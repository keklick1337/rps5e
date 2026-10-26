// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! SPIR-V emission of a structured SSA function: module setup, types and
//! constants, control flow, phis, ALU and subgroup operations. Memory,
//! images and the stage interface live in the sibling modules.
//!
//! Model: one host invocation per guest lane. Every guest branch is
//! wave-uniform (the IR's control flow is already structured), so the host
//! subgroup follows the IR's blocks in step and subgroup operations see
//! every lane of the guest wave the host subgroup holds. Two exceptions on
//! a 32-wide host subgroup: a wave64 program either runs two guest lanes
//! per invocation (`lanes`), or one per invocation over two subgroups
//! whose cross-lane operations go through workgroup memory
//! ([`super::wave_split`]).

use super::{EmitInput, Features};
use ps5shader::resources::{ResourceUse, ShaderStage};
use ps5shader::ssa::*;
use rspirv::dr::{Builder, InsertPoint, Instruction, Operand};
use rspirv::spirv::{self, Capability, Decoration, GlslStd450Op as GLOp, Op as S, StorageClass as SC, Word};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub type R<T> = Result<T, String>;

/// Subgroup scope.
pub const SCOPE_SUBGROUP: u32 = 3;
pub const SCOPE_WORKGROUP: u32 = 2;
pub const SCOPE_DEVICE: u32 = 1;

/// `SystemValue::Custom` payloads the compiler defines.
pub mod custom {
    /// Compute TG_SIZE word (wave index and wave count of the group).
    pub const THREAD_GROUP_SIZE: u32 = 0;
    /// Merged (ES+GS) wave info SGPR: vertex and primitive counts.
    pub const MERGED_WAVE_INFO: u32 = 1;
    /// Pixel POS_W: the clip-space w (1 / FragCoord.w).
    pub const POSITION_W: u32 = 2;
    /// Pixel POS_FIXED_PT: integer x in bits 0..15, y in 16..31.
    pub const POSITION_FIXED: u32 = 3;
    /// Compute: 1 for a lane past the group's thread count (paired
    /// wave64 lanes of a partial wave), else 0.
    pub const OUT_OF_RANGE: u32 = 4;
    /// Indirect draws on the host: the draw's base vertex / first
    /// instance (BaseVertex / BaseInstance) and the vertex / instance IDs
    /// without them.
    pub const BASE_VERTEX: u32 = 5;
    pub const BASE_INSTANCE: u32 = 6;
    pub const VERTEX_ID_REL: u32 = 7;
    pub const INSTANCE_ID_REL: u32 = 8;
    /// Merged (ES+GS) subgroup info SGPR (s2) of the vertex path: the
    /// subgroup's vertex and primitive counts.
    pub const SUBGROUP_INFO: u32 = 9;
    /// Mesh mode: hardware inputs computed by the entry point
    /// (`MESH_INPUT + k`, `k` from [`super::mesh`]'s `IN_*`).
    pub const MESH_INPUT: u32 = 16;
}

pub struct Emitter<'a, 'p> {
    pub b: Builder,
    pub f: &'a Function,
    pub input: &'a EmitInput<'a, 'p>,
    pub stage: ShaderStage,
    pub feat: Features,
    pub glsl: Word,
    types: HashMap<Type, Word>,
    consts: HashMap<(Type, u64), Word>,
    /// Ids of every scalar constant made by [`Self::konst`].
    const_ids: std::collections::HashSet<Word>,
    /// Type and bits of every scalar constant made by [`Self::konst`].
    const_vals: HashMap<Word, (Type, u64)>,
    undefs: HashMap<Type, Word>,
    ptrs: HashMap<(SC, Word), Word>,
    pub caps: BTreeMap<u32, Capability>,
    iconsts: HashMap<i32, Word>,
    i32_ty: Word,
    error: Option<String>,
    prologue_open: bool,
    pub exts: BTreeSet<&'static str>,
    /// Definitions per lane half.
    defs: Vec<[Word; 2]>,
    /// Values emitted in the branch of a lazy select rather than in their
    /// block ([`Self::plan_lazy_selects`]).
    deferred: Vec<bool>,
    /// The operations of each lazy select's true arm, in program order.
    lazy: HashMap<Value, Vec<Value>>,
    /// Guest lanes per host invocation (2: a wave64 program on a 32-wide
    /// host subgroup, guest lanes `lane` and `lane + 32`).
    pub lanes: usize,
    /// The lane half being emitted.
    pub half: usize,
    labels: Vec<Word>,
    exits: Vec<Word>,
    /// (phi id, rspirv block index, instruction index, IR value).
    phis: Vec<(usize, usize, Value, usize)>,
    pub cur_label: Word,
    prologue_block: usize,
    hoisted: HashMap<(u32, Word, Vec<u32>), Word>,
    /// Common subexpressions of pure integer / logical / composite /
    /// address instructions (and loads of invariant memory): (op, type,
    /// operands) -> (id, IR block it was made in). Block-local (for the
    /// block `cse_label`) without [`EmitOptions::global_cse`](super::EmitOptions);
    /// with it, an entry lives while the blocks emitted are dominated by
    /// its IR block, and entries made inside a branch the emitter adds die
    /// at its merge.
    cse: HashMap<CseKey, (Word, usize)>,
    cse_label: Word,
    /// Keys made inside each open emitter branch (global CSE).
    cse_scopes: Vec<Vec<CseKey>>,
    /// IR block being emitted ([`CSE_ROOT`] in the prologue).
    cse_block: usize,
    /// Function the CSE entries belong to.
    cse_function: Option<usize>,
    /// Pre / post order numbers of the IR dominator tree (global CSE).
    dom_order: Vec<(u32, u32)>,
    /// Pointers to memory no invocation writes while the module runs:
    /// loads through them are common subexpressions.
    pub invariant_ptrs: std::collections::HashSet<Word>,
    /// IR users of every value (built on first use).
    users: Option<HashMap<Value, Vec<Value>>>,
    /// Source-lane tables of lane-function values (see [`Self::lane_table`]).
    lane_tables: HashMap<Value, Option<Box<[u64; 64]>>>,
    /// Scalar loads of one instruction emitted together (see
    /// `memory::LoadGroups`), and the results of members emitted ahead.
    pub load_groups: Option<super::memory::LoadGroups>,
    pub group_results: HashMap<(Value, usize), Word>,
    /// A storage-buffer access whose index the module does not keep in
    /// range was emitted: the driver's robustness must stay on.
    pub unchecked_buffers: bool,
    /// The module stores to buffer or device memory (bound buffers, device
    /// addresses, atomics).
    pub writes_buffers: bool,
    /// The module stores to (or runs atomics on) images.
    pub writes_images: bool,
    /// The module invalidates the vector data caches (buffer_gl0_inv /
    /// buffer_gl1_inv): every buffer view is Coherent, so the reads after
    /// an invalidation see what other invocations wrote.
    pub coherent_buffers: bool,
    /// Nonzero while instructions go to a scratch block (`in_prologue`).
    cse_off: u32,
    /// Every global variable (SPIR-V 1.4+ entry point interface).
    pub globals: Vec<Word>,
    pub builtins: HashMap<(u32, u32), Word>,
    pub main: Word,
    pub res: super::memory::Resources,
    pub io: super::io::Interface,
    clock: Option<Word>,
    /// The memory access being emitted is globally coherent (volatile).
    pub volatile: bool,
    /// Geometry subgroup run as a mesh workgroup.
    pub mesh: Option<crate::pipeline::compiler::MeshConfig>,
    /// The module queries LOD in compute (quad derivatives).
    pub compute_derivatives: bool,
    /// The module uses compute derivatives over consecutive invocations.
    pub compute_derivatives_linear: bool,
    /// A native round-to-nearest-even f16 conversion was emitted (the
    /// module declares the 16-bit RoundingModeRTE mode).
    pub f16_rte_used: bool,
    /// (id, decoration) pairs applied without operands.
    decorated: std::collections::HashSet<(Word, u32)>,
    pub mesh_io: Option<super::mesh::MeshIo>,
    /// The entry point when it is not the guest function (mesh mode).
    entry: Word,
    /// A wave64 program run with one guest lane per invocation on a 32-wide
    /// host subgroup ([`super::wave_split`]).
    pub split: Option<super::wave_split::Split>,
}

/// A CSE key: (op, result type, operands).
type CseKey = (u32, Word, Vec<u64>);

/// [`Emitter::cse_block`] of values made in the prologue (they dominate
/// every block).
const CSE_ROOT: usize = usize::MAX;

/// Instructions without side effects or memory reads whose result depends
/// only on their operands (block-local CSE).
fn cse_pure(op: S) -> bool {
    matches!(
        op,
        S::IAdd
            | S::ISub
            | S::IMul
            | S::ShiftLeftLogical
            | S::ShiftRightLogical
            | S::ShiftRightArithmetic
            | S::BitwiseAnd
            | S::BitwiseOr
            | S::BitwiseXor
            | S::Not
            | S::BitFieldUExtract
            | S::BitFieldSExtract
            | S::BitFieldInsert
            | S::BitCount
            | S::CompositeExtract
            | S::CompositeConstruct
            | S::UConvert
            | S::SConvert
            | S::Bitcast
            | S::IEqual
            | S::INotEqual
            | S::ULessThan
            | S::ULessThanEqual
            | S::UGreaterThan
            | S::UGreaterThanEqual
            | S::SLessThan
            | S::SLessThanEqual
            | S::SGreaterThan
            | S::SGreaterThanEqual
            | S::LogicalAnd
            | S::LogicalOr
            | S::LogicalNot
            | S::LogicalEqual
            | S::LogicalNotEqual
            | S::Select
            | S::AccessChain
            | S::InBoundsAccessChain
            | S::ArrayLength
    )
}

/// Pre / post order numbers of every block in the function's dominator
/// tree (`(u32::MAX, 0)` for unreachable blocks): `a` dominates `b` when
/// `pre[a] <= pre[b] && post[b] <= post[a]`.
fn dominator_order(f: &Function) -> Vec<(u32, u32)> {
    let n = f.blocks.len();
    let mut order = vec![(u32::MAX, 0u32); n];
    if n == 0 || f.entry.index() >= n {
        return order;
    }
    let succs = ps5shader::ssa::build::block_successors(f);
    let idom = ps5shader::cfg::dom::dominators(&succs, f.entry.index());
    let mut children = vec![Vec::new(); n];
    for (b, &d) in idom.iter().enumerate() {
        if d != ps5shader::cfg::dom::NONE && d != b {
            children[d].push(b);
        }
    }
    let mut clock = 0u32;
    let mut stack = vec![(f.entry.index(), false)];
    while let Some((b, done)) = stack.pop() {
        if done {
            order[b].1 = clock;
            clock += 1;
            continue;
        }
        order[b].0 = clock;
        clock += 1;
        stack.push((b, true));
        for &c in children[b].iter().rev() {
            stack.push((c, false));
        }
    }
    order
}

/// The function stores to (or runs atomics on) buffer or device memory.
/// `RPS5E_DATA_CACHE_INVALIDATE=0`: buffer_gl0_inv / buffer_gl1_inv emit
/// no memory barrier and leave the buffer views as they are.
/// Part of the disk-cache key.
pub fn data_cache_invalidate_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_DATA_CACHE_INVALIDATE").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
}

fn invalidates_data_caches(f: &Function) -> bool {
    data_cache_invalidate_enabled() && f.order.iter().any(|&b| f.block(b).insts.iter().any(|&v| f.op(v) == Op::DataCacheInvalidate))
}

fn writes_buffers(f: &Function) -> bool {
    f.order.iter().any(|&b| {
        f.block(b).insts.iter().any(|&v| matches!(f.op(v), Op::BufferStore { .. } | Op::BufferAtomic { .. } | Op::GlobalStore { .. } | Op::GlobalAtomic { .. }))
    })
}

/// The function stores to (or runs atomics on) images.
fn writes_images(f: &Function) -> bool {
    f.order.iter().any(|&b| f.block(b).insts.iter().any(|&v| matches!(f.op(v), Op::Image(a) if matches!(a.op, ImageOp::Store | ImageOp::Atomic(_)))))
}

/// Memory semantics of an `s_barrier`: acquire-release of workgroup
/// memory (LDS), and of buffer and image memory when the program stores to
/// them: waves read what other waves of the group stored before the
/// barrier with plain loads after it.
pub(super) fn barrier_semantics(writes_buffers: bool, writes_images: bool) -> u32 {
    const ACQUIRE_RELEASE: u32 = 0x8;
    const UNIFORM_MEMORY: u32 = 0x40;
    const WORKGROUP_MEMORY: u32 = 0x100;
    const IMAGE_MEMORY: u32 = 0x800;
    let stored = if writes_buffers || writes_images { UNIFORM_MEMORY | IMAGE_MEMORY } else { 0 };
    ACQUIRE_RELEASE | WORKGROUP_MEMORY | stored
}

fn id_ops(ids: &[Word]) -> Vec<Operand> {
    ids.iter().map(|&i| Operand::IdRef(i)).collect()
}

impl<'a, 'p> Emitter<'a, 'p> {
    pub fn new(input: &'a EmitInput<'a, 'p>) -> Self {
        let mut b = Builder::new();
        b.set_version(1, 5);
        let glsl = b.ext_inst_import("GLSL.std.450");
        Emitter {
            b,
            f: input.func,
            input,
            stage: input.stage,
            feat: input.features,
            glsl,
            types: HashMap::new(),
            consts: HashMap::new(),
            const_ids: Default::default(),
            const_vals: HashMap::new(),
            undefs: HashMap::new(),
            ptrs: HashMap::new(),
            caps: BTreeMap::new(),
            iconsts: HashMap::new(),
            i32_ty: 0,
            error: None,
            prologue_open: false,
            exts: BTreeSet::new(),
            defs: vec![[0, 0]; input.func.values.len()],
            deferred: Vec::new(),
            lazy: HashMap::new(),
            lanes: input.lanes,
            half: 0,
            labels: vec![0; input.func.blocks.len()],
            exits: vec![0; input.func.blocks.len()],
            phis: Vec::new(),
            cur_label: 0,
            prologue_block: 0,
            hoisted: HashMap::new(),
            cse: HashMap::new(),
            cse_label: 0,
            cse_scopes: Vec::new(),
            cse_block: CSE_ROOT,
            cse_function: None,
            dom_order: dominator_order(input.func),
            invariant_ptrs: Default::default(),
            users: None,
            lane_tables: HashMap::new(),
            load_groups: None,
            group_results: HashMap::new(),
            unchecked_buffers: false,
            writes_buffers: writes_buffers(input.func),
            writes_images: writes_images(input.func),
            coherent_buffers: invalidates_data_caches(input.func),
            cse_off: 0,
            globals: Vec::new(),
            builtins: HashMap::new(),
            main: 0,
            res: Default::default(),
            io: Default::default(),
            clock: None,
            volatile: false,
            mesh: input.mesh,
            mesh_io: None,
            entry: 0,
            compute_derivatives: false,
            compute_derivatives_linear: false,
            f16_rte_used: false,
            decorated: Default::default(),
            split: input.split.clone().map(super::wave_split::Split::new),
        }
    }

    pub fn cap(&mut self, c: Capability) {
        self.caps.insert(c as u32, c);
    }

    /// [`Self::cse_scope_enter`] for branches built outside the helpers.
    pub fn open_scope(&mut self) {
        self.cse_scope_enter();
    }

    /// [`Self::cse_scope_exit`] for branches built outside the helpers.
    pub fn close_scope(&mut self) {
        self.cse_scope_exit();
    }

    /// Record an error raised inside a structured-helper closure.
    pub fn res_error(&mut self, m: String) {
        if self.error.is_none() {
            self.error = Some(m);
        }
    }

    pub fn take_error(&mut self) -> R<()> {
        match self.error.take() {
            Some(m) => Err(m),
            None => Ok(()),
        }
    }

    // ------------------------------------------------------------------
    // Types and constants
    // ------------------------------------------------------------------

    pub fn ty(&mut self, t: Type) -> Word {
        if let Some(&w) = self.types.get(&t) {
            return w;
        }
        let w = match t {
            Type::Void => self.b.type_void(),
            Type::Scalar(s) => match s {
                Scalar::Bool => self.b.type_bool(),
                Scalar::U16 => {
                    self.cap(Capability::Int16);
                    self.b.type_int(16, 0)
                }
                Scalar::U32 => self.b.type_int(32, 0),
                Scalar::U64 => {
                    self.cap(Capability::Int64);
                    self.b.type_int(64, 0)
                }
                Scalar::F16 => {
                    self.cap(Capability::Float16);
                    self.b.type_float(16, None)
                }
                Scalar::F32 => self.b.type_float(32, None),
                Scalar::F64 => {
                    self.cap(Capability::Float64);
                    self.b.type_float(64, None)
                }
            },
            Type::Vector(s, n) => {
                let e = self.ty(Type::Scalar(s));
                self.b.type_vector(e, n as u32)
            }
        };
        self.types.insert(t, w);
        w
    }

    pub fn t_u32(&mut self) -> Word {
        self.ty(Type::U32)
    }
    pub fn t_bool(&mut self) -> Word {
        self.ty(Type::BOOL)
    }
    pub fn t_f32(&mut self) -> Word {
        self.ty(Type::F32)
    }
    pub fn t_i32(&mut self) -> Word {
        if self.i32_ty == 0 {
            self.i32_ty = self.b.type_int(32, 1);
        }
        self.i32_ty
    }
    pub fn t_uvec(&mut self, n: u32) -> Word {
        self.ty(Type::vec(Scalar::U32, n))
    }
    pub fn t_fvec(&mut self, n: u32) -> Word {
        self.ty(Type::vec(Scalar::F32, n))
    }
    pub fn t_ivec(&mut self, n: u32) -> Word {
        let i = self.t_i32();
        if n == 1 {
            i
        } else {
            self.b.type_vector(i, n)
        }
    }

    pub fn ptr(&mut self, class: SC, pointee: Word) -> Word {
        if let Some(&p) = self.ptrs.get(&(class, pointee)) {
            return p;
        }
        let p = self.b.type_pointer(None, class, pointee);
        self.ptrs.insert((class, pointee), p);
        p
    }

    /// Scalar constant of `t` with the given bit pattern.
    pub fn konst(&mut self, t: Type, bits: u64) -> Word {
        let s = t.component();
        let bits = bits & s.mask();
        if let Some(&w) = self.consts.get(&(t, bits)) {
            return w;
        }
        let ty = self.ty(t);
        let w = match s {
            Scalar::Bool => {
                if bits != 0 {
                    self.b.constant_true(ty)
                } else {
                    self.b.constant_false(ty)
                }
            }
            Scalar::U64 | Scalar::F64 => self.b.constant_bit64(ty, bits),
            _ => self.b.constant_bit32(ty, bits as u32),
        };
        self.consts.insert((t, bits), w);
        self.const_ids.insert(w);
        self.const_vals.insert(w, (t, bits));
        w
    }

    /// `w` is a scalar constant.
    pub fn is_constant(&self, w: Word) -> bool {
        self.const_ids.contains(&w)
    }

    pub fn cu(&mut self, v: u32) -> Word {
        self.konst(Type::U32, v as u64)
    }
    pub fn cb(&mut self, v: bool) -> Word {
        self.konst(Type::BOOL, v as u64)
    }
    pub fn cf(&mut self, v: f32) -> Word {
        self.konst(Type::F32, v.to_bits() as u64)
    }
    /// The program's code address: from the shader data (the shader base)
    /// in a relocatable module, else the address it was translated at.
    fn code_address(&mut self) -> R<Word> {
        let layout = &self.input.request.bindings;
        if !self.input.request.relocatable_code || !layout.uses_shader_base() {
            return Ok(self.c64(self.f.code_address));
        }
        let at = layout.shader_base_dword;
        let lo = self.shader_data(at)?;
        let hi = self.shader_data(at + 1)?;
        let v2 = self.t_uvec(2);
        let pair = self.ins(S::CompositeConstruct, v2, &[lo, hi]);
        let t64 = self.ty(Type::U64);
        Ok(self.bitcast(t64, pair))
    }

    pub fn c64(&mut self, v: u64) -> Word {
        self.konst(Type::U64, v)
    }
    pub fn ci(&mut self, v: i32) -> Word {
        let t = self.t_i32();
        if let Some(&w) = self.iconsts.get(&v) {
            return w;
        }
        let w = self.b.constant_bit32(t, v as u32);
        self.iconsts.insert(v, w);
        w
    }

    pub fn undef(&mut self, t: Type) -> Word {
        if let Some(&w) = self.undefs.get(&t) {
            return w;
        }
        let ty = self.ty(t);
        let w = self.b.id();
        let inst = Instruction::new(S::Undef, Some(ty), Some(w), vec![]);
        self.b.module_mut().types_global_values.push(inst);
        self.undefs.insert(t, w);
        w
    }

    /// Composite constant made of scalar constants.
    pub fn const_vec(&mut self, t: Type, parts: &[Word]) -> Word {
        let ty = self.ty(t);
        self.b.constant_composite(ty, parts.iter().copied())
    }

    // ------------------------------------------------------------------
    // Instructions
    // ------------------------------------------------------------------

    /// Append `op` with id operands; returns the result id.
    pub fn ins(&mut self, op: S, ty: Word, args: &[Word]) -> Word {
        self.ins_ops(op, ty, id_ops(args))
    }

    pub fn ins_ops(&mut self, op: S, ty: Word, ops: Vec<Operand>) -> Word {
        if self.feat.emit.const_fold {
            if let Some(w) = self.fold_ins(op, ty, &ops) {
                return w;
            }
        }
        // An identical pure instruction in the same block (or, with global
        // CSE, in a dominating one) has the same value.
        let global = self.feat.emit.global_cse;
        let invariant_load = global && op == S::Load && matches!(ops.as_slice(), [Operand::IdRef(p)] if self.invariant_ptrs.contains(p));
        let key = if self.cse_off == 0 && (cse_pure(op) || invariant_load) {
            let mut k = Vec::with_capacity(ops.len());
            let mut ok = true;
            for o in &ops {
                match o {
                    Operand::IdRef(x) => k.push(*x as u64),
                    Operand::LiteralBit32(l) => k.push(*l as u64 | 1 << 40),
                    _ => ok = false,
                }
            }
            if !global && self.cse_label != self.cur_label {
                self.cse.clear();
                self.cse_label = self.cur_label;
            }
            ok.then_some((op as u32, ty, k))
        } else {
            None
        };
        if let Some(k) = &key {
            if let Some(&(w, _)) = self.cse.get(k) {
                return w;
            }
        }
        let id = self.ins_ops_new(op, ty, ops);
        if let Some(k) = key {
            if global {
                if let Some(scope) = self.cse_scopes.last_mut() {
                    scope.push(k.clone());
                }
            }
            self.cse.insert(k, (id, self.cse_block));
        }
        id
    }

    /// The value of a u32 or bool instruction on constants (or an identity
    /// of one constant operand: `x + 0`, `x & true`, a select on a constant),
    /// without emitting it.
    fn fold_ins(&mut self, op: S, ty: Word, ops: &[Operand]) -> Option<Word> {
        let tu = self.ty(Type::U32);
        let tb = self.ty(Type::BOOL);
        if ty != tu && ty != tb {
            return None;
        }
        let ids: Vec<Word> = ops.iter().map(|o| if let Operand::IdRef(x) = o { Some(*x) } else { None }).collect::<Option<_>>()?;
        let val = |e: &Self, w: Word| -> Option<u64> {
            match e.const_vals.get(&w) {
                Some(&(t, b)) if t == Type::U32 || t == Type::BOOL => Some(b),
                _ => None,
            }
        };
        let k: Vec<Option<u64>> = ids.iter().map(|&w| val(self, w)).collect();
        if op == S::Select && ids.len() == 3 {
            // Arms of u32/bool type only (`ty` checked above).
            if let Some(c) = k[0] {
                return Some(if c != 0 { ids[1] } else { ids[2] });
            }
            if ids[1] == ids[2] {
                return Some(ids[1]);
            }
            return None;
        }
        let u = |b: u64| b as u32;
        let r: Option<u64> = match (op, k.as_slice()) {
            (S::IAdd, [Some(a), Some(b)]) => Some(u(*a).wrapping_add(u(*b)) as u64),
            (S::ISub, [Some(a), Some(b)]) => Some(u(*a).wrapping_sub(u(*b)) as u64),
            (S::IMul, [Some(a), Some(b)]) => Some(u(*a).wrapping_mul(u(*b)) as u64),
            (S::BitwiseAnd, [Some(a), Some(b)]) => Some(a & b),
            (S::BitwiseOr, [Some(a), Some(b)]) => Some(a | b),
            (S::BitwiseXor, [Some(a), Some(b)]) => Some(a ^ b),
            (S::ShiftRightLogical, [Some(a), Some(b)]) if *b < 32 => Some((u(*a) >> b) as u64),
            (S::ShiftLeftLogical, [Some(a), Some(b)]) if *b < 32 => Some((u(*a) << b) as u64),
            (S::ULessThan, [Some(a), Some(b)]) => Some((a < b) as u64),
            (S::ULessThanEqual, [Some(a), Some(b)]) => Some((a <= b) as u64),
            (S::UGreaterThan, [Some(a), Some(b)]) => Some((a > b) as u64),
            (S::UGreaterThanEqual, [Some(a), Some(b)]) => Some((a >= b) as u64),
            (S::IEqual, [Some(a), Some(b)]) => Some((a == b) as u64),
            (S::INotEqual, [Some(a), Some(b)]) => Some((a != b) as u64),
            (S::LogicalAnd, [Some(a), Some(b)]) => Some((*a != 0 && *b != 0) as u64),
            (S::LogicalOr, [Some(a), Some(b)]) => Some((*a != 0 || *b != 0) as u64),
            (S::LogicalNot, [Some(a)]) => Some((*a == 0) as u64),
            _ => None,
        };
        if let Some(r) = r {
            return Some(self.konst(if ty == tb { Type::BOOL } else { Type::U32 }, r));
        }
        // One constant operand.
        let other = |i: usize| ids[1 - i];
        for i in 0..ids.len().min(2) {
            let Some(c) = k[i] else { continue };
            if ids.len() != 2 {
                break;
            }
            let x = other(i);
            match op {
                S::IAdd | S::BitwiseOr | S::BitwiseXor if c == 0 => return Some(x),
                S::ISub | S::ShiftRightLogical | S::ShiftLeftLogical if i == 1 && c == 0 => return Some(x),
                S::BitwiseAnd if u(c) == u32::MAX => return Some(x),
                S::IMul if c == 1 => return Some(x),
                S::LogicalAnd if c != 0 => return Some(x),
                S::LogicalAnd => return Some(ids[i]),
                S::LogicalOr if c == 0 => return Some(x),
                S::LogicalOr => return Some(ids[i]),
                _ => {}
            }
        }
        None
    }

    /// Open a branch the emitter adds: CSE entries made in it are dropped
    /// at [`Self::cse_scope_exit`] (its merge does not see them).
    fn cse_scope_enter(&mut self) {
        if self.feat.emit.global_cse {
            self.cse_scopes.push(Vec::new());
        }
    }

    fn cse_scope_exit(&mut self) {
        if self.feat.emit.global_cse {
            if let Some(keys) = self.cse_scopes.pop() {
                for k in keys {
                    self.cse.remove(&k);
                }
            }
        }
    }

    /// Global CSE at the start of IR block `blk`: keep the entries of the
    /// blocks that dominate it.
    fn cse_enter_block(&mut self, blk: usize) {
        if !self.feat.emit.global_cse {
            return;
        }
        self.cse_scopes.clear();
        self.cse_block = blk;
        let order = &self.dom_order;
        let (pre, post) = order.get(blk).copied().unwrap_or((u32::MAX, 0));
        self.cse.retain(|_, &mut (_, b)| {
            if b == CSE_ROOT {
                return true;
            }
            let Some(&(bp, bq)) = order.get(b) else { return false };
            pre != u32::MAX && bp != u32::MAX && bp <= pre && post <= bq
        });
    }

    fn ins_ops_new(&mut self, op: S, ty: Word, ops: Vec<Operand>) -> Word {
        let id = self.b.id();
        let inst = Instruction::new(op, Some(ty), Some(id), ops);
        self.b.insert_into_block(InsertPoint::End, inst).expect("instruction outside a block");
        id
    }

    /// Append an instruction without a result.
    pub fn void(&mut self, op: S, ops: Vec<Operand>) {
        let inst = Instruction::new(op, None, None, ops);
        self.b.insert_into_block(InsertPoint::End, inst).expect("instruction outside a block");
    }

    pub fn gl(&mut self, ty: Word, op: GLOp, args: &[Word]) -> Word {
        let set = self.glsl;
        self.b.ext_inst(ty, None, set, op as u32, id_ops(args)).expect("ext inst outside a block")
    }

    pub fn decorate(&mut self, id: Word, d: Decoration, lits: &[u32]) {
        // A value shared by block-local CSE may be decorated again: a
        // decoration without operands is applied once.
        if lits.is_empty() && !self.decorated.insert((id, d as u32)) {
            return;
        }
        self.b.decorate(id, d, lits.iter().map(|&l| Operand::LiteralBit32(l)));
    }

    pub fn bitcast(&mut self, ty: Word, x: Word) -> Word {
        self.ins(S::Bitcast, ty, &[x])
    }

    pub fn extract(&mut self, ty: Word, v: Word, i: u32) -> Word {
        self.ins_ops(S::CompositeExtract, ty, vec![Operand::IdRef(v), Operand::LiteralBit32(i)])
    }

    pub fn construct(&mut self, ty: Word, parts: &[Word]) -> Word {
        self.ins(S::CompositeConstruct, ty, parts)
    }

    pub fn select(&mut self, ty: Word, c: Word, a: Word, b: Word) -> Word {
        self.ins(S::Select, ty, &[c, a, b])
    }

    pub fn iadd(&mut self, a: Word, b: Word) -> Word {
        let t = self.t_u32();
        self.ins(S::IAdd, t, &[a, b])
    }
    pub fn isub(&mut self, a: Word, b: Word) -> Word {
        let t = self.t_u32();
        self.ins(S::ISub, t, &[a, b])
    }
    pub fn imul(&mut self, a: Word, b: Word) -> Word {
        let t = self.t_u32();
        self.ins(S::IMul, t, &[a, b])
    }
    pub fn band(&mut self, a: Word, b: Word) -> Word {
        let t = self.t_u32();
        self.ins(S::BitwiseAnd, t, &[a, b])
    }
    pub fn bor(&mut self, a: Word, b: Word) -> Word {
        let t = self.t_u32();
        self.ins(S::BitwiseOr, t, &[a, b])
    }
    pub fn shr(&mut self, a: Word, b: Word) -> Word {
        let t = self.t_u32();
        self.ins(S::ShiftRightLogical, t, &[a, b])
    }
    pub fn shl(&mut self, a: Word, b: Word) -> Word {
        let t = self.t_u32();
        self.ins(S::ShiftLeftLogical, t, &[a, b])
    }
    pub fn ult(&mut self, a: Word, b: Word) -> Word {
        let t = self.t_bool();
        self.ins(S::ULessThan, t, &[a, b])
    }
    pub fn ule(&mut self, a: Word, b: Word) -> Word {
        let t = self.t_bool();
        self.ins(S::ULessThanEqual, t, &[a, b])
    }
    pub fn ieq(&mut self, a: Word, b: Word) -> Word {
        let t = self.t_bool();
        self.ins(S::IEqual, t, &[a, b])
    }
    pub fn ine(&mut self, a: Word, b: Word) -> Word {
        let t = self.t_bool();
        self.ins(S::INotEqual, t, &[a, b])
    }
    pub fn land(&mut self, a: Word, b: Word) -> Word {
        let t = self.t_bool();
        self.ins(S::LogicalAnd, t, &[a, b])
    }
    pub fn lor(&mut self, a: Word, b: Word) -> Word {
        let t = self.t_bool();
        self.ins(S::LogicalOr, t, &[a, b])
    }
    pub fn lnot(&mut self, a: Word) -> Word {
        let t = self.t_bool();
        self.ins(S::LogicalNot, t, &[a])
    }
    pub fn umin(&mut self, a: Word, b: Word) -> Word {
        let t = self.t_u32();
        self.gl(t, GLOp::UMin, &[a, b])
    }
    /// Bits `offset .. offset + count` of a u32 (constant offsets).
    pub fn bits(&mut self, v: Word, offset: u32, count: u32) -> Word {
        let t = self.t_u32();
        let (o, c) = (self.cu(offset), self.cu(count));
        self.ins(S::BitFieldUExtract, t, &[v, o, c])
    }

    pub fn load(&mut self, ty: Word, ptr: Word) -> Word {
        self.ins(S::Load, ty, &[ptr])
    }

    pub fn store(&mut self, ptr: Word, v: Word) {
        self.void(S::Store, vec![Operand::IdRef(ptr), Operand::IdRef(v)]);
    }

    pub fn access(&mut self, ptr_ty: Word, base: Word, idx: &[Word]) -> Word {
        let mut ops = vec![base];
        ops.extend_from_slice(idx);
        self.ins(S::AccessChain, ptr_ty, &ops)
    }

    // ------------------------------------------------------------------
    // Globals and the prologue
    // ------------------------------------------------------------------

    pub fn global(&mut self, ty: Word, class: SC) -> Word {
        let p = self.ptr(class, ty);
        let v = self.b.id();
        let inst = Instruction::new(S::Variable, Some(p), Some(v), vec![Operand::StorageClass(class)]);
        self.b.module_mut().types_global_values.push(inst);
        self.globals.push(v);
        v
    }

    /// A built-in input variable of `ty` (created once).
    pub fn builtin_input(&mut self, builtin: spirv::BuiltIn, ty: Word) -> Word {
        if let Some(&v) = self.builtins.get(&(builtin as u32, 0)) {
            return v;
        }
        let v = self.global(ty, SC::Input);
        self.decorate(v, Decoration::BuiltIn, &[builtin as u32]);
        let integer = matches!(builtin, spirv::BuiltIn::SubgroupLocalInvocationId | spirv::BuiltIn::PrimitiveId | spirv::BuiltIn::SampleId | spirv::BuiltIn::SampleMask | spirv::BuiltIn::Layer | spirv::BuiltIn::ViewportIndex);
        if self.stage == ShaderStage::Pixel && integer {
            self.decorate(v, Decoration::Flat, &[]);
        }
        self.builtins.insert((builtin as u32, 0), v);
        v
    }

    /// Run `f` with the insertion point in the prologue block (before its
    /// terminator); values made there dominate the whole function.
    pub fn in_prologue<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        if self.prologue_open {
            return f(self);
        }
        let saved = self.b.selected_block();
        // Emit into a scratch block, then move its instructions.
        let func = self.b.selected_function().expect("no function");
        let blocks_before = self.b.module_ref().functions[func].blocks.len();
        self.b.select_block(None).ok();
        let scratch = self.b.id();
        self.b.begin_block(Some(scratch)).expect("scratch block");
        let saved_label = self.cur_label;
        self.cse_off += 1;
        let r = f(self);
        self.cse_off -= 1;
        self.cur_label = saved_label;
        let m = self.b.module_mut();
        let blk = m.functions[func].blocks.remove(blocks_before);
        let pro = &mut m.functions[func].blocks[self.prologue_block];
        let at = pro.instructions.len() - 1;
        for (k, inst) in blk.instructions.into_iter().enumerate() {
            pro.instructions.insert(at + k, inst);
        }
        self.b.select_block(saved).ok();
        r
    }

    /// A hoisted instruction (deduplicated).
    pub fn hoist(&mut self, op: S, ty: Word, args: &[Word]) -> Word {
        let key = (op as u32, ty, args.to_vec());
        if let Some(&w) = self.hoisted.get(&key) {
            return w;
        }
        let w = self.in_prologue(|e| e.ins(op, ty, args));
        self.hoisted.insert(key, w);
        w
    }

    pub fn hoist_extract(&mut self, ty: Word, v: Word, i: u32) -> Word {
        let key = (S::CompositeExtract as u32, ty, vec![v, i | 0x8000_0000]);
        if let Some(&w) = self.hoisted.get(&key) {
            return w;
        }
        let w = self.in_prologue(|e| e.extract(ty, v, i));
        self.hoisted.insert(key, w);
        w
    }

    /// Load of a built-in input (hoisted).
    pub fn builtin_value(&mut self, builtin: spirv::BuiltIn, ty: Word) -> Word {
        let var = self.builtin_input(builtin, ty);
        self.hoist(S::Load, ty, &[var])
    }

    // ------------------------------------------------------------------
    // Structured helpers
    // ------------------------------------------------------------------

    pub fn begin(&mut self, label: Word) {
        self.b.begin_block(Some(label)).expect("nested block");
        self.cur_label = label;
        let function = self.b.selected_function();
        if function != self.cse_function {
            // Ids of another function are not visible here.
            self.cse.clear();
            self.cse_scopes.clear();
            self.cse_function = function;
            self.cse_block = CSE_ROOT;
        }
    }

    /// `if (cond) { f }`.
    pub fn if_then(&mut self, cond: Word, f: impl FnOnce(&mut Self)) {
        let then_l = self.b.id();
        let merge = self.b.id();
        self.b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
        self.b.branch_conditional(cond, then_l, merge, []).unwrap();
        self.begin(then_l);
        self.cse_scope_enter();
        f(self);
        self.cse_scope_exit();
        self.b.branch(merge).unwrap();
        self.begin(merge);
    }

    /// `cond ? f() : otherwise` where `f` has side effects.
    pub fn if_value(&mut self, cond: Word, ty: Word, otherwise: Word, f: impl FnOnce(&mut Self) -> Word) -> Word {
        let head = self.cur_label;
        let then_l = self.b.id();
        let merge = self.b.id();
        self.b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
        self.b.branch_conditional(cond, then_l, merge, []).unwrap();
        self.begin(then_l);
        self.cse_scope_enter();
        let v = f(self);
        self.cse_scope_exit();
        let exit = self.cur_label;
        self.b.branch(merge).unwrap();
        self.begin(merge);
        self.b.phi(ty, None, [(v, exit), (otherwise, head)]).unwrap()
    }

    /// `cond ? f() : g()` with `n` results of type `ty`, where both arms
    /// have side effects.
    pub fn if_else_values(&mut self, cond: Word, ty: Word, n: usize, f: impl FnOnce(&mut Self) -> Vec<Word>, g: impl FnOnce(&mut Self) -> Vec<Word>) -> Vec<Word> {
        let (then_l, else_l, merge) = (self.b.id(), self.b.id(), self.b.id());
        self.b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
        self.b.branch_conditional(cond, then_l, else_l, []).unwrap();
        self.begin(then_l);
        self.cse_scope_enter();
        let a = f(self);
        self.cse_scope_exit();
        let a_exit = self.cur_label;
        self.b.branch(merge).unwrap();
        self.begin(else_l);
        self.cse_scope_enter();
        let b = g(self);
        self.cse_scope_exit();
        let b_exit = self.cur_label;
        self.b.branch(merge).unwrap();
        self.begin(merge);
        (0..n).map(|i| self.b.phi(ty, None, [(a[i], a_exit), (b[i], b_exit)]).unwrap()).collect()
    }

    /// `cond ? f() : otherwise` with several results of type `ty`, where
    /// `f` has side effects.
    pub fn if_values(&mut self, cond: Word, ty: Word, otherwise: &[Word], f: impl FnOnce(&mut Self) -> Vec<Word>) -> Vec<Word> {
        let head = self.cur_label;
        let then_l = self.b.id();
        let merge = self.b.id();
        self.b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
        self.b.branch_conditional(cond, then_l, merge, []).unwrap();
        self.begin(then_l);
        self.cse_scope_enter();
        let v = f(self);
        self.cse_scope_exit();
        let exit = self.cur_label;
        self.b.branch(merge).unwrap();
        self.begin(merge);
        v.iter().zip(otherwise).map(|(&a, &o)| self.b.phi(ty, None, [(a, exit), (o, head)]).unwrap()).collect()
    }

    /// `cond ? f() : g()` where both arms have side effects.
    pub fn if_else_value(&mut self, cond: Word, ty: Word, f: impl FnOnce(&mut Self) -> Word, g: impl FnOnce(&mut Self) -> Word) -> Word {
        let (then_l, else_l, merge) = (self.b.id(), self.b.id(), self.b.id());
        self.b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
        self.b.branch_conditional(cond, then_l, else_l, []).unwrap();
        self.begin(then_l);
        self.cse_scope_enter();
        let a = f(self);
        self.cse_scope_exit();
        let a_exit = self.cur_label;
        self.b.branch(merge).unwrap();
        self.begin(else_l);
        self.cse_scope_enter();
        let b = g(self);
        self.cse_scope_exit();
        let b_exit = self.cur_label;
        self.b.branch(merge).unwrap();
        self.begin(merge);
        self.b.phi(ty, None, [(a, a_exit), (b, b_exit)]).unwrap()
    }

    /// `switch (sel) { case k: f(k) }` for `k < count`, as a value: any
    /// other selector takes case 0.
    pub fn switch_value(&mut self, sel: Word, count: u32, ty: Word, mut f: impl FnMut(&mut Self, u32) -> Word) -> Word {
        let merge = self.b.id();
        let labels: Vec<Word> = (0..count).map(|_| self.b.id()).collect();
        self.b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
        let targets: Vec<(Operand, Word)> = (1..count).map(|k| (Operand::LiteralBit32(k), labels[k as usize])).collect();
        self.b.switch(sel, labels[0], targets).unwrap();
        let mut incoming = Vec::new();
        for k in 0..count {
            self.begin(labels[k as usize]);
            self.cse_scope_enter();
            let v = f(self, k);
            self.cse_scope_exit();
            incoming.push((v, self.cur_label));
            self.b.branch(merge).unwrap();
        }
        self.begin(merge);
        self.b.phi(ty, None, incoming).unwrap()
    }

    /// `switch (sel) { case k: f(k) }` for `k < count`; any other selector
    /// does nothing.
    pub fn switch_then(&mut self, sel: Word, count: u32, mut f: impl FnMut(&mut Self, u32)) {
        let merge = self.b.id();
        let labels: Vec<Word> = (0..count).map(|_| self.b.id()).collect();
        self.b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
        let targets: Vec<(Operand, Word)> = (0..count).map(|k| (Operand::LiteralBit32(k), labels[k as usize])).collect();
        self.b.switch(sel, merge, targets).unwrap();
        for k in 0..count {
            self.begin(labels[k as usize]);
            self.cse_scope_enter();
            f(self, k);
            self.cse_scope_exit();
            self.b.branch(merge).unwrap();
        }
        self.begin(merge);
    }

    // ------------------------------------------------------------------
    // Values
    // ------------------------------------------------------------------

    /// The SPIR-V id of an IR value (constants and undefs on demand).
    pub fn val(&mut self, v: Value) -> R<Word> {
        let d = self.defs[v.index()][self.half];
        if d != 0 {
            return Ok(d);
        }
        let data = self.f.value(v);
        let w = match data.op {
            Op::Const(bits) => self.konst(data.ty, bits),
            // Both halves of a split wave must see the same value.
            Op::Undef if self.split.is_some() => super::wave_split::zero(self, data.ty),
            Op::Undef => self.undef(data.ty),
            other => return Err(format!("value %{} ({}) used before its definition", v.0, other.name())),
        };
        self.defs[v.index()] = [w, w];
        Ok(w)
    }

    /// Argument `i` of `v` in lane half `half`.
    pub fn arg_half(&mut self, v: Value, i: usize, half: usize) -> R<Word> {
        let saved = self.half;
        self.half = half;
        let r = self.arg(v, i);
        self.half = saved;
        r
    }

    pub fn arg(&mut self, v: Value, i: usize) -> R<Word> {
        let a = self.f.args(v)[i];
        self.val(a)
    }

    pub fn arg_ty(&self, v: Value, i: usize) -> Type {
        self.f.ty(self.f.args(v)[i])
    }

    pub fn uses(&self, v: Value) -> Option<&Vec<ResourceUse>> {
        self.input.uses.get(&v)
    }

    // ------------------------------------------------------------------
    // Module
    // ------------------------------------------------------------------

    pub fn emit_function(&mut self) -> R<()> {
        let void = self.ty(Type::Void);
        let fnty = self.b.type_function(void, []);
        self.main = self.b.begin_function(void, None, spirv::FunctionControl::NONE, fnty).map_err(|e| format!("{e:?}"))?;
        for &blk in &self.f.order {
            self.labels[blk.index()] = self.b.id();
        }
        // Prologue block: hoisted values are inserted before its branch.
        let pro = self.b.id();
        self.begin(pro);
        self.prologue_block = self.b.selected_block().unwrap();
        self.prologue_open = true;
        self.prologue_init()?;
        self.prologue_open = false;
        let entry = self.labels[self.f.entry.index()];
        self.b.branch(entry).unwrap();
        if self.feat.emit.lazy_select {
            self.plan_lazy_selects();
        }
        let order = self.f.order.clone();
        for blk in order {
            self.emit_block(blk)?;
        }
        self.patch_phis()?;
        self.b.end_function().map_err(|e| format!("{e:?}"))?;
        if self.mesh.is_some() {
            self.entry = super::mesh::emit_entry(self, self.main)?;
        }
        Ok(())
    }

    fn prologue_init(&mut self) -> R<()> {
        super::io::prologue(self)
    }

    fn emit_block(&mut self, blk: Block) -> R<()> {
        let label = self.labels[blk.index()];
        if label == 0 {
            return Err(format!("block {} is not in the layout order", blk.0));
        }
        self.begin(label);
        self.cse_enter_block(blk.index());
        let bd = self.f.block(blk).clone();
        for &v in &bd.insts {
            if self.deferred.get(v.index()).copied().unwrap_or(false) {
                continue;
            }
            self.emit_value(v).map_err(|e| format!("%{} = {}: {e}", v.0, self.f.op(v).name()))?;
        }
        self.exits[blk.index()] = self.cur_label;
        let lab = |e: &Self, b: Block| -> R<Word> {
            let l = e.labels[b.index()];
            if l == 0 {
                Err(format!("branch to block {} outside the layout order", b.0))
            } else {
                Ok(l)
            }
        };
        match bd.term {
            Terminator::Branch(t) => {
                if let Merge::Loop { merge, continue_block } = bd.merge {
                    let (m, c) = (lab(self, merge)?, lab(self, continue_block)?);
                    self.b.loop_merge(m, c, spirv::LoopControl::NONE, []).unwrap();
                }
                let t = lab(self, t)?;
                self.b.branch(t).unwrap();
            }
            Terminator::CondBranch { cond, then_block, else_block } => {
                self.half = 0;
                let c = self.val(cond)?;
                let (t, e) = (lab(self, then_block)?, lab(self, else_block)?);
                match bd.merge {
                    Merge::Loop { merge, continue_block } => {
                        let (m, cb) = (lab(self, merge)?, lab(self, continue_block)?);
                        self.b.loop_merge(m, cb, spirv::LoopControl::NONE, []).unwrap();
                        self.b.branch_conditional(c, t, e, []).unwrap();
                    }
                    Merge::Selection { merge } if t != e => {
                        let m = lab(self, merge)?;
                        let control = super::wave_split::selection_control(self, cond);
                        self.b.selection_merge(m, control).unwrap();
                        self.b.branch_conditional(c, t, e, []).unwrap();
                    }
                    _ if t == e => {
                        self.b.branch(t).unwrap();
                    }
                    _ => {
                        self.b.branch_conditional(c, t, e, []).unwrap();
                    }
                }
            }
            Terminator::Return => {
                super::io::before_return(self)?;
                self.b.ret().unwrap();
            }
            Terminator::Unreachable => {
                self.b.unreachable().unwrap();
            }
            Terminator::None => return Err(format!("block {} has no terminator", blk.0)),
        }
        Ok(())
    }

    fn patch_phis(&mut self) -> R<()> {
        let func = self.b.selected_function().unwrap();
        let phis = std::mem::take(&mut self.phis);
        for (bi, ii, v, half) in phis {
            self.half = half;
            let blk = self.f.value(v).block.unwrap();
            let preds = self.f.block(blk).preds.clone();
            let args = self.f.args(v).to_vec();
            if preds.len() != args.len() {
                return Err(format!("phi %{} has {} arguments for {} predecessors", v.0, args.len(), preds.len()));
            }
            let mut ops = Vec::new();
            for (k, &a) in args.iter().enumerate() {
                let w = self.val(a)?;
                let exit = self.exits[preds[k].index()];
                if exit == 0 {
                    return Err(format!("phi %{} predecessor block {} was not emitted", v.0, preds[k].0));
                }
                ops.push(Operand::IdRef(w));
                ops.push(Operand::IdRef(exit));
            }
            // Constants created while patching went to the global section;
            // the block indices are unchanged.
            self.b.module_mut().functions[func].blocks[bi].instructions[ii].operands = ops;
        }
        self.half = 0;
        Ok(())
    }

    /// The operations each lazy select (`Function::lazy_selects`) needs
    /// only for its true arm: pure arithmetic in the select's block whose
    /// every use is another such operation or the select's true arm. They
    /// are emitted under a branch on the select's condition instead of in
    /// their block (the same values where the select takes them).
    fn plan_lazy_selects(&mut self) {
        let f = self.f;
        if f.lazy_selects.is_empty() {
            return;
        }
        // Uses of every value: (user, argument index); a terminator's use
        // has no user.
        let n = f.values.len();
        let mut users: Vec<Vec<(Option<Value>, usize)>> = vec![Vec::new(); n];
        for &b in &f.order {
            let bd = f.block(b);
            for &v in &bd.insts {
                for (k, &a) in f.args(v).iter().enumerate() {
                    users[a.index()].push((Some(v), k));
                }
            }
            if let Terminator::CondBranch { cond, .. } = bd.term {
                users[cond.index()].push((None, 0));
            }
        }
        let pure = |op: Op| {
            matches!(
                op,
                Op::Select
                    | Op::Bitcast
                    | Op::Construct
                    | Op::Extract(_)
                    | Op::IAdd
                    | Op::ISub
                    | Op::Shl
                    | Op::LShr
                    | Op::And
                    | Op::Or
                    | Op::Xor
                    | Op::Not
                    | Op::UMin
                    | Op::UMax
                    | Op::UConvert
                    | Op::ICmp(_)
                    | Op::FAdd
                    | Op::FSub
                    | Op::FMul
                    | Op::FFma
                    | Op::FNeg
                    | Op::FAbs
                    | Op::FCmp(_)
                    | Op::FConvert(_)
            )
        };
        let mut deferred = vec![false; n];
        let mut selects: Vec<Value> = f.lazy_selects.iter().copied().collect();
        selects.sort();
        for s in selects {
            let d = f.value(s);
            let Some(blk) = d.block else { continue };
            if d.op != Op::Select || !f.order.contains(&blk) {
                continue;
            }
            let insts = &f.block(blk).insts;
            let Some(pos) = insts.iter().position(|&v| v == s) else { continue };
            let mut cone: Vec<Value> = Vec::new();
            let mut in_cone: std::collections::HashSet<Value> = std::collections::HashSet::new();
            for &v in insts[..pos].iter().rev() {
                if deferred[v.index()] || !pure(f.op(v)) || users[v.index()].is_empty() {
                    continue;
                }
                let only_arm = users[v.index()].iter().all(|&(u, k)| match u {
                    Some(u) if u == s => k == 1,
                    Some(u) => in_cone.contains(&u),
                    None => false,
                });
                if only_arm {
                    in_cone.insert(v);
                    cone.push(v);
                }
            }
            if !in_cone.contains(&d.args[1]) {
                continue;
            }
            cone.reverse();
            for &v in &cone {
                deferred[v.index()] = true;
            }
            self.lazy.insert(s, cone);
        }
        self.deferred = deferred;
    }

    fn define(&mut self, v: Value, w: Word) {
        self.defs[v.index()][self.half] = w;
    }

    /// Operations done once per guest wave (their result, if any, is
    /// shared by the lane halves).
    fn wave_once(op: Op) -> bool {
        matches!(op, Op::Ballot | Op::Barrier | Op::SubgroupLdsBarrier | Op::DataCacheInvalidate | Op::LdsAppend { .. } | Op::GdsAppend { .. } | Op::Clock { .. } | Op::SendMessage { .. })
    }

    fn emit_value(&mut self, v: Value) -> R<()> {
        let op = self.f.op(v);
        // Scalar memory reads take SGPR operands and no EXEC: both lane
        // halves would load the same words.
        let smem = self.feat.emit.smem_once && matches!(op, Op::ScalarLoad { .. } | Op::ScalarBufferLoad { .. });
        if self.lanes == 2 && (Self::wave_once(op) || smem) {
            self.half = 0;
            self.emit_value_half(v)?;
            self.defs[v.index()][1] = self.defs[v.index()][0];
            return Ok(());
        }
        for h in 0..self.lanes {
            self.half = h;
            self.emit_value_half(v)?;
        }
        self.half = 0;
        Ok(())
    }

    fn emit_value_half(&mut self, v: Value) -> R<()> {
        let data = self.f.value(v).clone();
        if data.op == Op::Phi {
            let ty = self.ty(data.ty);
            let id = self.b.id();
            let inst = Instruction::new(S::Phi, Some(ty), Some(id), vec![]);
            self.b.insert_into_block(InsertPoint::End, inst).unwrap();
            let bi = self.b.selected_block().unwrap();
            let func = self.b.selected_function().unwrap();
            let ii = self.b.module_ref().functions[func].blocks[bi].instructions.len() - 1;
            self.phis.push((bi, ii, v, self.half));
            self.define(v, id);
            return Ok(());
        }
        let w = self.emit_op(v, &data)?;
        if let Some(w) = w {
            self.define(v, w);
        }
        Ok(())
    }

    /// Emit one operation; `None` for operations without a value.
    fn emit_op(&mut self, v: Value, d: &ValueData) -> R<Option<Word>> {
        if self.split.is_some() {
            if let Some(r) = super::wave_split::emit_op(self, v, d)? {
                return Ok(r);
            }
        }
        let ty = d.ty;
        let r = match d.op {
            Op::Const(bits) => self.konst(ty, bits),
            Op::Undef => self.undef(ty),
            Op::Phi | Op::GetReg(_) | Op::SetReg(_) => return Err("register-level operation after SSA construction".into()),
            Op::Select if self.lazy.contains_key(&v) => {
                // The true arm's operations run only where it is taken.
                let t = self.ty(ty);
                let (c, b) = (self.arg(v, 0)?, self.arg(v, 2)?);
                let cone = self.lazy[&v].clone();
                let err = std::cell::RefCell::new(None::<String>);
                let r = self.if_value(c, t, b, |e| {
                    for &x in &cone {
                        if let Err(m) = e.emit_value_half(x) {
                            err.borrow_mut().get_or_insert(m);
                        }
                    }
                    e.arg(v, 1).unwrap_or_else(|m| {
                        err.borrow_mut().get_or_insert(m);
                        b
                    })
                });
                if let Some(m) = err.into_inner() {
                    return Err(m);
                }
                r
            }
            Op::Select => {
                let t = self.ty(ty);
                let (c, a, b) = (self.arg(v, 0)?, self.arg(v, 1)?, self.arg(v, 2)?);
                self.select(t, c, a, b)
            }
            Op::Bitcast => {
                let t = self.ty(ty);
                let x = self.arg(v, 0)?;
                if self.arg_ty(v, 0) == ty {
                    x
                } else {
                    self.bitcast(t, x)
                }
            }
            Op::Construct => {
                let t = self.ty(ty);
                let parts = (0..d.args.len()).map(|i| self.arg(v, i)).collect::<R<Vec<_>>>()?;
                self.construct(t, &parts)
            }
            Op::Extract(i) => {
                let t = self.ty(ty);
                let x = self.arg(v, 0)?;
                if self.arg_ty(v, 0).is_vector() {
                    self.extract(t, x, i as u32)
                } else {
                    x
                }
            }
            Op::Insert(i) => {
                let t = self.ty(ty);
                let (vec, c) = (self.arg(v, 0)?, self.arg(v, 1)?);
                self.ins_ops(S::CompositeInsert, t, vec![Operand::IdRef(c), Operand::IdRef(vec), Operand::LiteralBit32(i as u32)])
            }
            Op::LaneId => self.lane_id(),
            Op::Ballot => {
                let p = self.arg(v, 0)?;
                let p1 = if self.lanes == 2 { Some(self.arg_half(v, 0, 1)?) } else { None };
                self.ballot2(p, p1)
            }
            Op::InverseBallot => {
                let (lo, hi) = (self.arg(v, 0)?, self.arg(v, 1)?);
                self.inverse_ballot(lo, hi)
            }
            Op::ReadLane | Op::Shuffle => {
                let (x, lane) = (self.arg(v, 0)?, self.arg(v, 1)?);
                if let Some(w) = self.lane_read(v, d.op, ty)? {
                    w
                } else if self.lanes == 2 {
                    // Lanes 0..31 live in half 0, 32..63 in half 1.
                    let x0 = self.arg_half(v, 0, 0)?;
                    let x1 = self.arg_half(v, 0, 1)?;
                    let _ = x;
                    let c31 = self.cu(31);
                    let l = self.band(lane, c31);
                    let s0 = self.shuffle(ty, x0, l);
                    let s1 = self.shuffle(ty, x1, l);
                    let c32 = self.cu(32);
                    let high = self.ule(c32, lane);
                    let t = self.ty(ty);
                    self.select(t, high, s1, s0)
                } else {
                    self.shuffle(ty, x, lane)
                }
            }
            Op::UserData(r) => super::memory::user_data(self, r)?,
            Op::SystemValue(s) => super::io::system_value(self, s)?,
            Op::CodeAddress => self.code_address()?,
            Op::InterpParam { attr, chan, param } => super::io::interp_param(self, attr, chan, param)?,
            Op::ScalarLoad { .. }
            | Op::ScalarBufferLoad { .. }
            | Op::BufferLoad { .. }
            | Op::BufferStore { .. }
            | Op::BufferAtomic { .. }
            | Op::GlobalLoad { .. }
            | Op::GlobalStore { .. }
            | Op::GlobalAtomic { .. }
            | Op::ScratchLoad { .. }
            | Op::ScratchStore { .. }
            | Op::LdsLoad { .. }
            | Op::LdsStore { .. }
            | Op::LdsAtomic { .. }
            | Op::LdsAppend { .. }
            | Op::GdsLoad { .. }
            | Op::GdsStore { .. }
            | Op::GdsAtomic { .. }
            | Op::GdsAppend { .. } => return super::memory::emit(self, v, d),
            Op::Image(attr) => return super::image::emit(self, v, d, attr),
            Op::Export(attr) => {
                super::io::export(self, v, attr)?;
                return Ok(None);
            }
            Op::Barrier => {
                if self.stage == ShaderStage::Compute || self.mesh.is_some() {
                    let (s, sem) = (self.cu(SCOPE_WORKGROUP), self.cu(barrier_semantics(self.writes_buffers, self.writes_images)));
                    self.void(S::ControlBarrier, vec![Operand::IdRef(s), Operand::IdRef(s), Operand::IdRef(sem)]);
                }
                return Ok(None);
            }
            Op::DataCacheInvalidate => {
                if !self.coherent_buffers {
                    return Ok(None);
                }
                // Device-scope acquire of uniform (buffer) and image memory.
                let (s, sem) = (self.cu(SCOPE_DEVICE), self.cu(0x842));
                self.void(S::MemoryBarrier, vec![Operand::IdRef(s), Operand::IdRef(sem)]);
                return Ok(None);
            }
            Op::SubgroupLdsBarrier => {
                if self.stage == ShaderStage::Compute || self.mesh.is_some() {
                    // A work-group of one wave keeps the control barrier:
                    // every invocation reaches it (the wave's control flow
                    // is uniform). On AMD's proprietary driver a
                    // subgroup-scope OpControlBarrier compiles to
                    // s_barrier, which waves in a branch paired with other
                    // waves' barriers hang on, so a work-group of several
                    // waves orders its LDS with a subgroup-scope memory
                    // barrier instead.
                    let sem = self.cu(0x108);
                    if self.feat.amd_proprietary && self.multi_wave_workgroup() {
                        let s = self.cu(SCOPE_SUBGROUP);
                        self.void(S::MemoryBarrier, vec![Operand::IdRef(s), Operand::IdRef(sem)]);
                    } else {
                        let (s, m) = (self.cu(SCOPE_SUBGROUP), self.cu(SCOPE_WORKGROUP));
                        self.void(S::ControlBarrier, vec![Operand::IdRef(s), Operand::IdRef(m), Operand::IdRef(sem)]);
                    }
                }
                return Ok(None);
            }
            Op::SendMessage { message } => {
                if self.mesh.is_some() {
                    let m0 = self.arg(v, 0)?;
                    super::mesh::send_message(self, message as u32, m0)?;
                }
                return Ok(None);
            }
            Op::Clock { realtime } => self.clock(realtime),
            _ => self.alu(v, d)?,
        };
        Ok(Some(r))
    }

    /// Whether the work-group this program runs in holds more than one
    /// host subgroup: several guest waves, or a work-group size past the
    /// host subgroup (a size the request does not carry counts: the safe
    /// answer). A split wave64 program exchanges through workgroup memory,
    /// so its work-group is several host subgroups by construction.
    fn multi_wave_workgroup(&self) -> bool {
        if self.split.is_some() {
            return true;
        }
        let s = self.input.request.local_size;
        let invocations = s[0].max(1) as u64 * s[1].max(1) as u64 * s[2].max(1) as u64 / self.lanes.max(1) as u64;
        let subgroup = self.feat.subgroup_size.max(1) as u64;
        invocations == 0 || invocations > subgroup
    }

    /// `s_memrealtime` (`realtime`) / `s_memtime`.
    ///
    /// The real-time clock is the device clock scaled to the 100 MHz
    /// reference clock of the host's CP timestamps ([`crate::clock`]):
    /// guest code compares the two and spins on it until a deadline passes.
    /// `s_memtime` (the shader core clock, read for intervals) is the
    /// subgroup clock, else the unscaled device clock. Without a device
    /// clock both are a per-invocation counter that grows by one per read.
    fn clock(&mut self, realtime: bool) -> Word {
        let t = self.ty(Type::U64);
        if !realtime && self.feat.subgroup_clock {
            return self.read_clock(spirv::Scope::Subgroup);
        }
        if let Some(ratio) = self.feat.device_clock {
            let ticks = self.read_clock(spirv::Scope::Device);
            if !realtime {
                return ticks;
            }
            // ticks / den * num + ticks % den * num / den: no overflow.
            let (num, den) = (self.c64(ratio.num as u64), self.c64(ratio.den as u64));
            let q = self.ins(S::UDiv, t, &[ticks, den]);
            let whole = self.ins(S::IMul, t, &[q, num]);
            let r = self.ins(S::UMod, t, &[ticks, den]);
            let rn = self.ins(S::IMul, t, &[r, num]);
            let part = self.ins(S::UDiv, t, &[rn, den]);
            return self.ins(S::IAdd, t, &[whole, part]);
        }
        let var = match self.clock {
            Some(v) => v,
            None => {
                // Initialised: an undefined Private start value lets the
                // driver fold the reads away.
                let zero = self.c64(0);
                let p = self.ptr(SC::Private, t);
                let v = self.b.id();
                let inst = Instruction::new(S::Variable, Some(p), Some(v), vec![Operand::StorageClass(SC::Private), Operand::IdRef(zero)]);
                self.b.module_mut().types_global_values.push(inst);
                self.globals.push(v);
                self.clock = Some(v);
                v
            }
        };
        let old = self.load(t, var);
        let one = self.c64(1);
        let new = self.ins(S::IAdd, t, &[old, one]);
        self.store(var, new);
        new
    }

    fn read_clock(&mut self, scope: spirv::Scope) -> Word {
        self.cap(Capability::ShaderClockKHR);
        self.exts.insert("SPV_KHR_shader_clock");
        let t = self.ty(Type::U64);
        let scope = self.cu(scope as u32);
        self.ins(S::ReadClockKHR, t, &[scope])
    }

    // ------------------------------------------------------------------
    // Subgroup
    // ------------------------------------------------------------------

    fn subgroup_caps(&mut self) {
        self.cap(Capability::GroupNonUniform);
    }

    /// The host subgroup invocation id.
    pub fn host_lane(&mut self) -> Word {
        self.subgroup_caps();
        let t = self.t_u32();
        self.builtin_value(spirv::BuiltIn::SubgroupLocalInvocationId, t)
    }

    pub fn lane_id(&mut self) -> Word {
        if self.split.is_some() {
            return super::wave_split::guest_lane(self);
        }
        let l = self.host_lane();
        if self.half == 1 {
            let t = self.t_u32();
            let c = self.cu(32);
            self.hoist(S::IAdd, t, &[l, c])
        } else {
            l
        }
    }

    /// Ballot of the predicate `p` (`p1`: the second lane half's).
    pub fn ballot2(&mut self, p: Word, p1: Option<Word>) -> Word {
        self.subgroup_caps();
        self.cap(Capability::GroupNonUniformBallot);
        let t4 = self.t_uvec(4);
        let scope = self.cu(SCOPE_SUBGROUP);
        let m = self.ins(S::GroupNonUniformBallot, t4, &[scope, p]);
        let tu = self.t_u32();
        let lo = self.extract(tu, m, 0);
        let hi = if let Some(p1) = p1 {
            let m1 = self.ins(S::GroupNonUniformBallot, t4, &[scope, p1]);
            self.extract(tu, m1, 0)
        } else if self.f.wave_size == 64 {
            self.extract(tu, m, 1)
        } else {
            self.cu(0)
        };
        let t2 = self.t_uvec(2);
        self.construct(t2, &[lo, hi])
    }

    pub fn inverse_ballot(&mut self, lo: Word, hi: Word) -> Word {
        let lane = self.lane_id();
        let tu = self.t_u32();
        let mask = if self.f.wave_size == 64 {
            let c32 = self.cu(32);
            let high = self.ule(c32, lane);
            self.select(tu, high, hi, lo)
        } else {
            lo
        };
        let c31 = self.cu(31);
        let sh = self.band(lane, c31);
        let bit = self.shr(mask, sh);
        let one = self.cu(1);
        let b = self.band(bit, one);
        let zero = self.cu(0);
        self.ine(b, zero)
    }

    /// Value of `x` in invocation `lane` (32-bit lanes; wider and narrower
    /// types are split or widened).
    pub fn shuffle(&mut self, ty: Type, x: Word, lane: Word) -> Word {
        self.lane_op(S::GroupNonUniformShuffle, ty, x, lane)
    }

    /// A subgroup operation that reads `x` from another invocation
    /// (Shuffle, ShuffleXor / Up / Down, Broadcast) with its lane operand
    /// (32-bit lanes; wider and narrower types are split or widened).
    pub fn lane_op(&mut self, op: S, ty: Type, x: Word, lane: Word) -> Word {
        self.subgroup_caps();
        match op {
            S::GroupNonUniformBroadcast => self.cap(Capability::GroupNonUniformBallot),
            S::GroupNonUniformShuffleUp | S::GroupNonUniformShuffleDown => self.cap(Capability::GroupNonUniformShuffleRelative),
            _ => self.cap(Capability::GroupNonUniformShuffle),
        }
        let scope = self.cu(SCOPE_SUBGROUP);
        let s = ty.component();
        match s.bits() {
            64 => {
                let n = ty.components() * 2;
                let carrier = Type::vec(Scalar::U32, n.min(4));
                if n > 4 {
                    // u64x3/x4: shuffle component-wise.
                    let et = self.ty(Type::Scalar(s));
                    let parts: Vec<Word> = (0..ty.components()).map(|i| {
                        let c = self.extract(et, x, i);
                        self.lane_op(op, Type::Scalar(s), c, lane)
                    }).collect();
                    let t = self.ty(ty);
                    return self.construct(t, &parts);
                }
                let ct = self.ty(carrier);
                let c = self.bitcast(ct, x);
                let r = self.ins(op, ct, &[scope, c, lane]);
                let t = self.ty(ty);
                self.bitcast(t, r)
            }
            16 => {
                let n = ty.components();
                let ut = Type::vec(Scalar::U16, n);
                let wt = Type::vec(Scalar::U32, n);
                let (ut_w, wt_w) = (self.ty(ut), self.ty(wt));
                let u = if s == Scalar::F16 { self.bitcast(ut_w, x) } else { x };
                let w = self.ins(S::UConvert, wt_w, &[u]);
                let r = self.ins(op, wt_w, &[scope, w, lane]);
                let n16 = self.ins(S::UConvert, ut_w, &[r]);
                if s == Scalar::F16 {
                    let t = self.ty(ty);
                    self.bitcast(t, n16)
                } else {
                    n16
                }
            }
            _ => {
                let t = self.ty(ty);
                self.ins(op, t, &[scope, x, lane])
            }
        }
    }

    /// [`EmitOptions::subgroup_ops`](super::EmitOptions): v_readlane /
    /// v_readfirstlane as a broadcast of their uniform lane, and a shuffle
    /// whose source lane is a fixed function of the lane as ShuffleXor /
    /// ShuffleUp / ShuffleDown / a broadcast, where that reads the same
    /// lane in every invocation whose result is used (None: the generic
    /// shuffle).
    fn lane_read(&mut self, v: Value, op: Op, ty: Type) -> R<Option<Word>> {
        if !self.feat.emit.subgroup_ops {
            return Ok(None);
        }
        let host = self.feat.subgroup_size;
        let lane_v = self.f.args(v)[1];
        if op == Op::ReadLane {
            // The lane is uniform (the IR's invariant for read_lane).
            if let Op::Const(k) = self.f.op(lane_v) {
                let k = k as u32;
                if let Some(w) = self.lane_reduction(v, k)? {
                    return Ok(Some(w));
                }
                let half = (k / 32) as usize;
                let within = if self.lanes == 2 { k % 32 } else { k };
                if within >= host || (self.lanes == 2 && half >= 2) {
                    return Ok(None);
                }
                let x = if self.lanes == 2 { self.arg_half(v, 0, half)? } else { self.arg(v, 0)? };
                let id = self.cu(within);
                return Ok(Some(self.lane_op(S::GroupNonUniformBroadcast, ty, x, id)));
            }
            if !self.feat.emit.broadcast_dynamic {
                return Ok(None);
            }
            let lane = self.arg(v, 1)?;
            if self.lanes == 2 {
                let x0 = self.arg_half(v, 0, 0)?;
                let x1 = self.arg_half(v, 0, 1)?;
                let c31 = self.cu(31);
                let l = self.band(lane, c31);
                let b0 = self.lane_op(S::GroupNonUniformBroadcast, ty, x0, l);
                let b1 = self.lane_op(S::GroupNonUniformBroadcast, ty, x1, l);
                let c32 = self.cu(32);
                let high = self.ule(c32, lane);
                let t = self.ty(ty);
                return Ok(Some(self.select(t, high, b1, b0)));
            }
            let x = self.arg(v, 0)?;
            return Ok(Some(self.lane_op(S::GroupNonUniformBroadcast, ty, x, lane)));
        }
        // A shuffle: classify its source lane over this half's lanes.
        let lanes = self.lanes.max(1) as u32;
        if self.f.wave_size != host * lanes || host > 64 {
            return Ok(None);
        }
        let Some(table) = self.lane_table(lane_v, 0) else { return Ok(None) };
        let care = self.used_lanes(v);
        let base = if self.lanes == 2 { 32 * self.half as u32 } else { 0 };
        let lanes_here: Vec<u32> = (base..base + host).filter(|&g| care[g as usize]).collect();
        if lanes_here.is_empty() {
            return Ok(None);
        }
        let src = |g: u32| table[g as usize];
        let inside = |t: u64| t >= base as u64 && t < (base + host) as u64;
        let x = self.arg(v, 0)?;
        if lanes_here.iter().all(|&g| src(g) == g as u64) {
            return Ok(Some(x));
        }
        let first = src(lanes_here[0]);
        if lanes_here.iter().all(|&g| src(g) == first) && first < self.f.wave_size as u64 {
            // Every used lane reads one lane: a broadcast of it.
            let half = if self.lanes == 2 { (first / 32) as usize } else { 0 };
            let within = (first - 32 * half as u64) as u32;
            if within < host {
                let x = if self.lanes == 2 { self.arg_half(v, 0, half)? } else { x };
                let id = self.cu(within);
                return Ok(Some(self.lane_op(S::GroupNonUniformBroadcast, ty, x, id)));
            }
        }
        let g0 = lanes_here[0];
        let mask = first ^ g0 as u64;
        if mask != 0 && mask < host as u64 && lanes_here.iter().all(|&g| src(g) == (g as u64 ^ mask) && inside(src(g))) {
            let c = self.cu(mask as u32);
            return Ok(Some(self.lane_op(S::GroupNonUniformShuffleXor, ty, x, c)));
        }
        if self.feat.emit.shuffle_relative {
            if first < g0 as u64 {
                let delta = g0 as u64 - first;
                if lanes_here.iter().all(|&g| src(g) + delta == g as u64 && inside(src(g))) {
                    let c = self.cu(delta as u32);
                    return Ok(Some(self.lane_op(S::GroupNonUniformShuffleUp, ty, x, c)));
                }
            } else if first > g0 as u64 {
                let delta = first - g0 as u64;
                if lanes_here.iter().all(|&g| src(g) == g as u64 + delta && inside(src(g))) {
                    let c = self.cu(delta as u32);
                    return Ok(Some(self.lane_op(S::GroupNonUniformShuffleDown, ty, x, c)));
                }
            }
        }
        Ok(None)
    }

    /// The value of `v` in each guest lane 0..64 when it is a function of
    /// the lane id and constants only (u32 / u64 integers and booleans as
    /// 0 / 1; None otherwise).
    pub(super) fn lane_table(&mut self, v: Value, depth: u32) -> Option<Box<[u64; 64]>> {
        if let Some(t) = self.lane_tables.get(&v) {
            return t.clone();
        }
        let t = self.lane_table_uncached(v, depth);
        self.lane_tables.insert(v, t.clone());
        t
    }

    fn lane_table_uncached(&mut self, v: Value, depth: u32) -> Option<Box<[u64; 64]>> {
        if depth > 24 {
            return None;
        }
        let d = self.f.value(v).clone();
        let mask = match d.ty {
            Type::Scalar(Scalar::Bool) => 1u64,
            Type::Scalar(Scalar::U32) => 0xFFFF_FFFF,
            Type::Scalar(Scalar::U64) => u64::MAX,
            _ => return None,
        };
        let mut out = Box::new([0u64; 64]);
        let arg = |e: &mut Self, i: usize| e.lane_table(d.args[i], depth + 1);
        match d.op {
            Op::Const(c) => out.fill(c & mask),
            Op::LaneId => {
                for (g, o) in out.iter_mut().enumerate() {
                    *o = g as u64;
                }
            }
            Op::Select => {
                let (c, a, b) = (arg(self, 0)?, arg(self, 1)?, arg(self, 2)?);
                for g in 0..64 {
                    out[g] = if c[g] != 0 { a[g] } else { b[g] };
                }
            }
            Op::ICmp(p) => {
                let (a, b) = (arg(self, 0)?, arg(self, 1)?);
                let wide = self.f.ty(d.args[0]) == Type::U64;
                for g in 0..64 {
                    let (x, y) = (a[g], b[g]);
                    let (sx, sy) = if wide { (x as i64, y as i64) } else { (x as u32 as i32 as i64, y as u32 as i32 as i64) };
                    out[g] = match p {
                        ICmp::Eq => x == y,
                        ICmp::Ne => x != y,
                        ICmp::ULt => x < y,
                        ICmp::ULe => x <= y,
                        ICmp::UGt => x > y,
                        ICmp::UGe => x >= y,
                        ICmp::SLt => sx < sy,
                        ICmp::SLe => sx <= sy,
                        ICmp::SGt => sx > sy,
                        ICmp::SGe => sx >= sy,
                    } as u64;
                }
            }
            Op::IAdd | Op::ISub | Op::IMul | Op::And | Op::Or | Op::Xor | Op::Shl | Op::LShr | Op::UMin | Op::UMax => {
                let (a, b) = (arg(self, 0)?, arg(self, 1)?);
                let bits = if mask == u64::MAX { 64 } else { 32 };
                for g in 0..64 {
                    let (x, y) = (a[g], b[g]);
                    let r = match d.op {
                        Op::IAdd => x.wrapping_add(y),
                        Op::ISub => x.wrapping_sub(y),
                        Op::IMul => x.wrapping_mul(y),
                        Op::And => x & y,
                        Op::Or => x | y,
                        Op::Xor => x ^ y,
                        Op::Shl => {
                            if y >= bits {
                                return None;
                            }
                            x << y
                        }
                        Op::LShr => {
                            if y >= bits {
                                return None;
                            }
                            x >> y
                        }
                        Op::UMin => x.min(y),
                        _ => x.max(y),
                    };
                    out[g] = r & mask;
                }
            }
            Op::Not => {
                let a = arg(self, 0)?;
                for g in 0..64 {
                    out[g] = !a[g] & mask;
                }
            }
            _ => return None,
        }
        Some(out)
    }

    /// The guest lanes in which the result of shuffle `v` is used: every
    /// lane, unless each use selects it under a condition that is false
    /// in some lanes as a function of the lane alone (a DPP source lane
    /// out of its row).
    fn used_lanes(&mut self, v: Value) -> [bool; 64] {
        if self.users.is_none() {
            let mut users: HashMap<Value, Vec<Value>> = HashMap::new();
            for &b in &self.f.order {
                for &u in &self.f.block(b).insts {
                    for &a in self.f.args(u) {
                        users.entry(a).or_default().push(u);
                    }
                }
                if let Terminator::CondBranch { cond, .. } = self.f.block(b).term {
                    // A branch use: every lane.
                    users.entry(cond).or_default().push(Value(u32::MAX));
                }
            }
            self.users = Some(users);
        }
        let list = self.users.as_ref().and_then(|u| u.get(&v)).cloned().unwrap_or_default();
        let mut care = [false; 64];
        if list.is_empty() {
            return [true; 64];
        }
        for u in list {
            if u.0 == u32::MAX || self.f.op(u) != Op::Select {
                return [true; 64];
            }
            let args = self.f.args(u).to_vec();
            if args[1] != v || args[0] == v || args[2] == v {
                return [true; 64];
            }
            let c = self.condition_lanes(args[0], 0);
            for g in 0..64 {
                care[g] |= c[g];
            }
        }
        care
    }

    /// Lanes in which the boolean `c` may be true (a conjunction of terms,
    /// those that are lane functions decided per lane).
    fn condition_lanes(&mut self, c: Value, depth: u32) -> [bool; 64] {
        if depth > 8 {
            return [true; 64];
        }
        if let Some(t) = self.lane_table(c, 0) {
            let mut r = [false; 64];
            for g in 0..64 {
                r[g] = t[g] != 0;
            }
            return r;
        }
        let d = self.f.value(c).clone();
        if d.op == Op::And && d.ty == Type::BOOL {
            let a = self.condition_lanes(d.args[0], depth + 1);
            let b = self.condition_lanes(d.args[1], depth + 1);
            let mut r = [false; 64];
            for g in 0..64 {
                r[g] = a[g] && b[g];
            }
            return r;
        }
        [true; 64]
    }

    // ------------------------------------------------------------------
    // ALU
    // ------------------------------------------------------------------

    /// f32 log2 with the series of ln(1 + d) / ln 2 (d = x - 1, eight
    /// terms) where |d| <= 1/8 and the host's log2 is more than 4 ULP away
    /// from it: host drivers lose relative precision there while the
    /// hardware stays within a few ULP (measured near 1.0 on RDNA2).
    fn log2_near_one(&mut self, t: Word, x: Word) -> Word {
        if self.feat.emit.log_series_branch {
            // Only an invocation whose input is near 1.0 computes the
            // series (the same result as computing it always).
            let value = self.gl(t, GLOp::Log2, &[x]);
            let one = self.cf(1.0);
            let d = self.ins(S::FSub, t, &[x, one]);
            let d = self.precise(d);
            let ad = self.gl(t, GLOp::FAbs, &[d]);
            let eighth = self.cf(0.125);
            let tb = self.t_bool();
            let near = self.ins(S::FOrdLessThanEqual, tb, &[ad, eighth]);
            return self.if_value(near, t, value, |e| {
                let series = e.log2_series(t, d);
                let tu = e.t_u32();
                let vb = e.bitcast(tu, value);
                let sb = e.bitcast(tu, series);
                let up = e.isub(vb, sb);
                let down = e.isub(sb, vb);
                let distance = e.umin(up, down);
                let four = e.cu(4);
                let far = e.ult(four, distance);
                e.select(t, far, series, value)
            });
        }
        let value = self.gl(t, GLOp::Log2, &[x]);
        let one = self.cf(1.0);
        let d = self.ins(S::FSub, t, &[x, one]);
        let d = self.precise(d);
        let mut poly = self.konst(Type::F32, 0xBE00_0000);
        for c in [0x3E12_4925u32, 0xBE2A_AAAB, 0x3E4C_CCCD, 0xBE80_0000, 0x3EAA_AAAB, 0xBF00_0000, 0x3F80_0000] {
            let k = self.konst(Type::F32, c as u64);
            let f = self.gl(t, GLOp::Fma, &[poly, d, k]);
            poly = self.precise(f);
        }
        let product = self.ins(S::FMul, t, &[d, poly]);
        let product = self.precise(product);
        let inv_ln2 = self.konst(Type::F32, 0x3FB8_AA3B);
        let series = self.ins(S::FMul, t, &[product, inv_ln2]);
        let series = self.precise(series);
        let tu = self.t_u32();
        let vb = self.bitcast(tu, value);
        let sb = self.bitcast(tu, series);
        let up = self.isub(vb, sb);
        let down = self.isub(sb, vb);
        let distance = self.umin(up, down);
        let ad = self.gl(t, GLOp::FAbs, &[d]);
        let eighth = self.cf(0.125);
        let tb = self.t_bool();
        let near = self.ins(S::FOrdLessThanEqual, tb, &[ad, eighth]);
        let four = self.cu(4);
        let far = self.ult(four, distance);
        let pick = self.land(near, far);
        self.select(t, pick, series, value)
    }

    /// The eight-term series of ln(1 + d) / ln 2.
    fn log2_series(&mut self, t: Word, d: Word) -> Word {
        let mut poly = self.konst(Type::F32, 0xBE00_0000);
        for c in [0x3E12_4925u32, 0xBE2A_AAAB, 0x3E4C_CCCD, 0xBE80_0000, 0x3EAA_AAAB, 0xBF00_0000, 0x3F80_0000] {
            let k = self.konst(Type::F32, c as u64);
            let f = self.gl(t, GLOp::Fma, &[poly, d, k]);
            poly = self.precise(f);
        }
        let product = self.ins(S::FMul, t, &[d, poly]);
        let product = self.precise(product);
        let inv_ln2 = self.konst(Type::F32, 0x3FB8_AA3B);
        let series = self.ins(S::FMul, t, &[product, inv_ln2]);
        self.precise(series)
    }

    fn precise(&mut self, id: Word) -> Word {
        self.decorate(id, Decoration::NoContraction, &[]);
        id
    }

    fn alu(&mut self, v: Value, d: &ValueData) -> R<Word> {
        let ty = d.ty;
        let t = self.ty(ty);
        let a0ty = if d.args.is_empty() { Type::Void } else { self.arg_ty(v, 0) };
        let bool_args = a0ty.is_bool();
        let a = |e: &mut Self, i: usize| e.arg(v, i);
        let bin = |e: &mut Self, op: S| -> R<Word> {
            let (x, y) = (a(e, 0)?, a(e, 1)?);
            Ok(e.ins(op, t, &[x, y]))
        };
        let un = |e: &mut Self, op: S| -> R<Word> {
            let x = a(e, 0)?;
            Ok(e.ins(op, t, &[x]))
        };
        let gl1 = |e: &mut Self, op: GLOp| -> R<Word> {
            let x = a(e, 0)?;
            Ok(e.gl(t, op, &[x]))
        };
        let gl2 = |e: &mut Self, op: GLOp| -> R<Word> {
            let (x, y) = (a(e, 0)?, a(e, 1)?);
            Ok(e.gl(t, op, &[x, y]))
        };
        let width = ty.scalar().map_or(0, Scalar::bits);
        Ok(match d.op {
            Op::IAdd => bin(self, S::IAdd)?,
            Op::ISub => bin(self, S::ISub)?,
            Op::IMul => bin(self, S::IMul)?,
            Op::UMulHi | Op::SMulHi => {
                let st = self.b.type_struct([t, t]);
                let (x, y) = (a(self, 0)?, a(self, 1)?);
                let op = if d.op == Op::UMulHi { S::UMulExtended } else { S::SMulExtended };
                let r = self.ins(op, st, &[x, y]);
                self.extract(t, r, 1)
            }
            Op::UDiv => bin(self, S::UDiv)?,
            Op::UMod => bin(self, S::UMod)?,
            Op::IAddCarry | Op::ISubBorrow => {
                let tu = self.t_u32();
                let st = self.b.type_struct([tu, tu]);
                let (x, y) = (a(self, 0)?, a(self, 1)?);
                let op = if d.op == Op::IAddCarry { S::IAddCarry } else { S::ISubBorrow };
                let r = self.ins(op, st, &[x, y]);
                let s0 = self.extract(tu, r, 0);
                let s1 = self.extract(tu, r, 1);
                self.construct(t, &[s0, s1])
            }
            Op::INeg => un(self, S::SNegate)?,
            Op::SMin => gl2(self, GLOp::SMin)?,
            Op::SMax => gl2(self, GLOp::SMax)?,
            Op::UMin => gl2(self, GLOp::UMin)?,
            Op::UMax => gl2(self, GLOp::UMax)?,
            Op::Shl => bin(self, S::ShiftLeftLogical)?,
            Op::LShr => bin(self, S::ShiftRightLogical)?,
            Op::AShr => bin(self, S::ShiftRightArithmetic)?,
            Op::And if bool_args => bin(self, S::LogicalAnd)?,
            Op::Or if bool_args => bin(self, S::LogicalOr)?,
            Op::Xor if bool_args => bin(self, S::LogicalNotEqual)?,
            Op::Not if bool_args => un(self, S::LogicalNot)?,
            Op::And => bin(self, S::BitwiseAnd)?,
            Op::Or => bin(self, S::BitwiseOr)?,
            Op::Xor => bin(self, S::BitwiseXor)?,
            Op::Not => un(self, S::Not)?,
            Op::BitReverse => {
                let x = a(self, 0)?;
                self.bit_reverse(ty, x)
            }
            Op::BitCount => {
                let x = a(self, 0)?;
                self.bit_count(a0ty, x)
            }
            Op::FindLsb => {
                let x = a(self, 0)?;
                self.find_lsb(a0ty, x)
            }
            Op::FindUMsb => {
                let x = a(self, 0)?;
                self.find_msb(a0ty, x, false)
            }
            Op::FindSMsb => {
                let x = a(self, 0)?;
                self.find_msb(a0ty, x, true)
            }
            Op::BitFieldUExtract | Op::BitFieldSExtract => {
                let (x, o, c) = (a(self, 0)?, a(self, 1)?, a(self, 2)?);
                let o = self.to_u32(self.arg_ty(v, 1), o);
                let c = self.to_u32(self.arg_ty(v, 2), c);
                self.bfe(ty, x, o, c, d.op == Op::BitFieldSExtract)
            }
            Op::BitFieldInsert => {
                let (x, ins, o, c) = (a(self, 0)?, a(self, 1)?, a(self, 2)?, a(self, 3)?);
                let o = self.to_u32(self.arg_ty(v, 2), o);
                let c = self.to_u32(self.arg_ty(v, 3), c);
                self.bfi(ty, x, ins, o, c)
            }
            Op::UConvert => {
                let x = a(self, 0)?;
                if a0ty.is_bool() {
                    let (one, zero) = (self.konst(ty, 1), self.konst(ty, 0));
                    self.select(t, x, one, zero)
                } else if a0ty.component().bits() == width {
                    x
                } else {
                    self.ins(S::UConvert, t, &[x])
                }
            }
            Op::SConvert => {
                let x = a(self, 0)?;
                if a0ty.component().bits() == width {
                    x
                } else {
                    self.ins(S::SConvert, t, &[x])
                }
            }
            Op::ICmp(p) => {
                let op = if bool_args {
                    match p {
                        ICmp::Eq => S::LogicalEqual,
                        ICmp::Ne => S::LogicalNotEqual,
                        _ => return Err("ordered comparison of booleans".into()),
                    }
                } else {
                    match p {
                        ICmp::Eq => S::IEqual,
                        ICmp::Ne => S::INotEqual,
                        ICmp::ULt => S::ULessThan,
                        ICmp::ULe => S::ULessThanEqual,
                        ICmp::UGt => S::UGreaterThan,
                        ICmp::UGe => S::UGreaterThanEqual,
                        ICmp::SLt => S::SLessThan,
                        ICmp::SLe => S::SLessThanEqual,
                        ICmp::SGt => S::SGreaterThan,
                        ICmp::SGe => S::SGreaterThanEqual,
                    }
                };
                bin(self, op)?
            }
            Op::FAdd => {
                let r = bin(self, S::FAdd)?;
                self.precise(r)
            }
            Op::FSub => {
                let r = bin(self, S::FSub)?;
                self.precise(r)
            }
            Op::FMul => {
                let r = bin(self, S::FMul)?;
                self.precise(r)
            }
            Op::FDiv => {
                let r = bin(self, S::FDiv)?;
                self.precise(r)
            }
            Op::FFma => {
                let (x, y, z) = (a(self, 0)?, a(self, 1)?, a(self, 2)?);
                self.gl(t, GLOp::Fma, &[x, y, z])
            }
            Op::FNeg => un(self, S::FNegate)?,
            Op::FAbs => gl1(self, GLOp::FAbs)?,
            Op::FMin | Op::FMax if !ty.is_vector() => {
                let (x, y) = (a(self, 0)?, a(self, 1)?);
                self.float_min_max(ty, x, y, d.op == Op::FMax)
            }
            Op::FMin => gl2(self, GLOp::NMin)?,
            Op::FMax => gl2(self, GLOp::NMax)?,
            Op::FFloor => gl1(self, GLOp::Floor)?,
            Op::FCeil => gl1(self, GLOp::Ceil)?,
            Op::FTrunc => gl1(self, GLOp::Trunc)?,
            Op::FRoundEven => gl1(self, GLOp::RoundEven)?,
            Op::FSqrt => gl1(self, GLOp::Sqrt)?,
            Op::FInvSqrt => gl1(self, GLOp::InverseSqrt)?,
            Op::FRcp => {
                let x = a(self, 0)?;
                let one = match ty.component() {
                    Scalar::F16 => self.konst(Type::F16, 0x3C00),
                    Scalar::F64 => self.konst(Type::F64, 1f64.to_bits()),
                    _ => self.cf(1.0),
                };
                let one = if ty.is_vector() {
                    let parts = vec![one; ty.components() as usize];
                    self.construct(t, &parts)
                } else {
                    one
                };
                let r = self.ins(S::FDiv, t, &[one, x]);
                self.precise(r)
            }
            Op::FExp2 | Op::FLog2 | Op::FSin | Op::FCos => {
                let op = match d.op {
                    Op::FExp2 => GLOp::Exp2,
                    Op::FLog2 => GLOp::Log2,
                    Op::FSin => GLOp::Sin,
                    _ => GLOp::Cos,
                };
                let x = a(self, 0)?;
                if ty.component() == Scalar::F64 {
                    // 16/32-bit only: evaluate in f32.
                    let ft = self.ty(Type::vec(Scalar::F32, ty.components()));
                    let n = self.ins(S::FConvert, ft, &[x]);
                    let r = self.gl(ft, op, &[n]);
                    self.ins(S::FConvert, t, &[r])
                } else if d.op == Op::FLog2 && ty == Type::F32 && super::log_series_enabled() && !self.feat.emit.fast_transcendental {
                    self.log2_near_one(t, x)
                } else {
                    self.gl(t, op, &[x])
                }
            }
            Op::FLdexp => {
                let (x, e) = (a(self, 0)?, a(self, 1)?);
                self.ldexp(ty, x, e)
            }
            Op::FCmp(p) => {
                let (x, y) = (a(self, 0)?, a(self, 1)?);
                self.fcmp(p, a0ty, x, y)
            }
            Op::FConvert(r) => {
                let x = a(self, 0)?;
                self.fconvert(a0ty, ty, x, r)
            }
            Op::FToS => un(self, S::ConvertFToS)?,
            Op::FToU => un(self, S::ConvertFToU)?,
            Op::SToF(r) | Op::UToF(r) => {
                let x = a(self, 0)?;
                self.int_to_float(a0ty, ty, x, d.op == Op::SToF(r), r)
            }
            other => return Err(format!("no SPIR-V lowering for {}", other.name())),
        })
    }

    pub fn to_u32(&mut self, ty: Type, x: Word) -> Word {
        match ty.component().bits() {
            32 => x,
            _ => {
                let t = self.t_u32();
                self.ins(S::UConvert, t, &[x])
            }
        }
    }

    /// Low / high dwords of a u64 scalar.
    pub fn split64(&mut self, x: Word) -> (Word, Word) {
        let t2 = self.t_uvec(2);
        let p = self.bitcast(t2, x);
        let tu = self.t_u32();
        (self.extract(tu, p, 0), self.extract(tu, p, 1))
    }

    pub fn join64(&mut self, lo: Word, hi: Word) -> Word {
        let t2 = self.t_uvec(2);
        let p = self.construct(t2, &[lo, hi]);
        let t = self.ty(Type::U64);
        self.bitcast(t, p)
    }

    fn widen16(&mut self, x: Word) -> Word {
        let tu = self.t_u32();
        self.ins(S::UConvert, tu, &[x])
    }

    fn narrow16(&mut self, x: Word) -> Word {
        let t = self.ty(Type::U16);
        self.ins(S::UConvert, t, &[x])
    }

    fn bit_reverse(&mut self, ty: Type, x: Word) -> Word {
        let tu = self.t_u32();
        match ty.component().bits() {
            64 => {
                let (lo, hi) = self.split64(x);
                let rl = self.ins(S::BitReverse, tu, &[lo]);
                let rh = self.ins(S::BitReverse, tu, &[hi]);
                self.join64(rh, rl)
            }
            16 => {
                let w = self.widen16(x);
                let r = self.ins(S::BitReverse, tu, &[w]);
                let c16 = self.cu(16);
                let s = self.shr(r, c16);
                self.narrow16(s)
            }
            _ => {
                let t = self.ty(ty);
                self.ins(S::BitReverse, t, &[x])
            }
        }
    }

    fn bit_count(&mut self, ty: Type, x: Word) -> Word {
        let tu = self.t_u32();
        match ty.component().bits() {
            64 => {
                let (lo, hi) = self.split64(x);
                let a = self.ins(S::BitCount, tu, &[lo]);
                let b = self.ins(S::BitCount, tu, &[hi]);
                self.iadd(a, b)
            }
            16 => {
                let w = self.widen16(x);
                self.ins(S::BitCount, tu, &[w])
            }
            _ => self.ins(S::BitCount, tu, &[x]),
        }
    }

    fn find_lsb(&mut self, ty: Type, x: Word) -> Word {
        let tu = self.t_u32();
        match ty.component().bits() {
            64 => {
                let (lo, hi) = self.split64(x);
                let l = self.gl(tu, GLOp::FindILsb, &[lo]);
                let h = self.gl(tu, GLOp::FindILsb, &[hi]);
                let c32 = self.cu(32);
                let h32 = self.iadd(h, c32);
                let zero = self.cu(0);
                let hz = self.ieq(hi, zero);
                let none = self.cu(u32::MAX);
                let hv = self.select(tu, hz, none, h32);
                let lnz = self.ine(lo, zero);
                self.select(tu, lnz, l, hv)
            }
            16 => {
                let w = self.widen16(x);
                self.gl(tu, GLOp::FindILsb, &[w])
            }
            _ => self.gl(tu, GLOp::FindILsb, &[x]),
        }
    }

    fn find_msb(&mut self, ty: Type, x: Word, signed: bool) -> Word {
        let tu = self.t_u32();
        let op = if signed { GLOp::FindSMsb } else { GLOp::FindUMsb };
        match ty.component().bits() {
            64 => {
                let x = if signed {
                    let t64 = self.ty(Type::U64);
                    let c63 = self.cu(63);
                    let sign = self.ins(S::ShiftRightArithmetic, t64, &[x, c63]);
                    self.ins(S::BitwiseXor, t64, &[x, sign])
                } else {
                    x
                };
                let (lo, hi) = self.split64(x);
                let l = self.gl(tu, GLOp::FindUMsb, &[lo]);
                let h = self.gl(tu, GLOp::FindUMsb, &[hi]);
                let c32 = self.cu(32);
                let h32 = self.iadd(h, c32);
                let zero = self.cu(0);
                let hnz = self.ine(hi, zero);
                self.select(tu, hnz, h32, l)
            }
            16 => {
                let w = if signed {
                    let ti = self.t_u32();
                    self.ins(S::SConvert, ti, &[x])
                } else {
                    self.widen16(x)
                };
                self.gl(tu, op, &[w])
            }
            _ => self.gl(tu, op, &[x]),
        }
    }

    fn bfe(&mut self, ty: Type, x: Word, o: Word, c: Word, signed: bool) -> Word {
        let t = self.ty(ty);
        match ty.component().bits() {
            32 => self.ins(if signed { S::BitFieldSExtract } else { S::BitFieldUExtract }, t, &[x, o, c]),
            16 => {
                let tu = self.t_u32();
                let w = if signed { self.ins(S::SConvert, tu, &[x]) } else { self.widen16(x) };
                let r = self.ins(if signed { S::BitFieldSExtract } else { S::BitFieldUExtract }, tu, &[w, o, c]);
                self.narrow16(r)
            }
            _ => {
                // 64-bit: (x << (64 - o - c)) >> (64 - c), zero for c == 0.
                let c64 = self.cu(64);
                let oc = self.iadd(o, c);
                let left = self.isub(c64, oc);
                let right = self.isub(c64, c);
                let zero = self.cu(0);
                let cz = self.ieq(c, zero);
                let c63 = self.cu(63);
                let left = self.umin(left, c63);
                let right = self.umin(right, c63);
                let sh = self.ins(S::ShiftLeftLogical, t, &[x, left]);
                let r = self.ins(if signed { S::ShiftRightArithmetic } else { S::ShiftRightLogical }, t, &[sh, right]);
                // c == 64 (o == 0): the value itself.
                let full = self.ieq(c, c64);
                let r = self.select(t, full, x, r);
                let z = self.konst(ty, 0);
                self.select(t, cz, z, r)
            }
        }
    }

    fn bfi(&mut self, ty: Type, x: Word, ins: Word, o: Word, c: Word) -> Word {
        let t = self.ty(ty);
        match ty.component().bits() {
            32 => self.ins(S::BitFieldInsert, t, &[x, ins, o, c]),
            16 => {
                let tu = self.t_u32();
                let (wx, wi) = (self.widen16(x), self.widen16(ins));
                let r = self.ins(S::BitFieldInsert, tu, &[wx, wi, o, c]);
                self.narrow16(r)
            }
            _ => {
                let one = self.konst(ty, 1);
                let c63 = self.cu(63);
                let cc = self.umin(c, c63);
                let m = self.ins(S::ShiftLeftLogical, t, &[one, cc]);
                let m = self.ins(S::ISub, t, &[m, one]);
                let c64 = self.cu(64);
                let full = self.ieq(c, c64);
                let all = self.konst(ty, u64::MAX);
                let m = self.select(t, full, all, m);
                let m = self.ins(S::ShiftLeftLogical, t, &[m, o]);
                let nm = self.ins(S::Not, t, &[m]);
                let keep = self.ins(S::BitwiseAnd, t, &[x, nm]);
                let si = self.ins(S::ShiftLeftLogical, t, &[ins, o]);
                let si = self.ins(S::BitwiseAnd, t, &[si, m]);
                self.ins(S::BitwiseOr, t, &[keep, si])
            }
        }
    }

    fn ldexp(&mut self, ty: Type, x: Word, e: Word) -> Word {
        let t = self.ty(ty);
        let ti = self.t_i32();
        let ei = self.bitcast(ti, e);
        // GLSL Ldexp is undefined past the exponent range: clamp (the
        // result saturates long before) and apply in two halves.
        let limit = match ty.component() {
            Scalar::F64 => 2200,
            Scalar::F16 => 80,
            _ => 300,
        };
        let (lo, hi) = (self.ci(-limit), self.ci(limit));
        let ec = self.gl(ti, GLOp::SClamp, &[ei, lo, hi]);
        let two = self.ci(2);
        let half = self.ins(S::SDiv, ti, &[ec, two]);
        let rest = self.ins(S::ISub, ti, &[ec, half]);
        let s1 = self.gl(t, GLOp::Ldexp, &[x, half]);
        self.gl(t, GLOp::Ldexp, &[s1, rest])
    }

    fn fcmp(&mut self, p: FCmp, ty: Type, x: Word, y: Word) -> Word {
        let tb = self.t_bool();
        if ty == Type::F32 && self.f32_compares_flush() && !matches!(p, FCmp::Ord | FCmp::Uno) {
            return self.fcmp_bits(p, x, y);
        }
        let op = match p {
            FCmp::OEq => S::FOrdEqual,
            FCmp::ONe => S::FOrdNotEqual,
            FCmp::OLt => S::FOrdLessThan,
            FCmp::OLe => S::FOrdLessThanEqual,
            FCmp::OGt => S::FOrdGreaterThan,
            FCmp::OGe => S::FOrdGreaterThanEqual,
            FCmp::UEq => S::FUnordEqual,
            FCmp::UNe => S::FUnordNotEqual,
            // Unordered relations as the negated inverse ordered relation
            // (the same result for every input, NaN included): a driver
            // compiler aborts on OpFUnordLessThanEqual /
            // OpFUnordGreaterThanEqual against +infinity.
            FCmp::ULt | FCmp::ULe | FCmp::UGt | FCmp::UGe => {
                let inverse = match p {
                    FCmp::ULt => S::FOrdGreaterThanEqual,
                    FCmp::ULe => S::FOrdGreaterThan,
                    FCmp::UGt => S::FOrdLessThanEqual,
                    _ => S::FOrdLessThan,
                };
                let ordered = self.ins(inverse, tb, &[x, y]);
                return self.lnot(ordered);
            }
            FCmp::Ord | FCmp::Uno => {
                let nx = self.ins(S::IsNan, tb, &[x]);
                let ny = self.ins(S::IsNan, tb, &[y]);
                let any = self.lor(nx, ny);
                return if p == FCmp::Uno { any } else { self.lnot(any) };
            }
        };
        self.ins(op, tb, &[x, y])
    }

    /// [`Self::fcmp`] of two f32 values through their bits: NaNs are
    /// unordered, -0 equals +0, and otherwise the sign-magnitude integers
    /// order as the values do (denormals included).
    /// A host that flushes f32 denormals in every module (no DenormPreserve
    /// 32) compares them as zero: unless the module runs on that flush on
    /// purpose, f32 comparisons compare the bits ([`Self::fcmp_bits`]).
    fn f32_compares_flush(&self) -> bool {
        let native32 = self.input.native_ftz && self.feat.denorm_flush_to_zero32;
        !self.feat.denorm_preserve[1] && !native32
    }

    fn fcmp_bits(&mut self, p: FCmp, x: Word, y: Word) -> Word {
        let tb = self.t_bool();
        let key = |e: &mut Self, v: Word| {
            let tu = e.t_u32();
            let bits = e.bitcast(tu, v);
            let m = e.cu(0x7FFF_FFFF);
            let mag = e.band(bits, m);
            let inf = e.cu(0x7F80_0000);
            let tb = e.t_bool();
            let nan = e.ins(S::UGreaterThan, tb, &[mag, inf]);
            let neg = e.ult(m, bits);
            let negated = e.ins(S::SNegate, tu, &[mag]);
            (e.select(tu, neg, negated, mag), nan)
        };
        let (kx, nx) = key(self, x);
        let (ky, ny) = key(self, y);
        let any_nan = self.lor(nx, ny);
        let ordered = self.lnot(any_nan);
        let rel = |e: &mut Self, op: S| e.ins(op, tb, &[kx, ky]);
        match p {
            FCmp::OEq => {
                let r = rel(self, S::IEqual);
                self.land(ordered, r)
            }
            FCmp::ONe => {
                let r = rel(self, S::INotEqual);
                self.land(ordered, r)
            }
            FCmp::OLt => {
                let r = rel(self, S::SLessThan);
                self.land(ordered, r)
            }
            FCmp::OLe => {
                let r = rel(self, S::SLessThanEqual);
                self.land(ordered, r)
            }
            FCmp::OGt => {
                let r = rel(self, S::SGreaterThan);
                self.land(ordered, r)
            }
            FCmp::OGe => {
                let r = rel(self, S::SGreaterThanEqual);
                self.land(ordered, r)
            }
            FCmp::UEq => {
                let r = rel(self, S::IEqual);
                self.lor(any_nan, r)
            }
            FCmp::UNe => {
                let r = rel(self, S::INotEqual);
                self.lor(any_nan, r)
            }
            FCmp::ULt => {
                let r = rel(self, S::SLessThan);
                self.lor(any_nan, r)
            }
            FCmp::ULe => {
                let r = rel(self, S::SLessThanEqual);
                self.lor(any_nan, r)
            }
            FCmp::UGt => {
                let r = rel(self, S::SGreaterThan);
                self.lor(any_nan, r)
            }
            FCmp::UGe => {
                let r = rel(self, S::SGreaterThanEqual);
                self.lor(any_nan, r)
            }
            FCmp::Ord => ordered,
            FCmp::Uno => any_nan,
        }
    }

    /// Float width conversion; round-toward-zero narrowing is the
    /// round-to-nearest result stepped one unit toward zero when it moved
    /// away from zero.
    /// Hardware minimum / maximum with explicit selects (no reliance on
    /// the host's NaN or signed-zero behaviour): a NaN operand yields the
    /// other one (the first when both are NaN); for two zeros min ORs and
    /// max ANDs the bits, so -0 < +0.
    fn float_min_max(&mut self, ty: Type, x: Word, y: Word, max: bool) -> Word {
        let t = self.ty(ty);
        let (it, mag, inf) = match ty.component() {
            Scalar::F16 => (Type::U16, 0x7FFFu64, 0x7C00u64),
            Scalar::F64 => (Type::U64, 0x7FFF_FFFF_FFFF_FFFF, 0x7FF0_0000_0000_0000),
            _ => (Type::U32, 0x7FFF_FFFF, 0x7F80_0000),
        };
        let iw = self.ty(it);
        let tb = self.t_bool();
        let xb = self.bitcast(iw, x);
        let yb = self.bitcast(iw, y);
        let mm = self.konst(it, mag);
        let infc = self.konst(it, inf);
        let zero = self.konst(it, 0);
        let xm = self.ins(S::BitwiseAnd, iw, &[xb, mm]);
        let ym = self.ins(S::BitwiseAnd, iw, &[yb, mm]);
        // Denormal operands are ordered by value also where the host
        // compares them as zero (med3 picks among denormals by them).
        let c = if ty == Type::F32 && self.f32_compares_flush() {
            self.fcmp_bits(if max { FCmp::OGe } else { FCmp::OLt }, x, y)
        } else {
            let cmp = if max { S::FOrdGreaterThanEqual } else { S::FOrdLessThan };
            self.ins(cmp, tb, &[x, y])
        };
        let r = self.select(iw, c, xb, yb);
        let xz = self.ins(S::IEqual, tb, &[xm, zero]);
        let yz = self.ins(S::IEqual, tb, &[ym, zero]);
        let zz = self.ins(S::LogicalAnd, tb, &[xz, yz]);
        let comb = self.ins(if max { S::BitwiseAnd } else { S::BitwiseOr }, iw, &[xb, yb]);
        let r = self.select(iw, zz, comb, r);
        let xn = self.ins(S::UGreaterThan, tb, &[xm, infc]);
        let r = self.select(iw, xn, yb, r);
        let yn = self.ins(S::UGreaterThan, tb, &[ym, infc]);
        let r = self.select(iw, yn, xb, r);
        self.bitcast(t, r)
    }

    /// f32 -> f16 bits (in a u32) rounded to nearest even, computed with
    /// integer operations: the host's FConvert rounding, overflow and NaN
    /// results are implementation-defined. Overflow gives infinity, NaN
    /// keeps the top mantissa bits and is quieted.
    pub fn f32_to_f16_bits(&mut self, value: Word) -> Word {
        let u = self.t_u32();
        let tb = self.t_bool();
        let round_even = |e: &mut Self, m: Word, shift: Word| {
            let one = e.cu(1);
            let trunc = e.ins(S::ShiftRightLogical, u, &[m, shift]);
            let unit = e.ins(S::ShiftLeftLogical, u, &[one, shift]);
            let low = e.ins(S::ISub, u, &[unit, one]);
            let rem = e.ins(S::BitwiseAnd, u, &[m, low]);
            let sm1 = e.ins(S::ISub, u, &[shift, one]);
            let half = e.ins(S::ShiftLeftLogical, u, &[one, sm1]);
            let above = e.ins(S::UGreaterThan, tb, &[rem, half]);
            let at = e.ins(S::IEqual, tb, &[rem, half]);
            let odd_bit = e.ins(S::BitwiseAnd, u, &[trunc, one]);
            let zero = e.cu(0);
            let odd = e.ins(S::INotEqual, tb, &[odd_bit, zero]);
            let tie = e.ins(S::LogicalAnd, tb, &[at, odd]);
            let up = e.ins(S::LogicalOr, tb, &[above, tie]);
            let inc = e.select(u, up, one, zero);
            e.ins(S::IAdd, u, &[trunc, inc])
        };
        let bits = self.bitcast(u, value);
        let c16 = self.cu(16);
        let s = self.ins(S::ShiftRightLogical, u, &[bits, c16]);
        let c8000 = self.cu(0x8000);
        let sign = self.ins(S::BitwiseAnd, u, &[s, c8000]);
        let mm = self.cu(0x7FFF_FFFF);
        let mag = self.ins(S::BitwiseAnd, u, &[bits, mm]);
        let c23 = self.cu(23);
        let exp = self.ins(S::ShiftRightLogical, u, &[mag, c23]);
        let c13 = self.cu(13);
        let rn = round_even(self, mag, c13);
        let bias = self.cu(112 << 10);
        let normal = self.ins(S::ISub, u, &[rn, bias]);
        let mant_mask = self.cu(0x7F_FFFF);
        let implicit = self.cu(0x80_0000);
        let mant = self.ins(S::BitwiseAnd, u, &[mag, mant_mask]);
        let mant = self.ins(S::BitwiseOr, u, &[mant, implicit]);
        let c102 = self.cu(102);
        let tiny = self.ins(S::ULessThan, tb, &[exp, c102]);
        let c126 = self.cu(126);
        let c31 = self.cu(31);
        let sh = self.ins(S::ISub, u, &[c126, exp]);
        let shift = self.select(u, tiny, c31, sh);
        let sub = round_even(self, mant, shift);
        let zero = self.cu(0);
        let subnormal = self.select(u, tiny, zero, sub);
        let min_normal = self.cu(0x3880_0000);
        let is_sub = self.ins(S::ULessThan, tb, &[mag, min_normal]);
        let r = self.select(u, is_sub, subnormal, normal);
        let ovf_lim = self.cu(0x477F_F000);
        let ovf = self.ins(S::UGreaterThanEqual, tb, &[mag, ovf_lim]);
        let inf = self.cu(0x7C00);
        let r = self.select(u, ovf, inf, r);
        let qnan = self.cu(0x7E00);
        let nm = self.ins(S::BitwiseAnd, u, &[mag, mant_mask]);
        let nm = self.ins(S::ShiftRightLogical, u, &[nm, c13]);
        let nan = self.ins(S::BitwiseOr, u, &[qnan, nm]);
        let inf32 = self.cu(0x7F80_0000);
        let is_nan = self.ins(S::UGreaterThan, tb, &[mag, inf32]);
        let r = self.select(u, is_nan, nan, r);
        self.ins(S::BitwiseOr, u, &[sign, r])
    }

    /// f32 → f16 bits (low 16 bits of a u32), rounded toward zero, with
    /// integer operations: a finite value past the largest half saturates
    /// to it (65504), subnormal halves are kept, infinities stay infinite
    /// and NaNs keep the quiet NaN with the high payload bits.
    pub fn f32_to_f16_bits_rtz(&mut self, value: Word) -> Word {
        let u = self.t_u32();
        let tb = self.t_bool();
        let bits = self.bitcast(u, value);
        let c16 = self.cu(16);
        let s = self.ins(S::ShiftRightLogical, u, &[bits, c16]);
        let c8000 = self.cu(0x8000);
        let sign = self.ins(S::BitwiseAnd, u, &[s, c8000]);
        let mm = self.cu(0x7FFF_FFFF);
        let abs = self.ins(S::BitwiseAnd, u, &[bits, mm]);
        let c23 = self.cu(23);
        let exp = self.ins(S::ShiftRightLogical, u, &[abs, c23]);
        let mant_mask = self.cu(0x7F_FFFF);
        let mant = self.ins(S::BitwiseAnd, u, &[abs, mant_mask]);
        // Subnormal half: the mantissa with its hidden bit, shifted to the
        // half's 2^-24 unit (shifts kept below 32).
        let c126 = self.cu(126);
        let c31 = self.cu(31);
        let sh = self.ins(S::ISub, u, &[c126, exp]);
        let sh = self.umin(sh, c31);
        let hidden = self.cu(0x80_0000);
        let full = self.ins(S::BitwiseOr, u, &[mant, hidden]);
        let sub = self.ins(S::ShiftRightLogical, u, &[full, sh]);
        // Normal half: drop 13 mantissa bits and rebias; finite overflow
        // saturates to the largest half.
        let c13 = self.cu(13);
        let truncated = self.ins(S::ShiftRightLogical, u, &[abs, c13]);
        let rebias = self.cu(0x1C000);
        let normal = self.ins(S::ISub, u, &[truncated, rebias]);
        let max_half = self.cu(0x7BFF);
        let normal = self.umin(normal, max_half);
        let min_normal = self.cu(0x3880_0000);
        let is_sub = self.ins(S::ULessThan, tb, &[abs, min_normal]);
        let finite = self.select(u, is_sub, sub, normal);
        let c3ff = self.cu(0x3FF);
        let payload = self.ins(S::BitwiseAnd, u, &[truncated, c3ff]);
        let qnan = self.cu(0x7E00);
        let nan = self.ins(S::BitwiseOr, u, &[payload, qnan]);
        let zero = self.cu(0);
        let no_mant = self.ins(S::IEqual, tb, &[mant, zero]);
        let inf = self.cu(0x7C00);
        let special = self.select(u, no_mant, inf, nan);
        let inf32 = self.cu(0x7F80_0000);
        let is_special = self.ins(S::UGreaterThanEqual, tb, &[abs, inf32]);
        let mag = self.select(u, is_special, special, finite);
        self.ins(S::BitwiseOr, u, &[sign, mag])
    }

    /// f16 bits (low 16 bits of a u32) → f32 bits, exactly, with integer
    /// operations: f16 denormals become normal f32 values whatever the
    /// host's f16 denormal mode; NaNs keep their payload.
    pub fn f16_bits_to_f32(&mut self, h: Word) -> Word {
        let u = self.t_u32();
        let tb = self.t_bool();
        let tf = self.t_f32();
        let c8000 = self.cu(0x8000);
        let c16 = self.cu(16);
        let s = self.ins(S::BitwiseAnd, u, &[h, c8000]);
        let sign = self.ins(S::ShiftLeftLogical, u, &[s, c16]);
        let c7fff = self.cu(0x7FFF);
        let mag = self.ins(S::BitwiseAnd, u, &[h, c7fff]);
        let c13 = self.cu(13);
        // Normal: rebias the exponent (+112); inf / NaN: exponent 0xFF.
        let shifted = self.ins(S::ShiftLeftLogical, u, &[mag, c13]);
        let rebias = self.cu(112 << 23);
        let normal = self.ins(S::IAdd, u, &[shifted, rebias]);
        let c7c00 = self.cu(0x7C00);
        let special = self.ins(S::UGreaterThanEqual, tb, &[mag, c7c00]);
        let infnan = self.cu(0x7F80_0000);
        let mant = self.cu(0x3FF);
        let m = self.ins(S::BitwiseAnd, u, &[h, mant]);
        let mm = self.ins(S::ShiftLeftLogical, u, &[m, c13]);
        let special_bits = self.ins(S::BitwiseOr, u, &[infnan, mm]);
        let r = self.select(u, special, special_bits, normal);
        // Zero / denormal: mantissa * 2^-24 is exact and a normal f32.
        let c400 = self.cu(0x400);
        let small = self.ins(S::ULessThan, tb, &[mag, c400]);
        let mf = self.ins(S::ConvertUToF, tf, &[mag]);
        let scale = self.cf(f32::from_bits(0x3380_0000));
        let sv = self.ins(S::FMul, tf, &[mf, scale]);
        let sb = self.bitcast(u, sv);
        let r = self.select(u, small, sb, r);
        self.ins(S::BitwiseOr, u, &[sign, r])
    }

    /// f32 → f16 bits (low 16 bits of a u32), round to nearest even: the
    /// native conversion under the 16-bit RoundingModeRTE, DenormPreserve
    /// and SignedZeroInfNanPreserve modes where the device guarantees them
    /// ([`Features::f16_rte`](super::Features)), else the integer
    /// conversion. NaNs keep the quiet NaN with the high payload bits.
    pub fn f32_to_f16_bits_rte(&mut self, value: Word) -> Word {
        if !self.feat.f16_rte {
            return self.f32_to_f16_bits(value);
        }
        self.cap(Capability::Float16);
        self.f16_rte_used = true;
        let u = self.t_u32();
        let tb = self.t_bool();
        let tf16 = self.ty(Type::F16);
        let half = self.ins(S::FConvert, tf16, &[value]);
        let t16 = self.ty(Type::U16);
        let h = self.bitcast(t16, half);
        let native = self.ins(S::UConvert, u, &[h]);
        let bits = self.bitcast(u, value);
        let mm = self.cu(0x7FFF_FFFF);
        let mag = self.ins(S::BitwiseAnd, u, &[bits, mm]);
        let c16 = self.cu(16);
        let s = self.ins(S::ShiftRightLogical, u, &[bits, c16]);
        let c8000 = self.cu(0x8000);
        let sign = self.ins(S::BitwiseAnd, u, &[s, c8000]);
        let mant_mask = self.cu(0x7F_FFFF);
        let nm = self.ins(S::BitwiseAnd, u, &[mag, mant_mask]);
        let c13 = self.cu(13);
        let payload = self.ins(S::ShiftRightLogical, u, &[nm, c13]);
        let qnan = self.cu(0x7E00);
        let nan = self.ins(S::BitwiseOr, u, &[qnan, payload]);
        let nan = self.ins(S::BitwiseOr, u, &[sign, nan]);
        let inf32 = self.cu(0x7F80_0000);
        let is_nan = self.ins(S::UGreaterThan, tb, &[mag, inf32]);
        self.select(u, is_nan, nan, native)
    }

    fn fconvert(&mut self, from: Type, to: Type, x: Word, r: Rounding) -> Word {
        if r == Rounding::Rte && from == Type::F32 && to == Type::F16 {
            let bits = self.f32_to_f16_bits_rte(x);
            let t16 = self.ty(Type::U16);
            let h = self.ins(S::UConvert, t16, &[bits]);
            let tf16 = self.ty(Type::F16);
            return self.bitcast(tf16, h);
        }
        if from == Type::F16 && to == Type::F32 {
            let t16 = self.ty(Type::U16);
            let h = self.bitcast(t16, x);
            let u = self.t_u32();
            let h = self.ins(S::UConvert, u, &[h]);
            let bits = self.f16_bits_to_f32(h);
            let tf = self.t_f32();
            return self.bitcast(tf, bits);
        }
        if r == Rounding::Rtz && from == Type::F32 && to == Type::F16 && !self.feat.f16_rte {
            // Without guaranteed 16-bit rounding and denormals the native
            // conversion's result is the driver's: convert on the bits.
            let bits = self.f32_to_f16_bits_rtz(x);
            let t16 = self.ty(Type::U16);
            let h = self.ins(S::UConvert, t16, &[bits]);
            let tf16 = self.ty(Type::F16);
            return self.bitcast(tf16, h);
        }
        let t = self.ty(to);
        // A directed f16 conversion steps from the native nearest one:
        // nearest even where the device guarantees it (the module then
        // declares 16-bit RoundingModeRTE).
        if r == Rounding::Rtz && to.component() == Scalar::F16 && self.feat.f16_rte {
            self.f16_rte_used = true;
        }
        let n = self.ins(S::FConvert, t, &[x]);
        if r == Rounding::Rte || to.component().bits() >= from.component().bits() || to.is_vector() {
            return n;
        }
        let ft = self.ty(from);
        let back = self.ins(S::FConvert, ft, &[n]);
        let ab = self.gl(ft, GLOp::FAbs, &[back]);
        let ax = self.gl(ft, GLOp::FAbs, &[x]);
        let tb = self.t_bool();
        let away = self.ins(S::FOrdGreaterThan, tb, &[ab, ax]);
        let it = self.ty(Type::Scalar(if to.component() == Scalar::F16 { Scalar::U16 } else { Scalar::U32 }));
        let bits = self.bitcast(it, n);
        let one = if to.component() == Scalar::F16 { self.konst(Type::U16, 1) } else { self.cu(1) };
        let dec = self.ins(S::ISub, it, &[bits, one]);
        let stepped = self.bitcast(t, dec);
        self.select(t, away, stepped, n)
    }

    fn int_to_float(&mut self, from: Type, to: Type, x: Word, signed: bool, r: Rounding) -> Word {
        let t = self.ty(to);
        let op = if signed { S::ConvertSToF } else { S::ConvertUToF };
        let n = self.ins(op, t, &[x]);
        if r == Rounding::Rte || to.is_vector() || from.component().bits() != 32 || to.component() == Scalar::F64 {
            return n;
        }
        if self.feat.no_float64 {
            return self.int_to_float_toward_zero(to, x, signed);
        }
        // Compare the rounded value with the exact one in f64.
        let t64 = self.ty(Type::F64);
        let exact = self.ins(op, t64, &[x]);
        let wide = self.ins(S::FConvert, t64, &[n]);
        let ae = self.gl(t64, GLOp::FAbs, &[exact]);
        let aw = self.gl(t64, GLOp::FAbs, &[wide]);
        let tb = self.t_bool();
        let away = self.ins(S::FOrdGreaterThan, tb, &[aw, ae]);
        let it = self.ty(Type::Scalar(if to.component() == Scalar::F16 { Scalar::U16 } else { Scalar::U32 }));
        let bits = self.bitcast(it, n);
        let one = if to.component() == Scalar::F16 { self.konst(Type::U16, 1) } else { self.cu(1) };
        let dec = self.ins(S::ISub, it, &[bits, one]);
        let stepped = self.bitcast(t, dec);
        self.select(t, away, stepped, n)
    }

    /// A 32-bit integer to f32 / f16 rounded toward zero with integer code
    /// only (devices without shaderFloat64): the magnitude truncated to the
    /// destination's significand (11 or 24 bits; f16 clamped to its largest
    /// finite value) converts exactly. The same result as stepping the
    /// nearest conversion toward zero where it rounded away.
    fn int_to_float_toward_zero(&mut self, to: Type, x: Word, signed: bool) -> Word {
        let t = self.ty(to);
        let tu = self.t_u32();
        let half = to.component() == Scalar::F16;
        let significand = if half { 11 } else { 24 };
        let zero = self.cu(0);
        let (negative, magnitude) = if signed {
            let tb = self.t_bool();
            let negative = self.ins(S::SLessThan, tb, &[x, zero]);
            let negated = self.isub(zero, x);
            (Some(negative), self.select(tu, negative, negated, x))
        } else {
            (None, x)
        };
        // Bits below the significand: msb - (significand - 1) when the
        // magnitude needs more than `significand` bits.
        let msb = self.gl(tu, GLOp::FindUMsb, &[magnitude]);
        let limit = self.cu(1 << significand);
        let wide = self.ule(limit, magnitude);
        let keep = self.cu(significand - 1);
        let excess = self.isub(msb, keep);
        let shift = self.select(tu, wide, excess, zero);
        let high = self.shr(magnitude, shift);
        let mut truncated = self.shl(high, shift);
        if half {
            let max = self.cu(65504);
            truncated = self.gl(tu, GLOp::UMin, &[truncated, max]);
        }
        let f = self.ins(S::ConvertUToF, t, &[truncated]);
        match negative {
            Some(negative) => {
                let flipped = self.ins(S::FNegate, t, &[f]);
                self.select(t, negative, flipped, f)
            }
            None => f,
        }
    }

    // ------------------------------------------------------------------
    // Finish
    // ------------------------------------------------------------------

    /// Capabilities, extensions, memory model, entry point and modes.
    pub fn finish(mut self, model: spirv::ExecutionModel, modes: &[(spirv::ExecutionMode, Vec<u32>)]) -> Vec<u32> {
        self.cap(Capability::Shader);
        let has = |e: &Self, c: Capability| e.caps.contains_key(&(c as u32));
        let inf_nan_widths = [
            (16, self.feat.preserve_inf_nan[0] && has(&self, Capability::Float16)),
            (32, self.feat.preserve_inf_nan[1]),
            (64, self.feat.preserve_inf_nan[2] && has(&self, Capability::Float64)),
        ];
        if inf_nan_widths.iter().any(|w| w.1) {
            self.cap(Capability::SignedZeroInfNanPreserve);
        }
        // A module whose FLOAT_MODE flushes f32 everywhere runs on the
        // host's flush: DenormFlushToZero instead of the (skipped) explicit
        // flush code and the DenormPreserve of the other widths.
        let native32 = self.input.native_ftz && self.feat.denorm_flush_to_zero32;
        let denorm_widths = [
            (16, self.feat.denorm_preserve[0] && has(&self, Capability::Float16)),
            (32, !native32 && self.feat.denorm_preserve[1]),
            (64, self.feat.denorm_preserve[2] && has(&self, Capability::Float64)),
        ];
        if denorm_widths.iter().any(|w| w.1) {
            self.cap(Capability::DenormPreserve);
        }
        if native32 {
            self.cap(Capability::DenormFlushToZero);
        }
        if self.f16_rte_used {
            self.cap(Capability::RoundingModeRTE);
        }
        let physical = self.caps.contains_key(&(Capability::PhysicalStorageBufferAddresses as u32));
        if physical {
            self.exts.insert("SPV_KHR_physical_storage_buffer");
        }
        for c in self.caps.values().copied().collect::<Vec<_>>() {
            self.b.capability(c);
        }
        for &e in &self.exts.clone() {
            self.b.extension(e);
        }
        self.b.memory_model(
            if physical { spirv::AddressingModel::PhysicalStorageBuffer64 } else { spirv::AddressingModel::Logical },
            spirv::MemoryModel::GLSL450,
        );
        let globals = self.globals.clone();
        let main = if self.entry != 0 { self.entry } else { self.main };
        self.b.entry_point(model, main, "main", globals);
        for (m, a) in modes {
            self.b.execution_mode(main, *m, a);
        }
        // Infinities, NaNs and signed zeros are data in guest shaders: keep
        // the driver from assuming them away (fast-math folding of the
        // explicit NaN / zero selects, min/max, clamps).
        for &(bits, used) in &inf_nan_widths {
            if used {
                self.b.execution_mode(main, spirv::ExecutionMode::SignedZeroInfNanPreserve, [bits]);
            }
        }
        // Denormals follow the guest's FLOAT_MODE, which the translator
        // applies explicitly; the host must not flush the others. A module
        // whose mode flushes f32 everywhere runs on the host's flush.
        for &(bits, used) in &denorm_widths {
            if used {
                self.b.execution_mode(main, spirv::ExecutionMode::DenormPreserve, [bits]);
            }
        }
        if native32 {
            self.b.execution_mode(main, spirv::ExecutionMode::DenormFlushToZero, [32]);
        }
        // Native f32 -> f16 conversions round to nearest even.
        if self.f16_rte_used {
            self.b.execution_mode(main, spirv::ExecutionMode::RoundingModeRTE, [16]);
        }
        use rspirv::binary::Assemble;
        self.b.module().assemble()
    }
}
