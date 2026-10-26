// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! IR function: value arena, blocks with structured terminators, builder.

use super::op::*;
use super::types::*;
use crate::fxhash::HashMap;

/// SSA value handle (index into [`Function::values`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Value(pub u32);

/// Block handle (index into [`Function::blocks`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Block(pub u32);

impl Value {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl Block {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// One instruction / value.
#[derive(Clone, Debug)]
pub struct ValueData {
    pub op: Op,
    pub ty: Type,
    pub args: Vec<Value>,
    /// Containing block; `None` for constants, undefs and removed values.
    pub block: Option<Block>,
}

/// Block terminator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Terminator {
    /// Not yet terminated (only while building).
    None,
    Branch(Block),
    /// `cond` is a `bool` value.
    CondBranch { cond: Value, then_block: Block, else_block: Block },
    Return,
    Unreachable,
}

impl Terminator {
    pub fn successors(&self) -> Vec<Block> {
        match *self {
            Terminator::Branch(b) => vec![b],
            Terminator::CondBranch { then_block, else_block, .. } => {
                if then_block == else_block {
                    vec![then_block]
                } else {
                    vec![then_block, else_block]
                }
            }
            _ => vec![],
        }
    }
}

/// Structured control flow declaration of a header block (SPIR-V
/// OpSelectionMerge / OpLoopMerge).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Merge {
    None,
    Selection { merge: Block },
    Loop { merge: Block, continue_block: Block },
}

#[derive(Clone, Debug)]
pub struct BlockData {
    /// Phis first, then ordinary instructions.
    pub insts: Vec<Value>,
    pub term: Terminator,
    pub merge: Merge,
    /// Predecessors; phi arguments align with this list.
    pub preds: Vec<Block>,
}

/// Shader stage the function was translated for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Compute,
    Pixel,
    Vertex,
    Geometry,
    Hull,
    Domain,
    Unknown,
}

/// A translated shader.
#[derive(Clone, Debug)]
pub struct Function {
    pub values: Vec<ValueData>,
    pub blocks: Vec<BlockData>,
    pub entry: Block,
    /// Structured layout order: every block appears after its dominator and
    /// constructs are contiguous (the order a SPIR-V emitter writes blocks).
    pub order: Vec<Block>,
    pub wave_size: u32,
    pub stage: Stage,
    /// Guest entry address (what S_GETPC_B64 is relative to).
    pub code_address: u64,
    /// Selects whose true arm need only be computed when the condition
    /// holds: a back-end may compute the operations used by nothing but
    /// that arm under a branch (the result is the same either way).
    pub lazy_selects: crate::fxhash::HashSet<Value>,
    consts: HashMap<(Type, u64), Value>,
    undefs: HashMap<Type, Value>,
}

impl Function {
    pub fn new(wave_size: u32, stage: Stage, code_address: u64) -> Self {
        Function {
            values: Vec::new(),
            blocks: Vec::new(),
            entry: Block(0),
            order: Vec::new(),
            wave_size,
            stage,
            code_address,
            lazy_selects: Default::default(),
            consts: HashMap::default(),
            undefs: HashMap::default(),
        }
    }

    pub fn value(&self, v: Value) -> &ValueData {
        &self.values[v.index()]
    }

    pub fn value_mut(&mut self, v: Value) -> &mut ValueData {
        &mut self.values[v.index()]
    }

    pub fn block(&self, b: Block) -> &BlockData {
        &self.blocks[b.index()]
    }

    pub fn block_mut(&mut self, b: Block) -> &mut BlockData {
        &mut self.blocks[b.index()]
    }

    pub fn op(&self, v: Value) -> Op {
        self.values[v.index()].op
    }

    pub fn ty(&self, v: Value) -> Type {
        self.values[v.index()].ty
    }

    pub fn args(&self, v: Value) -> &[Value] {
        &self.values[v.index()].args
    }

    pub fn new_block(&mut self) -> Block {
        let b = Block(self.blocks.len() as u32);
        self.blocks.push(BlockData { insts: Vec::new(), term: Terminator::None, merge: Merge::None, preds: Vec::new() });
        b
    }

    /// Create a detached value.
    pub fn new_value(&mut self, op: Op, ty: Type, args: Vec<Value>) -> Value {
        let v = Value(self.values.len() as u32);
        self.values.push(ValueData { op, ty, args, block: None });
        v
    }

    /// Append a new value to `block`.
    pub fn append(&mut self, block: Block, op: Op, ty: Type, args: Vec<Value>) -> Value {
        let v = self.new_value(op, ty, args);
        self.values[v.index()].block = Some(block);
        self.blocks[block.index()].insts.push(v);
        v
    }

    /// Interned constant of a scalar type. `bits` is masked to the width.
    pub fn constant(&mut self, ty: Type, bits: u64) -> Value {
        let s = ty.scalar().expect("constant of void type");
        assert!(!ty.is_vector(), "vector constants are built with Construct");
        let bits = bits & s.mask();
        if let Some(&v) = self.consts.get(&(ty, bits)) {
            return v;
        }
        let v = self.new_value(Op::Const(bits), ty, Vec::new());
        self.consts.insert((ty, bits), v);
        v
    }

    pub fn const_u32(&mut self, x: u32) -> Value {
        self.constant(Type::U32, x as u64)
    }

    pub fn const_u64(&mut self, x: u64) -> Value {
        self.constant(Type::U64, x)
    }

    pub fn const_bool(&mut self, x: bool) -> Value {
        self.constant(Type::BOOL, x as u64)
    }

    pub fn const_f32(&mut self, x: f32) -> Value {
        self.constant(Type::F32, x.to_bits() as u64)
    }

    pub fn undef(&mut self, ty: Type) -> Value {
        if let Some(&v) = self.undefs.get(&ty) {
            return v;
        }
        let v = self.new_value(Op::Undef, ty, Vec::new());
        self.undefs.insert(ty, v);
        v
    }

    /// Constant payload of `v`, if it is a constant.
    pub fn as_const(&self, v: Value) -> Option<u64> {
        match self.values[v.index()].op {
            Op::Const(c) => Some(c),
            _ => None,
        }
    }

    pub fn is_undef(&self, v: Value) -> bool {
        self.values[v.index()].op == Op::Undef
    }

    /// Recompute every block's predecessor list from the terminators,
    /// keeping existing order where possible (phi arguments must be remapped
    /// by the caller if the order changes).
    pub fn compute_preds(&self) -> Vec<Vec<Block>> {
        let mut preds = vec![Vec::new(); self.blocks.len()];
        for &b in &self.order {
            for s in self.blocks[b.index()].term.successors() {
                if !preds[s.index()].contains(&b) {
                    preds[s.index()].push(b);
                }
            }
        }
        preds
    }

    /// Rebuild `preds` from the terminators (for functions without phis yet).
    pub fn rebuild_preds(&mut self) {
        let preds = self.compute_preds();
        for (b, p) in preds.into_iter().enumerate() {
            self.blocks[b].preds = p;
        }
    }

    /// Replace the values listed in `map` (value -> replacement) in every
    /// operand and terminator condition. Chains are followed.
    pub fn replace_uses(&mut self, map: &HashMap<Value, Value>) {
        if map.is_empty() {
            return;
        }
        let resolve = |mut v: Value| {
            let mut guard = 0;
            while let Some(&n) = map.get(&v) {
                if n == v || guard > 1_000_000 {
                    break;
                }
                v = n;
                guard += 1;
            }
            v
        };
        for b in 0..self.blocks.len() {
            for i in 0..self.blocks[b].insts.len() {
                let v = self.blocks[b].insts[i];
                let data = &mut self.values[v.index()];
                for a in data.args.iter_mut() {
                    *a = resolve(*a);
                }
            }
            if let Terminator::CondBranch { cond, .. } = &mut self.blocks[b].term {
                *cond = resolve(*cond);
            }
        }
    }

    /// Remove instructions for which `dead` is true from their blocks.
    pub fn remove_values(&mut self, dead: &[bool]) {
        for b in 0..self.blocks.len() {
            let insts = std::mem::take(&mut self.blocks[b].insts);
            let mut kept = Vec::with_capacity(insts.len());
            for v in insts {
                if dead.get(v.index()).copied().unwrap_or(false) {
                    self.values[v.index()].block = None;
                } else {
                    kept.push(v);
                }
            }
            self.blocks[b].insts = kept;
        }
    }

    /// Number of instructions placed in blocks.
    pub fn instruction_count(&self) -> usize {
        self.order.iter().map(|b| self.blocks[b.index()].insts.len()).sum()
    }

    /// Use counts of every value (operands and branch conditions of blocks in
    /// `order`).
    pub fn use_counts(&self) -> Vec<u32> {
        let mut uses = vec![0u32; self.values.len()];
        for &b in &self.order {
            let bd = &self.blocks[b.index()];
            for &v in &bd.insts {
                for &a in &self.values[v.index()].args {
                    uses[a.index()] += 1;
                }
            }
            if let Terminator::CondBranch { cond, .. } = bd.term {
                uses[cond.index()] += 1;
            }
        }
        uses
    }
}

/// Appends instructions at the end of a block.
pub struct Builder<'f> {
    pub func: &'f mut Function,
    pub block: Block,
}

macro_rules! binop {
    ($($name:ident => $op:ident),* $(,)?) => {
        $(pub fn $name(&mut self, a: Value, b: Value) -> Value {
            let ty = self.func.ty(a);
            self.emit(Op::$op, ty, vec![a, b])
        })*
    };
}

macro_rules! unop {
    ($($name:ident => $op:ident),* $(,)?) => {
        $(pub fn $name(&mut self, a: Value) -> Value {
            let ty = self.func.ty(a);
            self.emit(Op::$op, ty, vec![a])
        })*
    };
}

impl<'f> Builder<'f> {
    pub fn new(func: &'f mut Function, block: Block) -> Self {
        Builder { func, block }
    }

    pub fn emit(&mut self, op: Op, ty: Type, args: Vec<Value>) -> Value {
        self.func.append(self.block, op, ty, args)
    }

    pub fn u32(&mut self, x: u32) -> Value {
        self.func.const_u32(x)
    }

    pub fn u64(&mut self, x: u64) -> Value {
        self.func.const_u64(x)
    }

    pub fn bool(&mut self, x: bool) -> Value {
        self.func.const_bool(x)
    }

    pub fn f32(&mut self, x: f32) -> Value {
        self.func.const_f32(x)
    }

    pub fn f64(&mut self, x: f64) -> Value {
        self.func.constant(Type::F64, x.to_bits())
    }

    pub fn f16_bits(&mut self, bits: u16) -> Value {
        self.func.constant(Type::F16, bits as u64)
    }

    pub fn constant(&mut self, ty: Type, bits: u64) -> Value {
        self.func.constant(ty, bits)
    }

    pub fn ty(&self, v: Value) -> Type {
        self.func.ty(v)
    }

    pub fn get_reg(&mut self, r: Reg) -> Value {
        self.emit(Op::GetReg(r), r.ty(), vec![])
    }

    pub fn set_reg(&mut self, r: Reg, v: Value) {
        debug_assert_eq!(self.func.ty(v), r.ty(), "set {r}");
        self.emit(Op::SetReg(r), Type::Void, vec![v]);
    }

    pub fn select(&mut self, c: Value, a: Value, b: Value) -> Value {
        let ty = self.func.ty(a);
        debug_assert_eq!(ty, self.func.ty(b));
        debug_assert!(self.func.ty(c).is_bool());
        self.emit(Op::Select, ty, vec![c, a, b])
    }

    pub fn bitcast(&mut self, x: Value, ty: Type) -> Value {
        if self.func.ty(x) == ty {
            return x;
        }
        self.emit(Op::Bitcast, ty, vec![x])
    }

    pub fn construct(&mut self, parts: &[Value]) -> Value {
        if parts.len() == 1 {
            return parts[0];
        }
        let s = self.func.ty(parts[0]).component();
        self.emit(Op::Construct, Type::vec(s, parts.len() as u32), parts.to_vec())
    }

    pub fn extract(&mut self, v: Value, i: u32) -> Value {
        let ty = self.func.ty(v);
        if !ty.is_vector() {
            assert_eq!(i, 0);
            return v;
        }
        self.emit(Op::Extract(i as u8), Type::Scalar(ty.component()), vec![v])
    }

    binop! {
        iadd => IAdd, isub => ISub, imul => IMul, umul_hi => UMulHi, smul_hi => SMulHi,
        udiv => UDiv, umod => UMod, smin => SMin, smax => SMax, umin => UMin, umax => UMax,
        shl => Shl, lshr => LShr, ashr => AShr, and => And, or => Or, xor => Xor,
        fadd => FAdd, fsub => FSub, fmul => FMul, fdiv => FDiv, fmin => FMin, fmax => FMax,
        ldexp => FLdexp,
    }

    unop! {
        ineg => INeg, not => Not, bit_reverse => BitReverse,
        fneg => FNeg, fabs => FAbs, floor => FFloor, ceil => FCeil, trunc => FTrunc,
        round_even => FRoundEven, sqrt => FSqrt, inv_sqrt => FInvSqrt, rcp => FRcp,
        exp2 => FExp2, log2 => FLog2, sin => FSin, cos => FCos,
    }

    pub fn fma(&mut self, a: Value, b: Value, c: Value) -> Value {
        let ty = self.func.ty(a);
        self.emit(Op::FFma, ty, vec![a, b, c])
    }

    pub fn bit_count(&mut self, a: Value) -> Value {
        self.emit(Op::BitCount, Type::U32, vec![a])
    }

    pub fn find_lsb(&mut self, a: Value) -> Value {
        self.emit(Op::FindLsb, Type::U32, vec![a])
    }

    pub fn find_umsb(&mut self, a: Value) -> Value {
        self.emit(Op::FindUMsb, Type::U32, vec![a])
    }

    pub fn find_smsb(&mut self, a: Value) -> Value {
        self.emit(Op::FindSMsb, Type::U32, vec![a])
    }

    pub fn bfe_u(&mut self, base: Value, offset: Value, count: Value) -> Value {
        let ty = self.func.ty(base);
        self.emit(Op::BitFieldUExtract, ty, vec![base, offset, count])
    }

    pub fn bfe_s(&mut self, base: Value, offset: Value, count: Value) -> Value {
        let ty = self.func.ty(base);
        self.emit(Op::BitFieldSExtract, ty, vec![base, offset, count])
    }

    pub fn bfi(&mut self, base: Value, insert: Value, offset: Value, count: Value) -> Value {
        let ty = self.func.ty(base);
        self.emit(Op::BitFieldInsert, ty, vec![base, insert, offset, count])
    }

    pub fn uconvert(&mut self, x: Value, ty: Type) -> Value {
        if self.func.ty(x) == ty {
            return x;
        }
        self.emit(Op::UConvert, ty, vec![x])
    }

    pub fn sconvert(&mut self, x: Value, ty: Type) -> Value {
        if self.func.ty(x) == ty {
            return x;
        }
        self.emit(Op::SConvert, ty, vec![x])
    }

    pub fn icmp(&mut self, p: ICmp, a: Value, b: Value) -> Value {
        self.emit(Op::ICmp(p), Type::BOOL, vec![a, b])
    }

    pub fn fcmp(&mut self, p: FCmp, a: Value, b: Value) -> Value {
        self.emit(Op::FCmp(p), Type::BOOL, vec![a, b])
    }

    pub fn fconvert(&mut self, x: Value, ty: Type, r: Rounding) -> Value {
        if self.func.ty(x) == ty {
            return x;
        }
        self.emit(Op::FConvert(r), ty, vec![x])
    }

    pub fn ftos(&mut self, x: Value, ty: Type) -> Value {
        self.emit(Op::FToS, ty, vec![x])
    }

    pub fn ftou(&mut self, x: Value, ty: Type) -> Value {
        self.emit(Op::FToU, ty, vec![x])
    }

    pub fn stof(&mut self, x: Value, ty: Type) -> Value {
        self.emit(Op::SToF(Rounding::Rte), ty, vec![x])
    }

    pub fn utof(&mut self, x: Value, ty: Type) -> Value {
        self.emit(Op::UToF(Rounding::Rte), ty, vec![x])
    }

    pub fn lnot(&mut self, a: Value) -> Value {
        self.emit(Op::Not, Type::BOOL, vec![a])
    }

    pub fn land(&mut self, a: Value, b: Value) -> Value {
        self.emit(Op::And, Type::BOOL, vec![a, b])
    }

    pub fn lor(&mut self, a: Value, b: Value) -> Value {
        self.emit(Op::Or, Type::BOOL, vec![a, b])
    }

    pub fn lxor(&mut self, a: Value, b: Value) -> Value {
        self.emit(Op::Xor, Type::BOOL, vec![a, b])
    }

    pub fn lane_id(&mut self) -> Value {
        self.emit(Op::LaneId, Type::U32, vec![])
    }

    /// `(lo, hi)` words of the ballot of `b`.
    pub fn ballot(&mut self, b: Value) -> (Value, Value) {
        let m = self.emit(Op::Ballot, Type::U32X2, vec![b]);
        let lo = self.extract(m, 0);
        let hi = self.extract(m, 1);
        (lo, hi)
    }

    pub fn inverse_ballot(&mut self, lo: Value, hi: Value) -> Value {
        self.emit(Op::InverseBallot, Type::BOOL, vec![lo, hi])
    }

    pub fn read_lane(&mut self, v: Value, lane: Value) -> Value {
        let ty = self.func.ty(v);
        self.emit(Op::ReadLane, ty, vec![v, lane])
    }

    pub fn shuffle(&mut self, v: Value, lane: Value) -> Value {
        let ty = self.func.ty(v);
        self.emit(Op::Shuffle, ty, vec![v, lane])
    }

    /// 0/1 as u32 from a bool.
    pub fn bool_to_u32(&mut self, b: Value) -> Value {
        let one = self.u32(1);
        let zero = self.u32(0);
        self.select(b, one, zero)
    }

    /// Low / high dwords of a u64.
    pub fn split_u64(&mut self, v: Value) -> (Value, Value) {
        let pair = self.bitcast(v, Type::U32X2);
        let lo = self.extract(pair, 0);
        let hi = self.extract(pair, 1);
        (lo, hi)
    }

    pub fn join_u64(&mut self, lo: Value, hi: Value) -> Value {
        let pair = self.construct(&[lo, hi]);
        self.bitcast(pair, Type::U64)
    }
}
