// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The bytes a compute dispatch can store to through each buffer.
//!
//! A written buffer is GPU-dirty over its whole descriptor range: the CPU
//! then faults on, and downloads, every page of it, even when the dispatch
//! stores to a few of them. This analysis bounds the byte offsets of a
//! program's buffer stores and atomics from the thread and group ids of a
//! dispatch: an abstract interpretation of the decoded program over
//! integer expressions of the inputs (local invocation ids, workgroup ids,
//! user-data SGPRs), evaluated as intervals per dispatch.
//!
//! It must never under-estimate (a page the GPU writes but the tracker
//! does not know about would serve stale bytes to the CPU), so it gives up
//! wherever it is not sure: loops (backward branches), indirect jumps,
//! register-indexed moves, values it does not model (memory loads,
//! floating point, lane operations), and vector registers written under an
//! EXEC mask that is not known to contain the lanes of the store. A buffer
//! with any store it cannot bound keeps its whole range.
//!
//! EXEC is tracked as a tree of lane sets: narrowing it (`s_and_saveexec`,
//! `s_and exec, exec, x`, `v_cmpx`) makes a child of the current set, and
//! restoring a saved mask returns to that ancestor. A vector register
//! written under set `c` holds its new value in the lanes of `c` (and of
//! every set below it); a store under any other set sees an unknown value.

use ps5shader::opcode::Opcode;
use ps5shader::{Control, Encoding, Instruction, OperandKind, Program};

/// An expression of the dispatch inputs (index into [`StoreBounds::exprs`]).
type E = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Expr {
    Unknown,
    Const(u32),
    /// `[lo, hi]`.
    Range(u32, u32),
    /// The user-data SGPR (absolute index).
    UserData(u32),
    WorkgroupId(u8),
    LocalId(u8),
    Add(E, E),
    Sub(E, E),
    Mul(E, E),
    /// Both operands' low 24 bits multiplied.
    Mul24(E, E),
    Shl(E, E),
    Shr(E, E),
    And(E, E),
    Or(E, E),
    Min(E, E),
    Max(E, E),
    /// Either value (control-flow joins, lane selects).
    Join(E, E),
    /// `(a >> offset) & ((1 << width) - 1)`, offset and width 5 bits each.
    Bfe(E, E, E),
}

const UNKNOWN: E = 0;

/// A store or atomic through a buffer: its byte offset from the V# base is
/// `soffset + imm + (offen ? offset : 0) + (idxen ? index * stride : 0)`.
#[derive(Clone, Copy, Debug)]
struct Store {
    pc: u32,
    index: Option<E>,
    offset: Option<E>,
    soffset: E,
    imm: u32,
}

/// Bytes past a store's offset it may write (the widest access, 16 bytes,
/// for every store: page granularity makes the excess irrelevant).
const STORE_BYTES: u64 = 16;

/// A program's buffer stores, analysed once per program.
#[derive(Debug)]
pub struct StoreBounds {
    exprs: Vec<Expr>,
    stores: Vec<Store>,
    /// Pcs of buffer stores the analysis could not follow (unreachable
    /// ones are not listed anywhere: they never run).
    unbounded: Vec<u32>,
}

/// The inputs of one dispatch.
#[derive(Clone, Copy, Debug)]
pub struct DispatchInputs<'a> {
    /// User-data SGPR values from `user_base` on.
    pub user_data: &'a [u32],
    pub user_base: u32,
    /// Workgroups per axis (direct dispatches only).
    pub groups: [u32; 3],
    /// Threads per group per axis.
    pub local: [u32; 3],
}

/// Where the hardware puts the compute system SGPRs (see
/// `ps5shader::ComputeSystemRegisters`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SystemSgprs {
    pub work_group: [Option<u32>; 3],
    pub thread_group_size: Option<u32>,
}

/// A scalar register: a value, or a half of a saved EXEC mask (the lane
/// set id; a wave64 mask restores only with both halves intact).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sv {
    Val(E),
    Exec(u32),
    ExecHi(u32),
}

/// A vector register: its value in the lanes of an EXEC lane set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Vv {
    value: E,
    lanes: u32,
}

const SGPRS: usize = 128;
const VGPRS: usize = 256;

#[derive(Clone)]
struct State {
    s: Vec<Sv>,
    v: Vec<Vv>,
    exec: u32,
}

struct Analyzer {
    exprs: Vec<Expr>,
    interned: crate::fxhash::HashMap<Expr, E>,
    /// Parent lane set of each EXEC lane set (None: unrelated root).
    parents: Vec<Option<u32>>,
    wave64: bool,
}

impl Analyzer {
    fn expr(&mut self, e: Expr) -> E {
        if e == Expr::Unknown {
            return UNKNOWN;
        }
        // Equal expressions stay equal (joins of identical values exact).
        // Operands always precede their users.
        if let Some(&i) = self.interned.get(&e) {
            return i;
        }
        self.exprs.push(e);
        let i = (self.exprs.len() - 1) as E;
        self.interned.insert(e, i);
        i
    }

    fn op2(&mut self, f: fn(E, E) -> Expr, a: E, b: E) -> E {
        if a == UNKNOWN || b == UNKNOWN {
            return UNKNOWN;
        }
        self.expr(f(a, b))
    }

    fn fresh_lanes(&mut self, parent: Option<u32>) -> u32 {
        self.parents.push(parent);
        (self.parents.len() - 1) as u32
    }

    /// `set` is `of` or lies below it.
    fn within(&self, mut set: u32, of: u32) -> bool {
        loop {
            if set == of {
                return true;
            }
            match self.parents[set as usize] {
                Some(p) => set = p,
                None => return false,
            }
        }
    }

    fn join(&mut self, a: &State, b: &State) -> State {
        let exec = if a.exec == b.exec { a.exec } else { self.fresh_lanes(None) };
        let s = a
            .s
            .iter()
            .zip(&b.s)
            .map(|(&x, &y)| match (x, y) {
                _ if x == y => x,
                (Sv::Val(p), Sv::Val(q)) => Sv::Val(self.op2(Expr::Join, p, q)),
                _ => Sv::Val(UNKNOWN),
            })
            .collect();
        let v = a
            .v
            .iter()
            .zip(&b.v)
            .map(|(&x, &y)| match (x, y) {
                _ if x == y => x,
                _ if x.lanes == y.lanes => Vv { value: self.op2(Expr::Join, x.value, y.value), lanes: x.lanes },
                _ => Vv { value: UNKNOWN, lanes: 0 },
            })
            .collect();
        State { s, v, exec }
    }

    /// A scalar source operand.
    fn scalar_src(&mut self, st: &State, op: &ps5shader::Operand) -> E {
        match op.kind {
            OperandKind::ScalarRegister => match st.s.get(op.value as usize) {
                Some(Sv::Val(e)) if op.value != 126 && op.value != 127 => *e,
                _ => UNKNOWN,
            },
            OperandKind::LiteralConstant => self.expr(Expr::Const(op.value)),
            OperandKind::EncodedConstant => match op.value {
                128..=192 => self.expr(Expr::Const(op.value - 128)),
                193..=208 => self.expr(Expr::Const((-((op.value - 192) as i32)) as u32)),
                125 => self.expr(Expr::Const(0)),
                _ => UNKNOWN,
            },
            OperandKind::VectorRegister => UNKNOWN,
        }
    }

    /// A vector-instruction source: VGPRs only in the lanes of the current
    /// EXEC set.
    fn vector_src(&mut self, st: &State, op: &ps5shader::Operand) -> E {
        match op.kind {
            OperandKind::VectorRegister => match st.v.get(op.value as usize) {
                Some(v) if self.within(st.exec, v.lanes) => v.value,
                _ => UNKNOWN,
            },
            _ => self.scalar_src(st, op),
        }
    }
}

fn simm16(word: u32) -> u32 {
    (word & 0xFFFF) as u16 as i16 as i32 as u32
}

/// Integer VALU modifiers that would change the value (none allowed).
fn plain_vop3(i: &Instruction) -> bool {
    match &i.control {
        None => true,
        Some(Control::Vop3(c)) => c.absolute_mask == 0 && c.negate_mask == 0 && c.output_modifier == 0 && !c.clamp && c.operand_select == 0,
        _ => false,
    }
}

const EXEC_LO: u32 = 126;
const VCC_LO: u32 = 106;

impl StoreBounds {
    /// Analyse `program` (a compute program with user data from SGPR
    /// `user_base`, `user_count` of them, and system SGPRs at `system`).
    /// `writes`: the pcs of its buffer stores and atomics. None when the
    /// program cannot be analysed at all (loops, indirect jumps).
    pub fn analyze(program: &Program, user_base: u32, user_count: u32, system: SystemSgprs, wave64: bool, writes: &[u32]) -> Option<StoreBounds> {
        let mut a = Analyzer { exprs: vec![Expr::Unknown], interned: Default::default(), parents: vec![None], wave64 };
        let mut entry = State { s: vec![Sv::Val(UNKNOWN); SGPRS], v: vec![Vv { value: UNKNOWN, lanes: 0 }; VGPRS], exec: 0 };
        for r in user_base..(user_base + user_count).min(106) {
            entry.s[r as usize] = Sv::Val(a.expr(Expr::UserData(r)));
        }
        for (axis, reg) in system.work_group.iter().enumerate() {
            if let Some(r) = reg.filter(|&r| r < 106) {
                entry.s[r as usize] = Sv::Val(a.expr(Expr::WorkgroupId(axis as u8)));
            }
        }
        if let Some(r) = system.thread_group_size.filter(|&r| r < 106) {
            entry.s[r as usize] = Sv::Val(UNKNOWN);
        }
        for c in 0..3u8 {
            entry.v[c as usize] = Vv { value: a.expr(Expr::LocalId(c)), lanes: 0 };
        }
        // States arriving at forward branch targets.
        let mut pending: std::collections::BTreeMap<u32, State> = std::collections::BTreeMap::new();
        let mut cur: Option<State> = Some(entry);
        let mut stores = Vec::new();
        let mut unbounded = Vec::new();
        for i in &program.instructions {
            if i.address_offset.is_some() {
                return None;
            }
            // A branch into the middle of an instruction: not code this
            // analysis can follow.
            if pending.range(..i.pc).next().is_some() {
                return None;
            }
            if let Some(p) = pending.remove(&i.pc) {
                cur = Some(match cur.take() {
                    Some(c) => a.join(&c, &p),
                    None => p,
                });
            }
            let Some(st) = cur.as_mut() else { continue };
            if i.encoding == Encoding::Sopp {
                match i.opcode {
                    Opcode::SEndpgm | Opcode::SEndpgmSaved | Opcode::SEndpgmOrderedPsDone => cur = None,
                    Opcode::SBranch | Opcode::SCbranchScc0 | Opcode::SCbranchScc1 | Opcode::SCbranchVccz | Opcode::SCbranchVccnz | Opcode::SCbranchExecz | Opcode::SCbranchExecnz => {
                        let target = (i.pc as i64 + 4 + simm16(i.words[0]) as i32 as i64 * 4) as i64;
                        if target <= i.pc as i64 {
                            return None;
                        }
                        let target = target as u32;
                        let taken = st.clone();
                        match pending.remove(&target) {
                            Some(p) => {
                                let j = a.join(&p, &taken);
                                pending.insert(target, j);
                            }
                            None => {
                                pending.insert(target, taken);
                            }
                        }
                        if i.opcode == Opcode::SBranch {
                            cur = None;
                        }
                    }
                    Opcode::SCbranchCdbgsys | Opcode::SCbranchCdbguser | Opcode::SCbranchCdbgsysOrUser | Opcode::SCbranchCdbgsysAndUser | Opcode::SSubvectorLoopBegin | Opcode::SSubvectorLoopEnd => return None,
                    _ => {}
                }
                continue;
            }
            if matches!(i.opcode, Opcode::SSetpcB64 | Opcode::SSwappcB64 | Opcode::SCallB64 | Opcode::SRfeB64)
                || matches!(i.opcode, Opcode::SMovrelsB32 | Opcode::SMovrelsB64 | Opcode::SMovreldB32 | Opcode::SMovreldB64 | Opcode::SMovrelsd2B32)
                || matches!(i.opcode, Opcode::VMovreldB32 | Opcode::VMovrelsB32 | Opcode::VMovrelsdB32 | Opcode::VMovrelsd2B32 | Opcode::VSwaprelB32)
            {
                return None;
            }
            if writes.contains(&i.pc) {
                match a.store(st, i) {
                    Some(s) => stores.push(s),
                    None => unbounded.push(i.pc),
                }
            }
            a.step(st, i);
        }
        // Branches past the last instruction lead to no code (the program
        // ends there); one into the last instruction is not followed.
        let end = program.instructions.last().map_or(0, |i| i.next_pc());
        if pending.range(..end).next().is_some() {
            return None;
        }
        Some(StoreBounds { exprs: a.exprs, stores, unbounded })
    }

    /// The bytes `[lo, end)` from the V# base the stores at `pcs` can
    /// write, for V# `stride`. None when one of them is not bounded (or
    /// was not reached by the analysis).
    pub fn extent(&self, pcs: &[u32], stride: u32, inputs: &DispatchInputs) -> Option<(u64, u64)> {
        let mut lo = u64::MAX;
        let mut end = 0u64;
        let mut memo = Vec::new();
        for &pc in pcs {
            if self.unbounded.contains(&pc) {
                return None;
            }
            let mut any = false;
            for s in self.stores.iter().filter(|s| s.pc == pc) {
                any = true;
                let (l, h) = self.store_range(s, stride, &mut memo, inputs)?;
                lo = lo.min(l);
                end = end.max(h + STORE_BYTES);
            }
            if !any {
                return None;
            }
        }
        (lo < end).then_some((lo, end))
    }

    fn store_range(&self, s: &Store, stride: u32, memo: &mut Vec<Option<Option<(u64, u64)>>>, inputs: &DispatchInputs) -> Option<(u64, u64)> {
        let mut r = self.eval(s.soffset, memo, inputs)?;
        r = add(r, (s.imm as u64, s.imm as u64))?;
        if let Some(o) = s.offset {
            r = add(r, self.eval(o, memo, inputs)?)?;
        }
        if let Some(ix) = s.index {
            let (a, b) = self.eval(ix, memo, inputs)?;
            r = add(r, (a * stride as u64, b * stride as u64))?;
        }
        Some(r)
    }

    /// The interval of expression `e` (None: unknown, or it may wrap).
    /// Each node is evaluated once per dispatch (`memo`), without
    /// recursion.
    fn eval(&self, e: E, memo: &mut Vec<Option<Option<(u64, u64)>>>, inp: &DispatchInputs) -> Option<(u64, u64)> {
        if memo.len() < self.exprs.len() {
            memo.resize(self.exprs.len(), None);
        }
        if let Some(r) = memo[e as usize] {
            return r;
        }
        // Nodes whose operands are not evaluated yet stay on the stack.
        let mut stack = vec![e];
        while let Some(&x) = stack.last() {
            if memo[x as usize].is_some() {
                stack.pop();
                continue;
            }
            let missing: smallvec::SmallVec<[E; 3]> = operands(&self.exprs[x as usize]).into_iter().filter(|&o| memo[o as usize].is_none()).collect();
            if missing.is_empty() {
                stack.pop();
                memo[x as usize] = Some(self.eval_node(x, memo, inp));
            } else {
                stack.extend(missing);
            }
        }
        memo[e as usize].flatten()
    }

    /// One node, its operands evaluated.
    fn eval_node(&self, e: E, memo: &[Option<Option<(u64, u64)>>], inp: &DispatchInputs) -> Option<(u64, u64)> {
        let v = |x: E| memo[x as usize].flatten();
        let r = match self.exprs[e as usize] {
            Expr::Unknown => return None,
            Expr::Const(c) => (c as u64, c as u64),
            Expr::Range(a, b) => (a as u64, b as u64),
            Expr::UserData(r) => {
                let x = *inp.user_data.get(r.checked_sub(inp.user_base)? as usize)? as u64;
                (x, x)
            }
            Expr::WorkgroupId(c) => (0, inp.groups[c as usize].max(1) as u64 - 1),
            Expr::LocalId(c) => (0, inp.local[c as usize].max(1) as u64 - 1),
            Expr::Add(a, b) => add(v(a)?, v(b)?)?,
            Expr::Sub(a, b) => {
                let (a, b) = (v(a)?, v(b)?);
                if a.0 < b.1 {
                    return None;
                }
                (a.0 - b.1, a.1 - b.0)
            }
            Expr::Mul(a, b) => mul(v(a)?, v(b)?)?,
            Expr::Mul24(a, b) => {
                let (a, b) = (v(a)?, v(b)?);
                if a.1 >= 1 << 24 || b.1 >= 1 << 24 {
                    return None;
                }
                mul(a, b)?
            }
            Expr::Shl(a, b) => {
                let (a, b) = (v(a)?, v(b)?);
                if b.1 >= 32 {
                    return None;
                }
                fits((a.0 << b.0, a.1 << b.1))?
            }
            Expr::Shr(a, b) => {
                let (a, b) = (v(a)?, v(b)?);
                if b.1 >= 32 {
                    return None;
                }
                (a.0 >> b.1, a.1 >> b.0)
            }
            Expr::And(a, b) => {
                let (a, b) = (v(a)?, v(b)?);
                (0, a.1.min(b.1))
            }
            Expr::Or(a, b) => {
                let (a, b) = (v(a)?, v(b)?);
                let top = (a.1.max(b.1) + 1).next_power_of_two() - 1;
                (a.0.max(b.0), top.min(u32::MAX as u64))
            }
            Expr::Min(a, b) => {
                let (a, b) = (v(a)?, v(b)?);
                (a.0.min(b.0), a.1.min(b.1))
            }
            Expr::Max(a, b) => {
                let (a, b) = (v(a)?, v(b)?);
                (a.0.max(b.0), a.1.max(b.1))
            }
            Expr::Join(a, b) => {
                let (a, b) = (v(a)?, v(b)?);
                (a.0.min(b.0), a.1.max(b.1))
            }
            Expr::Bfe(a, o, w) => {
                let (a, o, w) = (v(a)?, v(o)?, v(w)?);
                if o.0 != o.1 || w.0 != w.1 {
                    return None;
                }
                let (o, w) = (o.0 & 31, w.0 & 31);
                let mask = (1u64 << w) - 1;
                if a.1 >> o <= mask {
                    (a.0 >> o, a.1 >> o)
                } else {
                    (0, mask)
                }
            }
        };
        fits(r)
    }
}

/// The operands of an expression node.
fn operands(e: &Expr) -> smallvec::SmallVec<[E; 3]> {
    match *e {
        Expr::Unknown | Expr::Const(_) | Expr::Range(..) | Expr::UserData(_) | Expr::WorkgroupId(_) | Expr::LocalId(_) => smallvec::SmallVec::new(),
        Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Mul24(a, b)
        | Expr::Shl(a, b)
        | Expr::Shr(a, b)
        | Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Min(a, b)
        | Expr::Max(a, b)
        | Expr::Join(a, b) => smallvec::smallvec![a, b],
        Expr::Bfe(a, o, w) => smallvec::smallvec![a, o, w],
    }
}

fn fits(r: (u64, u64)) -> Option<(u64, u64)> {
    (r.0 <= r.1 && r.1 <= u32::MAX as u64).then_some(r)
}

fn add(a: (u64, u64), b: (u64, u64)) -> Option<(u64, u64)> {
    fits((a.0 + b.0, a.1 + b.1))
}

fn mul(a: (u64, u64), b: (u64, u64)) -> Option<(u64, u64)> {
    fits((a.0 * b.0, a.1.checked_mul(b.1)?))
}

impl Analyzer {
    /// The offset expressions of a buffer store at `i` (None: not a
    /// buffer store this analysis understands).
    fn store(&mut self, st: &State, i: &Instruction) -> Option<Store> {
        if !matches!(i.encoding, Encoding::Mubuf | Encoding::Mtbuf) {
            return None;
        }
        let Some(Control::BufferMemory(c)) = &i.control else { return None };
        let vreg = |a: &mut Analyzer, r: u32| -> E {
            match st.v.get(r as usize) {
                Some(v) if a.within(st.exec, v.lanes) => v.value,
                _ => UNKNOWN,
            }
        };
        let mut next = c.vector_address;
        let index = c.index_enabled.then(|| {
            let e = vreg(self, next);
            next += 1;
            e
        });
        let offset = c.offset_enabled.then(|| vreg(self, next));
        let soffset_field = *i.words.get(1)? >> 24;
        let soffset = match soffset_field {
            0..=105 | 124 => match st.s[soffset_field as usize] {
                Sv::Val(e) => e,
                Sv::Exec(_) | Sv::ExecHi(_) => UNKNOWN,
            },
            125 => self.expr(Expr::Const(0)),
            128..=192 => self.expr(Expr::Const(soffset_field - 128)),
            _ => UNKNOWN,
        };
        if index == Some(UNKNOWN) || offset == Some(UNKNOWN) || soffset == UNKNOWN || c.offset_bytes < 0 {
            return None;
        }
        Some(Store { pc: i.pc, index, offset, soffset, imm: c.offset_bytes as u32 })
    }

    /// Apply instruction `i` to the state.
    fn step(&mut self, st: &mut State, i: &Instruction) {
        let op = i.opcode;
        let dst = i.destinations.first().map(|d| d.value);
        // EXEC changes.
        if op.is_saveexec() {
            let d = dst.unwrap_or(u32::MAX);
            let saved = st.exec;
            st.exec = if matches!(op, Opcode::SAndSaveexecB64 | Opcode::SAndSaveexecB32) { self.fresh_lanes(Some(saved)) } else { self.fresh_lanes(None) };
            self.kill_scalar(st, d, 2);
            // A wave64 mask saved by a 32-bit op is half a mask.
            let whole = op.is_saveexec_b64() || !self.wave64;
            if whole && (d as usize) + 1 < SGPRS {
                st.s[d as usize] = Sv::Exec(saved);
                st.s[d as usize + 1] = Sv::ExecHi(saved);
            }
            return;
        }
        // Other writes of EXEC through a scalar destination
        // (s_andn2_wrexec ...): an unrelated lane set.
        if op.name().contains("exec") {
            st.exec = self.fresh_lanes(None);
            self.kill_scalar(st, dst.unwrap_or(u32::MAX), 2);
            return;
        }
        if op.is_vector_compare_exec() {
            st.exec = self.fresh_lanes(Some(st.exec));
            return;
        }
        if dst == Some(EXEC_LO) && matches!(i.encoding, Encoding::Sop1 | Encoding::Sop2) {
            let src = |k: usize| i.sources.get(k).copied();
            let is_exec = |o: Option<ps5shader::Operand>| o.is_some_and(|o| o.kind == OperandKind::ScalarRegister && o.value == EXEC_LO);
            let wave64 = self.wave64;
            // A saved mask in a register pair (wave64: both halves).
            let saved = |o: Option<ps5shader::Operand>| match o {
                Some(o) if o.kind == OperandKind::ScalarRegister => match (st.s.get(o.value as usize), st.s.get(o.value as usize + 1)) {
                    (Some(Sv::Exec(id)), Some(Sv::ExecHi(hi))) if id == hi => Some(*id),
                    (Some(Sv::Exec(id)), _) if !wave64 => Some(*id),
                    _ => None,
                },
                _ => None,
            };
            let width_ok = matches!(op, Opcode::SAndB64 | Opcode::SAndn2B64 | Opcode::SOrB64 | Opcode::SMovB64) || !self.wave64;
            st.exec = match op {
                _ if !width_ok => self.fresh_lanes(None),
                Opcode::SAndB64 | Opcode::SAndB32 if is_exec(src(0)) || is_exec(src(1)) => self.fresh_lanes(Some(st.exec)),
                Opcode::SAndn2B64 | Opcode::SAndn2B32 if is_exec(src(0)) => self.fresh_lanes(Some(st.exec)),
                Opcode::SOrB64 | Opcode::SOrB32 => {
                    let other = if is_exec(src(0)) { saved(src(1)) } else if is_exec(src(1)) { saved(src(0)) } else { None };
                    match other {
                        Some(p) if self.within(st.exec, p) => p,
                        _ => self.fresh_lanes(None),
                    }
                }
                Opcode::SMovB64 | Opcode::SMovB32 => match saved(src(0)) {
                    Some(p) => p,
                    None => self.fresh_lanes(None),
                },
                _ => self.fresh_lanes(None),
            };
            return;
        }
        // Any other write of EXEC (or of its high half).
        if i.destinations.iter().any(|d| d.kind == OperandKind::ScalarRegister && (d.value == EXEC_LO || d.value == EXEC_LO + 1 || (d.value == EXEC_LO - 1 && op.is_wide_scalar()))) {
            st.exec = self.fresh_lanes(None);
            self.kill_scalar(st, dst.unwrap_or(u32::MAX), 2);
            return;
        }
        // A saved mask copied to another pair.
        if matches!(op, Opcode::SMovB64) && i.encoding == Encoding::Sop1 {
            if let (Some(d), Some(s)) = (dst, i.sources.first()) {
                if s.kind == OperandKind::ScalarRegister && s.value == EXEC_LO {
                    self.kill_scalar(st, d, 2);
                    if (d as usize) + 1 < SGPRS {
                        st.s[d as usize] = Sv::Exec(st.exec);
                        st.s[d as usize + 1] = Sv::ExecHi(st.exec);
                    }
                    return;
                }
                if s.kind == OperandKind::ScalarRegister && (s.value as usize) + 1 < SGPRS && (d as usize) + 1 < SGPRS {
                    let (a, b) = (st.s[s.value as usize], st.s[s.value as usize + 1]);
                    st.s[d as usize] = a;
                    st.s[d as usize + 1] = b;
                    return;
                }
            }
        }
        if let Some(value) = self.scalar_value(st, i) {
            let d = dst.unwrap_or(u32::MAX);
            self.kill_scalar(st, d, 1);
            if (d as usize) < SGPRS && d != EXEC_LO {
                st.s[d as usize] = Sv::Val(value);
            }
            return;
        }
        if let Some(value) = self.vector_value(st, i) {
            let d = dst.unwrap_or(u32::MAX);
            if op == Opcode::VReadfirstlaneB32 {
                self.kill_scalar(st, d, 1);
                if (d as usize) < SGPRS {
                    st.s[d as usize] = Sv::Val(value);
                }
            } else if (d as usize) < VGPRS {
                st.v[d as usize] = Vv { value, lanes: st.exec };
            }
            // Carry-outs.
            if let Some(sd) = i.control.as_ref().and_then(|c| c.scalar_destination()) {
                self.kill_scalar(st, sd, 2);
            }
            return;
        }
        // Buffer stores list their data registers but write none.
        if matches!(i.encoding, Encoding::Mubuf | Encoding::Mtbuf) && i.opcode.name().contains("Store") {
            return;
        }
        self.clobber(st, i);
    }

    /// Unknown values in the `n` scalar registers from `d` (n + 1 for a
    /// pair base, to be safe).
    fn kill_scalar(&mut self, st: &mut State, d: u32, n: u32) {
        for r in d..d.saturating_add(n) {
            if (r as usize) < SGPRS {
                st.s[r as usize] = Sv::Val(UNKNOWN);
            }
        }
    }

    /// An instruction not modelled: every register it may write becomes
    /// unknown (wider than its destination list, which names only the
    /// first register of some wide results).
    fn clobber(&mut self, st: &mut State, i: &Instruction) {
        let extra = match i.encoding {
            Encoding::Mimg => 6,
            Encoding::Mubuf | Encoding::Mtbuf | Encoding::Flat | Encoding::Ds | Encoding::Smem | Encoding::Smrd => 2,
            _ => 2,
        };
        for d in &i.destinations {
            match d.kind {
                OperandKind::ScalarRegister => self.kill_scalar(st, d.value, extra),
                OperandKind::VectorRegister => {
                    for r in d.value..d.value + extra {
                        if (r as usize) < VGPRS {
                            st.v[r as usize] = Vv { value: UNKNOWN, lanes: 0 };
                        }
                    }
                }
                _ => {}
            }
        }
        if let Some(sd) = i.control.as_ref().and_then(|c| c.scalar_destination()) {
            self.kill_scalar(st, sd, 2);
        }
        // Implicit VCC writes: compares and carries.
        if (i.encoding == Encoding::Vopc && !i.opcode.is_vector_compare_exec()) || i.opcode.is_vector_carry() || (i.encoding == Encoding::Vop2 && i.opcode == Opcode::VCndmaskB32) {
            self.kill_scalar(st, VCC_LO, 2);
        }
        // Instructions that write registers they do not list.
        if matches!(i.opcode, Opcode::VSwapB32) {
            for s in &i.sources {
                if s.kind == OperandKind::VectorRegister && (s.value as usize) < VGPRS {
                    st.v[s.value as usize] = Vv { value: UNKNOWN, lanes: 0 };
                }
            }
        }
    }

    /// The value of a modelled scalar ALU instruction.
    fn scalar_value(&mut self, st: &State, i: &Instruction) -> Option<E> {
        if !matches!(i.encoding, Encoding::Sop1 | Encoding::Sop2 | Encoding::Sopk) {
            return None;
        }
        let s = |a: &mut Analyzer, k: usize| i.sources.get(k).map_or(UNKNOWN, |o| a.scalar_src(st, o));
        Some(match i.opcode {
            Opcode::SMovB32 => s(self, 0),
            Opcode::SMovkI32 => self.expr(Expr::Const(simm16(i.words[0]))),
            Opcode::SAddU32 | Opcode::SAddI32 => {
                let (a, b) = (s(self, 0), s(self, 1));
                self.op2(Expr::Add, a, b)
            }
            Opcode::SSubU32 | Opcode::SSubI32 => {
                let (a, b) = (s(self, 0), s(self, 1));
                self.op2(Expr::Sub, a, b)
            }
            Opcode::SMulI32 => {
                let (a, b) = (s(self, 0), s(self, 1));
                self.op2(Expr::Mul, a, b)
            }
            Opcode::SLshlB32 => {
                let (a, b) = (s(self, 0), s(self, 1));
                let b = self.mask5(b);
                self.op2(Expr::Shl, a, b)
            }
            Opcode::SLshrB32 => {
                let (a, b) = (s(self, 0), s(self, 1));
                let b = self.mask5(b);
                self.op2(Expr::Shr, a, b)
            }
            Opcode::SAndB32 => {
                let (a, b) = (s(self, 0), s(self, 1));
                self.op2(Expr::And, a, b)
            }
            Opcode::SOrB32 => {
                let (a, b) = (s(self, 0), s(self, 1));
                self.op2(Expr::Or, a, b)
            }
            Opcode::SMinU32 => {
                let (a, b) = (s(self, 0), s(self, 1));
                self.op2(Expr::Min, a, b)
            }
            Opcode::SMaxU32 => {
                let (a, b) = (s(self, 0), s(self, 1));
                self.op2(Expr::Max, a, b)
            }
            Opcode::SCselectB32 => {
                let (a, b) = (s(self, 0), s(self, 1));
                self.op2(Expr::Join, a, b)
            }
            Opcode::SLshl1AddU32 | Opcode::SLshl2AddU32 | Opcode::SLshl3AddU32 | Opcode::SLshl4AddU32 => {
                let k = match i.opcode {
                    Opcode::SLshl1AddU32 => 1,
                    Opcode::SLshl2AddU32 => 2,
                    Opcode::SLshl3AddU32 => 3,
                    _ => 4,
                };
                let (a, b) = (s(self, 0), s(self, 1));
                let k = self.expr(Expr::Const(k));
                let sh = self.op2(Expr::Shl, a, k);
                self.op2(Expr::Add, sh, b)
            }
            Opcode::SBfeU32 => {
                let (a, b) = (s(self, 0), s(self, 1));
                // Offset in bits 4:0, width in bits 22:16 of the second source.
                match self.exprs[b as usize] {
                    Expr::Const(c) if (c >> 16) & 0x7F <= 31 => {
                        let o = self.expr(Expr::Const(c & 31));
                        let w = self.expr(Expr::Const((c >> 16) & 0x7F));
                        if a == UNKNOWN {
                            UNKNOWN
                        } else {
                            self.expr(Expr::Bfe(a, o, w))
                        }
                    }
                    _ => UNKNOWN,
                }
            }
            Opcode::SAddkI32 | Opcode::SMulkI32 => {
                let d = i.destinations.first()?.value as usize;
                let cur = match st.s.get(d) {
                    Some(Sv::Val(e)) => *e,
                    _ => UNKNOWN,
                };
                let k = self.expr(Expr::Const(simm16(i.words[0])));
                if i.opcode == Opcode::SAddkI32 {
                    self.op2(Expr::Add, cur, k)
                } else {
                    self.op2(Expr::Mul, cur, k)
                }
            }
            _ => return None,
        })
    }

    /// `b & 31` for shift amounts (exact for constants, else unknown).
    fn mask5(&mut self, b: E) -> E {
        match self.exprs[b as usize] {
            Expr::Const(c) => self.expr(Expr::Const(c & 31)),
            _ => UNKNOWN,
        }
    }

    /// The value of a modelled vector ALU instruction (its lanes: the
    /// current EXEC set).
    fn vector_value(&mut self, st: &State, i: &Instruction) -> Option<E> {
        if !matches!(i.encoding, Encoding::Vop1 | Encoding::Vop2 | Encoding::Vop3) || !plain_vop3(i) {
            return None;
        }
        if i.encoding == Encoding::Vop3 && i.control.is_none() {
            return None;
        }
        let v = |a: &mut Analyzer, k: usize| i.sources.get(k).map_or(UNKNOWN, |o| a.vector_src(st, o));
        let (x, y, z) = (v(self, 0), v(self, 1), v(self, 2));
        Some(match i.opcode {
            Opcode::VMovB32 | Opcode::VReadfirstlaneB32 => x,
            Opcode::VAddI32 | Opcode::VAddNcI32 | Opcode::VAddCoU32 => self.op2(Expr::Add, x, y),
            Opcode::VSubI32 | Opcode::VSubNcI32 | Opcode::VSubCoU32 => self.op2(Expr::Sub, x, y),
            Opcode::VSubrevI32 | Opcode::VSubrevCoU32 => self.op2(Expr::Sub, y, x),
            Opcode::VLshlrevB32 => {
                let s = self.mask5(x);
                self.op2(Expr::Shl, y, s)
            }
            Opcode::VLshrrevB32 => {
                let s = self.mask5(x);
                self.op2(Expr::Shr, y, s)
            }
            Opcode::VLshlB32 => {
                let s = self.mask5(y);
                self.op2(Expr::Shl, x, s)
            }
            Opcode::VLshrB32 => {
                let s = self.mask5(y);
                self.op2(Expr::Shr, x, s)
            }
            Opcode::VAndB32 => self.op2(Expr::And, x, y),
            Opcode::VOrB32 => self.op2(Expr::Or, x, y),
            Opcode::VMulU32U24 => self.op2(Expr::Mul24, x, y),
            Opcode::VMulLoU32 => self.op2(Expr::Mul, x, y),
            Opcode::VMadU32U24 => {
                let m = self.op2(Expr::Mul24, x, y);
                self.op2(Expr::Add, m, z)
            }
            Opcode::VLshlAddU32 => {
                let s = self.mask5(y);
                let sh = self.op2(Expr::Shl, x, s);
                self.op2(Expr::Add, sh, z)
            }
            Opcode::VAddLshlU32 => {
                let sum = self.op2(Expr::Add, x, y);
                let s = self.mask5(z);
                self.op2(Expr::Shl, sum, s)
            }
            Opcode::VAdd3U32 => {
                let sum = self.op2(Expr::Add, x, y);
                self.op2(Expr::Add, sum, z)
            }
            Opcode::VLshlOrU32 => {
                let s = self.mask5(y);
                let sh = self.op2(Expr::Shl, x, s);
                self.op2(Expr::Or, sh, z)
            }
            Opcode::VOr3U32 => {
                let o = self.op2(Expr::Or, x, y);
                self.op2(Expr::Or, o, z)
            }
            Opcode::VAndOrB32 => {
                let a = self.op2(Expr::And, x, y);
                self.op2(Expr::Or, a, z)
            }
            Opcode::VMinU32 => self.op2(Expr::Min, x, y),
            Opcode::VMaxU32 => self.op2(Expr::Max, x, y),
            Opcode::VMin3U32 => {
                let m = self.op2(Expr::Min, x, y);
                self.op2(Expr::Min, m, z)
            }
            Opcode::VMax3U32 => {
                let m = self.op2(Expr::Max, x, y);
                self.op2(Expr::Max, m, z)
            }
            Opcode::VCndmaskB32 => self.op2(Expr::Join, x, y),
            Opcode::VBfeU32 => {
                if x == UNKNOWN {
                    UNKNOWN
                } else {
                    let (o, w) = (self.mask5(y), self.mask5(z));
                    if o == UNKNOWN || w == UNKNOWN {
                        UNKNOWN
                    } else {
                        self.expr(Expr::Bfe(x, o, w))
                    }
                }
            }
            Opcode::VMbcntLoU32B32 | Opcode::VMbcntHiU32B32 => {
                let lanes = self.expr(Expr::Range(0, 32));
                self.op2(Expr::Add, y, lanes)
            }
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = 0x10_0000;

    fn decode(words: &[u32]) -> Program {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        ps5shader::decode_blob(&bytes, BASE).expect("decodes")
    }

    fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
        (op << 25) | (vdst << 17) | ((vsrc1 & 0xFF) << 9) | src0
    }

    /// buffer_store_dword vdata, vaddr, s[srsrc..+3], soffset (offen / idxen).
    fn store(vdata: u32, vaddr: u32, srsrc: u32, soffset: u32, offen: bool, idxen: bool, imm: u32) -> [u32; 2] {
        [0xE000_0000 | (0x1C << 18) | ((offen as u32) << 12) | ((idxen as u32) << 13) | imm, vaddr | (vdata << 8) | ((srsrc / 4) << 16) | (soffset << 24)]
    }

    const ENDPGM: u32 = 0xBF81_0000;
    const V_LSHLREV: u32 = 0x1A;
    const V_ADD: u32 = 0x25;
    const ZERO: u32 = 0x80;

    /// The pcs of the program's buffer stores.
    fn stores(p: &Program) -> Vec<u32> {
        p.instructions.iter().filter(|i| i.opcode == Opcode::BufferStoreDword).map(|i| i.pc).collect()
    }

    fn bounds(words: &[u32], user_count: u32, system: SystemSgprs) -> (Program, Option<StoreBounds>) {
        let p = decode(words);
        let pcs = stores(&p);
        let b = StoreBounds::analyze(&p, 0, user_count, system, true, &pcs);
        (p, b)
    }

    fn inputs(user: &[u32], groups: [u32; 3], local: [u32; 3]) -> DispatchInputs<'_> {
        DispatchInputs { user_data: user, user_base: 0, groups, local }
    }

    #[test]
    fn a_store_at_the_thread_index_covers_the_group() {
        // v1 = v0 << 2; store v2 at v1 offen.
        let st = store(2, 1, 0, ZERO, true, false, 0);
        let (p, b) = bounds(&[vop2(V_LSHLREV, 1, 128 + 2, 0), st[0], st[1], ENDPGM], 4, SystemSgprs::default());
        let b = b.unwrap();
        assert_eq!(b.extent(&stores(&p), 0, &inputs(&[0; 4], [1, 1, 1], [64, 1, 1])), Some((0, 63 * 4 + STORE_BYTES)));
        assert_eq!(b.extent(&stores(&p), 0, &inputs(&[0; 4], [1, 1, 1], [256, 1, 1])), Some((0, 255 * 4 + STORE_BYTES)));
    }

    #[test]
    fn the_global_index_spans_every_group() {
        // v1 = (s4 << 6) + v0 (v_lshl_add_u32, s4 = group id x); store at
        // index v1 (idxen, the V# stride), offset 8.
        let lshl_add = [0xD746_0001, 0x0401_0C04];
        let st = store(2, 1, 0, ZERO, false, true, 8);
        let system = SystemSgprs { work_group: [Some(4), None, None], thread_group_size: None };
        let (p, b) = bounds(&[lshl_add[0], lshl_add[1], st[0], st[1], ENDPGM], 4, system);
        let b = b.unwrap();
        let pcs = stores(&p);
        let i = inputs(&[0; 4], [4, 1, 1], [64, 1, 1]);
        assert_eq!(b.extent(&pcs, 16, &i), Some((8, 255 * 16 + 8 + STORE_BYTES)));
        // A group id SGPR the dispatch does not provide is unknown.
        let (p, b) = bounds(&[lshl_add[0], lshl_add[1], st[0], st[1], ENDPGM], 4, SystemSgprs::default());
        assert_eq!(b.unwrap().extent(&stores(&p), 16, &i), None);
    }

    #[test]
    fn soffset_from_user_data_and_index_stride() {
        // v1 = v0 + s0 (user data); store at index v1 (stride 32), soffset s1, imm 4.
        let st = store(2, 1, 4, 1, false, true, 4);
        let (p, b) = bounds(&[vop2(V_ADD, 1, 0, 0), st[0], st[1], ENDPGM], 8, SystemSgprs::default());
        let b = b.unwrap();
        let user = [10u32, 0x40, 0, 0, 0, 0, 0, 0];
        let i = inputs(&user, [1, 1, 1], [64, 1, 1]);
        // index 10..=73, * 32, + 0x40 + 4.
        assert_eq!(b.extent(&stores(&p), 32, &i), Some((10 * 32 + 0x44, 73 * 32 + 0x44 + STORE_BYTES)));
    }

    /// v_cmp_gt_u32 vcc, s4, v1 (VOPC), s_and_saveexec_b64 s[6:7], vcc,
    /// s_cbranch_execz, the store, s_or_b64 exec, exec, s[6:7].
    fn bounds_checked(address_inside: bool) -> Vec<u32> {
        let cmp = (0x3E << 25) | (0xC4 << 17) | (1 << 9) | 4; // v_cmp_gt_u32 vcc, s4, v1
        let saveexec = 0xBE80_0000 | (6 << 16) | (0x24 << 8) | 106; // s_and_saveexec_b64 s[6:7], vcc
        let st = store(2, 3, 8, ZERO, true, false, 0);
        let shl = vop2(V_LSHLREV, 3, 128 + 2, 1); // v3 = v1 << 2
        let or = 0x8880_0000 | (126 << 16) | (6 << 8) | 126; // s_or_b64 exec, exec, s[6:7]
        let mut w = vec![vop2(V_ADD, 1, 0, 0)]; // v1 = s0 + v0
        if !address_inside {
            w.push(shl);
        }
        w.push(cmp);
        w.push(saveexec);
        let body = if address_inside { 3 } else { 2 };
        w.push(0xBF88_0000 | body); // s_cbranch_execz over the body
        if address_inside {
            w.push(shl);
        }
        w.extend_from_slice(&st);
        w.push(or);
        // After the restore: the same store again, of v3.
        w.extend_from_slice(&st);
        w.push(ENDPGM);
        w
    }

    #[test]
    fn exec_narrowing_keeps_values_restoring_it_forgets_them() {
        let user = [100u32, 0, 0, 0, 1000, 0, 0, 0];
        let i = inputs(&user, [1, 1, 1], [64, 1, 1]);
        // Address computed before the branch: both stores are bounded.
        let p = decode(&bounds_checked(false));
        let pcs = stores(&p);
        assert_eq!(pcs.len(), 2);
        let b = StoreBounds::analyze(&p, 0, 8, SystemSgprs::default(), true, &pcs).unwrap();
        assert_eq!(b.extent(&pcs[..1], 0, &i), Some((400, 163 * 4 + STORE_BYTES)), "inside the if");
        assert_eq!(b.extent(&pcs, 0, &i), Some((400, 163 * 4 + STORE_BYTES)), "after it");
        // Address computed inside the if: known there, not after the restore
        // (the other lanes keep their old v3).
        let p = decode(&bounds_checked(true));
        let pcs = stores(&p);
        let b = StoreBounds::analyze(&p, 0, 8, SystemSgprs::default(), true, &pcs).unwrap();
        assert_eq!(b.extent(&pcs[..1], 0, &i), Some((400, 163 * 4 + STORE_BYTES)));
        assert_eq!(b.extent(&pcs[1..], 0, &i), None);
    }

    #[test]
    fn loops_and_unknown_offsets_give_up() {
        // A backward branch: no analysis at all.
        let st = store(2, 1, 0, ZERO, true, false, 0);
        let p = decode(&[vop2(V_LSHLREV, 1, 128 + 2, 0), st[0], st[1], 0xBF82_FFFC, ENDPGM]);
        assert!(StoreBounds::analyze(&p, 0, 4, SystemSgprs::default(), true, &stores(&p)).is_none());
        // An offset loaded from memory (s_load into s4, v1 = s4).
        let load = [0xF400_0000 | (4 << 6), 0x7D << 25];
        let mov = 0x7E00_0000 | (1 << 17) | (1 << 9) | 4; // v_mov_b32 v1, s4
        let p = decode(&[load[0], load[1], 0xBF8C_C07F, mov, st[0], st[1], ENDPGM]);
        let pcs = stores(&p);
        let b = StoreBounds::analyze(&p, 0, 4, SystemSgprs::default(), true, &pcs).unwrap();
        assert_eq!(b.extent(&pcs, 0, &inputs(&[0; 4], [1, 1, 1], [64, 1, 1])), None);
        // A VGPR never written (not a thread id): unknown.
        let st5 = store(2, 5, 0, ZERO, true, false, 0);
        let p = decode(&[st5[0], st5[1], ENDPGM]);
        let pcs = stores(&p);
        let b = StoreBounds::analyze(&p, 0, 4, SystemSgprs::default(), true, &pcs).unwrap();
        assert_eq!(b.extent(&pcs, 0, &inputs(&[0; 4], [1, 1, 1], [64, 1, 1])), None);
        // An overflowing offset (tid << 30) is not bounded.
        let p = decode(&[vop2(V_LSHLREV, 1, 128 + 30, 0), st[0], st[1], ENDPGM]);
        let pcs = stores(&p);
        let b = StoreBounds::analyze(&p, 0, 4, SystemSgprs::default(), true, &pcs).unwrap();
        assert_eq!(b.extent(&pcs, 0, &inputs(&[0; 4], [1, 1, 1], [64, 1, 1])), None);
    }

    #[test]
    fn workgroup_ids_scale_with_the_groups() {
        // s_lshl_b32 s5, s4, 6 ; v1 = s5 + v0 ; v2 = v1 << 2 ; store v3 at v2.
        let lshl = 0x8F00_0000 | (5 << 16) | ((128 + 6) << 8) | 4;
        let st = store(3, 2, 0, ZERO, true, false, 0);
        let system = SystemSgprs { work_group: [Some(4), None, None], thread_group_size: None };
        let (p, b) = bounds(&[lshl, vop2(V_ADD, 1, 5, 0), vop2(V_LSHLREV, 2, 128 + 2, 1), st[0], st[1], ENDPGM], 4, system);
        let b = b.unwrap();
        let pcs = stores(&p);
        assert_eq!(b.extent(&pcs, 0, &inputs(&[0; 4], [8, 1, 1], [64, 1, 1])), Some((0, (8 * 64 - 1) * 4 + STORE_BYTES)));
        assert_eq!(b.extent(&pcs, 0, &inputs(&[0; 4], [1, 1, 1], [64, 1, 1])), Some((0, 63 * 4 + STORE_BYTES)));
    }

    /// A wave64 mask saved by a 32-bit op, or one whose high half changed,
    /// restores an unknown lane set: values known before are not trusted
    /// after it.
    #[test]
    fn partial_exec_masks_do_not_restore() {
        let user = [100u32, 0, 0, 0, 1000, 0, 0, 0];
        let i = inputs(&user, [1, 1, 1], [64, 1, 1]);
        let check = |w: &[u32]| {
            let p = decode(w);
            let pcs = stores(&p);
            let b = StoreBounds::analyze(&p, 0, 8, SystemSgprs::default(), true, &pcs).unwrap();
            (b.extent(&pcs[..1], 0, &i), b.extent(&pcs[1..], 0, &i))
        };
        let whole = bounds_checked(false);
        assert_eq!(check(&whole), (Some((400, 163 * 4 + STORE_BYTES)), Some((400, 163 * 4 + STORE_BYTES))));
        // s_and_saveexec_b32 (SOP1 0x3C) instead of the 64-bit form.
        let half: Vec<u32> = whole.iter().map(|&w| if w >> 16 == 0xBE86 && (w >> 8) & 0xFF == 0x24 { (w & !0xFF00) | (0x3C << 8) } else { w }).collect();
        assert_ne!(half, whole);
        assert_eq!(check(&half).1, None, "half a mask");
        // s_mov_b32 s7, 0 between the save and the restore (the branch over
        // the body skips it too).
        let mut clobbered = whole.clone();
        let branch = clobbered.iter().position(|&w| w >> 16 == 0xBE86).unwrap() + 1;
        clobbered[branch] += 1;
        clobbered.insert(branch + 1, 0xBE87_0380);
        assert_eq!(check(&clobbered).1, None, "a changed high half");
        // A branch into the middle of the store: no analysis.
        let mut inside = whole.clone();
        inside[branch] -= 1;
        let p = decode(&inside);
        assert!(StoreBounds::analyze(&p, 0, 8, SystemSgprs::default(), true, &stores(&p)).is_none());
    }
}
