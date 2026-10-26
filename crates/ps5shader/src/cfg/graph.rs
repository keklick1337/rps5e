// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Basic-block control-flow graph of a decoded program.
//!
//! Handles S_BRANCH / S_CBRANCH_* (SCC, VCCZ, EXECZ), the subvector loop
//! branches (S_SUBVECTOR_LOOP_BEGIN / END), S_ENDPGM, falling off
//! the end, S_SETPC_B64 built from S_GETPC_B64 plus a constant offset, and
//! bounded jump tables (a table of offsets in the code object indexed by a
//! clamped value). A jump table becomes a chain of comparisons of the target
//! address against every table entry. The debugger branches
//! (S_CBRANCH_CDBG*) are never taken.

use crate::ir::branch;
use crate::ir_types::*;
use crate::opcode::Opcode;
use crate::program::CodeSource;
use std::collections::{BTreeSet, HashMap};
use std::fmt;

/// Condition of a conditional branch, evaluated at the end of its block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BranchCond {
    Scc0,
    Scc1,
    Vccz,
    Vccnz,
    Execz,
    Execnz,
    /// Computed by the block's last instruction (S_SUBVECTOR_LOOP_BEGIN /
    /// END).
    Instruction,
    /// SGPR `sgpr` equals `value` (jump-table dispatch on the low dword of
    /// the computed target address).
    SgprEquals { sgpr: u8, value: u32 },
}

/// How a block ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    Return,
    Branch(usize),
    /// Goes to `taken` when `cond` holds, else to `fallthrough`.
    Cond { cond: BranchCond, taken: usize, fallthrough: usize },
}

impl Exit {
    pub fn successors(&self) -> Vec<usize> {
        match *self {
            Exit::Return => vec![],
            Exit::Branch(t) => vec![t],
            Exit::Cond { taken, fallthrough, .. } if taken == fallthrough => vec![taken],
            Exit::Cond { taken, fallthrough, .. } => vec![taken, fallthrough],
        }
    }

    pub fn retarget(&mut self, from: usize, to: usize) {
        match self {
            Exit::Return => {}
            Exit::Branch(t) => {
                if *t == from {
                    *t = to
                }
            }
            Exit::Cond { taken, fallthrough, .. } => {
                if *taken == from {
                    *taken = to;
                }
                if *fallthrough == from {
                    *fallthrough = to;
                }
            }
        }
    }
}

/// One basic block: instructions `[first, end)` of the program (branch
/// instructions included; they have no data effect).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BasicBlock {
    pub start_pc: u32,
    pub first: usize,
    pub end: usize,
    pub exit: Exit,
    /// Estimated SPIR-V words of the block (clone budgeting).
    pub words: u32,
    /// Index of the block this one was cloned from (itself if original).
    pub origin: usize,
}

/// The control-flow graph. Blocks are in program order; `entry` is 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cfg {
    pub blocks: Vec<BasicBlock>,
    pub entry: usize,
}

/// Why a graph could not be built or structured.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CfgError {
    EmptyProgram,
    InvalidBranchTarget { pc: u32, target: i64 },
    /// S_SETPC_B64 whose resolved target lies past the decoded program
    /// (`offset` from the program address): the code there was not decoded.
    TargetPastEnd { pc: u32, offset: u64 },
    /// S_SETPC_B64 whose target is not a recognised pattern.
    UnsupportedSetpc { pc: u32 },
    /// A subroutine call outside the call/return model ([`super::calls`]).
    UnsupportedCall { pc: u32, reason: String },
    /// A retreating edge whose target does not dominate its source.
    Irreducible { from_pc: u32, to_pc: u32 },
}

impl fmt::Display for CfgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CfgError::EmptyProgram => write!(f, "empty program"),
            CfgError::InvalidBranchTarget { pc, target } => write!(f, "branch at {pc:#x} targets invalid pc {target:#x}"),
            CfgError::TargetPastEnd { pc, offset } => write!(f, "s_setpc_b64 at {pc:#x} targets offset {offset:#x} past the decoded program"),
            CfgError::UnsupportedSetpc { pc } => write!(f, "unsupported dynamic s_setpc_b64 at {pc:#x}"),
            CfgError::UnsupportedCall { pc, reason } => write!(f, "unsupported scalar call at {pc:#x}: {reason}"),
            CfgError::Irreducible { from_pc, to_pc } => {
                write!(f, "irreducible control flow: edge {from_pc:#x} -> {to_pc:#x} enters a cycle not through its header")
            }
        }
    }
}

impl std::error::Error for CfgError {}

/// Estimated SPIR-V words an instruction expands to (budget heuristic).
pub fn estimated_words(i: &Instruction) -> u32 {
    use Opcode::*;
    match i.opcode {
        VReadfirstlaneB32 | VReadlaneB32 | VWritelaneB32 | VPermlane16B32 | VPermlanex16B32 | DsSwizzleB32
        | DsBpermuteB32 => return 120,
        ImageBvhIntersectRay | ImageBvh64IntersectRay => return 2000,
        _ => {}
    }
    match i.encoding {
        Encoding::Sopp => 0,
        Encoding::Sop1 | Encoding::Sop2 | Encoding::Sopk | Encoding::Sopc => 8,
        Encoding::Vop1 | Encoding::Vop2 | Encoding::Vop3 | Encoding::Vop3p | Encoding::Vopc | Encoding::Vintrp => 60,
        Encoding::Exp => 20,
        Encoding::Mubuf | Encoding::Mtbuf | Encoding::Flat => 120,
        Encoding::Smem | Encoding::Smrd => 180,
        Encoding::Mimg => 220,
        Encoding::Ds => 400,
    }
}

fn cond_of(op: Opcode) -> Option<BranchCond> {
    Some(match op {
        Opcode::SCbranchScc0 => BranchCond::Scc0,
        Opcode::SCbranchScc1 => BranchCond::Scc1,
        Opcode::SCbranchVccz => BranchCond::Vccz,
        Opcode::SCbranchVccnz => BranchCond::Vccnz,
        Opcode::SCbranchExecz => BranchCond::Execz,
        Opcode::SCbranchExecnz => BranchCond::Execnz,
        Opcode::SSubvectorLoopBegin | Opcode::SSubvectorLoopEnd => BranchCond::Instruction,
        _ => return None,
    })
}

/// Resolved computed jump.
enum SetpcTarget {
    /// Program offset of the single target.
    Direct(u64),
    /// Jump table: SGPR holding the target's low dword and the program
    /// offsets of every entry (in table order).
    Table { sgpr: u8, targets: Vec<u64> },
}

fn scalar_reg(o: &Operand) -> Option<u32> {
    (o.kind == OperandKind::ScalarRegister).then_some(o.value)
}

fn immediate(o: &Operand) -> Option<u32> {
    match o.kind {
        OperandKind::LiteralConstant => Some(o.value),
        OperandKind::EncodedConstant => crate::inline_constants::decode(o.value),
        _ => None,
    }
}

fn dst_is(i: &Instruction, reg: u32) -> bool {
    i.destinations.first().and_then(scalar_reg) == Some(reg)
}

/// `sD = sD op imm` or `sD = imm op sD`; returns imm.
fn adds_immediate(i: &Instruction, reg: u32) -> Option<u32> {
    if !dst_is(i, reg) || i.sources.len() < 2 {
        return None;
    }
    let (a, b) = (&i.sources[0], &i.sources[1]);
    if scalar_reg(a) == Some(reg) {
        immediate(b)
    } else if scalar_reg(b) == Some(reg) {
        immediate(a)
    } else {
        None
    }
}

fn subtracts_immediate(i: &Instruction, reg: u32) -> Option<u32> {
    if !dst_is(i, reg) || i.sources.len() < 2 || scalar_reg(&i.sources[0]) != Some(reg) {
        return None;
    }
    immediate(&i.sources[1])
}

/// `sD = sD + sR` (either order) where `sR` is `other`.
fn adds_register(i: &Instruction, reg: u32, other: u32) -> bool {
    dst_is(i, reg)
        && i.sources.len() >= 2
        && ((scalar_reg(&i.sources[0]) == Some(reg) && scalar_reg(&i.sources[1]) == Some(other))
            || (scalar_reg(&i.sources[1]) == Some(reg) && scalar_reg(&i.sources[0]) == Some(other)))
}

/// Program offset of the instruction after `i` (what S_GETPC_B64 returns,
/// relative to the program address).
fn next_offset(i: &Instruction) -> u64 {
    i.program_offset() + i.size_bytes() as u64
}

fn resolve_setpc(p: &Program, index: usize, code: Option<&dyn CodeSource>) -> Option<SetpcTarget> {
    let ins = &p.instructions;
    let setpc = &ins[index];
    let reg = scalar_reg(setpc.sources.first()?)?;
    let at = |k: usize| index.checked_sub(k).map(|j| &ins[j]);
    // getpc; add lo, imm; addc hi, imm; setpc  (or sub/subb)
    if let (Some(pc), Some(low), Some(high)) = (at(3), at(2), at(1)) {
        if reg % 2 == 0 && pc.opcode == Opcode::SGetpcB64 && dst_is(pc, reg) {
            let base = next_offset(pc);
            if low.opcode == Opcode::SAddU32 && high.opcode == Opcode::SAddcU32 {
                if let (Some(lo), Some(hi)) = (adds_immediate(low, reg), adds_immediate(high, reg + 1)) {
                    let offset = ((hi as u64) << 32) | lo as u64;
                    return Some(SetpcTarget::Direct(base.wrapping_add(offset)));
                }
            }
            if low.opcode == Opcode::SSubU32 && high.opcode == Opcode::SSubbU32 {
                if let (Some(lo), Some(hi)) = (subtracts_immediate(low, reg), subtracts_immediate(high, reg + 1)) {
                    let offset = ((hi as u64) << 32) | lo as u64;
                    return Some(SetpcTarget::Direct(base.wrapping_sub(offset)));
                }
            }
        }
    }
    // getpc; add lo, imm; setpc
    if let (Some(pc), Some(arith)) = (at(2), at(1)) {
        if pc.opcode == Opcode::SGetpcB64 && dst_is(pc, reg) {
            let base = next_offset(pc);
            match arith.opcode {
                Opcode::SAddU32 | Opcode::SAddI32 => {
                    if let Some(imm) = adds_immediate(arith, reg) {
                        return Some(SetpcTarget::Direct((base as u32).wrapping_add(imm) as u64 & !3));
                    }
                }
                Opcode::SSubU32 | Opcode::SSubI32 => {
                    if let Some(imm) = subtracts_immediate(arith, reg) {
                        return Some(SetpcTarget::Direct((base as u32).wrapping_sub(imm) as u64 & !3));
                    }
                }
                _ => {}
            }
        }
    }
    // getpc; setpc
    if let Some(pc) = at(1) {
        if pc.opcode == Opcode::SGetpcB64 && dst_is(pc, reg) {
            return Some(SetpcTarget::Direct(next_offset(pc)));
        }
    }
    let code = code?;
    resolve_qword_table(p, index, code).or_else(|| resolve_dword_table(p, index, code))
}

/// Program offset an S_SETPC_B64 / S_SWAPPC_B64 at `index` transfers to
/// when its address is S_GETPC_B64 plus constants.
pub(crate) fn direct_setpc_target(p: &Program, index: usize) -> Option<u64> {
    match resolve_setpc(p, index, None)? {
        SetpcTarget::Direct(off) => Some(off),
        SetpcTarget::Table { .. } => None,
    }
}

/// The S_SETPC_B64 at `index` jumps through a bounded table in the code.
pub(crate) fn resolves_jump_table(p: &Program, index: usize, code: &dyn CodeSource) -> bool {
    resolve_qword_table(p, index, code).is_some() || resolve_dword_table(p, index, code).is_some()
}

fn read_code(p: &Program, code: &dyn CodeSource, offset: u64) -> Option<u32> {
    code.read_u32(p.address.checked_add(offset)?)
}

/// `s_min_u32 sI, sI, max; s_lshl_b32 sI, sI, 3; getpc sB; add sB, disp;
/// addc sB+1, 0; s_load_dwordx2 sE, sB, sI; ...; getpc sP; add sP, sE;
/// addc sP+1, sE+1; setpc sP` — table of signed 64-bit offsets relative to
/// the second getpc.
fn resolve_qword_table(p: &Program, index: usize, code: &dyn CodeSource) -> Option<SetpcTarget> {
    let ins = &p.instructions;
    if index < 9 {
        return None;
    }
    let reg = scalar_reg(ins[index].sources.first()?)?;
    let (pc, low, high) = (&ins[index - 3], &ins[index - 2], &ins[index - 1]);
    if pc.opcode != Opcode::SGetpcB64 || !dst_is(pc, reg) || low.opcode != Opcode::SAddU32 || high.opcode != Opcode::SAddcU32 {
        return None;
    }
    let mut li = index - 3;
    while li > 5 {
        li -= 1;
        let load = &ins[li];
        if load.opcode != Opcode::SLoadDwordx2 {
            if load.opcode == Opcode::SWaitcnt {
                continue;
            }
            let writes_scalar = load.destinations.iter().any(|d| d.kind == OperandKind::ScalarRegister);
            if matches!(load.encoding, Encoding::Vop1 | Encoding::Vop2 | Encoding::Vop3 | Encoding::Vopc) && !writes_scalar {
                continue;
            }
            return None;
        }
        let dst = scalar_reg(load.destinations.first()?)?;
        let base_reg = scalar_reg(load.sources.first()?)?;
        let Some(Control::ScalarMemory(smem)) = &load.control else { return None };
        let index_reg = smem.dynamic_offset_register?;
        if !adds_register(low, reg, dst) || !adds_register(high, reg + 1, dst + 1) {
            return None;
        }
        if reg <= dst + 1 && dst <= reg + 1 {
            return None;
        }
        // The index's shift and bound are the last writers of the index
        // register before the base (other instructions may be scheduled
        // between them).
        let shift_i = last_writer(p, li - 3, &[index_reg])?;
        let bound_i = last_writer(p, shift_i, &[index_reg])?;
        let (base_pc, base_low, base_high, shift, bound) = (&ins[li - 3], &ins[li - 2], &ins[li - 1], &ins[shift_i], &ins[bound_i]);
        if base_pc.opcode != Opcode::SGetpcB64 || !dst_is(base_pc, base_reg) {
            return None;
        }
        let disp = (base_low.opcode == Opcode::SAddU32).then(|| adds_immediate(base_low, base_reg)).flatten()?;
        let carry = (base_high.opcode == Opcode::SAddcU32).then(|| adds_immediate(base_high, base_reg + 1)).flatten()?;
        if carry != 0 || shift.opcode != Opcode::SLshlB32 || !dst_is(shift, index_reg) {
            return None;
        }
        if scalar_reg(&shift.sources[0]) != Some(index_reg) || immediate(&shift.sources[1]) != Some(3) {
            return None;
        }
        if bound.opcode != Opcode::SMinU32 || !dst_is(bound, index_reg) || scalar_reg(&bound.sources[0]) != Some(index_reg) {
            return None;
        }
        let max = immediate(&bound.sources[1])?;
        if max > 255 || (base_reg..=base_reg + 1).contains(&index_reg) {
            return None;
        }
        if branch_into(p, bound.pc, ins[index].pc) {
            return None;
        }
        let table = next_offset(base_pc) as i64 + disp as i64 + smem.immediate_offset_bytes as i64;
        if table < 0 || table & 3 != 0 {
            return None;
        }
        let origin = next_offset(pc) as i64;
        let mut targets = Vec::new();
        for e in 0..=max as i64 {
            let lo = read_code(p, code, (table + e * 8) as u64)?;
            let hi = read_code(p, code, (table + e * 8 + 4) as u64)?;
            let offset = (((hi as u64) << 32) | lo as u64) as i64;
            let target = origin.checked_add(offset)?;
            if target < next_offset(&ins[index]) as i64 || target & 3 != 0 {
                return None;
            }
            targets.push(target as u64);
        }
        return Some(SetpcTarget::Table { sgpr: reg as u8, targets });
    }
    None
}

fn branch_into(p: &Program, from_pc: u32, to_pc: u32) -> bool {
    p.instructions.iter().any(|i| branch::branch_target(i).is_some_and(|t| t > from_pc && t <= to_pc))
}

/// Last instruction before `end` (exclusive) writing one of `regs`, in the
/// same straight-line code.
fn last_writer(p: &Program, end: usize, regs: &[u32]) -> Option<usize> {
    let mut i = end;
    while i > 0 {
        i -= 1;
        let ins = &p.instructions[i];
        if branch::is_branch(ins) || branch::is_computed_jump(ins) {
            return None;
        }
        if ins.destinations.iter().any(|d| d.kind == OperandKind::ScalarRegister && regs.contains(&d.value)) {
            return Some(i);
        }
        if ins.control.as_ref().and_then(Control::scalar_destination).is_some_and(|d| regs.contains(&d) || regs.contains(&(d + 1))) {
            return Some(i);
        }
    }
    None
}

/// `s_min_u32 sI, sI, max; s_lshl_b32 sI, sI, 2; getpc sP; add sP, disp;
/// addc sP+1, 0; s_load_dword sE, sP, sI; sub sP, sP, sE; subb sP+1, 0;
/// setpc sP` — table of 32-bit backward offsets from the table base.
fn resolve_dword_table(p: &Program, index: usize, code: &dyn CodeSource) -> Option<SetpcTarget> {
    let ins = &p.instructions;
    if index < 2 {
        return None;
    }
    let reg = scalar_reg(ins[index].sources.first()?)?;
    let (low, high) = (&ins[index - 2], &ins[index - 1]);
    if reg % 2 != 0 || low.opcode != Opcode::SSubU32 || !dst_is(low, reg) || scalar_reg(&low.sources[0]) != Some(reg) {
        return None;
    }
    let entry_reg = scalar_reg(&low.sources[1])?;
    if entry_reg == reg || entry_reg == reg + 1 || high.opcode != Opcode::SSubbU32 || subtracts_immediate(high, reg + 1) != Some(0) {
        return None;
    }
    let load_i = last_writer(p, index - 2, &[entry_reg, reg, reg + 1])?;
    let load = &ins[load_i];
    let Some(Control::ScalarMemory(smem)) = &load.control else { return None };
    if load.opcode != Opcode::SLoadDword || !dst_is(load, entry_reg) || scalar_reg(&load.sources[0]) != Some(reg) {
        return None;
    }
    let offset_reg = smem.dynamic_offset_register?;
    if offset_reg == reg || offset_reg == reg + 1 {
        return None;
    }
    let shift_i = last_writer(p, load_i, &[offset_reg])?;
    let bound_i = last_writer(p, shift_i, &[offset_reg])?;
    let base_i = last_writer(p, load_i, &[reg, reg + 1])?;
    if base_i < 2 {
        return None;
    }
    let (shift, bound) = (&ins[shift_i], &ins[bound_i]);
    let (base_pc, base_low, base_high) = (&ins[base_i - 2], &ins[base_i - 1], &ins[base_i]);
    if shift.opcode != Opcode::SLshlB32 || !dst_is(shift, offset_reg) || scalar_reg(&shift.sources[0]) != Some(offset_reg) || immediate(&shift.sources[1]) != Some(2) {
        return None;
    }
    if bound.opcode != Opcode::SMinU32 || !dst_is(bound, offset_reg) {
        return None;
    }
    let max = immediate(&bound.sources[1]).or_else(|| immediate(&bound.sources[0]))?;
    if max > 255 || base_pc.opcode != Opcode::SGetpcB64 || !dst_is(base_pc, reg) {
        return None;
    }
    let disp = (base_low.opcode == Opcode::SAddU32).then(|| adds_immediate(base_low, reg)).flatten()?;
    let carry = (base_high.opcode == Opcode::SAddcU32).then(|| adds_immediate(base_high, reg + 1)).flatten()?;
    if carry != 0 {
        return None;
    }
    let first_pc = ins[bound_i.min(base_i - 2)].pc;
    if branch_into(p, first_pc, ins[index].pc) {
        return None;
    }
    let base = next_offset(base_pc) as i64 + disp as i64;
    let table = base + smem.immediate_offset_bytes as i64;
    if table < 0 || table & 3 != 0 {
        return None;
    }
    let mut targets = Vec::new();
    for e in 0..=max as i64 {
        let value = read_code(p, code, (table + e * 4) as u64)?;
        let target = base - value as i64;
        if target < next_offset(&ins[index]) as i64 || target & 3 != 0 {
            return None;
        }
        targets.push(target as u64);
    }
    Some(SetpcTarget::Table { sgpr: reg as u8, targets })
}

impl Cfg {
    /// Build the graph. `code` (guest memory) is needed only to read jump
    /// tables embedded in the code object.
    pub fn build(p: &Program, code: Option<&dyn CodeSource>) -> Result<Cfg, CfgError> {
        let ins = &p.instructions;
        if ins.is_empty() {
            return Err(CfgError::EmptyProgram);
        }
        let end_pc = ins.last().unwrap().next_pc();
        let by_offset: HashMap<u64, u32> = ins.iter().map(|i| (i.program_offset(), i.pc)).collect();
        let pc_set: BTreeSet<u32> = ins.iter().map(|i| i.pc).collect();
        let end_offset = ins.last().unwrap().program_offset() + ins.last().unwrap().size_bytes() as u64;
        let offset_to_pc = |off: u64| -> Option<u32> {
            if let Some(&pc) = by_offset.get(&off) {
                return Some(pc);
            }
            let last = ins.last().unwrap();
            (off == last.program_offset() + last.size_bytes() as u64).then_some(end_pc)
        };

        let calls = super::calls::analyze(p, true).map_err(|e| CfgError::UnsupportedCall { pc: e.pc, reason: e.reason })?;
        let call_at: HashMap<usize, usize> = calls.iter().map(|c| (c.call, c.target)).collect();
        let return_at: HashMap<usize, usize> = calls.iter().map(|c| (c.ret, c.call)).collect();

        // Per-instruction control transfer.
        enum Transfer {
            None,
            End,
            Jump(u32),
            Cond(BranchCond, u32),
            Table { sgpr: u8, targets: Vec<u32> },
        }
        let mut transfers: Vec<Transfer> = Vec::with_capacity(ins.len());
        let mut leaders: BTreeSet<u32> = BTreeSet::new();
        leaders.insert(ins[0].pc);
        for (k, i) in ins.iter().enumerate() {
            let next = i.next_pc();
            let valid = |t: i64| t >= 0 && t <= u32::MAX as i64 && (t as u32 == end_pc || pc_set.contains(&(t as u32)));
            let t = match i.opcode {
                op if op.is_program_end() => Transfer::End,
                _ if call_at.contains_key(&k) => Transfer::Jump(ins[call_at[&k]].pc),
                _ if return_at.contains_key(&k) => Transfer::Jump(ins[return_at[&k]].next_pc()),
                Opcode::SBranch | Opcode::SCbranchScc0 | Opcode::SCbranchScc1 | Opcode::SCbranchVccz
                | Opcode::SCbranchVccnz | Opcode::SCbranchExecz | Opcode::SCbranchExecnz | Opcode::SSubvectorLoopBegin
                | Opcode::SSubvectorLoopEnd => {
                    let off = (i.words[0] & 0xFFFF) as u16 as i16 as i64;
                    let target = next as i64 + off * 4;
                    if !valid(target) {
                        return Err(CfgError::InvalidBranchTarget { pc: i.pc, target });
                    }
                    match cond_of(i.opcode) {
                        Some(c) => Transfer::Cond(c, target as u32),
                        None => Transfer::Jump(target as u32),
                    }
                }
                Opcode::SSetpcB64 | Opcode::SSwappcB64 => match resolve_setpc(p, k, code) {
                    Some(SetpcTarget::Direct(off)) => {
                        // A computed jump never targets the end of the decoded
                        // code: it continues there.
                        if off >= end_offset {
                            return Err(CfgError::TargetPastEnd { pc: i.pc, offset: off });
                        }
                        let Some(target) = offset_to_pc(off) else {
                            return Err(CfgError::InvalidBranchTarget { pc: i.pc, target: off as i64 });
                        };
                        Transfer::Jump(target)
                    }
                    Some(SetpcTarget::Table { sgpr, targets }) => {
                        if let Some(&far) = targets.iter().filter(|&&off| off >= end_offset).max() {
                            return Err(CfgError::TargetPastEnd { pc: i.pc, offset: far });
                        }
                        let mut pcs = Vec::new();
                        for off in targets {
                            match offset_to_pc(off) {
                                Some(pc) => pcs.push(pc),
                                _ => return Err(CfgError::InvalidBranchTarget { pc: i.pc, target: off as i64 }),
                            }
                        }
                        Transfer::Table { sgpr, targets: pcs }
                    }
                    None => return Err(CfgError::UnsupportedSetpc { pc: i.pc }),
                },
                _ => Transfer::None,
            };
            match &t {
                Transfer::None => {}
                Transfer::End => {
                    leaders.insert(next);
                }
                Transfer::Jump(target) | Transfer::Cond(_, target) => {
                    leaders.insert(*target);
                    leaders.insert(next);
                }
                Transfer::Table { targets, .. } => {
                    leaders.extend(targets.iter().copied());
                    leaders.insert(next);
                }
            }
            transfers.push(t);
        }
        leaders.insert(end_pc);

        // Blocks: one per leader; the leader at end_pc is the empty return
        // block reached by falling off the end or branching past it.
        let starts: Vec<u32> = leaders.into_iter().filter(|&l| l <= end_pc).collect();
        let index_of_pc = |pc: u32| ins.partition_point(|i| i.pc < pc);
        let mut blocks: Vec<BasicBlock> = Vec::new();
        let mut block_of: HashMap<u32, usize> = HashMap::new();
        for (n, &start) in starts.iter().enumerate() {
            let stop = starts.get(n + 1).copied().unwrap_or(end_pc);
            let (first, end) = (index_of_pc(start), index_of_pc(stop.max(start)));
            let (first, end) = if start == end_pc { (ins.len(), ins.len()) } else { (first, end) };
            block_of.insert(start, blocks.len());
            let words = ins[first..end].iter().map(estimated_words).sum();
            let id = blocks.len();
            blocks.push(BasicBlock { start_pc: start, first, end, exit: Exit::Return, words, origin: id });
        }
        let mut extra: Vec<BasicBlock> = Vec::new();
        let count = blocks.len();
        for b in 0..count {
            let (first, end) = (blocks[b].first, blocks[b].end);
            let next_block = (b + 1 < count).then_some(b + 1);
            let exit = if first == end {
                // The end-of-program block.
                Exit::Return
            } else {
                let last = end - 1;
                match &transfers[last] {
                    Transfer::End => Exit::Return,
                    Transfer::None => next_block.map_or(Exit::Return, Exit::Branch),
                    Transfer::Jump(t) => Exit::Branch(block_of[t]),
                    Transfer::Cond(c, t) => Exit::Cond { cond: *c, taken: block_of[t], fallthrough: next_block.expect("end block") },
                    Transfer::Table { sgpr, targets } => {
                        // Compare chain: entry k is taken when the computed
                        // address equals its target; the last is the
                        // default. Comparison 0 ends this block, the others
                        // are empty blocks.
                        let m = targets.len();
                        if m == 1 {
                            Exit::Branch(block_of[&targets[0]])
                        } else {
                            let cond_for = |k: usize| {
                                let off = ins.iter().find(|i| i.pc == targets[k]).map_or_else(
                                    || ins.last().map_or(0, |l| l.program_offset() + l.size_bytes() as u64),
                                    |i| i.program_offset(),
                                );
                                BranchCond::SgprEquals { sgpr: *sgpr, value: p.address.wrapping_add(off) as u32 }
                            };
                            let first_extra = count + extra.len();
                            // Node k (k >= 1) is extra block first_extra + k - 1.
                            let node = |k: usize| first_extra + k - 1;
                            for k in 1..m - 1 {
                                let fallthrough = if k + 1 == m - 1 { block_of[&targets[m - 1]] } else { node(k + 1) };
                                let id = node(k);
                                extra.push(BasicBlock {
                                    start_pc: blocks[b].start_pc,
                                    first: end,
                                    end,
                                    exit: Exit::Cond { cond: cond_for(k), taken: block_of[&targets[k]], fallthrough },
                                    words: 0,
                                    origin: id,
                                });
                            }
                            let fallthrough = if m == 2 { block_of[&targets[1]] } else { node(1) };
                            Exit::Cond { cond: cond_for(0), taken: block_of[&targets[0]], fallthrough }
                        }
                    }
                }
            };
            blocks[b].exit = exit;
        }
        blocks.extend(extra);
        let mut cfg = Cfg { blocks, entry: 0 };
        cfg.prune_unreachable();
        Ok(cfg)
    }

    pub fn successors(&self) -> Vec<Vec<usize>> {
        self.blocks.iter().map(|b| b.exit.successors()).collect()
    }

    /// Drop blocks unreachable from the entry, renumbering the rest.
    pub fn prune_unreachable(&mut self) {
        let succs = self.successors();
        let mut reachable = vec![false; self.blocks.len()];
        for b in super::dom::reverse_post_order(&succs, self.entry) {
            reachable[b] = true;
        }
        if reachable.iter().all(|&r| r) {
            return;
        }
        let mut map = vec![usize::MAX; self.blocks.len()];
        let mut kept = Vec::new();
        for (i, b) in self.blocks.iter().enumerate() {
            if reachable[i] {
                map[i] = kept.len();
                kept.push(b.clone());
            }
        }
        for b in kept.iter_mut() {
            b.exit = match b.exit {
                Exit::Return => Exit::Return,
                Exit::Branch(t) => Exit::Branch(map[t]),
                Exit::Cond { cond, taken, fallthrough } => Exit::Cond { cond, taken: map[taken], fallthrough: map[fallthrough] },
            };
            if b.origin < map.len() && map[b.origin] != usize::MAX {
                b.origin = map[b.origin];
            }
        }
        self.entry = map[self.entry];
        self.blocks = kept;
    }

    /// Error if the graph is irreducible.
    pub fn check_reducible(&self) -> Result<(), CfgError> {
        let succs = self.successors();
        match super::dom::irreducible_edge(&succs, self.entry) {
            None => Ok(()),
            Some((from, to)) => Err(CfgError::Irreducible { from_pc: self.blocks[from].start_pc, to_pc: self.blocks[to].start_pc }),
        }
    }
}

impl fmt::Display for Cfg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, b) in self.blocks.iter().enumerate() {
            writeln!(f, "B{i} @{:#x} [{}..{}) -> {:?}", b.start_pc, b.first, b.end, b.exit)?;
        }
        Ok(())
    }
}
