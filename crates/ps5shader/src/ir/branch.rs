// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Branch classification and SOPP branch-target resolution.

use crate::ir_types::{Encoding, Instruction, OperandKind, NULL_SOURCE};
use crate::opcode::Opcode;

/// S_BRANCH.
pub fn is_unconditional_branch(i: &Instruction) -> bool {
    i.opcode == Opcode::SBranch
}

/// S_SETPC_B64, or S_SWAPPC_B64 discarding its return address (null
/// destination), which is the same jump.
pub fn is_computed_jump(i: &Instruction) -> bool {
    match i.opcode {
        Opcode::SSetpcB64 => true,
        Opcode::SSwappcB64 => i.destinations.first().is_some_and(|d| d.kind == OperandKind::ScalarRegister && d.value == NULL_SOURCE),
        _ => false,
    }
}

/// S_ENDPGM.
pub fn is_terminator(i: &Instruction) -> bool {
    i.opcode.is_program_end()
}

/// S_SUBVECTOR_LOOP_BEGIN / END: SOPK branches whose condition the
/// instruction itself computes from EXEC.
pub fn is_subvector_loop(i: &Instruction) -> bool {
    matches!(i.opcode, Opcode::SSubvectorLoopBegin | Opcode::SSubvectorLoopEnd)
}

/// S_CBRANCH_* (including the debugger forms) and the subvector loop
/// branches.
pub fn is_conditional(i: &Instruction) -> bool {
    use Opcode::*;
    matches!(
        i.opcode,
        SSubvectorLoopBegin
            | SSubvectorLoopEnd
            | SCbranchScc0
            | SCbranchScc1
            | SCbranchVccz
            | SCbranchVccnz
            | SCbranchExecz
            | SCbranchExecnz
            | SCbranchCdbgsys
            | SCbranchCdbguser
            | SCbranchCdbgsysOrUser
            | SCbranchCdbgsysAndUser
    )
}

pub fn is_branch(i: &Instruction) -> bool {
    is_unconditional_branch(i) || is_conditional(i) || is_terminator(i)
}

/// Target pc of a (conditional) branch: next pc + simm16 * 4 (SOPP, or the
/// SOPK subvector loop branches).
pub fn branch_target(i: &Instruction) -> Option<u32> {
    if is_terminator(i) || !(is_unconditional_branch(i) || is_conditional(i)) {
        return None;
    }
    let encoding = if is_subvector_loop(i) { Encoding::Sopk } else { Encoding::Sopp };
    if i.encoding != encoding || i.words.is_empty() {
        return None;
    }
    let offset = (i.words[0] & 0xFFFF) as u16 as i16 as i64;
    let target = i.pc as i64 + i.words.len() as i64 * 4 + offset * 4;
    u32::try_from(target).ok()
}

/// Branch semantics a control-flow analysis needs.
pub trait BranchResolver {
    fn is_branch(&self, i: &Instruction) -> bool;
    fn is_conditional(&self, i: &Instruction) -> bool;
    fn branch_target(&self, i: &Instruction) -> Option<u32>;
}

/// The SOPP branch resolver.
#[derive(Clone, Copy, Debug, Default)]
pub struct Gen5BranchResolver;

impl BranchResolver for Gen5BranchResolver {
    fn is_branch(&self, i: &Instruction) -> bool {
        is_branch(i)
    }

    fn is_conditional(&self, i: &Instruction) -> bool {
        is_conditional(i)
    }

    fn branch_target(&self, i: &Instruction) -> Option<u32> {
        branch_target(i)
    }
}
