// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The values M0 can hold at each V_MOVRELS/V_MOVRELD/V_MOVRELSD, when they
//! follow from straight-line scalar code (bit-field extracts, masks, shifts,
//! adds and multiplies by constants). A back-end that knows the few
//! registers a relative move can reach selects among them with constant
//! register numbers instead of indexing the whole register file at run time.

use super::branch;
use crate::inline_constants;
use crate::ir_types::*;
use crate::opcode::Opcode;
use std::collections::{BTreeSet, HashMap};

const MAX_VALUES: usize = 64;
const MAX_DEPTH: u32 = 8;
const MAX_SCAN_STEPS: usize = 4096;

/// Maps the pc of every resolved relative move to its sorted M0 values; a
/// move missing from the map could not be bounded.
pub fn analyze(program: &Program) -> HashMap<u32, Vec<u32>> {
    let mut result = HashMap::new();
    let instructions = &program.instructions;
    let mut has_relative_move = false;
    for ins in instructions {
        has_relative_move |= ins.opcode.is_bounded_relative_move();
        // A computed jump can land anywhere; no straight-line reasoning holds.
        if ins.opcode.is_computed_jump() {
            return result;
        }
    }
    if !has_relative_move {
        return result;
    }

    // Branch target pc -> index of the single branch reaching it, or -1 when
    // several branches do.
    let mut targets: HashMap<u32, isize> = HashMap::new();
    for (index, ins) in instructions.iter().enumerate() {
        if let Some(t) = branch::branch_target(ins) {
            targets.entry(t).and_modify(|v| *v = -1).or_insert(index as isize);
        }
    }

    for (index, ins) in instructions.iter().enumerate() {
        if !ins.opcode.is_bounded_relative_move() {
            continue;
        }
        let mut values = BTreeSet::new();
        if resolve_register(instructions, &targets, index, M0, 0, &mut values) {
            result.insert(ins.pc, values.into_iter().collect());
        }
    }
    result
}

/// Values `register` holds when the instruction at `index` executes.
fn resolve_register(
    instructions: &[Instruction],
    targets: &HashMap<u32, isize>,
    index: usize,
    register: u32,
    depth: u32,
    values: &mut BTreeSet<u32>,
) -> bool {
    if depth > MAX_DEPTH {
        return false;
    }
    let mut steps = 0;
    let mut at = index as isize - 1;
    while at >= 0 {
        steps += 1;
        if steps > MAX_SCAN_STEPS {
            return false;
        }
        let a = at as usize;
        // A join: follow its one predecessor, or give up when paths meet.
        if let Some(&branch_index) = targets.get(&instructions[a + 1].pc) {
            let falls_through = !branch::is_unconditional_branch(&instructions[a]) && !branch::is_terminator(&instructions[a]);
            if branch_index < 0 || falls_through {
                return false;
            }
            at = branch_index - 1;
            continue;
        }
        let ins = &instructions[a];
        if let Some(exact) = writes(ins, register) {
            return exact && evaluate(instructions, targets, a, ins, depth, values);
        }
        at -= 1;
    }
    false
}

/// `Some(exact)` when the instruction writes `register`; `exact` when its
/// only effect on it is its single listed destination.
fn writes(ins: &Instruction, register: u32) -> Option<bool> {
    let opcode = ins.opcode;
    let wide = opcode.is_wide_scalar();
    for d in ins.destinations.iter().filter(|d| d.kind == OperandKind::ScalarRegister) {
        if d.value == register {
            return Some(!wide && ins.destinations.len() == 1);
        }
        if wide && d.value + 1 == register {
            return Some(false);
        }
    }
    if let Some(w) = ins.control.as_ref().and_then(Control::scalar_destination) {
        if w == register || w + 1 == register {
            return Some(false);
        }
    }
    // Vector compares and carries write VCC or EXEC without listing them;
    // on gfx10 V_CMPX writes EXEC only.
    if opcode.is_vector_compare_exec() {
        return matches!(register, EXEC_LO | EXEC_HI).then_some(false);
    }
    if matches!(register, VCC_LO | VCC_HI) && (opcode.is_vector_compare() || opcode.is_vector_carry()) {
        return Some(false);
    }
    None
}

fn evaluate(
    instructions: &[Instruction],
    targets: &HashMap<u32, isize>,
    index: usize,
    ins: &Instruction,
    depth: u32,
    values: &mut BTreeSet<u32>,
) -> bool {
    let s = &ins.sources;
    match ins.opcode {
        Opcode::SMovB32 if s.len() == 1 => resolve_operand(instructions, targets, index, &s[0], depth, values),
        Opcode::SBfeU32 if s.len() == 2 => {
            let Some(field) = constant(&s[1]) else { return false };
            let width = (field >> 16) & 0x7F;
            if width > 6 {
                return false;
            }
            values.extend(0..1u32 << width);
            true
        }
        Opcode::SAndB32 if s.len() == 2 => {
            let Some(mask) = constant(&s[1]).or_else(|| constant(&s[0])) else { return false };
            if mask as usize >= MAX_VALUES {
                return false;
            }
            values.extend((0..=mask).filter(|v| v & !mask == 0));
            true
        }
        Opcode::SLshlB32 if s.len() == 2 => {
            let Some(shift) = constant(&s[1]) else { return false };
            combine(instructions, targets, index, &s[0], depth, values, |v| v << (shift & 31))
        }
        Opcode::SMulI32 | Opcode::SAddU32 | Opcode::SAddI32 if s.len() == 2 => {
            let multiply = ins.opcode == Opcode::SMulI32;
            let apply = move |v: u32, k: u32| if multiply { v.wrapping_mul(k) } else { v.wrapping_add(k) };
            if let Some(right) = constant(&s[1]) {
                combine(instructions, targets, index, &s[0], depth, values, |v| apply(v, right))
            } else if let Some(left) = constant(&s[0]) {
                combine(instructions, targets, index, &s[1], depth, values, |v| apply(v, left))
            } else {
                false
            }
        }
        _ => false,
    }
}

fn combine(
    instructions: &[Instruction],
    targets: &HashMap<u32, isize>,
    index: usize,
    operand: &Operand,
    depth: u32,
    values: &mut BTreeSet<u32>,
    apply: impl Fn(u32) -> u32,
) -> bool {
    let mut inputs = BTreeSet::new();
    if !resolve_operand(instructions, targets, index, operand, depth, &mut inputs) {
        return false;
    }
    values.extend(inputs.into_iter().map(apply));
    values.len() <= MAX_VALUES
}

fn resolve_operand(
    instructions: &[Instruction],
    targets: &HashMap<u32, isize>,
    index: usize,
    operand: &Operand,
    depth: u32,
    values: &mut BTreeSet<u32>,
) -> bool {
    if let Some(c) = constant(operand) {
        values.insert(c);
        return true;
    }
    operand.kind == OperandKind::ScalarRegister
        && resolve_register(instructions, targets, index, operand.value, depth + 1, values)
        && values.len() <= MAX_VALUES
}

fn constant(operand: &Operand) -> Option<u32> {
    match operand.kind {
        OperandKind::LiteralConstant => Some(operand.value),
        OperandKind::EncodedConstant => inline_constants::decode(operand.value),
        _ => None,
    }
}
