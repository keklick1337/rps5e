// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Which instructions run with every EXEC lane set.
//!
//! EXEC starts full; it stays full through S_WQM of a full EXEC and comes
//! back when a saved copy of a full EXEC is moved or ORed back into it. Any
//! other EXEC write (V_CMPX, S_AND_SAVEEXEC, ...) makes it unknown, and a
//! join is full only when every path into it is. A back-end can skip the
//! EXEC guard of a vector register write there: no lane keeps its old value.

use super::branch;
use crate::ir_types::*;
use crate::opcode::Opcode;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct State {
    reached: bool,
    exec_full: bool,
    /// SGPRs 0..127 holding (part of) a copy of a full EXEC.
    full_copies: u128,
}

impl State {
    fn meet(left: State, right: State) -> State {
        if !left.reached {
            right
        } else if !right.reached {
            left
        } else {
            State { reached: true, exec_full: left.exec_full && right.exec_full, full_copies: left.full_copies & right.full_copies }
        }
    }
}

fn bit(register: u32) -> u128 {
    if register < 128 {
        1u128 << register
    } else {
        0
    }
}

/// Pcs of the instructions that start with a full EXEC. Empty when the
/// program has a computed jump or branches out of the decoded range.
pub fn analyze(program: &Program, wave32: bool) -> HashSet<u32> {
    let instructions = &program.instructions;
    let mut result = HashSet::new();
    if instructions.is_empty() {
        return result;
    }
    let mut index_by_pc = HashMap::with_capacity(instructions.len());
    for (index, ins) in instructions.iter().enumerate() {
        if ins.opcode.is_computed_jump() {
            return result;
        }
        index_by_pc.insert(ins.pc, index);
    }

    let mut states = vec![State::default(); instructions.len()];
    states[0] = State { reached: true, exec_full: true, full_copies: 0 };
    let mut pending = VecDeque::from([0usize]);
    let mut queued = vec![false; instructions.len()];
    queued[0] = true;
    while let Some(index) = pending.pop_front() {
        queued[index] = false;
        let ins = &instructions[index];
        let output = transfer(ins, states[index], wave32);
        let mut flow = |successor: usize, states: &mut Vec<State>| {
            let merged = State::meet(states[successor], output);
            if merged != states[successor] {
                states[successor] = merged;
                if !queued[successor] {
                    queued[successor] = true;
                    pending.push_back(successor);
                }
            }
        };
        if branch::is_terminator(ins) {
            continue;
        }
        if let Some(target_pc) = branch::branch_target(ins) {
            let Some(&target) = index_by_pc.get(&target_pc) else {
                // A branch out of the decoded range: give up on the program.
                return HashSet::new();
            };
            flow(target, &mut states);
            if branch::is_unconditional_branch(ins) {
                continue;
            }
        }
        if index + 1 < instructions.len() {
            flow(index + 1, &mut states);
        }
    }

    for (ins, state) in instructions.iter().zip(&states) {
        if state.reached && state.exec_full {
            result.insert(ins.pc);
        }
    }
    result
}

fn is_exec(op: &Operand) -> bool {
    op.is_scalar(EXEC_LO)
}

/// The operand is a full EXEC or a saved copy of one.
fn is_full_copy(op: &Operand, copies: u128, exec_full: bool, wide: bool) -> bool {
    if op.kind != OperandKind::ScalarRegister {
        return false;
    }
    if op.value == EXEC_LO {
        return exec_full;
    }
    let mask = bit(op.value) | if wide { bit(op.value + 1) } else { 0 };
    copies & mask == mask
}

fn transfer(ins: &Instruction, input: State, wave32: bool) -> State {
    if !input.reached {
        return input;
    }
    let exec_full = input.exec_full;
    let copies = input.full_copies;
    let opcode = ins.opcode;
    let sources = &ins.sources;
    let destinations = &ins.destinations;
    let wide = opcode.is_wide_scalar();

    let mut written = 0u128;
    for d in destinations.iter().filter(|d| d.kind == OperandKind::ScalarRegister) {
        written |= bit(d.value);
        if wide {
            written |= bit(d.value + 1);
        }
    }
    let scalar_destination = ins.control.as_ref().and_then(Control::scalar_destination);
    if let Some(sdst) = scalar_destination {
        written |= bit(sdst) | bit(sdst + 1);
    }
    if opcode.is_vector_compare_exec() {
        written |= bit(EXEC_LO) | bit(EXEC_HI);
    } else if opcode.is_vector_compare() && scalar_destination.is_none() {
        // VOPC compares write VCC.
        written |= bit(VCC_LO) | bit(VCC_HI);
    }
    if opcode.is_vector_carry() && scalar_destination.is_none() {
        // VOP2 carry forms write VCC without listing it.
        written |= bit(VCC_LO) | bit(VCC_HI);
    }
    if opcode.is_saveexec() {
        written |= bit(EXEC_LO) | bit(EXEC_HI);
    }

    let writes_exec = written & (bit(EXEC_LO) | bit(EXEC_HI)) != 0;
    let mut next_copies = copies & !written;
    let mut next_exec_full = exec_full;
    if writes_exec {
        next_exec_full = false;
        let only_exec_destination = destinations.len() == 1 && destinations[0].is_scalar(EXEC_LO);
        // In wave64 a 32-bit write leaves EXEC_HI as it was.
        if only_exec_destination && scalar_destination.is_none() && (wide || wave32) {
            match opcode {
                Opcode::SWqmB64 | Opcode::SWqmB32 if sources.len() == 1 && is_exec(&sources[0]) => {
                    next_exec_full = exec_full;
                }
                Opcode::SMovB64 | Opcode::SMovB32 if sources.len() == 1 => {
                    next_exec_full = is_full_copy(&sources[0], copies, exec_full, wide);
                }
                Opcode::SOrB64 | Opcode::SOrB32 if sources.len() == 2 => {
                    next_exec_full =
                        is_full_copy(&sources[0], copies, exec_full, wide) || is_full_copy(&sources[1], copies, exec_full, wide);
                }
                _ => {}
            }
        }
    } else if exec_full
        && matches!(opcode, Opcode::SMovB64 | Opcode::SMovB32)
        && destinations.len() == 1
        && destinations[0].kind == OperandKind::ScalarRegister
        && sources.len() == 1
        && is_exec(&sources[0])
    {
        // A saved copy of the full EXEC.
        next_copies |= bit(destinations[0].value);
        if wide {
            next_copies |= bit(destinations[0].value + 1);
        }
    }

    if opcode.is_saveexec()
        && exec_full
        && destinations.first().is_some_and(|d| d.kind == OperandKind::ScalarRegister && d.value < EXEC_LO)
    {
        // The saved destination receives the (full) EXEC from before.
        next_copies |= bit(destinations[0].value);
        if wide {
            next_copies |= bit(destinations[0].value + 1);
        }
    }

    State { reached: true, exec_full: next_exec_full, full_copies: next_copies }
}
