// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Wave64 lane-mask halves.
//!
//! When a wave64 program runs as two wave32 halves, a lane mask written by a
//! vector compare / carry (V_CMP*, V_CMPX*, V_ADD_CO*, ...) holds only the
//! running half's bits. Bitwise mask arithmetic on such "half" masks stays
//! per-half; any other read (branch on VCCZ/EXECZ, SCC-producing logic whose
//! SCC is consumed, scalar reads of the pair) needs the exact 64-bit value,
//! so both halves must be combined first. The analysis reports:
//!
//! * `exact_pairs_before`: per pc, the SGPR pair bases that must be made
//!   exact before the instruction;
//! * `half_mask_writes`: per pc, the pair base a half mask is written to;
//! * `shared_memory_barriers_before`: pcs needing an LDS barrier between the
//!   two halves (a half-exact sync point separates their LDS accesses).

use super::branch;
use crate::ir_types::*;
use crate::opcode::Opcode;
use std::collections::{HashMap, HashSet, VecDeque};

const VCC: u32 = VCC_LO;
const EXEC: u32 = EXEC_LO;
const EVEN_REGISTERS: u128 = 0x5555_5555_5555_5555_5555_5555_5555_5555;
const ODD_REGISTERS: u128 = EVEN_REGISTERS << 1;

/// Result of [`analyze`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Wave64HalfMaskPlan {
    pub exact_pairs_before: HashMap<u32, Vec<u32>>,
    pub half_mask_writes: HashMap<u32, u32>,
    pub shared_memory_barriers_before: HashSet<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct State {
    reached: bool,
    half: u128,
    pending_read: bool,
    pending_write: bool,
}

impl State {
    fn meet(left: State, right: State) -> State {
        if !left.reached {
            right
        } else if !right.reached {
            left
        } else {
            State {
                reached: true,
                half: left.half | right.half,
                pending_read: left.pending_read | right.pending_read,
                pending_write: left.pending_write | right.pending_write,
            }
        }
    }
}

fn is_bitwise_with_scc(op: Opcode) -> bool {
    use Opcode::*;
    matches!(
        op,
        SNotB64
            | SAndB64
            | SOrB64
            | SXorB64
            | SAndn2B64
            | SOrn2B64
            | SNandB64
            | SNorB64
            | SXnorB64
            | SAndSaveexecB64
            | SOrSaveexecB64
            | SXorSaveexecB64
            | SAndn1SaveexecB64
            | SAndn2SaveexecB64
            | SOrn1SaveexecB64
            | SOrn2SaveexecB64
            | SNandSaveexecB64
            | SNorSaveexecB64
            | SXnorSaveexecB64
    )
}

fn is_bitwise_without_scc(op: Opcode) -> bool {
    matches!(op, Opcode::SMovB64 | Opcode::SWqmB64 | Opcode::SCselectB64)
}

fn is_scc_reader_opcode(op: Opcode) -> bool {
    use Opcode::*;
    matches!(op, SCbranchScc0 | SCbranchScc1 | SAddcU32 | SSubbU32 | SCselectB32 | SCselectB64)
}

fn is_scc_writer_opcode(op: Opcode) -> bool {
    use Opcode::*;
    matches!(
        op,
        SNotB32
            | SAbsI32
            | SWqmB32
            | SBrevB32
            | SBcnt1I32B32
            | SBcnt1I32B64
            | SQuadmaskB32
            | SQuadmaskB64
            | SAddU32
            | SSubU32
            | SAddI32
            | SSubI32
            | SAddcU32
            | SSubbU32
            | SAndB32
            | SOrB32
            | SXorB32
            | SAndn2B32
            | SOrn2B32
            | SNandB32
            | SNorB32
            | SXnorB32
            | SLshlB32
            | SLshrB32
            | SAshrI32
            | SBfeU32
            | SBfeI32
            | SMinU32
            | SMinI32
            | SMaxU32
            | SMaxI32
            | SLshlB64
            | SLshrB64
            | SAshrI64
            | SBfeU64
            | SBfeI64
            | SBfmB64
    )
}

fn bit(register: u32) -> u128 {
    if register < 128 {
        1u128 << register
    } else {
        0
    }
}

fn expand(registers: u128) -> u128 {
    registers | ((registers & EVEN_REGISTERS) << 1) | ((registers & ODD_REGISTERS) >> 1)
}

fn pair(register: u32) -> u128 {
    expand(bit(register))
}

fn range(register: u32, count: u32) -> u128 {
    (0..count).fold(0, |acc, o| acc | bit(register + o))
}

fn pair_bases(registers: u128) -> Vec<u32> {
    (0..128).step_by(2).filter(|&r| registers & pair(r) != 0).collect()
}

/// Analyse `program`; `shared_flat_pcs` are FLAT accesses known to hit LDS.
/// `None` when the program has dynamic control flow or a branch into the
/// middle of the decoded range that is not an instruction boundary.
pub fn analyze(program: &Program, shared_flat_pcs: &HashSet<u32>) -> Option<Wave64HalfMaskPlan> {
    let instructions = &program.instructions;
    if instructions.is_empty() {
        return None;
    }
    let successors = build_successors(instructions)?;
    let scc_live_after = compute_scc_live_after(instructions, &successors);
    let mut states = vec![State::default(); instructions.len()];
    states[0] = State { reached: true, ..State::default() };
    let mut pending = VecDeque::from([0usize]);
    let mut queued = vec![false; instructions.len()];
    queued[0] = true;
    while let Some(index) = pending.pop_front() {
        queued[index] = false;
        let output = step(&instructions[index], states[index], scc_live_after[index], shared_flat_pcs).state;
        for &s in &successors[index] {
            let merged = State::meet(states[s], output);
            if merged != states[s] {
                states[s] = merged;
                if !queued[s] {
                    queued[s] = true;
                    pending.push_back(s);
                }
            }
        }
    }

    let mut plan = Wave64HalfMaskPlan::default();
    for (index, ins) in instructions.iter().enumerate() {
        if !states[index].reached {
            continue;
        }
        let r = step(ins, states[index], scc_live_after[index], shared_flat_pcs);
        if r.exact != 0 {
            plan.exact_pairs_before.insert(ins.pc, pair_bases(r.exact));
        }
        if r.shared_barrier {
            plan.shared_memory_barriers_before.insert(ins.pc);
        }
        if r.half_write {
            if let Some(d) = half_mask_destination(ins) {
                plan.half_mask_writes.insert(ins.pc, d);
            }
        }
    }
    Some(plan)
}

struct StepResult {
    state: State,
    exact: u128,
    shared_barrier: bool,
    half_write: bool,
}

fn step(ins: &Instruction, input: State, scc_live_after: bool, shared_flat_pcs: &HashSet<u32>) -> StepResult {
    let opcode = ins.opcode;
    let mut half = input.half;
    let bitwise = is_bitwise_mask_operation(ins);
    let mut needed = 0u128;
    let mut bitwise_inputs = 0u128;
    for (index, operand) in ins.sources.iter().enumerate() {
        if operand.kind == OperandKind::EncodedConstant {
            if operand.value == VCCZ_SOURCE {
                needed |= pair(VCC);
            } else if operand.value == EXECZ_SOURCE {
                needed |= pair(EXEC);
            }
            continue;
        }
        if operand.kind != OperandKind::ScalarRegister {
            continue;
        }
        if bitwise {
            bitwise_inputs |= pair(operand.value);
            continue;
        }
        if index == 2 && (opcode == Opcode::VCndmaskB32 || opcode.reads_vector_carry_in()) {
            continue;
        }
        needed |= read_registers(ins, index, operand.value);
    }
    match opcode {
        Opcode::SCbranchExecz | Opcode::SCbranchExecnz | Opcode::SSubvectorLoopBegin | Opcode::SSubvectorLoopEnd => {
            needed |= pair(EXEC)
        }
        Opcode::SCbranchVccz | Opcode::SCbranchVccnz => needed |= pair(VCC),
        _ => {}
    }
    let is_bitset = matches!(opcode, Opcode::SBitset0B32 | Opcode::SBitset0B64 | Opcode::SBitset1B32 | Opcode::SBitset1B64);
    if ins.encoding == Encoding::Sopk || is_bitset {
        if let Some(d) = ins.destinations.first().filter(|d| d.kind == OperandKind::ScalarRegister) {
            needed |= range(d.value, if opcode.name().contains("64") { 2 } else { 1 });
        }
    }
    if bitwise {
        if opcode.is_saveexec_b64() {
            bitwise_inputs |= pair(EXEC);
        }
        if scc_live_after && is_bitwise_with_scc(opcode) {
            needed |= bitwise_inputs;
        }
    }

    let exact = expand(needed & half);
    half &= !exact;

    let mut pending_read = input.pending_read;
    let mut pending_write = input.pending_write;
    if exact != 0 {
        pending_read = false;
        pending_write = false;
    }
    let (reads, writes) = shared_memory_access(ins, shared_flat_pcs);
    let shared_barrier = (writes && (pending_read || pending_write)) || (reads && pending_write);
    if shared_barrier {
        pending_read = false;
        pending_write = false;
    }
    pending_read |= reads;
    pending_write |= writes;
    if opcode == Opcode::SBarrier {
        pending_read = false;
        pending_write = false;
    }

    let mut half_write = false;
    if let Some(d) = half_mask_destination(ins) {
        half |= pair(d);
        half_write = true;
    } else if bitwise {
        let destination = ins.destinations[0].value;
        let output_half = bitwise_inputs & half != 0;
        half = if output_half { half | pair(destination) } else { half & !pair(destination) };
        if opcode.is_saveexec_b64() {
            half = if output_half { half | pair(EXEC) } else { half & !pair(EXEC) };
        }
    } else {
        let wide = opcode.is_wide_scalar();
        for d in ins.destinations.iter().filter(|d| d.kind == OperandKind::ScalarRegister) {
            half &= !bit(d.value);
            if wide {
                half &= !bit(d.value + 1);
            }
        }
        if opcode.is_saveexec_b32() {
            half &= !pair(EXEC);
        }
    }

    StepResult { state: State { reached: true, half, pending_read, pending_write }, exact, shared_barrier, half_write }
}

/// Pair base a vector compare / carry writes its lane mask to, when it is
/// an even SGPR below M0 or EXEC.
fn half_mask_destination(ins: &Instruction) -> Option<u32> {
    let opcode = ins.opcode;
    let destination = if opcode.is_vector_compare_exec() {
        EXEC
    } else if opcode.is_vector_compare() {
        ins.control.as_ref().and_then(Control::scalar_destination).unwrap_or(VCC)
    } else if opcode.writes_vector_carry_out() {
        match &ins.control {
            Some(Control::Vop3(c)) => c.scalar_destination.unwrap_or(VCC),
            _ => VCC,
        }
    } else {
        return None;
    };
    (destination & 1 == 0 && (destination < M0 || destination == EXEC)).then_some(destination)
}

fn is_bitwise_mask_operation(ins: &Instruction) -> bool {
    if !is_bitwise_with_scc(ins.opcode) && !is_bitwise_without_scc(ins.opcode) {
        return false;
    }
    let [d] = ins.destinations.as_slice() else { return false };
    let bad_pair = |v: u32| v & 1 != 0 || (v >= M0 && v != EXEC);
    if d.kind != OperandKind::ScalarRegister || bad_pair(d.value) {
        return false;
    }
    ins.sources.iter().all(|s| match s.kind {
        OperandKind::VectorRegister => false,
        OperandKind::ScalarRegister => !bad_pair(s.value),
        _ => true,
    })
}

fn shared_memory_access(ins: &Instruction, shared_flat_pcs: &HashSet<u32>) -> (bool, bool) {
    if let Some(Control::DataShare(c)) = &ins.control {
        if !c.gds {
            let opcode = ins.opcode;
            if matches!(opcode, Opcode::DsSwizzleB32 | Opcode::DsBpermuteB32) {
                return (false, false);
            }
            let name = opcode.name();
            if name.starts_with("DsRead") {
                return (true, false);
            }
            if name.starts_with("DsWrite") {
                return (false, true);
            }
            return (true, true);
        }
    }
    if shared_flat_pcs.contains(&ins.pc) {
        (true, true)
    } else {
        (false, false)
    }
}

fn build_successors(instructions: &[Instruction]) -> Option<Vec<Vec<usize>>> {
    let mut index_by_pc = HashMap::with_capacity(instructions.len());
    for (index, ins) in instructions.iter().enumerate() {
        if ins.opcode.is_computed_jump() {
            return None;
        }
        index_by_pc.insert(ins.pc, index);
    }
    let end_pc = instructions.last().unwrap().next_pc();
    let mut successors = Vec::with_capacity(instructions.len());
    for (index, ins) in instructions.iter().enumerate() {
        let next = (index + 1 < instructions.len()).then_some(index + 1);
        if branch::is_terminator(ins) {
            successors.push(vec![]);
            continue;
        }
        let unconditional = branch::is_unconditional_branch(ins);
        if !unconditional && !branch::is_conditional(ins) {
            successors.push(next.into_iter().collect());
            continue;
        }
        let target_pc = branch::branch_target(ins)?;
        let mut targets = Vec::with_capacity(2);
        if let Some(&t) = index_by_pc.get(&target_pc) {
            targets.push(t);
        } else if target_pc < end_pc {
            return None;
        }
        if !unconditional {
            targets.extend(next);
        }
        successors.push(targets);
    }
    Some(successors)
}

fn reads_scc(ins: &Instruction) -> bool {
    is_scc_reader_opcode(ins.opcode)
        || ins.sources.iter().any(|s| s.kind == OperandKind::EncodedConstant && s.value == SCC_SOURCE)
}

fn writes_scc(ins: &Instruction) -> bool {
    ins.encoding == Encoding::Sopc
        || ins.opcode.name().starts_with("SCmpk")
        || ins.opcode.is_saveexec_b32()
        || is_scc_writer_opcode(ins.opcode)
        || is_bitwise_with_scc(ins.opcode)
}

fn compute_scc_live_after(instructions: &[Instruction], successors: &[Vec<usize>]) -> Vec<bool> {
    let n = instructions.len();
    let mut live_before = vec![false; n];
    let mut live_after = vec![false; n];
    loop {
        let mut changed = false;
        for index in (0..n).rev() {
            let after = successors[index].iter().any(|&s| live_before[s]);
            let ins = &instructions[index];
            let before = reads_scc(ins) || (!writes_scc(ins) && after);
            if after != live_after[index] || before != live_before[index] {
                live_after[index] = after;
                live_before[index] = before;
                changed = true;
            }
        }
        if !changed {
            return live_after;
        }
    }
}

/// SGPRs read through scalar source `source_index` (resource descriptors
/// and 64-bit operands span several registers).
fn read_registers(ins: &Instruction, source_index: usize, register: u32) -> u128 {
    let scalar_ordinal = ins.sources[..source_index].iter().filter(|s| s.kind == OperandKind::ScalarRegister).count();
    let count = match ins.encoding {
        Encoding::Smem | Encoding::Smrd => {
            if scalar_ordinal != 0 {
                1
            } else if ins.opcode.name().starts_with("SLoad") {
                2
            } else {
                4
            }
        }
        Encoding::Mubuf | Encoding::Mtbuf => {
            if scalar_ordinal == 0 {
                4
            } else {
                1
            }
        }
        Encoding::Mimg => {
            if scalar_ordinal == 0 {
                8
            } else {
                4
            }
        }
        Encoding::Flat => 2,
        _ => {
            if ins.opcode.name().contains("64") {
                2
            } else {
                1
            }
        }
    };
    range(register, count)
}
