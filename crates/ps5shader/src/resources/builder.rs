// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Builds the value graph: walks the control-flow graph to a fixpoint.
//! Registers hold graph values, joins create phis per register (or selects
//! at uniform if/else joins), and a final recording pass stores each memory
//! access's handles and each branch's condition.

use super::graph::{MemoryAccessBinding, ScalarValueGraph};
use super::memory::is_scalar_buffer_load;
use super::value::{ScalarOp, ValueId, ValueKind, ValueType};
use super::ResourcePlanError;
use crate::inline_constants;
use crate::ir_types::*;
use crate::opcode::Opcode;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

const SCALAR_REGISTER_COUNT: usize = 256;
const VECTOR_REGISTER_COUNT: usize = 256;
const VCC_LOW: u32 = 106;
const EXEC_LOW: u32 = 126;
const EXEC_HIGH: u32 = 127;
const NULL_SCALAR_REGISTER: u32 = 125;
const FULL_WAVE_MASK: u32 = 0xFFFF_FFFF;
const BLOCK_VISIT_LIMIT: u32 = 64;

const EXEC_SLOT: usize = 4096;
const VCC_SLOT: usize = 4097;
const SCC_SLOT: usize = 4098;

/// `Some(signed)` for the SOPK compares (`s_cmpk_*`): whether the 16-bit
/// immediate is sign-extended (the _i32 forms) or zero-extended (_u32).
fn sopk_compare_signed(opcode: Opcode) -> Option<bool> {
    use Opcode::*;
    match opcode {
        SCmpkEqI32 | SCmpkLgI32 | SCmpkGtI32 | SCmpkGeI32 | SCmpkLtI32 | SCmpkLeI32 => Some(true),
        SCmpkEqU32 | SCmpkLgU32 | SCmpkGtU32 | SCmpkGeU32 | SCmpkLtU32 | SCmpkLeU32 => Some(false),
        _ => None,
    }
}

fn lane_slot(register: u32, lane: u32) -> usize {
    8192 + (register * 64 + lane) as usize
}

fn mask_slot(register: u32) -> usize {
    32768 + register as usize
}

/// The register file of one program point: scalar, uniform-vector and lane
/// values, the lane-mask meaning of scalar pairs, and EXEC, VCC and SCC.
#[derive(Clone)]
struct RegisterState {
    scalars: Box<[ValueId; SCALAR_REGISTER_COUNT]>,
    vectors: Box<[ValueId; VECTOR_REGISTER_COUNT]>,
    lanes: HashMap<(u32, u32), ValueId>,
    thread_bits: HashMap<u32, ValueId>,
    exec: ValueId,
    vcc: ValueId,
    scc: ValueId,
    carry_out: ValueId,
}

impl RegisterState {
    fn new(graph: &mut ScalarValueGraph) -> Self {
        let u = graph.undefined(ValueType::U32);
        let b = graph.undefined(ValueType::Bool);
        RegisterState {
            scalars: Box::new([u; SCALAR_REGISTER_COUNT]),
            vectors: Box::new([u; VECTOR_REGISTER_COUNT]),
            lanes: HashMap::new(),
            thread_bits: HashMap::new(),
            exec: b,
            vcc: b,
            scc: b,
            carry_out: b,
        }
    }

    fn read(&self, graph: &mut ScalarValueGraph, register: u32) -> ValueId {
        if (register as usize) < SCALAR_REGISTER_COUNT {
            self.scalars[register as usize]
        } else {
            graph.undefined(ValueType::U32)
        }
    }

    fn read_vector(&self, graph: &mut ScalarValueGraph, register: u32) -> ValueId {
        if (register as usize) < VECTOR_REGISTER_COUNT {
            self.vectors[register as usize]
        } else {
            graph.undefined(ValueType::U32)
        }
    }

    fn write_scalar(&mut self, graph: &mut ScalarValueGraph, register: u32, value: ValueId) {
        if register as usize >= SCALAR_REGISTER_COUNT {
            return;
        }
        self.scalars[register as usize] = value;
        self.thread_bits.remove(&register);
        self.thread_bits.remove(&register.wrapping_sub(1));
        if register == EXEC_LOW {
            // An all-or-nothing constant keeps the mask known, so
            // S_CBRANCH_EXECZ stays decidable.
            let high = self.scalars[EXEC_HIGH as usize];
            let known = graph.is_constant(value)
                && graph.constant_u32(high) == Some(0)
                && matches!(graph.constant_u32(value), Some(0 | FULL_WAVE_MASK));
            self.exec = if known {
                let full = graph.constant_u32(value) != Some(0);
                graph.constant_bool(full)
            } else {
                graph.undefined(ValueType::Bool)
            };
        } else if register == VCC_LOW {
            self.vcc = graph.undefined(ValueType::Bool);
        }
    }

    fn write_pair(&mut self, graph: &mut ScalarValueGraph, register: u32, low: ValueId, high: ValueId) {
        self.write_scalar(graph, register, low);
        self.write_scalar(graph, register + 1, high);
    }

    /// A vector write reaches the active lanes only; other lanes keep the
    /// old value.
    fn write_vector(&mut self, graph: &mut ScalarValueGraph, register: u32, value: ValueId) {
        if register as usize >= VECTOR_REGISTER_COUNT {
            return;
        }
        let old = self.vectors[register as usize];
        self.vectors[register as usize] = if graph.is_undefined(value) { value } else { graph.select(self.exec, value, old) };
    }

    fn clear_lanes(&mut self, register: u32) {
        if !self.lanes.is_empty() {
            self.lanes.retain(|k, _| k.0 != register);
        }
    }

    fn same_as(&self, other: &RegisterState, graph: &ScalarValueGraph) -> bool {
        // Two undefined values compare equal so an undefined slot does not
        // keep the walk alive.
        let same = |a: ValueId, b: ValueId| a == b || (graph.is_undefined(a) && graph.is_undefined(b));
        for i in 0..SCALAR_REGISTER_COUNT {
            if !same(self.scalars[i], other.scalars[i]) || !same(self.vectors[i], other.vectors[i]) {
                return false;
            }
        }
        if !same(self.exec, other.exec) || !same(self.vcc, other.vcc) || !same(self.scc, other.scc) {
            return false;
        }
        if self.lanes.len() != other.lanes.len() || self.thread_bits.len() != other.thread_bits.len() {
            return false;
        }
        self.lanes.iter().all(|(k, v)| other.lanes.get(k) == Some(v))
            && self.thread_bits.iter().all(|(k, v)| other.thread_bits.get(k) == Some(v))
    }
}

struct Builder<'g> {
    graph: &'g mut ScalarValueGraph,
    phis: HashMap<(usize, usize), ValueId>,
    block_by_pc: HashMap<u32, usize>,
    entry: Vec<Option<RegisterState>>,
    exit: Vec<Option<RegisterState>>,
    block_instructions: Vec<Vec<usize>>,
    recording: bool,
    /// The join of an if or if/else on SCC: every lane took the same side,
    /// so a merged value is a select on the branch condition.
    uniform_join: Option<(ValueId, usize)>,
}

pub(super) fn run(graph: &mut ScalarValueGraph) -> Result<(), ResourcePlanError> {
    let block_count = graph.cfg.blocks.len();
    let mut block_instructions = vec![Vec::new(); block_count];
    let mut block_by_pc = HashMap::new();
    for (index, instruction) in graph.program.instructions.iter().enumerate() {
        if let Some(block) = graph.cfg.block_of(instruction.pc) {
            block_by_pc.insert(instruction.pc, block);
            block_instructions[block].push(index);
        }
    }
    graph.accesses = vec![None; graph.memory.len()];
    if block_count == 0 {
        return Ok(());
    }
    let mut builder = Builder {
        graph,
        phis: HashMap::new(),
        block_by_pc,
        entry: vec![None; block_count],
        exit: vec![None; block_count],
        block_instructions,
        recording: false,
        uniform_join: None,
    };
    builder.run()
}

impl Builder<'_> {
    fn run(&mut self) -> Result<(), ResourcePlanError> {
        let block_count = self.graph.cfg.blocks.len();
        let mut visits = vec![0u32; block_count];
        let mut queued = vec![false; block_count];
        let priorities = self.block_processing_priorities();
        let mut worklist = BinaryHeap::new();
        worklist.push(Reverse((priorities[0], 0usize)));
        queued[0] = true;
        while let Some(Reverse((_, block))) = worklist.pop() {
            queued[block] = false;
            let v = visits[block];
            visits[block] += 1;
            if v > BLOCK_VISIT_LIMIT {
                return Err(ResourcePlanError::new(format!(
                    "value graph did not converge: block {block} pc=0x{:08X}",
                    self.graph.cfg.blocks[block].start_pc
                )));
            }
            let entry = self.merge_predecessors(block);
            self.entry[block] = Some(entry.clone());
            let exit = self.transfer(block, entry);
            let changed = match &self.exit[block] {
                Some(previous) => !previous.same_as(&exit, self.graph),
                None => true,
            };
            self.exit[block] = Some(exit);
            if !changed {
                continue;
            }
            for &successor in &self.graph.cfg.successors[block].clone() {
                if !queued[successor] {
                    queued[successor] = true;
                    worklist.push(Reverse((priorities[successor], successor)));
                }
            }
        }

        self.recording = true;
        for block in 0..block_count {
            if let Some(entry) = self.entry[block].clone() {
                self.transfer(block, entry);
            }
        }
        Ok(())
    }

    /// Reverse postorder completes acyclic predecessors before their joins;
    /// loop back edges still use the convergence check.
    fn block_processing_priorities(&self) -> Vec<usize> {
        let successors = &self.graph.cfg.successors;
        let mut priorities = vec![0usize; successors.len()];
        let mut visited = vec![false; successors.len()];
        let mut pending = vec![(0usize, 0usize)];
        let mut next = successors.len();
        visited[0] = true;
        while let Some((block, next_successor)) = pending.pop() {
            if next_successor == successors[block].len() {
                next -= 1;
                priorities[block] = next;
                continue;
            }
            pending.push((block, next_successor + 1));
            let successor = successors[block][next_successor];
            if !visited[successor] {
                visited[successor] = true;
                pending.push((successor, 0));
            }
        }
        priorities
    }

    fn initial_state(&mut self) -> RegisterState {
        let mut state = RegisterState::new(self.graph);
        for index in 0..self.graph.user_data_count {
            let register = self.graph.user_data_base + index;
            if (register as usize) < SCALAR_REGISTER_COUNT {
                state.scalars[register as usize] = self.graph.user_data(register);
            }
        }
        state.scalars[VCC_LOW as usize] = self.graph.constant(0);
        state.scalars[VCC_LOW as usize + 1] = self.graph.constant(0);
        state.scalars[EXEC_LOW as usize] = self.graph.constant(FULL_WAVE_MASK);
        state.scalars[EXEC_HIGH as usize] = self.graph.constant(0);
        state.exec = self.graph.constant_bool(true);
        state.vcc = self.graph.constant_bool(false);
        state.scc = self.graph.constant_bool(false);
        state
    }

    fn merge_predecessors(&mut self, block: usize) -> RegisterState {
        let predecessors = self.graph.cfg.predecessors[block].clone();
        if block == 0 && predecessors.is_empty() {
            return self.initial_state();
        }
        let mut visited: Vec<(i64, RegisterState)> = Vec::new();
        let mut pending = false;
        for &predecessor in &predecessors {
            match &self.exit[predecessor] {
                Some(exit) => visited.push((predecessor as i64, exit.clone())),
                None => pending = true,
            }
        }
        let mut merged = RegisterState::new(self.graph);
        if block == 0 {
            // The entry block joins the initial registers with its back edges.
            let initial = self.initial_state();
            visited.insert(0, (-1, initial));
        }
        if visited.is_empty() {
            return merged;
        }
        self.uniform_join = if pending { None } else { self.find_uniform_join(block, &visited) };

        for register in 0..SCALAR_REGISTER_COUNT {
            merged.scalars[register] = self.merge_slot(block, register, &visited, pending, |s| s.scalars[register]);
        }
        for register in 0..VECTOR_REGISTER_COUNT {
            merged.vectors[register] =
                self.merge_slot(block, SCALAR_REGISTER_COUNT + register, &visited, pending, |s| s.vectors[register]);
        }
        merged.exec = self.merge_slot(block, EXEC_SLOT, &visited, pending, |s| s.exec);
        merged.vcc = self.merge_slot(block, VCC_SLOT, &visited, pending, |s| s.vcc);
        merged.scc = self.merge_slot(block, SCC_SLOT, &visited, pending, |s| s.scc);

        let mut lane_keys: Vec<(u32, u32)> =
            visited[0].1.lanes.keys().copied().filter(|k| visited[1..].iter().all(|(_, s)| s.lanes.contains_key(k))).collect();
        lane_keys.sort_unstable();
        for key in lane_keys {
            let value = self.merge_slot(block, lane_slot(key.0, key.1), &visited, pending, |s| s.lanes[&key]);
            if !self.graph.is_undefined(value) {
                merged.lanes.insert(key, value);
            }
        }
        let mut mask_keys: Vec<u32> = visited[0]
            .1
            .thread_bits
            .keys()
            .copied()
            .filter(|k| visited[1..].iter().all(|(_, s)| s.thread_bits.contains_key(k)))
            .collect();
        mask_keys.sort_unstable();
        for key in mask_keys {
            let value = self.merge_slot(block, mask_slot(key), &visited, pending, |s| s.thread_bits[&key]);
            if !self.graph.is_undefined(value) {
                merged.thread_bits.insert(key, value);
            }
        }
        merged
    }

    fn find_uniform_join(&mut self, block: usize, visited: &[(i64, RegisterState)]) -> Option<(ValueId, usize)> {
        let cfg = &self.graph.cfg;
        let predecessors = &cfg.predecessors[block];
        if visited.len() != 2 || predecessors.len() != 2 || visited.iter().any(|(b, _)| *b < 0) || cfg.loop_headers.contains(&block)
        {
            return None;
        }
        let single_predecessor = |candidate: usize| -> Option<usize> {
            (cfg.predecessors[candidate].len() == 1 && cfg.successors[candidate].len() == 1 && cfg.successors[candidate][0] == block)
                .then(|| cfg.predecessors[candidate][0])
        };
        let (first, second) = (predecessors[0], predecessors[1]);
        let branch = if single_predecessor(first) == Some(second) {
            second
        } else if single_predecessor(second) == Some(first) {
            first
        } else {
            match (single_predecessor(first), single_predecessor(second)) {
                (Some(a), Some(b)) if a == b => a,
                _ => return None,
            }
        };
        if branch == block {
            return None;
        }
        let branch_exit_scc = self.exit[branch].as_ref()?.scc;
        let terminator = self.terminator(branch)?;
        let opcode = terminator.opcode;
        if !matches!(opcode, Opcode::SCbranchScc0 | Opcode::SCbranchScc1) {
            return None;
        }
        let target = (terminator.pc as i64 + 4 + (terminator.words[0] & 0xFFFF) as u16 as i16 as i64 * 4) as u32;
        let &target_block = self.block_by_pc.get(&target)?;
        // The predecessor reached through the taken edge: the join itself
        // (triangle) is entered from the branch block, otherwise it is the
        // arm that starts at the target.
        let taken_predecessor = if target_block == block { branch } else { target_block };
        if taken_predecessor != first && taken_predecessor != second {
            return None;
        }
        let taken = if opcode == Opcode::SCbranchScc1 {
            branch_exit_scc
        } else {
            self.graph.operation(ScalarOp::LogicalNot, ValueType::U32, &[branch_exit_scc])
        };
        // A constant condition only comes from the modelled initial SCC,
        // which the hardware leaves undefined; keep that join an opaque phi.
        if self.graph.is_undefined(taken) || self.graph.is_constant(taken) {
            None
        } else {
            Some((taken, taken_predecessor))
        }
    }

    fn terminator(&self, block: usize) -> Option<Instruction> {
        let range = self.graph.cfg.blocks[block];
        self.graph.program.instructions.iter().filter(|i| i.pc >= range.start_pc && i.pc < range.end_pc).last().cloned()
    }

    /// Equal incoming values pass through; differing or still unknown ones
    /// meet in a phi owned by this block and slot. Any undefined input stays
    /// undefined.
    fn merge_slot(
        &mut self,
        block: usize,
        slot: usize,
        visited: &[(i64, RegisterState)],
        pending: bool,
        read: impl Fn(&RegisterState) -> ValueId,
    ) -> ValueId {
        let first = read(&visited[0].1);
        let mut same = true;
        for (_, state) in visited {
            let value = read(state);
            if self.graph.is_undefined(value) {
                let ty = self.graph.node(first).ty;
                return self.graph.undefined(ty);
            }
            same &= value == first;
        }
        if same && !pending {
            return first;
        }
        if let Some((taken_condition, taken_predecessor)) = self.uniform_join {
            let taken = read(&visited.iter().find(|(b, _)| *b == taken_predecessor as i64).unwrap().1);
            let other = read(&visited.iter().find(|(b, _)| *b != taken_predecessor as i64).unwrap().1);
            return self.graph.select(taken_condition, taken, other);
        }
        let phi = match self.phis.get(&(block, slot)) {
            Some(&phi) => phi,
            None => {
                let ty = self.graph.node(first).ty;
                let phi = self.graph.phi(block, ty);
                self.phis.insert((block, slot), phi);
                phi
            }
        };
        let predecessors = &self.graph.cfg.predecessors[block];
        let order: Vec<i32> = if block == 0 {
            std::iter::once(-1).chain(predecessors.iter().map(|&p| p as i32)).collect()
        } else {
            predecessors.iter().map(|&p| p as i32).collect()
        };
        let operands: Vec<ValueId> = order
            .iter()
            .map(|&o| match visited.iter().find(|(b, _)| *b == o as i64) {
                Some((_, state)) => read(state),
                None => phi,
            })
            .collect();
        self.graph.set_phi_operands(phi, order, operands);
        phi
    }

    fn transfer(&mut self, block: usize, mut state: RegisterState) -> RegisterState {
        for k in 0..self.block_instructions[block].len() {
            let index = self.block_instructions[block][k];
            let instruction = self.graph.program.instructions[index].clone();
            self.apply(&instruction, &mut state);
        }
        state
    }

    fn apply(&mut self, instruction: &Instruction, state: &mut RegisterState) {
        // Lets the graph record which instruction gave up.
        self.graph.builder_instruction = Some((instruction.pc, instruction.opcode));
        match instruction.encoding {
            // SOPK compares read their SGPR and write only SCC.
            Encoding::Sopk if sopk_compare_signed(instruction.opcode).is_some() => self.apply_scalar_compare(instruction, state),
            Encoding::Sop1 | Encoding::Sop2 | Encoding::Sopk => self.apply_scalar_alu(instruction, state),
            Encoding::Sopc => self.apply_scalar_compare(instruction, state),
            Encoding::Sopp => {
                if self.recording {
                    let condition = match instruction.opcode {
                        Opcode::SCbranchScc1 => Some(state.scc),
                        Opcode::SCbranchScc0 => Some(self.unary(ScalarOp::LogicalNot, state.scc)),
                        Opcode::SCbranchVccnz => Some(state.vcc),
                        Opcode::SCbranchVccz => Some(self.unary(ScalarOp::LogicalNot, state.vcc)),
                        Opcode::SCbranchExecnz => Some(state.exec),
                        Opcode::SCbranchExecz => Some(self.unary(ScalarOp::LogicalNot, state.exec)),
                        _ => None,
                    };
                    if let Some(c) = condition {
                        self.graph.branch_conditions.insert(instruction.pc, c);
                    }
                }
            }
            Encoding::Smem | Encoding::Smrd => self.apply_scalar_load(instruction, state),
            Encoding::Vop1 | Encoding::Vop2 | Encoding::Vop3 | Encoding::Vopc | Encoding::Vop3p => {
                self.apply_vector_alu(instruction, state)
            }
            _ => self.apply_memory(instruction, state),
        }
    }

    // ---- value helpers ----

    fn c(&mut self, value: u32) -> ValueId {
        self.graph.constant(value)
    }

    fn unary(&mut self, op: ScalarOp, value: ValueId) -> ValueId {
        self.graph.operation(op, ValueType::U32, &[value])
    }

    fn binary(&mut self, op: ScalarOp, left: ValueId, right: ValueId) -> ValueId {
        self.graph.operation(op, ValueType::U32, &[left, right])
    }

    fn binary64(&mut self, op: ScalarOp, left: ValueId, right: ValueId) -> ValueId {
        self.graph.operation(op, ValueType::U64, &[left, right])
    }

    fn boolean(&mut self, op: ScalarOp, left: ValueId, right: ValueId) -> ValueId {
        self.graph.operation(op, ValueType::Bool, &[left, right])
    }

    fn not(&mut self, value: ValueId) -> ValueId {
        self.graph.operation(ScalarOp::LogicalNot, ValueType::Bool, &[value])
    }

    fn not_zero(&mut self, value: ValueId) -> ValueId {
        let zero = self.c(0);
        self.boolean(ScalarOp::INotEqual32, value, zero)
    }

    fn construct(&mut self, low: ValueId, high: ValueId) -> ValueId {
        self.graph.operation(ScalarOp::Construct64, ValueType::U64, &[low, high])
    }

    fn extract(&mut self, pair: ValueId, index: u32) -> ValueId {
        let i = self.c(index);
        self.graph.operation(ScalarOp::Extract64, ValueType::U32, &[pair, i])
    }

    fn bfe(&mut self, op: ScalarOp, value: ValueId, offset: ValueId, count: ValueId) -> ValueId {
        self.graph.operation(op, ValueType::U32, &[value, offset, count])
    }

    fn bfe_const(&mut self, value: ValueId, offset: u32, count: u32) -> ValueId {
        let (o, c) = (self.c(offset), self.c(count));
        self.bfe(ScalarOp::BitFieldUExtract, value, o, c)
    }

    fn right_mask32(&mut self, count: ValueId) -> ValueId {
        let (zero, ones) = (self.c(0), self.c(u32::MAX));
        self.graph.operation(ScalarOp::BitFieldInsert, ValueType::U32, &[zero, ones, zero, count])
    }

    fn right_mask64(&mut self, count: ValueId) -> ValueId {
        let c32 = self.c(32);
        let below32 = self.boolean(ScalarOp::ULessThan32, count, c32);
        let above32 = self.boolean(ScalarOp::UGreaterThan32, count, c32);
        let low_count = self.graph.select(below32, count, c32);
        let excess = self.binary(ScalarOp::ISub32, count, c32);
        let zero = self.c(0);
        let high_count = self.graph.select(above32, excess, zero);
        let low = self.right_mask32(low_count);
        let high = self.right_mask32(high_count);
        self.construct(low, high)
    }

    fn select_bit(&mut self, condition: ValueId) -> ValueId {
        let (one, zero) = (self.c(1), self.c(0));
        self.graph.select(condition, one, zero)
    }

    /// `dst` pair = shader base + offset of the next instruction (S_GETPC_B64
    /// and the return address a call writes to its link).
    fn write_next_address(&mut self, instruction: &Instruction, dst: u32, state: &mut RegisterState) {
        let base = self.graph.shader_base();
        let offset = self.graph.constant64(instruction.program_offset().wrapping_add(instruction.words.len() as u64 * 4));
        let address = self.graph.operation(ScalarOp::IAdd64, ValueType::U64, &[base, offset]);
        let (low, high) = (self.extract(address, 0), self.extract(address, 1));
        state.write_pair(self.graph, dst, low, high);
    }

    // ---- scalar ALU ----

    fn apply_scalar_alu(&mut self, instruction: &Instruction, state: &mut RegisterState) {
        let opcode = instruction.opcode;
        if instruction.destinations.len() != 1
            || instruction.destinations[0].kind != OperandKind::ScalarRegister
            || instruction.destinations[0].value as usize >= SCALAR_REGISTER_COUNT
        {
            return;
        }
        let dst = instruction.destinations[0].value;
        let src = |i: usize| instruction.sources[i];
        use Opcode::*;
        match opcode {
            SMovB32 => {
                let v = self.read(src(0), state);
                state.write_scalar(self.graph, dst, v);
                return;
            }
            SBitreplicateB64B32 => {
                let v = self.read(src(0), state);
                state.write_scalar(self.graph, dst, v);
                state.write_scalar(self.graph, dst + 1, v);
                return;
            }
            SMovkI32 => {
                let v = self.c(src(0).value as u16 as i16 as i32 as u32);
                state.write_scalar(self.graph, dst, v);
                return;
            }
            SAddkI32 | SMulkI32 => {
                let imm = src(0).value as u16 as i16 as i32 as u32;
                let k = self.c(imm);
                let old = state.scalars[dst as usize];
                let op = if opcode == SAddkI32 { ScalarOp::IAdd32 } else { ScalarOp::IMul32 };
                let v = self.binary(op, old, k);
                state.write_scalar(self.graph, dst, v);
                if opcode == SAddkI32 {
                    // SCC is the signed overflow, as for s_add_i32.
                    let c31 = self.c(31);
                    let old_sign = self.binary(ScalarOp::ShiftRightLogical32, old, c31);
                    let k_sign = self.c(imm >> 31);
                    let out_sign = self.binary(ScalarOp::ShiftRightLogical32, v, c31);
                    let same = self.boolean(ScalarOp::IEqual32, old_sign, k_sign);
                    let flipped = self.boolean(ScalarOp::INotEqual32, old_sign, out_sign);
                    state.scc = self.boolean(ScalarOp::LogicalAnd, same, flipped);
                }
                return;
            }
            SGetpcB64 | SCallB64 => {
                self.write_next_address(instruction, dst, state);
                return;
            }
            SSwappcB64 if crate::cfg::calls::is_call(instruction) => {
                self.write_next_address(instruction, dst, state);
                return;
            }
            SSetpcB64 | SSwappcB64 => return,
            SMovB64 => {
                let (low, high) = self.read_pair(src(0), state);
                let mask = self.mask_of(src(0), state);
                self.write_mask_pair(state, dst, low, high, mask);
                return;
            }
            SCselectB32 => {
                let (a, b) = (self.read(src(0), state), self.read(src(1), state));
                let v = self.graph.select(state.scc, a, b);
                state.write_scalar(self.graph, dst, v);
                return;
            }
            SCselectB64 => {
                let (ll, lh) = self.read_pair(src(0), state);
                let (rl, rh) = self.read_pair(src(1), state);
                let (lm, rm) = (self.mask_of(src(0), state), self.mask_of(src(1), state));
                let mask = self.graph.select(state.scc, lm, rm);
                let low = self.graph.select(state.scc, ll, rl);
                let high = self.graph.select(state.scc, lh, rh);
                self.write_mask_pair(state, dst, low, high, mask);
                return;
            }
            _ => {}
        }
        if opcode.is_saveexec() {
            self.apply_save_exec(instruction, state, dst);
            return;
        }
        if matches!(opcode, SAndB64 | SOrB64 | SXorB64 | SAndn2B64 | SOrn2B64 | SNandB64 | SNorB64 | SXnorB64 | SNotB64) {
            self.apply_mask_pair_operation(instruction, state, dst);
            return;
        }
        if matches!(opcode, SLshlB64 | SLshrB64 | SAshrI64) {
            let (low, high) = self.read_pair(src(0), state);
            let amount = self.read(src(1), state);
            let c63 = self.c(63);
            let count = self.binary(ScalarOp::And32, amount, c63);
            let op = match opcode {
                SLshlB64 => ScalarOp::ShiftLeft64,
                SLshrB64 => ScalarOp::ShiftRightLogical64,
                _ => ScalarOp::ShiftRightArithmetic64,
            };
            let pair = self.construct(low, high);
            let result = self.graph.operation(op, ValueType::U64, &[pair, count]);
            let (l, h) = (self.extract(result, 0), self.extract(result, 1));
            state.write_pair(self.graph, dst, l, h);
            let zero = self.graph.constant64(0);
            state.scc = self.boolean(ScalarOp::INotEqual64, result, zero);
            return;
        }
        if opcode == SBfeU64 {
            let (low, high) = self.read_pair(src(0), state);
            let field = self.read(src(1), state);
            let offset = self.bfe_const(field, 0, 6);
            let raw_count = self.bfe_const(field, 16, 7);
            let c64 = self.c(64);
            let room = self.binary(ScalarOp::ISub32, c64, offset);
            let count = self.binary(ScalarOp::UMin32, raw_count, room);
            let pair = self.construct(low, high);
            let shifted = self.graph.operation(ScalarOp::ShiftRightLogical64, ValueType::U64, &[pair, offset]);
            let mask = self.right_mask64(count);
            let result = self.graph.operation(ScalarOp::And64, ValueType::U64, &[shifted, mask]);
            let (l, h) = (self.extract(result, 0), self.extract(result, 1));
            state.write_pair(self.graph, dst, l, h);
            let zero = self.graph.constant64(0);
            state.scc = self.boolean(ScalarOp::INotEqual64, result, zero);
            return;
        }
        if opcode == SBfmB64 {
            let c63 = self.c(63);
            let (a, b) = (self.read(src(0), state), self.read(src(1), state));
            let count = self.binary(ScalarOp::And32, a, c63);
            let offset = self.binary(ScalarOp::And32, b, c63);
            let mask = self.right_mask64(count);
            let result = self.graph.operation(ScalarOp::ShiftLeft64, ValueType::U64, &[mask, offset]);
            let (l, h) = (self.extract(result, 0), self.extract(result, 1));
            state.write_pair(self.graph, dst, l, h);
            return;
        }
        if matches!(opcode, SQuadmaskB32 | SQuadmaskB64) {
            let wide = opcode == SQuadmaskB64;
            let (low, high) = self.read_pair(src(0), state);
            let mut result = self.c(0);
            for quad in 0..(if wide { 16 } else { 8 }) {
                let shift = self.c((quad % 8) * 4);
                let word = if quad < 8 { low } else { high };
                let shifted = self.binary(ScalarOp::ShiftRightLogical32, word, shift);
                let c15 = self.c(15);
                let nibble = self.binary(ScalarOp::And32, shifted, c15);
                let nz = self.not_zero(nibble);
                let (bit, zero) = (self.c(1 << quad), self.c(0));
                let sel = self.graph.select(nz, bit, zero);
                result = self.binary(ScalarOp::Or32, result, sel);
            }
            if wide {
                let zero = self.c(0);
                state.write_pair(self.graph, dst, result, zero);
            } else {
                state.write_scalar(self.graph, dst, result);
            }
            state.scc = self.not_zero(result);
            return;
        }
        if matches!(opcode, SBcnt1I32B64 | SFF1I32B64 | SWqmB64 | SBfeI64) {
            let (low, high) = self.read_pair(src(0), state);
            match opcode {
                SWqmB64 => {
                    let ql = self.unary(ScalarOp::QuadMask32, low);
                    let qh = self.unary(ScalarOp::QuadMask32, high);
                    state.write_pair(self.graph, dst, ql, qh);
                    let both = self.binary(ScalarOp::Or32, ql, qh);
                    state.scc = self.not_zero(both);
                }
                SBcnt1I32B64 => {
                    let a = self.unary(ScalarOp::BitCount32, low);
                    let b = self.unary(ScalarOp::BitCount32, high);
                    let v = self.binary(ScalarOp::IAdd32, a, b);
                    state.write_scalar(self.graph, dst, v);
                    state.scc = self.not_zero(state.scalars[dst as usize]);
                }
                SFF1I32B64 => {
                    let low_lsb = self.unary(ScalarOp::FindLowestBit32, low);
                    let high_scan = self.unary(ScalarOp::FindLowestBit32, high);
                    let c32 = self.c(32);
                    let high_lsb = self.binary(ScalarOp::IAdd32, high_scan, c32);
                    let low_nz = self.not_zero(low);
                    let high_nz = self.not_zero(high);
                    let none = self.c(u32::MAX);
                    let inner = self.graph.select(high_nz, high_lsb, none);
                    let result = self.graph.select(low_nz, low_lsb, inner);
                    state.write_scalar(self.graph, dst, result);
                }
                _ => {
                    let u = self.graph.undefined(ValueType::U32);
                    state.write_pair(self.graph, dst, u, u);
                    state.scc = self.graph.undefined(ValueType::Bool);
                }
            }
            return;
        }
        if instruction.sources.is_empty() {
            let u = self.graph.undefined(ValueType::U32);
            state.write_scalar(self.graph, dst, u);
            return;
        }
        let left = self.read(src(0), state);
        match opcode {
            SNotB32 => {
                let v = self.unary(ScalarOp::Not32, left);
                self.write_with_scc(state, dst, v);
                return;
            }
            SAbsI32 => {
                let (zero, c31) = (self.c(0), self.c(31));
                let sign_bit = self.binary(ScalarOp::ShiftRightLogical32, left, c31);
                let sign = self.binary(ScalarOp::ISub32, zero, sign_bit);
                let flipped = self.binary(ScalarOp::Xor32, left, sign);
                let magnitude = self.binary(ScalarOp::ISub32, flipped, sign);
                self.write_with_scc(state, dst, magnitude);
                return;
            }
            SWqmB32 => {
                let v = self.unary(ScalarOp::QuadMask32, left);
                self.write_with_scc(state, dst, v);
                return;
            }
            SBrevB32 => {
                let v = self.unary(ScalarOp::BitReverse32, left);
                state.write_scalar(self.graph, dst, v);
                return;
            }
            SBcnt1I32B32 => {
                let v = self.unary(ScalarOp::BitCount32, left);
                self.write_with_scc(state, dst, v);
                return;
            }
            SFF1I32B32 => {
                let v = self.graph.find_lowest_set_bit(left, instruction.pc);
                state.write_scalar(self.graph, dst, v);
                return;
            }
            SBitset0B32 | SBitset1B32 => {
                let (one, c31) = (self.c(1), self.c(31));
                let index = self.binary(ScalarOp::And32, left, c31);
                let bit = self.binary(ScalarOp::ShiftLeft32, one, index);
                let old = state.scalars[dst as usize];
                let v = if opcode == SBitset0B32 {
                    let inverted = self.unary(ScalarOp::Not32, bit);
                    self.binary(ScalarOp::And32, old, inverted)
                } else {
                    self.binary(ScalarOp::Or32, old, bit)
                };
                state.write_scalar(self.graph, dst, v);
                return;
            }
            _ => {}
        }
        if instruction.sources.len() < 2 {
            let u = self.graph.undefined(ValueType::U32);
            state.write_scalar(self.graph, dst, u);
            return;
        }
        let right = self.read(src(1), state);
        match opcode {
            SAddU32 => {
                let sum = self.binary64(ScalarOp::AddCarry32, left, right);
                let (l, carry) = (self.extract(sum, 0), self.extract(sum, 1));
                state.write_scalar(self.graph, dst, l);
                state.scc = self.not_zero(carry);
            }
            SAddcU32 => {
                let carry_in = self.select_bit(state.scc);
                let first = self.binary64(ScalarOp::AddCarry32, left, right);
                let first_low = self.extract(first, 0);
                let second = self.binary64(ScalarOp::AddCarry32, first_low, carry_in);
                let result = self.extract(second, 0);
                state.write_scalar(self.graph, dst, result);
                let (c1, c2) = (self.extract(first, 1), self.extract(second, 1));
                let carry = self.binary(ScalarOp::Or32, c1, c2);
                state.scc = self.not_zero(carry);
            }
            SSubU32 => {
                let v = self.binary(ScalarOp::ISub32, left, right);
                state.write_scalar(self.graph, dst, v);
                state.scc = self.boolean(ScalarOp::UGreaterThan32, right, left);
            }
            SSubbU32 => {
                let borrow_in = self.select_bit(state.scc);
                let partial = self.binary(ScalarOp::ISub32, left, right);
                let v = self.binary(ScalarOp::ISub32, partial, borrow_in);
                state.write_scalar(self.graph, dst, v);
                let a = self.boolean(ScalarOp::UGreaterThan32, right, left);
                let b = self.boolean(ScalarOp::UGreaterThan32, borrow_in, partial);
                state.scc = self.boolean(ScalarOp::LogicalOr, a, b);
            }
            SAddI32 | SSubI32 => {
                let subtract = opcode == SSubI32;
                let result = self.binary(if subtract { ScalarOp::ISub32 } else { ScalarOp::IAdd32 }, left, right);
                let c31 = self.c(31);
                let left_sign = self.binary(ScalarOp::ShiftRightLogical32, left, c31);
                let right_sign = self.binary(ScalarOp::ShiftRightLogical32, right, c31);
                let out_sign = self.binary(ScalarOp::ShiftRightLogical32, result, c31);
                let inputs = self.boolean(
                    if subtract { ScalarOp::INotEqual32 } else { ScalarOp::IEqual32 },
                    left_sign,
                    right_sign,
                );
                state.write_scalar(self.graph, dst, result);
                let flipped = self.boolean(ScalarOp::INotEqual32, left_sign, out_sign);
                state.scc = self.boolean(ScalarOp::LogicalAnd, inputs, flipped);
            }
            SMinI32 | SMinU32 | SMaxI32 | SMaxU32 => {
                let (op, cmp) = match opcode {
                    SMinI32 => (ScalarOp::SMin32, ScalarOp::SLessThan32),
                    SMinU32 => (ScalarOp::UMin32, ScalarOp::ULessThan32),
                    SMaxI32 => (ScalarOp::SMax32, ScalarOp::SGreaterThan32),
                    _ => (ScalarOp::UMax32, ScalarOp::UGreaterThan32),
                };
                let v = self.binary(op, left, right);
                state.write_scalar(self.graph, dst, v);
                state.scc = self.boolean(cmp, left, right);
            }
            SAndB32 | SOrB32 | SXorB32 => {
                let op = match opcode {
                    SAndB32 => ScalarOp::And32,
                    SOrB32 => ScalarOp::Or32,
                    _ => ScalarOp::Xor32,
                };
                let v = self.binary(op, left, right);
                self.write_with_scc(state, dst, v);
            }
            SAndn2B32 | SOrn2B32 => {
                let inverted = self.unary(ScalarOp::Not32, right);
                let v = self.binary(if opcode == SAndn2B32 { ScalarOp::And32 } else { ScalarOp::Or32 }, left, inverted);
                self.write_with_scc(state, dst, v);
            }
            SNandB32 | SNorB32 | SXnorB32 => {
                let op = match opcode {
                    SNandB32 => ScalarOp::And32,
                    SNorB32 => ScalarOp::Or32,
                    _ => ScalarOp::Xor32,
                };
                let inner = self.binary(op, left, right);
                let v = self.unary(ScalarOp::Not32, inner);
                self.write_with_scc(state, dst, v);
            }
            SLshlB32 | SLshrB32 | SAshrI32 => {
                let op = match opcode {
                    SLshlB32 => ScalarOp::ShiftLeft32,
                    SLshrB32 => ScalarOp::ShiftRightLogical32,
                    _ => ScalarOp::ShiftRightArithmetic32,
                };
                let c31 = self.c(31);
                let count = self.binary(ScalarOp::And32, right, c31);
                let v = self.binary(op, left, count);
                self.write_with_scc(state, dst, v);
            }
            SBfmB32 => {
                let c31 = self.c(31);
                let count = self.binary(ScalarOp::And32, left, c31);
                let offset = self.binary(ScalarOp::And32, right, c31);
                // ((1 << count) - 1) << offset, truncated to 32 bits (a field
                // past bit 31 loses its top bits).
                let mask = self.right_mask32(count);
                let v = self.binary(ScalarOp::ShiftLeft32, mask, offset);
                state.write_scalar(self.graph, dst, v);
            }
            SMulI32 => {
                let v = self.binary(ScalarOp::IMul32, left, right);
                state.write_scalar(self.graph, dst, v);
            }
            SMulHiU32 => {
                let v = self.binary(ScalarOp::UMulHi32, left, right);
                state.write_scalar(self.graph, dst, v);
            }
            SMulHiI32 => {
                let v = self.binary(ScalarOp::SMulHi32, left, right);
                state.write_scalar(self.graph, dst, v);
            }
            SBfeU32 | SBfeI32 => {
                let offset = self.bfe_const(right, 0, 5);
                let raw_count = self.bfe_const(right, 16, 7);
                let c32 = self.c(32);
                let room = self.binary(ScalarOp::ISub32, c32, offset);
                let count = self.binary(ScalarOp::UMin32, raw_count, room);
                let op = if opcode == SBfeU32 { ScalarOp::BitFieldUExtract } else { ScalarOp::BitFieldSExtract };
                let v = self.bfe(op, left, offset, count);
                self.write_with_scc(state, dst, v);
            }
            SAbsdiffI32 => {
                let d = self.binary(ScalarOp::ISub32, left, right);
                let v = self.unary(ScalarOp::IAbs32, d);
                self.write_with_scc(state, dst, v);
            }
            SLshl1AddU32 | SLshl2AddU32 | SLshl3AddU32 | SLshl4AddU32 => {
                let amount = match opcode {
                    SLshl1AddU32 => 1,
                    SLshl2AddU32 => 2,
                    SLshl3AddU32 => 3,
                    _ => 4,
                };
                let a = self.c(amount);
                let shifted = self.binary(ScalarOp::ShiftLeft32, left, a);
                let result = self.binary(ScalarOp::IAdd32, shifted, right);
                let add_carry = self.boolean(ScalarOp::ULessThan32, result, shifted);
                let back = self.c(32 - amount);
                let shifted_out = self.binary(ScalarOp::ShiftRightLogical32, left, back);
                state.write_scalar(self.graph, dst, result);
                let nz = self.not_zero(shifted_out);
                state.scc = self.boolean(ScalarOp::LogicalOr, add_carry, nz);
            }
            SPackLlB32B16 | SPackLhB32B16 | SPackHhB32B16 => {
                let c16 = self.c(16);
                let low = if opcode == SPackHhB32B16 { self.binary(ScalarOp::ShiftRightLogical32, left, c16) } else { left };
                let high = if opcode == SPackLlB32B16 { right } else { self.binary(ScalarOp::ShiftRightLogical32, right, c16) };
                let ffff = self.c(0xFFFF);
                let low_bits = self.binary(ScalarOp::And32, low, ffff);
                let high_masked = self.binary(ScalarOp::And32, high, ffff);
                let high_bits = self.binary(ScalarOp::ShiftLeft32, high_masked, c16);
                let v = self.binary(ScalarOp::Or32, low_bits, high_bits);
                state.write_scalar(self.graph, dst, v);
            }
            _ => {
                let u = self.graph.undefined(ValueType::U32);
                state.write_scalar(self.graph, dst, u);
                if opcode.name().ends_with("64") {
                    state.write_scalar(self.graph, dst + 1, u);
                }
                state.scc = self.graph.undefined(ValueType::Bool);
            }
        }
    }

    fn write_with_scc(&mut self, state: &mut RegisterState, dst: u32, value: ValueId) {
        state.write_scalar(self.graph, dst, value);
        state.scc = self.not_zero(value);
    }

    /// A lane mask a vector instruction writes to an SGPR: in wave32 it
    /// fills that register alone and the next keeps its value (often live:
    /// a pointer's high half, or VCC_HI used as an ordinary SGPR).
    fn write_vector_mask(&mut self, state: &mut RegisterState, dst: u32, low: ValueId, mask: ValueId) {
        if self.graph.wave_size != 32 {
            let zero = self.c(0);
            self.write_mask_pair(state, dst, low, zero, mask);
            return;
        }
        state.write_scalar(self.graph, dst, low);
        match dst {
            EXEC_LOW => state.exec = mask,
            VCC_LOW => state.vcc = mask,
            _ => {
                if !self.graph.is_undefined(mask) {
                    state.thread_bits.insert(dst, mask);
                }
            }
        }
    }

    /// Write a dword pair; a pair that is a lane mask also keeps its mask
    /// value, and a write to EXEC or VCC updates the active mask.
    fn write_mask_pair(&mut self, state: &mut RegisterState, dst: u32, low: ValueId, high: ValueId, mask: ValueId) {
        state.write_pair(self.graph, dst, low, high);
        match dst {
            EXEC_LOW => state.exec = mask,
            VCC_LOW => state.vcc = mask,
            _ => {
                if !self.graph.is_undefined(mask) {
                    state.thread_bits.insert(dst, mask);
                }
            }
        }
    }

    fn apply_save_exec(&mut self, instruction: &Instruction, state: &mut RegisterState, dst: u32) {
        let opcode = instruction.opcode;
        let wide = opcode.is_saveexec_b64();
        let old_exec = state.exec;
        let old_low = state.scalars[EXEC_LOW as usize];
        let old_high = state.scalars[EXEC_HIGH as usize];
        let source = self.mask_of(instruction.sources[0], state);
        let (source_low, source_high) = if wide {
            self.read_pair(instruction.sources[0], state)
        } else {
            (self.read(instruction.sources[0], state), self.c(0))
        };
        let full = opcode.name();
        let name = full[1..full.find("Saveexec").unwrap()].to_string();
        let combine = |this: &mut Self, old_mask: ValueId, source_mask: ValueId, old_bits: ValueId, source_bits: ValueId| {
            use ScalarOp as O;
            match name.as_str() {
                "And" => (this.binary(O::And32, old_bits, source_bits), this.boolean(O::LogicalAnd, old_mask, source_mask)),
                "Or" => (this.binary(O::Or32, old_bits, source_bits), this.boolean(O::LogicalOr, old_mask, source_mask)),
                "Xor" => (this.binary(O::Xor32, old_bits, source_bits), this.boolean(O::LogicalXor, old_mask, source_mask)),
                "Andn1" => {
                    let n = this.unary(O::Not32, source_bits);
                    let raw = this.binary(O::And32, n, old_bits);
                    let nm = this.not(source_mask);
                    (raw, this.boolean(O::LogicalAnd, nm, old_mask))
                }
                "Andn2" => {
                    let n = this.unary(O::Not32, old_bits);
                    let raw = this.binary(O::And32, source_bits, n);
                    let nm = this.not(old_mask);
                    (raw, this.boolean(O::LogicalAnd, source_mask, nm))
                }
                "Orn1" => {
                    let n = this.unary(O::Not32, source_bits);
                    let raw = this.binary(O::Or32, n, old_bits);
                    let nm = this.not(source_mask);
                    (raw, this.boolean(O::LogicalOr, nm, old_mask))
                }
                "Orn2" => {
                    let n = this.unary(O::Not32, old_bits);
                    let raw = this.binary(O::Or32, source_bits, n);
                    let nm = this.not(old_mask);
                    (raw, this.boolean(O::LogicalOr, source_mask, nm))
                }
                "Nand" => {
                    let a = this.binary(O::And32, source_bits, old_bits);
                    let raw = this.unary(O::Not32, a);
                    let m = this.boolean(O::LogicalAnd, source_mask, old_mask);
                    (raw, this.not(m))
                }
                "Nor" => {
                    let a = this.binary(O::Or32, source_bits, old_bits);
                    let raw = this.unary(O::Not32, a);
                    let m = this.boolean(O::LogicalOr, source_mask, old_mask);
                    (raw, this.not(m))
                }
                _ => {
                    let a = this.binary(O::Xor32, old_bits, source_bits);
                    let raw = this.unary(O::Not32, a);
                    let m = this.boolean(O::LogicalXor, old_mask, source_mask);
                    (raw, this.not(m))
                }
            }
        };
        let (new_low, new_exec) = combine(self, old_exec, source, old_low, source_low);
        let (mut new_high, _) = combine(self, old_exec, source, old_high, source_high);
        if wide {
            state.write_pair(self.graph, dst, old_low, old_high);
        } else {
            state.write_scalar(self.graph, dst, old_low);
            new_high = self.c(0);
        }
        state.thread_bits.insert(dst, old_exec);
        state.write_pair(self.graph, EXEC_LOW, new_low, new_high);
        state.exec = new_exec;
        state.scc = new_exec;
    }

    fn apply_mask_pair_operation(&mut self, instruction: &Instruction, state: &mut RegisterState, dst: u32) {
        use Opcode::*;
        let opcode = instruction.opcode;
        let unary = opcode == SNotB64;
        let (left_low, left_high) = self.read_pair(instruction.sources[0], state);
        let left_mask = self.mask_of(instruction.sources[0], state);
        let (right_low, right_high, right_mask) = if unary {
            let zero = self.c(0);
            (zero, zero, self.graph.constant_bool(false))
        } else {
            let (l, h) = self.read_pair(instruction.sources[1], state);
            (l, h, self.mask_of(instruction.sources[1], state))
        };
        let bit_op = match opcode {
            SAndB64 | SAndn2B64 | SNandB64 | SNotB64 => ScalarOp::And32,
            SOrB64 | SOrn2B64 | SNorB64 => ScalarOp::Or32,
            _ => ScalarOp::Xor32,
        };
        let logical_op = match bit_op {
            ScalarOp::And32 => ScalarOp::LogicalAnd,
            ScalarOp::Or32 => ScalarOp::LogicalOr,
            _ => ScalarOp::LogicalXor,
        };
        let negate_right = matches!(opcode, SAndn2B64 | SOrn2B64);
        let negate_result = matches!(opcode, SNandB64 | SNorB64 | SXnorB64);
        let combine = |this: &mut Self, left: ValueId, right: ValueId| {
            if unary {
                return this.unary(ScalarOp::Not32, left);
            }
            let r = if negate_right { this.unary(ScalarOp::Not32, right) } else { right };
            let v = this.binary(bit_op, left, r);
            if negate_result {
                this.unary(ScalarOp::Not32, v)
            } else {
                v
            }
        };
        let mask = if unary {
            self.not(left_mask)
        } else {
            let r = if negate_right { self.not(right_mask) } else { right_mask };
            let m = self.boolean(logical_op, left_mask, r);
            if negate_result {
                self.not(m)
            } else {
                m
            }
        };
        let low = combine(self, left_low, right_low);
        let high = combine(self, left_high, right_high);
        self.write_mask_pair(state, dst, low, high, mask);
        state.scc = if dst == EXEC_LOW || dst == VCC_LOW {
            mask
        } else {
            let both = self.binary(ScalarOp::Or32, low, high);
            self.not_zero(both)
        };
    }

    fn apply_scalar_compare(&mut self, instruction: &Instruction, state: &mut RegisterState) {
        use Opcode::*;
        let opcode = instruction.opcode;
        if let Some(signed) = sopk_compare_signed(opcode) {
            // s_cmpk_*: the SGPR in the SDST field against the 16-bit
            // immediate, sign-extended for the _i32 forms and zero-extended
            // for the _u32 forms.
            let (Some(reg), Some(imm)) = (instruction.destinations.first(), instruction.sources.first()) else {
                state.scc = self.graph.undefined(ValueType::Bool);
                return;
            };
            let left = self.read(*reg, state);
            let imm = imm.value & 0xFFFF;
            let right = self.c(if signed { imm as u16 as i16 as i32 as u32 } else { imm });
            state.scc = self.compare32(opcode, left, right);
            return;
        }
        if instruction.sources.len() != 2 {
            state.scc = self.graph.undefined(ValueType::Bool);
            return;
        }
        if matches!(opcode, SCmpEqU64 | SCmpLgU64) {
            let (ll, lh) = self.read_pair(instruction.sources[0], state);
            let (rl, rh) = self.read_pair(instruction.sources[1], state);
            let (l, r) = (self.construct(ll, lh), self.construct(rl, rh));
            state.scc = self.boolean(if opcode == SCmpEqU64 { ScalarOp::IEqual64 } else { ScalarOp::INotEqual64 }, l, r);
            return;
        }
        let left = self.read(instruction.sources[0], state);
        let right = self.read(instruction.sources[1], state);
        if matches!(opcode, SBitcmp0B32 | SBitcmp1B32) {
            let c31 = self.c(31);
            let offset = self.binary(ScalarOp::And32, right, c31);
            let one = self.c(1);
            let bit = self.bfe(ScalarOp::BitFieldUExtract, left, offset, one);
            let want = self.c((opcode == SBitcmp1B32) as u32);
            state.scc = self.boolean(ScalarOp::IEqual32, bit, want);
            return;
        }
        if matches!(opcode, SBitcmp0B64 | SBitcmp1B64) {
            let (low, high) = self.read_pair(instruction.sources[0], state);
            let (c63, c31, one, c32) = (self.c(63), self.c(31), self.c(1), self.c(32));
            let offset = self.binary(ScalarOp::And32, right, c63);
            let word_offset = self.binary(ScalarOp::And32, offset, c31);
            let low_bit = self.bfe(ScalarOp::BitFieldUExtract, low, word_offset, one);
            let high_bit = self.bfe(ScalarOp::BitFieldUExtract, high, word_offset, one);
            let in_low = self.boolean(ScalarOp::ULessThan32, offset, c32);
            let bit = self.graph.select(in_low, low_bit, high_bit);
            let want = self.c((opcode == SBitcmp1B64) as u32);
            state.scc = self.boolean(ScalarOp::IEqual32, bit, want);
            return;
        }
        state.scc = self.compare32(opcode, left, right);
    }

    /// SCC of a 32-bit s_cmp / s_cmpk predicate.
    fn compare32(&mut self, opcode: Opcode, left: ValueId, right: ValueId) -> ValueId {
        use Opcode::*;
        let op = match opcode {
            SCmpEqI32 | SCmpEqU32 | SCmpkEqI32 | SCmpkEqU32 => Some(ScalarOp::IEqual32),
            SCmpLgI32 | SCmpLgU32 | SCmpkLgI32 | SCmpkLgU32 => Some(ScalarOp::INotEqual32),
            SCmpGtU32 | SCmpkGtU32 => Some(ScalarOp::UGreaterThan32),
            SCmpGeU32 | SCmpkGeU32 => Some(ScalarOp::UGreaterThanEqual32),
            SCmpLtU32 | SCmpkLtU32 => Some(ScalarOp::ULessThan32),
            SCmpLeU32 | SCmpkLeU32 => Some(ScalarOp::ULessThanEqual32),
            SCmpGtI32 | SCmpkGtI32 => Some(ScalarOp::SGreaterThan32),
            SCmpGeI32 | SCmpkGeI32 => Some(ScalarOp::SGreaterThanEqual32),
            SCmpLtI32 | SCmpkLtI32 => Some(ScalarOp::SLessThan32),
            SCmpLeI32 | SCmpkLeI32 => Some(ScalarOp::SLessThanEqual32),
            _ => None,
        };
        match op {
            Some(op) => self.boolean(op, left, right),
            None => self.graph.undefined(ValueType::Bool),
        }
    }

    // ---- scalar loads ----

    fn apply_scalar_load(&mut self, instruction: &Instruction, state: &mut RegisterState) {
        let control = match &instruction.control {
            Some(Control::ScalarMemory(c))
                if !instruction.sources.is_empty() && instruction.sources[0].kind == OperandKind::ScalarRegister =>
            {
                c.clone()
            }
            _ => {
                for d in &instruction.destinations {
                    if d.kind == OperandKind::ScalarRegister {
                        let u = self.graph.undefined(ValueType::U32);
                        state.write_scalar(self.graph, d.value, u);
                    }
                }
                return;
            }
        };
        let is_buffer = is_scalar_buffer_load(instruction.opcode);
        let base = instruction.sources[0].value;
        let handle = if is_buffer {
            let dwords: Vec<ValueId> = (0..4).map(|i| state.read(self.graph, base + i)).collect();
            self.graph.handle(ValueKind::BufferHandle, &dwords)
        } else {
            let dwords = [state.read(self.graph, base), state.read(self.graph, base + 1)];
            self.graph.handle(ValueKind::AddressHandle, &dwords)
        };
        let offset = match control.dynamic_offset_register {
            Some(r) => state.read(self.graph, r),
            None => self.c(0),
        };
        for component in 0..control.destination_count {
            let Some(memory_index) = self.graph.memory.index_of(instruction.pc, component) else { continue };
            let kind = if is_buffer { ValueKind::ScalarBufferWord } else { ValueKind::ScalarAddressWord };
            let read = self.graph.memory_read(kind, handle, offset, memory_index);
            if let Some(d) = instruction.destinations.get(component as usize) {
                if d.kind == OperandKind::ScalarRegister {
                    state.write_scalar(self.graph, d.value, read);
                }
            }
            if self.recording {
                self.graph.accesses[memory_index] =
                    Some(MemoryAccessBinding { handle: Some(handle), read: Some(read), ..Default::default() });
            }
        }
    }

    // ---- vector ALU (uniform domain) ----

    fn apply_vector_alu(&mut self, instruction: &Instruction, state: &mut RegisterState) {
        use Opcode::*;
        let opcode = instruction.opcode;
        match opcode {
            VWritelaneB32 => return self.apply_write_lane(instruction, state),
            VReadlaneB32 => return self.apply_read_lane(instruction, state),
            VReadfirstlaneB32 => {
                if instruction.destinations.len() == 1
                    && instruction.destinations[0].kind == OperandKind::ScalarRegister
                    && instruction.sources.len() == 1
                {
                    let source = self.read_vector_operand(instruction.sources[0], state);
                    // Keep the selector shape even when its lane value is
                    // GPU-dependent.
                    let v = self.graph.first_lane(source, state.exec, instruction.pc);
                    state.write_scalar(self.graph, instruction.destinations[0].value, v);
                }
                return;
            }
            _ => {}
        }
        // Any other write to a vector register replaces its lanes and its
        // uniform value.
        for d in &instruction.destinations {
            if d.kind == OperandKind::VectorRegister {
                state.clear_lanes(d.value);
            }
        }
        if instruction.encoding == Encoding::Vopc || (instruction.encoding == Encoding::Vop3 && opcode.is_vector_compare()) {
            self.apply_vector_compare(instruction, state);
            return;
        }
        let has_modifiers = match &instruction.control {
            Some(Control::Vop3(c)) => {
                c.absolute_mask != 0 || c.negate_mask != 0 || c.clamp || c.output_modifier != 0 || c.operand_select != 0
            }
            Some(Control::Sdwa(s)) => {
                s.absolute_mask != 0
                    || s.negate_mask != 0
                    || s.output_modifier != 0
                    || s.clamp
                    || s.destination_select == 7
                    || s.source0_select == 7
                    || s.source1_select == 7
                    || s.destination_unused == 3
            }
            Some(Control::Dpp(_) | Control::Dpp8(_) | Control::Vop3p(_)) => true,
            _ => false,
        };
        let mut value = if has_modifiers { self.graph.undefined(ValueType::U32) } else { self.vector_result(instruction, state) };
        let carry_destination = match &instruction.control {
            Some(Control::Vop3(c)) => c.scalar_destination,
            _ => None,
        };
        if let (Some(carry_dst), false, true) = (
            carry_destination,
            has_modifiers,
            matches!(opcode, VAddCoU32 | VSubCoU32 | VSubrevCoU32 | VAddCoCiU32 | VSubCoCiU32 | VSubrevCoCiU32 | VMadU64U32),
        ) {
            let carry = if self.graph.is_undefined(value) { self.graph.undefined(ValueType::Bool) } else { state.carry_out };
            let raw = self.select_bit(carry);
            self.write_vector_mask(state, carry_dst, raw, carry);
        } else if matches!(opcode, VAddcU32 | VSubbU32 | VSubbrevU32 | VAddCoU32 | VSubCoU32 | VSubrevCoU32) {
            let carry = if self.graph.is_undefined(value) { self.graph.undefined(ValueType::Bool) } else { state.carry_out };
            let raw = self.select_bit(carry);
            self.write_vector_mask(state, VCC_LOW, raw, carry);
        }
        for d in &instruction.destinations {
            if d.kind == OperandKind::VectorRegister {
                let result = match &instruction.control {
                    Some(Control::Sdwa(s)) => {
                        let previous = state.read_vector(self.graph, d.value);
                        self.apply_sdwa_destination(s, value, previous)
                    }
                    _ => value,
                };
                state.write_vector(self.graph, d.value, result);
                // Multi-dword results are not modelled; the second dword is
                // undefined.
                value = self.graph.undefined(ValueType::U32);
            } else if d.kind == OperandKind::ScalarRegister {
                let u = self.graph.undefined(ValueType::U32);
                state.write_scalar(self.graph, d.value, u);
            }
        }
    }

    fn vsrc(&mut self, sources: &[Operand], sdwa: Option<&SdwaControl>, index: usize, state: &mut RegisterState) -> ValueId {
        if index < sources.len() {
            let v = self.read_vector_operand(sources[index], state);
            self.apply_sdwa_source(sdwa, index, v)
        } else {
            self.graph.undefined(ValueType::U32)
        }
    }

    fn vector_result(&mut self, instruction: &Instruction, state: &mut RegisterState) -> ValueId {
        use Opcode::*;
        let sources = &instruction.sources;
        let sdwa = match &instruction.control {
            Some(Control::Sdwa(s)) => Some(s.clone()),
            _ => None,
        };
        let shift = |this: &mut Self, value: ValueId, count: ValueId, op: ScalarOp| {
            let c31 = this.c(31);
            let masked = this.binary(ScalarOp::And32, count, c31);
            this.binary(op, value, masked)
        };
        let low24 = |this: &mut Self, value: ValueId| this.bfe_const(value, 0, 24);
        let signed_low24 = |this: &mut Self, value: ValueId| {
            let (zero, c24) = (this.c(0), this.c(24));
            this.bfe(ScalarOp::BitFieldSExtract, value, zero, c24)
        };
        let op = instruction.opcode;
        macro_rules! bin {
            ($o:expr, $a:expr, $b:expr) => {{
                let a = self.vsrc(sources, sdwa.as_ref(), $a, state);
                let b = self.vsrc(sources, sdwa.as_ref(), $b, state);
                self.binary($o, a, b)
            }};
        }
        match op {
            VMovB32 => self.vsrc(sources, sdwa.as_ref(), 0, state),
            VAddI32 => bin!(ScalarOp::IAdd32, 0, 1),
            VSubI32 => bin!(ScalarOp::ISub32, 0, 1),
            VSubrevI32 => bin!(ScalarOp::ISub32, 1, 0),
            VAddCoU32 => {
                let (a, b) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state));
                let sum = self.binary64(ScalarOp::AddCarry32, a, b);
                let carry = self.extract(sum, 1);
                let nz = self.not_zero(carry);
                state.carry_out = self.boolean(ScalarOp::LogicalAnd, state.exec, nz);
                self.extract(sum, 0)
            }
            VSubCoU32 | VSubrevCoU32 => {
                let (s0, s1) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state));
                let (left, right) = if op == VSubCoU32 { (s0, s1) } else { (s1, s0) };
                let borrow = self.boolean(ScalarOp::UGreaterThan32, right, left);
                state.carry_out = self.boolean(ScalarOp::LogicalAnd, state.exec, borrow);
                self.binary(ScalarOp::ISub32, left, right)
            }
            VAddcU32 | VAddCoCiU32 => {
                let mask = if sources.len() > 2 { self.mask_of(sources[2], state) } else { state.vcc };
                let carry_in = self.select_bit(mask);
                let (a, b) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state));
                let first = self.binary64(ScalarOp::AddCarry32, a, b);
                let first_low = self.extract(first, 0);
                let second = self.binary64(ScalarOp::AddCarry32, first_low, carry_in);
                let (c1, c2) = (self.extract(first, 1), self.extract(second, 1));
                let any = self.binary(ScalarOp::Or32, c1, c2);
                let nz = self.not_zero(any);
                state.carry_out = self.boolean(ScalarOp::LogicalAnd, state.exec, nz);
                self.extract(second, 0)
            }
            VSubCoCiU32 | VSubrevCoCiU32 => {
                let (s0, s1) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state));
                let (left, right) = if op == VSubCoCiU32 { (s0, s1) } else { (s1, s0) };
                let mask = if sources.len() > 2 { self.mask_of(sources[2], state) } else { state.vcc };
                let borrow_in = self.select_bit(mask);
                let partial = self.binary(ScalarOp::ISub32, left, right);
                let result = self.binary(ScalarOp::ISub32, partial, borrow_in);
                let a = self.boolean(ScalarOp::UGreaterThan32, right, left);
                let b = self.boolean(ScalarOp::UGreaterThan32, borrow_in, partial);
                let any = self.boolean(ScalarOp::LogicalOr, a, b);
                state.carry_out = self.boolean(ScalarOp::LogicalAnd, state.exec, any);
                result
            }
            VMulLoU32 | VMulLoI32 => bin!(ScalarOp::IMul32, 0, 1),
            VMulHiU32 => bin!(ScalarOp::UMulHi32, 0, 1),
            VMulHiI32 => bin!(ScalarOp::SMulHi32, 0, 1),
            VMulU32U24 => {
                let (a, b) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state));
                let (a, b) = (low24(self, a), low24(self, b));
                self.binary(ScalarOp::IMul32, a, b)
            }
            VMulI32I24 => {
                let (a, b) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state));
                let (a, b) = (signed_low24(self, a), signed_low24(self, b));
                self.binary(ScalarOp::IMul32, a, b)
            }
            VCvtPkU16U32 => {
                let ffff = self.c(0xFFFF);
                let (a, b) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state));
                let low = self.binary(ScalarOp::UMin32, a, ffff);
                let high = self.binary(ScalarOp::UMin32, b, ffff);
                let c16 = self.c(16);
                let shifted = self.binary(ScalarOp::ShiftLeft32, high, c16);
                self.binary(ScalarOp::Or32, low, shifted)
            }
            VCvtPkI16I32 => {
                let clamp = |this: &mut Self, v: ValueId| {
                    let (lo, hi) = (this.c(0xFFFF_8000), this.c(0x7FFF));
                    let m = this.binary(ScalarOp::SMax32, v, lo);
                    this.binary(ScalarOp::SMin32, m, hi)
                };
                let (a, b) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state));
                let ca = clamp(self, a);
                let ffff = self.c(0xFFFF);
                let low = self.binary(ScalarOp::And32, ca, ffff);
                let cb = clamp(self, b);
                let c16 = self.c(16);
                let high = self.binary(ScalarOp::ShiftLeft32, cb, c16);
                self.binary(ScalarOp::Or32, low, high)
            }
            VMadU32U24 => {
                let (a, b, c) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state), self.vsrc(sources, sdwa.as_ref(), 2, state));
                let (a, b) = (low24(self, a), low24(self, b));
                let m = self.binary(ScalarOp::IMul32, a, b);
                self.binary(ScalarOp::IAdd32, m, c)
            }
            VLshlrevB32 | VLshrrevB32 | VAshrrevI32 | VLshlB32 | VLshrB32 | VAshrI32 => {
                let (s0, s1) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state));
                let reversed = matches!(op, VLshlrevB32 | VLshrrevB32 | VAshrrevI32);
                let (value, count) = if reversed { (s1, s0) } else { (s0, s1) };
                let o = match op {
                    VLshlrevB32 | VLshlB32 => ScalarOp::ShiftLeft32,
                    VLshrrevB32 | VLshrB32 => ScalarOp::ShiftRightLogical32,
                    _ => ScalarOp::ShiftRightArithmetic32,
                };
                shift(self, value, count, o)
            }
            VAndB32 => bin!(ScalarOp::And32, 0, 1),
            VOrB32 => bin!(ScalarOp::Or32, 0, 1),
            VXorB32 => bin!(ScalarOp::Xor32, 0, 1),
            VXor3B32 => {
                let x = bin!(ScalarOp::Xor32, 0, 1);
                let c = self.vsrc(sources, sdwa.as_ref(), 2, state);
                self.binary(ScalarOp::Xor32, x, c)
            }
            VXnorB32 => {
                let x = bin!(ScalarOp::Xor32, 0, 1);
                self.unary(ScalarOp::Not32, x)
            }
            VNotB32 => {
                let a = self.vsrc(sources, sdwa.as_ref(), 0, state);
                self.unary(ScalarOp::Not32, a)
            }
            VMinU32 => bin!(ScalarOp::UMin32, 0, 1),
            VMaxU32 => bin!(ScalarOp::UMax32, 0, 1),
            VMinI32 => bin!(ScalarOp::SMin32, 0, 1),
            VMaxI32 => bin!(ScalarOp::SMax32, 0, 1),
            VMed3U32 | VMed3I32 => {
                // median(a, b, c) = max(min(a, b), min(max(a, b), c))
                let (min, max) = if op == VMed3U32 { (ScalarOp::UMin32, ScalarOp::UMax32) } else { (ScalarOp::SMin32, ScalarOp::SMax32) };
                let (a, b, c) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state), self.vsrc(sources, sdwa.as_ref(), 2, state));
                let lo = self.binary(min, a, b);
                let hi = self.binary(max, a, b);
                let mid = self.binary(min, hi, c);
                self.binary(max, lo, mid)
            }
            VBfeU32 | VBfeI32 => {
                let (a, b, c) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state), self.vsrc(sources, sdwa.as_ref(), 2, state));
                let c31 = self.c(31);
                let offset = self.binary(ScalarOp::And32, b, c31);
                let raw_count = self.binary(ScalarOp::And32, c, c31);
                let c32 = self.c(32);
                let room = self.binary(ScalarOp::ISub32, c32, offset);
                let count = self.binary(ScalarOp::UMin32, raw_count, room);
                self.bfe(if op == VBfeU32 { ScalarOp::BitFieldUExtract } else { ScalarOp::BitFieldSExtract }, a, offset, count)
            }
            VBfiB32 => {
                let (bits, b, c) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state), self.vsrc(sources, sdwa.as_ref(), 2, state));
                let x = self.binary(ScalarOp::And32, bits, b);
                let n = self.unary(ScalarOp::Not32, bits);
                let y = self.binary(ScalarOp::And32, n, c);
                self.binary(ScalarOp::Or32, x, y)
            }
            VBfmB32 => {
                let (a, b) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state));
                let c31 = self.c(31);
                let count = self.binary(ScalarOp::And32, a, c31);
                let offset = self.binary(ScalarOp::And32, b, c31);
                let (zero, ones) = (self.c(0), self.c(u32::MAX));
                self.graph.operation(ScalarOp::BitFieldInsert, ValueType::U32, &[zero, ones, offset, count])
            }
            VLshlAddU32 => {
                let (a, b, c) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state), self.vsrc(sources, sdwa.as_ref(), 2, state));
                let s = shift(self, a, b, ScalarOp::ShiftLeft32);
                self.binary(ScalarOp::IAdd32, s, c)
            }
            VAddLshlU32 => {
                let (a, b, c) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state), self.vsrc(sources, sdwa.as_ref(), 2, state));
                let sum = self.binary(ScalarOp::IAdd32, a, b);
                shift(self, sum, c, ScalarOp::ShiftLeft32)
            }
            VLshlOrU32 => {
                let (a, b, c) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state), self.vsrc(sources, sdwa.as_ref(), 2, state));
                let s = shift(self, a, b, ScalarOp::ShiftLeft32);
                self.binary(ScalarOp::Or32, s, c)
            }
            VAndOrB32 => {
                let x = bin!(ScalarOp::And32, 0, 1);
                let c = self.vsrc(sources, sdwa.as_ref(), 2, state);
                self.binary(ScalarOp::Or32, x, c)
            }
            VOr3U32 => {
                let x = bin!(ScalarOp::Or32, 0, 1);
                let c = self.vsrc(sources, sdwa.as_ref(), 2, state);
                self.binary(ScalarOp::Or32, x, c)
            }
            VAdd3U32 => {
                let x = bin!(ScalarOp::IAdd32, 0, 1);
                let c = self.vsrc(sources, sdwa.as_ref(), 2, state);
                self.binary(ScalarOp::IAdd32, x, c)
            }
            VXadU32 => {
                let x = bin!(ScalarOp::Xor32, 0, 1);
                let c = self.vsrc(sources, sdwa.as_ref(), 2, state);
                self.binary(ScalarOp::IAdd32, x, c)
            }
            VSadU32 => {
                let (a, b, c) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state), self.vsrc(sources, sdwa.as_ref(), 2, state));
                let low = self.binary(ScalarOp::UMin32, a, b);
                let high = self.binary(ScalarOp::UMax32, a, b);
                let d = self.binary(ScalarOp::ISub32, high, low);
                self.binary(ScalarOp::IAdd32, d, c)
            }
            VCndmaskB32 => {
                let condition = if sources.len() > 2 { self.mask_of(sources[2], state) } else { state.vcc };
                let (a, b) = (self.vsrc(sources, sdwa.as_ref(), 0, state), self.vsrc(sources, sdwa.as_ref(), 1, state));
                self.graph.select(condition, b, a)
            }
            VCvtF32U32 | VCvtU32F32 | VCvtF32I32 | VCvtI32F32 | VTruncF32 => {
                let a = self.vsrc(sources, sdwa.as_ref(), 0, state);
                let o = match op {
                    VCvtF32U32 => ScalarOp::ConvertF32U32,
                    VCvtU32F32 => ScalarOp::ConvertU32F32,
                    VCvtF32I32 => ScalarOp::ConvertF32S32,
                    VCvtI32F32 => ScalarOp::ConvertS32F32,
                    _ => ScalarOp::FTrunc,
                };
                self.unary(o, a)
            }
            VMulF32 => bin!(ScalarOp::FMul, 0, 1),
            VAddF32 => bin!(ScalarOp::FAdd, 0, 1),
            VSubF32 => bin!(ScalarOp::FSub, 0, 1),
            VSubrevF32 => bin!(ScalarOp::FSub, 1, 0),
            VMinF32 => bin!(ScalarOp::FMin, 0, 1),
            VMaxF32 => bin!(ScalarOp::FMax, 0, 1),
            _ => self.graph.undefined(ValueType::U32),
        }
    }

    fn apply_sdwa_source(&mut self, control: Option<&SdwaControl>, index: usize, value: ValueId) -> ValueId {
        let Some(control) = control else { return value };
        if index > 1 {
            return value;
        }
        let selector = if index == 0 { control.source0_select } else { control.source1_select };
        if selector == 6 {
            return value;
        }
        if selector > 5 {
            return self.graph.undefined(ValueType::U32);
        }
        let width = if selector <= 3 { 8 } else { 16 };
        let shift = if selector <= 3 { selector * 8 } else { (selector - 4) * 16 };
        let s = self.c(shift);
        let shifted = self.binary(ScalarOp::ShiftRightLogical32, value, s);
        let m = self.c(if width == 8 { 0xFF } else { 0xFFFF });
        let extracted = self.binary(ScalarOp::And32, shifted, m);
        let sign_extend = if index == 0 { control.source0_sign_extend } else { control.source1_sign_extend };
        if sign_extend {
            let (zero, w) = (self.c(0), self.c(width));
            self.bfe(ScalarOp::BitFieldSExtract, extracted, zero, w)
        } else {
            extracted
        }
    }

    fn apply_sdwa_destination(&mut self, control: &SdwaControl, value: ValueId, previous: ValueId) -> ValueId {
        if control.destination_select == 6 {
            return value;
        }
        if control.destination_select > 5 || control.destination_unused == 3 {
            return self.graph.undefined(ValueType::U32);
        }
        let select = control.destination_select;
        let width = if select <= 3 { 8 } else { 16 };
        let shift = if select <= 3 { select * 8 } else { (select - 4) * 16 };
        let low_mask: u32 = if width == 8 { 0xFF } else { 0xFFFF };
        let field_mask = low_mask << shift;
        let lm = self.c(low_mask);
        let masked = self.binary(ScalarOp::And32, value, lm);
        let s = self.c(shift);
        let positioned = self.binary(ScalarOp::ShiftLeft32, masked, s);
        match control.destination_unused {
            0 => positioned,
            1 => {
                let sign = self.c(1u32 << (shift + width - 1));
                let signed = self.binary(ScalarOp::And32, positioned, sign);
                let nz = self.not_zero(signed);
                let fill = self.c(u32::MAX.checked_shl(shift + width).unwrap_or(0));
                let zero = self.c(0);
                let ext = self.graph.select(nz, fill, zero);
                self.binary(ScalarOp::Or32, positioned, ext)
            }
            2 => {
                let keep = self.c(!field_mask);
                let kept = self.binary(ScalarOp::And32, previous, keep);
                self.binary(ScalarOp::Or32, kept, positioned)
            }
            _ => self.graph.undefined(ValueType::U32),
        }
    }

    fn apply_vector_compare(&mut self, instruction: &Instruction, state: &mut RegisterState) {
        let opcode = instruction.opcode;
        let mut left = match instruction.sources.first() {
            Some(&s) => self.read_vector_operand(s, state),
            None => self.graph.undefined(ValueType::U32),
        };
        let mut right = match instruction.sources.get(1) {
            Some(&s) => self.read_vector_operand(s, state),
            None => self.graph.undefined(ValueType::U32),
        };
        let updates_exec = opcode.is_vector_compare_exec();
        let name = opcode.name();
        let suffix = &name[if updates_exec { 5 } else { 4 }..];
        // Integer 16-bit compares consume the low word; signed forms
        // sign-extend it.
        if suffix.ends_with("16") && !suffix.ends_with("F16") {
            let signed = suffix.ends_with("I16");
            let narrow = |this: &mut Self, v: ValueId| {
                let ffff = this.c(0xFFFF);
                let low = this.binary(ScalarOp::And32, v, ffff);
                if signed {
                    let c16 = this.c(16);
                    let up = this.binary(ScalarOp::ShiftLeft32, low, c16);
                    this.binary(ScalarOp::ShiftRightArithmetic32, up, c16)
                } else {
                    low
                }
            };
            left = narrow(self, left);
            right = narrow(self, right);
        }
        let modified = match &instruction.control {
            Some(Control::Sdwa(_)) => true,
            Some(Control::Vop3(c)) => c.absolute_mask != 0 || c.negate_mask != 0,
            _ => false,
        };
        let op = if modified {
            None
        } else {
            match suffix {
                "EqU32" | "EqI32" | "EqU16" | "EqI16" => Some(ScalarOp::IEqual32),
                "NeU32" | "NeI32" | "NeU16" | "NeI16" => Some(ScalarOp::INotEqual32),
                "LtU32" | "LtU16" => Some(ScalarOp::ULessThan32),
                "GtU32" | "GtU16" => Some(ScalarOp::UGreaterThan32),
                "LeU32" | "LeU16" => Some(ScalarOp::ULessThanEqual32),
                "GeU32" | "GeU16" => Some(ScalarOp::UGreaterThanEqual32),
                "LtI32" | "LtI16" => Some(ScalarOp::SLessThan32),
                "GtI32" | "GtI16" => Some(ScalarOp::SGreaterThan32),
                "LeI32" | "LeI16" => Some(ScalarOp::SLessThanEqual32),
                "GeI32" | "GeI16" => Some(ScalarOp::SGreaterThanEqual32),
                "LeF32" => Some(ScalarOp::FLessThanEqual),
                "GeF32" => Some(ScalarOp::FGreaterThanEqual),
                "LtF32" => Some(ScalarOp::FLessThan),
                "GtF32" => Some(ScalarOp::FGreaterThan),
                "EqF32" => Some(ScalarOp::FEqual),
                "NeqF32" | "LgF32" => Some(ScalarOp::FNotEqual),
                _ => None,
            }
        };
        let result = match op {
            Some(op) => self.boolean(op, left, right),
            None => self.graph.undefined(ValueType::Bool),
        };
        let masked = if self.graph.is_undefined(result) { result } else { self.boolean(ScalarOp::LogicalAnd, state.exec, result) };
        let raw = self.select_bit(masked);
        // Execution-mask comparisons leave the vector condition registers
        // unchanged.
        if updates_exec {
            self.write_vector_mask(state, EXEC_LOW, raw, masked);
            return;
        }
        match instruction.destinations.iter().find(|d| d.kind == OperandKind::ScalarRegister) {
            Some(d) => self.write_vector_mask(state, d.value, raw, masked),
            None => self.write_vector_mask(state, VCC_LOW, raw, masked),
        }
    }

    fn apply_write_lane(&mut self, instruction: &Instruction, state: &mut RegisterState) {
        if instruction.destinations.len() != 1
            || instruction.destinations[0].kind != OperandKind::VectorRegister
            || instruction.sources.len() < 2
        {
            return;
        }
        let dst = instruction.destinations[0].value;
        let lane = self.read(instruction.sources[1], state);
        let u = self.graph.undefined(ValueType::U32);
        state.write_vector(self.graph, dst, u);
        let Some(lane) = self.graph.constant_u32(lane) else {
            state.clear_lanes(dst);
            return;
        };
        let value = self.read(instruction.sources[0], state);
        let key = (dst, lane & 63);
        if self.graph.is_undefined(value) {
            state.lanes.remove(&key);
        } else {
            state.lanes.insert(key, value);
        }
    }

    fn apply_read_lane(&mut self, instruction: &Instruction, state: &mut RegisterState) {
        if instruction.destinations.len() != 1 || instruction.destinations[0].kind != OperandKind::ScalarRegister {
            return;
        }
        let dst = instruction.destinations[0].value;
        let lane = match instruction.sources.get(1) {
            Some(&s) => self.read(s, state),
            None => self.graph.undefined(ValueType::U32),
        };
        if instruction.sources.len() < 2 || instruction.sources[0].kind != OperandKind::VectorRegister {
            let u = self.graph.undefined(ValueType::U32);
            state.write_scalar(self.graph, dst, u);
            return;
        }
        let source = instruction.sources[0].value;
        if let Some(l) = self.graph.constant_u32(lane) {
            if let Some(&v) = state.lanes.get(&(source, l & 63)) {
                state.write_scalar(self.graph, dst, v);
                return;
            }
        }
        let source_value = state.read_vector(self.graph, source);
        let v = if !self.graph.is_constant(lane) && self.graph.is_undefined(source_value) {
            self.graph.first_lane(source_value, state.exec, instruction.pc)
        } else {
            self.graph.undefined(ValueType::U32)
        };
        state.write_scalar(self.graph, dst, v);
    }

    // ---- memory instructions ----

    fn apply_memory(&mut self, instruction: &Instruction, state: &mut RegisterState) {
        let mut vector_read = None;
        if self.recording {
            if let Some(memory_index) = self.graph.memory.index_of(instruction.pc, 0) {
                let mut binding = match &instruction.control {
                    Some(Control::BufferMemory(buffer)) => {
                        let dwords: Vec<ValueId> = (0..4).map(|i| state.read(self.graph, buffer.scalar_resource + i)).collect();
                        Some(MemoryAccessBinding {
                            handle: Some(self.graph.handle(ValueKind::BufferHandle, &dwords)),
                            ..Default::default()
                        })
                    }
                    Some(Control::Image(image)) => {
                        let dwords: Vec<ValueId> = (0..8).map(|i| state.read(self.graph, image.scalar_resource + i)).collect();
                        let handle = self.graph.handle(ValueKind::ImageHandle, &dwords);
                        let sampler = if self.graph.memory.get(memory_index).needs_sampler {
                            let dwords: Vec<ValueId> = (0..4).map(|i| state.read(self.graph, image.scalar_sampler + i)).collect();
                            Some(self.graph.handle(ValueKind::SamplerHandle, &dwords))
                        } else {
                            None
                        };
                        Some(MemoryAccessBinding { handle: Some(handle), sampler_handle: sampler, ..Default::default() })
                    }
                    Some(Control::RayIntersect(ray)) => {
                        let dwords: Vec<ValueId> = (0..8).map(|i| state.read(self.graph, ray.scalar_resource + i)).collect();
                        Some(MemoryAccessBinding {
                            handle: Some(self.graph.handle(ValueKind::ImageHandle, &dwords)),
                            ..Default::default()
                        })
                    }
                    Some(Control::GlobalMemory(global)) => {
                        let handle = self.address_handle_of(global, state);
                        let offset = if global.uses_flat_address { None } else { Some(state.read_vector(self.graph, global.vector_address)) };
                        Some(MemoryAccessBinding { handle: Some(handle), offset, active: Some(state.exec), ..Default::default() })
                    }
                    _ => None,
                };
                if let (Some(b), Some(Control::BufferMemory(buffer))) = (&mut binding, &instruction.control) {
                    if !instruction.opcode.name().contains("Store") {
                        let handle = b.handle.unwrap();
                        let offset = state.read_vector(self.graph, buffer.vector_address);
                        // A lane-varying address loads a lane-varying value.
                        if !self.graph.is_undefined(offset) {
                            let read = self.graph.memory_read(ValueKind::ScalarBufferWord, handle, offset, memory_index);
                            vector_read = Some(read);
                            b.read = Some(read);
                        }
                    }
                }
                self.graph.accesses[memory_index] = binding;
            }
        }
        for d in &instruction.destinations {
            if d.kind == OperandKind::VectorRegister {
                state.clear_lanes(d.value);
                let v = vector_read.unwrap_or_else(|| self.graph.undefined(ValueType::U32));
                state.write_vector(self.graph, d.value, v);
            } else if d.kind == OperandKind::ScalarRegister {
                let u = self.graph.undefined(ValueType::U32);
                state.write_scalar(self.graph, d.value, u);
            }
        }
    }

    /// A scalar address pair when the access has one, else the vector pair.
    fn address_handle_of(&mut self, control: &GlobalMemoryControl, state: &mut RegisterState) -> ValueId {
        if (control.scalar_address as usize) < SCALAR_REGISTER_COUNT - 1 && control.scalar_address != NULL_SCALAR_REGISTER {
            let dwords = [state.read(self.graph, control.scalar_address), state.read(self.graph, control.scalar_address + 1)];
            return self.graph.handle(ValueKind::AddressHandle, &dwords);
        }
        let dwords = [
            state.read_vector(self.graph, control.vector_address),
            state.read_vector(self.graph, control.vector_address + 1),
        ];
        self.graph.handle(ValueKind::AddressHandle, &dwords)
    }

    // ---- operand helpers ----

    fn read(&mut self, operand: Operand, state: &mut RegisterState) -> ValueId {
        match operand.kind {
            OperandKind::ScalarRegister => state.read(self.graph, operand.value),
            OperandKind::LiteralConstant => self.c(operand.value),
            OperandKind::EncodedConstant => {
                let (zero, one) = (self.c(0), self.c(1));
                match operand.value {
                    VCCZ_SOURCE => return self.graph.select(state.vcc, zero, one),
                    EXECZ_SOURCE => return self.graph.select(state.exec, zero, one),
                    SCC_SOURCE => return self.graph.select(state.scc, one, zero),
                    _ => {}
                }
                // A 32-bit aperture read is the low dword, a constant (only
                // the high dword of a 64-bit read carries the aperture).
                match inline_constants::decode(operand.value) {
                    Some(v) => self.c(v),
                    None => self.graph.undefined(ValueType::U32),
                }
            }
            OperandKind::VectorRegister => self.graph.undefined(ValueType::U32),
        }
    }

    fn read_vector_operand(&mut self, operand: Operand, state: &mut RegisterState) -> ValueId {
        if operand.kind == OperandKind::VectorRegister {
            state.read_vector(self.graph, operand.value)
        } else {
            self.read(operand, state)
        }
    }

    fn read_pair(&mut self, operand: Operand, state: &mut RegisterState) -> (ValueId, ValueId) {
        if operand.kind == OperandKind::ScalarRegister {
            return (state.read(self.graph, operand.value), state.read(self.graph, operand.value + 1));
        }
        if operand.kind == OperandKind::EncodedConstant && inline_constants::is_aperture(operand.value) {
            // A 64-bit aperture read: the offset dword is a constant, the
            // high dword carries the aperture.
            let low = self.c(inline_constants::decode_aperture64(operand.value) as u32);
            return (low, self.graph.memory_aperture(operand.value));
        }
        // A float inline constant of a 64-bit operand is the f64 value.
        if operand.kind == OperandKind::EncodedConstant {
            if let Some(bits) = crate::translate::operand::f64_inline(operand.value) {
                return (self.c(bits as u32), self.c((bits >> 32) as u32));
            }
        }
        let low = self.read(operand, state);
        let negative = operand.kind == OperandKind::EncodedConstant && (193..=208).contains(&operand.value);
        (low, self.c(if negative { u32::MAX } else { 0 }))
    }

    /// The lane-mask meaning of an operand: EXEC, VCC, a pair written as a
    /// mask, or whether the raw pair is non-zero.
    fn mask_of(&mut self, operand: Operand, state: &mut RegisterState) -> ValueId {
        if operand.kind == OperandKind::ScalarRegister {
            match operand.value {
                EXEC_LOW => return state.exec,
                VCC_LOW => return state.vcc,
                _ => {}
            }
            if let Some(&mask) = state.thread_bits.get(&operand.value) {
                return mask;
            }
        }
        let (low, high) = self.read_pair(operand, state);
        let both = self.binary(ScalarOp::Or32, low, high);
        self.not_zero(both)
    }
}

