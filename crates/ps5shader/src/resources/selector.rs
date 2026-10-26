// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Finite over-estimates of the run-time selector of an indirect image
//! (`v_readfirstlane` of a lane-varying key): the set of values it can take
//! for one draw, so the host materialises only those candidates.

use super::evaluator::{ComputeSelectorState, ResourceRuntimeInputs, RuntimeValueEvaluator};
use super::info::ShaderStage;
use super::plan::ShaderResourcePlan;
use super::value::{evaluate_operation, ScalarOp, ValueId, ValueKind};
use crate::inline_constants;
use crate::ir_types::*;
use crate::opcode::Opcode;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

const MAXIMUM_VALUES: usize = 4096;
const MAXIMUM_COMBINATIONS: u64 = 65536;

#[derive(Debug)]
enum Expression {
    Values(Vec<u32>),
    Runtime(ValueId),
    /// Any of the inputs (several reaching definitions).
    Union(Vec<Arc<Expression>>),
    /// A binary operation over every combination of the inputs.
    Operation(ScalarOp, Vec<Arc<Expression>>),
}

/// What the selector evaluation did, for diagnostics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndirectSelectorDiagnostic {
    pub selection_mode: String,
    pub evaluation_failure: Option<String>,
    pub selector_indices: Vec<u32>,
    pub proven_offsets: Vec<u32>,
    /// `(probed offset, key)` of every distinct key read.
    pub key_probes: Vec<(u32, u32)>,
}

/// A finite over-estimate of a selector. Unsupported shapes keep the
/// full-domain scan.
#[derive(Debug)]
pub struct IndirectSelectorValues {
    root: Arc<Expression>,
    wave_bounds: Option<WaveMaskSelectorBounds>,
}

impl IndirectSelectorValues {
    pub(super) fn create(plan: &ShaderResourcePlan, selector: ValueId) -> Option<Self> {
        let node = plan.graph.node(selector);
        if node.kind != ValueKind::FirstLane {
            return None;
        }
        let instruction = plan.graph.program.instructions.iter().find(|i| i.pc as u64 == node.payload)?;
        if instruction.opcode != Opcode::VReadfirstlaneB32 || instruction.sources.len() != 1 {
            return None;
        }
        let mut builder = ExpressionBuilder { plan, active: HashSet::new(), requests: 0, has_bit_scan: false };
        let root = builder.read(instruction.sources[0], instruction.pc)?;
        if !builder.has_bit_scan {
            return None;
        }
        Some(IndirectSelectorValues { root, wave_bounds: WaveMaskSelectorBounds::try_create(plan, instruction) })
    }

    /// The selector values for one draw (memory read with the clean reader).
    pub(super) fn evaluate(
        &self,
        plan: &ShaderResourcePlan,
        inputs: &ResourceRuntimeInputs<'_>,
        mut diagnostic: Option<&mut IndirectSelectorDiagnostic>,
    ) -> Option<Vec<u32>> {
        let evaluator = RuntimeValueEvaluator::new(plan, inputs.with_reader(inputs.read_clean_memory));
        if let Some(bounds) = &self.wave_bounds {
            if let Some(values) = bounds.evaluate(&evaluator, inputs.compute_state) {
                return Some(values);
            }
        }
        let mut cache: HashMap<*const Expression, Vec<u32>> = HashMap::new();
        let values = evaluate_expression(&self.root, &evaluator, &mut cache, &mut diagnostic).unwrap_or_default();
        (!values.is_empty()).then_some(values)
    }
}

fn decline(diagnostic: &mut Option<&mut IndirectSelectorDiagnostic>, reason: &str) -> Option<Vec<u32>> {
    if let Some(d) = diagnostic {
        d.evaluation_failure.get_or_insert_with(|| reason.to_string());
    }
    None
}

fn evaluate_expression(
    expression: &Arc<Expression>,
    evaluator: &RuntimeValueEvaluator<'_>,
    cache: &mut HashMap<*const Expression, Vec<u32>>,
    diagnostic: &mut Option<&mut IndirectSelectorDiagnostic>,
) -> Option<Vec<u32>> {
    let key = Arc::as_ptr(expression);
    if let Some(v) = cache.get(&key) {
        return Some(v.clone());
    }
    let (inputs, op) = match &**expression {
        Expression::Values(v) => return Some(v.clone()),
        Expression::Runtime(value) => {
            return match evaluator.evaluate(*value) {
                Some(v) => Some(vec![v]),
                None => decline(diagnostic, "runtime_value_unavailable"),
            }
        }
        Expression::Union(inputs) => (inputs, ScalarOp::None),
        Expression::Operation(op, inputs) => (inputs, *op),
    };
    let mut operands = Vec::with_capacity(inputs.len());
    for input in inputs {
        operands.push(evaluate_expression(input, evaluator, cache, diagnostic)?);
    }
    let mut results: Vec<u32> = Vec::new();
    let mut seen = HashSet::new();
    if op == ScalarOp::None {
        for operand in &operands {
            for &v in operand {
                if seen.insert(v) {
                    results.push(v);
                    if results.len() > MAXIMUM_VALUES {
                        return decline(diagnostic, "value_limit");
                    }
                }
            }
        }
    } else {
        if operands[0].len() as u64 * operands[1].len() as u64 > MAXIMUM_COMBINATIONS {
            return decline(diagnostic, "combination_limit");
        }
        for &left in &operands[0] {
            for &right in &operands[1] {
                let Some(r) = evaluate_operation(op, &[left as u64, right as u64]) else {
                    return decline(diagnostic, "unsupported_operation");
                };
                if seen.insert(r as u32) {
                    results.push(r as u32);
                    if results.len() > MAXIMUM_VALUES {
                        return decline(diagnostic, "value_limit");
                    }
                }
            }
        }
    }
    cache.insert(key, results.clone());
    Some(results)
}

struct ExpressionBuilder<'p> {
    plan: &'p ShaderResourcePlan,
    active: HashSet<(Operand, u32)>,
    requests: u32,
    has_bit_scan: bool,
}

impl ExpressionBuilder<'_> {
    /// The values `operand` can hold just before pc `before`: the reaching
    /// definitions on every path, or the entry user data.
    fn read(&mut self, operand: Operand, before: u32) -> Option<Arc<Expression>> {
        self.requests += 1;
        if self.requests > 512 || !self.active.insert((operand, before)) {
            return None;
        }
        let result = self.read_inner(operand, before);
        self.active.remove(&(operand, before));
        result
    }

    fn read_inner(&mut self, operand: Operand, before: u32) -> Option<Arc<Expression>> {
        match operand.kind {
            OperandKind::LiteralConstant => return Some(Arc::new(Expression::Values(vec![operand.value]))),
            OperandKind::EncodedConstant => {
                return inline_constants::decode(operand.value).map(|c| Arc::new(Expression::Values(vec![c])))
            }
            _ => {}
        }
        let plan = self.plan;
        let flow = &plan.graph.cfg;
        let instructions = &plan.graph.program.instructions;
        let initial = flow.blocks.iter().position(|b| before >= b.start_pc && before < b.end_pc)?;
        let mut pending = VecDeque::from([(initial, before)]);
        let mut visited = HashSet::new();
        let mut definitions = Vec::new();
        while let Some(position) = pending.pop_front() {
            if !visited.insert(position) {
                continue;
            }
            let block = flow.blocks[position.0];
            let mut found = false;
            for instruction in instructions.iter().rev() {
                if instruction.pc < block.start_pc || instruction.pc >= position.1 {
                    continue;
                }
                let name = instruction.opcode.name();
                if name.contains("Movreld") || name.contains("Movrelsd") || name.contains("Swaprel") || name.contains("GprIdx") {
                    return None;
                }
                if writes_register(instruction, operand) {
                    definitions.push(self.define(instruction, operand)?);
                    found = true;
                    break;
                }
                // A later active lane must also have been active at the
                // defining write.
                if operand.kind == OperandKind::VectorRegister && may_expand_execution(instruction) {
                    return None;
                }
            }
            if found {
                continue;
            }
            if position.0 == 0 {
                if operand.kind != OperandKind::ScalarRegister
                    || operand.value < plan.user_data_base()
                    || operand.value - plan.user_data_base() >= plan.user_data_count()
                {
                    return None;
                }
                // The user-data leaf: the graph interns it, find it.
                let leaf = plan
                    .graph
                    .values()
                    .find(|&v| plan.graph.kind(v) == ValueKind::UserData && plan.graph.node(v).payload == operand.value as u64)?;
                definitions.push(Arc::new(Expression::Runtime(leaf)));
            }
            if flow.predecessors[position.0].is_empty() && position.0 != 0 {
                return None;
            }
            for &predecessor in &flow.predecessors[position.0] {
                pending.push_back((predecessor, flow.blocks[predecessor].end_pc));
            }
        }
        (!definitions.is_empty()).then(|| Arc::new(Expression::Union(definitions)))
    }

    fn define(&mut self, instruction: &Instruction, destination: Operand) -> Option<Arc<Expression>> {
        if !instruction.destinations.contains(&destination) {
            return None;
        }
        let modified = match &instruction.control {
            Some(Control::Vop3(c)) => c.absolute_mask != 0 || c.negate_mask != 0 || c.clamp || c.output_modifier != 0 || c.operand_select != 0,
            Some(Control::Sdwa(_) | Control::Dpp(_) | Control::Dpp8(_) | Control::Vop3p(_)) => true,
            _ => false,
        };
        if modified {
            return None;
        }
        if matches!(instruction.opcode, Opcode::SFF1I32B32 | Opcode::VFfblB32) {
            self.has_bit_scan = true;
            let mut values: Vec<u32> = (0..32).collect();
            values.push(u32::MAX);
            return Some(Arc::new(Expression::Values(values)));
        }
        let plan = self.plan;
        if instruction.encoding == Encoding::Smem {
            let component = instruction.destinations.iter().position(|d| *d == destination)?;
            let memory_index = plan.memory().index_of(instruction.pc, component as u32)?;
            let read = plan.graph.accesses[memory_index].and_then(|a| a.read)?;
            return plan.validate_runtime_value(read).then(|| Arc::new(Expression::Runtime(read)));
        }
        if matches!(instruction.opcode, Opcode::SMovB32 | Opcode::VMovB32) {
            return self.read(instruction.sources[0], instruction.pc);
        }
        let op = match instruction.opcode {
            Opcode::SAddU32 | Opcode::SAddI32 | Opcode::VAddI32 | Opcode::VAdd3U32 => ScalarOp::IAdd32,
            Opcode::SLshlB32 => ScalarOp::ShiftLeft32,
            _ => return None,
        };
        if instruction.sources.len() < 2 {
            return None;
        }
        let left = self.read(instruction.sources[0], instruction.pc)?;
        let right = self.read(instruction.sources[1], instruction.pc)?;
        let result = Arc::new(Expression::Operation(op, vec![left, right]));
        if instruction.opcode != Opcode::VAdd3U32 {
            return Some(result);
        }
        let third = self.read(*instruction.sources.get(2)?, instruction.pc)?;
        Some(Arc::new(Expression::Operation(ScalarOp::IAdd32, vec![result, third])))
    }
}

fn may_expand_execution(instruction: &Instruction) -> bool {
    if matches!(instruction.opcode, Opcode::SAndSaveexecB64 | Opcode::SAndSaveexecB32) {
        return false;
    }
    let name = instruction.opcode.name();
    name.contains("Saveexec")
        || name.contains("Wrexec")
        || name.starts_with("VCmpx")
        || writes_register(instruction, Operand::scalar(126))
        || writes_register(instruction, Operand::scalar(127))
}

fn writes_register(instruction: &Instruction, register: Operand) -> bool {
    let name = instruction.opcode.name();
    if register.kind == OperandKind::ScalarRegister && (register.value == 106 || register.value == 107) && name.starts_with('V') {
        return true;
    }
    if let Some(Control::Vop3(Vop3Control { scalar_destination: Some(d), .. })) = &instruction.control {
        if register.kind == OperandKind::ScalarRegister && register.value >= *d && register.value - d < 2 {
            return true;
        }
    }
    instruction.destinations.iter().any(|d| {
        *d == register
            || (d.kind == register.kind && name.contains("64") && register.value > d.value && register.value - d.value == 1)
    })
}

// ---- wave-mask selector bounds ----

/// Proves the range of a two-level mask built and consumed by one complete
/// wave: a compute kernel that builds per-record bit masks in LDS, then
/// visits the set bits. The selector then ranges over `first .. first +
/// count`.
#[derive(Debug)]
struct WaveMaskSelectorBounds {
    first_record: ValueId,
    record_count: ValueId,
    row_shift: u32,
}

impl WaveMaskSelectorBounds {
    fn evaluate(&self, evaluator: &RuntimeValueEvaluator<'_>, state: Option<ComputeSelectorState>) -> Option<Vec<u32>> {
        let compute = state?;
        if compute.wave_size != 64
            || compute.threads_z != 1
            || compute.has_partial_workgroups
            || compute.local_invocation_id_components < 2
            || self.row_shift > 6
            || compute.threads_x != 1 << self.row_shift
            || compute.threads_x as u64 * compute.threads_y as u64 != 64
        {
            return None;
        }
        let first = evaluator.evaluate(self.first_record)?;
        let count = evaluator.evaluate(self.record_count)?;
        if count == 0 || count > 1024 || count.div_ceil(64) * 2 > compute.local_data_share_dwords {
            return None;
        }
        Some((0..count).map(|i| first.wrapping_add(i)).collect())
    }

    fn try_create(plan: &ShaderResourcePlan, selector: &Instruction) -> Option<Self> {
        if plan.stage != ShaderStage::Compute {
            return None;
        }
        MaskConstructionProof::new(plan).try_create_bounds(selector)
    }
}

const EXECUTION: Operand = Operand::scalar(126);
const COMPARISON: Operand = Operand::scalar(106);

fn literal(value: u32) -> Operand {
    Operand::literal(value)
}

struct MaskConstructionProof<'p> {
    plan: &'p ShaderResourcePlan,
    instructions: Vec<&'p Instruction>,
}

impl<'p> MaskConstructionProof<'p> {
    fn new(plan: &'p ShaderResourcePlan) -> Self {
        let instructions =
            plan.graph.program.instructions.iter().filter(|i| !matches!(i.opcode, Opcode::SWaitcnt | Opcode::SNop)).collect();
        MaskConstructionProof { plan, instructions }
    }

    fn len(&self) -> usize {
        self.instructions.len()
    }

    fn try_create_bounds(&self, selector: &Instruction) -> Option<WaveMaskSelectorBounds> {
        // These can hide register writes or add control-flow edges.
        if self.instructions.iter().any(|i| {
            let name = i.opcode.name();
            name.to_ascii_lowercase().contains("rel") || name.contains("GprIdx") || matches!(i.opcode, Opcode::SSetpcB64 | Opcode::SSwappcB64)
        }) {
            return None;
        }
        let selector_index = self.instructions.iter().position(|i| i.pc == selector.pc)? as i64;
        if selector.sources.len() != 1 {
            return None;
        }
        let addition_index = self.find_previous_definition(selector.sources[0], selector_index);
        if addition_index < 0 {
            return None;
        }
        let addition = self.at(addition_index)?;
        if addition.opcode != Opcode::VAdd3U32 || addition.sources.len() != 3 || !has_unmodified_operands(addition) {
            return None;
        }
        let (first, group_offset, bit_index) = (addition.sources[0], addition.sources[1], addition.sources[2]);
        if first.kind != OperandKind::ScalarRegister || group_offset.kind != OperandKind::ScalarRegister || bit_index.kind != OperandKind::VectorRegister {
            return None;
        }
        let bit_scan_index = self.find_previous_definition(bit_index, addition_index);
        if bit_scan_index < 0 || !self.matches_operation(bit_scan_index, Opcode::VFfblB32, Some(bit_index)) {
            return None;
        }
        let word_mask = single(&self.at(bit_scan_index)?.sources)?;
        let word_load_index = self.find_previous_definition(word_mask, bit_scan_index);
        if word_load_index < 0 || !self.matches_operation(word_load_index, Opcode::DsReadB32, Some(word_mask)) {
            return None;
        }
        match &self.at(word_load_index)?.control {
            Some(Control::DataShare(d)) if !d.gds && d.offset0 == 0 && d.offset1 == 0 => {}
            _ => return None,
        }
        let offset_index = self.find_previous_definition(group_offset, addition_index);
        if offset_index < 0 || self.at(offset_index)?.sources.len() != 2 {
            return None;
        }
        let group_index = self.at(offset_index)?.sources[0];
        let group_scan_index = self.find_previous_definition(group_index, offset_index);
        if group_scan_index < 0
            || !self.matches_operation(group_scan_index, Opcode::SFF1I32B32, Some(group_index))
            || !self.matches_instruction(offset_index, Opcode::SLshlB32, Some(group_offset), &[group_index, literal(5)])
        {
            return None;
        }
        let summary_mask = single(&self.at(group_scan_index)?.sources)?;
        if summary_mask.kind != OperandKind::ScalarRegister
            || !self.matches_instruction(group_scan_index + 1, Opcode::VLshlrevB32, Some(word_mask), &[literal(2), group_index])
            || offset_index != group_scan_index + 2
            || word_load_index != group_scan_index + 3
            || !self.matches_instruction(word_load_index, Opcode::DsReadB32, Some(word_mask), &[word_mask])
        {
            return None;
        }

        let stores: Vec<i64> =
            (0..self.len() as i64).filter(|&i| self.instructions[i as usize].encoding == Encoding::Ds && i != word_load_index).collect();
        if stores.len() != 1 {
            return None;
        }
        let store_index = stores[0];
        let store = self.at(store_index)?;
        match &store.control {
            Some(Control::DataShare(d)) if store.opcode == Opcode::DsWrite2B32 && store.sources.len() == 3 && !d.gds && d.offset0 == 0 && d.offset1 == 1 => {}
            _ => return None,
        }
        if store_index < 14 || store_index + 12 >= group_scan_index {
            return None;
        }
        let (address, low_data, high_data) = (store.sources[0], store.sources[1], store.sources[2]);
        let word_counter = self.source_at(store_index - 3, 1);
        let saved_execution = self.at(store_index - 2).and_then(|i| single(&i.sources)).unwrap_or_default_operand();
        let inner_execution = self.at(store_index + 4).and_then(|i| single(&i.sources)).unwrap_or_default_operand();
        let next_word = self.destination(store_index - 8);
        let local_index = self.source_at(store_index - 7, 1);
        let predicate = self.source_at(store_index - 6, 1);
        let count = self.source_at(store_index - 11, 0);
        let record_index = self.source_at(store_index - 11, 1);
        let record_counter = self.destination(store_index + 11);
        if [word_counter, saved_execution, inner_execution, next_word, count, record_counter, summary_mask, first]
            .iter()
            .any(|o| o.kind != OperandKind::ScalarRegister)
            || [local_index, predicate, record_index, address, low_data, high_data].iter().any(|o| o.kind != OperandKind::VectorRegister)
        {
            return None;
        }
        if !have_disjoint_scalar_ranges(&[
            (word_counter, 1),
            (record_counter, 1),
            (summary_mask, 1),
            (first, 1),
            (count, 1),
            (saved_execution, 2),
            (inner_execution, 2),
            (next_word, 1),
            (COMPARISON, 2),
        ]) || [address, low_data, high_data, predicate, record_index].contains(&local_index)
        {
            return None;
        }
        if !self.matches_mask_store_and_summary_update(
            store_index,
            address,
            low_data,
            high_data,
            word_counter,
            next_word,
            saved_execution,
            inner_execution,
            summary_mask,
            local_index,
            predicate,
            count,
            record_index,
            record_counter,
        ) {
            return None;
        }

        let loop_start = self.branch_target_index(store_index + 12);
        if loop_start < 0
            || loop_start + 6 >= store_index - 13
            || !self.matches_instruction(loop_start, Opcode::VAddI32, Some(record_index), &[record_counter, local_index])
            || !self.matches_instruction(loop_start + 1, Opcode::SCmpLtU32, None, &[record_counter, count])
            || !self.matches_instruction(loop_start + 2, Opcode::VCmpGtU32, None, &[count, record_index])
            || !self.matches_branch(loop_start + 3, Opcode::SCbranchScc0, store_index + 13)
            || !self.matches_instruction(loop_start + 4, Opcode::SAndSaveexecB64, Some(saved_execution), &[COMPARISON])
            || !self.matches_branch(loop_start + 5, Opcode::SCbranchExecz, store_index - 12)
        {
            return None;
        }

        let initialized = [word_counter, record_counter, summary_mask];
        let initialization: Vec<i64> = initialized.iter().map(|&r| self.find_previous_definition(r, loop_start)).collect();
        if initialization.iter().any(|&i| i < 0)
            || !self.matches_instruction(initialization[0], Opcode::SMovB32, Some(word_counter), &[literal(0)])
            || !self.matches_instruction(initialization[1], Opcode::SMovB32, Some(record_counter), &[literal(0)])
            || !self.matches_instruction(initialization[2], Opcode::SMovB32, Some(summary_mask), &[literal(0)])
        {
            return None;
        }

        let local_definition = self.find_previous_definition(local_index, loop_start);
        if local_definition < 0 {
            return None;
        }
        let local = self.at(local_definition)?;
        if local.opcode != Opcode::VLshlAddU32
            || local.sources.len() != 3
            || !has_unmodified_operands(local)
            || local.sources[0] != Operand::vector(1)
            || local.sources[2] != Operand::vector(0)
        {
            return None;
        }
        let shift = try_get_constant(local.sources[1])?;
        if shift > 6
            || !self.dominates_instruction(local_definition, loop_start)
            || (local_definition..self.len() as i64)
                .any(|i| is_branch(self.instructions[i as usize]) && self.branch_target_index(i) <= local_definition)
            || self.instructions[..local_definition as usize]
                .iter()
                .any(|i| bounds_writes_register(i, Operand::vector(0)) || bounds_writes_register(i, Operand::vector(1)))
            || self.has_register_write(local_index, local_definition + 1, store_index - 7)
        {
            return None;
        }

        let min_initialization = *initialization.iter().min().unwrap();
        if !self.has_complete_entry_execution_mask(loop_start)
            || !self.verify_mask_construction_body(
                loop_start,
                store_index,
                saved_execution,
                inner_execution,
                record_index,
                &[word_counter, record_counter, summary_mask, first, count, local_index],
            )
            || !self.has_only_allowed_entries(min_initialization, store_index + 13, &[min_initialization])
            || initialization.iter().enumerate().any(|(register, &index)| {
                self.has_register_write(initialized[register], index + 1, loop_start) || !self.dominates_instruction(index, group_scan_index)
            })
        {
            return None;
        }

        if !self.verify_mask_consumption(
            group_scan_index,
            bit_scan_index,
            addition_index,
            selector_index,
            store_index,
            summary_mask,
            group_index,
            group_offset,
            word_mask,
            bit_index,
            first,
        ) {
            return None;
        }
        let first_value = self.runtime_read(first, loop_start)?;
        let count_value = self.runtime_read(count, loop_start)?;
        if self.has_register_write(first, loop_start, selector_index + 1) || self.has_register_write(count, loop_start, store_index + 13) {
            return None;
        }
        Some(WaveMaskSelectorBounds { first_record: first_value, record_count: count_value, row_shift: shift })
    }

    #[allow(clippy::too_many_arguments)]
    fn matches_mask_store_and_summary_update(
        &self,
        store: i64,
        address: Operand,
        low_data: Operand,
        high_data: Operand,
        word_counter: Operand,
        next_word: Operand,
        saved_execution: Operand,
        inner_execution: Operand,
        summary_mask: Operand,
        local_index: Operand,
        predicate: Operand,
        count: Operand,
        record_index: Operand,
        record_counter: Operand,
    ) -> bool {
        use Opcode::*;
        let s107 = Operand::scalar(107);
        let saved_high = Operand::scalar(saved_execution.value + 1);
        self.matches_instruction(store - 13, SMovB64, Some(EXECUTION), &[inner_execution])
            && self.matches_instruction(store - 12, SAndn2B64, Some(EXECUTION), &[saved_execution, EXECUTION])
            && self.matches_instruction(store - 11, VCmpGtU32, None, &[count, record_index])
            && self.matches_instruction(store - 10, VCndmaskB32, Some(predicate), &[literal(0), literal(1)])
            && self.matches_instruction(store - 9, SMovB64, Some(EXECUTION), &[saved_execution])
            && self.matches_instruction(store - 8, SAddI32, Some(next_word), &[word_counter, literal(1)])
            && self.matches_instruction(store - 7, VCmpEqU32, None, &[literal(0), local_index])
            && self.compare_destination(store - 6, saved_execution)
            && self.matches_instruction(store - 6, VCmpNeU32, Some(saved_execution), &[literal(0), predicate])
            && self.matches_instruction(store - 5, SAndSaveexecB64, Some(inner_execution), &[COMPARISON])
            && self.matches_branch(store - 4, SCbranchExecz, store + 1)
            && self.matches_instruction(store - 3, VLshlrevB32, Some(address), &[literal(2), word_counter])
            && self.matches_instruction(store - 2, VMovB32, Some(low_data), &[saved_execution])
            && self.matches_instruction(store - 1, VMovB32, Some(high_data), &[saved_high])
            && self.matches_instruction(store + 1, SMovB32, Some(COMPARISON), &[summary_mask])
            && self.matches_instruction(store + 2, SCmpLgU32, None, &[saved_execution, literal(0)])
            && self.matches_instruction(store + 3, SBitset1B32, Some(COMPARISON), &[word_counter])
            && self.matches_instruction(store + 4, SMovB64, Some(EXECUTION), &[inner_execution])
            && self.matches_instruction(store + 5, SCselectB32, Some(s107), &[COMPARISON, summary_mask])
            && self.matches_instruction(store + 6, SCmpLgU32, None, &[saved_high, literal(0)])
            && self.matches_instruction(store + 7, SMovB32, Some(COMPARISON), &[s107])
            && self.matches_instruction(store + 8, SBitset1B32, Some(COMPARISON), &[next_word])
            && self.matches_instruction(store + 9, SCselectB32, Some(summary_mask), &[COMPARISON, s107])
            && self.matches_instruction(store + 10, SAddI32, Some(word_counter), &[word_counter, literal(2)])
            && self.matches_instruction(store + 11, SAddI32, Some(record_counter), &[record_counter, literal(64)])
            && self.matches_operation(store + 12, SBranch, None)
    }

    fn verify_mask_construction_body(
        &self,
        start: i64,
        store: i64,
        saved_execution: Operand,
        inner_execution: Operand,
        record_index: Operand,
        preserved: &[Operand],
    ) -> bool {
        let inner_save: Vec<i64> = (start + 6..store - 13)
            .filter(|&i| self.matches_instruction(i, Opcode::SAndSaveexecB64, Some(inner_execution), &[COMPARISON]))
            .collect();
        if inner_save.len() != 1 || !self.matches_branch(inner_save[0] + 1, Opcode::SCbranchExecz, store - 13) {
            return false;
        }
        // Work outside the range keeps its original index until the false
        // predicate is written.
        for index in start + 6..store - 13 {
            let instruction = self.instructions[index as usize];
            if preserved.iter().any(|&r| bounds_writes_register(instruction, r))
                || bounds_writes_register(instruction, saved_execution)
                || bounds_writes_register(instruction, Operand::scalar(saved_execution.value + 1))
            {
                return false;
            }
            if index == inner_save[0] || index == inner_save[0] + 1 {
                continue;
            }
            let name = instruction.opcode.name();
            if bounds_writes_register(instruction, inner_execution)
                || bounds_writes_register(instruction, Operand::scalar(inner_execution.value + 1))
                || changes_execution(instruction)
                || is_branch(instruction)
                || name.contains("Store")
                || name.contains("Atomic")
            {
                return false;
            }
        }
        !self.has_register_write(record_index, start + 1, start + 6)
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_mask_consumption(
        &self,
        scan: i64,
        bit_scan: i64,
        addition: i64,
        selector: i64,
        store: i64,
        summary: Operand,
        group: Operand,
        group_offset: Operand,
        mask: Operand,
        bit: Operand,
        first: Operand,
    ) -> bool {
        use Opcode::*;
        let guard = scan - 5;
        if guard < 0
            || !self.matches_instruction(guard, SCmpLgU32, None, &[literal(0), summary])
            || !self.matches_operation(guard + 1, SCbranchScc0, None)
            || !self.matches_operation(scan - 3, VCmpGtF32, None)
            || !self.matches_instruction(scan - 2, SAndB64, Some(COMPARISON), &[COMPARISON, EXECUTION])
            || self.branch_target_index(scan - 1) != self.branch_target_index(guard + 1)
            || !self.matches_operation(scan - 1, SCbranchScc0, None)
            || !self.matches_instruction(scan + 4, VCmpNeU32, None, &[literal(0), mask])
            || !self.matches_operation(scan + 5, SCbranchVccz, None)
            || bit_scan != scan + 7
            || !self.matches_operation(scan + 6, VCmpGtF32, None)
            || addition != bit_scan + 3
        {
            return false;
        }
        let saved_execution = self.destination(bit_scan + 1);
        if !self.matches_instruction(bit_scan + 1, SAndSaveexecB64, Some(saved_execution), &[COMPARISON])
            || !self.matches_operation(bit_scan + 2, SCbranchExecz, None)
        {
            return false;
        }
        let outer_tail = self.branch_target_index(scan + 5);
        let inner_tail = outer_tail - 4;
        if inner_tail <= selector
            || !self.matches_branch(bit_scan + 2, SCbranchExecz, inner_tail)
            || !self.matches_instruction(inner_tail, SMovB64, Some(EXECUTION), &[saved_execution])
        {
            return false;
        }
        let clear_word = self.destination(inner_tail + 1);
        let clear_summary = self.destination(outer_tail);
        if !self.matches_instruction(inner_tail + 1, VLshlrevB32, Some(clear_word), &[bit, literal(1)])
            || !self.matches_instruction(inner_tail + 2, VXorB32, Some(mask), &[clear_word, mask])
            || !self.matches_branch(inner_tail + 3, SBranch, scan + 4)
            || !self.matches_instruction(outer_tail, SLshlB32, Some(clear_summary), &[literal(1), group])
            || !self.matches_instruction(outer_tail + 1, SXorB32, Some(summary), &[clear_summary, summary])
            || !self.matches_branch(outer_tail + 2, SBranch, guard)
            || !self.matches_branch(guard + 1, SCbranchScc0, outer_tail + 3)
        {
            return false;
        }
        let addition_destination = match self.at(addition).and_then(|i| i.destinations.first().copied()) {
            Some(d) => d,
            None => return false,
        };
        if self.has_register_write(summary, store + 10, outer_tail + 1)
            || self.has_register_write(group, scan + 1, outer_tail)
            || self.has_register_write(group_offset, scan + 3, addition)
            || self.has_register_write(mask, scan + 4, inner_tail + 2)
            || self.has_register_write(bit, bit_scan + 1, inner_tail + 1)
            || self.has_register_write(addition_destination, addition + 1, selector)
            || self.has_register_write(first, store + 13, outer_tail + 3)
        {
            return false;
        }
        // Nested execution masks may shrink the wave but must restore the
        // saved mask.
        if !have_disjoint_scalar_ranges(&[(summary, 1), (group, 1), (group_offset, 1), (first, 1), (saved_execution, 2), (COMPARISON, 2)])
            || mask == bit
            || mask == addition_destination
            || bit == addition_destination
        {
            return false;
        }
        let mut stack: Vec<Operand> = Vec::new();
        let mut scopes: HashMap<i64, Vec<Operand>> = HashMap::new();
        for index in addition + 1..inner_tail {
            scopes.insert(index, stack.iter().rev().copied().collect());
            let instruction = self.instructions[index as usize];
            if instruction.opcode == SAndSaveexecB64 && has_unmodified_operands(instruction) {
                if instruction.sources.len() != 1 || instruction.sources[0] != COMPARISON {
                    return false;
                }
                let saved = self.destination(index);
                let mut ranges = vec![(saved_execution, 2), (COMPARISON, 2), (saved, 2)];
                ranges.extend(stack.iter().map(|&p| (p, 2)));
                if !have_disjoint_scalar_ranges(&ranges) {
                    return false;
                }
                stack.push(saved);
            } else if !stack.is_empty() && self.matches_instruction(index, SMovB64, Some(EXECUTION), &[*stack.last().unwrap()]) {
                stack.pop();
            } else if changes_execution(instruction)
                || bounds_writes_register(instruction, saved_execution)
                || bounds_writes_register(instruction, Operand::scalar(saved_execution.value + 1))
                || stack
                    .iter()
                    .any(|&s| bounds_writes_register(instruction, s) || bounds_writes_register(instruction, Operand::scalar(s.value + 1)))
            {
                return false;
            }
        }
        scopes.insert(inner_tail, Vec::new());
        for (&index, scope) in &scopes {
            if !is_branch(self.instructions[index as usize]) {
                continue;
            }
            let target = self.branch_target_index(index);
            match scopes.get(&target) {
                Some(target_scope) if target_scope == scope => {}
                _ => return false,
            }
        }
        stack.is_empty()
            && self.has_only_allowed_entries(guard, outer_tail + 3, &[guard])
            && self.has_only_allowed_entries(addition + 1, inner_tail, &[addition + 1])
            && !self.has_register_write(addition_destination, addition + 1, inner_tail)
            && self.instructions[..(outer_tail + 3).clamp(0, self.len() as i64) as usize].iter().all(|i| {
                let name = i.opcode.name();
                !name.contains("Store") && !name.contains("Atomic")
            })
    }

    fn runtime_read(&self, register: Operand, before: i64) -> Option<ValueId> {
        let index = self.find_previous_definition(register, before);
        if index < 0 {
            return None;
        }
        let instruction = self.instructions[index as usize];
        if instruction.encoding != Encoding::Smem || !self.dominates_instruction(index, before) {
            return None;
        }
        let component = instruction.destinations.iter().position(|d| *d == register)?;
        let memory = self.plan.memory().index_of(instruction.pc, component as u32)?;
        let read = self.plan.graph.accesses[memory].and_then(|a| a.read)?;
        self.plan.validate_runtime_value(read).then_some(read)
    }

    fn dominates_instruction(&self, definition: i64, target: i64) -> bool {
        let mut pending = vec![0i64];
        let mut visited = HashSet::new();
        while let Some(index) = pending.pop() {
            if index == definition || index < 0 || index >= self.len() as i64 || !visited.insert(index) {
                continue;
            }
            if index == target {
                return false;
            }
            let instruction = self.instructions[index as usize];
            if instruction.opcode == Opcode::SEndpgm {
                continue;
            }
            if is_branch(instruction) {
                pending.push(self.branch_target_index(index));
            }
            if instruction.opcode != Opcode::SBranch {
                pending.push(index + 1);
            }
        }
        true
    }

    /// Before the producer, only saves and restores of the complete entry
    /// mask are allowed.
    fn has_complete_entry_execution_mask(&self, target: i64) -> bool {
        let mut saved: HashMap<Operand, i64> = HashMap::new();
        for index in 0..target {
            let instruction = self.instructions[index as usize];
            if instruction.opcode == Opcode::SOrn2SaveexecB64 && has_unmodified_operands(instruction) && instruction.sources.len() == 1 {
                let source = instruction.sources[0];
                let from_entry = source == EXECUTION || saved.get(&source).is_some_and(|&s| self.dominates_instruction(s, index));
                if from_entry {
                    saved.insert(self.destination(index), index);
                    continue;
                }
            }
            if instruction.opcode == Opcode::SMovB64
                && instruction.sources.len() == 1
                && instruction.destinations == [EXECUTION]
                && saved.get(&instruction.sources[0]).is_some_and(|&s| self.dominates_instruction(s, index))
            {
                continue;
            }
            if changes_execution(instruction) {
                return false;
            }
            saved.retain(|&r, _| !bounds_writes_register(instruction, r) && !bounds_writes_register(instruction, Operand::scalar(r.value + 1)));
        }
        true
    }

    fn has_only_allowed_entries(&self, start: i64, end: i64, allowed: &[i64]) -> bool {
        for index in 0..self.len() as i64 {
            if !is_branch(self.instructions[index as usize]) || (index >= start && index < end) {
                continue;
            }
            let target = self.branch_target_index(index);
            if target >= start && target < end && !allowed.contains(&target) {
                return false;
            }
        }
        true
    }

    fn find_previous_definition(&self, register: Operand, before: i64) -> i64 {
        let mut index = before - 1;
        while index >= 0 {
            if bounds_writes_register(self.instructions[index as usize], register) {
                return index;
            }
            index -= 1;
        }
        -1
    }

    fn has_register_write(&self, register: Operand, start: i64, end: i64) -> bool {
        let start = start.max(0);
        (start..end.min(self.len() as i64)).any(|i| bounds_writes_register(self.instructions[i as usize], register))
    }

    fn at(&self, index: i64) -> Option<&'p Instruction> {
        (index >= 0 && (index as usize) < self.len()).then(|| self.instructions[index as usize])
    }

    fn destination(&self, index: i64) -> Operand {
        self.at(index).and_then(|i| i.destinations.first().copied()).unwrap_or_default_operand()
    }

    fn source_at(&self, index: i64, source: usize) -> Operand {
        self.at(index).and_then(|i| i.sources.get(source).copied()).unwrap_or_default_operand()
    }

    fn matches_operation(&self, index: i64, opcode: Opcode, destination: Option<Operand>) -> bool {
        let Some(instruction) = self.at(index) else { return false };
        instruction.opcode == opcode
            && has_unmodified_operands(instruction)
            && match destination {
                None => instruction.destinations.is_empty(),
                Some(d) => instruction.destinations.contains(&d) || self.compare_destination(index, d),
            }
    }

    fn matches_instruction(&self, index: i64, opcode: Opcode, destination: Option<Operand>, sources: &[Operand]) -> bool {
        self.matches_operation(index, opcode, destination) && {
            let instruction = self.instructions[index as usize];
            instruction.sources.len() == sources.len()
                && instruction.sources.iter().zip(sources).all(|(&a, &b)| {
                    a == b
                        || match (try_get_constant(a), try_get_constant(b)) {
                            (Some(x), Some(y)) => x == y,
                            _ => false,
                        }
                })
        }
    }

    fn compare_destination(&self, index: i64, register: Operand) -> bool {
        matches!(self.at(index).map(|i| &i.control), Some(Some(Control::Sdwa(SdwaControl { scalar_destination: Some(d), .. }))) if register == Operand::scalar(*d))
    }

    fn matches_branch(&self, index: i64, opcode: Opcode, target: i64) -> bool {
        self.matches_operation(index, opcode, None) && self.branch_target_index(index) == target
    }

    fn branch_target_index(&self, index: i64) -> i64 {
        let Some(instruction) = self.at(index) else { return -1 };
        if !is_branch(instruction) {
            return -1;
        }
        let target = instruction.pc.wrapping_add(4).wrapping_add(((instruction.words[0] & 0xFFFF) as u16 as i16 as i32 * 4) as u32);
        // A branch may target a wait that was removed from the sequence.
        self.instructions.iter().position(|c| c.pc >= target).map_or(-1, |p| p as i64)
    }
}

trait OperandDefault {
    fn unwrap_or_default_operand(self) -> Operand;
}

impl OperandDefault for Option<Operand> {
    /// The zero operand (`s0`) a missing slot reads as.
    fn unwrap_or_default_operand(self) -> Operand {
        self.unwrap_or(Operand::scalar(0))
    }
}

fn single(sources: &[Operand]) -> Option<Operand> {
    (sources.len() == 1).then(|| sources[0])
}

fn is_branch(instruction: &Instruction) -> bool {
    instruction.opcode == Opcode::SBranch || instruction.opcode.name().starts_with("SCbranch")
}

fn changes_execution(instruction: &Instruction) -> bool {
    let name = instruction.opcode.name();
    bounds_writes_register(instruction, EXECUTION)
        || bounds_writes_register(instruction, Operand::scalar(127))
        || name.contains("Saveexec")
        || name.contains("Wrexec")
        || name.starts_with("VCmpx")
}

fn bounds_writes_register(instruction: &Instruction, register: Operand) -> bool {
    let scalar_destination = instruction.control.as_ref().and_then(|c| c.scalar_destination());
    if register.kind == OperandKind::ScalarRegister {
        if let Some(s) = scalar_destination {
            if register.value >= s && register.value - s < 2 {
                return true;
            }
        }
        if (register.value == 106 || register.value == 107) && instruction.opcode.is_vector_compare() && scalar_destination.is_none() {
            return true;
        }
    }
    let wide = instruction.opcode.name().contains("64");
    instruction.destinations.iter().any(|d| *d == register || (d.kind == register.kind && wide && register.value == d.value + 1))
}

fn have_disjoint_scalar_ranges(ranges: &[(Operand, u32)]) -> bool {
    let mut seen = HashSet::new();
    for &(register, width) in ranges {
        for component in 0..width {
            let r = register.value + component;
            if r >= 126 || !seen.insert(r) {
                return false;
            }
        }
    }
    true
}

fn try_get_constant(operand: Operand) -> Option<u32> {
    match operand.kind {
        OperandKind::LiteralConstant => Some(operand.value),
        OperandKind::EncodedConstant => inline_constants::decode(operand.value),
        _ => None,
    }
}

fn has_unmodified_operands(instruction: &Instruction) -> bool {
    match &instruction.control {
        Some(Control::Vop3(c)) => c.absolute_mask == 0 && c.negate_mask == 0 && !c.clamp && c.output_modifier == 0 && c.operand_select == 0,
        Some(Control::Sdwa(c)) => {
            c.destination_select == 6
                && c.source0_select == 6
                && c.source1_select == 6
                && !c.source0_sign_extend
                && !c.source1_sign_extend
                && c.absolute_mask == 0
                && c.negate_mask == 0
                && !c.clamp
                && c.output_modifier == 0
        }
        Some(Control::Dpp(_) | Control::Dpp8(_) | Control::Vop3p(_)) => false,
        _ => true,
    }
}
