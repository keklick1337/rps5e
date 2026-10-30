// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Per-draw evaluation of graph values against the draw's user data, shader
//! base and guest memory.

use super::plan::ShaderResourcePlan;
use super::value::{evaluate_operation, ValueId, ValueKind};
use crate::inline_constants;
use std::cell::RefCell;
use std::collections::HashMap;

/// Device addresses carry 48 bits; upper handle bits are ignored.
pub const ADDRESS_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

/// Reads one guest dword (little-endian, 4-byte aligned address); `None`
/// when the address cannot be read.
pub type WordReader<'a> = &'a dyn Fn(u64) -> Option<u32>;

/// Copies guest bytes only when they are already resident on the CPU side
/// (`clean`: refuse memory the GPU may still own). Never synchronises.
pub type ResidentReader<'a> = &'a dyn Fn(u64, &mut [u8], bool) -> bool;
/// Whether the word at an address is produced by GPU work recorded earlier
/// and not yet complete (see [`ResourceRuntimeInputs::gpu_written`]).
pub type GpuWrittenProbe<'a> = &'a dyn Fn(u64) -> bool;

/// Compute-dispatch facts the wave-mask selector proof needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComputeSelectorState {
    pub wave_size: u32,
    pub threads_x: u32,
    pub threads_y: u32,
    pub threads_z: u32,
    pub has_partial_workgroups: bool,
    pub local_data_share_dwords: u32,
    pub local_invocation_id_components: i32,
}

/// What one draw supplies to materialise a plan.
#[derive(Clone, Copy)]
pub struct ResourceRuntimeInputs<'a> {
    /// User-data SGPR values from the plan's user-data base on.
    pub user_data: &'a [u32],
    /// Guest address of the program entry (S_GETPC_B64 base).
    pub shader_base: u64,
    /// General guest memory reader.
    pub read_memory: Option<WordReader<'a>>,
    /// Reader that refuses memory the GPU may still own; required for
    /// indirect image tables and resource-branch predicates.
    pub read_clean_memory: Option<WordReader<'a>>,
    pub compute_state: Option<ComputeSelectorState>,
    /// Told `true` before the flattened table's words are evaluated and
    /// `false` after.
    pub table_phase: Option<&'a dyn Fn(bool)>,
    pub read_resident_memory: Option<ResidentReader<'a>>,
    pub reads_clean: bool,
    /// Words GPU work recorded earlier is still producing. A flattened-table
    /// slot that only copies such a word (its value feeds no descriptor,
    /// range or predicate) is not read on the CPU: the snapshot lists it in
    /// `deferred_table_reads` and the host fills the slot on the GPU, after
    /// that work, instead of waiting for it.
    pub gpu_written: Option<GpuWrittenProbe<'a>>,
    /// The room a buffer binding has from its base when its NUM_RECORDS is
    /// a word GPU work is still writing and the program reads that word
    /// itself (see `ReplayProgram::run_into_with`).
    pub records_room: Option<super::replay::RecordsRoom<'a>>,
}

impl<'a> ResourceRuntimeInputs<'a> {
    pub fn new(user_data: &'a [u32], shader_base: u64) -> Self {
        ResourceRuntimeInputs {
            user_data,
            shader_base,
            read_memory: None,
            read_clean_memory: None,
            compute_state: None,
            table_phase: None,
            read_resident_memory: None,
            reads_clean: false,
            gpu_written: None,
            records_room: None,
        }
    }

    /// These inputs with `reader` as the general reader.
    pub fn with_reader(&self, reader: Option<WordReader<'a>>) -> Self {
        let same = match (reader, self.read_clean_memory) {
            (Some(a), Some(b)) => std::ptr::addr_eq(a as *const _, b as *const _),
            (None, None) => true,
            _ => false,
        };
        ResourceRuntimeInputs { read_memory: reader, reads_clean: self.reads_clean || same, ..*self }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RawAddress {
    Read(u64),
    Zero,
    Failed,
}

/// The address of a raw scalar read: the immediate and dynamic offsets
/// added to the 48-bit handle base; a buffer read is checked against its
/// record count.
///
/// A buffer word follows the translated `s_buffer_load`: the immediate (of
/// this dword) and SOFFSET are dword-aligned each and added in 32 bits; a
/// sum that carries out of 32 bits, or a dword not wholly inside
/// `stride * records` bytes (`records` bytes with stride 0), reads 0.
pub(super) fn resolve_raw_address(kind: ValueKind, base: u64, high: u64, records: u64, immediate: i64, offset: u32) -> RawAddress {
    if kind == ValueKind::ScalarBufferWord {
        let (sum, carry) = (offset & !3).overflowing_add(immediate as u32 & !3);
        let stride = ((high as u32) >> 16) & 0x3FFF;
        let size = if stride == 0 { records as u32 as u64 } else { stride as u64 * records as u32 as u64 };
        if carry || sum as u64 + 4 > size {
            return RawAddress::Zero;
        }
        return RawAddress::Read((base & !3) + sum as u64);
    }
    let relative = (immediate & !3) + (offset & !3) as i64;
    match add_signed_address(base & !3, relative) {
        Some(a) => RawAddress::Read(a),
        None => RawAddress::Failed,
    }
}

fn add_signed_address(base: u64, offset: i64) -> Option<u64> {
    if base > ADDRESS_MASK {
        return None;
    }
    if offset < 0 {
        let magnitude = offset.unsigned_abs();
        (magnitude <= base).then(|| base - magnitude)
    } else {
        let forward = offset as u64;
        (forward <= ADDRESS_MASK - base).then(|| base + forward)
    }
}

/// Evaluates graph values against one draw's inputs. Results are memoised,
/// so values shared by several descriptors are computed once.
pub struct RuntimeValueEvaluator<'a> {
    plan: &'a ShaderResourcePlan,
    inputs: ResourceRuntimeInputs<'a>,
    clean_flat_slots: &'a [u8],
    clean: Option<&'a RuntimeValueEvaluator<'a>>,
    active_mask: Option<ValueId>,
    cache: RefCell<HashMap<ValueId, u64>>,
    visiting: RefCell<Vec<ValueId>>,
}

impl<'a> RuntimeValueEvaluator<'a> {
    pub fn new(plan: &'a ShaderResourcePlan, inputs: ResourceRuntimeInputs<'a>) -> Self {
        Self::with(plan, inputs, &[], None, None)
    }

    /// An evaluator whose flattened slots marked in `clean_flat_slots` are
    /// read through `clean`.
    pub fn with(
        plan: &'a ShaderResourcePlan,
        inputs: ResourceRuntimeInputs<'a>,
        clean_flat_slots: &'a [u8],
        clean: Option<&'a RuntimeValueEvaluator<'a>>,
        active_mask: Option<ValueId>,
    ) -> Self {
        RuntimeValueEvaluator {
            plan,
            inputs,
            clean_flat_slots,
            clean,
            active_mask,
            cache: RefCell::new(HashMap::new()),
            visiting: RefCell::new(Vec::new()),
        }
    }

    pub fn inputs(&self) -> &ResourceRuntimeInputs<'a> {
        &self.inputs
    }

    /// The low dword of a value.
    pub fn evaluate(&self, value: ValueId) -> Option<u32> {
        self.evaluate_wide(value).map(|v| v as u32)
    }

    pub fn evaluate_wide(&self, value: ValueId) -> Option<u64> {
        let graph = &self.plan.graph;
        let node = graph.node(value);
        if node.is_constant() {
            return Some(node.payload);
        }
        if node.kind == ValueKind::Select && self.active_mask == Some(node.operands[0]) {
            return self.evaluate_wide(node.operands[1]);
        }
        if let Some(&v) = self.cache.borrow().get(&value) {
            return Some(v);
        }
        if self.visiting.borrow().contains(&value) {
            return None;
        }
        self.visiting.borrow_mut().push(value);
        let result = self.evaluate_node(value);
        self.visiting.borrow_mut().pop();
        let result = result?;
        self.cache.borrow_mut().insert(value, result);
        Some(result)
    }

    fn evaluate_node(&self, value: ValueId) -> Option<u64> {
        let graph = &self.plan.graph;
        let node = graph.node(value);
        match node.kind {
            ValueKind::Undefined => None,
            ValueKind::MemoryAperture => Some(inline_constants::decode_aperture64(node.payload as u32) >> 32),
            ValueKind::UserData => {
                let register = node.payload as u32;
                let base = graph.user_data_base;
                if register < base || (register - base) as usize >= self.inputs.user_data.len() {
                    return None;
                }
                Some(self.inputs.user_data[(register - base) as usize] as u64)
            }
            ValueKind::ShaderBase => Some(self.inputs.shader_base),
            ValueKind::Phi => {
                let invariant = graph.resolve_invariant_phi(value)?;
                self.evaluate_wide(invariant)
            }
            ValueKind::FirstLane => {
                let nested = RuntimeValueEvaluator::with(self.plan, self.inputs, self.clean_flat_slots, self.clean, Some(node.operands[1]));
                nested.evaluate_wide(node.operands[0])
            }
            ValueKind::ResourceTableWord => self.evaluate_table_reference(node.payload as usize),
            ValueKind::ScalarAddressWord | ValueKind::ScalarBufferWord => self.evaluate_raw_read(value),
            ValueKind::Select => {
                let condition = self.evaluate_wide(node.operands[0])?;
                let when_true = self.evaluate_wide(node.operands[1])?;
                let when_false = self.evaluate_wide(node.operands[2])?;
                Some(if condition != 0 { when_true } else { when_false })
            }
            ValueKind::Operation => {
                if !node.op.is_uniform() {
                    return None;
                }
                let mut operands = [0u64; 4];
                for (i, &o) in node.operands.iter().enumerate().take(4) {
                    operands[i] = self.evaluate_wide(o)?;
                }
                evaluate_operation(node.op, &operands[..node.operands.len().min(4)])
            }
            _ => None,
        }
    }

    fn evaluate_raw_read(&self, value: ValueId) -> Option<u64> {
        let graph = &self.plan.graph;
        let node = graph.node(value);
        let index = node.memory_index();
        if index >= graph.memory.len() {
            return None;
        }
        let immediate = graph.memory.get(index).offset as i32 as i64;
        let handle = graph.node(node.operands[0]);
        if handle.operands.len() < 2 {
            return None;
        }
        let low = self.evaluate_wide(handle.operands[0])?;
        let high = self.evaluate_wide(handle.operands[1])?;
        let offset = self.evaluate_wide(node.operands[1])?;
        let mut records = 0;
        if node.kind == ValueKind::ScalarBufferWord {
            if handle.operands.len() != 4 {
                return None;
            }
            records = self.evaluate_wide(handle.operands[2])?;
            self.evaluate_wide(handle.operands[3])?;
        }
        self.read_raw_word(node.kind, low, high, offset as u32, records, immediate)
    }

    pub(super) fn read_raw_word(&self, kind: ValueKind, low: u64, high: u64, offset: u32, records: u64, immediate: i64) -> Option<u64> {
        let base = ((high << 32) | (low as u32 as u64)) & ADDRESS_MASK;
        let address = match resolve_raw_address(kind, base, high, records, immediate, offset) {
            RawAddress::Failed => return None,
            RawAddress::Zero => return Some(0),
            RawAddress::Read(a) => a,
        };
        match self.inputs.read_memory.and_then(|read| read(address)) {
            Some(word) => Some(word as u64),
            // Every read is evaluated up front, including ones in branches
            // the shader skips when a pointer is null: a load through a null
            // base cannot execute on hardware, so its value is never used.
            None if base & !3 == 0 => Some(0),
            None => None,
        }
    }

    /// The address a raw read node reads (None: not a raw read, or its
    /// address does not evaluate to a plain read).
    fn raw_read_address(&self, value: ValueId) -> Option<u64> {
        let graph = &self.plan.graph;
        let node = graph.node(value);
        if !matches!(node.kind, ValueKind::ScalarAddressWord | ValueKind::ScalarBufferWord) {
            return None;
        }
        let index = node.memory_index();
        if index >= graph.memory.len() {
            return None;
        }
        let immediate = graph.memory.get(index).offset as i32 as i64;
        let handle = graph.node(node.operands[0]);
        if handle.operands.len() < 2 {
            return None;
        }
        let low = self.evaluate_wide(handle.operands[0])?;
        let high = self.evaluate_wide(handle.operands[1])?;
        let offset = self.evaluate_wide(node.operands[1])?;
        let mut records = 0;
        if node.kind == ValueKind::ScalarBufferWord {
            if handle.operands.len() != 4 {
                return None;
            }
            records = self.evaluate_wide(handle.operands[2])?;
        }
        let base = ((high << 32) | (low as u32 as u64)) & ADDRESS_MASK;
        match resolve_raw_address(node.kind, base, high, records, immediate, offset as u32) {
            RawAddress::Read(a) if base & !3 != 0 => Some(a),
            _ => None,
        }
    }

    /// A flattened-table word: deferred (`Err(address)`) when it is a raw
    /// read of a word GPU work is still producing and nothing evaluated
    /// it on the CPU before.
    fn evaluate_table_word(&self, value: ValueId) -> Option<Result<u32, u64>> {
        if let Some(probe) = self.inputs.gpu_written {
            if !self.cache.borrow().contains_key(&value) {
                if let Some(address) = self.raw_read_address(value) {
                    if probe(address) {
                        return Some(Err(address));
                    }
                }
            }
        }
        self.evaluate(value).map(Ok)
    }

    fn evaluate_table_reference(&self, slot: usize) -> Option<u64> {
        let read = self.plan.table_reads.get(slot)?;
        let selected = match self.clean {
            Some(clean) if self.clean_flat_slots.get(slot).is_some_and(|&c| c != 0) => clean,
            _ => self,
        };
        selected.evaluate_wide(read.value)
    }

    /// Dword `word` of descriptor source `source`.
    pub fn evaluate_source_word(&self, source: usize, word: usize) -> Option<u32> {
        self.evaluate(self.plan.descriptor_sources[source].dwords[word])
    }
}

/// Descriptor words of the requested sources plus (optionally) the
/// flattened table, evaluated in one memoised walk.
pub struct EvaluatedSources {
    pub descriptors: Vec<Vec<u32>>,
    pub table: Vec<u32>,
    /// Per descriptor source: reachable this draw (empty when every source
    /// is assumed active).
    pub active_sources: Vec<bool>,
    /// Table slots left zero for the GPU to fill: (slot, guest address).
    pub deferred: Vec<(u32, u64)>,
}

/// Evaluate `sources` and, when `evaluate_table`, the flattened table
/// (followed by `additional_table_words` zeroed words). `None` when any
/// requested word cannot be evaluated.
pub fn evaluate_sources(
    plan: &ShaderResourcePlan,
    sources: &[u32],
    inputs: &ResourceRuntimeInputs<'_>,
    clean_flat_slots: &[u8],
    evaluate_table: bool,
    additional_table_words: usize,
) -> Option<EvaluatedSources> {
    let any_clean = clean_flat_slots.iter().any(|&c| c != 0);
    if any_clean && inputs.read_clean_memory.is_none() {
        return None;
    }
    let clean_inputs = inputs.with_reader(inputs.read_clean_memory);
    let clean_evaluator = RuntimeValueEvaluator::new(plan, clean_inputs);
    let evaluator = RuntimeValueEvaluator::with(plan, *inputs, clean_flat_slots, Some(&clean_evaluator), None);
    let active_sources =
        if evaluate_table && !plan.resource_branches.is_empty() { evaluate_active_sources(plan, inputs, &clean_evaluator) } else { Vec::new() };
    let mut descriptors = Vec::with_capacity(sources.len());
    for &source_index in sources {
        let source = plan.descriptor_sources.get(source_index as usize)?;
        let mut words = vec![0u32; source.dwords.len()];
        if active_sources.is_empty() || active_sources[source_index as usize] {
            for (i, w) in words.iter_mut().enumerate() {
                *w = evaluator.evaluate_source_word(source_index as usize, i)?;
            }
        }
        descriptors.push(words);
    }
    let mut table = Vec::new();
    let mut deferred = Vec::new();
    if evaluate_table {
        table = vec![0u32; plan.table_reads.len() + additional_table_words];
        if let Some(phase) = inputs.table_phase {
            phase(true);
        }
        let mut ok = true;
        for read in &plan.table_reads {
            let clean = clean_flat_slots.get(read.flat_offset as usize).is_some_and(|&c| c != 0);
            let word = if clean { clean_evaluator.evaluate(read.value).map(Ok) } else { evaluator.evaluate_table_word(read.value) };
            match (read.flat_offset as usize) < plan.table_reads.len() {
                true => match word {
                    Some(Ok(word)) => table[read.flat_offset as usize] = word,
                    Some(Err(address)) => deferred.push((read.flat_offset, address)),
                    None => {
                        ok = false;
                        break;
                    }
                },
                false => {
                    ok = false;
                    break;
                }
            }
        }
        if let Some(phase) = inputs.table_phase {
            phase(false);
        }
        if !ok {
            return None;
        }
    }
    Some(EvaluatedSources { descriptors, table, active_sources, deferred })
}

/// The descriptor sources reachable through the plan's resource branches:
/// a branch whose predicate the clean reader can evaluate follows one edge,
/// an unreadable predicate keeps both.
fn evaluate_active_sources(plan: &ShaderResourcePlan, inputs: &ResourceRuntimeInputs<'_>, clean: &RuntimeValueEvaluator<'_>) -> Vec<bool> {
    let mut active = vec![true; plan.descriptor_sources.len()];
    for block in &plan.resource_branches {
        for &s in &block.sources {
            active[s as usize] = false;
        }
    }
    let mut visited = vec![false; plan.resource_branches.len()];
    let mut pending = vec![0usize];
    while let Some(index) = pending.pop() {
        if visited[index] {
            continue;
        }
        visited[index] = true;
        let block = &plan.resource_branches[index];
        for &s in &block.sources {
            active[s as usize] = true;
        }
        let decided = match block.condition {
            Some(c) if inputs.read_clean_memory.is_some() => clean.evaluate(c),
            _ => None,
        };
        match decided {
            Some(v) => pending.push(block.successors[if v != 0 { 0 } else { 1 }]),
            None => pending.extend(block.successors.iter().copied()),
        }
    }
    active
}

/// One descriptor source's words.
pub fn evaluate_descriptor_source(plan: &ShaderResourcePlan, source: u32, inputs: &ResourceRuntimeInputs<'_>) -> Option<Vec<u32>> {
    evaluate_sources(plan, &[source], inputs, &[], false, 0).map(|mut e| e.descriptors.remove(0))
}

/// The flattened resource table alone.
pub fn flatten_resource_table(plan: &ShaderResourcePlan, inputs: &ResourceRuntimeInputs<'_>) -> Option<Vec<u32>> {
    evaluate_sources(plan, &[], inputs, &[], true, 0).map(|e| e.table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_addresses() {
        // Scalar address reads: signed immediates, dword aligned.
        assert_eq!(resolve_raw_address(ValueKind::ScalarAddressWord, 0x1000, 0, 0, 8, 4), RawAddress::Read(0x100C));
        assert_eq!(resolve_raw_address(ValueKind::ScalarAddressWord, 0x1000, 0, 0, -8, 0), RawAddress::Read(0xFF8));
        assert_eq!(resolve_raw_address(ValueKind::ScalarAddressWord, 0x4, 0, 0, -8, 0), RawAddress::Failed);
        // Buffer reads are bounded by the record count (stride 0: bytes);
        // a dword out of range reads 0, as the translated load does.
        assert_eq!(resolve_raw_address(ValueKind::ScalarBufferWord, 0x2000, 0, 16, 12, 0), RawAddress::Read(0x200C));
        assert_eq!(resolve_raw_address(ValueKind::ScalarBufferWord, 0x2000, 0, 16, 16, 0), RawAddress::Zero);
        assert_eq!(resolve_raw_address(ValueKind::ScalarBufferWord, 0x2000, 0, 0, 16, 0), RawAddress::Zero);
        // Stride 8, 2 records = 16 bytes.
        assert_eq!(resolve_raw_address(ValueKind::ScalarBufferWord, 0x2000, 8 << 16, 2, 8, 4), RawAddress::Read(0x200C));
        // A negative immediate wraps past 2^32 (no SOFFSET) or carries out
        // of the SOFFSET sum: out of range.
        assert_eq!(resolve_raw_address(ValueKind::ScalarBufferWord, 0x2000, 0, 16, -4, 0), RawAddress::Zero);
        assert_eq!(resolve_raw_address(ValueKind::ScalarBufferWord, 0x2000, 0, 16, -4, 8), RawAddress::Zero);
        // The immediate and SOFFSET are aligned each before the sum.
        assert_eq!(resolve_raw_address(ValueKind::ScalarBufferWord, 0x2000, 0, 64, 6, 6), RawAddress::Read(0x2008));
        // A partly out-of-range x8 load: its last dwords read 0, the first
        // ones memory.
        assert_eq!(resolve_raw_address(ValueKind::ScalarBufferWord, 0x2000, 0, 24, 20, 0), RawAddress::Read(0x2014));
        assert_eq!(resolve_raw_address(ValueKind::ScalarBufferWord, 0x2000, 0, 24, 24, 0), RawAddress::Zero);
    }
}
