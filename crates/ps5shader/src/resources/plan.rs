// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The immutable resource analysis of one program.

use super::branches::{self, ResourceBranchBlock};
use super::candidates::BufferCandidateTablePlan;
use super::device_ranges::{self, DeviceAddressRangePlan};
use super::graph::{MemoryAccessBinding, ScalarValueGraph};
use super::info::*;
use super::memory::{MemoryAccessTable, MemoryResourceKind};
use super::table_reads;
use super::tracker;
use super::validator::RuntimeValueValidator;
use super::value::{ValueId, ValueKind};
use super::ResourcePlanError;
use crate::ir_types::Program;
use std::collections::{HashMap, HashSet};

/// An image access whose descriptor comes from a run-time key: the emitter
/// reads the key and selects among the materialised candidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectImageAccess {
    pub memory_index: usize,
    /// The scalar read whose value is the key.
    pub key: ValueId,
    pub heap_source: u32,
    /// The key is a byte offset of the descriptor (not a heap record index).
    pub key_is_address_offset: bool,
}

/// How a plan treats memory the host could read for the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlanOptions {
    /// Scalar reads of data (not descriptors) at host-evaluable addresses
    /// are read by the host into the flattened table. Off, the program
    /// reads them itself (guest memory resident for the device).
    pub flatten_data_reads: bool,
    /// Buffer reads whose V# the program can decode itself (all but
    /// untyped formatted loads, whose conversion depends on the V#'s format:
    /// decoding it at run time is a switch over every format per load) read
    /// the V# from their registers and the data through device addresses:
    /// the V# is neither materialised nor bound.
    pub buffers_on_device: bool,
}

impl Default for PlanOptions {
    fn default() -> Self {
        PlanOptions { flatten_data_reads: true, buffers_on_device: false }
    }
}

impl PlanOptions {
    /// Guest memory resident for the device: the program reads its data
    /// and buffer descriptors itself.
    pub fn guest_memory() -> Self {
        PlanOptions { flatten_data_reads: false, buffers_on_device: true }
    }

    /// `memory` is a buffer read that goes through device addresses under
    /// these options.
    pub fn device_buffer_read(&self, memory: &super::memory::MemoryAccessInfo) -> bool {
        use super::memory::{MemoryAccess, MemoryResourceKind as K};
        self.buffers_on_device
            && matches!(memory.kind, K::Buffer | K::ScalarBuffer)
            && memory.access == MemoryAccess::Read
            && !(memory.formatted && !memory.typed)
    }
}

/// Dwords each written device-address range owns in the flattened table:
/// base low, base high, size.
pub const WRITTEN_RANGE_DWORD_COUNT: usize = 3;

/// The immutable resource analysis of one program: value graph, descriptor
/// sources, flattened reads and dense tables. Built once by
/// [`ShaderResourcePlan::extract`], materialised per draw, free of any
/// draw's user data.
#[derive(Clone, Debug)]
pub struct ShaderResourcePlan {
    /// Process-unique identity (materialisation caches key on it).
    pub id: u64,
    pub graph: ScalarValueGraph,
    pub stage: ShaderStage,
    pub hash: u64,
    /// Descriptor sources; dense tables index into this.
    pub descriptor_sources: Vec<DescriptorSource>,
    /// Sources evaluated per draw, in snapshot order: every buffer, every
    /// non-indirect image, every sampler.
    pub materialization_sources: Vec<u32>,
    /// Per-block reachability of descriptor sources (empty unless some
    /// branch condition is host-evaluable).
    pub resource_branches: Vec<ResourceBranchBlock>,
    /// Host reads that fill the flattened resource table.
    pub table_reads: Vec<ResourceTableRead>,
    /// Memory index of a flattened scalar load → its table slot.
    pub flattened_slot_by_memory_index: HashMap<usize, u32>,
    /// Scalar reads with a run-time offset (performed by the device).
    pub dynamic_reads: Vec<ValueId>,
    /// Flattened slots an indirect table depends on; read with the clean
    /// reader.
    pub clean_flat_slots: Vec<u8>,
    pub indirect_images: Vec<IndirectImageAccess>,
    /// Memory indices of descriptor reads resolved by the planner (the T#
    /// dwords of run-time selected images; they feed only the image).
    pub suppressed_memory: HashSet<usize>,
    pub buffer_candidate_tables: Vec<BufferCandidateTablePlan>,
    /// Some image is selected at run time; materialisation needs a clean
    /// memory reader.
    pub requires_specialization_memory: bool,
    pub info: ShaderResourceInfo,
    /// The handles each memory access reads, after flattened reads were
    /// replaced by table words.
    pub accesses: Vec<Option<MemoryAccessBinding>>,
    /// Bounds of every device-address handle (optional analysis).
    pub device_address_ranges: Vec<DeviceAddressRangePlan>,
    /// Written handle → its first flattened slot (after the table reads).
    pub written_range_slot_by_handle: HashMap<u32, u32>,
    pub options: PlanOptions,
}

impl ShaderResourcePlan {
    /// Analyse `program`: build the value graph, flatten host-evaluable
    /// scalar reads, track descriptors into dense tables, and plan resource
    /// branches and device-address ranges.
    ///
    /// * `user_data_base` / `user_data_count`: the SGPRs initialised from
    ///   user data (graphics: base 8 for the GS/vertex stage, 0 for pixel;
    ///   compute: 0).
    /// * `fixed_function_vertex_loads`: pcs of vertex-fetch buffer loads the
    ///   host replaces by fixed-function attributes.
    /// * `wave_size`: 32 or 64 (graphics stages compile as wave32).
    pub fn extract(
        program: Program,
        stage: ShaderStage,
        hash: u64,
        user_data_base: u32,
        user_data_count: u32,
        fixed_function_vertex_loads: Option<&HashSet<u32>>,
        wave_size: u32,
    ) -> Result<ShaderResourcePlan, ResourcePlanError> {
        Self::extract_with(program, stage, hash, user_data_base, user_data_count, fixed_function_vertex_loads, wave_size, PlanOptions::default())
    }

    /// [`Self::extract`] with explicit [`PlanOptions`].
    #[allow(clippy::too_many_arguments)]
    pub fn extract_with(
        program: Program,
        stage: ShaderStage,
        hash: u64,
        user_data_base: u32,
        user_data_count: u32,
        fixed_function_vertex_loads: Option<&HashSet<u32>>,
        wave_size: u32,
        options: PlanOptions,
    ) -> Result<ShaderResourcePlan, ResourcePlanError> {
        let graph = ScalarValueGraph::build(program, user_data_base, user_data_count, fixed_function_vertex_loads, wave_size)?;
        Self::from_graph_with(graph, stage, hash, options)
    }

    /// Plan an already built graph.
    pub fn from_graph(graph: ScalarValueGraph, stage: ShaderStage, hash: u64) -> Result<ShaderResourcePlan, ResourcePlanError> {
        Self::from_graph_with(graph, stage, hash, PlanOptions::default())
    }

    /// [`Self::from_graph`] with explicit [`PlanOptions`].
    pub fn from_graph_with(mut graph: ScalarValueGraph, stage: ShaderStage, hash: u64, options: PlanOptions) -> Result<ShaderResourcePlan, ResourcePlanError> {
        let reads = table_reads::plan(&mut graph, stage, hash, options)?;
        let mut memo = HashMap::new();
        let rewrite_read = |graph: &mut ScalarValueGraph, memo: &mut HashMap<ValueId, ValueId>, read: ValueId| {
            // A flattened read keeps its own root; only its handle and
            // offset are rewritten.
            let node = graph.node(read).clone();
            let handle = graph.substitute(node.operands[0], &reads.replacements, memo);
            let offset = graph.substitute(node.operands[1], &reads.replacements, memo);
            if handle == node.operands[0] && offset == node.operands[1] {
                read
            } else {
                graph.memory_read(node.kind, handle, offset, node.memory_index())
            }
        };
        let table_reads: Vec<ResourceTableRead> = reads
            .reads
            .iter()
            .map(|r| ResourceTableRead { value: rewrite_read(&mut graph, &mut memo, r.value), flat_offset: r.flat_offset })
            .collect();
        let dynamic_reads: Vec<ValueId> = reads.dynamic_reads.iter().map(|&r| rewrite_read(&mut graph, &mut memo, r)).collect();
        let mut accesses = vec![None; graph.accesses.len()];
        for index in 0..graph.accesses.len() {
            let Some(access) = graph.accesses[index] else { continue };
            let rewrite = |v: Option<ValueId>, g: &mut ScalarValueGraph, m: &mut HashMap<ValueId, ValueId>| {
                v.map(|v| g.substitute(v, &reads.replacements, m))
            };
            let handle = rewrite(access.handle, &mut graph, &mut memo);
            let sampler_handle = rewrite(access.sampler_handle, &mut graph, &mut memo);
            let read = access.read.map(|r| {
                if reads.replacements.contains_key(&r) {
                    rewrite_read(&mut graph, &mut memo, r)
                } else {
                    graph.substitute(r, &reads.replacements, &mut memo)
                }
            });
            let offset = rewrite(access.offset, &mut graph, &mut memo);
            let active = rewrite(access.active, &mut graph, &mut memo);
            accesses[index] = Some(MemoryAccessBinding { handle, sampler_handle, read, offset, active });
        }

        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let mut plan = ShaderResourcePlan {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            graph,
            stage,
            hash,
            descriptor_sources: Vec::new(),
            materialization_sources: Vec::new(),
            resource_branches: Vec::new(),
            table_reads,
            flattened_slot_by_memory_index: reads.flattened_slot_by_memory_index,
            dynamic_reads,
            clean_flat_slots: Vec::new(),
            indirect_images: Vec::new(),
            suppressed_memory: HashSet::new(),
            buffer_candidate_tables: Vec::new(),
            requires_specialization_memory: false,
            info: ShaderResourceInfo::default(),
            accesses,
            device_address_ranges: Vec::new(),
            written_range_slot_by_handle: HashMap::new(),
            options,
        };

        let tracked = tracker::track(&mut plan)?;
        plan.descriptor_sources = tracked.sources;
        plan.info = tracked.info;
        plan.indirect_images = tracked.indirect_images;
        plan.suppressed_memory = tracked.suppressed_memory;
        plan.buffer_candidate_tables = tracked.buffer_candidate_tables;
        plan.dynamic_reads.retain(|r| !tracked.indirect_reads.contains(r));
        // Scalar reads the planners took over without a flattened slot (the
        // descriptor reads of an indirect image, or code after an indirect
        // branch) are still instructions of the program: when one survives
        // into the module it reads guest memory through the device-address
        // page table, which must then be bound.
        let unslotted = plan.graph.memory.entries().iter().enumerate().any(|(i, m)| {
            m.planning_only
                && matches!(m.kind, MemoryResourceKind::ScalarAddress | MemoryResourceKind::ScalarBuffer)
                && plan.accesses.get(i).is_some_and(|a| a.is_some())
                && !plan.flattened_slot_by_memory_index.contains_key(&i)
        });
        if unslotted {
            plan.info.uses_device_addresses = true;
        }

        let mut materialization = Vec::new();
        materialization.extend(plan.info.buffers.iter().map(|b| b.source));
        for image in &plan.info.images {
            if plan.descriptor_sources[image.source as usize].indirect_image.is_some() {
                plan.requires_specialization_memory = true;
            } else {
                materialization.push(image.source);
            }
        }
        materialization.extend(plan.info.samplers.iter().map(|s| s.source));
        plan.materialization_sources = materialization;

        let mut clean_slots = vec![0u8; plan.table_reads.len()];
        for image in &plan.info.images {
            let Some(indirect) = &plan.descriptor_sources[image.source as usize].indirect_image else { continue };
            if let Some(candidates) = &indirect.direct_candidates {
                for candidate in candidates.iter() {
                    plan.mark_clean_flat_slots(candidate.source, &mut clean_slots);
                }
            } else if indirect.dense {
                plan.mark_clean_flat_slots(indirect.heap_source, &mut clean_slots);
                if indirect.wave_indexed.is_some() {
                    plan.mark_heap_read_slots(indirect.heap_source, &mut clean_slots);
                }
            } else {
                plan.mark_clean_flat_slots(indirect.material_source, &mut clean_slots);
                plan.mark_clean_flat_slots(indirect.heap_source, &mut clean_slots);
            }
        }
        plan.clean_flat_slots = clean_slots;
        plan.resource_branches = branches::build(&mut plan, &reads.replacements, &mut memo);
        plan.device_address_ranges = device_ranges::plan(&mut plan);

        // Each written handle owns three flattened slots after the table
        // reads: base low, base high and size, checked by every store.
        let mut written = HashMap::new();
        for range in &plan.device_address_ranges {
            if range.written {
                let slot = (plan.table_reads.len() + written.len() * WRITTEN_RANGE_DWORD_COUNT) as u32;
                written.insert(range.handle, slot);
            }
        }
        plan.written_range_slot_by_handle = written;
        Ok(plan)
    }

    pub fn user_data_base(&self) -> u32 {
        self.graph.user_data_base
    }

    pub fn user_data_count(&self) -> u32 {
        self.graph.user_data_count
    }

    pub fn memory(&self) -> &MemoryAccessTable {
        &self.graph.memory
    }

    pub fn program(&self) -> &Program {
        &self.graph.program
    }

    pub fn written_range_count(&self) -> usize {
        self.written_range_slot_by_handle.len()
    }

    /// The flattened table the host fills per draw: table reads, then the
    /// written ranges.
    pub fn flattened_table_reserved_count(&self) -> usize {
        self.table_reads.len() + self.written_range_count() * WRITTEN_RANGE_DWORD_COUNT
    }

    /// The value is host-evaluable per draw.
    pub fn validate_runtime_value(&self, value: ValueId) -> bool {
        self.validate_runtime_value_detail(value).0
    }

    /// `(valid, control_dependent)`.
    pub fn validate_runtime_value_detail(&self, value: ValueId) -> (bool, bool) {
        let mut validator = RuntimeValueValidator::new(&self.graph, self.table_reads.len());
        let ok = validator.validate(value);
        (ok, validator.control_dependent)
    }

    /// Every flattened slot an indirect table's descriptor depends on is read
    /// with the clean reader, including the slots those reads depend on.
    fn mark_clean_flat_slots(&self, source: u32, slots: &mut [u8]) {
        let mut pending: Vec<ValueId> = self.descriptor_sources[source as usize].dwords.clone();
        let mut visited = HashSet::new();
        while let Some(value) = pending.pop() {
            if !visited.insert(value) {
                continue;
            }
            let node = self.graph.node(value);
            if node.kind == ValueKind::ResourceTableWord {
                let slot = node.payload as usize;
                if slot < slots.len() {
                    slots[slot] = 1;
                    pending.push(self.table_reads[slot].value);
                }
                continue;
            }
            pending.extend(node.operands.iter().copied());
        }
    }

    /// A wave-indexed table picks its keys from a mask word read off the
    /// same heap.
    fn mark_heap_read_slots(&self, heap: u32, slots: &mut [u8]) {
        let heap = &self.descriptor_sources[heap as usize].dwords;
        for slot in 0..self.table_reads.len().min(slots.len()) {
            let value = self.table_reads[slot].value;
            let node = self.graph.node(value);
            if node.kind != ValueKind::ScalarAddressWord || node.operands.is_empty() {
                continue;
            }
            let handle = self.graph.node(node.operands[0]);
            if handle.kind == ValueKind::AddressHandle
                && handle.operands.len() == heap.len()
                && handle.operands.iter().zip(heap.iter()).all(|(&a, &b)| a == b || self.graph.equivalent(a, b))
            {
                slots[slot] = 1;
            }
        }
    }
}
