// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Interns descriptor sources, builds the dense buffer, image, sampler and
//! sampled-pair tables and patches each memory access with its index.
//! Also recognises images whose descriptor is selected at run time
//! (material/heap tables, dense key tables, finite S_FF1 candidate sets).

use super::candidates::{self, BufferCandidateTablePlan};
use super::graph::inline_constant;
use super::info::*;
use super::memory::*;
use super::plan::{IndirectImageAccess, ShaderResourcePlan};
use super::selector::IndirectSelectorValues;
use super::value::{ScalarOp, ValueId, ValueKind, ValueType};
use super::ResourcePlanError;
use crate::ir::branch;
use crate::ir_types::*;
use crate::opcode::Opcode;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

const SAMPLER_BORDER_CLAMP_MASK: u32 = (1 << 2) | (1 << 5) | (1 << 8);
const SAMPLER_DWORD3_RESERVED_MASK: u32 = 0x3FFF_F000;
const DENSE_INDIRECT_IMAGE_SHIFT: u32 = 5;
const MAX_DENSE_INDIRECT_IMAGE_ENTRIES: u32 = 4096;

pub(super) struct TrackResult {
    pub sources: Vec<DescriptorSource>,
    pub info: ShaderResourceInfo,
    pub indirect_images: Vec<IndirectImageAccess>,
    pub indirect_reads: HashSet<ValueId>,
    /// Memory indices of the suppressed descriptor reads of run-time
    /// selected images.
    pub suppressed_memory: HashSet<usize>,
    pub buffer_candidate_tables: Vec<BufferCandidateTablePlan>,
}

struct IndirectImagePlan {
    handle: ValueId,
    source: u32,
    key: ValueId,
    heap_source: u32,
    key_is_address_offset: bool,
    suppress_memory_reads: bool,
    memory: Vec<usize>,
    reads: Vec<ValueId>,
}

struct Tracker<'p> {
    plan: &'p mut ShaderResourcePlan,
    info: ShaderResourceInfo,
    sources: Vec<DescriptorSource>,
    memory_patches: Vec<(usize, u32, u32, bool)>,
    indirect_images: Vec<IndirectImagePlan>,
    buffer_candidate_tables: Vec<BufferCandidateTablePlan>,
    uses: HashMap<ValueId, Vec<ValueId>>,
    reads_by_memory: HashMap<usize, Vec<ValueId>>,
}

pub(super) fn track(plan: &mut ShaderResourcePlan) -> Result<TrackResult, ResourcePlanError> {
    let mut roots = Vec::new();
    for access in plan.accesses.iter().flatten() {
        roots.extend(access.handle);
        roots.extend(access.sampler_handle);
        roots.extend(access.read);
    }
    roots.extend(plan.table_reads.iter().map(|r| r.value));
    roots.extend(plan.dynamic_reads.iter().copied());
    let uses = plan.graph.collect_uses(&roots);
    let mut reads_by_memory: HashMap<usize, Vec<ValueId>> = HashMap::new();
    for &value in uses.keys().chain(roots.iter()) {
        let node = plan.graph.node(value);
        if matches!(node.kind, ValueKind::ScalarAddressWord | ValueKind::ScalarBufferWord) {
            let list = reads_by_memory.entry(node.memory_index()).or_default();
            if !list.contains(&value) {
                list.push(value);
            }
        }
    }
    Tracker {
        plan,
        info: ShaderResourceInfo::default(),
        sources: Vec::new(),
        memory_patches: Vec::new(),
        indirect_images: Vec::new(),
        buffer_candidate_tables: Vec::new(),
        uses,
        reads_by_memory,
    }
    .run()
}

impl Tracker<'_> {
    fn run(mut self) -> Result<TrackResult, ResourcePlanError> {
        self.plan_indirect_images();
        for index in 0..self.plan.graph.memory.len() {
            self.collect(index)?;
        }
        self.link_image_aliases();
        for &(index, resource, sampler, has_sampler) in &self.memory_patches {
            let memory = self.plan.graph.memory.get_mut(index);
            memory.resource = resource;
            if has_sampler {
                memory.sampler = sampler;
            }
        }
        let mut indirect_reads = HashSet::new();
        let mut suppressed_memory = HashSet::new();
        let mut indirect_accesses = Vec::new();
        for plan in &self.indirect_images {
            if plan.suppress_memory_reads {
                for &index in &plan.memory {
                    self.plan.graph.memory.get_mut(index).planning_only = true;
                    suppressed_memory.insert(index);
                }
                indirect_reads.extend(plan.reads.iter().copied());
            }
            for (index, access) in self.plan.accesses.iter().enumerate() {
                if access.and_then(|a| a.handle) == Some(plan.handle) {
                    indirect_accesses.push(IndirectImageAccess {
                        memory_index: index,
                        key: plan.key,
                        heap_source: plan.heap_source,
                        key_is_address_offset: plan.key_is_address_offset,
                    });
                }
            }
        }
        Ok(TrackResult {
            sources: self.sources,
            info: self.info,
            indirect_images: indirect_accesses,
            indirect_reads,
            suppressed_memory,
            buffer_candidate_tables: self.buffer_candidate_tables,
        })
    }

    fn failure(&self, pc: u32, reason: impl std::fmt::Display) -> ResourcePlanError {
        ResourcePlanError::new(format!(
            "shader resource tracking: hash=0x{:016X} stage={:?} pc=0x{pc:08X} {reason}",
            self.plan.hash, self.plan.stage
        ))
    }

    fn graph(&self) -> &super::graph::ScalarValueGraph {
        &self.plan.graph
    }

    fn node_kind(&self, v: ValueId) -> ValueKind {
        self.plan.graph.kind(v)
    }

    // ---- descriptor sources ----

    /// Copy a handle's dwords into a source. A sampler that no clamp axis
    /// sets to border mode drops its (unused) border colour.
    fn make_source(&mut self, handle: ValueId, width: usize, sampler: bool, sample_adjust: bool, pc: u32) -> Result<DescriptorSource, ResourcePlanError> {
        let operands = self.plan.graph.operands(handle).to_vec();
        if operands.len() != width {
            return Err(self.failure(
                pc,
                format!("{:?} has {} descriptor dwords, expected {width}", self.node_kind(handle), operands.len()),
            ));
        }
        let mut dwords = operands;
        if sample_adjust {
            dwords[3] = self.canonicalize_sample_adjust_dword3(dwords[3]);
        }
        if sampler {
            if let Some(d0) = self.graph().constant_u32(dwords[0]) {
                if self.graph().node(dwords[0]).ty == ValueType::U32 && d0 & SAMPLER_BORDER_CLAMP_MASK == 0 {
                    dwords[3] = self.plan.graph.constant(0);
                }
            }
        }
        Ok(DescriptorSource { dwords, indirect_image: None })
    }

    fn possible_bits(&self, value: ValueId) -> u32 {
        let graph = self.graph();
        let node = graph.node(value);
        if node.kind == ValueKind::Phi {
            return match graph.resolve_invariant_phi(value) {
                None => u32::MAX,
                Some(i) => self.possible_bits(i),
            };
        }
        if node.is_constant() {
            return if node.ty == ValueType::U32 { node.payload as u32 } else { u32::MAX };
        }
        if node.kind != ValueKind::Operation {
            if node.kind == ValueKind::Select {
                return self.possible_bits(node.operands[1]) | self.possible_bits(node.operands[2]);
            }
            return u32::MAX;
        }
        match node.op {
            ScalarOp::And32 => self.possible_bits(node.operands[0]) & self.possible_bits(node.operands[1]),
            ScalarOp::Or32 => self.possible_bits(node.operands[0]) | self.possible_bits(node.operands[1]),
            ScalarOp::ShiftLeft32 if graph.is_constant(node.operands[1]) => {
                self.possible_bits(node.operands[0]) << (graph.constant_u32(node.operands[1]).unwrap() & 31)
            }
            ScalarOp::BitFieldInsert
                if graph.is_constant(node.operands[2]) && graph.is_constant(node.operands[3]) =>
            {
                // (src0 & !field) | ((src1 << offset) & field) for an
                // in-range immediate field.
                let offset = graph.constant_u32(node.operands[2]).unwrap();
                let count = graph.constant_u32(node.operands[3]).unwrap();
                if offset < 32 && count <= 32 - offset {
                    let field = if count == 32 { u32::MAX } else { ((1u32 << count) - 1) << offset };
                    (self.possible_bits(node.operands[0]) & !field)
                        | ((self.possible_bits(node.operands[1]) << offset) & field)
                } else {
                    u32::MAX
                }
            }
            _ => u32::MAX,
        }
    }

    fn canonicalize_sample_adjust_dword3(&mut self, mut value: ValueId) -> ValueId {
        // A dword 3 whose possible bits all lie in the reserved bits (a
        // shifted field with nothing else) is dropped whole.
        if self.possible_bits(value) & !SAMPLER_DWORD3_RESERVED_MASK == 0 {
            return self.plan.graph.constant(0);
        }
        loop {
            let node = self.graph().node(value);
            if !node.is_operation(ScalarOp::Or32) {
                return value;
            }
            let (left, right) = (node.operands[0], node.operands[1]);
            let left_reserved = self.possible_bits(left) & !SAMPLER_DWORD3_RESERVED_MASK == 0;
            let right_reserved = self.possible_bits(right) & !SAMPLER_DWORD3_RESERVED_MASK == 0;
            if left_reserved && right_reserved {
                return self.plan.graph.constant(0);
            }
            if left_reserved {
                value = right;
            } else if right_reserved {
                value = left;
            } else {
                return value;
            }
        }
    }

    /// `Ok(())` when every dword is a host-evaluable U32 value; else the bad
    /// dword and whether the failure is control dependent.
    fn validate_source(&self, source: &DescriptorSource) -> Result<(), (usize, bool)> {
        for (i, &dword) in source.dwords.iter().enumerate() {
            if self.graph().node(dword).ty != ValueType::U32 {
                return Err((i, false));
            }
            let (ok, control_dependent) = self.plan.validate_runtime_value_detail(dword);
            if !ok {
                return Err((i, control_dependent));
            }
        }
        Ok(())
    }

    fn intern_source(&mut self, source: DescriptorSource) -> u32 {
        for (index, current) in self.sources.iter().enumerate() {
            if current.dwords.len() != source.dwords.len() || current.indirect_image != source.indirect_image {
                continue;
            }
            if current.dwords.iter().zip(&source.dwords).all(|(&a, &b)| self.plan.graph.equivalent(a, b)) {
                return index as u32;
            }
        }
        self.sources.push(source);
        (self.sources.len() - 1) as u32
    }

    /// Record a proven bounded SRT candidate table, reusing an equivalent
    /// one. `false` when the SRT root itself is not a valid runtime source.
    fn intern_buffer_candidate_table(&mut self, mut table: BufferCandidateTablePlan, pc: u32, memory_index: usize) -> Result<bool, ResourcePlanError> {
        table.source_srt_resource = self.intern_runtime_source(table.srt_handle, pc)?;
        if table.source_srt_resource == NO_INDEX {
            return Ok(false);
        }
        for existing in 0..self.buffer_candidate_tables.len() {
            let current = &self.buffer_candidate_tables[existing];
            if !self.plan.graph.equivalent(current.srt_handle, table.srt_handle)
                || !self.plan.graph.equivalent(current.offset_expression, table.offset_expression)
            {
                continue;
            }
            self.buffer_candidate_tables[existing].memory_indices.push(memory_index);
            return Ok(true);
        }
        self.buffer_candidate_tables.push(table);
        Ok(true)
    }

    fn intern_runtime_source(&mut self, handle: ValueId, pc: u32) -> Result<u32, ResourcePlanError> {
        let source = self.make_source(handle, 4, false, false, pc)?;
        Ok(if self.validate_source(&source).is_ok() { self.intern_source(source) } else { NO_INDEX })
    }

    fn is_host_buffer_handle(&self, handle: Option<ValueId>) -> bool {
        let Some(handle) = handle else { return false };
        let node = self.graph().node(handle);
        node.kind == ValueKind::BufferHandle
            && node.operands.len() == 4
            && node
                .operands
                .iter()
                .all(|&d| self.graph().node(d).ty == ValueType::U32 && self.plan.validate_runtime_value(d))
    }

    fn is_device_loaded_buffer_handle(&self, handle: Option<ValueId>) -> bool {
        let Some(handle) = handle else { return false };
        let node = self.graph().node(handle);
        node.kind == ValueKind::BufferHandle
            && node.operands.len() == 4
            && node.operands.iter().all(|&d| {
                self.graph().node(d).ty == ValueType::U32
                    && (self.plan.validate_runtime_value(d) || self.depends_on_scalar_buffer_word(d))
            })
            && node.operands.iter().any(|&d| self.depends_on_scalar_buffer_word(d))
    }

    fn depends_on_scalar_buffer_word(&self, value: ValueId) -> bool {
        let mut pending = vec![value];
        let mut visited = HashSet::new();
        while let Some(current) = pending.pop() {
            if !visited.insert(current) {
                continue;
            }
            let node = self.graph().node(current);
            if node.kind == ValueKind::ScalarBufferWord {
                return true;
            }
            pending.extend(node.operands.iter().copied());
        }
        false
    }

    fn get_handle_source(
        &mut self,
        handle: Option<ValueId>,
        expected: ValueKind,
        width: usize,
        pc: u32,
        sampler: bool,
        sample_adjust: bool,
        memory_opcode: Option<Opcode>,
        memory_access: MemoryAccess,
    ) -> Result<u32, ResourcePlanError> {
        let handle = match handle {
            Some(h) if self.node_kind(h) == expected => h,
            _ => return Err(self.failure(pc, format!("memory operation requires {expected:?}"))),
        };
        let mut source = self.make_source(handle, width, sampler, sample_adjust, pc)?;
        let non_contiguous_image = expected == ValueKind::ImageHandle && !self.is_contiguous_scalar_buffer_record(&source);
        // Image descriptors loaded straight from a scalar buffer at a
        // dynamically uniform offset are evaluated like buffer descriptors;
        // the validator is the single source of truth.
        let check = if non_contiguous_image { Err((0usize, false)) } else { self.validate_source(&source) };
        if let Err((bad, control_dependent)) = check {
            // A bindless image/sampler descriptor resolving through a
            // control-dependent phi (a hash-table material lookup) has no
            // single compile-time source; degrade that access to a null
            // descriptor. Buffers and other failures still fail the plan.
            let fallback = matches!(expected, ValueKind::ImageHandle | ValueKind::SamplerHandle)
                && (control_dependent
                    || self.graph().has_undefined_origin(source.dwords[bad], "BufferLoadFormat")
                    || (non_contiguous_image && source.dwords.iter().any(|&d| self.graph().has_undefined_origin(d, "SAndB32"))));
            if fallback {
                let zero = self.plan.graph.constant(0);
                source = DescriptorSource { dwords: vec![zero; source.dwords.len()], indirect_image: None };
            } else {
                let opcode = memory_opcode.map_or("memory".to_string(), |o| o.name().into_owned());
                return Err(self.failure(
                    pc,
                    format!(
                        "{opcode} ({memory_access:?}) {expected:?} dword {bad} is not a valid runtime value{} (value: {})",
                        self.graph().describe_undefined_leaves(source.dwords[bad]),
                        self.graph().describe(source.dwords[bad], 4)
                    ),
                ));
            }
        }
        Ok(self.intern_source(source))
    }

    /// An image descriptor read from a scalar buffer must be one contiguous
    /// record.
    fn is_contiguous_scalar_buffer_record(&self, source: &DescriptorSource) -> bool {
        let graph = self.graph();
        if !source.dwords.iter().any(|&d| graph.kind(d) == ValueKind::ScalarBufferWord) {
            return true;
        }
        let first = self.scalar_read_memory(source.dwords[0]);
        for (dword, &read) in source.dwords.iter().enumerate() {
            let memory = self.scalar_read_memory(read);
            match (first, memory) {
                (Some((_, f)), Some((_, m))) => {
                    if m.offset != f.offset.wrapping_add(dword as u32 * 4)
                        || !graph.equivalent(graph.operand(read, 0), graph.operand(source.dwords[0], 0))
                        || !graph.equivalent(graph.operand(read, 1), graph.operand(source.dwords[0], 1))
                    {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        true
    }

    /// A runtime V#: its dwords cannot be resolved at plan time but every
    /// non-constant leaf is a read from guest memory (a descriptor read from
    /// a device address or from the SRT at a dynamic offset). A handle that
    /// is fully resolvable keeps the native binding.
    fn is_runtime_descriptor_handle(&self, handle: Option<ValueId>, allow_computed_words: bool) -> bool {
        let Some(handle) = handle else { return false };
        let node = self.graph().node(handle);
        if node.kind != ValueKind::BufferHandle || node.operands.len() != 4 {
            return false;
        }
        if !node.operands.iter().all(|&d| self.is_runtime_descriptor_dword(d, &mut HashSet::new())) {
            // A fetch shader can combine one scalar address load with words
            // produced in the vector path: the shader still owns the whole
            // descriptor in its registers, so lower it through device
            // addresses.
            return allow_computed_words && node.operands.iter().any(|&d| self.has_read_kind(d, ValueKind::ScalarAddressWord, &mut HashSet::new()));
        }
        node.operands.iter().any(|&d| self.has_runtime_read(d, &mut HashSet::new()))
    }

    /// A word read from guest memory (consumed as a device-address word; it
    /// may have no host record), or one derivable from such words. (The
    /// derivability check runs with the word not yet visited: marking it
    /// first would reject every non-phi word, e.g. a constant stride dword.)
    fn is_runtime_descriptor_dword(&self, value: ValueId, visiting: &mut HashSet<ValueId>) -> bool {
        if matches!(self.node_kind(value), ValueKind::ScalarAddressWord | ValueKind::ScalarBufferWord) {
            return true;
        }
        self.is_runtime_derivable(value, visiting)
    }

    /// Known at plan time (constant, user data, shader base, a flattened
    /// table word) or ultimately a guest-memory read; a phi, select or
    /// operation qualifies when every operand does.
    fn is_runtime_derivable(&self, value: ValueId, visiting: &mut HashSet<ValueId>) -> bool {
        if !visiting.insert(value) {
            return self.node_kind(value) == ValueKind::Phi;
        }
        let node = self.graph().node(value);
        let result = match node.kind {
            ValueKind::Constant | ValueKind::ResourceTableWord | ValueKind::UserData | ValueKind::ShaderBase | ValueKind::MemoryAperture => {
                true
            }
            ValueKind::ScalarAddressWord | ValueKind::ScalarBufferWord => node.memory_index() < self.graph().memory.len(),
            ValueKind::Phi | ValueKind::Select | ValueKind::Operation | ValueKind::FirstLane => {
                node.operands.iter().all(|&o| self.is_runtime_derivable(o, visiting))
            }
            _ => false,
        };
        visiting.remove(&value);
        result
    }

    fn has_read_kind(&self, value: ValueId, kind: ValueKind, visiting: &mut HashSet<ValueId>) -> bool {
        if !visiting.insert(value) {
            return false;
        }
        let node = self.graph().node(value);
        let result = node.kind == kind || node.operands.iter().any(|&o| self.has_read_kind(o, kind, visiting));
        visiting.remove(&value);
        result
    }

    fn has_runtime_read(&self, value: ValueId, visiting: &mut HashSet<ValueId>) -> bool {
        if !visiting.insert(value) {
            return false;
        }
        let node = self.graph().node(value);
        let result = matches!(node.kind, ValueKind::ScalarAddressWord | ValueKind::ScalarBufferWord)
            || node.operands.iter().any(|&o| self.has_runtime_read(o, visiting));
        visiting.remove(&value);
        result
    }

    // ---- dense tables ----

    fn byte_extent(memory: &MemoryAccessInfo) -> u32 {
        let bytes = memory.data_bits.div_ceil(8).max(1) as u64;
        let count = memory.data_dwords.max(1) as u64;
        let end = memory.offset as u64 + bytes * count;
        end.min(u32::MAX as u64) as u32
    }

    fn merge_buffer(resource: &mut BufferResource, memory: &MemoryAccessInfo, pc: u32) {
        let atomic = memory.access == MemoryAccess::Atomic;
        let write = memory.access == MemoryAccess::Write || atomic;
        resource.first_use_pc = resource.first_use_pc.min(pc);
        resource.max_byte_extent = resource.max_byte_extent.max(Self::byte_extent(memory));
        resource.read |= !write || atomic;
        resource.written |= write;
        resource.atomic |= atomic;
        resource.formatted |= memory.formatted;
        resource.scalar |= memory.kind == MemoryResourceKind::ScalarBuffer;
        resource.constant_offsets &= memory.kind == MemoryResourceKind::ScalarBuffer && !memory.dynamic_offset && (memory.offset as i32) >= 0;
    }

    fn add_buffer(&mut self, source: u32, memory: &MemoryAccessInfo, pc: u32) -> u32 {
        if let Some(index) = self.info.buffers.iter().position(|b| b.source == source) {
            Self::merge_buffer(&mut self.info.buffers[index], memory, pc);
            return index as u32;
        }
        if self.info.buffers.len() >= ShaderResourceInfo::MAX_BUFFERS {
            return NO_INDEX;
        }
        let mut resource = BufferResource::new(source, pc);
        resource.constant_offsets = true;
        Self::merge_buffer(&mut resource, memory, pc);
        self.info.buffers.push(resource);
        (self.info.buffers.len() - 1) as u32
    }

    fn merge_image(image: &mut ImageResource, memory: &MemoryAccessInfo, pc: u32) {
        let atomic = memory.access == MemoryAccess::Atomic;
        let write = memory.access == MemoryAccess::Write || atomic;
        image.first_use_pc = image.first_use_pc.min(pc);
        image.read |= !write || atomic;
        image.written |= write;
        image.atomic |= atomic;
        image.fetched |= !write && !memory.needs_sampler && memory.opcode != Opcode::ImageGetResinfo;
        image.packed |= memory.image_sample_flags & image_sample_flags::PACKED != 0;
    }

    fn add_image(&mut self, source: u32, memory: &MemoryAccessInfo, pc: u32) -> u32 {
        let class = memory.image_class;
        let dynamic_mip = (class == ImageResourceClass::Storage && memory.image_has_mip)
            || memory.image_sample_flags & image_sample_flags::GATHER_LOD != 0;
        let mip = if dynamic_mip { ImageMipMode::Dynamic } else { ImageMipMode::None };
        let depth = memory.image_sample_flags & image_sample_flags::COMPARE != 0;
        let atomic64 = memory.access == MemoryAccess::Atomic && memory.data_bits == 64;
        if let Some(index) = self.info.images.iter().position(|i| {
            i.source == source
                && i.resource_class == class
                && i.dimension == memory.image_dimension
                && i.mip_mode == mip
                && i.depth_compare == depth
                && i.r128 == memory.image_r128
                && i.atomic64 == atomic64
        }) {
            Self::merge_image(&mut self.info.images[index], memory, pc);
            return index as u32;
        }
        if self.info.images.len() >= ShaderResourceInfo::MAX_IMAGES {
            return NO_INDEX;
        }
        let mut image = ImageResource::new(source, pc);
        image.resource_class = class;
        image.dimension = memory.image_dimension;
        image.mip_mode = mip;
        image.depth_compare = depth;
        image.r128 = memory.image_r128;
        image.atomic64 = atomic64;
        Self::merge_image(&mut image, memory, pc);
        self.info.images.push(image);
        (self.info.images.len() - 1) as u32
    }

    fn add_sampler(&mut self, source: u32, pc: u32) -> u32 {
        if let Some(index) = self.info.samplers.iter().position(|s| s.source == source) {
            let s = &mut self.info.samplers[index];
            s.first_use_pc = s.first_use_pc.min(pc);
            return index as u32;
        }
        if self.info.samplers.len() >= ShaderResourceInfo::MAX_SAMPLERS {
            return NO_INDEX;
        }
        self.info.samplers.push(SamplerResource { source, first_use_pc: pc, ..Default::default() });
        (self.info.samplers.len() - 1) as u32
    }

    fn add_sampled_pair(&mut self, image: u32, sampler: u32, pc: u32) -> Result<(), ResourcePlanError> {
        if let Some(pair) = self.info.sampled_pairs.iter_mut().find(|p| p.image == image && p.sampler == sampler) {
            pair.first_use_pc = pair.first_use_pc.min(pc);
            return Ok(());
        }
        self.info.sampled_pairs.push(SampledImagePair { image, sampler, first_use_pc: pc, point_sampler: None });
        Ok(())
    }

    fn add_memory_patch(&mut self, index: usize, resource: u32, sampler: u32, has_sampler: bool, pc: u32) -> Result<(), ResourcePlanError> {
        if let Some(position) = self.memory_patches.iter().position(|p| p.0 == index) {
            let existing = self.memory_patches[position];
            if existing.1 != resource || (has_sampler && existing.3 && existing.2 != sampler) {
                return Err(self.failure(pc, "memory metadata is reused with incompatible resources"));
            }
            if has_sampler {
                self.memory_patches[position] = (index, resource, sampler, true);
            }
            return Ok(());
        }
        self.memory_patches.push((index, resource, sampler, has_sampler));
        Ok(())
    }

    // ---- collection ----

    fn collect(&mut self, index: usize) -> Result<(), ResourcePlanError> {
        let memory = self.plan.graph.memory.get(index).clone();
        let access = self.plan.accesses[index];
        let is_buffer = matches!(memory.kind, MemoryResourceKind::Buffer | MemoryResourceKind::ScalarBuffer);
        let is_address = memory.is_address_kind();
        let is_image = memory.kind == MemoryResourceKind::Image;
        if memory.opcode.is_ray_intersect() {
            // BVH nodes are read through the device-address page table.
            self.info.uses_device_addresses = true;
            return Ok(());
        }
        if !is_buffer && !is_address && !is_image {
            return Ok(());
        }
        let Some(access) = access else {
            if self.has_indirect_pc_transfer_before(memory.pc) {
                self.plan.graph.memory.get_mut(index).planning_only = true;
                return Ok(());
            }
            return Err(self.failure(memory.pc, "memory operation has no resource handle"));
        };
        if memory.planning_only || self.is_indirect_planning_memory(index) {
            return Ok(());
        }
        let pc = memory.pc;
        let handle_node = access.handle.map(|h| self.graph().node(h).clone());

        if is_buffer && self.plan.options.device_buffer_read(&memory) {
            // Guest memory is resident for the device: the program reads
            // the V# from its registers and the data through its address.
            self.plan.graph.memory.get_mut(index).device_descriptor = true;
            self.info.uses_device_addresses = true;
            return Ok(());
        }
        if is_buffer {
            // A front GS/HS half can carry an unreachable tail after its
            // S_SETPC_B64 with undefined SGPRs; do not fail on it.
            if let Some(h) = &handle_node {
                if h.kind == ValueKind::BufferHandle
                    && h.operands.iter().any(|&v| self.graph().is_undefined(v))
                    && self.has_indirect_pc_transfer_before(pc)
                {
                    self.plan.graph.memory.get_mut(index).planning_only = true;
                    return Ok(());
                }
            }
            // A descriptor assembled in vector lanes: its record count or
            // stride is not host-evaluable, but the device-address lowering
            // reads the four words from the shader's registers.
            if matches!(memory.access, MemoryAccess::Read | MemoryAccess::Write) {
                if let Some(h) = handle_node.as_ref().filter(|h| h.kind == ValueKind::BufferHandle) {
                    let _ = h;
                    let source = self.make_source(access.handle.unwrap(), 4, false, false, pc)?;
                    let vector_derived = source.dwords.iter().any(|&d| {
                        let g = self.graph();
                        g.has_undefined_origin(d, "VAdd3U32") || g.has_undefined_origin(d, "VCvtU32F32") || g.has_undefined_origin(d, "VRcpIflagF32")
                    });
                    if vector_derived {
                        self.plan.graph.memory.get_mut(index).buffer_descriptor = Some(GuestBufferDescriptor::runtime());
                        self.info.uses_device_addresses = true;
                        return Ok(());
                    }
                }
            }
            // A V# read from a runtime scalar-memory address cannot be bound
            // ahead of time; record it as a runtime descriptor. A vector
            // access through a V# the draw can evaluate keeps its binding.
            let allow_computed = memory.formatted && memory.opcode.name().starts_with("BufferLoadFormat");
            if self.is_runtime_descriptor_handle(access.handle, allow_computed) {
                let host_valid = if memory.kind == MemoryResourceKind::Buffer {
                    let source = self.make_source(access.handle.unwrap(), 4, false, false, pc)?;
                    self.validate_source(&source).is_ok()
                } else {
                    false
                };
                if memory.kind != MemoryResourceKind::Buffer || !host_valid {
                    if memory.kind == MemoryResourceKind::Buffer {
                        if let Some(table) = candidates::try_plan(self.plan, access.handle.unwrap(), index) {
                            if self.intern_buffer_candidate_table(table, pc, index)? {
                                self.plan.graph.memory.get_mut(index).buffer_descriptor = Some(GuestBufferDescriptor::runtime());
                                return Ok(());
                            }
                        }
                        // Without a bounded candidate table, a V# loaded
                        // from scalar-buffer data is read through its
                        // registers on the device.
                        if self.is_device_loaded_buffer_handle(access.handle) {
                            self.plan.graph.memory.get_mut(index).device_descriptor = true;
                            self.info.uses_device_addresses = true;
                            return Ok(());
                        }
                    }
                    self.plan.graph.memory.get_mut(index).buffer_descriptor = Some(GuestBufferDescriptor::runtime());
                    self.info.uses_device_addresses = true;
                    return Ok(());
                }
            }
            // Scalar loads can address buffers with no host binding; vector
            // descriptors loaded from scalar-buffer data stay device-side.
            if (memory.kind == MemoryResourceKind::ScalarBuffer && !self.is_host_buffer_handle(access.handle))
                || (memory.kind == MemoryResourceKind::Buffer && self.is_device_loaded_buffer_handle(access.handle))
            {
                self.plan.graph.memory.get_mut(index).device_descriptor = true;
                self.info.uses_device_addresses = true;
                return Ok(());
            }
            let source = self.get_handle_source(
                access.handle,
                ValueKind::BufferHandle,
                4,
                pc,
                false,
                false,
                Some(memory.opcode),
                memory.access,
            )?;
            let resource = self.add_buffer(source, &memory, pc);
            if resource == NO_INDEX {
                // Past the binding budget the access reads its V# from its
                // SGPRs, through the device-address table; that path has no
                // typed or formatted store lowering.
                if memory.opcode.name().starts_with("TBuffer") || (memory.formatted && memory.access != MemoryAccess::Read) {
                    return Err(self.failure(pc, "buffer resource limit exceeded"));
                }
                self.plan.graph.memory.get_mut(index).device_descriptor = true;
                self.info.uses_device_addresses = true;
                return Ok(());
            }
            self.plan.graph.memory.get_mut(index).buffer_descriptor = Some(GuestBufferDescriptor::static_resource(resource));
            return self.add_memory_patch(index, resource, 0, false, pc);
        }

        if is_address {
            if memory.kind == MemoryResourceKind::Scratch {
                // Invocation-private storage: nothing to materialise.
                return Ok(());
            }
            match &handle_node {
                Some(h) if h.kind == ValueKind::AddressHandle => {
                    if h.operands.len() != 2 {
                        return Err(self.failure(pc, "an address handle must have two address dwords"));
                    }
                }
                _ => return Err(self.failure(pc, "address operation requires an address handle")),
            }
            self.info.uses_device_addresses = true;
            return Ok(());
        }

        // A fused continuation can consume an image descriptor carried in
        // the incoming SGPR state; leave such an access unplanned.
        if let Some(h) = &handle_node {
            if h.kind == ValueKind::ImageHandle
                && h.operands.iter().any(|&v| self.graph().is_undefined(v))
                && self.has_indirect_pc_transfer_before(pc)
            {
                self.plan.graph.memory.get_mut(index).planning_only = true;
                return Ok(());
            }
        }
        if memory.image_class == ImageResourceClass::None {
            return Err(self.failure(pc, "image operation has invalid resource kind"));
        }
        let image_source = match self.indirect_images.iter().find(|p| Some(p.handle) == access.handle) {
            Some(p) => p.source,
            None => self.get_handle_source(access.handle, ValueKind::ImageHandle, 8, pc, false, false, None, MemoryAccess::Read)?,
        };
        let image = self.add_image(image_source, &memory, pc);
        if image == NO_INDEX {
            return Err(self.failure(pc, "image resource limit exceeded"));
        }
        let mut sampler = 0;
        if memory.needs_sampler {
            if access.sampler_handle.is_none() {
                return Err(self.failure(pc, "sampled image operation has no sampler handle"));
            }
            let adjust = memory.image_sample_flags & image_sample_flags::ADJUST != 0;
            let sampler_source =
                self.get_handle_source(access.sampler_handle, ValueKind::SamplerHandle, 4, pc, true, adjust, None, MemoryAccess::Read)?;
            sampler = self.add_sampler(sampler_source, pc);
            if sampler == NO_INDEX {
                return Err(self.failure(pc, "sampler resource limit exceeded"));
            }
            self.add_sampled_pair(image, sampler, pc)?;
        }
        self.add_memory_patch(index, image, sampler, memory.needs_sampler, pc)
    }

    fn link_image_aliases(&mut self) {
        for b in 0..self.info.buffers.len() {
            let buffer_source = &self.sources[self.info.buffers[b].source as usize];
            if buffer_source.dwords.len() != 4 {
                continue;
            }
            for i in 0..self.info.images.len() {
                let image_source = &self.sources[self.info.images[i].source as usize];
                if image_source.dwords.len() != 8 || image_source.indirect_image.is_some() {
                    continue;
                }
                if (0..4).all(|d| self.plan.graph.equivalent(buffer_source.dwords[d], image_source.dwords[d])) {
                    self.info.buffers[b].image_alias = i as u32;
                    break;
                }
            }
        }
    }

    // ---- indirect images ----

    fn is_indirect_planning_memory(&self, index: usize) -> bool {
        self.indirect_images.iter().any(|p| p.suppress_memory_reads && p.memory.contains(&index))
    }

    fn has_indirect_pc_transfer_before(&self, pc: u32) -> bool {
        let mut previous_end = 0u32;
        for instruction in &self.graph().program.instructions {
            if instruction.pc >= pc {
                break;
            }
            if instruction.opcode == Opcode::SSetpcB64 {
                return true;
            }
            // Fused shaders replace the S_SETPC marker with a NOP and splice
            // the continuation at its real address: the pc discontinuity is
            // the remaining boundary.
            if previous_end != 0 && instruction.pc > previous_end {
                return true;
            }
            previous_end = instruction.next_pc();
        }
        false
    }

    fn plan_indirect_images(&mut self) {
        for index in 0..self.plan.graph.memory.len() {
            if self.plan.graph.memory.get(index).kind != MemoryResourceKind::Image {
                continue;
            }
            let Some(handle) = self.plan.accesses[index].and_then(|a| a.handle) else { continue };
            if self.indirect_images.iter().any(|p| p.handle == handle) {
                continue;
            }
            let pc = self.plan.graph.memory.get(index).pc;
            let plan = match self.try_make_indirect_image(handle, pc) {
                Some(p) => Some(p),
                None => match self.try_make_dense_indirect_image(handle, pc) {
                    Some(p) => Some(p),
                    None => match self.try_make_direct_image(handle) {
                        Some(p) => Some(p),
                        None => self.try_make_table_image(handle, pc),
                    },
                },
            };
            if let Some(p) = plan {
                self.indirect_images.push(p);
            }
        }
    }

    /// The memory record of a 32-bit scalar-buffer read.
    fn scalar_read_memory(&self, read: ValueId) -> Option<(usize, &MemoryAccessInfo)> {
        let node = self.graph().node(read);
        let index = node.memory_index();
        if node.kind != ValueKind::ScalarBufferWord || index >= self.graph().memory.len() {
            return None;
        }
        let memory = self.graph().memory.get(index);
        (matches!(memory.kind, MemoryResourceKind::ScalarBuffer | MemoryResourceKind::Buffer)
            && memory.data_bits == 32
            && memory.data_dwords == 1)
            .then_some((index, memory))
    }

    fn memory_index_belongs_to(&self, index: usize, owner: ValueId) -> bool {
        self.reads_by_memory.get(&index).is_none_or(|readers| readers.iter().all(|&r| r == owner))
    }

    fn uses_only(&self, value: ValueId, users: &[ValueId]) -> bool {
        self.uses.get(&value).is_some_and(|uses| !uses.is_empty() && uses.iter().all(|u| users.contains(u)))
    }

    fn make_runtime_source(&mut self, handle: ValueId, kind: ValueKind, width: usize, pc: u32) -> Option<(u32, DescriptorSource)> {
        if self.node_kind(handle) != kind {
            return None;
        }
        let source = self.make_source(handle, width, false, false, pc).ok()?;
        self.validate_source(&source).ok()?;
        let index = self.intern_source(source.clone());
        Some((index, source))
    }

    fn match_material_offset(&self, mut value: ValueId) -> Option<(ValueId, u32, u32)> {
        let graph = self.graph();
        let mut offset = 0;
        let node = graph.node(value);
        if node.is_operation(ScalarOp::IAdd32) {
            if let Some(c) = graph.constant_u32(node.operands[0]) {
                offset = c;
                value = node.operands[1];
            } else if let Some(c) = graph.constant_u32(node.operands[1]) {
                offset = c;
                value = node.operands[0];
            } else {
                return None;
            }
        }
        let node = graph.node(value);
        if !node.is_operation(ScalarOp::IMul32) {
            return None;
        }
        let (stride, selector) = if let Some(c) = graph.constant_u32(node.operands[0]) {
            (c, node.operands[1])
        } else if let Some(c) = graph.constant_u32(node.operands[1]) {
            (c, node.operands[0])
        } else {
            return None;
        };
        (stride != 0 && graph.kind(selector) == ValueKind::FirstLane).then_some((selector, stride, offset))
    }

    /// A material-table key selects a heap record whose eight dwords are the
    /// image descriptor. The key read, the heap reads and their shift must
    /// feed nothing else.
    fn try_make_indirect_image(&mut self, handle: ValueId, pc: u32) -> Option<IndirectImagePlan> {
        let node = self.graph().node(handle).clone();
        if node.kind != ValueKind::ImageHandle || node.operands.len() != 8 {
            return None;
        }
        let heap_reads: Vec<ValueId> = node.operands.to_vec();
        let mut heap_handle = None;
        let mut heap_offset = None;
        let mut memory_indices = vec![0usize; 8];
        for (dword, &read) in heap_reads.iter().enumerate() {
            let (memory_index, memory) = self.scalar_read_memory(read)?;
            if memory.offset != dword as u32 * 4 || !self.memory_index_belongs_to(memory_index, read) {
                return None;
            }
            let current = self.graph().operand(read, 0);
            if heap_handle.is_some_and(|h| h != current) {
                return None;
            }
            heap_handle = Some(current);
            let offset = self.graph().operand(read, 1);
            match heap_offset {
                None => heap_offset = Some(offset),
                Some(first) if !self.graph().equivalent(first, offset) => return None,
                _ => {}
            }
            memory_indices[dword] = memory_index;
        }
        let heap_offset = heap_offset?;
        let graph = self.graph();
        let shift = graph.node(heap_offset);
        if !shift.is_operation(ScalarOp::ShiftLeft32) || graph.constant_u32(shift.operands[1]) != Some(5) {
            return None;
        }
        let material_read = shift.operands[0];
        let (material_memory_index, material_memory) = self.scalar_read_memory(material_read)?;
        let material_immediate = material_memory.offset;
        if !self.memory_index_belongs_to(material_memory_index, material_read) {
            return None;
        }
        let material_handle = graph.operand(material_read, 0);
        let (selector, stride, selector_offset) = self.match_material_offset(graph.operand(material_read, 1))?;
        if !self.uses_only(material_read, &[heap_offset]) || !self.uses_only(heap_offset, &heap_reads) {
            return None;
        }
        if !heap_reads.iter().all(|&r| self.uses_only(r, &[handle])) {
            return None;
        }
        let (material_index, material_source) = self.make_runtime_source(material_handle, ValueKind::BufferHandle, 4, pc)?;
        let (heap_index, heap_source) = self.make_runtime_source(heap_handle?, ValueKind::BufferHandle, 4, pc)?;
        let mut dwords = material_source.dwords[..4].to_vec();
        dwords.extend_from_slice(&heap_source.dwords[..4]);
        let mut selector_info = IndirectImageSelector::new(material_index, heap_index, stride, selector_offset);
        selector_info.selector_values = IndirectSelectorValues::create(self.plan, selector).map(Arc::new);
        selector_info.material_immediate = material_immediate;
        let source = self.intern_source(DescriptorSource { dwords, indirect_image: Some(selector_info) });
        Some(IndirectImagePlan {
            handle,
            source,
            key: material_read,
            heap_source: heap_index,
            key_is_address_offset: false,
            suppress_memory_reads: true,
            memory: memory_indices,
            reads: heap_reads,
        })
    }

    fn try_make_dense_indirect_image(&mut self, handle: ValueId, pc: u32) -> Option<IndirectImagePlan> {
        let node = self.graph().node(handle).clone();
        if node.kind != ValueKind::ImageHandle || node.operands.len() != 8 {
            return None;
        }
        let mut reads = Vec::with_capacity(8);
        let mut memory_indices = Vec::with_capacity(8);
        let mut heap_handle: Option<ValueId> = None;
        let mut based: Option<ValueId> = None;
        let mut table_immediate = 0u32;
        let mut can_suppress = true;
        for (dword, &read) in node.operands.iter().enumerate() {
            let graph = self.graph();
            let read_node = graph.node(read);
            let memory_index = read_node.memory_index();
            if read_node.kind != ValueKind::ScalarAddressWord
                || memory_index >= graph.memory.len()
                || !self.memory_index_belongs_to(memory_index, read)
            {
                return None;
            }
            let memory = graph.memory.get(memory_index);
            if memory.kind != MemoryResourceKind::ScalarAddress || memory.data_bits != 32 || memory.data_dwords != 1 {
                return None;
            }
            let offset = read_node.operands[1];
            let mut extra = 0u32;
            match based {
                None => based = Some(offset),
                Some(b) if !graph.equivalent(offset, b) => {
                    let o = graph.node(offset);
                    if !o.is_operation(ScalarOp::IAdd32) || o.operands.len() != 2 {
                        return None;
                    }
                    let inner = if let Some(c) = graph.constant_u32(o.operands[1]) {
                        extra = c;
                        o.operands[0]
                    } else if let Some(c) = graph.constant_u32(o.operands[0]) {
                        extra = c;
                        o.operands[1]
                    } else {
                        return None;
                    };
                    if !graph.equivalent(inner, b) {
                        return None;
                    }
                }
                _ => {}
            }
            let component_offset = dword as u32 * 4;
            if (extra as u64) + (memory.offset as u64) < component_offset as u64 {
                return None;
            }
            let immediate = extra.wrapping_add(memory.offset).wrapping_sub(component_offset);
            if dword == 0 {
                table_immediate = immediate;
            } else if immediate != table_immediate {
                return None;
            }
            let current = read_node.operands[0];
            if graph.kind(current) != ValueKind::AddressHandle || heap_handle.is_some_and(|h| !graph.equivalent(current, h)) {
                return None;
            }
            heap_handle = Some(current);
            reads.push(read);
            memory_indices.push(memory_index);
            let memory = memory.clone();
            can_suppress &= self.has_only_image_consumers(&memory, handle);
        }
        let (based, heap_handle) = (based?, heap_handle?);
        let graph = self.graph();
        let mut table_offset = table_immediate;
        let mut scaled = based;
        let s = graph.node(scaled);
        if s.is_operation(ScalarOp::IAdd32) && s.operands.len() == 2 {
            if let Some(c) = graph.constant_u32(s.operands[1]) {
                table_offset = table_offset.wrapping_add(c);
                scaled = s.operands[0];
            } else if let Some(c) = graph.constant_u32(s.operands[0]) {
                table_offset = table_offset.wrapping_add(c);
                scaled = s.operands[1];
            } else {
                return None;
            }
        }
        let s = graph.node(scaled);
        if !s.is_operation(ScalarOp::ShiftLeft32) || s.operands.len() != 2 || graph.constant_u32(s.operands[1]) != Some(DENSE_INDIRECT_IMAGE_SHIFT) {
            return None;
        }
        let key = s.operands[0];
        let bound = self.bound_by_sampler_loads(self.dense_key_bound(key), heap_handle, table_offset);
        let wave_indexed = self.try_create_wave_indexed_image_selector(key, &reads);
        if bound == 0 && wave_indexed.is_none() {
            return None;
        }
        if !reads.iter().all(|&r| self.uses_only(r, &[handle])) {
            return None;
        }
        let (heap_index, heap_source) = self.make_runtime_source(heap_handle, ValueKind::AddressHandle, 2, pc)?;
        let mut dwords = vec![key; 8];
        dwords[0] = heap_source.dwords[0];
        dwords[1] = heap_source.dwords[1];
        let mut selector = IndirectImageSelector::new(0, heap_index, 0, 0);
        selector.dense = true;
        selector.table_offset = table_offset;
        selector.dynamic_offset_base = table_offset.wrapping_sub(table_immediate);
        selector.key_bound = bound;
        selector.wave_indexed = wave_indexed;
        let source = self.intern_source(DescriptorSource { dwords, indirect_image: Some(selector) });
        Some(IndirectImagePlan {
            handle,
            source,
            key: reads[0],
            heap_source: heap_index,
            key_is_address_offset: true,
            suppress_memory_reads: can_suppress,
            memory: memory_indices,
            reads,
        })
    }

    fn bound_by_sampler_loads(&self, mut bound: u32, heap_handle: ValueId, table_offset: u32) -> u32 {
        let graph = self.graph();
        for access in self.plan.accesses.iter().flatten() {
            let Some(sampler) = access.sampler_handle else { continue };
            for &operand in graph.operands(sampler) {
                let mut word = graph.resolve_invariant_phi(operand).unwrap_or(operand);
                let n = graph.node(word);
                if n.kind == ValueKind::ResourceTableWord && (n.payload as usize) < self.plan.table_reads.len() {
                    word = self.plan.table_reads[n.payload as usize].value;
                }
                let n = graph.node(word);
                if n.kind != ValueKind::ScalarAddressWord
                    || n.memory_index() >= graph.memory.len()
                    || !graph.is_constant(n.operands[1])
                    || !graph.equivalent(n.operands[0], heap_handle)
                {
                    continue;
                }
                let offset = graph.constant_u32(n.operands[1]).unwrap() as u64 + graph.memory.get(n.memory_index()).offset as u64;
                if offset > table_offset as u64 {
                    bound = bound.min(((offset - table_offset as u64) >> DENSE_INDIRECT_IMAGE_SHIFT) as u32);
                }
            }
        }
        bound
    }

    fn dense_key_bound(&self, key: ValueId) -> u32 {
        let graph = self.graph();
        if graph.node(key).is_operation(ScalarOp::FindLowestBit32) && graph.has_non_zero_bit_scan_input(key) {
            return 32;
        }
        let Some(uses) = self.uses.get(&key) else { return 0 };
        for &u in uses {
            let n = graph.node(u);
            if !n.is_operation(ScalarOp::ULessThan32) || n.operands.len() != 2 || n.operands[0] != key {
                continue;
            }
            if let Some(limit) = graph.constant_u32(n.operands[1]) {
                if limit > 0 && limit <= MAX_DENSE_INDIRECT_IMAGE_ENTRIES {
                    return limit;
                }
            }
        }
        0
    }

    /// Post-process kernels select a descriptor per active bit of a scalar
    /// mask: mask → s_ff1 → v_mov → (bit * stride + table) → global_load →
    /// v_readfirstlane → key << 5 → s_load_dwordx8. The host resolves the
    /// same finite key set from the mask and the index table.
    fn try_create_wave_indexed_image_selector(&self, key: ValueId, reads: &[ValueId]) -> Option<WaveIndexedImageSelector> {
        let graph = self.graph();
        let key_node = graph.node(key);
        if key_node.kind != ValueKind::FirstLane || key_node.operands.len() != 2 || reads.is_empty() {
            return None;
        }
        let instructions = &graph.program.instructions;
        let first_lane_index = find_instruction_index(instructions, key_node.payload as u32)?;
        let first_lane = &instructions[first_lane_index];
        if first_lane.destinations.len() != 1
            || first_lane.sources.is_empty()
            || first_lane.sources[0].kind != OperandKind::VectorRegister
            || first_lane.destinations[0].kind != OperandKind::ScalarRegister
            || !((first_lane.opcode == Opcode::VReadfirstlaneB32 && first_lane.sources.len() == 1)
                || reads_active_lane(instructions, first_lane_index))
        {
            return None;
        }
        let vector_source = first_lane.sources[0];
        let global_index = find_last_definition(instructions, first_lane_index, vector_source)?;
        let global = match &instructions[global_index].control {
            Some(Control::GlobalMemory(g)) if g.dword_count == 1 && !g.uses_flat_address && g.offset_bytes == 0 => g.clone(),
            _ => return None,
        };
        if global.destination_vector_register != vector_source.value {
            return None;
        }
        let descriptor_memory_index = graph.node(reads[0]).memory_index();
        if descriptor_memory_index >= graph.memory.len() {
            return None;
        }
        let descriptor_load_index = find_instruction_index(instructions, graph.memory.get(descriptor_memory_index).pc)?;
        let descriptor_load = &instructions[descriptor_load_index];
        match &descriptor_load.control {
            Some(Control::ScalarMemory(c)) if c.dynamic_offset_register.is_some() => {}
            _ => return None,
        }
        if descriptor_load.sources.is_empty() || descriptor_load.sources[0] != Operand::scalar(global.scalar_address) {
            return None;
        }
        let address = Operand::vector(global.vector_address);
        let address_definition_index = find_last_definition(instructions, global_index, address)?;
        let index_data_offset = added_constant(&instructions[address_definition_index])?;
        let heap_address = Operand::scalar(global.scalar_address);
        if self_added_constant(&instructions[address_definition_index], address).is_some() {
            if let Some((mask, bit, mask_offset, stride)) =
                wave_indexed_stride(instructions, address_definition_index, address, heap_address)
            {
                if index_data_offset != 0 && stride != 0 && clears_mask_bit(instructions, 0, instructions.len(), mask, bit) {
                    return Some(WaveIndexedImageSelector { mask_offset, index_table_offset: index_data_offset, index_stride: stride });
                }
            }
        }
        // Register reuse while preparing the lane mask: the scalar data
        // dependency is still exact (mask word, its FF1, the bit clear, a
        // scalar multiply giving the index stride).
        wave_indexed_mask_stride(instructions, address_definition_index, heap_address, index_data_offset)
    }

    /// A T# whose eight dwords are contiguous scalar reads of one V# (the
    /// table) at `entry + key * 32` (a shift by 5 or a multiply by 32) with a
    /// runtime key: a bindless image table. The key stays an ordinary value;
    /// the eight reads become planning-only. When the key is itself a
    /// scalar-buffer read at `selector * stride + offset` of another V# (the
    /// material table), the material V# lets materialisation enumerate the
    /// keys the draw can reach.
    fn try_make_table_image(&mut self, handle: ValueId, pc: u32) -> Option<IndirectImagePlan> {
        let node = self.graph().node(handle).clone();
        if node.kind != ValueKind::ImageHandle || node.operands.len() != 8 {
            return None;
        }
        let reads = node.operands.to_vec();
        let mut memory_indices = Vec::with_capacity(8);
        let mut heap_handle = None;
        let mut heap_offset = None;
        let mut immediate = 0;
        for (dword, &read) in reads.iter().enumerate() {
            let (memory_index, memory) = self.scalar_read_memory(read)?;
            if !self.memory_index_belongs_to(memory_index, read) {
                return None;
            }
            if dword == 0 {
                immediate = memory.offset;
            } else if memory.offset != immediate.wrapping_add(dword as u32 * 4) {
                return None;
            }
            let (h, o) = (self.graph().operand(read, 0), self.graph().operand(read, 1));
            if heap_handle.is_some_and(|x| x != h) {
                return None;
            }
            heap_handle = Some(h);
            match heap_offset {
                None => heap_offset = Some(o),
                Some(first) if !self.graph().equivalent(first, o) => return None,
                _ => {}
            }
            memory_indices.push(memory_index);
        }
        let (key, entry_offset) = self.match_table_offset(heap_offset?)?;
        if self.graph().node(key).ty != ValueType::U32 || !reads.iter().all(|&r| self.uses_only(r, &[handle])) {
            return None;
        }
        let (heap_index, heap_source) = self.make_runtime_source(heap_handle?, ValueKind::BufferHandle, 4, pc)?;
        let entry_offset = entry_offset.wrapping_add(immediate);
        let mut selector = IndirectImageSelector::new(heap_index, heap_index, 0, 0);
        let mut material_dwords = heap_source.dwords.clone();
        if let Some((material_index, material_memory)) = self.scalar_read_memory(key) {
            let material_offset = material_memory.offset;
            if self.memory_index_belongs_to(material_index, key) {
                let graph = self.graph();
                let (material_handle, material_key_offset) = (graph.operand(key, 0), graph.operand(key, 1));
                if let Some((_, stride, offset)) = self.match_material_offset(material_key_offset) {
                    if let Some((index, source)) = self.make_runtime_source(material_handle, ValueKind::BufferHandle, 4, pc) {
                        selector.material_source = index;
                        selector.selector_stride = stride;
                        selector.selector_offset = offset.wrapping_add(material_offset);
                        material_dwords = source.dwords;
                    }
                }
            }
        }
        selector.table = Some(TableImageSelector { entry_offset, has_material: selector.material_source != heap_index });
        let mut dwords = heap_source.dwords[..4].to_vec();
        dwords.extend_from_slice(&material_dwords[..4]);
        let source = self.intern_source(DescriptorSource { dwords, indirect_image: Some(selector) });
        Some(IndirectImagePlan {
            handle,
            source,
            key,
            heap_source: heap_index,
            key_is_address_offset: false,
            suppress_memory_reads: true,
            memory: memory_indices,
            reads,
        })
    }

    /// `key * 32` (a shift by 5 or a multiply), optionally plus an immediate:
    /// `(key, entry offset)`.
    fn match_table_offset(&self, mut value: ValueId) -> Option<(ValueId, u32)> {
        let graph = self.graph();
        let mut entry = 0;
        let n = graph.node(value);
        if n.is_operation(ScalarOp::IAdd32) {
            if let Some(c) = graph.constant_u32(n.operands[0]) {
                entry = c;
                value = n.operands[1];
            } else if let Some(c) = graph.constant_u32(n.operands[1]) {
                entry = c;
                value = n.operands[0];
            } else {
                return None;
            }
        }
        let n = graph.node(value);
        if n.operands.len() != 2 {
            return None;
        }
        if n.is_operation(ScalarOp::ShiftLeft32) {
            return (graph.constant_u32(n.operands[1]) == Some(5)).then_some((n.operands[0], entry));
        }
        if !n.is_operation(ScalarOp::IMul32) {
            return None;
        }
        if graph.constant_u32(n.operands[1]) == Some(32) {
            Some((n.operands[0], entry))
        } else if graph.constant_u32(n.operands[0]) == Some(32) {
            Some((n.operands[1], entry))
        } else {
            None
        }
    }

    fn try_make_direct_image(&mut self, handle: ValueId) -> Option<IndirectImagePlan> {
        let node = self.graph().node(handle).clone();
        if node.kind != ValueKind::ImageHandle || node.operands.len() != 8 {
            return None;
        }
        let reads = node.operands.to_vec();
        let mut memory_indices = Vec::with_capacity(8);
        let mut can_suppress = true;
        for &read in &reads {
            let graph = self.graph();
            let n = graph.node(read);
            let memory_index = n.memory_index();
            if n.kind != ValueKind::ScalarAddressWord
                || memory_index >= graph.memory.len()
                || !self.memory_index_belongs_to(memory_index, read)
                || !self.uses_only(read, &[handle])
            {
                return None;
            }
            let memory = graph.memory.get(memory_index).clone();
            if memory.kind != MemoryResourceKind::ScalarAddress || memory.data_bits != 32 || memory.data_dwords != 1 {
                return None;
            }
            can_suppress &= self.has_only_image_consumers(&memory, handle);
            memory_indices.push(memory_index);
        }
        let graph = self.graph();
        let key_read = reads[0];
        let key_memory = graph.memory.get(graph.node(key_read).memory_index()).clone();
        let key_instruction = graph.program.instructions.iter().find(|i| i.pc == key_memory.pc)?;
        match &key_instruction.control {
            Some(Control::ScalarMemory(c)) if key_memory.component_index == 0 && c.dynamic_offset_register.is_some() => {}
            _ => return None,
        }
        let block = graph.cfg.blocks.iter().find(|b| key_memory.pc >= b.start_pc && key_memory.pc < b.end_pc)?;
        if memory_indices.iter().any(|&i| {
            let pc = graph.memory.get(i).pc;
            pc < block.start_pc || pc >= block.end_pc
        }) {
            return None;
        }
        // Include the all-zero-input result unless the scan's incoming edge
        // proves it cannot occur.
        let mut pending: Vec<ValueId> = reads.iter().map(|&r| graph.operand(r, 1)).collect();
        let mut visited = HashSet::new();
        let mut selector = None;
        while let Some(value) = pending.pop() {
            if !visited.insert(value) {
                continue;
            }
            let n = graph.node(value);
            if n.is_operation(ScalarOp::FindLowestBit32) {
                if selector.is_some_and(|s| s != value) {
                    return None;
                }
                selector = Some(value);
                continue;
            }
            if !n.is_constant() && n.kind != ValueKind::Operation {
                return None;
            }
            pending.extend(n.operands.iter().copied());
        }
        let selector = selector?;
        let mut results: Vec<u32> = (0..32).collect();
        if !graph.has_non_zero_bit_scan_input(selector) {
            results.push(u32::MAX);
        }
        let mut candidates = Vec::new();
        let mut sources = Vec::new();
        let mut keys = HashSet::new();
        for result in results {
            let replacement = self.plan.graph.constant(result);
            let replacements = HashMap::from([(selector, replacement)]);
            let mut memo = HashMap::new();
            let key = self.plan.graph.substitute(self.plan.graph.operand(key_read, 1), &replacements, &mut memo);
            let key_value = match self.plan.graph.node(key) {
                n if n.is_constant() && n.ty == ValueType::U32 => n.payload as u32,
                _ => return None,
            };
            if !keys.insert(key_value) {
                return None;
            }
            let dwords: Vec<ValueId> = reads.iter().map(|&r| self.plan.graph.substitute(r, &replacements, &mut memo)).collect();
            let source = DescriptorSource { dwords, indirect_image: None };
            self.validate_source(&source).ok()?;
            sources.push(source);
            candidates.push(DirectImageCandidate { offset: key_value, source: 0 });
        }
        for (index, source) in sources.iter().enumerate() {
            candidates[index].source = self.intern_source(source.clone());
        }
        let mut selector_info = IndirectImageSelector::new(0, 0, 0, 0);
        selector_info.direct_candidates = Some(Arc::new(candidates));
        let source = self.intern_source(DescriptorSource { dwords: sources[0].dwords.clone(), indirect_image: Some(selector_info) });
        Some(IndirectImagePlan {
            handle,
            source,
            key: key_read,
            heap_source: 0,
            key_is_address_offset: true,
            suppress_memory_reads: can_suppress,
            memory: memory_indices,
            reads,
        })
    }

    /// The loaded register feeds only the image access (the resource-graph
    /// uses omit ordinary shader arithmetic, so the instructions are
    /// checked before a load is removed).
    fn has_only_image_consumers(&self, memory: &MemoryAccessInfo, handle: ValueId) -> bool {
        let graph = self.graph();
        let instructions = &graph.program.instructions;
        let Some(load) = instructions.iter().find(|i| i.pc == memory.pc) else { return false };
        let Some(&destination) = load.destinations.get(memory.component_index as usize) else { return false };
        if destination.kind != OperandKind::ScalarRegister || destination.value >= 106 {
            return false;
        }
        let flow = &graph.cfg;
        let Some(initial) = flow.blocks.iter().position(|b| load.pc >= b.start_pc && load.pc < b.end_pc) else { return false };
        let mut pending = VecDeque::from([(initial, load.next_pc())]);
        let mut visited = HashSet::new();
        while let Some(position) = pending.pop_front() {
            if !visited.insert(position) {
                continue;
            }
            let block = flow.blocks[position.0];
            let mut overwritten = false;
            for instruction in instructions.iter().filter(|i| i.pc >= position.1 && i.pc < block.end_pc) {
                let name = instruction.opcode.name();
                if name.to_ascii_lowercase().contains("rel") || name.contains("GprIdx") {
                    return false;
                }
                let allowed_image = matches!(instruction.control, Some(Control::Image(_)))
                    && graph.memory.index_of(instruction.pc, 0).and_then(|i| self.plan.accesses[i]).and_then(|a| a.handle) == Some(handle);
                if !allowed_image
                    && (reads_scalar(instruction, destination.value)
                        || (instruction.encoding == Encoding::Sopk
                            && matches!(instruction.opcode, Opcode::SAddkI32 | Opcode::SMulkI32)
                            && instruction.destinations.contains(&destination)))
                {
                    return false;
                }
                let wide = name.contains("64");
                let scalar_destination = instruction.control.as_ref().and_then(|c| c.scalar_destination());
                overwritten = instruction.destinations.iter().any(|t| {
                    t.kind == OperandKind::ScalarRegister && (*t == destination || (wide && t.value + 1 == destination.value))
                }) || scalar_destination.is_some_and(|s| destination.value >= s && destination.value - s < 2);
                if overwritten {
                    break;
                }
            }
            if !overwritten {
                for &successor in &flow.successors[position.0] {
                    pending.push_back((successor, flow.blocks[successor].start_pc));
                }
            }
        }
        true
    }
}

fn reads_scalar(instruction: &Instruction, register: u32) -> bool {
    let in_range = |first: u32, width: u32| register >= first && register - first < width;
    let name = instruction.opcode.name();
    let width = if name.contains("64") { 2 } else { 1 };
    if instruction.sources.iter().any(|s| s.kind == OperandKind::ScalarRegister && in_range(s.value, width)) {
        return true;
    }
    match &instruction.control {
        Some(Control::Image(image)) => in_range(image.scalar_resource, 8) || in_range(image.scalar_sampler, 4),
        Some(Control::BufferMemory(buffer)) => in_range(buffer.scalar_resource, 4),
        Some(Control::GlobalMemory(global)) => in_range(global.scalar_address, 2),
        Some(Control::ScalarMemory(scalar)) => {
            (!instruction.sources.is_empty()
                && in_range(instruction.sources[0].value, if name.starts_with("SBuffer") { 4 } else { 2 }))
                || scalar.dynamic_offset_register == Some(register)
        }
        _ => false,
    }
}

fn find_last_definition(instructions: &[Instruction], before: usize, destination: Operand) -> Option<usize> {
    (0..before).rev().find(|&i| instructions[i].destinations.contains(&destination))
}

fn find_instruction_index(instructions: &[Instruction], pc: u32) -> Option<usize> {
    instructions.iter().position(|i| i.pc == pc)
}

fn self_added_constant(instruction: &Instruction, destination: Operand) -> Option<u32> {
    if instruction.opcode != Opcode::VAddI32 || instruction.sources.len() != 2 || !instruction.destinations.contains(&destination) {
        return None;
    }
    let (left, right) = (instruction.sources[0], instruction.sources[1]);
    if left == destination {
        inline_constant(right)
    } else if right == destination {
        inline_constant(left)
    } else {
        None
    }
}

fn added_constant(instruction: &Instruction) -> Option<u32> {
    if instruction.opcode != Opcode::VAddI32 || instruction.sources.len() != 2 {
        return None;
    }
    inline_constant(instruction.sources[0]).or_else(|| inline_constant(instruction.sources[1]))
}

fn wave_indexed_mask_stride(instructions: &[Instruction], before: usize, heap_address: Operand, index_table_offset: u32) -> Option<WaveIndexedImageSelector> {
    if index_table_offset == 0 {
        return None;
    }
    for load_index in (0..before).rev() {
        let mask_load = &instructions[load_index];
        let mask_control = match &mask_load.control {
            Some(Control::ScalarMemory(c))
                if c.destination_count == 1 && c.immediate_offset_bytes >= 0 && c.dynamic_offset_register.is_none() =>
            {
                c
            }
            _ => continue,
        };
        if mask_load.sources.is_empty() || mask_load.destinations.len() != 1 || mask_load.sources[0] != heap_address {
            continue;
        }
        let mask_register = mask_load.destinations[0];
        if mask_register.kind != OperandKind::ScalarRegister {
            continue;
        }
        for scan_index in load_index + 1..before {
            let scan = &instructions[scan_index];
            if scan.opcode != Opcode::SFF1I32B32
                || scan.sources.len() != 1
                || scan.destinations.len() != 1
                || scan.sources[0] != mask_register
                || scan.destinations[0].kind != OperandKind::ScalarRegister
            {
                continue;
            }
            let bit_register = scan.destinations[0];
            if !clears_mask_bit(instructions, scan_index + 1, before, mask_register, bit_register) {
                continue;
            }
            for multiply in &instructions[scan_index + 1..before] {
                if multiply.opcode != Opcode::SMulI32 || multiply.sources.len() != 2 || !multiply.sources.contains(&bit_register) {
                    continue;
                }
                let stride_operand = if multiply.sources[0] == bit_register { multiply.sources[1] } else { multiply.sources[0] };
                match inline_constant(stride_operand) {
                    Some(stride) if stride != 0 => {
                        return Some(WaveIndexedImageSelector {
                            mask_offset: mask_control.immediate_offset_bytes as u32,
                            index_table_offset,
                            index_stride: stride,
                        })
                    }
                    _ => continue,
                }
            }
        }
    }
    None
}

/// `(mask register, bit register, mask offset, stride)`.
fn wave_indexed_stride(
    instructions: &[Instruction],
    address_add_index: usize,
    address: Operand,
    heap_address: Operand,
) -> Option<(Operand, Operand, u32, u32)> {
    let stride_add_index = find_last_definition(instructions, address_add_index, address)?;
    let stride_add = &instructions[stride_add_index];
    if stride_add.opcode != Opcode::VLshlAddU32 || stride_add.sources.len() != 3 || stride_add.sources[0] != address || stride_add.sources[2] != address {
        return None;
    }
    let outer_shift = inline_constant(stride_add.sources[1]).filter(|&s| s < 31)?;
    let inner_shift_index = find_last_definition(instructions, stride_add_index, address)?;
    let inner_shift = &instructions[inner_shift_index];
    if inner_shift.opcode != Opcode::VLshlrevB32 || inner_shift.sources.len() != 2 {
        return None;
    }
    let inner_amount = inline_constant(inner_shift.sources[0]).filter(|&s| s < 31)?;
    let bit_vector = inner_shift.sources[1];
    if bit_vector.kind != OperandKind::VectorRegister {
        return None;
    }
    let move_index = find_last_definition(instructions, inner_shift_index, bit_vector)?;
    let mv = &instructions[move_index];
    if mv.opcode != Opcode::VMovB32 || mv.sources.len() != 1 || mv.sources[0].kind != OperandKind::ScalarRegister {
        return None;
    }
    let bit_scalar = mv.sources[0];
    let bit_scan_index = find_last_definition(instructions, move_index, bit_scalar)?;
    let scan = &instructions[bit_scan_index];
    if scan.opcode != Opcode::SFF1I32B32 || scan.sources.len() != 1 || scan.sources[0].kind != OperandKind::ScalarRegister {
        return None;
    }
    let mask = scan.sources[0];
    let mask_load_index = find_last_definition(instructions, bit_scan_index, mask)?;
    let mask_load = &instructions[mask_load_index];
    let mask_control = match &mask_load.control {
        Some(Control::ScalarMemory(c)) if c.destination_count == 1 && c.immediate_offset_bytes >= 0 && c.dynamic_offset_register.is_none() => c,
        _ => return None,
    };
    if mask_load.sources.is_empty() || mask_load.sources[0] != heap_address {
        return None;
    }
    let combined = (1u64 << inner_amount) * ((1u64 << outer_shift) + 1);
    if combined == 0 || combined > u32::MAX as u64 {
        return None;
    }
    Some((mask, bit_scalar, mask_control.immediate_offset_bytes as u32, combined as u32))
}

fn clears_mask_bit(instructions: &[Instruction], start: usize, end: usize, mask: Operand, bit: Operand) -> bool {
    let end = end.min(instructions.len());
    for index in start..end {
        let instruction = &instructions[index];
        if instruction.opcode == Opcode::SBitset0B32 && instruction.destinations.contains(&mask) && instruction.sources.contains(&bit) {
            return true;
        }
        if instruction.opcode != Opcode::SLshlB32
            || instruction.sources.len() != 2
            || instruction.destinations.len() != 1
            || inline_constant(instruction.sources[0]) != Some(1)
            || instruction.sources[1] != bit
            || instruction.destinations[0].kind != OperandKind::ScalarRegister
        {
            continue;
        }
        let single = instruction.destinations[0];
        for clear in &instructions[index + 1..end] {
            if clear.destinations.contains(&mask)
                && clear.sources.len() == 2
                && ((clear.opcode == Opcode::SXorB32 && clear.sources.contains(&mask) && clear.sources.contains(&single))
                    || (clear.opcode == Opcode::SAndn2B32 && clear.sources[0] == mask && clear.sources[1] == single))
            {
                return true;
            }
            if clear.destinations.contains(&single)
                || clear.destinations.contains(&mask)
                || clear.destinations.contains(&bit)
                || ((single.value == 106 || single.value == 107) && clear.opcode.name().starts_with('V'))
            {
                break;
            }
        }
    }
    false
}

fn writes_scalar_pair(instruction: &Instruction, register: Operand) -> bool {
    let name = instruction.opcode.name();
    if (register.value == 106 || register.value == 107) && name.starts_with('V') {
        return true;
    }
    if let Some(Control::Vop3(Vop3Control { scalar_destination: Some(d), .. })) = &instruction.control {
        if d + 1 >= register.value && *d <= register.value + 1 {
            return true;
        }
    }
    if register.value == 126 && (name.contains("Saveexec") || name.starts_with("VCmpx")) {
        return true;
    }
    let width = if name.contains("64") { 2 } else { 1 };
    instruction
        .destinations
        .iter()
        .any(|d| d.kind == OperandKind::ScalarRegister && d.value + width > register.value && d.value <= register.value + 1)
}

fn branch_target_of(instruction: &Instruction) -> Option<u32> {
    if branch::is_branch(instruction) {
        branch::branch_target(instruction)
    } else {
        None
    }
}

/// A V_READLANE at the lane S_FF1 picks from a copy of EXEC, inside a loop
/// that clears the lane and repeats: every active lane is visited.
fn reads_active_lane(instructions: &[Instruction], read_lane_index: usize) -> bool {
    let exec = Operand::scalar(126);
    let read_lane = &instructions[read_lane_index];
    if read_lane.opcode != Opcode::VReadlaneB32 || read_lane.sources.len() < 2 || read_lane.sources[1].kind != OperandKind::ScalarRegister {
        return false;
    }
    let lane = read_lane.sources[1];
    let Some(scan_index) = find_last_definition(instructions, read_lane_index, lane) else { return false };
    let scan = &instructions[scan_index];
    if scan.opcode != Opcode::SFF1I32B64 || scan.sources.len() != 1 || scan.sources[0].kind != OperandKind::ScalarRegister {
        return false;
    }
    let candidates = scan.sources[0];
    let Some(copy_index) = find_last_definition(instructions, scan_index, candidates) else { return false };
    let copy = &instructions[copy_index];
    if copy.opcode != Opcode::SMovB64 || copy.sources.len() != 1 || copy.sources[0] != exec {
        return false;
    }
    for instruction in &instructions[copy_index + 1..read_lane_index] {
        if writes_scalar_pair(instruction, exec) || writes_scalar_pair(instruction, candidates) || branch_target_of(instruction).is_some() {
            return false;
        }
    }
    let mut back_index = None;
    for (index, instruction) in instructions.iter().enumerate().skip(read_lane_index + 1) {
        let Some(target) = branch_target_of(instruction) else { continue };
        if target > instructions[copy_index].pc && target <= instructions[scan_index].pc {
            back_index = Some(index);
            break;
        }
    }
    let Some(back_index) = back_index else { return false };
    let Some(restore_index) = (read_lane_index + 1..back_index).rev().find(|&i| writes_scalar_pair(&instructions[i], exec)) else {
        return false;
    };
    let restore = &instructions[restore_index];
    if restore.opcode != Opcode::SMovB64 || restore.sources.len() != 1 || restore.sources[0].kind != OperandKind::ScalarRegister {
        return false;
    }
    let saved = restore.sources[0];
    match find_last_definition(instructions, restore_index, saved) {
        Some(save_index) if save_index > read_lane_index && instructions[save_index].opcode == Opcode::SAndSaveexecB64 => {
            for index in read_lane_index + 1..back_index {
                let instruction = &instructions[index];
                if index > save_index && index < restore_index && writes_scalar_pair(instruction, saved) {
                    return false;
                }
                if writes_scalar_pair(instruction, candidates)
                    && (instruction.opcode != Opcode::SAndn2B64 || instruction.sources.len() != 2 || instruction.sources[0] != candidates)
                {
                    return false;
                }
                if let Some(target) = branch_target_of(instruction) {
                    if target <= instruction.pc || target > instructions[restore_index].pc {
                        return false;
                    }
                }
            }
            true
        }
        _ => false,
    }
}
