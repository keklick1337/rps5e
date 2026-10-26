// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Bounded runtime V# tables: a V# whose four dwords are read from one
//! static scalar buffer (the SRT) at an offset driven by a canonical unsigned
//! induction loop. The loop guard bounds the descriptors the access can
//! select, so a back-end can enumerate them into native candidates and pick
//! one at run time. No descriptor is invented and the SRT's size is never
//! taken as proof of which entries are valid; each candidate is decoded and
//! validated again at materialisation.

use super::info::NO_INDEX;
use super::memory::MemoryResourceKind;
use super::plan::ShaderResourcePlan;
use super::value::{ScalarOp, ValueId, ValueKind};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferCandidateTablePlan {
    /// Descriptor source of the static SRT the V# dwords are read from.
    pub source_srt_resource: u32,
    pub srt_handle: ValueId,
    /// The full dynamic byte offset: base + index * stride.
    pub offset_expression: ValueId,
    /// The induction variable whose guard bounds the candidates.
    pub index_expression: ValueId,
    pub initial: ValueId,
    pub step: u32,
    /// The guard's limit; `None` when it is a compile-time constant.
    pub limit: Option<ValueId>,
    pub limit_inclusive: bool,
    pub stride: u32,
    pub base_offset: u32,
    /// First candidate's byte offset.
    pub min_offset: u32,
    /// One past the last candidate's descriptor (`u32::MAX` when run-time).
    pub max_offset: u32,
    /// Statically proven candidate count, -1 for a run-time limit.
    pub count: i32,
    /// The SRT's declared byte size when constant (a memory-safety bound
    /// only), else `u32::MAX`.
    pub srt_byte_extent: u32,
    pub memory_indices: Vec<usize>,
    pub cap: i32,
}

impl BufferCandidateTablePlan {
    /// Hard ceiling on the candidates one table may bind.
    pub const DEFAULT_CAP: i32 = 64;
    pub const DESCRIPTOR_BYTE_SIZE: u32 = 16;

    pub fn is_statically_bounded(&self) -> bool {
        self.count >= 0
    }

    /// Byte distance between consecutive candidates.
    pub fn candidate_spacing(&self) -> u32 {
        self.step.wrapping_mul(self.stride)
    }

    pub fn static_candidate_offset(&self, index: i32) -> u32 {
        self.min_offset.wrapping_add((index as u32).wrapping_mul(self.candidate_spacing()))
    }

    /// The candidate count for a run-time `limit` (0 for an empty loop);
    /// `None` when the initial value is not constant or the count overflows.
    pub fn resolve_count(&self, initial: Option<u32>, limit: u32) -> Option<i32> {
        let first = initial? as u64;
        let bound = if self.limit_inclusive { limit as u64 + 1 } else { limit as u64 };
        if bound <= first {
            return Some(0);
        }
        let iterations = (bound - first).div_ceil(self.step as u64);
        (iterations <= i32::MAX as u64).then_some(iterations as i32)
    }
}

/// Recognise a bounded runtime V# table for the buffer access `memory_index`
/// through `handle`.
pub(super) fn try_plan(plan: &ShaderResourcePlan, handle: ValueId, memory_index: usize) -> Option<BufferCandidateTablePlan> {
    let graph = &plan.graph;
    let node = graph.node(handle);
    if node.kind != ValueKind::BufferHandle || node.operands.len() != 4 {
        return None;
    }
    // Every dword is a scalar-buffer read at one shared dynamic offset with
    // the component immediates 0, 4, 8, 12 of one dwordx4 read.
    let mut srt = None;
    let mut dynamic_offset = None;
    for (word, &read) in node.operands.iter().enumerate() {
        let r = graph.node(read);
        if r.kind != ValueKind::ScalarBufferWord || r.memory_index() >= graph.memory.len() {
            return None;
        }
        let memory = graph.memory.get(r.memory_index());
        if memory.kind != MemoryResourceKind::ScalarBuffer || memory.data_bits != 32 || memory.data_dwords != 1 || memory.offset != word as u32 * 4 {
            return None;
        }
        match srt {
            None => srt = Some(r.operands[0]),
            Some(s) if !graph.equivalent(s, r.operands[0]) => return None,
            _ => {}
        }
        match dynamic_offset {
            None => dynamic_offset = Some(r.operands[1]),
            Some(o) if !graph.equivalent(o, r.operands[1]) => return None,
            _ => {}
        }
    }
    let (srt, dynamic_offset) = (srt?, dynamic_offset?);

    // offset = base + index * stride
    let mut offset = dynamic_offset;
    let mut base_offset = 0;
    let n = graph.node(offset);
    if n.is_operation(ScalarOp::IAdd32) {
        if let Some(c) = graph.constant_u32(n.operands[0]) {
            base_offset = c;
            offset = n.operands[1];
        } else if let Some(c) = graph.constant_u32(n.operands[1]) {
            base_offset = c;
            offset = n.operands[0];
        }
    }
    let n = graph.node(offset);
    if !n.is_operation(ScalarOp::IMul32) {
        return None;
    }
    let (stride, index) = if let Some(c) = graph.constant_u32(n.operands[0]) {
        (c, n.operands[1])
    } else if let Some(c) = graph.constant_u32(n.operands[1]) {
        (c, n.operands[0])
    } else {
        return None;
    };
    if stride < BufferCandidateTablePlan::DESCRIPTOR_BYTE_SIZE {
        return None;
    }
    let (initial, step, limit, inclusive) = prove_induction_bound(plan, index)?;
    let initial_value = graph.constant_u32(initial)?;
    let srt_extent = srt_byte_extent(plan, srt);
    let min_offset = base_offset.wrapping_add(initial_value.wrapping_mul(stride));
    let mut table = BufferCandidateTablePlan {
        source_srt_resource: NO_INDEX,
        srt_handle: srt,
        offset_expression: dynamic_offset,
        index_expression: index,
        initial,
        step,
        limit: if graph.is_constant(limit) { None } else { Some(limit) },
        limit_inclusive: inclusive,
        stride,
        base_offset,
        min_offset,
        max_offset: 0,
        count: -1,
        srt_byte_extent: srt_extent,
        memory_indices: vec![memory_index],
        cap: BufferCandidateTablePlan::DEFAULT_CAP,
    };
    let max_offset;
    if let Some(limit_value) = graph.constant_u32(limit) {
        let count = table.resolve_count(Some(initial_value), limit_value)?;
        if count <= 0 || count > table.cap {
            return None;
        }
        table.count = count;
        max_offset = table.static_candidate_offset(count - 1).wrapping_add(BufferCandidateTablePlan::DESCRIPTOR_BYTE_SIZE);
    } else {
        if !plan.validate_runtime_value(limit) {
            return None;
        }
        // A run-time limit is capped and memory-checked at materialisation.
        table.count = -1;
        max_offset = u32::MAX;
    }
    if max_offset < min_offset || (max_offset != u32::MAX && max_offset > srt_extent) {
        return None;
    }
    table.max_offset = max_offset;
    Some(table)
}

/// A canonical unsigned induction: a phi with a constant initial value and
/// an add-recurrence, guarded by an unsigned compare against the phi at the
/// loop header or one of its predecessors. `(initial, step, limit,
/// inclusive)`.
fn prove_induction_bound(plan: &ShaderResourcePlan, index: ValueId) -> Option<(ValueId, u32, ValueId, bool)> {
    let graph = &plan.graph;
    let phi = graph.node(index);
    if phi.kind != ValueKind::Phi {
        return None;
    }
    let mut found = None;
    let mut initial_count = 0;
    let mut step = 0;
    for &operand in phi.operands.iter() {
        let n = graph.node(operand);
        if n.is_operation(ScalarOp::IAdd32) {
            if n.operands[0] == index && graph.is_constant(n.operands[1]) {
                step = graph.constant_u32(n.operands[1]).unwrap();
            } else if n.operands[1] == index && graph.is_constant(n.operands[0]) {
                step = graph.constant_u32(n.operands[0]).unwrap();
            } else {
                continue;
            }
        } else if operand != index {
            found = Some(operand);
            initial_count += 1;
        }
    }
    let initial = found?;
    if step == 0 || initial_count != 1 {
        return None;
    }
    let flow = &graph.cfg;
    let mut limit: Option<ValueId> = None;
    let mut inclusive = false;
    let mut guarded = false;
    for (&pc, &condition) in &graph.branch_conditions {
        let c = graph.node(condition);
        if c.kind != ValueKind::Operation {
            continue;
        }
        let (bound, bound_inclusive) = match c.op {
            ScalarOp::ULessThan32 if matches_index(plan, c.operands[0], index, step) => (c.operands[1], false),
            ScalarOp::ULessThanEqual32 if matches_index(plan, c.operands[0], index, step) => (c.operands[1], true),
            ScalarOp::UGreaterThan32 if matches_index(plan, c.operands[1], index, step) => (c.operands[0], false),
            ScalarOp::UGreaterThanEqual32 if matches_index(plan, c.operands[1], index, step) => (c.operands[0], true),
            _ => continue,
        };
        if !graph.is_constant(bound) && !plan.validate_runtime_value(bound) {
            continue;
        }
        let Some(block) = flow.blocks.iter().position(|b| pc >= b.start_pc && pc < b.end_pc) else { continue };
        // The guard must control the recurrence.
        let header = phi.phi_block();
        if block != header && !flow.predecessors[header].contains(&block) {
            continue;
        }
        if let Some(l) = limit {
            if !graph.equivalent(l, bound) {
                return None;
            }
        }
        limit = Some(bound);
        inclusive = bound_inclusive;
        guarded = true;
    }
    if !guarded {
        return None;
    }
    Some((initial, step, limit?, inclusive))
}

/// The guard compares the induction value itself or its next value.
fn matches_index(plan: &ShaderResourcePlan, operand: ValueId, index: ValueId, step: u32) -> bool {
    let graph = &plan.graph;
    if graph.equivalent(operand, index) {
        return true;
    }
    let n = graph.node(operand);
    if !n.is_operation(ScalarOp::IAdd32) {
        return false;
    }
    if graph.equivalent(n.operands[0], index) && graph.is_constant(n.operands[1]) {
        return graph.constant_u32(n.operands[1]) == Some(step);
    }
    if graph.equivalent(n.operands[1], index) && graph.is_constant(n.operands[0]) {
        return graph.constant_u32(n.operands[0]) == Some(step);
    }
    false
}

/// The SRT's declared byte size when stride and record count are constant.
fn srt_byte_extent(plan: &ShaderResourcePlan, srt: ValueId) -> u32 {
    let graph = &plan.graph;
    let operands = graph.operands(srt);
    if operands.len() != 4 {
        return u32::MAX;
    }
    let stride_word = graph.resolve_invariant_phi(operands[1]).unwrap_or(operands[1]);
    let records_word = graph.resolve_invariant_phi(operands[2]).unwrap_or(operands[2]);
    let (Some(stride_word), Some(records)) = (graph.constant_u32(stride_word), graph.constant_u32(records_word)) else {
        return u32::MAX;
    };
    let stride = (stride_word >> 16) & 0x3FFF;
    let size = if stride == 0 { records as u64 } else { stride as u64 * records as u64 };
    size.min(u32::MAX as u64) as u32
}
