// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Bounds of every device-address handle, planned after tracking. Optional:
//! an unplannable handle never fails the plan; it is reported so hosts can
//! decide.

use super::evaluator::{ResourceRuntimeInputs, RuntimeValueEvaluator, ADDRESS_MASK};
use super::memory::{FlatAddressSpace, MemoryAccess, MemoryResourceKind};
use super::plan::ShaderResourcePlan;
use super::value::{ScalarOp, ValueId, ValueKind, ValueType};
use crate::inline_constants;
use std::collections::HashSet;

/// One address handle's planned range: its base value and the signed byte
/// span its accesses reach (unbounded up to [`MAX_RANGE_BYTES`] when an
/// access has a run-time offset).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceAddressRangePlan {
    pub handle: u32,
    pub base_low: ValueId,
    pub base_high: ValueId,
    pub memory_indices: Vec<usize>,
    pub written: bool,
    pub bounded: bool,
    pub first_byte: i64,
    pub end_byte: i64,
    /// The base is host-evaluable.
    pub plannable: bool,
}

impl DeviceAddressRangePlan {
    pub fn extent(&self) -> u64 {
        if self.bounded && self.end_byte > self.first_byte {
            (self.end_byte - self.first_byte) as u64
        } else {
            0
        }
    }
}

/// One handle's evaluated base and extent for one draw.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceAddressRange {
    pub handle: u32,
    pub base: u64,
    pub size: u64,
    pub planned: bool,
    pub written: bool,
}

pub const MAX_RANGE_BYTES: u64 = 16 * 1024 * 1024;

/// Device-address paging every emitter uses.
pub mod paging {
    pub const PAGE_BITS: u32 = 14;
    pub const PAGE_SIZE: u64 = 1 << PAGE_BITS;
    pub const PAGE_OFFSET_MASK: u64 = PAGE_SIZE - 1;
    pub const ADDRESS_MASK: u64 = super::ADDRESS_MASK;
}

pub(super) fn plan(plan: &mut ShaderResourcePlan) -> Vec<DeviceAddressRangePlan> {
    let mut ranges: Vec<DeviceAddressRangePlan> = Vec::new();
    let mut handles: Vec<ValueId> = Vec::new();
    for index in 0..plan.graph.memory.len() {
        let memory = plan.graph.memory.get(index);
        if !matches!(memory.kind, MemoryResourceKind::Flat | MemoryResourceKind::Global | MemoryResourceKind::ScalarAddress) || memory.planning_only {
            continue;
        }
        let Some(access) = plan.accesses[index] else { continue };
        let Some(handle) = access.handle else { continue };
        if plan.graph.kind(handle) != ValueKind::AddressHandle || plan.graph.operands(handle).len() != 2 {
            continue;
        }
        // A FLAT address built from an aperture base is an LDS or scratch
        // pointer, routed per lane; it owns no device range.
        if memory.kind == MemoryResourceKind::Flat {
            let space = classify_flat_address(plan, plan.graph.operand(handle, 1), access.active);
            plan.graph.memory.get_mut(index).address_space = space;
            if space != FlatAddressSpace::Global {
                continue;
            }
        }
        let memory = plan.graph.memory.get(index).clone();
        let slot = match handles.iter().position(|&h| plan.graph.equivalent(h, handle)) {
            Some(s) => s,
            None => {
                handles.push(handle);
                ranges.push(DeviceAddressRangePlan {
                    handle: ranges.len() as u32,
                    base_low: plan.graph.operand(handle, 0),
                    base_high: plan.graph.operand(handle, 1),
                    memory_indices: Vec::new(),
                    written: false,
                    bounded: true,
                    first_byte: i64::MAX,
                    end_byte: i64::MIN,
                    plannable: false,
                });
                ranges.len() - 1
            }
        };
        // The immediate offset is signed; the span keeps the lowest and
        // highest byte.
        let written = matches!(memory.access, MemoryAccess::Write | MemoryAccess::Atomic);
        let bytes = memory.data_bits.div_ceil(8).max(1) as i64 * memory.data_dwords.max(1) as i64;
        let first_byte = memory.offset as i32 as i64;
        let bounded = is_bounded_offset(plan, index);
        let range = &mut ranges[slot];
        range.memory_indices.push(index);
        range.written |= written;
        range.bounded &= bounded;
        range.first_byte = range.first_byte.min(first_byte);
        range.end_byte = range.end_byte.max(first_byte + bytes);
    }
    for range in &mut ranges {
        range.plannable = plan.validate_runtime_value(range.base_low) && plan.validate_runtime_value(range.base_high);
    }
    ranges
}

type Range = (u32, u32);
const EMPTY: Range = (u32::MAX, 0);

fn is_empty(range: Range) -> bool {
    range.0 > range.1
}

fn union(left: Range, right: Range) -> Range {
    if is_empty(left) {
        right
    } else if is_empty(right) {
        left
    } else {
        (left.0.min(right.0), left.1.max(right.1))
    }
}

/// A FLAT address is local when its high dword provably lies in an
/// aperture: clamped between constants inside the apertures, or derived
/// from an aperture operand.
fn classify_flat_address(plan: &ShaderResourcePlan, high: ValueId, active: Option<ValueId>) -> FlatAddressSpace {
    let range = unsigned_range(plan, high, active, &mut HashSet::new());
    let (low, high_bound) = if is_empty(range) { (0, u32::MAX) } else { range };
    // Each aperture is one high dword (a 4 GiB window): only a range that
    // holds that single value proves it.
    if low == high_bound {
        if inline_constants::is_shared_aperture_high(low) {
            return FlatAddressSpace::Shared;
        }
        if inline_constants::is_private_aperture_high(low) {
            return FlatAddressSpace::Private;
        }
    }
    classify_by_aperture_operand(plan, high)
}

/// Conservative unsigned bounds of a 32-bit value as the access sees it.
fn unsigned_range(plan: &ShaderResourcePlan, value: ValueId, active: Option<ValueId>, visiting: &mut HashSet<ValueId>) -> Range {
    let graph = &plan.graph;
    let node = graph.node(value);
    if node.is_constant() && node.ty == ValueType::U32 {
        return (node.payload as u32, node.payload as u32);
    }
    if node.kind == ValueKind::MemoryAperture {
        let aperture = (inline_constants::decode_aperture64(node.payload as u32) >> 32) as u32;
        return (aperture, aperture);
    }
    if !visiting.insert(value) {
        return EMPTY;
    }
    let result = match node.kind {
        ValueKind::Select if node.operands.len() == 3 => {
            let on_active = active.is_some_and(|a| node.operands[0] == a || graph.equivalent(node.operands[0], a));
            if on_active {
                unsigned_range(plan, node.operands[1], active, visiting)
            } else {
                let a = unsigned_range(plan, node.operands[1], active, visiting);
                let b = unsigned_range(plan, node.operands[2], active, visiting);
                union(a, b)
            }
        }
        ValueKind::Phi => {
            let mut range = EMPTY;
            for &o in node.operands.iter() {
                range = union(range, unsigned_range(plan, o, active, visiting));
            }
            range
        }
        ValueKind::Operation if matches!(node.op, ScalarOp::UMin32 | ScalarOp::UMax32) => {
            let left = unsigned_range(plan, node.operands[0], active, visiting);
            let right = unsigned_range(plan, node.operands[1], active, visiting);
            if is_empty(left) || is_empty(right) {
                if is_empty(left) {
                    right
                } else {
                    left
                }
            } else if node.op == ScalarOp::UMin32 {
                (left.0.min(right.0), left.1.min(right.1))
            } else {
                (left.0.max(right.0), left.1.max(right.1))
            }
        }
        _ => (0, u32::MAX),
    };
    visiting.remove(&value);
    result
}

fn classify_by_aperture_operand(plan: &ShaderResourcePlan, high: ValueId) -> FlatAddressSpace {
    let (mut shared, mut private) = (false, false);
    let mut pending = vec![high];
    let mut visited = HashSet::new();
    while let Some(value) = pending.pop() {
        if !visited.insert(value) {
            continue;
        }
        let node = plan.graph.node(value);
        if node.kind == ValueKind::MemoryAperture {
            if inline_constants::is_shared_aperture(node.payload as u32) {
                shared = true;
            } else {
                private = true;
            }
            continue;
        }
        pending.extend(node.operands.iter().copied());
    }
    match (shared, private) {
        (true, true) => FlatAddressSpace::SharedOrPrivate,
        (true, false) => FlatAddressSpace::Shared,
        (false, true) => FlatAddressSpace::Private,
        _ => FlatAddressSpace::Global,
    }
}

/// Bounded when the only run-time term is the immediate offset: a scalar
/// read with a constant offset, or a global access whose vector offset is a
/// constant zero.
fn is_bounded_offset(plan: &ShaderResourcePlan, memory_index: usize) -> bool {
    let graph = &plan.graph;
    let Some(access) = plan.accesses[memory_index] else { return false };
    if graph.memory.get(memory_index).kind == MemoryResourceKind::ScalarAddress {
        return access.read.is_some_and(|r| graph.is_constant(graph.operand(r, 1)));
    }
    access.offset.is_some_and(|o| graph.constant_u32(o) == Some(0))
}

/// Evaluate the planned ranges for one draw: a bounded range spans its
/// lowest byte to its extent; an unbounded one starts at the lowest
/// immediate and takes the cap.
pub fn evaluate(plan: &ShaderResourcePlan, inputs: &ResourceRuntimeInputs<'_>) -> Vec<DeviceAddressRange> {
    let evaluator = RuntimeValueEvaluator::new(plan, *inputs);
    plan.device_address_ranges
        .iter()
        .map(|range| {
            let base = if range.plannable {
                match (evaluator.evaluate(range.base_low), evaluator.evaluate(range.base_high)) {
                    (Some(low), Some(high)) => Some((low as u64 | ((high as u64) << 32)) & ADDRESS_MASK),
                    _ => None,
                }
            } else {
                None
            };
            from_base(range, base)
        })
        .collect()
}

/// One range for a draw from its evaluated 48-bit base (None: unplanned).
pub(super) fn from_base(range: &DeviceAddressRangePlan, base: Option<u64>) -> DeviceAddressRange {
    let first_byte = if range.bounded { range.first_byte } else { range.first_byte.min(0) };
    let size = if range.bounded { range.extent() } else { MAX_RANGE_BYTES };
    DeviceAddressRange {
        handle: range.handle,
        base: base.map_or(0, |b| b.wrapping_add(first_byte as u64) & ADDRESS_MASK),
        size: if base.is_some() { size } else { 0 },
        planned: base.is_some(),
        written: range.written,
    }
}
