// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Which graph values the host can evaluate per draw.

use super::graph::ScalarValueGraph;
use super::memory::MemoryResourceKind;
use super::value::{ScalarOp, ValueId, ValueKind};
use std::collections::HashSet;

/// Accepts a value only when every node down to its leaves is one the host
/// can evaluate per draw: user data, shader base, scalar reads, flattened
/// table words, invariant phis and uniform operations.
pub struct RuntimeValueValidator<'g> {
    graph: &'g ScalarValueGraph,
    user_data_base: u32,
    user_data_count: u32,
    table_read_count: usize,
    active_mask: Option<ValueId>,
    visiting: HashSet<ValueId>,
    /// Set when a failure traces back to a phi that is not loop-invariant
    /// (e.g. a hash-table probe): the value genuinely has no single source,
    /// as opposed to an operation or memory shape that is not recognised.
    pub control_dependent: bool,
}

impl<'g> RuntimeValueValidator<'g> {
    pub fn new(graph: &'g ScalarValueGraph, table_read_count: usize) -> Self {
        Self::with_mask(graph, graph.user_data_base, graph.user_data_count, table_read_count, None)
    }

    fn with_mask(
        graph: &'g ScalarValueGraph,
        user_data_base: u32,
        user_data_count: u32,
        table_read_count: usize,
        active_mask: Option<ValueId>,
    ) -> Self {
        RuntimeValueValidator {
            graph,
            user_data_base,
            user_data_count,
            table_read_count,
            active_mask,
            visiting: HashSet::new(),
            control_dependent: false,
        }
    }

    pub fn validate(&mut self, value: ValueId) -> bool {
        if self.graph.is_constant(value) {
            return true;
        }
        if !self.visiting.insert(value) {
            return false;
        }
        let result = self.validate_node(value);
        self.visiting.remove(&value);
        result
    }

    fn validate_node(&mut self, value: ValueId) -> bool {
        let graph = self.graph;
        let node = graph.node(value);
        // Under a first-lane read, a select on that lane's active mask is its
        // active arm.
        if node.kind == ValueKind::Select && Some(node.operands[0]) == self.active_mask {
            return self.validate(node.operands[1]);
        }
        match node.kind {
            ValueKind::Undefined => return false,
            ValueKind::UserData => {
                let r = node.payload as u32;
                return r >= self.user_data_base && r - self.user_data_base < self.user_data_count;
            }
            ValueKind::ShaderBase | ValueKind::MemoryAperture => return true,
            ValueKind::Phi => {
                return match graph.resolve_invariant_phi(value) {
                    None => {
                        self.control_dependent = true;
                        false
                    }
                    Some(invariant) => self.validate(invariant),
                }
            }
            ValueKind::FirstLane => {
                if node.operands.len() != 2 {
                    return false;
                }
                let mut nested = RuntimeValueValidator::with_mask(
                    graph,
                    self.user_data_base,
                    self.user_data_count,
                    self.table_read_count,
                    Some(node.operands[1]),
                );
                let ok = nested.validate(node.operands[0]);
                if nested.control_dependent {
                    self.control_dependent = true;
                }
                return ok;
            }
            ValueKind::ResourceTableWord => return node.payload < self.table_read_count as u64,
            ValueKind::ScalarAddressWord => {
                if !self.is_raw_read(value) || graph.kind(node.operands[0]) != ValueKind::AddressHandle {
                    return false;
                }
            }
            ValueKind::ScalarBufferWord => {
                if !self.is_raw_read(value) || graph.kind(node.operands[0]) != ValueKind::BufferHandle {
                    return false;
                }
            }
            ValueKind::BufferHandle | ValueKind::SamplerHandle => {
                if node.operands.len() != 4 {
                    return false;
                }
            }
            ValueKind::ImageHandle => {
                if node.operands.len() != 8 {
                    return false;
                }
            }
            ValueKind::AddressHandle => {
                if node.operands.len() != 2 {
                    return false;
                }
            }
            ValueKind::Select => {}
            ValueKind::Operation => {
                if node.op == ScalarOp::Extract64 && !matches!(graph.constant_u32(node.operands[1]), Some(0 | 1)) {
                    return false;
                }
                if !node.op.is_uniform() {
                    return false;
                }
            }
            ValueKind::Constant => return true,
        }
        for i in 0..node.operands.len() {
            if !self.validate(graph.operand(value, i)) {
                return false;
            }
        }
        true
    }

    /// A raw read is a scalar load whose memory record has the matching
    /// scalar kind.
    pub fn is_raw_read(&self, value: ValueId) -> bool {
        is_raw_read(self.graph, value)
    }
}

pub(super) fn is_raw_read(graph: &ScalarValueGraph, value: ValueId) -> bool {
    let node = graph.node(value);
    if !matches!(node.kind, ValueKind::ScalarAddressWord | ValueKind::ScalarBufferWord) {
        return false;
    }
    let index = node.memory_index();
    if index >= graph.memory.len() {
        return false;
    }
    let kind = graph.memory.get(index).kind;
    (node.kind == ValueKind::ScalarAddressWord && kind == MemoryResourceKind::ScalarAddress)
        || (node.kind == ValueKind::ScalarBufferWord && matches!(kind, MemoryResourceKind::ScalarBuffer | MemoryResourceKind::Buffer))
}
