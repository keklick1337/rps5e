// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Collects the scalar reads reachable from descriptor handles. Reads with
//! an immediate offset get compact flattened-table slots the host fills per
//! draw; reads with a dynamic offset stay explicit.

use super::graph::ScalarValueGraph;
use super::info::{ResourceTableRead, ShaderStage};
use super::plan::PlanOptions;
use super::memory::MemoryResourceKind;
use super::validator::{is_raw_read, RuntimeValueValidator};
use super::value::{ValueId, ValueKind};
use super::ResourcePlanError;
use std::collections::{HashMap, HashSet};

pub(super) struct TableReadPlan {
    pub reads: Vec<ResourceTableRead>,
    pub dynamic_reads: Vec<ValueId>,
    pub replacements: HashMap<ValueId, ValueId>,
    pub flattened_slot_by_memory_index: HashMap<usize, u32>,
}

struct Planner<'g> {
    graph: &'g mut ScalarValueGraph,
    options: PlanOptions,
    stage: ShaderStage,
    hash: u64,
    reads: Vec<ResourceTableRead>,
    dynamic_reads: Vec<ValueId>,
    patches: Vec<(ValueId, u32)>,
    visiting: Vec<ValueId>,
    visited: HashSet<ValueId>,
}

/// With `options.flatten_data_reads` raw scalar reads of data at
/// host-evaluable addresses get slots too (else the program performs them);
/// with `options.buffers_on_device` the descriptors of buffer reads the
/// program addresses itself get none.
pub(super) fn plan(graph: &mut ScalarValueGraph, stage: ShaderStage, hash: u64, options: PlanOptions) -> Result<TableReadPlan, ResourcePlanError> {
    Planner { graph, options, stage, hash, reads: Vec::new(), dynamic_reads: Vec::new(), patches: Vec::new(), visiting: Vec::new(), visited: HashSet::new() }
        .run()
}

impl Planner<'_> {
    fn fail(&self, pc: u32, message: impl std::fmt::Display) -> ResourcePlanError {
        ResourcePlanError::new(format!(
            "shader resource-table planning failed: hash=0x{:016X} stage={:?} pc=0x{pc:08X} {message}",
            self.hash, self.stage
        ))
    }

    fn run(mut self) -> Result<TableReadPlan, ResourcePlanError> {
        for value in self.graph.values() {
            let node = self.graph.node(value);
            if matches!(node.kind, ValueKind::ScalarAddressWord | ValueKind::ScalarBufferWord) && node.memory_index() < self.graph.memory.len()
            {
                let entry = self.graph.memory.get(node.memory_index());
                let crosswired = (node.kind == ValueKind::ScalarAddressWord && entry.kind == MemoryResourceKind::ScalarBuffer)
                    || (node.kind == ValueKind::ScalarBufferWord && entry.kind == MemoryResourceKind::ScalarAddress);
                if crosswired {
                    return Err(self.fail(entry.pc, format!("{:?} has incompatible scalar memory metadata", node.kind)));
                }
            }
        }

        let accesses = self.graph.accesses.clone();
        for (index, access) in accesses.iter().enumerate() {
            let Some(access) = access else { continue };
            if index < self.graph.memory.len() && self.options.device_buffer_read(self.graph.memory.get(index)) {
                continue;
            }
            for handle in [access.handle, access.sampler_handle].into_iter().flatten() {
                for dword in self.graph.operands(handle).to_vec() {
                    self.collect(dword, 0)?;
                }
            }
        }

        let validator_read_count = self.reads.len();
        let flatten = self.options.flatten_data_reads;
        for access in accesses.iter().flatten().filter(|_| flatten) {
            let Some(read) = access.read else { continue };
            if self.graph.kind(read) == ValueKind::ScalarAddressWord
                && is_raw_read(self.graph, read)
                && self.graph.is_constant(self.graph.operand(read, 1))
                && RuntimeValueValidator::new(self.graph, validator_read_count).validate(read)
            {
                let pc = self.graph.memory.get(self.graph.node(read).memory_index()).pc;
                self.collect(read, pc)?;
            }
        }

        // Equivalent reads share one host read, but each instruction loads
        // its destination from that slot instead of reading guest memory.
        let mut replacements = HashMap::new();
        let mut flattened_slot_by_memory_index = HashMap::new();
        for &(read, slot) in &self.patches {
            let word = self.graph.resource_table_word(slot);
            replacements.insert(read, word);
            let memory_index = self.graph.node(read).memory_index();
            if memory_index < self.graph.memory.len() {
                self.graph.memory.get_mut(memory_index).planning_only = true;
                if flattened_slot_by_memory_index.insert(memory_index, slot).is_some() {
                    let pc = self.graph.memory.get(memory_index).pc;
                    return Err(self.fail(pc, "memory access is flattened twice"));
                }
            }
        }
        Ok(TableReadPlan { reads: self.reads, dynamic_reads: self.dynamic_reads, replacements, flattened_slot_by_memory_index })
    }

    fn collect(&mut self, value: ValueId, use_pc: u32) -> Result<(), ResourcePlanError> {
        if self.graph.is_constant(value) {
            return Ok(());
        }
        if let Some(cycle) = self.visiting.iter().position(|&v| v == value) {
            if self.visiting[cycle..].iter().any(|&v| self.graph.kind(v) == ValueKind::Phi) {
                return Ok(());
            }
            return Err(self.fail(use_pc, format!("cyclic planning value {:?} without a phi", self.graph.kind(value))));
        }
        if !self.visited.insert(value) {
            return Ok(());
        }
        self.visiting.push(value);
        for operand in self.graph.operands(value).to_vec() {
            self.collect(operand, use_pc)?;
        }
        self.visiting.pop();
        if !is_raw_read(self.graph, value) {
            return Ok(());
        }
        let offset = self.graph.operand(value, 1);
        if !self.graph.is_constant(offset) {
            if !self.dynamic_reads.contains(&value) {
                self.dynamic_reads.push(value);
            }
            return Ok(());
        }
        // A read whose address depends on values the host cannot know (a loop
        // over data the shader loaded itself) is performed on the device.
        if !RuntimeValueValidator::new(self.graph, 0).validate(value) {
            return Ok(());
        }
        for slot in 0..self.reads.len() {
            if self.graph.equivalent(value, self.reads[slot].value) {
                self.patches.push((value, slot as u32));
                return Ok(());
            }
        }
        let slot = self.reads.len() as u32;
        self.reads.push(ResourceTableRead { value, flat_offset: slot });
        self.patches.push((value, slot));
        Ok(())
    }
}
