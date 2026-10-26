// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Resource branches: which descriptor sources each basic block uses, and
//! the host-evaluable condition of each conditional branch, so a draw only
//! materialises the descriptors on paths its uniform predicates take.

use super::graph::ScalarValueGraph;
use super::memory::{MemoryAccess, MemoryResourceKind};
use super::plan::ShaderResourcePlan;
use super::value::ValueId;
use crate::ir::branch;
use crate::ir_types::Instruction;
use crate::opcode::Opcode;
use std::collections::{BTreeSet, HashMap, HashSet};

/// One basic block: its branch condition (taken edge first in
/// `successors`), and the descriptor sources its accesses use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceBranchBlock {
    /// Host-evaluable condition of the block's conditional branch.
    pub condition: Option<ValueId>,
    /// `[taken, fall-through]` for a conditional branch, else the CFG
    /// successors.
    pub successors: Vec<usize>,
    pub sources: Vec<u32>,
}

pub(super) fn build(
    plan: &mut ShaderResourcePlan,
    replacements: &HashMap<ValueId, ValueId>,
    memo: &mut HashMap<ValueId, ValueId>,
) -> Vec<ResourceBranchBlock> {
    // A shader write can change a predicate through an alias or a later
    // loop iteration.
    if plan.graph.memory.entries().iter().any(|a| matches!(a.access, MemoryAccess::Write | MemoryAccess::Atomic)) {
        return Vec::new();
    }
    if plan.graph.program.instructions.iter().any(|i| matches!(i.opcode, Opcode::SSetpcB64 | Opcode::SSwappcB64)) {
        return Vec::new();
    }
    let block_count = plan.graph.cfg.blocks.len();
    let mut last_by_block: Vec<Option<Instruction>> = vec![None; block_count];
    let mut instruction_pcs = HashSet::new();
    for instruction in &plan.graph.program.instructions {
        instruction_pcs.insert(instruction.pc);
        if let Some(b) = plan.graph.cfg.block_of(instruction.pc) {
            last_by_block[b] = Some(instruction.clone());
        }
    }
    let mut blocks = Vec::with_capacity(block_count);
    let mut has_condition = false;
    for block_index in 0..block_count {
        let range = plan.graph.cfg.blocks[block_index];
        let Some(last) = &last_by_block[block_index] else { return Vec::new() };
        let mut successors = plan.graph.cfg.successors[block_index].clone();
        let mut condition = None;
        if branch::is_conditional(last) || branch::is_unconditional_branch(last) {
            let Some(target) = branch::branch_target(last).filter(|t| instruction_pcs.contains(t)) else { return Vec::new() };
            if branch::is_conditional(last) {
                if block_index + 1 >= block_count {
                    return Vec::new();
                }
                // Keep taken and fall-through separate even when both name
                // the same block.
                successors = vec![plan.graph.cfg.block_by_start_pc[&target], block_index + 1];
                if let Some(&original) = plan.graph.branch_conditions.get(&last.pc) {
                    let candidate = rewrite(&mut plan.graph, original, replacements, memo);
                    if plan.validate_runtime_value(candidate) {
                        condition = Some(candidate);
                    }
                }
            }
        }
        let mut sources = BTreeSet::new();
        for access in plan.graph.memory.entries() {
            if access.pc < range.start_pc || access.pc >= range.end_pc || access.planning_only {
                continue;
            }
            let resource = access.resource as usize;
            if matches!(access.kind, MemoryResourceKind::Buffer | MemoryResourceKind::ScalarBuffer) && resource < plan.info.buffers.len() {
                sources.insert(plan.info.buffers[resource].source);
            }
            if access.kind == MemoryResourceKind::Image && resource < plan.info.images.len() {
                sources.insert(plan.info.images[resource].source);
                if access.needs_sampler && (access.sampler as usize) < plan.info.samplers.len() {
                    sources.insert(plan.info.samplers[access.sampler as usize].source);
                }
            }
        }
        has_condition |= condition.is_some();
        blocks.push(ResourceBranchBlock { condition, successors, sources: sources.into_iter().collect() });
    }
    if has_condition {
        blocks
    } else {
        Vec::new()
    }
}

fn rewrite(graph: &mut ScalarValueGraph, value: ValueId, replacements: &HashMap<ValueId, ValueId>, memo: &mut HashMap<ValueId, ValueId>) -> ValueId {
    graph.substitute(value, replacements, memo)
}
