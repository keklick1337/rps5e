// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Basic blocks and the control-flow graph over a decoded program.

use super::branch::BranchResolver;
use crate::ir_types::Instruction;
use std::collections::{BTreeSet, HashMap, HashSet};

/// Half-open pc range `[start_pc, end_pc)` of one basic block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockRange {
    pub start_pc: u32,
    pub end_pc: u32,
}

/// Basic blocks sorted by start pc; each ends where the next begins.
#[derive(Clone, Debug, Default)]
pub struct ControlFlowGraph {
    pub blocks: Vec<BlockRange>,
    pub block_by_start_pc: HashMap<u32, usize>,
    pub successors: Vec<Vec<usize>>,
    pub predecessors: Vec<Vec<usize>>,
    /// Blocks targeted by a back edge (successor index <= source index).
    pub loop_headers: HashSet<usize>,
}

fn block_of(blocks: &[BlockRange], pc: u32) -> Option<usize> {
    let (mut lo, mut hi) = (0isize, blocks.len() as isize - 1);
    while lo <= hi {
        let mid = lo + ((hi - lo) >> 1);
        let r = blocks[mid as usize];
        if pc < r.start_pc {
            hi = mid - 1;
        } else if pc >= r.end_pc {
            lo = mid + 1;
        } else {
            return Some(mid as usize);
        }
    }
    None
}

fn link(succ: &mut [Vec<usize>], pred: &mut [Vec<usize>], from: usize, to: usize) {
    if !succ[from].contains(&to) {
        succ[from].push(to);
    }
    if !pred[to].contains(&from) {
        pred[to].push(from);
    }
}

impl ControlFlowGraph {
    pub fn has_control_flow(&self) -> bool {
        self.blocks.len() > 1
    }

    /// Index of the block containing `pc`.
    pub fn block_of(&self, pc: u32) -> Option<usize> {
        block_of(&self.blocks, pc)
    }

    pub fn build(instructions: &[Instruction], resolver: &impl BranchResolver) -> Self {
        let mut leaders = BTreeSet::new();
        if let Some(first) = instructions.first() {
            leaders.insert(first.pc);
        }
        for (index, ins) in instructions.iter().enumerate() {
            if !resolver.is_branch(ins) {
                continue;
            }
            if let Some(target) = resolver.branch_target(ins) {
                leaders.insert(target);
            }
            if let Some(next) = instructions.get(index + 1) {
                leaders.insert(next.pc);
            }
        }

        let ordered: Vec<u32> = leaders.into_iter().collect();
        let mut blocks = Vec::with_capacity(ordered.len());
        let mut by_start = HashMap::new();
        for (index, &start) in ordered.iter().enumerate() {
            let end = match ordered.get(index + 1) {
                Some(&next) => next,
                None => instructions.last().map_or(start, |l| l.pc + 1),
            };
            by_start.insert(start, blocks.len());
            blocks.push(BlockRange { start_pc: start, end_pc: end });
        }

        let mut successors = vec![Vec::new(); blocks.len()];
        let mut predecessors = vec![Vec::new(); blocks.len()];
        let mut last_by_block: Vec<Option<&Instruction>> = vec![None; blocks.len()];
        for ins in instructions {
            if let Some(b) = block_of(&blocks, ins.pc) {
                last_by_block[b] = Some(ins);
            }
        }
        for (b, last) in last_by_block.iter().enumerate() {
            let Some(last) = last else { continue };
            let is_branch = resolver.is_branch(last);
            if is_branch {
                if let Some(&target) = resolver.branch_target(last).and_then(|t| by_start.get(&t)) {
                    link(&mut successors, &mut predecessors, b, target);
                }
            }
            let falls_through = !is_branch || resolver.is_conditional(last);
            if falls_through && b + 1 < blocks.len() {
                link(&mut successors, &mut predecessors, b, b + 1);
            }
        }

        let mut loop_headers = HashSet::new();
        for (b, succ) in successors.iter().enumerate() {
            loop_headers.extend(succ.iter().copied().filter(|&s| s <= b));
        }

        ControlFlowGraph { blocks, block_by_start_pc: by_start, successors, predecessors, loop_headers }
    }
}
