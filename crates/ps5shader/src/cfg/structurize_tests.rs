// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Structurizer tests: random reducible graphs are walked both directly and
//! through the region tree with the same branch decisions; the block traces
//! must match.

use super::structurize::*;
use super::graph::*;
use std::collections::HashMap;

/// Decision for the k-th evaluation of block b's condition.
fn decide(seed: u64, b: usize, k: u32) -> bool {
    let mut x = seed ^ (b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (k as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 29;
    x & 1 == 1
}

pub fn walk_cfg(cfg: &Cfg, seed: u64, limit: usize) -> Vec<usize> {
    let mut trace = Vec::new();
    let mut counts = HashMap::new();
    let mut at = cfg.entry;
    while trace.len() < limit {
        let o = cfg.blocks[at].origin;
        trace.push(o);
        match cfg.blocks[at].exit {
            Exit::Return => break,
            Exit::Branch(t) => at = t,
            Exit::Cond { taken, fallthrough, .. } => {
                let k = counts.entry(o).or_insert(0u32);
                let d = decide(seed, o, *k);
                *k += 1;
                at = if d { taken } else { fallthrough };
            }
        }
    }
    trace
}

enum Flow {
    Normal,
    Break,
    Continue,
    Return,
    Limit,
}

pub struct TreeWalk<'a> {
    pub cfg: &'a Cfg,
    pub seed: u64,
    pub trace: Vec<usize>,
    counts: HashMap<usize, u32>,
    flags: HashMap<FlagId, bool>,
    state: Option<u32>,
    pub limit: usize,
}

impl<'a> TreeWalk<'a> {
    pub fn new(cfg: &'a Cfg, seed: u64, limit: usize) -> Self {
        TreeWalk { cfg, seed, trace: Vec::new(), counts: HashMap::new(), flags: HashMap::new(), state: None, limit }
    }

    fn eval(&mut self, c: &Cond) -> bool {
        match c {
            Cond::Branch { block, .. } => {
                let o = self.cfg.blocks[*block].origin;
                let k = self.counts.entry(o).or_insert(0);
                let d = decide(self.seed, o, *k);
                *k += 1;
                d
            }
            Cond::Flag(f) => *self.flags.get(f).expect("goto variable read before write"),
            Cond::State(b) => self.state.expect("dispatcher state read before write") == *b,
            Cond::Not(c) => !self.eval(c),
            Cond::Or(cs) => {
                let mut r = false;
                for c in cs {
                    r |= self.eval(c);
                }
                r
            }
        }
    }

    pub fn run_tree(&mut self, n: &Node) {
        let _ = self.run(n);
    }

    fn run(&mut self, n: &Node) -> Flow {
        if self.trace.len() >= self.limit {
            return Flow::Limit;
        }
        match n {
            Node::Code(b) => {
                self.trace.push(self.cfg.blocks[*b].origin);
                Flow::Normal
            }
            Node::Seq(v) => {
                for item in v {
                    match self.run(item) {
                        Flow::Normal => {}
                        other => return other,
                    }
                }
                Flow::Normal
            }
            Node::If { cond, then_branch, else_branch } => {
                if self.eval(cond) {
                    self.run(then_branch)
                } else {
                    self.run(else_branch)
                }
            }
            Node::Loop(body) => loop {
                match self.run(body) {
                    Flow::Continue => continue,
                    Flow::Normal | Flow::Break => return Flow::Normal,
                    other => return other,
                }
            },
            Node::Break => Flow::Break,
            Node::Continue => Flow::Continue,
            Node::Return => Flow::Return,
            Node::SetFlag(f, v) => {
                self.flags.insert(*f, *v);
                Flow::Normal
            }
            Node::SetState(b) => {
                self.state = Some(*b);
                Flow::Normal
            }
        }
    }
}

fn rng(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn random_cfg(state: &mut u64, n: usize) -> Cfg {
    let mut blocks = Vec::new();
    for i in 0..n {
        let r = rng(state) % 10;
        let exit = if i == n - 1 || r == 0 {
            Exit::Return
        } else if r < 4 {
            Exit::Branch((rng(state) % n as u64) as usize)
        } else {
            let taken = (rng(state) % n as u64) as usize;
            let fallthrough = if r < 7 { i + 1 } else { (rng(state) % n as u64) as usize };
            Exit::Cond { cond: BranchCond::Scc1, taken, fallthrough }
        };
        blocks.push(BasicBlock { start_pc: i as u32 * 4, first: i, end: i + 1, exit, words: 8, origin: i });
    }
    let mut cfg = Cfg { blocks, entry: 0 };
    cfg.prune_unreachable();
    for (i, b) in cfg.blocks.iter_mut().enumerate() {
        b.origin = i;
    }
    cfg
}

#[test]
fn random_graphs_match() {
    let mut state = 0x1234_5678_9ABC_DEF1u64;
    let mut tested = 0;
    let mut gotos = 0;
    let mut clones = 0;
    let mut dispatched = 0;
    for round in 0..8000 {
        let n = 2 + (round % 16);
        let cfg = random_cfg(&mut state, n);
        let s = structurize(&cfg).unwrap();
        assert_eq!(s.stats.dispatcher, cfg.check_reducible().is_err());
        dispatched += s.stats.dispatcher as u32;
        gotos += s.stats.goto_variables;
        clones += s.stats.cloned_blocks;
        for seed in 0..12u64 {
            let expected = walk_cfg(&cfg, seed, 300);
            let mut tw = TreeWalk::new(&s.cfg, seed, 300);
            tw.run_tree(&s.tree);
            assert_eq!(tw.trace, expected, "round {round} seed {seed}\ncfg:\n{cfg}\ntree:\n{}", s.tree);
        }
        tested += 1;
    }
    assert!(tested - dispatched > 1000, "only {} reducible graphs", tested - dispatched);
    assert!(dispatched > 100, "only {dispatched} irreducible graphs");
    eprintln!("tested {tested} graphs ({dispatched} dispatched), {gotos} goto variables, {clones} cloned blocks");
}

/// Early discards that branch to one small shared ending block (several
/// conditional predecessors outside any loop): the reference structurizer
/// failed on this pixel-shader graph with "conditional block has no
/// structured merge".
#[test]
fn early_exits_sharing_an_ending_block() {
    let successors: [&[usize]; 8] = [&[1, 5], &[2, 4], &[3, 7], &[4, 7], &[5], &[6, 7], &[], &[]];
    let words = [444u32, 1944, 68, 1128, 8, 2084, 148, 28];
    let blocks = successors
        .iter()
        .enumerate()
        .map(|(b, s)| {
            let exit = match s {
                [] => Exit::Return,
                [t] => Exit::Branch(*t),
                [t, f] => Exit::Cond { cond: BranchCond::Scc1, taken: *t, fallthrough: *f },
                _ => unreachable!(),
            };
            BasicBlock { start_pc: b as u32 * 4, first: b, end: b + 1, exit, words: words[b], origin: b }
        })
        .collect();
    let cfg = Cfg { blocks, entry: 0 };
    let s = structurize(&cfg).unwrap();
    for seed in 0..64u64 {
        let expected = walk_cfg(&cfg, seed, 300);
        let mut tw = TreeWalk::new(&s.cfg, seed, 300);
        tw.run_tree(&s.tree);
        assert_eq!(tw.trace, expected, "seed {seed}\ntree:\n{}", s.tree);
    }
}

fn cfg_of(successors: &[&[usize]]) -> Cfg {
    let blocks = successors
        .iter()
        .enumerate()
        .map(|(b, s)| {
            let exit = match s {
                [] => Exit::Return,
                [t] => Exit::Branch(*t),
                [t, f] => Exit::Cond { cond: BranchCond::Scc1, taken: *t, fallthrough: *f },
                _ => unreachable!(),
            };
            BasicBlock { start_pc: b as u32 * 8, first: b, end: b + 1, exit, words: 2, origin: b }
        })
        .collect();
    Cfg { blocks, entry: 0 }
}

fn assert_walks_match(cfg: &Cfg, seeds: u64) {
    let s = structurize(cfg).unwrap();
    for seed in 0..seeds {
        let expected = walk_cfg(cfg, seed, 300);
        let mut tw = TreeWalk::new(&s.cfg, seed, 300);
        tw.run_tree(&s.tree);
        assert_eq!(tw.trace, expected, "seed {seed}\ntree:\n{}", s.tree);
    }
}

/// A loop (1 -> 3 -> 4 -> 1) whose merge (5) is also reached from inside
/// a nested selection (1 -> 2 -> 5, 3 -> 5): the loop merge must not be
/// placed inside that selection (a structural dominance regression of
/// the reference structurizer).
#[test]
fn loop_merge_outside_a_nested_selection() {
    assert_walks_match(&cfg_of(&[&[1, 5], &[2, 3], &[5], &[5, 4], &[1], &[]]), 64);
}

/// A loop whose selection arms leave the iteration (to the latch) before
/// the join an enclosing selection shares (the shape of the reference's
/// control-flow case; the blocks of its program).
#[test]
fn loop_selection_arms_leaving_the_iteration() {
    // 0 entry, 1 loop (cmp s8 == s2), 2 vcc, 3 +1/vcc, 4 arm, 5 join,
    // 6 branch latch, 7 away, 8 aside, 9 latch, 10 done.
    let successors: [&[usize]; 11] = [&[1], &[5, 2], &[4, 3], &[7, 4], &[8, 5], &[10, 6], &[9], &[9], &[9], &[1], &[]];
    assert_walks_match(&cfg_of(&successors), 64);
}

#[test]
fn irreducible_dispatched() {
    let blocks = vec![
        BasicBlock { start_pc: 0, first: 0, end: 1, exit: Exit::Cond { cond: BranchCond::Scc1, taken: 1, fallthrough: 2 }, words: 1, origin: 0 },
        BasicBlock { start_pc: 4, first: 1, end: 2, exit: Exit::Cond { cond: BranchCond::Scc1, taken: 2, fallthrough: 3 }, words: 1, origin: 1 },
        BasicBlock { start_pc: 8, first: 2, end: 3, exit: Exit::Branch(1), words: 1, origin: 2 },
        BasicBlock { start_pc: 12, first: 3, end: 4, exit: Exit::Return, words: 1, origin: 3 },
    ];
    let cfg = Cfg { blocks, entry: 0 };
    assert!(matches!(cfg.check_reducible(), Err(CfgError::Irreducible { .. })));
    let s = structurize(&cfg).unwrap();
    assert!(s.stats.dispatcher);
    for seed in 0..32u64 {
        let expected = walk_cfg(&cfg, seed, 300);
        let mut tw = TreeWalk::new(&s.cfg, seed, 300);
        tw.run_tree(&s.tree);
        assert_eq!(tw.trace, expected, "seed {seed}\ntree:\n{}", s.tree);
    }
}

#[test]
fn simple_loop_has_no_goto() {
    // 0 -> 1; 1 -> 1 | 2; 2 return
    let blocks = vec![
        BasicBlock { start_pc: 0, first: 0, end: 1, exit: Exit::Branch(1), words: 1, origin: 0 },
        BasicBlock { start_pc: 4, first: 1, end: 2, exit: Exit::Cond { cond: BranchCond::Execnz, taken: 1, fallthrough: 2 }, words: 1, origin: 1 },
        BasicBlock { start_pc: 8, first: 2, end: 3, exit: Exit::Return, words: 1, origin: 2 },
    ];
    let cfg = Cfg { blocks, entry: 0 };
    let s = structurize(&cfg).unwrap();
    assert_eq!(s.stats.goto_variables, 0, "{}", s.tree);
    assert_eq!(s.stats.loops, 1);
}

/// Graphs the reference structurizer refused or structured invalidly
/// (2026-10 fixes: loop exits that leave the loop before they meet, a loop
/// selection whose arms meet within the iteration while one arm continues,
/// a conditional branch whose two targets are one block, a single-block
/// loop): every one structures and runs the original's paths.
#[test]
fn reference_structurizer_regression_graphs() {
    let graphs: [&[&[usize]]; 9] = [
        &[&[1], &[2], &[3, 4], &[2, 5], &[], &[1]],
        &[&[1], &[2, 3], &[4, 5], &[], &[3, 6], &[1], &[2]],
        &[&[1, 3], &[2, 4], &[3, 1], &[4], &[]],
        &[&[1, 2], &[3, 2], &[4], &[4, 1], &[]],
        &[&[1, 2], &[4], &[3, 1], &[4, 2], &[]],
        &[&[1], &[2, 4], &[3, 4], &[4, 5], &[1, 6], &[1], &[]],
        &[&[1, 1], &[2, 2], &[]],
        &[&[1], &[1, 2], &[]],
        &[&[1], &[2, 4], &[2, 3], &[1, 4], &[]],
    ];
    for (i, g) in graphs.iter().enumerate() {
        let cfg = cfg_of(g);
        let s = structurize(&cfg).unwrap_or_else(|e| panic!("graph {i}: {e:?}"));
        for seed in 0..64u64 {
            let expected = walk_cfg(&cfg, seed, 300);
            let mut tw = TreeWalk::new(&s.cfg, seed, 300);
            tw.run_tree(&s.tree);
            assert_eq!(tw.trace, expected, "graph {i} seed {seed}\ntree:\n{}", s.tree);
        }
    }
}

/// An arm whose return path branches again (an early return inside a nested
/// if/else, the tail shared across arms): it structures as selections, with
/// no goto variable, and runs the original's paths (reference regression
/// graph of a2f497a05).
#[test]
fn arm_returning_through_nested_branches() {
    let cfg = cfg_of(&[&[5, 1], &[3, 2], &[13], &[9, 4], &[13], &[10, 6], &[8, 7], &[13], &[13, 9], &[13], &[12, 11], &[12], &[13], &[]]);
    assert_walks_match(&cfg, 64);
    let s = structurize(&cfg).unwrap();
    assert_eq!(s.stats.goto_variables, 0, "{}", s.tree);
}
