// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Structurizer: turns a reducible [`Cfg`] into a region tree whose
//! constructs map one-to-one onto SPIR-V structured control flow.
//!
//! 1. Analyses: reverse post-order, dominators, back edges, loop headers,
//!    merge nodes (two or more forward in-edges); irreducible graphs take
//!    the dispatcher form below instead.
//! 2. Translation over the dominator tree into nested labelled blocks,
//!    loops and multi-level branches: every merge node is placed after the
//!    code of its immediate dominator inside a labelled block, every other
//!    node is inlined at its single forward predecessor's branch, and loop
//!    headers open a loop (Ramsey, "Beyond Relooper", 2022).
//! 3. Lowering to SPIR-V shapes: a branch to the end of a labelled block in
//!    the same loop skips the rest of the sequences in between, which is
//!    expressed by moving the remaining code into the other arm of the `if`
//!    when that arm falls through (no variable), else by a goto variable set
//!    at the branch and tested by a guard selection. A branch out of a loop
//!    (to a block further out, or to an outer loop's continue) sets a goto
//!    variable, breaks, and is re-dispatched after the loop ("junction").
//!    Loops get a dedicated empty header and a continue block when emitted
//!    (canonical form), so a back edge from anywhere in the body is a
//!    `Continue`.
//! 4. Clean-up: goto variables never tested are dropped with their
//!    assignments; empty selections disappear.
//!
//! Merge tails that only lead to the program end are cloned per predecessor
//! (under a SPIR-V word budget) when that removes goto variables.
//!
//! An irreducible graph is not restructured: it becomes a dispatcher, one
//! loop around a selection per block keyed by a state variable that holds
//! the next block to run. Each block sets the state to its successor and
//! continues the loop; a block ending the program returns.

use super::dom::{self, NONE};
use super::graph::{BasicBlock, BranchCond, Cfg, CfgError, Exit};
use std::collections::{BTreeSet, HashMap};
use std::fmt;

/// Goto variable index.
pub type FlagId = u32;

/// A selection condition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cond {
    /// The branch condition of CFG block `block` (evaluated right after its
    /// code).
    Branch { block: usize, cond: BranchCond },
    Flag(FlagId),
    /// The dispatcher state names CFG block `n`.
    State(u32),
    Not(Box<Cond>),
    Or(Vec<Cond>),
}

impl Cond {
    fn negate(self) -> Cond {
        match self {
            Cond::Not(c) => *c,
            c => Cond::Not(Box::new(c)),
        }
    }

    fn flags(&self, out: &mut BTreeSet<FlagId>) {
        match self {
            Cond::Branch { .. } | Cond::State(_) => {}
            Cond::Flag(f) => {
                out.insert(*f);
            }
            Cond::Not(c) => c.flags(out),
            Cond::Or(cs) => cs.iter().for_each(|c| c.flags(out)),
        }
    }
}

/// Region tree node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    /// The (non-branch) instructions of a CFG block.
    Code(usize),
    Seq(Vec<Node>),
    /// Selection; the merge follows it.
    If { cond: Cond, then_branch: Box<Node>, else_branch: Box<Node> },
    /// Loop: the body repeats on `Continue`; falling off the body's end or
    /// `Break` leaves it.
    Loop(Box<Node>),
    /// Leave the innermost loop.
    Break,
    /// Next iteration of the innermost loop.
    Continue,
    /// End of program.
    Return,
    /// Assign a goto variable.
    SetFlag(FlagId, bool),
    /// Assign the dispatcher state (the next CFG block to run).
    SetState(u32),
}

impl Node {
    fn empty() -> Node {
        Node::Seq(Vec::new())
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, Node::Seq(v) if v.iter().all(Node::is_empty))
    }

    /// Visit every node.
    pub fn walk(&self, f: &mut impl FnMut(&Node)) {
        f(self);
        match self {
            Node::Seq(v) => v.iter().for_each(|n| n.walk(f)),
            Node::If { then_branch, else_branch, .. } => {
                then_branch.walk(f);
                else_branch.walk(f);
            }
            Node::Loop(b) => b.walk(f),
            _ => {}
        }
    }
}

/// Structurizer result.
#[derive(Clone, Debug)]
pub struct Structured {
    /// The graph the tree refers to (with cloned tails appended).
    pub cfg: Cfg,
    pub tree: Node,
    /// Goto variables `0..flag_count` (some may be unused).
    pub flag_count: u32,
    pub stats: StructureStats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StructureStats {
    pub loops: u32,
    pub selections: u32,
    /// Goto variables remaining after clean-up.
    pub goto_variables: u32,
    pub cloned_blocks: u32,
    pub cloned_words: u32,
    /// The graph is irreducible and runs through a dispatcher loop.
    pub dispatcher: bool,
}

/// Per-clone and total clone budgets in estimated SPIR-V words.
pub const CLONE_WORD_LIMIT: u32 = 1024;
pub const CLONE_WORD_FLOOR: u32 = 256;

/// Structurize `cfg`: a reducible graph becomes nested structured
/// constructs, an irreducible one a dispatcher loop.
pub fn structurize(cfg: &Cfg) -> Result<Structured, CfgError> {
    if cfg.check_reducible().is_err() {
        return Ok(dispatcher(cfg));
    }
    // Deep graphs recurse deeply; run on a thread with a large stack.
    std::thread::scope(|s| {
        std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn_scoped(s, || structurize_impl(cfg))
            .expect("spawn structurizer thread")
            .join()
            .expect("structurizer panicked")
    })
}

fn structurize_impl(cfg: &Cfg) -> Result<Structured, CfgError> {
    let plain = run(cfg.clone());
    if plain.stats.goto_variables == 0 {
        return Ok(plain);
    }
    let (cloned, blocks, words) = clone_terminal_tails(cfg);
    if blocks == 0 {
        return Ok(plain);
    }
    let mut with_clones = run(cloned);
    with_clones.stats.cloned_blocks = blocks;
    with_clones.stats.cloned_words = words;
    if with_clones.stats.goto_variables < plain.stats.goto_variables {
        Ok(with_clones)
    } else {
        Ok(plain)
    }
}

fn run(cfg: Cfg) -> Structured {
    let analysis = Analysis::new(&cfg);
    let w = analysis.do_tree(cfg.entry);
    let mut lowerer = Lowerer { frames: Vec::new(), flags: HashMap::new(), next_flag: 0 };
    let (tree, _) = lowerer.lower(&w);
    let flag_count = lowerer.next_flag;
    let tree = cleanup(tree);
    let mut stats = StructureStats::default();
    let mut used = BTreeSet::new();
    tree.walk(&mut |n| match n {
        Node::Loop(_) => stats.loops += 1,
        Node::If { cond, .. } => {
            stats.selections += 1;
            cond.flags(&mut used);
        }
        _ => {}
    });
    stats.goto_variables = used.len() as u32;
    Structured { cfg, tree, flag_count, stats }
}

/// Dispatcher form of an irreducible graph:
///
/// ```text
/// state = entry
/// loop {
///     if state == b { code(b); state = successor(b); continue }   // per block
///     ...
///     break
/// }
/// ```
///
/// A block ending the program returns from inside the loop.
pub fn dispatcher(cfg: &Cfg) -> Structured {
    let goto = |t: usize| Node::Seq(vec![Node::SetState(t as u32), Node::Continue]);
    let mut cases = Vec::with_capacity(cfg.blocks.len() + 1);
    for (b, block) in cfg.blocks.iter().enumerate() {
        let next = match block.exit {
            Exit::Return => Node::Return,
            Exit::Branch(t) => goto(t),
            Exit::Cond { cond, taken, fallthrough } => Node::If {
                cond: Cond::Branch { block: b, cond },
                then_branch: Box::new(goto(taken)),
                else_branch: Box::new(goto(fallthrough)),
            },
        };
        cases.push(Node::If {
            cond: Cond::State(b as u32),
            then_branch: Box::new(Node::Seq(vec![Node::Code(b), next])),
            else_branch: Box::new(Node::empty()),
        });
    }
    // No block matched: the state is always a block, so this is not reached.
    cases.push(Node::Break);
    let tree = Node::Seq(vec![Node::SetState(cfg.entry as u32), Node::Loop(Box::new(Node::Seq(cases)))]);
    let stats = StructureStats { loops: 1, selections: cfg.blocks.len() as u32, dispatcher: true, ..Default::default() };
    Structured { cfg: cfg.clone(), tree, flag_count: 0, stats }
}

/// Clone linear tails ending the program (no successors at the end) that
/// have several predecessors, within the word budget.
fn clone_terminal_tails(cfg: &Cfg) -> (Cfg, u32, u32) {
    let mut cfg = cfg.clone();
    let total: u32 = cfg.blocks.iter().map(|b| b.words).sum();
    let per_clone = CLONE_WORD_LIMIT.min(CLONE_WORD_FLOOR.max(total / 8));
    let mut budget = CLONE_WORD_FLOOR.max(total / 4);
    let (mut cloned_blocks, mut cloned_words) = (0, 0);
    let original = cfg.blocks.len();
    for m in 0..original {
        let preds = dom::predecessors(&cfg.successors());
        if preds[m].len() < 2 || m == cfg.entry {
            continue;
        }
        // Linear chain m -> ... -> return, each later node with one pred.
        let mut chain = vec![m];
        let mut ok = true;
        let mut at = m;
        loop {
            match cfg.blocks[at].exit {
                Exit::Return => break,
                Exit::Branch(t) if preds[t].len() == 1 && !chain.contains(&t) => {
                    chain.push(t);
                    at = t;
                }
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let words: u32 = chain.iter().map(|&b| cfg.blocks[b].words.max(1)).sum();
        if words > per_clone {
            continue;
        }
        for &p in preds[m].iter().skip(1) {
            if words > budget {
                break;
            }
            budget -= words;
            let base = cfg.blocks.len();
            for (k, &b) in chain.iter().enumerate() {
                let mut copy: BasicBlock = cfg.blocks[b].clone();
                copy.origin = cfg.blocks[b].origin;
                if let Exit::Branch(_) = copy.exit {
                    copy.exit = Exit::Branch(base + k + 1);
                }
                cfg.blocks.push(copy);
            }
            cfg.blocks[p].exit.retarget(m, base);
            cloned_blocks += chain.len() as u32;
            cloned_words += words;
        }
    }
    (cfg, cloned_blocks, cloned_words)
}

// ---------------------------------------------------------------------------
// Step 2: labelled-block tree
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Target {
    /// End of the labelled block followed by node `y`.
    Exit(usize),
    /// Restart of the loop headed by `h`.
    Continue(usize),
}

#[derive(Clone, Debug)]
enum W {
    Code(usize),
    Seq(Vec<W>),
    If(Cond, Box<W>, Box<W>),
    Block(usize, Box<W>),
    Loop(usize, Box<W>),
    Br(Target),
    Return,
}

struct Analysis<'a> {
    cfg: &'a Cfg,
    rpo_index: Vec<usize>,
    children: Vec<Vec<usize>>,
    merge: Vec<bool>,
    loop_header: Vec<bool>,
    exit_child: Vec<bool>,
}

impl<'a> Analysis<'a> {
    fn new(cfg: &'a Cfg) -> Self {
        let succs = cfg.successors();
        let rpo = dom::reverse_post_order(&succs, cfg.entry);
        let mut rpo_index = vec![NONE; succs.len()];
        for (i, &b) in rpo.iter().enumerate() {
            rpo_index[b] = i;
        }
        let idom = dom::dominators(&succs, cfg.entry);
        let children = dom::dominator_children(&idom);
        let mut forward_in = vec![0u32; succs.len()];
        let mut loop_header = vec![false; succs.len()];
        for &b in &rpo {
            for &s in &succs[b] {
                if rpo_index[s] > rpo_index[b] {
                    forward_in[s] += 1;
                } else {
                    loop_header[s] = true;
                }
            }
        }
        let mut merge: Vec<bool> = forward_in.iter().map(|&n| n >= 2).collect();
        // Natural loop bodies: a header's dominator-tree children outside its
        // body are loop exits; they are placed after the loop, like merge
        // nodes, instead of inside it.
        let preds = dom::predecessors(&succs);
        let mut loop_body: HashMap<usize, Vec<bool>> = HashMap::new();
        for &b in &rpo {
            for &h in &succs[b] {
                if rpo_index[h] > rpo_index[b] {
                    continue;
                }
                let body = loop_body.entry(h).or_insert_with(|| {
                    let mut v = vec![false; succs.len()];
                    v[h] = true;
                    v
                });
                let mut stack = vec![b];
                while let Some(n) = stack.pop() {
                    if body[n] {
                        continue;
                    }
                    body[n] = true;
                    stack.extend(preds[n].iter().copied().filter(|&p| rpo_index[p] != NONE));
                }
            }
        }
        let mut exit_child = vec![false; succs.len()];
        for (&h, body) in &loop_body {
            for &c in &children[h] {
                if !body[c] {
                    exit_child[c] = true;
                    merge[c] = true;
                }
            }
        }
        Analysis { cfg, rpo_index, children, merge, loop_header, exit_child }
    }

    fn do_tree(&self, x: usize) -> W {
        let by_rpo_desc = |mut v: Vec<usize>| {
            v.sort_by_key(|&y| std::cmp::Reverse(self.rpo_index[y]));
            v
        };
        if self.loop_header[x] {
            let inside = by_rpo_desc(self.children[x].iter().copied().filter(|&c| self.merge[c] && !self.exit_child[c]).collect());
            let outside = by_rpo_desc(self.children[x].iter().copied().filter(|&c| self.exit_child[c]).collect());
            let looped = W::Loop(x, Box::new(self.node_within(x, &inside)));
            self.wrap_followers(looped, &outside)
        } else {
            let ys = by_rpo_desc(self.children[x].iter().copied().filter(|&c| self.merge[c]).collect());
            self.node_within(x, &ys)
        }
    }

    /// `inner` wrapped in labelled blocks followed by `ys` (highest RPO
    /// outermost).
    fn wrap_followers(&self, inner: W, ys: &[usize]) -> W {
        match ys.split_first() {
            Some((&y, rest)) => W::Seq(vec![W::Block(y, Box::new(self.wrap_followers(inner, rest))), self.do_tree(y)]),
            None => inner,
        }
    }

    fn node_within(&self, x: usize, ys: &[usize]) -> W {
        match ys.split_first() {
            Some((&y, rest)) => W::Seq(vec![W::Block(y, Box::new(self.node_within(x, rest))), self.do_tree(y)]),
            None => {
                let tail = match self.cfg.blocks[x].exit {
                    Exit::Return => W::Return,
                    Exit::Branch(t) => self.do_branch(x, t),
                    Exit::Cond { cond, taken, fallthrough } => W::If(
                        Cond::Branch { block: x, cond },
                        Box::new(self.do_branch(x, taken)),
                        Box::new(self.do_branch(x, fallthrough)),
                    ),
                };
                W::Seq(vec![W::Code(x), tail])
            }
        }
    }

    fn do_branch(&self, from: usize, to: usize) -> W {
        if self.rpo_index[to] <= self.rpo_index[from] {
            W::Br(Target::Continue(to))
        } else if self.merge[to] {
            W::Br(Target::Exit(to))
        } else {
            self.do_tree(to)
        }
    }
}

// ---------------------------------------------------------------------------
// Step 3: lowering
// ---------------------------------------------------------------------------

/// Control summary of a lowered statement.
#[derive(Clone, Debug, Default)]
struct Sum {
    /// May complete normally.
    falls_through: bool,
    /// Labelled blocks (in the same loop) it may exit to by a pending
    /// goto variable.
    exits: BTreeSet<usize>,
}

impl Sum {
    fn normal() -> Sum {
        Sum { falls_through: true, exits: BTreeSet::new() }
    }

    fn terminal() -> Sum {
        Sum { falls_through: false, exits: BTreeSet::new() }
    }

    fn join(&self, other: &Sum) -> Sum {
        Sum { falls_through: self.falls_through || other.falls_through, exits: self.exits.union(&other.exits).copied().collect() }
    }
}

enum Frame {
    Block(usize),
    Loop { header: usize, escapes: BTreeSet<Target> },
}

struct Lowerer {
    frames: Vec<Frame>,
    flags: HashMap<Target, FlagId>,
    next_flag: FlagId,
}

impl Lowerer {
    fn flag(&mut self, t: Target) -> FlagId {
        if let Some(&f) = self.flags.get(&t) {
            return f;
        }
        let f = self.next_flag;
        self.next_flag += 1;
        self.flags.insert(t, f);
        f
    }

    fn lower(&mut self, w: &W) -> (Node, Sum) {
        match w {
            W::Code(b) => (Node::Code(*b), Sum::normal()),
            W::Return => (Node::Return, Sum::terminal()),
            W::Seq(items) => self.lower_seq(items),
            W::If(c, a, b) => {
                let (na, sa) = self.lower(a);
                let (nb, sb) = self.lower(b);
                let sum = sa.join(&sb);
                (Node::If { cond: c.clone(), then_branch: Box::new(na), else_branch: Box::new(nb) }, sum)
            }
            W::Block(y, body) => {
                self.frames.push(Frame::Block(*y));
                let (n, mut sum) = self.lower(body);
                self.frames.pop();
                if sum.exits.remove(y) {
                    sum.falls_through = true;
                }
                let node = match self.flags.get(&Target::Exit(*y)) {
                    Some(&f) => Node::Seq(vec![Node::SetFlag(f, false), n]),
                    None => n,
                };
                (node, sum)
            }
            W::Loop(h, body) => {
                self.frames.push(Frame::Loop { header: *h, escapes: BTreeSet::new() });
                let (n, sum) = self.lower(body);
                let Some(Frame::Loop { escapes, .. }) = self.frames.pop() else { unreachable!() };
                debug_assert!(sum.exits.is_empty(), "loop body leaks block exits");
                let body = match self.flags.get(&Target::Continue(*h)) {
                    Some(&f) => Node::Seq(vec![Node::SetFlag(f, false), n]),
                    None => n,
                };
                let mut out = vec![Node::Loop(Box::new(body))];
                // Dispatch after the loop.
                let targets: Vec<Target> = escapes.into_iter().collect();
                let mut arms: Vec<(Target, Node, Sum)> = Vec::new();
                for &t in &targets {
                    let (n, s) = self.lower_branch(t);
                    arms.push((t, n, s));
                }
                let mut sum = if sum.falls_through { Sum::normal() } else { Sum::terminal() };
                if arms.is_empty() {
                    return (Node::Seq(out), sum);
                }
                // Exhaustive chain unless the body can also fall off.
                let mut chain: Option<(Node, Sum)> = if sum.falls_through { Some((Node::empty(), Sum::normal())) } else { None };
                for (t, n, s) in arms.into_iter().rev() {
                    chain = Some(match chain {
                        None => (n, s),
                        Some((rest, rs)) => {
                            let f = self.flag(t);
                            let joined = s.join(&rs);
                            (Node::If { cond: Cond::Flag(f), then_branch: Box::new(n), else_branch: Box::new(rest) }, joined)
                        }
                    });
                }
                let (dispatch, ds) = chain.unwrap();
                out.push(dispatch);
                sum = ds;
                (Node::Seq(out), sum)
            }
            W::Br(t) => self.lower_branch(*t),
        }
    }

    fn lower_branch(&mut self, t: Target) -> (Node, Sum) {
        // Innermost loop between here and the target frame, if any.
        let mut crossed_loop: Option<usize> = None;
        for i in (0..self.frames.len()).rev() {
            match (&self.frames[i], t) {
                (Frame::Block(y), Target::Exit(z)) if *y == z => {
                    return match crossed_loop {
                        None => {
                            let f = self.flag(t);
                            (Node::SetFlag(f, true), Sum { falls_through: false, exits: [z].into_iter().collect() })
                        }
                        Some(l) => self.escape(l, t),
                    };
                }
                (Frame::Loop { header, .. }, Target::Continue(h)) if *header == h => {
                    return match crossed_loop {
                        None => (Node::Continue, Sum::terminal()),
                        Some(l) => self.escape(l, t),
                    };
                }
                (Frame::Loop { .. }, _) => {
                    if crossed_loop.is_none() {
                        crossed_loop = Some(i);
                    }
                }
                _ => {}
            }
        }
        panic!("branch target {t:?} has no enclosing frame");
    }

    fn escape(&mut self, loop_frame: usize, t: Target) -> (Node, Sum) {
        if let Frame::Loop { escapes, .. } = &mut self.frames[loop_frame] {
            escapes.insert(t);
        }
        let f = self.flag(t);
        (Node::Seq(vec![Node::SetFlag(f, true), Node::Break]), Sum::terminal())
    }

    fn lower_seq(&mut self, items: &[W]) -> (Node, Sum) {
        let mut out: Vec<Node> = Vec::new();
        let mut exits: BTreeSet<usize> = BTreeSet::new();
        for (i, item) in items.iter().enumerate() {
            let (node, sum) = self.lower(item);
            let rest = &items[i + 1..];
            if !sum.falls_through {
                // Nothing after this point runs: every path branches away or
                // exits to an enclosing block.
                out.push(node);
                exits.extend(sum.exits);
                return (Node::Seq(out), Sum { falls_through: false, exits });
            }
            if sum.exits.is_empty() || rest.is_empty() {
                out.push(node);
                exits.extend(sum.exits);
                continue;
            }
            // The statement may exit to an enclosing block: the rest runs
            // only when it completed normally.
            let (rest_node, rest_sum) = self.lower_seq(rest);
            exits.extend(sum.exits.iter().copied());
            let all_exits: BTreeSet<usize> = rest_sum.exits.union(&exits).copied().collect();
            match node {
                Node::If { cond, then_branch, else_branch } if self.definite_exit_arm(&then_branch, &else_branch).is_some() => {
                    let then_exits = self.definite_exit_arm(&then_branch, &else_branch) == Some(true);
                    let (t, e) = if then_exits {
                        (*then_branch, Node::Seq(vec![*else_branch, rest_node]))
                    } else {
                        (Node::Seq(vec![*then_branch, rest_node]), *else_branch)
                    };
                    out.push(Node::If { cond, then_branch: Box::new(t), else_branch: Box::new(e) });
                }
                node => {
                    out.push(node);
                    let guard = Cond::Or(sum.exits.iter().map(|&y| Cond::Flag(self.flags[&Target::Exit(y)])).collect());
                    out.push(Node::If { cond: guard.negate(), then_branch: Box::new(rest_node), else_branch: Box::new(Node::empty()) });
                }
            }
            return (Node::Seq(out), Sum { falls_through: rest_sum.falls_through, exits: all_exits });
        }
        (Node::Seq(out), Sum { falls_through: true, exits })
    }

    /// For an `if` whose one arm never completes normally and whose other arm
    /// completes normally without pending exits: `Some(true)` if the then arm
    /// is the exiting one.
    fn definite_exit_arm(&self, then_branch: &Node, else_branch: &Node) -> Option<bool> {
        let (t, e) = (summarize(then_branch, &self.flags), summarize(else_branch, &self.flags));
        if !t.falls_through && e.exits.is_empty() && e.falls_through {
            Some(true)
        } else if !e.falls_through && t.exits.is_empty() && t.falls_through {
            Some(false)
        } else {
            None
        }
    }
}

/// Summary of an already lowered node: completes normally? pending exits
/// (goto variables of block exits set as the last action).
fn summarize(n: &Node, flags: &HashMap<Target, FlagId>) -> Sum {
    let exit_of: HashMap<FlagId, usize> = flags
        .iter()
        .filter_map(|(t, &f)| match t {
            Target::Exit(y) => Some((f, *y)),
            _ => None,
        })
        .collect();
    fn go(n: &Node, exit_of: &HashMap<FlagId, usize>) -> Sum {
        match n {
            Node::Code(_) => Sum::normal(),
            Node::SetFlag(f, true) => match exit_of.get(f) {
                Some(&y) => Sum { falls_through: false, exits: [y].into_iter().collect() },
                None => Sum::normal(),
            },
            Node::SetFlag(..) | Node::SetState(_) => Sum::normal(),
            Node::Break | Node::Continue | Node::Return => Sum::terminal(),
            Node::Loop(_) => Sum::normal(),
            Node::If { then_branch, else_branch, .. } => go(then_branch, exit_of).join(&go(else_branch, exit_of)),
            Node::Seq(v) => {
                let mut exits = BTreeSet::new();
                for item in v {
                    let s = go(item, exit_of);
                    exits.extend(s.exits);
                    if !s.falls_through {
                        return Sum { falls_through: false, exits };
                    }
                }
                Sum { falls_through: true, exits }
            }
        }
    }
    go(n, &exit_of)
}

// ---------------------------------------------------------------------------
// Step 4: clean-up
// ---------------------------------------------------------------------------

fn cleanup(mut tree: Node) -> Node {
    loop {
        let mut used = BTreeSet::new();
        tree.walk(&mut |n| {
            if let Node::If { cond, .. } = n {
                cond.flags(&mut used);
            }
        });
        let before = tree.clone();
        tree = simplify(tree, &used);
        if tree == before {
            return tree;
        }
    }
}

fn simplify(n: Node, used: &BTreeSet<FlagId>) -> Node {
    match n {
        Node::SetFlag(f, _) if !used.contains(&f) => Node::empty(),
        Node::Seq(items) => {
            let mut out = Vec::new();
            for item in items {
                let s = simplify(item, used);
                let terminal = matches!(s, Node::Break | Node::Continue | Node::Return);
                match s {
                    Node::Seq(inner) => out.extend(inner),
                    s => out.push(s),
                }
                if terminal {
                    break;
                }
            }
            out.retain(|n| !n.is_empty());
            if out.len() == 1 {
                out.pop().unwrap()
            } else {
                Node::Seq(out)
            }
        }
        Node::If { cond, then_branch, else_branch } => {
            let t = simplify(*then_branch, used);
            let e = simplify(*else_branch, used);
            match (t.is_empty(), e.is_empty()) {
                (true, true) => Node::empty(),
                (true, false) => Node::If { cond: cond.negate(), then_branch: Box::new(e), else_branch: Box::new(Node::empty()) },
                _ => Node::If { cond, then_branch: Box::new(t), else_branch: Box::new(e) },
            }
        }
        Node::Loop(b) => Node::Loop(Box::new(simplify(*b, used))),
        other => other,
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn go(n: &Node, depth: usize, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let pad = "  ".repeat(depth);
            match n {
                Node::Code(b) => writeln!(f, "{pad}B{b}"),
                Node::Seq(v) => v.iter().try_for_each(|n| go(n, depth, f)),
                Node::If { cond, then_branch, else_branch } => {
                    writeln!(f, "{pad}if {cond:?} {{")?;
                    go(then_branch, depth + 1, f)?;
                    if !else_branch.is_empty() {
                        writeln!(f, "{pad}}} else {{")?;
                        go(else_branch, depth + 1, f)?;
                    }
                    writeln!(f, "{pad}}}")
                }
                Node::Loop(b) => {
                    writeln!(f, "{pad}loop {{")?;
                    go(b, depth + 1, f)?;
                    writeln!(f, "{pad}}}")
                }
                Node::Break => writeln!(f, "{pad}break"),
                Node::Continue => writeln!(f, "{pad}continue"),
                Node::Return => writeln!(f, "{pad}return"),
                Node::SetFlag(id, v) => writeln!(f, "{pad}goto{id} = {v}"),
                Node::SetState(b) => writeln!(f, "{pad}state = B{b}"),
            }
        }
        go(self, 0, f)
    }
}
