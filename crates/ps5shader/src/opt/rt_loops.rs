// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Hardware traversal of guest BVH loops.
//!
//! A guest ray-tracing shader walks its BVH in a loop of
//! `image_bvh_intersect_ray` queries with a node stack kept in VGPR lanes:
//! the loop starts at the root node (pointer 0x25), every box query pushes
//! the children its rays hit (one entry per child: node pointer plus the
//! lane mask of the rays that hit it, `v_writelane` at a counter), and each
//! iteration pops an entry (`v_readlane` at the counter minus one) until
//! the counter says the stack is empty. A top level's non-box nodes are
//! instances: their record gives the BLAS, whose own loop (a second stack
//! level on the same lanes) pops leaf lists, whose entries the loop tests
//! with triangle queries and the shader's own candidate code (alpha tests,
//! culling, the accept decision). The wave walks the union of its lanes'
//! paths with wave-uniform node pointers.
//!
//! The pass recognises such a nest by its properties, not by its address:
//! a loop whose header carries the root pointer into a BVH query, node
//! stacks popped by lane reads at `counter - 1` (one stack per loop level,
//! at most two: top and bottom), emptiness tested by comparing the counter
//! (with the stack's base), the pops' lane-mask words feeding inverse
//! ballots, and no memory writes in the loop proper (the candidates' order
//! changes).
//!
//! It then adds a copy of the nest whose nodes come from host ray queries
//! ([`RtOp`]): each invocation runs a query over the host's mirror of the
//! BVH (`Begin`, with the guest ray, its instance mask, and facing
//! culling and first-hit ending when the nest culls by facing and never
//! narrows its query extent); the RT cores commit opaque triangles (by the
//! leaf entries' and instances' flags) and hand out the others one at a
//! time, the closest committed hit last. The copy's stack pops become
//! wave pops of those candidates: the node of the first invocation with
//! one pending (its instance at the top level, the first entry of its leaf
//! list at the bottom) and every invocation sharing it as the popped mask;
//! emptiness tests become "no invocation has a candidate pending" (those
//! without one fetch first). Everything else stays the guest's: its
//! instance setup, its walk of each leaf list with its own triangle test
//! and candidate code, its masks, ballots and lane reads (the wave still
//! walks together); the top level's box queries return no children. An
//! instance whose BLAS the host does not mirror is a box candidate: for it
//! the bottom level walks the guest's own stack and box queries, as the
//! original. With [`Options::merge_bottom`] a leaf list popped at the
//! bottom of two levels is walked by every invocation that has a
//! candidate of the popped instance pending, not only by those whose
//! candidate is in that list. A pop never takes the guest counter below its base (the
//! guest's own pushes and clamps stay consistent). At run time the copy
//! runs when the host mirrors the BVH ([`RtOp::Lookup`], the same for the
//! whole wave); otherwise the original nest does.

use crate::fxhash::{HashMap, HashSet};
use crate::ssa::*;

/// The node pointer of a BVH's root (the BOX32 node after the header).
pub const ROOT_NODE: u64 = 0x25;

/// What [`replace_traversals`] did.
#[derive(Debug, Default)]
pub struct Outcome {
    /// Nests that got a hardware copy.
    pub replaced: usize,
    /// Why recognised loops (header block) were left alone.
    pub skipped: Vec<(Block, String)>,
}

/// How [`replace_traversals_with`] builds the copies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Options {
    /// At the bottom of a two-level nest, a popped leaf list is walked by
    /// every invocation with a candidate of the popped instance pending
    /// (each with its own ray: the guest's triangle tests decide), not
    /// only by those whose pending candidate is in that list. An
    /// invocation keeps a pending candidate that is in another list, and
    /// its query's later candidates in a list it walked are not handed out
    /// again. The invocations' queries reach the lists of an instance in
    /// different orders; walking a list once for all of them replaces a
    /// pop per invocation or group of invocations.
    pub merge_bottom: bool,
}

/// Add a hardware copy to every guest traversal nest of `func` that
/// qualifies. The function stays valid either way.
pub fn replace_traversals(func: &mut Function) -> Outcome {
    replace_traversals_with(func, Options::default())
}

/// [`replace_traversals`] with `options`.
pub fn replace_traversals_with(func: &mut Function, options: Options) -> Outcome {
    let mut out = Outcome::default();
    let mut done: HashSet<Block> = HashSet::default();
    let mut slot = 0u8;
    // Each replacement changes the blocks: analyse again after each.
    loop {
        let cfg = Loops::of(func);
        let users = users(func);
        let Some(h) = cfg.loops.iter().map(|l| l.header).find(|&h| !done.contains(&h) && top_loop(func, &cfg, &users, h).is_some() && !inside_other_top(func, &cfg, &users, h)) else { break };
        done.insert(h);
        match analyse(func, &cfg, &users, h) {
            Ok(nest) => {
                let copy = rewrite(func, &nest, slot, options);
                done.insert(copy);
                slot = slot.wrapping_add(1);
                out.replaced += 1;
            }
            Err(why) => out.skipped.push((h, why)),
        }
    }
    out
}

// ----------------------------------------------------------------------
// Loops
// ----------------------------------------------------------------------

struct LoopInfo {
    header: Block,
    merge: Block,
    /// Blocks of the construct (the order range from the header to the
    /// merge), in order.
    body: Vec<Block>,
}

struct Loops {
    loops: Vec<LoopInfo>,
    /// Position of each block in `order` (usize::MAX: not placed).
    pos: Vec<usize>,
}

impl Loops {
    fn of(func: &Function) -> Loops {
        let mut pos = vec![usize::MAX; func.blocks.len()];
        for (i, &b) in func.order.iter().enumerate() {
            pos[b.index()] = i;
        }
        let mut loops = Vec::new();
        for &b in &func.order {
            if let Merge::Loop { merge, .. } = func.block(b).merge {
                let (a, z) = (pos[b.index()], pos[merge.index()]);
                if z == usize::MAX || z <= a {
                    continue;
                }
                loops.push(LoopInfo { header: b, merge, body: func.order[a..z].to_vec() });
            }
        }
        Loops { loops, pos }
    }

    fn get(&self, h: Block) -> Option<&LoopInfo> {
        self.loops.iter().find(|l| l.header == h)
    }

    fn contains(&self, l: &LoopInfo, b: Block) -> bool {
        let (a, z, p) = (self.pos[l.header.index()], self.pos[l.merge.index()], self.pos[b.index()]);
        p != usize::MAX && p >= a && p < z
    }

    /// The innermost loop holding `b`.
    fn innermost(&self, b: Block) -> Option<&LoopInfo> {
        self.loops.iter().filter(|l| self.contains(l, b)).min_by_key(|l| l.body.len())
    }
}

/// Users of every value: (user, operand index); a terminator condition is
/// recorded with operand index `usize::MAX` and the block's first value.
fn users(func: &Function) -> Vec<Vec<(Value, usize)>> {
    let mut u = vec![Vec::new(); func.values.len()];
    for &b in &func.order {
        for &v in &func.block(b).insts {
            for (k, &a) in func.args(v).iter().enumerate() {
                u[a.index()].push((v, k));
            }
        }
    }
    u
}

fn is_bvh(func: &Function, v: Value) -> bool {
    matches!(func.op(v), Op::Image(a) if matches!(a.op, ImageOp::BvhIntersect { .. }))
}

/// `v` reaches the node operand of a BVH query of the loop `l` through
/// phis and selects.
fn reaches_node(func: &Function, cfg: &Loops, users: &[Vec<(Value, usize)>], l: &LoopInfo, v: Value) -> bool {
    let mut seen: HashSet<Value> = HashSet::default();
    let mut work = vec![v];
    while let Some(x) = work.pop() {
        if !seen.insert(x) || seen.len() > 4096 {
            continue;
        }
        for &(u, k) in &users[x.index()] {
            let Some(b) = func.value(u).block else { continue };
            if !cfg.contains(l, b) {
                continue;
            }
            match func.op(u) {
                Op::Image(_) if is_bvh(func, u) && k == 3 => return true,
                Op::Phi => work.push(u),
                Op::Select if k > 0 => work.push(u),
                _ => {}
            }
        }
    }
    false
}

/// The half of the wave (0 low, 1 high) of the lane mask word `v`: the
/// operand of an inverse ballot it reaches through phis, selects and
/// bitwise operations.
fn inverse_ballot_half(func: &Function, users: &[Vec<(Value, usize)>], v: Value) -> Option<u8> {
    let mut seen: HashSet<Value> = HashSet::default();
    let mut work = vec![v];
    while let Some(x) = work.pop() {
        if !seen.insert(x) || seen.len() > 4096 {
            continue;
        }
        for &(u, k) in &users[x.index()] {
            match func.op(u) {
                Op::InverseBallot => return Some(k as u8),
                Op::Phi | Op::And | Op::Or | Op::Xor | Op::Not => work.push(u),
                Op::Select if k > 0 => work.push(u),
                _ => {}
            }
        }
    }
    None
}

/// The root phi of a loop: a header phi entering with the root pointer and
/// reaching a BVH query's node operand.
fn top_loop(func: &Function, cfg: &Loops, users: &[Vec<(Value, usize)>], h: Block) -> Option<Value> {
    let l = cfg.get(h)?;
    let bd = func.block(h);
    bd.insts.iter().copied().take_while(|&v| func.op(v) == Op::Phi).find(|&phi| {
        func.ty(phi) == Type::U32
            && bd.preds.iter().zip(func.args(phi)).any(|(p, &a)| !cfg.contains(l, *p) && func.as_const(a) == Some(ROOT_NODE))
            && reaches_node(func, cfg, users, l, phi)
    })
}

fn inside_other_top(func: &Function, cfg: &Loops, users: &[Vec<(Value, usize)>], h: Block) -> bool {
    cfg.loops.iter().any(|o| o.header != h && cfg.contains(o, h) && top_loop(func, cfg, users, o.header).is_some())
}

// ----------------------------------------------------------------------
// Analysis
// ----------------------------------------------------------------------

/// One node stack: its counter, the lane reads that pop it, the counter
/// tests that say whether it is empty, its level.
struct Stack {
    counter: Value,
    /// `counter - 1` (the pop's index and the counter after it).
    decrement: Value,
    node: Value,
    /// Mask reads with the half of the wave they hold (0 low, 1 high).
    masks: Vec<(Value, u8)>,
    tests: Vec<Value>,
    level: u8,
}

/// A value at the nest's entry: one that exists there, or pure operations
/// on such values made again ahead of the nest.
#[derive(Clone)]
enum Entry {
    Value(Value),
    Op(Op, Type, Vec<Entry>),
    True,
}

impl Entry {
    /// The value, made at the end of `b`.
    fn make(&self, func: &mut Function, b: Block) -> Value {
        match self {
            Entry::Value(v) => *v,
            Entry::True => func.const_bool(true),
            Entry::Op(op, ty, args) => {
                let args = args.iter().map(|a| a.make(func, b)).collect();
                func.append(b, *op, *ty, args)
            }
        }
    }
}

struct Nest {
    header: Block,
    merge: Block,
    pre: Block,
    body: Vec<Block>,
    /// The root query's descriptor at the nest's entry.
    desc: Entry,
    /// Ray operands at the nest's entry: origin xyz, direction xyz, extent.
    ray: Vec<Entry>,
    /// The invocation takes part in the root query (its EXEC there): the
    /// others fetch no candidates.
    active: Entry,
    /// The ray's instance mask, when the nest tests instance records' mask
    /// byte (bits 31:24 of the word at +60) against a value from before
    /// the nest: the host query culls by it too.
    mask: Option<Entry>,
    /// The nest culls by facing (it tests the sign of a triangle query's
    /// `t_num`): the host query culls the same faces (the instances' flags
    /// may disable or flip it, as the guest's do). The guest keeps the
    /// triangles whose `t_num` is negative unless the instance is flipped
    /// (the console drivers' rule `sign(t_num) != flip`); Vulkan calls those
    /// back-facing, so the host culls front-facing ones.
    cull: bool,
    /// The root query's extent never changes in the nest (a header phi or
    /// a value made in it would): the guest accepts the first candidate it
    /// accepts, it never searches for the closest, so the host query may
    /// stop at its first opaque hit.
    first_hit: bool,
    stacks: Vec<Stack>,
    levels: u8,
    /// The BVH queries of the top stack's own loop (not inside the bottom
    /// one).
    top_queries: Vec<Value>,
}

/// `v` = `(counter - 1) & 63`: the counter.
fn pop_counter(func: &Function, idx: Value) -> Option<Value> {
    let d = func.value(idx);
    if d.op != Op::And || d.args.len() != 2 {
        return None;
    }
    let (a, m) = if func.as_const(d.args[1]) == Some(0x3F) { (d.args[0], d.args[0]) } else if func.as_const(d.args[0]) == Some(0x3F) { (d.args[1], d.args[1]) } else { return None };
    let _ = m;
    let s = func.value(a);
    if s.op != Op::IAdd {
        return None;
    }
    if func.as_const(s.args[1]) == Some(0xFFFF_FFFF) {
        Some(s.args[0])
    } else if func.as_const(s.args[0]) == Some(0xFFFF_FFFF) {
        Some(s.args[1])
    } else {
        None
    }
}

fn analyse(func: &Function, cfg: &Loops, users: &[Vec<(Value, usize)>], h: Block) -> Result<Nest, String> {
    if func.wave_size != 64 && func.wave_size != 32 {
        return Err("wave size".into());
    }
    let l = cfg.get(h).ok_or("no loop")?;
    let root = top_loop(func, cfg, users, h).ok_or("no root")?;
    let mut in_body = vec![false; func.blocks.len()];
    for &b in &l.body {
        in_body[b.index()] = true;
    }
    // Entered only through the header from one preheader; left only to the
    // merge.
    let outside: Vec<Block> = func.block(h).preds.iter().copied().filter(|p| !in_body[p.index()]).collect();
    let [pre] = outside[..] else { return Err(format!("{} entries", outside.len())) };
    if func.block(pre).term != Terminator::Branch(h) {
        return Err("preheader does not branch to the header alone".into());
    }
    for &b in &func.order {
        let succ = func.block(b).term.successors();
        if in_body[b.index()] {
            if let Some(s) = succ.iter().find(|s| !in_body[s.index()] && **s != l.merge) {
                return Err(format!("b{} leaves the nest to b{}", b.0, s.0));
            }
            let m = match func.block(b).merge {
                Merge::Selection { merge } => Some(merge),
                Merge::Loop { merge, continue_block } => {
                    if !in_body[continue_block.index()] {
                        return Err("continue outside".into());
                    }
                    Some(merge)
                }
                Merge::None => None,
            };
            if let Some(m) = m.filter(|m| !in_body[m.index()] && *m != l.merge) {
                return Err(format!("b{} merges at b{} outside", b.0, m.0));
            }
        } else if b != pre && succ.iter().any(|s| in_body[s.index()]) {
            return Err(format!("b{} enters the nest", b.0));
        }
    }
    // The value of `v` where the nest is entered: as it is, a header phi's
    // entry operand, or pure operations of those (made again ahead of the
    // nest).
    fn entry_value(func: &Function, in_body: &[bool], h: Block, pre: Block, v: Value, depth: u32) -> Option<Entry> {
        let d = func.value(v);
        match d.block {
            None => Some(Entry::Value(v)),
            Some(b) if !in_body[b.index()] => Some(Entry::Value(v)),
            Some(b) if b == h && d.op == Op::Phi => {
                let k = func.block(h).preds.iter().position(|&p| p == pre)?;
                let a = d.args[k];
                func.value(a).block.is_none_or(|ab| !in_body[ab.index()]).then_some(Entry::Value(a))
            }
            _ if depth < 12 && d.op.is_pure() && !d.op.is_subgroup() => {
                let args = d.args.iter().map(|&a| entry_value(func, in_body, h, pre, a, depth + 1)).collect::<Option<Vec<_>>>()?;
                Some(Entry::Op(d.op, d.ty, args))
            }
            _ => None,
        }
    }
    let entry_value = |v: Value| entry_value(func, &in_body, h, pre, v, 0);
    // The loop proper: blocks that reach the loop's continue block. The
    // others are its exits the structurizer kept inside (they run once per
    // invocation in the copy, after its traversal).
    let Merge::Loop { continue_block, .. } = func.block(h).merge else { return Err("not a loop".into()) };
    let mut cycle = vec![false; func.blocks.len()];
    cycle[continue_block.index()] = true;
    let mut grew = true;
    while grew {
        grew = false;
        for &b in &l.body {
            if !cycle[b.index()] && b != continue_block && func.block(b).term.successors().iter().any(|s| s.index() < cycle.len() && cycle[s.index()] && *s != h) {
                cycle[b.index()] = true;
                grew = true;
            }
        }
    }
    cycle[h.index()] = true;
    let mut stacks: HashMap<Value, (Value, Vec<Value>)> = HashMap::default();
    let mut root_query = None;
    for &b in &l.body {
        for &v in &func.block(b).insts {
            let d = func.value(v);
            match d.op {
                Op::Image(a) => match a.op {
                    ImageOp::BvhIntersect { bvh64: true, .. } => return Err("64-bit node pointers".into()),
                    ImageOp::BvhIntersect { .. } => {
                        if a.a16 || d.args.len() < 12 {
                            return Err("BVH query with packed or short operands".into());
                        }
                        let mut n = d.args[3];
                        for _ in 0..8 {
                            if func.op(n) == Op::Select {
                                n = func.args(n)[1];
                            }
                        }
                        if n == root {
                            root_query = Some(v);
                        }
                    }
                    ImageOp::Store | ImageOp::Atomic(_) => return Err("image write in the nest".into()),
                    _ => {}
                },
                Op::ReadLane => {
                    if let Some(c) = pop_counter(func, d.args[1]) {
                        stacks.entry(c).or_insert((c, Vec::new())).1.push(v);
                    }
                }
                Op::Clock { .. } | Op::RtQuery(_) | Op::Export(_) | Op::SendMessage { .. } => return Err(format!("{} in the nest", d.op.name())),
                // Memory writes only on the way out (never between
                // candidates, whose order the host changes).
                op if op.has_side_effects() && cycle[b.index()] && !matches!(op, Op::Barrier | Op::SubgroupLdsBarrier) => return Err(format!("{} %{} in the nest", op.name(), v.0)),
                _ => {}
            }
        }
    }
    let rq = root_query.ok_or("no root query")?;
    let rd = func.value(rq);
    let desc = entry_value(rd.args[0]).ok_or("root descriptor made in the nest")?;
    let mut ray = Vec::new();
    for i in [5usize, 6, 7, 8, 9, 10, 4] {
        ray.push(entry_value(rd.args[i]).ok_or("root ray made in the nest")?);
    }
    // (Unknown: every invocation queries; the guest's own masks drop the
    // nodes of inactive ones, at the cost of the wave's iterations.)
    let active = entry_value(*rd.args.last().unwrap()).unwrap_or(Entry::True);
    // `bfe(record word, 24, 8) & mask`: the instance mask test.
    let mut mask = None;
    'm: for &b in &l.body {
        for &v in &func.block(b).insts {
            let d = func.value(v);
            if d.op != Op::And || d.ty != Type::U32 {
                continue;
            }
            for (x, m) in [(d.args[0], d.args[1]), (d.args[1], d.args[0])] {
                let xd = func.value(x);
                let byte = xd.op == Op::BitFieldUExtract && func.as_const(xd.args[1]) == Some(24) && func.as_const(xd.args[2]) == Some(8);
                let loaded = byte && matches!(func.op(xd.args[0]), Op::ScalarLoad { .. } | Op::ScalarBufferLoad { .. } | Op::GlobalLoad { .. } | Op::BufferLoad { .. });
                if loaded {
                    if let Some(e) = entry_value(m) {
                        mask = Some(e);
                        break 'm;
                    }
                }
            }
        }
    }
    let first_hit = func.value(rd.args[4]).block.is_none_or(|b| !in_body[b.index()]);
    // `t_num >> 31` of a triangle query: the facing test.
    let strip = |mut x: Value| {
        for _ in 0..8 {
            if func.op(x) == Op::Select {
                x = func.args(x)[1];
            }
        }
        x
    };
    let cull = l.body.iter().flat_map(|&b| func.block(b).insts.iter().copied()).any(|v| {
        let d = func.value(v);
        d.op == Op::LShr && func.as_const(d.args[1]) == Some(31) && {
            let x = strip(d.args[0]);
            func.op(x) == Op::Extract(0) && is_bvh(func, func.args(x)[0])
        }
    });
    // Each stack: its node read (the one reaching a query), its mask reads,
    // its tests, its level.
    let mut list: Vec<Stack> = Vec::new();
    for (counter, reads) in stacks.into_values() {
        let decrement = func.args(func.args(reads[0])[1]).iter().copied().find(|&a| func.op(a) == Op::IAdd).ok_or("pop index")?;
        let mut s = Stack { counter, decrement, node: reads[0], masks: Vec::new(), tests: Vec::new(), level: 0 };
        let lp = cfg.innermost(func.value(s.node).block.unwrap()).ok_or("pop outside a loop")?;
        let nodes: Vec<Value> = reads.iter().copied().filter(|&r| reaches_node(func, cfg, users, l, r)).collect();
        let [node] = nodes[..] else { return Err(format!("stack %{}: {} node reads", s.counter.0, nodes.len())) };
        s.node = node;
        for r in reads.into_iter().filter(|&r| r != node) {
            let half = inverse_ballot_half(func, users, r).ok_or_else(|| format!("stack read %{} is neither node nor mask", r.0))?;
            s.masks.push((r, half));
        }
        let mut halves: Vec<u8> = s.masks.iter().map(|m| m.1).collect();
        halves.sort();
        if halves != [0, 1] && !(func.wave_size == 32 && halves == [0]) {
            return Err(format!("stack %{}: mask halves {halves:?}", s.counter.0));
        }
        let block = func.value(node).block;
        if s.masks.iter().any(|m| func.value(m.0).block != block) {
            return Err("stack popped in two blocks".into());
        }
        for &(u, _) in &users[s.counter.index()] {
            if matches!(func.op(u), Op::ICmp(ICmp::Eq | ICmp::Ne)) && func.value(u).block.is_some_and(|b| in_body[b.index()]) {
                s.tests.push(u);
            }
        }
        if s.tests.is_empty() {
            return Err(format!("stack %{} has no emptiness test", s.counter.0));
        }
        if lp.header == h {
            s.level = 0;
        } else {
            s.level = 1;
        }
        list.push(s);
    }
    list.sort_by_key(|s| s.level);
    let levels = match list.iter().map(|s| s.level).collect::<Vec<_>>()[..] {
        [0] => 1,
        [0, 1] => 2,
        ref other => return Err(format!("stack levels {other:?}")),
    };
    if levels == 2 {
        let inner = cfg.innermost(func.value(list[1].node).block.unwrap()).unwrap().header;
        if !cfg.loops.iter().any(|o| o.header == inner && cfg.contains(l, inner)) {
            return Err("bottom stack outside the nest".into());
        }
    }
    let bottom = (levels == 2).then(|| cfg.innermost(func.value(list[1].node).block.unwrap()).unwrap().header);
    let top_queries = l
        .body
        .iter()
        .filter(|&&b| bottom.is_none_or(|bh| cfg.get(bh).is_none_or(|bl| !cfg.contains(bl, b))))
        .flat_map(|&b| func.block(b).insts.iter().copied())
        .filter(|&v| is_bvh(func, v))
        .collect();
    Ok(Nest { header: h, merge: l.merge, pre, body: l.body.clone(), desc, ray, active, mask, cull, first_hit, stacks: list, levels, top_queries })
}

// ----------------------------------------------------------------------
// Rewrite
// ----------------------------------------------------------------------

fn place(func: &mut Function, b: Block, at: usize, op: Op, ty: Type, args: Vec<Value>) -> Value {
    let v = func.new_value(op, ty, args);
    func.value_mut(v).block = Some(b);
    func.block_mut(b).insts.insert(at, v);
    v
}

fn index_in(func: &Function, b: Block, v: Value) -> usize {
    func.block(b).insts.iter().position(|&x| x == v).expect("value in its block")
}

/// Returns the copy's header.
fn rewrite(func: &mut Function, n: &Nest, slot: u8, options: Options) -> Block {
    let (h, x, pre) = (n.header, n.merge, n.pre);
    // New blocks: the selection header and the copy's prelude, the two
    // loops' own merges.
    let s = func.new_block();
    let s1 = func.new_block();
    let x0 = func.new_block();
    let x1 = func.new_block();
    let x2 = func.new_block();
    let mut bmap: HashMap<Block, Block> = HashMap::default();
    for &b in &n.body {
        let nb = func.new_block();
        bmap.insert(b, nb);
    }
    bmap.insert(x, x1);
    let mut vmap: HashMap<Value, Value> = HashMap::default();
    for &b in &n.body {
        for i in 0..func.block(b).insts.len() {
            let v = func.block(b).insts[i];
            let d = func.value(v).clone();
            let nv = func.new_value(d.op, d.ty, d.args);
            vmap.insert(v, nv);
        }
    }
    let mv = |vmap: &HashMap<Value, Value>, v: Value| *vmap.get(&v).unwrap_or(&v);
    for &b in &n.body {
        let nb = bmap[&b];
        let src = func.block(b).clone();
        let insts: Vec<Value> = src.insts.iter().map(|&v| vmap[&v]).collect();
        for &nv in &insts {
            let args: Vec<Value> = func.value(nv).args.iter().map(|&a| mv(&vmap, a)).collect();
            let d = func.value_mut(nv);
            d.args = args;
            d.block = Some(nb);
        }
        let mb = |blk: Block| *bmap.get(&blk).unwrap_or(&blk);
        let term = match src.term {
            Terminator::Branch(t) => Terminator::Branch(mb(t)),
            Terminator::CondBranch { cond, then_block, else_block } => Terminator::CondBranch { cond: mv(&vmap, cond), then_block: mb(then_block), else_block: mb(else_block) },
            t => t,
        };
        let merge = match src.merge {
            Merge::Selection { merge } => Merge::Selection { merge: mb(merge) },
            Merge::Loop { merge, continue_block } => Merge::Loop { merge: mb(merge), continue_block: mb(continue_block) },
            Merge::None => Merge::None,
        };
        let preds: Vec<Block> = src.preds.iter().map(|&p| if p == pre { s1 } else { mb(p) }).collect();
        let bd = func.block_mut(nb);
        bd.insts = insts;
        bd.term = term;
        bd.merge = merge;
        bd.preds = preds;
    }
    // The original nest leaves to x0, the copy to x1; both join at x.
    for &b in &n.body {
        let retarget = |t: &mut Block| {
            if *t == x {
                *t = x0;
            }
        };
        let bd = func.block_mut(b);
        match &mut bd.term {
            Terminator::Branch(t) => retarget(t),
            Terminator::CondBranch { then_block, else_block, .. } => {
                retarget(then_block);
                retarget(else_block);
            }
            _ => {}
        }
        match &mut bd.merge {
            Merge::Selection { merge } => retarget(merge),
            Merge::Loop { merge, .. } => retarget(merge),
            Merge::None => {}
        }
    }
    let exit_preds = func.block(x).preds.clone();
    let phis: Vec<Value> = func.block(x).insts.iter().copied().take_while(|&v| func.op(v) == Op::Phi).collect();
    let rest: Vec<Value> = func.block(x).insts[phis.len()..].to_vec();
    // x0: the original exits' phis.
    for &p in &phis {
        func.value_mut(p).block = Some(x0);
    }
    {
        let bd = func.block_mut(x0);
        bd.insts = phis.clone();
        bd.preds = exit_preds.clone();
        bd.term = Terminator::Branch(x);
    }
    // x1: their copies over the copy's exits.
    let mut x1_phis = Vec::new();
    for &p in &phis {
        let d = func.value(p).clone();
        let args = d.args.iter().map(|&a| mv(&vmap, a)).collect();
        let np = func.new_value(Op::Phi, d.ty, args);
        func.value_mut(np).block = Some(x1);
        vmap.insert(p, np);
        x1_phis.push(np);
    }
    {
        let preds = exit_preds.iter().map(|p| bmap[p]).collect();
        let bd = func.block_mut(x1);
        bd.insts = x1_phis;
        bd.preds = preds;
        bd.term = Terminator::Branch(x2);
    }
    // x keeps its other instructions (the join phis come first, below).
    func.block_mut(x).insts = rest.clone();
    // Values of the nest used after it: joined at x.
    let mut inside: HashSet<Value> = HashSet::default();
    for &b in &n.body {
        inside.extend(func.block(b).insts.iter().copied());
    }
    inside.extend(phis.iter().copied());
    let skip: HashSet<Block> = n.body.iter().copied().chain(n.body.iter().map(|b| bmap[b])).chain([x0, x1, x2, s, s1]).collect();
    let mut used_after: Vec<Value> = Vec::new();
    let mut seen: HashSet<Value> = HashSet::default();
    for &b in &func.order {
        if skip.contains(&b) {
            continue;
        }
        let bd = func.block(b);
        let cond = match bd.term {
            Terminator::CondBranch { cond, .. } => Some(cond),
            _ => None,
        };
        for a in bd.insts.iter().flat_map(|&v| func.args(v).iter().copied()).chain(cond) {
            if inside.contains(&a) && seen.insert(a) {
                used_after.push(a);
            }
        }
    }
    // An invocation that skipped the copy (inactive at the root query)
    // leaves with the header's entry values (else undefined values: the
    // guest computes nothing for it).
    let pre_index = func.block(h).preds.iter().position(|&p| p == pre);
    let mut join: HashMap<Value, Value> = HashMap::default();
    let mut join_phis = Vec::new();
    let mut x2_phis = Vec::new();
    for &v in &used_after {
        let ty = func.ty(v);
        let d = func.value(v);
        let fallback = match (d.op, d.block, pre_index) {
            (Op::Phi, Some(b), Some(k)) if b == h => d.args[k],
            _ => func.undef(ty),
        };
        let k = func.new_value(Op::Phi, ty, vec![vmap[&v], fallback]);
        func.value_mut(k).block = Some(x2);
        x2_phis.push(k);
        let j = func.new_value(Op::Phi, ty, vec![v, k]);
        func.value_mut(j).block = Some(x);
        join.insert(v, j);
        join_phis.push(j);
    }
    {
        let bd = func.block_mut(x2);
        bd.insts = x2_phis;
        bd.preds = vec![x1, s1];
        bd.term = Terminator::Branch(x);
    }
    for i in 0..func.order.len() {
        let b = func.order[i];
        if skip.contains(&b) {
            continue;
        }
        for k in 0..func.block(b).insts.len() {
            let v = func.block(b).insts[k];
            let args: Vec<Value> = func.args(v).iter().map(|a| *join.get(a).unwrap_or(a)).collect();
            func.value_mut(v).args = args;
        }
        if let Terminator::CondBranch { cond, .. } = &mut func.block_mut(b).term {
            if let Some(&j) = join.get(cond) {
                *cond = j;
            }
        }
    }
    {
        let bd = func.block_mut(x);
        bd.insts = join_phis.into_iter().chain(rest).collect();
        bd.preds = vec![x0, x2];
    }
    // The loops' merges.
    if let Merge::Loop { continue_block, .. } = func.block(h).merge {
        func.block_mut(h).merge = Merge::Loop { merge: x0, continue_block };
    }
    // The selection: the host mirror or the guest's nest.
    func.block_mut(pre).term = Terminator::Branch(s);
    for p in func.block_mut(h).preds.iter_mut() {
        if *p == pre {
            *p = s;
        }
    }
    let desc = n.desc.make(func, s);
    let w0 = func.append(s, Op::Extract(0), Type::U32, vec![desc]);
    let w1 = func.append(s, Op::Extract(1), Type::U32, vec![desc]);
    let entry = func.append(s, Op::RtQuery(RtOp::Lookup { levels: n.levels }), Type::U32, vec![w0, w1]);
    let zero = func.const_u32(0);
    let host = func.append(s, Op::ICmp(ICmp::Ne), Type::BOOL, vec![entry, zero]);
    let h1 = bmap[&h];
    {
        let bd = func.block_mut(s);
        bd.preds = vec![pre];
        bd.merge = Merge::Selection { merge: x };
        bd.term = Terminator::CondBranch { cond: host, then_block: s1, else_block: h };
    }
    // The copy's prelude: every invocation starts its query (an inactive
    // one finds nothing); the wave enters the copy whole.
    let mut begin = vec![entry];
    for r in &n.ray {
        let v = r.make(func, s1);
        begin.push(v);
    }
    let mask = match &n.mask {
        Some(m) => m.make(func, s1),
        None => func.const_u32(0xFF),
    };
    begin.push(mask);
    let active = n.active.make(func, s1);
    begin.push(active);
    // Ray flags (SPIR-V `RayFlags`): cull front-facing triangles (the faces
    // the guest drops, see `Nest::cull`), terminate on the first hit.
    begin.push(func.const_u32(if n.cull { 0x20 } else { 0 } | if n.first_hit { 0x4 } else { 0 }));
    func.append(s1, Op::RtQuery(RtOp::Begin { slot }), Type::VOID, begin);
    let enter = func.const_bool(true);
    {
        let bd = func.block_mut(s1);
        bd.preds = vec![s];
        bd.merge = Merge::Selection { merge: x2 };
        bd.term = Terminator::CondBranch { cond: enter, then_block: h1, else_block: x2 };
    }
    // Block order: the selection, the copy, the original, the join.
    let ph = func.order.iter().position(|&b| b == h).unwrap();
    let px = func.order.iter().position(|&b| b == x).unwrap();
    let mut order: Vec<Block> = func.order[..ph].to_vec();
    order.push(s);
    order.push(s1);
    order.extend(n.body.iter().map(|b| bmap[b]));
    order.push(x1);
    order.push(x2);
    order.extend(n.body.iter().copied());
    order.push(x0);
    order.extend(func.order[px..].iter().copied());
    func.order = order;

    // ---- the copy's nodes come from the host's queries ----
    let levels = n.levels;
    for st in &n.stacks {
        let level = st.level;
        // A pop: the node of the first invocation with a candidate pending
        // at this level, taken by every invocation sharing it (their lanes
        // are the popped mask).
        let node = vmap[&st.node];
        let b = func.value(node).block.unwrap();
        let first_read = std::iter::once(node).chain(st.masks.iter().map(|m| vmap[&m.0])).map(|v| index_in(func, b, v)).min().unwrap();
        let mut at = first_read;
        let mut put = |func: &mut Function, op: Op, ty: Type, args: Vec<Value>| {
            let v = place(func, b, at, op, ty, args);
            at += 1;
            v
        };
        let pending = if level == 0 { RtGet::Ready } else { RtGet::Leaf };
        let r = put(func, Op::RtQuery(RtOp::Get { slot, what: pending }), Type::BOOL, vec![]);
        let bal = put(func, Op::Ballot, Type::U32X2, vec![r]);
        let lo = put(func, Op::Extract(0), Type::U32, vec![bal]);
        let hi = put(func, Op::Extract(1), Type::U32, vec![bal]);
        let flo = put(func, Op::FindLsb, Type::U32, vec![lo]);
        let fhi = put(func, Op::FindLsb, Type::U32, vec![hi]);
        let c32 = func.const_u32(32);
        let fhi = put(func, Op::IAdd, Type::U32, vec![fhi, c32]);
        let zero = func.const_u32(0);
        let lo_set = put(func, Op::ICmp(ICmp::Ne), Type::BOOL, vec![lo, zero]);
        let first = put(func, Op::Select, Type::U32, vec![lo_set, flo, fhi]);
        let c63 = func.const_u32(63);
        let first = put(func, Op::And, Type::U32, vec![first, c63]);
        let what = if level == 0 && levels == 2 { RtGet::Instance } else { RtGet::Entry };
        let key = put(func, Op::RtQuery(RtOp::Get { slot, what }), Type::U32, vec![]);
        let wave_key = put(func, Op::ReadLane, Type::U32, vec![key, first]);
        let same = put(func, Op::ICmp(ICmp::Eq), Type::BOOL, vec![key, wave_key]);
        let own = put(func, Op::And, Type::BOOL, vec![r, same]);
        // A merged bottom pop: every invocation with a candidate pending
        // at the bottom (always one of the popped instance) walks the list.
        let merge = options.merge_bottom && level == 1 && levels == 2;
        let take = if merge { r } else { own };
        let taken = put(func, Op::Ballot, Type::U32X2, vec![take]);
        let mut take_args = vec![take];
        if merge {
            take_args.push(own);
            take_args.push(wave_key);
        }
        if level == 0 && levels == 2 {
            // Whether the popped instance's BLAS walks the guest's stack.
            let px = put(func, Op::RtQuery(RtOp::Get { slot, what: RtGet::Proxy }), Type::BOOL, vec![]);
            let one = func.const_u32(1);
            let pxu = put(func, Op::Select, Type::U32, vec![px, one, zero]);
            let wave_px = put(func, Op::ReadLane, Type::U32, vec![pxu, first]);
            take_args.push(wave_px);
        }
        put(func, Op::RtQuery(RtOp::Take { slot, level, levels }), Type::VOID, take_args);
        // At the bottom of two levels, an instance without a host BLAS pops
        // the guest's own stack (its original reads, kept).
        let guest = (level == 1).then(|| put(func, Op::RtQuery(RtOp::Get { slot, what: RtGet::InGuest }), Type::BOOL, vec![]));
        let mut keep_original = |func: &mut Function, v: Value| -> Option<Value> {
            guest.map(|_| {
                let d = func.value(v).clone();
                put(func, d.op, d.ty, d.args)
            })
        };
        let original_node = keep_original(func, node);
        let original_masks: Vec<Option<Value>> = st.masks.iter().map(|&(m, _)| keep_original(func, vmap[&m])).collect();
        // The node: an instance node (record index << 3) or a leaf entry.
        let oracle_node = {
            let three = func.const_u32(3);
            if level == 0 && levels == 2 {
                put(func, Op::Shl, Type::U32, vec![wave_key, three])
            } else {
                wave_key
            }
        };
        let choose = |func: &mut Function, v: Value, oracle: Value, original: Option<Value>| {
            let d = func.value_mut(v);
            match (guest, original) {
                (Some(g), Some(o)) => {
                    d.op = Op::Select;
                    d.args = vec![g, o, oracle];
                }
                _ => {
                    d.op = Op::Bitcast;
                    d.args = vec![oracle];
                }
            }
        };
        choose(func, node, oracle_node, original_node);
        for (&(m, half), original) in st.masks.iter().zip(original_masks) {
            let w = place(func, b, index_in(func, b, vmap[&m]), Op::Extract(half), Type::U32, vec![taken]);
            choose(func, vmap[&m], w, original);
        }
        // An emptiness test: no invocation has a candidate pending at this
        // level (those without one fetch their next first: at the bottom,
        // the invocations in the popped instance, keeping a candidate in
        // it there).
        for t in &st.tests {
            let t1 = vmap[t];
            let b = func.value(t1).block.unwrap();
            let mut at = index_in(func, b, t1);
            let mut put = |func: &mut Function, op: Op, ty: Type, args: Vec<Value>| {
                let v = place(func, b, at, op, ty, args);
                at += 1;
                v
            };
            if level == 0 || levels == 2 {
                put(func, Op::RtQuery(RtOp::Fetch { slot, level }), Type::VOID, vec![]);
            }
            let r = put(func, Op::RtQuery(RtOp::Get { slot, what: pending }), Type::BOOL, vec![]);
            let bal = put(func, Op::Ballot, Type::U32X2, vec![r]);
            let lo = put(func, Op::Extract(0), Type::U32, vec![bal]);
            let hi = put(func, Op::Extract(1), Type::U32, vec![bal]);
            let any = put(func, Op::Or, Type::U32, vec![lo, hi]);
            let zero = func.const_u32(0);
            // `counter == base` is empty, `counter != base` is not.
            let d = func.value(t1).clone();
            let oracle = put(func, d.op, Type::BOOL, vec![any, zero]);
            if level == 1 {
                let original = put(func, d.op, Type::BOOL, d.args.clone());
                let guest = put(func, Op::RtQuery(RtOp::Get { slot, what: RtGet::InGuest }), Type::BOOL, vec![]);
                let d = func.value_mut(t1);
                d.op = Op::Select;
                d.args = vec![guest, original, oracle];
            } else {
                let d = func.value_mut(t1);
                d.op = Op::Bitcast;
                d.args = vec![oracle];
            }
        }
    }
    // An oracle pop takes no guest stack entry it did not push: the
    // counter stops at the emptiness test's operand (the stack's base),
    // the BLAS root an instance pushed is the only entry it takes. (The
    // guest's own clamps and pushes stay consistent with the counter.)
    for st in &n.stacks {
        let dec = vmap[&st.decrement];
        let Some(&test) = st.tests.first() else { continue };
        let t1 = vmap[&test];
        // The test's original operands: the one that is not the counter.
        let d = func.value(test).clone();
        let base = if d.args[0] == st.counter { d.args[1] } else { d.args[0] };
        let base = *vmap.get(&base).unwrap_or(&base);
        let c = vmap[&st.counter];
        let b = func.value(dec).block.unwrap();
        let at = index_in(func, b, dec);
        let minus = place(func, b, at, Op::IAdd, Type::U32, func.args(dec).to_vec());
        let above = place(func, b, at + 1, Op::ICmp(ICmp::Ne), Type::BOOL, vec![c, base]);
        let take_entry = if st.level == 1 {
            // Walking the guest's stack: its own pops.
            let g = place(func, b, at + 2, Op::RtQuery(RtOp::Get { slot, what: RtGet::InGuest }), Type::BOOL, vec![]);
            place(func, b, at + 3, Op::Or, Type::BOOL, vec![above, g])
        } else {
            above
        };
        let dd = func.value_mut(dec);
        dd.op = Op::Select;
        dd.args = vec![take_entry, minus, c];
        let _ = t1;
    }
    // The top level's box queries return no children: the host's queries
    // found its nodes (the bottom level's walk the BLASes the host lacks).
    for &v in &n.top_queries {
        let v1 = vmap[&v];
        if let Op::Image(mut a) = func.op(v1) {
            if let ImageOp::BvhIntersect { bvh64, .. } = a.op {
                a.op = ImageOp::BvhIntersect { bvh64, boxes: false };
                func.value_mut(v1).op = Op::Image(a);
            }
        }
    }
    h1
}
