// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Lane-level passes:
//!
//! * [`eliminate_read_lanes`] (ReadLaneEliminator): `read_lane(v, k)` of a
//!   value equal in every invocation is the value itself; a read of a lane a
//!   `v_writelane` (`select(lane_id == k, x, old)`) wrote is `x`, looking
//!   through EXEC selects and phis whose operands agree (SGPR spills to VGPR
//!   lanes).
//! * [`eliminate_masked_selects`] (MaskedSelectEliminator): `select(c, new,
//!   old)` is `new` when no invocation where `c` is false ever observes the
//!   result (every use is guarded by a condition implying `c`).
//! * [`project_lane_masks`] (LaneMaskProjection): in wave32, the lane's
//!   bit of a mask computed from ballots is computed from their predicates.
//! * [`insert_lds_barriers`] (SharedMemoryBarrierInserter): a wave's LDS
//!   writes become visible to its own other invocations only after a
//!   subgroup memory barrier on the host; one is inserted before an LDS read
//!   that may follow an LDS write without an intervening workgroup barrier.

use super::uniform::uniform_values;
use crate::cfg::dom;
use crate::ssa::build::block_successors;
use crate::ssa::*;
use crate::fxhash::{HashMap, HashSet};

fn def_dominates(func: &Function, idom: &[usize], position: &[(usize, usize)], def: Value, use_block: usize, use_index: usize) -> bool {
    let d = func.value(def);
    if matches!(d.op, Op::Const(_) | Op::Undef) {
        return true;
    }
    let (b, i) = position[def.index()];
    if b == usize::MAX {
        return false;
    }
    if b == use_block {
        return i < use_index;
    }
    dom::dominates(idom, b, use_block)
}

fn positions(func: &Function) -> Vec<(usize, usize)> {
    let mut p = vec![(usize::MAX, 0); func.values.len()];
    for &b in &func.order {
        for (i, &v) in func.block(b).insts.iter().enumerate() {
            p[v.index()] = (b.index(), i);
        }
    }
    p
}

/// Candidate values of lane `lane` of `v`; `None` when unknown.
fn lane_values(func: &Function, v: Value, lane: u64, uniform: &[bool], seen: &mut HashSet<Value>, out: &mut HashSet<Value>) -> bool {
    if !seen.insert(v) {
        return true;
    }
    if seen.len() > 4096 {
        return false;
    }
    let d = func.value(v);
    if uniform[v.index()] {
        out.insert(v);
        return true;
    }
    match d.op {
        Op::Select => {
            // v_writelane: select(lane_id == k, x, old)
            let c = func.value(d.args[0]);
            if c.op == Op::ICmp(ICmp::Eq) {
                let (l, r) = (c.args[0], c.args[1]);
                let (lane_side, k) = if func.op(l) == Op::LaneId { (true, func.as_const(r)) } else if func.op(r) == Op::LaneId { (true, func.as_const(l)) } else { (false, None) };
                if lane_side {
                    if let Some(k) = k {
                        return if k == lane {
                            lane_values(func, d.args[1], lane, uniform, seen, out)
                        } else {
                            lane_values(func, d.args[2], lane, uniform, seen, out)
                        };
                    }
                    return false;
                }
            }
            lane_values(func, d.args[1], lane, uniform, seen, out) && lane_values(func, d.args[2], lane, uniform, seen, out)
        }
        Op::Phi => d.args.iter().all(|&a| lane_values(func, a, lane, uniform, seen, out)),
        Op::Undef => true,
        _ => false,
    }
}

pub fn eliminate_read_lanes(func: &mut Function) -> usize {
    let uniform = uniform_values(func);
    let succs = block_successors(func);
    let idom = dom::dominators(&succs, func.entry.index());
    let pos = positions(func);
    let mut replace = HashMap::default();
    for &b in &func.order {
        for (i, &v) in func.block(b).insts.iter().enumerate() {
            if func.op(v) != Op::ReadLane {
                continue;
            }
            let (src, lane) = (func.args(v)[0], func.args(v)[1]);
            if uniform[src.index()] {
                replace.insert(v, src);
                continue;
            }
            let Some(k) = func.as_const(lane) else { continue };
            let mut seen = HashSet::default();
            let mut out = HashSet::default();
            if !lane_values(func, src, k, &uniform, &mut seen, &mut out) {
                continue;
            }
            out.remove(&v);
            if out.len() == 1 {
                let x = *out.iter().next().unwrap();
                if def_dominates(func, &idom, &pos, x, b.index(), i) && !depends_on(func, x, v) {
                    replace.insert(v, x);
                }
            }
        }
    }
    let n = replace.len();
    func.replace_uses(&replace);
    n
}

fn depends_on(func: &Function, v: Value, target: Value) -> bool {
    let mut stack = vec![v];
    let mut seen = HashSet::default();
    while let Some(x) = stack.pop() {
        if x == target {
            return true;
        }
        if !seen.insert(x) || seen.len() > 65536 {
            continue;
        }
        stack.extend(func.args(x).iter().copied());
    }
    false
}

/// Does `cond` (true) imply `mask` (true) in every invocation? Follows
/// And chains and phis outside loops (an exec saved and restored around a
/// kill or a branch merges as a phi: it implies the mask when every
/// incoming value does), up to 64 levels, memoised per walk; a cycle counts
/// as not implied.
struct Implications<'a> {
    func: &'a Function,
    block_of: &'a [u32],
    once: &'a [bool],
    memo: HashMap<(Value, Value), bool>,
}

impl Implications<'_> {
    fn implies(&mut self, cond: Value, mask: Value) -> bool {
        self.implies_at(cond, mask, 0)
    }

    fn implies_at(&mut self, cond: Value, mask: Value, depth: u32) -> bool {
        let func = self.func;
        if cond == mask || func.as_const(mask) == Some(1) {
            return true;
        }
        if func.as_const(cond) == Some(0) {
            return true;
        }
        if depth > 64 {
            return false;
        }
        if let Some(&known) = self.memo.get(&(cond, mask)) {
            return known;
        }
        let d = func.value(cond);
        let result = if d.op == Op::And && d.ty == Type::BOOL {
            let (x, y) = (d.args[0], d.args[1]);
            self.implies_at(x, mask, depth + 1) || self.implies_at(y, mask, depth + 1)
        } else if d.op == Op::Phi && d.ty == Type::BOOL && self.outside_loops(cond) {
            // Not implied while its own incoming values are checked.
            self.memo.insert((cond, mask), false);
            let args = d.args.clone();
            args.iter().all(|&x| (func.as_const(x).is_some() || self.outside_loops(x)) && self.implies_at(x, mask, depth + 1))
        } else {
            false
        };
        self.memo.insert((cond, mask), result);
        result
    }

    fn outside_loops(&self, v: Value) -> bool {
        let b = self.block_of[v.index()];
        b != u32::MAX && self.once[b as usize]
    }
}

/// Pure per-invocation computation (result in an invocation depends only on
/// that invocation's operands).
fn lane_local(op: Op) -> bool {
    op.is_pure() && !matches!(op, Op::Ballot | Op::ReadLane | Op::Shuffle | Op::Phi)
}

/// Users of every value by value index, as (user, operand index), in one
/// flat array (`start[v]..start[v + 1]`).
struct Users {
    start: Vec<u32>,
    list: Vec<(Value, u32)>,
}

impl Users {
    fn new(func: &Function) -> Users {
        let n = func.values.len();
        let mut count = vec![0u32; n + 1];
        for &b in &func.order {
            for &v in &func.block(b).insts {
                for &a in func.args(v) {
                    count[a.index() + 1] += 1;
                }
            }
        }
        for k in 0..n {
            count[k + 1] += count[k];
        }
        let mut fill = count.clone();
        let mut list = vec![(Value(0), 0u32); count[n] as usize];
        for &b in &func.order {
            for &v in &func.block(b).insts {
                for (k, &a) in func.args(v).iter().enumerate() {
                    let slot = &mut fill[a.index()];
                    list[*slot as usize] = (v, k as u32);
                    *slot += 1;
                }
            }
        }
        Users { start: count, list }
    }

    fn of(&self, v: Value) -> &[(Value, u32)] {
        &self.list[self.start[v.index()] as usize..self.start[v.index() + 1] as usize]
    }
}

/// Visited marks of the forward walks (one generation per walk).
struct Marks {
    gen: u32,
    mark: Vec<u32>,
    stack: Vec<Value>,
    /// Values one walk may visit: twice the program's instructions, at
    /// least 4096 (a walk visits each value derived from its select once).
    limit: usize,
}

/// An image operation whose result in an invocation depends only on that
/// invocation's operands: a sample or gather without implicit derivatives
/// (explicit LOD, LOD zero or explicit derivatives in a pixel shader; any
/// elsewhere), a size query.
fn explicit_lod_sample(func: &Function, op: Op) -> bool {
    let Op::Image(a) = op else { return false };
    match a.op {
        ImageOp::Sample | ImageOp::Gather4 => func.stage != Stage::Pixel || a.flags.derivatives || a.flags.lod || a.flags.lod_zero,
        ImageOp::GetResinfo => true,
        _ => false,
    }
}

/// Whether an operation with an EXEC operand reads its other operands only
/// in the invocations that operand keeps. Not so for a pixel shader's
/// implicit-LOD sample, gather or LOD query (their quad's coordinates
/// decide the LOD, inactive invocations included) or a BVH intersection
/// (its fault record covers every invocation).
fn reads_only_where_active(func: &Function, op: Op) -> bool {
    if !op.has_exec_operand() || op.is_lds_read() {
        return false;
    }
    match op {
        Op::Image(a) => match a.op {
            ImageOp::Sample | ImageOp::Gather4 => explicit_lod_sample(func, op),
            ImageOp::GetLod | ImageOp::BvhIntersect { .. } => false,
            _ => true,
        },
        _ => true,
    }
}

/// Blocks reachable from the entry and outside every loop (a depth-first
/// walk: a back edge's loop body is its target and the blocks reaching the
/// source without passing the target). A value there is computed at most
/// once per invocation, so a use sees the same instance of a mask as its
/// select.
fn blocks_outside_loops(func: &Function) -> Vec<bool> {
    let n = func.blocks.len();
    let mut once = vec![false; n];
    let Some(&first) = func.order.first() else { return once };
    let succs = block_successors(func);
    // 0 unvisited, 1 on the walk, 2 done.
    let mut state = vec![0u8; n];
    let mut looping = vec![false; n];
    let mut stack: Vec<(usize, usize)> = vec![(first.index(), 0)];
    state[first.index()] = 1;
    while let Some(&mut (block, ref mut next)) = stack.last_mut() {
        if *next == succs[block].len() {
            state[block] = 2;
            stack.pop();
            continue;
        }
        let successor = succs[block][*next];
        *next += 1;
        match state[successor] {
            0 => {
                state[successor] = 1;
                stack.push((successor, 0));
            }
            1 => {
                let mut body = vec![false; n];
                body[successor] = true;
                let mut pending = vec![block];
                while let Some(member) = pending.pop() {
                    if std::mem::replace(&mut body[member], true) {
                        continue;
                    }
                    pending.extend(func.blocks[member].preds.iter().map(|p| p.index()));
                }
                for (l, b) in looping.iter_mut().zip(body) {
                    *l |= b;
                }
            }
            _ => {}
        }
    }
    for b in 0..n {
        once[b] = state[b] == 2 && !looping[b];
    }
    once
}

/// Is the value of `select` in invocations where `mask` is false never
/// observed? Every use must be outside loops ([`blocks_outside_loops`]).
/// `eliminated`: selects already replaced by their `new` operand (their
/// other operands are no longer used).
#[allow(clippy::too_many_arguments)]
fn unobserved(func: &Function, users: &Users, block_of: &[u32], once: &[bool], branch_conds: &[bool], eliminated: &[bool], marks: &mut Marks, select: Value, mask: Value) -> bool {
    let mut implications = Implications { func, block_of, once, memo: HashMap::default() };
    marks.gen += 1;
    let gen = marks.gen;
    marks.stack.clear();
    marks.stack.push(select);
    marks.mark[select.index()] = gen;
    let mut seen = 1usize;
    while let Some(v) = marks.stack.pop() {
        if branch_conds[v.index()] {
            return false;
        }
        for &(user, k) in users.of(v) {
            let k = k as usize;
            if eliminated[user.index()] && k != 1 {
                continue;
            }
            let b = block_of[user.index()];
            if b == u32::MAX || !once[b as usize] {
                return false;
            }
            let d = func.value(user);
            if d.op == Op::Select && k == 1 && implications.implies(d.args[0], mask) {
                continue;
            }
            if d.op == Op::And && d.ty == Type::BOOL && implications.implies(d.args[1 - k], mask) {
                continue;
            }
            if reads_only_where_active(func, d.op) && k + 1 < d.args.len() && implications.implies(*d.args.last().unwrap(), mask) {
                continue;
            }
            // Branches are uniform here (the divergent ones come later, from
            // exec runs): a phi outside loops takes each invocation's value
            // from the edge it came along.
            if !lane_local(d.op) && d.op != Op::Phi && !explicit_lod_sample(func, d.op) {
                return false;
            }
            if marks.mark[user.index()] != gen {
                marks.mark[user.index()] = gen;
                seen += 1;
                if seen > marks.limit {
                    return false;
                }
                marks.stack.push(user);
            }
        }
    }
    true
}

/// MaskedSelectEliminator: `select(mask, new, old)` becomes `new` when the
/// lanes where `mask` is false never observe the result. The selects are
/// decided last to first, each after the selects using it: an eliminated
/// select no longer uses its `old` operand, so a chain of selects goes in
/// one call.
pub fn eliminate_masked_selects(func: &mut Function) -> usize {
    let once = blocks_outside_loops(func);
    masked_selects_round(func, &once)
}

/// How an edge from a value to one of its users counts in [`unobserved`]'s
/// walk for `mask`.
enum Edge {
    /// Not followed (an eliminated select's other operands, or a use under a
    /// condition implying the mask).
    Skip,
    /// The value may be observed outside the mask.
    Fail,
    Follow,
}

/// The common inputs of the walks of one round.
struct Walk<'a> {
    func: &'a Function,
    users: &'a Users,
    block_of: &'a [u32],
    once: &'a [bool],
    branch_conds: &'a [bool],
}

/// The and / phi trees [`Implications::implies_at`] recurses through, and
/// how deep they go (a query deeper than its depth limit may answer
/// differently from the plain walk's: `exact` turns false).
struct ImplicationTrees<'a> {
    func: &'a Function,
    block_of: &'a [u32],
    once: &'a [bool],
    /// Per value: the longest chain of and / phi levels a query starting
    /// there can recurse through (u32::MAX: not computed yet).
    height: Vec<u32>,
    exact: bool,
}

impl ImplicationTrees<'_> {
    fn outside_loops(&self, v: Value) -> bool {
        let b = self.block_of[v.index()];
        b != u32::MAX && self.once[b as usize]
    }

    /// The operands [`Implications::implies_at`] may recurse into from `v`.
    fn children(&self, v: Value) -> &[Value] {
        let func = self.func;
        let d = func.value(v);
        if d.ty == Type::BOOL && (d.op == Op::And || (d.op == Op::Phi && self.outside_loops(v))) {
            &d.args
        } else {
            &[]
        }
    }

    fn height_of(&mut self, root: Value) -> u32 {
        const UNKNOWN: u32 = u32::MAX;
        const ON_PATH: u32 = u32::MAX - 1;
        if self.height[root.index()] < ON_PATH {
            return self.height[root.index()];
        }
        let mut stack = vec![(root, 0usize)];
        self.height[root.index()] = ON_PATH;
        while let Some(&mut (v, ref mut next)) = stack.last_mut() {
            let children = self.children(v);
            if *next < children.len() {
                let c = children[*next];
                *next += 1;
                match self.height[c.index()] {
                    UNKNOWN => {
                        self.height[c.index()] = ON_PATH;
                        stack.push((c, 0));
                    }
                    ON_PATH => {
                        // A cycle: no bound holds.
                        self.exact = false;
                        self.height[root.index()] = UNKNOWN;
                        for (x, _) in stack {
                            self.height[x.index()] = UNKNOWN;
                        }
                        return u32::MAX;
                    }
                    _ => {}
                }
                continue;
            }
            let h = children.iter().map(|c| self.height[c.index()] + 1).max().unwrap_or(0);
            self.height[v.index()] = h.min(ON_PATH - 1);
            stack.pop();
        }
        self.height[root.index()]
    }
}

/// Walks that no guard stops (every condition taken as not implying the
/// mask, except those every mask implies). When no guard of the program
/// can imply a select's mask, its plain walk is exactly this one, and this
/// one does not depend on the mask: each value is walked once per round
/// (the masks of the f32 denormal flushes are one per select).
struct Unguarded {
    /// Per value: 0 not walked, 1 on the walk's path, 2 unobserved, 3
    /// observed.
    state: Vec<u8>,
    /// The values walked (their `state` is set).
    walked: Vec<Value>,
    /// Per value: uses as a guard (a select's condition, an operand of a
    /// boolean and, the EXEC operand of an operation reading only where it
    /// is active).
    guards: Vec<u32>,
    /// Per value: 0 unknown, 1 not implied by every mask, 2 implied by
    /// every mask (a constant false in its and / phi tree), 3 on the path.
    always: Vec<u8>,
    exact: bool,
}

impl Unguarded {
    fn new(w: &Walk<'_>) -> Unguarded {
        let func = w.func;
        let n = func.values.len();
        let mut guards = vec![0u32; n];
        for &b in &func.order {
            for &u in &func.block(b).insts {
                let d = func.value(u);
                for (k, &a) in d.args.iter().enumerate() {
                    let guard = (d.op == Op::Select && k == 0)
                        || (d.op == Op::And && d.ty == Type::BOOL)
                        || (reads_only_where_active(func, d.op) && d.args.len() >= 2 && k + 1 == d.args.len());
                    guards[a.index()] += guard as u32;
                }
            }
        }
        Unguarded { state: vec![0; n], walked: Vec::new(), guards, always: vec![0; n], exact: true }
    }

    /// `select` was eliminated: its other operands lose an edge, so the
    /// memo of a walk through one of them may be stale (every memo is
    /// dropped then).
    fn eliminated(&mut self, func: &Function, select: Value) {
        let d = func.value(select);
        if [d.args[0], d.args[2]].iter().any(|a| self.state[a.index()] != 0) {
            for v in self.walked.drain(..) {
                self.state[v.index()] = 0;
            }
        }
    }

    /// Whether no guard of the program can imply `mask` (the condition of
    /// `select`): the values implying it (`mask` and the boolean ands and
    /// phis built from it) guard nothing but `select`.
    fn no_guard_implies(&self, w: &Walk<'_>, select: Value, mask: Value) -> bool {
        let func = w.func;
        if func.as_const(mask).is_some() || func.args(select)[0] != mask {
            return false;
        }
        let mut uses = 0u32;
        let mut seen: HashSet<Value> = HashSet::default();
        let mut stack = vec![mask];
        seen.insert(mask);
        while let Some(c) = stack.pop() {
            uses += self.guards[c.index()];
            if uses > 1 || seen.len() > 256 {
                return false;
            }
            for &(u, _) in w.users.of(c) {
                let d = func.value(u);
                if d.ty == Type::BOOL && matches!(d.op, Op::And | Op::Phi) && seen.insert(u) {
                    stack.push(u);
                }
            }
        }
        // The one use left is `select`'s own condition.
        uses == 1
    }

    /// Whether a condition is implied by every mask (a constant false
    /// reached through boolean ands and phis outside loops), as
    /// [`Implications::implies_at`] finds it without its depth limit.
    fn always(&mut self, imp: &mut ImplicationTrees<'_>, root: Value) -> bool {
        const NO: u8 = 1;
        const YES: u8 = 2;
        const ON_PATH: u8 = 3;
        if matches!(self.always[root.index()], NO | YES) {
            return self.always[root.index()] == YES;
        }
        let func = imp.func;
        let mut stack = vec![(root, 0usize)];
        self.always[root.index()] = ON_PATH;
        while let Some(&mut (v, ref mut next)) = stack.last_mut() {
            let children = imp.children(v);
            if *next < children.len() {
                let c = children[*next];
                *next += 1;
                match self.always[c.index()] {
                    0 => {
                        self.always[c.index()] = ON_PATH;
                        stack.push((c, 0));
                    }
                    ON_PATH => {
                        self.exact = false;
                        return false;
                    }
                    _ => {}
                }
                continue;
            }
            let d = func.value(v);
            let yes = if func.as_const(v) == Some(0) {
                true
            } else if d.op == Op::And && d.ty == Type::BOOL {
                children.iter().any(|c| self.always[c.index()] == YES)
            } else if !children.is_empty() {
                children.iter().all(|&c| (func.as_const(c).is_some() || imp.outside_loops(c)) && self.always[c.index()] == YES)
            } else {
                false
            };
            self.always[v.index()] = if yes { YES } else { NO };
            stack.pop();
        }
        let yes = self.always[root.index()] == YES;
        // Found deeper than the depth limit, the plain walk may miss it.
        if yes && imp.height_of(root) > 64 {
            self.exact = false;
        }
        yes
    }

    /// [`unobserved`]'s decision for the edge to `user` (operand `k`)
    /// with no mask: a guard skips only when every mask implies it.
    fn edge(&mut self, w: &Walk<'_>, imp: &mut ImplicationTrees<'_>, eliminated: &[bool], user: Value, k: usize) -> Edge {
        if eliminated[user.index()] && k != 1 {
            return Edge::Skip;
        }
        let b = w.block_of[user.index()];
        if b == u32::MAX || !w.once[b as usize] {
            return Edge::Fail;
        }
        let func = w.func;
        let d = func.value(user);
        if d.op == Op::Select && k == 1 && self.always(imp, d.args[0]) {
            return Edge::Skip;
        }
        if d.op == Op::And && d.ty == Type::BOOL && self.always(imp, d.args[1 - k]) {
            return Edge::Skip;
        }
        if reads_only_where_active(func, d.op) && k + 1 < d.args.len() && self.always(imp, *d.args.last().unwrap()) {
            return Edge::Skip;
        }
        if !lane_local(d.op) && d.op != Op::Phi && !explicit_lod_sample(func, d.op) {
            return Edge::Fail;
        }
        Edge::Follow
    }

    /// Whether the walk from `select` meets an observation (memoised per
    /// value).
    fn observed(&mut self, w: &Walk<'_>, imp: &mut ImplicationTrees<'_>, eliminated: &[bool], select: Value) -> bool {
        const ON_PATH: u8 = 1;
        const UNOBSERVED: u8 = 2;
        const OBSERVED: u8 = 3;
        match self.state[select.index()] {
            UNOBSERVED => return false,
            OBSERVED => return true,
            _ => {}
        }
        let mut stack: Vec<(Value, usize)> = vec![(select, 0)];
        self.state[select.index()] = ON_PATH;
        self.walked.push(select);
        'walk: while let Some(&(v, next)) = stack.last() {
            let observed = if next == 0 && w.branch_conds[v.index()] {
                true
            } else {
                let users = w.users.of(v);
                let mut i = next;
                let mut verdict = false;
                while i < users.len() {
                    let (user, k) = users[i];
                    i += 1;
                    match self.edge(w, imp, eliminated, user, k as usize) {
                        Edge::Skip => {}
                        Edge::Fail => {
                            verdict = true;
                            break;
                        }
                        Edge::Follow => match self.state[user.index()] {
                            UNOBSERVED => {}
                            OBSERVED => {
                                verdict = true;
                                break;
                            }
                            ON_PATH => {
                                self.exact = false;
                                return true;
                            }
                            _ => {
                                stack.last_mut().unwrap().1 = i;
                                self.state[user.index()] = ON_PATH;
                                self.walked.push(user);
                                stack.push((user, 0));
                                continue 'walk;
                            }
                        },
                    }
                }
                verdict
            };
            if observed {
                for (x, _) in stack {
                    self.state[x.index()] = OBSERVED;
                }
                return true;
            }
            self.state[v.index()] = UNOBSERVED;
            stack.pop();
        }
        false
    }
}

fn masked_selects_round(func: &mut Function, once: &[bool]) -> usize {
    let users = Users::new(func);
    let mut branch_conds = vec![false; func.values.len()];
    let mut block_of = vec![u32::MAX; func.values.len()];
    for &b in &func.order {
        if let Terminator::CondBranch { cond, .. } = func.block(b).term {
            branch_conds[cond.index()] = true;
        }
        for &v in &func.block(b).insts {
            block_of[v.index()] = b.index() as u32;
        }
    }
    let w = Walk { func, users: &users, block_of: &block_of, once, branch_conds: &branch_conds };
    let replace = match memoised_round(&w) {
        Some(r) => r,
        None => plain_round(&w),
    };
    let n = replace.len();
    func.replace_uses(&replace);
    n
}

/// The selects of a round to replace by their new value, decided last to
/// first (a select after the selects using it): a select whose mask no
/// guard implies with the memoised [`Unguarded`] walk, any other with its
/// plain walk ([`unobserved`]). `None` when the memo might not be exact
/// (a cycle, an implication found deeper than the depth limit).
fn memoised_round(w: &Walk<'_>) -> Option<HashMap<Value, Value>> {
    let func = w.func;
    let n = func.values.len();
    let mut imp = ImplicationTrees { func, block_of: w.block_of, once: w.once, height: vec![u32::MAX; n], exact: true };
    let mut unguarded = Unguarded::new(w);
    let mut marks = Marks { gen: 0, mark: vec![0; n], stack: Vec::new(), limit: (2 * func.instruction_count()).max(4096) };
    let mut replace = HashMap::default();
    let mut eliminated = vec![false; n];
    for &b in func.order.iter().rev() {
        let single = w.once[b.index()];
        for &v in func.block(b).insts.iter().rev() {
            let d = func.value(v);
            if d.op != Op::Select || d.ty == Type::BOOL {
                continue;
            }
            let mask = d.args[0];
            let gone = func.as_const(mask) == Some(1)
                || (single
                    && if unguarded.no_guard_implies(w, v, mask) {
                        !unguarded.observed(w, &mut imp, &eliminated, v)
                    } else {
                        unobserved(func, w.users, w.block_of, w.once, w.branch_conds, &eliminated, &mut marks, v, mask)
                    });
            if gone {
                replace.insert(v, d.args[1]);
                eliminated[v.index()] = true;
                unguarded.eliminated(func, v);
            }
            if !imp.exact || !unguarded.exact {
                return None;
            }
        }
    }
    Some(replace)
}

/// [`memoised_round`] with one plain walk per select ([`unobserved`]).
fn plain_round(w: &Walk<'_>) -> HashMap<Value, Value> {
    let func = w.func;
    let mut marks = Marks { gen: 0, mark: vec![0; func.values.len()], stack: Vec::new(), limit: (2 * func.instruction_count()).max(4096) };
    let mut replace = HashMap::default();
    let mut eliminated = vec![false; func.values.len()];
    // Last to first: a select is decided after the selects using it.
    for &b in func.order.iter().rev() {
        let single = w.once[b.index()];
        for &v in func.block(b).insts.iter().rev() {
            let d = func.value(v);
            if d.op != Op::Select || d.ty == Type::BOOL {
                continue;
            }
            let mask = d.args[0];
            if func.as_const(mask) == Some(1) || (single && unobserved(func, w.users, w.block_of, w.once, w.branch_conds, &eliminated, &mut marks, v, mask)) {
                replace.insert(v, d.args[1]);
                eliminated[v.index()] = true;
            }
        }
    }
    replace
}

/// Insert subgroup LDS barriers before LDS reads that may follow an LDS
/// write (forward dataflow over the CFG).
pub fn insert_lds_barriers(func: &mut Function) -> usize {
    let succs = block_successors(func);
    let preds = func.compute_preds();
    let n = func.blocks.len();
    let mut dirty_out = vec![false; n];
    let transfer = |func: &Function, b: usize, mut dirty: bool| {
        for &v in &func.blocks[b].insts {
            let op = func.op(v);
            if op == Op::Barrier || op == Op::SubgroupLdsBarrier {
                dirty = false;
            } else if op.is_lds_read() && dirty {
                dirty = false; // a barrier will be inserted here
                if op.is_lds_write() {
                    dirty = true;
                }
            } else if op.is_lds_write() {
                dirty = true;
            }
        }
        dirty
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &func.order {
            let input = preds[b.index()].iter().any(|p| dirty_out[p.index()]);
            let out = transfer(func, b.index(), input);
            if out != dirty_out[b.index()] {
                dirty_out[b.index()] = out;
                changed = true;
            }
        }
    }
    let _ = succs;
    let mut inserted = 0;
    for &b in &func.order.clone() {
        let mut dirty = preds[b.index()].iter().any(|p| dirty_out[p.index()]);
        let insts = func.block(b).insts.clone();
        let mut out = Vec::with_capacity(insts.len());
        for v in insts {
            let op = func.op(v);
            if op == Op::Barrier || op == Op::SubgroupLdsBarrier {
                dirty = false;
            } else if op.is_lds_read() && dirty {
                let bar = func.new_value(Op::SubgroupLdsBarrier, Type::VOID, vec![]);
                func.value_mut(bar).block = Some(b);
                out.push(bar);
                inserted += 1;
                dirty = op.is_lds_write();
            } else if op.is_lds_write() {
                dirty = true;
            }
            out.push(v);
        }
        func.block_mut(b).insts = out;
    }
    inserted
}

// ---------------------------------------------------------------------------
// Lane-mask projection
// ---------------------------------------------------------------------------

/// LaneMaskProjection (wave32): this invocation's bit of a lane mask built
/// from ballots by scalar mask arithmetic (`s_and/or/not`, `s_cselect`,
/// loop-carried EXEC phis) is the same arithmetic on the ballots'
/// predicates: `inverse_ballot(and(ballot(a), ballot(b)))` becomes
/// `a && b`, so no subgroup operation is left. Every leaf must be a ballot
/// or the constant 0 / all ones, and at least one a ballot. Returns the
/// number of inverse ballots replaced.
pub fn project_lane_masks(func: &mut Function) -> usize {
    if func.wave_size != 32 {
        return 0;
    }
    let candidates: Vec<Value> = func
        .order
        .iter()
        .flat_map(|&b| func.block(b).insts.iter().copied())
        .filter(|&v| func.op(v) == Op::InverseBallot)
        .collect();
    let mut projected: HashMap<Value, Value> = HashMap::default();
    let mut replace: HashMap<Value, Value> = HashMap::default();
    for v in candidates {
        let mask = func.args(v)[0];
        let mut visited = HashSet::default();
        let mut grounded = false;
        if !can_project(func, mask, &projected, &mut visited, &mut grounded) || !grounded {
            continue;
        }
        let p = project(func, mask, &mut projected);
        replace.insert(v, p);
    }
    let n = replace.len();
    func.replace_uses(&replace);
    n
}

fn ballot_predicate(func: &Function, v: Value) -> Option<Value> {
    let d = func.value(v);
    if d.op != Op::Extract(0) {
        return None;
    }
    let src = d.args[0];
    (func.op(src) == Op::Ballot).then(|| func.args(src)[0])
}

fn can_project(func: &Function, v: Value, projected: &HashMap<Value, Value>, visited: &mut HashSet<Value>, grounded: &mut bool) -> bool {
    if let Some(c) = func.as_const(v) {
        let ok = c == 0 || c == 0xFFFF_FFFF;
        *grounded |= ok;
        return ok;
    }
    if projected.contains_key(&v) || ballot_predicate(func, v).is_some() {
        *grounded = true;
        return true;
    }
    if !visited.insert(v) {
        return true;
    }
    let d = func.value(v);
    if d.ty != Type::U32 || !matches!(d.op, Op::And | Op::Or | Op::Not | Op::Select | Op::Phi) {
        return false;
    }
    let first = if d.op == Op::Select { 1 } else { 0 };
    let args = d.args.clone();
    !args.is_empty() && args[first..].iter().all(|&a| can_project(func, a, projected, visited, grounded))
}

/// Insert `op` (bool) into the block of `after`, right after it (phis at
/// the block start).
fn insert_after(func: &mut Function, after: Value, op: Op, args: Vec<Value>) -> Value {
    let block = func.value(after).block.expect("placed value");
    let v = func.new_value(op, Type::BOOL, args);
    func.value_mut(v).block = Some(block);
    let insts = &mut func.block_mut(block).insts;
    let at = if op == Op::Phi { 0 } else { insts.iter().position(|&x| x == after).map_or(insts.len(), |i| i + 1) };
    insts.insert(at, v);
    v
}

fn project(func: &mut Function, v: Value, projected: &mut HashMap<Value, Value>) -> Value {
    if let Some(c) = func.as_const(v) {
        return func.const_bool(c != 0);
    }
    if let Some(&p) = projected.get(&v) {
        return p;
    }
    if let Some(p) = ballot_predicate(func, v) {
        return p;
    }
    let d = func.value(v).clone();
    let f = func.const_bool(false);
    let result = match d.op {
        Op::Phi => insert_after(func, v, Op::Phi, vec![f; d.args.len()]),
        Op::Select => insert_after(func, v, Op::Select, vec![d.args[0], f, f]),
        Op::Not => insert_after(func, v, Op::Not, vec![f]),
        op => insert_after(func, v, op, vec![f, f]),
    };
    projected.insert(v, result);
    let first = if d.op == Op::Select { 1 } else { 0 };
    for (i, &a) in d.args.iter().enumerate().skip(first) {
        let p = project(func, a, projected);
        func.value_mut(result).args[i] = p;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A function with blocks `b0..bn` laid out in order, `lane < limit`
    /// as the lane mask `m`, and user data 0..3 as plain values.
    struct Rig {
        f: Function,
        blocks: Vec<Block>,
        m: Value,
        data: Vec<Value>,
    }

    fn rig(blocks: usize) -> Rig {
        let mut f = Function::new(64, Stage::Compute, 0);
        let blocks: Vec<Block> = (0..blocks).map(|_| f.new_block()).collect();
        f.entry = blocks[0];
        f.order = blocks.clone();
        let b0 = blocks[0];
        let data = (0..4).map(|i| f.append(b0, Op::UserData(i), Type::U32, vec![])).collect::<Vec<_>>();
        let lane = f.append(b0, Op::LaneId, Type::U32, vec![]);
        let m = f.append(b0, Op::ICmp(ICmp::ULt), Type::BOOL, vec![lane, data[3]]);
        Rig { f, blocks, m, data }
    }

    fn store(f: &mut Function, b: Block, addr: Value, data: Value, exec: Value) -> Value {
        f.append(b, Op::LdsStore { access: Access::Dwords(1) }, Type::VOID, vec![addr, data, exec])
    }

    fn link(f: &mut Function, from: Block, term: Terminator) {
        for s in term.successors() {
            f.block_mut(s).preds.push(from);
        }
        f.block_mut(from).term = term;
    }

    #[test]
    fn a_select_feeding_an_eliminated_select_goes_in_the_same_call() {
        let Rig { mut f, blocks, m, data } = rig(1);
        let b = blocks[0];
        let (a, old, x) = (data[0], data[1], data[2]);
        // m2 = lane < a: not implied by m.
        let lane = f.append(b, Op::LaneId, Type::U32, vec![]);
        let m2 = f.append(b, Op::ICmp(ICmp::ULt), Type::BOOL, vec![lane, a]);
        let s1 = f.append(b, Op::Select, Type::U32, vec![m, a, old]);
        let s2 = f.append(b, Op::Select, Type::U32, vec![m2, x, s1]);
        let s3 = f.append(b, Op::IAdd, Type::U32, vec![s1, x]);
        let st2 = store(&mut f, b, a, s2, m2);
        let st3 = store(&mut f, b, a, s3, m);
        f.block_mut(b).term = Terminator::Return;
        // s2's invocations outside m2 are not stored; once s2 is gone, s1
        // only reaches a store under m.
        assert_eq!(eliminate_masked_selects(&mut f), 2);
        assert_eq!(f.args(st2)[1], x);
        assert_eq!(f.args(s3)[0], a);
        assert_eq!(f.args(st3)[1], s3);
    }

    #[test]
    fn a_merge_phi_outside_loops_passes_the_invocations_on() {
        let Rig { mut f, blocks, m, data } = rig(4);
        let (a, old, z) = (data[0], data[1], data[2]);
        let uniform = f.append(blocks[0], Op::ICmp(ICmp::Eq), Type::BOOL, vec![data[3], a]);
        f.block_mut(blocks[0]).merge = Merge::Selection { merge: blocks[3] };
        link(&mut f, blocks[0], Terminator::CondBranch { cond: uniform, then_block: blocks[1], else_block: blocks[2] });
        let s = f.append(blocks[1], Op::Select, Type::U32, vec![m, a, old]);
        link(&mut f, blocks[1], Terminator::Branch(blocks[3]));
        link(&mut f, blocks[2], Terminator::Branch(blocks[3]));
        let p = f.append(blocks[3], Op::Phi, Type::U32, vec![s, z]);
        let st = store(&mut f, blocks[3], a, p, m);
        f.block_mut(blocks[3]).term = Terminator::Return;
        assert_eq!(eliminate_masked_selects(&mut f), 1);
        assert_eq!(f.args(p), &[a, z]);
        assert_eq!(f.args(st)[1], p);
    }

    /// An exec saved and restored around a branch merges as a phi of the
    /// narrowed exec and the saved one: a use under it reads only lanes of
    /// the write's mask.
    #[test]
    fn an_exec_phi_whose_incoming_values_imply_the_mask_passes() {
        let Rig { mut f, blocks, m, data } = rig(4);
        let (a, old) = (data[0], data[1]);
        let s = f.append(blocks[0], Op::Select, Type::U32, vec![m, a, old]);
        let uniform = f.append(blocks[0], Op::ICmp(ICmp::Eq), Type::BOOL, vec![data[3], a]);
        f.block_mut(blocks[0]).merge = Merge::Selection { merge: blocks[3] };
        link(&mut f, blocks[0], Terminator::CondBranch { cond: uniform, then_block: blocks[1], else_block: blocks[2] });
        let lane = f.append(blocks[1], Op::LaneId, Type::U32, vec![]);
        let other = f.append(blocks[1], Op::ICmp(ICmp::Ne), Type::BOOL, vec![lane, a]);
        let narrowed = f.append(blocks[1], Op::And, Type::BOOL, vec![other, m]);
        link(&mut f, blocks[1], Terminator::Branch(blocks[3]));
        link(&mut f, blocks[2], Terminator::Branch(blocks[3]));
        let exec = f.append(blocks[3], Op::Phi, Type::BOOL, vec![narrowed, m]);
        let st = store(&mut f, blocks[3], a, s, exec);
        f.block_mut(blocks[3]).term = Terminator::Return;
        assert_eq!(eliminate_masked_selects(&mut f), 1);
        assert_eq!(f.args(st)[1], a);
    }

    /// A pixel shader's implicit-LOD sample takes its LOD from its quad,
    /// inactive invocations included: a select feeding its coordinates
    /// stays. With an explicit LOD it reads only its active invocations.
    #[test]
    fn an_implicit_lod_pixel_sample_keeps_the_select() {
        for (lod_zero, eliminated) in [(false, 0), (true, 1)] {
            let Rig { mut f, blocks, m, data } = rig(1);
            f.stage = Stage::Pixel;
            let b = blocks[0];
            let (a, old) = (data[0], data[1]);
            let s = f.append(b, Op::Select, Type::U32, vec![m, a, old]);
            let flags = SampleFlags { lod_zero, ..SampleFlags::default() };
            let attr = ImageAttr { pc: 0, op: ImageOp::Sample, flags, dim: 1, dmask: 1, a16: false, d16: false, glc: false, slc: false, address_dwords: 1, data_dwords: 0 };
            let sample = f.append(b, Op::Image(attr), Type::U32X4, vec![data[2], data[2], data[2], s, m]);
            let x = f.append(b, Op::Extract(0), Type::U32, vec![sample]);
            store(&mut f, b, a, x, m);
            f.block_mut(b).term = Terminator::Return;
            assert_eq!(eliminate_masked_selects(&mut f), eliminated, "lod_zero {lod_zero}");
        }
    }

    /// A walk may visit twice the program's instructions: a select whose
    /// derived values pass 4096 still goes when they all reach a use under
    /// its mask.
    #[test]
    fn the_walk_limit_scales_with_the_program() {
        let Rig { mut f, blocks, m, data } = rig(1);
        let b = blocks[0];
        let (a, old) = (data[0], data[1]);
        let s = f.append(b, Op::Select, Type::U32, vec![m, a, old]);
        let mut v = s;
        for _ in 0..5000 {
            v = f.append(b, Op::IAdd, Type::U32, vec![v, a]);
        }
        store(&mut f, b, a, v, m);
        f.block_mut(b).term = Terminator::Return;
        assert_eq!(eliminate_masked_selects(&mut f), 1);
    }

    #[test]
    fn a_select_in_a_loop_stays() {
        // b0 -> b1 (header) -> b2 (body, back to b1 or out) -> b3
        let Rig { mut f, blocks, m, data } = rig(4);
        let (a, old) = (data[0], data[1]);
        let stop = f.append(blocks[0], Op::ICmp(ICmp::Eq), Type::BOOL, vec![data[2], a]);
        link(&mut f, blocks[0], Terminator::Branch(blocks[1]));
        f.block_mut(blocks[1]).merge = Merge::Loop { merge: blocks[3], continue_block: blocks[2] };
        let p = f.append(blocks[1], Op::Phi, Type::U32, vec![old]);
        link(&mut f, blocks[1], Terminator::Branch(blocks[2]));
        let s = f.append(blocks[2], Op::Select, Type::U32, vec![m, a, p]);
        link(&mut f, blocks[2], Terminator::CondBranch { cond: stop, then_block: blocks[3], else_block: blocks[1] });
        f.value_mut(p).args.push(s);
        let st = store(&mut f, blocks[3], a, s, m);
        f.block_mut(blocks[3]).term = Terminator::Return;
        // Each iteration's lanes outside m keep the previous iteration's
        // value: the select stays.
        assert_eq!(eliminate_masked_selects(&mut f), 0);
        assert_eq!(f.args(st)[1], s);
        assert_eq!(blocks_outside_loops(&f), vec![true, false, false, true]);
    }

    /// A random diamond (b0 branches uniformly to b1 or b2, both join in
    /// b3 with phis) of selects, ands, adds, compares and masked stores.
    fn random_diamond(seed: &mut u64) -> Function {
        let mut rnd = |n: usize| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            (*seed % n as u64) as usize
        };
        let Rig { mut f, blocks, m, data } = rig(4);
        let uniform = f.append(blocks[0], Op::ICmp(ICmp::Eq), Type::BOOL, vec![data[3], data[0]]);
        f.block_mut(blocks[0]).merge = Merge::Selection { merge: blocks[3] };
        link(&mut f, blocks[0], Terminator::CondBranch { cond: uniform, then_block: blocks[1], else_block: blocks[2] });
        link(&mut f, blocks[1], Terminator::Branch(blocks[3]));
        link(&mut f, blocks[2], Terminator::Branch(blocks[3]));
        f.block_mut(blocks[3]).term = Terminator::Return;
        let lane = f.append(blocks[0], Op::LaneId, Type::U32, vec![]);
        let shared_u: Vec<Value> = data.clone();
        let shared_b: Vec<Value> = vec![m];
        let mut ends: Vec<(Vec<Value>, Vec<Value>)> = Vec::new();
        let grow = |f: &mut Function, b: Block, us: &mut Vec<Value>, bs: &mut Vec<Value>, count: usize, rnd: &mut dyn FnMut(usize) -> usize| {
            for _ in 0..count {
                match rnd(6) {
                    0 | 1 => {
                        let (c, x, y) = (bs[rnd(bs.len())], us[rnd(us.len())], us[rnd(us.len())]);
                        us.push(f.append(b, Op::Select, Type::U32, vec![c, x, y]));
                    }
                    2 => {
                        let (x, y) = (bs[rnd(bs.len())], bs[rnd(bs.len())]);
                        bs.push(f.append(b, Op::And, Type::BOOL, vec![x, y]));
                    }
                    3 => {
                        let x = us[rnd(us.len())];
                        bs.push(f.append(b, Op::ICmp(ICmp::ULt), Type::BOOL, vec![lane, x]));
                    }
                    4 => {
                        let (x, y) = (us[rnd(us.len())], us[rnd(us.len())]);
                        us.push(f.append(b, Op::IAdd, Type::U32, vec![x, y]));
                    }
                    _ => {
                        let (x, e) = (us[rnd(us.len())], bs[rnd(bs.len())]);
                        store(f, b, data[0], x, e);
                    }
                }
            }
        };
        let (mut us, mut bs) = (shared_u.clone(), shared_b.clone());
        let n0 = 4 + rnd(12);
        grow(&mut f, blocks[0], &mut us, &mut bs, n0, &mut rnd);
        for &b in &blocks[1..3] {
            let (mut u2, mut b2) = (us.clone(), bs.clone());
            let n = 2 + rnd(10);
            grow(&mut f, b, &mut u2, &mut b2, n, &mut rnd);
            ends.push((u2, b2));
        }
        let (mut u3, mut b3) = (us.clone(), bs.clone());
        for _ in 0..1 + rnd(3) {
            let (x, y) = (ends[0].0[rnd(ends[0].0.len())], ends[1].0[rnd(ends[1].0.len())]);
            u3.push(f.append(blocks[3], Op::Phi, Type::U32, vec![x, y]));
            let (x, y) = (ends[0].1[rnd(ends[0].1.len())], ends[1].1[rnd(ends[1].1.len())]);
            b3.push(f.append(blocks[3], Op::Phi, Type::BOOL, vec![x, y]));
        }
        let n3 = 2 + rnd(10);
        grow(&mut f, blocks[3], &mut u3, &mut b3, n3, &mut rnd);
        // Phis lead their block.
        let insts = std::mem::take(&mut f.block_mut(blocks[3]).insts);
        let (phis, rest): (Vec<Value>, Vec<Value>) = insts.into_iter().partition(|&v| f.op(v) == Op::Phi);
        f.block_mut(blocks[3]).insts = phis.into_iter().chain(rest).collect();
        let last = *u3.last().unwrap();
        store(&mut f, blocks[3], data[0], last, *b3.last().unwrap());
        f
    }

    /// The memoised walks decide every select as one plain walk per select
    /// does (random diamonds of selects, masks, phis and masked stores).
    #[test]
    fn memoised_walks_decide_as_the_plain_walks() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let (mut cases, mut eliminated) = (0, 0);
        for _ in 0..3000 {
            let f = random_diamond(&mut seed);
            crate::ssa::verify::verify(&f, true).unwrap();
            let (once, users, branch_conds, block_of) = round_inputs(&f);
            let w = Walk { func: &f, users: &users, block_of: &block_of, once: &once, branch_conds: &branch_conds };
            let memo = memoised_round(&w).expect("an exact memo");
            let plain = plain_round(&w);
            assert_eq!(memo, plain, "{f}");
            cases += 1;
            eliminated += plain.len();
        }
        assert!(cases == 3000 && eliminated > 1000, "{eliminated}");
    }

    /// A chain of `n` flush-style selects, each on its own condition, the
    /// chain's end stored under another mask: every select's walk reaches
    /// the store.
    fn flush_chain(n: usize) -> Function {
        let Rig { mut f, blocks, m, data } = rig(1);
        let b = blocks[0];
        let zero = f.const_u32(0);
        let mut v = data[0];
        for _ in 0..n {
            let small = f.append(b, Op::ICmp(ICmp::ULt), Type::BOOL, vec![v, data[1]]);
            let s = f.append(b, Op::Select, Type::U32, vec![small, zero, v]);
            v = f.append(b, Op::IAdd, Type::U32, vec![s, data[2]]);
        }
        store(&mut f, b, data[0], v, m);
        f.block_mut(b).term = Terminator::Return;
        f
    }

    fn round_inputs(f: &Function) -> (Vec<bool>, Users, Vec<bool>, Vec<u32>) {
        let once = blocks_outside_loops(f);
        let users = Users::new(f);
        let mut branch_conds = vec![false; f.values.len()];
        let mut block_of = vec![u32::MAX; f.values.len()];
        for &b in &f.order {
            if let Terminator::CondBranch { cond, .. } = f.block(b).term {
                branch_conds[cond.index()] = true;
            }
            for &v in &f.block(b).insts {
                block_of[v.index()] = b.index() as u32;
            }
        }
        (once, users, branch_conds, block_of)
    }

    /// Selects whose masks no guard implies (the f32 denormal flushes) are
    /// decided by one walk per value, not one per select: a chain of 40000
    /// takes milliseconds (minutes with a walk per select), and a shorter
    /// one is decided as the plain walks decide it.
    #[test]
    fn flush_selects_walk_each_value_once() {
        let f = flush_chain(2000);
        let (once, users, branch_conds, block_of) = round_inputs(&f);
        let w = Walk { func: &f, users: &users, block_of: &block_of, once: &once, branch_conds: &branch_conds };
        assert_eq!(memoised_round(&w).expect("an exact memo"), plain_round(&w));
        let mut f = flush_chain(40000);
        let t = std::time::Instant::now();
        assert_eq!(eliminate_masked_selects(&mut f), 0);
        let elapsed = t.elapsed();
        assert!(elapsed.as_secs() < 10, "{elapsed:?}");
    }
}
