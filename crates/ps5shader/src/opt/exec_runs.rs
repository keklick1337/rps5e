// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! ExecRunGrouping: a run of EXEC-masked vector register writes shares one
//! branch on the EXEC bit.
//!
//! A vector register write is `select(exec, new, old)`, so the old value of
//! every register the shader writes stays live across the write and every
//! write pays a select. A run of such writes, together with the pure
//! per-invocation arithmetic that only feeds their new values, can instead
//! run inside one `if (exec)`: inside the branch the selects are their new
//! values, and only the join needs the old ones (phis). Masked-off
//! invocations skip the whole run, which is what the hardware does with
//! them anyway.
//!
//! The pass works on optimized SSA and runs last: the branch it creates is
//! the only one that may diverge (every other branch is wave-uniform), so
//! the uniformity-based passes must not run after it. A run ends at
//! anything that is not per-invocation arithmetic: memory, side effects,
//! ballots, lane reads and shuffles (a lane read needs its neighbours to
//! have run), and phis. Loop headers and continue blocks keep their shape.

use crate::ssa::*;
use crate::fxhash::{HashMap, HashSet};

/// The default run length: shorter runs pay for the branch more often than
/// they save.
pub const DEFAULT_MINIMUM: usize = 6;

/// Pure per-invocation computation: its result in an invocation depends
/// only on that invocation's operands, so it may run under a divergent
/// branch.
fn lane_local(op: Op) -> bool {
    op.is_pure() && !op.is_subgroup()
}

/// The users of every value (by value index: `(user, operand index)`) and
/// the branch conditions, as the function was when the pass started.
struct Uses {
    /// `users[start[v]..start[v + 1]]` are the users of value `v`.
    start: Vec<u32>,
    users: Vec<(Value, u32)>,
    branch_conds: Vec<bool>,
}

impl Uses {
    fn of(func: &Function) -> Uses {
        let n = func.values.len();
        let mut count = vec![0u32; n + 1];
        let mut branch_conds = vec![false; n];
        for &b in &func.order {
            for &v in &func.block(b).insts {
                for &a in func.args(v) {
                    count[a.index()] += 1;
                }
            }
            if let Terminator::CondBranch { cond, .. } = func.block(b).term {
                branch_conds[cond.index()] = true;
            }
        }
        let mut start = vec![0u32; n + 1];
        for i in 0..n {
            start[i + 1] = start[i] + count[i];
        }
        let mut fill = start.clone();
        let mut users = vec![(Value(0), 0u32); start[n] as usize];
        for &b in &func.order {
            for &v in &func.block(b).insts {
                for (k, &a) in func.args(v).iter().enumerate() {
                    let at = &mut fill[a.index()];
                    users[*at as usize] = (v, k as u32);
                    *at += 1;
                }
            }
        }
        Uses { start, users, branch_conds }
    }

    fn of_value(&self, v: Value) -> &[(Value, u32)] {
        let i = v.index();
        if i + 1 >= self.start.len() {
            return &[];
        }
        &self.users[self.start[i] as usize..self.start[i + 1] as usize]
    }

    fn is_branch_cond(&self, v: Value) -> bool {
        self.branch_conds.get(v.index()).copied().unwrap_or(false)
    }
}

/// Positions of the instructions of the block being searched, by value
/// index (`NOT_HERE`: elsewhere).
struct Positions(Vec<u32>);

const NOT_HERE: u32 = u32::MAX;

impl Positions {
    fn at(&self, v: Value) -> u32 {
        self.0.get(v.index()).copied().unwrap_or(NOT_HERE)
    }

    fn set(&mut self, insts: &[Value]) {
        for (i, &v) in insts.iter().enumerate() {
            if v.index() >= self.0.len() {
                self.0.resize(v.index() + 1, NOT_HERE);
            }
            self.0[v.index()] = i as u32;
        }
    }

    fn clear(&mut self, insts: &[Value]) {
        for &v in insts {
            if let Some(p) = self.0.get_mut(v.index()) {
                *p = NOT_HERE;
            }
        }
    }
}

/// A run found in a block: instruction positions `start..end`, its
/// condition, the selects on it (`s`), the instructions moved under the
/// branch (`r`) and the ones hoisted before it (`p`), each in block order.
struct Run {
    start: usize,
    end: usize,
    cond: Value,
    s: Vec<Value>,
    r: Vec<Value>,
    p: Vec<Value>,
}

/// Classify `insts[start..end]` for condition `cond`. Returns the run, or
/// the position where it has to end (an instruction that would have to
/// follow the join). `pos`: the positions of `insts`.
fn classify(func: &Function, uses: &Uses, pos: &Positions, insts: &[Value], start: usize, end: usize, cond: Value) -> Result<Run, usize> {
    let window = &insts[start..end];
    let n = window.len();
    // Offset of `v` in the window.
    let slot = |v: Value| -> Option<usize> {
        let p = pos.at(v) as usize;
        (p >= start && p < end).then(|| p - start)
    };
    let is_s = |v: Value| func.op(v) == Op::Select && func.args(v)[0] == cond;
    let s_flags: Vec<bool> = window.iter().map(|&v| is_s(v)).collect();
    // Moved under the branch: every use is by another moved instruction or
    // is the new value of a select on `cond` in the run (greatest fixpoint).
    // A worklist: a value is checked again only when one of its users in
    // the window leaves `r` (`dependents`: user -> the values it uses).
    let mut r: Vec<bool> = s_flags.iter().map(|s| !s).collect();
    let mut dep_start = vec![0u32; n + 1];
    for (i, &v) in window.iter().enumerate() {
        if !r[i] {
            continue;
        }
        for &(u, _) in uses.of_value(v) {
            if let Some(j) = slot(u).filter(|&j| r[j]) {
                dep_start[j + 1] += 1;
            }
        }
    }
    for j in 0..n {
        dep_start[j + 1] += dep_start[j];
    }
    let mut fill = dep_start.clone();
    let mut dependents = vec![0u32; dep_start[n] as usize];
    for (i, &v) in window.iter().enumerate() {
        if !r[i] {
            continue;
        }
        for &(u, _) in uses.of_value(v) {
            if let Some(j) = slot(u).filter(|&j| r[j]) {
                dependents[fill[j] as usize] = i as u32;
                fill[j] += 1;
            }
        }
    }
    let keeps = |v: Value, r: &[bool]| {
        !uses.is_branch_cond(v)
            && uses.of_value(v).iter().all(|&(u, k)| match slot(u) {
                Some(j) => r[j] || (s_flags[j] && k == 1),
                None => false,
            })
    };
    let mut work: Vec<u32> = (0..n as u32).filter(|&i| r[i as usize]).collect();
    while let Some(i) = work.pop() {
        let i = i as usize;
        if !r[i] || keeps(window[i], &r) {
            continue;
        }
        r[i] = false;
        work.extend(dependents[dep_start[i] as usize..dep_start[i + 1] as usize].iter().copied().filter(|&d| r[d as usize]));
    }
    // Hoisted before the branch: must not depend on the selects (those
    // before it in the window).
    let mut run = Run { start, end, cond, s: Vec::new(), r: Vec::new(), p: Vec::new() };
    for (i, &v) in window.iter().enumerate() {
        if s_flags[i] {
            run.s.push(v);
        } else if r[i] {
            run.r.push(v);
        } else {
            if func.args(v).iter().any(|&a| slot(a).is_some_and(|j| j < i && s_flags[j])) {
                return Err(start + i);
            }
            run.p.push(v);
        }
    }
    Ok(run)
}

/// The most frequent non-constant select condition of `insts[start..end]`
/// with its count (first seen wins ties).
fn dominant_condition(func: &Function, insts: &[Value], start: usize, end: usize) -> Option<(Value, usize)> {
    // In order of first appearance (`slot`: condition -> its entry).
    let mut counts: Vec<(Value, usize)> = Vec::new();
    let mut slot: HashMap<Value, usize> = HashMap::default();
    for &v in &insts[start..end] {
        if func.op(v) != Op::Select {
            continue;
        }
        let c = func.args(v)[0];
        if func.as_const(c).is_some() {
            continue;
        }
        let i = *slot.entry(c).or_insert_with(|| {
            counts.push((c, 0));
            counts.len() - 1
        });
        counts[i].1 += 1;
    }
    let mut best: Option<(Value, usize)> = None;
    for (c, n) in counts {
        if best.is_none_or(|(_, m)| n > m) {
            best = Some((c, n));
        }
    }
    best
}

/// The first run of `block` with at least `minimum` selects.
fn find_run(func: &Function, uses: &Uses, pos: &mut Positions, block: Block, minimum: usize) -> Option<Run> {
    let insts = &func.block(block).insts;
    pos.set(insts);
    let run = find_run_in(func, uses, pos, insts, minimum);
    pos.clear(insts);
    run
}

/// [`find_run`] over `insts` (their positions in `pos`).
fn find_run_in(func: &Function, uses: &Uses, pos: &Positions, insts: &[Value], minimum: usize) -> Option<Run> {
    let mut a = insts.iter().position(|&v| func.op(v) != Op::Phi)?;
    while a < insts.len() {
        if !lane_local(func.op(insts[a])) {
            a += 1;
            continue;
        }
        let mut e = a;
        while e < insts.len() && lane_local(func.op(insts[e])) {
            e += 1;
        }
        // Candidate windows inside the maximal stretch a..e.
        let mut start = a;
        while start < e {
            let Some((cond, count)) = dominant_condition(func, insts, start, e) else { break };
            if count < minimum {
                break;
            }
            let mut end = e;
            loop {
                match classify(func, uses, pos, insts, start, end, cond) {
                    Ok(run) => {
                        if run.s.len() >= minimum && !run.r.is_empty() {
                            return Some(run);
                        }
                        break;
                    }
                    Err(stop) => end = stop,
                }
            }
            // `end` stops at an instruction that depends on a select of
            // the window: the next window starts there.
            if end == e {
                break;
            }
            start = end;
        }
        a = e;
    }
    None
}

/// Rewrite `block` around `run`: `block` keeps the instructions before the
/// run and the hoisted ones and branches on the condition; a new block runs
/// the moved instructions; a new join block turns the selects into phis
/// and takes the rest of `block` with its terminator and merge.
fn split(func: &mut Function, block: Block, run: Run) -> (Block, Block) {
    let insts = std::mem::take(&mut func.block_mut(block).insts);
    let then_block = func.new_block();
    let join = func.new_block();
    // Value of each select on either path.
    let s_set: HashSet<Value> = run.s.iter().copied().collect();
    let mut on_true: HashMap<Value, Value> = HashMap::default();
    let mut on_false: HashMap<Value, Value> = HashMap::default();
    for &s in &run.s {
        let (n, o) = (func.args(s)[1], func.args(s)[2]);
        let t = on_true.get(&n).copied().unwrap_or(n);
        let f = on_false.get(&o).copied().unwrap_or(o);
        on_true.insert(s, t);
        on_false.insert(s, f);
    }
    // Header: the code before the run, then the hoisted instructions.
    let mut head: Vec<Value> = insts[..run.start].to_vec();
    head.extend(run.p.iter().copied());
    // Branch body: the moved instructions read the selects' new values.
    for &v in &run.r {
        let args = func.args(v).to_vec();
        let args = args.into_iter().map(|a| if s_set.contains(&a) { on_true[&a] } else { a }).collect();
        func.value_mut(v).args = args;
        func.value_mut(v).block = Some(then_block);
    }
    // Join: one phi per select (preds: header, branch body), then the rest.
    let mut tail: Vec<Value> = Vec::with_capacity(run.s.len() + insts.len() - run.end);
    for &s in &run.s {
        let d = func.value_mut(s);
        d.op = Op::Phi;
        d.args = vec![on_false[&s], on_true[&s]];
        d.block = Some(join);
        tail.push(s);
    }
    for &v in &insts[run.end..] {
        func.value_mut(v).block = Some(join);
        tail.push(v);
    }
    let old_term = std::mem::replace(&mut func.block_mut(block).term, Terminator::CondBranch { cond: run.cond, then_block, else_block: join });
    let old_merge = std::mem::replace(&mut func.block_mut(block).merge, Merge::Selection { merge: join });
    for s in old_term.successors() {
        for p in func.block_mut(s).preds.iter_mut() {
            if *p == block {
                *p = join;
            }
        }
    }
    func.block_mut(block).insts = head;
    let tb = func.block_mut(then_block);
    tb.insts = run.r;
    tb.term = Terminator::Branch(join);
    tb.preds = vec![block];
    let jb = func.block_mut(join);
    jb.insts = tail;
    jb.term = old_term;
    jb.merge = old_merge;
    jb.preds = vec![block, then_block];
    (then_block, join)
}

/// Group runs of at least `minimum` selects on one condition under a
/// branch on it. Returns the number of runs grouped.
pub fn group_exec_runs(func: &mut Function, minimum: usize) -> usize {
    if minimum < 2 {
        return 0;
    }
    let mut skip: HashSet<Block> = HashSet::default();
    for &b in &func.order {
        if let Merge::Loop { continue_block, .. } = func.block(b).merge {
            skip.insert(b);
            skip.insert(continue_block);
        }
    }
    let uses = Uses::of(func);
    let mut pos = Positions(vec![NOT_HERE; func.values.len()]);
    let mut grouped = 0;
    let mut i = 0;
    while i < func.order.len() {
        let b = func.order[i];
        i += 1;
        if skip.contains(&b) {
            continue;
        }
        let Some(run) = find_run(func, &uses, &mut pos, b, minimum) else { continue };
        let (then_block, join) = split(func, b, run);
        // The join (with the rest of the block) is searched next; the branch
        // body only holds arithmetic.
        func.order.splice(i..i, [then_block, join]);
        skip.insert(then_block);
        grouped += 1;
    }
    grouped
}
