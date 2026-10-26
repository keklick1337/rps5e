// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! SplitDescriptorPhis: an image read whose descriptor is a phi of
//! different static descriptors (chosen by control flow) is duplicated once
//! per distinct incoming descriptor, each copy reading through a descriptor
//! rematerialized at the access, and the results are selected by an
//! edge-selector phi. Every copy then has a statically resolvable
//! descriptor, so the resource plan can bind each one; the reads have no
//! side effects, so executing every copy is harmless.

use crate::cfg::dom;
use crate::ssa::build::block_successors;
use crate::ssa::*;
use crate::fxhash::HashMap;

const DEPTH: u32 = 8;

fn splittable(op: Op) -> bool {
    matches!(op, Op::Image(a) if matches!(a.op, ImageOp::Sample | ImageOp::Gather4 | ImageOp::GetLod | ImageOp::GetResinfo | ImageOp::Load))
}

/// Ops a descriptor expression may be rebuilt from.
fn rebuildable(op: Op) -> bool {
    (op.is_pure() && !matches!(op, Op::LaneId | Op::InverseBallot | Op::Shuffle | Op::InterpParam { .. } | Op::SystemValue(_)))
        || matches!(op, Op::ScalarLoad { .. } | Op::ScalarBufferLoad { .. })
}

fn collect_phis(func: &Function, v: Value, depth: u32, out: &mut Vec<Value>) {
    let d = func.value(v);
    if d.op == Op::Phi {
        if !out.contains(&v) {
            out.push(v);
        }
        return;
    }
    if depth == 0 || !rebuildable(d.op) {
        return;
    }
    for &a in &d.args {
        collect_phis(func, a, depth - 1, out);
    }
}

struct Ctx<'a> {
    idom: &'a [usize],
    pos: Vec<(usize, usize)>,
    at_block: usize,
    at_index: usize,
}

impl Ctx<'_> {
    fn dominates_use(&self, func: &Function, v: Value) -> bool {
        if matches!(func.op(v), Op::Const(_) | Op::Undef) {
            return true;
        }
        let (b, i) = self.pos[v.index()];
        if b == usize::MAX {
            return false;
        }
        if b == self.at_block {
            i < self.at_index
        } else {
            dom::dominates(self.idom, b, self.at_block)
        }
    }
}

/// Plan of new instructions (op, type, args) emitted before the access;
/// returns the value (existing or index into `emitted` encoded as Err).
#[derive(Clone, Copy)]
enum Ref {
    Old(Value),
    New(usize),
}

struct Plan {
    emitted: Vec<(Op, Type, Vec<Ref>)>,
}

fn rematerialize(func: &Function, ctx: &Ctx, plan: &mut Plan, v: Value, depth: u32) -> Option<Ref> {
    if ctx.dominates_use(func, v) {
        return Some(Ref::Old(v));
    }
    let d = func.value(v);
    if depth == 0 || d.op == Op::Phi || !rebuildable(d.op) {
        return None;
    }
    let mut args = Vec::new();
    for &a in &d.args {
        args.push(rematerialize(func, ctx, plan, a, depth - 1)?);
    }
    plan.emitted.push((d.op, d.ty, args));
    Some(Ref::New(plan.emitted.len() - 1))
}

fn substitute(func: &Function, ctx: &Ctx, plan: &mut Plan, v: Value, phis: &[Value], pred_index: usize, depth: u32) -> Option<Ref> {
    let d = func.value(v);
    if let Some(_p) = phis.iter().find(|&&p| p == v) {
        let incoming = d.args[pred_index];
        return rematerialize(func, ctx, plan, incoming, DEPTH);
    }
    let mut inner = Vec::new();
    collect_phis(func, v, depth, &mut inner);
    if inner.iter().all(|p| !phis.contains(p)) {
        return Some(Ref::Old(v));
    }
    if depth == 0 || !rebuildable(d.op) {
        return None;
    }
    let mut args = Vec::new();
    for &a in &d.args {
        args.push(substitute(func, ctx, plan, a, phis, pred_index, depth - 1)?);
    }
    plan.emitted.push((d.op, d.ty, args));
    Some(Ref::New(plan.emitted.len() - 1))
}

pub fn split_descriptor_phis(func: &mut Function) -> usize {
    let mut split = 0;
    loop {
        let Some(target) = find_candidate(func) else { break };
        if !split_one(func, target) {
            // Mark as handled by moving on; avoid infinite retries.
            break;
        }
        split += 1;
        if split > 10_000 {
            break;
        }
    }
    split
}

fn descriptor_slots(func: &Function, v: Value) -> Vec<usize> {
    match func.op(v) {
        Op::Image(a) if matches!(a.op, ImageOp::Sample | ImageOp::Gather4 | ImageOp::GetLod) => vec![0, 1, 2],
        _ => vec![0, 1],
    }
}

fn find_candidate(func: &Function) -> Option<Value> {
    for &b in &func.order {
        for &v in &func.block(b).insts {
            if !splittable(func.op(v)) {
                continue;
            }
            let mut phis = Vec::new();
            for s in descriptor_slots(func, v) {
                collect_phis(func, func.args(v)[s], DEPTH, &mut phis);
            }
            if phis.is_empty() {
                continue;
            }
            if prepare(func, v).is_some() {
                return Some(v);
            }
        }
    }
    None
}

struct Prepared {
    phis: Vec<Value>,
    block: Block,
    arms: Vec<usize>,
    edge_arm: Vec<usize>,
}

fn prepare(func: &Function, v: Value) -> Option<Prepared> {
    let mut phis = Vec::new();
    for s in descriptor_slots(func, v) {
        collect_phis(func, func.args(v)[s], DEPTH, &mut phis);
    }
    let block = func.value(phis[0]).block?;
    let preds = &func.block(block).preds;
    if preds.len() < 2 || phis.iter().any(|&p| func.value(p).block != Some(block)) {
        return None;
    }
    let mut arms: Vec<usize> = Vec::new();
    let mut edge_arm = vec![0; preds.len()];
    for e in 0..preds.len() {
        let found = arms.iter().position(|&a| phis.iter().all(|&p| func.args(p)[e] == func.args(p)[a]));
        match found {
            Some(k) => edge_arm[e] = k,
            None => {
                edge_arm[e] = arms.len();
                arms.push(e);
            }
        }
    }
    if arms.len() < 2 {
        return None;
    }
    Some(Prepared { phis, block, arms, edge_arm })
}

fn split_one(func: &mut Function, v: Value) -> bool {
    let Some(p) = prepare(func, v) else { return false };
    let succs = block_successors(func);
    let idom = dom::dominators(&succs, func.entry.index());
    let mut pos = vec![(usize::MAX, 0); func.values.len()];
    for &b in &func.order {
        for (i, &x) in func.block(b).insts.iter().enumerate() {
            pos[x.index()] = (b.index(), i);
        }
    }
    let (at_block, at_index) = pos[v.index()];
    let ctx = Ctx { idom: &idom, pos, at_block, at_index };
    let slots = descriptor_slots(func, v);
    // Plan every arm first; give up if any arm cannot be rebuilt.
    let mut plans = Vec::new();
    for &arm_edge in &p.arms {
        let mut plan = Plan { emitted: Vec::new() };
        let mut slot_refs = Vec::new();
        for &s in &slots {
            let r = substitute(func, &ctx, &mut plan, func.args(v)[s], &p.phis, arm_edge, DEPTH);
            match r {
                Some(r) => slot_refs.push((s, r)),
                None => return false,
            }
        }
        plans.push((plan, slot_refs));
    }
    // Emit.
    let block = Block(at_block as u32);
    let original = func.value(v).clone();
    let mut new_insts: Vec<Value> = Vec::new();
    let mut copies = Vec::new();
    for (plan, slot_refs) in plans {
        let mut made: Vec<Value> = Vec::new();
        for (op, ty, args) in plan.emitted {
            let args: Vec<Value> = args
                .into_iter()
                .map(|r| match r {
                    Ref::Old(x) => x,
                    Ref::New(i) => made[i],
                })
                .collect();
            let x = func.new_value(op, ty, args);
            func.value_mut(x).block = Some(block);
            new_insts.push(x);
            made.push(x);
        }
        let mut args = original.args.clone();
        for (s, r) in slot_refs {
            args[s] = match r {
                Ref::Old(x) => x,
                Ref::New(i) => made[i],
            };
        }
        let copy = func.new_value(original.op, original.ty, args);
        func.value_mut(copy).block = Some(block);
        new_insts.push(copy);
        copies.push(copy);
    }
    // Edge selectors and result selects.
    let mut result = *copies.last().unwrap();
    if original.ty != Type::VOID {
        for arm in (0..copies.len() - 1).rev() {
            let npreds = func.block(p.block).preds.len();
            let mut args = Vec::with_capacity(npreds);
            for e in 0..npreds {
                args.push(func.const_bool(p.edge_arm[e] == arm));
            }
            let sel = func.new_value(Op::Phi, Type::BOOL, args);
            func.value_mut(sel).block = Some(p.block);
            func.block_mut(p.block).insts.insert(0, sel);
            let s = func.new_value(Op::Select, original.ty, vec![sel, copies[arm], result]);
            func.value_mut(s).block = Some(block);
            new_insts.push(s);
            result = s;
        }
    }
    // Splice before the original, then remove it.
    let insts = &mut func.block_mut(block).insts;
    let idx = insts.iter().position(|&x| x == v).unwrap();
    insts.splice(idx..idx + 1, new_insts);
    func.value_mut(v).block = None;
    let mut map = HashMap::default();
    map.insert(v, result);
    func.replace_uses(&map);
    true
}
