// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! IR validation: structure, types, SSA dominance.

use super::build::block_successors;
use super::func::*;
use super::op::*;
use super::types::*;
use crate::cfg::dom;

/// Check the function. `ssa` requires the absence of register operations.
pub fn verify(func: &Function, ssa: bool) -> Result<(), String> {
    let succs = block_successors(func);
    let idom = dom::dominators(&succs, func.entry.index());
    let mut position = vec![(usize::MAX, 0usize); func.values.len()];
    let mut in_order = vec![false; func.blocks.len()];
    for &b in &func.order {
        if in_order[b.index()] {
            return Err(format!("block b{} listed twice in order", b.0));
        }
        in_order[b.index()] = true;
        for (i, &v) in func.block(b).insts.iter().enumerate() {
            position[v.index()] = (b.index(), i);
            if func.value(v).block != Some(b) {
                return Err(format!("%{} placed in b{} but records {:?}", v.0, b.0, func.value(v).block));
            }
        }
    }
    if func.order.first() != Some(&func.entry) {
        return Err("order does not start with the entry block".into());
    }
    let preds = func.compute_preds();
    for &b in &func.order {
        let bd = func.block(b);
        let mut sorted_have = bd.preds.clone();
        sorted_have.sort();
        let mut sorted_want = preds[b.index()].clone();
        sorted_want.sort();
        if sorted_have != sorted_want {
            return Err(format!("b{} preds {:?} but terminators give {:?}", b.0, bd.preds, preds[b.index()]));
        }
        if bd.term == Terminator::None {
            return Err(format!("b{} is not terminated", b.0));
        }
        for s in bd.term.successors() {
            if !in_order[s.index()] {
                return Err(format!("b{} branches to b{} outside the order", b.0, s.0));
            }
        }
        match bd.merge {
            Merge::None => {}
            Merge::Selection { merge } => {
                if !in_order[merge.index()] {
                    return Err(format!("b{} selection merge b{} missing", b.0, merge.0));
                }
                if !matches!(bd.term, Terminator::CondBranch { .. }) {
                    return Err(format!("b{} selection header without a conditional branch", b.0));
                }
            }
            Merge::Loop { merge, continue_block } => {
                if !in_order[merge.index()] || !in_order[continue_block.index()] {
                    return Err(format!("b{} loop merge/continue missing", b.0));
                }
            }
        }
        let mut seen_non_phi = false;
        for (i, &v) in bd.insts.iter().enumerate() {
            let d = func.value(v);
            if d.op == Op::Phi {
                if seen_non_phi {
                    return Err(format!("phi %{} after non-phi in b{}", v.0, b.0));
                }
                if d.args.len() != bd.preds.len() {
                    return Err(format!("phi %{} has {} args for {} preds", v.0, d.args.len(), bd.preds.len()));
                }
            } else {
                seen_non_phi = true;
            }
            if ssa && matches!(d.op, Op::GetReg(_) | Op::SetReg(_)) {
                return Err(format!("register operation %{} in SSA form", v.0));
            }
            for (k, &a) in d.args.iter().enumerate() {
                if a.index() >= func.values.len() {
                    return Err(format!("%{} uses out-of-range value", v.0));
                }
                let ad = func.value(a);
                if matches!(ad.op, Op::Const(_) | Op::Undef) {
                    continue;
                }
                let Some(ab) = ad.block else {
                    return Err(format!("%{} uses removed value %{}", v.0, a.0));
                };
                let (pb, pi) = position[a.index()];
                if pb == usize::MAX || pb != ab.index() {
                    return Err(format!("%{} uses %{} not placed in order", v.0, a.0));
                }
                if d.op == Op::Phi {
                    let pred = bd.preds[k].index();
                    if !dom::dominates(&idom, pb, pred) {
                        return Err(format!("phi %{} arg %{} does not dominate pred b{}", v.0, a.0, pred));
                    }
                } else if pb == b.index() {
                    if pi >= i {
                        return Err(format!("%{} uses %{} before its definition", v.0, a.0));
                    }
                } else if !dom::dominates(&idom, pb, b.index()) {
                    return Err(format!("%{} in b{} uses %{} from non-dominating b{}", v.0, b.0, a.0, pb));
                }
            }
            check_types(func, v)?;
        }
        if let Terminator::CondBranch { cond, .. } = bd.term {
            if func.ty(cond) != Type::BOOL {
                return Err(format!("b{} branch condition is not bool", b.0));
            }
        }
    }
    Ok(())
}

fn check_types(func: &Function, v: Value) -> Result<(), String> {
    let d = func.value(v);
    let t = |i: usize| func.ty(d.args[i]);
    let err = |what: &str| Err(format!("%{} ({}): {what}", v.0, super::print::instruction(func, v)));
    let same_as_result = |i: usize| t(i) == d.ty;
    match d.op {
        Op::IAdd | Op::ISub | Op::IMul | Op::UMulHi | Op::SMulHi | Op::UDiv | Op::UMod | Op::SMin | Op::SMax
        | Op::UMin | Op::UMax => {
            if !d.ty.is_int() || !same_as_result(0) || !same_as_result(1) {
                return err("integer binary operand types");
            }
        }
        Op::And | Op::Or | Op::Xor => {
            if d.ty.is_float() || !same_as_result(0) || !same_as_result(1) {
                return err("bitwise operand types");
            }
        }
        Op::Shl | Op::LShr | Op::AShr => {
            if !d.ty.is_int() || !same_as_result(0) || !t(1).is_int() {
                return err("shift operand types");
            }
        }
        Op::FAdd | Op::FSub | Op::FMul | Op::FDiv | Op::FMin | Op::FMax => {
            if !d.ty.is_float() || !same_as_result(0) || !same_as_result(1) {
                return err("float binary operand types");
            }
        }
        Op::FFma => {
            if !d.ty.is_float() || !same_as_result(0) || !same_as_result(1) || !same_as_result(2) {
                return err("fma operand types");
            }
        }
        Op::Select => {
            if t(0) != Type::BOOL || !same_as_result(1) || !same_as_result(2) {
                return err("select operand types");
            }
        }
        Op::ICmp(_) => {
            if d.ty != Type::BOOL || t(0) != t(1) || !(t(0).is_int()) {
                return err("icmp operand types");
            }
        }
        Op::FCmp(_) => {
            if d.ty != Type::BOOL || t(0) != t(1) || !t(0).is_float() {
                return err("fcmp operand types");
            }
        }
        Op::Bitcast => {
            if t(0).bits() != d.ty.bits() {
                return err("bitcast width");
            }
        }
        Op::Extract(i) => {
            if i as u32 >= t(0).components() || t(0).component() != d.ty.component() {
                return err("extract");
            }
        }
        Op::Construct => {
            if d.args.len() as u32 != d.ty.components() || d.args.iter().any(|&a| func.ty(a) != Type::Scalar(d.ty.component())) {
                return err("construct");
            }
        }
        Op::Phi => {
            if d.args.iter().any(|&a| func.ty(a) != d.ty) {
                return err("phi operand types");
            }
        }
        _ => {}
    }
    if d.op.has_exec_operand() && func.ty(*d.args.last().unwrap()) != Type::BOOL {
        return err("exec operand is not bool");
    }
    Ok(())
}
