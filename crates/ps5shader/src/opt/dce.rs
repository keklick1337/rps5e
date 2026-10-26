// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Dead code elimination (the DeadCodeEliminator pass): removes pure
//! instructions and phis whose results never reach a side effect or a
//! branch condition, then trivial phis.

use crate::ssa::build::remove_trivial_phis;
use crate::ssa::*;

/// A conditional branch whose one arm is an empty block that only jumps to
/// the other arm, which has no phis, goes to the same place either way: its
/// condition is replaced by a constant (the edges and the structured merge
/// stay), so the values computed only for it die. Loop headers keep
/// theirs. Returns the branches changed.
fn drop_empty_detour_conditions(func: &mut Function) -> usize {
    let mut changed = 0;
    for i in 0..func.order.len() {
        let b = func.order[i];
        let Terminator::CondBranch { cond, then_block, else_block } = func.block(b).term else { continue };
        if func.as_const(cond).is_some() || matches!(func.block(b).merge, Merge::Loop { .. }) || then_block == else_block {
            continue;
        }
        for (empty, target, keep) in [(then_block, else_block, false), (else_block, then_block, true)] {
            let detour = func.block(empty);
            let joins = detour.insts.is_empty()
                && detour.term == Terminator::Branch(target)
                && !matches!(detour.merge, Merge::Loop { .. })
                && !matches!(func.block(target).merge, Merge::Loop { .. })
                && !func.block(target).insts.iter().any(|&v| func.op(v) == Op::Phi);
            if joins {
                // `keep`: the condition that takes the non-empty arm.
                let c = func.const_bool(keep);
                if let Terminator::CondBranch { cond, .. } = &mut func.block_mut(b).term {
                    *cond = c;
                }
                changed += 1;
                break;
            }
        }
    }
    changed
}

pub fn dce(func: &mut Function) -> usize {
    let mut removed = remove_trivial_phis(func);
    removed += drop_empty_detour_conditions(func);
    let mut live = vec![false; func.values.len()];
    let mut work: Vec<Value> = Vec::new();
    for &b in &func.order {
        let bd = func.block(b);
        for &v in &bd.insts {
            let op = func.op(v);
            if op.has_side_effects() || matches!(op, Op::GetReg(_)) {
                work.push(v);
            }
        }
        if let Terminator::CondBranch { cond, .. } = bd.term {
            work.push(cond);
        }
    }
    while let Some(v) = work.pop() {
        if live[v.index()] {
            continue;
        }
        live[v.index()] = true;
        for &a in &func.value(v).args {
            if !live[a.index()] {
                work.push(a);
            }
        }
    }
    let mut dead = vec![false; func.values.len()];
    for &b in &func.order {
        for &v in &func.block(b).insts {
            if !live[v.index()] {
                dead[v.index()] = true;
                removed += 1;
            }
        }
    }
    func.remove_values(&dead);
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// entry -> (detour | merge), detour -> merge; the condition is a
    /// compare of user data that nothing else uses. `mode`: 0 empty
    /// detour, 1 a store in the detour, 2 a phi in the merge, 3 the
    /// compared value also used in the merge.
    fn build(mode: u32, reverse: bool) -> (Function, Value, Block) {
        let mut f = Function::new(64, Stage::Compute, 0);
        let (entry, detour, merge) = (f.new_block(), f.new_block(), f.new_block());
        f.entry = entry;
        f.order = vec![entry, detour, merge];
        let x = f.append(entry, Op::UserData(0), Type::U32, vec![]);
        let zero = f.const_u32(0);
        let cond = f.append(entry, Op::ICmp(ICmp::Ne), Type::BOOL, vec![x, zero]);
        f.append(entry, Op::Barrier, Type::VOID, vec![]);
        let (then_block, else_block) = if reverse { (merge, detour) } else { (detour, merge) };
        f.block_mut(entry).term = Terminator::CondBranch { cond, then_block, else_block };
        f.block_mut(entry).merge = Merge::Selection { merge };
        f.block_mut(detour).term = Terminator::Branch(merge);
        f.block_mut(merge).term = Terminator::Return;
        f.block_mut(detour).preds = vec![entry];
        f.block_mut(merge).preds = vec![entry, detour];
        if mode == 1 {
            f.append(detour, Op::Barrier, Type::VOID, vec![]);
        }
        if mode == 2 {
            let (one, two) = (f.const_u32(1), f.const_u32(2));
            let phi = f.new_value(Op::Phi, Type::U32, vec![one, two]);
            f.block_mut(merge).insts.insert(0, phi);
            f.append(merge, Op::GdsStore { access: crate::ssa::Access::Dwords(1) }, Type::VOID, vec![zero, phi, cond]);
        }
        if mode == 3 {
            let t = f.const_bool(true);
            f.append(merge, Op::GdsStore { access: crate::ssa::Access::Dwords(1) }, Type::VOID, vec![zero, x, t]);
        }
        (f, x, entry)
    }

    /// Only an empty detour into a phi-free merge loses its condition; the
    /// compare feeding it dies, the edges stay, values used elsewhere and
    /// earlier side effects stay.
    #[test]
    fn empty_branch_detour_condition_dies() {
        for reverse in [false, true] {
            for mode in 0..4 {
                let (mut f, x, entry) = build(mode, reverse);
                dce(&mut f);
                let Terminator::CondBranch { cond, then_block, else_block } = f.block(entry).term else { panic!("edges changed") };
                assert_ne!(then_block, else_block);
                assert_eq!(f.as_const(cond).is_some(), mode != 1 && mode != 2, "mode {mode} reverse {reverse}");
                let has = |op: fn(Op) -> bool| f.block(entry).insts.iter().any(|&v| op(f.op(v)));
                assert_eq!(has(|o| matches!(o, Op::ICmp(_))), mode == 1 || mode == 2, "mode {mode}: the compare");
                assert_eq!(f.block(entry).insts.contains(&x), mode != 0, "mode {mode}: the compared value");
                assert!(has(|o| o == Op::Barrier), "the earlier side effect stays");
            }
        }
    }
}
