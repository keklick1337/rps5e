// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! SSA construction: turns the register-level IR (GetReg / SetReg) into SSA
//! with phis (Cytron et al. phi placement on iterated dominance frontiers,
//! semi-pruned to registers live across blocks, then renaming over the
//! dominator tree). Registers read before any write become `undef`.

use super::func::*;
use super::op::*;
use crate::cfg::dom;
use crate::fxhash::{HashMap, HashSet};

/// Successor lists indexed by block index (only blocks in `order`).
pub fn block_successors(func: &Function) -> Vec<Vec<usize>> {
    let mut succs = vec![Vec::new(); func.blocks.len()];
    for &b in &func.order {
        succs[b.index()] = func.block(b).term.successors().iter().map(|s| s.index()).collect();
    }
    succs
}

/// Construct SSA form. `preds` must be up to date. Registers read before
/// any write are `undef`.
pub fn construct_ssa(func: &mut Function) {
    construct_ssa_with(func, &|f, r| f.undef(r.ty()));
}

/// Construct SSA form; `initial` gives the value of a register read before
/// any write.
pub fn construct_ssa_with(func: &mut Function, initial: &dyn Fn(&mut Function, Reg) -> Value) {
    let n = func.blocks.len();
    let succs = block_successors(func);
    let entry = func.entry.index();
    let idom = dom::dominators(&succs, entry);
    let df = dom::dominance_frontiers(&succs, &idom);
    let children = dom::dominator_children(&idom);

    // Global registers and definition sites.
    let mut globals: HashSet<Reg> = HashSet::default();
    let mut def_blocks: HashMap<Reg, Vec<usize>> = HashMap::default();
    for &b in &func.order {
        let mut killed: HashSet<Reg> = HashSet::default();
        for &v in &func.block(b).insts {
            match func.op(v) {
                Op::GetReg(r) if !killed.contains(&r) => {
                    globals.insert(r);
                }
                Op::SetReg(r) => {
                    killed.insert(r);
                    let list = def_blocks.entry(r).or_default();
                    if list.last() != Some(&b.index()) {
                        list.push(b.index());
                    }
                }
                _ => {}
            }
        }
    }

    // Phi placement.
    let mut phi_reg: HashMap<Value, Reg> = HashMap::default();
    let mut block_phis: Vec<Vec<Value>> = vec![Vec::new(); n];
    let mut regs: Vec<Reg> = globals.iter().copied().collect();
    regs.sort();
    for r in regs {
        let Some(defs) = def_blocks.get(&r) else { continue };
        let mut has_phi = vec![false; n];
        let mut work: Vec<usize> = defs.clone();
        let mut in_work = vec![false; n];
        for &d in defs {
            in_work[d] = true;
        }
        while let Some(x) = work.pop() {
            for &y in &df[x] {
                if has_phi[y] {
                    continue;
                }
                has_phi[y] = true;
                let npreds = func.blocks[y].preds.len();
                let undef = func.undef(r.ty());
                let phi = func.new_value(Op::Phi, r.ty(), vec![undef; npreds]);
                func.values[phi.index()].block = Some(Block(y as u32));
                phi_reg.insert(phi, r);
                block_phis[y].push(phi);
                if !in_work[y] {
                    in_work[y] = true;
                    work.push(y);
                }
            }
        }
    }
    for (b, phis) in block_phis.iter().enumerate() {
        if !phis.is_empty() {
            let mut insts = phis.clone();
            insts.extend(func.blocks[b].insts.iter().copied());
            func.blocks[b].insts = insts;
        }
    }

    // Renaming.
    let mut stacks: HashMap<Reg, Vec<Value>> = HashMap::default();
    let mut replace: HashMap<Value, Value> = HashMap::default();
    let mut dead = vec![false; func.values.len()];
    enum Visit {
        Enter(usize),
        Exit(Vec<Reg>),
    }
    let mut work = vec![Visit::Enter(entry)];
    while let Some(item) = work.pop() {
        match item {
            Visit::Exit(pushed) => {
                for r in pushed {
                    stacks.get_mut(&r).unwrap().pop();
                }
            }
            Visit::Enter(b) => {
                let mut pushed = Vec::new();
                let insts = func.blocks[b].insts.clone();
                for v in insts {
                    match func.op(v) {
                        Op::Phi => {
                            if let Some(&r) = phi_reg.get(&v) {
                                stacks.entry(r).or_default().push(v);
                                pushed.push(r);
                            }
                        }
                        Op::GetReg(r) => {
                            let cur = match stacks.get(&r).and_then(|s| s.last()) {
                                Some(&c) => c,
                                None => initial(func, r),
                            };
                            replace.insert(v, cur);
                            dead[v.index()] = true;
                        }
                        Op::SetReg(r) => {
                            let mut a = func.args(v)[0];
                            while let Some(&n) = replace.get(&a) {
                                a = n;
                            }
                            stacks.entry(r).or_default().push(a);
                            pushed.push(r);
                            dead[v.index()] = true;
                        }
                        _ => {}
                    }
                }
                for s in succs[b].clone() {
                    let pred_index = func.blocks[s].preds.iter().position(|p| p.index() == b).expect("pred list");
                    let phis = block_phis[s].clone();
                    for phi in phis {
                        let r = phi_reg[&phi];
                        let cur = match stacks.get(&r).and_then(|st| st.last()) {
                            Some(&c) => c,
                            None => initial(func, r),
                        };
                        func.values[phi.index()].args[pred_index] = cur;
                    }
                }
                work.push(Visit::Exit(pushed));
                for &c in children[b].iter().rev() {
                    work.push(Visit::Enter(c));
                }
            }
        }
    }
    dead.resize(func.values.len(), false);
    func.remove_values(&dead);
    func.replace_uses(&replace);
    remove_trivial_phis(func);
}

/// Replace phis whose operands (ignoring self references) are one value.
pub fn remove_trivial_phis(func: &mut Function) -> usize {
    let mut removed = 0;
    loop {
        let mut replace: HashMap<Value, Value> = HashMap::default();
        let mut dead = vec![false; func.values.len()];
        for &b in &func.order {
            for &v in &func.block(b).insts {
                if func.op(v) != Op::Phi {
                    break;
                }
                let mut same: Option<Value> = None;
                let mut trivial = true;
                for &a in func.args(v) {
                    let a = resolve(&replace, a);
                    if a == v || Some(a) == same {
                        continue;
                    }
                    if same.is_some() {
                        trivial = false;
                        break;
                    }
                    same = Some(a);
                }
                if trivial {
                    // A phi of only itself: replaced by undef below.
                    replace.insert(v, same.unwrap_or(v));
                    dead[v.index()] = true;
                }
            }
        }
        let self_only: Vec<Value> = replace.iter().filter(|(k, v)| k == v).map(|(k, _)| *k).collect();
        for v in self_only {
            let u = func.undef(func.ty(v));
            replace.insert(v, u);
        }
        if replace.is_empty() {
            return removed;
        }
        removed += replace.len();
        dead.resize(func.values.len(), false);
        func.remove_values(&dead);
        func.replace_uses(&replace);
    }
}

fn resolve(map: &HashMap<Value, Value>, mut v: Value) -> Value {
    while let Some(&n) = map.get(&v) {
        if n == v {
            break;
        }
        v = n;
    }
    v
}
