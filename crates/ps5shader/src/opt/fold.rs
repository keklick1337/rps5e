// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Constant folding and algebraic simplification (the ConstantFolder pass),
//! plus dominator-scoped common subexpression elimination.

use crate::interp::eval::{eval, EvalError};
use crate::ssa::build::block_successors;
use crate::ssa::*;
use crate::cfg::dom;
use crate::fxhash::HashMap;

/// What a simplification produced.
enum Simplified {
    /// Replace the value by another value.
    Value(Value),
    /// Replace by a scalar constant.
    Const(u64),
}

fn is_const(f: &Function, v: Value, c: u64) -> bool {
    f.as_const(v) == Some(c)
}

fn all_ones(f: &Function, v: Value) -> bool {
    let ty = f.ty(v);
    ty.scalar().is_some_and(|s| f.as_const(v) == Some(s.mask()))
}

/// `Select(c, 1, 0)` as u32 -> c.
fn bool_of_u32(f: &Function, v: Value) -> Option<(Value, bool)> {
    let d = f.value(v);
    if d.op == Op::Select && d.ty == Type::U32 {
        let (t, e) = (d.args[1], d.args[2]);
        if is_const(f, t, 1) && is_const(f, e, 0) {
            return Some((d.args[0], true));
        }
        if is_const(f, t, 0) && is_const(f, e, 1) {
            return Some((d.args[0], false));
        }
    }
    None
}

fn simplify(f: &Function, v: Value) -> Option<Simplified> {
    let d = f.value(v);
    let a = |i: usize| d.args[i];
    let c = |i: usize| f.as_const(d.args[i]);
    let ty = d.ty;
    // Full constant evaluation of scalar results.
    if d.op.is_pure() && !d.args.is_empty() && !ty.is_vector() && ty != Type::VOID {
        if !matches!(d.op, Op::LaneId | Op::Ballot | Op::InverseBallot | Op::ReadLane | Op::Shuffle | Op::UserData(_) | Op::SystemValue(_) | Op::CodeAddress | Op::InterpParam { .. }) {
            if d.args.iter().all(|&x| matches!(f.value(x).op, Op::Const(_))) {
                let consts: Vec<_> = d
                    .args
                    .iter()
                    .map(|&x| match f.value(x).op {
                        Op::Const(k) => (f.ty(x), [k, 0, 0, 0]),
                        _ => unreachable!(),
                    })
                    .collect();
                if let Ok(r) = eval(d.op, ty, &consts) {
                    return Some(Simplified::Const(r[0]));
                }
            }
        }
    }
    match d.op {
        Op::Select => {
            if let Some(k) = c(0) {
                return Some(Simplified::Value(if k != 0 { a(1) } else { a(2) }));
            }
            if a(1) == a(2) {
                return Some(Simplified::Value(a(1)));
            }
            if f.is_undef(a(2)) {
                return Some(Simplified::Value(a(1)));
            }
            if f.is_undef(a(1)) {
                return Some(Simplified::Value(a(2)));
            }
            if ty == Type::BOOL && is_const(f, a(1), 1) && is_const(f, a(2), 0) {
                return Some(Simplified::Value(a(0)));
            }
            // select(c, select(c, x, y), z) -> select(c, x, z)
            let inner_t = f.value(a(1));
            if inner_t.op == Op::Select && inner_t.args[0] == a(0) {
                // Handled by rewriting args below (not a value replacement).
            }
            None
        }
        Op::And | Op::Or => {
            let is_and = d.op == Op::And;
            if a(0) == a(1) {
                return Some(Simplified::Value(a(0)));
            }
            // x & ~x = 0, x | ~x = all ones (s_orn2_saveexec's full EXEC).
            let ones = ty.component().mask();
            for (x, y) in [(0, 1), (1, 0)] {
                let other = f.value(a(y));
                if other.op == Op::Not && other.args[0] == a(x) && !ty.is_vector() {
                    return Some(Simplified::Const(if is_and { 0 } else { ones }));
                }
            }
            for (x, y) in [(0, 1), (1, 0)] {
                if let Some(k) = c(x) {
                    let ones = ty.component().mask();
                    if is_and {
                        if k == 0 {
                            return Some(Simplified::Const(0));
                        }
                        if k == ones {
                            return Some(Simplified::Value(a(y)));
                        }
                    } else {
                        if k == 0 {
                            return Some(Simplified::Value(a(y)));
                        }
                        if k == ones {
                            return Some(Simplified::Const(ones));
                        }
                    }
                }
            }
            None
        }
        Op::Xor => {
            if a(0) == a(1) {
                return Some(Simplified::Const(0));
            }
            if c(1) == Some(0) {
                return Some(Simplified::Value(a(0)));
            }
            if c(0) == Some(0) {
                return Some(Simplified::Value(a(1)));
            }
            None
        }
        Op::Not => {
            let inner = f.value(a(0));
            if inner.op == Op::Not {
                return Some(Simplified::Value(inner.args[0]));
            }
            None
        }
        Op::IAdd => {
            if c(1) == Some(0) {
                return Some(Simplified::Value(a(0)));
            }
            if c(0) == Some(0) {
                return Some(Simplified::Value(a(1)));
            }
            None
        }
        Op::ISub | Op::Shl | Op::LShr | Op::AShr => {
            if c(1) == Some(0) {
                return Some(Simplified::Value(a(0)));
            }
            None
        }
        Op::IMul => {
            if c(1) == Some(1) {
                return Some(Simplified::Value(a(0)));
            }
            if c(0) == Some(1) {
                return Some(Simplified::Value(a(1)));
            }
            if c(0) == Some(0) || c(1) == Some(0) {
                return Some(Simplified::Const(0));
            }
            None
        }
        Op::UMin | Op::UMax | Op::SMin | Op::SMax => {
            if a(0) == a(1) {
                return Some(Simplified::Value(a(0)));
            }
            None
        }
        // x <= +inf and -inf <= x hold for every x unordered (NaN too).
        Op::FCmp(p @ (FCmp::ULe | FCmp::UGe)) if f.ty(a(0)) == Type::F32 => {
            const INF: u64 = 0x7F80_0000;
            const NEG_INF: u64 = 0xFF80_0000;
            let (upper, lower) = if p == FCmp::ULe { (c(1), c(0)) } else { (c(0), c(1)) };
            (upper == Some(INF) || lower == Some(NEG_INF)).then_some(Simplified::Const(1))
        }
        Op::Bitcast => {
            if f.ty(a(0)) == ty {
                return Some(Simplified::Value(a(0)));
            }
            let inner = f.value(a(0));
            if inner.op == Op::Bitcast && f.ty(inner.args[0]) == ty {
                return Some(Simplified::Value(inner.args[0]));
            }
            None
        }
        Op::UConvert | Op::SConvert => {
            if f.ty(a(0)) == ty {
                return Some(Simplified::Value(a(0)));
            }
            // zext(trunc) etc. are not identities; trunc(zext x) is.
            let inner = f.value(a(0));
            if matches!(inner.op, Op::UConvert | Op::SConvert) && f.ty(inner.args[0]) == ty && f.ty(a(0)).bits() > ty.bits() {
                return Some(Simplified::Value(inner.args[0]));
            }
            None
        }
        Op::FConvert(_) => {
            // widen(narrow-from-wide-to-this)? only widen of a narrow value
            // back: fconvert(fconvert(x: f16 -> f32) -> f16) == x.
            let inner = f.value(a(0));
            if matches!(inner.op, Op::FConvert(_)) && f.ty(inner.args[0]) == ty && f.ty(a(0)).bits() > ty.bits() {
                return Some(Simplified::Value(inner.args[0]));
            }
            None
        }
        Op::Extract(i) => {
            let inner = f.value(a(0));
            if inner.op == Op::Construct {
                return Some(Simplified::Value(inner.args[i as usize]));
            }
            if inner.op == Op::Insert(i) {
                return Some(Simplified::Value(inner.args[1]));
            }
            // extract(bitcast(construct(lo, hi): u32x2 -> u64 -> u32x2))
            if inner.op == Op::Bitcast {
                let src = inner.args[0];
                if f.ty(src) == ty_vector_of(ty, f.ty(a(0))) {
                    let s = f.value(src);
                    if s.op == Op::Construct {
                        return Some(Simplified::Value(s.args[i as usize]));
                    }
                }
            }
            None
        }
        Op::ICmp(p) => {
            // (select(c,1,0) ==/!= 0/1) -> c / !c
            for (x, y) in [(0, 1), (1, 0)] {
                if let (Some((cond, positive)), Some(k)) = (bool_of_u32(f, a(x)), c(y)) {
                    let truthy = match (p, k) {
                        (ICmp::Ne, 0) | (ICmp::Eq, 1) => positive,
                        (ICmp::Eq, 0) | (ICmp::Ne, 1) => !positive,
                        _ => continue,
                    };
                    if truthy {
                        return Some(Simplified::Value(cond));
                    }
                }
            }
            if a(0) == a(1) {
                return Some(Simplified::Const(matches!(p, ICmp::Eq | ICmp::ULe | ICmp::UGe | ICmp::SLe | ICmp::SGe) as u64));
            }
            None
        }
        Op::InverseBallot => {
            let (lo, hi) = (a(0), a(1));
            let wave32 = f.wave_size == 32;
            if (all_ones(f, lo) || is_const(f, lo, 0xFFFF_FFFF)) && (wave32 || is_const(f, hi, 0xFFFF_FFFF)) {
                return Some(Simplified::Const(1));
            }
            if is_const(f, lo, 0) && (wave32 || is_const(f, hi, 0)) {
                return Some(Simplified::Const(0));
            }
            // inverse_ballot(ballot(b)) -> b
            let dl = f.value(lo);
            if let Op::Extract(0) = dl.op {
                let ballot = dl.args[0];
                if f.op(ballot) == Op::Ballot {
                    let hi_ok = wave32 || {
                        let dh = f.value(hi);
                        dh.op == Op::Extract(1) && dh.args[0] == ballot
                    };
                    if hi_ok {
                        return Some(Simplified::Value(f.args(ballot)[0]));
                    }
                }
            }
            None
        }
        Op::ReadLane => {
            // Reading any lane of a value equal in every invocation.
            None
        }
        _ => None,
    }
}

/// Vector type a bitcast source must have for extract-through-bitcast.
fn ty_vector_of(component: Type, of: Type) -> Type {
    if of.is_vector() && Type::Scalar(of.component()) == component {
        of
    } else {
        Type::VOID
    }
}

/// One folding sweep in block order. Returns the number of changes.
pub fn fold(func: &mut Function) -> usize {
    let mut changes = 0;
    let mut replace: HashMap<Value, Value> = HashMap::default();
    let order = func.order.clone();
    for &b in &order {
        let insts = func.block(b).insts.clone();
        for v in insts {
            // Apply pending replacements to operands first.
            if !replace.is_empty() {
                for k in 0..func.value(v).args.len() {
                    let mut x = func.value(v).args[k];
                    let mut touched = false;
                    while let Some(&n) = replace.get(&x) {
                        x = n;
                        touched = true;
                    }
                    if touched {
                        func.value_mut(v).args[k] = x;
                    }
                }
            }
            if func.op(v) == Op::Phi {
                continue;
            }
            // select(c, select(c, x, y), z) -> select(c, x, z) and
            // select(c, x, select(c, y, z)) -> select(c, x, z).
            if func.op(v) == Op::Select {
                let a = [func.args(v)[0], func.args(v)[1], func.args(v)[2]];
                let t = func.value(a[1]);
                if t.op == Op::Select && t.args[0] == a[0] {
                    let x = t.args[1];
                    func.value_mut(v).args[1] = x;
                    changes += 1;
                }
                let e = func.value(a[2]);
                if e.op == Op::Select && e.args[0] == a[0] {
                    let x = e.args[2];
                    func.value_mut(v).args[2] = x;
                    changes += 1;
                }
            }
            // bitcast(select(c, x, y)) -> select(c, x', y') when both arms
            // are free in the target type: constants (same bits), undef or
            // bitcasts from it. The register-level select of a converted
            // value then needs no casts.
            if func.op(v) == Op::Bitcast {
                let ty = func.ty(v);
                let src = func.args(v)[0];
                if func.op(src) == Op::Select && !ty.is_vector() && !func.ty(src).is_vector() {
                    let sd = func.value(src).clone();
                    let free = |f: &Function, x: Value| -> Option<Value> {
                        let d = f.value(x);
                        match d.op {
                            Op::Bitcast if f.ty(d.args[0]) == ty => Some(d.args[0]),
                            _ => None,
                        }
                    };
                    let arm = |f: &mut Function, x: Value| -> Option<Value> {
                        if let Some(k) = f.as_const(x) {
                            return Some(f.constant(ty, k));
                        }
                        if f.is_undef(x) {
                            return Some(f.undef(ty));
                        }
                        free(f, x)
                    };
                    let (c, x, y) = (sd.args[0], sd.args[1], sd.args[2]);
                    let either_free = free(func, x).is_some() || free(func, y).is_some();
                    if either_free {
                        if let (Some(x), Some(y)) = (arm(func, x), arm(func, y)) {
                            let d = func.value_mut(v);
                            d.op = Op::Select;
                            d.args = vec![c, x, y];
                            changes += 1;
                        }
                    }
                }
            }
            match simplify(func, v) {
                Some(Simplified::Value(n)) => {
                    replace.insert(v, n);
                    changes += 1;
                }
                Some(Simplified::Const(k)) => {
                    let ty = func.ty(v);
                    let cst = func.constant(ty, k);
                    replace.insert(v, cst);
                    changes += 1;
                }
                None => {}
            }
        }
        if let Terminator::CondBranch { cond, then_block, else_block } = func.block(b).term.clone() {
            let mut c = cond;
            while let Some(&n) = replace.get(&c) {
                c = n;
            }
            func.block_mut(b).term = Terminator::CondBranch { cond: c, then_block, else_block };
        }
    }
    func.replace_uses(&replace);
    changes
}

/// Dominator-scoped value numbering of pure instructions.
pub fn cse(func: &mut Function) -> usize {
    let succs = block_successors(func);
    let idom = dom::dominators(&succs, func.entry.index());
    let children = dom::dominator_children(&idom);
    // Replacements by value index (Value(u32::MAX): none).
    const NONE: Value = Value(u32::MAX);
    let mut replace = vec![NONE; func.values.len()];
    let mut replaced = 0usize;
    // Scoped table: an entry lives while its block's dominator subtree is
    // walked (a dominated duplicate is replaced, never inserted).
    let mut table: HashMap<CseKey, Value> = HashMap::default();
    enum Visit {
        Enter(usize),
        Exit(Vec<CseKey>),
    }
    let mut stack = vec![Visit::Enter(func.entry.index())];
    while let Some(item) = stack.pop() {
        match item {
            Visit::Exit(keys) => {
                for k in keys {
                    table.remove(&k);
                }
            }
            Visit::Enter(b) => {
                let mut added = Vec::new();
                for &v in &func.blocks[b].insts {
                    let d = func.value(v);
                    if !d.op.is_pure() || d.op == Op::Phi || d.ty == Type::VOID {
                        continue;
                    }
                    let arg = |a: &Value| {
                        let r = replace[a.index()];
                        if r == NONE {
                            *a
                        } else {
                            r
                        }
                    };
                    let args = if d.args.len() <= 4 {
                        let mut small = [NONE; 4];
                        for (k, a) in d.args.iter().enumerate() {
                            small[k] = arg(a);
                        }
                        CseArgs::Small(d.args.len() as u8, small)
                    } else {
                        CseArgs::Big(d.args.iter().map(arg).collect())
                    };
                    let key = CseKey { op: d.op, ty: d.ty, args };
                    match table.entry(key) {
                        std::collections::hash_map::Entry::Occupied(o) => {
                            replace[v.index()] = *o.get();
                            replaced += 1;
                        }
                        std::collections::hash_map::Entry::Vacant(e) => {
                            added.push(e.key().clone());
                            e.insert(v);
                        }
                    }
                }
                stack.push(Visit::Exit(added));
                for &c in children[b].iter().rev() {
                    stack.push(Visit::Enter(c));
                }
            }
        }
    }
    if replaced == 0 {
        return 0;
    }
    let map: HashMap<Value, Value> = replace.iter().enumerate().filter(|(_, r)| **r != NONE).map(|(k, r)| (Value(k as u32), *r)).collect();
    func.replace_uses(&map);
    replaced
}

/// Operands of a CSE key (inline up to four).
#[derive(Clone, PartialEq, Eq, Hash)]
enum CseArgs {
    Small(u8, [Value; 4]),
    Big(Box<[Value]>),
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct CseKey {
    op: Op,
    ty: Type,
    args: CseArgs,
}

/// Fold scalar loads from the code object (constant tables addressed from
/// S_GETPC_B64) when the code bytes are available.
pub fn fold_code_loads(func: &mut Function, code: &dyn crate::program::CodeSource, code_size_hint: u64) -> usize {
    let mut replace = HashMap::default();
    for &b in &func.order.clone() {
        for &v in &func.block(b).insts.clone() {
            if let Op::ScalarLoad { .. } = func.op(v) {
                let addr = func.args(v)[0];
                if let Some(a) = func.as_const(addr) {
                    let base = func.code_address;
                    if a >= base && a < base.saturating_add(code_size_hint) {
                        if let Some(w) = code.read_u32(a) {
                            let k = func.const_u32(w);
                            replace.insert(v, k);
                        }
                    }
                }
            }
        }
    }
    let n = replace.len();
    func.replace_uses(&replace);
    n
}

/// Express every 64-bit constant in `[code_address, code_address + span)`
/// as `CodeAddress + offset`: after folding, the values a program derives
/// from S_GETPC_B64 that still reach run time (a table it indexes with a
/// value known only then) are relative to where the program is loaded, so
/// the function no longer names its load address. Returns the count.
/// (Whether anything else of the function depends on the address is not
/// decided here: compare two translations at different addresses.)
pub fn unfold_code_address(func: &mut Function, span: u64) -> usize {
    let base = func.code_address;
    // Small values are offsets and counts, not code addresses.
    if base < span {
        return 0;
    }
    let end = base.saturating_add(span);
    let mut used = std::collections::BTreeSet::new();
    for v in func.values.iter() {
        if v.block.is_none() && !matches!(v.op, Op::Phi) {
            continue;
        }
        for &a in &v.args {
            let d = &func.values[a.index()];
            if let (Op::Const(c), Type::U64) = (d.op, d.ty) {
                if c >= base && c < end {
                    used.insert(a);
                }
            }
        }
    }
    if used.is_empty() {
        return 0;
    }
    let entry = func.entry;
    let code = func.new_value(Op::CodeAddress, Type::U64, Vec::new());
    let mut at = vec![code];
    let mut replace = HashMap::default();
    for c in used {
        let off = func.as_const(c).expect("constant") - base;
        let k = func.const_u64(off);
        let v = func.new_value(Op::IAdd, Type::U64, vec![code, k]);
        at.push(v);
        replace.insert(c, v);
    }
    for &v in &at {
        func.values[v.index()].block = Some(entry);
    }
    let n = replace.len();
    func.replace_uses(&replace);
    let insts = &mut func.blocks[entry.index()].insts;
    insts.splice(0..0, at);
    n
}

#[allow(dead_code)]
fn unused(_: EvalError) {}

#[cfg(test)]
mod unfold_tests {
    use super::*;

    /// 64-bit constants inside the program's code become `CodeAddress +
    /// offset` (defined first in the entry block); others stay.
    #[test]
    fn code_address_constants_become_relative() {
        let base = 0x11_e2a3_d200u64;
        let mut f = Function::new(64, Stage::Compute, base);
        let entry = f.new_block();
        f.entry = entry;
        f.order = vec![entry];
        let inside = f.const_u64(base + 0x40);
        let outside = f.const_u64(0x1234);
        let small = f.const_u32(base as u32);
        let sum = f.append(entry, Op::IAdd, Type::U64, vec![inside, outside]);
        let other = f.append(entry, Op::IAdd, Type::U32, vec![small, small]);
        f.block_mut(entry).term = Terminator::Return;
        assert_eq!(unfold_code_address(&mut f, 1 << 20), 1);
        let insts = f.block(entry).insts.clone();
        let (code, rel) = (insts[0], insts[1]);
        assert_eq!(f.op(code), Op::CodeAddress);
        assert_eq!((f.op(rel), f.args(rel)[0]), (Op::IAdd, code));
        assert_eq!(f.as_const(f.args(rel)[1]), Some(0x40));
        assert_eq!(f.args(sum), &[rel, outside]);
        assert_eq!(f.args(other), &[small, small], "32-bit constants stay");
        // A program near address zero has none: small values are not code
        // addresses.
        let mut g = Function::new(64, Stage::Compute, 0x100);
        let e = g.new_block();
        g.entry = e;
        let c = g.const_u64(0x140);
        g.append(e, Op::IAdd, Type::U64, vec![c, c]);
        assert_eq!(unfold_code_address(&mut g, 1 << 20), 0);
    }
}
