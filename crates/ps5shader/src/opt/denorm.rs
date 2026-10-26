// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Redundant f32 denormal flush elimination (the FlushElision pass).
//!
//! In the f32 flush modes every float operation flushes its denormal
//! inputs and results (`Translator::flush_denorm`): `select(|x| < 2^-126,
//! sign(x), x)` on the bits of `x` (the test on the magnitude bits or as a
//! float compare). A result flushed by one instruction is
//! flushed again as the input of the next, through the EXEC selects of its
//! register write and the phis of the register across blocks. Flushing a
//! value that can never be an f32 denormal returns it unchanged, so such a
//! flush is replaced by its input.
//!
//! "Never a denormal" (exponent field 0 with a non-zero mantissa) is the
//! greatest fixed point of these rules, on 32-bit values through bitcasts:
//! * a flush, a constant that is not a denormal, an integer converted to
//!   float (0 or at least 1 in magnitude), an f16 widened to f32, a float
//!   rounded to an integral value;
//! * `x & k` with `k` keeping the whole exponent field or clearing every
//!   bit but the sign; `x | k` with `k` setting an exponent bit (or only
//!   the sign); `x ^ 0x80000000`; `|x|`, `-x` — of a value that is never a
//!   denormal;
//! * a select, a phi, a min or a max whose every operand is never a
//!   denormal; a select arm taken only when that arm is a NaN
//!   (`select(isnan(y), y | quiet, r)`) counts as never a denormal.

use crate::ssa::*;
use crate::fxhash::HashMap;

const MAG: u64 = 0x7FFF_FFFF;
const SIGN: u64 = 0x8000_0000;
const EXP: u64 = 0x7F80_0000;
const SMALLEST_NORMAL: u64 = 0x0080_0000;

fn is32(f: &Function, v: Value) -> bool {
    f.ty(v).scalar().is_some_and(|s| s.bits() == 32 && s != Scalar::Bool) && !f.ty(v).is_vector()
}

/// `v` without 32-bit bitcasts (the same bits).
fn strip(f: &Function, mut v: Value) -> Value {
    while f.op(v) == Op::Bitcast && is32(f, v) && is32(f, f.args(v)[0]) {
        v = f.args(v)[0];
    }
    v
}

/// `a op b` with `k` constant in either order: `(other, k)`.
fn with_const(f: &Function, v: Value, op: Op) -> Option<(Value, u64)> {
    let d = f.value(v);
    if d.op != op || d.args.len() != 2 {
        return None;
    }
    match (f.as_const(d.args[0]), f.as_const(d.args[1])) {
        (_, Some(k)) => Some((d.args[0], k)),
        (Some(k), _) => Some((d.args[1], k)),
        _ => None,
    }
}

/// `x & k` with the bits of `base`.
fn is_and_of(f: &Function, v: Value, base: Value, k: u64) -> bool {
    with_const(f, strip(f, v), Op::And).is_some_and(|(x, c)| c == k && strip(f, x) == base)
}

/// The flushed value if `v` is `select(|x| < 2^-126, sign(x), x)` (or the
/// form that also tests `|x| != 0`).
pub fn flush_input(f: &Function, v: Value) -> Option<Value> {
    let d = f.value(v);
    if d.op != Op::Select || !is32(f, v) {
        return None;
    }
    let (c, s, x) = (d.args[0], d.args[1], strip(f, d.args[2]));
    if !is_and_of(f, s, x, SIGN) {
        return None;
    }
    let small = |c: Value| {
        let cd = f.value(c);
        match cd.op {
            Op::ICmp(ICmp::ULt) => f.as_const(cd.args[1]) == Some(SMALLEST_NORMAL) && is_and_of(f, cd.args[0], x, MAG),
            // |x| < 2^-126 as a float compare.
            Op::FCmp(FCmp::OLt) => {
                f.as_const(cd.args[1]) == Some(SMALLEST_NORMAL) && f.ty(cd.args[1]) == Type::F32 && {
                    let ad = f.value(cd.args[0]);
                    ad.op == Op::FAbs && strip(f, ad.args[0]) == x
                }
            }
            _ => false,
        }
    };
    let cd = f.value(c);
    let ok = small(c)
        || (cd.op == Op::And
            && f.ty(c).is_bool()
            && cd.args.iter().any(|&a| small(a))
            && cd.args.iter().any(|&a| {
                let ad = f.value(a);
                ad.op == Op::ICmp(ICmp::Ne) && f.as_const(ad.args[1]) == Some(0) && is_and_of(f, ad.args[0], x, MAG)
            }));
    ok.then_some(d.args[2])
}

/// Whether `arm` of a select on `c` is a NaN whenever it is selected:
/// `c` is `isnan(y)` and `arm` is `y` or `y | k`.
fn nan_arm(f: &Function, c: Value, arm: Value) -> bool {
    let cd = f.value(c);
    let y = match cd.op {
        Op::ICmp(ICmp::UGt) if f.as_const(cd.args[1]) == Some(EXP) => match with_const(f, strip(f, cd.args[0]), Op::And) {
            Some((y, MAG)) => strip(f, y),
            _ => return false,
        },
        Op::FCmp(FCmp::Uno) if cd.args[0] == cd.args[1] && is32(f, cd.args[0]) => strip(f, cd.args[0]),
        _ => return false,
    };
    let a = strip(f, arm);
    a == y || with_const(f, a, Op::Or).is_some_and(|(x, _)| strip(f, x) == y)
}

fn denormal_bits(k: u64) -> bool {
    let k = k & 0xFFFF_FFFF;
    k & EXP == 0 && k & 0x7F_FFFF != 0
}

/// Replace every flush of a value that is never an f32 denormal by the
/// value; returns the number of flushes removed.
pub fn elide_flushes(func: &mut Function) -> usize {
    let n = func.values.len();
    // Candidates: every 32-bit scalar placed in a block or constant.
    let mut nd = vec![false; n];
    let mut placed = Vec::new();
    for &b in &func.order {
        placed.extend(func.block(b).insts.iter().copied());
    }
    for (i, d) in func.values.iter().enumerate() {
        if let Op::Const(k) = d.op {
            nd[i] = d.ty.scalar().is_some_and(|s| s.bits() == 32) && !denormal_bits(k);
        }
    }
    for &v in &placed {
        nd[v.index()] = is32(func, v);
    }
    // Greatest fixed point: clear values whose rule fails until stable.
    let holds = |f: &Function, nd: &[bool], v: Value| -> bool {
        if flush_input(f, v).is_some() {
            return true;
        }
        let d = f.value(v);
        let a = |i: usize| nd[d.args[i].index()];
        match d.op {
            Op::Bitcast => is32(f, d.args[0]) && a(0),
            // The true arm is taken only when the condition holds.
            Op::Select => (a(1) || nan_arm(f, d.args[0], d.args[1])) && a(2),
            // A min / max returns one of its operands (or a NaN).
            Op::Phi | Op::FMin | Op::FMax => d.args.iter().all(|x| nd[x.index()]),
            Op::UToF(_) | Op::SToF(_) => true,
            Op::FConvert(_) => f.ty(d.args[0]).scalar() == Some(Scalar::F16),
            Op::FFloor | Op::FCeil | Op::FTrunc | Op::FRoundEven => true,
            Op::FAbs | Op::FNeg => a(0),
            Op::And => match with_const(f, v, Op::And) {
                Some((x, k)) => k & MAG == 0 || (k & EXP == EXP && nd[x.index()]),
                None => false,
            },
            Op::Or => match with_const(f, v, Op::Or) {
                Some((x, k)) => k & EXP != 0 || (k & 0x7F_FFFF == 0 && nd[x.index()]),
                None => false,
            },
            Op::Xor => with_const(f, v, Op::Xor).is_some_and(|(x, k)| k & 0x7FFF_FFFF == 0 && nd[x.index()]),
            _ => false,
        }
    };
    loop {
        let mut changed = false;
        for &v in &placed {
            if nd[v.index()] && !holds(func, &nd, v) {
                nd[v.index()] = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut map: HashMap<Value, Value> = HashMap::default();
    for &v in &placed {
        if let Some(x) = flush_input(func, v) {
            if nd[x.index()] && func.ty(x) == func.ty(v) {
                map.insert(v, x);
            }
        }
    }
    let removed = map.len();
    func.replace_uses(&map);
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flush(b: &mut Builder, x: Value) -> Value {
        let bits = b.bitcast(x, Type::U32);
        let mm = b.u32(0x7FFF_FFFF);
        let sm = b.u32(0x8000_0000);
        let mag = b.and(bits, mm);
        let sign = b.and(bits, sm);
        let lim = b.u32(0x0080_0000);
        let small = b.icmp(ICmp::ULt, mag, lim);
        let r = b.select(small, sign, bits);
        b.bitcast(r, Type::F32)
    }

    #[test]
    fn a_flushed_value_is_not_flushed_again() {
        let mut f = Function::new(32, Stage::Compute, 0);
        let b0 = f.new_block();
        f.entry = b0;
        f.order = vec![b0];
        let mut b = Builder::new(&mut f, b0);
        let x = b.emit(Op::UserData(0), Type::U32, vec![]);
        let x = b.bitcast(x, Type::F32);
        let once = flush(&mut b, x);
        let y = b.fmul(once, once);
        let twice = flush(&mut b, once);
        let again = flush(&mut b, y);
        let twice = b.bitcast(twice, Type::U32);
        let again = b.bitcast(again, Type::U32);
        b.emit(Op::SetReg(Reg::Vgpr(0)), Type::Void, vec![twice]);
        b.emit(Op::SetReg(Reg::Vgpr(1)), Type::Void, vec![again]);
        f.blocks[0].term = Terminator::Return;
        assert_eq!(elide_flushes(&mut f), 1, "the flush of a flushed value goes, the flush of a product stays");
    }

    /// `select(isnan(p), default, p)` keeps `p` when it is not a NaN: a
    /// product's NaN selection is not a flush.
    #[test]
    fn a_nan_selection_keeps_the_flush_of_its_other_arm() {
        let mut f = Function::new(32, Stage::Compute, 0);
        let b0 = f.new_block();
        f.entry = b0;
        f.order = vec![b0];
        let mut b = Builder::new(&mut f, b0);
        let x = b.emit(Op::UserData(0), Type::U32, vec![]);
        let x = b.bitcast(x, Type::F32);
        let once = flush(&mut b, x);
        let p = b.fmul(once, once);
        let bits = b.bitcast(p, Type::U32);
        let mm = b.u32(0x7FFF_FFFF);
        let mag = b.and(bits, mm);
        let inf = b.u32(0x7F80_0000);
        let nan = b.icmp(ICmp::UGt, mag, inf);
        let default = b.u32(0xFFC0_0000);
        let r = b.select(nan, default, bits);
        let r = b.bitcast(r, Type::F32);
        let flushed = flush(&mut b, r);
        let flushed = b.bitcast(flushed, Type::U32);
        b.emit(Op::SetReg(Reg::Vgpr(0)), Type::Void, vec![flushed]);
        // A NaN source taken when it is a NaN is never a denormal.
        let quiet = b.u32(0x0040_0000);
        let xb = b.bitcast(once, Type::U32);
        let xm = b.and(xb, mm);
        let xnan = b.icmp(ICmp::UGt, xm, inf);
        let q = b.or(xb, quiet);
        let pick = b.select(xnan, q, xb);
        let pick = b.bitcast(pick, Type::F32);
        let kept = flush(&mut b, pick);
        let kept = b.bitcast(kept, Type::U32);
        b.emit(Op::SetReg(Reg::Vgpr(1)), Type::Void, vec![kept]);
        f.blocks[0].term = Terminator::Return;
        assert_eq!(elide_flushes(&mut f), 1, "only the flush of the NaN-quieted flushed source goes");
    }

    /// The float-compare form of the flush is recognised too.
    #[test]
    fn the_float_compare_flush_is_elided_the_same() {
        let mut f = Function::new(32, Stage::Compute, 0);
        let b0 = f.new_block();
        f.entry = b0;
        f.order = vec![b0];
        let mut b = Builder::new(&mut f, b0);
        let fl = |b: &mut Builder, x: Value| {
            let xf = b.bitcast(x, Type::F32);
            let mag = b.fabs(xf);
            let lim = b.f32(f32::from_bits(0x0080_0000));
            let small = b.fcmp(FCmp::OLt, mag, lim);
            let bits = b.bitcast(x, Type::U32);
            let sm = b.u32(0x8000_0000);
            let sign = b.and(bits, sm);
            b.select(small, sign, bits)
        };
        let x = b.emit(Op::UserData(0), Type::U32, vec![]);
        let once = fl(&mut b, x);
        let twice = fl(&mut b, once);
        b.emit(Op::SetReg(Reg::Vgpr(0)), Type::Void, vec![twice]);
        let i = b.emit(Op::UserData(1), Type::U32, vec![]);
        let c = b.emit(Op::UToF(Rounding::Rte), Type::F32, vec![i]);
        let cf = fl(&mut b, c);
        b.emit(Op::SetReg(Reg::Vgpr(1)), Type::Void, vec![cf]);
        f.blocks[0].term = Terminator::Return;
        assert_eq!(elide_flushes(&mut f), 2);
    }
}
