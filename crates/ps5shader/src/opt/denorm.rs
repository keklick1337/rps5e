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
//!
//! On a host whose own f32 arithmetic flushes denormals ([`HostFloat`],
//! measured on the device, per group of operations) two more rules hold,
//! so the emulated flush is the host's own and goes:
//! * the result of a host add, sub, mul or div ([`HostFloat::arith`]), or
//!   fma, rcp, sqrt, inverse square root or log2 ([`HostFloat::fused`]) is
//!   never a denormal;
//! * a flushed value that only enters such an operation as an operand is
//!   replaced there by the unflushed value: the host flushes it to the
//!   same zero of its sign.
//! Except where the host compiler makes the operation a move: a multiply
//! by a constant ±1, an add or subtract of a constant ±0, a division by a
//! constant ±1 and an fma by ±1 with a ±0 addend return their other
//! operand (denormal or not, measured), and an operation on constants
//! alone is folded by the host compiler.

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

/// What the host does with f32 denormals in a module without an f32
/// denormal execution mode (measured on the device): which operations
/// flush denormal operands to zeros of their sign and never return a
/// denormal (the operations the host compiler makes moves aside, see the
/// module doc).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HostFloat {
    /// f32 add, sub, mul and div.
    pub arith: bool,
    /// f32 fma, rcp, sqrt, inverse square root and log2.
    pub fused: bool,
}

impl HostFloat {
    /// A host whose denormal behaviour is not relied on.
    pub const NONE: HostFloat = HostFloat { arith: false, fused: false };

    /// Whether any host rule applies.
    pub fn any(self) -> bool {
        self.arith || self.fused
    }
}

/// The bits of a 32-bit constant `v` (through bitcasts).
fn const32(f: &Function, v: Value) -> Option<u32> {
    f.as_const(strip(f, v)).filter(|_| is32(f, v)).map(|k| k as u32)
}

fn is_signed_zero(k: Option<u32>) -> bool {
    k.is_some_and(|k| k & 0x7FFF_FFFF == 0)
}

fn is_signed_one(k: Option<u32>) -> bool {
    k.is_some_and(|k| k & 0x7FFF_FFFF == 0x3F80_0000)
}

/// What a host f32 operation `v` is: `Some(None)` an operation that
/// flushes its operands and result, `Some(Some(x))` one the host compiler
/// makes a move of its operand `x`, `None` neither (another operation, or
/// one on constants alone).
fn host_operation(f: &Function, host: HostFloat, v: Value) -> Option<Option<Value>> {
    let d = f.value(v);
    if f.ty(v) != Type::F32 {
        return None;
    }
    let k = |i: usize| const32(f, d.args[i]);
    match d.op {
        Op::FAdd | Op::FSub | Op::FMul | Op::FDiv if host.arith => {}
        Op::FFma | Op::FRcp | Op::FSqrt | Op::FInvSqrt | Op::FLog2 if host.fused => {}
        _ => return None,
    }
    if d.args.iter().all(|&a| const32(f, a).is_some()) {
        return None;
    }
    let other = |i: usize| Some(Some(d.args[1 - i]));
    match d.op {
        Op::FMul => {
            if let Some(i) = (0..2).find(|&i| is_signed_one(k(i))) {
                return other(i);
            }
        }
        Op::FAdd | Op::FSub => {
            if let Some(i) = (0..2).find(|&i| is_signed_zero(k(i))) {
                return other(i);
            }
        }
        Op::FDiv if is_signed_one(k(1)) => return Some(Some(d.args[0])),
        Op::FFma if is_signed_zero(k(2)) => {
            if let Some(i) = (0..2).find(|&i| is_signed_one(k(i))) {
                return other(i);
            }
        }
        _ => {}
    }
    Some(None)
}

/// Whether a placed value or constant of `func` has an f16 type (the
/// emitter then declares the 16-bit float modes).
pub fn uses_f16(func: &Function) -> bool {
    let f16 = |t: Type| t.scalar() == Some(Scalar::F16);
    func.order.iter().any(|&b| func.block(b).insts.iter().any(|&v| f16(func.ty(v)) || func.args(v).iter().any(|&a| f16(func.ty(a)))))
}

/// Replace every flush of a value that is never an f32 denormal by the
/// value; returns the number of flushes removed.
pub fn elide_flushes(func: &mut Function) -> usize {
    elide_flushes_on(func, HostFloat::NONE)
}

/// Whether each value is never an f32 denormal (indexed by value), with
/// the rules of `host`: the greatest fixed point of the module doc's
/// rules. Values that are not 32-bit scalars are `false`.
pub fn never_denormal(func: &Function, host: HostFloat) -> Vec<bool> {
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
            _ if host.any() => match host_operation(f, host, v) {
                Some(None) => true,
                Some(Some(x)) => nd[x.index()],
                None => false,
            },
            _ => false,
        }
    };
    // A value that stops holding is re-checked in its users only.
    let mut users: Vec<Vec<Value>> = vec![Vec::new(); n];
    for &v in &placed {
        for &a in func.args(v) {
            users[a.index()].push(v);
        }
    }
    let mut work: Vec<Value> = placed.clone();
    while let Some(v) = work.pop() {
        if nd[v.index()] && !holds(func, &nd, v) {
            nd[v.index()] = false;
            work.extend(users[v.index()].iter().copied().filter(|u| nd[u.index()]));
        }
    }
    nd
}

/// [`elide_flushes`] with the rules of `host`: also the flushes of the
/// host operations' results, and the flushed operands of host operations
/// (replaced there by the unflushed value). Returns the number of flushes
/// and flushed operands removed.
pub fn elide_flushes_on(func: &mut Function, host: HostFloat) -> usize {
    let nd = never_denormal(func, host);
    let mut placed = Vec::new();
    for &b in &func.order {
        placed.extend(func.block(b).insts.iter().copied());
    }
    let mut map: HashMap<Value, Value> = HashMap::default();
    for &v in &placed {
        if let Some(x) = flush_input(func, v) {
            if nd[x.index()] && func.ty(x) == func.ty(v) {
                map.insert(v, x);
            }
        }
    }
    let mut removed = map.len();
    func.replace_uses(&map);
    if host.any() {
        removed += unflush_insensitive(func, host);
        removed += unflush_host_operands(func, host);
    }
    removed
}

/// Whether `icmp(p, w, k)` (`w_left`; else `icmp(p, k, w)`) gives one result
/// for every `w` in `[s, s | 0x7FFFFF]` for both signs `s`: a denormal and
/// the zero of its sign compare the same.
fn icmp_insensitive(p: ICmp, k: u32, w_left: bool) -> bool {
    use ICmp::*;
    let p = if w_left {
        p
    } else {
        match p {
            ULt => UGt,
            ULe => UGe,
            UGt => ULt,
            UGe => ULe,
            other => other,
        }
    };
    [0u32, 0x8000_0000].into_iter().all(|lo| {
        let hi = lo | 0x7F_FFFF;
        match p {
            ULt | UGe => k <= lo || k > hi,
            ULe | UGt => k < lo || k >= hi,
            Eq | Ne => k < lo || k > hi,
            _ => false,
        }
    })
}

/// Which 32-bit values may change between a denormal and the zero of its
/// sign without changing anything the program computes: every use flushes
/// it the same way (a host operation, a flush, a NaN / sign / range test,
/// a truncation, a conversion to an integer or to f16), or passes it on
/// unchanged up to such a change to values of that kind (selects, phis,
/// bitcasts, sign and magnitude bit operations, min / max, the host
/// compiler's moves). The greatest fixed point; a terminator condition,
/// a store or any other use is sensitive.
fn denormal_insensitive(func: &Function, host: HostFloat, placed: &[Value]) -> Vec<bool> {
    let n = func.values.len();
    let mut uses: Vec<Vec<(Value, usize)>> = vec![Vec::new(); n];
    let mut ok = vec![false; n];
    for &v in placed {
        ok[v.index()] = is32(func, v);
        for (i, &a) in func.args(v).iter().enumerate() {
            uses[a.index()].push((v, i));
        }
    }
    for &b in &func.order {
        if let Terminator::CondBranch { cond, .. } = func.block(b).term {
            ok[cond.index()] = false;
        }
    }
    // A use that absorbs the change (`Some(true)`), one that passes it on
    // to its result (`None`), or a sensitive one (`Some(false)`).
    let absorbs = |u: Value, i: usize| -> Option<bool> {
        let d = func.value(u);
        let other = |j: usize| func.as_const(strip(func, d.args[j])).map(|k| k as u32);
        match d.op {
            Op::Select if i == 0 => Some(false),
            Op::Select | Op::Phi | Op::FMin | Op::FMax | Op::FNeg | Op::FAbs => None,
            Op::Bitcast if is32(func, u) => None,
            Op::And if d.args.len() == 2 => other(1 - i).map(|_| None).unwrap_or(Some(false)),
            Op::Or | Op::Xor if d.args.len() == 2 => match other(1 - i) {
                Some(k) if k & EXP as u32 == 0 => None,
                _ => Some(false),
            },
            Op::ICmp(p) if d.args.len() == 2 => Some(other(1 - i).is_some_and(|k| icmp_insensitive(p, k, i == 0))),
            Op::FCmp(FCmp::Uno | FCmp::Ord) => Some(true),
            Op::FTrunc | Op::FRoundEven | Op::FToS | Op::FToU => Some(func.ty(d.args[0]) == Type::F32),
            Op::FConvert(_) => Some(func.ty(d.args[0]) == Type::F32 && func.ty(u).scalar() == Some(Scalar::F16)),
            _ => match host_operation(func, host, u) {
                Some(None) => Some(true),
                Some(Some(x)) if x == d.args[i] => None,
                _ => Some(false),
            },
        }
    };
    // A value that turns sensitive is re-checked in its operands only.
    let mut work: Vec<Value> = placed.to_vec();
    while let Some(v) = work.pop() {
        if ok[v.index()] && !uses[v.index()].iter().all(|&(u, i)| absorbs(u, i).unwrap_or_else(|| ok[u.index()])) {
            ok[v.index()] = false;
            work.extend(func.args(v).iter().copied().filter(|a| ok[a.index()]));
        }
    }
    ok
}

/// Flushes whose result is used only where a denormal and the zero of its
/// sign give the same results ([`denormal_insensitive`]) are replaced by
/// their unflushed value.
fn unflush_insensitive(func: &mut Function, host: HostFloat) -> usize {
    let mut placed = Vec::new();
    for &b in &func.order {
        placed.extend(func.block(b).insts.iter().copied());
    }
    let ok = denormal_insensitive(func, host, &placed);
    let used = func.use_counts();
    let mut map: HashMap<Value, Value> = HashMap::default();
    for &v in &placed {
        if let Some(x) = flush_input(func, v) {
            if ok[v.index()] && used[v.index()] > 0 && func.ty(x) == func.ty(v) {
                map.insert(v, x);
            }
        }
    }
    func.replace_uses(&map);
    map.len()
}

/// The f32 value of the flush `arg` (an operand of a host operation)
/// before flushing: `Ok(x)` an existing f32 value, `Err(bits)` 32-bit bits
/// that need a bitcast; `None` when `arg` is not a flush.
fn unflushed(f: &Function, arg: Value) -> Option<Result<Value, Value>> {
    let flush = strip(f, arg);
    let bits = flush_input(f, flush)?;
    let x = strip(f, bits);
    if f.ty(x) == Type::F32 {
        return Some(Ok(x));
    }
    // A bitcast chain may pass an f32 on the way.
    let mut v = bits;
    loop {
        if f.ty(v) == Type::F32 {
            return Some(Ok(v));
        }
        if f.op(v) == Op::Bitcast && is32(f, v) && is32(f, f.args(v)[0]) {
            v = f.args(v)[0];
        } else {
            return Some(Err(bits));
        }
    }
}

/// Operands of host operations that are flushes take the unflushed value
/// (a bitcast to f32 placed just before the operation where none exists).
fn unflush_host_operands(func: &mut Function, host: HostFloat) -> usize {
    let mut replaced = 0;
    for b in func.order.clone() {
        let insts = std::mem::take(&mut func.blocks[b.index()].insts);
        let mut kept = Vec::with_capacity(insts.len());
        for v in insts {
            if host_operation(func, host, v) == Some(None) {
                for i in 0..func.values[v.index()].args.len() {
                    let arg = func.values[v.index()].args[i];
                    let x = match unflushed(func, arg) {
                        Some(Ok(x)) => x,
                        Some(Err(bits)) => {
                            let c = func.new_value(Op::Bitcast, Type::F32, vec![bits]);
                            func.values[c.index()].block = Some(b);
                            kept.push(c);
                            c
                        }
                        None => continue,
                    };
                    func.values[v.index()].args[i] = x;
                    replaced += 1;
                }
            }
            kept.push(v);
        }
        func.blocks[b.index()].insts = kept;
    }
    replaced
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

    const HOST: HostFloat = HostFloat { arith: true, fused: true };

    /// `fmul(flush(x), flush(y))` and a later flush of the product: on a
    /// flushing host the operands take the unflushed values and the
    /// product's flush goes; the flushes of x and y are then dead.
    #[test]
    fn host_flushes_replace_the_emulated_ones() {
        let mut f = Function::new(32, Stage::Compute, 0);
        let b0 = f.new_block();
        f.entry = b0;
        f.order = vec![b0];
        let mut b = Builder::new(&mut f, b0);
        let x = b.emit(Op::UserData(0), Type::U32, vec![]);
        let x = b.bitcast(x, Type::F32);
        let y = b.emit(Op::UserData(1), Type::U32, vec![]);
        let y = b.bitcast(y, Type::F32);
        let fx = flush(&mut b, x);
        let fy = flush(&mut b, y);
        let p = b.fmul(fx, fy);
        let r = flush(&mut b, p);
        let r = b.bitcast(r, Type::U32);
        b.emit(Op::SetReg(Reg::Vgpr(0)), Type::Void, vec![r]);
        f.blocks[0].term = Terminator::Return;
        let mut plain = f.clone();
        assert_eq!(elide_flushes(&mut plain), 0, "without host rules nothing is provably flushed");
        assert_eq!(elide_flushes_on(&mut f, HOST), 3, "the product's flush and both operand flushes");
        let unflushed: Vec<Value> = f.args(p).iter().map(|&a| strip(&f, a)).collect();
        assert_eq!(unflushed, [strip(&f, x), strip(&f, y)]);
        crate::opt::dce::dce(&mut f);
        assert!(f.order.iter().all(|&b| f.block(b).insts.iter().all(|&v| flush_input(&f, v).is_none())), "no flush left");
    }

    /// A multiply by a constant one is a move on the host (measured: the
    /// host compiler drops it and keeps a denormal operand): its operand's
    /// flush stays, and its result is a denormal when the operand is.
    #[test]
    fn host_identities_keep_their_flushes() {
        let mut f = Function::new(32, Stage::Compute, 0);
        let b0 = f.new_block();
        f.entry = b0;
        f.order = vec![b0];
        let mut b = Builder::new(&mut f, b0);
        let x = b.emit(Op::UserData(0), Type::U32, vec![]);
        let x = b.bitcast(x, Type::F32);
        let fx = flush(&mut b, x);
        let one = b.f32(1.0);
        let m = b.fmul(fx, one);
        let zero = b.f32(-0.0);
        let s = b.fadd(x, zero);
        let fm = flush(&mut b, m);
        let fs = flush(&mut b, s);
        for (i, v) in [fm, fs].into_iter().enumerate() {
            let v = b.bitcast(v, Type::U32);
            b.emit(Op::SetReg(Reg::Vgpr(i as u16)), Type::Void, vec![v]);
        }
        f.blocks[0].term = Terminator::Return;
        // fm: the flush of `fx * 1` (never a denormal: fx is a flush) goes;
        // fs: `x + -0` is x, which may be a denormal: its flush stays.
        assert_eq!(elide_flushes_on(&mut f, HOST), 1);
        assert_eq!(f.args(m)[0], fx, "the operand of the move keeps its flush");
        assert_eq!(f.op(fs), Op::Bitcast, "the flush of x + -0 is kept");
        assert!(flush_input(&f, strip(&f, fs)).is_some());
    }

    /// With DenormPreserve(16) the host keeps f32 denormals in add / mul
    /// (measured): only the fused group's rules apply.
    #[test]
    fn host_rules_follow_the_operation_groups() {
        let mut f = Function::new(32, Stage::Compute, 0);
        let b0 = f.new_block();
        f.entry = b0;
        f.order = vec![b0];
        let mut b = Builder::new(&mut f, b0);
        let x = b.emit(Op::UserData(0), Type::U32, vec![]);
        let x = b.bitcast(x, Type::F32);
        let fx = flush(&mut b, x);
        let sum = b.fadd(fx, fx);
        let fused = b.fma(fx, fx, fx);
        for (i, v) in [sum, fused].into_iter().enumerate() {
            let v = flush(&mut b, v);
            let v = b.bitcast(v, Type::U32);
            b.emit(Op::SetReg(Reg::Vgpr(i as u16)), Type::Void, vec![v]);
        }
        f.blocks[0].term = Terminator::Return;
        let only_fused = HostFloat { arith: false, fused: true };
        // The fma's result flush and its three operands.
        assert_eq!(elide_flushes_on(&mut f, only_fused), 4);
        assert_eq!(f.args(sum), &[fx, fx], "the add keeps its flushed operands");
        assert_eq!(f.args(fused), &[x, x, x]);
    }

    /// A flush whose value reaches only host operations, NaN and sign
    /// tests through selects and sign bit operations goes; one that also
    /// reaches a register write stays.
    #[test]
    fn insensitive_uses_drop_the_flush() {
        let mut f = Function::new(32, Stage::Compute, 0);
        let b0 = f.new_block();
        f.entry = b0;
        f.order = vec![b0];
        let mut b = Builder::new(&mut f, b0);
        let c = b.emit(Op::UserData(2), Type::U32, vec![]);
        let zero = b.u32(0);
        let c = b.icmp(ICmp::Ne, c, zero);
        let mut flushes = Vec::new();
        for i in 0..2u32 {
            let x = b.emit(Op::UserData(i), Type::U32, vec![]);
            let x = b.bitcast(x, Type::F32);
            let fx = flush(&mut b, x);
            flushes.push(fx);
            let one = b.f32(2.0);
            let picked = b.select(c, fx, one);
            let bits = b.bitcast(picked, Type::U32);
            let sm = b.u32(0x8000_0000);
            let neg = b.xor(bits, sm);
            let neg = b.bitcast(neg, Type::F32);
            let p = b.fmul(neg, neg);
            let nan = b.fcmp(FCmp::Uno, picked, picked);
            let nan = b.select(nan, one, p);
            let r = b.bitcast(nan, Type::U32);
            b.emit(Op::SetReg(Reg::Vgpr(i as u16)), Type::Void, vec![r]);
            if i == 1 {
                // The flushed value itself is written too.
                let w = b.bitcast(fx, Type::U32);
                b.emit(Op::SetReg(Reg::Vgpr(7)), Type::Void, vec![w]);
            }
        }
        f.blocks[0].term = Terminator::Return;
        elide_flushes_on(&mut f, HOST);
        let flushed = |f: &Function, v: Value| flush_input(f, strip(f, v)).is_some();
        assert!(!flushed(&f, flushes[0]), "only absorbing uses: the flush goes");
        assert!(flushed(&f, flushes[1]), "a register write sees the flush: it stays");
    }
}
