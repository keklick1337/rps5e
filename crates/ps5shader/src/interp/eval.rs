// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Evaluation of pure IR operations on constant bits. Shared by the CPU
//! interpreter and the constant folder, so folding cannot drift from the
//! reference semantics.

use crate::ssa::*;
use std::fmt;

/// Component bits of a value (vector components in order, scalar in [0]).
pub type Val = [u64; 4];

/// Undefined behaviour or an operation the evaluator does not handle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvalError {
    /// The IR operation has undefined behaviour for these operands.
    Undefined(String),
    /// Not a pure operation (needs lane context or memory).
    NotPure,
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::Undefined(s) => write!(f, "undefined behaviour: {s}"),
            EvalError::NotPure => write!(f, "not a pure operation"),
        }
    }
}

fn ub<T>(what: impl Into<String>) -> Result<T, EvalError> {
    Err(EvalError::Undefined(what.into()))
}

// ---------------------------------------------------------------------------
// f16 conversions
// ---------------------------------------------------------------------------

/// f16 bits -> f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: normalize.
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// f64 -> f16 bits with rounding (`rtz` toward zero, else nearest even).
/// NaNs keep the top payload bits and are quieted.
pub fn f64_to_f16(x: f64, rtz: bool) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    let exp = ((bits >> 52) & 0x7FF) as i32;
    let mant = bits & 0x000F_FFFF_FFFF_FFFF;
    if exp == 0x7FF {
        if mant == 0 {
            return sign | 0x7C00;
        }
        return sign | 0x7E00 | ((mant >> 42) as u16 & 0x1FF);
    }
    if exp == 0 && mant == 0 {
        return sign;
    }
    // Value = 1.mant * 2^(exp-1023) (normal f64; f64 denormals are far below
    // the f16 range).
    let e = exp - 1023;
    let full = mant | (1u64 << 52); // 53 significant bits
    // Target: f16 normal has 11 significant bits, exponent -14..=15.
    let (shift, biased) = if e >= -14 {
        (52 - 10, e + 15) // keep 11 bits
    } else {
        // Subnormal: value = m * 2^-24 with m integer.
        let s = 52 - 10 + (-14 - e);
        (s, 0)
    };
    if biased > 30 {
        return if rtz { sign | 0x7BFF } else { sign | 0x7C00 };
    }
    let shift = shift as u32;
    let (kept, rem_nonzero, half, above_half) = if shift >= 64 {
        (0u64, full != 0, false, false)
    } else {
        let kept = full >> shift;
        let rem = full & ((1u64 << shift) - 1);
        let halfway = 1u64 << (shift - 1);
        (kept, rem != 0, rem == halfway, rem > halfway)
    };
    let mut m = kept;
    if !rtz && (above_half || (half && m & 1 == 1)) {
        m += 1;
    }
    let _ = rem_nonzero;
    if biased == 0 {
        // Subnormal (m may have rounded up to the smallest normal 0x400).
        return sign | m as u16;
    }
    // m in [0x400, 0x800]
    let mut be = biased as u32;
    if m == 0x800 {
        m = 0x400;
        be += 1;
        if be > 30 {
            return if rtz { sign | 0x7BFF } else { sign | 0x7C00 };
        }
    }
    sign | ((be as u16) << 10) | (m as u16 & 0x3FF)
}

pub fn f32_to_f16(x: f32, rtz: bool) -> u16 {
    if x.is_nan() {
        let b = x.to_bits();
        return ((b >> 16) & 0x8000) as u16 | 0x7E00 | ((b >> 13) & 0x1FF) as u16;
    }
    f64_to_f16(x as f64, rtz)
}

/// f64 -> f32 rounding toward zero.
fn f64_to_f32_rtz(x: f64) -> f32 {
    let r = x as f32;
    if r.is_nan() || r.is_infinite() && x.is_infinite() {
        return r;
    }
    if (r as f64).abs() > x.abs() {
        // Rounded away from zero: step toward zero.
        if r.is_infinite() {
            return f32::MAX.copysign(r);
        }
        let b = r.to_bits();
        return f32::from_bits(b - 1);
    }
    r
}

// ---------------------------------------------------------------------------
// Float helpers by width
// ---------------------------------------------------------------------------

fn to_f64(s: Scalar, bits: u64) -> f64 {
    match s {
        Scalar::F16 => f16_to_f32(bits as u16) as f64,
        Scalar::F32 => f32::from_bits(bits as u32) as f64,
        _ => f64::from_bits(bits),
    }
}

/// Round an exact (or f64-computed) result to the float type.
fn from_f64(s: Scalar, v: f64) -> u64 {
    match s {
        Scalar::F16 => f64_to_f16(v, false) as u64,
        Scalar::F32 => (v as f32).to_bits() as u64,
        _ => v.to_bits(),
    }
}

/// Binary float op evaluated in the native width (f32 natively; f16 via
/// f64, which is exact before the final rounding for +,-,*).
fn float_bin(s: Scalar, a: u64, b: u64, f32op: fn(f32, f32) -> f32, f64op: fn(f64, f64) -> f64) -> u64 {
    match s {
        Scalar::F32 => f32op(f32::from_bits(a as u32), f32::from_bits(b as u32)).to_bits() as u64,
        Scalar::F64 => f64op(f64::from_bits(a), f64::from_bits(b)).to_bits(),
        _ => from_f64(s, f64op(to_f64(s, a), to_f64(s, b))),
    }
}

fn float_un(s: Scalar, a: u64, f32op: fn(f32) -> f32, f64op: fn(f64) -> f64) -> u64 {
    match s {
        Scalar::F32 => f32op(f32::from_bits(a as u32)).to_bits() as u64,
        Scalar::F64 => f64op(f64::from_bits(a)).to_bits(),
        _ => from_f64(s, f64op(to_f64(s, a))),
    }
}

fn is_nan(s: Scalar, bits: u64) -> bool {
    to_f64(s, bits).is_nan()
}

fn is_zero(s: Scalar, bits: u64) -> bool {
    to_f64(s, bits) == 0.0
}

/// Minimum / maximum: a NaN operand yields the other operand (the first
/// when both are NaN); -0 < +0.
pub fn float_min_max(s: Scalar, a: u64, b: u64, max: bool) -> u64 {
    if is_nan(s, b) {
        return a;
    }
    if is_nan(s, a) {
        return b;
    }
    if is_zero(s, a) && is_zero(s, b) {
        return if max { a & b } else { a | b };
    }
    let (fa, fb) = (to_f64(s, a), to_f64(s, b));
    if max {
        if fa >= fb {
            a
        } else {
            b
        }
    } else if fa < fb {
        a
    } else {
        b
    }
}

/// `2^e` for a normal f64 exponent (-1022..=1023).
fn pow2(e: i64) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// `x * 2^e` rounded once (to nearest even); zeros, infinities and NaNs are
/// returned unchanged.
pub fn ldexp_f64(x: f64, e: i64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    // Normalize x = m * 2^ex with m in [1, 2).
    let (x, bias) = if x.abs() < f64::MIN_POSITIVE { (x * pow2(64), -64i64) } else { (x, 0) };
    let bits = x.to_bits();
    let ex = ((bits >> 52) & 0x7FF) as i64 - 1023 + bias;
    let m = f64::from_bits((bits & !(0x7FFu64 << 52)) | (1023u64 << 52));
    let t = ex + e.clamp(-4000, 4000);
    if t > 1023 {
        return f64::INFINITY.copysign(x);
    }
    if t < -1100 {
        return 0f64.copysign(x);
    }
    if t >= -1022 {
        return m * pow2(t);
    }
    // Denormal result: the first product is exact, the second rounds once.
    m * pow2(t + 100) * pow2(-100)
}

fn sext(v: u64, bits: u32) -> i64 {
    let sh = 64 - bits;
    ((v << sh) as i64) >> sh
}

fn fcmp(p: FCmp, a: f64, b: f64) -> bool {
    let uno = a.is_nan() || b.is_nan();
    match p {
        FCmp::OEq => !uno && a == b,
        FCmp::ONe => !uno && a != b,
        FCmp::OLt => !uno && a < b,
        FCmp::OLe => !uno && a <= b,
        FCmp::OGt => !uno && a > b,
        FCmp::OGe => !uno && a >= b,
        FCmp::Ord => !uno,
        FCmp::Uno => uno,
        FCmp::UEq => uno || a == b,
        FCmp::UNe => uno || a != b,
        FCmp::ULt => uno || a < b,
        FCmp::ULe => uno || a <= b,
        FCmp::UGt => uno || a > b,
        FCmp::UGe => uno || a >= b,
    }
}

fn icmp(p: ICmp, a: u64, b: u64, bits: u32) -> bool {
    let (sa, sb) = (sext(a, bits), sext(b, bits));
    match p {
        ICmp::Eq => a == b,
        ICmp::Ne => a != b,
        ICmp::ULt => a < b,
        ICmp::ULe => a <= b,
        ICmp::UGt => a > b,
        ICmp::UGe => a >= b,
        ICmp::SLt => sa < sb,
        ICmp::SLe => sa <= sb,
        ICmp::SGt => sa > sb,
        ICmp::SGe => sa >= sb,
    }
}

/// Split the bits of a value into `n` components of `bits` each.
fn split(v: &[u64], from: Type) -> Vec<u64> {
    // Flatten to a bit string of 64-bit words.
    let s = from.component();
    let mut out = Vec::new();
    for c in 0..from.components() as usize {
        out.push((v[c] & s.mask(), s.bits()));
    }
    let _ = out;
    v.to_vec()
}

fn bitcast(v: &Val, from: Type, to: Type) -> Val {
    let fs = from.component();
    let ts = to.component();
    // Concatenate source components little-endian into a u128.
    let mut acc: u128 = 0;
    let mut pos = 0u32;
    for c in 0..from.components() as usize {
        acc |= ((v[c] & fs.mask()) as u128) << pos;
        pos += fs.bits();
    }
    let mut out = [0u64; 4];
    let mut p = 0u32;
    for slot in out.iter_mut().take(to.components() as usize) {
        *slot = ((acc >> p) as u64) & ts.mask();
        p += ts.bits();
    }
    let _ = split;
    out
}

/// Evaluate a pure operation. `args` are (type, bits) pairs.
pub fn eval(op: Op, ty: Type, args: &[(Type, Val)]) -> Result<Val, EvalError> {
    let s = ty.scalar();
    let a = |i: usize| args[i].1[0];
    let at = |i: usize| args[i].0;
    let mask = s.map_or(0, Scalar::mask);
    let ret = |x: u64| -> Result<Val, EvalError> { Ok([x & mask, 0, 0, 0]) };
    let width = s.map_or(0, Scalar::bits);
    match op {
        Op::Const(c) => ret(c),
        Op::Select => Ok(if a(0) != 0 { args[1].1 } else { args[2].1 }),
        Op::Bitcast => Ok(bitcast(&args[0].1, at(0), ty)),
        Op::Construct => {
            let mut out = [0u64; 4];
            for (i, (_, v)) in args.iter().enumerate() {
                out[i] = v[0];
            }
            Ok(out)
        }
        Op::Extract(i) => ret(args[0].1[i as usize]),
        Op::Insert(i) => {
            let mut out = args[0].1;
            out[i as usize] = a(1);
            Ok(out)
        }
        Op::IAdd => ret(a(0).wrapping_add(a(1))),
        Op::ISub => ret(a(0).wrapping_sub(a(1))),
        Op::IMul => ret(a(0).wrapping_mul(a(1))),
        Op::UMulHi => ret(((a(0) as u128 * a(1) as u128) >> width) as u64),
        Op::SMulHi => {
            let p = sext(a(0), width) as i128 * sext(a(1), width) as i128;
            ret((p >> width) as u64)
        }
        Op::UDiv | Op::UMod => {
            if a(1) == 0 {
                return ub("division by zero");
            }
            ret(if op == Op::UDiv { a(0) / a(1) } else { a(0) % a(1) })
        }
        Op::IAddCarry => {
            let (x, y) = (a(0) & 0xFFFF_FFFF, a(1) & 0xFFFF_FFFF);
            let s = x + y;
            Ok([s & 0xFFFF_FFFF, s >> 32, 0, 0])
        }
        Op::ISubBorrow => {
            let (x, y) = (a(0) & 0xFFFF_FFFF, a(1) & 0xFFFF_FFFF);
            Ok([x.wrapping_sub(y) & 0xFFFF_FFFF, (x < y) as u64, 0, 0])
        }
        Op::INeg => ret(a(0).wrapping_neg()),
        Op::SMin => ret(if sext(a(0), width) <= sext(a(1), width) { a(0) } else { a(1) }),
        Op::SMax => ret(if sext(a(0), width) >= sext(a(1), width) { a(0) } else { a(1) }),
        Op::UMin => ret(a(0).min(a(1))),
        Op::UMax => ret(a(0).max(a(1))),
        Op::Shl | Op::LShr | Op::AShr => {
            let c = a(1);
            if c >= width as u64 {
                return ub(format!("shift by {c} of a {width}-bit value"));
            }
            let c = c as u32;
            ret(match op {
                Op::Shl => a(0) << c,
                Op::LShr => a(0) >> c,
                _ => (sext(a(0), width) >> c) as u64,
            })
        }
        Op::And => ret(a(0) & a(1)),
        Op::Or => ret(a(0) | a(1)),
        Op::Xor => ret(a(0) ^ a(1)),
        Op::Not => {
            if ty == Type::BOOL {
                ret((a(0) == 0) as u64)
            } else {
                ret(!a(0))
            }
        }
        Op::BitReverse => ret(a(0).reverse_bits() >> (64 - width)),
        Op::BitCount => ret((a(0) & at(0).component().mask()).count_ones() as u64),
        Op::FindLsb => {
            let v = a(0) & at(0).component().mask();
            ret(if v == 0 { u64::MAX } else { v.trailing_zeros() as u64 })
        }
        Op::FindUMsb => {
            let v = a(0) & at(0).component().mask();
            ret(if v == 0 { u64::MAX } else { 63 - v.leading_zeros() as u64 })
        }
        Op::FindSMsb => {
            let w = at(0).component().bits();
            let v = sext(a(0), w);
            let x = if v < 0 { !v } else { v } as u64 & at(0).component().mask();
            ret(if x == 0 { u64::MAX } else { 63 - x.leading_zeros() as u64 })
        }
        Op::BitFieldUExtract | Op::BitFieldSExtract => {
            let (off, cnt) = (a(1), a(2));
            if off + cnt > width as u64 {
                return ub(format!("bit field offset {off} count {cnt} past {width} bits"));
            }
            if cnt == 0 {
                return ret(0);
            }
            let f = (a(0) >> off) & ((1u128 << cnt) - 1) as u64;
            ret(if op == Op::BitFieldSExtract { sext(f, cnt as u32) as u64 } else { f })
        }
        Op::BitFieldInsert => {
            let (off, cnt) = (a(2), a(3));
            if off + cnt > width as u64 {
                return ub(format!("bit field insert offset {off} count {cnt} past {width} bits"));
            }
            if cnt == 0 {
                return ret(a(0));
            }
            let m = (((1u128 << cnt) - 1) as u64) << off;
            ret((a(0) & !m) | ((a(1) << off) & m))
        }
        Op::UConvert => ret(a(0) & at(0).component().mask()),
        Op::SConvert => ret(sext(a(0), at(0).component().bits()) as u64),
        Op::ICmp(p) => {
            let w = at(0).component().bits();
            ret(icmp(p, a(0), a(1), w) as u64)
        }
        // ---- float ----
        Op::FAdd | Op::FSub | Op::FMul | Op::FDiv => {
            let s = s.unwrap();
            ret(match op {
                Op::FAdd => float_bin(s, a(0), a(1), |x, y| x + y, |x, y| x + y),
                Op::FSub => float_bin(s, a(0), a(1), |x, y| x - y, |x, y| x - y),
                Op::FMul => float_bin(s, a(0), a(1), |x, y| x * y, |x, y| x * y),
                _ => float_bin(s, a(0), a(1), |x, y| x / y, |x, y| x / y),
            })
        }
        Op::FFma => {
            let s = s.unwrap();
            ret(match s {
                Scalar::F32 => f32::from_bits(a(0) as u32).mul_add(f32::from_bits(a(1) as u32), f32::from_bits(a(2) as u32)).to_bits() as u64,
                Scalar::F64 => f64::from_bits(a(0)).mul_add(f64::from_bits(a(1)), f64::from_bits(a(2))).to_bits(),
                _ => from_f64(s, to_f64(s, a(0)).mul_add(to_f64(s, a(1)), to_f64(s, a(2)))),
            })
        }
        Op::FNeg => ret(a(0) ^ (1u64 << (width - 1))),
        Op::FAbs => ret(a(0) & !(1u64 << (width - 1))),
        Op::FMin | Op::FMax => ret(float_min_max(s.unwrap(), a(0), a(1), op == Op::FMax)),
        Op::FFloor => ret(float_un(s.unwrap(), a(0), f32::floor, f64::floor)),
        Op::FCeil => ret(float_un(s.unwrap(), a(0), f32::ceil, f64::ceil)),
        Op::FTrunc => ret(float_un(s.unwrap(), a(0), f32::trunc, f64::trunc)),
        Op::FRoundEven => ret(float_un(s.unwrap(), a(0), f32::round_ties_even, f64::round_ties_even)),
        Op::FSqrt => ret(float_un(s.unwrap(), a(0), f32::sqrt, f64::sqrt)),
        Op::FInvSqrt => ret(float_un(s.unwrap(), a(0), |x| 1.0 / x.sqrt(), |x| 1.0 / x.sqrt())),
        Op::FRcp => ret(float_un(s.unwrap(), a(0), |x| 1.0 / x, |x| 1.0 / x)),
        Op::FExp2 => ret(float_un(s.unwrap(), a(0), f32::exp2, f64::exp2)),
        Op::FLog2 => ret(float_un(s.unwrap(), a(0), f32::log2, f64::log2)),
        Op::FSin => ret(float_un(s.unwrap(), a(0), f32::sin, f64::sin)),
        Op::FCos => ret(float_un(s.unwrap(), a(0), f32::cos, f64::cos)),
        Op::FLdexp => {
            let s = s.unwrap();
            let e = sext(a(1), at(1).component().bits());
            let r = ldexp_f64(to_f64(s, a(0)), e);
            ret(match s {
                Scalar::F64 => r.to_bits(),
                _ => from_f64(s, r),
            })
        }
        Op::FCmp(p) => {
            let fs = at(0).component();
            ret(fcmp(p, to_f64(fs, a(0)), to_f64(fs, a(1))) as u64)
        }
        Op::FConvert(r) => {
            let fs = at(0).component();
            let ts = s.unwrap();
            let v = a(0);
            ret(match (fs, ts) {
                (Scalar::F16, Scalar::F32) => f16_to_f32(v as u16).to_bits() as u64,
                (Scalar::F16, Scalar::F64) => (f16_to_f32(v as u16) as f64).to_bits(),
                (Scalar::F32, Scalar::F64) => (f32::from_bits(v as u32) as f64).to_bits(),
                (Scalar::F32, Scalar::F16) => f32_to_f16(f32::from_bits(v as u32), r == Rounding::Rtz) as u64,
                (Scalar::F64, Scalar::F16) => f64_to_f16(f64::from_bits(v), r == Rounding::Rtz) as u64,
                (Scalar::F64, Scalar::F32) => {
                    let x = f64::from_bits(v);
                    let y = if r == Rounding::Rtz { f64_to_f32_rtz(x) } else { x as f32 };
                    y.to_bits() as u64
                }
                _ => v,
            })
        }
        Op::FToS | Op::FToU => {
            let fs = at(0).component();
            let x = to_f64(fs, a(0)).trunc();
            if x.is_nan() {
                return ub("float to integer conversion of NaN");
            }
            if op == Op::FToS {
                let lo = -(2f64.powi(width as i32 - 1));
                let hi = 2f64.powi(width as i32 - 1);
                if x < lo || x >= hi {
                    return ub(format!("float {x} out of signed {width}-bit range"));
                }
                ret(x as i64 as u64)
            } else {
                if x < 0.0 || x >= 2f64.powi(width as i32) {
                    return ub(format!("float {x} out of unsigned {width}-bit range"));
                }
                ret(x as u64)
            }
        }
        Op::SToF(r) | Op::UToF(r) => {
            let iw = at(0).component().bits();
            let ts = s.unwrap();
            let x: f64 = if matches!(op, Op::SToF(_)) { sext(a(0), iw) as f64 } else { a(0) as f64 };
            // Integers up to 32 bits are exact in f64.
            ret(match ts {
                Scalar::F32 => {
                    let y = if r == Rounding::Rtz { f64_to_f32_rtz(x) } else { x as f32 };
                    y.to_bits() as u64
                }
                Scalar::F16 => f64_to_f16(x, r == Rounding::Rtz) as u64,
                _ => x.to_bits(),
            })
        }
        _ => Err(EvalError::NotPure),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip() {
        for h in 0..=0xFFFFu16 {
            let f = f16_to_f32(h);
            if f.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16(f, false), h, "{h:#x}");
            assert_eq!(f32_to_f16(f, true), h, "{h:#x}");
        }
    }

    #[test]
    fn f16_rounding() {
        // 1 + 2^-11 is halfway between 1 and the next f16: ties to even.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11), false), 0x3C00);
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11), false), 0x3C02);
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11), true), 0x3C01);
        assert_eq!(f32_to_f16(65520.0, false), 0x7C00);
        assert_eq!(f32_to_f16(65520.0, true), 0x7BFF);
        assert_eq!(f32_to_f16(1e-8, false), 0x0000);
        assert_eq!(f32_to_f16(2f32.powi(-24), false), 0x0001);
        assert_eq!(f32_to_f16(f32::INFINITY, true), 0x7C00);
    }

    #[test]
    fn min_max_signed_zero() {
        let z = 0f32.to_bits() as u64;
        let nz = (-0f32).to_bits() as u64;
        assert_eq!(float_min_max(Scalar::F32, z, nz, false), nz);
        assert_eq!(float_min_max(Scalar::F32, z, nz, true), z);
        let nan = f32::NAN.to_bits() as u64;
        assert_eq!(float_min_max(Scalar::F32, nan, z, false), z);
    }

    #[test]
    fn ldexp_rounds_once() {
        let max = f64::from_bits(0x7FEF_FFFF_FFFF_FFFF);
        assert_eq!(ldexp_f64(max, -2100).to_bits(), 0);
        assert_eq!(ldexp_f64(max, -2046).to_bits(), 0x0010_0000_0000_0000);
        assert_eq!(ldexp_f64(max, -2097).to_bits(), 2);
        assert_eq!(ldexp_f64(f64::from_bits(1), 1074), 1.0);
        assert_eq!(ldexp_f64(1.5, -1074).to_bits(), 2);
        assert_eq!(ldexp_f64(1.0, 1024), f64::INFINITY);
        assert_eq!(ldexp_f64(-1.0, i64::from(i32::MIN)).to_bits(), 0x8000_0000_0000_0000);
    }

    #[test]
    fn shifts_are_undefined_past_width() {
        let r = eval(Op::Shl, Type::U32, &[(Type::U32, [1, 0, 0, 0]), (Type::U32, [32, 0, 0, 0])]);
        assert!(matches!(r, Err(EvalError::Undefined(_))));
    }
}
