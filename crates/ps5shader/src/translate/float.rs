// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Float ALU (f32, f16, packed f16, f64) and conversions, with the RDNA2
//! special-value behaviour measured on hardware.

use super::operand::{DstMods, Mods};
use super::{Result, Translator};
use crate::ir_types::*;
use crate::opcode::Opcode;
use crate::ssa::*;

/// Unary f32 operations with hardware special cases.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Unary {
    Rcp,
    RcpIflag,
    Rsq,
    Sqrt,
    Log,
    Exp,
    Fract,
    Sin,
    Cos,
    Trunc,
    Ceil,
    Floor,
    Rndne,
}

impl<'a> Translator<'a> {
    fn bits(&mut self, v: Value) -> Value {
        self.b().bitcast(v, Type::U32)
    }

    fn float(&mut self, v: Value) -> Value {
        self.b().bitcast(v, Type::F32)
    }

    /// The quiet bit an operation sets in a NaN it returns: only with
    /// IEEE_MODE. Hardware rows (NaN quieting, RDNA2): with IEEE_MODE clear
    /// v_sqrt/rcp/fract/floor/trunc/frexp_mant/ldexp/div_fixup/mullit and
    /// their f16 forms return a signaling NaN unchanged.
    pub fn nan_quiet_bit(&mut self, half: bool) -> Value {
        let bit = match (self.float_mode.ieee, half) {
            (false, _) => 0,
            (true, false) => 0x0040_0000,
            (true, true) => 0x200,
        };
        self.b().u32(bit)
    }

    /// Whether the f32 bits `bits` are a NaN: an unordered self-compare
    /// with [`super::rewrites::FLOAT_CLASS_COMPARE`] (the host preserves
    /// NaNs), the magnitude bits above infinity otherwise.
    pub fn is_nan_bits(&mut self, bits: Value) -> Value {
        if self.options.rewrites & super::rewrites::FLOAT_CLASS_COMPARE != 0 {
            let f = self.float(bits);
            return self.b().fcmp(FCmp::Uno, f, f);
        }
        let mag_mask = self.b().u32(0x7FFF_FFFF);
        let inf = self.b().u32(0x7F80_0000);
        let mag = self.b().and(bits, mag_mask);
        self.b().icmp(ICmp::UGt, mag, inf)
    }

    /// The NaN an f32 add/sub/mul/fma returns, chosen on the bits
    /// (hardware rows, RDNA2): the first NaN source in operand order (after
    /// source modifiers; a subtracted operand counts negated), quieted only
    /// with IEEE_MODE; an invalid operation (inf - inf, inf * 0) gives
    /// 0xffc00000. With `invalid_product` (fma forms) a NaN addend is not
    /// returned when the product is inf * 0.
    pub fn nan_result_f32(&mut self, sources: &[Value], result: Value, invalid_product: Option<Value>) -> Value {
        let is_nan = |t: &mut Self, bits: Value| t.is_nan_bits(bits);
        let mut bits = self.bits(result);
        let result_nan = is_nan(self, bits);
        let default_nan = self.b().u32(0xFFC0_0000);
        bits = self.b().select(result_nan, default_nan, bits);
        let qb = self.nan_quiet_bit(false);
        for (i, &source) in sources.iter().enumerate().rev() {
            let source_bits = self.bits(source);
            let mut selected = is_nan(self, source_bits);
            if let Some(invalid) = invalid_product {
                if sources.len() == 3 && i == 2 {
                    let valid = self.b().lnot(invalid);
                    selected = self.b().land(selected, valid);
                }
            }
            let quiet = self.b().or(source_bits, qb);
            bits = self.b().select(selected, quiet, bits);
        }
        let r = self.float(bits);
        // A NaN is not a denormal: the selection keeps the result's class.
        if self.not_denormal.contains(&result) {
            self.not_denormal.insert(r);
        }
        r
    }

    /// `-v` as a sign-bit flip (NaN payloads kept on every backend).
    fn negate_bits(&mut self, v: Value) -> Value {
        let bits = self.bits(v);
        let sign = self.b().u32(0x8000_0000);
        let flipped = self.b().xor(bits, sign);
        self.float(flipped)
    }

    /// Whether an fma's NaN result comes from an invalid operation: the
    /// product is inf * 0 (`invalid_product`) or no source is a NaN (inf -
    /// inf). Hardware returns the default NaN then (RDNA2).
    pub fn invalid_operation_f32(&mut self, sources: &[Value], invalid_product: Value) -> Value {
        let mut no_nan = self.b().bool(true);
        for &source in sources {
            let bits = self.bits(source);
            let nan = self.is_nan_bits(bits);
            let not_nan = self.b().lnot(nan);
            no_nan = self.b().land(no_nan, not_nan);
        }
        self.b().lor(invalid_product, no_nan)
    }

    /// The f16 word `bits`, or the default NaN 0xfe00 when it is a NaN and
    /// `invalid` holds.
    pub(super) fn default_nan_f16_on_invalid(&mut self, bits: Value, invalid: Value) -> Value {
        let mm = self.b().u32(0x7FFF);
        let mag = self.b().and(bits, mm);
        let inf = self.b().u32(0x7C00);
        let nan = self.b().icmp(ICmp::UGt, mag, inf);
        let take = self.b().land(nan, invalid);
        let dn = self.b().u32(0xFE00);
        self.b().select(take, dn, bits)
    }

    /// Whether the product `a * b` is inf * 0 (either order).
    pub fn invalid_product_f32(&mut self, a: Value, b: Value) -> Value {
        let mag_mask = self.b().u32(0x7FFF_FFFF);
        let inf = self.b().u32(0x7F80_0000);
        let zero = self.b().u32(0);
        let ab = self.bits(a);
        let bb = self.bits(b);
        let am = self.b().and(ab, mag_mask);
        let bm = self.b().and(bb, mag_mask);
        let a_inf = self.b().icmp(ICmp::Eq, am, inf);
        let b_inf = self.b().icmp(ICmp::Eq, bm, inf);
        let a_zero = self.b().icmp(ICmp::Eq, am, zero);
        let b_zero = self.b().icmp(ICmp::Eq, bm, zero);
        let first = self.b().land(a_inf, b_zero);
        let second = self.b().land(b_inf, a_zero);
        self.b().lor(first, second)
    }

    /// With f32 results flushed, a product of two finite non-zero operands
    /// whose exponent underflows (and, for an fma, a zero addend) is a
    /// zero of the product's sign. Tininess is decided after rounding (in
    /// the f32 round mode): a product just below the smallest normal that
    /// rounds up to it is that normal (hardware rows, RDNA2).
    pub fn flush_tiny_product(&mut self, a: Value, b: Value, product: Value, addend: Option<Value>) -> Value {
        self.flush_tiny_product_rounded(a, b, product, addend, true)
    }

    /// [`Self::flush_tiny_product`]; `after_rounding` false decides
    /// tininess before rounding (v_mad/mac_legacy_f32, measured).
    pub fn flush_tiny_product_rounded(&mut self, a: Value, b: Value, product: Value, addend: Option<Value>, after_rounding: bool) -> Value {
        if self.f32_flush & 2 == 0 {
            return product;
        }
        let after_rounding = after_rounding && self.options.float_rules & super::float_rules::TINY_AFTER_ROUNDING != 0;
        let round_mode = (self.float_mode.mode & 3) as u32;
        if after_rounding && round_mode == 0 && self.options.float_rules & super::float_rules::TINY_ONE_MULTIPLY != 0 {
            return self.flush_tiny_product_nearest(a, b, product, addend);
        }
        let ab = self.bits(a);
        let bb = self.bits(b);
        let c23 = self.b().u32(23);
        let c8 = self.b().u32(8);
        let ea = self.b().bfe_u(ab, c23, c8);
        let eb = self.b().bfe_u(bb, c23, c8);
        let mant = self.b().u32(0x7F_FFFF);
        let implicit = self.b().u32(0x80_0000);
        let ma = self.b().and(ab, mant);
        let ma = self.b().or(ma, implicit);
        let mb = self.b().and(bb, mant);
        let mb = self.b().or(mb, implicit);
        let hi = self.b().umul_hi(ma, mb);
        let c7fff = self.b().u32(0x7FFF);
        let carry_c = self.b().icmp(ICmp::UGt, hi, c7fff);
        let one = self.b().u32(1);
        let zero = self.b().u32(0);
        let carry = self.b().select(carry_c, one, zero);
        // Without a carry the 48-bit significand product lies in
        // [2^46, 2^47); it rounds up to 2^47 (the next binade) when its
        // high half is 0x7fff and the dropped 23 bits round up.
        let rounds_up = if after_rounding && round_mode != 3 {
            let at_top = self.b().icmp(ICmp::Eq, hi, c7fff);
            let lo = self.b().imul(ma, mb);
            let limit = self.b().u32(if round_mode == 0 { 0xFFBF_FFFF } else { 0xFF80_0000 });
            let up = self.b().icmp(ICmp::UGt, lo, limit);
            Some(self.b().land(at_top, up))
        } else {
            None
        };
        let cff = self.b().u32(0xFF);
        let finite = |t: &mut Self, e: Value| {
            let nz = t.b().icmp(ICmp::Ne, e, zero);
            let nf = t.b().icmp(ICmp::Ne, e, cff);
            t.b().land(nz, nf)
        };
        let fa = finite(self, ea);
        let fb = finite(self, eb);
        let both = self.b().land(fa, fb);
        let sum = self.b().iadd(ea, eb);
        let sum = self.b().iadd(sum, carry);
        let c128 = self.b().u32(128);
        let under = self.b().icmp(ICmp::ULt, sum, c128);
        let mut tiny = self.b().land(both, under);
        if let Some(c) = addend {
            let cb = self.bits(c);
            let mm = self.b().u32(0x7FFF_FFFF);
            let cm = self.b().and(cb, mm);
            let cz = self.b().icmp(ICmp::Eq, cm, zero);
            tiny = self.b().land(tiny, cz);
        }
        let x = self.b().xor(ab, bb);
        let sm = self.b().u32(0x8000_0000);
        let sign = self.b().and(x, sm);
        let flushed = match rounds_up {
            Some(up) => {
                // Only an exponent sum of 127 (just below the smallest
                // normal) reaches it; the directed modes round up away
                // from zero only on their own side.
                let c127 = self.b().u32(127);
                let edge = self.b().icmp(ICmp::Eq, sum, c127);
                let mut up = self.b().land(edge, up);
                if round_mode != 0 {
                    let side = self.b().u32(if round_mode == 1 { 0 } else { 0x8000_0000 });
                    let ours = self.b().icmp(ICmp::Eq, sign, side);
                    up = self.b().land(up, ours);
                }
                let normal = self.b().u32(0x0080_0000);
                let smallest = self.b().or(sign, normal);
                self.b().select(up, smallest, sign)
            }
            None => sign,
        };
        let flushed = self.float(flushed);
        self.b().select(tiny, flushed, product)
    }

    /// [`Self::flush_tiny_product`] to nearest even with one multiply:
    /// `a * 2^64` is exact (or infinite, and then the product is not tiny),
    /// so `(a * 2^64) * b` is the exact product rounded once to 24 bits at
    /// an exponent where it is normal; 0 < |that| < 2^-62 is tininess after
    /// rounding. A product that rounds up to 2^-126 is not tiny and keeps
    /// the computed product, which is 2^-126 there. NaN, infinite and zero
    /// operands give no tiny product.
    fn flush_tiny_product_nearest(&mut self, a: Value, b: Value, product: Value, addend: Option<Value>) -> Value {
        let scale = self.b().f32(f32::from_bits(0x5F80_0000));
        let scaled = self.b().fmul(a, scale);
        let p = self.b().fmul(scaled, b);
        let magnitude = self.b().fabs(p);
        let zero = self.b().f32(0.0);
        let limit = self.b().f32(f32::from_bits(0x2080_0000));
        let nonzero = self.b().fcmp(FCmp::OGt, magnitude, zero);
        let small = self.b().fcmp(FCmp::OLt, magnitude, limit);
        let mut tiny = self.b().land(nonzero, small);
        if let Some(c) = addend {
            let cb = self.bits(c);
            let mm = self.b().u32(0x7FFF_FFFF);
            let cm = self.b().and(cb, mm);
            let z = self.b().u32(0);
            let cz = self.b().icmp(ICmp::Eq, cm, z);
            tiny = self.b().land(tiny, cz);
        }
        let ab = self.bits(a);
        let bb = self.bits(b);
        let x = self.b().xor(ab, bb);
        let sm = self.b().u32(0x8000_0000);
        let sign = self.b().and(x, sm);
        let flushed = self.float(sign);
        let r = self.b().select(tiny, flushed, product);
        if addend.is_none() && self.options.rewrites & super::rewrites::FLUSH_ELISION != 0 {
            // Not tiny: the product rounded to 24 bits is at least 2^-126,
            // so the exact product is within 2^-151 of it and the host's
            // product (rounded on the denormal grid) is at least 2^-126 too.
            self.not_denormal.insert(r);
        }
        r
    }

    /// Whether the f16 denormal mode rules apply (see
    /// [`super::float_rules::F16_DENORMALS`]).
    fn f16_denormal_rules(&self) -> bool {
        self.options.float_rules & super::float_rules::F16_DENORMALS != 0
    }

    /// Whether the f16/f64 denormal mode flushes: `keep_bit` 0x40 for
    /// inputs (FLOAT_MODE bit 6), 0x80 for results (bit 7).
    pub(super) fn f16_flushes(&self, keep_bit: u8) -> bool {
        self.f16_denormal_rules() && self.float_mode.mode & keep_bit == 0
    }

    /// f16 bits (in a u32) below the smallest normal become a zero of
    /// their sign when the f16 denormal mode flushes `keep_bit` (0x40
    /// inputs, 0x80 results); unchanged otherwise (no code).
    pub(super) fn flush_f16_bits(&mut self, bits: Value, keep_bit: u8) -> Value {
        if !self.f16_flushes(keep_bit) {
            return bits;
        }
        let mm = self.b().u32(0x7FFF);
        let mag = self.b().and(bits, mm);
        let lim = self.b().u32(0x0400);
        let tiny = self.b().icmp(ICmp::ULt, mag, lim);
        let sm = self.b().u32(0x8000);
        let sign = self.b().and(bits, sm);
        self.b().select(tiny, sign, bits)
    }

    /// An f16 input (widened to f32) below 2^-14 is a zero of its sign
    /// when FLOAT_MODE flushes f16 input denormals (bit 6 clear).
    pub(super) fn flush_f16_input_value(&mut self, v: Value) -> Value {
        if !self.f16_flushes(0x40) {
            return v;
        }
        self.flush_tiny_f16_value(v, 0x3880_0000)
    }

    /// An f64 input with a zero exponent is a zero of its sign when the
    /// f16/f64 denormal mode flushes inputs (FLOAT_MODE bit 6 clear).
    pub(super) fn flush_f64_input(&mut self, v: Value) -> Value {
        if !self.f16_flushes(0x40) {
            return v;
        }
        let u = self.b().bitcast(v, Type::U64);
        let (lo, hi) = self.b().split_u64(u);
        let em = self.b().u32(0x7FF0_0000);
        let e = self.b().and(hi, em);
        let zero = self.b().u32(0);
        let tiny = self.b().icmp(ICmp::Eq, e, zero);
        let sm = self.b().u32(0x8000_0000);
        let sign = self.b().and(hi, sm);
        let lo = self.b().select(tiny, zero, lo);
        let hi = self.b().select(tiny, sign, hi);
        let u = self.b().join_u64(lo, hi);
        self.b().bitcast(u, Type::F64)
    }

    /// An f32 value whose magnitude bits are below `threshold` is a zero of
    /// its sign.
    pub(super) fn flush_tiny_f16_value(&mut self, v: Value, threshold: u32) -> Value {
        let bits = self.bits(v);
        let mm = self.b().u32(0x7FFF_FFFF);
        let mag = self.b().and(bits, mm);
        let lim = self.b().u32(threshold);
        let tiny = self.b().icmp(ICmp::ULt, mag, lim);
        let sm = self.b().u32(0x8000_0000);
        let sign = self.b().and(bits, sm);
        let r = self.b().select(tiny, sign, bits);
        self.float(r)
    }

    /// v_ldexp_f32 on the bits (after source modifiers) in the current
    /// FLOAT_MODE: denormal inputs are normalized when f32 input denormals
    /// are kept and flush otherwise; a result below the smallest normal
    /// flushes when result denormals are flushed (tininess before rounding)
    /// and otherwise rounds once in the f32 round mode; overflow gives the
    /// infinity, or the largest finite value in the modes that round away
    /// from it; NaNs pass, quieted with IEEE_MODE.
    fn ldexp_f32_bits(&mut self, bits: Value, off: Value) -> Value {
        let denormals = self.float_mode.f32_denormals();
        let keep_in = denormals & 1 != 0;
        let keep_out = denormals & 2 != 0;
        let round_mode = (self.float_mode.mode & 3) as u32;
        let c = |t: &mut Self, x: u32| t.b().u32(x);
        let zero = c(self, 0);
        let one = c(self, 1);
        let sm = c(self, 0x8000_0000);
        let sign = self.b().and(bits, sm);
        let negative = self.b().icmp(ICmp::Ne, sign, zero);
        let c23 = c(self, 23);
        let c8 = c(self, 8);
        let exponent = self.b().bfe_u(bits, c23, c8);
        let fm = c(self, 0x7F_FFFF);
        let fraction = self.b().and(bits, fm);
        let implicit = c(self, 0x80_0000);
        let normal_sig = self.b().or(fraction, implicit);
        let ez = self.b().icmp(ICmp::Eq, exponent, zero);
        // Biased exponent and 24-bit significand (implicit bit 23).
        let (e, sig) = if keep_in {
            // A denormal 0.f * 2^-126 whose top set bit is `msb` is
            // 1.x * 2^(msb - 149): biased exponent msb - 22.
            let msb = self.b().find_umsb(fraction);
            let c22 = c(self, 22);
            let de = self.b().isub(msb, c22);
            let sh = self.b().isub(c23, msb);
            let c31 = c(self, 31);
            let sh = self.b().umin(sh, c31);
            let ds = self.b().shl(fraction, sh);
            let e = self.b().select(ez, de, exponent);
            let s = self.b().select(ez, ds, normal_sig);
            (e, s)
        } else {
            (exponent, normal_sig)
        };
        let hi = c(self, 512);
        let lo = c(self, (-512i32) as u32);
        let t = self.b().smin(off, hi);
        let clamped = self.b().smax(t, lo);
        let scaled = self.b().iadd(e, clamped);
        // Normal result: exact.
        let kept = self.b().and(sig, fm);
        let sh = self.b().shl(scaled, c23);
        let normal = self.b().or(kept, sh);
        let mut r = self.b().or(normal, sign);
        // Tiny result.
        let tiny = self.b().icmp(ICmp::SLe, scaled, zero);
        let tiny_value = if keep_out {
            let shift = self.b().isub(one, scaled);
            let c26 = c(self, 26);
            let shift = self.b().umin(shift, c26);
            let q = self.b().lshr(sig, shift);
            let unit = self.b().shl(one, shift);
            let mask = self.b().isub(unit, one);
            let rem = self.b().and(sig, mask);
            let inexact = self.b().icmp(ICmp::Ne, rem, zero);
            let up = match round_mode {
                0 => {
                    let half = self.b().lshr(unit, one);
                    let above = self.b().icmp(ICmp::UGt, rem, half);
                    let tie = self.b().icmp(ICmp::Eq, rem, half);
                    let odd = self.b().and(q, one);
                    let odd = self.b().icmp(ICmp::Ne, odd, zero);
                    let tie_up = self.b().land(tie, odd);
                    Some(self.b().lor(above, tie_up))
                }
                1 => {
                    let positive = self.b().lnot(negative);
                    Some(self.b().land(inexact, positive))
                }
                2 => Some(self.b().land(inexact, negative)),
                _ => None,
            };
            let q = match up {
                Some(up) => {
                    let inc = self.b().iadd(q, one);
                    self.b().select(up, inc, q)
                }
                None => q,
            };
            self.b().or(q, sign)
        } else {
            sign
        };
        r = self.b().select(tiny, tiny_value, r);
        // Overflow.
        let c255 = c(self, 255);
        let over = self.b().icmp(ICmp::SGe, scaled, c255);
        let inf = c(self, 0x7F80_0000);
        let max = c(self, 0x7F7F_FFFF);
        let to_max = match round_mode {
            0 => None,
            1 => Some(negative),
            2 => Some(self.b().lnot(negative)),
            _ => Some(self.b().bool(true)),
        };
        let magnitude = match to_max {
            Some(to_max) => self.b().select(to_max, max, inf),
            None => inf,
        };
        let over_v = self.b().or(sign, magnitude);
        r = self.b().select(over, over_v, r);
        // Zero (and flushed denormal) inputs.
        let input_zero = if keep_in {
            let fz = self.b().icmp(ICmp::Eq, fraction, zero);
            self.b().land(ez, fz)
        } else {
            ez
        };
        r = self.b().select(input_zero, sign, r);
        // Infinities and NaNs.
        let special = self.b().icmp(ICmp::Eq, exponent, c255);
        let qb = self.nan_quiet_bit(false);
        let mm = c(self, 0x7FFF_FFFF);
        let mag = self.b().and(bits, mm);
        let nan = self.b().icmp(ICmp::UGt, mag, inf);
        let quieted = self.b().or(bits, qb);
        let passed = self.b().select(nan, quieted, bits);
        self.b().select(special, passed, r)
    }

    /// v_cvt_f32_i32 / v_cvt_f32_u32 in a directed f32 round mode
    /// (`round_mode` 1 toward +inf, 2 toward -inf, 3 toward zero), as bits:
    /// the magnitude truncated to 24 significant bits converts exactly, and
    /// the next float away from zero is one more in the bits.
    fn int_to_f32_directed(&mut self, v: Value, signed: bool, round_mode: u32) -> Value {
        let zero = self.b().u32(0);
        let one = self.b().u32(1);
        let (magnitude, negative) = if signed {
            let negative = self.b().icmp(ICmp::SLt, v, zero);
            let negated = self.b().isub(zero, v);
            (self.b().select(negative, negated, v), Some(negative))
        } else {
            (v, None)
        };
        // find_umsb of 0 is -1: no excess bits.
        let msb = self.b().find_umsb(magnitude);
        let c23 = self.b().u32(23);
        let excess = self.b().isub(msb, c23);
        let shift = self.b().smax(excess, zero);
        let kept = self.b().lshr(magnitude, shift);
        let truncated = self.b().shl(kept, shift);
        let unit = self.b().shl(one, shift);
        let mask = self.b().isub(unit, one);
        let rem = self.b().and(magnitude, mask);
        let inexact = self.b().icmp(ICmp::Ne, rem, zero);
        let exact = self.b().utof(truncated, Type::F32);
        let toward_zero = self.b().bitcast(exact, Type::U32);
        let up = match (round_mode, negative) {
            (1, Some(n)) => {
                let p = self.b().lnot(n);
                Some(self.b().land(inexact, p))
            }
            (1, None) => Some(inexact),
            (2, Some(n)) => Some(self.b().land(inexact, n)),
            _ => None,
        };
        let r = match up {
            Some(up) => {
                let away = self.b().iadd(toward_zero, one);
                self.b().select(up, away, toward_zero)
            }
            None => toward_zero,
        };
        match negative {
            Some(n) => {
                let sm = self.b().u32(0x8000_0000);
                let signed_r = self.b().or(r, sm);
                self.b().select(n, signed_r, r)
            }
            None => r,
        }
    }

    /// Flush a denormal f32 to a zero of the same sign. Skipped under
    /// [`super::TranslateOptions::native_ftz`]: the host flushes (the
    /// unconditional flushes of the transcendental inputs included).
    pub fn flush_denorm(&mut self, v: Value) -> Value {
        if self.options.native_ftz {
            return v;
        }
        if self.not_denormal.contains(&v) {
            return v;
        }
        if self.options.rewrites & super::rewrites::FLOAT_CLASS_COMPARE != 0 {
            // |x| < 2^-126 as one float compare (exact on denormals, and
            // still true for a denormal a host flushes; false for NaN).
            let f = self.float(v);
            let mag = self.b().fabs(f);
            let lim = self.b().f32(f32::from_bits(0x0080_0000));
            let small = self.b().fcmp(FCmp::OLt, mag, lim);
            let bits = self.bits(v);
            let sm = self.b().u32(0x8000_0000);
            let sign = self.b().and(bits, sm);
            let r = self.b().select(small, sign, bits);
            return self.float(r);
        }
        let bits = self.bits(v);
        let mm = self.b().u32(0x7FFF_FFFF);
        let sm = self.b().u32(0x8000_0000);
        let mag = self.b().and(bits, mm);
        let sign = self.b().and(bits, sm);
        let lim = self.b().u32(0x0080_0000);
        let small = self.b().icmp(ICmp::ULt, mag, lim);
        let flush = self.nonzero_small(mag, small);
        let r = self.b().select(flush, sign, bits);
        self.float(r)
    }

    /// The flush condition of a value with magnitude bits `mag` below the
    /// smallest normal (`small`): a zero is already its own sign, so with
    /// [`super::rewrites::LEAN_FLUSH`] `small` alone decides (same result).
    fn nonzero_small(&mut self, mag: Value, small: Value) -> Value {
        if self.options.rewrites & super::rewrites::LEAN_FLUSH != 0 {
            return small;
        }
        let zero = self.b().u32(0);
        let nz = self.b().icmp(ICmp::Ne, mag, zero);
        self.b().land(nz, small)
    }

    /// The f32 IR computation of a unary op before special cases.
    fn unary_core(&mut self, op: Unary, x: Value) -> Value {
        match op {
            Unary::Rcp => {
                let f = self.flush_denorm(x);
                self.b().rcp(f)
            }
            Unary::RcpIflag => self.b().rcp(x),
            Unary::Rsq => {
                let f = self.flush_denorm(x);
                self.b().inv_sqrt(f)
            }
            Unary::Sqrt => {
                let f = self.flush_denorm(x);
                self.b().sqrt(f)
            }
            Unary::Log => {
                let f = self.flush_denorm(x);
                self.b().log2(f)
            }
            Unary::Exp => {
                let f = self.flush_denorm(x);
                self.b().exp2(f)
            }
            Unary::Fract => {
                let fl = self.b().floor(x);
                self.b().fsub(x, fl)
            }
            Unary::Sin | Unary::Cos => {
                // The input is in turns. The cycle x - roundEven(x) lies in
                // [-0.5, 0.5] and is exact for every finite x (it is x when
                // roundEven(x) is 0, Sterbenz's lemma otherwise); finite
                // inputs of 2^23 or more reduce to +0. Infinite and NaN
                // inputs take their results from the specials.
                let rounded = self.b().round_even(x);
                let mut cycle = self.b().fsub(x, rounded);
                let bits = self.bits(x);
                let mm = self.b().u32(0x7FFF_FFFF);
                if op == Unary::Sin {
                    let mag = self.b().and(bits, mm);
                    let zero = self.b().u32(0);
                    let is_zero = self.b().icmp(ICmp::Eq, mag, zero);
                    cycle = self.b().select(is_zero, x, cycle);
                }
                let absolute = self.b().fabs(cycle);
                let tau = self.b().f32(f32::from_bits(0x40C9_0FDB));
                let rad = self.b().fmul(absolute, tau);
                // Within 2^-12 of a turn from a zero of the function the
                // hardware is linear in the exact distance to that zero
                // (measured on RDNA2): the input times 6.2831826 at the
                // integers, the f32 2*pi times the distance at the half and
                // quarter turns.
                let near_linear = |t: &mut Self, value: Value, distance: Value, k: Value| {
                    let linear = t.b().fmul(distance, k);
                    let db = t.bits(distance);
                    let dm = t.b().and(db, mm);
                    let lim = t.b().u32(0x3980_0000);
                    let near = t.b().icmp(ICmp::ULt, dm, lim);
                    t.b().select(near, linear, value)
                };
                let series = self.options.float_rules & super::float_rules::SIN_COS_SERIES != 0;
                if op == Unary::Sin {
                    let value = self.b().sin(rad);
                    let quarter = self.b().f32(0.25);
                    let near_half = self.b().fcmp(FCmp::OGt, absolute, quarter);
                    let half = self.b().f32(0.5);
                    let to_half = self.b().fsub(half, absolute);
                    let distance = self.b().select(near_half, to_half, absolute);
                    let value = if series {
                        // sin(2*pi*d) for d in [0, 1/4] in turns: the sine
                        // series to 1/8, the cosine series of 1/4 - d past.
                        let eighth = self.b().f32(0.125);
                        let past_eighth = self.b().fcmp(FCmp::OGt, distance, eighth);
                        let rest = self.b().fsub(quarter, distance);
                        let cos_part = self.cos_turn_series(rest);
                        let sin_part = self.sin_turn_series(distance);
                        let s = self.b().select(past_eighth, cos_part, sin_part);
                        self.series_when_far(value, s)
                    } else {
                        value
                    };
                    let at_integer = self.b().f32(f32::from_bits(0x40C9_0FD5));
                    let k = self.b().select(near_half, tau, at_integer);
                    let r = near_linear(self, value, distance, k);
                    let cb = self.bits(cycle);
                    let sm = self.b().u32(0x8000_0000);
                    let sign = self.b().and(cb, sm);
                    let rb = self.bits(r);
                    let signed = self.b().or(rb, sign);
                    self.float(signed)
                } else {
                    let value = self.b().cos(rad);
                    let quarter = self.b().f32(0.25);
                    let distance = self.b().fsub(quarter, absolute);
                    let value = if series {
                        // cos(2*pi*a) for a in [0, 1/2] in turns: the cosine
                        // series to 1/8, the sine series of 1/4 - a to 3/8,
                        // minus the cosine series of 1/2 - a past.
                        let low = self.cos_turn_series(absolute);
                        let middle = self.sin_turn_series(distance);
                        let half = self.b().f32(0.5);
                        let to_half = self.b().fsub(half, absolute);
                        let high = self.cos_turn_series(to_half);
                        let high = self.b().fneg(high);
                        let eighth = self.b().f32(0.125);
                        let three_eighths = self.b().f32(0.375);
                        let past_eighth = self.b().fcmp(FCmp::OGt, absolute, eighth);
                        let past_three = self.b().fcmp(FCmp::OGt, absolute, three_eighths);
                        let s = self.b().select(past_eighth, middle, low);
                        let s = self.b().select(past_three, high, s);
                        self.series_when_far(value, s)
                    } else {
                        value
                    };
                    near_linear(self, value, distance, tau)
                }
            }
            Unary::Trunc => self.b().trunc(x),
            Unary::Ceil => self.b().ceil(x),
            Unary::Floor => self.b().floor(x),
            Unary::Rndne => self.b().round_even(x),
        }
    }

    /// sin(2*pi*t) for t in [0, 1/8] turns (odd polynomial, f32 fma).
    fn sin_turn_series(&mut self, t: Value) -> Value {
        let square = self.b().fmul(t, t);
        let mut p = self.b().f32(f32::from_bits(0xC171_83A8));
        for k in [0x4228_3C1Au32, 0xC299_6966, 0x42A3_35E3, 0xC225_5DE7, 0x40C9_0FDB] {
            let c = self.b().f32(f32::from_bits(k));
            p = self.b().fma(p, square, c);
        }
        self.b().fmul(t, p)
    }

    /// cos(2*pi*t) for t in [0, 1/8] turns (even polynomial, f32 fma).
    fn cos_turn_series(&mut self, t: Value) -> Value {
        let square = self.b().fmul(t, t);
        let mut p = self.b().f32(f32::from_bits(0x40FC_E9C5));
        for k in [0xC1D3_68F9u32, 0x4270_FA83, 0xC2AA_E9E4, 0x4281_E0F8, 0xC19D_E9E6, 0x3F80_0000] {
            let c = self.b().f32(f32::from_bits(k));
            p = self.b().fma(p, square, c);
        }
        p
    }

    /// `series` where the host's `value` is more than 4 ulp from it (a
    /// driver sine / cosine that loses precision), `value` otherwise.
    fn series_when_far(&mut self, value: Value, series: Value) -> Value {
        let vb = self.bits(value);
        let sb = self.bits(series);
        let up = self.b().isub(vb, sb);
        let down = self.b().isub(sb, vb);
        let distance = self.b().umin(up, down);
        let four = self.b().u32(4);
        let far = self.b().icmp(ICmp::UGt, distance, four);
        self.b().select(far, series, value)
    }

    /// Special-case results: returns (result f32, invalid bool).
    fn unary_specials(&mut self, op: Unary, x: Value, result: Value) -> (Value, Value) {
        let bits = self.bits(x);
        let mm = self.b().u32(0x7FFF_FFFF);
        let sm = self.b().u32(0x8000_0000);
        let mag = self.b().and(bits, mm);
        let sign = self.b().and(bits, sm);
        let flushes = matches!(op, Unary::Rcp | Unary::RcpIflag | Unary::Rsq | Unary::Sqrt | Unary::Log | Unary::Exp);
        let zero_u = self.b().u32(0);
        let zero = if flushes {
            let lim = self.b().u32(0x0080_0000);
            self.b().icmp(ICmp::ULt, mag, lim)
        } else {
            self.b().icmp(ICmp::Eq, mag, zero_u)
        };
        let inf_bits = self.b().u32(0x7F80_0000);
        let infinite = self.b().icmp(ICmp::Eq, mag, inf_bits);
        let has_sign = self.b().icmp(ICmp::Ne, sign, zero_u);
        let not_zero = self.b().lnot(zero);
        let negative = self.b().land(has_sign, not_zero);
        let signed_inf = self.b().or(sign, inf_bits);
        let mut value = self.bits(result);
        let mut invalid = self.b().bool(false);
        let pick = |t: &mut Self, c: Value, v: Value, cur: Value| t.b().select(c, v, cur);
        match op {
            Unary::Rcp | Unary::RcpIflag => {
                value = pick(self, infinite, sign, value);
                value = pick(self, zero, signed_inf, value);
            }
            Unary::Rsq => {
                value = pick(self, infinite, zero_u, value);
                value = pick(self, zero, signed_inf, value);
                invalid = negative;
            }
            Unary::Sqrt => {
                value = pick(self, infinite, bits, value);
                value = pick(self, zero, sign, value);
                invalid = negative;
            }
            Unary::Log => {
                value = pick(self, infinite, bits, value);
                let one = self.b().u32(0x3F80_0000);
                let is_one = self.b().icmp(ICmp::Eq, bits, one);
                value = pick(self, is_one, zero_u, value);
                let ninf = self.b().u32(0xFF80_0000);
                value = pick(self, zero, ninf, value);
                invalid = negative;
            }
            Unary::Exp => {
                let pos = self.b().icmp(ICmp::Eq, sign, zero_u);
                let big = self.b().u32(0x42FF_FFFF);
                let over = self.b().icmp(ICmp::UGt, mag, big);
                let overflow = self.b().land(pos, over);
                let small = self.b().u32(0x42FC_0000);
                let under = self.b().icmp(ICmp::UGt, mag, small);
                let underflow = self.b().land(negative, under);
                value = pick(self, overflow, inf_bits, value);
                // From -126 up the result is at least the smallest normal.
                let min_normal = self.b().u32(0x0080_0000);
                value = self.b().umax(value, min_normal);
                value = pick(self, underflow, zero_u, value);
                let one = self.b().u32(0x3F80_0000);
                value = pick(self, zero, one, value);
            }
            Unary::Fract => {
                let unsigned = self.b().and(value, mm);
                let lim = self.b().u32(0x3F7F_FFFF);
                let above = self.b().icmp(ICmp::UGt, unsigned, lim);
                value = pick(self, above, lim, unsigned);
                invalid = infinite;
            }
            Unary::Sin | Unary::Cos => {
                let whole = self.b().trunc(x);
                let diff = self.b().fsub(x, whole);
                let frac = self.b().fabs(diff);
                let is = |t: &mut Self, c: f32| {
                    let k = t.b().f32(c);
                    t.b().fcmp(FCmp::OEq, frac, k)
                };
                if op == Unary::Sin {
                    let a = is(self, 0.0);
                    let b = is(self, 0.5);
                    let cardinal = self.b().lor(a, b);
                    value = pick(self, cardinal, zero_u, value);
                    value = pick(self, zero, bits, value);
                } else {
                    let a = is(self, 0.25);
                    let b = is(self, 0.75);
                    let q = self.b().lor(a, b);
                    value = pick(self, q, zero_u, value);
                }
                invalid = infinite;
            }
            _ => {}
        }
        let r = self.float(value);
        (r, invalid)
    }

    /// Full f32 unary op: core, specials, NaN quieting, invalid -> NaN.
    pub fn float_unary(&mut self, op: Unary, x: Value) -> Value {
        if self.options.fast_transcendental && matches!(op, Unary::Rcp | Unary::Rsq | Unary::Sqrt | Unary::Log | Unary::Exp) {
            return self.fast_unary(op, x);
        }
        let core = self.unary_core(op, x);
        let (result, invalid) = self.unary_specials(op, x, core);
        let bits = self.bits(x);
        let nan = self.is_nan_bits(bits);
        let qb = self.nan_quiet_bit(false);
        let quiet = self.b().or(bits, qb);
        let default_nan = self.b().u32(0xFFC0_0000);
        let special = self.b().select(nan, quiet, default_nan);
        let special = self.float(special);
        let bad = self.b().lor(nan, invalid);
        self.b().select(bad, special, result)
    }

    /// [`TranslateOptions::fast_transcendental`](super::TranslateOptions):
    /// the host's operation on the flushed input; a NaN input returns
    /// itself (quieted as the full form does), every other special value
    /// and range is the host's.
    fn fast_unary(&mut self, op: Unary, x: Value) -> Value {
        let core = self.unary_core(op, x);
        let bits = self.bits(x);
        let mm = self.b().u32(0x7FFF_FFFF);
        let mag = self.b().and(bits, mm);
        let inf = self.b().u32(0x7F80_0000);
        let nan = self.b().icmp(ICmp::UGt, mag, inf);
        let qb = self.nan_quiet_bit(false);
        let quiet = self.b().or(bits, qb);
        let quiet = self.float(quiet);
        self.b().select(nan, quiet, core)
    }

    /// Minimum / maximum of three with hardware NaN handling.
    fn min3(&mut self, a: Value, b: Value, c: Value, max: bool) -> Value {
        if max {
            let ab = self.b().fmax(a, b);
            self.b().fmax(ab, c)
        } else {
            let ab = self.b().fmin(a, b);
            self.b().fmin(ab, c)
        }
    }

    fn med3(&mut self, a: Value, b: Value, c: Value) -> Value {
        let min_ab = self.b().fmin(a, b);
        let min3 = self.b().fmin(min_ab, c);
        let max_ab = self.b().fmax(a, b);
        let high_min = self.b().fmin(max_ab, c);
        let median = self.b().fmax(min_ab, high_min);
        let na = self.b().fcmp(FCmp::Uno, a, a);
        let nb = self.b().fcmp(FCmp::Uno, b, b);
        let nc = self.b().fcmp(FCmp::Uno, c, c);
        let n = self.b().lor(na, nb);
        let n = self.b().lor(n, nc);
        self.b().select(n, min3, median)
    }

    /// The median of v_med3_f32 (no NaN among the operands), as measured on
    /// RDNA2: the operand equal to the numeric maximum of the three (+0 and
    /// -0 compare equal) is dropped and the larger of the other two is the
    /// result, so med3(-0, +0, -0) is +0. The comparisons see the operands
    /// before the input denormal flush (`raw`): med3(-0, +denormal,
    /// -denormal) with flushed denormals is -0.
    fn med3_f32_median(&mut self, raw: [Value; 3]) -> Value {
        let [a, b, c] = raw;
        let max_ab = self.b().fmax(a, b);
        let max3 = self.b().fmax(max_ab, c);
        let max_bc = self.b().fmax(b, c);
        let max_ac = self.b().fmax(a, c);
        let a_is_max = self.b().fcmp(FCmp::OEq, max3, a);
        let b_is_max = self.b().fcmp(FCmp::OEq, max3, b);
        let not_a = self.b().select(b_is_max, max_ac, max_ab);
        let median = self.b().select(a_is_max, max_bc, not_a);
        if self.f32_flush & 1 != 0 {
            self.flush_denorm(median)
        } else {
            median
        }
    }

    /// f32 to f16 bits (in a u32) with the given rounding.
    pub fn f32_to_half_bits(&mut self, v: Value, r: Rounding) -> Value {
        let h = self.b().fconvert(v, Type::F16, r);
        let u = self.b().bitcast(h, Type::U16);
        self.b().uconvert(u, Type::U32)
    }

    /// The f16 bits of a conversion result (v_cvt_f16_f32, v_cvt_f16_u16,
    /// v_cvt_f16_i16): FP16_OVFL, the clamp modifier, rounding; with f16
    /// result denormals flushed (FLOAT_MODE bit 7 clear) a value that
    /// rounds below the smallest normal is a zero of its sign (measured on
    /// RDNA2).
    pub(super) fn f16_conversion_bits(&mut self, v: Value, clamp: bool, sources: &[Value]) -> Value {
        if self.f16_round_mode() != 0 {
            let modified = self.f16_clamp(v, clamp);
            let bits = self.f32_to_f16_bits_rounded(modified, sources);
            return self.flush_tiny_f16_conversion(modified, bits);
        }
        let v = self.clamp_f16_overflow(v, sources);
        let v = self.f16_clamp(v, clamp);
        let bits = self.f32_to_half_bits(v, Rounding::Rte);
        self.flush_tiny_f16_conversion(v, bits)
    }

    /// f16 bits (in a u32) of `v` in the f16 round mode, FP16_OVFL applied
    /// (`sources` as for [`Self::clamp_f16_overflow`]). The directed modes
    /// truncate the f32 bits to the f16 grid (normal or denormal) and step
    /// one unit away from zero when bits were discarded and the mode rounds
    /// away for the value's sign; a finite value beyond the f16 range is
    /// the infinity in the modes that round toward it and the largest
    /// finite value otherwise (always with FP16_OVFL). Infinities and NaNs
    /// convert as to nearest.
    pub(super) fn f32_to_f16_bits_rounded(&mut self, v: Value, sources: &[Value]) -> Value {
        let clamped = self.clamp_f16_overflow(v, sources);
        let native = self.f32_to_half_bits(clamped, Rounding::Rte);
        let rounding = self.f16_round_mode();
        if rounding == 0 {
            return native;
        }
        let overflow_clamps = self.float_mode.fp16_overflow;
        let c = |t: &mut Self, x: u32| t.b().u32(x);
        let word = self.bits(v);
        let sixteen = c(self, 16);
        let high = self.b().lshr(word, sixteen);
        let sm = c(self, 0x8000);
        let sign = self.b().and(high, sm);
        let mm = c(self, 0x7FFF_FFFF);
        let magnitude = self.b().and(word, mm);
        let c23 = c(self, 23);
        let exponent = self.b().lshr(magnitude, c23);
        let c112 = c(self, 112);
        let normal = self.b().icmp(ICmp::UGt, exponent, c112);
        let fm = c(self, 0x7F_FFFF);
        let fraction = self.b().and(magnitude, fm);
        let hidden = c(self, 0x80_0000);
        let significand = self.b().or(fraction, hidden);
        let c126 = c(self, 126);
        let distance = self.b().isub(c126, exponent);
        let c24 = c(self, 24);
        let shift = self.b().umin(distance, c24);
        let subnormal = self.b().lshr(significand, shift);
        let c13 = c(self, 13);
        let top = self.b().lshr(magnitude, c13);
        let rebias = c(self, 0x1_C000);
        let normal_bits = self.b().isub(top, rebias);
        let truncated = self.b().select(normal, normal_bits, subnormal);
        let c103 = c(self, 103);
        let tiny = self.b().icmp(ICmp::ULt, exponent, c103);
        let low13 = c(self, 0x1FFF);
        let normal_rest = self.b().and(magnitude, low13);
        let kept = self.b().shl(subnormal, shift);
        let subnormal_rest = self.b().isub(significand, kept);
        let small_rest = self.b().select(tiny, magnitude, subnormal_rest);
        let discarded = self.b().select(normal, normal_rest, small_rest);
        let zero = c(self, 0);
        let away = match rounding {
            1 => self.b().icmp(ICmp::Eq, sign, zero),
            2 => self.b().icmp(ICmp::Ne, sign, zero),
            _ => self.b().bool(false),
        };
        let inexact = self.b().icmp(ICmp::Ne, discarded, zero);
        let increment = self.b().land(away, inexact);
        let one = c(self, 1);
        let step = self.b().select(increment, one, zero);
        let rounded = self.b().iadd(truncated, step);
        let largest = c(self, 0x7BFF);
        let infinity = c(self, 0x7C00);
        let overflow = if overflow_clamps { largest } else { self.b().select(away, infinity, largest) };
        let c142 = c(self, 142);
        let beyond = self.b().icmp(ICmp::UGt, exponent, c142);
        let mut result = self.b().select(beyond, overflow, rounded);
        if overflow_clamps {
            let to_infinity = self.b().icmp(ICmp::Eq, result, infinity);
            result = self.b().select(to_infinity, largest, result);
        }
        let result = self.b().or(sign, result);
        let special = c(self, 0x7F80_0000);
        let finite = self.b().icmp(ICmp::ULt, magnitude, special);
        self.b().select(finite, result, native)
    }

    /// With f16 result denormals flushed, the f16 conversion `bits` of `v`
    /// is a zero of `v`'s sign when the rounded result is below the
    /// smallest normal: below 0x387ff000 to nearest even, 2^-14 toward
    /// zero and toward the infinity of the other sign, above the largest
    /// denormal toward the infinity of the value's sign.
    pub(super) fn flush_tiny_f16_conversion(&mut self, v: Value, bits: Value) -> Value {
        if !self.f16_flushes(0x80) {
            return bits;
        }
        let rounding = self.f16_round_mode();
        let word = self.bits(v);
        let mm = self.b().u32(0x7FFF_FFFF);
        let mag = self.b().and(word, mm);
        let sixteen = self.b().u32(16);
        let high = self.b().lshr(word, sixteen);
        let sm = self.b().u32(0x8000);
        let sign = self.b().and(high, sm);
        let limit = match rounding {
            0 => self.b().u32(0x387F_F000),
            1 | 2 => {
                let away_sign = self.b().u32(if rounding == 1 { 0 } else { 0x8000 });
                let away = self.b().icmp(ICmp::Eq, sign, away_sign);
                let above_denormal = self.b().u32(0x387F_E001);
                let normal = self.b().u32(0x3880_0000);
                self.b().select(away, above_denormal, normal)
            }
            _ => self.b().u32(0x3880_0000),
        };
        let tiny = self.b().icmp(ICmp::ULt, mag, limit);
        self.b().select(tiny, sign, bits)
    }

    /// The f16/f64 round mode (FLOAT_MODE[3:2]: 0 nearest even, 1 toward
    /// +infinity, 2 toward -infinity, 3 toward zero).
    pub(super) fn f16_round_mode(&self) -> u32 {
        if self.options.float_rules & super::float_rules::F16_ROUNDING == 0 {
            return 0;
        }
        self.f16_round_field()
    }

    /// FLOAT_MODE[3:2], whatever the float rules.
    pub(super) fn f16_round_field(&self) -> u32 {
        (self.float_mode.mode as u32 >> 2) & 3
    }

    /// f16 result modifiers (clamp in the f32 domain; NaN -> 0).
    fn f16_clamp(&mut self, v: Value, clamp: bool) -> Value {
        if !clamp {
            return v;
        }
        let zero = self.b().f32(0.0);
        let positive = self.b().fcmp(FCmp::OGt, v, zero);
        let one = self.b().f32(1.0);
        let lim = self.b().fmin(v, one);
        let r = self.b().select(positive, lim, zero);
        if self.float_mode.dx10_clamp {
            return r;
        }
        let nan = self.b().fcmp(FCmp::Uno, v, v);
        self.b().select(nan, v, r)
    }

    /// IEEE_MODE minimum / maximum pair: a signaling NaN operand is
    /// returned quieted (the first one first), otherwise the plain pair.
    fn ieee_pair(&mut self, a: Value, b: Value, max: bool) -> Value {
        let plain = if max { self.b().fmax(a, b) } else { self.b().fmin(a, b) };
        let rb = self.quiet_if_signaling(b, plain);
        self.quiet_if_signaling(a, rb)
    }

    fn quiet_if_signaling(&mut self, x: Value, otherwise: Value) -> Value {
        let bits = self.bits(x);
        let em = self.b().u32(0x7FC0_0000);
        let e = self.b().and(bits, em);
        let snan_exp = self.b().u32(0x7F80_0000);
        let is_e = self.b().icmp(ICmp::Eq, e, snan_exp);
        let pm = self.b().u32(0x003F_FFFF);
        let p = self.b().and(bits, pm);
        let zero = self.b().u32(0);
        let has_p = self.b().icmp(ICmp::Ne, p, zero);
        let signaling = self.b().land(is_e, has_p);
        let q = self.b().u32(0x0040_0000);
        let quiet = self.b().or(bits, q);
        let quiet = self.float(quiet);
        self.b().select(signaling, quiet, otherwise)
    }

    /// FP16_OVFL: an f16 result that would round to infinity is the largest
    /// finite value of its sign instead, unless a source is infinite
    /// (measured on RDNA2 at IEEE=0 and IEEE=1, f16 denormals kept and
    /// flushed). `v` is the f32 value before the f16 conversion.
    pub(super) fn clamp_f16_overflow(&mut self, v: Value, sources: &[Value]) -> Value {
        if !self.float_mode.fp16_overflow {
            return v;
        }
        let bits = self.bits(v);
        let mm = self.b().u32(0x7FFF_FFFF);
        let mag = self.b().and(bits, mm);
        let lo = self.b().u32(0x477F_F000);
        let hi = self.b().u32(0x7F80_0001);
        let at_least = self.b().icmp(ICmp::UGe, mag, lo);
        let below = self.b().icmp(ICmp::ULt, mag, hi);
        let mut overflow = self.b().land(at_least, below);
        for &source in sources {
            let sb = self.bits(source);
            let sm = self.b().and(sb, mm);
            let inf = self.b().u32(0x7F80_0000);
            let infinite = self.b().icmp(ICmp::Eq, sm, inf);
            let finite = self.b().lnot(infinite);
            overflow = self.b().land(overflow, finite);
        }
        let sign_mask = self.b().u32(0x8000_0000);
        let sign = self.b().and(bits, sign_mask);
        let largest = self.b().u32(0x477F_E000);
        let largest = self.b().or(sign, largest);
        let r = self.b().select(overflow, largest, bits);
        self.float(r)
    }

    /// v_min/max/min3/max3/med3_f16: NaN results are picked on the bits (a
    /// NaN operand gives the other one; a signaling NaN is quieted and
    /// returned only with IEEE_MODE), measured on RDNA2.
    fn min_max_f16(&mut self, ins: &Instruction, m: &Mods) -> Result<()> {
        use Opcode::*;
        let count = if matches!(ins.opcode, VMinF16 | VMaxF16) { 2 } else { 3 };
        let mut args = Vec::new();
        // With f16 input denormals flushed (FLOAT_MODE bit 6 clear) a
        // denormal operand is a zero of its sign (hardware rows, RDNA2).
        let flush_inputs = self.f16_denormal_rules() && self.float_mode.mode & 0x40 == 0;
        for i in 0..count {
            let mut value = self.src_f16(ins, m, i);
            let mut bits = self.src_f16_bits(ins, m, i);
            if flush_inputs {
                let mm = self.b().u32(0x7FFF);
                let mag = self.b().and(bits, mm);
                let lim = self.b().u32(0x0400);
                let tiny = self.b().icmp(ICmp::ULt, mag, lim);
                let sm = self.b().u32(0x8000);
                let sign = self.b().and(bits, sm);
                bits = self.b().select(tiny, sign, bits);
                let sixteen = self.b().u32(16);
                let wide = self.b().shl(sign, sixteen);
                let zero = self.float(wide);
                value = self.b().select(tiny, zero, value);
            }
            args.push((value, bits));
        }
        let ieee = self.float_mode.ieee;
        let is_nan = |t: &mut Self, bits: Value| {
            let mm = t.b().u32(0x7FFF);
            let mag = t.b().and(bits, mm);
            let inf = t.b().u32(0x7C00);
            t.b().icmp(ICmp::UGt, mag, inf)
        };
        let is_snan = |t: &mut Self, bits: Value| {
            let nan = is_nan(t, bits);
            let qb = t.b().u32(0x200);
            let q = t.b().and(bits, qb);
            let zero = t.b().u32(0);
            let signaling = t.b().icmp(ICmp::Eq, q, zero);
            t.b().land(nan, signaling)
        };
        let pair = |t: &mut Self, max: bool, l: (Value, Value), r: (Value, Value)| -> (Value, Value) {
            let value = if max { t.b().fmax(l.0, r.0) } else { t.b().fmin(l.0, r.0) };
            let vb = t.f32_to_half_bits(value, Rounding::Rte);
            let ln = is_nan(t, l.1);
            let mut bits = t.b().select(ln, r.1, vb);
            let rn = is_nan(t, r.1);
            bits = t.b().select(rn, l.1, bits);
            if ieee {
                let qb = t.b().u32(0x200);
                let rs = is_snan(t, r.1);
                let rq = t.b().or(r.1, qb);
                bits = t.b().select(rs, rq, bits);
                let ls = is_snan(t, l.1);
                let lq = t.b().or(l.1, qb);
                bits = t.b().select(ls, lq, bits);
            }
            (value, bits)
        };
        let result = match ins.opcode {
            VMinF16 | VMaxF16 => pair(self, ins.opcode == VMaxF16, args[0], args[1]).1,
            VMin3F16 | VMax3F16 => {
                let max = ins.opcode == VMax3F16;
                let ab = pair(self, max, args[0], args[1]);
                pair(self, max, ab, args[2]).1
            }
            _ => {
                let n0 = is_nan(self, args[0].1);
                let n1 = is_nan(self, args[1].1);
                let n2 = is_nan(self, args[2].1);
                let any = self.b().lor(n0, n1);
                let any = self.b().lor(any, n2);
                let ab = pair(self, false, args[0], args[1]);
                let minimum = pair(self, false, ab, args[2]).1;
                let median = self.med3(args[0].0, args[1].0, args[2].0);
                let mb = self.f32_to_half_bits(median, Rounding::Rte);
                self.b().select(any, minimum, mb)
            }
        };
        let r = self.clamp_f16_bits(result, m.clamp);
        let d = self.vdst(ins)?;
        self.write_vgpr_half(d, r, m.op_sel(3), &DstMods::of(m));
        Ok(())
    }

    /// Write an f16 result (from f32) into the destination half selected by
    /// VOP3 op_sel bit 3 (low otherwise), preserving the other half.
    /// `sources` are the operands (as f32) the FP16_OVFL rule looks at.
    fn write_f16(&mut self, ins: &Instruction, m: &Mods, v: Value, sources: &[Value]) -> Result<()> {
        self.write_f16_checked(ins, m, v, sources, None)
    }

    /// [`Self::write_f16`]; with `invalid` (an invalid operation, see
    /// [`Self::invalid_operation_f32`]) a NaN result is the default NaN
    /// 0xfe00 whatever NaN the host operation gave.
    fn write_f16_checked(&mut self, ins: &Instruction, m: &Mods, v: Value, sources: &[Value], invalid: Option<Value>) -> Result<()> {
        let v = self.clamp_f16_overflow(v, sources);
        let v = self.f16_clamp(v, m.clamp);
        let mut bits = self.f32_to_half_bits(v, Rounding::Rte);
        if let Some(invalid) = invalid {
            bits = self.default_nan_f16_on_invalid(bits, invalid);
        }
        let d = self.vdst(ins)?;
        self.write_vgpr_half(d, bits, m.op_sel(3), &DstMods::of(m));
        Ok(())
    }

    /// f16 unary: compute in f32, round, apply f16 special cases.
    fn float16_unary(&mut self, ins: &Instruction, m: &Mods, op: Unary) -> Result<()> {
        // The f16 denormal mode flushes the input (bit 6 clear) and the
        // result (bit 7 clear), measured on RDNA2.
        let x = self.src_f16(ins, m, 0);
        let x = self.flush_f16_input_value(x);
        let core = self.unary_core(op, x);
        let (result, invalid) = self.unary_specials(op, x, core);
        let result = self.clamp_f16_overflow(result, &[x]);
        let clamped = self.f16_clamp(result, m.clamp);
        let mut bits = self.f32_to_half_bits(clamped, Rounding::Rte);
        if op == Unary::Fract {
            let one = self.b().u32(0x3C00);
            let is_one = self.b().icmp(ICmp::Eq, bits, one);
            let below = self.b().u32(0x3BFF);
            bits = self.b().select(is_one, below, bits);
        }
        let bits = self.flush_f16_bits(bits, 0x80);
        let half = self.src_f16_bits(ins, m, 0);
        let mm = self.b().u32(0x7FFF);
        let mag = self.b().and(half, mm);
        let inf = self.b().u32(0x7C00);
        let nan = self.b().icmp(ICmp::UGt, mag, inf);
        let qb = self.nan_quiet_bit(true);
        let quiet = self.b().or(half, qb);
        let dn = self.b().u32(0xFE00);
        let special = self.b().select(nan, quiet, dn);
        let special = if m.clamp && self.float_mode.dx10_clamp { self.b().u32(0) } else { special };
        let bad = self.b().lor(nan, invalid);
        let value = self.b().select(bad, special, bits);
        let d = self.vdst(ins)?;
        self.write_vgpr_half(d, value, m.op_sel(3), &DstMods::of(m));
        Ok(())
    }

    /// Saturating f32 -> u32 conversion (truncating).
    pub fn f32_to_u32_sat(&mut self, v: Value, upper: f32, safe_upper: f32, high: u32) -> Value {
        let zero = self.b().f32(0.0);
        let nan = self.b().fcmp(FCmp::Uno, v, v);
        let low = self.b().fcmp(FCmp::OLe, v, zero);
        let up = self.b().f32(upper);
        let hi = self.b().fcmp(FCmp::OGe, v, up);
        let t = self.b().trunc(v);
        let nl = self.b().lor(nan, low);
        let safe_low = self.b().select(nl, zero, t);
        let su = self.b().f32(safe_upper);
        let safe = self.b().select(hi, su, safe_low);
        let conv = self.b().ftou(safe, Type::U32);
        let h = self.b().u32(high);
        self.b().select(hi, h, conv)
    }

    /// Saturating f32 -> i32 conversion (truncating).
    pub fn f32_to_i32_sat(&mut self, v: Value, lower: f32, upper: f32, safe_upper: f32, low_result: u32, high_result: u32) -> Value {
        let nan = self.b().fcmp(FCmp::Uno, v, v);
        let lo = self.b().f32(lower);
        let low = self.b().fcmp(FCmp::OLe, v, lo);
        let up = self.b().f32(upper);
        let high = self.b().fcmp(FCmp::OGe, v, up);
        let t = self.b().trunc(v);
        let safe_low = self.b().select(low, lo, t);
        let su = self.b().f32(safe_upper);
        let safe_high = self.b().select(high, su, safe_low);
        let zf = self.b().f32(0.0);
        let safe = self.b().select(nan, zf, safe_high);
        let conv = self.b().ftos(safe, Type::U32);
        let hr = self.b().u32(high_result);
        let lr = self.b().u32(low_result);
        let c1 = self.b().select(high, hr, conv);
        let c2 = self.b().select(low, lr, c1);
        let zero = self.b().u32(0);
        self.b().select(nan, zero, c2)
    }

    /// Floored conversion with NaN giving 0x7fffffff + sign.
    fn floored_to_i32(&mut self, source: Value, floored: Value) -> Value {
        let conv = self.f32_to_i32_sat(floored, -2147483648.0, 2147483648.0, 2147483520.0, 0x8000_0000, 0x7FFF_FFFF);
        let nan = self.b().fcmp(FCmp::Uno, source, source);
        let bits = self.bits(source);
        let c31 = self.b().u32(31);
        let sign = self.b().lshr(bits, c31);
        let mx = self.b().u32(0x7FFF_FFFF);
        let nr = self.b().iadd(mx, sign);
        self.b().select(nan, nr, conv)
    }

    /// f32 -> 16-bit normalized integer, round to nearest even.
    fn norm_f32(&mut self, bits: Value, signed: bool) -> Value {
        let c = |t: &mut Self, x: u32| t.b().u32(x);
        let mm = c(self, 0x7FFF_FFFF);
        let mag = self.b().and(bits, mm);
        let sm = c(self, 0x8000_0000);
        let sb = self.b().and(bits, sm);
        let zero = c(self, 0);
        let negative = self.b().icmp(ICmp::Ne, sb, zero);
        let inf = c(self, 0x7F80_0000);
        let nan = self.b().icmp(ICmp::UGt, mag, inf);
        let one = c(self, 0x3F80_0000);
        let saturated = self.b().icmp(ICmp::UGe, mag, one);
        let scale = if signed { 32767u32 } else { 65535 };
        if self.options.float_rules & super::float_rules::PKNORM_F32 != 0 {
            // p = |x| * scale rounded to f32 is the exact product rounded
            // once, so roundEven(p) is the answer unless p is exactly
            // k + 0.5, where the sign of the exact error decides. scale + 1
            // is a power of two: A = |x| * (scale + 1) is exact, p = A - |x|
            // and the error (A - p) - |x| come from correctly rounded
            // subtractions only (Fast2Sum, A >= |x|). Lanes that select a
            // constant (NaN, |x| >= 1) multiply 0.
            let special = self.b().lor(saturated, nan);
            let kept = self.b().select(special, zero, mag);
            let absolute = self.float(kept);
            let factor = self.b().f32((scale + 1) as f32);
            let scaled = self.b().fmul(absolute, factor);
            let product = self.b().fsub(scaled, absolute);
            let back = self.b().fsub(scaled, product);
            let error = self.b().fsub(back, absolute);
            let floored = self.b().floor(product);
            let fraction = self.b().fsub(product, floored);
            let half = self.b().f32(0.5);
            let tie = self.b().fcmp(FCmp::OEq, fraction, half);
            let fzero = self.b().f32(0.0);
            let below = self.b().fcmp(FCmp::OLt, error, fzero);
            let above = self.b().fcmp(FCmp::OGt, error, fzero);
            let tie_down = self.b().land(tie, below);
            let tie_up = self.b().land(tie, above);
            let nearest = self.b().round_even(product);
            let rounded = self.b().select(tie_down, floored, nearest);
            let fone = self.b().f32(1.0);
            let next = self.b().fadd(floored, fone);
            let rounded = self.b().select(tie_up, next, rounded);
            let integer = self.b().ftou(rounded, Type::U32);
            let sc = c(self, scale);
            let mut value = self.b().select(saturated, sc, integer);
            if signed {
                let n = self.b().ineg(value);
                value = self.b().select(negative, n, value);
            } else {
                value = self.b().select(negative, zero, value);
            }
            let value = self.b().select(nan, zero, value);
            let ffff = c(self, 0xFFFF);
            return self.b().and(value, ffff);
        }
        let c23 = c(self, 23);
        let exponent = self.b().lshr(mag, c23);
        let mmask = c(self, 0x7F_FFFF);
        let mant = self.b().and(mag, mmask);
        let hidden = c(self, 0x80_0000);
        let mantissa = self.b().or(mant, hidden);
        let c149 = c(self, 149);
        let sh = self.b().isub(c149, exponent);
        let c63 = c(self, 63);
        let shift = self.b().umin(sh, c63);
        let m64 = self.b().uconvert(mantissa, Type::U64);
        let s64 = self.b().u64(scale as u64);
        let product = self.b().imul(m64, s64);
        let shift64 = self.b().uconvert(shift, Type::U64);
        let halves = self.b().lshr(product, shift64);
        let back = self.b().shl(halves, shift64);
        let sticky = self.b().icmp(ICmp::Ne, back, product);
        let (doubled, _) = self.b().split_u64(halves);
        let one_u = c(self, 1);
        let truncated = self.b().lshr(doubled, one_u);
        let low_bit = self.b().and(doubled, one_u);
        let has_half = self.b().icmp(ICmp::Ne, low_bit, zero);
        let tb = self.b().and(truncated, one_u);
        let odd = self.b().icmp(ICmp::Ne, tb, zero);
        let so = self.b().lor(sticky, odd);
        let round_up = self.b().land(has_half, so);
        let inc = self.b().bool_to_u32(round_up);
        let rounded = self.b().iadd(truncated, inc);
        let sc = c(self, scale);
        let mut value = self.b().select(saturated, sc, rounded);
        if signed {
            let n = self.b().ineg(value);
            value = self.b().select(negative, n, value);
        } else {
            value = self.b().select(negative, zero, value);
        }
        let value = self.b().select(nan, zero, value);
        let ffff = c(self, 0xFFFF);
        self.b().and(value, ffff)
    }

    fn cube(&mut self, ins: &Instruction, m: &Mods, kind: u32) -> Result<()> {
        let x = self.src_f32(ins, m, 0);
        let y = self.src_f32(ins, m, 1);
        let z = self.src_f32(ins, m, 2);
        let neg = |t: &mut Self, v: Value| t.b().fneg(v);
        let exp_zero = |t: &mut Self, v: Value| {
            let b = t.bits(v);
            let em = t.b().u32(0x7F80_0000);
            let e = t.b().and(b, em);
            let zero = t.b().u32(0);
            t.b().icmp(ICmp::Eq, e, zero)
        };
        let flushed = |t: &mut Self, v: Value| {
            let ez = exp_zero(t, v);
            let b = t.bits(v);
            let zero = t.b().u32(0);
            let r = t.b().select(ez, zero, b);
            t.float(r)
        };
        let fx = flushed(self, x);
        let fy = flushed(self, y);
        let fz = flushed(self, z);
        let ax = self.b().fabs(fx);
        let ay = self.b().fabs(fy);
        let az = self.b().fabs(fz);
        let zx = self.b().fcmp(FCmp::OGe, az, ax);
        let zy = self.b().fcmp(FCmp::OGe, az, ay);
        let z_face = self.b().land(zx, zy);
        let y_face = self.b().fcmp(FCmp::OGe, ay, ax);
        let zero = self.b().f32(0.0);
        let xn = self.b().fcmp(FCmp::OLt, fx, zero);
        let yn = self.b().fcmp(FCmp::OLt, fy, zero);
        let zn = self.b().fcmp(FCmp::OLt, fz, zero);
        let face = |t: &mut Self, xv: Value, yv: Value, zv: Value| {
            let s = t.b().select(y_face, yv, xv);
            t.b().select(z_face, zv, s)
        };
        let r = match kind {
            0 => {
                let k = |t: &mut Self, c: f32| t.b().f32(c);
                let (k0, k1, k2, k3, k4, k5) = (k(self, 0.0), k(self, 1.0), k(self, 2.0), k(self, 3.0), k(self, 4.0), k(self, 5.0));
                let xr = self.b().select(xn, k1, k0);
                let yr = self.b().select(yn, k3, k2);
                let zr = self.b().select(zn, k5, k4);
                face(self, xr, yr, zr)
            }
            1 => {
                let nz = neg(self, z);
                let nx = neg(self, x);
                let xr = self.b().select(xn, z, nz);
                let zr = self.b().select(zn, nx, x);
                face(self, xr, x, zr)
            }
            2 => {
                let ny = neg(self, y);
                let nz = neg(self, z);
                let yr = self.b().select(yn, nz, z);
                face(self, ny, yr, ny)
            }
            _ => {
                let major = face(self, x, y, z);
                let two = self.b().f32(2.0);
                let twice = self.b().fmul(major, two);
                let mb = self.bits(major);
                let mm = self.b().u32(0x7FFF_FFFF);
                let mag = self.b().and(mb, mm);
                let inf = self.b().u32(0x7F80_0000);
                let nan = self.b().icmp(ICmp::UGt, mag, inf);
                let ez = exp_zero(self, major);
                let t = self.b().select(ez, zero, twice);
                self.b().select(nan, major, t)
            }
        };
        // A NaN result is quieted only with IEEE_MODE, keeping its sign.
        let rb = self.bits(r);
        let mm = self.b().u32(0x7FFF_FFFF);
        let mag = self.b().and(rb, mm);
        let inf = self.b().u32(0x7F80_0000);
        let nan = self.b().icmp(ICmp::UGt, mag, inf);
        let qb = self.nan_quiet_bit(false);
        let quiet = self.b().or(rb, qb);
        let r = self.b().select(nan, quiet, rb);
        let r = self.float(r);
        // The cube ops take their output modifier whenever IEEE_MODE is
        // clear, whatever the f32 denormal mode (hardware rows, RDNA2).
        let mut m = m.clone();
        let r = if m.omod != 0 && !self.float_mode.ieee { self.apply_omod(r, m.omod) } else { r };
        m.omod = 0;
        self.write_f32_result(ins, &m, r)
    }

    /// Returns false when the opcode is not a float operation.
    pub(crate) fn translate_float(&mut self, ins: &Instruction, m: &Mods) -> Result<bool> {
        use Opcode::*;
        let unary = match ins.opcode {
            VRcpF32 => Some(Unary::Rcp),
            VRcpIflagF32 => Some(Unary::RcpIflag),
            VRsqF32 => Some(Unary::Rsq),
            VSqrtF32 => Some(Unary::Sqrt),
            VLogF32 => Some(Unary::Log),
            VExpF32 => Some(Unary::Exp),
            VFractF32 => Some(Unary::Fract),
            VSinF32 => Some(Unary::Sin),
            VCosF32 => Some(Unary::Cos),
            VTruncF32 => Some(Unary::Trunc),
            VCeilF32 => Some(Unary::Ceil),
            VFloorF32 => Some(Unary::Floor),
            VRndneF32 => Some(Unary::Rndne),
            _ => None,
        };
        if let Some(op) = unary {
            let x = self.src_f32(ins, m, 0);
            let r = self.float_unary(op, x);
            self.write_f32_result(ins, m, r)?;
            return Ok(true);
        }
        let unary16 = match ins.opcode {
            VRcpF16 => Some(Unary::Rcp),
            VRsqF16 => Some(Unary::Rsq),
            VSqrtF16 => Some(Unary::Sqrt),
            VLogF16 => Some(Unary::Log),
            VExpF16 => Some(Unary::Exp),
            VFractF16 => Some(Unary::Fract),
            VSinF16 => Some(Unary::Sin),
            VCosF16 => Some(Unary::Cos),
            VTruncF16 => Some(Unary::Trunc),
            VCeilF16 => Some(Unary::Ceil),
            VFloorF16 => Some(Unary::Floor),
            VRndneF16 => Some(Unary::Rndne),
            _ => None,
        };
        if let Some(op) = unary16 {
            self.float16_unary(ins, m, op)?;
            return Ok(true);
        }
        match ins.opcode {
            VAddF32 | VSubF32 | VSubrevF32 | VMulF32 | VMinF32 | VMaxF32 => {
                let (ai, bi) = if ins.opcode == VSubrevF32 { (1, 0) } else { (0, 1) };
                let a = self.src_f32(ins, m, ai);
                let b = self.src_f32(ins, m, bi);
                let ieee = self.float_mode.ieee;
                let r = match ins.opcode {
                    VAddF32 => self.b().fadd(a, b),
                    VSubF32 | VSubrevF32 => self.b().fsub(a, b),
                    VMulF32 => {
                        let p = self.b().fmul(a, b);
                        self.flush_tiny_product(a, b, p, None)
                    }
                    VMinF32 if ieee => self.ieee_pair(a, b, false),
                    VMaxF32 if ieee => self.ieee_pair(a, b, true),
                    VMinF32 => self.b().fmin(a, b),
                    _ => self.b().fmax(a, b),
                };
                let r = match ins.opcode {
                    VAddF32 | VMulF32 => self.nan_result_f32(&[a, b], r, None),
                    VSubF32 => {
                        let nb = self.negate_bits(b);
                        self.nan_result_f32(&[a, nb], r, None)
                    }
                    VSubrevF32 => {
                        // a = src1, b = src0: src0 (negated) comes first.
                        let nb = self.negate_bits(b);
                        self.nan_result_f32(&[nb, a], r, None)
                    }
                    _ => r,
                };
                self.write_f32_result(ins, m, r)?;
            }
            VMadF32 | VMadMkF32 | VMadAkF32 | VMacF32 | VFmaF32 | VFmaMkF32 | VFmaAkF32 | VFmacF32 => {
                let a = self.src_f32(ins, m, 0);
                let b = self.src_f32(ins, m, 1);
                let c = if matches!(ins.opcode, VMacF32 | VFmacF32) {
                    let d = self.vdst(ins)?;
                    let acc = self.vgpr(d);
                    let acc = if m.abs(2) || m.neg(2) {
                        let mut v = acc;
                        if m.abs(2) {
                            let k = self.b().u32(0x7FFF_FFFF);
                            v = self.b().and(v, k);
                        }
                        if m.neg(2) {
                            let k = self.b().u32(0x8000_0000);
                            v = self.b().xor(v, k);
                        }
                        v
                    } else {
                        acc
                    };
                    let acc = self.float(acc);
                    if self.f32_flush & 1 != 0 {
                        self.flush_denorm(acc)
                    } else {
                        acc
                    }
                } else {
                    self.src_f32(ins, m, 2)
                };
                let fused = matches!(ins.opcode, VFmaF32 | VFmaMkF32 | VFmaAkF32 | VFmacF32);
                let r = if fused {
                    let r = self.b().fma(a, b, c);
                    let r = self.flush_tiny_product(a, b, r, Some(c));
                    let invalid = self.invalid_product_f32(a, b);
                    self.nan_result_f32(&[a, b, c], r, Some(invalid))
                } else {
                    // The separately rounded forms flush f32 denormals
                    // (operands, rounded product and result) in every
                    // FLOAT_MODE.
                    let p = self.b().fmul(a, b);
                    let p = self.flush_denorm(p);
                    self.b().fadd(p, c)
                };
                self.write_f32_result(ins, m, r)?;
            }
            VMin3F32 | VMax3F32 | VMed3F32 => {
                let a = self.src_f32(ins, m, 0);
                let b = self.src_f32(ins, m, 1);
                let c = self.src_f32(ins, m, 2);
                let raw = if ins.opcode == VMed3F32 {
                    let flush = self.f32_flush;
                    self.f32_flush &= !1;
                    let raw = [self.src_f32(ins, m, 0), self.src_f32(ins, m, 1), self.src_f32(ins, m, 2)];
                    self.f32_flush = flush;
                    raw
                } else {
                    [a, b, c]
                };
                let r = if self.float_mode.ieee {
                    match ins.opcode {
                        VMin3F32 | VMax3F32 => {
                            let max = ins.opcode == VMax3F32;
                            let ab = self.ieee_pair(a, b, max);
                            self.ieee_pair(ab, c, max)
                        }
                        _ => {
                            let ab = self.ieee_pair(a, b, false);
                            let minimum = self.ieee_pair(ab, c, false);
                            let median = self.med3_f32_median(raw);
                            let na = self.b().fcmp(FCmp::Uno, a, a);
                            let nb = self.b().fcmp(FCmp::Uno, b, b);
                            let nc = self.b().fcmp(FCmp::Uno, c, c);
                            let n = self.b().lor(na, nb);
                            let n = self.b().lor(n, nc);
                            self.b().select(n, minimum, median)
                        }
                    }
                } else {
                    match ins.opcode {
                        VMin3F32 => self.min3(a, b, c, false),
                        VMax3F32 => self.min3(a, b, c, true),
                        _ => {
                            let min_ab = self.b().fmin(a, b);
                            let min3 = self.b().fmin(min_ab, c);
                            let median = self.med3_f32_median(raw);
                            let na = self.b().fcmp(FCmp::Uno, a, a);
                            let nb = self.b().fcmp(FCmp::Uno, b, b);
                            let nc = self.b().fcmp(FCmp::Uno, c, c);
                            let n = self.b().lor(na, nb);
                            let n = self.b().lor(n, nc);
                            self.b().select(n, min3, median)
                        }
                    }
                };
                self.write_f32_result(ins, m, r)?;
            }
            VLdexpF32 => {
                let bits = self.src_bits(ins, m, 0);
                let bits = {
                    let mut v = bits;
                    if m.abs(0) {
                        let k = self.b().u32(0x7FFF_FFFF);
                        v = self.b().and(v, k);
                    }
                    if m.neg(0) {
                        let k = self.b().u32(0x8000_0000);
                        v = self.b().xor(v, k);
                    }
                    v
                };
                let off = self.src_u32(ins, m, 1);
                if self.options.float_rules & super::float_rules::LDEXP_DENORMALS != 0 {
                    let r = self.ldexp_f32_bits(bits, off);
                    let f = self.float(r);
                    self.write_f32_result(ins, m, f)?;
                    return Ok(true);
                }
                let c = |t: &mut Self, x: u32| t.b().u32(x);
                let sm = c(self, 0x8000_0000);
                let sign = self.b().and(bits, sm);
                let c23 = c(self, 23);
                let c8 = c(self, 8);
                let exponent = self.b().bfe_u(bits, c23, c8);
                let hi = c(self, 512);
                let lo = c(self, (-512i32) as u32);
                let t = self.b().smin(off, hi);
                let clamped = self.b().smax(t, lo);
                let scaled = self.b().iadd(exponent, clamped);
                let keep = c(self, 0x807F_FFFF);
                let kept = self.b().and(bits, keep);
                let sh = self.b().shl(scaled, c23);
                let normal = self.b().or(kept, sh);
                let inf = c(self, 0x7F80_0000);
                let over_v = self.b().or(sign, inf);
                let c255 = c(self, 255);
                let over = self.b().icmp(ICmp::SGe, scaled, c255);
                let mut r = self.b().select(over, over_v, normal);
                let zero = c(self, 0);
                let under = self.b().icmp(ICmp::SLe, scaled, zero);
                r = self.b().select(under, sign, r);
                let special = self.b().icmp(ICmp::Eq, exponent, c255);
                let qb = self.nan_quiet_bit(false);
                let mm = c(self, 0x7FFF_FFFF);
                let mag = self.b().and(bits, mm);
                let nan = self.b().icmp(ICmp::UGt, mag, inf);
                let quieted = self.b().or(bits, qb);
                let passed = self.b().select(nan, quieted, bits);
                r = self.b().select(special, passed, r);
                let ez = self.b().icmp(ICmp::Eq, exponent, zero);
                r = self.b().select(ez, sign, r);
                let f = self.float(r);
                self.write_f32_result(ins, m, f)?;
            }
            VCubeidF32 => self.cube(ins, m, 0)?,
            VCubescF32 => self.cube(ins, m, 1)?,
            VCubetcF32 => self.cube(ins, m, 2)?,
            VCubemaF32 => self.cube(ins, m, 3)?,
            // ---- conversions ----
            VCvtF32I32 | VCvtF32U32 => {
                let v = self.src_u32(ins, m, 0);
                let round_mode = (self.float_mode.mode & 3) as u32;
                let r = if round_mode != 0 && self.options.float_rules & super::float_rules::INT_TO_F32_ROUND != 0 {
                    let bits = self.int_to_f32_directed(v, ins.opcode == VCvtF32I32, round_mode);
                    self.float(bits)
                } else if ins.opcode == VCvtF32I32 {
                    self.b().stof(v, Type::F32)
                } else {
                    self.b().utof(v, Type::F32)
                };
                self.write_f32_result(ins, m, r)?;
            }
            VCvtU32F32 => {
                let v = self.src_f32(ins, m, 0);
                let r = self.f32_to_u32_sat(v, 4294967296.0, 4294967040.0, u32::MAX);
                self.write_u32_result(ins, m, r)?;
            }
            VCvtI32F32 => {
                let v = self.src_f32(ins, m, 0);
                let r = self.f32_to_i32_sat(v, -2147483648.0, 2147483648.0, 2147483520.0, 0x8000_0000, 0x7FFF_FFFF);
                self.write_u32_result(ins, m, r)?;
            }
            VCvtFlrI32F32 | VCvtRpiI32F32 => {
                let v = self.src_f32(ins, m, 0);
                let fl = self.b().floor(v);
                let rounded = if ins.opcode == VCvtRpiI32F32 {
                    let fr = self.b().fsub(v, fl);
                    let half = self.b().f32(0.5);
                    let up = self.b().fcmp(FCmp::OGe, fr, half);
                    let one = self.b().f32(1.0);
                    let zero = self.b().f32(0.0);
                    let inc = self.b().select(up, one, zero);
                    self.b().fadd(fl, inc)
                } else {
                    fl
                };
                let r = self.floored_to_i32(v, rounded);
                self.write_u32_result(ins, m, r)?;
            }
            VCvtF32Ubyte0 | VCvtF32Ubyte1 | VCvtF32Ubyte2 | VCvtF32Ubyte3 => {
                let byte = match ins.opcode {
                    VCvtF32Ubyte0 => 0,
                    VCvtF32Ubyte1 => 1,
                    VCvtF32Ubyte2 => 2,
                    _ => 3,
                };
                let v = self.src_u32(ins, m, 0);
                let o = self.b().u32(byte * 8);
                let w = self.b().u32(8);
                let b = self.b().bfe_u(v, o, w);
                let r = self.b().utof(b, Type::F32);
                self.write_f32_result(ins, m, r)?;
            }
            VCvtOffF32I4 => {
                let v = self.src_u32(ins, m, 0);
                let zero = self.b().u32(0);
                let four = self.b().u32(4);
                let n = self.b().bfe_s(v, zero, four);
                let f = self.b().stof(n, Type::F32);
                let k = self.b().f32(1.0 / 16.0);
                let r = self.b().fmul(f, k);
                self.write_f32_result(ins, m, r)?;
            }
            VCvtF16F32 => {
                let v = self.src_f32(ins, m, 0);
                let bits = self.f16_conversion_bits(v, m.clamp, &[v]);
                let d = self.vdst(ins)?;
                self.write_vgpr_half(d, bits, m.op_sel(3), &DstMods::of(m));
            }
            VCvtF32F16 => {
                let v = self.src_f16(ins, m, 0);
                let v = self.flush_f16_input_value(v);
                self.write_f32_result(ins, m, v)?;
            }
            VCvtU16F16 => {
                let v = self.src_f16(ins, m, 0);
                let r = self.f32_to_u32_sat(v, 65536.0, 65535.0, 0xFFFF);
                let d = self.vdst(ins)?;
                self.write_vgpr_half(d, r, m.op_sel(3), &DstMods::of(m));
            }
            VCvtPkrtzF16F32 => {
                let a = self.src_f32(ins, m, 0);
                let b = self.src_f32(ins, m, 1);
                let lo = self.f32_to_half_bits(a, Rounding::Rtz);
                let hi = self.f32_to_half_bits(b, Rounding::Rtz);
                // f16 result denormals follow FLOAT_MODE bit 7 (RDNA2).
                let lo = self.flush_f16_bits(lo, 0x80);
                let hi = self.flush_f16_bits(hi, 0x80);
                let r = self.pack_u16(lo, hi);
                self.write_u32_result(ins, m, r)?;
            }
            VCvtPknormI16F32 | VCvtPknormU16F32 => {
                let signed = ins.opcode == VCvtPknormI16F32;
                let a = self.src_f32(ins, m, 0);
                let b = self.src_f32(ins, m, 1);
                let ab = self.bits(a);
                let bb = self.bits(b);
                let lo = self.norm_f32(ab, signed);
                let hi = self.norm_f32(bb, signed);
                let r = self.pack_u16(lo, hi);
                self.write_u32_result(ins, m, r)?;
            }
            VCvtPkU8F32 => {
                let v = self.src_f32(ins, m, 0);
                let t = self.f32_to_u32_sat(v, 255.0, 255.0, 255);
                let whole = self.b().utof(t, Type::F32);
                let frac = self.b().fsub(v, whole);
                let one = self.b().u32(1);
                let zero = self.b().u32(0);
                let lowbit = self.b().and(t, one);
                let odd = self.b().icmp(ICmp::Ne, lowbit, zero);
                let half = self.b().f32(0.5);
                let above = self.b().fcmp(FCmp::OGt, frac, half);
                let eq = self.b().fcmp(FCmp::OEq, frac, half);
                let tie = self.b().land(eq, odd);
                let up = self.b().lor(above, tie);
                let c255 = self.b().u32(255);
                let room = self.b().icmp(ICmp::ULt, t, c255);
                let up = self.b().land(up, room);
                let inc = self.b().iadd(t, one);
                let byte = self.b().select(up, inc, t);
                let idx = self.src_u32(ins, m, 1);
                let three = self.b().u32(3);
                let idx = self.b().and(idx, three);
                let shift = self.b().shl(idx, three);
                let ff = self.b().u32(0xFF);
                let mask = self.b().shl(ff, shift);
                let base = self.src_u32(ins, m, 2);
                let nm = self.b().not(mask);
                let base = self.b().and(base, nm);
                let placed = self.b().shl(byte, shift);
                let r = self.b().or(base, placed);
                self.write_u32_result(ins, m, r)?;
            }
            // ---- f16 binary / ternary ----
            VMinF16 | VMaxF16 | VMin3F16 | VMax3F16 | VMed3F16 => self.min_max_f16(ins, m)?,
            VAddF16 | VSubF16 | VSubrevF16 | VMulF16 => {
                let (ai, bi) = if ins.opcode == VSubrevF16 { (1, 0) } else { (0, 1) };
                let a = self.src_f16(ins, m, ai);
                let b = self.src_f16(ins, m, bi);
                // In the f16 round-to-nearest mode the f16 denormal mode
                // applies (hardware rows, RDNA2): flushed inputs (bit 6
                // clear) and results (bit 7 clear) below the smallest
                // normal are zeros of their sign; a product counts as
                // tiny once it no longer rounds up to the smallest normal.
                let mode = self.float_mode.mode;
                let measured = self.f16_denormal_rules() && mode & 0x0C == 0;
                let flush = |t: &mut Self, v: Value, threshold: u32| {
                    let bits = t.bits(v);
                    let mm = t.b().u32(0x7FFF_FFFF);
                    let mag = t.b().and(bits, mm);
                    let lim = t.b().u32(threshold);
                    let tiny = t.b().icmp(ICmp::ULt, mag, lim);
                    let sm = t.b().u32(0x8000_0000);
                    let sign = t.b().and(bits, sm);
                    let r = t.b().select(tiny, sign, bits);
                    t.float(r)
                };
                let (a, b) = if measured && mode & 0x40 == 0 { (flush(self, a, 0x3880_0000), flush(self, b, 0x3880_0000)) } else { (a, b) };
                let mut r = match ins.opcode {
                    VAddF16 => self.b().fadd(a, b),
                    VSubF16 | VSubrevF16 => self.b().fsub(a, b),
                    _ => self.b().fmul(a, b),
                };
                if measured && mode & 0x80 == 0 {
                    r = flush(self, r, if ins.opcode == VMulF16 { 0x387F_F000 } else { 0x3880_0000 });
                }
                self.write_f16(ins, m, r, &[a, b])?;
            }
            VFmaF16 | VFmaMkF16 | VFmaAkF16 => {
                let a = self.src_f16(ins, m, 0);
                let b = self.src_f16(ins, m, 1);
                let c = self.src_f16(ins, m, 2);
                let (a, b, c) = (self.fma_f16_input(a), self.fma_f16_input(b), self.fma_f16_input(c));
                let r = self.fma_f16(a, b, c);
                let r = self.fma_f16_result(r);
                let product = self.invalid_product_f32(a, b);
                let invalid = self.invalid_operation_f32(&[a, b, c], product);
                self.write_f16_checked(ins, m, r, &[a, b, c], Some(invalid))?;
            }
            VPkAddF16 | VPkMulF16 | VPkMinF16 | VPkMaxF16 | VPkFmaF16 | VPkFmacF16 => self.packed_f16(ins, m)?,
            VFmaMixF32 | VFmaMixloF16 | VFmaMixhiF16 => {
                let mut args = Vec::new();
                for i in 0..3 {
                    args.push(self.src_mix(ins, m, i));
                }
                let r = self.b().fma(args[0], args[1], args[2]);
                if ins.opcode == VFmaMixF32 {
                    let r = self.flush_tiny_product(args[0], args[1], r, Some(args[2]));
                    let r = if (0..3).all(|i| !m.op_sel_hi(i)) {
                        let invalid = self.invalid_product_f32(args[0], args[1]);
                        self.nan_result_f32(&args, r, Some(invalid))
                    } else {
                        // With f16 sources an invalid operation gives the
                        // default NaN; other NaNs are left as computed.
                        let product = self.invalid_product_f32(args[0], args[1]);
                        let invalid = self.invalid_operation_f32(&args, product);
                        let bits = self.bits(r);
                        let nan = self.is_nan_bits(bits);
                        let take = self.b().land(nan, invalid);
                        let dn = self.b().u32(0xFFC0_0000);
                        let bits = self.b().select(take, dn, bits);
                        self.float(bits)
                    };
                    let r = if self.f32_flush & 2 != 0 { self.flush_denorm(r) } else { r };
                    let r = if m.clamp { self.saturate_f32(r) } else { r };
                    let bits = self.bits(r);
                    let d = self.vdst(ins)?;
                    self.write_vgpr(d, bits);
                } else {
                    // The f16 forms round the exact product-sum once.
                    let rounding = self.f16_round_mode();
                    let r = if rounding != 0 {
                        self.fma_f32_round_to_odd(args[0], args[1], args[2])
                    } else if self.options.rewrites & super::rewrites::HALF_FMA_FAST != 0 {
                        // f32 sources keep denormals unless the mode flushes
                        // f32 inputs (src_mix); f16 ones widen to normals.
                        let kept: Vec<Value> = if self.f32_flush & 1 == 0 { (0..3).filter(|&i| !m.op_sel_hi(i)).map(|i| args[i]).collect() } else { Vec::new() };
                        self.fma_for_half(args[0], args[1], args[2], &kept)
                    } else {
                        self.fma_f32_round_to_odd(args[0], args[1], args[2])
                    };
                    let h = if rounding != 0 {
                        // The round-to-odd f32 sum keeps the directed
                        // rounding of the exact sum; toward -infinity an
                        // exact zero sum of terms that are not both +0 is
                        // -0.
                        let r = if m.clamp { self.saturate_f32(r) } else { r };
                        let h = self.f32_to_f16_bits_rounded(r, &args);
                        if rounding == 2 {
                            self.mix_negative_zero(&args, r, h)
                        } else {
                            h
                        }
                    } else {
                        let r = self.clamp_f16_overflow(r, &args);
                        let r = if m.clamp { self.saturate_f32(r) } else { r };
                        self.f32_to_half_bits(r, Rounding::Rte)
                    };
                    let d = self.vdst(ins)?;
                    self.write_vgpr_half(d, h, ins.opcode == VFmaMixhiF16, &DstMods::plain());
                }
            }
            VDot2cF32F16 => self.dot2_f32_f16(ins, m, true)?,
            VInterpP1llF16 | VInterpP1lvF16 | VInterpP2F16 => self.interp_f16(ins, m)?,
            // ---- f64 ----
            VFmaF64 | VMulF64 => {
                let a = self.src_f64(ins, m, 0);
                let b = self.src_f64(ins, m, 1);
                let r = if ins.opcode == VFmaF64 {
                    let c = self.src_f64(ins, m, 2);
                    self.b().fma(a, b, c)
                } else {
                    self.b().fmul(a, b)
                };
                self.write_f64_result(ins, m, r)?;
            }
            VRcpF64 | VRsqF64 | VSqrtF64 => {
                let a = self.src_f64(ins, m, 0);
                let r = match ins.opcode {
                    VRcpF64 => self.b().rcp(a),
                    VRsqF64 => self.b().inv_sqrt(a),
                    _ => self.b().sqrt(a),
                };
                let r = self.f64_unary_nan(a, r);
                self.write_f64_result(ins, m, r)?;
            }
            VCvtF64I32 | VCvtF64U32 => {
                let v = self.src_u32(ins, m, 0);
                let r = if ins.opcode == VCvtF64I32 { self.b().stof(v, Type::F64) } else { self.b().utof(v, Type::F64) };
                self.write_f64_result(ins, m, r)?;
            }
            VCvtF32F64 => {
                let v = self.src_f64(ins, m, 0);
                let r = self.b().fconvert(v, Type::F32, Rounding::Rte);
                // The output modifier does not apply to f64-source results.
                let m = Mods { omod: 0, ..m.clone() };
                self.write_f32_result(ins, &m, r)?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// A v_fma_f16 / v_fmac_f16 / v_fmamk / v_fmaak_f16 source in the f16
    /// round-to-nearest mode: below 2^-14 a zero of its sign when f16
    /// input denormals are flushed (RDNA2).
    pub(super) fn fma_f16_input(&mut self, v: Value) -> Value {
        if self.f16_round_field() != 0 {
            return v;
        }
        self.flush_f16_input_value(v)
    }

    /// The f16 fma result (rounded to odd in f32) in the f16
    /// round-to-nearest mode: a zero of its sign when it rounds below the
    /// smallest normal and f16 result denormals are flushed (RDNA2).
    pub(super) fn fma_f16_result(&mut self, r: Value) -> Value {
        if self.f16_round_field() != 0 || !self.f16_flushes(0x80) {
            return r;
        }
        self.flush_tiny_f16_value(r, 0x387F_F000)
    }

    /// Toward -infinity an exact zero sum `sum` of `args[0] * args[1] +
    /// args[2]` is -0 unless the product and the addend are both +0; `h`
    /// is the f16 result otherwise.
    fn mix_negative_zero(&mut self, args: &[Value], sum: Value, h: Value) -> Value {
        let c = |t: &mut Self, x: u32| t.b().u32(x);
        let mm = c(self, 0x7FFF_FFFF);
        let zero = c(self, 0);
        let sb = self.bits(sum);
        let sum_mag = self.b().and(sb, mm);
        let sum_zero = self.b().icmp(ICmp::Eq, sum_mag, zero);
        let ab = self.bits(args[0]);
        let bb = self.bits(args[1]);
        let cb = self.bits(args[2]);
        let am = self.b().and(ab, mm);
        let bm = self.b().and(bb, mm);
        let a_zero = self.b().icmp(ICmp::Eq, am, zero);
        let b_zero = self.b().icmp(ICmp::Eq, bm, zero);
        let product_zero = self.b().lor(a_zero, b_zero);
        let x = self.b().xor(ab, bb);
        let sm = c(self, 0x8000_0000);
        let product_sign = self.b().and(x, sm);
        let product_positive = self.b().icmp(ICmp::Eq, product_sign, zero);
        let positive_product = self.b().land(product_zero, product_positive);
        let addend_positive_zero = self.b().icmp(ICmp::Eq, cb, zero);
        let positive_zeros = self.b().land(positive_product, addend_positive_zero);
        let not_positive = self.b().lnot(positive_zeros);
        let negative_zero = self.b().land(sum_zero, not_positive);
        let nz = c(self, 0x8000);
        self.b().select(negative_zero, nz, h)
    }

    /// f16 FMA rounded once: the product of two f16 values is exact in
    /// f32, the sum is rounded to odd in f32 (with its exact error term),
    /// and the f16 conversion of that rounds the exact result once.
    pub(super) fn fma_f16(&mut self, a: Value, b: Value, c: Value) -> Value {
        let product = self.b().fmul(a, b);
        let sum = self.b().fadd(product, c);
        let addend_part = self.b().fsub(sum, product);
        let product_part = self.b().fsub(sum, addend_part);
        let product_error = self.b().fsub(product, product_part);
        let addend_error = self.b().fsub(c, addend_part);
        let error = self.b().fadd(product_error, addend_error);
        let zero = self.b().f32(0.0);
        let inexact = self.b().fcmp(FCmp::ONe, error, zero);
        let sb = self.bits(sum);
        let eb = self.bits(error);
        let x = self.b().xor(sb, eb);
        let sm = self.b().u32(0x8000_0000);
        let xs = self.b().and(x, sm);
        let z = self.b().u32(0);
        let same_sign = self.b().icmp(ICmp::Eq, xs, z);
        let one = self.b().u32(1);
        let less = self.b().isub(sb, one);
        let toward_zero = self.b().select(same_sign, sb, less);
        let odd = self.b().or(toward_zero, one);
        let r = self.b().select(inexact, odd, sb);
        self.float(r)
    }

    /// v_interp_p1ll_f16 / v_interp_p1lv_f16 / v_interp_p2_f16 (VOP3, 16-bit
    /// pixel attributes): SRC0 holds the attribute (bits 5:0), the channel
    /// (7:6) and the half (8); SRC1 is the i / j VGPR; SRC2 the P0 half
    /// (p1lv) or the f32 partial result (p2). `delta * coordinate + base`
    /// is rounded once (to f32 for the p1 forms, to f16 for p2).
    fn interp_f16(&mut self, ins: &Instruction, m: &Mods) -> Result<()> {
        use Opcode::*;
        if ins.words.len() < 2 {
            return self.invalid(ins, "VOP3 interpolation without its second word");
        }
        let (w0, w1) = (ins.words[0], ins.words[1]);
        let (src0, src1, src2) = (w1 & 0x1FF, (w1 >> 9) & 0x1FF, (w1 >> 18) & 0x1FF);
        let (abs, neg) = ((w0 >> 8) & 7, w1 >> 29);
        let op_sel = (w0 >> 11) & 0xF;
        let has_src2 = ins.opcode != VInterpP1llF16;
        // op_sel[3] of v_interp_p2_f16 (the only 16-bit result) writes the
        // destination's high half and keeps the low one.
        let op_sel = if ins.opcode == VInterpP2F16 { op_sel & !8 } else { op_sel };
        if abs & 1 != 0 || neg & 1 != 0 || op_sel != 0 || (!has_src2 && (abs | neg) & 4 != 0) {
            return self.unsupported(ins, "VOP3 interpolation modifiers on the attribute operand");
        }
        if src1 < 256 || (has_src2 && src2 < 256) {
            return self.unsupported(ins, "VOP3 interpolation source that is not a VGPR");
        }
        let mode = self.float_mode.mode;
        if mode & 0xF != 0 {
            return self.unsupported(ins, "16-bit interpolation with a rounding mode other than nearest even");
        }
        let flush32 = mode >> 4 & 3 == 0;
        let flush16 = mode >> 6 & 3 == 0;
        let attr = (src0 & 0x3F) as u8;
        let chan = ((src0 >> 6) & 3) as u8;
        let high = src0 & 0x100 != 0;
        let modify = |t: &mut Self, mut bits: Value, abs: bool, neg: bool, sign: u32| {
            if abs {
                let k = t.b().u32(sign - 1);
                bits = t.b().and(bits, k);
            }
            if neg {
                let k = t.b().u32(sign);
                bits = t.b().xor(bits, k);
            }
            bits
        };
        let coordinate = self.vgpr(src1 - 256);
        let coordinate = modify(self, coordinate, abs & 2 != 0, neg & 2 != 0, 0x8000_0000);
        let coordinate = self.float(coordinate);
        let param = |t: &mut Self, p: InterpParam| t.b().emit(Op::InterpParam { attr, chan, param: p }, Type::F32, vec![]);
        let (delta, base) = match ins.opcode {
            VInterpP1llF16 => (param(self, InterpParam::HalfP10 { high }), param(self, InterpParam::HalfP0 { high })),
            VInterpP1lvF16 => {
                let delta = param(self, InterpParam::HalfP10 { high });
                let word = self.vgpr(src2 - 256);
                let o = self.b().u32(if high { 16 } else { 0 });
                let w = self.b().u32(16);
                let h = self.b().bfe_u(word, o, w);
                let h = modify(self, h, abs & 4 != 0, neg & 4 != 0, 0x8000);
                (delta, self.half_to_f32(h))
            }
            _ => {
                let delta = param(self, InterpParam::HalfP20 { high });
                let partial = self.vgpr(src2 - 256);
                let partial = modify(self, partial, abs & 4 != 0, neg & 4 != 0, 0x8000_0000);
                (delta, self.float(partial))
            }
        };
        let half = ins.opcode == VInterpP2F16;
        let r = self.interpolate(delta, coordinate, base, half, flush32, flush16);
        if half {
            self.write_f16(ins, m, r, &[delta, coordinate, base])
        } else {
            self.write_f32_result(ins, m, r)
        }
    }

    /// Flush a denormal (below `smallest_normal`, as f32 bits) to a zero of
    /// its sign.
    fn flush_below(&mut self, v: Value, smallest_normal: u32) -> Value {
        let bits = self.bits(v);
        let mm = self.b().u32(0x7FFF_FFFF);
        let mag = self.b().and(bits, mm);
        let lim = self.b().u32(smallest_normal);
        let small = self.b().icmp(ICmp::ULt, mag, lim);
        let flush = self.nonzero_small(mag, small);
        let sm = self.b().u32(0x8000_0000);
        let sign = self.b().and(bits, sm);
        let r = self.b().select(flush, sign, bits);
        self.float(r)
    }

    /// `delta * coordinate + base` of the 16-bit interpolation, rounded once
    /// (to f32, or to odd in f32 for an f16 result). f16 operands flush with
    /// the f16 denormal mode, f32 ones with the f32 mode. A non-finite
    /// operand gives the first NaN (delta, coordinate, then base unless the
    /// product is inf * 0), quieted for an f16 result or with IEEE_MODE, or
    /// the infinity / default NaN of the unbounded sum.
    fn interpolate(&mut self, delta: Value, coordinate: Value, base: Value, half: bool, flush32: bool, flush16: bool) -> Value {
        let delta = if flush16 { self.flush_below(delta, 0x3880_0000) } else { delta };
        let coordinate = if flush32 { self.flush_below(coordinate, 0x0080_0000) } else { coordinate };
        let base = match (half, flush32, flush16) {
            (true, true, _) => self.flush_below(base, 0x0080_0000),
            (false, _, true) => self.flush_below(base, 0x3880_0000),
            _ => base,
        };
        let finite_result = if half {
            self.fma_f32_round_to_odd(delta, coordinate, base)
        } else {
            let d64 = self.b().fconvert(delta, Type::F64, Rounding::Rte);
            let c64 = self.b().fconvert(coordinate, Type::F64, Rounding::Rte);
            let b64 = self.b().fconvert(base, Type::F64, Rounding::Rte);
            let p = self.b().fmul(d64, c64);
            let s = self.sum_round_to_odd(p, b64);
            self.b().fconvert(s, Type::F32, Rounding::Rte)
        };
        let d64 = self.b().fconvert(delta, Type::F64, Rounding::Rte);
        let c64 = self.b().fconvert(coordinate, Type::F64, Rounding::Rte);
        let b64 = self.b().fconvert(base, Type::F64, Rounding::Rte);
        let p = self.b().fmul(d64, c64);
        let unbounded = self.b().fadd(p, b64);
        let unbounded_nan = self.b().fcmp(FCmp::Uno, unbounded, unbounded);
        let ub = self.b().bitcast(unbounded, Type::U64);
        let (_, uh) = self.b().split_u64(ub);
        let sm = self.b().u32(0x8000_0000);
        let us = self.b().and(uh, sm);
        let inf = self.b().u32(0x7F80_0000);
        let signed_inf = self.b().or(us, inf);
        let dn = self.b().u32(0xFFC0_0000);
        let mut special = self.b().select(unbounded_nan, dn, signed_inf);
        let quiet = if half || self.float_mode.ieee { 0x0040_0000 } else { 0 };
        let inputs = [self.bits(delta), self.bits(coordinate), self.bits(base)];
        let mm = self.b().u32(0x7FFF_FFFF);
        let class = |t: &mut Self, bits: Value| {
            let mag = t.b().and(bits, mm);
            let nan = t.b().icmp(ICmp::UGt, mag, inf);
            let infinite = t.b().icmp(ICmp::Eq, mag, inf);
            let z = t.b().u32(0);
            let zero = t.b().icmp(ICmp::Eq, mag, z);
            let e = t.b().and(bits, inf);
            let non_finite = t.b().icmp(ICmp::Eq, e, inf);
            (nan, infinite, zero, non_finite)
        };
        let c: Vec<_> = inputs.iter().map(|&b| class(self, b)).collect();
        let first = self.b().land(c[0].1, c[1].2);
        let second = self.b().land(c[0].2, c[1].1);
        let invalid_product = self.b().lor(first, second);
        let qb = self.b().u32(quiet);
        let valid = self.b().lnot(invalid_product);
        let base_nan = self.b().land(c[2].0, valid);
        let q2 = self.b().or(inputs[2], qb);
        special = self.b().select(base_nan, q2, special);
        let q1 = self.b().or(inputs[1], qb);
        special = self.b().select(c[1].0, q1, special);
        let q0 = self.b().or(inputs[0], qb);
        special = self.b().select(c[0].0, q0, special);
        let any = self.b().lor(c[0].3, c[1].3);
        let any = self.b().lor(any, c[2].3);
        let fb = self.bits(finite_result);
        let r = self.b().select(any, special, fb);
        let r = self.float(r);
        match (half, flush16, flush32) {
            (true, true, _) => self.flush_below(r, 0x3880_0000),
            (false, _, true) => self.flush_below(r, 0x0080_0000),
            _ => r,
        }
    }

    /// `a * b + c` for one rounding to f16 (after the f16 overflow clamp and
    /// the [0, 1] clamp): an f32 value whose f16 rounding is that of
    /// [`Self::fma_f32_round_to_odd`].
    ///
    /// The f32 fma `r` (rounded to nearest) and the round-to-odd value lie
    /// on the same side of every f32 value `m` unless `r == m`: both
    /// roundings are monotonic. The f16 rounding, the overflow clamp (at
    /// the midpoint 65520) and the [0, 1] clamp change only at f16
    /// midpoints or f16 values, and an f16 value `m` lies strictly inside
    /// its rounding cell, so only an `r` that is an f16 midpoint (2^-25
    /// and up) can round differently. A NaN `r` takes the slow path for
    /// its payload; an infinite `r` is an overflow (whose round-to-odd
    /// value, the largest finite one, rounds and clamps the same) or an
    /// infinite source. A zero or denormal `r` is a zero of its sign (the
    /// f16 of anything below 2^-25; a host that flushes the fma's denormal
    /// result keeps the sign). A kept f32 denormal source (`kept`) also
    /// takes the slow path: a host may flush it in the fma.
    ///
    /// The round-to-odd sum is computed in f64 (a fraction of the f32 rate
    /// on most GPUs) only for the invocations that need it: it is the lazy
    /// arm of the select.
    fn fma_for_half(&mut self, a: Value, b: Value, c: Value, kept: &[Value]) -> Value {
        let r = self.b().fma(a, b, c);
        let bits = self.bits(r);
        let c23 = self.b().u32(23);
        let shifted = self.b().lshr(bits, c23);
        let ff = self.b().u32(0xFF);
        let e = self.b().and(shifted, ff);
        let zero = self.b().u32(0);
        let tiny = self.b().icmp(ICmp::Eq, e, zero);
        let sm = self.b().u32(0x8000_0000);
        let sign = self.b().and(bits, sm);
        let fast = self.b().select(tiny, sign, bits);
        // NaN.
        let mm = self.b().u32(0x7FFF_FFFF);
        let mag = self.b().and(bits, mm);
        let inf = self.b().u32(0x7F80_0000);
        let mut slow = self.b().icmp(ICmp::UGt, mag, inf);
        // A midpoint of normal f16 values (2^-14 <= |r| < 2^16): the 13
        // bits below the f16 mantissa are 1000000000000.
        let lo = self.b().u32(113);
        let hi = self.b().u32(142);
        let ge = self.b().icmp(ICmp::UGe, e, lo);
        let le = self.b().icmp(ICmp::ULe, e, hi);
        let normal = self.b().land(ge, le);
        let low = self.b().u32(0x1FFF);
        let low_bits = self.b().and(bits, low);
        let half = self.b().u32(0x1000);
        let at_half = self.b().icmp(ICmp::Eq, low_bits, half);
        let mid = self.b().land(normal, at_half);
        slow = self.b().lor(slow, mid);
        // A midpoint of f16 denormals (2^-25 <= |r| < 2^-14, exponent field
        // 102..=112): the 126 - e bits of the significand below 2^-24 are
        // 100...0.
        let lo = self.b().u32(102);
        let hi = self.b().u32(112);
        let ge = self.b().icmp(ICmp::UGe, e, lo);
        let le = self.b().icmp(ICmp::ULe, e, hi);
        let subnormal = self.b().land(ge, le);
        let c126 = self.b().u32(126);
        let n = self.b().isub(c126, e);
        let c24 = self.b().u32(24);
        let n = self.b().umin(n, c24);
        let one = self.b().u32(1);
        let unit = self.b().shl(one, n);
        let mask = self.b().isub(unit, one);
        let m23 = self.b().u32(0x7F_FFFF);
        let mantissa = self.b().and(bits, m23);
        let implicit = self.b().u32(0x80_0000);
        let significand = self.b().or(mantissa, implicit);
        let dropped = self.b().and(significand, mask);
        let half = self.b().lshr(unit, one);
        let at_half = self.b().icmp(ICmp::Eq, dropped, half);
        let mid = self.b().land(subnormal, at_half);
        slow = self.b().lor(slow, mid);
        for &k in kept {
            // 0 < |k| < 2^-126.
            let kb = self.bits(k);
            let km = self.b().and(kb, mm);
            let less = self.b().isub(km, one);
            let denormal = self.b().icmp(ICmp::ULt, less, m23);
            slow = self.b().lor(slow, denormal);
        }
        let exact = self.fma_f32_round_to_odd(a, b, c);
        let exact = self.bits(exact);
        let out = self.b().select(slow, exact, fast);
        self.func.lazy_selects.insert(out);
        self.float(out)
    }

    /// `a * b + c` of f32 values rounded to odd in f32 (an exact f64
    /// product, a sum rounded to odd in f64, then narrowed to odd), so that
    /// a later f16 conversion rounds the exact result once.
    fn fma_f32_round_to_odd(&mut self, a: Value, b: Value, c: Value) -> Value {
        let a64 = self.b().fconvert(a, Type::F64, Rounding::Rte);
        let b64 = self.b().fconvert(b, Type::F64, Rounding::Rte);
        let c64 = self.b().fconvert(c, Type::F64, Rounding::Rte);
        let p = self.b().fmul(a64, b64);
        let odd_sum = self.sum_round_to_odd(p, c64);
        // Infinite and NaN sums take the plain sum (its error term is NaN).
        let plain = self.b().fadd(p, c64);
        let magnitude = self.b().fabs(plain);
        let inf = self.b().f64(f64::INFINITY);
        let finite = self.b().fcmp(FCmp::OLt, magnitude, inf);
        let s = self.b().select(finite, odd_sum, plain);
        let r = self.b().fconvert(s, Type::F32, Rounding::Rte);
        let back = self.b().fconvert(r, Type::F64, Rounding::Rte);
        // Narrowing to odd: an inexact result moves toward zero and gets
        // its low bit set (NaN and infinite sums compare unordered / equal).
        let inexact = self.b().fcmp(FCmp::ONe, back, s);
        let ab = self.b().fabs(back);
        let asum = self.b().fabs(s);
        let away = self.b().fcmp(FCmp::OGt, ab, asum);
        let rb = self.bits(r);
        let one = self.b().u32(1);
        let less = self.b().isub(rb, one);
        let toward_zero = self.b().select(away, less, rb);
        let odd = self.b().or(toward_zero, one);
        let out = self.b().select(inexact, odd, rb);
        self.float(out)
    }

    /// v_fma_mix operand: f16 half (op_sel picks it) when op_sel_hi is set,
    /// else the f32 source; NEG_HI is abs for these opcodes.
    fn src_mix(&mut self, ins: &Instruction, m: &Mods, i: usize) -> Value {
        let abs = m.neg_hi(i);
        let neg = m.neg(i);
        let mut v = if m.op_sel_hi(i) {
            // f16 sources follow the f16 input denormal mode (RDNA2).
            let bits = self.src_half_bits(ins, m, i, m.op_sel(i));
            let bits = self.flush_f16_bits(bits, 0x40);
            self.half_to_f32(bits)
        } else {
            let bits = self.src_bits(ins, m, i);
            let f = self.float(bits);
            if self.f32_flush & 1 != 0 {
                self.flush_denorm(f)
            } else {
                f
            }
        };
        if abs {
            v = self.b().fabs(v);
        }
        if neg {
            v = self.b().fneg(v);
        }
        v
    }

    /// f16 bits of the packed half `high_lane` of source `i`, with the
    /// modifiers `src_packed_f16` applies (on the bits).
    fn packed_lane_bits(&mut self, ins: &Instruction, m: &Mods, i: usize, high_lane: bool) -> Value {
        let select_high = if high_lane { m.op_sel_hi(i) } else { m.op_sel(i) };
        let mut bits = self.src_half_bits(ins, m, i, select_high);
        let (abs, negate) = if m.vop3p { (false, if high_lane { m.neg_hi(i) } else { m.neg(i) }) } else { (m.abs(i), m.neg(i)) };
        if abs {
            let k = self.b().u32(0x7FFF);
            bits = self.b().and(bits, k);
        }
        if negate {
            let k = self.b().u32(0x8000);
            bits = self.b().xor(bits, k);
        }
        bits
    }

    /// Packed f16 add/mul/min/max/fma/fmac, each half computed in f32 and
    /// rounded once. NaN results are picked on the bits: min/max return the
    /// other operand for a NaN and a signaling NaN quieted only with
    /// IEEE_MODE; the others the first NaN source (quieted with IEEE_MODE,
    /// an fma addend skipped for inf * 0) or 0xfe00 for an invalid
    /// operation. With clamp and DX10_CLAMP a NaN result of the arithmetic
    /// forms is +0.
    fn packed_f16(&mut self, ins: &Instruction, m: &Mods) -> Result<()> {
        use Opcode::*;
        let min_max = matches!(ins.opcode, VPkMinF16 | VPkMaxF16);
        let count = match ins.opcode {
            VPkFmaF16 | VPkFmacF16 => 3,
            _ => 2,
        };
        let dx10 = self.float_mode.dx10_clamp;
        let ieee = self.float_mode.ieee;
        let mut halves = Vec::new();
        // The f16 denormal mode flushes the lane inputs (FLOAT_MODE bit 6
        // clear) and the add / mul / fma results that round below the
        // smallest normal (bit 7 clear), measured on RDNA2.
        let flush_results = !min_max && self.f16_flushes(0x80);
        let tiny = if ins.opcode == VPkAddF16 { 0x3880_0000 } else { 0x387F_F000 };
        for high in [false, true] {
            let a = self.src_packed_f16(ins, m, 0, high);
            let a = self.flush_f16_input_value(a);
            let b = self.src_packed_f16(ins, m, 1, high);
            let b = self.flush_f16_input_value(b);
            let b0 = self.packed_lane_bits(ins, m, 0, high);
            let b1 = self.packed_lane_bits(ins, m, 1, high);
            let mut bits = vec![self.flush_f16_bits(b0, 0x40), self.flush_f16_bits(b1, 0x40)];
            let r = match ins.opcode {
                VPkAddF16 => self.b().fadd(a, b),
                VPkMulF16 => self.b().fmul(a, b),
                VPkMinF16 => self.b().fmin(a, b),
                VPkMaxF16 => self.b().fmax(a, b),
                VPkFmaF16 => {
                    let c = self.src_packed_f16(ins, m, 2, high);
                    let c = self.flush_f16_input_value(c);
                    let cb = self.packed_lane_bits(ins, m, 2, high);
                    bits.push(self.flush_f16_bits(cb, 0x40));
                    let r = self.fma_f16(a, b, c);
                    let r = if flush_results { self.flush_tiny_f16_value(r, tiny) } else { r };
                    self.clamp_f16_overflow(r, &[a, b, c])
                }
                _ => {
                    let d = self.vdst(ins)?;
                    let acc = self.vgpr(d);
                    let o = self.b().u32(if high { 16 } else { 0 });
                    let w = self.b().u32(16);
                    let cb = self.b().bfe_u(acc, o, w);
                    let cb = self.flush_f16_bits(cb, 0x40);
                    bits.push(cb);
                    let c = self.half_to_f32(cb);
                    let r = self.fma_f16(a, b, c);
                    let r = if flush_results { self.flush_tiny_f16_value(r, tiny) } else { r };
                    self.clamp_f16_overflow(r, &[a, b, c])
                }
            };
            let r = if flush_results && count == 2 { self.flush_tiny_f16_value(r, tiny) } else { r };
            let r = if count == 2 { self.clamp_f16_overflow(r, &[a, b]) } else { r };
            let r = if m.clamp { self.saturate_f32(r) } else { r };
            let mut h = self.f32_to_half_bits(r, Rounding::Rte);
            if !m.clamp || !dx10 || min_max {
                let is_nan = |t: &mut Self, v: Value| {
                    let mm = t.b().u32(0x7FFF);
                    let mag = t.b().and(v, mm);
                    let inf = t.b().u32(0x7C00);
                    t.b().icmp(ICmp::UGt, mag, inf)
                };
                let qb = self.nan_quiet_bit(true);
                let mut lane = h;
                if min_max {
                    let n0 = is_nan(self, bits[0]);
                    lane = self.b().select(n0, bits[1], lane);
                    let n1 = is_nan(self, bits[1]);
                    lane = self.b().select(n1, bits[0], lane);
                    if ieee {
                        for k in (0..count).rev() {
                            let nan = is_nan(self, bits[k]);
                            let q = self.b().u32(0x200);
                            let qbit = self.b().and(bits[k], q);
                            let zero = self.b().u32(0);
                            let signaling = self.b().icmp(ICmp::Eq, qbit, zero);
                            let snan = self.b().land(nan, signaling);
                            let quiet = self.b().or(bits[k], qb);
                            lane = self.b().select(snan, quiet, lane);
                        }
                    }
                } else {
                    let ln = is_nan(self, lane);
                    let dn = self.b().u32(0xFE00);
                    lane = self.b().select(ln, dn, lane);
                    for k in (0..count).rev() {
                        let mut selected = is_nan(self, bits[k]);
                        if k == 2 {
                            let mm = self.b().u32(0x7FFF);
                            let m0 = self.b().and(bits[0], mm);
                            let m1 = self.b().and(bits[1], mm);
                            let inf = self.b().u32(0x7C00);
                            let zero = self.b().u32(0);
                            let i0 = self.b().icmp(ICmp::Eq, m0, inf);
                            let z1 = self.b().icmp(ICmp::Eq, m1, zero);
                            let i1 = self.b().icmp(ICmp::Eq, m1, inf);
                            let z0 = self.b().icmp(ICmp::Eq, m0, zero);
                            let first = self.b().land(i0, z1);
                            let second = self.b().land(i1, z0);
                            let invalid = self.b().lor(first, second);
                            let valid = self.b().lnot(invalid);
                            selected = self.b().land(selected, valid);
                        }
                        let quiet = self.b().or(bits[k], qb);
                        lane = self.b().select(selected, quiet, lane);
                    }
                }
                h = self.clamp_f16_bits(lane, m.clamp);
            }
            halves.push(h);
        }
        let r = self.pack_u16(halves[0], halves[1]);
        let d = self.vdst(ins)?;
        self.write_vgpr_mods(d, r, &DstMods::of(m));
        Ok(())
    }
}
