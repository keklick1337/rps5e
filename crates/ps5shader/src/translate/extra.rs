// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Less common instructions: legacy multiplies, division helpers, frexp /
//! ldexp, 16-bit and packed integer forms, dot products, SAD variants, f64
//! operations and conversions, M0-relative scalar moves.

use super::operand::{DstMods, Mods};
use super::{Result, Translator};
use crate::ir_types::*;
use crate::opcode::Opcode;
use crate::ssa::*;

/// Bits and class of an f64 value.
struct F64Class {
    lo: Value,
    hi: Value,
    nan: Value,
    snan: Value,
    inf: Value,
    zero: Value,
    /// Biased exponent field.
    exp: Value,
}

impl<'a> Translator<'a> {
    fn u(&mut self, x: u32) -> Value {
        self.b().u32(x)
    }

    fn f_bits(&mut self, v: Value) -> Value {
        self.b().bitcast(v, Type::U32)
    }

    fn to_f(&mut self, v: Value) -> Value {
        self.b().bitcast(v, Type::F32)
    }

    /// `|x|` bits of an f32 value given as bits.
    fn mag(&mut self, bits: Value) -> Value {
        let m = self.u(0x7FFF_FFFF);
        self.b().and(bits, m)
    }

    fn is_inf_bits(&mut self, bits: Value) -> Value {
        let m = self.mag(bits);
        let inf = self.u(0x7F80_0000);
        self.b().icmp(ICmp::Eq, m, inf)
    }

    fn is_zero_bits(&mut self, bits: Value) -> Value {
        let m = self.mag(bits);
        let z = self.u(0);
        self.b().icmp(ICmp::Eq, m, z)
    }

    fn exp_bits(&mut self, bits: Value) -> Value {
        let o = self.u(23);
        let w = self.u(8);
        self.b().bfe_u(bits, o, w)
    }

    /// Source with f32 abs/neg applied, as bits.
    fn src_mod_bits(&mut self, ins: &Instruction, m: &Mods, i: usize) -> Value {
        let f = self.src_f32(ins, m, i);
        self.f_bits(f)
    }

    /// Integer dot product over `element` bit lanes (accumulator `acc`).
    fn integer_dot(&mut self, ins: &Instruction, m: &Mods, element: u32, signed: bool, acc: Value) -> Value {
        let mut a = self.src_u32(ins, m, 0);
        let mut b = self.src_u32(ins, m, 1);
        if element == 16 && ins.encoding == Encoding::Vop3p {
            // The low product takes the half op_sel picks, the high one
            // the half op_sel_hi picks.
            for (i, v) in [&mut a, &mut b].into_iter().enumerate() {
                let (lo_high, hi_high) = (m.op_sel >> i & 1 != 0, m.op_sel_hi >> i & 1 != 0);
                if (lo_high, hi_high) == (false, true) {
                    continue;
                }
                let sixteen = self.u(16);
                let zero = self.u(0);
                let lo = self.b().bfe_u(*v, if lo_high { sixteen } else { zero }, sixteen);
                let hi = self.b().bfe_u(*v, if hi_high { sixteen } else { zero }, sixteen);
                *v = self.b().bfi(lo, hi, sixteen, sixteen);
            }
        }
        let mut r = acc;
        let mut off = 0;
        while off < 32 {
            let o = self.u(off);
            let w = self.u(element);
            let (x, y) = if signed { (self.b().bfe_s(a, o, w), self.b().bfe_s(b, o, w)) } else { (self.b().bfe_u(a, o, w), self.b().bfe_u(b, o, w)) };
            let p = self.b().imul(x, y);
            r = self.b().iadd(r, p);
            off += element;
        }
        r
    }

    /// f16 bits -> normalized 16-bit integer (round to nearest even).
    fn norm_f16(&mut self, bits: Value, signed: bool) -> Value {
        let c = |t: &mut Self, x: u32| t.b().u32(x);
        let mm = c(self, 0x7FFF);
        let mag = self.b().and(bits, mm);
        let sb = c(self, 0x8000);
        let s = self.b().and(bits, sb);
        let zero = c(self, 0);
        let negative = self.b().icmp(ICmp::Ne, s, zero);
        let inf = c(self, 0x7C00);
        let nan = self.b().icmp(ICmp::UGt, mag, inf);
        let one = c(self, 0x3C00);
        let saturated = self.b().icmp(ICmp::UGe, mag, one);
        let scale = if signed { 32767u32 } else { 65535 };
        let ten = c(self, 10);
        let exponent = self.b().lshr(mag, ten);
        let normal = self.b().icmp(ICmp::Ne, exponent, zero);
        let m3 = c(self, 0x3FF);
        let frac = self.b().and(mag, m3);
        let hb = c(self, 0x400);
        let hidden = self.b().select(normal, hb, zero);
        let mantissa = self.b().or(frac, hidden);
        let onec = c(self, 1);
        let e = self.b().select(normal, exponent, onec);
        // Saturated inputs (exponent >= 15, including inf / NaN) take the
        // constant result; clamp so the unused shift stays in range.
        let c15 = c(self, 15);
        let e = self.b().umin(e, c15);
        let c25 = c(self, 25);
        let shift = self.b().isub(c25, e);
        let sc = c(self, scale);
        let product = self.b().imul(mantissa, sc);
        let truncated = self.b().lshr(product, shift);
        let p1 = self.b().shl(onec, shift);
        let rmask = self.b().isub(p1, onec);
        let rem = self.b().and(product, rmask);
        let sm1 = self.b().isub(shift, onec);
        let half = self.b().shl(onec, sm1);
        let above = self.b().icmp(ICmp::UGt, rem, half);
        let tie = self.b().icmp(ICmp::Eq, rem, half);
        let tb = self.b().and(truncated, onec);
        let odd = self.b().icmp(ICmp::Ne, tb, zero);
        let tie_odd = self.b().land(tie, odd);
        let up = self.b().lor(above, tie_odd);
        let inc = self.b().bool_to_u32(up);
        let rounded = self.b().iadd(truncated, inc);
        let mut v = self.b().select(saturated, sc, rounded);
        if signed {
            let n = self.b().ineg(v);
            v = self.b().select(negative, n, v);
        } else {
            v = self.b().select(negative, zero, v);
        }
        let v = self.b().select(nan, zero, v);
        let ffff = c(self, 0xFFFF);
        self.b().and(v, ffff)
    }

    fn write_half_result(&mut self, ins: &Instruction, m: &Mods, bits: Value) -> Result<()> {
        let d = self.vdst(ins)?;
        self.write_vgpr_half(d, bits, m.op_sel(3), &DstMods::of(m));
        Ok(())
    }

    /// Clamp f16 result bits to [0, 1] (NaN and negatives give 0).
    pub(super) fn clamp_f16_bits(&mut self, bits: Value, clamp: bool) -> Value {
        if !clamp {
            return bits;
        }
        let m7 = self.u(0x7FFF);
        let mag = self.b().and(bits, m7);
        let lim7 = self.u(0x7FFF);
        let neg = self.b().icmp(ICmp::UGt, bits, lim7);
        let inf = self.u(0x7C00);
        let nan = self.b().icmp(ICmp::UGt, mag, inf);
        let zero_c = self.b().lor(neg, nan);
        let one = self.u(0x3C00);
        let big = self.b().icmp(ICmp::UGt, mag, one);
        let lim = self.b().select(big, one, bits);
        let z = self.u(0);
        let r = self.b().select(zero_c, z, lim);
        if self.float_mode.dx10_clamp {
            return r;
        }
        self.b().select(nan, bits, r)
    }

    /// f64 value (bits) -> (lo, hi) dwords.
    fn f64_parts(&mut self, v: Value) -> (Value, Value) {
        let u = self.b().bitcast(v, Type::U64);
        self.b().split_u64(u)
    }

    fn f64_from_parts(&mut self, lo: Value, hi: Value) -> Value {
        let u = self.b().join_u64(lo, hi);
        self.b().bitcast(u, Type::F64)
    }

    /// Write an f64 result: clamp to [0, 1] (NaN gives +0); the output
    /// modifier does not apply to f64 results.
    pub(super) fn write_f64_result(&mut self, ins: &Instruction, m: &Mods, v: Value) -> Result<()> {
        let v = if m.clamp {
            let zero = self.b().f64(0.0);
            let one = self.b().f64(1.0);
            let positive = self.b().fcmp(FCmp::OGt, v, zero);
            let below = self.b().fcmp(FCmp::OLt, v, one);
            let upper = self.b().select(below, v, one);
            self.b().select(positive, upper, zero)
        } else {
            v
        };
        let (lo, hi) = self.f64_parts(v);
        let d = self.vdst(ins)?;
        self.write_vgpr(d, lo);
        self.write_vgpr(d + 1, hi);
        Ok(())
    }

    fn f64_to_int_sat(&mut self, v: Value, signed: bool) -> Value {
        let nan = self.b().fcmp(FCmp::Uno, v, v);
        let (lo_f, hi_f, safe_hi, lo_r, hi_r) =
            if signed { (-2147483648.0f64, 2147483647.0f64, 2147483647.0f64, 0x8000_0000u32, 0x7FFF_FFFFu32) } else { (0.0, 4294967295.0, 4294967295.0, 0, u32::MAX) };
        let lo = self.b().f64(lo_f);
        let hi = self.b().f64(hi_f);
        let t = self.b().trunc(v);
        let below = self.b().fcmp(FCmp::OLe, t, lo);
        let above = self.b().fcmp(FCmp::OGe, t, hi);
        let s1 = self.b().select(below, lo, t);
        let sh = self.b().f64(safe_hi);
        let s2 = self.b().select(above, sh, s1);
        let z = self.b().f64(0.0);
        let safe = self.b().select(nan, z, s2);
        let conv = if signed { self.b().ftos(safe, Type::U32) } else { self.b().ftou(safe, Type::U32) };
        let lr = self.u(lo_r);
        let hr = self.u(hi_r);
        let r = self.b().select(below, lr, conv);
        let r = self.b().select(above, hr, r);
        let zero = self.u(0);
        self.b().select(nan, zero, r)
    }

    pub(crate) fn translate_extra_valu(&mut self, ins: &Instruction, m: &Mods) -> Result<bool> {
        use Opcode::*;
        let dm = DstMods::of(m);
        match ins.opcode {
            VPipeflush | VClrexcp => {}
            VMulLegacyF32 => {
                let a = self.src_f32(ins, m, 0);
                let b = self.src_f32(ins, m, 1);
                let ab = self.f_bits(a);
                let bb = self.f_bits(b);
                let za = self.is_zero_bits(ab);
                let zb = self.is_zero_bits(bb);
                let zero = self.b().lor(za, zb);
                let p = self.b().fmul(a, b);
                let p = self.flush_tiny_product(a, b, p, None);
                let p = self.nan_result_f32(&[a, b], p, None);
                let zf = self.b().f32(0.0);
                let r = self.b().select(zero, zf, p);
                self.write_f32_result(ins, m, r)?;
            }
            // Measured on gfx1010: the product of the factors (+0 when
            // either is zero) is rounded before the add; f32 denormal
            // sources, product and result are flushed in every mode.
            // v_mac_legacy_f32 is the same with the accumulator as addend.
            VMadLegacyF32 | VMacLegacyF32 => {
                let a = self.src_f32(ins, m, 0);
                let b = self.src_f32(ins, m, 1);
                let c = if ins.opcode == VMacLegacyF32 {
                    let d = self.vdst(ins)?;
                    let mut acc = self.vgpr(d);
                    if m.abs(2) {
                        let k = self.u(0x7FFF_FFFF);
                        acc = self.b().and(acc, k);
                    }
                    if m.neg(2) {
                        let k = self.u(0x8000_0000);
                        acc = self.b().xor(acc, k);
                    }
                    let acc = self.to_f(acc);
                    if self.f32_flush & 1 != 0 {
                        self.flush_denorm(acc)
                    } else {
                        acc
                    }
                } else {
                    self.src_f32(ins, m, 2)
                };
                let ab = self.f_bits(a);
                let bb = self.f_bits(b);
                let za = self.is_zero_bits(ab);
                let zb = self.is_zero_bits(bb);
                let zero = self.b().lor(za, zb);
                let zf = self.b().f32(0.0);
                let fa = self.b().select(zero, zf, a);
                let fb = self.b().select(zero, zf, b);
                let p = self.b().fmul(fa, fb);
                let p = self.flush_tiny_product_rounded(fa, fb, p, None, false);
                // A sum of two zeros takes its sign on the bits.
                let pb = self.f_bits(p);
                let cb = self.f_bits(c);
                let pz = self.is_zero_bits(pb);
                let cz = self.is_zero_bits(cb);
                let zero_sum = self.b().land(pz, cz);
                let both = self.b().and(pb, cb);
                let sm = self.u(0x8000_0000);
                let zs = self.b().and(both, sm);
                let zs = self.to_f(zs);
                let sum = self.b().fadd(p, c);
                let sum = self.b().select(zero_sum, zs, sum);
                let invalid = self.invalid_product_f32(fa, fb);
                let r = self.nan_result_f32(&[fa, fb, c], sum, Some(invalid));
                self.write_f32_result(ins, m, r)?;
            }
            VMulHiI32I24 => {
                let a = self.src_u32(ins, m, 0);
                let b = self.src_u32(ins, m, 1);
                let z = self.u(0);
                let w = self.u(24);
                let a = self.b().bfe_s(a, z, w);
                let b = self.b().bfe_s(b, z, w);
                let r = self.b().smul_hi(a, b);
                self.write_u32_result(ins, m, r)?;
            }
            VDot4cI32I8 => {
                let d = self.vdst(ins)?;
                let acc = self.vgpr(d);
                let r = self.integer_dot(ins, m, 8, true, acc);
                self.write_vgpr(d, r);
            }
            VDot2I32I16 | VDot2U32U16 | VDot4I32I8 | VDot4U32U8 | VDot8I32I4 | VDot8U32U4 => {
                let (element, signed) = match ins.opcode {
                    VDot2I32I16 => (16, true),
                    VDot2U32U16 => (16, false),
                    VDot4I32I8 => (8, true),
                    VDot4U32U8 => (8, false),
                    VDot8I32I4 => (4, true),
                    _ => (4, false),
                };
                let acc = self.src_u32(ins, m, 2);
                let r = self.integer_dot(ins, m, element, signed, acc);
                let d = self.vdst(ins)?;
                self.write_vgpr(d, r);
            }
            VDot2F32F16 => self.dot2_f32_f16(ins, m, false)?,
            VFmacF16 => {
                let a = self.src_f16(ins, m, 0);
                let b = self.src_f16(ins, m, 1);
                let d = self.vdst(ins)?;
                let acc = self.vgpr(d);
                let z = self.u(0);
                let s = self.u(16);
                let h = self.b().bfe_u(acc, z, s);
                let c = self.half_to_f32(h);
                let (a, b, c) = (self.fma_f16_input(a), self.fma_f16_input(b), self.fma_f16_input(c));
                let r = self.fma_f16(a, b, c);
                let r = self.fma_f16_result(r);
                let r = self.clamp_f16_overflow(r, &[a, b, c]);
                let r = if m.clamp { self.saturate_f32(r) } else { r };
                let bits = self.f32_to_half_bits(r, Rounding::Rte);
                let product = self.invalid_product_f32(a, b);
                let invalid = self.invalid_operation_f32(&[a, b, c], product);
                let bits = self.default_nan_f16_on_invalid(bits, invalid);
                self.write_vgpr_half(d, bits, false, &dm);
            }
            VLdexpF16 => {
                let source = self.src_f16_bits(ins, m, 0);
                let bits = self.flush_f16_bits(source, 0x40);
                let v = self.half_to_f32(bits);
                // src1 is an f16 operand: a VOP3 float inline constant is
                // its f16 encoding (0.5 is 0x3800, an exponent of 14336).
                let e = if ins.encoding == Encoding::Vop3 { self.src_f16_word(ins, m, 1) } else { self.src_u32(ins, m, 1) };
                let z = self.u(0);
                let s16 = self.u(16);
                let e = self.b().bfe_s(e, z, s16);
                let hi = self.u(64);
                let lo = self.u((-64i32) as u32);
                let e = self.b().smin(e, hi);
                let e = self.b().smax(e, lo);
                let c127 = self.u(127);
                let be = self.b().iadd(e, c127);
                let c23 = self.u(23);
                let pw = self.b().shl(be, c23);
                let pw = self.to_f(pw);
                let scaled = self.b().fmul(v, pw);
                let mut r = self.f32_to_f16_bits_rounded(scaled, &[v]);
                if self.f16_flushes(0x80) {
                    // A product below 2^-14 flushes to the source's sign
                    // (RDNA2).
                    let sb = self.f_bits(scaled);
                    let mm = self.u(0x7FFF_FFFF);
                    let smag = self.b().and(sb, mm);
                    let lim = self.u(0x3880_0000);
                    let tiny = self.b().icmp(ICmp::ULt, smag, lim);
                    let hs = self.u(0x8000);
                    let sign = self.b().and(bits, hs);
                    r = self.b().select(tiny, sign, r);
                }
                let m7 = self.u(0x7FFF);
                let mag = self.b().and(source, m7);
                let inf = self.u(0x7C00);
                let nan = self.b().icmp(ICmp::UGt, mag, inf);
                let qb = self.nan_quiet_bit(true);
                let q = self.b().or(source, qb);
                let r = self.b().select(nan, q, r);
                let r = self.clamp_f16_bits(r, m.clamp);
                self.write_half_result(ins, m, r)?;
            }
            VFrexpMantF16 | VFrexpExpI16F16 => {
                let bits = self.src_f16_bits(ins, m, 0);
                let bits = self.flush_f16_bits(bits, 0x40);
                let v = self.half_to_f32(bits);
                let word = self.f_bits(v);
                let m7 = self.u(0x7FFF);
                let mag = self.b().and(bits, m7);
                let zero = self.u(0);
                let is_zero = self.b().icmp(ICmp::Eq, mag, zero);
                let inf = self.u(0x7C00);
                let big = self.b().icmp(ICmp::UGe, mag, inf);
                let special = self.b().lor(is_zero, big);
                if ins.opcode == VFrexpExpI16F16 {
                    let e = self.exp_bits(word);
                    let c126 = self.u(126);
                    let unbiased = self.b().isub(e, c126);
                    let r = self.b().select(special, zero, unbiased);
                    let ffff = self.u(0xFFFF);
                    let r = self.b().and(r, ffff);
                    self.write_half_result(ins, m, r)?;
                } else {
                    let km = self.u(0x807F_FFFF);
                    let kept = self.b().and(word, km);
                    let e126 = self.u(126 << 23);
                    let mant = self.b().or(kept, e126);
                    let mant = self.to_f(mant);
                    let conv = self.f32_to_half_bits(mant, Rounding::Rte);
                    let nan = self.b().icmp(ICmp::UGt, mag, inf);
                    let qb = self.nan_quiet_bit(true);
                    let q = self.b().or(bits, qb);
                    let keep = self.b().select(nan, q, bits);
                    let r = self.b().select(special, keep, conv);
                    let r = self.clamp_f16_bits(r, m.clamp);
                    self.write_half_result(ins, m, r)?;
                }
            }
            VCvtNormI16F16 | VCvtNormU16F16 => {
                let bits = self.src_f16_bits(ins, m, 0);
                let bits = self.flush_f16_bits(bits, 0x40);
                let r = self.norm_f16(bits, ins.opcode == VCvtNormI16F16);
                self.write_half_result(ins, m, r)?;
            }
            VCvtPknormI16F16 | VCvtPknormU16F16 => {
                let signed = ins.opcode == VCvtPknormI16F16;
                let a = self.src_f16_bits(ins, m, 0);
                let a = self.flush_f16_bits(a, 0x40);
                let b = self.src_f16_bits(ins, m, 1);
                let b = self.flush_f16_bits(b, 0x40);
                let lo = self.norm_f16(a, signed);
                let hi = self.norm_f16(b, signed);
                let r = self.pack_u16(lo, hi);
                self.write_u32_result(ins, m, r)?;
            }
            VPackB32F16 => {
                // With IEEE_MODE each NaN half (after abs / neg) is quieted.
                let mut halves = Vec::new();
                for i in 0..2 {
                    let h = self.src_f16_bits(ins, m, i);
                    let h = self.flush_f16_bits(h, 0x40);
                    let m7 = self.u(0x7FFF);
                    let mag = self.b().and(h, m7);
                    let inf = self.u(0x7C00);
                    let nan = self.b().icmp(ICmp::UGt, mag, inf);
                    let qb = self.nan_quiet_bit(true);
                    let q = self.b().or(h, qb);
                    halves.push(self.b().select(nan, q, h));
                }
                let (a, b) = (halves[0], halves[1]);
                let r = self.pack_u16(a, b);
                self.write_u32_result(ins, m, r)?;
            }
            VCvtF16U16 | VCvtF16I16 => {
                let signed = ins.opcode == VCvtF16I16;
                let v = self.src_u16(ins, m, 0, signed);
                let f = if signed { self.b().stof(v, Type::F32) } else { self.b().utof(v, Type::F32) };
                let bits = self.f16_conversion_bits(f, m.clamp, &[]);
                self.write_half_result(ins, m, bits)?;
            }
            VCvtI16F16 => {
                let v = self.src_f16(ins, m, 0);
                let r = self.f32_to_i32_sat(v, -32768.0, 32768.0, 32767.0, 0xFFFF_8000, 0x7FFF);
                let ffff = self.u(0xFFFF);
                let r = self.b().and(r, ffff);
                self.write_half_result(ins, m, r)?;
            }
            VSatPkU8I16 => {
                let v = self.src_u32(ins, m, 0);
                let mut parts = Vec::new();
                for off in [0u32, 16] {
                    let o = self.u(off);
                    let w = self.u(16);
                    let x = self.b().bfe_s(v, o, w);
                    let zero = self.u(0);
                    let c255 = self.u(255);
                    let x = self.b().smax(x, zero);
                    let x = self.b().smin(x, c255);
                    parts.push(x);
                }
                let eight = self.u(8);
                let hi = self.b().shl(parts[1], eight);
                let r = self.b().or(parts[0], hi);
                self.write_u32_result(ins, m, r)?;
            }
            VFrexpMantF32 => {
                let bits = self.src_mod_bits(ins, m, 0);
                let e = self.exp_bits(bits);
                let mm = self.u(0x007F_FFFF);
                let mant = self.b().and(bits, mm);
                let sm = self.u(0x8000_0000);
                let sign = self.b().and(bits, sm);
                let half = self.u(0x3F00_0000);
                let base = self.b().or(sign, half);
                let normal = self.b().or(base, mant);
                let msb = self.b().find_umsb(mant);
                let c23 = self.u(23);
                let sh = self.b().isub(c23, msb);
                let m31 = self.u(31);
                let sh = self.b().and(sh, m31);
                let shifted = self.b().shl(mant, sh);
                let frac = self.b().and(shifted, mm);
                let sub = self.b().or(base, frac);
                let zero = self.u(0);
                let mz = self.b().icmp(ICmp::Eq, mant, zero);
                let enz = self.b().icmp(ICmp::Ne, e, zero);
                let zs = self.b().select(mz, bits, sub);
                let finite = self.b().select(enz, normal, zs);
                let ff = self.u(0xFF);
                let all = self.b().icmp(ICmp::Eq, e, ff);
                let qb = self.nan_quiet_bit(false);
                let q = self.b().or(bits, qb);
                let special = self.b().select(mz, bits, q);
                let r = self.b().select(all, special, finite);
                let r = self.to_f(r);
                self.write_f32_result(ins, m, r)?;
            }
            VFrexpExpI32F32 => {
                let bits = self.src_mod_bits(ins, m, 0);
                let e = self.exp_bits(bits);
                let mm = self.u(0x007F_FFFF);
                let mant = self.b().and(bits, mm);
                let c126 = self.u(126);
                let normal = self.b().isub(e, c126);
                let msb = self.b().find_umsb(mant);
                let c148 = self.u(148);
                let sub = self.b().isub(msb, c148);
                let zero = self.u(0);
                let mnz = self.b().icmp(ICmp::Ne, mant, zero);
                let den = self.b().select(mnz, sub, zero);
                let ff = self.u(0xFF);
                let not_all = self.b().icmp(ICmp::Ne, e, ff);
                let enz = self.b().icmp(ICmp::Ne, e, zero);
                let nd = self.b().select(enz, normal, den);
                let r = self.b().select(not_all, nd, zero);
                self.write_u32_result(ins, m, r)?;
            }
            VLerpU8 => {
                let a = self.src_u32(ins, m, 0);
                let b = self.src_u32(ins, m, 1);
                let c = self.src_u32(ins, m, 2);
                let mut r = self.u(0);
                for off in [0u32, 8, 16, 24] {
                    let o = self.u(off);
                    let w8 = self.u(8);
                    let one = self.u(1);
                    let x = self.b().bfe_u(a, o, w8);
                    let y = self.b().bfe_u(b, o, w8);
                    let rr = self.b().bfe_u(c, o, one);
                    let s = self.b().iadd(x, y);
                    let s = self.b().iadd(s, rr);
                    let avg = self.b().lshr(s, one);
                    let placed = self.b().shl(avg, o);
                    r = self.b().or(r, placed);
                }
                self.write_u32_result(ins, m, r)?;
            }
            VPermB32 => {
                let hi = self.src_u32(ins, m, 0);
                let lo = self.src_u32(ins, m, 1);
                let sel = self.src_u32(ins, m, 2);
                let mut r = self.u(0);
                for byte in 0..4u32 {
                    let c = |t: &mut Self, x: u32| t.b().u32(x);
                    let sh = c(self, byte * 8);
                    let ff = c(self, 255);
                    let s = self.b().lshr(sel, sh);
                    let s = self.b().and(s, ff);
                    let three = c(self, 3);
                    let two = c(self, 2);
                    let one = c(self, 1);
                    let s3 = self.b().and(s, three);
                    let s3 = self.b().imul(s3, two);
                    let sign_index = self.b().iadd(s3, one);
                    let eight = c(self, 8);
                    let lt8 = self.b().icmp(ICmp::ULt, s, eight);
                    let index = self.b().select(lt8, s, sign_index);
                    let four = c(self, 4);
                    let lt4 = self.b().icmp(ICmp::ULt, index, four);
                    let word = self.b().select(lt4, lo, hi);
                    let ib = self.b().and(index, three);
                    let shift = self.b().imul(ib, eight);
                    let v = self.b().lshr(word, shift);
                    let v = self.b().and(v, ff);
                    let c127 = c(self, 127);
                    let neg = self.b().icmp(ICmp::UGt, v, c127);
                    let zero = c(self, 0);
                    let sign = self.b().select(neg, ff, zero);
                    let selected = self.b().select(lt8, v, sign);
                    let c12 = c(self, 12);
                    let is12 = self.b().icmp(ICmp::Eq, s, c12);
                    let fill = self.b().select(is12, zero, ff);
                    let lt12 = self.b().icmp(ICmp::ULt, s, c12);
                    let out = self.b().select(lt12, selected, fill);
                    let placed = self.b().shl(out, sh);
                    r = self.b().or(r, placed);
                }
                self.write_u32_result(ins, m, r)?;
            }
            VMullitF32 => {
                let lhs = self.src_mod_bits(ins, m, 0);
                let rhs = self.src_mod_bits(ins, m, 1);
                let lim = self.src_mod_bits(ins, m, 2);
                let lz = self.is_zero_bits(lim);
                let sm = self.u(0x8000_0000);
                let ls = self.b().and(lim, sm);
                let zero = self.u(0);
                let lneg = self.b().icmp(ICmp::Ne, ls, zero);
                let not_pos = self.b().lor(lz, lneg);
                let nm = self.u(0xFF7F_FFFE);
                let rneg_max = self.b().icmp(ICmp::UGt, rhs, nm);
                let lnan = self.is_nan_bits(lim);
                let rnan = self.is_nan_bits(rhs);
                let lowest = self.b().lor(rneg_max, lnan);
                let lowest = self.b().lor(lowest, not_pos);
                let lowest = self.b().lor(lowest, rnan);
                let fl = self.to_f(lhs);
                let fr = self.to_f(rhs);
                let p = self.b().fmul(fl, fr);
                let p = self.flush_tiny_product(fl, fr, p, None);
                let p = self.f_bits(p);
                let lhs_nan = self.is_nan_bits(lhs);
                let qb = self.nan_quiet_bit(false);
                let q = self.b().or(lhs, qb);
                let r = self.b().select(lhs_nan, q, p);
                let za = self.is_zero_bits(lhs);
                let zb = self.is_zero_bits(rhs);
                let zz = self.b().lor(za, zb);
                let r = self.b().select(zz, zero, r);
                let low = self.u(0xFF7F_FFFF);
                let r = self.b().select(lowest, low, r);
                let r = self.to_f(r);
                self.write_f32_result(ins, m, r)?;
            }
            VDivScaleF32 => self.div_scale_f32(ins, m)?,
            VDivFmasF32 => self.div_fmas_f32(ins, m)?,
            VDivFixupF32 => self.div_fixup_f32(ins, m)?,
            VDivFixupF16 => self.div_fixup_f16(ins, m)?,
            VMsadU8 => {
                let a = self.src_u32(ins, m, 0);
                let b = self.src_u32(ins, m, 1);
                let c = self.src_u32(ins, m, 2);
                let s = self.masked_sad(a, b);
                let z = self.u(0);
                let r = self.wide_add(m.clamp, s, z, c, false);
                self.write_u32_result(ins, m, r)?;
            }
            VQsadPkU16U8 | VMqsadPkU16U8 | VMqsadU32U8 => self.quad_sad(ins, m)?,
            VMadI64I32 => {
                let a = self.src_u32(ins, m, 0);
                let b = self.src_u32(ins, m, 1);
                let (al, ah) = self.read_i64_parts(ins.sources[2]);
                let lo = self.b().imul(a, b);
                let hi = self.b().smul_hi(a, b);
                let low = self.b().iadd(lo, al);
                let carry_low = self.b().icmp(ICmp::ULt, low, lo);
                let high0 = self.b().iadd(hi, ah);
                let carry0 = self.b().icmp(ICmp::ULt, high0, hi);
                let cl = self.b().bool_to_u32(carry_low);
                let high = self.b().iadd(high0, cl);
                let carry1 = self.b().icmp(ICmp::ULt, high, high0);
                let carry = self.b().lor(carry0, carry1);
                let (rl, rh) = if m.clamp {
                    let x1 = self.b().xor(hi, high);
                    let x2 = self.b().xor(ah, high);
                    let fl = self.b().and(x1, x2);
                    let c31 = self.u(31);
                    let ov = self.b().lshr(fl, c31);
                    let zero = self.u(0);
                    let ov = self.b().icmp(ICmp::Ne, ov, zero);
                    let asign = self.b().ashr(ah, c31);
                    let ones = self.u(u32::MAX);
                    let mx = self.u(0x7FFF_FFFF);
                    let sl = self.b().xor(asign, ones);
                    let sh = self.b().xor(asign, mx);
                    (self.b().select(ov, sl, low), self.b().select(ov, sh, high))
                } else {
                    (low, high)
                };
                let d = self.vdst(ins)?;
                self.write_vgpr(d, rl);
                self.write_vgpr(d + 1, rh);
                let c31 = self.u(31);
                let x = self.b().xor(hi, ah);
                let sd = self.b().lshr(x, c31);
                let cu = self.b().bool_to_u32(carry);
                let co = self.b().icmp(ICmp::Ne, sd, cu);
                let cd = ins.control.as_ref().and_then(Control::scalar_destination).unwrap_or(VCC_LO);
                self.write_mask(cd, co);
            }
            VAshrrevI64 => {
                let c = self.src_u32(ins, m, 0);
                let v = self.src_i64(ins, 1);
                let r = self.shift64(Op::AShr, v, c);
                let (lo, hi) = self.b().split_u64(r);
                let d = self.vdst(ins)?;
                self.write_vgpr(d, lo);
                self.write_vgpr(d + 1, hi);
            }
            VMulLoU16 => {
                let a = self.src_u16(ins, m, 0, false);
                let b = self.src_u16(ins, m, 1, false);
                let r = self.b().imul(a, b);
                let ffff = self.u(0xFFFF);
                let r = self.b().and(r, ffff);
                self.write_half_result(ins, m, r)?;
            }
            VMadU16 | VMadI16 => {
                let signed = ins.opcode == VMadI16;
                let a = self.src_u16(ins, m, 0, signed);
                let b = self.src_u16(ins, m, 1, signed);
                let c = self.src_u16(ins, m, 2, signed);
                let p = self.b().imul(a, b);
                let r = self.b().iadd(p, c);
                let r = if m.clamp {
                    if signed {
                        let mx = self.u(0x7FFF);
                        let mn = self.u(0xFFFF_8000);
                        let t = self.b().smin(r, mx);
                        self.b().smax(t, mn)
                    } else {
                        let mx = self.u(0xFFFF);
                        self.b().umin(r, mx)
                    }
                } else {
                    r
                };
                let ffff = self.u(0xFFFF);
                let r = self.b().and(r, ffff);
                self.write_half_result(ins, m, r)?;
            }
            VMadI32I16 => {
                let a = self.src_u16(ins, m, 0, true);
                let b = self.src_u16(ins, m, 1, true);
                let p = self.b().imul(a, b);
                let c31 = self.u(31);
                let high = self.b().ashr(p, c31);
                let c = self.src_u32(ins, m, 2);
                let r = self.wide_add(m.clamp, p, high, c, true);
                self.write_u32_result(ins, m, r)?;
            }
            VMin3I16 | VMin3U16 | VMax3I16 | VMax3U16 | VMed3I16 | VMed3U16 => {
                let signed = matches!(ins.opcode, VMin3I16 | VMax3I16 | VMed3I16);
                let a = self.src_u16(ins, m, 0, signed);
                let b = self.src_u16(ins, m, 1, signed);
                let c = self.src_u16(ins, m, 2, signed);
                let (mn, mx) = if signed { (Op::SMin, Op::SMax) } else { (Op::UMin, Op::UMax) };
                let r = match ins.opcode {
                    VMin3I16 | VMin3U16 => {
                        let ab = self.b().emit(mn, Type::U32, vec![a, b]);
                        self.b().emit(mn, Type::U32, vec![ab, c])
                    }
                    VMax3I16 | VMax3U16 => {
                        let ab = self.b().emit(mx, Type::U32, vec![a, b]);
                        self.b().emit(mx, Type::U32, vec![ab, c])
                    }
                    _ => {
                        let lo = self.b().emit(mn, Type::U32, vec![a, b]);
                        let hi = self.b().emit(mx, Type::U32, vec![a, b]);
                        let hc = self.b().emit(mn, Type::U32, vec![hi, c]);
                        self.b().emit(mx, Type::U32, vec![lo, hc])
                    }
                };
                let ffff = self.u(0xFFFF);
                let r = self.b().and(r, ffff);
                self.write_half_result(ins, m, r)?;
            }
            VAddNcI32 | VSubNcI32 => {
                let a = self.src_u32(ins, m, 0);
                let b = self.src_u32(ins, m, 1);
                let sub = ins.opcode == VSubNcI32;
                let mut r = if sub { self.b().isub(a, b) } else { self.b().iadd(a, b) };
                if m.clamp {
                    let same = if sub { self.b().xor(a, b) } else { self.b().xor(b, r) };
                    let x = self.b().xor(a, r);
                    let fl = self.b().and(x, same);
                    let c31 = self.u(31);
                    let s = self.b().lshr(fl, c31);
                    let zero = self.u(0);
                    let ov = self.b().icmp(ICmp::Ne, s, zero);
                    let asign = self.b().ashr(a, c31);
                    let mx = self.u(0x7FFF_FFFF);
                    let sat = self.b().xor(asign, mx);
                    r = self.b().select(ov, sat, r);
                }
                self.write_u32_result(ins, m, r)?;
            }
            VPkMadI16 | VPkMadU16 | VPkMulLoU16 | VPkLshlrevB16 | VPkLshrrevB16 | VPkAshrrevI16 | VPkMaxI16
            | VPkMinI16 | VPkAddU16 | VPkSubU16 | VPkMaxU16 | VPkMinU16 => {
                let signed = matches!(ins.opcode, VPkMadI16 | VPkAshrrevI16 | VPkMaxI16 | VPkMinI16);
                let mut halves = Vec::new();
                for high in [false, true] {
                    let r = match ins.opcode {
                        VPkLshlrevB16 | VPkLshrrevB16 | VPkAshrrevI16 => {
                            let c = self.src_packed_u16(ins, m, 0, high, false);
                            let f = self.u(15);
                            let c = self.b().and(c, f);
                            let v = self.src_packed_u16(ins, m, 1, high, signed);
                            let op = match ins.opcode {
                                VPkLshlrevB16 => Op::Shl,
                                VPkLshrrevB16 => Op::LShr,
                                _ => Op::AShr,
                            };
                            self.b().emit(op, Type::U32, vec![v, c])
                        }
                        VPkMadI16 | VPkMadU16 => {
                            let a = self.src_packed_u16(ins, m, 0, high, signed);
                            let b = self.src_packed_u16(ins, m, 1, high, signed);
                            let c = self.src_packed_u16(ins, m, 2, high, signed);
                            let mut p = self.b().imul(a, b);
                            if m.clamp && !signed {
                                let ffff = self.u(0xFFFF);
                                p = self.b().umin(p, ffff);
                            }
                            let s = self.b().iadd(p, c);
                            self.sat16(s, signed, m.clamp)
                        }
                        _ => {
                            let a = self.src_packed_u16(ins, m, 0, high, signed);
                            let b = self.src_packed_u16(ins, m, 1, high, signed);
                            match ins.opcode {
                                VPkMulLoU16 => self.b().imul(a, b),
                                VPkAddU16 => {
                                    let s = self.b().iadd(a, b);
                                    self.sat16(s, false, m.clamp)
                                }
                                VPkSubU16 => {
                                    let s = self.b().isub(a, b);
                                    self.sat16(s, false, m.clamp)
                                }
                                VPkMaxI16 => self.b().smax(a, b),
                                VPkMinI16 => self.b().smin(a, b),
                                VPkMaxU16 => self.b().umax(a, b),
                                _ => self.b().umin(a, b),
                            }
                        }
                    };
                    halves.push(r);
                }
                let r = self.pack_u16(halves[0], halves[1]);
                let d = self.vdst(ins)?;
                self.write_vgpr(d, r);
            }
            VSwapB32 => {
                let src = ins.sources[0];
                if src.kind != OperandKind::VectorRegister {
                    return self.invalid(ins, "v_swap source is not a VGPR");
                }
                let d = self.vdst(ins)?;
                let a = self.vgpr(src.value);
                let b = self.vgpr(d);
                self.write_vgpr(d, a);
                self.write_vgpr(src.value, b);
            }
            // ---- f64 ----
            VAddF64 | VMinF64 | VMaxF64 => {
                let a = self.src_f64(ins, m, 0);
                let b = self.src_f64(ins, m, 1);
                // v_min/max_f64 flush input denormals by FLOAT_MODE bit 6.
                let (a, b) = if ins.opcode == VAddF64 { (a, b) } else { (self.flush_f64_input(a), self.flush_f64_input(b)) };
                let r = match ins.opcode {
                    VAddF64 => self.b().fadd(a, b),
                    VMinF64 => self.b().fmin(a, b),
                    _ => self.b().fmax(a, b),
                };
                let r = if ins.opcode == VAddF64 {
                    r
                } else if self.float_mode.ieee {
                    // A signaling NaN operand (the first one wins) is
                    // returned quieted.
                    let ca = self.f64_class(a);
                    let cb = self.f64_class(b);
                    let qb = self.f64_quiet(&cb);
                    let r = self.b().select(cb.snan, qb, r);
                    let qa = self.f64_quiet(&ca);
                    self.b().select(ca.snan, qa, r)
                } else {
                    // With IEEE_MODE clear a NaN operand gives the other
                    // one (measured on gfx1036).
                    let ca = self.f64_class(a);
                    let cb = self.f64_class(b);
                    let r = self.b().select(ca.nan, b, r);
                    self.b().select(cb.nan, a, r)
                };
                self.write_f64_result(ins, m, r)?;
            }
            VLdexpF64 => {
                let a = self.src_f64(ins, m, 0);
                let e = self.src_u32(ins, m, 1);
                let r = self.b().ldexp(a, e);
                let r = self.f64_unary_nan(a, r);
                self.write_f64_result(ins, m, r)?;
            }
            VTruncF64 | VCeilF64 | VRndneF64 | VFloorF64 | VFractF64 => {
                let a = self.src_f64(ins, m, 0);
                let r = match ins.opcode {
                    VTruncF64 => {
                        let r = self.b().trunc(a);
                        self.f64_unary_nan(a, r)
                    }
                    VCeilF64 => {
                        let r = self.b().ceil(a);
                        self.f64_unary_nan(a, r)
                    }
                    VRndneF64 => {
                        let r = self.b().round_even(a);
                        self.f64_unary_nan(a, r)
                    }
                    VFloorF64 => {
                        let r = self.b().floor(a);
                        self.f64_unary_nan(a, r)
                    }
                    _ => {
                        let fl = self.b().floor(a);
                        let fr = self.b().fsub(a, fl);
                        let lim = self.b().f64(f64::from_bits(0x3FEF_FFFF_FFFF_FFFF));
                        let over = self.b().fcmp(FCmp::OGt, fr, lim);
                        let r = self.b().select(over, lim, fr);
                        // Invalid (infinite input) gives the default NaN, a
                        // NaN input passes through quieted.
                        let invalid = self.b().fcmp(FCmp::Uno, r, r);
                        let dn = self.b().f64(f64::from_bits(0xFFF8_0000_0000_0000));
                        let r = self.b().select(invalid, dn, r);
                        self.f64_unary_nan(a, r)
                    }
                };
                self.write_f64_result(ins, m, r)?;
            }
            VCvtF64F32 => {
                let v = self.src_f32(ins, m, 0);
                let r = self.b().fconvert(v, Type::F64, Rounding::Rte);
                let vb = self.f_bits(v);
                let nan = self.is_nan_bits(vb);
                let mm = self.u(0x007F_FFFF);
                let mant = self.b().and(vb, mm);
                let c29 = self.u(29);
                let lo = self.b().shl(mant, c29);
                let sm = self.u(0x8000_0000);
                let sign = self.b().and(vb, sm);
                let e = self.u(0x7FF0_0000);
                let hi = self.b().or(sign, e);
                let c3 = self.u(3);
                let top = self.b().lshr(mant, c3);
                let hi = self.b().or(hi, top);
                let qb = self.u(if self.float_mode.ieee { 0x0008_0000 } else { 0 });
                let hi = self.b().or(hi, qb);
                let widened = self.f64_from_parts(lo, hi);
                let r = self.b().select(nan, widened, r);
                self.write_f64_result(ins, m, r)?;
            }
            VCvtI32F64 | VCvtU32F64 => {
                let v = self.src_f64(ins, m, 0);
                let r = self.f64_to_int_sat(v, ins.opcode == VCvtI32F64);
                self.write_u32_result(ins, m, r)?;
            }
            VFrexpExpI32F64 | VFrexpMantF64 => {
                let v = self.src_f64(ins, m, 0);
                let (lo, hi) = self.f64_parts(v);
                let o = self.u(20);
                let w = self.u(11);
                let e = self.b().bfe_u(hi, o, w);
                let zero = self.u(0);
                let c7ff = self.u(0x7FF);
                let special = self.b().icmp(ICmp::Eq, e, c7ff);
                let mag_hi = self.mag(hi);
                let mz = self.b().or(mag_hi, lo);
                let is_zero = self.b().icmp(ICmp::Eq, mz, zero);
                // Normal and zero inputs only (denormal f64 inputs are
                // treated as zero).
                let denorm = self.b().icmp(ICmp::Eq, e, zero);
                if ins.opcode == VFrexpExpI32F64 {
                    let c1022 = self.u(1022);
                    let ex = self.b().isub(e, c1022);
                    let bad = self.b().lor(special, denorm);
                    let r = self.b().select(bad, zero, ex);
                    self.write_u32_result(ins, m, r)?;
                } else {
                    let km = self.u(0x800F_FFFF);
                    let kept = self.b().and(hi, km);
                    let e1022 = self.u(1022 << 20);
                    let nh = self.b().or(kept, e1022);
                    let r = self.f64_from_parts(lo, nh);
                    let keep = self.b().lor(special, is_zero);
                    let r = self.b().select(keep, v, r);
                    let r = self.f64_unary_nan(v, r);
                    self.write_f64_result(ins, m, r)?;
                }
            }
            VDivScaleF64 => self.div_scale_f64(ins, m)?,
            VDivFmasF64 => self.div_fmas_f64(ins, m)?,
            VDivFixupF64 => self.div_fixup_f64(ins, m)?,
            VTrigPreopF64 => self.trig_preop_f64(ins, m)?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn sat16(&mut self, v: Value, signed: bool, clamp: bool) -> Value {
        if !clamp {
            return v;
        }
        let lo = self.u(if signed { 0xFFFF_8000 } else { 0 });
        let hi = self.u(if signed { 0x7FFF } else { 0xFFFF });
        let t = self.b().smax(v, lo);
        self.b().smin(t, hi)
    }

    fn masked_sad(&mut self, a: Value, b: Value) -> Value {
        let mut sum = self.u(0);
        for off in [0u32, 8, 16, 24] {
            let o = self.u(off);
            let w = self.u(8);
            let fa = self.b().bfe_u(a, o, w);
            let fb = self.b().bfe_u(b, o, w);
            let lo = self.b().umin(fa, fb);
            let hi = self.b().umax(fa, fb);
            let d = self.b().isub(hi, lo);
            let zero = self.u(0);
            let r = self.b().icmp(ICmp::Ne, fb, zero);
            let d = self.b().select(r, d, zero);
            sum = self.b().iadd(sum, d);
        }
        sum
    }

    fn byte_sad_plain(&mut self, a: Value, b: Value, masked: bool) -> Value {
        if masked {
            return self.masked_sad(a, b);
        }
        let mut sum = self.u(0);
        for off in [0u32, 8, 16, 24] {
            let o = self.u(off);
            let w = self.u(8);
            let fa = self.b().bfe_u(a, o, w);
            let fb = self.b().bfe_u(b, o, w);
            let lo = self.b().umin(fa, fb);
            let hi = self.b().umax(fa, fb);
            let d = self.b().isub(hi, lo);
            sum = self.b().iadd(sum, d);
        }
        sum
    }

    fn quad_sad(&mut self, ins: &Instruction, m: &Mods) -> Result<()> {
        use Opcode::*;
        let masked = ins.opcode != VQsadPkU16U8;
        let wide = ins.opcode == VMqsadU32U8;
        let (s0, s1) = self.read_u64_parts(ins.sources[0]);
        let reference = self.src_u32(ins, m, 1);
        let mut sads = Vec::new();
        for shift in [0u32, 8, 16, 24] {
            let window = if shift == 0 {
                s0
            } else {
                let sh = self.u(shift);
                let back = self.u(32 - shift);
                let a = self.b().lshr(s0, sh);
                let b = self.b().shl(s1, back);
                self.b().or(a, b)
            };
            sads.push(self.byte_sad_plain(window, reference, masked));
        }
        let d = self.vdst(ins)?;
        if wide {
            let src2 = ins.sources[2];
            let acc: Vec<Value> = if src2.kind == OperandKind::VectorRegister {
                (0..4).map(|k| self.vgpr(src2.value + k)).collect()
            } else {
                let (a, b) = self.read_u64_parts(src2);
                vec![a, b, a, b]
            };
            for k in 0..4 {
                let z = self.u(0);
                let r = self.wide_add(m.clamp, sads[k], z, acc[k], false);
                self.write_vgpr(d + k as u32, r);
            }
            return Ok(());
        }
        let (a0, a1) = self.read_u64_parts(ins.sources[2]);
        for (k, acc) in [a0, a1].into_iter().enumerate() {
            let ffff = self.u(0xFFFF);
            let lo_acc = self.b().and(acc, ffff);
            let sixteen = self.u(16);
            let hi_acc = self.b().lshr(acc, sixteen);
            let lo = self.b().iadd(lo_acc, sads[2 * k]);
            let hi = self.b().iadd(hi_acc, sads[2 * k + 1]);
            let (lo, hi) = if m.clamp { (self.b().umin(lo, ffff), self.b().umin(hi, ffff)) } else { (lo, hi) };
            let r = self.pack_u16(lo, hi);
            self.write_vgpr(d + k as u32, r);
        }
        Ok(())
    }

    fn div_scale_f32(&mut self, ins: &Instruction, m: &Mods) -> Result<()> {
        let value = self.src_f32(ins, m, 0);
        let bits = self.f_bits(value);
        let den = self.src_mod_bits(ins, m, 1);
        let num = self.src_mod_bits(ins, m, 2);
        let de = self.exp_bits(den);
        let ne = self.exp_bits(num);
        let diff = self.b().isub(ne, de);
        let c96 = self.u(96);
        let near_max = self.b().icmp(ICmp::SGe, diff, c96);
        let cm96 = self.u((-96i32) as u32);
        let near_min = self.b().icmp(ICmp::SLe, diff, cm96);
        let zero = self.u(0);
        let den_denorm = self.b().icmp(ICmp::Eq, de, zero);
        let c253 = self.u(253);
        let den_huge = self.b().icmp(ICmp::UGe, de, c253);
        let c24 = self.u(24);
        let num_tiny = self.b().icmp(ICmp::ULe, ne, c24);
        let is_den = self.b().icmp(ICmp::Eq, bits, den);
        let not_near_max = self.b().lnot(near_max);
        let not_denorm = self.b().lnot(den_denorm);
        let normal_range = self.b().land(not_near_max, not_denorm);
        let not_huge = self.b().lnot(den_huge);
        let below_huge = self.b().land(normal_range, not_huge);
        let t1 = self.b().land(near_max, is_den);
        let t2 = self.b().land(not_near_max, den_denorm);
        let not_is_den = self.b().lnot(is_den);
        let t3 = self.b().land(near_min, not_is_den);
        let not_near_min = self.b().lnot(near_min);
        let t4 = self.b().land(not_near_min, num_tiny);
        let t34 = self.b().lor(t3, t4);
        let t5 = self.b().land(below_huge, t34);
        let t12 = self.b().lor(t1, t2);
        let scale_up = self.b().lor(t12, t5);
        let nh = self.b().land(normal_range, den_huge);
        let nmd = self.b().lor(not_near_min, is_den);
        let scale_down = self.b().land(nh, nmd);
        let up_k = self.b().f32(18446744073709551616.0);
        let down_k = self.b().f32(5.421_010_862_427_522e-20);
        let up = self.b().fmul(value, up_k);
        let down = self.b().fmul(value, down_k);
        let vnan = self.is_nan_bits(bits);
        let not_nan = self.b().lnot(vnan);
        let su = self.b().land(scale_up, not_nan);
        let sd = self.b().land(scale_down, not_nan);
        let ub = self.f_bits(up);
        let db = self.f_bits(down);
        let s1 = self.b().select(sd, db, bits);
        let scaled = self.b().select(su, ub, s1);
        let dz = self.is_zero_bits(den);
        let nz = self.is_zero_bits(num);
        let z = self.b().lor(dz, nz);
        let dn = self.u(0xFFC0_0000);
        let r = self.b().select(z, dn, scaled);
        let r = self.to_f(r);
        self.write_f32_result(ins, m, r)?;
        let nm = self.b().land(not_denorm, near_min);
        let vcc = self.b().lor(near_max, nm);
        let cd = ins.control.as_ref().and_then(Control::scalar_destination).unwrap_or(VCC_LO);
        self.write_mask(cd, vcc);
        Ok(())
    }

    fn div_fmas_f32(&mut self, ins: &Instruction, m: &Mods) -> Result<()> {
        let scale = self.read_mask(Operand::scalar(VCC_LO));
        let lhs = self.src_f32(ins, m, 0);
        let rhs = self.src_f32(ins, m, 1);
        let add = self.src_f32(ins, m, 2);
        let addb = self.f_bits(add);
        let ae = self.exp_bits(addb);
        let c128 = self.u(128);
        let up = self.b().icmp(ICmp::UGe, ae, c128);
        let pu = self.b().f32(18446744073709551616.0);
        let pd = self.b().f32(5.421_010_862_427_522e-20);
        let power = self.b().select(up, pu, pd);
        let plain = self.b().fma(lhs, rhs, add);
        let pb = self.f_bits(plain);
        let p_inf = self.is_inf_bits(pb);
        // Only an overflowed sum scaled by 2^-64 is recomputed from the
        // scaled larger factor and addend. Scaling up (|src2| >= 2^64) a
        // denormal sum is impossible: a non-zero sum is at least 2^-47, so
        // only an exact cancellation (+0) reaches it, and +0 * 2^64 is +0.
        let not_up = self.b().lnot(up);
        let rescale = self.b().land(not_up, p_inf);
        let lb = self.f_bits(lhs);
        let rb = self.f_bits(rhs);
        let lm = self.mag(lb);
        let rm = self.mag(rb);
        let larger = self.b().icmp(ICmp::UGt, lm, rm);
        let scaled = self.b().select(larger, lhs, rhs);
        let other = self.b().select(larger, rhs, lhs);
        let sp = self.b().fmul(scaled, power);
        let ap = self.b().fmul(add, power);
        let rescaled = self.b().fma(sp, other, ap);
        let afterwards = self.b().fmul(plain, power);
        let scaled_result = self.b().select(rescale, rescaled, afterwards);
        let computed = self.b().select(scale, scaled_result, plain);
        if self.options.float_rules & super::float_rules::DIV_FMAS_NAN != 0 {
            // The NaN result follows v_fma_f32 (hardware rows): the first
            // NaN source, quieted only with IEEE_MODE, and no NaN addend
            // for inf * 0.
            let invalid = self.invalid_product_f32(lhs, rhs);
            let r = self.nan_result_f32(&[lhs, rhs, add], computed, Some(invalid));
            return self.write_f32_result(ins, m, r);
        }
        let cb = self.f_bits(computed);
        let cn = self.is_nan_bits(cb);
        let dn = self.u(0xFFC0_0000);
        let mut r = self.b().select(cn, dn, cb);
        let qb = self.u(0x0040_0000);
        for v in [addb, rb, lb] {
            let n = self.is_nan_bits(v);
            let q = self.b().or(v, qb);
            r = self.b().select(n, q, r);
        }
        let r = self.to_f(r);
        self.write_f32_result(ins, m, r)
    }

    fn div_fixup_f32(&mut self, ins: &Instruction, m: &Mods) -> Result<()> {
        // With f32 denormals flushed only the denominator and the numerator
        // are flushed: a denormal quotient is kept.
        let flush = self.f32_flush;
        self.f32_flush = 0;
        let q = self.src_mod_bits(ins, m, 0);
        self.f32_flush = flush;
        let den = self.src_mod_bits(ins, m, 1);
        let num = self.src_mod_bits(ins, m, 2);
        let x = self.b().xor(den, num);
        let sm = self.u(0x8000_0000);
        let sign = self.b().and(x, sm);
        let infb = self.u(0x7F80_0000);
        let inf = self.b().or(sign, infb);
        let ne = self.exp_bits(num);
        let de = self.exp_bits(den);
        let diff = self.b().isub(ne, de);
        let m150 = self.u((-150i32) as u32);
        let underflow = self.b().icmp(ICmp::SLt, diff, m150);
        let qm = self.mag(q);
        let mut r = self.b().or(sign, qm);
        let qnan = self.is_nan_bits(q);
        r = self.b().select(qnan, inf, r);
        r = self.b().select(underflow, sign, r);
        let dinf = self.is_inf_bits(den);
        let nzero = self.is_zero_bits(num);
        let c1 = self.b().lor(dinf, nzero);
        r = self.b().select(c1, sign, r);
        let dzero = self.is_zero_bits(den);
        let ninf = self.is_inf_bits(num);
        let c2 = self.b().lor(dzero, ninf);
        r = self.b().select(c2, inf, r);
        let zz = self.b().land(dzero, nzero);
        let ii = self.b().land(dinf, ninf);
        let invalid = self.b().lor(zz, ii);
        let dn = self.u(0xFFC0_0000);
        r = self.b().select(invalid, dn, r);
        let qb = self.nan_quiet_bit(false);
        let dnan = self.is_nan_bits(den);
        let dq = self.b().or(den, qb);
        r = self.b().select(dnan, dq, r);
        let nnan = self.is_nan_bits(num);
        let nq = self.b().or(num, qb);
        r = self.b().select(nnan, nq, r);
        let r = self.to_f(r);
        self.write_f32_result(ins, m, r)
    }

    fn div_fixup_f16(&mut self, ins: &Instruction, m: &Mods) -> Result<()> {
        let mut bits = Vec::new();
        let mut mags = Vec::new();
        for i in 0..3 {
            let b = self.src_f16_bits(ins, m, i);
            // The denominator and numerator follow the f16 input denormal
            // mode (RDNA2).
            let b = if i == 0 { b } else { self.flush_f16_bits(b, 0x40) };
            let m7 = self.u(0x7FFF);
            mags.push(self.b().and(b, m7));
            bits.push(b);
        }
        let inf = self.u(0x7C00);
        let zero = self.u(0);
        let is_nan = |t: &mut Self, i: usize, mags: &[Value]| t.b().icmp(ICmp::UGt, mags[i], inf);
        let is_inf = |t: &mut Self, i: usize, mags: &[Value]| t.b().icmp(ICmp::Eq, mags[i], inf);
        let is_zero = |t: &mut Self, i: usize, mags: &[Value]| t.b().icmp(ICmp::Eq, mags[i], zero);
        let x = self.b().xor(bits[1], bits[2]);
        let sb = self.u(0x8000);
        let sign = self.b().and(x, sb);
        let infinity = self.b().or(sign, inf);
        let base = self.b().or(sign, mags[0]);
        let n0 = is_nan(self, 0, &mags);
        let mut r = self.b().select(n0, infinity, base);
        let i1 = is_inf(self, 1, &mags);
        let z2 = is_zero(self, 2, &mags);
        let c = self.b().lor(i1, z2);
        r = self.b().select(c, sign, r);
        let z1 = is_zero(self, 1, &mags);
        let i2 = is_inf(self, 2, &mags);
        let c = self.b().lor(z1, i2);
        r = self.b().select(c, infinity, r);
        let bz = self.b().land(z1, z2);
        let bi = self.b().land(i1, i2);
        let c = self.b().lor(bz, bi);
        let dn = self.u(0xFE00);
        r = self.b().select(c, dn, r);
        let qb = self.nan_quiet_bit(true);
        for i in [1usize, 2] {
            let n = is_nan(self, i, &mags);
            let q = self.b().or(bits[i], qb);
            r = self.b().select(n, q, r);
        }
        if self.float_mode.fp16_overflow {
            // FP16_OVFL keeps the infinity a zero denominator or an
            // infinite numerator gives.
            let keep = self.b().lor(z1, i2);
            let not_keep = self.b().lnot(keep);
            let m7 = self.u(0x7FFF);
            let rm = self.b().and(r, m7);
            let is_inf = self.b().icmp(ICmp::Eq, rm, inf);
            let overflow = self.b().land(is_inf, not_keep);
            let sb = self.u(0x8000);
            let rs = self.b().and(r, sb);
            let largest = self.u(0x7BFF);
            let largest = self.b().or(rs, largest);
            r = self.b().select(overflow, largest, r);
        }
        let r = self.clamp_f16_bits(r, m.clamp);
        self.write_half_result(ins, m, r)
    }

    // ---- f64 helpers ----

    /// Classify an f64 value by its bits.
    fn f64_class(&mut self, v: Value) -> F64Class {
        let (lo, hi) = self.f64_parts(v);
        let ah = self.mag(hi);
        let inf_hi = self.u(0x7FF0_0000);
        let zero = self.u(0);
        let above = self.b().icmp(ICmp::UGt, ah, inf_hi);
        let at_inf = self.b().icmp(ICmp::Eq, ah, inf_hi);
        let lo_nz = self.b().icmp(ICmp::Ne, lo, zero);
        let lo_nan = self.b().land(at_inf, lo_nz);
        let nan = self.b().lor(above, lo_nan);
        let qb = self.u(0x0008_0000);
        let qbit = self.b().and(hi, qb);
        let signaling = self.b().icmp(ICmp::Eq, qbit, zero);
        let snan = self.b().land(nan, signaling);
        let lo_z = self.b().lnot(lo_nz);
        let inf = self.b().land(at_inf, lo_z);
        let mo = self.b().or(ah, lo);
        let is_zero = self.b().icmp(ICmp::Eq, mo, zero);
        let o = self.u(20);
        let w = self.u(11);
        let exp = self.b().bfe_u(hi, o, w);
        F64Class { lo, hi, nan, snan, inf, zero: is_zero, exp }
    }

    /// A NaN as an f64 operation returns it: quieted only with IEEE_MODE
    /// (measured on gfx1036 for the unary forms, v_ldexp_f64,
    /// v_div_fixup_f64 and v_cvt_f64_f32).
    fn f64_nan_result(&mut self, c: &F64Class) -> Value {
        if self.float_mode.ieee {
            self.f64_quiet(c)
        } else {
            self.f64_from_parts(c.lo, c.hi)
        }
    }

    /// An f64 unary result with a NaN input passed through
    /// [`Self::f64_nan_result`].
    pub(super) fn f64_unary_nan(&mut self, input: Value, result: Value) -> Value {
        let c = self.f64_class(input);
        let n = self.f64_nan_result(&c);
        self.b().select(c.nan, n, result)
    }

    /// The value with its quiet bit set.
    fn f64_quiet(&mut self, c: &F64Class) -> Value {
        let qb = self.u(0x0008_0000);
        let hi = self.b().or(c.hi, qb);
        self.f64_from_parts(c.lo, hi)
    }

    fn f64_const(&mut self, bits: u64) -> Value {
        self.b().f64(f64::from_bits(bits))
    }

    fn div_scale_f64(&mut self, ins: &Instruction, m: &Mods) -> Result<()> {
        let value = self.src_f64(ins, m, 0);
        let den = self.src_f64(ins, m, 1);
        let num = self.src_f64(ins, m, 2);
        let cv = self.f64_class(value);
        let cd = self.f64_class(den);
        let cn = self.f64_class(num);
        let diff = self.b().isub(cn.exp, cd.exp);
        let c768 = self.u(768);
        let near_max = self.b().icmp(ICmp::SGe, diff, c768);
        let cm768 = self.u((-768i32) as u32);
        let near_min = self.b().icmp(ICmp::SLe, diff, cm768);
        let zero = self.u(0);
        let den_denorm = self.b().icmp(ICmp::Eq, cd.exp, zero);
        let c2045 = self.u(2045);
        let den_huge = self.b().icmp(ICmp::UGe, cd.exp, c2045);
        let c53 = self.u(53);
        let num_tiny = self.b().icmp(ICmp::ULe, cn.exp, c53);
        let el = self.b().icmp(ICmp::Eq, cv.lo, cd.lo);
        let eh = self.b().icmp(ICmp::Eq, cv.hi, cd.hi);
        let is_den = self.b().land(el, eh);
        let not_near_max = self.b().lnot(near_max);
        let not_denorm = self.b().lnot(den_denorm);
        let normal_range = self.b().land(not_near_max, not_denorm);
        let not_huge = self.b().lnot(den_huge);
        let below_huge = self.b().land(normal_range, not_huge);
        let t1 = self.b().land(near_max, is_den);
        let t2 = self.b().land(not_near_max, den_denorm);
        let not_is_den = self.b().lnot(is_den);
        let t3 = self.b().land(near_min, not_is_den);
        let not_near_min = self.b().lnot(near_min);
        let t4 = self.b().land(not_near_min, num_tiny);
        let t34 = self.b().lor(t3, t4);
        let t5 = self.b().land(below_huge, t34);
        let t12 = self.b().lor(t1, t2);
        let scale_up = self.b().lor(t12, t5);
        let nh = self.b().land(normal_range, den_huge);
        let nmd = self.b().lor(not_near_min, is_den);
        let scale_down = self.b().land(nh, nmd);
        let up = self.u(128);
        let down = self.u((-128i32) as u32);
        let power = self.b().select(scale_up, up, down);
        let scaled = self.b().ldexp(value, power);
        let any = self.b().lor(scale_up, scale_down);
        let not_nan = self.b().lnot(cv.nan);
        let rescale = self.b().land(any, not_nan);
        let r = self.b().select(rescale, scaled, value);
        let z = self.b().lor(cd.zero, cn.zero);
        let dn = self.f64_const(0xFFF8_0000_0000_0000);
        let r = self.b().select(z, dn, r);
        self.write_f64_result(ins, m, r)?;
        let nm = self.b().land(not_denorm, near_min);
        let vcc = self.b().lor(near_max, nm);
        let sd = ins.control.as_ref().and_then(Control::scalar_destination).unwrap_or(VCC_LO);
        self.write_mask(sd, vcc);
        Ok(())
    }

    /// `a * b + c`, scaled by 2^±128 when VCC is set (up when |c| >= 2),
    /// rounded once.
    fn div_fmas_f64(&mut self, ins: &Instruction, m: &Mods) -> Result<()> {
        let scale = self.read_mask(Operand::scalar(VCC_LO));
        let lhs = self.src_f64(ins, m, 0);
        let rhs = self.src_f64(ins, m, 1);
        let add = self.src_f64(ins, m, 2);
        let ca = self.f64_class(lhs);
        let cb = self.f64_class(rhs);
        let cc = self.f64_class(add);
        let c1024 = self.u(1024);
        let up = self.b().icmp(ICmp::UGe, cc.exp, c1024);
        let pu = self.f64_const(0x47F0_0000_0000_0000);
        let pd = self.f64_const(0x37F0_0000_0000_0000);
        let power = self.b().select(up, pu, pd);
        let plain = self.b().fma(lhs, rhs, add);
        let cp = self.f64_class(plain);
        let zero = self.u(0);
        let pe_zero = self.b().icmp(ICmp::Eq, cp.exp, zero);
        let not_up = self.b().lnot(up);
        // Scaling up: an underflowed sum is recomputed from scaled operands
        // (the smaller factor and the addend, both exact).
        let a1 = self.b().land(up, pe_zero);
        // Scaling down: an overflowed sum is recomputed from the scaled
        // larger factor and addend.
        let a2 = self.b().land(not_up, cp.inf);
        let rescale = self.b().lor(a1, a2);
        let al = self.b().fabs(lhs);
        let ar = self.b().fabs(rhs);
        let larger = self.b().fcmp(FCmp::OGt, al, ar);
        let not_larger = self.b().lnot(larger);
        let s1 = self.b().land(up, not_larger);
        let s2 = self.b().land(not_up, larger);
        let scale_lhs = self.b().lor(s1, s2);
        let scaled = self.b().select(scale_lhs, lhs, rhs);
        let other = self.b().select(scale_lhs, rhs, lhs);
        let sp = self.b().fmul(scaled, power);
        let ap = self.b().fmul(add, power);
        let rescaled = self.b().fma(sp, other, ap);
        let afterwards = self.b().fmul(plain, power);
        let mut scaled_result = self.b().select(rescale, rescaled, afterwards);
        // Scaling down into the denormal range: the exact sum, computed at
        // 2^900 (|c| < 2, so the smaller factor is below 2 as well) and
        // rounded to odd, is scaled and rounded once.
        let c128 = self.u(128);
        let pe_low = self.b().icmp(ICmp::ULe, cp.exp, c128);
        let not_inf = self.b().lnot(cp.inf);
        let pe_low = self.b().land(pe_low, not_inf);
        let tiny = self.b().land(not_up, pe_low);
        let small = self.b().select(larger, rhs, lhs);
        let big = self.b().select(larger, lhs, rhs);
        let k900 = self.f64_const(0x7830_0000_0000_0000);
        let small = self.b().fmul(small, k900);
        let c900 = self.b().fmul(add, k900);
        let ph = self.b().fmul(small, big);
        let nph = self.b().fneg(ph);
        let pl = self.b().fma(small, big, nph);
        let (s_hi, s_lo) = self.two_sum(c900, ph);
        // The exact sum is s_hi + s_lo + pl; the remainder rounded to odd
        // keeps it exact enough for one rounding below.
        let err = self.add_round_to_odd(s_lo, pl);
        let odd = self.add_round_to_odd(s_hi, err);
        let back = self.u((-1028i32) as u32);
        let near = self.b().ldexp(odd, back);
        // The scaled sum (|x| < 2^6, the result below 2^-1022) rounds to
        // the denormal grid 2^-1074, which is 2^-46 here: add ±2^6 (whose
        // binade has that ulp), round once, and take the constant away.
        // Rounding to 53 bits first and then to the grid would round twice
        // (round-to-odd needs two spare bits; a denormal result at
        // 2^-1023 keeps 52 of the 53).
        let (t_hi, t_lo) = self.two_sum(s_hi, err);
        let zero_f = self.b().f64(0.0);
        let negative = self.b().fcmp(FCmp::OLt, t_hi, zero_f);
        let p64 = self.b().f64(64.0);
        let m64 = self.b().f64(-64.0);
        let bias = self.b().select(negative, m64, p64);
        let (u_hi, u_lo) = self.two_sum(bias, t_hi);
        let w = self.add_round_to_odd(u_lo, t_lo);
        let y = self.b().fadd(u_hi, w);
        let g = self.b().fsub(y, bias);
        let on_grid = self.b().ldexp(g, back);
        let at = self.b().fabs(t_hi);
        let below = self.b().fcmp(FCmp::OLt, at, p64);
        // A zero keeps the sign the narrowing gives it.
        let nonzero = self.b().fcmp(FCmp::ONe, g, zero_f);
        let use_grid = self.b().land(below, nonzero);
        let exact = self.b().select(use_grid, on_grid, near);
        scaled_result = self.b().select(tiny, exact, scaled_result);
        let computed = self.b().select(scale, scaled_result, plain);
        let invalid = self.b().fcmp(FCmp::Uno, computed, computed);
        let dn = self.f64_const(0xFFF8_0000_0000_0000);
        let mut r = self.b().select(invalid, dn, computed);
        for c in [&cc, &cb, &ca] {
            let q = self.f64_quiet(c);
            r = self.b().select(c.nan, q, r);
        }
        self.write_f64_result(ins, m, r)
    }

    fn div_fixup_f64(&mut self, ins: &Instruction, m: &Mods) -> Result<()> {
        let q = self.src_f64(ins, m, 0);
        // The denominator and numerator follow the f64 input denormal
        // mode (RDNA2).
        let den = self.src_f64(ins, m, 1);
        let den = self.flush_f64_input(den);
        let num = self.src_f64(ins, m, 2);
        let num = self.flush_f64_input(num);
        let cq = self.f64_class(q);
        let cd = self.f64_class(den);
        let cn = self.f64_class(num);
        let x = self.b().xor(cd.hi, cn.hi);
        let sm = self.u(0x8000_0000);
        let sign = self.b().and(x, sm);
        let zero = self.u(0);
        let signed_zero = self.f64_from_parts(zero, sign);
        let ih = self.u(0x7FF0_0000);
        let ihs = self.b().or(sign, ih);
        let inf = self.f64_from_parts(zero, ihs);
        let dn = self.f64_const(0xFFF8_0000_0000_0000);
        let diff = self.b().isub(cn.exp, cd.exp);
        let m1075 = self.u((-1075i32) as u32);
        let underflow = self.b().icmp(ICmp::SLt, diff, m1075);
        let qm = self.mag(cq.hi);
        let rh = self.b().or(sign, qm);
        let mut r = self.f64_from_parts(cq.lo, rh);
        r = self.b().select(cq.nan, inf, r);
        r = self.b().select(underflow, signed_zero, r);
        let c1 = self.b().lor(cd.inf, cn.zero);
        r = self.b().select(c1, signed_zero, r);
        let c2 = self.b().lor(cd.zero, cn.inf);
        r = self.b().select(c2, inf, r);
        let zz = self.b().land(cd.zero, cn.zero);
        let ii = self.b().land(cd.inf, cn.inf);
        let invalid = self.b().lor(zz, ii);
        r = self.b().select(invalid, dn, r);
        let dq = self.f64_nan_result(&cd);
        r = self.b().select(cd.nan, dq, r);
        let nq = self.f64_nan_result(&cn);
        r = self.b().select(cn.nan, nq, r);
        self.write_f64_result(ins, m, r)
    }

    /// v_trig_preop_f64: 53 bits of 2/pi starting `src1 * 53` bits (plus an
    /// exponent-dependent offset for large inputs) below the binary point,
    /// scaled accordingly.
    fn trig_preop_f64(&mut self, ins: &Instruction, m: &Mods) -> Result<()> {
        const TWO_OVER_PI: [u32; 40] = [
            0xa2f9836e, 0x4e441529, 0xfc2757d1, 0xf534ddc0, 0xdb629599, 0x3c439041, 0xfe5163ab, 0xdebbc561, 0xb7246e3a, 0x424dd2e0, 0x06492eea,
            0x09d1921c, 0xfe1deb1c, 0xb129a73e, 0xe88235f5, 0x2ebb4484, 0xe99c7026, 0xb45f7e41, 0x3991d639, 0x835339f4, 0x9c845f8b, 0xbdf9283b,
            0x1ff897ff, 0xde05980f, 0xef2f118b, 0x5a0a6d1f, 0x6d367ecf, 0x27cb09b7, 0x4f463f66, 0x9e5fea2d, 0x7527bac7, 0xebe5f17b, 0x3d0739f7,
            0x8a5292ea, 0x6bfb5fb1, 0x1f8d5d08, 0x56033046, 0x00000000, 0x00000000, 0x00000000,
        ];
        let v = self.src_f64(ins, m, 0);
        let sel = self.src_u32(ins, m, 1);
        let cv = self.f64_class(v);
        let c1077 = self.u(1077);
        let e = self.b().umax(cv.exp, c1077);
        let extra = self.b().isub(e, c1077);
        let c31 = self.u(31);
        let seg = self.b().and(sel, c31);
        let c53 = self.u(53);
        let seg = self.b().imul(seg, c53);
        let sh = self.b().iadd(seg, extra);
        let c1201 = self.u(1201);
        let shift = self.b().umin(sh, c1201);
        let c5 = self.u(5);
        let word = self.b().lshr(shift, c5);
        let offset = self.b().and(shift, c31);
        let zero = self.u(0);
        let aligned = self.b().icmp(ICmp::Eq, offset, zero);
        let c32 = self.u(32);
        let rest = self.b().isub(c32, offset);
        let rest = self.b().and(rest, c31);
        let load = |t: &mut Self, k: u32| {
            let kk = t.u(k);
            let idx = t.b().iadd(word, kk);
            let mut acc = t.u(0);
            for (i, w) in TWO_OVER_PI.iter().enumerate() {
                let iv = t.u(i as u32);
                let hit = t.b().icmp(ICmp::Eq, idx, iv);
                let wv = t.u(*w);
                acc = t.b().select(hit, wv, acc);
            }
            acc
        };
        let funnel = |t: &mut Self, upper: Value, lower: Value| {
            let a = t.b().shl(upper, offset);
            let b = t.b().lshr(lower, rest);
            let merged = t.b().or(a, b);
            t.b().select(aligned, upper, merged)
        };
        let first = load(self, 0);
        let second = load(self, 1);
        let third = load(self, 2);
        let hw = funnel(self, first, second);
        let high = self.b().utof(hw, Type::F64);
        let lw = funnel(self, second, third);
        let c11 = self.u(11);
        let lw = self.b().lshr(lw, c11);
        let low = self.b().utof(lw, Type::F64);
        let p21 = self.b().f64(2097152.0);
        let hs = self.b().fmul(high, p21);
        let mantissa = self.b().fadd(hs, low);
        let c1968 = self.u(1968);
        let large = self.b().icmp(ICmp::UGe, cv.exp, c1968);
        let bl = self.u(1023 + 600 - 53 + 128);
        let bs = self.u(1023 + 600 - 53);
        let bias = self.b().select(large, bl, bs);
        let se = self.b().isub(bias, shift);
        let c20 = self.u(20);
        let sh_hi = self.b().shl(se, c20);
        let scale = self.f64_from_parts(zero, sh_hi);
        let r = self.b().fmul(mantissa, scale);
        let k = self.f64_const(0x1A70_0000_0000_0000);
        let r = self.b().fmul(r, k);
        self.write_f64_result(ins, m, r)
    }

    // ---- f32 = f16 dot product ----

    /// Error-free f64 sum: (rounded sum, exact remainder).
    fn two_sum(&mut self, l: Value, r: Value) -> (Value, Value) {
        let s = self.b().fadd(l, r);
        let rp = self.b().fsub(s, l);
        let lp = self.b().fsub(s, rp);
        let el = self.b().fsub(l, lp);
        let er = self.b().fsub(r, rp);
        let e = self.b().fadd(el, er);
        (s, e)
    }

    /// f64 sum rounded to odd (sticky low bit), so that a later narrowing
    /// rounds the exact sum only once.
    pub(super) fn add_round_to_odd(&mut self, l: Value, r_in: Value) -> Value {
        let (s, e) = self.two_sum(l, r_in);
        let sb = self.b().bitcast(s, Type::U64);
        let eb = self.b().bitcast(e, Type::U64);
        let zero = self.b().f64(0.0);
        let inexact = self.b().fcmp(FCmp::UNe, e, zero);
        let x = self.b().xor(sb, eb);
        let z64 = self.b().u64(0);
        let away = self.b().icmp(ICmp::SLt, x, z64);
        let one = self.b().u64(1);
        let dec = self.b().isub(sb, one);
        let trunc = self.b().select(away, dec, sb);
        let odd = self.b().or(trunc, one);
        let r = self.b().select(inexact, odd, sb);
        let r = self.b().bitcast(r, Type::F64);
        // A zero `r` keeps `l` as is (including the sign of a zero `l`).
        let r_zero = self.b().fcmp(FCmp::OEq, r_in, zero);
        self.b().select(r_zero, l, r)
    }

    /// `add_round_to_odd` of two operands rather than a value and its error
    /// term: two zeros add as IEEE zeros do (-0 + +0 = +0).
    pub(super) fn sum_round_to_odd(&mut self, l: Value, r: Value) -> Value {
        let odd = self.add_round_to_odd(l, r);
        let zero = self.b().f64(0.0);
        let l_zero = self.b().fcmp(FCmp::OEq, l, zero);
        let r_zero = self.b().fcmp(FCmp::OEq, r, zero);
        let zeros = self.b().land(l_zero, r_zero);
        let zero_sum = self.b().fadd(l, r);
        self.b().select(zeros, zero_sum, odd)
    }

    /// v_dot2_f32_f16 / v_dot2c_f32_f16: `a.lo * b.lo + a.hi * b.hi + c`
    /// rounded once to f32. A denormal accumulator reads as zero and a
    /// denormal result is flushed; NaN operands win in the order a.lo, b.lo,
    /// a.hi, b.hi, c (quieted); clamp and omod do not apply.
    pub(super) fn dot2_f32_f16(&mut self, ins: &Instruction, m: &Mods, accumulate: bool) -> Result<()> {
        struct Half {
            bits: Value,
            sign: Value,
            nan: Value,
            inf: Value,
            zero: Value,
            value: Value,
        }
        let mut halves = Vec::new();
        for i in 0..2 {
            for high in [false, true] {
                let sel = if accumulate { high } else if high { m.op_sel_hi(i) } else { m.op_sel(i) };
                let mut bits = self.src_half_bits(ins, m, i, sel);
                if if high { m.neg_hi(i) } else { m.neg(i) } {
                    let k = self.u(0x8000);
                    bits = self.b().xor(bits, k);
                }
                let k = self.u(0x7FFF);
                let mag = self.b().and(bits, k);
                let k = self.u(0x8000);
                let sign = self.b().and(bits, k);
                let sixteen = self.u(16);
                let sign = self.b().shl(sign, sixteen);
                let inf = self.u(0x7C00);
                let nan = self.b().icmp(ICmp::UGt, mag, inf);
                let is_inf = self.b().icmp(ICmp::Eq, mag, inf);
                let zero = self.u(0);
                let is_zero = self.b().icmp(ICmp::Eq, mag, zero);
                let f = self.half_to_f32(bits);
                let value = self.b().fconvert(f, Type::F64, Rounding::Rte);
                halves.push(Half { bits, sign, nan, inf: is_inf, zero: is_zero, value });
            }
        }
        // halves: a.lo, a.hi, b.lo, b.hi
        let (al, ah, bl, bh) = (&halves[0], &halves[1], &halves[2], &halves[3]);
        let d = self.vdst(ins)?;
        // The addend is always the full 32-bit source: op_sel / op_sel_hi on
        // src2 are ignored, neg_hi on src2 is an absolute value and neg_lo
        // negates after it.
        let mut c = if accumulate { self.vgpr(d) } else { self.src_bits(ins, m, 2) };
        if !accumulate && m.neg_hi(2) {
            let k = self.u(0x7FFF_FFFF);
            c = self.b().and(c, k);
        }
        if !accumulate && m.neg(2) {
            let k = self.u(0x8000_0000);
            c = self.b().xor(c, k);
        }
        let cm = self.mag(c);
        let k = self.u(0x8000_0000);
        let sc = self.b().and(c, k);
        let inf32 = self.u(0x7F80_0000);
        let c_nan = self.b().icmp(ICmp::UGt, cm, inf32);
        let c_inf = self.b().icmp(ICmp::Eq, cm, inf32);
        let lim = self.u(0x0080_0000);
        let c_denorm = self.b().icmp(ICmp::ULt, cm, lim);
        let cf = self.b().select(c_denorm, sc, c);
        let cf = self.to_f(cf);
        let cv = self.b().fconvert(cf, Type::F64, Rounding::Rte);

        let pl = self.b().fmul(al.value, bl.value);
        let ph = self.b().fmul(ah.value, bh.value);
        let (uh, ul) = self.two_sum(ph, cv);
        let (th, tl) = self.two_sum(pl, uh);
        let inner = self.add_round_to_odd(tl, ul);
        let sum = self.add_round_to_odd(th, inner);
        let r32 = self.b().fconvert(sum, Type::F32, Rounding::Rte);
        let rb = self.f_bits(r32);
        let rm = self.mag(rb);
        let rs = self.b().and(rb, k);
        let r_denorm = self.b().icmp(ICmp::ULt, rm, lim);
        let mut r = self.b().select(r_denorm, rs, rb);

        let sl = self.b().xor(al.sign, bl.sign);
        let sh = self.b().xor(ah.sign, bh.sign);
        let zs = self.b().and(sl, sh);
        let zs = self.b().and(zs, sc);
        let z64 = self.b().f64(0.0);
        let sum_zero = self.b().fcmp(FCmp::OEq, sum, z64);
        r = self.b().select(sum_zero, zs, r);

        let inf_product = |t: &mut Self, x: &Half, y: &Half| {
            let nx = t.b().lnot(x.zero);
            let ny = t.b().lnot(y.zero);
            let a = t.b().land(x.inf, ny);
            let b = t.b().land(y.inf, nx);
            t.b().lor(a, b)
        };
        let invalid_product = |t: &mut Self, x: &Half, y: &Half| {
            let a = t.b().land(x.inf, y.zero);
            let b = t.b().land(y.inf, x.zero);
            t.b().lor(a, b)
        };
        let il = inf_product(self, al, bl);
        let ih = inf_product(self, ah, bh);
        let vl = invalid_product(self, al, bl);
        let vh = invalid_product(self, ah, bh);
        let mut invalid = self.b().lor(vl, vh);
        let conflict = |t: &mut Self, x: Value, y: Value, sx: Value, sy: Value| {
            let both = t.b().land(x, y);
            let differ = t.b().icmp(ICmp::Ne, sx, sy);
            t.b().land(both, differ)
        };
        let k1 = conflict(self, il, ih, sl, sh);
        let k2 = conflict(self, il, c_inf, sl, sc);
        let k3 = conflict(self, ih, c_inf, sh, sc);
        invalid = self.b().lor(invalid, k1);
        invalid = self.b().lor(invalid, k2);
        invalid = self.b().lor(invalid, k3);
        let any_inf = self.b().lor(il, ih);
        let any_inf = self.b().lor(any_inf, c_inf);
        let s1 = self.b().select(ih, sh, sc);
        let inf_sign = self.b().select(il, sl, s1);
        let signed_inf = self.b().or(inf_sign, inf32);
        r = self.b().select(any_inf, signed_inf, r);
        let dn = self.u(0xFFC0_0000);
        r = self.b().select(invalid, dn, r);

        let qb = self.u(0x0040_0000);
        let cq = self.b().or(c, qb);
        r = self.b().select(c_nan, cq, r);
        for h in [bh, ah, bl, al] {
            let qn = self.u(0x7FC0_0000);
            let q = self.b().or(h.sign, qn);
            let pm = self.u(0x3FF);
            let p = self.b().and(h.bits, pm);
            let thirteen = self.u(13);
            let p = self.b().shl(p, thirteen);
            let q = self.b().or(q, p);
            r = self.b().select(h.nan, q, r);
        }
        self.write_vgpr_mods(d, r, &DstMods::of(m));
        Ok(())
    }

    // ---- scalar extras ----

    pub(crate) fn translate_extra_salu(&mut self, ins: &Instruction) -> Result<bool> {
        use Opcode::*;
        let dst = ins.destinations.first().map(|d| d.value).unwrap_or(NULL_SOURCE);
        let set_scc_nz = |t: &mut Self, v: Value| {
            let z = t.b().u32(0);
            let c = t.b().icmp(ICmp::Ne, v, z);
            t.b().set_reg(Reg::Scc, c);
        };
        match ins.opcode {
            SCmovB32 => {
                let v = self.read_raw(ins.sources[0]);
                let old = self.sgpr(dst);
                let scc = self.b().get_reg(Reg::Scc);
                let r = self.b().select(scc, v, old);
                self.write_sgpr(dst, r);
            }
            SCmovkI32 => {
                let v = self.u((ins.sources[0].value & 0xFFFF) as i16 as i32 as u32);
                let old = self.sgpr(dst);
                let scc = self.b().get_reg(Reg::Scc);
                let r = self.b().select(scc, v, old);
                self.write_sgpr(dst, r);
            }
            SBrevB64 => {
                let (lo, hi) = self.read_u64_parts(ins.sources[0]);
                let nl = self.b().bit_reverse(hi);
                let nh = self.b().bit_reverse(lo);
                self.write_sgpr_pair(dst, nl, nh);
            }
            SBcnt0I32B32 | SBcnt0I32B64 => {
                let r = if ins.opcode == SBcnt0I32B64 {
                    let (lo, hi) = self.read_u64_parts(ins.sources[0]);
                    let nl = self.b().not(lo);
                    let nh = self.b().not(hi);
                    let a = self.b().bit_count(nl);
                    let b = self.b().bit_count(nh);
                    self.b().iadd(a, b)
                } else {
                    let v = self.read_raw(ins.sources[0]);
                    let n = self.b().not(v);
                    self.b().bit_count(n)
                };
                set_scc_nz(self, r);
                self.write_sgpr(dst, r);
            }
            SFF0I32B32 => {
                let v = self.read_raw(ins.sources[0]);
                let n = self.b().not(v);
                let r = self.b().find_lsb(n);
                self.write_sgpr(dst, r);
            }
            SFF0I32B64 => {
                let (lo, hi) = self.read_u64_parts(ins.sources[0]);
                let nl = self.b().not(lo);
                let nh = self.b().not(hi);
                let r = self.find_lsb64(nl, nh);
                self.write_sgpr(dst, r);
            }
            SFlbitI32 => {
                let v = self.read_raw(ins.sources[0]);
                let r = self.leading_zeros32(v, true);
                self.write_sgpr(dst, r);
            }
            SFlbitI32I64 => {
                // A literal source is sign-extended for this instruction.
                let (lo, hi) = self.read_i64_parts(ins.sources[0]);
                let c31 = self.u(31);
                let s = self.b().ashr(hi, c31);
                let lo = self.b().xor(lo, s);
                let hi = self.b().xor(hi, s);
                let hz = self.leading_zeros32(hi, false);
                let lz = self.leading_zeros32(lo, false);
                let c32 = self.u(32);
                let lz32 = self.b().iadd(lz, c32);
                let zero = self.u(0);
                let hnz = self.b().icmp(ICmp::Ne, hi, zero);
                let lnz = self.b().icmp(ICmp::Ne, lo, zero);
                let neg1 = self.u(u32::MAX);
                let lp = self.b().select(lnz, lz32, neg1);
                let r = self.b().select(hnz, hz, lp);
                self.write_sgpr(dst, r);
            }
            SSextI32I8 | SSextI32I16 => {
                let v = self.read_raw(ins.sources[0]);
                let z = self.u(0);
                let w = self.u(if ins.opcode == SSextI32I8 { 8 } else { 16 });
                let r = self.b().bfe_s(v, z, w);
                self.write_sgpr(dst, r);
            }
            SMovrelsB32 | SMovrelsB64 | SMovreldB32 | SMovreldB64 | SMovrelsd2B32 => self.s_movrel(ins)?,
            SAndn1WrexecB64 | SAndn2WrexecB64 | SAndn1WrexecB32 | SAndn2WrexecB32 => {
                let wide = matches!(ins.opcode, SAndn1WrexecB64 | SAndn2WrexecB64);
                let not_src = matches!(ins.opcode, SAndn1WrexecB64 | SAndn1WrexecB32);
                let (src_lo, src_hi) = if wide { self.read_u64_parts(ins.sources[0]) } else { (self.read_raw(ins.sources[0]), self.u(0)) };
                let old_lo = self.sgpr(EXEC_LO);
                let old_hi = self.sgpr(EXEC_HI);
                let comb = |t: &mut Self, e: Value, s: Value| {
                    if not_src {
                        let ns = t.b().not(s);
                        t.b().and(e, ns)
                    } else {
                        let ne = t.b().not(e);
                        t.b().and(ne, s)
                    }
                };
                let lo = comb(self, old_lo, src_lo);
                let hi = if wide { comb(self, old_hi, src_hi) } else { old_hi };
                self.b().set_reg(Reg::Sgpr(Reg::EXEC), lo);
                if wide {
                    self.set_exec_hi(hi);
                }
                self.refresh_exec_bit();
                if wide {
                    self.write_sgpr_pair(dst, lo, hi);
                    let o = self.b().or(lo, hi);
                    set_scc_nz(self, o);
                } else {
                    self.write_sgpr(dst, lo);
                    set_scc_nz(self, lo);
                }
            }
            SGetregB32 => {
                let field = ins.sources[0].value & 0xFFFF;
                let id = field & 0x3F;
                let offset = (field >> 6) & 0x1F;
                let size = ((field >> 11) & 0x1F) + 1;
                if id != 1 || offset + size > 4 {
                    return self.unsupported(ins, "hardware register read other than MODE round mode");
                }
                // The round fields as set at this point.
                let mask = (((1u32 << size) - 1) << offset) as u8;
                let bits = self.read_rounding(ins, mask)? as u32;
                let v = self.u(bits >> offset);
                self.write_sgpr(dst, v);
            }
            // Round-field writes are tracked through the control flow
            // (`rounding`); other fields and registers are refused when
            // strict, ignored otherwise.
            SSetregImm32B32 => self.write_rounding(ins)?,
            SVersion | SSleep | SWakeup | SIcacheInv | SIncperflevel | SDecperflevel | STtracedataImm => {}
            SRoundMode => self.write_rounding(ins)?,
            SDenormMode => return self.unsupported(ins, "denormal modes are not modelled"),
            SRfeB64 => return self.unsupported(ins, "s_rfe_b64 returns from a trap handler, and recompiled shaders run without one"),
            SCallB64 => self.write_next_address(ins)?,
            SEndpgmSaved => {}
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn s_movrel(&mut self, ins: &Instruction) -> Result<()> {
        use Opcode::*;
        let wide = matches!(ins.opcode, SMovrelsB64 | SMovreldB64);
        let rel_src = !matches!(ins.opcode, SMovreldB32 | SMovreldB64);
        let rel_dst = !matches!(ins.opcode, SMovrelsB32 | SMovrelsB64);
        let src = ins.sources[0];
        let dst = ins.destinations[0].value;
        if (rel_src && src.kind != OperandKind::ScalarRegister) || dst >= 128 {
            return self.invalid(ins, "s_movrel operand is not an SGPR");
        }
        let m0 = self.sgpr(M0);
        let split = ins.opcode == SMovrelsd2B32;
        let (soff, doff) = if split {
            let mask = self.u(0x3FF);
            let s = self.b().and(m0, mask);
            let sixteen = self.u(16);
            let d = self.b().lshr(m0, sixteen);
            (s, self.b().and(d, mask))
        } else if wide {
            let mask = self.u(!1);
            let s = self.b().and(m0, mask);
            (s, s)
        } else {
            (m0, m0)
        };
        let count = if wide { 2 } else { 1 };
        let mut values = Vec::new();
        for k in 0..count {
            let v = if rel_src {
                let base = src.value + k;
                let mut acc = self.u(0);
                for reg in base..106 {
                    let off = self.u(reg - base);
                    let hit = self.b().icmp(ICmp::Eq, soff, off);
                    let v = self.sgpr(reg);
                    acc = self.b().select(hit, v, acc);
                }
                acc
            } else if wide {
                let (lo, hi) = self.read_u64_parts(src);
                if k == 0 {
                    lo
                } else {
                    hi
                }
            } else {
                self.read_raw(src)
            };
            values.push(v);
        }
        for (k, v) in values.into_iter().enumerate() {
            let base = dst + k as u32;
            if !rel_dst {
                self.write_sgpr(base, v);
                continue;
            }
            for reg in base..106 {
                let off = self.u(reg - base);
                let hit = self.b().icmp(ICmp::Eq, doff, off);
                let old = self.sgpr(reg);
                let r = self.b().select(hit, v, old);
                self.write_sgpr(reg, r);
            }
        }
        Ok(())
    }
}
