// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Formatted buffer conversions in integer arithmetic, bit-exact with the
//! interpreter (`ps5shader::interp::buffer`): normalized and scaled
//! stores (exact product rounded to nearest even), small floats rounded
//! toward zero, and the 16-bit (D16) load and store conversions.

use super::emit::Emitter;
use ps5shader::interp::buffer::{FormatInfo, FormatKind};
use rspirv::spirv::{GlslStd450Op as GLOp, Op as S, Word};

/// FindUMsb (all ones for 0).
fn msb(e: &mut Emitter, v: Word) -> Word {
    let tu = e.t_u32();
    e.gl(tu, GLOp::FindUMsb, &[v])
}

fn ugt(e: &mut Emitter, a: Word, b: Word) -> Word {
    e.ult(b, a)
}

/// f32 bits to a UNORM (`signed == false`) or SNORM field of magnitude
/// `maximum` (at most 16 bits): `|value| * maximum` rounded to nearest
/// even, saturated at `maximum`; NaN, zero and non-positive UNORM inputs
/// give 0; a negative SNORM input is negated (not masked).
pub fn norm_store(e: &mut Emitter, bits: Word, maximum: u32, signed: bool) -> Word {
    let tu = e.t_u32();
    let t64 = e.ty(ps5shader::ssa::Type::U64);
    let exponent = e.bits(bits, 23, 8);
    let zero = e.cu(0);
    let denormal = e.ieq(exponent, zero);
    let mfield = e.bits(bits, 0, 23);
    let implicit = e.cu(0x80_0000);
    let with_one = e.bor(mfield, implicit);
    let mantissa = e.select(tu, denormal, mfield, with_one);
    // value = mantissa * 2^-shift
    let c150 = e.cu(150);
    let c149 = e.cu(149);
    let normal_shift = e.isub(c150, exponent);
    let shift = e.select(tu, denormal, c149, normal_shift);
    let c24 = e.cu(24);
    let extra = e.isub(shift, c24);
    let c16 = e.cu(16);
    let too_small = ugt(e, extra, c16);
    let ex = e.umin(extra, c16);
    let m64 = e.ins(S::UConvert, t64, &[mantissa]);
    let max64 = e.c64(maximum as u64);
    let product = e.ins(S::IMul, t64, &[m64, max64]);
    let s24 = e.cu(24);
    let hi = e.ins(S::ShiftRightLogical, t64, &[product, s24]);
    let integer = e.ins(S::UConvert, tu, &[hi]);
    let low = e.ins(S::UConvert, tu, &[product]);
    let fraction = e.bits(low, 0, 24);
    let c8 = e.cu(8);
    let ih = e.shl(integer, c16);
    let fl = e.shr(fraction, c8);
    let fixed = e.bor(ih, fl);
    let ff = e.cu(0xFF);
    let st = e.band(fraction, ff);
    let sticky = e.ine(st, zero);
    let q0 = e.shr(fixed, c16);
    let quotient = e.shr(q0, ex);
    let qs = e.shl(quotient, c16);
    let qs = e.shl(qs, ex);
    let remainder = e.isub(fixed, qs);
    let h0 = e.cu(1 << 15);
    let half = e.shl(h0, ex);
    let one = e.cu(1);
    let odd = e.band(quotient, one);
    let odd = e.ine(odd, zero);
    let eq = e.ieq(remainder, half);
    let so = e.lor(sticky, odd);
    let tie = e.land(eq, so);
    let above = ugt(e, remainder, half);
    let up = e.lor(above, tie);
    let inc = e.select(tu, up, one, zero);
    let rounded = e.iadd(quotient, inc);
    let mut result = e.select(tu, too_small, zero, rounded);
    let abs_mask = e.cu(0x7FFF_FFFF);
    let magnitude = e.band(bits, abs_mask);
    let one_f = e.cu(0x3F80_0000);
    let inf = e.cu(0x7F80_0000);
    let ge_one = e.ule(one_f, magnitude);
    let max = e.cu(maximum);
    result = e.select(tu, ge_one, max, result);
    // NaN, zero and (UNORM) negative inputs store 0.
    let nonzero = e.ine(magnitude, zero);
    let not_nan = e.ule(magnitude, inf);
    let mut positive = e.land(nonzero, not_nan);
    let sign_bit = e.cu(0x8000_0000);
    let sign = e.band(bits, sign_bit);
    let negative = e.ine(sign, zero);
    if !signed {
        let nn = e.lnot(negative);
        positive = e.land(positive, nn);
    }
    result = e.select(tu, positive, result, zero);
    if signed {
        let neg = e.isub(zero, result);
        result = e.select(tu, negative, neg, result);
    }
    result
}

/// f32 bits to a small float with a 5-bit exponent and `mantissa_bits`
/// mantissa bits, rounding toward zero: finite overflow gives the largest
/// finite value, infinities stay infinite, NaN keeps the top payload bits
/// (or becomes the pattern with only bit 0 when they are all zero).
/// Unsigned formats store negative non-NaN values as 0.
pub fn small_float_rtz(e: &mut Emitter, bits: Word, mantissa_bits: u32, signed: bool) -> Word {
    let tu = e.t_u32();
    let ti = e.t_i32();
    let zero = e.cu(0);
    let infinity = 0x1Fu32 << mantissa_bits;
    let sign_bit = e.bits(bits, 31, 1);
    let negative = e.ine(sign_bit, zero);
    let exponent = e.bits(bits, 23, 8);
    let mantissa = e.bits(bits, 0, 23);
    let tsh = e.cu(23 - mantissa_bits);
    let top = e.shr(mantissa, tsh);
    let c112 = e.cu(112);
    let biased_u = e.isub(exponent, c112);
    let biased = e.bitcast(ti, biased_u);
    let sg = |e: &mut Emitter, k: i32| {
        let tb = e.t_bool();
        let c = e.ci(k);
        e.ins(S::SGreaterThanEqual, tb, &[biased, c])
    };
    let overflow = sg(e, 31);
    let normal = sg(e, 1);
    let subnormal = sg(e, -(mantissa_bits as i32));
    let mb = e.cu(mantissa_bits);
    let bs = e.shl(biased_u, mb);
    let normal_v = e.bor(bs, top);
    let implicit = e.cu(0x80_0000);
    let full = e.bor(mantissa, implicit);
    let base = e.cu(24 - mantissa_bits);
    let sh = e.isub(base, biased_u);
    let c31 = e.cu(31);
    let sh = e.umin(sh, c31);
    let sub_v = e.shr(full, sh);
    let big = e.cu(infinity - 1);
    let mut magnitude = e.select(tu, subnormal, sub_v, zero);
    magnitude = e.select(tu, normal, normal_v, magnitude);
    magnitude = e.select(tu, overflow, big, magnitude);
    let ff = e.cu(0xFF);
    let special = e.ieq(exponent, ff);
    let top0 = e.ieq(top, zero);
    let mnz = e.ine(mantissa, zero);
    let nan_low = e.land(top0, mnz);
    let one = e.cu(1);
    let payload = e.select(tu, nan_low, one, top);
    let inf = e.cu(infinity);
    let sp = e.bor(inf, payload);
    magnitude = e.select(tu, special, sp, magnitude);
    if signed {
        let s = e.cu(1 << (5 + mantissa_bits));
        let sign = e.select(tu, negative, s, zero);
        e.bor(sign, magnitude)
    } else {
        let nan = e.land(special, mnz);
        let not_nan = e.lnot(nan);
        let clear = e.land(negative, not_nan);
        e.select(tu, clear, zero, magnitude)
    }
}

/// `value / maximum` (maximum at most 16 bits) as f16 bits, rounded to
/// nearest.
fn norm_to_f16(e: &mut Emitter, value: Word, maximum: u32) -> Word {
    let tu = e.t_u32();
    let zero = e.cu(0);
    let is_zero = e.ieq(value, zero);
    let width = 32 - maximum.leading_zeros();
    let m = msb(e, value);
    let w9 = e.cu(9 + width);
    let first = e.isub(w9, m);
    let c31 = e.cu(31);
    let first = e.umin(first, c31);
    let shifted = e.shl(value, first);
    let lim = e.cu(1024 * maximum);
    let below = e.ult(shifted, lim);
    let one = e.cu(1);
    let first1 = e.iadd(first, one);
    let adjusted = e.select(tu, below, first1, first);
    let c24 = e.cu(24);
    let shift = e.umin(adjusted, c24);
    let scaled = e.shl(value, shift);
    let max = e.cu(maximum);
    let quotient = e.ins(S::UDiv, tu, &[scaled, max]);
    let rem = e.ins(S::UMod, tu, &[scaled, max]);
    let halfm = e.cu(maximum >> 1);
    let up = ugt(e, rem, halfm);
    let upv = e.select(tu, up, one, zero);
    let c25 = e.cu(25);
    let es = e.isub(c25, shift);
    let c10 = e.cu(10);
    let exponent = e.shl(es, c10);
    let r = e.iadd(exponent, quotient);
    let r = e.iadd(r, upv);
    let k = e.cu(1024);
    let r = e.isub(r, k);
    e.select(tu, is_zero, zero, r)
}

/// Unsigned integer as f16 bits, rounded to nearest even, clamped to the
/// largest finite value.
fn integer_to_f16(e: &mut Emitter, value: Word) -> Word {
    let tu = e.t_u32();
    let zero = e.cu(0);
    let one = e.cu(1);
    let is_zero = e.ieq(value, zero);
    let m = msb(e, value);
    let c10 = e.cu(10);
    let wide = ugt(e, m, c10);
    let l = e.isub(c10, m);
    let r = e.isub(m, c10);
    let left = e.select(tu, wide, zero, l);
    let right = e.select(tu, wide, r, zero);
    let c31 = e.cu(31);
    let left = e.umin(left, c31);
    let right = e.umin(right, c31);
    let sl = e.shl(value, left);
    let quotient = e.shr(sl, right);
    let unit = e.shl(one, right);
    let um = e.isub(unit, one);
    let remainder = e.band(value, um);
    let half = e.shr(unit, one);
    let eq = e.ieq(remainder, half);
    let q1 = e.band(quotient, one);
    let odd = e.ine(q1, zero);
    let tie = e.land(eq, odd);
    let above = ugt(e, remainder, half);
    let up = e.lor(above, tie);
    let up = e.land(wide, up);
    let upv = e.select(tu, up, one, zero);
    let c15 = e.cu(15);
    let me = e.iadd(m, c15);
    let exponent = e.shl(me, c10);
    let s = e.iadd(exponent, quotient);
    let s = e.iadd(s, upv);
    let k = e.cu(1024);
    let s = e.isub(s, k);
    let cap = e.cu(0x7BFF);
    let s = e.umin(s, cap);
    e.select(tu, is_zero, zero, s)
}

/// Exact f16 to f32 bit conversion (NaN payloads kept).
fn f16_bits_to_f32(e: &mut Emitter, half: Word) -> Word {
    let tu = e.t_u32();
    let zero = e.cu(0);
    let sm = e.cu(0x8000);
    let s = e.band(half, sm);
    let c16 = e.cu(16);
    let sign = e.shl(s, c16);
    let exponent = e.bits(half, 10, 5);
    let mantissa = e.bits(half, 0, 10);
    let c13 = e.cu(13);
    let wide = e.shl(mantissa, c13);
    // Denormal: normalise the mantissa.
    let m = msb(e, mantissa);
    let c103 = e.cu(103);
    let de = e.iadd(m, c103);
    let c23 = e.cu(23);
    let des = e.shl(de, c23);
    let dsh = e.isub(c23, m);
    let c31 = e.cu(31);
    let dsh = e.umin(dsh, c31);
    let dm = e.shl(mantissa, dsh);
    let mmask = e.cu(0x7F_FFFF);
    let dm = e.band(dm, mmask);
    let denormal = e.bor(des, dm);
    let inf = e.cu(0x7F80_0000);
    let special = e.bor(inf, wide);
    let c112 = e.cu(112);
    let ne = e.iadd(exponent, c112);
    let nes = e.shl(ne, c23);
    let normal = e.bor(nes, wide);
    let e0 = e.ieq(exponent, zero);
    let m0 = e.ieq(mantissa, zero);
    let e31 = e.cu(31);
    let is31 = e.ieq(exponent, e31);
    let mut mag = e.select(tu, is31, special, normal);
    let d = e.select(tu, m0, zero, denormal);
    mag = e.select(tu, e0, d, mag);
    e.bor(sign, mag)
}

/// Store conversion of a 32-bit register value to component `c`'s field
/// (unshifted, masked to the field width).
pub fn store_component(e: &mut Emitter, info: &FormatInfo, c: u32, data: Word) -> Word {
    let tu = e.t_u32();
    let bits = info.bits[c as usize];
    let mask = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
    match info.kind {
        FormatKind::Uint => {
            let m = e.cu(mask);
            e.umin(data, m)
        }
        FormatKind::Sint if bits >= 32 => data,
        FormatKind::Sint => {
            let ti = e.t_i32();
            let di = e.bitcast(ti, data);
            let high = (mask >> 1) as i32;
            let (lo, hi) = (e.ci(!high), e.ci(high));
            let cl = e.gl(ti, GLOp::SClamp, &[di, lo, hi]);
            let cu = e.bitcast(tu, cl);
            let m = e.cu(mask);
            e.band(cu, m)
        }
        FormatKind::Unorm | FormatKind::Uscaled => norm_store(e, data, mask, false),
        FormatKind::Snorm | FormatKind::Sscaled => {
            let v = norm_store(e, data, mask >> 1, true);
            let m = e.cu(mask);
            e.band(v, m)
        }
        FormatKind::Float => match bits {
            32 => data,
            16 => small_float_rtz(e, data, 10, true),
            _ => small_float_rtz(e, data, bits - 5, false),
        },
    }
}

/// D16 store conversion of a 16-bit register half to component `c`'s field.
pub fn store_component_d16(e: &mut Emitter, info: &FormatInfo, c: u32, half: Word) -> Word {
    let tu = e.t_u32();
    let ffff = e.cu(0xFFFF);
    let half = e.band(half, ffff);
    match info.kind {
        FormatKind::Uint => store_component(e, info, c, half),
        FormatKind::Sint => {
            let (o, n) = (e.cu(0), e.cu(16));
            let s = e.ins(S::BitFieldSExtract, tu, &[half, o, n]);
            store_component(e, info, c, s)
        }
        FormatKind::Float if info.bits[c as usize] == 16 => half,
        _ => {
            let f = f16_bits_to_f32(e, half);
            store_component(e, info, c, f)
        }
    }
}

/// D16 load conversion of component `c`'s extracted field (sign-extended
/// for signed formats) given `full`, its 32-bit load conversion.
pub fn load_component_d16(e: &mut Emitter, info: &FormatInfo, c: u32, raw: Word, full: Word) -> Word {
    let tu = e.t_u32();
    let ti = e.t_i32();
    let bits = info.bits[c as usize];
    let ffff = e.cu(0xFFFF);
    let zero = e.cu(0);
    let negative = |e: &mut Emitter| {
        let r = e.bitcast(ti, raw);
        let tb = e.t_bool();
        let z = e.ci(0);
        e.ins(S::SLessThan, tb, &[r, z])
    };
    match info.kind {
        FormatKind::Uint if bits > 16 => e.umin(raw, ffff),
        FormatKind::Sint if bits > 16 => {
            let r = e.bitcast(ti, raw);
            let (lo, hi) = (e.ci(-0x8000), e.ci(0x7FFF));
            let cl = e.gl(ti, GLOp::SClamp, &[r, lo, hi]);
            let cu = e.bitcast(tu, cl);
            e.band(cu, ffff)
        }
        FormatKind::Uint | FormatKind::Sint => e.band(raw, ffff),
        FormatKind::Unorm => norm_to_f16(e, raw, if bits >= 32 { u32::MAX } else { (1 << bits) - 1 }),
        FormatKind::Snorm => {
            let max = (1u32 << (bits - 1)) - 1;
            let neg = negative(e);
            let n = e.isub(zero, raw);
            let mag = e.select(tu, neg, n, raw);
            let m = e.cu(max);
            let mag = e.umin(mag, m);
            let h = norm_to_f16(e, mag, max);
            let s = e.cu(0x8000);
            let sign = e.select(tu, neg, s, zero);
            e.bor(sign, h)
        }
        FormatKind::Uscaled => integer_to_f16(e, raw),
        FormatKind::Sscaled => {
            let neg = negative(e);
            let n = e.isub(zero, raw);
            let mag = e.select(tu, neg, n, raw);
            let h = integer_to_f16(e, mag);
            let s = e.cu(0x8000);
            let sign = e.select(tu, neg, s, zero);
            e.bor(sign, h)
        }
        FormatKind::Float if bits == 16 => e.band(raw, ffff),
        FormatKind::Float => small_float_rtz(e, full, 10, true),
    }
}
