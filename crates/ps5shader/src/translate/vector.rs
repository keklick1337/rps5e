// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Vector ALU: dispatch, integer operations, compares, lane operations.

use super::operand::{DstMods, Mods};
use super::{Result, Translator};
use crate::ir_types::*;
use crate::opcode::Opcode;
use crate::ssa::*;

/// Float compare kinds of the VOPC table.
#[derive(Clone, Copy)]
enum FKind {
    False,
    True,
    P(FCmp),
}

fn float_compare(op: Opcode) -> Option<(FKind, bool, bool)> {
    use Opcode::*;
    use FCmp::*;
    // (kind, half, cmpx)
    let (k, half, x) = match op {
        VCmpFF32 => (FKind::False, false, false),
        VCmpLtF32 => (FKind::P(OLt), false, false),
        VCmpEqF32 => (FKind::P(OEq), false, false),
        VCmpLeF32 => (FKind::P(OLe), false, false),
        VCmpGtF32 => (FKind::P(OGt), false, false),
        VCmpLgF32 => (FKind::P(ONe), false, false),
        VCmpGeF32 => (FKind::P(OGe), false, false),
        VCmpOF32 => (FKind::P(Ord), false, false),
        VCmpUF32 => (FKind::P(Uno), false, false),
        VCmpNgeF32 => (FKind::P(ULt), false, false),
        VCmpNlgF32 => (FKind::P(UEq), false, false),
        VCmpNgtF32 => (FKind::P(ULe), false, false),
        VCmpNleF32 => (FKind::P(UGt), false, false),
        VCmpNeqF32 => (FKind::P(UNe), false, false),
        VCmpNltF32 => (FKind::P(UGe), false, false),
        VCmpTruF32 => (FKind::True, false, false),
        VCmpxFF32 => (FKind::False, false, true),
        VCmpxLtF32 => (FKind::P(OLt), false, true),
        VCmpxEqF32 => (FKind::P(OEq), false, true),
        VCmpxLeF32 => (FKind::P(OLe), false, true),
        VCmpxGtF32 => (FKind::P(OGt), false, true),
        VCmpxLgF32 => (FKind::P(ONe), false, true),
        VCmpxGeF32 => (FKind::P(OGe), false, true),
        VCmpxOF32 => (FKind::P(Ord), false, true),
        VCmpxUF32 => (FKind::P(Uno), false, true),
        VCmpxNgeF32 => (FKind::P(ULt), false, true),
        VCmpxNlgF32 => (FKind::P(UEq), false, true),
        VCmpxNgtF32 => (FKind::P(ULe), false, true),
        VCmpxNleF32 => (FKind::P(UGt), false, true),
        VCmpxNeqF32 => (FKind::P(UNe), false, true),
        VCmpxNltF32 => (FKind::P(UGe), false, true),
        VCmpxTruF32 => (FKind::True, false, true),
        VCmpFF16 => (FKind::False, true, false),
        VCmpLtF16 => (FKind::P(OLt), true, false),
        VCmpEqF16 => (FKind::P(OEq), true, false),
        VCmpLeF16 => (FKind::P(OLe), true, false),
        VCmpGtF16 => (FKind::P(OGt), true, false),
        VCmpLgF16 => (FKind::P(ONe), true, false),
        VCmpGeF16 => (FKind::P(OGe), true, false),
        VCmpOF16 => (FKind::P(Ord), true, false),
        VCmpUF16 => (FKind::P(Uno), true, false),
        VCmpNgeF16 => (FKind::P(ULt), true, false),
        VCmpNlgF16 => (FKind::P(UEq), true, false),
        VCmpNgtF16 => (FKind::P(ULe), true, false),
        VCmpNleF16 => (FKind::P(UGt), true, false),
        VCmpNeqF16 => (FKind::P(UNe), true, false),
        VCmpNltF16 => (FKind::P(UGe), true, false),
        VCmpTruF16 => (FKind::True, true, false),
        VCmpxFF16 => (FKind::False, true, true),
        VCmpxLtF16 => (FKind::P(OLt), true, true),
        VCmpxEqF16 => (FKind::P(OEq), true, true),
        VCmpxLeF16 => (FKind::P(OLe), true, true),
        VCmpxGtF16 => (FKind::P(OGt), true, true),
        VCmpxLgF16 => (FKind::P(ONe), true, true),
        VCmpxGeF16 => (FKind::P(OGe), true, true),
        VCmpxOF16 => (FKind::P(Ord), true, true),
        VCmpxUF16 => (FKind::P(Uno), true, true),
        VCmpxNgeF16 => (FKind::P(ULt), true, true),
        VCmpxNlgF16 => (FKind::P(UEq), true, true),
        VCmpxNgtF16 => (FKind::P(ULe), true, true),
        VCmpxNleF16 => (FKind::P(UGt), true, true),
        VCmpxNeqF16 => (FKind::P(UNe), true, true),
        VCmpxNltF16 => (FKind::P(UGe), true, true),
        VCmpxTruF16 => (FKind::True, true, true),
        _ => return None,
    };
    Some((k, half, x))
}

fn f64_compare(op: Opcode) -> Option<(FKind, bool)> {
    use FCmp::*;
    use Opcode::*;
    Some(match op {
        VCmpFF64 => (FKind::False, false),
        VCmpLtF64 => (FKind::P(OLt), false),
        VCmpEqF64 => (FKind::P(OEq), false),
        VCmpLeF64 => (FKind::P(OLe), false),
        VCmpGtF64 => (FKind::P(OGt), false),
        VCmpLgF64 => (FKind::P(ONe), false),
        VCmpGeF64 => (FKind::P(OGe), false),
        VCmpOF64 => (FKind::P(Ord), false),
        VCmpUF64 => (FKind::P(Uno), false),
        VCmpNgeF64 => (FKind::P(ULt), false),
        VCmpNlgF64 => (FKind::P(UEq), false),
        VCmpNgtF64 => (FKind::P(ULe), false),
        VCmpNleF64 => (FKind::P(UGt), false),
        VCmpNeqF64 => (FKind::P(UNe), false),
        VCmpNltF64 => (FKind::P(UGe), false),
        VCmpTruF64 => (FKind::True, false),
        VCmpxFF64 => (FKind::False, true),
        VCmpxLtF64 => (FKind::P(OLt), true),
        VCmpxEqF64 => (FKind::P(OEq), true),
        VCmpxLeF64 => (FKind::P(OLe), true),
        VCmpxGtF64 => (FKind::P(OGt), true),
        VCmpxLgF64 => (FKind::P(ONe), true),
        VCmpxGeF64 => (FKind::P(OGe), true),
        VCmpxOF64 => (FKind::P(Ord), true),
        VCmpxUF64 => (FKind::P(Uno), true),
        VCmpxNgeF64 => (FKind::P(ULt), true),
        VCmpxNlgF64 => (FKind::P(UEq), true),
        VCmpxNgtF64 => (FKind::P(ULe), true),
        VCmpxNleF64 => (FKind::P(UGt), true),
        VCmpxNeqF64 => (FKind::P(UNe), true),
        VCmpxNltF64 => (FKind::P(UGe), true),
        VCmpxTruF64 => (FKind::True, true),
        _ => return None,
    })
}

/// v_cmp*_i64 / v_cmpx*_i64: the operands are signed 64-bit (a 32-bit
/// literal is sign-extended).
fn signed_i64_compare(op: Opcode) -> bool {
    use Opcode::*;
    matches!(
        op,
        VCmpLtI64
            | VCmpEqI64
            | VCmpLeI64
            | VCmpGtI64
            | VCmpNeI64
            | VCmpGeI64
            | VCmpxLtI64
            | VCmpxEqI64
            | VCmpxLeI64
            | VCmpxGtI64
            | VCmpxNeI64
            | VCmpxGeI64
    )
}

/// Integer compare: (predicate or constant, bits 16/32/64, cmpx).
fn int_compare(op: Opcode) -> Option<(Option<ICmp>, bool, u32, bool)> {
    use ICmp::*;
    use Opcode::*;
    // (predicate, constant value when predicate is None, width, cmpx)
    let r = match op {
        VCmpFI32 | VCmpFU32 => (None, false, 32, false),
        VCmpTI32 | VCmpTU32 => (None, true, 32, false),
        VCmpxFI32 | VCmpxFU32 => (None, false, 32, true),
        VCmpxTI32 | VCmpxTU32 => (None, true, 32, true),
        VCmpFI64 | VCmpFU64 => (None, false, 64, false),
        VCmpTI64 | VCmpTU64 => (None, true, 64, false),
        VCmpxFI64 | VCmpxFU64 => (None, false, 64, true),
        VCmpxTI64 | VCmpxTU64 => (None, true, 64, true),
        VCmpLtI32 => (Some(SLt), false, 32, false),
        VCmpEqI32 | VCmpEqU32 => (Some(Eq), false, 32, false),
        VCmpLeI32 => (Some(SLe), false, 32, false),
        VCmpGtI32 => (Some(SGt), false, 32, false),
        VCmpNeI32 | VCmpNeU32 => (Some(Ne), false, 32, false),
        VCmpGeI32 => (Some(SGe), false, 32, false),
        VCmpLtU32 => (Some(ULt), false, 32, false),
        VCmpLeU32 => (Some(ULe), false, 32, false),
        VCmpGtU32 => (Some(UGt), false, 32, false),
        VCmpGeU32 => (Some(UGe), false, 32, false),
        VCmpxLtI32 => (Some(SLt), false, 32, true),
        VCmpxEqI32 | VCmpxEqU32 => (Some(Eq), false, 32, true),
        VCmpxLeI32 => (Some(SLe), false, 32, true),
        VCmpxGtI32 => (Some(SGt), false, 32, true),
        VCmpxNeI32 | VCmpxNeU32 => (Some(Ne), false, 32, true),
        VCmpxGeI32 => (Some(SGe), false, 32, true),
        VCmpxLtU32 => (Some(ULt), false, 32, true),
        VCmpxLeU32 => (Some(ULe), false, 32, true),
        VCmpxGtU32 => (Some(UGt), false, 32, true),
        VCmpxGeU32 => (Some(UGe), false, 32, true),
        VCmpLtI64 => (Some(SLt), false, 64, false),
        VCmpEqI64 | VCmpEqU64 => (Some(Eq), false, 64, false),
        VCmpLeI64 => (Some(SLe), false, 64, false),
        VCmpGtI64 => (Some(SGt), false, 64, false),
        VCmpNeI64 | VCmpNeU64 => (Some(Ne), false, 64, false),
        VCmpGeI64 => (Some(SGe), false, 64, false),
        VCmpLtU64 => (Some(ULt), false, 64, false),
        VCmpLeU64 => (Some(ULe), false, 64, false),
        VCmpGtU64 => (Some(UGt), false, 64, false),
        VCmpGeU64 => (Some(UGe), false, 64, false),
        VCmpxLtI64 => (Some(SLt), false, 64, true),
        VCmpxEqI64 | VCmpxEqU64 => (Some(Eq), false, 64, true),
        VCmpxLeI64 => (Some(SLe), false, 64, true),
        VCmpxGtI64 => (Some(SGt), false, 64, true),
        VCmpxNeI64 | VCmpxNeU64 => (Some(Ne), false, 64, true),
        VCmpxGeI64 => (Some(SGe), false, 64, true),
        VCmpxLtU64 => (Some(ULt), false, 64, true),
        VCmpxLeU64 => (Some(ULe), false, 64, true),
        VCmpxGtU64 => (Some(UGt), false, 64, true),
        VCmpxGeU64 => (Some(UGe), false, 64, true),
        VCmpLtI16 => (Some(SLt), false, 16, false),
        VCmpEqI16 => (Some(Eq), false, 16, false),
        VCmpLeI16 => (Some(SLe), false, 16, false),
        VCmpGtI16 => (Some(SGt), false, 16, false),
        VCmpNeI16 => (Some(Ne), false, 16, false),
        VCmpGeI16 => (Some(SGe), false, 16, false),
        VCmpLtU16 => (Some(ULt), false, 16, false),
        VCmpEqU16 => (Some(Eq), false, 16, false),
        VCmpLeU16 => (Some(ULe), false, 16, false),
        VCmpGtU16 => (Some(UGt), false, 16, false),
        VCmpNeU16 => (Some(Ne), false, 16, false),
        VCmpGeU16 => (Some(UGe), false, 16, false),
        VCmpxLtI16 => (Some(SLt), false, 16, true),
        VCmpxEqI16 => (Some(Eq), false, 16, true),
        VCmpxLeI16 => (Some(SLe), false, 16, true),
        VCmpxGtI16 => (Some(SGt), false, 16, true),
        VCmpxNeI16 => (Some(Ne), false, 16, true),
        VCmpxGeI16 => (Some(SGe), false, 16, true),
        VCmpxLtU16 => (Some(ULt), false, 16, true),
        VCmpxEqU16 => (Some(Eq), false, 16, true),
        VCmpxLeU16 => (Some(ULe), false, 16, true),
        VCmpxGtU16 => (Some(UGt), false, 16, true),
        VCmpxNeU16 => (Some(Ne), false, 16, true),
        VCmpxGeU16 => (Some(UGe), false, 16, true),
        _ => return None,
    };
    Some(r)
}

/// Whether a VOP1/VOP2 opcode has a DPP16 / DPP8 encoding (64-bit operands,
/// inline literal constants, lane reads and register swaps have none).
fn has_dpp_form(op: Opcode) -> bool {
    use Opcode::*;
    !matches!(
        op,
        VReadfirstlaneB32
            | VCvtI32F64
            | VCvtF64I32
            | VCvtF32F64
            | VCvtF64F32
            | VCvtU32F64
            | VCvtF64U32
            | VTruncF64
            | VCeilF64
            | VRndneF64
            | VFloorF64
            | VFrexpExpI32F64
            | VFrexpMantF64
            | VFractF64
            | VRcpF64
            | VRsqF64
            | VSqrtF64
            | VSwapB32
            | VSwaprelB32
            | VMacLegacyF32
            | VMadMkF32
            | VMadAkF32
            | VFmaMkF32
            | VFmaAkF32
            | VFmaMkF16
            | VFmaAkF16
    )
}

impl<'a> Translator<'a> {
    /// Compare destination: EXEC for v_cmpx, else the scalar destination
    /// (VCC for VOPC).
    fn compare_result(&mut self, ins: &Instruction, value: Value, cmpx: bool) {
        if cmpx {
            let exec = self.exec_bit();
            let masked = self.b().land(exec, value);
            self.set_exec_bit(masked);
            return;
        }
        let dst = ins.control.as_ref().and_then(Control::scalar_destination).unwrap_or(VCC_LO);
        // write_mask clears the lanes EXEC disables.
        self.write_mask(dst, value);
    }

    /// v_cmp_class: `mask` bit per class (0 sNaN, 1 qNaN, 2 -inf, 3 -normal,
    /// 4 -denormal, 5 -0, 6 +0, 7 +denormal, 8 +normal, 9 +inf).
    pub fn float_class(&mut self, bits: Value, mask: Value, half: bool) -> Value {
        let c = |t: &mut Self, x: u32| t.b().u32(x);
        let (sign_bit, mag_mask, exp_inf, quiet_bit, min_normal) =
            if half { (0x8000, 0x7FFF, 0x7C00, 0x200, 0x400) } else { (0x8000_0000, 0x7FFF_FFFF, 0x7F80_0000, 0x40_0000, 0x80_0000) };
        // An immediate mask of no class, of every class or of exactly the
        // two NaN classes is a constant or one magnitude comparison.
        if let Some(m) = self.func.as_const(mask) {
            match m as u32 & 0x3FF {
                0 => return self.b().bool(false),
                0x3FF => return self.b().bool(true),
                3 => {
                    let mm = c(self, mag_mask);
                    let mag = self.b().and(bits, mm);
                    let einf = c(self, exp_inf);
                    return self.b().icmp(ICmp::UGt, mag, einf);
                }
                _ => {}
            }
        }
        let sb = c(self, sign_bit);
        let sign = self.b().and(bits, sb);
        let zero = c(self, 0);
        let negative = self.b().icmp(ICmp::Ne, sign, zero);
        let mm = c(self, mag_mask);
        let mag = self.b().and(bits, mm);
        let pick = |t: &mut Self, neg: u32, pos: u32| {
            let n = t.b().u32(neg);
            let p = t.b().u32(pos);
            t.b().select(negative, n, p)
        };
        let mut kind = pick(self, 3, 8);
        let mn = c(self, min_normal);
        let denorm = self.b().icmp(ICmp::ULt, mag, mn);
        let dk = pick(self, 4, 7);
        kind = self.b().select(denorm, dk, kind);
        let is_zero = self.b().icmp(ICmp::Eq, mag, zero);
        let zk = pick(self, 5, 6);
        kind = self.b().select(is_zero, zk, kind);
        let einf = c(self, exp_inf);
        let is_inf = self.b().icmp(ICmp::Eq, mag, einf);
        let ik = pick(self, 2, 9);
        kind = self.b().select(is_inf, ik, kind);
        let is_nan = self.b().icmp(ICmp::UGt, mag, einf);
        let qb = c(self, quiet_bit);
        let q = self.b().and(mag, qb);
        let quiet = self.b().icmp(ICmp::Ne, q, zero);
        let one = c(self, 1);
        let nan_kind = self.b().select(quiet, one, zero);
        kind = self.b().select(is_nan, nan_kind, kind);
        let shifted = self.b().lshr(mask, kind);
        let bit = self.b().and(shifted, one);
        self.b().icmp(ICmp::Ne, bit, zero)
    }

    /// v_cmp_class_f64 on the high word (low-word bits folded into bit 0).
    fn float_class64(&mut self, hi: Value, mask: Value) -> Value {
        let c = |t: &mut Self, x: u32| t.b().u32(x);
        let sb = c(self, 0x8000_0000);
        let s = self.b().and(hi, sb);
        let zero = c(self, 0);
        let negative = self.b().icmp(ICmp::Ne, s, zero);
        let mm = c(self, 0x7FFF_FFFF);
        let mag = self.b().and(hi, mm);
        let em = c(self, 0x7FF0_0000);
        let e = self.b().and(hi, em);
        let fm = c(self, 0x000F_FFFF);
        let frac = self.b().and(hi, fm);
        let frac_nz = self.b().icmp(ICmp::Ne, frac, zero);
        let qb = c(self, 0x0008_0000);
        let q = self.b().and(hi, qb);
        let quiet = self.b().icmp(ICmp::Ne, q, zero);
        let pick = |t: &mut Self, n: u32, p: u32| {
            let a = t.b().u32(n);
            let b = t.b().u32(p);
            t.b().select(negative, a, b)
        };
        let one = c(self, 1);
        let nan = self.b().select(quiet, one, zero);
        let inf = pick(self, 2, 9);
        let normal = pick(self, 3, 8);
        let denorm = pick(self, 4, 7);
        let szero = pick(self, 5, 6);
        let emax = self.b().select(frac_nz, nan, inf);
        let ezero = self.b().select(frac_nz, denorm, szero);
        let ez = self.b().icmp(ICmp::Eq, e, zero);
        let finite = self.b().select(ez, ezero, normal);
        let is_max = self.b().icmp(ICmp::Eq, e, em);
        let index = self.b().select(is_max, emax, finite);
        let _ = mag;
        let shifted = self.b().lshr(mask, index);
        let bit = self.b().and(shifted, one);
        self.b().icmp(ICmp::Ne, bit, zero)
    }

    /// v_cmp_*_f32 when f32 input denormals flush and one operand is a
    /// constant: one compare of the other operand, unflushed. A denormal
    /// and its flushed zero compare alike with a NaN, an infinity or a
    /// normal constant; against a zero, flush(x) < 0 is x <= -2^-126,
    /// flush(x) == 0 is |x| < 2^-126, and so on (unordered forms alike).
    /// None when the rule is off, nothing flushes or no operand is constant.
    fn f32_compare_against_constant(&mut self, ins: &Instruction, m: &Mods, p: FCmp) -> Option<Value> {
        use FCmp::*;
        if self.f32_flush & 1 == 0 || self.options.float_rules & super::float_rules::COMPARE_CONSTANT_FOLD == 0 {
            return None;
        }
        if matches!(p, Ord | Uno) {
            return None;
        }
        let constant = |t: &mut Self, i: usize| -> Option<u32> {
            let bits = t.src_bits(ins, m, i);
            let mut c = t.func.as_const(bits)? as u32;
            if m.abs(i) {
                c &= 0x7FFF_FFFF;
            }
            if m.neg(i) {
                c ^= 0x8000_0000;
            }
            Some(c)
        };
        let (ci, c) = match constant(self, 1) {
            Some(c) => (1, c),
            None => (0, constant(self, 0)?),
        };
        let xi = 1 - ci;
        let mut x = self.src_bits(ins, m, xi);
        if m.abs(xi) {
            let k = self.b().u32(0x7FFF_FFFF);
            x = self.b().and(x, k);
        }
        if m.neg(xi) {
            let k = self.b().u32(0x8000_0000);
            x = self.b().xor(x, k);
        }
        let x = self.b().bitcast(x, Type::F32);
        if c & 0x7F80_0000 != 0 {
            // Normal, infinite or NaN constant: the flush cannot change the
            // outcome.
            let k = self.b().f32(f32::from_bits(c));
            return Some(if ci == 1 { self.b().fcmp(p, x, k) } else { self.b().fcmp(p, k, x) });
        }
        // A zero (or a denormal, which flushes to one): x on the left.
        let p = if ci == 1 {
            p
        } else {
            match p {
                OLt => OGt,
                OLe => OGe,
                OGt => OLt,
                OGe => OLe,
                ULt => UGt,
                ULe => UGe,
                UGt => ULt,
                UGe => ULe,
                other => other,
            }
        };
        const MIN: f32 = f32::MIN_POSITIVE;
        let (q, magnitude, threshold) = match p {
            OLt => (OLe, false, -MIN),
            ULt => (ULe, false, -MIN),
            OLe => (OLt, false, MIN),
            ULe => (ULt, false, MIN),
            OGt => (OGe, false, MIN),
            UGt => (UGe, false, MIN),
            OGe => (OGt, false, -MIN),
            UGe => (UGt, false, -MIN),
            OEq => (OLt, true, MIN),
            UEq => (ULt, true, MIN),
            ONe => (OGe, true, MIN),
            UNe => (UGe, true, MIN),
            Ord | Uno => unreachable!(),
        };
        let x = if magnitude { self.b().fabs(x) } else { x };
        let k = self.b().f32(threshold);
        Some(self.b().fcmp(q, x, k))
    }

    fn translate_compare(&mut self, ins: &Instruction, m: &Mods) -> Result<bool> {
        use Opcode::*;
        if let Some((kind, half, cmpx)) = float_compare(ins.opcode) {
            let v = match kind {
                FKind::False => self.b().bool(false),
                FKind::True => self.b().bool(true),
                FKind::P(p) => {
                    let folded = if half { None } else { self.f32_compare_against_constant(ins, m, p) };
                    match folded {
                        Some(v) => v,
                        None => {
                            let (a, b) = if half {
                                // Inputs follow the f16 denormal mode (RDNA2).
                                let a = self.src_f16(ins, m, 0);
                                let b = self.src_f16(ins, m, 1);
                                (self.flush_f16_input_value(a), self.flush_f16_input_value(b))
                            } else {
                                (self.src_f32(ins, m, 0), self.src_f32(ins, m, 1))
                            };
                            self.b().fcmp(p, a, b)
                        }
                    }
                }
            };
            self.compare_result(ins, v, cmpx);
            return Ok(true);
        }
        if let Some((pred, constant, width, cmpx)) = int_compare(ins.opcode) {
            let v = match pred {
                None => self.b().bool(constant),
                Some(p) => {
                    let (a, b) = match width {
                        16 => {
                            let signed = matches!(p, ICmp::SLt | ICmp::SLe | ICmp::SGt | ICmp::SGe);
                            let a = self.src_half_bits_int(ins, m, 0, false);
                            let b = self.src_half_bits_int(ins, m, 1, false);
                            if signed {
                                let zero = self.b().u32(0);
                                let sixteen = self.b().u32(16);
                                (self.b().bfe_s(a, zero, sixteen), self.b().bfe_s(b, zero, sixteen))
                            } else {
                                (a, b)
                            }
                        }
                        64 if signed_i64_compare(ins.opcode) => (self.src_i64(ins, 0), self.src_i64(ins, 1)),
                        64 => (self.src_u64(ins, 0), self.src_u64(ins, 1)),
                        _ => (self.src_u32(ins, m, 0), self.src_u32(ins, m, 1)),
                    };
                    self.b().icmp(p, a, b)
                }
            };
            self.compare_result(ins, v, cmpx);
            return Ok(true);
        }
        if let Some((kind, cmpx)) = f64_compare(ins.opcode) {
            let v = match kind {
                FKind::False => self.b().bool(false),
                FKind::True => self.b().bool(true),
                FKind::P(p) => {
                    // f64 compares flush input denormals by FLOAT_MODE bit 6.
                    let a = self.src_f64(ins, m, 0);
                    let a = self.flush_f64_input(a);
                    let b = self.src_f64(ins, m, 1);
                    let b = self.flush_f64_input(b);
                    self.b().fcmp(p, a, b)
                }
            };
            self.compare_result(ins, v, cmpx);
            return Ok(true);
        }
        match ins.opcode {
            VCmpClassF16 | VCmpxClassF16 => {
                let bits = self.src_f16_bits(ins, m, 0);
                let mask = self.src_u32(ins, m, 1);
                let v = self.float_class(bits, mask, true);
                self.compare_result(ins, v, ins.opcode == VCmpxClassF16);
                Ok(true)
            }
            VCmpClassF64 | VCmpxClassF64 => {
                let (lo, mut hi) = self.read_f64_parts(ins.sources[0]);
                if m.abs(0) {
                    let c = self.b().u32(0x7FFF_FFFF);
                    hi = self.b().and(hi, c);
                }
                if m.neg(0) {
                    let c = self.b().u32(0x8000_0000);
                    hi = self.b().xor(hi, c);
                }
                // Classify through the high word, folding a non-zero low
                // word into the mantissa test.
                let zero = self.b().u32(0);
                let lnz = self.b().icmp(ICmp::Ne, lo, zero);
                let one = self.b().u32(1);
                let sticky = self.b().select(lnz, one, zero);
                let hi_s = self.b().or(hi, sticky);
                let mask = self.src_u32(ins, m, 1);
                let v = self.float_class64(hi_s, mask);
                self.compare_result(ins, v, ins.opcode == VCmpxClassF64);
                Ok(true)
            }
            VCmpClassF32 | VCmpxClassF32 => {
                let mut bits = self.src_bits(ins, m, 0);
                if m.abs(0) {
                    let c = self.b().u32(0x7FFF_FFFF);
                    bits = self.b().and(bits, c);
                }
                if m.neg(0) {
                    let c = self.b().u32(0x8000_0000);
                    bits = self.b().xor(bits, c);
                }
                let mask = self.src_u32(ins, m, 1);
                let v = self.float_class(bits, mask, false);
                self.compare_result(ins, v, ins.opcode == VCmpxClassF32);
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Source with float abs/neg applied as bit operations (for moves and
    /// selects).
    fn src_bits_mod(&mut self, ins: &Instruction, m: &Mods, i: usize) -> Value {
        let v = self.src_f32(ins, m, i);
        self.b().bitcast(v, Type::U32)
    }

    /// Carry-out destination of an add/sub with carry.
    fn carry_destination(&self, ins: &Instruction) -> u32 {
        ins.control.as_ref().and_then(Control::scalar_destination).unwrap_or(VCC_LO)
    }

    /// Carry-in mask operand: VCC for VOP2, source 2 for VOP3.
    fn carry_in(&mut self, ins: &Instruction) -> Value {
        if ins.encoding == Encoding::Vop3 && ins.sources.len() >= 3 {
            self.read_mask(ins.sources[2])
        } else {
            self.read_mask(Operand::scalar(VCC_LO))
        }
    }

    pub(crate) fn translate_valu(&mut self, ins: &Instruction) -> Result<()> {
        use Opcode::*;
        let m = Mods::of(ins);
        if (m.dpp.is_some() || m.dpp8.is_some()) && !has_dpp_form(ins.opcode) {
            return self.invalid(ins, "opcode has no DPP form");
        }
        if self.translate_compare(ins, &m)? {
            return Ok(());
        }
        if self.translate_float(ins, &m)? {
            return Ok(());
        }
        let dm = DstMods::of(&m);
        match ins.opcode {
            VMovB32 => {
                let v = self.src_f32(ins, &m, 0);
                self.write_f32_result(ins, &m, v)
            }
            VCndmaskB32 => {
                let mask = if ins.encoding == Encoding::Vop3 { self.read_mask(ins.sources[2]) } else { self.read_mask(Operand::scalar(VCC_LO)) };
                let f = self.src_bits_mod(ins, &m, 0);
                let t = self.src_bits_mod(ins, &m, 1);
                let r = self.b().select(mask, t, f);
                self.write_u32_result(ins, &m, r)
            }
            VNotB32 | VBfrevB32 | VFfbhU32 | VFfblB32 | VFfbhI32 => {
                let v = self.src_u32(ins, &m, 0);
                let r = match ins.opcode {
                    VNotB32 => self.b().not(v),
                    VBfrevB32 => self.b().bit_reverse(v),
                    VFfblB32 => self.b().find_lsb(v),
                    VFfbhU32 => self.leading_zeros32(v, false),
                    _ => self.leading_zeros32(v, true),
                };
                self.write_u32_result(ins, &m, r)
            }
            VAndB32 | VOrB32 | VXorB32 | VXnorB32 | VMinI32 | VMaxI32 | VMinU32 | VMaxU32 | VMulLoU32 | VMulLoI32
            | VMulHiU32 | VMulHiI32 => {
                let a = self.src_u32(ins, &m, 0);
                let b = self.src_u32(ins, &m, 1);
                let r = match ins.opcode {
                    VAndB32 => self.b().and(a, b),
                    VOrB32 => self.b().or(a, b),
                    VXorB32 => self.b().xor(a, b),
                    VXnorB32 => {
                        let x = self.b().xor(a, b);
                        self.b().not(x)
                    }
                    VMinI32 => self.b().smin(a, b),
                    VMaxI32 => self.b().smax(a, b),
                    VMinU32 => self.b().umin(a, b),
                    VMaxU32 => self.b().umax(a, b),
                    VMulLoU32 | VMulLoI32 => self.b().imul(a, b),
                    VMulHiU32 => self.b().umul_hi(a, b),
                    _ => self.b().smul_hi(a, b),
                };
                self.write_u32_result(ins, &m, r)
            }
            VLshrB32 | VLshrrevB32 | VAshrI32 | VAshrrevI32 | VLshlB32 | VLshlrevB32 => {
                let rev = matches!(ins.opcode, VLshrrevB32 | VAshrrevI32 | VLshlrevB32);
                let (vi, ci) = if rev { (1, 0) } else { (0, 1) };
                let v = self.src_u32(ins, &m, vi);
                let c = self.src_u32(ins, &m, ci);
                let m31 = self.b().u32(31);
                let c = self.b().and(c, m31);
                let op = match ins.opcode {
                    VLshrB32 | VLshrrevB32 => Op::LShr,
                    VAshrI32 | VAshrrevI32 => Op::AShr,
                    _ => Op::Shl,
                };
                let r = self.b().emit(op, Type::U32, vec![v, c]);
                self.write_u32_result(ins, &m, r)
            }
            VLshlrevB64 | VLshrrevB64 => {
                let c = self.src_u32(ins, &m, 0);
                let v = self.src_u64(ins, 1);
                let op = if ins.opcode == VLshlrevB64 { Op::Shl } else { Op::LShr };
                let r = self.shift64(op, v, c);
                let (lo, hi) = self.b().split_u64(r);
                let d = self.vdst(ins)?;
                self.write_vgpr(d, lo);
                self.write_vgpr(d + 1, hi);
                Ok(())
            }
            VAddI32 | VSubI32 | VSubrevI32 => {
                // gfx10 v_add_nc_u32 / v_sub_nc_u32 / v_subrev_nc_u32.
                let a = self.src_u32(ins, &m, 0);
                let b = self.src_u32(ins, &m, 1);
                let (l, r) = if ins.opcode == VSubrevI32 { (b, a) } else { (a, b) };
                let sub = ins.opcode != VAddI32;
                let mut res = if sub { self.b().isub(l, r) } else { self.b().iadd(l, r) };
                if m.clamp {
                    let carry = if sub { self.b().icmp(ICmp::ULt, l, r) } else { self.b().icmp(ICmp::ULt, res, l) };
                    let sat = self.b().u32(if sub { 0 } else { u32::MAX });
                    res = self.b().select(carry, sat, res);
                }
                self.write_u32_result(ins, &m, res)
            }
            VAddCoU32 | VAddcU32 | VAddCoCiU32 => {
                let a = self.src_u32(ins, &m, 0);
                let b = self.src_u32(ins, &m, 1);
                let cin = if ins.opcode == VAddCoU32 { None } else { Some(self.carry_in(ins)) };
                let (sum, carry) = self.add_with_carry(a, b, cin);
                let sum = if m.clamp {
                    let ones = self.b().u32(u32::MAX);
                    self.b().select(carry, ones, sum)
                } else {
                    sum
                };
                self.write_u32_result(ins, &m, sum)?;
                let cd = self.carry_destination(ins);
                self.write_mask(cd, carry);
                Ok(())
            }
            VSubCoU32 | VSubrevCoU32 | VSubbU32 | VSubbrevU32 | VSubCoCiU32 | VSubrevCoCiU32 => {
                let a = self.src_u32(ins, &m, 0);
                let b = self.src_u32(ins, &m, 1);
                let rev = matches!(ins.opcode, VSubrevCoU32 | VSubbrevU32 | VSubrevCoCiU32);
                let (l, r) = if rev { (b, a) } else { (a, b) };
                let bin = if matches!(ins.opcode, VSubCoU32 | VSubrevCoU32) { None } else { Some(self.carry_in(ins)) };
                let (diff, borrow) = self.sub_with_borrow(l, r, bin);
                let diff = if m.clamp {
                    let zero = self.b().u32(0);
                    self.b().select(borrow, zero, diff)
                } else {
                    diff
                };
                self.write_u32_result(ins, &m, diff)?;
                let cd = self.carry_destination(ins);
                self.write_mask(cd, borrow);
                Ok(())
            }
            VMadU64U32 => {
                let a = self.src_u32(ins, &m, 0);
                let b = self.src_u32(ins, &m, 1);
                let (al, ah) = self.read_u64_parts(ins.sources[2]);
                let lo = self.b().imul(a, b);
                let hi = self.b().umul_hi(a, b);
                let low = self.b().iadd(lo, al);
                let carry_low = self.b().icmp(ICmp::ULt, low, lo);
                let high0 = self.b().iadd(hi, ah);
                let carry0 = self.b().icmp(ICmp::ULt, high0, hi);
                let cl = self.b().bool_to_u32(carry_low);
                let high = self.b().iadd(high0, cl);
                let carry1 = self.b().icmp(ICmp::ULt, high, high0);
                let carry = self.b().lor(carry0, carry1);
                let (rl, rh) = if m.clamp {
                    let ones = self.b().u32(u32::MAX);
                    (self.b().select(carry, ones, low), self.b().select(carry, ones, high))
                } else {
                    (low, high)
                };
                let d = self.vdst(ins)?;
                self.write_vgpr(d, rl);
                self.write_vgpr(d + 1, rh);
                let cd = self.carry_destination(ins);
                self.write_mask(cd, carry);
                Ok(())
            }
            VMulI32I24 | VMulU32U24 | VMulHiU32U24 | VMadI32I24 | VMadU32U24 => {
                let signed = matches!(ins.opcode, VMulI32I24 | VMadI32I24);
                let addend = matches!(ins.opcode, VMadI32I24 | VMadU32U24);
                let high = ins.opcode == VMulHiU32U24;
                let a = self.src_u32(ins, &m, 0);
                let b = self.src_u32(ins, &m, 1);
                let zero = self.b().u32(0);
                let w = self.b().u32(24);
                let (a, b) = if signed {
                    (self.b().bfe_s(a, zero, w), self.b().bfe_s(b, zero, w))
                } else {
                    (self.b().bfe_u(a, zero, w), self.b().bfe_u(b, zero, w))
                };
                let mut r = if high {
                    if signed {
                        self.b().smul_hi(a, b)
                    } else {
                        self.b().umul_hi(a, b)
                    }
                } else {
                    self.b().imul(a, b)
                };
                if m.clamp && !high && !addend {
                    let top = if signed { self.b().smul_hi(a, b) } else { self.b().umul_hi(a, b) };
                    let expected = if signed {
                        let c31 = self.b().u32(31);
                        self.b().ashr(r, c31)
                    } else {
                        zero
                    };
                    let sat = if signed {
                        let c31 = self.b().u32(31);
                        let s = self.b().ashr(top, c31);
                        let mx = self.b().u32(0x7FFF_FFFF);
                        self.b().xor(s, mx)
                    } else {
                        self.b().u32(u32::MAX)
                    };
                    let ov = self.b().icmp(ICmp::Ne, top, expected);
                    r = self.b().select(ov, sat, r);
                }
                if addend {
                    let top = if m.clamp {
                        if signed {
                            self.b().smul_hi(a, b)
                        } else {
                            self.b().umul_hi(a, b)
                        }
                    } else {
                        zero
                    };
                    let c = self.src_u32(ins, &m, 2);
                    r = self.wide_add(m.clamp, r, top, c, signed);
                }
                self.write_u32_result(ins, &m, r)
            }
            VBfeU32 | VBfeI32 => {
                let src = self.src_u32(ins, &m, 0);
                let off = self.src_u32(ins, &m, 1);
                let cnt = self.src_u32(ins, &m, 2);
                let m31 = self.b().u32(31);
                let off = self.b().and(off, m31);
                let cnt = self.b().and(cnt, m31);
                let c32 = self.b().u32(32);
                let avail = self.b().isub(c32, off);
                let count = self.b().umin(cnt, avail);
                let r = if ins.opcode == VBfeI32 { self.b().bfe_s(src, off, count) } else { self.b().bfe_u(src, off, count) };
                self.write_u32_result(ins, &m, r)
            }
            VBfiB32 => {
                let mask = self.src_u32(ins, &m, 0);
                let ins_v = self.src_u32(ins, &m, 1);
                let base = self.src_u32(ins, &m, 2);
                let a = self.b().and(mask, ins_v);
                let nm = self.b().not(mask);
                let b = self.b().and(nm, base);
                let r = self.b().or(a, b);
                self.write_u32_result(ins, &m, r)
            }
            VBfmB32 => {
                let count = self.src_u32(ins, &m, 0);
                let off = self.src_u32(ins, &m, 1);
                let m31 = self.b().u32(31);
                let count = self.b().and(count, m31);
                let off = self.b().and(off, m31);
                let r = self.bfm32(count, off);
                self.write_u32_result(ins, &m, r)
            }
            VAlignbitB32 => {
                let hi = self.src_u32(ins, &m, 0);
                let lo = self.src_u32(ins, &m, 1);
                let s = self.align_count(ins, &m);
                let m31 = self.b().u32(31);
                let s = self.b().and(s, m31);
                let v = self.b().join_u64(lo, hi);
                let shifted = self.shift64(Op::LShr, v, s);
                let (r, _) = self.b().split_u64(shifted);
                let r = self.align_destination(&m, r);
                self.write_u32_result(ins, &m, r)
            }
            VAlignbyteB32 => {
                let hi = self.src_u32(ins, &m, 0);
                let lo = self.src_u32(ins, &m, 1);
                let s = self.align_count(ins, &m);
                let three = self.b().u32(3);
                let s = self.b().and(s, three);
                let s = self.b().shl(s, three);
                let v = self.b().join_u64(lo, hi);
                let shifted = self.shift64(Op::LShr, v, s);
                let (r, _) = self.b().split_u64(shifted);
                let r = self.align_destination(&m, r);
                self.write_u32_result(ins, &m, r)
            }
            VMin3I32 | VMax3I32 | VMed3I32 | VMin3U32 | VMax3U32 | VMed3U32 => {
                let a = self.src_u32(ins, &m, 0);
                let b = self.src_u32(ins, &m, 1);
                let c = self.src_u32(ins, &m, 2);
                let signed = matches!(ins.opcode, VMin3I32 | VMax3I32 | VMed3I32);
                let (mn, mx) = if signed { (Op::SMin, Op::SMax) } else { (Op::UMin, Op::UMax) };
                let r = match ins.opcode {
                    VMin3I32 | VMin3U32 => {
                        let ab = self.b().emit(mn, Type::U32, vec![a, b]);
                        self.b().emit(mn, Type::U32, vec![ab, c])
                    }
                    VMax3I32 | VMax3U32 => {
                        let ab = self.b().emit(mx, Type::U32, vec![a, b]);
                        self.b().emit(mx, Type::U32, vec![ab, c])
                    }
                    _ => {
                        // max(min(a, b), min(max(a, b), c))
                        let lo = self.b().emit(mn, Type::U32, vec![a, b]);
                        let hi = self.b().emit(mx, Type::U32, vec![a, b]);
                        let hc = self.b().emit(mn, Type::U32, vec![hi, c]);
                        self.b().emit(mx, Type::U32, vec![lo, hc])
                    }
                };
                self.write_u32_result(ins, &m, r)
            }
            VSadU8 | VSadHiU8 | VSadU16 | VSadU32 => {
                let a = self.src_u32(ins, &m, 0);
                let b = self.src_u32(ins, &m, 1);
                let c = self.src_u32(ins, &m, 2);
                let sum = match ins.opcode {
                    VSadU32 => {
                        let lo = self.b().umin(a, b);
                        let hi = self.b().umax(a, b);
                        self.b().isub(hi, lo)
                    }
                    VSadU16 => self.byte_sad(a, b, 16),
                    _ => self.byte_sad(a, b, 8),
                };
                let sum = if ins.opcode == VSadHiU8 {
                    let sixteen = self.b().u32(16);
                    self.b().shl(sum, sixteen)
                } else {
                    sum
                };
                let zero = self.b().u32(0);
                let r = self.wide_add(m.clamp, sum, zero, c, false);
                self.write_u32_result(ins, &m, r)
            }
            VBcntU32B32 => {
                let a = self.src_u32(ins, &m, 0);
                let b = self.src_u32(ins, &m, 1);
                let c = self.b().bit_count(a);
                let r = self.b().iadd(c, b);
                self.write_u32_result(ins, &m, r)
            }
            VMbcntLoU32B32 | VMbcntHiU32B32 => {
                let low = ins.opcode == VMbcntLoU32B32;
                let lane = self.b().lane_id();
                let m31 = self.b().u32(31);
                let local = self.b().and(lane, m31);
                let one = self.b().u32(1);
                let bit = self.b().shl(one, local);
                let below = self.b().isub(bit, one);
                let c32 = self.b().u32(32);
                let high_lane = self.b().icmp(ICmp::UGe, lane, c32);
                let ones = self.b().u32(u32::MAX);
                let zero = self.b().u32(0);
                let thread_mask = if low { self.b().select(high_lane, ones, below) } else { self.b().select(high_lane, below, zero) };
                let src = self.src_u32(ins, &m, 0);
                let act = self.b().and(src, thread_mask);
                let cnt = self.b().bit_count(act);
                let add = self.src_u32(ins, &m, 1);
                let r = self.b().iadd(cnt, add);
                self.write_u32_result(ins, &m, r)
            }
            VLshlAddU32 | VAddLshlU32 | VLshlOrU32 | VXadU32 | VAdd3U32 | VXor3B32 | VAndOrB32 | VOr3U32 => {
                let a = self.src_u32(ins, &m, 0);
                let b = self.src_u32(ins, &m, 1);
                let c = self.src_u32(ins, &m, 2);
                let m31 = self.b().u32(31);
                let r = match ins.opcode {
                    VLshlAddU32 => {
                        let s = self.b().and(b, m31);
                        let sh = self.b().shl(a, s);
                        self.b().iadd(sh, c)
                    }
                    VAddLshlU32 => {
                        let s = self.b().and(c, m31);
                        let sum = self.b().iadd(a, b);
                        self.b().shl(sum, s)
                    }
                    VLshlOrU32 => {
                        let s = self.b().and(b, m31);
                        let sh = self.b().shl(a, s);
                        self.b().or(sh, c)
                    }
                    VXadU32 => {
                        let x = self.b().xor(a, b);
                        self.b().iadd(x, c)
                    }
                    VAdd3U32 => {
                        let s = self.b().iadd(a, b);
                        self.b().iadd(s, c)
                    }
                    VXor3B32 => {
                        let x = self.b().xor(a, b);
                        self.b().xor(x, c)
                    }
                    VAndOrB32 => {
                        let x = self.b().and(a, b);
                        self.b().or(x, c)
                    }
                    _ => {
                        let x = self.b().or(a, b);
                        self.b().or(x, c)
                    }
                };
                self.write_u32_result(ins, &m, r)
            }
            VCvtPkU16U32 | VCvtPkI16I32 => {
                let signed = ins.opcode == VCvtPkI16I32;
                let mut parts = Vec::new();
                for i in 0..2 {
                    let v = self.src_u32(ins, &m, i);
                    let s = if signed {
                        let mx = self.b().u32(0x7FFF);
                        let mn = self.b().u32(0xFFFF_8000);
                        let t = self.b().smin(v, mx);
                        self.b().smax(t, mn)
                    } else {
                        let mx = self.b().u32(0xFFFF);
                        self.b().umin(v, mx)
                    };
                    parts.push(s);
                }
                let r = self.pack_u16(parts[0], parts[1]);
                self.write_u32_result(ins, &m, r)
            }
            VAddNcU16 | VSubNcU16 | VAddNcI16 | VSubNcI16 | VMaxU16 | VMaxI16 | VMinU16 | VMinI16 => {
                let signed = matches!(ins.opcode, VMaxI16 | VMinI16);
                let a = self.src_u16(ins, &m, 0, signed);
                let b = self.src_u16(ins, &m, 1, signed);
                let r = match ins.opcode {
                    VAddNcU16 | VAddNcI16 => self.b().iadd(a, b),
                    VSubNcU16 | VSubNcI16 => self.b().isub(a, b),
                    VMaxU16 => self.b().umax(a, b),
                    VMaxI16 => self.b().smax(a, b),
                    VMinU16 => self.b().umin(a, b),
                    _ => self.b().smin(a, b),
                };
                let r = if m.clamp && matches!(ins.opcode, VAddNcU16 | VSubNcU16 | VAddNcI16 | VSubNcI16) {
                    let s = matches!(ins.opcode, VAddNcI16 | VSubNcI16);
                    // Operands were read zero-extended; recompute signed.
                    if s {
                        let a = self.src_u16(ins, &m, 0, true);
                        let b = self.src_u16(ins, &m, 1, true);
                        let r = if ins.opcode == VAddNcI16 { self.b().iadd(a, b) } else { self.b().isub(a, b) };
                        self.saturate16(r, true)
                    } else {
                        self.saturate16(r, false)
                    }
                } else {
                    r
                };
                let d = self.vdst(ins)?;
                let ffff = self.b().u32(0xFFFF);
                let r = self.b().and(r, ffff);
                self.write_vgpr_half(d, r, m.op_sel(3), &dm);
                Ok(())
            }
            VLshlrevB16 | VLshrrevB16 | VAshrrevI16 => {
                let arith = ins.opcode == VAshrrevI16;
                let v = self.src_u16(ins, &m, 1, arith);
                let c = self.src_u16(ins, &m, 0, false);
                let f = self.b().u32(15);
                let c = self.b().and(c, f);
                let op = match ins.opcode {
                    VLshlrevB16 => Op::Shl,
                    VLshrrevB16 => Op::LShr,
                    _ => Op::AShr,
                };
                let r = self.b().emit(op, Type::U32, vec![v, c]);
                let ffff = self.b().u32(0xFFFF);
                let r = self.b().and(r, ffff);
                let d = self.vdst(ins)?;
                self.write_vgpr_half(d, r, m.op_sel(3), &dm);
                Ok(())
            }
            VMadU32U16 => {
                let a = self.src_u16(ins, &m, 0, false);
                let b = self.src_u16(ins, &m, 1, false);
                let p = self.b().imul(a, b);
                let c = self.src_u32(ins, &m, 2);
                let zero = self.b().u32(0);
                let r = self.wide_add(m.clamp, p, zero, c, false);
                self.write_u32_result(ins, &m, r)
            }
            VPkAddI16 | VPkSubI16 => {
                let mut halves = Vec::new();
                for high in [false, true] {
                    let a = self.src_packed_u16(ins, &m, 0, high, true);
                    let b = self.src_packed_u16(ins, &m, 1, high, true);
                    let r = if ins.opcode == VPkAddI16 { self.b().iadd(a, b) } else { self.b().isub(a, b) };
                    let r = if m.clamp { self.saturate16(r, true) } else { r };
                    halves.push(r);
                }
                let r = self.pack_u16(halves[0], halves[1]);
                let d = self.vdst(ins)?;
                self.write_vgpr(d, r);
                Ok(())
            }
            VReadfirstlaneB32 => {
                let v = self.src_u32(ins, &m, 0);
                let lane = self.first_active_lane();
                let r = self.b().read_lane(v, lane);
                let dst = ins.destinations[0].value;
                self.write_sgpr(dst, r);
                Ok(())
            }
            VReadlaneB32 => {
                let v = self.src_u32(ins, &m, 0);
                let l = self.src_u32(ins, &m, 1);
                let mask = self.wave_mask();
                let l = self.b().and(l, mask);
                let r = self.b().read_lane(v, l);
                let dst = ins.destinations[0].value;
                self.write_sgpr(dst, r);
                Ok(())
            }
            VWritelaneB32 => {
                let v = self.src_u32(ins, &m, 0);
                let l = self.src_u32(ins, &m, 1);
                let mask = self.wave_mask();
                let l = self.b().and(l, mask);
                let d = self.vdst(ins)?;
                let old = self.vgpr(d);
                let lane = self.b().lane_id();
                let hit = self.b().icmp(ICmp::Eq, lane, l);
                let r = self.b().select(hit, v, old);
                self.write_vgpr_unmasked(d, r);
                Ok(())
            }
            VPermlane16B32 | VPermlanex16B32 => {
                let x16 = ins.opcode == VPermlanex16B32;
                let fetch_inactive = m.op_sel(0);
                let bound_control = m.op_sel(1);
                let v = self.src_u32(ins, &m, 0);
                let sel_lo = self.src_u32(ins, &m, 1);
                let sel_hi = self.src_u32(ins, &m, 2);
                let d = self.vdst(ins)?;
                let old = self.vgpr(d);
                let lane = self.b().lane_id();
                let rm = self.b().u32(0xFFFF_FFF0);
                let row = self.b().and(lane, rm);
                let row = if x16 {
                    let sixteen = self.b().u32(16);
                    self.b().xor(row, sixteen)
                } else {
                    row
                };
                let fifteen = self.b().u32(15);
                let l = self.b().and(lane, fifteen);
                let seven = self.b().u32(7);
                let l8 = self.b().and(l, seven);
                let two = self.b().u32(2);
                let shift = self.b().shl(l8, two);
                let eight = self.b().u32(8);
                let upper = self.b().icmp(ICmp::UGe, l, eight);
                let sel = self.b().select(upper, sel_hi, sel_lo);
                let sh = self.b().lshr(sel, shift);
                let idx = self.b().and(sh, fifteen);
                let target = self.b().or(row, idx);
                let wm = self.wave_mask();
                let target = self.b().and(target, wm);
                let shuffled = self.b().shuffle(v, target);
                let r = if fetch_inactive {
                    shuffled
                } else {
                    let active = self.lane_active(target);
                    let fallback = if bound_control { self.b().u32(0) } else { old };
                    self.b().select(active, shuffled, fallback)
                };
                self.write_vgpr(d, r);
                Ok(())
            }
            VMovrelsB32 | VMovreldB32 | VMovrelsdB32 | VMovrelsd2B32 | VSwaprelB32 => self.v_movrel(ins, &m),
            _ => {
                if self.translate_extra_valu(ins, &m)? {
                    return Ok(());
                }
                self.unsupported(ins, "vector ALU")
            }
        }
    }

    /// Index of the first lane with EXEC set (0 when EXEC is zero).
    pub fn first_active_lane(&mut self) -> Value {
        let lo = self.sgpr(EXEC_LO);
        let lsb_lo = self.b().find_lsb(lo);
        if !self.wave64() {
            let zero = self.b().u32(0);
            let any = self.b().icmp(ICmp::Ne, lo, zero);
            return self.b().select(any, lsb_lo, zero);
        }
        let hi = self.sgpr(EXEC_HI);
        let lsb_hi = self.b().find_lsb(hi);
        let c32 = self.b().u32(32);
        let hpos = self.b().iadd(lsb_hi, c32);
        let zero = self.b().u32(0);
        let lnz = self.b().icmp(ICmp::Ne, lo, zero);
        let hnz = self.b().icmp(ICmp::Ne, hi, zero);
        let hsel = self.b().select(hnz, hpos, zero);
        self.b().select(lnz, lsb_lo, hsel)
    }

    /// `low + addend` with clamp saturation of a 64-bit `(low, high)` +
    /// 32-bit sum (unsigned, or signed with the addend's sign).
    pub fn wide_add(&mut self, clamp: bool, low: Value, high: Value, addend: Value, signed: bool) -> Value {
        let sum = self.b().iadd(low, addend);
        if !clamp {
            return sum;
        }
        let c = self.b().icmp(ICmp::ULt, sum, low);
        let carry = self.b().bool_to_u32(c);
        let top = self.b().iadd(high, carry);
        if !signed {
            let zero = self.b().u32(0);
            let nz = self.b().icmp(ICmp::Ne, top, zero);
            let ones = self.b().u32(u32::MAX);
            return self.b().select(nz, ones, sum);
        }
        let c31 = self.b().u32(31);
        let asign = self.b().ashr(addend, c31);
        let top = self.b().iadd(top, asign);
        let ssign = self.b().ashr(sum, c31);
        let ov = self.b().icmp(ICmp::Ne, top, ssign);
        let ts = self.b().ashr(top, c31);
        let mx = self.b().u32(0x7FFF_FFFF);
        let sat = self.b().xor(ts, mx);
        self.b().select(ov, sat, sum)
    }

    fn saturate16(&mut self, v: Value, signed: bool) -> Value {
        let lo = self.b().u32(if signed { 0xFFFF_8000 } else { 0 });
        let hi = self.b().u32(if signed { 0x7FFF } else { 0xFFFF });
        let t = self.b().smax(v, lo);
        self.b().smin(t, hi)
    }

    fn byte_sad(&mut self, a: Value, b: Value, field: u32) -> Value {
        let mut sum = self.b().u32(0);
        let mut off = 0;
        while off < 32 {
            let o = self.b().u32(off);
            let w = self.b().u32(field);
            let fa = self.b().bfe_u(a, o, w);
            let fb = self.b().bfe_u(b, o, w);
            let lo = self.b().umin(fa, fb);
            let hi = self.b().umax(fa, fb);
            let d = self.b().isub(hi, lo);
            sum = self.b().iadd(sum, d);
            off += field;
        }
        sum
    }

    /// M0-relative VGPR moves: a select over every VGPR the program names
    /// at or past the base.
    /// V_ALIGNBIT/ALIGNBYTE count: op_sel[1:0] selects a byte of src2.
    fn align_count(&mut self, ins: &Instruction, m: &Mods) -> Value {
        let s = self.src_u32(ins, m, 2);
        let byte = m.op_sel & 3;
        if byte == 0 {
            return s;
        }
        let off = self.b().u32(8 * byte);
        let eight = self.b().u32(8);
        self.b().bfe_u(s, off, eight)
    }

    /// V_ALIGNBIT/ALIGNBYTE op_sel[3]: the low 16 result bits go to the
    /// high half and the low half is zeroed.
    fn align_destination(&mut self, m: &Mods, r: Value) -> Value {
        if m.op_sel & 8 == 0 {
            return r;
        }
        let sixteen = self.b().u32(16);
        self.b().shl(r, sixteen)
    }

    fn v_movrel(&mut self, ins: &Instruction, m: &Mods) -> Result<()> {
        use Opcode::*;
        let m0 = self.sgpr(M0);
        let (src_off, dst_off) = if matches!(ins.opcode, VMovrelsd2B32 | VSwaprelB32) {
            let mask = self.b().u32(0x3FF);
            let s = self.b().and(m0, mask);
            let sixteen = self.b().u32(16);
            let d = self.b().lshr(m0, sixteen);
            let d = self.b().and(d, mask);
            (s, d)
        } else {
            (m0, m0)
        };
        let dm = DstMods::of(m);
        let d = self.vdst(ins)?;
        if ins.opcode == VSwaprelB32 {
            if m.dpp.is_some() || m.dpp8.is_some() || m.sdwa.is_some() {
                return self.invalid(ins, "v_swaprel_b32 has no DPP or SDWA form");
            }
            let base = ins.sources[0];
            if base.kind != OperandKind::VectorRegister {
                return self.invalid(ins, "relative source is not a VGPR");
            }
            // Neither side is written when either index is out of range.
            let src_value = self.relative_read(base.value, src_off);
            let dst_value = self.relative_read(d, dst_off);
            let src_in = self.relative_in_range(base.value, src_off);
            let dst_in = self.relative_in_range(d, dst_off);
            let both = self.b().land(src_in, dst_in);
            self.relative_write(d, dst_off, src_value, &dm, Some(both));
            self.relative_write(base.value, src_off, dst_value, &dm, Some(both));
            return Ok(());
        }
        let value = if ins.opcode == VMovreldB32 {
            self.src_u32(ins, m, 0)
        } else {
            let base = ins.sources[0];
            if base.kind != OperandKind::VectorRegister {
                return self.invalid(ins, "relative source is not a VGPR");
            }
            let v = self.relative_read(base.value, src_off);
            // DPP applies to the relatively addressed source.
            if let Some(p) = &m.dpp {
                self.dpp_read(v, p.control, p.row_mask, p.bank_mask, p.fetch_inactive)
            } else if let Some(p) = &m.dpp8 {
                self.dpp8_read(v, p.lane_selectors, p.fetch_inactive)
            } else {
                v
            }
        };
        if ins.opcode == VMovrelsB32 {
            self.write_vgpr_mods(d, value, &dm);
            return Ok(());
        }
        self.relative_write(d, dst_off, value, &dm, None);
        Ok(())
    }

    /// Whether `base + offset` lies in the relative-addressing window,
    /// which ends at the program's last named register (M0 is unsigned).
    fn relative_in_range(&mut self, base: u32, offset: Value) -> Value {
        let size = self.vector_limit.saturating_sub(base);
        let size = self.b().u32(size);
        self.b().icmp(ICmp::ULt, offset, size)
    }

    /// VGPR `base + offset` (uniform offset). An index at or past the
    /// window reads v0 (measured on RDNA2).
    fn relative_read(&mut self, base: u32, offset: Value) -> Value {
        let mut acc = self.vgpr(0);
        for reg in base..self.vector_limit {
            let k = self.b().u32(reg - base);
            let hit = self.b().icmp(ICmp::Eq, offset, k);
            let v = self.vgpr(reg);
            acc = self.b().select(hit, v, acc);
        }
        acc
    }

    /// Write VGPR `base + offset` under EXEC and the DPP write mask (and
    /// `enable`); an index past the window writes nothing.
    fn relative_write(&mut self, base: u32, offset: Value, value: Value, dm: &DstMods, enable: Option<Value>) {
        let dm = DstMods { dpp: dm.dpp.clone(), ..DstMods::plain() };
        for reg in base..self.vector_limit {
            let k = self.b().u32(reg - base);
            let mut hit = self.b().icmp(ICmp::Eq, offset, k);
            if let Some(e) = enable {
                hit = self.b().land(hit, e);
            }
            let old = self.vgpr(reg);
            let v = self.b().select(hit, value, old);
            self.write_vgpr_mods(reg, v, &dm);
        }
    }
}
