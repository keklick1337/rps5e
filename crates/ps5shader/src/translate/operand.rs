// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Operand access: register reads and writes, inline constants, source and
//! result modifiers (VOP3 abs/neg/omod/clamp/op_sel, SDWA, DPP16/DPP8,
//! VOP3P), lane masks.

use super::{Result, Translator};
use crate::inline_constants;
use crate::ir_types::*;
use crate::ssa::*;

/// SDWA operand selector: (bit offset, width), `None` for DWORD.
fn sdwa_field(sel: u32) -> Option<(u32, u32)> {
    match sel {
        0..=3 => Some((sel * 8, 8)),
        4 => Some((0, 16)),
        5 => Some((16, 16)),
        _ => None,
    }
}

/// f16 bits of a float inline constant (the 16-bit forms of 240..248).
pub fn f16_inline(encoded: u32) -> Option<u32> {
    Some(match encoded {
        240 => 0x3800,
        241 => 0xB800,
        242 => 0x3C00,
        243 => 0xBC00,
        244 => 0x4000,
        245 => 0xC000,
        246 => 0x4400,
        247 => 0xC400,
        248 => 0x3118,
        _ => return None,
    })
}

/// f64 bits of a float inline constant.
pub fn f64_inline(encoded: u32) -> Option<u64> {
    Some(match encoded {
        240 => 0.5f64.to_bits(),
        241 => (-0.5f64).to_bits(),
        242 => 1.0f64.to_bits(),
        243 => (-1.0f64).to_bits(),
        244 => 2.0f64.to_bits(),
        245 => (-2.0f64).to_bits(),
        246 => 4.0f64.to_bits(),
        247 => (-4.0f64).to_bits(),
        248 => 0x3FC4_5F30_6DC9_C882,
        _ => return None,
    })
}

/// Per-instruction modifiers, normalised over the encodings.
#[derive(Clone, Debug, Default)]
pub struct Mods {
    pub abs: u32,
    pub neg: u32,
    pub omod: u32,
    pub clamp: bool,
    /// VOP3 op_sel (bit i: source i high half; bit 3: destination high half).
    pub op_sel: u32,
    /// VOP3P op_sel_hi / neg_hi.
    pub op_sel_hi: u32,
    pub neg_hi: u32,
    pub sdwa: Option<SdwaControl>,
    pub dpp: Option<DppControl>,
    pub dpp8: Option<Dpp8Control>,
    pub vop3p: bool,
}

impl Mods {
    pub fn of(ins: &Instruction) -> Mods {
        // Packed operations outside VOP3P (v_pk_fmac_f16) take the high
        // halves for the high lane.
        let mut m = Mods { op_sel_hi: 7, ..Mods::default() };
        match &ins.control {
            Some(Control::Vop3(c)) => {
                m.abs = c.absolute_mask;
                m.neg = c.negate_mask;
                m.omod = c.output_modifier;
                m.clamp = c.clamp;
                m.op_sel = c.operand_select;
            }
            Some(Control::Sdwa(c)) => {
                m.abs = c.absolute_mask;
                m.neg = c.negate_mask;
                m.omod = c.output_modifier;
                m.clamp = c.clamp;
                m.sdwa = Some(c.clone());
            }
            Some(Control::Dpp(c)) => {
                m.abs = c.absolute_mask;
                m.neg = c.negate_mask;
                m.dpp = Some(c.clone());
            }
            Some(Control::Dpp8(c)) => m.dpp8 = Some(c.clone()),
            Some(Control::Vop3p(c)) => {
                m.vop3p = true;
                m.op_sel = c.op_sel_mask;
                m.op_sel_hi = c.op_sel_hi_mask;
                m.neg = c.neg_lo_mask;
                m.neg_hi = c.neg_hi_mask;
                m.clamp = c.clamp;
            }
            _ => {}
        }
        m
    }

    pub fn abs(&self, i: usize) -> bool {
        self.abs >> i & 1 != 0
    }

    pub fn neg(&self, i: usize) -> bool {
        self.neg >> i & 1 != 0
    }

    pub fn op_sel(&self, i: usize) -> bool {
        self.op_sel >> i & 1 != 0
    }

    pub fn op_sel_hi(&self, i: usize) -> bool {
        self.op_sel_hi >> i & 1 != 0
    }

    pub fn neg_hi(&self, i: usize) -> bool {
        self.neg_hi >> i & 1 != 0
    }

    /// SDWA selector of source `i`: (offset, width, sign extend).
    fn sdwa_src(&self, i: usize) -> Option<(u32, u32, bool)> {
        let s = self.sdwa.as_ref()?;
        let (sel, sext) = match i {
            0 => (s.source0_select, s.source0_sign_extend),
            1 => (s.source1_select, s.source1_sign_extend),
            _ => return None,
        };
        sdwa_field(sel).map(|(o, w)| (o, w, sext))
    }
}

/// Destination write controls.
#[derive(Clone, Debug, Default)]
pub struct DstMods {
    /// SDWA destination (offset, width, unused mode 0 zero / 1 sext / 2
    /// preserve).
    pub field: Option<(u32, u32, u32)>,
    pub dpp: Option<DppControl>,
    /// SDWA encoding: a 16-bit result with a DWORD destination is written
    /// zero-extended to the whole register.
    pub sdwa: bool,
}

impl DstMods {
    pub fn of(m: &Mods) -> DstMods {
        DstMods {
            field: m.sdwa.as_ref().and_then(|s| sdwa_field(s.destination_select).map(|(o, w)| (o, w, s.destination_unused))),
            dpp: m.dpp.clone(),
            sdwa: m.sdwa.is_some(),
        }
    }

    pub fn plain() -> DstMods {
        DstMods::default()
    }
}

impl<'a> Translator<'a> {
    // ---------------------------------------------------------------
    // Raw registers
    // ---------------------------------------------------------------

    pub fn sgpr(&mut self, n: u32) -> Value {
        match n {
            0..=127 => self.b().get_reg(Reg::Sgpr(n as u8)),
            _ => self.b().u32(0),
        }
    }

    pub fn vgpr(&mut self, n: u32) -> Value {
        self.b().get_reg(Reg::Vgpr((n & 0xFF) as u16))
    }

    /// Write an SGPR from scalar code: invalidates mask bits covering it and
    /// keeps EXEC's bit in sync.
    pub fn write_sgpr(&mut self, n: u32, v: Value) {
        if n >= 128 || n == NULL_SOURCE {
            return;
        }
        if n == EXEC_HI {
            self.set_exec_hi(v);
            self.refresh_exec_bit();
            return;
        }
        let n8 = n as u8;
        self.b().set_reg(Reg::Sgpr(n8), v);
        if n == EXEC_LO {
            self.refresh_exec_bit();
            return;
        }
        let f = self.b().bool(false);
        self.b().set_reg(Reg::SgprBitValid(n8), f);
        if n > 0 {
            self.b().set_reg(Reg::SgprBitValid(n8 - 1), f);
        }
    }

    pub fn write_sgpr_pair(&mut self, n: u32, lo: Value, hi: Value) {
        // A null destination discards both words (its high word is not
        // EXEC_LO).
        if n == NULL_SOURCE {
            return;
        }
        self.write_sgpr(n, lo);
        self.write_sgpr(n + 1, hi);
    }

    /// Raw write of the EXEC_HI word. Wave32 keeps a lane-mask provenance
    /// for EXEC_HI (a 32-bit mask stored there by `s_mov_b32 exec_hi, vcc_lo`);
    /// any other write of the word (the high half of `s_mov_b64 exec`, a
    /// 64-bit saveexec, a VALU EXEC write) replaces it.
    pub fn set_exec_hi(&mut self, hi: Value) {
        self.b().set_reg(Reg::Sgpr(Reg::EXEC + 1), hi);
        let f = self.b().bool(false);
        self.b().set_reg(Reg::SgprBitValid(Reg::EXEC + 1), f);
    }

    /// Recompute this invocation's EXEC bit from the EXEC words.
    pub fn refresh_exec_bit(&mut self) {
        let lo = self.b().get_reg(Reg::Sgpr(Reg::EXEC));
        let hi = if self.wave64() { self.b().get_reg(Reg::Sgpr(Reg::EXEC + 1)) } else { self.b().u32(0) };
        let bit = self.b().inverse_ballot(lo, hi);
        self.b().set_reg(Reg::SgprBit(Reg::EXEC), bit);
    }

    pub fn exec_bit(&mut self) -> Value {
        self.b().get_reg(Reg::SgprBit(Reg::EXEC))
    }

    /// Write EXEC from a per-invocation bit.
    pub fn set_exec_bit(&mut self, bit: Value) {
        let (lo, hi) = self.b().ballot(bit);
        let hi = if self.wave64() { hi } else { self.b().u32(0) };
        self.b().set_reg(Reg::Sgpr(Reg::EXEC), lo);
        self.set_exec_hi(hi);
        self.b().set_reg(Reg::SgprBit(Reg::EXEC), bit);
    }

    /// Write a lane mask (from a per-invocation bit) to the SGPR(s) at `n`:
    /// both words in wave64, one in wave32. A VALU mask write covers the
    /// whole wave: lanes disabled by EXEC get 0 (compares and carry-outs
    /// alike), so the per-invocation bit is masked like the ballot.
    pub fn write_mask(&mut self, n: u32, bit: Value) {
        if n == NULL_SOURCE || n >= 128 {
            return;
        }
        let exec = self.exec_bit();
        let bit = self.b().land(exec, bit);
        if n == EXEC_LO {
            self.set_exec_bit(bit);
            return;
        }
        let (lo, hi) = self.b().ballot(bit);
        let n8 = n as u8;
        self.b().set_reg(Reg::Sgpr(n8), lo);
        if self.wave64() && n < 127 {
            self.b().set_reg(Reg::Sgpr(n8 + 1), hi);
            let f = self.b().bool(false);
            self.b().set_reg(Reg::SgprBitValid(n8 + 1), f);
        }
        self.b().set_reg(Reg::SgprBit(n8), bit);
        let t = self.b().bool(true);
        self.b().set_reg(Reg::SgprBitValid(n8), t);
        if n > 0 {
            let f = self.b().bool(false);
            self.b().set_reg(Reg::SgprBitValid(n8 - 1), f);
        }
    }

    /// Write mask words computed by scalar code together with the per
    /// invocation bit they encode (always valid).
    pub fn write_mask_words(&mut self, n: u32, lo: Value, hi: Option<Value>, bit: Value) {
        if n == NULL_SOURCE || n >= 128 {
            return;
        }
        if n == EXEC_LO {
            self.b().set_reg(Reg::Sgpr(Reg::EXEC), lo);
            if let Some(hi) = hi {
                self.set_exec_hi(hi);
            }
            self.b().set_reg(Reg::SgprBit(Reg::EXEC), bit);
            return;
        }
        let n8 = n as u8;
        self.b().set_reg(Reg::Sgpr(n8), lo);
        if let Some(hi) = hi {
            self.write_sgpr(n + 1, hi);
        }
        self.b().set_reg(Reg::SgprBit(n8), bit);
        let t = self.b().bool(true);
        self.b().set_reg(Reg::SgprBitValid(n8), t);
        if n > 0 {
            let f = self.b().bool(false);
            self.b().set_reg(Reg::SgprBitValid(n8 - 1), f);
        }
    }

    /// This invocation's bit of the lane-mask operand.
    pub fn read_mask(&mut self, o: Operand) -> Value {
        match o.kind {
            OperandKind::ScalarRegister if o.value == EXEC_LO => self.exec_bit(),
            OperandKind::ScalarRegister if o.value < 127 || (o.value == 127 && !self.wave64()) => {
                let n = o.value as u8;
                let lo = self.sgpr(o.value);
                let hi = if self.wave64() { self.sgpr(o.value + 1) } else { self.b().u32(0) };
                let computed = self.b().inverse_ballot(lo, hi);
                let valid = self.b().get_reg(Reg::SgprBitValid(n));
                let bit = self.b().get_reg(Reg::SgprBit(n));
                self.b().select(valid, bit, computed)
            }
            _ => {
                let (lo, hi) = self.read_u64_parts(o);
                let hi = if self.wave64() { hi } else { self.b().u32(0) };
                self.b().inverse_ballot(lo, hi)
            }
        }
    }

    // ---------------------------------------------------------------
    // Raw operand values
    // ---------------------------------------------------------------

    /// 32-bit value of an operand without modifiers (float inline constants
    /// as f32 bits).
    pub fn read_raw(&mut self, o: Operand) -> Value {
        match o.kind {
            OperandKind::ScalarRegister => self.sgpr(o.value),
            OperandKind::VectorRegister => self.vgpr(o.value),
            OperandKind::LiteralConstant => self.b().u32(o.value),
            OperandKind::EncodedConstant => match o.value {
                SCC_SOURCE => {
                    let s = self.b().get_reg(Reg::Scc);
                    self.b().bool_to_u32(s)
                }
                VCCZ_SOURCE | EXECZ_SOURCE => {
                    let base = if o.value == VCCZ_SOURCE { Reg::VCC } else { Reg::EXEC };
                    let z = self.mask_is_zero(base);
                    self.b().bool_to_u32(z)
                }
                v => {
                    let c = inline_constants::decode(v).unwrap_or(0);
                    self.b().u32(c)
                }
            },
        }
    }

    /// Low / high dwords of a 64-bit operand. Integer inline constants sign
    /// extend; float inline constants are f64; literals zero extend.
    pub fn read_u64_parts(&mut self, o: Operand) -> (Value, Value) {
        match o.kind {
            OperandKind::ScalarRegister => {
                let lo = self.sgpr(o.value);
                let hi = self.sgpr(o.value + 1);
                (lo, hi)
            }
            OperandKind::VectorRegister => {
                let lo = self.vgpr(o.value);
                let hi = self.vgpr(o.value + 1);
                (lo, hi)
            }
            OperandKind::LiteralConstant => {
                let lo = self.b().u32(o.value);
                let hi = self.b().u32(0);
                (lo, hi)
            }
            OperandKind::EncodedConstant => {
                if let Some(bits) = f64_inline(o.value) {
                    let lo = self.b().u32(bits as u32);
                    let hi = self.b().u32((bits >> 32) as u32);
                    return (lo, hi);
                }
                if (193..=208).contains(&o.value) {
                    let lo = self.read_raw(o);
                    let hi = self.b().u32(u32::MAX);
                    return (lo, hi);
                }
                if inline_constants::is_aperture(o.value) {
                    let v = inline_constants::decode_aperture64(o.value);
                    let lo = self.b().u32(v as u32);
                    let hi = self.b().u32((v >> 32) as u32);
                    return (lo, hi);
                }
                let lo = self.read_raw(o);
                let hi = self.b().u32(0);
                (lo, hi)
            }
        }
    }

    /// A signed 64-bit operand: like [`Self::read_u64_parts`], but a 32-bit
    /// literal is sign-extended.
    pub fn read_i64_parts(&mut self, o: Operand) -> (Value, Value) {
        if o.kind == OperandKind::LiteralConstant {
            let lo = self.b().u32(o.value);
            let hi = self.b().u32(((o.value as i32) >> 31) as u32);
            return (lo, hi);
        }
        self.read_u64_parts(o)
    }

    pub fn read_i64(&mut self, o: Operand) -> Value {
        let (lo, hi) = self.read_i64_parts(o);
        self.b().join_u64(lo, hi)
    }

    pub fn read_u64(&mut self, o: Operand) -> Value {
        let (lo, hi) = self.read_u64_parts(o);
        self.b().join_u64(lo, hi)
    }

    /// f64 operand bits: literals give the high dword.
    pub fn read_f64_parts(&mut self, o: Operand) -> (Value, Value) {
        if o.kind == OperandKind::LiteralConstant {
            let lo = self.b().u32(0);
            let hi = self.b().u32(o.value);
            return (lo, hi);
        }
        self.read_u64_parts(o)
    }

    // ---------------------------------------------------------------
    // Sources with modifiers
    // ---------------------------------------------------------------

    /// Source `i` as raw 32 bits with DPP (source 0) and SDWA selection.
    pub fn src_bits(&mut self, ins: &Instruction, m: &Mods, i: usize) -> Value {
        let o = ins.sources[i];
        let mut v = self.read_raw(o);
        if i == 0 {
            if let Some(d) = &m.dpp {
                v = self.dpp_read(v, d.control, d.row_mask, d.bank_mask, d.fetch_inactive);
            } else if let Some(d) = &m.dpp8 {
                v = self.dpp8_read(v, d.lane_selectors, d.fetch_inactive);
            }
        }
        self.sdwa_select(m, i, v)
    }

    /// SDWA source selection (BYTE/WORD extract, sign-extended when SEXT)
    /// of source `i`; the value itself without SDWA.
    fn sdwa_select(&mut self, m: &Mods, i: usize, v: Value) -> Value {
        match m.sdwa_src(i) {
            Some((off, width, sext)) => {
                let off = self.b().u32(off);
                let w = self.b().u32(width);
                if sext {
                    self.b().bfe_s(v, off, w)
                } else {
                    self.b().bfe_u(v, off, w)
                }
            }
            None => v,
        }
    }

    /// Source `i` as u32 (integer operations take no abs/neg).
    pub fn src_u32(&mut self, ins: &Instruction, m: &Mods, i: usize) -> Value {
        self.src_bits(ins, m, i)
    }

    /// Source `i` as f32 with abs/neg applied on the bits.
    pub fn src_f32(&mut self, ins: &Instruction, m: &Mods, i: usize) -> Value {
        let mut v = self.src_bits(ins, m, i);
        if m.abs(i) {
            let c = self.b().u32(0x7FFF_FFFF);
            v = self.b().and(v, c);
        }
        if m.neg(i) {
            let c = self.b().u32(0x8000_0000);
            v = self.b().xor(v, c);
        }
        let f = self.b().bitcast(v, Type::F32);
        if self.f32_flush & 1 != 0 {
            self.flush_denorm(f)
        } else {
            f
        }
    }

    /// Raw 32-bit source of a 16-bit operand (no op_sel): a float inline
    /// constant is its f16 encoding (0.5 -> 0x3800), for float and integer
    /// operations alike, and SDWA selects from that encoding.
    pub fn src_f16_word(&mut self, ins: &Instruction, m: &Mods, i: usize) -> Value {
        let o = ins.sources[i];
        if o.kind == OperandKind::EncodedConstant {
            if let Some(h) = f16_inline(o.value) {
                let v = self.b().u32(h);
                return self.sdwa_select(m, i, v);
            }
        }
        self.src_bits(ins, m, i)
    }

    /// 16 bits (in a u32) of half `high` of source `i` before abs/neg.
    pub fn src_half_bits(&mut self, ins: &Instruction, m: &Mods, i: usize, high: bool) -> Value {
        let w = self.src_f16_word(ins, m, i);
        let off = self.b().u32(if high { 16 } else { 0 });
        let sixteen = self.b().u32(16);
        self.b().bfe_u(w, off, sixteen)
    }

    /// f16 bits of source `i` (selected half, abs/neg applied on bits).
    pub fn src_f16_bits(&mut self, ins: &Instruction, m: &Mods, i: usize) -> Value {
        let high = m.op_sel(i) && !m.vop3p;
        let mut v = self.src_half_bits(ins, m, i, high);
        if m.abs(i) {
            let c = self.b().u32(0x7FFF);
            v = self.b().and(v, c);
        }
        if m.neg(i) {
            let c = self.b().u32(0x8000);
            v = self.b().xor(v, c);
        }
        v
    }

    /// Convert 16 f16 bits (in a u32) to f32 exactly.
    pub fn half_to_f32(&mut self, bits: Value) -> Value {
        let h = self.b().uconvert(bits, Type::U16);
        let f = self.b().bitcast(h, Type::F16);
        self.b().fconvert(f, Type::F32, Rounding::Rte)
    }

    /// Source `i` of a 16-bit float operation, widened to f32 (abs/neg in
    /// the f32 domain).
    pub fn src_f16(&mut self, ins: &Instruction, m: &Mods, i: usize) -> Value {
        let high = m.op_sel(i) && !m.vop3p;
        let bits = self.src_half_bits(ins, m, i, high);
        let mut v = self.half_to_f32(bits);
        if m.abs(i) {
            v = self.b().fabs(v);
        }
        if m.neg(i) {
            v = self.b().fneg(v);
        }
        v
    }

    /// Packed (VOP3P) half `lane` of source `i` widened to f32.
    pub fn src_packed_f16(&mut self, ins: &Instruction, m: &Mods, i: usize, high_lane: bool) -> Value {
        let select_high = if high_lane { m.op_sel_hi(i) } else { m.op_sel(i) };
        let bits = self.src_half_bits(ins, m, i, select_high);
        let mut v = self.half_to_f32(bits);
        // VOP3P has a negate per half; a packed operation in another
        // encoding (v_pk_fmac_f16 with DPP) applies its abs/neg to both.
        let (abs, negate) = if m.vop3p {
            (false, if high_lane { m.neg_hi(i) } else { m.neg(i) })
        } else {
            (m.abs(i), m.neg(i))
        };
        if abs {
            v = self.b().fabs(v);
        }
        if negate {
            v = self.b().fneg(v);
        }
        v
    }

    /// Packed (VOP3P) half of source `i` as an integer (sign-extended when
    /// `signed`).
    pub fn src_packed_u16(&mut self, ins: &Instruction, m: &Mods, i: usize, high_lane: bool, signed: bool) -> Value {
        let select_high = if high_lane { m.op_sel_hi(i) } else { m.op_sel(i) };
        let mut bits = self.src_half_bits_int(ins, m, i, select_high);
        // VOP3P neg_lo / neg_hi flip bit 15 of an integer half too (UE5's
        // TSR clamps its -1 offsets with v_pk_max_i16 neg(0x7fff), x).
        if m.vop3p && if high_lane { m.neg_hi(i) } else { m.neg(i) } {
            let k = self.b().u32(0x8000);
            bits = self.b().xor(bits, k);
        }
        if signed {
            let zero = self.b().u32(0);
            let sixteen = self.b().u32(16);
            self.b().bfe_s(bits, zero, sixteen)
        } else {
            bits
        }
    }

    /// 16-bit integer half (integer inline constants keep their value,
    /// float inline constants are f16 encodings).
    pub fn src_half_bits_int(&mut self, ins: &Instruction, m: &Mods, i: usize, high: bool) -> Value {
        let w = self.src_f16_word(ins, m, i);
        let off = self.b().u32(if high { 16 } else { 0 });
        let sixteen = self.b().u32(16);
        self.b().bfe_u(w, off, sixteen)
    }

    /// 16-bit integer source (VOP3 op_sel picks the half).
    pub fn src_u16(&mut self, ins: &Instruction, m: &Mods, i: usize, signed: bool) -> Value {
        let high = m.op_sel(i);
        let v = self.src_half_bits_int(ins, m, i, high);
        if signed {
            let zero = self.b().u32(0);
            let sixteen = self.b().u32(16);
            self.b().bfe_s(v, zero, sixteen)
        } else {
            v
        }
    }

    pub fn src_u64(&mut self, ins: &Instruction, i: usize) -> Value {
        self.read_u64(ins.sources[i])
    }

    /// Signed 64-bit source (a 32-bit literal is sign-extended).
    pub fn src_i64(&mut self, ins: &Instruction, i: usize) -> Value {
        self.read_i64(ins.sources[i])
    }

    /// f64 source with abs/neg on the sign bit.
    pub fn src_f64(&mut self, ins: &Instruction, m: &Mods, i: usize) -> Value {
        let (lo, mut hi) = self.read_f64_parts(ins.sources[i]);
        if m.abs(i) {
            let c = self.b().u32(0x7FFF_FFFF);
            hi = self.b().and(hi, c);
        }
        if m.neg(i) {
            let c = self.b().u32(0x8000_0000);
            hi = self.b().xor(hi, c);
        }
        let u = self.b().join_u64(lo, hi);
        self.b().bitcast(u, Type::F64)
    }

    // ---------------------------------------------------------------
    // DPP
    // ---------------------------------------------------------------

    /// Source lane for a DPP16 control and whether it is valid.
    fn dpp_target(&mut self, control: u32) -> Option<(Value, Value)> {
        let lane = self.b().lane_id();
        let c = |t: &mut Self, x: u32| t.b().u32(x);
        let t_true = self.b().bool(true);
        if control <= 0xFF {
            // quad_perm
            let base_mask = c(self, !3);
            let base = self.b().and(lane, base_mask);
            let three = c(self, 3);
            let within = self.b().and(lane, three);
            let one = c(self, 1);
            let shift = self.b().shl(within, one);
            let ctrl = c(self, control);
            let sel = self.b().lshr(ctrl, shift);
            let sel = self.b().and(sel, three);
            let target = self.b().or(base, sel);
            return Some((target, t_true));
        }
        let row_mask = c(self, 0xFFFF_FFF0);
        let row = self.b().and(lane, row_mask);
        let fifteen = c(self, 15);
        let within = self.b().and(lane, fifteen);
        let amount = control & 0xF;
        match control {
            0x101..=0x10F => {
                // row_shl: read lane + amount
                let a = c(self, amount);
                let shifted = self.b().iadd(within, a);
                let lim = c(self, 16 - amount);
                let valid = self.b().icmp(ICmp::ULt, within, lim);
                let target = self.b().or(row, shifted);
                Some((target, valid))
            }
            0x111..=0x11F => {
                let a = c(self, amount);
                let shifted = self.b().isub(within, a);
                let valid = self.b().icmp(ICmp::UGe, within, a);
                let shifted = self.b().and(shifted, fifteen);
                let target = self.b().or(row, shifted);
                Some((target, valid))
            }
            0x121..=0x12F => {
                let a = c(self, amount);
                let minus = self.b().isub(within, a);
                let b16 = c(self, 16 - amount);
                let plus = self.b().iadd(within, b16);
                let ge = self.b().icmp(ICmp::UGe, within, a);
                let sel = self.b().select(ge, minus, plus);
                let target = self.b().or(row, sel);
                Some((target, t_true))
            }
            0x140 | 0x141 => {
                let half = control == 0x141;
                let (bm, lm) = if half { (!7u32, 7u32) } else { (!15u32, 15u32) };
                let bmv = c(self, bm);
                let base = self.b().and(lane, bmv);
                let lmv = c(self, lm);
                let l = self.b().and(lane, lmv);
                let mirrored = self.b().isub(lmv, l);
                let target = self.b().or(base, mirrored);
                Some((target, t_true))
            }
            0x150..=0x15F => {
                let a = c(self, amount);
                let target = self.b().or(row, a);
                Some((target, t_true))
            }
            0x160..=0x16F => {
                let a = c(self, amount);
                let target = self.b().xor(lane, a);
                Some((target, t_true))
            }
            _ => None,
        }
    }

    /// Whether lane `target` (u32 per invocation) has its EXEC bit set.
    pub fn lane_active(&mut self, target: Value) -> Value {
        if self.options.rewrites & super::rewrites::LANE_ACTIVE_EXEC != 0 {
            // The target's bit of the EXEC words (`target` is below the
            // wave size): the same bit every lane's EXEC bit encodes.
            let lo = self.sgpr(EXEC_LO);
            let word = if self.wave64() {
                let hi = self.sgpr(EXEC_HI);
                let c32 = self.b().u32(32);
                let high = self.b().icmp(ICmp::UGe, target, c32);
                self.b().select(high, hi, lo)
            } else {
                lo
            };
            let c31 = self.b().u32(31);
            let shift = self.b().and(target, c31);
            let bit = self.b().lshr(word, shift);
            let one = self.b().u32(1);
            let bit = self.b().and(bit, one);
            let zero = self.b().u32(0);
            return self.b().icmp(ICmp::Ne, bit, zero);
        }
        let e = self.exec_bit();
        let as_u32 = self.b().bool_to_u32(e);
        let other = self.b().shuffle(as_u32, target);
        let zero = self.b().u32(0);
        self.b().icmp(ICmp::Ne, other, zero)
    }

    pub(crate) fn dpp_read(&mut self, v: Value, control: u32, _row: u32, _bank: u32, fetch_inactive: bool) -> Value {
        let Some((target, valid)) = self.dpp_target(control) else {
            return v;
        };
        let wave_mask = self.wave_mask();
        let target = self.b().and(target, wave_mask);
        let shuffled = self.b().shuffle(v, target);
        let ok = if fetch_inactive {
            valid
        } else {
            let active = self.lane_active(target);
            self.b().land(valid, active)
        };
        let zero = self.b().u32(0);
        self.b().select(ok, shuffled, zero)
    }

    fn dpp8_target(&mut self, selectors: u32) -> Value {
        let lane = self.b().lane_id();
        let base_mask = self.b().u32(!7);
        let base = self.b().and(lane, base_mask);
        let seven = self.b().u32(7);
        let within = self.b().and(lane, seven);
        let three = self.b().u32(3);
        let shift = self.b().imul(within, three);
        let sel_v = self.b().u32(selectors & 0xFF_FFFF);
        let sel = self.b().lshr(sel_v, shift);
        let sel = self.b().and(sel, seven);
        self.b().or(base, sel)
    }

    pub(crate) fn dpp8_read(&mut self, v: Value, selectors: u32, fetch_inactive: bool) -> Value {
        let target = self.dpp8_target(selectors);
        let shuffled = self.b().shuffle(v, target);
        if fetch_inactive {
            return shuffled;
        }
        let active = self.lane_active(target);
        let zero = self.b().u32(0);
        self.b().select(active, shuffled, zero)
    }

    /// Per-invocation DPP16 write condition (without EXEC); `None` when
    /// the write reaches every active lane.
    fn dpp_write_condition(&mut self, d: &DppControl) -> Option<Value> {
        // Full row and bank masks with BOUND_CTRL reach every active lane:
        // the write is a plain EXEC select.
        if d.row_mask == 0xF && d.bank_mask == 0xF && d.bound_control {
            return None;
        }
        let lane = self.b().lane_id();
        let two = self.b().u32(2);
        let four = self.b().u32(4);
        let three = self.b().u32(3);
        let one = self.b().u32(1);
        let bank = self.b().lshr(lane, two);
        let bank = self.b().and(bank, three);
        let row = self.b().lshr(lane, four);
        let row = self.b().and(row, three);
        let bank_bit = self.b().shl(one, bank);
        let row_bit = self.b().shl(one, row);
        let bm = self.b().u32(d.bank_mask);
        let rm = self.b().u32(d.row_mask);
        let bank_hit = self.b().and(bm, bank_bit);
        let row_hit = self.b().and(rm, row_bit);
        let zero = self.b().u32(0);
        let bank_ok = self.b().icmp(ICmp::Ne, bank_hit, zero);
        let row_ok = self.b().icmp(ICmp::Ne, row_hit, zero);
        let mut ok = self.b().land(bank_ok, row_ok);
        if !d.bound_control {
            if let Some((target, valid)) = self.dpp_target(d.control) {
                ok = self.b().land(ok, valid);
                if !d.fetch_inactive {
                    let wave_mask = self.wave_mask();
                    let target = self.b().and(target, wave_mask);
                    let active = self.lane_active(target);
                    ok = self.b().land(ok, active);
                }
            }
        }
        Some(ok)
    }

    // ---------------------------------------------------------------
    // Destinations
    // ---------------------------------------------------------------

    /// Write VGPR `n` with EXEC masking, SDWA destination selection and DPP
    /// write masks.
    pub fn write_vgpr_mods(&mut self, n: u32, mut v: Value, d: &DstMods) {
        if let Some((off, width, unused)) = d.field {
            v = match unused {
                0 => {
                    let zero = self.b().u32(0);
                    let o = self.b().u32(off);
                    let w = self.b().u32(width);
                    self.b().bfi(zero, v, o, w)
                }
                1 => {
                    let zero = self.b().u32(0);
                    let w = self.b().u32(width);
                    let ext = self.b().bfe_s(v, zero, w);
                    let o = self.b().u32(off);
                    self.b().shl(ext, o)
                }
                _ => {
                    let old = self.vgpr(n);
                    let o = self.b().u32(off);
                    let w = self.b().u32(width);
                    self.b().bfi(old, v, o, w)
                }
            };
        }
        let old = self.vgpr(n);
        let mut cond = self.exec_bit();
        if let Some(dpp) = &d.dpp {
            if let Some(w) = self.dpp_write_condition(dpp) {
                cond = self.b().land(cond, w);
            }
        }
        let merged = self.b().select(cond, v, old);
        self.b().set_reg(Reg::Vgpr((n & 0xFF) as u16), merged);
    }

    /// Plain EXEC-masked VGPR write.
    pub fn write_vgpr(&mut self, n: u32, v: Value) {
        self.write_vgpr_mods(n, v, &DstMods::plain());
    }

    /// Write VGPR `n` in every invocation (no EXEC masking).
    pub fn write_vgpr_unmasked(&mut self, n: u32, v: Value) {
        self.b().set_reg(Reg::Vgpr((n & 0xFF) as u16), v);
    }

    /// Write the low or high 16 bits of VGPR `n`, preserving the rest.
    pub fn write_vgpr_half(&mut self, n: u32, bits: Value, high: bool, d: &DstMods) {
        if d.field.is_some() || d.sdwa {
            // SDWA destination selection decides the placement; DWORD
            // writes the zero-extended 16-bit result.
            let m = self.b().u32(0xFFFF);
            let bits = self.b().and(bits, m);
            self.write_vgpr_mods(n, bits, d);
            return;
        }
        let old = self.vgpr(n);
        let o = self.b().u32(if high { 16 } else { 0 });
        let w = self.b().u32(16);
        let merged = self.b().bfi(old, bits, o, w);
        self.write_vgpr_mods(n, merged, d);
    }

    /// Destination register of a vector instruction (first destination).
    pub fn vdst(&self, ins: &Instruction) -> Result<u32> {
        match ins.destinations.first() {
            Some(o) if o.kind == OperandKind::VectorRegister => Ok(o.value),
            _ => self.invalid(ins, "missing vector destination"),
        }
    }

    // ---------------------------------------------------------------
    // f32 result modifiers
    // ---------------------------------------------------------------

    /// Apply output modifier and clamp to an f32 result.
    pub fn f32_result(&mut self, v: Value, m: &Mods) -> Value {
        let mut v = v;
        if self.f32_flush & 2 != 0 {
            v = self.flush_denorm(v);
        }
        if m.omod != 0 && self.float_mode.output_modifier_applies() {
            v = self.apply_omod(v, m.omod);
        }
        if m.clamp {
            v = self.saturate_f32(v);
        }
        v
    }

    /// Output modifier on the bits (x2, x4, /2), flushing denormal results.
    pub(super) fn apply_omod(&mut self, v: Value, omod: u32) -> Value {
        let bits = self.b().bitcast(v, Type::U32);
        let mag_mask = self.b().u32(0x7FFF_FFFF);
        let sign_mask = self.b().u32(0x8000_0000);
        let mag = self.b().and(bits, mag_mask);
        let sign = self.b().and(bits, sign_mask);
        let result = if omod == 3 {
            let lim = self.b().u32(0x0100_0000);
            let small = self.b().icmp(ICmp::ULt, mag, lim);
            let step = self.b().u32(0x0080_0000);
            let dec = self.b().isub(bits, step);
            self.b().select(small, sign, dec)
        } else {
            let step = if omod == 1 { 0x0080_0000 } else { 0x0100_0000 };
            let lim = self.b().u32(0x7F80_0000 - step);
            let fits = self.b().icmp(ICmp::ULt, mag, lim);
            let stepv = self.b().u32(step);
            let inc = self.b().iadd(bits, stepv);
            let inf = self.b().u32(0x7F80_0000);
            let sinf = self.b().or(sign, inf);
            self.b().select(fits, inc, sinf)
        };
        let denorm_lim = self.b().u32(0x0080_0000);
        let denorm = self.b().icmp(ICmp::ULt, mag, denorm_lim);
        let zero = self.b().u32(0);
        let result = self.b().select(denorm, zero, result);
        let max_finite = self.b().u32(0x7F7F_FFFF);
        let special = self.b().icmp(ICmp::UGt, mag, max_finite);
        let result = self.b().select(special, bits, result);
        self.b().bitcast(result, Type::F32)
    }

    /// Clamp to [0, 1]; negatives (including -0) give +0, NaN gives +0
    /// with DX10_CLAMP and stays NaN without.
    pub fn saturate_f32(&mut self, v: Value) -> Value {
        let r = self.saturate_f32_dx10(v);
        if self.float_mode.dx10_clamp {
            return r;
        }
        let nan = self.b().fcmp(FCmp::Uno, v, v);
        self.b().select(nan, v, r)
    }

    fn saturate_f32_dx10(&mut self, v: Value) -> Value {
        let bits = self.b().bitcast(v, Type::U32);
        let one = self.b().u32(1);
        let m1 = self.b().isub(bits, one);
        let lim = self.b().u32(0x7F80_0000);
        let positive = self.b().icmp(ICmp::ULt, m1, lim);
        let one_f = self.b().u32(0x3F80_0000);
        let upper = self.b().umin(bits, one_f);
        let zero = self.b().u32(0);
        let r = self.b().select(positive, upper, zero);
        self.b().bitcast(r, Type::F32)
    }

    /// Write an f32 result to the first destination (VGPR or, for
    /// readlane-like forms, SGPR) with modifiers.
    pub fn write_f32_result(&mut self, ins: &Instruction, m: &Mods, v: Value) -> Result<()> {
        let v = self.f32_result(v, m);
        let bits = self.b().bitcast(v, Type::U32);
        let n = self.vdst(ins)?;
        self.write_vgpr_mods(n, bits, &DstMods::of(m));
        Ok(())
    }

    /// Write a u32 result to the first (vector) destination.
    pub fn write_u32_result(&mut self, ins: &Instruction, m: &Mods, v: Value) -> Result<()> {
        let n = self.vdst(ins)?;
        self.write_vgpr_mods(n, v, &DstMods::of(m));
        Ok(())
    }
}
