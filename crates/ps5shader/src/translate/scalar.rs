// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Scalar ALU (SOP1, SOP2, SOPC, SOPK) and scalar memory.

use super::{Result, Translator};
use crate::ir_types::*;
use crate::opcode::Opcode;
use crate::ssa::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Bit {
    And,
    Or,
    Xor,
}

impl<'a> Translator<'a> {
    fn sdst(&self, ins: &Instruction) -> Result<u32> {
        match ins.destinations.first() {
            Some(o) if o.kind == OperandKind::ScalarRegister => Ok(o.value),
            Some(o) if o.kind == OperandKind::EncodedConstant && o.value == NULL_SOURCE => Ok(NULL_SOURCE),
            _ => self.invalid(ins, "missing scalar destination"),
        }
    }

    /// Destination pair = code address + offset of the next instruction
    /// (S_GETPC_B64, and the return address of S_SWAPPC_B64 / S_CALL_B64).
    pub(super) fn write_next_address(&mut self, ins: &Instruction) -> Result<()> {
        let dst = self.sdst(ins)?;
        let offset = ins.program_offset() + ins.size_bytes() as u64;
        let base = if self.options.fold_code_address {
            let a = self.program.address;
            self.b().u64(a)
        } else {
            self.b().emit(Op::CodeAddress, Type::U64, vec![])
        };
        let off = self.b().u64(offset);
        let pc = self.b().iadd(base, off);
        let (lo, hi) = self.b().split_u64(pc);
        self.write_sgpr_pair(dst, lo, hi);
        Ok(())
    }

    fn set_scc(&mut self, v: Value) {
        self.b().set_reg(Reg::Scc, v);
    }

    fn scc_nonzero(&mut self, v: Value) {
        let zero = self.b().u32(0);
        let c = self.b().icmp(ICmp::Ne, v, zero);
        self.set_scc(c);
    }

    fn scc_nonzero64(&mut self, lo: Value, hi: Value) {
        let w = self.b().or(lo, hi);
        self.scc_nonzero(w);
    }

    fn s_src(&mut self, ins: &Instruction, i: usize) -> Value {
        self.read_raw(ins.sources[i])
    }

    /// Whether a scalar operation of this width operates on lane masks.
    fn mask_width(&self, wide: bool) -> bool {
        wide == self.wave64()
    }

    fn bitop(&mut self, op: Bit, a: Value, b: Value) -> Value {
        match op {
            Bit::And => self.b().and(a, b),
            Bit::Or => self.b().or(a, b),
            Bit::Xor => self.b().xor(a, b),
        }
    }

    /// Binary bitwise SALU op on 32 or 64 bits, with optional inversion of
    /// the second operand and of the result; mask-aware at wave width.
    fn s_bitwise(&mut self, ins: &Instruction, op: Bit, wide: bool, not_rhs: bool, not_result: bool, unary_not: bool) -> Result<()> {
        let dst = self.sdst(ins)?;
        let mask_bits = if self.mask_width(wide) {
            let a = self.read_mask(ins.sources[0]);
            let r = if unary_not {
                self.b().lnot(a)
            } else {
                let b = self.read_mask(ins.sources[1]);
                let b = if not_rhs { self.b().lnot(b) } else { b };
                let r = self.bitop(op, a, b);
                if not_result {
                    self.b().lnot(r)
                } else {
                    r
                }
            };
            Some(r)
        } else {
            None
        };
        let words = if wide { 2 } else { 1 };
        let mut out = Vec::new();
        for k in 0..words {
            let a = if wide {
                let (lo, hi) = self.read_u64_parts(ins.sources[0]);
                if k == 0 {
                    lo
                } else {
                    hi
                }
            } else {
                self.s_src(ins, 0)
            };
            let r = if unary_not {
                self.b().not(a)
            } else {
                let b = if wide {
                    let (lo, hi) = self.read_u64_parts(ins.sources[1]);
                    if k == 0 {
                        lo
                    } else {
                        hi
                    }
                } else {
                    self.s_src(ins, 1)
                };
                let b = if not_rhs { self.b().not(b) } else { b };
                let r = self.bitop(op, a, b);
                if not_result {
                    self.b().not(r)
                } else {
                    r
                }
            };
            out.push(r);
        }
        if wide {
            self.scc_nonzero64(out[0], out[1]);
        } else {
            self.scc_nonzero(out[0]);
        }
        match mask_bits {
            Some(bit) => self.write_mask_words(dst, out[0], out.get(1).copied(), bit),
            None if wide => self.write_sgpr_pair(dst, out[0], out[1]),
            None => self.write_sgpr(dst, out[0]),
        }
        Ok(())
    }

    /// Mask-aware move (s_mov_b32 in wave32, s_mov_b64 in wave64).
    fn s_move(&mut self, ins: &Instruction, wide: bool) -> Result<()> {
        let dst = self.sdst(ins)?;
        if wide {
            let (lo, hi) = self.read_u64_parts(ins.sources[0]);
            if self.mask_width(true) {
                let bit = self.read_mask(ins.sources[0]);
                self.write_mask_words(dst, lo, Some(hi), bit);
            } else {
                self.write_sgpr_pair(dst, lo, hi);
            }
        } else {
            let v = self.s_src(ins, 0);
            if self.mask_width(false) && dst != Reg::M0 as u32 {
                let bit = self.read_mask(ins.sources[0]);
                self.write_mask_words(dst, v, None, bit);
            } else {
                self.write_sgpr(dst, v);
            }
        }
        Ok(())
    }

    fn s_saveexec(&mut self, ins: &Instruction, op: Bit, not_exec: bool, not_src: bool, not_result: bool, wide: bool) -> Result<()> {
        let dst = self.sdst(ins)?;
        // Read the source before writing the destination: the destination
        // may alias it (s_and_saveexec_b64 vcc, vcc).
        let (src_lo, src_hi) = if wide { self.read_u64_parts(ins.sources[0]) } else { (self.s_src(ins, 0), self.b().u32(0)) };
        let src_bit = self.read_mask(ins.sources[0]);
        let old_lo = self.sgpr(EXEC_LO);
        let old_hi = self.sgpr(EXEC_HI);
        let old_bit = self.exec_bit();
        // Destination receives the old EXEC.
        if wide {
            self.write_mask_words(dst, old_lo, Some(old_hi), old_bit);
        } else if self.wave64() {
            self.write_sgpr(dst, old_lo);
        } else {
            self.write_mask_words(dst, old_lo, None, old_bit);
        }
        let combine = |t: &mut Self, e: Value, s: Value, logical: bool| {
            let e = if not_exec {
                if logical {
                    t.b().lnot(e)
                } else {
                    t.b().not(e)
                }
            } else {
                e
            };
            let s = if not_src {
                if logical {
                    t.b().lnot(s)
                } else {
                    t.b().not(s)
                }
            } else {
                s
            };
            let r = t.bitop(op, e, s);
            if not_result {
                if logical {
                    t.b().lnot(r)
                } else {
                    t.b().not(r)
                }
            } else {
                r
            }
        };
        let new_lo = combine(self, old_lo, src_lo, false);
        let new_hi = if wide { combine(self, old_hi, src_hi, false) } else { old_hi };
        self.b().set_reg(Reg::Sgpr(Reg::EXEC), new_lo);
        if wide {
            self.set_exec_hi(new_hi);
        }
        if wide == self.wave64() {
            let bit = combine(self, old_bit, src_bit, true);
            self.b().set_reg(Reg::SgprBit(Reg::EXEC), bit);
        } else {
            self.refresh_exec_bit();
        }
        if wide {
            self.scc_nonzero64(new_lo, new_hi);
        } else {
            self.scc_nonzero(new_lo);
        }
        Ok(())
    }

    fn s_cmp(&mut self, ins: &Instruction, p: ICmp, wide: bool) -> Result<()> {
        let (a, b) = if wide {
            let a = self.read_u64(ins.sources[0]);
            let b = self.read_u64(ins.sources[1]);
            (a, b)
        } else {
            (self.s_src(ins, 0), self.s_src(ins, 1))
        };
        let c = self.b().icmp(p, a, b);
        self.set_scc(c);
        Ok(())
    }

    /// Extract `count` bits at `offset` (count may be 0..=32+).
    fn extract_bits32(&mut self, src: Value, offset: Value, raw_count: Value, signed: bool) -> Value {
        let c32 = self.b().u32(32);
        let count = self.b().umin(raw_count, c32);
        let discard = self.b().isub(c32, count);
        let thirty_one = self.b().u32(31);
        let discard = self.b().and(discard, thirty_one);
        let shifted = if signed { self.b().ashr(src, offset) } else { self.b().lshr(src, offset) };
        let aligned = self.b().shl(shifted, discard);
        let field = if signed { self.b().ashr(aligned, discard) } else { self.b().lshr(aligned, discard) };
        let zero = self.b().u32(0);
        let nz = self.b().icmp(ICmp::Ne, count, zero);
        self.b().select(nz, field, zero)
    }

    /// Mask of the low `count` bits (count 0..=32).
    fn right_mask32(&mut self, count: Value) -> Value {
        let zero = self.b().u32(0);
        let ones = self.b().u32(u32::MAX);
        self.b().bfi(zero, ones, zero, count)
    }

    fn right_mask64(&mut self, count: Value) -> Value {
        let c32 = self.b().u32(32);
        let below = self.b().icmp(ICmp::ULt, count, c32);
        let above = self.b().icmp(ICmp::UGt, count, c32);
        let low_count = self.b().select(below, count, c32);
        let minus = self.b().isub(count, c32);
        let zero = self.b().u32(0);
        let high_count = self.b().select(above, minus, zero);
        let lo = self.right_mask32(low_count);
        let hi = self.right_mask32(high_count);
        self.b().join_u64(lo, hi)
    }

    /// 64-bit shift with a count in 0..=63.
    pub fn shift64(&mut self, op: Op, v: Value, count: Value) -> Value {
        let c63 = self.b().u32(63);
        let count = self.b().and(count, c63);
        let c = self.b().uconvert(count, Type::U64);
        self.b().emit(op, Type::U64, vec![v, c])
    }

    fn sdst_value(&mut self, dst: u32) -> Value {
        if dst == NULL_SOURCE {
            self.b().u32(0)
        } else {
            self.sgpr(dst)
        }
    }

    /// S_SUBVECTOR_LOOP_BEGIN / END: run a wave64 lane mask as two 32-lane
    /// passes, low half first. SDST keeps the half still to run; the branch
    /// condition (read at the end of the block) skips the loop when EXEC is
    /// zero (BEGIN) or repeats it for the saved high half (END).
    fn s_subvector_loop(&mut self, ins: &Instruction, begin: bool) -> Result<()> {
        let dst = self.sdst(ins)?;
        let zero = self.b().u32(0);
        let lo = self.sgpr(EXEC_LO);
        let hi = self.sgpr(EXEC_HI);
        let saved = self.sdst_value(dst);
        if begin {
            let low_active = self.b().icmp(ICmp::Ne, lo, zero);
            let any = self.b().or(lo, hi);
            let skip = self.b().icmp(ICmp::Eq, any, zero);
            self.b().set_reg(Reg::InstructionBranch, skip);
            let half = self.b().select(low_active, hi, lo);
            let d = self.b().select(skip, saved, half);
            self.write_sgpr(dst, d);
            // In ISA order: SDST may itself name an EXEC half.
            let hi_now = self.sgpr(EXEC_HI);
            let h = self.b().select(low_active, zero, hi_now);
            self.write_sgpr(EXEC_HI, h);
        } else {
            let high_active = self.b().icmp(ICmp::Ne, hi, zero);
            let not_high = self.b().lnot(high_active);
            let pending = self.b().icmp(ICmp::Ne, saved, zero);
            let repeat = self.b().land(not_high, pending);
            self.b().set_reg(Reg::InstructionBranch, repeat);
            let h = self.b().select(repeat, saved, hi);
            self.write_sgpr(EXEC_HI, h);
            let d_now = self.sdst_value(dst);
            let d = self.b().select(repeat, lo, d_now);
            self.write_sgpr(dst, d);
            let lo_now = self.sgpr(EXEC_LO);
            let cleared = self.b().select(repeat, zero, lo_now);
            let l = self.b().select(high_active, saved, cleared);
            self.write_sgpr(EXEC_LO, l);
        }
        Ok(())
    }

    pub(crate) fn translate_salu(&mut self, ins: &Instruction) -> Result<()> {
        use Opcode::*;
        match ins.opcode {
            SMovB32 => self.s_move(ins, false),
            SMovB64 => self.s_move(ins, true),
            SMovkI32 => {
                let dst = self.sdst(ins)?;
                let v = self.b().u32((ins.sources[0].value & 0xFFFF) as i16 as i32 as u32);
                self.write_sgpr(dst, v);
                Ok(())
            }
            SNotB32 => self.s_bitwise(ins, Bit::And, false, false, false, true),
            SNotB64 => self.s_bitwise(ins, Bit::And, true, false, false, true),
            SAndB32 => self.s_bitwise(ins, Bit::And, false, false, false, false),
            SAndB64 => self.s_bitwise(ins, Bit::And, true, false, false, false),
            SOrB32 => self.s_bitwise(ins, Bit::Or, false, false, false, false),
            SOrB64 => self.s_bitwise(ins, Bit::Or, true, false, false, false),
            SXorB32 => self.s_bitwise(ins, Bit::Xor, false, false, false, false),
            SXorB64 => self.s_bitwise(ins, Bit::Xor, true, false, false, false),
            SAndn2B32 => self.s_bitwise(ins, Bit::And, false, true, false, false),
            SAndn2B64 => self.s_bitwise(ins, Bit::And, true, true, false, false),
            SOrn2B32 => self.s_bitwise(ins, Bit::Or, false, true, false, false),
            SOrn2B64 => self.s_bitwise(ins, Bit::Or, true, true, false, false),
            SNandB32 => self.s_bitwise(ins, Bit::And, false, false, true, false),
            SNandB64 => self.s_bitwise(ins, Bit::And, true, false, true, false),
            SNorB32 => self.s_bitwise(ins, Bit::Or, false, false, true, false),
            SNorB64 => self.s_bitwise(ins, Bit::Or, true, false, true, false),
            SXnorB32 => self.s_bitwise(ins, Bit::Xor, false, false, true, false),
            SXnorB64 => self.s_bitwise(ins, Bit::Xor, true, false, true, false),
            SAndSaveexecB64 => self.s_saveexec(ins, Bit::And, false, false, false, true),
            SOrSaveexecB64 => self.s_saveexec(ins, Bit::Or, false, false, false, true),
            SXorSaveexecB64 => self.s_saveexec(ins, Bit::Xor, false, false, false, true),
            SAndn2SaveexecB64 => self.s_saveexec(ins, Bit::And, true, false, false, true),
            SOrn2SaveexecB64 => self.s_saveexec(ins, Bit::Or, true, false, false, true),
            SAndn1SaveexecB64 => self.s_saveexec(ins, Bit::And, false, true, false, true),
            SOrn1SaveexecB64 => self.s_saveexec(ins, Bit::Or, false, true, false, true),
            SNandSaveexecB64 => self.s_saveexec(ins, Bit::And, false, false, true, true),
            SNorSaveexecB64 => self.s_saveexec(ins, Bit::Or, false, false, true, true),
            SXnorSaveexecB64 => self.s_saveexec(ins, Bit::Xor, false, false, true, true),
            SAndSaveexecB32 => self.s_saveexec(ins, Bit::And, false, false, false, false),
            SOrSaveexecB32 => self.s_saveexec(ins, Bit::Or, false, false, false, false),
            SXorSaveexecB32 => self.s_saveexec(ins, Bit::Xor, false, false, false, false),
            SAndn2SaveexecB32 => self.s_saveexec(ins, Bit::And, true, false, false, false),
            SOrn2SaveexecB32 => self.s_saveexec(ins, Bit::Or, true, false, false, false),
            SAndn1SaveexecB32 => self.s_saveexec(ins, Bit::And, false, true, false, false),
            SOrn1SaveexecB32 => self.s_saveexec(ins, Bit::Or, false, true, false, false),
            SNandSaveexecB32 => self.s_saveexec(ins, Bit::And, false, false, true, false),
            SNorSaveexecB32 => self.s_saveexec(ins, Bit::Or, false, false, true, false),
            SXnorSaveexecB32 => self.s_saveexec(ins, Bit::Xor, false, false, true, false),
            SWqmB32 | SWqmB64 => {
                let wide = ins.opcode == SWqmB64;
                let dst = self.sdst(ins)?;
                let words = if wide {
                    let (lo, hi) = self.read_u64_parts(ins.sources[0]);
                    vec![lo, hi]
                } else {
                    vec![self.s_src(ins, 0)]
                };
                let mut out = Vec::new();
                for w in words {
                    out.push(self.wqm_word(w));
                }
                if wide {
                    self.scc_nonzero64(out[0], out[1]);
                    if self.mask_width(true) {
                        let hi = out[1];
                        let wave_lo = out[0];
                        let bit = self.b().inverse_ballot(wave_lo, hi);
                        self.write_mask_words(dst, out[0], Some(out[1]), bit);
                    } else {
                        self.write_sgpr_pair(dst, out[0], out[1]);
                    }
                } else {
                    self.scc_nonzero(out[0]);
                    self.write_sgpr(dst, out[0]);
                }
                Ok(())
            }
            SBrevB32 => {
                let dst = self.sdst(ins)?;
                let v = self.s_src(ins, 0);
                let r = self.b().bit_reverse(v);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SBcnt1I32B32 | SBcnt1I32B64 => {
                let dst = self.sdst(ins)?;
                let r = if ins.opcode == SBcnt1I32B64 {
                    let (lo, hi) = self.read_u64_parts(ins.sources[0]);
                    let a = self.b().bit_count(lo);
                    let b = self.b().bit_count(hi);
                    self.b().iadd(a, b)
                } else {
                    let v = self.s_src(ins, 0);
                    self.b().bit_count(v)
                };
                self.scc_nonzero(r);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SFF1I32B32 => {
                let dst = self.sdst(ins)?;
                let v = self.s_src(ins, 0);
                let r = self.b().find_lsb(v);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SFF1I32B64 => {
                let dst = self.sdst(ins)?;
                let (lo, hi) = self.read_u64_parts(ins.sources[0]);
                let r = self.find_lsb64(lo, hi);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SFlbitI32B32 => {
                let dst = self.sdst(ins)?;
                let v = self.s_src(ins, 0);
                let r = self.leading_zeros32(v, false);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SFlbitI32B64 => {
                let dst = self.sdst(ins)?;
                let (lo, hi) = self.read_u64_parts(ins.sources[0]);
                let hz = self.leading_zeros32(hi, false);
                let lz = self.leading_zeros32(lo, false);
                let c32 = self.b().u32(32);
                let lz32 = self.b().iadd(lz, c32);
                let zero = self.b().u32(0);
                let hi_nz = self.b().icmp(ICmp::Ne, hi, zero);
                let lo_nz = self.b().icmp(ICmp::Ne, lo, zero);
                let neg1 = self.b().u32(u32::MAX);
                let low_part = self.b().select(lo_nz, lz32, neg1);
                let r = self.b().select(hi_nz, hz, low_part);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SBitset0B32 | SBitset1B32 => {
                let dst = self.sdst(ins)?;
                let off = self.s_src(ins, 0);
                let m31 = self.b().u32(31);
                let off = self.b().and(off, m31);
                let one = self.b().u32(1);
                let bit = self.b().shl(one, off);
                let old = self.sgpr(dst);
                let r = if ins.opcode == SBitset1B32 {
                    self.b().or(old, bit)
                } else {
                    let nb = self.b().not(bit);
                    self.b().and(old, nb)
                };
                self.write_sgpr(dst, r);
                Ok(())
            }
            SBitset0B64 | SBitset1B64 => {
                let dst = self.sdst(ins)?;
                let off = self.s_src(ins, 0);
                let m63 = self.b().u32(63);
                let off = self.b().and(off, m63);
                let one = self.b().u64(1);
                let bit = self.shift64(Op::Shl, one, off);
                let (lo, hi) = (self.sgpr(dst), self.sgpr(dst + 1));
                let old = self.b().join_u64(lo, hi);
                let r = if ins.opcode == SBitset1B64 {
                    self.b().or(old, bit)
                } else {
                    let nb = self.b().not(bit);
                    self.b().and(old, nb)
                };
                let (rl, rh) = self.b().split_u64(r);
                self.write_sgpr_pair(dst, rl, rh);
                Ok(())
            }
            // S_SWAPPC_B64 and S_CALL_B64 write the same address to their
            // link register; the control flow graph makes the transfer.
            SGetpcB64 => self.write_next_address(ins),
            SSwappcB64 if crate::cfg::calls::is_call(ins) => self.write_next_address(ins),
            SSetpcB64 => Ok(()),
            SQuadmaskB32 | SQuadmaskB64 => {
                let dst = self.sdst(ins)?;
                if ins.opcode == SQuadmaskB32 {
                    let v = self.s_src(ins, 0);
                    let q = self.quad_compact(v);
                    self.scc_nonzero(q);
                    self.write_sgpr(dst, q);
                } else {
                    let (lo, hi) = self.read_u64_parts(ins.sources[0]);
                    let ql = self.quad_compact(lo);
                    let qh = self.quad_compact(hi);
                    let eight = self.b().u32(8);
                    let qh = self.b().shl(qh, eight);
                    let q = self.b().or(ql, qh);
                    let zero = self.b().u32(0);
                    self.scc_nonzero(q);
                    self.write_sgpr_pair(dst, q, zero);
                }
                Ok(())
            }
            SAbsI32 => {
                let dst = self.sdst(ins)?;
                let v = self.s_src(ins, 0);
                let r = self.iabs(v);
                self.scc_nonzero(r);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SBitreplicateB64B32 => {
                let dst = self.sdst(ins)?;
                let v = self.s_src(ins, 0);
                let ffff = self.b().u32(0xFFFF);
                let lo = self.b().and(v, ffff);
                let sixteen = self.b().u32(16);
                let hi = self.b().lshr(v, sixteen);
                let rl = self.replicate_bits(lo);
                let rh = self.replicate_bits(hi);
                self.write_sgpr_pair(dst, rl, rh);
                Ok(())
            }
            // ---- SOP2 arithmetic ----
            SAddU32 | SAddcU32 => {
                let dst = self.sdst(ins)?;
                let a = self.s_src(ins, 0);
                let b = self.s_src(ins, 1);
                let (sum, carry) = if ins.opcode == SAddcU32 {
                    let cin = self.b().get_reg(Reg::Scc);
                    self.add_with_carry(a, b, Some(cin))
                } else {
                    self.add_with_carry(a, b, None)
                };
                self.set_scc(carry);
                self.write_sgpr(dst, sum);
                Ok(())
            }
            SSubU32 | SSubbU32 => {
                let dst = self.sdst(ins)?;
                let a = self.s_src(ins, 0);
                let b = self.s_src(ins, 1);
                let bin = if ins.opcode == SSubbU32 { Some(self.b().get_reg(Reg::Scc)) } else { None };
                let (diff, borrow) = self.sub_with_borrow(a, b, bin);
                self.set_scc(borrow);
                self.write_sgpr(dst, diff);
                Ok(())
            }
            SAddI32 | SSubI32 => {
                let dst = self.sdst(ins)?;
                let a = self.s_src(ins, 0);
                let b = self.s_src(ins, 1);
                let sub = ins.opcode == SSubI32;
                let r = if sub { self.b().isub(a, b) } else { self.b().iadd(a, b) };
                let check = if sub {
                    let x1 = self.b().xor(a, b);
                    let x2 = self.b().xor(a, r);
                    self.b().and(x1, x2)
                } else {
                    let x1 = self.b().xor(a, r);
                    let x2 = self.b().xor(b, r);
                    self.b().and(x1, x2)
                };
                let sign = self.b().u32(0x8000_0000);
                let s = self.b().and(check, sign);
                self.scc_nonzero(s);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SAddkI32 => {
                let dst = self.sdst(ins)?;
                let a = self.sgpr(dst);
                let b = self.b().u32((ins.sources[0].value & 0xFFFF) as i16 as i32 as u32);
                let r = self.b().iadd(a, b);
                let x1 = self.b().xor(a, r);
                let x2 = self.b().xor(b, r);
                let check = self.b().and(x1, x2);
                let sign = self.b().u32(0x8000_0000);
                let s = self.b().and(check, sign);
                self.scc_nonzero(s);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SSubvectorLoopBegin | SSubvectorLoopEnd => self.s_subvector_loop(ins, ins.opcode == SSubvectorLoopBegin),
            SMulkI32 => {
                let dst = self.sdst(ins)?;
                let a = self.sgpr(dst);
                let b = self.b().u32((ins.sources[0].value & 0xFFFF) as i16 as i32 as u32);
                let r = self.b().imul(a, b);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SMinI32 | SMaxI32 | SMinU32 | SMaxU32 => {
                let dst = self.sdst(ins)?;
                let a = self.s_src(ins, 0);
                let b = self.s_src(ins, 1);
                let (op, p) = match ins.opcode {
                    SMinI32 => (Op::SMin, ICmp::SLt),
                    SMaxI32 => (Op::SMax, ICmp::SGt),
                    SMinU32 => (Op::UMin, ICmp::ULt),
                    _ => (Op::UMax, ICmp::UGt),
                };
                let r = self.b().emit(op, Type::U32, vec![a, b]);
                let c = self.b().icmp(p, a, b);
                self.set_scc(c);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SCselectB32 => {
                let dst = self.sdst(ins)?;
                let a = self.s_src(ins, 0);
                let b = self.s_src(ins, 1);
                let scc = self.b().get_reg(Reg::Scc);
                let r = self.b().select(scc, a, b);
                if self.mask_width(false) {
                    let ma = self.read_mask(ins.sources[0]);
                    let mb = self.read_mask(ins.sources[1]);
                    let bit = self.b().select(scc, ma, mb);
                    self.write_mask_words(dst, r, None, bit);
                } else {
                    self.write_sgpr(dst, r);
                }
                Ok(())
            }
            SCselectB64 => {
                let dst = self.sdst(ins)?;
                let (al, ah) = self.read_u64_parts(ins.sources[0]);
                let (bl, bh) = self.read_u64_parts(ins.sources[1]);
                let scc = self.b().get_reg(Reg::Scc);
                let lo = self.b().select(scc, al, bl);
                let hi = self.b().select(scc, ah, bh);
                if self.mask_width(true) {
                    let ma = self.read_mask(ins.sources[0]);
                    let mb = self.read_mask(ins.sources[1]);
                    let bit = self.b().select(scc, ma, mb);
                    self.write_mask_words(dst, lo, Some(hi), bit);
                } else {
                    self.write_sgpr_pair(dst, lo, hi);
                }
                Ok(())
            }
            SLshlB32 | SLshrB32 | SAshrI32 => {
                let dst = self.sdst(ins)?;
                let a = self.s_src(ins, 0);
                let b = self.s_src(ins, 1);
                let m31 = self.b().u32(31);
                let b = self.b().and(b, m31);
                let op = match ins.opcode {
                    SLshlB32 => Op::Shl,
                    SLshrB32 => Op::LShr,
                    _ => Op::AShr,
                };
                let r = self.b().emit(op, Type::U32, vec![a, b]);
                self.scc_nonzero(r);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SLshlB64 | SLshrB64 | SAshrI64 => {
                let dst = self.sdst(ins)?;
                let a = if ins.opcode == SAshrI64 { self.read_i64(ins.sources[0]) } else { self.read_u64(ins.sources[0]) };
                let b = self.s_src(ins, 1);
                let op = match ins.opcode {
                    SLshlB64 => Op::Shl,
                    SLshrB64 => Op::LShr,
                    _ => Op::AShr,
                };
                let r = self.shift64(op, a, b);
                let (lo, hi) = self.b().split_u64(r);
                self.scc_nonzero64(lo, hi);
                self.write_sgpr_pair(dst, lo, hi);
                Ok(())
            }
            SBfmB32 => {
                let dst = self.sdst(ins)?;
                let count = self.s_src(ins, 0);
                let off = self.s_src(ins, 1);
                let m31 = self.b().u32(31);
                let count = self.b().and(count, m31);
                let off = self.b().and(off, m31);
                let r = self.bfm32(count, off);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SBfmB64 => {
                let dst = self.sdst(ins)?;
                let count = self.s_src(ins, 0);
                let off = self.s_src(ins, 1);
                let m63 = self.b().u32(63);
                let count = self.b().and(count, m63);
                let mask = self.right_mask64(count);
                let r = self.shift64(Op::Shl, mask, off);
                let (lo, hi) = self.b().split_u64(r);
                self.write_sgpr_pair(dst, lo, hi);
                Ok(())
            }
            SMulI32 | SMulHiU32 | SMulHiI32 => {
                let dst = self.sdst(ins)?;
                let a = self.s_src(ins, 0);
                let b = self.s_src(ins, 1);
                let op = match ins.opcode {
                    SMulI32 => Op::IMul,
                    SMulHiU32 => Op::UMulHi,
                    _ => Op::SMulHi,
                };
                let r = self.b().emit(op, Type::U32, vec![a, b]);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SBfeU32 | SBfeI32 => {
                let dst = self.sdst(ins)?;
                let src = self.s_src(ins, 0);
                let field = self.s_src(ins, 1);
                let zero = self.b().u32(0);
                let five = self.b().u32(5);
                let off = self.b().bfe_u(field, zero, five);
                let sixteen = self.b().u32(16);
                let seven = self.b().u32(7);
                let count = self.b().bfe_u(field, sixteen, seven);
                let r = self.extract_bits32(src, off, count, ins.opcode == SBfeI32);
                self.scc_nonzero(r);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SBfeU64 | SBfeI64 => {
                let dst = self.sdst(ins)?;
                let src = if ins.opcode == SBfeI64 { self.read_i64(ins.sources[0]) } else { self.read_u64(ins.sources[0]) };
                let field = self.s_src(ins, 1);
                let zero = self.b().u32(0);
                let six = self.b().u32(6);
                let off = self.b().bfe_u(field, zero, six);
                let sixteen = self.b().u32(16);
                let seven = self.b().u32(7);
                let raw = self.b().bfe_u(field, sixteen, seven);
                let c64 = self.b().u32(64);
                let avail = self.b().isub(c64, off);
                let count = self.b().umin(raw, avail);
                let shifted = self.shift64(Op::LShr, src, off);
                let r = if ins.opcode == SBfeI64 {
                    let disc = self.b().isub(c64, count);
                    let aligned = self.shift64(Op::Shl, shifted, disc);
                    let ext = self.shift64(Op::AShr, aligned, disc);
                    let nz = self.b().icmp(ICmp::Ne, count, zero);
                    let z64 = self.b().u64(0);
                    self.b().select(nz, ext, z64)
                } else {
                    let mask = self.right_mask64(count);
                    self.b().and(shifted, mask)
                };
                let (lo, hi) = self.b().split_u64(r);
                self.scc_nonzero64(lo, hi);
                self.write_sgpr_pair(dst, lo, hi);
                Ok(())
            }
            SAbsdiffI32 => {
                let dst = self.sdst(ins)?;
                let a = self.s_src(ins, 0);
                let b = self.s_src(ins, 1);
                let d = self.b().isub(a, b);
                let r = self.iabs(d);
                self.scc_nonzero(r);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SLshl1AddU32 | SLshl2AddU32 | SLshl3AddU32 | SLshl4AddU32 => {
                let dst = self.sdst(ins)?;
                let amount = match ins.opcode {
                    SLshl1AddU32 => 1,
                    SLshl2AddU32 => 2,
                    SLshl3AddU32 => 3,
                    _ => 4,
                };
                let a = self.s_src(ins, 0);
                let b = self.s_src(ins, 1);
                let sh = self.b().u32(amount);
                let shifted = self.b().shl(a, sh);
                let r = self.b().iadd(shifted, b);
                let back = self.b().u32(32 - amount);
                let out = self.b().lshr(a, back);
                let zero = self.b().u32(0);
                let lost = self.b().icmp(ICmp::Ne, out, zero);
                let carry = self.b().icmp(ICmp::ULt, r, shifted);
                let c = self.b().lor(lost, carry);
                self.set_scc(c);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SPackLlB32B16 | SPackLhB32B16 | SPackHhB32B16 => {
                let dst = self.sdst(ins)?;
                let a = self.s_src(ins, 0);
                let b = self.s_src(ins, 1);
                let sixteen = self.b().u32(16);
                let lo = if ins.opcode == SPackHhB32B16 { self.b().lshr(a, sixteen) } else { a };
                let hi = if ins.opcode == SPackLlB32B16 { b } else { self.b().lshr(b, sixteen) };
                let r = self.pack_u16(lo, hi);
                self.write_sgpr(dst, r);
                Ok(())
            }
            SAddF32 | SSubF32 => {
                let dst = self.sdst(ins)?;
                let a = self.s_src(ins, 0);
                let b = self.s_src(ins, 1);
                let fa = self.b().bitcast(a, Type::F32);
                let fb = self.b().bitcast(b, Type::F32);
                let r = if ins.opcode == SAddF32 { self.b().fadd(fa, fb) } else { self.b().fsub(fa, fb) };
                let r = self.b().bitcast(r, Type::U32);
                self.write_sgpr(dst, r);
                Ok(())
            }
            // ---- SOPC ----
            SCmpEqI32 | SCmpEqU32 => self.s_cmp(ins, ICmp::Eq, false),
            SCmpLgI32 | SCmpLgU32 => self.s_cmp(ins, ICmp::Ne, false),
            SCmpGtI32 => self.s_cmp(ins, ICmp::SGt, false),
            SCmpGeI32 => self.s_cmp(ins, ICmp::SGe, false),
            SCmpLtI32 => self.s_cmp(ins, ICmp::SLt, false),
            SCmpLeI32 => self.s_cmp(ins, ICmp::SLe, false),
            SCmpGtU32 => self.s_cmp(ins, ICmp::UGt, false),
            SCmpGeU32 => self.s_cmp(ins, ICmp::UGe, false),
            SCmpLtU32 => self.s_cmp(ins, ICmp::ULt, false),
            SCmpLeU32 => self.s_cmp(ins, ICmp::ULe, false),
            SCmpEqU64 => self.s_cmp(ins, ICmp::Eq, true),
            SCmpLgU64 => self.s_cmp(ins, ICmp::Ne, true),
            SBitcmp0B32 | SBitcmp1B32 => {
                let v = self.s_src(ins, 0);
                let off = self.s_src(ins, 1);
                let m31 = self.b().u32(31);
                let off = self.b().and(off, m31);
                let one = self.b().u32(1);
                let bit = self.b().bfe_u(v, off, one);
                let want = self.b().u32((ins.opcode == SBitcmp1B32) as u32);
                let c = self.b().icmp(ICmp::Eq, bit, want);
                self.set_scc(c);
                Ok(())
            }
            SBitcmp0B64 | SBitcmp1B64 => {
                let v = self.read_u64(ins.sources[0]);
                let off = self.s_src(ins, 1);
                let shifted = self.shift64(Op::LShr, v, off);
                let one = self.b().u64(1);
                let bit = self.b().and(shifted, one);
                let want = self.b().u64((ins.opcode == SBitcmp1B64) as u64);
                let c = self.b().icmp(ICmp::Eq, bit, want);
                self.set_scc(c);
                Ok(())
            }
            // ---- SOPK compares ----
            SCmpkEqI32 | SCmpkLgI32 | SCmpkGtI32 | SCmpkGeI32 | SCmpkLtI32 | SCmpkLeI32 | SCmpkEqU32 | SCmpkLgU32
            | SCmpkGtU32 | SCmpkGeU32 | SCmpkLtU32 | SCmpkLeU32 => {
                let reg = self.sdst(ins)?;
                let a = self.sgpr(reg);
                let imm = ins.sources[0].value & 0xFFFF;
                let signed = matches!(ins.opcode, SCmpkEqI32 | SCmpkLgI32 | SCmpkGtI32 | SCmpkGeI32 | SCmpkLtI32 | SCmpkLeI32);
                let b = self.b().u32(if signed { imm as i16 as i32 as u32 } else { imm });
                let p = match ins.opcode {
                    SCmpkEqI32 | SCmpkEqU32 => ICmp::Eq,
                    SCmpkLgI32 | SCmpkLgU32 => ICmp::Ne,
                    SCmpkGtI32 => ICmp::SGt,
                    SCmpkGeI32 => ICmp::SGe,
                    SCmpkLtI32 => ICmp::SLt,
                    SCmpkLeI32 => ICmp::SLe,
                    SCmpkGtU32 => ICmp::UGt,
                    SCmpkGeU32 => ICmp::UGe,
                    SCmpkLtU32 => ICmp::ULt,
                    _ => ICmp::ULe,
                };
                let c = self.b().icmp(p, a, b);
                self.set_scc(c);
                Ok(())
            }
            // A MODE write from an SGPR makes the written round bits unknown
            // (`rounding`); other MODE fields are refused when strict, other
            // hardware registers ignored.
            SSetregB32 if ins.sources[1].value & 0x3F == 1 => self.write_rounding(ins),
            SSetregB32 | SWaitcnt => Ok(()),
            SSwappcB64 => Ok(()),
            _ => {
                if self.translate_extra_salu(ins)? {
                    return Ok(());
                }
                self.unsupported(ins, "scalar ALU")
            }
        }
    }

    // ---- shared integer helpers ----

    pub fn iabs(&mut self, v: Value) -> Value {
        let n = self.b().ineg(v);
        let zero = self.b().u32(0);
        let neg = self.b().icmp(ICmp::SLt, v, zero);
        self.b().select(neg, n, v)
    }

    /// `(sum, carry)` of `a + b (+ carry_in)`.
    pub fn add_with_carry(&mut self, a: Value, b: Value, carry_in: Option<Value>) -> (Value, Value) {
        let first = self.b().emit(Op::IAddCarry, Type::U32X2, vec![a, b]);
        let sum = self.b().extract(first, 0);
        let c1 = self.b().extract(first, 1);
        let zero = self.b().u32(0);
        let carry1 = self.b().icmp(ICmp::Ne, c1, zero);
        let Some(cin) = carry_in else { return (sum, carry1) };
        let cin = self.b().bool_to_u32(cin);
        let second = self.b().emit(Op::IAddCarry, Type::U32X2, vec![sum, cin]);
        let sum2 = self.b().extract(second, 0);
        let c2 = self.b().extract(second, 1);
        let carry2 = self.b().icmp(ICmp::Ne, c2, zero);
        let carry = self.b().lor(carry1, carry2);
        (sum2, carry)
    }

    /// `(difference, borrow)` of `a - b (- borrow_in)`.
    pub fn sub_with_borrow(&mut self, a: Value, b: Value, borrow_in: Option<Value>) -> (Value, Value) {
        let d = self.b().isub(a, b);
        let borrow1 = self.b().icmp(ICmp::ULt, a, b);
        let Some(bin) = borrow_in else { return (d, borrow1) };
        let bin = self.b().bool_to_u32(bin);
        let d2 = self.b().isub(d, bin);
        let borrow2 = self.b().icmp(ICmp::ULt, d, bin);
        let borrow = self.b().lor(borrow1, borrow2);
        (d2, borrow)
    }

    pub fn find_lsb64(&mut self, lo: Value, hi: Value) -> Value {
        let llsb = self.b().find_lsb(lo);
        let hlsb = self.b().find_lsb(hi);
        let c32 = self.b().u32(32);
        let hpos = self.b().iadd(hlsb, c32);
        let zero = self.b().u32(0);
        let lnz = self.b().icmp(ICmp::Ne, lo, zero);
        let hnz = self.b().icmp(ICmp::Ne, hi, zero);
        let neg1 = self.b().u32(u32::MAX);
        let hsel = self.b().select(hnz, hpos, neg1);
        self.b().select(lnz, llsb, hsel)
    }

    /// Leading zero count (or, signed, count of bits equal to the sign
    /// below it); -1 when no bit differs.
    pub fn leading_zeros32(&mut self, v: Value, signed: bool) -> Value {
        let x = if signed {
            let c31 = self.b().u32(31);
            let s = self.b().ashr(v, c31);
            self.b().xor(v, s)
        } else {
            v
        };
        let msb = self.b().find_umsb(x);
        let c31 = self.b().u32(31);
        let pos = self.b().isub(c31, msb);
        let zero = self.b().u32(0);
        let nz = self.b().icmp(ICmp::Ne, x, zero);
        let neg1 = self.b().u32(u32::MAX);
        self.b().select(nz, pos, neg1)
    }

    pub fn bfm32(&mut self, count: Value, offset: Value) -> Value {
        let zero = self.b().u32(0);
        let ones = self.b().u32(u32::MAX);
        // count + offset may exceed 32: the hardware mask is
        // ((1 << count) - 1) << offset truncated to 32 bits.
        let one = self.b().u32(1);
        let c32 = self.b().u32(32);
        let all = self.b().icmp(ICmp::UGe, count, c32);
        let m31 = self.b().u32(31);
        let cm = self.b().and(count, m31);
        let p = self.b().shl(one, cm);
        let mask = self.b().isub(p, one);
        let mask = self.b().select(all, ones, mask);
        let _ = zero;
        self.b().shl(mask, offset)
    }

    pub fn pack_u16(&mut self, lo: Value, hi: Value) -> Value {
        let ffff = self.b().u32(0xFFFF);
        let l = self.b().and(lo, ffff);
        let h = self.b().and(hi, ffff);
        let sixteen = self.b().u32(16);
        let h = self.b().shl(h, sixteen);
        self.b().or(l, h)
    }

    fn wqm_word(&mut self, w: Value) -> Value {
        let one = self.b().u32(1);
        let two = self.b().u32(2);
        let s1 = self.b().lshr(w, one);
        let m1 = self.b().or(w, s1);
        let s2 = self.b().lshr(m1, two);
        let m2 = self.b().or(m1, s2);
        let quad = self.b().u32(0x1111_1111);
        let q = self.b().and(m2, quad);
        let f = self.b().u32(0xF);
        self.b().imul(q, f)
    }

    fn quad_compact(&mut self, v: Value) -> Value {
        let c = |t: &mut Self, x: u32| t.b().u32(x);
        let one = c(self, 1);
        let s = self.b().lshr(v, one);
        let mut bits = self.b().or(v, s);
        let two = c(self, 2);
        let s = self.b().lshr(bits, two);
        bits = self.b().or(bits, s);
        let m = c(self, 0x1111_1111);
        bits = self.b().and(bits, m);
        for (shift, mask) in [(3u32, 0x0303_0303u32), (6, 0x000F_000F), (12, 0xFF)] {
            let sh = c(self, shift);
            let s = self.b().lshr(bits, sh);
            let o = self.b().or(bits, s);
            let mk = c(self, mask);
            bits = self.b().and(o, mk);
        }
        bits
    }

    fn replicate_bits(&mut self, v: Value) -> Value {
        let c = |t: &mut Self, x: u32| t.b().u32(x);
        let mut bits = v;
        for (shift, mask) in [(8u32, 0x00FF_00FFu32), (4, 0x0F0F_0F0F), (2, 0x3333_3333), (1, 0x5555_5555)] {
            let sh = c(self, shift);
            let s = self.b().shl(bits, sh);
            let o = self.b().or(bits, s);
            let mk = c(self, mask);
            bits = self.b().and(o, mk);
        }
        let one = c(self, 1);
        let s = self.b().shl(bits, one);
        self.b().or(bits, s)
    }

    // ---- scalar memory ----

    /// SMEM operations without a memory access: clocks, the wave index,
    /// cache control and address-translation probes (no-ops here).
    pub(crate) fn translate_smem_misc(&mut self, ins: &Instruction) -> Result<()> {
        use Opcode::*;
        let dst = ins.destinations.first().map_or(0, |d| d.value);
        match ins.opcode {
            SMemtime | SMemrealtime => {
                let clock = self.b().emit(Op::Clock { realtime: ins.opcode == SMemrealtime }, Type::U64, vec![]);
                let (lo, hi) = self.b().split_u64(clock);
                self.write_sgpr(dst, lo);
                self.write_sgpr(dst + 1, hi);
                Ok(())
            }
            SGetWaveidInWorkgroup => {
                let id = self.b().emit(Op::SystemValue(SystemValue::SubgroupId), Type::U32, vec![]);
                self.write_sgpr(dst, id);
                Ok(())
            }
            SGl1Inv | SDcacheInv | SDcacheWb | SAtcProbe | SAtcProbeBuffer | SDcacheDiscard | SDcacheDiscardX2 => Ok(()),
            _ => self.invalid(ins, "scalar memory without control"),
        }
    }

    pub(crate) fn translate_smem(&mut self, ins: &Instruction) -> Result<()> {
        use Opcode::*;
        if matches!(ins.opcode, SScratchLoadDword | SScratchLoadDwordx2 | SScratchLoadDwordx4) {
            return self.unsupported(ins, "scalar scratch load (the wave's scratch layout is not modelled)");
        }
        let Some(Control::ScalarMemory(c)) = &ins.control else {
            if matches!(ins.opcode, ScratchStoreDword | ScratchStoreDwordx2 | ScratchStoreDwordx4) {
                return self.unsupported(ins, "scalar scratch store");
            }
            return self.invalid(ins, "scalar memory without control");
        };
        // Every address component (base, immediate OFFSET, SOFFSET) is
        // dword-aligned on its own before they are added: the scalar cache
        // adds dword addresses (measured for the base of s_buffer_load on
        // RDNA2; the offsets follow the same rule).
        let base = ins.sources[0].value;
        let immediate = c.immediate_offset_bytes as u32 & !3;
        let mut offset = self.b().u32(immediate);
        let mut dynamic_offset = None;
        if let Some(src) = ins.sources.get(1) {
            let dynamic = match src.kind {
                OperandKind::ScalarRegister => Some(self.sgpr(src.value)),
                OperandKind::LiteralConstant => Some(self.b().u32(src.value)),
                _ => None,
            };
            if let Some(d) = dynamic {
                let m3 = self.b().u32(!3);
                let d = self.b().and(d, m3);
                offset = self.b().iadd(offset, d);
                dynamic_offset = Some(d);
            }
        }
        let count = c.destination_count;
        let dst = ins.destinations.first().map_or(0, |d| d.value);
        let buffer = matches!(ins.opcode, SBufferLoadDword | SBufferLoadDwordx2 | SBufferLoadDwordx4 | SBufferLoadDwordx8 | SBufferLoadDwordx16);
        let mut values = Vec::with_capacity(count as usize);
        if buffer {
            let parts: Vec<Value> = (0..4).map(|k| self.sgpr(base + k)).collect();
            let desc = self.b().construct(&parts);
            let m3 = self.b().u32(!3);
            let aligned = self.b().and(offset, m3);
            for k in 0..count {
                let off = self.b().u32(4 * k);
                let mut o = self.b().iadd(aligned, off);
                // Dword k reads `soffset + (imm + 4k)`; a sum past 2^32 is
                // out of range (unaligned u32::MAX tells the load).
                if let Some(d) = dynamic_offset {
                    let constant = self.b().u32(immediate.wrapping_add(4 * k));
                    let sum = self.b().iadd(d, constant);
                    let carry = self.b().icmp(ICmp::ULt, sum, d);
                    let invalid = self.b().u32(u32::MAX);
                    o = self.b().select(carry, invalid, o);
                }
                values.push(self.b().emit(Op::ScalarBufferLoad { pc: ins.pc, dword: k as u8 }, Type::U32, vec![desc, o]));
            }
        } else {
            // The base is a 48-bit address (s[base+1][15:0]); the
            // sign-extended 21-bit immediate and the zero-extended SOFFSET
            // are added to it (a negative immediate reaches below the
            // base), each dword-aligned.
            let lo = self.sgpr(base);
            let hi = self.sgpr(base + 1);
            let ffff = self.b().u32(0xFFFF);
            let hi = self.b().and(hi, ffff);
            let base_addr = self.b().join_u64(lo, hi);
            let m3 = self.b().u64(!3);
            let base_addr = self.b().and(base_addr, m3);
            let imm64 = self.b().u64(c.immediate_offset_bytes as i64 as u64 & !3);
            let off = match dynamic_offset {
                Some(d) => {
                    let d64 = self.b().uconvert(d, Type::U64);
                    self.b().iadd(imm64, d64)
                }
                None => imm64,
            };
            let addr = self.b().iadd(base_addr, off);
            for k in 0..count {
                let off = self.b().u64(4 * k as u64);
                let a = self.b().iadd(addr, off);
                values.push(self.b().emit(Op::ScalarLoad { pc: ins.pc, dword: k as u8 }, Type::U32, vec![a]));
            }
        }
        for (k, v) in values.into_iter().enumerate() {
            self.write_sgpr(dst + k as u32, v);
        }
        Ok(())
    }
}
