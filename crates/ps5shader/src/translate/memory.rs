// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Memory instructions (LDS, buffer, global/flat/scratch, image), exports
//! and attribute interpolation.

use super::operand::Mods;
use super::{Result, Translator};
use crate::ir_types::*;
use crate::opcode::{FlatOp, FlatSegment, Opcode};
use crate::ssa::*;

impl<'a> Translator<'a> {
    fn exec_operand(&mut self) -> Value {
        self.exec_bit()
    }

    /// `image_sample*_cd*` (coarse derivatives): every lane of a quad takes
    /// the derivative dwords of the quad's first active lane (measured; a
    /// quad without active lanes keeps its own).
    fn coarse_derivatives(&mut self, address: &mut [Value], f: &SampleFlags, dim: u32) {
        let deriv_dims = match dim {
            0 | 4 => 1,
            2 => 3,
            _ => 2,
        };
        let first = f.offset as usize + f.bias as usize + f.compare as usize;
        let count = if f.g16 { 2 * (deriv_dims as usize).div_ceil(2) } else { 2 * deriv_dims as usize };
        let exec = self.exec_bit();
        let wave64 = self.func.wave_size == 64;
        let mut b = self.b();
        let (lo, hi) = b.ballot(exec);
        let lane = b.lane_id();
        let not3 = b.u32(!3);
        let quad = b.and(lane, not3);
        let mask = if wave64 {
            let thirty_two = b.u32(32);
            let low = b.icmp(ICmp::ULt, lane, thirty_two);
            b.select(low, lo, hi)
        } else {
            lo
        };
        let thirty_one = b.u32(31);
        let shift = b.and(quad, thirty_one);
        let word = b.lshr(mask, shift);
        let fifteen = b.u32(15);
        let active = b.and(word, fifteen);
        let first_active = b.find_lsb(active);
        let zero = b.u32(0);
        let any = b.icmp(ICmp::Ne, active, zero);
        let leader = b.iadd(quad, first_active);
        let source = b.select(any, leader, lane);
        for v in address.iter_mut().skip(first).take(count) {
            *v = b.shuffle(*v, source);
        }
    }

    fn write_dwords(&mut self, first: u32, v: Value, count: u32) {
        for k in 0..count {
            let c = self.b().extract(v, k);
            self.write_vgpr(first + k, c);
        }
    }

    fn data_vector(&mut self, first: u32, count: u32) -> Value {
        let parts: Vec<Value> = (0..count).map(|k| self.vgpr(first + k)).collect();
        self.b().construct(&parts)
    }

    // ---------------------------------------------------------------
    // LDS (DS)
    // ---------------------------------------------------------------

    /// The load, store and atomic op of the data share an instruction
    /// addresses (LDS or GDS).
    fn ds_load_op(gds: bool, access: Access) -> Op {
        if gds {
            Op::GdsLoad { access }
        } else {
            Op::LdsLoad { access }
        }
    }

    fn ds_store_op(gds: bool, access: Access) -> Op {
        if gds {
            Op::GdsStore { access }
        } else {
            Op::LdsStore { access }
        }
    }

    fn ds_atomic_op(gds: bool, op: AtomicOp) -> Op {
        if gds {
            Op::GdsAtomic { op }
        } else {
            Op::LdsAtomic { op }
        }
    }

    /// Active invocations of the wave.
    fn active_count(&mut self) -> Value {
        let lo = self.sgpr(EXEC_LO);
        if self.wave64() {
            let hi = self.sgpr(EXEC_HI);
            let a = self.b().bit_count(lo);
            let b = self.b().bit_count(hi);
            self.b().iadd(a, b)
        } else {
            self.b().bit_count(lo)
        }
    }

    /// GDS DS_APPEND / DS_CONSUME: the counter is at M0 base plus the
    /// dword-aligned OFFSET, and the whole M0 region `{base[31:16],
    /// size[15:0]}` must cover the counter and fit the 48 KiB partition;
    /// otherwise the counter is untouched and 0 is returned.
    fn translate_gds_append(&mut self, ins: &Instruction, offset: u32) -> Result<()> {
        let vdst = ins.destinations.first().map(|o| o.value).unwrap_or(0);
        let offset = offset & 0xFFFC;
        let m0 = self.sgpr(Reg::M0 as u32);
        let sixteen = self.b().u32(16);
        let base = self.b().lshr(m0, sixteen);
        let ffff = self.b().u32(0xFFFF);
        let size = self.b().and(m0, ffff);
        let off = self.b().u32(offset);
        let addr = self.b().iadd(base, off);
        let last = self.b().u32(offset + 3);
        let covers = self.b().icmp(ICmp::ULt, last, size);
        let end = self.b().iadd(base, size);
        let limit = self.b().u32(0xC000);
        let fits = self.b().icmp(ICmp::ULe, end, limit);
        let valid = self.b().land(covers, fits);
        let none = self.b().u32(u32::MAX);
        let addr = self.b().select(valid, addr, none);
        let count = self.active_count();
        let r = self.b().emit(Op::GdsAppend { consume: ins.opcode == Opcode::DsConsume }, Type::U32, vec![addr, count]);
        self.write_vgpr(vdst, r);
        Ok(())
    }

    /// DS_ORDERED_COUNT as one GDS atomic per wave at M0 base (plus the
    /// packer slot when OFFSET1 selects it); the first active invocation's
    /// ADDR value is added (or swapped in) and every invocation receives
    /// the old counter. Wave ordering and release/done are not modelled.
    fn translate_ordered_count(&mut self, ins: &Instruction, c: &DataShareControl) -> Result<()> {
        let vdst = ins.destinations.first().map(|o| o.value).unwrap_or(0);
        let src = ins.sources.first().map(|o| o.value).unwrap_or(0);
        if !c.gds || (c.offset1 >> 4) & 3 > 1 || (c.offset1 >> 6) & 3 != 0 {
            return self.unsupported(ins, "ordered count variant");
        }
        let m0 = self.sgpr(Reg::M0 as u32);
        let sixteen = self.b().u32(16);
        let base = self.b().lshr(m0, sixteen);
        let mask = self.b().u32(0xFFFC);
        let base = self.b().and(base, mask);
        let off = self.b().u32(c.offset0 & 0xFC);
        let mut addr = self.b().iadd(base, off);
        if (c.offset1 >> 2) & 3 == 1 {
            let f = self.b().u32(0xF);
            let packer = self.b().and(m0, f);
            let two = self.b().u32(2);
            let p4 = self.b().shl(packer, two);
            addr = self.b().iadd(addr, p4);
        }
        let exec = self.exec_bit();
        let (lo, hi) = self.b().ballot(exec);
        let hi = if self.wave64() { hi } else { self.b().u32(0) };
        let zero = self.b().u32(0);
        let any = self.b().or(lo, hi);
        let active = self.b().icmp(ICmp::Ne, any, zero);
        let lo_set = self.b().icmp(ICmp::Ne, lo, zero);
        let first_lo = self.b().find_lsb(lo);
        let first_hi = self.b().find_lsb(hi);
        let thirty_two = self.b().u32(32);
        let first_hi = self.b().iadd(first_hi, thirty_two);
        let first = self.b().select(lo_set, first_lo, first_hi);
        let selected = self.b().select(active, first, zero);
        let lane = self.b().lane_id();
        let is_first = self.b().icmp(ICmp::Eq, lane, selected);
        let v = self.vgpr(src);
        let source = self.b().read_lane(v, selected);
        let value = self.b().select(active, source, zero);
        let result = if (c.offset1 >> 4) & 3 == 1 {
            let not_active = self.b().lnot(active);
            let swap_exec = self.b().land(is_first, active);
            let keep_exec = self.b().land(is_first, not_active);
            let swapped = self.b().emit(Op::GdsAtomic { op: AtomicOp::Swap }, Type::U32, vec![addr, value, swap_exec]);
            let unchanged = self.b().emit(Op::GdsAtomic { op: AtomicOp::Add }, Type::U32, vec![addr, zero, keep_exec]);
            self.b().select(active, swapped, unchanged)
        } else {
            self.b().emit(Op::GdsAtomic { op: AtomicOp::Add }, Type::U32, vec![addr, value, is_first])
        };
        // Every destination lane is written, whatever EXEC holds.
        let r = self.b().read_lane(result, selected);
        self.b().set_reg(Reg::Vgpr(vdst as u16), r);
        Ok(())
    }

    pub(crate) fn translate_ds(&mut self, ins: &Instruction) -> Result<()> {
        use Opcode::*;
        let Some(Control::DataShare(c)) = &ins.control else { return self.invalid(ins, "DS without control") };
        let c = c.clone();
        let gds = c.gds;
        let offset = c.single_offset_bytes();
        if ins.opcode == Opcode::DsOrderedCount {
            return self.translate_ordered_count(ins, &c);
        }
        if gds && matches!(ins.opcode, Opcode::DsAppend | Opcode::DsConsume) {
            return self.translate_gds_append(ins, offset);
        }
        if gds
            && matches!(
                ins.opcode,
                Opcode::DsSwizzleB32 | Opcode::DsPermuteB32 | Opcode::DsBpermuteB32 | Opcode::DsReadAddtidB32 | Opcode::DsWriteAddtidB32
            )
        {
            return self.unsupported(ins, "lane operation on GDS");
        }
        let vsrc = |k: usize| ins.sources.get(k).map(|o| o.value).unwrap_or(0);
        let vdst = ins.destinations.first().map(|o| o.value).unwrap_or(0);
        // GDS addresses are relative to the M0 partition base.
        let gds_base = if gds {
            let m0 = self.sgpr(Reg::M0 as u32);
            let sixteen = self.b().u32(16);
            Some(self.b().lshr(m0, sixteen))
        } else {
            None
        };
        let address = |t: &mut Self, extra: u32| {
            let a = t.vgpr(vsrc(0));
            let o = t.b().u32(extra);
            let a = t.b().iadd(a, o);
            match gds_base {
                Some(base) => t.b().iadd(base, a),
                None => a,
            }
        };
        match ins.opcode {
            DsReadB32 | DsReadB64 | DsReadB96 | DsReadB128 | DsReadU8 | DsReadI8 | DsReadU16 | DsReadI16 => {
                let access = match ins.opcode {
                    DsReadB32 => Access::Dwords(1),
                    DsReadB64 => Access::Dwords(2),
                    DsReadB96 => Access::Dwords(3),
                    DsReadB128 => Access::Dwords(4),
                    DsReadU8 => Access::U8,
                    DsReadI8 => Access::I8,
                    DsReadU16 => Access::U16,
                    _ => Access::I16,
                };
                let addr = address(self, offset);
                let exec = self.exec_operand();
                let n = access.dwords();
                let v = self.b().emit(Self::ds_load_op(gds, access), Type::u32xn(n), vec![addr, exec]);
                self.write_dwords(vdst, v, n);
            }
            DsRead2B32 | DsRead2St64B32 | DsRead2B64 | DsRead2St64B64 => {
                let wide = matches!(ins.opcode, DsRead2B64 | DsRead2St64B64);
                let scale = match ins.opcode {
                    DsRead2B32 => 4,
                    DsRead2St64B32 => 256,
                    DsRead2B64 => 8,
                    _ => 512,
                };
                let n = if wide { 2 } else { 1 };
                // Both loads use the original ADDR: issue them before any
                // destination write (VDST may overlap ADDR).
                let mut loaded = Vec::with_capacity(2);
                for off in [c.offset0, c.offset1] {
                    let addr = address(self, off * scale);
                    let exec = self.exec_operand();
                    loaded.push(self.b().emit(Self::ds_load_op(gds, Access::Dwords(n as u8)), Type::u32xn(n), vec![addr, exec]));
                }
                for (k, v) in loaded.into_iter().enumerate() {
                    self.write_dwords(vdst + k as u32 * n, v, n);
                }
            }
            DsWriteB32 | DsWriteB64 | DsWriteB96 | DsWriteB128 | DsWriteB8 | DsWriteB16 => {
                let access = match ins.opcode {
                    DsWriteB32 => Access::Dwords(1),
                    DsWriteB64 => Access::Dwords(2),
                    DsWriteB96 => Access::Dwords(3),
                    DsWriteB128 => Access::Dwords(4),
                    DsWriteB8 => Access::U8,
                    _ => Access::U16,
                };
                let addr = address(self, offset);
                let data = self.data_vector(vsrc(1), access.dwords());
                let exec = self.exec_operand();
                self.b().emit(Self::ds_store_op(gds, access), Type::VOID, vec![addr, data, exec]);
            }
            DsWrite2B32 | DsWrite2St64B32 | DsWrite2B64 | DsWrite2St64B64 => {
                let wide = matches!(ins.opcode, DsWrite2B64 | DsWrite2St64B64);
                let scale = match ins.opcode {
                    DsWrite2B32 => 4,
                    DsWrite2St64B32 => 256,
                    DsWrite2B64 => 8,
                    _ => 512,
                };
                let n = if wide { 2 } else { 1 };
                let (d0, d1) = if wide { (vsrc(1), vsrc(3)) } else { (vsrc(1), vsrc(2)) };
                // With equal offsets DATA0 is what memory keeps.
                let both = if c.offset0 == c.offset1 { 1 } else { 2 };
                for (off, data_reg) in [(c.offset0, d0), (c.offset1, d1)].into_iter().take(both) {
                    let addr = address(self, off * scale);
                    let data = self.data_vector(data_reg, n);
                    let exec = self.exec_operand();
                    self.b().emit(Self::ds_store_op(gds, Access::Dwords(n as u8)), Type::VOID, vec![addr, data, exec]);
                }
            }
            DsReadAddtidB32 | DsWriteAddtidB32 => {
                let m0 = self.sgpr(M0);
                let ffff = self.b().u32(0xFFFF);
                let base = self.b().and(m0, ffff);
                let lane = self.b().lane_id();
                let two = self.b().u32(2);
                let l4 = self.b().shl(lane, two);
                let o = self.b().u32(offset);
                let a = self.b().iadd(base, l4);
                let addr = self.b().iadd(a, o);
                let exec = self.exec_operand();
                if ins.opcode == DsReadAddtidB32 {
                    let v = self.b().emit(Op::LdsLoad { access: Access::Dwords(1) }, Type::U32, vec![addr, exec]);
                    self.write_vgpr(vdst, v);
                } else {
                    let data = self.vgpr(vsrc(1));
                    self.b().emit(Op::LdsStore { access: Access::Dwords(1) }, Type::VOID, vec![addr, data, exec]);
                }
            }
            // offset[15:13] = 7: the FFT swizzle, whose lane pattern is
            // not documented or measured.
            DsSwizzleB32 if offset >= 0xE000 => return self.unsupported(ins, "ds_swizzle_b32 FFT mode"),
            DsSwizzleB32 => {
                let v = self.vgpr(vsrc(0));
                let target = self.swizzle_target(offset);
                let wm = self.wave_mask();
                let target = self.b().and(target, wm);
                let shuffled = self.b().shuffle(v, target);
                let active = self.lane_active(target);
                let zero = self.b().u32(0);
                let r = self.b().select(active, shuffled, zero);
                self.write_vgpr(vdst, r);
            }
            DsBpermuteB32 => {
                let a = address(self, offset);
                let data = self.vgpr(vsrc(1));
                let two = self.b().u32(2);
                let idx = self.b().lshr(a, two);
                let m31 = self.b().u32(31);
                let idx = self.b().and(idx, m31);
                let lane = self.b().lane_id();
                let hm = self.b().u32(!31);
                let base = self.b().and(lane, hm);
                let target = self.b().or(base, idx);
                let shuffled = self.b().shuffle(data, target);
                let active = self.lane_active(target);
                let zero = self.b().u32(0);
                let r = self.b().select(active, shuffled, zero);
                self.write_vgpr(vdst, r);
            }
            DsAppend | DsConsume => {
                // LDS counters ignore M0 (its base/size apply to GDS only)
                // and use the dword-aligned OFFSET.
                let addr = self.b().u32(offset & 0xFFFC);
                let count = self.active_count();
                let r = self.b().emit(Op::LdsAppend { consume: ins.opcode == DsConsume }, Type::U32, vec![addr, count]);
                self.write_vgpr(vdst, r);
            }
            DsWrxchg2RtnB32 | DsWrxchg2St64RtnB32 | DsWrxchg2RtnB64 | DsWrxchg2St64RtnB64 => {
                let wide = matches!(ins.opcode, DsWrxchg2RtnB64 | DsWrxchg2St64RtnB64);
                let scale = match ins.opcode {
                    DsWrxchg2RtnB32 => 4,
                    DsWrxchg2St64RtnB32 => 256,
                    DsWrxchg2RtnB64 => 8,
                    _ => 512,
                };
                let w = if wide { 2 } else { 1 };
                // Both exchanges read ADDR and DATA before any destination
                // write (VDST may overlap them).
                let mut returned = Vec::with_capacity(2);
                for (k, off) in [c.offset0, c.offset1].into_iter().enumerate() {
                    let addr = address(self, off * scale);
                    let exec = self.exec_operand();
                    let first = 1 + k * w as usize;
                    let data = if wide {
                        let lo = self.vgpr(vsrc(first));
                        let hi = self.vgpr(vsrc(first + 1));
                        self.b().join_u64(lo, hi)
                    } else {
                        self.vgpr(vsrc(first))
                    };
                    let ty = if wide { Type::U64 } else { Type::U32 };
                    returned.push(self.b().emit(Self::ds_atomic_op(gds, AtomicOp::Swap), ty, vec![addr, data, exec]));
                }
                for (k, old) in returned.into_iter().enumerate() {
                    if wide {
                        let (lo, hi) = self.b().split_u64(old);
                        self.write_vgpr(vdst + 2 * k as u32, lo);
                        self.write_vgpr(vdst + 2 * k as u32 + 1, hi);
                    } else {
                        self.write_vgpr(vdst + k as u32, old);
                    }
                }
            }
            DsReadU8D16 | DsReadU8D16Hi | DsReadI8D16 | DsReadI8D16Hi | DsReadU16D16 | DsReadU16D16Hi => {
                let access = match ins.opcode {
                    DsReadU8D16 | DsReadU8D16Hi => Access::U8,
                    DsReadI8D16 | DsReadI8D16Hi => Access::I8,
                    _ => Access::U16,
                };
                let high = matches!(ins.opcode, DsReadU8D16Hi | DsReadI8D16Hi | DsReadU16D16Hi);
                let addr = address(self, offset);
                let exec = self.exec_operand();
                let v = self.b().emit(Self::ds_load_op(gds, access), Type::U32, vec![addr, exec]);
                let old = self.vgpr(vdst);
                let o = self.b().u32(if high { 16 } else { 0 });
                let w = self.b().u32(16);
                let merged = self.b().bfi(old, v, o, w);
                self.write_vgpr(vdst, merged);
            }
            DsWriteB8D16Hi | DsWriteB16D16Hi => {
                let access = if ins.opcode == DsWriteB8D16Hi { Access::U8 } else { Access::U16 };
                let addr = address(self, offset);
                let d = self.vgpr(vsrc(1));
                let sixteen = self.b().u32(16);
                let d = self.b().lshr(d, sixteen);
                let exec = self.exec_operand();
                self.b().emit(Self::ds_store_op(gds, access), Type::VOID, vec![addr, d, exec]);
            }
            DsPermuteB32 => {
                // Forward permute: invocation i sends its data to lane
                // ((addr_i + offset) / 4) mod 32 within its half; a lane
                // keeps the last active sender's value (0 when none).
                let a = address(self, offset);
                let data = self.vgpr(vsrc(1));
                let two = self.b().u32(2);
                let m31 = self.b().u32(31);
                let target = self.b().lshr(a, two);
                let target = self.b().and(target, m31);
                let lane = self.b().lane_id();
                let own = self.b().and(lane, m31);
                let hm = self.b().u32(!31);
                let base = self.b().and(lane, hm);
                let exec = self.exec_bit();
                let exec_u = self.b().bool_to_u32(exec);
                let mut result = self.b().u32(0);
                for src in 0..32u32 {
                    let sv = self.b().u32(src);
                    let from = self.b().or(base, sv);
                    let wm = self.wave_mask();
                    let from = self.b().and(from, wm);
                    let value = self.b().shuffle(data, from);
                    let t = self.b().shuffle(target, from);
                    let act = self.b().shuffle(exec_u, from);
                    let zero = self.b().u32(0);
                    let active = self.b().icmp(ICmp::Ne, act, zero);
                    let mine = self.b().icmp(ICmp::Eq, t, own);
                    let hit = self.b().land(active, mine);
                    result = self.b().select(hit, value, result);
                }
                self.write_vgpr(vdst, result);
            }
            // Two masked writes: each dword of DATA0 with bit 31 set stores
            // its low 31 bits (the reference's ds_condxchg32_rtn_b64); both
            // old dwords are returned.
            DsCondxchg32RtnB64 => {
                if gds {
                    return self.unsupported(ins, "ds_condxchg32_rtn_b64 on GDS");
                }
                let exec = self.exec_operand();
                let mut old = Vec::new();
                for k in 0..2u32 {
                    let data = self.vgpr(vsrc(1) + k);
                    let lim = self.b().u32(0x7FFF_FFFF);
                    let enabled = self.b().icmp(ICmp::UGt, data, lim);
                    let ones = self.b().u32(u32::MAX);
                    let zero = self.b().u32(0);
                    let mask = self.b().select(enabled, ones, zero);
                    let low = self.b().and(data, lim);
                    let value = self.b().select(enabled, low, zero);
                    let addr = address(self, offset + 4 * k);
                    old.push(self.b().emit(Self::ds_atomic_op(false, AtomicOp::MaskOr), Type::U32, vec![addr, mask, value, exec]));
                }
                self.write_vgpr(vdst, old[0]);
                self.write_vgpr(vdst + 1, old[1]);
            }
            // MEM[A] = op(MEM[A], MEM[B]) with B = A + 4 * the signed 15-bit
            // offset (offset1 bit 7 takes it from the address instead).
            DsMinSrc2F32 | DsMaxSrc2F32 | DsAddSrc2F32 => {
                return self.unsupported(ins, "f32 DS src2 operations depend on the f32 denormal mode, which is not modelled for LDS float atomics");
            }
            DsAddSrc2U32 | DsSubSrc2U32 | DsRsubSrc2U32 | DsIncSrc2U32 | DsDecSrc2U32 | DsMinSrc2I32 | DsMaxSrc2I32 | DsMinSrc2U32 | DsMaxSrc2U32
            | DsAndSrc2B32 | DsOrSrc2B32 | DsXorSrc2B32 | DsWriteSrc2B32 => {
                if gds {
                    return self.unsupported(ins, "GDS src2 operations");
                }
                if c.offset1 & 0x80 != 0 {
                    return self.unsupported(ins, "DS src2 operation with the offset taken from the address");
                }
                let combined = (c.offset1 << 8 | c.offset0) & 0x7FFF;
                let delta = ((combined << 17) as i32 >> 17).wrapping_mul(4) as u32;
                let a = address(self, 0);
                let b = address(self, delta);
                let exec = self.exec_operand();
                let value = self.b().emit(Self::ds_load_op(false, Access::Dwords(1)), Type::U32, vec![b, exec]);
                if ins.opcode == DsWriteSrc2B32 {
                    self.b().emit(Op::LdsStore { access: Access::Dwords(1) }, Type::VOID, vec![a, value, exec]);
                } else {
                    let aop = match ins.opcode {
                        DsAddSrc2U32 => AtomicOp::Add,
                        DsSubSrc2U32 => AtomicOp::Sub,
                        DsRsubSrc2U32 => AtomicOp::RSub,
                        DsIncSrc2U32 => AtomicOp::Inc,
                        DsDecSrc2U32 => AtomicOp::Dec,
                        DsMinSrc2I32 => AtomicOp::SMin,
                        DsMaxSrc2I32 => AtomicOp::SMax,
                        DsMinSrc2U32 => AtomicOp::UMin,
                        DsMaxSrc2U32 => AtomicOp::UMax,
                        DsAndSrc2B32 => AtomicOp::And,
                        DsOrSrc2B32 => AtomicOp::Or,
                        _ => AtomicOp::Xor,
                    };
                    self.b().emit(Self::ds_atomic_op(false, aop), Type::U32, vec![a, value, exec]);
                }
            }
            DsGwsSemaReleaseAll | DsGwsInit | DsGwsSemaV | DsGwsSemaBr | DsGwsSemaP | DsGwsBarrier => {
                return self.unsupported(ins, "global wave sync barriers and semaphores shared by waves of different workgroups");
            }
            op if op.is_data_share_atomic() => {
                let (wide, ndata, returns) = op.ds_atomic_shape().unwrap();
                let aop = match op {
                    DsAddU32 | DsAddRtnU32 | DsAddU64 | DsAddRtnU64 => AtomicOp::Add,
                    DsSubU32 | DsSubRtnU32 | DsSubU64 | DsSubRtnU64 => AtomicOp::Sub,
                    DsRsubU32 | DsRsubRtnU32 | DsRsubU64 | DsRsubRtnU64 => AtomicOp::RSub,
                    DsIncU32 | DsIncRtnU32 | DsIncU64 | DsIncRtnU64 => AtomicOp::Inc,
                    DsDecU32 | DsDecRtnU32 | DsDecU64 | DsDecRtnU64 => AtomicOp::Dec,
                    DsMinI32 | DsMinRtnI32 | DsMinI64 | DsMinRtnI64 => AtomicOp::SMin,
                    DsMaxI32 | DsMaxRtnI32 | DsMaxI64 | DsMaxRtnI64 => AtomicOp::SMax,
                    DsMinU32 | DsMinRtnU32 | DsMinU64 | DsMinRtnU64 => AtomicOp::UMin,
                    DsMaxU32 | DsMaxRtnU32 | DsMaxU64 | DsMaxRtnU64 => AtomicOp::UMax,
                    DsAndB32 | DsAndRtnB32 | DsAndB64 | DsAndRtnB64 => AtomicOp::And,
                    DsOrB32 | DsOrRtnB32 | DsOrB64 | DsOrRtnB64 => AtomicOp::Or,
                    DsXorB32 | DsXorRtnB32 | DsXorB64 | DsXorRtnB64 => AtomicOp::Xor,
                    DsWrxchgRtnB32 | DsWrxchgRtnB64 => AtomicOp::Swap,
                    DsCmpstB32 | DsCmpstRtnB32 | DsCmpstB64 | DsCmpstRtnB64 => AtomicOp::CmpSwap,
                    DsCmpstF32 | DsCmpstRtnF32 | DsCmpstF64 | DsCmpstRtnF64 => AtomicOp::FCmpSwap,
                    DsMskorB32 | DsMskorRtnB32 | DsMskorB64 | DsMskorRtnB64 => AtomicOp::MaskOr,
                    DsWrapRtnB32 => AtomicOp::Wrap,
                    DsAddF32 | DsAddRtnF32 => {
                        // The f32 denormal field of FLOAT_MODE: bit 0 keeps
                        // input denormals, bit 1 results.
                        let kept = self.float_mode.f32_denormals();
                        AtomicOp::DsFAdd { flush_inputs: kept & 1 == 0, flush_result: kept & 2 == 0 }
                    }
                    // DS_MIN/MAX_F32 (and RTN / F64) compare DATA1 and
                    // write DATA0; modelled as plain min/max of DATA0.
                    DsMinF32 | DsMinRtnF32 => AtomicOp::DsFMin,
                    DsMaxF32 | DsMaxRtnF32 => AtomicOp::DsFMax,
                    DsMinF64 | DsMinRtnF64 => AtomicOp::FMin,
                    DsMaxF64 | DsMaxRtnF64 => AtomicOp::FMax,
                    _ => return self.unsupported(ins, "LDS atomic"),
                };
                let addr = address(self, offset);
                let exec = self.exec_operand();
                let ty = if wide { Type::U64 } else { Type::U32 };
                let w = if wide { 2 } else { 1 };
                let read_data = |t: &mut Self, k: usize| {
                    let first = 1 + k * w;
                    if wide {
                        let lo = t.vgpr(vsrc(first));
                        let hi = t.vgpr(vsrc(first + 1));
                        t.b().join_u64(lo, hi)
                    } else {
                        t.vgpr(vsrc(first))
                    }
                };
                let d0 = read_data(self, 0);
                let mut args = vec![addr];
                if ndata == 2 && matches!(aop, AtomicOp::CmpSwap | AtomicOp::FCmpSwap) {
                    // DS_CMPST: DATA0 comparator, DATA1 new value; the IR
                    // takes (new value, comparator).
                    let d1 = read_data(self, 1);
                    args.push(d1);
                    args.push(d0);
                } else if ndata == 2 && matches!(aop, AtomicOp::MaskOr | AtomicOp::Wrap) {
                    let d1 = read_data(self, 1);
                    args.push(d0);
                    args.push(d1);
                } else {
                    args.push(d0);
                }
                args.push(exec);
                let old = self.b().emit(Self::ds_atomic_op(gds, aop), ty, args);
                if returns {
                    if wide {
                        let (lo, hi) = self.b().split_u64(old);
                        self.write_vgpr(vdst, lo);
                        self.write_vgpr(vdst + 1, hi);
                    } else {
                        self.write_vgpr(vdst, old);
                    }
                }
            }
            _ => return self.unsupported(ins, "LDS"),
        }
        Ok(())
    }

    /// DS_SWIZZLE source lane for a 16-bit pattern.
    fn swizzle_target(&mut self, control: u32) -> Value {
        let lane = self.b().lane_id();
        let c = |t: &mut Self, x: u32| t.b().u32(x);
        if control & 0xC000 == 0xC000 {
            let mask = control & 0x1F;
            let rotate = (control >> 5) & 0x1F;
            let delta = if control & 0x400 != 0 { (32 - rotate) & 0x1F } else { rotate };
            let m31 = c(self, 31);
            let l = self.b().and(lane, m31);
            let d = c(self, delta);
            let s = self.b().iadd(l, d);
            let rotated = self.b().and(s, m31);
            let mk = c(self, mask);
            let kept = self.b().and(l, mk);
            let nmk = c(self, !mask & 31);
            let moved = self.b().and(rotated, nmk);
            let comb = self.b().or(kept, moved);
            let hb = c(self, !31);
            let base = self.b().and(lane, hb);
            return self.b().or(base, comb);
        }
        if control & 0x8000 != 0 {
            let three = c(self, 3);
            let l2 = self.b().and(lane, three);
            let one = c(self, 1);
            let shift = self.b().shl(l2, one);
            let ctrl = c(self, control);
            let p = self.b().lshr(ctrl, shift);
            let p = self.b().and(p, three);
            let qb = c(self, !3);
            let base = self.b().and(lane, qb);
            return self.b().or(base, p);
        }
        let m31 = c(self, 31);
        let l = self.b().and(lane, m31);
        let am = c(self, control & 0x1F);
        let masked = self.b().and(l, am);
        let om = c(self, (control >> 5) & 0x1F);
        let ored = self.b().or(masked, om);
        let xm = c(self, (control >> 10) & 0x1F);
        let x = self.b().xor(ored, xm);
        let hb = c(self, !31);
        let base = self.b().and(lane, hb);
        self.b().or(base, x)
    }

    // ---------------------------------------------------------------
    // Buffers (MUBUF / MTBUF)
    // ---------------------------------------------------------------

    pub(crate) fn translate_buffer(&mut self, ins: &Instruction) -> Result<()> {
        use Opcode::*;
        let Some(Control::BufferMemory(c)) = &ins.control else { return self.invalid(ins, "buffer without control") };
        // MUBUF LDS (word 0 bit 16) loads into LDS at M0 instead of VGPRs.
        if ins.encoding == Encoding::Mubuf && (ins.words[0] >> 16) & 1 != 0 {
            return self.unsupported(ins, "MUBUF lds modifier (load into LDS)");
        }
        let c = c.clone();
        let attr = BufferAttr { pc: ins.pc, idxen: c.index_enabled, offen: c.offset_enabled, offset: c.offset_bytes as u32, glc: c.glc, slc: c.slc };
        let desc_parts: Vec<Value> = (0..4).map(|k| self.sgpr(c.scalar_resource + k)).collect();
        let desc = self.b().construct(&desc_parts);
        let vindex = if c.index_enabled { self.vgpr(c.vector_address) } else { self.b().u32(0) };
        let voffset = if c.offset_enabled { self.vgpr(c.vector_address + c.index_enabled as u32) } else { self.b().u32(0) };
        let soffset = self.read_raw(ins.sources[2]);
        let exec = self.exec_operand();
        let vdata = c.vector_data;
        let raw = |n: u8| BufferData::Raw(Access::Dwords(n));
        let typed = if c.typed { Some(c.typed_format) } else { None };
        let format = |n: u8| BufferData::Format { components: n, typed_format: typed, d16: false };
        let d16_format = |n: u8| BufferData::Format { components: n, typed_format: typed, d16: true };
        if let Some((load, n, high)) = match ins.opcode {
            BufferLoadFormatD16X | TBufferLoadFormatD16X => Some((true, 1, false)),
            BufferLoadFormatD16Xy | TBufferLoadFormatD16Xy => Some((true, 2, false)),
            BufferLoadFormatD16Xyz | TBufferLoadFormatD16Xyz => Some((true, 3, false)),
            BufferLoadFormatD16Xyzw | TBufferLoadFormatD16Xyzw => Some((true, 4, false)),
            BufferLoadFormatD16HiX => Some((true, 1, true)),
            BufferStoreFormatD16X | TBufferStoreFormatD16X => Some((false, 1, false)),
            BufferStoreFormatD16Xy | TBufferStoreFormatD16Xy => Some((false, 2, false)),
            BufferStoreFormatD16Xyz | TBufferStoreFormatD16Xyz => Some((false, 3, false)),
            BufferStoreFormatD16Xyzw | TBufferStoreFormatD16Xyzw => Some((false, 4, false)),
            BufferStoreFormatD16HiX => Some((false, 1, true)),
            _ => None,
        } {
            let data = d16_format(n);
            let n = n as u32;
            if load {
                // One 16-bit value per component: pairs pack into a dword,
                // a lone last (or the only) component writes one half and
                // keeps the other.
                let v = self.b().emit(Op::BufferLoad { attr, data }, Type::u32xn(n), vec![desc, vindex, voffset, soffset, exec]);
                for k in (0..n).step_by(2) {
                    let lo = self.b().extract(v, k);
                    if k + 1 < n {
                        let hi = self.b().extract(v, k + 1);
                        let sixteen = self.b().u32(16);
                        let hi = self.b().shl(hi, sixteen);
                        let word = self.b().or(lo, hi);
                        self.write_vgpr(vdata + k / 2, word);
                    } else {
                        let old = self.vgpr(vdata + k / 2);
                        let o = self.b().u32(if high { 16 } else { 0 });
                        let w = self.b().u32(16);
                        let merged = self.b().bfi(old, lo, o, w);
                        self.write_vgpr(vdata + k / 2, merged);
                    }
                }
            } else {
                let parts: Vec<Value> = (0..n)
                    .map(|k| {
                        let word = self.vgpr(vdata + k / 2);
                        if high || k % 2 == 1 {
                            let sixteen = self.b().u32(16);
                            self.b().lshr(word, sixteen)
                        } else {
                            let mask = self.b().u32(0xFFFF);
                            self.b().and(word, mask)
                        }
                    })
                    .collect();
                let value = self.b().construct(&parts);
                self.b().emit(Op::BufferStore { attr, data }, Type::VOID, vec![desc, vindex, voffset, soffset, value, exec]);
            }
            return Ok(());
        }
        let (load, data, d16_half): (Option<bool>, BufferData, Option<(bool, bool)>) = match ins.opcode {
            BufferLoadDword => (Some(true), raw(1), None),
            BufferLoadDwordx2 => (Some(true), raw(2), None),
            BufferLoadDwordx3 => (Some(true), raw(3), None),
            BufferLoadDwordx4 => (Some(true), raw(4), None),
            BufferLoadUbyte => (Some(true), BufferData::Raw(Access::U8), None),
            BufferLoadSbyte => (Some(true), BufferData::Raw(Access::I8), None),
            BufferLoadUshort => (Some(true), BufferData::Raw(Access::U16), None),
            BufferLoadSshort => (Some(true), BufferData::Raw(Access::I16), None),
            BufferLoadUbyteD16 => (Some(true), BufferData::Raw(Access::U8), Some((false, false))),
            BufferLoadUbyteD16Hi => (Some(true), BufferData::Raw(Access::U8), Some((true, false))),
            BufferLoadSbyteD16 => (Some(true), BufferData::Raw(Access::I8), Some((false, true))),
            BufferLoadSbyteD16Hi => (Some(true), BufferData::Raw(Access::I8), Some((true, true))),
            BufferLoadShortD16 => (Some(true), BufferData::Raw(Access::U16), Some((false, false))),
            BufferLoadShortD16Hi => (Some(true), BufferData::Raw(Access::U16), Some((true, false))),
            BufferLoadFormatX | TBufferLoadFormatX => (Some(true), format(1), None),
            BufferLoadFormatXy | TBufferLoadFormatXy => (Some(true), format(2), None),
            BufferLoadFormatXyz | TBufferLoadFormatXyz => (Some(true), format(3), None),
            BufferLoadFormatXyzw | TBufferLoadFormatXyzw => (Some(true), format(4), None),
            BufferStoreDword => (Some(false), raw(1), None),
            BufferStoreDwordx2 => (Some(false), raw(2), None),
            BufferStoreDwordx3 => (Some(false), raw(3), None),
            BufferStoreDwordx4 => (Some(false), raw(4), None),
            BufferStoreByte => (Some(false), BufferData::Raw(Access::U8), None),
            BufferStoreByteD16Hi => (Some(false), BufferData::Raw(Access::U8), Some((true, false))),
            BufferStoreShort => (Some(false), BufferData::Raw(Access::U16), None),
            BufferStoreShortD16Hi => (Some(false), BufferData::Raw(Access::U16), Some((true, false))),
            BufferStoreFormatX | TBufferStoreFormatX => (Some(false), format(1), None),
            BufferStoreFormatXy | TBufferStoreFormatXy => (Some(false), format(2), None),
            BufferStoreFormatXyz | TBufferStoreFormatXyz => (Some(false), format(3), None),
            BufferStoreFormatXyzw | TBufferStoreFormatXyzw => (Some(false), format(4), None),
            _ => (None, raw(1), None),
        };
        match load {
            Some(true) => {
                let n = match data {
                    BufferData::Raw(a) => a.dwords(),
                    BufferData::Format { components, .. } => components as u32,
                };
                let v = self.b().emit(Op::BufferLoad { attr, data }, Type::u32xn(n), vec![desc, vindex, voffset, soffset, exec]);
                match d16_half {
                    Some((high, _)) => {
                        let old = self.vgpr(vdata);
                        let o = self.b().u32(if high { 16 } else { 0 });
                        let w = self.b().u32(16);
                        let merged = self.b().bfi(old, v, o, w);
                        self.write_vgpr(vdata, merged);
                    }
                    None => self.write_dwords(vdata, v, n),
                }
            }
            Some(false) => {
                let n = match data {
                    BufferData::Raw(a) => a.dwords(),
                    BufferData::Format { components, .. } => components as u32,
                };
                let value = match d16_half {
                    Some((true, _)) => {
                        let v = self.vgpr(vdata);
                        let sixteen = self.b().u32(16);
                        self.b().lshr(v, sixteen)
                    }
                    _ => self.data_vector(vdata, n),
                };
                self.b().emit(Op::BufferStore { attr, data }, Type::VOID, vec![desc, vindex, voffset, soffset, value, exec]);
            }
            None => {
                let (op, wide) = match ins.opcode {
                    BufferAtomicSwap => (AtomicOp::Swap, false),
                    BufferAtomicCmpswap => (AtomicOp::CmpSwap, false),
                    BufferAtomicAdd => (AtomicOp::Add, false),
                    BufferAtomicSub => (AtomicOp::Sub, false),
                    BufferAtomicSmin => (AtomicOp::SMin, false),
                    BufferAtomicUmin => (AtomicOp::UMin, false),
                    BufferAtomicSmax => (AtomicOp::SMax, false),
                    BufferAtomicUmax => (AtomicOp::UMax, false),
                    BufferAtomicAnd => (AtomicOp::And, false),
                    BufferAtomicOr => (AtomicOp::Or, false),
                    BufferAtomicXor => (AtomicOp::Xor, false),
                    BufferAtomicInc => (AtomicOp::Inc, false),
                    BufferAtomicDec => (AtomicOp::Dec, false),
                    BufferAtomicFmin => (AtomicOp::FMin, false),
                    BufferAtomicFmax => (AtomicOp::FMax, false),
                    BufferAtomicCsub => (AtomicOp::CSub, false),
                    BufferAtomicFcmpswap => (AtomicOp::FCmpSwap, false),
                    BufferAtomicSwapX2 => (AtomicOp::Swap, true),
                    BufferAtomicOrX2 => (AtomicOp::Or, true),
                    BufferAtomicCmpswapX2 => (AtomicOp::CmpSwap, true),
                    BufferAtomicAddX2 => (AtomicOp::Add, true),
                    BufferAtomicSubX2 => (AtomicOp::Sub, true),
                    BufferAtomicSminX2 => (AtomicOp::SMin, true),
                    BufferAtomicUminX2 => (AtomicOp::UMin, true),
                    BufferAtomicSmaxX2 => (AtomicOp::SMax, true),
                    BufferAtomicUmaxX2 => (AtomicOp::UMax, true),
                    BufferAtomicAndX2 => (AtomicOp::And, true),
                    BufferAtomicXorX2 => (AtomicOp::Xor, true),
                    BufferAtomicIncX2 => (AtomicOp::Inc, true),
                    BufferAtomicDecX2 => (AtomicOp::Dec, true),
                    BufferAtomicFcmpswapX2 => (AtomicOp::FCmpSwap, true),
                    BufferAtomicFminX2 => (AtomicOp::FMin, true),
                    BufferAtomicFmaxX2 => (AtomicOp::FMax, true),
                    _ => return self.unsupported(ins, "buffer operation"),
                };
                let ty = if wide { Type::U64 } else { Type::U32 };
                let mut args = vec![desc, vindex, voffset, soffset];
                if wide {
                    let lo = self.vgpr(vdata);
                    let hi = self.vgpr(vdata + 1);
                    let d = self.b().join_u64(lo, hi);
                    args.push(d);
                    if matches!(op, AtomicOp::CmpSwap | AtomicOp::FCmpSwap) {
                        let lo = self.vgpr(vdata + 2);
                        let hi = self.vgpr(vdata + 3);
                        let c = self.b().join_u64(lo, hi);
                        args.push(c);
                    }
                } else if matches!(op, AtomicOp::CmpSwap | AtomicOp::FCmpSwap) {
                    let new = self.vgpr(vdata);
                    let cmp = self.vgpr(vdata + 1);
                    args.push(new);
                    args.push(cmp);
                } else {
                    let d = self.vgpr(vdata);
                    args.push(d);
                }
                args.push(exec);
                let old = self.b().emit(Op::BufferAtomic { attr, op }, ty, args);
                if c.glc {
                    if wide {
                        let (lo, hi) = self.b().split_u64(old);
                        self.write_vgpr(vdata, lo);
                        self.write_vgpr(vdata + 1, hi);
                    } else {
                        self.write_vgpr(vdata, old);
                    }
                }
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------------
    // FLAT / GLOBAL / SCRATCH
    // ---------------------------------------------------------------

    pub(crate) fn translate_flat(&mut self, ins: &Instruction) -> Result<()> {
        let Opcode::Flat(segment, op) = ins.opcode else { return self.invalid(ins, "flat opcode") };
        let Some(Control::GlobalMemory(c)) = &ins.control else { return self.invalid(ins, "flat without control") };
        let c = c.clone();
        let exec = self.exec_operand();
        let access = match op {
            FlatOp::LoadUbyte | FlatOp::StoreByte | FlatOp::LoadUbyteD16 | FlatOp::LoadUbyteD16Hi | FlatOp::StoreByteD16Hi => Access::U8,
            FlatOp::LoadSbyte | FlatOp::LoadSbyteD16 | FlatOp::LoadSbyteD16Hi => Access::I8,
            FlatOp::LoadUshort | FlatOp::StoreShort | FlatOp::LoadShortD16 | FlatOp::LoadShortD16Hi | FlatOp::StoreShortD16Hi => {
                Access::U16
            }
            FlatOp::LoadSshort => Access::I16,
            _ => Access::Dwords(op.dword_count() as u8),
        };
        let d16_high = matches!(op, FlatOp::LoadUbyteD16Hi | FlatOp::LoadSbyteD16Hi | FlatOp::LoadShortD16Hi | FlatOp::StoreByteD16Hi | FlatOp::StoreShortD16Hi);
        let d16 = op.is_d16_load() || matches!(op, FlatOp::StoreByteD16Hi | FlatOp::StoreShortD16Hi);
        let n = access.dwords();
        if segment == FlatSegment::Scratch {
            let base = if c.scalar_address < NULL_SOURCE { self.sgpr(c.scalar_address) } else { self.vgpr(c.vector_address) };
            let o = self.b().u32(c.offset_bytes as u32);
            let offset = self.b().iadd(base, o);
            if op.is_load() {
                let v = self.b().emit(Op::ScratchLoad { pc: ins.pc, access }, Type::u32xn(n), vec![offset, exec]);
                self.finish_load(c.destination_vector_register, v, n, d16, d16_high);
            } else if op.is_store() {
                let data = self.store_data(c.source_vector_register, n, d16_high);
                self.b().emit(Op::ScratchStore { pc: ins.pc, access }, Type::VOID, vec![offset, data, exec]);
            } else {
                return self.unsupported(ins, "scratch atomics");
            }
            return Ok(());
        }
        // Address: SADDR (64-bit) + zero-extended VADDR, or a 64-bit VADDR.
        let addtid = matches!(op, FlatOp::LoadDwordAddtid | FlatOp::StoreDwordAddtid);
        let addr = if addtid {
            // SADDR + TID * 4 (VADDR unused).
            let (lo, hi) = if c.scalar_address < NULL_SOURCE {
                (self.sgpr(c.scalar_address), self.sgpr(c.scalar_address + 1))
            } else {
                (self.b().u32(0), self.b().u32(0))
            };
            let base = self.b().join_u64(lo, hi);
            let lane = self.b().lane_id();
            let two = self.b().u32(2);
            let l4 = self.b().shl(lane, two);
            let l64 = self.b().uconvert(l4, Type::U64);
            self.b().iadd(base, l64)
        } else if segment == FlatSegment::Global && c.scalar_address < NULL_SOURCE {
            let lo = self.sgpr(c.scalar_address);
            let hi = self.sgpr(c.scalar_address + 1);
            let base = self.b().join_u64(lo, hi);
            let v = self.vgpr(c.vector_address);
            let v64 = self.b().uconvert(v, Type::U64);
            self.b().iadd(base, v64)
        } else {
            let lo = self.vgpr(c.vector_address);
            let hi = self.vgpr(c.vector_address + 1);
            self.b().join_u64(lo, hi)
        };
        let off = self.b().u64(c.offset_bytes as i64 as u64);
        let addr = self.b().iadd(addr, off);
        let flat = segment == FlatSegment::Flat;
        if op.is_load() {
            let v = self.b().emit(Op::GlobalLoad { pc: ins.pc, access, flat, glc: c.glc }, Type::u32xn(n), vec![addr, exec]);
            self.finish_load(c.destination_vector_register, v, n, d16, d16_high);
        } else if op.is_store() {
            let data = self.store_data(c.source_vector_register, n, d16_high);
            self.b().emit(Op::GlobalStore { pc: ins.pc, access, flat, glc: c.glc }, Type::VOID, vec![addr, data, exec]);
        } else {
            let wide = op.return_dwords() == 2;
            let aop = match op {
                FlatOp::AtomicSwap | FlatOp::AtomicSwapX2 => AtomicOp::Swap,
                FlatOp::AtomicCmpswap | FlatOp::AtomicCmpswapX2 => AtomicOp::CmpSwap,
                FlatOp::AtomicAdd | FlatOp::AtomicAddX2 => AtomicOp::Add,
                FlatOp::AtomicSub | FlatOp::AtomicSubX2 => AtomicOp::Sub,
                FlatOp::AtomicCsub => AtomicOp::CSub,
                FlatOp::AtomicSMin | FlatOp::AtomicSMinX2 => AtomicOp::SMin,
                FlatOp::AtomicUMin | FlatOp::AtomicUMinX2 => AtomicOp::UMin,
                FlatOp::AtomicSMax | FlatOp::AtomicSMaxX2 => AtomicOp::SMax,
                FlatOp::AtomicUMax | FlatOp::AtomicUMaxX2 => AtomicOp::UMax,
                FlatOp::AtomicAnd | FlatOp::AtomicAndX2 => AtomicOp::And,
                FlatOp::AtomicOr | FlatOp::AtomicOrX2 => AtomicOp::Or,
                FlatOp::AtomicXor | FlatOp::AtomicXorX2 => AtomicOp::Xor,
                FlatOp::AtomicInc | FlatOp::AtomicIncX2 => AtomicOp::Inc,
                FlatOp::AtomicDec | FlatOp::AtomicDecX2 => AtomicOp::Dec,
                FlatOp::AtomicFcmpswap | FlatOp::AtomicFcmpswapX2 => AtomicOp::FCmpSwap,
                FlatOp::AtomicFMin | FlatOp::AtomicFMinX2 => AtomicOp::FMin,
                FlatOp::AtomicFMax | FlatOp::AtomicFMaxX2 => AtomicOp::FMax,
                _ => return self.unsupported(ins, "flat atomic"),
            };
            let src = c.source_vector_register;
            let read = |t: &mut Self, r: u32| {
                if wide {
                    let lo = t.vgpr(r);
                    let hi = t.vgpr(r + 1);
                    t.b().join_u64(lo, hi)
                } else {
                    t.vgpr(r)
                }
            };
            let w = if wide { 2 } else { 1 };
            let mut args = vec![addr, read(self, src)];
            if matches!(aop, AtomicOp::CmpSwap | AtomicOp::FCmpSwap) {
                args.push(read(self, src + w));
            }
            args.push(exec);
            let ty = if wide { Type::U64 } else { Type::U32 };
            let old = self.b().emit(Op::GlobalAtomic { pc: ins.pc, op: aop, flat }, ty, args);
            if c.glc {
                if wide {
                    let (lo, hi) = self.b().split_u64(old);
                    self.write_vgpr(c.destination_vector_register, lo);
                    self.write_vgpr(c.destination_vector_register + 1, hi);
                } else {
                    self.write_vgpr(c.destination_vector_register, old);
                }
            }
        }
        Ok(())
    }

    fn finish_load(&mut self, vdst: u32, v: Value, n: u32, d16: bool, high: bool) {
        if d16 {
            let old = self.vgpr(vdst);
            let o = self.b().u32(if high { 16 } else { 0 });
            let w = self.b().u32(16);
            let merged = self.b().bfi(old, v, o, w);
            self.write_vgpr(vdst, merged);
        } else {
            self.write_dwords(vdst, v, n);
        }
    }

    fn store_data(&mut self, vdata: u32, n: u32, high: bool) -> Value {
        if high {
            let v = self.vgpr(vdata);
            let sixteen = self.b().u32(16);
            return self.b().lshr(v, sixteen);
        }
        self.data_vector(vdata, n)
    }

    // ---------------------------------------------------------------
    // Images
    // ---------------------------------------------------------------

    pub(crate) fn translate_image(&mut self, ins: &Instruction) -> Result<()> {
        use Opcode::*;
        if let Some(Control::RayIntersect(r)) = &ins.control {
            let r = r.clone();
            let bvh64 = ins.opcode == ImageBvh64IntersectRay;
            let count = if bvh64 { 2 } else { 1 } + 1 + 3 + if r.a16 { 3 } else { 6 };
            let t0: Vec<Value> = (0..4).map(|k| self.sgpr(r.scalar_resource + k)).collect();
            let t0 = self.b().construct(&t0);
            let undef4 = self.func.undef(Type::U32X4);
            let mut args = vec![t0, undef4, undef4];
            for k in 0..count {
                let reg = r.address_register(k);
                args.push(self.vgpr(reg));
            }
            let exec = self.exec_operand();
            args.push(exec);
            let attr = ImageAttr {
                pc: ins.pc,
                op: ImageOp::BvhIntersect { bvh64 },
                flags: SampleFlags::default(),
                dim: 0,
                dmask: 0xF,
                a16: r.a16,
                d16: false,
                glc: false,
                slc: false,
                address_dwords: count as u8,
                data_dwords: 0,
            };
            let v = self.b().emit(Op::Image(attr), Type::U32X4, args);
            self.write_dwords(r.vector_data, v, 4);
            return Ok(());
        }
        let Some(Control::Image(c)) = &ins.control else { return self.invalid(ins, "image without control") };
        let c = c.clone();
        if ins.opcode == ImageMsaaLoad {
            return self.translate_msaa_load(ins, &c);
        }
        let name = ins.opcode.name();
        let (op, mut flags) = image_kind(ins.opcode, &name);
        flags.mip |= matches!(ins.opcode, ImageLoadMip | ImageStoreMip);
        if matches!(
            ins.opcode,
            ImageStoreBy2 | ImageStoreBy4 | ImageStoreMipBy2 | ImageStoreMipBy4 | ImageStorePck2 | ImageStorePck4 | ImageStoreMipPck2 | ImageStoreMipPck4
        ) {
            return self.unsupported(ins, "MIMG BY2/BY4 and PCK2/PCK4 stores are not implemented");
        }
        if ins.opcode == ImageGather8hPck {
            return self.unsupported(ins, "image_gather8h_pck is not implemented");
        }
        let Some(op) = op else { return self.unsupported(ins, "image operation") };
        let r128 = (ins.words[0] >> 15) & 1 != 0;
        if flags.packed && c.d16 {
            return self.invalid(ins, "MIMG PCK forms have no 16-bit data");
        }
        if flags.elements != 0 {
            // As measured: 2D, 32-bit addresses and data, a full T#; BY
            // reads every element's components (BY2 0x3 or 0xF, BY4 0xF),
            // PCK2 / PCK4 one dword.
            let dmask = c.dmask & 0xF;
            let dmask_ok = if flags.packed { dmask == 1 } else { dmask == 0xF || (dmask == 0x3 && flags.elements == 2) };
            if !dmask_ok || c.a16 || c.d16 || c.dimension != 1 || r128 {
                return self.invalid(ins, "MIMG BY2/BY4 and PCK2/PCK4 need a data mask covering their elements, 32-bit addresses and data, a 2D image and a full T#");
            }
        }
        let address_dwords = image_address_dwords(op, &flags, c.dimension, c.a16);
        let dmask_count = (c.dmask & 0xF).count_ones().max(1);
        // Image atomics take one operand (DMASK 0x1) or a 64-bit one (0x3);
        // compare-swap a pair of either (0x3 / 0xF).
        let atomic_dwords = match op {
            ImageOp::Atomic(a) => {
                let pair = matches!(a, AtomicOp::CmpSwap | AtomicOp::FCmpSwap);
                let (mask32, mask64) = if pair { (0x3, 0xF) } else { (0x1, 0x3) };
                match c.dmask & 0xF {
                    m if m == mask32 => 1,
                    m if m == mask64 => {
                        if !matches!(
                            a,
                            AtomicOp::Swap | AtomicOp::CmpSwap | AtomicOp::Add | AtomicOp::Sub | AtomicOp::SMin | AtomicOp::UMin | AtomicOp::SMax | AtomicOp::UMax | AtomicOp::And | AtomicOp::Or | AtomicOp::Xor
                        ) {
                            return self.unsupported(ins, "64-bit image atomic");
                        }
                        2
                    }
                    _ => return self.unsupported(ins, "image atomic DMASK"),
                }
            }
            _ => 1,
        };
        let result_dwords = match op {
            // The packed horizontal gather returns the dwords its DMASK
            // selects of the four texels' bit stream.
            ImageOp::Gather4 if flags.packed => dmask_count,
            ImageOp::Gather4 => 4,
            ImageOp::Store => 0,
            ImageOp::Atomic(_) => {
                if c.glc {
                    atomic_dwords
                } else {
                    0
                }
            }
            _ => dmask_count,
        };
        let result_dwords = if c.d16 && result_dwords > 0 && !matches!(op, ImageOp::Atomic(_)) { result_dwords.div_ceil(2) } else { result_dwords };
        let data_dwords = match op {
            ImageOp::Store => {
                if c.d16 {
                    dmask_count.div_ceil(2)
                } else {
                    dmask_count
                }
            }
            ImageOp::Atomic(AtomicOp::CmpSwap | AtomicOp::FCmpSwap) => 2 * atomic_dwords,
            ImageOp::Atomic(_) => atomic_dwords,
            _ => 0,
        };
        let t_lo: Vec<Value> = (0..4).map(|k| self.sgpr(c.scalar_resource + k)).collect();
        let t_hi: Vec<Value> = (4..8).map(|k| self.sgpr(c.scalar_resource + k)).collect();
        let t_lo = self.b().construct(&t_lo);
        let t_hi = self.b().construct(&t_hi);
        let sampler = if matches!(op, ImageOp::Sample | ImageOp::Gather4 | ImageOp::GetLod) {
            let s: Vec<Value> = (0..4).map(|k| self.sgpr(c.scalar_sampler + k)).collect();
            self.b().construct(&s)
        } else {
            self.func.undef(Type::U32X4)
        };
        let mut args = vec![t_lo, t_hi, sampler];
        for k in 0..address_dwords {
            let reg = c.address_register(k as usize);
            args.push(self.vgpr(reg));
        }
        if flags.derivatives && name.contains("Cd") {
            self.coarse_derivatives(&mut args[3..], &flags, c.dimension);
        }
        for k in 0..data_dwords {
            args.push(self.vgpr(c.vector_data + k));
        }
        let exec = self.exec_operand();
        args.push(exec);
        let attr = ImageAttr {
            pc: ins.pc,
            op,
            flags,
            dim: c.dimension as u8,
            dmask: c.dmask as u8,
            a16: c.a16,
            d16: c.d16,
            glc: c.glc,
            slc: c.slc,
            address_dwords: address_dwords as u8,
            data_dwords: data_dwords as u8,
        };
        let ty = match op {
            ImageOp::Atomic(_) => Type::u32xn(atomic_dwords),
            _ if result_dwords == 0 => Type::VOID,
            _ => Type::u32xn(result_dwords),
        };
        let v = self.b().emit(Op::Image(attr), ty, args);
        if result_dwords > 0 {
            self.write_dwords(c.vector_data, v, result_dwords);
        }
        Ok(())
    }

    /// `image_msaa_load`: one component (a single DMASK bit) of the four
    /// samples from the addressed sample index on, one dword each, read
    /// like `image_load` of a multisampled image with the sample index
    /// raised by 0..3.
    fn translate_msaa_load(&mut self, ins: &Instruction, c: &ImageControl) -> Result<()> {
        if !matches!(c.dimension, 6 | 7) {
            return self.invalid(ins, "image_msaa_load of an image dimension without samples");
        }
        if (c.dmask & 0xF).count_ones() != 1 {
            return self.invalid(ins, "image_msaa_load reads one component (a single DMASK bit)");
        }
        if c.d16 {
            return self.unsupported(ins, "image_msaa_load with D16 data");
        }
        let flags = SampleFlags::default();
        let address_dwords = image_address_dwords(ImageOp::Load, &flags, c.dimension, c.a16);
        // The sample index is the last address component: x, y, [slice,]
        // sample.
        let sample = if c.dimension == 6 { 2 } else { 3 };
        let (sample_dword, sample_shift) = if c.a16 { (sample / 2, 16 * (sample % 2)) } else { (sample, 0) };
        let t_lo: Vec<Value> = (0..4).map(|k| self.sgpr(c.scalar_resource + k)).collect();
        let t_hi: Vec<Value> = (4..8).map(|k| self.sgpr(c.scalar_resource + k)).collect();
        let t_lo = self.b().construct(&t_lo);
        let t_hi = self.b().construct(&t_hi);
        let sampler = self.func.undef(Type::U32X4);
        let address: Vec<Value> = (0..address_dwords).map(|k| self.vgpr(c.address_register(k as usize))).collect();
        let exec = self.exec_operand();
        let attr = ImageAttr {
            pc: ins.pc,
            op: ImageOp::Load,
            flags,
            dim: c.dimension as u8,
            dmask: c.dmask as u8,
            a16: c.a16,
            d16: false,
            glc: c.glc,
            slc: c.slc,
            address_dwords: address_dwords as u8,
            data_dwords: 0,
        };
        for fragment in 0..4u32 {
            let mut args = vec![t_lo, t_hi, sampler];
            for (k, &a) in address.iter().enumerate() {
                if k as u32 == sample_dword && fragment != 0 {
                    let step = self.b().u32(fragment << sample_shift);
                    args.push(self.b().iadd(a, step));
                } else {
                    args.push(a);
                }
            }
            args.push(exec);
            let v = self.b().emit(Op::Image(attr), Type::U32, args);
            self.write_vgpr(c.vector_data + fragment, v);
        }
        Ok(())
    }

    // ---------------------------------------------------------------
    // Export and interpolation
    // ---------------------------------------------------------------

    pub(crate) fn translate_export(&mut self, ins: &Instruction) -> Result<()> {
        let Some(Control::Export(e)) = &ins.control else { return self.invalid(ins, "export without control") };
        let e = e.clone();
        let mut args: Vec<Value> = ins.sources.iter().map(|o| o.value).collect::<Vec<_>>().into_iter().map(|r| self.vgpr(r)).collect();
        let exec = self.exec_operand();
        args.push(exec);
        let attr = ExportAttr { target: e.target as u8, enable: e.enable_mask as u8, compressed: e.compressed, done: e.done, valid_mask: e.valid_mask };
        self.b().emit(Op::Export(attr), Type::VOID, args);
        Ok(())
    }

    pub(crate) fn translate_interp(&mut self, ins: &Instruction) -> Result<()> {
        use Opcode::*;
        // VINTRP: the coordinate VGPR (the parameter of the mov form) in
        // SRC, attribute and channel in the control.
        // VOP3 (v_interp_*_f32_e64, opcodes 0x200..0x202): attribute (5:0)
        // and channel (7:6) in SRC0, the coordinate VGPR (the parameter of
        // the mov form) in SRC1, SRC2 the inline constant 0; abs/neg of
        // SRC1, clamp and omod apply.
        let (attr, chan, source, coordinate_abs, coordinate_neg, m) = if ins.encoding == Encoding::Vop3 {
            if ins.words.len() < 2 {
                return self.invalid(ins, "VOP3 interpolation without its second word");
            }
            let (w0, w1) = (ins.words[0], ins.words[1]);
            let (src0, src1, src2) = (w1 & 0x1FF, (w1 >> 9) & 0x1FF, (w1 >> 18) & 0x1FF);
            let (abs, neg, op_sel) = ((w0 >> 8) & 7, w1 >> 29, (w0 >> 11) & 0xF);
            let moves = ins.opcode == VInterpMovF32;
            if src2 != 128 {
                return self.unsupported(ins, "VOP3 interpolation third source is not the inline constant 0");
            }
            if abs & 5 != 0 || neg & 5 != 0 || op_sel != 0 || src0 >> 8 != 0 || (moves && (abs | neg) != 0) {
                return self.unsupported(ins, "VOP3 interpolation modifiers");
            }
            if moves && src1 >= 3 {
                return self.unsupported(ins, "VOP3 interpolation source is not an interpolation parameter");
            }
            if !moves && src1 < 256 {
                return self.unsupported(ins, "VOP3 interpolation source is not a vector register");
            }
            let source = if moves { src1 } else { src1 - 256 };
            ((src0 & 0x3F) as u8, ((src0 >> 6) & 3) as u8, source, abs & 2 != 0, neg & 2 != 0, Mods::of(ins))
        } else {
            let Some(Control::Interpolation(c)) = &ins.control else { return self.invalid(ins, "interp without control") };
            (c.attribute as u8, c.channel as u8, ins.sources[0].value, false, false, Mods::default())
        };
        let d = self.vdst(ins)?;
        let param = |t: &mut Self, p: InterpParam| t.b().emit(Op::InterpParam { attr, chan, param: p }, Type::F32, vec![]);
        let coordinate = |t: &mut Self| {
            let c = t.vgpr(source);
            let mut c = t.b().bitcast(c, Type::F32);
            if coordinate_abs {
                c = t.b().fabs(c);
            }
            if coordinate_neg {
                c = t.b().fneg(c);
            }
            c
        };
        let r = match ins.opcode {
            VInterpP1F32 => {
                let i = coordinate(self);
                let p10 = param(self, InterpParam::P10);
                let p0 = param(self, InterpParam::P0);
                self.b().fma(i, p10, p0)
            }
            VInterpP2F32 => {
                let j = coordinate(self);
                let acc = self.vgpr(d);
                let acc = self.b().bitcast(acc, Type::F32);
                let p20 = param(self, InterpParam::P20);
                self.b().fma(j, p20, acc)
            }
            _ => {
                let p = match source & 3 {
                    0 => InterpParam::P10,
                    1 => InterpParam::P20,
                    _ => InterpParam::P0,
                };
                param(self, p)
            }
        };
        if ins.encoding == Encoding::Vop3 {
            return self.write_f32_result(ins, &m, r);
        }
        let bits = self.b().bitcast(r, Type::U32);
        self.write_vgpr(d, bits);
        Ok(())
    }
}

/// Operation and modifiers of an image opcode (from its name).
fn image_kind(op: Opcode, name: &str) -> (Option<ImageOp>, SampleFlags) {
    use Opcode::*;
    let mut f = SampleFlags::default();
    let kind = match op {
        ImageLoad | ImageLoadMip => Some(ImageOp::Load),
        ImageLoadPck | ImageLoadPckSgn | ImageLoadMipPck | ImageLoadMipPckSgn => {
            f.packed = true;
            f.packed_signed = matches!(op, ImageLoadPckSgn | ImageLoadMipPckSgn);
            f.mip = matches!(op, ImageLoadMipPck | ImageLoadMipPckSgn);
            Some(ImageOp::Load)
        }
        ImageStorePck | ImageStoreMipPck => {
            f.packed = true;
            f.mip = op == ImageStoreMipPck;
            Some(ImageOp::Store)
        }
        ImageLoadBy2 | ImageLoadBy4 | ImageLoadMipBy2 | ImageLoadMipBy4 => {
            f.elements = if matches!(op, ImageLoadBy4 | ImageLoadMipBy4) { 4 } else { 2 };
            f.mip = matches!(op, ImageLoadMipBy2 | ImageLoadMipBy4);
            Some(ImageOp::Load)
        }
        ImageLoadPck2 | ImageLoadPck4 | ImageLoadMipPck2 | ImageLoadMipPck4 => {
            f.packed = true;
            f.elements = if matches!(op, ImageLoadPck4 | ImageLoadMipPck4) { 4 } else { 2 };
            f.mip = matches!(op, ImageLoadMipPck2 | ImageLoadMipPck4);
            Some(ImageOp::Load)
        }
        ImageGather4hPck => {
            f.packed = true;
            f.horizontal = true;
            Some(ImageOp::Gather4)
        }
        // The BY / PCK2 / PCK4 stores and gather8h_pck: not implemented
        // (no measured behaviour).
        ImageStoreBy2 | ImageStoreBy4 | ImageStoreMipBy2 | ImageStoreMipBy4 | ImageStorePck2 | ImageStorePck4 | ImageStoreMipPck2 | ImageStoreMipPck4
        | ImageGather8hPck => None,
        ImageStore | ImageStoreMip => Some(ImageOp::Store),
        ImageGetResinfo => Some(ImageOp::GetResinfo),
        ImageGetLod => Some(ImageOp::GetLod),
        ImageAtomicSwap => Some(ImageOp::Atomic(AtomicOp::Swap)),
        ImageAtomicCmpswap => Some(ImageOp::Atomic(AtomicOp::CmpSwap)),
        ImageAtomicAdd => Some(ImageOp::Atomic(AtomicOp::Add)),
        ImageAtomicSub => Some(ImageOp::Atomic(AtomicOp::Sub)),
        ImageAtomicSmin => Some(ImageOp::Atomic(AtomicOp::SMin)),
        ImageAtomicUmin => Some(ImageOp::Atomic(AtomicOp::UMin)),
        ImageAtomicSmax => Some(ImageOp::Atomic(AtomicOp::SMax)),
        ImageAtomicUmax => Some(ImageOp::Atomic(AtomicOp::UMax)),
        ImageAtomicAnd => Some(ImageOp::Atomic(AtomicOp::And)),
        ImageAtomicOr => Some(ImageOp::Atomic(AtomicOp::Or)),
        ImageAtomicXor => Some(ImageOp::Atomic(AtomicOp::Xor)),
        ImageAtomicInc => Some(ImageOp::Atomic(AtomicOp::Inc)),
        ImageAtomicDec => Some(ImageOp::Atomic(AtomicOp::Dec)),
        ImageAtomicFcmpswap => Some(ImageOp::Atomic(AtomicOp::FCmpSwap)),
        ImageAtomicFmin => Some(ImageOp::Atomic(AtomicOp::FMin)),
        ImageAtomicFmax => Some(ImageOp::Atomic(AtomicOp::FMax)),
        _ if name.starts_with("ImageSample") || name.starts_with("ImageGather4") => {
            let gather = name.starts_with("ImageGather4");
            f.g16 = name.ends_with("G16");
            let rest = name.trim_start_matches("ImageSample").trim_start_matches("ImageGather4").trim_end_matches("G16");
            // Parse suffix tokens: C, Cl, D, Cd, L, B, Lz, O, A, H.
            let mut s = rest;
            while !s.is_empty() {
                let token = if s.starts_with("Cl") {
                    "Cl"
                } else if s.starts_with("Cd") {
                    "Cd"
                } else if s.starts_with("Lz") {
                    "Lz"
                } else {
                    &s[..1]
                };
                match token {
                    "C" => f.compare = true,
                    "Cl" => f.clamp = true,
                    "D" | "Cd" => f.derivatives = true,
                    "L" => f.lod = true,
                    "B" => f.bias = true,
                    "Lz" => f.lod_zero = true,
                    "O" => f.offset = true,
                    "H" => f.horizontal = true,
                    _ => {}
                }
                s = &s[token.len()..];
            }
            Some(if gather { ImageOp::Gather4 } else { ImageOp::Sample })
        }
        _ => None,
    };
    (kind, f)
}

/// Number of address dwords of an image operation (non-NSA layout).
fn image_address_dwords(op: ImageOp, f: &SampleFlags, dim: u32, a16: bool) -> u32 {
    // Coordinate count and derivative dimensions by MIMG DIM.
    let (coords, deriv_dims): (u32, u32) = match dim {
        0 => (1, 1),
        1 => (2, 2),
        2 => (3, 3),
        3 => (3, 2),
        4 => (2, 1),
        5 => (3, 2),
        6 => (3, 2),
        _ => (4, 2),
    };
    match op {
        ImageOp::GetResinfo => 1,
        ImageOp::BvhIntersect { .. } => 0,
        ImageOp::Load | ImageOp::Store | ImageOp::Atomic(_) => {
            let n = coords + f.mip as u32;
            if a16 {
                n.div_ceil(2)
            } else {
                n
            }
        }
        ImageOp::Sample | ImageOp::Gather4 | ImageOp::GetLod => {
            let full = f.offset as u32 + f.bias as u32 + f.compare as u32;
            // 16-bit derivatives (G16 only: A16 leaves them 32-bit) pack
            // each derivative vector into its own dwords (an odd last
            // component is padded).
            let derivatives = match (f.derivatives, f.g16) {
                (false, _) => 0,
                (true, false) => 2 * deriv_dims,
                (true, true) => 2 * deriv_dims.div_ceil(2),
            };
            let rest = coords + f.lod as u32 + f.clamp as u32;
            full + derivatives + if a16 { rest.div_ceil(2) } else { rest }
        }
    }
}
