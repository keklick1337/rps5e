// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! MODE round fields (MODE[1:0] f32, MODE[3:2] f16/f64) through the control
//! flow graph: `s_round_mode`, `s_setreg_imm32_b32` and `s_setreg_b32` of
//! the round fields change them at run time. Each block starts from the
//! join of its predecessors' exit states (bits that differ between paths,
//! or come from an SGPR, are unknown); an instruction runs in the declared
//! FLOAT_MODE with the round bits of the state at that point.
//!
//! Which instructions read which round bits ([`rounding_use`]) and what is
//! refused: an instruction that reads unknown bits; an f16 conversion in a
//! round mode set at run time when the declared FLOAT_MODE flushes f16
//! denormals; an instruction without a directed-rounding model in a
//! directed mode set at run time.

use crate::ir_types::Instruction;
use crate::opcode::Opcode;

/// MODE[3:0]: both round fields.
pub const ROUND_BITS: u8 = 0xF;
/// MODE[1:0]: the f32 round mode.
pub const ROUND32_BITS: u8 = 0x3;
/// MODE[3:2]: the f16/f64 round mode.
pub const ROUND1664_BITS: u8 = 0xC;

/// The round fields at a program point: `unknown` bits differ between the
/// paths reaching it or were written from an SGPR (`value` has them clear).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RoundState {
    pub value: u8,
    pub unknown: u8,
}

/// A write of round bits: `mask` bits become `value` (`known`) or unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundWrite {
    pub mask: u8,
    pub value: u8,
    pub known: bool,
}

/// The MODE round-field mask of a hwreg(id, offset, size) operand, when it
/// names MODE (id 1) within bits 3:0.
pub fn round_field(simm16: u32) -> Option<(u8, u32, u32)> {
    if simm16 & 0x3F != 1 {
        return None;
    }
    let offset = (simm16 >> 6) & 0x1F;
    let size = ((simm16 >> 11) & 0x1F) + 1;
    if offset + size > 4 {
        return None;
    }
    Some(((((1u32 << size) - 1) << offset) as u8, offset, size))
}

/// The round-field write of `ins`, if it is one.
pub fn decode_write(ins: &Instruction) -> Option<RoundWrite> {
    match ins.opcode {
        Opcode::SRoundMode => Some(RoundWrite { mask: ROUND_BITS, value: (ins.words[0] & 0xF) as u8, known: true }),
        Opcode::SSetregImm32B32 => {
            let (mask, offset, size) = round_field(ins.sources[1].value)?;
            let written = (ins.sources[0].value as u64 & ((1u64 << size) - 1)) as u32;
            Some(RoundWrite { mask, value: (written << offset) as u8, known: true })
        }
        Opcode::SSetregB32 => {
            let (mask, _, _) = round_field(ins.sources[1].value)?;
            Some(RoundWrite { mask, value: 0, known: false })
        }
        _ => None,
    }
}

pub fn apply(state: RoundState, write: RoundWrite) -> RoundState {
    let mask = write.mask & ROUND_BITS;
    if write.known {
        RoundState { value: (state.value & !mask) | (write.value & mask), unknown: state.unknown & !mask }
    } else {
        RoundState { value: state.value & !mask, unknown: state.unknown | mask }
    }
}

pub fn join(a: RoundState, b: RoundState) -> RoundState {
    let unknown = a.unknown | b.unknown | (a.value ^ b.value);
    RoundState { value: a.value & !unknown & ROUND_BITS, unknown }
}

/// How an instruction depends on the round fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundUse {
    /// Modelled in the f32 round mode (integer conversions, v_ldexp_f32).
    Directed32,
    /// Modelled in the f16 round mode (the f16 conversions, v_ldexp_f16).
    Directed16,
    /// Reads the round mode and models what it supports itself.
    SelfChecked,
    /// Rounds to nearest even only.
    Unmodelled,
}

/// The round bits `opcode` reads and how.
pub fn rounding_use(opcode: Opcode) -> Option<(u8, RoundUse)> {
    use Opcode::*;
    Some(match opcode {
        VCvtF32I32 | VCvtF32U32 | VLdexpF32 => (ROUND32_BITS, RoundUse::Directed32),
        VCvtF16F32 | VCvtF16U16 | VCvtF16I16 | VLdexpF16 => (ROUND1664_BITS, RoundUse::Directed16),
        VFmaMixloF16 | VFmaMixhiF16 | VInterpP1llF16 | VInterpP1lvF16 | VInterpP2F16 => (ROUND_BITS, RoundUse::SelfChecked),
        VPkAddF16 | VPkMulF16 | VPkFmaF16 | VPkFmacF16 => (ROUND1664_BITS, RoundUse::SelfChecked),
        VAddF32 | VSubF32 | VSubrevF32 | VMulF32 | VMulLegacyF32 | VMadF32 | VMacF32 | VMadLegacyF32 | VMacLegacyF32 | VMullitF32 | VFmaF32 | VMadMkF32 | VMadAkF32
        | VFractF32 | VCvtPkU8F32 | VCvtF32F64 | VDivScaleF32 | VDivFmasF32 | VDivFixupF32 | VCubemaF32 | VDot2F32F16 | VDot2cF32F16 | VInterpP1F32 | VInterpP2F32 => {
            (ROUND32_BITS, RoundUse::Unmodelled)
        }
        VAddF16 | VSubF16 | VSubrevF16 | VMulF16 | VFmacF16 | VFmaMkF16 | VFmaAkF16 | VFmaF16 | VDivFixupF16 | VFractF16 | VFmaF64 | VDivFmasF64 | VAddF64 | VMulF64
        | VLdexpF64 | VFractF64 => (ROUND1664_BITS, RoundUse::Unmodelled),
        _ => return None,
    })
}

/// Whether `value` sets a directed round mode (other than the declared
/// one) in a field of `fields`.
pub fn sets_directed(value: u8, initial: u8, fields: u8) -> bool {
    [ROUND32_BITS, ROUND1664_BITS].iter().any(|&group| {
        let used = fields & group;
        used != 0 && value & group != initial & group && value & group != 0
    })
}

/// The round state at the entry of each block of `cfg` (`initial` is the
/// declared FLOAT_MODE): a forward data-flow fixpoint over the successors.
/// Blocks not reached from the entry start in the declared state.
pub fn entry_states(cfg: &crate::cfg::Cfg, program: &crate::ir_types::Program, initial: u8) -> Vec<RoundState> {
    let start = RoundState { value: initial & ROUND_BITS, unknown: 0 };
    let n = cfg.blocks.len();
    let mut entries: Vec<Option<RoundState>> = vec![None; n];
    if n == 0 {
        return Vec::new();
    }
    // A program without round-field writes keeps the declared state.
    if !program.instructions.iter().any(|i| decode_write(i).is_some()) {
        return vec![start; n];
    }
    entries[cfg.entry] = Some(start);
    let mut pending = std::collections::VecDeque::from([cfg.entry]);
    while let Some(b) = pending.pop_front() {
        let block = &cfg.blocks[b];
        let mut exit = entries[b].unwrap_or(start);
        for ins in &program.instructions[block.first..block.end] {
            if let Some(w) = decode_write(ins) {
                exit = apply(exit, w);
            }
        }
        for s in block.exit.successors() {
            let next = match entries[s] {
                None => exit,
                Some(existing) => {
                    let joined = join(existing, exit);
                    if joined == existing {
                        continue;
                    }
                    joined
                }
            };
            entries[s] = Some(next);
            pending.push_back(s);
        }
    }
    entries.into_iter().map(|e| e.unwrap_or(start)).collect()
}
