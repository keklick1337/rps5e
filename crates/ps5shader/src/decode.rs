// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Instruction decoder: encoding detection, opcode tables, instruction
//! lengths and operand / control extraction.

use crate::ir_types::*;
use crate::opcode::{FlatOp, FlatSegment, Opcode};
use std::fmt;

/// Why a word could not be decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstructionError {
    /// The opcode field of a known encoding has no entry in its table.
    UnknownOpcode { encoding: Encoding, opcode: u32, word: u32 },
    /// FLAT-family word with segment 3 or an unnamed operation.
    UnknownFlat { segment: u32, opcode: u32, word: u32 },
    /// FLAT-family operation whose segment or address registers the
    /// hardware does not accept.
    InvalidFlat { reason: &'static str, word: u32 },
    /// DS operation with an encoding the hardware does not define.
    InvalidDs { reason: &'static str, word: u32 },
    /// The top bits match no encoding.
    UnknownEncoding { word: u32 },
    /// The second dword of a two-dword encoding could not be read.
    MissingExtraWord { encoding: Encoding },
}

impl fmt::Display for InstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InstructionError::UnknownOpcode { encoding, opcode, word } => {
                write!(f, "unknown-{} op=0x{opcode:02X} word=0x{word:08X}", format!("{encoding:?}").to_lowercase())
            }
            InstructionError::UnknownFlat { segment, opcode, word } => {
                write!(f, "unknown-flat segment=0x{segment:X} op=0x{opcode:02X} word=0x{word:08X}")
            }
            InstructionError::InvalidFlat { reason, word } => write!(f, "invalid-flat {reason} word=0x{word:08X}"),
            InstructionError::InvalidDs { reason, word } => write!(f, "invalid-ds {reason} word=0x{word:08X}"),
            InstructionError::UnknownEncoding { word } => write!(f, "unknown-top word=0x{word:08X}"),
            InstructionError::MissingExtraWord { encoding } => {
                write!(f, "{}-extra-read-failed", format!("{encoding:?}").to_lowercase())
            }
        }
    }
}

/// Encoding, opcode and size of the instruction starting with `word`.
/// `extra` is the following dword, needed for VOP3/VOP3P/MUBUF/MTBUF.
pub fn decode_opcode(word: u32, extra: Option<u32>) -> Result<(Encoding, Opcode, usize), InstructionError> {
    if word & 0x8000_0000 == 0 {
        let encoding = match (word >> 25) & 0x3F {
            0x3E => Encoding::Vopc,
            0x3F => Encoding::Vop1,
            _ => Encoding::Vop2,
        };
        let (op, size) = match encoding {
            Encoding::Vopc => decode_vopc(word)?,
            Encoding::Vop1 => decode_vop1(word)?,
            _ => decode_vop2(word)?,
        };
        return Ok((encoding, op, size));
    }
    if word & 0xF800_0000 == 0xC000_0000 {
        let (op, size) = decode_smrd(word)?;
        return Ok((Encoding::Smrd, op, size));
    }
    if word & 0xC000_0000 == 0x8000_0000 {
        return decode_sop(word);
    }
    // gfx10 VOP3P has its own 9-bit prefix 0b110011000 (top byte 0xCC);
    // match it before the coarse switch so packed math is not misread.
    if word & 0xFF80_0000 == 0xCC00_0000 {
        let extra = extra.ok_or(InstructionError::MissingExtraWord { encoding: Encoding::Vop3p })?;
        let (op, size) = decode_vop3p(word, extra);
        return Ok((Encoding::Vop3p, op, size));
    }
    match word >> 26 {
        0x33 | 0x3D => {
            let op = decode_smem(word)?;
            Ok((Encoding::Smem, op, 2))
        }
        0x32 => {
            let op = decode_vintrp(word)?;
            Ok((Encoding::Vintrp, op, 1))
        }
        0x34 | 0x35 => {
            let extra = extra.ok_or(InstructionError::MissingExtraWord { encoding: Encoding::Vop3 })?;
            let (op, size) = decode_vop3(word, extra)?;
            Ok((Encoding::Vop3, op, size))
        }
        0x36 => {
            let op = decode_ds(word)?;
            if let Some(extra) = extra {
                validate_ds(op, word, extra)?;
            }
            Ok((Encoding::Ds, op, 2))
        }
        0x37 => {
            let extra = extra.ok_or(InstructionError::MissingExtraWord { encoding: Encoding::Flat })?;
            Ok((Encoding::Flat, decode_flat(word, extra)?, 2))
        }
        0x38 => {
            let extra = extra.ok_or(InstructionError::MissingExtraWord { encoding: Encoding::Mubuf })?;
            let size = if extra >> 24 == 0xFF { 3 } else { 2 };
            Ok((Encoding::Mubuf, decode_mubuf(word), size))
        }
        0x3A => {
            let extra = extra.ok_or(InstructionError::MissingExtraWord { encoding: Encoding::Mtbuf })?;
            let size = if extra >> 24 == 0xFF { 3 } else { 2 };
            Ok((Encoding::Mtbuf, decode_mtbuf(word, extra), size))
        }
        0x3C => {
            let size = 2 + ((word >> 1) & 3) as usize;
            Ok((Encoding::Mimg, decode_mimg(word)?, size))
        }
        0x3E => Ok((Encoding::Exp, Opcode::Exp, 2)),
        0x3F => Ok((Encoding::Vop3p, Opcode::Vop3pTopRaw((word >> 24) as u8), 2)),
        _ => Err(InstructionError::UnknownEncoding { word }),
    }
}

fn unknown(encoding: Encoding, opcode: u32, word: u32) -> InstructionError {
    InstructionError::UnknownOpcode { encoding, opcode, word }
}

fn decode_sop(word: u32) -> Result<(Encoding, Opcode, usize), InstructionError> {
    match (word >> 23) & 0x7F {
        0x7D => decode_sop1(word).map(|(o, s)| (Encoding::Sop1, o, s)),
        0x7E => decode_sopc(word).map(|(o, s)| (Encoding::Sopc, o, s)),
        0x7F => decode_sopp(word).map(|o| (Encoding::Sopp, o, 1)),
        0x60.. => decode_sopk(word).map(|o| (Encoding::Sopk, o, if o == Opcode::SSetregImm32B32 { 2 } else { 1 })),
        _ => decode_sop2(word).map(|(o, s)| (Encoding::Sop2, o, s)),
    }
}

fn decode_sop1(word: u32) -> Result<(Opcode, usize), InstructionError> {
    use Opcode::*;
    let opcode = (word >> 8) & 0xFF;
    let size = if word & 0xFF == 0xFF { 2 } else { 1 };
    let op = match opcode {
        0x03 => SMovB32,
        0x04 => SMovB64,
        0x05 => SCmovB32,
        0x06 => SCmovB64,
        0x07 => SNotB32,
        0x08 => SNotB64,
        0x09 => SWqmB32,
        0x0A => SWqmB64,
        0x0B => SBrevB32,
        0x0C => SBrevB64,
        0x0D => SBcnt0I32B32,
        0x0E => SBcnt0I32B64,
        0x0F => SBcnt1I32B32,
        0x10 => SBcnt1I32B64,
        0x11 => SFF0I32B32,
        0x12 => SFF0I32B64,
        0x13 => SFF1I32B32,
        0x14 => SFF1I32B64,
        0x15 => SFlbitI32B32,
        0x16 => SFlbitI32B64,
        0x17 => SFlbitI32,
        0x18 => SFlbitI32I64,
        0x19 => SSextI32I8,
        0x1A => SSextI32I16,
        0x1B => SBitset0B32,
        0x1C => SBitset0B64,
        0x1D => SBitset1B32,
        0x1E => SBitset1B64,
        0x1F => SGetpcB64,
        0x20 => SSetpcB64,
        0x21 => SSwappcB64,
        0x22 => SRfeB64,
        0x24 => SAndSaveexecB64,
        0x25 => SOrSaveexecB64,
        0x26 => SXorSaveexecB64,
        0x27 => SAndn2SaveexecB64,
        0x28 => SOrn2SaveexecB64,
        0x29 => SNandSaveexecB64,
        0x2A => SNorSaveexecB64,
        0x2B => SXnorSaveexecB64,
        0x2C => SQuadmaskB32,
        0x2D => SQuadmaskB64,
        0x2E => SMovrelsB32,
        0x2F => SMovrelsB64,
        0x30 => SMovreldB32,
        0x31 => SMovreldB64,
        0x34 => SAbsI32,
        0x39 => SAndn1WrexecB64,
        0x3A => SAndn2WrexecB64,
        0x37 => SAndn1SaveexecB64,
        0x38 => SOrn1SaveexecB64,
        0x3B => SBitreplicateB64B32,
        0x3C => SAndSaveexecB32,
        0x3D => SOrSaveexecB32,
        0x3E => SXorSaveexecB32,
        0x3F => SAndn2SaveexecB32,
        0x40 => SOrn2SaveexecB32,
        0x41 => SNandSaveexecB32,
        0x42 => SNorSaveexecB32,
        0x43 => SXnorSaveexecB32,
        0x44 => SAndn1SaveexecB32,
        0x45 => SOrn1SaveexecB32,
        0x46 => SAndn1WrexecB32,
        0x47 => SAndn2WrexecB32,
        0x49 => SMovrelsd2B32,
        _ => return Err(unknown(Encoding::Sop1, opcode, word)),
    };
    Ok((op, size))
}

fn decode_sop2(word: u32) -> Result<(Opcode, usize), InstructionError> {
    use Opcode::*;
    let opcode = (word >> 23) & 0x7F;
    let size = if word & 0xFF == 0xFF || (word >> 8) & 0xFF == 0xFF { 2 } else { 1 };
    let op = match opcode {
        0x00 => SAddU32,
        0x01 => SSubU32,
        0x02 => SAddI32,
        0x03 => SSubI32,
        0x04 => SAddcU32,
        0x05 => SSubbU32,
        0x06 => SMinI32,
        0x07 => SMinU32,
        0x08 => SMaxI32,
        0x09 => SMaxU32,
        0x0A => SCselectB32,
        0x0B => SCselectB64,
        0x0E => SAndB32,
        0x0F => SAndB64,
        0x10 => SOrB32,
        0x11 => SOrB64,
        0x12 => SXorB32,
        0x13 => SXorB64,
        0x14 => SAndn2B32,
        0x15 => SAndn2B64,
        0x16 => SOrn2B32,
        0x17 => SOrn2B64,
        0x18 => SNandB32,
        0x19 => SNandB64,
        0x1A => SNorB32,
        0x1B => SNorB64,
        0x1C => SXnorB32,
        0x1D => SXnorB64,
        0x1E => SLshlB32,
        0x1F => SLshlB64,
        0x20 => SLshrB32,
        0x21 => SLshrB64,
        0x22 => SAshrI32,
        0x23 => SAshrI64,
        0x24 => SBfmB32,
        0x25 => SBfmB64,
        0x26 => SMulI32,
        0x27 => SBfeU32,
        0x28 => SBfeI32,
        0x29 => SBfeU64,
        0x2A => SBfeI64,
        0x2C => SAbsdiffI32,
        0x2E => SLshl1AddU32,
        0x2F => SLshl2AddU32,
        0x30 => SLshl3AddU32,
        0x31 => SLshl4AddU32,
        0x32 => SPackLlB32B16,
        0x33 => SPackLhB32B16,
        0x34 => SPackHhB32B16,
        0x35 => SMulHiU32,
        0x36 => SMulHiI32,
        // Scalar float forms in the extended SOP2 range: they keep the raw
        // SGPR bits and leave SCC alone.
        0x40 => SAddF32,
        0x41 => SSubF32,
        _ => return Err(unknown(Encoding::Sop2, opcode, word)),
    };
    Ok((op, size))
}

fn decode_sopc(word: u32) -> Result<(Opcode, usize), InstructionError> {
    use Opcode::*;
    let opcode = (word >> 16) & 0x7F;
    let size = if word & 0xFF == 0xFF || (word >> 8) & 0xFF == 0xFF { 2 } else { 1 };
    let op = match opcode {
        0x00 => SCmpEqI32,
        0x01 => SCmpLgI32,
        0x02 => SCmpGtI32,
        0x03 => SCmpGeI32,
        0x04 => SCmpLtI32,
        0x05 => SCmpLeI32,
        0x06 => SCmpEqU32,
        0x07 => SCmpLgU32,
        0x08 => SCmpGtU32,
        0x09 => SCmpGeU32,
        0x0A => SCmpLtU32,
        0x0B => SCmpLeU32,
        0x0C => SBitcmp0B32,
        0x0D => SBitcmp1B32,
        0x0E => SBitcmp0B64,
        0x0F => SBitcmp1B64,
        0x12 => SCmpEqU64,
        0x13 => SCmpLgU64,
        _ => return Err(unknown(Encoding::Sopc, opcode, word)),
    };
    Ok((op, size))
}

fn decode_sopp(word: u32) -> Result<Opcode, InstructionError> {
    use Opcode::*;
    let opcode = (word >> 16) & 0x7F;
    Ok(match opcode {
        0x00 => SNop,
        0x01 => SEndpgm,
        0x02 => SBranch,
        0x04 => SCbranchScc0,
        0x05 => SCbranchScc1,
        0x06 => SCbranchVccz,
        0x07 => SCbranchVccnz,
        0x08 => SCbranchExecz,
        0x09 => SCbranchExecnz,
        0x0A => SBarrier,
        0x03 => SWakeup,
        0x0B => SSetkill,
        0x0C => SWaitcnt,
        0x0E => SSleep,
        0x0D => SSethalt,
        0x0F => SSetprio,
        0x10 => SSendmsg,
        0x11 => SSendmsghalt,
        0x12 => STrap,
        0x16 => STtraceData,
        0x17 => SCbranchCdbgsys,
        0x18 => SCbranchCdbguser,
        0x19 => SCbranchCdbgsysOrUser,
        0x1A => SCbranchCdbgsysAndUser,
        0x1B => SEndpgmSaved,
        0x1E => SEndpgmOrderedPsDone,
        0x22 => SWaitIdle,
        0x13 => SIcacheInv,
        0x14 => SIncperflevel,
        0x15 => SDecperflevel,
        0x1F => SCodeEnd,
        0x20 => SInstPrefetch,
        0x21 => SClause,
        0x23 => SWaitcntDepctr,
        0x24 => SRoundMode,
        0x25 => SDenormMode,
        0x28 => STtracedataImm,
        _ => return Err(unknown(Encoding::Sopp, opcode, word)),
    })
}

fn decode_sopk(word: u32) -> Result<Opcode, InstructionError> {
    use Opcode::*;
    let opcode = ((word >> 23) & 0x7F) - 0x60;
    Ok(match opcode {
        0x00 => SMovkI32,
        0x01 => SVersion,
        0x02 => SCmovkI32,
        0x03 => SCmpkEqI32,
        0x04 => SCmpkLgI32,
        0x05 => SCmpkGtI32,
        0x06 => SCmpkGeI32,
        0x07 => SCmpkLtI32,
        0x08 => SCmpkLeI32,
        0x09 => SCmpkEqU32,
        0x0A => SCmpkLgU32,
        0x0B => SCmpkGtU32,
        0x0C => SCmpkGeU32,
        0x0D => SCmpkLtU32,
        0x0E => SCmpkLeU32,
        0x0F => SAddkI32,
        0x10 => SMulkI32,
        0x12 => SGetregB32,
        0x13 => SSetregB32,
        0x15 => SSetregImm32B32,
        0x16 => SCallB64,
        0x1B => SSubvectorLoopBegin,
        0x1C => SSubvectorLoopEnd,
        // RDNA2 waits for one counter with four SOPK forms; the counter does
        // not change the translated operation.
        0x17..=0x1A => SWaitcnt,
        _ => return Err(unknown(Encoding::Sopk, opcode, word)),
    })
}

/// SDWA (0xF9), DPP16 (0xFA), DPP8 (0xE9/0xEA) or literal (0xFF) src0.
fn vop_src0_extra(src0: u32) -> bool {
    matches!(src0, 0xE9 | 0xEA | 0xF9 | 0xFA | 0xFF)
}

fn decode_vop1(word: u32) -> Result<(Opcode, usize), InstructionError> {
    let opcode = (word >> 9) & 0xFF;
    let size = if vop_src0_extra(word & 0x1FF) { 2 } else { 1 };
    vop1_opcode(opcode).map(|o| (o, size)).ok_or(unknown(Encoding::Vop1, opcode, word))
}

/// VOP1 opcodes (also VOP3 opcodes 0x180 + op).
pub fn vop1_opcode(opcode: u32) -> Option<Opcode> {
    use Opcode::*;
    Some(match opcode {
        0x00 => VNop,
        0x01 => VMovB32,
        0x02 => VReadfirstlaneB32,
        0x03 => VCvtI32F64,
        0x04 => VCvtF64I32,
        0x05 => VCvtF32I32,
        0x06 => VCvtF32U32,
        0x07 => VCvtU32F32,
        0x08 => VCvtI32F32,
        0x0A => VCvtF16F32,
        0x0B => VCvtF32F16,
        0x0C => VCvtRpiI32F32,
        0x0D => VCvtFlrI32F32,
        0x0E => VCvtOffF32I4,
        0x0F => VCvtF32F64,
        0x10 => VCvtF64F32,
        0x11 => VCvtF32Ubyte0,
        0x12 => VCvtF32Ubyte1,
        0x13 => VCvtF32Ubyte2,
        0x14 => VCvtF32Ubyte3,
        0x15 => VCvtU32F64,
        0x16 => VCvtF64U32,
        0x17 => VTruncF64,
        0x18 => VCeilF64,
        0x19 => VRndneF64,
        0x1A => VFloorF64,
        0x1B => VPipeflush,
        0x20 => VFractF32,
        0x21 => VTruncF32,
        0x22 => VCeilF32,
        0x23 => VRndneF32,
        0x24 => VFloorF32,
        0x25 => VExpF32,
        0x27 => VLogF32,
        0x2A => VRcpF32,
        0x2B => VRcpIflagF32,
        0x2E => VRsqF32,
        0x2F => VRcpF64,
        0x31 => VRsqF64,
        0x33 => VSqrtF32,
        0x34 => VSqrtF64,
        0x35 => VSinF32,
        0x36 => VCosF32,
        0x37 => VNotB32,
        0x38 => VBfrevB32,
        0x39 => VFfbhU32,
        0x3A => VFfblB32,
        0x3B => VFfbhI32,
        0x3C => VFrexpExpI32F64,
        0x3D => VFrexpMantF64,
        0x3E => VFractF64,
        0x3F => VFrexpExpI32F32,
        0x40 => VFrexpMantF32,
        0x41 => VClrexcp,
        0x42 => VMovreldB32,
        0x43 => VMovrelsB32,
        0x44 => VMovrelsdB32,
        0x48 => VMovrelsd2B32,
        0x50 => VCvtF16U16,
        0x51 => VCvtF16I16,
        0x52 => VCvtU16F16,
        0x53 => VCvtI16F16,
        0x54 => VRcpF16,
        0x55 => VSqrtF16,
        0x56 => VRsqF16,
        0x57 => VLogF16,
        0x58 => VExpF16,
        0x59 => VFrexpMantF16,
        0x5A => VFrexpExpI16F16,
        0x5B => VFloorF16,
        0x5C => VCeilF16,
        0x5D => VTruncF16,
        0x5E => VRndneF16,
        0x5F => VFractF16,
        0x60 => VSinF16,
        0x61 => VCosF16,
        0x62 => VSatPkU8I16,
        0x63 => VCvtNormI16F16,
        0x64 => VCvtNormU16F16,
        0x65 => VSwapB32,
        0x68 => VSwaprelB32,
        _ => return None,
    })
}

fn decode_vop2(word: u32) -> Result<(Opcode, usize), InstructionError> {
    let opcode = (word >> 25) & 0x3F;
    // FMAMK/FMAAK/MADMK/MADAK carry a literal dword after the opcode word.
    let size = if matches!(opcode, 0x20 | 0x21 | 0x2C | 0x2D | 0x37 | 0x38) || vop_src0_extra(word & 0x1FF) { 2 } else { 1 };
    vop2_opcode(opcode).map(|o| (o, size)).ok_or(unknown(Encoding::Vop2, opcode, word))
}

/// VOP2 opcodes (also VOP3 opcodes 0x100 + op).
pub fn vop2_opcode(opcode: u32) -> Option<Opcode> {
    use Opcode::*;
    Some(match opcode {
        // Some streams use the opcode-zero VOP2 form as padding.
        0x00 => VNop,
        0x01 => VCndmaskB32,
        0x02 => VDot2cF32F16,
        0x03 => VAddF32,
        0x04 => VSubF32,
        0x05 => VSubrevF32,
        0x06 => VMacLegacyF32,
        0x07 => VMulLegacyF32,
        0x08 => VMulF32,
        0x09 => VMulI32I24,
        0x0A => VMulHiI32I24,
        0x0B => VMulU32U24,
        0x0C => VMulHiU32U24,
        0x0D => VDot4cI32I8,
        0x0F => VMinF32,
        0x10 => VMaxF32,
        0x11 => VMinI32,
        0x12 => VMaxI32,
        0x13 => VMinU32,
        0x14 => VMaxU32,
        // 0x15 / 0x17 / 0x19 (v_lshr/ashr/lshl_b32) and 0x22-0x24 (v_bcnt,
        // v_mbcnt_lo/hi) are gfx6-9 VOP2 encodings: gfx10 has only the
        // *rev shifts here and the VOP3 forms of the bit counts.
        0x16 => VLshrrevB32,
        0x18 => VAshrrevI32,
        0x1A => VLshlrevB32,
        0x1B => VAndB32,
        0x1C => VOrB32,
        0x1D => VXorB32,
        0x1E => VXnorB32,
        0x1F => VMacF32,
        0x20 => VMadMkF32,
        0x21 => VMadAkF32,
        0x25 => VAddI32,
        0x26 => VSubI32,
        0x27 => VSubrevI32,
        0x28 => VAddcU32,
        0x29 => VSubbU32,
        0x2A => VSubbrevU32,
        0x2B => VFmacF32,
        0x2C => VFmaMkF32,
        0x2D => VFmaAkF32,
        0x2F => VCvtPkrtzF16F32,
        0x30 => VCvtPkU16U32,
        0x31 => VCvtPkI16I32,
        0x32 => VAddF16,
        0x33 => VSubF16,
        0x34 => VSubrevF16,
        0x35 => VMulF16,
        0x36 => VFmacF16,
        0x37 => VFmaMkF16,
        0x38 => VFmaAkF16,
        0x39 => VMaxF16,
        0x3A => VMinF16,
        0x3B => VLdexpF16,
        0x3C => VPkFmacF16,
        _ => return None,
    })
}

fn decode_vopc(word: u32) -> Result<(Opcode, usize), InstructionError> {
    let opcode = (word >> 17) & 0xFF;
    let size = if vop_src0_extra(word & 0x1FF) { 2 } else { 1 };
    vopc_opcode(opcode).map(|o| (o, size)).ok_or(unknown(Encoding::Vopc, opcode, word))
}

/// VOPC compares, shared with the VOP3 encoding (VOP3 opcodes 0x000-0x0FF).
pub fn vopc_opcode(opcode: u32) -> Option<Opcode> {
    use Opcode::*;
    Some(match opcode {
        0x00 => VCmpFF32,
        0x01 => VCmpLtF32,
        0x02 => VCmpEqF32,
        0x03 => VCmpLeF32,
        0x04 => VCmpGtF32,
        0x05 => VCmpLgF32,
        0x06 => VCmpGeF32,
        0x07 => VCmpOF32,
        0x08 => VCmpUF32,
        0x09 => VCmpNgeF32,
        0x0A => VCmpNlgF32,
        0x0B => VCmpNgtF32,
        0x0C => VCmpNleF32,
        0x0D => VCmpNeqF32,
        0x0E => VCmpNltF32,
        0x0F => VCmpTruF32,
        0x10 => VCmpxFF32,
        0x11 => VCmpxLtF32,
        0x12 => VCmpxEqF32,
        0x13 => VCmpxLeF32,
        0x14 => VCmpxGtF32,
        0x15 => VCmpxLgF32,
        0x16 => VCmpxGeF32,
        0x17 => VCmpxOF32,
        0x18 => VCmpxUF32,
        0x19 => VCmpxNgeF32,
        0x1A => VCmpxNlgF32,
        0x1B => VCmpxNgtF32,
        0x1C => VCmpxNleF32,
        0x1D => VCmpxNeqF32,
        0x1E => VCmpxNltF32,
        0x1F => VCmpxTruF32,
        0x20 => VCmpFF64,
        0x21 => VCmpLtF64,
        0x22 => VCmpEqF64,
        0x23 => VCmpLeF64,
        0x24 => VCmpGtF64,
        0x25 => VCmpLgF64,
        0x26 => VCmpGeF64,
        0x27 => VCmpOF64,
        0x28 => VCmpUF64,
        0x29 => VCmpNgeF64,
        0x2A => VCmpNlgF64,
        0x2B => VCmpNgtF64,
        0x2C => VCmpNleF64,
        0x2D => VCmpNeqF64,
        0x2E => VCmpNltF64,
        0x2F => VCmpTruF64,
        0x30 => VCmpxFF64,
        0x31 => VCmpxLtF64,
        0x32 => VCmpxEqF64,
        0x33 => VCmpxLeF64,
        0x34 => VCmpxGtF64,
        0x35 => VCmpxLgF64,
        0x36 => VCmpxGeF64,
        0x37 => VCmpxOF64,
        0x38 => VCmpxUF64,
        0x39 => VCmpxNgeF64,
        0x3A => VCmpxNlgF64,
        0x3B => VCmpxNgtF64,
        0x3C => VCmpxNleF64,
        0x3D => VCmpxNeqF64,
        0x3E => VCmpxNltF64,
        0x3F => VCmpxTruF64,
        0x8F => VCmpClassF16,
        0x9F => VCmpxClassF16,
        0xA8 => VCmpClassF64,
        0xB8 => VCmpxClassF64,
        0x80 => VCmpFI32,
        0x81 => VCmpLtI32,
        0x82 => VCmpEqI32,
        0x83 => VCmpLeI32,
        0x84 => VCmpGtI32,
        0x85 => VCmpNeI32,
        0x86 => VCmpGeI32,
        0x87 => VCmpTI32,
        0x88 => VCmpClassF32,
        0x89 => VCmpLtI16,
        0x8A => VCmpEqI16,
        0x8B => VCmpLeI16,
        0x8C => VCmpGtI16,
        0x8D => VCmpNeI16,
        0x8E => VCmpGeI16,
        0x98 => VCmpxClassF32,
        0xA9 => VCmpLtU16,
        0xAA => VCmpEqU16,
        0xAB => VCmpLeU16,
        0xAC => VCmpGtU16,
        0xAD => VCmpNeU16,
        0xAE => VCmpGeU16,
        0x99 => VCmpxLtI16,
        0x9A => VCmpxEqI16,
        0x9B => VCmpxLeI16,
        0x9C => VCmpxGtI16,
        0x9D => VCmpxNeI16,
        0x9E => VCmpxGeI16,
        0xB9 => VCmpxLtU16,
        0xBA => VCmpxEqU16,
        0xBB => VCmpxLeU16,
        0xBC => VCmpxGtU16,
        0xBD => VCmpxNeU16,
        0xBE => VCmpxGeU16,
        0x90 => VCmpxFI32,
        0x91 => VCmpxLtI32,
        0x92 => VCmpxEqI32,
        0x93 => VCmpxLeI32,
        0x94 => VCmpxGtI32,
        0x95 => VCmpxNeI32,
        0x96 => VCmpxGeI32,
        0x97 => VCmpxTI32,
        0xA0 => VCmpFI64,
        0xA1 => VCmpLtI64,
        0xA2 => VCmpEqI64,
        0xA3 => VCmpLeI64,
        0xA4 => VCmpGtI64,
        0xA5 => VCmpNeI64,
        0xA6 => VCmpGeI64,
        0xA7 => VCmpTI64,
        0xB0 => VCmpxFI64,
        0xB1 => VCmpxLtI64,
        0xB2 => VCmpxEqI64,
        0xB3 => VCmpxLeI64,
        0xB4 => VCmpxGtI64,
        0xB5 => VCmpxNeI64,
        0xB6 => VCmpxGeI64,
        0xB7 => VCmpxTI64,
        0xC0 => VCmpFU32,
        0xC1 => VCmpLtU32,
        0xC2 => VCmpEqU32,
        0xC3 => VCmpLeU32,
        0xC4 => VCmpGtU32,
        0xC5 => VCmpNeU32,
        0xC6 => VCmpGeU32,
        0xC7 => VCmpTU32,
        0xC8 => VCmpFF16,
        0xC9 => VCmpLtF16,
        0xCA => VCmpEqF16,
        0xCB => VCmpLeF16,
        0xCC => VCmpGtF16,
        0xCD => VCmpLgF16,
        0xCE => VCmpGeF16,
        0xCF => VCmpOF16,
        0xD0 => VCmpxFU32,
        0xD1 => VCmpxLtU32,
        0xD2 => VCmpxEqU32,
        0xD3 => VCmpxLeU32,
        0xD4 => VCmpxGtU32,
        0xD5 => VCmpxNeU32,
        0xD6 => VCmpxGeU32,
        0xD7 => VCmpxTU32,
        0xD8 => VCmpxFF16,
        0xD9 => VCmpxLtF16,
        0xDA => VCmpxEqF16,
        0xDB => VCmpxLeF16,
        0xDC => VCmpxGtF16,
        0xDD => VCmpxLgF16,
        0xDE => VCmpxGeF16,
        0xDF => VCmpxOF16,
        0xE0 => VCmpFU64,
        0xE1 => VCmpLtU64,
        0xE2 => VCmpEqU64,
        0xE3 => VCmpLeU64,
        0xE4 => VCmpGtU64,
        0xE5 => VCmpNeU64,
        0xE6 => VCmpGeU64,
        0xE7 => VCmpTU64,
        0xE8 => VCmpUF16,
        0xE9 => VCmpNgeF16,
        0xEA => VCmpNlgF16,
        0xEB => VCmpNgtF16,
        0xEC => VCmpNleF16,
        0xED => VCmpNeqF16,
        0xEE => VCmpNltF16,
        0xEF => VCmpTruF16,
        0xF0 => VCmpxFU64,
        0xF1 => VCmpxLtU64,
        0xF2 => VCmpxEqU64,
        0xF3 => VCmpxLeU64,
        0xF4 => VCmpxGtU64,
        0xF5 => VCmpxNeU64,
        0xF6 => VCmpxGeU64,
        0xF7 => VCmpxTU64,
        0xF8 => VCmpxUF16,
        0xF9 => VCmpxNgeF16,
        0xFA => VCmpxNlgF16,
        0xFB => VCmpxNgtF16,
        0xFC => VCmpxNleF16,
        0xFD => VCmpxNeqF16,
        0xFE => VCmpxNltF16,
        0xFF => VCmpxTruF16,
        _ => return None,
    })
}

/// VOP3 opcodes using the VOP3B layout (SDST in bits 14:8).
pub fn is_vop3b_opcode(opcode: u32) -> bool {
    matches!(opcode, 0x128 | 0x129 | 0x12A | 0x16D | 0x16E | 0x176 | 0x177 | 0x30F | 0x310 | 0x319)
}

fn vop3_size(extra: u32) -> usize {
    let lit = |s: u32| s == 0xFF;
    if lit(extra & 0x1FF) || lit((extra >> 9) & 0x1FF) || lit((extra >> 18) & 0x1FF) {
        3
    } else {
        2
    }
}

fn decode_vop3(word: u32, extra: u32) -> Result<(Opcode, usize), InstructionError> {
    use Opcode::*;
    let opcode = (word >> 16) & 0x3FF;
    let size = vop3_size(extra);
    if is_vop3b_opcode(opcode) {
        let op = match opcode {
            0x128 => VAddCoCiU32,
            0x129 => VSubCoCiU32,
            0x12A => VSubrevCoCiU32,
            0x30F => VAddCoU32,
            0x310 => VSubCoU32,
            0x319 => VSubrevCoU32,
            0x176 => VMadU64U32,
            0x177 => VMadI64I32,
            0x16D => VDivScaleF32,
            0x16E => VDivScaleF64,
            _ => Vop3bRaw(opcode as u16),
        };
        return Ok((op, size));
    }
    if opcode < 0x100 {
        if let Some(op) = vopc_opcode(opcode) {
            return Ok((op, size));
        }
    }
    let op = match opcode {
        0x140 => VMadLegacyF32,
        0x14D => VLerpU8,
        0x150 => VMullitF32,
        0x15F => VDivFixupF32,
        0x160 => VDivFixupF64,
        0x164 => VAddF64,
        0x166 => VMinF64,
        0x167 => VMaxF64,
        0x168 => VLdexpF64,
        0x16F => VDivFmasF32,
        0x170 => VDivFmasF64,
        0x171 => VMsadU8,
        0x172 => VQsadPkU16U8,
        0x173 => VMqsadPkU16U8,
        0x174 => VTrigPreopF64,
        0x175 => VMqsadU32U8,
        0x301 => VAshrrevI64,
        0x305 => VMulLoU16,
        0x311 => VPackB32F16,
        0x312 => VCvtPknormI16F16,
        0x313 => VCvtPknormU16F16,
        0x340 => VMadU16,
        0x200 => VInterpP1F32,
        0x201 => VInterpP2F32,
        0x202 => VInterpMovF32,
        0x342 => VInterpP1llF16,
        0x343 => VInterpP1lvF16,
        0x344 => VPermB32,
        0x352 => VMin3I16,
        0x353 => VMin3U16,
        0x355 => VMax3I16,
        0x356 => VMax3U16,
        0x358 => VMed3I16,
        0x359 => VMed3U16,
        0x35A => VInterpP2F16,
        0x35E => VMadI16,
        0x35F => VDivFixupF16,
        0x375 => VMadI32I16,
        0x376 => VSubNcI32,
        0x37F => VAddNcI32,
        0x101 => VCndmaskB32,
        0x103 => VAddF32,
        0x104 => VSubF32,
        0x108 => VMulF32,
        0x109 => VMulI32I24,
        0x10F => VMinF32,
        0x110 => VMaxF32,
        0x11F => VMacF32,
        0x12B => VFmacF32,
        0x12F => VCvtPkrtzF16F32,
        0x141 => VMadF32,
        0x142 => VMadI32I24,
        0x143 => VMadU32U24,
        0x144 => VCubeidF32,
        0x145 => VCubescF32,
        0x146 => VCubetcF32,
        0x147 => VCubemaF32,
        0x14A => VBfiB32,
        0x14E => VAlignbitB32,
        0x14F => VAlignbyteB32,
        0x14B => VFmaF32,
        0x14C => VFmaF64,
        0x151 => VMin3F32,
        0x152 => VMin3I32,
        0x153 => VMin3U32,
        0x154 => VMax3F32,
        0x155 => VMax3I32,
        0x156 => VMax3U32,
        0x157 => VMed3F32,
        0x158 => VMed3I32,
        0x159 => VMed3U32,
        0x15A => VSadU8,
        0x15B => VSadHiU8,
        0x15C => VSadU16,
        0x15D => VSadU32,
        0x15E => VCvtPkU8F32,
        0x148 => VBfeU32,
        0x149 => VBfeI32,
        0x165 => VMulF64,
        0x169 => VMulLoU32,
        0x16A => VMulHiU32,
        0x16B => VMulLoI32,
        0x16C => VMulHiI32,
        // VOP3-encoded VOP1 (opcode + 0x180).
        0x184 => VCvtF64I32,
        0x18F => VCvtF32F64,
        0x196 => VCvtF64U32,
        0x1AF => VRcpF64,
        0x1B0 => VRsqF64,
        0x1B4 => VSqrtF64,
        0x1D2 => VCvtU16F16,
        0x1D4 => VRcpF16,
        0x1D5 => VSqrtF16,
        0x1D6 => VRsqF16,
        0x1D7 => VLogF16,
        0x1D8 => VExpF16,
        0x1DB => VFloorF16,
        0x1DC => VCeilF16,
        0x1DD => VTruncF16,
        0x1DE => VRndneF16,
        0x1DF => VFractF16,
        0x1E0 => VSinF16,
        0x1E1 => VCosF16,
        0x303 => VAddNcU16,
        0x34B => VFmaF16,
        0x351 => VMin3F16,
        0x354 => VMax3F16,
        0x357 => VMed3F16,
        0x360 => VReadlaneB32,
        0x361 => VWritelaneB32,
        0x362 => VLdexpF32,
        0x363 => VBfmB32,
        0x364 => VBcntU32B32,
        0x365 => VMbcntLoU32B32,
        0x366 => VMbcntHiU32B32,
        0x368 => VCvtPknormI16F32,
        0x369 => VCvtPknormU16F32,
        0x36A => VCvtPkU16U32,
        0x36B => VCvtPkI16I32,
        0x2FF => VLshlrevB64,
        0x300 => VLshrrevB64,
        0x373 => VMadU32U16,
        0x346 => VLshlAddU32,
        0x345 => VXadU32,
        0x347 => VAddLshlU32,
        0x36D => VAdd3U32,
        0x36F => VLshlOrU32,
        0x178 => VXor3B32,
        0x371 => VAndOrB32,
        0x372 => VOr3U32,
        0x377 => VPermlane16B32,
        0x378 => VPermlanex16B32,
        // gfx10 VOP3-only 16-bit integer ALU.
        0x304 => VSubNcU16,
        0x307 => VLshrrevB16,
        0x308 => VAshrrevI16,
        0x309 => VMaxU16,
        0x30A => VMaxI16,
        0x30B => VMinU16,
        0x30C => VMinI16,
        0x30D => VAddNcI16,
        0x30E => VSubNcI16,
        0x314 => VLshlrevB16,
        0x180..=0x1FF => vop1_opcode(opcode - 0x180).unwrap_or(Vop3Raw(opcode as u16)),
        // MADMK/MADAK/FMAMK/FMAAK have no VOP3 form.
        0x100..=0x13F if !matches!(opcode - 0x100, 0x20 | 0x21 | 0x2C | 0x2D | 0x37 | 0x38) => {
            vop2_opcode(opcode - 0x100).unwrap_or(Vop3Raw(opcode as u16))
        }
        _ => Vop3Raw(opcode as u16),
    };
    Ok((op, size))
}

fn decode_vop3p(word: u32, extra: u32) -> (Opcode, usize) {
    use Opcode::*;
    // Opcode numbers are shared by gfx9 and gfx10; 0x20-0x22 are the fused
    // V_FMA_MIX_* forms on gfx10.
    let opcode = (word >> 16) & 0x7F;
    let op = match opcode {
        0x00 => VPkMadI16,
        0x01 => VPkMulLoU16,
        0x02 => VPkAddI16,
        0x03 => VPkSubI16,
        0x04 => VPkLshlrevB16,
        0x05 => VPkLshrrevB16,
        0x06 => VPkAshrrevI16,
        0x07 => VPkMaxI16,
        0x08 => VPkMinI16,
        0x09 => VPkMadU16,
        0x0A => VPkAddU16,
        0x0B => VPkSubU16,
        0x0C => VPkMaxU16,
        0x0D => VPkMinU16,
        0x13 => VDot2F32F16,
        0x14 => VDot2I32I16,
        0x15 => VDot2U32U16,
        0x16 => VDot4I32I8,
        0x17 => VDot4U32U8,
        0x18 => VDot8I32I4,
        0x19 => VDot8U32U4,
        0x0E => VPkFmaF16,
        0x0F => VPkAddF16,
        0x10 => VPkMulF16,
        0x11 => VPkMinF16,
        0x12 => VPkMaxF16,
        0x20 => VFmaMixF32,
        0x21 => VFmaMixloF16,
        0x22 => VFmaMixhiF16,
        _ => Vop3pRaw(opcode as u8),
    };
    (op, vop3_size(extra))
}

fn decode_ds(word: u32) -> Result<Opcode, InstructionError> {
    use Opcode::*;
    let opcode = (word >> 18) & 0xFF;
    Ok(match opcode {
        0x00 => DsAddU32,
        0x01 => DsSubU32,
        0x02 => DsRsubU32,
        0x03 => DsIncU32,
        0x04 => DsDecU32,
        0x05 => DsMinI32,
        0x06 => DsMaxI32,
        0x07 => DsMinU32,
        0x08 => DsMaxU32,
        0x09 => DsAndB32,
        0x0A => DsOrB32,
        0x0B => DsXorB32,
        0x0C => DsMskorB32,
        0x0D => DsWriteB32,
        0x0E => DsWrite2B32,
        0x0F => DsWrite2St64B32,
        0x10 => DsCmpstB32,
        0x11 => DsCmpstF32,
        0x14 => DsNop,
        0x12 => DsMinF32,
        0x13 => DsMaxF32,
        0x15 => DsAddF32,
        0x18 => DsGwsSemaReleaseAll,
        0x19 => DsGwsInit,
        0x1A => DsGwsSemaV,
        0x1B => DsGwsSemaBr,
        0x1C => DsGwsSemaP,
        0x1D => DsGwsBarrier,
        0x7E => DsCondxchg32RtnB64,
        // gfx10 sub-dword LDS stores/loads.
        0x1E => DsWriteB8,
        0x1F => DsWriteB16,
        0x3A => DsReadU8,
        0x3B => DsReadI16,
        0x3C => DsReadU16,
        0x20 => DsAddRtnU32,
        0x21 => DsSubRtnU32,
        0x22 => DsRsubRtnU32,
        0x23 => DsIncRtnU32,
        0x24 => DsDecRtnU32,
        0x25 => DsMinRtnI32,
        0x26 => DsMaxRtnI32,
        0x27 => DsMinRtnU32,
        0x28 => DsMaxRtnU32,
        0x29 => DsAndRtnB32,
        0x2A => DsOrRtnB32,
        0x2B => DsXorRtnB32,
        0x2C => DsMskorRtnB32,
        0x2D => DsWrxchgRtnB32,
        0x2E => DsWrxchg2RtnB32,
        0x2F => DsWrxchg2St64RtnB32,
        0x30 => DsCmpstRtnB32,
        0x31 => DsCmpstRtnF32,
        0x32 => DsMinRtnF32,
        0x33 => DsMaxRtnF32,
        0x34 => DsWrapRtnB32,
        0x55 => DsAddRtnF32,
        0x35 => DsSwizzleB32,
        0x36 => DsReadB32,
        0x37 => DsRead2B32,
        0x38 => DsRead2St64B32,
        0x39 => DsReadI8,
        0x3D => DsConsume,
        0x3E => DsAppend,
        0x3F => DsOrderedCount,
        // 64-bit LDS atomics: only the no-return forms (the returning forms
        // cannot be split into two 32-bit atomics).
        0x40 => DsAddU64,
        0x41 => DsSubU64,
        0x42 => DsRsubU64,
        0x43 => DsIncU64,
        0x44 => DsDecU64,
        0x45 => DsMinI64,
        0x46 => DsMaxI64,
        0x47 => DsMinU64,
        0x48 => DsMaxU64,
        0x49 => DsAndB64,
        0x4A => DsOrB64,
        0x4B => DsXorB64,
        0x4C => DsMskorB64,
        0x50 => DsCmpstB64,
        0x51 => DsCmpstF64,
        0x52 => DsMinF64,
        0x53 => DsMaxF64,
        0x60 => DsAddRtnU64,
        0x61 => DsSubRtnU64,
        0x62 => DsRsubRtnU64,
        0x63 => DsIncRtnU64,
        0x64 => DsDecRtnU64,
        0x65 => DsMinRtnI64,
        0x66 => DsMaxRtnI64,
        0x67 => DsMinRtnU64,
        0x68 => DsMaxRtnU64,
        0x69 => DsAndRtnB64,
        0x6A => DsOrRtnB64,
        0x6B => DsXorRtnB64,
        0x6C => DsMskorRtnB64,
        0x6D => DsWrxchgRtnB64,
        0x6E => DsWrxchg2RtnB64,
        0x6F => DsWrxchg2St64RtnB64,
        0x70 => DsCmpstRtnB64,
        0x71 => DsCmpstRtnF64,
        0x72 => DsMinRtnF64,
        0x73 => DsMaxRtnF64,
        0xA0 => DsWriteB8D16Hi,
        0xA1 => DsWriteB16D16Hi,
        0xA2 => DsReadU8D16,
        0xA3 => DsReadU8D16Hi,
        0xA4 => DsReadI8D16,
        0xA5 => DsReadI8D16Hi,
        0xA6 => DsReadU16D16,
        0xA7 => DsReadU16D16Hi,
        0xB2 => DsPermuteB32,
        0x80 => DsAddSrc2U32,
        0x81 => DsSubSrc2U32,
        0x82 => DsRsubSrc2U32,
        0x83 => DsIncSrc2U32,
        0x84 => DsDecSrc2U32,
        0x85 => DsMinSrc2I32,
        0x86 => DsMaxSrc2I32,
        0x87 => DsMinSrc2U32,
        0x88 => DsMaxSrc2U32,
        0x89 => DsAndSrc2B32,
        0x8A => DsOrSrc2B32,
        0x8B => DsXorSrc2B32,
        0x8D => DsWriteSrc2B32,
        0x92 => DsMinSrc2F32,
        0x93 => DsMaxSrc2F32,
        0x95 => DsAddSrc2F32,
        0x4D => DsWriteB64,
        0x4E => DsWrite2B64,
        0x4F => DsWrite2St64B64,
        0x76 => DsReadB64,
        0x77 => DsRead2B64,
        0x78 => DsRead2St64B64,
        0xB0 => DsWriteAddtidB32,
        0xB1 => DsReadAddtidB32,
        0xB3 => DsBpermuteB32,
        0xDE => DsWriteB96,
        0xDF => DsWriteB128,
        0xFE => DsReadB96,
        0xFF => DsReadB128,
        _ => return Err(unknown(Encoding::Ds, opcode, word)),
    })
}

fn decode_mtbuf(word: u32, extra: u32) -> Opcode {
    use Opcode::*;
    // Opcode bit 3 lives in the second dword and selects the D16 forms.
    match ((word >> 16) & 7) | (((extra >> 21) & 1) << 3) {
        0x00 => TBufferLoadFormatX,
        0x01 => TBufferLoadFormatXy,
        0x02 => TBufferLoadFormatXyz,
        0x03 => TBufferLoadFormatXyzw,
        0x04 => TBufferStoreFormatX,
        0x05 => TBufferStoreFormatXy,
        0x06 => TBufferStoreFormatXyz,
        0x07 => TBufferStoreFormatXyzw,
        0x08 => TBufferLoadFormatD16X,
        0x09 => TBufferLoadFormatD16Xy,
        0x0A => TBufferLoadFormatD16Xyz,
        0x0B => TBufferLoadFormatD16Xyzw,
        0x0C => TBufferStoreFormatD16X,
        0x0D => TBufferStoreFormatD16Xy,
        0x0E => TBufferStoreFormatD16Xyz,
        _ => TBufferStoreFormatD16Xyzw,
    }
}

fn decode_mubuf(word: u32) -> Opcode {
    use Opcode::*;
    // OP[7] is word bit 25; bit 0 belongs to OFFSET.
    let opcode = ((word >> 18) & 0x7F) | (((word >> 25) & 1) << 7);
    match opcode {
        0x00 => BufferLoadFormatX,
        0x01 => BufferLoadFormatXy,
        0x02 => BufferLoadFormatXyz,
        0x03 => BufferLoadFormatXyzw,
        0x04 => BufferStoreFormatX,
        0x05 => BufferStoreFormatXy,
        0x06 => BufferStoreFormatXyz,
        0x07 => BufferStoreFormatXyzw,
        0x08 => BufferLoadUbyte,
        0x09 => BufferLoadSbyte,
        0x0A => BufferLoadUshort,
        0x0B => BufferLoadSshort,
        0x0C => BufferLoadDword,
        0x0D => BufferLoadDwordx2,
        0x0E => BufferLoadDwordx4,
        0x0F => BufferLoadDwordx3,
        0x18 => BufferStoreByte,
        0x19 => BufferStoreByteD16Hi,
        0x1A => BufferStoreShort,
        0x1B => BufferStoreShortD16Hi,
        0x1C => BufferStoreDword,
        0x1D => BufferStoreDwordx2,
        0x1E => BufferStoreDwordx4,
        0x1F => BufferStoreDwordx3,
        0x20 => BufferLoadUbyteD16,
        0x21 => BufferLoadUbyteD16Hi,
        0x22 => BufferLoadSbyteD16,
        0x23 => BufferLoadSbyteD16Hi,
        0x24 => BufferLoadShortD16,
        0x25 => BufferLoadShortD16Hi,
        0x26 => BufferLoadFormatD16HiX,
        0x27 => BufferStoreFormatD16HiX,
        0x30 => BufferAtomicSwap,
        0x31 => BufferAtomicCmpswap,
        0x32 => BufferAtomicAdd,
        0x33 => BufferAtomicSub,
        0x34 => BufferAtomicCsub,
        0x35 => BufferAtomicSmin,
        0x36 => BufferAtomicUmin,
        0x37 => BufferAtomicSmax,
        0x38 => BufferAtomicUmax,
        0x39 => BufferAtomicAnd,
        0x3A => BufferAtomicOr,
        0x3B => BufferAtomicXor,
        0x3C => BufferAtomicInc,
        0x3D => BufferAtomicDec,
        0x3E => BufferAtomicFcmpswap,
        0x3F => BufferAtomicFmin,
        0x40 => BufferAtomicFmax,
        // gfx9 buffer atomic encodings seen in guest shaders.
        0x41 => BufferAtomicCmpswap,
        0x42 => BufferAtomicAdd,
        0x43 => BufferAtomicSub,
        0x45 => BufferAtomicSmin,
        0x46 => BufferAtomicUmin,
        0x47 => BufferAtomicSmax,
        0x48 => BufferAtomicUmax,
        0x49 => BufferAtomicAnd,
        0x4A => BufferAtomicOr,
        0x4B => BufferAtomicXor,
        0x4C => BufferAtomicInc,
        0x50 => BufferAtomicSwapX2,
        0x51 => BufferAtomicCmpswapX2,
        0x52 => BufferAtomicAddX2,
        0x53 => BufferAtomicSubX2,
        0x55 => BufferAtomicSminX2,
        0x56 => BufferAtomicUminX2,
        0x57 => BufferAtomicSmaxX2,
        0x58 => BufferAtomicUmaxX2,
        0x59 => BufferAtomicAndX2,
        0x5A => BufferAtomicOrX2,
        0x5B => BufferAtomicXorX2,
        0x5C => BufferAtomicIncX2,
        0x5D => BufferAtomicDecX2,
        0x5E => BufferAtomicFcmpswapX2,
        0x5F => BufferAtomicFminX2,
        0x60 => BufferAtomicFmaxX2,
        0x71 => BufferGl0Inv,
        0x72 => BufferGl1Inv,
        0x80 => BufferLoadFormatD16X,
        0x81 => BufferLoadFormatD16Xy,
        0x82 => BufferLoadFormatD16Xyz,
        0x83 => BufferLoadFormatD16Xyzw,
        0x84 => BufferStoreFormatD16X,
        0x85 => BufferStoreFormatD16Xy,
        0x86 => BufferStoreFormatD16Xyz,
        0x87 => BufferStoreFormatD16Xyzw,
        _ => MubufRaw(opcode as u8),
    }
}

/// DS_WRXCHG2*: the returned register range must fit in the VGPR file and
/// the two offsets must name different elements. DS_CONDXCHG32_RTN_B64:
/// its data and returned register pairs must fit in the VGPR file.
fn validate_ds(op: Opcode, word: u32, extra: u32) -> Result<(), InstructionError> {
    use Opcode::*;
    if op == DsCondxchg32RtnB64 {
        if ((extra >> 8) & 0xFF) + 2 > 256 {
            return Err(InstructionError::InvalidDs { reason: "conditional exchange source range overflows the VGPR file", word });
        }
        if (extra >> 24) + 2 > 256 {
            return Err(InstructionError::InvalidDs { reason: "conditional exchange destination range overflows the VGPR file", word });
        }
        return Ok(());
    }
    let dwords = match op {
        DsWrxchg2RtnB32 | DsWrxchg2St64RtnB32 => 2,
        DsWrxchg2RtnB64 | DsWrxchg2St64RtnB64 => 4,
        _ => return Ok(()),
    };
    if (extra >> 24) + dwords > 256 {
        return Err(InstructionError::InvalidDs { reason: "destination range overflows the VGPR file", word });
    }
    if word & 0xFF == (word >> 8) & 0xFF {
        return Err(InstructionError::InvalidDs { reason: "both exchanges address the same element", word });
    }
    Ok(())
}

/// Scalar base of a global/scratch access: an SGPR pair up to s[104:105]
/// or vcc.
fn valid_flat_scalar_base(saddr: u32) -> bool {
    saddr == 106 || (saddr % 2 == 0 && saddr + 1 <= 105)
}

fn decode_flat(word: u32, extra: u32) -> Result<Opcode, InstructionError> {
    let segment = (word >> 14) & 3;
    // OP is bits 24:18; word bit 0 belongs to OFFSET.
    let opcode = (word >> 18) & 0x7F;
    let saddr = (extra >> 16) & 0x7F;
    let seg = match segment {
        0 => Some(FlatSegment::Flat),
        1 => Some(FlatSegment::Scratch),
        2 => Some(FlatSegment::Global),
        _ => None,
    };
    let op = match opcode {
        0x08 => Some(FlatOp::LoadUbyte),
        0x09 => Some(FlatOp::LoadSbyte),
        0x0A => Some(FlatOp::LoadUshort),
        0x0B => Some(FlatOp::LoadSshort),
        0x0C => Some(FlatOp::LoadDword),
        0x0D => Some(FlatOp::LoadDwordx2),
        0x0E => Some(FlatOp::LoadDwordx4),
        0x0F => Some(FlatOp::LoadDwordx3),
        0x18 => Some(FlatOp::StoreByte),
        0x19 => Some(FlatOp::StoreByteD16Hi),
        0x1A => Some(FlatOp::StoreShort),
        0x1B => Some(FlatOp::StoreShortD16Hi),
        0x1C => Some(FlatOp::StoreDword),
        0x1D => Some(FlatOp::StoreDwordx2),
        0x1E => Some(FlatOp::StoreDwordx4),
        0x1F => Some(FlatOp::StoreDwordx3),
        0x20 => Some(FlatOp::LoadUbyteD16),
        0x21 => Some(FlatOp::LoadUbyteD16Hi),
        0x22 => Some(FlatOp::LoadSbyteD16),
        0x23 => Some(FlatOp::LoadSbyteD16Hi),
        0x24 => Some(FlatOp::LoadShortD16),
        0x25 => Some(FlatOp::LoadShortD16Hi),
        0x32 => Some(FlatOp::AtomicAdd),
        0x38 => Some(FlatOp::AtomicUMax),
        0x30 => Some(FlatOp::AtomicSwap),
        0x31 => Some(FlatOp::AtomicCmpswap),
        0x33 => Some(FlatOp::AtomicSub),
        0x34 => Some(FlatOp::AtomicCsub),
        0x35 => Some(FlatOp::AtomicSMin),
        0x36 => Some(FlatOp::AtomicUMin),
        0x37 => Some(FlatOp::AtomicSMax),
        0x39 => Some(FlatOp::AtomicAnd),
        0x3A => Some(FlatOp::AtomicOr),
        0x3B => Some(FlatOp::AtomicXor),
        0x3C => Some(FlatOp::AtomicInc),
        0x3D => Some(FlatOp::AtomicDec),
        0x3E => Some(FlatOp::AtomicFcmpswap),
        0x3F => Some(FlatOp::AtomicFMin),
        0x40 => Some(FlatOp::AtomicFMax),
        0x50 => Some(FlatOp::AtomicSwapX2),
        0x51 => Some(FlatOp::AtomicCmpswapX2),
        0x52 => Some(FlatOp::AtomicAddX2),
        0x53 => Some(FlatOp::AtomicSubX2),
        0x55 => Some(FlatOp::AtomicSMinX2),
        0x56 => Some(FlatOp::AtomicUMinX2),
        0x57 => Some(FlatOp::AtomicSMaxX2),
        0x58 => Some(FlatOp::AtomicUMaxX2),
        0x59 => Some(FlatOp::AtomicAndX2),
        0x5A => Some(FlatOp::AtomicOrX2),
        0x5B => Some(FlatOp::AtomicXorX2),
        0x5C => Some(FlatOp::AtomicIncX2),
        0x5D => Some(FlatOp::AtomicDecX2),
        0x5E => Some(FlatOp::AtomicFcmpswapX2),
        0x5F => Some(FlatOp::AtomicFMinX2),
        0x60 => Some(FlatOp::AtomicFMaxX2),
        0x16 => Some(FlatOp::LoadDwordAddtid),
        0x17 => Some(FlatOp::StoreDwordAddtid),

        _ => None,
    };
    let (Some(seg), Some(op)) = (seg, op) else {
        return Err(InstructionError::UnknownFlat { segment, opcode, word });
    };
    let invalid = |reason| Err(InstructionError::InvalidFlat { reason, word });
    let lds = (word >> 13) & 1 != 0;
    if lds || (op.is_atomic() && seg == FlatSegment::Scratch) {
        return invalid("unsupported modifiers or segment");
    }
    if op == FlatOp::AtomicCsub && seg != FlatSegment::Global {
        return invalid("atomic_csub outside the global segment");
    }
    if matches!(op, FlatOp::LoadDwordAddtid | FlatOp::StoreDwordAddtid) {
        if seg != FlatSegment::Global {
            return invalid("addtid outside the global segment");
        }
        if !valid_flat_scalar_base(saddr) {
            return invalid("addtid needs an SGPR pair base");
        }
    } else if seg != FlatSegment::Flat && saddr != 0x7D && saddr != 0x7F && !valid_flat_scalar_base(saddr) {
        return invalid("scalar address register range");
    }
    Ok(Opcode::Flat(seg, op))
}

fn decode_smrd(word: u32) -> Result<(Opcode, usize), InstructionError> {
    use Opcode::*;
    let opcode = (word >> 22) & 0x1F;
    let immediate = (word >> 8) & 1 != 0;
    let size = if !immediate && word & 0xFF == 0xFF { 2 } else { 1 };
    let op = match opcode {
        0x00 => SLoadDword,
        0x01 => SLoadDwordx2,
        0x02 => SLoadDwordx4,
        0x03 => SLoadDwordx8,
        0x04 => SLoadDwordx16,
        0x08 => SBufferLoadDword,
        0x09 => SBufferLoadDwordx2,
        0x0A => SBufferLoadDwordx4,
        0x0B => SBufferLoadDwordx8,
        0x0C => SBufferLoadDwordx16,
        0x15 => ScratchStoreDword,
        0x16 => ScratchStoreDwordx2,
        0x17 => ScratchStoreDwordx4,
        _ => return Err(unknown(Encoding::Smrd, opcode, word)),
    };
    Ok((op, size))
}

fn decode_smem(word: u32) -> Result<Opcode, InstructionError> {
    use Opcode::*;
    let opcode = (word >> 18) & 0xFF;
    Ok(match opcode {
        0x00 => SLoadDword,
        0x01 => SLoadDwordx2,
        0x02 => SLoadDwordx4,
        0x03 => SLoadDwordx8,
        0x04 => SLoadDwordx16,
        0x05 => SScratchLoadDword,
        0x06 => SScratchLoadDwordx2,
        0x07 => SScratchLoadDwordx4,
        0x08 => SBufferLoadDword,
        0x09 => SBufferLoadDwordx2,
        0x0A => SBufferLoadDwordx4,
        0x0B => SBufferLoadDwordx8,
        0x0C => SBufferLoadDwordx16,
        0x1F => SGl1Inv,
        0x20 => SDcacheInv,
        0x21 => SDcacheWb,
        0x24 => SMemtime,
        0x25 => SMemrealtime,
        0x26 => SAtcProbe,
        0x27 => SAtcProbeBuffer,
        0x28 => SDcacheDiscard,
        0x29 => SDcacheDiscardX2,
        0x2A => SGetWaveidInWorkgroup,
        _ => return Err(unknown(Encoding::Smem, opcode, word)),
    })
}

fn decode_mimg(word: u32) -> Result<Opcode, InstructionError> {
    use Opcode::*;
    // MIMG OP[7] is word bit 0; OP[6:0] is bits 24:18.
    let opcode = ((word >> 18) & 0x7F) | ((word & 1) << 7);
    Ok(match opcode {
        0x00 => ImageLoad,
        0x01 => ImageLoadMip,
        0x08 => ImageStore,
        0x09 => ImageStoreMip,
        0x0E => ImageGetResinfo,
        0x0F => ImageAtomicSwap,
        0x10 => ImageAtomicCmpswap,
        0x11 => ImageAtomicAdd,
        0x12 => ImageAtomicSub,
        0x14 => ImageAtomicSmin,
        0x15 => ImageAtomicUmin,
        0x16 => ImageAtomicSmax,
        0x17 => ImageAtomicUmax,
        0x18 => ImageAtomicAnd,
        0x19 => ImageAtomicOr,
        0x1A => ImageAtomicXor,
        0x1B => ImageAtomicInc,
        0x1C => ImageAtomicDec,
        0x1D => ImageAtomicFcmpswap,
        0x1E => ImageAtomicFmin,
        0x1F => ImageAtomicFmax,
        0x20 => ImageSample,
        0x21 => ImageSampleCl,
        0x22 => ImageSampleD,
        0x23 => ImageSampleDCl,
        0x24 => ImageSampleL,
        0x25 => ImageSampleB,
        0x26 => ImageSampleBCl,
        0x27 => ImageSampleLz,
        0x28 => ImageSampleC,
        0x29 => ImageSampleCCl,
        0x2A => ImageSampleCD,
        0x2B => ImageSampleCDCl,
        0x2C => ImageSampleCL,
        0x2D => ImageSampleCB,
        0x2E => ImageSampleCBCl,
        0x2F => ImageSampleCLz,
        0x30 => ImageSampleO,
        0x31 => ImageSampleClO,
        0x32 => ImageSampleDO,
        0x33 => ImageSampleDClO,
        0x34 => ImageSampleLO,
        0x35 => ImageSampleBO,
        0x36 => ImageSampleBClO,
        0x37 => ImageSampleLzO,
        0x38 => ImageSampleCO,
        0x39 => ImageSampleCClO,
        0x3A => ImageSampleCDO,
        0x3B => ImageSampleCDClO,
        0x3C => ImageSampleCLO,
        0x3D => ImageSampleCBO,
        0x3E => ImageSampleCBClO,
        0x3F => ImageSampleCLzO,
        0x40 => ImageGather4,
        0x41 => ImageGather4Cl,
        0x44 => ImageGather4L,
        0x45 => ImageGather4B,
        0x46 => ImageGather4BCl,
        0x47 => ImageGather4Lz,
        0x48 => ImageGather4C,
        0x49 => ImageGather4CCl,
        0x4C => ImageGather4CL,
        0x4D => ImageGather4CB,
        0x4E => ImageGather4CBCl,
        0x4F => ImageGather4CLz,
        0x50 => ImageGather4O,
        0x51 => ImageGather4ClO,
        0x54 => ImageGather4LO,
        0x55 => ImageGather4BO,
        0x56 => ImageGather4BClO,
        0x57 => ImageGather4LzO,
        0x58 => ImageGather4CO,
        0x59 => ImageGather4CClO,
        0x5C => ImageGather4CLO,
        0x5D => ImageGather4CBO,
        0x5E => ImageGather4CBClO,
        0x5F => ImageGather4CLzO,
        0x60 => ImageGetLod,
        0x61 => ImageGather4H,
        0x68 => ImageSampleCd,
        0x69 => ImageSampleCdCl,
        0x6A => ImageSampleCCd,
        0x6B => ImageSampleCCdCl,
        0x6C => ImageSampleCdO,
        0x6D => ImageSampleCdClO,
        0x6E => ImageSampleCCdO,
        0x6F => ImageSampleCCdClO,
        0xA0 => ImageSampleA,
        0xA1 => ImageSampleClA,
        0xA5 => ImageSampleBA,
        0xA6 => ImageSampleBClA,
        0xA8 => ImageSampleCA,
        0xA9 => ImageSampleCClA,
        0xAD => ImageSampleCBA,
        0xAE => ImageSampleCBClA,
        0xB0 => ImageSampleAO,
        0xB1 => ImageSampleClAO,
        0xB5 => ImageSampleBAO,
        0xB6 => ImageSampleBClAO,
        0xB8 => ImageSampleCAO,
        0xB9 => ImageSampleCClAO,
        0xBD => ImageSampleCBAO,
        0xBE => ImageSampleCBClAO,
        0xC8 => ImageGather4CA,
        0xD8 => ImageGather4CAO,
        0xA2 => ImageSampleDG16,
        0xA3 => ImageSampleDClG16,
        0xAA => ImageSampleCDG16,
        0xAB => ImageSampleCDClG16,
        0xB2 => ImageSampleDOG16,
        0xB3 => ImageSampleDClOG16,
        0xBA => ImageSampleCDOG16,
        0xBB => ImageSampleCDClOG16,
        0xE8 => ImageSampleCdG16,
        0xE9 => ImageSampleCdClG16,
        0xEA => ImageSampleCCdG16,
        0xEB => ImageSampleCCdClG16,
        0xEC => ImageSampleCdOG16,
        0xED => ImageSampleCdClOG16,
        0xEE => ImageSampleCCdOG16,
        0xEF => ImageSampleCCdClOG16,
        0xE6 => ImageBvhIntersectRay,
        0xE7 => ImageBvh64IntersectRay,
        0x80 => ImageMsaaLoad,
        0x02 => ImageLoadPck,
        0x03 => ImageLoadPckSgn,
        0x04 => ImageLoadMipPck,
        0x05 => ImageLoadMipPckSgn,
        0x0A => ImageStorePck,
        0x0B => ImageStoreMipPck,
        0x42 => ImageLoadBy2,
        0x43 => ImageLoadBy4,
        0x4A => ImageLoadMipBy2,
        0x4B => ImageLoadMipBy4,
        0x52 => ImageStoreBy2,
        0x53 => ImageStoreBy4,
        0x5A => ImageStoreMipBy2,
        0x5B => ImageStoreMipBy4,
        0x62 => ImageGather4hPck,
        0x63 => ImageGather8hPck,
        0x70 => ImageLoadPck2,
        0x71 => ImageLoadPck4,
        0x73 => ImageLoadMipPck2,
        0x74 => ImageLoadMipPck4,
        0x76 => ImageStorePck2,
        0x77 => ImageStorePck4,
        0x79 => ImageStoreMipPck2,
        0x7A => ImageStoreMipPck4,
        _ => return Err(unknown(Encoding::Mimg, opcode, word)),
    })
}

fn decode_vintrp(word: u32) -> Result<Opcode, InstructionError> {
    match (word >> 16) & 3 {
        0 => Ok(Opcode::VInterpP1F32),
        1 => Ok(Opcode::VInterpP2F32),
        2 => Ok(Opcode::VInterpMovF32),
        op => Err(unknown(Encoding::Vintrp, op, word)),
    }
}

// ---------------------------------------------------------------------------
// Operand / control extraction
// ---------------------------------------------------------------------------

fn sign_extend(value: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((value << shift) as i32) >> shift
}

fn vectors(start: u32, count: u32) -> impl Iterator<Item = Operand> {
    (start..start + count).map(Operand::vector)
}

fn dpp_control(w: u32) -> Control {
    Control::Dpp(DppControl {
        control: (w >> 8) & 0x1FF,
        fetch_inactive: (w >> 18) & 1 != 0,
        bound_control: (w >> 19) & 1 != 0,
        absolute_mask: ((w >> 21) & 1) | (((w >> 23) & 1) << 1),
        negate_mask: ((w >> 20) & 1) | (((w >> 22) & 1) << 1),
        bank_mask: (w >> 24) & 0xF,
        row_mask: (w >> 28) & 0xF,
    })
}

fn sdwa_control(w: u32, is_compare: bool, has_source1: bool) -> SdwaControl {
    let scalar_destination = if is_compare { Some(if (w >> 15) & 1 != 0 { (w >> 8) & 0x7F } else { VCC_LO }) } else { None };
    SdwaControl {
        destination_select: if is_compare { 6 } else { (w >> 8) & 7 },
        destination_unused: if is_compare { 0 } else { (w >> 11) & 3 },
        source0_select: (w >> 16) & 7,
        source1_select: if has_source1 { (w >> 24) & 7 } else { 6 },
        source0_sign_extend: (w >> 19) & 1 != 0,
        source1_sign_extend: has_source1 && (w >> 27) & 1 != 0,
        absolute_mask: ((w >> 21) & 1) | if has_source1 { ((w >> 29) & 1) << 1 } else { 0 },
        negate_mask: ((w >> 20) & 1) | if has_source1 { ((w >> 28) & 1) << 1 } else { 0 },
        output_modifier: if is_compare { 0 } else { (w >> 14) & 3 },
        clamp: !is_compare && (w >> 13) & 1 != 0,
        scalar_destination,
    }
}

/// SDWA source: 8-bit register number plus the S0/S1 "is SGPR" bit.
fn sdwa_source(reg: u32, sgpr_bit: u32) -> Operand {
    Operand::source(reg + if sgpr_bit == 0 { 256 } else { 0 }, None)
}

/// Build the instruction model for already-sized `words`.
pub fn create_instruction(pc: u32, mut encoding: Encoding, mut opcode: Opcode, words: Vec<u32>) -> Instruction {
    use Opcode::*;
    let word = words[0];
    let vop_form = matches!(encoding, Encoding::Vop1 | Encoding::Vop2 | Encoding::Vopc);
    let src0_field = word & 0x1FF;
    let is_sdwa = vop_form && src0_field == 0xF9;
    let is_dpp = vop_form && src0_field == 0xFA;
    let is_dpp8 = vop_form && matches!(src0_field, 0xE9 | 0xEA);
    let literal = if !is_sdwa && !is_dpp && !is_dpp8 && words.len() > encoding.minimum_dwords() { words.last().copied() } else { None };
    let mut sources: Vec<Operand> = Vec::new();
    let mut destinations: Vec<Operand> = Vec::new();
    let mut control: Option<Control> = None;

    match encoding {
        Encoding::Sopp if opcode == STrap => {
            sources = vec![Operand::literal(word & 0xFF)];
        }
        Encoding::Sop1 => {
            sources = vec![Operand::source(word & 0xFF, literal)];
            destinations = vec![Operand::scalar((word >> 16) & 0x7F)];
            if opcode == SCmovB64 {
                // CMOV is CSELECT with the old destination as the false
                // source; reuse its SCC, register-pair and EXEC handling.
                opcode = SCselectB64;
                encoding = Encoding::Sop2;
                sources.push(destinations[0]);
            }
        }
        Encoding::Sop2 => {
            sources = vec![Operand::source(word & 0xFF, literal), Operand::source((word >> 8) & 0xFF, literal)];
            destinations = vec![Operand::scalar((word >> 16) & 0x7F)];
        }
        Encoding::Sopc => {
            sources = vec![Operand::source(word & 0xFF, literal), Operand::source((word >> 8) & 0xFF, literal)];
        }
        Encoding::Sopk => {
            let simm = Operand::encoded(word & 0xFFFF);
            let sdst = (word >> 16) & 0x7F;
            if opcode == SWaitcnt {
                sources = vec![simm];
                if sdst != NULL_SOURCE {
                    sources.push(Operand::scalar(sdst));
                }
            } else if opcode == SSetregB32 {
                sources = vec![Operand::scalar(sdst), simm];
            } else if opcode == SSetregImm32B32 {
                sources = vec![Operand::literal(words.get(1).copied().unwrap_or(0)), simm];
            } else if opcode == SVersion {
                sources = vec![simm];
            } else {
                sources = vec![simm];
                destinations = vec![Operand::scalar(sdst)];
            }
        }
        Encoding::Smrd => {
            let scalar_base = ((word >> 9) & 0x3F) * 2;
            let sdst = (word >> 15) & 0x7F;
            let immediate = (word >> 8) & 1 != 0;
            let offset = word & 0xFF;
            let mut dynamic = None;
            let mut imm_bytes = 0i32;
            if immediate {
                imm_bytes = (offset * 4) as i32;
            } else if offset == 0xFF && literal.is_some() {
                imm_bytes = literal.unwrap() as i32;
            } else {
                dynamic = Some(offset);
            }
            let store = opcode.scalar_store_dwords();
            if store != 0 {
                sources = std::iter::once(Operand::scalar(scalar_base)).chain((sdst..sdst + store).map(Operand::scalar)).collect();
                if let Some(d) = dynamic {
                    sources.push(Operand::scalar(d));
                }
                control = Some(Control::GlobalMemory(GlobalMemoryControl {
                    dword_count: store,
                    vector_address: 0,
                    source_vector_register: sdst,
                    destination_vector_register: 0,
                    scalar_address: scalar_base,
                    offset_bytes: imm_bytes,
                    glc: false,
                    slc: false,
                    uses_flat_address: false,
                    dynamic_offset_register: dynamic,
                    source_is_scalar: true,
                }));
            } else {
                let count = opcode.scalar_load_dwords();
                sources = vec![Operand::scalar(scalar_base)];
                if let Some(d) = dynamic {
                    sources.push(Operand::scalar(d));
                }
                destinations = (sdst..sdst + count).map(Operand::scalar).collect();
                control = Some(Control::ScalarMemory(ScalarMemoryControl {
                    destination_count: count,
                    immediate_offset_bytes: imm_bytes,
                    dynamic_offset_register: dynamic,
                }));
            }
        }
        Encoding::Smem => {
            let extra = words[1];
            let scalar_base = (word & 0x3F) * 2;
            let sdst = (word >> 6) & 0x7F;
            let soffset = Operand::source((extra >> 25) & 0x7F, None);
            let count = opcode.scalar_load_dwords();
            let dynamic = (soffset.kind == OperandKind::ScalarRegister).then_some(soffset.value);
            if count > 0 {
                sources = vec![Operand::scalar(scalar_base), soffset];
                destinations = (sdst..sdst + count).map(Operand::scalar).collect();
                control = Some(Control::ScalarMemory(ScalarMemoryControl {
                    destination_count: count,
                    immediate_offset_bytes: sign_extend(extra & 0x1F_FFFF, 21),
                    dynamic_offset_register: dynamic,
                }));
            } else {
                // Clocks, wave id and cache control access no memory.
                let written = match opcode {
                    SMemtime | SMemrealtime => 2,
                    SGetWaveidInWorkgroup => 1,
                    _ => 0,
                };
                destinations = (sdst..sdst + written).map(Operand::scalar).collect();
            }
        }
        Encoding::Vop1 => {
            if is_dpp8 {
                let extra = words[1];
                sources = vec![Operand::vector(extra & 0xFF)];
                control = Some(Control::Dpp8(Dpp8Control { lane_selectors: extra >> 8, fetch_inactive: src0_field == 0xEA }));
            } else if is_dpp {
                let extra = words[1];
                sources = vec![Operand::vector(extra & 0xFF)];
                control = Some(dpp_control(extra));
            } else if is_sdwa {
                let extra = words[1];
                sources = vec![sdwa_source(extra & 0xFF, (extra >> 23) & 1)];
                control = Some(Control::Sdwa(sdwa_control(extra, false, false)));
            } else {
                sources = vec![Operand::source(src0_field, literal)];
            }
            if opcode == VMovrelsB32 {
                sources = vec![sources[0], Operand::scalar(M0)];
            }
            // V_READFIRSTLANE_B32 is VOP1 but its destination names an SGPR.
            destinations = if opcode == VReadfirstlaneB32 {
                vec![Operand::scalar((word >> 17) & 0x7F)]
            } else {
                vec![Operand::vector((word >> 17) & 0xFF)]
            };
        }
        Encoding::Vop2 | Encoding::Vopc => {
            let vsrc1 = Operand::vector((word >> 9) & 0xFF);
            if is_dpp8 {
                let extra = words[1];
                sources = vec![Operand::vector(extra & 0xFF), vsrc1];
                control = Some(Control::Dpp8(Dpp8Control { lane_selectors: extra >> 8, fetch_inactive: src0_field == 0xEA }));
            } else if is_dpp {
                let extra = words[1];
                sources = vec![Operand::vector(extra & 0xFF), vsrc1];
                control = Some(dpp_control(extra));
            } else if is_sdwa {
                let extra = words[1];
                sources = vec![sdwa_source(extra & 0xFF, (extra >> 23) & 1), sdwa_source((word >> 9) & 0xFF, (extra >> 31) & 1)];
                let is_compare = encoding == Encoding::Vopc;
                let sdwa = sdwa_control(extra, is_compare, true);
                if is_compare {
                    if let Some(sd) = sdwa.scalar_destination.filter(|&d| d != VCC_LO) {
                        destinations = vec![Operand::scalar(sd)];
                    }
                }
                control = Some(Control::Sdwa(sdwa));
            } else {
                sources = vec![Operand::source(src0_field, literal), vsrc1];
                if let Some(lit) = literal {
                    if matches!(opcode, VMadMkF32 | VFmaMkF32 | VFmaMkF16) {
                        sources = vec![sources[0], Operand::literal(lit), sources[1]];
                    } else if matches!(opcode, VMadAkF32 | VFmaAkF32 | VFmaAkF16) {
                        sources.push(Operand::literal(lit));
                    }
                }
            }
            if encoding == Encoding::Vop2 {
                destinations = vec![Operand::vector((word >> 17) & 0xFF)];
            }
        }
        Encoding::Vop3 => {
            let extra = words[1];
            sources = vec![
                Operand::source(extra & 0x1FF, literal),
                Operand::source((extra >> 9) & 0x1FF, literal),
                Operand::source((extra >> 18) & 0x1FF, literal),
            ];
            destinations = vec![Operand::vector(word & 0xFF)];
            if opcode == VReadlaneB32 {
                // V_READLANE writes the SGPR named by the VOP3A vdst byte.
                destinations = vec![Operand::scalar(word & 0xFF)];
            }
            let vop3b = is_vop3b_opcode((word >> 16) & 0x3FF);
            if opcode.is_vector_compare() {
                // Two sources; the mask goes to the SGPR pair in the vdst
                // byte. gfx10 V_CMPX writes EXEC only.
                let exec_compare = opcode.is_vector_compare_exec();
                sources.truncate(2);
                destinations = if exec_compare { vec![] } else { vec![Operand::scalar(word & 0xFF)] };
                control = Some(Control::Vop3(Vop3Control {
                    absolute_mask: (word >> 8) & 7,
                    negate_mask: (extra >> 29) & 7,
                    output_modifier: 0,
                    clamp: false,
                    operand_select: 0,
                    scalar_destination: if exec_compare { None } else { Some(word & 0xFF) },
                }));
            } else {
                control = Some(Control::Vop3(Vop3Control {
                    absolute_mask: if vop3b { 0 } else { (word >> 8) & 7 },
                    negate_mask: (extra >> 29) & 7,
                    output_modifier: (extra >> 27) & 3,
                    clamp: (word >> 15) & 1 != 0,
                    operand_select: if vop3b { 0 } else { (word >> 11) & 0xF },
                    scalar_destination: vop3b.then_some((word >> 8) & 0x7F),
                }));
            }
        }
        Encoding::Vop3p => {
            let extra = words[1];
            sources = vec![
                Operand::source(extra & 0x1FF, literal),
                Operand::source((extra >> 9) & 0x1FF, literal),
                Operand::source((extra >> 18) & 0x1FF, literal),
            ];
            destinations = vec![Operand::vector(word & 0xFF)];
            // op_sel_hi bits 1:0 are word1 28:27, bit 2 is word0 14.
            control = Some(Control::Vop3p(Vop3pControl {
                op_sel_mask: (word >> 11) & 7,
                op_sel_hi_mask: ((extra >> 27) & 3) | (((word >> 14) & 1) << 2),
                neg_lo_mask: (extra >> 29) & 7,
                neg_hi_mask: (word >> 8) & 7,
                clamp: (word >> 15) & 1 != 0,
            }));
        }
        Encoding::Ds => {
            let extra = words[1];
            let addr = Operand::vector(extra & 0xFF);
            let d0 = (extra >> 8) & 0xFF;
            let d1 = (extra >> 16) & 0xFF;
            let vdst = (extra >> 24) & 0xFF;
            // GFX10 DS: offset0 7:0, offset1 15:8, GDS 17, op 25:18.
            control = Some(Control::DataShare(DataShareControl { offset0: word & 0xFF, offset1: (word >> 8) & 0xFF, gds: (word >> 17) & 1 != 0 }));
            let v = Operand::vector;
            sources = match opcode {
                DsAppend | DsConsume | DsReadAddtidB32 => vec![Operand::scalar(M0)],
                DsOrderedCount => vec![addr, Operand::scalar(M0)],
                DsWriteAddtidB32 => vec![Operand::scalar(M0), v(d0)],
                DsWriteB32 | DsWriteB8 | DsWriteB16 => vec![addr, v(d0)],
                DsWriteB64 => vec![addr, v(d0), v(d0 + 1)],
                DsWrite2B64 | DsWrite2St64B64 => vec![addr, v(d0), v(d0 + 1), v(d1), v(d1 + 1)],
                DsWriteB96 => vec![addr, v(d0), v(d0 + 1), v(d0 + 2)],
                DsWriteB128 => vec![addr, v(d0), v(d0 + 1), v(d0 + 2), v(d0 + 3)],
                DsWrite2B32 | DsWrite2St64B32 => vec![addr, v(d0), v(d1)],
                DsSwizzleB32 => vec![addr],
                DsNop => vec![],
                DsBpermuteB32 => vec![addr, v(d0)],
                DsPermuteB32 => vec![addr, v(d0)],
                DsWriteB8D16Hi | DsWriteB16D16Hi => vec![addr, v(d0)],
                DsCondxchg32RtnB64 => vec![addr, v(d0), v(d0 + 1)],
                // D16 reads keep the other half of the destination.
                DsReadU8D16 | DsReadU8D16Hi | DsReadI8D16 | DsReadI8D16Hi | DsReadU16D16 | DsReadU16D16Hi => vec![addr, v(vdst)],
                // Atomics: DATA0 (and DATA1 for compare / mask forms; for
                // DS_CMPST DATA0 is the comparator, DATA1 the new value),
                // a register pair each for 64-bit forms.
                _ => match opcode.ds_atomic_shape() {
                    Some((wide, n, _)) => {
                        let mut v0 = vec![addr];
                        let w = if wide { 2 } else { 1 };
                        v0.extend(vectors(d0, w));
                        if n == 2 {
                            v0.extend(vectors(d1, w));
                        }
                        v0
                    }
                    None => vec![addr],
                },
            };
            destinations = match opcode {
                DsAppend | DsConsume | DsOrderedCount | DsReadB32 | DsReadI8 | DsReadU8 | DsReadI16 | DsReadU16 | DsReadAddtidB32 | DsSwizzleB32 | DsBpermuteB32 => vec![v(vdst)],
                DsReadB64 | DsRead2B32 | DsRead2St64B32 => vectors(vdst, 2).collect(),
                DsReadB96 => vectors(vdst, 3).collect(),
                DsCondxchg32RtnB64 => vectors(vdst, 2).collect(),
                DsReadB128 | DsRead2B64 | DsRead2St64B64 => vectors(vdst, 4).collect(),
                DsPermuteB32 | DsReadU8D16 | DsReadU8D16Hi | DsReadI8D16 | DsReadI8D16Hi | DsReadU16D16 | DsReadU16D16Hi => vec![v(vdst)],
                _ => match opcode.ds_atomic_shape() {
                    Some((wide, n, true)) => {
                        let count = if wide { 2 } else { 1 } * if matches!(opcode, DsWrxchg2RtnB32 | DsWrxchg2St64RtnB32 | DsWrxchg2RtnB64 | DsWrxchg2St64RtnB64) { n } else { 1 };
                        vectors(vdst, count).collect()
                    }
                    _ => vec![],
                },
            };
        }
        Encoding::Vintrp => {
            sources = vec![Operand::vector(word & 0xFF)];
            destinations = vec![Operand::vector((word >> 18) & 0xFF)];
            control = Some(Control::Interpolation(InterpolationControl { attribute: (word >> 10) & 0x3F, channel: (word >> 8) & 3 }));
        }
        Encoding::Flat => {
            let extra = words[1];
            let vaddr = extra & 0xFF;
            let vdata = (extra >> 8) & 0xFF;
            let vdst = (extra >> 24) & 0xFF;
            let saddr = (extra >> 16) & 0x7F;
            let Opcode::Flat(segment, op) = opcode else { unreachable!() };
            let uses_flat = segment == FlatSegment::Flat;
            let uses_scratch = segment == FlatSegment::Scratch;
            let dword_count = op.dword_count();
            sources = if uses_flat {
                vec![Operand::vector(vaddr), Operand::vector(vaddr + 1)]
            } else if uses_scratch && saddr < NULL_SOURCE {
                vec![Operand::scalar(saddr)]
            } else if uses_scratch {
                vec![Operand::vector(vaddr)]
            } else {
                vec![Operand::vector(vaddr), Operand::scalar(saddr)]
            };
            let glc = (word >> 16) & 1 != 0;
            if op.is_store() || op.is_atomic() {
                sources.extend(vectors(vdata, dword_count));
            } else if op.is_d16_load() {
                // A partial load preserves the other half of the destination.
                sources.push(Operand::vector(vdst));
            }
            if op.is_load() {
                destinations = vectors(vdst, dword_count).collect();
            } else if op.is_atomic() && glc {
                destinations = vectors(vdst, op.return_dwords()).collect();
            }
            control = Some(Control::GlobalMemory(GlobalMemoryControl {
                dword_count,
                vector_address: vaddr,
                source_vector_register: vdata,
                destination_vector_register: vdst,
                scalar_address: if uses_flat { u32::MAX } else { saddr },
                offset_bytes: if uses_flat { (word & 0x7FF) as i32 } else { sign_extend(word & 0xFFF, 12) },
                glc,
                slc: (word >> 17) & 1 != 0,
                uses_flat_address: uses_flat,
                dynamic_offset_register: None,
                source_is_scalar: false,
            }));
        }
        Encoding::Mubuf | Encoding::Mtbuf => {
            let extra = words[1];
            let vaddr = extra & 0xFF;
            let vdata = (extra >> 8) & 0xFF;
            let srsrc = ((extra >> 16) & 0x1F) * 4;
            let soffset = (extra >> 24) & 0xFF;
            let dword_count = opcode.buffer_dwords();
            sources = vec![Operand::vector(vaddr), Operand::scalar(srsrc), Operand::source(soffset, literal)];
            destinations = vectors(vdata, dword_count).collect();
            let typed = encoding == Encoding::Mtbuf;
            control = Some(Control::BufferMemory(BufferMemoryControl {
                dword_count,
                vector_address: vaddr,
                vector_data: vdata,
                scalar_resource: srsrc,
                offset_bytes: (word & 0xFFF) as i32,
                index_enabled: (word >> 13) & 1 != 0,
                offset_enabled: (word >> 12) & 1 != 0,
                glc: (word >> 14) & 1 != 0,
                slc: (extra >> 22) & 1 != 0,
                typed,
                typed_format: if typed { (word >> 19) & 0x7F } else { 0 },
            }));
        }
        Encoding::Mimg => {
            let extra = words[1];
            let vaddr = extra & 0xFF;
            let vdata = (extra >> 8) & 0xFF;
            let srsrc = ((extra >> 16) & 0x1F) * 4;
            let ssamp = ((extra >> 21) & 0x1F) * 4;
            let mut address_registers = vec![vaddr];
            for w in &words[2..] {
                address_registers.extend((0..4).map(|i| (w >> (i * 8)) & 0xFF));
            }
            if opcode.is_ray_intersect() {
                let a16 = (extra >> 30) & 1 != 0;
                let ray = RayIntersectControl { vector_address: vaddr, address_registers, vector_data: vdata, scalar_resource: srsrc, a16 };
                // Node pointer (2 dwords for BVH64), extent, origin, then
                // direction and inverse direction (A16 packs them into 3).
                let count = if opcode == ImageBvh64IntersectRay { 2 } else { 1 } + 1 + 3 + if a16 { 3 } else { 6 };
                sources = (0..count).map(|c| Operand::vector(ray.address_register(c))).collect();
                sources.push(Operand::scalar(srsrc));
                destinations = vectors(vdata, RayIntersectControl::RESULT_DWORDS).collect();
                control = Some(Control::RayIntersect(ray));
            } else {
                sources = address_registers.iter().map(|&r| Operand::vector(r)).collect();
                sources.push(Operand::scalar(srsrc));
                sources.push(Operand::scalar(ssamp));
                destinations = if opcode.is_image_store() { vec![] } else { vec![Operand::vector(vdata)] };
                let dimension = (word >> 3) & 7;
                control = Some(Control::Image(ImageControl {
                    dmask: (word >> 8) & 0xF,
                    vector_address: vaddr,
                    address_registers,
                    vector_data: vdata,
                    scalar_resource: srsrc,
                    scalar_sampler: ssamp,
                    dimension,
                    is_array: matches!(dimension, 4 | 5 | 7),
                    glc: (word >> 13) & 1 != 0,
                    slc: (word >> 25) & 1 != 0,
                    a16: (extra >> 30) & 1 != 0,
                    d16: (extra >> 31) & 1 != 0,
                }));
            }
        }
        Encoding::Exp => {
            let extra = words[1];
            sources = (0..4).map(|i| Operand::vector((extra >> (i * 8)) & 0xFF)).collect();
            control = Some(Control::Export(ExportControl {
                target: (word >> 4) & 0x3F,
                enable_mask: word & 0xF,
                compressed: (word >> 10) & 1 != 0,
                done: (word >> 11) & 1 != 0,
                valid_mask: (word >> 12) & 1 != 0,
            }));
        }
        Encoding::Sopp => {}
    }

    Instruction { pc, encoding, opcode, words, sources, destinations, control, address_offset: None }
}

/// Decode one instruction from a dword slice (`words[0]` is the first word).
/// Returns `None` in the error slot if the slice is too short.
pub fn decode_instruction(pc: u32, words: &[u32]) -> Result<Instruction, InstructionError> {
    let first = *words.first().ok_or(InstructionError::UnknownEncoding { word: 0 })?;
    let (encoding, opcode, size) = decode_opcode(first, words.get(1).copied())?;
    if words.len() < size {
        return Err(InstructionError::MissingExtraWord { encoding });
    }
    Ok(create_instruction(pc, encoding, opcode, words[..size].to_vec()))
}
