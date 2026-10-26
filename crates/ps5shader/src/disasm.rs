// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Disassembly (`Display`) for operands, instructions and programs.
//!
//! The output is AMD-assembler-like but not byte-for-byte LLVM syntax: the
//! operand lists are the decoder's sources/destinations (`dst <- src`), and
//! encoding-specific fields are appended in braces.

use crate::inline_constants;
use crate::ir_types::*;
use std::fmt;

/// Name of a scalar register index.
pub fn scalar_name(index: u32) -> String {
    match index {
        VCC_LO => "vcc_lo".into(),
        VCC_HI => "vcc_hi".into(),
        108..=123 => format!("ttmp{}", index - 108),
        M0 => "m0".into(),
        EXEC_LO => "exec_lo".into(),
        EXEC_HI => "exec_hi".into(),
        _ => format!("s{index}"),
    }
}

fn encoded_name(value: u32) -> String {
    match value {
        125 => "null".into(),
        128..=208 => format!("{}", inline_constants::decode(value).unwrap() as i32),
        240 => "0.5".into(),
        241 => "-0.5".into(),
        242 => "1.0".into(),
        243 => "-1.0".into(),
        244 => "2.0".into(),
        245 => "-2.0".into(),
        246 => "4.0".into(),
        247 => "-4.0".into(),
        248 => "0.15915494".into(),
        235 => "src_shared_base".into(),
        236 => "src_shared_limit".into(),
        237 => "src_private_base".into(),
        238 => "src_private_limit".into(),
        239 => "src_pops_exiting_wave_id".into(),
        VCCZ_SOURCE => "vccz".into(),
        EXECZ_SOURCE => "execz".into(),
        SCC_SOURCE => "scc".into(),
        254 => "lds_direct".into(),
        _ => format!("src[{value}]"),
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            OperandKind::ScalarRegister => f.write_str(&scalar_name(self.value)),
            OperandKind::VectorRegister => write!(f, "v{}", self.value),
            OperandKind::LiteralConstant => write!(f, "0x{:08x}", self.value),
            OperandKind::EncodedConstant => f.write_str(&encoded_name(self.value)),
        }
    }
}

fn flag(f: &mut fmt::Formatter<'_>, on: bool, name: &str) -> fmt::Result {
    if on {
        write!(f, " {name}")
    } else {
        Ok(())
    }
}

impl fmt::Display for Control {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Control::Image(c) => {
                write!(f, "dmask:0x{:x} dim:{}", c.dmask, c.dimension)?;
                flag(f, c.is_array, "da")?;
                flag(f, c.glc, "glc")?;
                flag(f, c.slc, "slc")?;
                flag(f, c.a16, "a16")?;
                flag(f, c.d16, "d16")
            }
            Control::RayIntersect(c) => {
                write!(f, "bvh")?;
                flag(f, c.a16, "a16")
            }
            Control::GlobalMemory(c) => {
                write!(f, "dwords:{} offset:{}", c.dword_count, c.offset_bytes)?;
                if c.scalar_address != u32::MAX {
                    write!(f, " saddr:{}", scalar_name(c.scalar_address))?;
                }
                if let Some(d) = c.dynamic_offset_register {
                    write!(f, " soffset:{}", scalar_name(d))?;
                }
                flag(f, c.glc, "glc")?;
                flag(f, c.slc, "slc")
            }
            Control::BufferMemory(c) => {
                write!(f, "dwords:{} offset:{}", c.dword_count, c.offset_bytes)?;
                flag(f, c.index_enabled, "idxen")?;
                flag(f, c.offset_enabled, "offen")?;
                flag(f, c.glc, "glc")?;
                flag(f, c.slc, "slc")?;
                if c.typed {
                    write!(f, " format:{}", c.typed_format)?;
                }
                Ok(())
            }
            Control::Export(c) => {
                let target = match c.target {
                    0..=7 => format!("mrt{}", c.target),
                    8 => "mrtz".into(),
                    9 => "null".into(),
                    12..=15 => format!("pos{}", c.target - 12),
                    20 => "prim".into(),
                    32..=63 => format!("param{}", c.target - 32),
                    t => format!("target{t}"),
                };
                write!(f, "{target} en:0x{:x}", c.enable_mask)?;
                flag(f, c.compressed, "compr")?;
                flag(f, c.done, "done")?;
                flag(f, c.valid_mask, "vm")
            }
            Control::Interpolation(c) => write!(f, "attr{}.{}", c.attribute, ['x', 'y', 'z', 'w'][c.channel as usize & 3]),
            Control::Vop3(c) => {
                write!(f, "vop3")?;
                for (name, v) in [("abs", c.absolute_mask), ("neg", c.negate_mask), ("omod", c.output_modifier), ("op_sel", c.operand_select)] {
                    if v != 0 {
                        write!(f, " {name}:{v}")?;
                    }
                }
                flag(f, c.clamp, "clamp")?;
                if let Some(d) = c.scalar_destination {
                    write!(f, " sdst:{}", scalar_name(d))?;
                }
                Ok(())
            }
            Control::Sdwa(c) => {
                write!(
                    f,
                    "sdwa dst_sel:{} dst_unused:{} src0_sel:{} src1_sel:{} abs:{} neg:{}",
                    c.destination_select, c.destination_unused, c.source0_select, c.source1_select, c.absolute_mask, c.negate_mask
                )?;
                flag(f, c.clamp, "clamp")?;
                if let Some(d) = c.scalar_destination {
                    write!(f, " sdst:{}", scalar_name(d))?;
                }
                Ok(())
            }
            Control::Vop3p(c) => {
                write!(
                    f,
                    "op_sel:{} op_sel_hi:{} neg_lo:{} neg_hi:{}",
                    c.op_sel_mask, c.op_sel_hi_mask, c.neg_lo_mask, c.neg_hi_mask
                )?;
                flag(f, c.clamp, "clamp")
            }
            Control::Dpp(c) => {
                write!(f, "dpp_ctrl:0x{:x} row_mask:0x{:x} bank_mask:0x{:x}", c.control, c.row_mask, c.bank_mask)?;
                flag(f, c.bound_control, "bound_ctrl")?;
                flag(f, c.fetch_inactive, "fi")
            }
            Control::Dpp8(c) => {
                write!(f, "dpp8:[")?;
                for i in 0..8 {
                    write!(f, "{}{}", if i == 0 { "" } else { "," }, (c.lane_selectors >> (i * 3)) & 7)?;
                }
                write!(f, "]")?;
                flag(f, c.fetch_inactive, "fi")
            }
            Control::ScalarMemory(c) => {
                write!(f, "dwords:{} offset:{}", c.destination_count, c.immediate_offset_bytes)?;
                if let Some(d) = c.dynamic_offset_register {
                    write!(f, " soffset:{}", scalar_name(d))?;
                }
                Ok(())
            }
            Control::DataShare(c) => {
                write!(f, "offset0:{} offset1:{}", c.offset0, c.offset1)?;
                flag(f, c.gds, "gds")
            }
        }
    }
}

fn operand_list(f: &mut fmt::Formatter<'_>, ops: &[Operand]) -> fmt::Result {
    for (i, op) in ops.iter().enumerate() {
        if i != 0 {
            f.write_str(", ")?;
        }
        write!(f, "{op}")?;
    }
    Ok(())
}

impl fmt::Display for Instruction {
    /// `mnemonic dsts <- srcs {control}`; SOPP branches show their target.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.opcode)?;
        if !self.destinations.is_empty() {
            f.write_str(" ")?;
            operand_list(f, &self.destinations)?;
            if !self.sources.is_empty() {
                f.write_str(" <-")?;
            }
        }
        if !self.sources.is_empty() {
            f.write_str(" ")?;
            operand_list(f, &self.sources)?;
        }
        if self.encoding == Encoding::Sopp {
            if let Some(target) = crate::ir::branch::branch_target(self) {
                write!(f, " -> 0x{target:x}")?;
            } else if self.sources.is_empty() && self.words[0] & 0xFFFF != 0 {
                write!(f, " 0x{:x}", self.words[0] & 0xFFFF)?;
            }
        }
        if let Some(c) = &self.control {
            write!(f, " {{{c}}}")?;
        }
        Ok(())
    }
}

impl fmt::Display for Program {
    /// One line per instruction: `pc: words  disassembly`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for ins in &self.instructions {
            let words: Vec<String> = ins.words.iter().map(|w| format!("{w:08x}")).collect();
            writeln!(f, "{:06x}: {:<28} {}", ins.pc, words.join(" "), ins)?;
        }
        Ok(())
    }
}
