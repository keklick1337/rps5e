// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! x86-64 / Orbis relocations and symbol table.

use bytemuck::{Pod, Zeroable};

// Relocation types.
pub const R_X86_64_NONE: u32 = 0;
pub const R_X86_64_64: u32 = 1;
pub const R_X86_64_PC32: u32 = 2;
pub const R_X86_64_PLT32: u32 = 4;
pub const R_X86_64_GLOB_DAT: u32 = 6;
pub const R_X86_64_JUMP_SLOT: u32 = 7;
pub const R_X86_64_RELATIVE: u32 = 8;
pub const R_X86_64_UNSIGNED32: u32 = 10;
pub const R_X86_64_SIGNED32: u32 = 11;
pub const R_X86_64_DTPMOD64: u32 = 16;
pub const R_X86_64_DTPOFF64: u32 = 17;
pub const R_X86_64_TPOFF64: u32 = 18;
pub const R_X86_64_PC64: u32 = 24;
pub const R_X86_64_SIZE32: u32 = 32;
pub const R_X86_64_SIZE64: u32 = 33;
pub const R_X86_64_RELATIVE64: u32 = 38;

// Symbol binding (st_info high nibble).
pub const STB_LOCAL: u8 = 0;
pub const STB_GLOBAL: u8 = 1;
pub const STB_WEAK: u8 = 2;
// Symbol type (st_info low nibble).
pub const STT_NOTYPE: u8 = 0;
pub const STT_OBJECT: u8 = 1;
pub const STT_FUNC: u8 = 2;

/// Symbol table entry (Elf64_Sym, 24 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
pub struct Elf64Sym {
    pub st_name: u32,
    pub st_info: u8,
    pub st_other: u8,
    pub st_shndx: u16,
    pub st_value: u64,
    pub st_size: u64,
}

impl Elf64Sym {
    pub fn bind(&self) -> u8 {
        self.st_info >> 4
    }
    pub fn type_(&self) -> u8 {
        self.st_info & 0x0f
    }
}

/// Relocation entry (Elf64_Rela, 24 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Elf64Rela {
    pub r_offset: u64,
    pub r_info: u64,
    pub r_addend: i64,
}

impl Elf64Rela {
    pub fn sym(&self) -> u32 {
        (self.r_info >> 32) as u32
    }
    pub fn rtype(&self) -> u32 {
        (self.r_info & 0xffff_ffff) as u32
    }
}

/// How a relocation value is computed from the resolved symbol value S,
/// the addend A, and the patch target address P.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    /// S + A (or just B+A for relative relocs where S=image_base).
    Pointer,
    /// S + A - P.
    PcRelative,
    /// The symbol's size (st_size).
    SymbolSize,
}

/// How many bytes to write and how to interpret them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteKind {
    UInt64,
    UInt32,
    Int32,
}

/// Write size in bytes for a relocation type (4 or 8).
pub fn write_size(rtype: u32) -> usize {
    match rtype {
        R_X86_64_PC32
        | R_X86_64_PLT32
        | R_X86_64_SIGNED32
        | R_X86_64_UNSIGNED32
        | R_X86_64_SIZE32 => 4,
        _ => 8,
    }
}

/// Whether the linker knows how to compute and write this relocation type.
/// Others (TLS descriptors, IRELATIVE, 32-bit TLS offsets) must not be
/// written as 8-byte pointers: that would corrupt the neighbouring bytes.
pub fn is_supported(rtype: u32) -> bool {
    matches!(
        rtype,
        R_X86_64_NONE
            | R_X86_64_64
            | R_X86_64_PC32
            | R_X86_64_PLT32
            | R_X86_64_GLOB_DAT
            | R_X86_64_JUMP_SLOT
            | R_X86_64_RELATIVE
            | R_X86_64_UNSIGNED32
            | R_X86_64_SIGNED32
            | R_X86_64_DTPMOD64
            | R_X86_64_DTPOFF64
            | R_X86_64_TPOFF64
            | R_X86_64_PC64
            | R_X86_64_SIZE32
            | R_X86_64_SIZE64
            | R_X86_64_RELATIVE64
    )
}

/// Value kind for a symbol-based relocation type.
pub fn value_kind(rtype: u32) -> ValueKind {
    match rtype {
        R_X86_64_PC32 | R_X86_64_PLT32 | R_X86_64_PC64 => ValueKind::PcRelative,
        R_X86_64_SIZE32 | R_X86_64_SIZE64 => ValueKind::SymbolSize,
        _ => ValueKind::Pointer,
    }
}

/// Write kind for a relocation type.
pub fn write_kind(rtype: u32) -> WriteKind {
    match rtype {
        R_X86_64_PC32 | R_X86_64_PLT32 | R_X86_64_SIGNED32 => WriteKind::Int32,
        R_X86_64_UNSIGNED32 | R_X86_64_SIZE32 => WriteKind::UInt32,
        _ => WriteKind::UInt64,
    }
}

/// Add a signed addend to an unsigned value with wrapping.
pub fn add_signed(value: u64, addend: i64) -> u64 {
    if addend >= 0 {
        value.wrapping_add(addend as u64)
    } else {
        let mag = addend.wrapping_neg() as u64;
        value.wrapping_sub(mag)
    }
}

/// Compute the final value for a symbol-based relocation given the resolved
/// symbol value `s`, the addend `a`, the patch target `p`, and the kind.
pub fn compute_value(kind: ValueKind, s: u64, a: i64, p: u64) -> u64 {
    let base = add_signed(s, a);
    match kind {
        ValueKind::PcRelative => base.wrapping_sub(p),
        ValueKind::Pointer | ValueKind::SymbolSize => base,
    }
}

/// Write a relocation value into a byte buffer at the given offset.
pub fn write_value(buf: &mut [u8], off: usize, value: u64, kind: WriteKind) {
    match kind {
        WriteKind::UInt64 => buf[off..off + 8].copy_from_slice(&value.to_le_bytes()),
        WriteKind::UInt32 => buf[off..off + 4].copy_from_slice(&(value as u32).to_le_bytes()),
        WriteKind::Int32 => buf[off..off + 4].copy_from_slice(&(value as i32).to_le_bytes()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_types_cover_every_written_kind() {
        for t in [R_X86_64_NONE, R_X86_64_64, R_X86_64_PC32, R_X86_64_PLT32, R_X86_64_GLOB_DAT, R_X86_64_JUMP_SLOT, R_X86_64_RELATIVE, R_X86_64_DTPMOD64, R_X86_64_DTPOFF64, R_X86_64_TPOFF64, R_X86_64_RELATIVE64] {
            assert!(is_supported(t), "type {t}");
        }
        // TPOFF32, DTPOFF32, TLSDESC, IRELATIVE: 4-byte or special fields.
        for t in [21, 23, 36, 37] {
            assert!(!is_supported(t), "type {t}");
        }
    }
}
