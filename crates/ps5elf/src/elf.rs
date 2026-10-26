// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! ELF64 structures and helpers (Orbis flavor).

use bytemuck::{Pod, Zeroable};

/// ELF type. Orbis uses custom values: `0xfe10` (dyn-exec), `0xfe18` (shared).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElfType {
    DynExec, // 0xfe10
    Shared,  // 0xfe18
    Other(u16),
}

impl ElfType {
    pub const ET_DYNEXEC: u16 = 0xfe10;
    pub const ET_DYNAMIC: u16 = 0xfe18;
    pub fn from_raw(v: u16) -> Self {
        match v {
            Self::ET_DYNEXEC => Self::DynExec,
            Self::ET_DYNAMIC => Self::Shared,
            other => Self::Other(other),
        }
    }
}

/// Program (segment) header type. Standard + Orbis-specific.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramType {
    Load,
    Dynamic,
    Tls,
    SceDynlibData, // 0x61000000
    SceProcParam,  // 0x61000001
    SceRelRo,      // 0x61000010
    Other(u32),
}

impl ProgramType {
    pub const PT_LOAD: u32 = 1;
    pub const PT_DYNAMIC: u32 = 2;
    pub const PT_TLS: u32 = 7;
    pub const PT_SCE_DYNLIBDATA: u32 = 0x6100_0000;
    pub const PT_SCE_PROCPARAM: u32 = 0x6100_0001;
    pub const PT_SCE_RELRO: u32 = 0x6100_0010;
    pub fn from_raw(v: u32) -> Self {
        match v {
            Self::PT_LOAD => Self::Load,
            Self::PT_DYNAMIC => Self::Dynamic,
            Self::PT_TLS => Self::Tls,
            Self::PT_SCE_DYNLIBDATA => Self::SceDynlibData,
            Self::PT_SCE_PROCPARAM => Self::SceProcParam,
            Self::PT_SCE_RELRO => Self::SceRelRo,
            other => Self::Other(other),
        }
    }

    /// Segments mapped into the image: PT_LOAD and PT_SCE_RELRO.
    pub fn is_mapped(self) -> bool {
        matches!(self, Self::Load | Self::SceRelRo)
    }
}

pub const PT_GNU_EH_FRAME: u32 = 0x6474_e550;

pub const PF_X: u32 = 0x1;
pub const PF_W: u32 = 0x2;
pub const PF_R: u32 = 0x4;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Elf64Ehdr {
    pub e_ident: [u8; 16],
    pub e_type: u16,
    pub e_machine: u16,
    pub e_version: u32,
    pub e_entry: u64,
    pub e_phoff: u64,
    pub e_shoff: u64,
    pub e_flags: u32,
    pub e_ehsize: u16,
    pub e_phentsize: u16,
    pub e_phnum: u16,
    pub e_shentsize: u16,
    pub e_shnum: u16,
    pub e_shstrndx: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Elf64Phdr {
    pub p_type: u32,
    pub p_flags: u32,
    pub p_offset: u64,
    pub p_vaddr: u64,
    pub p_paddr: u64,
    pub p_filesz: u64,
    pub p_memsz: u64,
    pub p_align: u64,
}

impl Elf64Phdr {
    pub fn is_executable(&self) -> bool {
        self.p_flags & PF_X != 0
    }
    pub fn is_writable(&self) -> bool {
        self.p_flags & PF_W != 0
    }
    pub fn is_readable(&self) -> bool {
        self.p_flags & PF_R != 0
    }
}

/// Validate an Orbis ELF header (FreeBSD ABI, x86-64, ET_DYNEXEC/ET_DYNAMIC).
pub fn validate_orbis_ehdr(ehdr: &Elf64Ehdr) -> Result<(), &'static str> {
    if &ehdr.e_ident[0..4] != b"\x7fELF" {
        return Err("not an ELF file (bad magic)");
    }
    if ehdr.e_ident[4] != 2 /* ELFCLASS64 */ {
        return Err("not ELF64");
    }
    if ehdr.e_ident[5] != 1 /* ELFDATA2LSB */ {
        return Err("not little-endian");
    }
    if ehdr.e_ident[6] != 1 /* EV_CURRENT */ {
        return Err("e_ident[EI_VERSION] != EV_CURRENT");
    }
    if ehdr.e_ident[7] != 9 /* ELFOSABI_FREEBSD */ {
        return Err("OS/ABI is not FreeBSD");
    }
    if ehdr.e_ident[8] != 0 && ehdr.e_ident[8] != 2 {
        return Err("ABI version is not 0 or 2");
    }
    if ehdr.e_type != ElfType::ET_DYNEXEC && ehdr.e_type != ElfType::ET_DYNAMIC {
        return Err("type is neither ET_DYNEXEC nor ET_DYNAMIC");
    }
    if ehdr.e_machine != 62 /* EM_X86_64 */ {
        return Err("not x86-64");
    }
    if ehdr.e_version != 1 {
        return Err("e_version != EV_CURRENT");
    }
    if ehdr.e_phentsize as usize != std::mem::size_of::<Elf64Phdr>() {
        return Err("phentsize mismatch");
    }
    if ehdr.e_shentsize != 0 && ehdr.e_shentsize != 64 {
        return Err("shentsize mismatch");
    }
    Ok(())
}
