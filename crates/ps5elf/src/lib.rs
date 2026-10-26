// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! PlayStation 5 SELF / ELF loader.
//!
//! PS5 executables are Orbis (FreeBSD-ABI) ELF64 files wrapped in a SELF
//! container. The SELF header magic is `4F 15 3D 1D` (debug / fakepkg) or
//! `54 14 F5 EE`. On a dumped fSELF the segment payloads are already in
//! plaintext (decompressed, unencrypted); this loader does not perform
//! runtime decryption of retail images.

pub mod header;
pub mod self_loader;
pub mod elf;
pub mod dynamic;
pub mod reloc;

pub use self_loader::{
    parse_self, parse_self_headers, self_to_elf, LoadedProgramSegment, ProgramKind, SelfExtInfo, SelfHeaders, SelfImage,
    SelfLoadError, SelfSegment,
};
pub use elf::{validate_orbis_ehdr, Elf64Ehdr, Elf64Phdr, ElfType, ProgramType};
pub use header::SELF_MAGIC;
pub use reloc::{
    add_signed, compute_value, value_kind, write_kind, write_size, write_value, Elf64Rela,
    Elf64Sym, R_X86_64_DTPMOD64, R_X86_64_DTPOFF64, R_X86_64_NONE, R_X86_64_RELATIVE,
    R_X86_64_RELATIVE64, R_X86_64_TPOFF64, STB_GLOBAL, STB_LOCAL, STB_WEAK, STT_OBJECT,
    ValueKind, WriteKind,
};
