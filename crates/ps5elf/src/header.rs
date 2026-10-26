// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! SELF container constants.

/// SELF magic for PS5 debug / fakepkg builds (little-endian: `4F 15 3D 1D`).
pub const SELF_MAGIC: [u8; 4] = [0x4F, 0x15, 0x3D, 0x1D];

/// Alternative SELF magic (second variant observed in the wild).
pub const SELF_MAGIC_ALT: [u8; 4] = [0x54, 0x14, 0xF5, 0xEE];

/// Segment "blocked" flag. When set the segment entry carries real file
/// payload and is associated with a program header through
/// `(bits >> 20) & 0xFFF`.
pub const SELF_SEG_FLAG_BLOCKED: u64 = 0x800;

/// Per-segment encryption flag (data needs runtime decryption).
pub const SELF_SEG_FLAG_ENCRYPTED: u64 = 0x2;

/// Per-segment compression flag (payload is compressed; on a dumped fSELF
/// the data is already decompressed in-place).
pub const SELF_SEG_FLAG_COMPRESSED: u64 = 0x8;

/// Native page size used throughout SELF layout.
pub const PAGE_SIZE: u64 = 0x1000;

/// ELF e_ident[EI_ABIVERSION] == 2 marks a next-generation (PS5) image.
pub const ABI_VERSION_NEXT_GEN: u8 = 2;

/// Default load base for the main PS5 executable. Program-header virtual
/// addresses are relative; the chosen base is added at runtime.
pub const PS5_MAIN_IMAGE_BASE: u64 = 0x0000_0008_0000_0000;

/// Default load base for a PS4 executable.
pub const PS4_MAIN_IMAGE_BASE: u64 = 0x0000_0000_0040_0000;

/// Module placement search window for additional modules (PRX/Sprx).
pub const PS5_MODULE_SEARCH_START: u64 = 0x0000_0008_0400_0000;
pub const PS5_MODULE_SEARCH_END: u64 = 0x0000_0009_0000_0000;
pub const PS4_MODULE_SEARCH_START: u64 = 0x0000_0000_0200_0000;
pub const PS4_MODULE_SEARCH_END: u64 = 0x0000_0000_4000_0000;
pub const MODULE_PLACEMENT_STEP: u64 = 0x0020_0000;
