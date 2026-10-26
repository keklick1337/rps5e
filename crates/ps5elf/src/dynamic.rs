// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Dynamic table + Orbis/SCE custom tags.

/// Orbis custom dynamic tags (`DT_OS_*`).
pub mod dt {
    pub const DT_NULL: i64 = 0;
    pub const DT_NEEDED: i64 = 0x01;
    pub const DT_PLTRELSZ: i64 = 0x02;
    pub const DT_PLTGOT: i64 = 0x03;
    pub const DT_HASH: i64 = 0x04;
    pub const DT_STRTAB: i64 = 0x05;
    pub const DT_SYMTAB: i64 = 0x06;
    pub const DT_RELA: i64 = 0x07;
    pub const DT_RELASZ: i64 = 0x08;
    pub const DT_RELAENT: i64 = 0x09;
    pub const DT_STRSZ: i64 = 0x0a;
    pub const DT_SYMENT: i64 = 0x0b;
    pub const DT_INIT: i64 = 0x0c;
    pub const DT_FINI: i64 = 0x0d;
    pub const DT_SONAME: i64 = 0x0e;
    pub const DT_INIT_ARRAY: i64 = 0x19;
    pub const DT_FINI_ARRAY: i64 = 0x1a;
    pub const DT_INIT_ARRAYSZ: i64 = 0x1b;
    pub const DT_PREINIT_ARRAY: i64 = 0x20;
    pub const DT_PREINIT_ARRAYSZ: i64 = 0x21;
    pub const DT_FLAGS: i64 = 0x1e;
    pub const DT_REL: i64 = 0x11;
    pub const DT_RELCOUNT: i64 = 0x6fff_fff9;
    pub const DT_JMPREL: i64 = 0x17;
    pub const DT_PLTREL: i64 = 0x14;
    pub const DT_FINI_ARRAYSZ: i64 = 0x1c;

    // SCE / Orbis custom tags (0x61000000 family).
    pub const DT_SCE_NEEDED_MODULE: i64 = 0x6100_000f;
    pub const DT_SCE_MODULE_ATTR: i64 = 0x6100_0011;
    pub const DT_SCE_MODULE_INFO: i64 = 0x6100_000d;
    pub const DT_SCE_EXPORT_LIB: i64 = 0x6100_0013;
    pub const DT_SCE_IMPORT_LIB: i64 = 0x6100_0015;
    pub const DT_SCE_FINGERPRINT: i64 = 0x6100_0007;
    pub const DT_SCE_ORIGINAL_FILENAME: i64 = 0x6100_0009;
    pub const DT_SCE_HASH: i64 = 0x6100_0025;
    pub const DT_SCE_HASHSZ: i64 = 0x6100_003d;
    pub const DT_SCE_PLTGOT: i64 = 0x6100_0027;
    pub const DT_SCE_JMPREL: i64 = 0x6100_0029;
    pub const DT_SCE_PLTREL: i64 = 0x6100_002b;
    pub const DT_SCE_PLTRELSZ: i64 = 0x6100_002d;
    pub const DT_SCE_RELA: i64 = 0x6100_002f;
    pub const DT_SCE_RELASZ: i64 = 0x6100_0031;
    pub const DT_SCE_RELAENT: i64 = 0x6100_0033;
    pub const DT_SCE_STRTAB: i64 = 0x6100_0035;
    pub const DT_SCE_STRSZ: i64 = 0x6100_0037;
    pub const DT_SCE_SYMTAB: i64 = 0x6100_0039;
    pub const DT_SCE_SYMENT: i64 = 0x6100_003b;
    pub const DT_SCE_SYMTABSZ: i64 = 0x6100_003f;
    pub const DT_SCE_ORIGINAL_FILENAME_1: i64 = 0x6100_0041;
    pub const DT_SCE_MODULE_INFO_1: i64 = 0x6100_0043;
    pub const DT_SCE_NEEDED_MODULE_1: i64 = 0x6100_0045;
    pub const DT_SCE_EXPORT_LIB_1: i64 = 0x6100_0047;
    pub const DT_SCE_IMPORT_LIB_1: i64 = 0x6100_0049;
    pub const DT_SCE_EXPORT_LIB_ATTR: i64 = 0x6100_0017;
    pub const DT_SCE_IMPORT_LIB_ATTR: i64 = 0x6100_0019;
    pub const DT_DEBUG: i64 = 0x15;
    pub const DT_TEXTREL: i64 = 0x16;
    pub const DT_RELACOUNT: i64 = 0x6fff_fff9;
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Elf64Dyn {
    pub d_tag: i64,
    pub d_un: u64,
}

/// Parsed dynamic table.
#[derive(Default, Debug, Clone)]
pub struct DynamicTable {
    pub entries: Vec<(i64, u64)>,
}

impl DynamicTable {
    pub fn parse(bytes: &[u8]) -> Self {
        let mut entries = Vec::new();
        let mut off = 0;
        while off + 16 <= bytes.len() {
            let tag = i64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
            let val = u64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap());
            off += 16;
            entries.push((tag, val));
            if tag == dt::DT_NULL {
                break;
            }
        }
        Self { entries }
    }

    pub fn get(&self, tag: i64) -> Option<u64> {
        self.entries
            .iter()
            .find(|(t, _)| *t == tag)
            .map(|(_, v)| *v)
    }

    pub fn has(&self, tag: i64) -> bool {
        self.get(tag).is_some()
    }

    /// Every value of `tag`, in table order.
    pub fn get_all(&self, tag: i64) -> impl Iterator<Item = u64> + '_ {
        self.entries.iter().filter(move |(t, _)| *t == tag).map(|(_, v)| *v)
    }
}
