// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

// Vop3Aliases: hardware-measured reference rows (raw RDNA code and tables).
#[allow(dead_code)]
mod tables {
    pub const THREADS: u32 = 64;
    pub const INPUTS: u32 = 4;
    pub const RESULTS: u32 = 16;
    pub const ALIAS_CODE: [u32; 52] = [
        0x34020082, 0x34060084, 0xe0302000, 0x80000401, 0xe0302004, 0x80000501, 0xe0302008, 0x80000601,
        0x7e1402ff, 0xabcd0000, 0x7e1602ff, 0xabcd0000, 0x7e1802ff, 0xabcd0000, 0xbf8c3f70, 0xd539020a,
        0x20020b04, 0xd53a010b, 0x40020b04, 0xd535010c, 0x20020b04, 0xd58b000d, 0x10000105, 0xd58b010e,
        0x20000104, 0xd591000f, 0x08000106, 0xd5920010, 0x18000106, 0xd5938011, 0x00000106, 0xd5940012,
        0x00000106, 0xe0702000, 0x80010a03, 0xe0702004, 0x80010b03, 0xe0702008, 0x80010c03, 0xe070200c,
        0x80010d03, 0xe0702010, 0x80010e03, 0xe0702014, 0x80010f03, 0xe0702018, 0x80011003, 0xe070201c,
        0x80011103, 0xe0702020, 0x80011203, 0xbf810000,
    ];
}
use tables::*;
