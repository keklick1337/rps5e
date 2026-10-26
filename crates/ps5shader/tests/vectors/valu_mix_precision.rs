// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

// MixPrecision: hardware-measured reference rows (raw RDNA code and tables).
#[allow(dead_code)]
mod tables {
    pub const THREADS: u32 = 64;
    pub const INPUTS: u32 = 4;
    pub const RESULTS: u32 = 16;
    pub const MIX_CODE: [u32; 52] = [
        0x34020082, 0x34060084, 0xe0302000, 0x80000401, 0xe0302004, 0x80000501, 0xe0302008, 0x80000601,
        0xe030200c, 0x80000701, 0x7e2002ff, 0xabcd0000, 0x7e2202ff, 0x0000abcd, 0xbf8c3f70, 0xcc20400a,
        0x1c1a0b04, 0xcc20780b, 0x1c1a0b04, 0xcc20400c, 0x941a0af2, 0xcc20700d, 0x941a0af2, 0xcc20100e,
        0x141e0907, 0xcc204a0f, 0x1c1a0b04, 0xcc214010, 0x1c1a0b04, 0xcc227811, 0x1c1a0b04, 0xcc203812,
        0x041e0f07, 0xe0702000, 0x80010a03, 0xe0702004, 0x80010b03, 0xe0702008, 0x80010c03, 0xe070200c,
        0x80010d03, 0xe0702010, 0x80010e03, 0xe0702014, 0x80010f03, 0xe0702018, 0x80011003, 0xe070201c,
        0x80011103, 0xe0702020, 0x80011203, 0xbf810000,
    ];
}
use tables::*;
