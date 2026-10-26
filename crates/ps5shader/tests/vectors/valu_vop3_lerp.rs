// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

// Vop3Lerp: hardware-measured reference rows (raw RDNA code and tables).
#[allow(dead_code)]
mod tables {
    pub const THREADS: u32 = 64;
    pub const INPUTS: u32 = 4;
    pub const RESULTS: u32 = 16;
    pub const LITERAL: u32 = 0x01fe01fe;
    pub const EDGES: [[u32; 4]; 12] = [
        [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff],
        [0xffffffff, 0, 0x01010101, 0x80808080],
        [0xffffffff, 0, 0, 0x7f7f7f7f],
        [0, 0, 0xffffffff, 0],
        [0x01010101, 0, 0x01010101, 0x01010101],
        [0x01010101, 0, 0xfefefefe, 0],
        [0xff00ff00, 0x00ff00ff, 0x01000100, 0x807f807f],
        [0x80808080, 0x7f7f7f7f, 0x00010001, 0x7f807f80],
        [0x000000ff, 0x0000ff00, 0x00010000, 0x00007f7f],
        [0x12345678, 0x9abcdef0, 0xffffffff, 0x56789ab4],
        [0xfe01fe01, 0x01fe01fe, 0x80808080, 0x7f7f7f7f],
        [0, 0, 0, 0],
    ];
    pub const LERP_CODE: [u32; 44] = [
        0x34020082, 0x34060084, 0xe0302000, 0x80000401, 0xe0302004, 0x80000501, 0xe0302008, 0x80000601,
        0xbf8c3f70, 0xd54d000a, 0x041a0b04, 0xd54d000b, 0x041a0905, 0xd54d000c, 0x02020b04, 0xd54d000d,
        0x03060b04, 0xd54d000e, 0x03fe0b04, LITERAL, 0xd54d000f, 0x041a0ac1, 0xd54d0010, 0x041a0904,
        0x7e220304, 0xd54d0011, 0x041a0b11, 0xe0702000, 0x80010a03, 0xe0702004, 0x80010b03, 0xe0702008,
        0x80010c03, 0xe070200c, 0x80010d03, 0xe0702010, 0x80010e03, 0xe0702014, 0x80010f03, 0xe0702018,
        0x80011003, 0xe070201c, 0x80011103, 0xbf810000,
    ];
}
use tables::*;
