// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

// Vop3Sad: hardware-measured reference rows (raw RDNA code and tables).
#[allow(dead_code)]
mod tables {
    pub const THREADS: u32 = 64;
    pub const INPUTS: u32 = 4;
    pub const RESULTS: u32 = 16;
    pub const LITERAL: u32 = 0x8000ffff;
    pub const EDGES: [[u32; 3]; 12] = [
        [0xffffffff, 0, 0],
        [0, 0xffffffff, 0xffffffff],
        [0xff00ff00, 0x00ff00ff, 0xfffffc04],
        [0x80000000, 0x7fffffff, 1],
        [0x7f80ff00, 0x807f00ff, 0xffff0000],
        [0x01020304, 0x04030201, 0x10],
        [0x12345678, 0x12345678, 0x9abcdef0],
        [0x80808080, 0x7f7f7f7f, 0],
        [0x8000ffff, 0x7fff0000, 0xfffe0002],
        [0x00ff0001, 0x000000ff, 0x7fffffff],
        [0xffffffff, 0x00ff0000, 0x03fc],
        [0, 0, 0],
    ];
    pub const SAD_CODE: [u32; 51] = [
        0x34020082, 0x34060084, 0xe0302000, 0x80000401, 0xe0302004, 0x80000501, 0xe0302008, 0x80000601,
        0xbf8c3f70, 0xd55a000a, 0x041a0b04, 0xd55b000b, 0x041a0b04, 0xd55c000c, 0x041a0b04, 0xd571000d,
        0x041a0b04, 0xd55d000e, 0x041a0b04, 0xd55a000f, 0x04190104, 0xd55a0010, 0x041a0ac1, 0xd55b0011,
        0x042a0905, 0xd55c0012, 0x0419ff04, LITERAL, 0xd5710013, 0x041a0905, 0xe0702000, 0x80010a03,
        0xe0702004, 0x80010b03, 0xe0702008, 0x80010c03, 0xe070200c, 0x80010d03, 0xe0702010, 0x80010e03,
        0xe0702014, 0x80010f03, 0xe0702018, 0x80011003, 0xe070201c, 0x80011103, 0xe0702020, 0x80011203,
        0xe0702024, 0x80011303, 0xbf810000,
    ];
}
use tables::*;
