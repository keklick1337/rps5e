// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

// Vop3CarryOut: hardware-measured reference rows (raw RDNA code and tables).
#[allow(dead_code)]
mod tables {
    pub const THREADS: u32 = 64;
    pub const INPUTS: u32 = 4;
    pub const RESULTS: u32 = 16;
    pub const LITERAL: u32 = 0x80000000;
    pub const EDGES: [[u32; 8]; 14] = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0xffffffff, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 0xffffffff, 1],
        [0xffffffff, 1, 0, 1, 0xfffffffe, 0, 2, 1],
        [1, 0xffffffff, 0, 1, 2, 1, 0xfffffffe, 0],
        [0xffffffff, 0xffffffff, 0xfffffffe, 1, 0, 0, 0, 0],
        [0x80000000, 0x80000000, 0, 1, 0, 0, 0, 0],
        [0x7fffffff, 1, 0x80000000, 0, 0x7ffffffe, 0, 0x80000002, 1],
        [0x80000000, 0x7fffffff, 0xffffffff, 0, 1, 0, 0xffffffff, 1],
        [0x7fffffff, 0x80000000, 0xffffffff, 0, 0xffffffff, 1, 1, 0],
        [0, 0xffffffff, 0xffffffff, 0, 1, 1, 0xffffffff, 0],
        [0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0, 1, 1],
        [0x80000001, 0x80000000, 1, 1, 1, 0, 0xffffffff, 1],
        [0x12345678, 0x9abcdef0, 0xacf13568, 0, 0x77777788, 1, 0x88888878, 0],
    ];
    pub const CARRY_OUT_CODE: [u32; 73] = [
        0x34020082, 0x34060084, 0xe0302000, 0x80000401, 0xe0302004, 0x80000501, 0xbf8c3f70, 0xd70f080a,
        0x00020b04, 0xd501000b, 0x00210280, 0xd710080c, 0x00020b04, 0xd501000d, 0x00210280, 0xd719080e,
        0x00020b04, 0xd501000f, 0x00210280, 0xd70f6a10, 0x00020b04, 0xd5010011, 0x01a90280, 0xd7106a12,
        0x00020b04, 0xd5010013, 0x01a90280, 0xd7196a14, 0x00020b04, 0xd5010015, 0x01a90280, 0xd7100a16,
        0x000208ff, LITERAL, 0xd5010017, 0x00290280, 0xd7196a18, 0x00020a81, 0xd5010019, 0x01a90280,
        0xe0702000, 0x80010a03, 0xe0702004, 0x80010b03, 0xe0702008, 0x80010c03, 0xe070200c, 0x80010d03,
        0xe0702010, 0x80010e03, 0xe0702014, 0x80010f03, 0xe0702018, 0x80011003, 0xe070201c, 0x80011103,
        0xe0702020, 0x80011203, 0xe0702024, 0x80011303, 0xe0702028, 0x80011403, 0xe070202c, 0x80011503,
        0xe0702030, 0x80011603, 0xe0702034, 0x80011703, 0xe0702038, 0x80011803, 0xe070203c, 0x80011903,
        0xbf810000,
    ];
}
use tables::*;
