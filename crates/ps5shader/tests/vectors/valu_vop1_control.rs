// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

// Vop1Control: hardware-measured reference rows (raw RDNA code and tables).
#[allow(dead_code)]
mod tables {
    pub const THREADS: u32 = 64;
    pub const STRIDE: u32 = 4;
    pub const VOP1_CONTROL_CODE: [u32; 17] = [
        0x34020082, 0xe0302000, 0x80000401, 0xbf8c3f70, 0x7e000000, 0x7e003600, 0x7e008200, 0x4a080881,
        0xd5800000, 0x00000000, 0xd59b0000, 0x00000000, 0xd5c10000, 0x00000000, 0xe0702000, 0x80010401,
        0xbf810000,
    ];
}
use tables::*;
