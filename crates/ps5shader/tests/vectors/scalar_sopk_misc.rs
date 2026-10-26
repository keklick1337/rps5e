// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

// Generated from hardware-measured reference rows; do not edit by hand.

#[allow(dead_code)]
pub const CODE: [u32; 31] = [
    0x34060086, 0xbe9403ff, 0x11111111, 0xbe9503ff, 0x22222222, 0xbe9703ff, 0x33333333, 0xbf068080,
    0xb1148001, 0xbf068180, 0xb1157fff, 0xbf068080, 0xb1171234, 0xb0801234, 0xba801801, 0x00000000,
    0xbe9603ff, 0x44444444, 0x7e140214, 0x7e160215, 0x7e180216, 0x7e1a0217, 0xe0701000, 0x80010a03,
    0xe0701004, 0x80010b03, 0xe0701008, 0x80010c03, 0xe070100c, 0x80010d03, 0xbf810000,
];

#[allow(dead_code)]
pub const ROUND_TOWARD_ZERO_CODE: [u32; 3] = [
    0xba801801, 0x0000000f, 0xbf810000,
];

#[allow(dead_code)]
pub const EXPECTED: [u32; 4] = [0xffff8001, 0x22222222, 0x44444444, 0x00001234];

#[allow(dead_code)]
pub const NAMES: [Option<&str>; 4] = [
    Some("cmovk_taken"), Some("cmovk_not_taken"), Some("after_setreg_imm32"), Some("cmovk_positive"),
];
