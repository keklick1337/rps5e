// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! v_min_i32 with SDWA (a title's encoding `0x220404f9 0x00060605`:
//! v_min_i32_sdwa v2, v5, v2 src1_sel:BYTE_0, zero-extended): the signed
//! minimum of src0 and the low byte of src1. Cases from the reference's
//! Vop2SdwaMinI32CapturedByte0 test; the last lane is inactive and keeps v2.

use ps5shader::testing::asm::*;
use ps5shader::testing::IoTest;

const CASES: [(u32, u32, u32); 9] = [
    (0xfffffffb, 0x12345678, 0xfffffffb),
    (0x00000064, 0xffffff80, 0x00000064),
    (0x000000c8, 0xffffff80, 0x00000080),
    (0x7fffffff, 0x000000ff, 0x000000ff),
    (0x80000000, 0x7fffffff, 0x80000000),
    (0x00000100, 0x00000105, 0x00000005),
    (0xffffffff, 0x00000000, 0xffffffff),
    (0x00000010, 0xdeadbe0f, 0x0000000f),
    (0x80000001, 0x7fff0123, 0x7fff0123),
];

#[test]
fn sdwa_min_i32_byte0() {
    let mut code = vec![
        vop2(0x1A, 1, inline_u32(1), 0), // v_lshlrev_b32 v1, 1, v0
        vop2(0x25, 3, inline_u32(1), 1), // v_add_nc_u32 v3, 1, v1
    ];
    code.extend(mubuf(OP_BUFFER_LOAD_DWORD, 5, 1, 0, inline_u32(0), 0, true, false, false));
    code.extend(mubuf(OP_BUFFER_LOAD_DWORD, 2, 3, 0, inline_u32(0), 0, true, false, false));
    code.extend([s_mov_b32(126, 255), 0xFF]); // exec_lo = 0xff: lane 8 inactive
    code.extend([0x2204_04F9, 0x0006_0605]);
    code.extend([s_mov_b32(126, 255), 0x1FF]);
    code.extend(mubuf(OP_BUFFER_STORE_DWORD, 2, 0, 1, inline_u32(0), 0, true, false, false));
    code.push(S_ENDPGM);
    let input: Vec<u32> = CASES.iter().flat_map(|c| [c.0, c.1]).collect();
    let out = IoTest::new(&code, 9, 18).run(&input).unwrap();
    for (lane, c) in CASES.iter().enumerate() {
        assert_eq!(out[lane], c.2, "lane {lane}: min_i32({:#x}, byte0({:#x}))", c.0, c.1);
    }
}
