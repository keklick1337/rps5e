// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Scalar memory addressing: base + sign-extended 21-bit immediate +
//! zero-extended SOFFSET.

use ps5shader::interp::Machine;
use ps5shader::testing::asm::*;
use ps5shader::testing::{buffer_descriptor, ComputeRun};

const OUT: u64 = 0x2_0000_0000;
const DATA: u64 = 0x5_0000_0100;
const SGPR_NULL: u32 = 0x7D;

/// s_load_dword s8, s[4:5], `offset` (+ s6 when `soffset`), stored to the
/// output buffer in s[0:3].
fn load(offset: i32, soffset: Option<u32>) -> u32 {
    let mut code = smem(0x00, 8, 2, offset as u32 & 0x1F_FFFF, if soffset.is_some() { 6 } else { SGPR_NULL });
    code.push(sopp(0x0C, 0)); // s_waitcnt
    code.push(vop1(OP_V_MOV_B32, 1, 8));
    code.extend(mubuf(OP_BUFFER_STORE_DWORD, 1, 0, 0, inline_u32(0), 0, true, false, false));
    code.push(S_ENDPGM);
    let mut user = buffer_descriptor(OUT, 4, 1, 0x1101_6FAC).to_vec();
    user.extend([DATA as u32, (DATA >> 32) as u32, soffset.unwrap_or(0)]);
    let run = ComputeRun::new(code, 32, 1, user);
    let words: Vec<u32> = (0..128).map(|i| 0x1000 + i).collect();
    let setup = move |m: &mut Machine| {
        m.memory.write_words(DATA - 0x100, &words);
        m.memory.write_words(OUT, &[0xDEAD_BEEF]);
    };
    let m = run.run_checked(&setup, &[(OUT, 1)]).unwrap();
    m.memory.read_words(OUT, 1)[0]
}

#[test]
fn negative_immediate_reaches_below_the_base() {
    // DATA - 0x100 holds 0x1000 + i at word i: DATA is word 64.
    assert_eq!(load(0, None), 0x1000 + 64);
    assert_eq!(load(-4, None), 0x1000 + 63);
    assert_eq!(load(-0x100, None), 0x1000);
}

#[test]
fn soffset_adds_unsigned() {
    assert_eq!(load(-8, Some(16)), 0x1000 + 66);
    assert_eq!(load(4, Some(8)), 0x1000 + 67);
}
