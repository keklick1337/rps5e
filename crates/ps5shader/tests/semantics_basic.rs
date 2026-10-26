// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Hand-written semantics tests: control flow, EXEC masking, lane masks and
//! cross-lane operations, run through translation + SSA + optimization in
//! the interpreter (optimized and unoptimized runs must agree).

use ps5shader::testing::asm::*;
use ps5shader::testing::IoTest;

/// v1 = tid * 4 (byte offset), load input[tid] into v2.
fn prologue() -> Vec<u32> {
    let mut c = vec![vop2(0x1A, 1, inline_u32(2), 0)]; // v_lshlrev_b32 v1, 2, v0
    c.extend(mubuf(OP_BUFFER_LOAD_DWORD, 2, 0, 0, inline_u32(0), 0, true, false, false));
    c
}

/// Store v`reg` to output[tid].
fn store(reg: u32) -> Vec<u32> {
    mubuf(OP_BUFFER_STORE_DWORD, reg, 0, 1, inline_u32(0), 0, true, false, false)
}

#[test]
fn float_add_and_store() {
    let mut code = prologue();
    code.push(vop2(0x03, 3, F_ONE, 2)); // v_add_f32 v3, 1.0, v2
    code.extend(store(3));
    code.push(S_ENDPGM);
    let t = IoTest::new(&code, 32, 32);
    let input: Vec<u32> = (0..32).map(|i| (i as f32).to_bits()).collect();
    let out = t.run(&input).unwrap();
    for i in 0..32 {
        assert_eq!(f32::from_bits(out[i]), i as f32 + 1.0);
    }
}

#[test]
fn exec_if_else() {
    // if (v2 < 16) v3 = 1 else v3 = 2, via v_cmp + s_and_saveexec /
    // s_xor + s_cbranch_execz.
    let mut code = prologue();
    code.push(vop1(OP_V_MOV_B32, 3, inline_u32(0)));
    code.push(vopc(0xC1, inline_u32(16), 2)); // v_cmp_lt_u32 vcc, 16, v2  (16 < v2)
    code.push(sop1(0x24, 10, VCC_LO)); // s_and_saveexec_b64 s[10:11], vcc
    code.push(branch(SOPP_CBRANCH_EXECZ, 1));
    code.push(vop1(OP_V_MOV_B32, 3, inline_u32(2)));
    // else
    code.push(sop2(0x13, EXEC_LO, 10, EXEC_LO)); // s_xor_b64 exec, s[10:11], exec
    code.push(branch(SOPP_CBRANCH_EXECZ, 1));
    code.push(vop1(OP_V_MOV_B32, 3, inline_u32(1)));
    code.push(sop2(0x11, EXEC_LO, EXEC_LO, 10)); // s_or_b64 exec, exec, s[10:11]
    code.extend(store(3));
    code.push(S_ENDPGM);
    for wave in [32, 64] {
        let t = IoTest::with_wave(&code, wave, 32, 32);
        let input: Vec<u32> = (0..32).collect();
        let out = t.run(&input).unwrap();
        for i in 0..32 {
            assert_eq!(out[i], if i > 16 { 2 } else { 1 }, "wave{wave} lane {i}");
        }
    }
}

#[test]
fn divergent_loop() {
    // v3 = 0; loop { exec = v3 < v2 (per lane); if none: break; v3 += 1 }
    let mut code = prologue();
    code.push(vop1(OP_V_MOV_B32, 3, inline_u32(0)));
    code.push(sop1(0x04, 12, EXEC_LO)); // s_mov_b64 s[12:13], exec
    // loop:
    let loop_start = code.len();
    code.push(vopc(0xD1, vgpr(3), 2)); // v_cmpx_lt_u32 v3, v2
    code.push(branch(SOPP_CBRANCH_EXECZ, 2));
    code.push(vop2(0x25, 3, inline_u32(1), 3)); // v_add_nc_u32 v3, 1, v3
    let delta = loop_start as i32 - (code.len() as i32 + 1);
    code.push(branch(SOPP_BRANCH, delta));
    code.push(sop1(0x04, EXEC_LO, 12)); // s_mov_b64 exec, s[12:13]
    code.extend(store(3));
    code.push(S_ENDPGM);
    for wave in [32, 64] {
        let t = IoTest::with_wave(&code, wave, 32, 32);
        let input: Vec<u32> = (0..32).map(|i| (i * 7) % 13).collect();
        let out = t.run(&input).unwrap();
        for i in 0..32 {
            assert_eq!(out[i], input[i], "wave{wave} lane {i}");
        }
    }
}

#[test]
fn lane_mask_count_and_readlane() {
    // s[4:5] = mask of lanes with odd input; s6 = popcount; v3 = s6
    // + readlane(v2, 5) + readfirstlane.
    let mut code = prologue();
    code.push(vop2(0x1B, 4, inline_u32(1), 2)); // v_and_b32 v4, 1, v2
    code.extend(vop3(0xC5, 20, inline_u32(0), 256 + 4, 0)); // v_cmp_ne_u32 s[20:21], 0, v4
    code.push(sop1(0x10, 16, 20)); // s_bcnt1_i32_b64 s16, s[20:21]
    code.extend(vop3(0x360, 17, 256 + 2, inline_u32(5), 0)); // v_readlane_b32 s17, v2, 5
    code.push(sop2(0x00, 16, 16, 17)); // s_add_u32 s16, s16, s17
    code.push(vop1(OP_V_MOV_B32, 3, 16));
    code.extend(store(3));
    code.push(S_ENDPGM);
    for wave in [32, 64] {
        let t = IoTest::with_wave(&code, wave, 32, 32);
        let input: Vec<u32> = (0..32).map(|i| i * 3).collect();
        let out = t.run(&input).unwrap();
        let odd = input.iter().filter(|&&x| x & 1 == 1).count() as u32;
        for i in 0..32 {
            assert_eq!(out[i], odd + input[5], "wave{wave} lane {i}");
        }
    }
}

#[test]
fn writelane_spill_round_trip() {
    // s8 = input[3] via readlane; spill s8 into v9 lane 2 with writelane,
    // restore with readlane into s9 and store.
    let mut code = prologue();
    code.extend(vop3(0x360, 8, 256 + 2, inline_u32(3), 0)); // v_readlane_b32 s8, v2, 3
    code.extend(vop3(0x361, 9, 8, inline_u32(2), 0)); // v_writelane_b32 v9, s8, 2
    code.extend(vop3(0x360, 9, 256 + 9, inline_u32(2), 0)); // v_readlane_b32 s9, v9, 2
    code.push(vop1(OP_V_MOV_B32, 3, 9));
    code.extend(store(3));
    code.push(S_ENDPGM);
    let t = IoTest::new(&code, 32, 32);
    let input: Vec<u32> = (0..32).map(|i| 1000 + i).collect();
    let out = t.run(&input).unwrap();
    assert!(out.iter().all(|&x| x == 1003));
}
