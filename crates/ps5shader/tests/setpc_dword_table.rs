// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! A subtractive S_SETPC_B64 jump table of 32-bit offsets
//! (`s_load_dword sE, sP, sI; s_sub sP, sP, sE; setpc sP`) resolves whether
//! the loaded entry goes to its own register or back into the scaled
//! selector register (`s_load_dword s17, s[22:23], s17`).

use ps5shader::testing::asm::*;
use ps5shader::testing::{raw_buffer_descriptor, ComputeRun};

const OUTPUT: u64 = 0x2_0000_0000;
const VCC_LO: u32 = 106;

/// Selects case 0 (s1 = 1), case 1 (s1 = 2) or, for a selector clamped to
/// 2, case 0 again, then stores s1 to the output buffer (s[8:11]).
fn program(entry_sgpr: u32) -> Vec<u32> {
    let mut code = vec![
        sop2(0x07, VCC_LO, 0, inline_u32(2)),      // 0x00 s_min_u32 vcc_lo, s0, 2
        sop2(0x1e, VCC_LO, VCC_LO, inline_u32(2)), // 0x04 s_lshl_b32 vcc_lo, vcc_lo, 2
        sop1(0x1f, 4, 0),                          // 0x08 s_getpc_b64 s[4:5] (0x0c)
        sop2(0x00, 4, 4, 255),                     // 0x0c s_add_u32 s4, s4, literal
        60,                                        //      base 0x48
        sop2(0x04, 5, 5, inline_u32(0)),           // 0x14 s_addc_u32 s5, s5, 0
    ];
    code.extend(smem(0x00, entry_sgpr, 2, 4, VCC_LO)); // 0x18 s_load_dword sE, s[4:5], vcc_lo offset:4
    code.extend([
        sopp(0x0c, 0),                   // 0x20 s_waitcnt 0
        sop2(0x01, 4, 4, entry_sgpr),    // 0x24 s_sub_u32 s4, s4, sE
        sop2(0x05, 5, 5, inline_u32(0)), // 0x28 s_subb_u32 s5, s5, 0
        sop1(0x20, 0, 4),                // 0x2c s_setpc_b64 s[4:5]
        s_mov_b32(1, inline_u32(1)),     // 0x30 case 0
        sopp(0x02, 1),                   // 0x34 s_branch 0x3c
        s_mov_b32(1, inline_u32(2)),     // 0x38 case 1
        vop1(OP_V_MOV_B32, 1, 1),        // 0x3c v1 = s1
    ]);
    code.extend(mubuf(OP_BUFFER_STORE_DWORD, 1, 0, 2, inline_u32(0), 0, false, false, false)); // 0x40
    code.push(S_ENDPGM); // 0x48: the table base
    // Entries at base + 4: targets base - value (0x30, 0x38, 0x30).
    code.extend([0x18, 0x10, 0x18]);
    code
}

fn run(entry_sgpr: u32, selector: u32) -> u32 {
    let mut user = vec![0u32; 12];
    user[0] = selector;
    user[8..12].copy_from_slice(&raw_buffer_descriptor(OUTPUT, 64));
    let run = ComputeRun::new(program(entry_sgpr), 32, 1, user);
    let m = run.run_checked(&|m| m.memory.write_words(OUTPUT, &[0xDEAD_BEEF]), &[(OUTPUT, 1)]).unwrap_or_else(|e| panic!("entry s{entry_sgpr}: {e}"));
    m.memory.read_words(OUTPUT, 1)[0]
}

#[test]
fn dword_setpc_table_with_the_entry_in_its_own_register() {
    assert_eq!([0, 1, 2, 7].map(|s| run(6, s)), [1, 2, 1, 1]);
}

#[test]
fn dword_setpc_table_whose_load_reuses_the_selector() {
    assert_eq!([0, 1, 2, 7].map(|s| run(VCC_LO, s)), [1, 2, 1, 1]);
}
