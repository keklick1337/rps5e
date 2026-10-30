// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The EXEC-zero branch tests only the lanes the host runs
//! (`rewrites::PRESENT_EXEC`): a waterfall loop over a partial wave whose
//! missing lanes were enabled by `s_mov_b64 exec, -1` ends once every
//! present lane is served. The EXEC words keep the guest's bits.

use ps5shader::interp::Machine;
use ps5shader::testing::asm::*;
use ps5shader::testing::ComputeRun;
use ps5shader::translate::rewrites;

const OUT: u64 = 0x2_0000_0000;
const GUARD: u32 = 200;

/// s[0:3] = output V#. Enables all 64 lanes, then serves one lane per
/// iteration (v_readfirstlane, compare, clear the served lanes from EXEC)
/// until EXEC is empty or the guard count is reached; lane 0 stores the
/// iteration count.
fn waterfall() -> Vec<u32> {
    let mut c = vec![
        sop1(0x04, EXEC_LO, inline_neg(1)),  // 0: s_mov_b64 exec, -1
        sop1(0x03, 4, inline_u32(0)),        // 1: s_mov_b32 s4, 0
        sop2(0x00, 4, 4, inline_u32(1)),     // 2: loop: s_add_u32 s4, s4, 1
        sopc(0x09, 4, LITERAL),              // 3: s_cmp_ge_u32 s4, GUARD
        GUARD,                               // 4
        sopp(0x05, 4),                       // 5: s_cbranch_scc1 done
        vop1(0x02, 5, vgpr(0)),              // 6: v_readfirstlane_b32 s5, v0
        vopc(0xC2, 5, 0),                    // 7: v_cmp_eq_u32 vcc, s5, v0
        sop2(0x15, EXEC_LO, EXEC_LO, VCC_LO), // 8: s_andn2_b64 exec, exec, vcc
        sopp(0x09, 0xFFF8),                  // 9: s_cbranch_execnz loop
        sop1(0x04, EXEC_LO, inline_u32(1)),  // 10: done: s_mov_b64 exec, 1
        vop1(0x01, 1, 4),                    // 11: v_mov_b32 v1, s4
        vop1(0x01, 2, inline_u32(0)),        // 12: v_mov_b32 v2, 0
    ];
    c.extend(mubuf(0x1C, 1, 2, 0, inline_u32(0), 0, false, true, false)); // buffer_store_dword v1, v2 offen
    c.push(sopp(0x01, 0)); // s_endpgm
    c
}

fn iterations(lanes: u32, rewrites: u32) -> u32 {
    let user = vec![OUT as u32, (OUT >> 32) as u32, 64, 0x3101_6FAC];
    let mut run = ComputeRun::new(waterfall(), 64, lanes, user);
    run.rewrites = rewrites;
    let setup = |m: &mut Machine| m.memory.write_words(OUT, &[0xdead_beef; 4]);
    let m = run.run_with(true, &setup).expect("runs");
    m.memory.read_words(OUT, 1)[0]
}

#[test]
fn waterfall_over_a_partial_wave_ends() {
    // One iteration per present lane (distinct ids).
    assert_eq!(iterations(40, rewrites::ALL), 40);
    assert_eq!(iterations(64, rewrites::ALL), 64);
    // Without the rule the enabled missing lanes never clear their bits.
    assert_eq!(iterations(40, rewrites::ALL & !rewrites::PRESENT_EXEC), GUARD);
}
