// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! A loop whose selection arms leave the iteration (to the latch, past
//! the join an enclosing selection shares): the structured program must
//! run every path the branches take. The program is the reference's
//! control-flow case; its result is compared with a direct walk of the
//! branches.
//!
//! ```text
//!   v_mov_b32 v1, 0
//!   s_mov_b32 s8, 0
//! loop:
//!   s_cmp_eq_u32 s8, s2
//!   s_cbranch_scc1 join
//!   v_cmp_gt_u32 vcc, 8, v0
//!   s_cbranch_vccz arm
//!   v_add_nc_u32 v1, 1, v1
//!   v_cmp_gt_u32 vcc, 16, v0
//!   s_cbranch_vccz away
//! arm:
//!   v_add_nc_u32 v1, 2, v1
//!   v_cmp_gt_u32 vcc, 4, v0
//!   s_cbranch_vccz aside
//! join:
//!   v_add_nc_u32 v1, 3, v1
//!   s_cmp_lt_u32 s8, 4
//!   s_cbranch_scc0 done
//!   s_branch latch
//! away:
//!   v_add_nc_u32 v1, 4, v1
//!   s_branch latch
//! aside:
//!   v_add_nc_u32 v1, 5, v1
//! latch:
//!   s_add_u32 s8, s8, 1
//!   s_branch loop
//! done:
//!   buffer_store_dword v1, off, s[0:3], 0
//!   s_endpgm
//! ```

use ps5shader::interp::Machine;
use ps5shader::ssa::SystemValue;
use ps5shader::testing::{raw_buffer_descriptor, ComputeRun};

const CODE: [u32; 24] = [
    0x7e020280, 0xbe880380, 0xbf060208, 0xbf850008, 0x7d880088, 0xbf860003, 0x4a020281, 0x7d880090, 0xbf860007, 0x4a020282, 0x7d880084, 0xbf860006, 0x4a020283, 0xbf0a8408,
    0xbf840006, 0xbf820003, 0x4a020284, 0xbf820001, 0x4a020285, 0x80088108, 0xbf82ffed, 0xe0700000, 0x80000100, 0xbf810000,
];
const OUT: u64 = 0x2_0000_0000;

/// The program's branches walked directly (one lane with thread id `v0`;
/// `s2` is the V#'s NUM_RECORDS). None when it does not end.
fn walk(s2: u32, v0: u32) -> Option<u32> {
    let (mut v1, mut s8) = (0u32, 0u32);
    for _ in 0..64 {
        // loop:
        let mut at_join = s8 == s2;
        if !at_join {
            let mut at_arm = 8 <= v0;
            if !at_arm {
                v1 += 1;
                if 16 <= v0 {
                    v1 += 4; // away
                    s8 += 1;
                    continue;
                }
                at_arm = true;
            }
            if at_arm {
                v1 += 2;
                if 4 <= v0 {
                    v1 += 5; // aside
                    s8 += 1;
                    continue;
                }
                at_join = true;
            }
        }
        if at_join {
            v1 += 3;
            if s8 >= 4 {
                return Some(v1);
            }
            s8 += 1;
        }
    }
    None
}

#[test]
fn loop_selection_arms_leaving_the_iteration() {
    let mut cases = 0;
    for s2 in [4u32, 5, 8] {
        for v0 in [0u32, 1, 3, 5, 9, 20] {
            let Some(expected) = walk(s2, v0) else { continue };
            let user = raw_buffer_descriptor(OUT, s2).to_vec();
            let run = ComputeRun::new(CODE.to_vec(), 32, 1, user);
            let setup = |m: &mut Machine| {
                m.memory.write_words(OUT, &[0xDEAD_BEEF]);
                m.system.insert(SystemValue::LocalInvocationId(0), vec![v0]);
            };
            let m = run.run_checked(&setup, &[(OUT, 1)]).unwrap();
            assert_eq!(m.memory.read_words(OUT, 1)[0], expected, "s2 {s2} v0 {v0}");
            cases += 1;
        }
    }
    assert!(cases >= 6, "only {cases} terminating cases");
}
