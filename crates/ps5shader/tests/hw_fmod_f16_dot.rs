// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! v_dot2_f32_f16 / v_dot2c_f32_f16 (VOP3P modifiers, inline/literal,
//! DPP) against hardware-measured reference rows.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_f16_dot.rs");

const THREADS: usize = 32;
const INPUTS: usize = 4;
const RESULTS: usize = 16;
const NAMES: [&str; 10] = ["v_dot2c", "v_dot2", "v_dot2 neg_lo src0/src2 neg_hi src1", "v_dot2 neg_lo src1 neg_hi src0", "v_dot2 op_sel src0", "v_dot2 clamp", "v_dot2c inline 1.0", "v_dot2c literal", "v_dot2c dpp quad_perm", "v_dot2 op_sel src1"];

fn run() -> Vec<u32> {
    let mut input = vec![0u32; THREADS * INPUTS];
    for (tid, row) in ROWS.iter().enumerate() {
        input[tid * INPUTS..tid * INPUTS + 4].copy_from_slice(row);
    }
    fmod_dispatch(&CODE, 32, THREADS as u32, &input, THREADS * RESULTS, 0).unwrap()
}

fn check(columns: &[usize]) {
    let out = run();
    let mut failures = Vec::new();
    for tid in 0..THREADS {
        for &i in columns {
            let (a, e) = (out[tid * RESULTS + i], EXPECTED[tid][i]);
            if a != e {
                failures.push(format!("lane {tid} {:?} {}: {a:#010x}, expected {e:#010x}", ROWS[tid], NAMES[i]));
            }
        }
    }
    fmod_report(&failures);
}

// The reference also checks that its own compiler refuses accumulator
// op_sel / neg_hi / cleared op_sel_hi; that is a limitation of that
// compiler, not hardware behaviour, so it is not ported.

/// Accumulator modifiers of v_dot2_f32_f16: op_sel and op_sel_hi on src2
/// are ignored (the full 32-bit addend is read), neg_hi on src2 takes the
/// absolute value and neg_lo negates after it.
#[test]
fn dot2_accumulator_modifiers() {
    const ACC_NAMES: [&str; 5] = ["op_sel src2", "op_sel_hi src2 cleared", "neg_hi src2", "neg_lo src2", "neg_lo and neg_hi src2"];
    let mut input = vec![0u32; THREADS * INPUTS];
    for (tid, row) in ROWS.iter().enumerate() {
        input[tid * INPUTS..tid * INPUTS + 4].copy_from_slice(row);
    }
    let out = fmod_dispatch(&ACC_CODE, 32, THREADS as u32, &input, THREADS * RESULTS, 0).unwrap();
    let mut failures = Vec::new();
    for tid in 0..THREADS {
        for (i, name) in ACC_NAMES.iter().enumerate() {
            let (a, e) = (out[tid * RESULTS + i], ACC_EXPECTED[tid][i]);
            if a != e {
                failures.push(format!("lane {tid} {:?} v_dot2 {name}: {a:#010x}, expected {e:#010x}", ROWS[tid]));
            }
        }
    }
    fmod_report(&failures);
}

#[test]
fn dot2_all_forms() {
    check(&(0..10).collect::<Vec<_>>());
}
