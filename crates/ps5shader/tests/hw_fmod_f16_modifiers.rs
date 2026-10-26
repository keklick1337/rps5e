// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! f16 VOP2/VOP3 clamp and output modifiers against hardware-measured
//! reference rows (low half compared; any NaN matches any NaN).

include!("vectors/fmod_common.rs");
include!("vectors/fmod_f16_modifiers.rs");

const THREADS: usize = 32;
const INPUTS: usize = 4;
const RESULTS: usize = 16;
const NAMES: [&str; 16] = [
    "v_add_f16 clamp", "v_add_f16 mul:2", "v_sub_f16 clamp", "v_subrev_f16 clamp div:2", "v_mul_f16 clamp", "v_min_f16 clamp", "v_max_f16 mul:4", "v_ldexp_f16 clamp",
    "v_ldexp_f16 mul:2", "v_frexp_mant_f16 clamp", "v_cvt_norm_i16_f16 clamp", "v_cvt_norm_u16_f16 mul:2", "v_max3_f16 mul:2", "v_med3_f16 clamp mul:4", "v_min3_f16 div:2", "v_fma_f16 mul:2",
];

fn is_nan16(h: u32) -> bool {
    (h & 0x7fff) > 0x7c00
}

fn check(i: usize) {
    let mut input = vec![0u32; THREADS * INPUTS];
    for (tid, row) in ROWS.iter().enumerate() {
        input[tid * INPUTS..tid * INPUTS + 4].copy_from_slice(row);
    }
    let out = fmod_dispatch(&CODE, 32, THREADS as u32, &input, THREADS * RESULTS, 0).unwrap();
    let mut failures = Vec::new();
    for tid in 0..THREADS {
        let actual = out[tid * RESULTS + i] & 0xffff;
        let expected = EXPECTED[tid][i];
        if is_nan16(expected) && is_nan16(actual) {
            continue;
        }
        if actual != expected {
            failures.push(format!("lane {tid} {:08x?} {}: {actual:#06x}, expected {expected:#06x}", ROWS[tid], NAMES[i]));
        }
    }
    fmod_report(&failures);
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => $i:expr;)*) => { $( #[test] $(#[$m])* fn $name() { check($i); } )* };
}

columns! {
    add_f16_clamp => 0;
    add_f16_mul2 => 1;
    sub_f16_clamp => 2;
    subrev_f16_clamp_div2 => 3;
    mul_f16_clamp => 4;
    min_f16_clamp => 5;
    max_f16_mul4 => 6;
    ldexp_f16_clamp => 7;
    ldexp_f16_mul2 => 8;
    frexp_mant_f16_clamp => 9;
    cvt_norm_i16_f16_clamp => 10;
    cvt_norm_u16_f16_mul2 => 11;
    max3_f16_mul2 => 12;
    med3_f16_clamp_mul4 => 13;
    min3_f16_div2 => 14;
    fma_f16_mul2 => 15;
}
