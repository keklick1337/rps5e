// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! f16 result modifiers (clamp / omod on transcendental, conversion, FMA,
//! SDWA and pack forms) against hardware-measured reference rows.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_f16_result_modifiers.rs");

const THREADS: usize = 32;
const INPUTS: usize = 4;
const RESULTS: usize = 16;
const NAMES: [&str; 16] = ["rcp_clamp", "rcp_mul2", "sqrt_clamp_mul2", "sin_mul2", "sin_clamp", "cvt_f16_f32_clamp", "cvt_f16_f32_div2", "cvt_f16_u16_clamp", "fma_clamp", "fma_mul2", "add_sdwa_clamp", "mul_sdwa_mul2", "pkrtz_clamp", "pkrtz_div2", "floor_clamp", "add_sdwa_hi_clamp"];

fn is_nan16(h: u32) -> bool {
    (h & 0x7fff) > 0x7c00
}

fn same_half(a: u32, e: u32) -> bool {
    a == e || (is_nan16(a) && is_nan16(e))
}

fn half_value(bits: u32) -> f64 {
    let m = if bits & 0x7c00 == 0 {
        (bits & 0x3ff) as f64 * 2f64.powi(-24)
    } else {
        ((bits & 0x3ff) | 0x400) as f64 * 2f64.powi(((bits >> 10) & 0x1f) as i32 - 25)
    };
    if bits & 0x8000 != 0 {
        -m
    } else {
        m
    }
}

fn near_half(a: u32, e: u32, clamped: bool) -> bool {
    if clamped && a & 0x8000 != 0 {
        return false;
    }
    if !clamped && e & 0x7fff == 0 {
        return a == e;
    }
    if is_nan16(a) || is_nan16(e) || a & 0x7fff == 0x7c00 || e & 0x7fff == 0x7c00 {
        return same_half(a, e);
    }
    (half_value(a) - half_value(e)).abs() <= 2f64.powi(-10)
}

fn check(i: usize) {
    let mut input = vec![0u32; THREADS * INPUTS];
    for (tid, row) in ROWS.iter().enumerate() {
        input[tid * INPUTS..tid * INPUTS + 4].copy_from_slice(row);
    }
    let out = fmod_dispatch(&CODE, 32, THREADS as u32, &input, THREADS * RESULTS, 0).unwrap();
    let mut failures = Vec::new();
    for tid in 0..THREADS {
        let a = out[tid * RESULTS + i];
        let e = EXPECTED[tid][i];
        let ok = if i == 3 || i == 4 {
            a >> 16 == e >> 16 && near_half(a & 0xffff, e & 0xffff, i == 4)
        } else {
            same_half(a & 0xffff, e & 0xffff) && same_half(a >> 16, e >> 16)
        };
        if !ok {
            failures.push(format!("lane {tid} {:08x?} {}: {a:#010x}, expected {e:#010x}", ROWS[tid], NAMES[i]));
        }
    }
    fmod_report(&failures);
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => $i:expr;)*) => { $( #[test] $(#[$m])* fn $name() { check($i); } )* };
}

columns! {
    rcp_clamp => 0;
    rcp_mul2 => 1;
    sqrt_clamp_mul2 => 2;
    sin_mul2 => 3;
    sin_clamp => 4;
    cvt_f16_f32_clamp => 5;
    cvt_f16_f32_div2 => 6;
    cvt_f16_u16_clamp => 7;
    fma_clamp => 8;
    fma_mul2 => 9;
    add_sdwa_clamp => 10;
    mul_sdwa_mul2 => 11;
    pkrtz_clamp => 12;
    pkrtz_div2 => 13;
    floor_clamp => 14;
    add_sdwa_hi_clamp => 15;
}
