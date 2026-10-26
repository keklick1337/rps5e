// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! f64 conversions, clamp, omod, literals, inline 1/(2*pi).
//! Hardware-measured reference rows, exact compare.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_f64_conversions.rs");

const NAMES: [&str; 16] = [
    "cvt f64 f32 lo",
    "cvt f64 f32 hi",
    "cvt f64 i32 lo",
    "cvt f64 i32 hi",
    "cvt f64 u32 lo",
    "cvt f64 u32 hi",
    "add clamp lo",
    "add clamp hi",
    "mul abs neg omod lo",
    "mul abs neg omod hi",
    "literal lo",
    "literal hi",
    "ceil neg lo",
    "ceil neg hi",
    "fma inv2pi omod lo",
    "fma inv2pi omod hi",
];

fn check(i: usize) {
    // Measured with FLOAT_MODE 0xF0 and IEEE_MODE.
    fmod_set_float_mode(FMOD_IEEE_F0);
    fmod_check_rows(&CODE, &ROWS, &EXPECTED, None, i, NAMES[i], &fmod_exact);
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => $i:expr;)*) => { $( #[test] $(#[$m])* fn $name() { check($i); } )* };
}

columns! {
    c00_cvt_f64_f32_lo => 0;
    c01_cvt_f64_f32_hi => 1;
    c02_cvt_f64_i32_lo => 2;
    c03_cvt_f64_i32_hi => 3;
    c04_cvt_f64_u32_lo => 4;
    c05_cvt_f64_u32_hi => 5;
    c06_add_clamp_lo => 6;
    c07_add_clamp_hi => 7;
    c08_mul_abs_neg_omod_lo => 8;
    c09_mul_abs_neg_omod_hi => 9;
    c10_literal_lo => 10;
    c11_literal_hi => 11;
    c12_ceil_neg_lo => 12;
    c13_ceil_neg_hi => 13;
    c14_fma_inv2pi_omod_lo => 14;
    c15_fma_inv2pi_omod_hi => 15;
}
