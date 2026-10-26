// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! f64 trunc/ceil/floor/rndne/fract/frexp and f64 -> f32/i32/u32 conversions.
//! Hardware-measured reference rows, exact compare.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_f64_rounding.rs");

const NAMES: [&str; 16] = [
    "trunc lo",
    "trunc hi",
    "ceil lo",
    "ceil hi",
    "floor lo",
    "floor hi",
    "rndne lo",
    "rndne hi",
    "fract lo",
    "fract hi",
    "frexp mant lo",
    "frexp mant hi",
    "cvt f32",
    "frexp exp",
    "cvt i32",
    "cvt u32",
];

fn check(i: usize) {
    fmod_check_rows(&CODE, &ROWS, &EXPECTED, None, i, NAMES[i], &fmod_exact);
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => $i:expr;)*) => { $( #[test] $(#[$m])* fn $name() { check($i); } )* };
}

columns! {
    c00_trunc_lo => 0;
    c01_trunc_hi => 1;
    c02_ceil_lo => 2;
    c03_ceil_hi => 3;
    c04_floor_lo => 4;
    c05_floor_hi => 5;
    c06_rndne_lo => 6;
    c07_rndne_hi => 7;
    c08_fract_lo => 8;
    c09_fract_hi => 9;
    c10_frexp_mant_lo => 10;
    c11_frexp_mant_hi => 11;
    c12_cvt_f32 => 12;
    c13_frexp_exp => 13;
    c14_cvt_i32 => 14;
    c15_cvt_u32 => 15;
}
