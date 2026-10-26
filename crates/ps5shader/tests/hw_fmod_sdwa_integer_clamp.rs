// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! VOP2 SDWA integer operations with clamp.
//! Hardware-measured reference rows, exact compare.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_sdwa_integer_clamp.rs");

const NAMES: [&str; 15] = [
    "add_nc_u32",
    "sub_nc_u32",
    "subrev_nc_u32",
    "add_co_ci_u32",
    "add_co_ci_u32 carry",
    "sub_co_ci_u32",
    "sub_co_ci_u32 carry",
    "subrev_co_ci_u32",
    "subrev_co_ci_u32 carry",
    "mul_i32_i24",
    "mul_u32_u24",
    "mul_hi_i32_i24",
    "min_u32",
    "ashrrev_i32",
    "xnor_b32",
];

fn check(i: usize) {
    fmod_check_rows(&CODE, &ROWS, &EXPECTED, None, i, NAMES[i], &fmod_exact);
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => $i:expr;)*) => { $( #[test] $(#[$m])* fn $name() { check($i); } )* };
}

columns! {
    c00_add_nc_u32 => 0;
    c01_sub_nc_u32 => 1;
    c02_subrev_nc_u32 => 2;
    c03_add_co_ci_u32 => 3;
    c04_add_co_ci_u32_carry => 4;
    c05_sub_co_ci_u32 => 5;
    c06_sub_co_ci_u32_carry => 6;
    c07_subrev_co_ci_u32 => 7;
    c08_subrev_co_ci_u32_carry => 8;
    c09_mul_i32_i24 => 9;
    c10_mul_u32_u24 => 10;
    c11_mul_hi_i32_i24 => 11;
    c12_min_u32 => 12;
    c13_ashrrev_i32 => 13;
    c14_xnor_b32 => 14;
}
