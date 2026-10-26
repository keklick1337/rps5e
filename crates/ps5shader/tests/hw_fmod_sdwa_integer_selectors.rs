// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! VOP2 SDWA integer operations with byte/word selectors and sign extension.
//! Hardware-measured reference rows, exact compare.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_sdwa_integer_selectors.rs");

const NAMES: [&str; 16] = [
    "min_i32",
    "max_i32",
    "sub_co_ci_u32",
    "sub_co_ci_u32 carry",
    "subrev_co_ci_u32",
    "subrev_co_ci_u32 carry",
    "add_co_ci_u32",
    "add_co_ci_u32 carry",
    "mul_i32_i24",
    "max_u32",
    "lshrrev_b32",
    "ashrrev_i32",
    "lshlrev_b32",
    "and_b32",
    "xnor_b32",
    "subrev_nc_u32",
];

fn check(i: usize) {
    fmod_check_rows(&CODE, &ROWS, &EXPECTED, None, i, NAMES[i], &fmod_exact);
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => $i:expr;)*) => { $( #[test] $(#[$m])* fn $name() { check($i); } )* };
}

columns! {
    c00_min_i32 => 0;
    c01_max_i32 => 1;
    c02_sub_co_ci_u32 => 2;
    c03_sub_co_ci_u32_carry => 3;
    c04_subrev_co_ci_u32 => 4;
    c05_subrev_co_ci_u32_carry => 5;
    c06_add_co_ci_u32 => 6;
    c07_add_co_ci_u32_carry => 7;
    c08_mul_i32_i24 => 8;
    c09_max_u32 => 9;
    c10_lshrrev_b32 => 10;
    c11_ashrrev_i32 => 11;
    c12_lshlrev_b32 => 12;
    c13_and_b32 => 13;
    c14_xnor_b32 => 14;
    c15_subrev_nc_u32 => 15;
}
