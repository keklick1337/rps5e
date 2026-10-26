// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! DPP16 with pk_fmac, movrel forms, masks and EXEC.
//! Hardware-measured reference rows, exact compare.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_dpp16_operands.rs");

const NAMES: [&str; 16] = [
    "quad_perm_pk_fmac_f16",
    "quad_perm_movrels",
    "quad_perm_movrelsd",
    "quad_perm_movreld",
    "quad_perm_movrelsd_2",
    "row_shr_pk_fmac_f16",
    "row_shr_movrels",
    "row_shr_movrelsd",
    "row_shr_movreld",
    "row_shr_movrelsd_2",
    "masks_exec_pk_fmac_f16",
    "masks_exec_movrels",
    "masks_exec_movrelsd",
    "masks_exec_movreld",
    "masks_exec_movrelsd_2",
    "relative_bases",
];

fn check(i: usize) {
    fmod_check_rows(&CODE, &ROWS, &EXPECTED, None, i, NAMES[i], &fmod_exact);
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => $i:expr;)*) => { $( #[test] $(#[$m])* fn $name() { check($i); } )* };
}

columns! {
    c00_quad_perm_pk_fmac_f16 => 0;
    c01_quad_perm_movrels => 1;
    c02_quad_perm_movrelsd => 2;
    c03_quad_perm_movreld => 3;
    c04_quad_perm_movrelsd_2 => 4;
    c05_row_shr_pk_fmac_f16 => 5;
    c06_row_shr_movrels => 6;
    c07_row_shr_movrelsd => 7;
    c08_row_shr_movreld => 8;
    c09_row_shr_movrelsd_2 => 9;
    c10_masks_exec_pk_fmac_f16 => 10;
    c11_masks_exec_movrels => 11;
    c12_masks_exec_movrelsd => 12;
    c13_masks_exec_movreld => 13;
    c14_masks_exec_movrelsd_2 => 14;
    c15_relative_bases => 15;
}
