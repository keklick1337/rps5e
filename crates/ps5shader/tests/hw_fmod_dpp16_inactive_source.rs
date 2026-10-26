// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! DPP16 reading inactive source lanes (fetch_inactive, bound_ctrl, masks).
//! Hardware-measured reference rows, exact compare, wave32 and wave64 runs.
//! (The reference's extra "wave64 on a 32-wide subgroup" run only changes
//! its Vulkan backend's subgroup size and is not applicable here.)

include!("vectors/fmod_common.rs");
include!("vectors/fmod_dpp16_inactive_source.rs");

const NAMES: [&str; 15] = [
    "mov_quad_perm",
    "mov_quad_perm_fi",
    "mov_quad_perm_bc",
    "mov_row_shr",
    "mov_row_shr_fi",
    "mov_row_shr_bc",
    "mov_row_shr_bc_fi",
    "mov_row_mirror_masks",
    "add_quad_perm",
    "add_quad_perm_fi",
    "add_quad_perm_bc",
    "add_row_shr",
    "add_row_shr_fi",
    "add_row_shr_bc",
    "add_row_shr_bc_fi",
];

fn check(wave: u32, i: usize) {
    if wave == 32 {
        fmod_check_wave(&WAVE32_CODE, 32, &ROWS, &EXPECTED32, i, NAMES[i]);
    } else {
        fmod_check_wave(&WAVE64_CODE, 64, &ROWS, &EXPECTED64, i, NAMES[i]);
    }
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => ($w:expr, $i:expr);)*) => { $( #[test] $(#[$m])* fn $name() { check($w, $i); } )* };
}

columns! {
    w32_mov_quad_perm => (32, 0);
    w64_mov_quad_perm => (64, 0);
    w32_mov_quad_perm_fi => (32, 1);
    w64_mov_quad_perm_fi => (64, 1);
    w32_mov_quad_perm_bc => (32, 2);
    w64_mov_quad_perm_bc => (64, 2);
    w32_mov_row_shr => (32, 3);
    w64_mov_row_shr => (64, 3);
    w32_mov_row_shr_fi => (32, 4);
    w64_mov_row_shr_fi => (64, 4);
    w32_mov_row_shr_bc => (32, 5);
    w64_mov_row_shr_bc => (64, 5);
    w32_mov_row_shr_bc_fi => (32, 6);
    w64_mov_row_shr_bc_fi => (64, 6);
    w32_mov_row_mirror_masks => (32, 7);
    w64_mov_row_mirror_masks => (64, 7);
    w32_add_quad_perm => (32, 8);
    w64_add_quad_perm => (64, 8);
    w32_add_quad_perm_fi => (32, 9);
    w64_add_quad_perm_fi => (64, 9);
    w32_add_quad_perm_bc => (32, 10);
    w64_add_quad_perm_bc => (64, 10);
    w32_add_row_shr => (32, 11);
    w64_add_row_shr => (64, 11);
    w32_add_row_shr_fi => (32, 12);
    w64_add_row_shr_fi => (64, 12);
    w32_add_row_shr_bc => (32, 13);
    w64_add_row_shr_bc => (64, 13);
    w32_add_row_shr_bc_fi => (32, 14);
    w64_add_row_shr_bc_fi => (64, 14);
}
