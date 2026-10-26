// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! DPP8 lane selects (fi, inactive sources, VOP2/VOPC-carry forms).
//! Hardware-measured reference rows, exact compare, wave32 and wave64 runs.
//! (The reference's extra "wave64 on a 32-wide subgroup" run only changes
//! its Vulkan backend's subgroup size and is not applicable here.)

include!("vectors/fmod_common.rs");
include!("vectors/fmod_dpp8.rs");

const NAMES: [&str; 12] = [
    "mov",
    "mov_fi",
    "add_nc_u32",
    "add_f32",
    "fmac_f32",
    "cndmask",
    "cvt_f32_i32",
    "add_co_ci_u32",
    "mov_inactive",
    "mov_inactive_fi",
    "sub_nc_u32_inactive",
    "mov_uniform_inactive",
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
    w32_mov => (32, 0);
    w64_mov => (64, 0);
    w32_mov_fi => (32, 1);
    w64_mov_fi => (64, 1);
    w32_add_nc_u32 => (32, 2);
    w64_add_nc_u32 => (64, 2);
    w32_add_f32 => (32, 3);
    w64_add_f32 => (64, 3);
    w32_fmac_f32 => (32, 4);
    w64_fmac_f32 => (64, 4);
    w32_cndmask => (32, 5);
    w64_cndmask => (64, 5);
    w32_cvt_f32_i32 => (32, 6);
    w64_cvt_f32_i32 => (64, 6);
    w32_add_co_ci_u32 => (32, 7);
    w64_add_co_ci_u32 => (64, 7);
    w32_mov_inactive => (32, 8);
    w64_mov_inactive => (64, 8);
    w32_mov_inactive_fi => (32, 9);
    w64_mov_inactive_fi => (64, 9);
    w32_sub_nc_u32_inactive => (32, 10);
    w64_sub_nc_u32_inactive => (64, 10);
    w32_mov_uniform_inactive => (32, 11);
    w64_mov_uniform_inactive => (64, 11);
}
