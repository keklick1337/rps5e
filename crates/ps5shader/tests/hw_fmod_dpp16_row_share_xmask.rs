// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! DPP16 row_share / row_xmask with fi, bound_ctrl and row/bank masks.
//! Hardware-measured reference rows, exact compare, wave32 and wave64 runs.
//! (The reference's extra "wave64 on a 32-wide subgroup" run only changes
//! its Vulkan backend's subgroup size and is not applicable here.)

include!("vectors/fmod_common.rs");
include!("vectors/fmod_dpp16_row_share_xmask.rs");

const NAMES: [&str; 16] = [
    "mov_row_share0",
    "mov_row_share5_fi",
    "mov_row_share12_bc",
    "mov_row_share8_bc_fi",
    "mov_row_share15_masks",
    "mov_row_xmask1",
    "mov_row_xmask6_fi",
    "mov_row_xmask15_bc",
    "mov_row_xmask10_masks",
    "add_row_share3",
    "add_row_share12_bc",
    "add_row_share4_fi_masks",
    "add_row_xmask4",
    "sub_row_xmask13_bc_fi",
    "max_f32_row_share4_bc",
    "max_f32_row_xmask9_fi",
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
    w32_mov_row_share0 => (32, 0);
    w64_mov_row_share0 => (64, 0);
    w32_mov_row_share5_fi => (32, 1);
    w64_mov_row_share5_fi => (64, 1);
    w32_mov_row_share12_bc => (32, 2);
    w64_mov_row_share12_bc => (64, 2);
    w32_mov_row_share8_bc_fi => (32, 3);
    w64_mov_row_share8_bc_fi => (64, 3);
    w32_mov_row_share15_masks => (32, 4);
    w64_mov_row_share15_masks => (64, 4);
    w32_mov_row_xmask1 => (32, 5);
    w64_mov_row_xmask1 => (64, 5);
    w32_mov_row_xmask6_fi => (32, 6);
    w64_mov_row_xmask6_fi => (64, 6);
    w32_mov_row_xmask15_bc => (32, 7);
    w64_mov_row_xmask15_bc => (64, 7);
    w32_mov_row_xmask10_masks => (32, 8);
    w64_mov_row_xmask10_masks => (64, 8);
    w32_add_row_share3 => (32, 9);
    w64_add_row_share3 => (64, 9);
    w32_add_row_share12_bc => (32, 10);
    w64_add_row_share12_bc => (64, 10);
    w32_add_row_share4_fi_masks => (32, 11);
    w64_add_row_share4_fi_masks => (64, 11);
    w32_add_row_xmask4 => (32, 12);
    w64_add_row_xmask4 => (64, 12);
    w32_sub_row_xmask13_bc_fi => (32, 13);
    w64_sub_row_xmask13_bc_fi => (64, 13);
    w32_max_f32_row_share4_bc => (32, 14);
    w64_max_f32_row_share4_bc => (64, 14);
    w32_max_f32_row_xmask9_fi => (32, 15);
    w64_max_f32_row_xmask9_fi => (64, 15);
}
