// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! v_permlane16_b32 / v_permlanex16_b32 including inactive-lane fi/bc forms.
//! Hardware-measured reference rows, exact compare, wave32 and wave64 runs.
//! (The reference's extra "wave64 on a 32-wide subgroup" run only changes
//! its Vulkan backend's subgroup size and is not applicable here.)

include!("vectors/fmod_common.rs");
include!("vectors/fmod_permlane16.rs");

const NAMES: [&str; 16] = [
    "permlane16_identity",
    "permlane16_reverse",
    "permlane16_mixed",
    "permlane16_zero",
    "permlane16_all_ones",
    "permlanex16_identity",
    "permlanex16_reverse",
    "permlanex16_mixed",
    "permlanex16_broadcast",
    "inactive_permlane16",
    "inactive_permlane16_fi",
    "inactive_permlane16_bc",
    "inactive_permlane16_fi_bc",
    "inactive_permlanex16",
    "inactive_permlanex16_fi",
    "inactive_permlanex16_bc",
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
    w32_permlane16_identity => (32, 0);
    w64_permlane16_identity => (64, 0);
    w32_permlane16_reverse => (32, 1);
    w64_permlane16_reverse => (64, 1);
    w32_permlane16_mixed => (32, 2);
    w64_permlane16_mixed => (64, 2);
    w32_permlane16_zero => (32, 3);
    w64_permlane16_zero => (64, 3);
    w32_permlane16_all_ones => (32, 4);
    w64_permlane16_all_ones => (64, 4);
    w32_permlanex16_identity => (32, 5);
    w64_permlanex16_identity => (64, 5);
    w32_permlanex16_reverse => (32, 6);
    w64_permlanex16_reverse => (64, 6);
    w32_permlanex16_mixed => (32, 7);
    w64_permlanex16_mixed => (64, 7);
    w32_permlanex16_broadcast => (32, 8);
    w64_permlanex16_broadcast => (64, 8);
    w32_inactive_permlane16 => (32, 9);
    w64_inactive_permlane16 => (64, 9);
    w32_inactive_permlane16_fi => (32, 10);
    w64_inactive_permlane16_fi => (64, 10);
    w32_inactive_permlane16_bc => (32, 11);
    w64_inactive_permlane16_bc => (64, 11);
    w32_inactive_permlane16_fi_bc => (32, 12);
    w64_inactive_permlane16_fi_bc => (64, 12);
    w32_inactive_permlanex16 => (32, 13);
    w64_inactive_permlanex16 => (64, 13);
    w32_inactive_permlanex16_fi => (32, 14);
    w64_inactive_permlanex16_fi => (64, 14);
    w32_inactive_permlanex16_bc => (32, 15);
    w64_inactive_permlanex16_bc => (64, 15);
}
