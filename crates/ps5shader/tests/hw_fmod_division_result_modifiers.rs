// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! v_div_fmas/fixup f32 and f64 with clamp and omod.
//! Hardware-measured reference rows, exact compare with a per-lane checked mask.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_division_result_modifiers.rs");

const NAMES: [&str; 16] = [
    "v_div_fmas_f32 v10, v4, v5, v6 clamp",
    "v_div_fixup_f32 v11, v4, v5, v6 clamp",
    "v_div_fixup_f32 v12, -v4, |v5|, v6 clamp",
    "v_div_fmas_f32 v13, v5, -v4, v6 clamp",
    "v_div_fmas_f64 v[14:15], v[4:5], v[6:7], v[4:5] clamp lo",
    "v_div_fmas_f64 v[14:15], v[4:5], v[6:7], v[4:5] clamp hi",
    "v_div_fmas_f64 v[16:17], v[6:7], v[6:7], v[4:5] mul:2 lo",
    "v_div_fmas_f64 v[16:17], v[6:7], v[6:7], v[4:5] mul:2 hi",
    "v_div_fixup_f64 v[18:19], v[4:5], v[6:7], v[4:5] clamp lo",
    "v_div_fixup_f64 v[18:19], v[4:5], v[6:7], v[4:5] clamp hi",
    "v_div_fixup_f64 v[20:21], v[6:7], v[4:5], v[6:7] div:2 lo",
    "v_div_fixup_f64 v[20:21], v[6:7], v[4:5], v[6:7] div:2 hi",
    "v_div_fixup_f64 v[22:23], v[6:7], v[4:5], v[6:7] clamp mul:4 lo",
    "v_div_fixup_f64 v[22:23], v[6:7], v[4:5], v[6:7] clamp mul:4 hi",
    "v_div_fixup_f64 v[24:25], -v[4:5], v[6:7], v[4:5] clamp lo",
    "v_div_fixup_f64 v[24:25], -v[4:5], v[6:7], v[4:5] clamp hi",
];

fn check(i: usize) {
    fmod_check_rows(&CODE, &ROWS, &EXPECTED, Some(&CHECKED), i, NAMES[i], &fmod_exact);
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => $i:expr;)*) => { $( #[test] $(#[$m])* fn $name() { check($i); } )* };
}

columns! {
    c00_v_div_fmas_f32_v10_v4_v5_v6_clamp => 0;
    c01_v_div_fixup_f32_v11_v4_v5_v6_clamp => 1;
    c02_v_div_fixup_f32_v12_v4_v5_v6_clamp => 2;
    c03_v_div_fmas_f32_v13_v5_v4_v6_clamp => 3;
    c04_v_div_fmas_f64_v_14_15_v_4_5_v_6_7_v_4_5_clamp_lo => 4;
    c05_v_div_fmas_f64_v_14_15_v_4_5_v_6_7_v_4_5_clamp_hi => 5;
    c06_v_div_fmas_f64_v_16_17_v_6_7_v_6_7_v_4_5_mul_2_lo => 6;
    c07_v_div_fmas_f64_v_16_17_v_6_7_v_6_7_v_4_5_mul_2_hi => 7;
    c08_v_div_fixup_f64_v_18_19_v_4_5_v_6_7_v_4_5_clamp_lo => 8;
    c09_v_div_fixup_f64_v_18_19_v_4_5_v_6_7_v_4_5_clamp_hi => 9;
    c10_v_div_fixup_f64_v_20_21_v_6_7_v_4_5_v_6_7_div_2_lo => 10;
    c11_v_div_fixup_f64_v_20_21_v_6_7_v_4_5_v_6_7_div_2_hi => 11;
    c12_v_div_fixup_f64_v_22_23_v_6_7_v_4_5_v_6_7_clamp_mul_4_lo => 12;
    c13_v_div_fixup_f64_v_22_23_v_6_7_v_4_5_v_6_7_clamp_mul_4_hi => 13;
    c14_v_div_fixup_f64_v_24_25_v_4_5_v_6_7_v_4_5_clamp_lo => 14;
    c15_v_div_fixup_f64_v_24_25_v_4_5_v_6_7_v_4_5_clamp_hi => 15;
}
