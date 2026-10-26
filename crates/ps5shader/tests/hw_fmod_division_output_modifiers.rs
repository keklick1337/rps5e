// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! v_div_fmas/fixup/scale f32 and v_div_scale_f64 with clamp and omod.
//! Hardware-measured reference rows, exact compare with a per-lane checked mask.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_division_output_modifiers.rs");

const NAMES: [&str; 16] = [
    "v_div_fmas_f32 v10, v4, v5, v6 mul:2",
    "v_div_fmas_f32 v11, v5, -v4, v6 div:2",
    "v_div_fixup_f32 v12, v4, v5, v6 mul:4",
    "v_div_fixup_f32 v13, -v4, |v5|, v6 div:2",
    "v_div_scale_f32 v14, s12, v4, v5, v6 clamp",
    "v_div_scale_f32 v15, s12, v4, v5, v6 mul:2",
    "v_div_scale_f32 v16, s12, v6, v5, v6 div:2",
    "v_div_scale_f32 v17, s12, v6, v5, v6 clamp mul:4",
    "v_div_scale_f64 v[18:19], s12, v[4:5], v[6:7], v[4:5] clamp lo",
    "v_div_scale_f64 v[18:19], s12, v[4:5], v[6:7], v[4:5] clamp hi",
    "v_div_scale_f64 v[20:21], s12, v[6:7], v[6:7], v[4:5] mul:2 lo",
    "v_div_scale_f64 v[20:21], s12, v[6:7], v[6:7], v[4:5] mul:2 hi",
    "v_div_scale_f64 v[22:23], s12, v[4:5], v[6:7], v[4:5] div:2 lo",
    "v_div_scale_f64 v[22:23], s12, v[4:5], v[6:7], v[4:5] div:2 hi",
    "v_div_scale_f64 v[24:25], s12, v[6:7], v[6:7], v[4:5] clamp mul:4 lo",
    "v_div_scale_f64 v[24:25], s12, v[6:7], v[6:7], v[4:5] clamp mul:4 hi",
];

fn check(i: usize) {
    fmod_check_rows(&CODE, &ROWS, &EXPECTED, Some(&CHECKED), i, NAMES[i], &fmod_exact);
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => $i:expr;)*) => { $( #[test] $(#[$m])* fn $name() { check($i); } )* };
}

columns! {
    c00_v_div_fmas_f32_v10_v4_v5_v6_mul_2 => 0;
    c01_v_div_fmas_f32_v11_v5_v4_v6_div_2 => 1;
    c02_v_div_fixup_f32_v12_v4_v5_v6_mul_4 => 2;
    c03_v_div_fixup_f32_v13_v4_v5_v6_div_2 => 3;
    c04_v_div_scale_f32_v14_s12_v4_v5_v6_clamp => 4;
    c05_v_div_scale_f32_v15_s12_v4_v5_v6_mul_2 => 5;
    c06_v_div_scale_f32_v16_s12_v6_v5_v6_div_2 => 6;
    c07_v_div_scale_f32_v17_s12_v6_v5_v6_clamp_mul_4 => 7;
    c08_v_div_scale_f64_v_18_19_s12_v_4_5_v_6_7_v_4_5_clamp_lo => 8;
    c09_v_div_scale_f64_v_18_19_s12_v_4_5_v_6_7_v_4_5_clamp_hi => 9;
    c10_v_div_scale_f64_v_20_21_s12_v_6_7_v_6_7_v_4_5_mul_2_lo => 10;
    c11_v_div_scale_f64_v_20_21_s12_v_6_7_v_6_7_v_4_5_mul_2_hi => 11;
    c12_v_div_scale_f64_v_22_23_s12_v_4_5_v_6_7_v_4_5_div_2_lo => 12;
    c13_v_div_scale_f64_v_22_23_s12_v_4_5_v_6_7_v_4_5_div_2_hi => 13;
    c14_v_div_scale_f64_v_24_25_s12_v_6_7_v_6_7_v_4_5_clamp_mul => 14;
    c15_v_div_scale_f64_v_24_25_s12_v_6_7_v_6_7_v_4_5_clamp_mul => 15;
}
