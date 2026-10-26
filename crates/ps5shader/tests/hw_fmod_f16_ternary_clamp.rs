// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! f16 max3/min3/med3/div_fixup with clamp, op_sel and source modifiers
//! against hardware-measured reference rows (exact, per-lane checked mask).

include!("vectors/fmod_common.rs");
include!("vectors/fmod_f16_ternary_clamp.rs");

const NAMES: [&str; 9] = [
    "v_max3_f16 v10, v4, v5, v6 clamp",
    "v_min3_f16 v11, v4, v5, v6 clamp",
    "v_med3_f16 v12, v4, v5, v6 clamp",
    "v_div_fixup_f16 v13, v4, v5, v6 clamp",
    "v_max3_f16 v14, -v5, |v6|, v4 op_sel:[1,1,1,1] clamp",
    "v_min3_f16 v15, v6, -v4, v5 op_sel:[1,0,1,0] clamp",
    "v_med3_f16 v16, v5, v6, -|v4| op_sel:[0,1,1,1] clamp",
    "v_div_fixup_f16 v17, v6, v4, v5 op_sel:[1,1,0,1] clamp",
    "v_div_fixup_f16 v18, -v5, |v6|, -v4 op_sel:[0,0,1,0] clamp",
];

fn check(i: usize) {
    fmod_check_rows(&CODE, &ROWS, &EXPECTED, Some(&CHECKED), i, NAMES[i], &fmod_exact);
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => $i:expr;)*) => { $( #[test] $(#[$m])* fn $name() { check($i); } )* };
}

columns! {
    max3_clamp => 0;
    min3_clamp => 1;
    med3_clamp => 2;
    div_fixup_clamp => 3;
    max3_op_sel_clamp => 4;
    min3_op_sel_clamp => 5;
    med3_op_sel_clamp => 6;
    div_fixup_op_sel_clamp => 7;
    div_fixup_op_sel_mods_clamp => 8;
}
