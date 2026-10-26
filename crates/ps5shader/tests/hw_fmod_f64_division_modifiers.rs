// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! v_div_scale/fmas/fixup_f64 with source modifiers and VCC forms against
//! hardware-measured reference rows (exact compare; word 14 is the
//! v_div_scale VCC result).

include!("vectors/fmod_common.rs");
include!("vectors/fmod_f64_division_modifiers.rs");

const NAMES: [&str; 7] = ["div_scale -a", "div_scale -b", "div_fmas modifiers", "div_fmas b*b+a", "div_fmas vcc 0", "div_fixup modifiers", "div_fmas vcc per lane"];

fn check(i: usize) {
    // Measured with FLOAT_MODE 0xF0 and IEEE_MODE.
    fmod_set_float_mode(FMOD_IEEE_F0);
    let name = if i == 14 { "div_scale vcc".to_string() } else { format!("{} {}", NAMES[i / 2], if i % 2 == 0 { "lo" } else { "hi" }) };
    fmod_check_rows(&CODE, &ROWS, &EXPECTED, None, i, &name, &fmod_exact);
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => $i:expr;)*) => { $( #[test] $(#[$m])* fn $name() { check($i); } )* };
}

columns! {
    div_scale_neg_a_lo => 0;
    div_scale_neg_a_hi => 1;
    div_scale_neg_b_lo => 2;
    div_scale_neg_b_hi => 3;
    div_fmas_modifiers_lo => 4;
    div_fmas_modifiers_hi => 5;
    div_fmas_bb_plus_a_lo => 6;
    div_fmas_bb_plus_a_hi => 7;
    div_fmas_vcc0_lo => 8;
    div_fmas_vcc0_hi => 9;
    div_fixup_modifiers_lo => 10;
    div_fixup_modifiers_hi => 11;
    div_fmas_vcc_per_lane_lo => 12;
    div_fmas_vcc_per_lane_hi => 13;
    div_scale_vcc => 14;
}
