// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! v_div_scale/fmas/fixup_f64 against hardware-measured reference rows
//! (exact compare; word 14 is the v_div_scale VCC result).

include!("vectors/fmod_common.rs");
include!("vectors/fmod_f64_division.rs");

const NAMES: [&str; 7] = ["div_scale denominator", "div_scale numerator", "div_fmas a*b+a", "div_fmas a*b+b", "div_fixup a", "div_fixup b", "div_fixup nan"];

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
    div_scale_denominator_lo => 0;
    div_scale_denominator_hi => 1;
    div_scale_numerator_lo => 2;
    div_scale_numerator_hi => 3;
    div_fmas_ab_plus_a_lo => 4;
    div_fmas_ab_plus_a_hi => 5;
    div_fmas_ab_plus_b_lo => 6;
    div_fmas_ab_plus_b_hi => 7;
    div_fixup_a_lo => 8;
    div_fixup_a_hi => 9;
    div_fixup_b_lo => 10;
    div_fixup_b_hi => 11;
    div_fixup_nan_lo => 12;
    div_fixup_nan_hi => 13;
    div_scale_vcc => 14;
}
