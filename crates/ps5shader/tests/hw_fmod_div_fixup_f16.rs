// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! v_div_fixup_f16 forms (op_sel halves, modifiers, literals).
//! Hardware-measured reference rows, exact compare.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_div_fixup_f16.rs");

const NAMES: [&str; 8] = [
    "lo",
    "hi sources",
    "hi dst",
    "neg abs",
    "mixed halves",
    "neg denominator",
    "literal quotient",
    "literal denominator",
];

fn check(i: usize) {
    // Measured with FLOAT_MODE 0xF0 and IEEE_MODE.
    fmod_set_float_mode(FMOD_IEEE_F0);
    fmod_check_rows(&CODE, &ROWS, &EXPECTED, None, i, NAMES[i], &fmod_exact);
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => $i:expr;)*) => { $( #[test] $(#[$m])* fn $name() { check($i); } )* };
}

columns! {
    c00_lo => 0;
    c01_hi_sources => 1;
    c02_hi_dst => 2;
    c03_neg_abs => 3;
    c04_mixed_halves => 4;
    c05_neg_denominator => 5;
    c06_literal_quotient => 6;
    c07_literal_denominator => 7;
}
