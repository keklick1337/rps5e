// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! v_ldexp_f16 SDWA with source/destination selectors, clamp and omod.
//! Hardware-measured reference rows, exact compare with a per-lane checked mask.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_sdwa_ldexp_f16.rs");

// The reference also checks that its own compiler refuses some SDWA forms
// (clamp with byte selectors, source modifiers); that is a limitation of
// that compiler, not hardware behaviour, so it is not ported.

const NAMES: [&str; 16] = [
    "BYTE_0/DWORD -> DWORD",
    "BYTE_1/BYTE_0 -> WORD_1",
    "BYTE_2/BYTE_1 -> WORD_0",
    "BYTE_3/BYTE_2 -> BYTE_0",
    "WORD_0/BYTE_3 -> BYTE_1",
    "WORD_1/WORD_0 -> BYTE_2",
    "DWORD/WORD_1 -> BYTE_3",
    "WORD_1/WORD_1 -> DWORD",
    "DWORD/WORD_0 -> WORD_1",
    "WORD_0/DWORD -> WORD_1 clamp",
    "WORD_1/WORD_1 -> WORD_0 clamp",
    "DWORD/WORD_0 -> DWORD clamp",
    "DWORD/DWORD -> WORD_0 clamp",
    "WORD_0/DWORD -> DWORD mul:2",
    "WORD_1/WORD_0 -> WORD_1 clamp div:2",
    "BYTE_3/WORD_1 -> WORD_0",
];

fn check(i: usize) {
    // Measured with FLOAT_MODE 0xF0 and IEEE_MODE.
    fmod_set_float_mode(FMOD_IEEE_F0);
    fmod_check_rows(&CODE, &ROWS, &EXPECTED, Some(&CHECKED), i, NAMES[i], &fmod_exact);
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => $i:expr;)*) => { $( #[test] $(#[$m])* fn $name() { check($i); } )* };
}

columns! {
    c00_byte_0_dword_dword => 0;
    c01_byte_1_byte_0_word_1 => 1;
    c02_byte_2_byte_1_word_0 => 2;
    c03_byte_3_byte_2_byte_0 => 3;
    c04_word_0_byte_3_byte_1 => 4;
    c05_word_1_word_0_byte_2 => 5;
    c06_dword_word_1_byte_3 => 6;
    c07_word_1_word_1_dword => 7;
    c08_dword_word_0_word_1 => 8;
    c09_word_0_dword_word_1_clamp => 9;
    c10_word_1_word_1_word_0_clamp => 10;
    c11_dword_word_0_dword_clamp => 11;
    c12_dword_dword_word_0_clamp => 12;
    c13_word_0_dword_dword_mul_2 => 13;
    c14_word_1_word_0_word_1_clamp_div_2 => 14;
    c15_byte_3_word_1_word_0 => 15;
}
