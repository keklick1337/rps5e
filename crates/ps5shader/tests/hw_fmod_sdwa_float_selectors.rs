// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! VOP2 SDWA float operations with byte/word source and destination selectors.
//! Hardware-measured reference rows, exact compare with a per-lane checked mask.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_sdwa_float_selectors.rs");

// The reference also checks that its own compiler refuses VOP2 SDWA clamp
// with byte selectors; that is a limitation of that compiler, not hardware
// behaviour, so it is not ported.

const NAMES: [&str; 16] = [
    "add_f32 BYTE_0/DWORD -> DWORD",
    "sub_f32 DWORD/WORD_1 -> WORD_1",
    "subrev_f32 WORD_0/DWORD -> BYTE_2",
    "mul_f32 DWORD/BYTE_3 -> WORD_0",
    "mul_legacy_f32 BYTE_1/DWORD -> DWORD",
    "min_f32 BYTE_2/DWORD -> DWORD",
    "max_f32 DWORD/BYTE_0 -> BYTE_1",
    "max_f32 WORD_1/DWORD -> DWORD",
    "add_f16 BYTE_1/WORD_1 -> BYTE_0",
    "sub_f16 WORD_1/BYTE_2 -> WORD_1",
    "subrev_f16 BYTE_3/WORD_0 -> BYTE_3",
    "mul_f16 WORD_0/BYTE_0 -> DWORD",
    "min_f16 BYTE_0/WORD_1 -> WORD_0",
    "max_f16 WORD_1/BYTE_1 -> BYTE_2",
    "add_f32 WORD_1/DWORD -> BYTE_3",
    "mul_f16 BYTE_2/DWORD -> BYTE_1",
];

fn check(i: usize) {
    fmod_check_rows(&CODE, &ROWS, &EXPECTED, Some(&CHECKED), i, NAMES[i], &fmod_exact);
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => $i:expr;)*) => { $( #[test] $(#[$m])* fn $name() { check($i); } )* };
}

columns! {
    c00_add_f32_byte_0_dword_dword => 0;
    c01_sub_f32_dword_word_1_word_1 => 1;
    c02_subrev_f32_word_0_dword_byte_2 => 2;
    c03_mul_f32_dword_byte_3_word_0 => 3;
    c04_mul_legacy_f32_byte_1_dword_dword => 4;
    c05_min_f32_byte_2_dword_dword => 5;
    c06_max_f32_dword_byte_0_byte_1 => 6;
    c07_max_f32_word_1_dword_dword => 7;
    c08_add_f16_byte_1_word_1_byte_0 => 8;
    c09_sub_f16_word_1_byte_2_word_1 => 9;
    c10_subrev_f16_byte_3_word_0_byte_3 => 10;
    c11_mul_f16_word_0_byte_0_dword => 11;
    c12_min_f16_byte_0_word_1_word_0 => 12;
    c13_max_f16_word_1_byte_1_byte_2 => 13;
    c14_add_f32_word_1_dword_byte_3 => 14;
    c15_mul_f16_byte_2_dword_byte_1 => 15;
}
