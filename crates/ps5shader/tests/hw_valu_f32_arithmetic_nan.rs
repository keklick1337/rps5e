// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! NaN results of f32 arithmetic (add/sub/subrev/mul/fma/fmac/legacy forms,
//! fmamk/fmaak, fma_mix with f32 sources, source and output modifiers,
//! clamp) against hardware-measured reference rows, with and without
//! IEEE_MODE (exact compare: the NaN payload and sign are chosen on the
//! bits of the first NaN source, an invalid operation yields 0xffc00000).

include!("vectors/fmod_common.rs");
include!("vectors/valu_f32_arithmetic_nan.rs");

use ps5shader::translate::FloatMode;

const NAMES: [&str; 16] = [
    "v_add_f32",
    "v_sub_f32",
    "v_subrev_f32",
    "v_mul_f32",
    "v_fma_f32",
    "v_fmac_f32",
    "v_mul_legacy_f32",
    "v_fma_legacy_f32",
    "v_fmamk_f32",
    "v_fmaak_f32",
    "v_fma_mix_f32",
    "v_add_f32 -v4 |v5|",
    "v_mul_f32 mul:2",
    "v_fma_f32 -v4 v5 -v6",
    "v_sub_f32 |v4| -v5",
    "v_add_f32 clamp",
];

fn check_all(mode: FloatMode, expected: &[[u32; 16]; 32], what: &str) {
    fmod_set_float_mode(mode);
    for (column, name) in NAMES.iter().enumerate() {
        fmod_check_rows(&CODE, &ROWS, expected, None, column, &format!("{what} {name}"), &fmod_exact);
    }
}

#[test]
fn default_float_mode() {
    check_all(FloatMode::default(), &EXPECTED_NO_IEEE, "no float mode");
}

#[test]
fn ieee_off() {
    check_all(FloatMode { mode: 0xC0, dx10_clamp: true, ieee: false, fp16_overflow: false }, &EXPECTED_NO_IEEE, "IEEE=0");
}

#[test]
fn ieee_on() {
    check_all(FloatMode { mode: 0xC0, dx10_clamp: true, ieee: true, fp16_overflow: false }, &EXPECTED_IEEE, "IEEE=1");
}
