// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! NaN payloads through v_cvt_f64_f32 / v_cvt_f32_f64 against
//! hardware-measured reference rows (exact compare).

include!("vectors/fmod_common.rs");
include!("vectors/fmod_f64_nan_conversions.rs");

#[test]
fn cvt_f64_f32_lo() {
    // Measured with FLOAT_MODE 0xF0 and IEEE_MODE.
    fmod_set_float_mode(FMOD_IEEE_F0);
    fmod_check_rows(&CODE, &ROWS, &EXPECTED, None, 0, "cvt f64 f32 lo", &fmod_exact);
}

#[test]
fn cvt_f64_f32_hi() {
    // Measured with FLOAT_MODE 0xF0 and IEEE_MODE.
    fmod_set_float_mode(FMOD_IEEE_F0);
    fmod_check_rows(&CODE, &ROWS, &EXPECTED, None, 1, "cvt f64 f32 hi", &fmod_exact);
}

#[test]
fn cvt_f32_f64() {
    // Measured with FLOAT_MODE 0xF0 and IEEE_MODE.
    fmod_set_float_mode(FMOD_IEEE_F0);
    fmod_check_rows(&CODE, &ROWS, &EXPECTED, None, 2, "cvt f32 f64", &fmod_exact);
}
