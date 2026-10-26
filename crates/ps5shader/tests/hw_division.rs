// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Division helpers (v_div_scale/fmas/fixup_f32) and the compiler's f32
//! division macro against hardware-measured rows.

use ps5shader::testing::IoTest;
use ps5shader::translate::FloatMode;

include!("vectors/division.rs");

const THREADS: usize = 32;
const STRIDE: usize = 4;

/// The rows were measured with FLOAT_MODE 0xF0 and IEEE_MODE.
fn run(code: &[u32], input: &[u32]) -> Vec<u32> {
    let mode = FloatMode { mode: 0xF0, dx10_clamp: true, ieee: true, fp16_overflow: false };
    IoTest::new(code, THREADS as u32, THREADS * STRIDE).with_float_mode(mode).run(input).unwrap()
}

#[test]
fn fma_is_fused() {
    let mut input = vec![0u32; THREADS * STRIDE];
    input[0] = 0xbff2fb6d;
    input[1] = 0xbcaac14c;
    input[2] = 0xbd22126e;
    let out = run(&FMA_CODE, &input);
    assert_eq!(out[0], 0x2f8e8ae0);
}

#[test]
fn division_helpers() {
    let mut failures = Vec::new();
    for chunk in HELPER_VECTORS.chunks(THREADS) {
        let mut input = vec![0u32; THREADS * STRIDE];
        for (lane, v) in chunk.iter().enumerate() {
            input[lane * STRIDE..lane * STRIDE + 4].copy_from_slice(&v[..4]);
        }
        let out = run(&HELPER_CODE, &input);
        for (lane, v) in chunk.iter().enumerate() {
            let o = &out[lane * STRIDE..lane * STRIDE + 4];
            let (s0, s1, s2) = (v[0], v[1], v[2]);
            if (s0 == s1 || s0 == s2) && (o[0] != v[4] || o[1] != v[5]) {
                failures.push(format!("div_scale({s0:#x}, {s1:#x}, {s2:#x}) = {:#x} vcc {} expected {:#x} vcc {}", o[0], o[1], v[4], v[5]));
            }
            if o[2] != v[6] {
                failures.push(format!("div_fmas({s0:#x}, {s1:#x}, {s2:#x}) vcc {} = {:#x} expected {:#x}", v[3], o[2], v[6]));
            }
            if o[3] != v[7] {
                failures.push(format!("div_fixup({s0:#x}, {s1:#x}, {s2:#x}) = {:#x} expected {:#x}", o[3], v[7]));
            }
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn division_macro() {
    let mut failures = Vec::new();
    for chunk in DIVISION_VECTORS.chunks(THREADS) {
        let mut input = vec![0u32; THREADS * STRIDE];
        for (lane, v) in chunk.iter().enumerate() {
            input[lane * STRIDE] = v[0];
            input[lane * STRIDE + 1] = v[1];
        }
        let out = run(&DIVISION_CODE, &input);
        for (lane, v) in chunk.iter().enumerate() {
            if out[lane * STRIDE] != v[2] {
                failures.push(format!("{:#x} / {:#x} = {:#x} expected {:#x}", v[0], v[1], out[lane * STRIDE], v[2]));
            }
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

fn run_mode(code: &[u32], input: &[u32], mode: FloatMode) -> Vec<u32> {
    IoTest::new(code, THREADS as u32, THREADS * STRIDE).with_float_mode(mode).run(input).unwrap()
}

const fn mode(bits: u8, ieee: bool) -> FloatMode {
    FloatMode { mode: bits, dx10_clamp: true, ieee, fp16_overflow: false }
}

/// v_div_scale_f32 flushes its input denormals in the f32 flush modes
/// (rows measured in FLOAT_MODE 0xC0 and 0xF0, with and without
/// IEEE_MODE; the default float mode flushes).
#[test]
fn division_scale_denormals() {
    let mut failures = Vec::new();
    for m in [FloatMode::default(), mode(0xC0, false), mode(0xC0, true), mode(0xF0, false), mode(0xF0, true)] {
        let mut input = vec![0u32; THREADS * STRIDE];
        for (lane, v) in SCALE_VECTORS.iter().enumerate() {
            input[lane * STRIDE..lane * STRIDE + 3].copy_from_slice(&v[..3]);
        }
        let out = run_mode(&SCALE_CODE, &input, m);
        for (lane, v) in SCALE_VECTORS.iter().enumerate() {
            let expected = if (m.mode >> 4) & 3 == 3 { v[4] } else { v[3] };
            let o = &out[lane * STRIDE..];
            if o[0] != expected || o[1] != v[5] {
                failures.push(format!(
                    "mode {:#x} ieee {} lane {lane}: div_scale({:#x}, {:#x}, {:#x}) = {:#x} vcc {}, expected {expected:#x} vcc {}",
                    m.mode, m.ieee, v[0], v[1], v[2], o[0], o[1], v[5]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// v_div_fmas_f32 NaN results follow IEEE_MODE (the first NaN source,
/// quieted only with IEEE_MODE; no NaN addend for inf * 0), with VCC clear
/// and set.
#[test]
fn division_fmas_nan() {
    let mut failures = Vec::new();
    for m in [FloatMode::default(), mode(0xF0, true), mode(0xC0, false), mode(0xF0, false), mode(0xC0, true)] {
        let mut input = vec![0u32; THREADS * STRIDE];
        for lane in 0..FMAS_NAN_VECTORS.len() * 2 {
            let v = FMAS_NAN_VECTORS[lane / 2];
            input[lane * STRIDE..lane * STRIDE + 4].copy_from_slice(&[v[0], v[1], v[2], lane as u32 & 1]);
        }
        let out = run_mode(&HELPER_CODE, &input, m);
        for lane in 0..FMAS_NAN_VECTORS.len() * 2 {
            let v = FMAS_NAN_VECTORS[lane / 2];
            let expected = if m.ieee { v[4] } else { v[3] };
            let actual = out[lane * STRIDE + 2];
            if actual != expected {
                failures.push(format!(
                    "mode {:#x} ieee {} lane {lane} vcc {}: div_fmas({:#x}, {:#x}, {:#x}) = {actual:#x}, expected {expected:#x}",
                    m.mode,
                    m.ieee,
                    lane & 1,
                    v[0],
                    v[1],
                    v[2]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}
