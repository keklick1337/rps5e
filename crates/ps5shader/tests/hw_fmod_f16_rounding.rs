// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! f16 rounding (v_cvt_f16_f32, v_add_f16, v_mul_f16) against
//! hardware-measured reference rows. 64 threads per workgroup, wave32.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_f16_rounding.rs");

const THREADS: usize = 64;
const INPUTS: usize = 4;
const RESULTS: usize = 16;
const NAMES: [&str; 3] = ["v_cvt_f16_f32", "v_add_f16", "v_mul_f16"];

fn is_nan16(b: u32) -> bool {
    (b & 0x7fff) > 0x7c00
}

fn matches(a: u32, e: u32) -> bool {
    a == e || (a >> 16 == e >> 16 && is_nan16(a) && is_nan16(e))
}

fn check(c: usize) {
    let mut failures = Vec::new();
    for (batch, chunk) in VECTORS.chunks(THREADS).enumerate() {
        let mut input = vec![0u32; THREADS * INPUTS];
        for (lane, v) in chunk.iter().enumerate() {
            input[lane * INPUTS..lane * INPUTS + 3].copy_from_slice(&v[..3]);
        }
        let out = fmod_dispatch(&ROUNDING_CODE, 32, THREADS as u32, &input, THREADS * RESULTS, 4).unwrap();
        for (lane, v) in chunk.iter().enumerate() {
            let (a, e) = (out[lane * RESULTS + c], v[3 + c]);
            if !matches(a, e) {
                failures.push(format!("vector {} ({:#010x}, {:#010x}, {:#010x}) {}: {a:#010x}, expected {e:#010x}", batch * THREADS + lane, v[0], v[1], v[2], NAMES[c]));
            }
        }
    }
    fmod_report(&failures);
}

/// v_add_f16 and v_mul_f16 in the four f16 denormal modes (FLOAT_MODE
/// denorm16 << 6), with and without IEEE_MODE and FP16_OVFL: in the
/// round-to-nearest mode flushed inputs and results below the smallest
/// normal are zeros of their sign.
#[test]
fn add_mul_f16_denormal_modes() {
    let mut failures = Vec::new();
    for ieee in [false, true] {
        for overflow in [false, true] {
            for denorm16 in 0..4u8 {
                let mut vectors = DENORMAL_VECTORS[denorm16 as usize];
                if overflow {
                    vectors[27][5] = 0xabcd_7bff;
                }
                let mut input = vec![0u32; 32 * INPUTS];
                for (lane, v) in vectors.iter().enumerate() {
                    input[lane * INPUTS..lane * INPUTS + 3].copy_from_slice(&v[..3]);
                }
                let mode = ps5shader::translate::FloatMode { mode: denorm16 << 6, dx10_clamp: true, ieee, fp16_overflow: overflow };
                let out = fmod_dispatch_mode(&ROUNDING_CODE, 32, 32, &input, 32 * RESULTS, 4, mode).unwrap();
                for (lane, v) in vectors.iter().enumerate() {
                    for c in 1..3 {
                        let (a, e) = (out[lane * RESULTS + c], v[3 + c]);
                        if !matches(a, e) {
                            failures.push(format!(
                                "ieee {ieee} overflow {overflow} denorm16 {denorm16} vector {lane} ({:#010x}, {:#010x}, {:#010x}) {}: {a:#010x}, expected {e:#010x}",
                                v[0], v[1], v[2], NAMES[c]
                            ));
                        }
                    }
                }
            }
        }
    }
    fmod_report(&failures);
}

#[test]
fn cvt_f16_f32() {
    check(0);
}

#[test]
fn add_f16() {
    check(1);
}

#[test]
fn mul_f16() {
    check(2);
}
