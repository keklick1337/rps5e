// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! v_log_f32 and v_log_f16 near 1.0 (and a few far inputs) against
//! hardware-measured rows (FLOAT_MODE 0xC0), within the per-row ULP
//! tolerance the rows were published with; zeros, infinities and NaNs
//! are exact. 64 threads (two wave32 waves), raw buffers.

include!("vectors/fmod_common.rs");
include!("vectors/valu_log_near_one.rs");

/// Position of `bits` on the line of ordered floats of `width` bits.
fn ordered(bits: u32, width: u32) -> u32 {
    let sign = 1u32 << (width - 1);
    if bits & sign != 0 {
        sign - (bits & !sign)
    } else {
        bits | sign
    }
}

fn distance(a: u32, e: u32, width: u32) -> u32 {
    ordered(a, width).abs_diff(ordered(e, width))
}

fn pinned(bits: u32, width: u32) -> bool {
    let (magnitude, infinity) = if width == 16 { (bits & 0x7fff, 0x7c00) } else { (bits & 0x7fff_ffff, 0x7f80_0000) };
    magnitude == 0 || magnitude >= infinity
}

#[test]
fn log_near_one_within_the_measured_tolerance() {
    let input: Vec<u32> = ROWS.iter().flatten().copied().collect();
    let words = THREADS as usize * RESULTS as usize;
    let out = fmod_dispatch(&CODE, 32, THREADS, &input, words, 0).unwrap();
    let mut failures = Vec::new();
    for tid in 0..THREADS as usize {
        for column in 0..2 {
            let width = if column == 0 { 32 } else { 16 };
            let mask = if width == 16 { 0xffff } else { 0xffff_ffff };
            let a = out[tid * RESULTS as usize + column] & mask;
            let e = EXPECTED[tid][column] & mask;
            let ok = if pinned(e, width) { a == e } else { distance(a, e, width) <= TOLERANCE[tid] };
            if !ok {
                failures.push(format!(
                    "lane {tid} {} of {:#010x} is {a:#010x}, expected {e:#010x} (tolerance {} ulp)",
                    if column == 0 { "v_log_f32" } else { "v_log_f16" },
                    ROWS[tid][column],
                    TOLERANCE[tid]
                ));
            }
        }
    }
    fmod_report(&failures);
}
