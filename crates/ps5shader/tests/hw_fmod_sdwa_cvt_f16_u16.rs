// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! v_cvt_f16_u16 with SDWA byte source selects (BYTE_0..BYTE_3) and
//! WORD_0 / WORD_1 / DWORD destinations, checked against the reference
//! model: 64 threads (two wave32 waves), stride-4 buffers.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_sdwa_cvt_f16_u16.rs");

const THREADS: usize = 64;
const INPUTS: usize = 4;
const RESULTS: usize = 4;
const GARBAGE: u32 = 0xabcd_1234;

/// f16 bits of a small unsigned integer (exact).
fn half_bits(value: u32) -> u32 {
    if value == 0 {
        return 0;
    }
    let exponent = 31 - value.leading_zeros();
    let fraction = (value << (10 - exponent)) & 0x3ff;
    ((exponent + 15) << 10) | fraction
}

#[test]
fn byte_sources_convert_and_destinations_keep_the_other_half() {
    let mut input = vec![0u32; THREADS * INPUTS];
    for tid in 0..THREADS as u32 {
        input[tid as usize * INPUTS] = tid | ((tid + 64) << 8) | ((255 - tid) << 16) | (((tid * 3 + 128) & 0xff) << 24);
    }
    let out = fmod_dispatch(&SDWA_CODE, 32, THREADS as u32, &input, THREADS * RESULTS, 4).unwrap();
    let mut failures = Vec::new();
    for tid in 0..THREADS {
        let source = input[tid * INPUTS];
        let half = |byte: u32| half_bits((source >> (byte * 8)) & 0xff);
        let o = &out[tid * RESULTS..];
        let expected = [(GARBAGE & 0xffff_0000) | half(3), half(0), (half(1) << 16) | (GARBAGE & 0xffff), half(2)];
        let what = [
            "src0_sel:BYTE_3 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE must convert the top byte and keep the high half",
            "src0_sel:BYTE_0 dst_sel:DWORD must convert the low byte",
            "src0_sel:BYTE_1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE must keep the low half",
            "src0_sel:BYTE_2 dst_sel:DWORD must convert the third byte",
        ];
        for r in 0..4 {
            if o[r] != expected[r] {
                failures.push(format!("thread {tid} result {r} is {:#010x}, expected {:#010x}: {}", o[r], expected[r], what[r]));
            }
        }
    }
    fmod_report(&failures);
}
