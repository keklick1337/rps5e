// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! SDWA destination selection (dst_sel / dst_unused) against the reference
//! model checks: 64 threads (two wave32 waves), stride-4 buffers.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_sdwa_destination.rs");

const THREADS: usize = 64;
const INPUTS: usize = 4;
const RESULTS: usize = 16;
const GARBAGE: u32 = 0xabcd_1234;

fn quarter(index: u32) -> f32 {
    ((index % 17) as i32 - 8) as f32 / 4.0
}

/// f16 bits of an exactly representable normal value (or zero).
fn half_bits(v: f32) -> u32 {
    let b = v.to_bits();
    let sign = (b >> 16) & 0x8000;
    if v == 0.0 {
        return sign;
    }
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&exp) && b & 0x1fff == 0, "model value {v} is not a normal f16");
    sign | (exp as u32) << 10 | (b >> 13) & 0x3ff
}

fn run() -> (Vec<u32>, Vec<u32>) {
    let mut input = vec![0u32; THREADS * INPUTS];
    for tid in 0..THREADS as u32 {
        let w = &mut input[tid as usize * INPUTS..];
        w[0] = quarter(tid * 3 + 1).to_bits();
        w[1] = quarter(tid * 5 + 2).to_bits();
        w[2] = half_bits(quarter(tid * 7 + 3)) | half_bits(quarter(tid * 11 + 4)) << 16;
    }
    let out = fmod_dispatch(&SDWA_CODE, 32, THREADS as u32, &input, THREADS * RESULTS, 4).unwrap();
    (input, out)
}

fn check(result: usize) {
    let (input, out) = run();
    let mut failures = Vec::new();
    for tid in 0..THREADS {
        let inw = &input[tid * INPUTS..];
        let a = half_bits(f32::from_bits(inw[0]));
        let b = half_bits(f32::from_bits(inw[1]));
        let sum = half_bits(quarter(tid as u32 * 7 + 3) + quarter(tid as u32 * 11 + 4));
        let low = inw[0] & 0xff;
        let (expected, what) = match result {
            0 => (a, "v_cvt_f16_f32_sdwa dst_sel:DWORD dst_unused:UNUSED_PAD must zero the high half"),
            1 => ((GARBAGE & 0xffff_0000) | a, "v_cvt_f16_f32_e32 must keep the high half"),
            2 => ((b << 16) | (GARBAGE & 0xffff), "dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE must keep the low half"),
            3 => (a | (b << 16), "two padded conversions packed with v_lshl_add_u32"),
            4 => (sum, "v_add_f16_sdwa dst_sel:DWORD dst_unused:UNUSED_PAD must zero the high half"),
            5 => ((GARBAGE & 0xffff_ff00) | low, "v_mov_b32_sdwa dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE must keep the other bytes"),
            6 => ((GARBAGE & 0xff00_ffff) | (low << 16), "v_mov_b32_sdwa dst_sel:BYTE_2 dst_unused:UNUSED_PRESERVE must keep the other bytes"),
            7 => (low << 8, "v_mov_b32_sdwa dst_sel:BYTE_1 dst_unused:UNUSED_PAD must zero the other bytes"),
            8 => ((GARBAGE & 0xffff_0000) | (!inw[0] & 0xffff), "v_not_b32_sdwa dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE must keep the high half"),
            9 => ((GARBAGE & 0xffff_0000) | half_bits((inw[0] >> 24) as f32), "v_cvt_f16_u16_sdwa dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_3 must convert the zero-extended top byte"),
            _ => ((GARBAGE & 0xffff_0000) | half_bits((inw[0] >> 24) as u8 as i8 as f32), "v_cvt_f16_i16_sdwa dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_3 src0_sext must convert the sign-extended top byte"),
        };
        let actual = out[tid * RESULTS + result];
        if actual != expected {
            failures.push(format!("thread {tid} result {result} is {actual:#010x}, expected {expected:#010x}: {what}"));
        }
    }
    fmod_report(&failures);
}

#[test]
fn cvt_f16_f32_sdwa_dword_pad() {
    check(0);
}

#[test]
fn cvt_f16_f32_e32_keeps_high() {
    check(1);
}

#[test]
fn word1_preserve() {
    check(2);
}

#[test]
fn padded_pack() {
    check(3);
}

#[test]
fn add_f16_sdwa_dword_pad() {
    check(4);
}

#[test]
fn mov_sdwa_byte0_preserve() {
    check(5);
}

#[test]
fn mov_sdwa_byte2_preserve() {
    check(6);
}

#[test]
fn mov_sdwa_byte1_pad() {
    check(7);
}

#[test]
fn not_sdwa_word0_preserve() {
    check(8);
}

#[test]
fn cvt_f16_u16_sdwa_byte3_preserve() {
    check(9);
}

#[test]
fn cvt_f16_i16_sdwa_sext_byte3_preserve() {
    check(10);
}
