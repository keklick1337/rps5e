// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Buffer float atomics (fcmpswap / fmin / fmax, 32- and 64-bit) against
//! hardware-measured reference rows.
//!
//! Per lane the shader stores row word 0 (and 1) into the output, then
//! runs the atomic on it with row word 2 (and 3) as data (the 32-bit
//! fcmpswap compares against word 3; the 64-bit one swaps in words 2/3
//! when memory equals words 0/3).

include!("vectors/mem_common.rs");
include!("vectors/mem_float.rs");

const THREADS: u32 = 32;
const RESULTS: usize = 16;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const NAMES: [&str; 14] = [
    "fcmpswap_ret0", "fcmpswap_mem0", "fmin_mem0", "fmax_mem0", "fcmpswap_x2_ret0", "fcmpswap_x2_ret1", "fcmpswap_x2_mem0",
    "fcmpswap_x2_mem1", "fmin_x2_ret0", "fmin_x2_ret1", "fmin_x2_mem0", "fmin_x2_mem1", "fmax_x2_mem0", "fmax_x2_mem1",
];

fn run() -> Vec<u32> {
    let input: Vec<u32> = ROWS.iter().flatten().copied().collect();
    let n_out = THREADS as usize * RESULTS;
    let mut user = Vec::new();
    put_desc(&mut user, 0, vdesc(IN_BASE, (input.len() * 4) as u32));
    put_desc(&mut user, 4, vdesc(OUT_BASE, (n_out * 4) as u32));
    let setup = move |m: &mut Machine| {
        m.memory.write_words(IN_BASE, &input);
        m.memory.write_words(OUT_BASE, &vec![0xdeadbeef; n_out]);
    };
    dispatch(&CODE, 32, THREADS, user, &setup, &[(OUT_BASE, n_out)]).unwrap().remove(0)
}

fn nan32(x: u32) -> bool {
    x & 0x7fff_ffff > 0x7f80_0000
}

fn snan32(x: u32) -> bool {
    nan32(x) && x & 0x0040_0000 == 0
}

fn nan64(x: u64) -> bool {
    x & 0x7fff_ffff_ffff_ffff > 0x7ff0_0000_0000_0000
}

fn snan64(x: u64) -> bool {
    nan64(x) && x & 0x0008_0000_0000_0000 == 0
}

/// Rows whose min/max operands include a signaling NaN, or are both NaN.
fn special32(r: &[u32; 4]) -> bool {
    snan32(r[0]) || snan32(r[2]) || (nan32(r[0]) && nan32(r[2]))
}

fn special64(r: &[u32; 4]) -> bool {
    let a = r[0] as u64 | ((r[1] as u64) << 32);
    let b = r[2] as u64 | ((r[3] as u64) << 32);
    snan64(a) || snan64(b) || (nan64(a) && nan64(b))
}

fn check(columns: &[usize], rows: &dyn Fn(&[u32; 4]) -> bool) {
    let out = run();
    let mut c = Checker::new();
    for tid in 0..THREADS as usize {
        if !rows(&ROWS[tid]) {
            continue;
        }
        for &k in columns {
            c.eq(|| format!("lane {tid} {} (row {:08x?})", NAMES[k], ROWS[tid]), out[tid * RESULTS + k], EXPECTED[tid][k]);
        }
    }
    assert!(c.checks > 0);
    c.finish();
}

#[test]
fn fcmpswap_f32() {
    check(&[0, 1], &|_| true);
}

#[test]
fn fcmpswap_f64() {
    check(&[4, 5, 6, 7], &|_| true);
}

#[test]
fn fmin_fmax_f32() {
    check(&[2, 3], &|r| !special32(r));
}

#[test]
fn fmin_fmax_f32_signaling_or_two_nans() {
    check(&[2, 3], &|r| special32(r));
}

#[test]
fn fmin_fmax_f64() {
    check(&[8, 9, 10, 11, 12, 13], &|r| !special64(r));
}

#[test]
fn fmin_f64_returned_value() {
    check(&[8, 9], &|_| true);
}

#[test]
fn fmin_fmax_f64_signaling_or_two_nans() {
    check(&[10, 11, 12, 13], &|r| special64(r));
}
