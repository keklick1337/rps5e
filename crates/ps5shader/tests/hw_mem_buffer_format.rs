// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Formatted buffer loads/stores (buffer_load_format_*, tbuffer_load_format_*,
//! buffer_store_format_*) against hardware-measured reference rows.

include!("vectors/mem_common.rs");
include!("vectors/mem_buffer_format.rs");

const THREADS: u32 = 32;
const INPUTS: usize = 8;
const RESULTS: usize = 32;
const LOADS: usize = 24;
const STORED: usize = 7;
const SCRATCH_WORDS: usize = 16;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;
const SCRATCH_BASE: u64 = 0x3_0000_0000;

/// dword3 of the ten formatted descriptors at user data 8, 12, ... (the
/// first six address the input rows, the last four the scratch buffer).
const FORMATTED_WORD3: [u32; 10] = [0x31038fac, 0x31041fac, 0x31042fac, 0x31039fac, 0x31041977, 0x3103b80d, 0x31038fac, 0x31041fac, 0x31039fac, 0x31042fac];

const NAMES: [&str; LOADS + STORED] = [
    "buffer_load_format_x 8_8_8_8_UNORM x", "buffer_load_format_xy 16_16_16_16_UNORM x", "buffer_load_format_xy 16_16_16_16_UNORM y",
    "buffer_load_format_xyz 16_16_16_16_SNORM x", "buffer_load_format_xyz 16_16_16_16_SNORM y", "buffer_load_format_xyz 16_16_16_16_SNORM z",
    "buffer_load_format_xyzw 8_8_8_8_SNORM x", "buffer_load_format_xyzw 8_8_8_8_SNORM y", "buffer_load_format_xyzw 8_8_8_8_SNORM z",
    "buffer_load_format_xyzw 8_8_8_8_SNORM w", "tbuffer_load_format_x 16_UNORM x", "tbuffer_load_format_xy 10_11_11_FLOAT x",
    "tbuffer_load_format_xy 10_11_11_FLOAT y", "tbuffer_load_format_xyz 2_10_10_10_SNORM x", "tbuffer_load_format_xyz 2_10_10_10_SNORM y",
    "tbuffer_load_format_xyz 2_10_10_10_SNORM z", "buffer_load_format_xyzw 16_16_16_16_UNORM x", "buffer_load_format_xyzw 16_16_16_16_UNORM y",
    "buffer_load_format_xyzw 16_16_16_16_UNORM z", "buffer_load_format_xyzw 16_16_16_16_UNORM w", "buffer_load_format_xyzw 8_8_8_8_SSCALED x",
    "buffer_load_format_xyzw 8_8_8_8_SSCALED y", "buffer_load_format_xyzw 8_8_8_8_SSCALED z", "buffer_load_format_xyzw 8_8_8_8_SSCALED w",
    "scratch dword 0 (store_format_x 8_8_8_8_UNORM)", "scratch dword 1 (store_format_xy 16_16_16_16_UNORM low)", "scratch dword 2 (store_format_xy 16_16_16_16_UNORM high)",
    "scratch dword 3 (store_format_xyz 8_8_8_8_SNORM)", "scratch dword 4 (store_format_xyzw 16_16_16_16_SNORM low)", "scratch dword 5 (store_format_xyzw 16_16_16_16_SNORM high)",
    "scratch dword 6 (store_format_xyzw 8_8_8_8_UNORM)",
];

fn run() -> (Vec<u32>, Vec<u32>) {
    let input: Vec<u32> = ROWS.iter().flatten().copied().collect();
    assert_eq!(input.len(), THREADS as usize * INPUTS);
    let n_out = THREADS as usize * RESULTS;
    let n_scratch = THREADS as usize * SCRATCH_WORDS;
    let mut user = vec![0u32; 8 + 4 * FORMATTED_WORD3.len()];
    put_desc(&mut user, 4, vdesc_full(OUT_BASE, 0, (n_out * 4) as u32, 0x3101_6fac));
    for (k, &w3) in FORMATTED_WORD3.iter().enumerate() {
        let d = if k >= 6 { vdesc_full(SCRATCH_BASE, 0, (n_scratch * 4) as u32, w3) } else { vdesc_full(IN_BASE, 0, (input.len() * 4) as u32, w3) };
        put_desc(&mut user, 8 + 4 * k, d);
    }
    let inp = input.clone();
    let setup = move |m: &mut Machine| {
        m.memory.write_words(IN_BASE, &inp);
        m.memory.write_words(OUT_BASE, &vec![0xdeadbeef; n_out]);
        m.memory.write_words(SCRATCH_BASE, &vec![0xaaaaaaaa; n_scratch]);
    };
    let mut r = dispatch(&CODE, 32, THREADS, user, &setup, &[(OUT_BASE, n_out), (SCRATCH_BASE, n_scratch)]).unwrap();
    let scratch = r.pop().unwrap();
    (r.pop().unwrap(), scratch)
}

/// Check result columns `cols` (indices into `EXPECTED` rows / `NAMES`).
fn check(cols: &[usize]) {
    let (out, scratch) = run();
    let mut c = Checker::new();
    for tid in 0..THREADS as usize {
        for &j in cols {
            let actual = if j < LOADS { out[tid * RESULTS + j] } else { scratch[tid * SCRATCH_WORDS + j - LOADS] };
            c.eq(|| format!("lane {tid} {}", NAMES[j]), actual, EXPECTED[tid][j]);
        }
    }
    c.finish();
}

#[test]
fn load_format_x_8888_unorm() {
    check(&[0]);
}

#[test]
fn load_format_xy_16161616_unorm() {
    check(&[1, 2]);
}

#[test]
fn load_format_xyz_16161616_snorm() {
    check(&[3, 4, 5]);
}

#[test]
fn load_format_xyzw_8888_snorm() {
    check(&[6, 7, 8, 9]);
}

#[test]
fn tbuffer_load_format_x_16_unorm() {
    check(&[10]);
}

#[test]
fn tbuffer_load_format_xy_10_11_11_float() {
    check(&[11, 12]);
}

#[test]
fn tbuffer_load_format_xyz_2_10_10_10_snorm() {
    check(&[13, 14, 15]);
}

#[test]
fn load_format_xyzw_16161616_unorm() {
    check(&[16, 17, 18, 19]);
}

#[test]
fn load_format_xyzw_8888_sscaled() {
    check(&[20, 21, 22, 23]);
}

#[test]
fn store_format_x_8888_unorm() {
    check(&[24]);
}

#[test]
fn store_format_xy_16161616_unorm() {
    check(&[25, 26]);
}

#[test]
fn store_format_xyz_8888_snorm() {
    check(&[27]);
}

#[test]
fn store_format_xyzw_16161616_snorm() {
    check(&[28, 29]);
}

#[test]
fn store_format_xyzw_8888_unorm() {
    check(&[30]);
}
