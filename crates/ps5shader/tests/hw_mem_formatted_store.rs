// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Formatted buffer stores (tbuffer_store_format_*) against
//! hardware-measured reference rows.

include!("vectors/mem_common.rs");
include!("vectors/mem_formatted_store.rs");

const THREADS: u32 = 32;
const RESULTS: usize = 16;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const NAMES: [&str; RESULTS] = ["8unorm", "8snorm", "16unorm", "16snorm", "8uscaled", "16sscaled", "8uint", "16sint", "16float", "uf11_x", "unorm1010102_xyzw", "f16x2_xy", "uint32x2_x", "uint32x2_y", "sint8888_xyz", "snorm1010102_xy"];

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

fn check(cols: &[usize]) {
    let out = run();
    let mut c = Checker::new();
    for tid in 0..THREADS as usize {
        for &j in cols {
            c.eq(|| format!("lane {tid} {}", NAMES[j]), out[tid * RESULTS + j], EXPECTED[tid][j]);
        }
    }
    c.finish();
}

macro_rules! cols {
    ($($(#[$m:meta])* $name:ident => [$($c:expr),*];)*) => {
        $( $(#[$m])* #[test] fn $name() { check(&[$($c),*]); } )*
    };
}

cols! {
    store_x_8_unorm => [0];
    store_x_8_snorm => [1];
    store_x_16_unorm => [2];
    store_x_16_snorm => [3];
    store_x_8_uscaled => [4];
    store_x_16_sscaled => [5];
    store_x_8_uint => [6];
    store_x_16_sint => [7];
    store_x_16_float => [8];
    store_x_uf11 => [9];
    store_xyzw_unorm_10_10_10_2 => [10];
    store_xy_16_16_float => [11];
    store_x_32_32_uint_x => [12];
    store_x_32_32_uint_y => [13];
    store_xyz_8888_sint => [14];
    store_xy_snorm_10_10_10_2 => [15];
}
