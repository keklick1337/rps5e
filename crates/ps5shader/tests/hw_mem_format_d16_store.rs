// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! D16 formatted buffer stores (tbuffer_store_format_d16_*, including the
//! d16_hi forms) against hardware-measured reference rows.

include!("vectors/mem_common.rs");
include!("vectors/mem_format_d16_store.rs");

const THREADS: u32 = 32;
const RESULTS: usize = 16;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const NAMES: [&str; RESULTS] = ["8unorm", "8snorm", "16unorm", "8uscaled", "8uint", "8sint", "16float", "32float", "32uint", "32sint", "uf11", "xy_8_8unorm", "xyz_8888uint", "xyzw_f16_lo", "xyzw_f16_hi", "hi_x_32float"];

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
    store_d16_x_8_unorm => [0];
    store_d16_x_8_snorm => [1];
    store_d16_x_16_unorm => [2];
    store_d16_x_8_uscaled => [3];
    store_d16_x_8_uint => [4];
    store_d16_x_8_sint => [5];
    store_d16_x_16_float => [6];
    store_d16_x_32_float => [7];
    store_d16_x_32_uint => [8];
    store_d16_x_32_sint => [9];
    store_d16_x_uf11 => [10];
    store_d16_xy_8_8_unorm => [11];
    store_d16_xyz_8888_uint => [12];
    store_d16_xyzw_16_float => [13, 14];
    store_d16_hi_x_32_float => [15];
}
