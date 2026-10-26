// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! D16 formatted buffer loads (buffer_load_format_d16_*, including the
//! d16_hi forms) against hardware-measured reference rows.

include!("vectors/mem_common.rs");
include!("vectors/mem_format_d16.rs");

const THREADS: u32 = 32;
const RESULTS: usize = 16;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const NAMES: [&str; RESULTS] = ["8unorm", "8snorm", "16unorm", "16snorm", "16uscaled", "16sscaled", "8sint", "32float", "32uint", "32sint", "xy_8_8unorm", "xyz_lo", "xyz_hi", "hi_x_32float", "xyzw_lo", "xyzw_hi"];

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
    format_d16_x_8_unorm => [0];
    format_d16_x_8_snorm => [1];
    format_d16_x_16_unorm => [2];
    format_d16_x_16_snorm => [3];
    format_d16_x_16_uscaled => [4];
    format_d16_x_16_sscaled => [5];
    format_d16_x_8_sint => [6];
    format_d16_x_32_float => [7];
    format_d16_x_32_uint => [8];
    format_d16_x_32_sint => [9];
    format_d16_xy_8_8_unorm => [10];
    format_d16_xyz => [11, 12];
    format_d16_hi_x_32_float => [13];
    format_d16_xyzw => [14, 15];
}
