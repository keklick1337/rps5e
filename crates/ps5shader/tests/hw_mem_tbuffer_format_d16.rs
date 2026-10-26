// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! D16 typed buffer loads (tbuffer_load_format_d16_xyzw with explicit and
//! descriptor formats) against hardware-measured reference rows.

include!("vectors/mem_common.rs");
include!("vectors/mem_tbuffer_format_d16.rs");

const THREADS: u32 = 32;
const RESULTS: usize = 16;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const NAMES: [&str; RESULTS] = [
    "16_16_16_16_UNORM offset 0 xy", "16_16_16_16_UNORM offset 0 zw",
    "16_16_16_16_SNORM offset 8 xy", "16_16_16_16_SNORM offset 8 zw",
    "16_16_16_16_USCALED offset 0 xy", "16_16_16_16_USCALED offset 0 zw",
    "16_16_16_16_SSCALED offset 8 xy", "16_16_16_16_SSCALED offset 8 zw",
    "8_8_8_8_UNORM swizzled offset 0 xy", "8_8_8_8_UNORM swizzled offset 0 zw",
    "8_8_8_8_SINT swizzled offset 4 xy", "8_8_8_8_SINT swizzled offset 4 zw",
    "32_32_32_32_FLOAT offset 0 xy", "32_32_32_32_FLOAT offset 0 zw",
    "16_16_16_16_FLOAT swizzled offset 8 xy", "16_16_16_16_FLOAT swizzled offset 8 zw",
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
    d16_xyzw_16_16_16_16_unorm => [0, 1];
    d16_xyzw_16_16_16_16_snorm => [2, 3];
    d16_xyzw_16_16_16_16_uscaled => [4, 5];
    d16_xyzw_16_16_16_16_sscaled => [6, 7];
    d16_xyzw_8_8_8_8_unorm_swizzled => [8, 9];
    d16_xyzw_8_8_8_8_sint_swizzled => [10, 11];
    d16_xyzw_32_32_32_32_float => [12, 13];
    d16_xyzw_16_16_16_16_float_swizzled => [14, 15];
}
