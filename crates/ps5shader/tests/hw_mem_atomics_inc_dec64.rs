// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! buffer_atomic_inc_x2 / buffer_atomic_dec_x2 against hardware-measured
//! reference rows.
//!
//! Per lane the shader stores row words 0/1 into the output, then runs the
//! 64-bit atomic on it with words 2/3 as the limit.

include!("vectors/mem_common.rs");
include!("vectors/mem_inc_dec64.rs");

const THREADS: u32 = 32;
const RESULTS: usize = 16;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const NAMES: [&str; 8] = ["inc_x2_ret0", "inc_x2_ret1", "inc_x2_mem0", "inc_x2_mem1", "dec_x2_ret0", "dec_x2_ret1", "dec_x2_mem0", "dec_x2_mem1"];

fn check(columns: &[usize]) {
    let input: Vec<u32> = ROWS.iter().flatten().copied().collect();
    let n_out = THREADS as usize * RESULTS;
    let mut user = Vec::new();
    put_desc(&mut user, 0, vdesc(IN_BASE, (input.len() * 4) as u32));
    put_desc(&mut user, 4, vdesc(OUT_BASE, (n_out * 4) as u32));
    let setup = move |m: &mut Machine| {
        m.memory.write_words(IN_BASE, &input);
        m.memory.write_words(OUT_BASE, &vec![0xdeadbeef; n_out]);
    };
    let out = dispatch(&CODE, 32, THREADS, user, &setup, &[(OUT_BASE, n_out)]).unwrap().remove(0);
    let mut c = Checker::new();
    for tid in 0..THREADS as usize {
        for &k in columns {
            c.eq(|| format!("lane {tid} {} (row {:08x?})", NAMES[k], ROWS[tid]), out[tid * RESULTS + k], EXPECTED[tid][k]);
        }
    }
    c.finish();
}

#[test]
fn inc_x2() {
    check(&[0, 1, 2, 3]);
}

#[test]
fn dec_x2() {
    check(&[4, 5, 6, 7]);
}
