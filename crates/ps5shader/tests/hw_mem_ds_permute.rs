// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! ds_permute_b32 / ds_bpermute_b32 (wave32) against hardware-measured
//! reference rows.

include!("vectors/mem_common.rs");
include!("vectors/mem_ds_permute.rs");

const THREADS: u32 = 32;
const INPUTS: usize = 4;
const RESULTS: usize = 16;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

fn run() -> Vec<u32> {
    let mut input = vec![0u32; THREADS as usize * INPUTS];
    for tid in 0..THREADS as usize {
        input[tid * INPUTS] = ROWS[tid][0];
        input[tid * INPUTS + 1] = ROWS[tid][1];
    }
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

fn check(slots: &[usize]) {
    let out = run();
    let mut c = Checker::new();
    for tid in 0..THREADS as usize {
        for &j in slots {
            c.eq(|| format!("lane {tid} {}", NAMES[j]), out[tid * RESULTS + j], EXPECTED[tid][j]);
        }
    }
    c.finish();
}

#[test]
fn ds_permute() {
    check(&[0]);
}

#[test]
fn ds_bpermute() {
    check(&[1]);
}

#[test]
fn ds_permute_offset8() {
    check(&[2]);
}

#[test]
fn ds_permute_exec_low16() {
    check(&[3]);
}
