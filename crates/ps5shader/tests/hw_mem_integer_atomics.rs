// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! 32-bit buffer integer atomics (returned values, EXEC-masked results,
//! no-return memory results, and all lanes hitting one shared dword)
//! against hardware-measured reference rows.

include!("vectors/mem_common.rs");
include!("vectors/mem_integer_atomics.rs");

const INPUTS: usize = 4;
const RESULTS: usize = 128;
const SHARED_WORDS: usize = 16;
const CHECKED: usize = 33;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const NAMES: [&str; CHECKED] = [
    "buffer_atomic_swap returned", "buffer_atomic_add returned", "buffer_atomic_sub returned", "buffer_atomic_smin returned",
    "buffer_atomic_umin returned", "buffer_atomic_smax returned", "buffer_atomic_umax returned", "buffer_atomic_and returned",
    "buffer_atomic_or returned", "buffer_atomic_xor returned", "buffer_atomic_cmpswap returned", "buffer_atomic_swap (exec-masked) result",
    "buffer_atomic_add (exec-masked) result", "buffer_atomic_sub (exec-masked) result", "buffer_atomic_smin (exec-masked) result", "buffer_atomic_umin (exec-masked) result",
    "buffer_atomic_smax (exec-masked) result", "buffer_atomic_umax (exec-masked) result", "buffer_atomic_and (exec-masked) result", "buffer_atomic_or (exec-masked) result",
    "buffer_atomic_xor (exec-masked) result", "buffer_atomic_cmpswap (exec-masked) result", "buffer_atomic_swap (no return) result", "buffer_atomic_add (no return) result",
    "buffer_atomic_sub (no return) result", "buffer_atomic_smin (no return) result", "buffer_atomic_umin (no return) result", "buffer_atomic_smax (no return) result",
    "buffer_atomic_umax (no return) result", "buffer_atomic_and (no return) result", "buffer_atomic_or (no return) result", "buffer_atomic_xor (no return) result",
    "buffer_atomic_cmpswap (no return) result",
];

const SHARED_NAMES: [&str; 6] = [
    "buffer_atomic_add, all lanes on one dword", "buffer_atomic_and, all lanes on one dword", "buffer_atomic_or, all lanes on one dword", "buffer_atomic_xor, all lanes on one dword",
    "buffer_atomic_smin, all lanes on one dword", "buffer_atomic_umax, all lanes on one dword",
];

/// Descriptor dword3 used by this test (raw buffer, OOB_SELECT 3).
const DWORD3: u32 = 0x3101_6fac;

/// Returns (per-lane outputs, input buffer after the dispatch).
fn run(code: &[u32], wave: u32) -> (Vec<u32>, Vec<u32>) {
    let lanes = wave as usize;
    let mut input: Vec<u32> = ROWS[..lanes].iter().flatten().copied().collect();
    input.extend_from_slice(&SHARED_INIT);
    let n_in = input.len();
    let n_out = lanes * RESULTS;
    let mut user = Vec::new();
    put_desc(&mut user, 0, vdesc_full(IN_BASE, 0, ((lanes * INPUTS + SHARED_WORDS) * 4) as u32, DWORD3));
    put_desc(&mut user, 4, vdesc_full(OUT_BASE, 0, (n_out * 4) as u32, DWORD3));
    let setup = move |m: &mut Machine| {
        m.memory.write_words(IN_BASE, &input);
        m.memory.write_words(OUT_BASE, &vec![0xdeadbeef; n_out]);
    };
    let mut r = dispatch(code, wave, wave, user, &setup, &[(OUT_BASE, n_out), (IN_BASE, n_in)]).unwrap();
    let inp = r.remove(1);
    (r.remove(0), inp)
}

fn check(code: &[u32], wave: u32, expected: &[[u32; CHECKED]], columns: std::ops::Range<usize>) {
    let (out, _) = run(code, wave);
    let mut c = Checker::new();
    for (tid, exp) in expected.iter().enumerate() {
        for k in columns.clone() {
            c.eq(|| format!("wave{wave} lane {tid} {} (row {:08x?})", NAMES[k], ROWS[tid]), out[tid * RESULTS + k], exp[k]);
        }
    }
    c.finish();
}

fn check_shared(code: &[u32], wave: u32, expected: &[u32; 6]) {
    let (_, input) = run(code, wave);
    let base = wave as usize * INPUTS;
    let mut c = Checker::new();
    for (k, &e) in expected.iter().enumerate() {
        c.eq(|| format!("wave{wave} {}", SHARED_NAMES[k]), input[base + k], e);
    }
    c.finish();
}

#[test]
fn wave32_returned() {
    check(&WAVE32_CODE, 32, &EXPECTED32, 0..11);
}

#[test]
fn wave32_exec_masked() {
    check(&WAVE32_CODE, 32, &EXPECTED32, 11..22);
}

#[test]
fn wave32_no_return() {
    check(&WAVE32_CODE, 32, &EXPECTED32, 22..33);
}

#[test]
fn wave32_shared_dword() {
    check_shared(&WAVE32_CODE, 32, &SHARED_EXPECTED32);
}

#[test]
fn wave64_returned() {
    check(&WAVE64_CODE, 64, &EXPECTED64, 0..11);
}

#[test]
fn wave64_exec_masked() {
    check(&WAVE64_CODE, 64, &EXPECTED64, 11..22);
}

#[test]
fn wave64_no_return() {
    check(&WAVE64_CODE, 64, &EXPECTED64, 22..33);
}

#[test]
fn wave64_shared_dword() {
    check_shared(&WAVE64_CODE, 64, &SHARED_EXPECTED64);
}
