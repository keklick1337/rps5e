// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! buffer_atomic_or_x2 / buffer_atomic_swap_x2 (with and without return,
//! under a partial EXEC mask, and through a V# whose range ends inside the
//! lane's slots) against hardware-measured reference rows.

include!("vectors/mem_common.rs");
include!("vectors/mem_or_swap64.rs");

const INPUTS: usize = 4;
const RESULTS: usize = 128;
const CHECKED: usize = 28;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const NAMES: [&str; CHECKED] = [
    "buffer_atomic_or_x2 returned lo", "buffer_atomic_or_x2 returned hi", "buffer_atomic_swap_x2 returned lo", "buffer_atomic_swap_x2 returned hi",
    "buffer_atomic_or_x2 returned lo (exec-masked)", "buffer_atomic_or_x2 returned hi (exec-masked)", "buffer_atomic_swap_x2 returned lo (exec-masked)", "buffer_atomic_swap_x2 returned hi (exec-masked)",
    "buffer_atomic_or_x2 returned lo (V# end)", "buffer_atomic_or_x2 returned hi (V# end)", "buffer_atomic_swap_x2 returned lo (V# end)", "buffer_atomic_swap_x2 returned hi (V# end)",
    "or_x2 memory lo", "or_x2 memory hi", "or_x2 (no return) memory lo", "or_x2 (no return) memory hi",
    "swap_x2 memory lo", "swap_x2 memory hi", "swap_x2 (no return) memory lo", "swap_x2 (no return) memory hi",
    "or_x2 (exec-masked) memory lo", "or_x2 (exec-masked) memory hi", "swap_x2 (exec-masked) memory lo", "swap_x2 (exec-masked) memory hi",
    "or_x2 (V# end) memory lo", "or_x2 (V# end) memory hi", "swap_x2 (V# end) memory lo", "swap_x2 (V# end) memory hi",
];

/// Descriptor dword3 used by this test (raw buffer, OOB_SELECT 3).
const DWORD3: u32 = 0x3101_6fac;

fn run(code: &[u32], wave: u32) -> Vec<u32> {
    let lanes = wave as usize;
    let input: Vec<u32> = ROWS[..lanes].iter().flatten().copied().collect();
    let n_out = lanes * RESULTS;
    let mut user = Vec::new();
    put_desc(&mut user, 0, vdesc_full(IN_BASE, 0, (lanes * INPUTS * 4) as u32, DWORD3));
    put_desc(&mut user, 4, vdesc_full(OUT_BASE, 0, (n_out * 4) as u32, DWORD3));
    let setup = move |m: &mut Machine| {
        m.memory.write_words(IN_BASE, &input);
        m.memory.write_words(OUT_BASE, &vec![0xdeadbeef; n_out]);
    };
    dispatch(code, wave, wave, user, &setup, &[(OUT_BASE, n_out)]).unwrap().remove(0)
}

fn check(code: &[u32], wave: u32, expected: &[[u32; CHECKED]], columns: &[usize]) {
    let out = run(code, wave);
    let mut c = Checker::new();
    for (tid, exp) in expected.iter().enumerate() {
        for &k in columns {
            c.eq(|| format!("wave{wave} lane {tid} {} (row {:08x?})", NAMES[k], ROWS[tid]), out[tid * RESULTS + k], exp[k]);
        }
    }
    c.finish();
}

const RETURNED: [usize; 4] = [0, 1, 2, 3];
const MASKED: [usize; 8] = [4, 5, 6, 7, 20, 21, 22, 23];
const VSHARP_END: [usize; 8] = [8, 9, 10, 11, 24, 25, 26, 27];
const MEMORY: [usize; 8] = [12, 13, 14, 15, 16, 17, 18, 19];

#[test]
fn wave32_returned() {
    check(&WAVE32_CODE, 32, &EXPECTED32, &RETURNED);
}

#[test]
fn wave32_memory() {
    check(&WAVE32_CODE, 32, &EXPECTED32, &MEMORY);
}

#[test]
fn wave32_exec_masked() {
    check(&WAVE32_CODE, 32, &EXPECTED32, &MASKED);
}

#[test]
fn wave32_vsharp_end() {
    check(&WAVE32_CODE, 32, &EXPECTED32, &VSHARP_END);
}

#[test]
fn wave64_returned() {
    check(&WAVE64_CODE, 64, &EXPECTED64, &RETURNED);
}

#[test]
fn wave64_memory() {
    check(&WAVE64_CODE, 64, &EXPECTED64, &MEMORY);
}

#[test]
fn wave64_exec_masked() {
    check(&WAVE64_CODE, 64, &EXPECTED64, &MASKED);
}

#[test]
fn wave64_vsharp_end() {
    check(&WAVE64_CODE, 64, &EXPECTED64, &VSHARP_END);
}
