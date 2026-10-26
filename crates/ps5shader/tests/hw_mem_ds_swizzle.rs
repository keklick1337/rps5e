// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! ds_swizzle_b32 (bitmask, quad-perm, rotate, inactive sources) in wave32
//! and wave64 against hardware-measured reference rows.

include!("vectors/mem_common.rs");
include!("vectors/mem_ds_swizzle.rs");

const MAX_THREADS: usize = 64;
const INPUTS: usize = 4;
const RESULTS: usize = 16;
const CHECKED: usize = 16;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

fn run(code: &[u32], wave: u32) -> Vec<u32> {
    let mut input = vec![0u32; MAX_THREADS * INPUTS];
    for tid in 0..MAX_THREADS {
        input[tid * INPUTS..][..INPUTS].copy_from_slice(&ROWS[tid]);
    }
    let n_out = MAX_THREADS * RESULTS;
    let mut user = Vec::new();
    put_desc(&mut user, 0, vdesc(IN_BASE, (input.len() * 4) as u32));
    put_desc(&mut user, 4, vdesc(OUT_BASE, (n_out * 4) as u32));
    let setup = move |m: &mut Machine| {
        m.memory.write_words(IN_BASE, &input);
        m.memory.write_words(OUT_BASE, &vec![0xdeadbeef; n_out]);
    };
    dispatch(code, wave, wave, user, &setup, &[(OUT_BASE, n_out)]).unwrap().remove(0)
}

fn check<const L: usize>(code: &[u32], wave: u32, expected: &[[u32; CHECKED]; L], slots: &[usize]) {
    let out = run(code, wave);
    let mut c = Checker::new();
    for tid in 0..L {
        for &j in slots {
            c.eq(|| format!("wave{wave} lane {tid} {}", NAMES[j]), out[tid * RESULTS + j], expected[tid][j]);
        }
    }
    c.finish();
}

const ALL: [usize; CHECKED] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

#[test]
fn ds_swizzle_wave32() {
    check(&WAVE32_CODE, 32, &EXPECTED32, &ALL);
}

#[test]
fn ds_swizzle_wave64() {
    check(&WAVE64_CODE, 64, &EXPECTED64, &ALL);
}

/// The FFT swizzle (offset 0xE000 and up) is refused, not read as a
/// rotate: its lane pattern is neither documented nor measured.
#[test]
fn ds_swizzle_fft_mode_is_refused() {
    use ps5shader::testing::{asm, ComputeRun};
    for offset in [0xE000u32, 0xE00F, 0xFFFF] {
        let mut code = asm::ds(0x35, 1, 0, 0, 0, offset & 0xFF, offset >> 8);
        code.push(0xBF81_0000);
        let err = ComputeRun::new(code, 32, 32, vec![0; 8]).compile(true).unwrap_err();
        assert!(err.contains("FFT"), "{offset:#x}: {err}");
    }
    // The rotate mode just below still translates.
    let mut code = asm::ds(0x35, 1, 0, 0, 0, 0xFF, 0xDF);
    code.push(0xBF81_0000);
    ComputeRun::new(code, 32, 32, vec![0; 8]).compile(true).unwrap();
}
