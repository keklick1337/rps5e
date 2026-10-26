// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! LDS reads and writes (sub-dword, read2/write2, st64, b64, b96, d16_hi,
//! past the end of the allocation) in wave32 and wave64 against
//! hardware-measured reference rows.

include!("vectors/mem_common.rs");
include!("vectors/mem_lds_read_write.rs");

const MAX_THREADS: usize = 64;
const INPUTS: usize = 4;
const RESULTS: usize = 64;
const CHECKED: usize = 58;
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
    let lds_bytes = wave as usize * 32 * 4;
    let setup = move |m: &mut Machine| {
        m.lds = vec![0; lds_bytes];
        m.memory.write_words(IN_BASE, &input);
        m.memory.write_words(OUT_BASE, &vec![0xdeadbeef; n_out]);
    };
    dispatch(code, wave, wave, user, &setup, &[(OUT_BASE, n_out)]).unwrap().remove(0)
}

fn check<const L: usize>(code: &[u32], wave: u32, expected: &[[u32; CHECKED]; L], names: &[&str; CHECKED], slots: std::ops::Range<usize>) {
    let out = run(code, wave);
    let mut c = Checker::new();
    for tid in 0..L {
        for j in slots.clone() {
            c.eq(|| format!("wave{wave} lane {tid} {}", names[j]), out[tid * RESULTS + j], expected[tid][j]);
        }
    }
    c.finish();
}

/// Slots 0..26: values returned by the reads.
const READS: std::ops::Range<usize> = 0..26;
/// Slots 26..58: the 32 LDS planes after the writes.
const PLANES: std::ops::Range<usize> = 26..58;

#[test]
fn lds_read_write_wave32_reads() {
    check(&WAVE32_CODE, 32, &EXPECTED32, &NAMES32, READS);
}

#[test]
fn lds_read_write_wave32_planes() {
    check(&WAVE32_CODE, 32, &EXPECTED32, &NAMES32, PLANES);
}

#[test]
fn lds_read_write_wave64_reads() {
    check(&WAVE64_CODE, 64, &EXPECTED64, &NAMES64, READS);
}

#[test]
fn lds_read_write_wave64_planes() {
    check(&WAVE64_CODE, 64, &EXPECTED64, &NAMES64, PLANES);
}
