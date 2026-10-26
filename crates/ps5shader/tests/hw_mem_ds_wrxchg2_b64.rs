// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! ds_wrxchg2_rtn_b64 / ds_wrxchg2st64_rtn_b64 against hardware-measured
//! reference rows, plus the forms the reference compiler rejects.

include!("vectors/mem_common.rs");

const THREADS: u32 = 32;
const INPUTS: usize = 8;
const RESULTS: usize = 16;
const LDS_DWORDS: usize = 3072;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const CODE: [u32; 35] = [
    0x34020085, 0xe0381000, 0x80000201, 0xe0381010, 0x80000601, 0x345c0085, 0x4a5c5cff, 0x00000480,
    0x345e0083, 0x4a5e5eff, 0x00000880, 0xbf8c3f70, 0xd9380301, 0x0004022e, 0xd9b80301, 0x1208062e,
    0xd9dc0301, 0x1600002e, 0xd93c0100, 0x0004022f, 0xd9bc0100, 0x1a08062f, 0xd9e00100, 0x1e00002f,
    0x34020086, 0xbf8cc07f, 0xe0781000, 0x80011201, 0xe0781010, 0x80011601, 0xe0781020, 0x80011a01,
    0xe0781030, 0x80011e01, 0xbf810000,
];

const COINCIDING_OFFSETS: [u32; 3] = [0xd9b80101, 0x1208062e, 0xbf810000];
const COINCIDING_OFFSETS_ST64: [u32; 3] = [0xd9bc0000, 0x1a08062f, 0xbf810000];
const DESTINATION_OVERFLOW: [u32; 3] = [0xd9b80301, 0xfd08062e, 0xbf810000];

fn fill(tid: u32, words: &mut [u32]) {
    let mut seed = ((tid + 1) as u64).wrapping_mul(0x9e3779b97f4a7c15);
    for w in words.iter_mut().take(INPUTS) {
        seed = seed.wrapping_mul(0xbf58476d1ce4e5b9).wrapping_add(0x94d049bb133111eb);
        *w = (seed >> 32) as u32;
    }
}

fn user() -> Vec<u32> {
    let mut user = Vec::new();
    put_desc(&mut user, 0, vdesc(IN_BASE, (THREADS as usize * INPUTS * 4) as u32));
    put_desc(&mut user, 4, vdesc(OUT_BASE, (THREADS as usize * RESULTS * 4) as u32));
    user
}

#[test]
fn ds_wrxchg2_b64() {
    run_and_check(&CODE, false);
}

fn run_and_check(code: &[u32], skip_st64: bool) {
    let mut inp = vec![0u32; THREADS as usize * INPUTS];
    for tid in 0..THREADS {
        fill(tid, &mut inp[tid as usize * INPUTS..][..INPUTS]);
    }
    let n_out = THREADS as usize * RESULTS;
    let data = inp.clone();
    let setup = move |m: &mut Machine| {
        m.lds = vec![0; LDS_DWORDS * 4];
        m.memory.write_words(IN_BASE, &data);
        m.memory.write_words(OUT_BASE, &vec![0xdeadbeef; n_out]);
    };
    let out = dispatch(code, 32, THREADS, user(), &setup, &[(OUT_BASE, n_out)]).unwrap().remove(0);
    let names = ["ds_wrxchg2_rtn_b64", "ds_wrxchg2st64_rtn_b64"];
    let mut c = Checker::new();
    for tid in 0..THREADS as usize {
        let i = &inp[tid * INPUTS..][..INPUTS];
        let o = &out[tid * RESULTS..][..RESULTS];
        for op in 0..if skip_st64 { 1 } else { 2 } {
            for k in 0..4 {
                c.eq(|| format!("lane {tid} {} returned dword {k}", names[op]), o[op * 8 + k], i[k]);
                c.eq(|| format!("lane {tid} {} stored dword {k}", names[op]), o[op * 8 + 4 + k], i[4 + k]);
            }
        }
    }
    c.finish();
}

fn reject(code: &[u32], reason: &str) {
    let run = ComputeRun::new(code.to_vec(), 32, THREADS, user());
    for optimize in [false, true] {
        match run.compile(optimize) {
            Ok(_) => panic!("expected a rejection ({reason}), but the program compiled"),
            Err(e) => eprintln!("{reason}: {e}"),
        }
    }
}

#[test]
fn ds_wrxchg2_b64_coinciding_offsets_rejected() {
    reject(&COINCIDING_OFFSETS, "DS write exchange of one location through both offsets is not supported");
    reject(&COINCIDING_OFFSETS_ST64, "DS write exchange of one location through both offsets is not supported");
}

#[test]
fn ds_wrxchg2_b64_destination_overflow_rejected() {
    reject(&DESTINATION_OVERFLOW, "DS write exchange destination register range overflow");
}
