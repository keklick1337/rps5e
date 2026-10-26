// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! buffer_atomic_csub (clamped subtract) against hardware-measured
//! reference behaviour: per-lane edge rows, a partial EXEC mask, and all
//! lanes of the workgroup subtracting from one shared dword (one outcome
//! stays positive, the other clamps at zero).
//!
//! A workgroup larger than the wave runs as consecutive waves over the
//! same memory (the shared-dword results do not depend on wave order).

include!("vectors/mem_common.rs");

use ps5shader::interp::Memory;
use ps5shader::ssa::SystemValue;

const MAX_THREADS: u32 = 64;
const INPUTS: usize = 4;
const RESULTS: usize = 8;
const WRITTEN: usize = 5;
const SHARED_WORDS: usize = 2;
const UNTOUCHED: u32 = 0xdeadbeef;
const MASKED_EXEC_LO: u32 = 0x5555_5555;
const SHARED_PARTIAL: u32 = 0x8001_86a0;
const SHARED_EXHAUSTED: u32 = 1000;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const CODE: [u32; 37] = [
    0x34020084, 0x34040085, 0xe0301000, 0x80000401, 0xe0301004, 0x80000501, 0xe0301008, 0x80000601,
    0xbf8c3f70, 0x7e140280, 0xd765000a, 0x000214c1, 0xd766000a, 0x000214c1, 0x7e0e0304, 0xe0d05000,
    0x80010702, 0x7e100304, 0xbefe03ff, 0x55555555, 0xe0d05004, 0x80010802, 0xbefe03c1, 0x7e120305,
    0xe0d04800, 0x80010900, 0x7e160306, 0xe0d04804, 0x80010b00, 0xbf8c3f70, 0xe0701008, 0x80010702,
    0xe070100c, 0x80010802, 0xe0701010, 0x80010a02, 0xbf810000,
];

/// (memory, subtrahend)
const EDGES: [(u32, u32); 16] = [
    (0, 0), (0, 1), (1, 0), (1, 1),
    (5, 7), (7, 5), (0xffffffff, 1), (1, 0xffffffff),
    (0x80000000, 0x7fffffff), (0x7fffffff, 0x80000000), (0xffffffff, 0xffffffff), (0x80000000, 0x80000000),
    (0x80000001, 0x80000000), (0, 0xffffffff), (0xffffffff, 0), (0x12345678, 0x12345679),
];

fn row_of(tid: u32) -> (u32, u32) {
    if (tid as usize) < EDGES.len() {
        return EDGES[tid as usize];
    }
    let memory = (tid + 1).wrapping_mul(0x9e3779b9);
    let subtrahend = (tid + 1).wrapping_mul(0x85ebca6b);
    (memory, if tid & 3 == 0 { subtrahend >> 8 } else { subtrahend })
}

fn shared_partial_step(tid: u32) -> u32 {
    tid + 1
}

fn shared_exhausted_step(tid: u32) -> u32 {
    tid + 100
}

fn subtract_clamped(memory: u32, subtrahend: u32) -> u32 {
    memory.saturating_sub(subtrahend)
}

fn user_data() -> Vec<u32> {
    let n_in = MAX_THREADS as usize * INPUTS;
    let n_out = MAX_THREADS as usize * RESULTS + SHARED_WORDS;
    let mut user = Vec::new();
    put_desc(&mut user, 0, vdesc(IN_BASE, (n_in * 4) as u32));
    put_desc(&mut user, 4, vdesc(OUT_BASE, (n_out * 4) as u32));
    user
}

/// Run `threads` invocations as consecutive waves of `wave` lanes over one
/// memory; returns the output words.
fn run(wave: u32, threads: u32) -> Vec<u32> {
    let mut input = vec![0u32; MAX_THREADS as usize * INPUTS];
    let mut output = vec![UNTOUCHED; MAX_THREADS as usize * RESULTS + SHARED_WORDS];
    for tid in 0..MAX_THREADS {
        let (memory, subtrahend) = row_of(tid);
        let t = tid as usize;
        input[t * INPUTS] = subtrahend;
        input[t * INPUTS + 1] = shared_partial_step(tid);
        input[t * INPUTS + 2] = shared_exhausted_step(tid);
        output[t * RESULTS] = memory;
        output[t * RESULTS + 1] = memory;
    }
    let n = output.len();
    output[n - 2] = SHARED_PARTIAL;
    output[n - 1] = SHARED_EXHAUSTED;
    let mut memory: Option<Memory> = None;
    for w in 0..threads.div_ceil(wave) {
        let first = w * wave;
        let lanes = wave.min(threads - first);
        let prev = memory.clone();
        let (inp, out) = (input.clone(), output.clone());
        let setup = move |m: &mut Machine| {
            match &prev {
                Some(p) => m.memory = p.clone(),
                None => {
                    m.memory.write_words(IN_BASE, &inp);
                    m.memory.write_words(OUT_BASE, &out);
                }
            }
            m.system.insert(SystemValue::LocalInvocationId(0), (first..first + lanes).collect());
        };
        let m = ComputeRun::new(CODE.to_vec(), wave, lanes, user_data()).run_checked(&setup, &[(OUT_BASE, n)]).unwrap();
        memory = Some(m.memory);
    }
    memory.unwrap().read_words(OUT_BASE, n)
}

fn check(wave: u32, threads: u32) {
    let out = run(wave, threads);
    let name = format!("wave{wave} {threads} threads");
    let names = ["memory", "memory under a partial exec mask", "returned value", "data register under a partial exec mask"];
    let mut c = Checker::new();
    let (mut partial_total, mut exhausted_total) = (0u32, 0u32);
    for tid in 0..MAX_THREADS {
        let (memory, subtrahend) = row_of(tid);
        let o = &out[tid as usize * RESULTS..][..RESULTS];
        if tid >= threads {
            c.eq(|| format!("{name} thread {tid} (outside) memory"), o[0], memory);
            c.eq(|| format!("{name} thread {tid} (outside) masked memory"), o[1], memory);
            for (j, &x) in o.iter().enumerate().skip(2) {
                c.eq(|| format!("{name} thread {tid} (outside) result {j}"), x, UNTOUCHED);
            }
            continue;
        }
        partial_total += shared_partial_step(tid);
        exhausted_total += shared_exhausted_step(tid);
        let lane = o[4];
        c.eq(|| format!("{name} thread {tid} lane id"), lane, tid % wave);
        let active = lane >= 32 || (MASKED_EXEC_LO >> (lane & 31)) & 1 != 0;
        let clamped = subtract_clamped(memory, subtrahend);
        let expected = [clamped, if active { clamped } else { memory }, memory, if active { memory } else { subtrahend }];
        for (j, &e) in expected.iter().enumerate() {
            c.eq(|| format!("{name} thread {tid} lane {lane} {} (memory {memory:#x}, subtrahend {subtrahend:#x})", names[j]), o[j], e);
        }
        for (j, &x) in o.iter().enumerate().skip(WRITTEN) {
            c.eq(|| format!("{name} thread {tid} result {j} was written"), x, UNTOUCHED);
        }
    }
    let shared = [subtract_clamped(SHARED_PARTIAL, partial_total), subtract_clamped(SHARED_EXHAUSTED, exhausted_total)];
    assert!(shared[0] != 0 && shared[1] == 0, "{name} does not cover both shared outcomes");
    let base = MAX_THREADS as usize * RESULTS;
    c.eq(|| format!("{name} dword shared by all lanes"), out[base], shared[0]);
    c.eq(|| format!("{name} dword exhausted by all lanes"), out[base + 1], shared[1]);
    c.finish();
}

#[test]
fn wave32_32_threads() {
    check(32, 32);
}

#[test]
fn wave32_64_threads() {
    check(32, 64);
}

#[test]
fn wave64_64_threads() {
    check(64, 64);
}
