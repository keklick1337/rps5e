// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! global_atomic_csub (clamped subtract, GLC) against hardware-measured
//! reference behaviour: 64 invocations, run as wave32 and wave64. Each
//! lane subtracts from its own dword, and all lanes subtract from two
//! shared dwords (one that stays positive, one that clamps at zero).

use std::sync::OnceLock;

include!("vectors/mem_common.rs");
include!("vectors/mem_global_common.rs");

const THREADS: u32 = 64;
const INPUTS: usize = 4;
const RESULTS: usize = 4;
const WRITTEN: usize = 3;
const SHARED_OFFSET: usize = THREADS as usize * 4;
const UNTOUCHED: u32 = 0xdead_beef;
const SHARED_PARTIAL: u32 = 0x8001_86a0;
const SHARED_EXHAUSTED: u32 = 1000;
/// The reference block is 4 KiB (4 KiB aligned).
const CSUB_BLOCK_BYTES: usize = 4096;

const CODE: [u32; 25] = [
    0x34020084, 0x34060082, 0xe0301000, 0x80000a01, 0xe0301004, 0x80000b01, 0xe0301008, 0x80000c01,
    0x7e0e02ff, 0x00000100, 0xbf8c3f70, 0xdcd18000, 0x14080a03, 0xdcd18000, 0x15080b07, 0xdcd18004,
    0x16080c07, 0xbf8c3f70, 0xe0701000, 0x80011401, 0xe0701004, 0x80011501, 0xe0701008, 0x80011601,
    0xbf810000,
];

/// flat_atomic_csub (FLAT segment): the operation exists only in the
/// global segment.
const FLAT_SEGMENT_CODE: [u32; 3] = [0xdcd10000, 0x147f0a03, 0xbf810000];

const EDGES: [(u32, u32); 16] = [
    (0, 0), (0, 1), (1, 0), (1, 1),
    (5, 7), (7, 5), (0xffffffff, 1), (1, 0xffffffff),
    (0x80000000, 0x7fffffff), (0x7fffffff, 0x80000000), (0xffffffff, 0xffffffff), (0x80000000, 0x80000000),
    (0x80000001, 0x80000000), (0, 0xffffffff), (0xffffffff, 0), (0x12345678, 0x12345679),
];

/// `(memory, subtrahend)` of lane `tid`.
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
    if memory >= subtrahend { memory - subtrahend } else { 0 }
}

fn initial() -> Vec<u8> {
    let mut image = vec![BLOCK_FILL; CSUB_BLOCK_BYTES];
    for tid in 0..THREADS {
        put_le(&mut image, tid as usize * 4, row_of(tid).0 as u64, 4);
    }
    put_le(&mut image, SHARED_OFFSET, SHARED_PARTIAL as u64, 4);
    put_le(&mut image, SHARED_OFFSET + 4, SHARED_EXHAUSTED as u64, 4);
    image
}

fn expected() -> Vec<u8> {
    let mut image = initial();
    let mut partial = SHARED_PARTIAL;
    let mut exhausted = SHARED_EXHAUSTED;
    for tid in 0..THREADS {
        let (memory, subtrahend) = row_of(tid);
        put_le(&mut image, tid as usize * 4, subtract_clamped(memory, subtrahend) as u64, 4);
        partial = subtract_clamped(partial, shared_partial_step(tid));
        exhausted = subtract_clamped(exhausted, shared_exhausted_step(tid));
    }
    assert!(partial != 0 && exhausted == 0, "the shared dwords do not cover both outcomes");
    put_le(&mut image, SHARED_OFFSET, partial as u64, 4);
    put_le(&mut image, SHARED_OFFSET + 4, exhausted as u64, 4);
    image
}

struct CsubRun {
    output: Vec<u32>,
    guest: Vec<u32>,
}

fn result(wave: u32) -> &'static Result<CsubRun, String> {
    static RUN32: OnceLock<Result<CsubRun, String>> = OnceLock::new();
    static RUN64: OnceLock<Result<CsubRun, String>> = OnceLock::new();
    let cell = if wave == 32 { &RUN32 } else { &RUN64 };
    cell.get_or_init(|| {
        let mut input = vec![0u32; THREADS as usize * INPUTS];
        for tid in 0..THREADS as usize {
            input[tid * INPUTS] = row_of(tid as u32).1;
            input[tid * INPUTS + 1] = shared_partial_step(tid as u32);
            input[tid * INPUTS + 2] = shared_exhausted_step(tid as u32);
        }
        let out_words = THREADS as usize * RESULTS;
        let user = global_user_data((input.len() * 4) as u32, (out_words * 4) as u32);
        let regions = vec![(GUEST_BASE, bytes_to_words(&initial())), (GLOBAL_IN_BASE, input), (GLOBAL_OUT_BASE, vec![UNTOUCHED; out_words])];
        let mut r = run_waves(&CODE, wave, THREADS, user, &regions)?;
        let output = r.remove(2);
        let guest = r.remove(0);
        Ok(CsubRun { output, guest })
    })
}

fn run(wave: u32) -> &'static CsubRun {
    result(wave).as_ref().unwrap_or_else(|e| panic!("{e}"))
}

fn check_returned(wave: u32) {
    let r = run(wave);
    let mut c = Checker::new();
    for tid in 0..THREADS as usize {
        c.eq(|| format!("wave{wave} lane {tid} returned"), r.output[tid * RESULTS], row_of(tid as u32).0);
        for j in WRITTEN..RESULTS {
            c.eq(|| format!("wave{wave} lane {tid} result {j} written"), r.output[tid * RESULTS + j], UNTOUCHED);
        }
    }
    c.finish();
}

/// The values returned from a shared dword, sorted descending, must form
/// the chain of clamped subtractions in that order.
fn check_chain(wave: u32, column: usize, initial: u32, step: fn(u32) -> u32) {
    let r = run(wave);
    let mut order: Vec<(u32, u32)> = (0..THREADS).map(|tid| (r.output[tid as usize * RESULTS + column], tid)).collect();
    order.sort_by(|a, b| b.0.cmp(&a.0));
    let mut c = Checker::new();
    let mut running = initial;
    for (returned, tid) in order {
        c.eq(|| format!("wave{wave} column {column} returned to lane {tid}"), returned, running);
        running = subtract_clamped(running, step(tid));
    }
    c.finish();
}

fn check_block(wave: u32) {
    let r = run(wave);
    let e = expected();
    let mut c = Checker::new();
    for (k, b) in e.iter().enumerate() {
        c.eq(|| format!("wave{wave} guest byte {k}"), get_le(&r.guest, k, 1) as u32, *b as u32);
    }
    c.finish();
}

#[test]
fn returned_wave32() {
    check_returned(32);
}
#[test]
fn returned_wave64() {
    check_returned(64);
}
#[test]
fn shared_partial_chain_wave32() {
    check_chain(32, 1, SHARED_PARTIAL, shared_partial_step);
}
#[test]
fn shared_partial_chain_wave64() {
    check_chain(64, 1, SHARED_PARTIAL, shared_partial_step);
}
#[test]
fn shared_exhausted_chain_wave32() {
    check_chain(32, 2, SHARED_EXHAUSTED, shared_exhausted_step);
}
#[test]
fn shared_exhausted_chain_wave64() {
    check_chain(64, 2, SHARED_EXHAUSTED, shared_exhausted_step);
}
#[test]
fn memory_wave32() {
    check_block(32);
}
#[test]
fn memory_wave64() {
    check_block(64);
}

/// flat_atomic_csub is not an operation of the FLAT segment; the
/// compiler must reject it rather than translate it.
#[test]
fn flat_segment_rejected() {
    let user = global_user_data(THREADS * INPUTS as u32 * 4, THREADS * RESULTS as u32 * 4);
    let r = ComputeRun::new(FLAT_SEGMENT_CODE.to_vec(), 32, THREADS, user).compile(true);
    assert!(r.is_err(), "flat_atomic_csub compiled");
}
