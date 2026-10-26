// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Global/flat atomics across lanes against hardware-measured reference
//! behaviour: 64 invocations, run as wave32 and wave64, on a 64 KiB guest
//! block at s[8:9]:
//!   - global_atomic_add (no GLC) of tid + 1 into each lane's own dword,
//!   - flat_atomic_xor (GLC, 64-bit VADDR) of tid << 8 at +0x200,
//!   - global_atomic_add (GLC, offset 4) of (tid + 1) << 8 into one dword
//!     shared by all lanes at +0x404,
//!   - global_atomic_add_x2 (no GLC) of 0x100 + tid at +0x800,
//!   - global_atomic_or (no GLC) of 0xf0 at +0x600 for lanes below 16
//!     (v_cmpx_gt_u32).

use std::sync::OnceLock;

include!("vectors/mem_common.rs");
include!("vectors/mem_global_common.rs");

const THREADS: u32 = 64;
const OUT_WORDS: usize = (2 * THREADS * 2) as usize;
/// The reference places the special regions after zero table ops.
const SPECIAL: usize = 0;

const GLOBAL_ATOMICS_CODE: [u32; 57] = [
    0x34020083, 0x34040084, 0x340a0083, 0x7e2802ff, 0x05e471e1, 0x7e2a02ff, 0x05e471e1, 0x4a060280,
    0x7e2802ff, 0x05e471e1, 0x4a140081, 0xdcc88000, 0x00080a03, 0xbf8c3f70, 0xd70f6a1e, 0x02020608,
    0x7e3e0209, 0x503e3e80, 0xd70f6a1e, 0x02023cff, 0x00000200, 0x503e3e80, 0x34140088, 0xdced0000,
    0x147d0a1e, 0xbf8c0070, 0x4a0c0a80, 0xe0701000, 0x80011406, 0x7e0602ff, 0x00000400, 0x4a1414ff,
    0x00000100, 0xdcc98004, 0x14080a03, 0xbf8c3f70, 0x4a0c0aff, 0x00000200, 0xe0701000, 0x80011406,
    0x4a0602ff, 0x00000800, 0x4a1400ff, 0x00000100, 0x7e160280, 0xdd488000, 0x00080a03, 0xbf8c3f70,
    0x7da80090, 0x4a0602ff, 0x00000600, 0x7e1402ff, 0x000000f0, 0xdce88000, 0x00080a03, 0xbf8c3f70,
    0xbf810000,
];

fn initial() -> Vec<u8> {
    let mut image = vec![BLOCK_FILL; BLOCK_BYTES];
    let t = THREADS as usize;
    for tid in 0..t {
        put_le(&mut image, SPECIAL + tid * 8, 0x1000 * tid as u64, 4);
        put_le(&mut image, SPECIAL + t * 8 + tid * 8, (0xa5a5a5a5u32 ^ tid as u32) as u64, 4);
        put_le(&mut image, SPECIAL + t * 24 + tid * 8, 0x0f0f0f00 + tid as u64, 4);
        put_le(&mut image, SPECIAL + t * 32 + tid * 8, 0x7fffffffffffff00 + tid as u64, 8);
    }
    put_le(&mut image, SPECIAL + t * 16 + 4, 0x11, 4);
    image
}

fn expected() -> Vec<u8> {
    let mut image = initial();
    let t = THREADS as usize;
    let mut total = 0x11u32;
    for tid in 0..t {
        let id = tid as u32;
        put_le(&mut image, SPECIAL + tid * 8, (0x1000 * id + id + 1) as u64, 4);
        put_le(&mut image, SPECIAL + t * 8 + tid * 8, ((0xa5a5a5a5u32 ^ id) ^ (id << 8)) as u64, 4);
        if tid < 16 {
            put_le(&mut image, SPECIAL + t * 24 + tid * 8, ((0x0f0f0f00 + id) | 0xf0) as u64, 4);
        }
        put_le(&mut image, SPECIAL + t * 32 + tid * 8, 0x7fffffffffffff00u64 + tid as u64 + 0x100 + tid as u64, 8);
        total = total.wrapping_add((id + 1) << 8);
    }
    put_le(&mut image, SPECIAL + t * 16 + 4, total as u64, 4);
    image
}

struct LanesRun {
    output: Vec<u32>,
    guest: Vec<u32>,
}

fn result(wave: u32) -> &'static Result<LanesRun, String> {
    static RUN32: OnceLock<Result<LanesRun, String>> = OnceLock::new();
    static RUN64: OnceLock<Result<LanesRun, String>> = OnceLock::new();
    let cell = if wave == 32 { &RUN32 } else { &RUN64 };
    cell.get_or_init(|| {
        // The input buffer has no table rows (zero bytes).
        let user = global_user_data(0, (OUT_WORDS * 4) as u32);
        let regions = vec![(GUEST_BASE, bytes_to_words(&initial())), (GLOBAL_OUT_BASE, vec![TABLE_SENTINEL; OUT_WORDS])];
        let mut r = run_waves(&GLOBAL_ATOMICS_CODE, wave, THREADS, user, &regions)?;
        let output = r.remove(1);
        let guest = r.remove(0);
        Ok(LanesRun { output, guest })
    })
}

fn run(wave: u32) -> &'static LanesRun {
    result(wave).as_ref().unwrap_or_else(|e| panic!("{e}"))
}

fn check_flat_xor_returned(wave: u32) {
    let r = run(wave);
    let mut c = Checker::new();
    for tid in 0..THREADS {
        c.eq(|| format!("wave{wave} flat xor returned to lane {tid}"), r.output[(tid * 2) as usize], 0xa5a5a5a5 ^ tid);
    }
    c.finish();
}

fn check_contended_add(wave: u32) {
    let r = run(wave);
    let mut order: Vec<(u32, u32)> = (0..THREADS).map(|tid| (r.output[((THREADS + tid) * 2) as usize], tid)).collect();
    order.sort();
    let mut c = Checker::new();
    let mut running = 0x11u32;
    for (returned, tid) in order {
        c.eq(|| format!("wave{wave} contended add returned to lane {tid}"), returned, running);
        running = running.wrapping_add((tid + 1) << 8);
    }
    c.finish();
}

fn check_block(wave: u32) {
    let r = run(wave);
    let e = expected();
    let mut c = Checker::new();
    for (k, b) in e.iter().enumerate() {
        c.eq(|| format!("wave{wave} guest byte {k:#x}"), get_le(&r.guest, k, 1) as u32, *b as u32);
    }
    c.finish();
}

#[test]
fn flat_xor_returned_wave32() {
    check_flat_xor_returned(32);
}
#[test]
fn flat_xor_returned_wave64() {
    check_flat_xor_returned(64);
}
#[test]
fn contended_add_wave32() {
    check_contended_add(32);
}
#[test]
fn contended_add_wave64() {
    check_contended_add(64);
}
#[test]
fn memory_wave32() {
    check_block(32);
}
#[test]
fn memory_wave64() {
    check_block(64);
}
