// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! 64-bit global atomics (swap_x2 .. xor_x2, all with GLC) against
//! hardware-measured reference rows. One wave32 dispatch of 64
//! invocations; each operation has its own 0x200-byte region of a guest
//! block addressed through s[8:9] (SADDR) + a per-lane VADDR offset.

use std::sync::OnceLock;

include!("vectors/mem_common.rs");
include!("vectors/mem_global_common.rs");
include!("vectors/mem_global_atomics64.rs");

const THREADS: u32 = 64;
const ROWS_PER_OP: u32 = 16;
const OP_COUNT: u32 = 11;
const NARROW_OPS: u32 = 0;
const OUT_WORDS: usize = ((OP_COUNT + 2) * THREADS * 2) as usize;

const GLOBAL_ATOMICS_CODE: [u32; 159] = [
    0x34020083, 0x34040084, 0x340a0083, 0x7e2802ff, 0x05e471e1, 0x7e2a02ff, 0x05e471e1, 0x4a060280,
    0x4a080480, 0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0a80, 0xdd418000, 0x14080a03, 0xbf8c3f70,
    0xe0741000, 0x80011406, 0x4a0602ff, 0x00000200, 0x4a0804ff, 0x00000400, 0xe0381000, 0x80000a04,
    0xbf8c3f70, 0x4a0c0aff, 0x00000200, 0xdd458000, 0x14080a03, 0xbf8c3f70, 0xe0741000, 0x80011406,
    0x4a0602ff, 0x00000400, 0x4a0804ff, 0x00000800, 0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff,
    0x00000400, 0xdd498000, 0x14080a03, 0xbf8c3f70, 0xe0741000, 0x80011406, 0x4a0602ff, 0x00000600,
    0x4a0804ff, 0x00000c00, 0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff, 0x00000600, 0xdd4d8000,
    0x14080a03, 0xbf8c3f70, 0xe0741000, 0x80011406, 0x4a0602ff, 0x00000800, 0x4a0804ff, 0x00001000,
    0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff, 0x00000800, 0xdd558000, 0x14080a03, 0xbf8c3f70,
    0xe0741000, 0x80011406, 0x4a0602ff, 0x00000a00, 0x4a0804ff, 0x00001400, 0xe0381000, 0x80000a04,
    0xbf8c3f70, 0x4a0c0aff, 0x00000a00, 0xdd598000, 0x14080a03, 0xbf8c3f70, 0xe0741000, 0x80011406,
    0x4a0602ff, 0x00000c00, 0x4a0804ff, 0x00001800, 0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff,
    0x00000c00, 0xdd5d8000, 0x14080a03, 0xbf8c3f70, 0xe0741000, 0x80011406, 0x4a0602ff, 0x00000e00,
    0x4a0804ff, 0x00001c00, 0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff, 0x00000e00, 0xdd618000,
    0x14080a03, 0xbf8c3f70, 0xe0741000, 0x80011406, 0x4a0602ff, 0x00001000, 0x4a0804ff, 0x00002000,
    0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff, 0x00001000, 0xdd658000, 0x14080a03, 0xbf8c3f70,
    0xe0741000, 0x80011406, 0x4a0602ff, 0x00001200, 0x4a0804ff, 0x00002400, 0xe0381000, 0x80000a04,
    0xbf8c3f70, 0x4a0c0aff, 0x00001200, 0xdd698000, 0x14080a03, 0xbf8c3f70, 0xe0741000, 0x80011406,
    0x4a0602ff, 0x00001400, 0x4a0804ff, 0x00002800, 0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff,
    0x00001400, 0xdd6d8000, 0x14080a03, 0xbf8c3f70, 0xe0741000, 0x80011406, 0xbf810000,
];

const OP_NAMES: [&str; 11] = [
    "global_atomic_swap_x2", "global_atomic_cmpswap_x2", "global_atomic_add_x2", "global_atomic_sub_x2",
    "global_atomic_smin_x2", "global_atomic_umin_x2", "global_atomic_smax_x2", "global_atomic_umax_x2",
    "global_atomic_and_x2", "global_atomic_or_x2", "global_atomic_xor_x2",
];

fn row_of(op: u32, tid: u32) -> &'static [u64; 6] {
    &ROWS[(op * ROWS_PER_OP + tid % ROWS_PER_OP) as usize]
}

fn op_bytes(op: u32) -> usize {
    if op < NARROW_OPS { 4 } else { 8 }
}

fn result() -> &'static Result<TableRun, String> {
    static RUN: OnceLock<Result<TableRun, String>> = OnceLock::new();
    RUN.get_or_init(|| {
        run_table(&GLOBAL_ATOMICS_CODE, 32, THREADS, OP_COUNT, NARROW_OPS, OUT_WORDS, &|op, tid| {
            let r = row_of(op, tid);
            [r[1], r[2], r[3]]
        })
    })
}

fn check_op(op: u32) {
    let run = result().as_ref().unwrap_or_else(|e| panic!("{e}"));
    let mut c = Checker::new();
    let name = OP_NAMES[op as usize];
    for tid in 0..THREADS {
        let r = row_of(op, tid);
        let w = &run.output[((op * THREADS + tid) * 2) as usize..];
        let returned = if op_bytes(op) == 4 { w[0] as u64 } else { w[0] as u64 | ((w[1] as u64) << 32) };
        let what = |s: &str| format!("{name} lane {tid} (memory {:#x} data {:#x} comparator {:#x}) {s}", r[1], r[2], r[3]);
        c.eq(|| what("returned lo"), returned as u32, r[4] as u32);
        c.eq(|| what("returned hi"), (returned >> 32) as u32, (r[4] >> 32) as u32);
        let actual = get_le(&run.guest, op as usize * OP_STRIDE + tid as usize * 8, op_bytes(op));
        c.eq(|| what("final lo"), actual as u32, r[5] as u32);
        c.eq(|| what("final hi"), (actual >> 32) as u32, (r[5] >> 32) as u32);
    }
    c.finish();
}

#[test]
fn untouched_bytes() {
    let run = result().as_ref().unwrap_or_else(|e| panic!("{e}"));
    let mut expected = run.initial.clone();
    for op in 0..OP_COUNT {
        for tid in 0..THREADS {
            put_le(&mut expected, op as usize * OP_STRIDE + tid as usize * 8, row_of(op, tid)[5], op_bytes(op));
        }
    }
    let mut c = Checker::new();
    for (k, e) in expected.iter().enumerate() {
        c.eq(|| format!("guest byte {k:#x}"), get_le(&run.guest, k, 1) as u32, *e as u32);
    }
    c.finish();
}

#[test]
fn swap_x2() {
    check_op(0);
}
#[test]
fn cmpswap_x2() {
    check_op(1);
}
#[test]
fn add_x2() {
    check_op(2);
}
#[test]
fn sub_x2() {
    check_op(3);
}
#[test]
fn smin_x2() {
    check_op(4);
}
#[test]
fn umin_x2() {
    check_op(5);
}
#[test]
fn smax_x2() {
    check_op(6);
}
#[test]
fn umax_x2() {
    check_op(7);
}
#[test]
fn and_x2() {
    check_op(8);
}
#[test]
fn or_x2() {
    check_op(9);
}
#[test]
fn xor_x2() {
    check_op(10);
}
