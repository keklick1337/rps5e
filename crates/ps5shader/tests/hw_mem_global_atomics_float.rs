// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Global/flat float atomics (fcmpswap, fmin, fmax, their _x2 forms) plus
//! inc_x2/dec_x2 against hardware-measured reference rows. 64
//! invocations, run as wave32 and wave64. Each operation has its own
//! 0x200-byte region of a guest block; the global forms use SADDR
//! s[8:9] + VADDR, the flat forms a 64-bit VADDR built with
//! v_add_co_u32/v_addc_u32. Every atomic returns the pre-op memory value.

use std::sync::OnceLock;

include!("vectors/mem_common.rs");
include!("vectors/mem_global_common.rs");
include!("vectors/mem_global_atomics_float.rs");

const THREADS: u32 = 64;
const OP_COUNT: u32 = 8;
const NARROW_OPS: u32 = 3;
const OUT_WORDS: usize = (OP_COUNT * THREADS * 2) as usize;

const GLOBAL_ATOMICS_FLOAT_CODE: [u32; 129] = [
    0x34020083, 0x34040084, 0x340a0083, 0x7e2802ff, 0x05e471e1, 0x7e2a02ff, 0x05e471e1, 0x4a060280,
    0x4a080480, 0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0a80, 0xdcf98000, 0x14080a03, 0xbf8c0070,
    0xe0701000, 0x80011406, 0x4a0602ff, 0x00000200, 0x4a0804ff, 0x00000400, 0xe0381000, 0x80000a04,
    0xbf8c3f70, 0x4a0c0aff, 0x00000200, 0xd70f6a1e, 0x02020608, 0x7e3e0209, 0x503e3e80, 0xdcfd0000,
    0x147d0a1e, 0xbf8c0070, 0xe0701000, 0x80011406, 0x4a0602ff, 0x00000400, 0x4a0804ff, 0x00000800,
    0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff, 0x00000400, 0xdd018000, 0x14080a03, 0xbf8c0070,
    0xe0701000, 0x80011406, 0x4a0602ff, 0x00000600, 0x4a0804ff, 0x00000c00, 0xe0381000, 0x80000a04,
    0xbf8c3f70, 0x4a0c0aff, 0x00000600, 0xdd798000, 0x14080a03, 0xbf8c0070, 0xe0741000, 0x80011406,
    0x4a0602ff, 0x00000800, 0x4a0804ff, 0x00001000, 0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff,
    0x00000800, 0xdd7d8000, 0x14080a03, 0xbf8c0070, 0xe0741000, 0x80011406, 0x4a0602ff, 0x00000a00,
    0x4a0804ff, 0x00001400, 0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff, 0x00000a00, 0xd70f6a1e,
    0x02020608, 0x7e3e0209, 0x503e3e80, 0xdd810000, 0x147d0a1e, 0xbf8c0070, 0xe0741000, 0x80011406,
    0x4a0602ff, 0x00000c00, 0x4a0804ff, 0x00001800, 0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff,
    0x00000c00, 0xdd718000, 0x14080a03, 0xbf8c0070, 0xe0741000, 0x80011406, 0x4a0602ff, 0x00000e00,
    0x4a0804ff, 0x00001c00, 0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff, 0x00000e00, 0xd70f6a1e,
    0x02020608, 0x7e3e0209, 0x503e3e80, 0xdd750000, 0x147d0a1e, 0xbf8c0070, 0xe0741000, 0x80011406,
    0xbf810000,
];

const OP_NAMES: [&str; 8] = [
    "global_atomic_fcmpswap", "flat_atomic_fmin", "global_atomic_fmax", "global_atomic_fcmpswap_x2",
    "global_atomic_fmin_x2", "flat_atomic_fmax_x2", "global_atomic_inc_x2", "flat_atomic_dec_x2",
];

/// `[op, memory, data, comparator, final]`.
fn row_of(op: u32, tid: u32) -> &'static [u64; 5] {
    &ROWS[(op * THREADS + tid) as usize]
}

fn op_bytes(op: u32) -> usize {
    if op < NARROW_OPS { 4 } else { 8 }
}

fn result(wave: u32) -> &'static Result<TableRun, String> {
    static RUN32: OnceLock<Result<TableRun, String>> = OnceLock::new();
    static RUN64: OnceLock<Result<TableRun, String>> = OnceLock::new();
    let cell = if wave == 32 { &RUN32 } else { &RUN64 };
    cell.get_or_init(|| {
        run_table(&GLOBAL_ATOMICS_FLOAT_CODE, wave, THREADS, OP_COUNT, NARROW_OPS, OUT_WORDS, &|op, tid| {
            let r = row_of(op, tid);
            [r[1], r[2], r[3]]
        })
    })
}

/// Signaling NaN of the operation's width.
fn is_snan(op: u32, v: u64) -> bool {
    if op_bytes(op) == 4 {
        let v = v as u32;
        v & 0x7fff_ffff > 0x7f80_0000 && v & 0x0040_0000 == 0
    } else {
        v & 0x7fff_ffff_ffff_ffff > 0x7ff0_0000_0000_0000 && v & 0x0008_0000_0000_0000 == 0
    }
}

/// Rows of `op` whose memory or data operand is a signaling NaN.
fn snan_row(op: u32, tid: u32) -> bool {
    let r = row_of(op, tid);
    is_snan(op, r[1]) || is_snan(op, r[2])
}

#[derive(Clone, Copy, PartialEq)]
enum Rows {
    All,
    /// Rows without a signaling-NaN memory/data operand.
    NoSnan,
    /// Only rows with a signaling-NaN memory/data operand.
    Snan,
}

fn check_op(wave: u32, op: u32) {
    check_rows(wave, op, Rows::All);
}

fn check_rows(wave: u32, op: u32, rows: Rows) {
    let run = result(wave).as_ref().unwrap_or_else(|e| panic!("{e}"));
    let mut c = Checker::new();
    let name = OP_NAMES[op as usize];
    for tid in 0..THREADS {
        let snan = snan_row(op, tid);
        if (rows == Rows::NoSnan && snan) || (rows == Rows::Snan && !snan) {
            continue;
        }
        let r = row_of(op, tid);
        let w = &run.output[((op * THREADS + tid) * 2) as usize..];
        let returned = if op_bytes(op) == 4 { w[0] as u64 } else { w[0] as u64 | ((w[1] as u64) << 32) };
        let what = |s: &str| format!("wave{wave} {name} lane {tid} (memory {:#x} data {:#x} comparator {:#x}) {s}", r[1], r[2], r[3]);
        c.eq(|| what("returned lo"), returned as u32, r[1] as u32);
        c.eq(|| what("returned hi"), (returned >> 32) as u32, (r[1] >> 32) as u32);
        let actual = get_le(&run.guest, op as usize * OP_STRIDE + tid as usize * 8, op_bytes(op));
        c.eq(|| what("final lo"), actual as u32, r[4] as u32);
        c.eq(|| what("final hi"), (actual >> 32) as u32, (r[4] >> 32) as u32);
    }
    c.finish();
}

fn check_untouched(wave: u32) {
    let run = result(wave).as_ref().unwrap_or_else(|e| panic!("{e}"));
    // The reference compares the whole block against the initial image
    // with every slot replaced by its final value; the slots are checked
    // per operation, so this covers every other byte.
    let mut c = Checker::new();
    for (k, e) in run.initial.iter().enumerate() {
        let (op, within) = ((k / OP_STRIDE) as u32, k % OP_STRIDE);
        if op < OP_COUNT && within % 8 < op_bytes(op) {
            continue;
        }
        c.eq(|| format!("wave{wave} guest byte {k:#x}"), get_le(&run.guest, k, 1) as u32, *e as u32);
    }
    c.finish();
}

#[test]
fn untouched_bytes_wave32() {
    check_untouched(32);
}
#[test]
fn untouched_bytes_wave64() {
    check_untouched(64);
}

#[test]
fn fcmpswap_wave32() {
    check_op(32, 0);
}
#[test]
fn fcmpswap_wave64() {
    check_op(64, 0);
}
#[test]
fn fcmpswap_x2_wave32() {
    check_op(32, 3);
}
#[test]
fn fcmpswap_x2_wave64() {
    check_op(64, 3);
}

/// Float min/max: the rows without a signaling NaN and the signaling-NaN
/// rows (hardware returns the quieted sNaN operand) are separate tests.
macro_rules! min_max_tests {
    ($($name:ident, $snan:ident, $wave:expr, $op:expr;)*) => {$(
        #[test]
        fn $name() {
            check_rows($wave, $op, Rows::NoSnan);
        }
        #[test]
        fn $snan() {
            check_rows($wave, $op, Rows::Snan);
        }
    )*};
}

min_max_tests! {
    flat_fmin_wave32, flat_fmin_snan_wave32, 32, 1;
    flat_fmin_wave64, flat_fmin_snan_wave64, 64, 1;
    fmax_wave32, fmax_snan_wave32, 32, 2;
    fmax_wave64, fmax_snan_wave64, 64, 2;
    fmin_x2_wave32, fmin_x2_snan_wave32, 32, 4;
    fmin_x2_wave64, fmin_x2_snan_wave64, 64, 4;
    flat_fmax_x2_wave32, flat_fmax_x2_snan_wave32, 32, 5;
    flat_fmax_x2_wave64, flat_fmax_x2_snan_wave64, 64, 5;
}

#[test]
fn inc_x2_wave32() {
    check_op(32, 6);
}
#[test]
fn inc_x2_wave64() {
    check_op(64, 6);
}
#[test]
fn flat_dec_x2_wave32() {
    check_op(32, 7);
}
#[test]
fn flat_dec_x2_wave64() {
    check_op(64, 7);
}
