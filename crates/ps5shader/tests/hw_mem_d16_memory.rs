// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! D16 sub-dword memory ops (buffer/global/ds *_d16 and *_d16_hi loads,
//! ds_write_b8_d16_hi, buffer_store_*_d16_hi) against hardware-measured
//! reference behaviour.

include!("vectors/mem_common.rs");
include!("vectors/mem_d16_memory.rs");

const THREADS: u32 = 32;
const INPUT_BYTES: usize = 16;
const RESULTS: usize = 16;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const NAMES: [&str; 12] = [
    "buffer_load_ubyte_d16(_hi)", "buffer_load_sbyte_d16(_hi)", "buffer_load_short_d16(_hi)",
    "global_load_ubyte_d16(_hi)", "global_load_sbyte_d16(_hi)", "global_load_short_d16(_hi)",
    "ds_read_u8_d16(_hi)", "ds_read_i8_d16(_hi)", "ds_read_u16_d16(_hi)", "ds_write_b8_d16_hi",
    "buffer_store_byte_d16_hi", "buffer_store_short_d16_hi",
];

fn input() -> Vec<u8> {
    (0..THREADS as usize * INPUT_BYTES).map(|i| (i as u32 * 37 + 0x81) as u8).collect()
}

fn run() -> Vec<u32> {
    let bytes = input();
    let n_out = THREADS as usize * RESULTS;
    let mut user = Vec::new();
    put_desc(&mut user, 0, vdesc(IN_BASE, bytes.len() as u32));
    put_desc(&mut user, 4, vdesc(OUT_BASE, (n_out * 4) as u32));
    let setup = move |m: &mut Machine| {
        write_bytes(m, IN_BASE, &bytes);
        m.memory.write_words(OUT_BASE, &vec![0xdeadbeef; n_out]);
    };
    dispatch(&CODE, 32, THREADS, user, &setup, &[(OUT_BASE, n_out)]).unwrap().remove(0)
}

fn expected(tid: usize) -> [u32; 12] {
    let all = input();
    let inp = &all[tid * INPUT_BYTES..][..INPUT_BYTES];
    let u8_ = |at: usize| inp[at] as u32;
    let i8_ = |at: usize| (inp[at] as i8 as i16 as u16) as u32;
    let u16_ = |at: usize| u8_(at) | (u8_(at + 1) << 8);
    let loads = [u8_(1) | (u8_(2) << 16), i8_(3) | (i8_(5) << 16), u16_(6) | (u16_(8) << 16)];
    let dword = u32::from_le_bytes([inp[0], inp[1], inp[2], inp[3]]);
    let written = (dword & 0xffff00ff) | (0xc1 << 8);
    [
        loads[0], loads[1], loads[2], loads[0], loads[1], loads[2],
        u8_(1) | (u8_(2) << 16), i8_(3) | (i8_(1) << 16), u16_(2) | (u16_(0) << 16), written,
        0xdeadbec1, 0xdead80c1,
    ]
}

fn check(cols: &[usize]) {
    let out = run();
    let mut c = Checker::new();
    for tid in 0..THREADS as usize {
        let e = expected(tid);
        for &j in cols {
            c.eq(|| format!("lane {tid} {}", NAMES[j]), out[tid * RESULTS + j], e[j]);
        }
    }
    c.finish();
}

#[test]
fn buffer_load_d16() {
    check(&[0, 1, 2]);
}

#[test]
fn global_load_d16() {
    check(&[3, 4, 5]);
}

#[test]
fn ds_read_d16() {
    check(&[6, 7, 8]);
}

#[test]
fn ds_write_b8_d16_hi() {
    check(&[9]);
}

#[test]
fn buffer_store_d16_hi() {
    check(&[10, 11]);
}
