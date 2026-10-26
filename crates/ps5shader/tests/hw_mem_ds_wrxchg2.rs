// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! ds_wrxchg2_rtn_b32 / ds_wrxchg2st64_rtn_b32 (plain, aliased, exec-masked)
//! against hardware-measured reference rows, plus the forms the reference
//! compiler rejects.

include!("vectors/mem_common.rs");

const THREADS: u32 = 32;
const INPUTS: usize = 4;
const RESULTS: usize = 16;
const ACTIVE_LANES: u32 = 16;
const LDS_DWORDS: usize = 256;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const CODE: [u32; 94] = [
    0x34020082, 0x34060084, 0x343c0082, 0xe0302000, 0x80000401, 0xe0302004, 0x80000501, 0xe0302008,
    0x80000601, 0xe030200c, 0x80000701, 0xbf8c3f70, 0xd8340000, 0x0000041e, 0xd8340080, 0x0000051e,
    0xd8340100, 0x0000051e, 0xd8340300, 0x0000041e, 0xd8340180, 0x0000051e, 0xd8340280, 0x0000041e,
    0xd8340380, 0x0000041e, 0xd8340200, 0x0000051e, 0xbf8cc07f, 0x7e1c0306, 0x7e1e0307, 0x7e200306,
    0x7e220307, 0xd8b82000, 0x0a07061e, 0xd8bc0103, 0x0c06071e, 0xd8b8a060, 0x0e0f0e1e, 0x7da80090,
    0xd8b880e0, 0x1007061e, 0xbefe03c1, 0xbf8cc07f, 0xd8d80000, 0x1400001e, 0xd8d80080, 0x1500001e,
    0xd8d80300, 0x1600001e, 0xd8d80100, 0x1700001e, 0xd8d80180, 0x1800001e, 0xd8d80280, 0x1900001e,
    0xd8d80380, 0x1a00001e, 0xd8d80200, 0x1b00001e, 0xbf8cc07f, 0xe0702000, 0x80010a03, 0xe0702004,
    0x80010b03, 0xe0702008, 0x80010c03, 0xe070200c, 0x80010d03, 0xe0702010, 0x80010e03, 0xe0702014,
    0x80010f03, 0xe0702018, 0x80011003, 0xe070201c, 0x80011103, 0xe0702020, 0x80011403, 0xe0702024,
    0x80011503, 0xe0702028, 0x80011603, 0xe070202c, 0x80011703, 0xe0702030, 0x80011803, 0xe0702034,
    0x80011903, 0xe0702038, 0x80011a03, 0xe070203c, 0x80011b03, 0xbf810000,
];

const COINCIDING_OFFSETS: [u32; 3] = [0xd8b80505, 0x0a07061e, 0xbf810000];
const COINCIDING_OFFSETS_ST64: [u32; 3] = [0xd8bc0000, 0x0a07061e, 0xbf810000];
const DESTINATION_OVERFLOW: [u32; 3] = [0xd8b82000, 0xff07061e, 0xbf810000];

fn input() -> Vec<u32> {
    let mut v = vec![0u32; THREADS as usize * INPUTS];
    for tid in 0..THREADS {
        for word in 0..INPUTS as u32 {
            v[(tid * INPUTS as u32 + word) as usize] = ((word + 1) << 28) | (tid.wrapping_mul(0x0001_0203) + word);
        }
    }
    v
}

fn user() -> Vec<u32> {
    let mut user = Vec::new();
    put_desc(&mut user, 0, vdesc_full(IN_BASE, 4, (THREADS as usize * INPUTS) as u32, 0x1101_6fac));
    put_desc(&mut user, 4, vdesc_full(OUT_BASE, 4, (THREADS as usize * RESULTS) as u32, 0x1101_6fac));
    user
}

#[test]
fn ds_wrxchg2_b32() {
    let inp = input();
    let n_out = THREADS as usize * RESULTS;
    let data = inp.clone();
    let setup = move |m: &mut Machine| {
        m.lds = vec![0; LDS_DWORDS * 4];
        m.memory.write_words(IN_BASE, &data);
        m.memory.write_words(OUT_BASE, &vec![0xdeadbeef; n_out]);
    };
    let out = dispatch(&CODE, 32, THREADS, user(), &setup, &[(OUT_BASE, n_out)]).unwrap().remove(0);
    let names = [
        "ds_wrxchg2_rtn_b32 offset0 result", "ds_wrxchg2_rtn_b32 offset1 result",
        "ds_wrxchg2st64_rtn_b32 offset0 result", "ds_wrxchg2st64_rtn_b32 offset1 result",
        "aliased ds_wrxchg2_rtn_b32 offset0 result", "aliased ds_wrxchg2_rtn_b32 offset1 result",
        "masked ds_wrxchg2_rtn_b32 offset0 result", "masked ds_wrxchg2_rtn_b32 offset1 result",
        "ds_wrxchg2_rtn_b32 offset0 memory", "ds_wrxchg2_rtn_b32 offset1 memory",
        "ds_wrxchg2st64_rtn_b32 offset0 memory", "ds_wrxchg2st64_rtn_b32 offset1 memory",
        "aliased ds_wrxchg2_rtn_b32 offset0 memory", "aliased ds_wrxchg2_rtn_b32 offset1 memory",
        "masked ds_wrxchg2_rtn_b32 offset0 memory", "masked ds_wrxchg2_rtn_b32 offset1 memory",
    ];
    let mut c = Checker::new();
    for tid in 0..THREADS as usize {
        let (a, b, cc, d) = (inp[tid * INPUTS], inp[tid * INPUTS + 1], inp[tid * INPUTS + 2], inp[tid * INPUTS + 3]);
        let active = (tid as u32) < ACTIVE_LANES;
        let expected = [
            a, b, a, b, b, a, if active { a } else { cc }, if active { b } else { d },
            cc, d, d, cc, cc, d, if active { cc } else { a }, if active { d } else { b },
        ];
        for j in 0..RESULTS {
            c.eq(|| format!("lane {tid} {}", names[j]), out[tid * RESULTS + j], expected[j]);
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
fn ds_wrxchg2_b32_coinciding_offsets_rejected() {
    reject(&COINCIDING_OFFSETS, "DS write exchange of one location through both offsets is not supported");
    reject(&COINCIDING_OFFSETS_ST64, "DS write exchange of one location through both offsets is not supported");
}

#[test]
fn ds_wrxchg2_b32_destination_overflow_rejected() {
    reject(&DESTINATION_OVERFLOW, "DS write exchange destination register range overflow");
}
