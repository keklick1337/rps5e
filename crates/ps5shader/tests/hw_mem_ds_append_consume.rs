// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! LDS ds_append / ds_consume (plain and exec-masked, M0 base) against
//! hardware-measured reference rows.

include!("vectors/mem_common.rs");

const THREADS: u32 = 32;
const RESULTS: usize = 16;
const ACTIVE_LANES: u32 = 16;
const LDS_DWORDS: usize = 256;
const APPEND_START: u32 = 1000;
const CONSUME_START: u32 = 2000;
const MASKED_APPEND_START: u32 = 3000;
const MASKED_CONSUME_START: u32 = 4000;
const BASE_SHIFTED_START: u32 = 5000;
const UNTOUCHED_APPEND: u32 = 0x5eed_0001;
const UNTOUCHED_CONSUME: u32 = 0x5eed_0002;
const OUT_BASE: u64 = 0x2_0000_0000;

const CODE: [u32; 98] = [
    0x34060084, 0x7e100280, 0x7e0802ff, APPEND_START, 0x7e0a02ff, CONSUME_START, 0x7e0c02ff, MASKED_APPEND_START,
    0x7e0e02ff, MASKED_CONSUME_START, 0x7e1c02ff, BASE_SHIFTED_START, 0x7e1e02ff, BASE_SHIFTED_START + 1, 0x7e2002ff, BASE_SHIFTED_START + 2,
    0x7e2202ff, BASE_SHIFTED_START + 3, 0x7e1802ff, UNTOUCHED_APPEND, 0x7e1a02ff, UNTOUCHED_CONSUME, 0x7da40080, 0xd8340010,
    0x00000408, 0xd8340014, 0x00000508, 0xd8340018, 0x00000608, 0xd834001c, 0x00000708, 0xd8340050,
    0x00000e08, 0xd8340054, 0x00000f08, 0xd8340058, 0x00001008, 0xd834005c, 0x00001108, 0xbefe03c1,
    0xbf8cc07f, 0xbefc03ff, 0x0040ffff, 0xd8f80010, 0x0a000000, 0xd8f40014, 0x0b000000, 0xbefc03ff,
    0x00400000, 0x7da80090, 0xd8f80018, 0x0c000000, 0xd8f4001c, 0x0d000000, 0xbefe03c1, 0xbf8cc07f,
    0xd8d80010, 0x14000008, 0xd8d80014, 0x15000008, 0xd8d80018, 0x16000008, 0xd8d8001c, 0x17000008,
    0xd8d80050, 0x18000008, 0xd8d80054, 0x19000008, 0xd8d80058, 0x1a000008, 0xd8d8005c, 0x1b000008,
    0xbf8cc07f, 0xe0702000, 0x80000a03, 0xe0702004, 0x80000b03, 0xe0702008, 0x80000c03, 0xe070200c,
    0x80000d03, 0xe0702010, 0x80001403, 0xe0702014, 0x80001503, 0xe0702018, 0x80001603, 0xe070201c,
    0x80001703, 0xe0702020, 0x80001803, 0xe0702024, 0x80001903, 0xe0702028, 0x80001a03, 0xe070202c,
    0x80001b03, 0xbf810000,
];

/// Checks result slots `slots` of every lane.
fn check(slots: &[usize]) {
    let n_out = THREADS as usize * RESULTS;
    let mut user = Vec::new();
    put_desc(&mut user, 0, vdesc_full(OUT_BASE, 4, n_out as u32, 0x1101_6fac));
    let setup = move |m: &mut Machine| {
        m.lds = vec![0; LDS_DWORDS * 4];
        m.memory.write_words(OUT_BASE, &vec![0xdeadbeef; n_out]);
    };
    let out = dispatch(&CODE, 32, THREADS, user, &setup, &[(OUT_BASE, n_out)]).unwrap().remove(0);
    let names = [
        "ds_append result", "ds_consume result", "masked ds_append result", "masked ds_consume result",
        "ds_append counter", "ds_consume counter", "masked ds_append counter", "masked ds_consume counter",
        "dword at M0.base + ds_append offset", "dword at M0.base + ds_consume offset",
        "dword at M0.base + masked ds_append offset", "dword at M0.base + masked ds_consume offset",
    ];
    let mut c = Checker::new();
    for tid in 0..THREADS {
        let active = tid < ACTIVE_LANES;
        let expected = [
            APPEND_START, CONSUME_START,
            if active { MASKED_APPEND_START } else { UNTOUCHED_APPEND },
            if active { MASKED_CONSUME_START } else { UNTOUCHED_CONSUME },
            APPEND_START + THREADS, CONSUME_START - THREADS, MASKED_APPEND_START + ACTIVE_LANES, MASKED_CONSUME_START - ACTIVE_LANES,
            BASE_SHIFTED_START, BASE_SHIFTED_START + 1, BASE_SHIFTED_START + 2, BASE_SHIFTED_START + 3,
        ];
        for &j in slots {
            c.eq(|| format!("lane {tid} {}", names[j]), out[tid as usize * RESULTS + j], expected[j]);
        }
    }
    c.finish();
}

/// ds_append / ds_consume with M0 = 0x0040ffff: the hardware uses only the
/// instruction offset as the LDS address (M0 does not shift it).
#[test]
fn ds_append_consume_m0_ffff() {
    check(&[0, 1, 4, 5]);
}

/// Exec-masked ds_append / ds_consume (M0 = 0x00400000) and the dwords at
/// M0.base + offset, which must stay untouched.
#[test]
fn ds_append_consume_masked_and_m0_base() {
    check(&[2, 3, 6, 7, 8, 9, 10, 11]);
}
