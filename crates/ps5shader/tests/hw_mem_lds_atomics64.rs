// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! 64-bit LDS atomics (per-lane and contended) against hardware-measured
//! reference rows.

include!("vectors/mem_common.rs");
include!("vectors/mem_lds_atomics64.rs");

const THREADS: u32 = 32;
const INPUTS: usize = 8;
const RESULTS: usize = 128;
const OPERATIONS: usize = 35;
const RETURNING: usize = 18;
const LDS_DWORDS: usize = 2244;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;
const SIGN: u64 = 0x8000_0000_0000_0000;

const FLOATS: [u64; 12] = [
    0x0000000000000000, 0x8000000000000000, 0x3ff0000000000000, 0xbff0000000000000,
    0x7ff8000000000000, 0x7ff0000000000000, 0x4004000000000000, 0xc008000000000000,
    0x0000000000000001, 0x7ff0000000000001, 0xfff0000000000000, 0x800fffffffffffff,
];

fn nan(bits: u64) -> bool {
    (bits & !SIGN) > 0x7ff0_0000_0000_0000
}

fn order_key(bits: u64) -> u64 {
    if bits & SIGN != 0 { !bits } else { bits | SIGN }
}

fn min_max(old: u64, data: u64, max: bool) -> u64 {
    if nan(old) {
        return data;
    }
    if nan(data) {
        return old;
    }
    let take = if max { order_key(data) > order_key(old) } else { order_key(data) < order_key(old) };
    if take { data } else { old }
}

fn float_equal(a: u64, b: u64) -> bool {
    (a == b && !nan(a)) || ((a | b) & !SIGN) == 0
}

fn word(words: &[u32], index: usize) -> u64 {
    words[index * 2] as u64 | ((words[index * 2 + 1] as u64) << 32)
}

fn fill(tid: u32, words: &mut [u32]) {
    let kind = tid % 4;
    let seed = ((tid + 1) as u64).wrapping_mul(0x9e3779b97f4a7c15);
    let values: [u64; 3] = match kind {
        0 => [seed, seed.wrapping_mul(0xbf58476d1ce4e5b9), seed ^ 0x94d049bb133111eb],
        1 => [(tid % 7) as u64, ((tid * 3) % 7) as u64, (tid % 5) as u64],
        2 => [FLOATS[(tid % 12) as usize], FLOATS[((tid / 2 + 3) % 12) as usize], FLOATS[((tid + 5) % 12) as usize]],
        _ => [seed, if tid & 4 != 0 { seed } else { seed.wrapping_add(0x1_0000_0000) }, seed.wrapping_mul(3)],
    };
    for (i, v) in values.iter().enumerate() {
        words[i * 2] = *v as u32;
        words[i * 2 + 1] = (*v >> 32) as u32;
    }
}

fn input() -> Vec<u32> {
    let mut input = vec![0u32; THREADS as usize * INPUTS];
    for tid in 0..THREADS {
        fill(tid, &mut input[tid as usize * INPUTS..][..INPUTS]);
    }
    input
}

fn run() -> Result<Vec<u32>, String> {
    let input = input();
    let n_out = THREADS as usize * RESULTS;
    let mut user = Vec::new();
    put_desc(&mut user, 0, vdesc(IN_BASE, (input.len() * 4) as u32));
    put_desc(&mut user, 4, vdesc(OUT_BASE, (n_out * 4) as u32));
    let setup = move |m: &mut Machine| {
        m.lds = vec![0; LDS_DWORDS * 4];
        m.memory.write_words(IN_BASE, &input);
        m.memory.write_words(OUT_BASE, &vec![0xdeadbeef; n_out]);
    };
    Ok(dispatch(&CODE, 32, THREADS, user, &setup, &[(OUT_BASE, n_out)])?.remove(0))
}

const NAMES: [&str; OPERATIONS] = [
    "ds_add_u64", "ds_sub_u64", "ds_rsub_u64", "ds_inc_u64", "ds_dec_u64", "ds_min_i64", "ds_max_i64", "ds_min_u64", "ds_max_u64",
    "ds_and_b64", "ds_or_b64", "ds_xor_b64", "ds_mskor_b64", "ds_cmpst_b64", "ds_cmpst_f64", "ds_min_f64", "ds_max_f64",
    "ds_add_rtn_u64", "ds_sub_rtn_u64", "ds_rsub_rtn_u64", "ds_inc_rtn_u64", "ds_dec_rtn_u64", "ds_min_rtn_i64", "ds_max_rtn_i64",
    "ds_min_rtn_u64", "ds_max_rtn_u64", "ds_and_rtn_b64", "ds_or_rtn_b64", "ds_xor_rtn_b64", "ds_mskor_rtn_b64", "ds_wrxchg_rtn_b64",
    "ds_cmpst_rtn_b64", "ds_cmpst_rtn_f64", "ds_min_rtn_f64", "ds_max_rtn_f64",
];

/// Each lane works on its own LDS qwords: final memory and returned values.
#[test]
fn lds_atomics64_lanes() {
    let input = input();
    let out = run().unwrap();
    let mut c = Checker::new();
    for tid in 0..THREADS as usize {
        let i = &input[tid * INPUTS..][..INPUTS];
        let o = &out[tid * RESULTS..][..RESULTS];
        let (a, b, cc) = (word(i, 0), word(i, 1), word(i, 2));
        let (sa, sb) = (a as i64, b as i64);
        let common: [u64; 17] = [
            a.wrapping_add(b), a.wrapping_sub(b), b.wrapping_sub(a),
            if a >= b { 0 } else { a.wrapping_add(1) },
            if a == 0 || a > b { b } else { a - 1 },
            if sb < sa { b } else { a }, if sb > sa { b } else { a }, if b < a { b } else { a }, if b > a { b } else { a },
            a & b, a | b, a ^ b, (a & !b) | cc, if a == b { cc } else { a }, if float_equal(a, b) { cc } else { a },
            min_max(a, b, false), min_max(a, b, true),
        ];
        let mut memory = [0u64; OPERATIONS];
        memory[..17].copy_from_slice(&common);
        memory[17..30].copy_from_slice(&common[..13]);
        memory[30] = b;
        memory[31..35].copy_from_slice(&common[13..]);
        for j in 0..OPERATIONS {
            let v = word(o, j);
            c.eq(|| format!("lane {tid} {} lo", NAMES[j]), v as u32, memory[j] as u32);
            c.eq(|| format!("lane {tid} {} hi", NAMES[j]), (v >> 32) as u32, (memory[j] >> 32) as u32);
        }
        for j in 0..RETURNING {
            let v = word(o, OPERATIONS + 2 + j);
            c.eq(|| format!("lane {tid} {} returned value lo", NAMES[17 + j]), v as u32, a as u32);
            c.eq(|| format!("lane {tid} {} returned value hi", NAMES[17 + j]), (v >> 32) as u32, (a >> 32) as u32);
        }
    }
    c.finish();
}

/// All lanes on one qword: contended ds_add_rtn_u64 / ds_max_rtn_u64.
#[test]
fn lds_atomics64_contention() {
    let input = input();
    let out = run().unwrap();
    let inw = |tid: usize, k: usize| word(&input[tid * INPUTS..], k);
    let outw = |tid: usize, k: usize| word(&out[tid * RESULTS..], k);
    let mut total = 0u64;
    let mut maximum = 0u64;
    for tid in 0..THREADS as usize {
        total = total.wrapping_add(inw(tid, 1));
        maximum = maximum.max(inw(tid, 2));
    }
    assert_eq!(outw(0, OPERATIONS), total, "contended ds_add_rtn_u64 total");
    assert_eq!(outw(0, OPERATIONS + 1), maximum, "contended ds_max_rtn_u64 result");
    let mut pending: Vec<usize> = (0..THREADS as usize).collect();
    let mut current = 0u64;
    while !pending.is_empty() {
        let mut next: Option<usize> = None;
        for (k, &t) in pending.iter().enumerate() {
            if outw(t, OPERATIONS + 2 + RETURNING) != current {
                continue;
            }
            if next.is_none() || inw(t, 1) == 0 {
                next = Some(k);
            }
        }
        let k = next.unwrap_or_else(|| panic!("contended ds_add_rtn_u64 returned values do not form one serial order at {current:#018x}"));
        current = current.wrapping_add(inw(pending[k], 1));
        pending.remove(k);
    }
    for tid in 0..THREADS as usize {
        let old = outw(tid, OPERATIONS + 3 + RETURNING);
        let seen = old == 0 || (0..THREADS as usize).any(|l| inw(l, 2) == old);
        assert!(seen && old <= maximum, "contended ds_max_rtn_u64 returned {old:#018x} on lane {tid}");
    }
}
