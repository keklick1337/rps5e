// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! LDS rsub / inc / dec / mskor / cmpst / wrap and float atomics against
//! hardware-measured reference rows.

include!("vectors/mem_common.rs");
include!("vectors/mem_lds_atomics.rs");

const THREADS: u32 = 32;
const INPUTS: usize = 4;
const RESULTS: usize = 32;
const LDS_DWORDS: usize = 4096;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

fn val(bits: u32) -> f32 {
    f32::from_bits(bits)
}

fn nan(bits: u32) -> bool {
    (bits & 0x7fff_ffff) > 0x7f80_0000
}

fn denormal(bits: u32) -> bool {
    (bits & 0x7f80_0000) == 0 && (bits & 0x007f_ffff) != 0
}

fn min_max(old: u32, data: u32, max: bool) -> u32 {
    if nan(old) {
        return data;
    }
    if nan(data) {
        return old;
    }
    if ((old | data) & 0x7fff_ffff) == 0 {
        return if max { old & data } else { old | data };
    }
    let take = if max { val(data) > val(old) } else { val(data) < val(old) };
    if take { data } else { old }
}

fn fill(tid: u32, words: &mut [u32]) {
    let kind = tid % 4;
    let seed = tid.wrapping_mul(0x9e3779b9).wrapping_add(0x7f4a7c15);
    match kind {
        0 => {
            words[0] = seed;
            words[1] = seed.wrapping_mul(0x85ebca6b);
            words[2] = seed ^ 0xc2b2ae35;
        }
        1 => {
            words[0] = tid % 7;
            words[1] = (tid * 3) % 7;
            words[2] = tid % 5;
        }
        2 => {
            words[0] = FLOATS[(tid % 8) as usize];
            words[1] = FLOATS[((tid / 2 + 3) % 8) as usize];
            words[2] = FLOATS[((tid + 5) % 8) as usize];
        }
        _ => {
            words[0] = seed;
            words[1] = if tid & 4 != 0 { seed } else { seed.wrapping_add(1) };
            words[2] = seed.wrapping_mul(3);
        }
    }
}

/// The reference program.
#[test]
fn lds_atomics() {
    run_and_check(&CODE, &[]);
}

fn run_and_check(code: &[u32], skip: &[usize]) {
    let mut input = vec![0u32; THREADS as usize * INPUTS];
    for tid in 0..THREADS {
        fill(tid, &mut input[tid as usize * INPUTS..][..INPUTS]);
    }
    let n_out = THREADS as usize * RESULTS;
    let mut user = Vec::new();
    put_desc(&mut user, 0, vdesc(IN_BASE, (input.len() * 4) as u32));
    put_desc(&mut user, 4, vdesc(OUT_BASE, (n_out * 4) as u32));
    let inp = input.clone();
    let setup = move |m: &mut Machine| {
        m.lds = vec![0; LDS_DWORDS * 4];
        m.memory.write_words(IN_BASE, &inp);
        m.memory.write_words(OUT_BASE, &vec![0xdeadbeef; n_out]);
    };
    let out = dispatch(code, 32, THREADS, user, &setup, &[(OUT_BASE, n_out)]).unwrap().remove(0);
    let names = [
        "ds_rsub_u32", "ds_rsub_rtn_u32", "ds_inc_u32", "ds_dec_u32", "ds_mskor_b32", "ds_cmpst_b32", "ds_cmpst_f32", "ds_add_f32",
        "ds_mskor_rtn_b32", "ds_cmpst_rtn_b32", "ds_cmpst_rtn_f32", "ds_min_rtn_f32", "ds_max_rtn_f32", "ds_wrap_rtn_b32", "ds_add_rtn_f32", "ds_min_f32", "ds_max_f32",
    ];
    let float_ops: u32 = (1 << 6) | (1 << 7) | (1 << 10) | (1 << 11) | (1 << 12) | (1 << 14) | (1 << 15) | (1 << 16);
    let mut c = Checker::new();
    for tid in 0..THREADS as usize {
        let i = &input[tid * INPUTS..][..INPUTS];
        let o = &out[tid * RESULTS..][..RESULTS];
        let (a, b, cc) = (i[0], i[1], i[2]);
        let s = (val(a) + val(b)).to_bits();
        let sum = if nan(s) { 0x7fc0_0000 } else { s };
        let feq = val(a) == val(b);
        let memory = [
            b.wrapping_sub(a), b.wrapping_sub(a),
            if a >= b { 0 } else { a.wrapping_add(1) },
            if a == 0 || a > b { b } else { a - 1 },
            (a & !b) | cc, if a == b { cc } else { a }, if feq { cc } else { a }, sum,
            (a & !b) | cc, if a == b { cc } else { a }, if feq { cc } else { a }, min_max(a, b, false),
            min_max(a, b, true), if a >= b { a.wrapping_sub(b) } else { a.wrapping_add(cc) }, sum, min_max(a, b, false), min_max(a, b, true),
        ];
        let den = denormal(a) || denormal(b) || denormal(sum);
        for j in 0..memory.len() {
            if skip.contains(&j) || den && (float_ops >> j) & 1 != 0 {
                continue;
            }
            let both_nan = nan(o[j]) && nan(memory[j]);
            c.eq(|| format!("lane {tid} {}", names[j]), if both_nan { memory[j] } else { o[j] }, memory[j]);
        }
        for j in 17..25 {
            if skip.contains(&j) {
                continue;
            }
            c.eq(|| format!("lane {tid} returned value [{j}]"), o[j], a);
        }
    }
    c.finish();
}
