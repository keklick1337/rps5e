// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Buffer atomics (inc/dec and 64-bit integer atomics) against
//! hardware-measured reference rows.

include!("vectors/mem_common.rs");

const THREADS: u32 = 32;
const INPUTS: usize = 8;
const RESULTS: usize = 32;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const CODE: [u32; 78] = [
    0x34020085, 0x34060087, 0xe0301000, 0x80000401, 0xe0301004, 0x80000501, 0xe0301008, 0x80000601,
    0xe030100c, 0x80000701, 0xe0301010, 0x80000801, 0xe0301014, 0x80000901, 0xbf8c3f70, 0xe0701000,
    0x80010403, 0xe0701008, 0x80010403, 0xe0741010, 0x80010403, 0xe0741018, 0x80010403, 0xe0741020,
    0x80010403, 0xe0741028, 0x80010403, 0xe0741030, 0x80010403, 0xe0741038, 0x80010403, 0xe0741040,
    0x80010403, 0xe0741048, 0x80010403, 0xe0741050, 0x80010403, 0xbf8c3f70, 0x7e280306, 0xe0f05000,
    0x80011403, 0x7e2a0306, 0xe0f45008, 0x80011503, 0x7e2c0306, 0x7e2e0307, 0xe1485010, 0x80011603,
    0xe14c1018, 0x80010603, 0xe1541020, 0x80010603, 0xe1581028, 0x80010603, 0xe15c1030, 0x80010603,
    0xe1601038, 0x80010603, 0xe1641040, 0x80010603, 0xe16c1048, 0x80010603, 0x7e300306, 0x7e320307,
    0x7e340308, 0x7e360309, 0xe1445050, 0x80011803, 0xbf8c3f70, 0xe0701058, 0x80011403, 0xe070105c,
    0x80011503, 0xe0741060, 0x80011603, 0xe0741068, 0x80011803, 0xbf810000,
];

fn fill(tid: u32, words: &mut [u32]) {
    let seed = ((tid + 1) as u64).wrapping_mul(0x9e3779b97f4a7c15);
    let edges = [0u64, !0u64, 0x8000000000000000, 0x7fffffffffffffff];
    let a = if tid < 8 { edges[(tid % 4) as usize] } else { seed };
    let b = if tid < 8 { edges[((tid / 4 + tid + 1) % 4) as usize] } else { seed.wrapping_mul(0xc2b2ae3d27d4eb4f) };
    let c = if tid & 1 != 0 { a } else { seed ^ 1 };
    for (i, v) in [a, b, c].iter().enumerate() {
        words[2 * i] = *v as u32;
        words[2 * i + 1] = (*v >> 32) as u32;
    }
    if tid % 3 == 0 {
        words[2] = words[0] % 7;
        words[0] %= 9;
    }
}

#[test]
fn buffer_atomics() {
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
        m.memory.write_words(IN_BASE, &inp);
        m.memory.write_words(OUT_BASE, &vec![0xdeadbeef; n_out]);
    };
    let out = dispatch(&CODE, 32, THREADS, user, &setup, &[(OUT_BASE, n_out)]).unwrap().remove(0);
    let mut c = Checker::new();
    let w64 = |w: &[u32]| w[0] as u64 | ((w[1] as u64) << 32);
    for tid in 0..THREADS as usize {
        let i = &input[tid * INPUTS..][..INPUTS];
        let o = &out[tid * RESULTS..][..RESULTS];
        let (a, b, cc) = (w64(&i[0..]), w64(&i[2..]), w64(&i[4..]));
        let (a32, b32) = (i[0], i[2]);
        c.eq(|| format!("lane {tid} buffer_atomic_inc"), o[0], if a32 >= b32 { 0 } else { a32 + 1 });
        c.eq(|| format!("lane {tid} buffer_atomic_dec"), o[2], if a32 == 0 || a32 > b32 { b32 } else { a32 - 1 });
        let (sa, sb) = (a as i64, b as i64);
        let expected = [a.wrapping_add(b), a.wrapping_sub(b), sa.min(sb) as u64, a.min(b), sa.max(sb) as u64, a.max(b), a & b, a ^ b, if a == cc { b } else { a }];
        let names = ["add_x2", "sub_x2", "smin_x2", "umin_x2", "smax_x2", "umax_x2", "and_x2", "xor_x2", "cmpswap_x2"];
        for j in 0..9 {
            c.eq(|| format!("lane {tid} {} lo", names[j]), o[4 + 2 * j], expected[j] as u32);
            c.eq(|| format!("lane {tid} {} hi", names[j]), o[5 + 2 * j], (expected[j] >> 32) as u32);
        }
        c.eq(|| format!("lane {tid} inc returned value"), o[22], a32);
        c.eq(|| format!("lane {tid} dec returned value"), o[23], a32);
        c.eq(|| format!("lane {tid} add_x2 returned low"), o[24], i[0]);
        c.eq(|| format!("lane {tid} add_x2 returned high"), o[25], i[1]);
        c.eq(|| format!("lane {tid} cmpswap_x2 returned low"), o[26], i[0]);
        c.eq(|| format!("lane {tid} cmpswap_x2 returned high"), o[27], i[1]);
    }
    c.finish();
}
