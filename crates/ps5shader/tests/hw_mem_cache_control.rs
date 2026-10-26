// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Cache-control instructions (s_icache_inv, s_incperflevel /
//! s_decperflevel, s_dcache_wb / s_dcache_inv / s_gl1_inv,
//! buffer_gl1_inv / buffer_gl0_inv, s_wait_idle, s_ttracedata_imm) must
//! not change the result of a strided load / add / store around them.
//!
//! 64 threads of wave32 run as two consecutive waves over one memory.

include!("vectors/mem_common.rs");

use ps5shader::interp::Memory;
use ps5shader::ssa::SystemValue;

const THREADS: u32 = 64;
const STRIDE: usize = 4;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const CODE: [u32; 23] = [
    0x34020082, 0xbf930000, 0xbf940001, 0xe0302000, 0x80000401, 0xf4840000, 0x00000000, 0xf4800000,
    0x00000000, 0xf47c0000, 0x00000000, 0xe1c80000, 0x00000000, 0xe1c40000, 0x00000000, 0xbfa20000,
    0xbf8c3f70, 0xbfa80001, 0x4a080881, 0xbf950001, 0xe0702000, 0x80010401, 0xbf810000,
];

/// Stride-4-byte descriptor of `count` records.
fn desc(base: u64, count: u32) -> [u32; 4] {
    vdesc_full(base, 4, count, 0x0101_6fac)
}

#[test]
fn cache_control() {
    let n = THREADS as usize * STRIDE;
    let mut input = vec![0u32; n];
    for tid in 0..THREADS as usize {
        input[tid * STRIDE] = (tid as u32).wrapping_mul(0x0101_0101) + 7;
    }
    let mut user = Vec::new();
    put_desc(&mut user, 0, desc(IN_BASE, n as u32));
    put_desc(&mut user, 4, desc(OUT_BASE, n as u32));
    let wave = 32;
    let mut memory: Option<Memory> = None;
    for w in 0..THREADS / wave {
        let first = w * wave;
        let prev = memory.clone();
        let inp = input.clone();
        let setup = move |m: &mut Machine| {
            match &prev {
                Some(p) => m.memory = p.clone(),
                None => {
                    m.memory.write_words(IN_BASE, &inp);
                    m.memory.write_words(OUT_BASE, &vec![0xdeadbeef; n]);
                }
            }
            m.system.insert(SystemValue::LocalInvocationId(0), (first..first + wave).collect());
        };
        let m = ComputeRun::new(CODE.to_vec(), wave, wave, user.clone()).run_checked(&setup, &[(OUT_BASE, n)]).unwrap();
        memory = Some(m.memory);
    }
    let out = memory.unwrap().read_words(OUT_BASE, n);
    let mut c = Checker::new();
    for tid in 0..THREADS as usize {
        c.eq(|| format!("thread {tid}"), out[tid * STRIDE], input[tid * STRIDE] + 1);
    }
    c.finish();
}
