// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! FLAT *_d16 / *_d16_hi loads at varied offsets against hardware-measured
//! reference rows.

include!("vectors/mem_common.rs");
include!("vectors/mem_flat_d16_loads.rs");
include!("vectors/mem_flat_global_region.rs");

const THREADS: u32 = 32;
const RESULTS: usize = 16;
const BLOCK_BYTES: usize = 65536;
const REGION_OFFSET: u64 = THREADS as u64 * 16;
const SENTINEL: u32 = 0x05e471e1;
const BLOCK_BASE: u64 = 0x3_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

#[test]
fn flat_d16_loads() {
    let n_out = THREADS as usize * RESULTS;
    let mut user = vec![0u32; 10];
    put_desc(&mut user, 4, vdesc(OUT_BASE, (n_out * 4) as u32));
    user[8] = BLOCK_BASE as u32;
    user[9] = (BLOCK_BASE >> 32) as u32;
    let setup = move |m: &mut Machine| {
        write_bytes(m, BLOCK_BASE, &vec![0u8; BLOCK_BYTES]);
        for (tid, row) in ROWS.iter().enumerate() {
            m.memory.write_words(BLOCK_BASE + tid as u64 * 16, &row.0);
        }
        write_bytes(m, BLOCK_BASE + REGION_OFFSET, &FLAT_GLOBAL_LOADS_REGION);
        m.memory.write_words(OUT_BASE, &vec![SENTINEL; n_out]);
    };
    let out = dispatch(&CODE, 32, THREADS, user, &setup, &[(OUT_BASE, n_out)]).unwrap().remove(0);
    let mut c = Checker::new();
    for tid in 0..THREADS as usize {
        for r in 0..RESULTS {
            c.eq(|| format!("lane {tid} v{}", 10 + r), out[tid * RESULTS + r], ROWS[tid].1[r]);
        }
    }
    c.finish();
}
