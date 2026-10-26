// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! FLAT loads/stores/atomics carrying cache-policy bits (glc/slc/dlc)
//! against hardware-measured reference rows: results, final memory, and
//! no stores outside the rows.

include!("vectors/mem_common.rs");
include!("vectors/mem_flat_cache_bits.rs");

const THREADS: u32 = 32;
const RESULTS: usize = 4;
const BLOCK_BYTES: usize = 65536;
const FILL: u8 = 0xcd;
const SENTINEL: u32 = 0x05e471e1;
const BLOCK_BASE: u64 = 0x3_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

fn run() -> Machine<'static> {
    let n_out = THREADS as usize * RESULTS;
    let mut user = vec![0u32; 10];
    put_desc(&mut user, 4, vdesc(OUT_BASE, (n_out * 4) as u32));
    user[8] = BLOCK_BASE as u32;
    user[9] = (BLOCK_BASE >> 32) as u32;
    let setup = move |m: &mut Machine| {
        write_bytes(m, BLOCK_BASE, &vec![FILL; BLOCK_BYTES]);
        for tid in 0..THREADS as usize {
            m.memory.write_words(BLOCK_BASE + tid as u64 * 16, &ROWS[tid % ROWS.len()].0);
        }
        m.memory.write_words(OUT_BASE, &vec![SENTINEL; n_out]);
    };
    dispatch_machine(&CODE, 32, THREADS, user, &setup, &[(OUT_BASE, n_out), (BLOCK_BASE, BLOCK_BYTES / 4)]).unwrap()
}

#[test]
fn flat_cache_bits_results() {
    let m = run();
    let out = m.memory.read_words(OUT_BASE, THREADS as usize * RESULTS);
    let mut c = Checker::new();
    for tid in 0..THREADS as usize {
        let row = &ROWS[tid % ROWS.len()];
        for r in 0..RESULTS {
            c.eq(|| format!("wave32 lane {tid} v{}", 10 + r), out[tid * RESULTS + r], row.1[r]);
        }
    }
    c.finish();
}

#[test]
fn flat_cache_bits_final_memory() {
    let m = run();
    let mut c = Checker::new();
    for tid in 0..THREADS as usize {
        let row = &ROWS[tid % ROWS.len()];
        let mem = m.memory.read_words(BLOCK_BASE + tid as u64 * 16, 4);
        for d in 0..4 {
            c.eq(|| format!("wave32 lane {tid} memory dword {d}"), mem[d], row.2[d]);
        }
    }
    c.finish();
}

#[test]
fn flat_cache_bits_no_stray_stores() {
    let m = run();
    let mut c = Checker::new();
    for offset in THREADS as u64 * 16..BLOCK_BYTES as u64 {
        c.eq(|| format!("byte {offset} outside the rows"), m.memory.read_u8(BLOCK_BASE + offset) as u32, FILL as u32);
    }
    c.finish();
}
