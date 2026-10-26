// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! FLAT and GLOBAL loads (byte/short/dword, x2/x3/x4, sign/zero
//! extension, d16 hi/lo, saddr and vaddr forms, exec-masked lanes)
//! against hardware-measured reference rows.
//!
//! Layout: a 64 KiB guest block (64 KiB aligned) holds 32 per-lane input
//! rows of 4 dwords at `tid * 16`, followed by a 6144-byte data region at
//! offset 512. s[4:7] = output V#, s[8:9] = block address; each lane
//! writes 16 result dwords at `tid * 64` of the output.

include!("vectors/mem_common.rs");
include!("vectors/mem_flat_global_region.rs");
include!("vectors/mem_flat_global_loads.rs");

const THREADS: u32 = 32;
const RESULTS: usize = 16;
const SENTINEL: u32 = 0x05e4_71e1;
const BLOCK: u64 = 0x3_0000_0000;
const BLOCK_BYTES: usize = 65536;
const REGION_OFFSET: u64 = THREADS as u64 * 16;
const OUT_BASE: u64 = 0x2_0000_0000;

fn run(name: &str, code: &[u32], rows: &[([u32; 4], [u32; 16]); 32]) {
    let n_out = THREADS as usize * RESULTS;
    let mut user = vec![0u32; 10];
    put_desc(&mut user, 4, vdesc(OUT_BASE, (n_out * 4) as u32));
    user[8] = BLOCK as u32;
    user[9] = (BLOCK >> 32) as u32;
    let inputs: Vec<u32> = rows.iter().flat_map(|r| r.0).collect();
    let setup = move |m: &mut Machine| {
        write_bytes(m, BLOCK, &vec![0u8; BLOCK_BYTES]);
        m.memory.write_words(BLOCK, &inputs);
        write_bytes(m, BLOCK + REGION_OFFSET, &FLAT_GLOBAL_LOADS_REGION);
        m.memory.write_words(OUT_BASE, &vec![SENTINEL; n_out]);
    };
    let out = dispatch(code, 32, THREADS, user, &setup, &[(OUT_BASE, n_out)]).unwrap().remove(0);
    let mut c = Checker::new();
    for tid in 0..THREADS as usize {
        for r in 0..RESULTS {
            c.eq(|| format!("{name}: lane {tid} v{}", 10 + r), out[tid * RESULTS + r], rows[tid].1[r]);
        }
    }
    c.finish();
}

#[test]
fn flat_loads() {
    run("flat loads", &FLAT_LOADS_CODE, &FLAT_LOADS_ROWS);
}

#[test]
fn flat_loads_x2_x3() {
    run("flat loads x2 x3", &FLAT_LOADS_X2X3_CODE, &FLAT_LOADS_X2X3_ROWS);
}

#[test]
fn flat_loads_x4() {
    run("flat loads x4", &FLAT_LOADS_X4_CODE, &FLAT_LOADS_X4_ROWS);
}

#[test]
fn global_loads() {
    run("global loads", &GLOBAL_LOADS_CODE, &GLOBAL_LOADS_ROWS);
}

#[test]
fn global_loads_x2_x3() {
    run("global loads x2 x3", &GLOBAL_LOADS_X2X3_CODE, &GLOBAL_LOADS_X2X3_ROWS);
}

#[test]
fn global_loads_x4() {
    run("global loads x4", &GLOBAL_LOADS_X4_CODE, &GLOBAL_LOADS_X4_ROWS);
}
