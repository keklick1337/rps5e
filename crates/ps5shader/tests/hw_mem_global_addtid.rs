// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! global_load_dword_addtid / global_store_dword_addtid against
//! hardware-measured reference rows: positive and negative offsets,
//! exec-masked lanes, M0 ignored, multi-wave workgroups, and the
//! encodings that must be rejected (FLAT/scratch segment, no SGPR base).

include!("vectors/mem_common.rs");

use ps5shader::ssa::SystemValue;

/// Run a workgroup of `threads` invocations as consecutive waves of
/// `wave` lanes (local ids continue across waves), carrying the memory
/// regions from one wave to the next. Returns the final region words.
fn run_waves(code: &[u32], wave: u32, threads: u32, user: &[u32], regions: Vec<(u64, Vec<u32>)>) -> Result<Vec<Vec<u32>>, String> {
    let mut regions = regions;
    let mut first = 0;
    while first < threads {
        let lanes = wave.min(threads - first);
        let ids: Vec<u32> = (first..first + lanes).collect();
        let cur = regions.clone();
        let setup = move |m: &mut Machine| {
            for (a, w) in &cur {
                m.memory.write_words(*a, w);
            }
            m.system.insert(SystemValue::LocalInvocationId(0), ids.clone());
        };
        let outs: Vec<(u64, usize)> = regions.iter().map(|(a, w)| (*a, w.len())).collect();
        let m = dispatch_machine(code, wave, lanes, user.to_vec(), &setup, &outs)?;
        for (a, w) in regions.iter_mut() {
            *w = m.memory.read_words(*a, w.len());
        }
        first += lanes;
    }
    Ok(regions.into_iter().map(|(_, w)| w).collect())
}

/// Every case must fail to decode/compile; lists the accepted ones.
fn check_rejected(what: &str, cases: &[[u32; 3]], user_len: usize) {
    let accepted: Vec<String> = cases
        .iter()
        .filter(|code| ComputeRun::new(code.to_vec(), 32, 32, vec![0; user_len]).compile(false).is_ok())
        .map(|code| format!("{:#010x} {:#010x}", code[0], code[1]))
        .collect();
    assert!(accepted.is_empty(), "{what}: accepted {}", accepted.join(", "));
}

// ---------------------------------------------------------------- load

const MAX_THREADS: u32 = 128;
const RESULTS: usize = 8;
const WRITTEN: usize = 5;
const TABLE_DWORDS: usize = 256;
const UNTOUCHED: u32 = 0xdead_beef;
const SENTINEL: u32 = 0xabcd_abcd;
const POSITIVE_OFFSET_DWORD: u32 = 16;
const NEGATIVE_OFFSET_DWORD: u32 = 121;
const MASKED_EXEC_LO: u32 = 0x5555_5555;
const TABLE_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const LOAD_ADDTID_CODE: [u32; 41] = [
    0x34020085, 0x34040082, 0xbe880300, 0x8709ff01, 0x0000ffff, 0x800aff08, 0x00000200, 0x820b8009,
    0x7e080280, 0xd7650004, 0x000208c1, 0xd7660004, 0x000208c1, 0x7e0a02ff, 0xabcdabcd, 0x7e0c02ff,
    0xabcdabcd, 0x7e0e02ff, 0xabcdabcd, 0xe0301000, 0x80000802, 0xdc588040, 0x05080000, 0xdc588fe4,
    0x060a0000, 0xbefe03ff, 0x55555555, 0xdc588000, 0x07080000, 0xbefe03c1, 0xe0701000, 0x80010401,
    0xe0701004, 0x80010501, 0xe0701008, 0x80010601, 0xe070100c, 0x80010701, 0xe0701010, 0x80010801,
    0xbf810000,
];

fn run_load(threads: u32, wave: u32) {
    let table: Vec<u32> = (0..TABLE_DWORDS as u32).map(|i| 0x5100_0000u32.wrapping_add(i.wrapping_mul(0x0001_0203))).collect();
    let n_out = MAX_THREADS as usize * RESULTS;
    let mut user = Vec::new();
    put_desc(&mut user, 0, vdesc(TABLE_BASE, (TABLE_DWORDS * 4) as u32));
    put_desc(&mut user, 4, vdesc(OUT_BASE, (n_out * 4) as u32));
    let out = run_waves(&LOAD_ADDTID_CODE, wave, threads, &user, vec![(TABLE_BASE, table.clone()), (OUT_BASE, vec![UNTOUCHED; n_out])]).unwrap().remove(1);
    let names = ["positive offset", "negative offset", "masked by exec", "buffer load"];
    let mut c = Checker::new();
    for tid in 0..MAX_THREADS {
        let o = &out[tid as usize * RESULTS..][..RESULTS];
        let where_ = format!("global load addtid: wave{wave} {threads} threads thread {tid}");
        if tid >= threads {
            for j in 0..RESULTS {
                c.eq(|| format!("{where_} (outside the workgroup) result {j}"), o[j], UNTOUCHED);
            }
            continue;
        }
        let lane = o[0];
        c.eq(|| format!("{where_} lane id"), lane, tid % wave);
        let lane = lane.min(wave - 1);
        let active = lane >= 32 || (MASKED_EXEC_LO >> lane) & 1 != 0;
        let exp = [table[(POSITIVE_OFFSET_DWORD + lane) as usize], table[(NEGATIVE_OFFSET_DWORD + lane) as usize], if active { table[lane as usize] } else { SENTINEL }, table[tid as usize]];
        for j in 0..4 {
            c.eq(|| format!("{where_} lane {lane} {}", names[j]), o[j + 1], exp[j]);
        }
        for j in WRITTEN..RESULTS {
            c.eq(|| format!("{where_} result {j} (not written)"), o[j], UNTOUCHED);
        }
    }
    c.finish();
}

#[test]
fn load_addtid_wave32_32_threads() {
    run_load(32, 32);
}

#[test]
fn load_addtid_wave32_128_threads() {
    run_load(128, 32);
}

#[test]
fn load_addtid_wave64_64_threads() {
    run_load(64, 64);
}

#[test]
fn load_addtid_wave64_128_threads() {
    run_load(128, 64);
}

#[test]
fn load_addtid_rejections() {
    // FLAT segment, scratch segment, saddr = off.
    let cases = [[0xdc580040, 0x05080000, 0xbf810000], [0xdc584040, 0x05080000, 0xbf810000], [0xdc588040, 0x057d0000, 0xbf810000]];
    check_rejected("global_load_dword_addtid", &cases, 8);
}

// --------------------------------------------------------------- store

const BLOCK: u64 = 0x3_0000_0000;
const BLOCK_DWORDS: usize = 65536 / 4;
const STORE_BASE: u32 = 2048;
const FILL: u32 = 0x5a5a_5a5a;

const STORE_ADDTID_CODE: [u32; 41] = [
    0x800aff08, 0x00000800, 0x820b8009, 0x160800ff, 0x00000101, 0x4a0808ff, 0x00005100, 0x36500083,
    0x3a520880, 0xbefc0380, 0xdc5c8000, 0x000a2900, 0x3a5208ff, 0x11110000, 0xbefc03c0, 0xdc5c8f00,
    0x000a2900, 0x3a5208ff, 0x22220000, 0xbefc03ff, 0x0000ffff, 0xdc5c8c00, 0x000a2900, 0x3a5208ff,
    0x33330000, 0xbefc0380, 0x7daa5082, 0xdc5c8200, 0x000a2900, 0xbefe04c1, 0x3a5208ff, 0x44440000,
    0xbefc0380, 0xdc5c8800, 0x000a2900, 0x3a5208ff, 0x55550000, 0xbefc0380, 0xdc5c87fc, 0x000a2900,
    0xbf810000,
];

/// (offset, pattern, masked for lanes with (lane & 3) == 2)
const STORES: [(i32, u32, bool); 6] = [
    (0, 0x0000_0000, false),
    (-256, 0x1111_0000, false),
    (-1024, 0x2222_0000, false),
    (512, 0x3333_0000, true),
    (-2048, 0x4444_0000, false),
    (2044, 0x5555_0000, false),
];

fn run_store(wave: u32) {
    let mut user = vec![0u32; 10];
    user[8] = BLOCK as u32;
    user[9] = (BLOCK >> 32) as u32;
    let mem = run_waves(&STORE_ADDTID_CODE, wave, wave, &user, vec![(BLOCK, vec![FILL; BLOCK_DWORDS])]).unwrap().remove(0);
    let mut exp = vec![FILL; BLOCK_DWORDS];
    for &(offset, pattern, masked) in &STORES {
        for lane in 0..wave {
            if masked && lane & 3 == 2 {
                continue;
            }
            exp[((STORE_BASE as i32 + offset) as u32 + lane * 4) as usize / 4] = (0x5100 + lane * 0x101) ^ pattern;
        }
    }
    let mut c = Checker::new();
    for d in 0..BLOCK_DWORDS {
        c.eq(|| format!("global store addtid: wave{wave} byte {}", d * 4), mem[d], exp[d]);
    }
    c.finish();
}

#[test]
fn store_addtid_wave32() {
    run_store(32);
}

#[test]
fn store_addtid_wave64() {
    run_store(64);
}

#[test]
fn store_addtid_rejections() {
    // FLAT segment, scratch segment, saddr = off.
    let cases = [[0xdc5c0000, 0x000a2900, 0xbf810000], [0xdc5c4000, 0x000a2900, 0xbf810000], [0xdc5c8000, 0x007d2900, 0xbf810000]];
    check_rejected("global_store_dword_addtid", &cases, 10);
}
