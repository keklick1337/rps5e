// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! FLAT/GLOBAL stores (byte, short, d16-hi, dword, x2/x3/x4, saddr and
//! vaddr forms, negative offsets, exec-masked lanes) against
//! hardware-measured reference rows.
//!
//! Layout: s[0:1] = address of a 64 KiB guest block (64 KiB aligned),
//! prefilled with 0xcd; 64 invocations; the first 0x2000 bytes are checked.

include!("vectors/mem_common.rs");

const THREADS: u32 = 64;
const MASKED_THREADS: u32 = 16;
const BLOCK: u64 = 0x3_0000_0000;
const BLOCK_BYTES: usize = 65536;
const CHECKED_BYTES: usize = 0x2000;
const FILL: u8 = 0xcd;

const FLAT_STORE_CODE: [u32; 45] = [
    0x34020082, 0x34040084, 0x340c0081, 0x340e0083, 0x4a060200, 0x7e080201, 0x4a1806ff, 0x00000204,
    0x7e1a0201, 0x4a1c04ff, 0x00001000, 0x4a1000ff, 0xa1b2c300, 0x4a1202ff, 0x51000000, 0x4a1404ff,
    0x00c0ffee, 0x4a160eff, 0x7e570000, 0xdc708000, 0x00000801, 0xdc700100, 0x007d0903, 0xdc708ffc,
    0x007d0a0c, 0xdc608400, 0x00000800, 0xdc688480, 0x00000906, 0xdc708601, 0x00000b07, 0xdc788000,
    0x0000080e, 0xdc748400, 0x0000080e, 0xdc7c8800, 0x0000090e, 0xdc308000, 0x0f000001, 0xdc708500,
    0x00000f01, 0x7da80090, 0xdc708300, 0x00000a01, 0xbf810000,
];

/// Run a workgroup of `threads` invocations as consecutive waves of
/// `wave` lanes (local ids continue across waves), carrying the guest
/// block from one wave to the next. Returns the block bytes.
fn run_waves(code: &[u32], wave: u32, threads: u32, user: &[u32], block: Vec<u8>) -> Result<Vec<u8>, String> {
    let mut words = bytes_to_words(&block);
    let mut first = 0;
    while first < threads {
        let lanes = wave.min(threads - first);
        let ids: Vec<u32> = (first..first + lanes).collect();
        let cur = words.clone();
        let setup = move |m: &mut Machine| {
            m.memory.write_words(BLOCK, &cur);
            m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(0), ids.clone());
        };
        let m = dispatch_machine(code, wave, lanes, user.to_vec(), &setup, &[(BLOCK, words.len())])?;
        words = m.memory.read_words(BLOCK, words.len());
        first += lanes;
    }
    Ok(words.iter().flat_map(|w| w.to_le_bytes()).collect())
}

fn expected() -> Vec<u8> {
    let mut image = vec![FILL; CHECKED_BYTES];
    let mut put = |offset: u32, value: u32, bytes: u32| {
        for b in 0..bytes {
            image[(offset + b) as usize] = (value >> (b * 8)) as u8;
        }
    };
    for tid in 0..THREADS {
        let data = [0xa1b2_c300u32.wrapping_add(tid), 0x5100_0000 + tid * 4, 0x00c0_ffee + tid * 16, 0x7e57_0000 + tid * 8];
        put(tid * 4, data[0], 4);
        put(0x100 + tid * 4, data[1], 4);
        put(0x200 + tid * 4, data[2], 4);
        if tid < MASKED_THREADS {
            put(0x300 + tid * 4, data[2], 4);
        }
        put(0x400 + tid, data[0], 1);
        put(0x480 + tid * 2, data[1], 2);
        put(0x500 + tid * 4, data[0], 4);
        put(0x601 + tid * 8, data[3], 4);
        for d in 0..3 {
            put(0x800 + tid * 16 + d * 4, data[d as usize + 1], 4);
        }
        for d in 0..4 {
            put(0x1000 + tid * 16 + d * 4, data[d as usize], 4);
        }
        for d in 0..2 {
            put(0x1400 + tid * 16 + d * 4, data[d as usize], 4);
        }
    }
    image
}

fn run_stores(wave: u32) {
    let user = vec![BLOCK as u32, (BLOCK >> 32) as u32, 0, 0, 0, 0, 0, 0];
    let block = run_waves(&FLAT_STORE_CODE, wave, THREADS, &user, vec![FILL; BLOCK_BYTES]).unwrap();
    let exp = expected();
    let mut c = Checker::new();
    for off in 0..CHECKED_BYTES {
        c.eq(|| format!("flat store: wave{wave} byte {off:#x}"), block[off] as u32, exp[off] as u32);
    }
    c.finish();
}

#[test]
fn flat_store_wave32() {
    run_stores(32);
}

#[test]
fn flat_store_wave64() {
    run_stores(64);
}
