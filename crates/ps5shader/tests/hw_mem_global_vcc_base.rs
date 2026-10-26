// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! GLOBAL loads/stores/atomics and addtid with VCC as the scalar base
//! address against hardware-measured reference rows, plus the scalar base
//! registers that must be rejected (vcc_hi, m0, exec pairs).
//!
//! Layout: s[0:1] = address of a 64 KiB guest block (64 KiB aligned);
//! dwords 0..768 hold 0x51000000 + i * 0x00010203, the rest 0xcdcdcdcd;
//! 64 invocations; the whole block is checked.

include!("vectors/mem_common.rs");

use ps5shader::ssa::SystemValue;

const THREADS: u32 = 64;
const BLOCK: u64 = 0x3_0000_0000;
const BLOCK_DWORDS: usize = 65536 / 4;
const INPUT_DWORDS: u32 = 768;
const FILL: u32 = 0xcdcd_cdcd;
const VECTOR_BASE: usize = 512;
const DWORD_BASE: usize = 4;
const ADDTID_STORE_BASE: usize = 256;
const ADDTID_BASE: usize = 384;
const ATOMIC_BASE: usize = 448;
const OUTPUT_BASE: usize = 1024;
const OUTPUT_DWORDS: usize = 8;

const VCC_BASE_CODE: [u32; 22] = [
    0xbeea0400, 0x34020082, 0x34040084, 0x340c0085, 0x4a0c0cff, 0x00001000, 0x4ad404ff, 0x00000048,
    0xdc308010, 0x046a0001, 0xdcc98700, 0x0c6a0201, 0xdc3887b8, 0x006a006a, 0xbf8c3f70, 0xdc788000,
    0x006a0006, 0xdc708010, 0x006a0406, 0xdc708014, 0x006a0c06, 0xbf810000,
];

const VCC_ADDTID_CODE: [u32; 12] = [
    0xbeea0400, 0x340c0085, 0x4a0c0cff, 0x00001000, 0xdc588600, 0x056a0000, 0xbf8c3f70, 0xdc708018,
    0x006a0506, 0xdc5c8400, 0x006a0500, 0xbf810000,
];

/// Run the 64-invocation workgroup as consecutive waves of `wave` lanes
/// (local ids continue across waves), carrying the guest block between
/// waves. Returns the block words.
fn run_waves(code: &[u32], wave: u32) -> Result<Vec<u32>, String> {
    let user = vec![BLOCK as u32, (BLOCK >> 32) as u32];
    let mut words = initial();
    let mut first = 0;
    while first < THREADS {
        let lanes = wave.min(THREADS - first);
        let ids: Vec<u32> = (first..first + lanes).collect();
        let cur = words.clone();
        let setup = move |m: &mut Machine| {
            m.memory.write_words(BLOCK, &cur);
            m.system.insert(SystemValue::LocalInvocationId(0), ids.clone());
        };
        let m = dispatch_machine(code, wave, lanes, user.clone(), &setup, &[(BLOCK, BLOCK_DWORDS)])?;
        words = m.memory.read_words(BLOCK, BLOCK_DWORDS);
        first += lanes;
    }
    Ok(words)
}

fn initial() -> Vec<u32> {
    let mut m = vec![FILL; BLOCK_DWORDS];
    for d in 0..INPUT_DWORDS {
        m[d as usize] = 0x5100_0000u32.wrapping_add(d.wrapping_mul(0x0001_0203));
    }
    m
}

fn expected_base() -> Vec<u32> {
    let init = initial();
    let mut m = init.clone();
    for tid in 0..THREADS as usize {
        let o = OUTPUT_BASE + tid * OUTPUT_DWORDS;
        for comp in 0..4 {
            m[o + comp] = init[VECTOR_BASE + tid * 4 + comp];
        }
        m[o + 4] = init[DWORD_BASE + tid];
        m[o + 5] = init[ATOMIC_BASE + tid];
        m[ATOMIC_BASE + tid] = init[ATOMIC_BASE + tid].wrapping_add(tid as u32 * 16);
    }
    m
}

fn expected_addtid(wave: u32) -> Vec<u32> {
    let init = initial();
    let mut m = init.clone();
    for tid in 0..THREADS as usize {
        let lane = tid % wave as usize;
        m[OUTPUT_BASE + tid * OUTPUT_DWORDS + 6] = init[ADDTID_BASE + lane];
        m[ADDTID_STORE_BASE + lane] = init[ADDTID_BASE + lane];
    }
    m
}

fn check(name: &str, code: &[u32], wave: u32, exp: &[u32]) {
    let mem = run_waves(code, wave).unwrap();
    let mut c = Checker::new();
    for d in 0..BLOCK_DWORDS {
        c.eq(|| format!("global vcc base: {name} byte {}", d * 4), mem[d], exp[d]);
    }
    c.finish();
}

#[test]
fn vcc_base_wave32() {
    check("wave32", &VCC_BASE_CODE, 32, &expected_base());
}

#[test]
fn vcc_base_wave64() {
    check("wave64", &VCC_BASE_CODE, 64, &expected_base());
}

#[test]
fn vcc_addtid_wave32() {
    check("addtid wave32", &VCC_ADDTID_CODE, 32, &expected_addtid(32));
}

#[test]
fn vcc_addtid_wave64() {
    check("addtid wave64", &VCC_ADDTID_CODE, 64, &expected_addtid(64));
}

#[test]
fn vcc_base_rejections() {
    // saddr = vcc_hi (load), m0 (atomic), exec_lo (store), vcc_hi (addtid).
    let cases: [[u32; 3]; 4] = [
        [0xdc308000, 0x046b0001, 0xbf810000],
        [0xdcc88000, 0x007c0201, 0xbf810000],
        [0xdc708000, 0x007e0201, 0xbf810000],
        [0xdc588000, 0x056b0000, 0xbf810000],
    ];
    let accepted: Vec<String> = cases
        .iter()
        .filter(|code| ComputeRun::new(code.to_vec(), 32, 32, vec![0; 2]).compile(false).is_ok())
        .map(|code| format!("{:#010x} {:#010x}", code[0], code[1]))
        .collect();
    assert!(accepted.is_empty(), "accepted {}", accepted.join(", "));
}
