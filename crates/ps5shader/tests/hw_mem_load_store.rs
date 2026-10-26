// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Buffer loads/stores (sub-dword loads with sign/zero extension, multi-
//! dword loads, soffset/offset combinations, and a per-lane "scratch"
//! region of stored dwords) against hardware-measured reference rows.

include!("vectors/mem_common.rs");
include!("vectors/mem_load_store.rs");

const INPUTS: usize = 4;
const RESULTS: usize = 128;
const SHARED_WORDS: usize = 16;
const CHECKED: usize = 30;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const NAMES: [&str; CHECKED] = [
    "load_ubyte", "load_sbyte", "load_ushort", "load_sshort",
    "load_dwordx2[0]", "load_dwordx2[1]", "load_dwordx3[0]", "load_dwordx3[1]",
    "load_dwordx3[2]", "buffer_load_ubyte (soffset 3, offset 5)", "buffer_load_sbyte (offset 15)", "buffer_load_sshort (soffset 3, offset 11)",
    "scratch dword 0", "scratch dword 1", "scratch dword 2", "scratch dword 3",
    "scratch dword 4", "scratch dword 5", "scratch dword 6", "scratch dword 7",
    "scratch dword 8", "scratch dword 9", "scratch dword 10", "scratch dword 11",
    "scratch dword 12", "scratch dword 13", "scratch dword 14", "scratch dword 15",
    "scratch dword 62", "scratch dword 63",
];

/// Descriptor dword3 used by this test (raw buffer, OOB_SELECT 3).
const DWORD3: u32 = 0x3101_6fac;

fn run(code: &[u32], wave: u32) -> Vec<u32> {
    let lanes = wave as usize;
    let mut input: Vec<u32> = ROWS[..lanes].iter().flatten().copied().collect();
    input.extend_from_slice(&SHARED_INIT);
    let n_out = lanes * RESULTS;
    let mut user = Vec::new();
    put_desc(&mut user, 0, vdesc_full(IN_BASE, 0, ((lanes * INPUTS + SHARED_WORDS) * 4) as u32, DWORD3));
    put_desc(&mut user, 4, vdesc_full(OUT_BASE, 0, (n_out * 4) as u32, DWORD3));
    let setup = move |m: &mut Machine| {
        m.memory.write_words(IN_BASE, &input);
        m.memory.write_words(OUT_BASE, &vec![0xdeadbeef; n_out]);
    };
    dispatch(code, wave, wave, user, &setup, &[(OUT_BASE, n_out)]).unwrap().remove(0)
}

fn check(code: &[u32], wave: u32, expected: &[[u32; CHECKED]], columns: std::ops::Range<usize>) {
    let out = run(code, wave);
    let mut c = Checker::new();
    for (tid, exp) in expected.iter().enumerate() {
        for k in columns.clone() {
            c.eq(|| format!("wave{wave} lane {tid} {} (row {:08x?})", NAMES[k], ROWS[tid]), out[tid * RESULTS + k], exp[k]);
        }
    }
    c.finish();
}

#[test]
fn wave32_loads() {
    check(&WAVE32_CODE, 32, &EXPECTED32, 0..12);
}

#[test]
fn wave32_stores() {
    check(&WAVE32_CODE, 32, &EXPECTED32, 12..30);
}

#[test]
fn wave64_loads() {
    check(&WAVE64_CODE, 64, &EXPECTED64, 0..12);
}

#[test]
fn wave64_stores() {
    check(&WAVE64_CODE, 64, &EXPECTED64, 12..30);
}
