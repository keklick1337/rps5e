// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! A 32-bit buffer atomic whose byte address is off a DWORD boundary acts on
//! the dword that holds the address (RDNA2 ISA 8.1.7: the two low address
//! bits of dword accesses are ignored in DWORD alignment mode). Each lane
//! adds 1 at byte `4 * lane + offset`, offsets 0..3, and returns the old
//! dword; the dword after the last lane stays untouched.

include!("vectors/mem_common.rs");

const THREADS: u32 = 32;
const SENTINEL: u32 = 0x5eed_0001;
const BUF: u64 = 0x1_0000_0000;
const OUT: u64 = 0x2_0000_0000;

/// v1 = v0 << 2; v2 = 1; buffer_atomic_add v2, v1, s[0:3] offen
/// offset:N glc; s_waitcnt; buffer_store_dword v2, v1, s[4:7] offen.
const CODE: [u32; 8] = [0x34020082, 0x7e040281, 0xe0c85000, 0x80000201, 0xbf8c3f70, 0xe0701000, 0x80010201, 0xbf810000];

#[test]
fn unaligned_32bit_atomic_acts_on_its_dword() {
    let mut c = Checker::new();
    for offset in 0..4u32 {
        let mut code = CODE;
        code[2] = 0xe0c8_5000 | offset;
        let mut user = Vec::new();
        put_desc(&mut user, 0, vdesc(BUF, (THREADS + 1) * 4));
        put_desc(&mut user, 4, vdesc(OUT, THREADS * 4));
        let setup = |m: &mut Machine| {
            let mut words: Vec<u32> = (0..THREADS).map(|l| l * 100).collect();
            words.push(SENTINEL);
            m.memory.write_words(BUF, &words);
            m.memory.write_words(OUT, &[0xdead_beef; THREADS as usize]);
        };
        let out = dispatch(&code, 32, THREADS, user, &setup, &[(BUF, THREADS as usize + 1), (OUT, THREADS as usize)]).unwrap();
        for lane in 0..THREADS as usize {
            c.eq(|| format!("offset {offset} lane {lane} returned"), out[1][lane], lane as u32 * 100);
            c.eq(|| format!("offset {offset} lane {lane} dword"), out[0][lane], lane as u32 * 100 + 1);
        }
        c.eq(|| format!("offset {offset} sentinel"), out[0][THREADS as usize], SENTINEL);
    }
    c.finish();
}
