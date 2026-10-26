// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Scalar subroutine calls (S_CALL_B64, S_SWAPPC_B64 with a link register)
//! and their S_SETPC_B64 returns: hardware-measured execution rows and the
//! call/return model's acceptance rules.

include!("vectors/scalar_common.rs");

use ps5shader::cfg::{Cfg, CfgError, Exit};
use ps5shader::{decode_instruction, Program};

// Each lane loads (selector, value, x, y) and stores
// (selector + 0x13579bdf, value ^ 0xa5a5a5a5, x + y, y ^ 0x5a5a5a5a); the
// arithmetic runs in called subroutines.
const SWAPPC_FORWARD: [u32; 21] = [
    0x34020084, 0xe0381000, 0x80000401, 0xbf8c3f70, 0xbe8a1f00, 0x800aff0a, 0x00000014, 0x820b800b, 0xbe88210a, 0xbf820008, 0x4a0808ff, 0x13579bdf,
    0x3a0a0aff, 0xa5a5a5a5, 0x4a0c0f06, 0x3a0e0eff, 0x5a5a5a5a, 0xbe802008, 0xe0781000, 0x80010401, 0xbf810000,
];
const CALL_FORWARD: [u32; 17] = [
    0x34020084, 0xe0381000, 0x80000401, 0xbf8c3f70, 0xbb080001, 0xbf820008, 0x4a0808ff, 0x13579bdf, 0x3a0a0aff, 0xa5a5a5a5, 0x4a0c0f06, 0x3a0e0eff,
    0x5a5a5a5a, 0xbe802008, 0xe0781000, 0x80010401, 0xbf810000,
];
const CALL_BACKWARD: [u32; 18] = [
    0x34020084, 0xe0381000, 0x80000401, 0xbf8c3f70, 0xbf820008, 0x4a0808ff, 0x13579bdf, 0x3a0a0aff, 0xa5a5a5a5, 0x4a0c0f06, 0x3a0e0eff, 0x5a5a5a5a,
    0xbe802008, 0xbb08fff7, 0xbf820000, 0xe0781000, 0x80010401, 0xbf810000,
];
const CALL_AFTER_END: [u32; 17] = [
    0x34020084, 0xe0381000, 0x80000401, 0xbf8c3f70, 0xbb080003, 0xe0781000, 0x80010401, 0xbf810000, 0x4a0808ff, 0x13579bdf, 0x3a0a0aff, 0xa5a5a5a5,
    0x4a0c0f06, 0x3a0e0eff, 0x5a5a5a5a, 0xbe802008, 0xbf810000,
];
const MIXED_NESTED: [u32; 24] = [
    0x34020084, 0xe0381000, 0x80000401, 0xbf8c3f70, 0xbe8a1f00, 0x800aff0a, 0x00000014, 0x820b800b, 0xbe88210a, 0xbf82000b, 0x4a0808ff, 0x13579bdf,
    0xbb0c0001, 0xbf820006, 0x3a0a0aff, 0xa5a5a5a5, 0x4a0c0f06, 0x3a0e0eff, 0x5a5a5a5a, 0xbe80200c, 0xbe802008, 0xe0781000, 0x80010401, 0xbf810000,
];

const SELECTORS: [u32; 8] = [0, 1, 2, 3, 4, 7, 0x8000_0000, 0xffff_ffff];

fn run_static(name: &str, code: &[u32], wave: u32) {
    let lanes = wave as usize;
    let mut input = vec![0u32; 64 * 4];
    for tid in 0..64usize {
        input[tid * 4] = SELECTORS[tid % SELECTORS.len()];
        input[tid * 4 + 1] = 0x9e37_79b9u32.wrapping_mul(tid as u32 + 1);
        input[tid * 4 + 2] = 0xffff_ffff - tid as u32;
        input[tid * 4 + 3] = 17 * tid as u32 + 1;
    }
    let ud = user_data(raw_desc(IN_BASE, 64 * 16), raw_desc(OUT_BASE, 64 * 16), 8);
    let out = Dispatch::new(code, wave, wave, ud).buffer(IN_BASE, input.clone()).output(OUT_BASE, 64 * 4).run().unwrap().remove(0);
    let mut failures = Vec::new();
    for tid in 0..lanes {
        let i = &input[tid * 4..tid * 4 + 4];
        let want = [i[0].wrapping_add(0x1357_9bdf), i[1] ^ 0xa5a5_a5a5, i[2].wrapping_add(i[3]), i[3] ^ 0x5a5a_5a5a];
        for (k, w) in want.iter().enumerate() {
            let got = out[tid * 4 + k];
            if got != *w {
                failures.push(format!("{name} wave{wave}: lane {tid} result {k} is {got:#010x}, expected {w:#010x}"));
            }
        }
    }
    for (k, &w) in out.iter().enumerate().skip(lanes * 4) {
        if w != UNTOUCHED {
            failures.push(format!("{name} wave{wave}: inactive lane wrote word {k}"));
        }
    }
    assert_clean(failures);
}

#[test]
fn static_calls_execute() {
    let variants: [(&str, &[u32]); 5] = [
        ("SWAPPC forward", &SWAPPC_FORWARD),
        ("CALL forward", &CALL_FORWARD),
        ("CALL backward", &CALL_BACKWARD),
        ("CALL after endpgm", &CALL_AFTER_END),
        ("mixed nested calls", &MIXED_NESTED),
    ];
    for (name, code) in variants {
        run_static(name, code, 32);
        run_static(name, code, 64);
    }
}

/// Program of the given instructions, one per word sequence, at consecutive
/// pcs (no decode termination rules).
fn build(words: &[&[u32]]) -> Result<Cfg, CfgError> {
    let mut pc = 0u32;
    let mut instructions = Vec::new();
    for w in words {
        let i = decode_instruction(pc, w).unwrap();
        pc += i.words.len() as u32 * 4;
        instructions.push(i);
    }
    Cfg::build(&Program::new(0x1000, instructions), None)
}

fn call(link: u32, imm: u32) -> [u32; 1] {
    [0xbb00_0000 | (link << 16) | (imm & 0xffff)]
}
fn sop1(dst: u32, op: u32, src: u32) -> [u32; 1] {
    [0xbe80_0000 | (dst << 16) | (op << 8) | src]
}
fn sopp(op: u32, imm: u32) -> [u32; 1] {
    [0xbf80_0000 | (op << 16) | imm]
}
fn setpc(link: u32) -> [u32; 1] {
    sop1(0, 0x20, link)
}
const ENDPGM: [u32; 1] = [0xbf81_0000];
const NOP: [u32; 1] = [0xbf80_0000];

fn expect(needle: &str, r: Result<Cfg, CfgError>) {
    match r {
        Err(e) => assert!(e.to_string().contains(needle), "expected {needle}, got {e}"),
        Ok(cfg) => panic!("expected {needle}, built\n{cfg}"),
    }
}

fn block_at(cfg: &Cfg, pc: u32) -> usize {
    cfg.blocks.iter().position(|b| b.start_pc == pc && b.first < b.end).unwrap_or_else(|| panic!("no block at {pc:#x}\n{cfg}"))
}

#[test]
fn call_after_end_links_caller_callee_and_return() {
    let cfg = build(&[&call(8, 1), &ENDPGM, &NOP, &setpc(8), &ENDPGM]).unwrap();
    let (caller, callee, after) = (block_at(&cfg, 0), block_at(&cfg, 8), block_at(&cfg, 4));
    assert_eq!(cfg.blocks[caller].exit, Exit::Branch(callee));
    // The callee's block [8, 12] ends in the return.
    let ret = cfg.blocks.iter().position(|b| b.start_pc <= 12 && b.end > b.first && b.exit == Exit::Branch(after));
    assert!(ret.is_some(), "{cfg}");
}

#[test]
fn rejected_calls() {
    expect("invalid instruction boundary", build(&[&call(8, 16), &ENDPGM]));
    expect("no paired s_setpc_b64 return", build(&[&call(8, 0), &ENDPGM]));
    for link in [9, 105, 106, 125] {
        expect("ordinary aligned scalar register pair", build(&[&call(link, 0), &ENDPGM]));
    }
    expect("escapes", build(&[&call(8, 1), &ENDPGM, &sop1(8, 3, 128), &setpc(8)]));
    expect("recursive scalar call", build(&[&call(8, 0xffff), &setpc(8), &ENDPGM]));
    expect("is shared with another call", build(&[&call(8, 2), &call(8, 1), &ENDPGM, &setpc(8)]));
    expect("multiple s_setpc_b64 returns", build(&[&call(8, 1), &ENDPGM, &setpc(8), &setpc(8)]));
    expect("control flow enters the call region", build(&[&call(8, 2), &sopp(2, 1), &ENDPGM, &NOP, &setpc(8)]));
    expect("control flow leaves the call region", build(&[&call(8, 1), &ENDPGM, &sopp(2, 2), &setpc(8), &ENDPGM, &ENDPGM]));
}

#[test]
fn call_guards() {
    // s_load_dwordx8 s[4:11] inside the region overwrites the link.
    expect("escapes", build(&[&call(8, 1), &ENDPGM, &[0xf40c_0100, 0xfa00_0000], &setpc(8)]));
    // Return through s[9:10], not the link.
    expect("escapes", build(&[&call(8, 1), &ENDPGM, &setpc(9)]));
    expect("control flow enters the call region", build(&[&call(8, 2), &NOP, &NOP, &NOP, &setpc(8)]));
    expect("control flow enters the call region", build(&[&call(8, 2), &sopp(2, 0), &sopp(4, 2), &NOP, &setpc(8), &ENDPGM]));
    expect("return precedes the call target", build(&[&call(8, 2), &ENDPGM, &setpc(8), &NOP, &ENDPGM]));
    // buffer_load_dword v1, v0, s[8:11] reads the link as a descriptor.
    expect("escapes", build(&[&call(10, 1), &ENDPGM, &[0xe030_1000, 0x8002_0100], &setpc(10)]));
    // s_movrels_b32 can read any register.
    expect("escapes", build(&[&call(8, 1), &ENDPGM, &sop1(0, 0x2e, 0), &setpc(8)]));
    // A compare or a read of the link before the call is not an escape.
    let cfg = build(&[&[0xbf06_8008], &call(8, 1), &ENDPGM, &setpc(8)]).unwrap();
    assert_eq!(cfg.blocks[block_at(&cfg, 0)].exit, Exit::Branch(block_at(&cfg, 12)));
    let cfg = build(&[&sop1(0, 3, 8), &call(8, 1), &ENDPGM, &setpc(8)]).unwrap();
    assert_eq!(cfg.blocks[block_at(&cfg, 0)].exit, Exit::Branch(block_at(&cfg, 12)));
    // ... but reading it after the return, while it holds the return
    // address, is.
    expect("escapes", build(&[&sop1(0, 3, 8), &call(8, 2), &sopp(2, 0xfffd), &ENDPGM, &setpc(8)]));
    for link in [0, 104] {
        let cfg = build(&[&call(link, 1), &ENDPGM, &setpc(link)]).unwrap();
        assert!(matches!(cfg.blocks[block_at(&cfg, 0)].exit, Exit::Branch(_)));
    }
}
