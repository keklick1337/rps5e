// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! s_subvector_loop_begin / s_subvector_loop_end in wave64: the body runs
//! once per half of the wave that has lanes in EXEC (the low half first),
//! with EXEC narrowed to that half; s38 holds the other half's mask in the
//! pass and EXEC is restored after the loop. Two programs (the body keeps
//! EXEC, the body resets it to all lanes) over seven initial EXEC masks,
//! the expectations of the reference's SubvectorLoop execution test.
//! Columns: passes, last pass, s38 in the pass, exec lo / hi in the pass,
//! s38 after the loop, exec lo / hi after the loop, total passes.

use ps5shader::interp::Machine;
use ps5shader::testing::{raw_buffer_descriptor, ComputeRun};

const LANES: usize = 64;
const COLUMNS: usize = 9;
const ONES: u32 = 0xffff_ffff;
const MARKER: u32 = 0x5a5a_5a5a;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const EXEC_KEPT: [u32; 49] = [
    0x34020082, 0xe0301000, 0x80000401, 0xbf8c3f70, 0x7e140280, 0x7e160280, 0x7e180280, 0x7e1a0280, 0x7e1c0280, 0xbe940380, 0xbea603ff, 0x5a5a5a5a, 0x7daa0880, 0xbda6000a,
    0x80148114, 0x4a141481, 0x7e160214, 0x7e180226, 0xbe96047e, 0x7e1a0216, 0x7e1c0217, 0xbf098414, 0xbf850001, 0xbe26fff6, 0xbe98047e, 0xbefe04c1, 0x7e1e0226, 0x7e200218,
    0x7e220219, 0x7e240214, 0xe0701000, 0x80010a01, 0xe0701100, 0x80010b01, 0xe0701200, 0x80010c01, 0xe0701300, 0x80010d01, 0xe0701400, 0x80010e01, 0xe0701500, 0x80010f01,
    0xe0701600, 0x80011001, 0xe0701700, 0x80011101, 0xe0701800, 0x80011201, 0xbf810000,
];

const EXEC_RESET: [u32; 50] = [
    0x34020082, 0xe0301000, 0x80000401, 0xbf8c3f70, 0x7e140280, 0x7e160280, 0x7e180280, 0x7e1a0280, 0x7e1c0280, 0xbe940380, 0xbea603ff, 0x5a5a5a5a, 0x7daa0880, 0xbda6000b,
    0x80148114, 0x4a141481, 0x7e160214, 0x7e180226, 0xbe96047e, 0x7e1a0216, 0x7e1c0217, 0xbf098414, 0xbf850002, 0xbefe04c1, 0xbe26fff5, 0xbe98047e, 0xbefe04c1, 0x7e1e0226,
    0x7e200218, 0x7e220219, 0x7e240214, 0xe0701000, 0x80010a01, 0xe0701100, 0x80010b01, 0xe0701200, 0x80010c01, 0xe0701300, 0x80010d01, 0xe0701400, 0x80010e01, 0xe0701500,
    0x80010f01, 0xe0701600, 0x80011001, 0xe0701700, 0x80011101, 0xe0701800, 0x80011201, 0xbf810000,
];

/// Lanes `first..=last` expect `values` (later groups override).
type Group = (usize, usize, [u32; COLUMNS]);

fn exec_kept_cases() -> Vec<(&'static str, u64, Vec<Group>)> {
    vec![
        ("all lanes", !0, vec![(0, 31, [1, 1, ONES, ONES, 0, ONES, ONES, ONES, 2]), (32, 63, [1, 2, ONES, 0, ONES, ONES, ONES, ONES, 2])]),
        ("low half", 0xffff_ffff, vec![(0, 31, [1, 1, 0, ONES, 0, 0, ONES, 0, 1]), (32, 63, [0, 0, 0, 0, 0, 0, ONES, 0, 1])]),
        ("high half", 0xffff_ffff << 32, vec![(0, 31, [0, 0, 0, 0, 0, 0, 0, ONES, 1]), (32, 63, [1, 1, 0, 0, ONES, 0, 0, ONES, 1])]),
        ("no lane", 0, vec![(0, 63, [0, 0, 0, 0, 0, MARKER, 0, 0, 0])]),
        (
            "lanes 3 and 40",
            (1 << 3) | (1 << 40),
            vec![(0, 63, [0, 0, 0, 0, 0, 8, 8, 0x100, 2]), (3, 3, [1, 1, 0x100, 8, 0, 8, 8, 0x100, 2]), (40, 40, [1, 2, 8, 0, 0x100, 8, 8, 0x100, 2])],
        ),
        ("lane 3", 1 << 3, vec![(0, 63, [0, 0, 0, 0, 0, 0, 8, 0, 1]), (3, 3, [1, 1, 0, 8, 0, 0, 8, 0, 1])]),
        ("lane 40", 1 << 40, vec![(0, 63, [0, 0, 0, 0, 0, 0, 0, 0x100, 1]), (40, 40, [1, 1, 0, 0, 0x100, 0, 0, 0x100, 1])]),
    ]
}

fn exec_reset_cases() -> Vec<(&'static str, u64, Vec<Group>)> {
    vec![
        ("all lanes", !0, vec![(0, 63, [0, 0, 0, 0, 0, ONES, ONES, ONES, 1]), (0, 31, [1, 1, ONES, ONES, 0, ONES, ONES, ONES, 1])]),
        ("low half", 0xffff_ffff, vec![(0, 63, [0, 0, 0, 0, 0, 0, 0, ONES, 1]), (0, 31, [1, 1, 0, ONES, 0, 0, 0, ONES, 1])]),
        ("high half", 0xffff_ffff << 32, vec![(0, 63, [0, 0, 0, 0, 0, 0, 0, ONES, 1]), (32, 63, [1, 1, 0, 0, ONES, 0, 0, ONES, 1])]),
        ("no lane", 0, vec![(0, 63, [0, 0, 0, 0, 0, MARKER, 0, 0, 0])]),
        ("lanes 3 and 40", (1 << 3) | (1 << 40), vec![(0, 63, [0, 0, 0, 0, 0, 0x100, 0x100, ONES, 1]), (3, 3, [1, 1, 0x100, 8, 0, 0x100, 0x100, ONES, 1])]),
        ("lane 3", 1 << 3, vec![(0, 63, [0, 0, 0, 0, 0, 0, 0, ONES, 1]), (3, 3, [1, 1, 0, 8, 0, 0, 0, ONES, 1])]),
        ("lane 40", 1 << 40, vec![(0, 63, [0, 0, 0, 0, 0, 0, 0, ONES, 1]), (40, 40, [1, 1, 0, 0, 0x100, 0, 0, ONES, 1])]),
    ]
}

fn run(code: &[u32], mask: u64) -> Vec<u32> {
    let input: Vec<u32> = (0..LANES).map(|l| (mask >> l & 1) as u32).collect();
    let out_words = LANES * COLUMNS;
    let mut user = raw_buffer_descriptor(IN_BASE, LANES as u32 * 4).to_vec();
    user.extend(raw_buffer_descriptor(OUT_BASE, out_words as u32 * 4));
    let run = ComputeRun::new(code.to_vec(), 64, LANES as u32, user);
    let setup = |m: &mut Machine| {
        m.memory.write_words(IN_BASE, &input);
        m.memory.write_words(OUT_BASE, &vec![0xdead_beef; out_words]);
    };
    let m = run.run_checked(&setup, &[(OUT_BASE, out_words)]).unwrap();
    m.memory.read_words(OUT_BASE, out_words)
}

fn check(variant: &str, code: &[u32], cases: Vec<(&'static str, u64, Vec<Group>)>) -> Vec<String> {
    let mut failures = Vec::new();
    for (name, mask, groups) in cases {
        let out = run(code, mask);
        let mut expected = [[0u32; COLUMNS]; LANES];
        for (first, last, values) in groups {
            for e in &mut expected[first..=last] {
                *e = values;
            }
        }
        for (lane, want) in expected.iter().enumerate() {
            for (column, &w) in want.iter().enumerate() {
                let got = out[column * LANES + lane];
                if got != w {
                    failures.push(format!("{variant}, initial exec {name}: lane {lane} column {column} is {got:#x}, expected {w:#x}"));
                }
            }
        }
    }
    failures
}

#[test]
fn subvector_loop_body_keeps_exec() {
    let failures = check("exec kept by the body", &EXEC_KEPT, exec_kept_cases());
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.iter().take(30).cloned().collect::<Vec<_>>().join("\n"));
}

#[test]
fn subvector_loop_body_resets_exec() {
    let failures = check("body resets exec", &EXEC_RESET, exec_reset_cases());
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.iter().take(30).cloned().collect::<Vec<_>>().join("\n"));
}
