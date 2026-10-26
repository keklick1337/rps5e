// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! v_cvt_i16_f16 with output modifiers (mul:2, mul:4, div:2, clamp, clamp
//! mul:2) in the VOP3 encoding: the hardware ignores them (measured on
//! gfx1036 in FLOAT_MODE 0xF0 with IEEE and 0xC0 without), truncates
//! toward zero, saturates to the i16 range, sends NaN to 0 and keeps the
//! destination's high half. Rows of the reference's
//! VCvtI16F16OutputModifiers execution test.

use ps5shader::interp::Machine;
use ps5shader::testing::{raw_buffer_descriptor, ComputeRun};
use ps5shader::translate::FloatMode;

const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;
const LANES: usize = 16;

const ROWS: [u32; LANES] = [
    0x00003c00, 0x0000bc00, 0x0000c100, 0x00004200, 0x00006400, 0x00005640, 0x0000d640, 0x00007bff, 0x0000fbff, 0x00007e00, 0xbeef3c00, 0xbeefc100, 0x00007c00, 0x0000fc00,
    0x00007c01, 0x00003e00,
];

const EXPECTED: [u32; LANES] = [
    0x5a5a0001, 0x5a5affff, 0x5a5afffe, 0x5a5a0003, 0x5a5a0400, 0x5a5a0064, 0x5a5aff9c, 0x5a5a7fff, 0x5a5a8000, 0x5a5a0000, 0x5a5a0001, 0x5a5afffe, 0x5a5a7fff, 0x5a5a8000,
    0x5a5a0000, 0x5a5a0001,
];

/// v1 = tid * 16 (input), v3 = tid * 16 (output); v4 = input word;
/// v10 = 0x5a5a5a5a; v_cvt_i16_f16 v10, v4 (clamp / omod); store v10.
fn code(clamp: u32, omod: u32) -> Vec<u32> {
    vec![0x34020084, 0x34060084, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x7e1402ff, 0x5a5a5a5a, 0xd5d3000a | clamp << 15, 0x00000104 | omod << 27, 0xe0781000, 0x80010a03, 0xbf810000]
}

fn run(code: Vec<u32>, mode: FloatMode) -> Vec<u32> {
    let mut input = vec![0u32; LANES * 4];
    for (lane, &r) in ROWS.iter().enumerate() {
        input[lane * 4] = r;
    }
    let mut user = raw_buffer_descriptor(IN_BASE, input.len() as u32 * 4).to_vec();
    user.extend(raw_buffer_descriptor(OUT_BASE, LANES as u32 * 16));
    let mut run = ComputeRun::new(code, 32, LANES as u32, user);
    run.float_mode = mode;
    let setup = |m: &mut Machine| {
        m.memory.write_words(IN_BASE, &input);
        m.memory.write_words(OUT_BASE, &vec![0xdead_beef; LANES * 4]);
    };
    let m = run.run_checked(&setup, &[(OUT_BASE, LANES * 4)]).unwrap();
    m.memory.read_words(OUT_BASE, LANES * 4).chunks(4).map(|w| w[0]).collect()
}

#[test]
fn output_modifiers_are_ignored() {
    let modes = [("M1", FloatMode { mode: 0xF0, dx10_clamp: true, ieee: true, fp16_overflow: false }), ("M2", FloatMode { mode: 0xC0, dx10_clamp: true, ieee: false, fp16_overflow: false })];
    let variants = [("mul:2", 0, 1), ("mul:4", 0, 2), ("div:2", 0, 3), ("clamp", 1, 0), ("clamp mul:2", 1, 1)];
    let mut failures = Vec::new();
    for (name, clamp, omod) in variants {
        for (m, mode) in modes {
            for (lane, got) in run(code(clamp, omod), mode).into_iter().enumerate() {
                if got != EXPECTED[lane] {
                    failures.push(format!("v_cvt_i16_f16 {name} {m}: lane {lane} of {:#010x} is {got:#010x}, expected {:#010x}", ROWS[lane], EXPECTED[lane]));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}
