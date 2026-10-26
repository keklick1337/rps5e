// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! v_fma_f64 and v_div_fmas_f64 (VCC set) round the exact result once for
//! subnormal operands and results: hardware-measured reference rows.

use ps5shader::interp::Machine;
use ps5shader::ssa::SystemValue;
use ps5shader::testing::{buffer_descriptor, ComputeRun};

include!("vectors/valu_f64_fma_subnormals.rs");

const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

/// One wave32: lane `t` reads a, b, c (two dwords each) from 8 input dwords
/// and writes a 64-bit result to 2 output dwords (raw descriptors).
fn run(code: &[u32], rows: &[[u64; 4]]) -> Vec<u64> {
    let mut input = vec![0u32; 32 * 8];
    for (tid, row) in rows.iter().enumerate() {
        for operand in 0..3 {
            input[tid * 8 + operand * 2] = row[operand] as u32;
            input[tid * 8 + operand * 2 + 1] = (row[operand] >> 32) as u32;
        }
    }
    let i = buffer_descriptor(IN_BASE, 0, input.len() as u32 * 4, 0x3101_6FAC);
    let o = buffer_descriptor(OUT_BASE, 0, 32 * 2 * 4, 0x3101_6FAC);
    let user: Vec<u32> = i.iter().chain(o.iter()).copied().collect();
    let run = ComputeRun::new(code.to_vec(), 32, 32, user);
    let setup = |m: &mut Machine| {
        m.memory.write_words(IN_BASE, &input);
        m.memory.write_words(OUT_BASE, &[0xDEAD_BEEF; 64]);
        m.system.insert(SystemValue::LocalInvocationId(0), (0..32).collect());
    };
    let m = run.run_checked(&setup, &[(OUT_BASE, 64)]).unwrap();
    let words = m.memory.read_words(OUT_BASE, 64);
    (0..32).map(|t| words[t * 2] as u64 | (words[t * 2 + 1] as u64) << 32).collect()
}

fn check(what: &str, code: &[u32], rows: &[[u64; 4]]) {
    let out = run(code, rows);
    let mut fails = Vec::new();
    for (lane, row) in rows.iter().enumerate() {
        if out[lane] != row[3] {
            fails.push(format!("{what}: lane {lane} ({:#018x}, {:#018x}, {:#018x}) is {:#018x}, expected {:#018x}", row[0], row[1], row[2], out[lane], row[3]));
        }
    }
    assert!(fails.is_empty(), "{} failures:\n{}", fails.len(), fails.join("\n"));
}

#[test]
fn fma_f64_subnormals() {
    check("v_fma_f64", &FMA_CODE, &FMA_ROWS);
}

#[test]
fn div_fmas_f64_vcc_subnormals() {
    check("v_div_fmas_f64 with VCC set", &FMAS_CODE, &FMAS_ROWS);
}
