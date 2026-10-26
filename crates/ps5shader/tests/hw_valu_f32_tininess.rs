// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The tininess of flushed f32 products (v_mul_f32, v_fma_f32 with a zero
//! addend) to nearest even, decided by one scaled multiply
//! (`float_rules::TINY_ONE_MULTIPLY`), against the integer significand
//! check it replaces: the same results on products around 2^-126, and
//! fewer instructions (the reference's vector_constant_factor_tininess
//! pins the count).

include!("vectors/fmod_common.rs");

use ps5shader::translate::float_rules::{ALL, TINY_ONE_MULTIPLY};

/// v_mul_f32 v10, v4, v5; v_fma_f32 v11, v4, v5, v6; rows of four
/// inputs, four results.
const CODE: [u32; 13] = [
    0x34020084, 0x34040084, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x10140b04, 0xd54b000b, 0x041a0b04, 0x7e180280, 0x7e1a0280, 0xe0781000, 0x80010a02,
    0xbf810000,
];

fn run(rules: u32, input: &[u32], rows: usize) -> (Vec<u32>, usize) {
    let mut user = vec![0u32; 8];
    user[..4].copy_from_slice(&fmod_descriptor(FMOD_INPUT_BASE, input.len(), 0));
    user[4..].copy_from_slice(&fmod_descriptor(FMOD_OUTPUT_BASE, rows * 4, 0));
    let mut out = Vec::with_capacity(rows * 4);
    let mut count = 0;
    for chunk in 0..rows.div_ceil(32) {
        let lanes = (rows - chunk * 32).min(32) as u32;
        let mut run = ComputeRun::new(CODE.to_vec(), 32, lanes, user.clone());
        run.float_rules = rules;
        run.user_data[0] = (FMOD_INPUT_BASE + chunk as u64 * 32 * 16) as u32;
        run.user_data[4] = (FMOD_OUTPUT_BASE + chunk as u64 * 32 * 16) as u32;
        let f = run.compile(true).unwrap_or_else(|e| panic!("{e}"));
        count = f.instruction_count();
        let piece = &input[chunk * 32 * 4..(chunk * 32 + lanes as usize) * 4];
        let mut memory = ps5shader::interp::Memory::default();
        memory.write_words(run.user_data[0] as u64 | (FMOD_INPUT_BASE & !0xFFFF_FFFF), piece);
        let mut m = ps5shader::interp::Machine::new(32, lanes);
        m.user_data = run.user_data.clone();
        memory.write_words(run.code_address, &run.code);
        m.memory = memory;
        m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(0), (0..lanes).collect());
        m.run(&f).unwrap_or_else(|e| panic!("run: {e}"));
        out.extend(m.memory.read_words(run.user_data[4] as u64 | (FMOD_OUTPUT_BASE & !0xFFFF_FFFF), lanes as usize * 4));
    }
    (out, count)
}

/// xorshift64*.
fn next(state: &mut u64) -> u32 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
}

#[test]
fn one_multiply_tininess_matches_the_significand_check() {
    let mut rows: Vec<[u32; 4]> = Vec::new();
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    // Products whose exponent sum lands around the smallest normal, with
    // significands near the rounding thresholds and random ones.
    for _ in 0..20000 {
        let ea = 1 + next(&mut state) % 253;
        let d = (next(&mut state) % 4) as i32 - 2;
        let eb = (128 - ea as i32 + d).clamp(1, 254) as u32;
        let ma = match next(&mut state) % 4 {
            0 => 0x7F_FFFF - next(&mut state) % 4,
            1 => next(&mut state) % 4,
            _ => next(&mut state) & 0x7F_FFFF,
        };
        let mb = match next(&mut state) % 3 {
            0 => 0x7F_FFFF - next(&mut state) % 4,
            _ => next(&mut state) & 0x7F_FFFF,
        };
        let sa = next(&mut state) & 0x8000_0000;
        let sb = next(&mut state) & 0x8000_0000;
        let a = sa | ea << 23 | ma;
        let b = sb | eb << 23 | mb;
        let c = if next(&mut state) % 4 == 0 { 0x0080_0000 } else { next(&mut state) & 0x8000_0000 };
        rows.push([a, b, c, 0]);
    }
    // Exact pairs just below, at and above 2^-126 and specials.
    for &(a, b) in &[
        (0x3f7f_ffff, 0x0080_0001),
        (0x3f7f_fffe, 0x0080_0001),
        (0x3f7f_ffff, 0x0080_0000),
        (0x3f80_0000, 0x0080_0000),
        (0x3f00_0000, 0x0100_0000),
        (0x3f00_0000, 0x00ff_ffff),
        (0x7f80_0000, 0x0000_0000),
        (0x7fc0_0000, 0x0080_0000),
        (0x5f80_0000, 0x0080_0000),
        (0x7f7f_ffff, 0x0000_0001),
        (0x0000_0001, 0x3f80_0000),
        (0x2000_0000, 0x2000_0000),
        (0xa000_0000, 0x1f80_0000),
    ] {
        rows.push([a, b, 0, 0]);
        rows.push([b, a ^ 0x8000_0000, 0x8000_0000, 0]);
    }
    let input: Vec<u32> = rows.iter().flatten().copied().collect();
    let (one, one_count) = run(ALL, &input, rows.len());
    let (old, old_count) = run(ALL & !TINY_ONE_MULTIPLY, &input, rows.len());
    let mut failures = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        for (k, name) in [(0, "v_mul_f32"), (1, "v_fma_f32")] {
            if one[i * 4 + k] != old[i * 4 + k] {
                failures.push(format!("{name} {:#010x} * {:#010x} (+ {:#010x}): one multiply {:#010x}, significand check {:#010x}", r[0], r[1], r[2], one[i * 4 + k], old[i * 4 + k]));
            }
        }
    }
    fmod_report(&failures);
    assert!(one_count + 20 < old_count, "one multiply: {one_count} instructions, significand check: {old_count}");
}
