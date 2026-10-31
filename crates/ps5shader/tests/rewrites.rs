// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The equivalent rewrites of the translation (`translate::rewrites`):
//! every program gives the same words with and without them (hardware
//! rows and pseudo-random rows with special values), and the lean flush
//! makes a flush-heavy program smaller.

use ps5shader::testing::{buffer_descriptor, ComputeRun};
use ps5shader::testing::float_rows::*;
use ps5shader::translate::rewrites;

const INPUT: u64 = 0x1_0000_0000;
const OUTPUT: u64 = 0x2_0000_0000;

macro_rules! program {
    ($m:ident, $file:literal, $($code:ident: $wave:expr),+) => {
        #[allow(dead_code, unused_imports, clippy::all)]
        mod $m {
            include!(concat!("vectors/", $file));
            pub fn programs() -> Vec<(&'static [u32], u32)> {
                vec![$((&$code[..], $wave)),+]
            }
        }
    };
}

program!(denormal_flush, "valu_f32_denormal_flush.rs", CODE: 32, TINY_CODE: 32, MIXED_TINY_CODE: 32);
program!(float_unary, "valu_vop1_float_unary.rs", CODE: 32);
program!(dpp16_inactive, "fmod_dpp16_inactive_source.rs", WAVE32_CODE: 32, WAVE64_CODE: 64);
program!(dpp16_share, "fmod_dpp16_row_share_xmask.rs", WAVE32_CODE: 32, WAVE64_CODE: 64);
program!(dpp8, "fmod_dpp8.rs", WAVE32_CODE: 32, WAVE64_CODE: 64);
program!(dpp16_operands, "fmod_dpp16_operands.rs", CODE: 32);
program!(arithmetic_nan, "valu_f32_arithmetic_nan.rs", CODE: 32);
program!(min_max_nan, "valu_f32_min_max_nan.rs", CODE: 32);
program!(output_modifier, "valu_f32_output_modifier.rs", CODE: 32);
program!(result_clamp, "valu_f32_result_clamp.rs", CODE: 32);
program!(fma_legacy, "valu_fma_legacy.rs", FMA_LEGACY_CODE: 32, MAC_LEGACY_CODE: 32);
program!(fma_rounding, "valu_fma_rounding.rs", FMA_CODE: 32);
program!(modifier_forms, "valu_vop2_modifier_forms.rs", CODE: 32);
program!(legacy_mods, "valu_vop3_legacy_source_modifiers.rs", CODE: 32);
program!(mad_clamp, "valu_vop3_mad_clamp.rs", CODE: 32);
program!(pack_ldexp, "valu_vop3_pack_ldexp.rs", CODE: 32);
program!(aliases, "valu_vop3_aliases.rs", ALIAS_CODE: 32);
program!(mix_precision, "valu_mix_precision.rs", MIX_CODE: 32);
program!(compare_modifiers, "valu_vop3_compare_modifiers.rs", MODIFIER_CODE: 32);
program!(nan_quieting, "valu_nan_quieting.rs", CODE: 32);
program!(division_results, "fmod_division_result_modifiers.rs", CODE: 32);

/// Rows of four words: special f32 values, denormals and random bits.
fn rows(n: usize) -> Vec<u32> {
    const SPECIAL: [u32; 20] = [
        0, 0x8000_0000, 0x0000_0001, 0x8000_0001, 0x007F_FFFF, 0x807F_FFFF, 0x0080_0000, 0x8080_0000, 0x0040_0000, 0x3F80_0000, 0xBF80_0000, 0x7F80_0000,
        0xFF80_0000, 0x7FC0_0000, 0x7F80_0001, 0xFFFF_FFFF, 0x3F7F_FFFF, 0x1F80_0000, 0x0000_7FFF, 0x0012_3456,
    ];
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    (0..n * 4)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if x % 3 == 0 {
                SPECIAL[(x >> 8) as usize % SPECIAL.len()]
            } else {
                (x >> 20) as u32
            }
        })
        .collect()
}

/// Output words of one work-group of `wave` lanes (16-word-aligned
/// results, up to 64 per lane) translated with `bits`.
fn run(code: &[u32], wave: u32, input: &[u32], bits: u32) -> Vec<u32> {
    run_in(code, wave, input, bits, ps5shader::translate::FloatMode::default())
}

/// [`run`] with the float controls `mode`.
fn run_in(code: &[u32], wave: u32, input: &[u32], bits: u32, mode: ps5shader::translate::FloatMode) -> Vec<u32> {
    let out_words = wave as usize * 64;
    let mut user = buffer_descriptor(INPUT, 0, input.len() as u32 * 4, 0x3101_6FAC).to_vec();
    user.extend(buffer_descriptor(OUTPUT, 0, out_words as u32 * 4, 0x3101_6FAC));
    let mut run = ComputeRun::new(code.to_vec(), wave, wave, user);
    run.rewrites = bits;
    run.float_mode = mode;
    let f = run.compile(true).unwrap();
    let mut m = ps5shader::interp::Machine::new(wave, wave);
    m.user_data = run.user_data.clone();
    m.memory.write_words(INPUT, input);
    m.memory.write_words(OUTPUT, &vec![0xDEAD_BEEF; out_words]);
    m.memory.write_words(run.code_address, &run.code);
    m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(0), (0..wave).collect());
    m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(1), vec![0; wave as usize]);
    m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(2), vec![0; wave as usize]);
    m.run(&f).unwrap();
    m.memory.read_words(OUTPUT, out_words)
}

#[test]
fn rewrites_give_the_same_words() {
    let mut programs = Vec::new();
    for list in [denormal_flush::programs(), float_unary::programs(), dpp16_inactive::programs(), dpp16_share::programs(), dpp8::programs(), dpp16_operands::programs()] {
        programs.extend(list);
    }
    for (i, (code, wave)) in programs.iter().enumerate() {
        let input = rows(64);
        let all = run(code, *wave, &input, rewrites::ALL);
        for bits in [0, rewrites::LEAN_FLUSH, rewrites::LANE_ACTIVE_EXEC] {
            let other = run(code, *wave, &input, bits);
            let first = all.iter().zip(&other).position(|(a, b)| a != b);
            assert!(first.is_none(), "program {i} (wave{wave}) rewrites {bits:#x}: word {} differs: {:#010x} vs {:#010x}", first.unwrap(), all[first.unwrap()], other[first.unwrap()]);
        }
    }
}

#[test]
fn lean_flush_makes_flushes_smaller() {
    let (code, wave) = denormal_flush::programs()[0];
    let size = |bits: u32| {
        let run = ComputeRun { rewrites: bits, ..ComputeRun::new(code.to_vec(), wave, wave, vec![0; 8]) };
        let f = run.compile(true).unwrap();
        f.order.iter().map(|&b| f.block(b).insts.len()).sum::<usize>()
    };
    let (lean, full) = (size(rewrites::ALL), size(0));
    assert!(lean < full, "{lean} values with the lean flush, {full} without");
}

/// Every float program under every f32 denormal mode, round mode and
/// IEEE_MODE: the flush elision and the float class compares give the same
/// words bit for bit (NaN payloads and signed zeros included) as the
/// translation without them.
#[test]
fn float_rewrites_are_exact_in_every_float_mode() {
    use ps5shader::translate::FloatMode;
    let mut programs = vec![(&CHAIN_W32[..], 32u32)];
    for list in [
        denormal_flush::programs(),
        float_unary::programs(),
        arithmetic_nan::programs(),
        min_max_nan::programs(),
        output_modifier::programs(),
        result_clamp::programs(),
        fma_legacy::programs(),
        fma_rounding::programs(),
        modifier_forms::programs(),
        legacy_mods::programs(),
        mad_clamp::programs(),
        pack_ldexp::programs(),
        aliases::programs(),
        mix_precision::programs(),
        compare_modifiers::programs(),
        nan_quieting::programs(),
        division_results::programs(),
    ] {
        programs.extend(list);
    }
    let variants = [rewrites::ALL & !rewrites::FLUSH_ELISION, rewrites::ALL & !rewrites::FLOAT_CLASS_COMPARE, rewrites::ALL & !rewrites::NAN_ONCE, rewrites::LEAN_FLUSH | rewrites::LANE_ACTIVE_EXEC, 0];
    let mut fails = Vec::new();
    for (i, (code, wave)) in programs.iter().enumerate() {
        for mode in [0xC0u8, 0xF0, 0xD0, 0xE0, 0xC1, 0xC2, 0xC3] {
            for ieee in [false, true] {
                let fm = FloatMode { mode, dx10_clamp: true, ieee, fp16_overflow: false };
                for seed in 1..=3u64 {
                    let input = special_rows(*wave as usize * 16, seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    let all = run_in(code, *wave, &input, rewrites::ALL, fm);
                    for bits in variants {
                        let other = run_in(code, *wave, &input, bits, fm);
                        if let Some(w) = all.iter().zip(&other).position(|(a, b)| a != b) {
                            fails.push(format!("program {i} mode {mode:#x} ieee {ieee} rewrites {bits:#x} seed {seed}: word {w}: {:#010x} with all, {:#010x} without", all[w], other[w]));
                        }
                    }
                }
            }
        }
    }
    assert!(fails.is_empty(), "{} mismatches:\n{}", fails.len(), fails.iter().take(40).cloned().collect::<Vec<_>>().join("\n"));
}

/// The f16 fma-mix rounding through the f32 fma (HALF_FMA_FAST) gives the
/// same words as the f64 round-to-odd sum, at and around f16 midpoints,
/// in every f32 / f16 denormal mode and with FP16_OVFL and IEEE_MODE.
#[test]
fn half_fma_rounding_is_exact_at_midpoints() {
    use ps5shader::translate::FloatMode;
    let mut fails = Vec::new();
    let mut slow_words = 0usize;
    for mode in [0xC0u8, 0xF0, 0x30, 0x00, 0xC1, 0xC4] {
        for ieee in [false, true] {
            for fp16_overflow in [false, true] {
                let fm = FloatMode { mode, dx10_clamp: true, ieee, fp16_overflow };
                for seed in 1..=8u64 {
                    let input = midpoint_rows(32, seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
                    let all = run_in(&MIX_W32, 32, &input, rewrites::ALL, fm);
                    let other = run_in(&MIX_W32, 32, &input, rewrites::ALL & !rewrites::HALF_FMA_FAST, fm);
                    slow_words += all.len();
                    if let Some(w) = all.iter().zip(&other).position(|(a, b)| a != b) {
                        fails.push(format!("mode {mode:#x} ieee {ieee} ovfl {fp16_overflow} seed {seed}: word {w}: {:#010x} fast, {:#010x} f64 (inputs {:08x?})", all[w], other[w], &input[(w / 64) * 4..(w / 64) * 4 + 4]));
                    }
                }
            }
        }
    }
    assert!(slow_words > 0);
    assert!(fails.is_empty(), "{} mismatches:\n{}", fails.len(), fails.iter().take(40).cloned().collect::<Vec<_>>().join("\n"));
}

/// The midpoint rows reach the double-rounding cases: sums whose f32 fma
/// is an f16 midpoint while the exact sum is not.
#[test]
fn midpoint_rows_reach_double_rounding() {
    let mut double = 0;
    for seed in 1..=8u64 {
        let input = midpoint_rows(32, seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
        for lane in 0..32 {
            if lane % 4 == 3 {
                continue;
            }
            let row = &input[lane * 4..lane * 4 + 4];
            let a = half_value(row[0] & 0xFFFF);
            for c in [f32::from_bits(row[2]), f32::from_bits(row[3])] {
                let exact = a + c as f64;
                let fma = (a as f32).mul_add(1.0, c);
                // An f16 midpoint has a zero below its 12th significant bit
                // (and is at least 2^-25).
                let b = fma.to_bits() & 0x7FFF_FFFF;
                let e = b >> 23;
                let n = if e >= 113 { 13 } else { 126u32.saturating_sub(e).min(24) };
                let significand = (b & 0x7F_FFFF) | 0x80_0000;
                let mid = (102..=142).contains(&e) && significand & ((1 << n) - 1) == 1 << (n - 1);
                if mid && exact != fma as f64 {
                    double += 1;
                }
            }
        }
    }
    assert!(double >= 16, "{double} rows round twice");
}

/// The NaN choice of the f32 arithmetic results is off the results'
/// dependency chain with NAN_ONCE: the longest chain of selects through the
/// float chain program is shorter (one select per operation instead of one
/// per source plus one), so the host compiler schedules the arithmetic
/// without waiting for the NaN tests. Exactness: the test above.
#[test]
fn nan_once_shortens_the_select_chain() {
    use ps5shader::ssa::Op;
    let depth = |bits: u32| {
        let run = ComputeRun { rewrites: bits, ..ComputeRun::new(CHAIN_W32.to_vec(), 32, 32, vec![0; 8]) };
        let f = run.compile(true).unwrap();
        let mut d: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        let mut longest = 0;
        for &b in &f.order {
            for &v in &f.block(b).insts {
                let own = usize::from(f.op(v) == Op::Select);
                let deepest = f.args(v).iter().filter_map(|a| d.get(&a.0)).max().copied().unwrap_or(0);
                let n = deepest + own;
                d.insert(v.0, n);
                longest = longest.max(n);
            }
        }
        longest
    };
    let (once, chain) = (depth(rewrites::ALL), depth(rewrites::ALL & !rewrites::NAN_ONCE));
    assert!(once < chain, "select chain {once} with NAN_ONCE, {chain} without");
}
