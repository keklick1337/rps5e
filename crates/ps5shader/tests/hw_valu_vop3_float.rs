// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! VOP3 float semantics against hardware-measured reference rows: VOP3
//! encodings of VOP2/VOP1 aliases with modifiers, compare source
//! modifiers, cube ops with clamp/modifiers, and the legacy
//! v_mullit/v_cvt_pk_u8 source modifiers.

use ps5shader::interp::Machine;
use ps5shader::ssa::SystemValue;
use ps5shader::testing::{buffer_descriptor, ComputeRun};
use ps5shader::translate::FloatMode;

const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;
const POISON: u32 = 0xDEAD_BEEF;

/// One compute dispatch of `threads` invocations split into waves of
/// `wave`: s[0:3] input V#, s[4:7] output V#, v0 = local invocation id.
/// `raw` selects stride-0 descriptors sized in bytes, otherwise stride-4
/// descriptors sized in records. The output starts out as 0xdeadbeef.
fn dispatch(code: &[u32], wave: u32, threads: u32, input: &[u32], out_words: usize, raw: bool) -> Result<Vec<u32>, String> {
    dispatch_mode(code, wave, threads, input, out_words, raw, FloatMode::default())
}

/// `dispatch` compiled for the float controls `mode`.
fn dispatch_mode(code: &[u32], wave: u32, threads: u32, input: &[u32], out_words: usize, raw: bool, mode: FloatMode) -> Result<Vec<u32>, String> {
    let (i, o) = if raw {
        (buffer_descriptor(IN_BASE, 0, input.len() as u32 * 4, 0x3101_6FAC), buffer_descriptor(OUT_BASE, 0, out_words as u32 * 4, 0x3101_6FAC))
    } else {
        (buffer_descriptor(IN_BASE, 4, input.len() as u32, 0x1101_6FAC), buffer_descriptor(OUT_BASE, 4, out_words as u32, 0x1101_6FAC))
    };
    let user: Vec<u32> = i.iter().chain(o.iter()).copied().collect();
    let mut out = vec![POISON; out_words];
    let mut first = 0;
    while first < threads {
        let lanes = (threads - first).min(wave);
        let mut run = ComputeRun::new(code.to_vec(), wave, lanes, user.clone());
        run.float_mode = mode;
        let setup = |m: &mut Machine| {
            m.memory.write_words(IN_BASE, input);
            m.memory.write_words(OUT_BASE, &vec![POISON; out_words]);
            m.system.insert(SystemValue::LocalInvocationId(0), (first..first + lanes).collect());
        };
        let m = run.run_checked(&setup, &[(OUT_BASE, out_words)])?;
        for (k, w) in m.memory.read_words(OUT_BASE, out_words).into_iter().enumerate() {
            if w != POISON {
                out[k] = w;
            }
        }
        first += lanes;
    }
    Ok(out)
}

/// The table layout: one row of 4 input dwords per lane, `results` output
/// dwords per lane, raw descriptors, one wave32. Returns the failures of
/// the `checked` cells.
fn table<const N: usize>(code: &[u32], rows: &[[u32; 4]], results: usize, expected: &[[u32; N]], names: &[&str], checked: &dyn Fn(usize, usize) -> bool) -> Vec<String> {
    table_mode(code, rows, results, expected, names, checked, FloatMode::default())
}

/// `table` compiled for the float controls `mode`.
fn table_mode<const N: usize>(code: &[u32], rows: &[[u32; 4]], results: usize, expected: &[[u32; N]], names: &[&str], checked: &dyn Fn(usize, usize) -> bool, mode: FloatMode) -> Vec<String> {
    let input: Vec<u32> = rows.iter().flatten().copied().collect();
    let out = match dispatch_mode(code, 32, rows.len() as u32, &input, rows.len() * results, true, mode) {
        Ok(o) => o,
        Err(e) => return vec![e],
    };
    let mut fails = Vec::new();
    for (lane, row) in expected.iter().enumerate() {
        for (col, &e) in row.iter().enumerate() {
            if !checked(lane, col) {
                continue;
            }
            let a = out[lane * results + col];
            if a != e {
                let r = rows[lane];
                fails.push(format!("lane {lane} {} ({:#010x}, {:#010x}, {:#010x}, {:#010x}) is {a:#010x}, expected {e:#010x}", names[col], r[0], r[1], r[2], r[3]));
            }
        }
    }
    fails
}

fn assert_clean(what: &str, fails: &[String]) {
    assert!(fails.is_empty(), "{what}: {} failures:\n{}", fails.len(), fails.join("\n"));
}

fn half_value(bits: u32) -> f32 {
    let exponent = (bits >> 10) & 0x1F;
    let fraction = bits & 0x3FF;
    let magnitude = if exponent == 0x1F {
        if fraction != 0 {
            f32::NAN
        } else {
            f32::INFINITY
        }
    } else if exponent == 0 {
        fraction as f32 * 2f32.powi(-24)
    } else {
        (fraction | 0x400) as f32 * 2f32.powi(exponent as i32 - 25)
    };
    if bits & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// f16 bits of a value that is exactly a normal f16 (or +-0).
fn half_bits(value: f32) -> u32 {
    let b = value.to_bits();
    let sign = (b >> 16) & 0x8000;
    if value == 0.0 {
        return sign;
    }
    let exponent = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    assert!((1..31).contains(&exponent) && b & 0x1FFF == 0, "a model value is not a normal f16");
    sign | ((exponent as u32) << 10) | ((b >> 13) & 0x3FF)
}

fn quarter(index: u32) -> f32 {
    ((index % 17) as i32 - 8) as f32 / 4.0
}

fn pack(low: f32, high: f32) -> u32 {
    half_bits(low) | (half_bits(high) << 16)
}

fn low(word: u32) -> f32 {
    half_value(word & 0xFFFF)
}

mod vop3_aliases {
    use super::*;
    include!("vectors/valu_vop3_aliases.rs");

    #[test]
    fn results() {
        let mut input = vec![0u32; (THREADS * INPUTS) as usize];
        for tid in 0..THREADS {
            let w = &mut input[(tid * INPUTS) as usize..];
            w[0] = pack(quarter(tid), quarter(tid * 3 + 5));
            w[1] = pack(quarter(tid * 5 + 2), quarter(tid * 7 + 11));
            w[2] = tid.wrapping_mul(0x0103_0507) ^ 0x80C0_E0F0;
        }
        let out = dispatch(&ALIAS_CODE, 32, THREADS, &input, (THREADS * RESULTS) as usize, false).unwrap();
        let mut fails = Vec::new();
        for tid in 0..THREADS as usize {
            let i = &input[tid * INPUTS as usize..];
            let o = &out[tid * RESULTS as usize..];
            let byte = |index: u32| ((i[2] >> (index * 8)) & 0xFF) as f32;
            let halves = [(-low(i[0])).max(low(i[1]).abs()), low(i[0]).abs().min(-low(i[1])), -low(i[0]).abs() * low(i[1])];
            let floats = [low(i[1]) * 4.0, -low(i[0]).abs(), byte(0) * 2.0, byte(1) / 2.0, byte(2).min(1.0), byte(3)];
            for (j, &e) in halves.iter().enumerate() {
                if low(o[j]) != e {
                    fails.push(format!("thread {tid} result {j} is {:#x} ({}), expected {e}", o[j], low(o[j])));
                }
            }
            for (j, &e) in floats.iter().enumerate() {
                let a = f32::from_bits(o[halves.len() + j]);
                if a != e {
                    fails.push(format!("thread {tid} result {} is {a}, expected {e}", halves.len() + j));
                }
            }
        }
        assert_clean("vop3 aliases", &fails);
    }
}

mod vop3_compare_modifiers {
    use super::*;
    include!("vectors/valu_vop3_compare_modifiers.rs");

    #[test]
    fn results() {
        let mut input = vec![0u32; (THREADS * INPUTS) as usize];
        let (nf, nh) = (FLOAT_EDGES.len() as u32, HALF_EDGES.len() as u32);
        for tid in 0..THREADS {
            let w = &mut input[(tid * INPUTS) as usize..];
            w[0] = FLOAT_EDGES[(tid % nf) as usize];
            w[1] = FLOAT_EDGES[((tid / nf + tid * 7) % nf) as usize] ^ if tid & 4 != 0 { 0x8000_0000 } else { 0 };
            w[2] = 0xABCD_0000 | HALF_EDGES[(tid % nh) as usize] as u32;
            w[3] = 0x1234_0000 | (HALF_EDGES[((tid / nh + tid * 5) % nh) as usize] as u32 ^ if tid & 8 != 0 { 0x8000 } else { 0 });
        }
        let out = dispatch(&MODIFIER_CODE, 32, THREADS, &input, (THREADS * RESULTS) as usize, false).unwrap();
        let mut fails = Vec::new();
        for tid in 0..THREADS as usize {
            let i = &input[tid * INPUTS as usize..];
            let (fa, fb) = (f32::from_bits(i[0]), f32::from_bits(i[1]));
            let (ha, hb) = (half_value(i[2] & 0xFFFF), half_value(i[3] & 0xFFFF));
            let expected = [
                -fa < fb,
                -fa.abs() > -fb,
                !(fa >= -fb.abs()),
                -fa == fb,
                -ha == hb,
                -ha.abs() < -hb,
                !(ha == -hb),
                -fa <= fb,
                -ha > hb.abs(),
            ];
            for (k, &e) in expected.iter().enumerate() {
                let actual = (out[tid * RESULTS as usize] >> k) & 1 != 0;
                if actual != e {
                    fails.push(format!("thread {tid} ({:#x}, {:#x}, {:#x}, {:#x}) compare {k} is {actual}, expected {e}", i[0], i[1], i[2], i[3]));
                }
            }
        }
        assert_clean("vop3 compare modifiers", &fails);
    }
}

mod vop3_cube_clamp {
    use super::*;
    include!("vectors/valu_vop3_cube_clamp.rs");

    #[test]
    fn rows() {
        assert_clean("vop3 cube clamp", &table(&CODE, &ROWS, RESULTS as usize, &EXPECTED, &NAMES, &|_, _| true));
    }

    include!("vectors/valu_vop3_cube_clamp_ieee.rs");

    /// With IEEE_MODE a signaling NaN cube result is quieted (its sign
    /// kept); with IEEE_MODE clear the bits pass through.
    #[test]
    fn rows_ieee() {
        let ieee = FloatMode { mode: 0xC0, dx10_clamp: true, ieee: true, fp16_overflow: false };
        assert_clean("vop3 cube clamp, IEEE", &table_mode(&CODE, &ROWS, RESULTS as usize, &EXPECTED_IEEE, &NAMES, &|_, _| true, ieee));
    }
}

mod vop3_cube_omod {
    use super::*;
    include!("vectors/valu_vop3_cube_omod.rs");

    /// The cube ops apply their output modifier whenever IEEE_MODE is
    /// clear, whatever the f32 denormal mode, and ignore it with IEEE_MODE.
    #[test]
    fn rows() {
        let mut fails = Vec::new();
        for (mode, ieee, name) in [(0xC0, false, "IEEE=0 f32 denormals flushed"), (0xF0, false, "IEEE=0 f32 denormals kept"), (0xC0, true, "IEEE=1 f32 denormals flushed"), (0xF0, true, "IEEE=1 f32 denormals kept")] {
            let fm = FloatMode { mode, dx10_clamp: true, ieee, fp16_overflow: false };
            let expected = if ieee { &OMOD_EXPECTED_IEEE } else { &OMOD_EXPECTED };
            fails.extend(table_mode(&OMOD_CODE, &OMOD_ROWS, 16, expected, &OMOD_NAMES, &|_, _| true, fm).into_iter().map(|f| format!("{name}: {f}")));
        }
        assert_clean("vop3 cube output modifiers", &fails);
    }
}

mod pack_b32_f16_nan {
    use super::*;
    include!("vectors/valu_pack_b32_f16_nan.rs");

    const NAMES: [&str; 4] = ["v_pack_b32_f16 v10, v4, v5", "v_pack_b32_f16 v11, v4, v5 op_sel:[1,1,0]", "v_pack_b32_f16 v12, -v4, |v5|", "v_pack_b32_f16 v13, |v5|, -v4 op_sel:[1,0,0]"];

    /// v_pack_b32_f16 copies NaN halves with IEEE_MODE clear and quiets
    /// them (after abs / neg) with IEEE_MODE.
    #[test]
    fn ieee_off() {
        let mode = FloatMode { mode: 0xC0, dx10_clamp: true, ieee: false, fp16_overflow: false };
        assert_clean("pack_b32_f16, IEEE=0", &table_mode(&PACK_CODE, &PACK_ROWS, 16, &PACK_EXPECTED, &NAMES, &|_, _| true, mode));
    }

    #[test]
    fn ieee_on() {
        let mode = FloatMode { mode: 0xC0, dx10_clamp: true, ieee: true, fp16_overflow: false };
        assert_clean("pack_b32_f16, IEEE=1", &table_mode(&PACK_CODE, &PACK_ROWS, 16, &PACK_EXPECTED_IEEE, &NAMES, &|_, _| true, mode));
    }
}

mod ldexp_f16_inline_constants {
    use super::*;
    include!("vectors/valu_ldexp_f16_inline_constants.rs");

    const NAMES: [&str; 16] = ["0.5", "-0.5", "1.0", "-1.0", "2.0", "-2.0", "4.0", "-4.0", "0.15915494", "2", "-1", "64", "-16", "literal 0x3f000000", "s8 = 0x3f000000", "v5 = 0x3800"];

    /// The VOP3 v_ldexp_f16 exponent is an f16 operand: a float inline
    /// constant is its f16 encoding; registers and literals give their low
    /// 16 bits.
    #[test]
    fn rows() {
        let input: Vec<u32> = LDEXP_ROWS.iter().map(|r| r[0]).collect();
        let mode = FloatMode { mode: 0xC0, dx10_clamp: true, ieee: false, fp16_overflow: false };
        let out = dispatch_mode(&LDEXP_CODE, 32, 32, &input, 32 * 16, false, mode).unwrap();
        let mut fails = Vec::new();
        for (lane, row) in LDEXP_ROWS.iter().enumerate() {
            for col in 0..16 {
                let a = out[lane * 16 + col] & 0xFFFF;
                if a != row[1 + col] {
                    fails.push(format!("v_ldexp_f16 {:#06x}, {} is {a:#06x}, expected {:#06x}", row[0], NAMES[col], row[1 + col]));
                }
            }
        }
        assert_clean("ldexp f16 inline constants", &fails);
    }
}

mod vop3_legacy_source_modifiers {
    use super::*;
    include!("vectors/valu_vop3_legacy_source_modifiers.rs");

    #[test]
    fn rows() {
        let checked = |lane: usize, col: usize| (CHECKED[lane] >> col) & 1 != 0;
        assert_clean("vop3 legacy source modifiers", &table(&CODE, &ROWS, RESULTS as usize, &EXPECTED, &NAMES, &checked));
    }
}
