// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! VOP1/VOP2 semantics against hardware-measured reference rows: control
//! forms, f32/f16 unary float ops, integer VOP1/VOP2 ops and the modifier
//! forms (DPP/SDWA/VOP3) of v_cndmask, v_mul_legacy and v_ldexp_f16.

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
fn dispatch(code: &[u32], wave: u32, threads: u32, input: &[u32], out_words: usize, raw: bool, mode: FloatMode) -> Result<Vec<u32>, String> {
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
/// dwords per lane, raw descriptors, one wave32.
fn run_rows(code: &[u32], rows: &[[u32; 4]], results: usize) -> Vec<u32> {
    run_rows_mode(code, rows, results, FloatMode::default())
}

/// `run_rows` for rows measured with other float controls.
fn run_rows_mode(code: &[u32], rows: &[[u32; 4]], results: usize, mode: FloatMode) -> Vec<u32> {
    let input: Vec<u32> = rows.iter().flatten().copied().collect();
    dispatch(code, 32, rows.len() as u32, &input, rows.len() * results, true, mode).unwrap()
}

/// Per-column failures of a measured table; `skip` columns are left to
/// their own (ignored) tests.
fn table_failures<const N: usize>(out: &[u32], results: usize, expected: &[[u32; N]], names: &[&str], checked: &dyn Fn(usize, usize) -> bool, skip: &[usize]) -> Vec<String> {
    let mut fails = Vec::new();
    for (lane, row) in expected.iter().enumerate() {
        for (col, &e) in row.iter().enumerate() {
            if skip.contains(&col) || !checked(lane, col) {
                continue;
            }
            let a = out[lane * results + col];
            if a != e {
                fails.push(format!("lane {lane} {} is {a:#010x}, expected {e:#010x}", names[col]));
            }
        }
    }
    fails
}

fn assert_clean(what: &str, fails: &[String]) {
    assert!(fails.is_empty(), "{what}: {} failures:\n{}", fails.len(), fails.join("\n"));
}

mod vop1_control {
    use super::*;
    include!("vectors/valu_vop1_control.rs");

    #[test]
    fn vop1_control() {
        let mut input = vec![0u32; (THREADS * STRIDE) as usize];
        for tid in 0..THREADS {
            input[(tid * STRIDE) as usize] = tid.wrapping_mul(0x0101_0101).wrapping_add(7);
        }
        let out = dispatch(&VOP1_CONTROL_CODE, 32, THREADS, &input, input.len(), false, FloatMode::default()).unwrap();
        let mut fails = Vec::new();
        for tid in 0..THREADS as usize {
            let e = input[tid * STRIDE as usize].wrapping_add(1);
            let a = out[tid * STRIDE as usize];
            if a != e {
                fails.push(format!("thread {tid} is {a:#x}, expected {e:#x}"));
            }
        }
        assert_clean("vop1 control", &fails);
    }
}

mod vop1_float_unary {
    use super::*;
    include!("vectors/valu_vop1_float_unary.rs");

    fn half(bits: u32) -> f64 {
        let exponent = ((bits >> 10) & 0x1F) as i32;
        let magnitude = if exponent == 0x1F {
            if bits & 0x3FF != 0 {
                f64::NAN
            } else {
                f64::INFINITY
            }
        } else {
            let m = ((bits & 0x3FF) | if exponent != 0 { 0x400 } else { 0 }) as f64;
            m * 2f64.powi((if exponent != 0 { exponent } else { 1 }) - 25)
        };
        if bits & 0x8000 != 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    fn finite(v: f64) -> bool {
        v.is_finite() && v != 0.0
    }

    /// The reference test's tolerance classes (exact unless listed).
    fn near(tid: usize, index: usize, actual: u32, expected: u32) -> bool {
        let is_half = TOLERANCE[index] >= 5;
        let (a, e) = if is_half { (half(actual & 0xFFFF), half(expected & 0xFFFF)) } else { (f32::from_bits(actual) as f64, f32::from_bits(expected) as f64) };
        if !finite(e) || (is_half && (actual >> 16) != (expected >> 16)) {
            return actual == expected;
        }
        let x = if is_half { half((ROWS[tid][2] >> if index % 2 == 1 { 16 } else { 0 }) & 0xFFFF) } else { f32::from_bits(ROWS[tid][index % 2]) as f64 };
        let error = (a - e).abs();
        let relative = error / e.abs();
        match TOLERANCE[index] {
            1 => relative <= 2f64.powi(-20),
            2 => error <= 2f64.powi(-21) || relative <= 2f64.powi(-20),
            3 => relative <= (4.0 + 2.0 * x.abs()) * 2f64.powi(-23),
            4 => error <= 2f64.powi(-11),
            5 => relative <= 2f64.powi(-9),
            _ => error <= 2f64.powi(-10),
        }
    }

    fn failures() -> Vec<String> {
        // Measured with FLOAT_MODE 0xF0 and IEEE_MODE.
        let mode = FloatMode { mode: 0xF0, dx10_clamp: true, ieee: true, fp16_overflow: false };
        let out = run_rows_mode(&CODE, &ROWS, RESULTS as usize, mode);
        let mut fails = Vec::new();
        for tid in 0..THREADS as usize {
            for i in 0..54 {
                if (CHECKED[tid] >> i) & 1 == 0 {
                    continue;
                }
                let (a, e) = (out[tid * RESULTS as usize + i], EXPECTED[tid][i]);
                let ok = if TOLERANCE[i] == 0 { a == e } else { near(tid, i, a, e) };
                if !ok {
                    fails.push(format!("lane {tid} {} ({:#010x}, {:#010x}, {:#010x}) is {a:#010x}, expected {e:#010x}", NAMES[i], ROWS[tid][0], ROWS[tid][1], ROWS[tid][2]));
                }
            }
        }
        fails
    }

    #[test]
    fn vop1_float_unary() {
        assert_clean("vop1 float unary", &failures());
    }
}

mod vop1_vop2_integer {
    use super::*;
    include!("vectors/valu_vop1_vop2_integer.rs");

    #[test]
    fn vop1_vop2_integer() {
        let out = run_rows(&CODE, &ROWS, RESULTS as usize);
        assert_clean("vop1 vop2 integer", &table_failures(&out, RESULTS as usize, &EXPECTED, &NAMES, &|_, _| true, &[]));
    }
}

mod vop2_modifier_forms {
    use super::*;
    include!("vectors/valu_vop2_modifier_forms.rs");

    /// v_ldexp_f16_sdwa v23 (dst_sel DWORD, dst_unused pad).
    const SDWA_LDEXP_F16: usize = 13;

    fn failures(skip: &[usize], only: Option<usize>) -> Vec<String> {
        let out = run_rows(&CODE, &ROWS, RESULTS as usize);
        let checked = |lane: usize, col: usize| (CHECKED[lane] >> col) & 1 != 0 && only.is_none_or(|o| o == col);
        table_failures(&out, RESULTS as usize, &EXPECTED, &NAMES, &checked, skip)
    }

    #[test]
    fn vop2_modifier_forms() {
        assert_clean("vop2 modifier forms", &failures(&[SDWA_LDEXP_F16], None));
    }

    #[test]
    fn sdwa_ldexp_f16_dword_destination() {
        assert_clean("vop2 modifier forms", &failures(&[], Some(SDWA_LDEXP_F16)));
    }
}

/// Distance in ULP between two float bit patterns of `width` bits.
fn ulp_distance(actual: u32, expected: u32, width: u32) -> u32 {
    let ordered = |bits: u32| {
        let sign = 1u32 << (width - 1);
        if bits & sign != 0 {
            sign - (bits & !sign)
        } else {
            bits | sign
        }
    };
    ordered(actual).abs_diff(ordered(expected))
}

mod sin_cos_near_zero {
    use super::*;
    include!("vectors/valu_sin_cos_near_zero.rs");

    /// Zero, infinite and NaN results are exact; the others within the
    /// lane's ULP tolerance (the host's sine of the radians away from the
    /// linear bands).
    #[test]
    fn sin_cos_near_zero() {
        let out = run_rows_mode(&CODE, &ROWS, 4, FloatMode { mode: 0xC0, dx10_clamp: true, ieee: false, fp16_overflow: false });
        let mut fails = Vec::new();
        for lane in 0..32 {
            for col in 0..4 {
                let width = if col < 2 { 32 } else { 16 };
                let mask = if width == 16 { 0xFFFF } else { u32::MAX };
                let (a, e) = (out[lane * 4 + col] & mask, EXPECTED[lane][col] & mask);
                let (magnitude, inf) = if width == 16 { (e & 0x7FFF, 0x7C00) } else { (e & 0x7FFF_FFFF, 0x7F80_0000) };
                let ok = if magnitude == 0 || magnitude >= inf { a == e } else { ulp_distance(a, e, width) <= TOLERANCE[lane] };
                if !ok {
                    fails.push(format!("lane {lane} {}({:#010x}) is {a:#010x}, expected {e:#010x} (tolerance {} ulp)", NAMES[col], ROWS[lane][col / 2], TOLERANCE[lane]));
                }
            }
        }
        assert_clean("sin cos near zero", &fails);
    }
}

mod vop3_add_sub_clamp {
    use super::*;
    include!("vectors/valu_vop3_add_sub_clamp.rs");

    const NAMES: [&str; 8] = [
        "v_add_nc_u32 v10, v4, v5 clamp",
        "v_sub_nc_u32 v11, v4, v5 clamp",
        "v_subrev_nc_u32 v12, v4, v5 clamp",
        "v_add_nc_u16 v13, v4, v5 clamp",
        "v_sub_nc_u16 v14, v4, v5 clamp",
        "v_add_nc_i16 v15, v4, v5 clamp",
        "v_sub_nc_i16 v16, v4, v5 clamp",
        "v_add_nc_u32 v17, v4, v5",
    ];

    /// VOP3 clamp saturates v_add/sub/subrev_nc_u32 and the 16-bit
    /// add/sub (signed for the i16 forms).
    #[test]
    fn vop3_add_sub_clamp() {
        let out = run_rows(&CODE, &ROWS, 16);
        assert_clean("vop3 add sub clamp", &table_failures(&out, 16, &EXPECTED, &NAMES, &|_, _| true, &[]));
    }
}

mod f16_fma_rounding {
    use super::*;
    include!("vectors/valu_f16_fma_rounding.rs");

    /// The f16 fused multiply-adds round the exact product-sum once.
    #[test]
    fn f16_fma_rounding() {
        let input: Vec<u32> = ROWS.iter().flatten().copied().collect();
        let out = dispatch(&CODE, 32, 64, &input, 64 * 8, true, FloatMode::default()).unwrap();
        let mut fails = Vec::new();
        for (lane, row) in EXPECTED.iter().enumerate() {
            for (k, &e) in row.iter().enumerate() {
                let a = out[lane * 8 + k] & MASKS[k];
                if a != e {
                    fails.push(format!("lane {lane} {:08x?} {}: {a:#010x}, expected {e:#010x}", ROWS[lane], NAMES[k]));
                }
            }
        }
        assert_clean("f16 fma rounding", &fails);
    }
}

mod sin_cos_reduction {
    use super::*;
    include!("vectors/valu_sin_cos_reduction.rs");

    /// Zero, infinite and NaN results are exact, the others within 2^-10 of
    /// the hardware's (relative). The reduction keeps the low bits of small
    /// negative inputs, which a floor-based fraction rounds away.
    #[test]
    fn sin_cos_reduction() {
        let input: Vec<u32> = ROWS.iter().map(|r| r[0]).collect();
        let out = dispatch(&CODE, 32, 32, &input, 64, false, FloatMode::default()).unwrap();
        let mut fails = Vec::new();
        for (lane, row) in ROWS.iter().enumerate() {
            for (k, name) in ["v_sin_f32", "v_cos_f32"].into_iter().enumerate() {
                let (a, e) = (out[lane * 2 + k], row[1 + k]);
                let value = f32::from_bits(e);
                let ok = if !value.is_finite() || value == 0.0 { a == e } else { (f32::from_bits(a) as f64 - value as f64).abs() <= value.abs() as f64 * 2f64.powi(-10) };
                if !ok {
                    fails.push(format!("lane {lane} {name}({:#010x}) is {a:#010x}, expected near {e:#010x}", row[0]));
                }
            }
        }
        assert_clean("sin cos reduction", &fails);
    }
}
