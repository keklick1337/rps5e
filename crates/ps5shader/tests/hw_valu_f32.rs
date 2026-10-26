// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! f32 arithmetic semantics against hardware-measured reference rows:
//! v_fma_legacy_f32, fused vs. separate rounding of the fma/mad family,
//! min/max/min3/max3/med3 with NaN and signed zero, output modifiers
//! (omod) and result clamping.

use ps5shader::interp::Machine;
use ps5shader::ssa::SystemValue;
use ps5shader::testing::{buffer_descriptor, ComputeRun};

const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;
const POISON: u32 = 0xDEAD_BEEF;

/// One compute dispatch of `threads` invocations split into waves of
/// `wave`: s[0:3] input V#, s[4:7] output V#, v0 = local invocation id.
/// `raw` selects stride-0 descriptors sized in bytes, otherwise stride-4
/// descriptors sized in records. The output starts out as 0xdeadbeef.
fn dispatch(code: &[u32], wave: u32, threads: u32, input: &[u32], out_words: usize, raw: bool) -> Result<Vec<u32>, String> {
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
        let run = ComputeRun::new(code.to_vec(), wave, lanes, user.clone());
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

fn assert_clean(what: &str, fails: &[String]) {
    assert!(fails.is_empty(), "{what}: {} failures:\n{}", fails.len(), fails.join("\n"));
}

fn is_nan(bits: u32) -> bool {
    bits & 0x7FFF_FFFF > 0x7F80_0000
}

/// The table layout: one row of 4 input dwords per lane, `results` output
/// dwords per lane, raw descriptors, one wave32. Checks the `checked`
/// cells exactly and the `any_nan` cells for a NaN.
#[allow(clippy::too_many_arguments)]
fn table<const N: usize>(code: &[u32], rows: &[[u32; 4]], results: usize, expected: &[[u32; N]], names: &[&str], checked: &dyn Fn(usize, usize) -> bool, any_nan: &dyn Fn(usize, usize) -> bool, first_row: usize) -> Vec<String> {
    let input: Vec<u32> = rows.iter().flatten().copied().collect();
    let out = match dispatch(code, 32, rows.len() as u32, &input, rows.len() * results, true) {
        Ok(o) => o,
        Err(e) => return vec![e],
    };
    let mut fails = Vec::new();
    for (lane, row) in expected.iter().enumerate() {
        for (col, &e) in row.iter().enumerate() {
            let a = out[lane * results + col];
            let r = rows[lane];
            let at = format!("row {} {} ({:#010x}, {:#010x}, {:#010x})", first_row + lane, names[col], r[0], r[1], r[2]);
            if checked(lane, col) && a != e {
                fails.push(format!("{at} is {a:#010x}, expected {e:#010x}"));
            }
            if any_nan(lane, col) && !is_nan(a) {
                fails.push(format!("{at} is {a:#010x}, expected a NaN"));
            }
        }
    }
    fails
}

mod fma_legacy {
    use super::*;
    include!("vectors/valu_fma_legacy.rs");

    const NAMES: [&str; 5] = ["v_mad_legacy_f32", "v_mad_legacy_f32 -src0", "v_mad_legacy_f32 |src0| -src2", "v_mad_legacy_f32 src0=0", "v_mad_legacy_f32 -src0 -|src1|"];
    const MAC_NAMES: [&str; 4] = ["v_mac_legacy_f32", "v_mac_legacy_f32_e64 -src0", "v_mac_legacy_f32_e64 |src0| -|src1|", "v_mac_legacy_f32 src0=0"];

    /// Rows of (a, b, c, expected...) through `code`, compared exactly
    /// (NaN bits included).
    fn check<const N: usize>(what: &str, code: &[u32], rows: &[[u32; N]], names: &[&str]) {
        let mut input = vec![0u32; (THREADS * INPUTS) as usize];
        for (tid, row) in rows.iter().enumerate() {
            input[tid * INPUTS as usize..tid * INPUTS as usize + 3].copy_from_slice(&row[..3]);
        }
        let out = dispatch(code, 32, THREADS, &input, (THREADS * RESULTS) as usize, false).unwrap();
        let mut fails = Vec::new();
        for (tid, row) in rows.iter().enumerate() {
            for (j, &e) in row[3..].iter().enumerate() {
                let actual = out[tid * RESULTS as usize + j];
                if actual != e {
                    fails.push(format!("lane {tid} ({:#x}, {:#x}, {:#x}) {} is {actual:#x}, expected {e:#x}", row[0], row[1], row[2], names[j]));
                }
            }
        }
        assert_clean(what, &fails);
    }

    /// v_mad_legacy_f32 rounds the product before the add and flushes f32
    /// denormal sources, product and result in every mode.
    #[test]
    fn results() {
        check("mad legacy", &FMA_LEGACY_CODE, &ROWS, &NAMES);
    }

    /// v_mac_legacy_f32 (VOP2 and VOP3) measures the same as
    /// v_mad_legacy_f32 with the accumulator as the addend.
    #[test]
    fn mac_legacy() {
        check("mac legacy", &MAC_LEGACY_CODE, &MAC_ROWS, &MAC_NAMES);
    }
}

mod fma_rounding {
    use super::*;
    include!("vectors/valu_fma_rounding.rs");

    const NAMES: [&str; 11] = ["v_fma_f32", "v_fmac_f32", "v_fmaak_f32", "v_fmamk_f32", "v_fma_mix_f32", "v_mad_f32", "v_mac_f32", "v_madak_f32", "v_madmk_f32", "v_mul_f32 + v_add_f32", "v_mac_f32_e64"];

    #[test]
    fn results() {
        let mut input = vec![0u32; (THREADS * INPUTS) as usize];
        for (tid, row) in ROWS.iter().enumerate() {
            // a, b, c, akB, mkC
            input[tid * INPUTS as usize..tid * INPUTS as usize + 5].copy_from_slice(&row[..5]);
        }
        let out = dispatch(&FMA_CODE, 32, THREADS, &input, (THREADS * RESULTS) as usize, false).unwrap();
        let mut fails = Vec::new();
        // The fused rows must show fused rounding: the interpreter always
        // has a fused multiply-add, so all eleven forms are checked
        // (v_fma_mix_f32 with f32 sources rounds once, as v_fma_f32).
        for (tid, row) in ROWS.iter().enumerate() {
            let (fused, fused_ak, fused_mk, separate, separate_ak, separate_mk) = (row[5], row[6], row[7], row[8], row[9], row[10]);
            let expected = [fused, fused, fused_ak, fused_mk, fused, separate, separate, separate_ak, separate_mk, separate, separate];
            for (j, &e) in expected.iter().enumerate() {
                let actual = out[tid * RESULTS as usize + j];
                if actual != e {
                    fails.push(format!("lane {tid} ({:#x}, {:#x}, {:#x}) {} is {actual:#x}, expected {e:#x}", row[0], row[1], row[2], NAMES[j]));
                }
            }
        }
        assert_clean("fma rounding", &fails);
    }
}

mod f32_min_max_nan {
    use super::*;
    include!("vectors/valu_f32_min_max_nan.rs");

    #[test]
    fn rows() {
        let mut fails = Vec::new();
        for chunk in 0..CHUNKS as usize {
            let range = chunk * THREADS as usize..(chunk + 1) * THREADS as usize;
            fails.extend(table(&CODE, &ROWS[range.clone()], RESULTS as usize, &EXPECTED[range], &NAMES, &|_, _| true, &|_, _| false, chunk * THREADS as usize));
        }
        assert_clean("f32 min max nan", &fails);
    }
}

mod f32_output_modifier {
    use super::*;
    include!("vectors/valu_f32_output_modifier.rs");

    #[test]
    fn rows() {
        let checked = |lane: usize, col: usize| (CHECKED[lane] >> col) & 1 != 0;
        let any_nan = |lane: usize, col: usize| (ANY_NAN[lane] >> col) & 1 != 0;
        assert_clean("f32 output modifier", &table(&CODE, &ROWS, RESULTS as usize, &EXPECTED, &NAMES, &checked, &any_nan, 0));
    }
}

mod f32_result_clamp {
    use super::*;
    include!("vectors/valu_f32_result_clamp.rs");

    #[test]
    fn rows() {
        let checked = |lane: usize, col: usize| (CHECKED[lane] >> col) & 1 != 0;
        assert_clean("f32 result clamp", &table(&CODE, &ROWS, RESULTS as usize, &EXPECTED, &NAMES, &checked, &|_, _| false, 0));
    }
}

/// v_med3_f32 ties between signed zeros: the median is picked against the
/// numeric maximum of the three (+0 and -0 compare equal), as measured on
/// RDNA2; any NaN input takes the min3 path.
mod f32_med3_signed_zero {
    use super::*;
    include!("vectors/valu_f32_min_max_nan.rs");

    const CASES: [[u32; 4]; 22] = [
        [0x40000000, 0x40400000, 0x7fa00001, 0x40000000],
        [0x40000000, 0x40400000, 0x7fc00000, 0x40000000],
        [0x7fa00001, 0x40000000, 0x40400000, 0x40000000],
        [0x40400000, 0x7fc01234, 0x40000000, 0x40000000],
        [0x80000000, 0x00000000, 0x3f800000, 0x00000000],
        [0x00000000, 0x80000000, 0xbf800000, 0x80000000],
        [0x80000000, 0x80000000, 0x00000000, 0x00000000],
        [0x00000000, 0x00000000, 0x80000000, 0x00000000],
        [0x80000000, 0x00000000, 0x80000000, 0x00000000],
        [0x00000000, 0x80000000, 0x80000000, 0x80000000],
        [0xbf800000, 0x80000000, 0x00000000, 0x00000000],
        [0xbf800000, 0x00000000, 0x80000000, 0x80000000],
        [0x80000000, 0xbf800000, 0x00000000, 0x00000000],
        [0x00000000, 0xbf800000, 0x80000000, 0x80000000],
        [0x80000000, 0x00000000, 0xbf800000, 0x00000000],
        [0x7fc01234, 0x80000000, 0x00000000, 0x80000000],
        [0x3f800000, 0x40400000, 0x40000000, 0x40000000],
        [0xc0400000, 0xbf800000, 0xc0000000, 0xc0000000],
        [0xff800000, 0x7f800000, 0x3f800000, 0x3f800000],
        [0x00000001, 0x7f800000, 0x7fc01234, 0x00000001],
        [0x80000001, 0x80000001, 0x00000000, 0x80000001],
        [0x7fa00001, 0xffc05678, 0x40000000, 0x40000000],
    ];
    // Not ported: med3(0x7fc01234, 0x7fa00001, 0xffc05678) = 0xffc05678 of
    // the reference contradicts the measured table above, where three NaNs
    // return the first one.

    #[test]
    fn ties() {
        let rows: Vec<[u32; 4]> = CASES.iter().map(|c| [c[0], c[1], c[2], 0]).collect();
        let input: Vec<u32> = rows.iter().flatten().copied().collect();
        let out = dispatch(&CODE, 32, rows.len() as u32, &input, rows.len() * RESULTS as usize, true).expect("dispatch");
        let mut fails = Vec::new();
        for (lane, c) in CASES.iter().enumerate() {
            let a = out[lane * RESULTS as usize + 4];
            // Denormal results depend on the float mode (the reference ran
            // with FP32 denormals kept).
            if c[3] & 0x7F80_0000 == 0 && c[3] & 0x7FFF_FFFF != 0 {
                continue;
            }
            if a != c[3] {
                fails.push(format!("v_med3_f32({:#010x}, {:#010x}, {:#010x}) is {a:#010x}, expected {:#010x}", c[0], c[1], c[2], c[3]));
            }
        }
        assert_clean("f32 med3 signed zero", &fails);
    }
}
