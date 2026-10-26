// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Float-control dependent VALU semantics against hardware-measured
//! reference rows (RDNA2):
//! * NaN quieting: with IEEE_MODE clear, v_sqrt/rcp/fract/floor/trunc/
//!   frexp_mant/ldexp/div_fixup/mullit and their f16 forms return a
//!   signaling NaN unchanged; with IEEE_MODE set they quiet it.
//! * f32 denormal flushing in FLOAT_MODE 0xC0 (inputs and results of the
//!   arithmetic, inputs of compares and conversions), the same with
//!   FLOAT_MODE 0x00 and IEEE_MODE.

use ps5shader::interp::Machine;
use ps5shader::ssa::SystemValue;
use ps5shader::testing::{buffer_descriptor, ComputeRun};
use ps5shader::translate::FloatMode;

const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;
const POISON: u32 = 0xDEAD_BEEF;
const RESULTS: usize = 16;

/// One wave32 of `rows.len()` lanes: s[0:3] input V#, s[4:7] output V#
/// (raw buffers), v0 = lane; `RESULTS` words per lane.
fn run(code: &[u32], rows: &[[u32; 4]], mode: FloatMode) -> Vec<u32> {
    run_words(code, rows, mode, RESULTS)
}

/// `run` with `results` words per lane.
fn run_words(code: &[u32], rows: &[[u32; 4]], mode: FloatMode, results: usize) -> Vec<u32> {
    let input: Vec<u32> = rows.iter().flatten().copied().collect();
    let out_words = rows.len() * results;
    let i = buffer_descriptor(IN_BASE, 0, input.len() as u32 * 4, 0x3101_6FAC);
    let o = buffer_descriptor(OUT_BASE, 0, out_words as u32 * 4, 0x3101_6FAC);
    let user: Vec<u32> = i.iter().chain(o.iter()).copied().collect();
    let mut run = ComputeRun::new(code.to_vec(), 32, rows.len() as u32, user);
    run.float_mode = mode;
    let setup = |m: &mut Machine| {
        m.memory.write_words(IN_BASE, &input);
        m.memory.write_words(OUT_BASE, &vec![POISON; out_words]);
        m.system.insert(SystemValue::LocalInvocationId(0), (0..rows.len() as u32).collect());
    };
    let m = run.run_checked(&setup, &[(OUT_BASE, out_words)]).unwrap();
    m.memory.read_words(OUT_BASE, out_words)
}

fn is_nan32(v: u32) -> bool {
    v & 0x7FFF_FFFF > 0x7F80_0000
}

fn is_nan16(v: u32) -> bool {
    v & 0x7FFF > 0x7C00
}

const IEEE_OFF: FloatMode = FloatMode { mode: 0xC0, dx10_clamp: true, ieee: false, fp16_overflow: false };
const IEEE_ON: FloatMode = FloatMode { mode: 0xC0, dx10_clamp: true, ieee: true, fp16_overflow: false };

mod nan_quieting {
    use super::*;
    include!("vectors/valu_nan_quieting.rs");

    fn check(mode: FloatMode, expected: &[[u32; 16]; 32]) {
        let out = run(&CODE, &ROWS, mode);
        let mut fails = Vec::new();
        for (tid, row) in expected.iter().enumerate() {
            for (i, &e) in row.iter().enumerate() {
                // f16 results in the low half; sqrt/rcp results only when
                // they are NaNs (their precision is the host's).
                let half = i >= 8 && i != 14;
                let (a, e) = if half { (out[tid * RESULTS + i] & 0xFFFF, e & 0xFFFF) } else { (out[tid * RESULTS + i], e) };
                let approximate = matches!(i, 0 | 1 | 8 | 9);
                if approximate && !(if half { is_nan16(e) } else { is_nan32(e) }) {
                    continue;
                }
                if a != e {
                    fails.push(format!("lane {tid} {:08x?} {}: {a:#010x}, expected {e:#010x}", ROWS[tid], NAMES[i]));
                }
            }
        }
        assert!(fails.is_empty(), "{} failures:\n{}", fails.len(), fails.join("\n"));
    }

    #[test]
    fn ieee_mode_clear_keeps_signaling_nans() {
        check(IEEE_OFF, &EXPECTED_NO_IEEE);
    }

    #[test]
    fn ieee_mode_quiets_signaling_nans() {
        check(IEEE_ON, &EXPECTED_IEEE);
    }
}

mod f32_denormal_flush {
    use super::*;
    include!("vectors/valu_f32_denormal_flush.rs");

    fn check(mode: FloatMode) {
        let words = RESULTS as usize;
        let out = run_words(&CODE, &ROWS, mode, words);
        let mut fails = Vec::new();
        for (tid, row) in EXPECTED.iter().enumerate() {
            for (i, &e) in row.iter().enumerate() {
                let a = out[tid * words + i];
                let float_result = i < 12 || i >= 15;
                let ok = if float_result && is_nan32(e) { is_nan32(a) } else { a == e };
                if !ok {
                    fails.push(format!("lane {tid} {:08x?} {}: {a:#010x}, expected {e:#010x}", ROWS[tid], NAMES[i]));
                }
            }
        }
        assert!(fails.is_empty(), "{} failures:\n{}", fails.len(), fails.join("\n"));
    }

    #[test]
    fn float_mode_c0() {
        check(IEEE_OFF);
    }

    #[test]
    fn float_mode_00_ieee() {
        check(FloatMode { mode: 0x00, dx10_clamp: false, ieee: true, fp16_overflow: false });
    }

    const FLUSH_MODES: [FloatMode; 3] = [IEEE_OFF, IEEE_OFF, FloatMode { mode: 0x00, dx10_clamp: false, ieee: true, fp16_overflow: false }];

    fn directed(round: u8) -> FloatMode {
        FloatMode { mode: 0xC0 | round, ..IEEE_OFF }
    }

    const TINY_NAMES: [&str; 10] = [
        "v_mul_f32 v10, v4, v5",
        "v_mul_legacy_f32 v11, v4, v5",
        "v_fma_f32 v12, v4, v5, v6",
        "v_fmac_f32 v13, v4, v5",
        "v_fmamk_f32 v14, v4, 0x3f7ffffe, v6",
        "v_fmaak_f32 v15, v4, v5, 0x80000000",
        "v_fma_mix_f32 v16, v4, v5, v6",
        "v_mul_f32 v17, -v4, v5",
        "v_mad_legacy_f32 v18, v4, v5, v6",
        "v_mac_legacy_f32 v19, v4, v5",
    ];

    /// Tiny products: tininess is decided after rounding (a product that
    /// rounds up to the smallest normal is that normal) except in the
    /// legacy multiply-adds, which flush before rounding.
    #[test]
    fn tiny_products_round_before_the_flush() {
        let words = TINY_RESULTS as usize;
        let mut fails = Vec::new();
        for (k, mode) in FLUSH_MODES.iter().enumerate() {
            let out = run_words(&TINY_CODE, &TINY_ROWS, *mode, words);
            for tid in 0..32 {
                for i in 0..10 {
                    let e = if i < 8 { TINY_EXPECTED[tid][i] } else { TINY_LEGACY_EXPECTED[tid] };
                    let a = out[tid * words + i];
                    if a != e {
                        fails.push(format!("mode {k} lane {tid} {:08x?} {}: {a:#010x}, expected {e:#010x}", TINY_ROWS[tid], TINY_NAMES[i]));
                    }
                }
            }
        }
        for round in 1..4u8 {
            let out = run_words(&TINY_CODE, &TINY_ROWS, directed(round), words);
            for tid in 0..32 {
                for (col, i) in [(0, 0), (1, 7), (2, 8), (2, 9)] {
                    let e = if col < 2 { TINY_DIRECTED[round as usize - 1][tid][col] } else { TINY_LEGACY_EXPECTED[tid] };
                    let a = out[tid * words + i];
                    if a != e {
                        fails.push(format!("round mode {round} lane {tid} {:08x?} {}: {a:#010x}, expected {e:#010x}", TINY_ROWS[tid], TINY_NAMES[i]));
                    }
                }
            }
        }
        assert!(fails.is_empty(), "{} failures:\n{}", fails.len(), fails.join("\n"));
    }

    const MIXED_NAMES: [&str; 8] = [
        "v_fma_mix_f32 v10, v4, v5, v6 op_sel_hi:[0,1,0]",
        "v_fma_mix_f32 v11, v5, v4, v6 op_sel_hi:[1,0,0]",
        "v_fma_mix_f32 v12, v4, v5, v6 op_sel:[0,1,0] op_sel_hi:[0,1,0]",
        "v_fma_mix_f32 v13, v5, v4, v6 op_sel:[1,0,0] op_sel_hi:[1,0,0]",
        "v_fma_mix_f32 v14, v4, v5, v7 op_sel_hi:[0,1,1]",
        "v_fma_mix_f32 v15, v5, v4, v7 op_sel_hi:[1,0,1]",
        "v_fma_mix_f32 v16, v4, v8, v7 op_sel_hi:[0,0,1]",
        "v_fma_mix_f32 v17, v4, v8, v7 op_sel:[0,0,1] op_sel_hi:[0,0,1]",
    ];

    /// The same with f16 factors and addends of v_fma_mix_f32 (tininess
    /// after rounding of a mixed-source product), in the flush modes and
    /// the directed round modes.
    #[test]
    fn mixed_source_tiny_products() {
        let words = TINY_RESULTS as usize;
        let mut fails = Vec::new();
        let mut check = |what: String, out: &[u32], round: usize| {
            for tid in 0..MIXED_TINY_EXPECTED.len() {
                for i in 0..8 {
                    let high_factor = i == 2 || i == 3;
                    let mut e = MIXED_TINY_EXPECTED[tid][if i >= 6 { 2 } else if high_factor { 1 } else { 0 }];
                    if round != 0 {
                        if i >= 6 {
                            if tid < 14 {
                                continue;
                            }
                            e = MIXED_ADDEND_DIRECTED[round - 1][tid - 14];
                        } else {
                            if MIXED_TINY_ROWS[tid][0] & 0x7fff_ffff == 0x0080_1003 {
                                continue;
                            }
                            e = MIXED_TINY_DIRECTED[round - 1][tid][high_factor as usize];
                        }
                    }
                    let a = out[tid * words + i];
                    if a != e {
                        fails.push(format!("{what} lane {tid} {:08x?} {}: {a:#010x}, expected {e:#010x}", MIXED_TINY_ROWS[tid], MIXED_NAMES[i]));
                    }
                }
            }
        };
        for (k, mode) in FLUSH_MODES.iter().enumerate() {
            let out = run_words(&MIXED_TINY_CODE, &MIXED_TINY_ROWS, *mode, words);
            check(format!("mode {k}"), &out, 0);
        }
        for round in 1..4u8 {
            let out = run_words(&MIXED_TINY_CODE, &MIXED_TINY_ROWS, directed(round), words);
            check(format!("round mode {round}"), &out, round as usize);
        }
        assert!(fails.is_empty(), "{} failures:\n{}", fails.len(), fails.join("\n"));
    }
}

mod fp16_overflow {
    use super::*;
    include!("vectors/valu_fp16_overflow.rs");

    const NAMES: [&str; 16] = [
        "v_add_f16",
        "v_sub_f16",
        "v_mul_f16",
        "v_fma_f16",
        "v_div_fixup_f16",
        "v_pk_mul_f16 op_sel_hi:[0,0]",
        "v_pk_add_f16 op_sel_hi:[0,0]",
        "v_fma_mixlo_f16",
        "v_add_f16_sdwa dst_sel:WORD_1",
        "v_exp_f16",
        "v_log_f16",
        "v_rcp_f16",
        "v_rsq_f16",
        "v_ldexp_f16",
        "v_cvt_f16_f32",
        "v_cvt_f16_u16",
    ];

    fn is_nan16(h: u32) -> bool {
        h & 0x7FFF > 0x7C00
    }

    /// Exact halves, except a NaN the host produces (columns other than
    /// div_fixup, the packed forms, ldexp and cvt_f16_u16 accept any NaN).
    fn check(mode: FloatMode, expected: &[[u32; 16]; 32]) {
        let out = run(&CODE, &ROWS, mode);
        let mut fails = Vec::new();
        for (lane, row) in expected.iter().enumerate() {
            for (col, &e) in row.iter().enumerate() {
                let a = out[lane * RESULTS + col];
                let whole = matches!(col, 5 | 6 | 8);
                let driver_nan = !matches!(col, 4 | 5 | 6 | 13 | 15);
                for shift in [0, 16] {
                    if shift == 16 && !whole {
                        break;
                    }
                    let (h, eh) = ((a >> shift) & 0xFFFF, (e >> shift) & 0xFFFF);
                    let ok = if driver_nan && is_nan16(eh) { is_nan16(h) } else { h == eh };
                    if !ok {
                        fails.push(format!("lane {lane} {:08x?} {}: {a:#010x}, expected {e:#010x}", ROWS[lane], NAMES[col]));
                        break;
                    }
                }
            }
        }
        assert!(fails.is_empty(), "{} failures:\n{}", fails.len(), fails.join("\n"));
    }

    #[test]
    fn without_fp16_overflow() {
        check(IEEE_OFF, &EXPECTED_INFINITY);
    }

    /// With FP16_OVFL an f16 result that would round to infinity from
    /// finite sources is the largest finite value of its sign.
    #[test]
    fn with_fp16_overflow() {
        check(FloatMode { fp16_overflow: true, ..IEEE_OFF }, &EXPECTED_OVERFLOW);
    }
}

mod f16_min_max_nan {
    use super::*;
    include!("vectors/valu_f16_min_max_nan.rs");

    const NAMES: [&str; 7] = [
        "v_min_f16 v10, v4, v5",
        "v_max_f16 v11, v4, v5",
        "v_min3_f16 v12, v4, v5, v6",
        "v_max3_f16 v13, v4, v5, v6",
        "v_med3_f16 v14, v4, v5, v6",
        "v_min_f16_e64 v15, -v4, |v5|",
        "v_max_f16_e64 v16, |v4|, -v5",
    ];

    /// The 128 rows run as four wave32 dispatches; every word is exact.
    fn check(mode: FloatMode, expected: &[[u32; 7]; 128]) {
        let mut fails = Vec::new();
        for chunk in 0..4 {
            let rows = &ROWS[chunk * 32..chunk * 32 + 32];
            let out = run(&CODE, rows, mode);
            for lane in 0..32 {
                for col in 0..7 {
                    let (a, e) = (out[lane * RESULTS + col], expected[chunk * 32 + lane][col]);
                    if a != e {
                        fails.push(format!("row {} {:08x?} {}: {a:#010x}, expected {e:#010x}", chunk * 32 + lane, rows[lane], NAMES[col]));
                    }
                }
            }
        }
        assert!(fails.is_empty(), "{} failures:\n{}", fails.len(), fails.join("\n"));
    }

    #[test]
    fn ieee_off() {
        check(FloatMode::default(), &EXPECTED);
    }

    #[test]
    fn ieee_on() {
        check(FloatMode { mode: 0xF0, dx10_clamp: true, ieee: true, fp16_overflow: false }, &EXPECTED_IEEE);
    }

    /// The f16 denormal modes (FLOAT_MODE denorm16 << 6, f32 denormals
    /// flushed): with input denormals flushed (bit 6 clear) a denormal
    /// operand of min/max/min3/max3/med3 is a zero of its sign. Also through
    /// the SDWA WORD_0 and WORD_1 source forms of v_min/max_f16.
    #[test]
    fn denormal_modes() {
        let mut fails = Vec::new();
        for ieee in [false, true] {
            for form in 0..3 {
                for denorm16 in 0..4u8 {
                    let mode = FloatMode { mode: denorm16 << 6, dx10_clamp: true, ieee, fp16_overflow: false };
                    let rows: Vec<[u32; 4]> = DENORMAL_ROWS
                        .iter()
                        .map(|r| {
                            let mut r = *r;
                            if form == 2 {
                                r[0] = (r[0] << 16) | 0x1234;
                                r[1] = (r[1] << 16) | 0x5678;
                            }
                            r
                        })
                        .collect();
                    let code: &[u32] = match form {
                        1 => &SDWA_LOW_CODE,
                        2 => &SDWA_HIGH_CODE,
                        _ => &CODE,
                    };
                    let out = run(code, &rows, mode);
                    let kept = denorm16 & 1 != 0;
                    let binary = match (ieee, kept) {
                        (false, false) => &DENORMAL_FLUSHED,
                        (false, true) => &DENORMAL_KEPT,
                        (true, false) => &DENORMAL_FLUSHED_IEEE,
                        (true, true) => &DENORMAL_KEPT_IEEE,
                    };
                    let ternary = match (ieee, kept) {
                        (false, false) => &TERNARY_DENORMAL_FLUSHED,
                        (false, true) => &TERNARY_DENORMAL_KEPT,
                        (true, false) => &TERNARY_DENORMAL_FLUSHED_IEEE,
                        (true, true) => &TERNARY_DENORMAL_KEPT_IEEE,
                    };
                    for lane in 0..32 {
                        let mut checks: Vec<(usize, u32)> = [0usize, 1, 5, 6].iter().enumerate().map(|(k, &col)| (col, binary[lane][k])).collect();
                        if form == 0 {
                            checks.extend((0..3).map(|k| (k + 2, ternary[lane][k])));
                        }
                        for (col, e) in checks {
                            let a = out[lane * RESULTS + col];
                            if a != e {
                                fails.push(format!("ieee {ieee} form {form} denorm16 {denorm16} row {lane} {:08x?} {}: {a:#010x}, expected {e:#010x}", rows[lane], NAMES[col]));
                            }
                        }
                    }
                }
            }
        }
        assert!(fails.is_empty(), "{} failures:\n{}", fails.len(), fails.iter().take(60).cloned().collect::<Vec<_>>().join("\n"));
    }
}

mod f64_nan_quieting {
    use super::*;
    include!("vectors/valu_f64_nan_quieting.rs");

    fn is_nan64(lo: u32, hi: u32) -> bool {
        let m = hi & 0x7FFF_FFFF;
        m > 0x7FF0_0000 || (m == 0x7FF0_0000 && lo != 0)
    }

    /// The eight rows repeat over the wave. The first `approximate` result
    /// pairs (rcp, rsq, sqrt) are checked only where a NaN is expected.
    fn check(code: &[u32], mode: FloatMode, expected: &[[u32; 16]; 8], approximate: usize, what: &str) {
        let rows: Vec<[u32; 4]> = (0..32).map(|t| ROWS_F64[t % 8]).collect();
        let out = run(code, &rows, mode);
        let mut fails = Vec::new();
        for lane in 0..32 {
            let want = &expected[lane % 8];
            for i in 0..16 {
                let pair = i / 2;
                if pair < approximate && !is_nan64(want[pair * 2], want[pair * 2 + 1]) {
                    continue;
                }
                let a = out[lane * RESULTS + i];
                if a != want[i] {
                    fails.push(format!("{what}: lane {lane} word {i}: {a:#010x}, expected {:#010x}", want[i]));
                }
            }
        }
        assert!(fails.is_empty(), "{} failures:\n{}", fails.len(), fails.join("\n"));
    }

    /// v_fract/floor/ceil/trunc/rndne/frexp_mant/ldexp_f64 and
    /// v_cvt_f64_f32 return a signaling NaN unchanged with IEEE_MODE clear.
    #[test]
    fn rounding_ieee_off() {
        check(&CODE_ROUNDING, IEEE_OFF, &EXPECTED_ROUNDING_NO_IEEE, 0, "f64 rounding IEEE=0");
    }

    #[test]
    fn rounding_ieee_on() {
        check(&CODE_ROUNDING, IEEE_ON, &EXPECTED_ROUNDING_IEEE, 0, "f64 rounding IEEE=1");
    }

    /// v_rcp/rsq/sqrt_f64, v_div_fixup_f64 and v_cvt_f32_f64.
    #[test]
    fn approximate_ieee_off() {
        check(&CODE_APPROXIMATE, IEEE_OFF, &EXPECTED_APPROXIMATE_NO_IEEE, 3, "f64 approximate IEEE=0");
    }

    #[test]
    fn approximate_ieee_on() {
        check(&CODE_APPROXIMATE, IEEE_ON, &EXPECTED_APPROXIMATE_IEEE, 3, "f64 approximate IEEE=1");
    }
}
