// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! FLOAT_MODE denormal and f16 round-mode fields against hardware-measured
//! reference rows (RDNA2): the f16/f64 denormal field (bit 6 keeps input
//! denormals, bit 7 result denormals) for f16 unary operations and
//! conversions, compares, the fused multiply-adds, packed f16, the pack
//! instructions, v_fma_mix_f32 f16 sources, v_div_fixup_f16/f64 and
//! v_cvt_pknorm_*_f16; the f32 field for v_ldexp_f32 and compares; the
//! directed f16 round modes of the f16 conversions, v_ldexp_f16 and
//! v_fma_mixlo/mixhi_f16.

include!("vectors/fmod_common.rs");
include!("vectors/valu_denormal_modes.rs");

use ps5shader::translate::FloatMode;

fn mode(bits: u8, ieee: bool) -> FloatMode {
    FloatMode { mode: bits, dx10_clamp: true, ieee, fp16_overflow: false }
}

/// Run `code` over `lanes` (4 input words each) on raw buffers;
/// `results` output words per lane.
fn run_raw(code: &[u32], lanes: &[[u32; 4]], results: usize, mode: FloatMode) -> Vec<u32> {
    let input: Vec<u32> = lanes.iter().flatten().copied().collect();
    match fmod_dispatch_mode(code, 32, lanes.len() as u32, &input, lanes.len() * results, 0, mode) {
        Ok(o) => o,
        Err(e) => panic!("{}", &e[..e.len().min(2000)]),
    }
}

/// `run_raw` on structured buffers of 4-byte records (the reference's
/// idxen layout).
fn run_records(code: &[u32], lanes: &[[u32; 4]], results: usize, mode: FloatMode) -> Vec<u32> {
    let input: Vec<u32> = lanes.iter().flatten().copied().collect();
    match fmod_dispatch_mode(code, 32, lanes.len() as u32, &input, lanes.len() * results, 4, mode) {
        Ok(o) => o,
        Err(e) => panic!("{}", &e[..e.len().min(2000)]),
    }
}

fn nan16(h: u32) -> bool {
    h & 0x7FFF > 0x7C00
}

/// The four f16 denormal modes (FLOAT_MODE[7:6]) with f32 denormals kept,
/// IEEE_MODE on and off, as the reference runs them.
fn f16_modes() -> Vec<(u32, FloatMode)> {
    let mut v = Vec::new();
    for field in 0..4u32 {
        for ieee in [true, false] {
            v.push((field, mode(0x30 | (field << 6) as u8, ieee)));
        }
    }
    v
}

#[test]
fn f16_unary_and_conversions() {
    use f16_unary::*;
    let cases: [(&str, u32, &[[u32; 5]]); 14] = [
        ("v_ceil_f16", 0x7e14b904, &CEIL_ROWS),
        ("v_cvt_f32_f16", 0x7e141704, &CVT_F32_ROWS),
        ("v_cvt_norm_i16_f16", 0x7e14c704, &CVT_NORM_I16_ROWS),
        ("v_cvt_norm_u16_f16", 0x7e14c904, &CVT_NORM_U16_ROWS),
        ("v_exp_f16", 0x7e14b104, &EXP_ROWS),
        ("v_floor_f16", 0x7e14b704, &FLOOR_ROWS),
        ("v_fract_f16", 0x7e14bf04, &FRACT_ROWS),
        ("v_frexp_exp_i16_f16", 0x7e14b504, &FREXP_EXP_ROWS),
        ("v_frexp_mant_f16", 0x7e14b304, &FREXP_MANT_ROWS),
        ("v_log_f16", 0x7e14af04, &LOG_ROWS),
        ("v_rcp_f16", 0x7e14a904, &RCP_ROWS),
        ("v_rsq_f16", 0x7e14ad04, &RSQ_ROWS),
        ("v_sin_f16", 0x7e14c104, &SIN_ROWS),
        ("v_sqrt_f16", 0x7e14ab04, &SQRT_ROWS),
    ];
    let mut failures = Vec::new();
    for (name, word, rows) in cases {
        let code = [0x34020084, 0x34040082, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x7e1402ff, 0x5a5a5a5a, word, 0xe0701000, 0x80010a02, 0xbf810000];
        let mut lanes = vec![[0u32; 4]; 32];
        for (i, r) in rows.iter().enumerate() {
            lanes[i][0] = r[0];
        }
        for denorm16 in 0..4u8 {
            for ieee in [false, true] {
                let out = run_raw(&code, &lanes, 1, mode(0x30 | denorm16 << 6, ieee));
                for (i, r) in rows.iter().enumerate() {
                    let e = r[1 + denorm16 as usize];
                    if out[i] != e {
                        failures.push(format!("{name} denorm16 {denorm16} ieee {ieee} input {:#06x}: {:#010x}, expected {e:#010x}", r[0], out[i]));
                    }
                }
            }
        }
    }
    fmod_report(&failures);
}

/// The reference's compare tables: 256 pairs of 16 values, 12 conditions,
/// result bits per wave of 32 pairs.
fn compares(values: &[u32; 16], code: &[u32], rows: &[[[u32; 8]; 2]; 12], modes: &[(FloatMode, bool)], form: &str) -> Vec<String> {
    const NAMES: [&str; 12] = ["eq", "ge", "gt", "le", "lg", "lt", "neq", "nge", "ngt", "nle", "nlg", "nlt"];
    let lanes: Vec<[u32; 4]> = (0..256).map(|p| [values[p / 16], values[p % 16], 0, 0]).collect();
    let mut failures = Vec::new();
    for &(mode, flushed) in modes {
        let out = run_raw(code, &lanes, 16, mode);
        for (index, row) in rows.iter().enumerate() {
            let bits = &row[flushed as usize];
            for pair in 0..256 {
                let e = (bits[pair / 32] >> (pair % 32)) & 1;
                let a = out[pair * 16 + index];
                if a != e {
                    failures.push(format!("{form}{} mode {:#04x} ieee {} of {:#x} and {:#x}: {a:#x}, expected {e}", NAMES[index], mode.mode, mode.ieee, values[pair / 16], values[pair % 16]));
                }
            }
        }
    }
    failures
}

#[test]
fn compare_f16_denormal_modes() {
    use vcmp_f16::*;
    let modes: Vec<(FloatMode, bool)> = f16_modes().into_iter().map(|(field, m)| (m, field & 1 == 0)).collect();
    let mut failures = compares(&VALUES, &COMPARE_CODE, &ROWS, &modes, "v_cmp_");
    failures.extend(compares(&VALUES, &COMPARE_EXEC_CODE, &ROWS, &modes, "v_cmpx_"));
    fmod_report(&failures);
}

#[test]
fn compare_f32_denormal_modes() {
    use vcmp_f32::*;
    let modes = [(mode(0xD0, true), false), (mode(0xD0, false), false), (mode(0xE0, true), true), (mode(0xE0, false), true)];
    let mut failures = compares(&VALUES, &COMPARE_CODE, &ROWS, &modes, "v_cmp_");
    failures.extend(compares(&VALUES, &COMPARE_EXEC_CODE, &ROWS, &modes, "v_cmpx_"));
    fmod_report(&failures);
}

#[test]
fn packed_f16_denormal_modes() {
    use packed_f16::*;
    const NAMES: [&str; 3] = ["v_pk_add_f16", "v_pk_mul_f16", "v_pk_fma_f16"];
    let lanes: Vec<[u32; 4]> = ROWS.iter().map(|r| [r[0], r[1], r[2], 0]).collect();
    let mut failures = Vec::new();
    for (field, m) in f16_modes() {
        let out = run_raw(&CODE, &lanes, 4, m);
        for (tid, r) in ROWS.iter().enumerate() {
            for form in 0..3 {
                let (a, e) = (out[tid * 4 + form], EXPECTED[field as usize][tid][form]);
                if a != e {
                    failures.push(format!("{} field {field} ieee {} of {r:08x?}: {a:#010x}, expected {e:#010x}", NAMES[form], m.ieee));
                }
            }
        }
    }
    fmod_report(&failures);
}

#[test]
fn pack_f16_denormal_modes() {
    use f16_pack::*;
    let lanes: Vec<[u32; 4]> = (0..256).map(|p| [SINGLES[p / 16], SINGLES[p % 16], HALVES[p / 16], HALVES[p % 16]]).collect();
    let flush = |h: u32, flushed: bool| if flushed && h & 0x7FFF < 0x0400 { h & 0x8000 } else { h };
    let mut failures = Vec::new();
    for (field, m) in f16_modes() {
        let out = run_raw(&CODE, &lanes, 2, m);
        for p in 0..256 {
            let (l, r) = (p / 16, p % 16);
            let results_flushed = field & 2 == 0;
            let packed = flush(SINGLES_TOWARD_ZERO[l], results_flushed) | flush(SINGLES_TOWARD_ZERO[r], results_flushed) << 16;
            if out[p * 2] != packed {
                failures.push(format!("v_cvt_pkrtz_f16_f32 field {field} ieee {} of {:#x}, {:#x}: {:#010x}, expected {packed:#010x}", m.ieee, SINGLES[l], SINGLES[r], out[p * 2]));
            }
            let half = |v: u32| {
                let bits = flush(v, field & 1 == 0);
                if m.ieee && nan16(bits) {
                    bits | 0x0200
                } else {
                    bits
                }
            };
            let joined = half(HALVES[l]) | half(HALVES[r]) << 16;
            if out[p * 2 + 1] != joined {
                failures.push(format!("v_pack_b32_f16 field {field} ieee {} of {:#x}, {:#x}: {:#010x}, expected {joined:#010x}", m.ieee, HALVES[l], HALVES[r], out[p * 2 + 1]));
            }
        }
    }
    fmod_report(&failures);
}

#[test]
fn fma_f16_denormal_modes() {
    use f16_fma::*;
    const NAMES: [&str; 2] = ["v_fma_f16", "v_fmac_f16"];
    let lanes: Vec<[u32; 4]> = ROWS.iter().map(|r| [r[0], r[1], r[2], 0]).collect();
    let mut failures = Vec::new();
    for (field, m) in f16_modes() {
        let out = run_raw(&CODE, &lanes, 2, m);
        for (tid, r) in ROWS.iter().enumerate() {
            let e = EXPECTED[tid][field as usize];
            for form in 0..2 {
                let a = out[tid * 2 + form];
                if !(a == e || (nan16(e) && a <= 0xFFFF && nan16(a))) {
                    failures.push(format!("{} field {field} ieee {} of {r:08x?}: {a:#010x}, expected {e:#010x}", NAMES[form], m.ieee));
                }
            }
        }
    }
    fmod_report(&failures);
}

#[test]
fn fma_mix_f16_source_denormal_modes() {
    use fma_mix_f16::*;
    let lanes: Vec<[u32; 4]> = ROWS.iter().map(|r| [r[0], r[1], r[2], 0]).collect();
    let nan32 = |v: u32| v & 0x7FFF_FFFF > 0x7F80_0000;
    let mut failures = Vec::new();
    for f32_kept in 0..2u8 {
        for f16 in 0..4u8 {
            for ieee in [false, true] {
                let m = mode(f16 << 6 | if f32_kept != 0 { 0x30 } else { 0 }, ieee);
                let out = run_raw(&CODE, &lanes, 2, m);
                for (tid, r) in ROWS.iter().enumerate() {
                    for form in 0..2 {
                        let (a, e) = (out[tid * 2 + form], EXPECTED[f32_kept as usize][(f16 & 1) as usize][tid][form]);
                        if !(a == e || (nan32(e) && nan32(a))) {
                            failures.push(format!("v_fma_mix_f32 form {form} mode {:#04x} ieee {ieee} of {r:08x?}: {a:#010x}, expected {e:#010x}", m.mode));
                        }
                    }
                }
            }
        }
    }
    fmod_report(&failures);
}

fn fixup_f16(quotient: u32, denominator: u32, numerator: u32, flushed: bool, ieee: bool) -> u32 {
    let magnitude = |h: u32| h & 0x7FFF;
    let flush = |h: u32| if flushed && magnitude(h) < 0x0400 { h & 0x8000 } else { h };
    let (denominator, numerator) = (flush(denominator), flush(numerator));
    let nan = |h: u32| magnitude(h) > 0x7C00;
    let infinite = |h: u32| magnitude(h) == 0x7C00;
    let zero = |h: u32| magnitude(h) == 0;
    let sign = (denominator ^ numerator) & 0x8000;
    if nan(numerator) {
        return if ieee { numerator | 0x0200 } else { numerator };
    }
    if nan(denominator) {
        return if ieee { denominator | 0x0200 } else { denominator };
    }
    if (zero(denominator) && zero(numerator)) || (infinite(denominator) && infinite(numerator)) {
        return 0xFE00;
    }
    if zero(denominator) || infinite(numerator) {
        return sign | 0x7C00;
    }
    if infinite(denominator) || zero(numerator) {
        return sign;
    }
    if nan(quotient) {
        return sign | 0x7C00;
    }
    sign | magnitude(quotient)
}

#[test]
fn div_fixup_f16_denormal_modes() {
    use div_fixup_f16::*;
    let lanes: Vec<[u32; 4]> = (0..1024usize)
        .map(|tid| {
            let row = if tid < 1000 { tid } else { 0 };
            [VALUES[row / 100], VALUES[row / 10 % 10], VALUES[row % 10], 0]
        })
        .collect();
    let mut failures = Vec::new();
    for (field, m) in f16_modes() {
        let out = run_raw(&CODE, &lanes, 1, m);
        for (tid, l) in lanes.iter().enumerate() {
            let e = fixup_f16(l[0], l[1], l[2], field & 1 == 0, m.ieee);
            if out[tid] != e {
                failures.push(format!("v_div_fixup_f16 field {field} ieee {} of {:#x}, {:#x}, {:#x}: {:#010x}, expected {e:#010x}", m.ieee, l[0], l[1], l[2], out[tid]));
            }
        }
    }
    fmod_report(&failures);
}

fn fixup_f64(quotient: u64, denominator: u64, numerator: u64, flushed: bool, ieee: bool) -> u64 {
    const MAGNITUDE: u64 = 0x7FFF_FFFF_FFFF_FFFF;
    const INFINITY: u64 = 0x7FF0_0000_0000_0000;
    let flush = |v: u64| if flushed && v & INFINITY == 0 { v & !MAGNITUDE } else { v };
    let (denominator, numerator) = (flush(denominator), flush(numerator));
    let nan = |v: u64| v & MAGNITUDE > INFINITY;
    let infinite = |v: u64| v & MAGNITUDE == INFINITY;
    let zero = |v: u64| v & MAGNITUDE == 0;
    let exponent = |v: u64| ((v >> 52) & 0x7FF) as i32;
    let sign = (denominator ^ numerator) & !MAGNITUDE;
    if nan(numerator) {
        return if ieee { numerator | 0x0008_0000_0000_0000 } else { numerator };
    }
    if nan(denominator) {
        return if ieee { denominator | 0x0008_0000_0000_0000 } else { denominator };
    }
    if (zero(denominator) && zero(numerator)) || (infinite(denominator) && infinite(numerator)) {
        return 0xFFF8_0000_0000_0000;
    }
    if zero(denominator) || infinite(numerator) {
        return sign | INFINITY;
    }
    if infinite(denominator) || zero(numerator) {
        return sign;
    }
    if exponent(numerator) - exponent(denominator) < -1075 {
        return sign;
    }
    if nan(quotient) {
        return sign | INFINITY;
    }
    sign | (quotient & MAGNITUDE)
}

#[test]
fn div_fixup_f64_denormal_modes() {
    use div_fixup_f64::*;
    let sources = |tid: usize| {
        let row = if tid < 1000 { tid } else { 0 };
        [VALUES[row / 100], VALUES[row / 10 % 10], VALUES[row % 10]]
    };
    let mut input = Vec::new();
    for tid in 0..1024 {
        for s in sources(tid) {
            input.push(s as u32);
            input.push((s >> 32) as u32);
        }
        input.extend([0, 0]);
    }
    let mut failures = Vec::new();
    for (field, m) in f16_modes() {
        let out = match fmod_dispatch_mode(&CODE, 32, 1024, &input, 2048, 0, m) {
            Ok(o) => o,
            Err(e) => panic!("{}", &e[..e.len().min(2000)]),
        };
        for tid in 0..1024 {
            let s = sources(tid);
            let e = fixup_f64(s[0], s[1], s[2], field & 1 == 0, m.ieee);
            let a = out[tid * 2] as u64 | (out[tid * 2 + 1] as u64) << 32;
            if a != e {
                failures.push(format!("v_div_fixup_f64 field {field} ieee {} of {:#x}, {:#x}, {:#x}: {a:#018x}, expected {e:#018x}", m.ieee, s[0], s[1], s[2]));
            }
        }
    }
    fmod_report(&failures);
}

#[test]
fn pknorm_f16_denormal_modes() {
    use pknorm_f16::*;
    let cases: [(&str, u32, &[[u32; 6]]); 2] = [("v_cvt_pknorm_i16_f16", 0xd712000a, &PKNORM_I16_ROWS), ("v_cvt_pknorm_u16_f16", 0xd713000a, &PKNORM_U16_ROWS)];
    let mut failures = Vec::new();
    for (name, word, rows) in cases {
        let code = [0x34020084, 0x34040082, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x7e140280, word, 0x00020b04, 0xe0701000, 0x80010a02, 0xbf810000];
        let mut lanes = vec![[0u32; 4]; 32];
        for (i, r) in rows.iter().enumerate() {
            lanes[i][0] = r[0];
            lanes[i][1] = r[1];
        }
        for denorm16 in 0..4u8 {
            for ieee in [false, true] {
                let out = run_raw(&code, &lanes, 1, mode(denorm16 << 6, ieee));
                for (i, r) in rows.iter().enumerate() {
                    let e = r[2 + denorm16 as usize];
                    if out[i] != e {
                        failures.push(format!("{name} denorm16 {denorm16} ieee {ieee} of {:#x}, {:#x}: {:#010x}, expected {e:#010x}", r[0], r[1], out[i]));
                    }
                }
            }
        }
    }
    fmod_report(&failures);
}

#[test]
fn ldexp_f32_denormal_modes() {
    use ldexp_f32::*;
    let mut lanes = vec![[0u32; 4]; 64];
    for (i, v) in VECTORS.iter().enumerate() {
        lanes[i][0] = v[0];
        lanes[i][1] = v[1];
    }
    let mut failures = Vec::new();
    for denormals in [1u8, 2] {
        for rounding in 0..4u8 {
            for ieee in [false, true] {
                let out = run_raw(&CODE, &lanes, 16, mode(0xC0 | denormals << 4 | rounding, ieee));
                for (i, v) in VECTORS.iter().enumerate() {
                    let e = if denormals == 1 { v[2 + rounding as usize] } else { v[6 + rounding as usize] };
                    let a = out[i * 16];
                    if a != e {
                        failures.push(format!("v_ldexp_f32 denormals {denormals} round {rounding} ieee {ieee} of {:#x}, {:#x}: {a:#010x}, expected {e:#010x}", v[0], v[1]));
                    }
                }
            }
        }
    }
    fmod_report(&failures);
}

/// F16ConversionFloatMode in the f16 round modes `rounds` (FLOAT_MODE[3:2])
/// and the four f16 denormal modes.
fn f16_conversions(rounds: &[u32]) {
    use f16_conversion::*;
    let shader = |word: u32, second: Option<u32>| {
        let mut code = vec![0x34020084, 0x34040082, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x7e140280, word];
        code.extend(second);
        code.extend([0xe0701000, 0x80010a02, 0xbf810000]);
        code
    };
    let cases: [(&str, Vec<u32>, &[[u32; 18]]); 5] = [
        ("v_cvt_f16_f32", shader(0x7e141504, None), &V_CVT_F16_F32_ROWS),
        ("v_cvt_f16_f32 clamp", shader(0xd58a800a, Some(0x00000104)), &V_CVT_F16_F32_CLAMP_ROWS),
        ("v_cvt_f16_u16", shader(0x7e14a104, None), &V_CVT_F16_U16_ROWS),
        ("v_cvt_f16_i16", shader(0x7e14a304, None), &V_CVT_F16_I16_ROWS),
        ("v_ldexp_f16", shader(0x76140b04, None), &V_LDEXP_F16_ROWS),
    ];
    let mut failures = Vec::new();
    for (name, code, rows) in cases {
        let mut lanes = vec![[0u32; 4]; 64];
        for (i, r) in rows.iter().enumerate() {
            lanes[i][0] = r[0];
            lanes[i][1] = r[1];
        }
        for &round16 in rounds {
            for denorm16 in 0..4u32 {
                let out = run_raw(&code, &lanes, 1, mode((round16 << 2 | denorm16 << 6) as u8, false));
                for (i, r) in rows.iter().enumerate() {
                    let e = r[2 + (round16 * 4 + denorm16) as usize];
                    if out[i] != e {
                        failures.push(format!("{name} round16 {round16} denorm16 {denorm16} of {:#x}, {:#x}: {:#010x}, expected {e:#010x}", r[0], r[1], out[i]));
                    }
                }
            }
        }
    }
    fmod_report(&failures);
}

#[test]
fn f16_conversion_denormal_modes() {
    f16_conversions(&[0]);
}

#[test]
fn f16_ldexp_denormal_modes() {
    use f16_rounding::*;
    let lanes: Vec<[u32; 4]> = (0..64).map(|i| LDEXP_DENORMAL_VECTORS.get(i).map_or([0; 4], |v| [v[0], v[1], 0, 0])).collect();
    let mut failures = Vec::new();
    for denorm16 in 0..4u8 {
        let out = run_records(&LDEXP_DENORMAL_CODE, &lanes, 16, mode(denorm16 << 6, true));
        for (i, v) in LDEXP_DENORMAL_VECTORS.iter().enumerate() {
            let (a, e) = (out[i * 16], v[2 + denorm16 as usize]);
            if !(a == e || (a >> 16 == e >> 16 && nan16(a) && nan16(e))) {
                failures.push(format!("v_ldexp_f16 denorm16 {denorm16} of {:#x}, {:#x}: {a:#010x}, expected {e:#010x}", v[0], v[1]));
            }
        }
    }
    fmod_report(&failures);
}

#[test]
fn f16_conversion_round_modes() {
    f16_conversions(&[1, 2, 3]);
}

/// Float16Rounding: v_cvt_f16_f32, v_cvt_f16_i16, v_cvt_f16_u16 and
/// v_ldexp_f16 in the directed f16 round modes, with and without FP16_OVFL,
/// with the f32 round mode nearest and toward zero.
#[test]
fn directed_f16_conversions() {
    use f16_rounding::*;
    let lanes: Vec<[u32; 4]> = (0..64).map(|i| DIRECTED_VECTORS.get(i).map_or([0; 4], |v| [v[0], v[1], v[2], v[3]])).collect();
    let mut failures = Vec::new();
    for rounding in 1..4u32 {
        for overflow in 0..2u32 {
            for round32 in [0u32, 3] {
                let m = FloatMode { mode: (0xF0 | rounding << 2 | round32) as u8, dx10_clamp: true, ieee: true, fp16_overflow: overflow != 0 };
                let out = run_raw(&DIRECTED_CODE, &lanes, 16, m);
                for (lane, v) in DIRECTED_VECTORS.iter().enumerate() {
                    for column in 0..4 {
                        let e = v[4 + ((rounding - 1) * 2 + overflow) as usize * 4 + column];
                        let a = out[lane * 16 + column];
                        let ok = if column == 0 { a == e || (a >> 16 == e >> 16 && nan16(a) && nan16(e)) } else { a == e };
                        if !ok {
                            failures.push(format!("round {rounding} overflow {overflow} round32 {round32} lane {lane} {:08x?} column {column}: {a:#010x}, expected {e:#010x}", &v[..4]));
                        }
                    }
                }
            }
        }
    }
    fmod_report(&failures);
}

/// MixPrecision: v_fma_mixlo_f16 and v_fma_mixhi_f16 in the four f16
/// round modes.
#[test]
fn mix_f16_round_modes() {
    use mix_directed::*;
    let lanes: Vec<[u32; 4]> = (0..64).map(|i| DIRECTED_ROWS.get(i).map_or([0; 4], |r| [r[0], r[1], r[2], 0])).collect();
    let mut failures = Vec::new();
    for rounding in 0..4u32 {
        let out = run_records(&ROUNDING_CODE, &lanes, 16, mode((0xC0 | rounding << 2) as u8, false));
        for (tid, r) in DIRECTED_ROWS.iter().enumerate() {
            let e = r[3 + rounding as usize];
            let (lo, hi) = (out[tid * 16], out[tid * 16 + 1]);
            if lo != 0xABCD_0000 | e {
                failures.push(format!("v_fma_mixlo_f16 round {rounding} of {:08x?}: {lo:#010x}, expected {:#010x}", &r[..3], 0xABCD_0000 | e));
            }
            if hi != (e << 16 | 0xABCD) {
                failures.push(format!("v_fma_mixhi_f16 round {rounding} of {:08x?}: {hi:#010x}, expected {:#010x}", &r[..3], e << 16 | 0xABCD));
            }
        }
    }
    fmod_report(&failures);
}
