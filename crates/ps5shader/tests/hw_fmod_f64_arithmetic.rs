// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! f64 add/mul/fma/min/max/ldexp with modifiers.
//! Hardware-measured reference rows, exact compare.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_f64_arithmetic.rs");

const NAMES: [&str; 16] = [
    "add lo",
    "add hi",
    "mul lo",
    "mul hi",
    "fma lo",
    "fma hi",
    "fma error lo",
    "fma error hi",
    "min lo",
    "min hi",
    "max lo",
    "max hi",
    "ldexp lo",
    "ldexp hi",
    "add abs neg lo",
    "add abs neg hi",
];

include!("vectors/fmod_f64_min_max_no_ieee.rs");

/// The rows were measured with FLOAT_MODE 0xF0 and IEEE_MODE. With IEEE_MODE
/// clear v_min/max_f64 return the other operand for a NaN
/// (MIN_MAX_NO_IEEE) and v_ldexp_f64 returns a signaling NaN unchanged
/// (lane 13).
fn check(i: usize) {
    fmod_set_float_mode(FMOD_IEEE_F0);
    fmod_check_rows(&CODE, &ROWS, &EXPECTED, None, i, NAMES[i], &fmod_exact);
    let mut no_ieee = EXPECTED;
    for (lane, row) in no_ieee.iter_mut().enumerate() {
        row[8..12].copy_from_slice(&MIN_MAX_NO_IEEE[lane]);
    }
    no_ieee[13][13] = ROWS[13][1];
    fmod_set_float_mode(ps5shader::translate::FloatMode::default());
    fmod_check_rows(&CODE, &ROWS, &no_ieee, None, i, &format!("{} (IEEE=0)", NAMES[i]), &fmod_exact);
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => $i:expr;)*) => { $( #[test] $(#[$m])* fn $name() { check($i); } )* };
}

columns! {
    c00_add_lo => 0;
    c01_add_hi => 1;
    c02_mul_lo => 2;
    c03_mul_hi => 3;
    c04_fma_lo => 4;
    c05_fma_hi => 5;
    c06_fma_error_lo => 6;
    c07_fma_error_hi => 7;
    c08_min_lo => 8;
    c09_min_hi => 9;
    c10_max_lo => 10;
    c11_max_hi => 11;
    c12_ldexp_lo => 12;
    c13_ldexp_hi => 13;
    c14_add_abs_neg_lo => 14;
    c15_add_abs_neg_hi => 15;
}


/// Reference rule of v_min/max_f64: with IEEE_MODE a signaling NaN (the
/// first one wins) is returned quieted; otherwise a NaN gives the other
/// operand; two zeros give -0 for min and +0 for max.
fn min_max_reference(lhs: u64, rhs: u64, maximum: bool, ieee: bool) -> u64 {
    const MAG: u64 = 0x7fff_ffff_ffff_ffff;
    const INF: u64 = 0x7ff0_0000_0000_0000;
    const QUIET: u64 = 0x0008_0000_0000_0000;
    let lhs_nan = lhs & MAG > INF;
    let rhs_nan = rhs & MAG > INF;
    if ieee && lhs_nan && lhs & QUIET == 0 {
        return lhs | QUIET;
    }
    if ieee && rhs_nan && rhs & QUIET == 0 {
        return rhs | QUIET;
    }
    if rhs_nan {
        return lhs;
    }
    if lhs_nan {
        return rhs;
    }
    if lhs & MAG == 0 && rhs & MAG == 0 {
        return if maximum { lhs & rhs } else { lhs | rhs };
    }
    let (a, b) = (f64::from_bits(lhs), f64::from_bits(rhs));
    if maximum {
        if a > b {
            lhs
        } else {
            rhs
        }
    } else if a < b {
        lhs
    } else {
        rhs
    }
}

const DENORMAL_ROWS: [[u64; 2]; 32] = [
    [0x0000000000000001, 0x0000000000000000],
    [0x0000000000000000, 0x0000000000000001],
    [0x8000000000000001, 0x0000000000000000],
    [0x0000000000000000, 0x8000000000000001],
    [0x0000000000000001, 0x8000000000000001],
    [0x8000000000000001, 0x0000000000000001],
    [0x000fffffffffffff, 0x800fffffffffffff],
    [0x800fffffffffffff, 0x000fffffffffffff],
    [0x000fffffffffffff, 0x0010000000000000],
    [0x0010000000000000, 0x000fffffffffffff],
    [0x800fffffffffffff, 0x8010000000000000],
    [0x8010000000000000, 0x800fffffffffffff],
    [0x0000000000000001, 0x000fffffffffffff],
    [0x000fffffffffffff, 0x0000000000000001],
    [0x8000000000000001, 0x800fffffffffffff],
    [0x800fffffffffffff, 0x8000000000000001],
    [0x7ff8000000000123, 0x0000000000000001],
    [0x0000000000000001, 0x7ff8000000000123],
    [0xfff8000000000123, 0x800fffffffffffff],
    [0x800fffffffffffff, 0xfff8000000000123],
    [0x7ff0000000000123, 0x0000000000000001],
    [0x0000000000000001, 0x7ff0000000000123],
    [0xfff0000000000123, 0x800fffffffffffff],
    [0x800fffffffffffff, 0xfff0000000000123],
    [0x7ff0000000000000, 0x000fffffffffffff],
    [0x000fffffffffffff, 0x7ff0000000000000],
    [0xfff0000000000000, 0x800fffffffffffff],
    [0x800fffffffffffff, 0xfff0000000000000],
    [0x0000000000000000, 0x8000000000000000],
    [0x8000000000000000, 0x0000000000000000],
    [0x0010000000000000, 0x8010000000000000],
    [0x8010000000000000, 0x0010000000000000],
];

/// v_min/max_f64 inputs with a zero exponent are zeros of their sign when
/// FLOAT_MODE flushes f64 input denormals (bit 6 clear), after the abs/neg
/// modifiers, in both IEEE modes.
#[test]
fn min_max_denormal_inputs() {
    let mut input = vec![0u32; 32 * 4];
    for (lane, row) in DENORMAL_ROWS.iter().enumerate() {
        for s in 0..2 {
            input[lane * 4 + s * 2] = row[s] as u32;
            input[lane * 4 + s * 2 + 1] = (row[s] >> 32) as u32;
        }
    }
    let mut failures = Vec::new();
    for (abs, neg) in [(0u32, 0u32), (1, 2), (2, 1), (3, 3)] {
        let mut code = CODE;
        for w in [13usize, 15] {
            code[w] |= abs << 8;
            code[w + 1] |= neg << 29;
        }
        for denormals in [-1i32, 0, 2, 3] {
            for ieee in [false, true] {
                if denormals < 0 && ieee {
                    continue;
                }
                let mode = if denormals < 0 {
                    ps5shader::translate::FloatMode::default()
                } else {
                    ps5shader::translate::FloatMode {
                        mode: 0x30 | ((denormals as u8) << 6),
                        dx10_clamp: true,
                        ieee,
                        fp16_overflow: false,
                    }
                };
                let out = fmod_dispatch_mode(&code, 32, 32, &input, 32 * 16, 0, mode)
                    .unwrap_or_else(|e| panic!("{}", &e[..e.len().min(600)]));
                for (lane, row) in DENORMAL_ROWS.iter().enumerate() {
                    let mut pair = *row;
                    for (s, v) in pair.iter_mut().enumerate() {
                        if (abs >> s) & 1 != 0 {
                            *v &= 0x7fff_ffff_ffff_ffff;
                        }
                        if (neg >> s) & 1 != 0 {
                            *v ^= 0x8000_0000_0000_0000;
                        }
                        if (denormals == 0 || denormals == 2) && *v & 0x7ff0_0000_0000_0000 == 0 {
                            *v &= 0x8000_0000_0000_0000;
                        }
                    }
                    for op in 0..2 {
                        let e = min_max_reference(pair[0], pair[1], op != 0, ieee);
                        let a = out[lane * 16 + 8 + op * 2] as u64
                            | (out[lane * 16 + 9 + op * 2] as u64) << 32;
                        if a != e {
                            failures.push(format!("mode {denormals} ieee {ieee} abs {abs} neg {neg} lane {lane} {}: {a:#018x}, expected {e:#018x}", if op == 0 { "min" } else { "max" }));
                        }
                    }
                }
            }
        }
    }
    fmod_report(&failures);
}
