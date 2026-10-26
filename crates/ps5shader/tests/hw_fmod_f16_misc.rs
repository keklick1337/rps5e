// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Miscellaneous f16 instructions (ldexp, frexp, cvt_norm, pknorm,
//! cmp_class, sat_pk, legacy multiply) against hardware-measured reference
//! rows.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_f16_misc.rs");

const THREADS: usize = 32;
const INPUTS: usize = 4;
const RESULTS: usize = 16;
const NAMES: [&str; 11] = [
    "v_ldexp_f16", "v_frexp_mant_f16", "v_frexp_exp_i16_f16", "v_cvt_norm_i16_f16", "v_cvt_norm_u16_f16", "v_cvt_pknorm_i16_f16",
    "v_cvt_pknorm_u16_f16", "v_cmp_class_f16", "v_sat_pk_u8_i16", "v_mul_legacy_f32", "v_mac_legacy_f32",
];

fn is_nan32(b: u32) -> bool {
    (b & 0x7fff_ffff) > 0x7f80_0000
}

fn is_nan16(b: u32) -> bool {
    (b & 0x7fff) > 0x7c00
}

fn is_denormal32(b: u32) -> bool {
    b & 0x7f80_0000 == 0 && b & 0x007f_ffff != 0
}

fn matches(column: usize, a: u32, e: u32) -> bool {
    if a == e {
        return true;
    }
    if column <= 1 {
        return a >> 16 == e >> 16 && is_nan16(a) && is_nan16(e);
    }
    column >= 9 && is_nan32(a) && is_nan32(e)
}

fn check(columns: &[usize]) {
    let mut failures = Vec::new();
    for (batch, chunk) in VECTORS.chunks(THREADS).enumerate() {
        let mut input = vec![0u32; THREADS * INPUTS];
        for (lane, v) in chunk.iter().enumerate() {
            input[lane * INPUTS..lane * INPUTS + 3].copy_from_slice(&v[..3]);
        }
        let out = match fmod_dispatch(&MISC_CODE, 32, THREADS as u32, &input, THREADS * RESULTS, 4) {
            Ok(o) => o,
            Err(e) => panic!("{}", &e[..e.len().min(400)]),
        };
        for (lane, v) in chunk.iter().enumerate() {
            let denormal = is_denormal32(v[0]) || is_denormal32(v[1]) || is_denormal32(v[3 + 9]);
            for &c in columns {
                if c >= 9 && denormal {
                    continue;
                }
                let (a, e) = (out[lane * RESULTS + c], v[3 + c]);
                if !matches(c, a, e) {
                    failures.push(format!("vector {} ({:#010x}, {:#010x}, {:#010x}) {}: {a:#010x}, expected {e:#010x}", batch * THREADS + lane, v[0], v[1], v[2], NAMES[c]));
                }
            }
        }
    }
    fmod_report(&failures);
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => $i:expr;)*) => { $( #[test] $(#[$m])* fn $name() { check(&[$i]); } )* };
}

columns! {
    ldexp_f16 => 0;
    frexp_mant_f16 => 1;
    frexp_exp_i16_f16 => 2;
    cvt_norm_i16_f16 => 3;
    cvt_norm_u16_f16 => 4;
    cvt_pknorm_i16_f16 => 5;
    cvt_pknorm_u16_f16 => 6;
    cmp_class_f16 => 7;
    sat_pk_u8_i16 => 8;
    mul_legacy_f32 => 9;
    mac_legacy_f32 => 10;
}
