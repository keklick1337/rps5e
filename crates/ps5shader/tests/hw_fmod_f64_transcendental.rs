// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! v_rcp/rsq/sqrt/trig_preop_f64 against hardware-measured reference rows.
//! Approximate results with a normal expected value may differ by a
//! relative error of 2^-23 (as in the reference check); everything else is
//! exact.

include!("vectors/fmod_common.rs");
include!("vectors/fmod_f64_transcendental.rs");

const NAMES: [&str; 8] = ["rcp", "rsq", "sqrt", "trig preop", "trig preop inline", "trig preop abs neg omod", "rcp clamp", "sqrt abs neg"];
const APPROX: [bool; 8] = [true, true, true, false, false, false, true, true];

fn f64_of(lo: u32, hi: u32) -> f64 {
    f64::from_bits((hi as u64) << 32 | lo as u64)
}

fn check(i: usize) {
    // Measured with FLOAT_MODE 0xF0 and IEEE_MODE.
    fmod_set_float_mode(FMOD_IEEE_F0);
    let out = match fmod_run_rows(&CODE, &ROWS) {
        Ok(o) => o,
        Err(e) => panic!("{}", &e[..e.len().min(600)]),
    };
    let mut failures = Vec::new();
    for (lane, row) in ROWS.iter().enumerate() {
        let (alo, ahi) = (out[lane * 16 + 2 * i], out[lane * 16 + 2 * i + 1]);
        let (elo, ehi) = (EXPECTED[lane][2 * i], EXPECTED[lane][2 * i + 1]);
        let exponent = (ehi >> 20) & 0x7ff;
        let ok = if APPROX[i] && exponent != 0 && exponent != 0x7ff {
            let (a, e) = (f64_of(alo, ahi), f64_of(elo, ehi));
            (a - e).abs() / e.abs() <= 2f64.powi(-23)
        } else {
            alo == elo && ahi == ehi
        };
        if !ok {
            failures.push(format!("lane {lane} {row:08x?} {}: {ahi:08x}_{alo:08x}, expected {ehi:08x}_{elo:08x}", NAMES[i]));
        }
    }
    fmod_report(&failures);
}

macro_rules! columns {
    ($($(#[$m:meta])* $name:ident => $i:expr;)*) => { $( #[test] $(#[$m])* fn $name() { check($i); } )* };
}

columns! {
    rcp => 0;
    rsq => 1;
    sqrt => 2;
    trig_preop => 3;
    trig_preop_inline => 4;
    trig_preop_abs_neg_omod => 5;
    rcp_clamp => 6;
    sqrt_abs_neg => 7;
}
