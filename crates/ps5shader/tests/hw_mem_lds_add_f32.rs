// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! ds_add_f32 and ds_add_rtn_f32 against hardware-measured rows (RDNA1
//! gfx1010, the same rule measured on RDNA2 gfx1031): both operands are
//! flushed to a signed zero when bit 0 of the f32 denormal field of
//! FLOAT_MODE is clear and the sum when bit 1 is clear, the sum is rounded
//! to nearest even, a NaN data operand is stored quieted, else a NaN old
//! value is stored quieted, and +inf + -inf stores 0xffc00000. The returned
//! value is the old one. Rows of the reference's LdsAddF32 execution test.

use ps5shader::interp::Machine;
use ps5shader::testing::{buffer_descriptor, ComputeRun};
use ps5shader::translate::FloatMode;

const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;
const LANES: usize = 32;
const RESULTS: usize = 4;

include!("vectors/lds_add_f32.rs");

/// The 32 lanes of rows `first..first + 32` in `mode`: (returned old,
/// stored by ds_add_rtn_f32, stored by ds_add_f32) per lane.
fn run(mode: u8, first: usize) -> Vec<[u32; 3]> {
    let input: Vec<u32> = OPERANDS[first..first + LANES].iter().flatten().copied().collect();
    let out_words = LANES * RESULTS;
    let mut user = buffer_descriptor(IN_BASE, 0, input.len() as u32 * 4, 0x3101_6FAC).to_vec();
    user.extend(buffer_descriptor(OUT_BASE, 0, out_words as u32 * 4, 0x3101_6FAC));
    let mut run = ComputeRun::new(CODE.to_vec(), 32, LANES as u32, user);
    run.float_mode = FloatMode { mode, dx10_clamp: true, ieee: false, fp16_overflow: false };
    let setup = |m: &mut Machine| {
        m.memory.write_words(IN_BASE, &input);
        m.memory.write_words(OUT_BASE, &vec![0xDEAD_BEEF; out_words]);
    };
    let m = run.run_checked(&setup, &[(OUT_BASE, out_words)]).unwrap();
    let out = m.memory.read_words(OUT_BASE, out_words);
    out.chunks(RESULTS).map(|w| [w[0], w[1], w[2]]).collect()
}

#[test]
fn ds_add_f32_measured_rows() {
    let mut failures = Vec::new();
    for (k, &mode) in MODES.iter().enumerate() {
        for first in [0, LANES] {
            for (lane, got) in run(mode, first).into_iter().enumerate() {
                let row = first + lane;
                let [a, b] = OPERANDS[row];
                let want = MEASURED[k][row];
                if got != [a, want, want] {
                    failures.push(format!("FLOAT_MODE {mode:#04x} row {row} ({a:#010x} + {b:#010x}): returned / stored {:08x?}, expected {a:#010x} / {want:#010x}", got));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}
