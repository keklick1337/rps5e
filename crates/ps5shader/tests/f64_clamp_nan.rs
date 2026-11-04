// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The clamp of an f64 result follows MODE.DX10_CLAMP like f32 and f16
//! (RDNA2 ISA 3.5 and 6.2): with the bit set a NaN result clamps to +0,
//! without it the NaN is written unchanged; other values are limited to
//! [0, 1] either way.

use ps5shader::interp::Machine;
use ps5shader::ssa::SystemValue;
use ps5shader::testing::{buffer_descriptor, ComputeRun};
use ps5shader::translate::FloatMode;

const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const CODE: [u32; 9] = [
    0x3402_0083, // v_lshlrev_b32 v1, 3, v0
    0xE034_1000, // buffer_load_dwordx2 v[4:5], v1, s[0:3] offen
    0x8000_0401,
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xD564_8006, // v_add_f64 v[6:7], v[4:5], 0 clamp
    0x0001_0104,
    0xE074_1000, // buffer_store_dwordx2 v[6:7], v1, s[4:7] offen
    0x8001_0601,
    0xBF81_0000, // s_endpgm
];

const INPUTS: [u64; 6] = [0x7FF8_0000_0000_0000, 0xFFF8_0000_0000_1234, 0x4000_0000_0000_0000, 0xBFF0_0000_0000_0000, 0x3FE0_0000_0000_0000, 0x3FF0_0000_0000_0001];

fn run(dx10_clamp: bool) -> Vec<u64> {
    let n = INPUTS.len();
    let i = buffer_descriptor(IN_BASE, 0, n as u32 * 8, 0x3101_6FAC);
    let o = buffer_descriptor(OUT_BASE, 0, n as u32 * 8, 0x3101_6FAC);
    let user: Vec<u32> = i.iter().chain(o.iter()).copied().collect();
    let mut run = ComputeRun::new(CODE.to_vec(), 32, n as u32, user);
    run.float_mode = FloatMode { mode: 0xC0, dx10_clamp, ieee: false, fp16_overflow: false };
    let input: Vec<u32> = INPUTS.iter().flat_map(|v| [*v as u32, (*v >> 32) as u32]).collect();
    let setup = |m: &mut Machine| {
        m.memory.write_words(IN_BASE, &input);
        m.memory.write_words(OUT_BASE, &vec![0xDEAD_BEEF; n * 2]);
        m.system.insert(SystemValue::LocalInvocationId(0), (0..n as u32).collect());
    };
    let m = run.run_checked(&setup, &[(OUT_BASE, n * 2)]).unwrap();
    m.memory.read_words(OUT_BASE, n * 2).chunks(2).map(|w| w[0] as u64 | (w[1] as u64) << 32).collect()
}

fn is_nan(v: u64) -> bool {
    v & 0x7FFF_FFFF_FFFF_FFFF > 0x7FF0_0000_0000_0000
}

#[test]
fn f64_clamp_of_nan_follows_dx10_clamp() {
    let numbers = [0x3FF0_0000_0000_0000, 0, 0x3FE0_0000_0000_0000, 0x3FF0_0000_0000_0000];
    let on = run(true);
    assert_eq!(&on[..2], &[0, 0], "DX10_CLAMP: NaN clamps to +0");
    assert_eq!(&on[2..], &numbers);
    let off = run(false);
    assert!(is_nan(off[0]) && is_nan(off[1]), "without DX10_CLAMP the NaN stays: {off:016x?}");
    assert_eq!(off[1] >> 63, 1, "the sign of the NaN is kept");
    assert_eq!(&off[2..], &numbers);
}
