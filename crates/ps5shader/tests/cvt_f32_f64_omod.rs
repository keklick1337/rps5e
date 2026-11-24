// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The output modifier of v_cvt_f32_f64 (mul:2, mul:4, div:2) applies to
//! the converted f32 value under the f32 rule: with IEEE mode off and f32
//! output denormals flushed, ignored otherwise; clamp comes after it.
//! Rows measured on RDNA2 (gfx1032) and RDNA3 (gfx1101): 22 f64 inputs by
//! eight forms (plain, mul:2, mul:4, div:2, then the four with clamp).

use ps5shader::interp::Machine;
use ps5shader::ssa::SystemValue;
use ps5shader::testing::{buffer_descriptor, ComputeRun};
use ps5shader::translate::FloatMode;

const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

/// v1 = tid * 16, v2 = tid * 32; load v[4:7]; v10..v13 = cvt with omod
/// 0..3, v14..v17 the same with clamp; store v[10:13] and v[14:17].
const CODE: [u32; 26] = [
    0x34020084, 0x34040085, 0xe0381000, 0x80000401, 0xbf8c3f70, 0xd58f000a, 0x02010104, 0xd58f000b, 0x0a010104, 0xd58f000c, 0x12010104, 0xd58f000d, 0x1a010104, 0xd58f800e, 0x02010104, 0xd58f800f, 0x0a010104, 0xd58f8010, 0x12010104, 0xd58f8011, 0x1a010104, 0xe0781000, 0x80010a02, 0xe0781010, 0x80010e02, 0xbf810000,
];

const INPUTS: [u64; 22] = [
    0x3ff0000000000000, 0xbff0000000000000, 0x4000000000000000, 0x3fe0000000000000, 0x4330000000000000, 0x47efffffe0000000, 0x47f0000000000000, 0xc7f0000000000000, 0x3810000000000000, 0x380fffffffffffff, 0x381fffffefffffff, 0x3690000000000000, 0x3e80000000000000, 0x7ff8000000000000, 0x7ff4000000000000, 0xfff8000000000000, 0x7ff0000000000000, 0xfff0000000000000, 0x0000000000000000, 0x8000000000000000, 0xbfe0000000000000, 0xc000000000000000,
];

const APPLY: [[u32; 8]; 22] = [
    [0x3f800000, 0x40000000, 0x40800000, 0x3f000000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f000000],
    [0xbf800000, 0xc0000000, 0xc0800000, 0xbf000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0x40000000, 0x40800000, 0x41000000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000],
    [0x3f000000, 0x3f800000, 0x40000000, 0x3e800000, 0x3f000000, 0x3f800000, 0x3f800000, 0x3e800000],
    [0x59800000, 0x5a000000, 0x5a800000, 0x59000000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000],
    [0x7f7fffff, 0x7f800000, 0x7f800000, 0x7effffff, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000],
    [0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000],
    [0xff800000, 0xff800000, 0xff800000, 0xff800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0x00800000, 0x01000000, 0x01800000, 0x00000000, 0x00800000, 0x01000000, 0x01800000, 0x00000000],
    [0x00800000, 0x01000000, 0x01800000, 0x00000000, 0x00800000, 0x01000000, 0x01800000, 0x00000000],
    [0x00ffffff, 0x017fffff, 0x01ffffff, 0x00000000, 0x00ffffff, 0x017fffff, 0x01ffffff, 0x00000000],
    [0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0x34000000, 0x34800000, 0x35000000, 0x33800000, 0x34000000, 0x34800000, 0x35000000, 0x33800000],
    [0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0x7fe00000, 0x7fe00000, 0x7fe00000, 0x7fe00000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0xffc00000, 0xffc00000, 0xffc00000, 0xffc00000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000],
    [0xff800000, 0xff800000, 0xff800000, 0xff800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0xbf000000, 0xbf800000, 0xc0000000, 0xbe800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0xc0000000, 0xc0800000, 0xc1000000, 0xbf800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
];

const IGNORE: [[u32; 8]; 22] = [
    [0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000],
    [0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0x40000000, 0x40000000, 0x40000000, 0x40000000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000],
    [0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000],
    [0x59800000, 0x59800000, 0x59800000, 0x59800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000],
    [0x7f7fffff, 0x7f7fffff, 0x7f7fffff, 0x7f7fffff, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000],
    [0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000],
    [0xff800000, 0xff800000, 0xff800000, 0xff800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0x00800000, 0x00800000, 0x00800000, 0x00800000, 0x00800000, 0x00800000, 0x00800000, 0x00800000],
    [0x00800000, 0x00800000, 0x00800000, 0x00800000, 0x00800000, 0x00800000, 0x00800000, 0x00800000],
    [0x00ffffff, 0x00ffffff, 0x00ffffff, 0x00ffffff, 0x00ffffff, 0x00ffffff, 0x00ffffff, 0x00ffffff],
    [0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0x34000000, 0x34000000, 0x34000000, 0x34000000, 0x34000000, 0x34000000, 0x34000000, 0x34000000],
    [0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0x7fe00000, 0x7fe00000, 0x7fe00000, 0x7fe00000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0xffc00000, 0xffc00000, 0xffc00000, 0xffc00000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000],
    [0xff800000, 0xff800000, 0xff800000, 0xff800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0xbf000000, 0xbf000000, 0xbf000000, 0xbf000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
    [0xc0000000, 0xc0000000, 0xc0000000, 0xc0000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
];

fn run(mode: FloatMode) -> Vec<[u32; 8]> {
    let n = INPUTS.len();
    let i = buffer_descriptor(IN_BASE, 0, n as u32 * 16, 0x3101_6FAC);
    let o = buffer_descriptor(OUT_BASE, 0, n as u32 * 32, 0x3101_6FAC);
    let user: Vec<u32> = i.iter().chain(o.iter()).copied().collect();
    let mut run = ComputeRun::new(CODE.to_vec(), 32, n as u32, user);
    run.float_mode = mode;
    let input: Vec<u32> = INPUTS.iter().flat_map(|v| [*v as u32, (*v >> 32) as u32, 0, 0]).collect();
    let setup = |m: &mut Machine| {
        m.memory.write_words(IN_BASE, &input);
        m.memory.write_words(OUT_BASE, &vec![0xDEAD_BEEF; n * 8]);
        m.system.insert(SystemValue::LocalInvocationId(0), (0..n as u32).collect());
    };
    let m = run.run_checked(&setup, &[(OUT_BASE, n * 8)]).unwrap();
    m.memory.read_words(OUT_BASE, n * 8).chunks(8).map(|w| w.try_into().unwrap()).collect()
}

fn is_nan(v: u32) -> bool {
    v & 0x7FFF_FFFF > 0x7F80_0000
}

fn check(mode: FloatMode, expected: &[[u32; 8]; 22]) {
    for (row, (got, want)) in run(mode).iter().zip(expected).enumerate() {
        for col in 0..8 {
            // NaN payloads of the unclamped forms are not compared.
            if is_nan(want[col]) {
                assert!(is_nan(got[col]), "mode {mode:?} input {:#018x} form {col}: {:#010x} is not a NaN", INPUTS[row], got[col]);
            } else {
                assert_eq!(got[col], want[col], "mode {mode:?} input {:#018x} form {col}", INPUTS[row]);
            }
        }
    }
}

#[test]
fn cvt_f32_f64_output_modifier_applies_without_ieee_and_with_flushed_f32() {
    for mode in [0xC0, 0x40, 0xD0] {
        check(FloatMode { mode, dx10_clamp: true, ieee: false, fp16_overflow: false }, &APPLY);
    }
}

#[test]
fn cvt_f32_f64_output_modifier_is_ignored_in_ieee_mode_or_with_f32_denormals() {
    check(FloatMode { mode: 0xC0, dx10_clamp: true, ieee: true, fp16_overflow: false }, &IGNORE);
    check(FloatMode { mode: 0xF0, dx10_clamp: true, ieee: false, fp16_overflow: false }, &IGNORE);
}
