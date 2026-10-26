// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! F32 transcendental input semantics in the interpreter: v_sin/v_cos take
//! their input in revolutions and reduce it to [0, 1) exactly (inputs of
//! 2^23 and above are whole numbers of revolutions), and v_rcp / v_rsq /
//! v_sqrt / v_log / v_exp flush a denormal input to a zero of its sign.

use ps5shader::testing::asm::*;
use ps5shader::testing::IoTest;

const OP_V_EXP_F32: u32 = 0x25;
const OP_V_LOG_F32: u32 = 0x27;
const OP_V_RCP_F32: u32 = 0x2A;
const OP_V_RSQ_F32: u32 = 0x2E;
const OP_V_SQRT_F32: u32 = 0x33;
const OP_V_SIN_F32: u32 = 0x35;
const OP_V_COS_F32: u32 = 0x36;

/// output[tid] = op(input[tid]) for a VOP1 `op`.
fn unary(op: u32, inputs: &[u32]) -> Vec<u32> {
    let mut code = vec![vop2(0x1A, 1, inline_u32(2), 0)]; // v_lshlrev_b32 v1, 2, v0
    code.extend(mubuf(OP_BUFFER_LOAD_DWORD, 2, 0, 0, inline_u32(0), 0, true, false, false));
    code.push(vop1(op, 3, vgpr(2)));
    code.extend(mubuf(OP_BUFFER_STORE_DWORD, 3, 0, 1, inline_u32(0), 0, true, false, false));
    code.push(S_ENDPGM);
    let t = IoTest::new(&code, inputs.len() as u32, inputs.len());
    t.run(inputs).unwrap()
}

fn assert_bits(what: &str, inputs: &[u32], out: &[u32], expected: &[u32]) {
    for (k, (&o, &e)) in out.iter().zip(expected).enumerate() {
        assert_eq!(o, e, "{what}({:#010x}) is {o:#010x}, expected {e:#010x}", inputs[k]);
    }
}

fn assert_near(what: &str, inputs: &[u32], out: &[u32], expected: &[f32]) {
    for (k, (&o, &e)) in out.iter().zip(expected).enumerate() {
        let a = f32::from_bits(o);
        assert!((a - e).abs() <= 4e-6, "{what}({}) is {a}, expected {e}", f32::from_bits(inputs[k]));
    }
}

#[test]
fn sin_cos_large_inputs_are_whole_revolutions() {
    // 2^23, -2^24, 1e10, f32::MAX: no fraction left.
    let inputs = [0x4B00_0000, 0xCB80_0000, 0x5015_02F9, 0x7F7F_FFFF];
    assert_bits("v_sin", &inputs, &unary(OP_V_SIN_F32, &inputs), &[0; 4]);
    assert_bits("v_cos", &inputs, &unary(OP_V_COS_F32, &inputs), &[0x3F80_0000; 4]);
}

#[test]
fn sin_cos_reduce_the_fraction_exactly() {
    // Large arguments keep their exact fraction: x * 2pi in f32 would lose it.
    let inputs = [1000.25f32, 12345.75, -0.25, 4_194_303.25, 0.125, -77.875].map(f32::to_bits);
    let tau = std::f64::consts::TAU;
    let fract = |x: f32| {
        let x = x as f64;
        x - x.floor()
    };
    let sin: Vec<f32> = inputs.iter().map(|&b| (fract(f32::from_bits(b)) * tau).sin() as f32).collect();
    let cos: Vec<f32> = inputs.iter().map(|&b| (fract(f32::from_bits(b)) * tau).cos() as f32).collect();
    assert_near("v_sin", &inputs, &unary(OP_V_SIN_F32, &inputs), &sin);
    assert_near("v_cos", &inputs, &unary(OP_V_COS_F32, &inputs), &cos);
}

#[test]
fn sin_cos_signed_zero_and_specials() {
    // +0, -0, half a revolution below 2^23, +inf, NaN.
    let inputs = [0x0000_0000, 0x8000_0000, 0x4AFF_FFFF, 0x7F80_0000, 0x7FC0_0000];
    assert_bits("v_sin", &inputs, &unary(OP_V_SIN_F32, &inputs), &[0x0000_0000, 0x8000_0000, 0x0000_0000, 0xFFC0_0000, 0x7FC0_0000]);
    assert_bits("v_cos", &inputs, &unary(OP_V_COS_F32, &inputs), &[0x3F80_0000, 0x3F80_0000, 0xBF80_0000, 0xFFC0_0000, 0x7FC0_0000]);
}

/// From -126 up v_exp_f32 stays at or above the smallest normal (the host's
/// approximation may round just below it and flush); below -126 it is +0.
#[test]
fn exp_keeps_the_smallest_normal_from_minus_126() {
    let inputs = [(-126.0f32).to_bits(), (-125.999_99f32).to_bits(), (-126.000_01f32).to_bits(), (-127.0f32).to_bits()];
    let out = unary(OP_V_EXP_F32, &inputs);
    assert_eq!(out[0], 0x0080_0000, "v_exp(-126)");
    assert!(out[1] >= 0x0080_0000 && out[1] < 0x0081_0000, "v_exp(-125.99999) is {:#010x}", out[1]);
    assert_eq!(out[2], 0, "v_exp(-126.00001)");
    assert_eq!(out[3], 0, "v_exp(-127)");
}

#[test]
fn transcendental_inputs_flush_denormals() {
    // Smallest and largest denormals of either sign.
    let inputs = [0x0000_0001, 0x8000_0001, 0x007F_FFFF, 0x807F_FFFF];
    let run = |op| unary(op, &inputs);
    assert_bits("v_rcp", &inputs, &run(OP_V_RCP_F32), &[0x7F80_0000, 0xFF80_0000, 0x7F80_0000, 0xFF80_0000]);
    assert_bits("v_rsq", &inputs, &run(OP_V_RSQ_F32), &[0x7F80_0000, 0xFF80_0000, 0x7F80_0000, 0xFF80_0000]);
    assert_bits("v_sqrt", &inputs, &run(OP_V_SQRT_F32), &[0x0000_0000, 0x8000_0000, 0x0000_0000, 0x8000_0000]);
    assert_bits("v_log", &inputs, &run(OP_V_LOG_F32), &[0xFF80_0000; 4]);
    assert_bits("v_exp", &inputs, &run(OP_V_EXP_F32), &[0x3F80_0000; 4]);
}
