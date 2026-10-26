// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! VOP3 16-bit interpolation (v_interp_p1ll_f16, v_interp_p1lv_f16,
//! v_interp_p2_f16): decoded and translated with the reference's model, the
//! product-sum of each step rounded once (to f32 for p1, to f16 for p2).
//! The second case lands on an f16 tie after an f32 rounding: rounding the
//! exact sum once gives 0x3c01, rounding twice 0x3c00.

use ps5shader::interp::Machine;
use ps5shader::ssa::InterpParam;
use ps5shader::testing::asm::*;
use ps5shader::testing::IoTest;

const fn vop3_w0(op: u32, dst: u32) -> u32 {
    0xD400_0000 | op << 16 | dst
}
const fn vop3_w1(src0: u32, src1: u32, src2: u32) -> u32 {
    src0 | src1 << 9 | src2 << 18
}

/// v4 = i, v5 = j, v6 = P0 halves (p1lv); results v2 (p1ll), v3 (p2 of
/// v2), v7 (p1lv) stored at output[0..3].
fn program(i: f32, j: f32, p0_word: u32) -> Vec<u32> {
    let mut code = vec![vop1(OP_V_MOV_B32, 4, 255), i.to_bits(), vop1(OP_V_MOV_B32, 5, 255), j.to_bits(), vop1(OP_V_MOV_B32, 6, 255), p0_word, vop1(OP_V_MOV_B32, 3, inline_u32(0))];
    // v_interp_p1ll_f16 v2, v4, attr0.x
    code.extend([vop3_w0(0x342, 2), vop3_w1(0, 256 + 4, 0)]);
    // v_interp_p2_f16 v3, v5, attr0.x, v2
    code.extend([vop3_w0(0x35A, 3), vop3_w1(0, 256 + 5, 256 + 2)]);
    // v_interp_p1lv_f16 v7, v4, attr0.x.high, v6 (P0 from v6's high half)
    code.extend([vop3_w0(0x343, 7), vop3_w1(0x100, 256 + 4, 256 + 6)]);
    for (k, reg) in [2u32, 3, 7].into_iter().enumerate() {
        code.extend(mubuf(OP_BUFFER_STORE_DWORD, reg, 0, 1, inline_u32(0), k as u32 * 4, true, false, false));
    }
    code.push(S_ENDPGM);
    code
}

fn run(code: &[u32], params: [f32; 6]) -> Vec<u32> {
    let t = IoTest::new(code, 1, 3);
    let (ib, ob) = (t.input_base, t.output_base);
    let setup = move |m: &mut Machine| {
        m.memory.write_words(ib, &[0; 3]);
        m.memory.write_words(ob, &[0xDEAD_BEEF; 3]);
        m.interp = Some(Box::new(move |_, _, p, _| match p {
            InterpParam::HalfP0 { high: false } => params[0],
            InterpParam::HalfP10 { high: false } => params[1],
            InterpParam::HalfP20 { high: false } => params[2],
            InterpParam::HalfP0 { high: true } => params[3],
            InterpParam::HalfP10 { high: true } => params[4],
            InterpParam::HalfP20 { high: true } => params[5],
            _ => f32::NAN,
        }));
    };
    let m = t.run.run_checked(&setup, &[(ob, 3)]).unwrap();
    m.memory.read_words(ob, 3)
}

#[test]
fn interpolation_steps() {
    // P0 0.25, P10 1.5, P20 -0.75 (the reference's LowX column); i 0.25,
    // j 0.5. p1lv takes P0 = 2.0 from v6's high half and the high P10 3.0.
    let out = run(&program(0.25, 0.5, 0x4000_0000), [0.25, 1.5, -0.75, 9.0, 3.0, 0.0]);
    assert_eq!(out[0], 0.625f32.to_bits(), "v_interp_p1ll_f16");
    assert_eq!(out[1], 0x3400, "v_interp_p2_f16 (0.25, high half kept)");
    assert_eq!(out[2], 2.75f32.to_bits(), "v_interp_p1lv_f16");
}

#[test]
fn second_step_rounds_once_to_f16() {
    let tiny = f32::from_bits(0x3980_0000); // 2^-12
    let out = run(&program(1.0, tiny, 0), [1.0, f32::from_bits(0x3A00_0000), tiny, 0.0, 0.0, 0.0]);
    assert_eq!(out[0], 0x3F80_1000, "1 + 2^-11");
    assert_eq!(out[1], 0x3C01, "1 + 2^-11 + 2^-24 rounds once to 1 + 2^-10");
}

/// v_interp_p2_f16 with op_sel[3] writes the high half of its destination
/// and keeps the low half (GFX10 16-bit VOP3 results).
#[test]
fn p2_f16_writes_the_high_half_with_op_sel() {
    let mut code = vec![vop1(OP_V_MOV_B32, 5, 255), 0.5f32.to_bits(), vop1(OP_V_MOV_B32, 2, 255), 0.25f32.to_bits(), vop1(OP_V_MOV_B32, 3, 255), 0x1234];
    // v_interp_p2_f16 v3, v5, attr0.x, v2 op_sel:[0,0,0,1]
    code.extend([vop3_w0(0x35A, 3) | 8 << 11, vop3_w1(0, 256 + 5, 256 + 2)]);
    code.extend(mubuf(OP_BUFFER_STORE_DWORD, 3, 0, 1, inline_u32(0), 0, true, false, false));
    code.push(S_ENDPGM);
    let out = run(&code, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    // 0.5 * 1.0 + 0.25 = 0.75 (0x3a00) in the high half.
    assert_eq!(out[0], 0x3A00_1234);
}

/// v_interp_p1_f32_e64 / v_interp_p2_f32_e64 / v_interp_mov_f32_e64 (VOP3
/// opcodes 0x200..0x202): attribute and channel in SRC0, the coordinate
/// (the parameter of the mov form) in SRC1, SRC2 the inline constant 0;
/// a negated coordinate, clamp and omod apply.
#[test]
fn vop3_encoded_f32_interpolation() {
    let mut code = vec![vop1(OP_V_MOV_B32, 4, 255), 0.25f32.to_bits(), vop1(OP_V_MOV_B32, 5, 255), 0.5f32.to_bits()];
    // v_interp_p1_f32_e64 v2, v4, attr1.y
    code.extend([vop3_w0(0x200, 2), vop3_w1(1 | 1 << 6, 256 + 4, 128)]);
    // v_interp_p2_f32_e64 v2, v5, attr1.y
    code.extend([vop3_w0(0x201, 2), vop3_w1(1 | 1 << 6, 256 + 5, 128)]);
    // v_interp_p1_f32_e64 v3, -v4, attr1.y mul:2
    code.extend([vop3_w0(0x200, 3), vop3_w1(1 | 1 << 6, 256 + 4, 128) | 1 << 27 | 1 << 30]);
    // v_interp_mov_f32_e64 v6, p0, attr1.y clamp
    code.extend([vop3_w0(0x202, 6) | 1 << 15, vop3_w1(1 | 1 << 6, 2, 128)]);
    for (k, reg) in [2u32, 3, 6].into_iter().enumerate() {
        code.extend(mubuf(OP_BUFFER_STORE_DWORD, reg, 0, 1, inline_u32(0), k as u32 * 4, true, false, false));
    }
    code.push(S_ENDPGM);
    let t = IoTest::new(&code, 1, 3);
    let (ib, ob) = (t.input_base, t.output_base);
    let setup = move |m: &mut Machine| {
        m.memory.write_words(ib, &[0; 3]);
        m.memory.write_words(ob, &[0xDEAD_BEEF; 3]);
        m.interp = Some(Box::new(move |attr, chan, p, _| {
            if (attr, chan) != (1, 1) {
                return f32::NAN;
            }
            match p {
                InterpParam::P0 => 1.5,
                InterpParam::P10 => 2.0,
                InterpParam::P20 => -1.0,
                _ => f32::NAN,
            }
        }));
    };
    let m = t.run.run_checked(&setup, &[(ob, 3)]).unwrap();
    let out = m.memory.read_words(ob, 3);
    // p1: 1.5 + 0.25 * 2.0 = 2.0; p2: 2.0 + 0.5 * -1.0 = 1.5.
    assert_eq!(out[0], 1.5f32.to_bits(), "p1 then p2");
    // (1.5 - 0.25 * 2.0) * 2 = 2.0.
    assert_eq!(out[1], 2.0f32.to_bits(), "negated coordinate, mul:2");
    // P0 = 1.5 clamped to 1.0.
    assert_eq!(out[2], 1.0f32.to_bits(), "mov p0 with clamp");
}
