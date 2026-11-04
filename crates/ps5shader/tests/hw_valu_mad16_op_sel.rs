// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! v_mad_u32_u16 and v_mad_i32_i16 with VOP3 op_sel bits 2 and 3: bit 2
//! takes the high half of src2 as the 32-bit addend, zero-extended for both
//! instructions; bit 3 writes the low 16 bits of the (clamped) sum into the
//! high half of the destination and clears the low half. The expected
//! values are RDNA2 hardware results (8 operand sets x 16 forms).

use ps5shader::testing::{buffer_descriptor, ComputeRun};

const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;
const LANES: usize = 8;
const FORMS: usize = 16;

/// VOP3 word 0: opcode `op` (0x373 v_mad_u32_u16, 0x375 v_mad_i32_i16),
/// destination `vdst`, op_sel bits [0..3], clamp.
const fn vop3(op: u32, vdst: u32, op_sel: u32, clamp: bool) -> u32 {
    0xD400_0000 | (op << 16) | ((clamp as u32) << 15) | (op_sel << 11) | vdst
}

/// VOP3 word 1: src0 = v4, src1 = v5, src2 = `src2` (0x106 v6, 0x107 v7,
/// 0xFF literal).
const fn srcs(src2: u32) -> u32 {
    0x104 | (0x105 << 9) | (src2 << 18)
}

const LITERAL: u32 = 0x1234_5678;

/// (op_sel, clamp, src2) of the 16 forms; forms 0-7 are v_mad_u32_u16,
/// 8-15 the same with v_mad_i32_i16.
const FORM: [(u32, bool, u32); 8] = [(0, false, 0x106), (4, false, 0x106), (8, false, 0x106), (15, false, 0x106), (4, false, 0xFF), (8, true, 0x106), (4, true, 0x107), (12, true, 0x107)];

fn program() -> Vec<u32> {
    let mut code = vec![
        0x3402_0084, // v_lshlrev_b32 v1, 4, v0: input offset
        0x3404_0086, // v_lshlrev_b32 v2, 6, v0: output offset
        0xE038_1000, // buffer_load_dwordx4 v[4:7], v1, s[0:3] offen
        0x8000_0401,
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
    ];
    for (k, op) in [0x373u32, 0x375].into_iter().enumerate() {
        for (i, &(op_sel, clamp, src2)) in FORM.iter().enumerate() {
            code.push(vop3(op, 10 + (k * 8 + i) as u32, op_sel, clamp));
            code.push(srcs(src2));
            if src2 == 0xFF {
                code.push(LITERAL);
            }
        }
    }
    for i in 0..4u32 {
        // buffer_store_dwordx4 v[10 + 4i : 13 + 4i], v2, s[4:7] offen offset:16i
        code.push(0xE078_1000 | (i * 16));
        code.push(0x8001_0002 | ((10 + 4 * i) << 8));
    }
    code.push(0xBF81_0000); // s_endpgm
    code
}

/// (v4, v5, v6, v7) and the 16 results measured on RDNA2.
const VECTORS: [([u32; 4], [u32; FORMS]); LANES] = [
    (
        [0x00030002, 0x00050007, 0x80000001, 0x7fffffff],
        [0x8000000f, 0x0000800e, 0x000f0000, 0x800f0000, 0x00001242, 0x000f0000, 0x0000800d, 0x800d0000, 0x8000000f, 0x0000800e, 0x000f0000, 0x800f0000, 0x00001242, 0x000f0000, 0x0000800d, 0x800d0000],
    ),
    (
        [0xffff0002, 0x0003fffe, 0x7fff8000, 0xffffffff],
        [0x80017ffc, 0x00027ffb, 0x7ffc0000, 0x7ffc0000, 0x00021230, 0x7ffc0000, 0x0002fffb, 0xfffb0000, 0x7fff7ffc, 0x00007ffb, 0x7ffc0000, 0x7ffc0000, 0x00001230, 0x7ffc0000, 0x0000fffb, 0xfffb0000],
    ),
    (
        [0x00020002, 0x00020002, 0xfffe0005, 0x80000000],
        [0xfffe0009, 0x00010002, 0x00090000, 0x00020000, 0x00001238, 0x00090000, 0x00008004, 0x80040000, 0xfffe0009, 0x00010002, 0x00090000, 0x00020000, 0x00001238, 0x00090000, 0x00008004, 0x80040000],
    ),
    (
        [0xffffffff, 0xffffffff, 0xffffffff, 0x7fff7fff],
        [0xfffe0000, 0xffff0000, 0x00000000, 0x00000000, 0xfffe1235, 0xffff0000, 0xfffe8000, 0x80000000, 0x00000000, 0x00010000, 0x00000000, 0x00000000, 0x00001235, 0x00000000, 0x00008000, 0x80000000],
    ),
    (
        [0x7fff7fff, 0x7fff7fff, 0x7fffffff, 0x80008000],
        [0xbfff0000, 0x3fff8000, 0x00000000, 0x80000000, 0x3fff1235, 0x00000000, 0x3fff8001, 0x80010000, 0xbfff0000, 0x3fff8000, 0x00000000, 0x80000000, 0x3fff1235, 0xffff0000, 0x3fff8001, 0x80010000],
    ),
    (
        [0x80008000, 0x80008000, 0x00000000, 0xffff0000],
        [0x40000000, 0x40000000, 0x00000000, 0x00000000, 0x40001234, 0x00000000, 0x4000ffff, 0xffff0000, 0x40000000, 0x40000000, 0x00000000, 0x00000000, 0x40001234, 0x00000000, 0x4000ffff, 0xffff0000],
    ),
    (
        [0x1234abcd, 0xfedc5678, 0x9abc0def, 0x0000ffff],
        [0xd4c37407, 0x3a0800d4, 0x74070000, 0xd76c0000, 0x3a07784c, 0x74070000, 0x3a076618, 0x66180000, 0x7e4b7407, 0xe39000d4, 0x74070000, 0xd76c0000, 0xe38f784c, 0x00000000, 0xe38f6618, 0x66180000],
    ),
    (
        [0x00000000, 0x00000000, 0x0000ffff, 0x00010000],
        [0x0000ffff, 0x00000000, 0xffff0000, 0x00000000, 0x00001234, 0xffff0000, 0x00000001, 0x00010000, 0x0000ffff, 0x00000000, 0xffff0000, 0x00000000, 0x00001234, 0xffff0000, 0x00000001, 0x00010000],
    ),
];

#[test]
fn mad16_addend_and_result_halves() {
    let i = buffer_descriptor(IN_BASE, 0, (LANES * 16) as u32, 0x3101_6FAC);
    let o = buffer_descriptor(OUT_BASE, 0, (LANES * FORMS * 4) as u32, 0x3101_6FAC);
    let user: Vec<u32> = i.iter().chain(o.iter()).copied().collect();
    let run = ComputeRun::new(program(), 32, LANES as u32, user);
    let input: Vec<u32> = VECTORS.iter().flat_map(|(ops, _)| ops.iter().copied()).collect();
    for optimize in [false, true] {
        let f = run.compile(optimize).unwrap();
        let m = run
            .run_function(&f, &|m: &mut ps5shader::interp::Machine| {
                m.memory.write_words(IN_BASE, &input);
                m.memory.write_words(OUT_BASE, &[0xDEAD_BEEF; LANES * FORMS]);
            })
            .unwrap();
        let out = m.memory.read_words(OUT_BASE, LANES * FORMS);
        let mut fails = Vec::new();
        for (lane, (ops, expected)) in VECTORS.iter().enumerate() {
            for (form, &e) in expected.iter().enumerate() {
                let got = out[lane * FORMS + form];
                if got != e {
                    fails.push(format!("optimize {optimize}, operands {ops:08x?}, form {form}: {got:#010x}, expected {e:#010x}"));
                }
            }
        }
        assert!(fails.is_empty(), "{} failures:\n{}", fails.len(), fails.join("\n"));
    }
}
