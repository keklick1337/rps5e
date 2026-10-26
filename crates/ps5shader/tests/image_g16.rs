// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! MIMG samples with 16-bit gradients (image_sample_*_g16) decode, and the
//! address layout packs each derivative vector into its own dwords (an odd
//! component padded) with A16 or G16; the coordinates follow, packed only
//! under A16.

use ps5shader::ssa::{ImageAttr, Op};
use ps5shader::testing::asm::S_ENDPGM;
use ps5shader::testing::ComputeRun;
use ps5shader::{decode_instruction, Opcode};

/// MIMG word 0 for opcode `op`, DIM `dim`, DMASK 0xF.
fn mimg0(op: u32, dim: u32) -> u32 {
    0xF000_0000 | (op & 0x7F) << 18 | op >> 7 | dim << 3 | 0xF << 8
}

/// MIMG word 1: VADDR v0, VDATA v8, T# s[0:7], S# s[8:11], A16.
fn mimg1(a16: bool) -> u32 {
    8 << 8 | 2 << 21 | (a16 as u32) << 30
}

fn image_attr(words: [u32; 2]) -> ImageAttr {
    let code = vec![words[0], words[1], S_ENDPGM];
    let f = ComputeRun::new(code, 32, 1, vec![0; 12]).compile(false).unwrap();
    f.values.iter().find_map(|v| if let Op::Image(a) = v.op { Some(a) } else { None }).expect("no image op")
}

#[test]
fn g16_opcodes_decode() {
    let names = [
        (0xA2, Opcode::ImageSampleDG16),
        (0xA3, Opcode::ImageSampleDClG16),
        (0xAA, Opcode::ImageSampleCDG16),
        (0xAB, Opcode::ImageSampleCDClG16),
        (0xB2, Opcode::ImageSampleDOG16),
        (0xB3, Opcode::ImageSampleDClOG16),
        (0xBA, Opcode::ImageSampleCDOG16),
        (0xBB, Opcode::ImageSampleCDClOG16),
        (0xE8, Opcode::ImageSampleCdG16),
        (0xE9, Opcode::ImageSampleCdClG16),
        (0xEA, Opcode::ImageSampleCCdG16),
        (0xEB, Opcode::ImageSampleCCdClG16),
        (0xEC, Opcode::ImageSampleCdOG16),
        (0xED, Opcode::ImageSampleCdClOG16),
        (0xEE, Opcode::ImageSampleCCdOG16),
        (0xEF, Opcode::ImageSampleCCdClOG16),
    ];
    for (op, opcode) in names {
        let ins = decode_instruction(0, &[mimg0(op, 1), mimg1(false)]).unwrap();
        assert_eq!(ins.opcode, opcode, "MIMG {op:#x}");
    }
}

#[test]
fn g16_address_layout() {
    // image_sample_d_g16 2D: (dsdh, dtdh), (dsdv, dtdv), s, t.
    let a = image_attr([mimg0(0xA2, 1), mimg1(false)]);
    assert!(a.flags.g16 && a.flags.derivatives);
    assert_eq!(a.address_dwords, 4);
    // image_sample_c_d_cl_o_g16 3D: offset, compare, 2 + 2 derivative
    // dwords, s, t, r, clamp.
    let a = image_attr([mimg0(0xBB, 2), mimg1(false)]);
    assert_eq!(a.address_dwords, 2 + 4 + 4);
    // image_sample_d_g16 1D: dsdh, dsdv (one dword each), s.
    let a = image_attr([mimg0(0xA2, 0), mimg1(false)]);
    assert_eq!(a.address_dwords, 3);
    // image_sample_d with A16 1D: 32-bit derivatives (A16 packs the
    // coordinates only), then s.
    let a = image_attr([mimg0(0x22, 0), mimg1(true)]);
    assert!(!a.flags.g16);
    assert_eq!(a.address_dwords, 3);
    // image_sample_d with A16 2D: four derivative dwords, (s, t) packed
    // (measured: ImageSampleDerivatives "image_sample_d a16").
    let a = image_attr([mimg0(0x22, 1), mimg1(true)]);
    assert_eq!(a.address_dwords, 5);
    // image_sample_d 2D without 16-bit components: 4 + 2.
    let a = image_attr([mimg0(0x22, 1), mimg1(false)]);
    assert_eq!(a.address_dwords, 6);
}
