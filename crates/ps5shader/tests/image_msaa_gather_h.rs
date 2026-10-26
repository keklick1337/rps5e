// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! image_msaa_load reads one component of four samples (the addressed
//! sample index and the three after it), and image_gather4h is a gather of
//! four horizontally adjacent texels, not a plain gather4.

use ps5shader::interp::{ImageHandler, ImageLane, Machine};
use ps5shader::ssa::{ImageAttr, ImageOp, Op};
use ps5shader::testing::asm::*;
use ps5shader::testing::{raw_buffer_descriptor, ComputeRun};
use ps5shader::{decode_instruction, Opcode};

/// MIMG word 0 for opcode `op`, DIM `dim`, DMASK `dmask`.
fn mimg0(op: u32, dim: u32, dmask: u32) -> u32 {
    0xF000_0000 | (op & 0x7F) << 18 | op >> 7 | dim << 3 | dmask << 8
}

/// MIMG word 1: VADDR v0, VDATA v8, T# s[0:7], S# s[8:11].
fn mimg1(a16: bool) -> u32 {
    8 << 8 | 2 << 21 | (a16 as u32) << 30
}

fn image_ops(code: Vec<u32>) -> Vec<ImageAttr> {
    let f = ComputeRun::new(code, 32, 1, vec![0; 16]).compile(false).unwrap();
    f.values.iter().filter_map(|v| if let Op::Image(a) = v.op { Some(a) } else { None }).collect()
}

#[test]
fn msaa_load_and_horizontal_gather_decode() {
    let ins = decode_instruction(0, &[mimg0(0x80, 6, 1), mimg1(false)]).unwrap();
    assert_eq!(ins.opcode, Opcode::ImageMsaaLoad);
    let ins = decode_instruction(0, &[mimg0(0x61, 1, 1), mimg1(false)]).unwrap();
    assert_eq!(ins.opcode, Opcode::ImageGather4H);
}

#[test]
fn gather4h_is_a_horizontal_gather() {
    let ops = image_ops(vec![mimg0(0x61, 1, 1), mimg1(false), S_ENDPGM]);
    assert_eq!(ops.len(), 1);
    assert_eq!(ops[0].op, ImageOp::Gather4);
    assert!(ops[0].flags.horizontal, "image_gather4h translated as a plain gather4");
    let ops = image_ops(vec![mimg0(0x40, 1, 1), mimg1(false), S_ENDPGM]);
    assert!(!ops[0].flags.horizontal);
}

/// PS5 image_gather4_c_a (0xC8) and image_gather4_c_o_a (0xD8): compare
/// gathers (with an offset for _o) whose address has no extra dword.
#[test]
fn gather4_compare_adjust_forms_decode_and_translate() {
    let ins = decode_instruction(0, &[mimg0(0xC8, 1, 1), mimg1(false)]).unwrap();
    assert_eq!(ins.opcode, Opcode::ImageGather4CA);
    assert_eq!(ins.opcode.mnemonic(), "image_gather4_c_a");
    let ins = decode_instruction(0, &[mimg0(0xD8, 1, 1), mimg1(false)]).unwrap();
    assert_eq!(ins.opcode, Opcode::ImageGather4CAO);
    for (op, offset) in [(0xC8, false), (0xD8, true)] {
        let ops = image_ops(vec![mimg0(op, 1, 1), mimg1(false), S_ENDPGM]);
        assert_eq!(ops.len(), 1, "{op:#x}");
        assert_eq!(ops[0].op, ImageOp::Gather4);
        assert!(ops[0].flags.compare && ops[0].flags.offset == offset && !ops[0].flags.horizontal, "{op:#x}: {:?}", ops[0].flags);
    }
}

#[test]
fn msaa_load_needs_samples_and_one_component() {
    let compile = |words: [u32; 2]| ComputeRun::new(vec![words[0], words[1], S_ENDPGM], 32, 1, vec![0; 16]).compile(false);
    assert!(compile([mimg0(0x80, 1, 1), mimg1(false)]).is_err(), "a 2D (single-sample) dimension");
    assert!(compile([mimg0(0x80, 6, 3), mimg1(false)]).is_err(), "two DMASK bits");
    assert!(compile([mimg0(0x80, 6, 1), mimg1(false)]).is_ok());
}

/// Returns `sample * 16 + x` for loads (address x, y, [slice,] sample).
struct SampleImage;

impl ImageHandler for SampleImage {
    fn image(&mut self, attr: &ImageAttr, lanes: &[ImageLane]) -> Result<Vec<[u32; 4]>, String> {
        if attr.op != ImageOp::Load || !matches!(attr.dim, 6 | 7) || attr.dmask != 1 {
            return Err(format!("unexpected image operation {attr:?}"));
        }
        let sample = if attr.dim == 6 { 2 } else { 3 };
        Ok(lanes.iter().map(|l| [l.address[sample] * 16 + l.address[0], 0, 0, 0]).collect())
    }
}

fn run_msaa_load(dim: u32) -> Vec<u32> {
    const OUTPUT: u64 = 0x2_0000_0000;
    const LANES: u32 = 8;
    let mut code = vec![
        // v1 = 0 (y), v2 = 1 (slice, or the sample index of a 2D image),
        // v3 = 1 (the sample index of an array).
        vop1(OP_V_MOV_B32, 1, inline_u32(0)),
        vop1(OP_V_MOV_B32, 2, inline_u32(1)),
        vop1(OP_V_MOV_B32, 3, inline_u32(1)),
        mimg0(0x80, dim, 1),
        mimg1(false),
        // v12 = lane * 16.
        vop2(0x1a, 12, inline_u32(4), 0),
    ];
    for f in 0..4 {
        code.extend(mubuf(OP_BUFFER_STORE_DWORD, 8 + f, 12, 3, inline_u32(0), 4 * f, false, true, false));
    }
    code.push(S_ENDPGM);
    let mut user = vec![0u32; 16];
    user[12..16].copy_from_slice(&raw_buffer_descriptor(OUTPUT, LANES * 16));
    let run = ComputeRun::new(code, 32, LANES, user);
    let setup = |m: &mut Machine| {
        m.images = Some(Box::new(SampleImage));
        m.memory.write_words(OUTPUT, &[0xdead_beef; (LANES * 4) as usize]);
    };
    let m = run.run_checked(&setup, &[(OUTPUT, (LANES * 4) as usize)]).unwrap();
    m.memory.read_words(OUTPUT, (LANES * 4) as usize)
}

#[test]
fn msaa_load_reads_four_samples() {
    for dim in [6, 7] {
        let out = run_msaa_load(dim);
        for lane in 0..8u32 {
            for f in 0..4u32 {
                assert_eq!(out[(lane * 4 + f) as usize], (1 + f) * 16 + lane, "dim {dim} lane {lane} sample {}", 1 + f);
            }
        }
    }
}
