// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! v_bfm_b32 and s_bfm_b32: the mask is ((1 << width) - 1) << offset in
//! 32-bit arithmetic, so a mask reaching past bit 31 is truncated; the
//! width and offset use their low five bits and s_bfm_b32 keeps SCC. The
//! reference execution program (vector and per-lane scalar results).

use ps5shader::testing::{buffer_descriptor, ComputeRun};

const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const CODE: [u32; 31] = [
    0x34020083, 0x34040084, 0xe0341000, 0x80000401, 0xbf8c3f70, 0xd7630008, 0x00020b04, 0xbe8c0380, 0xbe8d0380, 0xf4240200, 0x18000000, 0xbf8cc07f, 0xbf060d0d, 0x920a0908, 0x850b8081, 0xd7610009,
    0x00001a0a, 0xd761000a, 0x00001a0b, 0xbf070d0d, 0x920a0908, 0x850b8081, 0xd761000b, 0x00001a0b, 0x800c880c, 0x800d810d, 0xbf0aa00d, 0xbf85ffed, 0xe0781000, 0x80010802, 0xbf810000,
];

fn expected_mask(width: u32, offset: u32) -> u32 {
    let mut mask = 0u32;
    let mut bit = offset;
    while bit < 32 && bit - offset < width {
        mask |= 1 << bit;
        bit += 1;
    }
    mask
}

#[test]
fn bitfield_masks() {
    let i = buffer_descriptor(IN_BASE, 0, 32 * 2 * 4, 0x3101_6FAC);
    let o = buffer_descriptor(OUT_BASE, 0, 32 * 4 * 4, 0x3101_6FAC);
    let user: Vec<u32> = i.iter().chain(o.iter()).copied().collect();
    let run = ComputeRun::new(CODE.to_vec(), 32, 32, user);
    let f = run.compile(true).unwrap();
    let mut fails = Vec::new();
    for upper in [0u32, 32, 0x8000_0000, 0xFFFF_FFE0] {
        for width in 0..32u32 {
            let input: Vec<u32> = (0..32u32).flat_map(|offset| [width | upper, offset | upper]).collect();
            let setup = |m: &mut ps5shader::interp::Machine| {
                m.memory.write_words(IN_BASE, &input);
                m.memory.write_words(OUT_BASE, &[0xDEAD_BEEF; 128]);
            };
            let m = run.run_function(&f, &setup).unwrap();
            let out = m.memory.read_words(OUT_BASE, 128);
            for offset in 0..32u32 {
                let e = expected_mask(width, offset);
                let w = &out[offset as usize * 4..offset as usize * 4 + 4];
                if w != [e, e, 1, 0] {
                    fails.push(format!("width {:#x} offset {:#x}: {w:08x?}, expected [{e:#010x}, {e:#010x}, 1, 0]", width | upper, offset | upper));
                }
            }
        }
    }
    assert!(fails.is_empty(), "{} failures:\n{}", fails.len(), fails.iter().take(40).cloned().collect::<Vec<_>>().join("\n"));
}
