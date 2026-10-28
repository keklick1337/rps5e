// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Pixel stage inputs and exports (the reference driver's
//! PixelAncillaryWord, RelativeAncillary, RelativeStageInputs and
//! PixelUint16Export tests): the ancillary VGPR's primitive class and
//! sample fields, stage input VGPRs read and written through M0-relative
//! moves, and UINT16_ABGR exports into a 16_16_16_16 UINT target under
//! each component order.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored pixel_inputs`.

use super::gfx_exec::*;
use super::*;

/// Exports (ancillary & 0xffff, ancillary >> 16, 0, 1) from the ancillary
/// VGPR `vgpr`, or (its sample field, 0, 0, 1) unless `whole`.
fn ancillary_program(vgpr: u32, whole: bool) -> Vec<u32> {
    let mut c = Vec::new();
    if whole {
        c.extend([0x3600_0000 | 20 << 17 | vgpr << 9 | 0xff, 0x0000_ffff, 0x2c00_0000 | 21 << 17 | vgpr << 9 | 0x90]);
    } else {
        c.extend([0xd548_0000 | 20, 0x84 << 18 | 0x88 << 9 | (256 + vgpr), 0x7e00_0280 | 21 << 17]);
    }
    c.extend([0x7e00_0c00 | 20 << 17 | (256 + 20), 0x7e00_0c00 | 21 << 17 | (256 + 21), 0x7e00_0280 | 22 << 17, 0x7e00_02f2 | 23 << 17]);
    c.extend([0xf800_180f, 0x1716_1514, 0xbf81_0000]);
    c
}

/// Horizontal lines through the centres of `h` rows, past both sides.
fn row_lines(h: u32) -> Vec<f32> {
    let mut v = Vec::new();
    for y in 0..h {
        let ndc = 1.0 - 2.0 * (y as f32 + 0.5) / h as f32;
        v.extend_from_slice(&[-1.5, ndc, 0.5, 1.0, 1.5, ndc, 0.5, 1.0]);
    }
    v
}

#[test]
#[ignore]
fn pixel_inputs_ancillary_word() {
    const W: u32 = 64;
    const H: u32 = 16;
    let mut b = Bench::new();
    let vs = b.words(&VERTEX);
    let cover = b.positions(&COVER);
    let lines = row_lines(H);
    let line_user = b.positions(&lines);
    // (inputs, ancillary VGPR, primitive type, class, whole word)
    let cases = [
        ("ANCILLARY alone, triangle", ANCILLARY, 0, primitive::TRI_LIST, 2, true),
        ("ANCILLARY after the perspective center, rectangle", PERSP_CENTER | ANCILLARY, 2, primitive::RECT_LIST, 3, true),
        ("ANCILLARY alone, line", ANCILLARY, 0, primitive::LINE_LIST, 1, true),
        ("sample field only, triangle", ANCILLARY, 0, primitive::TRI_LIST, 0, false),
        ("sample field only, line", ANCILLARY, 0, primitive::LINE_LIST, 0, false),
    ];
    for (what, inputs, vgpr, prim, word, whole) in cases {
        let ps = b.words(&ancillary_program(vgpr, whole));
        // 32_32_32_32 FLOAT.
        let target = b.target_of(W, H, 14, 7, 0, 0xff);
        let (user, count) = if prim == primitive::LINE_LIST { (&line_user, (lines.len() / 4) as u32) } else { (&cover, 3) };
        b.draw(target, vs, user, ps, &[0; 8], count, false, |d| {
            d.primitive_type = prim;
            d.topology = crate::render::topology(prim);
            d.ps_input_ena = inputs;
            d.ps_input_addr = inputs;
            d.shader_col_format = 9;
        });
        let texels = b.texels(&target, 16);
        let want = [(word & 0xffff) as f32, (word >> 16) as f32, 0.0, 1.0];
        for (i, t) in texels.chunks_exact(16).enumerate() {
            let got: Vec<f32> = t.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            assert_eq!(got, want, "{what}: pixel ({}, {})", i as u32 % W, i as u32 / W);
        }
    }
}

/// Draws `code` over a 64 x 64 target (pixel user data s0 = `offset`,
/// ANCILLARY plus POS_X..W when `implicit`) and expects every byte to be
/// `expected`.
fn relative_ancillary(b: &mut Bench, vs: u64, vs_user: &[u32], code: &[u32], offset: u32, wave32: bool, expected: u8, implicit: bool, what: &str) {
    const EXTENT: u32 = 64;
    let ps = b.words(code);
    let target = b.target(EXTENT, EXTENT, KEPT);
    let inputs = ANCILLARY | if implicit { POS_XY | POS_ZW } else { 0 };
    b.draw(target, vs, vs_user, ps, &[offset], 3, false, |d| {
        d.shaders.stages |= 1 << 22;
        if wave32 {
            d.ps_in_control |= 1 << 15;
        }
        d.ps_input_ena = inputs;
        d.ps_input_addr = inputs;
        d.index_offset = 3;
    });
    let pixels = b.rig.pixels(&target);
    let wrong = pixels.iter().position(|&p| p != expected);
    assert!(wrong.is_none(), "relative ancillary {what} wave{}: byte {} is {}, expected {expected}", if wave32 { 32 } else { 64 }, wrong.unwrap(), pixels[wrong.unwrap()]);
}

/// Three degenerate vertices at the origin, then a triangle over the
/// target.
const OFFSET_POSITIONS: [f32; 24] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.5, 1.0, 3.0, -1.0, 0.5, 1.0, -1.0, 3.0, 0.5, 1.0];

#[test]
#[ignore]
fn pixel_inputs_relative_ancillary() {
    let mut b = Bench::new();
    let vs = b.words(&VERTEX);
    let vs_user = b.positions(&OFFSET_POSITIONS);
    for wave32 in [true, false] {
        let mut run = |code: &[u32], offset: u32, expected: u8, implicit: bool, what: &str| relative_ancillary(&mut b, vs, &vs_user, code, offset, wave32, expected, implicit, what);
        // s_mov_b32 m0, s0; v_movrels_b32 v1, v0; v_bfe_u32 v1, v1, 8, 4;
        // v_cvt_f32_u32 v1, v1; v_add_f32 v1, 1.0, v1; exp v1 x4.
        let read = [0xbefc0300, 0x7e028700, 0xd5480001, 0x02111101, 0x7e020d01, 0x060202f2, 0xf800180f, 0x01010101, 0xbf810000];
        run(&read, 0, 255, false, "packed read");
        run(&read, 4, 255, true, "packed read after the position inputs");
        // v0 = the sample field; v1 = 0; v2 = 1.0; m0 = s0; v_movreld_b32
        // v0, v2; v_movrels_b32 v0, v2; exp v1 x4.
        let prefix = [0xd5480000u32, 0x02111100, 0x7e000d00, 0x7e020280, 0x7e0402f2, 0xbefc0300, 0x7e008502, 0x7e008702];
        let mut code = prefix.to_vec();
        code.extend([0xf800180f, 0x01010101, 0xbf810000]);
        run(&code, 1, 255, false, "in-range write");
        run(&code, 8, 0, false, "out-of-range write");
        // v_swaprel_b32 v1, v0.
        code[6] = 0x7e02d100;
        code.remove(7);
        run(&code, 0x0001_0001, 255, false, "in-range swap");
        run(&code, 0x0001_0008, 0, false, "out-of-range swap");
        // A write with EXEC empty.
        let mut code = prefix[..6].to_vec();
        code.extend([0xbefe0480, 0x7e008502, 0xbefe04c1, 0xf800180f, 0x01010101, 0xbf810000]);
        run(&code, 1, 0, false, "inactive write");
        // DPP writes: rows and banks off, then on.
        for modifier in [0x0000_e402u32, 0xff00_e402] {
            let mut code = prefix[..6].to_vec();
            code.extend([0x7e0084fa, modifier, 0xf800180f, 0x01010101, 0xbf810000]);
            run(&code, 1, if modifier == 0xff00_e402 { 255 } else { 0 }, false, &format!("DPP write {modifier:#x}"));
            run(&code, 8, 0, false, &format!("DPP write {modifier:#x} out of range"));
        }
        // s_mov_b32 m0, s0; v_movrels_b32 v1, v0; v_cvt_f32_u32 v1, v1:
        // the whole word (a triangle's primitive class is not 0).
        let whole = [0xbefc0300, 0x7e028700, 0x7e020d01, 0xf800180f, 0x01010101, 0xbf810000];
        run(&whole, 0, 255, false, "whole word");
        run(&whole, 4, 255, true, "whole word after the position inputs");
    }
}

#[test]
#[ignore]
fn pixel_inputs_relative_stage_inputs() {
    const EXTENT: u32 = 64;
    let mut b = Bench::new();
    let vs_user = b.positions(&OFFSET_POSITIONS);
    // m0 = 5: v1 = v5 (vertex index); m0 = 8: v0 = v8 (instance index);
    // v0 += v1; buffer_load_format_xyzw v[0:3], v0, s[8:11] idxen; exp
    // pos0.
    let relative_vertex = b.words(&[0xbefc0385, 0x7e028700, 0xbefc0388, 0x7e008700, 0x4a000300, 0xe0382000, 0x80020000, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000]);
    let vertex = b.words(&VERTEX);
    // v0 = 1.0; exp v0 x4.
    let white = b.words(&[0x7e0002f2, 0xf800180f, 0x00000000, 0xbf810000]);
    // m0 = 3: v0 = v3 (POS_W_FLOAT, 1.0); exp v0 x4.
    let relative_pixel = b.words(&[0xbefc0383, 0x7e008700, 0xf800180f, 0x00000000, 0xbf810000]);
    for relative_vs in [true, false] {
        let target = b.target(EXTENT, EXTENT, KEPT);
        let (vs, ps) = if relative_vs { (relative_vertex, white) } else { (vertex, relative_pixel) };
        b.draw(target, vs, &vs_user, ps, &[], 3, true, |d| {
            if relative_vs {
                // Vertices 1..3 of instances 0..2: only instance 2's
                // vertex + instance indices reach the covering triangle.
                d.index_offset = 1;
                d.packet.instance_count = 3;
            } else {
                d.index_offset = 3;
                d.ps_input_ena = POS_XY | POS_ZW;
                d.ps_input_addr = POS_XY | POS_ZW;
            }
        });
        let pixels = b.rig.pixels(&target);
        assert!(pixels.iter().all(|&p| p == 255), "relative stage input {} draw did not cover every pixel in white", if relative_vs { "vertex" } else { "pixel" });
    }
    // Compute: m0 = 2: v1 = v2 (thread z); store it at (x + z * 8) * 4.
    let cs = b.words(&[0xbefc0382, 0x7e028700, 0x34020283, 0x4a000300, 0x34000082, 0x7e028700, 0xe0701000, 0x80000100, 0xbf810000]);
    let out = b.words(&[0xdead_beef; 16]);
    let user = ps5shader::testing::buffer_descriptor(out, 0, 64, 0x3101_6fac);
    let mut shader = stage(cs, &user);
    // TIDIG_COMP_CNT 2: thread x, y and z in v0..v2.
    shader.rsrc2 |= 2 << 11;
    let d = ResolvedDispatch {
        packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1 | (1 << 15), ..DispatchPacket::default() },
        shader,
        rsrc3: 0,
        threads: [8, 1, 2],
        start: [0; 3],
    };
    assert_eq!(b.rig.dispatch(&d), DrawOutcome::Recorded);
    let words = b.read(out, 16);
    for (i, &w) in words.iter().enumerate() {
        assert_eq!(w, i as u32 / 8, "relative stage input dispatch did not keep the implicit Z input at output {i}");
    }
}

#[test]
#[ignore]
fn pixel_inputs_uint16_export() {
    const W: u32 = 64;
    const ROWS: [(u32, u32); 5] = [(0x0000ffff, 0xffff0000), (0x00010002, 0xffffffff), (0xffffffff, 0x00010002), (0xffff0000, 0x0000ffff), (0x22221111, 0x44443333)];
    const H: u32 = ROWS.len() as u32 + 1;
    const KEPT16: u8 = 0x5a;
    let mut b = Bench::new();
    let vs = b.words(&VERTEX);
    let cover = b.positions(&COVER);
    // (component order, export mapping, name)
    for (order, mapping, name) in [(0, 0xe4u32, "identity"), (2, 0x1b, "STD_REV"), (3, 0x93, "ALT_REV")] {
        // 16_16_16_16 UINT.
        let target = b.target_of(W, H, 12, 4, order, KEPT16);
        for (row, &(low, high)) in ROWS.iter().enumerate() {
            // v4 = low; v5 = high; exp mrt0 v4, v5 compr done vm.
            let ps = b.words(&[0x7e0802ff, low, 0x7e0a02ff, high, 0xf8001c0f, 0x00000504, 0xbf810000]);
            b.draw(target, vs, &cover, ps, &[0; 8], 3, false, |d| {
                d.shader_col_format = 7;
                let r = Rect { x0: 0, y0: row as i32, x1: W as i32, y1: row as i32 + 1 };
                d.scissor = r;
                d.scissors[0] = r;
            });
        }
        let texels = b.texels(&target, 8);
        for y in 0..H {
            for x in 0..W {
                let o = ((y * W + x) * 8) as usize;
                for c in 0..4u32 {
                    let got = u16::from_le_bytes([texels[o + 2 * c as usize], texels[o + 2 * c as usize + 1]]);
                    let want = match ROWS.get(y as usize) {
                        Some(&(low, high)) => [low as u16, (low >> 16) as u16, high as u16, (high >> 16) as u16][((mapping >> (2 * c)) & 3) as usize],
                        None => KEPT16 as u16 * 0x101,
                    };
                    assert_eq!(got, want, "a UINT16_ABGR export under the {name} mapping stored {got:#x} in component {c} of pixel ({x}, {y}), expected {want:#x}");
                }
            }
        }
    }
}

/// FP16_ABGR, UNORM16_ABGR and SNORM16_ABGR exports into 16_16_16_16 UINT
/// and SINT targets: each exported 16-bit field is stored as it is (a
/// G-buffer clear program writing half-float 1.0 into an integer target
/// leaves 0x3c00, not 1 or 0xffff).
#[test]
#[ignore]
fn pixel_inputs_float16_export_into_integer_target() {
    const W: u32 = 64;
    const H: u32 = 2;
    // Half 1.0, half 0.5, half -2.0, 0xffff; and a row of mixed fields.
    const ROWS: [(u32, u32); 2] = [(0x3800_3c00, 0xffff_c000), (0x8001_7bff, 0x0000_1234)];
    let mut b = Bench::new();
    let vs = b.words(&VERTEX);
    let cover = b.positions(&COVER);
    for (number_type, number_name) in [(4u32, "UINT"), (5, "SINT")] {
        for (col_format, format_name) in [(4u32, "FP16_ABGR"), (5, "UNORM16_ABGR"), (6, "SNORM16_ABGR")] {
            let target = b.target_of(W, H, 12, number_type, 0, 0x5a);
            for (row, &(low, high)) in ROWS.iter().enumerate() {
                // v4 = low; v5 = high; exp mrt0 v4, v5 compr done vm.
                let ps = b.words(&[0x7e0802ff, low, 0x7e0a02ff, high, 0xf8001c0f, 0x00000504, 0xbf810000]);
                b.draw(target, vs, &cover, ps, &[0; 8], 3, false, |d| {
                    d.shader_col_format = col_format;
                    let r = Rect { x0: 0, y0: row as i32, x1: W as i32, y1: row as i32 + 1 };
                    d.scissor = r;
                    d.scissors[0] = r;
                });
            }
            let texels = b.texels(&target, 8);
            for (y, &(low, high)) in ROWS.iter().enumerate() {
                let want = [low as u16, (low >> 16) as u16, high as u16, (high >> 16) as u16];
                for x in 0..W {
                    let o = ((y as u32 * W + x) * 8) as usize;
                    for c in 0..4usize {
                        let got = u16::from_le_bytes([texels[o + 2 * c], texels[o + 2 * c + 1]]);
                        assert_eq!(got, want[c], "a {format_name} export into a 16_16_16_16 {number_name} target stored {got:#x} in component {c} of pixel ({x}, {y}), expected {:#x}", want[c]);
                    }
                }
            }
        }
    }
}
