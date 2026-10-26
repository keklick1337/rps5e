// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The lanes a pixel wave runs with (the reference driver's
//! PixelHelperLanes, QuadPixelMask and PixelValidMask tests): EXEC holds
//! only the covered pixels, PA_SC_AA_MASK removes the pixels of a quad it
//! masks while implicit derivatives still see them, and a done export
//! with VM writes only the lanes left in EXEC.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored pixel_lanes`.

use super::gfx_exec::*;
use super::*;
use std::collections::BTreeMap;

const SENTINEL: u32 = 0xdead_beef;

/// Six by four skewed triangles over the target.
fn triangles() -> Vec<f32> {
    let (columns, rows) = (6u32, 4u32);
    let mut v = Vec::new();
    for cell in 0..columns * rows {
        let (w, h) = (2.0 / columns as f32, 2.0 / rows as f32);
        let x = -1.0 + w * (cell % columns) as f32;
        let y = -1.0 + h * (cell / columns) as f32;
        let skew = 0.07 * (cell % 5) as f32;
        v.extend_from_slice(&[x + (0.05 + skew) * w, y + 0.05 * h, 0.5, 1.0]);
        v.extend_from_slice(&[x + 0.93 * w, y + (0.11 + skew) * h, 0.5, 1.0]);
        v.extend_from_slice(&[x + (0.37 - skew * 0.5) * w, y + 0.91 * h, 0.5, 1.0]);
    }
    v
}

/// v6 = pixel index; v4 = the lane's place in EXEC (v_mbcnt); s8 = the
/// EXEC count << 23 | the first lane's pixel; stores s8 | v4 << 16 at
/// index v6 through s[4:7].
const HELPER_PIXEL: [u32; 21] = [
    0x7e040f00, 0x7e060f01, 0xd5430006, 0x0409ff03, 0x000000c0, 0xd7650004, 0x0001007e, 0xd7660004, 0x0002087f, 0xbe88107e, 0x7e120506, 0x8f089708,
    0x88080908, 0x34080890, 0x380a0808, 0xe0702000, 0x80010506, 0x7e0e02f2, 0xf800180f, 0x07070707, 0xbf810000,
];

#[test]
#[ignore]
fn pixel_lanes_exec_holds_only_covered_pixels() {
    const W: u32 = 192;
    const H: u32 = 128;
    let mut b = Bench::new();
    let tris = triangles();
    let vs_user = b.positions(&tris);
    let vs = b.words(&VERTEX);
    let ps = b.words(&HELPER_PIXEL);
    for wave32 in [false, true] {
        for pass in 0..4 {
            let what = format!("pixel helper lanes wave{} pass {pass}", if wave32 { 32 } else { 64 });
            let target = b.target(W, H, 0);
            let lanes = b.words(&vec![SENTINEL; (W * H) as usize]);
            let mut ps_user = vec![0u32; 4];
            ps_user.extend_from_slice(&[lanes as u32, ((lanes >> 32) as u32 & 0xffff) | (4 << 16), W * H, 0x0101_6fac]);
            b.draw(target, vs, &vs_user, ps, &ps_user, (tris.len() / 4) as u32, wave32, |d| {
                d.ps_input_ena = POS_XY;
                d.ps_input_addr = POS_XY;
            });
            let pixels = b.rig.pixels(&target);
            let words = b.read(lanes, (W * H) as usize);
            let mut waves: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
            let mut covered = 0;
            for pixel in 0..W * H {
                let word = words[pixel as usize];
                if pixels[pixel as usize * 4] != 255 {
                    assert_eq!(word, SENTINEL, "{what}: uncovered pixel {pixel} stored {word:#x}");
                    continue;
                }
                covered += 1;
                assert_ne!(word, SENTINEL, "{what}: covered pixel {pixel} stored nothing");
                waves.entry(word & 0xffff).or_default().push(((word >> 16) & 0x7f, pixel));
                assert_ne!(word >> 23, 0, "{what}: pixel {pixel} counted no lanes in EXEC");
            }
            assert!(covered > 1000, "{what}: the triangles covered only {covered} pixels");
            for (first, lanes) in &mut waves {
                lanes.sort();
                for (lane, &(place, pixel)) in lanes.iter().enumerate() {
                    assert_eq!(place, lane as u32, "{what}: the wave whose first lane is pixel {first} has {} covered pixels, but pixel {pixel} is lane {place} of EXEC", lanes.len());
                    assert_eq!(words[pixel as usize] >> 23, lanes.len() as u32, "{what}: the wave whose first lane is pixel {first} has {} covered pixels, but EXEC holds {} lanes", lanes.len(), words[pixel as usize] >> 23);
                }
                assert_eq!(lanes[0].1, *first, "{what}: the first lane of EXEC, pixel {first}, is not a covered pixel of its wave");
            }
        }
    }
}

/// s_mov_b64 s[12:13], exec; s_wqm_b64 exec, exec; u = x / 64, v = y /
/// 32; image_sample v[2:5], v[0:1], s[0:7], s[8:11]; s_and_b64 exec,
/// exec, s[12:13]; exp mrt0 v2..v5 done vm.
const SAMPLED_PIXEL: [u32; 13] = [0xbe8c047e, 0xbefe0a7e, 0x100000ff, 0x3c800000, 0x100202ff, 0x3d000000, 0xf0800f08, 0x00400200, 0xbf8c3f70, 0x87fe0c7e, 0xf800180f, 0x05040302, 0xbf810000];

fn level_color(level: u32) -> [u8; 4] {
    [(32 * (level + 1)) as u8, (255 - 16 * level) as u8, (0x40 + level) as u8, 255]
}

#[test]
#[ignore]
fn pixel_lanes_quad_pixel_mask() {
    const W: u32 = 64;
    const H: u32 = 32;
    const LEVELS: u32 = 9;
    let mut b = Bench::new();
    let vs_user = b.positions(&COVER);
    let vs = b.words(&VERTEX);
    let constant = b.words(&WHITE);
    let sampled = b.words(&SAMPLED_PIXEL);
    // A linear 256 x 128 RGBA8 texture whose levels hold their colours.
    let (tw, th) = (W * 4, H * 4);
    let layout = crate::tiling_layout::describe(0, 4, 1, tw, th, 1, LEVELS, false).expect("texture layout");
    let mut texels = vec![0u8; layout.guest_bytes as usize];
    for level in 0..LEVELS {
        let m = layout.mips[level as usize];
        for y in 0..m.height {
            for x in 0..m.width {
                let o = layout.element_offset(level, 0, x, y).unwrap() as usize;
                texels[o..o + 4].copy_from_slice(&level_color(level));
            }
        }
    }
    let base = b.g.alloc(&texels);
    let mut ps_user = vec![
        (base >> 8) as u32,
        ((base >> 40) as u32 & 0xff) | (56 << 20) | (((tw - 1) & 3) << 30),
        ((tw - 1) >> 2) | ((th - 1) << 14),
        0xfac | ((LEVELS - 1) << 16) | (9 << 28),
        0,
        (LEVELS - 1) << 4,
        0,
        0,
    ];
    ps_user.extend_from_slice(&[0, 0xfff << 12, 1 << 26, 0]);
    for mask in [0xfu32, 0x2, 0x9] {
        for (what, ps, color) in [("constant color", constant, [255u8; 4]), ("implicit-LOD sample", sampled, level_color(2))] {
            let target = b.target(W, H, 0);
            b.draw(target, vs, &vs_user, ps, &ps_user, 3, false, |d| {
                d.ps_input_ena = POS_XY;
                d.ps_input_addr = POS_XY;
                // PA_SC_AA_MASK: one sample per pixel, kept where the mask
                // has the pixel's bit.
                let bit = |p: u32| (mask >> p) & 1;
                d.aa_mask = [bit(0) | bit(1) << 16, bit(2) | bit(3) << 16];
            });
            let pixels = b.rig.pixels(&target);
            for y in 0..H {
                for x in 0..W {
                    let covered = (mask >> ((x & 1) | ((y & 1) << 1))) & 1 != 0;
                    let want = if covered { color } else { [0; 4] };
                    assert_eq!(px(&pixels, W, x, y), want, "{what} with quad pixel mask {mask:#x}: pixel ({x}, {y})");
                }
            }
        }
    }
}

/// v2 = x; v_cmpx_eq_u32 0, x & 1 (odd columns leave EXEC); white
/// export done vm.
const MASKED_PIXEL: [u32; 7] = [0x7e040f00, 0x36060481, 0x7da40680, 0x7e0e02f2, 0xf800180f, 0x07070707, 0xbf810000];

#[test]
#[ignore]
fn pixel_lanes_valid_mask() {
    const W: u32 = 64;
    const H: u32 = 32;
    let mut b = Bench::new();
    let vs_user = b.positions(&COVER);
    let vs = b.words(&VERTEX);
    let full = b.words(&WHITE);
    let masked = b.words(&MASKED_PIXEL);
    for wave32 in [false, true] {
        for (ps, is_masked) in [(full, false), (masked, true)] {
            let what = format!("pixel valid mask wave{}{}", if wave32 { 32 } else { 64 }, if is_masked { " odd columns cleared" } else { " full EXEC" });
            let target = b.target(W, H, KEPT);
            b.draw(target, vs, &vs_user, ps, &[0; 8], 3, wave32, |d| {
                d.ps_input_ena = POS_XY;
                d.ps_input_addr = POS_XY;
            });
            let pixels = b.rig.pixels(&target);
            for pixel in 0..W * H {
                let exported = !is_masked || (pixel % W) % 2 == 0;
                let want = if exported { [255; 4] } else { [KEPT; 4] };
                assert_eq!(px(&pixels, W, pixel % W, pixel / W), want, "{what}: pixel {pixel}");
            }
        }
    }
}
