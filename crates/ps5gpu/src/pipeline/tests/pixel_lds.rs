// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! LDS and ordered memory in pixel programs (the reference driver's
//! PixelLds, PixelLdsLaneSlots, PixelLdsReadModifyWrite and PixelInterlock
//! tests): each pixel invocation sees its own LDS writes, LDS
//! read-modify-write instructions return and store per invocation, and a
//! primitive-ordered pixel program's buffer read-modify-writes of
//! overlapping triangles do not race.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored pixel_lds`.

use super::gfx_exec::*;

const SENTINEL: u32 = 0xdead_beef;
const W: u32 = 64;

/// v2/v3 = pixel x/y; v6 = pixel index (y * 64 + x).
fn pixel_index_prologue() -> Vec<u32> {
    vec![0x7e040f00, 0x7e060f01, 0xd5430006, 0x0409ff03, W]
}

fn vop2(code: &mut Vec<u32>, opcode: u32, dst: u32, src: u32, vector: u32) {
    code.push(opcode | dst << 17 | vector << 9 | src);
}

/// Every pixel writes 32 slots (x + i) & 31 of three banks with pixel *
/// 256 + i + bank * 32, then reads slot y (& 31 when `bounded`) of each
/// bank and stores the three words at its index (16-byte records).
fn lds_code(bounded: bool) -> Vec<u32> {
    let mut code = pixel_index_prologue();
    vop2(&mut code, 0x3400_0000, 8, 0x88, 6);
    for index in 0..32 {
        vop2(&mut code, 0x4a00_0000, 4, 0x80 + index, 2);
        vop2(&mut code, 0x3600_0000, 9, 0x9f, 4);
        vop2(&mut code, 0x3400_0000, 13, 0x82, 9);
        vop2(&mut code, 0x4a00_0000, 5, 0x80 + index, 8);
        for bank in 0..3 {
            if bank != 0 {
                vop2(&mut code, 0x4a00_0000, 5, 0xa0, 5);
            }
            code.extend([0xd834_0000 | bank * 256, 0x0000_050d]);
        }
    }
    code.push(0xbf8c_c07f);
    if bounded {
        vop2(&mut code, 0x3600_0000, 14, 0x9f, 3);
    }
    vop2(&mut code, 0x3400_0000, 15, 0x82, if bounded { 14 } else { 3 });
    for bank in 0..3 {
        code.extend([0xd8d8_0000 | bank * 256, (10 + bank) << 24 | 15]);
    }
    code.push(0xbf8c_c07f);
    for bank in 0..3 {
        code.extend([0xe070_2000 | bank * 4, 0x8001_0006 | (10 + bank) << 8]);
    }
    code.extend(WHITE);
    code
}

/// Draws `code` over a 64 x `h` target with s[4:7] a V# of 64 * `h`
/// records of `stride` bytes (filled with the sentinel); returns the
/// pixels and the records' words.
fn draw_lds(b: &mut Bench, code: &[u32], h: u32, stride: u32, wave32: bool) -> (Vec<u8>, Vec<u32>) {
    let vs = b.words(&VERTEX);
    let cover = b.positions(&COVER);
    let ps = b.words(code);
    let target = b.target(W, h, 0);
    let words = (W * h * stride / 4) as usize;
    let results = b.words(&vec![SENTINEL; words]);
    let mut user = vec![0u32; 4];
    user.extend_from_slice(&[results as u32, ((results >> 32) as u32 & 0xffff) | stride << 16, W * h, 0x0101_6fac]);
    b.draw(target, vs, &cover, ps, &user, 3, wave32, |d| {
        d.ps_input_ena = POS_XY;
        d.ps_input_addr = POS_XY;
    });
    (b.rig.pixels(&target), b.read(results, words))
}

#[test]
#[ignore]
fn pixel_lds_invocations_see_their_writes() {
    const H: u32 = 32;
    let mut b = Bench::new();
    for wave32 in [false, true] {
        for bounded in [true, false] {
            let what = format!("pixel LDS wave{} {}", if wave32 { 32 } else { 64 }, if bounded { "bounded" } else { "unbounded" });
            for _ in 0..3 {
                let (pixels, results) = draw_lds(&mut b, &lds_code(bounded), H, 16, wave32);
                for pixel in 0..W * H {
                    assert!(pixels[pixel as usize * 4..pixel as usize * 4 + 4].iter().all(|&p| p == 255), "{what}: missing color at pixel {pixel}");
                    let (x, y) = (pixel % W, pixel / W);
                    let index = (y + 32 - x) & 31;
                    for bank in 0..3 {
                        let want = pixel * 256 + index + bank * 32;
                        let got = results[(pixel * 4 + bank) as usize];
                        assert_eq!(got, want, "{what}: pixel {pixel} bank {bank}");
                    }
                    assert_eq!(results[(pixel * 4 + 3) as usize], SENTINEL, "{what}: output stride guard overwritten");
                }
            }
        }
    }
}

#[test]
#[ignore]
fn pixel_lds_lane_slots() {
    const H: u32 = 32;
    let mut code = pixel_index_prologue();
    vop2(&mut code, 0x3400_0000, 8, 0x82, 6);
    // s_mov_b32 m0, 0.
    code.push(0xbefc_0380);
    for bank in 0..3 {
        vop2(&mut code, 0x4a00_0000, 5, 0x80 + bank, 8);
        code.extend([0xdac0_0000 | bank * 256, 0x0000_0500]);
    }
    code.push(0xbf8c_c07f);
    for bank in 0..3 {
        code.extend([0xdac4_0000 | ((bank + 1) % 3) * 256, (10 + bank) << 24]);
    }
    code.push(0xbf8c_c07f);
    for bank in 0..3 {
        code.extend([0xe070_2000 | bank * 4, 0x8001_0006 | (10 + bank) << 8]);
    }
    code.extend(WHITE);
    let mut b = Bench::new();
    for wave32 in [false, true] {
        let what = format!("pixel LDS lane slots wave{}", if wave32 { 32 } else { 64 });
        for _ in 0..3 {
            let (pixels, results) = draw_lds(&mut b, &code, H, 16, wave32);
            for pixel in 0..W * H {
                assert!(pixels[pixel as usize * 4..pixel as usize * 4 + 4].iter().all(|&p| p == 255), "{what}: missing color at pixel {pixel}");
                for bank in 0..3 {
                    let want = pixel * 4 + (bank + 1) % 3;
                    assert_eq!(results[(pixel * 4 + bank) as usize], want, "{what}: pixel {pixel} bank {bank}");
                }
                assert_eq!(results[(pixel * 4 + 3) as usize], SENTINEL, "{what}: output stride guard overwritten");
            }
        }
    }
}

#[test]
#[ignore]
fn pixel_lds_read_modify_write() {
    const H: u32 = 8;
    const RESULTS: u32 = 8;
    const FACTOR: u32 = 0x00a5_c3e1;
    const COUNTER: u32 = 0x1234;
    const SLOTS: u32 = 64;
    const COUNTER_OFFSET: u32 = SLOTS * 16;
    let mut code = pixel_index_prologue();
    let ds = |code: &mut Vec<u32>, opcode: u32, offset: u32, address: u32, data0: u32, data1: u32, dst: u32| {
        code.extend([0xd800_0000 | opcode << 18 | offset, address | data0 << 8 | data1 << 16 | dst << 24]);
    };
    vop2(&mut code, 0x3600_0000, 8, 0xff, 6);
    code.push(SLOTS - 1);
    vop2(&mut code, 0x3400_0000, 8, 0x84, 8);
    code.push(0x7e12_6f06);
    vop2(&mut code, 0x1600_0000, 10, 0xff, 6);
    code.push(FACTOR);
    code.extend([0x7e1c_02c1, 0x7e20_02ff, 0xffff_0000, 0x7e24_0280, 0x7e26_02ff, COUNTER, 0xbefc_03ff, 0x0000_ffff]);
    for offset in (0..16).step_by(4) {
        ds(&mut code, 0x0d, offset, 8, 9, 0, 0);
    }
    ds(&mut code, 0x1e, 0, 8, 10, 0, 0);
    ds(&mut code, 0x1e, 2, 8, 6, 0, 0);
    ds(&mut code, 0xa1, 6, 8, 10, 0, 0);
    ds(&mut code, 0x1f, 8, 8, 6, 0, 0);
    ds(&mut code, 0xa0, 11, 8, 10, 0, 0);
    ds(&mut code, 0x20, 12, 8, 10, 0, 12);
    ds(&mut code, 0x23, 12, 8, 14, 0, 13);
    ds(&mut code, 0x2c, 12, 8, 16, 6, 15);
    ds(&mut code, 0x0d, COUNTER_OFFSET, 18, 19, 0, 0);
    ds(&mut code, 0x3e, COUNTER_OFFSET, 0, 0, 0, 17);
    code.push(0xbf8c_c07f);
    for dword in 0..4 {
        ds(&mut code, 0x36, dword * 4, 8, 0, 0, 20 + dword);
    }
    code.push(0xbf8c_c07f);
    let stored = [20u32, 21, 22, 23, 12, 13, 15, 17];
    for (slot, &v) in stored.iter().enumerate() {
        code.extend([0xe070_2000 | slot as u32 * 4, 0x8001_0006 | v << 8]);
    }
    code.extend(WHITE);
    let expected = |pixel: u32| -> [u32; 8] {
        let initial = !pixel;
        let product = (FACTOR as u64 * pixel as u64) as u32;
        let added = initial.wrapping_add(product);
        let incremented = if added == u32::MAX { 0 } else { added + 1 };
        let masked = (incremented & 0xffff) | pixel;
        [
            (initial & 0xff00_ff00) | (product & 0xff) | ((pixel & 0xff) << 16),
            (initial & 0xffff) | (product & 0xffff_0000),
            (initial & 0x00ff_0000) | (pixel & 0xffff) | (((product >> 16) & 0xff) << 24),
            masked,
            initial,
            added,
            incremented,
            COUNTER,
        ]
    };
    let names = ["ds_write_b8 dword", "ds_write_b16_d16_hi dword", "ds_write_b16 and ds_write_b8_d16_hi dword", "atomic dword", "ds_add_rtn_u32", "ds_inc_rtn_u32", "ds_mskor_rtn_b32", "ds_append"];
    let mut b = Bench::new();
    for wave32 in [false, true] {
        let what = format!("pixel LDS read-modify-write wave{}", if wave32 { 32 } else { 64 });
        for _ in 0..2 {
            let (pixels, results) = draw_lds(&mut b, &code, H, RESULTS * 4, wave32);
            for pixel in 0..W * H {
                assert!(pixels[pixel as usize * 4..pixel as usize * 4 + 4].iter().all(|&p| p == 255), "{what}: missing color at pixel {pixel}");
                for (slot, want) in expected(pixel).into_iter().enumerate() {
                    assert_eq!(results[(pixel * RESULTS) as usize + slot], want, "{what}: pixel {pixel} {}", names[slot]);
                }
            }
        }
    }
}

/// Per pixel: v2 = (y * 64 + x) * 4; buffer_load_dword v4 glc; v4 += 1;
/// buffer_store_dword v4 glc; s_waitcnt_vscnt; s_sendmsg
/// MSG_ORDERED_PS_DONE; white export.
const INTERLOCK_PIXEL: [u32; 17] = [
    0x7e040f00, 0x7e060f01, 0x34060686, 0x4a040702, 0x34040482, 0xe030d000, 0x80000402, 0xbf8c3f70, 0x4a080881, 0xe0705000, 0x80000402, 0xbbfd0000, 0xbf900007, 0x7e0e02f2,
    0xf800180f, 0x07070707, 0xbf810000,
];

#[test]
#[ignore]
fn pixel_lds_interlock_orders_overlapping_triangles() {
    const H: u32 = 32;
    const LAYERS: u32 = 64;
    let mut b = Bench::new();
    if !b.rig.dev.caps.fragment_shader_pixel_interlock {
        eprintln!("skipped, the device lacks fragmentShaderPixelInterlock");
        return;
    }
    let positions: Vec<f32> = (0..LAYERS).flat_map(|_| COVER).collect();
    let vs_user = b.positions(&positions);
    let vs = b.words(&VERTEX);
    let ps = b.words(&INTERLOCK_PIXEL);
    for wave32 in [false, true] {
        for pass in 0..3 {
            let what = format!("pixel interlock wave{} pass {pass}", if wave32 { 32 } else { 64 });
            let target = b.target(W, H, 0);
            let counters = b.words(&vec![0; (W * H) as usize]);
            let user = ps5shader::testing::buffer_descriptor(counters, 0, W * H * 4, 0x3101_6fac);
            b.draw(target, vs, &vs_user, ps, &user, LAYERS * 3, wave32, |d| {
                d.ps_input_ena = POS_XY;
                d.ps_input_addr = POS_XY;
                // DB_SHADER_CONTROL.PRIMITIVE_ORDERED_PIXEL_SHADER.
                d.db_shader_control |= 1 << 16;
            });
            let pixels = b.rig.pixels(&target);
            let counts = b.read(counters, (W * H) as usize);
            for pixel in 0..W * H {
                assert!(pixels[pixel as usize * 4..pixel as usize * 4 + 4].iter().all(|&p| p == 255), "{what}: missing color at pixel {pixel}");
                assert_eq!(counts[pixel as usize], LAYERS, "{what}: pixel {pixel} counted {} of {LAYERS} overlapping read-modify-writes", counts[pixel as usize]);
            }
        }
    }
}
