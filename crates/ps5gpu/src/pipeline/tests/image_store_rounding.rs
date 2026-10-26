// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! image_store to UNORM / SNORM storage images writes the level nearest to
//! the exact product of the clamped value and the format's scale, ties to
//! even (the reference's driver execution test ImageStoreUnormRounding).
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored image_store_rounding`.

use super::image_dims::*;
use super::*;

const WIDTH: u32 = 32;
const HEIGHT: u32 = 8;
const TEXEL_BYTES: usize = 8192;
/// buffer_load_dwordx4 v[20:23] at tid * 16; image_store DMASK 0xF at
/// (tid, 0) and DMASK 0x5 at (tid, 1).
const STORE_CODE: [u32; 12] = [0x34020084, 0xe0381000, 0x80001401, 0xbf8c3f70, 0x7e3c0300, 0x7e3e0280, 0xf0201f08, 0x0001141e, 0x7e3e0281, 0xf0201508, 0x0001141e, 0xbf810000];

struct Format {
    name: &'static str,
    format: u32,
    components: u32,
    bits: u32,
    snorm: bool,
}

const FORMATS: [Format; 5] = [
    Format { name: "8_UNORM", format: 1, components: 1, bits: 8, snorm: false },
    Format { name: "16_16_UNORM", format: 23, components: 2, bits: 16, snorm: false },
    Format { name: "8_8_8_8_UNORM", format: 56, components: 4, bits: 8, snorm: false },
    Format { name: "8_8_8_8_SNORM", format: 57, components: 4, bits: 8, snorm: true },
    Format { name: "16_16_16_16_UNORM", format: 65, components: 4, bits: 16, snorm: false },
];

fn scale(f: &Format) -> f32 {
    ((1u32 << if f.snorm { f.bits - 1 } else { f.bits }) - 1) as f32
}

fn next_after(v: f32, toward: f32) -> f32 {
    if toward > v {
        v.next_up()
    } else if toward < v {
        v.next_down()
    } else {
        v
    }
}

/// The value lane `tid` stores in component `index`: edges, NaN and
/// infinities first, then values a few ulps around the midpoints between
/// levels.
fn value(f: &Format, tid: u32, index: u32) -> f32 {
    const EDGES: [f32; 16] = [0.0, -0.0, 1.0, 0.5, -0.25, 1.5, -1.5, 1e-30, -1e-30, 0.999, 0.0019607844, 0.9980392, 0.33333334, -0.6666667, 2e-5, 0.75];
    let slot = tid * 4 + index;
    if (slot as usize) < EDGES.len() {
        return EDGES[slot as usize];
    }
    match slot as usize - EDGES.len() {
        0 => return f32::NAN,
        1 => return f32::INFINITY,
        2 => return f32::NEG_INFINITY,
        _ => {}
    }
    let s = scale(f);
    let level = ((slot * 37) % s as u32) as f32;
    let mut v = (level + 0.5) / s;
    for _ in 0..slot % 5 {
        v = next_after(v, if slot % 2 == 0 { 2.0 } else { -2.0 });
    }
    if f.snorm && slot % 3 == 0 {
        -v
    } else {
        v
    }
}

/// The stored level: nearest to the exact product, ties to even.
fn level(f: &Format, v: f32) -> u32 {
    let lower = if f.snorm { -1.0 } else { 0.0 };
    let clamped = if v.is_nan() { 0.0 } else { v.clamp(lower, 1.0) };
    let product = clamped as f64 * scale(f) as f64;
    (product.round_ties_even() as i32 as u32) & ((1u32 << f.bits) - 1)
}

#[test]
#[ignore]
fn image_store_rounding_unorm_snorm() {
    let mut failures = Vec::new();
    for f in &FORMATS {
        let mut g = Guest::new(16 << 20);
        let mut rig = rig(&g);
        let bpe = f.components * f.bits / 8;
        let mut shape = Shape { format: f.format, bpe, ..Shape::new(sharps::image_type::TYPE_2D, WIDTH, HEIGHT, 1, 1) };
        shape.tile = 0;
        let layout = shape.layout();
        let base = g.alloc(&initial_bytes(TEXEL_BYTES));
        let mut d = shape.descriptor(base);
        if f.components == 2 {
            d[3] = (d[3] & !0xFFF) | 0x22C;
        }
        let input: Vec<u32> = (0..THREADS).flat_map(|tid| (0..4).map(move |i| (tid, i))).map(|(tid, i)| value(f, tid, i).to_bits()).collect();
        store_dispatch(&mut g, &mut rig, &STORE_CODE, &input, &d);
        let bytes = written_back(&g, &mut rig, base, TEXEL_BYTES as u64);
        for tid in 0..THREADS {
            for (row, dmask) in [(0u32, 0xFu32), (1, 0x5)] {
                let o = layout.element_offset(0, 0, tid, row).unwrap() as usize;
                for c in 0..f.components {
                    let wanted = if (dmask >> c) & 1 == 0 {
                        0
                    } else {
                        let index = (0..c).map(|l| (dmask >> l) & 1).sum();
                        level(f, value(f, tid, index))
                    };
                    let width = (f.bits / 8) as usize;
                    let at = o + c as usize * width;
                    let mut word = [0u8; 4];
                    word[..width].copy_from_slice(&bytes[at..at + width]);
                    let actual = u32::from_le_bytes(word);
                    if actual != wanted {
                        failures.push(format!("image_store of {}: texel ({tid}, {row}) component {c} is {actual:#x}, expected {wanted:#x}", f.name));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.iter().take(40).cloned().collect::<Vec<_>>().join("\n"));
}
