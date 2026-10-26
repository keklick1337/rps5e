// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Wave reductions in pixel programs on fragment subgroups that hold only
//! the covered pixels (the reference driver's PixelWaveReduction test):
//! each pixel keys itself from its position, reduces the keys over the
//! wave with row_shr DPP scans, v_permlanex16_b32 and v_readlane_b32 of
//! lanes 31 and 63, and stores (first lane's pixel << 16 | the reduction)
//! at its pixel index. The lanes a partial subgroup lacks must not change
//! the result (`RPS5E_LANE_REDUCTIONS`).
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored pixel_reduction`.

use super::*;
use std::collections::BTreeMap;

const WIDTH: u32 = 192;
const HEIGHT: u32 = 128;
const SENTINEL: u32 = 0xdead_beef;

struct Reduction {
    name: &'static str,
    identity: u32,
    vector: u32,
    scalar: u32,
    combine: fn(u32, u32) -> u32,
}

const REDUCTIONS: [Reduction; 5] = [
    Reduction { name: "UMin", identity: 0xffff_ffff, vector: 0x2600_0000, scalar: 0x8380_0000, combine: |a, b| a.min(b) },
    Reduction { name: "UMax", identity: 0, vector: 0x2800_0000, scalar: 0x8480_0000, combine: |a, b| a.max(b) },
    Reduction { name: "IAdd", identity: 0, vector: 0x4a00_0000, scalar: 0x8000_0000, combine: |a, b| a.wrapping_add(b) },
    Reduction { name: "AND", identity: 0xffff_ffff, vector: 0x3600_0000, scalar: 0x8700_0000, combine: |a, b| a & b },
    Reduction { name: "OR", identity: 0, vector: 0x3800_0000, scalar: 0x8800_0000, combine: |a, b| a | b },
];

fn key(pixel: u32) -> u32 {
    pixel.wrapping_mul(0x9e37_79b1) >> 22
}

/// v2/v3 = pixel x/y; v6 = pixel index; v11 = key; seed v13 = key in the
/// entry EXEC (VCC), the identity elsewhere; four row_shr scans, a
/// v_permlanex16 and the readlanes of 31 and 63; v5 = first lane's pixel
/// << 16 | the reduction, stored at index v6 through s[4:7].
fn pixel_code(r: &Reduction) -> Vec<u32> {
    let dpp = r.vector | 0x001a_1afa;
    vec![
        0x7e040f00, 0x7e060f01, 0xd5430006, 0x0409ff03, 0x000000c0, 0x7e120506, 0xd569000b, 0x0001ff06, 0x9e3779b1, 0x2c161696, 0xbe98047e, 0x879a187e,
        0xbeea287e, 0xd501000d, if r.identity == 0 { 0x006a1680 } else { 0x006a16c1 }, dpp, 0xff01110d, dpp, 0xff01120d, dpp, 0xff01140d, dpp, 0xff01180d,
        0xd778100c, 0x0305830d, r.vector | 0x001a190d, 0xbefe046a, 0xd760000a, 0x00013f0d, 0xd760000b, 0x00017f0d, r.scalar | 0x000c0b0a, 0x8f099009,
        0x880c090c, 0x7e0a020c, 0xe0702000, 0x80010506, 0x7e0e02f2, 0xf800180f, 0x07070707, 0xbf810000,
    ]
}

/// The row variant (the reference's PixelRowReduction): the scans' row
/// totals read at lanes 15, 31, 47 and 63 and combined by scalar ops.
fn row_code(r: &Reduction) -> Vec<u32> {
    let dpp = r.vector | 0x001a_1afa;
    vec![
        0x7e040f00, 0x7e060f01, 0xd5430006, 0x0409ff03, 0x000000c0, 0x7e120506, 0xd569000b, 0x0001ff06, 0x9e3779b1, 0x2c161696, 0xbe98047e, 0x879a187e,
        0xbeea287e, 0xd501000d, if r.identity == 0 { 0x006a1680 } else { 0x006a16c1 }, dpp, 0xff01110d, dpp, 0xff01120d, dpp, 0xff01140d, dpp, 0xff01180d,
        0xbefe046a, 0xd760000a, 0x00011f0d, 0xd760000b, 0x00013f0d, r.scalar | 0x000c0b0a, 0xd760000a, 0x00015f0d, r.scalar | 0x000c0a0c, 0xd760000a,
        0x00017f0d, r.scalar | 0x000c0a0c, 0x8f099009, 0x880c090c, 0x7e0a020c, 0xe0702000, 0x80010506, 0x7e0e02f2, 0xf800180f, 0x07070707, 0xbf810000,
    ]
}

/// `code` with a v_movreld_b32 v0, s20 window before the final colour
/// (the reference's windowed variant).
fn windowed(mut code: Vec<u32>) -> Vec<u32> {
    let at = code.iter().position(|&w| w == 0x7e0e02f2).unwrap();
    code.insert(at, 0x7e00_8514);
    code
}

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

#[test]
#[ignore]
fn pixel_reduction_over_partial_fragment_subgroups() {
    let mut g = Guest::new(32 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let tris = triangles();
    let verts = g.alloc(&floats(&tris));
    let count = (tris.len() / 4) as u32;
    let mut programs: Vec<(usize, &str, u64)> = Vec::new();
    for (k, r) in REDUCTIONS.iter().enumerate() {
        for (shape, code) in [("wave", pixel_code(r)), ("rows", row_code(r))] {
            programs.push((k, shape, g.program(&code)));
            programs.push((k, shape, g.program(&windowed(code))));
        }
    }
    let rest = TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 });
    let inputs = vec4_attrs(&[(verts, count as u64 * 16, 16, false)], &[(0, 0, 0)]);
    let pixel_programs: Vec<u64> = programs.iter().map(|&(_, _, p)| p).collect();
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(SpirvVertices { spirv: crate::shader::SpirvCompiler::for_device(dev, g.mem), vertices: pixel_programs, rest }), Some(inputs), None);
    let host = rig.dev.caps.subgroup_size;
    let mut failures = Vec::new();
    for (k, shape, ps) in programs {
        let r = &REDUCTIONS[k];
        let name = format!("{} {shape}", r.name);
        let target = rgba8_target(g.zeroed((WIDTH * HEIGHT * 4) as usize), WIDTH, HEIGHT);
        let results = g.alloc(&asm::bytes(&vec![SENTINEL; (WIDTH * HEIGHT) as usize]));
        let mut user = vec![0u32; 4];
        user.extend_from_slice(&[results as u32, ((results >> 32) as u32 & 0xffff) | (4 << 16), WIDTH * HEIGHT, 0x0101_6fac]);
        let mut d = draw_into(target, stage(vs, &[]), stage(ps, &user), primitive::TRI_LIST, auto(count, 1), None);
        // POS_X_FLOAT and POS_Y_FLOAT in v0 / v1.
        d.ps_input_ena = (1 << 8) | (1 << 9);
        d.ps_input_addr = d.ps_input_ena;
        assert_eq!(rig.draw(&d), DrawOutcome::Recorded, "{}", r.name);
        let pixels = rig.pixels(&target);
        assert!(rig.buffers.download(&mut rig.sched, results, (WIDTH * HEIGHT * 4) as u64));
        let mut bytes = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
        assert!(g.mem.read_backing(results, &mut bytes));
        let out: Vec<u32> = bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut waves: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for pixel in 0..WIDTH * HEIGHT {
            let covered = pixels[pixel as usize * 4] == 255;
            let v = out[pixel as usize];
            if !covered {
                if v != SENTINEL {
                    failures.push(format!("{name}: uncovered pixel {pixel} stored {v:#x}"));
                }
                continue;
            }
            if v == SENTINEL {
                failures.push(format!("{name}: covered pixel {pixel} stored nothing"));
                continue;
            }
            waves.entry(v >> 16).or_default().push(pixel);
        }
        let (mut covered, mut partial) = (0usize, 0usize);
        for (first, members) in &waves {
            if !members.contains(first) {
                failures.push(format!("{name}: the first lane's pixel {first} is not a covered pixel of its wave"));
            }
            let want = members.iter().fold(r.identity, |a, &p| (r.combine)(a, key(p)));
            for &p in members {
                if out[p as usize] & 0xffff != want & 0xffff {
                    failures.push(format!("{name}: pixel {p} of a wave of {} read {:#x}, the wave holds {:#x}", members.len(), out[p as usize] & 0xffff, want & 0xffff));
                }
            }
            covered += members.len();
            if members.len() < host as usize {
                partial += 1;
            }
        }
        assert!(covered > 1000, "{}: the triangles covered only {covered} pixels", r.name);
        assert!(partial > 0, "{}: no wave had fewer covered pixels than the host subgroup", r.name);
    }
    failures.truncate(20);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
