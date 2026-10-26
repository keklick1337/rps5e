// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Geometry subgroups (GS_EN) run as mesh workgroups through the SPIR-V
//! compiler: the hardware inputs the entry point computes (s2 subgroup
//! counts, s3 wave info) reach the guest program as the hardware gives
//! them, for wave32 and wave64 programs; strip restart in mesh draws; the
//! vertex path's one-wave subgroup inputs; alpha-to-coverage.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored ngg_mesh`.

use super::*;

const STAGES_GS_EN: u32 = 1 << 5;
const STAGES_GS_W32_EN: u32 = 1 << 22;

fn sop2(op: u32, sdst: u32, ssrc0: u32, ssrc1: u32) -> u32 {
    (0b10 << 30) | (op << 23) | (sdst << 16) | (ssrc1 << 8) | ssrc0
}

fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    (0x3F << 25) | (vdst << 17) | (op << 9) | src0
}

fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    (op << 25) | (vdst << 17) | ((vsrc1 & 0xFF) << 9) | src0
}

fn vop3(op: u32, vdst: u32, src0: u32, src1: u32) -> [u32; 2] {
    [(0x35 << 26) | (op << 16) | vdst, (src1 << 9) | src0]
}

const fn v(n: u32) -> u32 {
    256 + n
}

const fn int(n: u32) -> u32 {
    128 + n
}

const LITERAL: u32 = 255;
const M0: u32 = 124;
const MINUS_ONE: u32 = 193;
const ONE_F32: u32 = 242;
const MINUS_ONE_F32: u32 = 243;
const FOUR_F32: u32 = 246;

/// A geometry subgroup program of `wave` lanes per wave: lane `t` of the
/// subgroup (wave index from s3 bits 24..27 times the wave size, plus the
/// lane in the wave) exports corner `t % 3` of triangle `k = t / 3`, which
/// covers column `k` of a 16-column target, and primitive `t` of vertices
/// `3t..3t+2`; GS_ALLOC_REQ asks for the subgroup counts of s2. A lane
/// that takes a wrong wave index places its corner in another column.
fn column_program(g: &mut Guest, wave: u32) -> u64 {
    let shift = if wave == 32 { 5 } else { 6 };
    let mut w = vec![
        // s4 = wave index * wave size
        sop2(32, 4, 3, int(24)),
        sop2(14, 4, 4, int(15)),
        sop2(30, 4, 4, int(shift)),
        // m0 = subgroup vertices | primitives << 12 (from s2)
        sop2(39, 5, 2, LITERAL),
        0x0009_000C,
        sop2(39, 6, 2, LITERAL),
        0x0009_0016,
        sop2(30, 6, 6, int(12)),
        sop2(16, M0, 5, 6),
        // s_sendmsg GS_ALLOC_REQ
        0xBF90_0009,
    ];
    // v1 = t
    w.extend(vop3(0x365, 1, MINUS_ONE, int(0)));
    w.extend(vop3(0x366, 1, MINUS_ONE, v(1)));
    w.push(vop2(0x25, 1, 4, v(1)));
    w.extend([
        // v2 = k = t * 0xAAAB >> 17, v3 = c = t - 3k
        vop2(0x0B, 2, LITERAL, v(1)),
        0xAAAB,
        vop2(0x16, 2, int(17), v(2)),
        vop2(0x0B, 3, int(3), v(2)),
        vop2(0x26, 3, v(1), v(3)),
        // x = -1 + (k + 2 * (c & 1)) / 8
        vop2(0x1B, 4, int(1), v(3)),
        vop2(0x1A, 4, int(1), v(4)),
        vop2(0x25, 4, v(2), v(4)),
        vop1(0x06, 4, v(4)),
        vop2(0x08, 4, LITERAL, v(4)),
        0.125f32.to_bits(),
        vop2(0x03, 4, MINUS_ONE_F32, v(4)),
        // y = -1 + 4 * (c >> 1)
        vop2(0x16, 5, int(1), v(3)),
        vop1(0x06, 5, v(5)),
        vop2(0x08, 5, FOUR_F32, v(5)),
        vop2(0x03, 5, MINUS_ONE_F32, v(5)),
        vop1(0x01, 6, int(0)),
        vop1(0x01, 7, ONE_F32),
        // exp pos0 v4, v5, v6, v7 done
        (0x3E << 26) | (1 << 11) | (12 << 4) | 0xF,
        0x0706_0504,
        // v8 = 3t | (3t + 1) << 10 | (3t + 2) << 20
        vop2(0x0B, 8, int(3), v(1)),
        vop2(0x25, 9, int(1), v(8)),
        vop2(0x25, 10, int(2), v(8)),
        vop2(0x1A, 9, int(10), v(9)),
        vop2(0x1A, 10, int(20), v(10)),
        vop2(0x1C, 8, v(8), v(9)),
        vop2(0x1C, 8, v(8), v(10)),
        // exp prim v8 done
        (0x3E << 26) | (1 << 11) | (20 << 4) | 0x1,
        8,
        asm::S_ENDPGM,
    ]);
    g.program(&w)
}

/// Sixteen triangles from one 64-lane subgroup (two waves of a wave32
/// program, one of a wave64 one): every column is covered only when each
/// wave reads its own index in s3.
#[test]
#[ignore]
fn ngg_mesh_subgroup_waves_read_their_own_wave_info() {
    let mut g = Guest::new(32 << 20);
    let programs = [(32u32, column_program(&mut g, 32)), (64, column_program(&mut g, 64))];
    let ps = solid_program(&mut g);
    let rest = TestCompiler::new().with(ps, TestProgram::SolidPixel);
    let vertices: Vec<u64> = programs.iter().map(|p| p.1).collect();
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(SpirvVertices { spirv: crate::shader::SpirvCompiler::for_device(dev, g.mem), vertices, rest }), None, None);
    if !rig.dev.caps.mesh_shader || rig.dev.mesh.is_none() {
        eprintln!("ngg_mesh: no mesh shaders on this device, skipped");
        return;
    }
    let green = color_words([0.0, 1.0, 0.0, 1.0]);
    for (wave, vs) in programs {
        let target = rgba8_target(g.zeroed(64 * 16 * 4), 64, 16);
        let mut d = draw_into(target, stage(vs, &[]), stage(ps, &green), primitive::TRI_LIST, auto(48, 1), None);
        d.shaders.stages = STAGES_GS_EN | if wave == 32 { STAGES_GS_W32_EN } else { 0 };
        d.shaders.geometry = crate::render::GeometryRegs {
            ge_cntl: 16 | (48 << 9),
            max_output_per_subgroup: 48,
            max_vert_out: 3,
            esgs_ring_item_size: 1,
            out_prim_type: 2,
            user_data_address: 0,
        };
        assert_eq!(rig.draw(&d), DrawOutcome::Recorded, "wave{wave}");
        let p = rig.pixels(&target);
        let missing: Vec<u32> = (0..16u32).filter(|k| px(&p, 64, 4 * k + 2, 12) != [0, 255, 0, 255]).collect();
        assert!(missing.is_empty(), "wave{wave}: columns {missing:?} not covered by their triangles");
    }
}

/// Only a primitive subgroup that generates its primitives without a
/// geometry stage, pass-through or tessellation runs as a mesh workgroup
/// (a pass-through subgroup reads its packed primitive from v0).
#[test]
fn primitive_subgroups_need_primgen_without_passthrough() {
    use crate::pipeline::provider::primitive_subgroup_stages;
    assert!(primitive_subgroup_stages(0x2000), "PRIMGEN_EN");
    assert!(primitive_subgroup_stages(0x2000 | (1 << 22)), "PRIMGEN_EN, wave32");
    assert!(!primitive_subgroup_stages(0x0200_2000), "pass-through");
    assert!(!primitive_subgroup_stages(0x2004), "tessellation");
    assert!(!primitive_subgroup_stages(0x2020), "a geometry stage");
    assert!(!primitive_subgroup_stages(0), "no primitive generation");
}

/// The wave info of every lane of a subgroup, as the entry point computes
/// it on the device.
#[test]
fn merged_wave_info_numbers_the_guest_waves() {
    use crate::shader::mesh::merged_wave_info;
    // 48 vertices and 16 primitives in a 64-lane subgroup.
    assert_eq!(merged_wave_info(0, 48, 16, 64, 32), (2 << 28) | (16 << 8) | 32);
    assert_eq!(merged_wave_info(31, 48, 16, 64, 32), (2 << 28) | (16 << 8) | 32);
    assert_eq!(merged_wave_info(32, 48, 16, 64, 32), (2 << 28) | (1 << 24) | 16);
    assert_eq!(merged_wave_info(63, 48, 16, 64, 32), (2 << 28) | (1 << 24) | 16);
    assert_eq!(merged_wave_info(40, 48, 16, 64, 64), (1 << 28) | (16 << 8) | 48);
    assert_eq!(merged_wave_info(64, 100, 70, 128, 64), (2 << 28) | (1 << 24) | (6 << 8) | 36);
}

/// A passthrough geometry subgroup: lane `i` exports vertex `i` at the
/// corner its vertex ID (v5) names (quad `id / 4`, corner `id % 4`: quad 0
/// over x in [-1, -0.25], quad 1 over [0.25, 1]), lane `t` exports
/// primitive `t` of the ES vertices its v0/v1 offsets name (ESGS item size
/// 1: the offsets are the vertex indices).
fn passthrough_strip_program(g: &mut Guest) -> u64 {
    let w = vec![
        sop2(39, 5, 2, LITERAL),
        0x0009_000C,
        sop2(39, 6, 2, LITERAL),
        0x0009_0016,
        sop2(30, 6, 6, int(12)),
        sop2(16, M0, 5, 6),
        0xBF90_0009,
        vop2(0x16, 10, int(2), v(5)),
        vop2(0x1B, 11, int(3), v(5)),
        vop2(0x1B, 12, int(1), v(11)),
        vop2(0x16, 13, int(1), v(11)),
        vop1(0x06, 10, v(10)),
        vop1(0x06, 12, v(12)),
        vop1(0x06, 13, v(13)),
        vop2(0x08, 10, LITERAL, v(10)),
        1.25f32.to_bits(),
        vop2(0x08, 12, LITERAL, v(12)),
        0.75f32.to_bits(),
        vop2(0x03, 10, v(10), v(12)),
        vop2(0x03, 14, MINUS_ONE_F32, v(10)),
        vop2(0x08, 13, 244, v(13)),
        vop2(0x03, 15, MINUS_ONE_F32, v(13)),
        vop1(0x01, 16, int(0)),
        vop1(0x01, 17, ONE_F32),
        (0x3E << 26) | (1 << 11) | (12 << 4) | 0xF,
        0x1110_0F0E,
        vop2(0x1B, 20, LITERAL, v(0)),
        0xFFFF,
        vop2(0x16, 21, int(16), v(0)),
        vop2(0x1B, 22, LITERAL, v(1)),
        0xFFFF,
        vop2(0x1A, 21, int(10), v(21)),
        vop2(0x1A, 22, int(20), v(22)),
        vop2(0x1C, 20, v(20), v(21)),
        vop2(0x1C, 20, v(20), v(22)),
        (0x3E << 26) | (1 << 11) | (20 << 4) | 0x1,
        20,
        asm::S_ENDPGM,
    ];
    g.program(&w)
}

/// An indexed strip of two quads separated by a restart index, drawn by a
/// geometry subgroup: the primitives that reach across the restart are
/// dropped and the second strip starts its own winding, so the gap between
/// the quads stays empty.
#[test]
#[ignore]
fn ngg_mesh_strip_restart_splits_the_strip() {
    let mut g = Guest::new(32 << 20);
    let vs = passthrough_strip_program(&mut g);
    let ps = solid_program(&mut g);
    let rest = TestCompiler::new().with(ps, TestProgram::SolidPixel);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(SpirvVertices { spirv: crate::shader::SpirvCompiler::for_device(dev, g.mem), vertices: vec![vs], rest }), None, None);
    if !rig.dev.caps.mesh_shader || rig.dev.mesh.is_none() {
        eprintln!("ngg_mesh: no mesh shaders on this device, skipped");
        return;
    }
    let green = color_words([0.0, 1.0, 0.0, 1.0]);
    let idx = [0u16, 1, 2, 3, 0xFFFF, 4, 5, 6, 7];
    let indices = g.alloc(&idx.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>());
    for wave in [32u32, 64] {
        let target = rgba8_target(g.zeroed(64 * 16 * 4), 64, 16);
        let count = idx.len() as u32;
        let packet = DrawPacket { kind: DrawKind::Index, count, instance_count: 1, index_address: indices, index_type: IndexType::U16 as u32, ..DrawPacket::default() };
        let index = IndexBuffer { address: indices, kind: IndexType::U16, count };
        let mut d = draw_into(target, stage(vs, &[]), stage(ps, &green), primitive::TRI_STRIP, packet, Some(index));
        d.primitive_restart = true;
        d.shaders.stages = STAGES_GS_EN | if wave == 32 { STAGES_GS_W32_EN } else { 0 };
        d.shaders.geometry = crate::render::GeometryRegs {
            ge_cntl: 14 | (16 << 9),
            max_output_per_subgroup: 48,
            max_vert_out: 3,
            esgs_ring_item_size: 1,
            out_prim_type: 2,
            user_data_address: 0,
        };
        assert_eq!(rig.draw(&d), DrawOutcome::Recorded, "wave{wave}");
        let p = rig.pixels(&target);
        for (x, y) in [(4, 3), (12, 8), (20, 12), (44, 3), (52, 8), (60, 12)] {
            assert_eq!(px(&p, 64, x, y), [0, 255, 0, 255], "wave{wave}: quad pixel ({x}, {y})");
        }
        for (x, y) in [(28, 2), (32, 8), (36, 14)] {
            assert_eq!(px(&p, 64, x, y), [0, 0, 0, 0], "wave{wave}: gap pixel ({x}, {y})");
        }
    }
    assert!(rig.provider.stats().mesh_restart_tables >= 2);
}

#[test]
fn restart_table_marks_strip_starts() {
    use crate::shader::mesh::restart_table;
    let bytes = |v: &[u16]| v.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>();
    assert_eq!(restart_table(&bytes(&[0, 1, 2, 3]), 2, 0xFFFF_FFFF), None, "no restart, no table");
    assert_eq!(restart_table(&bytes(&[0, 1, 0xFFFF, 2, 3, 4]), 2, 0xFFFF_FFFF), Some(vec![6, 0xFFFF, 0, 0, 3, 3, 3, 3]));
    let words: Vec<u8> = [5u32, u32::MAX, 6].iter().flat_map(|i| i.to_le_bytes()).collect();
    assert_eq!(restart_table(&words, 4, u32::MAX), Some(vec![3, u32::MAX, 0, 2, 2]));
}

/// Alpha-to-coverage (DB_ALPHA_TO_MASK 0xAA01): a pixel whose MRT0 alpha
/// is below one half writes nothing, one at one half or above is written.
#[test]
#[ignore]
fn alpha_to_coverage_drops_pixels_below_half_alpha() {
    let mut g = Guest::new(16 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    // v0..v3 = s0..s3; exp mrt0 v0..v3 done vm.
    let ps = g.program(&[
        asm::v_mov_from_sgpr(0, 0),
        asm::v_mov_from_sgpr(1, 1),
        asm::v_mov_from_sgpr(2, 2),
        asm::v_mov_from_sgpr(3, 3),
        (0x3E << 26) | (1 << 12) | (1 << 11) | 0xF,
        0x0302_0100,
        asm::S_ENDPGM,
    ]);
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0]));
    let rest = TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 });
    let inputs = vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)]);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(SpirvVertices { spirv: crate::shader::SpirvCompiler::for_device(dev, g.mem), vertices: vec![ps], rest }), Some(inputs), None);
    for (alpha, written) in [(0.25f32, false), (0.5, true), (0.75, true), (f32::NAN, false)] {
        let target = rgba8_target(g.zeroed(16 * 16 * 4), 16, 16);
        let mut d = draw_into(target, stage(vs, &[]), stage(ps, &color_words([0.0, 1.0, 0.0, alpha])), primitive::TRI_LIST, auto(3, 1), None);
        d.ps_input_ena = 2;
        d.ps_input_addr = 2;
        d.alpha_to_mask = 0xAA01;
        assert_eq!(rig.draw(&d), DrawOutcome::Recorded, "alpha {alpha}");
        let p = rig.pixels(&target);
        let got = px(&p, 16, 8, 8);
        if written {
            let a = alpha * 255.0;
            assert!(got[..3] == [0, 255, 0] && (got[3] as f32 - a).abs() <= 1.0, "alpha {alpha}: {got:?}");
        } else {
            assert_eq!(got, [0, 0, 0, 0], "alpha {alpha}");
        }
    }
}

fn sop1(op: u32, sdst: u32, ssrc0: u32) -> u32 {
    0xBE80_0000 | (sdst << 16) | (op << 8) | ssrc0
}

fn vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
    (0x3E << 25) | (op << 17) | ((vsrc1 & 0xFF) << 9) | src0
}

/// The full-screen triangle of every three vertex IDs (v5) on the merged
/// layout: v0 = x, v1 = y (corner from the bits of ID % 3, in v10), v2 = 0.
pub(super) fn full_screen_corner() -> Vec<u32> {
    vec![
        vop2(0x0B, 10, LITERAL, v(5)),
        0xAAAB,
        vop2(0x16, 10, int(17), v(10)),
        vop2(0x0B, 10, int(3), v(10)),
        vop2(0x26, 10, v(5), v(10)),
        vop2(0x1B, 0, int(1), v(10)),
        vop2(0x1A, 0, int(2), v(0)),
        vop2(0x1B, 1, int(2), v(10)),
        vop2(0x1A, 1, int(1), v(1)),
        vop1(0x06, 0, v(0)),
        vop1(0x06, 1, v(1)),
        vop2(0x03, 0, MINUS_ONE_F32, v(0)),
        vop2(0x03, 1, MINUS_ONE_F32, v(1)),
        vop1(0x01, 2, int(0)),
    ]
}

/// A merged program run per vertex rebuilds EXEC from the vertex count in
/// s3 and walks it lane by lane (a waterfall loop over v5), marking each
/// vertex it visits with W = 1; a vertex the walk misses keeps W = 0 and
/// its triangle vanishes. The count must be the subgroup's active
/// invocations whatever the draw's size.
#[test]
#[ignore]
fn ngg_vertex_waterfall_over_s3_visits_every_vertex() {
    let mut g = Guest::new(32 << 20);
    let mut w = vec![
        vop1(0x01, 9, int(0)),
        // exec = (1 << primitives) - 1, then (1 << vertices) - 1
        0xBEFE_04C1,
        0x906A_8803,
        0x81EA_6A80,
        0x90FE_6AC1,
        0x81EA_0380,
        0x90FE_6AC1,
        sop1(0x04, 10, 126),
    ];
    let top = w.len();
    w.push(sop1(0x14, 4, 10));
    w.extend(vop3(0x360, 6, v(5), 4));
    w.push(vopc(0xC2, 6, v(5)));
    w.push(sop1(0x24, 18, 106));
    w.extend([vop1(0x01, 9, LITERAL), 1.0f32.to_bits()]);
    w.push(sop2(0x15, 10, 10, 106));
    w.push(sop1(0x04, 126, 18));
    let back = (top as i32 - w.len() as i32 - 1) as u32 & 0xFFFF;
    w.push(0xBF85_0000 | back);
    w.push(0xBEFE_04C1);
    w.extend(full_screen_corner());
    w.extend([(0x3E << 26) | (1 << 11) | (12 << 4) | 0xF, 0x0902_0100, asm::S_ENDPGM]);
    let vs = g.program(&w);
    let ps = solid_program(&mut g);
    let rest = TestCompiler::new().with(ps, TestProgram::SolidPixel);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(SpirvVertices { spirv: crate::shader::SpirvCompiler::for_device(dev, g.mem), vertices: vec![vs], rest }), None, None);
    let green = color_words([0.0, 1.0, 0.0, 1.0]);
    for count in [3u32, 6, 33] {
        let target = rgba8_target(g.zeroed(32 * 32 * 4), 32, 32);
        let d = draw_into(target, stage(vs, &[]), stage(ps, &green), primitive::TRI_LIST, auto(count, 1), None);
        assert_eq!(rig.draw(&d), DrawOutcome::Recorded, "{count} vertices");
        let p = rig.pixels(&target);
        for (x, y) in [(2, 2), (16, 16), (30, 30)] {
            assert_eq!(px(&p, 32, x, y), [0, 255, 0, 255], "{count} vertices at ({x}, {y})");
        }
    }
}

/// On the vertex path each host subgroup is a one-wave subgroup: s3 sets
/// the wave count (bits 28..31) to 1 and s2 carries the subgroup's vertex
/// count (bits 12..20), equal to the wave's (s3 bits 0..7). The program
/// takes W from the wave count and moves the triangle off the target by
/// any difference between the two vertex counts.
#[test]
#[ignore]
fn ngg_vertex_one_wave_subgroup_inputs() {
    // The one-wave inputs are opt-in (`RPS5E_NGG_VERTEX_WAVE_INFO=1`).
    if !crate::shader::vertex_wave_info_enabled() {
        eprintln!("skipped: run with RPS5E_NGG_VERTEX_WAVE_INFO=1");
        return;
    }
    let mut g = Guest::new(32 << 20);
    let mut w = full_screen_corner();
    w.extend([
        // v3 = float(s3 >> 28)
        vop1(0x01, 6, 3),
        vop2(0x16, 3, int(28), v(6)),
        vop1(0x06, 3, v(3)),
        // v8 = s3 & 0xFF, v9 = (s2 >> 12) & 0x1FF
        vop2(0x1B, 8, LITERAL, v(6)),
        0xFF,
        vop1(0x01, 9, 2),
        vop2(0x16, 9, int(12), v(9)),
        vop2(0x1B, 9, LITERAL, v(9)),
        0x1FF,
        // x += 100 * (v9 - v8)
        vop2(0x26, 9, v(9), v(8)),
        vop1(0x05, 9, v(9)),
        vop2(0x08, 9, LITERAL, v(9)),
        100.0f32.to_bits(),
        vop2(0x03, 0, v(9), v(0)),
        (0x3E << 26) | (1 << 11) | (12 << 4) | 0xF,
        0x0302_0100,
        asm::S_ENDPGM,
    ]);
    let vs = g.program(&w);
    let ps = solid_program(&mut g);
    let rest = TestCompiler::new().with(ps, TestProgram::SolidPixel);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(SpirvVertices { spirv: crate::shader::SpirvCompiler::for_device(dev, g.mem), vertices: vec![vs], rest }), None, None);
    let green = color_words([0.0, 1.0, 0.0, 1.0]);
    let target = rgba8_target(g.zeroed(32 * 32 * 4), 32, 32);
    let d = draw_into(target, stage(vs, &[]), stage(ps, &green), primitive::TRI_LIST, auto(3, 1), None);
    assert_eq!(rig.draw(&d), DrawOutcome::Recorded);
    let p = rig.pixels(&target);
    for (x, y) in [(2, 2), (16, 16), (30, 30)] {
        assert_eq!(px(&p, 32, x, y), [0, 255, 0, 255], "at ({x}, {y})");
    }
}

/// A vertex whose position is all zeros clips its primitive (the hardware
/// rule; the host would draw it at w = 0, an edge to infinity): the
/// full-screen triangle of vertices 0..2 draws nothing when vertex 2 is
/// (0, 0, 0, 0), and is drawn whole when it is not.
#[test]
#[ignore]
fn all_zero_vertex_position_clips_the_primitive() {
    let mut g = Guest::new(32 << 20);
    let mut w = full_screen_corner();
    w.extend([vop1(0x01, 9, ONE_F32), vop1(0x01, 11, int(0))]);
    // vcc = v5 == zero_id; v0, v1, v9 = vcc ? 0 : value.
    let zero_id = w.len();
    w.push(vopc(0xC2, int(2), v(5)));
    w.extend([vop2(0x01, 0, v(0), 11), vop2(0x01, 1, v(1), 11), vop2(0x01, 9, v(9), 11)]);
    w.extend([(0x3E << 26) | (1 << 11) | (12 << 4) | 0xF, 0x0902_0100, asm::S_ENDPGM]);
    let zeroed = g.program(&w);
    // The same program comparing with vertex 7, which the draw never has.
    w[zero_id] = vopc(0xC2, int(7), v(5));
    let whole = g.program(&w);
    let ps = solid_program(&mut g);
    let rest = TestCompiler::new().with(ps, TestProgram::SolidPixel);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(SpirvVertices { spirv: crate::shader::SpirvCompiler::for_device(dev, g.mem), vertices: vec![zeroed, whole], rest }), None, None);
    let green = color_words([0.0, 1.0, 0.0, 1.0]);
    for (vs, drawn) in [(whole, true), (zeroed, false)] {
        let target = rgba8_target(g.zeroed(32 * 32 * 4), 32, 32);
        let d = draw_into(target, stage(vs, &[]), stage(ps, &green), primitive::TRI_LIST, auto(3, 1), None);
        assert_eq!(rig.draw(&d), DrawOutcome::Recorded);
        let p = rig.pixels(&target);
        let lit = (0..32 * 32).filter(|i| p[i * 4 + 1] != 0).count();
        if drawn {
            assert_eq!(lit, 32 * 32, "the full-screen triangle");
        } else {
            assert_eq!(lit, 0, "a triangle with an all-zero vertex draws nothing");
        }
    }
}
