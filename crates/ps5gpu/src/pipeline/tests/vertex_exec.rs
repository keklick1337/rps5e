// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Vertex programs (the reference driver's VertexClipCullDistances,
//! InactiveVertexPointSize, NggVertexWaveInfo, NggVertexWaveReduction and
//! VertexUserDataTail tests): clip and cull distances by
//! PA_CL_VS_OUT_CNTL from a vertex and a geometry-subgroup program, a
//! point size export without effect on triangles, the subgroup inputs the
//! NGG prologue builds EXEC from, wave reductions over them, and a vertex
//! program continuing in code its user data points at.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored vertex_exec`.

use super::gfx_exec::*;
use super::*;

/// v0..v3 = 1.0; exp mrt0 v0..v3 done.
const WHITE_PIXEL: [u32; 7] = [0x7e0002f2, 0x7e0202f2, 0x7e0402f2, 0x7e0602f2, 0xf800080f, 0x03020100, 0xbf810000];

/// Loads the position and the distance vector of record v5 from the V# in
/// s[8:11] and exports them as pos0 and pos1.
const CLIP_VERTEX: [u32; 10] = [0xe0382000, 0x80020005, 0xe0382010, 0x80020405, 0xbf8c3f70, 0xf80000cf, 0x03020100, 0xf80008df, 0x07060504, 0xbf810000];

/// The same as a geometry subgroup (the reference's mesh program): the V#
/// in s[8:11], one triangle a subgroup.
const CLIP_MESH: [u32; 106] = [
    0x8f6a9003, 0x94fe6ac1, 0xbf88000b, 0xd7650006, 0x000100c1, 0xd7660006, 0x00020cc1, 0x93ebff03, 0x00040018, 0xd7460006, 0x04190c6b, 0x340c0c82, 0xd8340000, 0x00000506,
    0xbf8cc07f, 0xbefe04c1, 0xbf8a0000, 0x938dff02, 0x00090016, 0x9382ff03, 0x00040018, 0x938cff03, 0x00080008, 0xd7650009, 0x000100c1, 0xd7660009, 0x000212c1, 0xd746000a,
    0x04250c02, 0x7da8120c, 0xbf88002e, 0x361600ff, 0x0000ffff, 0x2c180090, 0x361a02ff, 0x0000ffff, 0xd8d80000, 0x0b00000b, 0xd8d80000, 0x0c00000c, 0xd8d80000, 0x0d00000d,
    0xbf8cc07f, 0xe0382000, 0x8002100b, 0xe0382010, 0x8002140b, 0xe0382000, 0x8002180c, 0xe0382010, 0x80021c0c, 0xe0382000, 0x8002200d, 0xe0382010, 0x8002240d, 0x161c1483,
    0x161e14ff, 0x00000060, 0xbf8c3f70, 0xdb7c0400, 0x0000100f, 0xdb7c0410, 0x0000140f, 0xdb7c0420, 0x0000180f, 0xdb7c0430, 0x00001c0f, 0xdb7c0440, 0x0000200f, 0xdb7c0450,
    0x0000240f, 0x4a501c81, 0x4a521c82, 0x3450508a, 0x34525294, 0xd772002a, 0x04a6510e, 0xbf8cc07f, 0xbefe04c1, 0xbf8a0000, 0x930e830d, 0xbf078002, 0xbf850003, 0x8f0f8c0d,
    0x887c0f0e, 0xbf900009, 0x7da8140d, 0xbf880002, 0xf8000941, 0x0000002a, 0xbefe04c1, 0x7da8140e, 0xbf88000c, 0x34561485, 0xdbfc0400, 0x2c00002b, 0xdbfc0410, 0x3000002b,
    0xbf8cc07f, 0xf80000cf, 0x2f2e2d2c, 0xf80008df, 0x33323130, 0xf800020f, 0x33323130, 0xbf810000,
];

#[derive(Clone, Copy, PartialEq, Debug)]
enum Visible {
    All,
    None,
    X,
    Y,
    XAndY,
}

#[test]
#[ignore]
fn vertex_exec_clip_cull_distances() {
    const W: u32 = 192;
    const H: u32 = 128;
    const DISTANCE_VECTOR: u32 = 0x0040_0000;
    const X: [f32; 3] = [-3.0, 1.0, 1.0];
    const Y: [f32; 3] = [-1.0, -1.0, 3.0];
    const NEGATIVE: [f32; 3] = [-1.0; 3];
    const MIXED: [f32; 3] = [-1.0, 1.0, -1.0];
    let rows: [(&str, u32, Visible, [f32; 3], [f32; 3], [f32; 3]); 7] = [
        ("no enable bits", DISTANCE_VECTOR, Visible::All, NEGATIVE, NEGATIVE, NEGATIVE),
        ("clip plane 0 from component 0", DISTANCE_VECTOR | 0x1, Visible::X, X, NEGATIVE, NEGATIVE),
        ("clip plane 1 alone, from component 1", DISTANCE_VECTOR | 0x2, Visible::Y, NEGATIVE, Y, NEGATIVE),
        ("clip planes 0x5, components 0 and 2", DISTANCE_VECTOR | 0x5, Visible::XAndY, X, NEGATIVE, Y),
        ("cull plane 0, every vertex negative", DISTANCE_VECTOR | 0x100, Visible::None, NEGATIVE, NEGATIVE, NEGATIVE),
        ("cull plane 0, one vertex not negative", DISTANCE_VECTOR | 0x100, Visible::All, MIXED, NEGATIVE, NEGATIVE),
        ("cull plane 0 from component 2, sparse 0x4", DISTANCE_VECTOR | 0x400, Visible::None, MIXED, NEGATIVE, NEGATIVE),
    ];
    let mut b = Bench::new();
    let vertex = b.words(&CLIP_VERTEX);
    let mesh = b.words(&CLIP_MESH);
    let ps = b.words(&WHITE_PIXEL);
    let meshes = b.rig.dev.caps.mesh_shader && b.rig.dev.mesh.is_some();
    if !meshes {
        eprintln!("mesh case skipped, the device has no VK_EXT_mesh_shader");
    }
    let mut cases = vec![(false, false), (false, true)];
    if meshes {
        cases.push((true, false));
    }
    for (is_mesh, wave32) in cases {
        for &(name, control, visible, c0, c1, c2) in &rows {
            let what = format!("{} clip and cull distances, {name}, wave{}", if is_mesh { "mesh" } else { "vertex" }, if wave32 { 32 } else { 64 });
            let mut records = Vec::new();
            for i in 0..3 {
                records.extend_from_slice(&[X[i], Y[i], 0.5, 1.0, c0[i], c1[i], c2[i], 0.0]);
            }
            let at = b.g.alloc(&floats(&records));
            let sharp = v_sharp(at, 32, 3);
            // User data from s8 in both layouts.
            let user = sharp.to_vec();
            let target = b.target(W, H, 0);
            b.draw(target, if is_mesh { mesh } else { vertex }, &user, ps, &[0; 4], 3, wave32, |d| {
                d.raster.vs_out_cntl = control;
                if is_mesh {
                    d.shaders.stages |= 1 << 5;
                    // Four primitives of twelve vertices a subgroup, 1024 LDS
                    // dwords (SPI_SHADER_PGM_RSRC2_GS.LDS_SIZE in 128-dword
                    // units).
                    d.shaders.geometry = crate::render::GeometryRegs { ge_cntl: 4 | (12 << 9), max_output_per_subgroup: 12, max_vert_out: 3, esgs_ring_item_size: 4, out_prim_type: 2, user_data_address: 0 };
                    d.shaders.gs.rsrc2 |= 8 << 19;
                }
            });
            let pixels = b.rig.pixels(&target);
            let written = pixels.chunks(4).filter(|p| p[0] == 255).count();
            for pixel in 0..W * H {
                let value = pixels[pixel as usize * 4];
                assert!(value == 0 || value == 255, "{what}: pixel {pixel} stored {value}");
                let x = -1.0 + 2.0 * ((pixel % W) as f64 + 0.5) / W as f64;
                let y = 1.0 - 2.0 * ((pixel / W) as f64 + 0.5) / H as f64;
                let expected = match visible {
                    Visible::All => true,
                    Visible::None => false,
                    Visible::X => x >= 0.0,
                    Visible::Y => y >= 0.0,
                    Visible::XAndY => x >= 0.0 && y >= 0.0,
                };
                assert_eq!(value == 255, expected, "{what}: pixel ({}, {}) ({written} pixels written)", pixel % W, pixel / W);
            }
        }
    }
}

/// Positions from the vertex index v5 (a triangle over the lower left
/// half of the target): v1 = 2i(2 - i) - 1, v2 = i(i - 1) - 1, v3 = 0,
/// v4 = 1.0.
const TAIL_PREFIX: [u32; 13] = [0x7e000d05, 0x080200f4, 0x10020300, 0x100202f4, 0x060202f3, 0x060400f3, 0x10040500, 0x060404f3, 0x7e060280, 0x7e0802f2, 0xbe8c03ff, 0x3f800000, 0xbf060c0c];

/// v_mov_b32 v0, s0; v_mov_b32 v1, 0; exp mrt0 (v0, v1, v1, v0) done vm.
const RED_PIXEL: [u32; 5] = [0x7e000200, 0x7e020280, 0xf800180f, 0x00010100, 0xbf810000];

#[test]
#[ignore]
fn vertex_exec_inactive_point_size() {
    // The prefix's positions, v5 = 1024.0; exp pos1 (point size) v5;
    // exp pos0 v1..v4 done.
    let mut code = TAIL_PREFIX[..10].to_vec();
    code.extend([0x7e0a02ff, 0x44800000, 0xf80000d1, 0x00000005, 0xf80008cf, 0x04030201, 0xbf810000]);
    let mut b = Bench::new();
    let vs = b.words(&code);
    let ps = b.words(&RED_PIXEL);
    let mut images = Vec::new();
    // PA_CL_VS_OUT_CNTL without and with USE_VTX_POINT_SIZE.
    for control in [0x0120_0000u32, 0x0121_0000] {
        let target = b.target(64, 4, KEPT);
        b.draw(target, vs, &[0], ps, &[1.0f32.to_bits()], 3, false, |d| d.raster.vs_out_cntl = control);
        images.push(b.rig.pixels(&target));
    }
    let kept = images[0].iter().filter(|&&p| p == KEPT).count();
    assert!(kept != 0 && kept != images[0].len(), "the triangle without USE_VTX_POINT_SIZE covered {} of {} bytes, expected an edge through the target", images[0].len() - kept, images[0].len());
    assert!(images[0] == images[1], "USE_VTX_POINT_SIZE on a triangle draw changed the pixels");
}

#[test]
#[ignore]
fn vertex_exec_ngg_wave_info() {
    const W: u32 = 64;
    const H: u32 = 32;
    // EXEC = the vertex count of s3 (merged wave info), then the load and
    // the export of VERTEX.
    let mut merged = vec![0x8f008803, 0x94fe00c1, 0x99030380, 0x94fe03c1];
    merged.extend(VERTEX);
    // EXEC = the vertex count of s2 (subgroup info) bits 12..20.
    let mut group = vec![0x9380ff02, 0x0009000c, 0x8f009000, 0x94fe00c1];
    group.extend(VERTEX);
    let mut b = Bench::new();
    let cover = b.positions(&COVER);
    let ps = b.words(&WHITE);
    let mut programs = vec![("merged wave info", b.words(&merged))];
    // s2 carries the subgroup counts only with the one-wave inputs
    // (`RPS5E_NGG_VERTEX_WAVE_INFO=1`, off by default: GPU_PLAN §43).
    if crate::shader::vertex_wave_info_enabled() {
        programs.push(("group info", b.words(&group)));
    } else {
        eprintln!("group info case skipped: RPS5E_NGG_VERTEX_WAVE_INFO is off");
    }
    for wave32 in [false, true] {
        for &(source, vs) in &programs {
            let target = b.target(W, H, KEPT);
            b.draw(target, vs, &cover, ps, &[0; 8], 3, wave32, |_| {});
            let pixels = b.rig.pixels(&target);
            let wrong = pixels.iter().position(|&p| p != 255);
            assert!(wrong.is_none(), "NGG vertex prologue wave{} ({source}): byte {} is {}, expected 255 (EXEC built from the {source} must cover every vertex)", if wave32 { 32 } else { 64 }, wrong.unwrap(), pixels[wrong.unwrap()]);
        }
    }
}

/// A wave64 vertex program: EXEC = lanes below the vertex count of s3;
/// each odd vertex sets its lane's bit, OR-reduced over the wave with
/// row_shr DPP, v_permlanex16_b32 and the readlanes of 31 and 63; the
/// position is scaled by 1.0 when the reduction equals the ballot of the
/// odd vertices, by 0 (a collapsed triangle) otherwise.
const REDUCTION_VERTEX: [u32; 44] = [
    0xd7650015, 0x000100c1, 0xd7660015, 0x00022ac1, 0x7e0e0280, 0x7da206f9, 0x81060015, 0x360e0a81, 0xbefe04c1, 0xd4c50002, 0x00020e80, 0xd763000f, 0x00022a81, 0xd5010000,
    0x000a1e80, 0x380000fa, 0xff091100, 0x380000fa, 0xff091200, 0x380000fa, 0xff091400, 0x380000fa, 0xff091800, 0xd778000f, 0x03058300, 0x38001f00, 0x88040302, 0xd7600002,
    0x00013f00, 0xd760006b, 0x00017f00, 0x88026b02, 0xbf060402, 0x850580f2, 0xe0382000, 0x80020005, 0xbf8c3f70, 0x10000005, 0x10020205, 0x10040405, 0x10060605, 0xf80008cf,
    0x03020100, 0xbf810000,
];

#[test]
#[ignore]
fn vertex_exec_ngg_wave_reduction() {
    const W: u32 = 64;
    const H: u32 = 32;
    const COLUMNS: u32 = 7;
    const ROWS: u32 = 5;
    let mut b = Bench::new();
    if b.rig.dev.caps.subgroup_size != 32 {
        eprintln!("skipped, s3 describes a 32-lane wave only on 32-lane subgroups, and the device has {} lanes", b.rig.dev.caps.subgroup_size);
        return;
    }
    let mut v = Vec::new();
    for cell in 0..COLUMNS * ROWS {
        let left = -1.0 + 2.0 * (cell % COLUMNS) as f32 / COLUMNS as f32;
        let right = -1.0 + 2.0 * (cell % COLUMNS + 1) as f32 / COLUMNS as f32;
        let bottom = -1.0 + 2.0 * (cell / COLUMNS) as f32 / ROWS as f32;
        let top = -1.0 + 2.0 * (cell / COLUMNS + 1) as f32 / ROWS as f32;
        for [x, y] in [[left, bottom], [right, bottom], [right, top], [left, bottom], [right, top], [left, top]] {
            v.extend_from_slice(&[x, y, 0.5, 1.0]);
        }
    }
    let user = b.positions(&v);
    let vs = b.words(&REDUCTION_VERTEX);
    let ps = b.words(&WHITE);
    for _ in 0..3 {
        let target = b.target(W, H, 0);
        b.draw(target, vs, &user, ps, &[0; 8], (v.len() / 4) as u32, false, |_| {});
        let pixels = b.rig.pixels(&target);
        for pixel in 0..W * H {
            assert_eq!(pixels[pixel as usize * 4], 255, "NGG vertex wave reduction: pixel {pixel} (a vertex whose wave OR of its lanes' bits, read from lanes 31 and 63, differs from its wave's ballot collapses its triangle)");
        }
    }
}

/// s_setpc_b64 s[8:9].
const S_SETPC_S8: u32 = 0xbe80_2008;
const S_CODE_END: u32 = 0xbf9f_0000;

struct Tail {
    b: Bench,
    ps: u64,
    color_ps: u64,
}

impl Tail {
    /// Pixels of a draw of vertex program `vs` with s[8:9] = `next`
    /// (PA_CL_VS_OUT_CNTL `control`; the pixel colour is the vertex's
    /// param0 x when `color`).
    fn draw(&mut self, vs: u64, next: u64, control: u32, color: bool) -> Vec<u8> {
        let target = self.b.target(64, 4, KEPT);
        let ps = if color { self.color_ps } else { self.ps };
        self.b.draw(target, vs, &[next as u32, (next >> 32) as u32], ps, &[1.0f32.to_bits()], 3, false, |d| {
            d.raster.vs_out_cntl = control;
            if color {
                d.ps_input_ena = PERSP_CENTER;
                d.ps_input_addr = PERSP_CENTER;
                d.ps_in_control = 1;
            }
        });
        self.b.rig.pixels(&target)
    }

    /// The linked body (`prefix`, s_setpc_b64 s[8:9]) draws what `prefix`
    /// joined with the continuation's first `words` words draws.
    fn check(&mut self, prefix: &[u32], next: u64, words: usize, control: u32, color: bool) -> Vec<u8> {
        let mut body = prefix.to_vec();
        body.push(S_SETPC_S8);
        body.resize(64, S_CODE_END);
        let mut reference = prefix.to_vec();
        let mut continuation = vec![0u8; words * 4];
        assert!(self.b.g.mem.read_backing(next, &mut continuation));
        reference.extend(continuation.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())));
        reference.resize(64, S_CODE_END);
        let (body, reference) = (self.b.words(&body), self.b.words(&reference));
        let expected = self.draw(reference, next, control, color);
        for _ in 0..2 {
            assert!(self.draw(body, next, control, color) == expected, "linked vertex tail pixels differ from the monolithic program");
        }
        expected
    }
}

fn write_words(addr: u64, words: &[u32]) {
    let bytes = asm::bytes(words);
    // SAFETY: a chunk of the test's guest view, read by the GPU only
    // through uploads made after this write.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), addr as *mut u8, bytes.len()) };
}

fn branch_colors(written: &[u8], skipped: &[u8], state: &str) {
    let mut covered = 0;
    for (w, s) in written.chunks(4).zip(skipped.chunks(4)) {
        let untouched = w[0] == KEPT;
        assert!(w.iter().all(|&c| c == if untouched { KEPT } else { 0xff }) && s.iter().all(|&c| c == if untouched { KEPT } else { 0 }), "{state} did not select the expected vertex color");
        covered += !untouched as usize;
    }
    assert!(covered != 0 && covered != written.len() / 4, "{state} reference triangle did not cross the render target");
}

#[test]
#[ignore]
fn vertex_exec_user_data_tail() {
    let mut b = Bench::new();
    let padded = |w: &[u32]| {
        let mut w = w.to_vec();
        w.resize(64, S_CODE_END);
        w
    };
    // s_cbranch_scc0 +3; v_mov_b32 v4, s12; exp pos0 v1..v4 done.
    const CONTINUATION: [u32; 5] = [0xbf840003, 0x7e08020c, 0xf80008cf, 0x04030201, 0xbf810000];
    // The same with v_mov_b32 v4, 2.0.
    const OTHER: [u32; 5] = [0xbf840003, 0x7e0802f4, 0xf80008cf, 0x04030201, 0xbf810000];
    let continuation = b.words(&padded(&CONTINUATION));
    let other = b.words(&padded(&OTHER));
    // s_cbranch_scc0 +5; v_mov_b32 v4, s12; exp pos0 v1..v4; exp pos1
    // (clip and cull distances) v1 x4 done.
    let clip = b.words(&padded(&[0xbf840005, 0x7e08020c, 0xf80000cf, 0x04030201, 0xf80008df, 0x01010101, 0xbf810000]));
    // s_cbranch_scc0 +1; v_mov_b32 v6, 1.0; exp param0 v6 x4; exp pos0.
    let scc = b.words(&padded(&[0xbf840001, 0x7e0c02f2, 0xf800020f, 0x06060606, 0xf80008cf, 0x04030201, 0xbf810000]));
    // v_mov_b32 v6, 1.0; s_mov_b64 exec, s[14:15]; exp param0 v6 x4; exp
    // pos0.
    let exec = b.words(&padded(&[0x7e0c02f2, 0xbefe040e, 0xf800020f, 0x06060606, 0xf80008cf, 0x04030201, 0xbf810000]));
    let ps = b.words(&RED_PIXEL);
    // v_interp_p1_f32 v2, v0, attr0.x; v_interp_p2_f32 v2, v1, attr0.x;
    // exp mrt0 v2 x4 done vm.
    let color_ps = b.words(&[0xc8080000, 0xc8090001, 0xf800180f, 0x02020202, 0xbf810000]);
    let mut t = Tail { b, ps, color_ps };
    let mut prefix = TAIL_PREFIX.to_vec();
    let original = t.check(&prefix, continuation, 5, 0, false);
    let kept = original.iter().filter(|&&p| p == KEPT).count();
    assert!(kept != 0 && kept != original.len(), "reference triangle did not cross the render target");
    let changed = t.check(&prefix, other, 5, 0, false);
    assert!(changed != original, "changing the continuation did not change the reference pixels");
    write_words(continuation + 4, &OTHER[1..2]);
    crate::memory::note_sync_point();
    assert!(t.check(&prefix, continuation, 5, 0, false) == changed, "rewritten continuation kept the old pixels");
    write_words(continuation + 4, &CONTINUATION[1..2]);
    crate::memory::note_sync_point();
    assert!(t.check(&prefix, continuation, 5, 0, false) == original, "restored continuation did not restore the pixels");
    let clipped = t.check(&prefix, clip, 7, 0x0040_0001, false);
    assert!(clipped != original && clipped.iter().filter(|&&p| p == KEPT).count() > kept, "a clip distance exported by the continuation did not clip the triangle");
    let culled = t.check(&prefix, clip, 7, 0x0040_0100, false);
    assert!(culled == original, "a mixed-sign cull distance discarded the triangle");
    // SCC and EXEC carry across the jump (v6 = 0 before it).
    prefix.insert(prefix.len() - 1, 0x7e0c0280);
    let true_scc = t.check(&prefix, scc, 7, 0, true);
    *prefix.last_mut().unwrap() = 0xbf070c0c;
    let false_scc = t.check(&prefix, scc, 7, 0, true);
    branch_colors(&true_scc, &false_scc, "SCC");
    *prefix.last_mut().unwrap() = 0xbf060c0c;
    // s_mov_b64 s[14:15], exec.
    prefix.push(0xbe8e047e);
    let active = t.check(&prefix, exec, 7, 0, true);
    // s_mov_b64 exec, 0.
    prefix.push(0xbefe0480);
    let empty = t.check(&prefix, exec, 7, 0, true);
    branch_colors(&active, &empty, "EXEC");
}
