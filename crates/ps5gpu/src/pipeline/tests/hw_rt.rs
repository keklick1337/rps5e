// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! `RPS5E_HW_RT`: a box query at the root of a guest TLAS the host mirrors
//! keeps the children a host ray query found hit triangles under, and
//! drops the others; without the mirror (or before it is built) the guest
//! result stays.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored hw_rt`.

use super::*;
use ps5shader::testing::{asm as sa, raw_buffer_descriptor};

const S_WAITCNT_0: u32 = 0xBF8C_0000;

/// image_bvh_intersect_ray of the root (pointer 0x25) with the ray from
/// `origin` along (0.001, 0.001, 1), extent 10; `out[tid]` = child 0.
fn root_program(g: &mut Guest, origin: [f32; 3]) -> u64 {
    let mut code = vec![sa::vop1(sa::OP_V_MOV_B32, 4, sa::LITERAL), 0x25, sa::vop1(sa::OP_V_MOV_B32, 5, sa::LITERAL), 10.0f32.to_bits()];
    for (k, x) in origin.iter().enumerate() {
        code.extend([sa::vop1(sa::OP_V_MOV_B32, 6 + k as u32, sa::LITERAL), x.to_bits()]);
    }
    for (v, x) in [(9, 0.001f32), (10, 0.001), (11, 1.0), (12, 1000.0), (13, 1000.0), (14, 1.0)] {
        code.extend([sa::vop1(sa::OP_V_MOV_B32, v, sa::LITERAL), x.to_bits()]);
    }
    code.extend([0xF198_8F07, 0x0002_1804, 0x0807_0605, 0x0C0B_0A09, 0x0000_0E0D]);
    code.push(S_WAITCNT_0);
    code.push((0x1A << 25) | (1 << 17) | (128 + 2));
    code.extend(asm::buffer_store_dword(24, 1, 0));
    code.push(sa::S_ENDPGM);
    g.program(&code)
}

fn put(b: &mut [u8], at: usize, w: u32) {
    b[at..at + 4].copy_from_slice(&w.to_le_bytes());
}

fn put_box(b: &mut [u8], at: usize, lo: [f32; 3], hi: [f32; 3]) {
    for k in 0..3 {
        put(b, at + k * 4, lo[k].to_bits());
        put(b, at + 12 + k * 4, hi[k].to_bits());
    }
}

/// A BLAS with one triangle (0,0,1) (1,0,1) (0,1,1) under its root's
/// child 0 (box [0,1]x[0,1]x[0.5,1.5]), and a TLAS with one identity
/// instance of it.
fn scene(g: &mut Guest) -> u64 {
    use crate::rt::layout::tests::blas_bytes;
    let mut blas = blas_bytes([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]], 1);
    let (lo, hi) = ([0.0f32, 0.0, 0.5], [1.0f32, 1.0, 1.5]);
    put_box(&mut blas, 272, lo, hi);
    for c in 1..4 {
        put_box(&mut blas, 272 + c * 24, [f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    }
    let b = g.alloc(&blas);
    let mut t = vec![0u8; 1024];
    put(&mut t, 0, crate::rt::layout::MAGIC[0]);
    put(&mut t, 4, crate::rt::layout::MAGIC[1]);
    put(&mut t, 16, 1024);
    put(&mut t, 252, crate::rt::layout::VERSION);
    put(&mut t, 48, 512);
    put(&mut t, 112, 1);
    // Root: child 0 the instance (index 0), the rest empty.
    put(&mut t, 256, 0);
    for c in 1..4 {
        put(&mut t, 256 + c * 4, u32::MAX);
        put_box(&mut t, 272 + c * 24, [f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    }
    put_box(&mut t, 272, lo, hi);
    let rec = 512;
    for (r, row) in [[1.0f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]].iter().enumerate() {
        for k in 0..4 {
            put(&mut t, rec + r * 16 + k * 4, row[k].to_bits());
            put(&mut t, rec + 64 + r * 16 + k * 4, row[k].to_bits());
        }
    }
    let q = b >> 3;
    put(&mut t, rec + 48, q as u32);
    put(&mut t, rec + 52, (q >> 32) as u32);
    put(&mut t, rec + 56, 0x25);
    put(&mut t, rec + 60, 0xFF00_0000);
    g.alloc(&t)
}

fn query(cs: u64, tlas: u64, out: u64) -> ResolvedDispatch {
    let mut user = raw_buffer_descriptor(out, 256).to_vec();
    user.extend([0, 0, 0, 0]);
    user.extend([(tlas >> 8) as u32, ((tlas >> 40) & 0xFF) as u32, 0x100, 0]);
    let mut shader = stage(cs, &user);
    shader.rsrc1 = (0xC0 << 12) | (1 << 21) | 7;
    ResolvedDispatch { packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() }, shader, rsrc3: 0, threads: [64, 1, 1], start: [0; 3] }
}

fn child0(rig: &mut Rig, mem: &crate::memory::GuestGpuMemory, d: &ResolvedDispatch, out: u64) -> u32 {
    assert_eq!(rig.dispatch(d), DrawOutcome::Recorded);
    rig.sched.flush_and_wait();
    assert!(rig.buffers.download(&mut rig.sched, out, 256));
    let mut b = vec![0u8; 4];
    assert!(mem.read_backing(out, &mut b));
    u32::from_le_bytes(b.try_into().unwrap())
}

#[test]
#[ignore]
fn hw_rt_root_queries_drop_boxes_without_hit_triangles() {
    let mut g = Guest::new(16 << 20);
    let mut rig = Rig::with_compiler(
        &g,
        |dev| {
            let f = crate::shader::Features::of(dev);
            let emit = crate::shader::EmitOptions { hw_rt: dev.caps.ray_query, hw_rt_prune: true, ..f.emit };
            Arc::new(crate::shader::SpirvCompiler::new(crate::shader::Features { emit, ..f }, Some(g.mem)))
        },
        None,
        None,
    );
    if !rig.provider.enable_hw_rt() {
        eprintln!("no ray queries on this device: skipped");
        return;
    }
    let mem = g.mem;
    let tlas = scene(&mut g);
    let out = g.zeroed(256);
    // Through the box, beside the triangle (x + y > 1) / onto it.
    let miss = root_program(&mut g, [0.8, 0.8, 0.0]);
    let hit = root_program(&mut g, [0.2, 0.2, 0.0]);
    // Before the mirror exists the guest's box test decides.
    assert_eq!(child0(&mut rig, mem, &query(miss, tlas, out), out), 0, "the box is hit");
    // Report, probe, register, mirror the BLAS, build: a few dispatches.
    let mut dropped = false;
    for _ in 0..8 {
        if child0(&mut rig, mem, &query(miss, tlas, out), out) == u32::MAX {
            dropped = true;
            break;
        }
    }
    assert!(dropped, "the mirror never dropped the box without a hit triangle");
    assert_eq!(child0(&mut rig, mem, &query(hit, tlas, out), out), 0, "the box of a hit triangle was dropped");
    assert_eq!(child0(&mut rig, mem, &query(miss, tlas, out), out), u32::MAX);
}

// ----------------------------------------------------------------------
// Traversal loops on host ray queries (`ps5shader::opt::rt_loops`)
// ----------------------------------------------------------------------

/// Guest code with labels: branches name their target.
#[derive(Default)]
struct Code {
    words: Vec<u32>,
    labels: std::collections::HashMap<&'static str, usize>,
    fixups: Vec<(usize, &'static str)>,
}

impl Code {
    fn put(&mut self, w: impl IntoIterator<Item = u32>) {
        self.words.extend(w);
    }
    fn label(&mut self, name: &'static str) {
        self.labels.insert(name, self.words.len());
    }
    fn branch(&mut self, op: u32, to: &'static str) {
        self.fixups.push((self.words.len(), to));
        self.words.push(sa::sopp(op, 0));
    }
    fn finish(mut self) -> Vec<u32> {
        for (at, to) in std::mem::take(&mut self.fixups) {
            let delta = self.labels[to] as i32 - (at as i32 + 1);
            self.words[at] = sa::branch(self.words[at] >> 16 & 0x7F, delta);
        }
        self.words
    }
}

const S_ADD_I32: u32 = 0x02;
const S_AND_B32: u32 = 0x0E;
const S_AND_B64: u32 = 0x0F;
const S_ANDN2_B64: u32 = 0x15;
const S_CSELECT_B64: u32 = 0x0B;
const S_ADD_U32: u32 = 0x00;
const S_MOV_B64: u32 = 0x04;
const S_FF1_I32_B64: u32 = 0x14;
const S_AND_SAVEEXEC_B64: u32 = 0x24;
const S_CMP_EQ_U32: u32 = 0x06;
const S_CMP_GT_U32: u32 = 0x08;
const S_BITCMP1_B32: u32 = 0x0D;
const S_CMP_EQ_U64: u32 = 0x12;
const V_CMP_NE_U32: u32 = 0xC5;
const V_CMPX_GT_U32: u32 = 0xD4;
const V_READLANE_B32: u32 = 0x360;
const V_WRITELANE_B32: u32 = 0x361;
const V_CVT_F32_U32: u32 = 0x06;
const V_ADD_F32: u32 = 0x03;
const V_MUL_F32: u32 = 0x08;
const V_RCP_F32: u32 = 0x2A;
const V_LSHLREV_B32: u32 = 0x1A;
const V_LSHRREV_B32: u32 = 0x16;
const SMEM_BUFFER_LOAD_DWORDX2: u32 = 0x09;
const LIT: u32 = sa::LITERAL;
const fn inl(n: u32) -> u32 {
    128 + n
}
const NEG1: u32 = 193;
const VGPR: u32 = 256;

/// image_bvh_intersect_ray v[24:27], node v4, extent v5, origin v6-8,
/// direction v9-11, inverse v12-14, T# s[8:11].
const BVH_QUERY: [u32; 5] = [0xF198_8F07, 0x0002_1804, 0x0807_0605, 0x0C0B_0A09, 0x0000_0E0D];

/// A guest traversal in the shape of the console's: the ray of lane `l`
/// starts at ((l + 0.5) / 48, 0.3, 0) along +z (extent 10); the loop starts
/// at the root, box results are pushed on a node stack in lanes of v20
/// (pointer), v21/v22 (lane mask) at counter s21, each iteration pops one
/// entry; a leaf list's entries are tested with triangle queries and a
/// candidate is accepted unless its primitive index is odd and the entry
/// not opaque; an accepted lane retires (s[40:41]). Lanes 60..63 take no
/// part. Lane `l` writes its hit flag and t to `out[2l]`, `out[2l+1]`.
fn traversal_program(g: &mut Guest) -> u64 {
    g.program(&traversal_code(false, false))
}

/// `store`: each pop also writes the lane's t (memory written between
/// candidates). `facing`: a candidate is accepted only when its `t_num` is
/// negative (the console's facing test without a flipped instance: the
/// sign of `t_num` is the triangle's facing, compared with the instance's
/// flip bit).
fn traversal_code(store: bool, facing: bool) -> Vec<u32> {
    let mut c = Code::default();
    // Ray.
    c.put([sa::vop1(V_CVT_F32_U32, 6, VGPR)]);
    c.put([sa::vop2(V_ADD_F32, 6, sa::F_HALF, 6)]);
    c.put([sa::vop2(V_MUL_F32, 6, LIT, 6), (1.0f32 / 48.0).to_bits()]);
    c.put([sa::vop1(sa::OP_V_MOV_B32, 7, LIT), 0.3f32.to_bits()]);
    c.put([sa::vop1(sa::OP_V_MOV_B32, 8, inl(0))]);
    c.put([sa::vop1(sa::OP_V_MOV_B32, 5, LIT), 10.0f32.to_bits()]);
    c.put([sa::vop1(sa::OP_V_MOV_B32, 9, inl(0)), sa::vop1(sa::OP_V_MOV_B32, 10, inl(0)), sa::vop1(sa::OP_V_MOV_B32, 11, sa::F_ONE)]);
    c.put([sa::vop1(sa::OP_V_MOV_B32, 12, LIT), 1e30f32.to_bits(), sa::vop1(sa::OP_V_MOV_B32, 13, LIT), 1e30f32.to_bits(), sa::vop1(sa::OP_V_MOV_B32, 14, sa::F_ONE)]);
    c.put([sa::vop1(sa::OP_V_MOV_B32, 16, inl(0)), sa::vop1(sa::OP_V_MOV_B32, 17, LIT), 10000.0f32.to_bits()]);
    // Root, empty stack, lanes 0..59 active (the others keep their initial
    // results).
    c.put([sa::vopc(V_CMPX_GT_U32, inl(60), 0)]);
    c.put([sa::s_mov_b32(20, LIT), 0x25, sa::s_mov_b32(21, inl(0)), sa::sop1(S_MOV_B64, 40, sa::EXEC_LO)]);
    c.label("loop");
    c.branch(sa::SOPP_CBRANCH_EXECZ, "pop");
    c.put([sa::sop2(S_AND_B32, 24, 20, inl(7)), sa::sopc(S_CMP_GT_U32, 24, inl(3))]);
    c.branch(sa::SOPP_CBRANCH_SCC0, "leaf");
    // A box: push each hit child with the lanes that hit it.
    c.put([sa::vop1(sa::OP_V_MOV_B32, 4, 20)]);
    c.put(BVH_QUERY);
    c.put([S_WAITCNT_0]);
    for k in 0..4u32 {
        let skip: &'static str = ["skip0", "skip1", "skip2", "skip3"][k as usize];
        c.put([sa::vopc(V_CMP_NE_U32, NEG1, 24 + k)]);
        c.put([sa::sop2(S_AND_B64, sa::VCC_LO, sa::VCC_LO, sa::EXEC_LO)]);
        c.branch(sa::SOPP_CBRANCH_VCCZ, skip);
        c.put([sa::sop1(S_FF1_I32_B64, 25, sa::VCC_LO)]);
        c.put(sa::vop3(V_READLANE_B32, 26, VGPR + 24 + k, 25, 0));
        c.put(sa::vop3(V_WRITELANE_B32, 20, 26, 21, 0));
        c.put(sa::vop3(V_WRITELANE_B32, 21, sa::VCC_LO, 21, 0));
        c.put(sa::vop3(V_WRITELANE_B32, 22, sa::VCC_HI, 21, 0));
        c.put([sa::sop2(S_ADD_I32, 21, 21, inl(1))]);
        c.label(skip);
    }
    c.branch(sa::SOPP_BRANCH, "pop");
    // A leaf list: its entries (8 bytes from the pointer) until the last.
    c.label("leaf");
    c.put([sa::sop2(S_AND_B32, 27, 20, LIT), 0xFFFF_FFF8]);
    c.label("entry");
    c.put(sa::smem(SMEM_BUFFER_LOAD_DWORDX2, 28, 6, 0, 27));
    c.put([S_WAITCNT_0]);
    c.put([sa::vop1(sa::OP_V_MOV_B32, 4, 28)]);
    c.put(BVH_QUERY);
    c.put([S_WAITCNT_0]);
    // A candidate (t_num not +inf) the any-hit rule takes.
    c.put([sa::vopc(V_CMP_NE_U32, LIT, 24), 0x7F80_0000]);
    c.put([sa::sop2(S_AND_B64, sa::VCC_LO, sa::VCC_LO, sa::EXEC_LO)]);
    if facing {
        c.put([sa::sop1(S_MOV_B64, 38, sa::VCC_LO)]);
        c.put([sa::vop2(V_LSHRREV_B32, 31, inl(31), 24)]);
        c.put([sa::vopc(V_CMP_NE_U32, inl(0), 31)]);
        c.put([sa::sop2(S_AND_B64, sa::VCC_LO, sa::VCC_LO, 38)]);
    }
    c.put([sa::sop2(S_AND_B32, 30, 29, LIT), 0x4000_0001, sa::sopc(S_CMP_EQ_U32, 30, inl(1))]);
    c.put([sa::sop2(S_CSELECT_B64, 32, inl(0), NEG1)]);
    c.put([sa::sop2(S_AND_B64, sa::VCC_LO, sa::VCC_LO, 32)]);
    c.put([sa::sop1(S_AND_SAVEEXEC_B64, 34, sa::VCC_LO)]);
    c.put([sa::vop1(V_RCP_F32, 18, VGPR + 25)]);
    c.put([sa::vop2(V_MUL_F32, 17, VGPR + 24, 18)]);
    c.put([sa::vop1(sa::OP_V_MOV_B32, 16, inl(1))]);
    c.put([sa::sop1(S_MOV_B64, sa::EXEC_LO, 34)]);
    // Accepted lanes retire.
    c.put([sa::sop2(S_ANDN2_B64, 40, 40, sa::VCC_LO), sa::sop2(S_ANDN2_B64, sa::EXEC_LO, sa::EXEC_LO, sa::VCC_LO)]);
    c.put([sa::sop2(S_ADD_U32, 27, 27, inl(8)), sa::sopc(S_BITCMP1_B32, 29, inl(31))]);
    c.branch(sa::SOPP_CBRANCH_SCC1, "pop");
    c.put([sa::sopc(S_CMP_EQ_U64, sa::EXEC_LO, inl(0))]);
    c.branch(sa::SOPP_CBRANCH_SCC1, "pop");
    c.branch(sa::SOPP_BRANCH, "entry");
    // Pop the next entry with the lanes still active.
    c.label("pop");
    if store {
        c.put([sa::vop2(V_LSHLREV_B32, 30, inl(3), VGPR)]);
        c.put(asm::buffer_store_dword(17, 30, 0));
    }
    c.put([sa::sopc(S_CMP_EQ_U64, 40, inl(0))]);
    c.branch(sa::SOPP_CBRANCH_SCC1, "exit");
    c.put([sa::sopc(S_CMP_EQ_U32, 21, inl(0))]);
    c.branch(sa::SOPP_CBRANCH_SCC1, "exit");
    c.put([sa::sop2(S_ADD_I32, 21, 21, NEG1)]);
    c.put(sa::vop3(V_READLANE_B32, 20, VGPR + 20, 21, 0));
    c.put(sa::vop3(V_READLANE_B32, 36, VGPR + 21, 21, 0));
    c.put(sa::vop3(V_READLANE_B32, 37, VGPR + 22, 21, 0));
    c.put([sa::sop2(S_AND_B64, sa::EXEC_LO, 36, 40)]);
    c.branch(sa::SOPP_BRANCH, "loop");
    c.label("exit");
    c.put([sa::sop1(S_MOV_B64, sa::EXEC_LO, NEG1)]);
    c.put([sa::vop2(V_LSHLREV_B32, 30, inl(3), VGPR)]);
    c.put(asm::buffer_store_dword(16, 30, 0));
    c.put(sa::mubuf(sa::OP_BUFFER_STORE_DWORD, 17, 30, 0, 128, 4, false, true, false));
    c.put([sa::S_ENDPGM]);
    c.finish()
}

/// A BLAS of three triangles under one leaf list: primitive 0 (opaque) at
/// z = 1 over x in [-0.5, 0.5] (at y = 0.3), primitive 1 at z = 1 over
/// [0.5, 1.175] (rejected by the program's rule), primitive 2 at z = 2 over
/// [0.5, 0.918]. The root's child box ends at x = `box_x` (the guest's
/// traversal misses what lies beyond it; the host's mirror does not).
fn leaf_scene(g: &mut Guest, box_x: f32) -> u64 {
    let mut b = vec![0u8; 1024];
    put(&mut b, 0, crate::rt::layout::MAGIC[0]);
    put(&mut b, 4, crate::rt::layout::MAGIC[1]);
    put(&mut b, 16, 1024);
    put(&mut b, 252, crate::rt::layout::VERSION);
    let tris: [[[f32; 3]; 3]; 3] = [
        [[-1.0, -1.0, 1.0], [0.5, -1.0, 1.0], [0.5, 3.0, 1.0]],
        [[0.5, -1.0, 1.0], [1.5, -1.0, 1.0], [0.5, 3.0, 1.0]],
        [[0.5, -1.0, 2.0], [1.12, -1.0, 2.0], [0.5, 3.0, 2.0]],
    ];
    // Triangle nodes at 512 + 64 j (index 8 + j); leaf entries at 704
    // (index 11).
    put(&mut b, 124, 8);
    put(&mut b, 104, 3);
    put(&mut b, 88, 11);
    put(&mut b, 40, 704 + 3 * 8);
    for (j, t) in tris.iter().enumerate() {
        let node = 512 + 64 * j;
        for (s, v) in t.iter().chain(&[t[2], t[2]]).enumerate() {
            for k in 0..3 {
                put(&mut b, node + s * 12 + k * 4, v[k].to_bits());
            }
        }
        put(&mut b, 704 + j * 8, (node / 8) as u32);
        // Primitive 0 is opaque (bit 30): the host query commits it.
        put(&mut b, 708 + j * 8, j as u32 | if j == 2 { 1 << 31 } else { 0 } | if j == 0 { 1 << 30 } else { 0 });
    }
    put(&mut b, 256, 704);
    for c in 1..4 {
        put(&mut b, 256 + c * 4, u32::MAX);
        put_box(&mut b, 272 + c * 24, [f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    }
    put_box(&mut b, 272, [-1.0, -1.0, 0.5], [box_x, 3.0, 2.5]);
    g.alloc(&b)
}

fn traversal(cs: u64, bvh: u64, out: u64) -> ResolvedDispatch {
    let mut user = raw_buffer_descriptor(out, 512).to_vec();
    user.extend([0, 0, 0, 0]);
    user.extend([(bvh >> 8) as u32, ((bvh >> 40) & 0xFF) as u32, 0xFFFF, 0]);
    user.extend(raw_buffer_descriptor(bvh, 1024));
    let mut shader = stage(cs, &user);
    shader.rsrc1 = (0xC0 << 12) | (1 << 21) | 7;
    ResolvedDispatch { packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() }, shader, rsrc3: 0, threads: [64, 1, 1], start: [0; 3] }
}

/// (hit, t) of every lane.
fn lanes(rig: &mut Rig, mem: &crate::memory::GuestGpuMemory, d: &ResolvedDispatch, out: u64) -> Vec<(u32, f32)> {
    assert_eq!(rig.dispatch(d), DrawOutcome::Recorded);
    rig.sched.flush_and_wait();
    assert!(rig.buffers.download(&mut rig.sched, out, 512));
    let mut b = vec![0u8; 512];
    assert!(mem.read_backing(out, &mut b));
    (0..64).map(|l| (u32::from_le_bytes(b[l * 8..l * 8 + 4].try_into().unwrap()), f32::from_le_bytes(b[l * 8 + 4..l * 8 + 8].try_into().unwrap()))).collect()
}

/// What the program finds for lane `l` when every triangle is reachable
/// (`box_x` beyond the geometry) or only those within the root box, with
/// primitive 2 ending at `far_x` (at y = 0.3).
fn expected(l: usize, box_x: f32, far_x: f32) -> (u32, f32) {
    let x = (l as f32 + 0.5) / 48.0;
    if x > box_x || l >= 60 {
        (0, 10000.0)
    } else if x < 0.5 {
        (1, 1.0)
    } else if x < far_x {
        (1, 2.0)
    } else {
        (0, 10000.0)
    }
}

/// Moves primitive 2 to x in [1.2, 1.47] (at y = 0.3), where no other
/// triangle lies, with buffer stores (as the console's refit programs write
/// triangle nodes).
fn move_triangle(g: &mut Guest, bvh: u64, xs: [f32; 3]) -> ResolvedDispatch {
    let mut code = Vec::new();
    for (at, x) in [(640u32, xs[0]), (652, xs[1]), (664, xs[2])] {
        code.extend([sa::vop1(sa::OP_V_MOV_B32, 1, LIT), x.to_bits(), sa::vop1(sa::OP_V_MOV_B32, 2, LIT), at]);
        code.extend(asm::buffer_store_dword(1, 2, 0));
    }
    code.push(sa::S_ENDPGM);
    let cs = g.program(&code);
    let mut shader = stage(cs, &raw_buffer_descriptor(bvh, 1024));
    shader.rsrc1 = (0xC0 << 12) | (1 << 21) | 7;
    ResolvedDispatch { packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() }, shader, rsrc3: 0, threads: [64, 1, 1], start: [0; 3] }
}

/// After [`move_triangle`].
fn expected_moved(l: usize) -> (u32, f32) {
    let x = (l as f32 + 0.5) / 48.0;
    if l >= 60 {
        (0, 10000.0)
    } else if x < 0.5 {
        (1, 1.0)
    } else if (1.2..1.47).contains(&x) {
        (1, 2.0)
    } else {
        (0, 10000.0)
    }
}

#[test]
#[ignore]
fn hw_rt_traversal_loops_run_on_ray_queries() {
    let mut g = Guest::new(16 << 20);
    let mut rig = Rig::with_compiler(
        &g,
        |dev| {
            let f = crate::shader::Features::of(dev);
            let emit = crate::shader::EmitOptions { hw_rt: dev.caps.ray_query, hw_rt_prune: false, ..f.emit };
            Arc::new(crate::shader::SpirvCompiler::new(crate::shader::Features { emit, ..f }, Some(g.mem)))
        },
        None,
        None,
    );
    if !rig.provider.enable_hw_rt() {
        eprintln!("no ray queries on this device: skipped");
        return;
    }
    let mem = g.mem;
    // The root box stops at x = 0.7: the guest's traversal misses the
    // triangles beyond it, the host's ray queries do not.
    let bvh = leaf_scene(&mut g, 0.7);
    let out = g.zeroed(512);
    let cs = traversal_program(&mut g);
    let d = traversal(cs, bvh, out);
    // Before the host knows the BVH the guest's loop runs.
    let guest = lanes(&mut rig, mem, &d, out);
    for (l, r) in guest.iter().enumerate() {
        assert_eq!(*r, expected(l, 0.7, 0.918), "guest traversal, lane {l}");
    }
    // Report, probe, register, mirror, build: a few dispatches.
    let mut host = None;
    for _ in 0..12 {
        let r = lanes(&mut rig, mem, &d, out);
        if r != guest {
            host = Some(r);
            break;
        }
    }
    let host = host.expect("the traversal never ran on the host's ray queries");
    for (l, r) in host.iter().enumerate() {
        assert_eq!(*r, expected(l, f32::INFINITY, 0.918), "host traversal, lane {l}");
    }
    // It stays there.
    assert_eq!(lanes(&mut rig, mem, &d, out), host);
    // GPU work rewrites a triangle (once a frame, like the console's
    // refits): the next updates find the change and refit the host BLAS.
    // A write that changes nothing first (the BLAS's own first build counts
    // as a change: refit here), then the move.
    let touch = move_triangle(&mut g, bvh, [0.5, 1.12, 0.5]);
    assert_eq!(rig.dispatch(&touch), DrawOutcome::Recorded);
    assert_eq!(lanes(&mut rig, mem, &d, out), host);
    let store = move_triangle(&mut g, bvh, [1.2, 1.6, 1.2]);
    let mut moved = false;
    for _ in 0..4 {
        assert_eq!(rig.dispatch(&store), DrawOutcome::Recorded);
        let r = lanes(&mut rig, mem, &d, out);
        // The guest's t = t_num * rcp(t_denom): within an ulp or two.
        let near = |a: (u32, f32), b: (u32, f32)| a.0 == b.0 && (a.1 - b.1).abs() <= 1e-5 * b.1;
        if r.iter().enumerate().all(|(l, r)| near(*r, expected_moved(l))) {
            moved = true;
            break;
        }
    }
    assert!(moved, "the host BLAS never followed the moved triangle");
}

/// Stores `words` (byte offset, value) into the BVH with a dispatch (as the
/// console's refit programs write triangle nodes).
fn store_words(g: &mut Guest, bvh: u64, words: &[(u32, f32)]) -> ResolvedDispatch {
    let mut code = Vec::new();
    for &(at, x) in words {
        code.extend([sa::vop1(sa::OP_V_MOV_B32, 1, LIT), x.to_bits(), sa::vop1(sa::OP_V_MOV_B32, 2, LIT), at]);
        code.extend(asm::buffer_store_dword(1, 2, 0));
    }
    code.push(sa::S_ENDPGM);
    let cs = g.program(&code);
    let mut shader = stage(cs, &raw_buffer_descriptor(bvh, 1024));
    shader.rsrc1 = (0xC0 << 12) | (1 << 21) | 7;
    ResolvedDispatch { packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() }, shader, rsrc3: 0, threads: [64, 1, 1], start: [0; 3] }
}

/// A traversal that tests the facing of its candidates (the console's
/// shadow traversals keep the triangles whose `t_num` is negative: seen
/// clockwise from the ray origin) runs on host ray queries that cull the
/// same faces: an opaque triangle the host commits and a non-opaque one it
/// hands to the guest are the ones the guest keeps; turned around, the
/// opaque one is no hit.
#[test]
#[ignore]
fn hw_rt_traversal_loops_keep_the_faces_the_guest_keeps() {
    let mut g = Guest::new(16 << 20);
    let mut rig = Rig::with_compiler(
        &g,
        |dev| {
            let f = crate::shader::Features::of(dev);
            let emit = crate::shader::EmitOptions { hw_rt: dev.caps.ray_query, hw_rt_prune: false, ..f.emit };
            Arc::new(crate::shader::SpirvCompiler::new(crate::shader::Features { emit, ..f }, Some(g.mem)))
        },
        None,
        None,
    );
    if !rig.provider.enable_hw_rt() {
        eprintln!("no ray queries on this device: skipped");
        return;
    }
    let mem = g.mem;
    // Every triangle of the scene is seen clockwise from the rays' origins
    // (z = 0, along +z): the guest keeps them all.
    let bvh = leaf_scene(&mut g, 0.7);
    let out = g.zeroed(512);
    let cs = g.program(&traversal_code(false, true));
    let d = traversal(cs, bvh, out);
    let guest = lanes(&mut rig, mem, &d, out);
    for (l, r) in guest.iter().enumerate() {
        assert_eq!(*r, expected(l, 0.7, 0.918), "guest traversal, lane {l}");
    }
    let mut host = None;
    for _ in 0..12 {
        let r = lanes(&mut rig, mem, &d, out);
        if r != guest {
            host = Some(r);
            break;
        }
    }
    let host = host.expect("the traversal never ran on the host's ray queries (or culled the faces the guest keeps)");
    for (l, r) in host.iter().enumerate() {
        assert_eq!(*r, expected(l, f32::INFINITY, 0.918), "host traversal, lane {l}");
    }
    // Primitive 0 (opaque) turned around: v1 and v2 swap their y.
    let turn = store_words(&mut g, bvh, &[(528, 3.0), (540, -1.0)]);
    let want = |l: usize| {
        let x = (l as f32 + 0.5) / 48.0;
        if l < 60 && (0.5..0.918).contains(&x) {
            (1, 2.0)
        } else {
            (0, 10000.0)
        }
    };
    let near = |a: (u32, f32), b: (u32, f32)| a.0 == b.0 && (a.1 - b.1).abs() <= 1e-5 * b.1;
    let mut last = Vec::new();
    for _ in 0..4 {
        assert_eq!(rig.dispatch(&turn), DrawOutcome::Recorded);
        last = lanes(&mut rig, mem, &d, out);
        if last.iter().enumerate().all(|(l, r)| near(*r, want(l))) {
            return;
        }
    }
    panic!("a back-facing opaque triangle still hits: {last:?}");
}

const S_LSHL_B32: u32 = 0x1E;
const S_LSHL_B64: u32 = 0x1F;
const S_LSHR_B32: u32 = 0x20;
const S_LSHR_B64: u32 = 0x21;
const S_CMP_LG_U32: u32 = 0x07;
const S_MIN_U32: u32 = 0x07;
const SMEM_BUFFER_LOAD_DWORD: u32 = 0x08;
const SMEM_NO_SOFFSET: u32 = 125;

/// The node pushes of a box query's four children (results v24..v27) on
/// the stack v20 (pointer) v21/v22 (lanes) at counter s21.
fn push_children(c: &mut Code, skips: [&'static str; 4]) {
    for k in 0..4u32 {
        c.put([sa::vopc(V_CMP_NE_U32, NEG1, 24 + k)]);
        c.put([sa::sop2(S_AND_B64, sa::VCC_LO, sa::VCC_LO, sa::EXEC_LO)]);
        c.branch(sa::SOPP_CBRANCH_VCCZ, skips[k as usize]);
        c.put([sa::sop1(S_FF1_I32_B64, 25, sa::VCC_LO)]);
        c.put(sa::vop3(V_READLANE_B32, 26, VGPR + 24 + k, 25, 0));
        c.put(sa::vop3(V_WRITELANE_B32, 20, 26, 21, 0));
        c.put(sa::vop3(V_WRITELANE_B32, 21, sa::VCC_LO, 21, 0));
        c.put(sa::vop3(V_WRITELANE_B32, 22, sa::VCC_HI, 21, 0));
        c.put([sa::sop2(S_ADD_I32, 21, 21, inl(1))]);
        c.label(skips[k as usize]);
    }
    // The console's clamp of the stack depth (an overflow overwrites the
    // last entry).
    c.put([sa::sop2(S_MIN_U32, 21, 21, inl(60))]);
}

/// A two-level traversal in the shape of the console's: the top level's
/// non-box nodes are instances (record at `[TLAS+48] + 128 i`: the BLAS
/// address in 8-byte units at +48), whose BLAS root goes on the same stack
/// above a base (s60); the BLAS loop pops down to the base, its leaf lists
/// are tested as in [`traversal_program`]. Rays, any-hit rule, inactive
/// lanes and outputs as there. User data: out V#, -, TLAS T# (s8),
/// TLAS bytes V# (s12).
fn two_level_program(g: &mut Guest) -> u64 {
    g.program(&two_level_code())
}

fn two_level_code() -> Vec<u32> {
    let mut c = Code::default();
    c.put([sa::vop1(V_CVT_F32_U32, 6, VGPR)]);
    c.put([sa::vop2(V_ADD_F32, 6, sa::F_HALF, 6)]);
    c.put([sa::vop2(V_MUL_F32, 6, LIT, 6), (1.0f32 / 48.0).to_bits()]);
    c.put([sa::vop1(sa::OP_V_MOV_B32, 7, LIT), 0.3f32.to_bits()]);
    c.put([sa::vop1(sa::OP_V_MOV_B32, 8, inl(0))]);
    c.put([sa::vop1(sa::OP_V_MOV_B32, 5, LIT), 10.0f32.to_bits()]);
    c.put([sa::vop1(sa::OP_V_MOV_B32, 9, inl(0)), sa::vop1(sa::OP_V_MOV_B32, 10, inl(0)), sa::vop1(sa::OP_V_MOV_B32, 11, sa::F_ONE)]);
    c.put([sa::vop1(sa::OP_V_MOV_B32, 12, LIT), 1e30f32.to_bits(), sa::vop1(sa::OP_V_MOV_B32, 13, LIT), 1e30f32.to_bits(), sa::vop1(sa::OP_V_MOV_B32, 14, sa::F_ONE)]);
    c.put([sa::vop1(sa::OP_V_MOV_B32, 16, inl(0)), sa::vop1(sa::OP_V_MOV_B32, 17, LIT), 10000.0f32.to_bits()]);
    c.put([sa::vopc(V_CMPX_GT_U32, inl(60), 0)]);
    c.put([sa::s_mov_b32(20, LIT), 0x25, sa::s_mov_b32(21, inl(0)), sa::sop1(S_MOV_B64, 40, sa::EXEC_LO)]);
    // The instance array's offset.
    c.put(sa::smem(SMEM_BUFFER_LOAD_DWORD, 56, 6, 48, SMEM_NO_SOFFSET));
    c.put([S_WAITCNT_0]);
    c.label("loop");
    c.branch(sa::SOPP_CBRANCH_EXECZ, "pop");
    c.put([sa::sop2(S_AND_B32, 24, 20, inl(7)), sa::sopc(S_CMP_GT_U32, 24, inl(3))]);
    c.branch(sa::SOPP_CBRANCH_SCC0, "instance");
    c.put([sa::vop1(sa::OP_V_MOV_B32, 4, 20)]);
    c.put(BVH_QUERY);
    c.put([S_WAITCNT_0]);
    push_children(&mut c, ["t0", "t1", "t2", "t3"]);
    c.branch(sa::SOPP_BRANCH, "pop");
    // An instance: its BLAS's T# (s[28:31]) and bytes (s[32:35]); its root
    // on the stack above the base s60.
    c.label("instance");
    c.put([sa::sop2(S_LSHR_B32, 25, 20, inl(3)), sa::sop2(S_LSHL_B32, 25, 25, inl(7)), sa::sop2(S_ADD_U32, 25, 25, 56)]);
    c.put(sa::smem(SMEM_BUFFER_LOAD_DWORDX2, 26, 6, 48, 25));
    c.put([S_WAITCNT_0]);
    c.put([sa::sop2(S_LSHL_B64, 26, 26, inl(3))]);
    c.put([sa::sop2(S_LSHR_B64, 28, 26, inl(8)), sa::sop2(S_AND_B32, 29, 29, LIT), 0xFF]);
    c.put([sa::s_mov_b32(30, LIT), 0xFFFF, sa::s_mov_b32(31, inl(0))]);
    c.put([sa::s_mov_b32(32, 26), sa::sop2(S_AND_B32, 33, 27, LIT), 0xFFFF, sa::s_mov_b32(34, LIT), 1024, sa::s_mov_b32(35, LIT), (3 << 28) | 0x0001_6FAC]);
    c.put([sa::s_mov_b32(60, 21), sa::s_mov_b32(61, LIT), 0x25]);
    c.put(sa::vop3(V_WRITELANE_B32, 20, 61, 21, 0));
    c.put(sa::vop3(V_WRITELANE_B32, 21, sa::EXEC_LO, 21, 0));
    c.put(sa::vop3(V_WRITELANE_B32, 22, sa::EXEC_HI, 21, 0));
    c.put([sa::sop2(S_ADD_I32, 21, 21, inl(1)), sa::sop1(S_MOV_B64, 62, sa::EXEC_LO)]);
    c.label("blas");
    c.put([sa::sopc(S_CMP_LG_U32, 21, 60)]);
    c.branch(sa::SOPP_CBRANCH_SCC0, "blas_done");
    c.put([sa::sop2(S_ADD_I32, 21, 21, NEG1)]);
    c.put(sa::vop3(V_READLANE_B32, 42, VGPR + 20, 21, 0));
    c.put(sa::vop3(V_READLANE_B32, 36, VGPR + 21, 21, 0));
    c.put(sa::vop3(V_READLANE_B32, 37, VGPR + 22, 21, 0));
    c.put([sa::sop2(S_AND_B64, sa::EXEC_LO, 36, 40)]);
    c.branch(sa::SOPP_CBRANCH_EXECZ, "blas");
    c.put([sa::sop2(S_AND_B32, 24, 42, inl(7)), sa::sopc(S_CMP_GT_U32, 24, inl(3))]);
    c.branch(sa::SOPP_CBRANCH_SCC0, "leaf");
    c.put([sa::vop1(sa::OP_V_MOV_B32, 4, 42)]);
    let mut q = BVH_QUERY;
    q[1] = 0x0007_1804;
    c.put(q);
    c.put([S_WAITCNT_0]);
    push_children(&mut c, ["b0", "b1", "b2", "b3"]);
    c.branch(sa::SOPP_BRANCH, "blas");
    c.label("leaf");
    c.put([sa::sop2(S_AND_B32, 43, 42, LIT), 0xFFFF_FFF8]);
    c.label("entry");
    c.put(sa::smem(SMEM_BUFFER_LOAD_DWORDX2, 44, 16, 0, 43));
    c.put([S_WAITCNT_0]);
    c.put([sa::vop1(sa::OP_V_MOV_B32, 4, 44)]);
    c.put(q);
    c.put([S_WAITCNT_0]);
    c.put([sa::vopc(V_CMP_NE_U32, LIT, 24), 0x7F80_0000]);
    c.put([sa::sop2(S_AND_B64, sa::VCC_LO, sa::VCC_LO, sa::EXEC_LO)]);
    c.put([sa::sop2(S_AND_B32, 30 + 16, 45, LIT), 0x4000_0001, sa::sopc(S_CMP_EQ_U32, 46, inl(1))]);
    c.put([sa::sop2(S_CSELECT_B64, 48, inl(0), NEG1)]);
    c.put([sa::sop2(S_AND_B64, sa::VCC_LO, sa::VCC_LO, 48)]);
    c.put([sa::sop1(S_AND_SAVEEXEC_B64, 50, sa::VCC_LO)]);
    c.put([sa::vop1(V_RCP_F32, 18, VGPR + 25)]);
    c.put([sa::vop2(V_MUL_F32, 17, VGPR + 24, 18)]);
    c.put([sa::vop1(sa::OP_V_MOV_B32, 16, inl(1))]);
    c.put([sa::sop1(S_MOV_B64, sa::EXEC_LO, 50)]);
    c.put([sa::sop2(S_ANDN2_B64, 40, 40, sa::VCC_LO), sa::sop2(S_ANDN2_B64, sa::EXEC_LO, sa::EXEC_LO, sa::VCC_LO)]);
    c.put([sa::sop2(S_ADD_U32, 43, 43, inl(8)), sa::sopc(S_BITCMP1_B32, 45, inl(31))]);
    c.branch(sa::SOPP_CBRANCH_SCC1, "blas");
    c.put([sa::sopc(S_CMP_EQ_U64, sa::EXEC_LO, inl(0))]);
    c.branch(sa::SOPP_CBRANCH_SCC1, "blas");
    c.branch(sa::SOPP_BRANCH, "entry");
    c.label("blas_done");
    c.put([sa::sop1(S_MOV_B64, sa::EXEC_LO, 62)]);
    c.label("pop");
    c.put([sa::sopc(S_CMP_EQ_U64, 40, inl(0))]);
    c.branch(sa::SOPP_CBRANCH_SCC1, "exit");
    c.put([sa::sopc(S_CMP_EQ_U32, 21, inl(0))]);
    c.branch(sa::SOPP_CBRANCH_SCC1, "exit");
    c.put([sa::sop2(S_ADD_I32, 21, 21, NEG1)]);
    c.put(sa::vop3(V_READLANE_B32, 20, VGPR + 20, 21, 0));
    c.put(sa::vop3(V_READLANE_B32, 36, VGPR + 21, 21, 0));
    c.put(sa::vop3(V_READLANE_B32, 37, VGPR + 22, 21, 0));
    c.put([sa::sop2(S_AND_B64, sa::EXEC_LO, 36, 40)]);
    c.branch(sa::SOPP_BRANCH, "loop");
    c.label("exit");
    c.put([sa::sop1(S_MOV_B64, sa::EXEC_LO, NEG1)]);
    c.put([sa::vop2(V_LSHLREV_B32, 30, inl(3), VGPR)]);
    c.put(asm::buffer_store_dword(16, 30, 0));
    c.put(sa::mubuf(sa::OP_BUFFER_STORE_DWORD, 17, 30, 0, 128, 4, false, true, false));
    c.put([sa::S_ENDPGM]);
    c.finish()
}

/// A BLAS the host cannot mirror (its leaf list also names a procedural
/// node, type 7) holding one triangle at z = 3 over x in [1.2, 1.47] (at
/// y = 0.3).
fn proxy_scene(g: &mut Guest) -> u64 {
    let mut b = vec![0u8; 1024];
    put(&mut b, 0, crate::rt::layout::MAGIC[0]);
    put(&mut b, 4, crate::rt::layout::MAGIC[1]);
    put(&mut b, 16, 1024);
    put(&mut b, 252, crate::rt::layout::VERSION);
    put(&mut b, 124, 8);
    put(&mut b, 104, 1);
    put(&mut b, 88, 9);
    put(&mut b, 40, 576 + 2 * 8);
    let t = [[1.2f32, -1.0, 3.0], [1.6, -1.0, 3.0], [1.2, 3.0, 3.0]];
    for (s, v) in t.iter().chain(&[t[2], t[2]]).enumerate() {
        for k in 0..3 {
            put(&mut b, 512 + s * 12 + k * 4, v[k].to_bits());
        }
    }
    put(&mut b, 576, 512 / 8);
    put(&mut b, 580, 0);
    put(&mut b, 584, (512 / 8) | 7);
    put(&mut b, 588, 1 | 1 << 31);
    put(&mut b, 256, 576);
    for c in 1..4 {
        put(&mut b, 256 + c * 4, u32::MAX);
        put_box(&mut b, 272 + c * 24, [f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    }
    put_box(&mut b, 272, [1.1, -1.0, 2.5], [1.7, 3.0, 3.5]);
    g.alloc(&b)
}

/// A TLAS of two identity instances: of `a` (bounds [-1, 1.5] in x) and of
/// `b` (bounds [1.1, 1.7]).
fn two_instances(g: &mut Guest, a: u64, b: u64) -> u64 {
    let mut t = vec![0u8; 1024];
    put(&mut t, 0, crate::rt::layout::MAGIC[0]);
    put(&mut t, 4, crate::rt::layout::MAGIC[1]);
    put(&mut t, 16, 1024);
    put(&mut t, 252, crate::rt::layout::VERSION);
    put(&mut t, 48, 512);
    put(&mut t, 112, 2);
    put(&mut t, 256, 0);
    put(&mut t, 260, 8);
    put(&mut t, 264, u32::MAX);
    put(&mut t, 268, u32::MAX);
    put_box(&mut t, 272, [-1.0, -1.0, 0.5], [1.5, 3.0, 2.5]);
    put_box(&mut t, 296, [1.1, -1.0, 2.5], [1.7, 3.0, 3.5]);
    for c in 2..4 {
        put_box(&mut t, 272 + c * 24, [f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    }
    for (i, blas) in [a, b].into_iter().enumerate() {
        let rec = 512 + i * 128;
        for (r, row) in [[1.0f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]].iter().enumerate() {
            for k in 0..4 {
                put(&mut t, rec + r * 16 + k * 4, row[k].to_bits());
                put(&mut t, rec + 64 + r * 16 + k * 4, row[k].to_bits());
            }
        }
        let q = blas >> 3;
        put(&mut t, rec + 48, q as u32);
        put(&mut t, rec + 52, (q >> 32) as u32);
        put(&mut t, rec + 56, 0x25);
        put(&mut t, rec + 60, 0xFF00_0000 | i as u32);
    }
    g.alloc(&t)
}

#[test]
#[ignore]
fn hw_rt_two_level_traversals_walk_unmirrored_blases_themselves() {
    let mut g = Guest::new(16 << 20);
    let mut rig = Rig::with_compiler(
        &g,
        |dev| {
            let f = crate::shader::Features::of(dev);
            let emit = crate::shader::EmitOptions { hw_rt: dev.caps.ray_query, hw_rt_prune: false, ..f.emit };
            Arc::new(crate::shader::SpirvCompiler::new(crate::shader::Features { emit, ..f }, Some(g.mem)))
        },
        None,
        None,
    );
    if !rig.provider.enable_hw_rt() {
        eprintln!("no ray queries on this device: skipped");
        return;
    }
    let mem = g.mem;
    // BLAS a's root box stops at x = 0.7 (the guest misses what lies
    // beyond); b has no host mirror (the guest's stack walks it).
    let a = leaf_scene(&mut g, 0.7);
    let b = proxy_scene(&mut g);
    let tlas = two_instances(&mut g, a, b);
    let out = g.zeroed(512);
    let cs = two_level_program(&mut g);
    let mut user = raw_buffer_descriptor(out, 512).to_vec();
    user.extend([0, 0, 0, 0]);
    user.extend([(tlas >> 8) as u32, ((tlas >> 40) & 0xFF) as u32, 0xFFFF, 0]);
    user.extend(raw_buffer_descriptor(tlas, 1024));
    let mut shader = stage(cs, &user);
    shader.rsrc1 = (0xC0 << 12) | (1 << 21) | 7;
    let d = ResolvedDispatch { packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() }, shader, rsrc3: 0, threads: [64, 1, 1], start: [0; 3] };
    let expect = |l: usize, box_x: f32| -> (u32, f32) {
        let x = (l as f32 + 0.5) / 48.0;
        if l >= 60 {
            (0, 10000.0)
        } else if (1.2..1.47).contains(&x) {
            (1, 3.0)
        } else {
            expected(l, box_x, 0.918)
        }
    };
    let near = |a: (u32, f32), b: (u32, f32)| a.0 == b.0 && (a.1 - b.1).abs() <= 1e-5 * b.1;
    let guest = lanes(&mut rig, mem, &d, out);
    for (l, r) in guest.iter().enumerate() {
        assert!(near(*r, expect(l, 0.7)), "guest traversal, lane {l}: {r:?}");
    }
    let mut host = None;
    for _ in 0..12 {
        let r = lanes(&mut rig, mem, &d, out);
        if r != guest {
            host = Some(r);
            break;
        }
    }
    let host = host.expect("the traversal never ran on the host's ray queries");
    for (l, r) in host.iter().enumerate() {
        assert!(near(*r, expect(l, f32::INFINITY)), "host traversal, lane {l}: {r:?}");
    }
    assert_eq!(lanes(&mut rig, mem, &d, out), host);
}

// ----------------------------------------------------------------------
// Recognition (no device)
// ----------------------------------------------------------------------

/// The optimized SSA of a wave64 compute program.
fn ssa_of(words: &[u32]) -> ps5shader::ssa::Function {
    use ps5shader::translate::{compile, TranslateOptions};
    let bytes = asm::bytes(words);
    let p = ps5shader::decode_blob(&bytes, 0x6000_0000).expect("decodes");
    let mut opts = TranslateOptions::with_user_data(64, ps5shader::ssa::Stage::Compute, 16);
    opts.optimize = true;
    compile(&p, &opts, None).expect("translates").function
}

fn rt_ops(f: &ps5shader::ssa::Function) -> Vec<ps5shader::ssa::RtOp> {
    f.order.iter().flat_map(|&b| f.block(b).insts.iter()).filter_map(|&v| match f.op(v) {
        ps5shader::ssa::Op::RtQuery(r) => Some(r),
        _ => None,
    }).collect()
}

#[test]
fn rt_loops_recognise_one_and_two_level_traversals() {
    use ps5shader::ssa::{RtOp, RtGet};
    for (words, levels, cull) in [(traversal_code(false, false), 1u8, false), (traversal_code(false, true), 1u8, true), (two_level_code(), 2, false)] {
        let mut f = ssa_of(&words);
        let out = ps5shader::opt::rt_loops::replace_traversals(&mut f);
        assert_eq!(out.replaced, 1, "levels {levels}: {:?}", out.skipped);
        ps5shader::ssa::verify::verify(&f, true).expect("valid after the rewrite");
        let ops = rt_ops(&f);
        assert!(ops.contains(&RtOp::Lookup { levels }), "{ops:?}");
        assert!(ops.iter().any(|o| matches!(o, RtOp::Begin { .. })));
        assert!(ops.iter().any(|o| matches!(o, RtOp::Fetch { level: 0, .. })));
        assert_eq!(ops.iter().any(|o| matches!(o, RtOp::Get { what: RtGet::InGuest, .. })), levels == 2);
        // The programs read no instance mask; their extent never changes:
        // first hit, back faces culled when they test the facing.
        let begins = f.order.iter().flat_map(|&b| f.block(b).insts.iter()).copied().filter(|&v| matches!(f.op(v), ps5shader::ssa::Op::RtQuery(RtOp::Begin { .. }))).collect::<Vec<_>>();
        let flags = f.as_const(*f.args(begins[0]).last().unwrap());
        assert_eq!(flags, Some(if cull { 0x24 } else { 0x4 }));
        // The guest's nest stays for BVHs the host does not mirror.
        let bvh = f.order.iter().flat_map(|&b| f.block(b).insts.iter()).filter(|&&v| matches!(f.op(v), ps5shader::ssa::Op::Image(a) if matches!(a.op, ps5shader::ssa::ImageOp::BvhIntersect { boxes: true, .. }))).count();
        assert!(bvh > 0);
    }
}

#[test]
fn rt_loops_leave_loops_that_write_memory_between_candidates() {
    // A store in the loop depends on the order of the candidates: no
    // hardware copy.
    let mut f = ssa_of(&traversal_code(true, false));
    let out = ps5shader::opt::rt_loops::replace_traversals(&mut f);
    assert_eq!(out.replaced, 0);
    assert!(out.skipped.iter().any(|(_, why)| why.contains("buffer_store")), "{:?}", out.skipped);
}
