// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Reads at device addresses the host cannot bound (a ray query's BVH
//! nodes) may reach any page: mirrored pages must hold the CPU's latest
//! bytes, and pages no mirror holds are read from guest memory.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored unranged`.

use super::*;
use ps5shader::testing::{asm as sa, raw_buffer_descriptor};

const S_WAITCNT_0: u32 = 0xBF8C_0000;

/// image_bvh_intersect_ray v[24:27] <- v4 (node), v5 (extent),
/// v[6:8] (origin), v[9:11] (direction), v[12:14] (inverse direction),
/// T# s[8:11]; `out[tid] = v24` through the V# in s[0:3].
fn first_child_program(g: &mut Guest) -> u64 {
    let mut code = vec![
        sa::vop1(sa::OP_V_MOV_B32, 4, sa::inline_u32(5)),
        sa::vop1(sa::OP_V_MOV_B32, 5, sa::LITERAL),
        10.0f32.to_bits(),
    ];
    for v in 6..9 {
        code.push(sa::vop1(sa::OP_V_MOV_B32, v, sa::inline_u32(0)));
    }
    for v in 9..15 {
        code.push(sa::vop1(sa::OP_V_MOV_B32, v, sa::F_ONE));
    }
    code.extend([0xF198_8F07, 0x0002_1804, 0x0807_0605, 0x0C0B_0A09, 0x0000_0E0D]);
    code.push(S_WAITCNT_0);
    // v1 = v0 << 2
    code.push((0x1A << 25) | (1 << 17) | (128 + 2));
    code.extend(asm::buffer_store_dword(24, 1, 0));
    code.push(sa::S_ENDPGM);
    g.program(&code)
}

/// An fp32 box node: four children around the origin (every one hit).
fn box_node(children: [u32; 4]) -> Vec<u8> {
    let mut words = children.to_vec();
    for _ in 0..4 {
        words.extend([(-1.0f32).to_bits(); 3]);
        words.extend([1.0f32.to_bits(); 3]);
    }
    asm::bytes(&words)
}

fn out_words(rig: &mut Rig, mem: &crate::memory::GuestGpuMemory, out: u64) -> Vec<u32> {
    assert!(rig.buffers.download(&mut rig.sched, out, 256));
    let mut b = vec![0u8; 256];
    assert!(mem.read_backing(out, &mut b));
    b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// A one-wave dispatch of `cs` querying the node at `node`, writing the
/// first child pointer of every lane to `out`.
fn query(cs: u64, node: u64, out: u64) -> ResolvedDispatch {
    // Raw output V#, then the BVH T#: base in 256-byte units, last node 256.
    let mut user = raw_buffer_descriptor(out, 256).to_vec();
    user.extend([0, 0, 0, 0]);
    user.extend([(node >> 8) as u32, ((node >> 40) & 0xFF) as u32, 0x100, 0]);
    let mut shader = stage(cs, &user);
    shader.rsrc1 = (0xC0 << 12) | (1 << 21) | 7;
    ResolvedDispatch { packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() }, shader, rsrc3: 0, threads: [64, 1, 1], start: [0; 3] }
}

fn spirv_rig(g: &Guest) -> Rig {
    Rig::with_compiler(g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None)
}

/// A BVH node in a mirror (made by an earlier use of its range) that the
/// CPU rewrites: the next ray query reads the new node, with nothing
/// obtaining the node's range in between.
#[test]
#[ignore]
fn unranged_bvh_reads_see_cpu_writes_into_mirrors() {
    let mut g = Guest::new(16 << 20);
    let mut rig = spirv_rig(&g);
    let cs = first_child_program(&mut g);
    let node = g.alloc(&box_node([0x1000, 0x2000, 0x3000, 0x4000]));
    let out = g.zeroed(256);
    let d = query(cs, node, out);
    let mem = g.mem;

    // An earlier use mirrors the node (as the GPU build of a BVH does).
    rig.buffers.obtain(&mut rig.sched, node, 128, false).unwrap();
    assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
    assert_eq!(out_words(&mut rig, mem, out)[..4], [0x1000; 4]);

    assert!(g.mem.write_backing(node, &box_node([0x5000, 0x6000, 0x7000, 0x8000])));
    rig.buffers.tracker.mark_cpu_dirty_pages(node, 128);
    assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
    assert_eq!(out_words(&mut rig, mem, out)[..4], [0x5000; 4], "the query read the mirror's old node");
    assert!(rig.buffers.stats().unranged_syncs > 0);

    // Unchanged since: nothing is uploaded for the next query.
    let uploads = rig.buffers.stats().uploads;
    assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
    assert_eq!(out_words(&mut rig, mem, out)[..4], [0x5000; 4]);
    assert_eq!(rig.buffers.stats().uploads, uploads);
}

/// A BVH node only the CPU wrote (no stage used its range, no mirror holds
/// it): the query reads it in guest memory, not zeros (a zero leaf list has
/// no end marker, and a traversal walking it never ends).
#[test]
#[ignore]
fn unranged_bvh_reads_reach_memory_without_mirrors() {
    let mut g = Guest::new(16 << 20);
    let mut rig = spirv_rig(&g);
    let cs = first_child_program(&mut g);
    let node = g.alloc(&box_node([0x1000, 0x2000, 0x3000, 0x4000]));
    let out = g.zeroed(256);
    let mem = g.mem;
    assert!(rig.buffers.find_containing(node, 128).is_none());
    assert_eq!(rig.dispatch(&query(cs, node, out)), DrawOutcome::Recorded);
    assert_eq!(out_words(&mut rig, mem, out)[..4], [0x1000; 4], "the query read a page without a mirror as zeros");
    assert!(rig.buffers.find_containing(node, 128).is_none(), "the query made a mirror");

    // Another node the CPU wrote later: read where it is too.
    let other = g.alloc(&box_node([0x9000, 0xA000, 0xB000, 0xC000]));
    assert_eq!(rig.dispatch(&query(cs, other, out)), DrawOutcome::Recorded);
    assert_eq!(out_words(&mut rig, mem, out)[..4], [0x9000; 4]);
}


/// image_bvh_intersect_ray of the node pointer `pointer`, origin (0.25,
/// 0.5, 0.125), direction and inverse direction 1: the four result dwords
/// of lane `tid` go to `out[tid]`, `out[64 + tid]`, `out[128 + tid]` and
/// `out[192 + tid]`.
fn all_results_program(g: &mut Guest, pointer: u32) -> u64 {
    let mut code = vec![sa::vop1(sa::OP_V_MOV_B32, 4, sa::LITERAL), pointer, sa::vop1(sa::OP_V_MOV_B32, 5, sa::LITERAL), 10.0f32.to_bits()];
    for (v, x) in [(6, 0.25f32), (7, 0.5), (8, 0.125)] {
        code.extend([sa::vop1(sa::OP_V_MOV_B32, v, sa::LITERAL), x.to_bits()]);
    }
    for v in 9..15 {
        code.push(sa::vop1(sa::OP_V_MOV_B32, v, sa::F_ONE));
    }
    code.extend([0xF198_8F07, 0x0002_1804, 0x0807_0605, 0x0C0B_0A09, 0x0000_0E0D]);
    code.push(S_WAITCNT_0);
    // v1 = v0 << 2
    code.push((0x1A << 25) | (1 << 17) | (128 + 2));
    for (k, v) in (24..28).enumerate() {
        let [w0, w1] = asm::buffer_store_dword(v, 1, 0);
        code.extend([w0 | (k as u32 * 256), w1]);
    }
    code.push(sa::S_ENDPGM);
    g.program(&code)
}

/// Triangle (types 0 and 1), fp16 box and fp32 box nodes read with
/// `RPS5E_BVH_VECTOR_LOADS` on give the results of the per-dword loads.
#[test]
#[ignore]
fn bvh_vector_node_loads_match_dword_loads() {
    use crate::shader::{EmitOptions, Features, SpirvCompiler};
    let rig_with = |g: &Guest, vector: bool| {
        Rig::with_compiler(
            g,
            |dev| {
                let features = Features::of(dev);
                Arc::new(SpirvCompiler::new(Features { emit: EmitOptions { bvh_vector_loads: vector, ..features.emit }, ..features }, Some(g.mem)))
            },
            None,
            None,
        )
    };
    // A triangle across the ray (vertices 0..3, the second triangle's
    // third vertex, triangle id), and node words for the boxes: fp16
    // halves and fp32 bounds around and off the ray, sorted children.
    let mut triangle: Vec<u32> = [[2.0f32, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [1.5, 1.5, 0.0]].iter().flatten().map(|f| f.to_bits()).collect();
    triangle.extend([0, 0, 0, 0x0000_0123]);
    let mut box32 = vec![0x1005u32, 0x2005, 0x3005, 0x4005];
    for (lo, hi) in [(-1.0f32, 1.0f32), (2.0, 3.0), (0.5, 4.0), (-3.0, -2.0)] {
        box32.extend([lo.to_bits(); 3]);
        box32.extend([hi.to_bits(); 3]);
    }
    let half = |x: f32| half_bits(x) as u32;
    let mut box16 = vec![0x1004u32, 0x2004, 0x3004, 0x4004];
    for (lo, hi) in [(-1.0f32, 1.0f32), (2.0, 3.0), (0.5, 4.0), (-3.0, -2.0)] {
        box16.extend([half(lo) | half(lo) << 16, half(lo) | half(hi) << 16, half(hi) | half(hi) << 16]);
    }
    let mut results = Vec::new();
    for vector in [false, true] {
        let mut g = Guest::new(16 << 20);
        let mut rig = rig_with(&g, vector);
        let mem = g.mem;
        let mut node_bytes = asm::bytes(&triangle);
        node_bytes.resize(128, 0);
        let tri = g.alloc(&node_bytes);
        let b32 = g.alloc(&asm::bytes(&box32));
        let b16 = g.alloc(&asm::bytes(&box16));
        let out = g.zeroed(1024);
        let mut rows = Vec::new();
        for (node, pointer) in [(tri, 0u32), (tri, 1), (b16, 4), (b32, 5)] {
            let cs = all_results_program(&mut g, pointer);
            let mut d = query(cs, node, out);
            // Box sorting on (T# word 1 bit 31).
            d.shader.user_data[9] |= 0x8000_0000;
            assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
            assert!(rig.buffers.download(&mut rig.sched, out, 1024));
            let mut b = vec![0u8; 1024];
            assert!(mem.read_backing(out, &mut b));
            let w: Vec<u32> = b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            rows.push([w[0], w[64], w[128], w[192]]);
        }
        results.push(rows);
    }
    assert_eq!(results[0], results[1], "vector node loads changed a result");
    // The node words reached the tests (not the zeros of a missing page).
    assert_ne!(results[0][0], [0; 4]);
    assert_ne!(results[0][3], [0xFFFF_FFFF; 4]);
}

/// f32 → f16 bits (normal values only).
fn half_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = ((b >> 13) & 0x3FF) as u16;
    if x == 0.0 {
        sign
    } else {
        sign | ((exp as u16) << 10) | mant
    }
}
