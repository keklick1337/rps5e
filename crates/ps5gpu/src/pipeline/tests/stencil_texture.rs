// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! An 8-bit texture over the stencil plane of a depth/stencil target reads
//! the stencil the GPU wrote. Demon's Souls (PPSA01342) marks lit pixels
//! with stencil bit 0x40 in its geometry passes; its light-tile classifier
//! samples the stencil buffer as an 8_UINT texture. The stencil lives only
//! in the depth image's stencil aspect, so a texture loaded from the
//! plane's guest memory saw no lit pixel and every lighting dispatch ran
//! no groups (unlit scene: dark silhouettes, flat fog-coloured walls).
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored stencil_texture`.

use super::*;
use ps5shader::testing::raw_buffer_descriptor;

#[test]
#[ignore]
fn stencil_texture_reads_the_depth_targets_stencil() {
    let mut g = Guest::new(16 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    let (w, h) = (64u32, 64u32);
    let z = g.zeroed((w * h * 4) as usize);
    // The plane's guest memory holds an older value than the GPU's stencil.
    let s = g.alloc(&vec![0x11u8; (w * h) as usize]);
    let mut target = d32_target(z, w, h);
    target.host_format = vk::Format::D32_SFLOAT_S8_UINT;
    target.has_stencil = true;
    target.stencil_read_base = s;
    target.stencil_write_base = s;
    target.stencil_slice_size = (w * h) as u64;
    let id = rig.images.find_or_create(&mut rig.sched, &mut rig.buffers, &crate::images::depth_key(&target)).unwrap();
    rig.images.prepare(&mut rig.sched, &mut rig.buffers, id, false);
    rig.images.clear_depth_stencil(&mut rig.sched, id, Some(0.0), Some(0x40), None, None);
    let tick = rig.sched.current_tick();
    rig.images.mark_gpu_written(id, tick);

    // v20 = 1, v21 = 2; image_load v4, v[20:21], s[0:7] dmask 0x1 2D;
    // buffer_store_dword v4, s[8:11] offset 0.
    let cs = g.program(&[0x7E28_0281, 0x7E2A_0282, 0xF000_0108, 0x0000_0414, 0xBF8C_3F70, 0xE070_0000, 0x8002_0400, asm::S_ENDPGM]);
    let out = g.alloc(&asm::bytes(&[0xDEAD_BEEFu32; 4]));
    let t_sharp = [
        (s >> 8) as u32,
        ((s >> 40) as u32 & 0xFF) | (5 << 20) | (((w - 1) & 3) << 30),
        ((w - 1) >> 2) | ((h - 1) << 14),
        0xFAC | (sharps::image_type::TYPE_2D << 28),
        0,
        0,
        0,
        0,
    ];
    let mut user = t_sharp.to_vec();
    user.extend_from_slice(&raw_buffer_descriptor(out, 16));
    let mut shader = stage(cs, &user);
    shader.rsrc1 = (0xC0 << 12) | (1 << 21);
    let d = ResolvedDispatch {
        packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() },
        shader,
        rsrc3: 0,
        threads: [1, 1, 1],
        start: [0; 3],
    };
    assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
    assert!(rig.buffers.download(&mut rig.sched, out, 16));
    let mut b = [0u8; 4];
    assert!(g.mem.read_backing(out, &mut b));
    assert_eq!(u32::from_le_bytes(b), 0x40, "stencil texel (1, 2)");
}

/// A deferred decal samples the scene depth while testing it and clearing
/// a stencil bit (SILENT HILL: stencil EQUAL 1, operations ZERO, depth
/// tested, not written). Depth is only read and the stencil aspect is a
/// separate subresource: the draw shares rendering scopes (no isolation
/// as a feedback loop), samples the depth, and its stencil writes reach
/// the next draws.
#[test]
#[ignore]
fn decal_samples_depth_while_writing_stencil() {
    let mut g = Guest::new(16 << 20);
    let (w, h) = (64u32, 64u32);
    let vs = g.program(&[asm::S_ENDPGM]);
    let sample = asm::image_sample(0, 0, 0, 8);
    let textured = g.program(&[sample[0], sample[1], asm::S_ENDPGM]);
    let solid = solid_program(&mut g);
    let z = g.zeroed((w * h * 4) as usize);
    let s = g.zeroed((w * h) as usize);
    let mut depth = d32_target(z, w, h);
    depth.host_format = vk::Format::D32_SFLOAT_S8_UINT;
    depth.has_stencil = true;
    depth.stencil_read_base = s;
    depth.stencil_write_base = s;
    depth.stencil_slice_size = (w * h) as u64;
    let compiler = Arc::new(
        TestCompiler::new()
            .with(vs, TestProgram::PassthroughVertex { inputs: 2 })
            .with(textured, TestProgram::TexturedPixel)
            .with(solid, TestProgram::SolidPixel),
    );
    let verts = g.alloc(&floats(&[
        -1.0, -1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, //
        1.0, -1.0, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0, //
        -1.0, 1.0, 0.5, 1.0, 0.0, 1.0, 0.0, 0.0,
    ]));
    let mut rig = Rig::new(&g, compiler, Some(vec4_attrs(&[(verts, 96, 32, false)], &[(0, 0, 0), (1, 0, 16)])), None);
    let id = rig.images.find_or_create(&mut rig.sched, &mut rig.buffers, &crate::images::depth_key(&depth)).unwrap();
    rig.images.prepare(&mut rig.sched, &mut rig.buffers, id, true);
    rig.images.clear_depth_stencil(&mut rig.sched, id, Some(0.25), Some(1), None, None);
    let tick = rig.sched.current_tick();
    rig.images.mark_gpu_written(id, tick);
    let target = rgba8_target(g.zeroed((w * h * 4) as usize), w, h);
    // The depth surface as a 32-bit float texture; clamp, point filtering.
    let t_sharp = [
        (z >> 8) as u32,
        ((z >> 40) as u32 & 0xFF) | (22 << 20) | (((w - 1) & 3) << 30),
        ((w - 1) >> 2) | ((h - 1) << 14),
        0xFAC | (sharps::image_type::TYPE_2D << 28),
        0,
        0,
        0,
        0,
    ];
    let mut user = t_sharp.to_vec();
    user.extend_from_slice(&[2 | (2 << 3) | (2 << 6), 0, 0, 0]);
    // Stencil EQUAL (ref 1, mask 1, write mask 1), every operation ZERO;
    // depth tested (ALWAYS), not written.
    let decal = DepthStencilState { depth_test: true, depth_compare: 7, stencil_test: true, stencil_compare: 2, stencil_control: 0x11_1111, stencil_ref_mask: 0x0101_0101, ..DepthStencilState::default() };
    let draw = |ps: Stage, ds: DepthStencilState| {
        let mut d = draw_into(target, stage(vs, &[]), ps, primitive::RECT_LIST, auto(3, 1), None);
        d.depth = Some(depth);
        d.depth_state = ds;
        d
    };
    assert_eq!(rig.draw(&draw(stage(textured, &user), decal)), DrawOutcome::Recorded);
    assert_eq!(rig.provider.stats().isolated[3], 0, "not a feedback loop");
    let p = rig.pixels(&target);
    assert_eq!(px(&p, w, 10, 10)[0], 64, "the decal sampled depth 0.25");
    // The decal cleared stencil bit 0: EQUAL 1 now fails, EQUAL 0 passes.
    let keep = |reference: u32| DepthStencilState { stencil_test: true, stencil_compare: 2, stencil_ref_mask: reference | 0x0100, ..DepthStencilState::default() };
    assert_eq!(rig.draw(&draw(stage(solid, &color_words([0.0, 1.0, 0.0, 1.0])), keep(1))), DrawOutcome::Recorded);
    assert_eq!(px(&rig.pixels(&target), w, 10, 10)[1], 0, "stencil 1 is gone");
    assert_eq!(rig.draw(&draw(stage(solid, &color_words([0.0, 0.0, 1.0, 1.0])), keep(0))), DrawOutcome::Recorded);
    assert_eq!(px(&rig.pixels(&target), w, 10, 10), [0, 0, 255, 255], "stencil is 0");
}
