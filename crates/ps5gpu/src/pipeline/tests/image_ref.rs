// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Image instructions through the SPIR-V compiler on the device, against
//! hardware-measured results.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored image_ref`.

use super::*;

/// 4x4 linear RGBA8 storage texture (row pitch 64 texels) with DST_SEL
/// `dst_sel`; the program stores (1.0, 0.50196, 0.25098, 0.74902) at
/// texel (1, 2). Returns the 16 texels after the dispatch.
fn store_rgba8(dst_sel: u32) -> Vec<u32> {
    store_rgba8_with(dst_sel, Fetch::WriteBack)
}

/// [`store_rgba8`]; with `collect` the image reaches guest memory through
/// the per-submission read-back of linear storage images, not an explicit
/// write-back.
/// How the stored texels are fetched: an explicit write-back, the read-back
/// on a CPU access, or the read-back at a collection.
#[derive(Clone, Copy, PartialEq)]
enum Fetch {
    WriteBack,
    OnAccess,
    Collection,
}

fn store_rgba8_with(dst_sel: u32, fetch: Fetch) -> Vec<u32> {
    let mut g = Guest::new(16 << 20);
    // The reference program (ImageStoreBgraUsesInverseSwizzle): v20 = 1,
    // v21 = 2, v22 = 0, v[0:3] = data, image_store v[0:3], v[20:21],
    // s[0:7] dmask 0xF 2D.
    let code = [
        0x7e28_0281, 0x7e2a_0282, 0x7e2c_0280, 0x7e00_02ff, 0x3f80_0000, 0x7e02_02ff, 0x3f00_8081, 0x7e04_02ff, 0x3e80_8081, 0x7e06_02ff, 0x3f3f_bfc0,
        0xf020_0f08, 0x0000_0014, asm::S_ENDPGM,
    ];
    let cs = g.program(&code);
    let tex = g.zeroed(64 * 4 * 4);
    let t_sharp = [
        (tex >> 8) as u32,
        ((tex >> 40) as u32 & 0xFF) | (56 << 20) | (3 << 30),
        (3 >> 2) | (3 << 14),
        dst_sel | (sharps::image_type::TYPE_2D << 28),
        0,
        0,
        0,
        0,
    ];
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    let mut shader = stage(cs, &t_sharp);
    shader.rsrc1 = (0xC0 << 12) | (1 << 21);
    shader.rsrc2 = (t_sharp.len() as u32) << 1;
    rig.images.readback_linear = match fetch {
        Fetch::OnAccess => crate::images::LinearReadback::OnAccess,
        _ => crate::images::LinearReadback::Eager,
    };
    let d = ResolvedDispatch {
        packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() },
        shader,
        rsrc3: 0,
        threads: [1, 1, 1],
        start: [0; 3],
    };
    assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
    let size = 64 * 4 * 4;
    if fetch == Fetch::Collection {
        rig.images.process_downloads(&mut rig.sched, &mut rig.buffers);
    } else if fetch == Fetch::OnAccess {
        // As a CPU access would: the image's pages block reads until the
        // fault writes it back.
        let page = tex & !(crate::memory::PAGE_BYTES - 1);
        assert!(g.mem.pages.watchers(page).0, "the GPU-written linear image is not read-watched");
        let hits = rig.images.table.take_read_backs(tex).unwrap();
        assert_eq!(hits.len(), 1);
        let tracker = rig.buffers.tracker.clone();
        rig.images.resolve_read_backs(&mut rig.sched, &mut rig.buffers, &tracker, &hits, false);
        assert!(!g.mem.pages.watchers(page).0);
    } else {
        rig.images.writeback_range(&mut rig.sched, &mut rig.buffers, tex, size);
    }
    rig.buffers.download(&mut rig.sched, tex, size);
    let mut out = vec![0u8; size as usize];
    assert!(g.mem.read_backing(tex, &mut out));
    (0..16)
        .map(|i| {
            let (x, y) = (i % 4, i / 4);
            let o = (y * 64 + x) * 4;
            u32::from_le_bytes(out[o..o + 4].try_into().unwrap())
        })
        .collect()
}

#[test]
#[ignore]
fn image_ref_store_applies_inverse_dst_sel() {
    // BGRA (DST_SEL Z, Y, X, W): memory R takes data Z, B takes data X.
    let texels = store_rgba8(0xF2E);
    for (i, &t) in texels.iter().enumerate() {
        assert_eq!(t, if i == 9 { 0xbfff_8040 } else { 0 }, "texel {i}: {t:#010x}");
    }
    // Identity.
    let texels = store_rgba8(0xFAC);
    assert_eq!(texels[9], 0xbf40_80ff, "{:#010x}", texels[9]);
}

/// A linear storage image a shader writes reaches guest memory when the
/// CPU touches its pages (the CPU reads such read-back surfaces directly).
#[test]
#[ignore]
fn image_ref_linear_storage_image_is_read_back() {
    let texels = store_rgba8_with(0xFAC, Fetch::OnAccess);
    assert_eq!(texels[9], 0xbf40_80ff, "{:#010x}", texels[9]);
}

/// The same at the next collection (the default mode).
#[test]
#[ignore]
fn image_ref_linear_storage_image_is_read_back_at_collection() {
    let texels = store_rgba8_with(0xFAC, Fetch::Collection);
    assert_eq!(texels[9], 0xbf40_80ff, "{:#010x}", texels[9]);
}

/// The hardware test `ImageSampleForceDegamma`: a compute program loads
/// (u, v) per lane, samples a 4x4 linear texture and stores the four
/// channels. Returns the 32 lanes' results.
fn sample_force_degamma(g: &mut Guest, rig: &mut Rig, format: u32, filter: u32, forced: bool, centers: bool) -> Vec<f32> {
    const THREADS: u32 = 32;
    const CODE: [u32; 13] = [
        0x1614_008c, 0xe03c_1000, 0x8000_010a, 0xbf8c_3f70, 0xf090_0f08, 0x0082_0401, 0xbf8c_3f70, 0x3416_0084, 0xe078_1000, 0x8001_040b, 0xbf81_0000, 0xbf81_0000, 0xbf81_0000,
    ];
    let code8 = |x: u32, y: u32, c: u32| ((y * 61 + x * 17 + c * 97 + 5) % 256) as u8;
    // Compute programs take 16 user SGPRs here: the S# the reference passes
    // in s[16:19] is moved there first (s_mov_b32 sN, literal).
    let s_sharp = [0x92 | if forced { 1 << 20 } else { 0 }, 0x00ff_f000, (filter << 20) | (filter << 22) | (1 << 24), 0u32];
    let mut code = Vec::new();
    for (k, w) in s_sharp.iter().enumerate() {
        code.extend_from_slice(&[0xBE80_03FF | (16 + k as u32) << 16, *w]);
    }
    code.extend_from_slice(&CODE);
    let cs = g.program(&code);
    // Linear 4x4: rows 64 texels (256 bytes) apart.
    let mut texels = vec![0u8; 4096];
    for y in 0..4u32 {
        for x in 0..4u32 {
            for c in 0..4u32 {
                texels[(y * 256 + x * 4 + c) as usize] = code8(x, y, c);
            }
        }
    }
    let tex = g.alloc(&texels);
    let mut input = Vec::new();
    for tid in 0..THREADS {
        let (x, y) = (tid % 4, (tid / 4) % 4);
        let (u, v) = if centers {
            ((x as f32 + 0.5) / 4.0, (y as f32 + 0.5) / 4.0)
        } else {
            ((((tid * 5 + 1) % 13) as f32 + 0.3) / 13.0, (((tid * 7 + 3) % 13) as f32 + 0.6) / 13.0)
        };
        input.extend_from_slice(&floats(&[u, v, 0.0]));
    }
    let input_addr = g.alloc(&input);
    let out_bytes = THREADS as u64 * 16;
    let output = g.zeroed(out_bytes as usize);
    let v_sharp = |a: u64, bytes: u32| [a as u32, (a >> 32) as u32 & 0xFFFF, bytes, 0x3101_6fac];
    let t_sharp = [
        (tex >> 8) as u32,
        ((tex >> 40) as u32 & 0xFF) | (format << 20) | (3 << 30),
        (3 >> 2) | (3 << 14),
        0xFAC | (sharps::image_type::TYPE_2D << 28),
        0,
        0,
        0,
        0,
    ];
    let mut user = Vec::new();
    user.extend_from_slice(&v_sharp(input_addr, input.len() as u32));
    user.extend_from_slice(&v_sharp(output, out_bytes as u32));
    user.extend_from_slice(&t_sharp);
    let mut shader = stage(cs, &user);
    shader.rsrc1 = 3 | (0xC0 << 12) | (1 << 21);
    shader.rsrc2 = (user.len() as u32) << 1;
    let d = ResolvedDispatch {
        packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1 | (1 << 15), ..DispatchPacket::default() },
        shader,
        rsrc3: 0,
        threads: [THREADS, 1, 1],
        start: [0; 3],
    };
    assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
    assert!(rig.buffers.download(&mut rig.sched, output, out_bytes));
    let mut out = vec![0u8; out_bytes as usize];
    assert!(g.mem.read_backing(output, &mut out));
    out.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

/// FORCE_DEGAMMA decodes 8_8_8_8_UNORM as sRGB, exactly as sampling the
/// texture as 8_8_8_8_SRGB (hardware test `ImageSampleForceDegamma`).
#[test]
#[ignore]
fn image_ref_sample_force_degamma() {
    let srgb = |c: u8| {
        let c = c as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let mut g = Guest::new(16 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    let mut sample = |format, filter, forced, centers| sample_force_degamma(&mut g, &mut rig, format, filter, forced, centers);
    let forced = sample(56, 0, true, true);
    for tid in 0..32u32 {
        let (x, y) = (tid % 4, (tid / 4) % 4);
        for c in 0..4u32 {
            let code = ((y * 61 + x * 17 + c * 97 + 5) % 256) as u8;
            let expected = if c == 3 { code as f32 / 255.0 } else { srgb(code) };
            let v = forced[(tid * 4 + c) as usize];
            assert!((v - expected).abs() <= 0.005, "lane {tid} channel {c}: {v} expected {expected}");
        }
    }
    assert_eq!(forced, sample(130, 0, false, true), "as 8_8_8_8_SRGB, point");
    assert_ne!(sample(56, 0, false, true)[..8], forced[..8], "without FORCE_DEGAMMA");
    assert_eq!(sample(56, 1, true, false), sample(130, 1, false, false), "as 8_8_8_8_SRGB, bilinear");
}
