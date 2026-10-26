// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! A synthetic draw-path benchmark shaped like a UE4 frame: every draw
//! takes another material (a T# and an S# read from a descriptor table in
//! guest memory, a material constant buffer) and another slice of an
//! object-constant ring the CPU rewrites every frame; passes switch render
//! targets, two of which share memory (transient render targets), and each
//! pass starts with a clear; the command processor writes labels between
//! draws. Reports the recording thread's CPU time per
//! draw and the per-draw caches' counters.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored draw_path_bench
//! --nocapture` (`RPS5E_GPU_PROFILE=1` adds the per-stage breakdown;
//! `RPS5E_DRAW_BENCH_FRAMES=<n>` sets the frames, default 20).

use super::*;

const LEN: usize = 64 << 20;
const MATERIALS: u64 = 64;
const PASS_DRAWS: u64 = 200;
const PASSES: u64 = 6;
const RING_SLOTS: u64 = 4096;

/// buffer_load_dword vdata, vaddr, srsrc, 0 offen.
fn buffer_load_dword(vdata: u32, vaddr: u32, srsrc: u32) -> [u32; 2] {
    [0xE000_0000 | (0x0C << 18) | (1 << 12), vaddr | (vdata << 8) | ((srsrc / 4) << 16) | (0x80 << 24)]
}

/// s_load_dword{x4,x8} sdata, s[0:1], offset (no SOFFSET).
fn s_load(op: u32, sdata: u32, offset: u32) -> [u32; 2] {
    [0xF400_0000 | (op << 18) | (sdata << 6), 0xFA00_0000 | offset]
}

const S_WAITCNT_LGKM0: u32 = 0xBF8C_C07F;
const S_WAITCNT_VM0: u32 = 0xBF8C_3F70;

/// A 4x4 linear RGBA8 T# at `tex` (row pitch 64 texels).
fn texture_sharp(tex: u64) -> [u32; 8] {
    [(tex >> 8) as u32, ((tex >> 40) as u32 & 0xFF) | (56 << 20) | (3 << 30), 3 << 14, 0xFAC | (sharps::image_type::TYPE_2D << 28), 0, 0, 0, 0]
}

fn texture_bytes(seed: u8) -> Vec<u8> {
    let mut tex = vec![0u8; 64 * 4 * 4];
    for y in 0..4u32 {
        for x in 0..4u32 {
            let o = ((y * 64 + x) * 4) as usize;
            tex[o..o + 4].copy_from_slice(&[(x * 60) as u8, (y * 60) as u8, seed, 255]);
        }
    }
    tex
}

fn raw_buffer(addr: u64, bytes: u32) -> [u32; 4] {
    [addr as u32, (addr >> 32) as u32 & 0xFFFF, bytes, (3 << 28) | (20 << 12) | 0xFAC]
}

/// Textures per material.
const TEXTURES: u64 = 4;

/// The descriptor table words of every material (its T#s then an S#, 256
/// bytes each).
fn tables(textures: &[u64]) -> Vec<u32> {
    let mut words = Vec::new();
    for m in textures.chunks(TEXTURES as usize) {
        for &t in m {
            words.extend_from_slice(&texture_sharp(t));
        }
        words.extend_from_slice(&[2 | (2 << 3) | (2 << 6), 0, 0, 0]);
        words.resize(words.len().next_multiple_of(64), 0);
    }
    words
}

#[test]
#[ignore]
fn draw_path_bench() {
    // A benchmark: it loads the GPU and would disturb the timing tests that
    // run beside it, so it runs only when asked for.
    let Some(frames) = std::env::var("RPS5E_DRAW_BENCH_FRAMES").ok().and_then(|v| v.parse::<u64>().ok()) else { return };
    let _ = env_logger::builder().is_test(true).try_init();
    let mut g = Guest::new(LEN);
    // Vertex program (a merged stage: user data from s8): loads through the
    // object-constant V# in s[8:11] and the scene-data V# in s[12:15].
    let vl = buffer_load_dword(1, 0, 8);
    let vl2 = buffer_load_dword(2, 0, 12);
    let vs = g.program(&[vl[0], vl[1], vl2[0], vl2[1], S_WAITCNT_VM0, asm::S_ENDPGM]);
    // Pixel program: four T#s and an S# from the table at s[0:1], a load
    // through the material V# in s[4:7], four samples.
    let mut code = Vec::new();
    for k in 0..TEXTURES as u32 {
        code.extend_from_slice(&s_load(3, 12 + 8 * k, 0x20 * k));
    }
    code.extend_from_slice(&s_load(2, 44, 0x80));
    code.push(S_WAITCNT_LGKM0);
    code.extend_from_slice(&buffer_load_dword(16, 0, 4));
    code.push(S_WAITCNT_VM0);
    for k in 0..TEXTURES as u32 {
        code.extend_from_slice(&asm::image_sample(4 * k, 0, 12 + 8 * k, 44));
    }
    code.push(asm::S_ENDPGM);
    let ps = g.program(&code);
    let textures: Vec<u64> = (0..MATERIALS * TEXTURES).map(|t| g.alloc(&texture_bytes(t as u8))).collect();
    let table_words = tables(&textures);
    let table = g.alloc(&asm::bytes(&table_words));
    let material = g.zeroed((MATERIALS * 256) as usize);
    let ring = g.zeroed((RING_SLOTS * 256) as usize);
    let scene = g.zeroed(64 << 10);
    let labels = g.zeroed(4096);
    let verts = g.alloc(&floats(&[
        -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
        -0.9, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, //
        -1.0, -0.9, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
    ]));
    // Targets: A and B, and C over A's memory (another shape).
    let a = g.zeroed(128 * 128 * 4);
    let b_mem = g.zeroed(128 * 128 * 4);
    let targets = [rgba8_target(a, 128, 128), rgba8_target(b_mem, 128, 128), rgba8_target(a, 256, 64)];

    let dev = GpuDevice::create(false).expect("device");
    let mut b = Backend::new(&dev, g.mem).unwrap();
    crate::backend::attach_faults(&b, g.mem);
    let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 2 }).with(ps, TestProgram::TexturedPixel));
    let mut p = ShaderPipelines::new(&dev, g.mem, Some(compiler), None).unwrap();
    p.set_async_compile(false);
    p.set_vertex_input_resolver(Some(Box::new(FixedInputs(vec4_attrs(&[(verts, 96, 32, false)], &[(0, 0, 0), (1, 0, 16)])))));
    b.pipelines = Box::new(p);

    let draw = |i: u64, target: ColorTarget| {
        // Draws come sorted by material, a few per material.
        let m = (i / 3 * 7) % MATERIALS;
        let slot = i % RING_SLOTS;
        let mut vs_user = [0u32; 8];
        vs_user[..4].copy_from_slice(&raw_buffer(ring + slot * 256, 256));
        vs_user[4..].copy_from_slice(&raw_buffer(scene, 64 << 10));
        let mut ps_user = [0u32; 8];
        let tab = table + m * 256;
        ps_user[0] = tab as u32;
        ps_user[1] = (tab >> 32) as u32;
        ps_user[4..8].copy_from_slice(&raw_buffer(material + m * 256, 256));
        draw_into(target, stage(vs, &vs_user), stage(ps, &ps_user), primitive::TRI_LIST, auto(3, 1), None)
    };
    let mut cpu_ns = 0u64;
    let mut timed_draws = 0u64;
    let mut i = 0u64;
    crate::backend::profile::flush_local();
    let base = crate::backend::profile::NANOS.iter().map(|n| n.load(std::sync::atomic::Ordering::Relaxed)).collect::<Vec<_>>();
    for frame in 0..frames {
        // The CPU writes this frame's object constants and descriptor
        // tables (the same words: the pages fault and turn CPU-dirty).
        let fill: Vec<u8> = (0..RING_SLOTS * 256).map(|x| (x as u8).wrapping_add(frame as u8)).collect();
        // SAFETY: the ring's and the table's bytes in the guest view.
        unsafe {
            std::ptr::copy_nonoverlapping(fill.as_ptr(), ring as *mut u8, fill.len());
            let words = asm::bytes(&table_words);
            std::ptr::copy_nonoverlapping(words.as_ptr(), table as *mut u8, words.len());
        }
        crate::memory::note_sync_point();
        for pass in 0..PASSES {
            let target = targets[(pass % 3) as usize];
            let id = b.images.find_or_create(&mut b.sched, &mut b.buffers, &crate::images::color_key(&target)).unwrap();
            b.images.clear_color(&mut b.sched, id, vk::ClearColorValue { float32: [0.0; 4] }, None);
            for _ in 0..PASS_DRAWS {
                let d = draw(i, target);
                // Another CPU thread updates scene data now and then (its
                // pages are watched once uploaded: the write faults).
                if i % 50 == 10 {
                    // SAFETY: a word of the scene data in the guest view.
                    unsafe { ((scene + (i / 50 % 16) * 4096) as *mut u32).write_volatile(i as u32) };
                }
                // The command processor writes a label now and then.
                if i % 25 == 0 {
                    assert!(b.buffers.fill(&mut b.sched, labels + (i / 25 % 256) * 16, 16, i as u32).unwrap());
                }
                let t0 = crate::backend::profile::thread_cpu();
                b.draw(&d);
                // The first frame compiles and creates: not timed.
                if frame > 0 {
                    cpu_ns += crate::backend::profile::thread_cpu() - t0;
                    timed_draws += 1;
                }
                i += 1;
            }
        }
        b.flush();
        b.sched.flush_and_wait();
    }
    let s = b.pipelines.summary().unwrap_or_default();
    assert_eq!(b.stats.recorded_draws, i, "every draw is recorded: {s}");
    eprintln!("draw_path_bench: {timed_draws} draws, {:.2} us CPU per draw", cpu_ns as f64 / timed_draws.max(1) as f64 / 1000.0);
    eprintln!("draw_path_bench: {s}");
    if crate::backend::profile::on() {
        crate::backend::profile::flush_local();
        let names = crate::backend::profile::NAMES;
        let parts: Vec<String> = names
            .iter()
            .zip(crate::backend::profile::NANOS.iter())
            .zip(&base)
            .map(|((n, v), b0)| (n, v.load(std::sync::atomic::Ordering::Relaxed) - b0))
            .filter(|(_, v)| *v > 0)
            .map(|(n, v)| format!("{n}={:.2}", v as f64 / i as f64 / 1000.0))
            .collect();
        eprintln!("draw_path_bench: us/draw by stage: {}", parts.join(" "));
    }
}
