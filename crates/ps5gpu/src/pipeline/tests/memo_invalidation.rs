// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The per-draw memos (texture lookups, stage bindings, prepared stages,
//! stream copies) against the changes that must invalidate them: a draw
//! repeated after the change sees the new contents, and a change elsewhere
//! leaves the memos in place.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored memo_`.

use super::*;

/// A 4x4 linear RGBA8 texture (row pitch 64 texels) of one colour.
fn solid_texture(rgba: [u8; 4]) -> Vec<u8> {
    let mut tex = vec![0u8; 64 * 4 * 4];
    for y in 0..4usize {
        for x in 0..4usize {
            let o = (y * 64 + x) * 4;
            tex[o..o + 4].copy_from_slice(&rgba);
        }
    }
    tex
}

fn texture_sharp(tex: u64) -> [u32; 8] {
    [(tex >> 8) as u32, ((tex >> 40) as u32 & 0xFF) | (56 << 20) | (3 << 30), 3 << 14, 0xFAC | (sharps::image_type::TYPE_2D << 28), 0, 0, 0, 0]
}

struct Scene {
    vs: u64,
    textured: u64,
    solid: u64,
    tex: u64,
    rig: Rig,
}

fn scene(g: &mut Guest) -> Scene {
    let vs = g.program(&[asm::S_ENDPGM]);
    let sample = asm::image_sample(0, 0, 0, 8);
    let textured = g.program(&[sample[0], sample[1], asm::S_ENDPGM]);
    let solid = solid_program(g);
    let tex = g.alloc(&solid_texture([200, 0, 0, 255]));
    let verts = g.alloc(&floats(&[
        -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
        3.0, -1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, //
        -1.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0,
    ]));
    let compiler = Arc::new(
        TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 2 }).with(textured, TestProgram::TexturedPixel).with(solid, TestProgram::SolidPixel),
    );
    let rig = Rig::new(g, compiler, Some(vec4_attrs(&[(verts, 96, 32, false)], &[(0, 0, 0), (1, 0, 16)])), None);
    Scene { vs, textured, solid, tex, rig }
}

impl Scene {
    fn sample_into(&mut self, target: ColorTarget) {
        let mut user = texture_sharp(self.tex).to_vec();
        user.extend_from_slice(&[2 | (2 << 3) | (2 << 6), 0, 0, 0]);
        let d = draw_into(target, stage(self.vs, &[]), stage(self.textured, &user), primitive::TRI_LIST, auto(3, 1), None);
        assert_eq!(self.rig.draw(&d), DrawOutcome::Recorded);
    }

    fn fill(&mut self, target: ColorTarget, c: [f32; 4]) {
        let d = draw_into(target, stage(self.vs, &[]), stage(self.solid, &color_words(c)), primitive::TRI_LIST, auto(3, 1), None);
        assert_eq!(self.rig.draw(&d), DrawOutcome::Recorded);
    }
}

/// A texture rendered into by another image over its memory (newer GPU
/// contents) is sampled with those contents by the next draw that binds the
/// same T#, while rendering into an unrelated target keeps the memoised
/// texture binding.
#[test]
#[ignore]
fn memo_texture_binding_follows_gpu_writes_over_its_memory() {
    let mut g = Guest::new(16 << 20);
    let mut s = scene(&mut g);
    let (a, b, c) = (rgba8_target(g.zeroed(64 * 64 * 4), 64, 64), rgba8_target(g.zeroed(64 * 64 * 4), 64, 64), rgba8_target(g.zeroed(64 * 64 * 4), 64, 64));
    // The first lookup loads the texture; the second, which changes
    // nothing, is memoised.
    s.sample_into(a);
    s.sample_into(a);
    assert_eq!(px(&s.rig.pixels(&a), 64, 10, 10), [200, 0, 0, 255], "the texture from guest memory");
    // Unrelated targets written and cleared: the texture binding stays.
    s.fill(c, [0.0, 0.0, 1.0, 1.0]);
    let reuses = s.rig.provider.stats.binding_reuses;
    let lookups = s.rig.provider.stats.view_lookups;
    s.sample_into(b);
    assert_eq!(px(&s.rig.pixels(&b), 64, 10, 10), [200, 0, 0, 255]);
    if crate::images::image_generations_enabled() {
        assert!(s.rig.provider.stats.binding_reuses >= reuses + 2, "image and sampler bindings reused across an unrelated target's first write");
        assert_eq!(s.rig.provider.stats.view_lookups, lookups, "no texture lookup");
    }
    // A render target over the texture's memory (the same 64-texel rows)
    // is filled green: the texture's newest contents are the target's.
    let over = rgba8_target(s.tex, 64, 4);
    s.fill(over, [0.0, 1.0, 0.0, 1.0]);
    s.sample_into(b);
    assert_eq!(px(&s.rig.pixels(&b), 64, 10, 10), [0, 255, 0, 255], "sampled after the GPU wrote the texture's memory");
}

/// A CPU write to a texture between two draws binding the same T# reaches
/// the second draw.
#[test]
#[ignore]
fn memo_texture_binding_follows_cpu_writes() {
    let mut g = Guest::new(16 << 20);
    let mut s = scene(&mut g);
    let (a, b) = (rgba8_target(g.zeroed(64 * 64 * 4), 64, 64), rgba8_target(g.zeroed(64 * 64 * 4), 64, 64));
    let images = s.rig.images.table.clone();
    g.mem.attach(Arc::new(NoBufferFaults), Arc::new(TableFaults(images)), Arc::new(NoUnmapping));
    s.sample_into(a);
    assert_eq!(px(&s.rig.pixels(&a), 64, 10, 10), [200, 0, 0, 255]);
    let new = solid_texture([0, 0, 255, 255]);
    // SAFETY: the texture's bytes in the guest view (watched: the write faults).
    unsafe { std::ptr::copy_nonoverlapping(new.as_ptr(), s.tex as *mut u8, new.len()) };
    crate::memory::note_sync_point();
    s.sample_into(b);
    assert_eq!(px(&s.rig.pixels(&b), 64, 10, 10), [0, 0, 255, 255], "sampled after the CPU rewrote the texture");
}

/// s[12:19] = T# and s[20:23] = S# from the table at s[0:1]; a sample.
fn table_program() -> Vec<u32> {
    let sample = asm::image_sample(0, 0, 12, 20);
    vec![0xF40C_0300, 0xFA00_0000, 0xF408_0500, 0xFA00_0020, 0xBF8C_C07F, sample[0], sample[1], asm::S_ENDPGM]
}

fn table_words(tex: u64) -> Vec<u32> {
    let mut w = texture_sharp(tex).to_vec();
    w.extend_from_slice(&[2 | (2 << 3) | (2 << 6), 0, 0, 0]);
    w
}

/// A stage whose T# is read from a table in guest memory is reused while
/// the table's page is unchanged (GPU writes elsewhere do not matter) and
/// materialised again after a CPU or a GPU write to it.
#[test]
#[ignore]
fn memo_prepared_stage_follows_its_table() {
    let mut g = Guest::new(16 << 20);
    let mut s = scene(&mut g);
    let ps = g.program(&table_program());
    let blue = g.alloc(&solid_texture([0, 0, 255, 255]));
    let table = g.alloc(&asm::bytes(&table_words(s.tex)));
    let labels = g.zeroed(4096);
    let targets: Vec<ColorTarget> = (0..4).map(|_| rgba8_target(g.zeroed(64 * 64 * 4), 64, 64)).collect();
    let compiler = Arc::new(TestCompiler::new().with(s.vs, TestProgram::PassthroughVertex { inputs: 2 }).with(ps, TestProgram::TexturedPixel));
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, -1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, -1.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0]));
    s.rig = Rig::new(&g, compiler, Some(vec4_attrs(&[(verts, 96, 32, false)], &[(0, 0, 0), (1, 0, 16)])), None);
    g.mem.attach(Arc::new(TrackerFaults(s.rig.buffers.tracker.clone())), Arc::new(TableFaults(s.rig.images.table.clone())), Arc::new(NoUnmapping));
    // The table's page is watched once a mirror holds it.
    s.rig.buffers.obtain(&mut s.rig.sched, table, 64, false).unwrap();
    let user = [table as u32, (table >> 32) as u32];
    let draw = |s: &mut Scene, t: ColorTarget| {
        let d = draw_into(t, stage(s.vs, &[]), stage(ps, &user), primitive::TRI_LIST, auto(3, 1), None);
        assert_eq!(s.rig.draw(&d), DrawOutcome::Recorded);
    };
    draw(&mut s, targets[0]);
    fn frontend_of(s: &Scene) -> std::sync::MutexGuard<'_, crate::pipeline::frontend::ShaderFrontend> {
        s.rig.provider.frontend.lock().unwrap_or_else(|e| e.into_inner())
    }
    let reuses = frontend_of(&s).stats.prepared_reuses;
    draw(&mut s, targets[0]);
    assert!(frontend_of(&s).stats.prepared_reuses > reuses, "the same stage again");
    assert_eq!(px(&s.rig.pixels(&targets[0]), 64, 10, 10), [200, 0, 0, 255]);
    // The command processor writes elsewhere.
    assert!(s.rig.buffers.fill(&mut s.rig.sched, labels, 16, 7).unwrap());
    let reuses = frontend_of(&s).stats.prepared_reuses;
    draw(&mut s, targets[0]);
    if crate::pipeline::frontend::prepare_pages_enabled() {
        assert!(frontend_of(&s).stats.prepared_reuses > reuses, "a GPU write to another page");
    }
    // The CPU points the table at the blue texture (no synchronisation
    // point: the page is watched, the write faults).
    let words = asm::bytes(&texture_sharp(blue));
    // SAFETY: the table's bytes in the guest view.
    unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), table as *mut u8, words.len()) };
    draw(&mut s, targets[1]);
    assert_eq!(px(&s.rig.pixels(&targets[1]), 64, 10, 10), [0, 0, 255, 255], "after the CPU rewrote the table");
    draw(&mut s, targets[1]);
    // The command processor points it back at the red texture.
    assert!(s.rig.buffers.write(&mut s.rig.sched, table, &asm::bytes(&texture_sharp(s.tex))).unwrap());
    draw(&mut s, targets[2]);
    assert_eq!(px(&s.rig.pixels(&targets[2]), 64, 10, 10), [200, 0, 0, 255], "after the GPU rewrote the table");
}

/// A small constant buffer the CPU rewrites is bound from a copy of its
/// guest bytes: the copy is reused while its pages are unchanged (GPU
/// writes elsewhere do not matter), made again after a CPU write and a
/// synchronisation point, and not used once the GPU writes the range.
#[test]
#[ignore]
fn memo_stream_copy_follows_its_pages() {
    let mut g = Guest::new(16 << 20);
    let mut s = scene(&mut g);
    // Vertex program (merged: user data from s8): a load through the V#
    // in s[8:11].
    let load = [0xE000_0000 | (0x0C << 18) | (1 << 12), (1 << 8) | (2 << 16) | (0x80 << 24)];
    let vs = g.program(&[load[0], load[1], 0xBF8C_3F70, asm::S_ENDPGM]);
    let cb = g.zeroed(4096);
    let labels = g.zeroed(4096);
    let target = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
    let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 2 }).with(s.solid, TestProgram::SolidPixel));
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, -1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, -1.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0]));
    s.rig = Rig::new(&g, compiler, Some(vec4_attrs(&[(verts, 96, 32, false)], &[(0, 0, 0), (1, 0, 16)])), None);
    let v = [cb as u32, (cb >> 32) as u32 & 0xFFFF, 256, (3 << 28) | (20 << 12) | 0xFAC];
    let draw = |s: &mut Scene| {
        let d = draw_into(target, stage(vs, &v), stage(s.solid, &color_words([1.0; 4])), primitive::TRI_LIST, auto(3, 1), None);
        assert_eq!(s.rig.draw(&d), DrawOutcome::Recorded);
    };
    let counts = |s: &Scene| (s.rig.provider.stats.stream_copies, s.rig.provider.stats.stream_memo_hits);
    draw(&mut s);
    let (copies, hits) = counts(&s);
    assert!(copies >= 1, "streamed");
    // Buffer bindings are not memoised across a content change: the copy is
    // looked up again.
    let tick = s.rig.sched.current_tick();
    assert!(s.rig.buffers.fill(&mut s.rig.sched, labels, 16, 1).unwrap());
    draw(&mut s);
    // Copies belong to their tick: a submit in between (an early submit
    // of a slow, loaded test run) makes them again.
    if super::super::provider::stream_pages_enabled() && s.rig.sched.current_tick() == tick {
        // (The vertex buffer is streamed as well.)
        assert_eq!(counts(&s).0, copies, "a GPU write elsewhere: the copies are reused");
        assert!(counts(&s).1 > hits);
    }
    let (copies, _) = counts(&s);
    // SAFETY: the constant buffer's bytes (CPU-dirty, unwatched).
    unsafe { (cb as *mut u32).write_volatile(0x1234) };
    crate::memory::note_sync_point();
    draw(&mut s);
    assert!(counts(&s).0 > copies, "copied again after a CPU write and a synchronisation point");
    // The GPU writes the range: bound from its mirror, not from a copy.
    assert!(s.rig.buffers.fill(&mut s.rig.sched, cb, 16, 2).unwrap());
    let before = counts(&s);
    draw(&mut s);
    if super::super::provider::stream_pages_enabled() {
        assert_eq!(counts(&s).0, before.0, "no copy of GPU-written bytes");
        assert_eq!(counts(&s).1, before.1 + 1, "only the vertex buffer's copy is reused");
    } else {
        // Any GPU write drops every copy: the vertex buffer is copied again.
        assert_eq!(counts(&s).0, before.0 + 1, "no copy of GPU-written bytes");
    }
}

/// The buffers a draw binds are the ranges its reservation resolved: a
/// small CPU-written buffer from a copy (also when a second V# names the
/// same range), a null V# as the null buffer, a large buffer from its
/// mirror.
#[test]
#[ignore]
fn memo_resolved_ranges_bind_copies_mirrors_and_null_descriptors() {
    let mut g = Guest::new(16 << 20);
    let mut s = scene(&mut g);
    // Loads through the V#s in s[8:11], s[12:15], s[16:19] and s[20:23].
    let mut code = Vec::new();
    for (k, sgpr) in [8u32, 12, 16, 20].into_iter().enumerate() {
        code.extend_from_slice(&[0xE000_0000 | (0x0C << 18) | (1 << 12), ((k as u32 + 1) << 8) | ((sgpr / 4) << 16) | (0x80 << 24)]);
    }
    code.extend_from_slice(&[0xBF8C_3F70, asm::S_ENDPGM]);
    let vs = g.program(&code);
    let cb = g.zeroed(4096);
    let big = g.zeroed(64 << 10);
    let target = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
    let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 2 }).with(s.solid, TestProgram::SolidPixel));
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, -1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, -1.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0]));
    s.rig = Rig::new(&g, compiler, Some(vec4_attrs(&[(verts, 96, 32, false)], &[(0, 0, 0), (1, 0, 16)])), None);
    let v = |a: u64, n: u32| [a as u32, (a >> 32) as u32 & 0xFFFF, n, (3 << 28) | (20 << 12) | 0xFAC];
    let mut user = Vec::new();
    for words in [v(cb, 256), v(cb, 256), v(0, 0), v(big, 64 << 10)] {
        user.extend_from_slice(&words);
    }
    let d = draw_into(target, stage(vs, &user), stage(s.solid, &color_words([0.0, 1.0, 0.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
    assert_eq!(s.rig.draw(&d), DrawOutcome::Recorded);
    assert_eq!(px(&s.rig.pixels(&target), 64, 10, 10), [0, 255, 0, 255]);
    assert!(s.rig.provider.stats.stream_copies >= 1, "the small buffer from a copy");
    assert!(s.rig.buffers.find_containing(big, 64 << 10).is_some(), "the large buffer from its mirror");
    assert!(s.rig.buffers.find_containing(cb, 256).is_none(), "no mirror of the copied buffer");
    assert_eq!(s.rig.draw(&d), DrawOutcome::Recorded);
}

/// Draws into the same target with and without blending take recipes of
/// their own (the blend state is part of the key) and keep both: with
/// draws alternating between them, only the first of each is a miss.
#[test]
#[ignore]
fn memo_recipes_keep_targets_that_differ_only_in_state() {
    let mut g = Guest::new(16 << 20);
    let mut s = scene(&mut g);
    let plain = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
    let additive = ColorTarget { blend: crate::render::BlendState::decode(1 | (1 << 8) | (1 << 30)), ..plain };
    for round in 0..3 {
        s.fill(plain, [1.0, 0.0, 0.0, 1.0]);
        s.fill(additive, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(px(&s.rig.pixels(&plain), 64, 10, 10), [255, 255, 0, 255], "round {round}: red, then green added");
    }
    if super::super::provider::compact_recipe_key_enabled() {
        assert_eq!(s.rig.provider.stats.recipe_misses, 2, "one recipe per blend state, both kept");
        assert_eq!(s.rig.provider.stats.recipe_hits, 4);
    }
}

/// A texture seen through many swizzles (a specialization and a
/// materialisation template each, more than the cache keeps): the first
/// swizzle bound again gets its earlier binding layout back, so its draw
/// recipe still hits, and samples as before.
#[test]
#[ignore]
fn memo_layouts_survive_dropped_templates() {
    let mut g = Guest::new(16 << 20);
    let mut s = scene(&mut g);
    let target = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
    let selects = [4u32, 5, 6, 7, 0, 1];
    let swizzles: Vec<u32> = (0..20).map(|i| selects[i / 6 % 6] | selects[i % 6] << 3 | 6 << 6 | 7 << 9).collect();
    let draw = |s: &mut Scene, swizzle: u32| {
        let mut user = texture_sharp(s.tex).to_vec();
        user[3] = (user[3] & !0xFFF) | swizzle;
        user.extend_from_slice(&[2 | (2 << 3) | (2 << 6), 0, 0, 0]);
        let d = draw_into(target, stage(s.vs, &[]), stage(s.textured, &user), primitive::TRI_LIST, auto(3, 1), None);
        assert_eq!(s.rig.draw(&d), DrawOutcome::Recorded);
    };
    for &sw in &swizzles {
        draw(&mut s, sw);
    }
    let layouts = s.rig.provider.stats.recipe_hits;
    draw(&mut s, swizzles[0]);
    assert_eq!(px(&s.rig.pixels(&target), 64, 10, 10), [200, 200, 0, 255], "the first swizzle (red, red, blue, alpha) again");
    if crate::pipeline::frontend::layout_intern_enabled() {
        assert_eq!(s.rig.provider.stats.recipe_hits, layouts + 1, "the earlier layout, and with it the recipe");
    }
    // A swizzle moving red into green.
    draw(&mut s, 0 | 4 << 3 | 6 << 6 | 7 << 9);
    assert_eq!(px(&s.rig.pixels(&target), 64, 10, 10), [0, 200, 0, 255]);
}

/// A buffer binding whose words changed (a constant buffer moved) binds
/// its unchanged buffers from the previous draw's results, until buffer
/// contents may have changed (a GPU write): then every buffer is obtained
/// again. The draws render the same either way.
#[test]
#[ignore]
fn memo_unchanged_buffers_of_a_changed_binding_are_reused() {
    let mut g = Guest::new(16 << 20);
    let mut s = scene(&mut g);
    // Loads through the V#s in s[8:11] (a large buffer: its mirror) and
    // s[12:15] (a small CPU-written one: a copy).
    let mut code = Vec::new();
    for (k, sgpr) in [8u32, 12].into_iter().enumerate() {
        code.extend_from_slice(&[0xE000_0000 | (0x0C << 18) | (1 << 12), ((k as u32 + 1) << 8) | ((sgpr / 4) << 16) | (0x80 << 24)]);
    }
    code.extend_from_slice(&[0xBF8C_3F70, asm::S_ENDPGM]);
    let vs = g.program(&code);
    let big = g.zeroed(64 << 10);
    let cbs = [g.zeroed(4096), g.zeroed(4096), g.zeroed(4096)];
    let other = g.zeroed(4096);
    let target = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
    let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 2 }).with(s.solid, TestProgram::SolidPixel));
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, -1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, -1.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0]));
    s.rig = Rig::new(&g, compiler, Some(vec4_attrs(&[(verts, 96, 32, false)], &[(0, 0, 0), (1, 0, 16)])), None);
    let v = |a: u64, n: u32| [a as u32, (a >> 32) as u32 & 0xFFFF, n, (3 << 28) | (20 << 12) | 0xFAC];
    let draw = |s: &mut Scene, cb: u64, c: [f32; 4]| {
        let mut user = v(big, 64 << 10).to_vec();
        user.extend_from_slice(&v(cb, 256));
        let d = draw_into(target, stage(vs, &user), stage(s.solid, &color_words(c)), primitive::TRI_LIST, auto(3, 1), None);
        assert_eq!(s.rig.draw(&d), DrawOutcome::Recorded);
    };
    let counts = |s: &Scene| (s.rig.provider.stats.resource_reuses, s.rig.provider.stats.resource_binds);
    draw(&mut s, cbs[0], [1.0, 0.0, 0.0, 1.0]);
    let (reuses, binds) = counts(&s);
    let tick = s.rig.sched.current_tick();
    // The constant buffer moved: the large buffer's binding is reused
    // (within the tick: buffer bindings belong to it).
    draw(&mut s, cbs[1], [0.0, 1.0, 0.0, 1.0]);
    if super::super::provider::resource_memo_enabled() && s.rig.sched.current_tick() == tick {
        assert!(counts(&s).0 > reuses, "the unchanged buffer reused");
        assert_eq!(counts(&s).1, binds, "nothing obtained again");
    }
    assert_eq!(px(&s.rig.pixels(&target), 64, 10, 10), [0, 255, 0, 255]);
    // The GPU writes elsewhere: buffer contents may have changed, every
    // buffer is obtained again.
    assert!(s.rig.buffers.fill(&mut s.rig.sched, other, 16, 1).unwrap());
    let (reuses, binds) = counts(&s);
    draw(&mut s, cbs[2], [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(px(&s.rig.pixels(&target), 64, 10, 10), [0, 0, 255, 255]);
    assert_eq!(counts(&s).0, reuses, "no reuse across a content change");
    if super::super::provider::resource_memo_enabled() {
        assert!(counts(&s).1 > binds, "the large buffer obtained again");
    }
}

pub(super) struct TrackerFaults(pub(super) Arc<crate::memory::tracker::GuestPageTracker>);
impl crate::memory::BufferStore for TrackerFaults {
    fn mark_cpu_write(&self, addr: u64, size: u64) -> bool {
        matches!(self.0.invalidate_region(addr, size, true), Some((true, false)))
    }
    fn download_to_cpu(&self, _: u64, _: u64) -> bool {
        false
    }
}

pub(super) struct TableFaults(pub(super) Arc<crate::images::WatchTable>);
impl crate::memory::ImageStore for TableFaults {
    fn mark_cpu_write(&self, addr: u64, size: u64) -> bool {
        self.0.mark_cpu_write(addr, size, true)
    }
    fn mapped(&self, addr: u64, size: u64) {
        self.0.mark_mapped(addr, size);
    }
}
struct NoBufferFaults;
impl crate::memory::BufferStore for NoBufferFaults {
    fn mark_cpu_write(&self, _: u64, _: u64) -> bool {
        false
    }
    fn download_to_cpu(&self, _: u64, _: u64) -> bool {
        false
    }
}
pub(super) struct NoUnmapping;
impl crate::memory::UnmapListener for NoUnmapping {
    fn unmapped(&self, _: u64, _: u64) {}
}
