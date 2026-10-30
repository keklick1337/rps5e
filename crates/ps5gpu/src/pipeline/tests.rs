// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The pipeline provider on the real device: guest programs in guest
//! memory, the test compiler, draws and dispatches through
//! `PipelineProvider`, pixels and buffers read back.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored pipeline` (add
//! `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` for validation).

use super::testing::{asm, TestCompiler, TestProgram};
use super::*;
use crate::backend::{Backend, DrawContext, DrawOutcome, DrawTargets, PipelineProvider};
use crate::buffers::BufferCache;
use crate::device::{Buffer, MemoryUse};
use crate::images::{color_key, depth_key, ImageCache};
use crate::memory::tracker::GuestPageTracker;
use crate::render::formats::render_target_format;
use crate::render::{
    color_mode, primitive, BlendState, ColorTarget, DepthStencilState, DepthTarget, DispatchPacket, DrawKind, DrawPacket, GraphicsShaders, IndexBuffer,
    IndexType, RasterState, Rect, ResolvedDispatch, ResolvedDraw, ShaderStage as Stage, Viewport,
};
use crate::scheduler::Scheduler;
use crate::testutil::{guest_mapping, test_memory};
use ash::vk;

const CHUNK: u64 = 64 << 10;

/// Guest memory with a bump allocator (every allocation starts a fresh
/// 64 KiB chunk, so writing it never touches pages the GPU watches).
struct Guest {
    mem: &'static crate::memory::GuestGpuMemory,
    next: u64,
    end: u64,
}

impl Guest {
    fn new(len: usize) -> Guest {
        let mem = test_memory();
        let (view, _) = guest_mapping(mem, len);
        Guest { mem, next: view, end: view + len as u64 }
    }

    fn alloc(&mut self, bytes: &[u8]) -> u64 {
        let a = self.next;
        let size = (bytes.len() as u64).max(1).div_ceil(CHUNK) * CHUNK;
        assert!(a + size <= self.end, "guest memory exhausted");
        self.next += size;
        // SAFETY: a fresh, unwatched chunk of the guest view.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), a as *mut u8, bytes.len()) };
        a
    }

    fn zeroed(&mut self, len: usize) -> u64 {
        self.alloc(&vec![0u8; len])
    }

    fn program(&mut self, words: &[u32]) -> u64 {
        self.alloc(&asm::bytes(words))
    }
}

fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// A fixed vertex input for every draw.
struct FixedInputs(VertexInputs);

impl VertexInputResolver for FixedInputs {
    fn resolve(&mut self, _: &ResolvedDraw, _: &frontend::PreparedStage) -> Option<VertexInputs> {
        Some(self.0.clone())
    }
}

fn vec4_attrs(buffers: &[(u64, u64, u32, bool)], attrs: &[(u32, u32, u32)]) -> VertexInputs {
    VertexInputs {
        buffers: buffers.iter().map(|&(address, size, stride, per_instance)| VertexBufferSource { address, size, stride, per_instance }).collect(),
        attributes: attrs
            .iter()
            .map(|&(location, binding, offset)| VertexAttribute { location, binding, format: vk::Format::R32G32B32A32_SFLOAT, offset })
            .collect(),
    }
}

fn stage(address: u64, user: &[u32]) -> Stage {
    let mut user_data = [0u32; 32];
    user_data[..user.len()].copy_from_slice(user);
    Stage { address, rsrc1: 0, rsrc2: (user.len() as u32) << 1, user_data }
}

fn unbound() -> Stage {
    Stage { address: 0, rsrc1: 0, rsrc2: 0, user_data: [0; 32] }
}

/// A linear RGBA8 color target of `w` x `h` at `base`.
fn rgba8_target(base: u64, w: u32, h: u32) -> ColorTarget {
    ColorTarget {
        slot: 0,
        base,
        format: render_target_format(10, 0, 0).unwrap(),
        layout: 10,
        number_type: 0,
        order: 0,
        width: w,
        height: h,
        pitch: w,
        swizzle: crate::tiling::SWIZZLE_LINEAR,
        samples: 1,
        mip_level: 0,
        mip_levels: 1,
        base_layer: 0,
        layers: 1,
        slice_size: (w * h * 4) as u64,
        write_mask: 0xF,
        fast_clear: false,
        dcc_enabled: false,
        dcc_address: 0,
        cmask_address: 0,
        fmask_address: 0,
        clear_word0: 0,
        clear_word1: 0,
        dcc_control: 0,
        blend: BlendState::default(),
        info: 0,
        attrib: 0,
        attrib2: 0,
        attrib3: 0,
        view: 0,
    }
}

/// A multisampled target with FMASK compression keeps its fast-clear state
/// in CMASK like a single-sampled one (a zero CMASK is its register clear);
/// DCC, VRS hints in CMASK and mip chains still have none.
#[test]
fn msaa_targets_keep_fast_clears_in_cmask() {
    let mut t = rgba8_target(0x1_0000_0000, 512, 256);
    t.fast_clear = true;
    t.cmask_address = 0x1_0080_0000;
    assert_eq!(t.cmask_slice_size(), 0x1000);
    t.samples = 4;
    t.info |= 1 << 14;
    assert_eq!(t.cmask_slice_size(), 0x1000, "4x MSAA with FMASK compression");
    assert_eq!(t.clear_metadata(), Some((0x1_0080_0000, 0x1000, true)));
    t.attrib3 |= 1 << 31;
    assert_eq!(t.cmask_slice_size(), 0, "VRS hints in CMASK");
    t.attrib3 = 0;
    t.mip_levels = 2;
    assert_eq!(t.cmask_slice_size(), 0, "mip chain");
}

/// A draw of `vs` / `ps` into `target` (viewport covering it, DX clip
/// space, no culling).
fn draw_into(target: ColorTarget, vs: Stage, ps: Stage, prim: u32, packet: DrawPacket, index: Option<IndexBuffer>) -> ResolvedDraw {
    let (w, h) = (target.width as f32, target.height as f32);
    let mut viewports = [Viewport::default(); 16];
    viewports[0] = Viewport {
        x: 0.0,
        y: 0.0,
        width: w,
        height: h,
        min_depth: 0.0,
        max_depth: 1.0,
        scale: [w / 2.0, h / 2.0, 1.0],
        offset: [w / 2.0, h / 2.0, 0.0],
        z_min: 0.0,
        z_max: 1.0,
    };
    ResolvedDraw {
        packet,
        index,
        primitive_type: prim,
        topology: crate::render::topology(prim),
        primitive_restart: false,
        restart_index: 0xFFFF_FFFF,
        index_offset: 0,
        color_targets: smallvec::smallvec![target],
        resolve_targets: None,
        depth: None,
        depth_state: DepthStencilState::default(),
        color_mode: color_mode::NORMAL,
        rop3: 0xCC,
        target_mask: 0xF,
        shader_mask: 0xF,
        blend_constants: [0.0; 4],
        viewports,
        scissor: Rect { x0: 0, y0: 0, x1: target.width as i32, y1: target.height as i32 },
        scissors: [Rect { x0: 0, y0: 0, x1: target.width as i32, y1: target.height as i32 }; 16],
        raster: RasterState { vte_cntl: 0x43F, clip_cntl: 1 << 19, dx_clip_space: true, ..RasterState::default() },
        shaders: GraphicsShaders { ps, gs: vs, es: unbound(), vs: unbound(), hs: unbound(), ls: unbound(), stages: 0, geometry: Default::default() },
        ps_input_cntl: [0; 32],
        ps_input_ena: 0,
        ps_input_addr: 0,
        ps_in_control: 0,
        shader_z_format: 0,
        shader_col_format: 0x4,
        shader_pos_format: 0,
        aa_config: 0,
        db_shader_control: 0,
        alpha_to_mask: 0,
        front_face_all_bits: false,
        msaa_enable: true,
        ps_iter_sample: false,
        aa_mask: [u32::MAX; 2],
    }
}

fn auto(count: u32, instances: u32) -> DrawPacket {
    DrawPacket { kind: DrawKind::Auto, count, instance_count: instances, ..DrawPacket::default() }
}

struct Rig {
    dev: Arc<GpuDevice>,
    sched: Scheduler,
    buffers: BufferCache,
    images: ImageCache,
    provider: ShaderPipelines,
}

impl Rig {
    fn new(g: &Guest, compiler: Arc<TestCompiler>, inputs: Option<VertexInputs>, cache: Option<std::path::PathBuf>) -> Rig {
        Rig::with_compiler(g, |_| compiler, inputs, cache)
    }

    /// A rig whose compiler is made for the created device.
    fn with_compiler(
        g: &Guest,
        compiler: impl FnOnce(&GpuDevice) -> Arc<dyn ShaderCompiler>,
        inputs: Option<VertexInputs>,
        cache: Option<std::path::PathBuf>,
    ) -> Rig {
        let dev = GpuDevice::create(false).expect("device");
        let sched = Scheduler::new(&dev).unwrap();
        let tracker = Arc::new(GuestPageTracker::new(g.mem.pages.clone()));
        let buffers = BufferCache::new(&dev, g.mem, tracker).unwrap();
        let images = ImageCache::new(&dev, g.mem).unwrap();
        let compiler = compiler(&dev);
        let mut provider = ShaderPipelines::new(&dev, g.mem, Some(compiler), cache).unwrap();
        // Deterministic draws: compile on the calling thread.
        provider.set_async_compile(false);
        provider.set_vertex_input_resolver(inputs.map(|i| Box::new(FixedInputs(i)) as Box<dyn VertexInputResolver>));
        Rig { dev, sched, buffers, images, provider }
    }

    /// Bind the draw's targets (as the back-end does) and draw.
    fn draw(&mut self, d: &ResolvedDraw) -> DrawOutcome {
        let mut targets = DrawTargets { colors: Default::default(), depth: None };
        for t in &d.color_targets {
            let id = self.images.find_or_create(&mut self.sched, &mut self.buffers, &color_key(t)).unwrap();
            self.images.prepare(&mut self.sched, &mut self.buffers, id, false);
            targets.colors.push((*t, id));
        }
        if let Some(z) = d.depth {
            let id = self.images.find_or_create(&mut self.sched, &mut self.buffers, &depth_key(&z)).unwrap();
            self.images.prepare(&mut self.sched, &mut self.buffers, id, false);
            targets.depth = Some((z, id));
        }
        let mut ctx = DrawContext { dev: &self.dev, sched: &mut self.sched, buffers: (&mut self.buffers).into(), images: (&mut self.images).into(), targets: &targets };
        let o = self.provider.draw(&mut ctx, d);
        if o == DrawOutcome::Recorded {
            let tick = self.sched.current_tick();
            for &(_, id) in &targets.colors {
                self.images.mark_gpu_written(id, tick);
            }
        }
        o
    }

    fn dispatch(&mut self, d: &ResolvedDispatch) -> DrawOutcome {
        let targets = DrawTargets { colors: Default::default(), depth: None };
        let mut ctx = DrawContext { dev: &self.dev, sched: &mut self.sched, buffers: (&mut self.buffers).into(), images: (&mut self.images).into(), targets: &targets };
        self.provider.dispatch(&mut ctx, d)
    }

    /// RGBA8 pixels of a color target's image.
    fn pixels(&mut self, t: &ColorTarget) -> Vec<u8> {
        let id = self.images.find_or_create(&mut self.sched, &mut self.buffers, &color_key(t)).unwrap();
        self.images.transit(&mut self.sched, id, crate::images::TRANSFER_SRC, None);
        read_image(&self.dev, &mut self.sched, self.images.get(id).unwrap().image.image, t.width, t.height)
    }
}

impl Rig {
    /// RGBA8 pixels of a multisampled color target, resolved (the average
    /// of each pixel's samples).
    fn resolved_pixels(&mut self, t: &ColorTarget) -> Vec<u8> {
        let id = self.images.find_or_create(&mut self.sched, &mut self.buffers, &color_key(t)).unwrap();
        self.images.transit(&mut self.sched, id, crate::images::TRANSFER_SRC, None);
        let src = self.images.get(id).unwrap().image.image;
        let usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let extent = vk::Extent3D { width: t.width, height: t.height, depth: 1 };
        let single = crate::device::Image::new(&self.dev, vk::Format::R8G8B8A8_UNORM, extent, 1, 1, vk::SampleCountFlags::TYPE_1, usage, vk::ImageCreateFlags::empty()).unwrap();
        let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
        let barrier = |old, new, src_access, dst_access| {
            vk::ImageMemoryBarrier::default()
                .old_layout(old)
                .new_layout(new)
                .src_access_mask(src_access)
                .dst_access_mask(dst_access)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(single.image)
                .subresource_range(range)
        };
        self.sched.full_barrier();
        let cb = self.sched.cmd();
        let d = &self.dev.device;
        // SAFETY: recording buffer; the target is in TRANSFER_SRC layout,
        // the resolve image is moved to TRANSFER_DST and then TRANSFER_SRC.
        unsafe {
            let to_dst = barrier(vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE);
            d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_dst]);
            let region = vk::ImageResolve::default().src_subresource(layers).dst_subresource(layers).extent(extent);
            d.cmd_resolve_image(cb, src, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, single.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
            let to_src = barrier(vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::TRANSFER_READ);
            d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_src]);
        }
        read_image(&self.dev, &mut self.sched, single.image, t.width, t.height)
    }
}

fn read_image(dev: &Arc<GpuDevice>, sched: &mut Scheduler, image: vk::Image, w: u32, h: u32) -> Vec<u8> {
    let size = (w * h * 4) as u64;
    let out = Buffer::new(dev, size, vk::BufferUsageFlags::TRANSFER_DST, MemoryUse::Download).unwrap();
    sched.full_barrier();
    let region = vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
        .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
    let cb = sched.cmd();
    // SAFETY: recording buffer; image in TRANSFER_SRC layout.
    unsafe { dev.device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, out.buffer, &[region]) };
    sched.flush_and_wait();
    out.invalidate(0, size);
    out.bytes_mut()[..size as usize].to_vec()
}

fn px(p: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
    let o = ((y * w + x) * 4) as usize;
    [p[o], p[o + 1], p[o + 2], p[o + 3]]
}

/// Solid pixel shader program: v0..v3 = s0..s3.
fn solid_program(g: &mut Guest) -> u64 {
    g.program(&[asm::v_mov_from_sgpr(0, 0), asm::v_mov_from_sgpr(1, 1), asm::v_mov_from_sgpr(2, 2), asm::v_mov_from_sgpr(3, 3), asm::S_ENDPGM])
}

fn color_words(c: [f32; 4]) -> [u32; 4] {
    c.map(f32::to_bits)
}

fn temp_cache(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("rps5e-pipeline-test-{}-{name}.bin", std::process::id()))
}

#[test]
#[ignore]
fn pipeline_compiles_on_workers_and_skips_draws_meanwhile() {
    let mut g = Guest::new(16 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = solid_program(&mut g);
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0]));
    let target = rgba8_target(g.zeroed(32 * 32 * 4), 32, 32);
    let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
    let inputs = vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)]);
    let mut rig = Rig::new(&g, compiler, Some(inputs), None);
    rig.provider.set_async_compile(true);
    rig.provider.set_compile_wait(std::time::Duration::ZERO);
    let d = draw_into(target, stage(vs, &[]), stage(ps, &color_words([0.0, 1.0, 0.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
    // With no wait budget the first draw only queues its modules: it
    // writes nothing but its attachment, so it is skipped while they
    // compile.
    assert_eq!(rig.draw(&d), DrawOutcome::Skipped);
    assert!(rig.provider.stats.compiling_skips >= 1);
    let mut tries = 0;
    while rig.draw(&d) != DrawOutcome::Recorded {
        tries += 1;
        assert!(tries < 2000, "the pipeline never became ready");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let p = rig.pixels(&target);
    assert_eq!(px(&p, 32, 31, 31), [0, 255, 0, 255], "drawn once compiled");
}

#[test]
#[ignore]
fn pipeline_compiled_on_workers_is_waited_for_by_default() {
    let mut g = Guest::new(16 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = solid_program(&mut g);
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0]));
    let target = rgba8_target(g.zeroed(32 * 32 * 4), 32, 32);
    let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
    let inputs = vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)]);
    let mut rig = Rig::new(&g, compiler, Some(inputs), None);
    rig.provider.set_async_compile(true);
    rig.provider.set_compile_wait(std::time::Duration::MAX);
    let d = draw_into(target, stage(vs, &[]), stage(ps, &color_words([0.0, 1.0, 0.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
    // A skipped pass would leave the target without this frame's result.
    assert_eq!(rig.draw(&d), DrawOutcome::Recorded);
    assert_eq!(rig.provider.stats.compiling_skips, 0);
    let p = rig.pixels(&target);
    assert_eq!(px(&p, 32, 31, 31), [0, 255, 0, 255], "drawn by the first draw");
}

#[test]
#[ignore]
fn fast_pipeline_is_used_then_replaced_by_its_optimised_build() {
    let mut g = Guest::new(16 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = solid_program(&mut g);
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0]));
    let target = rgba8_target(g.zeroed(32 * 32 * 4), 32, 32);
    let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
    let inputs = vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)]);
    let mut rig = Rig::new(&g, compiler, Some(inputs), None);
    if !rig.dev.caps.pipeline_creation_cache_control {
        eprintln!("skipped: no pipelineCreationCacheControl");
        return;
    }
    rig.provider.set_async_compile(true);
    rig.provider.set_fast_pipelines(true);
    rig.provider.set_compile_wait(std::time::Duration::MAX);
    let d = draw_into(target, stage(vs, &[]), stage(ps, &color_words([0.0, 1.0, 0.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
    assert_eq!(rig.draw(&d), DrawOutcome::Recorded);
    assert_eq!(px(&rig.pixels(&target), 32, 31, 31), [0, 255, 0, 255], "drawn with the first-tier pipeline");
    let t = rig.provider.compile_times();
    // The driver may already hold the pipeline (its own disk cache): then
    // there is nothing to replace.
    assert_eq!(t.fast.count + t.probe_hits, 1, "{t:?}");
    rig.provider.settle_rebuilds();
    let t = rig.provider.compile_times();
    assert_eq!(t.swaps, t.fast.count, "the optimised build replaced the fast one: {t:?}");
    eprintln!("fast {} probe hits {} swaps {}", t.fast.count, t.probe_hits, t.swaps);
    // The next draw records with the optimised pipeline (recipes started
    // over; the fast one is destroyed once the GPU is past it).
    let d2 = draw_into(target, stage(vs, &[]), stage(ps, &color_words([1.0, 0.0, 0.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
    assert_eq!(rig.draw(&d2), DrawOutcome::Recorded);
    assert_eq!(px(&rig.pixels(&target), 32, 31, 31), [255, 0, 0, 255], "drawn with the optimised pipeline");
    assert_eq!(rig.provider.stats().graphics_pipelines, 1);
}

#[test]
#[ignore]
fn pipeline_triangle_and_cache_hits() {
    let mut g = Guest::new(16 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = solid_program(&mut g);
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0]));
    let target = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
    let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
    let path = temp_cache("triangle");
    let _ = std::fs::remove_file(&path);
    let inputs = vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)]);
    let mut rig = Rig::new(&g, compiler.clone(), Some(inputs.clone()), Some(path.clone()));
    let d = draw_into(target, stage(vs, &[]), stage(ps, &color_words([1.0, 0.6, 0.2, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
    assert_eq!(rig.draw(&d), DrawOutcome::Recorded);
    let p = rig.pixels(&target);
    assert_eq!(px(&p, 64, 8, 8), [255, 153, 51, 255], "inside the triangle");
    assert_eq!(px(&p, 64, 60, 60), [0, 0, 0, 0], "outside the triangle");
    // Same state again: every cache hits.
    let d2 = draw_into(target, stage(vs, &[]), stage(ps, &color_words([0.0, 1.0, 0.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
    assert_eq!(rig.draw(&d2), DrawOutcome::Recorded);
    let s = rig.provider.stats();
    assert_eq!(s.graphics_pipelines, 1);
    assert!(s.graphics_hits >= 1);
    assert_eq!(compiler.compiled(), 2, "one module per stage");
    let p = rig.pixels(&target);
    assert_eq!(px(&p, 64, 8, 8), [0, 255, 0, 255], "new user data, same pipeline");
    // The on-disk cache is written on drop and loaded by the next provider.
    drop(rig);
    assert!(std::fs::metadata(&path).map(|m| m.len() > 0).unwrap_or(false), "pipeline cache saved");
    let rig = Rig::new(&g, compiler, Some(inputs), Some(path.clone()));
    assert!(rig.provider.t.disk.loaded_bytes > 0, "pipeline cache loaded");
    drop(rig);
    let _ = std::fs::remove_file(&path);
}

#[test]
#[ignore]
fn pipeline_textured_rect_list() {
    let mut g = Guest::new(16 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let sample = asm::image_sample(0, 0, 0, 8);
    let ps = g.program(&[sample[0], sample[1], asm::S_ENDPGM]);
    // 4x4 linear RGBA8 texture (row pitch 64 texels).
    let mut tex = vec![0u8; 64 * 4 * 4];
    for y in 0..4u32 {
        for x in 0..4u32 {
            let o = ((y * 64 + x) * 4) as usize;
            tex[o..o + 4].copy_from_slice(&[(x * 60) as u8, (y * 60) as u8, 200, 255]);
        }
    }
    let tex_addr = g.alloc(&tex);
    let t_sharp = [
        (tex_addr >> 8) as u32,
        ((tex_addr >> 40) as u32 & 0xFF) | (56 << 20) | (3 << 30),
        (3 >> 2) | (3 << 14),
        0xFAC | (sharps::image_type::TYPE_2D << 28),
        0,
        0,
        0,
        0,
    ];
    // Clamp to edge, point filtering, no mips.
    let s_sharp = [2 | (2 << 3) | (2 << 6), 0, 0, 0];
    let mut user = t_sharp.to_vec();
    user.extend_from_slice(&s_sharp);
    // Three corners of a full-screen rectangle with texture coordinates.
    let verts = g.alloc(&floats(&[
        -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
        1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, //
        -1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
    ]));
    let target = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
    let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 2 }).with(ps, TestProgram::TexturedPixel));
    let inputs = vec4_attrs(&[(verts, 96, 32, false)], &[(0, 0, 0), (1, 0, 16)]);
    let mut rig = Rig::new(&g, compiler, Some(inputs), None);
    let d = draw_into(target, stage(vs, &[]), stage(ps, &user), primitive::RECT_LIST, auto(3, 1), None);
    assert_eq!(rig.draw(&d), DrawOutcome::Recorded);
    let p = rig.pixels(&target);
    for (x, y) in [(2u32, 2u32), (20, 5), (40, 40), (63, 63), (63, 0), (0, 63)] {
        let (tx, ty) = (x / 16, y / 16);
        assert_eq!(px(&p, 64, x, y), [(tx * 60) as u8, (ty * 60) as u8, 200, 255], "pixel ({x},{y})");
    }
    assert_eq!(rig.provider.stats().rect_draws, 1);
}

/// A sampler with FORCE_DEGAMMA decodes a UNORM texture as sRGB.
#[test]
#[ignore]
fn force_degamma_sampler_decodes_srgb() {
    for (degamma, expected) in [(false, 128u8), (true, 55)] {
        let mut g = Guest::new(16 << 20);
        let vs = g.program(&[asm::S_ENDPGM]);
        let sample = asm::image_sample(0, 0, 0, 8);
        let ps = g.program(&[sample[0], sample[1], asm::S_ENDPGM]);
        let tex_addr = g.alloc(&[128u8, 128, 128, 255].repeat(64 * 4));
        let t_sharp = [
            (tex_addr >> 8) as u32,
            ((tex_addr >> 40) as u32 & 0xFF) | (56 << 20) | (3 << 30),
            (3 >> 2) | (3 << 14),
            0xFAC | (sharps::image_type::TYPE_2D << 28),
            0,
            0,
            0,
            0,
        ];
        let s_sharp = [2 | (2 << 3) | (2 << 6) | (degamma as u32) << 20, 0, 0, 0];
        let mut user = t_sharp.to_vec();
        user.extend_from_slice(&s_sharp);
        let verts = g.alloc(&floats(&[
            -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
            1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, //
            -1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        ]));
        let target = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
        let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 2 }).with(ps, TestProgram::TexturedPixel));
        let inputs = vec4_attrs(&[(verts, 96, 32, false)], &[(0, 0, 0), (1, 0, 16)]);
        let mut rig = Rig::new(&g, compiler, Some(inputs), None);
        let d = draw_into(target, stage(vs, &[]), stage(ps, &user), primitive::RECT_LIST, auto(3, 1), None);
        assert_eq!(rig.draw(&d), DrawOutcome::Recorded);
        let p = rig.pixels(&target);
        let c = px(&p, 64, 20, 20);
        assert!(c[0].abs_diff(expected) <= 1 && c[3] == 255, "degamma {degamma}: {c:?}");
    }
}

/// A T# with DCC compression reads its surface through the keys: a fill
/// of a fixed clear code over the keys of a surface that is only sampled
/// clears what it reads; comp-to-single keys (0x10) give every 256-byte
/// block its first element; uncompressed keys read the bytes.
#[test]
#[ignore]
fn sampled_dcc_keys_decide_what_a_texture_reads() {
    for case in ["clear fill", "single", "uncompressed"] {
        let mut g = Guest::new(16 << 20);
        let vs = g.program(&[asm::S_ENDPGM]);
        let sample = asm::image_sample(0, 0, 0, 8);
        let ps = g.program(&[sample[0], sample[1], asm::S_ENDPGM]);
        // 64x64 linear RGBA8: one row is one 256-byte block.
        let mut tex = vec![0u8; 64 * 64 * 4];
        for y in 0..64u32 {
            for x in 0..64u32 {
                let o = ((y * 64 + x) * 4) as usize;
                tex[o..o + 4].copy_from_slice(&[(x * 4) as u8, (y * 4) as u8, 100, 255]);
            }
        }
        let tex_addr = g.alloc(&tex);
        let keys = g.alloc(&[0xFFu8; 64]);
        let t_sharp = [
            (tex_addr >> 8) as u32,
            ((tex_addr >> 40) as u32 & 0xFF) | (56 << 20) | (3 << 30),
            (63 >> 2) | (63 << 14),
            0xFAC | (sharps::image_type::TYPE_2D << 28),
            0,
            0,
            // COMPRESSION_EN, ALPHA_IS_ON_MSB, META_DATA_ADDRESS.
            1 << 21 | 1 << 22 | ((keys >> 8) as u32 & 0xFF) << 24,
            (keys >> 16) as u32,
        ];
        let s_sharp = [2 | (2 << 3) | (2 << 6), 0, 0, 0];
        let mut user = t_sharp.to_vec();
        user.extend_from_slice(&s_sharp);
        let verts = g.alloc(&floats(&[
            -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
            1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, //
            -1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        ]));
        let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 2 }).with(ps, TestProgram::TexturedPixel));
        let inputs = vec4_attrs(&[(verts, 96, 32, false)], &[(0, 0, 0), (1, 0, 16)]);
        let mut rig = Rig::new(&g, compiler, Some(inputs), None);
        // The texture is read once with uncompressed keys.
        let first = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
        assert_eq!(rig.draw(&draw_into(first, stage(vs, &[]), stage(ps, &user), primitive::RECT_LIST, auto(3, 1), None)), DrawOutcome::Recorded);
        assert_eq!(px(&rig.pixels(&first), 64, 10, 20), [40, 80, 100, 255], "{case}: stored bytes");
        match case {
            "clear fill" => {
                // A DMA fill of 0x80 (colour 1, alpha 0) over the keys.
                assert!(g.mem.write_backing(keys, &[0x80u8; 64]));
                let id = rig.images.find_existing(&crate::pipeline::sharps::texture_key(&sharps::TextureSharp::from_words(&t_sharp), &sharps::texel_format(56).unwrap()).unwrap()).unwrap();
                let generation = rig.images.image_generation(id);
                rig.images.note_fill(keys, 64, 0x8080_8080);
                // Memoised lookups of the texture see the fill.
                assert_ne!(rig.images.image_generation(id), generation);
            }
            "single" => {
                // The surface is written again with comp-to-single blocks.
                assert!(g.mem.write_backing(keys, &[0x10u8; 64]));
                rig.buffers.tracker.mark_cpu_dirty_pages(keys, 64);
                let mut other = tex.clone();
                for y in 0..64usize {
                    for x in 1..64usize {
                        other[(y * 64 + x) * 4..][..4].copy_from_slice(&[1, 2, 3, 4]);
                    }
                }
                assert!(g.mem.write_backing(tex_addr, &other));
                rig.buffers.tracker.mark_cpu_dirty_pages(tex_addr, other.len() as u64);
                rig.images.invalidate_range(tex_addr, other.len() as u64);
            }
            _ => {}
        }
        let target = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
        assert_eq!(rig.draw(&draw_into(target, stage(vs, &[]), stage(ps, &user), primitive::RECT_LIST, auto(3, 1), None)), DrawOutcome::Recorded);
        let p = rig.pixels(&target);
        let expected = match case {
            "clear fill" => [255, 255, 255, 0],
            "single" => [0, 80, 100, 255],
            _ => [40, 80, 100, 255],
        };
        assert_eq!(px(&p, 64, 10, 20), expected, "{case}");
        if case != "uncompressed" {
            let mut k = [0u8; 64];
            assert!(g.mem.read_backing(keys, &mut k));
            assert_eq!(k, [0xFF; 64], "{case}: the keys are spent");
        }
    }
}

#[test]
#[ignore]
fn pipeline_indexed_instanced_and_indirect() {
    let mut g = Guest::new(32 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = g.program(&[asm::S_ENDPGM]);
    // A quad over the left half of the target (x in [-1, 0]).
    let positions = g.alloc(&floats(&[-1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0]));
    // Per instance: offset and color.
    let instances = g.alloc(&floats(&[
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, //
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
    ]));
    let idx16 = g.alloc(&[0u16, 1, 2, 0, 2, 3].iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>());
    let idx8 = g.alloc(&[0u8, 1, 2, 0, 2, 3]);
    let quads = g.alloc(&[0u16, 1, 2, 3].iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>());
    // DrawIndirect: 3 vertices of the first triangle, 2 instances.
    let args = g.alloc(&[3u32, 2, 0, 0].iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>());
    let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::OffsetVertex).with(ps, TestProgram::VaryingPixel));
    let inputs = vec4_attrs(&[(positions, 64, 16, false), (instances, 64, 32, true)], &[(0, 0, 0), (1, 1, 0), (2, 1, 16)]);
    let mut rig = Rig::new(&g, compiler, Some(inputs), None);
    let expect_halves = |p: &[u8], what: &str| {
        assert_eq!(px(p, 64, 10, 32), [255, 0, 0, 255], "{what}: left half, instance 0");
        assert_eq!(px(p, 64, 50, 32), [0, 255, 0, 255], "{what}: right half, instance 1");
    };
    let cases: [(&str, u64, IndexType, u32, u32); 3] =
        [("16-bit", idx16, IndexType::U16, primitive::TRI_LIST, 6), ("8-bit", idx8, IndexType::U8, primitive::TRI_LIST, 6), ("quads", quads, IndexType::U16, primitive::QUAD_LIST_LEGACY, 4)];
    for (what, addr, kind, prim, count) in cases {
        let target = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
        let packet = DrawPacket { kind: DrawKind::Index, count, instance_count: 2, index_address: addr, index_type: kind as u32, ..DrawPacket::default() };
        let index = IndexBuffer { address: addr, kind, count };
        let d = draw_into(target, stage(vs, &[]), stage(ps, &[]), prim, packet, Some(index));
        assert_eq!(rig.draw(&d), DrawOutcome::Recorded, "{what}");
        let p = rig.pixels(&target);
        expect_halves(&p, what);
    }
    assert_eq!(rig.provider.stats().cpu_index_conversions, 2, "8-bit and quad indices go through the CPU");
    // Indirect: the first triangle (top-left) of both instances.
    let target = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
    let packet = DrawPacket { kind: DrawKind::Indirect, indirect_address: args, ..DrawPacket::default() };
    let d = draw_into(target, stage(vs, &[]), stage(ps, &[]), primitive::TRI_LIST, packet, None);
    assert_eq!(rig.draw(&d), DrawOutcome::Recorded);
    let p = rig.pixels(&target);
    assert_eq!(px(&p, 64, 20, 4), [255, 0, 0, 255], "indirect instance 0");
    assert_eq!(px(&p, 64, 52, 4), [0, 255, 0, 255], "indirect instance 1");
    assert_eq!(px(&p, 64, 4, 60), [0, 0, 0, 0], "outside the triangle");
    assert_eq!(rig.provider.stats().indirect_draws, 1);
}

#[test]
#[ignore]
fn pipeline_compute_dispatch_writes_guest_buffer() {
    let mut g = Guest::new(16 << 20);
    let store = asm::buffer_store_dword(1, 0, 0);
    let cs = g.program(&[asm::v_mov_from_sgpr(1, 4), store[0], store[1], asm::S_ENDPGM]);
    let region = g.zeroed(8192);
    // An address that is not storage-offset aligned: the shader applies
    // the adjustment from the push data.
    let base = region + 0x44;
    let v_sharp = [base as u32, (base >> 32) as u32, 256 * 4, (3 << 28) | (20 << 12) | 0xFAC];
    let mut user = v_sharp.to_vec();
    user.push(1000);
    let compiler = Arc::new(TestCompiler::new().with(cs, TestProgram::StoreCompute { local_size: 64 }));
    let mut rig = Rig::new(&g, compiler, None, None);
    let mut shader = stage(cs, &user);
    shader.rsrc2 = (user.len() as u32) << 1;
    let d = ResolvedDispatch {
        packet: DispatchPacket { groups: [4, 1, 1], dispatch_initiator: 1 | (1 << 15), ..DispatchPacket::default() },
        shader,
        rsrc3: 0,
        threads: [64, 1, 1],
        start: [0; 3],
    };
    assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
    assert!(rig.buffers.has_gpu_dirty(base, 1024), "the written range is GPU-dirty");
    assert!(rig.buffers.download(&mut rig.sched, region, 8192));
    let mut out = vec![0u8; 1024];
    assert!(g.mem.read_backing(base, &mut out));
    for i in 0..256usize {
        let w = u32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(w, 1000 + i as u32, "word {i}");
    }
    let mut before = [0u8; 4];
    assert!(g.mem.read_backing(base - 4, &mut before));
    assert_eq!(before, [0; 4], "nothing written before the buffer");
    // Second dispatch: pipeline cache hit.
    assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
    let s = rig.provider.stats();
    assert_eq!((s.compute_pipelines, s.compute_hits), (1, 1));
}

/// The command thread records a dispatch holding the back-end lock; a
/// dispatch whose module still compiles used to wait for it inside the
/// lock, so a guest CPU fault needing the back-end (a download of a
/// GPU-written page) waited for the compile too: seconds with a cold
/// shader cache. Here a mutex stands for the back-end lock and the compile
/// is held back by a gate that only the "faulting" thread opens, once it
/// got the lock. With the waits unlocked the dispatch gives itself up,
/// the lock is free, and the dispatch records after the compile. With
/// `RPS5E_UNLOCKED_COMPILE_WAIT=0` (the old in-lock wait) the faulting
/// thread does not get the lock while the compile runs and the test fails.
#[test]
#[ignore]
fn dispatch_waits_for_its_compile_with_the_backend_unlocked() {
    use super::testing::CompileGate;
    use std::sync::atomic::{AtomicBool, Ordering};
    let mut g = Guest::new(16 << 20);
    let store = asm::buffer_store_dword(1, 0, 0);
    let cs = g.program(&[asm::v_mov_from_sgpr(1, 4), store[0], store[1], asm::S_ENDPGM]);
    let base = g.zeroed(8192);
    let v_sharp = [base as u32, (base >> 32) as u32, 64 * 4, (3 << 28) | (20 << 12) | 0xFAC];
    let mut user = v_sharp.to_vec();
    user.push(7);
    let gate = Arc::new(CompileGate::default());
    let compiler = Arc::new(TestCompiler::new().with(cs, TestProgram::StoreCompute { local_size: 64 }).gated(gate.clone()));
    let mut rig = Rig::new(&g, compiler, None, None);
    rig.provider.set_async_compile(true);
    let mut shader = stage(cs, &user);
    shader.rsrc2 = (user.len() as u32) << 1;
    let d = ResolvedDispatch {
        packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1 | (1 << 15), ..DispatchPacket::default() },
        shader,
        rsrc3: 0,
        threads: [64, 1, 1],
        start: [0; 3],
    };
    let backend = std::sync::Mutex::new(());
    let got_lock_while_compiling = AtomicBool::new(false);
    let outcome = std::thread::scope(|s| {
        s.spawn(|| {
            // The faulting thread: give the dispatch time to start, then
            // take the lock while the compile is still held back.
            std::thread::sleep(std::time::Duration::from_millis(100));
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while std::time::Instant::now() < deadline {
                if let Ok(_guard) = backend.try_lock() {
                    got_lock_while_compiling.store(!gate.is_open(), Ordering::SeqCst);
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            // Never leave the dispatch waiting for ever.
            gate.open();
        });
        unlocked_wait::record(true, || {
            let _guard = backend.lock().unwrap();
            let o = rig.dispatch(&d);
            (o, rig.provider.take_compile_pending())
        })
    });
    assert!(got_lock_while_compiling.load(Ordering::SeqCst), "the back-end lock stayed held while the dispatch waited for its compile");
    assert_eq!(outcome, DrawOutcome::Recorded);
    assert!(rig.provider.stats().pending_tries >= 1, "the dispatch was given up while its module compiled");
    assert_eq!(rig.provider.stats().dispatches, 1, "tries given up are not counted");
    assert!(rig.buffers.download(&mut rig.sched, base, 8192));
    let mut out = vec![0u8; 256];
    assert!(g.mem.read_backing(base, &mut out));
    for i in 0..64usize {
        assert_eq!(u32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap()), 7 + i as u32, "word {i}");
    }
}

#[test]
#[ignore]
fn pipeline_through_backend_and_needs_shader_without_compiler() {
    let mut g = Guest::new(16 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = solid_program(&mut g);
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0]));
    let target = rgba8_target(g.zeroed(32 * 32 * 4), 32, 32);
    let dev = GpuDevice::create(false).expect("device");
    let mut b = Backend::new(&dev, g.mem).unwrap();
    let d = draw_into(target, stage(vs, &[]), stage(ps, &color_words([0.0, 0.0, 1.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
    // No compiler: the draw needs a shader.
    b.pipelines = Box::new(ShaderPipelines::new(&dev, g.mem, None, None).unwrap());
    b.draw(&d);
    assert_eq!(b.stats.needs_shader_draws, 1);
    let compiler: Arc<dyn ShaderCompiler> = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
    let mut p = ShaderPipelines::new(&dev, g.mem, Some(compiler), None).unwrap();
    p.set_async_compile(false);
    p.set_vertex_input_resolver(Some(Box::new(FixedInputs(vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)])))));
    b.pipelines = Box::new(p);
    b.draw(&d);
    assert_eq!(b.stats.recorded_draws, 1);
    let id = b.images.find_or_create(&mut b.sched, &mut b.buffers, &color_key(&target)).unwrap();
    assert!(b.images.get(id).unwrap().gpu_modified);
    b.images.transit(&mut b.sched, id, crate::images::TRANSFER_SRC, None);
    let image = b.images.get(id).unwrap().image.image;
    let pixels = read_image(&dev, &mut b.sched, image, 32, 32);
    assert_eq!(px(&pixels, 32, 31, 31), [0, 0, 255, 255], "full-screen triangle");
}

/// A fast clear consumed at a bind stays consumed when the GPU holds the
/// metadata (the zero fill of the CMASK ran on the GPU): once those bytes
/// reach guest memory, a later bind of the target must not find the clear
/// state again and wipe what was drawn since.
#[test]
#[ignore]
fn consumed_fast_clear_of_gpu_metadata_does_not_clear_again() {
    let mut g = Guest::new(16 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = solid_program(&mut g);
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0]));
    let cmask = g.zeroed(0x1000);
    let mut target = rgba8_target(g.zeroed(32 * 32 * 4), 32, 32);
    target.fast_clear = true;
    target.cmask_address = cmask;
    target.clear_word0 = 0xFF00_00FF;
    assert_eq!(target.cmask_slice_size(), 0x1000);
    let dev = GpuDevice::create(false).expect("device");
    let mut b = Backend::new(&dev, g.mem).unwrap();
    let compiler: Arc<dyn ShaderCompiler> = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
    let mut p = ShaderPipelines::new(&dev, g.mem, Some(compiler), None).unwrap();
    p.set_async_compile(false);
    p.set_vertex_input_resolver(Some(Box::new(FixedInputs(vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)])))));
    b.pipelines = Box::new(p);
    // The CMASK is GPU memory, zero-filled on the GPU (the clear).
    assert!(b.buffers.fill(&mut b.sched, cmask, 0x1000, 0x5555_5555).unwrap());
    assert!(b.fill_memory(cmask, 0x1000, 0));
    b.memory_filled(cmask, 0x1000, 0);
    let d = draw_into(target, stage(vs, &[]), stage(ps, &color_words([0.0, 0.0, 1.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
    b.draw(&d);
    assert_eq!((b.stats.recorded_draws, b.stats.color_clears), (1, 1), "the CMASK clear is applied at the bind");
    // The GPU's metadata bytes reach guest memory.
    assert!(b.buffers.download(&mut b.sched, cmask, 0x1000));
    let mut bytes = vec![0u8; 0x1000];
    assert!(g.mem.read_backing(cmask, &mut bytes));
    assert!(bytes.iter().all(|&x| x == 0xFF), "the consumed clear left the CMASK expanded");
    // The next bind finds no clear (it would wipe the first draw).
    b.draw(&d);
    assert_eq!((b.stats.recorded_draws, b.stats.color_clears), (2, 1), "no second clear");
    let id = b.images.find_or_create(&mut b.sched, &mut b.buffers, &color_key(&target)).unwrap();
    b.images.transit(&mut b.sched, id, crate::images::TRANSFER_SRC, None);
    let image = b.images.get(id).unwrap().image.image;
    let pixels = read_image(&dev, &mut b.sched, image, 32, 32);
    assert_eq!(px(&pixels, 32, 5, 5), [0, 0, 255, 255]);
}

/// Draws through the shader pipelines; every dispatch is reported as one
/// that overwrote `overwrites` without reading them (a metadata clear
/// kernel).
struct OverwritingDispatches {
    inner: ShaderPipelines,
    overwrites: Vec<(u64, u64)>,
    pending: Vec<(u64, u64)>,
}

impl PipelineProvider for OverwritingDispatches {
    fn draw(&mut self, ctx: &mut DrawContext, d: &ResolvedDraw) -> DrawOutcome {
        self.inner.draw(ctx, d)
    }

    fn dispatch(&mut self, _ctx: &mut DrawContext, _d: &ResolvedDispatch) -> DrawOutcome {
        self.pending = self.overwrites.clone();
        DrawOutcome::Recorded
    }

    fn take_overwrites(&mut self) -> Vec<(u64, u64)> {
        std::mem::take(&mut self.pending)
    }
}

/// A compute program that writes the DCC keys of a layered target (an
/// array, or the depth slices of a volume) decides the fast clear of each
/// bound layer on the GPU, with no read-back: layers whose keys all hold
/// the "1111" code become white, the others keep what was drawn, and the
/// keys of the cleared layers become uncompressed.
fn compute_written_dcc_keys_clear_their_layers(volume: bool) {
    const W: u32 = 32;
    const LAYERS: u32 = 4;
    const SLICE: u64 = 0x10000;
    let mut g = Guest::new(32 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = solid_program(&mut g);
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0]));
    let mut target = rgba8_target(g.zeroed((SLICE * LAYERS as u64) as usize + 0x10000), W, W);
    target.swizzle = crate::tiling::SWIZZLE_64KB_R_X;
    target.pitch = 128;
    target.slice_size = SLICE;
    target.layers = LAYERS;
    if volume {
        target.attrib3 = (2 << 24) | (LAYERS - 1);
    }
    target.dcc_enabled = true;
    // Keys start uncompressed: one 4 KiB key block per layer.
    let keys = 0x1000u64;
    target.dcc_address = g.alloc(&vec![0xFFu8; (keys * LAYERS as u64) as usize]);
    assert_eq!(target.dcc_slice_size(), keys);
    let dev = GpuDevice::create(false).expect("device");
    if dev.conditional.is_none() {
        eprintln!("no conditional rendering: skipped");
        return;
    }
    let mut b = Backend::new(&dev, g.mem).unwrap();
    let compiler: Arc<dyn ShaderCompiler> = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
    let mut p = ShaderPipelines::new(&dev, g.mem, Some(compiler), None).unwrap();
    p.set_async_compile(false);
    p.set_vertex_input_resolver(Some(Box::new(FixedInputs(vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)])))));
    let meta = target.dcc_address;
    b.pipelines = Box::new(OverwritingDispatches { inner: p, overwrites: vec![(meta, keys * LAYERS as u64)], pending: Vec::new() });
    // Blue into layer 0 (the draw renders into the first bound layer).
    let d = draw_into(target, stage(vs, &[]), stage(ps, &color_words([0.0, 0.0, 1.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
    b.draw(&d);
    assert_eq!(b.stats.recorded_draws, 1);
    // The kernel's keys, on the GPU: layers 0 and 2 uniformly "1111",
    // layer 1 mixed, layer 3 uncompressed.
    assert!(b.buffers.fill(&mut b.sched, meta, keys * 3, 0xC0C0_C0C0).unwrap());
    assert!(b.buffers.fill(&mut b.sched, meta + keys, 4, 0xC0C0_C0FF).unwrap());
    assert!(b.buffers.fill(&mut b.sched, meta + 3 * keys, keys, u32::MAX).unwrap());
    let shader = stage(0x1000, &[]);
    let dispatch = ResolvedDispatch { packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() }, shader, rsrc3: 0, threads: [64, 1, 1], start: [0; 3] };
    b.dispatch(&dispatch);
    assert_eq!((b.stats.dcc_gpu_decisions, b.stats.dcc_cpu_reads), (1, 0), "decided on the GPU");
    // Every layer (depth slice) read back.
    let id = b.images.find_or_create(&mut b.sched, &mut b.buffers, &color_key(&target)).unwrap();
    b.images.transit(&mut b.sched, id, crate::images::TRANSFER_SRC, None);
    let image = b.images.get(id).unwrap().image.image;
    let size = (W * W * 4 * LAYERS) as u64;
    let out = Buffer::new(&dev, size, vk::BufferUsageFlags::TRANSFER_DST, MemoryUse::Download).unwrap();
    b.sched.full_barrier();
    let (layer_count, depth) = if volume { (1, LAYERS) } else { (LAYERS, 1) };
    let region = vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count })
        .image_extent(vk::Extent3D { width: W, height: W, depth });
    let cb = b.sched.cmd();
    // SAFETY: recording buffer; image in TRANSFER_SRC layout.
    unsafe { dev.device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, out.buffer, &[region]) };
    b.sched.flush_and_wait();
    out.invalidate(0, size);
    let pixels = out.bytes_mut()[..size as usize].to_vec();
    let layer = |l: u32| &pixels[(l * W * W * 4) as usize..((l + 1) * W * W * 4) as usize];
    let expect = [[255u8; 4], [0, 0, 0, 0], [255; 4], [0, 0, 0, 0]];
    for (l, e) in expect.iter().enumerate() {
        assert_eq!(px(layer(l as u32), W, 5, 5), *e, "layer {l} (volume: {volume})");
    }
    // The cleared layers' keys are uncompressed; the others are as written.
    assert!(b.buffers.download(&mut b.sched, meta, keys * LAYERS as u64));
    let mut k = vec![0u8; (keys * LAYERS as u64) as usize];
    assert!(g.mem.read_backing(meta, &mut k));
    let first = |l: u64| k[(l * keys) as usize];
    assert_eq!([first(0), first(1), first(2), first(3)], [0xFF, 0xFF, 0xFF, 0xFF]);
    assert_eq!(k[(keys + 1) as usize], 0xC0, "the mixed layer keeps its keys");
    // The next bind clears nothing and reads nothing back.
    b.draw(&d);
    assert_eq!((b.stats.recorded_draws, b.stats.dcc_cpu_reads), (2, 0));
}

/// Draws through the shader pipelines; every dispatch is reported as a
/// store through an image descriptor with DCC compression on into the
/// surface whose keys are `stores`.
struct CompressedStores {
    inner: ShaderPipelines,
    stores: Vec<crate::render::StoredKeys>,
    pending: Vec<crate::render::StoredKeys>,
}

impl PipelineProvider for CompressedStores {
    fn draw(&mut self, ctx: &mut DrawContext, d: &ResolvedDraw) -> DrawOutcome {
        self.inner.draw(ctx, d)
    }

    fn dispatch(&mut self, _ctx: &mut DrawContext, _d: &ResolvedDispatch) -> DrawOutcome {
        self.pending = self.stores.clone();
        DrawOutcome::Recorded
    }

    fn take_compressed_stores(&mut self) -> Vec<crate::render::StoredKeys> {
        std::mem::take(&mut self.pending)
    }
}

/// The keys a storage descriptor updates: T# word 6 COMPRESSION_EN (bit
/// 21) with META_DATA_ADDRESS (word 6 bits 24..31, word 7, in 256-byte
/// units); every layer of a modelled surface, nothing without the bit or
/// for unmodelled (linear) surfaces.
#[test]
fn compressed_store_keys_follow_the_descriptor() {
    let mut t = rgba8_target(0x10_0000_0000, 144, 81);
    t.swizzle = crate::tiling::SWIZZLE_64KB_R_X;
    t.layers = 96;
    let key = color_key(&t);
    let mut words = [0u32; 8];
    words[6] = 0x006b_0000 | (0x12 << 24);
    words[7] = 0x0021_a669;
    let keys = super::provider::compressed_store_keys(&words, Some(&key));
    assert_eq!(keys, Some(crate::render::StoredKeys { address: 0x21_a669_1200, slice: 0x1000, layers: 96 }));
    words[6] &= !(1 << 21);
    assert_eq!(super::provider::compressed_store_keys(&words, Some(&key)), None, "COMPRESSION_EN clear");
    words[6] |= 1 << 21;
    t.swizzle = crate::tiling::SWIZZLE_LINEAR;
    assert_eq!(super::provider::compressed_store_keys(&words, Some(&color_key(&t))), None, "keys of a linear surface are not modelled");
}

/// A store through an image descriptor with DCC compression on rewrites the
/// keys of what it writes. Keys left holding a clear code from earlier
/// ("0001" in guest memory here) are spent by the store, and the next bind
/// of the surface as a colour target clears nothing: the clear would wipe
/// what the store wrote (SILENT HILL's volumetric fog volumes went black
/// for a frame). A surface no store wrote is still cleared.
#[test]
#[ignore]
fn compressed_stores_spend_stale_dcc_clear_codes() {
    const W: u32 = 32;
    let mut g = Guest::new(32 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = solid_program(&mut g);
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0]));
    let keys = 0x1000u64;
    let target = |g: &mut Guest| {
        let mut t = rgba8_target(g.zeroed(0x20000), W, W);
        t.swizzle = crate::tiling::SWIZZLE_64KB_R_X;
        t.pitch = 128;
        t.slice_size = 0x10000;
        t.dcc_enabled = true;
        t.dcc_address = g.alloc(&vec![0x40u8; keys as usize]);
        assert_eq!(t.dcc_slice_size(), keys);
        t
    };
    let (stored, untouched) = (target(&mut g), target(&mut g));
    let dev = GpuDevice::create(false).expect("device");
    let mut b = Backend::new(&dev, g.mem).unwrap();
    let compiler: Arc<dyn ShaderCompiler> = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
    let mut p = ShaderPipelines::new(&dev, g.mem, Some(compiler), None).unwrap();
    p.set_async_compile(false);
    p.set_vertex_input_resolver(Some(Box::new(FixedInputs(vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)])))));
    let meta = stored.dcc_address;
    b.pipelines = Box::new(CompressedStores { inner: p, stores: vec![crate::render::StoredKeys { address: meta, slice: keys, layers: 1 }], pending: Vec::new() });
    let dispatch = ResolvedDispatch { packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() }, shader: stage(0x1000, &[]), rsrc3: 0, threads: [64, 1, 1], start: [0; 3] };
    b.dispatch(&dispatch);
    assert_eq!(b.stats.dcc_store_spent, 1, "the store spent the clear code");
    let mut k = vec![0u8; keys as usize];
    assert!(g.mem.read_backing(meta, &mut k));
    assert!(k.iter().all(|&x| x == 0xFF), "the keys read as uncompressed");
    let blue = color_words([0.0, 0.0, 1.0, 1.0]);
    b.draw(&draw_into(stored, stage(vs, &[]), stage(ps, &blue), primitive::TRI_LIST, auto(3, 1), None));
    assert_eq!((b.stats.recorded_draws, b.stats.color_clears), (1, 0), "no clear at the bind of the stored surface");
    b.draw(&draw_into(untouched, stage(vs, &[]), stage(ps, &blue), primitive::TRI_LIST, auto(3, 1), None));
    assert_eq!((b.stats.recorded_draws, b.stats.color_clears), (2, 1), "the surface no store wrote is cleared to 0001");
}

/// A compute write of one dword at a colour target's CMASK address (an
/// atomic counter over reused memory) is no metadata clear: the word keeps
/// the value the program wrote, and nothing is decided or cleared.
#[test]
#[ignore]
fn small_compute_writes_at_a_metadata_address_are_not_clears() {
    let mut g = Guest::new(16 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = solid_program(&mut g);
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0]));
    let cmask = g.zeroed(0x1000);
    let mut target = rgba8_target(g.zeroed(32 * 32 * 4), 32, 32);
    target.fast_clear = true;
    target.cmask_address = cmask;
    target.clear_word0 = 0xFF00_00FF;
    let dev = GpuDevice::create(false).expect("device");
    if dev.conditional.is_none() {
        eprintln!("no conditional rendering: skipped");
        return;
    }
    let mut b = Backend::new(&dev, g.mem).unwrap();
    let compiler: Arc<dyn ShaderCompiler> = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
    let mut p = ShaderPipelines::new(&dev, g.mem, Some(compiler), None).unwrap();
    p.set_async_compile(false);
    p.set_vertex_input_resolver(Some(Box::new(FixedInputs(vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)])))));
    b.pipelines = Box::new(OverwritingDispatches { inner: p, overwrites: vec![(cmask, 4)], pending: Vec::new() });
    let d = draw_into(target, stage(vs, &[]), stage(ps, &color_words([0.0, 0.0, 1.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
    b.draw(&d);
    let clears = b.stats.color_clears;
    // The program's result on the GPU: the whole range reads as the CMASK clear state.
    assert!(b.buffers.fill(&mut b.sched, cmask, 0x1000, 0).unwrap());
    let shader = stage(0x1000, &[]);
    let dispatch = ResolvedDispatch { packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() }, shader, rsrc3: 0, threads: [64, 1, 1], start: [0; 3] };
    b.dispatch(&dispatch);
    assert_eq!((b.stats.dcc_gpu_decisions, b.stats.color_clears), (0, clears), "no clear decided");
    assert!(b.buffers.download(&mut b.sched, cmask, 4));
    let mut word = [0xAAu8; 4];
    assert!(g.mem.read_backing(cmask, &mut word));
    assert_eq!(word, [0; 4], "the written word is kept");
}

#[test]
#[ignore]
fn compute_written_dcc_keys_clear_array_layers_on_the_gpu() {
    compute_written_dcc_keys_clear_their_layers(false);
}

#[test]
#[ignore]
fn compute_written_dcc_keys_clear_volume_slices_on_the_gpu() {
    compute_written_dcc_keys_clear_their_layers(true);
}

/// A "single-colour" DCC clear: the SDK fills every key with 0x10 and
/// stores the clear colour at the start of each 256-byte block of the
/// surface; each block then reads as that colour. The next bind of the
/// target clears it to the colour of the block starts (not only the
/// texels the kernel stored), and the keys become uncompressed. With
/// `gpu_heads` the block starts are stored by GPU work (the layer is then
/// filled from the first one on the GPU, without a read-back).
fn single_colour_dcc_keys_clear_to_the_block_colour(gpu_heads: bool) {
    const W: u32 = 32;
    const SLICE: u64 = 0x10000;
    let mut g = Guest::new(32 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = solid_program(&mut g);
    // A small triangle in one corner: the texels checked are not drawn.
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.0, 1.0, -0.9, -1.0, 0.0, 1.0, -1.0, -0.9, 0.0, 1.0]));
    // The surface's blocks: the clear colour (green) at each block start,
    // zeros elsewhere (what the clear kernel leaves in memory).
    let green = [0x00u8, 0xFF, 0x00, 0xFF];
    let mut surface = vec![0u8; SLICE as usize];
    if !gpu_heads {
        for block in surface.chunks_mut(256) {
            block[..4].copy_from_slice(&green);
        }
    }
    let mut target = rgba8_target(g.alloc(&surface), W, W);
    target.swizzle = crate::tiling::SWIZZLE_64KB_R_X;
    target.pitch = 128;
    target.slice_size = SLICE;
    target.dcc_enabled = true;
    let keys = 0x1000u64;
    target.dcc_address = g.alloc(&vec![0xFFu8; keys as usize]);
    assert_eq!(target.dcc_slice_size(), keys);
    let dev = GpuDevice::create(false).expect("device");
    let mut b = Backend::new(&dev, g.mem).unwrap();
    let compiler: Arc<dyn ShaderCompiler> = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
    let mut p = ShaderPipelines::new(&dev, g.mem, Some(compiler), None).unwrap();
    p.set_async_compile(false);
    p.set_vertex_input_resolver(Some(Box::new(FixedInputs(vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)])))));
    b.pipelines = Box::new(p);
    let d = draw_into(target, stage(vs, &[]), stage(ps, &color_words([0.0, 0.0, 1.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
    b.draw(&d);
    assert_eq!(b.stats.recorded_draws, 1);
    let id = b.images.find_or_create(&mut b.sched, &mut b.buffers, &color_key(&target)).unwrap();
    let read = |b: &mut Backend| {
        b.images.transit(&mut b.sched, id, crate::images::TRANSFER_SRC, None);
        let image = b.images.get(id).unwrap().image.image;
        read_image(&dev, &mut b.sched, image, W, W)
    };
    // Not a block start: the surface holds zero there.
    assert_eq!(px(&read(&mut b), W, 17, 18), [0, 0, 0, 0]);
    // The SDK clear: every key "single colour" (a fill on the GPU), then
    // the colour at the block starts.
    assert!(b.run_fill_kernel(target.dcc_address, keys, [0x1010_1010; 4]));
    if gpu_heads {
        assert!(b.buffers.fill(&mut b.sched, target.base, 4, u32::from_le_bytes(green)).unwrap());
    }
    b.draw(&d);
    assert_eq!(b.stats.recorded_draws, 2);
    let pixels = read(&mut b);
    for (x, y) in [(17, 18), (31, 31), (8, 3)] {
        assert_eq!(px(&pixels, W, x, y), green, "texel ({x}, {y})");
    }
    // The keys of the cleared slice are uncompressed: the next bind keeps
    // what is drawn.
    assert!(b.buffers.download(&mut b.sched, target.dcc_address, keys));
    let mut k = vec![0u8; keys as usize];
    assert!(g.mem.read_backing(target.dcc_address, &mut k));
    assert!(k.iter().all(|&x| x == 0xFF));
    assert_eq!(b.stats.dcc_cpu_reads, 0);
}

#[test]
#[ignore]
fn single_colour_dcc_keys_clear_to_the_block_colour_in_memory() {
    single_colour_dcc_keys_clear_to_the_block_colour(false);
}

#[test]
#[ignore]
fn single_colour_dcc_keys_clear_to_the_block_colour_on_the_gpu() {
    single_colour_dcc_keys_clear_to_the_block_colour(true);
}

/// GE_INDX_OFFSET moves the generated indices of an auto draw: with an
/// offset of 3 the draw takes vertices 3..6 (a full-screen triangle), not
/// 0..3 (one outside the viewport).
#[test]
#[ignore]
fn auto_draw_starts_at_the_index_offset() {
    let mut g = Guest::new(16 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = solid_program(&mut g);
    let verts = g.alloc(&floats(&[
        5.0, 5.0, 0.0, 1.0, 6.0, 5.0, 0.0, 1.0, 5.0, 6.0, 0.0, 1.0, // outside
        -1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0, // full screen
    ]));
    let target = rgba8_target(g.zeroed(32 * 32 * 4), 32, 32);
    let dev = GpuDevice::create(false).expect("device");
    let mut b = Backend::new(&dev, g.mem).unwrap();
    let compiler: Arc<dyn ShaderCompiler> = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
    let mut p = ShaderPipelines::new(&dev, g.mem, Some(compiler), None).unwrap();
    p.set_async_compile(false);
    p.set_vertex_input_resolver(Some(Box::new(FixedInputs(vec4_attrs(&[(verts, 96, 16, false)], &[(0, 0, 0)])))));
    b.pipelines = Box::new(p);
    let mut d = draw_into(target, stage(vs, &[]), stage(ps, &color_words([0.0, 1.0, 0.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
    d.index_offset = 3;
    b.draw(&d);
    assert_eq!(b.stats.recorded_draws, 1);
    let id = b.images.find_or_create(&mut b.sched, &mut b.buffers, &color_key(&target)).unwrap();
    b.images.transit(&mut b.sched, id, crate::images::TRANSFER_SRC, None);
    let image = b.images.get(id).unwrap().image.image;
    let pixels = read_image(&dev, &mut b.sched, image, 32, 32);
    assert_eq!(px(&pixels, 32, 16, 16), [0, 255, 0, 255], "vertices 3..6 drawn");
}

/// Cleared HTILE clears the whole depth surface at the next bind, not only
/// the binding draw's scissor: a later full-screen draw testing LESS
/// against the cleared value 1.0 passes outside that scissor too. Only a
/// fill whose ZMask is 0 over the whole pipe-aligned metadata (32 KiB for
/// this surface) clears: an expanded fill (ZMask 0xF) or a partial one
/// leaves the depth memory (0.0, nearest), where nothing passes.
#[test]
#[ignore]
fn htile_clear_covers_the_whole_surface_whatever_the_scissor() {
    for (what, size, value, cleared) in [("clear", 0x8000u64, 0u32, true), ("expanded", 0x8000, 0xFFFF_FFFF, false), ("partial", 0x1000, 0, false)] {
        let mut g = Guest::new(16 << 20);
        let vs = g.program(&[asm::S_ENDPGM]);
        let ps = solid_program(&mut g);
        let verts = g.alloc(&floats(&[-1.0, -1.0, 0.5, 1.0, 3.0, -1.0, 0.5, 1.0, -1.0, 3.0, 0.5, 1.0]));
        let target = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
        // Depth memory holds 0.0 (nearest): only a clear lets z 0.5 pass LESS.
        let htile = g.zeroed(0x8000);
        let mut depth = d32_target(g.zeroed(64 * 64 * 4), 64, 64);
        depth.htile_enabled = true;
        depth.htile_base = htile;
        let dev = GpuDevice::create(false).expect("device");
        let mut b = Backend::new(&dev, g.mem).unwrap();
        let compiler: Arc<dyn ShaderCompiler> = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
        let mut p = ShaderPipelines::new(&dev, g.mem, Some(compiler), None).unwrap();
        p.set_async_compile(false);
        p.set_vertex_input_resolver(Some(Box::new(FixedInputs(vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)])))));
        b.pipelines = Box::new(p);
        // The HTILE is filled, DB_DEPTH_CLEAR = 1.0.
        b.memory_filled(htile, size, value);
        let mut first = draw_into(target, stage(vs, &[]), stage(ps, &color_words([1.0, 0.0, 0.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
        first.depth = Some(depth);
        first.depth_state = DepthStencilState { depth_clear_value: 1.0, ..DepthStencilState::default() };
        let corner = Rect { x0: 0, y0: 0, x1: 8, y1: 8 };
        first.scissor = corner;
        first.scissors = [corner; 16];
        b.draw(&first);
        let mut second = draw_into(target, stage(vs, &[]), stage(ps, &color_words([0.0, 1.0, 0.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
        second.depth = Some(depth);
        second.depth_state = DepthStencilState { depth_test: true, depth_compare: 1, depth_clear_value: 1.0, ..DepthStencilState::default() };
        b.draw(&second);
        assert_eq!(b.stats.recorded_draws, 2, "{what}");
        let id = b.images.find_or_create(&mut b.sched, &mut b.buffers, &color_key(&target)).unwrap();
        b.images.transit(&mut b.sched, id, crate::images::TRANSFER_SRC, None);
        let image = b.images.get(id).unwrap().image.image;
        let pixels = read_image(&dev, &mut b.sched, image, 64, 64);
        if cleared {
            assert_eq!(px(&pixels, 64, 2, 2), [0, 255, 0, 255], "{what}: inside the first draw's scissor");
            assert_eq!(px(&pixels, 64, 40, 40), [0, 255, 0, 255], "{what}: outside it: the depth was cleared there too");
        } else {
            assert_eq!(px(&pixels, 64, 2, 2), [255, 0, 0, 255], "{what}: the depth stayed 0.0 inside the first draw's scissor");
            assert_eq!(px(&pixels, 64, 40, 40), [0, 0, 0, 0], "{what}: the depth stayed 0.0 outside it");
        }
    }
}

/// HTILE a compute program stored decides the clear by its words at the
/// next bind: ZMask 0 on every tile clears the depth (and the HTILE is
/// stored back expanded, so the clear is not applied again and readers of
/// the metadata see the written depth); an expanded store (0xFFFFFFFF) or
/// one tile left expanded keeps the depth memory. The reference's
/// `DepthHtileWrite` execution test, through a kernel.
#[test]
#[ignore]
fn htile_written_by_a_compute_program_decides_the_clear_by_its_words() {
    for (what, value, odd_tile, cleared) in [("ZMask 0", 0u32, None, true), ("expanded", 0xFFFF_FFFF, None, false), ("one tile expanded", 0, Some(0xFFFF_FFFFu32), false)] {
        let mut g = Guest::new(16 << 20);
        let vs = g.program(&[asm::S_ENDPGM]);
        let ps = solid_program(&mut g);
        let verts = g.alloc(&floats(&[-1.0, -1.0, 0.5, 1.0, 3.0, -1.0, 0.5, 1.0, -1.0, 3.0, 0.5, 1.0]));
        let target = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
        // Depth memory holds 0.0 (nearest): only a clear lets z 0.5 pass LESS.
        let htile = g.alloc(&vec![0xFFu8; 0x8000]);
        let mut depth = d32_target(g.zeroed(64 * 64 * 4), 64, 64);
        depth.htile_enabled = true;
        depth.htile_base = htile;
        let dev = GpuDevice::create(false).expect("device");
        let mut b = Backend::new(&dev, g.mem).unwrap();
        let compiler: Arc<dyn ShaderCompiler> = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
        let mut p = ShaderPipelines::new(&dev, g.mem, Some(compiler), None).unwrap();
        p.set_async_compile(false);
        p.set_vertex_input_resolver(Some(Box::new(FixedInputs(vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)])))));
        b.pipelines = Box::new(OverwritingDispatches { inner: p, overwrites: vec![(htile, 0x8000)], pending: Vec::new() });
        // Red everywhere; the bind makes the HTILE known.
        let mut first = draw_into(target, stage(vs, &[]), stage(ps, &color_words([1.0, 0.0, 0.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
        first.depth = Some(depth);
        first.depth_state = DepthStencilState { depth_clear_value: 1.0, ..DepthStencilState::default() };
        b.draw(&first);
        // The kernel's words, on the GPU.
        assert!(b.buffers.fill(&mut b.sched, htile, 0x8000, value).unwrap());
        if let Some(w) = odd_tile {
            assert!(b.buffers.fill(&mut b.sched, htile + 0x100, 4, w).unwrap());
        }
        let dispatch = ResolvedDispatch { packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() }, shader: stage(0x1000, &[]), rsrc3: 0, threads: [64, 1, 1], start: [0; 3] };
        b.dispatch(&dispatch);
        // Green where z 0.5 passes LESS.
        let mut second = draw_into(target, stage(vs, &[]), stage(ps, &color_words([0.0, 1.0, 0.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
        second.depth = Some(depth);
        second.depth_state = DepthStencilState { depth_test: true, depth_compare: 1, depth_clear_value: 1.0, ..DepthStencilState::default() };
        b.draw(&second);
        assert_eq!(b.stats.recorded_draws, 2, "{what}");
        let id = b.images.find_or_create(&mut b.sched, &mut b.buffers, &color_key(&target)).unwrap();
        b.images.transit(&mut b.sched, id, crate::images::TRANSFER_SRC, None);
        let image = b.images.get(id).unwrap().image.image;
        let pixels = read_image(&dev, &mut b.sched, image, 64, 64);
        let expected = if cleared { [0, 255, 0, 255] } else { [255, 0, 0, 255] };
        assert_eq!(px(&pixels, 64, 40, 40), expected, "{what}");
        if cleared {
            b.flush();
            b.buffers.download(&mut b.sched, htile, 0x8000);
            let mut bytes = vec![0u8; 0x8000];
            assert!(g.mem.read_backing(htile, &mut bytes));
            assert!(bytes.chunks(4).all(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) == 0xFFFC_000F), "{what}: the HTILE was stored back expanded");
        }
    }
}

/// Pixel inputs naming a parameter the vertex stage does not export read
/// their default 0 (interpolated inputs only; FP16 ones in both halves).
#[test]
fn unexported_pixel_inputs_read_zero() {
    use crate::pipeline::provider::default_unexported_inputs;
    let mut cntl = [0u32; 32];
    cntl[..6].copy_from_slice(&[0x000, 0x401, 0x302, 0x123, 0x68_0004, 0x005]);
    let out = default_unexported_inputs(&cntl, 5, 0b1).unwrap();
    assert_eq!(&out[..6], &[0x000, 0x021, 0x022, 0x123, 0x18_0024, 0x005], "input 5 is past NUM_INTERP");
    assert_eq!(default_unexported_inputs(&cntl, 5, 0b11_0111), None, "every read parameter exported");
    assert_eq!(default_unexported_inputs(&cntl, 0, 0), None);
}

/// DB_SHADER_CONTROL.ALPHA_TO_MASK_DISABLE (bit 11) turns alpha-to-coverage
/// off while DB_ALPHA_TO_MASK keeps it programmed.
#[test]
fn alpha_to_mask_disable_turns_coverage_off() {
    let mut d = draw_into(rgba8_target(0x10_0000, 4, 4), stage(0x1000, &[]), stage(0x2000, &[]), primitive::TRI_LIST, auto(3, 1), None);
    d.alpha_to_mask = 0x18701;
    assert!(d.alpha_to_coverage() && d.pixel_side_effects());
    d.db_shader_control = 0x800;
    assert!(!d.alpha_to_coverage() && !d.pixel_side_effects());
    d.db_shader_control = 0;
    d.alpha_to_mask = 0xAA00;
    assert!(!d.alpha_to_coverage());
}

/// PA_CL_CLIP_CNTL.UCP_ENA takes the PA_CL_UCP planes in the plain mode;
/// CLIP_DISABLE, the special modes and guest clip distances leave them out.
#[test]
fn user_clip_planes_follow_clip_cntl() {
    use super::state::user_clip_planes;
    let mut d = draw_into(rgba8_target(0x10_0000, 4, 4), stage(0x1000, &[]), stage(0x2000, &[]), primitive::TRI_LIST, auto(3, 1), None);
    d.raster.user_clip_planes = std::array::from_fn(|i| i as u32 + 1);
    assert_eq!(user_clip_planes(&d, true).mask, 0);
    d.raster.clip_cntl = 1 << 19 | 0b10_0001;
    let u = user_clip_planes(&d, true);
    assert_eq!((u.mask, u.count(), u.planes), (0b10_0001, 2, d.raster.user_clip_planes));
    assert_eq!(user_clip_planes(&d, false).mask, 0, "RPS5E_USER_CLIP_PLANES=0");
    for extra in [1 << 16, 1 << 13, 1 << 14, 1 << 17] {
        d.raster.clip_cntl = 1 << 19 | 0b1 | extra;
        assert_eq!(user_clip_planes(&d, true).mask, 0, "{extra:#x}");
    }
    d.raster.clip_cntl = 0b1;
    d.raster.vs_out_cntl = 0x1;
    assert_eq!(user_clip_planes(&d, true).mask, 0, "guest clip distances");
}

/// Each slice of a depth array owns the HTILE at base + slice * (the
/// slice's metadata bytes): a fill of slice 0's HTILE leaves slice 1, a fill
/// of slice 1's own HTILE or one spanning both clears it (reference
/// 8fec47a4b).
#[test]
#[ignore]
fn htile_fills_clear_the_slices_they_span() {
    const B: u64 = 0x8000;
    for (what, at, size, cleared) in [("slice 0 only", 0u64, B, false), ("slice 1's own", B, B, true), ("both slices", 0, 2 * B, true)] {
        let mut g = Guest::new(16 << 20);
        let vs = g.program(&[asm::S_ENDPGM]);
        let ps = solid_program(&mut g);
        let verts = g.alloc(&floats(&[-1.0, -1.0, 0.5, 1.0, 3.0, -1.0, 0.5, 1.0, -1.0, 3.0, 0.5, 1.0]));
        let target = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
        let htile = g.zeroed(2 * B as usize);
        let mut depth = d32_target(g.zeroed(2 * 64 * 64 * 4), 64, 64);
        depth.htile_enabled = true;
        depth.htile_base = htile;
        depth.base_layer = 1;
        let dev = GpuDevice::create(false).expect("device");
        let mut b = Backend::new(&dev, g.mem).unwrap();
        let compiler: Arc<dyn ShaderCompiler> = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
        let mut p = ShaderPipelines::new(&dev, g.mem, Some(compiler), None).unwrap();
        p.set_async_compile(false);
        p.set_vertex_input_resolver(Some(Box::new(FixedInputs(vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)])))));
        b.pipelines = Box::new(p);
        b.memory_filled(htile + at, size, 0);
        let mut d = draw_into(target, stage(vs, &[]), stage(ps, &color_words([0.0, 1.0, 0.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
        d.depth = Some(depth);
        d.depth_state = DepthStencilState { depth_test: true, depth_compare: 1, depth_clear_value: 1.0, ..DepthStencilState::default() };
        b.draw(&d);
        assert_eq!(b.stats.recorded_draws, 1, "{what}");
        let id = b.images.find_or_create(&mut b.sched, &mut b.buffers, &color_key(&target)).unwrap();
        b.images.transit(&mut b.sched, id, crate::images::TRANSFER_SRC, None);
        let image = b.images.get(id).unwrap().image.image;
        let pixels = read_image(&dev, &mut b.sched, image, 64, 64);
        let expected = if cleared { [0, 255, 0, 255] } else { [0, 0, 0, 0] };
        assert_eq!(px(&pixels, 64, 40, 40), expected, "{what}");
    }
}

/// A metadata fill that later writes replace is no clear any more: an
/// HTILE fill (ZMask 0) overwritten by a command-processor copy of
/// expanded words before the surface is bound leaves the depth memory.
#[test]
#[ignore]
fn metadata_fill_overwritten_before_its_bind_does_not_clear() {
    for (what, copied, cleared) in [("fill only", false, true), ("fill then copy", true, false)] {
        let mut g = Guest::new(16 << 20);
        let vs = g.program(&[asm::S_ENDPGM]);
        let ps = solid_program(&mut g);
        let verts = g.alloc(&floats(&[-1.0, -1.0, 0.5, 1.0, 3.0, -1.0, 0.5, 1.0, -1.0, 3.0, 0.5, 1.0]));
        let target = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
        let htile = g.zeroed(0x8000);
        let expanded = g.alloc(&vec![0xFFu8; 0x8000]);
        let mut depth = d32_target(g.zeroed(64 * 64 * 4), 64, 64);
        depth.htile_enabled = true;
        depth.htile_base = htile;
        let dev = GpuDevice::create(false).expect("device");
        let mut b = Backend::new(&dev, g.mem).unwrap();
        let compiler: Arc<dyn ShaderCompiler> = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
        let mut p = ShaderPipelines::new(&dev, g.mem, Some(compiler), None).unwrap();
        p.set_async_compile(false);
        p.set_vertex_input_resolver(Some(Box::new(FixedInputs(vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)])))));
        b.pipelines = Box::new(p);
        b.memory_filled(htile, 0x8000, 0);
        if copied {
            // The command processor copies on the CPU when the backend
            // does not (no GPU-written pages).
            if !b.copy_memory(htile, expanded, 0x8000) {
                assert!(g.mem.write_backing(htile, &vec![0xFFu8; 0x8000]));
            }
        }
        let mut d = draw_into(target, stage(vs, &[]), stage(ps, &color_words([0.0, 1.0, 0.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
        d.depth = Some(depth);
        d.depth_state = DepthStencilState { depth_test: true, depth_compare: 1, depth_clear_value: 1.0, ..DepthStencilState::default() };
        b.draw(&d);
        let id = b.images.find_or_create(&mut b.sched, &mut b.buffers, &color_key(&target)).unwrap();
        b.images.transit(&mut b.sched, id, crate::images::TRANSFER_SRC, None);
        let image = b.images.get(id).unwrap().image.image;
        let pixels = read_image(&dev, &mut b.sched, image, 64, 64);
        let expected = if cleared { [0, 255, 0, 255] } else { [0, 0, 0, 0] };
        assert_eq!(px(&pixels, 64, 40, 40), expected, "{what}");
    }
}

/// HTILE words decide the clear tile by tile; a cleared plane is stored
/// back expanded (the values the reference driver writes).
#[test]
fn htile_words_decide_the_clear_and_expand() {
    use crate::backend::{expanded_htile_word, htile_words_clear};
    assert_eq!(htile_words_clear(&[0; 8], false, true), (true, false));
    assert_eq!(htile_words_clear(&[0, 0, 0xF, 0], false, true), (false, false), "one tile expanded");
    assert_eq!(htile_words_clear(&[0xFFFF_F00F; 4], true, true), (false, true), "SMem 0 only");
    assert_eq!(htile_words_clear(&[0x0000_00F0, 0x0000_0030], true, true), (true, true));
    assert_eq!(htile_words_clear(&[0x0000_00F0, 0x0000_0030], true, false), (false, false), "a later slice needs uniform words");
    assert_eq!(htile_words_clear(&[], false, true), (false, false));
    assert_eq!(expanded_htile_word(0, false, true, false), 0xFFFC_000F);
    assert_eq!(expanded_htile_word(0x1234, false, false, false), 0x1234);
    assert_eq!(expanded_htile_word(0x0000_00F0, true, true, false), 0xFFFF_F0FF);
    assert_eq!(expanded_htile_word(0xFFFF_F00F, true, false, true), 0xFFFF_F3FF);
    assert_eq!(expanded_htile_word(0x0000_0000, true, true, true), 0xFFFF_F3FF);
}

/// HTILE fill patterns: ZMask 0 clears the depth, SMem 0 the stencil when
/// it keeps its state in HTILE; the metadata is 32 KiB per 1024x512.
#[test]
fn htile_fill_patterns_and_metadata_size() {
    use crate::backend::{htile_fill_clears, htile_metadata_bytes};
    assert_eq!(htile_fill_clears(0x0003_fff0, false), (true, false), "a depth-only fill with ZMask 0 clears the depth");
    assert_eq!(htile_fill_clears(0x0000_00f0, true), (true, true), "ZMask 0, SMem 0 clears both");
    assert_eq!(htile_fill_clears(0x0000_03f0, true), (true, false), "SMem 3 leaves the stencil");
    assert_eq!(htile_fill_clears(0xffff_f0ff, true), (false, true), "ZMask 0xF with SMem 0 clears only the stencil");
    assert_eq!(htile_fill_clears(0xffff_ffff, true), (false, false), "an expanded fill clears nothing");
    assert_eq!(htile_fill_clears(0xffff_ffff, false), (false, false));
    assert_eq!(htile_metadata_bytes(1024, 512), 0x8000);
    assert_eq!(htile_metadata_bytes(1025, 512), 0x10000);
    assert_eq!(htile_metadata_bytes(3840, 2160), 20 * 0x8000);
    assert_eq!(htile_metadata_bytes(16, 16), 0x8000);
}

/// A linear D32 depth target.
fn d32_target(base: u64, w: u32, h: u32) -> DepthTarget {
    DepthTarget {
        z_format: crate::render::formats::depth::Z32_FLOAT,
        host_format: vk::Format::D32_SFLOAT,
        bytes_per_element: 4,
        has_stencil: false,
        width: w,
        height: h,
        pitch: w,
        swizzle: crate::tiling::SWIZZLE_LINEAR,
        samples: 1,
        base_layer: 0,
        layers: 1,
        z_read_base: base,
        z_write_base: base,
        stencil_read_base: 0,
        stencil_write_base: 0,
        htile_base: 0,
        htile_enabled: false,
        depth_slice_size: (w * h * 4) as u64,
        stencil_slice_size: 0,
        z_info: 0,
        stencil_info: 0,
        depth_view: 0,
    }
}

/// The DB depth copy writes the depth the GPU holds in the depth image
/// into the R32 float colour target (reference: DepthStorage.cpp
/// one-sample DB depth copy).
#[test]
#[ignore]
fn db_depth_copy_writes_the_depth_into_the_r32_target() {
    let mut g = Guest::new(16 << 20);
    let mut rig = Rig::new(&g, Arc::new(TestCompiler::new()), None, None);
    let (w, h) = (64u32, 32u32);
    let z = g.zeroed((w * h * 4) as usize);
    let c = g.alloc(&vec![0xAAu8; (w * h * 4) as usize]);
    let depth = d32_target(z, w, h);
    let mut color = rgba8_target(c, w, h);
    color.format = render_target_format(4, 7, 0).unwrap();
    color.layout = 4;
    color.number_type = 7;
    assert_eq!(color.format.host, vk::Format::R32_SFLOAT);
    let src = rig.images.find_or_create(&mut rig.sched, &mut rig.buffers, &crate::images::depth_key(&depth)).unwrap();
    rig.images.prepare(&mut rig.sched, &mut rig.buffers, src, false);
    rig.images.clear_depth_stencil(&mut rig.sched, src, Some(0.375), None, None, None);
    let tick = rig.sched.current_tick();
    rig.images.mark_gpu_written(src, tick);
    let dst = rig.images.find_or_create(&mut rig.sched, &mut rig.buffers, &crate::images::color_key(&color)).unwrap();
    rig.images.prepare(&mut rig.sched, &mut rig.buffers, dst, false);
    assert!(rig.images.copy_depth_to_color(&mut rig.sched, src, 0, dst, 0));
    let size = (w * h * 4) as u64;
    rig.images.writeback_range(&mut rig.sched, &mut rig.buffers, c, size);
    rig.buffers.download(&mut rig.sched, c, size);
    let mut out = vec![0u8; size as usize];
    assert!(g.mem.read_backing(c, &mut out));
    assert!(out.chunks(4).all(|t| f32::from_le_bytes(t.try_into().unwrap()) == 0.375), "first texels {:x?}", &out[..16]);
    // A colour target of another size, or a multisampled plane, is refused.
    let small = rgba8_target(c, w / 2, h);
    let other = rig.images.find_or_create(&mut rig.sched, &mut rig.buffers, &crate::images::color_key(&small)).unwrap();
    assert!(!rig.images.copy_depth_to_color(&mut rig.sched, src, 0, other, 0));
}

/// Which draws are DB depth copies of the form the copy is done in
/// (reference State.cpp DecodeDepthCopyPass).
#[test]
fn db_depth_copy_decode() {
    let (w, h) = (64u32, 32u32);
    let mut color = rgba8_target(0x10_0000, w, h);
    color.format = render_target_format(4, 7, 0).unwrap();
    let mut d = draw_into(color, unbound(), unbound(), 4, auto(3, 1), None);
    d.depth = Some(d32_target(0x20_0000, w, h));
    d.target_mask = 0xF;
    d.shader_mask = 0x1;
    assert_eq!(d.db_depth_copy(), None, "DEPTH_COPY clear");
    d.depth_state.render_control = 4 | 0x80;
    assert_eq!(d.db_depth_copy(), Some(Ok(0)));
    d.depth_state.render_control = 4 | (1 << 8);
    assert!(matches!(d.db_depth_copy(), Some(Err(_))), "COPY_SAMPLE 1 of a one-sample plane");
    let ok = d.clone();
    let refused = |f: &dyn Fn(&mut ResolvedDraw)| {
        let mut x = ok.clone();
        x.depth_state.render_control = 4;
        assert_eq!(x.db_depth_copy(), Some(Ok(0)));
        f(&mut x);
        matches!(x.db_depth_copy(), Some(Err(_)))
    };
    assert!(refused(&|x| x.depth_state.render_control |= 8), "with a stencil copy");
    assert!(refused(&|x| x.depth_state.depth_control = 2), "with a depth test");
    assert!(refused(&|x| x.shader_mask = 3), "into green too");
    assert!(refused(&|x| x.aa_mask[0] = 0xFFFF), "under a sample mask");
    assert!(refused(&|x| x.color_targets[0].width = w / 2), "into a smaller target");
    assert!(refused(&|x| x.color_targets[0] = rgba8_target(0x10_0000, w, h)), "into an RGBA8 target");
    assert!(refused(&|x| x.scissor.x1 = 16), "over part of the surface");
    assert!(refused(&|x| x.depth = Some(DepthTarget { z_format: crate::render::formats::depth::Z16, ..d32_target(0x20_0000, w, h) })), "from 16-bit depth");
}

#[test]
#[ignore]
fn pipeline_depth_test_and_blending() {
    let mut g = Guest::new(16 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = solid_program(&mut g);
    let verts = g.alloc(&floats(&[
        -1.0, -1.0, 0.5, 1.0, 3.0, -1.0, 0.5, 1.0, -1.0, 3.0, 0.5, 1.0, // A: full screen, z 0.5
        -1.0, -1.0, 0.7, 1.0, 1.0, -1.0, 0.7, 1.0, -1.0, 1.0, 0.7, 1.0, // B: top-left half, z 0.7
        -1.0, -1.0, 0.3, 1.0, 0.0, -1.0, 0.3, 1.0, -1.0, 0.0, 0.3, 1.0, // C: top-left corner, z 0.3
    ]));
    let indices = g.alloc(&[0u16, 1, 2, 3, 4, 5, 6, 7, 8].iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>());
    let target = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
    let depth = d32_target(g.alloc(&floats(&[1.0; 64 * 64])), 64, 64);
    let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
    let mut rig = Rig::new(&g, compiler, Some(vec4_attrs(&[(verts, 144, 16, false)], &[(0, 0, 0)])), None);
    let draw = |tri: u64, color: [f32; 4], ds: DepthStencilState, blend: Option<BlendState>| {
        let packet = DrawPacket { kind: DrawKind::Index, count: 3, instance_count: 1, index_address: indices + tri * 6, ..DrawPacket::default() };
        let index = IndexBuffer { address: indices + tri * 6, kind: IndexType::U16, count: 3 };
        let mut t = target;
        if let Some(b) = blend {
            t.blend = b;
        }
        let mut d = draw_into(t, stage(vs, &[]), stage(ps, &color_words(color)), primitive::TRI_LIST, packet, Some(index));
        d.depth = Some(depth);
        d.depth_state = ds;
        d
    };
    let test = |compare: u32| DepthStencilState { depth_test: true, depth_write: true, depth_compare: compare, ..DepthStencilState::default() };
    assert_eq!(rig.draw(&draw(0, [1.0, 0.0, 0.0, 1.0], test(7), None)), DrawOutcome::Recorded);
    assert_eq!(rig.draw(&draw(1, [0.0, 1.0, 0.0, 1.0], test(1), None)), DrawOutcome::Recorded);
    assert_eq!(rig.draw(&draw(2, [0.0, 0.0, 1.0, 1.0], test(1), None)), DrawOutcome::Recorded);
    let p = rig.pixels(&target);
    assert_eq!(px(&p, 64, 2, 2), [0, 0, 255, 255], "nearer triangle passes");
    assert_eq!(px(&p, 64, 20, 20), [255, 0, 0, 255], "farther triangle fails the depth test");
    // Additive blending (ONE, ONE) without depth test.
    let add = BlendState::decode(1 | (1 << 8) | (1 << 30));
    assert_eq!(rig.draw(&draw(0, [0.0, 0.2, 0.0, 0.0], DepthStencilState::default(), Some(add))), DrawOutcome::Recorded);
    let p = rig.pixels(&target);
    assert_eq!(px(&p, 64, 2, 2), [0, 51, 255, 255]);
    assert_eq!(px(&p, 64, 40, 40), [255, 51, 0, 255]);
    assert_eq!(rig.provider.stats().graphics_pipelines, 2, "blending is part of the pipeline key; depth state is dynamic");
}

/// GE_INDX_OFFSET moves the vertex IDs of an auto-indexed draw (the
/// reference drivers' direct auto draw with GE_INDX_OFFSET 5): the draw of
/// three vertices with offset 3 fetches vertices 3..5.
#[test]
#[ignore]
fn pipeline_auto_draw_index_offset() {
    let mut g = Guest::new(16 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = solid_program(&mut g);
    // Vertices 0..2: a triangle outside the target; 3..5: one covering it.
    let verts = g.alloc(&floats(&[
        5.0, 5.0, 0.0, 1.0, 6.0, 5.0, 0.0, 1.0, 5.0, 6.0, 0.0, 1.0, //
        -1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0,
    ]));
    let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
    let mut rig = Rig::new(&g, compiler, Some(vec4_attrs(&[(verts, 96, 16, false)], &[(0, 0, 0)])), None);
    for (offset, drawn) in [(0u32, false), (3, true)] {
        let target = rgba8_target(g.zeroed(32 * 32 * 4), 32, 32);
        let mut d = draw_into(target, stage(vs, &[]), stage(ps, &color_words([0.0, 1.0, 0.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
        d.index_offset = offset;
        assert_eq!(rig.draw(&d), DrawOutcome::Recorded);
        let p = rig.pixels(&target);
        let expected = if drawn { [0, 255, 0, 255] } else { [0; 4] };
        assert_eq!(px(&p, 32, 16, 16), expected, "GE_INDX_OFFSET {offset}");
    }
}

/// A draw whose faces need different depth biases records front faces
/// (back faces culled) with the front bias, then back faces with theirs.
#[test]
fn per_face_depth_bias_splits_the_draw() {
    let target = rgba8_target(0x10_0000, 16, 16);
    let mut d = draw_into(target, unbound(), unbound(), primitive::TRI_LIST, auto(3, 1), None);
    d.raster.mode_cntl = 1 << 11;
    d.raster.poly_offset = [0, 0, (-4.0f32).to_bits(), 0, 0];
    d.raster.poly_offset_db_fmt = 1 << 8;
    let s = state::dynamic_state(&d, 16, 16, vk::Format::D32_SFLOAT, true, false, false, 1, None);
    assert_eq!(s.cull_mode, vk::CullModeFlags::BACK);
    assert_eq!(s.depth_bias, Some([-4.0, 0.0, 0.0]));
    let back = s.back_faces().expect("a back-face pass");
    assert_eq!(back.cull_mode, vk::CullModeFlags::FRONT);
    assert_eq!(back.depth_bias, Some([0.0, 0.0, 0.0]));
    assert!(back.back_faces().is_none());
    // Back faces culled by the guest: one pass with the front bias.
    d.raster.cull_back = true;
    let s = state::dynamic_state(&d, 16, 16, vk::Format::D32_SFLOAT, true, false, false, 1, None);
    assert_eq!((s.cull_mode, s.depth_bias), (vk::CullModeFlags::BACK, Some([-4.0, 0.0, 0.0])));
    assert!(s.back_faces().is_none());
    // Rectangles and draws without a depth attachment never split.
    d.raster.cull_back = false;
    assert!(state::dynamic_state(&d, 16, 16, vk::Format::D32_SFLOAT, true, false, true, 1, None).back_faces().is_none());
    assert!(state::dynamic_state(&d, 16, 16, vk::Format::UNDEFINED, false, false, false, 1, None).back_faces().is_none());
}

/// PA_SU_POLY_OFFSET_FRONT_* and BACK_* on one draw of a front-facing and
/// a back-facing triangle at depth 0.5 over a depth buffer of 0.5 (LESS,
/// no depth writes): a face is drawn exactly when its bias pulls it closer.
/// The cases and expectations are those of the reference driver's
/// per-face depth bias test.
#[test]
#[ignore]
fn pipeline_per_face_depth_bias() {
    const W: u32 = 64;
    const H: u32 = 16;
    const BIAS: f32 = 1_048_576.0;
    let mut g = Guest::new(16 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = solid_program(&mut g);
    // Framebuffer corners (y down): the first triangle is counter-clockwise
    // (front-facing), the second clockwise (back-facing).
    let corners = [(1.0f32, 1.0f32), (1.0, 15.0), (15.0, 1.0), (17.0, 1.0), (31.0, 1.0), (17.0, 15.0)];
    let mut v = Vec::new();
    for (x, y) in corners {
        v.extend_from_slice(&[x * 2.0 / W as f32 - 1.0, y * 2.0 / H as f32 - 1.0, 0.5, 1.0]);
    }
    let verts = g.alloc(&floats(&v));
    let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
    let mut rig = Rig::new(&g, compiler, Some(vec4_attrs(&[(verts, 96, 16, false)], &[(0, 0, 0)])), None);
    // (front enable, back enable, front offset, back offset, front drawn, back drawn, what)
    let cases = [
        (true, true, -BIAS, BIAS, true, false, "front faces pulled closer, back faces pushed away"),
        (true, true, BIAS, -BIAS, false, true, "front faces pushed away, back faces pulled closer"),
        (true, true, -BIAS, -BIAS, true, true, "both faces pulled closer"),
        (true, false, -BIAS, BIAS, true, false, "only front faces biased"),
        (false, true, BIAS, -BIAS, false, true, "only back faces biased"),
    ];
    for (front_on, back_on, front, back, front_drawn, back_drawn, what) in cases {
        let target = rgba8_target(g.alloc(&vec![0x40u8; (W * H * 4) as usize]), W, H);
        let depth = d32_target(g.alloc(&floats(&[0.5; (W * H) as usize])), W, H);
        let mut d = draw_into(target, stage(vs, &[]), stage(ps, &color_words([1.0; 4])), primitive::TRI_LIST, auto(6, 1), None);
        d.depth = Some(depth);
        d.depth_state = DepthStencilState { depth_test: true, depth_write: false, depth_compare: 1, ..DepthStencilState::default() };
        d.raster.mode_cntl = (front_on as u32) << 11 | (back_on as u32) << 12;
        d.raster.poly_offset = [0, 0, front.to_bits(), 0, back.to_bits()];
        d.raster.poly_offset_db_fmt = 1 << 8;
        assert_eq!(rig.draw(&d), DrawOutcome::Recorded, "{what}");
        let p = rig.pixels(&target);
        let expect = |drawn: bool| if drawn { [255; 4] } else { [0x40; 4] };
        assert_eq!(px(&p, W, 4, 4), expect(front_drawn), "{what}: the front-facing triangle");
        assert_eq!(px(&p, W, 20, 4), expect(back_drawn), "{what}: the back-facing triangle");
    }
}

/// PA_SU_SC_MODE_CNTL.PROVOKING_VTX_LAST: a flat attribute comes from the
/// first vertex of each triangle, or from the last with the bit set
/// (triangle lists and strips; the cases of the reference driver's
/// provoking vertex test). Without provokingVertexLast the first vertex
/// provokes either way.
#[test]
#[ignore]
fn pipeline_provoking_vertex() {
    const S: u32 = 64;
    let mut g = Guest::new(16 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = g.program(&[asm::S_ENDPGM]);
    let red = [1.0, 0.0, 0.0, 1.0];
    let green = [0.0, 1.0, 0.0, 1.0];
    let blue = [0.0, 0.0, 1.0, 1.0];
    let yellow = [1.0, 1.0, 0.0, 1.0];
    let cyan = [0.0, 1.0, 1.0, 1.0];
    let magenta = [1.0, 0.0, 1.0, 1.0];
    // Framebuffer corners: the first triangle covers the top left, the
    // second (list or strip) the bottom right.
    let vertices = [((8.0f32, 8.0f32), red), ((8.0, 56.0), green), ((56.0, 8.0), blue), ((56.0, 56.0), yellow), ((56.0, 8.0), cyan), ((8.0, 56.0), magenta)];
    let mut v = Vec::new();
    for ((x, y), c) in vertices {
        v.extend_from_slice(&[x * 2.0 / S as f32 - 1.0, y * 2.0 / S as f32 - 1.0, 0.5, 1.0]);
        v.extend_from_slice(&c);
    }
    let verts = g.alloc(&floats(&v));
    let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 2 }).with(ps, TestProgram::FlatPixel));
    let mut rig = Rig::new(&g, compiler, Some(vec4_attrs(&[(verts, 6 * 32, 32, false)], &[(0, 0, 0), (1, 0, 16)])), None);
    let last_supported = rig.dev.caps.provoking_vertex_last;
    let byte = |c: [f32; 4]| c.map(|x| (x * 255.0) as u8);
    for (prim, count) in [(primitive::TRI_LIST, 6), (primitive::TRI_STRIP, 4)] {
        for last in [false, true] {
            let target = rgba8_target(g.zeroed((S * S * 4) as usize), S, S);
            let mut d = draw_into(target, stage(vs, &[]), stage(ps, &[]), prim, auto(count, 1), None);
            d.raster.provoking_vertex_last = last;
            d.raster.mode_cntl = (last as u32) << 19;
            assert_eq!(rig.draw(&d), DrawOutcome::Recorded);
            let p = rig.pixels(&target);
            let last = last && last_supported;
            let first_triangle = if last { blue } else { red };
            let second_triangle = match (prim == primitive::TRI_STRIP, last) {
                (true, true) => yellow,
                (true, false) => green,
                (false, true) => magenta,
                (false, false) => yellow,
            };
            let what = format!("primitive {prim}, last vertex {last}");
            for o in 0..4 {
                assert_eq!(px(&p, S, 16 + o, 16), byte(first_triangle), "{what}: first triangle");
                assert_eq!(px(&p, S, 48 - o, 48), byte(second_triangle), "{what}: second triangle");
            }
            assert_eq!(px(&p, S, 0, 0), [0; 4], "{what}: background");
        }
    }
    if last_supported {
        assert!(rig.provider.stats().graphics_pipelines >= 4, "the provoking vertex mode is part of the pipeline key");
    }
}

#[allow(dead_code)]
mod ref_int16 {
    /// The fields of a reference compute case that the device run uses.
    pub struct Compute {
        pub threads: [u32; 3],
        pub lds_dwords: u32,
        pub wave_size: u32,
        pub float_mode: u32,
        pub dispatch_threads: [u32; 3],
        pub workgroup_register: u32,
        pub thread_ids: u32,
        pub group_id: [bool; 3],
        pub dispatch_dims: bool,
        pub tg_size_en: bool,
    }

    pub struct RefCase {
        pub name: &'static str,
        pub code: &'static [u32],
        pub initial: &'static [u32],
        pub expected: &'static [u32],
        pub user_data: Option<&'static [u32]>,
        pub compute: Compute,
        pub dispatch: [u32; 3],
        pub compile_only: bool,
        pub tolerance: f32,
        pub bda: &'static [(u64, u32)],
    }

    include!("../../../ps5shader/tests/vectors/ref_int16.rs");
}

/// The 16-bit integer reference cases through the SPIR-V compiler on the
/// device: the same words the interpreter tests run, with their result
/// stores redirected from s[48:51] to a V# in s[0:3].
#[test]
#[ignore]
fn pipeline_spirv_int16_alu_reference_cases() {
    use ref_int16::*;
    let mut g = Guest::new(16 << 20);
    let cases = [&VOP3P_INT16_OPS, &VOP3_INT16_OPS, &VECTOR_COMPARE_I16_U16_OPS];
    let programs: Vec<(u64, &RefCase)> = cases
        .iter()
        .map(|&c| {
            let mut words = c.code.to_vec();
            for i in 1..words.len() {
                // buffer_store_dword ... offen: SRSRC (word 1, bits 16..21) 12 -> 0.
                if words[i - 1] == 0xE070_1000 && (words[i] >> 16) & 0x1F == 12 {
                    words[i] &= !(0x1F << 16);
                }
            }
            (g.program(&words), c)
        })
        .collect();
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    for (cs, case) in programs {
        assert_eq!(case.compute.wave_size, 64, "{}", case.name);
        let region = g.zeroed(4096);
        let v_sharp = [region as u32, (region >> 32) as u32, 4096, (3 << 28) | (20 << 12) | 0xFAC];
        let mut shader = stage(cs, &v_sharp);
        shader.rsrc2 = (v_sharp.len() as u32) << 1;
        let d = ResolvedDispatch {
            packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() },
            shader,
            rsrc3: 0,
            threads: [64, 1, 1],
            start: [0; 3],
        };
        assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded, "{}", case.name);
        assert!(rig.buffers.download(&mut rig.sched, region, 4096), "{}", case.name);
        let mut out = vec![0u8; case.expected.len() * 4];
        assert!(g.mem.read_backing(region, &mut out));
        let got: Vec<u32> = out.chunks(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect();
        for (i, (&g, &e)) in got.iter().zip(case.expected).enumerate() {
            assert_eq!(g, e, "{} word {i}: got {g:#010x}, expected {e:#010x}", case.name);
        }
    }
}

/// Vertex programs through the SPIR-V compiler, everything else through
/// the test compiler.
struct SpirvVertices {
    spirv: crate::shader::SpirvCompiler,
    vertices: Vec<u64>,
    rest: TestCompiler,
}

impl ShaderCompiler for SpirvVertices {
    fn compile(&self, input: &compiler::ShaderCompileInput<'_, '_>) -> Result<compiler::CompiledShader, compiler::CompileError> {
        if self.vertices.contains(&input.guest_address) {
            self.spirv.compile(input)
        } else {
            self.rest.compile(input)
        }
    }
}

/// A native vertex program of the merged (ES) layout that, like guest
/// vertex fetch, adds the vertex offset in s8 and the instance offset in
/// s9 to the vertex ID (v5) and instance ID (v8). Only vertex IDs 7..9 of
/// instance 5 produce the full-screen triangle; any other ID lands
/// outside the target.
fn offset_vertex_program(g: &mut Guest) -> u64 {
    let vop1 = |op: u32, dst: u32, src0: u32| (0x3F << 25) | (dst << 17) | (op << 9) | src0;
    let vop2 = |op: u32, dst: u32, src0: u32, src1: u32| (op << 25) | (dst << 17) | ((src1 & 0xFF) << 9) | src0;
    let v = |n: u32| 256 + n;
    let inline = |n: u32| 128 + n;
    const ADD_U32: u32 = 0x25;
    const SUBREV_U32: u32 = 0x27;
    const AND: u32 = 0x1B;
    const LSHL: u32 = 0x1A;
    const LSHR: u32 = 0x16;
    const MUL_U24: u32 = 0x0B;
    const SUBREV_F32: u32 = 0x05;
    const ONE_F32: u32 = 242;
    let words = [
        vop2(ADD_U32, 5, 8, v(5)),
        vop2(ADD_U32, 8, 9, v(8)),
        vop2(SUBREV_U32, 5, inline(7), v(5)),
        vop2(SUBREV_U32, 8, inline(5), v(8)),
        // x = (id & 1) * 4, y = (id & 2) * 2, both + ((id >> 2) + instance) * 100.
        vop2(AND, 0, inline(1), v(5)),
        vop2(LSHL, 0, inline(2), v(0)),
        vop2(AND, 1, inline(2), v(5)),
        vop2(LSHL, 1, inline(1), v(1)),
        vop2(LSHR, 2, inline(2), v(5)),
        vop2(ADD_U32, 2, v(2), v(8)),
        vop2(MUL_U24, 2, 255, v(2)),
        100,
        vop2(ADD_U32, 0, v(0), v(2)),
        vop2(ADD_U32, 1, v(1), v(2)),
        vop1(0x06, 0, v(0)),
        vop1(0x06, 1, v(1)),
        vop2(SUBREV_F32, 0, ONE_F32, v(0)),
        vop2(SUBREV_F32, 1, ONE_F32, v(1)),
        vop1(0x01, 2, inline(0)),
        vop1(0x01, 3, ONE_F32),
        // exp pos0 v0, v1, v2, v3 done
        (0x3E << 26) | (12 << 4) | 0xF | (1 << 11),
        0x0302_0100,
        asm::S_ENDPGM,
    ];
    g.program(&words)
}

/// Indirect draws write their argument offsets to the user-data registers
/// the packet names, and the host draw adds none of them itself: a native
/// vertex program that adds those registers sees each offset exactly once,
/// for automatic and indexed, single and multi-draw packets.
#[test]
#[ignore]
fn pipeline_indirect_draw_offsets_reach_user_sgprs() {
    use crate::render::IndirectRegisters;
    let mut g = Guest::new(32 << 20);
    let vs = offset_vertex_program(&mut g);
    let ps = solid_program(&mut g);
    let words = |w: &[u32]| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let indices = g.alloc(&words(&[0, 1, 2]));
    let auto_args = g.alloc(&words(&[3, 1, 7, 5]));
    let indexed_args = g.alloc(&words(&[3, 1, 0, 7, 5]));
    // Records 24 bytes apart: the first draws no instances, the third is
    // past the count read from memory.
    let multi_args = g.alloc(&words(&[3, 0, 0, 7, 5, 0, 3, 1, 0, 7, 5, 0, 3, 1, 0, 0, 0, 0]));
    let count = g.alloc(&words(&[2]));
    let rest = TestCompiler::new().with(ps, TestProgram::SolidPixel);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(SpirvVertices { spirv: crate::shader::SpirvCompiler::for_device(dev, g.mem), vertices: vec![vs], rest }), None, None);
    let green = color_words([0.0, 1.0, 0.0, 1.0]);
    let regs = IndirectRegisters { vertex_offset: 0x8C, instance_offset: 0x8D, ..IndirectRegisters::default() };
    let mut check_prim = |what: &str, prim: u32, packet: DrawPacket, index: Option<IndexBuffer>, user: [u32; 2], covered: bool| {
        let target = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
        let d = draw_into(target, stage(vs, &user), stage(ps, &green), prim, packet, index);
        assert_eq!(rig.draw(&d), DrawOutcome::Recorded, "{what}");
        let p = rig.pixels(&target);
        let expect = if covered { [0, 255, 0, 255] } else { [0, 0, 0, 0] };
        for (x, y) in [(2, 2), (32, 32), (60, 60)] {
            assert_eq!(px(&p, 64, x, y), expect, "{what} at ({x}, {y})");
        }
    };
    let mut check = |what: &str, packet: DrawPacket, index: Option<IndexBuffer>, user: [u32; 2], covered: bool| check_prim(what, primitive::TRI_LIST, packet, index, user, covered);
    // Direct draws: the program sees the registers the guest wrote.
    check("direct", auto(3, 1), None, [7, 5], true);
    check("direct without offsets", auto(3, 1), None, [0, 0], false);
    // Indirect: stale register values are replaced by the record's.
    let packet = DrawPacket { kind: DrawKind::Indirect, indirect_address: auto_args, indirect_registers: regs, ..DrawPacket::default() };
    check("auto", packet, None, [99, 99], true);
    let index = IndexBuffer { address: indices, kind: IndexType::U32, count: 3 };
    let indexed = DrawPacket { index_base: indices, index_type: 1, index_buffer_size: 3, indirect_registers: regs, ..DrawPacket::default() };
    check("indexed", DrawPacket { kind: DrawKind::IndexIndirect, indirect_address: indexed_args, ..indexed }, Some(index), [99, 99], true);
    let multi = DrawPacket { kind: DrawKind::IndexIndirectMulti, indirect_address: multi_args, indirect_count: 3, count_address: count, stride: 24, ..indexed };
    check("multi", multi, Some(index), [99, 99], true);
    // Without destinations the program cannot see the offsets.
    let packet = DrawPacket { kind: DrawKind::Indirect, indirect_address: auto_args, ..DrawPacket::default() };
    check("no destinations", packet, None, [0, 0], false);
    // Rectangle lists: the corner (7) and its two neighbours span the
    // whole target once the fourth corner is added; the host indirect
    // draw reads the records as they are.
    let packet = DrawPacket { kind: DrawKind::Indirect, indirect_address: auto_args, indirect_registers: regs, ..DrawPacket::default() };
    check_prim("rectangles", primitive::RECT_LIST, packet, None, [99, 99], true);
    check_prim("indexed multi-draw rectangles", primitive::RECT_LIST, multi, Some(index), [99, 99], true);
    let stats = rig.provider.stats();
    assert_eq!(stats.indirect_draws, 6);
    if rig.dev.caps.tessellation_shader || rig.dev.caps.fill_rectangle {
        assert_eq!(stats.gpu_indirect_draws, 6, "every record read on the GPU");
    }
}

/// A multi-draw writes the index of each record to the draw-index register
/// its packet names: a vertex program that looks its per-draw data up by
/// that register sees 0, 1, 2… and not the stale register value. Here the
/// draw index lands in s8, which the program adds to the vertex ID: only
/// record 7 (the one record with instances) at draw index 7 covers the
/// target.
#[test]
#[ignore]
fn pipeline_multi_draw_writes_the_draw_index() {
    use crate::render::{IndirectRegisters, SH_NOP};
    let mut g = Guest::new(32 << 20);
    let vs = offset_vertex_program(&mut g);
    let ps = solid_program(&mut g);
    let words = |w: &[u32]| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let indices = g.alloc(&words(&[0, 1, 2]));
    // Eight records 20 bytes apart; only the last one has instances (first
    // instance 5 for s9).
    let mut records = Vec::new();
    for r in 0..8u32 {
        records.extend_from_slice(&[3, (r == 7) as u32, 0, 0, 5]);
    }
    let multi_args = g.alloc(&words(&records));
    let rest = TestCompiler::new().with(ps, TestProgram::SolidPixel);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(SpirvVertices { spirv: crate::shader::SpirvCompiler::for_device(dev, g.mem), vertices: vec![vs], rest }), None, None);
    let green = color_words([0.0, 1.0, 0.0, 1.0]);
    let index = IndexBuffer { address: indices, kind: IndexType::U32, count: 3 };
    let regs = IndirectRegisters { vertex_offset: SH_NOP, instance_offset: 0x8D, first_index: SH_NOP, draw_index: 0x8C };
    let packet = DrawPacket {
        kind: DrawKind::IndexIndirectMulti,
        index_base: indices,
        index_type: 1,
        index_buffer_size: 3,
        indirect_address: multi_args,
        indirect_count: 8,
        stride: 20,
        indirect_registers: regs,
        ..DrawPacket::default()
    };
    let target = rgba8_target(g.zeroed(64 * 64 * 4), 64, 64);
    let d = draw_into(target, stage(vs, &[99, 99]), stage(ps, &green), primitive::TRI_LIST, packet, Some(index));
    assert_eq!(rig.draw(&d), DrawOutcome::Recorded);
    let p = rig.pixels(&target);
    for (x, y) in [(2, 2), (32, 32), (60, 60)] {
        assert_eq!(px(&p, 64, x, y), [0, 255, 0, 255], "record 7 drawn with draw index 7 at ({x}, {y})");
    }
}

/// Remapped alpha blending reads the logical alpha from the second blend
/// source; guest SRC1 factors make MRT1 the second source.
/// CB_COLOR_ATTRIB.FORCE_DST_ALPHA_1: destination-alpha factors read one.
#[test]
fn forced_opaque_destination_alpha_blend_factors() {
    use ps5shader::resources::PixelAlphaBlendSource as A;
    let mut t = rgba8_target(0x1000, 4, 4);
    // DST_ALPHA / ONE_MINUS_DST_ALPHA for colour, ONE_MINUS_DST_COLOR / DST_COLOR for alpha.
    t.blend = BlendState { enable: true, separate_alpha: true, color_src: 6, color_dst: 7, alpha_src: 9, alpha_dst: 8, ..BlendState::default() };
    let k = state::blend_key(&t, true, A::None);
    assert_eq!((k.src_color, k.dst_color), (vk::BlendFactor::DST_ALPHA.as_raw(), vk::BlendFactor::ONE_MINUS_DST_ALPHA.as_raw()));
    t.attrib |= 1 << 17;
    let k = state::blend_key(&t, true, A::None);
    assert_eq!((k.src_color, k.dst_color), (vk::BlendFactor::ONE.as_raw(), vk::BlendFactor::ZERO.as_raw()));
    assert_eq!((k.src_alpha, k.dst_alpha), (vk::BlendFactor::ZERO.as_raw(), vk::BlendFactor::ONE.as_raw()));
}

#[test]
fn second_blend_source_selection() {
    use ps5shader::resources::PixelAlphaBlendSource as A;
    let mut t = rgba8_target(0x1000, 4, 4);
    // ARGB-style mapping: physical alpha holds logical red.
    t.format.export_mapping = crate::render::formats::ComponentMap::RABG;
    t.blend = BlendState { enable: true, color_src: 4, color_dst: 5, alpha_src: 4, alpha_dst: 5, ..BlendState::default() };
    let colors = [&t];
    assert_eq!(state::pixel_blend_sources(&colors, 9, true), (A::SourceAlpha, false));
    // No device dual-source blending, another exported MRT, or an integer
    // export: blended as programmed.
    assert_eq!(state::pixel_blend_sources(&colors, 9, false), (A::None, false));
    assert_eq!(state::pixel_blend_sources(&colors, 0x99, true), (A::None, false));
    assert_eq!(state::pixel_blend_sources(&colors, 7, true), (A::None, false));
    let k = state::blend_key(&t, true, A::SourceAlpha);
    assert!(k.enable);
    assert_eq!(k.src_color, vk::BlendFactor::SRC1_ALPHA.as_raw());
    assert_eq!(k.dst_color, vk::BlendFactor::ONE_MINUS_SRC1_ALPHA.as_raw());
    // SRC_ALPHA / ONE_MINUS_SRC_ALPHA with a separate ONE / ZERO alpha source.
    t.blend = BlendState { enable: true, separate_alpha: true, color_src: 4, color_dst: 5, alpha_src: 1, alpha_dst: 5, ..BlendState::default() };
    let colors = [&t];
    assert_eq!(state::pixel_blend_sources(&colors, 9, true), (A::SourceAlphaOne, false));
    let k = state::blend_key(&t, true, A::SourceAlphaOne);
    assert_eq!(k.src_color, vk::BlendFactor::SRC1_COLOR.as_raw());
    assert_eq!(k.dst_color, vk::BlendFactor::ONE_MINUS_SRC1_ALPHA.as_raw());
    assert_eq!(k.src_alpha, k.src_color);
    // Guest dual-source factors.
    t.format.export_mapping = crate::render::formats::ComponentMap::IDENTITY;
    t.blend = BlendState { enable: true, color_src: 15, color_dst: 18, alpha_src: 15, alpha_dst: 18, ..BlendState::default() };
    let colors = [&t];
    assert_eq!(state::pixel_blend_sources(&colors, 9, true), (A::None, true));
}

/// A vertex stage that selects the viewport gets all 16 with their own
/// scissors; without clipping the host viewport covers the largest area.
#[test]
fn indexed_viewports_and_clip_disable() {
    let t = rgba8_target(0x1000, 64, 64);
    let mut d = draw_into(t, unbound(), unbound(), primitive::TRI_LIST, auto(3, 1), None);
    d.viewports[3] = d.viewports[0];
    d.viewports[3].x = 16.0;
    d.viewports[3].width = 8.0;
    d.scissors[3] = Rect { x0: 16, y0: 0, x1: 24, y1: 8 };
    let one = state::dynamic_state(&d, 64, 64, vk::Format::UNDEFINED, false, false, false, 1, None);
    assert!(one.indexed.is_none());
    let all = state::dynamic_state(&d, 64, 64, vk::Format::UNDEFINED, false, false, false, 16, None);
    let v = all.indexed.unwrap();
    assert_eq!(v.viewports[0].x, one.viewport.x);
    assert_eq!((v.viewports[3].x, v.viewports[3].width), (16.0, 8.0));
    assert_eq!((v.scissors[3].offset.x, v.scissors[3].extent.width), (16, 8));
    let limits = vk::PhysicalDeviceLimits { max_viewport_dimensions: [32768, 32768], ..Default::default() };
    assert!(!state::clip_space_transform(&d, &limits).enabled);
    d.raster.clip_cntl |= 1 << 16;
    let c = state::clip_space_transform(&d, &limits);
    assert!(c.enabled && c.half_extent_x == 8192.0);
    let extent = state::clip_disabled_extent(&limits);
    let free = state::dynamic_state(&d, 64, 64, vk::Format::UNDEFINED, false, false, false, 1, Some(extent));
    assert_eq!((free.viewport.x, free.viewport.width), (0.0, 16384.0));
}

mod float_ref;
mod exec_ref;
mod emit_opt;
mod image_ref;
mod image_dims;
mod image_sample;
mod image_storage;
mod image_packed;
mod image_store_rounding;
mod ref_device;
mod clock;
mod thread_dims;
mod copy_kernel;
mod tile_classify;
mod stencil_texture;
mod bindless;
mod bda;
mod ngg_mesh;
mod capture_replay;
mod dispatch_sync;
mod wave64_split;
mod draw_bench;
mod slice_bench;
mod memo_invalidation;
mod scalar_calls;
mod pixel_reduction;
mod image_atomics;
mod user_data_tail;
mod gfx_exec;
mod pixel_lanes;
mod pixel_inputs;
mod pixel_lds;
mod vertex_exec;
mod unranged_reads;

/// A 2x target drawn without PA_SC_MODE_CNTL_0.MSAA_ENABLE rasterizes both
/// samples at the pixel center: a pixel whose center a triangle's edge
/// leaves outside gets neither sample, one whose center is inside gets
/// both. With the bit, the standard locations (0.75, 0.75) and
/// (0.25, 0.25) split such a pixel.
/// PA_SC_AA_MASK removes the samples a quad's pixels may not cover (a
/// uniform mask is the pipeline sample mask); PS_ITER_SAMPLE runs the
/// pixel program per sample.
#[test]
#[ignore]
fn pipeline_aa_mask_and_per_sample_shading() {
    let mut g = Guest::new(16 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = solid_program(&mut g);
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0]));
    let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
    let inputs = vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)]);
    let cases: [([u32; 2], u32, bool, std::ops::RangeInclusive<u8>); 5] = [
        ([u32::MAX; 2], 2, false, 255..=255),
        ([0x0001_0001; 2], 2, false, 100..=155),
        ([0; 2], 2, false, 0..=0),
        ([0x0003_0003; 2], 2, true, 255..=255),
        ([0; 2], 1, false, 0..=0),
    ];
    let mut rig = Rig::new(&g, compiler, Some(inputs), None);
    for (mask, samples, iter, green) in cases {
        let t = ColorTarget { samples, ..rgba8_target(g.zeroed(32 * 32 * 4 * 2), 32, 32) };
        let mut d = draw_into(t, stage(vs, &[]), stage(ps, &color_words([0.0, 1.0, 0.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
        d.aa_mask = mask;
        d.ps_iter_sample = iter;
        assert_eq!(rig.draw(&d), DrawOutcome::Recorded);
        let p = if samples > 1 { rig.resolved_pixels(&t) } else { rig.pixels(&t) };
        let c = px(&p, 32, 9, 9);
        assert!(green.contains(&c[1]) && c[0] == 0, "mask {mask:x?} samples {samples}: {c:?}");
    }
}

#[test]
#[ignore]
fn pipeline_multisampled_target_without_msaa_enable_is_centered() {
    let mut g = Guest::new(16 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let ps = solid_program(&mut g);
    // Triangles whose right edge is the vertical line x = 16.4 or 16.6
    // pixels of a 32-pixel-wide target (covering every row near it).
    let edge = |px: f32| px / 16.0 - 1.0;
    let tri = |x: f32| floats(&[-3.0, -1.0, 0.0, 1.0, x, -1.0, 0.0, 1.0, x, 3.0, 0.0, 1.0]);
    let left = g.alloc(&tri(edge(16.4)));
    let right = g.alloc(&tri(edge(16.6)));
    let compiler = Arc::new(TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 }).with(ps, TestProgram::SolidPixel));
    let target = |g: &mut Guest| ColorTarget { samples: 2, ..rgba8_target(g.zeroed(32 * 32 * 4 * 2), 32, 32) };
    let cases = [(left, false, target(&mut g)), (left, true, target(&mut g)), (right, false, target(&mut g))];
    let mut results = Vec::new();
    for (verts, msaa_enable, t) in cases {
        let inputs = vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)]);
        let mut rig = Rig::new(&g, compiler.clone(), Some(inputs), None);
        if !rig.dev.caps.sample_location_counts.contains(vk::SampleCountFlags::TYPE_2) {
            eprintln!("skipped: no VK_EXT_sample_locations for 2 samples");
            return;
        }
        let mut d = draw_into(t, stage(vs, &[]), stage(ps, &color_words([0.0, 1.0, 0.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
        d.msaa_enable = msaa_enable;
        assert_eq!(rig.draw(&d), DrawOutcome::Recorded);
        let p = rig.resolved_pixels(&t);
        assert_eq!(px(&p, 32, 15, 16), [0, 255, 0, 255], "a pixel inside the edge has both samples");
        assert_eq!(px(&p, 32, 17, 16), [0, 0, 0, 0], "a pixel outside the edge has none");
        results.push(px(&p, 32, 16, 16));
    }
    assert_eq!(results[0], [0, 0, 0, 0], "centered: the edge at 16.4 leaves the center of pixel 16 out");
    assert!((100..=155).contains(&results[1][1]) && results[1][0] == 0, "standard locations: one of two samples of pixel 16 ({:?})", results[1]);
    assert_eq!(results[2], [0, 255, 0, 255], "centered: the edge at 16.6 covers the center of pixel 16");
}
