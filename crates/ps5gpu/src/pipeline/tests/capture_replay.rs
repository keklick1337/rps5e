// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Frame capture and replay (`crate::capture`): a
//! small frame through `Backend` (two textured draws with a CPU write to
//! the texture between them, a compute dispatch through the SPIR-V
//! compiler, a GPU copy of its output, a flip) is captured; guest memory is
//! then replaced by a fresh mapping at the same addresses full of garbage
//! and the capture is replayed into a new back-end. The rendered and
//! written bytes are identical, in every loop.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored capture_replay`.
//! `RPS5E_CAPTURE_TEST_OUT=<file>` keeps a copy of the capture (for
//! trying `rps5e-cli replay` on it).

use super::*;
use crate::capture::{self, format::Capture, replay};
use crate::memory::GuestGpuMemory;
use crate::SurfaceDesc;

const LEN: usize = 16 << 20;
const W: u32 = 64;

/// Guest addresses of the workload.
#[derive(Clone, Copy)]
struct Frame {
    vs: u64,
    ps: u64,
    cs: u64,
    tex: u64,
    verts: u64,
    a: u64,
    b: u64,
    region: u64,
    copy: u64,
}

fn texture(blue: u8) -> Vec<u8> {
    // 4x4 linear RGBA8 (row pitch 64 texels).
    let mut tex = vec![0u8; 64 * 4 * 4];
    for y in 0..4u32 {
        for x in 0..4u32 {
            let o = ((y * 64 + x) * 4) as usize;
            tex[o..o + 4].copy_from_slice(&[(x * 60) as u8, (y * 60) as u8, blue, 255]);
        }
    }
    tex
}

fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    (op << 25) | (vdst << 17) | ((vsrc1 & 0xFF) << 9) | src0
}

fn setup(g: &mut Guest) -> Frame {
    let vs = g.program(&[asm::S_ENDPGM]);
    let sample = asm::image_sample(0, 0, 0, 8);
    let ps = g.program(&[sample[0], sample[1], asm::S_ENDPGM]);
    // word[tid] = s4 + tid: v1 = tid << 2, v2 = s4 + tid, store v2 at v1.
    let store = asm::buffer_store_dword(2, 1, 0);
    let cs = g.program(&[vop2(0x1A, 1, 128 + 2, 0), vop2(0x25, 2, 4, 0), store[0], store[1], asm::S_ENDPGM]);
    let tex = g.alloc(&texture(200));
    let verts = g.alloc(&floats(&[
        -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
        1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, //
        -1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
    ]));
    let a = g.zeroed((W * W * 4) as usize);
    let b = g.zeroed((W * W * 4) as usize);
    let region = g.zeroed(4096);
    let copy = g.zeroed(4096);
    Frame { vs, ps, cs, tex, verts, a, b, region, copy }
}

fn backend(mem: &'static GuestGpuMemory, f: &Frame) -> Backend {
    let dev = GpuDevice::create(false).expect("device");
    let mut b = Backend::new(&dev, mem).unwrap();
    crate::backend::attach_faults(&b, mem);
    let rest = TestCompiler::new().with(f.vs, TestProgram::PassthroughVertex { inputs: 2 }).with(f.ps, TestProgram::TexturedPixel);
    let compiler: Arc<dyn ShaderCompiler> = Arc::new(SpirvVertices { spirv: crate::shader::SpirvCompiler::for_device(&dev, mem), vertices: vec![f.cs], rest });
    let mut p = ShaderPipelines::new(&dev, mem, Some(compiler), None).unwrap();
    p.set_async_compile(false);
    p.set_vertex_input_resolver(Some(Box::new(FixedInputs(vec4_attrs(&[(f.verts, 96, 32, false)], &[(0, 0, 0), (1, 0, 16)])))));
    b.pipelines = Box::new(p);
    b
}

fn textured(f: &Frame, target: u64) -> ResolvedDraw {
    let t_sharp = [
        (f.tex >> 8) as u32,
        ((f.tex >> 40) as u32 & 0xFF) | (56 << 20) | (3 << 30),
        3 << 14,
        0xFAC | (sharps::image_type::TYPE_2D << 28),
        0,
        0,
        0,
        0,
    ];
    let s_sharp = [2 | (2 << 3) | (2 << 6), 0, 0, 0];
    let mut user = t_sharp.to_vec();
    user.extend_from_slice(&s_sharp);
    draw_into(rgba8_target(target, W, W), stage(f.vs, &[]), stage(f.ps, &user), primitive::RECT_LIST, auto(3, 1), None)
}

fn store_dispatch(f: &Frame) -> ResolvedDispatch {
    let v_sharp = [f.region as u32, (f.region >> 32) as u32, 4096, (3 << 28) | (20 << 12) | 0xFAC];
    let mut user = v_sharp.to_vec();
    user.push(1000);
    let mut shader = stage(f.cs, &user);
    shader.rsrc2 = (user.len() as u32) << 1;
    ResolvedDispatch { packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() }, shader, rsrc3: 0, threads: [64, 1, 1], start: [0; 3] }
}

fn display(w: u32) -> SurfaceDesc {
    SurfaceDesc { width: w, height: w, pitch_pixels: w, pixel_format: crate::format::A8B8G8R8_SRGB, tiling: crate::TILING_LINEAR }
}

/// Guest bytes of `[addr, addr + size)` once the GPU's results reached them.
fn guest_bytes(b: &mut Backend, mem: &GuestGpuMemory, addr: u64, size: u64) -> Vec<u8> {
    b.images.writeback_range(&mut b.sched, &mut b.buffers, addr, size);
    b.buffers.download(&mut b.sched, addr, size);
    let mut out = vec![0u8; size as usize];
    assert!(mem.read_backing(addr, &mut out));
    out
}

fn outputs(b: &mut Backend, mem: &GuestGpuMemory, f: &Frame) -> [Vec<u8>; 4] {
    let target = (W * W * 4) as u64;
    [guest_bytes(b, mem, f.a, target), guest_bytes(b, mem, f.b, target), guest_bytes(b, mem, f.region, 256), guest_bytes(b, mem, f.copy, 256)]
}

/// A fresh memfd mapped at `[view, view + LEN)` (the old view unmapped),
/// registered with a new memory manager and filled with garbage.
fn remap(old: &GuestGpuMemory, view: u64) -> &'static GuestGpuMemory {
    old.note_unmapped(view, LEN as u64);
    let mem = crate::testutil::test_memory();
    // SAFETY: the test's own view is replaced by a new mapping of the same
    // size; nothing else uses the range.
    unsafe {
        assert_eq!(libc::munmap(view as *mut libc::c_void, LEN), 0);
        let fd = libc::memfd_create(c"capture-replay".as_ptr(), 0);
        assert!(fd >= 0);
        assert_eq!(libc::ftruncate(fd, LEN as libc::off_t), 0);
        let v = libc::mmap(view as *mut libc::c_void, LEN, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED | libc::MAP_FIXED_NOREPLACE, fd, 0);
        assert_eq!(v as u64, view, "the view's addresses are free again");
        let alias = libc::mmap(std::ptr::null_mut(), LEN, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0);
        assert_ne!(alias, libc::MAP_FAILED);
        std::ptr::write_bytes(alias as *mut u8, 0xA5, LEN);
        mem.set_backing_alias(alias as u64, LEN as u64);
    }
    mem.note_mapped(view, LEN as u64, libc::PROT_READ | libc::PROT_WRITE, Some(0));
    ps5vm::ranges::track(view, LEN as u64, ps5vm::MemProtect::RW);
    mem
}

#[test]
#[ignore]
fn capture_replay_reproduces_the_frame_bytes() {
    let mut g = Guest::new(LEN);
    let view = g.end - LEN as u64;
    let f = setup(&mut g);
    let dir = std::env::temp_dir().join(format!("rps5e-capture-replay-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("frame.ps5cap");

    // The game side: the frame through the back-end while the capture records.
    let mut b = backend(g.mem, &f);
    capture::start(g.mem, &path, 0).unwrap();
    b.draw(&textured(&f, f.a));
    b.flush();
    // The CPU rewrites the texture (blue 200 -> 50) before the next draw
    // reads it: its pages are watched, the write faults.
    let new = texture(50);
    // SAFETY: the texture's bytes in the guest view.
    unsafe { std::ptr::copy_nonoverlapping(new.as_ptr(), f.tex as *mut u8, new.len()) };
    crate::memory::note_sync_point();
    b.draw(&textured(&f, f.b));
    b.dispatch(&store_dispatch(&f));
    assert!(b.copy_memory(f.copy, f.region, 256), "the copy of GPU-written bytes runs on the GPU");
    b.end_of_pipe(Box::new(|| {}));
    capture::flip_begin(&display(W), f.a, true);
    assert!(b.capture_display(&display(W), f.a).is_some(), "the display image is the GPU's");
    capture::flip_end();
    b.flush();
    let summary = capture::finish().unwrap();
    let game = outputs(&mut b, g.mem, &f);
    assert_eq!(b.stats.recorded_draws, 2);
    drop(b);

    let px = |o: &[u8], x: u32, y: u32| o[((y * W + x) * 4) as usize..][..4].to_vec();
    assert_eq!(px(&game[0], 20, 40), [60, 120, 200, 255], "first draw: the original texture");
    assert_eq!(px(&game[1], 20, 40), [60, 120, 50, 255], "second draw: the texture the CPU rewrote");
    let words: Vec<u32> = game[2].chunks(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect();
    assert_eq!(words, (1000..1064).collect::<Vec<u32>>(), "the dispatch through the SPIR-V compiler");
    assert_eq!(game[3], game[2], "the GPU copy");

    let t = summary.totals;
    assert_eq!((t.flips, t.draws, t.dispatches), (1, 2, 1));
    assert!(t.cpu_accesses >= 1, "the texture write fault is recorded: {t:?}");
    assert!(t.patches >= 4, "{t:?}");
    if let Some(out) = std::env::var_os("RPS5E_CAPTURE_TEST_OUT") {
        std::fs::copy(&path, out).unwrap();
    }

    // The replay: new memory at the same addresses (garbage), a new back-end.
    let mem = remap(g.mem, view);
    let cap = Capture::open(&path).unwrap();
    assert_eq!(cap.totals, Some(t));
    let mut space = replay::Space::existing(mem, &cap);
    let mut b2 = backend(mem, &f);
    let opts = replay::Options { loops: 1, present: true, dump_dir: None, slices: 0 };
    let report = replay::run(&cap, &mut space, &mut replay::BackendSink(&mut b2), &opts);
    assert_eq!((report.frames, report.draws, report.dispatches), (1, 2, 1));
    assert_eq!(report.loops[0].transfer_mismatches, 0);
    assert_eq!(report.failed_writes, 0);
    let replayed = outputs(&mut b2, mem, &f);
    for (i, (a, b)) in game.iter().zip(&replayed).enumerate() {
        assert!(a == b, "output {i} differs after the replay (first byte {:?})", a.iter().zip(b).position(|(x, y)| x != y));
    }

    // Two more loops over the same back-end (warm caches) and one with
    // dumps, in slice mode: still the same bytes, and the dumped frame is
    // target A.
    let dumps = dir.join("frames");
    let opts = replay::Options { loops: 2, present: true, dump_dir: Some(dumps.clone()), slices: 1 };
    let report = replay::run(&cap, &mut space, &mut replay::BackendSink(&mut b2), &opts);
    assert_eq!(report.loops.len(), 2);
    assert_eq!(report.loops[1].draws, 2);
    assert!(report.loops.iter().all(|l| l.slices >= 1));
    let warm = outputs(&mut b2, mem, &f);
    assert!(warm == game, "outputs differ after warm loops");
    assert_eq!(report.dumped.len(), 1);
    let ppm = std::fs::read(&report.dumped[0]).unwrap();
    let header = format!("P6\n{W} {W}\n255\n");
    assert_eq!(&ppm[..header.len()], header.as_bytes());
    let rgb = &ppm[header.len()..];
    let o = ((40 * W + 20) * 3) as usize;
    assert_eq!(&rgb[o..o + 3], &[60, 120, 200], "the dumped frame shows target A");
    drop(b2);
    std::fs::remove_dir_all(&dir).ok();
}

/// A capture for trying `rps5e-cli replay` by hand (does nothing unless
/// `RPS5E_CAPTURE_BENCH_OUT=<file>`): one frame of 2000 compute
/// dispatches through the SPIR-V compiler, each with its own constants and
/// output range, a CPU write every 100 dispatches, and a GPU copy of the
/// results.
#[test]
#[ignore]
fn capture_replay_synthetic_benchmark() {
    let Some(out) = std::env::var_os("RPS5E_CAPTURE_BENCH_OUT") else { return };
    let mut g = Guest::new(LEN);
    let f = setup(&mut g);
    let results = g.zeroed(2000 * 256);
    let copy = g.zeroed(2000 * 256);
    let mut b = backend(g.mem, &f);
    capture::start(g.mem, std::path::Path::new(&out), 0).unwrap();
    for i in 0..2000u64 {
        if i % 100 == 0 {
            let new = texture(i as u8);
            // SAFETY: the texture's bytes in the guest view (a CPU write).
            unsafe { std::ptr::copy_nonoverlapping(new.as_ptr(), f.tex as *mut u8, new.len()) };
            crate::memory::note_sync_point();
        }
        let mut d = store_dispatch(&Frame { region: results + i * 256, ..f });
        d.shader.user_data[4] = i as u32 * 64;
        b.dispatch(&d);
        if i % 500 == 499 {
            b.flush();
        }
    }
    assert!(b.copy_memory(copy, results, 2000 * 256));
    b.end_of_pipe(Box::new(|| {}));
    capture::flip_begin(&display(W), f.a, false);
    capture::flip_end();
    b.flush();
    eprintln!("{}", capture::finish().unwrap());
}
