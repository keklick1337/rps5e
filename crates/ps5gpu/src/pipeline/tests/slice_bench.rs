// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! A synthetic multi-slice recording benchmark (R7b, GPU_PLAN §67 F1):
//! UE4-shaped draws (another material per few draws: four T#s and an S#
//! from a descriptor table in guest memory, a material V#, an
//! object-constant slice and scene data; passes switching render targets)
//! recorded by K slice threads in waves through [`WavePool`] against the
//! frozen back-end, the segments committed in order on this thread. It
//! measures what the game's capture replay showed: the slice threads' CPU
//! per draw at K = 1, 2, 4, without a capture.
//!
//! Run with
//! `RPS5E_SLICE_BENCH=1 cargo test -p ps5gpu --release -- --ignored slice_record_bench --nocapture`.
//! `RPS5E_SLICE_BENCH_K=1,2,4` (thread counts), `RPS5E_SLICE_BENCH_LOOPS`
//! (default 30), `RPS5E_SLICE_BENCH_MATERIALS` (default 256: four T#s
//! each), `RPS5E_SLICE_BENCH_PROGRAMS` (pixel programs, default 16),
//! `RPS5E_SLICE_BENCH_CONST=<bytes>` (object constants per draw, default
//! 256; with `RPS5E_SLICE_BENCH_WRITES=1` they are copied to staging),
//! `RPS5E_SLICE_BENCH_AB=<n>` (slice affinity off and on in blocks of n
//! loops, medians of each), `RPS5E_SLICE_BENCH_WRITES=1` (the CPU rewrites the object
//! constants before every loop). The slice size and wave are the usual
//! `RPS5E_SLICE_DRAWS` / `RPS5E_SLICE_WAVE`. Per-thread hardware counters:
//! run one K under `scripts/perf/hwcount.py -g`.

use super::*;
use crate::backend::{SegItem, SliceItem, SliceUnit};
use crate::slices::pool::{WavePool, WaveSlice};

const LEN: usize = 256 << 20;
const TEXTURES: u64 = 4;
const PASS_DRAWS: u64 = 400;
const PASSES: u64 = 6;
const RING_SLOTS: u64 = 2400;

fn buffer_load_dword(vdata: u32, vaddr: u32, srsrc: u32) -> [u32; 2] {
    [0xE000_0000 | (0x0C << 18) | (1 << 12), vaddr | (vdata << 8) | ((srsrc / 4) << 16) | (0x80 << 24)]
}

fn s_load(op: u32, sdata: u32, offset: u32) -> [u32; 2] {
    [0xF400_0000 | (op << 18) | (sdata << 6), 0xFA00_0000 | offset]
}

const S_WAITCNT_LGKM0: u32 = 0xBF8C_C07F;
const S_WAITCNT_VM0: u32 = 0xBF8C_3F70;

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

/// A merged (ES) vertex program through the SPIR-V compiler: the
/// full-screen triangle of the vertex ID, loads through the object-constant
/// V# in s[8:11] and the scene V# in s[12:15], position and param 0.
fn vertex_program(g: &mut Guest) -> u64 {
    let mut w = super::ngg_mesh::full_screen_corner();
    // v_mov_b32 v9, 1.0
    w.push((0x3F << 25) | (9 << 17) | (0x01 << 9) | 242);
    // The second object-constant load's offset is data: the whole V# range
    // is bound (copied when its pages are CPU-dirty).
    let l1 = buffer_load_dword(10, 5, 8);
    let l2 = buffer_load_dword(11, 5, 12);
    let l3 = buffer_load_dword(12, 10, 8);
    w.extend_from_slice(&[l1[0], l1[1], l2[0], l2[1], S_WAITCNT_VM0, l3[0], l3[1], S_WAITCNT_VM0]);
    // exp param0 v0, v1, v2, v9; exp pos0 v0, v1, v2, v9 done.
    w.extend_from_slice(&[(0x3E << 26) | (32 << 4) | 0xF, 0x0902_0100, (0x3E << 26) | (1 << 11) | (12 << 4) | 0xF, 0x0902_0100, asm::S_ENDPGM]);
    g.program(&w)
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

/// (minor faults, voluntary and involuntary context switches) of the process.
fn rusage() -> (i64, i64, i64) {
    // SAFETY: valid out pointer.
    let mut r: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut r) };
    (r.ru_minflt, r.ru_nvcsw, r.ru_nivcsw)
}

/// System-wide TLB shootdown interrupts so far.
fn tlb_shootdowns() -> u64 {
    std::fs::read_to_string("/proc/interrupts")
        .ok()
        .and_then(|s| s.lines().find(|l| l.trim_start().starts_with("TLB:")).map(|l| l.split_whitespace().skip(1).filter_map(|x| x.parse::<u64>().ok()).sum()))
        .unwrap_or(0)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

#[derive(Default, Debug)]
struct Outcome {
    waves: u64,
    segments: u64,
    spec_draws: u64,
    conflicts: u64,
    holes: u64,
}

#[test]
#[ignore]
fn slice_record_bench() {
    if std::env::var_os("RPS5E_SLICE_BENCH").is_none() {
        return;
    }
    let _ = env_logger::builder().is_test(true).try_init();
    let ks: Vec<usize> = std::env::var("RPS5E_SLICE_BENCH_K").unwrap_or_else(|_| "1,2,4".into()).split(',').filter_map(|k| k.trim().parse().ok()).collect();
    let loops = env_u64("RPS5E_SLICE_BENCH_LOOPS").unwrap_or(30);
    let writes = std::env::var("RPS5E_SLICE_BENCH_WRITES").is_ok_and(|v| v == "1");
    let mut g = Guest::new(LEN);
    let vs = vertex_program(&mut g);
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
    // Distinct pixel programs (another address each: other modules and
    // pipelines), the same code.
    let programs = env_u64("RPS5E_SLICE_BENCH_PROGRAMS").unwrap_or(16).max(1);
    let materials = env_u64("RPS5E_SLICE_BENCH_MATERIALS").unwrap_or(256).max(1);
    let pss: Vec<u64> = (0..programs).map(|_| g.program(&code)).collect();
    // All textures in one allocation (1 KiB each).
    let all: Vec<u8> = (0..materials * TEXTURES).flat_map(|t| texture_bytes(t as u8)).collect();
    let base = g.alloc(&all);
    let textures: Vec<u64> = (0..materials * TEXTURES).map(|t| base + t * 1024).collect();
    let table_words = tables(&textures);
    let table = g.alloc(&asm::bytes(&table_words));
    let material = g.zeroed((materials * 256) as usize);
    // Object constants: `RPS5E_SLICE_BENCH_CONST` bytes per draw (default 256).
    let cbytes = env_u64("RPS5E_SLICE_BENCH_CONST").unwrap_or(256).clamp(256, 64 << 10).next_multiple_of(256);
    let ring = g.zeroed((RING_SLOTS * cbytes) as usize);
    let scene = g.zeroed(64 << 10);
    let a = g.zeroed(128 * 128 * 4);
    let b_mem = g.zeroed(128 * 128 * 4);
    let c_mem = g.zeroed(128 * 128 * 4);
    let targets = [rgba8_target(a, 128, 128), rgba8_target(b_mem, 128, 128), rgba8_target(c_mem, 128, 128)];

    let dev = GpuDevice::create(false).expect("device");
    let mut b = Backend::new(&dev, g.mem).unwrap();
    crate::backend::attach_faults(&b, g.mem);
    let rest = pss.iter().fold(TestCompiler::new(), |c, &ps| c.with(ps, TestProgram::TexturedPixel));
    let compiler: Arc<dyn ShaderCompiler> = Arc::new(SpirvVertices { spirv: crate::shader::SpirvCompiler::for_device(&dev, g.mem), vertices: vec![vs], rest });
    let mut p = ShaderPipelines::new(&dev, g.mem, Some(compiler), None).unwrap();
    p.set_async_compile(false);
    b.pipelines = Box::new(p);

    let draw = |i: u64, target: ColorTarget| {
        let m = (i / 3 * 7) % materials;
        let ps = pss[((i / 3 * 7) / materials.min(64) % programs) as usize];
        let slot = i % RING_SLOTS;
        let mut vs_user = [0u32; 16];
        vs_user[8..12].copy_from_slice(&raw_buffer(ring + slot * cbytes, cbytes as u32));
        vs_user[12..16].copy_from_slice(&raw_buffer(scene, 64 << 10));
        let mut ps_user = [0u32; 8];
        let tab = table + m * 256;
        ps_user[0] = tab as u32;
        ps_user[1] = (tab >> 32) as u32;
        ps_user[4..8].copy_from_slice(&raw_buffer(material + m * 256, 256));
        draw_into(target, stage(vs, &vs_user), stage(ps, &ps_user), primitive::TRI_LIST, auto(3, 1), None)
    };
    let passes: Vec<(ColorTarget, Vec<ResolvedDraw>)> = (0..PASSES)
        .map(|pass| {
            let t = targets[(pass % 3) as usize];
            (t, (0..PASS_DRAWS).map(|k| draw(pass * PASS_DRAWS + k, t)).collect())
        })
        .collect();
    let total_draws = PASSES * PASS_DRAWS;

    let clear = |b: &mut Backend, t: &ColorTarget| {
        let id = b.images.find_or_create(&mut b.sched, &mut b.buffers, &crate::images::color_key(t)).unwrap();
        b.images.clear_color(&mut b.sched, id, vk::ClearColorValue { float32: [0.0; 4] }, None);
    };
    let rewrite = |frame: u64| {
        let fill: Vec<u8> = (0..RING_SLOTS * cbytes).map(|x| (x as u8).wrapping_add(frame as u8)).collect();
        // SAFETY: the ring's bytes in the guest view (a CPU write).
        unsafe { std::ptr::copy_nonoverlapping(fill.as_ptr(), ring as *mut u8, fill.len()) };
        crate::memory::note_sync_point();
    };
    // The serial frame: compiles, creates and uploads everything.
    let serial_loop = |b: &mut Backend| {
        for (t, draws) in &passes {
            clear(b, t);
            for d in draws {
                b.draw(d);
            }
        }
        b.flush();
        b.sched.flush_and_wait();
    };
    serial_loop(&mut b);
    serial_loop(&mut b);
    let s = b.pipelines.summary().unwrap_or_default();
    assert_eq!(b.stats.recorded_draws, 2 * total_draws, "every warm-up draw is recorded: {s}");
    assert!(b.slices_allowed(), "the provider shares its tables with slice instances");

    // Serial reference.
    let (mut cpu, mut wall) = (Vec::new(), Vec::new());
    for l in 0..loops {
        if writes {
            rewrite(l);
        }
        let t0 = crate::backend::profile::thread_cpu();
        let w0 = std::time::Instant::now();
        serial_loop(&mut b);
        cpu.push((crate::backend::profile::thread_cpu() - t0) as f64 / 1e6);
        wall.push(w0.elapsed().as_secs_f64() * 1e3);
    }
    let serial_cpu = median(&mut cpu) * 1000.0 / total_draws as f64;
    let serial_wall = median(&mut wall);
    eprintln!("slice_record_bench: serial: {serial_cpu:.2} us CPU/draw, {serial_wall:.2} ms wall/loop ({total_draws} draws/loop)");

    let budget = crate::slices::budget() as usize;
    for &k in &ks {
        let pool = WavePool::new(k);
        if let Ok(v) = std::env::var("RPS5E_SLICE_BENCH_AFFINITY") {
            pool.set_affinity(v == "1");
        }
        let per_wave = crate::slices::wave_slices(k);
        let mut out = Outcome::default();
        let (rss0, protect0, tlb0) = (rusage(), crate::memory::guard::PROTECT_CALLS.load(std::sync::atomic::Ordering::Relaxed), tlb_shootdowns());
        // Per warm loop (the first two of a phase fill the slice threads'
        // own memos, recipes and views): slice-thread CPU, commit CPU, wall.
        let (mut busy, mut commits, mut walls) = (Vec::new(), Vec::new(), Vec::new());
        let ab = env_u64("RPS5E_SLICE_BENCH_AB").filter(|&n| n >= 2);
        let (mut ab_busy, mut ab_wall): ([Vec<f64>; 2], [Vec<f64>; 2]) = Default::default();
        std::thread::scope(|scope| {
            pool.spawn(scope);
            for l in 0..loops {
                if writes {
                    rewrite(l);
                }
                // A/B: affinity toggles every `ab` loops; the first loop of
                // a block (caches of the other mode) is not counted.
                let side = ab.map(|n| (l / n) % 2 == 1);
                if let Some(on) = side {
                    pool.set_affinity(on);
                }
                let busy0 = pool.busy_ns();
                let mut commit_ns = 0;
                let w0 = std::time::Instant::now();
                for (t, draws) in &passes {
                    clear(&mut b, t);
                    let slices: Vec<Vec<SliceItem>> = draws.chunks(budget).enumerate().map(|(c, ch)| ch.iter().enumerate().map(|(j, d)| SliceItem::Draw(c * budget + j, d)).collect()).collect();
                    let mut slices = slices.into_iter().peekable();
                    while slices.peek().is_some() {
                        let wave: Vec<WaveSlice> = slices.by_ref().take(per_wave).map(|s| (0, s)).collect();
                        let outs = pool.record(&b, Vec::new(), wave).expect("slice recorders");
                        let c0 = crate::backend::profile::thread_cpu();
                        out.waves += 1;
                        for units in outs {
                            for unit in units {
                                match unit {
                                    SliceUnit::Exact(pos, hole) => {
                                        out.holes += hole.is_some() as u64;
                                        b.draw(&draws[pos]);
                                    }
                                    SliceUnit::Segment(mut seg) => {
                                        let r = pool.with_recorder(seg.worker, |rec| b.commit_segment(rec, &mut seg, &[], &[], true, false)).expect("recorder");
                                        if r.is_ok() {
                                            out.segments += 1;
                                            out.spec_draws += seg.draws() as u64;
                                        } else {
                                            out.conflicts += 1;
                                            for it in &seg.items {
                                                if let SegItem::Draw(dr) = it {
                                                    b.draw(&draws[dr.pos]);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        commit_ns += crate::backend::profile::thread_cpu() - c0;
                    }
                }
                b.flush();
                b.sched.flush_and_wait();
                if l >= 2 {
                    busy.push((pool.busy_ns() - busy0) as f64 / 1e6);
                    commits.push(commit_ns as f64 / 1e6);
                    walls.push(w0.elapsed().as_secs_f64() * 1e3);
                }
                if let (Some(on), Some(n)) = (side, ab) {
                    if l % n != 0 {
                        ab_busy[on as usize].push((pool.busy_ns() - busy0) as f64 / total_draws as f64 / 1000.0);
                        ab_wall[on as usize].push(w0.elapsed().as_secs_f64() * 1e3);
                    }
                }
            }
            pool.stop();
        });
        if std::env::var_os("RPS5E_SLICE_BENCH_SUMMARY").is_some() {
            for w in 0..k {
                let s = pool.with_recorder(w, |r| r.summary().unwrap_or_default()).unwrap_or_default();
                eprintln!("slice_record_bench: K={k} worker {w}: {s}");
            }
        }
        pool.drop_recorders();
        let rss1 = rusage();
        let busy_ms = median(&mut busy);
        eprintln!(
            "slice_record_bench: K={k}: slice threads {:.2} us CPU/draw ({busy_ms:.2} ms/loop), commit {:.2} ms/loop, wall {:.2} ms/loop (medians of {} warm loops); {out:?}; minflt {} vcs {} ivcs {} mprotect {} TLB shootdowns (system) {}",
            busy_ms * 1000.0 / total_draws as f64,
            median(&mut commits),
            median(&mut walls),
            busy.len(),
            rss1.0 - rss0.0,
            rss1.1 - rss0.1,
            rss1.2 - rss0.2,
            crate::memory::guard::PROTECT_CALLS.load(std::sync::atomic::Ordering::Relaxed) - protect0,
            tlb_shootdowns() - tlb0,
        );
        if ab.is_some() {
            let med = median;
            eprintln!(
                "slice_record_bench: K={k} A/B affinity: off {:.2} us/draw {:.2} ms wall, on {:.2} us/draw {:.2} ms wall (medians of {} loops each)",
                med(&mut ab_busy[0]),
                med(&mut ab_wall[0]),
                med(&mut ab_busy[1]),
                med(&mut ab_wall[1]),
                ab_busy[0].len().min(ab_busy[1].len())
            );
        }
        assert!(out.spec_draws > 0, "draws are recorded speculatively: {out:?}");
    }
}
