// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Ordering of dispatches and the CPU: barriers only between dispatches
//! whose ranges overlap (read or write after a write, write after a read),
//! a chain of dispatches reading each other's results, early submits of a
//! long slice and write-throughs recorded right after their writer.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored dispatch_sync`.

use super::*;
use ps5shader::testing::raw_buffer_descriptor;

const WORDS: u64 = 64;

fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    (op << 25) | (vdst << 17) | ((vsrc1 & 0xFF) << 9) | src0
}

/// buffer_load_dword vdata, vaddr, srsrc, 0 offen.
fn buffer_load_dword(vdata: u32, vaddr: u32, srsrc: u32) -> [u32; 2] {
    [0xE000_0000 | (0x0C << 18) | (1 << 12), vaddr | (vdata << 8) | ((srsrc / 4) << 16) | (0x80 << 24)]
}

const S_WAITCNT_0: u32 = 0xBF8C_0000;

/// `out[tid] = s4 + tid` through the V# in s0..s3.
fn store_program(g: &mut Guest) -> u64 {
    let store = asm::buffer_store_dword(2, 1, 0);
    g.program(&[vop2(0x1A, 1, 128 + 2, 0), vop2(0x25, 2, 4, 0), store[0], store[1], asm::S_ENDPGM])
}

/// `out[tid] = in[tid] + s8`: `in` through the V# in s0..s3, `out` in s4..s7.
fn add_program(g: &mut Guest) -> u64 {
    let load = buffer_load_dword(2, 1, 0);
    let store = asm::buffer_store_dword(2, 1, 4);
    g.program(&[vop2(0x1A, 1, 128 + 2, 0), load[0], load[1], S_WAITCNT_0, vop2(0x25, 2, 8, 2), store[0], store[1], asm::S_ENDPGM])
}

fn one_wave(cs: u64, user: &[u32]) -> ResolvedDispatch {
    let mut shader = stage(cs, user);
    shader.rsrc2 = (user.len() as u32) << 1;
    ResolvedDispatch { packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() }, shader, rsrc3: 0, threads: [64, 1, 1], start: [0; 3] }
}

fn store(cs: u64, out: u64, k: u32) -> ResolvedDispatch {
    let mut user = raw_buffer_descriptor(out, (WORDS * 4) as u32).to_vec();
    user.push(k);
    one_wave(cs, &user)
}

fn add(cs: u64, input: u64, out: u64, k: u32) -> ResolvedDispatch {
    let mut user = raw_buffer_descriptor(input, (WORDS * 4) as u32).to_vec();
    user.extend_from_slice(&raw_buffer_descriptor(out, (WORDS * 4) as u32));
    user.push(k);
    one_wave(cs, &user)
}

fn words(b: &mut BufferCache, sched: &mut Scheduler, mem: &crate::memory::GuestGpuMemory, addr: u64) -> Vec<u32> {
    assert!(b.download(sched, addr, WORDS * 4));
    let mut out = vec![0u8; (WORDS * 4) as usize];
    assert!(mem.read_backing(addr, &mut out));
    out.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn expect(base: u32) -> Vec<u32> {
    (0..WORDS as u32).map(|i| base + i).collect()
}

/// (compute barriers, full barriers) recorded so far.
fn barriers(s: &Scheduler) -> (u64, u64) {
    let st = s.stats();
    (st.compute_barriers, st.full_barriers)
}

/// Independent dispatches get no barrier between them; a dispatch reading
/// what an earlier one wrote, or writing what an earlier one read or
/// wrote, gets one barrier from the compute stage (no full barrier); the
/// chain reads every result in order (`RPS5E_RANGE_BARRIERS`).
#[test]
#[ignore]
fn dispatch_sync_barriers_follow_overlapping_ranges() {
    let mut g = Guest::new(16 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    let (st, ad) = (store_program(&mut g), add_program(&mut g));
    let (x, y, z, w) = (g.zeroed(4096), g.zeroed(4096), g.zeroed(4096), g.zeroed(4096));
    let round = |rig: &mut Rig, base: u32| -> Vec<(u64, u64)> {
        let mut deltas = Vec::new();
        let mut last = barriers(&rig.sched);
        let mut step = |rig: &mut Rig, d: ResolvedDispatch| {
            assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
            let now = barriers(&rig.sched);
            deltas.push((now.0 - last.0, now.1 - last.1));
            last = now;
        };
        step(rig, store(st, x, base)); // x = base + i
        step(rig, store(st, z, base + 1000)); // independent
        step(rig, add(ad, x, y, 5)); // reads x (RAW)
        step(rig, store(st, x, base + 2000)); // writes x: read by the add (WAR), written before (WAW)
        step(rig, add(ad, z, z, 7)); // z's store is visible since the barriers above; nothing read z
        step(rig, add(ad, x, w, 5)); // reads the second store (RAW)
        deltas
    };
    // Warm: buffers, uploads, pipelines.
    round(&mut rig, 10_000);
    rig.sched.full_barrier();
    let deltas = round(&mut rig, 100);
    assert_eq!(deltas, [(0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (1, 0)], "(compute, full) barriers before each dispatch");
    let mem = g.mem;
    assert_eq!(words(&mut rig.buffers, &mut rig.sched, mem, y), expect(105), "the first add read the first store");
    assert_eq!(words(&mut rig.buffers, &mut rig.sched, mem, w), expect(2105), "the last add read the second store");
    assert_eq!(words(&mut rig.buffers, &mut rig.sched, mem, x), expect(2100), "the second store came after the first add's read");
    assert_eq!(words(&mut rig.buffers, &mut rig.sched, mem, z), expect(1107), "in place, after the store");
}

/// A long run of dispatches through the back-end is submitted before the
/// slice ends (`RPS5E_EARLY_SUBMIT`), with the write-throughs of the small
/// outputs written so far: the first dispatch's output lands at a tick
/// submitted long before the slice ends, so the CPU reads it from guest
/// memory without a download once that tick completed.
#[test]
#[ignore]
fn dispatch_sync_early_submits_carry_write_throughs() {
    let mut g = Guest::new(16 << 20);
    // The plan sees the store through s0..s3; the test compiler emits it.
    let st = asm::buffer_store_dword(1, 0, 0);
    let cs = g.program(&[asm::v_mov_from_sgpr(1, 4), st[0], st[1], asm::S_ENDPGM]);
    let n = 200u64;
    let out = g.zeroed((n * 4096) as usize);
    let dev = GpuDevice::create(false).expect("device");
    let mut b = Backend::new(&dev, g.mem).unwrap();
    crate::backend::attach_faults(&b, g.mem);
    let compiler: Arc<dyn ShaderCompiler> = Arc::new(TestCompiler::new().with(cs, TestProgram::StoreCompute { local_size: 64 }));
    let mut p = ShaderPipelines::new(&dev, g.mem, Some(compiler), None).unwrap();
    p.set_async_compile(false);
    b.pipelines = Box::new(p);
    let first_tick = b.sched.current_tick();
    for i in 0..n {
        let mut d = store(cs, out + i * 4096, i as u32 * 1000);
        d.packet.dispatch_initiator |= 1 << 15;
        b.dispatch(&d);
    }
    assert_eq!(b.stats.dispatches, n);
    assert!(b.stats.early_submits >= 1, "a slice of {n} dispatches is submitted before its end");
    assert!(b.sched.timeline.current_tick() > first_tick + 1);
    // The first outputs were written through by a submitted tick: waiting
    // for it needs no back-end and no download.
    let t = b.tracker.landing(out);
    assert_ne!(t, 0, "the first output is written through");
    assert!(t < b.sched.current_tick(), "submitted before the slice ended");
    b.sched.timeline.wait(t);
    let mut read = 0;
    for i in 0..n {
        let region = out + i * 4096;
        let landed = b.tracker.landing(region);
        if landed == 0 || landed > b.sched.timeline.completed_tick() {
            break;
        }
        for w in [0u64, 17, 63] {
            // SAFETY: a guest page of the test view (readable: written through).
            let v = unsafe { std::ptr::read_volatile((region + w * 4) as *const u32) };
            assert_eq!(v, i as u32 * 1000 + w as u32, "dispatch {i} word {w}");
        }
        read += 1;
    }
    assert!(read >= 1);
    // Eager write-throughs are on by default: every writing
    // dispatch's copies follow it at once, which the landing ticks above
    // already covered; the submit-carried path stays for
    // RPS5E_EAGER_THROUGH=0.
    if std::env::var("RPS5E_EAGER_THROUGH").is_ok_and(|v| v == "0") {
        assert_eq!(b.stats.eager_throughs, 0, "write-throughs follow at the submits (RPS5E_EAGER_THROUGH=1: after each dispatch)");
    } else {
        assert!(b.stats.eager_throughs >= 1, "eager write-throughs on by default");
    }
    b.flush();
    b.sched.finish();
}

/// `out[tid] = s2 + tid` through a V# loaded from the table at the
/// user-data pointer s0:s1 (the host reads the V# words to bind it).
fn table_store_program(g: &mut Guest) -> u64 {
    let store = asm::buffer_store_dword(2, 1, 4);
    g.program(&[0xF408_0000 | (4 << 6), 0x7D << 25, 0xBF8C_C07F, vop2(0x1A, 1, 0x82, 0), vop2(0x25, 2, 2, 0), store[0], store[1], asm::S_ENDPGM])
}

/// A capture for `rps5e-cli replay` (does nothing unless
/// `RPS5E_CAPTURE_SYNC_OUT=<file>`): frames of dependent dispatches and
/// CPU reads of their results. Each frame: four dispatches copy buffer
/// descriptors into tables inside a 2 MiB range they declare written; four
/// dispatches store through those tables (the host reads the descriptor
/// words the GPU wrote); 16 store/add pairs (the add reads the store's
/// output) next to 16 independent stores; an end of pipe; the CPU reads the
/// last add's output and a table (no fault: written through, downloaded).
#[test]
#[ignore]
fn dispatch_sync_capture_benchmark() {
    let Some(path) = std::env::var_os("RPS5E_CAPTURE_SYNC_OUT") else { return };
    let frames = 40u32;
    let mut g = Guest::new(64 << 20);
    let dev = GpuDevice::create(false).expect("device");
    let mut b = Backend::new(&dev, g.mem).unwrap();
    crate::backend::attach_faults(&b, g.mem);
    let mut p = ShaderPipelines::new(&dev, g.mem, Some(Arc::new(crate::shader::SpirvCompiler::for_device(&dev, g.mem))), None).unwrap();
    p.set_async_compile(false);
    b.pipelines = Box::new(p);
    let (st, ad, ts) = (store_program(&mut g), add_program(&mut g), table_store_program(&mut g));
    let region = g.zeroed(2 << 20);
    let outs: Vec<u64> = (0..4).map(|_| g.zeroed(4096)).collect();
    // The tables' sources: a V# of each output, then zeros.
    let sources: Vec<u64> = outs
        .iter()
        .map(|&o| {
            let mut w = raw_buffer_descriptor(o, (WORDS * 4) as u32).to_vec();
            w.resize(WORDS as usize, 0);
            g.alloc(&asm::bytes(&w))
        })
        .collect();
    let pairs: Vec<(u64, u64, u64)> = (0..16).map(|_| (g.zeroed(4096), g.zeroed(4096), g.zeroed(4096))).collect();
    // A copy into the table at `table` whose output V# declares the rest of
    // the 2 MiB region.
    let copy = |input: u64, table: u64| {
        let mut user = raw_buffer_descriptor(input, (WORDS * 4) as u32).to_vec();
        user.extend_from_slice(&raw_buffer_descriptor(table, ((2 << 20) - (table - region)) as u32));
        user.push(0);
        one_wave(ad, &user)
    };
    let tables: Vec<u64> = (0..4u64).map(|i| region + i * 64 * crate::memory::PAGE_BYTES).collect();
    crate::capture::start(g.mem, std::path::Path::new(&path), 0).unwrap();
    for f in 0..frames {
        for (&src, &table) in sources.iter().zip(&tables) {
            b.dispatch(&copy(src, table));
        }
        for (i, &table) in tables.iter().enumerate() {
            b.dispatch(&one_wave(ts, &[table as u32, (table >> 32) as u32, f * 100 + i as u32]));
        }
        for (i, &(x, y, z)) in pairs.iter().enumerate() {
            b.dispatch(&store(st, x, f * 1000 + i as u32));
            b.dispatch(&store(st, z, 7));
            b.dispatch(&add(ad, x, y, 5));
        }
        b.end_of_pipe(Box::new(|| {}));
        b.flush();
        b.sched.finish();
        let y = pairs[15].1;
        // SAFETY: a guest page of the test view (a fault downloads it).
        let v = unsafe { std::ptr::read_volatile((y + 8) as *const u32) };
        assert_eq!(v, f * 1000 + 15 + 5 + 2, "frame {f}");
        // A table inside the large written range (downloaded for the
        // stores through it).
        // SAFETY: as above.
        let w = unsafe { std::ptr::read_volatile((tables[3] + 4) as *const u32) };
        assert_eq!(w, raw_buffer_descriptor(outs[3], (WORDS * 4) as u32)[1], "frame {f}");
        let desc = crate::SurfaceDesc { width: 64, height: 64, pitch_pixels: 64, pixel_format: crate::format::A8B8G8R8_SRGB, tiling: crate::TILING_LINEAR };
        crate::capture::flip_begin(&desc, outs[0], false);
        crate::capture::flip_end();
    }
    b.flush();
    for (i, &o) in outs.iter().enumerate() {
        assert_eq!(words(&mut b.buffers, &mut b.sched, g.mem, o), expect((frames - 1) * 100 + i as u32), "store through table {i}");
    }
    eprintln!("{}", crate::capture::finish().unwrap());
    let s = b.sched.stats();
    eprintln!(
        "submits {} full barriers {} compute barriers {} downloads {} early submits {}",
        s.submits,
        s.full_barriers,
        s.compute_barriers,
        b.buffers.stats().downloads,
        b.stats.early_submits
    );
}

/// A dispatch storing at its thread index through a V# that declares
/// 2 MiB makes only the bytes its threads reach GPU-dirty; the rest of the
/// range stays CPU-readable (`RPS5E_WRITE_EXTENT`). A store the analysis
/// cannot bound (its offset loaded from memory) keeps the whole range.
#[test]
#[ignore]
fn dispatch_sync_bounded_stores_dirty_only_their_bytes() {
    let mut g = Guest::new(32 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    let st = store_program(&mut g);
    // The same store at an offset read from the V# table at s0:s1 + 16.
    let unbounded = {
        let store = asm::buffer_store_dword(2, 1, 4);
        let mov = 0x7E00_0000 | (1 << 17) | (1 << 9) | 12; // v_mov_b32 v1, s12
        g.program(&[0xF408_0000 | (4 << 6), 0x7D << 25, 0xF400_0000 | (12 << 6), (0x7D << 25) | 16, 0xBF8C_C07F, mov, vop2(0x25, 2, 2, 0), store[0], store[1], asm::S_ENDPGM])
    };
    let big = 2u64 << 20;
    let out = g.zeroed(big as usize);
    let mut user = raw_buffer_descriptor(out, big as u32).to_vec();
    user.push(500);
    assert_eq!(rig.dispatch(&one_wave(st, &user)), DrawOutcome::Recorded);
    assert!(rig.buffers.has_gpu_dirty(out, 4) && rig.buffers.has_gpu_dirty(out + 63 * 4, 4), "the stored words");
    assert!(!rig.buffers.has_gpu_dirty(out + 64 * 1024, 4), "bytes no thread stores to");
    assert!(!rig.buffers.has_gpu_dirty(out + big - 4, 4));
    assert_eq!(rig.provider.stats().bounded_writes, 1);
    assert_eq!(words(&mut rig.buffers, &mut rig.sched, g.mem, out), expect(500));
    // Unbounded: the offset (0) comes from memory.
    let out2 = g.zeroed(big as usize);
    let mut table = raw_buffer_descriptor(out2, big as u32).to_vec();
    table.extend_from_slice(&[0, 0, 0, 0]);
    let t = g.alloc(&asm::bytes(&table));
    assert_eq!(rig.dispatch(&one_wave(unbounded, &[t as u32, (t >> 32) as u32, 7])), DrawOutcome::Recorded);
    assert!(rig.buffers.has_gpu_dirty(out2 + big - 4, 4), "a store the analysis cannot bound keeps the whole range");
    assert_eq!(rig.provider.stats().bounded_writes, 1);
    assert!(rig.buffers.download(&mut rig.sched, out2, 4));
    let mut w = [0u8; 4];
    assert!(g.mem.read_backing(out2, &mut w));
    assert!((7..7 + 64).contains(&u32::from_le_bytes(w)), "every lane stores to offset 0");
}
