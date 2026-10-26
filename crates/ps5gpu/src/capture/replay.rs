// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Replay of a frame capture through the GPU back-end, as a benchmark
//!.
//!
//! The guest address space of the capture is mapped again at the same
//! addresses (one memfd as backing, the captured mappings as views), and
//! the records run in order through the entry points the game used:
//! guest memory records are applied as CPU writes where the bytes differ
//! (so the caches see the same invalidations), CPU accesses resolve the
//! same faults, and the back-end items run on the calling thread, which
//! plays the command thread. Every loop ends with the GPU idle; the first
//! one compiles the pipelines and fills the caches, the later ones are
//! warm. A later loop starts from the memory the loop before left, as the
//! game's next frame does: it applies only the records of bytes the CPU
//! rewrote within the capture ([`carried_over`]).

use super::format::{Capture, Event, Record};
use crate::backend::{profile, Backend, GpuExecutor};
use crate::memory::{FaultKind, GuestGpuMemory};
use crate::render::{GdsTransfer, RenderExecutor, ResolvedDispatch, ResolvedDraw};
use crate::slice_shadow::{self as shadow, Kind};
use crate::scheduler::gpu_timer::GpuTime;
use crate::SurfaceDesc;
use std::path::{Path, PathBuf};
use std::time::Instant;

const PAGE: u64 = crate::memory::PAGE_BYTES;

/// GPU time per draw / dispatch label: (label, (ns, items, longest ns)).
pub type ShaderTimes = Vec<(u64, (u64, u64, u64))>;

/// Replay settings.
#[derive(Clone, Debug)]
pub struct Options {
    /// Timed loops over the capture.
    pub loops: u32,
    /// Flips the game took from the back-end (a window) copy the display
    /// image as it did; false: flips only count frames.
    pub present: bool,
    /// After the timed loops, one more loop writes every flipped display
    /// buffer into this folder (`frame_NNNN.ppm`).
    pub dump_dir: Option<PathBuf>,
    /// Slice threads (0: the replaying thread runs every record itself).
    /// With slices a sequencer thread cuts the records into slices as the
    /// command thread cuts its log (`crate::slices`) and hands them over
    /// the same ordered queue; the replaying thread executes them.
    pub slices: u32,
}

impl Default for Options {
    fn default() -> Options {
        Options { loops: 3, present: true, dump_dir: None, slices: 0 }
    }
}

/// Where the replayed items go: the process-wide back-end through the
/// executor the PM4 interpreter uses ([`GlobalSink`]), or a back-end value
/// ([`BackendSink`], tests).
pub trait Sink {
    fn draw(&mut self, d: &ResolvedDraw);
    fn dispatch(&mut self, d: &ResolvedDispatch);
    fn end_of_pipe(&mut self);
    fn after_pipe(&mut self);
    fn memory_filled(&mut self, addr: u64, size: u64, value: u32);
    fn copy_memory(&mut self, dst: u64, src: u64, size: u64) -> bool;
    fn fill_memory(&mut self, dst: u64, size: u64, value: u32) -> bool;
    fn write_memory(&mut self, dst: u64, data: &[u8]) -> bool;
    fn gds(&mut self, op: GdsTransfer) -> bool;
    fn flush(&mut self);
    fn submission_complete(&mut self);
    fn sync_point(&mut self) {
        crate::memory::note_sync_point();
    }
    /// A flip the presenter took from the back-end.
    fn flip(&mut self, desc: &SurfaceDesc, addr: u64);
    /// Submit everything recorded and wait for the GPU.
    fn settle(&mut self);
    /// GPU time so far and per label (`RPS5E_GPU_PROFILE=1`).
    fn gpu(&mut self) -> Option<(GpuTime, ShaderTimes)>;
    /// Write the GPU image of a display buffer back to guest memory.
    fn writeback_display(&mut self, desc: &SurfaceDesc, addr: u64) -> bool;
    /// Wait for background compiles (optimised pipeline rebuilds).
    fn settle_compiles(&mut self) {}
    /// Module and pipeline build times so far.
    fn compile_report(&mut self) -> Option<String> {
        None
    }
}

/// The process-wide back-end, driven as the game drives it.
pub struct GlobalSink;

impl Sink for GlobalSink {
    fn draw(&mut self, d: &ResolvedDraw) {
        GpuExecutor.draw(d);
    }
    fn dispatch(&mut self, d: &ResolvedDispatch) {
        GpuExecutor.dispatch(d);
    }
    fn end_of_pipe(&mut self) {
        GpuExecutor.end_of_pipe(Box::new(|| {}));
    }
    fn after_pipe(&mut self) {
        GpuExecutor.after_pipe(Box::new(|| {}));
    }
    fn memory_filled(&mut self, addr: u64, size: u64, value: u32) {
        GpuExecutor.memory_filled(addr, size, value);
    }
    fn copy_memory(&mut self, dst: u64, src: u64, size: u64) -> bool {
        GpuExecutor.copy_memory(dst, src, size)
    }
    fn fill_memory(&mut self, dst: u64, size: u64, value: u32) -> bool {
        GpuExecutor.fill_memory(dst, size, value)
    }
    fn write_memory(&mut self, dst: u64, data: &[u8]) -> bool {
        GpuExecutor.write_memory(dst, data)
    }
    fn gds(&mut self, op: GdsTransfer) -> bool {
        GpuExecutor.gds_transfer(op)
    }
    fn flush(&mut self) {
        GpuExecutor.flush();
    }
    fn submission_complete(&mut self) {
        GpuExecutor.submission_complete();
    }
    fn sync_point(&mut self) {
        GpuExecutor.sync_point();
    }
    fn flip(&mut self, desc: &SurfaceDesc, addr: u64) {
        let _ = crate::backend::capture_display(desc, addr);
    }
    fn settle(&mut self) {
        crate::backend::with_backend(|b| {
            b.flush();
            b.sched.flush_and_wait();
        });
    }
    fn gpu(&mut self) -> Option<(GpuTime, ShaderTimes)> {
        crate::backend::with_ready_backend(|b| (b.sched.stats().gpu, b.sched.gpu_labels())).filter(|_| profile::on())
    }
    fn writeback_display(&mut self, desc: &SurfaceDesc, addr: u64) -> bool {
        crate::backend::writeback_display(desc, addr)
    }
    fn settle_compiles(&mut self) {
        crate::backend::with_backend(|b| b.pipelines.settle_compiles());
    }
    fn compile_report(&mut self) -> Option<String> {
        crate::backend::with_backend(|b| b.pipelines.compile_report()).flatten()
    }
}

/// A back-end value (tests).
pub struct BackendSink<'a>(pub &'a mut Backend);

impl Sink for BackendSink<'_> {
    fn draw(&mut self, d: &ResolvedDraw) {
        self.0.draw(d);
    }
    fn dispatch(&mut self, d: &ResolvedDispatch) {
        self.0.dispatch(d);
    }
    fn end_of_pipe(&mut self) {
        self.0.end_of_pipe(Box::new(|| {}));
    }
    fn after_pipe(&mut self) {
        self.0.after_pipe(Box::new(|| {}));
    }
    fn memory_filled(&mut self, addr: u64, size: u64, value: u32) {
        self.0.memory_filled(addr, size, value);
    }
    fn copy_memory(&mut self, dst: u64, src: u64, size: u64) -> bool {
        self.0.copy_memory(dst, src, size)
    }
    fn fill_memory(&mut self, dst: u64, size: u64, value: u32) -> bool {
        self.0.fill_memory(dst, size, value)
    }
    fn write_memory(&mut self, dst: u64, data: &[u8]) -> bool {
        self.0.write_memory(dst, data)
    }
    fn gds(&mut self, op: GdsTransfer) -> bool {
        self.0.gds_transfer(op)
    }
    fn flush(&mut self) {
        self.0.flush();
    }
    fn submission_complete(&mut self) {
        self.0.pressure_gc();
    }
    fn flip(&mut self, desc: &SurfaceDesc, addr: u64) {
        let _ = self.0.capture_display(desc, addr);
    }
    fn settle(&mut self) {
        self.0.flush();
        self.0.sched.flush_and_wait();
    }
    fn gpu(&mut self) -> Option<(GpuTime, ShaderTimes)> {
        profile::on().then(|| (self.0.sched.stats().gpu, self.0.sched.gpu_labels()))
    }
    fn writeback_display(&mut self, desc: &SurfaceDesc, addr: u64) -> bool {
        self.0.writeback_display(desc, addr)
    }
    fn settle_compiles(&mut self) {
        self.0.pipelines.settle_compiles();
    }
    fn compile_report(&mut self) -> Option<String> {
        self.0.pipelines.compile_report()
    }
}

/// The guest address space of a replay.
pub struct Space {
    pub mem: &'static GuestGpuMemory,
    /// The backing memfd (None: the mappings exist already, as in tests).
    memfd: Option<Backing>,
    /// Index of the first record after the mappings the capture started with.
    pub body_start: usize,
    scratch: Vec<u8>,
    /// Records that could not be applied (no memory there).
    pub failed_writes: u64,
    pub skipped_maps: u64,
    /// A loop ran: the memory holds the state a loop leaves.
    primed: bool,
}

impl Drop for Space {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(fd) = self.memfd {
            // SAFETY: our own descriptor; the mappings stay valid without it.
            unsafe { libc::close(fd) };
        }
    }
}

fn initial_maps(cap: &Capture) -> usize {
    cap.records.iter().take_while(|r| matches!(r, Record::Event(Event::Map { .. }))).count()
}

impl Space {
    /// The mappings of the capture exist already in `mem` (a test that
    /// captured and replays in one process).
    pub fn existing(mem: &'static GuestGpuMemory, cap: &Capture) -> Space {
        Space { mem, memfd: None, body_start: initial_maps(cap), scratch: Vec::new(), failed_writes: 0, skipped_maps: 0, primed: false }
    }

    /// Map the capture's guest address space for the process-wide memory
    /// manager: a memfd of the capture's backing size, its alias, and the
    /// views the capture started with at their addresses. Before the GPU
    /// device exists (its allocations could take the addresses).
    pub fn create_global(cap: &Capture) -> Result<Space, String> {
        let mem = crate::memory::global();
        if mem.has_backing_alias() {
            return Err("guest memory is mapped already in this process".into());
        }
        let maps = initial_maps(cap);
        let mut size = cap.header_value("alias_bytes").and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        for r in &cap.records {
            if let Record::Event(Event::Map { size: n, backing: Some(b), .. }) = r {
                size = size.max(b + n);
            }
        }
        let size = size.max(PAGE).next_multiple_of(PAGE);
        let (fd, alias) = create_backing(size)?;
        mem.set_backing_alias(alias, size);
        let mut space = Space { mem, memfd: Some(fd), body_start: maps, scratch: Vec::new(), failed_writes: 0, skipped_maps: 0, primed: false };
        for r in &cap.records[..maps] {
            if let Record::Event(Event::Map { addr, size, backing, .. }) = *r {
                space.map(addr, size, backing, true)?;
            }
        }
        Ok(space)
    }

    /// Map `[addr, addr+size)` (the memfd at `backing`, or private memory)
    /// and register it as the memory manager does.
    fn map(&mut self, addr: u64, size: u64, backing: Option<u64>, fresh: bool) -> Result<(), String> {
        let Some(fd) = self.memfd.clone() else {
            self.skipped_maps += 1;
            return Ok(());
        };
        let prot = map_view(&fd, addr, size, backing, fresh)?;
        ps5vm::ranges::track(addr, size, ps5vm::MemProtect::RW);
        self.mem.note_mapped(addr, size, prot, backing);
        Ok(())
    }

    fn unmap(&mut self, addr: u64, size: u64) {
        if self.memfd.is_none() {
            self.skipped_maps += 1;
            return;
        }
        self.mem.note_unmapped(addr, size);
        hold(addr, size);
        ps5vm::ranges::untrack(addr, size);
    }

    /// Memory the capture read outside every guest view (module segments,
    /// stacks): private pages at the same addresses.
    fn ensure_mapped(&mut self, addr: u64, len: u64) {
        if self.memfd.is_none() || self.mem.backing_ptr(addr, len).is_some() || ps5vm::ranges::accessible(addr, len, true) {
            return;
        }
        let first = addr & !(PAGE - 1);
        let end = (addr + len).next_multiple_of(PAGE);
        let mut page = first;
        while page < end {
            if !ps5vm::ranges::accessible(page, PAGE, true) && self.mem.backing_ptr(page, PAGE).is_none() && fresh_page(page) {
                ps5vm::ranges::track(page, PAGE, ps5vm::MemProtect::RW);
            }
            page += PAGE;
        }
    }

    /// Resolve the fault a CPU access of kind `kind` to `addr` takes.
    fn fault(&self, addr: u64, kind: FaultKind) {
        if !self.mem.is_tracked(addr, 1) {
            return;
        }
        for _ in 0..64 {
            if self.mem.pages.allows(addr, kind) {
                return;
            }
            self.mem.try_resolve_fault(kind, addr);
        }
    }

    /// A CPU write of `data` at `addr`: the faults of watched pages first.
    fn write_cpu(&mut self, addr: u64, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        self.ensure_mapped(addr, data.len() as u64);
        let end = addr + data.len() as u64;
        let mut page = addr & !(PAGE - 1);
        while page < end {
            self.fault(page.max(addr), FaultKind::Write);
            page += PAGE;
        }
        if !self.mem.write_backing(addr, data) {
            self.failed_writes += 1;
        }
    }

    /// Guest bytes at `addr` as the capture recorded them: written as the
    /// CPU would when they differ from what memory holds. True when
    /// written.
    fn patch(&mut self, addr: u64, data: &[u8]) -> bool {
        let mut cur = std::mem::take(&mut self.scratch);
        cur.resize(data.len(), 0);
        let same = self.mem.read_backing(addr, &mut cur) && cur == data;
        self.scratch = cur;
        if same {
            return false;
        }
        self.write_cpu(addr, data);
        true
    }

    /// A patch (`data`) or zero fill (None) of `len` bytes at `addr` would
    /// change guest memory.
    fn would_change(&mut self, addr: u64, data: Option<&[u8]>, len: u64) -> bool {
        let mut cur = std::mem::take(&mut self.scratch);
        cur.resize(len as usize, 0);
        let same = self.mem.read_backing(addr, &mut cur)
            && match data {
                Some(d) => cur == d,
                None => cur.iter().all(|&b| b == 0),
            };
        self.scratch = cur;
        !same
    }

    fn zero(&mut self, addr: u64, len: u64) -> bool {
        let mut cur = std::mem::take(&mut self.scratch);
        cur.resize(len as usize, 0);
        let same = self.mem.read_backing(addr, &mut cur) && cur.iter().all(|&b| b == 0);
        if !same {
            cur.fill(0);
            self.write_cpu(addr, &cur);
        }
        self.scratch = cur;
        !same
    }

    fn cpu_fill(&mut self, dst: u64, size: u64, value: u32) {
        const CHUNK: u64 = 1 << 20;
        let pattern: Vec<u8> = (0..CHUNK.min(size)).map(|i| value.to_le_bytes()[(i % 4) as usize]).collect();
        let mut done = 0;
        while done < size {
            let n = (size - done).min(CHUNK);
            self.write_cpu(dst + done, &pattern[..n as usize]);
            done += n;
        }
    }

    fn cpu_copy(&mut self, dst: u64, src: u64, size: u64) {
        const CHUNK: u64 = 1 << 20;
        let mut buf = vec![0u8; CHUNK.min(size) as usize];
        let mut done = 0;
        while done < size {
            let n = (size - done).min(CHUNK) as usize;
            if !self.mem.read_committed(src + done, &mut buf[..n]) {
                self.failed_writes += 1;
            }
            self.write_cpu(dst + done, &buf[..n]);
            done += n as u64;
        }
    }
}

/// The /proc/self/maps lines overlapping `[addr, addr+size)`.
#[cfg(unix)]
fn mapping_at(addr: u64, size: u64) -> String {
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    let hits: Vec<&str> = maps
        .lines()
        .filter(|l| {
            let Some((range, _)) = l.split_once(' ') else { return false };
            let Some((s, e)) = range.split_once('-') else { return false };
            let (Ok(s), Ok(e)) = (u64::from_str_radix(s, 16), u64::from_str_radix(e, 16)) else { return false };
            s < addr + size && addr < e
        })
        .take(4)
        .collect();
    hits.join(" | ")
}

/// Measurements of one loop over the capture.
#[derive(Clone, Debug, Default)]
pub struct LoopStats {
    pub wall_ns: u64,
    /// CPU time of the replaying thread over the loop, and its parts.
    pub thread_ns: u64,
    pub draw_ns: u64,
    pub dispatch_ns: u64,
    /// Other back-end items: flushes, completions, transfers, flips.
    pub other_ns: u64,
    /// Guest memory records and CPU accesses (the game's CPU threads did
    /// this work, not its command thread).
    pub memory_ns: u64,
    /// The final submit and GPU wait.
    pub settle_ns: u64,
    pub draws: u64,
    pub dispatches: u64,
    pub flips: u64,
    pub patches_written: u64,
    pub patch_bytes_written: u64,
    pub cpu_accesses: u64,
    /// Transfers the back-end executed differently from the capture.
    pub transfer_mismatches: u64,
    pub gpu: Option<GpuTime>,
    /// Shaders translated and on-disk module cache hits in the loop.
    pub translated: u64,
    pub translate_ns: u64,
    pub disk_hits: u64,
    /// `RPS5E_GPU_PROFILE` laps over the loop (ns, by `profile::NAMES`).
    pub laps: Vec<u64>,
    /// `RPS5E_SLICE_SHADOW`: the slice analysis of the loop's frames.
    pub shadow: Option<shadow::Totals>,
    /// Slice mode: slices handed over, the sequencer thread's CPU time and
    /// the time it waited for room in the queue.
    pub slices: u64,
    pub seq_ns: u64,
    pub seq_wait_ns: u64,
    /// Slice mode: time the executing thread waited for a slice.
    pub idle_ns: u64,
    /// Waves (stage 2): waves recorded, draws recorded speculatively and
    /// committed, segments committed, segments recorded exactly instead
    /// (conflicts), holes by kind, the slice threads' CPU time, verified
    /// draws and digest mismatches.
    pub waves: u64,
    pub spec_draws: u64,
    pub segments: u64,
    pub conflicts: u64,
    pub conflict_draws: u64,
    pub holes: [u64; crate::slices::spec::Hole::COUNT],
    pub worker_ns: u64,
    pub verified: u64,
    pub mismatches: u64,
    /// Verified draws bound the same ranges from another source (a copy
    /// instead of the mirror or the other way round).
    pub source_diffs: u64,
    /// Conflicts by the read that moved ([`CONFLICT_NAMES`]).
    pub conflict_why: [u64; CONFLICT_NAMES.len()],
}

/// Why a speculative segment was recorded exactly instead.
pub const CONFLICT_NAMES: [&str; 15] = ["poisoned", "pipeline", "cpu", "metadata", "maps", "sync", "pages", "mirror", "upload", "image", "target", "key", "write", "gone", "other"];

/// The outcome of a replay.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub path: String,
    pub frames: u64,
    pub draws: u64,
    pub dispatches: u64,
    pub records: usize,
    pub memory_bytes: u64,
    pub complete: bool,
    pub loops: Vec<LoopStats>,
    /// GPU time per label over the warm loops: (label, ns, items, longest ns).
    pub warm_shaders: Vec<(u64, u64, u64, u64)>,
    pub dumped: Vec<PathBuf>,
    pub failed_writes: u64,
    pub skipped_maps: u64,
    pub summary: Option<String>,
    /// Module and pipeline builds of the first loop, and the wait for
    /// the background (optimised) rebuilds after it.
    pub first_compiles: Option<String>,
    pub settle_compiles_ns: u64,
}

fn laps() -> Vec<u64> {
    profile::flush_local();
    profile::NANOS.iter().map(|n| n.load(std::sync::atomic::Ordering::Relaxed)).collect()
}

fn translations() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let s = &crate::shader::stats::TRANSLATE;
    (s.count.load(Relaxed), s.nanos.load(Relaxed), crate::shader::stats::DISK_HITS.count.load(Relaxed))
}

/// Replay the records of `cap` after its initial mappings `opts.loops`
/// times (and once more with dumps).
pub fn run(cap: &Capture, space: &mut Space, sink: &mut dyn Sink, opts: &Options) -> Report {
    if opts.slices <= 1 {
        return run_with(cap, space, sink, opts, None);
    }
    // Stage 2 (R7b): K slice threads record waves of slices.
    // `RPS5E_SLICE_THREADS`: threads of the pool when not K (diagnostics:
    // one thread measures the speculative path without contention).
    let threads = std::env::var("RPS5E_SLICE_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(opts.slices as usize).max(1);
    let pool = crate::slices::pool::WavePool::new(threads);
    let report = std::thread::scope(|sc| {
        pool.spawn(sc);
        let r = run_with(cap, space, sink, opts, Some(&pool));
        sink.settle();
        pool.stop();
        r
    });
    pool.drop_recorders();
    report
}

fn run_with<'a>(cap: &'a Capture, space: &mut Space, sink: &mut dyn Sink, opts: &Options, pool: Option<&crate::slices::pool::WavePool<'a>>) -> Report {
    let body = &cap.records[space.body_start.min(cap.records.len())..];
    let mut report = Report { records: cap.records.len(), complete: cap.totals.is_some(), ..Report::default() };
    for r in body {
        match r {
            Record::Event(Event::Flip { .. }) => report.frames += 1,
            Record::Event(Event::Draw(_)) => report.draws += 1,
            Record::Event(Event::Dispatch(_)) => report.dispatches += 1,
            Record::Patch { data, .. } => report.memory_bytes += data.len() as u64,
            _ => {}
        }
    }
    let carry = carried_over(body);
    let mut after_first: Option<ShaderTimes> = None;
    for i in 0..opts.loops.max(1) {
        let s = run_loop(cap, body, &carry, space, sink, opts, None, &mut report.dumped, pool);
        if i == 0 {
            after_first = sink.gpu().map(|(_, l)| l);
            report.first_compiles = sink.compile_report();
            let t = Instant::now();
            sink.settle_compiles();
            report.settle_compiles_ns = t.elapsed().as_nanos() as u64;
        }
        report.loops.push(s);
    }
    if let (Some(first), Some((_, last))) = (after_first, sink.gpu()) {
        if opts.loops > 1 {
            let before: crate::fxhash::HashMap<u64, (u64, u64, u64)> = first.into_iter().collect();
            let mut warm: Vec<(u64, u64, u64, u64)> = last
                .into_iter()
                .map(|(l, (ns, n, max))| {
                    let (ns0, n0, _) = before.get(&l).copied().unwrap_or_default();
                    (l, ns - ns0, n - n0, max)
                })
                .filter(|w| w.2 != 0)
                .collect();
            warm.sort_unstable_by_key(|w| std::cmp::Reverse(w.1));
            report.warm_shaders = warm;
        }
    }
    if let Some(dir) = &opts.dump_dir {
        let _ = std::fs::create_dir_all(dir);
        run_loop(cap, body, &carry, space, sink, opts, Some(dir), &mut report.dumped, pool);
    }
    report.failed_writes = space.failed_writes;
    report.skipped_maps = space.skipped_maps;
    report
}

/// The guest memory records a loop after the first one leaves out: the
/// first record of an address that has no later record. Such bytes the
/// CPU never rewrote within the capture, so the next frame of the game
/// found them as the frame before left them (unchanged, or rewritten by
/// the GPU), and so does the next loop. The first record of an address the
/// CPU rewrote later in the capture is applied again (the frame reads that
/// version first).
fn carried_over(body: &[Record]) -> Vec<bool> {
    let addr = |r: &Record| match r {
        Record::Patch { addr, .. } | Record::Zero { addr, .. } => Some(*addr),
        _ => None,
    };
    let mut count: crate::fxhash::HashMap<u64, u32> = Default::default();
    for a in body.iter().filter_map(addr) {
        *count.entry(a).or_default() += 1;
    }
    let mut seen: crate::fxhash::HashSet<u64> = Default::default();
    body.iter().map(|r| addr(r).is_some_and(|a| seen.insert(a) && count[&a] == 1)).collect()
}

#[allow(clippy::too_many_arguments)]
fn run_loop<'a>(
    cap: &'a Capture,
    body: &'a [Record],
    carry: &[bool],
    space: &mut Space,
    sink: &mut dyn Sink,
    opts: &Options,
    dump: Option<&Path>,
    dumped: &mut Vec<PathBuf>,
    pool: Option<&crate::slices::pool::WavePool<'a>>,
) -> LoopStats {
    let present = opts.present;
    let primed = std::mem::replace(&mut space.primed, true);
    let cpu = profile::thread_cpu;
    let laps0 = laps();
    // TEMPORARY diagnostic: what a frame's draws are.
    let histogram = std::env::var_os("RPS5E_REPLAY_DRAW_STATS").is_some();
    let mut groups: std::collections::HashMap<(u64, u64, u32, u32, u8, u64), u64> = Default::default();
    let mut dup_runs = 0u64;
    let mut dup_draws = 0u64;
    let mut prev: Option<(u64, u64, u32, u32, u8, u64)> = None;
    let mut run_len = 0u64;
    // TEMPORARY diagnostic: of consecutive draws of the same
    // (vs, ps) pair, how many user-data words differ — and how many would
    // still differ with the V# base/limit dwords (0,1,2) treated as data.
    let mut same_pair = 0u64;
    let mut any_change = 0u64;
    let mut words_changed = [0u64; 4];
    let mut prev_user: Vec<Vec<u32>> = Vec::new();
    let mut prev_pair: (u64, u64) = (0, 0);

    let gpu0 = sink.gpu().map(|g| g.0);
    let tr0 = translations();
    let mut s = LoopStats::default();
    let wall = Instant::now();
    let thread0 = cpu();
    // `query`: whether the record would change guest memory (CPU writes
    // inside a slice segment), without running it.
    let mut step = |i: usize, s: &mut LoopStats, query: bool| -> bool {
        let r = &body[i];
        if primed && carry[i] {
            return false;
        }
        if query {
            return match r {
                Record::Patch { addr, data } => space.would_change(*addr, Some(cap.data(data)), data.len() as u64),
                Record::Zero { addr, len } => space.would_change(*addr, None, *len),
                _ => true,
            };
        }
        let t = cpu();
        match r {
            Record::Patch { addr, data } => {
                let written = shadow::item(Kind::Patch, 0, || {
                    shadow::changed(*addr, data.len() as u64);
                    space.patch(*addr, cap.data(data))
                });
                if written {
                    s.patches_written += 1;
                    s.patch_bytes_written += data.len() as u64;
                }
                s.memory_ns += cpu() - t;
            }
            Record::Zero { addr, len } => {
                let written = shadow::item(Kind::Patch, 0, || {
                    shadow::changed(*addr, *len);
                    space.zero(*addr, *len)
                });
                if written {
                    s.patches_written += 1;
                    s.patch_bytes_written += len;
                }
                s.memory_ns += cpu() - t;
            }
            Record::Event(e) => match e {
                Event::Map { addr, size, backing, .. } => {
                    shadow::item(Kind::Map, 0, || {
                        shadow::changed(*addr, *size);
                        if let Err(e) = space.map(*addr, *size, *backing, false) {
                            log::warn!("ps5gpu: replay: {e}");
                        }
                    });
                    s.memory_ns += cpu() - t;
                }
                Event::Unmap { addr, size } => {
                    shadow::item(Kind::Map, 0, || {
                        shadow::changed(*addr, *size);
                        space.unmap(*addr, *size);
                    });
                    s.memory_ns += cpu() - t;
                }
                Event::CpuAccess { addr, write } => {
                    shadow::item(Kind::Fault, 0, || space.fault(*addr, if *write { FaultKind::Write } else { FaultKind::Read }));
                    s.cpu_accesses += 1;
                    s.memory_ns += cpu() - t;
                }
                Event::Draw(d) => {
                    if histogram {
                        let mut h = 0x9E3779B97F4A7C15u64;
                        for &w in d.shaders.vertex_program().user_data.iter() {
                            h = (h ^ w as u64).wrapping_mul(0x100000001B3);
                        }
                        for &w in d.shaders.ps.user_data.iter() {
                            h = (h ^ w as u64).wrapping_mul(0x100000001B3);
                        }
                        let key = (
                            d.shaders.vertex_program_address(),
                            if d.shaders.ps.bound() { d.shaders.ps.address } else { 0 },
                            d.packet.count,
                            d.packet.instance_count,
                            d.color_targets.len() as u8,
                            h,
                        );
                        // Which user-data words rotate between consecutive
                        // draws of the same shader pair (VS words, then PS).
                        let cur_user: Vec<Vec<u32>> = vec![d.shaders.vertex_program().user_data.to_vec(), d.shaders.ps.user_data.to_vec()];
                        let pair = (key.0, key.1);
                        if pair == prev_pair {
                            same_pair += 1;
                            let mut diff = 0;
                            for (stage_i, (a, b)) in prev_user.iter().zip(&cur_user).enumerate() {
                                for (wi, (x, y)) in a.iter().zip(b).enumerate() {
                                    if x != y {
                                        diff += 1;
                                        let slot = if wi % 4 == 0 { 0 } else if wi % 4 == 1 { 1 } else if wi % 4 == 2 { 2 } else { 3 };
                                        words_changed[slot] += 1;
                                        let _ = stage_i;
                                    }
                                }
                            }
                            if diff > 0 {
                                any_change += 1;
                            }
                        }
                        prev_user = cur_user;
                        prev_pair = pair;
                        *groups.entry(key).or_default() += 1;
                        if prev == Some(key) {
                            run_len += 1;
                        } else {
                            dup_runs += (run_len > 0) as u64;
                            dup_draws += run_len;
                            run_len = 0;
                        }
                        prev = Some(key);
                    }
                    sink.draw(d);
                    s.draws += 1;
                    s.draw_ns += cpu() - t;
                }
                Event::Dispatch(d) => {
                    sink.dispatch(d);
                    s.dispatches += 1;
                    s.dispatch_ns += cpu() - t;
                }
                other => {
                    match other {
                        Event::SyncPoint => shadow::item(Kind::SyncPoint, 0, || sink.sync_point()),
                        Event::EndOfPipe => sink.end_of_pipe(),
                        Event::AfterPipe => sink.after_pipe(),
                        Event::MemoryFilled { addr, size, value } => sink.memory_filled(*addr, *size, *value),
                        Event::CopyMemory { dst, src, size, gpu } => {
                            let ok = sink.copy_memory(*dst, *src, *size);
                            if !ok {
                                space.cpu_copy(*dst, *src, *size);
                            }
                            s.transfer_mismatches += (ok != *gpu) as u64;
                        }
                        Event::FillMemory { dst, size, value, gpu } => {
                            let ok = sink.fill_memory(*dst, *size, *value);
                            if !ok {
                                space.cpu_fill(*dst, *size, *value);
                            }
                            s.transfer_mismatches += (ok != *gpu) as u64;
                        }
                        Event::WriteMemory { dst, data, gpu } => {
                            let ok = sink.write_memory(*dst, data);
                            if !ok {
                                space.write_cpu(*dst, data);
                            }
                            s.transfer_mismatches += (ok != *gpu) as u64;
                        }
                        Event::Gds(op) => {
                            sink.gds(op.clone());
                        }
                        Event::Flush => sink.flush(),
                        Event::SubmissionComplete => sink.submission_complete(),
                        Event::Flip { desc, addr, gpu } => {
                            s.flips += 1;
                            if histogram {
                                dup_runs += (run_len > 0) as u64;
                                dup_draws += run_len;
                                run_len = 0;
                                prev = None;
                                let mut top: Vec<(&(u64, u64, u32, u32, u8, u64), &u64)> = groups.iter().collect();
                                top.sort_unstable_by_key(|(_, n)| std::cmp::Reverse(**n));
                                let total: u64 = groups.values().sum();
                                eprintln!(
                                    "ps5gpu: replay frame: {} draws, {} distinct (vs,ps,count,inst,targets,userdata), {} in back-to-back duplicate runs ({} runs); same-pair consecutive: {} ({} with word changes; changed words by dword%4: {:?}); top: {:?}",
                                    total,
                                    groups.len(),
                                    dup_draws,
                                    dup_runs,
                                    same_pair,
                                    any_change,
                                    words_changed,
                                    top.iter().take(6).map(|(k, n)| (format!("{:#x}/{:#x} c{} i{} t{}", k.0, k.1, k.2, k.3, k.4), n)).collect::<Vec<_>>()
                                );
                                groups.clear();
                                dup_runs = 0;
                                dup_draws = 0;
                            }
                            shadow::item(Kind::Flip, 0, || {
                                if present && *gpu {
                                    sink.flip(desc, *addr);
                                } else {
                                    // The frame trace counts the frames a flip
                                    // would have presented.
                                    crate::pipeline::trace::frame_presented();
                                }
                            });
                            shadow::frame_end();
                            if let Some(dir) = dump {
                                let path = dir.join(format!("frame_{:04}.ppm", dumped.len() + 1));
                                if dump_frame(space, sink, desc, *addr, &path) {
                                    dumped.push(path);
                                }
                            }
                        }
                        _ => {}
                    }
                    s.other_ns += cpu() - t;
                }
            },
        }
        true
    };
    if let Some(pool) = pool {
        let busy0 = pool.busy_ns();
        drive_waves(cap, body, carry, primed, pool, crate::slices::budget(), &mut step, &mut s);
        s.worker_ns = pool.busy_ns() - busy0;
    } else if opts.slices == 0 {
        for i in 0..body.len() {
            step(i, &mut s, false);
        }
    } else {
        let d = drive_slices(body, crate::slices::budget(), crate::slices::depth(), &mut |i| {
            step(i, &mut s, false);
        });
        (s.slices, s.seq_ns, s.seq_wait_ns, s.idle_ns) = (d.slices, d.seq_ns, d.seq_wait_ns, d.idle_ns);
    }
    let t = Instant::now();
    sink.settle();
    s.settle_ns = t.elapsed().as_nanos() as u64;
    s.thread_ns = cpu() - thread0;
    s.wall_ns = wall.elapsed().as_nanos() as u64;
    s.gpu = match (gpu0, sink.gpu().map(|g| g.0)) {
        (Some(a), Some(b)) => Some(GpuTime { busy_ns: b.busy_ns - a.busy_ns, max_ns: b.max_ns, buffers: b.buffers - a.buffers, long: b.long - a.long }),
        _ => None,
    };
    let tr1 = translations();
    (s.translated, s.translate_ns, s.disk_hits) = (tr1.0 - tr0.0, tr1.1 - tr0.1, tr1.2 - tr0.2);
    s.laps = laps().iter().zip(&laps0).map(|(a, b)| a.wrapping_sub(*b)).collect();
    // Analysed after the timed loop (the frames of the loop flipped).
    s.shadow = shadow::enabled().then(shadow::take);
    s
}

/// What the slice hand-over of one loop cost.
struct SliceRun {
    slices: u64,
    seq_ns: u64,
    seq_wait_ns: u64,
    idle_ns: u64,
}

/// Slice mode of the replay: a sequencer thread cuts the records as the
/// command thread cuts its log (dispatches, end of pipe, flips and
/// submission ends are hard cuts; draws are counted against the budget and
/// cut at a render-target change) and hands the slices (record ranges) over
/// the bounded, ordered slice queue; the calling thread executes them in
/// order with `step`. Guest memory records and CPU accesses stay at their
/// positions between the items.
fn drive_slices(body: &[Record], budget: u32, depth: usize, step: &mut dyn FnMut(usize)) -> SliceRun {
    use crate::slices::cut::Cutter;
    use crate::slices::queue::SliceQueue;
    let queue: SliceQueue<(usize, usize)> = SliceQueue::new(depth);
    std::thread::scope(|sc| {
        let q = &queue;
        let sequencer = std::thread::Builder::new()
            .name("replay-seq".into())
            .spawn_scoped(sc, move || {
                ps5vm::topology::pin_current_thread_to_fast_cores("replay-seq");
                let t0 = profile::thread_cpu();
                let mut cutter = Cutter::new(budget);
                let (mut start, mut seq, mut waited) = (0usize, 0u64, 0u64);
                let mut ship = |end: usize, start: &mut usize, cutter: &mut Cutter| {
                    cutter.reset();
                    if end > *start {
                        seq += 1;
                        waited += q.push(seq, (*start, end)).unwrap_or(0);
                        *start = end;
                    }
                };
                for (i, r) in body.iter().enumerate() {
                    match r {
                        Record::Event(Event::Draw(d)) => {
                            cutter.draw_targets(shadow::target_key(d));
                            if cutter.before_draw().is_some() {
                                ship(i, &mut start, &mut cutter);
                                cutter.before_draw();
                            }
                        }
                        Record::Event(Event::Dispatch(_) | Event::EndOfPipe | Event::Flip { .. } | Event::Flush | Event::SubmissionComplete) => {
                            ship(i + 1, &mut start, &mut cutter)
                        }
                        _ => {}
                    }
                }
                ship(body.len(), &mut start, &mut cutter);
                q.end();
                (seq, profile::thread_cpu() - t0, waited)
            })
            .expect("spawn the replay sequencer");
        let mut idle = 0u64;
        loop {
            let t = Instant::now();
            let Some((n, (a, b))) = queue.pop() else { break };
            idle += t.elapsed().as_nanos() as u64;
            for i in a..b {
                step(i);
            }
            queue.complete(n);
        }
        let (slices, seq_ns, seq_wait_ns) = sequencer.join().unwrap_or((0, 0, 0));
        SliceRun { slices, seq_ns, seq_wait_ns, idle_ns: idle }
    })
}

/// The display buffer of a flip as a PPM: the GPU image written back to
/// guest memory, read, detiled and converted.
fn dump_frame(space: &mut Space, sink: &mut dyn Sink, desc: &SurfaceDesc, addr: u64, path: &Path) -> bool {
    sink.writeback_display(desc, addr);
    let Some(layout) = crate::format::decode(desc.pixel_format) else { return false };
    let bpp = layout.bytes_per_pixel();
    let (mode, pitch) = match desc.tiling {
        crate::TILING_RENDER_TARGET => (crate::tiling::SWIZZLE_64KB_R_X, desc.pitch_pixels.max(desc.width)),
        crate::TILING_LINEAR => (crate::tiling::SWIZZLE_LINEAR, desc.pitch_pixels.max(desc.width)),
        _ => return false,
    };
    let Some(params) = crate::tiling::DetileParams::new(mode, bpp, pitch) else { return false };
    let mut tiled = vec![0u8; params.surface_size(desc.height)];
    if !space.mem.read_committed(addr, &mut tiled) {
        return false;
    }
    let Ok(bgra) = crate::convert_surface(desc, &params, layout, &tiled) else { return false };
    std::fs::write(path, crate::ppm_scaled(desc.width, desc.height, &bgra, 1)).is_ok()
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64;
    (m, var.sqrt())
}

fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    match v.len() {
        0 => 0.0,
        n if n % 2 == 1 => v[n / 2],
        n => (v[n / 2 - 1] + v[n / 2]) / 2.0,
    }
}

fn per(ns: u64, n: u64) -> f64 {
    if n == 0 {
        0.0
    } else {
        ns as f64 / n as f64 / 1e3
    }
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ms = |ns: u64| ns as f64 / 1e6;
        writeln!(
            f,
            "capture {}: {} frames, {} draws, {} dispatches per loop, {} records, {:.1} MiB of guest memory records{}",
            self.path,
            self.frames,
            self.draws,
            self.dispatches,
            self.records,
            self.memory_bytes as f64 / (1 << 20) as f64,
            if self.complete { "" } else { " (INCOMPLETE capture: no END record)" }
        )?;
        writeln!(f, "loop      wall ms    fps  thread ms  draw us  disp us  other ms  memory ms  settle ms  gpu busy ms  translated (ms)  mem writes (MiB)")?;
        for (i, l) in self.loops.iter().enumerate() {
            writeln!(
                f,
                "{:>4} {:>12.2} {:>6.1} {:>10.2} {:>8.2} {:>8.2} {:>9.2} {:>10.2} {:>10.2} {:>12} {:>8} ({:.0}) {:>8} ({:.1})",
                i + 1,
                ms(l.wall_ns),
                if l.wall_ns == 0 { 0.0 } else { l.flips as f64 * 1e9 / l.wall_ns as f64 },
                ms(l.thread_ns),
                per(l.draw_ns, l.draws),
                per(l.dispatch_ns, l.dispatches),
                ms(l.other_ns),
                ms(l.memory_ns),
                ms(l.settle_ns),
                l.gpu.map_or("-".to_string(), |g| format!("{:.2}", ms(g.busy_ns))),
                l.translated,
                ms(l.translate_ns),
                l.patches_written,
                l.patch_bytes_written as f64 / (1 << 20) as f64
            )?;
        }
        let warm: Vec<&LoopStats> = self.loops.iter().skip(1).collect();
        if !warm.is_empty() {
            let col = |g: &dyn Fn(&LoopStats) -> f64| mean_sd(&warm.iter().map(|l| g(l)).collect::<Vec<_>>());
            let pct = |(m, sd): (f64, f64)| if m == 0.0 { 0.0 } else { sd / m * 100.0 };
            let wall = col(&|l| ms(l.wall_ns));
            let draw = col(&|l| per(l.draw_ns, l.draws));
            let disp = col(&|l| per(l.dispatch_ns, l.dispatches));
            let rec = col(&|l| ms(l.draw_ns + l.dispatch_ns + l.other_ns));
            let frames = self.frames.max(1) as f64;
            // The game's CPU threads wrote guest memory; here the replaying
            // thread does, between the items.
            let bare = col(&|l| ms(l.wall_ns.saturating_sub(l.memory_ns)));
            writeln!(
                f,
                "warm loops 2..{}: wall {:.2} ms ±{:.1}% (median {:.2} ms; {:.1} ms/frame, {:.1} fps; {:.2} ms ±{:.1}% without the memory replay) | recording thread {:.2} ms ±{:.1}% (draw {:.2} us ±{:.1}%, dispatch {:.2} us ±{:.1}%){}",
                self.loops.len(),
                wall.0,
                pct(wall),
                median(&warm.iter().map(|l| ms(l.wall_ns)).collect::<Vec<_>>()),
                wall.0 / frames,
                if wall.0 == 0.0 { 0.0 } else { frames * 1e3 / wall.0 },
                bare.0,
                pct(bare),
                rec.0,
                pct(rec),
                draw.0,
                pct(draw),
                disp.0,
                pct(disp),
                if warm.iter().all(|l| l.gpu.is_some()) {
                    let g = col(&|l| ms(l.gpu.map_or(0, |g| g.busy_ns)));
                    format!(" | GPU busy {:.2} ms ±{:.1}% ({:.0}% of wall)", g.0, pct(g), if wall.0 == 0.0 { 0.0 } else { g.0 / wall.0 * 100.0 })
                } else {
                    String::new()
                }
            )?;
            let first = &self.loops[0];
            writeln!(
                f,
                "first loop (compiles, fills the caches): {:+.2} ms wall over the warm mean; {} shaders translated in {:.0} ms, {} from the module cache",
                ms(first.wall_ns) - wall.0,
                first.translated,
                ms(first.translate_ns),
                first.disk_hits
            )?;
            if warm.iter().all(|l| l.slices != 0) {
                let n = warm.len() as f64;
                let sum = |g: &dyn Fn(&LoopStats) -> u64| warm.iter().map(|l| g(l)).sum::<u64>() as f64 / n;
                writeln!(
                    f,
                    "slices: {:.0} per warm loop ({:.1} draws/slice) | sequencer thread {:.2} ms (waited for room {:.2} ms) | executing thread {:.2} ms CPU, waited for slices {:.2} ms",
                    sum(&|l| l.slices),
                    self.draws as f64 / sum(&|l| l.slices).max(1.0),
                    sum(&|l| l.seq_ns) / 1e6,
                    sum(&|l| l.seq_wait_ns) / 1e6,
                    sum(&|l| l.thread_ns) / 1e6,
                    sum(&|l| l.idle_ns) / 1e6
                )?;
            }
            if warm.iter().all(|l| l.waves != 0) {
                let n = warm.len() as f64;
                let sum = |g: &dyn Fn(&LoopStats) -> u64| warm.iter().map(|l| g(l)).sum::<u64>() as f64 / n;
                let holes: Vec<String> = crate::slices::spec::Hole::NAMES
                    .iter()
                    .enumerate()
                    .filter_map(|(k, name)| {
                        let v = sum(&|l| l.holes[k]);
                        (v > 0.0).then(|| format!("{name}:{v:.0}"))
                    })
                    .collect();
                let why: Vec<String> = CONFLICT_NAMES
                    .iter()
                    .enumerate()
                    .filter_map(|(k, name)| {
                        let v = sum(&|l| l.conflict_why[k]);
                        (v > 0.0).then(|| format!("{name}:{v:.0}"))
                    })
                    .collect();
                writeln!(
                    f,
                    "waves: {:.0} per warm loop | speculative draws {:.0} in {:.0} segments | conflicts {:.0} segments ({:.0} draws) [{}] | holes {:.0} [{}] | slice threads {:.2} ms CPU{}",
                    sum(&|l| l.waves),
                    sum(&|l| l.spec_draws),
                    sum(&|l| l.segments),
                    sum(&|l| l.conflicts),
                    sum(&|l| l.conflict_draws),
                    why.join(" "),
                    sum(&|l| l.holes.iter().sum()),
                    holes.join(" "),
                    sum(&|l| l.worker_ns) / 1e6,
                    if warm.iter().any(|l| l.verified != 0) { format!(" | verify: {:.0} draws, {:.0} mismatches, {:.0} with other binding sources", sum(&|l| l.verified), sum(&|l| l.mismatches), sum(&|l| l.source_diffs)) } else { String::new() }
                )?;
            }
            if let Some(c) = &self.first_compiles {
                writeln!(f, "first loop builds: {c}; background rebuilds settled {:.0} ms after it", ms(self.settle_compiles_ns))?;
            }
            if warm.iter().all(|l| l.laps.iter().any(|&x| x != 0)) {
                let n = warm.len() as f64;
                let detailed = profile::detailed();
                let laps: Vec<String> = profile::NAMES
                    .iter()
                    .enumerate()
                    .filter(|&(k, _)| k != 34)
                    .filter(|(k, _)| *k < profile::BASE_SPANS || detailed)
                    .filter_map(|(k, name)| {
                        let total: u64 = warm.iter().map(|l| l.laps.get(k).copied().unwrap_or(0)).sum();
                        (total != 0).then(|| format!("{name}={:.2}", total as f64 / n / 1e6))
                    })
                    .collect();
                writeln!(f, "profile laps, ms per warm loop: {}", laps.join(" "))?;
            }
            if warm.iter().all(|l| l.shadow.is_some()) {
                let mut t = shadow::Totals::default();
                for l in &warm {
                    if let Some(x) = &l.shadow {
                        t.add(x);
                    }
                }
                let gpu = warm.iter().all(|l| l.gpu.is_some()).then(|| col(&|l| ms(l.gpu.map_or(0, |g| g.busy_ns))).0 / frames);
                write!(f, "{}", shadow::report(&t, gpu))?;
            }
            if !self.warm_shaders.is_empty() {
                let n = warm.len() as u64;
                use crate::scheduler::gpu_timer::{label, label_text};
                let fmt = |&(l, ns, k, max): &(u64, u64, u64, u64)| format!("{}:{:.2}ms/{}/{:.2}ms", label_text(l), ms(ns / n), k / n, ms(max));
                let (cats, shaders): (Vec<_>, Vec<_>) = self.warm_shaders.iter().partition(|w| w.0 < label::CATEGORIES);
                let top: Vec<String> = shaders.iter().take(15).map(|w| fmt(w)).collect();
                writeln!(f, "GPU time per warm loop by shader (label:ms/items/longest): {}", top.join(" "))?;
                let sum = |v: &[(u64, u64, u64, u64)]| v.iter().map(|w| w.1).sum::<u64>() / n;
                let (cat_ns, shader_ns) = (sum(&cats), sum(&shaders));
                let listed: Vec<String> = cats.iter().map(|w| fmt(w)).collect();
                writeln!(
                    f,
                    "GPU time per warm loop by segment: {:.2} ms = draws/dispatches {:.2} ms + other work {:.2} ms ({})",
                    ms(cat_ns + shader_ns),
                    ms(shader_ns),
                    ms(cat_ns),
                    listed.join(" ")
                )?;
            }
        }
        if !self.dumped.is_empty() {
            writeln!(f, "frames dumped: {} ({} ...)", self.dumped.len(), self.dumped[0].display())?;
        }
        if self.failed_writes != 0 || self.skipped_maps != 0 {
            writeln!(f, "guest memory records not applied: {} (no memory there), mapping records skipped: {}", self.failed_writes, self.skipped_maps)?;
        }
        let mismatches: u64 = self.loops.iter().map(|l| l.transfer_mismatches).sum();
        if mismatches != 0 {
            writeln!(f, "transfers the back-end executed differently from the capture: {mismatches}")?;
        }
        if let Some(s) = &self.summary {
            writeln!(f, "[gpu] {s}")?;
        }
        Ok(())
    }
}

#[cfg(unix)]
extern "C" fn replay_segv(sig: libc::c_int, info: *mut libc::siginfo_t, uc: *mut libc::c_void) {
    // SAFETY: kernel-provided signal context.
    unsafe {
        let uc = uc as *mut libc::ucontext_t;
        let err = (*uc).uc_mcontext.gregs[libc::REG_ERR as usize] as u64;
        let addr = (*info).si_addr() as u64;
        if (*info).si_code == 2 && crate::memory::handle_fault(addr, err & 2 != 0) {
            return;
        }
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = libc::SIG_DFL;
        libc::sigaction(sig, &sa, std::ptr::null_mut());
    }
}

/// Replay the capture file at `path` through the process-wide back-end:
/// map its guest memory, create the device headless, run the loops.
pub fn replay_file(path: &Path, opts: &Options) -> Result<Report, String> {
    let cap = Capture::open(path).map_err(|e| e.to_string())?;
    // The replaying thread records like the game's command thread, pinned
    // the same way.
    ps5vm::topology::pin_current_thread_to_fast_cores("replay");
    if cap.totals.is_none() {
        log::warn!("ps5gpu: {}: the capture has no END record (interrupted); replaying what it holds", path.display());
    }
    let mut space = Space::create_global(&cap)?;
    install_fault_handler();
    if crate::backend::with_backend(|_| ()).is_none() {
        return Err("no GPU back-end (no Vulkan device?)".into());
    }
    let mut report = run(&cap, &mut space, &mut GlobalSink, opts);
    report.path = path.display().to_string();
    if profile::on() {
        report.summary = crate::backend::summary();
    }
    Ok(report)
}

/// How a record takes part in a wave.
enum WaveRole<'a> {
    Draw(&'a crate::render::ResolvedDraw),
    /// A CPU write of guest memory: (address, size, bytes).
    Cpu(u64, u64, crate::backend::CpuBytes<'a>),
    /// Run exactly at its position inside the wave.
    Exact,
    /// Ends the wave (run alone between waves).
    Boundary,
}

fn wave_role<'a>(cap: &'a Capture, r: &'a Record) -> WaveRole<'a> {
    use crate::backend::CpuBytes;
    match r {
        Record::Event(Event::Draw(d)) => WaveRole::Draw(d),
        Record::Patch { addr, data } => WaveRole::Cpu(*addr, data.len() as u64, CpuBytes::Data(cap.data(data))),
        Record::Zero { addr, len } => WaveRole::Cpu(*addr, *len, CpuBytes::Zero),
        Record::Event(Event::CpuAccess { addr, write: true }) => WaveRole::Cpu(*addr & !(crate::memory::PAGE_BYTES - 1), crate::memory::PAGE_BYTES, CpuBytes::Fault),
        Record::Event(
            Event::SyncPoint
            | Event::AfterPipe
            | Event::MemoryFilled { .. }
            | Event::CopyMemory { .. }
            | Event::FillMemory { .. }
            | Event::WriteMemory { .. }
            | Event::Flush
            | Event::SubmissionComplete,
        ) => WaveRole::Exact,
        _ => WaveRole::Boundary,
    }
}

/// Stage 2 of the replay (R7b): the records are cut into slices (soft cuts
/// near `budget` draws at a render-target change, as the command thread
/// cuts its log); consecutive slices form a wave (`RPS5E_SLICE_WAVE` per
/// thread), ended early by
/// dispatches, end of pipe, flips, mappings, GDS and CPU reads. The slice
/// threads record a wave against the frozen back-end; this thread then
/// commits each slice's units in order — segments checked and spliced, or
/// recorded exactly; holes and positioned events run exactly at their
/// positions — and runs the boundary record.
#[allow(clippy::too_many_arguments)]
fn drive_waves<'a>(
    cap: &'a Capture,
    body: &'a [Record],
    carry: &[bool],
    primed: bool,
    pool: &crate::slices::pool::WavePool<'a>,
    budget: u32,
    step: &mut dyn FnMut(usize, &mut LoopStats, bool) -> bool,
    s: &mut LoopStats,
) {
    use crate::backend::{SegItem, SliceItem, SliceUnit};
    use crate::slices::cut::Cutter;
    let per_wave = crate::slices::wave_slices(pool.len());
    let verify = crate::slices::verify();
    let skip = |i: usize| primed && carry[i];
    let mut i = 0;
    while i < body.len() {
        if skip(i) {
            i += 1;
            continue;
        }
        if matches!(wave_role(cap, &body[i]), WaveRole::Boundary) {
            step(i, s, false);
            i += 1;
            continue;
        }
        // The wave: up to K slices, each with the wave's CPU writes before it.
        let mut slices: Vec<crate::slices::pool::WaveSlice<'a>> = Vec::new();
        let mut cur: Vec<SliceItem<'a>> = Vec::new();
        let mut writes: Vec<(u64, u64, crate::backend::CpuBytes<'a>)> = Vec::new();
        let mut prefix_len = 0;
        let mut cutter = Cutter::new(budget);
        let mut draws = 0usize;
        let mut j = i;
        while j < body.len() {
            if skip(j) {
                j += 1;
                continue;
            }
            match wave_role(cap, &body[j]) {
                WaveRole::Boundary => break,
                WaveRole::Draw(d) => {
                    cutter.draw_targets(crate::slice_shadow::target_key(d));
                    if cutter.before_draw().is_some() {
                        slices.push((prefix_len, std::mem::take(&mut cur)));
                        prefix_len = writes.len();
                        cutter.reset();
                        if slices.len() == per_wave {
                            break;
                        }
                        cutter.before_draw();
                    }
                    cur.push(SliceItem::Draw(j, d));
                    draws += 1;
                }
                WaveRole::Cpu(a, n, b) => {
                    cur.push(SliceItem::Cpu(j, a, n, b));
                    writes.push((a, n, b));
                }
                WaveRole::Exact => cur.push(SliceItem::Exact(j)),
            }
            j += 1;
        }
        if !cur.is_empty() {
            slices.push((prefix_len, cur));
        }
        let positions = |slices: &[crate::slices::pool::WaveSlice]| -> Vec<usize> {
            slices
                .iter()
                .flat_map(|s| s.1.iter())
                .map(|it| match it {
                    SliceItem::Draw(p, _) | SliceItem::Cpu(p, ..) | SliceItem::Exact(p) => *p,
                })
                .collect()
        };
        // Too little to share out: run it here.
        let recorded = if draws < 2 {
            None
        } else {
            crate::backend::with_backend(|b| if b.slices_allowed() { pool.record(b, std::mem::take(&mut writes), std::mem::take(&mut slices)) } else { None }).flatten()
        };
        let Some(outs) = recorded else {
            for p in positions(&slices) {
                step(p, s, false);
            }
            i = j;
            continue;
        };
        s.waves += 1;
        // Guest memory written on the CPU side since the wave was recorded.
        let mut prior: Vec<crate::backend::PriorWrite> = Vec::new();
        // (address, size, bytes known: the slices composed them).
        let written = |p: usize| -> Option<crate::backend::PriorWrite> {
            match &body[p] {
                Record::Patch { addr, data } => Some((*addr, data.len() as u64, true)),
                Record::Zero { addr, len } => Some((*addr, *len, true)),
                Record::Event(Event::CpuAccess { addr, write: true }) => Some((*addr & !(crate::memory::PAGE_BYTES - 1), crate::memory::PAGE_BYTES, true)),
                Record::Event(Event::WriteMemory { dst, data, .. }) => Some((*dst, data.len() as u64, false)),
                Record::Event(Event::FillMemory { dst, size, .. } | Event::CopyMemory { dst, size, .. }) => Some((*dst, *size, false)),
                _ => None,
            }
        };
        for units in outs {
            for unit in units {
                match unit {
                    SliceUnit::Exact(p, hole) => {
                        if let Some(h) = hole {
                            s.holes[h as usize] += 1;
                        }
                        if let Some(w) = written(p) {
                            if step(p, s, true) {
                                prior.push(w);
                            }
                        }
                        step(p, s, false);
                    }
                    SliceUnit::Segment(mut seg) => {
                        let changing: Vec<bool> = seg.items.iter().filter_map(|it| if let SegItem::Cpu(p, ..) = it { Some(step(*p, s, true)) } else { None }).collect();
                        let w = seg.worker;
                        let r = crate::backend::with_backend(|b| pool.with_recorder(w, |rec| b.commit_segment(rec, &mut seg, &changing, &prior, true, verify))).flatten();
                        let mut ci = 0;
                        for it in &seg.items {
                            if let SegItem::Cpu(_, a, n, known) = it {
                                if changing[ci] {
                                    prior.push((*a, *n, *known));
                                }
                                ci += 1;
                            }
                        }
                        match r {
                            Some(Ok(())) => {
                                s.segments += 1;
                                s.spec_draws += seg.draws() as u64;
                                s.draws += seg.draws() as u64;
                                for it in &seg.items {
                                    if let SegItem::Cpu(p, ..) = it {
                                        step(*p, s, false);
                                    }
                                }
                            }
                            other => {
                                let verifying = matches!(other, Some(Err("verify")));
                                if !verifying {
                                    s.conflicts += 1;
                                    s.conflict_draws += seg.draws() as u64;
                                    let why = match other {
                                        Some(Err(w)) => CONFLICT_NAMES.iter().position(|n| *n == w).unwrap_or(CONFLICT_NAMES.len() - 1),
                                        _ => CONFLICT_NAMES.len() - 1,
                                    };
                                    s.conflict_why[why] += 1;
                                }
                                for it in &seg.items {
                                    match it {
                                        SegItem::Draw(dr) => {
                                            step(dr.pos, s, false);
                                            if verifying {
                                                let exact = crate::backend::with_backend(|b| b.take_digest()).unwrap_or(0);
                                                s.verified += 1;
                                                // Bytes 2..5: pipeline and draw, rendering area,
                                                // guest ranges, images. The binding source of a
                                                // range (a copy or the mirror, with its offset
                                                // dwords: bytes 6..7) may differ between
                                                // equivalent recordings.
                                                const DECISIONS: u64 = 0x0000_FFFF_FFFF_0000;
                                                if exact != dr.digest && (exact ^ dr.digest) & DECISIONS == 0 {
                                                    s.source_diffs += 1;
                                                } else if exact != dr.digest {
                                                    if let Record::Event(Event::Draw(d)) = &body[dr.pos] {
                                                        log::warn!(
                                                            "ps5gpu: slice verify: draw {} vs={:#x} ps={:#x} decisions differ (speculative {:#x}, exact {exact:#x})",
                                                            dr.pos,
                                                            d.shaders.vertex_program().address,
                                                            d.shaders.ps.address,
                                                            dr.digest
                                                        );
                                                    }
                                                    s.mismatches += 1;
                                                    crate::slices::stats::VERIFY_MISMATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                                }
                                            }
                                        }
                                        SegItem::Cpu(p, ..) => {
                                            step(*p, s, false);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        i = j;
    }
}

// ---------------------------------------------------------------------------
// Host memory of a replay
// ---------------------------------------------------------------------------

#[cfg(unix)]
type Backing = i32;
#[cfg(windows)]
type Backing = std::sync::Arc<ps5vm::os::Section>;

/// The backing of `size` bytes and its read/write alias.
#[cfg(unix)]
fn create_backing(size: u64) -> Result<(Backing, u64), String> {
    // SAFETY: plain syscalls on our own descriptor and fresh mappings.
    unsafe {
        let fd = libc::memfd_create(c"ps5-replay".as_ptr(), libc::MFD_CLOEXEC);
        if fd < 0 || libc::ftruncate(fd, size as libc::off_t) != 0 {
            return Err(format!("memfd of {size:#x} bytes: {}", std::io::Error::last_os_error()));
        }
        let alias = libc::mmap(std::ptr::null_mut(), size as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED | libc::MAP_NORESERVE, fd, 0);
        if alias == libc::MAP_FAILED {
            return Err(format!("backing alias of {size:#x} bytes: {}", std::io::Error::last_os_error()));
        }
        Ok((fd, alias as u64))
    }
}

#[cfg(windows)]
fn create_backing(size: u64) -> Result<(Backing, u64), String> {
    let sec = ps5vm::os::Section::create(size).ok_or_else(|| format!("section of {size:#x} bytes: {}", std::io::Error::last_os_error()))?;
    let alias = sec.map_anywhere(0, size).ok_or_else(|| format!("backing alias of {size:#x} bytes: {}", std::io::Error::last_os_error()))?;
    Ok((sec, alias))
}

/// Map `[addr, addr+size)` read/write: the backing at `backing`, or private
/// memory. `fresh`: the range must be free. Returns the `PROT_*` bits.
#[cfg(unix)]
fn map_view(fd: &Backing, addr: u64, size: u64, backing: Option<u64>, fresh: bool) -> Result<i32, String> {
    let fd = *fd;
    let fixed = if fresh { libc::MAP_FIXED_NOREPLACE } else { libc::MAP_FIXED };
    let prot = libc::PROT_READ | libc::PROT_WRITE;
    // SAFETY: fixed mappings at the capture's guest addresses; without
    // `fresh` they replace a guest view this replay made.
    let r = unsafe {
        match backing {
            Some(off) => libc::mmap(addr as *mut libc::c_void, size as usize, prot, libc::MAP_SHARED | libc::MAP_NORESERVE | fixed, fd, off as libc::off_t),
            None => libc::mmap(addr as *mut libc::c_void, size as usize, prot, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | fixed, -1, 0),
        }
    };
    if r == libc::MAP_FAILED || r as u64 != addr {
        let e = std::io::Error::last_os_error();
        if r != libc::MAP_FAILED {
            // SAFETY: the mapping the kernel placed elsewhere.
            unsafe { libc::munmap(r, size as usize) };
        }
        return Err(format!("cannot map guest memory {addr:#x}+{size:#x} in this process ({e}); in use by: {}", mapping_at(addr, size)));
    }
    Ok(prot)
}

#[cfg(windows)]
fn map_view(sec: &Backing, addr: u64, size: u64, backing: Option<u64>, fresh: bool) -> Result<i32, String> {
    use ps5vm::os::space;
    if fresh && !space::is_free(addr, size) {
        return Err(format!("cannot map guest memory {addr:#x}+{size:#x} in this process: in use"));
    }
    let ok = match backing {
        Some(off) => space::map_view(addr, size, sec, off, ps5vm::MemProtect::RW),
        None => space::commit(addr, size, ps5vm::MemProtect::RW),
    };
    if !ok {
        return Err(format!("cannot map guest memory {addr:#x}+{size:#x} in this process ({})", std::io::Error::last_os_error()));
    }
    Ok(ps5vm::os::PROT_READ | ps5vm::os::PROT_WRITE)
}

/// Replace a guest range of this replay with an inaccessible hold.
#[cfg(unix)]
fn hold(addr: u64, size: u64) {
    // SAFETY: a guest range of this replay becomes an inaccessible hold.
    unsafe { libc::mmap(addr as *mut libc::c_void, size as usize, libc::PROT_NONE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | libc::MAP_FIXED, -1, 0) };
}

#[cfg(windows)]
fn hold(addr: u64, size: u64) {
    ps5vm::os::space::hold(addr, size);
}

/// A fresh private page at `page` that never replaces a mapping.
#[cfg(unix)]
fn fresh_page(page: u64) -> bool {
    // SAFETY: MAP_FIXED_NOREPLACE never replaces a mapping.
    let r = unsafe {
        libc::mmap(page as *mut libc::c_void, PAGE as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED_NOREPLACE, -1, 0)
    };
    if r as u64 == page {
        return true;
    }
    if r != libc::MAP_FAILED {
        // SAFETY: the mapping the kernel placed elsewhere.
        unsafe { libc::munmap(r, PAGE as usize) };
    }
    false
}

#[cfg(windows)]
fn fresh_page(page: u64) -> bool {
    ps5vm::os::map_anon_noreplace(page, PAGE, ps5vm::MemProtect::RW).is_ok()
}

/// A process-wide handler for accesses to watched guest pages (the
/// back-end protects the views it caches, as in the game).
#[cfg(unix)]
fn install_fault_handler() {
    // SAFETY: installing a SIGSEGV handler.
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = replay_segv as *const () as usize;
        sa.sa_flags = libc::SA_SIGINFO | libc::SA_NODEFER;
        libc::sigemptyset(&mut sa.sa_mask);
        libc::sigaction(libc::SIGSEGV, &sa, std::ptr::null_mut());
    }
}

#[cfg(windows)]
fn install_fault_handler() {
    use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS};
    unsafe extern "system" fn veh(p: *mut EXCEPTION_POINTERS) -> i32 {
        let rec = &*(*p).ExceptionRecord;
        if rec.ExceptionCode == windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION
            && rec.NumberParameters >= 2
            && crate::memory::handle_fault(rec.ExceptionInformation[1] as u64, rec.ExceptionInformation[0] == 1)
        {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        EXCEPTION_CONTINUE_SEARCH
    }
    // SAFETY: registering a process-wide handler.
    unsafe { AddVectoredExceptionHandler(1, Some(veh)) };
}
