// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Performance overlay: a compact semi-transparent panel in the top-left
//! corner of the game window with presented FPS, frame time (+ graph and
//! 1% low), the guest flip rate, CPU/RAM/GPU usage and emulator counters.
//!
//! - Toggle: [`toggle_overlay`] (F10 by default, bound in ps5input), or
//!   start with it on via `RPS5E_PERF_OVERLAY=1` (`--perf-overlay`).
//! - Frame times are recorded per shown frame on the render thread
//!   ([`frames::FrameTimes`], a fixed ring).
//! - System statistics are sampled at 2 Hz on a background thread
//!   (`rps5e-perf`, started the first time the overlay is shown): `/proc`
//!   ([`proc`]), NVML / amdgpu sysfs / VK_EXT_memory_budget ([`gpu`]) and
//!   the emulator counters below.
//! - The panel is rasterised on the CPU with a cached glyph atlas
//!   ([`panel`]) into reused buffers and blended over the swapchain image by
//!   a small Vulkan pass ([`shaders`]). Nothing is allocated per frame.

pub mod frames;
pub mod gpu;
pub mod items;
pub mod panel;
pub mod proc;
pub mod shaders;
pub mod sys;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Mutex, Once, OnceLock};
use std::time::{Duration, Instant};

/// Draws the GPU back-end received.
pub static DRAWS: AtomicU64 = AtomicU64::new(0);
/// Compute dispatches the GPU back-end received.
pub static DISPATCHES: AtomicU64 = AtomicU64::new(0);
/// Guest shader programs compiled to SPIR-V.
pub static SHADER_COMPILES: AtomicU64 = AtomicU64::new(0);
/// Shader compiles running now.
pub static SHADER_COMPILING: AtomicU32 = AtomicU32::new(0);
/// Pipeline lookups served from the pipeline cache / that created one.
pub static PIPELINE_HITS: AtomicU64 = AtomicU64::new(0);
pub static PIPELINE_MISSES: AtomicU64 = AtomicU64::new(0);
/// Display flips the guest submitted.
pub static GUEST_FLIPS: AtomicU64 = AtomicU64::new(0);

static HLE_CALLS: OnceLock<fn() -> u64> = OnceLock::new();

/// Install the source of the HLE call count (the CPU layer's import
/// counter; ps5gpu does not depend on it).
pub fn set_hle_call_counter(f: fn() -> u64) {
    let _ = HLE_CALLS.set(f);
}

/// Count a running shader compile for the overlay (dropped when done).
pub struct CompileGuard(());

impl CompileGuard {
    pub fn start() -> Self {
        SHADER_COMPILES.fetch_add(1, Ordering::Relaxed);
        SHADER_COMPILING.fetch_add(1, Ordering::Relaxed);
        CompileGuard(())
    }
}

impl Drop for CompileGuard {
    fn drop(&mut self) {
        SHADER_COMPILING.fetch_sub(1, Ordering::Relaxed);
    }
}

/// 0: not read from the environment yet, 1: off, 2: on.
static ENABLED: AtomicU8 = AtomicU8::new(0);

/// Whether the overlay is shown (initially `RPS5E_PERF_OVERLAY`).
pub fn overlay_enabled() -> bool {
    match ENABLED.load(Ordering::Relaxed) {
        0 => {
            let on = std::env::var("RPS5E_PERF_OVERLAY").is_ok_and(|v| !v.is_empty() && v != "0");
            let _ = ENABLED.compare_exchange(0, if on { 2 } else { 1 }, Ordering::Relaxed, Ordering::Relaxed);
            ENABLED.load(Ordering::Relaxed) == 2
        }
        v => v == 2,
    }
}

/// Show or hide the overlay.
pub fn set_overlay_enabled(on: bool) {
    ENABLED.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    crate::input::wake();
}

/// Flip the overlay; returns the new state.
pub fn toggle_overlay() -> bool {
    let on = !overlay_enabled();
    log::info!("performance overlay {}", if on { "on" } else { "off" });
    set_overlay_enabled(on);
    on
}

/// Busiest threads shown.
pub const TOP_THREADS: usize = 3;

/// A thread and its CPU use (percent of one core).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ThreadLoad {
    name: [u8; 16],
    len: u8,
    pub percent: f32,
}

impl ThreadLoad {
    fn new(name: &str, percent: f32) -> Self {
        let mut t = ThreadLoad { percent, ..Default::default() };
        let n = name.len().min(15);
        // Keep whole UTF-8 characters.
        let n = (0..=n).rev().find(|&i| name.is_char_boundary(i)).unwrap_or(0);
        t.name[..n].copy_from_slice(&name.as_bytes()[..n]);
        t.len = n as u8;
        t
    }

    pub fn name(&self) -> &str {
        std::str::from_utf8(&self.name[..self.len as usize]).unwrap_or("?")
    }
}

/// One 2 Hz sample of everything the overlay shows besides frame times.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SystemSample {
    /// Increments with every sample.
    pub seq: u64,
    /// Machine CPU use, percent of all CPUs.
    pub cpu_system: Option<f32>,
    /// Emulator process CPU use, percent of one core.
    pub cpu_process: Option<f32>,
    pub cpus: u32,
    pub threads: [ThreadLoad; TOP_THREADS],
    pub thread_count: usize,
    pub rss: Option<u64>,
    /// System memory (total, used), bytes.
    pub mem: Option<(u64, u64)>,
    pub gpu: gpu::GpuSample,
    pub flips_per_sec: f32,
    pub draws_per_flip: f32,
    pub dispatches_per_flip: f32,
    pub compiles_per_sec: f32,
    pub compiling: u32,
    /// Pipeline cache hit rate over the period (None without lookups).
    pub pipeline_hit_percent: Option<f32>,
    pub pipelines_per_sec: f32,
    pub hle_per_sec: Option<f32>,
}

static SAMPLE: Mutex<SystemSample> = Mutex::new(SystemSample {
    seq: 0,
    cpu_system: None,
    cpu_process: None,
    cpus: 0,
    threads: [ThreadLoad { name: [0; 16], len: 0, percent: 0.0 }; TOP_THREADS],
    thread_count: 0,
    rss: None,
    mem: None,
    gpu: gpu::GpuSample { busy: None, vram_used: None, vram_total: None, vram_process: None, temp_c: None, clock_mhz: None },
    flips_per_sec: 0.0,
    draws_per_flip: 0.0,
    dispatches_per_flip: 0.0,
    compiles_per_sec: 0.0,
    compiling: 0,
    pipeline_hit_percent: None,
    pipelines_per_sec: 0.0,
    hle_per_sec: None,
});
static SEQ: AtomicU64 = AtomicU64::new(0);
static SAMPLER: Once = Once::new();

/// Sampling period of the system statistics.
pub const SAMPLE_PERIOD: Duration = Duration::from_millis(500);

/// Start the sampler thread (idempotent).
pub fn ensure_sampler() {
    SAMPLER.call_once(|| {
        let spawned = std::thread::Builder::new().name("rps5e-perf".into()).spawn(|| {
            let mut s = Sampler::new();
            loop {
                if overlay_enabled() {
                    let sample = s.sample();
                    *SAMPLE.lock().unwrap_or_else(|e| e.into_inner()) = sample;
                    SEQ.store(sample.seq, Ordering::Release);
                    crate::input::wake();
                } else {
                    s.reset();
                }
                std::thread::sleep(SAMPLE_PERIOD);
            }
        });
        if let Err(e) = spawned {
            log::warn!("perf overlay: cannot start the sampler thread: {e}");
        }
    });
}

/// Sequence number of the newest sample (0: none yet).
pub fn sample_seq() -> u64 {
    SEQ.load(Ordering::Acquire)
}

/// The newest sample.
pub fn latest_sample() -> SystemSample {
    *SAMPLE.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Clone, Copy, Default)]
struct Counters {
    draws: u64,
    dispatches: u64,
    flips: u64,
    compiles: u64,
    hits: u64,
    misses: u64,
    hle: Option<u64>,
}

impl Counters {
    fn now() -> Self {
        Counters {
            draws: DRAWS.load(Ordering::Relaxed),
            dispatches: DISPATCHES.load(Ordering::Relaxed),
            flips: GUEST_FLIPS.load(Ordering::Relaxed),
            compiles: SHADER_COMPILES.load(Ordering::Relaxed),
            hits: PIPELINE_HITS.load(Ordering::Relaxed),
            misses: PIPELINE_MISSES.load(Ordering::Relaxed),
            hle: HLE_CALLS.get().map(|f| f()),
        }
    }
}

struct Prev {
    at: Instant,
    cpu: Option<proc::CpuTimes>,
    process_ticks: Option<u64>,
    counters: Counters,
}

struct Sampler {
    seq: u64,
    prev: Option<Prev>,
    threads: HashMap<u32, u64>,
    gpu: gpu::GpuSources,
    clk_tck: f64,
    page_size: u64,
    cpus: u32,
}

fn read(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

impl Sampler {
    fn new() -> Self {
        #[cfg(unix)]
        // SAFETY: plain sysconf queries.
        let (clk, page) = unsafe { (libc::sysconf(libc::_SC_CLK_TCK), libc::sysconf(libc::_SC_PAGESIZE)) };
        #[cfg(windows)]
        let (clk, page) = (100i64, ps5vm::os::page_size() as i64);
        let cpus = read("/proc/stat").map(|s| proc::count_cpus(&s) as u32).unwrap_or(0);
        Sampler {
            seq: 0,
            prev: None,
            threads: HashMap::new(),
            gpu: gpu::GpuSources::new(),
            clk_tck: if clk > 0 { clk as f64 } else { 100.0 },
            page_size: if page > 0 { page as u64 } else { 4096 },
            cpus,
        }
    }

    /// Forget the previous sample (the overlay was hidden).
    fn reset(&mut self) {
        self.prev = None;
        self.threads.clear();
    }

    fn sample(&mut self) -> SystemSample {
        let now = Instant::now();
        let cpu = read("/proc/stat").and_then(|s| proc::parse_proc_stat(&s));
        let process_ticks = read("/proc/self/stat").and_then(|s| proc::parse_task_stat(&s).map(|(_, t)| t));
        let counters = Counters::now();
        self.seq += 1;
        let mut out = SystemSample { seq: self.seq, cpus: self.cpus, ..Default::default() };
        let dt = self.prev.as_ref().map(|p| now.duration_since(p.at).as_secs_f64()).filter(|&d| d > 0.0);
        // Per-thread CPU (percent of one core over the period).
        let mut loads: Vec<(f32, ThreadLoad)> = Vec::new();
        let mut seen = HashMap::with_capacity(self.threads.len());
        if let Ok(dir) = std::fs::read_dir("/proc/self/task") {
            for e in dir.flatten() {
                let Some(tid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else { continue };
                let Some(text) = read(&format!("/proc/self/task/{tid}/stat")) else { continue };
                let Some((name, ticks)) = proc::parse_task_stat(&text) else { continue };
                if let (Some(dt), Some(&before)) = (dt, self.threads.get(&tid)) {
                    let pct = (ticks.saturating_sub(before) as f64 / self.clk_tck / dt * 100.0) as f32;
                    loads.push((pct, ThreadLoad::new(name, pct)));
                }
                seen.insert(tid, ticks);
            }
        }
        out.thread_count = seen.len();
        self.threads = seen;
        loads.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (slot, (_, t)) in out.threads.iter_mut().zip(loads.into_iter().filter(|(p, _)| *p >= 0.5)) {
            *slot = t;
        }
        out.rss = read("/proc/self/statm").and_then(|s| proc::parse_statm_resident(&s)).map(|p| p * self.page_size);
        out.mem = read("/proc/meminfo").and_then(|s| proc::parse_meminfo(&s));
        out.gpu = self.gpu.sample();
        if let (Some(prev), Some(dt)) = (&self.prev, dt) {
            if let (Some(a), Some(b)) = (prev.cpu, cpu) {
                out.cpu_system = Some(b.busy_percent(&a));
            }
            if let (Some(a), Some(b)) = (prev.process_ticks, process_ticks) {
                out.cpu_process = Some((b.saturating_sub(a) as f64 / self.clk_tck / dt * 100.0) as f32);
            }
            let (a, b) = (prev.counters, counters);
            let per_sec = |x: u64, y: u64| (y.saturating_sub(x) as f64 / dt) as f32;
            let flips = b.flips.saturating_sub(a.flips);
            out.flips_per_sec = per_sec(a.flips, b.flips);
            if flips > 0 {
                out.draws_per_flip = b.draws.saturating_sub(a.draws) as f32 / flips as f32;
                out.dispatches_per_flip = b.dispatches.saturating_sub(a.dispatches) as f32 / flips as f32;
            }
            out.compiles_per_sec = per_sec(a.compiles, b.compiles);
            let (hits, misses) = (b.hits.saturating_sub(a.hits), b.misses.saturating_sub(a.misses));
            if hits + misses > 0 {
                out.pipeline_hit_percent = Some(hits as f32 * 100.0 / (hits + misses) as f32);
            }
            out.pipelines_per_sec = per_sec(a.misses, b.misses);
            if let (Some(x), Some(y)) = (a.hle, b.hle) {
                out.hle_per_sec = Some(per_sec(x, y));
            }
        }
        out.compiling = SHADER_COMPILING.load(Ordering::Relaxed);
        self.prev = Some(Prev { at: now, cpu, process_ticks, counters });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_names_are_truncated_on_char_boundaries() {
        assert_eq!(ThreadLoad::new("rps5e-render", 1.0).name(), "rps5e-render");
        assert_eq!(ThreadLoad::new("a-very-long-thread-name", 1.0).name(), "a-very-long-thr");
        assert_eq!(ThreadLoad::new("ééééééééé", 1.0).name(), "ééééééé");
    }

    #[test]
    fn sampler_reads_this_process() {
        let mut s = Sampler::new();
        let first = s.sample();
        assert!(first.cpu_process.is_none(), "no rate without a previous sample");
        assert!(first.rss.is_some_and(|r| r > 0));
        assert!(first.mem.is_some_and(|(t, u)| t > 0 && u <= t));
        assert!(first.thread_count >= 1);
        std::thread::sleep(Duration::from_millis(30));
        let second = s.sample();
        assert_eq!(second.seq, first.seq + 1);
        assert!(second.cpu_process.is_some());
        assert!(second.cpu_system.is_some_and(|c| (0.0..=100.0).contains(&c)));
    }
}
