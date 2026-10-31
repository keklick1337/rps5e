// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Frame capture: everything the GPU back-end consumes over a few guest
//! frames, written to a file that [`replay`] executes again through the
//! same back-end entry points, offline.
//!
//! Turned on with `RPS5E_CAPTURE_FRAMES=<dir>`: at the first flip at or
//! after `RPS5E_CAPTURE_AT` seconds of the process (default 0) the
//! capture starts and it ends with the `RPS5E_CAPTURE_COUNT`-th flip
//! after that (default 1). Off, every hook costs one relaxed atomic load.
//!
//! What is recorded ([`format`]):
//! - the back-end entry points ([`crate::backend::Backend`]'s draw,
//!   dispatch, transfers, completion callbacks, flushes), CPU
//!   synchronisation points and flips, in the order they run;
//! - guest memory as the back-end reads it: a page the first time it is
//!   read, then again whenever it is read with other contents (the CPU
//!   wrote it meanwhile). A page is hashed on every read; GPU results the
//!   back-end writes into guest memory update the hash without a record;
//! - guest mappings at the start and their changes;
//! - CPU accesses to pages the GPU caches watch (the faults the tracking
//!   resolves), so the replay pays for the same invalidations and
//!   downloads.
//!
//! When the capture starts, the GPU results are written back into guest
//! memory and the image and buffer caches are emptied
//! ([`crate::backend::Backend::capture_reset`]): every resource the
//! captured frames use is then read from guest memory, and recorded, when
//! it is first used. The game itself runs on with cold caches for a moment.

pub mod format;
pub mod replay;
#[cfg(test)]
mod tests;

use crate::memory::GuestGpuMemory;
use crate::SurfaceDesc;
use format::{Event, Totals};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Mutex;

const UNCONFIGURED: u8 = 0;
const OFF: u8 = 1;
const ARMED: u8 = 2;
const ACTIVE: u8 = 3;

static STATE: AtomicU8 = AtomicU8::new(UNCONFIGURED);
/// The memory manager whose reads are recorded (address of the value).
static MEMORY: AtomicUsize = AtomicUsize::new(0);
static RECORDER: Mutex<Option<Recorder>> = Mutex::new(None);

const PAGE: u64 = crate::memory::PAGE_BYTES;

thread_local! {
    /// Back-end entry points running on this thread: nested calls (a
    /// flush inside a draw) belong to the outer one.
    static DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// A capture is being recorded.
#[inline]
pub fn active() -> bool {
    STATE.load(Ordering::Relaxed) == ACTIVE
}

/// A capture is recording `mem`'s reads.
pub(crate) fn recording(mem: &GuestGpuMemory) -> bool {
    active() && captured(mem)
}

fn captured(mem: &GuestGpuMemory) -> bool {
    MEMORY.load(Ordering::Acquire) == mem as *const GuestGpuMemory as usize
}

fn lock() -> std::sync::MutexGuard<'static, Option<Recorder>> {
    RECORDER.lock().unwrap_or_else(|e| e.into_inner())
}

struct Recorder {
    out: std::io::BufWriter<std::fs::File>,
    part: PathBuf,
    path: PathBuf,
    scratch: Vec<u8>,
    /// Hash of the contents last recorded or written by the back-end, per
    /// page (or per readable piece of a page).
    pages: crate::fxhash::HashMap<u64, u64>,
    totals: Totals,
    /// Flips after which an environment-driven capture ends (0: ended by
    /// [`finish`]).
    frames_wanted: u64,
    ring_tail: usize,
    bytes: u64,
    error: Option<String>,
}

impl Recorder {
    fn emit(&mut self, f: impl FnOnce(&mut Vec<u8>)) {
        let mut buf = std::mem::take(&mut self.scratch);
        buf.clear();
        f(&mut buf);
        if self.error.is_none() {
            if let Err(e) = self.out.write_all(&buf) {
                self.error = Some(format!("{}: {e}", self.part.display()));
            }
        }
        self.bytes += buf.len() as u64;
        self.scratch = buf;
    }

    fn event(&mut self, ev: &Event) {
        let t = &mut self.totals;
        match ev {
            Event::Draw(_) => t.draws += 1,
            Event::Dispatch(_) => t.dispatches += 1,
            Event::Flip { .. } => t.flips += 1,
            Event::CpuAccess { .. } => t.cpu_accesses += 1,
            Event::Map { .. } | Event::Unmap { .. } => t.maps += 1,
            _ => {}
        }
        if !matches!(ev, Event::CpuAccess { .. } | Event::Map { .. } | Event::Unmap { .. }) {
            t.items += 1;
        }
        self.emit(|b| format::event_record(b, ev));
    }

    fn patch(&mut self, addr: u64, data: &[u8]) {
        if data.iter().all(|&b| b == 0) {
            self.totals.zero_bytes += data.len() as u64;
        } else {
            self.totals.patch_bytes += data.len() as u64;
        }
        self.totals.patches += 1;
        self.emit(|b| format::memory_record(b, addr, data));
    }

    /// CPU accesses noted since the last drain, as records.
    fn drain_accesses(&mut self) {
        let head = RING_HEAD.load(Ordering::Acquire);
        while self.ring_tail < head {
            let v = RING[self.ring_tail % RING_SLOTS].swap(0, Ordering::AcqRel);
            if v == 0 {
                // Reserved, not stored yet: the next drain takes it.
                break;
            }
            self.ring_tail += 1;
            self.event(&Event::CpuAccess { addr: v & !(PAGE - 1), write: v & 1 != 0 });
        }
    }
}

// ------------------------------------------------------------- CPU faults --

const RING_SLOTS: usize = 1 << 14;
static RING: [AtomicU64; RING_SLOTS] = [const { AtomicU64::new(0) }; RING_SLOTS];
static RING_HEAD: AtomicUsize = AtomicUsize::new(0);
static RING_DROPPED: AtomicU64 = AtomicU64::new(0);

/// A CPU access the GPU memory tracking resolved (fault handler: no locks,
/// no allocation).
#[inline]
pub(crate) fn note_fault(mem: &GuestGpuMemory, addr: u64, write: bool) {
    if !active() || !captured(mem) {
        return;
    }
    let i = RING_HEAD.fetch_add(1, Ordering::AcqRel);
    let v = (addr & !(PAGE - 1)) | 2 | write as u64;
    if RING[i % RING_SLOTS].swap(v, Ordering::AcqRel) != 0 {
        RING_DROPPED.fetch_add(1, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------- memory reads --

/// The back-end read `[addr, addr+len)` of guest memory.
#[inline]
pub(crate) fn note_read(mem: &GuestGpuMemory, addr: u64, len: u64) {
    if active() {
        touch(mem, addr, len, true);
    }
}

/// The back-end wrote `[addr, addr+len)` of guest memory (GPU results).
#[inline]
pub(crate) fn note_write(mem: &GuestGpuMemory, addr: u64, len: u64) {
    if active() {
        touch(mem, addr, len, false);
    }
}

/// Copy guest bytes without tripping the GPU watches (as the back-end
/// reads them): through the backing alias, or directly from untracked
/// readable memory.
fn read_guest(mem: &GuestGpuMemory, addr: u64, out: &mut [u8]) -> bool {
    let n = out.len() as u64;
    if let Some(p) = mem.backing_ptr(addr, n) {
        // SAFETY: the alias maps the memfd range backing this guest range.
        unsafe { std::ptr::copy_nonoverlapping(p, out.as_mut_ptr(), out.len()) };
        return true;
    }
    if !mem.pages.is_registered(addr, n) && ps5vm::ranges::accessible(addr, n, false) {
        // SAFETY: readable guest memory the GPU tracking never protects.
        unsafe { std::ptr::copy_nonoverlapping(addr as *const u8, out.as_mut_ptr(), out.len()) };
        return true;
    }
    false
}

/// Hash of a piece of guest memory (four independent lanes).
pub(crate) fn content_hash(b: &[u8]) -> u64 {
    let mut lanes = [0x9E37_79B9_7F4A_7C15u64, 0xBF58_476D_1CE4_E5B9, 0x94D0_49BB_1331_11EB, 0x2545_F491_4F6C_DD1D];
    let (chunks, r) = b.as_chunks::<32>();
    for c in chunks {
        for (i, lane) in lanes.iter_mut().enumerate() {
            let w = u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap());
            *lane = (*lane ^ w).wrapping_mul(0xFF51_AFD7_ED55_8CCD).rotate_left(29);
        }
    }
    let mut tail = [0u8; 32];
    tail[..r.len()].copy_from_slice(r);
    for (i, lane) in lanes.iter_mut().enumerate() {
        let w = u64::from_le_bytes(tail[i * 8..i * 8 + 8].try_into().unwrap());
        *lane = (*lane ^ w).wrapping_mul(0xC4CE_B9FE_1A85_EC53).rotate_left(31);
    }
    let mut h = b.len() as u64;
    for lane in lanes {
        h = (h ^ lane).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(27);
    }
    h ^ (h >> 31)
}

fn touch(mem: &GuestGpuMemory, addr: u64, len: u64, record: bool) {
    if len == 0 || !captured(mem) {
        return;
    }
    let end = addr.saturating_add(len);
    let mut buf = [0u8; PAGE as usize];
    let mut page = addr & !(PAGE - 1);
    while page < end {
        let next = page + PAGE;
        // The whole page when it is readable, else the piece read.
        let (at, n) = if read_guest(mem, page, &mut buf) {
            (page, PAGE as usize)
        } else {
            let (a, e) = (page.max(addr), next.min(end));
            let n = (e - a) as usize;
            if !read_guest(mem, a, &mut buf[..n]) {
                page = next;
                continue;
            }
            (a, n)
        };
        let h = content_hash(&buf[..n]);
        let mut g = lock();
        let Some(r) = g.as_mut() else { return };
        if r.pages.insert(at, h) != Some(h) && record {
            r.patch(at, &buf[..n]);
        }
        drop(g);
        page = next;
    }
}

// ------------------------------------------------------------ the stream --

/// A guest mapping changed (after the memory manager's own locks).
#[inline]
pub(crate) fn note_map(mem: &GuestGpuMemory, ev: Event) {
    if !active() || !captured(mem) {
        return;
    }
    if let Some(r) = lock().as_mut() {
        r.event(&ev);
    }
}

/// A CPU synchronisation point (see [`crate::memory::note_sync_point`]).
#[inline]
pub(crate) fn note_sync_point() {
    if active() && DEPTH.with(|d| d.get()) == 0 {
        item(Event::SyncPoint);
    }
}

fn item(ev: Event) {
    if let Some(r) = lock().as_mut() {
        r.drain_accesses();
        r.event(&ev);
    }
}

/// Run back-end entry point `f`; while a capture of `mem` records, its
/// record `ev` follows the guest memory `f` read.
#[inline]
pub(crate) fn around<R>(mem: &GuestGpuMemory, f: impl FnOnce() -> R, ev: impl FnOnce(&R) -> Event) -> R {
    if !active() {
        return f();
    }
    around_slow(mem, f, ev)
}

#[cold]
fn around_slow<R>(mem: &GuestGpuMemory, f: impl FnOnce() -> R, ev: impl FnOnce(&R) -> Event) -> R {
    if !captured(mem) || DEPTH.with(|d| d.get()) != 0 {
        return f();
    }
    if let Some(r) = lock().as_mut() {
        r.drain_accesses();
    }
    DEPTH.with(|d| d.set(1));
    let out = f();
    DEPTH.with(|d| d.set(0));
    item(ev(&out));
    out
}

/// A display buffer is flipped (`gpu`: the presenter takes the image from
/// the back-end). Starts with the presentation; [`flip_end`] follows it.
pub fn flip_begin(desc: &SurfaceDesc, addr: u64, gpu: bool) {
    if STATE.load(Ordering::Relaxed) == UNCONFIGURED {
        configure();
    }
    if active() {
        item(Event::Flip { desc: *desc, addr, gpu });
    }
}

/// The presentation of a flip is done: an environment-driven capture
/// starts or ends here.
pub fn flip_end() {
    match STATE.load(Ordering::Relaxed) {
        ACTIVE => {
            let done = lock().as_ref().is_some_and(|r| r.frames_wanted != 0 && r.totals.flips >= r.frames_wanted);
            if done {
                match finish() {
                    Ok(s) => log::info!("{s}"),
                    Err(e) => log::warn!("ps5gpu: frame capture failed: {e}"),
                }
            }
        }
        ARMED => {
            let Some(cfg) = config() else { return };
            if crate::process_seconds() >= cfg.at {
                start_configured(cfg);
            }
        }
        _ => {}
    }
}

struct Config {
    dir: PathBuf,
    at: f64,
    count: u64,
}

fn config() -> Option<&'static Config> {
    static CFG: std::sync::OnceLock<Option<Config>> = std::sync::OnceLock::new();
    CFG.get_or_init(|| {
        let dir = std::env::var_os("RPS5E_CAPTURE_FRAMES").filter(|d| !d.is_empty())?;
        let at = std::env::var("RPS5E_CAPTURE_AT").ok().and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(0.0);
        let count = std::env::var("RPS5E_CAPTURE_COUNT").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(1).max(1);
        Some(Config { dir: dir.into(), at, count })
    })
    .as_ref()
}

fn configure() {
    let next = if config().is_some() { ARMED } else { OFF };
    let _ = STATE.compare_exchange(UNCONFIGURED, next, Ordering::AcqRel, Ordering::Acquire);
    if let Some(c) = config() {
        log::info!("ps5gpu: frame capture armed: {} frame(s) from {:.1} s into {}", c.count, c.at, c.dir.display());
    }
}

fn start_configured(cfg: &Config) {
    // One capture per process.
    if STATE.compare_exchange(ARMED, OFF, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return;
    }
    if let Err(e) = std::fs::create_dir_all(&cfg.dir) {
        log::warn!("ps5gpu: frame capture: {}: {e}", cfg.dir.display());
        return;
    }
    if crate::buffers::guest::bda_configured() {
        log::warn!("ps5gpu: frame capture with RPS5E_BDA=1: memory the GPU reads through device addresses is not recorded; the replay may differ");
    }
    let t = std::time::Instant::now();
    let reset = crate::backend::with_ready_backend(|b| b.capture_reset());
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let path = cfg.dir.join(format!("capture-{secs}-{}.ps5cap", std::process::id()));
    match start(crate::memory::global(), &path, cfg.count) {
        Ok(()) => log::info!(
            "ps5gpu: frame capture started at {:.1} s ({} frame(s)) into {}; caches emptied in {:.0} ms ({:?} images/buffers)",
            crate::process_seconds(),
            cfg.count,
            path.display(),
            t.elapsed().as_secs_f64() * 1e3,
            reset
        ),
        Err(e) => log::warn!("ps5gpu: frame capture: {}: {e}", path.display()),
    }
}

/// Start recording the back-end work over `mem` into `path` (written as
/// `<path>.part` until [`finish`]). `frames`: the capture ends by itself
/// after that many flips (0: only [`finish`] ends it).
pub fn start(mem: &'static GuestGpuMemory, path: &Path, frames: u64) -> std::io::Result<()> {
    let mut g = lock();
    if g.is_some() {
        return Err(std::io::Error::other("a capture is already recording"));
    }
    let part = PathBuf::from(format!("{}.part", path.display()));
    let file = std::fs::File::create(&part)?;
    let mut out = std::io::BufWriter::with_capacity(8 << 20, file);
    let mut header: Vec<(String, String)> = vec![
        ("format".into(), "rps5e-frame-capture".into()),
        ("version".into(), format::VERSION.to_string()),
        ("page_bytes".into(), PAGE.to_string()),
        ("frames".into(), frames.to_string()),
        ("alias_bytes".into(), mem.backing_alias().1.to_string()),
        ("start_seconds".into(), format!("{:.3}", crate::process_seconds())),
    ];
    let mut env: Vec<(String, String)> = std::env::vars().filter(|(k, _)| k.starts_with("RPS5E_") || k.starts_with("PS5_")).map(|(k, v)| (format!("env.{k}"), v)).collect();
    env.sort();
    header.extend(env);
    out.write_all(&format::header_bytes(&header))?;
    let mut r = Recorder {
        out,
        part,
        path: path.to_path_buf(),
        scratch: Vec::new(),
        pages: Default::default(),
        totals: Totals::default(),
        frames_wanted: frames,
        ring_tail: RING_HEAD.load(Ordering::Acquire),
        bytes: 0,
        error: None,
    };
    for slot in &RING {
        slot.store(0, Ordering::Relaxed);
    }
    RING_DROPPED.store(0, Ordering::Relaxed);
    for (addr, size, backing) in mem.spans_snapshot() {
        r.event(&Event::Map { addr, size, backing, prot: ps5vm::os::PROT_READ | ps5vm::os::PROT_WRITE });
    }
    *g = Some(r);
    MEMORY.store(mem as *const GuestGpuMemory as usize, Ordering::Release);
    // Programs are revalidated at the next use and read their code again
    // (recorded; the point itself is not, the capture is not active yet).
    crate::memory::note_sync_point();
    STATE.store(ACTIVE, Ordering::Release);
    Ok(())
}

/// A finished capture.
#[derive(Clone, Debug)]
pub struct Summary {
    pub path: PathBuf,
    pub totals: Totals,
    pub bytes: u64,
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let t = &self.totals;
        write!(
            f,
            "ps5gpu: frame capture written: {} ({:.1} MiB): {} flips, {} draws, {} dispatches, {} items, {} memory records ({:.1} MiB data, {:.1} MiB zero), {} CPU accesses ({} dropped), {} mapping records",
            self.path.display(),
            self.bytes as f64 / (1 << 20) as f64,
            t.flips,
            t.draws,
            t.dispatches,
            t.items,
            t.patches,
            t.patch_bytes as f64 / (1 << 20) as f64,
            t.zero_bytes as f64 / (1 << 20) as f64,
            t.cpu_accesses,
            t.dropped_accesses,
            t.maps
        )
    }
}

/// Stop recording and complete the file.
pub fn finish() -> std::io::Result<Summary> {
    STATE.store(OFF, Ordering::Release);
    MEMORY.store(0, Ordering::Release);
    let Some(mut r) = lock().take() else { return Err(std::io::Error::other("no capture is recording")) };
    r.drain_accesses();
    r.totals.dropped_accesses = RING_DROPPED.load(Ordering::Relaxed);
    let totals = r.totals;
    r.emit(|b| format::event_record(b, &Event::End(totals)));
    if let Some(e) = r.error.take() {
        return Err(std::io::Error::other(e));
    }
    r.out.flush()?;
    drop(r.out);
    std::fs::rename(&r.part, &r.path)?;
    let bytes = std::fs::metadata(&r.path).map_or(r.bytes, |m| m.len());
    Ok(Summary { path: r.path, totals, bytes })
}
