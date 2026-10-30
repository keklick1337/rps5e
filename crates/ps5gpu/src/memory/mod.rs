// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest GPU memory tracking.
//!
//! Guest RAM that the GPU caches mirror is watched with host page
//! protection ([`guard::PageGuard`]): pages whose contents were uploaded are
//! write-protected, pages the GPU wrote are made inaccessible. The SIGSEGV
//! handler (`ps5cpu::exec::set_fault_hook`) calls [`handle_fault`], which
//! asks the attached buffer and image stores to mark the CPU write (or to
//! download GPU data before a CPU access) and lets the access retry.
//!
//! Guest physical memory is a memfd. The memory manager registers every
//! memfd-backed guest mapping here together with its backing offset, and
//! maps the whole memfd a second time as a *backing alias*: the GPU side
//! reads and writes guest memory through the alias
//! ([`GuestGpuMemory::read_backing`] / [`GuestGpuMemory::write_backing`])
//! without tripping the protection it installed on the guest view.
//!
//! Layout: 4 KiB pages, 4 MiB tracking blocks, a 1 TiB guest space.

pub mod deps;
pub mod guard;
pub mod history;
pub mod layout_memo;
pub mod lock;
pub mod mask;
pub mod region;
pub mod tracker;
pub mod views;

use guard::PageGuard;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

pub const PAGE_BYTES: u64 = 4096;
pub const BLOCK_BYTES: u64 = 4 << 20;
/// Guest addresses the tracking covers: user memory and the AMM window at
/// 8 TiB (`0x800_0000_0000`, 512 GiB), where titles that use AMM keep
/// textures and buffers.
pub const SPACE_BYTES: u64 = 1 << 44;
pub const PAGES_PER_BLOCK: usize = (BLOCK_BYTES / PAGE_BYTES) as usize;
pub const BLOCK_COUNT: usize = (SPACE_BYTES / BLOCK_BYTES) as usize;

/// Values whose all-zero bit pattern is valid (null / 0).
pub(crate) trait ZeroInit {}
impl<T> ZeroInit for std::sync::atomic::AtomicPtr<T> {}
impl ZeroInit for AtomicU64 {}

/// A slice of `n` zeroed values. The allocation comes from zero pages that
/// take memory only once written, so per-block tables of the whole space
/// cost what the mapped blocks use.
pub(crate) fn zeroed_slice<T: ZeroInit>(n: usize) -> Box<[T]> {
    let layout = std::alloc::Layout::array::<T>(n).expect("table size");
    if layout.size() == 0 {
        return Vec::new().into_boxed_slice();
    }
    // SAFETY: the layout has a non-zero size; all-zero bytes are a valid
    // `T` (ZeroInit); the box takes ownership of the allocation with the
    // same layout.
    unsafe {
        let p = std::alloc::alloc_zeroed(layout) as *mut T;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Box::from_raw(std::ptr::slice_from_raw_parts_mut(p, n))
    }
}

/// CPU synchronisation points seen by the command processor: a submission
/// (or a slice of one) starts, or a wait on guest memory is satisfied.
/// The CPU wrote what the following commands read before the point, so a
/// hot page (see [`region`]) is compared with its upload once between two
/// points.
static SYNC_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Guest views mapped over backing bytes another view already maps.
static BACKING_ALIAS_VIEWS: AtomicU64 = AtomicU64::new(0);

/// Guest views mapped over backing bytes another guest view already maps
/// (the GPU caches do not connect the two views).
pub fn backing_alias_views() -> u64 {
    BACKING_ALIAS_VIEWS.load(Ordering::Relaxed)
}

/// Mappings that replaced views of other backing bytes in place (a direct
/// mapping over a direct mapping, an AMM remap), without an unmap first.
static REPLACED_VIEWS: AtomicU64 = AtomicU64::new(0);

pub fn replaced_views() -> u64 {
    REPLACED_VIEWS.load(Ordering::Relaxed)
}

/// Imports of guest pages whose guest view now maps other backing bytes
/// (diagnostics): the guest range, the backing offset the import holds and
/// the one the range maps now (None: not one memfd range).
#[derive(Clone, Copy, Debug)]
pub struct StaleImport {
    pub addr: u64,
    pub size: u64,
    pub imported: u64,
    pub current: Option<u64>,
}

static STALE_IMPORTS: std::sync::Mutex<Vec<StaleImport>> = std::sync::Mutex::new(Vec::new());
static STALE_IMPORT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Note an import found stale (see [`StaleImport`]).
pub fn note_stale_import(s: StaleImport) {
    let n = STALE_IMPORT_COUNT.fetch_add(1, Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        log::warn!(
            "ps5gpu: the import of guest {:#x}+{:#x} holds backing {:#x}, the range now maps {:?} ({} such imports so far)",
            s.addr,
            s.size,
            s.imported,
            s.current.map(|c| format!("{c:#x}")),
            n + 1
        );
    }
    let mut v = STALE_IMPORTS.lock().unwrap_or_else(|e| e.into_inner());
    if v.len() >= 4096 {
        v.drain(..2048);
    }
    v.push(s);
}

pub fn stale_imports() -> u64 {
    STALE_IMPORT_COUNT.load(Ordering::Relaxed)
}

/// Per-thread copies of the last spans [`GuestGpuMemory::backing_ptr`]
/// resolved, most recent first (span lookups take the spans lock; a draw
/// reads constant rings, vertex data and code in different mappings).
#[derive(Clone, Copy)]
struct SpanCache {
    owner: usize,
    generation: u64,
    base: u64,
    start: u64,
    end: u64,
    backing: u64,
}

const SPAN_CACHE_ENTRIES: usize = 4;
const NO_SPAN: SpanCache = SpanCache { owner: 0, generation: 0, base: 0, start: 0, end: 0, backing: 0 };

thread_local! {
    static SPAN_CACHE: std::cell::Cell<[SpanCache; SPAN_CACHE_ENTRIES]> = const { std::cell::Cell::new([NO_SPAN; SPAN_CACHE_ENTRIES]) };
}

/// `RPS5E_SAME_BACKING_REMAP=0`: a mapping onto the bytes a range already
/// shows is reported to the buffer and image caches as new contents (they
/// drop GPU results there and load the range again), as before.
fn same_backing_remap_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_SAME_BACKING_REMAP").is_ok_and(|v| v == "0"))
}

/// `RPS5E_FAULT_LANDING=0`: a CPU access to GPU-dirty pages always goes
/// through the back-end (it takes the back-end lock and waits for a copy
/// recorded behind all queued work). On (default), copies into guest
/// memory recorded after the GPU work (write-through, settling, downloads)
/// note their tick on the pages, and a CPU access whose page's copy
/// completed just unprotects it; one still in flight is waited for
/// without the back-end.
pub fn landing_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_FAULT_LANDING"))
}

pub fn note_sync_point() {
    crate::capture::note_sync_point();
    SYNC_GENERATION.fetch_add(1, Ordering::AcqRel);
}

pub fn sync_generation() -> u64 {
    SYNC_GENERATION.load(Ordering::Acquire)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteOrigin {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultKind {
    Read,
    Write,
}

/// The buffer cache as seen from the fault handler.
pub trait BufferStore: Send + Sync {
    /// A CPU write fault: true when the range is tracked and any GPU data
    /// reached guest memory (the access may retry).
    fn mark_cpu_write(&self, addr: u64, size: u64) -> bool;
    /// A CPU access to GPU-owned pages: download them; true on success.
    fn download_to_cpu(&self, addr: u64, size: u64) -> bool;
    /// Guest memory was mapped at `[addr, addr+size)`: what the mirror
    /// holds there (uncommitted pages read as zeros, or another mapping's
    /// bytes) is stale. Not called from the fault handler.
    fn mapped(&self, _addr: u64, _size: u64) {}
    /// A fault on the page of `addr` that keeps coming back: resolve it
    /// with the GPU side held still (no work recorded meanwhile can dirty
    /// or watch the page again): GPU-written bytes of the page land in
    /// guest memory, images read back, and a write is marked in every
    /// cache. Waits for the back-end. True when the watches allow the
    /// access afterwards; None when it cannot be done (no back-end, or
    /// this thread holds it).
    fn settle(&self, _addr: u64, _write: bool) -> Option<bool> {
        None
    }
}

/// The image cache as seen from the fault handler / memory manager.
pub trait ImageStore: Send + Sync {
    /// A CPU write fault: true when an image page covers the range.
    fn mark_cpu_write(&self, addr: u64, size: u64) -> bool;
    /// Any CPU access to read-watched pages of GPU-written images: a read
    /// writes them back to guest memory first, a write lets the CPU bytes
    /// land and the images follow them; true when it handled the page.
    fn cpu_access(&self, _addr: u64, _size: u64, _write: bool) -> bool {
        false
    }
    /// Guest memory was mapped at `[addr, addr+size)`: images over it hold
    /// stale bytes on exactly those pages. Not called from the fault
    /// handler.
    fn mapped(&self, _addr: u64, _size: u64) {}
}

/// Mapping changes the GPU caches must see before the memory goes away.
pub trait UnmapListener: Send + Sync {
    fn unmapped(&self, addr: u64, size: u64);
}

#[derive(Clone, Copy, Debug)]
struct Span {
    end: u64,
    /// Offset of `start` in the backing memfd (None: not aliasable).
    backing: Option<u64>,
}

/// A change of the guest mappings, as the GPU's import of guest memory
/// sees it (see [`GuestGpuMemory::enable_map_log`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapEvent {
    /// `[addr, addr+size)` maps the memfd at `backing` (None: not aliasable).
    Mapped { addr: u64, size: u64, backing: Option<u64> },
    Unmapped { addr: u64, size: u64 },
}

/// The committed parts of a range (see [`GuestGpuMemory::committed`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Commitment {
    /// `[begin, end)` runs in address order.
    pub runs: Vec<(u64, u64)>,
    /// The runs cover the whole range.
    pub whole: bool,
}

/// Diagnostics counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryStats {
    pub mapped: u64,
    pub unmapped: u64,
    pub faults_resolved: u64,
    pub faults_declined: u64,
    pub protect_calls: u64,
    /// Resolved faults by kind: CPU writes into watched pages, CPU reads
    /// of GPU-written pages.
    pub write_faults: u64,
    pub read_faults: u64,
    /// Write faults on pages of a watched image.
    pub image_faults: u64,
    /// Faults that kept coming back, resolved with the GPU side held still
    /// ([`BufferStore::settle`]).
    pub settled_faults: u64,
}

pub struct GuestGpuMemory {
    pub pages: Arc<PageGuard>,
    spans: RwLock<BTreeMap<u64, Span>>,
    /// The spans with a backing, by backing offset (changed under the
    /// spans lock).
    views: std::sync::Mutex<views::BackingViews>,
    alias_base: AtomicU64,
    alias_size: AtomicU64,
    buffers: OnceLock<Arc<dyn BufferStore>>,
    images: OnceLock<Arc<dyn ImageStore>>,
    unmap: OnceLock<Arc<dyn UnmapListener>>,
    /// Mapping changes since the last [`GuestGpuMemory::take_map_events`]
    /// (None until [`GuestGpuMemory::enable_map_log`]); appended under the
    /// spans lock, so the log continues the snapshot exactly.
    map_log: std::sync::Mutex<Option<Vec<MapEvent>>>,
    map_events: AtomicU64,
    mapped: AtomicU64,
    unmapped: AtomicU64,
    /// Moved after every change of the spans and the page registry is
    /// complete: the stamp of [`layout_memo`] answers.
    layout: AtomicU64,
    /// Process-unique number of this instance (keys of [`layout_memo`]).
    id: u64,
    resolved: AtomicU64,
    declined: AtomicU64,
    write_faults: AtomicU64,
    read_faults: AtomicU64,
    image_faults: AtomicU64,
    settled_faults: AtomicU64,
}

impl Default for GuestGpuMemory {
    fn default() -> Self {
        Self::new()
    }
}

thread_local! {
    /// (page, consecutive faults) of the last fault on this thread.
    static LAST_FAULT: std::cell::Cell<(u64, u32)> = const { std::cell::Cell::new((0, 0)) };
}

/// Consecutive faults on one page resolved the quick way (no back-end
/// lock unless a download needs it). A store may answer "handled" while
/// the page stays protected: the GPU side re-dirtied or re-watched it
/// meanwhile, or the fault cannot make progress without the back-end.
/// Past this count every fault on the page is settled with the GPU side
/// held still ([`BufferStore::settle`]).
const SETTLE_AFTER: u32 = 1 << 10;

/// Settles of one page that leave it protected after which the fault is
/// declined: under the back-end lock nothing else changes the page's
/// watches, so a watch left after a settle has no owner to release it.
const SETTLE_TRIES: u32 = 16;

/// Without a settle (no back-end, or this thread holds it): consecutive
/// faults on a still-watched page retried as a race with the GPU side (a
/// region lock held by an upload or download on another thread can
/// outlast a few dozen yields) before the fault is declined.
const TRANSIENT_RETRIES: u32 = 1 << 16;

impl GuestGpuMemory {
    pub fn new() -> Self {
        Self {
            pages: Arc::new(PageGuard::new()),
            spans: RwLock::new(BTreeMap::new()),
            views: std::sync::Mutex::new(views::BackingViews::default()),
            alias_base: AtomicU64::new(0),
            alias_size: AtomicU64::new(0),
            buffers: OnceLock::new(),
            images: OnceLock::new(),
            unmap: OnceLock::new(),
            map_log: std::sync::Mutex::new(None),
            map_events: AtomicU64::new(0),
            mapped: AtomicU64::new(0),
            unmapped: AtomicU64::new(0),
            layout: AtomicU64::new(0),
            id: layout_memo::instance_id(),
            resolved: AtomicU64::new(0),
            declined: AtomicU64::new(0),
            write_faults: AtomicU64::new(0),
            read_faults: AtomicU64::new(0),
            image_faults: AtomicU64::new(0),
            settled_faults: AtomicU64::new(0),
        }
    }

    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            mapped: self.mapped.load(Ordering::Relaxed),
            unmapped: self.unmapped.load(Ordering::Relaxed),
            faults_resolved: self.resolved.load(Ordering::Relaxed),
            faults_declined: self.declined.load(Ordering::Relaxed),
            protect_calls: guard::PROTECT_CALLS.load(Ordering::Relaxed),
            write_faults: self.write_faults.load(Ordering::Relaxed),
            read_faults: self.read_faults.load(Ordering::Relaxed),
            image_faults: self.image_faults.load(Ordering::Relaxed),
            settled_faults: self.settled_faults.load(Ordering::Relaxed),
        }
    }

    /// Attach the GPU caches (once).
    pub fn attach(&self, buffers: Arc<dyn BufferStore>, images: Arc<dyn ImageStore>, unmap: Arc<dyn UnmapListener>) {
        let _ = self.buffers.set(buffers);
        let _ = self.images.set(images);
        let _ = self.unmap.set(unmap);
    }

    /// The whole backing memfd mapped read/write at `base` (`size` bytes).
    pub fn set_backing_alias(&self, base: u64, size: u64) {
        self.alias_size.store(size, Ordering::Release);
        self.alias_base.store(base, Ordering::Release);
    }

    pub fn has_backing_alias(&self) -> bool {
        self.alias_base.load(Ordering::Acquire) != 0
    }

    /// The backing alias: (host base, size); base 0 when there is none.
    pub fn backing_alias(&self) -> (u64, u64) {
        (self.alias_base.load(Ordering::Acquire), self.alias_size.load(Ordering::Acquire))
    }

    /// Start logging mapping changes: returns the current mappings as
    /// events, after which every change is logged until taken with
    /// [`Self::take_map_events`]. Calling it again restarts the log.
    pub fn enable_map_log(&self) -> Vec<MapEvent> {
        let m = self.spans.write().unwrap_or_else(|e| e.into_inner());
        let snapshot = m.iter().map(|(&addr, sp)| MapEvent::Mapped { addr, size: sp.end - addr, backing: sp.backing }).collect();
        *self.map_log.lock().unwrap_or_else(|e| e.into_inner()) = Some(Vec::new());
        self.map_events.store(0, Ordering::Release);
        snapshot
    }

    /// Mapping changes logged so far (0 after a take): a lock-free check
    /// before [`Self::take_map_events`].
    pub fn pending_map_events(&self) -> u64 {
        self.map_events.load(Ordering::Acquire)
    }

    /// The logged mapping changes, oldest first.
    pub fn take_map_events(&self) -> Vec<MapEvent> {
        let mut log = self.map_log.lock().unwrap_or_else(|e| e.into_inner());
        self.map_events.store(0, Ordering::Release);
        log.as_mut().map(std::mem::take).unwrap_or_default()
    }

    /// The memfd-backed pieces of `[addr, addr+size)`: (guest address,
    /// backing offset, bytes) in address order.
    pub fn backing_runs(&self, addr: u64, size: u64) -> smallvec::SmallVec<[(u64, u64, u64); 4]> {
        let mut out = smallvec::SmallVec::new();
        let end = addr.saturating_add(size);
        let m = self.spans();
        let first = m.range(..=addr).next_back().map_or(addr, |(&s, _)| s);
        for (&s, sp) in m.range(first..end) {
            let (a, e) = (s.max(addr), sp.end.min(end));
            if a >= e {
                continue;
            }
            if let Some(b) = sp.backing {
                out.push((a, b + (a - s), e - a));
            }
        }
        out
    }

    /// Log a change (called with the spans write lock held).
    fn log_map(&self, e: MapEvent) {
        let mut log = self.map_log.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(l) = log.as_mut() {
            l.push(e);
            self.map_events.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn spans(&self) -> std::sync::RwLockReadGuard<'_, BTreeMap<u64, Span>> {
        self.spans.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Cut the backing views of the spans `[start, end)` removes, as
    /// [`Self::remove_spans`] cuts the spans.
    fn unindex(views: &mut views::BackingViews, map: &BTreeMap<u64, Span>, start: u64, end: u64) {
        for (&s, sp) in map.range(..end).filter(|(&s, sp)| s < end && sp.end > start) {
            if let Some(b) = sp.backing {
                views.cut(s, sp.end, b, start, end);
            }
        }
    }

    fn remove_spans(map: &mut BTreeMap<u64, Span>, start: u64, end: u64) {
        let keys: Vec<u64> = map.range(..end).filter(|(&s, sp)| s < end && sp.end > start).map(|(&s, _)| s).collect();
        for s in keys {
            let sp = map.remove(&s).unwrap();
            if s < start {
                map.insert(s, Span { end: start, backing: sp.backing });
            }
            if sp.end > end {
                map.insert(end, Span { end: sp.end, backing: sp.backing.map(|b| b + (end - s)) });
            }
        }
    }

    /// Diagnostic: the new view `[addr, addr+size)` of backing `backing`
    /// shares backing bytes with another guest view. The page tracker and
    /// the GPU caches work on guest addresses, so a CPU write through one
    /// view is not seen by what the caches hold for the other (and a GPU
    /// write mirrored for one view is not read back through the other).
    /// Logged for the first views, then counted ([`backing_alias_views`]).
    fn report_backing_alias(views: &views::BackingViews, addr: u64, size: u64, backing: u64) {
        let Some((there, here, bytes)) = views.shared(addr, size, backing) else { return };
        let n = BACKING_ALIAS_VIEWS.fetch_add(1, Ordering::Relaxed);
        if n < 16 || n.is_power_of_two() {
            log::warn!(
                "ps5gpu: guest memory {here:#x} and {there:#x} map the same {bytes:#x} backing bytes ({} such views so far): the GPU caches track each view on its own",
                n + 1
            );
        }
    }

    /// A guest mapping appeared with host protection `prot` (`PROT_*`).
    /// `backing` is its offset in the backing memfd (None for memory that
    /// has no alias; such ranges are never protected).
    pub fn note_mapped(&self, addr: u64, size: u64, prot: i32, backing: Option<u64>) {
        if addr == 0 || size == 0 || addr >= SPACE_BYTES || size > SPACE_BYTES - addr {
            return;
        }
        // A view of other backing bytes replaced in place is an unmap of
        // that view for the caches: what they hold for it (GPU results,
        // imports of its pages) belongs to the old bytes. The spans still
        // describe the old view here, so downloads land in the old bytes,
        // where the GPU wrote them on the console.
        let replaced = self.replaced_parts(addr, size, backing);
        // Parts mapped again onto the bytes they already showed keep their
        // contents: the caches keep what they hold there (GPU results not
        // yet downloaded included).
        let same = if same_backing_remap_enabled() { self.same_backing_parts(addr, size, backing) } else { smallvec::SmallVec::new() };
        if !replaced.is_empty() {
            let n = REPLACED_VIEWS.fetch_add(1, Ordering::Relaxed);
            if n < 8 || n.is_power_of_two() {
                log::info!("ps5gpu: guest memory {addr:#x}+{size:#x} (backing {backing:x?}) replaces views of other backing bytes in place: {replaced:x?} ({} such mappings so far)", n + 1);
            }
            if let Some(l) = self.unmap.get() {
                for &(a, n) in &replaced {
                    l.unmapped(a, n);
                }
            }
        }
        {
            let mut m = self.spans.write().unwrap_or_else(|e| e.into_inner());
            let mut views = self.views.lock().unwrap_or_else(|e| e.into_inner());
            Self::unindex(&mut views, &m, addr, addr + size);
            Self::remove_spans(&mut m, addr, addr + size);
            if let Some(b) = backing {
                Self::report_backing_alias(&views, addr, size, b);
                views.insert(addr, size, b);
            }
            m.insert(addr, Span { end: addr + size, backing });
            self.log_map(MapEvent::Mapped { addr, size, backing });
            history::note(history::Event::Map, addr, size, backing.unwrap_or(u64::MAX));
        }
        crate::capture::note_map(self, crate::capture::format::Event::Map { addr, size, backing, prot });
        // After the change: a reader of the old generation looks again.
        self.mapped.fetch_add(1, Ordering::AcqRel);
        if backing.is_some() {
            self.pages.register(addr, size, prot as u8);
        } else {
            self.pages.unregister(addr, size);
        }
        self.layout.fetch_add(1, Ordering::AcqRel);
        // The pages' contents changed under the caches: a page of a
        // partially resident resource was committed, or memory was mapped
        // over another mapping. Pages watched while uncommitted were
        // protected by `register` from their watch counts already.
        let mut cur = addr;
        for (a, n) in same.into_iter().chain(std::iter::once((addr + size, 0))) {
            if a > cur {
                if let Some(b) = self.buffers.get() {
                    b.mapped(cur, a - cur);
                }
                if let Some(i) = self.images.get() {
                    i.mapped(cur, a - cur);
                }
            }
            cur = cur.max(a + n);
        }
    }

    /// The parts of `[addr, addr+size)` a new mapping at `backing` maps onto
    /// the memfd bytes the current views there already show (in address
    /// order, merged).
    fn same_backing_parts(&self, addr: u64, size: u64, backing: Option<u64>) -> smallvec::SmallVec<[(u64, u64); 4]> {
        let mut out: smallvec::SmallVec<[(u64, u64); 4]> = smallvec::SmallVec::new();
        let Some(new) = backing else { return out };
        let end = addr + size;
        let m = self.spans();
        let first = m.range(..=addr).next_back().map_or(addr, |(&s, _)| s);
        for (&s, sp) in m.range(first..end) {
            let (a, e) = (s.max(addr), sp.end.min(end));
            if a >= e || sp.backing.map(|old| old + (a - s)) != Some(new + (a - addr)) {
                continue;
            }
            match out.last_mut() {
                Some(last) if last.0 + last.1 == a => last.1 += e - a,
                _ => out.push((a, e - a)),
            }
        }
        out
    }

    /// The parts of `[addr, addr+size)` that views of memfd bytes other
    /// than those a new mapping at `backing` puts there map now.
    fn replaced_parts(&self, addr: u64, size: u64, backing: Option<u64>) -> smallvec::SmallVec<[(u64, u64); 4]> {
        let end = addr + size;
        let m = self.spans();
        let first = m.range(..=addr).next_back().map_or(addr, |(&s, _)| s);
        let mut out: smallvec::SmallVec<[(u64, u64); 4]> = smallvec::SmallVec::new();
        for (&s, sp) in m.range(first..end) {
            let (a, e) = (s.max(addr), sp.end.min(end));
            let Some(old) = sp.backing else { continue };
            if a >= e || backing.map(|b| b + (a - addr)) == Some(old + (a - s)) {
                continue;
            }
            match out.last_mut() {
                Some(last) if last.0 + last.1 == a => last.1 += e - a,
                _ => out.push((a, e - a)),
            }
        }
        out
    }

    /// Diagnostics for a guest crash: how the GPU side sees the page
    /// holding `addr` (its view and backing offset, and imports of other
    /// guest ranges that still copy into those backing bytes).
    pub fn describe_address(&self, addr: u64) -> Option<String> {
        if addr >= SPACE_BYTES || addr < PAGE_BYTES {
            return None;
        }
        let m = self.spans.try_read().ok()?;
        let (&start, sp) = m.range(..=addr).next_back()?;
        if addr >= sp.end {
            return None;
        }
        let mut s = format!("view {start:#x}..{:#x}", sp.end);
        if let Some(b) = sp.backing {
            let off = b + (addr - start);
            s += &format!(" backing {b:#x} (address at {off:#x})");
            if let Ok(v) = STALE_IMPORTS.try_lock() {
                for i in v.iter().filter(|i| off >= i.imported && off < i.imported + i.size) {
                    s += &format!("; STALE import of guest {:#x}+{:#x} (imported backing {:#x}, now {:x?}) copies into it", i.addr, i.size, i.imported, i.current);
                }
            }
        } else {
            s += " (private)";
        }
        Some(s)
    }

    /// The guest changed the protection of a range. True when the range is
    /// registered and the derived protection was applied here (the caller
    /// must then not mprotect the range itself).
    pub fn note_protected(&self, addr: u64, size: u64, prot: i32) -> bool {
        addr < SPACE_BYTES && self.pages.set_guest_protection(addr, size, prot as u8)
    }

    /// The guest unmapped (or replaced) a range: the caches drop or
    /// download what they hold there first.
    pub fn note_unmapped(&self, addr: u64, size: u64) {
        if addr == 0 || size == 0 || addr >= SPACE_BYTES {
            return;
        }
        let overlaps = {
            let m = self.spans();
            m.range(..addr + size).any(|(&s, sp)| s < addr + size && sp.end > addr)
        };
        if !overlaps {
            return;
        }
        if let Some(l) = self.unmap.get() {
            l.unmapped(addr, size);
        }
        {
            let mut m = self.spans.write().unwrap_or_else(|e| e.into_inner());
            Self::unindex(&mut self.views.lock().unwrap_or_else(|e| e.into_inner()), &m, addr, addr + size);
            Self::remove_spans(&mut m, addr, addr + size);
            self.log_map(MapEvent::Unmapped { addr, size });
            history::note(history::Event::Unmap, addr, size, 0);
        }
        crate::capture::note_map(self, crate::capture::format::Event::Unmap { addr, size });
        self.unmapped.fetch_add(1, Ordering::AcqRel);
        self.pages.unregister(addr, size);
        self.layout.fetch_add(1, Ordering::AcqRel);
    }

    /// Changes whenever a mapping comes or goes.
    pub fn map_generation(&self) -> u64 {
        self.mapped.load(Ordering::Acquire).wrapping_mul(0x9E37_79B9).wrapping_add(self.unmap_generation())
    }

    /// Changes whenever a mapping goes (readable ranges stay readable
    /// while it stays).
    pub fn unmap_generation(&self) -> u64 {
        self.unmapped.load(Ordering::Acquire).wrapping_add(self.alias_base.load(Ordering::Acquire))
    }

    /// True when the whole range lies in registered (aliasable, watched)
    /// mappings.
    pub fn is_tracked(&self, addr: u64, size: u64) -> bool {
        self.pages.is_registered(addr, size)
    }

    /// Host pointer of the backing alias for `[addr, addr+size)` when the
    /// range lies in one aliasable mapping (None on a slice thread when
    /// guest writes of its wave are pending over the range: see
    /// [`pending`]).
    pub fn backing_ptr(&self, addr: u64, size: u64) -> Option<*mut u8> {
        if pending::overlaps(addr, size) {
            return None;
        }
        self.backing_ptr_raw(addr, size)
    }

    fn backing_ptr_raw(&self, addr: u64, size: u64) -> Option<*mut u8> {
        let base = self.alias_base.load(Ordering::Acquire);
        if base == 0 || size == 0 {
            return None;
        }
        // The span this thread resolved last, at unchanged span counters
        // (read before the lookup: a change made meanwhile moves them).
        let generation = self.mapped.load(Ordering::Acquire).wrapping_mul(0x9E37_79B9).wrapping_add(self.unmapped.load(Ordering::Acquire));
        let me = self as *const Self as usize;
        if let Ok(Some(p)) = SPAN_CACHE.try_with(|cell| {
            let mut all = cell.get();
            let i = all.iter().position(|c| {
                c.owner == me && c.generation == generation && c.base == base && addr >= c.start && addr.checked_add(size).is_some_and(|e| e <= c.end)
            })?;
            let c = all[i];
            if i != 0 {
                all.copy_within(0..i, 1);
                all[0] = c;
                cell.set(all);
            }
            Some((base + c.backing + (addr - c.start)) as *mut u8)
        }) {
            return Some(p);
        }
        let m = self.spans();
        let (&start, sp) = m.range(..=addr).next_back()?;
        if let Some(backing) = sp.backing {
            if sp.end - start <= self.alias_size.load(Ordering::Acquire).saturating_sub(backing) {
                let _ = SPAN_CACHE.try_with(|cell| {
                    let mut all = cell.get();
                    all.copy_within(0..SPAN_CACHE_ENTRIES - 1, 1);
                    all[0] = SpanCache { owner: me, generation, base, start, end: sp.end, backing };
                    cell.set(all);
                });
            }
        }
        if addr + size > sp.end {
            // Contiguous spans with contiguous backing also qualify.
            let mut cur_end = sp.end;
            let mut cur_backing = sp.backing? + (sp.end - start);
            while cur_end < addr + size {
                let next = m.get(&cur_end)?;
                if next.backing? != cur_backing {
                    return None;
                }
                cur_backing += next.end - cur_end;
                cur_end = next.end;
            }
        }
        let off = sp.backing? + (addr - start);
        (off + size <= self.alias_size.load(Ordering::Acquire)).then(|| (base + off) as *mut u8)
    }

    /// Copy guest memory without tripping the GPU watches: through the
    /// backing alias, or directly for untracked accessible memory.
    pub fn read_backing(&self, addr: u64, out: &mut [u8]) -> bool {
        if out.is_empty() {
            return true;
        }
        let ok = self.read_piece(addr, out) || {
            // Spans with discontiguous backing: copy span by span.
            let pieces = self.span_pieces(addr, out.len() as u64);
            pieces.len() > 1 && pieces.iter().all(|&(a, n)| self.read_piece(a, &mut out[(a - addr) as usize..(a - addr + n) as usize]))
        };
        if ok {
            pending::compose(addr, out);
        }
        ok
    }

    fn read_piece(&self, addr: u64, out: &mut [u8]) -> bool {
        if let Some(p) = self.backing_ptr_raw(addr, out.len() as u64) {
            // SAFETY: the alias maps the memfd range backing this guest range.
            unsafe { std::ptr::copy_nonoverlapping(p, out.as_mut_ptr(), out.len()) };
            crate::capture::note_read(self, addr, out.len() as u64);
            return true;
        }
        if !self.pages.is_registered(addr, out.len() as u64) && ps5vm::ranges::accessible(addr, out.len() as u64, false) {
            // SAFETY: tracked, readable guest memory that is never protected by us.
            unsafe { std::ptr::copy_nonoverlapping(addr as *const u8, out.as_mut_ptr(), out.len()) };
            crate::capture::note_read(self, addr, out.len() as u64);
            return true;
        }
        false
    }

    /// The guest mappings: (address, bytes, backing offset) in address
    /// order.
    pub fn spans_snapshot(&self) -> Vec<(u64, u64, Option<u64>)> {
        self.spans().iter().map(|(&addr, sp)| (addr, sp.end - addr, sp.backing)).collect()
    }

    /// `[addr, addr + size)` cut at span boundaries.
    fn span_pieces(&self, addr: u64, size: u64) -> Vec<(u64, u64)> {
        let m = self.spans();
        let end = addr + size;
        let mut out = Vec::new();
        let mut cur = addr;
        while cur < end {
            let next = match m.range(..=cur).next_back() {
                Some((_, sp)) if sp.end > cur => sp.end.min(end),
                _ => m.range(cur + 1..).next().map_or(end, |(&s, _)| s.min(end)),
            };
            out.push((cur, next - cur));
            cur = next;
        }
        out
    }

    /// The pieces of `[addr, addr+size)` that hold guest memory, in
    /// address order: `(start, len, aliased)`, cut at span boundaries.
    /// Aliased pieces lie in memfd-backed mappings (read and written
    /// through the backing alias, whatever their CPU protection); the
    /// others are untracked memory the CPU may access directly (`write`:
    /// writable). Everything else is uncommitted: a page of a partially
    /// resident range the guest has not mapped (yet).
    fn committed_pieces(&self, addr: u64, size: u64, write: bool) -> Vec<(u64, u64, bool)> {
        let Some(end) = addr.checked_add(size) else { return Vec::new() };
        let mut out = Vec::new();
        let direct = |a: u64, e: u64, out: &mut Vec<(u64, u64, bool)>| {
            for (s, e) in ps5vm::ranges::accessible_runs(a, e - a, write) {
                out.push((s, e - s, false));
            }
        };
        let m = self.spans();
        let mut cur = addr;
        while cur < end {
            match m.range(..=cur).next_back() {
                Some((_, sp)) if sp.end > cur => {
                    let e = sp.end.min(end);
                    if sp.backing.is_some() {
                        out.push((cur, e - cur, true));
                    } else {
                        direct(cur, e, &mut out);
                    }
                    cur = e;
                }
                _ => {
                    let e = m.range(cur + 1..).next().map_or(end, |(&s, _)| s.min(end));
                    direct(cur, e, &mut out);
                    cur = e;
                }
            }
        }
        out
    }

    /// The committed parts of `[addr, addr+size)` (adjacent pieces
    /// merged) and whether they cover all of it.
    pub fn committed(&self, addr: u64, size: u64) -> Commitment {
        let mut runs: Vec<(u64, u64)> = Vec::new();
        for (a, n, _) in self.committed_pieces(addr, size, false) {
            match runs.last_mut() {
                Some(last) if last.1 == a => last.1 = a + n,
                _ => runs.push((a, a + n)),
            }
        }
        let whole = size == 0 || (runs.len() == 1 && runs[0] == (addr, addr + size));
        Commitment { runs, whole }
    }

    /// The number of committed runs of `[addr, addr+size)` (adjacent
    /// pieces merged, as in [`Self::committed`]) and the end of the last
    /// one (0 without any).
    pub fn committed_extent(&self, addr: u64, size: u64) -> (u64, u64) {
        fn fold(it: &mut dyn Iterator<Item = layout_memo::Piece>) -> (u64, u64) {
            let (mut runs, mut last) = (0u64, 0u64);
            for (a, n, _, _) in it {
                if runs == 0 || last != a {
                    runs += 1;
                }
                last = a + n;
            }
            (runs, last)
        }
        if let Some(st) = self.layout_memo_stamp(addr, size) {
            return match layout_memo::with_pieces(self.id, st, addr, size, fold) {
                Some(v) => v,
                None => self.walk_layout(st, addr, size, fold),
            };
        }
        let c = self.committed(addr, size);
        (c.runs.len() as u64, c.runs.last().map_or(0, |r| r.1))
    }

    /// The stamp a memoised layout is valid at ([`layout_memo`]; None when
    /// the memo is off or the range is not in the space): the spans and
    /// the page registry here, and the guest's tracked ranges (which decide
    /// the untracked gaps of a range). Read before the pieces are walked;
    /// a change moves it only once it is complete, so pieces found while a
    /// change was under way carry the old stamp.
    fn layout_memo_stamp(&self, addr: u64, size: u64) -> Option<(u64, u64)> {
        if !layout_memo::enabled() || addr == 0 || size == 0 || addr >= SPACE_BYTES || size > SPACE_BYTES - addr {
            return None;
        }
        Some((self.layout.load(Ordering::Acquire), ps5vm::ranges::generation()))
    }

    /// Walk the committed pieces of `[addr, addr+size)`, remember them at
    /// `stamp` and give them to `f`.
    fn walk_layout<R>(&self, stamp: (u64, u64), addr: u64, size: u64, f: impl FnOnce(&mut dyn Iterator<Item = layout_memo::Piece>) -> R) -> R {
        // The aligned window around the range: the next ranges of a ring
        // in the same pool fall inside it.
        let (start, end) = layout_memo::window(addr, addr + size, SPACE_BYTES);
        let pieces: Vec<layout_memo::Piece> = self.committed_pieces(start, end - start, false).into_iter().map(|(a, n, aliased)| (a, n, aliased, aliased && self.pages.is_registered(a, n))).collect();
        layout_memo::insert(self.id, stamp, start, end, pieces);
        layout_memo::with_pieces(self.id, stamp, addr, size, f).expect("the range just walked")
    }

    /// True when the GPU caches can watch `[addr, addr+size)`: every
    /// committed page is registered (memfd-backed), so CPU writes fault,
    /// and the uncommitted pages of a partially resident range only change
    /// through a mapping, which [`Self::note_mapped`] reports to the
    /// caches. Untracked accessible memory makes the range unwatchable.
    pub fn is_watchable(&self, addr: u64, size: u64) -> bool {
        // Remembered pieces answer `true` only; anything else is asked
        // again (a page found unregistered may be the registry's "cannot
        // tell" for a block this thread holds locked).
        let all = |it: &mut dyn Iterator<Item = layout_memo::Piece>| Iterator::all(&mut &mut *it, |(_, _, aliased, registered)| aliased && registered);
        let Some(st) = self.layout_memo_stamp(addr, size) else { return self.is_watchable_uncached(addr, size) };
        if layout_memo::with_pieces(self.id, st, addr, size, all) == Some(true) || self.pages.is_registered(addr, size) {
            return true;
        }
        self.walk_layout(st, addr, size, all) || self.is_watchable_uncached(addr, size)
    }

    fn is_watchable_uncached(&self, addr: u64, size: u64) -> bool {
        if self.pages.is_registered(addr, size) {
            return true;
        }
        if addr == 0 || size == 0 || addr >= SPACE_BYTES || size > SPACE_BYTES - addr {
            return false;
        }
        self.committed_pieces(addr, size, false).iter().all(|&(a, n, aliased)| aliased && self.pages.is_registered(a, n))
    }

    /// [`Self::read_backing`] for resources in partially resident ranges:
    /// uncommitted pages read as zeros (as the GPU reads an unmapped page
    /// of a partially resident texture) and committed pages are copied.
    /// False only when a committed piece could not be read.
    pub fn read_committed(&self, addr: u64, out: &mut [u8]) -> bool {
        if out.is_empty() {
            return true;
        }
        if self.read_piece(addr, out) {
            pending::compose(addr, out);
            return true;
        }
        out.fill(0);
        let mut ok = true;
        for (a, n, _) in self.committed_pieces(addr, out.len() as u64, false) {
            ok &= self.read_piece(a, &mut out[(a - addr) as usize..(a - addr + n) as usize]);
        }
        ok
    }

    /// [`Self::write_backing`] that stores only into committed pages:
    /// bytes of uncommitted pages are dropped, never written anywhere.
    /// False only when a committed piece could not be written.
    pub fn write_committed(&self, addr: u64, data: &[u8]) -> bool {
        if data.is_empty() || self.write_piece(addr, data) {
            return true;
        }
        let mut ok = true;
        for (a, n, _) in self.committed_pieces(addr, data.len() as u64, true) {
            ok &= self.write_piece(a, &data[(a - addr) as usize..(a - addr + n) as usize]);
        }
        ok
    }

    /// Write guest memory without tripping the GPU watches.
    pub fn write_backing(&self, addr: u64, data: &[u8]) -> bool {
        if data.is_empty() {
            return true;
        }
        if self.write_piece(addr, data) {
            return true;
        }
        let pieces = self.span_pieces(addr, data.len() as u64);
        pieces.len() > 1 && pieces.iter().all(|&(a, n)| self.write_piece(a, &data[(a - addr) as usize..(a - addr + n) as usize]))
    }

    /// Change the guest bytes of `[addr, addr+len)` in place with `f`, as a
    /// [`Self::read_backing`] followed by a [`Self::write_backing`] of the
    /// result would, without the two copies. Only for a range in one
    /// aliasable mapping (false otherwise, and `f` is not called).
    pub fn modify_backing(&self, addr: u64, len: u64, f: impl FnOnce(&mut [u8])) -> bool {
        if len == 0 {
            return true;
        }
        let Some(p) = self.backing_ptr(addr, len) else { return false };
        crate::capture::note_read(self, addr, len);
        history::note(history::Event::HostWrite, addr, len, 0);
        // SAFETY: the alias maps the memfd range backing this guest range;
        // nothing else holds a Rust reference into guest memory.
        f(unsafe { std::slice::from_raw_parts_mut(p, len as usize) });
        crate::capture::note_write(self, addr, len);
        true
    }

    fn write_piece(&self, addr: u64, data: &[u8]) -> bool {
        history::note(history::Event::HostWrite, addr, data.len() as u64, 0);
        if let Some(p) = self.backing_ptr(addr, data.len() as u64) {
            // SAFETY: the alias maps the memfd range backing this guest range.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len()) };
            crate::capture::note_write(self, addr, data.len() as u64);
            return true;
        }
        if !self.pages.is_registered(addr, data.len() as u64) && ps5vm::ranges::accessible(addr, data.len() as u64, true) {
            // SAFETY: tracked, writable guest memory that is never protected by us.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), addr as *mut u8, data.len()) };
            crate::capture::note_write(self, addr, data.len() as u64);
            return true;
        }
        false
    }

    /// Resolve a fault on `addr`. True when the access should be retried.
    /// Runs in the signal handler: no allocation on this path.
    pub fn try_resolve_fault(&self, kind: FaultKind, addr: u64) -> bool {
        let page = addr & !(PAGE_BYTES - 1);
        let repeats = LAST_FAULT.with(|c| {
            let (p, n) = c.get();
            let n = if p == page { n + 1 } else { 0 };
            c.set((page, n));
            n
        });
        if repeats > 16 {
            // The tracker region may be locked by an upload or download in
            // progress on another thread: let it finish.
            // sched_yield: async-signal-safe.
            std::thread::yield_now();
        }
        if !self.pages.guest_permits(addr, kind) {
            self.declined.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        if repeats >= SETTLE_AFTER {
            return self.settle_fault(kind, addr, page, repeats);
        }
        self.resolve_quick(kind, addr, repeats)
    }

    /// A fault that kept coming back ([`SETTLE_AFTER`]): settled with the
    /// GPU side held still. A settle that leaves the page accessible
    /// restarts the count at [`SETTLE_AFTER`] (the next fault there, if
    /// any, settles again); [`SETTLE_TRIES`] that do not end in a decline
    /// with the page's state. Without a settle the quick path goes on up
    /// to [`TRANSIENT_RETRIES`].
    fn settle_fault(&self, kind: FaultKind, addr: u64, page: u64, repeats: u32) -> bool {
        let settled = self.buffers.get().and_then(|b| b.settle(addr, kind == FaultKind::Write));
        if settled == Some(true) {
            LAST_FAULT.with(|c| c.set((page, SETTLE_AFTER)));
            self.settled_faults.fetch_add(1, Ordering::Relaxed);
            self.resolved.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        if settled.is_none() && repeats < TRANSIENT_RETRIES {
            return self.resolve_quick(kind, addr, repeats);
        }
        if settled.is_some() && repeats < SETTLE_AFTER + SETTLE_TRIES {
            return true;
        }
        let (read_watch, write_watches) = self.pages.watchers(addr);
        log::error!(
            "ps5gpu: fault at {addr:#x} ({kind:?}) unresolved after {repeats} tries, {SETTLE_TRIES} of them with the GPU side held still (settle {settled:?}): read watch {read_watch}, write watches {write_watches}"
        );
        self.declined.fetch_add(1, Ordering::Relaxed);
        false
    }

    /// [`Self::try_resolve_fault`] without holding the GPU side still: the
    /// stores resolve what they can without waiting for the back-end.
    fn resolve_quick(&self, kind: FaultKind, addr: u64, repeats: u32) -> bool {
        const SIZE: u64 = 8;
        let mut handled = false;
        if let Some(i) = self.images.get() {
            handled |= i.cpu_access(addr, SIZE, kind == FaultKind::Write);
        }
        match kind {
            FaultKind::Write => crate::backend::profile::time(21, || {
                if let Some(b) = self.buffers.get() {
                    handled |= b.mark_cpu_write(addr, SIZE);
                }
                if let Some(i) = self.images.get() {
                    if i.mark_cpu_write(addr, SIZE) {
                        self.image_faults.fetch_add(1, Ordering::Relaxed);
                        handled = true;
                    }
                }
            }),
            // A read-through page became readable meanwhile: just retry.
            FaultKind::Read if self.pages.allows(addr, kind) => {}
            FaultKind::Read => {
                if let Some(b) = self.buffers.get() {
                    handled |= b.download_to_cpu(addr, SIZE);
                }
            }
        }
        // A watch may have been removed between the fault and this handler.
        if handled || self.pages.allows(addr, kind) {
            // A handled page may be GPU-written again before the access
            // retries (the GPU thread re-dirties it while a CPU thread polls
            // it): retried up to the repeat limit above.
            self.resolved.fetch_add(1, Ordering::Relaxed);
            match kind {
                FaultKind::Write => self.write_faults.fetch_add(1, Ordering::Relaxed),
                FaultKind::Read => self.read_faults.fetch_add(1, Ordering::Relaxed),
            };
            crate::capture::note_fault(self, addr, kind == FaultKind::Write);
            return true;
        }
        // No store handled the fault and the watches disallowed it at the
        // check above. A page still watched is retried a few times (the GPU
        // side may be changing it). A page unwatched since that check is
        // retried at once: its watch was removed under the region lock, which
        // also restored the page's host protection, so the access can
        // proceed. Declining it would kill the guest on a writable page.
        let (read_watch, write_watches) = self.pages.watchers(addr);
        let watched = read_watch || write_watches != 0;
        if !watched || repeats < TRANSIENT_RETRIES {
            return true;
        }
        log::warn!(
            "ps5gpu: fault at {addr:#x} ({kind:?}) unresolved: handled {handled}, read watch {read_watch}, write watches {write_watches}"
        );
        self.declined.fetch_add(1, Ordering::Relaxed);
        false
    }
}

static GLOBAL: OnceLock<GuestGpuMemory> = OnceLock::new();

/// The process-wide guest GPU memory manager.
pub fn global() -> &'static GuestGpuMemory {
    GLOBAL.get_or_init(GuestGpuMemory::new)
}

/// Description hook for fatal guest fault reports: the words at `addr`
/// (read through the backing alias, so GPU watches do not fault) and the
/// view of its page ([`GuestGpuMemory::describe_address`]).
pub fn describe_address(addr: u64) -> Option<String> {
    let m = GLOBAL.get()?;
    let d = m.describe_address(addr)?;
    let at = addr & !7;
    let mut out = format!("{d}");
    if let Some(h) = history::describe(addr, 24) {
        out += &format!("\n      history {h}");
    }
    // The words at the address and, one level down, what the pointers
    // among them point at.
    let mut words = [0u8; 0x80];
    if !m.read_backing(at, &mut words) {
        out += "\n      (words unreadable)";
        return Some(out);
    }
    let qwords: Vec<u64> = words.chunks(8).map(|w| u64::from_le_bytes(w.try_into().unwrap())).collect();
    out += &format!("\n      {at:#x}: {}", qwords.iter().map(|q| format!("{q:016x}")).collect::<Vec<_>>().join(" "));
    let mut seen: Vec<u64> = vec![addr & !(PAGE_BYTES - 1)];
    for (i, &q) in qwords.iter().enumerate() {
        let page = q & !(PAGE_BYTES - 1);
        if seen.contains(&page) {
            continue;
        }
        let Some(qd) = m.describe_address(q) else { continue };
        seen.push(page);
        let mut inner = [0u8; 0x100];
        let dump = if m.read_backing(q & !7, &mut inner) {
            inner.chunks(8).map(|w| format!("{:016x}", u64::from_le_bytes(w.try_into().unwrap()))).collect::<Vec<_>>().join(" ")
        } else {
            String::from("(unreadable)")
        };
        out += &format!("\n      +{:#x} -> {q:#x}: {qd}\n        {dump}", i * 8);
        if let Some(h) = history::describe(q, 24) {
            out += &format!("\n        history {h}");
        }
    }
    Some(out)
}

/// Fault hook for the CPU core: `(fault address, is write) -> retry`.
pub fn handle_fault(addr: u64, is_write: bool) -> bool {
    match GLOBAL.get() {
        Some(m) => m.try_resolve_fault(if is_write { FaultKind::Write } else { FaultKind::Read }, addr),
        None => false,
    }
}

#[cfg(test)]
mod tests;

/// Guest memory writes a slice thread's reads see before they happen (R7b).
///
/// The CPU writes of a wave (patches of the capture replay) run on the
/// committing thread at their positions after the wave was recorded. A slice
/// thread recording a draw at position `r` reads guest memory as it was when
/// the wave began, with the known writes positioned before `r` composed over
/// it: [`compose`] applies them to copies, [`super::GuestGpuMemory::backing_ptr`]
/// gives no pointer into a range they overlap.
pub mod pending {
    use std::cell::Cell;
    use std::collections::BTreeMap;

    /// A write: `len` bytes at `addr`, from `data` (null: zeros).
    #[derive(Clone, Copy, Debug)]
    pub struct Write {
        pub addr: u64,
        pub len: u64,
        pub data: *const u8,
    }

    /// The writes before the draw being recorded, in order, with the
    /// ranges they (and writes of unknown bytes) reach merged for lookups.
    #[derive(Default)]
    pub struct Pending {
        writes: Vec<Write>,
        /// Merged reached ranges: start -> end.
        spans: BTreeMap<u64, u64>,
    }

    impl Pending {
        pub fn clear(&mut self) {
            self.writes.clear();
            self.spans.clear();
        }

        /// A write of known bytes (`data`: `len` bytes, or None: zeros).
        pub fn push_bytes(&mut self, addr: u64, len: u64, data: Option<&[u8]>) {
            let len = data.map_or(len, |d| len.min(d.len() as u64));
            self.writes.push(Write { addr, len, data: data.map_or(std::ptr::null(), |d| d.as_ptr()) });
            self.reach(addr, len);
        }

        /// A range the wave's writes reach without known bytes.
        pub fn reach(&mut self, addr: u64, len: u64) {
            if len == 0 {
                return;
            }
            let (mut s, mut e) = (addr, addr.saturating_add(len));
            // Merge with every span touching [s, e).
            while let Some((&ps, &pe)) = self.spans.range(..=e).next_back() {
                if pe < s {
                    break;
                }
                self.spans.remove(&ps);
                s = s.min(ps);
                e = e.max(pe);
            }
            self.spans.insert(s, e);
        }

        pub fn overlaps(&self, addr: u64, size: u64) -> bool {
            let end = addr.saturating_add(size.max(1));
            self.spans.range(..end).next_back().is_some_and(|(_, &e)| e > addr)
        }

        pub fn is_empty(&self) -> bool {
            self.spans.is_empty()
        }

        fn compose(&self, addr: u64, out: &mut [u8]) {
            if !self.overlaps(addr, out.len() as u64) {
                return;
            }
            let end = addr + out.len() as u64;
            for w in &self.writes {
                let (s, e) = (w.addr.max(addr), (w.addr + w.len).min(end));
                if s >= e {
                    continue;
                }
                let dst = &mut out[(s - addr) as usize..(e - addr) as usize];
                if w.data.is_null() {
                    dst.fill(0);
                } else {
                    // SAFETY: `data` holds `len` bytes (see `set`).
                    let src = unsafe { std::slice::from_raw_parts(w.data.add((s - w.addr) as usize), (e - s) as usize) };
                    dst.copy_from_slice(src);
                }
            }
        }
    }

    thread_local! {
        static ACTIVE: Cell<*const Pending> = const { Cell::new(std::ptr::null()) };
    }

    /// Make `p` the calling thread's pending writes until [`clear`]; it and
    /// the bytes of its writes must stay alive (and unchanged) until then.
    pub fn set(p: &Pending) {
        ACTIVE.with(|a| a.set(p as *const Pending));
    }

    pub fn clear() {
        ACTIVE.with(|a| a.set(std::ptr::null()));
    }

    #[inline]
    fn with<R>(f: impl FnOnce(&Pending) -> R) -> Option<R> {
        let p = ACTIVE.with(|a| a.get());
        if p.is_null() {
            return None;
        }
        // SAFETY: `set`'s caller keeps it alive until `clear`.
        Some(f(unsafe { &*p }))
    }

    /// A pending write reaches `[addr, addr+size)`.
    #[inline]
    pub fn overlaps(addr: u64, size: u64) -> bool {
        with(|p| p.overlaps(addr, size)).unwrap_or(false)
    }

    /// Apply the pending writes over `out` (the bytes at `addr`).
    #[inline]
    pub fn compose(addr: u64, out: &mut [u8]) {
        with(|p| p.compose(addr, out));
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn writes_compose_in_order_and_only_while_set() {
            let data = [1u8, 2, 3, 4];
            let later = [9u8, 9];
            let mut p = Pending::default();
            p.push_bytes(0x1002, 4, Some(&data));
            p.push_bytes(0x1000, 3, None);
            p.push_bytes(0x1004, 2, Some(&later));
            p.reach(0x3000, 0x10);
            let mut out = [0xEEu8; 8];
            compose(0x1000, &mut out);
            assert_eq!(out, [0xEE; 8], "nothing pending");
            set(&p);
            assert!(overlaps(0x1005, 1) && !overlaps(0x1008, 8) && !overlaps(0x0FF0, 0x10) && overlaps(0x300F, 4) && !overlaps(0x2000, 0x1000));
            compose(0x1000, &mut out);
            // Zeros over the first bytes of the patch, then the later patch.
            assert_eq!(out, [0, 0, 0, 2, 9, 9, 0xEE, 0xEE]);
            clear();
            assert!(!overlaps(0x1000, 8));
        }

        #[test]
        fn reached_ranges_merge() {
            let mut p = Pending::default();
            for (a, n) in [(0x100, 0x10), (0x120, 0x10), (0x110, 0x10), (0x50, 0x20), (0x200, 1)] {
                p.reach(a, n);
            }
            assert_eq!(p.spans.iter().map(|(&s, &e)| (s, e)).collect::<Vec<_>>(), vec![(0x50, 0x70), (0x100, 0x130), (0x200, 0x201)]);
            assert!(p.overlaps(0x12F, 1) && !p.overlaps(0x130, 0xD0) && p.overlaps(0x6F, 0x92));
        }
    }
}
