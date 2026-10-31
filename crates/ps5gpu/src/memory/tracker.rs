// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Per-page CPU/GPU dirty tracking over the guest address space, in lazily
//! created 4 MiB [`TrackedRegion`]s. A missing region means "unknown" and
//! counts as CPU-dirty (it is created fully dirty).

use super::guard::PageGuard;
pub use super::region::Landing;
use super::region::{CpuDirtySummary, Ctx, TrackedRegion};
use super::{WriteOrigin, BLOCK_BYTES, BLOCK_COUNT, PAGE_BYTES, SPACE_BYTES};
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Slots of the [`DirtyLog`] ring.
const DIRTY_LOG_SLOTS: usize = 4096;

/// Ranges that became CPU-dirty, for consumers that must react to CPU
/// writes without querying the tracker per use (the guest-memory import
/// table points pages at mirrors only while they are current). Lock-free
/// and allocation-free: written from the fault handler.
pub struct DirtyLog {
    on: std::sync::atomic::AtomicBool,
    head: AtomicU64,
    /// (first 4 KiB page << 24) | page count (saturated).
    slots: Box<[AtomicU64]>,
    /// Sequence number + 1 of the entry a slot holds (published last).
    seqs: Box<[AtomicU64]>,
}

/// What [`GuestPageTracker::drain_dirty_log`] found.
pub enum DirtyDrain {
    /// The ranges logged since the previous drain.
    Ranges(Vec<(u64, u64)>),
    /// The ring overflowed (or an entry was not published in time): any
    /// page may have become CPU-dirty.
    Lost,
}

impl DirtyLog {
    fn new() -> DirtyLog {
        DirtyLog {
            on: std::sync::atomic::AtomicBool::new(false),
            head: AtomicU64::new(0),
            slots: (0..DIRTY_LOG_SLOTS).map(|_| AtomicU64::new(0)).collect(),
            seqs: (0..DIRTY_LOG_SLOTS).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    fn push(&self, addr: u64, size: u64) {
        if !self.on.load(Ordering::Relaxed) || size == 0 {
            return;
        }
        let first = addr / PAGE_BYTES;
        let count = ((addr + size).div_ceil(PAGE_BYTES) - first).min((1 << 24) - 1);
        let i = self.head.fetch_add(1, Ordering::AcqRel);
        let slot = (i % DIRTY_LOG_SLOTS as u64) as usize;
        self.slots[slot].store(first << 24 | count, Ordering::Relaxed);
        self.seqs[slot].store(i + 1, Ordering::Release);
    }
}

pub struct GuestPageTracker {
    pages: Arc<PageGuard>,
    dirty_log: DirtyLog,
    /// Next log entry the (single) consumer reads.
    dirty_tail: AtomicU64,
    regions: Box<[AtomicPtr<TrackedRegion>]>,
    present: Box<[AtomicU64]>,
    summary: CpuDirtySummary,
    gate: Mutex<()>,
}

/// The block index and region of the block a page walk asked last (see
/// [`GuestPageTracker::page_generation_in`]).
pub type BlockRegion<'a> = Option<(u64, Option<&'a TrackedRegion>)>;

fn valid(addr: u64, size: u64) -> bool {
    addr != 0 && size != 0 && addr < SPACE_BYTES && size <= SPACE_BYTES - addr
}

/// Split `[addr, addr+size)` into per-block pieces `(index, offset, bytes)`.
fn pieces(addr: u64, size: u64) -> impl Iterator<Item = (usize, u64, u64)> {
    let mut remaining = if valid(addr, size) { size } else { 0 };
    let mut index = (addr / BLOCK_BYTES) as usize;
    let mut offset = addr % BLOCK_BYTES;
    std::iter::from_fn(move || {
        if remaining == 0 {
            return None;
        }
        let bytes = (BLOCK_BYTES - offset).min(remaining);
        let r = (index, offset, bytes);
        remaining -= bytes;
        offset = 0;
        index += 1;
        Some(r)
    })
}

impl GuestPageTracker {
    pub fn new(pages: Arc<PageGuard>) -> Self {
        // Read the switches the fault handler asks outside of it.
        if super::region::fault_runs_enabled() {
            crate::backend::ab::release_fast("RPS5E_FAULT_RUNS");
        }
        Self {
            pages,
            dirty_log: DirtyLog::new(),
            dirty_tail: AtomicU64::new(0),
            regions: super::zeroed_slice(BLOCK_COUNT),
            present: super::zeroed_slice(BLOCK_COUNT.div_ceil(64)),
            summary: CpuDirtySummary::new(BLOCK_COUNT),
            gate: Mutex::new(()),
        }
    }

    pub fn pages(&self) -> &Arc<PageGuard> {
        &self.pages
    }

    /// Log the ranges that become CPU-dirty from now on (see [`DirtyLog`]).
    pub fn enable_dirty_log(&self) {
        self.dirty_tail.store(self.dirty_log.head.load(Ordering::Acquire), Ordering::Release);
        self.dirty_log.on.store(true, Ordering::Release);
    }

    /// Entries logged and not drained yet (lock-free check).
    pub fn dirty_log_pending(&self) -> bool {
        self.dirty_log.head.load(Ordering::Acquire) != self.dirty_tail.load(Ordering::Acquire)
    }

    /// The ranges that became CPU-dirty since the previous drain (one
    /// consumer).
    pub fn drain_dirty_log(&self) -> DirtyDrain {
        let log = &self.dirty_log;
        let tail = self.dirty_tail.load(Ordering::Acquire);
        let head = log.head.load(Ordering::Acquire);
        self.dirty_tail.store(head, Ordering::Release);
        if head - tail > DIRTY_LOG_SLOTS as u64 {
            return DirtyDrain::Lost;
        }
        let mut out = Vec::with_capacity((head - tail) as usize);
        for i in tail..head {
            let slot = (i % DIRTY_LOG_SLOTS as u64) as usize;
            let mut spins = 0;
            while log.seqs[slot].load(Ordering::Acquire) != i + 1 {
                spins += 1;
                if spins > 1 << 16 {
                    return DirtyDrain::Lost;
                }
                if spins < 64 {
                    std::hint::spin_loop();
                } else {
                    // The writer that claimed the slot may be preempted
                    // (it runs in a fault handler): let it finish.
                    std::thread::yield_now();
                }
            }
            let v = log.slots[slot].load(Ordering::Relaxed);
            // A writer that wrapped around meanwhile overwrote the slot.
            if log.head.load(Ordering::Acquire) - i > DIRTY_LOG_SLOTS as u64 {
                return DirtyDrain::Lost;
            }
            out.push(((v >> 24) * PAGE_BYTES, (v & ((1 << 24) - 1)) * PAGE_BYTES));
        }
        DirtyDrain::Ranges(out)
    }

    fn ctx(&self, fault: bool) -> Ctx<'_> {
        Ctx { pages: &self.pages, summary: &self.summary, fault }
    }

    fn find(&self, index: usize) -> Option<&TrackedRegion> {
        let p = self.regions.get(index)?.load(Ordering::Acquire);
        // SAFETY: regions are never freed once published.
        (!p.is_null()).then(|| unsafe { &*p })
    }

    fn get_or_create(&self, index: usize) -> &TrackedRegion {
        if let Some(r) = self.find(index) {
            return r;
        }
        let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(r) = self.find(index) {
            return r;
        }
        let r = Box::into_raw(Box::new(TrackedRegion::new(index as u64 * BLOCK_BYTES, &self.summary)));
        self.regions[index].store(r, Ordering::Release);
        self.present[index / 64].fetch_or(1 << (index % 64), Ordering::AcqRel);
        // SAFETY: just published, never freed.
        unsafe { &*r }
    }

    fn region(&self, index: usize, create: bool) -> Option<&TrackedRegion> {
        if create {
            Some(self.get_or_create(index))
        } else {
            self.find(index)
        }
    }

    pub fn cpu_dirty_epoch(&self) -> u64 {
        self.summary.epoch()
    }

    /// Lock-free: every block of the range has a region without CPU-dirty
    /// pages.
    pub fn known_cpu_clean(&self, addr: u64, size: u64) -> bool {
        if !valid(addr, size) {
            return false;
        }
        let last = (addr + size - 1) / BLOCK_BYTES;
        (addr / BLOCK_BYTES..=last).all(|i| self.find(i as usize).is_some() && !self.summary.is_dirty(i))
    }

    pub fn has_cpu_dirty_pages(&self, addr: u64, size: u64) -> bool {
        if self.known_cpu_clean(addr, size) {
            return false;
        }
        pieces(addr, size).any(|(i, off, bytes)| {
            let r = self.get_or_create(i);
            let _l = r.lock.lock();
            // SAFETY: lock held.
            unsafe { r.state() }.is_modified(WriteOrigin::Cpu, off, bytes)
        })
    }

    /// The content generation of the page holding `addr` (see
    /// [`super::region::PageGenerations`]; None without a region: every
    /// page of the block counts as CPU-dirty). Lock-free.
    pub fn page_generation(&self, addr: u64) -> Option<u32> {
        if !valid(addr, 1) {
            return None;
        }
        let r = self.find((addr / BLOCK_BYTES) as usize)?;
        Some(r.generations.get(((addr % BLOCK_BYTES) / PAGE_BYTES) as usize))
    }

    /// [`Self::page_generation`] for walks over consecutive pages: `block`
    /// keeps the region of the block asked last (regions are never freed),
    /// so pages of one block look it up once.
    pub fn page_generation_in<'a>(&'a self, block: &mut BlockRegion<'a>, addr: u64) -> Option<u32> {
        if !valid(addr, 1) {
            return None;
        }
        let index = addr / BLOCK_BYTES;
        let r = match *block {
            Some((i, r)) if i == index => r,
            _ => {
                let r = self.find(index as usize);
                *block = Some((index, r));
                r
            }
        }?;
        Some(r.generations.get(((addr % BLOCK_BYTES) / PAGE_BYTES) as usize))
    }

    /// The page holding `addr` may be written without a fault (CPU-dirty,
    /// or in a block without a region). Creates no region.
    pub fn page_maybe_cpu_dirty(&self, addr: u64) -> bool {
        if !valid(addr, 1) {
            return true;
        }
        let block = addr / BLOCK_BYTES;
        let Some(r) = self.find(block as usize) else { return true };
        if !self.summary.is_dirty(block) {
            return false;
        }
        let _l = r.lock.lock();
        // SAFETY: lock held.
        unsafe { r.state() }.is_modified(WriteOrigin::Cpu, (addr % BLOCK_BYTES) & !(PAGE_BYTES - 1), PAGE_BYTES)
    }

    /// Host code wrote guest bytes of the range without a fault (GPU
    /// contents copied into guest memory): their content generations move.
    pub fn note_written(&self, addr: u64, size: u64) {
        for (i, off, bytes) in pieces(addr, size) {
            if let Some(r) = self.find(i) {
                let _l = r.lock.lock();
                // SAFETY: lock held.
                unsafe { r.state() }.note_written(r.base + off, bytes);
            }
        }
    }

    /// Some page of the range is CPU-dirty and none is GPU-dirty (the
    /// guest bytes are the current data and the mirror is behind).
    pub fn cpu_dirty_only(&self, addr: u64, size: u64) -> bool {
        if !valid(addr, size) || self.known_cpu_clean(addr, size) {
            return false;
        }
        let mut cpu = false;
        for (i, off, bytes) in pieces(addr, size) {
            let r = self.get_or_create(i);
            let _l = r.lock.lock();
            // SAFETY: lock held.
            let st = unsafe { r.state() };
            if st.is_modified(WriteOrigin::Gpu, off, bytes) {
                return false;
            }
            cpu |= st.is_modified(WriteOrigin::Cpu, off, bytes);
        }
        cpu
    }

    pub fn has_gpu_dirty_pages(&self, addr: u64, size: u64) -> bool {
        pieces(addr, size).any(|(i, off, bytes)| {
            self.find(i).is_some_and(|r| {
                let _l = r.lock.lock();
                // SAFETY: lock held.
                unsafe { r.state() }.is_modified(WriteOrigin::Gpu, off, bytes)
            })
        })
    }

    /// Fault-path variant: None when a region lock is held by this thread.
    pub fn has_gpu_dirty_pages_in_fault(&self, addr: u64, size: u64) -> Option<bool> {
        for (i, off, bytes) in pieces(addr, size) {
            if let Some(r) = self.find(i) {
                let _l = r.lock.lock_for_fault()?;
                // SAFETY: lock held.
                if unsafe { r.state() }.is_modified(WriteOrigin::Gpu, off, bytes) {
                    return Some(true);
                }
            }
        }
        Some(false)
    }

    /// True when every block of the range has a region.
    pub fn has_region(&self, addr: u64, size: u64) -> bool {
        valid(addr, size) && (addr / BLOCK_BYTES..=(addr + size - 1) / BLOCK_BYTES).all(|i| self.find(i as usize).is_some())
    }

    fn mark(&self, addr: u64, size: u64, side: WriteOrigin, enable: bool, create: bool) {
        let ctx = self.ctx(false);
        for (i, off, bytes) in pieces(addr, size) {
            if let Some(r) = self.region(i, create) {
                let _l = r.lock.lock();
                // SAFETY: lock held.
                let s = unsafe { r.state() };
                if side == WriteOrigin::Cpu && enable {
                    s.mark_cpu_write(&ctx, r.base + off, bytes);
                } else {
                    s.change_state(&ctx, side, enable, r.base + off, bytes);
                }
            }
        }
    }

    pub fn mark_cpu_dirty_pages(&self, addr: u64, size: u64) {
        self.mark(addr, size, WriteOrigin::Cpu, true, true);
        self.dirty_log.push(addr, size);
    }

    pub fn mark_gpu_dirty_pages(&self, addr: u64, size: u64) {
        crate::backend::profile::note_gpu_write(addr, size);
        super::history::note(super::history::Event::GpuWritten, addr, size, 0);
        self.mark(addr, size, WriteOrigin::Gpu, true, true);
    }

    pub fn clear_gpu_dirty_pages(&self, addr: u64, size: u64) {
        super::history::note(super::history::Event::GpuCleared, addr, size, 0);
        self.mark(addr, size, WriteOrigin::Gpu, false, false);
    }

    /// GPU writes to the range are not written through (any more): its
    /// GPU-dirty pages block CPU reads.
    pub fn end_read_through(&self, addr: u64, size: u64) {
        let ctx = self.ctx(false);
        for (i, off, bytes) in pieces(addr, size) {
            if let Some(r) = self.find(i) {
                let _l = r.lock.lock();
                // SAFETY: lock held.
                unsafe { r.state() }.end_read_through(&ctx, r.base + off, bytes);
            }
        }
    }

    /// A copy of the GPU contents of the range's GPU-dirty pages into guest
    /// memory is recorded in `tick` (see [`super::region::RegionState::set_landing`]).
    pub fn set_landing(&self, addr: u64, size: u64, tick: u64) {
        for (i, off, bytes) in pieces(addr, size) {
            if let Some(r) = self.find(i) {
                let _l = r.lock.lock();
                // SAFETY: lock held.
                unsafe { r.state() }.set_landing(r.base + off, bytes, tick);
            }
        }
    }

    /// Before recording a copy of the GPU-dirty pages of the range into
    /// guest memory in `tick` (0: staged, written by the back-end later):
    /// the pages land with that copy (see
    /// [`super::region::RegionState::claim_landing`]); `visit` sees the
    /// runs still GPU-dirty, in address order, which are the ones to copy.
    pub fn claim_landing(&self, addr: u64, size: u64, tick: u64, visit: &mut dyn FnMut(u64, u64)) {
        for (i, off, bytes) in pieces(addr, size) {
            if let Some(r) = self.find(i) {
                let _l = r.lock.lock();
                // SAFETY: lock held.
                unsafe { r.state() }.claim_landing(r.base + off, bytes, tick, visit);
            }
        }
    }

    /// The landing tick of the GPU-dirty page holding `addr` (0: none, or
    /// the page is not GPU-dirty).
    pub fn landing(&self, addr: u64) -> u64 {
        let Some(r) = self.find((addr / BLOCK_BYTES) as usize) else { return 0 };
        let _l = r.lock.lock();
        // SAFETY: lock held.
        unsafe { r.state() }.landing(addr)
    }

    /// Fault path (allocation-free, no back-end): the GPU-dirty page holding
    /// `addr` becomes clean when its copy into guest memory completed by
    /// tick `completed`. None when the region lock is held by this thread.
    pub fn take_landed_in_fault(&self, addr: u64, completed: u64) -> Option<Landing> {
        if !valid(addr, 1) {
            return Some(Landing::Clean);
        }
        let Some(r) = self.find((addr / BLOCK_BYTES) as usize) else { return Some(Landing::Clean) };
        let ctx = self.ctx(true);
        let _l = r.lock.lock_for_fault()?;
        // SAFETY: lock held.
        Some(unsafe { r.state() }.take_landed(&ctx, addr, completed))
    }

    /// Stop tracking: every page becomes CPU-dirty (unwatched). GPU-dirty
    /// pages must have been downloaded first.
    pub fn untrack_memory(&self, addr: u64, size: u64) {
        let ctx = self.ctx(false);
        for (i, off, bytes) in pieces(addr, size) {
            if let Some(r) = self.find(i) {
                let _l = r.lock.lock();
                // SAFETY: lock held.
                unsafe { r.state() }.change_state(&ctx, WriteOrigin::Cpu, true, r.base + off, bytes);
            }
        }
    }

    /// A CPU write: mark it in every region whose bytes the GPU has not
    /// modified. Returns `(tracked, needs_gpu_flush)`; with a flush needed
    /// the caller downloads first and calls this again. Allocation-free;
    /// safe in the fault handler (`fault`). `None` when a lock could not be
    /// taken from the fault path.
    pub fn invalidate_region(&self, addr: u64, size: u64, fault: bool) -> Option<(bool, bool)> {
        let ctx = self.ctx(fault);
        let mut tracked = false;
        let mut flush = false;
        for (i, off, bytes) in pieces(addr, size) {
            let Some(r) = self.find(i) else { continue };
            tracked = true;
            let _l = if fault { r.lock.lock_for_fault()? } else { r.lock.lock() };
            // SAFETY: lock held.
            let s = unsafe { r.state() };
            if s.is_modified(WriteOrigin::Gpu, off, bytes) {
                flush = true;
                continue;
            }
            let (run, run_bytes) = if fault && super::region::fault_runs_enabled() && crate::backend::ab::release_fast("RPS5E_FAULT_RUNS") {
                s.fault_run(r.base + off, bytes)
            } else {
                (r.base + off, bytes)
            };
            if !s.mark_cpu_write(&ctx, run, run_bytes) {
                return None;
            }
            if run_bytes != bytes {
                // The opened pages may change without a fault from now on.
                self.dirty_log.push(run, run_bytes);
                super::note_fault_run(run, run_bytes);
            }
        }
        if tracked && !flush {
            self.dirty_log.push(addr, size);
            if fault {
                super::history::note(super::history::Event::CpuWrite, addr, size, 0);
            }
        }
        Some((tracked, flush))
    }

    /// Visit GPU-dirty runs of the range (holding the region locks), then
    /// clear them when `clear`.
    pub fn for_each_download_range(&self, addr: u64, size: u64, clear: bool, visit: &mut dyn FnMut(u64, u64)) {
        let regions: Vec<(&TrackedRegion, u64, u64)> =
            pieces(addr, size).filter_map(|(i, off, bytes)| self.find(i).map(|r| (r, off, bytes))).collect();
        for (r, _, _) in &regions {
            r.lock.enter();
        }
        let ctx = self.ctx(false);
        for (r, off, bytes) in &regions {
            // SAFETY: lock held.
            unsafe { r.state() }.for_each_modified_range(&ctx, WriteOrigin::Gpu, false, r.base + off, *bytes, visit);
        }
        if clear {
            for (r, off, bytes) in &regions {
                // SAFETY: lock held.
                unsafe { r.state() }.for_each_modified_range(&ctx, WriteOrigin::Gpu, true, r.base + off, *bytes, &mut |_, _| {});
            }
        }
        for (r, _, _) in &regions {
            r.lock.unlock();
        }
    }

    /// Plan and perform an upload of the CPU-dirty pages of the range.
    /// `range_fn` sees every run to upload (region locks held; the runs
    /// become clean and write-watched), `upload_fn` performs the upload and
    /// returns success. With `is_written` the range becomes GPU-dirty
    /// afterwards (locks held until then). A failed upload makes the
    /// cleared pages CPU-dirty again. `read` (a page's guest bytes in the
    /// backing alias) enables hot pages, see [`super::region`].
    #[allow(clippy::too_many_arguments)]
    pub fn for_each_upload_range(
        &self,
        addr: u64,
        size: u64,
        is_written: bool,
        read: Option<&dyn Fn(u64) -> Option<*const u8>>,
        range_fn: &mut dyn FnMut(u64, u64),
        upload_fn: &mut dyn FnMut() -> bool,
    ) -> bool {
        let none: &dyn Fn(u64) -> bool = &|_| false;
        self.for_each_upload_range_through(addr, size, is_written.then_some(none), read, range_fn, upload_fn)
    }

    /// [`Self::for_each_upload_range`] for a written range whose pages
    /// `through` selects are written through to guest memory after every
    /// GPU write: those stay CPU-readable (read-through) when they were.
    #[allow(clippy::too_many_arguments)]
    pub fn for_each_upload_range_through(
        &self,
        addr: u64,
        size: u64,
        written: Option<&dyn Fn(u64) -> bool>,
        read: Option<&dyn Fn(u64) -> Option<*const u8>>,
        range_fn: &mut dyn FnMut(u64, u64),
        upload_fn: &mut dyn FnMut() -> bool,
    ) -> bool {
        let is_written = written.is_some();
        let ctx = self.ctx(false);
        let regions: smallvec::SmallVec<[(&TrackedRegion, u64, u64); 4]> =
            pieces(addr, size).map(|(i, off, bytes)| (self.get_or_create(i), off, bytes)).collect();
        let mut cleared: smallvec::SmallVec<[(usize, u64, u64); 8]> = smallvec::SmallVec::new();
        for (idx, (r, off, bytes)) in regions.iter().enumerate() {
            r.lock.enter();
            // SAFETY: lock held.
            unsafe { r.state() }.for_each_cpu_upload_range(&ctx, r.base + off, *bytes, if is_written { None } else { read }, &mut |a, s| {
                super::history::note(super::history::Event::Upload, a, s, 0);
                cleared.push((idx, a, s));
                range_fn(a, s);
            });
            if !is_written {
                r.lock.unlock();
            }
        }
        let ok = upload_fn();
        if !ok {
            for &(idx, a, s) in &cleared {
                let r = regions[idx].0;
                if !is_written {
                    r.lock.enter();
                }
                // SAFETY: lock held.
                let st = unsafe { r.state() };
                st.change_state(&ctx, WriteOrigin::Cpu, true, a, s);
                st.forget_shadows();
                if !is_written {
                    r.lock.unlock();
                }
            }
        } else if let Some(through) = written {
            crate::backend::profile::note_gpu_write(addr, size);
            super::history::note(super::history::Event::GpuWritten, addr, size, 0);
            for (r, off, bytes) in &regions {
                // SAFETY: lock held.
                unsafe { r.state() }.mark_gpu_written(&ctx, r.base + off, *bytes, through);
            }
        }
        if is_written {
            for (r, _, _) in &regions {
                r.lock.unlock();
            }
        }
        ok
    }

    /// Visit maximal runs whose blocks may hold CPU-dirty pages (no region
    /// yet, or the summary bit set). Lock-free.
    pub fn for_each_possibly_cpu_dirty_range(&self, addr: u64, size: u64, visit: &mut dyn FnMut(u64, u64)) {
        if !valid(addr, size) {
            visit(addr, size);
            return;
        }
        let end = addr + size;
        let mut run: Option<u64> = None;
        let first = addr / BLOCK_BYTES;
        let last = (end - 1) / BLOCK_BYTES;
        for i in first..=last {
            let maybe = self.find(i as usize).is_none() || self.summary.is_dirty(i);
            match (maybe, run) {
                (true, None) => run = Some((i * BLOCK_BYTES).max(addr)),
                (false, Some(s)) => {
                    visit(s, i * BLOCK_BYTES - s);
                    run = None;
                }
                _ => {}
            }
        }
        if let Some(s) = run {
            visit(s, end - s);
        }
    }

    /// Debug: page-granular state of `addr` as (cpu_dirty, gpu_dirty).
    pub fn page_state(&self, addr: u64) -> Option<(bool, bool)> {
        let r = self.find((addr / BLOCK_BYTES) as usize)?;
        let _l = r.lock.lock();
        // SAFETY: lock held.
        let s = unsafe { r.state() };
        let off = (addr % BLOCK_BYTES) & !(PAGE_BYTES - 1);
        Some((s.is_modified(WriteOrigin::Cpu, off, PAGE_BYTES), s.is_modified(WriteOrigin::Gpu, off, PAGE_BYTES)))
    }
}
