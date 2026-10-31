// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Host page protection derived from the guest's own protection and the
//! GPU caches' watches.
//!
//! Every 4 KiB page of a registered guest mapping carries the guest
//! protection (as the host would apply it) and two watch counters: write
//! watches (the page is clean in some GPU cache; a CPU write must be seen)
//! and a read watch (the GPU owns newer data; any CPU access must be seen).
//! The host protection is `guest ∩ watchers`: a read watch makes the page
//! inaccessible, a write watch read-only.
//!
//! Pages outside registered mappings keep their counters but their host
//! protection is never touched.
//!
//! Pages of mappings registered with userfaultfd ([`super::uffd`]) carry
//! their write watches as userfaultfd write protection instead: their host
//! protection is then `guest` (none with a read watch), and the write
//! protection bit follows the write watch count.
//!
//! Per-block state sits behind a [`RegionLock`] so the fault handler can
//! consult it without allocating.

use super::lock::RegionLock;
use super::mask::PageMask;
use super::{FaultKind, BLOCK_BYTES, BLOCK_COUNT, PAGES_PER_BLOCK, PAGE_BYTES, SPACE_BYTES};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::Mutex;

/// Guest protection bits (host `PROT_*` values).
pub const PROT_READ: u8 = 1;
pub const PROT_WRITE: u8 = 2;
pub const PROT_EXEC: u8 = 4;
const REGISTERED: u8 = 0x80;
/// The page's write watches are userfaultfd write protection.
const UFFD: u8 = 0x40;
const PROT_MASK: u8 = PROT_READ | PROT_WRITE | PROT_EXEC;

const READ_WATCH: u8 = 0x80;
const WRITE_WATCH_MASK: u8 = 0x7F;

struct PageBlock {
    lock: RegionLock,
    /// Watch counters: bit 7 read watch, bits 0..7 write watch count.
    counts: UnsafeCell<[u8; PAGES_PER_BLOCK]>,
    /// Guest protection | REGISTERED.
    guest: UnsafeCell<[u8; PAGES_PER_BLOCK]>,
    /// Every page of the block is registered (lock-free reads; written
    /// with `lock` held: cleared before a page loses REGISTERED, set
    /// after the last page got it).
    all_registered: AtomicBool,
    /// Its [`UFFD`] pages keep their write watches as write protection
    /// now (written with `lock` held; see [`PageGuard::sync_mode`]).
    uffd_mode: UnsafeCell<bool>,
}

impl PageBlock {
    /// Recompute [`PageBlock::all_registered`] (lock held).
    fn note_registered(&self, guest: &[u8; PAGES_PER_BLOCK]) {
        let all = guest.iter().fold(REGISTERED, |a, &g| a & g) & REGISTERED != 0;
        self.all_registered.store(all, Ordering::Release);
    }
}

// SAFETY: the cells are only accessed with `lock` held.
unsafe impl Sync for PageBlock {}

/// Counters of protection changes (diagnostics).
pub static PROTECT_CALLS: AtomicU64 = AtomicU64::new(0);
pub static PROTECT_FAILURES: AtomicU64 = AtomicU64::new(0);
pub static WATCH_ERRORS: AtomicU64 = AtomicU64::new(0);
/// Userfaultfd write protection found on unwatched pages and dropped.
pub static STALE_WRITE_PROTECTION: AtomicU64 = AtomicU64::new(0);

fn allowed(counts: u8) -> u8 {
    if counts & READ_WATCH != 0 {
        0
    } else if counts & WRITE_WATCH_MASK != 0 {
        PROT_READ
    } else {
        PROT_READ | PROT_WRITE
    }
}

/// Host protection of a page, or None when the page is not registered.
fn derive(guest: u8, counts: u8) -> Option<u8> {
    if guest & REGISTERED == 0 {
        return None;
    }
    let guest = guest & PROT_MASK;
    let watchers = allowed(counts);
    if watchers == 0 {
        return Some(0);
    }
    let mut d = guest & (PROT_READ | PROT_EXEC);
    if watchers & PROT_WRITE != 0 {
        d |= guest & PROT_WRITE;
    }
    Some(d)
}

/// Host state of a page: its `mprotect` protection and its userfaultfd
/// write protection (always false for pages without [`UFFD`]); None when
/// unregistered.
fn derive_state(guest: u8, counts: u8, uffd_mode: bool) -> Option<(u8, bool)> {
    if guest & UFFD == 0 || !uffd_mode {
        return derive(guest, counts).map(|p| (p, false));
    }
    if guest & REGISTERED == 0 {
        return None;
    }
    let prot = if counts & READ_WATCH != 0 { 0 } else { guest & PROT_MASK };
    Some((prot, counts & WRITE_WATCH_MASK != 0))
}

/// Maximal runs of one kind of host change over a block's pages (see
/// [`PageGuard::update_block`]).
struct Runs<T: Copy + PartialEq> {
    /// The open run: first page and the value it gets.
    run: Option<(usize, T)>,
    /// Unchanged pages at the end of the run that already have its value
    /// (bridged when the run goes on after them).
    gap: Option<usize>,
}

impl<T: Copy + PartialEq> Runs<T> {
    fn new() -> Self {
        Runs { run: None, gap: None }
    }

    /// Page `i`: `change` is its new value when it changes, `current` its
    /// value when it does not (None: not part of any run). Returns a run
    /// that ended: (first page, end page, value).
    fn step(&mut self, i: usize, change: Option<T>, current: Option<T>, bridge: bool) -> Option<(usize, usize, T)> {
        let same = change.is_none() && bridge && self.run.is_some_and(|(_, p)| current == Some(p) && self.gap.is_none_or(|g| i - g < BRIDGE_PAGES));
        match (self.run, change) {
            (Some((_, p)), Some(nd)) if nd == p => {
                self.gap = None;
                None
            }
            (Some(_), None) if same => {
                self.gap.get_or_insert(i);
                None
            }
            (Some((s, p)), _) => {
                let stop = self.gap.take().unwrap_or(i);
                self.run = change.map(|nd| (i, nd));
                Some((s, stop, p))
            }
            (None, Some(nd)) => {
                self.run = Some((i, nd));
                None
            }
            (None, None) => None,
        }
    }
}

/// `RPS5E_PROTECT_BRIDGE=0`: one host protection change per run of
/// pages whose protection changes. By default a change of a block's pages
/// also covers the unchanged pages between two such runs that already
/// have the runs' protection (up to [`BRIDGE_PAGES`] in a row): one
/// `mprotect` instead of one per run, and so one TLB shootdown and fewer
/// split mappings (an upload re-watching the scattered pages the guest
/// rewrote in a 16 MiB buffer window changed ~8 runs, ~7 us each, in GTA V).
/// The bridged pages keep their protection. Not on Windows, where one
/// protection change must stay inside one allocation.
fn bridge_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    cfg!(unix) && *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_PROTECT_BRIDGE")) && crate::backend::ab::release_fast("protect-bridge")
}

#[cfg(test)]
thread_local! {
    /// Protection changes made by the calling thread (tests).
    pub static APPLIED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Unchanged pages a bridge spans at most.
const BRIDGE_PAGES: usize = 32;

fn apply(addr: u64, size: u64, prot: u8, why: crate::perf::sys::Call) {
    if size == 0 {
        return;
    }
    PROTECT_CALLS.fetch_add(1, Ordering::Relaxed);
    #[cfg(test)]
    APPLIED.with(|a| a.set(a.get() + 1));
    // SAFETY: the range is part of a registered guest mapping owned by the
    // emulator; only its protection changes.
    #[cfg(unix)]
    let ok = crate::perf::sys::timed_bytes(why, size, || unsafe { libc::mprotect(addr as *mut libc::c_void, size as usize, prot as i32) } == 0);
    #[cfg(windows)]
    let ok = crate::perf::sys::timed_bytes(why, size, || ps5vm::os::protect(addr, size, ps5vm::MemProtect::from_prot_bits(prot as i32)));
    if !ok {
        PROTECT_FAILURES.fetch_add(1, Ordering::Relaxed);
    }
}

/// Set or clear the userfaultfd write protection of a range (timed like
/// [`apply`]).
fn apply_wp(addr: u64, size: u64, on: bool, why: crate::perf::sys::Call) {
    if size == 0 {
        return;
    }
    PROTECT_CALLS.fetch_add(1, Ordering::Relaxed);
    #[cfg(test)]
    APPLIED.with(|a| a.set(a.get() + 1));
    if !crate::perf::sys::timed_bytes(why, size, || super::uffd::protect(addr, size, on)) {
        PROTECT_FAILURES.fetch_add(1, Ordering::Relaxed);
    }
}

/// Userfaultfd write watches are wanted now: available, not turned off
/// (`RPS5E_UFFD_WATCH=0`), and not in the slow period of an A/B run
/// (release name `uffd-watch`).
fn uffd_wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| super::uffd::init() && crate::backend::ab::switch_on("RPS5E_UFFD_WATCH")) && crate::backend::ab::release_fast("uffd-watch")
}

/// Page watches and protection for the whole guest address space.
pub struct PageGuard {
    blocks: Box<[AtomicPtr<PageBlock>]>,
    gate: Mutex<()>,
    /// Write watches of [`UFFD`] pages are write protection now (the mode
    /// new blocks start in; blocks follow it in [`Self::sync_mode`]).
    uffd_mode: AtomicBool,
}

impl Default for PageGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl PageGuard {
    pub fn new() -> Self {
        let blocks = super::zeroed_slice::<AtomicPtr<PageBlock>>(BLOCK_COUNT);
        // Read the switches now: the fault handler must not be the first.
        let _ = bridge_enabled();
        Self { blocks, gate: Mutex::new(()), uffd_mode: AtomicBool::new(uffd_wanted()) }
    }

    fn find(&self, addr: u64) -> Option<&PageBlock> {
        if addr >= SPACE_BYTES {
            return None;
        }
        let p = self.blocks[(addr / BLOCK_BYTES) as usize].load(Ordering::Acquire);
        // SAFETY: blocks are never freed once published.
        (!p.is_null()).then(|| unsafe { &*p })
    }

    fn get_or_create(&self, addr: u64) -> &PageBlock {
        if let Some(b) = self.find(addr) {
            return b;
        }
        let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(b) = self.find(addr) {
            return b;
        }
        let b = Box::into_raw(Box::new(PageBlock {
            lock: RegionLock::new(),
            counts: UnsafeCell::new([0; PAGES_PER_BLOCK]),
            guest: UnsafeCell::new([0; PAGES_PER_BLOCK]),
            all_registered: AtomicBool::new(false),
            uffd_mode: UnsafeCell::new(self.uffd_mode.load(Ordering::Acquire)),
        }));
        self.blocks[(addr / BLOCK_BYTES) as usize].store(b, Ordering::Release);
        // SAFETY: just published, never freed.
        unsafe { &*b }
    }

    /// Visit `[addr, addr+size)` (page-aligned outward) per block:
    /// `(block, block_base, first_page, end_page)`.
    fn chunks(addr: u64, size: u64) -> impl Iterator<Item = (u64, usize, usize)> {
        let begin = addr & !(PAGE_BYTES - 1);
        let end = if size == 0 { begin } else { ((addr + size - 1) & !(PAGE_BYTES - 1)) + PAGE_BYTES };
        let end = end.min(SPACE_BYTES);
        let mut cur = begin;
        std::iter::from_fn(move || {
            if cur >= end {
                return None;
            }
            let base = cur / BLOCK_BYTES * BLOCK_BYTES;
            let chunk_end = end.min(base + BLOCK_BYTES);
            let r = (base, ((cur - base) / PAGE_BYTES) as usize, ((chunk_end - base) / PAGE_BYTES) as usize);
            cur = chunk_end;
            Some(r)
        })
    }

    /// Re-apply the derived protection of pages `[first, end)` of a block
    /// in maximal runs (lock held).
    /// With `write_protect` the userfaultfd write protection of watched
    /// pages is set as well (a fresh registration has none).
    fn reapply_locked(b: &PageBlock, base: u64, first: usize, end: usize, write_protect: bool) {
        // SAFETY: lock held by the caller.
        let (counts, guest, mode) = unsafe { (&*b.counts.get(), &*b.guest.get(), *b.uffd_mode.get()) };
        let mut prot = Runs::<u8>::new();
        let mut wp = Runs::<bool>::new();
        for i in first..=end {
            let d = if i < end { derive_state(guest[i], counts[i], mode) } else { None };
            if let Some((s, stop, p)) = prot.step(i, d.map(|d| d.0), None, false) {
                apply(base + s as u64 * PAGE_BYTES, (stop - s) as u64 * PAGE_BYTES, p, crate::perf::sys::Call::Protect);
            }
            let w = d.filter(|d| write_protect && d.1).map(|_| true);
            if let Some((s, stop, _)) = wp.step(i, w, None, false) {
                apply_wp(base + s as u64 * PAGE_BYTES, (stop - s) as u64 * PAGE_BYTES, true, crate::perf::sys::Call::Protect);
            }
        }
    }

    /// A guest mapping appeared with host protection `prot` (`PROT_*`).
    pub fn register(&self, addr: u64, size: u64, prot: u8) {
        // Write watches through userfaultfd where the kernel takes the
        // mapping (memfd views), else through the protection.
        let begin = addr & !(PAGE_BYTES - 1);
        let stop = addr.saturating_add(size).div_ceil(PAGE_BYTES) * PAGE_BYTES;
        let uffd = if stop > begin && stop <= SPACE_BYTES && super::uffd::register(begin, stop - begin) { UFFD } else { 0 };
        for (base, first, end) in Self::chunks(addr, size) {
            let b = self.get_or_create(base);
            let _l = b.lock.lock();
            // SAFETY: lock held.
            let guest = unsafe { &mut *b.guest.get() };
            for g in &mut guest[first..end] {
                *g = (prot & PROT_MASK) | REGISTERED | uffd;
            }
            b.note_registered(guest);
            Self::reapply_locked(b, base, first, end, true);
        }
    }

    /// The guest mapping is gone: stop managing these pages' protection.
    pub fn unregister(&self, addr: u64, size: u64) {
        for (base, first, end) in Self::chunks(addr, size) {
            if let Some(b) = self.find(base) {
                let _l = b.lock.lock();
                // SAFETY: lock held.
                let guest = unsafe { &mut *b.guest.get() };
                b.all_registered.store(false, Ordering::Release);
                for g in &mut guest[first..end] {
                    *g = 0;
                }
            }
        }
    }

    /// The guest changed the protection of registered pages. Returns false
    /// (nothing applied) unless every page of the range is registered.
    pub fn set_guest_protection(&self, addr: u64, size: u64, prot: u8) -> bool {
        if !self.is_registered(addr, size) {
            return false;
        }
        for (base, first, end) in Self::chunks(addr, size) {
            let b = self.get_or_create(base);
            let _l = b.lock.lock();
            // SAFETY: lock held.
            let guest = unsafe { &mut *b.guest.get() };
            for g in &mut guest[first..end] {
                // The write protection bits survive a protection change.
                *g = (prot & PROT_MASK) | REGISTERED | (*g & UFFD);
            }
            b.note_registered(guest);
            Self::reapply_locked(b, base, first, end, false);
        }
        true
    }

    /// True when every page of the range belongs to a registered mapping.
    pub fn is_registered(&self, addr: u64, size: u64) -> bool {
        if addr >= SPACE_BYTES || size == 0 || size > SPACE_BYTES - addr {
            return false;
        }
        Self::chunks(addr, size).all(|(base, first, end)| {
            let Some(b) = self.find(base) else { return false };
            // A wholly registered block answers without its lock (unless
            // this thread holds it: the answer is then "cannot tell").
            if b.all_registered.load(Ordering::Acquire) && !b.lock.held_by_me() {
                return true;
            }
            let Some(_l) = b.lock.lock_for_fault() else { return false };
            // SAFETY: lock held.
            let guest = unsafe { &*b.guest.get() };
            // Branch-free AND over the pages (vectorises; buffers span
            // thousands of pages and this runs on every buffer use).
            guest[first..end].iter().fold(REGISTERED, |a, &g| a & g) & REGISTERED != 0
        })
    }

    /// Guest protection of the page holding `addr` (None when unregistered).
    pub fn guest_protection(&self, addr: u64) -> Option<u8> {
        let b = self.find(addr)?;
        let _l = b.lock.lock_for_fault()?;
        // SAFETY: lock held.
        let g = unsafe { (*b.guest.get())[((addr % BLOCK_BYTES) / PAGE_BYTES) as usize] };
        (g & REGISTERED != 0).then_some(g & PROT_MASK)
    }

    /// The guest itself allows the access.
    pub fn guest_permits(&self, addr: u64, kind: FaultKind) -> bool {
        self.guest_protection(addr).is_some_and(|p| match kind {
            FaultKind::Write => p & PROT_WRITE != 0,
            FaultKind::Read => p & (PROT_READ | PROT_WRITE | PROT_EXEC) != 0,
        })
    }

    /// The watchers (not the guest) allow the access.
    pub fn allows(&self, addr: u64, kind: FaultKind) -> bool {
        let Some(b) = self.find(addr) else { return true };
        let Some(_l) = b.lock.lock_for_fault() else { return false };
        // SAFETY: lock held.
        let c = unsafe { (*b.counts.get())[((addr % BLOCK_BYTES) / PAGE_BYTES) as usize] };
        let a = allowed(c);
        match kind {
            FaultKind::Write => a == PROT_READ | PROT_WRITE,
            FaultKind::Read => a != 0,
        }
    }

    /// Watch counters of the page holding `addr`: (read watch, write watches).
    pub fn watchers(&self, addr: u64) -> (bool, u8) {
        let Some(b) = self.find(addr) else { return (false, 0) };
        let Some(_l) = b.lock.lock_for_fault() else { return (false, 0) };
        // SAFETY: lock held.
        let c = unsafe { (*b.counts.get())[((addr % BLOCK_BYTES) / PAGE_BYTES) as usize] };
        (c & READ_WATCH != 0, c & WRITE_WATCH_MASK)
    }

    /// Follow a change of [`uffd_wanted`] (an A/B period): every block
    /// moves its [`UFFD`] pages' write watches between write protection
    /// and the page protection. Called between draws.
    pub fn sync_mode(&self) {
        self.sync_mode_to(uffd_wanted());
    }

    /// [`Self::sync_mode`] to `wanted` (false: page protection only).
    pub fn sync_mode_to(&self, wanted: bool) {
        if self.uffd_mode.load(Ordering::Acquire) == wanted {
            return;
        }
        self.uffd_mode.store(wanted, Ordering::Release);
        let t0 = std::time::Instant::now();
        let calls = PROTECT_CALLS.load(Ordering::Relaxed);
        self.convert_blocks(wanted);
        log::info!(
            "ps5gpu: write watches now {} ({} protection changes, {} us)",
            if wanted { "userfaultfd write protection" } else { "page protection" },
            PROTECT_CALLS.load(Ordering::Relaxed) - calls,
            t0.elapsed().as_micros()
        );
    }

    fn convert_blocks(&self, wanted: bool) {
        for (index, slot) in self.blocks.iter().enumerate() {
            let p = slot.load(Ordering::Acquire);
            if p.is_null() {
                continue;
            }
            // SAFETY: blocks are never freed once published.
            let b = unsafe { &*p };
            let base = index as u64 * BLOCK_BYTES;
            let _l = b.lock.lock();
            // SAFETY: lock held.
            let (counts, guest, mode) = unsafe { (&*b.counts.get(), &*b.guest.get(), &mut *b.uffd_mode.get()) };
            if *mode == wanted {
                continue;
            }
            let mut prot = Runs::<u8>::new();
            let mut wp = Runs::<bool>::new();
            for i in 0..=PAGES_PER_BLOCK {
                let (mut dp, mut dw) = (None, None);
                if i < PAGES_PER_BLOCK {
                    if let (Some(old), Some(new)) = (derive_state(guest[i], counts[i], *mode), derive_state(guest[i], counts[i], wanted)) {
                        dp = (new.0 != old.0).then_some(new.0);
                        dw = (new.1 != old.1).then_some(new.1);
                    }
                }
                if let Some((s, stop, p)) = prot.step(i, dp, None, false) {
                    apply(base + s as u64 * PAGE_BYTES, (stop - s) as u64 * PAGE_BYTES, p, crate::perf::sys::Call::Protect);
                }
                if let Some((s, stop, w)) = wp.step(i, dw, None, false) {
                    apply_wp(base + s as u64 * PAGE_BYTES, (stop - s) as u64 * PAGE_BYTES, w, crate::perf::sys::Call::Protect);
                }
            }
            *mode = wanted;
        }
    }

    /// The page holding `addr` keeps its write watches as userfaultfd
    /// write protection (fault path: true when the block is busy).
    pub fn write_protected(&self, addr: u64) -> bool {
        let Some(b) = self.find(addr) else { return false };
        // A busy block: the fault path's own retries handle it (as for a
        // read-only page whose block is busy).
        let Some(_l) = b.lock.lock_for_fault() else { return true };
        // SAFETY: lock held.
        let g = unsafe { (*b.guest.get())[((addr % BLOCK_BYTES) / PAGE_BYTES) as usize] };
        g & (UFFD | REGISTERED) == UFFD | REGISTERED
    }

    /// A userfaultfd write fault on the page of `addr` while it has no
    /// write watch: the protection is stale (a change that failed), so it
    /// goes. True when it went; false when the page is watched (the fault
    /// is a CPU write the watchers must see) or not managed here.
    pub fn drop_stale_write_protection(&self, addr: u64) -> bool {
        let Some(b) = self.find(addr) else { return false };
        let Some(_l) = b.lock.lock_for_fault() else { return false };
        let i = ((addr % BLOCK_BYTES) / PAGE_BYTES) as usize;
        // SAFETY: lock held.
        let (g, c) = unsafe { ((*b.guest.get())[i], (*b.counts.get())[i]) };
        if g & (UFFD | REGISTERED) != UFFD | REGISTERED || c & WRITE_WATCH_MASK != 0 {
            return false;
        }
        apply_wp(addr & !(PAGE_BYTES - 1), PAGE_BYTES, false, crate::perf::sys::Call::UnwatchWrites);
        STALE_WRITE_PROTECTION.fetch_add(1, Ordering::Relaxed);
        true
    }

    fn change(c: &mut u8, track: bool, block_reads: bool) {
        if block_reads {
            match (track, *c & READ_WATCH != 0) {
                (true, false) => *c |= READ_WATCH,
                (false, true) => *c &= !READ_WATCH,
                _ => {
                    WATCH_ERRORS.fetch_add(1, Ordering::Relaxed);
                }
            }
        } else {
            let n = *c & WRITE_WATCH_MASK;
            match (track, n) {
                // A saturated count refuses the add (counted). It must not
                // refuse the remove: a write fault resolves by removing the
                // watches, and a page that can never lose them keeps the
                // faulting store retrying for ever.
                (true, WRITE_WATCH_MASK) | (false, 0) => {
                    WATCH_ERRORS.fetch_add(1, Ordering::Relaxed);
                }
                (true, n) => *c = (*c & READ_WATCH) | (n + 1),
                (false, n) => *c = (*c & READ_WATCH) | (n - 1),
            }
        }
    }

    /// Update pages `[first, end)` of the block at `base` (only those set
    /// in `mask` when given) and re-protect the pages whose derived
    /// protection changed. With `fault` the lock is taken in fault mode;
    /// returns false when that failed.
    fn update_block(&self, base: u64, first: usize, end: usize, mask: Option<&PageMask>, track: bool, block_reads: bool, fault: bool) -> bool {
        let b = if track && !fault {
            self.get_or_create(base)
        } else {
            match self.find(base) {
                Some(b) => b,
                None => {
                    if !track {
                        WATCH_ERRORS.fetch_add(1, Ordering::Relaxed);
                    }
                    return !fault || !track;
                }
            }
        };
        let guard = if fault {
            match b.lock.lock_for_fault() {
                Some(g) => g,
                None => return false,
            }
        } else {
            b.lock.lock()
        };
        // SAFETY: lock held.
        let (counts, guest, mode) = unsafe { (&mut *b.counts.get(), &*b.guest.get(), *b.uffd_mode.get()) };
        let bridge = bridge_enabled();
        let why = match (track, block_reads) {
            (true, false) => crate::perf::sys::Call::WatchWrites,
            (false, false) => crate::perf::sys::Call::UnwatchWrites,
            (true, true) => crate::perf::sys::Call::WatchReads,
            (false, true) => crate::perf::sys::Call::UnwatchReads,
        };
        let mut prot = Runs::<u8>::new();
        let mut wp = Runs::<bool>::new();
        for i in first..=end {
            let (mut dp, mut dw, mut current) = (None, None, None);
            if i < end {
                if mask.is_none_or(|m| m.get(i)) {
                    let old = derive_state(guest[i], counts[i], mode);
                    Self::change(&mut counts[i], track, block_reads);
                    let new = derive_state(guest[i], counts[i], mode);
                    if let (Some(old), Some(new)) = (old, new) {
                        dp = (new.0 != old.0).then_some(new.0);
                        dw = (new.1 != old.1).then_some(new.1);
                    }
                }
                current = derive_state(guest[i], counts[i], mode);
            }
            if let Some((s, stop, p)) = prot.step(i, dp, current.map(|c| c.0), bridge) {
                apply(base + s as u64 * PAGE_BYTES, (stop - s) as u64 * PAGE_BYTES, p, why);
            }
            // Write protection is bridged over pages that already have it
            // (or lack it) like the run: one call, one TLB flush.
            let current_wp = current.filter(|_| guest[i] & UFFD != 0).map(|c| c.1);
            if let Some((s, stop, w)) = wp.step(i, dw, current_wp, true) {
                apply_wp(base + s as u64 * PAGE_BYTES, (stop - s) as u64 * PAGE_BYTES, w, why);
            }
        }
        drop(guard);
        true
    }

    fn update(&self, addr: u64, size: u64, track: bool, block_reads: bool, fault: bool) -> bool {
        let mut ok = true;
        for (base, first, end) in Self::chunks(addr, size) {
            ok &= self.update_block(base, first, end, None, track, block_reads, fault);
        }
        ok
    }

    pub fn add_watch(&self, addr: u64, size: u64, block_reads: bool) {
        self.update(addr, size, true, block_reads, false);
    }

    pub fn remove_watch(&self, addr: u64, size: u64, block_reads: bool) {
        self.update(addr, size, false, block_reads, false);
    }

    /// Fault-path variant of [`PageGuard::remove_watch`].
    pub fn remove_watch_in_fault(&self, addr: u64, size: u64, block_reads: bool) -> bool {
        self.update(addr, size, false, block_reads, true)
    }

    pub fn add_watch_mask(&self, block_base: u64, pages: &PageMask, block_reads: bool) {
        if let Some((first, end)) = pages.bounds() {
            self.update_block(block_base, first, end, Some(pages), true, block_reads, false);
        }
    }

    pub fn remove_watch_mask(&self, block_base: u64, pages: &PageMask, block_reads: bool, fault: bool) -> bool {
        match pages.bounds() {
            Some((first, end)) => self.update_block(block_base, first, end, Some(pages), false, block_reads, fault),
            None => true,
        }
    }
}
