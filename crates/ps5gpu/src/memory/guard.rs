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

fn apply(addr: u64, size: u64, prot: u8) {
    if size == 0 {
        return;
    }
    PROTECT_CALLS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: the range is part of a registered guest mapping owned by the
    // emulator; only its protection changes.
    #[cfg(unix)]
    let ok = unsafe { libc::mprotect(addr as *mut libc::c_void, size as usize, prot as i32) } == 0;
    #[cfg(windows)]
    let ok = ps5vm::os::protect(addr, size, ps5vm::MemProtect::from_prot_bits(prot as i32));
    if !ok {
        PROTECT_FAILURES.fetch_add(1, Ordering::Relaxed);
    }
}

/// Page watches and protection for the whole guest address space.
pub struct PageGuard {
    blocks: Box<[AtomicPtr<PageBlock>]>,
    gate: Mutex<()>,
}

impl Default for PageGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl PageGuard {
    pub fn new() -> Self {
        let blocks = super::zeroed_slice::<AtomicPtr<PageBlock>>(BLOCK_COUNT);
        Self { blocks, gate: Mutex::new(()) }
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
    fn reapply_locked(b: &PageBlock, base: u64, first: usize, end: usize) {
        // SAFETY: lock held by the caller.
        let (counts, guest) = unsafe { (&*b.counts.get(), &*b.guest.get()) };
        let mut run: Option<(usize, u8)> = None;
        for i in first..=end {
            let d = if i < end { derive(guest[i], counts[i]) } else { None };
            match (run, d) {
                (Some((s, p)), Some(nd)) if nd == p => {
                    let _ = s;
                }
                (Some((s, p)), _) => {
                    apply(base + s as u64 * PAGE_BYTES, (i - s) as u64 * PAGE_BYTES, p);
                    run = d.map(|nd| (i, nd));
                }
                (None, Some(nd)) => run = Some((i, nd)),
                (None, None) => {}
            }
        }
    }

    /// A guest mapping appeared with host protection `prot` (`PROT_*`).
    pub fn register(&self, addr: u64, size: u64, prot: u8) {
        for (base, first, end) in Self::chunks(addr, size) {
            let b = self.get_or_create(base);
            let _l = b.lock.lock();
            // SAFETY: lock held.
            let guest = unsafe { &mut *b.guest.get() };
            for g in &mut guest[first..end] {
                *g = (prot & PROT_MASK) | REGISTERED;
            }
            b.note_registered(guest);
            Self::reapply_locked(b, base, first, end);
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
                *g = (prot & PROT_MASK) | REGISTERED;
            }
            b.note_registered(guest);
            Self::reapply_locked(b, base, first, end);
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
        let (counts, guest) = unsafe { (&mut *b.counts.get(), &*b.guest.get()) };
        let mut run: Option<(usize, u8)> = None;
        for i in first..=end {
            let mut d = None;
            if i < end && mask.is_none_or(|m| m.get(i)) {
                let old = derive(guest[i], counts[i]);
                Self::change(&mut counts[i], track, block_reads);
                let new = derive(guest[i], counts[i]);
                if new != old {
                    d = new;
                }
            }
            match (run, d) {
                (Some((_, p)), Some(nd)) if nd == p => {}
                (Some((s, p)), _) => {
                    apply(base + s as u64 * PAGE_BYTES, (i - s) as u64 * PAGE_BYTES, p);
                    run = d.map(|nd| (i, nd));
                }
                (None, Some(nd)) => run = Some((i, nd)),
                (None, None) => {}
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
