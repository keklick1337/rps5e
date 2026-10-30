// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Page tracker state machine and fault resolution with real mprotect and a
//! real SIGSEGV handler.

use super::tracker::GuestPageTracker;
use super::*;
use std::sync::atomic::AtomicUsize;
use crate::testutil::{guest_mapping, perms, test_memory};

struct TestBuffers {
    tracker: Arc<GuestPageTracker>,
    mem: &'static GuestGpuMemory,
    downloads: AtomicUsize,
    /// What a "download" writes into the first byte of every page.
    gpu_byte: u8,
}

impl BufferStore for TestBuffers {
    fn mark_cpu_write(&self, addr: u64, size: u64) -> bool {
        match self.tracker.invalidate_region(addr, size, true) {
            Some((tracked, false)) => tracked,
            Some((_, true)) => {
                self.download_to_cpu(addr, size);
                matches!(self.tracker.invalidate_region(addr, size, true), Some((true, false)))
            }
            None => false,
        }
    }

    fn download_to_cpu(&self, addr: u64, size: u64) -> bool {
        if !self.tracker.has_region(addr, size) {
            return false;
        }
        let mut pages = [0u64; 64];
        let mut n = 0;
        self.tracker.for_each_download_range(addr & !(PAGE_BYTES - 1), PAGE_BYTES, true, &mut |a, s| {
            let mut p = a;
            while p < a + s && n < 64 {
                pages[n] = p;
                n += 1;
                p += PAGE_BYTES;
            }
        });
        for &p in &pages[..n] {
            self.mem.write_backing(p, &[self.gpu_byte]);
        }
        self.downloads.fetch_add(1, Ordering::SeqCst);
        true
    }
}

struct NoImages;
impl ImageStore for NoImages {
    fn mark_cpu_write(&self, _addr: u64, _size: u64) -> bool {
        false
    }
}
struct NoUnmap;
impl UnmapListener for NoUnmap {
    fn unmapped(&self, _addr: u64, _size: u64) {}
}

#[test]
fn page_tracker_state_machine_and_fault_hook() {
    let mem = test_memory();
    let len = 8 * BLOCK_BYTES as usize;
    let (view, _alias) = guest_mapping(mem, len);
    assert!(mem.is_tracked(view, len as u64));
    let read = |page: u64| mem.backing_ptr(page, PAGE_BYTES).map(|p| p as *const u8);

    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let buffers = Arc::new(TestBuffers { tracker: tracker.clone(), mem, downloads: AtomicUsize::new(0), gpu_byte: 0xAB });
    mem.attach(buffers.clone(), Arc::new(NoImages), Arc::new(NoUnmap));

    let a = view + BLOCK_BYTES - 4 * PAGE_BYTES; // straddles two blocks
    let size = 8 * PAGE_BYTES;
    // Fresh memory is CPU-dirty and unprotected.
    assert!(tracker.has_cpu_dirty_pages(a, size));
    assert_eq!(perms(a), "rw-");

    // --- upload: pages become clean and write-watched ---
    let mut runs = Vec::new();
    assert!(tracker.for_each_upload_range(a, size, false, Some(&read), &mut |x, n| runs.push((x, n)), &mut || true));
    assert_eq!(runs, vec![(a, 4 * PAGE_BYTES), (a + 4 * PAGE_BYTES, 4 * PAGE_BYTES)]);
    assert!(!tracker.has_cpu_dirty_pages(a, size));
    assert_eq!(perms(a), "r--");
    assert_eq!(perms(a + 7 * PAGE_BYTES), "r--");
    assert_eq!(mem.pages.watchers(a), (false, 1));

    // Reading is fine; writing faults, is marked and retried.
    // SAFETY: the page is mapped readable.
    assert_eq!(unsafe { std::ptr::read_volatile(a as *const u8) }, 0);
    let target = a + 5 * PAGE_BYTES + 16;
    // SAFETY: mapped guest view; the write faults and is resolved.
    unsafe { std::ptr::write_volatile(target as *mut u32, 0xDEAD_BEEF) };
    // SAFETY: as above.
    assert_eq!(unsafe { std::ptr::read_volatile(target as *const u32) }, 0xDEAD_BEEF);
    assert_eq!(tracker.page_state(target), Some((true, false)));
    assert_eq!(tracker.page_state(a), Some((false, false)));
    assert_eq!(perms(target), "rw-");
    assert_eq!(perms(a), "r--");
    // The write is visible through the alias.
    let mut back = [0u8; 4];
    assert!(mem.read_backing(target, &mut back));
    assert_eq!(u32::from_le_bytes(back), 0xDEAD_BEEF);
    // Writing through the alias does not fault or dirty anything.
    assert!(mem.write_backing(a + 8, &[1, 2, 3]));
    assert_eq!(tracker.page_state(a), Some((false, false)));

    // A second upload only visits the dirty page.
    let mut runs = Vec::new();
    tracker.for_each_upload_range(a, size, false, Some(&read), &mut |x, n| runs.push((x, n)), &mut || true);
    assert_eq!(runs, vec![(target & !(PAGE_BYTES - 1), PAGE_BYTES)]);
    assert_eq!(perms(target), "r--");

    // --- repeated writes: every upload re-arms the write watch ---
    let page = target & !(PAGE_BYTES - 1);
    // Two uploads so far; the one reaching HOT_UPLOADS makes it hot.
    for _ in 2..super::region::HOT_UPLOADS - 1 {
        // SAFETY: faulting write resolved by the handler.
        unsafe { std::ptr::write_volatile(target as *mut u8, 7) };
        assert_eq!(tracker.page_state(target), Some((true, false)));
        let mut runs = Vec::new();
        tracker.for_each_upload_range(a, size, false, Some(&read), &mut |x, n| runs.push((x, n)), &mut || true);
        assert_eq!(runs, vec![(page, PAGE_BYTES)]);
        assert_eq!(perms(target), "r--");
    }

    // --- a page rewritten again and again turns hot: unwatched, uploaded
    // once per synchronisation point ---
    // SAFETY: faulting write resolved by the handler.
    unsafe { std::ptr::write_volatile(target as *mut u8, 8) };
    let mut runs = Vec::new();
    tracker.for_each_upload_range(a, size, false, Some(&read), &mut |x, n| runs.push((x, n)), &mut || true);
    assert_eq!(runs, vec![(page, PAGE_BYTES)]);
    assert_eq!(perms(target), "rw-");
    assert_eq!(tracker.page_state(target), Some((true, false)));
    // Unchanged bytes are not uploaded again; changed ones are, without a fault.
    super::note_sync_point();
    let mut runs = Vec::new();
    tracker.for_each_upload_range(a, size, false, Some(&read), &mut |x, n| runs.push((x, n)), &mut || true);
    assert!(runs.is_empty(), "unchanged hot page uploaded: {runs:x?}");
    // SAFETY: writable page, no fault.
    unsafe { std::ptr::write_volatile(target as *mut u8, 9) };
    // Within one synchronisation interval the page is checked once.
    tracker.for_each_upload_range(a, size, false, Some(&read), &mut |x, n| runs.push((x, n)), &mut || true);
    assert!(runs.is_empty(), "hot page checked twice in a generation: {runs:x?}");
    super::note_sync_point();
    tracker.for_each_upload_range(a, size, false, Some(&read), &mut |x, n| runs.push((x, n)), &mut || true);
    assert_eq!(runs, vec![(page, PAGE_BYTES)]);
    // A GPU write cools it: cleared, then GPU-dirty.
    tracker.for_each_upload_range(page, PAGE_BYTES, true, None, &mut |_, _| {}, &mut || true);
    assert_eq!(tracker.page_state(target), Some((false, true)));
    assert_eq!(perms(target), "---");
    // SAFETY: the read faults and downloads.
    unsafe { std::ptr::read_volatile(target as *const u8) };
    assert_eq!(tracker.page_state(target), Some((false, false)));

    // --- GPU write: pages read-blocked, CPU access downloads ---
    let g = view + 2 * BLOCK_BYTES;
    tracker.for_each_upload_range(g, 2 * PAGE_BYTES, true, None, &mut |_, _| {}, &mut || true);
    assert_eq!(tracker.page_state(g), Some((false, true)));
    assert_eq!(perms(g), "---");
    let before = buffers.downloads.load(Ordering::SeqCst);
    // SAFETY: the read faults; the handler downloads and unblocks.
    let v = unsafe { std::ptr::read_volatile(g as *const u8) };
    assert_eq!(v, 0xAB);
    assert_eq!(buffers.downloads.load(Ordering::SeqCst), before + 1);
    assert_eq!(tracker.page_state(g), Some((false, false)));
    assert_eq!(perms(g), "r--");
    // A write to the second (still GPU-dirty) page downloads, then dirties.
    // SAFETY: faulting write resolved by the handler.
    unsafe { std::ptr::write_volatile((g + PAGE_BYTES + 1) as *mut u8, 0x11) };
    assert_eq!(tracker.page_state(g + PAGE_BYTES), Some((true, false)));
    // SAFETY: readable now.
    assert_eq!(unsafe { std::ptr::read_volatile((g + PAGE_BYTES) as *const u8) }, 0xAB);

    // --- read-through: GPU pages written through stay readable ---
    let t = view + 5 * BLOCK_BYTES;
    assert!(mem.write_backing(t, &[0x5A]));
    let first: &dyn Fn(u64) -> bool = &|p| p == t;
    tracker.for_each_upload_range_through(t, 2 * PAGE_BYTES, Some(first), None, &mut |_, _| {}, &mut || true);
    assert_eq!(tracker.page_state(t), Some((false, true)));
    assert_eq!(perms(t), "r--");
    assert_eq!(perms(t + PAGE_BYTES), "---");
    let before = buffers.downloads.load(Ordering::SeqCst);
    // SAFETY: readable without a fault.
    assert_eq!(unsafe { std::ptr::read_volatile(t as *const u8) }, 0x5A);
    assert_eq!(buffers.downloads.load(Ordering::SeqCst), before);
    // Written again (and through): still readable.
    tracker.for_each_upload_range_through(t, PAGE_BYTES, Some(first), None, &mut |_, _| {}, &mut || true);
    assert_eq!(perms(t), "r--");
    // A GPU write that is not written through blocks reads; a later
    // write-through does not unblock them (guest memory is stale).
    tracker.for_each_upload_range(t, PAGE_BYTES, true, None, &mut |_, _| {}, &mut || true);
    assert_eq!(perms(t), "---");
    tracker.for_each_upload_range_through(t, PAGE_BYTES, Some(first), None, &mut |_, _| {}, &mut || true);
    assert_eq!(perms(t), "---");
    // SAFETY: the read faults and downloads.
    assert_eq!(unsafe { std::ptr::read_volatile(t as *const u8) }, 0xAB);
    assert_eq!(perms(t), "r--");
    tracker.for_each_upload_range_through(t, PAGE_BYTES, Some(first), None, &mut |_, _| {}, &mut || true);
    assert_eq!(perms(t), "r--");
    tracker.end_read_through(t, PAGE_BYTES);
    assert_eq!(perms(t), "---");
    tracker.for_each_upload_range_through(t, PAGE_BYTES, Some(first), None, &mut |_, _| {}, &mut || true);
    // SAFETY: the read faults and downloads.
    assert_eq!(unsafe { std::ptr::read_volatile(t as *const u8) }, 0xAB);
    tracker.for_each_upload_range_through(t, PAGE_BYTES, Some(first), None, &mut |_, _| {}, &mut || true);
    assert_eq!(perms(t), "r--");
    // A CPU write to a read-through page downloads first.
    let before = buffers.downloads.load(Ordering::SeqCst);
    // SAFETY: faulting write resolved by the handler.
    unsafe { std::ptr::write_volatile((t + 1) as *mut u8, 0x22) };
    assert_eq!(buffers.downloads.load(Ordering::SeqCst), before + 1);
    assert_eq!(tracker.page_state(t), Some((true, false)));
    assert_eq!(perms(t), "rw-");

    // --- failed upload restores the CPU-dirty state ---
    let f = view + 3 * BLOCK_BYTES;
    assert!(!tracker.for_each_upload_range(f, PAGE_BYTES, false, None, &mut |_, _| {}, &mut || false));
    assert_eq!(tracker.page_state(f), Some((true, false)));
    assert_eq!(perms(f), "rw-");

    // --- guest protection is part of the derived protection ---
    let r = view + 4 * BLOCK_BYTES;
    tracker.for_each_upload_range(r, PAGE_BYTES, false, None, &mut |_, _| {}, &mut || true);
    assert_eq!(perms(r), "r--");
    assert!(mem.note_protected(r, PAGE_BYTES, libc::PROT_READ));
    assert!(!mem.pages.guest_permits(r, FaultKind::Write));
    tracker.untrack_memory(r, PAGE_BYTES);
    assert_eq!(perms(r), "r--", "guest read-only stays read-only without watches");
    assert!(mem.note_protected(r, PAGE_BYTES, libc::PROT_READ | libc::PROT_WRITE));
    assert_eq!(perms(r), "rw-");

    // --- untracked memory is never protected ---
    // SAFETY: anonymous mapping owned by the test.
    let anon = unsafe {
        libc::mmap(std::ptr::null_mut(), PAGE_BYTES as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
    } as u64;
    assert!(!mem.is_tracked(anon, PAGE_BYTES));
    mem.pages.add_watch(anon, PAGE_BYTES, false);
    assert_eq!(perms(anon), "rw-");
    mem.pages.remove_watch(anon, PAGE_BYTES, false);

    // --- unmap stops managing the range ---
    mem.note_unmapped(view + 6 * BLOCK_BYTES, BLOCK_BYTES);
    assert!(!mem.is_tracked(view + 6 * BLOCK_BYTES, PAGE_BYTES));
    assert!(mem.backing_ptr(view + 6 * BLOCK_BYTES, 16).is_none());
    assert!(mem.backing_ptr(view + 7 * BLOCK_BYTES, 16).is_some());
    assert!(mem.stats().faults_resolved >= 4);
    assert_eq!(super::region::STATE_CONFLICTS.load(Ordering::Relaxed), 0);
}

/// Page-guard watcher counting (shared watchers, read/write interplay,
/// ranges crossing a 4 MiB block, mask updates, concurrent watchers).
#[test]
fn page_guard_watcher_counts_and_protection() {
    let mem = test_memory();
    let (view, _) = guest_mapping(mem, 3 * BLOCK_BYTES as usize);
    let g = &mem.pages;
    let p = view + 5 * PAGE_BYTES;

    // Two write watchers on one page: the page stays read-only until both leave.
    g.add_watch(p, PAGE_BYTES, false);
    g.add_watch(p, 1, false);
    assert_eq!(g.watchers(p), (false, 2));
    assert_eq!(perms(p), "r--");
    g.remove_watch(p, PAGE_BYTES, false);
    assert_eq!(perms(p), "r--");
    g.remove_watch(p, PAGE_BYTES, false);
    assert_eq!(perms(p), "rw-");

    // Read watch dominates a write watch; dropping it falls back to read-only.
    g.add_watch(p, PAGE_BYTES, false);
    g.add_watch(p, PAGE_BYTES, true);
    assert_eq!(perms(p), "---");
    assert!(!g.allows(p, FaultKind::Read));
    g.remove_watch(p, PAGE_BYTES, true);
    assert_eq!(perms(p), "r--");
    assert!(g.allows(p, FaultKind::Read) && !g.allows(p, FaultKind::Write));
    g.remove_watch(p, PAGE_BYTES, false);
    assert!(g.allows(p, FaultKind::Write));

    // A range crossing a block boundary: every page changes, neighbours do not.
    let cross = view + BLOCK_BYTES - 2 * PAGE_BYTES;
    g.add_watch(cross, 4 * PAGE_BYTES, false);
    for i in 0..4 {
        assert_eq!(perms(cross + i * PAGE_BYTES), "r--", "page {i}");
    }
    assert_eq!(perms(cross - PAGE_BYTES), "rw-");
    assert_eq!(perms(cross + 4 * PAGE_BYTES), "rw-");
    g.remove_watch(cross, 4 * PAGE_BYTES, false);
    assert_eq!(perms(cross), "rw-");

    // Mask updates touch exactly the masked pages.
    let base = view + BLOCK_BYTES;
    let mut m = mask::PageMask::empty();
    m.set(3);
    m.set_range(10, 12);
    g.add_watch_mask(base, &m, false);
    assert_eq!(perms(base + 3 * PAGE_BYTES), "r--");
    assert_eq!(perms(base + 4 * PAGE_BYTES), "rw-");
    assert_eq!(perms(base + 11 * PAGE_BYTES), "r--");
    assert!(g.remove_watch_mask(base, &m, false, false));
    assert_eq!(perms(base + 11 * PAGE_BYTES), "rw-");

    // Concurrent shared watchers on the same pages balance out.
    let r = view + 2 * BLOCK_BYTES;
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| {
                for _ in 0..200 {
                    g.add_watch(r, 8 * PAGE_BYTES, false);
                    g.remove_watch(r, 8 * PAGE_BYTES, false);
                }
            });
        }
    });
    assert_eq!(g.watchers(r), (false, 0));
    assert_eq!(perms(r), "rw-");
    assert_eq!(guard::WATCH_ERRORS.load(Ordering::Relaxed), 0);
}

struct MapLog(std::sync::Mutex<Vec<(u64, u64)>>);
impl BufferStore for MapLog {
    fn mark_cpu_write(&self, _addr: u64, _size: u64) -> bool {
        false
    }
    fn download_to_cpu(&self, _addr: u64, _size: u64) -> bool {
        false
    }
    fn mapped(&self, addr: u64, size: u64) {
        self.0.lock().unwrap().push((addr, size));
    }
}

/// A mapping onto the bytes a range already shows changes nothing for the
/// caches: only the parts that were not mapped there (or showed other
/// bytes) are reported as mapped.
#[test]
fn remap_onto_the_same_bytes_is_not_new_contents() {
    use crate::testutil::{map_guest, remap_guest, unmap_guest};
    let mem = test_memory();
    let (view, _) = guest_mapping(mem, BLOCK_BYTES as usize);
    let log = Arc::new(MapLog(std::sync::Mutex::new(Vec::new())));
    mem.attach(log.clone(), Arc::new(NoImages), Arc::new(NoUnmap));
    unmap_guest(mem, view + 2 * PAGE_BYTES, PAGE_BYTES);
    log.0.lock().unwrap().clear();
    map_guest(mem, view, 4 * PAGE_BYTES);
    assert_eq!(*log.0.lock().unwrap(), vec![(view + 2 * PAGE_BYTES, PAGE_BYTES)]);
    // Other bytes in place: reported.
    log.0.lock().unwrap().clear();
    remap_guest(mem, view, PAGE_BYTES, 8 * PAGE_BYTES);
    assert_eq!(*log.0.lock().unwrap(), vec![(view, PAGE_BYTES)]);
}

/// A finite buffer over a partially resident range keeps the committed
/// pages past an uncommitted hole (they read through the binding, the hole
/// as zeros); a range ending in the hole, an oversized descriptor and an
/// unreadable start keep the plain clamp.
#[test]
fn finite_buffer_range_spans_uncommitted_holes() {
    use crate::pipeline::provider::extend_over_holes;
    use crate::testutil::unmap_guest;
    let mem = test_memory();
    let (view, _) = guest_mapping(mem, BLOCK_BYTES as usize);
    let a = view;
    // Pages 2..4 are not committed; the buffer spans pages 0..8.
    unmap_guest(mem, a + 2 * PAGE_BYTES, 2 * PAGE_BYTES);
    let size = 8 * PAGE_BYTES;
    let cut = Some((a, 2 * PAGE_BYTES));
    assert_eq!(extend_over_holes(mem, a, size, size, cut), Some((a, size)));
    // The bound size limits the extension.
    assert_eq!(extend_over_holes(mem, a, size, 6 * PAGE_BYTES, cut), Some((a, 6 * PAGE_BYTES)));
    // Nothing committed past the hole: the cut stays.
    assert_eq!(extend_over_holes(mem, a, 4 * PAGE_BYTES, 4 * PAGE_BYTES, cut), cut);
    // Oversized (1 GiB or more) descriptors keep the plain clamp.
    assert_eq!(extend_over_holes(mem, a, 1 << 30, size, cut), cut);
    assert_eq!(extend_over_holes(mem, a, size, size, None), None);
}

/// A partially resident range (pages the guest has not mapped): it can
/// be watched, its uncommitted pages read as zeros and are never written,
/// and committing a page is reported to the caches.
#[test]
fn partially_committed_range_reads_zeros_and_stays_watchable() {
    use crate::testutil::{map_guest, unmap_guest};
    let mem = test_memory();
    let len = 2 * BLOCK_BYTES;
    let (view, alias) = guest_mapping(mem, len as usize);
    let log = Arc::new(MapLog(std::sync::Mutex::new(Vec::new())));
    mem.attach(log.clone(), Arc::new(NoImages), Arc::new(NoUnmap));
    let a = view + BLOCK_BYTES - 4 * PAGE_BYTES;
    let size = 8 * PAGE_BYTES;
    // Every byte holds its page index + 1 (through the alias: the memfd
    // keeps the bytes of unmapped pages).
    for i in 0..8u64 {
        // SAFETY: the alias maps the whole test memfd.
        unsafe { std::ptr::write_bytes((alias + (a - view) + i * PAGE_BYTES) as *mut u8, i as u8 + 1, PAGE_BYTES as usize) };
    }
    // Pages 2..4 and 6 are not committed.
    unmap_guest(mem, a + 2 * PAGE_BYTES, 2 * PAGE_BYTES);
    unmap_guest(mem, a + 6 * PAGE_BYTES, PAGE_BYTES);
    assert!(!mem.is_tracked(a, size), "not every page is registered");
    assert!(mem.is_watchable(a, size), "the committed pages can be watched");
    let c = mem.committed(a, size);
    assert!(!c.whole);
    assert_eq!(c.runs, vec![(a, a + 2 * PAGE_BYTES), (a + 4 * PAGE_BYTES, a + 6 * PAGE_BYTES), (a + 7 * PAGE_BYTES, a + 8 * PAGE_BYTES)]);
    assert!(mem.committed(a, 2 * PAGE_BYTES).whole);

    // Reads: zeros where nothing is committed.
    let mut out = vec![0xEEu8; size as usize];
    assert!(!mem.read_backing(a, &mut out), "plain reads still refuse holes");
    assert!(mem.read_committed(a, &mut out));
    for i in 0..8u64 {
        let want = if matches!(i, 2 | 3 | 6) { 0 } else { i as u8 + 1 };
        assert!(out[(i * PAGE_BYTES) as usize..((i + 1) * PAGE_BYTES) as usize].iter().all(|&b| b == want), "page {i}");
    }
    // Writes: the uncommitted pages' backing is left alone.
    assert!(mem.write_committed(a, &vec![0x55u8; size as usize]));
    for i in 0..8u64 {
        // SAFETY: the alias maps the whole test memfd.
        let b = unsafe { std::ptr::read_volatile((alias + (a - view) + i * PAGE_BYTES + 100) as *const u8) };
        let want = if matches!(i, 2 | 3 | 6) { i as u8 + 1 } else { 0x55 };
        assert_eq!(b, want, "page {i}");
    }

    // A watch over the whole range protects the committed pages only; a
    // page committed later comes in already watched and is reported.
    mem.pages.add_watch(a, size, false);
    assert_eq!(perms(a), "r--");
    assert_eq!(perms(a + 2 * PAGE_BYTES), "---");
    log.0.lock().unwrap().clear();
    map_guest(mem, a + 6 * PAGE_BYTES, PAGE_BYTES);
    assert_eq!(*log.0.lock().unwrap(), vec![(a + 6 * PAGE_BYTES, PAGE_BYTES)]);
    assert_eq!(perms(a + 6 * PAGE_BYTES), "r--", "the watch count survived the hole");
    assert!(mem.is_watchable(a, size));
    let mut page = vec![0u8; PAGE_BYTES as usize];
    assert!(mem.read_committed(a + 6 * PAGE_BYTES, &mut page));
    assert!(page.iter().all(|&b| b == 7), "the committed page shows its memory");
    mem.pages.remove_watch(a, size, false);
    assert_eq!(perms(a + 6 * PAGE_BYTES), "rw-");

    // Untracked accessible memory inside the range: not watchable.
    // SAFETY: replaces a test guest page with private anonymous memory.
    let r = unsafe {
        libc::mmap((a + 2 * PAGE_BYTES) as *mut libc::c_void, PAGE_BYTES as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED, -1, 0)
    };
    assert_ne!(r, libc::MAP_FAILED);
    ps5vm::ranges::track(a + 2 * PAGE_BYTES, PAGE_BYTES, ps5vm::MemProtect::RW);
    mem.note_mapped(a + 2 * PAGE_BYTES, PAGE_BYTES, libc::PROT_READ | libc::PROT_WRITE, None);
    assert!(!mem.is_watchable(a, size));
    assert!(mem.read_committed(a, &mut out));
    assert!(out[(2 * PAGE_BYTES) as usize..(3 * PAGE_BYTES) as usize].iter().all(|&b| b == 0), "fresh anonymous memory");
    assert_eq!(out[(3 * PAGE_BYTES) as usize], 0, "still uncommitted");
}

/// Landing copies: a GPU-dirty page whose copy into guest memory is
/// recorded at a tick becomes clean (readable, write-watched) on a CPU
/// access once that tick completed, without the buffer cache; before, the
/// access learns the tick to wait for. A later GPU write of the page
/// forgets the copy, and clean or copy-less pages are reported as such.
#[test]
fn landing_copies_make_gpu_pages_clean_without_the_cache() {
    use super::tracker::Landing;
    let mem = test_memory();
    let (view, _) = guest_mapping(mem, 2 * BLOCK_BYTES as usize);
    let tracker = GuestPageTracker::new(mem.pages.clone());
    // Three GPU-written pages across the block boundary.
    let g = view + BLOCK_BYTES - PAGE_BYTES;
    tracker.for_each_upload_range(g, 3 * PAGE_BYTES, true, None, &mut |_, _| {}, &mut || true);
    assert_eq!(perms(g), "---");
    assert_eq!(tracker.take_landed_in_fault(g, 100), Some(Landing::Missing));
    // Copied out at tick 7 (the first two pages only).
    tracker.set_landing(g, 2 * PAGE_BYTES, 7);
    assert_eq!(tracker.landing(g), 7);
    assert_eq!(tracker.landing(g + PAGE_BYTES), 7);
    assert_eq!(tracker.landing(g + 2 * PAGE_BYTES), 0);
    assert_eq!(tracker.take_landed_in_fault(g + 8, 6), Some(Landing::InFlight(7)));
    assert_eq!(tracker.page_state(g), Some((false, true)), "still GPU-dirty while in flight");
    assert_eq!(tracker.take_landed_in_fault(g + 8, 7), Some(Landing::Landed));
    assert_eq!(tracker.page_state(g), Some((false, false)));
    assert_eq!(perms(g), "r--", "clean pages are write-watched");
    assert_eq!(tracker.take_landed_in_fault(g, 7), Some(Landing::Clean));
    assert_eq!(perms(g + 2 * PAGE_BYTES), "---");
    assert_eq!(tracker.take_landed_in_fault(g + 2 * PAGE_BYTES, 100), Some(Landing::Missing));
    // A GPU write after the copy: the page's guest bytes are old again.
    tracker.for_each_upload_range(g + PAGE_BYTES, PAGE_BYTES, true, None, &mut |_, _| {}, &mut || true);
    assert_eq!(tracker.landing(g + PAGE_BYTES), 0);
    assert_eq!(tracker.take_landed_in_fault(g + PAGE_BYTES, 100), Some(Landing::Missing));
    // A landing noted on clean pages means nothing.
    tracker.set_landing(g, PAGE_BYTES, 9);
    assert_eq!(tracker.landing(g), 0);
    // Read-through pages (written through after every GPU write) lose the
    // read-through state with the GPU-dirty one.
    let t = view + 4 * PAGE_BYTES;
    let all: &dyn Fn(u64) -> bool = &|_| true;
    tracker.for_each_upload_range_through(t, PAGE_BYTES, Some(all), None, &mut |_, _| {}, &mut || true);
    assert_eq!(perms(t), "r--");
    tracker.set_landing(t, PAGE_BYTES, 11);
    assert_eq!(tracker.take_landed_in_fault(t, 11), Some(Landing::Landed));
    assert_eq!(tracker.page_state(t), Some((false, false)));
    assert_eq!(perms(t), "r--");
    // Downloads clearing the pages forget their copies too.
    tracker.for_each_upload_range(t, PAGE_BYTES, true, None, &mut |_, _| {}, &mut || true);
    tracker.set_landing(t, PAGE_BYTES, 12);
    tracker.clear_gpu_dirty_pages(t, PAGE_BYTES);
    tracker.for_each_upload_range(t, PAGE_BYTES, true, None, &mut |_, _| {}, &mut || true);
    assert_eq!(tracker.landing(t), 0);
}

/// A write fault on a page whose write watch another thread removes while
/// the handler is checking it must retry. The watch owner restores the
/// page's host protection under the region lock before it releases the
/// lock, so the access succeeds; declining it kills the guest.
#[test]
fn write_fault_retries_when_its_watch_is_removed_during_the_check() {
    let mem = test_memory();
    let (view, _alias) = guest_mapping(mem, BLOCK_BYTES as usize);
    let page = view + 16 * PAGE_BYTES;
    let other = view + 17 * PAGE_BYTES;
    let stop = std::sync::atomic::AtomicBool::new(false);
    let declined = AtomicUsize::new(0);
    std::thread::scope(|s| {
        s.spawn(|| {
            // Watch and unwatch the page: each removal is a window in which
            // the handler's watch snapshots can disagree.
            while !stop.load(Ordering::Acquire) {
                mem.pages.add_watch(page, PAGE_BYTES, false);
                mem.pages.remove_watch(page, PAGE_BYTES, false);
            }
        });
        for _ in 0..200_000 {
            // A call on another page in between keeps the handler's count of
            // consecutive faults on `page` from growing.
            assert!(mem.try_resolve_fault(FaultKind::Write, other));
            if !mem.try_resolve_fault(FaultKind::Write, page) {
                declined.fetch_add(1, Ordering::SeqCst);
            }
        }
        stop.store(true, Ordering::Release);
    });
    assert_eq!(declined.load(Ordering::SeqCst), 0);
    assert_eq!(mem.pages.watchers(page), (false, 0));
}

/// Memory in the AMM window (8 TiB) is tracked like user memory: after an
/// upload its pages are write-watched, a CPU write faults, is marked dirty
/// and retried. Outside the tracked space the pages could not be watched
/// and every use of an image there loaded it again.
#[test]
fn amm_window_memory_is_tracked() {
    let mem = test_memory();
    let len = 2 * BLOCK_BYTES as usize;
    let (view, _alias) = crate::testutil::amm_guest_mapping(mem, len);
    assert!(view >= 0x800_0000_0000, "the mapping is in the AMM window");
    assert!(mem.is_tracked(view, len as u64), "AMM memory is registered");
    assert!(mem.is_watchable(view, len as u64), "AMM memory can be watched");
    let read = |page: u64| mem.backing_ptr(page, PAGE_BYTES).map(|p| p as *const u8);
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let buffers = Arc::new(TestBuffers { tracker: tracker.clone(), mem, downloads: AtomicUsize::new(0), gpu_byte: 0xAB });
    mem.attach(buffers.clone(), Arc::new(NoImages), Arc::new(NoUnmap));

    let a = view + BLOCK_BYTES - 2 * PAGE_BYTES;
    let size = 4 * PAGE_BYTES;
    let mut runs = Vec::new();
    assert!(tracker.for_each_upload_range(a, size, false, Some(&read), &mut |x, n| runs.push((x, n)), &mut || true));
    assert!(!runs.is_empty(), "the pages were uploaded");
    assert!(!tracker.has_cpu_dirty_pages(a, size));
    assert_eq!(perms(a), "r--", "uploaded pages are write-watched");
    let target = a + 3 * PAGE_BYTES + 8;
    // SAFETY: mapped guest view; the write faults and is resolved.
    unsafe { std::ptr::write_volatile(target as *mut u32, 0x1234_5678) };
    // SAFETY: as above.
    assert_eq!(unsafe { std::ptr::read_volatile(target as *const u32) }, 0x1234_5678);
    assert_eq!(tracker.page_state(target), Some((true, false)), "the write marked the page CPU-dirty");
    assert_eq!(perms(a), "r--", "other pages stay watched");
}

/// Page dependencies (memoised materialisations, stream copies): a result
/// stays current while its pages' bytes cannot have changed, and only then.
#[test]
fn page_deps_follow_writes_to_their_pages_only() {
    use super::deps::PageDeps;
    let mem = test_memory();
    let (view, _alias) = guest_mapping(mem, 8 * BLOCK_BYTES as usize);
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let buffers = Arc::new(TestBuffers { tracker: tracker.clone(), mem, downloads: AtomicUsize::new(0), gpu_byte: 0xAB });
    mem.attach(buffers.clone(), Arc::new(NoImages), Arc::new(NoUnmap));
    let read = |page: u64| mem.backing_ptr(page, PAGE_BYTES).map(|p| p as *const u8);
    let a = view + 2 * BLOCK_BYTES;
    let upload = || assert!(tracker.for_each_upload_range(a, 16 * PAGE_BYTES, false, Some(&read), &mut |_, _| {}, &mut || true));
    let write = |addr: u64| {
        // SAFETY: a mapped guest page (a watched page faults and is resolved).
        unsafe { std::ptr::write_volatile(addr as *mut u32, 0x5A5A_5A5A) }
    };
    upload();
    let mut deps = PageDeps::begin(mem);
    deps.note(&tracker, a + 8, 8);
    deps.note(&tracker, a + PAGE_BYTES + 4, 4);
    assert_eq!(deps.pages(), 2);
    assert!(!deps.volatile(), "watched pages");
    assert!(deps.current(&tracker, mem));
    note_sync_point();
    assert!(deps.current(&tracker, mem), "a synchronisation point changes no watched page");
    write(a + 10 * PAGE_BYTES);
    assert!(deps.current(&tracker, mem), "a CPU write to another page");
    write(a + PAGE_BYTES + 64);
    assert!(!deps.current(&tracker, mem), "a CPU write to a page read");

    // A page read while CPU-dirty may change without a fault: current
    // until the next synchronisation point only.
    let mut deps = PageDeps::begin(mem);
    deps.note(&tracker, a + PAGE_BYTES, 4);
    assert!(deps.volatile());
    assert!(deps.current(&tracker, mem));
    note_sync_point();
    assert!(!deps.current(&tracker, mem), "a synchronisation point passed");

    // GPU writes, and GPU contents landing in guest memory.
    upload();
    let mut deps = PageDeps::begin(mem);
    deps.note(&tracker, a, 4);
    tracker.mark_gpu_dirty_pages(a + 3 * PAGE_BYTES, PAGE_BYTES);
    assert!(deps.current(&tracker, mem), "the GPU writes another page");
    tracker.mark_gpu_dirty_pages(a, PAGE_BYTES);
    assert!(!deps.current(&tracker, mem), "the GPU writes a page read");
    let mut deps = PageDeps::begin(mem);
    deps.note(&tracker, a, 4);
    assert!(deps.current(&tracker, mem));
    tracker.clear_gpu_dirty_pages(a, PAGE_BYTES);
    assert!(!deps.current(&tracker, mem), "the GPU contents reached guest memory");
    let mut deps = PageDeps::begin(mem);
    deps.note(&tracker, a, 4);
    tracker.note_written(a, 4);
    assert!(!deps.current(&tracker, mem), "host code wrote the page");

    // Memory without a region counts as CPU-dirty.
    let far = view + 6 * BLOCK_BYTES;
    let mut deps = PageDeps::begin(mem);
    deps.note(&tracker, far, 4);
    assert!(deps.volatile());
    assert!(deps.current(&tracker, mem));
    note_sync_point();
    assert!(!deps.current(&tracker, mem));

    // A mapping that comes.
    let mut deps = PageDeps::begin(mem);
    deps.note(&tracker, a + 8 * PAGE_BYTES, 4);
    assert!(deps.current(&tracker, mem));
    let _ = guest_mapping(mem, PAGE_BYTES as usize);
    assert!(!deps.current(&tracker, mem), "a mapping came");
}

/// A page walk keeping the region of its block answers as the per-page
/// lookup does, across block boundaries and blocks without a region; a
/// read spanning two blocks follows writes to the pages of either.
#[test]
fn page_walks_across_blocks_match_per_page_generations() {
    use super::deps::PageDeps;
    let mem = test_memory();
    let (view, _alias) = guest_mapping(mem, 8 * BLOCK_BYTES as usize);
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let buffers = Arc::new(TestBuffers { tracker: tracker.clone(), mem, downloads: AtomicUsize::new(0), gpu_byte: 0xAB });
    mem.attach(buffers.clone(), Arc::new(NoImages), Arc::new(NoUnmap));
    let read = |page: u64| mem.backing_ptr(page, PAGE_BYTES).map(|p| p as *const u8);
    // Blocks 2 and 3 have regions (uploaded, watched), block 4 has none.
    let a = view + 3 * BLOCK_BYTES - 4 * PAGE_BYTES;
    assert!(tracker.for_each_upload_range(a, 8 * PAGE_BYTES, false, Some(&read), &mut |_, _| {}, &mut || true));
    tracker.mark_gpu_dirty_pages(a + 5 * PAGE_BYTES, PAGE_BYTES);
    let mut block = None;
    let mut page = a - 2 * PAGE_BYTES;
    while page < view + 4 * BLOCK_BYTES + 4 * PAGE_BYTES {
        assert_eq!(tracker.page_generation_in(&mut block, page), tracker.page_generation(page), "{page:#x}");
        page += PAGE_BYTES;
    }
    assert_eq!(tracker.page_generation_in(&mut block, 0), None, "invalid address");

    let mut deps = PageDeps::begin(mem);
    deps.note(&tracker, a + 3 * PAGE_BYTES + 8, 2 * PAGE_BYTES);
    assert_eq!(deps.pages(), 3);
    assert!(deps.current(&tracker, mem));
    tracker.mark_gpu_dirty_pages(a + 2 * PAGE_BYTES, PAGE_BYTES);
    assert!(deps.current(&tracker, mem), "another page of the first block");
    tracker.mark_gpu_dirty_pages(a + 5 * PAGE_BYTES, PAGE_BYTES);
    assert!(!deps.current(&tracker, mem), "a page read in the second block");
}

/// Registration answers stay exact with the lock-free summary of wholly
/// registered blocks: a page unregistered in the middle of a block is seen
/// (and its neighbours stay registered), registering it again restores
/// the whole range, and a range reaching unregistered memory is not.
#[test]
fn registration_summary_follows_register_and_unregister() {
    let mem = test_memory();
    let (view, _alias) = guest_mapping(mem, 3 * BLOCK_BYTES as usize);
    let g = &mem.pages;
    let all = 3 * BLOCK_BYTES;
    assert!(g.is_registered(view, all));
    assert!(g.is_registered(view + BLOCK_BYTES + 7, 100));
    let hole = view + BLOCK_BYTES + 5 * PAGE_BYTES;
    g.unregister(hole, PAGE_BYTES);
    assert!(!g.is_registered(view, all));
    assert!(!g.is_registered(hole + 8, 8));
    assert!(g.is_registered(view, BLOCK_BYTES), "another block");
    assert!(g.is_registered(view + BLOCK_BYTES, 5 * PAGE_BYTES), "the pages before the hole");
    assert!(g.is_registered(hole + PAGE_BYTES, BLOCK_BYTES - 6 * PAGE_BYTES), "the pages after the hole");
    assert!(!g.is_registered(view + BLOCK_BYTES, BLOCK_BYTES));
    g.register(hole, PAGE_BYTES, guard::PROT_READ | guard::PROT_WRITE);
    assert!(g.is_registered(view, all));
    assert!(g.is_registered(hole, PAGE_BYTES));
    // Past the mapping: not registered (the next block has no pages of it).
    assert!(!g.is_registered(view + all - PAGE_BYTES, 2 * PAGE_BYTES));
    // Protection changes keep the registration.
    assert!(g.set_guest_protection(view + BLOCK_BYTES, BLOCK_BYTES, guard::PROT_READ | guard::PROT_WRITE));
    assert!(g.is_registered(view, all));
}

/// Unmaps as the caches see them, with the backing each range mapped at
/// that moment.
struct UnmapLog {
    mem: &'static GuestGpuMemory,
    calls: std::sync::Mutex<Vec<(u64, u64, Option<u64>)>>,
}

impl UnmapListener for UnmapLog {
    fn unmapped(&self, addr: u64, size: u64) {
        let alias = self.mem.backing_alias().0;
        let at = self.mem.backing_ptr(addr, size).map(|p| p as u64 - alias);
        self.calls.lock().unwrap().push((addr, size, at));
    }
}

/// A mapping that replaces views of other backing bytes in place (a direct
/// mapping over a direct mapping, an AMM remap; no unmap first) is an
/// unmap of those views for the caches, told while the spans still map
/// the old bytes (downloads land there). The same bytes mapped again, and
/// memory mapped into a hole, are not.
#[test]
fn replaced_view_is_an_unmap_for_the_caches() {
    use crate::testutil::{map_guest, remap_guest, unmap_guest};
    let mem = test_memory();
    let (view, _alias) = guest_mapping(mem, 2 * BLOCK_BYTES as usize);
    let log = Arc::new(UnmapLog { mem, calls: std::sync::Mutex::new(Vec::new()) });
    mem.attach(Arc::new(MapLog(std::sync::Mutex::new(Vec::new()))), Arc::new(NoImages), log.clone());
    let calls = || std::mem::take(&mut *log.calls.lock().unwrap());
    let a = view + 4 * PAGE_BYTES;
    let n = 4 * PAGE_BYTES;
    let other = BLOCK_BYTES + 16 * PAGE_BYTES;
    let before = replaced_views();

    // Other bytes over the view: an unmap of the old view, which the spans
    // still describe when the caches hear of it.
    remap_guest(mem, a, n, other);
    assert_eq!(calls(), vec![(a, n, Some(a - view))]);
    assert!(replaced_views() > before);
    let alias = mem.backing_alias().0;
    assert_eq!(mem.backing_ptr(a, n).map(|p| p as u64 - alias), Some(other), "the new view afterwards");

    // A mapping over the moved pages and their unmoved neighbours: only
    // the parts whose bytes change.
    remap_guest(mem, a - 2 * PAGE_BYTES, 4 * PAGE_BYTES, a - 2 * PAGE_BYTES - view);
    assert_eq!(calls(), vec![(a, 2 * PAGE_BYTES, Some(other))]);

    // The same bytes mapped again (a protection change): nothing to tell.
    map_guest(mem, a - 2 * PAGE_BYTES, 2 * PAGE_BYTES);
    assert!(calls().is_empty());

    // Memory mapped into a hole: the unmap told the caches already.
    unmap_guest(mem, view + 20 * PAGE_BYTES, PAGE_BYTES);
    assert_eq!(calls(), vec![(view + 20 * PAGE_BYTES, PAGE_BYTES, Some(20 * PAGE_BYTES))]);
    remap_guest(mem, view + 20 * PAGE_BYTES, PAGE_BYTES, other + 8 * PAGE_BYTES);
    assert!(calls().is_empty());
}

/// The per-thread layout memo answers as the walk does after every change:
/// pages committed and released at random (the span map, the page registry
/// and the guest range registry all move), spans dropped while the range
/// registry stays (only the layout stamp moves), and untracked memory
/// appearing in a hole (only the range registry moves).
#[test]
fn layout_memo_answers_as_the_walk_after_mapping_changes() {
    use crate::testutil::{map_guest, unmap_guest};
    let mem = test_memory();
    let pages = 32u64;
    let (view, _) = guest_mapping(mem, (pages * PAGE_BYTES) as usize);
    let asked: Vec<(u64, u64)> = [(0, 32), (0, 8), (4, 12), (10, 3), (31, 1), (5, 20), (2, 2)].iter().map(|&(p, n)| (view + p * PAGE_BYTES, n * PAGE_BYTES)).collect();
    let check = |what: &str| {
        for &(a, n) in &asked {
            assert_eq!(mem.is_watchable(a, n), mem.is_watchable_uncached(a, n), "{what}: watchable {a:#x}+{n:#x}");
            let c = mem.committed(a, n);
            assert_eq!(mem.committed_extent(a, n), (c.runs.len() as u64, c.runs.last().map_or(0, |r| r.1)), "{what}: committed {a:#x}+{n:#x}");
        }
    };
    let hits = layout_memo::HITS.load(Ordering::Relaxed);
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for _ in 0..160 {
        check("before");
        check("again");
        let r = next();
        let p = r % pages;
        let n = (1 + (r >> 8) % 3).min(pages - p);
        let (a, len) = (view + p * PAGE_BYTES, n * PAGE_BYTES);
        match (r >> 16) % 4 {
            0 => unmap_guest(mem, a, len),
            // The spans go while the range registry still calls the pages
            // readable: untracked accessible memory until mapped again.
            1 => mem.note_unmapped(a, len),
            _ => map_guest(mem, a, len),
        }
        check("after a change");
    }
    // A hole of uncommitted memory, then untracked memory appearing in it.
    unmap_guest(mem, view + 3 * PAGE_BYTES, PAGE_BYTES);
    map_guest(mem, view, 3 * PAGE_BYTES);
    map_guest(mem, view + 4 * PAGE_BYTES, 4 * PAGE_BYTES);
    check("hole");
    assert!(mem.is_watchable(view, 8 * PAGE_BYTES), "a hole of uncommitted memory");
    ps5vm::ranges::track(view + 3 * PAGE_BYTES, PAGE_BYTES, ps5vm::MemProtect::RW);
    assert!(!mem.is_watchable(view, 8 * PAGE_BYTES), "untracked accessible memory in the hole");
    check("untracked memory");
    ps5vm::ranges::track(view + 3 * PAGE_BYTES, PAGE_BYTES, ps5vm::MemProtect::NONE);
    assert!(layout_memo::HITS.load(Ordering::Relaxed) > hits, "the memo answered");
}

/// A buffer store as the fault handler meets it while the GPU side keeps
/// working: every quick resolution marks the write (the page becomes
/// writable), but before the access retries the command thread uploads the
/// page again and re-arms its write watch. Only a settle holds the GPU side
/// still. `settles`: what [`BufferStore::settle`] can do (None: no
/// back-end; Some(false): the watch is left behind).
struct RewatchedBuffers {
    tracker: Arc<GuestPageTracker>,
    settles: Option<bool>,
    settled: AtomicUsize,
}

impl RewatchedBuffers {
    fn upload(&self, addr: u64) {
        let page = addr & !(PAGE_BYTES - 1);
        self.tracker.for_each_upload_range(page, PAGE_BYTES, false, None, &mut |_, _| {}, &mut || true);
    }
}

impl BufferStore for RewatchedBuffers {
    fn mark_cpu_write(&self, addr: u64, size: u64) -> bool {
        let handled = matches!(self.tracker.invalidate_region(addr, size, true), Some((true, false)));
        // The command thread, not held still, uses the page again.
        self.upload(addr);
        handled
    }

    fn download_to_cpu(&self, _addr: u64, _size: u64) -> bool {
        false
    }

    fn settle(&self, addr: u64, write: bool) -> Option<bool> {
        let can = self.settles?;
        self.settled.fetch_add(1, Ordering::SeqCst);
        if can && write {
            let _ = self.tracker.invalidate_region(addr, 8, true);
        }
        Some(self.tracker.pages().allows(addr, if write { FaultKind::Write } else { FaultKind::Read }))
    }
}

/// Write faults on `page` until one is declined or the page is writable:
/// (faults taken, declined).
fn fault_until_resolved(mem: &GuestGpuMemory, page: u64, limit: u32) -> (u32, bool) {
    for n in 1..=limit {
        if !mem.try_resolve_fault(FaultKind::Write, page + 8) {
            return (n, true);
        }
        if mem.pages.allows(page + 8, FaultKind::Write) {
            return (n, false);
        }
    }
    (limit, false)
}

/// A store that answers "handled" while the GPU side re-watches the page
/// before every retry kept the faulting store spinning (a million retries,
/// then a declined fault that killed the guest). Past [`SETTLE_AFTER`]
/// faults on one page the fault is settled with the GPU side held still:
/// the store resolves it, and the access goes through.
#[test]
fn fault_rewatched_before_every_retry_is_settled() {
    let mem = test_memory();
    let (view, _alias) = guest_mapping(mem, BLOCK_BYTES as usize);
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let buffers = Arc::new(RewatchedBuffers { tracker: tracker.clone(), settles: Some(true), settled: AtomicUsize::new(0) });
    mem.attach(buffers.clone(), Arc::new(NoImages), Arc::new(NoUnmap));
    let page = view + 3 * PAGE_BYTES;
    buffers.upload(page);
    assert_eq!(perms(page), "r--", "write-watched");
    let (faults, declined) = fault_until_resolved(mem, page, 4 * SETTLE_AFTER);
    assert!(!declined, "declined after {faults} faults");
    assert!(mem.pages.allows(page + 8, FaultKind::Write), "still watched after {faults} faults");
    assert_eq!(faults, SETTLE_AFTER + 1, "settled once the quick path kept failing");
    assert_eq!(buffers.settled.load(Ordering::SeqCst), 1);
    assert_eq!(perms(page), "rw-");
    assert_eq!(mem.stats().settled_faults, 1);
    // The next write fault on another page starts over on the quick path.
    let other = view + 5 * PAGE_BYTES;
    buffers.upload(other);
    assert!(mem.try_resolve_fault(FaultKind::Write, other));
    assert_eq!(buffers.settled.load(Ordering::SeqCst), 1);
}

/// A watch no settle can release, or no settle at all (no back-end): the
/// fault is declined after a bounded number of tries instead of spinning.
#[test]
fn unresolvable_fault_is_declined_after_bounded_tries() {
    for (settles, bound) in [(Some(false), SETTLE_AFTER + SETTLE_TRIES + 1), (None, TRANSIENT_RETRIES + 1)] {
        let mem = test_memory();
        let (view, _alias) = guest_mapping(mem, BLOCK_BYTES as usize);
        let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
        let buffers = Arc::new(RewatchedBuffers { tracker: tracker.clone(), settles, settled: AtomicUsize::new(0) });
        mem.attach(buffers.clone(), Arc::new(NoImages), Arc::new(NoUnmap));
        let page = view + 2 * PAGE_BYTES;
        buffers.upload(page);
        let (faults, declined) = fault_until_resolved(mem, page, 4 * TRANSIENT_RETRIES);
        assert!(declined, "{settles:?}: not declined after {faults} faults");
        assert_eq!(faults, bound, "{settles:?}");
        assert_eq!(mem.stats().faults_declined, 1);
    }
}
