// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Buffer cache upload / download on the real device with real page
//! protection. Run with `cargo test -p ps5gpu --release -- --ignored`.

use super::*;
use crate::memory::{BufferStore, ImageStore, UnmapListener, PAGE_BYTES};
use crate::testutil::{guest_mapping, map_guest, perms, test_memory, unmap_guest};
use std::sync::Mutex;

struct Shared {
    sched: Scheduler,
    cache: BufferCache,
}

struct Store {
    shared: Arc<Mutex<Shared>>,
    tracker: Arc<GuestPageTracker>,
}

impl BufferStore for Store {
    fn mark_cpu_write(&self, addr: u64, size: u64) -> bool {
        match self.tracker.invalidate_region(addr, size, true) {
            Some((tracked, false)) => tracked,
            Some((_, true)) => {
                self.download_to_cpu(addr, size) && matches!(self.tracker.invalidate_region(addr, size, true), Some((_, false)))
            }
            None => false,
        }
    }

    fn download_to_cpu(&self, addr: u64, size: u64) -> bool {
        if self.tracker.has_gpu_dirty_pages_in_fault(addr, size) != Some(true) {
            return false;
        }
        let mut g = self.shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        cache.download(sched, addr, size)
    }

    fn mapped(&self, addr: u64, size: u64) {
        if self.tracker.has_gpu_dirty_pages(addr, size) {
            self.tracker.clear_gpu_dirty_pages(addr, size);
        }
        let _ = self.tracker.invalidate_region(addr, size, false);
    }
}

struct NoImages;
impl ImageStore for NoImages {
    fn mark_cpu_write(&self, _: u64, _: u64) -> bool {
        false
    }
}
struct NoUnmap;
impl UnmapListener for NoUnmap {
    fn unmapped(&self, _: u64, _: u64) {}
}

/// Copy `size` bytes of a device buffer back to the host.
fn read_device(dev: &Arc<GpuDevice>, sched: &mut Scheduler, buf: vk::Buffer, offset: u64, size: u64) -> Vec<u8> {
    let host = Buffer::new(dev, size, vk::BufferUsageFlags::TRANSFER_DST, MemoryUse::Download).unwrap();
    sched.full_barrier();
    let cb = sched.cmd();
    // SAFETY: recording buffer; both buffers alive until the wait below.
    unsafe { dev.device.cmd_copy_buffer(cb, buf, host.buffer, &[vk::BufferCopy { src_offset: offset, dst_offset: 0, size }]) };
    sched.flush_and_wait();
    host.invalidate(0, size);
    host.bytes_mut()[..size as usize].to_vec()
}

fn pattern(i: u64) -> u8 {
    (i.wrapping_mul(2654435761) >> 11) as u8
}

#[test]
#[ignore]
fn buffer_cache_uploads_downloads_and_merges() {
    let mem = test_memory();
    let len = 16 << 20;
    let (view, _) = guest_mapping(mem, len);
    // SAFETY: freshly mapped, unwatched guest view.
    let guest = unsafe { std::slice::from_raw_parts_mut(view as *mut u8, len) };
    for (i, b) in guest.iter_mut().enumerate() {
        *b = pattern(i as u64);
    }
    let dev = GpuDevice::create(false).expect("device");
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let shared = Arc::new(Mutex::new(Shared {
        sched: Scheduler::new(&dev).unwrap(),
        cache: BufferCache::new(&dev, mem, tracker.clone()).unwrap(),
    }));
    mem.attach(Arc::new(Store { shared: shared.clone(), tracker: tracker.clone() }), Arc::new(NoImages), Arc::new(NoUnmap));

    // --- upload on use ---
    let a = view + 0x10000 + 123;
    let n = 300_000u64;
    {
        let mut g = shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        let s = cache.obtain(sched, a, n, false).unwrap();
        assert_eq!(s.offset % CACHING_PAGE, a % CACHING_PAGE);
        let got = read_device(&dev, sched, s.buffer, s.offset, n);
        assert!(got.iter().enumerate().all(|(i, &b)| b == pattern(a - view + i as u64)));
        assert!(cache.stats().uploaded_bytes >= n);
    }
    assert_eq!(perms(a), "r--", "uploaded pages are write-watched");

    // --- CPU write: only the written page is uploaded again ---
    let w = a + 5 * PAGE_BYTES + 7;
    // SAFETY: the write faults and the store marks the page CPU-dirty.
    unsafe { std::ptr::write_volatile(w as *mut u8, 0xEE) };
    assert_eq!(perms(w), "rw-");
    {
        let mut g = shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        let before = cache.stats().uploaded_bytes;
        let s = cache.obtain(sched, a, n, false).unwrap();
        assert_eq!(cache.stats().uploaded_bytes - before, PAGE_BYTES);
        let got = read_device(&dev, sched, s.buffer, s.offset + (w - a), 1);
        assert_eq!(got[0], 0xEE);
    }

    // --- GPU write of a small range: written through, the pages stay
    // readable (read-through) and hold the GPU data once it completed ---
    let gw = view + (4 << 20);
    let gn = 3 * PAGE_BYTES;
    {
        let mut g = shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        let s = cache.obtain(sched, gw, gn, true).unwrap();
        let cb = sched.cmd();
        // SAFETY: recording buffer, range inside the buffer.
        unsafe { dev.device.cmd_fill_buffer(cb, s.buffer, s.offset, gn, 0x55AA_55AA) };
        cache.flush_write_through(sched);
        sched.flush_and_wait();
        assert!(cache.has_gpu_dirty(gw, gn));
    }
    assert_eq!(perms(gw), "r--");
    // SAFETY: readable without a fault.
    let v = unsafe { std::ptr::read_volatile((gw + PAGE_BYTES) as *const u32) };
    assert_eq!(v, 0x55AA_55AA);
    assert!(tracker.has_gpu_dirty_pages(gw + PAGE_BYTES, PAGE_BYTES));
    // A CPU write still faults: the page comes back and becomes CPU-dirty.
    // SAFETY: the write faults; the store resolves the write-through first.
    unsafe { std::ptr::write_volatile((gw + PAGE_BYTES + 8) as *mut u32, 7) };
    assert!(!tracker.has_gpu_dirty_pages(gw + PAGE_BYTES, PAGE_BYTES));
    // SAFETY: CPU-dirty page, readable.
    assert_eq!(unsafe { std::ptr::read_volatile((gw + PAGE_BYTES + 4) as *const u32) }, 0x55AA_55AA);

    // --- GPU write of a large range: read-blocked; a CPU read downloads ---
    let lw = view + (10 << 20);
    let ln = 2 << 20;
    {
        let mut g = shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        let s = cache.obtain(sched, lw, ln, true).unwrap();
        let cb = sched.cmd();
        // SAFETY: recording buffer, range inside the buffer.
        unsafe { dev.device.cmd_fill_buffer(cb, s.buffer, s.offset, ln, 0x3C3C_3C3C) };
        sched.flush();
    }
    assert_eq!(perms(lw + PAGE_BYTES), "---");
    // SAFETY: the read faults; the store downloads the GPU data first.
    assert_eq!(unsafe { std::ptr::read_volatile((lw + PAGE_BYTES) as *const u32) }, 0x3C3C_3C3C);
    assert!(!tracker.has_gpu_dirty_pages(lw + PAGE_BYTES, PAGE_BYTES));
    // The GPU data reached guest memory without a staging copy: written
    // through after the fill (small written range) or downloaded straight
    // into the imported pages.
    let st = shared.lock().unwrap().cache.stats();
    assert!(st.written_through_waits + st.direct_downloads >= 1);
    // A CPU write into another GPU-dirty page downloads it, then dirties it.
    // SAFETY: as above.
    unsafe { std::ptr::write_volatile((gw + 2 * PAGE_BYTES + 4) as *mut u8, 0x01) };
    // SAFETY: readable now.
    assert_eq!(unsafe { std::ptr::read_volatile((gw + 2 * PAGE_BYTES) as *const u32) }, 0x55AA_55AA);
    assert_eq!(tracker.page_state(gw + 2 * PAGE_BYTES), Some((true, false)));

    // --- merge: a range spanning both buffers keeps their contents ---
    {
        let mut g = shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        let first = cache.stats().buffers;
        assert_eq!(first, 3, "two mirrors in the merge window, one large range");
        let s = cache.obtain(sched, view, 8 << 20, false).unwrap();
        assert_eq!(cache.stats().buffers, 2);
        assert_eq!(cache.stats().merges, 1);
        let got = read_device(&dev, sched, s.buffer, s.offset + (gw - view), 8);
        assert_eq!(&got[..4], &0x55AA_55AAu32.to_le_bytes());
        let got = read_device(&dev, sched, s.buffer, s.offset + (w - view), 1);
        assert_eq!(got[0], 0xEE);

        // --- BDA page table maps every 16 KiB page of the buffer ---
        let (table, table_addr) = cache.bda_page_table(sched).unwrap();
        assert_ne!(table_addr, 0);
        let first_page = view >> CACHING_PAGE_BITS;
        let entries = read_device(&dev, sched, table, first_page * 8, 16);
        let e0 = u64::from_le_bytes(entries[..8].try_into().unwrap());
        let e1 = u64::from_le_bytes(entries[8..].try_into().unwrap());
        assert_eq!(e0, s.device_address - s.offset);
        assert_eq!(e1, e0 + CACHING_PAGE);

        // --- unmap downloads GPU data and forgets the buffers ---
        let s = cache.obtain(sched, gw, PAGE_BYTES, true).unwrap();
        let cb = sched.cmd();
        // SAFETY: recording buffer.
        unsafe { dev.device.cmd_fill_buffer(cb, s.buffer, s.offset, PAGE_BYTES, 0x0BAD_F00D) };
        cache.unmap(sched, view, len as u64);
        assert_eq!(cache.stats().buffers, 0);
    }
    let mut b = [0u8; 4];
    assert!(mem.read_backing(gw, &mut b));
    assert_eq!(u32::from_le_bytes(b), 0x0BAD_F00D);
    assert_eq!(perms(gw), "rw-");
    dev.wait_idle();
}

/// A test cache over a fresh 4 MiB guest mapping with the fault hook
/// attached; returns (mapping, device, shared state, tracker).
fn setup(fill: u8) -> (u64, Arc<GpuDevice>, Arc<Mutex<Shared>>, Arc<GuestPageTracker>) {
    let mem = test_memory();
    let len = 4 << 20;
    let (view, _) = guest_mapping(mem, len);
    // SAFETY: freshly mapped, unwatched guest view.
    unsafe { std::ptr::write_bytes(view as *mut u8, fill, len) };
    let dev = GpuDevice::create(false).expect("device");
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let shared = Arc::new(Mutex::new(Shared {
        sched: Scheduler::new(&dev).unwrap(),
        cache: BufferCache::new(&dev, mem, tracker.clone()).unwrap(),
    }));
    mem.attach(Arc::new(Store { shared: shared.clone(), tracker: tracker.clone() }), Arc::new(NoImages), Arc::new(NoUnmap));
    (view, dev, shared, tracker)
}

#[test]
#[ignore]
fn buffer_download_through_staging_ring() {
    let (view, dev, shared, tracker) = setup(0x11);
    let gw = view + PAGE_BYTES * 3;
    {
        let mut g = shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        cache.set_zero_copy(false);
        let s = cache.obtain(sched, gw, 2 * PAGE_BYTES, true).unwrap();
        let cb = sched.cmd();
        // SAFETY: recording buffer, range inside the buffer.
        unsafe { dev.device.cmd_fill_buffer(cb, s.buffer, s.offset, 2 * PAGE_BYTES, 0x1234_5678) };
        sched.flush();
    }
    // SAFETY: the read faults; the store downloads through the ring.
    assert_eq!(unsafe { std::ptr::read_volatile((gw + PAGE_BYTES) as *const u32) }, 0x1234_5678);
    // Only the faulting page came back.
    assert!(!tracker.has_gpu_dirty_pages(gw + PAGE_BYTES, PAGE_BYTES));
    assert!(tracker.has_gpu_dirty_pages(gw, PAGE_BYTES));
    let st = shared.lock().unwrap().cache.stats();
    assert_eq!(st.direct_downloads, 0);
    assert!(st.downloads >= 1);
    // SAFETY: readable now; untouched bytes of the page kept their value.
    assert_eq!(unsafe { std::ptr::read_volatile((gw - 4) as *const u32) }, 0x1111_1111);
    dev.wait_idle();
}

#[test]
#[ignore]
fn command_processor_copy_fill_write_on_gpu() {
    let (view, dev, shared, tracker) = setup(0x22);
    let src = view + 0x10000;
    let dst = view + 0x200000 + 0x40;
    let n = 3 * PAGE_BYTES;
    {
        let mut g = shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        // GPU-written source.
        let s = cache.obtain(sched, src, n, true).unwrap();
        let cb = sched.cmd();
        // SAFETY: recording buffer, range inside the buffer.
        unsafe { dev.device.cmd_fill_buffer(cb, s.buffer, s.offset, n, 0xA5A5_A5A5) };
        sched.full_barrier();
        // Overlapping ranges are refused.
        assert!(!cache.copy(sched, src + 4, src, 64).unwrap());
        assert!(cache.copy(sched, dst, src, n).unwrap());
        assert!(cache.fill(sched, dst + n, 64, 0x0F0F_0F0F).unwrap());
        assert!(!cache.fill(sched, dst + n + 1, 64, 0).unwrap(), "unaligned fills stay on the CPU");
        assert!(cache.write(sched, dst + n + 64, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap());
        sched.flush();
        assert!(cache.has_gpu_dirty(dst, n + 72));
        let st = cache.stats();
        assert_eq!((st.gpu_copies, st.gpu_fills, st.gpu_writes), (1, 1, 1));
    }
    // The results are written through to guest memory: once the GPU is
    // done, CPU reads see them without a fault (read-through pages).
    {
        let mut g = shared.lock().unwrap();
        g.sched.flush_and_wait();
    }
    let faults = crate::memory::global().stats().faults_resolved;
    // SAFETY: readable pages.
    unsafe {
        assert_eq!(std::ptr::read_volatile((dst + n - 4) as *const u32), 0xA5A5_A5A5);
        assert_eq!(std::ptr::read_volatile((dst + n) as *const u32), 0x0F0F_0F0F);
        assert_eq!(std::ptr::read_volatile((dst + n + 64) as *const u64), 0x0807_0605_0403_0201);
        // Bytes before the copy kept their guest value.
        assert_eq!(std::ptr::read_volatile((dst - 4) as *const u32), 0x2222_2222);
    }
    // A page only the copy wrote is served by its write-through too.
    // SAFETY: readable page.
    assert_eq!(unsafe { std::ptr::read_volatile((dst + PAGE_BYTES) as *const u32) }, 0xA5A5_A5A5);
    assert_eq!(crate::memory::global().stats().faults_resolved, faults, "no read faulted");
    assert!(tracker.has_gpu_dirty_pages((dst + n) & !(PAGE_BYTES - 1), PAGE_BYTES));
    // A CPU write resolves the write-through (no copy) and dirties the page.
    // SAFETY: the write faults and is resolved.
    unsafe { std::ptr::write_volatile((dst + n + 8) as *mut u32, 0x7777_7777) };
    assert!(!tracker.has_gpu_dirty_pages((dst + n) & !(PAGE_BYTES - 1), PAGE_BYTES));
    // SAFETY: CPU-dirty page.
    unsafe {
        assert_eq!(std::ptr::read_volatile((dst + n) as *const u32), 0x0F0F_0F0F);
        assert_eq!(std::ptr::read_volatile((dst + n + 8) as *const u32), 0x7777_7777);
    }
    dev.wait_idle();
}

/// Downloads that do not hold the cache while the GPU works: the copies
/// are submitted, the caller waits for their tick, and the next call
/// finishes (staged or straight into the imported pages). GPU work recorded
/// in between that writes the pages again cancels the copies, so the CPU
/// receives the newest data.
#[test]
#[ignore]
fn buffer_cache_async_download() {
    let mem = test_memory();
    let len = 8 << 20;
    let (view, _) = guest_mapping(mem, len);
    let dev = GpuDevice::create(false).expect("device");
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut sched = Scheduler::new(&dev).unwrap();
    let mut cache = BufferCache::new(&dev, mem, tracker.clone()).unwrap();
    for (round, zero_copy) in [false, true].into_iter().enumerate() {
        cache.set_zero_copy(zero_copy);
        let gw = view + ((1 + 3 * round as u64) << 20);
        let gn = 2u64 << 20;
        let fill = |cache: &mut BufferCache, sched: &mut Scheduler, value: u32| {
            let s = cache.obtain(sched, gw, gn, true).unwrap();
            sched.full_barrier();
            // SAFETY: recording buffer, range inside the buffer.
            unsafe { dev.device.cmd_fill_buffer(sched.cmd(), s.buffer, s.offset, gn, value) };
            sched.full_barrier();
        };
        fill(&mut cache, &mut sched, 0x1111_1111);
        let DownloadStep::Wait(t) = cache.download_async(&mut sched, gw, gn) else { panic!("copies submitted") };
        // The GPU writes the range again before the caller comes back.
        fill(&mut cache, &mut sched, 0x2222_2222);
        sched.timeline.wait(t);
        let mut steps = 0;
        loop {
            match cache.download_async(&mut sched, gw, gn) {
                DownloadStep::Done(ok) => {
                    assert!(ok);
                    break;
                }
                DownloadStep::Wait(t) => sched.timeline.wait(t),
            }
            steps += 1;
            assert!(steps < 4);
        }
        assert!(steps >= 1, "the cancelled copies were recorded again");
        assert!(!tracker.has_gpu_dirty_pages(gw, gn));
        let mut now = vec![0u8; gn as usize];
        assert!(mem.read_backing(gw, &mut now));
        assert!(now.chunks(4).all(|c| c == [0x22; 4]), "zero copy {zero_copy}: the newest GPU data reached guest memory");
    }
    sched.flush_and_wait();
    dev.wait_idle();
}

/// A constant page the CPU rewrites before every read-only use, with no
/// command-processor synchronisation point in between (a submission still
/// being processed while the guest's CPU threads write ahead): every use
/// must bind the bytes current at that use. With per-interval hot-page
/// copies the uses after the fourth upload kept the first copy, and depth
/// EQUAL base-pass draws of SILENT HILL failed against the depth prepass.
#[test]
#[ignore]
fn cpu_rewrites_between_uses_reach_every_use() {
    let (view, dev, shared, _tracker) = setup(0);
    let page = view + 0x40000;
    let mut g = shared.lock().unwrap();
    let Shared { sched, cache } = &mut *g;
    cache.set_zero_copy(false);
    for round in 1..=4 * crate::memory::region::HOT_UPLOADS as u32 {
        // SAFETY: mapped guest view; a watched page faults and is resolved.
        unsafe { std::ptr::write_volatile((page + 0x100) as *mut u32, round) };
        let s = cache.obtain(sched, page, PAGE_BYTES, false).unwrap();
        let bytes = read_device(&dev, sched, s.buffer, s.offset + 0x100, 4);
        assert_eq!(u32::from_le_bytes(bytes.try_into().unwrap()), round, "use {round} bound a stale copy");
    }
    sched.flush_and_wait();
    dev.wait_idle();
}

/// Uploads ahead of the tick (the upload prologue) never reach work
/// recorded earlier in the tick: the CPU rewrites a page between two uses
/// recorded in one tick, each use copies the mirror out in order, and each
/// copy holds the bytes current at its use. A page of a buffer used earlier
/// in the tick, but outside what that use touched, is uploaded ahead.
#[test]
#[ignore]
fn uploads_ahead_of_the_tick_keep_earlier_uses() {
    let (view, dev, shared, _tracker) = setup(0);
    let page = view + 0x40000;
    let other = page + 2 * PAGE_BYTES;
    let mut g = shared.lock().unwrap();
    let Shared { sched, cache } = &mut *g;
    cache.set_zero_copy(false);
    // One buffer over both pages, made in an earlier tick.
    cache.obtain(sched, page, 4 * PAGE_BYTES, false).unwrap();
    sched.flush_and_wait();
    let out = Buffer::new(&dev, 16, vk::BufferUsageFlags::TRANSFER_DST, MemoryUse::Download).unwrap();
    let copy_out = |sched: &mut Scheduler, s: BufferSlice, slot: u64| {
        let cb = sched.cmd();
        // SAFETY: recording buffer; both buffers alive until the wait below.
        unsafe { dev.device.cmd_copy_buffer(cb, s.buffer, out.buffer, &[vk::BufferCopy { src_offset: s.offset + 0x100, dst_offset: slot * 4, size: 4 }]) };
        sched.full_barrier();
    };
    let before = cache.stats().prologue_runs;
    for (slot, (addr, value)) in [(page, 1u32), (page, 2), (other, 3), (page, 4)].into_iter().enumerate() {
        // SAFETY: mapped guest view; a watched page faults and is resolved.
        unsafe { std::ptr::write_volatile((addr + 0x100) as *mut u32, value) };
        let s = cache.obtain(sched, addr, PAGE_BYTES, false).unwrap();
        copy_out(sched, s, slot as u64);
    }
    sched.flush_and_wait();
    out.invalidate(0, 16);
    let got: Vec<u32> = out.bytes_mut()[..16].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(got, vec![1, 2, 3, 4], "a use saw bytes of another use");
    // The first use of each page in the tick went ahead.
    assert_eq!(cache.stats().prologue_runs - before, if super::prologue_enabled() { 2 } else { 0 });
    dev.wait_idle();
}

/// A buffer over a partially resident range: uncommitted pages upload as
/// zeros, the committed pages are watched, a page committed later is
/// uploaded on its own, and GPU writes never reach uncommitted pages.
#[test]
#[ignore]
fn buffer_cache_partially_resident_range() {
    let mem = test_memory();
    let len = 4 << 20;
    let (view, alias) = guest_mapping(mem, len);
    // SAFETY: freshly mapped, unwatched guest view.
    let guest = unsafe { std::slice::from_raw_parts_mut(view as *mut u8, len) };
    for (i, b) in guest.iter_mut().enumerate() {
        *b = pattern(i as u64) | 1;
    }
    let a = view + 0x20000;
    let n = 16 * PAGE_BYTES;
    let pg = |i: u64| a + i * PAGE_BYTES;
    unmap_guest(mem, pg(3), 2 * PAGE_BYTES);
    unmap_guest(mem, pg(10), PAGE_BYTES);
    let dev = GpuDevice::create(false).expect("device");
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let shared = Arc::new(Mutex::new(Shared {
        sched: Scheduler::new(&dev).unwrap(),
        cache: BufferCache::new(&dev, mem, tracker.clone()).unwrap(),
    }));
    mem.attach(Arc::new(Store { shared: shared.clone(), tracker: tracker.clone() }), Arc::new(NoImages), Arc::new(NoUnmap));
    let uncommitted = |o: u64, holes: &[u64]| holes.contains(&(o / PAGE_BYTES));
    {
        let mut g = shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        let s = cache.obtain(sched, a, n, false).unwrap();
        let got = read_device(&dev, sched, s.buffer, s.offset, n);
        for (o, &b) in got.iter().enumerate() {
            let o = o as u64;
            assert_eq!(b, if uncommitted(o, &[3, 4, 10]) { 0 } else { pattern(a - view + o) | 1 }, "byte {o:#x}");
        }
        let before = cache.stats().uploaded_bytes;
        cache.obtain(sched, a, n, false).unwrap();
        assert_eq!(cache.stats().uploaded_bytes, before, "watched: nothing uploaded again");
    }
    assert_eq!(perms(pg(0)), "r--");

    // Page 4 is committed with new bytes: only it is uploaded.
    // SAFETY: the alias maps the whole test memfd.
    unsafe { std::ptr::write_bytes((alias + (pg(4) - view)) as *mut u8, 0x5A, PAGE_BYTES as usize) };
    map_guest(mem, pg(4), PAGE_BYTES);
    {
        let mut g = shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        let before = cache.stats().uploaded_bytes;
        let s = cache.obtain(sched, a, n, false).unwrap();
        assert_eq!(cache.stats().uploaded_bytes - before, PAGE_BYTES);
        let got = read_device(&dev, sched, s.buffer, s.offset + 4 * PAGE_BYTES, PAGE_BYTES);
        assert!(got.iter().all(|&b| b == 0x5A));

        // The GPU fills the whole range; the download skips page 3 and 10.
        // SAFETY: the alias maps the whole test memfd.
        unsafe { std::ptr::write_bytes((alias + (pg(10) - view)) as *mut u8, 0x3C, PAGE_BYTES as usize) };
        let s = cache.obtain(sched, a, n, true).unwrap();
        let cb = sched.cmd();
        // SAFETY: recording buffer; the mirror is alive.
        unsafe { dev.device.cmd_fill_buffer(cb, s.buffer, s.offset, n, 0x9999_9999) };
        assert!(cache.download(sched, a, n));
    }
    let mut back = vec![0u8; n as usize];
    assert!(mem.read_committed(a, &mut back));
    for (o, &b) in back.iter().enumerate() {
        assert_eq!(b, if uncommitted(o as u64, &[3, 10]) { 0 } else { 0x99 }, "byte {o:#x}");
    }
    // SAFETY: as above.
    let hole = unsafe { std::slice::from_raw_parts((alias + (pg(10) - view)) as *const u8, PAGE_BYTES as usize) };
    assert!(hole.iter().all(|&b| b == 0x3C), "nothing stored into an uncommitted page");
    let mut g = shared.lock().unwrap();
    g.sched.flush_and_wait();
    dev.wait_idle();
}

/// Copies into guest memory note their tick on the pages (landing): once
/// it completed, a CPU access makes the page clean without the cache and
/// without new GPU work. A page the CPU accessed is written through after
/// later GPU writes of a large range (only that page), so the next access
/// finds it landed too; the rest of the range stays GPU-dirty.
#[test]
#[ignore]
fn landing_copies_serve_cpu_accesses_without_the_cache() {
    use crate::memory::tracker::Landing;
    let (view, dev, shared, tracker) = setup(0x33);
    let gw = view + (1 << 20);
    let gn = 2u64 << 20;
    let p = gw + 5 * PAGE_BYTES;
    let fill = |cache: &mut BufferCache, sched: &mut Scheduler, value: u32| {
        let s = cache.obtain(sched, gw, gn, true).unwrap();
        sched.full_barrier();
        // SAFETY: recording buffer, range inside the buffer.
        unsafe { dev.device.cmd_fill_buffer(sched.cmd(), s.buffer, s.offset, gn, value) };
        sched.full_barrier();
    };
    let t = {
        let mut g = shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        fill(cache, sched, 0x4444_4444);
        sched.flush();
        assert_eq!(tracker.landing(p), 0, "large written ranges are not written through");
        cache.note_cpu_access(p, PAGE_BYTES);
        let DownloadStep::Wait(t) = cache.download_async(sched, p, PAGE_BYTES) else { panic!("copy submitted") };
        assert_eq!(tracker.landing(p), t, "a direct download lands at its tick");
        t
    };
    // Without the cache: wait for the tick, then the page is clean.
    let timeline = shared.lock().unwrap().sched.timeline.clone();
    assert_eq!(tracker.take_landed_in_fault(p, t - 1), Some(Landing::InFlight(t)));
    timeline.wait(t);
    assert_eq!(tracker.take_landed_in_fault(p, timeline.completed_tick()), Some(Landing::Landed));
    assert_eq!(perms(p), "r--");
    // SAFETY: readable page.
    assert_eq!(unsafe { std::ptr::read_volatile((p + 64) as *const u32) }, 0x4444_4444);
    assert_eq!(tracker.take_landed_in_fault(p + PAGE_BYTES, timeline.completed_tick()), Some(Landing::Missing));

    // The GPU writes the range again: the accessed page is written through
    // after the work and stays readable meanwhile.
    let t2 = {
        let mut g = shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        fill(cache, sched, 0x5555_5555);
        assert_eq!(perms(p), "r--", "read-through page");
        assert_eq!(perms(p + PAGE_BYTES), "---");
        cache.flush_write_through(sched);
        let t2 = sched.flush();
        assert_eq!(tracker.landing(p), t2);
        assert_eq!(tracker.landing(p + PAGE_BYTES), 0);
        t2
    };
    timeline.wait(t2);
    // Once landed, the page's guest bytes are the new GPU data.
    assert_eq!(tracker.take_landed_in_fault(p, timeline.completed_tick()), Some(Landing::Landed));
    // SAFETY: readable page.
    assert_eq!(unsafe { std::ptr::read_volatile((p + 8) as *const u32) }, 0x5555_5555);
    assert!(tracker.has_gpu_dirty_pages(p + PAGE_BYTES, PAGE_BYTES));
    shared.lock().unwrap().sched.flush_and_wait();
    dev.wait_idle();
}

/// A buffer's import of its guest pages copies into the memfd bytes the
/// range mapped when the import was made. Once the range maps other bytes
/// (a view replaced in place, an AMM remap), the old bytes belong to the
/// guest elsewhere: GPU writes of the range must land in the new bytes and
/// never in the old ones (Spider-Man 2 found float data in its job
/// system's pointers). The listener here does nothing, so
/// the cache itself must notice.
#[test]
#[ignore]
fn gpu_writes_never_land_in_bytes_the_range_no_longer_maps() {
    use crate::testutil::remap_guest;
    let mem = test_memory();
    let len = 4 << 20;
    let (view, alias) = guest_mapping(mem, len);
    // SAFETY: freshly mapped, unwatched guest view.
    unsafe { std::ptr::write_bytes(view as *mut u8, 0x11, len) };
    let dev = GpuDevice::create(false).expect("device");
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let shared = Arc::new(Mutex::new(Shared { sched: Scheduler::new(&dev).unwrap(), cache: BufferCache::new(&dev, mem, tracker.clone()).unwrap() }));
    mem.attach(Arc::new(Store { shared: shared.clone(), tracker }), Arc::new(NoImages), Arc::new(NoUnmap));
    let a = view + (1 << 20);
    let n = 16 * PAGE_BYTES;
    let moved = 3u64 << 20;
    let byte = |off: u64| -> u8 {
        // SAFETY: the alias maps the whole test memfd.
        unsafe { std::ptr::read_volatile((alias + off) as *const u8) }
    };
    {
        let mut g = shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        assert!(cache.fill(sched, a, n, 0x2222_2222).unwrap());
        sched.flush_and_wait();
        assert!(cache.stats().host_imports >= 1, "the write-through went through an import");
    }
    assert_eq!(byte(a - view + 100), 0x22, "written through into the range's bytes");

    // The range moves to other bytes; the guest reuses the old ones.
    remap_guest(mem, a, n, moved);
    // SAFETY: the alias maps the whole test memfd.
    unsafe { std::ptr::write_bytes((alias + (a - view)) as *mut u8, 0x77, n as usize) };
    {
        let mut g = shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        assert!(cache.fill(sched, a, n, 0x3333_3333).unwrap());
        sched.flush_and_wait();
    }
    for off in [0, 100, n - 1] {
        assert_eq!(byte(a - view + off), 0x77, "the old bytes keep what the guest wrote there (offset {off:#x})");
        assert_eq!(byte(moved + off), 0x33, "the GPU write lands in the bytes the range maps now (offset {off:#x})");
    }
    assert!(crate::memory::stale_imports() >= 1);
    shared.lock().unwrap().sched.flush_and_wait();
    dev.wait_idle();
}

/// A synchronous download copies a GPU-dirty page into guest memory again
/// while an older copy of the same contents already landed (its span was
/// forgotten: the written-through list is capped). Until the new copy
/// lands, a CPU access must not make the page clean on the old copy: the
/// CPU's writes after that would be overwritten by the new copy of the
/// mirror (Spider-Man 2's job threads found their freshly written pointers
/// reverted to older bytes). The CPU access below follows
/// the fault path: landed, wait for an in-flight tick, or (no landing)
/// wait for the back-end, i.e. for the download to finish.
#[test]
#[ignore]
fn cpu_writes_survive_a_download_recorded_over_a_landed_page() {
    use crate::memory::tracker::Landing;
    for staged in [false, true] {
        let (view, dev, shared, tracker) = setup(0x44);
        let p = view + (1 << 20) + 3 * PAGE_BYTES;
        let timeline = shared.lock().unwrap().sched.timeline.clone();
        let mut g = shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        cache.set_zero_copy(!staged);
        // The GPU writes the page; a copy of it into guest memory lands
        // (its write-through through the import, or any earlier copy).
        assert!(cache.fill(sched, p, PAGE_BYTES, 0x5555_5555).unwrap());
        sched.flush_and_wait();
        assert!(tracker.has_gpu_dirty_pages(p, PAGE_BYTES));
        if staged {
            tracker.set_landing(p, PAGE_BYTES, timeline.completed_tick());
        }
        assert_eq!(tracker.take_landed_in_fault(p, 0), Some(Landing::InFlight(tracker.landing(p))), "an older copy landed");
        // The span is forgotten (as when the list overflows); a download of
        // the page records a new copy of the unchanged mirror.
        cache.written_through.clear();
        // Not submitted yet: the CPU access below comes before the copy.
        let mut rec = Some(cache.record_download_runs(sched, vec![(p, PAGE_BYTES)]).expect("recorded"));
        // A CPU thread writes the page meanwhile, through the fault path.
        loop {
            match tracker.take_landed_in_fault(p, timeline.completed_tick()) {
                Some(Landing::Landed | Landing::Clean) => break,
                Some(Landing::InFlight(t)) if t < sched.current_tick() => timeline.wait(t),
                // A copy still being recorded, or none: the access needs
                // the back-end, which the download holds until it is done.
                _ => assert!(cache.finish_download_runs(sched, rec.take().expect("one download"))),
            }
        }
        // SAFETY: a guest page; a write fault is resolved by the store.
        unsafe { std::ptr::write_volatile((p + 64) as *mut u32, 0xC0DE_C0DE) };
        if let Some(rec) = rec {
            assert!(cache.finish_download_runs(sched, rec));
        }
        sched.flush_and_wait();
        let mut w = [0u8; 4];
        assert!(cache.memory.read_backing(p + 64, &mut w));
        assert_eq!(u32::from_le_bytes(w), 0xC0DE_C0DE, "the CPU write survives the download (staged {staged})");
        assert!(cache.memory.read_backing(p + 128, &mut w));
        assert_eq!(u32::from_le_bytes(w), 0x5555_5555, "the GPU data is in guest memory (staged {staged})");
        drop(g);
        dev.wait_idle();
    }
}

#[test]
fn descriptor_window_sizes() {
    assert_eq!(parse_descriptor_window(None), 1 << 20);
    assert_eq!(parse_descriptor_window(Some("0")), 0);
    assert_eq!(parse_descriptor_window(Some("64")), 64 << 10);
    assert_eq!(parse_descriptor_window(Some("100")), 128 << 10, "rounded up to a power of two");
    assert_eq!(parse_descriptor_window(Some("junk")), 1 << 20);
}

/// A descriptor word the GPU wrote, read on the CPU, downloads the
/// GPU-written pages of its aligned window at once: the other words of its
/// table cost no further GPU wait (`RPS5E_DESC_WINDOW_KIB`).
#[test]
#[ignore]
fn descriptor_reads_download_their_window_once() {
    let (view, dev, shared, tracker) = setup(0x21);
    let window = 1u64 << 20;
    let base = (view + window - 1) & !(window - 1);
    assert!(base + 2 * window <= view + (4 << 20));
    let (a, b, far) = (base + 3 * PAGE_BYTES, base + 200 * PAGE_BYTES, base + window + PAGE_BYTES);
    let mut g = shared.lock().unwrap();
    let Shared { sched, cache } = &mut *g;
    let s = cache.obtain(sched, base, 2 * window, true).unwrap();
    sched.full_barrier();
    // SAFETY: recording buffer, range inside the buffer.
    unsafe { dev.device.cmd_fill_buffer(sched.cmd(), s.buffer, s.offset, 2 * window, 0x0BAD_F00D) };
    sched.full_barrier();
    let submits = sched.stats().submits;
    assert!(cache.download_window(sched, a));
    assert_eq!(sched.stats().submits - submits, 1, "one GPU wait");
    assert!(!tracker.has_gpu_dirty_pages(a, 4) && !tracker.has_gpu_dirty_pages(b, 4), "the word and its window are in guest memory");
    assert!(tracker.has_gpu_dirty_pages(far, 4), "pages past the window stay on the GPU");
    let mut w = [0u8; 4];
    assert!(cache.memory.read_backing(b, &mut w));
    assert_eq!(u32::from_le_bytes(w), 0x0BAD_F00D);
    // Another word of the window: nothing to wait for.
    let submits = sched.stats().submits;
    assert!(cache.download_window(sched, b));
    assert_eq!(sched.stats().submits, submits);
    sched.flush_and_wait();
    drop(g);
    dev.wait_idle();
}

/// A large GPU-written range the CPU read back (a fault downloaded it) is
/// written through after the next GPU write of it, whatever its size: the
/// next CPU access finds every page landed instead of waiting for the GPU
/// again (`RPS5E_READBACK_ANY_SIZE`).
#[test]
#[ignore]
fn large_read_back_ranges_are_written_through_next_time() {
    use crate::memory::tracker::Landing;
    let (view, dev, shared, tracker) = setup(0x31);
    let gw = view + (1 << 20);
    let gn = 2u64 << 20;
    let fill = |cache: &mut BufferCache, sched: &mut Scheduler, value: u32| {
        let s = cache.obtain(sched, gw, gn, true).unwrap();
        sched.full_barrier();
        // SAFETY: recording buffer, range inside the buffer.
        unsafe { dev.device.cmd_fill_buffer(sched.cmd(), s.buffer, s.offset, gn, value) };
        sched.full_barrier();
    };
    let timeline = shared.lock().unwrap().sched.timeline.clone();
    {
        let mut g = shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        fill(cache, sched, 0x6666_6666);
        // The CPU copies the whole range out (a fault downloads it).
        loop {
            match cache.download_async(sched, gw, gn) {
                DownloadStep::Done(ok) => break assert!(ok),
                DownloadStep::Wait(t) => timeline.wait(t),
            }
        }
        assert!(!tracker.has_gpu_dirty_pages(gw, gn));
    }
    let t = {
        let mut g = shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        fill(cache, sched, 0x7777_7777);
        cache.flush_write_through(sched);
        sched.flush()
    };
    timeline.wait(t);
    let middle = gw + gn / 2 + 5 * PAGE_BYTES;
    for p in [gw, middle, gw + gn - PAGE_BYTES] {
        assert_eq!(tracker.take_landed_in_fault(p, timeline.completed_tick()), Some(Landing::Landed), "page {:#x}", p - gw);
        // SAFETY: readable page.
        assert_eq!(unsafe { std::ptr::read_volatile((p + 16) as *const u32) }, 0x7777_7777);
    }
    shared.lock().unwrap().sched.flush_and_wait();
    dev.wait_idle();
}

/// Staged downloads (pages without an import) reuse their host-visible
/// buffers once the copies landed (`RPS5E_DOWNLOAD_POOL`).
#[test]
#[ignore]
fn staged_downloads_reuse_their_buffers() {
    let (view, dev, shared, tracker) = setup(0x41);
    let gw = view + (1 << 20);
    let gn = 256 << 10;
    let timeline = shared.lock().unwrap().sched.timeline.clone();
    for (round, value) in [0x1357_9BDFu32, 0x2468_ACE0].into_iter().enumerate() {
        let mut g = shared.lock().unwrap();
        let Shared { sched, cache } = &mut *g;
        cache.set_zero_copy(false);
        let s = cache.obtain(sched, gw, gn, true).unwrap();
        sched.full_barrier();
        // SAFETY: recording buffer, range inside the buffer.
        unsafe { dev.device.cmd_fill_buffer(sched.cmd(), s.buffer, s.offset, gn, value) };
        sched.full_barrier();
        loop {
            match cache.download_async(sched, gw, gn) {
                DownloadStep::Done(ok) => break assert!(ok),
                DownloadStep::Wait(t) => timeline.wait(t),
            }
        }
        sched.run_completed();
        assert!(!tracker.has_gpu_dirty_pages(gw, gn));
        let mut w = [0u8; 4];
        assert!(cache.memory.read_backing(gw + gn - 4, &mut w));
        assert_eq!(u32::from_le_bytes(w), value);
        assert_eq!(cache.download_pool.free_count(), 1, "round {round}: the staging buffer went back to the pool");
    }
    shared.lock().unwrap().sched.flush_and_wait();
    dev.wait_idle();
}

/// The touched spans (a sorted vector) behave as the ordered-map model
/// they replaced: merged on contact, queried by overlap, reset per tick.
#[test]
fn touched_spans_match_the_ordered_map_model() {
    use std::collections::BTreeMap;
    struct Model {
        tick: u64,
        spans: BTreeMap<u64, u64>,
    }
    impl Model {
        fn add(&mut self, tick: u64, addr: u64, size: u64) {
            if self.tick != tick {
                self.tick = tick;
                self.spans.clear();
            }
            let page = crate::memory::PAGE_BYTES;
            let (mut start, mut end) = (addr & !(page - 1), (addr + size.max(1)).div_ceil(page) * page);
            if let Some((_, &e)) = self.spans.range(..=start).next_back() {
                if end <= e {
                    return;
                }
            }
            while let Some((&s, &e)) = self.spans.range(..=end).next_back() {
                if e < start {
                    break;
                }
                self.spans.remove(&s);
                start = start.min(s);
                end = end.max(e);
            }
            self.spans.insert(start, end);
        }
        fn overlaps(&self, tick: u64, addr: u64, size: u64) -> bool {
            self.tick == tick && self.spans.range(..addr + size).next_back().is_some_and(|(_, &e)| e > addr)
        }
    }
    let page = crate::memory::PAGE_BYTES;
    let mut t = Touched::default();
    let mut m = Model { tick: 0, spans: BTreeMap::new() };
    // A deterministic pseudo-random walk over 64 pages, the tick changing
    // now and then.
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut tick = 1;
    for step in 0..20_000 {
        if next() % 500 == 0 {
            tick += 1;
        }
        let addr = 0x1000_0000 + next() % (64 * page);
        let size = match next() % 4 {
            0 => 1 + next() % 256,
            1 => next() % page,
            2 => next() % (4 * page),
            _ => next() % (12 * page),
        };
        if next() % 3 == 0 {
            t.add(tick, addr, size);
            m.add(tick, addr, size);
            let spans: Vec<(u64, u64)> = m.spans.iter().map(|(&s, &e)| (s, e)).collect();
            assert_eq!(t.spans, spans, "step {step}");
        } else {
            let size = size.max(1);
            let qtick = if next() % 10 == 0 { tick + 1 } else { tick };
            assert_eq!(t.overlaps(qtick, addr, size), m.overlaps(qtick, addr, size), "step {step} {addr:#x}+{size:#x}");
        }
    }
}

/// The page memos answer as a plain map through their front: entries
/// replaced, pages colliding in one front slot, and clears.
#[test]
fn page_memo_front_matches_the_map() {
    let mut memos = PageMemos::default();
    let mut model: std::collections::HashMap<u64, (u64, u64)> = std::collections::HashMap::new();
    let mut x: u64 = 0x2545_f491_4f6c_dd1d;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for step in 0..50_000u64 {
        // Pages that collide in the front (multiples of its size apart).
        let page = (next() % 8) * PAGE_MEMO_FRONT as u64 + next() % 32;
        match next() % 10 {
            0..=3 => {
                let m = PageMemo { key: step, serial: next(), dirty_epoch: 0, sync: None };
                memos.insert(page, m);
                model.insert(page, (m.key, m.serial));
            }
            4 if next() % 50 == 0 => {
                memos.clear();
                model.clear();
            }
            _ => {
                let got = memos.get(page).map(|m| (m.key, m.serial));
                assert_eq!(got, model.get(&page).copied(), "step {step} page {page}");
            }
        }
        assert_eq!(memos.len(), model.len());
    }
}
