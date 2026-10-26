// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Windows implementation of the host primitives.
//!
//! Guest memory needs what Linux gives with `mmap(MAP_FIXED)`: replace any
//! part of a mapping with another mapping, at page granularity, and map one
//! physical backing at several addresses. Windows offers that through
//! placeholders (`VirtualAlloc2(MEM_RESERVE_PLACEHOLDER)`,
//! `MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)`,
//! `UnmapViewOfFile2(MEM_PRESERVE_PLACEHOLDER)`): a placeholder is reserved
//! address space that can be split at any page and replaced by exactly one
//! view. [`space`] keeps the model of every range this process manages
//! that way. All guest-visible memory is a view of a pagefile-backed
//! section, private memory included (each commit gets its own section), so
//! a partial unmap can always re-map the remaining pieces with their
//! contents.

use crate::MemProtect;
use std::sync::Arc;
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::*;
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const MEM_COALESCE_PLACEHOLDERS: u32 = 0x1;
const MEM_PRESERVE_PLACEHOLDER_FREE: u32 = 0x2;
/// Windows reserves address space in units of 64 KiB.
pub const ALLOCATION_GRANULARITY: u64 = 0x1_0000;

/// `PAGE_*` value of a protection.
pub fn native_prot(p: MemProtect) -> u32 {
    match (p.read || p.write, p.write, p.exec) {
        (_, true, true) => PAGE_EXECUTE_READWRITE,
        (_, true, false) => PAGE_READWRITE,
        (true, false, true) => PAGE_EXECUTE_READ,
        (false, false, true) => PAGE_EXECUTE,
        (true, false, false) => PAGE_READONLY,
        (false, false, false) => PAGE_NOACCESS,
    }
}

/// Protection of a `PAGE_*` value (guard and cache bits ignored).
pub fn prot_of_native(n: u32) -> MemProtect {
    match n & 0xFF {
        PAGE_READONLY => MemProtect::R,
        PAGE_READWRITE | PAGE_WRITECOPY => MemProtect::RW,
        PAGE_EXECUTE => MemProtect { read: false, write: false, exec: true },
        PAGE_EXECUTE_READ => MemProtect::RX,
        PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY => MemProtect::RWX,
        _ => MemProtect::NONE,
    }
}

/// Host page size.
pub fn page_size() -> u64 {
    0x1000
}

/// Leave the process at once, without the atexit chain or DLL detach.
pub fn exit_now(code: i32) -> ! {
    // SAFETY: immediate process exit.
    unsafe { windows_sys::Win32::System::Threading::TerminateProcess(GetCurrentProcess(), code as u32) };
    loop {
        std::hint::spin_loop();
    }
}

/// Kernel id of the calling thread.
pub fn current_tid() -> u64 {
    // SAFETY: plain query.
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() as u64 }
}

fn qpc_ns() -> u64 {
    use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
    static FREQ: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let f = *FREQ.get_or_init(|| {
        let mut f = 0i64;
        // SAFETY: valid out pointer.
        unsafe { QueryPerformanceFrequency(&mut f) };
        f.max(1) as u64
    });
    let mut c = 0i64;
    // SAFETY: valid out pointer.
    unsafe { QueryPerformanceCounter(&mut c) };
    ((c as u128 * 1_000_000_000) / f as u128) as u64
}

/// Monotonic clock in nanoseconds.
pub fn monotonic_ns() -> u64 {
    qpc_ns()
}

/// CPU time consumed by the calling thread, in nanoseconds.
pub fn thread_cpu_ns() -> u64 {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    let z = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut a, mut b, mut k, mut u) = (z, z, z, z);
    // SAFETY: valid out pointers.
    unsafe { GetThreadTimes(GetCurrentThread(), &mut a, &mut b, &mut k, &mut u) };
    let t = |f: FILETIME| (f.dwHighDateTime as u64) << 32 | f.dwLowDateTime as u64;
    (t(k) + t(u)) * 100
}

/// Map anonymous zeroed memory at exactly `addr` without replacing
/// anything mapped there.
pub fn map_anon_noreplace(addr: u64, len: u64, prot: MemProtect) -> Result<(), String> {
    if !space::is_free(addr, len) {
        return Err(format!("{addr:#x}+{len:#x} is in use"));
    }
    if space::commit(addr, len, prot) {
        Ok(())
    } else {
        Err(format!("cannot map {addr:#x}+{len:#x}: {}", std::io::Error::last_os_error()))
    }
}

/// Map anonymous zeroed memory at exactly `addr`, replacing what is there.
pub fn map_anon_fixed(addr: u64, len: u64, prot: MemProtect) -> Result<(), String> {
    if space::commit(addr, len, prot) {
        Ok(())
    } else {
        Err(format!("cannot map {addr:#x}+{len:#x}: {}", std::io::Error::last_os_error()))
    }
}

/// Map anonymous zeroed memory anywhere.
pub fn map_anon(len: u64, prot: MemProtect) -> Option<u64> {
    // SAFETY: a fresh allocation.
    let p = unsafe { VirtualAlloc(std::ptr::null(), len as usize, MEM_RESERVE | MEM_COMMIT, native_prot(prot)) };
    (!p.is_null()).then_some(p as u64)
}

/// Unmap `[addr, addr+len)`. Ranges of the placeholder space stay reserved
/// (inaccessible); other allocations are released whole.
pub fn unmap(addr: u64, len: u64) {
    if !space::release(addr, len) {
        // SAFETY: an allocation made by `map_anon`.
        unsafe { VirtualFree(addr as *mut _, 0, MEM_RELEASE) };
    }
}

/// Change the protection of `[addr, addr+len)`, which may span several
/// allocations and views. Reserved pages are left alone.
pub fn protect(addr: u64, len: u64, prot: MemProtect) -> bool {
    protect_native(addr, len, native_prot(prot))
}

/// [`protect`] with a `PAGE_*` value.
pub fn protect_native(addr: u64, len: u64, native: u32) -> bool {
    let end = addr + len;
    let mut cur = addr;
    let mut ok = true;
    while cur < end {
        let Some(r) = query(cur) else { return false };
        let stop = r.end.min(end);
        if r.state == MEM_COMMIT {
            let mut old = 0;
            // SAFETY: committed pages of one region.
            if unsafe { VirtualProtect(cur as *const _, (stop - cur) as usize, native, &mut old) } == 0 {
                ok = false;
            }
        } else if native != PAGE_NOACCESS {
            ok = false;
        }
        cur = stop;
    }
    ok
}

/// One region as `VirtualQuery` reports it.
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub start: u64,
    pub end: u64,
    pub alloc_base: u64,
    pub state: u32,
    pub protect: u32,
    pub kind: u32,
}

/// The region holding `addr`.
pub fn query(addr: u64) -> Option<Region> {
    // SAFETY: plain query into a local.
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let n = unsafe { VirtualQuery(addr as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) };
    if n == 0 {
        return None;
    }
    let start = mbi.BaseAddress as u64;
    Some(Region {
        start,
        end: start + mbi.RegionSize as u64,
        alloc_base: mbi.AllocationBase as u64,
        state: mbi.State,
        protect: mbi.Protect,
        kind: mbi.Type,
    })
}

/// End of the host allocations overlapping `[addr, addr+len)`, if any
/// (what a candidate range has to skip).
pub fn conflict_end(addr: u64, len: u64) -> Option<u64> {
    let end = addr + len;
    let mut cur = addr;
    let mut best = None;
    while cur < end {
        let r = query(cur)?;
        if r.state != MEM_FREE && !space::is_spare(r.start.max(addr), r.end.min(end)) {
            best = Some(r.end);
        }
        cur = r.end;
    }
    best
}

/// A pagefile-backed section (shared memory object).
pub struct Section {
    handle: HANDLE,
    size: u64,
}

// SAFETY: a kernel handle, usable from any thread.
unsafe impl Send for Section {}
unsafe impl Sync for Section {}

impl Drop for Section {
    fn drop(&mut self) {
        // SAFETY: our own handle.
        unsafe { CloseHandle(self.handle) };
    }
}

impl Section {
    /// A zero-filled section of `size` bytes. Commit is charged for the
    /// whole size up front; pages take memory once touched.
    pub fn create(size: u64) -> Option<Arc<Section>> {
        // SAFETY: anonymous section, no name.
        let h = unsafe {
            CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_EXECUTE_READWRITE | SEC_COMMIT, (size >> 32) as u32, size as u32, std::ptr::null())
        };
        if h.is_null() {
            return None;
        }
        Some(Arc::new(Section { handle: h, size }))
    }

    /// A section over the open file `file` (a raw `HANDLE`), `size` bytes.
    /// Executable views need the file opened with execute access.
    pub fn from_file(file: HANDLE, size: u64) -> Option<Arc<Section>> {
        for prot in [PAGE_EXECUTE_READWRITE, PAGE_READWRITE] {
            // SAFETY: a mapping object over a file handle we were given.
            let h = unsafe { CreateFileMappingW(file, std::ptr::null(), prot, (size >> 32) as u32, size as u32, std::ptr::null()) };
            if !h.is_null() {
                return Some(Arc::new(Section { handle: h, size }));
            }
        }
        None
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn handle(&self) -> HANDLE {
        self.handle
    }

    /// Map `[offset, offset+len)` read/write at an address the system picks.
    pub fn map_anywhere(&self, offset: u64, len: u64) -> Option<u64> {
        // SAFETY: a view of our own section.
        let v = unsafe { MapViewOfFile3(self.handle, GetCurrentProcess(), std::ptr::null(), offset, len as usize, 0, PAGE_READWRITE, std::ptr::null_mut(), 0) };
        (!v.Value.is_null()).then_some(v.Value as u64)
    }
}

/// The placeholder-managed part of the address space.
pub mod space {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    #[derive(Clone)]
    enum Kind {
        /// A placeholder: reserved, inaccessible.
        Hold,
        /// A view of `section` at `offset`.
        View { section: Arc<Section>, offset: u64 },
    }

    #[derive(Clone)]
    struct Entry {
        end: u64,
        kind: Kind,
        /// False for placeholder space reserved only because Windows
        /// reserves in 64 KiB units, or given back by `release`: a later
        /// "map without replacing" may take it.
        owned: bool,
    }

    static MAP: Mutex<BTreeMap<u64, Entry>> = Mutex::new(BTreeMap::new());

    /// Ranges being re-mapped right now. Linux replaces part of a mapping
    /// atomically; here a view is unmapped and its pieces mapped again, so
    /// another thread can fault in between. The exception handler waits
    /// for the range to settle and retries the access (`wait_transient`).
    const SLOTS: usize = 16;
    static TRANSIENT: [(AtomicU64, AtomicU64); SLOTS] = [const { (AtomicU64::new(0), AtomicU64::new(0)) }; SLOTS];

    /// Marks `[start, end)` transient until dropped.
    struct Transient(Option<usize>);

    impl Transient {
        fn new(map: &BTreeMap<u64, Entry>, addr: u64, end: u64) -> Self {
            let mut lo = addr;
            let mut hi = end;
            for (&s, e) in map.range(..end) {
                if e.end > addr && matches!(e.kind, Kind::View { .. }) {
                    lo = lo.min(s);
                    hi = hi.max(e.end);
                }
            }
            // The map lock is held: slots are only taken under it.
            for (i, (a, b)) in TRANSIENT.iter().enumerate() {
                if b.load(Ordering::Acquire) == 0 {
                    a.store(lo, Ordering::Release);
                    b.store(hi, Ordering::Release);
                    return Transient(Some(i));
                }
            }
            Transient(None)
        }
    }

    impl Drop for Transient {
        fn drop(&mut self) {
            if let Some(i) = self.0 {
                TRANSIENT[i].1.store(0, Ordering::Release);
            }
        }
    }

    /// If `addr` lies in a range being re-mapped, wait until it settles and
    /// return true (the faulting access should be retried). Lock-free:
    /// usable from an exception handler.
    pub fn wait_transient(addr: u64) -> bool {
        let mut waited = false;
        for (a, b) in TRANSIENT.iter() {
            loop {
                let end = b.load(Ordering::Acquire);
                if end == 0 || addr < a.load(Ordering::Acquire) || addr >= end {
                    break;
                }
                waited = true;
                std::thread::yield_now();
            }
        }
        waited
    }

    fn lock() -> std::sync::MutexGuard<'static, BTreeMap<u64, Entry>> {
        MAP.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn align_down(v: u64, a: u64) -> u64 {
        v & !(a - 1)
    }

    fn align_up(v: u64, a: u64) -> u64 {
        (v + a - 1) & !(a - 1)
    }

    /// The entry containing `addr`.
    fn containing(map: &BTreeMap<u64, Entry>, addr: u64) -> Option<(u64, Entry)> {
        let (&s, e) = map.range(..=addr).next_back()?;
        (addr < e.end).then(|| (s, e.clone()))
    }

    /// Reserve placeholders over the free host space in `[addr, end)` that
    /// the model does not cover yet.
    fn fill(map: &mut BTreeMap<u64, Entry>, addr: u64, end: u64) -> bool {
        let mut cur = addr;
        while cur < end {
            if let Some((_, e)) = containing(map, cur) {
                cur = e.end;
                continue;
            }
            let next_entry = map.range(cur..).next().map(|(&s, _)| s).unwrap_or(u64::MAX);
            let Some(r) = query(cur) else { return false };
            if r.state != MEM_FREE {
                log::error!("[vm] {cur:#x} belongs to a host allocation ({:#x}..{:#x})", r.start, r.end);
                return false;
            }
            // `VirtualQuery` reports a free region from the queried page
            // on; a reservation starts at a 64 KiB boundary, so the space
            // down to it must be free as well.
            let lo = align_down(cur, ALLOCATION_GRANULARITY);
            let rs = if lo == cur {
                cur
            } else {
                match query(lo) {
                    Some(q) if q.state == MEM_FREE && q.end >= cur => lo,
                    _ => {
                        log::error!("[vm] cannot reserve at {cur:#x}: the 64 KiB block is partly in use");
                        return false;
                    }
                }
            };
            let re = r.end.min(align_up(end, ALLOCATION_GRANULARITY)).min(next_entry.max(cur + 1));
            if re <= rs {
                log::error!("[vm] cannot reserve {rs:#x}..{re:#x}: not at a 64 KiB boundary");
                return false;
            }
            // SAFETY: free address space.
            let p = unsafe {
                VirtualAlloc2(GetCurrentProcess(), rs as *const _, (re - rs) as usize, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, std::ptr::null_mut(), 0)
            };
            if p.is_null() {
                log::error!("[vm] placeholder {rs:#x}..{re:#x}: {}", std::io::Error::last_os_error());
                return false;
            }
            map.insert(rs, Entry { end: re, kind: Kind::Hold, owned: false });
            cur = re;
        }
        true
    }

    /// Split the placeholder entry starting at `s` at `at`.
    fn split(map: &mut BTreeMap<u64, Entry>, s: u64, at: u64) -> bool {
        let e = map[&s].clone();
        debug_assert!(matches!(e.kind, Kind::Hold) && s < at && at < e.end);
        // SAFETY: a placeholder of the model; releasing a leading part
        // with MEM_PRESERVE_PLACEHOLDER splits it.
        if unsafe { VirtualFree(s as *mut _, (at - s) as usize, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER_FREE) } == 0 {
            log::error!("[vm] split {s:#x}..{:#x} at {at:#x}: {}", e.end, std::io::Error::last_os_error());
            return false;
        }
        map.insert(s, Entry { end: at, ..e.clone() });
        map.insert(at, e);
        true
    }

    /// Replace the placeholder `[s, e)` with a view.
    fn map_piece(map: &mut BTreeMap<u64, Entry>, s: u64, e: u64, section: &Arc<Section>, offset: u64, native: u32) -> bool {
        let initial = if native == PAGE_NOACCESS { PAGE_READWRITE } else { native };
        // SAFETY: an exact placeholder of the model.
        let v = unsafe {
            MapViewOfFile3(section.handle, GetCurrentProcess(), s as *const _, offset, (e - s) as usize, MEM_REPLACE_PLACEHOLDER, initial, std::ptr::null_mut(), 0)
        };
        if v.Value.is_null() {
            log::error!("[vm] view {s:#x}..{e:#x} offset {offset:#x}: {}", std::io::Error::last_os_error());
            return false;
        }
        if initial != native {
            let mut old = 0;
            // SAFETY: the view just mapped.
            unsafe { VirtualProtect(s as *const _, (e - s) as usize, native, &mut old) };
        }
        map.insert(s, Entry { end: e, kind: Kind::View { section: section.clone(), offset }, owned: true });
        true
    }

    /// Make `[addr, end)` consist of placeholder entries only, split
    /// exactly at `addr` and `end`. Views reaching outside the range keep
    /// their outside parts (re-mapped with their protections).
    fn carve(map: &mut BTreeMap<u64, Entry>, addr: u64, end: u64) -> bool {
        if !fill(map, addr, end) {
            return false;
        }
        // Views overlapping the range become placeholders.
        let views: Vec<(u64, Entry)> = map
            .range(..end)
            .filter(|(_, e)| e.end > addr && matches!(e.kind, Kind::View { .. }))
            .map(|(&s, e)| (s, e.clone()))
            .collect();
        let mut remap: Vec<(u64, u64, Arc<Section>, u64, Vec<Region>)> = Vec::new();
        for (s, e) in views {
            let Kind::View { section, offset } = &e.kind else { continue };
            let mut keep = |ps: u64, pe: u64| {
                if ps < pe {
                    let mut prots = Vec::new();
                    let mut c = ps;
                    while c < pe {
                        let Some(r) = query(c) else { break };
                        prots.push(Region { start: c, end: r.end.min(pe), ..r });
                        c = r.end;
                    }
                    remap.push((ps, pe, section.clone(), offset + (ps - s), prots));
                }
            };
            keep(s, addr.max(s));
            keep(end.min(e.end), e.end);
            // SAFETY: a view of the model, mapped over a placeholder.
            let ok = unsafe { UnmapViewOfFile2(GetCurrentProcess(), MEMORY_MAPPED_VIEW_ADDRESS { Value: s as *mut _ }, MEM_PRESERVE_PLACEHOLDER) };
            if ok == 0 {
                log::error!("[vm] unmap view {s:#x}..{:#x}: {}", e.end, std::io::Error::last_os_error());
                return false;
            }
            map.insert(s, Entry { end: e.end, kind: Kind::Hold, owned: e.owned });
        }
        // Split at the boundaries.
        if let Some((s, _)) = containing(map, addr) {
            if s < addr && !split(map, s, addr) {
                return false;
            }
        }
        if let Some((s, _)) = containing(map, end) {
            if s < end && !split(map, s, end) {
                return false;
            }
        }
        // Put the outside parts of the views back.
        for (ps, pe, section, offset, prots) in remap {
            if !map_piece(map, ps, pe, &section, offset, PAGE_READWRITE) {
                return false;
            }
            for r in prots {
                let mut old = 0;
                // SAFETY: part of the view just mapped.
                unsafe { VirtualProtect(r.start as *const _, (r.end - r.start) as usize, r.protect, &mut old) };
            }
        }
        true
    }

    /// Merge the placeholders of `[addr, end)` into one where Windows
    /// allows it (pieces of one reservation).
    fn coalesce(map: &mut BTreeMap<u64, Entry>, addr: u64, end: u64) {
        let n = map.range(addr..end).count();
        if n < 2 {
            return;
        }
        // SAFETY: placeholders of the model covering exactly the range.
        if unsafe { VirtualFree(addr as *mut _, (end - addr) as usize, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) } != 0 {
            let keys: Vec<u64> = map.range(addr..end).map(|(&s, _)| s).collect();
            for k in keys {
                map.remove(&k);
            }
            map.insert(addr, Entry { end, kind: Kind::Hold, owned: true });
        }
    }

    fn set_owned(map: &mut BTreeMap<u64, Entry>, addr: u64, end: u64, owned: bool) {
        for (_, e) in map.range_mut(addr..end) {
            e.owned = owned;
        }
    }

    /// Hold `[addr, addr+len)` as an inaccessible reservation, replacing
    /// whatever the range held.
    pub fn hold(addr: u64, len: u64) -> bool {
        let mut map = lock();
        let end = addr + len;
        let _transient = Transient::new(&map, addr, end);
        if !carve(&mut map, addr, end) {
            return false;
        }
        coalesce(&mut map, addr, end);
        set_owned(&mut map, addr, end, true);
        true
    }

    /// Whether `[addr, addr+len)` is free: unreserved host space or spare
    /// placeholder space.
    pub fn is_free(addr: u64, len: u64) -> bool {
        let map = lock();
        let end = addr + len;
        let mut cur = addr;
        while cur < end {
            if let Some((_, e)) = containing(&map, cur) {
                if e.owned {
                    return false;
                }
                cur = e.end;
                continue;
            }
            let Some(r) = query(cur) else { return false };
            if r.state != MEM_FREE {
                return false;
            }
            let next_entry = map.range(cur..).next().map(|(&s, _)| s).unwrap_or(u64::MAX);
            cur = r.end.min(next_entry);
        }
        true
    }

    /// Whether `[addr, end)` is entirely spare placeholder space.
    pub fn is_spare(addr: u64, end: u64) -> bool {
        let map = lock();
        let mut cur = addr;
        while cur < end {
            match containing(&map, cur) {
                Some((_, e)) if !e.owned => cur = e.end,
                _ => return false,
            }
        }
        true
    }

    /// [`hold`] only when the range is free.
    pub fn hold_noreplace(addr: u64, len: u64) -> bool {
        is_free(addr, len) && hold(addr, len)
    }

    /// Map `[offset, offset+len)` of `section` at `addr` with `prot`,
    /// replacing whatever the range held.
    pub fn map_view(addr: u64, len: u64, section: &Arc<Section>, offset: u64, prot: MemProtect) -> bool {
        let mut map = lock();
        let end = addr + len;
        let _transient = Transient::new(&map, addr, end);
        if !carve(&mut map, addr, end) {
            return false;
        }
        coalesce(&mut map, addr, end);
        let pieces: Vec<(u64, u64)> = map.range(addr..end).map(|(&s, e)| (s, e.end)).collect();
        let native = native_prot(prot);
        for (s, e) in pieces {
            if !map_piece(&mut map, s, e, section, offset + (s - addr), native) {
                return false;
            }
        }
        true
    }

    /// Fresh zeroed private memory at `addr` (a view of its own section).
    pub fn commit(addr: u64, len: u64, prot: MemProtect) -> bool {
        let Some(section) = Section::create(len) else {
            log::error!("[vm] section of {len:#x} bytes: {}", std::io::Error::last_os_error());
            return false;
        };
        map_view(addr, len, &section, 0, prot)
    }

    /// Give `[addr, addr+len)` back: it stays reserved as spare
    /// placeholder space. False when the range is not managed here.
    pub fn release(addr: u64, len: u64) -> bool {
        let mut map = lock();
        let end = addr + len;
        if containing(&map, addr).is_none() && map.range(addr..end).next().is_none() {
            return false;
        }
        let _transient = Transient::new(&map, addr, end);
        if !carve(&mut map, addr, end) {
            return false;
        }
        coalesce(&mut map, addr, end);
        set_owned(&mut map, addr, end, false);
        true
    }

    /// The section and offset backing `addr`, if it is a view.
    pub fn backing_of(addr: u64) -> Option<(Arc<Section>, u64)> {
        let map = lock();
        let (s, e) = containing(&map, addr)?;
        match e.kind {
            Kind::View { section, offset } => Some((section, offset + (addr - s))),
            Kind::Hold => None,
        }
    }
}
