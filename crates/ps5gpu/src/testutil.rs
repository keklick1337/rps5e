// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Test helpers: guest-like memfd mappings inside the tracked guest space and
//! a SIGSEGV handler that resolves faults through registered test memory
//! managers (several tests run in parallel, each with its own manager).

use crate::memory::GuestGpuMemory;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::Once;

const SLOTS: usize = 2048;

/// The start of the next slot of the guest space handed to a test mapping.
static NEXT: AtomicU64 = AtomicU64::new(0x60_0000_0000);

/// Where the slots of [`guest_mapping`] must end: device addresses
/// translate only below 1 TiB (the BDA page table), and memory above it
/// takes other paths than the tests mean to check.
const LOW_SPACE_END: u64 = 1 << 40;

/// A slot of `len` bytes from `next`, sized to the mapping (aligned to the
/// tracker's blocks, so every slot starts on a block whatever slots other
/// tests took before it) with a 64 MiB unmapped gap after it. Slots sized 4 GiB each ran past
/// 1 TiB in a long run of the device tests, and every test after that
/// read its results through the wrong path (order-dependent failures).
fn slot(next: &AtomicU64, len: usize, end: u64) -> u64 {
    const GAP: u64 = 64 << 20;
    let stride = (len as u64).next_multiple_of(crate::memory::BLOCK_BYTES) + GAP;
    let start = next.fetch_add(stride, Ordering::SeqCst);
    assert!(start + len as u64 <= end, "test guest slots exhausted at {start:#x} (+{len:#x})");
    start
}

/// The memfd behind each guest view made by [`guest_mapping`].
static VIEWS: std::sync::Mutex<Vec<(u64, u64, i32)>> = std::sync::Mutex::new(Vec::new());
static MEMORIES: [AtomicPtr<GuestGpuMemory>; SLOTS] = [const { AtomicPtr::new(std::ptr::null_mut()) }; SLOTS];

extern "C" fn test_segv(sig: libc::c_int, info: *mut libc::siginfo_t, uc: *mut libc::c_void) {
    // SAFETY: kernel-provided signal context.
    unsafe {
        let uc = uc as *mut libc::ucontext_t;
        let err = (*uc).uc_mcontext.gregs[libc::REG_ERR as usize] as u64;
        let addr = (*info).si_addr() as u64;
        if (*info).si_code == 2 {
            let kind = if err & 2 != 0 || sig == libc::SIGBUS { crate::memory::FaultKind::Write } else { crate::memory::FaultKind::Read };
            for slot in &MEMORIES {
                let p = slot.load(Ordering::Acquire);
                if p.is_null() || !(*p).is_tracked(addr, 1) {
                    continue;
                }
                // SIGBUS: a store hit a page's userfaultfd write protection.
                let resolved = if sig == libc::SIGBUS { (*p).try_resolve_wp_fault(addr) } else { (*p).try_resolve_fault(kind, addr) };
                if resolved {
                    return;
                }
            }
        }
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = libc::SIG_DFL;
        libc::sigaction(sig, &sa, std::ptr::null_mut());
    }
}

/// A fresh memory manager (leaked) whose faults the test handler resolves.
pub fn test_memory() -> &'static GuestGpuMemory {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        // SAFETY: installs a process-wide SIGSEGV handler for this test binary.
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = test_segv as *const () as usize;
            sa.sa_flags = libc::SA_SIGINFO | libc::SA_NODEFER;
            libc::sigemptyset(&mut sa.sa_mask);
            libc::sigaction(libc::SIGSEGV, &sa, std::ptr::null_mut());
            libc::sigaction(libc::SIGBUS, &sa, std::ptr::null_mut());
        }
    });
    let m: &'static GuestGpuMemory = Box::leak(Box::new(GuestGpuMemory::new()));
    let slot = MEMORIES
        .iter()
        .find(|s| s.compare_exchange(std::ptr::null_mut(), m as *const _ as *mut _, Ordering::AcqRel, Ordering::Acquire).is_ok());
    assert!(slot.is_some(), "too many test memory managers");
    m
}

/// Map `len` bytes of a fresh memfd twice: (guest view inside the tracked
/// guest space, alias anywhere), register the view with `mem` (backing
/// offset 0, alias as its backing) and with the guest mapping registry.
pub fn guest_mapping(mem: &GuestGpuMemory, len: usize) -> (u64, u64) {
    guest_mapping_from(mem, len, || slot(&NEXT, len, LOW_SPACE_END))
}

/// [`guest_mapping`] in the AMM window (8 TiB), from its own slots.
pub fn amm_guest_mapping(mem: &GuestGpuMemory, len: usize) -> (u64, u64) {
    static NEXT_AMM: AtomicU64 = AtomicU64::new(0x800_4000_0000);
    guest_mapping_from(mem, len, || slot(&NEXT_AMM, len, crate::memory::SPACE_BYTES))
}

fn guest_mapping_from(mem: &GuestGpuMemory, len: usize, mut next: impl FnMut() -> u64) -> (u64, u64) {
    // SAFETY: plain memfd/mmap calls; the mappings live for the process.
    unsafe {
        let fd = libc::memfd_create(c"gpu-test".as_ptr(), 0);
        assert!(fd >= 0);
        assert_eq!(libc::ftruncate(fd, len as libc::off_t), 0);
        let mut view = libc::MAP_FAILED;
        for _ in 0..64 {
            let hint = next();
            view = libc::mmap(
                hint as *mut libc::c_void,
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED_NOREPLACE,
                fd,
                0,
            );
            if view != libc::MAP_FAILED {
                break;
            }
        }
        let alias = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0);
        assert_ne!(view, libc::MAP_FAILED);
        assert_ne!(alias, libc::MAP_FAILED);
        let (view, alias) = (view as u64, alias as u64);
        VIEWS.lock().unwrap().push((view, len as u64, fd));
        mem.set_backing_alias(alias, len as u64);
        mem.note_mapped(view, len as u64, libc::PROT_READ | libc::PROT_WRITE, Some(0));
        ps5vm::ranges::track(view, len as u64, ps5vm::MemProtect::RW);
        (view, alias)
    }
}

/// Map `len` bytes of private anonymous read/write memory (no memfd
/// behind it) at a free slot of the guest space, from the same slots as
/// [`guest_mapping`] so parallel tests never collide. The caller registers
/// it.
pub fn private_mapping(len: usize) -> u64 {
    for _ in 0..64 {
        let hint = slot(&NEXT, len, LOW_SPACE_END);
        // SAFETY: a fresh anonymous mapping that never replaces another.
        let p = unsafe {
            libc::mmap(
                hint as *mut libc::c_void,
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED_NOREPLACE,
                -1,
                0,
            )
        };
        if p != libc::MAP_FAILED {
            return p as u64;
        }
    }
    panic!("no free guest slot for a private mapping");
}

/// Unmap `[addr, addr+len)` of a guest view the way the memory manager
/// does: the caches are told, then the range becomes an inaccessible hold
/// (an uncommitted page of a partially resident range).
pub fn unmap_guest(mem: &GuestGpuMemory, addr: u64, len: u64) {
    mem.note_unmapped(addr, len);
    // SAFETY: the range belongs to a test guest view.
    let r = unsafe {
        libc::mmap(addr as *mut libc::c_void, len as usize, libc::PROT_NONE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | libc::MAP_FIXED, -1, 0)
    };
    assert_ne!(r, libc::MAP_FAILED);
    ps5vm::ranges::track(addr, len, ps5vm::MemProtect::NONE);
}

/// Map `[addr, addr+len)` of a guest view made by [`guest_mapping`] again
/// onto its own memfd pages (whatever bytes they hold), the way the memory
/// manager commits a page.
pub fn map_guest(mem: &GuestGpuMemory, addr: u64, len: u64) {
    let (view, _, fd) = *VIEWS.lock().unwrap().iter().find(|&&(v, n, _)| addr >= v && addr + len <= v + n).expect("a test guest view");
    // SAFETY: the range belongs to a test guest view.
    let r = unsafe {
        libc::mmap(addr as *mut libc::c_void, len as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED | libc::MAP_FIXED, fd, (addr - view) as libc::off_t)
    };
    assert_ne!(r, libc::MAP_FAILED);
    ps5vm::ranges::track(addr, len, ps5vm::MemProtect::RW);
    mem.note_mapped(addr, len, libc::PROT_READ | libc::PROT_WRITE, Some(addr - view));
}

/// Map `[addr, addr+len)` of a guest view made by [`guest_mapping`] onto
/// other bytes of its memfd (`backing`), in place over the current view
/// (no unmap first), the way the memory manager replaces a direct mapping
/// or an AMM remap moves a page.
pub fn remap_guest(mem: &GuestGpuMemory, addr: u64, len: u64, backing: u64) {
    let (_, n, fd) = *VIEWS.lock().unwrap().iter().find(|&&(v, n, _)| addr >= v && addr + len <= v + n).expect("a test guest view");
    assert!(backing + len <= n);
    // SAFETY: the range belongs to a test guest view.
    let r = unsafe {
        libc::mmap(addr as *mut libc::c_void, len as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED | libc::MAP_FIXED, fd, backing as libc::off_t)
    };
    assert_ne!(r, libc::MAP_FAILED);
    ps5vm::ranges::track(addr, len, ps5vm::MemProtect::RW);
    mem.note_mapped(addr, len, libc::PROT_READ | libc::PROT_WRITE, Some(backing));
}

/// Host permissions ("rw-") of the page holding `addr`.
pub fn perms(addr: u64) -> String {
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
    for line in maps.lines() {
        let mut it = line.split_whitespace();
        let range = it.next().unwrap();
        let p = it.next().unwrap();
        let (s, e) = range.split_once('-').unwrap();
        let (s, e) = (u64::from_str_radix(s, 16).unwrap(), u64::from_str_radix(e, 16).unwrap());
        if addr >= s && addr < e {
            let mut p = p[..3].to_string();
            // A page under userfaultfd write protection is read-only too.
            if uffd_write_protected(addr) {
                p.replace_range(1..2, "-");
            }
            return p;
        }
    }
    "---".into()
}

/// The page of `addr` has userfaultfd write protection (pagemap bit 57).
pub fn uffd_write_protected(addr: u64) -> bool {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open("/proc/self/pagemap") else { return false };
    let mut entry = [0u8; 8];
    if f.seek(SeekFrom::Start(addr / 4096 * 8)).is_err() || f.read_exact(&mut entry).is_err() {
        return false;
    }
    u64::from_le_bytes(entry) & (1 << 57) != 0
}
