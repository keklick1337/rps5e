// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Write watches through userfaultfd write protection (Linux).
//!
//! A write watch makes a page read-only so the first CPU store faults. With
//! `mprotect` every watch and every resolved fault changes the protection
//! of a mapping range: the kernel splits and merges mappings (GTA V kept
//! ~14 000 of them), takes the address space's lock for writing (page
//! faults of every other thread wait meanwhile) and flushes the TLBs. With
//! userfaultfd write protection the bit sits in the page table entry: the
//! mappings stay whole, the address space is only read-locked, and only
//! write protection is flushed from the TLBs. Measured on this host with 20
//! busy threads: protecting a page 3.9 us instead of 8.3 us, a write fault
//! with its unprotection 0.9 us instead of 7.1 us.
//!
//! The userfaultfd is opened in user-mode-only mode (no privilege needed)
//! with `UFFD_FEATURE_SIGBUS`: a store to a protected page raises SIGBUS
//! (`BUS_ADRERR`) in the storing thread, which resolves it like the SIGSEGV
//! of a read-only page ([`super::handle_wp_fault`]). Stores of the kernel on
//! the process's behalf (a `read` into a watched buffer) fail with EFAULT,
//! as they do on a read-only page. Read watches still use `mprotect`.
//!
//! Only memfd-backed (shmem) guest mappings are registered. Where the
//! kernel lacks the features (write protection of shmem, of unpopulated
//! pages) or refuses a registration, pages keep `mprotect` write watches.
//! `RPS5E_UFFD_WATCH=0` turns it off. Not on other systems.

#[cfg(target_os = "linux")]
mod imp {
    use std::sync::atomic::{AtomicI32, Ordering};

    const UFFD_API: u64 = 0xAA;
    const UFFDIO_API: libc::c_ulong = 0xc018_aa3f;
    const UFFDIO_REGISTER: libc::c_ulong = 0xc020_aa00;
    const UFFDIO_WRITEPROTECT: libc::c_ulong = 0xc018_aa06;
    const UFFD_USER_MODE_ONLY: libc::c_int = 1;
    const UFFD_FEATURE_SIGBUS: u64 = 1 << 7;
    const UFFD_FEATURE_WP_HUGETLBFS_SHMEM: u64 = 1 << 12;
    const UFFD_FEATURE_WP_UNPOPULATED: u64 = 1 << 13;
    const UFFDIO_REGISTER_MODE_WP: u64 = 1 << 1;
    const UFFDIO_WRITEPROTECT_MODE_WP: u64 = 1 << 0;

    #[repr(C)]
    struct Api {
        api: u64,
        features: u64,
        ioctls: u64,
    }

    #[repr(C)]
    struct Range {
        start: u64,
        len: u64,
    }

    #[repr(C)]
    struct Register {
        range: Range,
        mode: u64,
        ioctls: u64,
    }

    #[repr(C)]
    struct WriteProtect {
        range: Range,
        mode: u64,
    }

    /// The userfaultfd (-1: none; -2: not opened yet).
    static FD: AtomicI32 = AtomicI32::new(-2);

    fn open() -> i32 {
        if std::env::var("RPS5E_UFFD_WATCH").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")) {
            return -1;
        }
        // SAFETY: plain syscalls on a descriptor this function owns.
        unsafe {
            let fd = libc::syscall(libc::SYS_userfaultfd, libc::O_CLOEXEC | libc::O_NONBLOCK | UFFD_USER_MODE_ONLY) as i32;
            if fd < 0 {
                log::info!("ps5gpu: userfaultfd unavailable ({}): write watches use mprotect", std::io::Error::last_os_error());
                return -1;
            }
            let wanted = UFFD_FEATURE_SIGBUS | UFFD_FEATURE_WP_HUGETLBFS_SHMEM | UFFD_FEATURE_WP_UNPOPULATED;
            let mut api = Api { api: UFFD_API, features: wanted, ioctls: 0 };
            if libc::ioctl(fd, UFFDIO_API, &mut api) != 0 || api.features & wanted != wanted {
                log::info!("ps5gpu: userfaultfd lacks write protection of shared memory ({}): write watches use mprotect", std::io::Error::last_os_error());
                libc::close(fd);
                return -1;
            }
            log::info!("ps5gpu: write watches through userfaultfd write protection");
            fd
        }
    }

    /// The descriptor, opened on first use (not from a signal handler:
    /// the guard opens it when it is made).
    pub fn fd() -> i32 {
        match FD.load(Ordering::Acquire) {
            -2 => {
                let fd = open();
                match FD.compare_exchange(-2, fd, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => fd,
                    Err(other) => {
                        if fd >= 0 {
                            // SAFETY: our own descriptor, never published.
                            unsafe { libc::close(fd) };
                        }
                        other
                    }
                }
            }
            fd => fd,
        }
    }

    /// Register `[addr, addr+size)` (page-aligned, every page mapped) for
    /// write protection. False when the kernel refuses (not shmem, say).
    pub fn register(addr: u64, size: u64) -> bool {
        let fd = fd();
        if fd < 0 || size == 0 {
            return false;
        }
        let mut r = Register { range: Range { start: addr, len: size }, mode: UFFDIO_REGISTER_MODE_WP, ioctls: 0 };
        // SAFETY: plain ioctl on our descriptor.
        unsafe { libc::ioctl(fd, UFFDIO_REGISTER, &mut r) == 0 }
    }

    /// Write-protect (`on`) or unprotect `[addr, addr+size)` of registered
    /// pages. Async-signal-safe (the fault handler calls it).
    pub fn protect(addr: u64, size: u64, on: bool) -> bool {
        let fd = FD.load(Ordering::Acquire);
        if fd < 0 {
            return false;
        }
        let mut w = WriteProtect { range: Range { start: addr, len: size }, mode: if on { UFFDIO_WRITEPROTECT_MODE_WP } else { 0 } };
        // SAFETY: plain ioctl on our descriptor.
        unsafe { libc::ioctl(fd, UFFDIO_WRITEPROTECT, &mut w) == 0 }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    pub fn fd() -> i32 {
        -1
    }
    pub fn register(_addr: u64, _size: u64) -> bool {
        false
    }
    pub fn protect(_addr: u64, _size: u64, _on: bool) -> bool {
        false
    }
}

pub use imp::{protect, register};

/// Open the userfaultfd now (outside any signal handler); true when write
/// watches can use it.
pub fn init() -> bool {
    imp::fd() >= 0
}
