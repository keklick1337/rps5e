// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Linux implementation of the host primitives.

use crate::MemProtect;

fn native_prot(p: MemProtect) -> i32 {
    let mut n = libc::PROT_NONE;
    if p.read {
        n |= libc::PROT_READ;
    }
    if p.write {
        n |= libc::PROT_WRITE;
    }
    if p.exec {
        n |= libc::PROT_EXEC;
    }
    n
}

/// Host page size.
pub fn page_size() -> u64 {
    // SAFETY: plain sysconf.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as u64 }
}

/// Leave the process at once, without the atexit chain.
pub fn exit_now(code: i32) -> ! {
    // SAFETY: immediate process exit.
    unsafe { libc::_exit(code) }
}

/// Kernel id of the calling thread.
pub fn current_tid() -> u64 {
    // SAFETY: plain syscall.
    unsafe { libc::gettid() as u64 }
}

/// Monotonic clock in nanoseconds.
pub fn monotonic_ns() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: valid out pointer.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// CPU time consumed by the calling thread, in nanoseconds.
pub fn thread_cpu_ns() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: valid out pointer.
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// Map anonymous zeroed memory at exactly `addr` without replacing
/// anything mapped there.
pub fn map_anon_noreplace(addr: u64, len: u64, prot: MemProtect) -> Result<(), String> {
    // SAFETY: MAP_FIXED_NOREPLACE never clobbers existing mappings.
    let p = unsafe {
        libc::mmap(
            addr as *mut libc::c_void,
            len as usize,
            native_prot(prot),
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED_NOREPLACE | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    if p == libc::MAP_FAILED || p as u64 != addr {
        return Err(std::io::Error::last_os_error().to_string());
    }
    Ok(())
}

/// Map anonymous zeroed memory anywhere.
pub fn map_anon(len: u64, prot: MemProtect) -> Option<u64> {
    // SAFETY: a fresh anonymous mapping.
    let p = unsafe { libc::mmap(std::ptr::null_mut(), len as usize, native_prot(prot), libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
    (p != libc::MAP_FAILED).then_some(p as u64)
}

/// Unmap `[addr, addr+len)`.
pub fn unmap(addr: u64, len: u64) {
    // SAFETY: the caller owns the range.
    unsafe { libc::munmap(addr as *mut libc::c_void, len as usize) };
}

/// Change the protection of `[addr, addr+len)`.
pub fn protect(addr: u64, len: u64, prot: MemProtect) -> bool {
    // SAFETY: the caller owns the range.
    unsafe { libc::mprotect(addr as *mut libc::c_void, len as usize, native_prot(prot)) == 0 }
}
