// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest memory operations of the HLE C library (memcpy, memset, memmove,
//! the string copies).
//!
//! The guest range registry (`ps5vm::ranges`) is checked first; a range it
//! allows is copied in place, without an intermediate buffer (pages the GPU
//! write-watches fault and are tracked as for any host store). A range the
//! registry does not know is counted and logged (what, which range, what
//! the registry holds there, the guest caller) and then copied by the kernel
//! (`process_vm_readv` on this process), which checks every page itself: a
//! mapped host page is copied as on the console, an unmapped one makes the
//! operation stop there instead of faulting. A silent no-op would leave
//! stale bytes behind (stale vertex or skinning data for the GPU).
//!
//! A destination spanning several pages the GPU caches write-watch is
//! marked CPU-written in one call first
//! (`ps5gpu::memory::GuestGpuMemory::prepare_cpu_write`), so the copy does
//! not fault once per page.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Per-operation failure counts (name, count).
static FAILURES: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());
/// Failures of all operations.
static TOTAL: AtomicU64 = AtomicU64::new(0);
/// Bytes copied by the kernel fallback.
static FALLBACK_BYTES: AtomicU64 = AtomicU64::new(0);

/// The guest return address of an HLE call (for the failure log).
pub fn caller(ctx: &crate::context::CpuContext) -> u64 {
    ctx.read_u64(ctx.get(crate::context::CpuRegister::Rsp)).unwrap_or(0)
}

/// Failures counted so far, per operation.
pub fn failures() -> Vec<(&'static str, u64)> {
    FAILURES.lock().map(|v| v.clone()).unwrap_or_default()
}

/// All failures counted so far.
pub fn total_failures() -> u64 {
    TOTAL.load(Ordering::Relaxed)
}

/// Bytes copied through the kernel fallback so far.
pub fn fallback_bytes() -> u64 {
    FALLBACK_BYTES.load(Ordering::Relaxed)
}

/// Record a failed registry check of `[addr, addr+len)` by `op` (`write`:
/// the destination). Logged for the first 32 failures of an operation and
/// then at every power of two.
pub fn note_failure(op: &'static str, addr: u64, len: u64, write: bool, caller: u64) {
    TOTAL.fetch_add(1, Ordering::Relaxed);
    let n = {
        let Ok(mut v) = FAILURES.lock() else { return };
        match v.iter_mut().find(|(o, _)| *o == op) {
            Some((_, c)) => {
                *c += 1;
                *c
            }
            None => {
                v.push((op, 1));
                1
            }
        }
    };
    if n > 32 && !n.is_power_of_two() {
        return;
    }
    let runs = ps5vm::ranges::accessible_runs(addr, len, write);
    let missing = first_gap(addr, len, &runs);
    let near = ps5vm::ranges::containing(missing).map(|(s, e, p)| format!("{s:#x}..{e:#x} r={} w={}", p.read, p.write));
    log::warn!(
        "libc {op}: {} [{addr:#x}, +{len:#x}) not accessible in the guest range registry (first bad byte {missing:#x}, registry there: {}, accessible runs {:?}), caller {caller:#x}, failure #{n} of {op}",
        if write { "destination" } else { "source" },
        near.as_deref().unwrap_or("nothing"),
        &runs[..runs.len().min(4)],
    );
}

/// The first byte of `[addr, addr+len)` outside `runs`.
fn first_gap(addr: u64, len: u64, runs: &[(u64, u64)]) -> u64 {
    let mut cur = addr;
    for &(s, e) in runs {
        if s > cur {
            break;
        }
        cur = cur.max(e);
    }
    cur.min(addr.saturating_add(len.saturating_sub(1)))
}

/// Copy `len` bytes from `src` to `dst` with memmove semantics (`dst` and
/// `src` may overlap). Returns the bytes copied (`len` unless a page is not
/// mapped on the host).
pub fn copy(op: &'static str, dst: u64, src: u64, len: u64, caller: u64) -> u64 {
    if len == 0 {
        return 0;
    }
    let src_ok = ps5vm::ranges::accessible(src, len, false);
    let dst_ok = ps5vm::ranges::accessible(dst, len, true);
    if src_ok && dst_ok {
        ps5gpu::memory::global().prepare_cpu_write(dst, len);
        // SAFETY: both ranges are mapped with the needed access; `copy`
        // handles overlap (memmove).
        unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len as usize) };
        return len;
    }
    if !src_ok {
        note_failure(op, src, len, false, caller);
    }
    if !dst_ok {
        note_failure(op, dst, len, true, caller);
    }
    kernel_copy(dst, src, len)
}

/// Store host bytes `data` at guest `dst` (string copies). Returns the
/// bytes written.
pub fn store(op: &'static str, dst: u64, data: &[u8], caller: u64) -> u64 {
    let len = data.len() as u64;
    if len == 0 {
        return 0;
    }
    if ps5vm::ranges::accessible(dst, len, true) {
        ps5gpu::memory::global().prepare_cpu_write(dst, len);
        // SAFETY: the range is mapped writable; `data` is a host buffer.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst as *mut u8, data.len()) };
        return len;
    }
    note_failure(op, dst, len, true, caller);
    vm_copy(dst, data.as_ptr() as u64, len)
}

/// Fill `len` bytes at `dst` with `value`. Returns the bytes written.
pub fn fill(op: &'static str, dst: u64, value: u8, len: u64, caller: u64) -> u64 {
    if len == 0 {
        return 0;
    }
    if ps5vm::ranges::accessible(dst, len, true) {
        ps5gpu::memory::global().prepare_cpu_write(dst, len);
        // SAFETY: the range is mapped writable.
        unsafe { std::ptr::write_bytes(dst as *mut u8, value, len as usize) };
        return len;
    }
    note_failure(op, dst, len, true, caller);
    let chunk = [value; 4096];
    let mut done = 0;
    while done < len {
        let n = (len - done).min(chunk.len() as u64);
        let got = kernel_copy(dst + done, chunk.as_ptr() as u64, n);
        done += got;
        if got < n {
            break;
        }
    }
    done
}

/// Copy through the kernel, which checks both ranges page by page (no
/// fault in this process). Overlapping ranges go through a bounded bounce
/// buffer, chunk by chunk, in the direction that never overwrites source
/// bytes still to be read (`len` comes from the guest: never allocate it).
fn kernel_copy(dst: u64, src: u64, len: u64) -> u64 {
    let overlap = dst < src.saturating_add(len) && src < dst.saturating_add(len);
    if !overlap {
        return vm_copy(dst, src, len);
    }
    const CHUNK: u64 = 64 << 10;
    let mut tmp = vec![0u8; CHUNK as usize];
    let mut done = 0u64;
    if dst <= src {
        // Forward: each chunk's source lies at or after its destination.
        while done < len {
            let n = (len - done).min(CHUNK);
            let got = vm_copy(tmp.as_mut_ptr() as u64, src + done, n);
            let put = vm_copy(dst + done, tmp.as_ptr() as u64, got);
            done += put;
            if put < n {
                break;
            }
        }
    } else {
        // Backward: from the end, so a chunk's destination never covers
        // source bytes of a later chunk.
        let mut remaining = len;
        while remaining > 0 {
            let n = remaining.min(CHUNK);
            let off = remaining - n;
            if vm_copy(tmp.as_mut_ptr() as u64, src + off, n) < n {
                break;
            }
            let put = vm_copy(dst + off, tmp.as_ptr() as u64, n);
            done += put;
            if put < n {
                break;
            }
            remaining = off;
        }
    }
    done
}

/// Page by page (a transfer stops at the first page either side lacks).
fn vm_copy(dst: u64, src: u64, len: u64) -> u64 {
    const PAGE: u64 = 4096;
    let mut done = 0u64;
    while done < len {
        let (d, s) = (dst + done, src + done);
        let n = (len - done).min(PAGE - (d % PAGE)).min(PAGE - (s % PAGE)) as usize;
        let local = libc::iovec { iov_base: (dst + done) as *mut libc::c_void, iov_len: n };
        let remote = libc::iovec { iov_base: (src + done) as *mut libc::c_void, iov_len: n };
        // SAFETY: the kernel validates both ranges; nothing is accessed
        // directly by this process.
        let got = unsafe { libc::process_vm_readv(libc::getpid(), &local, 1, &remote, 1, 0) };
        if got <= 0 {
            break;
        }
        done += got as u64;
        if (got as usize) < n {
            break;
        }
    }
    FALLBACK_BYTES.fetch_add(done, Ordering::Relaxed);
    done
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps5vm::{MemProtect, VirtualMemory};

    #[test]
    fn copies_overlap_fill_and_unregistered_host_memory() {
        const BASE: u64 = 0x7B32_0000_0000;
        let vm = VirtualMemory::new();
        vm.map(BASE, 0x4000, MemProtect::RW).unwrap();
        let p = BASE as *mut u8;
        for i in 0..64u8 {
            unsafe { *p.add(i as usize) = i };
        }
        // memmove forward and backward overlap.
        assert_eq!(copy("memmove", BASE + 4, BASE, 32, 0), 32);
        let got: Vec<u8> = (0..36).map(|i| unsafe { *p.add(i) }).collect();
        assert_eq!(&got[4..36], &(0..32u8).collect::<Vec<_>>()[..]);
        assert_eq!(copy("memmove", BASE, BASE + 4, 32, 0), 32);
        let got: Vec<u8> = (0..32).map(|i| unsafe { *p.add(i) }).collect();
        assert_eq!(got, (0..32u8).collect::<Vec<_>>());
        assert_eq!(fill("memset", BASE + 0x100, 0xAB, 0x1000, 0), 0x1000);
        assert!((0..0x1000).all(|i| unsafe { *p.add(0x100 + i) } == 0xAB));
        // Host memory the registry does not know: copied by the kernel and
        // counted.
        let before = total_failures();
        let host = vec![7u8; 100];
        assert_eq!(copy("memcpy", BASE + 0x2000, host.as_ptr() as u64, 100, 0), 100);
        assert!((0..100).all(|i| unsafe { *p.add(0x2000 + i) } == 7));
        let mut out = vec![0u8; 50];
        assert_eq!(fill("memset", out.as_mut_ptr() as u64, 3, 50, 0), 50);
        assert!(out.iter().all(|&b| b == 3));
        assert!(total_failures() >= before + 2);
        assert!(failures().iter().any(|(o, n)| *o == "memcpy" && *n >= 1));
        // A copy running into an unmapped page stops there.
        let end = BASE + 0x4000;
        assert_eq!(copy("memcpy", end - 16, BASE, 64, 0), 16);
        vm.unmap(BASE, 0x4000).unwrap();
    }

    /// Host memory the registry does not know, overlapping both ways and
    /// longer than one bounce chunk: the kernel path copies it like memmove.
    #[test]
    fn kernel_path_moves_overlapping_ranges_in_both_directions() {
        let n = 200 * 1024usize;
        let pattern = |i: usize| (i.wrapping_mul(2654435761) >> 13) as u8;
        let mut buf: Vec<u8> = (0..n + 64).map(pattern).collect();
        let base = buf.as_mut_ptr() as u64;
        // Backward (dst > src).
        assert_eq!(copy("memmove", base + 24, base, n as u64, 0), n as u64);
        assert!((0..n).all(|i| buf[i + 24] == pattern(i)));
        buf.iter_mut().enumerate().for_each(|(i, b)| *b = pattern(i));
        // Forward (dst < src).
        assert_eq!(copy("memmove", base, base + 24, n as u64, 0), n as u64);
        assert!((0..n).all(|i| buf[i] == pattern(i + 24)));
    }

    /// A zero-length operation never looks at its pointers: null or
    /// unmapped ones are neither touched nor reported.
    #[test]
    fn zero_length_operations_do_not_touch_the_pointers() {
        assert_eq!(copy("zero-length-copy", 0, 0, 0, 0), 0);
        assert_eq!(copy("zero-length-copy", 0x10, 0xdead_0000_0000, 0, 0), 0);
        assert_eq!(fill("zero-length-fill", 0, 0xAB, 0, 0), 0);
        assert_eq!(store("zero-length-store", 0, &[], 0), 0);
        assert!(failures().iter().all(|(op, _)| !op.starts_with("zero-length")));
    }

    #[test]
    fn first_gap_finds_the_first_unlisted_byte() {
        assert_eq!(first_gap(0x1000, 0x3000, &[(0x1000, 0x2000)]), 0x2000);
        assert_eq!(first_gap(0x1000, 0x3000, &[]), 0x1000);
        assert_eq!(first_gap(0x1000, 0x1000, &[(0x1000, 0x2000)]), 0x1fff);
    }
}
