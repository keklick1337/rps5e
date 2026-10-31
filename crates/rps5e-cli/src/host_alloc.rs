// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The host heap of the emulator's own (Rust) allocations on Linux:
//! the C library's allocator, or mimalloc with `RPS5E_MIMALLOC=1`. (Other
//! targets keep the system allocator: the MinGW build of mimalloc needs
//! emulated TLS from libgcc, which does not link there.)
//!
//! The GPU command thread allocates and frees small vectors and maps for
//! every draw; with glibc's allocator that was ~10 % of its time (arena
//! locks, `realloc` growth, heap trimming), with mimalloc's per-thread
//! free lists a few percent. Guest code keeps its own allocator (the
//! game's libc module), and the GPU driver keeps the C library's.
//!
//! The choice is made once, at the first allocation, from the environment
//! (read without allocating). For an A/B inside one run
//! (`RPS5E_AB_SCOPE=release RPS5E_AB_ONLY=mimalloc`) both heaps are used:
//! [`start_ab_ticker`] switches new allocations between them with the A/B
//! periods, and a block is freed or grown by the heap that owns it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU8, Ordering::Relaxed};

/// 0: not decided yet, 1: system, 2: mimalloc, 3: both (A/B; new blocks
/// follow [`AB_MI`]).
static MODE: AtomicU8 = AtomicU8::new(0);
/// In A/B mode: new allocations go to mimalloc.
static AB_MI: AtomicU8 = AtomicU8::new(1);

const SYSTEM: u8 = 1;
const MIMALLOC: u8 = 2;
const BOTH: u8 = 3;

fn env_is(name: &core::ffi::CStr, test: impl Fn(&[u8]) -> bool) -> Option<bool> {
    // SAFETY: getenv does not allocate; the returned string is read at once.
    let p = unsafe { libc::getenv(name.as_ptr()) };
    if p.is_null() {
        return None;
    }
    // SAFETY: a NUL-terminated string from the environment.
    Some(test(unsafe { core::ffi::CStr::from_ptr(p) }.to_bytes()))
}

#[cold]
fn decide() -> u8 {
    // Opt-in (`RPS5E_MIMALLOC=1`): with it SILENT HILL 2 (Remake) stalls
    // after START GAME (~600 flips in 7 minutes instead of ~2 400; the
    // shader translation workers ran 1.5-4x slower), and GTA V's in-run
    // A/B showed no gain.
    let off = !env_is(c"RPS5E_MIMALLOC", |v| v == b"1" || v.eq_ignore_ascii_case(b"on")).unwrap_or(false)
        || env_is(c"RPS5E_SAFE", |v| v == b"1").unwrap_or(false);
    let ab = env_is(c"RPS5E_AB_PERIOD", |v| !v.is_empty() && v != b"0").unwrap_or(false)
        && env_is(c"RPS5E_AB_SCOPE", |v| v == b"release").unwrap_or(false)
        && env_is(c"RPS5E_AB_ONLY", |v| v.split(|&c| c == b',').any(|s| s.trim_ascii() == b"mimalloc")).unwrap_or(false);
    let mode = if ab && !off {
        BOTH
    } else if off {
        SYSTEM
    } else {
        MIMALLOC
    };
    // Two threads deciding at once reach the same answer.
    let _ = MODE.compare_exchange(0, mode, Relaxed, Relaxed);
    MODE.load(Relaxed)
}

#[inline]
fn mode() -> u8 {
    match MODE.load(Relaxed) {
        0 => decide(),
        m => m,
    }
}

/// The heap a new block comes from.
#[inline]
fn new_from_mimalloc() -> bool {
    match mode() {
        MIMALLOC => true,
        BOTH => AB_MI.load(Relaxed) == 1,
        _ => false,
    }
}

/// The heap owning `p` (A/B mode asks mimalloc).
#[inline]
fn owned_by_mimalloc(p: *mut u8) -> bool {
    match mode() {
        MIMALLOC => true,
        // SAFETY: a lookup in mimalloc's segment map; any pointer is fine.
        BOTH => unsafe { libmimalloc_sys::mi_is_in_heap_region(p as *const _) },
        _ => false,
    }
}

pub struct HostAlloc;

// SAFETY: every block is returned to (and resized by) the heap that made
// it; mimalloc's aligned functions honour the layout's alignment.
unsafe impl GlobalAlloc for HostAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if new_from_mimalloc() {
            unsafe { libmimalloc_sys::mi_malloc_aligned(layout.size(), layout.align()) as *mut u8 }
        } else {
            unsafe { System.alloc(layout) }
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if new_from_mimalloc() {
            unsafe { libmimalloc_sys::mi_zalloc_aligned(layout.size(), layout.align()) as *mut u8 }
        } else {
            unsafe { System.alloc_zeroed(layout) }
        }
    }

    #[inline]
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        if owned_by_mimalloc(p) {
            unsafe { libmimalloc_sys::mi_free(p as *mut _) }
        } else {
            unsafe { System.dealloc(p, layout) }
        }
    }

    #[inline]
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if owned_by_mimalloc(p) {
            unsafe { libmimalloc_sys::mi_realloc_aligned(p as *mut _, new_size, layout.align()) as *mut u8 }
        } else {
            unsafe { System.realloc(p, layout, new_size) }
        }
    }
}

/// In A/B mode: a thread that follows the A/B periods of switch `mimalloc`
/// (the fast periods allocate from mimalloc). Does nothing otherwise.
pub fn start_ab_ticker() {
    if mode() != BOTH {
        return;
    }
    let _ = std::thread::Builder::new().name("ab-alloc".into()).spawn(|| loop {
        let fast = ps5gpu::backend::ab::release_fast("mimalloc");
        AB_MI.store(fast as u8, Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(50));
    });
}

/// Which heap new allocations use (for the log).
pub fn describe() -> &'static str {
    match mode() {
        MIMALLOC => "mimalloc",
        BOTH => "mimalloc/system (A/B)",
        _ => "system",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Blocks of either heap are freed and resized by their owner, at
    /// every alignment, while new blocks switch heaps.
    #[test]
    fn blocks_go_back_to_their_heap() {
        let _ = mode();
        let h = HostAlloc;
        let mut blocks = Vec::new();
        for (i, align) in [1usize, 8, 16, 64, 4096].iter().cycle().take(40).enumerate() {
            let layout = Layout::from_size_align(24 + i * 40, *align).unwrap();
            // SAFETY: matching layouts, every block freed once below.
            unsafe {
                let p = if i % 2 == 0 { libmimalloc_sys::mi_malloc_aligned(layout.size(), layout.align()) as *mut u8 } else { System.alloc(layout) };
                assert!(!p.is_null() && p as usize % align == 0);
                p.write_bytes(0xA5, layout.size());
                blocks.push((p, layout, i % 2 == 0));
            }
        }
        for (p, _, mi) in &blocks {
            // SAFETY: live blocks.
            assert_eq!(unsafe { libmimalloc_sys::mi_is_in_heap_region(*p as *const _) }, *mi);
        }
        MODE.store(BOTH, Relaxed);
        for (p, layout, _) in blocks {
            // SAFETY: live blocks with their layouts.
            unsafe {
                let q = h.realloc(p, layout, layout.size() * 3);
                assert!(!q.is_null() && q as usize % layout.align() == 0);
                assert!(std::slice::from_raw_parts(q, layout.size()).iter().all(|&b| b == 0xA5));
                h.dealloc(q, Layout::from_size_align(layout.size() * 3, layout.align()).unwrap());
            }
        }
        for fast in [0u8, 1] {
            AB_MI.store(fast, Relaxed);
            let layout = Layout::from_size_align(100, 32).unwrap();
            // SAFETY: allocated and freed through the same allocator.
            unsafe {
                let p = h.alloc_zeroed(layout);
                assert!(std::slice::from_raw_parts(p, 100).iter().all(|&b| b == 0));
                assert_eq!(libmimalloc_sys::mi_is_in_heap_region(p as *const _), fast == 1);
                h.dealloc(p, layout);
            }
        }
    }
}
