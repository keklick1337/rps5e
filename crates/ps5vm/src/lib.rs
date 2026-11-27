// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! PS5 virtual memory manager.
//!
//! Backed by host `mmap`. Provides a flat 64-bit address space with the same
//! address layout as the console:
//! - main image: `0x0000_0008_0000_0000`
//! - direct memory arena (managed): `0x0000_0000_4000_0000 ..`
//! - flexible / shared: higher ranges.
//!
//! The manager is a thin wrapper around mmap MAP_ANON with explicit commit/
//! protect so guest PT_LOAD segments can be mapped at their preferred vaddr.

pub mod appenv;
mod backend;
pub mod os;
pub mod ranges;
pub mod shutdown;
pub mod test_games;
pub mod topology;

/// Instruction pointer of the access fault being resolved on this thread
/// (set by the CPU core before it asks the GPU memory tracker; diagnostics).
pub mod fault_site {
    use std::cell::Cell;
    thread_local! {
        static RIP: Cell<u64> = const { Cell::new(0) };
    }
    pub fn set(rip: u64) {
        RIP.with(|c| c.set(rip));
    }
    pub fn get() -> u64 {
        RIP.with(|c| c.get())
    }

    use std::sync::atomic::{AtomicU64, Ordering};

    /// Protection faults being resolved right now, one slot per fault:
    /// instruction address (0: free), fault address with the write flag in
    /// bit 63, and the time it began (microseconds since the first fault).
    static IN_FLIGHT: [[AtomicU64; 3]; 64] = [const { [const { AtomicU64::new(0) }; 3] }; 64];

    fn micros() -> u64 {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64
    }

    /// A fault at `rip` on `addr` is handed to its resolver; the slot to
    /// pass to [`leave`] (None: every slot is taken). Atomics only: called
    /// in signal handlers.
    pub fn enter(rip: u64, addr: u64, write: bool) -> Option<usize> {
        let now = micros();
        for (i, s) in IN_FLIGHT.iter().enumerate() {
            if s[0].compare_exchange(0, rip.max(1), Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                s[1].store(addr | (write as u64) << 63, Ordering::Relaxed);
                s[2].store(now, Ordering::Release);
                return Some(i);
            }
        }
        None
    }

    pub fn leave(slot: Option<usize>) {
        if let Some(i) = slot {
            IN_FLIGHT[i][0].store(0, Ordering::Release);
        }
    }

    /// The faults being resolved: (instruction address, fault address,
    /// write, milliseconds so far). For reports of a stalled GPU: a guest
    /// thread waiting in a fault is one that has not finished its writes.
    pub fn in_flight() -> Vec<(u64, u64, bool, u64)> {
        let now = micros();
        IN_FLIGHT
            .iter()
            .filter_map(|s| {
                let rip = s[0].load(Ordering::Acquire);
                let (a, t) = (s[1].load(Ordering::Relaxed), s[2].load(Ordering::Acquire));
                (rip != 0).then(|| (rip, a & !(1 << 63), a >> 63 != 0, now.saturating_sub(t) / 1000))
            })
            .collect()
    }
}

/// What the calling host thread has handed to foreign code (a driver
/// compiling a pipeline): the report of a host fault on that thread names
/// it.
pub mod thread_note {
    use std::cell::RefCell;
    thread_local! {
        static NOTE: RefCell<Option<String>> = const { RefCell::new(None) };
    }

    /// Clears the thread's note when dropped.
    pub struct Guard(());

    impl Drop for Guard {
        fn drop(&mut self) {
            let _ = NOTE.try_with(|n| n.borrow_mut().take());
        }
    }

    /// Note `what` for this thread until the guard is dropped.
    pub fn set(what: String) -> Guard {
        NOTE.with(|n| *n.borrow_mut() = Some(what));
        Guard(())
    }

    /// The thread's note (None also while it is being replaced: fault
    /// handlers call this).
    pub fn get() -> Option<String> {
        NOTE.try_with(|n| n.try_borrow().ok().and_then(|n| n.clone())).ok().flatten()
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn a_note_lasts_as_long_as_its_guard() {
            assert_eq!(super::get(), None);
            let guard = super::set("compute cs 1".into());
            assert_eq!(super::get().as_deref(), Some("compute cs 1"));
            // Another thread has its own.
            assert_eq!(std::thread::spawn(super::get).join().unwrap(), None);
            drop(guard);
            assert_eq!(super::get(), None);
        }
    }
}

use std::sync::Mutex;

#[derive(Debug, thiserror::Error)]
pub enum VmError {
    #[error("unaligned address/size: {0:#x}")]
    Unaligned(u64),
    #[error("out of memory mapping {1:#x} bytes at {0:#x}: {2}")]
    MapFailed(u64, u64, String),
    #[error("unmap failed: {0}")]
    UnmapFailed(String),
    #[error("protect failed: {0}")]
    ProtectFailed(String),
    #[error("address {0:#x} not mapped")]
    NotMapped(u64),
}

/// Memory protection bits (match ELF PF_*).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemProtect {
    pub read: bool,
    pub write: bool,
    pub exec: bool,
}

impl MemProtect {
    pub const NONE: Self = Self { read: false, write: false, exec: false };
    pub const R: Self = Self { read: true, write: false, exec: false };
    pub const RW: Self = Self { read: true, write: true, exec: false };
    pub const RX: Self = Self { read: true, write: false, exec: true };
    pub const RWX: Self = Self { read: true, write: true, exec: true };

    pub fn from_flags(pf: u32) -> Self {
        Self {
            read: pf & 0x4 != 0,
            write: pf & 0x2 != 0,
            exec: pf & 0x1 != 0,
        }
    }
}

/// mmap-backed address space. Internally synchronized.
pub struct VirtualMemory {
    inner: Mutex<backend::MmapBackend>,
}

impl Default for VirtualMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualMemory {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(backend::MmapBackend::new()),
        }
    }

    /// Map `size` bytes at `addr`. `addr` and `size` must be page-aligned.
    pub fn map(&self, addr: u64, size: u64, prot: MemProtect) -> Result<(), VmError> {
        self.inner.lock().unwrap().map(addr, size, prot)
    }

    pub fn unmap(&self, addr: u64, size: u64) -> Result<(), VmError> {
        self.inner.lock().unwrap().unmap(addr, size)
    }

    pub fn protect(&self, addr: u64, size: u64, prot: MemProtect) -> Result<(), VmError> {
        self.inner.lock().unwrap().protect(addr, size, prot)
    }

    /// Write guest bytes into mapped memory (panics if unmapped).
    /// Accesses are validated against the shared range registry only; they
    /// do not take the mapping lock, so threads never serialize on it.
    pub fn write(&self, addr: u64, src: &[u8]) -> Result<(), VmError> {
        if !ranges::accessible(addr, src.len() as u64, true) {
            return Err(VmError::NotMapped(addr));
        }
        // SAFETY: the range is mapped writable.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), addr as *mut u8, src.len()) };
        Ok(())
    }

    pub fn read(&self, addr: u64, dst: &mut [u8]) -> Result<(), VmError> {
        if !ranges::accessible(addr, dst.len() as u64, false) {
            return Err(VmError::NotMapped(addr));
        }
        // SAFETY: the range is mapped readable.
        unsafe { std::ptr::copy_nonoverlapping(addr as *const u8, dst.as_mut_ptr(), dst.len()) };
        Ok(())
    }

    /// Page size of the host (typically 4096).
    pub fn page_size() -> u64 {
        backend::page_size()
    }
}
