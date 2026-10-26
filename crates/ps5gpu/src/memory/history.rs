// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Diagnostics (`RPS5E_MEM_HISTORY=1`): a ring of what the GPU side did
//! to guest pages (uploads, GPU-dirty marks, copies into guest memory, CPU
//! faults, mapping changes), so a fatal guest fault can name the last
//! writers of the memory its registers point at.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Event {
    /// CPU-dirty runs uploaded into a mirror.
    Upload = 1,
    /// Pages marked GPU-written.
    GpuWritten,
    /// GPU-dirty pages cleared without a copy.
    GpuCleared,
    /// A CPU write fault on a watched page.
    CpuWrite,
    /// A CPU read fault on a GPU-written page.
    CpuRead,
    /// A write-through copy into guest memory (through an import).
    Through,
    /// A settling copy into guest memory.
    Settle,
    /// A download copy straight into guest memory.
    DownloadDirect,
    /// A download through a staging buffer.
    DownloadStaged,
    /// Host code wrote guest bytes through the backing alias.
    HostWrite,
    Map,
    Unmap,
}

impl Event {
    fn name(v: u64) -> &'static str {
        match v {
            1 => "upload",
            2 => "gpu-written",
            3 => "gpu-cleared",
            4 => "cpu-write-fault",
            5 => "cpu-read-fault",
            6 => "write-through",
            7 => "settle",
            8 => "download-direct",
            9 => "download-staged",
            10 => "host-write",
            11 => "map",
            12 => "unmap",
            _ => "?",
        }
    }
}

const ENTRIES: usize = 1 << 20;

struct Ring {
    slots: Box<[[AtomicU64; 4]]>,
    next: AtomicU64,
    start: std::time::Instant,
}

fn ring() -> Option<&'static Ring> {
    static R: OnceLock<Option<Ring>> = OnceLock::new();
    R.get_or_init(|| {
        std::env::var("RPS5E_MEM_HISTORY").is_ok_and(|v| v == "1").then(|| {
            log::warn!("ps5gpu: RPS5E_MEM_HISTORY=1: guest page history kept for fault reports");
            Ring { slots: (0..ENTRIES).map(|_| [const { AtomicU64::new(0) }; 4]).collect(), next: AtomicU64::new(0), start: std::time::Instant::now() }
        })
    })
    .as_ref()
}

#[inline]
pub fn enabled() -> bool {
    ring().is_some()
}

/// Record `event` over `[addr, addr+size)` (`tick`: the copy's tick, or 0).
#[inline]
pub fn note(event: Event, addr: u64, size: u64, tick: u64) {
    let Some(r) = ring() else { return };
    let i = r.next.fetch_add(1, Ordering::Relaxed) as usize % ENTRIES;
    let us = r.start.elapsed().as_micros() as u64;
    let s = &r.slots[i];
    s[0].store(addr, Ordering::Relaxed);
    s[1].store(addr.saturating_add(size), Ordering::Relaxed);
    s[2].store(tick, Ordering::Relaxed);
    s[3].store((us << 8) | event as u64, Ordering::Release);
}

/// The last `max` events touching the page holding `addr`, newest last.
pub fn describe(addr: u64, max: usize) -> Option<String> {
    let r = ring()?;
    let page = addr & !(super::PAGE_BYTES - 1);
    let (lo, hi) = (page, page + super::PAGE_BYTES);
    let now = r.start.elapsed().as_micros() as u64;
    let n = r.next.load(Ordering::Acquire) as usize;
    let count = n.min(ENTRIES);
    let mut hits: Vec<String> = Vec::new();
    for k in 0..count {
        let i = (n - 1 - k) % ENTRIES;
        let s = &r.slots[i];
        let (a, e) = (s[0].load(Ordering::Relaxed), s[1].load(Ordering::Relaxed));
        if a >= hi || e <= lo {
            continue;
        }
        let meta = s[3].load(Ordering::Acquire);
        let tick = s[2].load(Ordering::Relaxed);
        hits.push(format!("{}ms ago {} {a:#x}+{:#x}{}", now.saturating_sub(meta >> 8) / 1000, Event::name(meta & 0xFF), e - a, if tick != 0 { format!(" tick {tick}") } else { String::new() }));
        if hits.len() >= max {
            break;
        }
    }
    hits.reverse();
    Some(if hits.is_empty() { format!("page {page:#x}: no events in {count} recorded") } else { format!("page {page:#x}: {}", hits.join("; ")) })
}
