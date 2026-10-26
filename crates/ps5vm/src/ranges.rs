// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Process-wide registry of guest-accessible address ranges.
//!
//! Every component that maps guest memory (image loader, CPU stacks/TLS,
//! the kernel memory manager) records the range and its protection here, so
//! HLE code can validate guest pointers before touching them, no matter who
//! created the mapping. Ranges never overlap: inserting carves out whatever
//! was there before.

use crate::MemProtect;
use std::collections::BTreeMap;

use std::sync::RwLock;

#[derive(Clone, Copy)]
struct Range {
    end: u64,
    prot: MemProtect,
}

static RANGES: RwLock<BTreeMap<u64, Range>> = RwLock::new(BTreeMap::new());


/// The tracked range containing `addr`: `(start, end, prot)`.
pub fn containing(addr: u64) -> Option<(u64, u64, MemProtect)> {
    let map = RANGES.read().unwrap();
    let (&start, r) = map.range(..=addr).next_back()?;
    (addr < r.end).then_some((start, r.end, r.prot))
}

/// Every tracked range in address order: `(start, end, prot)`.
pub fn snapshot() -> Vec<(u64, u64, MemProtect)> {
    RANGES.read().unwrap().iter().map(|(&start, r)| (start, r.end, r.prot)).collect()
}

/// Bumped by every change of the registry (see [`generation`]).
static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bumped when bytes stop being tracked or change protection.
static SHRINK_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Changes whenever a range is tracked or forgotten: answers of
/// [`accessible`] made at one value hold while it stays.
pub fn generation() -> u64 {
    GENERATION.load(std::sync::atomic::Ordering::Acquire)
}

/// Changes whenever tracked bytes are forgotten or re-tracked with another
/// protection: a `true` answer of [`accessible`] holds while it stays (new
/// ranges only make more bytes accessible).
pub fn shrink_generation() -> u64 {
    SHRINK_GENERATION.load(std::sync::atomic::Ordering::Acquire)
}

/// Remove `[start, end)` from the map; true when an existing range lost
/// bytes.
fn carve(map: &mut BTreeMap<u64, Range>, start: u64, end: u64) -> bool {
    let first = map.range(..start).next_back().map(|(&k, _)| k).unwrap_or(start);
    let keys: Vec<u64> = map.range(first..end).map(|(&k, _)| k).collect();
    let mut shrunk = false;
    for k in keys {
        let r = map.remove(&k).unwrap();
        if r.end <= start {
            map.insert(k, r);
            continue;
        }
        shrunk = true;
        if k < start {
            map.insert(k, Range { end: start, prot: r.prot });
        }
        if r.end > end {
            map.insert(end, r);
        }
    }
    shrunk
}

/// Record `[addr, addr+len)` as guest memory with `prot`.
pub fn track(addr: u64, len: u64, prot: MemProtect) {
    let Some(end) = addr.checked_add(len) else { return };
    if len == 0 {
        return;
    }
    let mut map = RANGES.write().unwrap();
    if carve(&mut map, addr, end) {
        SHRINK_GENERATION.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }
    // Coalesce with equal neighbours: allocators map in 64 KiB pieces and
    // lookups walk every range they cross.
    let (mut start, mut end) = (addr, end);
    if let Some((&k, r)) = map.range(..start).next_back() {
        if r.end == start && r.prot == prot {
            start = k;
        }
    }
    if let Some(r) = map.get(&end).copied() {
        if r.prot == prot {
            map.remove(&end);
            end = r.end;
        }
    }
    map.insert(start, Range { end, prot });
    GENERATION.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
}

/// Forget `[addr, addr+len)`.
pub fn untrack(addr: u64, len: u64) {
    let Some(end) = addr.checked_add(len) else { return };
    let mut map = RANGES.write().unwrap();
    if carve(&mut map, addr, end) {
        SHRINK_GENERATION.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }
    GENERATION.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
}

/// True when every byte of `[addr, addr+len)` is tracked and allows the
/// requested access.
pub fn accessible(addr: u64, len: u64, write: bool) -> bool {
    let Some(end) = addr.checked_add(len) else { return false };
    if len == 0 {
        return true;
    }
    // Every HLE guest access lands here: a hit in one of the ranges this
    // thread saw last, at an unchanged registry generation, needs no lock.
    let generation = generation();
    if let Ok(Some(c)) = CACHE.try_with(|c| c.lookup(generation, addr, end)) {
        return if write { c.write } else { c.read };
    }
    let map = RANGES.read().unwrap();
    if let Some((&start, r)) = map.range(..=addr).next_back() {
        if end <= r.end {
            let cached = Cached { generation, start, end: r.end, read: r.prot.read, write: r.prot.write };
            let _ = CACHE.try_with(|c| c.insert(cached));
            return if write { r.prot.write } else { r.prot.read };
        }
    }
    let mut cur = addr;
    let first = map.range(..=addr).next_back().map(|(&k, _)| k).unwrap_or(addr);
    for (&start, r) in map.range(first..end) {
        if r.end <= cur {
            continue;
        }
        if start > cur {
            return false;
        }
        let ok = if write { r.prot.write } else { r.prot.read };
        if !ok {
            return false;
        }
        cur = r.end;
        if cur >= end {
            return true;
        }
    }
    false
}

/// The parts of `[addr, addr+len)` that are tracked and allow the
/// requested access, in address order (adjacent ranges merged).
pub fn accessible_runs(addr: u64, len: u64, write: bool) -> Vec<(u64, u64)> {
    let Some(end) = addr.checked_add(len) else { return Vec::new() };
    let mut out: Vec<(u64, u64)> = Vec::new();
    if len == 0 {
        return out;
    }
    let map = RANGES.read().unwrap();
    let first = map.range(..=addr).next_back().map(|(&k, _)| k).unwrap_or(addr);
    for (&start, r) in map.range(first..end) {
        if r.end <= addr || !(if write { r.prot.write } else { r.prot.read }) {
            continue;
        }
        let (s, e) = (start.max(addr), r.end.min(end));
        match out.last_mut() {
            Some(last) if last.1 == s => last.1 = e,
            _ => out.push((s, e)),
        }
    }
    out
}

/// Per-thread copy of a range [`accessible`] resolved.
#[derive(Clone, Copy)]
struct Cached {
    generation: u64,
    start: u64,
    end: u64,
    read: bool,
    write: bool,
}

const EMPTY: Cached = Cached { generation: u64::MAX, start: 0, end: 0, read: false, write: false };

/// Entries of the per-thread cache: a copy checks both its source and its
/// destination, so one entry would evict itself on every call.
const CACHE_WAYS: usize = 4;

/// The last ranges [`accessible`] resolved on this thread, replaced round
/// robin.
struct RangeCache {
    entries: [std::cell::Cell<Cached>; CACHE_WAYS],
    next: std::cell::Cell<usize>,
}

impl RangeCache {
    #[inline]
    fn lookup(&self, generation: u64, addr: u64, end: u64) -> Option<Cached> {
        self.entries.iter().map(|c| c.get()).find(|c| c.generation == generation && addr >= c.start && end <= c.end)
    }

    fn insert(&self, c: Cached) {
        let ways = cache_ways();
        let i = self.next.get() % ways;
        self.entries[i].set(c);
        self.next.set((i + 1) % ways);
    }
}

/// `RPS5E_RANGE_CACHE4=0`: keep one cached range per thread.
fn cache_ways() -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static WAYS: AtomicUsize = AtomicUsize::new(0);
    match WAYS.load(Ordering::Relaxed) {
        0 => {
            let off = std::env::var("RPS5E_RANGE_CACHE4").is_ok_and(|v| matches!(v.trim(), "0" | "off" | "false" | "no"));
            let ways = if off { 1 } else { CACHE_WAYS };
            WAYS.store(ways, Ordering::Relaxed);
            ways
        }
        n => n,
    }
}

thread_local! {
    static CACHE: RangeCache = const {
        RangeCache {
            entries: [const { std::cell::Cell::new(EMPTY) }; CACHE_WAYS],
            next: std::cell::Cell::new(0),
        }
    };
}

/// True when every byte of `[addr, addr+len)` is tracked, whatever its
/// protection.
pub fn covered(addr: u64, len: u64) -> bool {
    let Some(end) = addr.checked_add(len) else { return false };
    if len == 0 {
        return true;
    }
    let map = RANGES.read().unwrap();
    let mut cur = addr;
    let first = map.range(..=addr).next_back().map(|(&k, _)| k).unwrap_or(addr);
    for (&start, r) in map.range(first..end) {
        if r.end <= cur {
            continue;
        }
        if start > cur {
            return false;
        }
        cur = r.end;
        if cur >= end {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_ranges_cover_and_carve() {
        track(0x10_0000, 0x1000, MemProtect::RW);
        track(0x10_1000, 0x1000, MemProtect::R);
        assert!(accessible(0x10_0800, 0x1000, false));
        assert!(!accessible(0x10_0800, 0x1000, true));
        track(0x10_0800, 0x100, MemProtect::NONE);
        assert!(!accessible(0x10_0700, 0x200, false));
        assert!(accessible(0x10_0000, 0x800, true));
        assert!(covered(0x10_0700, 0x200));
        untrack(0x10_0000, 0x2000);
        assert!(!accessible(0x10_0000, 1, false));
        assert!(!covered(0x10_0000, 1));
    }

    #[test]
    fn equal_neighbours_coalesce_and_split_again() {
        for i in 0..4u64 {
            track(0x20_0000 + i * 0x1000, 0x1000, MemProtect::RW);
        }
        let n = RANGES.read().unwrap().range(0x20_0000..0x20_4000).count();
        assert_eq!(n, 1);
        assert!(accessible(0x20_0000, 0x4000, true));
        untrack(0x20_1000, 0x1000);
        assert!(accessible(0x20_0000, 0x1000, true));
        assert!(!covered(0x20_0000, 0x2000));
        assert!(accessible(0x20_2000, 0x2000, true));
        track(0x20_1000, 0x1000, MemProtect::R);
        assert!(accessible(0x20_0000, 0x4000, false));
        assert!(!accessible(0x20_0000, 0x4000, true));
        untrack(0x20_0000, 0x4000);
    }

    #[test]
    fn accessible_runs_skip_holes_and_merge_neighbours() {
        track(0x40_0000, 0x1000, MemProtect::RW);
        track(0x40_1000, 0x1000, MemProtect::R);
        track(0x40_3000, 0x1000, MemProtect::NONE);
        track(0x40_5000, 0x2000, MemProtect::RW);
        assert_eq!(accessible_runs(0x40_0800, 0x6000, false), vec![(0x40_0800, 0x40_2000), (0x40_5000, 0x40_6800)]);
        assert_eq!(accessible_runs(0x40_0000, 0x7000, true), vec![(0x40_0000, 0x40_1000), (0x40_5000, 0x40_7000)]);
        assert!(accessible_runs(0x40_2000, 0x2000, false).is_empty());
        untrack(0x40_0000, 0x7000);
    }

    #[test]
    fn cached_range_follows_changes() {
        track(0x30_0000, 0x4000, MemProtect::RW);
        assert!(accessible(0x30_0100, 8, true));
        assert!(accessible(0x30_0200, 8, true), "cached hit");
        track(0x30_0000, 0x1000, MemProtect::R);
        assert!(!accessible(0x30_0100, 8, true), "protection change invalidates the cache");
        untrack(0x30_0000, 0x4000);
        assert!(!accessible(0x30_2000, 8, false), "untrack invalidates the cache");
    }

    /// Alternating lookups in separate ranges (a copy's source and
    /// destination) all hit the per-thread cache.
    #[test]
    fn cache_keeps_several_ranges() {
        track(0x50_0000, 0x1000, MemProtect::R);
        track(0x50_2000, 0x1000, MemProtect::RW);
        track(0x50_4000, 0x1000, MemProtect::RW);
        // Other tests change the registry concurrently: count only when the
        // generation stayed put.
        let g = generation();
        assert!(accessible(0x50_0000, 8, false));
        assert!(accessible(0x50_2000, 8, true));
        assert!(accessible(0x50_4000, 8, true));
        let hits = CACHE.with(|c| {
            [(0x50_0010, 0x50_0018), (0x50_2010, 0x50_2018), (0x50_4010, 0x50_4018)]
                .iter()
                .filter(|&&(a, e)| c.lookup(g, a, e).is_some())
                .count()
        });
        if cache_ways() == CACHE_WAYS && g == generation() {
            assert_eq!(hits, 3, "all three ranges stay cached");
        }
        // Cached answers keep the protection of each range.
        assert!(!accessible(0x50_0010, 8, true));
        assert!(accessible(0x50_2010, 8, true));
        untrack(0x50_0000, 0x5000);
        assert!(!accessible(0x50_2010, 8, false));
    }
}
