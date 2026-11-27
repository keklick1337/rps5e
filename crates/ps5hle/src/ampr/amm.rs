// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! AMM: address-mapping commands executed by the AMM submission engine.
//!
//! Direct maps place allocated direct memory at a fixed address. Automatic
//! maps draw their backing from direct memory given to AMM with the
//! automatic usage (`sceAmprAmmGiveDirectMemory`), first fit across the
//! free automatic ranges; unmapping returns that backing to the pool
//! unless a MultiMap alias still maps it. Remap moves mapped pages,
//! MultiMap aliases them, the protect commands change protection under a
//! mask. MapAsPrt turns a range into a partially resident one whose pages
//! AllocatePaForPrt backs from the pool and UnmapToPrt takes away again;
//! unbacked PRT pages stay unmapped (no zero page is mapped for GPU reads
//! of them).

use super::command_buffer::{AmmKind, AmmMap, AMM_PAGE_SIZE};
use crate::memory;
use crate::memory::spans::{RefSpans, SpanIndex};
use crate::registry::HleResult;
use crate::sync::{ERR_INVALID_ARGUMENT, ERR_TRY_AGAIN, OK};
use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

/// Virtual range AMM maps into.
pub const VA_BASE: u64 = 0x0800_0000_0000;
pub const VA_SIZE: u64 = 512 * 1024 * 1024 * 1024;

const PROT_CPU_READ: i32 = 0x01;
const PROT_CPU_WRITE: i32 = 0x02;
const PROT_CPU_EXEC: i32 = 0x04;
const PROT_GPU_READ: i32 = 0x10;
const PROT_GPU_WRITE: i32 = 0x20;
const PROT_AMPR_READ: i32 = 0x40;
const PROT_AMPR_WRITE: i32 = 0x80;
const PROT_ACP_READ: i32 = 0x100;
const PROT_ACP_WRITE: i32 = 0x200;

/// Automatic mappings and their aliases, `va -> (length, physical
/// start)`, with the number of mappings over every byte of direct memory
/// (what none maps any more goes back to the pool).
#[derive(Default)]
struct Mapped {
    spans: SpanIndex<u64>,
    backing: RefSpans,
}

impl Mapped {
    fn insert(&mut self, va: u64, (len, phys): (u64, u64)) {
        self.backing.add(phys, len);
        if let Some((l, ph)) = self.spans.insert(va, (len, phys)) {
            self.backing.sub(ph, l);
        }
    }

    fn remove(&mut self, va: &u64) {
        if let Some((l, ph)) = self.spans.remove(va) {
            self.backing.sub(ph, l);
        }
    }

    #[cfg(test)]
    fn get(&self, va: &u64) -> Option<&(u64, u64)> {
        self.spans.get(va)
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    fn values(&self) -> impl Iterator<Item = &(u64, u64)> {
        self.spans.values()
    }
}

#[derive(Default)]
struct Pool {
    /// Free automatic direct memory: start -> length (coalesced).
    free: BTreeMap<u64, u64>,
    mapped: Mapped,
    /// Partially resident (PRT) ranges: start -> length (coalesced).
    prt: BTreeMap<u64, u64>,
}

static POOL: Mutex<Option<Pool>> = Mutex::new(None);

fn with_pool<R>(f: impl FnOnce(&mut Pool) -> R) -> R {
    let mut g: MutexGuard<'_, Option<Pool>> = POOL.lock().unwrap_or_else(|p| p.into_inner());
    f(g.get_or_insert_with(Pool::default))
}

/// The automatic pool cannot back `len` bytes at `va`: EAGAIN, as the
/// console answers a title that asked for more than it gave AMM. Logged
/// with what the pool holds (a title treats this as fatal).
fn pool_exhausted(what: &str, va: u64, len: u64) -> HleResult {
    let (free, mapped, prt) =
        with_pool(|p| (p.free.values().sum::<u64>(), p.mapped.values().map(|&(l, _)| l).sum::<u64>(), p.prt.values().sum::<u64>()));
    log::warn!(
        "[ampr] AMM {what} {va:#x} (+{len:#x}): the automatic pool has {free:#x} bytes free ({mapped:#x} mapped, {prt:#x} of PRT ranges)"
    );
    ERR_TRY_AGAIN
}

fn add_free(free: &mut BTreeMap<u64, u64>, start: u64, len: u64) {
    let (mut start, mut end) = (start, start + len);
    if let Some((&s, &l)) = free.range(..=start).next_back() {
        if s + l >= start {
            start = s;
            end = end.max(s + l);
            free.remove(&s);
        }
    }
    while let Some((&s, &l)) = free.range(start..).next() {
        if s > end {
            break;
        }
        end = end.max(s + l);
        free.remove(&s);
    }
    free.insert(start, end - start);
}

fn remove_free(free: &mut BTreeMap<u64, u64>, start: u64, len: u64) {
    let end = start + len;
    // The ranges do not overlap: the one before `start` is the only one
    // starting below it that can reach into the span.
    let mut at = match free.range(..start).next_back() {
        Some((&s, &l)) if s + l > start => s,
        _ => start,
    };
    while at < end {
        let Some((s, l)) = free.range(at..end).next().map(|(&s, &l)| (s, l)) else { break };
        free.remove(&s);
        if s < start {
            free.insert(s, start - s);
        }
        if s + l > end {
            free.insert(end, s + l - end);
        }
        at = s + l;
    }
}

/// Make direct memory at `[start, start + len)` available to automatic maps.
pub fn give_automatic(start: u64, len: u64) {
    with_pool(|p| add_free(&mut p.free, start, len));
}

/// Take free direct memory back from the automatic pool (tests release
/// what they gave).
#[cfg(test)]
pub(crate) fn withdraw_automatic(start: u64, len: u64) {
    with_pool(|p| remove_free(&mut p.free, start, len));
}

/// CPU/GPU protection for AMPR/ACP access bits.
pub fn normalize_protection(prot: i32) -> i32 {
    let mut n = prot & (PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC | PROT_GPU_READ | PROT_GPU_WRITE);
    if prot & (PROT_AMPR_READ | PROT_ACP_READ) != 0 {
        n |= PROT_CPU_READ;
    }
    if prot & (PROT_AMPR_WRITE | PROT_ACP_WRITE) != 0 {
        n |= PROT_CPU_READ | PROT_CPU_WRITE;
    }
    n
}

fn map_direct(va: u64, len: u64, prot: i32, phys: u64) -> HleResult {
    match memory::map_direct_at(va, len, normalize_protection(prot), memory::MAP_FIXED, phys, AMM_PAGE_SIZE) {
        Ok(_) => OK,
        Err(e) => e,
    }
}

fn map_automatic(va: u64, size: u64, prot: i32) -> HleResult {
    if size == 0
        || va % AMM_PAGE_SIZE != 0
        || size % AMM_PAGE_SIZE != 0
        || !(VA_BASE..VA_BASE + VA_SIZE).contains(&va)
        || size > VA_BASE + VA_SIZE - va
    {
        return ERR_INVALID_ARGUMENT;
    }
    // The target is released first: an automatic mapping already there
    // gives its pages back to the pool before the new backing is taken.
    let _ = memory::unmap(va, size);
    let ranges = with_pool(|p| {
        p.reclaim(va, size);
        p.take(size)
    });
    let Some(ranges) = ranges else { return pool_exhausted("map", va, size) };
    let mut mapped = 0;
    for &(phys, len) in &ranges {
        let r = map_direct(va + mapped, len, prot, phys);
        if r != OK {
            if mapped != 0 {
                let _ = memory::unmap(va, mapped);
            }
            with_pool(|p| {
                for &(s, l) in &ranges {
                    add_free(&mut p.free, s, l);
                }
                for a in p.mapped.spans.starts_in(va, va + mapped) {
                    p.mapped.remove(&a);
                }
            });
            return r;
        }
        with_pool(|p| p.mapped.insert(va + mapped, (len, phys)));
        mapped += len;
    }
    OK
}

/// Return the automatic backing of `[va, va + size)` to the pool.
fn reclaim_automatic(va: u64, size: u64) {
    with_pool(|p| p.reclaim(va, size));
}

impl Pool {
    /// The automatic mappings overlapping `[va, va + size)`, clipped to it:
    /// (va, length, physical start).
    fn entries(&self, va: u64, size: u64) -> Vec<(u64, u64, u64)> {
        let end = va + size;
        let mut hits = Vec::new();
        self.mapped.spans.overlapping(va, size, &mut hits);
        for (a, l, ph) in &mut hits {
            let first = (*a).max(va);
            let last = (*a + *l).min(end);
            (*a, *l, *ph) = (first, last - first, *ph + (first - *a));
        }
        hits
    }

    /// Forget the automatic mappings of `[va, va + size)` without freeing
    /// their backing; returns them clipped as [`Pool::entries`].
    fn detach(&mut self, va: u64, size: u64) -> Vec<(u64, u64, u64)> {
        let end = va + size;
        let mut hits = Vec::new();
        self.mapped.spans.overlapping(va, size, &mut hits);
        for (a, l, ph) in &mut hits {
            let (start, len, phys) = (*a, *l, *ph);
            let first = start.max(va);
            let last = (start + len).min(end);
            // The parts outside stay mapped: their backing stays counted
            // while the entry is cut.
            if start < first {
                self.mapped.insert(start, (first - start, phys));
            } else {
                self.mapped.remove(&start);
            }
            if start + len > last {
                self.mapped.insert(last, (start + len - last, phys + (last - start)));
            }
            (*a, *l, *ph) = (first, last - first, phys + (first - start));
        }
        hits
    }

    /// Return the automatic backing of `[va, va + size)` to the free ranges
    /// (pages an alias still maps stay taken).
    fn reclaim(&mut self, va: u64, size: u64) {
        for (_, l, ph) in self.detach(va, size) {
            let (free, mapped) = (&mut self.free, &self.mapped);
            mapped.backing.unreferenced(ph, l, |s, len| add_free(free, s, len));
        }
    }

    fn mark_prt(&mut self, va: u64, size: u64) {
        if size != 0 {
            add_free(&mut self.prt, va, size);
        }
    }

    fn clear_prt(&mut self, va: u64, size: u64) {
        remove_free(&mut self.prt, va, size);
    }

    fn is_prt(&self, va: u64, size: u64) -> bool {
        size == 0 || self.prt.range(..=va).next_back().is_some_and(|(&s, &l)| s + l >= va + size)
    }

    /// Take `size` bytes of free backing, first fit; None (and nothing
    /// taken) when the pool cannot cover it.
    fn take(&mut self, size: u64) -> Option<Vec<(u64, u64)>> {
        let mut ranges = Vec::new();
        let mut left = size;
        for (&start, &avail) in &self.free {
            let len = left.min(avail);
            ranges.push((start, len));
            left -= len;
            if left == 0 {
                break;
            }
        }
        if left != 0 {
            return None;
        }
        for &(s, l) in &ranges {
            remove_free(&mut self.free, s, l);
        }
        Some(ranges)
    }
}

/// `[va, va + size)` lies in the AMM virtual range.
pub(super) fn in_amm_range(va: u64, size: u64) -> bool {
    va >= VA_BASE && size <= VA_SIZE && va - VA_BASE <= VA_SIZE - size
}

/// The direct mappings covering all of `[va, va + size)`; None when part of
/// it is not mapped.
fn mapped_pieces(va: u64, size: u64) -> Option<Vec<memory::DirectPiece>> {
    let pieces = memory::direct_pieces(va, size);
    let mut at = va;
    for p in &pieces {
        if p.start != at {
            return None;
        }
        at += p.size;
    }
    (at == va + size).then_some(pieces)
}

/// Remap (`alias == false`) moves the pages of `source` to `va`; MultiMap
/// (`alias`) maps them at `va` as well. The pages get the command's
/// protection; the source range of a move is left unmapped.
fn remap(m: &AmmMap, alias: bool) -> HleResult {
    let (va, source, size) = (m.va, m.source, m.size);
    if !in_amm_range(va, size) || !in_amm_range(source, size) {
        log::error!("[ampr] AMM remap {source:#x} -> {va:#x} (+{size:#x}): outside the AMM virtual range");
        return ERR_INVALID_ARGUMENT;
    }
    if alias && va < source + size && source < va + size {
        log::error!("[ampr] AMM multimap {source:#x} -> {va:#x} (+{size:#x}): onto its own source range");
        return ERR_INVALID_ARGUMENT;
    }
    let Some(pieces) = mapped_pieces(source, size) else {
        log::error!("[ampr] AMM remap {source:#x} -> {va:#x} (+{size:#x}): the source is not fully mapped");
        return ERR_INVALID_ARGUMENT;
    };
    let automatic = with_pool(|p| if alias { p.entries(source, size) } else { p.detach(source, size) });
    if !alias {
        let r = memory::unmap(source, size);
        if r != OK {
            return r;
        }
        with_pool(|p| p.clear_prt(source, size));
    }
    let _ = memory::unmap(va, size);
    with_pool(|p| {
        p.reclaim(va, size);
        p.clear_prt(va, size);
    });
    let prot = normalize_protection(m.prot);
    for piece in &pieces {
        let at = va + (piece.start - source);
        let r = memory::map_direct_typed_at(at, piece.size, prot, piece.phys, AMM_PAGE_SIZE, piece.mtype);
        if r != OK {
            log::error!("[ampr] AMM remap: mapping {at:#x} (+{:#x}) from {:#x} failed: {r:#x}", piece.size, piece.phys);
            return r;
        }
    }
    with_pool(|p| {
        for (a, l, ph) in automatic {
            p.mapped.insert(va + (a - source), (l, ph));
        }
    });
    OK
}

/// ModifyProtect / ModifyMtypeProtect: every page of the (fully mapped)
/// range keeps its protection bits outside `mask` and takes the command's
/// inside it.
fn modify_protect(m: &AmmMap, with_type: bool) -> HleResult {
    let Some(pieces) = mapped_pieces(m.va, m.size) else {
        log::error!("[ampr] AMM modify protect {:#x} (+{:#x}): the range is not fully mapped", m.va, m.size);
        return ERR_INVALID_ARGUMENT;
    };
    for piece in pieces {
        let prot = normalize_protection((piece.prot & !m.mask) | (m.prot & m.mask));
        let r = if with_type {
            memory::mtype_protect(piece.start, piece.size, m.mtype, prot)
        } else {
            memory::protect(piece.start, piece.size, prot)
        };
        if r != OK {
            return r;
        }
    }
    OK
}

/// MapAsPrt and UnmapToPrt: the range loses its backing (automatic pages
/// go back to the pool) and is partially resident; its unbacked pages
/// stay unmapped.
fn map_as_prt(va: u64, size: u64) -> HleResult {
    if !in_amm_range(va, size) {
        log::error!("[ampr] AMM map as PRT {va:#x} (+{size:#x}): outside the AMM virtual range");
        return ERR_INVALID_ARGUMENT;
    }
    let _ = memory::unmap(va, size);
    with_pool(|p| {
        p.reclaim(va, size);
        p.mark_prt(va, size);
    });
    OK
}

/// AllocatePaForPrt: the backed pages of the PRT range take the command's
/// type and protection, the unbacked ones are backed from the automatic
/// pool.
fn allocate_pa_for_prt(m: &AmmMap) -> HleResult {
    let (va, size) = (m.va, m.size);
    if !in_amm_range(va, size) || !with_pool(|p| p.is_prt(va, size)) {
        log::error!("[ampr] AMM allocate PA for PRT {va:#x} (+{size:#x}): not a PRT range");
        return ERR_INVALID_ARGUMENT;
    }
    let prot = normalize_protection(m.prot);
    let mut at = va;
    let mut runs = Vec::new();
    for piece in memory::direct_pieces(va, size) {
        if piece.start > at {
            runs.push((at, piece.start - at, false));
        }
        runs.push((piece.start, piece.size, true));
        at = piece.start + piece.size;
    }
    if at < va + size {
        runs.push((at, va + size - at, false));
    }
    for (start, len, backed) in runs {
        if backed {
            let r = memory::mtype_protect(start, len, m.mtype, prot);
            if r != OK {
                return r;
            }
            continue;
        }
        let Some(ranges) = with_pool(|p| p.take(len)) else { return pool_exhausted("allocate PA for PRT", start, len) };
        let mut mapped = 0;
        for (i, &(phys, l)) in ranges.iter().enumerate() {
            let r = memory::map_direct_typed_at(start + mapped, l, prot, phys, AMM_PAGE_SIZE, m.mtype);
            if r != OK {
                with_pool(|p| {
                    for &(s, l) in &ranges[i..] {
                        add_free(&mut p.free, s, l);
                    }
                });
                return r;
            }
            with_pool(|p| p.mapped.insert(start + mapped, (l, phys)));
            mapped += l;
        }
    }
    OK
}

/// Execute one AMM command.
pub fn execute(m: &AmmMap) -> HleResult {
    if !super::command_buffer::valid_map_args(m.va, m.size) {
        return ERR_INVALID_ARGUMENT;
    }
    let r = match m.kind {
        AmmKind::Unmap => {
            let r = memory::unmap(m.va, m.size);
            if r == OK {
                reclaim_automatic(m.va, m.size);
            }
            r
        }
        AmmKind::MapAuto => map_automatic(m.va, m.size, m.prot),
        AmmKind::MapDirect => map_direct(m.va, m.size, m.prot, m.dmem_offset),
        AmmKind::Remap => return remap(m, false),
        AmmKind::MultiMap => return remap(m, true),
        AmmKind::ModifyProtect => return modify_protect(m, false),
        AmmKind::ModifyMtypeProtect => return modify_protect(m, true),
        AmmKind::MapAsPrt | AmmKind::UnmapToPrt => return map_as_prt(m.va, m.size),
        AmmKind::AllocatePaForPrt => return allocate_pa_for_prt(m),
    };
    if r == OK {
        with_pool(|p| p.clear_prt(m.va, m.size));
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_ranges_coalesce_and_split() {
        let mut f = BTreeMap::new();
        add_free(&mut f, 0x4000, 0x4000);
        add_free(&mut f, 0x8000, 0x4000);
        add_free(&mut f, 0x10000, 0x4000);
        assert_eq!(f.iter().map(|(&a, &b)| (a, b)).collect::<Vec<_>>(), vec![(0x4000, 0x8000), (0x10000, 0x4000)]);
        remove_free(&mut f, 0x6000, 0x1000);
        assert_eq!(
            f.iter().map(|(&a, &b)| (a, b)).collect::<Vec<_>>(),
            vec![(0x4000, 0x2000), (0x7000, 0x5000), (0x10000, 0x4000)]
        );
    }

    /// Mapping over an automatic mapping reuses its pages instead of
    /// leaking them; a map the pool cannot cover leaves the target's old
    /// pages in the pool.
    #[test]
    fn remap_returns_pages_to_the_pool() {
        const PAGE: u64 = AMM_PAGE_SIZE;
        let mut p = Pool::default();
        add_free(&mut p.free, 0x10_0000, PAGE);
        let (orig, fresh) = (VA_BASE, VA_BASE + 0x100_0000);
        let map = |p: &mut Pool, va: u64, size: u64| {
            p.reclaim(va, size);
            let ranges = p.take(size)?;
            let mut at = va;
            for &(phys, len) in &ranges {
                p.mapped.insert(at, (len, phys));
                at += len;
            }
            Some(ranges)
        };
        assert_eq!(map(&mut p, orig, PAGE), Some(vec![(0x10_0000, PAGE)]));
        assert_eq!(map(&mut p, orig, PAGE), Some(vec![(0x10_0000, PAGE)]), "the same page backs the remap");
        assert!(p.free.is_empty());
        assert_eq!(map(&mut p, orig, 2 * PAGE), None, "one page cannot cover two");
        assert_eq!(p.free.iter().map(|(&a, &l)| (a, l)).collect::<Vec<_>>(), vec![(0x10_0000, PAGE)]);
        assert!(p.mapped.is_empty(), "the failed map left the target unmapped");
        assert_eq!(map(&mut p, fresh, PAGE), Some(vec![(0x10_0000, PAGE)]));
    }

    /// Pages a MultiMap alias still maps are not returned to the pool when
    /// the original goes; the last mapping returns them. Detach (Remap)
    /// moves entries without freeing them.
    #[test]
    fn aliases_keep_pages_taken() {
        const PAGE: u64 = AMM_PAGE_SIZE;
        let mut p = Pool::default();
        let (orig, alias, moved) = (VA_BASE, VA_BASE + 0x100_0000, VA_BASE + 0x200_0000);
        p.mapped.insert(orig, (2 * PAGE, 0x10_0000));
        for (a, l, ph) in p.entries(orig + PAGE, 2 * PAGE) {
            p.mapped.insert(alias + (a - orig - PAGE), (l, ph));
        }
        assert_eq!(p.mapped.get(&alias), Some(&(PAGE, 0x10_0000 + PAGE)));
        p.reclaim(orig, 2 * PAGE);
        assert_eq!(p.free.iter().map(|(&a, &l)| (a, l)).collect::<Vec<_>>(), vec![(0x10_0000, PAGE)], "the aliased page stays taken");
        let detached = p.detach(alias, PAGE);
        assert_eq!(detached, vec![(alias, PAGE, 0x10_0000 + PAGE)]);
        assert!(p.mapped.is_empty());
        p.mapped.insert(moved, (PAGE, 0x10_0000 + PAGE));
        assert_eq!(p.free.len(), 1, "detaching frees nothing");
        p.reclaim(moved, PAGE);
        assert_eq!(p.free.iter().map(|(&a, &l)| (a, l)).collect::<Vec<_>>(), vec![(0x10_0000, 2 * PAGE)]);
    }

    #[test]
    fn prt_pages() {
        const PAGE: u64 = AMM_PAGE_SIZE;
        let mut p = Pool::default();
        p.mark_prt(VA_BASE, 3 * PAGE);
        assert!(p.is_prt(VA_BASE, 3 * PAGE));
        assert!(!p.is_prt(VA_BASE, 4 * PAGE));
        p.clear_prt(VA_BASE + PAGE, PAGE);
        assert!(p.is_prt(VA_BASE, PAGE) && !p.is_prt(VA_BASE, 2 * PAGE) && p.is_prt(VA_BASE + 2 * PAGE, PAGE));
    }

    /// The pool stays cheap under a title that commits and releases
    /// pieces of a partially resident window all the time: 20 000
    /// automatic mappings, then 50 000 commands (commit, release, and now
    /// and then an alias) of 16 KiB to 1 MiB at pseudo-random places of a
    /// 64 GiB window, as `allocate_pa_for_prt`, `map_as_prt` and `remap`
    /// drive it. Every backing taken and the final state are compared with
    /// a page-by-page model, and the replay has a time bound that work
    /// proportional to the number of mappings per command misses by far.
    #[test]
    fn pool_keeps_up_with_prt_commit_streams() {
        const PAGE: u64 = AMM_PAGE_SIZE;
        const SPAN: u64 = 64 << 30;
        const PAGES: usize = (SPAN / PAGE) as usize;
        const POOL_PAGES: usize = PAGES;
        const MAPPINGS: usize = 20_000;
        const COMMANDS: usize = 50_000;

        // Page model: backing page + 1 of every window page, and how many
        // window pages map each backing page.
        struct Model {
            phys: Vec<u32>,
            refs: Vec<u8>,
            /// One bit per backing page: set while nothing maps it.
            free: Vec<u64>,
            /// No word below this one has a free page.
            lowest_free: usize,
        }
        impl Model {
            fn take(&mut self) -> u32 {
                while self.free[self.lowest_free] == 0 {
                    self.lowest_free += 1;
                }
                let bit = self.free[self.lowest_free].trailing_zeros() as usize;
                self.free[self.lowest_free] &= !(1 << bit);
                let page = self.lowest_free * 64 + bit;
                self.refs[page] = 1;
                page as u32
            }
            fn drop_page(&mut self, page: usize) {
                let ph = std::mem::take(&mut self.phys[page]);
                if ph != 0 {
                    let ph = ph as usize - 1;
                    self.refs[ph] -= 1;
                    if self.refs[ph] == 0 {
                        self.free[ph / 64] |= 1 << (ph % 64);
                        self.lowest_free = self.lowest_free.min(ph / 64);
                    }
                }
            }
        }
        let mut model = Model { phys: vec![0; PAGES], refs: vec![0; POOL_PAGES], free: vec![u64::MAX; POOL_PAGES / 64], lowest_free: 0 };
        let mut p = Pool::default();
        add_free(&mut p.free, 0, POOL_PAGES as u64 * PAGE);
        p.mark_prt(VA_BASE, SPAN);
        let va_of = |page: usize| VA_BASE + page as u64 * PAGE;

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // Time spent in the pool (the model's bookkeeping is not counted).
        let spent = std::cell::Cell::new(std::time::Duration::ZERO);
        let timed = |started: std::time::Instant| spent.set(spent.get() + started.elapsed());
        let commit = |p: &mut Pool, model: &mut Model, page: usize, pages: usize| {
            let started = std::time::Instant::now();
            assert!(p.is_prt(va_of(page), pages as u64 * PAGE));
            timed(started);
            let mut at = page;
            while at < page + pages {
                if model.phys[at] != 0 {
                    at += 1;
                    continue;
                }
                let mut end = at;
                while end < page + pages && model.phys[end] == 0 {
                    end += 1;
                }
                let started = std::time::Instant::now();
                let taken = p.take((end - at) as u64 * PAGE).expect("the pool covers the window");
                let mut q = at;
                for &(phys, len) in &taken {
                    p.mapped.insert(va_of(q), (len, phys));
                    q += (len / PAGE) as usize;
                }
                timed(started);
                let mut q = at;
                for (phys, len) in taken {
                    for i in 0..len / PAGE {
                        let expected = model.take();
                        assert_eq!(phys / PAGE + i, expected as u64, "first fit");
                        model.phys[q] = expected + 1;
                        q += 1;
                    }
                }
                assert_eq!(q, end);
                at = end;
            }
        };
        let release = |p: &mut Pool, model: &mut Model, page: usize, pages: usize| {
            let started = std::time::Instant::now();
            p.reclaim(va_of(page), pages as u64 * PAGE);
            p.mark_prt(va_of(page), pages as u64 * PAGE);
            timed(started);
            for q in page..page + pages {
                model.drop_page(q);
            }
        };

        let mut mappings = 0;
        while mappings < MAPPINGS {
            let r = next();
            let pages = 1 + (r >> 8) as usize % 64;
            let page = (r >> 20) as usize % (PAGES - pages);
            mappings += (page..page + pages).filter(|&q| model.phys[q] == 0 && (q == page || model.phys[q - 1] != 0)).count();
            commit(&mut p, &mut model, page, pages);
        }
        let commands: Vec<u64> = (0..COMMANDS * 2).map(|_| next()).collect();

        spent.set(std::time::Duration::ZERO);
        for pair in commands.chunks(2) {
            let (r, r2) = (pair[0], pair[1]);
            let pages = 1 + (r >> 8) as usize % 64;
            let page = (r >> 20) as usize % (PAGES - pages);
            match r & 15 {
                0..=7 => commit(&mut p, &mut model, page, pages),
                8..=14 => release(&mut p, &mut model, page, pages),
                _ => {
                    // MultiMap: the pages of the source are mapped at the
                    // target as well, whose own backing goes first.
                    let target = (r2 >> 20) as usize % (PAGES - pages);
                    if target < page + pages && page < target + pages {
                        continue;
                    }
                    let started = std::time::Instant::now();
                    let entries = p.entries(va_of(page), pages as u64 * PAGE);
                    p.reclaim(va_of(target), pages as u64 * PAGE);
                    for (a, l, ph) in entries {
                        p.mapped.insert(va_of(target) + (a - va_of(page)), (l, ph));
                    }
                    timed(started);
                    for i in 0..pages {
                        model.drop_page(target + i);
                    }
                    for i in 0..pages {
                        let ph = model.phys[page + i];
                        if ph != 0 {
                            model.refs[ph as usize - 1] += 1;
                            model.phys[target + i] = ph;
                        }
                    }
                }
            }
        }
        let elapsed = spent.get();

        let mut mapped = vec![0u32; PAGES];
        let entries = p.entries(VA_BASE, SPAN);
        for &(a, l, ph) in &entries {
            for i in 0..l / PAGE {
                let page = ((a - VA_BASE) / PAGE + i) as usize;
                assert_eq!(mapped[page], 0, "one mapping per page");
                mapped[page] = (ph / PAGE + i + 1) as u32;
            }
        }
        assert!(mapped == model.phys, "the mappings differ from the page model");
        let mut free = vec![false; POOL_PAGES];
        for (&s, &l) in &p.free {
            for q in s / PAGE..(s + l) / PAGE {
                free[q as usize] = true;
            }
        }
        assert!(free.iter().zip(&model.refs).all(|(&f, &n)| f == (n == 0)), "the free ranges differ from the page model");
        assert!(p.free.iter().zip(p.free.iter().skip(1)).all(|((&s, &l), (&n, _))| s + l < n), "free ranges are merged");
        assert!(p.is_prt(VA_BASE, SPAN));

        eprintln!("pool replay: {COMMANDS} commands over {} mappings in {elapsed:?}", entries.len());
        if !cfg!(debug_assertions) {
            assert!(elapsed < std::time::Duration::from_millis(1500), "{COMMANDS} commands took {elapsed:?}");
        }
    }

    #[test]
    fn protection_normalization() {
        assert_eq!(normalize_protection(PROT_AMPR_READ), PROT_CPU_READ);
        assert_eq!(normalize_protection(PROT_ACP_WRITE | PROT_GPU_READ), PROT_CPU_READ | PROT_CPU_WRITE | PROT_GPU_READ);
        assert_eq!(normalize_protection(0x1000 | PROT_CPU_WRITE), PROT_CPU_WRITE);
    }
}
