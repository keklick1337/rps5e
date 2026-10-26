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
//! AllocatePaForPrt backs from the pool; unbacked PRT pages stay unmapped
//! (no zero page is mapped for GPU reads of them).

use super::command_buffer::{AmmKind, AmmMap, AMM_PAGE_SIZE};
use crate::memory;
use crate::registry::HleResult;
use crate::sync::{ERR_INVALID_ARGUMENT, ERR_TRY_AGAIN, OK};
use std::collections::{BTreeMap, BTreeSet};
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

#[derive(Default)]
struct Pool {
    /// Free automatic direct memory: start -> length (coalesced).
    free: BTreeMap<u64, u64>,
    /// Automatic mappings and their aliases: va -> (length, physical start).
    mapped: BTreeMap<u64, (u64, u64)>,
    /// Pages of partially resident (PRT) ranges.
    prt: BTreeSet<u64>,
}

static POOL: Mutex<Option<Pool>> = Mutex::new(None);

fn with_pool<R>(f: impl FnOnce(&mut Pool) -> R) -> R {
    let mut g: MutexGuard<'_, Option<Pool>> = POOL.lock().unwrap_or_else(|p| p.into_inner());
    f(g.get_or_insert_with(Pool::default))
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
    let overlapping: Vec<(u64, u64)> = free
        .range(..end)
        .filter(|(&s, &l)| s + l > start)
        .map(|(&s, &l)| (s, l))
        .collect();
    for (s, l) in overlapping {
        free.remove(&s);
        if s < start {
            free.insert(s, start - s);
        }
        if s + l > end {
            free.insert(end, s + l - end);
        }
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
    let Some(ranges) = ranges else { return ERR_TRY_AGAIN };
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
                p.mapped.retain(|&a, _| !(va..va + mapped).contains(&a));
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
        self.mapped
            .range(..end)
            .filter(|(&a, &(l, _))| a + l > va)
            .map(|(&a, &(l, ph))| {
                let first = a.max(va);
                let last = (a + l).min(end);
                (first, last - first, ph + (first - a))
            })
            .collect()
    }

    /// Forget the automatic mappings of `[va, va + size)` without freeing
    /// their backing; returns them clipped as [`Pool::entries`].
    fn detach(&mut self, va: u64, size: u64) -> Vec<(u64, u64, u64)> {
        let end = va + size;
        let clipped = self.entries(va, size);
        let hits: Vec<(u64, u64, u64)> =
            self.mapped.range(..end).filter(|(&a, &(l, _))| a + l > va).map(|(&a, &(l, ph))| (a, l, ph)).collect();
        for (a, l, ph) in hits {
            self.mapped.remove(&a);
            let first = a.max(va);
            let last = (a + l).min(end);
            if a < first {
                self.mapped.insert(a, (first - a, ph));
            }
            if a + l > last {
                self.mapped.insert(last, (a + l - last, ph + (last - a)));
            }
        }
        clipped
    }

    /// The parts of physical `[ph, ph + len)` no automatic mapping (or
    /// alias) still uses.
    fn unreferenced(&self, ph: u64, len: u64) -> Vec<(u64, u64)> {
        let mut parts = vec![(ph, ph + len)];
        for &(l, p) in self.mapped.values() {
            let (s, e) = (p, p + l);
            parts = parts
                .into_iter()
                .flat_map(|(a, b)| {
                    if e <= a || s >= b {
                        vec![(a, b)]
                    } else {
                        [(a, s.max(a)), (e.min(b), b)].into_iter().filter(|(x, y)| x < y).collect()
                    }
                })
                .collect();
        }
        parts.into_iter().map(|(a, b)| (a, b - a)).collect()
    }

    /// Return the automatic backing of `[va, va + size)` to the free ranges
    /// (pages an alias still maps stay taken).
    fn reclaim(&mut self, va: u64, size: u64) {
        for (_, l, ph) in self.detach(va, size) {
            for (s, len) in self.unreferenced(ph, l) {
                add_free(&mut self.free, s, len);
            }
        }
    }

    fn mark_prt(&mut self, va: u64, size: u64) {
        self.prt.extend((va..va + size).step_by(AMM_PAGE_SIZE as usize));
    }

    fn clear_prt(&mut self, va: u64, size: u64) {
        let pages: Vec<u64> = self.prt.range(va..va + size).copied().collect();
        for p in pages {
            self.prt.remove(&p);
        }
    }

    fn is_prt(&self, va: u64, size: u64) -> bool {
        self.prt.range(va..va + size).count() as u64 == size / AMM_PAGE_SIZE
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
fn in_amm_range(va: u64, size: u64) -> bool {
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

/// MapAsPrt: the range loses its backing and becomes partially resident;
/// its unbacked pages stay unmapped.
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
        let Some(ranges) = with_pool(|p| p.take(len)) else { return ERR_TRY_AGAIN };
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
        AmmKind::MapAsPrt => return map_as_prt(m.va, m.size),
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

    #[test]
    fn protection_normalization() {
        assert_eq!(normalize_protection(PROT_AMPR_READ), PROT_CPU_READ);
        assert_eq!(normalize_protection(PROT_ACP_WRITE | PROT_GPU_READ), PROT_CPU_READ | PROT_CPU_WRITE | PROT_GPU_READ);
        assert_eq!(normalize_protection(0x1000 | PROT_CPU_WRITE), PROT_CPU_WRITE);
    }
}
