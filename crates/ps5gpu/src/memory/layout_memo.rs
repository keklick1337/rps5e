// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Per-thread memo of the guest memory layout of ranges: the committed
//! pieces [`GuestGpuMemory::is_watchable`](super::GuestGpuMemory::is_watchable)
//! and [`GuestGpuMemory::committed_extent`](super::GuestGpuMemory::committed_extent)
//! are answered from.
//!
//! Both walk the span map piece by piece. Buffers of partially resident
//! pools cross dozens of spans, and the recording thread asks about ranges
//! inside the same pools for every draw that binds them (each draw at
//! another offset of a ring, most up to the pool's end), while the layout
//! changes only when the guest maps or unmaps memory. The pieces of a range
//! walked once are kept with the stamp they were found at (the instance's
//! layout counter and the guest range registry's generation); a later
//! range inside it takes its pieces from there while both stay.
//! `RPS5E_LAYOUT_MEMO=0` turns the memo off; A/B scope `release` toggles it.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Ranges remembered per thread (most recently used first).
const EXTENTS: usize = 16;

/// A committed piece of a range: `(start, len, aliased, registered)`, where
/// `registered` says every page of the piece is registered.
pub(super) type Piece = (u64, u64, bool, bool);

struct Extent {
    id: u64,
    stamp: (u64, u64),
    start: u64,
    end: u64,
    pieces: Vec<Piece>,
}

thread_local! {
    static TABLE: RefCell<Vec<Extent>> = const { RefCell::new(Vec::new()) };
}

/// Answers served from the memo and walks made (diagnostics).
pub static HITS: AtomicU64 = AtomicU64::new(0);
pub static MISSES: AtomicU64 = AtomicU64::new(0);

/// A new instance number (never 0).
pub(super) fn instance_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// The memo is on now (`RPS5E_LAYOUT_MEMO`, A/B scope `release`).
#[inline]
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_LAYOUT_MEMO")) && crate::backend::ab::release_fast("RPS5E_LAYOUT_MEMO")
}

/// `f` over the pieces of `[addr, addr+size)` (clipped to it) when a
/// remembered range of instance `id` at `stamp` holds it.
pub(super) fn with_pieces<R>(id: u64, stamp: (u64, u64), addr: u64, size: u64, f: impl FnOnce(&mut dyn Iterator<Item = Piece>) -> R) -> Option<R> {
    let end = addr.checked_add(size)?;
    let r = TABLE
        .try_with(|t| {
            let mut t = t.borrow_mut();
            let i = t.iter().position(|e| e.id == id && e.stamp == stamp && e.start <= addr && end <= e.end)?;
            if i != 0 {
                let e = t.remove(i);
                t.insert(0, e);
            }
            let e = &t[0];
            // Pieces are in address order: start at the first one that
            // ends after `addr`.
            let first = e.pieces.partition_point(|p| p.0 + p.1 <= addr);
            let mut it = e.pieces[first..].iter().take_while(|p| p.0 < end).map(|&(s, n, a, r)| {
                let (cs, ce) = (s.max(addr), (s + n).min(end));
                (cs, ce - cs, a, r)
            });
            Some(f(&mut it))
        })
        .ok()
        .flatten();
    if r.is_some() {
        HITS.fetch_add(1, Ordering::Relaxed);
    } else {
        MISSES.fetch_add(1, Ordering::Relaxed);
    }
    r
}

/// Alignment of the window a walk covers.
const WINDOW: u64 = 32 << 20;

/// The range a walk for `[start, end)` covers: the range widened to
/// [`WINDOW`] alignment (within `space`), unless that more than doubles it
/// for a large range.
pub(super) fn window(start: u64, end: u64, space: u64) -> (u64, u64) {
    let (s, e) = (start & !(WINDOW - 1), end.div_ceil(WINDOW).saturating_mul(WINDOW).min(space));
    if e - s > 2 * (end - start) + 2 * WINDOW || e <= s {
        (start, end)
    } else {
        (s.max(1), e)
    }
}

/// Remember the pieces of `[start, end)` of instance `id`, found at `stamp`.
pub(super) fn insert(id: u64, stamp: (u64, u64), start: u64, end: u64, pieces: Vec<Piece>) {
    let _ = TABLE.try_with(|t| {
        let mut t = t.borrow_mut();
        // Stale ranges of this instance go first; then the least recently used.
        t.retain(|e| e.id != id || e.stamp == stamp);
        if t.len() >= EXTENTS {
            t.pop();
        }
        t.insert(0, Extent { id, stamp, start, end, pieces });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_inside_a_remembered_one_take_its_clipped_pieces() {
        let (a, b) = (instance_id(), instance_id());
        let pieces = vec![(0x1000, 0x1000, true, true), (0x3000, 0x2000, true, false), (0x6000, 0x1000, false, false)];
        insert(a, (1, 2), 0x1000, 0x8000, pieces);
        let got = |id, stamp, addr, size| with_pieces(id, stamp, addr, size, |it| it.collect::<Vec<_>>());
        assert_eq!(got(a, (1, 2), 0x1800, 0x2000), Some(vec![(0x1800, 0x800, true, true), (0x3000, 0x800, true, false)]));
        assert_eq!(got(a, (1, 2), 0x2000, 0x1000), Some(vec![]), "a hole");
        assert_eq!(got(a, (1, 2), 0x5800, 0x2800), Some(vec![(0x6000, 0x1000, false, false)]));
        assert_eq!(got(a, (1, 2), 0x800, 0x1000), None, "starts before");
        assert_eq!(got(a, (1, 2), 0x7000, 0x2000), None, "ends after");
        assert_eq!(got(b, (1, 2), 0x1800, 0x100), None, "another instance");
        assert_eq!(got(a, (2, 2), 0x1800, 0x100), None, "the layout moved");
        assert_eq!(got(a, (1, 3), 0x1800, 0x100), None, "the range registry moved");
    }
}
