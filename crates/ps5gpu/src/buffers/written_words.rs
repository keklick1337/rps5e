// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! [`WrittenWords`]: the exact byte ranges GPU work wrote into buffers, and
//! the write-through that brings each to guest memory.
//!
//! Write-through spans and GPU-dirty state are kept per page. A small
//! write next to a table (a dispatch writing its output a few bytes after
//! a descriptor another dispatch wrote long ago) gives the whole page a
//! new write-through tick, and a CPU read of the old descriptor would wait
//! for all the work recorded so far. Here every GPU write is recorded with
//! its exact range: pending until a write-through of its page is recorded,
//! then landed by that write-through's tick. A word covered only by landed
//! ranges is in guest memory once the newest of their ticks completed.
//!
//! The record is complete only while nothing is dropped: when it is full
//! it is cleared, and a word no range covers is unknown (the caller falls
//! back to the page's tick).

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// Written, no write-through recorded yet.
    Pending,
    /// In guest memory once this tick completed.
    Landed(u64),
}

/// `RPS5E_WRITTEN_WORDS_IN_PLACE=0`: a write-through re-inserts every
/// pending range it lands instead of updating it in place.
fn in_place_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_WRITTEN_WORDS_IN_PLACE").is_ok_and(|v| v == "0")) && crate::backend::ab::release_fast("written-words-in-place")
}

#[derive(Default)]
pub(super) struct WrittenWords {
    /// start -> (end, state); the ranges never overlap.
    map: BTreeMap<u64, (u64, State)>,
}

impl WrittenWords {
    /// Ranges kept at most; a full record is cleared.
    const MAX: usize = 16384;

    /// Take `[start, end)` out of every range (the parts outside stay).
    fn cut(&mut self, start: u64, end: u64) {
        let keys: smallvec::SmallVec<[u64; 8]> = self.overlapping(start, end).map(|(&k, _)| k).collect();
        for k in keys {
            let (e, st) = self.map.remove(&k).expect("listed range");
            if k < start {
                self.map.insert(k, (start, st));
            }
            if e > end {
                self.map.insert(end, (e, st));
            }
        }
    }

    fn overlapping(&self, start: u64, end: u64) -> impl Iterator<Item = (&u64, &(u64, State))> {
        // A range starting before `start` that reaches into it is the one
        // just before; ranges never overlap.
        let before = self.map.range(..start).next_back().filter(|(_, &(e, _))| e > start);
        before.into_iter().chain(self.map.range(start..end))
    }

    fn insert(&mut self, start: u64, end: u64, state: State) {
        if start >= end {
            return;
        }
        if self.map.len() >= Self::MAX {
            self.map.clear();
        }
        self.cut(start, end);
        self.map.insert(start, (end, state));
    }

    /// GPU work recorded now writes `[addr, addr+size)`.
    pub fn written(&mut self, addr: u64, size: u64) {
        self.insert(addr, addr + size, State::Pending);
    }

    /// A write-through of the pages `[page_start, page_end)` is recorded in
    /// `tick`; `[addr, addr+size)` (inside them) was just written (a
    /// command-processor write, or the range the write-through is for).
    /// Every pending range of the pages lands with it.
    pub fn written_through(&mut self, page_start: u64, page_end: u64, addr: u64, size: u64, tick: u64) {
        if !in_place_enabled() {
            return self.written_through_by_inserts(page_start, page_end, addr, size, tick);
        }
        // Pending ranges reaching over the pages' ends are split there; the
        // pending ranges inside then land in place (no remove and insert).
        self.split_pending_at(page_start);
        self.split_pending_at(page_end);
        for (_, v) in self.map.range_mut(page_start..page_end) {
            if v.1 == State::Pending {
                v.1 = State::Landed(tick);
            }
        }
        self.insert(addr.max(page_start), (addr + size).min(page_end), State::Landed(tick));
    }

    /// Split the pending range strictly containing `at` into its parts
    /// before and after `at`.
    fn split_pending_at(&mut self, at: u64) {
        if let Some((&s, &(e, st))) = self.map.range(..at).next_back() {
            if e > at && st == State::Pending {
                self.map.insert(s, (at, st));
                self.map.insert(at, (e, st));
            }
        }
    }

    /// [`Self::written_through`] by cutting and inserting every pending
    /// range of the pages (`RPS5E_WRITTEN_WORDS_IN_PLACE=0`).
    fn written_through_by_inserts(&mut self, page_start: u64, page_end: u64, addr: u64, size: u64, tick: u64) {
        let pending: Vec<(u64, u64)> = self
            .overlapping(page_start, page_end)
            .filter(|(_, &(_, st))| st == State::Pending)
            .map(|(&s, &(e, _))| (s.max(page_start), e.min(page_end)))
            .collect();
        for (s, e) in pending {
            self.insert(s, e, State::Landed(tick));
        }
        self.insert(addr.max(page_start), (addr + size).min(page_end), State::Landed(tick));
    }

    /// The guest pages `[addr, addr+size)` hold no GPU data any more (they
    /// were downloaded or unmapped).
    pub fn forget(&mut self, addr: u64, size: u64) {
        self.cut(addr, addr + size);
    }

    /// Diagnostics: the recorded ranges overlapping `[start, end)`.
    pub fn describe(&self, start: u64, end: u64) -> Vec<String> {
        self.overlapping(start, end)
            .take(24)
            .map(|(&s, &(e, st))| match st {
                State::Pending => format!("{s:#x}+{:#x}:P", e - s),
                State::Landed(t) => format!("{s:#x}+{:#x}:L{t}", e - s),
            })
            .collect()
    }

    /// The tick after which guest memory holds the GPU data of every byte
    /// of `[addr, addr+size)`: None when a byte is not covered by a landed
    /// range (pending, or not recorded).
    pub fn landed(&self, addr: u64, size: u64) -> Option<u64> {
        let end = addr + size;
        let mut cur = addr;
        let mut tick = 0;
        for (&s, &(e, st)) in self.overlapping(addr, end) {
            if s > cur {
                return None;
            }
            match st {
                State::Pending => return None,
                State::Landed(t) => tick = tick.max(t),
            }
            cur = e;
            if cur >= end {
                return Some(tick);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A descriptor written by a large write that landed long ago stays
    /// landed at that tick when a small write next to it on the same page
    /// gets a new write-through.
    #[test]
    fn a_neighbouring_write_keeps_the_old_landing() {
        let mut w = WrittenWords::default();
        w.written(0x1800, 0x39300);
        w.written_through(0x1000, 0x3b000, 0x1800, 0x39300, 10);
        assert_eq!(w.landed(0x1850, 4), Some(10));
        w.written(0x1900, 0xc0);
        assert_eq!(w.landed(0x1850, 4), Some(10));
        assert_eq!(w.landed(0x1900, 4), None, "pending");
        w.written_through(0x1000, 0x2000, 0x1900, 0xc0, 12);
        assert_eq!(w.landed(0x1850, 4), Some(10));
        assert_eq!(w.landed(0x1904, 4), Some(12));
        assert_eq!(w.landed(0x18fe, 4), Some(12), "a word across both ranges");
    }

    /// A large write without a write-through of its own lands with the
    /// next write-through of its page (which copies the whole page).
    #[test]
    fn a_pending_write_lands_with_the_next_write_through_of_its_page() {
        let mut w = WrittenWords::default();
        w.written(0x0, 0x10000);
        assert_eq!(w.landed(0x2050, 4), None);
        w.written(0x2400, 0x10);
        w.written_through(0x2000, 0x3000, 0x2400, 0x10, 7);
        assert_eq!(w.landed(0x2050, 4), Some(7));
        assert_eq!(w.landed(0x3050, 4), None, "another page is still pending");
        assert_eq!(w.landed(0x10000, 4), None, "not written");
    }

    /// A newer write over a landed range makes it pending again; a
    /// download or unmap forgets the range.
    #[test]
    fn rewrites_and_downloads() {
        let mut w = WrittenWords::default();
        w.written(0x0, 0x100);
        w.written_through(0x0, 0x1000, 0x0, 0x100, 3);
        w.written(0x40, 0x8);
        assert_eq!(w.landed(0x40, 4), None);
        assert_eq!(w.landed(0x80, 4), Some(3));
        w.forget(0x0, 0x1000);
        assert_eq!(w.landed(0x80, 4), None);
    }

    /// Landing in place leaves the same ranges as cutting and inserting
    /// them, over random writes, write-throughs and forgets.
    #[test]
    fn in_place_landing_matches_inserts() {
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..50 {
            let (mut a, mut b) = (WrittenWords::default(), WrittenWords::default());
            for tick in 0..400u64 {
                let r = next();
                let addr = (r >> 8) % 0x20000 & !3;
                let size = 4 + (r >> 40) % 0x3000 & !3;
                match r % 5 {
                    0 | 1 => {
                        a.written(addr, size);
                        b.written(addr, size);
                    }
                    2 | 3 => {
                        let (ps, pe) = (addr & !0xFFF, (addr + size + 0xFFF) & !0xFFF);
                        let ps = ps.saturating_sub((r >> 20) % 3 * 0x1000);
                        a.written_through(ps, pe, addr, size, tick);
                        b.written_through_by_inserts(ps, pe, addr, size, tick);
                    }
                    _ => {
                        a.forget(addr & !0xFFF, 0x1000);
                        b.forget(addr & !0xFFF, 0x1000);
                    }
                }
                assert_eq!(a.map, b.map);
            }
        }
    }

    #[test]
    fn a_full_record_is_cleared() {
        let mut w = WrittenWords::default();
        for i in 0..WrittenWords::MAX as u64 {
            w.written(i * 16, 8);
            w.written_through(i * 16 & !0xFFF, (i * 16 & !0xFFF) + 0x1000, i * 16, 8, 1);
        }
        assert!(w.map.len() <= WrittenWords::MAX);
        w.written(1 << 40, 8);
        assert!(w.map.len() < WrittenWords::MAX);
    }
}
