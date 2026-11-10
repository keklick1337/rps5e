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

fn writers_enabled() -> bool {
    crate::pipeline::frontend::word_speculation() != 0
}

impl WrittenWords {
    /// Diagnostics: the newest recorded GPU write covering `addr`: its
    /// range and the work that recorded it.
    pub fn writer_of(&self, addr: u64) -> Option<(u64, u64, u64)> {
        self.writers.range(..=addr).rev().take(64).find(|(_, &(e, _))| e > addr).map(|(&s, &(e, w))| (s, e, w))
    }

    /// The newest tick of the GPU writes recorded over `[addr, addr+size)`
    /// when every byte is covered by a record (pending or landed); None
    /// when a byte has none (never written by GPU work as far as the record
    /// knows, or its record was dropped).
    pub fn settled(&self, addr: u64, size: u64) -> Option<u64> {
        let end = addr + size;
        if self.lost_all || self.lost.range(..end).next_back().is_some_and(|(_, &le)| le > addr) {
            return None;
        }
        let mut cur = addr;
        let mut tick = 0;
        for (&s, &(e, st)) in self.overlapping(addr, end) {
            if s > cur {
                return None;
            }
            tick = tick.max(match st {
                State::Pending(t) | State::Landed(t) => t,
            });
            cur = e;
            if cur >= end {
                return Some(tick);
            }
        }
        None
    }
}

/// The tick of the newest draw or dispatch recorded that stores through
/// device addresses without an evaluated range (its writes reach no write
/// record; diagnostics, see `SharedWrittenWords::writes_complete`).
pub static UNRANGED_WRITE_TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Note a draw or dispatch recorded in `tick` that stores through device
/// addresses without an evaluated range.
pub fn note_unranged_write(tick: u64) {
    UNRANGED_WRITE_TICK.fetch_max(tick, std::sync::atomic::Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// Written by GPU work recorded in this tick, no write-through
    /// recorded yet.
    Pending(u64),
    /// In guest memory once this tick completed.
    Landed(u64),
}

/// `RPS5E_WRITTEN_REPEAT_SKIP=0`: a GPU write recorded again over exactly
/// the range and state already recorded (a run of dispatches writing one
/// buffer in one tick) cuts and re-inserts it as any other write (A/B
/// release name `written-repeat-skip`).
fn repeat_skip_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_WRITTEN_REPEAT_SKIP")) && crate::backend::ab::release_fast("written-repeat-skip")
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
    /// Diagnostics (`RPS5E_WORD_SPECULATION`): the work that recorded each
    /// GPU write (start -> (end, kind << 56 | shader address)).
    writers: BTreeMap<u64, (u64, u64)>,
    /// Ranges of records dropped when the record was full (start -> end,
    /// merged, never overlapping): GPU work may have written there.
    lost: BTreeMap<u64, u64>,
    /// Too much was dropped to keep: no byte is known unwritten.
    lost_all: bool,
}

impl WrittenWords {
    /// Ranges kept at most; a full record is cleared.
    const MAX: usize = 16384;
    /// Dropped ranges kept at most (then nothing is known unwritten).
    const LOST_MAX: usize = 65536;

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
        // The same range in the same state is already there (a run of
        // dispatches writing one buffer in one tick): ranges never overlap,
        // so the cut and insert below would put back exactly this entry.
        if repeat_skip_enabled() && self.map.len() < Self::MAX && self.map.get(&start) == Some(&(end, state)) {
            return;
        }
        self.insert_always(start, end, state);
    }

    fn insert_always(&mut self, start: u64, end: u64, state: State) {
        if self.map.len() >= Self::MAX {
            let dropped = std::mem::take(&mut self.map);
            for (s, (e, _)) in dropped {
                self.lose(s, e);
            }
        }
        self.cut(start, end);
        self.map.insert(start, (end, state));
    }

    /// GPU work recorded now (in `tick`) writes `[addr, addr+size)`.
    pub fn written(&mut self, addr: u64, size: u64, tick: u64) {
        if writers_enabled() {
            if self.writers.len() > 1 << 16 {
                self.writers.clear();
            }
            self.writers.insert(addr, (addr + size, crate::backend::profile::work()));
        }
        self.insert(addr, addr + size, State::Pending(tick));
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
            if matches!(v.1, State::Pending(_)) {
                v.1 = State::Landed(tick);
            }
        }
        self.insert(addr.max(page_start), (addr + size).min(page_end), State::Landed(tick));
    }

    /// Split the pending range strictly containing `at` into its parts
    /// before and after `at`.
    fn split_pending_at(&mut self, at: u64) {
        if let Some((&s, &(e, st))) = self.map.range(..at).next_back() {
            if e > at && matches!(st, State::Pending(_)) {
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
            .filter(|(_, &(_, st))| matches!(st, State::Pending(_)))
            .map(|(&s, &(e, _))| (s.max(page_start), e.min(page_end)))
            .collect();
        for (s, e) in pending {
            self.insert(s, e, State::Landed(tick));
        }
        self.insert(addr.max(page_start), (addr + size).min(page_end), State::Landed(tick));
    }

    /// Remember that GPU work may have written `[start, end)` although its
    /// record is gone.
    fn lose(&mut self, start: u64, end: u64) {
        if self.lost_all || start >= end {
            return;
        }
        let (mut s, mut e) = (start, end);
        // Merge with every lost range touching it.
        let touching: smallvec::SmallVec<[u64; 8]> = self.lost.range(..=end).rev().take_while(|(_, &le)| le >= start).map(|(&ls, _)| ls).collect();
        for k in touching {
            let le = self.lost.remove(&k).expect("listed range");
            s = s.min(k);
            e = e.max(le);
        }
        self.lost.insert(s, e);
        if self.lost.len() > Self::LOST_MAX {
            self.lost.clear();
            self.lost_all = true;
        }
    }

    /// The guest pages `[addr, addr+size)` hold no GPU data any more (they
    /// were downloaded or unmapped).
    pub fn forget(&mut self, addr: u64, size: u64) {
        self.cut(addr, addr + size);
        let end = addr + size;
        let keys: smallvec::SmallVec<[u64; 8]> = self.lost.range(..end).rev().take_while(|(_, &le)| le > addr).map(|(&ls, _)| ls).collect();
        for k in keys {
            let le = self.lost.remove(&k).expect("listed range");
            if k < addr {
                self.lost.insert(k, addr);
            }
            if le > end {
                self.lost.insert(end, le);
            }
        }
    }

    /// What guest memory holds of `[addr, addr+size)`: no GPU work wrote
    /// any byte of it (`Some(0)`), or the bytes GPU work wrote all landed
    /// with write-throughs, the newest at the returned tick (`Some(tick)`);
    /// None when a byte is pending or its record was dropped.
    pub fn in_guest_memory_after(&self, addr: u64, size: u64) -> Option<u64> {
        let end = addr + size;
        if self.lost_all || self.lost.range(..end).next_back().is_some_and(|(_, &le)| le > addr) {
            return None;
        }
        let mut tick = 0;
        for (_, &(_, st)) in self.overlapping(addr, end) {
            match st {
                State::Pending(_) => return None,
                State::Landed(t) => tick = tick.max(t),
            }
        }
        Some(tick)
    }

    /// The records of `[start, end)` (one page) for many questions without
    /// the lock: see [`PageWords`].
    pub fn page_words(&self, start: u64, end: u64) -> PageWords {
        let lost = self.lost_all || self.lost.range(..end).next_back().is_some_and(|(_, &le)| le > start);
        let ranges = if lost {
            smallvec::SmallVec::new()
        } else {
            self.overlapping(start, end)
                .map(|(&s, &(e, st))| (s, e, match st {
                    State::Pending(_) => None,
                    State::Landed(t) => Some(t),
                }))
                .collect()
        };
        PageWords { start, end, lost, ranges }
    }

    /// Diagnostics: the recorded ranges overlapping `[start, end)`.
    pub fn describe(&self, start: u64, end: u64) -> Vec<String> {
        self.overlapping(start, end)
            .take(24)
            .map(|(&s, &(e, st))| match st {
                State::Pending(t) => format!("{s:#x}+{:#x}:P{t}", e - s),
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
                State::Pending(_) => return None,
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

/// A copy of the write records of one page (see
/// [`WrittenWords::page_words`]).
#[derive(Clone)]
pub struct PageWords {
    pub start: u64,
    pub end: u64,
    lost: bool,
    /// (start, end, landing tick or None while pending).
    ranges: smallvec::SmallVec<[(u64, u64, Option<u64>); 8]>,
}

impl PageWords {
    /// [`WrittenWords::in_guest_memory_after`] for a range inside the page.
    pub fn in_guest_memory_after(&self, addr: u64, size: u64) -> Option<u64> {
        let end = addr + size;
        if self.lost || addr < self.start || end > self.end {
            return None;
        }
        let mut tick = 0;
        for &(s, e, st) in &self.ranges {
            if s < end && e > addr {
                tick = tick.max(st?);
            }
        }
        Some(tick)
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
        w.written(0x1800, 0x39300, 1);
        w.written_through(0x1000, 0x3b000, 0x1800, 0x39300, 10);
        assert_eq!(w.landed(0x1850, 4), Some(10));
        w.written(0x1900, 0xc0, 1);
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
        w.written(0x0, 0x10000, 1);
        assert_eq!(w.landed(0x2050, 4), None);
        w.written(0x2400, 0x10, 1);
        w.written_through(0x2000, 0x3000, 0x2400, 0x10, 7);
        assert_eq!(w.landed(0x2050, 4), Some(7));
        assert_eq!(w.landed(0x3050, 4), None, "another page is still pending");
        assert_eq!(w.landed(0x10000, 4), None, "not written");
    }

    /// Every byte covered by GPU write records: the newest writer's tick
    /// (pending or landed); a byte without a record: none.
    #[test]
    fn settled_ranges_report_their_newest_writer() {
        let mut w = WrittenWords::default();
        w.written(0x1000, 0x100, 4);
        w.written(0x1080, 0x10, 7);
        assert_eq!(w.settled(0x1000, 4), Some(4));
        assert_eq!(w.settled(0x1078, 0x10), Some(7));
        w.written_through(0x0, 0x4000, 0x1000, 4, 9);
        assert_eq!(w.settled(0x1000, 4), Some(9), "landed by a later write-through");
        assert_eq!(w.settled(0x10fc, 8), None, "a byte past the records");
    }

    /// A newer write over a landed range makes it pending again; a
    /// download or unmap forgets the range.
    #[test]
    fn rewrites_and_downloads() {
        let mut w = WrittenWords::default();
        w.written(0x0, 0x100, 1);
        w.written_through(0x0, 0x1000, 0x0, 0x100, 3);
        w.written(0x40, 0x8, 1);
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
                        a.written(addr, size, 1);
                        b.written(addr, size, 1);
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

    /// A word next to a written range on the same page is known unwritten;
    /// ranges dropped from a full record are not, until they are forgotten.
    #[test]
    fn unwritten_words_stay_known_across_a_full_record() {
        let mut w = WrittenWords::default();
        w.written(0x15c0, 0xb8, 1);
        assert!(w.in_guest_memory_after(0x15c0, 4) != Some(0));
        assert!(w.in_guest_memory_after(0x1674, 4) != Some(0), "last word of the range");
        assert!(w.in_guest_memory_after(0x1690, 4) == Some(0), "after the range on its page");
        assert!(w.in_guest_memory_after(0x1000, 4) == Some(0), "before it");
        w.written_through(0x1000, 0x2000, 0x15c0, 0xb8, 5);
        assert!(w.in_guest_memory_after(0x15c0, 4) != Some(0), "landed is still written");
        for i in 0..WrittenWords::MAX as u64 {
            w.written(0x100000 + i * 16, 8, 1);
        }
        assert!(w.in_guest_memory_after(0x15c0, 4) != Some(0), "dropped from the full record");
        assert!(w.in_guest_memory_after(0x1690, 4) == Some(0));
        assert!(w.in_guest_memory_after(0x100000 + 5 * 16, 4) != Some(0));
        assert!(w.in_guest_memory_after(0x100000 + 5 * 16 + 8, 4) == Some(0), "a gap between dropped ranges");
        w.forget(0x1000, 0x1000);
        assert!(w.in_guest_memory_after(0x15c0, 4) == Some(0), "downloaded and forgotten");
    }

    /// Guest memory holds a word once nothing pending covers it: at once
    /// when unwritten, after the newest landing tick otherwise.
    #[test]
    fn guest_memory_holds_unwritten_and_landed_words() {
        let mut w = WrittenWords::default();
        assert_eq!(w.in_guest_memory_after(0x1000, 4), Some(0));
        w.written(0x1000, 0x10, 1);
        assert_eq!(w.in_guest_memory_after(0x1000, 4), None, "pending");
        assert_eq!(w.in_guest_memory_after(0x1010, 4), Some(0), "next to it");
        w.written_through(0x1000, 0x2000, 0x1000, 0x10, 9);
        assert_eq!(w.in_guest_memory_after(0x100c, 8), Some(9), "half landed, half unwritten");
        let page = w.page_words(0x1000, 0x2000);
        assert_eq!(page.in_guest_memory_after(0x100c, 8), Some(9), "the page copy agrees");
        assert_eq!(page.in_guest_memory_after(0x1f00, 4), Some(0));
        assert_eq!(page.in_guest_memory_after(0x2000, 4), None, "outside the page");
        w.written(0x1004, 4, 1);
        assert_eq!(w.in_guest_memory_after(0x1000, 0x10), None, "rewritten");
        assert_eq!(w.page_words(0x1000, 0x2000).in_guest_memory_after(0x1004, 4), None);
    }

    /// Writes repeated over exactly a recorded range (skipped) leave the
    /// same record as cutting and inserting every one, among random other
    /// writes, write-throughs and forgets.
    #[test]
    fn repeated_writes_leave_the_record_unchanged() {
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut skipped = 0;
        for _ in 0..50 {
            let (mut a, mut b) = (WrittenWords::default(), WrittenWords::default());
            let mut last = (0x1000u64, 0x100u64);
            for tick in 0..400u64 {
                let r = next();
                let addr = (r >> 8) % 0x20000 & !3;
                let size = 4 + (r >> 40) % 0x3000 & !3;
                let t = tick / 8;
                match r % 7 {
                    0..=2 => {
                        // The previous write again (a run of dispatches).
                        skipped += (a.map.get(&last.0) == Some(&(last.0 + last.1, State::Pending(t)))) as usize;
                        a.insert(last.0, last.0 + last.1, State::Pending(t));
                        b.insert_always(last.0, last.0 + last.1, State::Pending(t));
                    }
                    3 => {
                        last = (addr, size);
                        a.insert(addr, addr + size, State::Pending(t));
                        b.insert_always(addr, addr + size, State::Pending(t));
                    }
                    4 | 5 => {
                        let (ps, pe) = (addr & !0xFFF, (addr + size + 0xFFF) & !0xFFF);
                        a.written_through(ps, pe, addr, size, tick);
                        b.written_through(ps, pe, addr, size, tick);
                    }
                    _ => {
                        a.forget(addr & !0xFFF, 0x1000);
                        b.forget(addr & !0xFFF, 0x1000);
                    }
                }
                assert_eq!(a.map, b.map);
            }
        }
        assert!(skipped > 1000, "repeats skipped: {skipped}");
    }

    #[test]
    fn a_full_record_is_cleared() {
        let mut w = WrittenWords::default();
        for i in 0..WrittenWords::MAX as u64 {
            w.written(i * 16, 8, 1);
            w.written_through(i * 16 & !0xFFF, (i * 16 & !0xFFF) + 0x1000, i * 16, 8, 1);
        }
        assert!(w.map.len() <= WrittenWords::MAX);
        w.written(1 << 40, 8, 1);
        assert!(w.map.len() < WrittenWords::MAX);
    }
}
