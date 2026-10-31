// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Buffer hazards between work recorded without a full barrier.
//!
//! Dispatches are recorded back to back without barriers unless one of
//! them touches what an earlier one did:
//!
//! * read after write and write after write (a dispatch reads or writes a
//!   range an earlier dispatch wrote): a memory dependency from the
//!   compute stage, its writes made visible to compute shaders and
//!   indirect-argument reads;
//! * write after read (a dispatch writes a range earlier work read): an
//!   execution dependency on the stages that read it (compute, graphics,
//!   transfer).
//!
//! Every write tracked here comes from a compute shader: draws that write
//! memory and command-processor transfers are ordered by full barriers on
//! both sides. A write stays tracked per kind of later reader until a
//! barrier makes it visible to that kind: compute shaders (and indirect
//! reads), transfers, or everything (a full barrier). Reads stay tracked
//! until an execution dependency orders a later compute writer after them.
//! Work whose reads have no ranges (device addresses without a planned
//! range, buffers read in guest memory, draws that did not report their
//! ranges) counts as reading everything.
//!
//! Guest dispatches the console may run at the same time (one queue, no
//! partial flush, cache action, wait or command-processor write between
//! them: [`crate::render::DispatchPacket::unsynced`]) form a run: their
//! compute accesses are kept apart ([`Hazards::note_guest_dispatch`]) and
//! order only dispatches after the run, not the run's later members
//! ([`Hazards::guest_dispatch_barrier`]), as on the console, where such
//! dispatches overlap and the title orders dependent ones itself.

use std::collections::BTreeMap;

/// `RPS5E_RANGE_BARRIERS=0`: a dispatch that writes memory after any
/// shader work, or touches a range an earlier dispatch wrote, gets a full
/// barrier (all commands, all memory) before it, and copies of GPU results
/// out to guest memory (write-throughs) are ordered by full barriers.
/// Otherwise only overlapping ranges order them, through barriers from the
/// stages involved.
pub fn range_barriers_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_RANGE_BARRIERS"))
}

/// Draw reads kept unmerged at most (memory bound while no dispatch asks).
const MAX_PENDING_READS: usize = 1 << 16;

/// Disjoint, merged, half-open guest ranges.
#[derive(Clone, Debug, Default)]
pub struct RangeSet {
    /// start -> end.
    spans: BTreeMap<u64, u64>,
    /// Bytes covered.
    bytes: u64,
}

impl RangeSet {
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Number of disjoint spans.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn clear(&mut self) {
        self.spans.clear();
        self.bytes = 0;
    }

    /// Bytes covered by the spans.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Add `[addr, addr + size)` (an empty range adds nothing). Spans that
    /// touch or overlap it merge with it.
    pub fn insert(&mut self, addr: u64, size: u64) {
        if size == 0 {
            return;
        }
        let (mut start, mut end) = (addr, addr.saturating_add(size));
        // Most inserts fall inside a span already present.
        if let Some((_, &e)) = self.spans.range(..=start).next_back() {
            if end <= e {
                return;
            }
        }
        while let Some((&s, &e)) = self.spans.range(..=end).next_back() {
            if e < start {
                break;
            }
            self.spans.remove(&s);
            self.bytes -= e - s;
            start = start.min(s);
            end = end.max(e);
        }
        self.spans.insert(start, end);
        self.bytes += end - start;
    }

    /// Take `[addr, addr + size)` out of the spans.
    pub fn remove(&mut self, addr: u64, size: u64) {
        if size == 0 {
            return;
        }
        let end = addr.saturating_add(size);
        while let Some((&s, &e)) = self.spans.range(..end).next_back() {
            if e <= addr {
                break;
            }
            self.spans.remove(&s);
            self.bytes -= e - s;
            if s < addr {
                self.spans.insert(s, addr);
                self.bytes += addr - s;
            }
            if e > end {
                self.spans.insert(end, e);
                self.bytes += e - end;
            }
        }
    }

    /// The parts of the spans inside `[addr, addr + size)`, in address order.
    pub fn clipped(&self, addr: u64, size: u64) -> impl Iterator<Item = (u64, u64)> + '_ {
        use std::ops::Bound::Excluded;
        let end = addr.saturating_add(size);
        let first = self.spans.range(..=addr).next_back().map(|(&s, &e)| (s, e));
        let rest = (addr < end).then(|| self.spans.range((Excluded(addr), Excluded(end))).map(|(&s, &e)| (s, e)));
        first
            .into_iter()
            .chain(rest.into_iter().flatten())
            .map(move |(s, e)| (s.max(addr), e.min(end)))
            .filter(|&(s, e)| s < e)
            .map(|(s, e)| (s, e - s))
    }

    /// `[addr, addr + size)` overlaps a span (an empty range overlaps
    /// nothing).
    pub fn overlaps(&self, addr: u64, size: u64) -> bool {
        if size == 0 {
            return false;
        }
        let end = addr.saturating_add(size);
        self.spans.range(..end).next_back().is_some_and(|(_, &e)| e > addr)
    }

    /// Add disjoint, non-touching half-open spans `(start, end)` in
    /// address order (as [`Self::insert`] of each, built in one pass when
    /// the set is empty).
    fn insert_sorted(&mut self, spans: &[(u64, u64)]) {
        if self.spans.is_empty() {
            self.spans = spans.iter().copied().collect();
            self.bytes = spans.iter().map(|&(s, e)| e - s).sum();
        } else {
            for &(s, e) in spans {
                self.insert(s, e - s);
            }
        }
    }

    /// One of `ranges` overlaps a span.
    pub fn overlaps_any(&self, ranges: impl IntoIterator<Item = (u64, u64)>) -> bool {
        !self.spans.is_empty() && ranges.into_iter().any(|(a, n)| self.overlaps(a, n))
    }
}

/// The dependency a dispatch needs before it, besides image barriers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchBarrier {
    None,
    /// A barrier whose source is the compute stage (with its shader
    /// writes) and, for write-after-read hazards, the graphics and transfer
    /// stages that read what the dispatch writes; its destination is the
    /// compute stage and indirect-argument reads.
    Compute { graphics: bool, transfer: bool },
}

/// Ranges written by dispatches and read by work recorded since the last
/// full barrier (see the module documentation).
#[derive(Clone, Debug, Default)]
pub struct Hazards {
    /// Written since the last full barrier: not yet visible to graphics,
    /// host or command-processor work.
    pub writes: RangeSet,
    /// Not yet visible to compute shaders and indirect-argument reads.
    compute_writes: RangeSet,
    /// Not yet visible to transfer reads.
    transfer_writes: RangeSet,
    compute_reads: RangeSet,
    compute_reads_unknown: bool,
    /// Compute writes and reads of the current run of guest dispatches
    /// (see the module documentation): they order the dispatches after the
    /// run, not its members.
    run_writes: RangeSet,
    run_reads: RangeSet,
    run_reads_unknown: bool,
    graphics_reads: RangeSet,
    /// Reads of draws recorded since `graphics_reads` was last looked at,
    /// `(start, end)` in recording order (a range touching the previous
    /// one is merged with it): every draw adds its reads, only dispatches
    /// that write ask about them ([`Self::settle_graphics_reads`]).
    graphics_pending: Vec<(u64, u64)>,
    graphics_reads_unknown: bool,
    transfer_reads: RangeSet,
    /// The reads the draw about to begin (or continue) a rendering scope
    /// reported ([`Self::note_draw_reads`]): its ranges, whether it also
    /// reads memory without ranges, and whether it reported at all.
    /// Barriers recorded before the draw itself do not order it.
    draw_reads: Vec<(u64, u64)>,
    draw_unknown: bool,
    draw_noted: bool,
}

impl Hazards {
    /// A full barrier was recorded: every earlier access is ordered before
    /// every later one.
    pub fn full_barrier(&mut self) {
        self.writes.clear();
        self.compute_writes.clear();
        self.transfer_writes.clear();
        self.compute_reads.clear();
        self.compute_reads_unknown = false;
        self.clear_run();
        self.graphics_reads.clear();
        self.graphics_pending.clear();
        self.graphics_reads_unknown = false;
        self.transfer_reads.clear();
    }

    fn clear_run(&mut self) {
        self.run_writes.clear();
        self.run_reads.clear();
        self.run_reads_unknown = false;
    }

    /// The current run of guest dispatches ended: its accesses order every
    /// later dispatch.
    fn end_run(&mut self) {
        if self.run_writes.is_empty() && self.run_reads.is_empty() && !self.run_reads_unknown {
            return;
        }
        for (s, e) in std::mem::take(&mut self.run_writes).spans {
            self.compute_writes.insert(s, e - s);
        }
        for (s, e) in std::mem::take(&mut self.run_reads).spans {
            self.compute_reads.insert(s, e - s);
        }
        self.compute_reads_unknown |= std::mem::take(&mut self.run_reads_unknown);
    }

    /// Merge the pending draw reads into `graphics_reads`.
    fn settle_graphics_reads(&mut self) {
        if self.graphics_pending.is_empty() {
            return;
        }
        let pending = &mut self.graphics_pending;
        pending.sort_unstable();
        // Coalesce in place: overlapping or touching spans merge, as
        // `RangeSet::insert` merges them.
        let mut out = 0;
        for i in 1..pending.len() {
            let (s, e) = pending[i];
            if s <= pending[out].1 {
                pending[out].1 = pending[out].1.max(e);
            } else {
                out += 1;
                pending[out] = (s, e);
            }
        }
        pending.truncate(out + 1);
        self.graphics_reads.insert_sorted(pending);
        pending.clear();
    }

    /// Spans tracked in all sets (diagnostics, bounds).
    pub fn spans(&mut self) -> usize {
        self.settle_graphics_reads();
        self.settled_spans()
    }

    fn settled_spans(&self) -> usize {
        self.writes.len() + self.compute_writes.len() + self.transfer_writes.len() + self.compute_reads.len() + self.run_writes.len() + self.run_reads.len() + self.graphics_reads.len() + self.transfer_reads.len()
    }

    /// [`Self::spans`] `> max`, settling the pending draw reads only when
    /// their count could decide it.
    pub fn spans_exceed(&mut self, max: usize) -> bool {
        if self.settled_spans() + self.graphics_pending.len() <= max {
            return false;
        }
        self.spans() > max
    }

    /// The dependency a dispatch reading `reads` and writing `writes` needs
    /// (`unknown_reads`: it also reads memory it has no ranges for).
    pub fn dispatch_barrier(&mut self, reads: &[(u64, u64)], writes: &[(u64, u64)], unknown_reads: bool) -> DispatchBarrier {
        self.end_run();
        self.ordered_barrier(reads, writes, unknown_reads)
    }

    /// [`Self::dispatch_barrier`] of a guest dispatch; `follow`: it
    /// continues the run of the dispatches noted before it (the console may
    /// run it alongside them), so their accesses do not order it.
    pub fn guest_dispatch_barrier(&mut self, reads: &[(u64, u64)], writes: &[(u64, u64)], unknown_reads: bool, follow: bool) -> DispatchBarrier {
        if !follow {
            self.end_run();
        }
        self.ordered_barrier(reads, writes, unknown_reads)
    }

    /// The dependency on the accesses outside the current run.
    fn ordered_barrier(&mut self, reads: &[(u64, u64)], writes: &[(u64, u64)], unknown_reads: bool) -> DispatchBarrier {
        if !self.graphics_reads_unknown && writes.iter().any(|w| w.1 != 0) {
            self.settle_graphics_reads();
        }
        let raw = (unknown_reads && !self.compute_writes.is_empty()) || self.compute_writes.overlaps_any(reads.iter().chain(writes).copied());
        let writing = writes.iter().any(|w| w.1 != 0);
        let war_compute = writing && (self.compute_reads_unknown || self.compute_reads.overlaps_any(writes.iter().copied()));
        let graphics = writing && (self.graphics_reads_unknown || self.graphics_reads.overlaps_any(writes.iter().copied()));
        let transfer = writing && self.transfer_reads.overlaps_any(writes.iter().copied());
        if raw || war_compute || graphics || transfer {
            DispatchBarrier::Compute { graphics, transfer }
        } else {
            DispatchBarrier::None
        }
    }

    /// Compute writes not yet visible to compute shaders overlap
    /// `[addr, addr + size)`.
    pub fn compute_write_pending(&self, addr: u64, size: u64) -> bool {
        self.compute_writes.overlaps(addr, size) || self.run_writes.overlaps(addr, size)
    }

    /// The barrier [`Self::dispatch_barrier`] returned was recorded.
    pub fn compute_barrier_recorded(&mut self, graphics: bool, transfer: bool) {
        self.compute_writes.clear();
        self.compute_reads.clear();
        self.compute_reads_unknown = false;
        self.clear_run();
        if graphics {
            self.graphics_reads.clear();
            self.graphics_pending.clear();
            self.graphics_reads_unknown = false;
        }
        if transfer {
            self.transfer_reads.clear();
        }
    }

    /// A dispatch reading `reads` and writing `writes` was recorded.
    pub fn note_dispatch(&mut self, reads: &[(u64, u64)], writes: &[(u64, u64)], unknown_reads: bool) {
        self.end_run();
        for &(a, n) in writes {
            self.writes.insert(a, n);
            self.compute_writes.insert(a, n);
            self.transfer_writes.insert(a, n);
        }
        for &(a, n) in reads {
            self.compute_reads.insert(a, n);
        }
        self.compute_reads_unknown |= unknown_reads;
    }

    /// A guest dispatch reading `reads` and writing `writes` was recorded
    /// after [`Self::guest_dispatch_barrier`]: it joins the current run.
    /// Its writes are unordered for graphics, host and transfer work as any
    /// dispatch's.
    pub fn note_guest_dispatch(&mut self, reads: &[(u64, u64)], writes: &[(u64, u64)], unknown_reads: bool) {
        for &(a, n) in writes {
            self.writes.insert(a, n);
            self.run_writes.insert(a, n);
            self.transfer_writes.insert(a, n);
        }
        for &(a, n) in reads {
            self.run_reads.insert(a, n);
        }
        self.run_reads_unknown |= unknown_reads;
    }

    /// The next draw recorded into a rendering scope reads `reads` (and
    /// memory it has no ranges for when `unknown`).
    pub fn note_draw_reads(&mut self, reads: impl IntoIterator<Item = (u64, u64)>, unknown: bool) {
        self.draw_reads.clear();
        self.draw_reads.extend(reads);
        self.draw_unknown = unknown;
        self.draw_noted = true;
    }

    /// A draw is recorded into a rendering scope: it reads what it reported
    /// last, and without reported ranges anything.
    pub fn draw_recorded(&mut self) {
        if !std::mem::take(&mut self.draw_noted) {
            self.graphics_reads_unknown = true;
            return;
        }
        for &(a, n) in &self.draw_reads {
            if n == 0 {
                continue;
            }
            let end = a.saturating_add(n);
            match self.graphics_pending.last_mut() {
                Some(last) if a <= last.1 && last.0 <= end => *last = (last.0.min(a), last.1.max(end)),
                _ => self.graphics_pending.push((a, end)),
            }
        }
        if self.graphics_pending.len() >= MAX_PENDING_READS {
            self.settle_graphics_reads();
        }
        self.graphics_reads_unknown |= self.draw_unknown;
        self.draw_reads.clear();
    }

    /// Transfers reading `ranges` need a memory dependency on the compute
    /// writes there first.
    pub fn transfer_read_barrier(&self, ranges: &[(u64, u64)]) -> bool {
        self.transfer_writes.overlaps_any(ranges.iter().copied())
    }

    /// A barrier from the compute stage's writes to transfer reads was
    /// recorded.
    pub fn transfer_barrier_recorded(&mut self) {
        self.transfer_writes.clear();
    }

    /// Transfers reading `ranges` were recorded.
    pub fn note_transfer_reads(&mut self, ranges: &[(u64, u64)]) {
        for &(a, n) in ranges {
            self.transfer_reads.insert(a, n);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_set_merges_and_answers_overlaps_exactly() {
        let mut s = RangeSet::default();
        s.insert(0x1000, 0x100);
        s.insert(0x3000, 0x100);
        assert_eq!(s.len(), 2);
        assert!(s.overlaps(0x10ff, 1));
        assert!(!s.overlaps(0x1100, 0x1f00), "the gap between the spans");
        assert!(!s.overlaps(0x0, 0x1000), "ends where the first span starts");
        assert!(s.overlaps(0x0, 0x1001));
        assert!(!s.overlaps(0x1050, 0), "an empty range");
        // Touching spans merge; the union stays the same.
        s.insert(0x1100, 0x1f00);
        assert_eq!(s.len(), 1);
        assert!(s.overlaps(0x30ff, 1));
        assert!(!s.overlaps(0x3100, 0x100));
        // A span swallowing several.
        s.insert(0x8000, 0x10);
        s.insert(0x9000, 0x10);
        s.insert(0x7000, 0x3000);
        assert_eq!(s.len(), 2);
        assert!(s.overlaps(0x9fff, 1) && !s.overlaps(0xa000, 1));
        assert!(s.overlaps_any([(0x5000, 0x10), (0x7000, 1)]));
        assert!(!s.overlaps_any([(0x5000, 0x10), (0x6000, 0x1000)]));
        // Ranges at the top of the address space do not wrap.
        s.insert(u64::MAX - 0x10, 0x100);
        assert!(s.overlaps(u64::MAX - 1, 1));
    }

    #[test]
    fn range_set_removes_and_clips() {
        let mut s = RangeSet::default();
        s.insert(0x1000, 0x3000);
        s.insert(0x8000, 0x1000);
        assert_eq!(s.bytes(), 0x4000);
        assert_eq!(s.clipped(0x0, 0x10000).collect::<Vec<_>>(), [(0x1000, 0x3000), (0x8000, 0x1000)]);
        assert_eq!(s.clipped(0x2000, 0x6800).collect::<Vec<_>>(), [(0x2000, 0x2000), (0x8000, 0x800)]);
        assert_eq!(s.clipped(0x1000, 0x1000).collect::<Vec<_>>(), [(0x1000, 0x1000)]);
        assert_eq!(s.clipped(0x4000, 0x4000).count(), 0);
        // A hole in the middle of a span, and a span removed whole.
        s.remove(0x2000, 0x1000);
        s.remove(0x7000, 0x3000);
        assert_eq!(s.clipped(0x0, 0x10000).collect::<Vec<_>>(), [(0x1000, 0x1000), (0x3000, 0x1000)]);
        assert_eq!(s.bytes(), 0x2000);
        assert!(!s.overlaps(0x2000, 0x1000) && s.overlaps(0x3fff, 1));
        s.clear();
        assert_eq!((s.len(), s.bytes()), (0, 0));
    }

    #[test]
    fn range_set_matches_a_linear_scan() {
        // Pseudo-random ranges against the pairwise comparison the
        // scheduler used before.
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut set = RangeSet::default();
        let mut list: Vec<(u64, u64)> = Vec::new();
        for _ in 0..400 {
            let (a, n) = (next() % 0x10_0000, next() % 0x800);
            set.insert(a, n);
            list.push((a, n));
            let (b, m) = (next() % 0x10_0000, next() % 0x800);
            let linear = list.iter().any(|&(a, n)| n != 0 && m != 0 && b < a + n && a < b + m);
            assert_eq!(set.overlaps(b, m), linear, "{b:#x}+{m:#x}");
        }
    }

    #[test]
    fn independent_dispatches_need_no_barrier() {
        let mut h = Hazards::default();
        h.note_dispatch(&[(0x1000, 0x100), (0x10_0000, 0x40)], &[(0x1000, 0x100)], false);
        // Another output, reading the same constants.
        assert_eq!(h.dispatch_barrier(&[(0x2000, 0x100), (0x10_0000, 0x40)], &[(0x2000, 0x100)], false), DispatchBarrier::None);
    }

    #[test]
    fn read_after_write_and_write_after_write_need_a_compute_barrier() {
        let mut h = Hazards::default();
        h.note_dispatch(&[(0x1000, 0x100)], &[(0x1000, 0x100)], false);
        let raw = h.dispatch_barrier(&[(0x10f0, 0x10), (0x5000, 0x100)], &[(0x5000, 0x100)], false);
        assert_eq!(raw, DispatchBarrier::Compute { graphics: false, transfer: false });
        let waw = h.dispatch_barrier(&[(0x1080, 0x10)], &[(0x1080, 0x10)], false);
        assert_eq!(waw, DispatchBarrier::Compute { graphics: false, transfer: false });
        // A read of memory without ranges waits for every write.
        assert_ne!(h.dispatch_barrier(&[], &[], true), DispatchBarrier::None);
        h.compute_barrier_recorded(false, false);
        assert_eq!(h.dispatch_barrier(&[(0x1000, 0x100)], &[], true), DispatchBarrier::None, "the writes are visible to compute now");
        // Still unordered for graphics and transfers.
        assert!(h.writes.overlaps(0x1000, 1));
        assert!(h.transfer_read_barrier(&[(0x1000, 4)]));
    }

    #[test]
    fn write_after_read_waits_for_the_readers_stages() {
        let mut h = Hazards::default();
        h.note_dispatch(&[(0x1000, 0x100)], &[], false);
        assert_eq!(h.dispatch_barrier(&[], &[(0x2000, 0x100)], false), DispatchBarrier::None, "writes elsewhere");
        assert_eq!(h.dispatch_barrier(&[], &[(0x1040, 0x10)], false), DispatchBarrier::Compute { graphics: false, transfer: false });
        h.compute_barrier_recorded(false, false);
        assert_eq!(h.dispatch_barrier(&[], &[(0x1040, 0x10)], false), DispatchBarrier::None);
        // Reads of draws.
        h.note_draw_reads([(0x3000, 0x100)], false);
        h.draw_recorded();
        assert_eq!(h.dispatch_barrier(&[], &[(0x3000, 4)], false), DispatchBarrier::Compute { graphics: true, transfer: false });
        assert_eq!(h.dispatch_barrier(&[], &[(0x4000, 4)], false), DispatchBarrier::None);
        // A barrier recorded between the report and the draw does not
        // order the draw.
        h.note_draw_reads([(0x5000, 0x100)], false);
        h.full_barrier();
        h.draw_recorded();
        assert_eq!(h.dispatch_barrier(&[], &[(0x5000, 4)], false), DispatchBarrier::Compute { graphics: true, transfer: false });
        h.full_barrier();
        // A draw without reported ranges reads everything.
        h.draw_recorded();
        assert_eq!(h.dispatch_barrier(&[], &[(0x4000, 4)], false), DispatchBarrier::Compute { graphics: true, transfer: false });
        h.compute_barrier_recorded(true, false);
        assert_eq!(h.dispatch_barrier(&[], &[(0x4000, 4)], false), DispatchBarrier::None);
        // Reads of transfers.
        h.note_transfer_reads(&[(0x6000, 0x1000)]);
        assert_eq!(h.dispatch_barrier(&[(0x6000, 4)], &[], false), DispatchBarrier::None, "read after read");
        assert_eq!(h.dispatch_barrier(&[], &[(0x6ffc, 4)], false), DispatchBarrier::Compute { graphics: false, transfer: true });
        // Unknown reads of a dispatch: every later writer waits.
        h.note_dispatch(&[], &[], true);
        assert_ne!(h.dispatch_barrier(&[], &[(0x9_0000, 4)], false), DispatchBarrier::None);
        h.full_barrier();
        assert_eq!(h.dispatch_barrier(&[], &[(0x6ffc, 4)], false), DispatchBarrier::None);
        assert_eq!(h.spans(), 0);
    }

    /// Guest dispatches with nothing between them on the console overlap:
    /// a run member is not ordered after the earlier members, the first
    /// dispatch after the run is ordered after all of them.
    #[test]
    fn guest_dispatch_runs_order_only_the_dispatches_after_them() {
        let mut h = Hazards::default();
        // Writes before the run order its members.
        h.note_dispatch(&[], &[(0x9000, 0x100)], false);
        assert_ne!(h.guest_dispatch_barrier(&[(0x9000, 4)], &[], false, false), DispatchBarrier::None);
        h.compute_barrier_recorded(false, false);
        // A run writing one range (write after write) and reading it.
        assert_eq!(h.guest_dispatch_barrier(&[(0x1000, 0x100)], &[(0x1000, 0x100)], false, false), DispatchBarrier::None);
        h.note_guest_dispatch(&[(0x1000, 0x100)], &[(0x1000, 0x100)], false);
        for _ in 0..3 {
            assert_eq!(h.guest_dispatch_barrier(&[(0x1000, 0x100)], &[(0x1000, 0x100)], true, true), DispatchBarrier::None, "a member of the run");
            h.note_guest_dispatch(&[(0x1000, 0x100)], &[(0x1000, 0x100)], true);
        }
        assert!(h.compute_write_pending(0x10f0, 4) && !h.compute_write_pending(0x1100, 4));
        // Unordered for transfers and graphics as any compute write.
        assert!(h.transfer_read_barrier(&[(0x1000, 4)]));
        assert!(h.writes.overlaps(0x1000, 4));
        // The dispatch after the run (a synchronisation point between).
        let after = h.guest_dispatch_barrier(&[(0x1000, 4)], &[], false, false);
        assert_eq!(after, DispatchBarrier::Compute { graphics: false, transfer: false });
        h.compute_barrier_recorded(false, false);
        assert!(!h.compute_write_pending(0x1000, 0x100));
        // Write after read of a run member, seen by a dispatch outside the
        // run, and by an ordinary dispatch.
        h.note_guest_dispatch(&[(0x4000, 0x10)], &[], false);
        assert_eq!(h.guest_dispatch_barrier(&[], &[(0x4000, 4)], false, true), DispatchBarrier::None);
        assert_eq!(h.dispatch_barrier(&[], &[(0x4000, 4)], false), DispatchBarrier::Compute { graphics: false, transfer: false });
        // A full barrier ends everything.
        h.note_guest_dispatch(&[], &[(0x5000, 0x10)], true);
        h.full_barrier();
        assert_eq!(h.guest_dispatch_barrier(&[(0x5000, 4)], &[], true, false), DispatchBarrier::None);
        assert_eq!(h.spans(), 0);
    }

    #[test]
    fn transfer_reads_wait_for_compute_writes_once() {
        let mut h = Hazards::default();
        h.note_dispatch(&[], &[(0x1000, 0x100)], false);
        assert!(!h.transfer_read_barrier(&[(0x2000, 0x100)]));
        assert!(h.transfer_read_barrier(&[(0x1000, 0x1000)]));
        h.transfer_barrier_recorded();
        assert!(!h.transfer_read_barrier(&[(0x1000, 0x1000)]));
        // Graphics still waits for a full barrier.
        assert!(h.writes.overlaps(0x1000, 4));
    }

    /// The tracker as it was before draw reads were merged lazily: every
    /// draw's reads go into the set when it is recorded.
    #[derive(Default)]
    struct Eager {
        writes: RangeSet,
        compute_writes: RangeSet,
        transfer_writes: RangeSet,
        compute_reads: RangeSet,
        compute_reads_unknown: bool,
        graphics_reads: RangeSet,
        graphics_reads_unknown: bool,
        transfer_reads: RangeSet,
        draw_reads: Vec<(u64, u64)>,
        draw_unknown: bool,
        draw_noted: bool,
    }

    impl Eager {
        fn full_barrier(&mut self) {
            self.writes.clear();
            self.compute_writes.clear();
            self.transfer_writes.clear();
            self.compute_reads.clear();
            self.compute_reads_unknown = false;
            self.graphics_reads.clear();
            self.graphics_reads_unknown = false;
            self.transfer_reads.clear();
        }

        fn spans(&self) -> usize {
            self.writes.len() + self.compute_writes.len() + self.transfer_writes.len() + self.compute_reads.len() + self.graphics_reads.len() + self.transfer_reads.len()
        }

        fn dispatch_barrier(&self, reads: &[(u64, u64)], writes: &[(u64, u64)], unknown_reads: bool) -> DispatchBarrier {
            let raw = (unknown_reads && !self.compute_writes.is_empty()) || self.compute_writes.overlaps_any(reads.iter().chain(writes).copied());
            let writing = writes.iter().any(|w| w.1 != 0);
            let war_compute = writing && (self.compute_reads_unknown || self.compute_reads.overlaps_any(writes.iter().copied()));
            let graphics = writing && (self.graphics_reads_unknown || self.graphics_reads.overlaps_any(writes.iter().copied()));
            let transfer = writing && self.transfer_reads.overlaps_any(writes.iter().copied());
            if raw || war_compute || graphics || transfer {
                DispatchBarrier::Compute { graphics, transfer }
            } else {
                DispatchBarrier::None
            }
        }

        fn compute_barrier_recorded(&mut self, graphics: bool, transfer: bool) {
            self.compute_writes.clear();
            self.compute_reads.clear();
            self.compute_reads_unknown = false;
            if graphics {
                self.graphics_reads.clear();
                self.graphics_reads_unknown = false;
            }
            if transfer {
                self.transfer_reads.clear();
            }
        }

        fn note_dispatch(&mut self, reads: &[(u64, u64)], writes: &[(u64, u64)], unknown_reads: bool) {
            for &(a, n) in writes {
                self.writes.insert(a, n);
                self.compute_writes.insert(a, n);
                self.transfer_writes.insert(a, n);
            }
            for &(a, n) in reads {
                self.compute_reads.insert(a, n);
            }
            self.compute_reads_unknown |= unknown_reads;
        }

        fn draw_recorded(&mut self) {
            if !std::mem::take(&mut self.draw_noted) {
                self.graphics_reads_unknown = true;
                return;
            }
            for &(a, n) in &self.draw_reads {
                self.graphics_reads.insert(a, n);
            }
            self.graphics_reads_unknown |= self.draw_unknown;
            self.draw_reads.clear();
        }
    }

    #[test]
    fn lazy_draw_reads_decide_as_eager_inserts() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for round in 0..64u64 {
            let mut lazy = Hazards::default();
            let mut eager = Eager::default();
            // Narrow address windows make overlaps and touching spans common.
            let window = 0x400u64 << (round % 8);
            let range = |next: &mut dyn FnMut() -> u64| {
                let a = (next() % window) & !3;
                let n = match next() % 8 {
                    0 => 0,
                    1 => 4,
                    _ => (next() % 0x80 + 1) * 4,
                };
                (a, n)
            };
            for _ in 0..3000 {
                match next() % 16 {
                    0..=6 => {
                        let reads: Vec<_> = (0..next() % 6).map(|_| range(&mut next)).collect();
                        let unknown = next() % 50 == 0;
                        lazy.note_draw_reads(reads.iter().copied(), unknown);
                        eager.draw_reads = reads;
                        eager.draw_unknown = unknown;
                        eager.draw_noted = true;
                        if next() % 40 == 0 {
                            // A barrier between the report and the draw.
                            lazy.full_barrier();
                            eager.full_barrier();
                        }
                        lazy.draw_recorded();
                        eager.draw_recorded();
                    }
                    7 => {
                        // A draw that reported nothing.
                        if next() % 4 == 0 {
                            lazy.draw_recorded();
                            eager.draw_recorded();
                        }
                    }
                    8..=12 => {
                        let reads: Vec<_> = (0..next() % 3).map(|_| range(&mut next)).collect();
                        let writes: Vec<_> = (0..next() % 3).map(|_| range(&mut next)).collect();
                        let unknown = next() % 20 == 0;
                        let b = lazy.dispatch_barrier(&reads, &writes, unknown);
                        assert_eq!(b, eager.dispatch_barrier(&reads, &writes, unknown), "round {round}");
                        if let DispatchBarrier::Compute { graphics, transfer } = b {
                            lazy.compute_barrier_recorded(graphics, transfer);
                            eager.compute_barrier_recorded(graphics, transfer);
                        }
                        lazy.note_dispatch(&reads, &writes, unknown);
                        eager.note_dispatch(&reads, &writes, unknown);
                        let max = next() as usize % 64;
                        assert_eq!(lazy.spans_exceed(max), eager.spans() > max, "round {round}");
                    }
                    13 => {
                        let ranges: Vec<_> = (0..next() % 3 + 1).map(|_| range(&mut next)).collect();
                        assert_eq!(lazy.transfer_read_barrier(&ranges), eager.transfer_writes.overlaps_any(ranges.iter().copied()));
                        lazy.note_transfer_reads(&ranges);
                        for &(a, n) in &ranges {
                            eager.transfer_reads.insert(a, n);
                        }
                    }
                    14 => {
                        if next() % 4 == 0 {
                            lazy.full_barrier();
                            eager.full_barrier();
                        } else {
                            assert_eq!(lazy.spans(), eager.spans(), "round {round}");
                        }
                    }
                    _ => {
                        let (a, n) = range(&mut next);
                        let write = [(a, n.max(4))];
                        assert_eq!(lazy.dispatch_barrier(&[], &write, false), eager.dispatch_barrier(&[], &write, false), "round {round}");
                    }
                }
            }
            assert_eq!(lazy.spans(), eager.spans(), "round {round}");
        }
    }
}
