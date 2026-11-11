// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Two recording lanes by guest command queue, predicted from a frame's
//! logged items (measurement only, part of `RPS5E_SLICE_SHADOW=1`).
//!
//! The items of queue 0 (the graphics queue) form one lane, the items of
//! every other queue the second. Each lane records its items in their
//! logged order; an item also starts only after the latest earlier item of
//! the other lane it depends on has finished:
//! - it reads a range, an image or word pages that item changed;
//! - it changes a range or an image that item read or changed;
//! - one of the two reads or writes memory it has no ranges for and the
//!   other writes or accesses anything.
//!
//! A flip joins the lanes. The frame's predicted recording time is the
//! later of the two lanes' ends; the waits are counted per cause. Three
//! predictions are made: with the word pages stage preparations read (the
//! pages are what their validity is checked on, so a write to another word
//! of a page counts); without them (ranges and images only); and without
//! the accesses that have no ranges either: those leave nothing in the
//! caches when recorded, so they order the lanes only on the GPU, which
//! submitting the lanes' work in item order keeps.

use super::{Cut, Frame, ItemRec, Kind, UNKNOWN_READS, UNKNOWN_WRITES};
use crate::fxhash::HashMap;
use std::collections::BTreeMap;

/// What made an item wait for the other lane.
pub const CAUSES: usize = 6;
pub const CAUSE_NAMES: [&str; CAUSES] = ["read-of-change", "change-of-access", "image", "words", "unknown", "flip"];
const READ: usize = 0;
const CHANGE: usize = 1;
const IMAGE: usize = 2;
const WORDS: usize = 3;
const UNKNOWN: usize = 4;
const FLIP: usize = 5;

/// The predictions: everything; without word pages; without word pages
/// and accesses without ranges.
const MODES: usize = 3;
const MODE_NAMES: [&str; MODES] = ["two-lane", "without-words", "ranges-only"];

/// Sums over frames.
#[derive(Clone, Debug, Default)]
pub struct Lanes {
    /// Items and their recording time per lane (graphics, others).
    pub items: [u64; 2],
    pub ns: [u64; 2],
    /// Items and time per guest queue (queues above 7 count as 7).
    pub queue_items: [u64; 8],
    pub queue_ns: [u64; 8],
    /// Changes of lane along the item order.
    pub switches: u64,
    /// The predicted recording time per mode.
    pub two_ns: [u64; MODES],
    /// Items that waited for the other lane and how long, per mode and
    /// lane.
    pub waits: [[u64; 2]; MODES],
    pub wait_ns: [[u64; 2]; MODES],
    /// Wait time per mode and cause.
    pub cause_ns: [[u64; CAUSES]; MODES],
}

impl Lanes {
    pub const ZERO: Lanes =
        Lanes { items: [0; 2], ns: [0; 2], queue_items: [0; 8], queue_ns: [0; 8], switches: 0, two_ns: [0; MODES], waits: [[0; 2]; MODES], wait_ns: [[0; 2]; MODES], cause_ns: [[0; CAUSES]; MODES] };

    pub fn add(&mut self, o: &Lanes) {
        fn add_all(a: &mut [u64], b: &[u64]) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
        add_all(&mut self.items, &o.items);
        add_all(&mut self.ns, &o.ns);
        add_all(&mut self.queue_items, &o.queue_items);
        add_all(&mut self.queue_ns, &o.queue_ns);
        self.switches += o.switches;
        add_all(&mut self.two_ns, &o.two_ns);
        for m in 0..MODES {
            add_all(&mut self.waits[m], &o.waits[m]);
            add_all(&mut self.wait_ns[m], &o.wait_ns[m]);
            add_all(&mut self.cause_ns[m], &o.cause_ns[m]);
        }
    }

    /// Per-frame means over `frames`.
    pub fn line(&self, frames: u64) -> String {
        let fr = frames.max(1) as f64;
        let ms = |ns: u64| ns as f64 / fr / 1e6;
        let queues: Vec<String> = (0..8).filter(|&q| self.queue_items[q] != 0).map(|q| format!("q{q}={:.0}/{:.1}ms", self.queue_items[q] as f64 / fr, ms(self.queue_ns[q]))).collect();
        let mode = |m: usize| {
            let causes: Vec<String> = (0..CAUSES).filter(|&c| self.cause_ns[m][c] != 0).map(|c| format!("{}={:.1}", CAUSE_NAMES[c], ms(self.cause_ns[m][c]))).collect();
            format!(
                "{:.1}ms (waits g {:.0}/{:.1}ms c {:.0}/{:.1}ms: {})",
                ms(self.two_ns[m]),
                self.waits[m][0] as f64 / fr,
                ms(self.wait_ns[m][0]),
                self.waits[m][1] as f64 / fr,
                ms(self.wait_ns[m][1]),
                causes.join(" ")
            )
        };
        format!(
            "lanes: graphics {:.0}/{:.1}ms others {:.0}/{:.1}ms switches {:.1} [{}] {}",
            self.items[0] as f64 / fr,
            ms(self.ns[0]),
            self.items[1] as f64 / fr,
            ms(self.ns[1]),
            self.switches as f64 / fr,
            queues.join(" "),
            (0..MODES).map(|m| format!("{} {}", MODE_NAMES[m], mode(m))).collect::<Vec<_>>().join(" ")
        )
    }
}

/// When the latest item touching each part of an address range finished.
/// The times inserted never decrease, so a new range replaces what it
/// covers.
#[derive(Default)]
struct TimeMap(BTreeMap<u64, (u64, u64)>);

impl TimeMap {
    /// The first key whose range may reach into `[a, ..)`.
    fn first(&self, a: u64) -> u64 {
        self.0.range(..=a).next_back().filter(|(_, &(e, _))| e > a).map_or(a, |(&k, _)| k)
    }

    fn insert(&mut self, (a, b): (u64, u64), t: u64) {
        if a >= b {
            return;
        }
        let covered: smallvec::SmallVec<[(u64, u64, u64); 8]> = self.0.range(self.first(a)..b).map(|(&k, &(e, t))| (k, e, t)).collect();
        for (k, e, old) in covered {
            self.0.remove(&k);
            if k < a {
                self.0.insert(k, (a, old));
            }
            if e > b {
                self.0.insert(b, (e, old));
            }
        }
        self.0.insert(a, (b, t));
    }

    /// The latest time over `[a, b)` (0: untouched).
    fn latest(&self, (a, b): (u64, u64)) -> u64 {
        if a >= b {
            return 0;
        }
        self.0.range(self.first(a)..b).map(|(_, &(_, t))| t).max().unwrap_or(0)
    }
}

/// What one lane did so far in the frame, by finish time.
#[derive(Default)]
struct LaneState {
    reads: TimeMap,
    changes: TimeMap,
    image_reads: HashMap<u64, u64>,
    image_changes: HashMap<u64, u64>,
    /// The last item at all, the last that changed anything, the last
    /// with unknown reads and with unknown writes.
    any: u64,
    any_change: u64,
    unknown_reads: u64,
    unknown_writes: u64,
}

/// The lane of an item: the graphics queue, or the others.
fn lane(it: &ItemRec) -> usize {
    (it.queue != 0) as usize
}

/// `log`: the waits of 0.2 ms and more, as text.
fn simulate(f: &Frame, mode: usize, out: &mut Lanes, mut log: Option<&mut Vec<String>>) {
    let words = mode == 0;
    let unknown = mode != 2;
    let mut free = [0u64; 2];
    let mut state = [LaneState::default(), LaneState::default()];
    for (index, it) in f.items.iter().enumerate() {
        if it.kind.cut() == Cut::Memory {
            continue;
        }
        let l = lane(it);
        let o = 1 - l;
        if it.kind == Kind::Flip {
            let start = free[0].max(free[1]);
            if start > free[l] {
                out.waits[mode][l] += 1;
                out.wait_ns[mode][l] += start - free[l];
                out.cause_ns[mode][FLIP] += start - free[l];
            }
            free = [start + it.ns; 2];
            continue;
        }
        let other = &state[o];
        let mut dep = (0u64, READ, (0u64, 0u64));
        let mut need = |t: u64, cause: usize, range: (u64, u64)| {
            if t > dep.0 {
                dep = (t, cause, range);
            }
        };
        let word_pages: &[(u64, u64)] = if words { f.words(it) } else { &[] };
        let reads = || f.reads(it).iter().chain(f.soft_reads(it)).copied();
        for r in reads() {
            need(other.changes.latest(r), READ, r);
        }
        for &r in word_pages {
            need(other.changes.latest(r), WORDS, r);
        }
        for &w in f.changes(it) {
            need(other.changes.latest(w).max(other.reads.latest(w)), CHANGE, w);
        }
        for &id in f.read_ids(it) {
            need(other.image_changes.get(&id).copied().unwrap_or(0), IMAGE, (id, id));
        }
        for &id in f.changed_ids(it) {
            need(other.image_changes.get(&id).copied().unwrap_or(0).max(other.image_reads.get(&id).copied().unwrap_or(0)), IMAGE, (id, id));
        }
        let reads_any = reads().next().is_some() || !word_pages.is_empty() || !f.read_ids(it).is_empty();
        let changes_any = !f.changes(it).is_empty() || !f.changed_ids(it).is_empty();
        if unknown {
            if reads_any || changes_any {
                need(other.unknown_writes, UNKNOWN, (0, 0));
            }
            if changes_any {
                need(other.unknown_reads, UNKNOWN, (0, 0));
            }
            if it.unknown & UNKNOWN_READS != 0 {
                need(other.any_change, UNKNOWN, (0, 0));
            }
            if it.unknown & UNKNOWN_WRITES != 0 {
                need(other.any, UNKNOWN, (0, 0));
            }
        }
        if dep.0 > free[l] {
            let wait = dep.0 - free[l];
            out.waits[mode][l] += 1;
            out.wait_ns[mode][l] += wait;
            out.cause_ns[mode][dep.1] += wait;
            if let Some(log) = log.as_deref_mut().filter(|_| wait >= 200_000) {
                log.push(format!("#{index} {:?} q{} waits {:.1} ms: {} {:#x}..{:#x}", it.kind, it.queue, wait as f64 / 1e6, CAUSE_NAMES[dep.1], dep.2 .0, dep.2 .1));
            }
        }
        let end = free[l].max(dep.0) + it.ns;
        free[l] = end;
        let mine = &mut state[l];
        for r in reads() {
            mine.reads.insert(r, end);
        }
        for &r in word_pages {
            mine.reads.insert(r, end);
        }
        for &w in f.changes(it) {
            mine.changes.insert(w, end);
        }
        for &id in f.read_ids(it) {
            mine.image_reads.insert(id, end);
        }
        for &id in f.changed_ids(it) {
            mine.image_changes.insert(id, end);
        }
        mine.any = end;
        if changes_any || it.unknown & UNKNOWN_WRITES != 0 {
            mine.any_change = end;
        }
        if it.unknown & UNKNOWN_READS != 0 {
            mine.unknown_reads = end;
        }
        if it.unknown & UNKNOWN_WRITES != 0 {
            mine.unknown_writes = end;
        }
    }
    out.two_ns[mode] += free[0].max(free[1]);
}

/// Predict one frame.
pub fn analyze(f: &Frame) -> Lanes {
    let mut out = Lanes::default();
    let mut last = None;
    for it in f.items.iter().filter(|it| it.kind.cut() != Cut::Memory) {
        let l = lane(it);
        out.items[l] += 1;
        out.ns[l] += it.ns;
        let q = (it.queue as usize).min(7);
        out.queue_items[q] += 1;
        out.queue_ns[q] += it.ns;
        out.switches += last.is_some_and(|p| p != l) as u64;
        last = Some(l);
    }
    static FRAMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let logged = block_log() && FRAMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 64 == 0;
    let mut waits = Vec::new();
    for mode in 0..MODES {
        simulate(f, mode, &mut out, (logged && mode == MODES - 1).then_some(&mut waits));
    }
    if logged {
        let ms = |ns: u64| ns as f64 / 1e6;
        log::info!(
            "[lanes] frame blocks: {} -> two-lane {:.1} ms, without words {:.1} ms, ranges only {:.1} ms; its waits: {}",
            blocks(f),
            ms(out.two_ns[0]),
            ms(out.two_ns[1]),
            ms(out.two_ns[2]),
            waits.join("; ")
        );
    }
    out
}

/// `RPS5E_SLICE_SHADOW=2`: every 64th frame's runs of items of one queue
/// are logged.
fn block_log() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_SLICE_SHADOW").is_ok_and(|v| v == "2"))
}

/// The frame as runs of items of one queue: `q<queue> <items>/<ms>` with
/// `r<n>` / `w<n>` items reading / writing memory without ranges; runs
/// shorter than 0.2 ms are summed as `+<runs>/<ms>`.
fn blocks(f: &Frame) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut small = (0u32, 0u64);
    let mut run: Option<(u8, u32, u64, u32, u32)> = None;
    let close = |run: Option<(u8, u32, u64, u32, u32)>, small: &mut (u32, u64), out: &mut Vec<String>| {
        let Some((q, n, ns, r, w)) = run else { return };
        if ns < 200_000 {
            small.0 += 1;
            small.1 += ns;
            return;
        }
        if small.0 != 0 {
            out.push(format!("+{}/{:.1}", small.0, small.1 as f64 / 1e6));
            *small = (0, 0);
        }
        out.push(format!("q{q} {n}/{:.1}{}{}", ns as f64 / 1e6, if r != 0 { format!(" r{r}") } else { String::new() }, if w != 0 { format!(" w{w}") } else { String::new() }));
    };
    for it in f.items.iter().filter(|it| it.kind.cut() != Cut::Memory) {
        if run.is_some_and(|r| r.0 != it.queue) {
            close(run.take(), &mut small, &mut out);
        }
        let r = run.get_or_insert((it.queue, 0, 0, 0, 0));
        r.1 += 1;
        r.2 += it.ns;
        r.3 += (it.unknown & UNKNOWN_READS != 0) as u32;
        r.4 += (it.unknown & UNKNOWN_WRITES != 0) as u32;
    }
    close(run, &mut small, &mut out);
    if small.0 != 0 {
        out.push(format!("+{}/{:.1}", small.0, small.1 as f64 / 1e6));
    }
    out.join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: u64 = 1_000_000;

    fn item(f: &mut Frame, kind: Kind, queue: u8, ns: u64, reads: &[(u64, u64)], changes: &[(u64, u64)], unknown: u8, words: &[(u64, u64)]) {
        f.push(kind, 0, ns, 0, 0, reads, changes, &[], &[]);
        f.annotate_last(queue, unknown, words);
    }

    #[test]
    fn time_maps_keep_the_latest_time_of_every_part() {
        let mut m = TimeMap::default();
        m.insert((0x1000, 0x3000), 5);
        m.insert((0x2000, 0x2800), 9);
        assert_eq!(m.latest((0x1000, 0x2000)), 5);
        assert_eq!(m.latest((0x27ff, 0x2800)), 9);
        assert_eq!(m.latest((0x2800, 0x3000)), 5);
        assert_eq!(m.latest((0x0, 0x1000)), 0);
        assert_eq!(m.latest((0x3000, 0x4000)), 0);
        assert_eq!(m.latest((0x0, 0x10000)), 9);
        m.insert((0x0, 0x10000), 12);
        assert_eq!(m.0.len(), 1);
    }

    #[test]
    fn independent_lanes_run_side_by_side_and_dependencies_wait() {
        // Compute queue: three dispatches of 2 ms writing A, B, C. The
        // graphics queue: draws of 1 ms that read nothing of them.
        let mut f = Frame::default();
        for i in 0..3u64 {
            item(&mut f, Kind::Dispatch, 1, 2 * MS, &[], &[(0x10000 * (i + 1), 0x10000 * (i + 1) + 0x100)], 0, &[]);
        }
        for _ in 0..4 {
            item(&mut f, Kind::Draw, 0, MS, &[(0x90_0000, 0x90_0100)], &[], 0, &[]);
        }
        let t = analyze(&f);
        assert_eq!(t.items, [4, 3]);
        assert_eq!(t.ns, [4 * MS, 6 * MS]);
        assert_eq!(t.switches, 1);
        assert_eq!(t.two_ns, [6 * MS; MODES]);
        assert_eq!(t.waits, [[0; 2]; MODES]);

        // A draw reading what the last dispatch wrote starts after it.
        item(&mut f, Kind::Draw, 0, MS, &[(0x30000, 0x30010)], &[], 0, &[]);
        let t = analyze(&f);
        assert_eq!(t.two_ns, [7 * MS; MODES]);
        assert_eq!(t.waits[0], [1, 0]);
        assert_eq!(t.wait_ns[0], [2 * MS, 0]);
        assert_eq!(t.cause_ns[0][READ], 2 * MS);
    }

    #[test]
    fn word_pages_unknown_accesses_and_flips_order_the_lanes() {
        let page = (0x40_0000, 0x40_4000);
        // A dispatch writes a few bytes; a draw's stage reads words of the
        // same page: a dependency only for the prediction with word pages.
        let mut f = Frame::default();
        item(&mut f, Kind::Dispatch, 9, 3 * MS, &[], &[(0x40_0100, 0x40_0200)], 0, &[]);
        item(&mut f, Kind::Draw, 0, MS, &[], &[], 0, &[page]);
        let t = analyze(&f);
        assert_eq!(t.two_ns, [4 * MS, 3 * MS, 3 * MS]);
        assert_eq!(t.cause_ns[0][WORDS], 3 * MS);
        assert_eq!(t.cause_ns[1], [0; CAUSES]);

        // A dispatch writing memory without ranges: every draw after it
        // that reads anything waits for it; one that reads nothing does not.
        let mut f = Frame::default();
        item(&mut f, Kind::Dispatch, 1, 3 * MS, &[], &[], UNKNOWN_WRITES, &[]);
        item(&mut f, Kind::EndOfPipe, 0, MS, &[], &[], 0, &[]);
        item(&mut f, Kind::Draw, 0, MS, &[(0x1000, 0x2000)], &[], 0, &[]);
        let t = analyze(&f);
        // Without ranges it changes no recorded state: the prediction that
        // keeps only the submission order does not wait.
        assert_eq!(t.two_ns, [4 * MS, 4 * MS, 3 * MS]);
        assert_eq!(t.cause_ns[1][UNKNOWN], 2 * MS);
        assert_eq!(t.cause_ns[2], [0; CAUSES]);

        // A flip waits for both lanes.
        let mut f = Frame::default();
        item(&mut f, Kind::Dispatch, 1, 5 * MS, &[], &[], 0, &[]);
        item(&mut f, Kind::Draw, 0, MS, &[], &[], 0, &[]);
        item(&mut f, Kind::Flip, 0, MS, &[], &[], 0, &[]);
        let t = analyze(&f);
        assert_eq!(t.two_ns, [6 * MS; MODES]);
        assert_eq!(t.cause_ns[0][FLIP], 4 * MS);
    }
}
