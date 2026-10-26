// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Shadow analysis of slice recording (R7b stage 0):
//! measurement only, no behaviour change, `RPS5E_SLICE_SHADOW=1`.
//!
//! The serial recorder classifies every back-end item it executes:
//! - a draw is either *speculative* (a slice thread could record it against
//!   frozen committed state) or a *hole* (it creates, uploads or changes
//!   state that must be exact: image creation/upload/refresh, buffer misses
//!   and CPU uploads, written ranges, metadata clears, a full staging ring,
//!   a submit or GPU wait, GDS, occlusion queries);
//! - every other item is a *serial event*: a hard cut (dispatches, `write_memory`,
//!   GDS transfers, occlusion dumps; end-of-pipe and flips also end a wave)
//!   or a positioned event that does not cut (flushes, completions, CP
//!   copies and fills, metadata fills, after-pipe callbacks);
//! - guest memory work done on the same thread in a replay (patches, CPU
//!   faults, mappings) is the game's CPU side: counted apart.
//!
//! Each item carries the thread CPU time it took, the ranges and image ids
//! it read (speculative draws) and the ones whose state it changed. At a
//! flip the frame's items are cut into slices for several slice sizes N
//! (soft cuts every N draws moved to the next render-target change within
//! N/4, hard cuts at serial cut events), the slices are grouped into waves
//! of K, and every wave is checked for read-set conflicts: a speculative
//! draw reading a range or image an earlier item of its wave changed makes
//! its whole slice re-record serially. The frame time is predicted with
//! the design's models (waves: P/K·1.15 + S; rolling: max(P/K, S)) and with
//! a wave schedule simulated from the actual slices.
//!
//! With the switch off every entry point is one relaxed load and a branch.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

/// Slice sizes simulated (`RPS5E_SLICE_DRAWS` candidates).
pub const NS: [usize; 4] = [32, 64, 128, 256];
/// Slice threads per wave simulated.
pub const KS: [usize; 3] = [4, 6, 8];
/// The design's wave imbalance factor.
const IMBALANCE: f64 = 1.15;
/// A frame without a flip is analysed after this many items (live loading
/// screens).
const MAX_ITEMS: usize = 200_000;
/// Frames waiting for analysis before the flip analyses one at once.
const MAX_PENDING: usize = 4;

/// Whether the shadow analysis runs (`RPS5E_SLICE_SHADOW=1`).
#[inline]
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_SLICE_SHADOW").is_ok_and(|v| v != "0" && !v.is_empty()))
}

/// What a draw did that a speculative slice thread could not do against
/// frozen committed state (one bit each).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Hole {
    /// An image created.
    ImageCreate,
    /// An image loaded from guest memory (whole, or CPU pages merged).
    ImageUpload,
    /// An image refilled from newer overlapping images.
    ImageRefresh,
    /// An image discarded for a whole overwrite (clears).
    ImageDiscard,
    /// An image retiled or converted from another shape.
    ImageRetile,
    /// An unwatched image re-armed after its bytes matched.
    ImageRevalidate,
    /// Any lookup generation bump (`changed`): the image's state moved.
    ImageGeneration,
    /// A render target became the newest image of its memory.
    TargetChange,
    /// A colour or depth image cleared outside a pass.
    ImageClear,
    /// A buffer created or merged (find miss).
    BufferMiss,
    /// CPU-dirty pages uploaded into a buffer mirror.
    BufferUpload,
    /// A shader-written buffer range (GPU-dirty, write-through).
    WrittenRange,
    /// A shader-written storage image.
    StorageImage,
    /// A DCC/CMASK or HTILE clear code found at a target bind.
    MetadataClear,
    /// A recorded metadata fill consumed.
    TakeFill,
    /// A metadata clear consumed (0xFF written).
    ConsumeClear,
    /// The staging ring was full (a wait or a submit to recycle it).
    StagingFull,
    /// A submit or a wait for the GPU inside the draw.
    FlushWait,
    /// GPU bytes downloaded to guest memory inside the draw.
    Download,
    /// The draw uses the global data share.
    Gds,
    /// The draw runs inside an occlusion query.
    Occlusion,
    /// A metadata operation, resolve or depth copy draw.
    MetaOp,
}

pub const HOLES: usize = Hole::MetaOp as usize + 1;
pub const HOLE_NAMES: [&str; HOLES] = [
    "create", "upload", "refresh", "discard", "retile", "revalidate", "generation", "target", "clear", "bufmiss", "bufupload", "written", "storage", "metaclear", "takefill",
    "consume", "staging", "flushwait", "download", "gds", "occlusion", "metaop",
];

impl Hole {
    fn bit(self) -> u32 {
        1 << self as u32
    }
}

/// Holes Stage 3 makes speculative (uploads journaled with page generations).
const STAGE3_SPECULATIVE: u32 = 1 << Hole::BufferUpload as u32;

/// A back-end item, as the replay and the executor see it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Draw,
    Dispatch,
    EndOfPipe,
    AfterPipe,
    Flush,
    Completion,
    MemoryFilled,
    CpCopy,
    CpFill,
    CpWrite,
    Gds,
    Occlusion,
    SyncPoint,
    Flip,
    Fault,
    Patch,
    Map,
}

pub const KINDS: usize = Kind::Map as usize + 1;
pub const KIND_NAMES: [&str; KINDS] = ["draw", "dispatch", "eop", "afterpipe", "flush", "completion", "memfilled", "cpcopy", "cpfill", "cpwrite", "gds", "occlusion", "syncpoint", "flip", "fault", "patch", "map"];

/// How an item splits the draw stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cut {
    /// A draw: recorded inside a slice.
    Slice,
    /// Serial, positioned in a slice (no cut).
    Positioned,
    /// Serial, ends the slice.
    Hard,
    /// Serial, ends the slice and the wave (the guard is released).
    Wave,
    /// Guest memory work of the game's CPU threads (no cut, not serial
    /// recording time).
    Memory,
}

impl Kind {
    fn cut(self) -> Cut {
        match self {
            Kind::Draw => Cut::Slice,
            Kind::Dispatch | Kind::CpWrite | Kind::Gds | Kind::Occlusion => Cut::Hard,
            Kind::EndOfPipe | Kind::Flip => Cut::Wave,
            Kind::Fault | Kind::Patch | Kind::Map => Cut::Memory,
            Kind::AfterPipe | Kind::Flush | Kind::Completion | Kind::MemoryFilled | Kind::CpCopy | Kind::CpFill | Kind::SyncPoint => Cut::Positioned,
        }
    }
}

// ---------------------------------------------------------------------
// Per-item bookkeeping on the recording thread
// ---------------------------------------------------------------------

#[derive(Default)]
struct Acc {
    /// Open items on this thread (nested ones add to the outer item).
    depth: u32,
    /// Inside commit-side work (`commit_side`): holes are not noted.
    suppress: u32,
    holes: u32,
    tail_ns: u64,
    reads: Vec<(u64, u64)>,
    soft_reads: Vec<(u64, u64)>,
    changes: Vec<(u64, u64)>,
    read_ids: Vec<u64>,
    changed_ids: Vec<u64>,
}

thread_local! {
    static ACC: RefCell<Acc> = RefCell::new(Acc::default());
}

/// Apply `f` to this thread's open item (nothing outside items; never
/// re-entrant, e.g. from a fault taken while noting).
#[inline]
fn with_open(f: impl FnOnce(&mut Acc)) {
    if !enabled() {
        return;
    }
    ACC.with(|a| {
        if let Ok(mut a) = a.try_borrow_mut() {
            if a.depth > 0 {
                f(&mut a);
            }
        }
    });
}

/// The open draw hit `h`.
#[inline]
pub fn hole(h: Hole) {
    with_open(|a| {
        if a.suppress == 0 {
            a.holes |= h.bit();
        }
    });
}

/// The open item read guest bytes `[addr, addr+size)` through state that
/// must be exact (a buffer mirror, an image).
#[inline]
pub fn read(addr: u64, size: u64) {
    if size != 0 {
        with_open(|a| a.reads.push((addr, addr.saturating_add(size))));
    }
}

/// The open item copied guest bytes `[addr, addr+size)` (a stream copy:
/// a positioned CPU write before it composes into the copy).
#[inline]
pub fn read_streamed(addr: u64, size: u64) {
    if size != 0 {
        with_open(|a| a.soft_reads.push((addr, addr.saturating_add(size))));
    }
}

/// The open item changed the cached state of `[addr, addr+size)`.
#[inline]
pub fn changed(addr: u64, size: u64) {
    if size != 0 {
        with_open(|a| a.changes.push((addr, addr.saturating_add(size))));
    }
}

/// The open item used image `id` covering `range` (`[begin, end)`).
#[inline]
pub fn read_image(id: u64, range: Option<(u64, u64)>) {
    with_open(|a| {
        a.read_ids.push(id);
        if let Some(r) = range.filter(|r| r.1 > r.0) {
            a.reads.push(r);
        }
    });
}

/// The open item changed the state (lookup generation) of image `id`.
#[inline]
pub fn changed_image(id: u64, range: Option<(u64, u64)>) {
    with_open(|a| {
        a.changed_ids.push(id);
        if let Some(r) = range.filter(|r| r.1 > r.0) {
            a.changes.push(r);
        }
    });
}

fn thread_cpu() -> u64 {
    crate::backend::profile::thread_cpu()
}

/// Run commit-side work of the open item (the commit evaluates early
/// submits and collections, not the slice thread): its time is serial and
/// its submits are no holes. Its state changes still count.
pub fn commit_side<R>(f: impl FnOnce() -> R) -> R {
    if !enabled() || !ACC.with(|a| a.try_borrow().is_ok_and(|a| a.depth > 0)) {
        return f();
    }
    with_open(|a| a.suppress += 1);
    let t = thread_cpu();
    let r = f();
    let ns = thread_cpu() - t;
    with_open(|a| {
        a.suppress -= 1;
        a.tail_ns += ns;
    });
    r
}

/// Run item `f` of `kind` (`rt`: the render-target key of a draw) and log
/// it with its thread CPU time. Items nested in an open one are part of it.
pub fn item<R>(kind: Kind, rt: u64, f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let outer = ACC.with(|a| match a.try_borrow_mut() {
        Ok(mut a) => {
            a.depth += 1;
            if a.depth == 1 {
                a.holes = 0;
                a.suppress = 0;
                a.tail_ns = 0;
                a.reads.clear();
                a.soft_reads.clear();
                a.changes.clear();
                a.read_ids.clear();
                a.changed_ids.clear();
                true
            } else {
                false
            }
        }
        Err(_) => false,
    });
    if !outer {
        let r = f();
        ACC.with(|a| {
            if let Ok(mut a) = a.try_borrow_mut() {
                a.depth = a.depth.saturating_sub(1);
            }
        });
        return r;
    }
    let t = thread_cpu();
    let r = f();
    let ns = thread_cpu() - t;
    ACC.with(|a| {
        let Ok(mut a) = a.try_borrow_mut() else { return };
        a.depth = 0;
        let mut log = LOG.lock().unwrap_or_else(|e| e.into_inner());
        log.current.push(kind, a.holes, ns, a.tail_ns, rt, &a.reads, &a.changes, &a.read_ids, &a.changed_ids);
        log.current.push_soft_reads(&a.soft_reads);
        if log.current.items.len() >= MAX_ITEMS {
            log.end_frame();
        }
    });
    r
}

/// The render-target key of a draw: soft cuts move to where it changes.
pub fn target_key(d: &crate::render::ResolvedDraw) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |v: u64| h = (h ^ v).wrapping_mul(0x0000_0100_0000_01b3);
    for t in &d.color_targets {
        mix(t.slot as u64);
        mix(t.base);
    }
    mix(d.depth.map_or(0, |z| z.z_read_base));
    h
}

/// A frame was flipped: its items go to the analysis.
pub fn frame_end() {
    if enabled() {
        LOG.lock().unwrap_or_else(|e| e.into_inner()).end_frame();
    }
}

// ---------------------------------------------------------------------
// Frames and their analysis
// ---------------------------------------------------------------------

/// One logged item.
#[derive(Clone, Copy, Debug)]
pub struct ItemRec {
    pub kind: Kind,
    pub holes: u32,
    /// Thread CPU time of the whole item.
    pub ns: u64,
    /// Of it, commit-side work (serial even for a speculative draw).
    pub tail_ns: u64,
    pub rt: u64,
    reads: (u32, u32),
    soft_reads: (u32, u32),
    changes: (u32, u32),
    read_ids: (u32, u32),
    changed_ids: (u32, u32),
}

/// The items of one frame, ranges and ids in flat arrays.
#[derive(Clone, Debug, Default)]
pub struct Frame {
    pub items: Vec<ItemRec>,
    ranges: Vec<(u64, u64)>,
    ids: Vec<u64>,
}

impl Frame {
    #[allow(clippy::too_many_arguments)]
    pub fn push(&mut self, kind: Kind, holes: u32, ns: u64, tail_ns: u64, rt: u64, reads: &[(u64, u64)], changes: &[(u64, u64)], read_ids: &[u64], changed_ids: &[u64]) {
        let span = |v: &[(u64, u64)], to: &mut Vec<(u64, u64)>| {
            let s = to.len() as u32;
            to.extend_from_slice(v);
            (s, to.len() as u32)
        };
        let reads = span(reads, &mut self.ranges);
        let changes = span(changes, &mut self.ranges);
        let mut ids = |v: &[u64]| {
            let s = self.ids.len() as u32;
            self.ids.extend_from_slice(v);
            (s, self.ids.len() as u32)
        };
        let read_ids = ids(read_ids);
        let changed_ids = ids(changed_ids);
        let at = self.ranges.len() as u32;
        self.items.push(ItemRec { kind, holes, ns, tail_ns, rt, reads, soft_reads: (at, at), changes, read_ids, changed_ids });
    }

    /// The composable (stream-copied) reads of the item pushed last.
    pub fn push_soft_reads(&mut self, v: &[(u64, u64)]) {
        let s = self.ranges.len() as u32;
        self.ranges.extend_from_slice(v);
        let e = self.ranges.len() as u32;
        if let Some(i) = self.items.last_mut() {
            i.soft_reads = (s, e);
        }
    }

    fn reads(&self, i: &ItemRec) -> &[(u64, u64)] {
        &self.ranges[i.reads.0 as usize..i.reads.1 as usize]
    }

    fn soft_reads(&self, i: &ItemRec) -> &[(u64, u64)] {
        &self.ranges[i.soft_reads.0 as usize..i.soft_reads.1 as usize]
    }

    fn changes(&self, i: &ItemRec) -> &[(u64, u64)] {
        &self.ranges[i.changes.0 as usize..i.changes.1 as usize]
    }

    fn read_ids(&self, i: &ItemRec) -> &[u64] {
        &self.ids[i.read_ids.0 as usize..i.read_ids.1 as usize]
    }

    fn changed_ids(&self, i: &ItemRec) -> &[u64] {
        &self.ids[i.changed_ids.0 as usize..i.changed_ids.1 as usize]
    }
}

/// A speculative draw (recorded by a slice thread).
fn speculative(i: &ItemRec) -> bool {
    i.kind == Kind::Draw && i.holes == 0
}

/// A slice: items `[start, end)` from its first draw to its last.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slice {
    pub start: usize,
    pub end: usize,
    pub draws: usize,
}

/// Cut the frame's draws into slices of about `n`: hard cuts at serial cut
/// events, soft cuts every `n` draws moved forward to the next
/// render-target change within `n/4` (else at `n`).
pub fn cut_slices(f: &Frame, n: usize) -> Vec<Slice> {
    let n = n.max(1);
    let mut out = Vec::new();
    let mut seg: Vec<usize> = Vec::new();
    let flush = |seg: &mut Vec<usize>, out: &mut Vec<Slice>| {
        let len = seg.len();
        let mut start = 0;
        while start < len {
            let cut = if len - start <= n {
                len
            } else {
                let nominal = start + n;
                let last = (nominal + n / 4).min(len - 1);
                (nominal..=last).find(|&j| f.items[seg[j]].rt != f.items[seg[j - 1]].rt).unwrap_or(nominal)
            };
            out.push(Slice { start: seg[start], end: seg[cut - 1] + 1, draws: cut - start });
            start = cut;
        }
        seg.clear();
    };
    for (i, it) in f.items.iter().enumerate() {
        match it.kind.cut() {
            Cut::Slice => seg.push(i),
            Cut::Hard | Cut::Wave => flush(&mut seg, &mut out),
            Cut::Positioned | Cut::Memory => {}
        }
    }
    flush(&mut seg, &mut out);
    out
}

/// Waves: up to `k` consecutive slices, ended early by a wave cut
/// (end-of-pipe, flip) between two slices. Returns `(first, last+1)`
/// slice indices.
pub fn waves(f: &Frame, slices: &[Slice], k: usize) -> Vec<(usize, usize)> {
    let k = k.max(1);
    let mut out = Vec::new();
    let mut first = 0;
    for i in 0..slices.len() {
        let broken = i > first && f.items[slices[i - 1].end..slices[i].start].iter().any(|it| it.kind.cut() == Cut::Wave);
        if i - first == k || broken {
            out.push((first, i));
            first = i;
        }
    }
    if first < slices.len() {
        out.push((first, slices.len()));
    }
    out
}

/// Disjoint half-open ranges, each with the sources (bit mask) of the
/// changes merged into it.
#[derive(Default)]
struct RangeSet(BTreeMap<u64, (u64, u64)>);

impl RangeSet {
    fn insert(&mut self, (mut a, mut b): (u64, u64), mut mask: u64) {
        if b <= a {
            return;
        }
        // Merge every range touching [a, b).
        while let Some((&s, &(e, m))) = self.0.range(..=b).next_back() {
            if e < a {
                break;
            }
            self.0.remove(&s);
            a = a.min(s);
            b = b.max(e);
            mask |= m;
        }
        self.0.insert(a, (b, mask));
    }

    /// The sources of the changes overlapping `[a, b)` (0: none).
    fn overlaps(&self, (a, b): (u64, u64)) -> u64 {
        if b <= a {
            return 0;
        }
        self.0.range(..b).next_back().filter(|(_, &(e, _))| e > a).map_or(0, |(_, &(_, m))| m)
    }
}

/// Source bits of a conflict: the hole kinds of a draw (bits 0..), the
/// commit-side work of a speculative draw, serial events by kind (bits 32..).
const SOURCE_TAIL: u32 = 30;
pub const SOURCES: usize = 32 + KINDS;

fn source_mask(it: &ItemRec) -> u64 {
    match it.kind {
        Kind::Draw if it.holes != 0 => it.holes as u64,
        Kind::Draw => 1 << SOURCE_TAIL,
        k => 1 << (32 + k as u32),
    }
}

/// The name of source bit `b`.
pub fn source_name(b: usize) -> &'static str {
    match b {
        b if b < HOLES => HOLE_NAMES[b],
        b if b == SOURCE_TAIL as usize => "commit-side",
        b if (32..32 + KINDS).contains(&b) => KIND_NAMES[b - 32],
        _ => "?",
    }
}

/// Read-set conflicts of one wave layout. A speculative draw reading a
/// range or image an earlier item of its wave changed (holes and serial
/// events, including those between the wave's slices) is:
/// - under the strict check (the design's read set of page and image
///   generations), a conflict: its slice re-records;
/// - under a value check, an *induced hole* when every change it read is
///   work its frozen view still shows pending and would redo itself (an
///   upload, an image creation or refresh, a first target write, a
///   metadata clear): the draw hits the same hole and is recorded exactly
///   at commit. Changes it cannot see coming (GPU writes of dispatches and
///   written ranges, buffer merges, downloads) still re-record the slice.
///
/// Guest memory written by the CPU side (patches, faults, mappings) within
/// the wave is a positioned event instead: a stream copy composes the
/// known bytes (`composed`), an exact read (mirror, image) makes the draw a
/// hole (`pending_holes`); the slice does not re-record.
///
/// Draw counts and times of holes are of draws in slices that do not
/// re-record (a re-recorded slice is all serial anyway).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Conflicts {
    /// Slices re-recorded under the value check.
    pub slices: Vec<bool>,
    /// Slices re-recorded under the strict check.
    pub strict: Vec<bool>,
    pub range_hits: u64,
    pub image_hits: u64,
    /// Induced holes (value check), their time, and the time of those
    /// induced by CPU uploads alone (speculative in Stage 3).
    pub induced: u64,
    pub induced_ns: u64,
    pub induced_upload_ns: u64,
    /// Draws reading CPU-side writes of the wave through exact state, and
    /// their time (value check; `strict_pending_ns` under the strict one).
    pub pending_holes: u64,
    pub pending_ns: u64,
    pub strict_pending_ns: u64,
    /// Draws only copying CPU-side writes of the wave.
    pub composed: u64,
    /// Per slice, the speculative time of its draws turned holes (value
    /// check; with Stage 3's upload-induced ones apart).
    pub moved: Vec<u64>,
    pub moved_upload: Vec<u64>,
    /// Speculative draws hit by a change, by the source of the change.
    pub sources: Vec<u64>,
}

/// Holes whose change a later reader's frozen view shows still pending (it
/// would redo the work and be a hole itself).
const REDONE: u64 = (1 << Hole::ImageCreate as u32)
    | (1 << Hole::ImageUpload as u32)
    | (1 << Hole::ImageRefresh as u32)
    | (1 << Hole::ImageDiscard as u32)
    | (1 << Hole::ImageRetile as u32)
    | (1 << Hole::ImageRevalidate as u32)
    | (1 << Hole::ImageGeneration as u32)
    | (1 << Hole::TargetChange as u32)
    | (1 << Hole::ImageClear as u32)
    | (1 << Hole::BufferUpload as u32)
    | (1 << Hole::MetadataClear as u32)
    | (1 << Hole::TakeFill as u32)
    | (1 << Hole::ConsumeClear as u32);

pub fn conflicts(f: &Frame, slices: &[Slice], waves: &[(usize, usize)]) -> Conflicts {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        Induced { upload_only: bool },
        Pending,
        Hit,
    }
    let n = slices.len();
    let mut c = Conflicts { slices: vec![false; n], strict: vec![false; n], moved: vec![0; n], moved_upload: vec![0; n], sources: vec![0; SOURCES], ..Conflicts::default() };
    // (slice, time, what) of the draws hit, settled once the slices are.
    let mut marks: Vec<(usize, u64, Mark)> = Vec::new();
    let mut prev_end = 0usize;
    for &(w0, w1) in waves {
        if w0 >= w1 {
            continue;
        }
        // The wave's recording starts after the last wave cut before its
        // first slice (state committed until then is what it sees).
        let first = slices[w0].start;
        let from = f.items[prev_end..first].iter().rposition(|it| it.kind.cut() == Cut::Wave).map_or(prev_end, |p| prev_end + p + 1);
        let to = slices[w1 - 1].end;
        prev_end = to;
        let mut ranges = RangeSet::default();
        let mut pending = RangeSet::default();
        let mut ids: crate::fxhash::HashMap<u64, u64> = Default::default();
        let mut s = w0;
        for idx in from..to {
            while s < w1 && slices[s].end <= idx {
                s += 1;
            }
            let it = &f.items[idx];
            if speculative(it) && s < w1 && slices[s].start <= idx {
                let mut hit = 0u64;
                let (mut exact, mut soft) = (false, false);
                for (&r, streamed) in f.reads(it).iter().map(|r| (r, false)).chain(f.soft_reads(it).iter().map(|r| (r, true))) {
                    let m = ranges.overlaps(r);
                    if m != 0 {
                        c.range_hits += 1;
                        hit |= m;
                    } else if pending.overlaps(r) != 0 {
                        if streamed {
                            soft = true;
                        } else {
                            exact = true;
                        }
                    }
                }
                for id in f.read_ids(it) {
                    if let Some(&m) = ids.get(id) {
                        c.image_hits += 1;
                        hit |= m;
                    }
                }
                let ns = parallel_ns(it);
                if hit != 0 {
                    c.strict[s] = true;
                    for (b, n) in c.sources.iter_mut().enumerate() {
                        if hit & 1 << b != 0 {
                            *n += 1;
                        }
                    }
                    if hit & !REDONE == 0 {
                        marks.push((s, ns, Mark::Induced { upload_only: hit == 1 << Hole::BufferUpload as u32 }));
                    } else {
                        c.slices[s] = true;
                        marks.push((s, ns, Mark::Hit));
                    }
                } else if exact {
                    marks.push((s, ns, Mark::Pending));
                } else if soft {
                    c.composed += 1;
                }
            }
            if it.kind.cut() == Cut::Memory {
                for &r in f.changes(it) {
                    pending.insert(r, 1);
                }
            } else {
                let m = source_mask(it);
                for &r in f.changes(it) {
                    ranges.insert(r, m);
                }
                for &id in f.changed_ids(it) {
                    *ids.entry(id).or_default() |= m;
                }
            }
        }
    }
    for (s, ns, mark) in marks {
        match mark {
            Mark::Pending => {
                if !c.strict[s] {
                    c.strict_pending_ns += ns;
                }
                if !c.slices[s] {
                    c.pending_holes += 1;
                    c.pending_ns += ns;
                    c.moved[s] += ns;
                }
            }
            Mark::Induced { upload_only } if !c.slices[s] => {
                c.induced += 1;
                c.induced_ns += ns;
                c.moved[s] += ns;
                if upload_only {
                    c.induced_upload_ns += ns;
                    c.moved_upload[s] += ns;
                }
            }
            _ => {}
        }
    }
    c
}

/// One (N, K) simulation, summed over frames.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sim {
    pub waves: u64,
    /// Value check: re-recorded slices and their speculative time.
    pub conflicts: u64,
    pub conflict_ns: u64,
    /// Strict check (page and image generations).
    pub strict_conflicts: u64,
    pub strict_ns: u64,
    pub range_hits: u64,
    pub image_hits: u64,
    pub induced: u64,
    pub induced_ns: u64,
    pub induced_upload_ns: u64,
    pub pending_holes: u64,
    pub pending_ns: u64,
    pub composed: u64,
    /// Predicted recording time (ns, summed over frames) under the value
    /// check: the design's wave model, its rolling model, the wave schedule
    /// of the actual slices; the strict check's wave and rolling models;
    /// Stage 3 (speculative uploads) wave and rolling models.
    pub waves_ns: u64,
    pub rolling_ns: u64,
    pub schedule_ns: u64,
    pub strict_waves_ns: u64,
    pub strict_rolling_ns: u64,
    pub waves3_ns: u64,
    pub rolling3_ns: u64,
    pub schedule3_ns: u64,
}

impl Sim {
    const ZERO: Sim = Sim {
        waves: 0,
        conflicts: 0,
        conflict_ns: 0,
        strict_conflicts: 0,
        strict_ns: 0,
        range_hits: 0,
        image_hits: 0,
        induced: 0,
        induced_ns: 0,
        induced_upload_ns: 0,
        pending_holes: 0,
        pending_ns: 0,
        composed: 0,
        waves_ns: 0,
        rolling_ns: 0,
        schedule_ns: 0,
        strict_waves_ns: 0,
        strict_rolling_ns: 0,
        waves3_ns: 0,
        rolling3_ns: 0,
        schedule3_ns: 0,
    };

    fn fields(&mut self) -> [&mut u64; 21] {
        [
            &mut self.waves,
            &mut self.conflicts,
            &mut self.conflict_ns,
            &mut self.strict_conflicts,
            &mut self.strict_ns,
            &mut self.range_hits,
            &mut self.image_hits,
            &mut self.induced,
            &mut self.induced_ns,
            &mut self.induced_upload_ns,
            &mut self.pending_holes,
            &mut self.pending_ns,
            &mut self.composed,
            &mut self.waves_ns,
            &mut self.rolling_ns,
            &mut self.schedule_ns,
            &mut self.strict_waves_ns,
            &mut self.strict_rolling_ns,
            &mut self.waves3_ns,
            &mut self.rolling3_ns,
            &mut self.schedule3_ns,
        ]
    }

    fn add(&mut self, o: &Sim) {
        let mut o = *o;
        for (a, b) in self.fields().into_iter().zip(o.fields()) {
            *a += *b;
        }
    }
}

/// What the analysis found, summed over frames.
#[derive(Clone, Debug, Default)]
pub struct Totals {
    pub frames: u64,
    pub items: u64,
    pub draws: u64,
    pub hole_draws: u64,
    /// Holes left once Stage 3 records CPU uploads speculatively.
    pub hole_draws3: u64,
    pub holes: [u64; HOLES],
    pub hole_ns: [u64; HOLES],
    pub events: [u64; KINDS],
    pub event_ns: [u64; KINDS],
    /// Speculative draw time (without commit-side work): P.
    pub p_ns: u64,
    pub hole_ns_total: u64,
    pub hole_ns3: u64,
    /// Commit-side work of speculative draws (early submits, collections).
    pub tail_ns: u64,
    /// Serial events (not guest memory work).
    pub serial_ns: u64,
    /// Guest memory work (replay patches, faults, mappings).
    pub memory_ns: u64,
    /// CPU faults that downloaded GPU data or waited for the GPU (the design runs each between
    /// waves: a wave break when it lands mid-wave).
    pub fault_downloads: u64,
    pub slices: [u64; NS.len()],
    pub slice_draws: [u64; NS.len()],
    pub max_slice: [u64; NS.len()],
    pub sims: [[Sim; KS.len()]; NS.len()],
    /// Speculative draws hit by a change, by source, at [`REFERENCE`].
    pub sources: Vec<u64>,
}

/// The (N, K) indices whose conflict sources are reported (N=128, K=6).
pub const REFERENCE: (usize, usize) = (2, 1);

impl Totals {
    const fn new_const() -> Totals {
        Totals {
            frames: 0,
            items: 0,
            draws: 0,
            hole_draws: 0,
            hole_draws3: 0,
            holes: [0; HOLES],
            hole_ns: [0; HOLES],
            events: [0; KINDS],
            event_ns: [0; KINDS],
            p_ns: 0,
            hole_ns_total: 0,
            hole_ns3: 0,
            tail_ns: 0,
            serial_ns: 0,
            memory_ns: 0,
            fault_downloads: 0,
            slices: [0; NS.len()],
            slice_draws: [0; NS.len()],
            max_slice: [0; NS.len()],
            sims: [[Sim::ZERO; KS.len()]; NS.len()],
            sources: Vec::new(),
        }
    }

    /// Serial time S (holes, commit-side work, serial events).
    pub fn s_ns(&self) -> u64 {
        self.hole_ns_total + self.tail_ns + self.serial_ns
    }

    pub fn add(&mut self, o: &Totals) {
        fn add_all(a: &mut [u64], b: &[u64]) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
        self.frames += o.frames;
        self.items += o.items;
        self.draws += o.draws;
        self.hole_draws += o.hole_draws;
        self.hole_draws3 += o.hole_draws3;
        add_all(&mut self.holes, &o.holes);
        add_all(&mut self.hole_ns, &o.hole_ns);
        add_all(&mut self.events, &o.events);
        add_all(&mut self.event_ns, &o.event_ns);
        self.p_ns += o.p_ns;
        self.hole_ns_total += o.hole_ns_total;
        self.hole_ns3 += o.hole_ns3;
        self.tail_ns += o.tail_ns;
        self.serial_ns += o.serial_ns;
        self.memory_ns += o.memory_ns;
        self.fault_downloads += o.fault_downloads;
        add_all(&mut self.slices, &o.slices);
        add_all(&mut self.slice_draws, &o.slice_draws);
        for (a, b) in self.max_slice.iter_mut().zip(&o.max_slice) {
            *a = (*a).max(*b);
        }
        for (ra, rb) in self.sims.iter_mut().zip(&o.sims) {
            for (a, b) in ra.iter_mut().zip(rb) {
                a.add(b);
            }
        }
        if self.sources.len() < o.sources.len() {
            self.sources.resize(o.sources.len(), 0);
        }
        add_all(&mut self.sources, &o.sources);
    }
}

/// The parallel time of a speculative draw (its commit-side tail is serial).
fn parallel_ns(it: &ItemRec) -> u64 {
    it.ns.saturating_sub(it.tail_ns)
}

/// The design's models for parallel time `p` and serial time `s` (ns).
fn models(p: u64, s: u64, k: usize) -> (u64, u64) {
    let (p, s, k) = (p as f64, s as f64, k as f64);
    ((p / k * IMBALANCE + s) as u64, (p / k).max(s) as u64)
}

/// Analyse one frame.
pub fn analyze(f: &Frame) -> Totals {
    let mut t = Totals { frames: 1, items: f.items.len() as u64, ..Totals::default() };
    for it in &f.items {
        let k = it.kind as usize;
        t.events[k] += 1;
        t.event_ns[k] += it.ns;
        match it.kind.cut() {
            Cut::Slice => {
                t.draws += 1;
                if it.holes == 0 {
                    t.p_ns += parallel_ns(it);
                    t.tail_ns += it.tail_ns;
                } else {
                    t.hole_draws += 1;
                    t.hole_ns_total += it.ns;
                    if it.holes & !STAGE3_SPECULATIVE != 0 {
                        t.hole_draws3 += 1;
                        t.hole_ns3 += it.ns;
                    }
                    for (h, n) in t.holes.iter_mut().enumerate() {
                        if it.holes & 1 << h != 0 {
                            *n += 1;
                            t.hole_ns[h] += it.ns;
                        }
                    }
                }
            }
            Cut::Memory => {
                t.memory_ns += it.ns;
                t.fault_downloads += (it.kind == Kind::Fault && it.holes & (Hole::Download.bit() | Hole::FlushWait.bit()) != 0) as u64;
            }
            _ => t.serial_ns += it.ns,
        }
    }
    let s = t.s_ns();
    // Stage 3: upload-only holes become speculative (the whole hole time:
    // an upper bound of what moves).
    let upload_holes = t.hole_ns_total - t.hole_ns3;
    for (ni, &n) in NS.iter().enumerate() {
        let slices = cut_slices(f, n);
        t.slices[ni] = slices.len() as u64;
        t.slice_draws[ni] = slices.iter().map(|s| s.draws as u64).sum();
        t.max_slice[ni] = slices.iter().map(|s| s.draws as u64).max().unwrap_or(0);
        let slice_p: Vec<u64> = slices.iter().map(|s| f.items[s.start..s.end].iter().filter(|i| speculative(i)).map(parallel_ns).sum()).collect();
        // Upload-only holes per slice (speculative in Stage 3).
        let slice_up: Vec<u64> =
            slices.iter().map(|s| f.items[s.start..s.end].iter().filter(|i| i.kind == Kind::Draw && i.holes != 0 && i.holes & !STAGE3_SPECULATIVE == 0).map(|i| i.ns).sum()).collect();
        for (ki, &k) in KS.iter().enumerate() {
            let w = waves(f, &slices, k);
            let c = conflicts(f, &slices, &w);
            let sum_of = |marked: &[bool]| -> u64 { slice_p.iter().zip(marked).filter(|(_, &x)| x).map(|(p, _)| *p).sum() };
            let conflict_ns = sum_of(&c.slices);
            let strict_ns = sum_of(&c.strict);
            // Re-recorded slices and draws turned holes are serial.
            let moved = conflict_ns + c.induced_ns + c.pending_ns;
            let (waves_ns, rolling_ns) = models(t.p_ns.saturating_sub(moved), s + moved, k);
            let strict_moved = strict_ns + c.strict_pending_ns;
            let (strict_waves_ns, strict_rolling_ns) = models(t.p_ns.saturating_sub(strict_moved), s + strict_moved, k);
            let back3 = upload_holes + c.induced_upload_ns;
            let (waves3_ns, rolling3_ns) = models(t.p_ns.saturating_sub(moved) + back3, (s + moved).saturating_sub(back3), k);
            // The schedule: per wave its longest slice of speculative work.
            let longest = |work: &dyn Fn(usize) -> u64| -> u64 { w.iter().map(|&(a, b)| (a..b).filter(|&i| !c.slices[i]).map(work).max().unwrap_or(0)).sum() };
            let schedule = longest(&|i| slice_p[i] - c.moved[i]);
            let schedule3 = longest(&|i| slice_p[i] - c.moved[i] + c.moved_upload[i] + slice_up[i]);
            if (ni, ki) == REFERENCE {
                t.sources = c.sources.clone();
            }
            t.sims[ni][ki] = Sim {
                waves: w.len() as u64,
                conflicts: c.slices.iter().filter(|&&x| x).count() as u64,
                conflict_ns,
                strict_conflicts: c.strict.iter().filter(|&&x| x).count() as u64,
                strict_ns,
                range_hits: c.range_hits,
                image_hits: c.image_hits,
                induced: c.induced,
                induced_ns: c.induced_ns,
                induced_upload_ns: c.induced_upload_ns,
                pending_holes: c.pending_holes,
                pending_ns: c.pending_ns,
                composed: c.composed,
                waves_ns,
                rolling_ns,
                schedule_ns: schedule + s + moved,
                strict_waves_ns,
                strict_rolling_ns,
                waves3_ns,
                rolling3_ns,
                schedule3_ns: schedule3 + (s + moved).saturating_sub(back3),
            };
        }
    }
    t
}

// ---------------------------------------------------------------------
// The process-wide log
// ---------------------------------------------------------------------

#[derive(Default)]
struct Log {
    current: Frame,
    pending: std::collections::VecDeque<Frame>,
    /// Since the start.
    cumulative: Totals,
    /// Since the last [`take`].
    window: Totals,
}

impl Log {
    fn end_frame(&mut self) {
        let f = std::mem::take(&mut self.current);
        if f.items.is_empty() {
            return;
        }
        self.pending.push_back(f);
        while self.pending.len() > MAX_PENDING {
            self.analyze_one();
        }
    }

    fn analyze_one(&mut self) {
        if let Some(f) = self.pending.pop_front() {
            let t = analyze(&f);
            self.cumulative.add(&t);
            self.window.add(&t);
        }
    }

    fn analyze_pending(&mut self) {
        while !self.pending.is_empty() {
            self.analyze_one();
        }
    }
}

static LOG: Mutex<Log> = Mutex::new(Log {
    current: Frame { items: Vec::new(), ranges: Vec::new(), ids: Vec::new() },
    pending: std::collections::VecDeque::new(),
    cumulative: Totals::new_const(),
    window: Totals::new_const(),
});

/// The totals of the frames flipped since the last call (analysed now).
pub fn take() -> Totals {
    let mut log = LOG.lock().unwrap_or_else(|e| e.into_inner());
    log.analyze_pending();
    std::mem::take(&mut log.window)
}

/// The `[gpu]` line part: cumulative per-frame means (None when off).
pub fn summary() -> Option<String> {
    if !enabled() {
        return None;
    }
    let t = {
        let mut log = LOG.lock().unwrap_or_else(|e| e.into_inner());
        log.analyze_pending();
        log.cumulative.clone()
    };
    Some(format!("slice-shadow {}", compact(&t)))
}

fn ms_per_frame(ns: u64, frames: u64) -> f64 {
    if frames == 0 {
        0.0
    } else {
        ns as f64 / frames as f64 / 1e6
    }
}

fn fps(ms: f64) -> f64 {
    if ms <= 0.0 {
        0.0
    } else {
        1e3 / ms
    }
}

/// The holes per frame by kind (most frequent first).
fn holes_line(t: &Totals) -> String {
    let fr = t.frames.max(1) as f64;
    let mut v: Vec<(usize, u64)> = t.holes.iter().copied().enumerate().filter(|&(_, n)| n != 0).collect();
    v.sort_unstable_by_key(|&(_, n)| std::cmp::Reverse(n));
    v.iter().map(|&(h, n)| format!("{}={:.1}/{:.2}ms", HOLE_NAMES[h], n as f64 / fr, ms_per_frame(t.hole_ns[h], t.frames))).collect::<Vec<_>>().join(" ")
}

fn events_line(t: &Totals) -> String {
    let fr = t.frames.max(1) as f64;
    (0..KINDS)
        .filter(|&k| k != Kind::Draw as usize && t.events[k] != 0)
        .map(|k| format!("{}={:.1}/{:.2}ms", KIND_NAMES[k], t.events[k] as f64 / fr, ms_per_frame(t.event_ns[k], t.frames)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn parallel_fraction(t: &Totals) -> f64 {
    let all = t.p_ns + t.s_ns();
    if all == 0 {
        0.0
    } else {
        t.p_ns as f64 / all as f64 * 100.0
    }
}

/// One line: per-frame holes, serial time and the N=128 predictions.
pub fn compact(t: &Totals) -> String {
    if t.frames == 0 {
        return "no frames".into();
    }
    let fr = t.frames as f64;
    let ni = REFERENCE.0;
    let preds: Vec<String> = KS
        .iter()
        .enumerate()
        .map(|(ki, k)| {
            let s = &t.sims[ni][ki];
            format!(
                "K{k} {:.1}/{:.1}ms (strict {:.1}, S3 {:.1}) re{:.1} ind{:.0}",
                ms_per_frame(s.waves_ns, t.frames),
                ms_per_frame(s.rolling_ns, t.frames),
                ms_per_frame(s.strict_rolling_ns, t.frames),
                ms_per_frame(s.rolling3_ns, t.frames),
                s.conflicts as f64 / fr,
                s.induced as f64 / fr
            )
        })
        .collect();
    format!(
        "frames={} draws={:.0} holes={:.0} ({:.1}%) P={:.2}ms S={:.2}ms (holes {:.2} tail {:.2} events {:.2}) par={:.1}% mem={:.2}ms N{} slices={:.0} waves/rolling: {} | holes: {}",
        t.frames,
        t.draws as f64 / fr,
        t.hole_draws as f64 / fr,
        if t.draws == 0 { 0.0 } else { t.hole_draws as f64 / t.draws as f64 * 100.0 },
        ms_per_frame(t.p_ns, t.frames),
        ms_per_frame(t.s_ns(), t.frames),
        ms_per_frame(t.hole_ns_total, t.frames),
        ms_per_frame(t.tail_ns, t.frames),
        ms_per_frame(t.serial_ns, t.frames),
        parallel_fraction(t),
        ms_per_frame(t.memory_ns, t.frames),
        NS[ni],
        t.slices[ni] as f64 / fr,
        preds.join(" "),
        holes_line(t)
    )
}

/// The replay report's section (per frame means; `gpu_ms`: GPU busy per
/// frame when measured).
pub fn report(t: &Totals, gpu_ms: Option<f64>) -> String {
    use std::fmt::Write;
    if t.frames == 0 {
        return "slice shadow: no frames analysed\n".into();
    }
    let fr = t.frames as f64;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "slice shadow ({} frames; per frame): {:.0} items, {:.0} draws, {:.0} holes ({:.1}%; {:.0} after Stage 3's speculative uploads)",
        t.frames,
        t.items as f64 / fr,
        t.draws as f64 / fr,
        t.hole_draws as f64 / fr,
        if t.draws == 0 { 0.0 } else { t.hole_draws as f64 / t.draws as f64 * 100.0 },
        t.hole_draws3 as f64 / fr
    );
    let _ = writeln!(out, "  holes by kind (draws/frame / ms of those draws): {}", holes_line(t));
    let _ = writeln!(out, "  serial events (count/frame / ms): {}", events_line(t));
    let _ = writeln!(
        out,
        "  P {:.2} ms (speculative draws) | S {:.2} ms = holes {:.2} + commit-side {:.2} + events {:.2} | parallel fraction {:.1}% | Stage 3 S {:.2} ms | guest memory side {:.2} ms (not recording; {:.1} faults downloaded or waited for the GPU)",
        ms_per_frame(t.p_ns, t.frames),
        ms_per_frame(t.s_ns(), t.frames),
        ms_per_frame(t.hole_ns_total, t.frames),
        ms_per_frame(t.tail_ns, t.frames),
        ms_per_frame(t.serial_ns, t.frames),
        parallel_fraction(t),
        ms_per_frame(t.s_ns() - (t.hole_ns_total - t.hole_ns3), t.frames),
        ms_per_frame(t.memory_ns, t.frames),
        t.fault_downloads as f64 / fr
    );
    let gpu = gpu_ms.map_or(String::new(), |g| format!(" GPU busy {g:.2} ms/frame (caps at {:.1} fps).", fps(g)));
    let _ = writeln!(
        out,
        "  predicted recording ms/frame (fps): waves P/K*{IMBALANCE}+S, rolling max(P/K,S), schedule = the waves of the actual slices; re-recorded slices and induced/pending holes move from P to S. strict: generation read sets; S3: Stage 3 uploads.{gpu}"
    );
    for (ni, &n) in NS.iter().enumerate() {
        let slices = t.slices[ni] as f64 / fr;
        let mean = if t.slices[ni] == 0 { 0.0 } else { t.slice_draws[ni] as f64 / t.slices[ni] as f64 };
        let _ = writeln!(out, "  N={n}: {slices:.0} slices/frame, {mean:.1} draws/slice (max {})", t.max_slice[ni]);
        for (ki, &k) in KS.iter().enumerate() {
            let s = &t.sims[ni][ki];
            let pct = |ns: u64| if t.p_ns == 0 { 0.0 } else { ns as f64 / t.p_ns as f64 * 100.0 };
            let ms = |ns: u64| ms_per_frame(ns, t.frames);
            let (w, r, sc) = (ms(s.waves_ns), ms(s.rolling_ns), ms(s.schedule_ns));
            let (sw, sr) = (ms(s.strict_waves_ns), ms(s.strict_rolling_ns));
            let (w3, r3, sc3) = (ms(s.waves3_ns), ms(s.rolling3_ns), ms(s.schedule3_ns));
            let _ = writeln!(
                out,
                "    K={k}: {:.1} waves | strict: {:.1} slices re-record ({:.0}% of P) -> waves {sw:.1} ({:.0} fps), rolling {sr:.1} ({:.0}) | value: {:.1} re-record ({:.1}% of P), {:.0} induced holes ({:.2} ms), {:.1} pending holes ({:.2} ms), {:.0} composed -> waves {w:.1} ({:.0}), rolling {r:.1} ({:.0}), schedule {sc:.1} ({:.0}) | S3 waves {w3:.1} ({:.0}), rolling {r3:.1} ({:.0}), schedule {sc3:.1} ({:.0})",
                s.waves as f64 / fr,
                s.strict_conflicts as f64 / fr,
                pct(s.strict_ns),
                fps(sw),
                fps(sr),
                s.conflicts as f64 / fr,
                pct(s.conflict_ns),
                s.induced as f64 / fr,
                ms(s.induced_ns),
                s.pending_holes as f64 / fr,
                ms(s.pending_ns),
                s.composed as f64 / fr,
                fps(w),
                fps(r),
                fps(sc),
                fps(w3),
                fps(r3),
                fps(sc3)
            );
        }
    }
    let mut sources: Vec<(usize, u64)> = t.sources.iter().copied().enumerate().filter(|&(_, n)| n != 0).collect();
    sources.sort_unstable_by_key(|&(_, n)| std::cmp::Reverse(n));
    let line: Vec<String> = sources.iter().map(|&(b, n)| format!("{}={:.1}", source_name(b), n as f64 / fr)).collect();
    let _ = writeln!(out, "  changes read by speculative draws at N={} K={} (draws/frame by source): {}", NS[REFERENCE.0], KS[REFERENCE.1], line.join(" "));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw(f: &mut Frame, rt: u64, holes: u32, ns: u64) {
        f.push(Kind::Draw, holes, ns, 0, rt, &[], &[], &[], &[]);
    }

    fn event(f: &mut Frame, kind: Kind, ns: u64) {
        f.push(kind, 0, ns, 0, 0, &[], &[], &[], &[]);
    }

    fn sizes(s: &[Slice]) -> Vec<usize> {
        s.iter().map(|s| s.draws).collect()
    }

    #[test]
    fn soft_cuts_move_to_the_next_target_change_within_a_quarter() {
        // 20 draws on target 1, then 20 on target 2: with N=16 the first
        // cut (nominal 16) moves to the change at 20 (within 16/4 = 4).
        let mut f = Frame::default();
        for i in 0..40 {
            draw(&mut f, if i < 20 { 1 } else { 2 }, 0, 1);
        }
        assert_eq!(sizes(&cut_slices(&f, 16)), vec![20, 16, 4]);
        // The change is 5 draws past the nominal cut: out of reach, the cut
        // stays at N.
        let mut f = Frame::default();
        for i in 0..42 {
            draw(&mut f, if i < 21 { 1 } else { 2 }, 0, 1);
        }
        assert_eq!(sizes(&cut_slices(&f, 16)), vec![16, 16, 10]);
        // A remainder of at most N stays one slice.
        let mut f = Frame::default();
        for _ in 0..16 {
            draw(&mut f, 1, 0, 1);
        }
        assert_eq!(sizes(&cut_slices(&f, 16)), vec![16]);
    }

    #[test]
    fn hard_cuts_end_slices_and_positioned_events_do_not() {
        let mut f = Frame::default();
        draw(&mut f, 1, 0, 1);
        event(&mut f, Kind::Flush, 1);
        draw(&mut f, 1, 0, 1);
        event(&mut f, Kind::Dispatch, 1);
        draw(&mut f, 1, 0, 1);
        event(&mut f, Kind::Patch, 1);
        draw(&mut f, 1, 0, 1);
        event(&mut f, Kind::EndOfPipe, 1);
        event(&mut f, Kind::Dispatch, 1);
        draw(&mut f, 1, 0, 1);
        let s = cut_slices(&f, 128);
        assert_eq!(s, vec![Slice { start: 0, end: 3, draws: 2 }, Slice { start: 4, end: 7, draws: 2 }, Slice { start: 9, end: 10, draws: 1 }]);
        // Waves of two slices, the end-of-pipe between slices 1 and 2 ends
        // a wave early.
        assert_eq!(waves(&f, &s, 2), vec![(0, 2), (2, 3)]);
        assert_eq!(waves(&f, &s, 8), vec![(0, 2), (2, 3)]);
        assert_eq!(waves(&f, &s, 1), vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn a_change_earlier_in_the_wave_conflicts_with_a_later_speculative_read() {
        let mut f = Frame::default();
        // Slice 0: a hole uploading [0x1000, 0x2000) and a speculative draw.
        f.push(Kind::Draw, Hole::BufferUpload.bit(), 5, 0, 1, &[(0x1000, 0x2000)], &[(0x1000, 0x2000)], &[], &[]);
        f.push(Kind::Draw, 0, 7, 0, 1, &[(0x8000, 0x9000)], &[], &[7], &[]);
        event(&mut f, Kind::Dispatch, 3);
        // Slice 1: reads the uploaded range (conflict) and image 7.
        f.push(Kind::Draw, 0, 11, 0, 1, &[(0x1800, 0x1900)], &[], &[], &[]);
        // Slice 2: reads image 9, which the hole of slice 3 creates later
        // (no conflict: changes only count for later reads).
        f.push(Kind::Draw, 0, 13, 0, 1, &[], &[], &[9], &[]);
        let s = cut_slices(&f, 1);
        assert_eq!(s.len(), 4);
        let w = waves(&f, &s, 8);
        assert_eq!(w, vec![(0, 4)]);
        let c = conflicts(&f, &s, &w);
        // Strictly the slice re-records; by value the reader would upload
        // the pages itself: an induced hole.
        assert_eq!(c.strict, vec![false, false, true, false]);
        assert_eq!(c.slices, vec![false; 4]);
        assert_eq!((c.induced, c.induced_ns, c.induced_upload_ns), (1, 11, 11));
        assert_eq!((c.range_hits, c.image_hits), (1, 0));
        // In waves of one slice nothing conflicts: every wave starts from
        // the committed state of the one before.
        let w1 = waves(&f, &s, 1);
        let c1 = conflicts(&f, &s, &w1);
        assert!(c1.slices.iter().all(|&x| !x));
        // An image changed earlier in the wave.
        let mut g = Frame::default();
        g.push(Kind::Draw, Hole::TargetChange.bit(), 1, 0, 1, &[], &[], &[], &[42]);
        g.push(Kind::Draw, 0, 1, 0, 1, &[], &[], &[42], &[]);
        let s = cut_slices(&g, 1);
        let c = conflicts(&g, &s, &waves(&g, &s, 4));
        assert_eq!(c.strict, vec![false, true]);
        assert_eq!((c.induced, c.induced_upload_ns), (1, 0));
        assert_eq!(c.image_hits, 1);
        // A storage write's image change is not redone by the reader: the
        // slice re-records.
        let mut g = Frame::default();
        g.push(Kind::Draw, Hole::StorageImage.bit() | Hole::ImageGeneration.bit(), 1, 0, 1, &[], &[], &[], &[42]);
        g.push(Kind::Draw, 0, 1, 0, 1, &[], &[], &[42], &[]);
        let s = cut_slices(&g, 1);
        let c = conflicts(&g, &s, &waves(&g, &s, 4));
        assert_eq!(c.slices, vec![false, true]);
        assert_eq!(c.induced, 0);
    }

    #[test]
    fn changes_before_a_wave_cut_are_committed_state() {
        let mut f = Frame::default();
        f.push(Kind::Draw, Hole::WrittenRange.bit(), 1, 0, 1, &[], &[(0, 0x1000)], &[], &[]);
        event(&mut f, Kind::EndOfPipe, 1);
        f.push(Kind::Draw, 0, 1, 0, 1, &[(0, 0x10)], &[], &[], &[]);
        let s = cut_slices(&f, 64);
        let w = waves(&f, &s, 8);
        assert_eq!(w.len(), 2);
        assert!(conflicts(&f, &s, &w).slices.iter().all(|&x| !x));
    }

    #[test]
    fn analysis_splits_parallel_and_serial_time() {
        let mut f = Frame::default();
        for i in 0..8 {
            // Speculative draws of 10 ns with a 2 ns commit-side tail.
            f.push(Kind::Draw, 0, 10, 2, i / 4, &[], &[], &[], &[]);
        }
        f.push(Kind::Draw, Hole::ImageCreate.bit() | Hole::ImageGeneration.bit(), 50, 0, 9, &[], &[], &[], &[]);
        f.push(Kind::Draw, Hole::BufferUpload.bit(), 30, 0, 9, &[], &[], &[], &[]);
        event(&mut f, Kind::Dispatch, 40);
        event(&mut f, Kind::Patch, 100);
        event(&mut f, Kind::Flip, 5);
        let t = analyze(&f);
        assert_eq!((t.draws, t.hole_draws, t.hole_draws3), (10, 2, 1));
        assert_eq!(t.p_ns, 8 * 8);
        assert_eq!(t.tail_ns, 8 * 2);
        assert_eq!(t.hole_ns_total, 80);
        assert_eq!(t.hole_ns3, 50);
        assert_eq!(t.serial_ns, 45);
        assert_eq!(t.memory_ns, 100);
        assert_eq!(t.s_ns(), 80 + 16 + 45);
        assert_eq!(t.holes[Hole::ImageCreate as usize], 1);
        assert_eq!(t.holes[Hole::ImageGeneration as usize], 1);
        assert_eq!(t.holes[Hole::BufferUpload as usize], 1);
        assert_eq!(t.hole_ns[Hole::BufferUpload as usize], 30);
        assert_eq!(t.events[Kind::Dispatch as usize], 1);
        // N=32: one slice of 10 draws; K=4: P/K*1.15 + S, max(P/K, S).
        let ni = 0;
        assert_eq!(t.slices[ni], 1);
        let s = t.sims[ni][0];
        assert_eq!(s.conflicts, 0);
        assert_eq!(s.waves_ns, (64.0 / 4.0 * IMBALANCE + 141.0) as u64);
        assert_eq!(s.rolling_ns, 141);
        // One wave, one slice: its whole P plus S.
        assert_eq!(s.schedule_ns, 64 + 141);
        // Stage 3: the upload hole joins its slice's parallel work.
        assert_eq!(s.schedule3_ns, 64 + 30 + 141 - 30);
        // Stage 3 moves the upload hole into P.
        assert_eq!(s.rolling3_ns, 111);
        let mut sum = Totals::default();
        sum.add(&t);
        sum.add(&t);
        assert_eq!((sum.frames, sum.p_ns, sum.sims[ni][0].rolling_ns), (2, 128, 282));
    }

    #[test]
    fn items_log_holes_reads_and_tails_on_their_thread() {
        // The process-wide switch may be off in tests: exercise the
        // accumulator directly.
        ACC.with(|a| {
            let mut a = a.borrow_mut();
            *a = Acc { depth: 1, ..Acc::default() };
        });
        let note = |f: &dyn Fn(&mut Acc)| ACC.with(|a| f(&mut a.borrow_mut()));
        note(&|a| a.holes |= Hole::StagingFull.bit());
        note(&|a| a.suppress += 1);
        // Suppressed (commit-side) work notes no holes.
        note(&|a| {
            if a.suppress == 0 {
                a.holes |= Hole::FlushWait.bit();
            }
        });
        note(&|a| a.suppress -= 1);
        let holes = ACC.with(|a| a.borrow().holes);
        assert_eq!(holes, Hole::StagingFull.bit());
        ACC.with(|a| *a.borrow_mut() = Acc::default());
        // Outside items nothing is noted.
        read(0x1000, 0x10);
        assert!(ACC.with(|a| a.borrow().reads.is_empty()));
    }

    #[test]
    fn range_sets_merge_and_answer_overlaps() {
        let mut r = RangeSet::default();
        r.insert((10, 20), 1);
        r.insert((30, 40), 2);
        assert_eq!(r.overlaps((15, 16)), 1);
        assert_eq!(r.overlaps((20, 30)), 0);
        assert_ne!(r.overlaps((19, 31)), 0);
        r.insert((18, 32), 4);
        assert_eq!(r.0.len(), 1);
        // The merged range carries every source.
        assert_eq!(r.overlaps((25, 26)), 7);
        assert_eq!(r.overlaps((40, 50)), 0);
        assert_eq!(r.overlaps((0, 10)), 0);
        assert_eq!(r.overlaps((12, 12)), 0);
    }

    #[test]
    fn cpu_side_writes_compose_into_copies_and_make_exact_reads_holes() {
        let mut f = Frame::default();
        // A patch of [0x1000, 0x1100), then a draw copying it (composes), a
        // draw binding its mirror (a pending hole) and a draw reading
        // memory a dispatch wrote (a conflict, attributed to the dispatch).
        f.push(Kind::Patch, 0, 1, 0, 0, &[], &[(0x1000, 0x1100)], &[], &[]);
        draw(&mut f, 1, 0, 3);
        f.push_soft_reads(&[(0x1000, 0x1010)]);
        f.push(Kind::Draw, 0, 5, 0, 1, &[(0x1080, 0x1090)], &[], &[], &[]);
        f.push(Kind::Dispatch, 0, 1, 0, 0, &[], &[(0x9000, 0xA000)], &[], &[]);
        f.push(Kind::Draw, 0, 7, 0, 1, &[(0x9800, 0x9900)], &[], &[], &[]);
        let s = cut_slices(&f, 64);
        assert_eq!(s.len(), 2);
        let c = conflicts(&f, &s, &waves(&f, &s, 4));
        assert_eq!((c.composed, c.pending_holes, c.pending_ns), (1, 1, 5));
        assert_eq!(c.slices, vec![false, true]);
        assert_eq!(c.sources[32 + Kind::Dispatch as usize], 1);
        assert_eq!(source_name(32 + Kind::Dispatch as usize), "dispatch");
        assert_eq!(source_name(Hole::BufferUpload as usize), "bufupload");
        // The analysis moves the pending hole and the conflicting slice
        // from P to S: K=4 rolling = max((15 - 7 - 5)/4, S + 12).
        let t = analyze(&f);
        assert_eq!(t.p_ns, 15);
        let k4 = t.sims[1][0];
        assert_eq!((k4.conflict_ns, k4.pending_ns), (7, 5));
        assert_eq!(k4.rolling_ns, t.s_ns() + 12);
    }
}
