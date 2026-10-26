// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Speculative recording against frozen committed state (R7b stage 2).
//!
//! A slice thread records draws with the same code as the committing
//! thread, through [`Buffers`] and [`Images`]: exact access forwards to the
//! caches, speculative access only reads them (they are frozen while a wave
//! records). What a speculative draw read is noted ([`DrawReads`]), what it
//! would change is journaled ([`Effects`]), and image layouts go through a
//! per-segment [`LayoutOverlay`]. Anything that needs the exact state (an
//! image or buffer to create, upload, refresh or clear, a written range, a
//! submit or a wait, a missing pipeline) sets the sticky bail flag of the
//! [`SpecCtx`]: the draw becomes a hole, recorded exactly by the committing
//! thread at its position.

use crate::buffers::{BufferCache, BufferSlice, GpuBuffer};
use crate::device::ImageView;
use crate::images::layout::{LayoutTracker, SubRange, UNKNOWN};
use crate::images::{CachedImage, ImageCache, ImageKey, ImageUse, ViewPlan, ViewRange};
use crate::memory::tracker::GuestPageTracker;
use crate::memory::{GuestGpuMemory, PAGE_BYTES};
use crate::scheduler::Scheduler;
use ash::vk;
use std::cell::{Cell, RefCell};
use std::sync::Arc;

/// Why a speculative draw became a hole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Hole {
    /// No buffer contains a range (the cache would create or merge one).
    BufferMiss,
    /// CPU-dirty pages to upload into a mirror.
    Upload,
    /// A range that cannot be watched (uploaded at every use).
    Untracked,
    /// The draw writes buffer ranges or storage images.
    Written,
    /// An image to create.
    ImageMiss,
    /// An image to upload, refresh or revalidate before use.
    ImagePrepare,
    /// A first GPU write that changes image lookups.
    TargetChange,
    /// A DCC/CMASK fast clear found in metadata bytes.
    Clear,
    /// A metadata fill waits for its consumer.
    ClearFill,
    /// A compute program overwrote the metadata (its flag is set).
    ClearFlag,
    /// HTILE slices cleared by a fill.
    ClearHtile,
    /// A depth or stencil clear draw.
    ClearDepth,
    /// A module, pipeline, layout, sampler or dummy resource is missing.
    Pipeline,
    /// The staging ring is full.
    Staging,
    /// Indirect arguments, GDS, mesh launches from records, metadata
    /// operations, resolves or copies.
    Special,
    /// A submit, wait, download, host read-back or callback was asked for.
    Submit,
    /// The front end could not materialise without the exact state (a
    /// GPU-written descriptor word).
    Frontend,
    /// An error the exact path reports.
    Other,
}

impl Hole {
    pub const COUNT: usize = 18;
    pub const NAMES: [&'static str; Self::COUNT] = [
        "bufmiss", "upload", "untracked", "written", "imgmiss", "prepare", "target", "clear", "fill", "flag", "htile", "zclear", "pipeline", "staging", "special", "submit",
        "frontend", "other",
    ];
}

/// Guest pages a speculative draw read the bytes of (descriptor words,
/// stream copies, metadata), with their generations when read: the bytes
/// are what the draw saw while every generation holds (see
/// [`crate::memory::deps::PageDeps`]).
#[derive(Clone, Default)]
pub struct ReadSet {
    pages: Vec<(u64, Option<u32>)>,
    volatile: bool,
    sync: u64,
    maps: u64,
}

impl ReadSet {
    pub fn begin(memory: &GuestGpuMemory) -> ReadSet {
        ReadSet { sync: crate::memory::sync_generation(), maps: memory.map_generation(), ..ReadSet::default() }
    }

    /// `[addr, addr+size)` is about to be read.
    pub fn note(&mut self, tracker: &GuestPageTracker, addr: u64, size: u64) {
        let first = addr & !(PAGE_BYTES - 1);
        let end = addr.saturating_add(size.max(1));
        let mut page = first;
        let mut block = None;
        while page < end {
            if self.pages.last().is_none_or(|&(p, _)| p != page) && !self.pages.iter().rev().take(8).any(|&(p, _)| p == page) {
                let generation = tracker.page_generation_in(&mut block, page);
                // Pages the CPU may write without a fault are found when
                // checked (`current_except`): no region lock per read here.
                self.volatile = self.volatile || generation.is_none();
                self.pages.push((page, generation));
            }
            page += PAGE_BYTES;
        }
    }

    /// [`Self::current`], except that pages `composed` says known writes
    /// reached (the reads saw those writes) may have moved; they must hold
    /// no GPU data.
    /// `writes_known`: every CPU write since the reads is among the writes
    /// the caller checked (the capture replay): a synchronisation point
    /// then stands for no unseen write of a CPU-dirty page.
    pub fn current_except(&self, tracker: &GuestPageTracker, memory: &GuestGpuMemory, composed: &dyn Fn(u64) -> bool, writes_known: bool) -> Result<(), &'static str> {
        if self.maps != memory.map_generation() {
            return Err("maps");
        }
        if !writes_known && self.sync != crate::memory::sync_generation() && (self.volatile || self.pages.iter().any(|&(p, _)| tracker.page_maybe_cpu_dirty(p))) {
            return Err("sync");
        }
        let mut block = None;
        if self.pages.iter().all(|&(p, g)| tracker.page_generation_in(&mut block, p) == g || (composed(p) && !tracker.has_gpu_dirty_pages(p, PAGE_BYTES))) {
            Ok(())
        } else {
            Err("pages")
        }
    }

    /// The pages still hold the bytes read.
    pub fn current(&self, tracker: &GuestPageTracker, memory: &GuestGpuMemory) -> bool {
        self.maps == memory.map_generation()
            && (self.sync == crate::memory::sync_generation() || !(self.volatile || self.pages.iter().any(|&(p, _)| tracker.page_maybe_cpu_dirty(p))))
            && {
            let mut block = None;
            self.pages.iter().all(|&(p, g)| tracker.page_generation_in(&mut block, p) == g)
        }
    }

    pub fn clear(&mut self, memory: &GuestGpuMemory) {
        self.pages.clear();
        self.volatile = false;
        self.sync = crate::memory::sync_generation();
        self.maps = memory.map_generation();
    }

    /// Whether a page of `[addr, addr+size)` was read.
    pub fn overlaps(&self, addr: u64, size: u64) -> bool {
        let (first, end) = (addr & !(PAGE_BYTES - 1), addr.saturating_add(size.max(1)));
        self.pages.iter().any(|&(p, _)| p >= first && p < end)
    }

    pub fn len(&self) -> usize {
        self.pages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetaCheck {
    Dcc(u64),
    Htile(u64, u32, u64),
}

/// What a speculative draw's decisions depended on, re-checked against the
/// committed state when its segment commits (value-based: the answers must
/// be the ones the frozen state gave).
#[derive(Clone, Default)]
pub struct DrawReads {
    /// Mirrors bound without an upload: (address, size, buffer key, serial).
    pub mirrors: Vec<(u64, u64, u64, u64)>,
    /// Mirrors bound that need an upload of CPU-dirty pages (made by the
    /// committing thread before the segment, stage 3): (address, size,
    /// buffer key, serial). Their dirty pages are in `pages`.
    pub uploads: Vec<(u64, u64, u64, u64)>,
    /// Stream decisions: (address, size, streamed).
    pub streams: Vec<(u64, u64, bool)>,
    /// Guest bytes read.
    pub pages: ReadSet,
    /// Images looked up or bound with their lookup generation.
    pub images: Vec<(u64, u64)>,
    /// Render targets bound (found by key, current without an upload: their
    /// generation may move by writes that leave both true).
    pub targets: Vec<u64>,
    /// Image keys looked up: (key, image found).
    pub keys: Vec<(ImageKey, u64)>,
    /// Metadata decisions taken (no fast clear pending): DCC/CMASK at a
    /// base (no fill waits, no overwrite flag), HTILE at (base, slice) (no
    /// fill waits, the slice's clear bits are consumed).
    pub meta: Vec<MetaCheck>,
    /// Guest ranges whose bytes or GPU contents the draw used (positioned
    /// CPU writes before the draw in its segment conflict with them).
    pub ranges: Vec<(u64, u64)>,
}

impl DrawReads {
    pub fn clear(&mut self, memory: &GuestGpuMemory) {
        self.mirrors.clear();
        self.uploads.clear();
        self.streams.clear();
        self.pages.clear(memory);
        self.images.clear();
        self.targets.clear();
        self.keys.clear();
        self.meta.clear();
        self.ranges.clear();
    }

    pub fn overlaps(&self, addr: u64, size: u64) -> bool {
        let end = addr.saturating_add(size.max(1));
        self.ranges.iter().any(|&(a, n)| a < end && addr < a + n) || self.pages.overlaps(addr, size)
    }
}

/// A view made by a slice thread, cached at commit.
pub struct NewView {
    pub plan: ViewPlan,
    pub view: ImageView,
}

/// What a speculative segment changes, applied in order at commit with the
/// real tick.
#[derive(Default)]
pub struct Effects {
    /// Mirrors used: (buffer key, address, size).
    pub buffer_uses: Vec<(u64, u64, u64)>,
    /// Images used.
    pub image_uses: Vec<u64>,
    /// Images written as attachments (draw order).
    pub gpu_written: Vec<u64>,
    /// Views made.
    pub views: Vec<NewView>,
    /// Back-end metadata state: DCC targets, CMASK bases, HTILE bases, DCC
    /// flags reset (`dcc[base] = false`).
    pub dcc_targets: Vec<(u64, crate::render::ColorTarget)>,
    pub cmask: Vec<u64>,
    pub htile: Vec<u64>,
    pub dcc_reset: Vec<u64>,
}

impl Effects {
    pub fn clear(&mut self) {
        self.buffer_uses.clear();
        self.image_uses.clear();
        self.gpu_written.clear();
        self.views.clear();
        self.dcc_targets.clear();
        self.cmask.clear();
        self.htile.clear();
        self.dcc_reset.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.buffer_uses.is_empty()
            && self.image_uses.is_empty()
            && self.gpu_written.is_empty()
            && self.views.is_empty()
            && self.dcc_targets.is_empty()
            && self.cmask.is_empty()
            && self.htile.is_empty()
            && self.dcc_reset.is_empty()
    }
}

/// Image layouts as a slice segment uses them: the first use of each
/// subresource is a requirement on the committed state (no barrier is
/// recorded for it); later uses get barriers from the segment's own state.
/// At commit the requirements become barriers from the committed state
/// (the seam) and the committed state takes the segment's last one.
#[derive(Default)]
pub struct LayoutOverlay {
    pub images: Vec<OverlayImage>,
}

pub struct OverlayImage {
    pub id: u64,
    pub tracker: LayoutTracker,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    /// (range, use, stage) per first use.
    pub requirements: Vec<(Option<SubRange>, ImageUse, vk::PipelineStageFlags2)>,
}

impl LayoutOverlay {
    pub fn clear(&mut self) {
        self.images.clear();
    }

    fn entry(&mut self, id: u64, shape: impl FnOnce() -> Option<(LayoutTracker, vk::Image, vk::ImageAspectFlags)>) -> Option<&mut OverlayImage> {
        if let Some(i) = self.images.iter().position(|o| o.id == id) {
            return self.images.get_mut(i);
        }
        let (t, image, aspect) = shape()?;
        self.images.push(OverlayImage { id, tracker: t.unknown_like(), image, aspect, requirements: Vec::new() });
        self.images.last_mut()
    }

    /// [`ImageCache::transition`] against the overlay.
    pub fn transition(&mut self, images: &ImageCache, id: u64, usage: ImageUse, range: Option<SubRange>, out: &mut Vec<vk::ImageMemoryBarrier2<'static>>) {
        self.transition_with(id, || images.layout_of(id).map(|(t, i, a)| (t.clone(), i, a)), usage, range, out)
    }

    /// [`Self::transition`] of an image of the shape `shape` gives.
    pub fn transition_with(
        &mut self,
        id: u64,
        shape: impl FnOnce() -> Option<(LayoutTracker, vk::Image, vk::ImageAspectFlags)>,
        (layout, access): ImageUse,
        range: Option<SubRange>,
        out: &mut Vec<vk::ImageMemoryBarrier2<'static>>,
    ) {
        let Some(o) = self.entry(id, shape) else { return };
        let stage = crate::images::layout::stages_for(access);
        let at = out.len();
        o.tracker.barriers(o.image, o.aspect, layout, access, stage, range, out);
        // Barriers from the unknown state are requirements.
        let mut i = at;
        while i < out.len() {
            if out[i].old_layout == UNKNOWN {
                let b = out.remove(i);
                let r = b.subresource_range;
                let range = (r.level_count != vk::REMAINING_MIP_LEVELS).then_some(SubRange {
                    base_level: r.base_mip_level,
                    levels: r.level_count,
                    base_layer: r.base_array_layer,
                    layers: r.layer_count,
                });
                o.requirements.push((range, (b.new_layout, b.dst_access_mask), b.dst_stage_mask));
            } else {
                i += 1;
            }
        }
    }

    /// The segment used image `id`.
    pub fn knows(&self, id: u64) -> bool {
        self.images.iter().any(|o| o.id == id)
    }

    /// [`ImageCache::uniform_layout`] against the overlay (None for
    /// subresources not used yet: the committed layout is not known).
    pub fn uniform_layout(&self, id: u64, range: SubRange) -> Option<vk::ImageLayout> {
        let o = self.images.iter().find(|o| o.id == id)?;
        let t = &o.tracker;
        let first = if t.is_uniform() { t.state().layout } else { t.subresource_state(range.base_level, range.base_layer).layout };
        if first == UNKNOWN {
            return None;
        }
        if !t.is_uniform() {
            for level in range.base_level..range.base_level + range.levels.max(1) {
                for layer in range.base_layer..range.base_layer + range.layers.max(1) {
                    if t.subresource_state(level, layer).layout != first {
                        return None;
                    }
                }
            }
        }
        Some(first)
    }
}

/// The speculative state of one slice thread (one segment at a time).
pub struct SpecCtx {
    pub memory: &'static GuestGpuMemory,
    bail: Cell<Option<Hole>>,
    pub reads: RefCell<DrawReads>,
    pub effects: RefCell<Effects>,
    pub overlay: RefCell<LayoutOverlay>,
    /// Views made in the open segment (not cached yet).
    pending_views: RefCell<Vec<(ViewPlan, vk::ImageView)>>,
    /// Mirrors with CPU-dirty pages are bound and uploaded at commit
    /// (stage 3) instead of making the draw a hole.
    uploads: Cell<bool>,
    /// Answers of the frozen state within the wave being recorded (they
    /// cannot change while it records): mirror reads and image prepares.
    memo: RefCell<WaveMemo>,
}

#[derive(Default)]
struct WaveMemo {
    wave: u64,
    reads: crate::fxhash::HashMap<(u64, u64), Result<(BufferSlice, u64, u64), Hole>>,
    noop: crate::fxhash::HashMap<u64, bool>,
}

impl SpecCtx {
    pub fn new(memory: &'static GuestGpuMemory) -> SpecCtx {
        SpecCtx {
            memory,
            bail: Cell::new(None),
            reads: RefCell::new(DrawReads { pages: ReadSet::begin(memory), ..DrawReads::default() }),
            effects: RefCell::default(),
            overlay: RefCell::default(),
            pending_views: RefCell::default(),
            uploads: Cell::new(true),
            memo: RefCell::default(),
        }
    }

    /// Fail the draw being recorded (the first reason sticks).
    pub fn bail(&self, why: Hole) {
        if self.bail.get().is_none() {
            self.bail.set(Some(why));
        }
    }

    pub fn bailed(&self) -> Option<Hole> {
        self.bail.get()
    }

    /// Start a draw: the bail flag and its reads are reset.
    pub fn begin_draw(&self) {
        self.bail.set(None);
        self.reads.borrow_mut().clear(self.memory);
    }

    /// Start a segment: its effects, overlay and views are reset.
    pub fn begin_segment(&self) {
        self.effects.borrow_mut().clear();
        self.overlay.borrow_mut().clear();
        self.pending_views.borrow_mut().clear();
    }

    /// A CPU write of the wave before this draw reaches the range (mirrors
    /// it reaches need an upload; ranges it makes CPU-dirty prefer a copy).
    pub fn pending_overlaps(&self, addr: u64, size: u64) -> bool {
        crate::memory::pending::overlaps(addr, size)
    }

    /// A new wave records: answers of the frozen state of the last one are
    /// dropped.
    pub fn begin_wave(&self, wave: u64) {
        let mut m = self.memo.borrow_mut();
        if m.wave != wave {
            m.wave = wave;
            m.reads.clear();
            m.noop.clear();
        }
    }

    fn peek_read(&self, b: &BufferCache, addr: u64, size: u64) -> Result<(BufferSlice, u64, u64), Hole> {
        if let Some(r) = self.memo.borrow().reads.get(&(addr, size)) {
            return *r;
        }
        let r = b.peek_read(addr, size);
        self.memo.borrow_mut().reads.insert((addr, size), r);
        r
    }

    pub fn prepare_noop(&self, images: &ImageCache, id: u64) -> bool {
        if let Some(&r) = self.memo.borrow().noop.get(&id) {
            return r;
        }
        let r = images.prepare_noop(id, false);
        self.memo.borrow_mut().noop.insert(id, r);
        r
    }

    pub fn set_uploads(&self, on: bool) {
        self.uploads.set(on);
    }

    pub fn note_range(&self, addr: u64, size: u64) {
        self.reads.borrow_mut().ranges.push((addr, size));
    }

    pub fn note_pages(&self, tracker: &GuestPageTracker, addr: u64, size: u64) {
        self.reads.borrow_mut().pages.note(tracker, addr, size);
    }

    pub fn note_deps(&self, tracker: &GuestPageTracker, deps: &crate::memory::deps::PageDeps) {
        let mut r = self.reads.borrow_mut();
        for p in deps.page_list() {
            r.pages.note(tracker, p, PAGE_BYTES);
        }
        if deps.volatile() {
            r.pages.volatile = true;
        }
    }

    pub fn note_target(&self, images: &ImageCache, id: u64) {
        let mut r = self.reads.borrow_mut();
        if !r.targets.contains(&id) {
            r.targets.push(id);
            if let Some((b, e)) = images.guest_range(id) {
                r.ranges.push((b, e - b));
            }
        }
    }

    pub fn note_image(&self, images: &ImageCache, id: u64) {
        let g = images.image_generation(id).unwrap_or(u64::MAX);
        let mut r = self.reads.borrow_mut();
        if !r.images.iter().any(|&(i, _)| i == id) {
            r.images.push((id, g));
            if let Some((b, e)) = images.guest_range(id) {
                r.ranges.push((b, e - b));
            }
        }
    }
}

/// Where a draw's reads stood (see [`SpecCtx::mark`]).
#[derive(Clone, Copy)]
pub struct ReadMark(usize, usize, usize);

/// The reads of a memoised binding, replayed into a later draw that reuses
/// it ([`SpecCtx::replay`]).
#[derive(Clone, Default)]
pub struct BindingReads {
    mirrors: Vec<(u64, u64, u64, u64)>,
    ranges: Vec<(u64, u64)>,
    pages: Vec<u64>,
}

impl SpecCtx {
    pub fn mark(&self) -> ReadMark {
        let r = self.reads.borrow();
        ReadMark(r.mirrors.len(), r.ranges.len(), r.pages.pages.len())
    }

    /// The reads since `m`.
    pub fn since(&self, m: ReadMark) -> BindingReads {
        let r = self.reads.borrow();
        BindingReads {
            mirrors: r.mirrors.get(m.0..).unwrap_or_default().to_vec(),
            ranges: r.ranges.get(m.1..).unwrap_or_default().to_vec(),
            pages: r.pages.pages.get(m.2..).unwrap_or_default().iter().map(|&(p, _)| p).collect(),
        }
    }

    /// A memoised binding is reused: its reads count for this draw too
    /// (its pages at their current generations: memory is frozen while a
    /// wave records, and the memo is only valid while nothing changed).
    pub fn replay(&self, b: &BindingReads, tracker: &GuestPageTracker) {
        let mut r = self.reads.borrow_mut();
        r.mirrors.extend_from_slice(&b.mirrors);
        r.ranges.extend_from_slice(&b.ranges);
        for &p in &b.pages {
            r.pages.note(tracker, p, PAGE_BYTES);
        }
        drop(r);
        let mut e = self.effects.borrow_mut();
        for &(a, n, k, _) in &b.mirrors {
            e.buffer_uses.push((k, a, n));
        }
    }
}

/// Buffer cache access of a draw: exact, or speculative (read-only).
pub enum Buffers<'a> {
    Exact(&'a mut BufferCache),
    Spec(&'a BufferCache, &'a SpecCtx),
}

impl<'a> From<&'a mut BufferCache> for Buffers<'a> {
    fn from(b: &'a mut BufferCache) -> Self {
        Buffers::Exact(b)
    }
}

impl Buffers<'_> {
    pub fn cache(&self) -> &BufferCache {
        match self {
            Buffers::Exact(b) => b,
            Buffers::Spec(b, _) => b,
        }
    }

    pub fn spec(&self) -> Option<&SpecCtx> {
        match self {
            Buffers::Exact(_) => None,
            Buffers::Spec(_, s) => Some(s),
        }
    }

    /// The exact cache (None on a slice thread: the caller bails).
    pub fn exact(&mut self) -> Option<&mut BufferCache> {
        match self {
            Buffers::Exact(b) => Some(b),
            Buffers::Spec(..) => None,
        }
    }

    pub fn tracker(&self) -> &Arc<GuestPageTracker> {
        &self.cache().tracker
    }

    /// [`BufferCache::obtain`]: speculatively only a read-only use of a
    /// mirror that holds the range and needs no upload.
    pub fn obtain(&mut self, sched: &mut Scheduler, addr: u64, size: u64, written: bool) -> Result<BufferSlice, vk::Result> {
        match self {
            Buffers::Exact(b) => b.obtain(sched, addr, size, written),
            Buffers::Spec(b, s) => {
                if written {
                    s.bail(Hole::Written);
                    return Err(vk::Result::ERROR_UNKNOWN);
                }
                let r = if s.pending_overlaps(addr, size) {
                    // The wave's CPU writes make the range CPU-dirty first.
                    match b.find_containing(addr, size.max(1)) {
                        Some(_) => Err(Hole::Upload),
                        None => Err(Hole::BufferMiss),
                    }
                } else {
                    s.peek_read(b, addr, size)
                };
                match r {
                    Ok((slice, key, serial)) => {
                        let mut r = s.reads.borrow_mut();
                        r.mirrors.push((addr, size.max(1), key, serial));
                        r.ranges.push((addr, size.max(1)));
                        drop(r);
                        s.effects.borrow_mut().buffer_uses.push((key, addr, size.max(1)));
                        Ok(slice)
                    }
                    Err(Hole::Upload) if s.uploads.get() => match b.peek_read_dirty(addr, size) {
                        Some((slice, key, serial)) => {
                            let n = size.max(1);
                            let tracker = b.tracker.clone();
                            let mut r = s.reads.borrow_mut();
                            r.uploads.push((addr, n, key, serial));
                            r.ranges.push((addr, n));
                            tracker.for_each_possibly_cpu_dirty_range(addr, n, &mut |a, m| r.pages.note(&tracker, a, m));
                            drop(r);
                            s.effects.borrow_mut().buffer_uses.push((key, addr, n));
                            Ok(slice)
                        }
                        None => {
                            s.bail(Hole::Upload);
                            Err(vk::Result::ERROR_UNKNOWN)
                        }
                    },
                    Err(why) => {
                        s.bail(why);
                        Err(vk::Result::ERROR_UNKNOWN)
                    }
                }
            }
        }
    }

    pub fn find_or_create(&mut self, sched: &mut Scheduler, addr: u64, size: u64) -> Result<u64, vk::Result> {
        match self {
            Buffers::Exact(b) => b.find_or_create(sched, addr, size),
            Buffers::Spec(b, s) => match b.find_containing(addr, size) {
                Some(k) => Ok(k),
                None => {
                    s.bail(Hole::BufferMiss);
                    Err(vk::Result::ERROR_UNKNOWN)
                }
            },
        }
    }

    pub fn has_gpu_dirty(&self, addr: u64, size: u64) -> bool {
        self.cache().has_gpu_dirty(addr, size)
    }

    pub fn download(&mut self, sched: &mut Scheduler, addr: u64, size: u64) -> bool {
        match self {
            Buffers::Exact(b) => b.download(sched, addr, size),
            Buffers::Spec(_, s) => {
                s.bail(Hole::Submit);
                false
            }
        }
    }

    pub fn content_stamp(&self) -> [u64; 4] {
        self.cache().content_stamp()
    }

    pub fn layout_epoch(&self) -> u64 {
        self.cache().layout_epoch()
    }

    pub fn get(&self, key: u64) -> Option<&GpuBuffer> {
        self.cache().get(key)
    }

    pub fn find_containing(&self, addr: u64, size: u64) -> Option<u64> {
        self.cache().find_containing(addr, size)
    }

    pub fn prefers_stream(&mut self, addr: u64, size: u64) -> bool {
        match self {
            Buffers::Exact(b) => b.prefers_stream(addr, size),
            Buffers::Spec(b, s) => {
                let r = if s.pending_overlaps(addr, size) {
                    // CPU-dirty once the wave's writes before the draw ran.
                    b.stream_eligible(addr, size) && !b.has_gpu_dirty(addr, size)
                } else {
                    b.prefers_stream_query(addr, size)
                };
                s.reads.borrow_mut().streams.push((addr, size, r));
                r
            }
        }
    }

    pub fn bda_backed(&self, addr: u64, size: u64) -> bool {
        self.cache().bda_backed(addr, size)
    }

    pub fn bda_resident(&self, addr: u64) -> bool {
        self.cache().bda_resident(addr)
    }

    pub fn bda_residency(&self) -> bool {
        self.cache().bda_residency()
    }

    pub fn bda_active(&self) -> bool {
        self.cache().bda_active()
    }

    /// [`BufferCache::prepare_unranged_reads`]: speculatively only when it
    /// has nothing to do.
    pub fn prepare_unranged_reads(&mut self, sched: &mut Scheduler) {
        match self {
            Buffers::Exact(b) => {
                b.prepare_unranged_reads(sched);
            }
            Buffers::Spec(b, s) => {
                if !b.unranged_reads_ready() {
                    s.bail(Hole::Upload);
                }
            }
        }
    }

    pub fn bda_make_current(&mut self, sched: &mut Scheduler, addr: u64, size: u64) {
        match self {
            Buffers::Exact(b) => b.bda_make_current(sched, addr, size),
            Buffers::Spec(b, s) => {
                if b.bda_active() {
                    s.bail(Hole::Special);
                }
            }
        }
    }

    pub fn bda_page_table(&mut self, sched: &mut Scheduler) -> Result<(vk::Buffer, u64), vk::Result> {
        match self {
            Buffers::Exact(b) => b.bda_page_table(sched),
            Buffers::Spec(b, s) => b.bda_page_table_existing().ok_or_else(|| {
                s.bail(Hole::Special);
                vk::Result::ERROR_UNKNOWN
            }),
        }
    }

    pub fn bda_imports(&mut self, sched: &mut Scheduler) -> Option<vk::Buffer> {
        match self {
            Buffers::Exact(b) => b.bda_imports(sched),
            Buffers::Spec(b, s) => {
                if b.bda_active() || !b.bda_tried() {
                    s.bail(Hole::Special);
                }
                None
            }
        }
    }

    pub fn bda_capture(&mut self, sched: &mut Scheduler, addr: u64, size: u64, done: impl FnOnce(Vec<u8>) + Send + 'static) {
        if let Buffers::Exact(b) = self {
            b.bda_capture(sched, addr, size, done);
        }
    }

    pub fn download_window(&mut self, sched: &mut Scheduler, addr: u64) -> bool {
        match self {
            Buffers::Exact(b) => b.download_window(sched, addr),
            Buffers::Spec(_, s) => {
                s.bail(Hole::Frontend);
                false
            }
        }
    }
}

/// Image cache access of a draw: exact, or speculative (read-only, with the
/// segment's layout overlay and views).
pub enum Images<'a> {
    Exact(&'a mut ImageCache),
    Spec(&'a ImageCache, &'a SpecCtx),
}

impl<'a> From<&'a mut ImageCache> for Images<'a> {
    fn from(i: &'a mut ImageCache) -> Self {
        Images::Exact(i)
    }
}

impl Images<'_> {
    pub fn cache(&self) -> &ImageCache {
        match self {
            Images::Exact(i) => i,
            Images::Spec(i, _) => i,
        }
    }

    pub fn spec(&self) -> Option<&SpecCtx> {
        match self {
            Images::Exact(_) => None,
            Images::Spec(_, s) => Some(s),
        }
    }

    pub fn exact(&mut self) -> Option<&mut ImageCache> {
        match self {
            Images::Exact(i) => Some(i),
            Images::Spec(..) => None,
        }
    }

    pub fn lookup_stamp(&self) -> (u64, u64) {
        self.cache().lookup_stamp()
    }

    pub fn upload_covered(&self, id: u64) -> bool {
        self.cache().upload_covered(id)
    }

    pub fn pinned_memory_changes(&self) -> u64 {
        self.cache().pinned_memory_changes()
    }

    /// See [`ImageCache::pinned_memory_changed_since`].
    pub fn pinned_memory_changed_since(&self, since: u64, out: &mut Vec<u64>) -> Option<u64> {
        self.cache().pinned_memory_changed_since(since, out)
    }

    pub fn pinned_cursor(&self) -> u64 {
        self.cache().pinned_cursor()
    }

    pub fn pinned_since(&self, cursor: u64) -> Option<&[u64]> {
        self.cache().pinned_since(cursor)
    }

    /// [`ImageCache::pin`] (the bindless set, committing thread only).
    pub fn pin(&mut self, id: u64) {
        match self {
            Images::Exact(i) => i.pin(id),
            Images::Spec(_, s) => s.bail(Hole::Special),
        }
    }

    pub fn lookup_epoch(&self) -> u64 {
        self.cache().lookup_epoch()
    }

    pub fn get(&self, id: u64) -> Option<&CachedImage> {
        self.cache().get(id)
    }

    pub fn needs_upload(&self, id: u64) -> bool {
        self.cache().needs_upload(id)
    }

    pub fn image_generation(&self, id: u64) -> Option<u64> {
        self.cache().image_generation(id)
    }

    pub fn guest_range(&self, id: u64) -> Option<(u64, u64)> {
        self.cache().guest_range(id)
    }

    pub fn find_alias(&self, key: &ImageKey, levels_needed: u32) -> Option<u64> {
        let r = self.cache().find_alias(key, levels_needed);
        if let (Images::Spec(i, s), Some(id)) = (self, r) {
            s.note_image(i, id);
            s.reads.borrow_mut().keys.push((*key, id | 1 << 63));
        }
        r
    }

    pub fn stencil_owner(&self, key: &ImageKey) -> Option<u64> {
        let r = self.cache().stencil_owner(key);
        if let (Images::Spec(i, s), Some(id)) = (self, r) {
            s.note_image(i, id);
        }
        r
    }

    pub fn is_stencil_view(&self, view: vk::ImageView) -> bool {
        match self {
            Images::Exact(i) => i.is_stencil_view(view),
            Images::Spec(i, s) => i.is_stencil_view(view) || s.pending_views.borrow().iter().any(|(p, v)| *v == view && p_stencil(p)),
        }
    }

    pub fn view_subresources(&self, view: vk::ImageView) -> Option<SubRange> {
        match self {
            Images::Exact(i) => i.view_subresources(view),
            Images::Spec(i, s) => i.view_subresources(view).or_else(|| s.pending_views.borrow().iter().find(|(_, v)| *v == view).map(|(p, _)| p.range)),
        }
    }

    pub fn find_or_create(&mut self, sched: &mut Scheduler, buffers: &mut Buffers, key: &ImageKey) -> Result<u64, vk::Result> {
        match self {
            Images::Exact(i) => match buffers.exact() {
                Some(b) => i.find_or_create(sched, b, key),
                None => Err(vk::Result::ERROR_UNKNOWN),
            },
            Images::Spec(i, s) => match i.find_existing(key) {
                Some(id) => {
                    s.note_image(i, id);
                    s.reads.borrow_mut().keys.push((*key, id));
                    Ok(id)
                }
                None => {
                    s.bail(Hole::ImageMiss);
                    Err(vk::Result::ERROR_UNKNOWN)
                }
            },
        }
    }

    pub fn prepare(&mut self, sched: &mut Scheduler, buffers: &mut Buffers, id: u64, discard: bool) -> bool {
        match self {
            Images::Exact(i) => match buffers.exact() {
                Some(b) => i.prepare(sched, b, id, discard),
                None => false,
            },
            Images::Spec(i, s) => {
                if !discard && s.prepare_noop(i, id) {
                    s.note_image(i, id);
                    s.effects.borrow_mut().image_uses.push(id);
                    true
                } else {
                    s.bail(Hole::ImagePrepare);
                    false
                }
            }
        }
    }

    pub fn touch(&mut self, id: u64, tick: u64) {
        match self {
            Images::Exact(i) => i.touch(id, tick),
            Images::Spec(_, s) => s.effects.borrow_mut().image_uses.push(id),
        }
    }

    pub fn mark_gpu_written(&mut self, id: u64, tick: u64) {
        match self {
            Images::Exact(i) => i.mark_gpu_written(id, tick),
            Images::Spec(i, s) => {
                if i.gpu_write_changes(id) {
                    s.bail(Hole::TargetChange);
                } else {
                    s.effects.borrow_mut().gpu_written.push(id);
                }
            }
        }
    }

    pub fn note_storage_write(&mut self, id: u64) {
        match self {
            Images::Exact(i) => i.note_storage_write(id),
            Images::Spec(_, s) => s.bail(Hole::Written),
        }
    }

    pub fn invalidate_shader_write(&mut self, addr: u64, size: u64) {
        match self {
            Images::Exact(i) => i.invalidate_shader_write(addr, size),
            Images::Spec(_, s) => s.bail(Hole::Written),
        }
    }

    pub fn view(&mut self, id: u64, format: vk::Format, aspect: vk::ImageAspectFlags, base_layer: u32, layers: u32) -> Option<vk::ImageView> {
        self.view_level(id, format, aspect, 0, base_layer, layers)
    }

    pub fn view_level(&mut self, id: u64, format: vk::Format, aspect: vk::ImageAspectFlags, level: u32, base_layer: u32, layers: u32) -> Option<vk::ImageView> {
        let ty = if layers > 1 { vk::ImageViewType::TYPE_2D_ARRAY } else { vk::ImageViewType::TYPE_2D };
        let range = ViewRange { ty, base_level: level, levels: 1, base_layer, layers, min_lod: 0.0 };
        self.view_range(id, format, aspect, range, vk::ComponentMapping::default())
    }

    pub fn view_range(&mut self, id: u64, format: vk::Format, aspect: vk::ImageAspectFlags, range: ViewRange, components: vk::ComponentMapping) -> Option<vk::ImageView> {
        match self {
            Images::Exact(i) => i.view_range(id, format, aspect, range, components),
            Images::Spec(i, s) => {
                let plan = i.view_plan(id, format, aspect, range, components)?;
                if let Some(v) = i.find_view(&plan) {
                    return Some(v);
                }
                if let Some(&(_, v)) = s.pending_views.borrow().iter().find(|(p, _)| p.id == plan.id && p.same_key(&plan)) {
                    return Some(v);
                }
                let v = i.create_view(&plan)?;
                let handle = v.view;
                s.pending_views.borrow_mut().push((plan, handle));
                s.effects.borrow_mut().views.push(NewView { plan, view: v });
                Some(handle)
            }
        }
    }

    /// [`ImageCache::transition`] (speculatively against the overlay).
    pub fn transition(&mut self, id: u64, usage: ImageUse, range: Option<SubRange>, out: &mut Vec<vk::ImageMemoryBarrier2<'static>>) {
        match self {
            Images::Exact(i) => i.transition(id, usage, range, out),
            Images::Spec(i, s) => s.overlay.borrow_mut().transition(i, id, usage, range, out),
        }
    }

    /// [`ImageCache::uniform_layout`] (speculatively from the overlay).
    pub fn uniform_layout(&self, id: u64, range: SubRange) -> Option<vk::ImageLayout> {
        match self {
            Images::Exact(i) => i.uniform_layout(id, range),
            // Before the segment uses it: the frozen layout (the decision the
            // committing thread would take; the seam brings the image there).
            Images::Spec(i, s) => s.overlay.borrow().uniform_layout(id, range).or_else(|| if s.overlay.borrow().knows(id) { None } else { i.uniform_layout(id, range) }),
        }
    }
}

fn p_stencil(p: &ViewPlan) -> bool {
    p.is_stencil()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::images::layout::stages_for;

    /// Per-subresource layouts of a tracker (levels x layers).
    fn layouts(t: &LayoutTracker, levels: u32, layers: u32) -> Vec<vk::ImageLayout> {
        (0..levels).flat_map(|l| (0..layers).map(move |y| (l, y))).map(|(l, y)| t.subresource_state(l, y).layout).collect()
    }

    /// Apply barriers to per-subresource layouts, checking each barrier's
    /// old layout is the one the subresource is in.
    fn apply(cur: &mut [vk::ImageLayout], levels: u32, layers: u32, barriers: &[vk::ImageMemoryBarrier2]) -> Result<(), String> {
        for b in barriers {
            let r = b.subresource_range;
            let lc = if r.level_count == vk::REMAINING_MIP_LEVELS { levels - r.base_mip_level } else { r.level_count };
            let yc = if r.layer_count == vk::REMAINING_ARRAY_LAYERS { layers - r.base_array_layer } else { r.layer_count };
            for l in r.base_mip_level..r.base_mip_level + lc {
                for y in r.base_array_layer..r.base_array_layer + yc {
                    let i = (l * layers + y) as usize;
                    if cur[i] != b.old_layout && b.old_layout != vk::ImageLayout::UNDEFINED {
                        return Err(format!("subresource {l}/{y} in {:?}, barrier from {:?}", cur[i], b.old_layout));
                    }
                    cur[i] = b.new_layout;
                }
            }
        }
        Ok(())
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    const USES: [ImageUse; 5] = [crate::images::SAMPLED, crate::images::COLOR_ATTACHMENT, crate::images::STORAGE, crate::images::TRANSFER_DST, crate::images::TRANSFER_SRC];

    fn random_use(rng: &mut Rng, levels: u32, layers: u32) -> (ImageUse, Option<SubRange>) {
        let u = USES[rng.below(USES.len() as u64) as usize];
        let range = (rng.below(2) == 0).then(|| {
            let base_level = rng.below(levels as u64) as u32;
            let base_layer = rng.below(layers as u64) as u32;
            SubRange { base_level, levels: 1 + rng.below((levels - base_level) as u64) as u32, base_layer, layers: 1 + rng.below((layers - base_layer) as u64) as u32 }
        });
        (u, range)
    }

    /// A segment's uses recorded against the overlay, stitched onto the
    /// committed state (requirement barriers, then the segment's barriers),
    /// leave every subresource in the layout the serial tracker leaves, and
    /// every barrier starts from the layout the subresource is in.
    #[test]
    fn overlay_stitching_matches_the_serial_tracker() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for case in 0..400 {
            let (levels, layers) = (1 + rng.below(3) as u32, 1 + rng.below(3) as u32);
            let mut committed = LayoutTracker::new(levels, layers, false);
            let mut scratch = Vec::new();
            // Some history before the segment.
            for _ in 0..rng.below(4) {
                let (u, r) = random_use(&mut rng, levels, layers);
                committed.barriers(vk::Image::null(), vk::ImageAspectFlags::COLOR, u.0, u.1, stages_for(u.1), r, &mut scratch);
            }
            let uses: Vec<(ImageUse, Option<SubRange>)> = (0..1 + rng.below(6)).map(|_| random_use(&mut rng, levels, layers)).collect();
            // Serial.
            let mut serial = committed.clone();
            for &(u, r) in &uses {
                serial.barriers(vk::Image::null(), vk::ImageAspectFlags::COLOR, u.0, u.1, stages_for(u.1), r, &mut scratch);
            }
            // Overlay.
            let mut overlay = LayoutOverlay::default();
            let mut segment = Vec::new();
            for &(u, r) in &uses {
                overlay.transition_with(7, || Some((committed.clone(), vk::Image::null(), vk::ImageAspectFlags::COLOR)), u, r, &mut segment);
            }
            assert!(segment.iter().all(|b| b.old_layout != crate::images::layout::UNKNOWN), "case {case}: a barrier from the unknown state was recorded");
            let o = &overlay.images[0];
            let mut stitched = committed.clone();
            let mut seam = Vec::new();
            for &(range, (layout, access), stage) in &o.requirements {
                stitched.barriers(vk::Image::null(), vk::ImageAspectFlags::COLOR, layout, access, stage, range, &mut seam);
            }
            // The GPU sees: the seam, then the segment's barriers.
            let mut cur = layouts(&committed, levels, layers);
            apply(&mut cur, levels, layers, &seam).unwrap_or_else(|e| panic!("case {case}: seam: {e}"));
            apply(&mut cur, levels, layers, &segment).unwrap_or_else(|e| panic!("case {case}: segment: {e}"));
            stitched.adopt(&o.tracker);
            assert_eq!(cur, layouts(&serial, levels, layers), "case {case}: layouts on the GPU");
            assert_eq!(layouts(&stitched, levels, layers), layouts(&serial, levels, layers), "case {case}: tracked layouts after the commit");
        }
    }

    #[test]
    fn unknown_subresources_give_no_uniform_layout() {
        let mut overlay = LayoutOverlay::default();
        let t = LayoutTracker::new(2, 2, false);
        let mut out = Vec::new();
        let r = SubRange { base_level: 0, levels: 1, base_layer: 0, layers: 2 };
        overlay.transition_with(1, || Some((t.clone(), vk::Image::null(), vk::ImageAspectFlags::COLOR)), crate::images::SAMPLED, Some(r), &mut out);
        assert!(out.is_empty(), "first uses are requirements");
        assert_eq!(overlay.uniform_layout(1, r), Some(crate::images::SAMPLED.0));
        assert_eq!(overlay.uniform_layout(1, SubRange { base_level: 1, levels: 1, base_layer: 0, layers: 1 }), None);
        assert_eq!(overlay.uniform_layout(2, r), None);
        assert!(overlay.knows(1) && !overlay.knows(2));
    }
}
