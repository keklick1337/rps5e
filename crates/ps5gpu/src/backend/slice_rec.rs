// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Slice recording and commit, the back-end side (R7b stage 2/3).
//!
//! A slice thread records the draws of one slice against the frozen
//! back-end ([`Backend::record_slice`], `&self` only) into its own
//! scheduler, through its own pipeline provider and speculative cache
//! access. The output is a list of units in order: segments (one command
//! buffer each, with the reads of every draw, the journaled effects and the
//! layout overlay) and exact items (holes and events the committing thread
//! runs at their positions). The committing thread checks each segment's
//! reads against the committed state and splices it, or records its items
//! exactly ([`Backend::commit_segment`]).

use super::*;
use crate::images::ImageKey;
use crate::slices::spec::{Buffers, DrawReads, Effects, Hole, Images, LayoutOverlay, MetaCheck, SpecCtx};
use std::any::Any;

/// One item of a slice, by its position in the work stream.
pub enum SliceItem<'a> {
    Draw(usize, &'a ResolvedDraw),
    /// A guest memory write of the CPU at this position: `(address, size,
    /// bytes)` (patches, zero fills, write faults). It records nothing on
    /// the GPU.
    Cpu(usize, u64, u64, CpuBytes<'a>),
    /// Work the committing thread runs exactly at this position (it ends
    /// the open segment).
    Exact(usize),
}

/// What a CPU write puts into guest memory.
#[derive(Clone, Copy, Debug)]
pub enum CpuBytes<'a> {
    Data(&'a [u8]),
    Zero,
    /// A write fault: the page becomes CPU-dirty, its bytes as they are
    /// (the bytes follow as patches).
    Fault,
    /// Bytes not known ahead (a command-processor transfer run on the CPU).
    Unknown,
}

impl CpuBytes<'_> {
    pub fn known(&self) -> bool {
        !matches!(self, CpuBytes::Unknown)
    }
}

/// A CPU write a slice's draws see composed (see
/// [`crate::memory::pending`]): `(address, size, bytes known)`.
pub type PriorWrite = (u64, u64, bool);

/// A draw recorded speculatively.
pub struct SegDraw {
    pub pos: usize,
    pub reads: DrawReads,
    /// `RPS5E_SLICE_VERIFY`: its decision digest.
    pub digest: u64,
    pub recorded: bool,
}

pub enum SegItem {
    Draw(SegDraw),
    /// (position, address, size, bytes known).
    Cpu(usize, u64, u64, bool),
}

/// A speculative segment of a slice.
pub struct SegmentOut {
    /// The worker that recorded it (its recorder gets the buffers back).
    pub worker: usize,
    pub cbs: Vec<vk::CommandBuffer>,
    pub items: Vec<SegItem>,
    pub effects: Effects,
    pub overlay: LayoutOverlay,
    pub retired: Vec<Box<dyn Any + Send>>,
    /// A draw that became a hole recorded commands into the buffer first:
    /// the segment is recorded exactly.
    pub poisoned: bool,
    /// The provisional tick it was recorded at.
    pub tick: u64,
    /// The provider's pipeline swaps when it was recorded.
    pub swaps: u64,
}

impl SegmentOut {
    pub fn draws(&self) -> usize {
        self.items.iter().filter(|i| matches!(i, SegItem::Draw(_))).count()
    }

    /// Positions of the segment's items, in order.
    pub fn positions(&self) -> impl Iterator<Item = usize> + '_ {
        self.items.iter().map(|i| match i {
            SegItem::Draw(d) => d.pos,
            SegItem::Cpu(p, ..) => *p,
        })
    }
}

pub enum SliceUnit {
    Segment(SegmentOut),
    /// Recorded exactly at the position (a hole, with why, or an event).
    Exact(usize, Option<Hole>),
}

/// The state of one slice thread: its scheduler, provider, speculative
/// access and target-key memos.
pub struct SliceRecorder {
    pub worker: usize,
    sched: Scheduler,
    pipelines: Box<dyn PipelineProvider>,
    spec: SpecCtx,
    color_keys: Vec<(ColorTarget, ImageKey)>,
    depth_keys: Vec<(DepthTarget, ImageKey)>,
    /// Targets an earlier hole of the slice writes or prepares: later draws
    /// of the slice take them as current and newest (checked at commit).
    assumed: Vec<u64>,
    /// HTILE slices an earlier hole of the slice cleared (consumed).
    assumed_htile: Vec<(u64, u32)>,
    /// The CPU writes of wave `pending_wave` applied so far (its first
    /// `pending_applied`): the slices of a wave come in order.
    pending: crate::memory::pending::Pending,
    pending_wave: u64,
    pending_applied: usize,
}

/// Add a CPU write of the wave to what later draws of the slice see.
fn add_write(p: &mut crate::memory::pending::Pending, a: u64, n: u64, b: &CpuBytes) {
    match b {
        CpuBytes::Data(d) => p.push_bytes(a, n, Some(d)),
        CpuBytes::Zero => p.push_bytes(a, n, None),
        CpuBytes::Fault | CpuBytes::Unknown => p.reach(a, n),
    }
}

impl SliceRecorder {
    /// The slice provider's counters (memo hits, recipes, lookups).
    pub fn summary(&self) -> Option<String> {
        self.pipelines.summary()
    }
}

// SAFETY: the recorder is used by one thread at a time (its worker while a
// wave records, the committing thread between waves); its scheduler and
// provider are Send, the speculative state is only touched by that thread.
unsafe impl Send for SliceRecorder {}

/// `RPS5E_SLICE_UPLOADS=0`: a slice draw that binds a mirror with CPU-dirty
/// pages is a hole; otherwise the upload is made at commit, before the
/// segment (stage 3: the draw records speculatively, the bytes it needs
/// are the ones its read set holds).
fn speculative_uploads() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_SLICE_UPLOADS").is_ok_and(|v| v == "0"))
}

impl Backend {
    /// A recorder for slice thread `worker` (None: the provider has no
    /// slice instances).
    pub fn slice_recorder(&self, worker: usize) -> Option<SliceRecorder> {
        let pipelines = self.pipelines.slice_provider()?;
        let sched = Scheduler::new_slice(&self.sched).ok()?;
        let spec = SpecCtx::new(self.memory);
        spec.set_uploads(speculative_uploads());
        Some(SliceRecorder { worker, sched, pipelines, spec, color_keys: Vec::new(), depth_keys: Vec::new(), assumed: Vec::new(), assumed_htile: Vec::new(), pending: Default::default(), pending_wave: 0, pending_applied: 0 })
    }

    /// Slices may record now: no occlusion query is open, no capture
    /// records, no frame trace or per-item submits run, and the provider
    /// set up what its slice instances share.
    pub fn slices_allowed(&self) -> bool {
        !self.sched.occlusion_active() && !crate::capture::active() && !crate::pipeline::trace::active() && !submit_each() && self.pipelines.slice_ready()
    }

    /// Record `items` speculatively (a slice thread, against the frozen
    /// back-end).
    /// `prefix`: the CPU writes of wave `wave` before the slice (a
    /// recorder keeps those it applied for the wave's later slices).
    pub fn record_slice(&self, rec: &mut SliceRecorder, wave: u64, prefix: &[(u64, u64, CpuBytes)], items: &[SliceItem]) -> Vec<SliceUnit> {
        // The wave's CPU writes before each draw, composed into its view of
        // guest memory.
        rec.spec.begin_wave(wave);
        let mut pending = std::mem::take(&mut rec.pending);
        if rec.pending_wave != wave || rec.pending_applied > prefix.len() {
            pending.clear();
            rec.pending_applied = 0;
            rec.pending_wave = wave;
        }
        for (a, n, b) in &prefix[rec.pending_applied..] {
            add_write(&mut pending, *a, *n, b);
        }
        let (units, own) = self.record_slice_inner(rec, items, &mut pending);
        crate::memory::pending::clear();
        rec.pending_applied = prefix.len() + own;
        rec.pending = pending;
        units
    }

    /// The units, and the slice's own CPU writes applied to `pending`.
    fn record_slice_inner(&self, rec: &mut SliceRecorder, items: &[SliceItem], pending: &mut crate::memory::pending::Pending) -> (Vec<SliceUnit>, usize) {
        let mut own = 0;
        let mut units = Vec::new();
        rec.assumed.clear();
        rec.assumed_htile.clear();
        let mut seg = self.open_segment(rec);
        for item in items {
            match *item {
                SliceItem::Draw(pos, d) => {
                    crate::memory::pending::set(pending);
                    rec.spec.begin_draw();
                    let mark = rec.sched.recorded_count();
                    let r = self.draw_spec(rec, d);
                    let asked = rec.sched.take_slice_bail();
                    let r = match r {
                        Ok(recorded) if !asked => Ok(recorded),
                        Ok(_) => Err(Hole::Submit),
                        Err(h) => Err(h),
                    };
                    match r {
                        Ok(recorded) => {
                            let reads = std::mem::take(&mut *rec.spec.reads.borrow_mut());
                            let digest = rec.pipelines.take_digest();
                            seg.items.push(SegItem::Draw(SegDraw { pos, reads, digest, recorded }));
                        }
                        Err(h) => {
                            if rec.sched.recorded_count() != mark {
                                seg.poisoned = true;
                            }
                            rec.pipelines.take_digest();
                            self.close_segment(rec, &mut seg, &mut units);
                            units.push(SliceUnit::Exact(pos, Some(h)));
                        }
                    }
                }
                SliceItem::Cpu(pos, a, n, ref b) => {
                    add_write(pending, a, n, b);
                    own += 1;
                    if seg.items.is_empty() {
                        units.push(SliceUnit::Exact(pos, None));
                    } else {
                        seg.items.push(SegItem::Cpu(pos, a, n, b.known()));
                    }
                }
                SliceItem::Exact(pos) => {
                    self.close_segment(rec, &mut seg, &mut units);
                    units.push(SliceUnit::Exact(pos, None));
                }
            }
        }
        self.close_segment(rec, &mut seg, &mut units);
        (units, own)
    }

    fn open_segment(&self, rec: &SliceRecorder) -> SegmentOut {
        rec.spec.begin_segment();
        SegmentOut {
            worker: rec.worker,
            cbs: Vec::new(),
            items: Vec::new(),
            effects: Effects::default(),
            overlay: LayoutOverlay::default(),
            retired: Vec::new(),
            poisoned: false,
            tick: rec.sched.current_tick(),
            swaps: self.pipelines.swap_generation(),
        }
    }

    /// End the open segment: emitted when it holds draws (or commands),
    /// else its CPU items become exact items; a new one opens.
    fn close_segment(&self, rec: &mut SliceRecorder, seg: &mut SegmentOut, units: &mut Vec<SliceUnit>) {
        let next = self.open_segment_keep(rec, seg);
        let mut done = std::mem::replace(seg, next);
        let (prologue, cb) = rec.sched.end_segment();
        done.cbs.extend(prologue.iter().chain(cb.iter()).copied());
        done.effects = std::mem::take(&mut *rec.spec.effects.borrow_mut());
        done.overlay = std::mem::take(&mut *rec.spec.overlay.borrow_mut());
        done.retired = rec.sched.take_retired();
        rec.spec.begin_segment();
        if done.draws() == 0 && !done.poisoned && done.cbs.is_empty() && done.effects.views.is_empty() {
            for p in done.positions() {
                units.push(SliceUnit::Exact(p, None));
            }
            return;
        }
        units.push(SliceUnit::Segment(done));
    }

    fn open_segment_keep(&self, rec: &SliceRecorder, _seg: &SegmentOut) -> SegmentOut {
        SegmentOut {
            worker: rec.worker,
            cbs: Vec::new(),
            items: Vec::new(),
            effects: Effects::default(),
            overlay: LayoutOverlay::default(),
            retired: Vec::new(),
            poisoned: false,
            tick: rec.sched.current_tick(),
            swaps: self.pipelines.swap_generation(),
        }
    }

    /// The draw would copy depth/stencil planes instead of drawing.
    fn depth_copy_applies(d: &ResolvedDraw) -> bool {
        let Some(z) = d.depth else { return false };
        // A DB depth copy into colour target 0 is the committing thread's.
        if super::db_depth_copy_enabled() && d.db_depth_copy().is_some() {
            return true;
        }
        if d.color_mode != color_mode::DISABLE {
            return false;
        }
        let (depth_flag, stencil_flag) = d.depth_state.depth_copy();
        let depth = depth_flag && z.z_format != crate::render::formats::depth::INVALID && z.z_read_base != 0 && z.z_write_base != 0 && z.z_read_base != z.z_write_base;
        let stencil = stencil_flag && z.has_stencil && z.stencil_read_base != 0 && z.stencil_write_base != 0 && z.stencil_read_base != z.stencil_write_base;
        depth || stencil
    }

    /// One draw on a slice thread ([`Self::draw_inner`] against the frozen
    /// state). Ok(recorded) or the hole it needs.
    fn draw_spec(&self, rec: &mut SliceRecorder, d: &ResolvedDraw) -> Result<bool, Hole> {
        match d.color_mode {
            color_mode::ELIMINATE_FAST_CLEAR | color_mode::FMASK_DECOMPRESS | color_mode::DCC_DECOMPRESS | color_mode::RESOLVE => return Err(Hole::Special),
            _ => {}
        }
        if Self::depth_copy_applies(d) || skipped_shader(&[d.shaders.vertex_program().address, d.shaders.ps.address]) {
            return Err(Hole::Special);
        }
        let targets = self.bind_targets_spec(rec, d)?;
        // A first write that changes lookups is the committing thread's.
        let mut change = false;
        for &(_, id) in &targets.colors {
            if !rec.assumed.contains(&id) && self.images.gpu_write_changes(id) {
                rec.assumed.push(id);
                change = true;
            }
        }
        let depth_written = targets.depth.and_then(|(z, id)| {
            let has_depth = z.z_format != crate::render::formats::depth::INVALID;
            let (depth, stencil) = crate::pipeline::state::depth_attachment_writes(d, has_depth, z.has_stencil);
            (depth || stencil).then_some(id)
        });
        if let Some(id) = depth_written.filter(|&id| !rec.assumed.contains(&id) && self.images.gpu_write_changes(id)) {
            rec.assumed.push(id);
            change = true;
        }
        if change {
            return Err(Hole::TargetChange);
        }
        let outcome = {
            let mut ctx = DrawContext {
                dev: &self.dev,
                sched: &mut rec.sched,
                buffers: Buffers::Spec(&self.buffers, &rec.spec),
                images: Images::Spec(&self.images, &rec.spec),
                targets: &targets,
            };
            rec.pipelines.draw(&mut ctx, d)
        };
        if let Some(h) = rec.spec.bailed() {
            return Err(h);
        }
        match outcome {
            DrawOutcome::Hole => Err(Hole::Other),
            DrawOutcome::Recorded => {
                let mut e = rec.spec.effects.borrow_mut();
                e.gpu_written.extend(targets.colors.iter().map(|&(_, id)| id));
                e.gpu_written.extend(depth_written);
                Ok(true)
            }
            DrawOutcome::Skipped | DrawOutcome::NeedsShader => Ok(false),
        }
    }

    /// [`Self::bind_targets`] against the frozen state: the images must
    /// exist and be current, no fast clear may be pending.
    fn bind_targets_spec(&self, rec: &mut SliceRecorder, d: &ResolvedDraw) -> Result<DrawTargets, Hole> {
        let mut targets = DrawTargets { colors: Default::default(), depth: None };
        let spec = &rec.spec;
        if d.color_mode != color_mode::DISABLE {
            for t in &d.color_targets {
                let key = recent_key(&mut rec.color_keys, t, color_key);
                let id = self.images.find_existing(&key).ok_or(Hole::ImageMiss)?;
                spec.note_target(&self.images, id);
                spec.reads.borrow_mut().keys.push((key, id));
                {
                    let mut e = spec.effects.borrow_mut();
                    if t.dcc_address != 0 && self.dcc_targets.get(&t.dcc_address) != Some(t) {
                        e.dcc_targets.push((t.dcc_address, *t));
                    }
                    if let Some((base, _, true)) = t.clear_metadata() {
                        if !self.cmask.contains(&base) {
                            e.cmask.push(base);
                        }
                        if self.dcc_targets.get(&base) != Some(t) {
                            e.dcc_targets.push((base, *t));
                        }
                    }
                }
                self.dcc_spec(spec, t)?;
                if !rec.assumed.contains(&id) && !rec.spec.prepare_noop(&self.images, id) {
                    rec.assumed.push(id);
                    return Err(Hole::ImagePrepare);
                }
                spec.effects.borrow_mut().image_uses.push(id);
                targets.colors.push((*t, id));
            }
        }
        if let Some(z) = d.depth {
            let key = recent_key(&mut rec.depth_keys, &z, depth_key);
            let id = self.images.find_existing(&key).ok_or(Hole::ImageMiss)?;
            spec.note_target(&self.images, id);
            spec.reads.borrow_mut().keys.push((key, id));
            if z.htile_enabled {
                let slice_bytes = crate::backend::htile_metadata_bytes(z.width, z.height);
                spec.reads.borrow_mut().meta.push(MetaCheck::Htile(z.htile_base, z.base_layer, slice_bytes));
                if !self.htile.contains(&z.htile_base) {
                    spec.effects.borrow_mut().htile.push(z.htile_base);
                }
                let slice = (z.htile_base, z.base_layer);
                if !rec.assumed_htile.contains(&slice) && !self.htile_clean(z.htile_base, z.base_layer, slice_bytes) {
                    rec.assumed_htile.push(slice);
                    return Err(Hole::ClearHtile);
                }
            }
            let ds = &d.depth_state;
            if ds.depth_clear_enable || (z.has_stencil && ds.stencil_clear_enable && !ds.stencil_write_disable) {
                return Err(Hole::ClearDepth);
            }
            if !rec.assumed.contains(&id) && !rec.spec.prepare_noop(&self.images, id) {
                rec.assumed.push(id);
                return Err(Hole::ImagePrepare);
            }
            spec.effects.borrow_mut().image_uses.push(id);
            targets.depth = Some((z, id));
        }
        crate::backend::drop_mixed_sample_targets(&mut targets, d);
        Ok(targets)
    }

    /// No HTILE fill waits at `base` and slice `layer` has no clear bits:
    /// a bind clears nothing ([`Self::htile_slice_cleared`]).
    fn htile_clean(&self, base: u64, layer: u32, slice_bytes: u64) -> bool {
        !self.images.has_fill(base)
            && (layer == 0 || !self.images.has_fill(base + layer as u64 * slice_bytes))
            && !self.htile_kernel_written.contains_key(&base)
            && self.htile_cleared.get(&base).is_none_or(|(depth, stencil)| !depth.test(layer) && !stencil.test(layer))
    }

    /// The fast-clear decision of [`Self::pending_dcc_clears`], read-only:
    /// Ok when no slice of `t` is cleared (a clear, a pending fill, an
    /// overwrite flag or GPU-written clear codes are holes).
    fn dcc_spec(&self, spec: &SpecCtx, t: &ColorTarget) -> Result<(), Hole> {
        let Some((base, slice, cmask)) = t.clear_metadata() else { return Ok(()) };
        spec.reads.borrow_mut().meta.push(MetaCheck::Dcc(base));
        if self.images.has_fill(base) {
            return Err(Hole::ClearFill);
        }
        match self.dcc.get(&base) {
            Some(true) => return Err(Hole::ClearFlag),
            Some(false) => {}
            None => spec.effects.borrow_mut().dcc_reset.push(base),
        }
        for layer in t.base_layer..t.base_layer + t.layers.max(1) {
            let addr = base + layer as u64 * slice;
            let head_bytes = slice.min(64);
            if self.buffers.has_gpu_dirty(addr, head_bytes) {
                continue;
            }
            spec.note_pages(&self.tracker, addr, head_bytes);
            let mut head = [0u8; 64];
            let head = &mut head[..head_bytes as usize];
            if !self.memory.read_backing(addr, head) || !ColorTarget::is_clear_code(head[0], cmask) || head.iter().any(|&x| x != head[0]) {
                continue;
            }
            if self.buffers.has_gpu_dirty(addr, slice) {
                return Err(Hole::Clear);
            }
            spec.note_pages(&self.tracker, addr, slice);
            let mut bytes = vec![0u8; slice as usize];
            if !self.memory.read_backing(addr, &mut bytes) {
                continue;
            }
            let b = bytes[0];
            if ColorTarget::is_clear_code(b, cmask) && bytes.iter().all(|&x| x == b) && t.metadata_clear(b, cmask).is_some() {
                return Err(Hole::Clear);
            }
        }
        Ok(())
    }

    /// A speculative draw's reads still give what they gave (else which
    /// kind of read moved).
    fn reads_current(&self, r: &DrawReads) -> Result<(), &'static str> {
        if !r.mirrors.iter().all(|&(a, n, k, serial)| matches!(self.buffers.peek_read(a, n), Ok((_, k2, s2)) if k2 == k && s2 == serial)) {
            return Err("mirror");
        }
        if !r.uploads.iter().all(|&(a, n, k, serial)| self.buffers.find_containing(a, n) == Some(k) && self.buffers.get(k).is_some_and(|b| b.serial == serial)) {
            return Err("upload");
        }
        if !r.images.iter().all(|&(id, g)| self.images.image_generation(id) == Some(g) && !self.images.needs_upload(id)) {
            return Err("image");
        }
        if !r.targets.iter().all(|&id| self.images.prepare_noop(id, false)) {
            return Err("target");
        }
        if !r.keys.iter().all(|(key, id)| id & (1 << 63) != 0 || self.images.find_existing(key) == Some(*id)) {
            return Err("key");
        }
        Ok(())
    }

    /// Whether `seg` is what recording its draws at their positions now
    /// would give: every read still holds, no CPU write inside it reaches a
    /// later draw's reads (`changing[i]`: the segment's i-th CPU item
    /// changes memory), metadata state unchanged where read, its target
    /// writes still change no lookup.
    fn segment_valid(&self, seg: &SegmentOut, changing: &[bool], prior: &[PriorWrite], writes_known: bool) -> Result<(), &'static str> {
        if seg.poisoned {
            return Err("poisoned");
        }
        if seg.swaps != self.pipelines.swap_generation() {
            return Err("pipeline");
        }
        // CPU writes of the segment before each draw (run after it is
        // spliced): (address, size, bytes known).
        let mut cpu: smallvec::SmallVec<[PriorWrite; 4]> = smallvec::SmallVec::new();
        let mut ci = 0;
        for item in &seg.items {
            match item {
                SegItem::Cpu(_, a, n, known) => {
                    if changing.get(ci).copied().unwrap_or(true) {
                        cpu.push((*a, *n, *known));
                    }
                    ci += 1;
                }
                SegItem::Draw(dr) => {
                    let r = &dr.reads;
                    // Unknown bytes reach no read; the segment's own known
                    // writes reach no read of the exact state (mirrors,
                    // uploads, images: they run after the segment's uploads),
                    // only composed byte reads.
                    for &(a, n, known) in prior.iter().chain(cpu.iter()) {
                        if !known && r.overlaps(a, n) {
                            return Err("cpu");
                        }
                    }
                    for &(a, n, _) in cpu.iter() {
                        let end = a.saturating_add(n);
                        if r.ranges.iter().any(|&(b, m)| b < end && a < b + m) {
                            return Err("cpu");
                        }
                    }
                    for m in &r.meta {
                        let ok = match *m {
                            MetaCheck::Dcc(base) => !self.images.has_fill(base) && self.dcc.get(&base) != Some(&true),
                            MetaCheck::Htile(base, layer, slice_bytes) => self.htile_clean(base, layer, slice_bytes),
                        };
                        if !ok {
                            return Err("metadata");
                        }
                    }
                    // Pages the known writes reached moved by them (composed
                    // into the reads): they only must not hold GPU data.
                    let composed = |p: u64| prior.iter().chain(cpu.iter()).any(|&(a, n, known)| known && a < p + crate::memory::PAGE_BYTES && p < a.saturating_add(n));
                    r.pages.current_except(&self.tracker, self.memory, &composed, writes_known)?;
                    self.reads_current(r)?;
                }
            }
        }
        let mut seen: smallvec::SmallVec<[u64; 8]> = smallvec::SmallVec::new();
        for &id in &seg.effects.gpu_written {
            if !seen.contains(&id) {
                seen.push(id);
                if self.images.gpu_write_changes(id) {
                    return Err("write");
                }
            }
        }
        if seg.overlay.images.iter().any(|o| self.images.layout_of(o.id).is_none()) {
            return Err("gone");
        }
        Ok(())
    }

    /// Commit a speculative segment at the current position: checked,
    /// then its uploads are made, the seam recorded and the segment
    /// spliced, its effects applied with the real tick. Err(why) when it
    /// must be recorded exactly instead (the caller runs its items in
    /// order); its buffers are freed, its views kept.
    /// `prior`: guest memory the CPU (or a command-processor transfer run
    /// on the CPU) wrote since the wave was recorded, before this segment:
    /// writes to CPU-dirty pages move no page generation.
    /// `writes_known`: `prior` and the segment's CPU items are every CPU
    /// write since the wave was recorded (the capture replay).
    #[allow(clippy::too_many_arguments)]
    pub fn commit_segment(&mut self, rec: &mut SliceRecorder, seg: &mut SegmentOut, changing: &[bool], prior: &[PriorWrite], writes_known: bool, verify: bool) -> Result<(), &'static str> {
        let valid = self.segment_valid(seg, changing, prior, writes_known);
        if verify || valid.is_err() {
            self.drop_segment(rec, seg);
            return valid.and(Err("verify"));
        }
        // Stage 3: the mirrors the draws bound with CPU-dirty pages get
        // the bytes the draws' read sets hold, before the segment.
        for item in &seg.items {
            if let SegItem::Draw(dr) = item {
                for &(a, n, _, _) in &dr.reads.uploads {
                    if let Err(e) = self.buffers.obtain(&mut self.sched, a, n, false) {
                        log::warn!("ps5gpu: slice upload {a:#x}+{n:#x}: {e:?}");
                    }
                }
            }
        }
        let mut barriers = Vec::new();
        for o in &seg.overlay.images {
            self.images.apply_layouts(o.id, &o.requirements, &o.tracker, &mut barriers);
        }
        let tick = self.sched.current_tick();
        let cbs = std::mem::take(&mut seg.cbs);
        self.sched.splice(&barriers, &cbs);
        rec.sched.release(&cbs, tick);
        rec.pipelines.retag_tick(seg.tick, tick);
        let e = std::mem::take(&mut seg.effects);
        for &(k, a, n) in &e.buffer_uses {
            self.buffers.note_used(k, a, n, tick);
        }
        for &id in &e.image_uses {
            self.images.touch(id, tick);
        }
        for &id in &e.gpu_written {
            self.images.mark_gpu_written(id, tick);
        }
        self.adopt_views(e.views);
        for &(a, t) in &e.dcc_targets {
            self.dcc_targets.insert(a, t);
        }
        for &b in &e.cmask {
            self.cmask.insert(b);
        }
        for &b in &e.htile {
            self.htile.insert(b);
        }
        for &b in &e.dcc_reset {
            self.dcc.entry(b).or_insert(false);
        }
        for o in std::mem::take(&mut seg.retired) {
            self.sched.retire(o);
        }
        let draws = seg.draws() as u64;
        let recorded = seg.items.iter().filter(|i| matches!(i, SegItem::Draw(d) if d.recorded)).count() as u64;
        self.stats.draws += draws;
        self.stats.recorded_draws += recorded;
        crate::perf::DRAWS.fetch_add(draws, Ordering::Relaxed);
        self.housekeeping();
        if recorded != 0 {
            self.early_submit(false);
        }
        Ok(())
    }

    /// A segment recorded exactly instead: its buffers are free again, its
    /// views are cached (the slice thread's memos may hold them) and what
    /// it retired is kept until the open tick completes.
    fn drop_segment(&mut self, rec: &mut SliceRecorder, seg: &mut SegmentOut) {
        let cbs = std::mem::take(&mut seg.cbs);
        rec.sched.release(&cbs, 0);
        let e = std::mem::take(&mut seg.effects);
        self.adopt_views(e.views);
        for o in std::mem::take(&mut seg.retired) {
            self.sched.retire(o);
        }
    }

    fn adopt_views(&mut self, views: Vec<crate::slices::spec::NewView>) {
        for v in views {
            self.images.adopt_view(v.plan.id, v.plan, v.view);
        }
        let orphans = self.images.take_orphan_views();
        if !orphans.is_empty() {
            self.sched.retire(orphans);
        }
    }

    /// `RPS5E_SLICE_VERIFY`: the decision digest of the last exact draw.
    pub fn take_digest(&mut self) -> u64 {
        self.pipelines.take_digest()
    }
}
