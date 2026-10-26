// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest occlusion queries counted with Vulkan occlusion queries.
//!
//! A guest occlusion query is a pair of PIXEL_PIPE_STAT_DUMP events
//! (EVENT_WRITE ZPASS_DONE) into one 256-byte block: the begin dump writes
//! each depth block's sample counter to `block + 16 * db`, the end dump to
//! `block + 16 * db + 8`, and the guest sums `end - begin` over the 16 depth
//! blocks once every slot carries the ready bit (bit 63).
//!
//! The host brackets the draws between the dumps with precise Vulkan
//! occlusion queries. Vulkan queries cannot span rendering scopes, so a
//! query is split at every scope boundary, and segment `k` is published as
//! depth block `k`'s count: the guest's own sum adds the segments. The
//! results are copied into a host-visible buffer after the scope that ended
//! the query (never inside one) and written to guest memory once the GPU
//! retired them, as a command-processor memory write. More than 16
//! segments, an end without its begin or an exhausted pool publish a
//! conservative visible result.
//!
//! Opt-in with `RPS5E_OCCLUSION_QUERIES=1`: SILENT HILL's title-menu
//! character is intermittently reported occluded and culled for seconds at
//! a time with real counts. Without it (or on a device without precise
//! occlusion queries or host query reset) the command processor publishes a
//! synthetic always-visible result.

use super::Scheduler;
use crate::device::{Buffer, GpuDevice, MemoryUse};
use crate::render::OcclusionWrite;
use ash::vk;
use std::sync::{Arc, Mutex, OnceLock};

/// Bytes of a guest occlusion result block.
pub const BLOCK_BYTES: usize = 256;
/// Depth blocks (begin/end counter pairs) of a result block.
pub const DEPTH_BLOCKS: usize = 16;
/// Set in every counter slot once its value is valid.
pub const READY_BIT: u64 = 1 << 63;
/// Reported when the samples could not be counted, so the guest keeps the
/// object visible.
pub const CONSERVATIVE_COUNT: u64 = 0x00FF_FFFF;
/// Host queries in the pool (segments in flight at once).
const POOL_SIZE: u32 = 4096;

/// `RPS5E_OCCLUSION_QUERIES=1`: count guest occlusion queries on the GPU.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_enabled(std::env::var("RPS5E_OCCLUSION_QUERIES").ok().as_deref()))
}

fn parse_enabled(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim() == "1")
}

/// A counter dump: the begin or end of the query of a result block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dump {
    Begin(u64),
    End(u64),
}

/// Which side of which query a dump to `addr` is (None: not a qword-aligned
/// address).
pub fn classify(addr: u64) -> Option<Dump> {
    if addr == 0 || addr & 7 != 0 {
        return None;
    }
    Some(if addr & 0xF == 0 { Dump::Begin(addr) } else { Dump::End(addr - 8) })
}

/// The result block of a query counted in `segments` host segments with
/// `counts[k]` samples in segment `k` (a conservative query reports
/// [`CONSERVATIVE_COUNT`] samples). None with more segments than depth
/// blocks or fewer counts than segments.
pub fn build_block(segments: usize, conservative: bool, counts: &[u64]) -> Option<[u8; BLOCK_BYTES]> {
    if segments > DEPTH_BLOCKS || (!conservative && counts.len() < segments) {
        return None;
    }
    let mut slots = [READY_BIT; DEPTH_BLOCKS * 2];
    if conservative {
        slots[1] |= CONSERVATIVE_COUNT;
    } else {
        for (segment, count) in counts[..segments].iter().enumerate() {
            slots[segment * 2 + 1] = READY_BIT | (count & (READY_BIT - 1));
        }
    }
    let mut block = [0u8; BLOCK_BYTES];
    for (bytes, slot) in block.chunks_exact_mut(8).zip(slots) {
        bytes.copy_from_slice(&slot.to_le_bytes());
    }
    Some(block)
}

/// A guest query: its result block and the host queries counting it.
struct Query {
    block: u64,
    segments: Vec<u32>,
    /// More scopes than depth blocks, no free host query, or an end without
    /// its begin: publish a visible result.
    conservative: bool,
    write: OcclusionWrite,
}

/// Query bookkeeping (no Vulkan calls): the open guest query, its active
/// segment, the ended queries waiting to be published and the free host
/// queries (returned once their results were read).
struct Tracker {
    free: Arc<Mutex<Vec<u32>>>,
    open: Option<Query>,
    active: Option<u32>,
    pending: Vec<Query>,
}

impl Tracker {
    fn new(pool_size: u32) -> Tracker {
        Tracker { free: Arc::new(Mutex::new((0..pool_size).rev().collect())), open: None, active: None, pending: Vec::new() }
    }

    fn open_block(&self) -> Option<u64> {
        self.open.as_ref().map(|q| q.block)
    }

    /// Open the query of `block` (the previous one was closed).
    fn open(&mut self, block: u64, write: &OcclusionWrite) {
        debug_assert!(self.open.is_none());
        self.open = Some(Query { block, segments: Vec::new(), conservative: false, write: write.clone() });
    }

    /// A rendering scope began: the host query to begin, if any.
    fn begin_segment(&mut self) -> Option<u32> {
        let query = self.open.as_mut()?;
        if self.active.is_some() {
            return None;
        }
        let index = if query.segments.len() < DEPTH_BLOCKS { self.free.lock().unwrap_or_else(|e| e.into_inner()).pop() } else { None };
        let Some(index) = index else {
            query.conservative = true;
            return None;
        };
        query.segments.push(index);
        self.active = Some(index);
        Some(index)
    }

    /// A rendering scope ends: the host query to end, if any.
    fn end_segment(&mut self) -> Option<u32> {
        self.active.take()
    }

    /// The end dump of `block`: the query is ready to publish. Returns the
    /// active host query to end.
    fn close(&mut self, block: u64, write: &OcclusionWrite) -> Option<u32> {
        match self.open.take() {
            Some(query) if query.block == block => {
                self.pending.push(query);
                self.end_segment()
            }
            other => {
                // An end without its begin: nothing was counted.
                self.open = other;
                self.pending.push(Query { block, segments: Vec::new(), conservative: true, write: write.clone() });
                None
            }
        }
    }
}

/// The host query pool and the buffer its results are copied into (one
/// qword per host query).
pub(super) struct Occlusion {
    dev: Arc<GpuDevice>,
    pool: vk::QueryPool,
    results: Arc<Buffer>,
    tracker: Tracker,
}

impl Occlusion {
    fn new(dev: &Arc<GpuDevice>) -> Result<Occlusion, vk::Result> {
        let results = Arc::new(Buffer::new(dev, POOL_SIZE as u64 * 8, vk::BufferUsageFlags::TRANSFER_DST, MemoryUse::Download)?);
        let info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::OCCLUSION).query_count(POOL_SIZE);
        // SAFETY: valid create info.
        let pool = unsafe { dev.device.create_query_pool(&info, None) }?;
        // SAFETY: the pool is new and unused; hostQueryReset is enabled.
        unsafe { dev.device.reset_query_pool(pool, 0, POOL_SIZE) };
        Ok(Occlusion { dev: dev.clone(), pool, results, tracker: Tracker::new(POOL_SIZE) })
    }
}

impl Drop for Occlusion {
    fn drop(&mut self) {
        // SAFETY: dropped with the scheduler, after it waited for its work.
        unsafe { self.dev.device.destroy_query_pool(self.pool, None) };
    }
}

impl Scheduler {
    /// A guest occlusion query is open: draws are counted into it.
    pub fn occlusion_active(&self) -> bool {
        self.occlusion.as_ref().is_some_and(|o| o.tracker.open_block().is_some())
    }

    /// Record a guest occlusion counter dump to `addr` (see
    /// [`crate::render::RenderExecutor::occlusion_counter_dump`]). False
    /// when disabled or the device cannot count samples.
    pub fn occlusion_counter_dump(&mut self, addr: u64, write: &OcclusionWrite) -> bool {
        enabled() && self.record_occlusion_dump(addr, write)
    }

    /// [`Self::occlusion_counter_dump`] whatever `RPS5E_OCCLUSION_QUERIES` says.
    fn record_occlusion_dump(&mut self, addr: u64, write: &OcclusionWrite) -> bool {
        if !self.dev.caps.occlusion_query_precise || !self.dev.caps.host_query_reset {
            return false;
        }
        let Some(dump) = classify(addr) else { return false };
        if self.occlusion.is_none() {
            match Occlusion::new(&self.dev) {
                Ok(o) => self.occlusion = Some(Box::new(o)),
                Err(e) => {
                    crate::log_once(&format!("ps5gpu: occlusion query pool: {e:?}; occlusion queries report visible"));
                    return false;
                }
            }
        }
        match dump {
            Dump::Begin(block) => {
                // A begin without an end: close the previous query so its
                // slots still become ready.
                if let Some(previous) = self.occlusion.as_ref().and_then(|o| o.tracker.open_block()) {
                    self.end_occlusion_query(previous, write);
                }
                if let Some(o) = self.occlusion.as_mut() {
                    o.tracker.open(block, write);
                }
                if self.rendering.is_some() {
                    let cb = self.raw_cmd();
                    self.begin_occlusion_segment(cb);
                }
            }
            Dump::End(block) => self.end_occlusion_query(block, write),
        }
        true
    }

    fn end_occlusion_query(&mut self, block: u64, write: &OcclusionWrite) {
        let Some(o) = self.occlusion.as_mut() else { return };
        if let Some(index) = o.tracker.close(block, write) {
            let pool = o.pool;
            let cb = self.raw_cmd();
            // SAFETY: the query was begun in the open rendering scope.
            unsafe { self.dev.device.cmd_end_query(cb, pool, index) };
        }
        if self.rendering.is_none() {
            let cb = self.raw_cmd();
            self.publish_occlusion_results(cb);
        }
    }

    /// Called right after a rendering scope begins in `cb`.
    pub(super) fn begin_occlusion_segment(&mut self, cb: vk::CommandBuffer) {
        let Some(o) = self.occlusion.as_mut() else { return };
        if let Some(index) = o.tracker.begin_segment() {
            // SAFETY: recording buffer inside a rendering scope; the query
            // was reset after its last use and no other query is active.
            unsafe { self.dev.device.cmd_begin_query(cb, o.pool, index, vk::QueryControlFlags::PRECISE) };
        }
    }

    /// Called right before the rendering scope recorded in `cb` ends.
    pub(super) fn end_occlusion_segment(&mut self, cb: vk::CommandBuffer) {
        let Some(o) = self.occlusion.as_mut() else { return };
        if let Some(index) = o.tracker.end_segment() {
            // SAFETY: the query was begun in this rendering scope.
            unsafe { self.dev.device.cmd_end_query(cb, o.pool, index) };
        }
    }

    /// Record the results of every ended query into `cb` (outside any
    /// rendering scope); once the GPU retired them, the result blocks are
    /// written to guest memory on the priority thread.
    pub(super) fn publish_occlusion_results(&mut self, cb: vk::CommandBuffer) {
        let Some(o) = self.occlusion.as_mut() else { return };
        if o.tracker.pending.is_empty() {
            return;
        }
        let queries = std::mem::take(&mut o.tracker.pending);
        let d = &self.dev.device;
        let flags = vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT;
        for &index in queries.iter().flat_map(|q| &q.segments) {
            // SAFETY: recording buffer outside any rendering scope; the
            // query ended earlier on this queue; the slot is inside the buffer.
            unsafe { d.cmd_copy_query_pool_results(cb, o.pool, index, 1, o.results.buffer, index as u64 * 8, 8, flags) };
        }
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ);
        // SAFETY: recording buffer outside any rendering scope.
        unsafe { d.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&barrier))) };
        for &index in queries.iter().flat_map(|q| &q.segments) {
            // SAFETY: query commands execute in order: the reset follows
            // the copy above. The slot stays out of the free list until the
            // copied value was read.
            unsafe { d.cmd_reset_query_pool(cb, o.pool, index, 1) };
        }
        let results = o.results.clone();
        let free = o.tracker.free.clone();
        self.queue_priority(move || {
            results.invalidate(0, results.size);
            let bytes = results.bytes_mut();
            let mut counts = [0u64; DEPTH_BLOCKS];
            for query in queries {
                for (count, &index) in counts.iter_mut().zip(&query.segments) {
                    let at = index as usize * 8;
                    *count = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap_or_default());
                }
                free.lock().unwrap_or_else(|e| e.into_inner()).extend_from_slice(&query.segments);
                if let Some(block) = build_block(query.segments.len(), query.conservative, &counts) {
                    (query.write)(query.block, &block);
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(block: &[u8; BLOCK_BYTES], depth_block: usize, end: bool) -> u64 {
        let at = depth_block * 16 + if end { 8 } else { 0 };
        u64::from_le_bytes(block[at..at + 8].try_into().unwrap())
    }

    /// What the guest computes: the sum of end - begin over every ready
    /// depth block.
    fn guest_sum(block: &[u8; BLOCK_BYTES]) -> u64 {
        (0..DEPTH_BLOCKS)
            .map(|db| {
                let (begin, end) = (slot(block, db, false), slot(block, db, true));
                assert_ne!(begin & READY_BIT, 0);
                assert_ne!(end & READY_BIT, 0);
                (end & !READY_BIT) - (begin & !READY_BIT)
            })
            .sum()
    }

    fn nowrite() -> OcclusionWrite {
        Arc::new(|_, _| {})
    }

    #[test]
    fn env_gate_accepts_only_one() {
        assert!(parse_enabled(Some("1")));
        assert!(parse_enabled(Some(" 1\n")));
        assert!(!parse_enabled(None));
        assert!(!parse_enabled(Some("0")));
        assert!(!parse_enabled(Some("")));
        assert!(!parse_enabled(Some("yes")));
    }

    #[test]
    fn dumps_classify_by_slot() {
        assert_eq!(classify(0x1000), Some(Dump::Begin(0x1000)));
        assert_eq!(classify(0x1008), Some(Dump::End(0x1000)));
        assert_eq!(classify(0x1004), None);
        assert_eq!(classify(0), None);
    }

    #[test]
    fn segments_add_up_through_the_guest_sum() {
        let block = build_block(3, false, &[10, 0, 32]).unwrap();
        assert_eq!(guest_sum(&block), 42);
        assert_eq!(slot(&block, 0, true), READY_BIT | 10);
        assert_eq!(slot(&block, 2, true), READY_BIT | 32);
        assert_eq!(slot(&block, 5, false), READY_BIT);
    }

    #[test]
    fn a_query_without_draws_reports_zero_samples() {
        assert_eq!(guest_sum(&build_block(0, false, &[]).unwrap()), 0);
    }

    #[test]
    fn an_uncounted_query_stays_visible() {
        assert_eq!(guest_sum(&build_block(0, true, &[]).unwrap()), CONSERVATIVE_COUNT);
    }

    #[test]
    fn more_segments_than_depth_blocks_are_rejected() {
        assert!(build_block(17, false, &[0; 17]).is_none());
        assert!(build_block(2, false, &[1]).is_none());
    }

    /// Needs a Vulkan 1.3 GPU: `cargo test -p ps5gpu --release -- --ignored`.
    /// Queries spanning two rendering scopes without draws count zero
    /// samples; an end without its begin reports visible.
    #[test]
    #[ignore]
    fn host_queries_publish_counted_blocks() {
        use crate::device::{Image, ImageView};
        use crate::scheduler::RenderingKey;
        let dev = GpuDevice::create(false).expect("device");
        if !dev.caps.occlusion_query_precise || !dev.caps.host_query_reset {
            return;
        }
        let mut s = Scheduler::new(&dev).unwrap();
        let published: Arc<Mutex<Vec<(u64, [u8; BLOCK_BYTES])>>> = Arc::default();
        let p = published.clone();
        let write: OcclusionWrite = Arc::new(move |block, data| p.lock().unwrap().push((block, data.try_into().unwrap())));
        let extent = vk::Extent3D { width: 16, height: 16, depth: 1 };
        let format = vk::Format::R8G8B8A8_UNORM;
        let image = Image::new(&dev, format, extent, 1, 1, vk::SampleCountFlags::TYPE_1, vk::ImageUsageFlags::COLOR_ATTACHMENT, vk::ImageCreateFlags::empty()).unwrap();
        let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let view = ImageView::new(&dev, &vk::ImageViewCreateInfo::default().image(image.image).view_type(vk::ImageViewType::TYPE_2D).format(format).subresource_range(range)).unwrap();
        let to_general = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(image.image)
            .subresource_range(range)];
        let attachment = [vk::RenderingAttachmentInfo::default()
            .image_view(view.view)
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: 16, height: 16 } })
            .layer_count(1)
            .color_attachments(&attachment);
        let mut colors = [vk::ImageView::null(); 8];
        colors[0] = view.view;
        let key = RenderingKey { colors, color_count: 1, depth: vk::ImageView::null(), depth_attachment: false, stencil_attachment: false, width: 16, height: 16, layers: 1, general: true, depth_layout: vk::ImageLayout::UNDEFINED, provoking_last: false };

        let (a, b) = (0x1000u64, 0x2000u64);
        // Query a: begun outside a scope, counted in two scopes, ended outside.
        assert!(s.record_occlusion_dump(a, &write));
        s.begin_rendering_scope(key, &info, &to_general);
        s.end_rendering_scope();
        s.begin_rendering_scope(key, &info, &[]);
        // Query b's end without its begin, inside a scope.
        assert!(s.record_occlusion_dump(b + 8, &write));
        s.end_rendering_scope();
        assert!(s.record_occlusion_dump(a + 8, &write));
        assert_eq!(s.occlusion.as_ref().unwrap().tracker.free.lock().unwrap().len(), POOL_SIZE as usize - 2);
        s.finish();
        let blocks = published.lock().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].0, b);
        assert_eq!(guest_sum(&blocks[0].1), CONSERVATIVE_COUNT);
        assert_eq!(blocks[1].0, a);
        assert_eq!(guest_sum(&blocks[1].1), 0);
        // Both host queries returned to the pool once read.
        assert_eq!(s.occlusion.as_ref().unwrap().tracker.free.lock().unwrap().len(), POOL_SIZE as usize);
        s.retire((image, view));
        s.finish();
    }

    #[test]
    fn a_query_splits_at_scope_boundaries() {
        let w = nowrite();
        let mut t = Tracker::new(8);
        // No query open: scopes count nothing.
        assert_eq!(t.begin_segment(), None);
        t.open(0x1000, &w);
        let first = t.begin_segment().unwrap();
        // A scope is counted once.
        assert_eq!(t.begin_segment(), None);
        assert_eq!(t.end_segment(), Some(first));
        let second = t.begin_segment().unwrap();
        assert_ne!(first, second);
        // The end inside a scope ends the active segment.
        assert_eq!(t.close(0x1000, &w), Some(second));
        assert_eq!(t.open_block(), None);
        assert_eq!(t.end_segment(), None);
        assert_eq!(t.pending.len(), 1);
        assert_eq!(t.pending[0].segments, vec![first, second]);
        assert!(!t.pending[0].conservative);
    }

    #[test]
    fn an_end_without_its_begin_is_conservative() {
        let w = nowrite();
        let mut t = Tracker::new(8);
        t.open(0x1000, &w);
        assert_eq!(t.close(0x2000, &w), None);
        assert_eq!(t.open_block(), Some(0x1000), "the open query stays open");
        assert_eq!(t.pending.len(), 1);
        assert_eq!(t.pending[0].block, 0x2000);
        assert!(t.pending[0].conservative && t.pending[0].segments.is_empty());
    }

    #[test]
    fn too_many_scopes_or_an_empty_pool_are_conservative() {
        let w = nowrite();
        let mut t = Tracker::new(64);
        t.open(0x1000, &w);
        for _ in 0..DEPTH_BLOCKS {
            assert!(t.begin_segment().is_some());
            t.end_segment();
        }
        assert_eq!(t.begin_segment(), None);
        t.close(0x1000, &w);
        assert!(t.pending[0].conservative);
        assert_eq!(t.pending[0].segments.len(), DEPTH_BLOCKS);

        let mut t = Tracker::new(1);
        t.open(0x2000, &w);
        assert!(t.begin_segment().is_some());
        t.end_segment();
        assert_eq!(t.begin_segment(), None, "the pool is exhausted");
        t.close(0x2000, &w);
        assert!(t.pending[0].conservative);
    }
}
