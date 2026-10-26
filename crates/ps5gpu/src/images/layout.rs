// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Layout and access tracking of cached images.
//!
//! Every image remembers the layout, access and pipeline stages of its last
//! use, for the whole image or, once a use covered only some levels or
//! layers, for each subresource (level × array layer; a volume has one
//! layer). A new use asks for the barriers that bring the image (or the
//! used subresources) from that state into the one it needs: a barrier is
//! recorded only when the layout or the access changes, or when the last
//! use wrote through transfers, shaders or generic memory writes (a write
//! must be made visible even when the next use is the same).
//!
//! Callers record the barriers outside any rendering scope.

use ash::vk;

/// The state of an image (or one subresource) after its last use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccessState {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
    pub layout: vk::ImageLayout,
}

impl Default for AccessState {
    fn default() -> Self {
        AccessState { stage: vk::PipelineStageFlags2::ALL_COMMANDS, access: vk::AccessFlags2::NONE, layout: vk::ImageLayout::UNDEFINED }
    }
}

/// The layout of a subresource a slice segment's overlay does not know yet
/// (never given to Vulkan: see [`LayoutTracker::unknown_like`]).
pub const UNKNOWN: vk::ImageLayout = vk::ImageLayout::from_raw(0x7FFF_0001);

/// Levels and layers a use covers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SubRange {
    pub base_level: u32,
    pub levels: u32,
    pub base_layer: u32,
    pub layers: u32,
}

/// Accesses after which the next use needs a barrier even in the same
/// layout and access.
const WRITE_ACCESS: vk::AccessFlags2 =
    vk::AccessFlags2::from_raw(vk::AccessFlags2::TRANSFER_WRITE.as_raw() | vk::AccessFlags2::SHADER_WRITE.as_raw() | vk::AccessFlags2::MEMORY_WRITE.as_raw());

/// Every write access (attachment writes included).
const ANY_WRITE: vk::AccessFlags2 = vk::AccessFlags2::from_raw(
    WRITE_ACCESS.as_raw() | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw() | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw(),
);

/// The pipeline stages a use with `access` waits in: transfers, and every
/// graphics and compute stage for other accesses (or none given).
pub fn stages_for(access: vk::AccessFlags2) -> vk::PipelineStageFlags2 {
    let transfer = vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE;
    let mut stage = vk::PipelineStageFlags2::NONE;
    if access.intersects(transfer) {
        stage |= vk::PipelineStageFlags2::TRANSFER;
    }
    if access.is_empty() || !(access & !transfer).is_empty() {
        stage |= vk::PipelineStageFlags2::ALL_GRAPHICS | vk::PipelineStageFlags2::COMPUTE_SHADER;
    }
    stage
}

/// Every aspect of an image of `format`.
pub fn full_aspect(format: vk::Format) -> vk::ImageAspectFlags {
    crate::device::aspect_of(format)
}

/// The tracked state of one image.
#[derive(Clone, Debug)]
pub struct LayoutTracker {
    state: AccessState,
    /// Per level × layer once a use covered part of the image (empty
    /// while the whole image shares `state`).
    subresources: Vec<AccessState>,
    levels: u32,
    layers: u32,
    volume: bool,
}

impl LayoutTracker {
    /// A new image (contents undefined): `levels` levels of `layers`
    /// array layers (a volume has one).
    pub fn new(levels: u32, layers: u32, volume: bool) -> LayoutTracker {
        LayoutTracker { state: AccessState::default(), subresources: Vec::new(), levels: levels.max(1), layers: if volume { 1 } else { layers.max(1) }, volume }
    }

    /// A tracker of the same shape whose state is not known yet (a slice
    /// segment's overlay, R7b): a use of a subresource in [`UNKNOWN`]
    /// layout gives a barrier from [`UNKNOWN`], which the overlay turns into
    /// a requirement on the committed state instead.
    pub fn unknown_like(&self) -> LayoutTracker {
        LayoutTracker {
            state: AccessState { layout: UNKNOWN, ..AccessState::default() },
            subresources: Vec::new(),
            levels: self.levels,
            layers: self.layers,
            volume: self.volume,
        }
    }

    /// Take the state `last` (of the same shape) left for every
    /// subresource it knows (not [`UNKNOWN`]).
    pub fn adopt(&mut self, last: &LayoutTracker) {
        if last.levels != self.levels || last.layers != self.layers {
            return;
        }
        if last.subresources.is_empty() {
            if last.state.layout != UNKNOWN {
                self.state = last.state;
                self.subresources.clear();
            }
            return;
        }
        if self.subresources.is_empty() {
            self.subresources = vec![self.state; (self.levels * self.layers) as usize];
        }
        for (mine, theirs) in self.subresources.iter_mut().zip(&last.subresources) {
            if theirs.layout != UNKNOWN {
                *mine = *theirs;
            }
        }
        if self.subresources.windows(2).all(|w| w[0] == w[1]) {
            self.state = self.subresources[0];
            self.subresources.clear();
        }
    }

    /// The state of the whole image when it shares one (else that of the
    /// first subresource).
    pub fn state(&self) -> AccessState {
        self.subresources.first().copied().unwrap_or(self.state)
    }

    /// The state of one subresource.
    pub fn subresource_state(&self, level: u32, layer: u32) -> AccessState {
        if self.subresources.is_empty() {
            return self.state;
        }
        let layer = if self.volume { 0 } else { layer };
        self.subresources.get((level * self.layers + layer) as usize).copied().unwrap_or(self.state)
    }

    /// True while every subresource shares one state.
    pub fn is_uniform(&self) -> bool {
        self.subresources.is_empty()
    }

    /// Barriers taking `range` (None: the whole image) of `image` into
    /// `layout` with `access` in `stage`, appended to `out`; the tracked
    /// state becomes the new one. A volume's layer ranges are its depth
    /// slices and cover its one layer.
    #[allow(clippy::too_many_arguments)]
    pub fn barriers(
        &mut self,
        image: vk::Image,
        aspect: vk::ImageAspectFlags,
        layout: vk::ImageLayout,
        access: vk::AccessFlags2,
        stage: vk::PipelineStageFlags2,
        range: Option<SubRange>,
        out: &mut Vec<vk::ImageMemoryBarrier2<'static>>,
    ) {
        let range = range.map(|mut r| {
            if self.volume {
                r.base_layer = 0;
                r.layers = 1;
            }
            r.base_level = r.base_level.min(self.levels - 1);
            r.levels = r.levels.clamp(1, self.levels - r.base_level);
            r.base_layer = r.base_layer.min(self.layers - 1);
            r.layers = r.layers.clamp(1, self.layers - r.base_layer);
            r
        });
        let partial = range.is_some_and(|r| r.base_level != 0 || r.levels != self.levels || r.base_layer != 0 || r.layers != self.layers);
        let next = AccessState { stage, access, layout };
        let barrier = |from: &AccessState, base_level: u32, levels: u32, base_layer: u32, layers: u32| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(from.stage)
                .src_access_mask(from.access)
                .dst_stage_mask(stage)
                .dst_access_mask(access)
                .old_layout(from.layout)
                .new_layout(layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: base_level, level_count: levels, base_array_layer: base_layer, layer_count: layers })
        };
        // Reads after reads in one layout need no barrier: the state keeps
        // every reader (stages and accesses accumulate), so the next writer
        // still waits for all of them.
        let read_after_read = |from: &AccessState| {
            from.layout == layout && from.layout != vk::ImageLayout::UNDEFINED && !from.access.is_empty() && !access.is_empty() && !from.access.intersects(ANY_WRITE) && !access.intersects(ANY_WRITE)
        };
        let needs = |from: &AccessState| from.layout != layout || from.access != access || from.access.intersects(WRITE_ACCESS);
        if partial || !self.subresources.is_empty() {
            if self.subresources.is_empty() {
                self.subresources = vec![self.state; (self.levels * self.layers) as usize];
            }
            let r = if partial { range.unwrap() } else { SubRange { base_level: 0, levels: self.levels, base_layer: 0, layers: self.layers } };
            for level in r.base_level..r.base_level + r.levels {
                for layer in r.base_layer..r.base_layer + r.layers {
                    let s = &mut self.subresources[(level * self.layers + layer) as usize];
                    if read_after_read(s) {
                        *s = AccessState { stage: s.stage | stage, access: s.access | access, layout };
                    } else if needs(s) {
                        out.push(barrier(s, level, 1, layer, 1));
                        *s = next;
                    }
                }
            }
            if !partial {
                self.subresources.clear();
            }
        } else {
            if read_after_read(&self.state) {
                self.state = AccessState { stage: self.state.stage | stage, access: self.state.access | access, layout };
                return;
            }
            if !needs(&self.state) {
                return;
            }
            out.push(barrier(&self.state, 0, vk::REMAINING_MIP_LEVELS, 0, vk::REMAINING_ARRAY_LAYERS));
        }
        self.state = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMG: vk::Image = vk::Image::null();
    const COLOR: vk::ImageAspectFlags = vk::ImageAspectFlags::COLOR;

    fn run(t: &mut LayoutTracker, layout: vk::ImageLayout, access: vk::AccessFlags2, range: Option<SubRange>) -> Vec<vk::ImageMemoryBarrier2<'static>> {
        let mut out = Vec::new();
        t.barriers(IMG, COLOR, layout, access, stages_for(access), range, &mut out);
        out
    }

    #[test]
    fn same_state_needs_no_barrier_unless_written() {
        let mut t = LayoutTracker::new(1, 1, false);
        let read = vk::AccessFlags2::SHADER_READ;
        let b = run(&mut t, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, read, None);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(b[0].src_stage_mask, vk::PipelineStageFlags2::ALL_COMMANDS);
        assert!(run(&mut t, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, read, None).is_empty());
        // Attachment accesses repeat without a barrier.
        let att = vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE;
        assert_eq!(run(&mut t, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, att, None).len(), 1);
        assert!(run(&mut t, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, att, None).is_empty());
        // Transfer and shader writes are made visible to the next use.
        let w = vk::AccessFlags2::TRANSFER_WRITE;
        let b = run(&mut t, vk::ImageLayout::TRANSFER_DST_OPTIMAL, w, None);
        assert_eq!(b[0].old_layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert_eq!(b[0].src_access_mask, att);
        assert_eq!(b[0].dst_stage_mask, vk::PipelineStageFlags2::TRANSFER);
        assert_eq!(run(&mut t, vk::ImageLayout::TRANSFER_DST_OPTIMAL, w, None).len(), 1);
        let sw = vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE;
        assert_eq!(run(&mut t, vk::ImageLayout::GENERAL, sw, None).len(), 1);
        assert_eq!(run(&mut t, vk::ImageLayout::GENERAL, sw, None).len(), 1);
    }

    #[test]
    fn partial_uses_track_subresources_and_merge_back() {
        let mut t = LayoutTracker::new(3, 2, false);
        let read = vk::AccessFlags2::SHADER_READ;
        let att = vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE;
        // Whole image first: one barrier for every subresource.
        let b = run(&mut t, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, read, None);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].subresource_range.level_count, vk::REMAINING_MIP_LEVELS);
        // Level 1 of both layers as an attachment.
        let r = SubRange { base_level: 1, levels: 1, base_layer: 0, layers: 2 };
        let b = run(&mut t, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, att, Some(r));
        assert_eq!(b.len(), 2);
        assert!(b.iter().all(|b| b.subresource_range.base_mip_level == 1 && b.subresource_range.level_count == 1 && b.subresource_range.layer_count == 1));
        assert!(!t.is_uniform());
        assert_eq!(t.subresource_state(1, 1).layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert_eq!(t.subresource_state(0, 1).layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        // Level 0 sampled again: already there.
        let r0 = SubRange { base_level: 0, levels: 1, base_layer: 0, layers: 2 };
        assert!(run(&mut t, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, read, Some(r0)).is_empty());
        // The whole image as a transfer source: every subresource changes
        // layout (one barrier each), then the image shares one state again.
        let b = run(&mut t, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags2::TRANSFER_READ, None);
        assert_eq!(b.len(), 6);
        assert!(t.is_uniform());
        assert_eq!(t.state().layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        assert!(run(&mut t, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags2::TRANSFER_READ, None).is_empty());
    }

    #[test]
    fn full_ranges_and_volumes_are_whole_image_uses() {
        let mut t = LayoutTracker::new(2, 1, true);
        let att = vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE;
        // Slices 3..5 of level 0 of a volume: its layer 0 at level 0 only.
        let b = run(&mut t, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, att, Some(SubRange { base_level: 0, levels: 1, base_layer: 3, layers: 2 }));
        assert_eq!(b.len(), 1);
        assert_eq!((b[0].subresource_range.base_array_layer, b[0].subresource_range.layer_count), (0, 1));
        assert_eq!(t.subresource_state(1, 0).layout, vk::ImageLayout::UNDEFINED);
        // A range that is the whole image behaves like None.
        let mut u = LayoutTracker::new(1, 4, false);
        let b = run(&mut u, vk::ImageLayout::GENERAL, vk::AccessFlags2::SHADER_READ, Some(SubRange { base_level: 0, levels: 1, base_layer: 0, layers: 4 }));
        assert_eq!(b.len(), 1);
        assert!(u.is_uniform());
        // Ranges past the image are clamped.
        let b = run(&mut u, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::AccessFlags2::SHADER_READ, Some(SubRange { base_level: 0, levels: 9, base_layer: 3, layers: 9 }));
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].subresource_range.base_array_layer, 3);
    }

    #[test]
    fn stages_follow_the_access() {
        assert_eq!(stages_for(vk::AccessFlags2::TRANSFER_WRITE), vk::PipelineStageFlags2::TRANSFER);
        let all = vk::PipelineStageFlags2::ALL_GRAPHICS | vk::PipelineStageFlags2::COMPUTE_SHADER;
        assert_eq!(stages_for(vk::AccessFlags2::SHADER_READ), all);
        assert_eq!(stages_for(vk::AccessFlags2::NONE), all);
        assert_eq!(stages_for(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::TRANSFER_READ), all | vk::PipelineStageFlags2::TRANSFER);
    }
}
