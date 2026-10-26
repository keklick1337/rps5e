// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Redundant-state filtering for draws recorded into one rendering scope.
//!
//! Inside a dynamic-rendering scope nothing but draws is recorded (any other
//! command ends the scope first), so the pipeline, dynamic state, push
//! constants and push descriptors a draw left bound are still bound for the
//! next draw of the same scope. A draw records only what differs from the
//! previous one; a new scope starts from nothing.

use super::state::{DynamicState, StencilFace};
use ash::vk;

/// Dynamic state as recorded (`depth_bounds` and `depth_bias` as passed to
/// the commands).
#[derive(Clone, Copy, PartialEq)]
struct Dynamic {
    viewport: [u32; 6],
    scissor: [i32; 4],
    blend_constants: [u32; 4],
    cull_mode: vk::CullModeFlags,
    front_face: vk::FrontFace,
    depth_test: bool,
    depth_write: bool,
    depth_compare: vk::CompareOp,
    depth_bounds_test: bool,
    depth_bounds: [u32; 2],
    stencil_test: bool,
    stencil: [StencilFace; 2],
    depth_bias_enable: bool,
    depth_bias: [u32; 3],
}

fn bits<const N: usize>(v: [f32; N]) -> [u32; N] {
    v.map(f32::to_bits)
}

impl Dynamic {
    fn of(d: &DynamicState, bias_clamp: bool) -> Dynamic {
        let v = d.viewport;
        let s = d.scissor;
        let bias = d.depth_bias.unwrap_or([0.0; 3]);
        Dynamic {
            viewport: bits([v.x, v.y, v.width, v.height, v.min_depth, v.max_depth]),
            scissor: [s.offset.x, s.offset.y, s.extent.width as i32, s.extent.height as i32],
            blend_constants: bits(d.blend_constants),
            cull_mode: d.cull_mode,
            front_face: d.front_face,
            depth_test: d.depth_test,
            depth_write: d.depth_write,
            depth_compare: d.depth_compare,
            depth_bounds_test: d.depth_bounds_test,
            depth_bounds: bits([d.depth_bounds[0], d.depth_bounds[1].max(d.depth_bounds[0])]),
            stencil_test: d.stencil_test,
            stencil: [d.stencil_front, d.stencil_back],
            depth_bias_enable: d.depth_bias.is_some(),
            depth_bias: bits([bias[0], if bias_clamp { bias[1] } else { 0.0 }, bias[2]]),
        }
    }
}

/// What the previous draw of the open scope left bound.
#[derive(Default)]
pub struct ScopeState {
    scope: u64,
    pipeline: vk::Pipeline,
    dynamic: Option<Dynamic>,
    /// The layout the push constants and descriptors were recorded with
    /// (another layout forgets both).
    layout: vk::PipelineLayout,
    push: Option<[u32; 32]>,
    /// Push-descriptor writes: (binding, type, count, buffer?) per entry,
    /// then the infos as plain words (kept allocated across scopes).
    descriptors_valid: bool,
    entries: Vec<(u32, i32, usize, bool)>,
    words: Vec<u64>,
    index: Option<(vk::Buffer, u64, vk::IndexType)>,
    pub skipped: u64,
}

impl ScopeState {
    /// Start tracking for the draw recorded in scope `scope` (the
    /// scheduler's scope serial): a new scope forgets everything.
    pub fn enter(&mut self, scope: u64) {
        if self.scope != scope {
            self.scope = scope;
            self.pipeline = vk::Pipeline::null();
            self.dynamic = None;
            self.layout = vk::PipelineLayout::null();
            self.push = None;
            self.descriptors_valid = false;
            self.index = None;
        }
    }

    /// Bind `pipeline` and set the dynamic state that differs.
    pub fn bind(&mut self, dev: &crate::device::GpuDevice, cb: vk::CommandBuffer, pipeline: vk::Pipeline, d: &DynamicState) {
        let device = &dev.device;
        let new = Dynamic::of(d, dev.caps.depth_bias_clamp);
        let old = self.dynamic;
        let changed = |f: &dyn Fn(&Dynamic) -> bool| old.as_ref().is_none_or(|o| !f(o));
        // SAFETY: recording inside the rendering scope; the pipeline declares this state dynamic.
        unsafe {
            if self.pipeline != pipeline {
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
                self.pipeline = pipeline;
            } else {
                self.skipped += 1;
            }
            if let Some(v) = &d.indexed {
                // Viewport-selecting draws set all of them.
                device.cmd_set_viewport(cb, 0, &v.viewports);
                device.cmd_set_scissor(cb, 0, &v.scissors);
            } else {
                if changed(&|o| o.viewport == new.viewport) {
                    device.cmd_set_viewport(cb, 0, &[d.viewport]);
                }
                if changed(&|o| o.scissor == new.scissor) {
                    device.cmd_set_scissor(cb, 0, &[d.scissor]);
                }
            }
            if old.is_none() {
                device.cmd_set_line_width(cb, 1.0);
            }
            if changed(&|o| o.blend_constants == new.blend_constants) {
                device.cmd_set_blend_constants(cb, &d.blend_constants);
            }
            if changed(&|o| o.cull_mode == new.cull_mode) {
                device.cmd_set_cull_mode(cb, d.cull_mode);
            }
            if changed(&|o| o.front_face == new.front_face) {
                device.cmd_set_front_face(cb, d.front_face);
            }
            if changed(&|o| o.depth_test == new.depth_test) {
                device.cmd_set_depth_test_enable(cb, d.depth_test);
            }
            if changed(&|o| o.depth_write == new.depth_write) {
                device.cmd_set_depth_write_enable(cb, d.depth_write);
            }
            if changed(&|o| o.depth_compare == new.depth_compare) {
                device.cmd_set_depth_compare_op(cb, d.depth_compare);
            }
            if dev.caps.depth_bounds {
                if changed(&|o| o.depth_bounds_test == new.depth_bounds_test) {
                    device.cmd_set_depth_bounds_test_enable(cb, d.depth_bounds_test);
                }
                if changed(&|o| o.depth_bounds == new.depth_bounds) {
                    device.cmd_set_depth_bounds(cb, d.depth_bounds[0], d.depth_bounds[1].max(d.depth_bounds[0]));
                }
            }
            if changed(&|o| o.stencil_test == new.stencil_test) {
                device.cmd_set_stencil_test_enable(cb, d.stencil_test);
            }
            for (i, (face, f)) in [(vk::StencilFaceFlags::FRONT, d.stencil_front), (vk::StencilFaceFlags::BACK, d.stencil_back)].into_iter().enumerate() {
                let o = old.map(|o| o.stencil[i]);
                if o.is_none_or(|o| (o.fail, o.pass, o.depth_fail, o.compare) != (f.fail, f.pass, f.depth_fail, f.compare)) {
                    device.cmd_set_stencil_op(cb, face, f.fail, f.pass, f.depth_fail, f.compare);
                }
                if o.is_none_or(|o| o.compare_mask != f.compare_mask) {
                    device.cmd_set_stencil_compare_mask(cb, face, f.compare_mask);
                }
                if o.is_none_or(|o| o.write_mask != f.write_mask) {
                    device.cmd_set_stencil_write_mask(cb, face, f.write_mask);
                }
                if o.is_none_or(|o| o.reference != f.reference) {
                    device.cmd_set_stencil_reference(cb, face, f.reference);
                }
            }
            if changed(&|o| o.depth_bias_enable == new.depth_bias_enable) {
                device.cmd_set_depth_bias_enable(cb, new.depth_bias_enable);
            }
            if changed(&|o| o.depth_bias == new.depth_bias) {
                let b = new.depth_bias.map(f32::from_bits);
                device.cmd_set_depth_bias(cb, b[0], b[1], b[2]);
            }
        }
        self.dynamic = Some(new);
    }

    /// True when `push` under `layout` is already pushed (and remembers it
    /// otherwise).
    pub fn same_push(&mut self, layout: vk::PipelineLayout, push: &[u32; 32]) -> bool {
        self.use_layout(layout);
        if self.push.as_ref() == Some(push) {
            self.skipped += 1;
            return true;
        }
        self.push = Some(*push);
        false
    }

    fn use_layout(&mut self, layout: vk::PipelineLayout) {
        if self.layout != layout {
            self.layout = layout;
            self.push = None;
            self.descriptors_valid = false;
        }
    }

    /// True when these push-descriptor writes under `layout` are already
    /// pushed (and remembers them otherwise).
    pub fn same_descriptors(
        &mut self,
        layout: vk::PipelineLayout,
        entries: &[(u32, vk::DescriptorType, usize, usize, bool)],
        buffers: &[vk::DescriptorBufferInfo],
        images: &[vk::DescriptorImageInfo],
    ) -> bool {
        use ash::vk::Handle;
        self.use_layout(layout);
        let same = self.descriptors_valid
            && self.entries.len() == entries.len()
            && self.entries.iter().zip(entries).all(|(a, &(b, t, _, n, buf))| *a == (b, t.as_raw(), n, buf))
            && self.words.len() == (buffers.len() + images.len()) * 3
            && buffers
                .iter()
                .map(|b| [b.buffer.as_raw(), b.offset, b.range])
                .chain(images.iter().map(|i| [i.sampler.as_raw(), i.image_view.as_raw(), i.image_layout.as_raw() as u64]))
                .zip(self.words.chunks_exact(3))
                .all(|(a, b)| a[..] == *b);
        if same {
            self.skipped += 1;
            return true;
        }
        self.entries.clear();
        self.entries.extend(entries.iter().map(|&(b, t, _, n, buf)| (b, t.as_raw(), n, buf)));
        self.words.clear();
        for b in buffers {
            self.words.extend_from_slice(&[b.buffer.as_raw(), b.offset, b.range]);
        }
        for i in images {
            self.words.extend_from_slice(&[i.sampler.as_raw(), i.image_view.as_raw(), i.image_layout.as_raw() as u64]);
        }
        self.descriptors_valid = true;
        false
    }

    /// A pool set was bound under `layout`.
    pub fn bound_set(&mut self, layout: vk::PipelineLayout) {
        self.use_layout(layout);
        self.descriptors_valid = false;
    }

    /// True when this index buffer is already bound.
    pub fn same_index(&mut self, binding: (vk::Buffer, u64, vk::IndexType)) -> bool {
        if self.index == Some(binding) {
            self.skipped += 1;
            return true;
        }
        self.index = Some(binding);
        false
    }
}
