// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! DCC fast clears decided on the GPU.
//!
//! A compute program (the SDK fill kernel) that overwrites a colour
//! target's DCC metadata may have written a clear code: when every key
//! byte of the surface holds the same code (0x00, 0x20, 0x40, 0x80, 0xC0)
//! the surface reads as the matching clear colour. Reading the keys back
//! on the CPU waits for the GPU; instead three passes of one small compute
//! shader decide it in order with the rest of the work:
//!
//! 1. *scan*: every key word is compared with the first one; a mismatch
//!    sets the "mixed" flag of a result slot, word 0 records the first
//!    word;
//! 2. *decide* (one invocation): for each candidate code `k` the slot's
//!    predicate word `4 + k` becomes 1 when the keys are uniformly `k`,
//!    and word 2 when any candidate matched;
//! 3. *consume* (after the conditional clears): when word 2 is set, the
//!    keys become 0xFF (uncompressed) — the surface now holds the colour.
//!
//! Between 2 and 3 the backend records one clear per decodable candidate
//! under conditional rendering on its predicate word
//! (`vkCmdClearAttachments` is a conditional command).
//!
//! Push constants: mode (0 scan, 1 decide, 2 consume), first key word in
//! the key binding, key word count, the slot's first word in the result
//! binding, then the five candidate codes (0xFFFF_FFFF: never matches).
//! Binding 0 is read-only: the keys (scan) or the result slot (consume);
//! binding 1 is written: the result slot (scan, decide) or the keys
//! (consume).

use crate::device::{Buffer, GpuDevice, MemoryUse};
use crate::tiler::{Spv, ARGUMENT_COUNT, R};
use ash::vk;
use rspirv::spirv;
use std::sync::Arc;

/// Clear codes, in predicate-word order.
pub const CODES: [u8; 5] = [0x00, 0x20, 0x40, 0x80, 0xC0];
const SLOT_BYTES: u64 = 256;
const SLOTS: u64 = 1024;
const GROUP: u32 = 64;

/// Byte offset of candidate `k`'s predicate word in a slot.
pub fn predicate_offset(k: usize) -> u64 {
    (4 + k as u64) * 4
}

fn kernel() -> R<Vec<u32>> {
    let mut s = Spv::new();
    let (main, x, _, _) = s.begin_main()?;
    let mode = s.arg(0)?;
    let base = s.arg(1)?;
    let count = s.arg(2)?;
    let slot = s.arg(3)?;
    let (uint, boolean) = (s.uint, s.boolean);
    let scope = s.c(spirv::Scope::Device as u32);
    let sem = s.c(0);
    let zero = s.c(0);
    let one = s.c(1);
    let two = s.c(2);

    // Mode 0: scan.
    let is_scan = s.b.i_equal(boolean, None, mode, zero)?;
    let scan_block = s.b.id();
    let after_scan = s.b.id();
    s.b.selection_merge(after_scan, spirv::SelectionControl::NONE)?;
    s.b.branch_conditional(is_scan, scan_block, after_scan, [])?;
    s.b.begin_block(Some(scan_block))?;
    {
        let inside = s.b.u_less_than(boolean, None, x, count)?;
        let body = s.b.id();
        let end = s.b.id();
        s.b.selection_merge(end, spirv::SelectionControl::NONE)?;
        s.b.branch_conditional(inside, body, end, [])?;
        s.b.begin_block(Some(body))?;
        let idx = s.add(base, x)?;
        let w = s.load_word(s.src, idx)?;
        let f = s.load_word(s.src, base)?;
        let differs = s.b.i_not_equal(boolean, None, w, f)?;
        let set = s.b.id();
        let next = s.b.id();
        s.b.selection_merge(next, spirv::SelectionControl::NONE)?;
        s.b.branch_conditional(differs, set, next, [])?;
        s.b.begin_block(Some(set))?;
        let flag = s.word_ptr(s.dst, slot)?;
        s.b.atomic_or(uint, None, flag, scope, sem, one)?;
        s.b.branch(next)?;
        s.b.begin_block(Some(next))?;
        let first = s.b.i_equal(boolean, None, x, zero)?;
        let store = s.b.id();
        let done = s.b.id();
        s.b.selection_merge(done, spirv::SelectionControl::NONE)?;
        s.b.branch_conditional(first, store, done, [])?;
        s.b.begin_block(Some(store))?;
        let i1 = s.add(slot, one)?;
        let p = s.word_ptr(s.dst, i1)?;
        s.b.store(p, f, None, [])?;
        s.b.branch(done)?;
        s.b.begin_block(Some(done))?;
        s.b.branch(end)?;
        s.b.begin_block(Some(end))?;
    }
    s.b.branch(after_scan)?;
    s.b.begin_block(Some(after_scan))?;

    // Mode 1: decide.
    let is_decide = s.b.i_equal(boolean, None, mode, one)?;
    let first_lane = s.b.i_equal(boolean, None, x, zero)?;
    let decide = s.b.logical_and(boolean, None, is_decide, first_lane)?;
    let decide_block = s.b.id();
    let after_decide = s.b.id();
    s.b.selection_merge(after_decide, spirv::SelectionControl::NONE)?;
    s.b.branch_conditional(decide, decide_block, after_decide, [])?;
    s.b.begin_block(Some(decide_block))?;
    {
        let mixed = s.load_word(s.dst, slot)?;
        let i1 = s.add(slot, one)?;
        let f = s.load_word(s.dst, i1)?;
        let byte = s.and_c(f, 0xFF)?;
        let splat = s.mul_c(byte, 0x0101_0101)?;
        let same_bytes = s.b.i_equal(boolean, None, f, splat)?;
        let not_mixed = s.b.i_equal(boolean, None, mixed, zero)?;
        let uniform = s.b.logical_and(boolean, None, same_bytes, not_mixed)?;
        let mut any = s.b.constant_false(boolean);
        for k in 0..CODES.len() as u32 {
            let code = s.arg(4 + k)?;
            let eq = s.b.i_equal(boolean, None, byte, code)?;
            let hit = s.b.logical_and(boolean, None, uniform, eq)?;
            any = s.b.logical_or(boolean, None, any, hit)?;
            let v = s.b.select(uint, None, hit, one, zero)?;
            let off = s.c(4 + k);
            let i = s.add(slot, off)?;
            let p = s.word_ptr(s.dst, i)?;
            s.b.store(p, v, None, [])?;
        }
        let v = s.b.select(uint, None, any, one, zero)?;
        let i2 = s.add(slot, two)?;
        let p = s.word_ptr(s.dst, i2)?;
        s.b.store(p, v, None, [])?;
    }
    s.b.branch(after_decide)?;
    s.b.begin_block(Some(after_decide))?;

    // Mode 2: consume.
    let is_consume = s.b.i_equal(boolean, None, mode, two)?;
    let inside = s.b.u_less_than(boolean, None, x, count)?;
    let consume = s.b.logical_and(boolean, None, is_consume, inside)?;
    let consume_block = s.b.id();
    let after_consume = s.b.id();
    s.b.selection_merge(after_consume, spirv::SelectionControl::NONE)?;
    s.b.branch_conditional(consume, consume_block, after_consume, [])?;
    s.b.begin_block(Some(consume_block))?;
    {
        let i2 = s.add(slot, two)?;
        let cleared = s.load_word(s.src, i2)?;
        let yes = s.b.i_not_equal(boolean, None, cleared, zero)?;
        let write = s.b.id();
        let end = s.b.id();
        s.b.selection_merge(end, spirv::SelectionControl::NONE)?;
        s.b.branch_conditional(yes, write, end, [])?;
        s.b.begin_block(Some(write))?;
        let idx = s.add(base, x)?;
        let p = s.word_ptr(s.dst, idx)?;
        let ones = s.c(0xFFFF_FFFF);
        s.b.store(p, ones, None, [])?;
        s.b.branch(end)?;
        s.b.begin_block(Some(end))?;
    }
    s.b.branch(after_consume)?;
    s.b.begin_block(Some(after_consume))?;
    s.finish(main, GROUP, 1)
}

/// A buffer range bound to one kernel binding.
#[derive(Clone, Copy, Debug)]
pub struct Range {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub size: u64,
}

/// The kernel, its layout and the ring of result slots.
pub struct DccResolver {
    dev: Arc<GpuDevice>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pub results: Buffer,
    next: u64,
}

impl DccResolver {
    /// None when the device lacks conditional rendering or the kernel
    /// cannot be built.
    pub fn new(dev: &Arc<GpuDevice>) -> Option<DccResolver> {
        dev.conditional.as_ref()?;
        let code = kernel().ok()?;
        let bindings = [0u32, 1].map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::CONDITIONAL_RENDERING_EXT;
        let results = Buffer::new(dev, SLOT_BYTES * SLOTS, usage, MemoryUse::Device).ok()?;
        // SAFETY: valid create infos; destroyed in Drop.
        unsafe {
            let set_layout = dev
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR).bindings(&bindings),
                    None,
                )
                .ok()?;
            let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: (ARGUMENT_COUNT * 4) as u32 }];
            let layout = dev
                .device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout)).push_constant_ranges(&ranges), None)
                .ok()?;
            let module = dev.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).ok()?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            let r = dev.device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None);
            dev.device.destroy_shader_module(module, None);
            let pipeline = match r {
                Ok(p) => p[0],
                Err((_, e)) => {
                    log::warn!("ps5gpu: DCC clear kernel: {e:?}");
                    return None;
                }
            };
            Some(DccResolver { dev: dev.clone(), set_layout, layout, pipeline, results, next: 0 })
        }
    }

    /// A fresh result slot (byte offset in `results`). Slots are reused
    /// round-robin; work on the queue is ordered by full barriers.
    pub fn slot(&mut self) -> u64 {
        let s = self.next % SLOTS * SLOT_BYTES;
        self.next += 1;
        s
    }

    fn slot_range(&self, slot: u64) -> Range {
        Range { buffer: self.results.buffer, offset: slot, size: SLOT_BYTES }
    }

    /// Record one pass. `keys` is the key range (whole dwords), `slot` the
    /// result slot, `codes` the candidate codes (0xFFFF_FFFF: skip).
    pub fn record(&self, cb: vk::CommandBuffer, mode: u32, keys: Range, slot: u64, codes: &[u32; 5]) {
        let align = self.dev.caps.min_storage_buffer_offset_alignment.max(4);
        let bind = |r: Range| {
            let base = r.offset / align * align;
            (vk::DescriptorBufferInfo { buffer: r.buffer, offset: base, range: (r.offset - base + r.size).div_ceil(4) * 4 }, ((r.offset - base) / 4) as u32)
        };
        let (key_info, key_word) = bind(keys);
        let (slot_info, slot_word) = bind(self.slot_range(slot));
        let count = (keys.size / 4) as u32;
        // Binding 0 is read, binding 1 written.
        let (read, write) = if mode == 2 { (slot_info, key_info) } else { (key_info, slot_info) };
        let mut words = [0u32; ARGUMENT_COUNT];
        words[0] = mode;
        words[1] = key_word;
        words[2] = count;
        words[3] = slot_word;
        words[4..9].copy_from_slice(codes);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let writes = [
            vk::WriteDescriptorSet::default().dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&read)),
            vk::WriteDescriptorSet::default().dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&write)),
        ];
        let groups = if mode == 1 { 1 } else { count.div_ceil(GROUP).max(1) };
        crate::scheduler::disturb_state();
        // SAFETY: recording command buffer; all handles valid.
        unsafe {
            let d = &self.dev.device;
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.dev.push_descriptor.cmd_push_descriptor_set(cb, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &writes);
            d.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
            d.cmd_dispatch(cb, groups, 1, 1);
        }
    }
}

impl Drop for DccResolver {
    fn drop(&mut self) {
        // SAFETY: the device is idle when the back-end is dropped.
        unsafe {
            self.dev.device.destroy_pipeline(self.pipeline, None);
            self.dev.device.destroy_pipeline_layout(self.layout, None);
            self.dev.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn kernel_assembles() {
        let words = super::kernel().expect("kernel");
        assert_eq!(words[0], 0x0723_0203);
    }
}
