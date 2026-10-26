// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Indirect dispatches whose argument record counts threads
//! (`DISPATCH_INITIATOR.USE_THREAD_DIMENSIONS`) on the GPU timeline.
//!
//! The command processor reads the record when it executes the dispatch:
//! each axis launches ceil(threads / group size) groups, and the lanes of
//! the partial last groups past the thread count do not run. The dispatch
//! program masks those lanes by comparing its global invocation id with the
//! three thread counts in its shader data (`uses_dispatch_thread_limits`).
//!
//! A one-invocation compute kernel, recorded after the record's writers and
//! before the dispatch, does the command processor's part:
//!
//! ```text
//!   (tx, ty, tz) = record
//!   g[i]    = tx / size[i] + (tx % size[i] != 0)     (0 threads: 0 groups)
//!   command = any g[i] > maxComputeWorkGroupCount[i] ? (0, 0, 0) : g
//!   data[limits + i] = t[i]
//! ```
//!
//! `command` is a `VkDispatchIndirectCommand` the dispatch reads with
//! `vkCmdDispatchIndirect`; `data` is the program's shader data (a fresh
//! staging copy the dispatch binds, so its thread limits live in a buffer
//! the GPU can write, never in push constants). A launch over the device's
//! group limit runs nothing, as the direct path skips such a dispatch.
//! [`launch`] is the same rule on the CPU.

use crate::device::GpuDevice;
use crate::tiler::{Spv, TilerBuffer, R};
use ash::vk;
use std::sync::Arc;

/// Push-constant dwords of the kernel.
pub const ARGUMENT_COUNT: usize = 10;
/// Bytes of the argument record (three thread counts).
pub const RECORD_BYTES: u64 = 12;
/// Bytes of a `VkDispatchIndirectCommand`.
pub const COMMAND_BYTES: u64 = 12;

/// How a record's thread counts become thread groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThreadRules {
    /// COMPUTE_NUM_THREAD_X/Y/Z (threads per group, at least 1).
    pub local_size: [u32; 3],
    /// maxComputeWorkGroupCount.
    pub max_groups: [u32; 3],
}

impl ThreadRules {
    pub fn new(threads: [u32; 3], max_groups: [u32; 3]) -> ThreadRules {
        ThreadRules { local_size: threads.map(|t| t.max(1)), max_groups }
    }
}

/// Thread groups of a dispatch of `threads` threads per axis: the groups
/// covering them (the last ones partial), all zero when an axis exceeds the
/// device's group limit. An axis without threads launches no group.
pub fn launch(threads: [u32; 3], r: &ThreadRules) -> [u32; 3] {
    let g: [u32; 3] = std::array::from_fn(|i| threads[i].div_ceil(r.local_size[i].max(1)));
    if (0..3).any(|i| g[i] > r.max_groups[i]) {
        return [0; 3];
    }
    g
}

/// Push words: byte bases of the bindings (record, command, shader data)
/// within their aligned descriptors, the shader-data dword of the limits,
/// the group size and the device limits.
fn words(bases: [u32; 3], limits_dword: u32, r: &ThreadRules) -> [u32; ARGUMENT_COUNT] {
    let mut w = [0u32; ARGUMENT_COUNT];
    w[..3].copy_from_slice(&bases.map(|b| b / 4));
    w[3] = limits_dword;
    w[4..7].copy_from_slice(&r.local_size.map(|n| n.max(1)));
    w[7..10].copy_from_slice(&r.max_groups);
    w
}

/// Bindings: 0 the record (read), 1 the command (written), 2 the shader
/// data (written).
fn kernel() -> R<Vec<u32>> {
    let mut s = Spv::with_arguments(ARGUMENT_COUNT);
    let (record, command) = (s.src, s.dst);
    let data = s.storage_buffer(2, "shader_data", false);
    let (main, _, _, _) = s.begin_main()?;
    let [rb, kb, db, limits] = [s.arg(0)?, s.arg(1)?, s.arg(2)?, s.arg(3)?];
    let zero = s.c(0);
    let one = s.c(1);
    let mut threads = [0; 3];
    let mut groups = [0; 3];
    let mut over = s.b.constant_false(s.boolean);
    for i in 0..3u32 {
        let at = s.add_c(rb, i)?;
        let t = s.load_word(record, at)?;
        let size = s.arg(4 + i)?;
        let whole = s.b.u_div(s.uint, None, t, size)?;
        let rest = s.b.u_mod(s.uint, None, t, size)?;
        let partial = s.b.i_not_equal(s.boolean, None, rest, zero)?;
        let extra = s.select(partial, one, zero)?;
        let g = s.add(whole, extra)?;
        let max = s.arg(7 + i)?;
        let above = s.b.u_greater_than(s.boolean, None, g, max)?;
        over = s.b.logical_or(s.boolean, None, over, above)?;
        threads[i as usize] = t;
        groups[i as usize] = g;
    }
    for i in 0..3u32 {
        let g = s.select(over, zero, groups[i as usize])?;
        let at = s.add_c(kb, i)?;
        s.store_word(command, at, g)?;
    }
    let first = s.add(db, limits)?;
    for i in 0..3u32 {
        let at = s.add_c(first, i)?;
        s.store_word(data, at, threads[i as usize])?;
    }
    s.finish(main, 1, 1)
}

/// The kernel and its layout (push descriptors and push constants).
pub struct IndirectThreadsKernel {
    dev: Arc<GpuDevice>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl IndirectThreadsKernel {
    /// None when the kernel cannot be built (such dispatches then read
    /// their record on the CPU).
    pub fn new(dev: &Arc<GpuDevice>) -> Option<IndirectThreadsKernel> {
        let code = match kernel() {
            Ok(c) => c,
            Err(e) => {
                log::warn!("ps5gpu: thread-dimension kernel assembly failed: {e:?}");
                return None;
            }
        };
        let bindings = [0u32, 1, 2].map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        // SAFETY: valid create infos; destroyed in Drop (or here on failure).
        unsafe {
            let d = &dev.device;
            let set_layout = d
                .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR).bindings(&bindings), None)
                .ok()?;
            let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: (ARGUMENT_COUNT * 4) as u32 }];
            let layout = match d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout)).push_constant_ranges(&ranges), None) {
                Ok(l) => l,
                Err(_) => {
                    d.destroy_descriptor_set_layout(set_layout, None);
                    return None;
                }
            };
            let pipeline = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).ok().and_then(|module| {
                let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
                let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
                let r = d.create_compute_pipelines(vk::PipelineCache::null(), &[info], None);
                d.destroy_shader_module(module, None);
                r.map_err(|(_, e)| log::warn!("ps5gpu: thread-dimension kernel: {e:?}")).ok().map(|p| p[0])
            });
            let Some(pipeline) = pipeline else {
                d.destroy_pipeline_layout(layout, None);
                d.destroy_descriptor_set_layout(set_layout, None);
                return None;
            };
            Some(IndirectThreadsKernel { dev: dev.clone(), set_layout, layout, pipeline })
        }
    }

    /// Record one run (outside any rendering scope): the thread counts at
    /// `record` (4-byte aligned), the groups written to `command` (at least
    /// [`COMMAND_BYTES`]) and the counts to dwords `limits_dword..+3` of the
    /// shader data `data`.
    pub fn record(&self, cb: vk::CommandBuffer, record: TilerBuffer, command: TilerBuffer, data: TilerBuffer, limits_dword: u32, rules: &ThreadRules) {
        let align = self.dev.caps.min_storage_buffer_offset_alignment.max(4);
        let (ri, rb) = record.binding(align);
        let (ki, kb) = command.binding(align);
        let (di, db) = data.binding(align);
        let words = words([rb, kb, db], limits_dword, rules);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let infos = [ri, ki, di];
        let writes: [vk::WriteDescriptorSet; 3] = std::array::from_fn(|i| {
            vk::WriteDescriptorSet::default().dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&infos[i]))
        });
        crate::scheduler::disturb_state();
        // SAFETY: recording command buffer outside a rendering scope; all handles valid.
        unsafe {
            let d = &self.dev.device;
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.dev.push_descriptor.cmd_push_descriptor_set(cb, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &writes);
            d.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
            d.cmd_dispatch(cb, 1, 1, 1);
        }
    }
}

impl Drop for IndirectThreadsKernel {
    fn drop(&mut self) {
        // SAFETY: the provider waits for the device before dropping.
        unsafe {
            self.dev.device.destroy_pipeline(self.pipeline, None);
            self.dev.device.destroy_pipeline_layout(self.layout, None);
            self.dev.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{DispatchPacket, ResolvedDispatch, ShaderStage as Stage, DISPATCH_USE_THREAD_DIMENSIONS};

    const MAX: [u32; 3] = [65535, 65535, 65535];

    fn direct(threads: [u32; 3], local: [u32; 3]) -> ResolvedDispatch {
        ResolvedDispatch {
            packet: DispatchPacket { groups: threads, dispatch_initiator: 1 | DISPATCH_USE_THREAD_DIMENSIONS, ..DispatchPacket::default() },
            shader: Stage { address: 0, rsrc1: 0, rsrc2: 0, user_data: [0; 32] },
            rsrc3: 0,
            threads: local,
            start: [0; 3],
        }
    }

    #[test]
    fn kernel_assembles() {
        let words = kernel().expect("assembles");
        assert_eq!(words[0], rspirv::spirv::MAGIC_NUMBER);
        let mut loader = rspirv::dr::Loader::new();
        rspirv::binary::parse_words(&words, &mut loader).unwrap();
    }

    #[test]
    fn launch_rounds_up_per_axis() {
        let r = ThreadRules::new([8, 4, 2], MAX);
        assert_eq!(launch([13, 5, 3], &r), [2, 2, 2], "partial groups on every axis");
        assert_eq!(launch([16, 8, 4], &r), [2, 2, 2], "exact multiples");
        assert_eq!(launch([1, 1, 1], &r), [1, 1, 1], "one thread is one group");
        assert_eq!(launch([0, 8, 4], &r), [0, 2, 2], "an axis without threads launches nothing");
        assert_eq!(launch([16, 0, 0], &r), [2, 0, 0]);
        // No overflow at the top of the range.
        let wide = ThreadRules::new([1, 1, 64], [u32::MAX; 3]);
        assert_eq!(launch([u32::MAX, u32::MAX, u32::MAX], &wide), [u32::MAX, u32::MAX, u32::MAX.div_ceil(64)]);
        // A group size of 0 counts as 1 (as the direct path does).
        assert_eq!(launch([5, 1, 1], &ThreadRules::new([0, 0, 0], MAX)), [5, 1, 1]);
    }

    #[test]
    fn launch_over_the_device_limit_runs_nothing() {
        let r = ThreadRules::new([64, 1, 1], [100, 50, 10]);
        assert_eq!(launch([6400, 50, 10], &r), [100, 50, 10], "at the limit");
        assert_eq!(launch([6401, 50, 10], &r), [0; 3], "x one group over");
        assert_eq!(launch([64, 51, 1], &r), [0; 3], "y over");
        assert_eq!(launch([64, 1, 11], &r), [0; 3], "z over");
    }

    /// The rule matches the groups and limits of the equivalent direct
    /// dispatch (`ResolvedDispatch::group_counts`).
    #[test]
    fn launch_matches_direct_dispatch() {
        for local in [[64, 1, 1], [8, 4, 2], [3, 5, 7], [1, 1, 1]] {
            let r = ThreadRules::new(local, MAX);
            for t in [[1, 1, 1], [63, 1, 1], [64, 2, 3], [100, 9, 14], [7, 25, 6]] {
                assert_eq!(launch(t, &r), direct(t, local).group_counts(), "{t:?} in groups of {local:?}");
            }
        }
    }

    #[test]
    fn argument_words() {
        let r = ThreadRules::new([8, 0, 2], [10, 20, 30]);
        let w = words([4, 16, 64], 7, &r);
        assert_eq!(w, [1, 4, 16, 7, 8, 1, 2, 10, 20, 30]);
    }

    /// The kernel on the device against [`launch`].
    #[test]
    #[ignore]
    fn thread_dims_kernel_matches_cpu_rule() {
        use crate::device::{Buffer, MemoryUse};
        use crate::scheduler::Scheduler;
        let dev = GpuDevice::create(false).expect("device");
        let mut sched = Scheduler::new(&dev).unwrap();
        let kernel = IndirectThreadsKernel::new(&dev).expect("kernel");
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
        let record = Buffer::new(&dev, 256, usage, MemoryUse::Upload).unwrap();
        let out = Buffer::new(&dev, 256, usage, MemoryUse::Download).unwrap();
        let cases: [([u32; 3], ThreadRules); 7] = [
            ([13, 5, 3], ThreadRules::new([8, 4, 2], MAX)),
            ([16, 8, 4], ThreadRules::new([8, 4, 2], MAX)),
            ([0, 8, 4], ThreadRules::new([8, 4, 2], MAX)),
            ([100, 1, 1], ThreadRules::new([64, 1, 1], MAX)),
            ([u32::MAX, 1, 1], ThreadRules::new([1, 1, 1], [u32::MAX; 3])),
            ([6401, 1, 1], ThreadRules::new([64, 1, 1], [100, 50, 10])),
            ([64, 1, 11], ThreadRules::new([64, 1, 1], [100, 50, 10])),
        ];
        for (threads, rules) in cases {
            // The record at an offset below the descriptor alignment.
            let rb = record.bytes_mut();
            rb.fill(0);
            for (k, t) in threads.iter().enumerate() {
                rb[20 + k * 4..24 + k * 4].copy_from_slice(&t.to_le_bytes());
            }
            record.flush(0, 256);
            out.bytes_mut().fill(0xA5);
            out.flush(0, 256);
            let cb = sched.cmd();
            kernel.record(
                cb,
                TilerBuffer { buffer: record.buffer, offset: 20, size: RECORD_BYTES },
                TilerBuffer { buffer: out.buffer, offset: 4, size: COMMAND_BYTES },
                TilerBuffer { buffer: out.buffer, offset: 128, size: 64 },
                5,
                &rules,
            );
            sched.flush_and_wait();
            out.invalidate(0, 256);
            let w = |i: usize| u32::from_le_bytes(out.bytes_mut()[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!([w(1), w(2), w(3)], launch(threads, &rules), "command of {threads:?} {rules:?}");
            let data: Vec<u32> = (0..16).map(|i| w(32 + i)).collect();
            let mut expected = [0xA5A5_A5A5u32; 16];
            expected[5..8].copy_from_slice(&threads);
            assert_eq!(data, expected, "shader data of {threads:?}");
        }
        drop(kernel);
        dev.wait_idle();
    }
}
