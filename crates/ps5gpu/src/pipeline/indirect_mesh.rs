// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Indirect draws of mesh-workgroup draws (geometry subgroups and primitive
//! subgroups, see `shader::mesh`) from their argument record on the GPU.
//!
//! The mesh entry point reads the draw's index count and index address
//! from its shader data, and the guest program reads the record's offsets
//! from the user SGPRs the packet names. On the CPU path the record is read
//! back (waiting for the GPU work that wrote it) and becomes a direct draw
//! (`indirect::direct_draw`). Here a one-invocation compute kernel does the
//! same work on the GPU, ordered after the record's writers and before the
//! draw:
//!
//! ```text
//!   (count, instances, first, ...) = record          (one record)
//!   draws  = count word present ? count word != 0 : true
//!   count  = min(count, INDEX_BUFFER_SIZE - first)   (indexed; 0 past it)
//!   groups = count >= size ? (count - size) / step / perGroup + 1 : 0
//!   groups = 0 when over maxMeshWorkGroupCount / TotalCount
//!   command = draws && instances != 0 && groups != 0 ? (groups, instances, 1) : 0
//!   data[draw + 0]     = count
//!   data[draw + 4..6]  = index base + first index * index bytes   (indexed)
//!   data[sgpr dwords]  = base vertex / first vertex, first instance, first index
//! ```
//!
//! `command` is a `VkDrawMeshTasksIndirectCommandEXT` the draw reads with
//! `vkCmdDrawMeshTasksIndirectEXT`; `data` is the stage's shader data the
//! draw binds (a fresh staging copy, see the provider). [`launch`] is the
//! same rule on the CPU (direct mesh draws use it too).

use super::compiler::MeshConfig;
use super::indirect::{AUTO_RECORD, INDEXED_RECORD};
use crate::device::GpuDevice;
use crate::tiler::{Spv, TilerBuffer, R};
use ash::vk;
use rspirv::spirv;
use std::sync::Arc;

/// Push-constant dwords of the kernel.
pub const ARGUMENT_COUNT: usize = 24;
/// A shader-data dword the kernel does not write.
pub const NO_DWORD: u32 = u32::MAX;
/// Bytes of a `VkDrawMeshTasksIndirectCommandEXT`.
pub const COMMAND_BYTES: u64 = 12;

/// How a record's counts become mesh workgroups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshRules {
    /// Vertices of one input primitive and between consecutive ones.
    pub input_size: u32,
    pub input_step: u32,
    pub primitives_per_group: u32,
    /// maxMeshWorkGroupCount[0..2] and maxMeshWorkGroupTotalCount.
    pub max_groups: [u32; 2],
    pub max_total: u32,
}

impl MeshRules {
    pub fn new(m: &MeshConfig, max_groups: [u32; 3], max_total: u32) -> MeshRules {
        MeshRules {
            input_size: m.input_size(),
            input_step: m.input_step(),
            primitives_per_group: m.primitives_per_group.max(1),
            max_groups: [max_groups[0], max_groups[1]],
            max_total,
        }
    }
}

/// Workgroups (x, y = instances, z) of a mesh draw of `count` vertices or
/// indices and `instances` instances: all zero when it draws nothing or
/// exceeds the device's workgroup limits.
pub fn launch(count: u32, instances: u32, r: &MeshRules) -> [u32; 3] {
    if instances == 0 || count < r.input_size {
        return [0; 3];
    }
    let groups = (count - r.input_size) / r.input_step / r.primitives_per_group + 1;
    if groups > r.max_groups[0] || instances > r.max_groups[1] || groups > r.max_total / instances {
        return [0; 3];
    }
    [groups, instances, 1]
}

/// The arguments of one kernel run (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshIndirectArgs {
    pub indexed: bool,
    /// INDEX_BUFFER_SIZE (0: the count is not clamped).
    pub index_limit: u32,
    /// Bytes per index (0: not indexed) and the index buffer base.
    pub index_bytes: u32,
    pub index_base: u64,
    pub rules: MeshRules,
    /// First shader-data dword of the draw parameters.
    pub draw_dword: u32,
    /// Shader-data dwords of the user SGPRs receiving the base (first)
    /// vertex, the first instance and the first index ([`NO_DWORD`]: none).
    /// Written in this order, as the command processor writes them.
    pub vertex_dword: u32,
    pub instance_dword: u32,
    pub first_index_dword: u32,
}

impl MeshIndirectArgs {
    /// Push words: byte bases of the bindings (records, command, count,
    /// shader data) within their aligned descriptors, then the arguments.
    fn words(&self, bases: [u32; 4], has_count: bool) -> [u32; ARGUMENT_COUNT] {
        let mut w = [0u32; ARGUMENT_COUNT];
        w[..4].copy_from_slice(&bases.map(|b| b / 4));
        w[4] = self.indexed as u32 | (has_count as u32) << 1;
        w[5] = self.index_limit;
        w[6] = self.index_bytes;
        w[7] = self.index_base as u32;
        w[8] = (self.index_base >> 32) as u32;
        w[9] = self.rules.input_size;
        w[10] = self.rules.input_step;
        w[11] = self.rules.primitives_per_group;
        w[12] = self.rules.max_groups[0];
        w[13] = self.rules.max_groups[1];
        w[14] = self.rules.max_total;
        w[15] = self.draw_dword;
        w[16] = self.vertex_dword;
        w[17] = self.instance_dword;
        w[18] = self.first_index_dword;
        w
    }

    /// Bytes of the record the kernel reads.
    pub fn record_bytes(&self) -> u64 {
        if self.indexed { INDEXED_RECORD } else { AUTO_RECORD }
    }
}

/// Bindings: 0 the record (read), 1 the command (written), 2 the count
/// word (read; the record again when there is none), 3 the shader data
/// (written).
fn kernel() -> R<Vec<u32>> {
    let mut s = Spv::with_arguments(ARGUMENT_COUNT);
    let (records, command) = (s.src, s.dst);
    let count_word = s.storage_buffer(2, "count_buffer", true);
    let data = s.storage_buffer(3, "shader_data", false);
    let (main, _, _, _) = s.begin_main()?;
    let [rb, cmd, cb, db] = [s.arg(0)?, s.arg(1)?, s.arg(2)?, s.arg(3)?];
    let flags = s.arg(4)?;
    let zero = s.c(0);
    let one = s.c(1);
    let indexed_bit = s.and_c(flags, 1)?;
    let indexed = s.b.i_not_equal(s.boolean, None, indexed_bit, zero)?;
    let has_count_bit = s.and_c(flags, 2)?;
    let has_count = s.b.i_not_equal(s.boolean, None, has_count_bit, zero)?;
    // Record words: count, instances, first index / vertex, then (indexed)
    // base vertex and first instance, else first instance.
    let raw_count = s.load_word(records, rb)?;
    let i1 = s.add_c(rb, 1)?;
    let instances = s.load_word(records, i1)?;
    let i2 = s.add_c(rb, 2)?;
    let first = s.load_word(records, i2)?;
    let vi = s.add(i2, indexed_bit)?;
    let vertex_offset = s.load_word(records, vi)?;
    let ii = s.add_c(vi, 1)?;
    let first_instance = s.load_word(records, ii)?;
    // The draw count in memory (clamped to the one record).
    let stored = s.load_word(count_word, cb)?;
    let stored_some = s.b.i_not_equal(s.boolean, None, stored, zero)?;
    let true_ = s.b.constant_true(s.boolean);
    let draws = s.b.select(s.boolean, None, has_count, stored_some, true_)?;
    // INDEX_BUFFER_SIZE bounds the indices one record may read, counted
    // from the index base: the record's first index uses part of it.
    let size_limit = s.arg(5)?;
    let limit_set = s.b.i_not_equal(s.boolean, None, size_limit, zero)?;
    let inside = s.b.u_less_than(s.boolean, None, first, size_limit)?;
    let left = s.b.i_sub(s.uint, None, size_limit, first)?;
    let limit = s.select(inside, left, zero)?;
    let above = s.b.u_greater_than(s.boolean, None, raw_count, limit)?;
    let limited = s.b.logical_and(s.boolean, None, limit_set, above)?;
    let clamp = s.b.logical_and(s.boolean, None, limited, indexed)?;
    let count = s.select(clamp, limit, raw_count)?;
    // Workgroups (as `launch`).
    let [size, step, per_group, max_x, max_y, max_total] = [s.arg(9)?, s.arg(10)?, s.arg(11)?, s.arg(12)?, s.arg(13)?, s.arg(14)?];
    let complete = s.b.u_greater_than_equal(s.boolean, None, count, size)?;
    let rest = s.b.i_sub(s.uint, None, count, size)?;
    let primitives = s.b.u_div(s.uint, None, rest, step)?;
    let whole = s.b.u_div(s.uint, None, primitives, per_group)?;
    let groups = s.add(whole, one)?;
    let some_instances = s.b.i_not_equal(s.boolean, None, instances, zero)?;
    let divisor = s.select(some_instances, instances, one)?;
    let total = s.b.u_div(s.uint, None, max_total, divisor)?;
    let over_x = s.b.u_greater_than(s.boolean, None, groups, max_x)?;
    let over_y = s.b.u_greater_than(s.boolean, None, instances, max_y)?;
    let over_total = s.b.u_greater_than(s.boolean, None, groups, total)?;
    let over_xy = s.b.logical_or(s.boolean, None, over_x, over_y)?;
    let over = s.b.logical_or(s.boolean, None, over_xy, over_total)?;
    let within = s.b.logical_not(s.boolean, None, over)?;
    let a = s.b.logical_and(s.boolean, None, draws, some_instances)?;
    let b = s.b.logical_and(s.boolean, None, complete, within)?;
    let drawn = s.b.logical_and(s.boolean, None, a, b)?;
    let gx = s.select(drawn, groups, zero)?;
    let gy = s.select(drawn, instances, zero)?;
    let gz = s.select(drawn, one, zero)?;
    for (k, v) in [gx, gy, gz].into_iter().enumerate() {
        let i = s.add_c(cmd, k as u32)?;
        s.store_word(command, i, v)?;
    }
    // Draw parameters: the index count and the record's index address.
    let draw = s.arg(15)?;
    let draw_at = s.add(db, draw)?;
    s.store_word(data, draw_at, count)?;
    let index_bytes = s.arg(6)?;
    let pair = s.b.type_struct([s.uint, s.uint]);
    let product = s.b.u_mul_extended(pair, None, first, index_bytes)?;
    let product_lo = s.b.composite_extract(s.uint, None, product, [0])?;
    let product_hi = s.b.composite_extract(s.uint, None, product, [1])?;
    let base_lo = s.arg(7)?;
    let base_hi = s.arg(8)?;
    let sum = s.b.i_add_carry(pair, None, base_lo, product_lo)?;
    let lo = s.b.composite_extract(s.uint, None, sum, [0])?;
    let carry = s.b.composite_extract(s.uint, None, sum, [1])?;
    let hi_part = s.add(base_hi, product_hi)?;
    let hi = s.add(hi_part, carry)?;
    let lo = s.select(indexed, lo, base_lo)?;
    let hi = s.select(indexed, hi, base_hi)?;
    let lo_at = s.add_c(draw_at, crate::shader::mesh::DRAW_INDEX_ADDRESS)?;
    s.store_word(data, lo_at, lo)?;
    let hi_at = s.add_c(lo_at, 1)?;
    s.store_word(data, hi_at, hi)?;
    // The offsets in the user SGPRs, in the order the packet writes them.
    for (arg, value) in [(16, vertex_offset), (17, first_instance), (18, first)] {
        let dword = s.arg(arg)?;
        let none = s.c(NO_DWORD);
        let set = s.b.i_not_equal(s.boolean, None, dword, none)?;
        let then = s.b.id();
        let merge = s.b.id();
        s.b.selection_merge(merge, spirv::SelectionControl::NONE)?;
        s.b.branch_conditional(set, then, merge, [])?;
        s.b.begin_block(Some(then))?;
        let at = s.add(db, dword)?;
        s.store_word(data, at, value)?;
        s.b.branch(merge)?;
        s.b.begin_block(Some(merge))?;
    }
    s.finish(main, 1, 1)
}

/// The kernel and its layout (push descriptors and push constants).
pub struct IndirectMeshKernel {
    dev: Arc<GpuDevice>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl IndirectMeshKernel {
    /// None when the kernel cannot be built (mesh indirect draws then read
    /// their records on the CPU).
    pub fn new(dev: &Arc<GpuDevice>) -> Option<IndirectMeshKernel> {
        let code = match kernel() {
            Ok(c) => c,
            Err(e) => {
                log::warn!("ps5gpu: mesh indirect kernel assembly failed: {e:?}");
                return None;
            }
        };
        let bindings = [0u32, 1, 2, 3].map(|i| {
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
                r.map_err(|(_, e)| log::warn!("ps5gpu: mesh indirect kernel: {e:?}")).ok().map(|p| p[0])
            });
            let Some(pipeline) = pipeline else {
                d.destroy_pipeline_layout(layout, None);
                d.destroy_descriptor_set_layout(set_layout, None);
                return None;
            };
            Some(IndirectMeshKernel { dev: dev.clone(), set_layout, layout, pipeline })
        }
    }

    /// Record one run (outside any rendering scope): the record at
    /// `record`, the optional draw count word at `count`, the command
    /// written to `command` (at least [`COMMAND_BYTES`]) and the draw
    /// parameters and offsets to the shader data `data`.
    pub fn record(&self, cb: vk::CommandBuffer, record: TilerBuffer, count: Option<TilerBuffer>, command: TilerBuffer, data: TilerBuffer, args: &MeshIndirectArgs) {
        let align = self.dev.caps.min_storage_buffer_offset_alignment.max(4);
        let (ri, rb) = record.binding(align);
        let (ki, kb) = command.binding(align);
        let (ci, cb_base) = count.unwrap_or(record).binding(align);
        let (di, db) = data.binding(align);
        let words = args.words([rb, kb, cb_base, db], count.is_some());
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let infos = [ri, ki, ci, di];
        let writes: [vk::WriteDescriptorSet; 4] = std::array::from_fn(|i| {
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

impl Drop for IndirectMeshKernel {
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
    use crate::pipeline::indirect::direct_draw;
    use crate::render::{DrawKind, DrawPacket, IndirectRegisters, ResolvedDraw};

    const GS_0: u32 = 0x8C;

    fn config(input_primitive: u32, primitives_per_group: u32) -> MeshConfig {
        MeshConfig {
        lines: false,
        points: false,
            input_primitive,
            primitives_per_group,
            vertices_per_group: 0,
            max_vertices: 0,
            max_primitives: 0,
            threads: 64,
            esgs_item_size: 4,
            draw_dword: 0,
        }
    }

    fn rules(input_primitive: u32, primitives_per_group: u32) -> MeshRules {
        MeshRules::new(&config(input_primitive, primitives_per_group), [1000, 600, 600], 4000)
    }

    /// What the kernel does, on the CPU: the command it writes, with the
    /// shader-data words updated in `data`.
    fn resolve(a: &MeshIndirectArgs, record: [u32; 5], count_word: Option<u32>, data: &mut [u32]) -> [u32; 3] {
        let (first, vertex, instance) = if a.indexed { (record[2], record[3], record[4]) } else { (record[2], record[2], record[3]) };
        let count = if a.indexed && a.index_limit != 0 { record[0].min(a.index_limit.saturating_sub(first)) } else { record[0] };
        let command = if count_word.is_none_or(|c| c != 0) { launch(count, record[1], &a.rules) } else { [0; 3] };
        let draw = a.draw_dword as usize;
        data[draw] = count;
        let address = if a.indexed { a.index_base.wrapping_add(first as u64 * a.index_bytes as u64) } else { a.index_base };
        data[draw + 4] = address as u32;
        data[draw + 5] = (address >> 32) as u32;
        for (dword, value) in [(a.vertex_dword, vertex), (a.instance_dword, instance), (a.first_index_dword, first)] {
            if dword != NO_DWORD {
                data[dword as usize] = value;
            }
        }
        command
    }

    #[test]
    fn kernel_assembles() {
        let words = kernel().expect("assembles");
        assert_eq!(words[0], rspirv::spirv::MAGIC_NUMBER);
        let mut loader = rspirv::dr::Loader::new();
        rspirv::binary::parse_words(&words, &mut loader).unwrap();
    }

    #[test]
    fn launch_rules() {
        let (points, triangles, strip, fan) = (rules(1, 1), rules(4, 32), rules(6, 8), rules(5, 8));
        assert_eq!(launch(1, 512, &points), [1, 512, 1], "one point, 512 instances");
        assert_eq!(launch(1, 0, &points), [0; 3], "no instances draws nothing");
        assert_eq!(launch(0, 4, &points), [0; 3], "no vertices draws nothing");
        assert_eq!(launch(96, 2, &triangles), [1, 2, 1], "one full group");
        assert_eq!(launch(99, 2, &triangles), [2, 2, 1], "a partial second group");
        assert_eq!(launch(2, 1, &triangles), [0; 3], "no complete triangle draws nothing");
        assert_eq!(launch(10, 1, &strip), [1, 1, 1], "eight strip triangles are one group");
        assert_eq!(launch(11, 1, &strip), [2, 1, 1], "a ninth strip triangle starts a group");
        assert_eq!(launch(10, 1, &fan), [1, 1, 1], "eight fan triangles are one group");
        assert_eq!(launch(11, 1, &fan), [2, 1, 1], "a ninth fan triangle starts a group");
        assert_eq!(launch(1, 601, &points), [0; 3], "instances over the device limit");
        assert_eq!(launch(96 * 7, 600, &triangles), [0; 3], "groups times instances over the device limit");
        assert_eq!(launch(96 * 6, 600, &triangles), [6, 600, 1], "groups times instances at the device limit");
        assert_eq!(launch(u32::MAX, 1, &points), [0; 3], "groups over the device limit");
        // The ceil(primitives / per group) rule of direct draws.
        for count in 0..400u32 {
            let primitives = if count < 3 { 0 } else { (count - 3) / 3 + 1 };
            assert_eq!(launch(count, 1, &triangles)[0], primitives.div_ceil(32), "{count} indices");
        }
    }

    fn draw(packet: DrawPacket) -> ResolvedDraw {
        let b = crate::render::RegisterBanks { context: &[], shader: &[], uconfig: &[] };
        crate::render::resolve_draw(&b, &packet)
    }

    /// The kernel's rule gives the launch and the shader data the CPU path
    /// gets from the record (`indirect::direct_draw`, then the mesh draw
    /// parameters and workgroups of the direct draw).
    #[test]
    fn records_resolve_as_on_the_cpu() {
        let regs = IndirectRegisters { vertex_offset: GS_0 + 9, instance_offset: GS_0 + 10, first_index: GS_0 + 11, draw_index: crate::render::SH_NOP };
        let r = rules(4, 32);
        let records: [[u32; 5]; 7] = [
            [96, 2, 5, (-17i32) as u32, 3],
            [200, 1, 0, 0, 0],
            [99, 3, 0x4000_0001, 9, 1],
            [0, 4, 5, 6, 7],
            [96, 0, 5, 6, 7],
            [2, 1, 0, 0, 0],
            [1 << 20, 1, 1, 1, 1],
        ];
        for indexed in [true, false] {
            let mut d = draw(DrawPacket {
                kind: if indexed { DrawKind::IndexIndirect } else { DrawKind::Indirect },
                index_base: 0x1_FFFF_FF00,
                index_type: 1,
                index_buffer_size: 150,
                indirect_registers: if indexed { regs } else { IndirectRegisters { first_index: crate::render::SH_NOP, ..regs } },
                ..DrawPacket::default()
            });
            d.index_offset = 7;
            let args = MeshIndirectArgs {
                indexed,
                index_limit: if indexed { 150 } else { 0 },
                index_bytes: if indexed { 4 } else { 0 },
                index_base: if indexed { 0x1_FFFF_FF00 } else { 0 },
                rules: r,
                draw_dword: 8,
                vertex_dword: 1,
                instance_dword: 2,
                first_index_dword: if indexed { 3 } else { NO_DWORD },
            };
            for record in records {
                let mut data = [0xDEAD_BEEFu32; 16];
                let command = resolve(&args, record, None, &mut data);
                match direct_draw(&d, &record, 0) {
                    None => assert_eq!(command, [0; 3], "{record:?} draws nothing"),
                    Some(sub) => {
                        assert_eq!(command, launch(sub.packet.count, sub.packet.instance_count.max(1), &r), "{record:?}");
                        assert_eq!(data[8], sub.packet.count, "{record:?} index count");
                        let address = sub.index.map_or(0, |i| i.address);
                        assert_eq!((data[12], data[13]), (address as u32, (address >> 32) as u32), "{record:?} index address");
                        assert_eq!(sub.index_offset, 0, "no GE_INDX_OFFSET (the template's base stays 0)");
                        let user = &sub.shaders.gs.user_data;
                        assert_eq!((data[1], data[2]), (user[9], user[10]), "{record:?} offsets");
                        if indexed {
                            assert_eq!(data[3], user[11], "{record:?} first index");
                        }
                    }
                }
            }
            // A draw count of zero in memory draws nothing.
            let mut data = [0u32; 16];
            assert_eq!(resolve(&args, records[0], Some(0), &mut data), [0; 3]);
            assert_ne!(resolve(&args, records[0], Some(5), &mut data), [0; 3]);
        }
    }

    #[test]
    fn argument_words() {
        let args = MeshIndirectArgs {
            indexed: true,
            index_limit: 9,
            index_bytes: 2,
            index_base: 0x1234_5678_9ABC,
            rules: rules(6, 8),
            draw_dword: 40,
            vertex_dword: 3,
            instance_dword: NO_DWORD,
            first_index_dword: 5,
        };
        let w = args.words([0, 16, 4, 64], true);
        assert_eq!(&w[..4], &[0, 4, 1, 16]);
        assert_eq!(w[4], 3);
        assert_eq!(&w[5..9], &[9, 2, 0x5678_9ABC, 0x1234]);
        assert_eq!(&w[9..15], &[3, 1, 8, 1000, 600, 4000]);
        assert_eq!(&w[15..19], &[40, 3, NO_DWORD, 5]);
        assert_eq!(args.record_bytes(), INDEXED_RECORD);
    }

    /// The kernel on the device against [`resolve`].
    #[test]
    #[ignore]
    fn kernel_matches_cpu_rule() {
        use crate::device::{Buffer, MemoryUse};
        use crate::scheduler::Scheduler;
        let dev = GpuDevice::create(false).expect("device");
        let mut sched = Scheduler::new(&dev).unwrap();
        let kernel = IndirectMeshKernel::new(&dev).expect("kernel");
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
        let indexed = MeshIndirectArgs {
            indexed: true,
            index_limit: 150,
            index_bytes: 4,
            index_base: 0x1_FFFF_FF00,
            rules: rules(4, 32),
            draw_dword: 8,
            vertex_dword: 1,
            instance_dword: 2,
            first_index_dword: 3,
        };
        let auto = MeshIndirectArgs { indexed: false, index_limit: 0, index_bytes: 0, index_base: 0, first_index_dword: NO_DWORD, ..indexed };
        // Both offsets to one register: the later write (first index) wins.
        let strip = MeshIndirectArgs { rules: rules(6, 8), vertex_dword: NO_DWORD, instance_dword: 1, first_index_dword: 1, ..indexed };
        let cases: [(MeshIndirectArgs, [u32; 5], Option<u32>); 10] = [
            (indexed, [96, 2, 5, (-17i32) as u32, 3], None),
            // INDEX_BUFFER_SIZE 150 from the base: 140 indices after index 10.
            (indexed, [148, 1, 10, 0, 0], None),
            (indexed, [200, 1, 0x4000_0001, 9, 1], None),
            (indexed, [0, 4, 5, 6, 7], None),
            (indexed, [96, 0, 5, 6, 7], None),
            (indexed, [96, 2, 5, 6, 7], Some(0)),
            (indexed, [96, 2, 5, 6, 7], Some(3)),
            (auto, [99, 3, 31, 5, 0xFFFF_FFFF], None),
            (auto, [1 << 20, 1, 1, 1, 1], None),
            (strip, [11, 1, 4, 8, 12], None),
        ];
        let record = Buffer::new(&dev, 256, usage, MemoryUse::Upload).unwrap();
        let out = Buffer::new(&dev, 256, usage, MemoryUse::Download).unwrap();
        for (args, words, count_word) in cases {
            // The record at an offset below the descriptor alignment, the
            // count word after it.
            let rb = record.bytes_mut();
            rb.fill(0);
            for (k, w) in words.iter().enumerate() {
                rb[20 + k * 4..24 + k * 4].copy_from_slice(&w.to_le_bytes());
            }
            rb[64..68].copy_from_slice(&count_word.unwrap_or(0).to_le_bytes());
            record.flush(0, 256);
            out.bytes_mut().fill(0xA5);
            out.flush(0, 256);
            let cb = sched.cmd();
            kernel.record(
                cb,
                TilerBuffer { buffer: record.buffer, offset: 20, size: args.record_bytes() },
                count_word.map(|_| TilerBuffer { buffer: record.buffer, offset: 64, size: 4 }),
                TilerBuffer { buffer: out.buffer, offset: 4, size: COMMAND_BYTES },
                TilerBuffer { buffer: out.buffer, offset: 128, size: 64 },
                &args,
            );
            sched.flush_and_wait();
            out.invalidate(0, 256);
            let w = |i: usize| u32::from_le_bytes(out.bytes_mut()[i * 4..i * 4 + 4].try_into().unwrap());
            let mut data = [0xA5A5_A5A5u32; 16];
            let command = resolve(&args, words, count_word, &mut data);
            assert_eq!([w(1), w(2), w(3)], command, "command of {words:?} {count_word:?}");
            let got: Vec<u32> = (0..16).map(|i| w(32 + i)).collect();
            assert_eq!(got, data, "shader data of {words:?}");
        }
        drop(kernel);
        dev.wait_idle();
    }
}
