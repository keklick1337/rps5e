// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The guest inputs of a module translation, kept in the prewarm list so a
//! module can be translated again by another translator build.
//!
//! A module is the guest program (the code words its decode read, where it
//! continues, the plan inputs) and the compile request the provider built
//! for it: the resource specialization, the binding layout and every stage
//! option, plus the code words the translation itself read (jump tables).
//! [`retranslate`] rebuilds the request from these and runs the compiler on
//! it, exactly as the provider does for a draw: the same permutation key
//! then names the module the current translator makes of the same inputs.
//!
//! The request encoding destructures every field of the types it writes, so
//! a field added to one of them fails the build until it is written here
//! too (and [`REQUEST_VERSION`] is bumped).

use super::compiler::{CompileError, CompiledShader, MeshConfig, ShaderCompileInput, ShaderCompiler};
use super::prewarm::{Reader, Writer};
use ps5shader::resources::{
    apply_specialization, BindingLayout, BufferCandidateTableSpecialization, BufferSpecialization, DescriptorBinding, DescriptorBindingKind,
    ImageDimension, ImageNumericClass, ImageSpecialization, PixelAlphaBlendSource, PlanOptions, ResourceSpecialization, ShaderClipSpaceTransform,
    ShaderCompileRequest, ShaderUserClipPlanes, ShaderResourcePlan, ShaderStage, ShaderVertexInput,
};
use ps5shader::{ColorComponentMapping, ComputeSystemRegisters, PixelOutputBinding, PixelOutputKind};
use std::sync::Arc;

/// Version of the request encoding (a request of another version is not
/// translated again). Version 3 adds the user clip planes, version 4 point
/// meshes and quantized storage writes, version 5 the packed 4_4_4_4
/// UNORM export, version 6 the per-pixel sample masks in the shader data;
/// older requests are not read.
pub const REQUEST_VERSION: u32 = 6;
/// The oldest request encoding read.
const OLDEST_REQUEST_VERSION: u32 = 4;

/// Guest code words by address (sorted runs of consecutive words).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct CodeImage {
    runs: Vec<(u64, Vec<u32>)>,
}

impl CodeImage {
    /// The image of `words` (address, word); later duplicates are ignored.
    pub fn from_words(mut words: Vec<(u64, u32)>) -> CodeImage {
        words.sort_by_key(|&(a, _)| a);
        words.dedup_by_key(|&mut (a, _)| a);
        let mut runs: Vec<(u64, Vec<u32>)> = Vec::new();
        for (a, w) in words {
            match runs.last_mut() {
                Some((start, run)) if *start + run.len() as u64 * 4 == a => run.push(w),
                _ => runs.push((a, vec![w])),
            }
        }
        CodeImage { runs }
    }

    /// The image of `bytes` mapped at `base`.
    pub fn from_bytes(base: u64, bytes: &[u8]) -> CodeImage {
        CodeImage::from_words(bytes.chunks_exact(4).enumerate().map(|(i, c)| (base + i as u64 * 4, u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect())
    }

    pub fn read(&self, address: u64) -> Option<u32> {
        let i = self.runs.partition_point(|(start, _)| *start <= address).checked_sub(1)?;
        let (start, run) = &self.runs[i];
        let off = address.checked_sub(*start)?;
        if off % 4 != 0 {
            return None;
        }
        run.get((off / 4) as usize).copied()
    }

    /// True when the runs are what [`CodeImage::from_words`] makes: word
    /// aligned, non-empty, sorted, apart and inside the 48-bit space (an
    /// image read from a file is checked before use).
    pub fn runs_valid(&self) -> bool {
        let mut end = 0u64;
        for (i, (start, run)) in self.runs.iter().enumerate() {
            let Some(e) = (run.len() as u64).checked_mul(4).and_then(|n| start.checked_add(n)) else { return false };
            if start % 4 != 0 || run.is_empty() || e >> 48 != 0 || (i > 0 && *start <= end) {
                return false;
            }
            end = e;
        }
        true
    }

    /// Every (address, word) of the image.
    pub fn words_iter(&self) -> impl Iterator<Item = (u64, u32)> + '_ {
        self.runs.iter().flat_map(|(start, run)| run.iter().enumerate().map(move |(i, &w)| (start + i as u64 * 4, w)))
    }

    pub fn words(&self) -> usize {
        self.runs.iter().map(|(_, r)| r.len()).sum()
    }

    pub(super) fn put(&self, w: &mut Writer) {
        w.u(self.runs.len() as u32);
        for (start, run) in &self.runs {
            w.q(*start);
            w.words(run);
        }
    }

    pub(super) fn get(r: &mut Reader) -> Option<CodeImage> {
        let n = r.len(1 << 20)?;
        let mut runs = Vec::with_capacity(n);
        for _ in 0..n {
            runs.push((r.q()?, r.words()?));
        }
        Some(CodeImage { runs })
    }
}

impl ps5shader::CodeSource for CodeImage {
    fn read_u32(&self, address: u64) -> Option<u32> {
        self.read(address)
    }
}

/// A guest program as the front end decoded and planned it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProgramSource {
    pub address: u64,
    /// The back half of a split geometry program (0: none).
    pub continuation: u64,
    /// `ShaderStage as u8`.
    pub stage: u8,
    pub user_data_base: u32,
    pub user_data_count: u32,
    pub wave_size: u32,
    /// The hash the plan was extracted under.
    pub plan_hash: u64,
    /// Planned for a stage reading its buffers in guest memory.
    pub guest_plan: bool,
    /// Every word the decode read.
    pub image: Arc<CodeImage>,
}

impl ProgramSource {
    pub fn of(p: &super::frontend::StageProgram) -> ProgramSource {
        ProgramSource {
            address: p.address,
            continuation: p.continuation,
            stage: p.stage as u8,
            user_data_base: p.user_data_base,
            user_data_count: p.user_data_count,
            wave_size: p.wave_size,
            plan_hash: p.plan.hash,
            guest_plan: p.guest_plan,
            image: p.image.clone(),
        }
    }

    pub(super) fn put(&self, w: &mut Writer) {
        let ProgramSource { address, continuation, stage, user_data_base, user_data_count, wave_size, plan_hash, guest_plan, image } = self;
        w.q(*address);
        w.q(*continuation);
        w.u(*stage as u32);
        w.u(*user_data_base);
        w.u(*user_data_count);
        w.u(*wave_size);
        w.q(*plan_hash);
        w.b(*guest_plan);
        image.put(w);
    }

    pub(super) fn get(r: &mut Reader) -> Option<ProgramSource> {
        Some(ProgramSource {
            address: r.q()?,
            continuation: r.q()?,
            stage: u8::try_from(r.u()?).ok()?,
            user_data_base: r.u()?,
            user_data_count: r.u()?,
            wave_size: r.u()?,
            plan_hash: r.q()?,
            guest_plan: r.b()?,
            image: Arc::new(CodeImage::get(r)?),
        })
    }
}

/// Everything one module translation depends on.
#[derive(Clone, Debug)]
pub struct ModuleSource {
    pub program: Arc<ProgramSource>,
    /// [`encode_request`] of the request (stage options applied).
    pub request: Arc<[u32]>,
    /// The code words the translation read (jump tables).
    pub reads: Arc<CodeImage>,
}

pub fn stage_of(v: u8) -> Option<ShaderStage> {
    [ShaderStage::Unknown, ShaderStage::Vertex, ShaderStage::Pixel, ShaderStage::Compute].into_iter().find(|&s| s as u8 == v)
}

// ---------------------------------------------------------------------
// The request
// ---------------------------------------------------------------------

fn opt(w: &mut Writer, v: Option<u32>) {
    match v {
        Some(v) => {
            w.u(1);
            w.u(v);
        }
        None => w.u(0),
    }
}

fn get_opt(r: &mut Reader) -> Option<Option<u32>> {
    Some(if r.b()? { Some(r.u()?) } else { None })
}

fn numeric_code(c: ImageNumericClass) -> u32 {
    match c {
        ImageNumericClass::Unsupported => 0,
        ImageNumericClass::Float => 1,
        ImageNumericClass::Uint => 2,
        ImageNumericClass::Sint => 3,
    }
}

fn numeric_of(v: u32) -> Option<ImageNumericClass> {
    [ImageNumericClass::Unsupported, ImageNumericClass::Float, ImageNumericClass::Uint, ImageNumericClass::Sint].into_iter().find(|&c| numeric_code(c) == v)
}

const DIMENSIONS: [ImageDimension; 8] = [
    ImageDimension::Unknown,
    ImageDimension::Dim1D,
    ImageDimension::Dim1DArray,
    ImageDimension::Dim2D,
    ImageDimension::Dim3D,
    ImageDimension::Dim2DArray,
    ImageDimension::Dim2DMsaa,
    ImageDimension::Dim2DMsaaArray,
];

fn dimension_code(d: ImageDimension) -> u32 {
    match d {
        ImageDimension::Unknown => 0,
        ImageDimension::Dim1D => 1,
        ImageDimension::Dim1DArray => 2,
        ImageDimension::Dim2D => 3,
        ImageDimension::Dim3D => 4,
        ImageDimension::Dim2DArray => 5,
        ImageDimension::Dim2DMsaa => 6,
        ImageDimension::Dim2DMsaaArray => 7,
    }
}

fn kind_code(k: DescriptorBindingKind) -> (u32, u32) {
    match k {
        DescriptorBindingKind::Buffers => (0, 0),
        DescriptorBindingKind::Image(i) => (1, i),
        DescriptorBindingKind::Samplers => (2, 0),
        DescriptorBindingKind::GlobalDataShare => (3, 0),
        DescriptorBindingKind::DeviceAddressPageTable => (4, 0),
        DescriptorBindingKind::FaultBuffer => (5, 0),
        DescriptorBindingKind::FlattenedResourceTable => (6, 0),
        DescriptorBindingKind::ShaderData => (7, 0),
    }
}

fn kind_of(tag: u32, index: u32) -> Option<DescriptorBindingKind> {
    Some(match tag {
        0 => DescriptorBindingKind::Buffers,
        1 => DescriptorBindingKind::Image(index),
        2 => DescriptorBindingKind::Samplers,
        3 => DescriptorBindingKind::GlobalDataShare,
        4 => DescriptorBindingKind::DeviceAddressPageTable,
        5 => DescriptorBindingKind::FaultBuffer,
        6 => DescriptorBindingKind::FlattenedResourceTable,
        7 => DescriptorBindingKind::ShaderData,
        _ => return None,
    })
}

fn put_mesh(w: &mut Writer, mesh: Option<MeshConfig>) {
    let Some(m) = mesh else {
        w.u(0);
        return;
    };
    w.u(1);
    let MeshConfig { input_primitive, primitives_per_group, vertices_per_group, max_vertices, max_primitives, threads, esgs_item_size, draw_dword, lines, points } = m;
    for v in [input_primitive, primitives_per_group, vertices_per_group, max_vertices, max_primitives, threads, esgs_item_size, draw_dword] {
        w.u(v);
    }
    w.b(lines);
    w.b(points);
}

fn get_mesh(r: &mut Reader) -> Option<Option<MeshConfig>> {
    if !r.b()? {
        return Some(None);
    }
    Some(Some(MeshConfig {
        input_primitive: r.u()?,
        primitives_per_group: r.u()?,
        vertices_per_group: r.u()?,
        max_vertices: r.u()?,
        max_primitives: r.u()?,
        threads: r.u()?,
        esgs_item_size: r.u()?,
        draw_dword: r.u()?,
        lines: r.b()?,
        points: r.b()?,
    }))
}

fn put_specialization(w: &mut Writer, s: &ResourceSpecialization) {
    let ResourceSpecialization { base_buffer_count, buffers, images, buffer_candidate_tables } = s;
    w.u(*base_buffer_count as u32);
    w.u(buffers.len() as u32);
    for b in buffers {
        let BufferSpecialization { packed_stride, descriptor_format, descriptor_swizzle } = *b;
        w.u(packed_stride);
        w.u(descriptor_format);
        w.u(descriptor_swizzle);
    }
    w.u(images.len() as u32);
    for i in images {
        let ImageSpecialization {
            numeric_class,
            dimension,
            mip_count,
            conversion_format,
            shader_swizzle,
            indirect_root,
            indirect_mapping_offset,
            indirect_search_iterations,
            cube,
            emulated_compare,
            bindless,
            packed_format,
            store_quantize,
        } = *i;
        w.u(numeric_code(numeric_class));
        w.u(dimension_code(dimension));
        for v in [mip_count, conversion_format, shader_swizzle, indirect_root, indirect_mapping_offset, indirect_search_iterations] {
            w.u(v);
        }
        w.b(cube);
        w.u(emulated_compare);
        w.b(bindless);
        w.u(packed_format);
        w.u(store_quantize);
    }
    w.u(buffer_candidate_tables.len() as u32);
    for t in buffer_candidate_tables {
        let BufferCandidateTableSpecialization { first_candidate, candidate_count, mapping_offset, search_iterations } = *t;
        for v in [first_candidate, candidate_count, mapping_offset, search_iterations] {
            w.u(v);
        }
    }
}

fn get_specialization(r: &mut Reader) -> Option<ResourceSpecialization> {
    let base_buffer_count = r.u()? as usize;
    let n = r.len(1 << 16)?;
    let buffers = (0..n).map(|_| Some(BufferSpecialization { packed_stride: r.u()?, descriptor_format: r.u()?, descriptor_swizzle: r.u()? })).collect::<Option<Vec<_>>>()?;
    let n = r.len(1 << 16)?;
    let images = (0..n)
        .map(|_| {
            Some(ImageSpecialization {
                numeric_class: numeric_of(r.u()?)?,
                dimension: *DIMENSIONS.get(r.u()? as usize)?,
                mip_count: r.u()?,
                conversion_format: r.u()?,
                shader_swizzle: r.u()?,
                indirect_root: r.u()?,
                indirect_mapping_offset: r.u()?,
                indirect_search_iterations: r.u()?,
                cube: r.b()?,
                emulated_compare: r.u()?,
                bindless: r.b()?,
                packed_format: r.u()?,
                store_quantize: r.u()?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let n = r.len(1 << 16)?;
    let buffer_candidate_tables = (0..n)
        .map(|_| Some(BufferCandidateTableSpecialization { first_candidate: r.u()?, candidate_count: r.u()?, mapping_offset: r.u()?, search_iterations: r.u()? }))
        .collect::<Option<Vec<_>>>()?;
    Some(ResourceSpecialization { base_buffer_count, buffers, images, buffer_candidate_tables })
}

fn put_layout(w: &mut Writer, l: &BindingLayout) {
    let BindingLayout {
        push_data_start_dword,
        allocation_cursor,
        shader_base_dword,
        memory_offset_dword,
        memory_offset_count,
        wide_offsets,
        uses_dispatch_thread_limits,
        user_data_registers,
        descriptors,
        bindless_tables,
        quad_masks_dword,
    } = l;
    for v in [*push_data_start_dword, *allocation_cursor, *shader_base_dword, *memory_offset_dword, *memory_offset_count] {
        w.u(v);
    }
    w.b(*wide_offsets);
    w.b(*uses_dispatch_thread_limits);
    w.words(user_data_registers);
    w.u(descriptors.len() as u32);
    for d in descriptors {
        let DescriptorBinding { kind, resources } = d;
        let (tag, index) = kind_code(*kind);
        w.u(tag);
        w.u(index);
        w.words(resources);
    }
    w.b(*bindless_tables);
    w.u(*quad_masks_dword);
}

/// A layout of request `version` (before 6: no per-pixel sample masks in
/// the shader data; such a module keeps them as constants).
fn get_layout(r: &mut Reader, version: u32) -> Option<BindingLayout> {
    let push_data_start_dword = r.u()?;
    let allocation_cursor = r.u()?;
    let shader_base_dword = r.u()?;
    let memory_offset_dword = r.u()?;
    let memory_offset_count = r.u()?;
    let wide_offsets = r.b()?;
    let uses_dispatch_thread_limits = r.b()?;
    let user_data_registers = r.words()?;
    let n = r.len(1024)?;
    let descriptors = (0..n).map(|_| Some(DescriptorBinding { kind: kind_of(r.u()?, r.u()?)?, resources: r.words()? })).collect::<Option<Vec<_>>>()?;
    let bindless_tables = r.b()?;
    let quad_masks_dword = if version >= 6 { r.u()? } else { ps5shader::resources::NO_QUAD_MASKS };
    Some(BindingLayout {
        push_data_start_dword,
        allocation_cursor,
        shader_base_dword,
        memory_offset_dword,
        memory_offset_count,
        wide_offsets,
        uses_dispatch_thread_limits,
        user_data_registers,
        descriptors,
        bindless_tables,
        quad_masks_dword,
    })
}

fn output_kind_code(k: PixelOutputKind) -> u32 {
    match k {
        PixelOutputKind::Float => 0,
        PixelOutputKind::Uint => 1,
        PixelOutputKind::Sint => 2,
    }
}

fn alpha_source_code(s: PixelAlphaBlendSource) -> u32 {
    match s {
        PixelAlphaBlendSource::None => 0,
        PixelAlphaBlendSource::SourceAlpha => 1,
        PixelAlphaBlendSource::SourceAlphaOne => 2,
        PixelAlphaBlendSource::SourceAlphaZero => 3,
    }
}

/// The stage options of a request (every field not derived from the plan
/// and the resources by `ShaderCompileRequest::new`).
fn put_options(w: &mut Writer, r: &ShaderCompileRequest<'_>) {
    let ShaderCompileRequest {
        // Derived from the plan and the resources (`new`).
        plan: _,
        resources: _,
        bindings: _,
        uses_flattened_table: _,
        uses_global_data_share: _,
        reads_shader_base: _,
        flattened_slot_by_memory_index: _,
        flattened_table_reserved_words: _,
        indirect_key_memory_indices: _,
        indirect_offset_key_memory_indices: _,
        indirect_root_by_memory_index: _,
        buffer_candidate_table_by_memory_index: _,
        written_range_slot_by_memory_index: _,
        // Stage options.
        trace_device_address_faults,
        wave_size,
        enable_exec_guard_elision,
        scratch_dwords,
        enable_graphics_subgroup_operations,
        supports_shared_int64_atomics,
        compute_system_registers,
        float_mode,
        pixel_outputs,
        color_export_format,
        pixel_input_enable,
        pixel_custom_interpolation_mask,
        supports_per_vertex_pixel_inputs,
        pixel_input_address,
        pixel_input_cntl,
        front_face_all_bits,
        pixel_raster_primitive,
        pixel_ordered,
        pixel_sample_rate,
        required_vertex_output_count,
        vertex_inputs,
        position_export_control,
        supports_clip_distance,
        pixel_dual_source_blend,
        pixel_depth_export,
        pixel_alpha_to_coverage,
        pixel_stencil_export,
        pixel_stencil_op_export,
        pixel_early_fragment_tests,
        pixel_quad_sample_masks,
        pixel_alpha_blend_source,
        clip_space,
        user_clip_planes,
        local_data_share_dwords,
        local_size,
        thread_count,
        indirect_draw,
        relocatable_code,
    } = r;
    w.b(*trace_device_address_faults);
    w.u(*wave_size);
    w.b(*enable_exec_guard_elision);
    w.u(*scratch_dwords);
    w.b(*enable_graphics_subgroup_operations);
    w.b(*supports_shared_int64_atomics);
    match compute_system_registers {
        Some(ComputeSystemRegisters { work_group_x, work_group_y, work_group_z, thread_group_size }) => {
            w.u(1);
            for v in [*work_group_x, *work_group_y, *work_group_z, *thread_group_size] {
                opt(w, v);
            }
        }
        None => w.u(0),
    }
    let ps5shader::translate::FloatMode { mode, dx10_clamp, ieee, fp16_overflow } = *float_mode;
    w.u(mode as u32);
    w.b(dx10_clamp);
    w.b(ieee);
    w.b(fp16_overflow);
    w.u(pixel_outputs.len() as u32);
    for o in pixel_outputs {
        let PixelOutputBinding { guest_slot, host_location, kind, component_mapping, export_target, packed_unorm_11_11_10, packed_unorm_4_4_4_4 } = *o;
        let ColorComponentMapping { packed } = component_mapping;
        w.u(guest_slot);
        w.u(host_location);
        w.u(output_kind_code(kind));
        w.u(packed as u32);
        opt(w, export_target);
        opt(w, packed_unorm_11_11_10.map(u32::from));
        w.b(packed_unorm_4_4_4_4);
    }
    w.u(*color_export_format);
    w.u(*pixel_input_enable);
    w.u(*pixel_custom_interpolation_mask);
    w.b(*supports_per_vertex_pixel_inputs);
    w.u(*pixel_input_address);
    match pixel_input_cntl {
        Some(c) => {
            w.u(1);
            w.words(c);
        }
        None => w.u(0),
    }
    w.b(*front_face_all_bits);
    w.u(*pixel_raster_primitive);
    w.b(*pixel_ordered);
    w.b(*pixel_sample_rate);
    w.i(*required_vertex_output_count);
    w.u(vertex_inputs.len() as u32);
    for v in vertex_inputs {
        let ShaderVertexInput { pc, location, fetch_component_count, component_count, number_format, destination_select, per_instance, alias_pcs } = v;
        for x in [*pc, *location, *fetch_component_count, *component_count, *number_format, *destination_select] {
            w.u(x);
        }
        w.b(*per_instance);
        w.words(alias_pcs);
    }
    w.u(*position_export_control);
    w.b(*supports_clip_distance);
    w.b(*pixel_dual_source_blend);
    w.b(*pixel_depth_export);
    w.b(*pixel_alpha_to_coverage);
    w.b(*pixel_stencil_export);
    w.b(*pixel_stencil_op_export);
    w.b(*pixel_early_fragment_tests);
    match pixel_quad_sample_masks {
        Some(m) => {
            w.u(1);
            for v in m {
                w.u(*v);
            }
        }
        None => w.u(0),
    }
    w.u(alpha_source_code(*pixel_alpha_blend_source));
    let ShaderClipSpaceTransform { enabled, scale_x, scale_y, offset_x, offset_y, half_extent_x, half_extent_y } = *clip_space;
    w.b(enabled);
    for f in [scale_x, scale_y, offset_x, offset_y, half_extent_x, half_extent_y] {
        w.u(f.to_bits());
    }
    w.u(*local_data_share_dwords);
    for v in local_size.iter().chain(thread_count) {
        w.u(*v);
    }
    opt(w, indirect_draw.map(|[a, b]| a as u32 | (b as u32) << 8));
    let ShaderUserClipPlanes { mask, planes } = *user_clip_planes;
    w.u(mask);
    for v in planes {
        w.u(v);
    }
    // Last: [`relocatable`] reads it from the end.
    w.b(*relocatable_code);
}

fn get_options(r: &mut Reader, q: &mut ShaderCompileRequest<'_>, version: u32) -> Option<()> {
    q.trace_device_address_faults = r.b()?;
    q.wave_size = r.u()?;
    q.enable_exec_guard_elision = r.b()?;
    q.scratch_dwords = r.u()?;
    q.enable_graphics_subgroup_operations = r.b()?;
    q.supports_shared_int64_atomics = r.b()?;
    q.compute_system_registers = if r.b()? {
        Some(ComputeSystemRegisters { work_group_x: get_opt(r)?, work_group_y: get_opt(r)?, work_group_z: get_opt(r)?, thread_group_size: get_opt(r)? })
    } else {
        None
    };
    q.float_mode = ps5shader::translate::FloatMode { mode: u8::try_from(r.u()?).ok()?, dx10_clamp: r.b()?, ieee: r.b()?, fp16_overflow: r.b()? };
    let n = r.len(64)?;
    q.pixel_outputs = (0..n)
        .map(|_| {
            let (guest_slot, host_location, kind) = (r.u()?, r.u()?, r.u()?);
            Some(PixelOutputBinding {
                guest_slot,
                host_location,
                kind: [PixelOutputKind::Float, PixelOutputKind::Uint, PixelOutputKind::Sint].into_iter().find(|&k| output_kind_code(k) == kind)?,
                component_mapping: ColorComponentMapping { packed: u8::try_from(r.u()?).ok()? },
                export_target: get_opt(r)?,
                packed_unorm_11_11_10: get_opt(r)?.map(|v| v != 0),
                packed_unorm_4_4_4_4: version >= 5 && r.b()?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    q.color_export_format = r.u()?;
    q.pixel_input_enable = r.u()?;
    q.pixel_custom_interpolation_mask = r.u()?;
    q.supports_per_vertex_pixel_inputs = r.b()?;
    q.pixel_input_address = r.u()?;
    q.pixel_input_cntl = if r.b()? { Some(r.words()?) } else { None };
    q.front_face_all_bits = r.b()?;
    q.pixel_raster_primitive = r.u()?;
    q.pixel_ordered = r.b()?;
    q.pixel_sample_rate = r.b()?;
    q.required_vertex_output_count = r.i()?;
    let n = r.len(1024)?;
    q.vertex_inputs = (0..n)
        .map(|_| {
            Some(ShaderVertexInput {
                pc: r.u()?,
                location: r.u()?,
                fetch_component_count: r.u()?,
                component_count: r.u()?,
                number_format: r.u()?,
                destination_select: r.u()?,
                per_instance: r.b()?,
                alias_pcs: r.words()?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    q.position_export_control = r.u()?;
    q.supports_clip_distance = r.b()?;
    q.pixel_dual_source_blend = r.b()?;
    q.pixel_depth_export = r.b()?;
    q.pixel_alpha_to_coverage = r.b()?;
    q.pixel_stencil_export = r.b()?;
    q.pixel_stencil_op_export = r.b()?;
    q.pixel_early_fragment_tests = r.b()?;
    q.pixel_quad_sample_masks = if r.b()? { Some([r.u()?, r.u()?, r.u()?, r.u()?]) } else { None };
    let code = r.u()?;
    q.pixel_alpha_blend_source = [PixelAlphaBlendSource::None, PixelAlphaBlendSource::SourceAlpha, PixelAlphaBlendSource::SourceAlphaOne, PixelAlphaBlendSource::SourceAlphaZero]
        .into_iter()
        .find(|&s| alpha_source_code(s) == code)?;
    let enabled = r.b()?;
    let f = |r: &mut Reader| r.u().map(f32::from_bits);
    q.clip_space =
        ShaderClipSpaceTransform { enabled, scale_x: f(r)?, scale_y: f(r)?, offset_x: f(r)?, offset_y: f(r)?, half_extent_x: f(r)?, half_extent_y: f(r)? };
    q.local_data_share_dwords = r.u()?;
    q.local_size = [r.u()?, r.u()?, r.u()?];
    q.thread_count = [r.u()?, r.u()?, r.u()?];
    q.indirect_draw = get_opt(r)?.map(|v| [v as u8, (v >> 8) as u8]);
    if version >= 3 {
        let mask = r.u()?;
        let mut planes = [0u32; 24];
        for v in &mut planes {
            *v = r.u()?;
        }
        q.user_clip_planes = ShaderUserClipPlanes { mask, planes };
    }
    if version >= 6 {
        q.relocatable_code = r.u()? != 0;
    }
    Some(())
}

/// The words of a compile request (its stage options applied), the
/// specialization it was made for and the geometry configuration.
pub fn encode_request(r: &ShaderCompileRequest<'_>, specialization: &ResourceSpecialization, mesh: Option<MeshConfig>) -> Arc<[u32]> {
    let mut w = Writer(Vec::new());
    w.u(REQUEST_VERSION);
    put_mesh(&mut w, mesh);
    put_specialization(&mut w, specialization);
    w.b(r.resources.info.buffers_in_guest_memory);
    put_layout(&mut w, &r.bindings);
    put_options(&mut w, r);
    w.0.into()
}

/// Whether the encoded `request` is of a relocatable module (the code
/// address read from the shader data): its last word, 1, or 2 once a
/// translation at another address proved the module the same.
pub fn relocatable(request: &[u32]) -> bool {
    request.first().is_some_and(|&v| v >= 6) && matches!(request.last(), Some(1 | 2))
}

/// Whether `request` is of a relocatable module proven address independent.
pub fn proven_relocatable(request: &[u32]) -> bool {
    request.first().is_some_and(|&v| v >= 6) && request.last() == Some(&2)
}

/// `request` of a relocatable module, marked proven.
pub fn mark_proven(request: &[u32]) -> Arc<[u32]> {
    let mut r = request.to_vec();
    if let Some(l) = r.last_mut().filter(|_| relocatable(request)) {
        *l = 2;
    }
    r.into()
}

/// The parts of an encoded request (mesh, specialization, the
/// guest-memory flag, binding layout, stage options), as word slices: what
/// two permutations of one program differ in (`shader-cache info`).
pub fn request_parts(request: &[u32]) -> Option<[&[u32]; 5]> {
    let mut r = Reader(request);
    let version = r.u().filter(|v| (OLDEST_REQUEST_VERSION..=REQUEST_VERSION).contains(v))?;
    let mut parts = [&request[..0]; 5];
    let mut at = request.len() - r.0.len();
    for (i, part) in parts.iter_mut().enumerate().take(4) {
        match i {
            0 => get_mesh(&mut r).map(|_| ())?,
            1 => get_specialization(&mut r).map(|_| ())?,
            2 => r.b().map(|_| ())?,
            _ => get_layout(&mut r, version).map(|_| ())?,
        }
        let end = request.len() - r.0.len();
        *part = &request[at..end];
        at = end;
    }
    parts[4] = &request[at..];
    Some(parts)
}

fn failed(what: &str) -> CompileError {
    CompileError::Failed(format!("prewarm inputs: {what}"))
}

/// Translate the module `key` of `stage` again from its inputs `src`.
pub fn retranslate(compiler: &dyn ShaderCompiler, key: u64, src: &ModuleSource) -> Result<CompiledShader, CompileError> {
    let address = src.program.address;
    with_request(src, |stage, mesh, specialization, request| {
        let input = ShaderCompileInput { stage, mesh, request, specialization, guest_address: address, key, code: Some(&*src.reads) };
        compiler.compile(&input)
    })?
}

/// The image moved by `delta` bytes (wrapping in the 64-bit space).
impl CodeImage {
    pub fn rebased(&self, delta: u64) -> CodeImage {
        CodeImage { runs: self.runs.iter().map(|(a, r)| (a.wrapping_add(delta), r.clone())).collect() }
    }
}

/// How far a relocation trial moves a program: a 256-byte multiple that
/// changes the bits a page or a cache line would keep and the high half
/// (a 32-bit half of the address left in the module differs too).
const TRIAL_DELTA: u64 = 0x0001_2345_6700;

/// The inputs of `src` as if the game had loaded the program `delta` bytes
/// further (its code, and the words its translation read, moved along).
pub fn rebased(src: &ModuleSource, delta: u64) -> ModuleSource {
    let p = &*src.program;
    ModuleSource {
        program: Arc::new(ProgramSource {
            address: p.address.wrapping_add(delta),
            continuation: if p.continuation == 0 { 0 } else { p.continuation.wrapping_add(delta) },
            image: Arc::new(p.image.rebased(delta)),
            ..p.clone()
        }),
        request: src.request.clone(),
        reads: Arc::new(src.reads.rebased(delta)),
    }
}

/// Whether the relocatable module `compiled` of `src` (translated at its
/// address) is the module of the same program loaded anywhere else: the
/// program is translated again at another address (past any module
/// cache) and must give the same SPIR-V. Anything of the program that
/// still depended on where it is loaded would differ there.
pub fn proves_relocatable(compiler: &dyn ShaderCompiler, key: u64, src: &ModuleSource, compiled: &CompiledShader) -> bool {
    if !relocatable(&src.request) {
        return false;
    }
    let a = src.program.address;
    let delta = trial_delta(a);
    let moved = rebased(src, delta);
    let address = moved.program.address;
    with_request(&moved, |stage, mesh, specialization, request| {
        let input = ShaderCompileInput { stage, mesh, request, specialization, guest_address: address, key, code: Some(&*moved.reads) };
        compiler.compile_uncached(&input).is_ok_and(|c| c.spirv == compiled.spirv)
    })
    .unwrap_or(false)
}

/// `src` with its request asking for a relocatable module (None: the
/// program does not read its address, is split, or has no shader base).
pub fn as_relocatable(src: &ModuleSource) -> Option<ModuleSource> {
    if src.program.continuation != 0 {
        return None;
    }
    let request = with_request(src, |_, mesh, specialization, r| {
        if !r.reads_shader_base || !r.bindings.uses_shader_base() {
            return None;
        }
        let mut r = r.clone();
        r.relocatable_code = true;
        Some(encode_request(&r, specialization, mesh))
    })
    .ok()??;
    Some(ModuleSource { program: src.program.clone(), request, reads: src.reads.clone() })
}

/// Whether the plain (not relocatable) modules of `src` at its address and
/// at the trial address differ: the program's address reaches the module
/// (None: not translated).
pub fn plain_translations_differ(compiler: &dyn ShaderCompiler, key: u64, src: &ModuleSource) -> Option<bool> {
    let plain = |s: &ModuleSource| {
        let address = s.program.address;
        with_request(s, |stage, mesh, specialization, request| {
            let mut request = request.clone();
            request.relocatable_code = false;
            let input = ShaderCompileInput { stage, mesh, request: &request, specialization, guest_address: address, key, code: Some(&*s.reads) };
            compiler.compile_uncached(&input).ok().map(|c| c.spirv)
        })
        .ok()
        .flatten()
    };
    let moved = rebased(src, trial_delta(src.program.address));
    Some(plain(src)? != plain(&moved)?)
}

fn trial_delta(a: u64) -> u64 {
    if a.checked_add(TRIAL_DELTA).is_some_and(|e| e >> 47 == 0) {
        TRIAL_DELTA
    } else {
        TRIAL_DELTA.wrapping_neg()
    }
}

/// Translate the relocatable module of `src` at its address (past any
/// module cache).
pub fn translate_uncached(compiler: &dyn ShaderCompiler, key: u64, src: &ModuleSource) -> Result<CompiledShader, CompileError> {
    let address = src.program.address;
    with_request(src, |stage, mesh, specialization, request| {
        let input = ShaderCompileInput { stage, mesh, request, specialization, guest_address: address, key, code: Some(&*src.reads) };
        compiler.compile_uncached(&input)
    })?
}

/// The inputs of the module of `src`, one `name = value` line per field
/// (`shader-cache info`: what two permutations of a program differ in).
pub fn request_fields(src: &ModuleSource) -> Result<Vec<String>, CompileError> {
    with_request(src, |_, mesh, specialization, r| {
        let mut out = vec![format!("mesh = {mesh:?}")];
        out.extend(specialization.buffers.iter().enumerate().map(|(i, b)| format!("specialization.buffer[{i}] = {b:?}")));
        out.extend(specialization.images.iter().enumerate().map(|(i, b)| format!("specialization.image[{i}] = {b:?}")));
        out.push(format!("specialization.candidate_tables = {:?}", specialization.buffer_candidate_tables));
        out.push(format!("buffers_in_guest_memory = {}", r.resources.info.buffers_in_guest_memory));
        out.push(format!("layout = {:?}", r.bindings));
        let ShaderCompileRequest {
            plan: _,
            resources: _,
            bindings: _,
            uses_flattened_table: _,
            uses_global_data_share: _,
            reads_shader_base: _,
            flattened_slot_by_memory_index: _,
            flattened_table_reserved_words: _,
            indirect_key_memory_indices: _,
            indirect_offset_key_memory_indices: _,
            indirect_root_by_memory_index: _,
            buffer_candidate_table_by_memory_index: _,
            written_range_slot_by_memory_index: _,
            trace_device_address_faults,
            wave_size,
            enable_exec_guard_elision,
            scratch_dwords,
            enable_graphics_subgroup_operations,
            supports_shared_int64_atomics,
            compute_system_registers,
            float_mode,
            pixel_outputs,
            color_export_format,
            pixel_input_enable,
            pixel_custom_interpolation_mask,
            supports_per_vertex_pixel_inputs,
            pixel_input_address,
            pixel_input_cntl,
            front_face_all_bits,
            pixel_raster_primitive,
            pixel_ordered,
            pixel_sample_rate,
            required_vertex_output_count,
            vertex_inputs,
            position_export_control,
            supports_clip_distance,
            pixel_dual_source_blend,
            pixel_depth_export,
            pixel_alpha_to_coverage,
            pixel_stencil_export,
            pixel_stencil_op_export,
            pixel_early_fragment_tests,
            pixel_quad_sample_masks,
            pixel_alpha_blend_source,
            clip_space,
            user_clip_planes,
            local_data_share_dwords,
            local_size,
            thread_count,
            indirect_draw,
            relocatable_code,
        } = r;
        macro_rules! fields {
            ($($f:ident),*) => { $( out.push(format!(concat!(stringify!($f), " = {:?}"), $f)); )* };
        }
        fields!(
            trace_device_address_faults, wave_size, enable_exec_guard_elision, scratch_dwords, enable_graphics_subgroup_operations, supports_shared_int64_atomics,
            compute_system_registers, float_mode, pixel_outputs, color_export_format, pixel_input_enable, pixel_custom_interpolation_mask,
            supports_per_vertex_pixel_inputs, pixel_input_address, pixel_input_cntl, front_face_all_bits, pixel_raster_primitive, pixel_ordered,
            pixel_sample_rate, required_vertex_output_count, vertex_inputs, position_export_control, supports_clip_distance, pixel_dual_source_blend,
            pixel_depth_export, pixel_alpha_to_coverage, pixel_stencil_export, pixel_stencil_op_export, pixel_early_fragment_tests,
            pixel_quad_sample_masks, pixel_alpha_blend_source, clip_space, user_clip_planes, local_data_share_dwords, local_size, thread_count,
            indirect_draw, relocatable_code
        );
        out
    })
}

/// Decode the request of `src` (the program decoded and planned from its
/// code image) and hand it to `f`.
fn with_request<R>(
    src: &ModuleSource,
    f: impl FnOnce(ShaderStage, Option<MeshConfig>, &ResourceSpecialization, &ShaderCompileRequest<'_>) -> R,
) -> Result<R, CompileError> {
    let p = &*src.program;
    let stage = stage_of(p.stage).ok_or_else(|| failed("stage"))?;
    let program = super::frontend::decode_code(&*p.image, p.address, p.continuation).map_err(|e| CompileError::Failed(format!("decode: {e}")))?;
    let options = if p.guest_plan { crate::buffers::guest::guest_plan_options() } else { PlanOptions::default() };
    let plan = ShaderResourcePlan::extract_with(program, stage, p.plan_hash, p.user_data_base, p.user_data_count, None, p.wave_size, options)
        .map_err(|e| CompileError::Failed(format!("plan: {e}")))?;
    let mut r = Reader(&src.request);
    let version = r.u().filter(|v| (OLDEST_REQUEST_VERSION..=REQUEST_VERSION).contains(v)).ok_or_else(|| failed("request version"))?;
    let mesh = get_mesh(&mut r).ok_or_else(|| failed("mesh"))?;
    let specialization = get_specialization(&mut r).ok_or_else(|| failed("specialization"))?;
    let in_guest_memory = r.b().ok_or_else(|| failed("resources"))?;
    let layout = get_layout(&mut r, version).ok_or_else(|| failed("layout"))?;
    let mut resources = apply_specialization(&plan, &specialization).map_err(|e| CompileError::Failed(format!("specialize: {e}")))?;
    resources.info.buffers_in_guest_memory = in_guest_memory;
    let mut request = ShaderCompileRequest::new(&plan, resources, layout);
    request.wave_size = p.wave_size;
    get_options(&mut r, &mut request, version).ok_or_else(|| failed("options"))?;
    if !r.0.is_empty() {
        return Err(failed("trailing words"));
    }
    Ok(f(stage, mesh, &specialization, &request))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::prewarm::PrewarmList;
    use ps5shader::resources::ShaderResourceInfo;

    fn features() -> crate::shader::Features {
        crate::shader::Features {
            barycentric: true,
            demote: true,
            subgroup_size: 32,
            subgroup64_compute: false,
            max_compute_workgroup_subgroups: 32,
            max_shared_bytes: 49152,
            sample_offset: false,
            image_int64_atomics: true,
            shared_int64_atomics: true,
            compute_derivatives: true,
            compute_derivatives_linear: true,
            preserve_inf_nan: [true; 3],
            denorm_preserve: [true; 3],
            denorm_flush_to_zero32: true,
            f16_rte: false,
            device_clock: Some(crate::clock::DeviceClock { num: 1, den: 10 }),
            subgroup_clock: true,
            bindless_tables: true,
            no_float64: false,
            amd_proprietary: false,
            emit: crate::shader::EmitOptions::ALL,
        }
    }

    /// A captured compute program as the front end would prepare it for a
    /// dispatch: its source, request (stage options set) and specialization.
    fn captured_compute() -> (ProgramSource, impl Fn(&dyn Fn(&ShaderCompileRequest<'_>, &ResourceSpecialization))) {
        captured_compute_from("0081df7de03c5d27", 0x50_0a00_0000)
    }

    /// [`captured_compute`] of corpus program `name` loaded at `address`.
    fn captured_compute_from(name: &str, address: u64) -> (ProgramSource, impl Fn(&dyn Fn(&ShaderCompileRequest<'_>, &ResourceSpecialization))) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../ps5shader/tests/corpus/{name}.bin"));
        let bytes = std::fs::read(path).expect("corpus program");
        let image = Arc::new(CodeImage::from_bytes(address, &bytes));
        let source = ProgramSource {
            address,
            continuation: 0,
            stage: ShaderStage::Compute as u8,
            user_data_base: 0,
            user_data_count: 14,
            wave_size: 64,
            plan_hash: 0x1234,
            guest_plan: false,
            image,
        };
        let s = source.clone();
        let with_request = move |f: &dyn Fn(&ShaderCompileRequest<'_>, &ResourceSpecialization)| {
            let program = super::super::frontend::decode_code(&*s.image, s.address, 0).expect("decodes");
            let user = BindingLayout::collect_user_data_registers(&program, 0, 14);
            let gds = BindingLayout::uses_global_data_share(&program);
            let base = BindingLayout::reads_shader_base(&program);
            let plan = ShaderResourcePlan::extract(program, ShaderStage::Compute, s.plan_hash, 0, 14, None, 64).expect("plans");
            let info: &ShaderResourceInfo = &plan.info;
            let mut spec = ResourceSpecialization::default_for(info);
            for b in &mut spec.buffers {
                b.descriptor_format = 77;
            }
            let resources = apply_specialization(&plan, &spec).expect("specializes");
            let flat = ShaderCompileRequest::requires_flattened_table(&plan, &resources);
            let layout = BindingLayout::allocate(&resources.info, user, gds, flat, base, 0, false, false).expect("lays out");
            let mut r = ShaderCompileRequest::new(&plan, resources, layout);
            r.wave_size = 64;
            r.local_size = [64, 1, 1];
            r.compute_system_registers = Some(ComputeSystemRegisters { work_group_x: Some(14), work_group_y: Some(15), work_group_z: None, thread_group_size: None });
            r.local_data_share_dwords = 512;
            r.float_mode = ps5shader::translate::FloatMode { mode: 0xF0, dx10_clamp: false, ieee: true, fp16_overflow: false };
            f(&r, &spec);
        };
        (source, with_request)
    }

    /// Game compute programs that read their own address (S_GETPC_B64):
    /// their plain modules name the address (a module per address the game
    /// loads them at), their relocatable modules are the same SPIR-V
    /// wherever they are loaded, and that is what the trial proves.
    #[test]
    fn programs_reading_their_address_translate_to_one_relocatable_module() {
        let compiler = crate::shader::SpirvCompiler::new(features(), None);
        for name in ["7fe2487938a7183d", "d00319ddf5bbd854"] {
            let at = |address: u64, relocatable: bool| {
                let (program, with_request) = captured_compute_from(name, address);
                let out = std::cell::RefCell::new(None);
                with_request(&|r, spec| {
                    assert!(r.reads_shader_base && r.bindings.uses_shader_base(), "{name} reads its address");
                    let mut r = r.clone();
                    r.relocatable_code = relocatable;
                    let c = compiler.compile_request(ShaderStage::Compute, &r, 7).expect("compiles");
                    let src = ModuleSource { program: Arc::new(program.clone()), request: encode_request(&r, spec, None), reads: Arc::new(CodeImage::from_words(c.code_reads.clone())) };
                    *out.borrow_mut() = Some((c, src));
                });
                out.into_inner().unwrap()
            };
            let (a, b) = (0x50_0a00_0000u64, 0x77_1234_5600u64);
            let (plain_a, _) = at(a, false);
            let (plain_b, _) = at(b, false);
            assert_ne!(plain_a.spirv, plain_b.spirv, "{name}: the plain module names its address");
            let (reloc_a, src_a) = at(a, true);
            let (reloc_b, _) = at(b, true);
            validate_spirv(&reloc_a.spirv);
            assert_eq!(reloc_a.spirv, reloc_b.spirv, "{name}: one relocatable module");
            assert!(relocatable(&src_a.request) && !proven_relocatable(&src_a.request));
            assert!(proves_relocatable(&compiler, 7, &src_a, &reloc_a), "{name}: the trial proves it");
            assert!(proven_relocatable(&mark_proven(&src_a.request)));
            // A module made at another address is not the trial's.
            assert!(!proves_relocatable(&compiler, 7, &src_a, &plain_a), "{name}: a plain module is not relocatable");
        }
    }

    fn validate_spirv(words: &[u32]) {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../local/tmp/reloc-spv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.spv");
        std::fs::write(&path, words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let out = std::process::Command::new("spirv-val").arg("--target-env").arg("vulkan1.3").arg(&path).output();
        let _ = std::fs::remove_dir_all(&dir);
        if let Ok(o) = out {
            assert!(o.status.success(), "spirv-val: {}", String::from_utf8_lossy(&o.stderr));
        }
    }

    /// The module a list made by another translator holds is translated
    /// again from its guest inputs: the same SPIR-V the compiler makes of
    /// the request the provider built.
    #[test]
    fn modules_translate_again_from_their_inputs() {
        let compiler = crate::shader::SpirvCompiler::new(features(), None);
        let (program, with_request) = captured_compute();
        let direct = std::cell::RefCell::new(None);
        let encoded = std::cell::RefCell::new(None);
        with_request(&|r, spec| {
            *direct.borrow_mut() = Some(compiler.compile_request(ShaderStage::Compute, r, 7).expect("compiles"));
            *encoded.borrow_mut() = Some(encode_request(r, spec, None));
        });
        let direct = direct.into_inner().unwrap();
        let source = Arc::new(ModuleSource { program: Arc::new(program), request: encoded.into_inner().unwrap(), reads: Arc::new(CodeImage::from_words(direct.code_reads.clone())) });

        // A list written by "another translator" (stamp 1) with a module
        // whose SPIR-V that translator made.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../local/tmp/prewarm-retranslate-test-{}", std::process::id()));
        let bin = dir.join("TEST00002.bin");
        {
            let mut list = PrewarmList::load(Some(&bin), 1);
            list.add_module(7, ShaderStage::Compute as u8, vec![0x0723_0203, 0xdead], direct.reflection.clone(), Some(source));
        }
        // This translator (stamp 2): kept for translation, not dropped.
        let mut list = PrewarmList::load(Some(&bin), 2);
        let stale: Vec<(u64, Arc<ModuleSource>)> = list.stale().map(|m| (m.key, m.source.clone().unwrap())).collect();
        assert_eq!(stale.len(), 1, "the module is kept to be translated again");
        let (key, src) = &stale[0];
        let again = retranslate(&compiler, *key, src).expect("translates again");
        assert_eq!(again.spirv, direct.spirv, "same module as the provider's request gives");
        assert_eq!(again.reflection, direct.reflection);
        list.add_module(*key, ShaderStage::Compute as u8, again.spirv.clone(), again.reflection, Some(src.clone()));
        assert_eq!(list.stale().count(), 0);
        drop(list);
        let list = PrewarmList::load(Some(&bin), 2);
        assert!(list.modules[0].fresh);
        assert_eq!(&list.modules[0].spirv[..], &direct.spirv[..]);
        drop(list);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every stage option survives the encoding.
    #[test]
    fn requests_round_trip() {
        let (program, with_request) = captured_compute();
        with_request(&|r, spec| {
            let mut q = r.clone();
            q.pixel_outputs = vec![PixelOutputBinding {
                guest_slot: 2,
                host_location: 1,
                kind: PixelOutputKind::Sint,
                component_mapping: ColorComponentMapping { packed: 0x1b },
                export_target: Some(3),
                packed_unorm_11_11_10: Some(true),
                packed_unorm_4_4_4_4: true,
            }];
            q.pixel_input_cntl = Some(vec![1, 2, 3]);
            q.vertex_inputs = vec![ShaderVertexInput::new(0x40, 2, 3, 7, true, vec![0x80])];
            q.pixel_quad_sample_masks = Some([1, 2, 4, 8]);
            q.pixel_alpha_blend_source = PixelAlphaBlendSource::SourceAlphaZero;
            q.clip_space = ShaderClipSpaceTransform { enabled: true, scale_x: 0.5, scale_y: -2.0, offset_x: 1.0, offset_y: 3.0, half_extent_x: 960.0, half_extent_y: 540.0 };
            q.indirect_draw = Some([4, 0xFF]);
            q.user_clip_planes = ps5shader::resources::ShaderUserClipPlanes { mask: 0x21, planes: std::array::from_fn(|i| 0x3F00_0000 + i as u32) };
            q.required_vertex_output_count = -3;
            let mesh = Some(MeshConfig { input_primitive: 4, primitives_per_group: 64, vertices_per_group: 96, max_vertices: 256, max_primitives: 128, threads: 128, esgs_item_size: 12, draw_dword: 5, lines: true, points: false });
            let words = encode_request(&q, spec, mesh);
            let mut rd = Reader(&words);
            assert_eq!(rd.u(), Some(REQUEST_VERSION));
            assert_eq!(get_mesh(&mut rd), Some(mesh));
            assert_eq!(get_specialization(&mut rd).as_ref(), Some(spec));
            assert_eq!(rd.b(), Some(q.resources.info.buffers_in_guest_memory));
            assert_eq!(get_layout(&mut rd, REQUEST_VERSION).as_ref(), Some(&q.bindings));
            let mut back = ShaderCompileRequest::new(q.plan, q.resources.clone(), q.bindings.clone());
            get_options(&mut rd, &mut back, REQUEST_VERSION).expect("options decode");
            assert!(rd.0.is_empty());
            assert_eq!(format!("{back:?}"), format!("{q:?}"));
        });
        let _ = program;
    }

    /// `rps5e-cli precompile` on a synthetic list made by another
    /// translator: its module is translated again on the workers, its
    /// pipeline built into the pipeline cache, and the list and the cache
    /// are written (the list now with this translator's module).
    #[test]
    #[ignore]
    fn precompile_builds_a_synthetic_list_on_the_device() {
        let dev = crate::device::GpuDevice::create(false).expect("device");
        let compiler = Arc::new(crate::shader::SpirvCompiler::new(crate::shader::Features::of(&dev), None));
        let stamp = compiler.translator_stamp().unwrap();
        let (program, with_request) = captured_compute();
        let made = std::cell::RefCell::new(None);
        with_request(&|r, spec| {
            let c = compiler.compile_request(ShaderStage::Compute, r, 7).expect("compiles");
            let layout = super::super::descriptors::set_bindings(&[(ShaderStage::Compute, &r.bindings)]);
            *made.borrow_mut() = Some((c, encode_request(r, spec, None), layout));
        });
        let (direct, request, layout) = made.into_inner().unwrap();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../local/tmp/precompile-device-test-{}", std::process::id()));
        let bin = dir.join("TEST00003.bin");
        {
            let mut list = PrewarmList::load(Some(&bin), stamp ^ 1);
            let source = Arc::new(ModuleSource { program: Arc::new(program), request, reads: Arc::new(CodeImage::from_words(direct.code_reads.clone())) });
            list.add_module(7, ShaderStage::Compute as u8, vec![0x0723_0203, 0xdead], direct.reflection.clone(), Some(source));
            list.add_compute(crate::pipeline::prewarm::ComputeRecord { cs: 7, layout, subgroup32: false });
        }
        let memory = crate::testutil::test_memory();
        let mut reports = 0;
        let summary = crate::pipeline::precompile::precompile_with(&dev, memory, compiler.clone(), &bin, 4, false, &mut |p| {
            reports += 1;
            assert!(p.done <= p.total);
            true
        })
        .expect("precompiles");
        eprintln!("{summary}");
        assert!(reports >= 1);
        assert_eq!((summary.modules, summary.translated, summary.translate_failures), (1, 1, 0));
        assert_eq!((summary.pipelines, summary.built, summary.pipeline_failures), (1, 1, 0));
        assert!(bin.is_file(), "pipeline cache written");
        let list = PrewarmList::load(Some(&bin), stamp);
        assert_eq!(list.stale().count(), 0);
        assert_eq!(&list.modules[0].spirv[..], &direct.spirv[..]);
        drop(list);
        // A recompile clears the pipeline cache and translates the module
        // of this translator again.
        let summary = crate::pipeline::precompile::precompile_with(&dev, memory, compiler.clone(), &bin, 4, true, &mut |_| true).expect("recompiles");
        assert_eq!((summary.modules, summary.translated, summary.translate_failures), (1, 1, 0));
        assert_eq!((summary.pipelines, summary.built, summary.pipeline_failures), (1, 1, 0));
        assert!(bin.is_file(), "pipeline cache written again");
        let list = PrewarmList::load(Some(&bin), stamp);
        assert_eq!(list.stale().count(), 0);
        assert_eq!(&list.modules[0].spirv[..], &direct.spirv[..]);
        drop(list);
        // A list of an older format builds nothing, says so, is moved aside
        // and leaves the caches of other builds alone.
        let old = dir.join("TEST00004.bin");
        std::fs::write(&old, b"another build's pipeline cache").unwrap();
        std::fs::write(crate::pipeline::prewarm::list_path(&old), b"rps5ePW3\nolder list").unwrap();
        let err = crate::pipeline::precompile::precompile_with(&dev, memory, compiler, &old, 4, false, &mut |_| true).err().expect("an older list is refused");
        assert!(err.contains("older format PW3"), "{err}");
        assert_eq!(std::fs::read(&old).unwrap(), b"another build's pipeline cache");
        assert!(!crate::pipeline::prewarm::list_path(&old).exists());
        assert_eq!(std::fs::read(dir.join("TEST00004.prewarm.pw3.old")).unwrap(), b"rps5ePW3\nolder list");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn code_images_read_what_they_hold() {
        let img = CodeImage::from_words(vec![(0x108, 3), (0x100, 1), (0x104, 2), (0x200, 9), (0x100, 7)]);
        assert_eq!(img.read(0x100), Some(1));
        assert_eq!(img.read(0x104), Some(2));
        assert_eq!(img.read(0x108), Some(3));
        assert_eq!(img.read(0x10c), None);
        assert_eq!(img.read(0x102), None);
        assert_eq!(img.read(0x200), Some(9));
        assert_eq!(img.read(0xfc), None);
        assert_eq!(img.words(), 4);
        let mut w = Writer(Vec::new());
        img.put(&mut w);
        assert_eq!(CodeImage::get(&mut Reader(&w.0)), Some(img));
    }
}
