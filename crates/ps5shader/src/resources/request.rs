// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Everything an emitter needs to compile one permutation of a program.

use super::binding::BindingLayout;
use super::materialize::SpecializedResourceInfo;
use super::memory::MemoryAccessTable;
use super::plan::{ShaderResourcePlan, WRITTEN_RANGE_DWORD_COUNT};
use super::ShaderStage;
use crate::ir_types::{ComputeSystemRegisters, PixelOutputBinding, Program};
use std::collections::{HashMap, HashSet};

/// One fixed-function vertex attribute a fetch instruction reads instead of
/// its buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderVertexInput {
    pub pc: u32,
    pub location: u32,
    pub fetch_component_count: u32,
    pub component_count: u32,
    pub number_format: u32,
    pub destination_select: u32,
    pub per_instance: bool,
    /// Later loads of the same attribute.
    pub alias_pcs: Vec<u32>,
}

impl ShaderVertexInput {
    /// An input whose destination select is the identity for its components.
    pub fn new(pc: u32, location: u32, component_count: u32, number_format: u32, per_instance: bool, alias_pcs: Vec<u32>) -> Self {
        let destination_select = match component_count {
            1 => 4,
            2 => 4 | (5 << 3),
            3 => 4 | (5 << 3) | (6 << 6),
            4 => 4 | (5 << 3) | (6 << 6) | (7 << 9),
            _ => 0,
        };
        ShaderVertexInput {
            pc,
            location,
            fetch_component_count: component_count,
            component_count,
            number_format,
            destination_select,
            per_instance,
            alias_pcs,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShaderClipSpaceTransform {
    pub enabled: bool,
    pub scale_x: f32,
    pub scale_y: f32,
    pub offset_x: f32,
    pub offset_y: f32,
    pub half_extent_x: f32,
    pub half_extent_y: f32,
}

/// User clip planes (PA_CL_CLIP_CNTL.UCP_ENA and PA_CL_UCP_n_X..W): the
/// clipper keeps the half space where dot(plane, position) >= 0 for each
/// enabled plane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ShaderUserClipPlanes {
    /// UCP_ENA: bit n enables plane n (0..5).
    pub mask: u32,
    /// The plane coefficients as raw f32 bits, four per plane.
    pub planes: [u32; 24],
}

impl ShaderUserClipPlanes {
    /// The enabled planes, in plane order.
    pub fn count(&self) -> u32 {
        (self.mask & 0x3F).count_ones()
    }
}

/// How the pixel shader supplies the logical source alpha as the second
/// blend source of target 0 when the target's export mapping moves alpha
/// to another channel (the host blend then reads SRC1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PixelAlphaBlendSource {
    #[default]
    None,
    /// Source alpha in every channel (SRC_ALPHA factors become SRC1_ALPHA).
    SourceAlpha,
    /// Source alpha for color, one / zero in the physical alpha channel
    /// (the color factor becomes SRC1_COLOR).
    SourceAlphaOne,
    SourceAlphaZero,
}

/// What one component of a misc position export (POS1..POS4) carries, by
/// PA_CL_VS_OUT_CNTL.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PositionExportComponent {
    /// Index into the clip / cull distance arrays.
    pub clip_distance: Option<u32>,
    pub cull_distance: Option<u32>,
    pub point_size: bool,
    pub layer: bool,
    pub viewport: bool,
}

/// Decode component `component` of position export `pos_index` (1..4):
/// the enabled vectors (misc, clip/cull 0..3, clip/cull 4..7, VS_OUT_CCDIST0/1
/// and VS_OUT_MISC_VEC_ENA bits 21..23) take the exports in order.
pub fn decode_position_export_component(control: u32, pos_index: u32, component: u32) -> PositionExportComponent {
    let mut r = PositionExportComponent::default();
    if pos_index == 0 || component >= 4 {
        return r;
    }
    let mut slot = pos_index - 1;
    let mut vector = 3;
    for i in 0..3 {
        if control & (1 << (21 + i)) != 0 {
            if slot == 0 {
                vector = i;
                break;
            }
            slot -= 1;
        }
    }
    match vector {
        3 => {}
        0 => {
            r.point_size = component == 0 && control & (1 << 16) != 0;
            r.layer = component == 2 && control & (1 << 18) != 0;
            r.viewport = component == 2 && control & (1 << 19) != 0;
        }
        _ => {
            let scalar = (vector - 1) * 4 + component;
            let lower = (1u32 << scalar) - 1;
            let (clip, cull) = (control & 0xFF, (control >> 8) & 0xFF);
            if clip & (1 << scalar) != 0 {
                r.clip_distance = Some((clip & lower).count_ones());
            }
            if cull & (1 << scalar) != 0 {
                r.cull_distance = Some((cull & lower).count_ones());
            }
        }
    }
    r
}

/// One bounded runtime V# table as the emitter sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferCandidateTableUse {
    pub first_candidate: u32,
    pub candidate_count: u32,
    pub mapping_offset: u32,
    pub search_iterations: u32,
}

pub const UNBOUNDED_THREAD_COUNT: u32 = u32::MAX;

/// The decoded program, its resource plan applied to one specialization,
/// the binding layout, and the stage options of one permutation.
#[derive(Clone, Debug)]
pub struct ShaderCompileRequest<'p> {
    pub plan: &'p ShaderResourcePlan,
    pub resources: SpecializedResourceInfo,
    pub bindings: BindingLayout,
    pub uses_flattened_table: bool,
    pub uses_global_data_share: bool,
    pub reads_shader_base: bool,
    /// Host-flattened scalar reads: memory index → table slot.
    pub flattened_slot_by_memory_index: HashMap<usize, u32>,
    /// The materialised table always holds at least this many words.
    pub flattened_table_reserved_words: u32,
    /// Scalar reads whose dword selects an indirect image later.
    pub indirect_key_memory_indices: HashSet<usize>,
    pub indirect_offset_key_memory_indices: HashSet<usize>,
    /// Indirect image access → memory index of its key read.
    pub indirect_root_by_memory_index: HashMap<usize, usize>,
    pub buffer_candidate_table_by_memory_index: HashMap<usize, BufferCandidateTableUse>,
    /// Written device-address access → flattened slot of its range.
    pub written_range_slot_by_memory_index: HashMap<usize, u32>,

    pub trace_device_address_faults: bool,
    pub wave_size: u32,
    pub enable_exec_guard_elision: bool,
    pub scratch_dwords: u32,
    pub enable_graphics_subgroup_operations: bool,
    pub supports_shared_int64_atomics: bool,
    pub compute_system_registers: Option<ComputeSystemRegisters>,
    /// FLOAT_MODE / DX10_CLAMP / IEEE_MODE of the program's RSRC1.
    pub float_mode: crate::translate::FloatMode,
    pub pixel_outputs: Vec<PixelOutputBinding>,
    /// SPI_SHADER_COL_FORMAT: export format of each MRT (4 bits each).
    pub color_export_format: u32,
    pub pixel_input_enable: u32,
    pub pixel_custom_interpolation_mask: u32,
    pub supports_per_vertex_pixel_inputs: bool,
    pub pixel_input_address: u32,
    pub pixel_input_cntl: Option<Vec<u32>>,
    /// SPI_BARYC_CNTL.FRONT_FACE_ALL_BITS while the program loads the
    /// front-face input: it reads 0xFFFFFFFF/0 instead of +1.0f/-1.0f
    /// (measured on hardware).
    pub front_face_all_bits: bool,
    /// Pixel: the class of the rasterized primitives (0 point, 1 line, 2
    /// triangle, 3 rectangle), bits 0..1 of the ancillary VGPR.
    pub pixel_raster_primitive: u32,
    /// Pixel: a primitive-ordered pixel program
    /// (DB_SHADER_CONTROL.PRIMITIVE_ORDERED_PIXEL_SHADER): the invocations
    /// of a pixel run their memory accesses in primitive order (the host's
    /// pixel-ordered fragment interlock around the whole program).
    pub pixel_ordered: bool,
    /// Pixel: the program runs once per sample (PA_SC_MODE_CNTL_1.
    /// PS_ITER_SAMPLE on a multisampled target). Otherwise the ancillary
    /// sample number reads 0, the iterated sample of a pixel the hardware
    /// does not iterate.
    pub pixel_sample_rate: bool,
    pub required_vertex_output_count: i32,
    pub vertex_inputs: Vec<ShaderVertexInput>,
    pub position_export_control: u32,
    pub supports_clip_distance: bool,
    /// Pixel: MRT1 is the second blend source of target 0 (dual-source
    /// blending), or target 0 also exports its logical alpha there.
    pub pixel_dual_source_blend: bool,
    /// Pixel: the depth export (mrtz x) reaches the depth buffer
    /// (DB_SHADER_CONTROL.Z_EXPORT_ENABLE with a SPI_SHADER_Z_FORMAT).
    pub pixel_depth_export: bool,
    /// Pixel: alpha-to-coverage (DB_ALPHA_TO_MASK): the MRT0 alpha decides
    /// the sample mask.
    pub pixel_alpha_to_coverage: bool,
    /// Pixel: the stencil test value export (mrtz y, an integer) reaches
    /// the stencil test (DB_SHADER_CONTROL.STENCIL_TEST_VAL_EXPORT_ENABLE
    /// with SPI_SHADER_Z_FORMAT 32_GR or 32_ABGR).
    pub pixel_stencil_export: bool,
    /// Pixel: MRTZ.y bits 8..15 are the stencil operation value
    /// (DB_SHADER_CONTROL.STENCIL_OP_VAL_EXPORT_ENABLE), exported as the
    /// host stencil reference REPLACE_OP writes.
    pub pixel_stencil_op_export: bool,
    /// Pixel: the depth and stencil tests run before the program
    /// (DB_SHADER_CONTROL.DEPTH_BEFORE_SHADER, or an early Z order without
    /// kill or exports): occluded pixels do not run it.
    pub pixel_early_fragment_tests: bool,
    /// Pixel: the samples each pixel of a 2x2 quad keeps (PA_SC_AA_MASK
    /// X0Y0, X1Y0, X0Y1, X1Y1) when they differ: the sample mask output
    /// starts from the mask of the pixel's quad position.
    pub pixel_quad_sample_masks: Option<[u32; 4]>,
    pub pixel_alpha_blend_source: PixelAlphaBlendSource,
    pub clip_space: ShaderClipSpaceTransform,
    /// Vertex: the user clip planes, written as the first clip distances.
    pub user_clip_planes: ShaderUserClipPlanes,
    /// LDS the dispatch allocates (COMPUTE_PGM_RSRC2.LDS_SIZE), 0 if unknown.
    pub local_data_share_dwords: u32,
    pub local_size: [u32; 3],
    pub thread_count: [u32; 3],
    /// Vertex stage of an indirect draw executed by the host from the
    /// guest's argument records: the user SGPRs (indices into the stage's
    /// user data, 0xFF for none) that receive the record's base vertex
    /// and first instance come from the host draw parameters, and the
    /// vertex and instance IDs exclude them (as the guest's do).
    pub indirect_draw: Option<[u8; 2]>,
    /// The program's code address (S_GETPC_B64) is read from the shader
    /// data (the shader base) instead of being a constant of the module:
    /// the module serves the program wherever the game loads it.
    pub relocatable_code: bool,
}

impl<'p> ShaderCompileRequest<'p> {
    pub fn new(plan: &'p ShaderResourcePlan, resources: SpecializedResourceInfo, bindings: BindingLayout) -> Self {
        let program = plan.program();
        let mut written = HashMap::new();
        for range in &plan.device_address_ranges {
            if let (true, Some(&slot)) = (range.written, plan.written_range_slot_by_handle.get(&range.handle)) {
                for &m in &range.memory_indices {
                    written.insert(m, slot);
                }
            }
        }
        let mut candidates = HashMap::new();
        for (table, info) in plan.buffer_candidate_tables.iter().zip(&resources.info.buffer_candidate_tables) {
            let use_ = BufferCandidateTableUse {
                first_candidate: info.first_candidate,
                candidate_count: info.candidate_count,
                mapping_offset: info.mapping_offset,
                search_iterations: info.search_iterations,
            };
            for &m in &table.memory_indices {
                candidates.insert(m, use_);
            }
        }
        let key_memory = |key| plan.graph.node(key).memory_index();
        ShaderCompileRequest {
            uses_flattened_table: Self::requires_flattened_table(plan, &resources),
            uses_global_data_share: BindingLayout::uses_global_data_share(program),
            reads_shader_base: BindingLayout::reads_shader_base(program),
            flattened_slot_by_memory_index: plan.flattened_slot_by_memory_index.clone(),
            flattened_table_reserved_words: plan.flattened_table_reserved_count() as u32,
            indirect_key_memory_indices: plan.indirect_images.iter().map(|a| key_memory(a.key)).collect(),
            indirect_offset_key_memory_indices: plan.indirect_images.iter().filter(|a| a.key_is_address_offset).map(|a| key_memory(a.key)).collect(),
            indirect_root_by_memory_index: plan.indirect_images.iter().map(|a| (a.memory_index, key_memory(a.key))).collect(),
            buffer_candidate_table_by_memory_index: candidates,
            written_range_slot_by_memory_index: written,
            plan,
            resources,
            bindings,
            trace_device_address_faults: false,
            wave_size: 32,
            enable_exec_guard_elision: true,
            scratch_dwords: 0,
            enable_graphics_subgroup_operations: true,
            supports_shared_int64_atomics: false,
            compute_system_registers: None,
            float_mode: crate::translate::FloatMode::default(),
            pixel_outputs: Vec::new(),
            color_export_format: 0,
            pixel_input_enable: 0,
            pixel_custom_interpolation_mask: 0,
            supports_per_vertex_pixel_inputs: true,
            pixel_input_address: 0,
            pixel_input_cntl: None,
            front_face_all_bits: false,
            pixel_raster_primitive: 2,
            pixel_ordered: false,
            pixel_sample_rate: false,
            required_vertex_output_count: 0,
            vertex_inputs: Vec::new(),
            position_export_control: 0,
            supports_clip_distance: true,
            pixel_dual_source_blend: false,
            pixel_depth_export: true,
            pixel_alpha_to_coverage: false,
            pixel_stencil_export: false,
            pixel_stencil_op_export: false,
            pixel_early_fragment_tests: false,
            pixel_quad_sample_masks: None,
            pixel_alpha_blend_source: PixelAlphaBlendSource::None,
            clip_space: ShaderClipSpaceTransform::default(),
            user_clip_planes: ShaderUserClipPlanes::default(),
            local_data_share_dwords: 0,
            local_size: [1, 1, 1],
            thread_count: [UNBOUNDED_THREAD_COUNT; 3],
            indirect_draw: None,
            relocatable_code: false,
        }
    }

    /// The flattened table is bound when host reads, written ranges,
    /// indirect mappings or candidate mappings fill it.
    pub fn requires_flattened_table(plan: &ShaderResourcePlan, resources: &SpecializedResourceInfo) -> bool {
        !plan.table_reads.is_empty()
            || plan.written_range_count() != 0
            || !plan.buffer_candidate_tables.is_empty()
            || resources.info.images.iter().any(|i| i.indirect_search_iterations != 0 || i.bindless)
    }

    pub fn program(&self) -> &Program {
        self.plan.program()
    }

    pub fn stage(&self) -> ShaderStage {
        self.plan.stage
    }

    pub fn hash(&self) -> u64 {
        self.plan.hash
    }

    pub fn memory(&self) -> &MemoryAccessTable {
        self.plan.memory()
    }

    pub fn user_data_base(&self) -> u32 {
        self.plan.user_data_base()
    }

    pub fn user_data_count(&self) -> u32 {
        self.plan.user_data_count()
    }

    pub const WRITTEN_RANGE_DWORD_COUNT: usize = WRITTEN_RANGE_DWORD_COUNT;
}
