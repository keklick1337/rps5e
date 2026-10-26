// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The dense resource tables of a program and the descriptor sources they
//! are materialised from.

use super::format::ImageNumericClass;
use super::memory::{ImageDimension, ImageResourceClass};
use super::selector::IndirectSelectorValues;
use super::value::ValueId;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    #[default]
    Unknown,
    Vertex,
    Pixel,
    Compute,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ImageMipMode {
    #[default]
    None,
    /// One binding element per mip of the view, the shader selects one: a
    /// storage image accessed with an explicit mip, or an image gathered
    /// with an explicit LOD (gathers read level 0 of their view).
    Dynamic,
}

/// The identity V# DST_SEL (X, Y, Z, W in place).
pub const IDENTITY_DESTINATION_SELECT: u32 = 4 | (5 << 3) | (6 << 6) | (7 << 9);
/// The identity T# DST_SEL as the image path packs it.
pub const IDENTITY_IMAGE_SWIZZLE: u32 = 0xFAC;
pub const INVALID_FORMAT: u32 = 0;
/// "No index" in every `u32` index field.
pub const NO_INDEX: u32 = u32::MAX;

/// One buffer resource and how the program uses it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferResource {
    /// Index into [`ShaderResourcePlan::descriptor_sources`](super::ShaderResourcePlan).
    pub source: u32,
    pub first_use_pc: u32,
    /// Highest byte (immediate offset + access size) any access reaches.
    pub max_byte_extent: u32,
    /// Filled by specialization: stride | swizzle<<14 | index stride<<16 |
    /// add-tid<<20 | OOB_SELECT<<24.
    pub packed_stride: u32,
    pub descriptor_format: u32,
    pub descriptor_swizzle: u32,
    /// The image whose first four T# dwords are this V#.
    pub image_alias: u32,
    pub read: bool,
    pub written: bool,
    pub atomic: bool,
    pub formatted: bool,
    /// Read by a scalar (S_BUFFER_LOAD) access.
    pub scalar: bool,
    /// Every access is a scalar load at a constant byte offset (no offset
    /// register): nothing past `max_byte_extent` is ever read.
    pub constant_offsets: bool,
}

impl BufferResource {
    pub fn new(source: u32, first_use_pc: u32) -> Self {
        BufferResource {
            source,
            first_use_pc,
            max_byte_extent: 0,
            packed_stride: 0,
            descriptor_format: INVALID_FORMAT,
            descriptor_swizzle: IDENTITY_DESTINATION_SELECT,
            image_alias: NO_INDEX,
            read: false,
            written: false,
            atomic: false,
            formatted: false,
            scalar: false,
            constant_offsets: false,
        }
    }
}

/// One image resource: a descriptor source with its view class, dimension
/// and use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageResource {
    pub source: u32,
    pub first_use_pc: u32,
    pub resource_class: ImageResourceClass,
    /// Filled by specialization from the T# format.
    pub numeric_class: ImageNumericClass,
    pub dimension: ImageDimension,
    pub mip_mode: ImageMipMode,
    pub mip_count: u32,
    pub conversion_format: u32,
    pub shader_swizzle: u32,
    pub read: bool,
    pub written: bool,
    pub atomic: bool,
    /// Updated by 64-bit atomics: bound as an R64 unsigned view of a
    /// 32_32 unsigned image.
    pub atomic64: bool,
    pub depth_compare: bool,
    /// Depth-compare sampling of an image whose format has no host depth
    /// equivalent, evaluated in the shader ([`emulated_compare`] bits); 0
    /// when not emulated.
    pub emulated_compare: u32,
    pub cube: bool,
    /// Read without a sampler (image_load): such a cube is addressed by
    /// face, not sampled through a direction.
    pub fetched: bool,
    pub r128: bool,
    /// The root of an indirect candidate run (an image selected at run time).
    pub indirect_root: u32,
    pub indirect_mapping_offset: u32,
    pub indirect_search_iterations: u32,
    pub indirect_resources: Vec<u32>,
    /// A bindless table too large to bind per draw: the T# is selected
    /// through the host's table map (see [`super::binding::bindless`]); the
    /// map location sits in the flattened table at
    /// `indirect_mapping_offset`.
    pub bindless: bool,
    /// Accessed as raw texel bits (MIMG PCK forms) or several texels at
    /// once (BY2 / BY4): the specialization carries the T# format.
    pub packed: bool,
    /// The T# format of such an image (filled by specialization).
    pub packed_format: u32,
    /// Stores to this storage image quantize their float data in the
    /// shader (filled by specialization): 0 for none, else the component
    /// width in bits, | 0x100 for SNORM ([`store_quantize_code`]).
    pub store_quantize: u32,
}

/// The [`ImageResource::store_quantize`] code of a T# format: a UNORM or
/// SNORM format whose components all have one width of 2 to 16 bits (not
/// packed fields) is quantized to the level nearest to the exact product,
/// ties to even, as hardware rounds; 0 otherwise.
pub fn store_quantize_code(format: u32) -> u32 {
    let Some((_, nfmt)) = crate::resources::format::unified_format::decode(format) else {
        return 0;
    };
    if nfmt > 1 {
        return 0;
    }
    let Some(f) = crate::interp::buffer::FormatInfo::new(format) else {
        return 0;
    };
    let bits = f.bits[0];
    if f.packed || !(2..=16).contains(&bits) || f.bits[..f.count as usize].iter().any(|&b| b != bits) {
        return 0;
    }
    bits | if nfmt == 1 { 0x100 } else { 0 }
}

impl ImageResource {
    pub fn new(source: u32, first_use_pc: u32) -> Self {
        ImageResource {
            source,
            first_use_pc,
            resource_class: ImageResourceClass::None,
            numeric_class: ImageNumericClass::Unsupported,
            dimension: ImageDimension::Unknown,
            mip_mode: ImageMipMode::None,
            mip_count: 1,
            conversion_format: INVALID_FORMAT,
            shader_swizzle: IDENTITY_IMAGE_SWIZZLE,
            read: false,
            written: false,
            atomic: false,
            atomic64: false,
            depth_compare: false,
            emulated_compare: 0,
            cube: false,
            fetched: false,
            r128: false,
            indirect_root: NO_INDEX,
            indirect_mapping_offset: 0,
            indirect_search_iterations: 0,
            indirect_resources: Vec::new(),
            bindless: false,
            packed: false,
            packed_format: INVALID_FORMAT,
            store_quantize: 0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SamplerResource {
    pub source: u32,
    pub first_use_pc: u32,
    pub force_point_filtering: bool,
    pub depth_compare: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SampledImagePair {
    pub image: u32,
    pub sampler: u32,
    pub first_use_pc: u32,
    /// The point-filtering copy of the sampler an image selected through
    /// the host table map samples its integer T#s with (filled by
    /// specialization).
    pub point_sampler: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StageInputKind {
    VertexIndex,
    InstanceIndex,
    FragCoord,
    FrontFacing,
    BaryCoordSmooth,
    BaryCoordNoPerspective,
    WorkgroupId,
    LocalInvocationId,
    LocalInvocationIndex,
    GlobalInvocationId,
    Parameter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StageOutputKind {
    Position,
    Parameter,
    Mrt,
    Depth,
    SampleMask,
    PointSize,
    ClipDistance,
    CullDistance,
    Layer,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageInput {
    pub kind: StageInputKind,
    pub location: u32,
    pub component_count: u32,
    pub debug_name: String,
    pub per_vertex: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageOutput {
    pub kind: StageOutputKind,
    pub index: u32,
    pub location: u32,
    pub debug_name: String,
}

/// One bounded runtime V# table lowered to a contiguous run of native buffer
/// candidates, selected through a key mapping in the flattened table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferCandidateTableInfo {
    pub first_candidate: u32,
    pub candidate_count: u32,
    pub mapping_offset: u32,
    pub search_iterations: u32,
}

pub const NO_SCALAR_REGISTER: i32 = -1;

/// The dense resource tables of a program plus the facts the pipeline layout
/// needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderResourceInfo {
    pub buffers: Vec<BufferResource>,
    pub images: Vec<ImageResource>,
    pub samplers: Vec<SamplerResource>,
    pub sampled_pairs: Vec<SampledImagePair>,
    pub buffer_candidate_tables: Vec<BufferCandidateTableInfo>,
    pub inputs: Vec<StageInput>,
    pub outputs: Vec<StageOutput>,
    pub vertex_fetch_components: [u8; 32],
    pub vertex_offset_scalar_register: i32,
    pub instance_offset_scalar_register: i32,
    pub has_bitwise_exclusive_or: bool,
    /// Some access goes through the device-address page table.
    pub uses_device_addresses: bool,
    /// Every buffer is read through its V#'s device address (guest memory
    /// resident for the GPU): no buffer is bound. Set by hosts for stages
    /// that write no memory through buffers or addresses.
    pub buffers_in_guest_memory: bool,
}

impl Default for ShaderResourceInfo {
    fn default() -> Self {
        ShaderResourceInfo {
            buffers: Vec::new(),
            images: Vec::new(),
            samplers: Vec::new(),
            sampled_pairs: Vec::new(),
            buffer_candidate_tables: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            vertex_fetch_components: [0; 32],
            vertex_offset_scalar_register: NO_SCALAR_REGISTER,
            instance_offset_scalar_register: NO_SCALAR_REGISTER,
            has_bitwise_exclusive_or: false,
            uses_device_addresses: false,
            buffers_in_guest_memory: false,
        }
    }
}

impl ShaderResourceInfo {
    pub const MAX_BUFFERS: usize = 32;
    pub const MAX_IMAGES: usize = 64;
    pub const MAX_SAMPLERS: usize = 32;
}

/// One candidate of an image whose descriptor is selected by a known finite
/// set of keys (an S_FF1 selector).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectImageCandidate {
    /// The key (dynamic byte offset of the descriptor load).
    pub offset: u32,
    /// Descriptor source of this candidate.
    pub source: u32,
}

/// A wave-uniform selector: each descriptor key comes from a set bit of one
/// scalar mask, through a compact global index table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaveIndexedImageSelector {
    pub mask_offset: u32,
    pub index_table_offset: u32,
    pub index_stride: u32,
}

/// An image whose descriptor is selected at run time: a material-table key
/// picks one of several heap descriptors (or one of a finite candidate set).
#[derive(Clone, Debug)]
pub struct IndirectImageSelector {
    pub material_source: u32,
    pub heap_source: u32,
    pub selector_stride: u32,
    pub selector_offset: u32,
    pub key_argument: u32,
    pub selector_values: Option<Arc<IndirectSelectorValues>>,
    pub direct_candidates: Option<Arc<Vec<DirectImageCandidate>>>,
    pub dense: bool,
    pub table_offset: u32,
    pub dynamic_offset_base: u32,
    pub key_bound: u32,
    pub wave_indexed: Option<WaveIndexedImageSelector>,
    /// The key read's immediate offset (added after the 32-bit selector
    /// offset, without wrapping).
    pub material_immediate: u32,
    /// A bindless table: the T# is read from the heap V# at
    /// `entry_offset + key * 32` with a wave-uniform key.
    pub table: Option<TableImageSelector>,
}

/// A bindless image table: eight contiguous dwords of the heap V# at
/// `entry_offset + key * 32`. When the key is itself read from a material
/// V# at `selector_stride * i + selector_offset`, the reachable keys are
/// enumerated from the material records; otherwise the whole table is bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableImageSelector {
    pub entry_offset: u32,
    pub has_material: bool,
}

impl IndirectImageSelector {
    pub fn new(material_source: u32, heap_source: u32, selector_stride: u32, selector_offset: u32) -> Self {
        IndirectImageSelector {
            material_source,
            heap_source,
            selector_stride,
            selector_offset,
            key_argument: 0,
            selector_values: None,
            direct_candidates: None,
            dense: false,
            table_offset: 0,
            dynamic_offset_base: 0,
            key_bound: 0,
            wave_indexed: None,
            material_immediate: 0,
            table: None,
        }
    }
}

impl PartialEq for IndirectImageSelector {
    /// Structural on plain fields; the shared analyses compare by identity.
    fn eq(&self, other: &Self) -> bool {
        let same_values = match (&self.selector_values, &other.selector_values) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        let same_candidates = match (&self.direct_candidates, &other.direct_candidates) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        same_values
            && same_candidates
            && self.material_source == other.material_source
            && self.heap_source == other.heap_source
            && self.selector_stride == other.selector_stride
            && self.selector_offset == other.selector_offset
            && self.key_argument == other.key_argument
            && self.dense == other.dense
            && self.table_offset == other.table_offset
            && self.dynamic_offset_base == other.dynamic_offset_base
            && self.key_bound == other.key_bound
            && self.wave_indexed == other.wave_indexed
            && self.material_immediate == other.material_immediate
            && self.table == other.table
    }
}

/// The graph values one descriptor is assembled from (2, 4 or 8 dwords).
#[derive(Clone, Debug, PartialEq)]
pub struct DescriptorSource {
    pub dwords: Vec<ValueId>,
    pub indirect_image: Option<IndirectImageSelector>,
}

impl DescriptorSource {
    pub fn dword_count(&self) -> u32 {
        self.dwords.len() as u32
    }
}

/// One immediate-offset scalar read the host evaluates into the flattened
/// resource table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceTableRead {
    pub value: ValueId,
    pub flat_offset: u32,
}

/// Bits of [`ImageResource::emulated_compare`].
pub mod emulated_compare {
    pub const ENABLED: u32 = 1;
    /// Guest compare function (SQ_IMG_SAMP DEPTH_COMPARE_FUNC, 0..7).
    pub const FUNCTION_SHIFT: u32 = 1;
    pub const LINEAR: u32 = 1 << 4;
    pub const CLAMP_X_SHIFT: u32 = 5;
    pub const CLAMP_Y_SHIFT: u32 = 7;
    pub const REFERENCE_SHIFT: u32 = 9;
    pub const SINGLE_LEVEL: u32 = 1 << 11;
    /// Clamp-to-border reads opaque white (red = 1), else transparent black.
    pub const BORDER_WHITE: u32 = 1 << 12;
    pub const ADDRESS_WRAP: u32 = 0;
    pub const ADDRESS_EDGE: u32 = 1;
    pub const ADDRESS_BORDER: u32 = 2;
    pub const REFERENCE_FLOAT: u32 = 0;
    pub const REFERENCE_UNORM: u32 = 1;
    pub const REFERENCE_SNORM: u32 = 2;
    /// 8-bit sRGB colour: the comparison returns 0 (measured on gfx1035).
    pub const REFERENCE_ZERO: u32 = 3;

    pub fn function(state: u32) -> u32 {
        (state >> FUNCTION_SHIFT) & 7
    }
    pub fn address_x(state: u32) -> u32 {
        (state >> CLAMP_X_SHIFT) & 3
    }
    pub fn address_y(state: u32) -> u32 {
        (state >> CLAMP_Y_SHIFT) & 3
    }
    pub fn reference(state: u32) -> u32 {
        (state >> REFERENCE_SHIFT) & 3
    }
}
