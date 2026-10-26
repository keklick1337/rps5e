// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Descriptor-set layout and push data of one compiled program.

use super::format::ImageNumericClass;
use super::info::*;
use super::memory::{ImageDimension, ImageResourceClass};
use super::ResourcePlanError;
use crate::ir::ControlFlowGraph;
use crate::ir_types::*;
use crate::opcode::Opcode;

/// The fixed binding numbers of one stage. Image bindings occupy
/// `FIRST_IMAGE_BINDING .. FIRST_IMAGE_BINDING + IMAGE_BINDING_COUNT`
/// ([`ImageBinding`]); a pixel stage adds [`DescriptorBindingKind::COUNT`] to
/// every number ([`native_binding_index`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DescriptorBindingKind {
    Buffers,
    /// One image binding array (`0..IMAGE_BINDING_COUNT`).
    Image(u32),
    Samplers,
    GlobalDataShare,
    DeviceAddressPageTable,
    FaultBuffer,
    FlattenedResourceTable,
    ShaderData,
}

pub const FIRST_IMAGE_BINDING: u32 = 1;
pub const FIRST_STORAGE_IMAGE_BINDING: u32 = 22;
pub const IMAGE_BINDING_COUNT: u32 = 50;

impl DescriptorBindingKind {
    /// Bindings per stage.
    pub const COUNT: u32 = FIRST_IMAGE_BINDING + IMAGE_BINDING_COUNT + 6;

    /// The binding number within a stage.
    pub fn number(self) -> u32 {
        let samplers = FIRST_IMAGE_BINDING + IMAGE_BINDING_COUNT;
        match self {
            DescriptorBindingKind::Buffers => 0,
            DescriptorBindingKind::Image(i) => FIRST_IMAGE_BINDING + i,
            DescriptorBindingKind::Samplers => samplers,
            DescriptorBindingKind::GlobalDataShare => samplers + 1,
            DescriptorBindingKind::DeviceAddressPageTable => samplers + 2,
            DescriptorBindingKind::FaultBuffer => samplers + 3,
            DescriptorBindingKind::FlattenedResourceTable => samplers + 4,
            DescriptorBindingKind::ShaderData => samplers + 5,
        }
    }
}

/// The binding number of `kind` in `stage`'s range.
pub fn native_binding_index(stage: ShaderStage, kind: DescriptorBindingKind) -> u32 {
    kind.number() + if stage == ShaderStage::Pixel { DescriptorBindingKind::COUNT } else { 0 }
}

/// Thirty-two dwords of push data shared by the stages of one pipeline.
pub mod push_data {
    pub const DWORD_COUNT: u32 = 32;
    pub const BYTE_SIZE: u32 = DWORD_COUNT * 4;
    pub const NO_START: u32 = u32::MAX;

    pub fn can_fit(start: u32, size: u32) -> bool {
        size != 0 && start <= DWORD_COUNT && size <= DWORD_COUNT - start
    }

    pub fn start_for(cursor: u32, size: u32) -> u32 {
        if can_fit(cursor, size) {
            cursor
        } else {
            NO_START
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DescriptorBinding {
    pub kind: DescriptorBindingKind,
    /// Dense-table indices, one per array element.
    pub resources: Vec<u32>,
}

/// Images selected through a host table map (bindless tables too large to
/// bind per draw): a descriptor set shared by every program, kept by the
/// host. The program reads the slot of a key from the map and samples the
/// image view in that slot.
pub mod bindless {
    /// The descriptor set (after the program's own set 0).
    pub const SET: u32 = 1;
    /// A storage buffer of u32 words holding every table map.
    pub const MAP_BINDING: u32 = 0;
    /// A sampled-image array: one view per slot, of any view type.
    pub const IMAGE_BINDING: u32 = 1;
    /// Flattened-table words of each bindless image: the map's first word,
    /// its key count and the slot of the null view (for keys past the map).
    pub const TABLE_WORDS: u32 = 3;
    /// A map word: the slot in the low bits and the view's class above.
    pub const SLOT_MASK: u32 = (1 << 24) - 1;
    pub const MODE_SHIFT: u32 = 24;
    /// The view classes a slot holds (the program declares one image type
    /// per class on the array): float, unsigned and signed integer.
    pub const MODE_FLOAT: u32 = 0;
    pub const MODE_UINT: u32 = 1;
    pub const MODE_SINT: u32 = 2;
    pub const MODES: u32 = 3;
}

/// Which image binding array an image occupies (by class, numeric class,
/// dimension and cube/compare flags).
pub mod image_binding {
    use super::*;

    const SAMPLED_FLOAT: u32 = 1;
    const SAMPLED_UINT: u32 = 8;
    const SAMPLED_SINT: u32 = 15;
    const STORAGE_FLOAT: u32 = 22;
    const STORAGE_UINT: u32 = 27;
    const ATOMIC_UINT: u32 = 32;
    const SAMPLED_CUBE_FLOAT: u32 = 37;
    const SAMPLED_CUBE_UINT: u32 = 38;
    const SAMPLED_CUBE_SINT: u32 = 39;
    const STORAGE_CUBE_FLOAT: u32 = 40;
    const STORAGE_CUBE_UINT: u32 = 41;
    const ATOMIC_CUBE_UINT: u32 = 42;
    // Depth-reference sampling has its own arrays: validation checks every
    // element of an array sampled with Dref.
    const SAMPLED_COMPARE_2D: u32 = 43;
    const SAMPLED_COMPARE_2D_ARRAY: u32 = 44;
    const SAMPLED_COMPARE_CUBE: u32 = 45;
    /// 64-bit integer atomics (R64 unsigned views), one per storage
    /// dimension.
    const ATOMIC64_UINT: u32 = 46;
    const SAMPLERS: u32 = FIRST_IMAGE_BINDING + IMAGE_BINDING_COUNT;

    fn kind(number: u32) -> Option<DescriptorBindingKind> {
        Some(DescriptorBindingKind::Image(number - FIRST_IMAGE_BINDING))
    }

    /// The binding of `image`, `None` when no binding class fits.
    pub fn for_image(image: &ImageResource) -> Option<DescriptorBindingKind> {
        use ImageDimension as D;
        use ImageNumericClass as N;
        // A sampled cube is bound as a cube (sampled through a direction)
        // unless the program also reads it by face (image_load).
        let cube = image.cube && !(image.resource_class == ImageResourceClass::Sampled && image.fetched);
        if image.depth_compare && image.resource_class == ImageResourceClass::Sampled && !image.atomic && image.numeric_class == N::Float {
            if cube {
                return matches!(image.dimension, D::Dim2D | D::Dim2DArray).then(|| kind(SAMPLED_COMPARE_CUBE)).flatten();
            }
            if image.dimension == D::Dim2D {
                return kind(SAMPLED_COMPARE_2D);
            }
            if image.dimension == D::Dim2DArray {
                return kind(SAMPLED_COMPARE_2D_ARRAY);
            }
        }
        if cube {
            if !matches!(image.dimension, D::Dim2D | D::Dim2DArray) {
                return None;
            }
            return match image.resource_class {
                ImageResourceClass::Sampled if !image.atomic => match image.numeric_class {
                    N::Float => kind(SAMPLED_CUBE_FLOAT),
                    N::Uint => kind(SAMPLED_CUBE_UINT),
                    N::Sint => kind(SAMPLED_CUBE_SINT),
                    _ => None,
                },
                ImageResourceClass::Storage if image.atomic64 => None,
                ImageResourceClass::Storage if image.atomic => (image.numeric_class == N::Uint).then(|| kind(ATOMIC_CUBE_UINT)).flatten(),
                ImageResourceClass::Storage => match image.numeric_class {
                    N::Float => kind(STORAGE_CUBE_FLOAT),
                    N::Uint => kind(STORAGE_CUBE_UINT),
                    _ => None,
                },
                _ => None,
            };
        }
        if image.atomic64 {
            if image.resource_class != ImageResourceClass::Storage || image.numeric_class != N::Uint {
                return None;
            }
            let dimension = STORAGE_DIMENSIONS.iter().position(|&d| d == image.dimension)?;
            return kind(ATOMIC64_UINT + dimension as u32);
        }
        let (base, sampled) = match image.resource_class {
            ImageResourceClass::Sampled => {
                if image.atomic {
                    return None;
                }
                let base = match image.numeric_class {
                    N::Float => SAMPLED_FLOAT,
                    N::Uint => SAMPLED_UINT,
                    N::Sint => SAMPLED_SINT,
                    _ => return None,
                };
                (base, true)
            }
            ImageResourceClass::Storage => {
                let base = if image.atomic {
                    if image.numeric_class != N::Uint {
                        return None;
                    }
                    ATOMIC_UINT
                } else {
                    match image.numeric_class {
                        N::Float => STORAGE_FLOAT,
                        N::Uint => STORAGE_UINT,
                        _ => return None,
                    }
                };
                (base, false)
            }
            ImageResourceClass::None => return None,
        };
        let dimension = match image.dimension {
            D::Dim1D => 0,
            D::Dim1DArray => 1,
            D::Dim2D => 2,
            D::Dim2DArray => 3,
            D::Dim2DMsaa if sampled => 4,
            D::Dim2DMsaaArray if sampled => 5,
            D::Dim3D => {
                if sampled {
                    6
                } else {
                    4
                }
            }
            _ => return None,
        };
        kind(base + dimension)
    }

    fn number(kind: DescriptorBindingKind) -> u32 {
        kind.number()
    }

    pub fn resource_class(kind: DescriptorBindingKind) -> ImageResourceClass {
        let v = number(kind);
        if (SAMPLED_CUBE_FLOAT..=SAMPLED_CUBE_SINT).contains(&v) || (SAMPLED_COMPARE_2D..=SAMPLED_COMPARE_CUBE).contains(&v) {
            return ImageResourceClass::Sampled;
        }
        if (STORAGE_CUBE_FLOAT..=ATOMIC_CUBE_UINT).contains(&v) {
            return ImageResourceClass::Storage;
        }
        if (FIRST_IMAGE_BINDING..FIRST_STORAGE_IMAGE_BINDING).contains(&v) {
            return ImageResourceClass::Sampled;
        }
        if (ATOMIC64_UINT..ATOMIC64_UINT + 5).contains(&v) {
            return ImageResourceClass::Storage;
        }
        if (FIRST_STORAGE_IMAGE_BINDING..SAMPLERS).contains(&v) {
            return ImageResourceClass::Storage;
        }
        ImageResourceClass::None
    }

    pub fn array_index(kind: DescriptorBindingKind) -> u32 {
        number(kind) - FIRST_IMAGE_BINDING
    }

    /// The binding holds R64 unsigned views for 64-bit atomics.
    pub fn is_atomic64(kind: DescriptorBindingKind) -> bool {
        (ATOMIC64_UINT..ATOMIC64_UINT + 5).contains(&number(kind))
    }

    /// The binding samples with a depth reference (depth views).
    pub fn is_compare(kind: DescriptorBindingKind) -> bool {
        (SAMPLED_COMPARE_2D..=SAMPLED_COMPARE_CUBE).contains(&number(kind))
    }

    pub fn is_cube(kind: DescriptorBindingKind) -> bool {
        let v = number(kind);
        (SAMPLED_CUBE_FLOAT..=ATOMIC_CUBE_UINT).contains(&v) || v == SAMPLED_COMPARE_CUBE
    }

    const SAMPLED_DIMENSIONS: [ImageDimension; 7] = [
        ImageDimension::Dim1D,
        ImageDimension::Dim1DArray,
        ImageDimension::Dim2D,
        ImageDimension::Dim2DArray,
        ImageDimension::Dim2DMsaa,
        ImageDimension::Dim2DMsaaArray,
        ImageDimension::Dim3D,
    ];
    const STORAGE_DIMENSIONS: [ImageDimension; 5] =
        [ImageDimension::Dim1D, ImageDimension::Dim1DArray, ImageDimension::Dim2D, ImageDimension::Dim2DArray, ImageDimension::Dim3D];

    /// `(class, numeric class, dimension, atomic)` a binding declares.
    pub fn describe(kind: DescriptorBindingKind) -> (ImageResourceClass, ImageNumericClass, ImageDimension, bool) {
        use ImageNumericClass as N;
        let i = number(kind);
        match i {
            SAMPLED_COMPARE_2D => return (ImageResourceClass::Sampled, N::Float, ImageDimension::Dim2D, false),
            SAMPLED_COMPARE_2D_ARRAY | SAMPLED_COMPARE_CUBE => {
                return (ImageResourceClass::Sampled, N::Float, ImageDimension::Dim2DArray, false)
            }
            _ => {}
        }
        if (SAMPLED_CUBE_FLOAT..=SAMPLED_CUBE_SINT).contains(&i) {
            let n = match i {
                SAMPLED_CUBE_FLOAT => N::Float,
                SAMPLED_CUBE_UINT => N::Uint,
                _ => N::Sint,
            };
            return (ImageResourceClass::Sampled, n, ImageDimension::Dim2DArray, false);
        }
        if (STORAGE_CUBE_FLOAT..=STORAGE_CUBE_UINT).contains(&i) {
            let n = if i == STORAGE_CUBE_FLOAT { N::Float } else { N::Uint };
            return (ImageResourceClass::Storage, n, ImageDimension::Dim2DArray, false);
        }
        if i == ATOMIC_CUBE_UINT {
            return (ImageResourceClass::Storage, N::Uint, ImageDimension::Dim2DArray, true);
        }
        if (SAMPLED_FLOAT..STORAGE_FLOAT).contains(&i) {
            let offset = i - SAMPLED_FLOAT;
            let n = match offset / 7 {
                0 => N::Float,
                1 => N::Uint,
                _ => N::Sint,
            };
            return (ImageResourceClass::Sampled, n, SAMPLED_DIMENSIONS[(offset % 7) as usize], false);
        }
        if (STORAGE_FLOAT..ATOMIC_UINT).contains(&i) {
            let offset = i - STORAGE_FLOAT;
            let n = if offset / 5 == 0 { N::Float } else { N::Uint };
            return (ImageResourceClass::Storage, n, STORAGE_DIMENSIONS[(offset % 5) as usize], false);
        }
        if (ATOMIC64_UINT..ATOMIC64_UINT + 5).contains(&i) {
            return (ImageResourceClass::Storage, N::Uint, STORAGE_DIMENSIONS[(i - ATOMIC64_UINT) as usize], true);
        }
        if (ATOMIC_UINT..SAMPLED_CUBE_FLOAT).contains(&i) {
            return (ImageResourceClass::Storage, N::Uint, STORAGE_DIMENSIONS[(i - ATOMIC_UINT) as usize], true);
        }
        (ImageResourceClass::None, N::Unsupported, ImageDimension::Unknown, false)
    }
}

pub const NO_SHADER_BASE: u32 = u32::MAX;
pub const SHADER_BASE_DWORD_COUNT: u32 = 2;
/// No per-pixel sample masks in the shader data.
pub const NO_QUAD_MASKS: u32 = u32::MAX;
/// The per-pixel sample masks of a pixel program: four 16-bit masks, two
/// per dword (the pixels at (0,0), (1,0) | (0,1), (1,1) of a 2x2 quad).
pub const QUAD_MASKS_DWORD_COUNT: u32 = 2;
const SCALAR_REGISTER_COUNT: usize = 256;

/// The descriptor-set layout of one compiled program and its push data:
/// live user-data registers, the shader base, packed buffer offsets and
/// optional dispatch limits ("shader data"). Shader data lives in push
/// constants when it fits, else in a [`DescriptorBindingKind::ShaderData`]
/// buffer.
#[derive(Clone, Debug)]
pub struct BindingLayout {
    pub push_data_start_dword: u32,
    /// The push cursor the layout was allocated at (context, not layout).
    pub allocation_cursor: u32,
    pub shader_base_dword: u32,
    pub memory_offset_dword: u32,
    pub memory_offset_count: u32,
    /// Every buffer's byte offset takes a full shader-data dword (not four
    /// 8-bit slots in one): the substrate for binding whole ring buffers
    /// with the rotating base patched into the offset.
    pub wide_offsets: bool,
    pub uses_dispatch_thread_limits: bool,
    /// User-data SGPRs live at entry, ascending; shader data dword i holds
    /// `user_data_registers[i]`.
    pub user_data_registers: Vec<u32>,
    pub descriptors: Vec<DescriptorBinding>,
    /// Some image is selected through the host table map: the pipeline
    /// layout adds the [`bindless`] set.
    pub bindless_tables: bool,
    /// A pixel program's per-pixel sample masks (PA_SC_AA_MASK), set per
    /// draw ([`NO_QUAD_MASKS`]: none). Runtime data, so a game that steps
    /// the masks every frame (dithered fades) keeps one module.
    pub quad_masks_dword: u32,
}

impl BindingLayout {
    /// Where buffer `element`'s byte offset lives in the shader data:
    /// (dword index, bit shift, bit width).
    pub fn memory_offset_slot(&self, element: u32) -> (u32, u8, u8) {
        if self.wide_offsets {
            (self.memory_offset_dword + element, 0, 32)
        } else {
            (self.memory_offset_dword + element / 4, (element % 4) as u8 * 8, 8)
        }
    }

    /// Dwords the buffer offsets occupy.
    pub fn memory_offset_dwords(&self) -> u32 {
        if self.wide_offsets {
            self.memory_offset_count
        } else {
            self.memory_offset_count.div_ceil(4)
        }
    }
}

impl PartialEq for BindingLayout {
    fn eq(&self, other: &Self) -> bool {
        self.push_data_start_dword == other.push_data_start_dword
            && self.shader_base_dword == other.shader_base_dword
            && self.memory_offset_dword == other.memory_offset_dword
            && self.memory_offset_count == other.memory_offset_count
            && self.wide_offsets == other.wide_offsets
            && self.uses_dispatch_thread_limits == other.uses_dispatch_thread_limits
            && self.user_data_registers == other.user_data_registers
            && self.descriptors == other.descriptors
            && self.bindless_tables == other.bindless_tables
            && self.quad_masks_dword == other.quad_masks_dword
    }
}

impl BindingLayout {
    pub fn dispatch_thread_limits_dword(&self) -> u32 {
        self.memory_offset_dword + self.memory_offset_dwords()
    }

    pub fn shader_data_dword_count(&self) -> u32 {
        self.dispatch_thread_limits_dword() + if self.uses_dispatch_thread_limits { 3 } else { 0 } + if self.uses_quad_masks() { QUAD_MASKS_DWORD_COUNT } else { 0 }
    }

    pub fn uses_quad_masks(&self) -> bool {
        self.quad_masks_dword != NO_QUAD_MASKS
    }

    /// Reserve the per-pixel sample masks at the end of the shader data
    /// (a pixel program's layout); the push data start is chosen again
    /// for the larger size.
    pub fn add_quad_masks(&mut self) {
        if self.uses_quad_masks() {
            return;
        }
        self.quad_masks_dword = self.shader_data_dword_count();
        let size = self.shader_data_dword_count();
        self.push_data_start_dword = push_data::start_for(self.allocation_cursor, size);
        if self.push_data_start_dword == push_data::NO_START && self.find(DescriptorBindingKind::ShaderData).is_none() {
            self.descriptors.push(DescriptorBinding { kind: DescriptorBindingKind::ShaderData, resources: vec![] });
        }
    }

    pub fn uses_push_data(&self) -> bool {
        self.push_data_start_dword != push_data::NO_START
    }

    pub fn uses_shader_base(&self) -> bool {
        self.shader_base_dword != NO_SHADER_BASE
    }

    pub fn advance_push_data(&self, cursor: &mut u32) {
        if self.uses_push_data() {
            *cursor = self.push_data_start_dword + self.shader_data_dword_count();
        }
    }

    pub fn find(&self, kind: DescriptorBindingKind) -> Option<&DescriptorBinding> {
        self.descriptors.iter().find(|b| b.kind == kind)
    }

    /// The user-data registers live at program entry (some path reads them
    /// before writing them), ascending.
    pub fn collect_user_data_registers(program: &Program, user_data_base: u32, user_data_count: u32) -> Vec<u32> {
        let cfg = ControlFlowGraph::build(&program.instructions, &crate::cfg::calls::CallResolver::new(program, true));
        let block_count = cfg.blocks.len();
        let mut uses = vec![[false; SCALAR_REGISTER_COUNT]; block_count];
        let mut definitions = vec![[false; SCALAR_REGISTER_COUNT]; block_count];
        for instruction in &program.instructions {
            if let Some(block) = cfg.block_of(instruction.pc) {
                record_uses_and_definitions(instruction, &mut uses[block], &mut definitions[block]);
            }
        }
        // live-in = uses | (live-out & not defined), to a fixpoint.
        const WORDS: usize = SCALAR_REGISTER_COUNT / 64;
        let mut use_bits = vec![0u64; block_count * WORDS];
        let mut pass_bits = vec![0u64; block_count * WORDS];
        let mut live_in = vec![0u64; block_count * WORDS];
        for block in 0..block_count {
            for register in 0..SCALAR_REGISTER_COUNT {
                let bit = 1u64 << (register & 63);
                if uses[block][register] {
                    use_bits[block * WORDS + (register >> 6)] |= bit;
                }
                if !definitions[block][register] {
                    pass_bits[block * WORDS + (register >> 6)] |= bit;
                }
            }
        }
        let mut changed = block_count != 0;
        while changed {
            changed = false;
            for block in (0..block_count).rev() {
                for word in 0..WORDS {
                    let live_out = cfg.successors[block].iter().fold(0, |acc, &s| acc | live_in[s * WORDS + word]);
                    let index = block * WORDS + word;
                    let live = use_bits[index] | (live_out & pass_bits[index]);
                    if live != live_in[index] {
                        live_in[index] = live;
                        changed = true;
                    }
                }
            }
        }
        (0..user_data_count)
            .map(|i| user_data_base + i)
            .filter(|&r| block_count != 0 && (r as usize) < SCALAR_REGISTER_COUNT && live_in[(r >> 6) as usize] & (1u64 << (r & 63)) != 0)
            .collect()
    }

    pub fn uses_global_data_share(program: &Program) -> bool {
        program.instructions.iter().any(|i| matches!(&i.control, Some(Control::DataShare(d)) if d.gds))
    }

    /// Does the program read or write LDS (data-share instructions other
    /// than GDS and the lane permutes, which do not touch memory)?
    pub fn uses_local_data_share(program: &Program) -> bool {
        program.instructions.iter().any(|i| {
            matches!(&i.control, Some(Control::DataShare(d)) if !d.gds)
                && !matches!(i.opcode, Opcode::DsSwizzleB32 | Opcode::DsBpermuteB32 | Opcode::DsPermuteB32)
        })
    }

    pub fn reads_shader_base(program: &Program) -> bool {
        program.instructions.iter().any(|i| i.opcode == Opcode::SGetpcB64)
    }

    /// Allocate one program's bindings: buffers, image groups (one element
    /// per mip for a dynamic-mip storage image), samplers, shared buffers,
    /// push data or its fallback buffer.
    pub fn allocate(
        info: &ShaderResourceInfo,
        user_data_registers: Vec<u32>,
        uses_global_data_share: bool,
        uses_flattened_table: bool,
        uses_shader_base: bool,
        push_data_start_dword: u32,
        uses_dispatch_thread_limits: bool,
        wide_offsets: bool,
    ) -> Result<BindingLayout, ResourcePlanError> {
        let user = user_data_registers.len() as u32;
        let shader_base_dword = if uses_shader_base { user } else { NO_SHADER_BASE };
        let memory_offset_dword = user + if uses_shader_base { SHADER_BASE_DWORD_COUNT } else { 0 };
        let memory_offset_count = info.buffers.len() as u32;
        // Narrow packs four 8-bit byte offsets per dword (the < 256 alignment
        // remainder of a binding); wide gives every buffer a full dword, the
        // substrate for binding a whole ring buffer and patching the rotating
        // base offset into it (a data-word split).
        let offset_dwords = if wide_offsets { memory_offset_count } else { memory_offset_count.div_ceil(4) };
        let shader_data_dwords = memory_offset_dword + offset_dwords + if uses_dispatch_thread_limits { 3 } else { 0 };
        let push_start = push_data::start_for(push_data_start_dword, shader_data_dwords);
        let mut descriptors = Vec::new();
        if !info.buffers.is_empty() && !info.buffers_in_guest_memory {
            descriptors.push(DescriptorBinding { kind: DescriptorBindingKind::Buffers, resources: (0..info.buffers.len() as u32).collect() });
        }
        let mut groups: Vec<Vec<u32>> = vec![Vec::new(); IMAGE_BINDING_COUNT as usize];
        for (index, image) in info.images.iter().enumerate() {
            let kind = image_binding::for_image(image)
                .ok_or_else(|| ResourcePlanError::new(format!("shader binding layout failed: image {index} has an invalid binding class")))?;
            let group = image_binding::array_index(kind);
            if group >= IMAGE_BINDING_COUNT {
                return Err(ResourcePlanError::new(format!("shader binding layout failed: image {index} has an unmapped binding class")));
            }
            let dynamic = image.mip_mode == ImageMipMode::Dynamic;
            let count = if dynamic { image.mip_count } else { 1 };
            if count == 0 || (!dynamic && image.mip_count != 1) {
                return Err(ResourcePlanError::new(format!(
                    "shader binding layout failed: image {index} has invalid specialized mip count {}",
                    image.mip_count
                )));
            }
            groups[group as usize].extend(std::iter::repeat_n(index as u32, count as usize));
        }
        for (group, resources) in groups.into_iter().enumerate() {
            if !resources.is_empty() {
                descriptors.push(DescriptorBinding { kind: DescriptorBindingKind::Image(group as u32), resources });
            }
        }
        if !info.samplers.is_empty() {
            descriptors.push(DescriptorBinding { kind: DescriptorBindingKind::Samplers, resources: (0..info.samplers.len() as u32).collect() });
        }
        if uses_global_data_share {
            descriptors.push(DescriptorBinding { kind: DescriptorBindingKind::GlobalDataShare, resources: vec![] });
        }
        if info.uses_device_addresses || (info.buffers_in_guest_memory && !info.buffers.is_empty()) {
            descriptors.push(DescriptorBinding { kind: DescriptorBindingKind::DeviceAddressPageTable, resources: vec![] });
            descriptors.push(DescriptorBinding { kind: DescriptorBindingKind::FaultBuffer, resources: vec![] });
        }
        if uses_flattened_table || info.images.iter().any(|i| i.indirect_search_iterations != 0 || i.bindless) {
            descriptors.push(DescriptorBinding { kind: DescriptorBindingKind::FlattenedResourceTable, resources: vec![] });
        }
        if shader_data_dwords != 0 && push_start == push_data::NO_START {
            descriptors.push(DescriptorBinding { kind: DescriptorBindingKind::ShaderData, resources: vec![] });
        }
        Ok(BindingLayout {
            push_data_start_dword: push_start,
            allocation_cursor: push_data_start_dword,
            shader_base_dword,
            memory_offset_dword,
            memory_offset_count,
            wide_offsets,
            uses_dispatch_thread_limits,
            user_data_registers,
            descriptors,
            bindless_tables: info.images.iter().any(|i| i.bindless),
            quad_masks_dword: NO_QUAD_MASKS,
        })
    }

    /// Recompute the layout from the same inputs and cursor and fail when it
    /// differs, so declarations and descriptor writes never drift.
    #[allow(clippy::too_many_arguments)]
    pub fn validate(
        &self,
        info: &ShaderResourceInfo,
        user_data_registers: &[u32],
        uses_global_data_share: bool,
        uses_flattened_table: bool,
        uses_shader_base: bool,
        hash: u64,
        stage: ShaderStage,
    ) -> Result<(), ResourcePlanError> {
        if self.uses_dispatch_thread_limits && stage != ShaderStage::Compute {
            return Err(ResourcePlanError::new("Only a compute shader can use dispatch thread limits."));
        }
        let expected = BindingLayout::allocate(
            info,
            user_data_registers.to_vec(),
            uses_global_data_share,
            uses_flattened_table,
            uses_shader_base,
            self.allocation_cursor,
            self.uses_dispatch_thread_limits,
            self.wide_offsets,
        )?;
        if expected != *self {
            return Err(ResourcePlanError::new(format!(
                "shader binding layout does not match its resources: hash=0x{hash:016X} stage={stage:?} expected=[{}] actual=[{}]",
                expected.describe(),
                self.describe()
            )));
        }
        Ok(())
    }

    pub fn describe(&self) -> String {
        let bindings: Vec<String> = self.descriptors.iter().map(|b| format!("{:?}:{:?}", b.kind, b.resources)).collect();
        format!(
            "push={} base={} offsets={}+{} user={:?} {}",
            self.push_data_start_dword,
            self.shader_base_dword,
            self.memory_offset_dword,
            self.memory_offset_count,
            self.user_data_registers,
            bindings.join(" ")
        )
    }
}

/// Reads count as uses only before the block defines the register: scalar
/// sources at their width plus the descriptor, address and offset registers
/// of memory instructions.
fn record_uses_and_definitions(instruction: &Instruction, uses: &mut [bool; SCALAR_REGISTER_COUNT], definitions: &mut [bool; SCALAR_REGISTER_COUNT]) {
    let use_range = |uses: &mut [bool; SCALAR_REGISTER_COUNT], definitions: &[bool; SCALAR_REGISTER_COUNT], register: u32, count: u32| {
        for i in 0..count {
            let r = (register + i) as usize;
            if r < SCALAR_REGISTER_COUNT && !definitions[r] {
                uses[r] = true;
            }
        }
    };
    let name = instruction.opcode.name();
    let width = if name.contains("64") { 2 } else { 1 };
    for source in &instruction.sources {
        if source.kind == OperandKind::ScalarRegister {
            use_range(uses, definitions, source.value, width);
        }
    }
    // These read the register their destination names: bit sets, SOPK
    // accumulations and the subvector loop branches modify it, SOPK compares
    // only read it.
    let compares_destination = instruction.encoding == Encoding::Sopk && name.starts_with("SCmpk");
    if compares_destination
        || matches!(instruction.opcode, Opcode::SBitset0B32 | Opcode::SBitset1B32)
        || (instruction.encoding == Encoding::Sopk
            && matches!(instruction.opcode, Opcode::SAddkI32 | Opcode::SMulkI32 | Opcode::SSubvectorLoopBegin | Opcode::SSubvectorLoopEnd))
    {
        for d in &instruction.destinations {
            if d.kind == OperandKind::ScalarRegister {
                use_range(uses, definitions, d.value, 1);
            }
        }
    }
    if compares_destination {
        return;
    }
    match &instruction.control {
        Some(Control::Image(image)) => {
            use_range(uses, definitions, image.scalar_resource, 8);
            if name.starts_with("ImageSample") || name.starts_with("ImageGather") {
                use_range(uses, definitions, image.scalar_sampler, 4);
            }
        }
        Some(Control::ScalarMemory(scalar)) => {
            if let Some(s) = instruction.sources.first().filter(|s| s.kind == OperandKind::ScalarRegister) {
                use_range(uses, definitions, s.value, if name.starts_with("SBufferLoad") { 4 } else { 2 });
            }
            if let Some(r) = scalar.dynamic_offset_register {
                use_range(uses, definitions, r, 1);
            }
        }
        Some(Control::GlobalMemory(global)) if global.scalar_address < 255 => use_range(uses, definitions, global.scalar_address, 2),
        Some(Control::BufferMemory(buffer)) => use_range(uses, definitions, buffer.scalar_resource, 4),
        // A ray query reads its whole four-word BVH T#.
        Some(Control::RayIntersect(ray)) => use_range(uses, definitions, ray.scalar_resource, 4),
        _ => {}
    }
    // A record listing every written register defines each once; a single
    // listed register of a multiword instruction defines the whole width.
    let scalar_destinations: Vec<u32> =
        instruction.destinations.iter().filter(|d| d.kind == OperandKind::ScalarRegister).map(|d| d.value).collect();
    if scalar_destinations.is_empty() {
        return;
    }
    if let Some(Control::ScalarMemory(memory)) = &instruction.control {
        let first = *scalar_destinations.iter().min().unwrap();
        let count = memory.destination_count.max(scalar_destinations.len() as u32);
        for i in 0..count {
            if ((first + i) as usize) < SCALAR_REGISTER_COUNT {
                definitions[(first + i) as usize] = true;
            }
        }
        return;
    }
    let per_destination = if scalar_destinations.len() > 1 { 1 } else { width };
    for d in scalar_destinations {
        for i in 0..per_destination {
            if ((d + i) as usize) < SCALAR_REGISTER_COUNT {
                definitions[(d + i) as usize] = true;
            }
        }
    }
}
