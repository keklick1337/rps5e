// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The memory accesses of a program: one record per memory instruction, one
//! per loaded dword for scalar loads.

use super::format::unified_format;
use crate::ir_types::*;
use crate::opcode::Opcode;
use std::collections::{HashMap, HashSet};

/// Where a FLAT access can land.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FlatAddressSpace {
    /// No aperture reaches the address: an ordinary device address.
    #[default]
    Global,
    /// Built from SHARED_BASE/SHARED_LIMIT (LDS).
    Shared,
    /// Built from PRIVATE_BASE/PRIVATE_LIMIT (scratch).
    Private,
    /// Both apertures reach the address; chosen per lane.
    SharedOrPrivate,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MemoryResourceKind {
    #[default]
    None,
    ScalarBuffer,
    ScalarAddress,
    Buffer,
    Flat,
    Global,
    Scratch,
    LocalDataShare,
    GlobalDataShare,
    Image,
    Sampler,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MemoryAccess {
    #[default]
    None,
    Read,
    Write,
    Atomic,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ImageDimension {
    #[default]
    Unknown,
    Dim1D,
    Dim1DArray,
    Dim2D,
    Dim3D,
    Dim2DArray,
    Dim2DMsaa,
    Dim2DMsaaArray,
}

impl ImageDimension {
    /// The MIMG DIM field.
    pub fn from_mimg(dimension: u32) -> Self {
        match dimension {
            0 => ImageDimension::Dim1D,
            1 => ImageDimension::Dim2D,
            2 => ImageDimension::Dim3D,
            3 => ImageDimension::Dim2DArray, // cube
            4 => ImageDimension::Dim1DArray,
            5 => ImageDimension::Dim2DArray,
            6 => ImageDimension::Dim2DMsaa,
            7 => ImageDimension::Dim2DMsaaArray,
            _ => ImageDimension::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ImageResourceClass {
    #[default]
    None,
    Sampled,
    Storage,
}

/// Image sample flags (bit values follow the MIMG opcode bits they mirror).
pub mod image_sample_flags {
    pub const COMPARE: u32 = 1 << 3;
    pub const ADJUST: u32 = 1 << 10;
    /// A gather with an explicit LOD.
    pub const GATHER_LOD: u32 = 1 << 11;
    /// Raw texel bits (MIMG PCK forms) or several texels at once (BY2 /
    /// BY4): the access needs the T# format.
    pub const PACKED: u32 = 1 << 12;
}

/// Where a guest V#'s dwords come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferDescriptorProvenance {
    /// Evaluated by the host ahead of the draw into the dense buffer table.
    Static,
    /// Fetched by the shader itself from guest memory.
    Runtime,
}

/// How a back-end lowers one access through a guest buffer descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferLoweringStrategy {
    /// A compile-time V# bound as a native storage-buffer descriptor.
    NativeBinding,
    /// A runtime V# addressed through the device-address page table; base,
    /// stride and bounds are read from its raw words at run time.
    PhysicalStorageBuffer,
    /// A runtime V# enumerated into a bounded set of native candidate
    /// bindings selected at run time (needed when the format or type cannot
    /// be reconstructed from the raw words alone).
    BoundedCandidateTable,
}

/// The guest V# behind a buffer access, independent of how the host binds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GuestBufferDescriptor {
    pub provenance: BufferDescriptorProvenance,
    /// A static descriptor's index in [`ShaderResourceInfo::buffers`](super::ShaderResourceInfo).
    pub static_resource: u32,
}

impl GuestBufferDescriptor {
    pub fn runtime() -> Self {
        GuestBufferDescriptor { provenance: BufferDescriptorProvenance::Runtime, static_resource: super::NO_INDEX }
    }

    pub fn static_resource(resource: u32) -> Self {
        GuestBufferDescriptor { provenance: BufferDescriptorProvenance::Static, static_resource: resource }
    }

    /// A static descriptor always binds natively; a runtime one is addressed
    /// through the page table for raw accesses and needs candidate
    /// enumeration otherwise (Vulkan fixes a binding's format at pipeline
    /// creation).
    pub fn choose_strategy(&self, typed: bool, formatted: bool, atomic: bool) -> BufferLoweringStrategy {
        if self.provenance == BufferDescriptorProvenance::Static {
            BufferLoweringStrategy::NativeBinding
        } else if typed || formatted || atomic {
            BufferLoweringStrategy::BoundedCandidateTable
        } else {
            BufferLoweringStrategy::PhysicalStorageBuffer
        }
    }
}

pub const NO_RESOURCE: u32 = u32::MAX;

/// The decoded facts of one memory access. The tracker fills `resource` and
/// `sampler`; the planners set `planning_only`, `address_space`,
/// `device_descriptor` and `buffer_descriptor`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryAccessInfo {
    pub pc: u32,
    pub opcode: Opcode,
    pub kind: MemoryResourceKind,
    pub access: MemoryAccess,
    /// Dense table index (buffer or image), [`NO_RESOURCE`] when unbound.
    pub resource: u32,
    pub sampler: u32,
    /// Immediate byte offset (two's complement for signed fields).
    pub offset: u32,
    pub dmask: u32,
    pub data_dwords: u32,
    pub data_bits: u32,
    pub component_index: u32,
    pub component_count: u32,
    pub data_format: u32,
    pub number_format: u32,
    /// [`image_sample_flags`] bits.
    pub image_sample_flags: u32,
    pub image_dimension: ImageDimension,
    pub image_class: ImageResourceClass,
    pub needs_sampler: bool,
    pub address_is_full: bool,
    pub data_signed: bool,
    pub typed: bool,
    pub formatted: bool,
    pub image_has_mip: bool,
    pub image_r128: bool,
    pub glc: bool,
    pub slc: bool,
    pub index_enabled: bool,
    pub offset_enabled: bool,
    /// A scalar load adds an offset register to its immediate offset.
    pub dynamic_offset: bool,
    /// The access is served by the host (a flattened table read) or by an
    /// indirect-image table and emits no device access of its own.
    pub planning_only: bool,
    pub address_space: FlatAddressSpace,
    /// A buffer access whose V# exists only on the device: the shader reads
    /// through the descriptor in its registers.
    pub device_descriptor: bool,
    pub buffer_descriptor: Option<GuestBufferDescriptor>,
}

impl MemoryAccessInfo {
    fn new(pc: u32, opcode: Opcode, kind: MemoryResourceKind, access: MemoryAccess) -> Self {
        MemoryAccessInfo {
            pc,
            opcode,
            kind,
            access,
            resource: NO_RESOURCE,
            sampler: NO_RESOURCE,
            offset: 0,
            dmask: 0,
            data_dwords: 1,
            data_bits: 32,
            component_index: 0,
            component_count: 1,
            data_format: 0,
            number_format: 0,
            image_sample_flags: 0,
            image_dimension: ImageDimension::Unknown,
            image_class: ImageResourceClass::None,
            needs_sampler: false,
            address_is_full: false,
            data_signed: false,
            typed: false,
            formatted: false,
            image_has_mip: false,
            image_r128: false,
            glc: false,
            slc: false,
            index_enabled: false,
            offset_enabled: false,
            dynamic_offset: false,
            planning_only: false,
            address_space: FlatAddressSpace::Global,
            device_descriptor: false,
            buffer_descriptor: None,
        }
    }

    pub fn is_address_kind(&self) -> bool {
        matches!(
            self.kind,
            MemoryResourceKind::ScalarAddress | MemoryResourceKind::Flat | MemoryResourceKind::Global | MemoryResourceKind::Scratch
        )
    }

    /// Two accesses are the same when every decoded field but the pc and the
    /// opcode agrees.
    pub fn same_access(&self, other: &MemoryAccessInfo) -> bool {
        self.kind == other.kind
            && self.access == other.access
            && self.offset == other.offset
            && self.dmask == other.dmask
            && self.data_dwords == other.data_dwords
            && self.data_bits == other.data_bits
            && self.component_index == other.component_index
            && self.component_count == other.component_count
            && self.data_format == other.data_format
            && self.number_format == other.number_format
            && self.image_sample_flags == other.image_sample_flags
            && self.image_dimension == other.image_dimension
            && self.image_class == other.image_class
            && self.needs_sampler == other.needs_sampler
            && self.address_is_full == other.address_is_full
            && self.data_signed == other.data_signed
            && self.typed == other.typed
            && self.formatted == other.formatted
            && self.image_has_mip == other.image_has_mip
            && self.image_r128 == other.image_r128
            && self.glc == other.glc
            && self.slc == other.slc
            && self.index_enabled == other.index_enabled
            && self.offset_enabled == other.offset_enabled
            && self.dynamic_offset == other.dynamic_offset
            && self.resource == other.resource
            && self.sampler == other.sampler
            && self.planning_only == other.planning_only
            && self.device_descriptor == other.device_descriptor
            && self.address_space == other.address_space
    }
}

/// Every memory access of a program, indexed by pc and component.
#[derive(Clone, Debug, Default)]
pub struct MemoryAccessTable {
    entries: Vec<MemoryAccessInfo>,
    index_by_pc: HashMap<(u32, u32), usize>,
}

impl MemoryAccessTable {
    pub fn entries(&self) -> &[MemoryAccessInfo] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, index: usize) -> &MemoryAccessInfo {
        &self.entries[index]
    }

    pub fn get_mut(&mut self, index: usize) -> &mut MemoryAccessInfo {
        &mut self.entries[index]
    }

    pub fn add(&mut self, entry: MemoryAccessInfo) -> usize {
        let index = self.entries.len();
        self.index_by_pc.insert((entry.pc, entry.component_index), index);
        self.entries.push(entry);
        index
    }

    pub fn index_of(&self, pc: u32, component: u32) -> Option<usize> {
        self.index_by_pc.get(&(pc, component)).copied()
    }

    pub fn find(&self, pc: u32, component: u32) -> Option<&MemoryAccessInfo> {
        self.index_of(pc, component).map(|i| &self.entries[i])
    }

    /// One entry per memory instruction; scalar loads get one per dword.
    /// Buffer loads at `fixed_function_vertex_loads` pcs (replaced by
    /// fixed-function vertex attributes) are left out.
    pub fn build(program: &Program, fixed_function_vertex_loads: Option<&HashSet<u32>>) -> Self {
        let mut table = MemoryAccessTable::default();
        for instruction in &program.instructions {
            match &instruction.control {
                Some(Control::BufferMemory(_)) if fixed_function_vertex_loads.is_some_and(|s| s.contains(&instruction.pc)) => {}
                Some(Control::ScalarMemory(scalar)) => add_scalar_load(&mut table, instruction, scalar),
                Some(Control::BufferMemory(buffer)) => {
                    table.add(from_buffer(instruction, buffer));
                }
                Some(Control::GlobalMemory(global)) => {
                    table.add(from_global(instruction, global));
                }
                Some(Control::Image(image)) => {
                    table.add(from_image(instruction, image));
                }
                Some(Control::RayIntersect(_)) => {
                    // A BVH T# is not an image descriptor: the back-end reads
                    // the nodes through device addresses.
                    let mut entry =
                        MemoryAccessInfo::new(instruction.pc, instruction.opcode, MemoryResourceKind::None, MemoryAccess::None);
                    entry.data_dwords = 4;
                    entry.component_count = 4;
                    table.add(entry);
                }
                Some(Control::DataShare(share)) => {
                    if matches!(instruction.opcode, Opcode::DsSwizzleB32 | Opcode::DsBpermuteB32) {
                        continue;
                    }
                    let kind = if share.gds { MemoryResourceKind::GlobalDataShare } else { MemoryResourceKind::LocalDataShare };
                    let mut entry = MemoryAccessInfo::new(instruction.pc, instruction.opcode, kind, data_share_access(instruction.opcode));
                    entry.offset = share.single_offset_bytes();
                    table.add(entry);
                }
                _ => {}
            }
        }
        table
    }
}

pub(crate) fn is_scalar_buffer_load(opcode: Opcode) -> bool {
    matches!(
        opcode,
        Opcode::SBufferLoadDword
            | Opcode::SBufferLoadDwordx2
            | Opcode::SBufferLoadDwordx4
            | Opcode::SBufferLoadDwordx8
            | Opcode::SBufferLoadDwordx16
    )
}

fn add_scalar_load(table: &mut MemoryAccessTable, instruction: &Instruction, control: &ScalarMemoryControl) {
    let kind = if is_scalar_buffer_load(instruction.opcode) {
        MemoryResourceKind::ScalarBuffer
    } else {
        MemoryResourceKind::ScalarAddress
    };
    for component in 0..control.destination_count {
        let mut entry = MemoryAccessInfo::new(instruction.pc, instruction.opcode, kind, MemoryAccess::Read);
        entry.offset = (control.immediate_offset_bytes as u32).wrapping_add(component * 4);
        entry.data_dwords = 1;
        entry.component_index = component;
        entry.component_count = control.destination_count;
        entry.dynamic_offset = control.dynamic_offset_register.is_some();
        table.add(entry);
    }
}

fn access_of(name: &str) -> MemoryAccess {
    if name.contains("Atomic") {
        MemoryAccess::Atomic
    } else if name.contains("Store") {
        MemoryAccess::Write
    } else {
        MemoryAccess::Read
    }
}

fn subword_bits(name: &str) -> (u32, bool) {
    let lower = name.to_ascii_lowercase();
    if lower.contains("byte") {
        (8, name.contains("Sbyte"))
    } else if lower.contains("short") {
        (16, name.contains("Sshort"))
    } else {
        (32, false)
    }
}

fn from_buffer(instruction: &Instruction, control: &BufferMemoryControl) -> MemoryAccessInfo {
    let name = instruction.opcode.name();
    let (bits, signed) = subword_bits(&name);
    let mut entry = MemoryAccessInfo::new(instruction.pc, instruction.opcode, MemoryResourceKind::Buffer, access_of(&name));
    if control.typed {
        if let Some((data_format, number_format)) = unified_format::decode(control.typed_format) {
            entry.data_format = data_format;
            entry.number_format = number_format;
        }
    }
    entry.offset = control.offset_bytes as u32;
    entry.data_dwords = control.dword_count.max(1);
    entry.data_bits = bits;
    entry.component_count = control.dword_count.max(1);
    entry.data_signed = signed;
    entry.typed = control.typed;
    entry.formatted = name.contains("Format");
    entry.glc = control.glc;
    entry.slc = control.slc;
    entry.index_enabled = control.index_enabled;
    entry.offset_enabled = control.offset_enabled;
    entry
}

fn from_global(instruction: &Instruction, control: &GlobalMemoryControl) -> MemoryAccessInfo {
    let name = instruction.opcode.name();
    let (bits, signed) = subword_bits(&name);
    let kind = if name.starts_with("Scratch") {
        MemoryResourceKind::Scratch
    } else if control.uses_flat_address {
        MemoryResourceKind::Flat
    } else {
        MemoryResourceKind::Global
    };
    let mut entry = MemoryAccessInfo::new(instruction.pc, instruction.opcode, kind, access_of(&name));
    entry.offset = control.offset_bytes as u32;
    entry.data_dwords = control.dword_count.max(1);
    entry.data_bits = bits;
    entry.component_count = control.dword_count.max(1);
    entry.data_signed = signed;
    entry.address_is_full = control.uses_flat_address || control.scalar_address == NULL_SOURCE;
    entry.glc = control.glc;
    entry.slc = control.slc;
    entry
}

fn from_image(instruction: &Instruction, control: &ImageControl) -> MemoryAccessInfo {
    let name = instruction.opcode.name();
    let atomic = name.starts_with("ImageAtomic");
    let store = name.starts_with("ImageStore");
    let sampled = name.starts_with("ImageSample") || name.starts_with("ImageGather") || instruction.opcode == Opcode::ImageGetLod;
    let compare = (name.starts_with("ImageSampleC") && !name.starts_with("ImageSampleCd")) || name.starts_with("ImageGather4C");
    let word0 = instruction.words.first().copied().unwrap_or(0);
    let mimg_opcode = if instruction.words.is_empty() { 0 } else { ((word0 >> 18) & 0x7F) | ((word0 & 1) << 7) };
    let adjust = matches!(
        mimg_opcode,
        0xA0 | 0xA1 | 0xA5 | 0xA6 | 0xA8 | 0xA9 | 0xAD | 0xAE | 0xB0 | 0xB1 | 0xB5 | 0xB6 | 0xB8 | 0xB9 | 0xBD | 0xBE | 0xC8 | 0xD8
    );
    let access = if atomic {
        MemoryAccess::Atomic
    } else if store {
        MemoryAccess::Write
    } else {
        MemoryAccess::Read
    };
    let mut entry = MemoryAccessInfo::new(instruction.pc, instruction.opcode, MemoryResourceKind::Image, access);
    entry.image_class = if atomic || store { ImageResourceClass::Storage } else { ImageResourceClass::Sampled };
    entry.needs_sampler = sampled;
    entry.dmask = control.dmask;
    entry.image_dimension = ImageDimension::from_mimg(control.dimension);
    let gather_lod = matches!(instruction.opcode, Opcode::ImageGather4L | Opcode::ImageGather4LO | Opcode::ImageGather4CL | Opcode::ImageGather4CLO);
    let packed = name.contains("Pck") || name.contains("By2") || name.contains("By4");
    entry.image_sample_flags = if compare { image_sample_flags::COMPARE } else { 0 }
        | if adjust { image_sample_flags::ADJUST } else { 0 }
        | if gather_lod { image_sample_flags::GATHER_LOD } else { 0 }
        | if packed { image_sample_flags::PACKED } else { 0 };
    entry.image_has_mip = name.starts_with("ImageLoadMip") || name.starts_with("ImageStoreMip");
    entry.image_r128 = !instruction.words.is_empty() && (word0 >> 15) & 1 != 0;
    // A 64-bit image atomic has two data dwords per operand (DMASK 0x3, or
    // 0xF for the compare-swap pair).
    if atomic {
        let mask64 = if matches!(instruction.opcode, Opcode::ImageAtomicCmpswap | Opcode::ImageAtomicFcmpswap) { 0xF } else { 0x3 };
        if control.dmask & 0xF == mask64 {
            entry.data_bits = 64;
        }
    }
    entry.glc = control.glc;
    entry.slc = control.slc;
    entry
}

fn data_share_access(opcode: Opcode) -> MemoryAccess {
    if opcode.is_data_share_atomic() {
        MemoryAccess::Atomic
    } else if opcode.name().starts_with("DsWrite") {
        MemoryAccess::Write
    } else {
        MemoryAccess::Read
    }
}
