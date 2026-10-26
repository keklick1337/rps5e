// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Program / instruction model: encodings, operands, per-encoding control
//! fields and pixel-output helpers.

use crate::opcode::Opcode;

/// Microcode format of an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Encoding {
    Sop1,
    Sop2,
    Sopc,
    Sopp,
    Sopk,
    /// Legacy one-dword scalar memory format (top bits 0b11000).
    Smrd,
    Smem,
    Mubuf,
    Mtbuf,
    Vop1,
    Vop2,
    Vopc,
    Vop3,
    Vintrp,
    Ds,
    /// FLAT, SCRATCH and GLOBAL.
    Flat,
    Vop3p,
    Mimg,
    Exp,
}

impl Encoding {
    /// Dwords every instruction of this encoding has before a literal.
    pub fn minimum_dwords(self) -> usize {
        match self {
            Encoding::Vop3
            | Encoding::Smem
            | Encoding::Mubuf
            | Encoding::Mtbuf
            | Encoding::Ds
            | Encoding::Flat
            | Encoding::Vop3p
            | Encoding::Mimg
            | Encoding::Exp => 2,
            _ => 1,
        }
    }
}

/// Kind of an instruction operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OperandKind {
    /// SGPR file index 0..=127 (includes VCC 106/107, TTMP 108..=123, M0 124,
    /// EXEC 126/127).
    ScalarRegister,
    VectorRegister,
    /// 9-bit source encoding that is neither a register nor a literal
    /// (inline constants, SCC/VCCZ/EXECZ, apertures, NULL 125).
    EncodedConstant,
    /// 32-bit literal dword.
    LiteralConstant,
}

/// One instruction operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Operand {
    pub kind: OperandKind,
    pub value: u32,
}

pub const VCC_LO: u32 = 106;
pub const VCC_HI: u32 = 107;
pub const M0: u32 = 124;
pub const NULL_SOURCE: u32 = 125;
pub const EXEC_LO: u32 = 126;
pub const EXEC_HI: u32 = 127;
pub const VCCZ_SOURCE: u32 = 251;
pub const EXECZ_SOURCE: u32 = 252;
pub const SCC_SOURCE: u32 = 253;
pub const LITERAL_SOURCE: u32 = 255;

impl Operand {
    pub const fn scalar(index: u32) -> Self {
        Operand { kind: OperandKind::ScalarRegister, value: index }
    }

    pub const fn vector(index: u32) -> Self {
        Operand { kind: OperandKind::VectorRegister, value: index }
    }

    pub const fn literal(value: u32) -> Self {
        Operand { kind: OperandKind::LiteralConstant, value }
    }

    pub const fn encoded(value: u32) -> Self {
        Operand { kind: OperandKind::EncodedConstant, value }
    }

    /// Decode a 9-bit (or 8-bit scalar) source field. `literal` is the
    /// instruction's trailing literal dword, if it has one.
    pub fn source(encoded: u32, literal: Option<u32>) -> Self {
        if encoded >= 256 {
            return Operand::vector(encoded - 256);
        }
        if let (249 | 255, Some(lit)) = (encoded, literal) {
            return Operand::literal(lit);
        }
        // 108..=123 are the trap temporaries (TTMP0..15). They share the
        // scalar register file with ordinary SGPRs; treating them as encoded
        // constants loses values in continuations that spill through them.
        if encoded <= 123 || matches!(encoded, 124 | 126 | 127) {
            return Operand::scalar(encoded);
        }
        Operand::encoded(encoded)
    }

    pub fn is_scalar(&self, index: u32) -> bool {
        self.kind == OperandKind::ScalarRegister && self.value == index
    }
}

/// Image (MIMG) instruction fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageControl {
    pub dmask: u32,
    pub vector_address: u32,
    /// VADDR followed by the NSA address bytes, if any.
    pub address_registers: Vec<u32>,
    pub vector_data: u32,
    pub scalar_resource: u32,
    pub scalar_sampler: u32,
    pub dimension: u32,
    pub is_array: bool,
    pub glc: bool,
    pub slc: bool,
    pub a16: bool,
    pub d16: bool,
}

impl ImageControl {
    /// VGPR holding address component `component`.
    pub fn address_register(&self, component: usize) -> u32 {
        self.address_registers.get(component).copied().unwrap_or(self.vector_address + component as u32)
    }
}

/// BVH ray-intersect fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RayIntersectControl {
    pub vector_address: u32,
    pub address_registers: Vec<u32>,
    pub vector_data: u32,
    pub scalar_resource: u32,
    pub a16: bool,
}

impl RayIntersectControl {
    pub const RESULT_DWORDS: u32 = 4;

    pub fn address_register(&self, component: usize) -> u32 {
        self.address_registers.get(component).copied().unwrap_or(self.vector_address + component as u32)
    }
}

/// FLAT/GLOBAL/SCRATCH (and SMRD scalar scratch store) fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlobalMemoryControl {
    pub dword_count: u32,
    pub vector_address: u32,
    pub source_vector_register: u32,
    pub destination_vector_register: u32,
    /// SADDR; `u32::MAX` for FLAT until the base is resolved.
    pub scalar_address: u32,
    pub offset_bytes: i32,
    pub glc: bool,
    pub slc: bool,
    pub uses_flat_address: bool,
    pub dynamic_offset_register: Option<u32>,
    pub source_is_scalar: bool,
}

/// MUBUF/MTBUF fields. A typed access carries the unified format from the
/// instruction; an untyped formatted access reads the descriptor format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferMemoryControl {
    pub dword_count: u32,
    pub vector_address: u32,
    pub vector_data: u32,
    pub scalar_resource: u32,
    pub offset_bytes: i32,
    pub index_enabled: bool,
    pub offset_enabled: bool,
    pub glc: bool,
    pub slc: bool,
    pub typed: bool,
    pub typed_format: u32,
}

/// EXP fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportControl {
    pub target: u32,
    pub enable_mask: u32,
    pub compressed: bool,
    pub done: bool,
    pub valid_mask: bool,
}

/// VINTRP fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InterpolationControl {
    pub attribute: u32,
    pub channel: u32,
}

/// VOP3 modifiers (VOP3A, or VOP3B with a scalar carry destination).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vop3Control {
    pub absolute_mask: u32,
    pub negate_mask: u32,
    pub output_modifier: u32,
    pub clamp: bool,
    pub operand_select: u32,
    pub scalar_destination: Option<u32>,
}

/// SDWA modifiers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SdwaControl {
    pub destination_select: u32,
    pub destination_unused: u32,
    pub source0_select: u32,
    pub source1_select: u32,
    pub source0_sign_extend: bool,
    pub source1_sign_extend: bool,
    pub absolute_mask: u32,
    pub negate_mask: u32,
    pub output_modifier: u32,
    pub clamp: bool,
    pub scalar_destination: Option<u32>,
}

/// Packed (VOP3P) modifiers; each mask has one bit per source. OpSel /
/// OpSelHi pick the 16-bit half feeding the low / high result lane; NegLo /
/// NegHi negate the value routed to each lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vop3pControl {
    pub op_sel_mask: u32,
    pub op_sel_hi_mask: u32,
    pub neg_lo_mask: u32,
    pub neg_hi_mask: u32,
    pub clamp: bool,
}

/// DPP16 fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DppControl {
    pub control: u32,
    pub fetch_inactive: bool,
    pub bound_control: bool,
    pub absolute_mask: u32,
    pub negate_mask: u32,
    pub bank_mask: u32,
    pub row_mask: u32,
}

/// DPP8 fields: eight 3-bit lane selectors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dpp8Control {
    pub lane_selectors: u32,
    pub fetch_inactive: bool,
}

/// Scalar load fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarMemoryControl {
    pub destination_count: u32,
    pub immediate_offset_bytes: i32,
    pub dynamic_offset_register: Option<u32>,
}

/// LDS/GDS fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataShareControl {
    pub offset0: u32,
    pub offset1: u32,
    pub gds: bool,
}

impl DataShareControl {
    /// Single-address forms use OFFSET1:OFFSET0 as one 16-bit byte offset;
    /// read2/write2 forms use them as two scaled 8-bit offsets.
    pub fn single_offset_bytes(&self) -> u32 {
        self.offset0 | (self.offset1 << 8)
    }
}

/// Encoding-specific instruction fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Control {
    Image(ImageControl),
    RayIntersect(RayIntersectControl),
    GlobalMemory(GlobalMemoryControl),
    BufferMemory(BufferMemoryControl),
    Export(ExportControl),
    Interpolation(InterpolationControl),
    Vop3(Vop3Control),
    Sdwa(SdwaControl),
    Vop3p(Vop3pControl),
    Dpp(DppControl),
    Dpp8(Dpp8Control),
    ScalarMemory(ScalarMemoryControl),
    DataShare(DataShareControl),
}

impl Control {
    /// Explicit SGPR destination of a VOP3B / VOP3 compare / SDWA compare.
    pub fn scalar_destination(&self) -> Option<u32> {
        match self {
            Control::Vop3(c) => c.scalar_destination,
            Control::Sdwa(c) => c.scalar_destination,
            _ => None,
        }
    }
}

/// One decoded instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instruction {
    /// Byte offset in the (possibly fused) program.
    pub pc: u32,
    pub encoding: Encoding,
    pub opcode: Opcode,
    /// Raw dwords, including any literal / NSA dwords.
    pub words: Vec<u32>,
    pub sources: Vec<Operand>,
    pub destinations: Vec<Operand>,
    pub control: Option<Control>,
    /// Offset from the program entry address when it differs from `pc`
    /// (a continuation spliced at a different pc than its real distance).
    pub address_offset: Option<u64>,
}

impl Instruction {
    /// Byte offset from the program's entry address.
    pub fn program_offset(&self) -> u64 {
        self.address_offset.unwrap_or(self.pc as u64)
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> u32 {
        self.words.len() as u32 * 4
    }

    /// Pc of the next instruction in memory order.
    pub fn next_pc(&self) -> u32 {
        self.pc + self.size_bytes()
    }
}

const PIXEL_COLOR_TARGET_COUNT: u32 = 8;
const PIXEL_COLOR_MASK_BITS: u32 = 4;

/// A decoded shader program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    /// Guest address of the entry instruction.
    pub address: u64,
    pub instructions: Vec<Instruction>,
}

impl Program {
    pub fn new(address: u64, instructions: Vec<Instruction>) -> Self {
        Program { address, instructions }
    }

    /// Per MRT 0..7, the 4-bit component enable mask of its color exports.
    pub fn pixel_color_export_masks(&self) -> u32 {
        let mut masks = 0;
        for e in self.exports() {
            if e.target < PIXEL_COLOR_TARGET_COUNT {
                masks |= (e.enable_mask & 0xF) << (e.target * PIXEL_COLOR_MASK_BITS);
            }
        }
        masks
    }

    /// Bit i set when parameter i (EXP target 32 + i) is exported.
    pub fn parameter_export_mask(&self) -> u32 {
        let mut mask = 0;
        for e in self.exports() {
            if (32..64).contains(&e.target) && e.enable_mask != 0 {
                mask |= 1 << (e.target - 32);
            }
        }
        mask
    }

    pub fn exports(&self) -> impl Iterator<Item = &ExportControl> {
        self.instructions.iter().filter_map(|i| match &i.control {
            Some(Control::Export(e)) => Some(e),
            _ => None,
        })
    }

    pub fn image_resources(&self) -> impl Iterator<Item = &ImageControl> {
        self.instructions.iter().filter_map(|i| match &i.control {
            Some(Control::Image(c)) => Some(c),
            _ => None,
        })
    }

    /// Index of the instruction at `pc`.
    pub fn index_of_pc(&self, pc: u32) -> Option<usize> {
        self.instructions.binary_search_by_key(&pc, |i| i.pc).ok()
    }
}

/// Pixel output component type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelOutputKind {
    Float,
    Uint,
    Sint,
}

/// Selects the logical shader component for each physical color component;
/// two bits per component.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorComponentMapping {
    pub packed: u8,
}

impl Default for ColorComponentMapping {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl ColorComponentMapping {
    pub const IDENTITY_PACKED: u8 = 0xE4;
    pub const IDENTITY: ColorComponentMapping = ColorComponentMapping { packed: Self::IDENTITY_PACKED };

    pub fn new(packed: u8) -> Self {
        ColorComponentMapping { packed }
    }

    pub fn map(self, physical: u32) -> u32 {
        if physical < 4 {
            (self.packed as u32 >> (physical * 2)) & 3
        } else {
            physical
        }
    }

    pub fn apply_mask(self, logical_mask: u32) -> u32 {
        (0..4).fold(0, |m, p| m | (((logical_mask >> self.map(p)) & 1) << p))
    }

    pub fn then(self, next: ColorComponentMapping) -> ColorComponentMapping {
        let packed = (0..4).fold(0u32, |acc, p| acc | (next.map(self.map(p)) << (p * 2)));
        ColorComponentMapping::new(packed as u8)
    }

    pub fn is_identity(self) -> bool {
        self.packed == Self::IDENTITY_PACKED
    }

    /// CB_COLORn_INFO.COMP_SWAP for a format with `component_count`
    /// components.
    pub fn resolve_render_target(component_swap: u32, component_count: u32) -> Option<Self> {
        let packed = match (component_swap, component_count) {
            (0, 1..=4) => Self::IDENTITY_PACKED,
            (1, 1) => 0xE1,
            (1, 2) => 0x6C,
            (1, 3) => 0xB4,
            (1, 4) => 0xC6,
            (2, 1) => 0xC6,
            (2, 2) => 0xE1,
            (2, 3) => 0xC6,
            (2, 4) => 0x1B,
            (3, 1) => 0x27,
            (3, 2) => 0x63,
            (3, 3) => 0x87,
            (3, 4) => 0x93,
            _ => return None,
        };
        Some(ColorComponentMapping::new(packed))
    }
}

/// Binding of one guest pixel output slot to a host attachment location.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PixelOutputBinding {
    pub guest_slot: u32,
    pub host_location: u32,
    pub kind: PixelOutputKind,
    pub component_mapping: ColorComponentMapping,
    /// The EXP MRT target feeding this slot. It differs from the slot when
    /// the program skips targets: the hardware packs color exports into the
    /// slots CB_SHADER_MASK enables.
    pub export_target: Option<u32>,
    /// The target stores 11_11_10 UNORM texels in a 32-bit unsigned host
    /// format: the export packs its first three physical components
    /// (true: truncated, CB_COLOR_INFO.ROUND_MODE; false: rounded half up).
    pub packed_unorm_11_11_10: Option<bool>,
    /// The target stores 4_4_4_4 UNORM texels in a 16-bit unsigned host
    /// format: the export packs its four physical components (X in the
    /// lowest field, matching the texture path's packed-channel order)
    /// rounded half up; a truncating ROUND_MODE is not modelled.
    pub packed_unorm_4_4_4_4: bool,
}

impl PixelOutputBinding {
    pub fn new(guest_slot: u32, host_location: u32, kind: PixelOutputKind) -> Self {
        PixelOutputBinding {
            guest_slot,
            host_location,
            kind,
            component_mapping: ColorComponentMapping::IDENTITY,
            export_target: None,
            packed_unorm_11_11_10: None,
            packed_unorm_4_4_4_4: false,
        }
    }

    pub fn export_target(&self) -> u32 {
        self.export_target.unwrap_or(self.guest_slot)
    }
}

/// SGPRs holding the compute work-group id / thread-group size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ComputeSystemRegisters {
    pub work_group_x: Option<u32>,
    pub work_group_y: Option<u32>,
    pub work_group_z: Option<u32>,
    pub thread_group_size: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_mapping() {
        let id = ColorComponentMapping::IDENTITY;
        assert!(id.is_identity());
        assert_eq!(id.apply_mask(0b0101), 0b0101);
        let bgra = ColorComponentMapping::resolve_render_target(1, 4).unwrap();
        assert_eq!(bgra.packed, 0xC6);
        assert_eq!(bgra.map(0), 2);
        assert_eq!(bgra.map(2), 0);
        assert!(bgra.then(bgra).is_identity());
        assert!(ColorComponentMapping::resolve_render_target(4, 4).is_none());
    }

    #[test]
    fn source_operands() {
        assert_eq!(Operand::source(5, None), Operand::scalar(5));
        assert_eq!(Operand::source(110, None), Operand::scalar(110));
        assert_eq!(Operand::source(124, None), Operand::scalar(124));
        assert_eq!(Operand::source(125, None), Operand::encoded(125));
        assert_eq!(Operand::source(255, Some(7)), Operand::literal(7));
        assert_eq!(Operand::source(255, None), Operand::encoded(255));
        assert_eq!(Operand::source(300, None), Operand::vector(44));
        assert_eq!(Operand::source(128, None), Operand::encoded(128));
    }
}
