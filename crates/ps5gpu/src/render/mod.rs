// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Draw / dispatch state resolution and the render-executor interface.
//!
//! The PM4 interpreter decodes a draw or dispatch packet into a
//! [`DrawPacket`] / [`DispatchPacket`] and calls [`resolve_draw`] /
//! [`resolve_dispatch`] with its register banks. The resolved state —
//! index buffer, draw arguments, color and depth targets (CB/DB registers),
//! viewports and scissor, blend / depth / raster state, and every shader
//! stage's program address, resource words and user data — goes to the
//! [`RenderExecutor`] trait object (the Vulkan back-end in
//! [`crate::backend`], or a recorder in tests).

pub mod formats;
pub mod regfile;
pub mod regs;

use formats::{ComponentMap, RenderTargetFormat};
use regs::{cx, sh, uc};

/// The three register banks of a queue (dword arrays indexed by offset).
#[derive(Clone, Copy)]
pub struct RegisterBanks<'a> {
    pub context: &'a [u32],
    pub shader: &'a [u32],
    pub uconfig: &'a [u32],
}

impl RegisterBanks<'_> {
    #[inline]
    pub fn cx(&self, off: u32) -> u32 {
        self.context.get(off as usize).copied().unwrap_or(0)
    }
    #[inline]
    pub fn sh(&self, off: u32) -> u32 {
        self.shader.get(off as usize).copied().unwrap_or(0)
    }
    #[inline]
    pub fn uc(&self, off: u32) -> u32 {
        self.uconfig.get(off as usize).copied().unwrap_or(0)
    }
    fn cxf(&self, off: u32) -> f32 {
        f32::from_bits(self.cx(off))
    }
}

/// A 40-bit address split into a low register (bits 8..39) and an
/// extension register (bits 40..47).
pub fn split_address(lo: u32, hi: u32) -> u64 {
    ((lo as u64) << 8) | (((hi & 0xFF) as u64) << 40)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DrawKind {
    /// DRAW_INDEX_AUTO (or the private NOP form): `count` vertices.
    #[default]
    Auto,
    /// DRAW_INDEX_2: `count` indices at `index_address`.
    Index,
    /// DRAW_INDEX_OFFSET_2: `count` indices from INDEX_BASE + offset.
    IndexOffset,
    /// DRAW_INDIRECT: arguments at `indirect_address`.
    Indirect,
    /// DRAW_INDEX_INDIRECT: arguments at `indirect_address`.
    IndexIndirect,
    /// DRAW_INDIRECT_MULTI.
    IndirectMulti,
    /// DRAW_INDEX_INDIRECT_MULTI.
    IndexIndirectMulti,
    /// Multi-instanced indexed draw (DISPATCH_DRAW_PREAMBLE form).
    IndexMultiInstanced,
}

impl DrawKind {
    pub fn indexed(self) -> bool {
        !matches!(self, DrawKind::Auto | DrawKind::Indirect | DrawKind::IndirectMulti)
    }
    pub fn indirect(self) -> bool {
        matches!(self, DrawKind::Indirect | DrawKind::IndexIndirect | DrawKind::IndirectMulti | DrawKind::IndexIndirectMulti)
    }
}

/// What the interpreter decoded from a draw packet plus the draw state it
/// keeps outside the register banks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawPacket {
    pub kind: DrawKind,
    pub queue: usize,
    pub packet_address: u64,
    /// Vertex count (auto) or index count (indexed) of a direct draw.
    pub count: u32,
    /// Address of the first index (direct indexed draws).
    pub index_address: u64,
    /// INDEX_BASE (indirect indexed draws take indices from here).
    pub index_base: u64,
    /// Index buffer size in elements (INDEX_BUFFER_SIZE / max size).
    pub index_buffer_size: u32,
    /// Raw VGT_INDEX_TYPE state.
    pub index_type: u32,
    /// NUM_INSTANCES state.
    pub instance_count: u32,
    /// Argument record address of an indirect draw.
    pub indirect_address: u64,
    /// Draw count of a multi-indirect draw (or the maximum with a count address).
    pub indirect_count: u32,
    pub count_address: u64,
    pub stride: u32,
    pub draw_initiator: u32,
    /// Shader registers an indirect draw writes with its argument record
    /// (vertex offset, instance offset, first index): offsets in the SH
    /// register space, [`SH_NOP`] (or any non-user-data offset) for none.
    pub indirect_registers: IndirectRegisters,
}

/// The SH register offset that disables an indirect-draw register write.
pub const SH_NOP: u32 = 0x280;

/// Where an indirect draw stores its offsets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectRegisters {
    pub vertex_offset: u32,
    pub instance_offset: u32,
    pub first_index: u32,
    /// Multi-draws: where the CP writes the index of the record being
    /// drawn (0, 1, …), the draw ID the vertex program looks its per-draw
    /// data up with.
    pub draw_index: u32,
}

impl Default for IndirectRegisters {
    fn default() -> IndirectRegisters {
        IndirectRegisters { vertex_offset: SH_NOP, instance_offset: SH_NOP, first_index: SH_NOP, draw_index: SH_NOP }
    }
}

/// `RPS5E_MSAA_CMASK=0`: multisampled (or FMASK-compressed) targets keep
/// no fast-clear state in CMASK, so their CMASK clears are not applied.
pub fn msaa_cmask_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_MSAA_CMASK").map_or(true, |v| v != "0" && !v.eq_ignore_ascii_case("off")))
}

/// `RPS5E_DRAW_INDEX_REGISTER=0`: multi-draws leave the draw-index
/// register alone (the behaviour before the register was written).
pub fn draw_index_register_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_DRAW_INDEX_REGISTER").map_or(true, |v| v != "0" && !v.eq_ignore_ascii_case("off")))
}

impl IndirectRegisters {
    /// Decode the destination words of DRAW_(INDEX_)INDIRECT_MULTI:
    /// as [`Self::decode`] with `control` (word 3) as the enable word; its
    /// bit 31 (DRAW_INDEX_ENABLE) enables the draw-index destination in
    /// its low half.
    pub fn decode_multi(locations: u32, instance: u32, control: u32, indexed: bool) -> IndirectRegisters {
        let location = control & 0xFFFF;
        IndirectRegisters {
            draw_index: if control & (1 << 31) != 0 && location != 0 { location } else { SH_NOP },
            ..IndirectRegisters::decode(locations, instance, control, indexed)
        }
    }

    /// The register the record index is written to (SH_NOP: none, or the
    /// write is switched off).
    pub fn draw_index_target(&self) -> u32 {
        if draw_index_register_enabled() { self.draw_index } else { SH_NOP }
    }

    /// Decode the destination words of DRAW_(INDEX_)INDIRECT(_MULTI):
    /// `locations` is word 1 (vertex offset low, first index high),
    /// `instance` word 2 (instance offset low) and `enable` the word whose
    /// bit 28 enables the first-index destination (word 2 of the single
    /// indexed packet, word 3 of the multi-draw).
    pub fn decode(locations: u32, instance: u32, enable: u32, indexed: bool) -> IndirectRegisters {
        IndirectRegisters {
            vertex_offset: locations & 0xFFFF,
            instance_offset: instance & 0xFFFF,
            first_index: if indexed && enable & (1 << 28) != 0 { locations >> 16 } else { SH_NOP },
            draw_index: SH_NOP,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexType {
    U16,
    U32,
    U8,
}

impl IndexType {
    pub fn from_raw(v: u32) -> IndexType {
        match v & 3 {
            1 => IndexType::U32,
            2 => IndexType::U8,
            _ => IndexType::U16,
        }
    }
    pub fn bytes(self) -> u64 {
        match self {
            IndexType::U16 => 2,
            IndexType::U32 => 4,
            IndexType::U8 => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexBuffer {
    pub address: u64,
    pub kind: IndexType,
    /// Indices the draw reads (direct draws) or the buffer capacity.
    pub count: u32,
}

impl IndexBuffer {
    pub fn bytes(&self) -> u64 {
        self.count as u64 * self.kind.bytes()
    }
}

/// VGT_PRIMITIVE_TYPE values.
pub mod primitive {
    pub const NONE: u32 = 0;
    pub const POINT_LIST: u32 = 1;
    pub const LINE_LIST: u32 = 2;
    pub const LINE_STRIP: u32 = 3;
    pub const TRI_LIST: u32 = 4;
    pub const TRI_FAN: u32 = 5;
    pub const TRI_STRIP: u32 = 6;
    pub const RECT_LIST: u32 = 7;
    pub const RECT_LIST_LEGACY: u32 = 17;
    pub const QUAD_LIST_LEGACY: u32 = 19;
    pub const POLYGON: u32 = 21;
}

/// Host topology of a guest primitive type (RECT_LIST maps to a patch list
/// the pipeline expands; legacy quads draw as fans).
pub fn topology(prim: u32) -> Option<ash::vk::PrimitiveTopology> {
    use ash::vk::PrimitiveTopology as T;
    Some(match prim {
        primitive::POINT_LIST => T::POINT_LIST,
        primitive::LINE_LIST => T::LINE_LIST,
        primitive::LINE_STRIP => T::LINE_STRIP,
        primitive::TRI_LIST => T::TRIANGLE_LIST,
        primitive::TRI_FAN | primitive::POLYGON | primitive::QUAD_LIST_LEGACY => T::TRIANGLE_FAN,
        primitive::TRI_STRIP => T::TRIANGLE_STRIP,
        primitive::RECT_LIST | primitive::RECT_LIST_LEGACY => T::PATCH_LIST,
        _ => return None,
    })
}

/// One shader stage's program and inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderStage {
    /// Program address (0 when the stage is unbound).
    pub address: u64,
    pub rsrc1: u32,
    pub rsrc2: u32,
    /// User data SGPR values (count from RSRC2.USER_SGPR).
    pub user_data: [u32; 32],
}

impl ShaderStage {
    fn read(b: &RegisterBanks, lo: u32, rsrc1: u32, user0: Option<u32>, user_count: u32) -> ShaderStage {
        let mut user_data = [0u32; 32];
        if let Some(u0) = user0 {
            for (i, v) in user_data.iter_mut().enumerate().take(user_count as usize) {
                *v = b.sh(u0 + i as u32);
            }
        }
        ShaderStage { address: split_address(b.sh(lo), b.sh(lo + 1)), rsrc1: b.sh(rsrc1), rsrc2: b.sh(rsrc1 + 1), user_data }
    }

    pub fn bound(&self) -> bool {
        self.address != 0
    }

    /// RSRC2.USER_SGPR (with the extra high bit).
    pub fn user_sgpr_count(&self) -> u32 {
        ((self.rsrc2 >> 1) & 0x1F) | (((self.rsrc2 >> 27) & 1) << 5)
    }
}

/// Graphics shader stages. On the NGG pipeline the vertex work runs in the
/// GS stage (primitive shader); the legacy VS/ES/HS/LS registers are
/// reported as written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphicsShaders {
    pub ps: ShaderStage,
    pub gs: ShaderStage,
    pub es: ShaderStage,
    pub vs: ShaderStage,
    pub hs: ShaderStage,
    pub ls: ShaderStage,
    /// VGT_SHADER_STAGES_EN.
    pub stages: u32,
    /// Geometry subgroup registers (used when GS_EN is set).
    pub geometry: GeometryRegs,
}

/// The registers that shape a geometry (ES+GS) subgroup.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GeometryRegs {
    /// GE_CNTL: primitives (bits 0..8) and vertices (9..17) per subgroup.
    pub ge_cntl: u32,
    /// GE_MAX_OUTPUT_PER_SUBGROUP: vertices a subgroup may emit.
    pub max_output_per_subgroup: u32,
    /// VGT_GS_MAX_VERT_OUT: vertices one input primitive may emit.
    pub max_vert_out: u32,
    /// VGT_ESGS_RING_ITEMSIZE.
    pub esgs_ring_item_size: u32,
    /// VGT_GS_OUT_PRIM_TYPE.
    pub out_prim_type: u32,
    /// SPI_SHADER_USER_DATA_ADDR_LO/HI_GS (the merged program's s[0:1]).
    pub user_data_address: u64,
}

impl GraphicsShaders {
    /// Write user-data register `reg` (an SH register offset) of the stage
    /// it belongs to; false when `reg` is no graphics user-data register.
    pub fn write_user_data(&mut self, reg: u32, value: u32) -> bool {
        let stages: [(u32, &mut ShaderStage); 5] = [
            (sh::SPI_SHADER_USER_DATA_PS_0, &mut self.ps),
            (sh::SPI_SHADER_USER_DATA_VS_0, &mut self.vs),
            (sh::SPI_SHADER_USER_DATA_GS_0, &mut self.gs),
            (sh::SPI_SHADER_USER_DATA_ES_0, &mut self.es),
            (sh::SPI_SHADER_USER_DATA_HS_0, &mut self.hs),
        ];
        for (base, stage) in stages {
            if let Some(i) = reg.checked_sub(base).filter(|&i| i < 32) {
                stage.user_data[i as usize] = value;
                return true;
            }
        }
        false
    }

    /// The vertex-stage program. Titles on this platform write the NGG
    /// (primitive shader) program address to SPI_SHADER_PGM_LO_ES while its
    /// resources and user data live in the GS registers; a program written
    /// to the GS (or legacy VS) address registers is used as is.
    /// The address of [`Self::vertex_program`] (without copying the stage).
    pub fn vertex_program_address(&self) -> u64 {
        if self.es.bound() {
            self.es.address
        } else if self.gs.bound() {
            self.gs.address
        } else {
            self.vs.address
        }
    }

    pub fn vertex_program(&self) -> ShaderStage {
        if self.es.bound() {
            ShaderStage { address: self.es.address, ..self.gs }
        } else if self.gs.bound() {
            self.gs
        } else {
            self.vs
        }
    }
}

/// A bound color target (CB registers of one slot).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorTarget {
    pub slot: u32,
    pub base: u64,
    pub format: RenderTargetFormat,
    /// CB_COLOR_INFO channel layout / number type / component order.
    pub layout: u32,
    pub number_type: u32,
    pub order: u32,
    pub width: u32,
    pub height: u32,
    /// Tiled row pitch in elements.
    pub pitch: u32,
    /// GFX10 swizzle mode (`tiling::SWIZZLE_*`).
    pub swizzle: u32,
    pub samples: u32,
    pub mip_level: u32,
    pub mip_levels: u32,
    pub base_layer: u32,
    pub layers: u32,
    /// Bytes of one layer of the surface.
    pub slice_size: u64,
    /// CB_TARGET_MASK nibble of the slot (logical components).
    pub write_mask: u32,
    pub fast_clear: bool,
    pub dcc_enabled: bool,
    pub dcc_address: u64,
    pub cmask_address: u64,
    pub fmask_address: u64,
    pub clear_word0: u32,
    pub clear_word1: u32,
    pub dcc_control: u32,
    pub blend: BlendState,
    /// Raw CB_COLOR_INFO / ATTRIB / ATTRIB2 / ATTRIB3 / VIEW.
    pub info: u32,
    pub attrib: u32,
    pub attrib2: u32,
    pub attrib3: u32,
    pub view: u32,
}

impl ColorTarget {
    /// Bytes the target occupies in guest memory (all layers).
    pub fn size(&self) -> u64 {
        self.slice_size * (self.base_layer + self.layers) as u64
    }

    /// The fast-clear color of CB_COLOR_CLEAR_WORD0/1 in the target format
    /// (the words hold the stored texel, moved to the host layout first).
    pub fn register_clear(&self) -> Option<ash::vk::ClearColorValue> {
        self.stored_clear([self.clear_word0, self.clear_word1])
    }

    /// The clear value of one element as the target stores it in memory
    /// (`words`: its bytes, little-endian; elements narrower than 8 bytes
    /// in the low bits), moved to the host layout first.
    pub fn stored_clear(&self, mut words: [u32; 2]) -> Option<ash::vk::ClearColorValue> {
        self.format.transform().apply(&mut words, true);
        formats::decode_clear_color(self.format.host, words)
    }

    /// The clear value a DCC clear code stands for in this target.
    pub fn dcc_clear(&self, code: u8) -> Option<ash::vk::ClearColorValue> {
        if self.format.guest == formats::GuestFormat::R11G11B10Unorm {
            // Packed 11_11_10 UNORM: the colour flag sets every field.
            return match code {
                0x00 | 0x40 => Some(ash::vk::ClearColorValue { uint32: [0; 4] }),
                0x80 | 0xC0 => Some(ash::vk::ClearColorValue { uint32: [u32::MAX, 0, 0, 0] }),
                0x20 => self.register_clear(),
                _ => None,
            };
        }
        if self.format.guest == formats::GuestFormat::Rgba4Unorm {
            // Packed 4_4_4_4 UNORM: the colour flag sets every field (the
            // stored word is the element the program packs).
            return match code {
                0x00 | 0x40 => Some(ash::vk::ClearColorValue { uint32: [0; 4] }),
                0x80 | 0xC0 => Some(ash::vk::ClearColorValue { uint32: [0xFFFF, 0, 0, 0] }),
                0x20 => self.register_clear(),
                _ => None,
            };
        }
        formats::decode_dcc_clear(code, self.format.host, self.register_clear(), formats::dcc_alpha_on_msb(self.layout, self.number_type, self.order))
    }

    /// Bytes of CMASK per layer when the target keeps its fast-clear state
    /// in CMASK (fast clears enabled, no DCC, no VRS hints in CMASK, one
    /// level; multisampled targets with FMASK compression too): 4 KiB per
    /// 1024x512 pixels, whatever the sample count. 0 otherwise.
    pub fn cmask_slice_size(&self) -> u64 {
        let compression = self.info & (1 << 14) != 0;
        let vrs_hint = self.attrib3 & (1 << 31) != 0;
        let msaa = self.samples != 1 || compression;
        if self.dcc_enabled || !self.fast_clear || self.cmask_address == 0 || (msaa && !msaa_cmask_enabled()) || vrs_hint || self.mip_levels != 1 {
            return 0;
        }
        (self.width as u64).div_ceil(1024) * (self.height as u64).div_ceil(512) * 4096
    }

    /// The fast-clear metadata of the target: (base address, bytes per
    /// layer, CMASK rather than DCC). DCC owns the clear when both exist.
    pub fn clear_metadata(&self) -> Option<(u64, u64, bool)> {
        let dcc = self.dcc_slice_size();
        if dcc != 0 {
            return Some((self.dcc_address, dcc, false));
        }
        let cmask = self.cmask_slice_size();
        (cmask != 0).then_some((self.cmask_address, cmask, true))
    }

    /// True for a metadata byte that marks a whole block cleared: a DCC
    /// clear code, or CMASK 0 (every tile holds the register clear).
    pub fn is_clear_code(code: u8, cmask: bool) -> bool {
        if cmask {
            code == 0
        } else {
            formats::is_dcc_clear_code(code)
        }
    }

    /// The clear value a metadata clear code stands for.
    pub fn metadata_clear(&self, code: u8, cmask: bool) -> Option<ash::vk::ClearColorValue> {
        if cmask {
            if code == 0 {
                self.register_clear()
            } else {
                None
            }
        } else {
            self.dcc_clear(code)
        }
    }

    /// Bytes of DCC metadata per layer (64 KiB render-target swizzle, one
    /// mip, one sample): 4 KiB per compressed block.
    pub fn dcc_slice_size(&self) -> u64 {
        if !self.dcc_enabled || self.dcc_address == 0 {
            return 0;
        }
        dcc_slice_bytes(self.width, self.height, self.format.bytes_per_element, self.swizzle, self.samples, self.mip_levels)
    }
}

/// The DCC keys of a surface a shader stored into: `layers` slices of
/// `slice` bytes from `address` (the layout of [`dcc_slice_bytes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoredKeys {
    pub address: u64,
    pub slice: u64,
    pub layers: u32,
}

/// Bytes of DCC metadata per layer of a surface (64 KiB render-target
/// swizzle, one mip, one sample): 4 KiB per compressed block. 0 for other
/// surfaces (their keys are not modelled).
pub fn dcc_slice_bytes(width: u32, height: u32, bpe: u32, swizzle: u32, samples: u32, mip_levels: u32) -> u64 {
    if swizzle != crate::tiling::SWIZZLE_64KB_R_X || samples != 1 || mip_levels != 1 {
        return 0;
    }
    if bpe == 0 || bpe > 16 || !bpe.is_power_of_two() {
        return 0;
    }
    let coverage = 20 - bpe.trailing_zeros();
    let bw = 1u64 << coverage.div_ceil(2);
    let bh = 1u64 << (coverage / 2);
    (width as u64).div_ceil(bw) * (height as u64).div_ceil(bh) * 4096
}

/// CB_BLENDn_CONTROL decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct BlendState {
    pub raw: u32,
    pub enable: bool,
    pub color_src: u32,
    pub color_op: u32,
    pub color_dst: u32,
    pub alpha_src: u32,
    pub alpha_op: u32,
    pub alpha_dst: u32,
    pub separate_alpha: bool,
}

impl BlendState {
    pub fn decode(v: u32) -> BlendState {
        BlendState {
            raw: v,
            color_src: v & 0x1F,
            color_op: (v >> 5) & 7,
            color_dst: (v >> 8) & 0x1F,
            alpha_src: (v >> 16) & 0x1F,
            alpha_op: (v >> 21) & 7,
            alpha_dst: (v >> 24) & 0x1F,
            separate_alpha: v & (1 << 29) != 0,
            enable: v & (1 << 30) != 0,
        }
    }
}

/// The bound depth/stencil target (DB registers).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthTarget {
    pub z_format: u32,
    pub host_format: ash::vk::Format,
    /// Guest bytes per depth element.
    pub bytes_per_element: u32,
    pub has_stencil: bool,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub swizzle: u32,
    pub samples: u32,
    pub base_layer: u32,
    pub layers: u32,
    pub z_read_base: u64,
    pub z_write_base: u64,
    pub stencil_read_base: u64,
    pub stencil_write_base: u64,
    pub htile_base: u64,
    pub htile_enabled: bool,
    /// Bytes of one depth layer.
    pub depth_slice_size: u64,
    pub stencil_slice_size: u64,
    pub z_info: u32,
    pub stencil_info: u32,
    pub depth_view: u32,
}

/// Depth / stencil test state and clears.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct DepthStencilState {
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare: u32,
    pub depth_bounds_enable: bool,
    pub depth_bounds: [f32; 2],
    pub stencil_test: bool,
    pub stencil_compare: u32,
    pub stencil_compare_back: u32,
    pub backface: bool,
    pub stencil_control: u32,
    pub stencil_ref_mask: u32,
    pub stencil_ref_mask_back: u32,
    /// DB_RENDER_CONTROL clear enables and DB_DEPTH_CLEAR / DB_STENCIL_CLEAR.
    pub depth_clear_enable: bool,
    pub stencil_clear_enable: bool,
    pub depth_clear_value: f32,
    pub stencil_clear_value: u8,
    pub render_control: u32,
    pub render_override: u32,
    pub depth_control: u32,
    /// DB_DEPTH_VIEW.Z_READ_ONLY / STENCIL_READ_ONLY: the target accepts no
    /// depth / stencil writes.
    pub depth_write_disable: bool,
    pub stencil_write_disable: bool,
}

impl DepthStencilState {
    /// DB_RENDER_OVERRIDE forces depth / stencil both valid and dirty: the
    /// draw copies the plane from its read base to its write base.
    pub fn depth_copy(&self) -> (bool, bool) {
        let o = self.render_override;
        (o & (1 << 27) != 0 && o & (1 << 29) != 0, o & (1 << 28) != 0 && o & (1 << 30) != 0)
    }
}

/// One viewport in host terms (from PA_CL_VPORT_* and PA_SC_VPORT_ZMIN/MAX).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
    /// Raw scale/offset (x, y, z).
    pub scale: [f32; 3],
    pub offset: [f32; 3],
    pub z_min: f32,
    pub z_max: f32,
}

/// Integer rectangle `[x0, x1) x [y0, y1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Rect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl Rect {
    pub fn intersect(self, o: Rect) -> Rect {
        Rect { x0: self.x0.max(o.x0), y0: self.y0.max(o.y0), x1: self.x1.min(o.x1), y1: self.y1.min(o.y1) }
    }
    pub fn is_empty(&self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }
    pub fn covers(&self, w: u32, h: u32) -> bool {
        self.x0 <= 0 && self.y0 <= 0 && self.x1 >= w as i32 && self.y1 >= h as i32
    }
}

/// Rasterizer state (PA_SU_SC_MODE_CNTL, PA_CL_CLIP_CNTL, ...).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct RasterState {
    pub cull_front: bool,
    pub cull_back: bool,
    pub front_face_cw: bool,
    pub polygon_mode: u32,
    pub provoking_vertex_last: bool,
    pub mode_cntl: u32,
    pub clip_cntl: u32,
    pub vte_cntl: u32,
    /// PA_CL_VS_OUT_CNTL: what the misc position exports carry.
    pub vs_out_cntl: u32,
    pub dx_clip_space: bool,
    /// PA_SU_POLY_OFFSET_DB_FMT_CNTL and the raw float words of
    /// PA_SU_POLY_OFFSET_CLAMP, FRONT_SCALE, FRONT_OFFSET, BACK_SCALE,
    /// BACK_OFFSET.
    pub poly_offset_db_fmt: u32,
    pub poly_offset: [u32; 5],
    /// PA_CL_UCP_0_X..PA_CL_UCP_5_W (raw float words, four per plane);
    /// zero unless PA_CL_CLIP_CNTL.UCP_ENA enables a plane.
    pub user_clip_planes: [u32; 24],
}

impl RasterState {
    /// The depth biases of front and back faces (each as [`Self::depth_bias`]
    /// gives it, zero factors for a face without POLY_OFFSET_*_ENABLE) when
    /// both faces are rasterised and their biases differ. The host has one
    /// bias per draw, so such a draw is drawn twice: front faces, then back
    /// faces. None when one bias serves the draw.
    pub fn face_depth_biases(&self, depth_format: ash::vk::Format) -> Option<([f32; 3], [f32; 3])> {
        if self.cull_front || self.cull_back {
            return None;
        }
        let front = self.mode_cntl & (1 << 11) != 0;
        let back = self.mode_cntl & (1 << 12) != 0;
        if !front && !back {
            return None;
        }
        if front == back && self.poly_offset[1] == self.poly_offset[3] && self.poly_offset[2] == self.poly_offset[4] {
            return None;
        }
        let f = |i: usize| f32::from_bits(self.poly_offset[i]);
        let face = |on: bool, scale: usize| {
            if on {
                [poly_offset_constant(f(scale + 1), self.poly_offset_db_fmt, depth_format), f(0), f(scale) / 16.0]
            } else {
                [0.0, f(0), 0.0]
            }
        };
        Some((face(front, 1), face(back, 3)))
    }

    /// The depth bias of a draw (constant factor, clamp, slope factor) for
    /// a host depth attachment of `depth_format`, None when no polygon
    /// offset applies. A draw whose visible faces need different biases
    /// takes them from [`Self::face_depth_biases`]; otherwise a visible
    /// front face takes precedence.
    pub fn depth_bias(&self, depth_format: ash::vk::Format) -> Option<[f32; 3]> {
        let front = self.mode_cntl & (1 << 11) != 0 && !self.cull_front;
        let back = self.mode_cntl & (1 << 12) != 0 && !self.cull_back;
        if !front && !back {
            return None;
        }
        let f = |i: usize| f32::from_bits(self.poly_offset[i]);
        let (scale, offset) = if front { (f(1), f(2)) } else { (f(3), f(4)) };
        let constant = poly_offset_constant(offset, self.poly_offset_db_fmt, depth_format);
        Some([constant, f(0), scale / 16.0])
    }
}

/// The host constant depth bias of a guest polygon offset: a fixed-point
/// guest depth buffer (PA_SU_POLY_OFFSET_DB_FMT_CNTL.DB_IS_FLOAT_FMT clear)
/// counts the offset in units of 2^NEG_NUM_DB_BITS, which a fixed-point
/// host attachment of N bits expresses as offset * 2^(N + NEG_NUM_DB_BITS).
/// A floating-point host attachment takes the guest value as it is.
pub fn poly_offset_constant(offset: f32, db_fmt_cntl: u32, depth_format: ash::vk::Format) -> f32 {
    use ash::vk::Format as F;
    if db_fmt_cntl & (1 << 8) != 0 {
        return offset;
    }
    let neg_num_db_bits = (db_fmt_cntl & 0xFF) as u8 as i8 as i32;
    let host_bits = match depth_format {
        F::D16_UNORM | F::D16_UNORM_S8_UINT => 16,
        F::D24_UNORM_S8_UINT => 24,
        _ => return offset,
    };
    offset * 2f32.powi(host_bits + neg_num_db_bits)
}

/// CB_COLOR_CONTROL.MODE values.
pub mod color_mode {
    pub const DISABLE: u32 = 0;
    pub const NORMAL: u32 = 1;
    pub const ELIMINATE_FAST_CLEAR: u32 = 2;
    pub const RESOLVE: u32 = 3;
    pub const FMASK_DECOMPRESS: u32 = 5;
    pub const DCC_DECOMPRESS: u32 = 6;
}

/// `RPS5E_COLOR_VIEW_CLAMP=0`: a color view's SLICE_MAX is taken as given
/// even past the array's ATTRIB3.MIP0_DEPTH (the earlier behaviour).
fn color_view_clamp_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_COLOR_VIEW_CLAMP").is_ok_and(|v| v == "0"))
}

/// `RPS5E_LAYERED_VIEW_SLICES=0`: a layered draw's color view is clamped
/// to the array's ATTRIB3 slices as any other (the earlier behaviour).
fn layered_view_slices_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_LAYERED_VIEW_SLICES").is_ok_and(|v| v == "0"))
}

/// Everything a draw needs, resolved from the register banks.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedDraw {
    pub packet: DrawPacket,
    pub index: Option<IndexBuffer>,
    pub primitive_type: u32,
    pub topology: Option<ash::vk::PrimitiveTopology>,
    pub primitive_restart: bool,
    pub restart_index: u32,
    /// GE_INDEX_OFFSET (added to every index).
    pub index_offset: u32,
    pub color_targets: smallvec::SmallVec<[ColorTarget; 8]>,
    /// Slots 0 and 1 regardless of the target mask when CB_COLOR_CONTROL
    /// selects a resolve (slot 0 is resolved into slot 1).
    pub resolve_targets: Option<(ColorTarget, ColorTarget)>,
    pub depth: Option<DepthTarget>,
    pub depth_state: DepthStencilState,
    pub color_mode: u32,
    pub rop3: u32,
    pub target_mask: u32,
    pub shader_mask: u32,
    pub blend_constants: [f32; 4],
    pub viewports: [Viewport; 16],
    /// Combined screen ∩ window ∩ generic ∩ viewport-0 scissor.
    pub scissor: Rect,
    /// The combined scissor of each viewport (`scissors[0] == scissor`).
    pub scissors: [Rect; 16],
    pub raster: RasterState,
    pub shaders: GraphicsShaders,
    /// SPI_PS_INPUT_CNTL_0..31.
    pub ps_input_cntl: [u32; 32],
    pub ps_input_ena: u32,
    pub ps_input_addr: u32,
    pub ps_in_control: u32,
    pub shader_z_format: u32,
    pub shader_col_format: u32,
    pub shader_pos_format: u32,
    pub aa_config: u32,
    /// DB_SHADER_CONTROL.
    pub db_shader_control: u32,
    /// DB_ALPHA_TO_MASK.
    pub alpha_to_mask: u32,
    /// SPI_BARYC_CNTL.FRONT_FACE_ALL_BITS while the pixel program loads the
    /// front-face input: the input reads 0xFFFFFFFF/0 instead of
    /// +1.0f/-1.0f (measured on hardware).
    pub front_face_all_bits: bool,
    /// PA_SC_MODE_CNTL_0.MSAA_ENABLE: the rasterizer uses the programmed
    /// sample locations. Clear on a multisampled target, every sample sits
    /// at the pixel center.
    pub msaa_enable: bool,
    /// PA_SC_MODE_CNTL_1.PS_ITER_SAMPLE: the pixel program runs once per
    /// sample of a multisampled target.
    pub ps_iter_sample: bool,
    /// PA_SC_AA_MASK_X0Y0_X1Y0 and PA_SC_AA_MASK_X0Y1_X1Y1: the samples each
    /// pixel of a 2x2 quad may cover (16 bits per pixel).
    pub aa_mask: [u32; 2],
}

/// The sample masks of the four pixels of a quad (X0Y0, X1Y0, X0Y1, X1Y1)
/// for `samples` samples: with one sample (MSAA_NUM_SAMPLES 0) bit 0 keeps
/// or removes the whole pixel.
pub fn quad_sample_masks(aa_mask: [u32; 2], samples: u32) -> [u32; 4] {
    let full = if samples >= 16 { 0xFFFF } else { (1u32 << samples.max(1)) - 1 };
    let fields = [aa_mask[0] & 0xFFFF, aa_mask[0] >> 16, aa_mask[1] & 0xFFFF, aa_mask[1] >> 16];
    fields.map(|f| if samples <= 1 { if f & 1 != 0 { full } else { 0 } } else { f & full })
}

impl ResolvedDraw {
    /// The pixel shader has effects besides its color exports (depth,
    /// coverage or mask exports, kills, or it runs without a target:
    /// DB_SHADER_CONTROL Z_EXPORT, KILL, MASK_EXPORT, DUAL_EXPORT,
    /// EXEC_ON_NOOP, or DB_ALPHA_TO_MASK's coverage from the MRT0 alpha).
    pub fn pixel_side_effects(&self) -> bool {
        self.db_shader_control & 0x741 != 0 || self.alpha_to_coverage()
    }

    /// DB_ALPHA_TO_MASK.ALPHA_TO_MASK_ENABLE, unless DB_SHADER_CONTROL
    /// bit 11 (ALPHA_TO_MASK_DISABLE) turns it off: titles keep the mask
    /// setting programmed and switch coverage off per draw with the bit.
    pub fn alpha_to_coverage(&self) -> bool {
        self.alpha_to_mask & 1 != 0 && self.db_shader_control & 0x800 == 0
    }

    /// True when no color or depth target is bound and no pixel shader runs.
    pub fn targetless(&self) -> bool {
        self.color_targets.is_empty() && self.depth.is_none()
    }

    /// Color mode is a metadata operation rather than a draw.
    pub fn is_metadata_operation(&self) -> bool {
        matches!(self.color_mode, color_mode::ELIMINATE_FAST_CLEAR | color_mode::FMASK_DECOMPRESS | color_mode::DCC_DECOMPRESS)
    }

    /// A DB depth copy: DB_RENDER_CONTROL.DEPTH_COPY makes the draw write
    /// one sample of the Z_32_FLOAT depth plane into colour target 0
    /// (COPY_CENTROID: the first covered sample, sample 0 under the full
    /// sample mask required here; else COPY_SAMPLE). None when the draw is
    /// no depth copy; Ok(sample) for a copy of the whole surface into the
    /// red channel of a single-sample R32 float target of its size, without
    /// depth or stencil tests, the form the copy is done in; Err(why)
    /// otherwise.
    pub fn db_depth_copy(&self) -> Option<Result<u32, &'static str>> {
        let rc = self.depth_state.render_control;
        if rc & 4 == 0 {
            return None;
        }
        Some(self.db_depth_copy_sample(rc))
    }

    fn db_depth_copy_sample(&self, rc: u32) -> Result<u32, &'static str> {
        if rc & !0xF84 != 0 {
            return Err("combined with a stencil copy, clear, resummarize or decompress");
        }
        if self.depth_state.depth_control & 0xF != 0 {
            return Err("with depth or stencil tests");
        }
        let z = self.depth.as_ref().ok_or("without a depth target")?;
        if z.z_format != formats::depth::Z32_FLOAT {
            return Err("from a depth plane that is not Z_32_FLOAT");
        }
        if z.z_info & 0x000F_1000 != 0 || z.depth_view & 0x3C00_0000 != 0 {
            return Err("from a mipmapped or partially resident depth plane");
        }
        let mut mask = self.target_mask & self.shader_mask;
        for slot in 0..8 {
            if !self.color_targets.iter().any(|t| t.slot == slot) {
                mask &= !(0xF << (slot * 4));
            }
        }
        if mask != 1 {
            return Err("into anything but the red channel of color target 0");
        }
        if self.aa_mask != [u32::MAX; 2] {
            return Err("with a sample mask");
        }
        let sample = if rc & 0x80 != 0 { 0 } else { (rc >> 8) & 0xF };
        if sample >= z.samples {
            return Err("of a sample the depth plane does not have");
        }
        let Some(t) = self.color_targets.iter().find(|t| t.slot == 0) else { return Err("without color target 0") };
        if t.format.host != ash::vk::Format::R32_SFLOAT || t.samples != 1 || t.mip_level != 0 || t.layers != 1 {
            return Err("into a color target that is not a single-sample R32 float");
        }
        if (t.width, t.height) != (z.width, z.height) {
            return Err("into a color target of another size");
        }
        if t.fast_clear {
            return Err("into a color target with CMASK fast clears");
        }
        let (w, h) = (z.width as f32, z.height as f32);
        let v = &self.viewports[0];
        let (x0, x1) = (v.x.min(v.x + v.width), v.x.max(v.x + v.width));
        let (y0, y1) = (v.y.min(v.y + v.height), v.y.max(v.y + v.height));
        let s = self.scissor;
        if x0 > 0.0 || y0 > 0.0 || x1 < w || y1 < h || s.x0 > 0 || s.y0 > 0 || s.x1 < z.width as i32 || s.y1 < z.height as i32 {
            return Err("over part of the depth surface");
        }
        Ok(sample)
    }
}

/// A decoded dispatch packet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DispatchPacket {
    pub queue: usize,
    pub packet_address: u64,
    /// Thread groups (direct dispatches).
    pub groups: [u32; 3],
    /// Argument record address (indirect dispatches; 0 when direct).
    pub indirect_address: u64,
    pub dispatch_initiator: u32,
    /// Only register writes and markers stand between the previous dispatch
    /// of the queue and this one: no partial flush, cache action, wait or
    /// command-processor write, so the console may run both at once and the
    /// title does not rely on one finishing before the other. Not kept in
    /// frame captures (a replay orders every dispatch by its ranges).
    pub unsynced: bool,
}

/// A dispatch resolved from the shader bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedDispatch {
    pub packet: DispatchPacket,
    pub shader: ShaderStage,
    pub rsrc3: u32,
    /// COMPUTE_NUM_THREAD_X/Y/Z (threads per group).
    pub threads: [u32; 3],
    /// COMPUTE_START_X/Y/Z.
    pub start: [u32; 3],
}

/// COMPUTE_DISPATCH_INITIATOR.USE_THREAD_DIMENSIONS: the dimensions of a
/// direct dispatch count threads, not thread groups.
pub const DISPATCH_USE_THREAD_DIMENSIONS: u32 = 1 << 5;

impl ResolvedDispatch {
    /// A direct dispatch whose dimensions count threads.
    pub fn uses_thread_dimensions(&self) -> bool {
        self.packet.indirect_address == 0 && self.packet.dispatch_initiator & DISPATCH_USE_THREAD_DIMENSIONS != 0
    }

    /// Thread groups a direct dispatch launches: its dimensions, or with
    /// thread dimensions the groups covering them (the last ones partial).
    pub fn group_counts(&self) -> [u32; 3] {
        let g = self.packet.groups;
        if !self.uses_thread_dimensions() {
            return g;
        }
        std::array::from_fn(|i| g[i].div_ceil(self.threads[i].max(1)))
    }

    /// The thread counts of a dispatch with thread dimensions whose last
    /// group along some axis is partial: invocations at or past them do
    /// not run (the hardware launches no wave lanes for them).
    pub fn thread_limits(&self) -> Option<[u32; 3]> {
        let g = self.packet.groups;
        (self.uses_thread_dimensions() && (0..3).any(|i| g[i] % self.threads[i].max(1) != 0)).then_some(g)
    }
}

/// Receives resolved draws and dispatches (the GPU back-end).
pub trait RenderExecutor: Send + Sync {
    fn draw(&self, draw: &ResolvedDraw);
    fn dispatch(&self, dispatch: &ResolvedDispatch);
    /// Run `done` once all GPU work recorded so far has completed (end of
    /// pipe). The default runs it immediately.
    fn end_of_pipe(&self, done: Box<dyn FnOnce() + Send>) {
        done();
    }
    /// Like [`Self::end_of_pipe`] without submitting now: the work recorded
    /// so far goes to the GPU with the end of the slice ([`Self::flush`]).
    /// For end-of-pipe label writes, which the CPU must not see before the
    /// work in front of them (or its write-throughs) completed.
    fn after_pipe(&self, done: Box<dyn FnOnce() + Send>) {
        done();
    }
    /// Whether work recorded so far still copies GPU results into guest
    /// memory the CPU can already read (write-throughs of read-through
    /// pages not yet completed): a label written now would let the CPU
    /// read those pages before the GPU results reach them. The default
    /// (work runs at once) has none.
    fn writes_through_pending(&self) -> bool {
        false
    }
    /// A command-processor fill of guest memory (DMA_DATA with an immediate
    /// source); the bytes already reached guest memory.
    fn memory_filled(&self, _addr: u64, _size: u64, _value: u32) {}
    /// A command-processor copy between guest ranges. True when the GPU
    /// executes it (GPU-written memory is involved); false: the caller
    /// copies on the CPU.
    fn copy_memory(&self, _dst: u64, _src: u64, _size: u64) -> bool {
        false
    }
    /// A command-processor fill of guest memory. True when the GPU executes
    /// it; false: the caller fills on the CPU.
    fn fill_memory(&self, _dst: u64, _size: u64, _value: u32) -> bool {
        false
    }
    /// A command-processor write of `data` (e.g. a GDS read-back) to guest
    /// memory. True when the GPU executes it; false: the caller writes on
    /// the CPU.
    fn write_memory(&self, _dst: u64, _data: &[u8]) -> bool {
        false
    }
    /// A command-processor transfer touching the global data share. True
    /// when the GPU executes it (GDS lives on the device, next to the
    /// shader programs using it); false: the caller keeps a CPU copy.
    fn gds_transfer(&self, _op: GdsTransfer) -> bool {
        false
    }
    /// A PIXEL_PIPE_STAT_DUMP occlusion counter dump (EVENT_WRITE
    /// ZPASS_DONE) to `addr`: the begin dump of a query targets its 256-byte
    /// result block, the end dump the block + 8. True when the host counts
    /// the samples of the draws in between and publishes the whole block
    /// through `write` once the GPU retired them; false: the caller
    /// publishes a conservative result itself.
    fn occlusion_counter_dump(&self, _addr: u64, _write: &OcclusionWrite) -> bool {
        false
    }
    /// [`Self::occlusion_counter_dump`] without waiting for its answer:
    /// when the dump is not counted on the GPU, `cpu` (publishing the
    /// conservative result) runs in order with the executor's work. False
    /// when the executor does not take dumps this way (nothing was done:
    /// the caller asks [`Self::occlusion_counter_dump`]).
    fn occlusion_counter_dump_or(&self, _addr: u64, _write: &OcclusionWrite, _cpu: Box<dyn FnOnce() + Send>) -> bool {
        false
    }
    /// End of a submission slice: submit what was recorded.
    fn flush(&self) {}
    /// A CPU synchronisation point (a slice starts, a wait on memory was
    /// satisfied): what the CPU wrote before it is visible to the commands
    /// after it. Executors that record later mark it in order.
    fn sync_point(&self) {
        crate::memory::note_sync_point();
    }
    /// Return once every call made so far was executed (before work that
    /// reads its results outside the executor, such as flips).
    fn drain(&self) {}
    /// Run `f` after every call made so far (flips: they capture what the
    /// draws before them rendered). Now when calls run at once.
    fn run_ordered(&self, f: Box<dyn FnOnce() + Send>) {
        f();
    }
    /// A command-processor fill of guest memory: on the GPU when it holds
    /// GPU-written data, else by `cpu` (which writes guest memory and
    /// returns false when it cannot); `memory_filled` follows a fill that
    /// happened. False when nothing was written. Executors that record
    /// later may run it later, in order with the calls before it.
    fn fill_memory_or(&self, dst: u64, size: u64, value: u32, cpu: Box<dyn FnOnce() -> bool + Send>) -> bool {
        let ok = self.fill_memory(dst, size, value) || cpu();
        if ok {
            self.memory_filled(dst, size, value);
        }
        ok
    }
    /// A command-processor copy: on the GPU when GPU-written data is
    /// involved, else by `cpu` (see [`Self::fill_memory_or`]).
    fn copy_memory_or(&self, dst: u64, src: u64, size: u64, cpu: Box<dyn FnOnce() -> bool + Send>) -> bool {
        self.copy_memory(dst, src, size) || cpu()
    }
    /// The interpreter is about to read or write `[addr, addr+len)` on the
    /// CPU: transfers queued over it must have happened first.
    fn memory_hazard(&self, _addr: u64, _len: u64) {}
    /// A command-processor write by the interpreter (labels, immediate
    /// data): run `write` after every call made before (now when calls run
    /// at once).
    /// `data` is what `write` writes (the bytes later reads see).
    fn write_ordered(&self, _addr: u64, _data: &[u8], write: Box<dyn FnOnce() + Send>) {
        write();
    }
    /// A CPU read of the interpreter: what `read` finds in guest memory
    /// with the transfers queued over the range applied.
    fn read_ordered(&self, _addr: u64, out: &mut [u8], read: &dyn Fn(&mut [u8]) -> bool) -> bool {
        read(out)
    }
    /// A read a one-shot decision of the command processor depends on
    /// (conditional execution, predication, branches, atomics): like
    /// [`Self::read_ordered`], once the GPU work recorded before it would
    /// have been recorded on the calling thread (its written pages then
    /// fault and download on the read).
    fn read_control(&self, addr: u64, out: &mut [u8], read: &dyn Fn(&mut [u8]) -> bool) -> bool {
        self.read_ordered(addr, out, read)
    }
    /// Slice mode (`RPS5E_SLICES`): the interpreter logs register writes,
    /// draws and dispatches here instead of resolving them.
    fn slices(&self) -> Option<&crate::slices::SliceExecutor> {
        None
    }
    /// A submission was processed to its end, or a frame boundary passed:
    /// the point where the caches collect under memory pressure (one
    /// collection tick).
    fn submission_complete(&self) {}
}

/// Writes a finished occlusion result block (`(block address, bytes)`) to
/// guest memory, as a command-processor memory write would (called on the
/// GPU completion thread).
pub type OcclusionWrite = std::sync::Arc<dyn Fn(u64, &[u8]) + Send + Sync>;

/// A command-processor transfer to or from GDS (byte offsets).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GdsTransfer {
    /// Immediate data into GDS.
    Write { offset: u64, data: Vec<u8> },
    /// A dword pattern into GDS.
    Fill { offset: u64, size: u64, value: u32 },
    /// Guest memory into GDS.
    FromMemory { offset: u64, src: u64, size: u64 },
    /// GDS into guest memory.
    ToMemory { offset: u64, dst: u64, size: u64 },
}

fn i15(v: u32) -> i32 {
    (v & 0x7FFF) as u16 as i16 as i32
}
fn i16c(v: u32) -> i32 {
    (v & 0xFFFF) as u16 as i16 as i32
}

/// Resolve color target `slot`. None when it is unbound or masked off
/// (`ignore_mask` keeps masked targets, for resolves/copies).
pub fn resolve_color_target(b: &RegisterBanks, slot: u32, ignore_mask: bool) -> Option<ColorTarget> {
    let r = |off: u32| b.cx(off + slot * cx::CB_COLOR_STRIDE);
    let base = split_address(r(cx::CB_COLOR0_BASE), b.cx(cx::CB_COLOR0_BASE_EXT + slot));
    let mut mask = (b.cx(cx::CB_TARGET_MASK) >> (slot * 4)) & 0xF;
    if ignore_mask && base != 0 && mask == 0 {
        mask = 0xF;
    }
    if !ignore_mask {
        // The CB writes only the components the pixel shader exports:
        // CB_TARGET_MASK & CB_SHADER_MASK (components the shader leaves
        // out keep their contents; a target no export reaches is untouched).
        mask &= (b.cx(cx::CB_SHADER_MASK) >> (slot * 4)) & 0xF;
    }
    if base == 0 || mask == 0 {
        return None;
    }
    let info = r(cx::CB_COLOR0_INFO);
    let attrib = r(cx::CB_COLOR0_ATTRIB);
    let view = r(cx::CB_COLOR0_VIEW);
    let attrib2 = b.cx(cx::CB_COLOR0_ATTRIB2 + slot);
    let attrib3 = b.cx(cx::CB_COLOR0_ATTRIB3 + slot);
    let layout = (info >> 2) & 0x1F;
    let number_type = (info >> 8) & 7;
    let order = (info >> 11) & 3;
    let format = formats::render_target_format(layout, number_type, order)?;
    let fragments_log2 = (attrib >> 15) & 3;
    let samples = 1u32 << fragments_log2;
    let width = ((attrib2 >> 14) & 0x3FFF) + 1;
    let height = (attrib2 & 0x3FFF) + 1;
    let mip_levels = ((attrib2 >> 28) & 0xF) + 1;
    let swizzle = (attrib3 >> 14) & 0x1F;
    let slice_start = view & 0x1FFF;
    let mip_level = (view >> 26) & 0xF;
    // CB_COLOR_VIEW bounds the exported slices; for a 3D target the storage
    // depth is ATTRIB3.MIP0_DEPTH + 1 (per mip), and SLICE_MAX may point past
    // it (titles program depth, not depth - 1).
    let slice_max = if (attrib3 >> 24) & 3 == 2 {
        let depth = (attrib3 & 0x1FFF) + 1;
        ((view >> 13) & 0x1FFF).min((depth >> mip_level).max(1) - 1)
    } else {
        // An array target holds ATTRIB3.MIP0_DEPTH + 1 slices: a view past
        // them (a 2D target bound with a several-slice view) renders into
        // the slices that exist. A layered draw (the vertex stage exports
        // the render target array index, PA_CL_VS_OUT_CNTL bit 18) is
        // clamped by the CB to the view's SLICE_MAX, not to the surface:
        // each layer lands one slice pitch further on, so the view renders
        // as an array of its own slices (a cube map drawn into a one-slice
        // surface).
        let last = (view >> 13) & 0x1FFF;
        let layered = b.cx(cx::PA_CL_VS_OUT_CNTL) & (1 << 18) != 0 && layered_view_slices_enabled();
        if color_view_clamp_enabled() && !layered {
            last.min((attrib3 & 0x1FFF).max(slice_start))
        } else {
            last
        }
    };
    if crate::tiling::pipe_bank_xor(base, swizzle) != 0 {
        crate::tiling::report_pipe_bank_xor("colour target", base, swizzle);
    }
    let bpe = format.bytes_per_element;
    let (pitch, slice_size) = match crate::tiling::thin_surface(swizzle, bpe as usize, width, height) {
        Some((pitch, bytes)) => (pitch, crate::tiling::thin_surface_bytes(swizzle, bpe as usize, width, height, samples).unwrap_or(bytes * samples as usize) as u64),
        None => (width, width as u64 * height as u64 * bpe as u64 * samples as u64),
    };
    let dcc_address = split_address(r(cx::CB_COLOR0_DCC_BASE), b.cx(cx::CB_COLOR0_DCC_BASE_EXT + slot));
    let cmask_address = split_address(r(cx::CB_COLOR0_CMASK), b.cx(cx::CB_COLOR0_CMASK_BASE_EXT + slot));
    let fmask_address = split_address(r(cx::CB_COLOR0_FMASK), b.cx(cx::CB_COLOR0_FMASK_BASE_EXT + slot));
    Some(ColorTarget {
        slot,
        base,
        format,
        layout,
        number_type,
        order,
        width,
        height,
        pitch,
        swizzle,
        samples,
        mip_level,
        mip_levels,
        base_layer: slice_start,
        layers: slice_max.saturating_sub(slice_start) + 1,
        slice_size,
        write_mask: mask,
        fast_clear: info & (1 << 13) != 0,
        dcc_enabled: info & (1 << 28) != 0,
        dcc_address,
        cmask_address,
        fmask_address,
        clear_word0: r(cx::CB_COLOR0_CLEAR_WORD0),
        clear_word1: r(cx::CB_COLOR0_CLEAR_WORD1),
        dcc_control: r(cx::CB_COLOR0_DCC_CONTROL),
        blend: BlendState::decode(b.cx(cx::CB_BLEND0_CONTROL + slot)),
        info,
        attrib,
        attrib2,
        attrib3,
        view,
    })
}

/// Resolve the depth target. None when no depth/stencil state is active or
/// the target is unbound.
pub fn resolve_depth_target(b: &RegisterBanks, ds: &DepthStencilState) -> Option<DepthTarget> {
    let z_info = b.cx(cx::DB_Z_INFO);
    let stencil_info = b.cx(cx::DB_STENCIL_INFO);
    let z_format = z_info & 3;
    let has_stencil = stencil_info & 1 != 0;
    // DB_RENDER_CONTROL.DEPTH_COPY / STENCIL_COPY (bits 2, 3) also read the target.
    let rc = ds.render_control;
    let (depth_copy, stencil_copy) = ds.depth_copy();
    let depth_active = ds.depth_test || ds.depth_bounds_enable || ds.depth_clear_enable || rc & (1 << 2) != 0 || depth_copy;
    let stencil_active = has_stencil && (ds.stencil_test || ds.stencil_clear_enable || rc & (1 << 3) != 0 || stencil_copy);
    if !depth_active && !stencil_active {
        return None;
    }
    let z_read_base = split_address(b.cx(cx::DB_Z_READ_BASE), b.cx(cx::DB_Z_READ_BASE_HI));
    if z_format == formats::depth::INVALID && !has_stencil || z_read_base == 0 && z_format != formats::depth::INVALID {
        return None;
    }
    let size = b.cx(cx::DB_DEPTH_SIZE_XY);
    let width = (size & 0x3FFF) + 1;
    let height = ((size >> 16) & 0x3FFF) + 1;
    let (host_format, bpe) = formats::depth_format(z_format, has_stencil)?;
    let sw_mode = (z_info >> 4) & 0x1F;
    let swizzle = if sw_mode != 0 { sw_mode } else { crate::tiling::SWIZZLE_64KB_Z_X };
    let samples = 1u32 << ((z_info >> 2) & 3);
    let (pitch, depth_slice_size) = match crate::tiling::thin_surface(swizzle, bpe as usize, width, height) {
        Some((pitch, bytes)) => (pitch, crate::tiling::thin_surface_bytes(swizzle, bpe as usize, width, height, samples).unwrap_or(bytes * samples as usize) as u64),
        None => (width, width as u64 * height as u64 * bpe as u64),
    };
    let stencil_slice_size = crate::tiling::thin_surface_bytes(swizzle, 1, width, height, samples).map_or(0, |bytes| bytes as u64);
    let view = b.cx(cx::DB_DEPTH_VIEW);
    // A mipped depth surface (DB_Z_INFO.MAXMIP) stores level 0 after the
    // smaller levels, and DB_DEPTH_VIEW.MIPID picks the level drawn into;
    // only level 0 of a one-level surface is modelled.
    if (z_info >> 16) & 0xF != 0 || (view >> 26) & 0xF != 0 {
        crate::log_once(&format!("ps5gpu: depth target with mip levels (MAXMIP {}, MIPID {}) drawn as a one-level surface at its base", (z_info >> 16) & 0xF, (view >> 26) & 0xF));
    }
    let slice_start = (view & 0x7FF) | (((view >> 11) & 3) << 11);
    let slice_max = ((view >> 13) & 0x7FF) | (((view >> 30) & 3) << 11);
    Some(DepthTarget {
        z_format,
        host_format,
        bytes_per_element: bpe,
        has_stencil,
        width,
        height,
        pitch,
        swizzle,
        samples,
        base_layer: slice_start,
        layers: slice_max.saturating_sub(slice_start) + 1,
        z_read_base,
        z_write_base: split_address(b.cx(cx::DB_Z_WRITE_BASE), b.cx(cx::DB_Z_WRITE_BASE_HI)),
        stencil_read_base: split_address(b.cx(cx::DB_STENCIL_READ_BASE), b.cx(cx::DB_STENCIL_READ_BASE_HI)),
        stencil_write_base: split_address(b.cx(cx::DB_STENCIL_WRITE_BASE), b.cx(cx::DB_STENCIL_WRITE_BASE_HI)),
        htile_base: split_address(b.cx(cx::DB_HTILE_DATA_BASE), b.cx(cx::DB_HTILE_DATA_BASE_HI)),
        htile_enabled: z_info & (1 << 29) != 0,
        depth_slice_size,
        stencil_slice_size,
        z_info,
        stencil_info,
        depth_view: view,
    })
}

fn resolve_depth_state(b: &RegisterBanks) -> DepthStencilState {
    let dc = b.cx(cx::DB_DEPTH_CONTROL);
    let rc = b.cx(cx::DB_RENDER_CONTROL);
    let view = b.cx(cx::DB_DEPTH_VIEW);
    DepthStencilState {
        stencil_test: dc & 1 != 0,
        depth_test: dc & 2 != 0,
        depth_write: dc & 4 != 0,
        depth_bounds_enable: dc & 8 != 0,
        depth_compare: (dc >> 4) & 7,
        backface: dc & (1 << 7) != 0,
        stencil_compare: (dc >> 8) & 7,
        stencil_compare_back: (dc >> 20) & 7,
        depth_bounds: [b.cxf(cx::DB_DEPTH_BOUNDS_MIN), b.cxf(cx::DB_DEPTH_BOUNDS_MAX)],
        stencil_control: b.cx(cx::DB_STENCIL_CONTROL),
        stencil_ref_mask: b.cx(cx::DB_STENCILREFMASK),
        stencil_ref_mask_back: b.cx(cx::DB_STENCILREFMASK_BF),
        depth_clear_enable: rc & 1 != 0,
        stencil_clear_enable: rc & 2 != 0,
        depth_clear_value: b.cxf(cx::DB_DEPTH_CLEAR),
        stencil_clear_value: b.cx(cx::DB_STENCIL_CLEAR) as u8,
        render_control: rc,
        render_override: b.cx(cx::DB_RENDER_OVERRIDE),
        depth_control: dc,
        depth_write_disable: view & (1 << 24) != 0,
        stencil_write_disable: view & (1 << 25) != 0,
    }
}

fn resolve_viewports(b: &RegisterBanks, dx_clip: bool) -> [Viewport; 16] {
    let mut out = [Viewport::default(); 16];
    for (i, v) in out.iter_mut().enumerate() {
        let base = cx::PA_CL_VPORT_XSCALE + i as u32 * 6;
        let f = |k: u32| b.cxf(base + k);
        let (xs, xo, ys, yo, zs, zo) = (f(0), f(1), f(2), f(3), f(4), f(5));
        let z_min = b.cxf(cx::PA_SC_VPORT_ZMIN_0 + i as u32 * 2);
        let z_max = b.cxf(cx::PA_SC_VPORT_ZMIN_0 + i as u32 * 2 + 1);
        let (min_depth, max_depth) = if dx_clip { (zo, zo + zs) } else { (zo - zs, zo + zs) };
        *v = Viewport {
            x: xo - xs,
            y: yo - ys,
            width: 2.0 * xs,
            height: 2.0 * ys,
            min_depth,
            max_depth,
            scale: [xs, ys, zs],
            offset: [xo, yo, zo],
            z_min,
            z_max,
        };
    }
    out
}

/// The scissor of each of the 16 viewports (screen ∩ window ∩ generic ∩
/// viewport scissor `i` when enabled ∩ clip rectangles). `extent` recovers
/// a stale offscreen window scissor (see [`stale_offscreen_window`]).
fn resolve_scissors(b: &RegisterBanks, extent: Option<(u32, u32)>) -> [Rect; 16] {
    let rect15 = |tl: u32, br: u32| Rect { x0: i15(tl), y0: i15(tl >> 16), x1: i15(br), y1: i15(br >> 16) };
    let wo = b.cx(cx::PA_SC_WINDOW_OFFSET);
    let (wox, woy) = (i16c(wo), i16c(wo >> 16));
    let offset = |r: Rect, tl: u32| {
        if tl & (1 << 31) == 0 {
            Rect { x0: r.x0 + wox, y0: r.y0 + woy, x1: r.x1 + wox, y1: r.y1 + woy }
        } else {
            r
        }
    };
    let first = resolve_scissor_with(b, Some(0), extent);
    let mut out = [first; 16];
    if b.cx(cx::PA_SC_MODE_CNTL_0) & 2 == 0 {
        return out;
    }
    // Viewport 0's scissor is part of `first`; the others replace it.
    let without = resolve_scissor_with(b, None, extent);
    for (i, r) in out.iter_mut().enumerate().skip(1) {
        let tl = b.cx(cx::PA_SC_VPORT_SCISSOR_0_TL + 2 * i as u32);
        let vp = offset(rect15(tl, b.cx(cx::PA_SC_VPORT_SCISSOR_0_TL + 2 * i as u32 + 1)), tl);
        *r = without.intersect(vp);
    }
    out
}

/// The scissor of viewport 0 without stale-window recovery (tests; the
/// draw path passes the bound targets' extent).
#[cfg(test)]
fn resolve_scissor(b: &RegisterBanks) -> Rect {
    resolve_scissor_with(b, Some(0), None)
}

/// A window scissor left over from a smaller offscreen pass: some
/// submissions restore a full-target viewport without rewriting
/// PA_SC_WINDOW_SCISSOR, and the stale window can be smaller in only one
/// dimension. The active viewport then covers the whole target (its
/// scissor is exactly the target and its transform spans it), so the
/// window must not clip the draw. Both axes are checked independently.
fn stale_offscreen_window(b: &RegisterBanks, width: u32, height: u32, window: Rect) -> bool {
    if b.cx(cx::PA_SC_MODE_CNTL_0) & 2 == 0 || b.cx(cx::PA_SC_WINDOW_SCISSOR_TL) & (1 << 31) != 0 {
        return false;
    }
    if width == 0 || height == 0 || window.x0 != 0 || window.y0 != 0 || window.x1 <= 0 || window.y1 <= 0 {
        return false;
    }
    if window.x1 as i32 >= width as i32 && window.y1 as i32 >= height as i32 {
        return false;
    }
    let vtl = b.cx(cx::PA_SC_VPORT_SCISSOR_0_TL);
    let vbr = b.cx(cx::PA_SC_VPORT_SCISSOR_0_TL + 1);
    if vtl & (1 << 31) != 0
        || i15(vtl) != 0
        || i15(vtl >> 16) != 0
        || i15(vbr) as i32 != width as i32
        || i15(vbr >> 16) as i32 != height as i32
    {
        return false;
    }
    let (xs, ys, xo, yo) = (b.cxf(cx::PA_CL_VPORT_XSCALE), b.cxf(cx::PA_CL_VPORT_XSCALE + 1), b.cxf(cx::PA_CL_VPORT_XSCALE + 2), b.cxf(cx::PA_CL_VPORT_XSCALE + 3));
    xs.abs() * 2.0 >= width as f32
        && ys.abs() * 2.0 >= height as f32
        && (xo - xs).abs() < 1.0
        && (yo - ys.abs()).abs() < 1.0
}

/// The combined scissor with the scissor of viewport `viewport` (None:
/// without a viewport scissor). `extent` recovers a stale offscreen
/// window scissor against the bound targets' size.
fn resolve_scissor_with(b: &RegisterBanks, viewport: Option<u32>, extent: Option<(u32, u32)>) -> Rect {
    let rect16 = |tl: u32, br: u32| Rect { x0: i16c(tl), y0: i16c(tl >> 16), x1: i16c(br), y1: i16c(br >> 16) };
    let rect15 = |tl: u32, br: u32| Rect { x0: i15(tl), y0: i15(tl >> 16), x1: i15(br), y1: i15(br >> 16) };
    let screen = rect16(b.cx(cx::PA_SC_SCREEN_SCISSOR_TL), b.cx(cx::PA_SC_SCREEN_SCISSOR_BR));
    let wo = b.cx(cx::PA_SC_WINDOW_OFFSET);
    let (wox, woy) = (i16c(wo), i16c(wo >> 16));
    let offset = |r: Rect, tl: u32| {
        if tl & (1 << 31) == 0 {
            Rect { x0: r.x0 + wox, y0: r.y0 + woy, x1: r.x1 + wox, y1: r.y1 + woy }
        } else {
            r
        }
    };
    let wtl = b.cx(cx::PA_SC_WINDOW_SCISSOR_TL);
    let raw_window = rect15(wtl, b.cx(cx::PA_SC_WINDOW_SCISSOR_BR));
    let window = offset(raw_window, wtl);
    let stale = extent.is_some_and(|(w, h)| stale_offscreen_window(b, w, h, raw_window));
    let gtl = b.cx(cx::PA_SC_GENERIC_SCISSOR_TL);
    let generic = offset(rect15(gtl, b.cx(cx::PA_SC_GENERIC_SCISSOR_BR)), gtl);
    let mut r = if stale { screen.intersect(generic) } else { screen.intersect(window).intersect(generic) };
    if let (true, Some(i)) = (b.cx(cx::PA_SC_MODE_CNTL_0) & 2 != 0, viewport) {
        let vtl = b.cx(cx::PA_SC_VPORT_SCISSOR_0_TL + 2 * i);
        let vp = offset(rect15(vtl, b.cx(cx::PA_SC_VPORT_SCISSOR_0_TL + 2 * i + 1)), vtl);
        r = r.intersect(vp);
    }
    // Clip rectangles: the rule is a 16-entry truth table over the four
    // rectangles; rule 0 passes no pixel, 0xFFFF every pixel. A rule that
    // accepts exactly the combinations containing a set of rectangles is
    // their intersection; other rules (unions) are left out.
    let rule = b.cx(cx::PA_SC_CLIPRECT_RULE) & 0xFFFF;
    if rule == 0 {
        return Rect::default();
    }
    if rule != 0xFFFF {
        match clip_rule_intersection(rule as u16) {
            Some(mask) => {
                for i in 0..4u32 {
                    if mask & (1 << i) != 0 {
                        let tl = b.cx(cx::PA_SC_CLIPRECT_0_TL + i * 2);
                        r = r.intersect(offset(rect15(tl, b.cx(cx::PA_SC_CLIPRECT_0_TL + i * 2 + 1)), tl));
                    }
                }
            }
            None => {
                static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 32 {
                    log::warn!("ps5gpu: clip-rect rule {rule:#06x} is not an intersection; scissor left unchanged");
                }
            }
        }
    }
    r
}

/// The rectangles (bit mask) whose intersection a PA_SC_CLIPRECT_RULE
/// truth table describes: the rule accepts combination `c` (bit `c` of the
/// rule, bit `i` of `c` = inside rectangle `i`) exactly when `c` contains
/// every rectangle of the mask. None for other rules.
pub fn clip_rule_intersection(rule: u16) -> Option<u8> {
    (0..16u8).find(|&candidate| {
        let accepted = (0..16u32).filter(|c| c & candidate as u32 == candidate as u32).fold(0u16, |r, c| r | 1 << c);
        accepted == rule
    })
}

/// Register banks written since a draw state was last resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirtyBanks {
    pub context: bool,
    pub shader: bool,
    pub uconfig: bool,
}

impl DirtyBanks {
    pub const ALL: DirtyBanks = DirtyBanks { context: true, shader: true, uconfig: true };
    pub const NONE: DirtyBanks = DirtyBanks { context: false, shader: false, uconfig: false };

    pub fn any(&self) -> bool {
        self.context || self.shader || self.uconfig
    }
}

/// Draw state that only the context bank determines.
struct ContextFields {
    resolve_targets: Option<(ColorTarget, ColorTarget)>,
    depth: Option<DepthTarget>,
    depth_state: DepthStencilState,
    color_mode: u32,
    rop3: u32,
    target_mask: u32,
    shader_mask: u32,
    blend_constants: [f32; 4],
    viewports: [Viewport; 16],
    scissor: Rect,
    scissors: [Rect; 16],
    raster: RasterState,
    ps_input_cntl: [u32; 32],
    ps_input_ena: u32,
    ps_input_addr: u32,
    ps_in_control: u32,
    shader_z_format: u32,
    shader_col_format: u32,
    shader_pos_format: u32,
    aa_config: u32,
    db_shader_control: u32,
    alpha_to_mask: u32,
    restart_index: u32,
    /// SPI_BARYC_CNTL.FRONT_FACE_ALL_BITS while the pixel program loads the
    /// front-face input (see `context_fields`).
    front_face_all_bits: bool,
    msaa_enable: bool,
    ps_iter_sample: bool,
    aa_mask: [u32; 2],
}

/// Resolve the context-bank state; the bound color targets go to
/// `color_targets` (cleared first, its capacity reused).
fn context_fields(b: &RegisterBanks, color_targets: &mut smallvec::SmallVec<[ColorTarget; 8]>) -> ContextFields {
    let color_control = b.cx(cx::CB_COLOR_CONTROL);
    let ds = resolve_depth_state(b);
    let mode_cntl = b.cx(cx::PA_SU_SC_MODE_CNTL);
    let clip_cntl = b.cx(cx::PA_CL_CLIP_CNTL);
    let dx_clip = clip_cntl & (1 << 19) != 0;
    color_targets.clear();
    color_targets.extend((0..8).filter_map(|s| resolve_color_target(b, s, false)));
    let depth = resolve_depth_target(b, &ds);
    // The largest bound extent, for stale offscreen window recovery.
    let mut extent = (0u32, 0u32);
    for t in color_targets.iter() {
        extent.0 = extent.0.max(t.width);
        extent.1 = extent.1.max(t.height);
    }
    if let Some(d) = depth.as_ref() {
        extent.0 = extent.0.max(d.width);
        extent.1 = extent.1.max(d.height);
    }
    let extent = (extent.0 != 0 && extent.1 != 0).then_some(extent);
    let mut ps_input_cntl = [0u32; 32];
    for (i, v) in ps_input_cntl.iter_mut().enumerate() {
        *v = b.cx(cx::SPI_PS_INPUT_CNTL_0 + i as u32);
    }
    ContextFields {
        resolve_targets: if (color_control >> 4) & 7 == color_mode::RESOLVE {
            resolve_color_target(b, 0, true).zip(resolve_color_target(b, 1, true))
        } else {
            None
        },
        depth,
        depth_state: ds,
        color_mode: (color_control >> 4) & 7,
        rop3: (color_control >> 16) & 0xFF,
        target_mask: b.cx(cx::CB_TARGET_MASK),
        shader_mask: b.cx(cx::CB_SHADER_MASK),
        blend_constants: [b.cxf(cx::CB_BLEND_RED), b.cxf(cx::CB_BLEND_RED + 1), b.cxf(cx::CB_BLEND_RED + 2), b.cxf(cx::CB_BLEND_RED + 3)],
        viewports: resolve_viewports(b, dx_clip),
        scissor: resolve_scissor_with(b, Some(0), extent),
        scissors: resolve_scissors(b, extent),
        raster: RasterState {
            cull_front: mode_cntl & 1 != 0,
            cull_back: mode_cntl & 2 != 0,
            front_face_cw: mode_cntl & 4 != 0,
            polygon_mode: (mode_cntl >> 3) & 3,
            provoking_vertex_last: mode_cntl & (1 << 19) != 0,
            mode_cntl,
            clip_cntl,
            vte_cntl: b.cx(cx::PA_CL_VTE_CNTL),
            vs_out_cntl: b.cx(cx::PA_CL_VS_OUT_CNTL),
            dx_clip_space: dx_clip,
            user_clip_planes: if clip_cntl & 0x3F != 0 { std::array::from_fn(|i| b.cx(cx::PA_CL_UCP_0_X + i as u32)) } else { [0; 24] },
            poly_offset_db_fmt: b.cx(cx::PA_SU_POLY_OFFSET_DB_FMT_CNTL),
            poly_offset: [
                b.cx(cx::PA_SU_POLY_OFFSET_CLAMP),
                b.cx(cx::PA_SU_POLY_OFFSET_FRONT_SCALE),
                b.cx(cx::PA_SU_POLY_OFFSET_FRONT_OFFSET),
                b.cx(cx::PA_SU_POLY_OFFSET_BACK_SCALE),
                b.cx(cx::PA_SU_POLY_OFFSET_BACK_OFFSET),
            ],
        },
        ps_input_cntl,
        ps_input_ena: b.cx(cx::SPI_PS_INPUT_ENA),
        ps_input_addr: b.cx(cx::SPI_PS_INPUT_ADDR),
        ps_in_control: b.cx(cx::SPI_PS_IN_CONTROL),
        shader_z_format: b.cx(cx::SPI_SHADER_Z_FORMAT),
        shader_col_format: b.cx(cx::SPI_SHADER_COL_FORMAT),
        shader_pos_format: b.cx(cx::SPI_SHADER_POS_FORMAT),
        aa_config: b.cx(cx::PA_SC_AA_CONFIG),
        db_shader_control: b.cx(cx::DB_SHADER_CONTROL),
        alpha_to_mask: b.cx(cx::DB_ALPHA_TO_MASK),
        restart_index: b.cx(cx::VGT_MULTI_PRIM_IB_RESET_INDX),
        // SPI_BARYC_CNTL.FRONT_FACE_ALL_BITS (bit 24) changes what the
        // front-face input reads (0xFFFFFFFF/0 instead of +1.0f/-1.0f);
        // it matters only while the program loads that input (bit 12 of
        // SPI_PS_INPUT_ADDR). Measured on hardware.
        front_face_all_bits: b.cx(cx::SPI_PS_INPUT_ADDR) >> 12 & 1 != 0 && b.cx(cx::SPI_BARYC_CNTL) >> 24 & 1 != 0,
        msaa_enable: b.cx(cx::PA_SC_MODE_CNTL_0) & 1 != 0,
        ps_iter_sample: b.cx(cx::PA_SC_MODE_CNTL_1) & 0x1_0000 != 0,
        aa_mask: [b.cx(cx::PA_SC_AA_MASK_X0Y0_X1Y0), b.cx(cx::PA_SC_AA_MASK_X0Y1_X1Y1)],
    }
}

/// Every shader stage (shader bank, plus the context and user-config
/// registers the stages depend on).
fn resolve_shaders(b: &RegisterBanks) -> GraphicsShaders {
    let gs_rsrc2 = b.sh(sh::SPI_SHADER_PGM_RSRC1_GS + 1);
    let user_count = |rsrc2: u32| (((rsrc2 >> 1) & 0x1F) | (((rsrc2 >> 27) & 1) << 5)).min(32);
    let ps_rsrc2 = b.sh(sh::SPI_SHADER_PGM_RSRC1_PS + 1);
    GraphicsShaders {
        ps: ShaderStage::read(b, sh::SPI_SHADER_PGM_LO_PS, sh::SPI_SHADER_PGM_RSRC1_PS, Some(sh::SPI_SHADER_USER_DATA_PS_0), user_count(ps_rsrc2).max(16)),
        gs: ShaderStage::read(b, sh::SPI_SHADER_PGM_LO_GS, sh::SPI_SHADER_PGM_RSRC1_GS, Some(sh::SPI_SHADER_USER_DATA_GS_0), user_count(gs_rsrc2).max(16)),
        es: ShaderStage::read(b, sh::SPI_SHADER_PGM_LO_ES, sh::SPI_SHADER_PGM_RSRC1_ES, Some(sh::SPI_SHADER_USER_DATA_ES_0), 32),
        vs: ShaderStage::read(b, sh::SPI_SHADER_PGM_LO_VS, sh::SPI_SHADER_PGM_RSRC1_VS, Some(sh::SPI_SHADER_USER_DATA_VS_0), 32),
        hs: ShaderStage::read(b, sh::SPI_SHADER_PGM_LO_HS, sh::SPI_SHADER_PGM_RSRC1_HS, Some(sh::SPI_SHADER_USER_DATA_HS_0), 32),
        ls: ShaderStage::read(b, sh::SPI_SHADER_PGM_LO_LS, sh::SPI_SHADER_PGM_RSRC1_LS, None, 0),
        stages: b.cx(cx::VGT_SHADER_STAGES_EN),
        geometry: GeometryRegs {
            ge_cntl: b.uc(uc::GE_CNTL),
            max_output_per_subgroup: b.cx(cx::GE_MAX_OUTPUT_PER_SUBGROUP),
            max_vert_out: b.cx(cx::VGT_GS_MAX_VERT_OUT),
            esgs_ring_item_size: b.cx(cx::VGT_ESGS_RING_ITEMSIZE),
            out_prim_type: b.cx(cx::VGT_GS_OUT_PRIM_TYPE),
            user_data_address: b.sh(sh::SPI_SHADER_USER_DATA_ADDR_LO_GS) as u64 | (b.sh(sh::SPI_SHADER_USER_DATA_ADDR_LO_GS + 1) as u64) << 32,
        },
    }
}

/// The index buffer a draw packet reads.
fn packet_index(packet: &DrawPacket) -> Option<IndexBuffer> {
    let kind = IndexType::from_raw(packet.index_type);
    match packet.kind {
        DrawKind::Index | DrawKind::IndexOffset | DrawKind::IndexMultiInstanced => {
            Some(IndexBuffer { address: packet.index_address, kind, count: packet.count })
        }
        DrawKind::IndexIndirect | DrawKind::IndexIndirectMulti => Some(IndexBuffer { address: packet.index_base, kind, count: packet.index_buffer_size }),
        _ => None,
    }
}

/// (primitive type, GE_INDEX_OFFSET) from the user-config bank.
fn uconfig_fields(b: &RegisterBanks) -> (u32, u32) {
    // The legacy rectangle list encoding draws as a rectangle list.
    let primitive_type = match b.uc(uc::VGT_PRIMITIVE_TYPE) {
        primitive::RECT_LIST_LEGACY => primitive::RECT_LIST,
        p => p,
    };
    (primitive_type, b.uc(uc::GE_INDEX_OFFSET))
}

/// Whether primitive restart applies to a draw of `packet`:
/// GE_MULTI_PRIM_IB_RESET_EN.RESET_EN on a strip or fan. With MATCH_ALL_BITS
/// (bit 1) the zero-extended index is compared with all 32 bits of the
/// reset index, so a reset index wider than the index type never matches.
/// Read for every draw: it depends on the packet's index type.
fn primitive_restart(b: &RegisterBanks, primitive_type: u32, packet: &DrawPacket) -> bool {
    let enable = b.uc(uc::GE_MULTI_PRIM_IB_RESET_EN);
    if enable & 1 == 0 || !matches!(primitive_type, primitive::LINE_STRIP | primitive::TRI_FAN | primitive::TRI_STRIP) {
        return false;
    }
    let width_mask = match IndexType::from_raw(packet.index_type) {
        IndexType::U8 => 0xFF,
        IndexType::U16 => 0xFFFF,
        IndexType::U32 => u32::MAX,
    };
    enable & 2 == 0 || b.cx(cx::VGT_MULTI_PRIM_IB_RESET_INDX) & !width_mask == 0
}

/// Resolve a draw's full state from the register banks.
pub fn resolve_draw(b: &RegisterBanks, packet: &DrawPacket) -> ResolvedDraw {
    let mut color_targets = smallvec::SmallVec::new();
    let c = context_fields(b, &mut color_targets);
    let (primitive_type, index_offset) = uconfig_fields(b);
    let primitive_restart = primitive_restart(b, primitive_type, packet);
    ResolvedDraw {
        packet: *packet,
        index: packet_index(packet),
        primitive_type,
        topology: topology(primitive_type),
        primitive_restart,
        restart_index: c.restart_index,
        index_offset,
        color_targets,
        resolve_targets: c.resolve_targets,
        depth: c.depth,
        depth_state: c.depth_state,
        color_mode: c.color_mode,
        rop3: c.rop3,
        target_mask: c.target_mask,
        shader_mask: c.shader_mask,
        blend_constants: c.blend_constants,
        viewports: c.viewports,
        scissor: c.scissor,
        scissors: c.scissors,
        raster: c.raster,
        shaders: resolve_shaders(b),
        ps_input_cntl: c.ps_input_cntl,
        ps_input_ena: c.ps_input_ena,
        ps_input_addr: c.ps_input_addr,
        ps_in_control: c.ps_in_control,
        shader_z_format: c.shader_z_format,
        shader_col_format: c.shader_col_format,
        shader_pos_format: c.shader_pos_format,
        aa_config: c.aa_config,
        db_shader_control: c.db_shader_control,
        alpha_to_mask: c.alpha_to_mask,
        front_face_all_bits: c.front_face_all_bits,
        msaa_enable: c.msaa_enable,
        ps_iter_sample: c.ps_iter_sample,
        aa_mask: c.aa_mask,
    }
}

/// Bring a previously resolved draw up to date for `packet`, re-resolving
/// only what the banks in `dirty` determine (the result equals
/// [`resolve_draw`]). Draws between which only some banks change (most
/// draws change user data alone) skip the rest.
pub fn resolve_draw_into(b: &RegisterBanks, packet: &DrawPacket, dirty: DirtyBanks, d: &mut ResolvedDraw) {
    d.packet = *packet;
    d.index = packet_index(packet);
    if dirty.context {
        let c = context_fields(b, &mut d.color_targets);
        d.restart_index = c.restart_index;
        d.resolve_targets = c.resolve_targets;
        d.depth = c.depth;
        d.depth_state = c.depth_state;
        d.color_mode = c.color_mode;
        d.rop3 = c.rop3;
        d.target_mask = c.target_mask;
        d.shader_mask = c.shader_mask;
        d.blend_constants = c.blend_constants;
        d.viewports = c.viewports;
        d.scissor = c.scissor;
        d.scissors = c.scissors;
        d.raster = c.raster;
        d.ps_input_cntl = c.ps_input_cntl;
        d.ps_input_ena = c.ps_input_ena;
        d.ps_input_addr = c.ps_input_addr;
        d.ps_in_control = c.ps_in_control;
        d.shader_z_format = c.shader_z_format;
        d.shader_col_format = c.shader_col_format;
        d.shader_pos_format = c.shader_pos_format;
        d.aa_config = c.aa_config;
        d.db_shader_control = c.db_shader_control;
        d.alpha_to_mask = c.alpha_to_mask;
        d.front_face_all_bits = c.front_face_all_bits;
        d.msaa_enable = c.msaa_enable;
        d.ps_iter_sample = c.ps_iter_sample;
        d.aa_mask = c.aa_mask;
    }
    if dirty.uconfig {
        let (primitive_type, index_offset) = uconfig_fields(b);
        d.primitive_type = primitive_type;
        d.topology = topology(primitive_type);
        d.index_offset = index_offset;
    }
    d.primitive_restart = primitive_restart(b, d.primitive_type, packet);
    if dirty.any() {
        d.shaders = resolve_shaders(b);
    }
}

/// Resolve a dispatch from the shader bank.
pub fn resolve_dispatch(b: &RegisterBanks, packet: &DispatchPacket) -> ResolvedDispatch {
    let rsrc2 = b.sh(sh::COMPUTE_PGM_RSRC2);
    let count = ((rsrc2 >> 1) & 0x1F).min(16);
    let mut shader = ShaderStage::read(b, sh::COMPUTE_PGM_LO, sh::COMPUTE_PGM_RSRC1, Some(sh::COMPUTE_USER_DATA_0), count.max(16));
    for v in shader.user_data.iter_mut().skip(16) {
        *v = 0;
    }
    ResolvedDispatch {
        packet: *packet,
        shader,
        rsrc3: b.sh(sh::COMPUTE_PGM_RSRC3),
        threads: [b.sh(sh::COMPUTE_NUM_THREAD_X), b.sh(sh::COMPUTE_NUM_THREAD_X + 1), b.sh(sh::COMPUTE_NUM_THREAD_X + 2)],
        start: [b.sh(sh::COMPUTE_START_X), b.sh(sh::COMPUTE_START_X + 1), b.sh(sh::COMPUTE_START_X + 2)],
    }
}

/// Export mapping helper for the shader back-end: physical write mask of
/// a target from its logical CB_TARGET_MASK nibble.
pub fn physical_write_mask(t: &ColorTarget) -> u32 {
    ComponentMap::apply_mask(t.format.export_mapping, t.write_mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Registers of the 64x64x64 RGBA16F volume target a title binds with
    /// SLICE_MAX = 64 (one past the last slice).
    fn volume_target_banks(view: u32, attrib3: u32) -> Vec<u32> {
        let mut ctx = vec![0u32; 0x400];
        ctx[cx::CB_TARGET_MASK as usize] = 0xF;
        ctx[cx::CB_SHADER_MASK as usize] = 0xF;
        ctx[cx::CB_COLOR0_BASE as usize] = (0x50_2367_0000u64 >> 8) as u32;
        ctx[cx::CB_COLOR0_INFO as usize] = 0x1004_0730;
        ctx[cx::CB_COLOR0_VIEW as usize] = view;
        ctx[cx::CB_COLOR0_ATTRIB2 as usize] = 0x000F_C03F;
        ctx[cx::CB_COLOR0_ATTRIB3 as usize] = attrib3;
        ctx
    }

    fn target(ctx: &[u32]) -> ColorTarget {
        let b = RegisterBanks { context: ctx, shader: &[], uconfig: &[] };
        resolve_color_target(&b, 0, false).expect("bound target")
    }

    #[test]
    fn volume_target_layers_clamp_to_storage_depth() {
        let t = target(&volume_target_banks(0x0008_0000, 0x4EC6_C03F));
        assert_eq!((t.width, t.height, t.swizzle), (64, 64, 27));
        assert_eq!(t.base_layer, 0);
        assert_eq!(t.layers, 64);
        // 64 slices of one 64 KiB block: the neighbour 4 MiB above is disjoint.
        assert_eq!(t.size(), 64 * 65536);
    }

    #[test]
    fn array_target_layers_follow_the_view() {
        // A 2D target keeps SLICE_MAX as the last exported slice.
        let t = target(&volume_target_banks((3 << 13) | 1, 0x4DC6_C003));
        assert_eq!((t.base_layer, t.layers), (1, 3));
        // A view past the array (ATTRIB3 last slice 0) renders one slice:
        // the same target as a one-slice view.
        let wide = target(&volume_target_banks(35 << 13, 0x4DC6_C000));
        let one = target(&volume_target_banks(0, 0x4DC6_C000));
        assert_eq!((wide.base_layer, wide.layers), (0, 1));
        assert_eq!((wide.size(), crate::images::color_key(&wide)), (one.size(), crate::images::color_key(&one)));
        let t = target(&volume_target_banks((5 << 13) | 1, 0x4DC6_C003));
        assert_eq!((t.base_layer, t.layers), (1, 3));
    }

    /// A layered draw (render target array index exported) into a view of
    /// slices 0..5 of a one-slice surface renders six slices, one slice
    /// pitch apart (the CB clamps the layer to SLICE_MAX); without the
    /// export the view stays clamped to the surface.
    #[test]
    fn layered_draw_renders_the_views_slices() {
        let mut ctx = volume_target_banks(5 << 13, 0x4DC6_C000);
        assert_eq!(target(&ctx).layers, 1);
        ctx[cx::PA_CL_VS_OUT_CNTL as usize] = 1 << 18 | 1 << 21;
        let t = target(&ctx);
        assert_eq!((t.base_layer, t.layers), (0, 6));
        assert_eq!(t.size(), 6 * t.slice_size);
        // A view inside a larger array is unchanged by the export.
        let mut array = volume_target_banks((3 << 13) | 1, 0x4DC6_C007);
        array[cx::PA_CL_VS_OUT_CNTL as usize] = 1 << 18 | 1 << 21;
        assert_eq!((target(&array).base_layer, target(&array).layers), (1, 3));
    }

    /// Updating a resolved draw for only the banks that changed gives the
    /// same state as resolving it again.
    #[test]
    fn incremental_resolve_matches_full_resolve() {
        let mut ctx = volume_target_banks(0x0008_0000, 0x4EC6_C03F);
        let mut shr = vec![0u32; 0x300];
        let mut ucr = vec![0u32; 0x4000];
        ucr[uc::VGT_PRIMITIVE_TYPE as usize] = primitive::TRI_LIST;
        shr[sh::SPI_SHADER_PGM_LO_PS as usize] = 0x1234;
        shr[sh::SPI_SHADER_USER_DATA_PS_0 as usize] = 7;
        let packet = DrawPacket { kind: DrawKind::Auto, count: 3, instance_count: 1, ..Default::default() };
        let full = |c: &[u32], s: &[u32], u: &[u32], p: &DrawPacket| resolve_draw(&RegisterBanks { context: c, shader: s, uconfig: u }, p);
        let mut cached = full(&ctx, &shr, &ucr, &packet);
        // User data only.
        shr[sh::SPI_SHADER_USER_DATA_PS_0 as usize] = 8;
        let p2 = DrawPacket { count: 6, ..packet };
        let dirty = DirtyBanks { shader: true, ..DirtyBanks::NONE };
        resolve_draw_into(&RegisterBanks { context: &ctx, shader: &shr, uconfig: &ucr }, &p2, dirty, &mut cached);
        assert_eq!(cached, full(&ctx, &shr, &ucr, &p2));
        // Context: the target goes away, blend constants change.
        ctx[cx::CB_TARGET_MASK as usize] = 0;
        ctx[cx::CB_BLEND_RED as usize] = 1.0f32.to_bits();
        let dirty = DirtyBanks { context: true, ..DirtyBanks::NONE };
        resolve_draw_into(&RegisterBanks { context: &ctx, shader: &shr, uconfig: &ucr }, &p2, dirty, &mut cached);
        assert_eq!(cached, full(&ctx, &shr, &ucr, &p2));
        assert!(cached.color_targets.is_empty());
        // User config: strips with restart.
        ucr[uc::VGT_PRIMITIVE_TYPE as usize] = primitive::TRI_STRIP;
        ucr[uc::GE_MULTI_PRIM_IB_RESET_EN as usize] = 1;
        let p3 = DrawPacket { kind: DrawKind::Index, index_address: 0x1000, ..p2 };
        let dirty = DirtyBanks { uconfig: true, ..DirtyBanks::NONE };
        resolve_draw_into(&RegisterBanks { context: &ctx, shader: &shr, uconfig: &ucr }, &p3, dirty, &mut cached);
        assert_eq!(cached, full(&ctx, &shr, &ucr, &p3));
        assert!(cached.primitive_restart);
        // MATCH_ALL_BITS: a 32-bit reset index never matches 16-bit
        // indices; with 32-bit indices of the next draw it does.
        ucr[uc::GE_MULTI_PRIM_IB_RESET_EN as usize] = 3;
        ctx[cx::VGT_MULTI_PRIM_IB_RESET_INDX as usize] = u32::MAX;
        let dirty = DirtyBanks { context: true, uconfig: true, ..DirtyBanks::NONE };
        resolve_draw_into(&RegisterBanks { context: &ctx, shader: &shr, uconfig: &ucr }, &p3, dirty, &mut cached);
        assert_eq!(cached, full(&ctx, &shr, &ucr, &p3));
        assert!(!cached.primitive_restart);
        let p4 = DrawPacket { index_type: 1, ..p3 };
        resolve_draw_into(&RegisterBanks { context: &ctx, shader: &shr, uconfig: &ucr }, &p4, DirtyBanks::NONE, &mut cached);
        assert_eq!(cached, full(&ctx, &shr, &ucr, &p4));
        assert!(cached.primitive_restart);
    }

    /// Primitive type 17 is the legacy encoding of a rectangle list, not a
    /// strip: three vertices per rectangle.
    #[test]
    fn legacy_rect_list_is_a_rect_list() {
        let ctx = vec![0u32; 0x400];
        let shr = vec![0u32; 0x300];
        let mut ucr = vec![0u32; 0x4000];
        ucr[uc::VGT_PRIMITIVE_TYPE as usize] = primitive::RECT_LIST_LEGACY;
        let packet = DrawPacket { kind: DrawKind::Auto, count: 3, instance_count: 1, ..Default::default() };
        let d = resolve_draw(&RegisterBanks { context: &ctx, shader: &shr, uconfig: &ucr }, &packet);
        assert_eq!(d.primitive_type, primitive::RECT_LIST);
        assert_eq!(topology(primitive::RECT_LIST_LEGACY), Some(ash::vk::PrimitiveTopology::PATCH_LIST));
    }

    /// PA_SC_AA_MASK: 16 bits per pixel of a quad; one sample keeps or
    /// removes the whole pixel with bit 0.
    #[test]
    fn quad_sample_masks_per_pixel() {
        assert_eq!(quad_sample_masks([0x0003_0001, 0x000F_0002], 4), [1, 3, 2, 0xF]);
        assert_eq!(quad_sample_masks([0xFFFF_FFFF; 2], 8), [0xFF; 4]);
        assert_eq!(quad_sample_masks([0x0002_0001, 0xFFFF_0000], 1), [1, 0, 0, 1]);
    }

    #[test]
    fn clip_rect_rules() {
        // The reset rule passes everything; intersections of one or more
        // rectangles are recognised, unions are not.
        assert_eq!(clip_rule_intersection(0xFFFF), Some(0));
        assert_eq!(clip_rule_intersection(0xAAAA), Some(1));
        assert_eq!(clip_rule_intersection(0x8888), Some(3));
        assert_eq!(clip_rule_intersection(0x8000), Some(0xF));
        assert_eq!(clip_rule_intersection(0xFFFE), None);
        let mut ctx = vec![0u32; 0x400];
        ctx[cx::PA_SC_SCREEN_SCISSOR_BR as usize] = 0x4000_4000;
        ctx[cx::PA_SC_WINDOW_SCISSOR_TL as usize] = 1 << 31;
        ctx[cx::PA_SC_WINDOW_SCISSOR_BR as usize] = 0x4000_4000;
        ctx[cx::PA_SC_GENERIC_SCISSOR_TL as usize] = 1 << 31;
        ctx[cx::PA_SC_GENERIC_SCISSOR_BR as usize] = 0x4000_4000;
        ctx[cx::PA_SC_CLIPRECT_RULE as usize] = 0xAAAA;
        ctx[cx::PA_SC_CLIPRECT_0_TL as usize] = (1 << 31) | (20 << 16) | 10;
        ctx[cx::PA_SC_CLIPRECT_0_TL as usize + 1] = (200 << 16) | 100;
        let b = RegisterBanks { context: &ctx, shader: &[], uconfig: &[] };
        assert_eq!(resolve_scissor(&b), Rect { x0: 10, y0: 20, x1: 100, y1: 200 });
        ctx[cx::PA_SC_CLIPRECT_RULE as usize] = 0;
        let b = RegisterBanks { context: &ctx, shader: &[], uconfig: &[] };
        assert!(resolve_scissor(&b).is_empty());
    }

    #[test]
    fn polygon_offset_bias() {
        let mut r = RasterState { mode_cntl: 1 << 11, ..RasterState::default() };
        r.poly_offset = [0.5f32.to_bits(), 32.0f32.to_bits(), 2.0f32.to_bits(), 0, 0];
        // Fixed-point guest depth (24 bits) on a D16 host attachment.
        r.poly_offset_db_fmt = (-24i8) as u8 as u32;
        assert_eq!(r.depth_bias(ash::vk::Format::D16_UNORM), Some([2.0 / 256.0, 0.5, 2.0]));
        // Float host attachment and float guest format take the value as is.
        assert_eq!(r.depth_bias(ash::vk::Format::D32_SFLOAT), Some([2.0, 0.5, 2.0]));
        r.poly_offset_db_fmt = 1 << 8;
        assert_eq!(r.depth_bias(ash::vk::Format::D16_UNORM), Some([2.0, 0.5, 2.0]));
        // A culled front face uses the back offset; none enabled: no bias.
        r.cull_front = true;
        assert_eq!(r.depth_bias(ash::vk::Format::D32_SFLOAT), None);
    }

    #[test]
    fn per_face_polygon_offset() {
        let f = ash::vk::Format::D32_SFLOAT;
        let bits = |v: [f32; 5]| v.map(f32::to_bits);
        // Only front faces biased, both rasterised: back faces get none
        // (one host bias for both would move the back faces too).
        let mut r = RasterState { mode_cntl: 1 << 11, poly_offset_db_fmt: 1 << 8, ..RasterState::default() };
        r.poly_offset = bits([0.25, 16.0, -8.0, 0.0, 0.0]);
        assert_eq!(r.face_depth_biases(f), Some(([-8.0, 0.25, 1.0], [0.0, 0.25, 0.0])));
        // Both enabled with different offsets.
        r.mode_cntl = 1 << 11 | 1 << 12;
        r.poly_offset = bits([0.0, 0.0, -8.0, 0.0, 8.0]);
        assert_eq!(r.face_depth_biases(f), Some(([-8.0, 0.0, 0.0], [8.0, 0.0, 0.0])));
        // Equal offsets on both faces: one bias serves the draw.
        r.poly_offset = bits([0.0, 4.0, 2.0, 4.0, 2.0]);
        assert_eq!(r.face_depth_biases(f), None);
        assert_eq!(r.depth_bias(f), Some([2.0, 0.0, 0.25]));
        // One face culled: the visible face's bias alone.
        r.poly_offset = bits([0.0, 0.0, -8.0, 0.0, 8.0]);
        r.cull_back = true;
        assert_eq!(r.face_depth_biases(f), None);
        assert_eq!(r.depth_bias(f), Some([-8.0, 0.0, 0.0]));
        r.cull_back = false;
        r.cull_front = true;
        assert_eq!(r.depth_bias(f), Some([8.0, 0.0, 0.0]));
        // No offset enabled.
        let none = RasterState { poly_offset: bits([0.0, 1.0, 1.0, 2.0, 2.0]), ..RasterState::default() };
        assert_eq!(none.face_depth_biases(f), None);
    }
}
