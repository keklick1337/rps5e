// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! AGC shader-object metadata the planner needs: the user-data layout
//! (`sceAgcCreateShader` header +0x08) and the stage parameters.
//!
//! * Direct resources: one optional user-data SGPR offset per
//!   [`DirectResource`] kind (0xFFFF = absent), e.g. the extended-user-data
//!   pointer or the vertex buffer/attribute tables of a fetch shader.
//! * Sharp resources: per descriptor class (texture, buffer, sampler, and a
//!   fourth class), the dword offset of each statically placed descriptor in
//!   the *flat user-data space*: dwords 0..16 are the user SGPRs, dwords 16..
//!   are the extended user data (EUD) the [`DirectResource::PtrExtendedUserData`]
//!   pointer addresses. Bit 15 marks a 4-dword (V#/S#-sized) slot.

/// Direct resource kinds (index into the direct-offset table).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum DirectResource {
    GdsCounterRange = 0,
    ShaderResourceTable = 1,
    SubPtrFetchShader = 2,
    PtrSoBufferTable = 3,
    PtrInternalGlobalTable = 4,
    PtrExtendedUserData = 5,
    GsFlags = 6,
    GdsMemoryRange = 7,
    PtrVertexBufferTable = 8,
    StereoXOffset = 9,
    PtrVertexAttribDescTable = 10,
}

pub const DIRECT_RESOURCE_COUNT: usize = 11;
pub const ILLEGAL_DIRECT_OFFSET: u16 = 0xFFFF;
/// User SGPR dwords before the extended user data in the flat space.
pub const USER_SGPR_DWORDS: u32 = 16;

/// One statically placed descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SharpSlot {
    /// Dword offset in the flat user-data space.
    pub offset_dw: u32,
    /// Bit 15: a 4-dword slot.
    pub size_flag: bool,
}

impl SharpSlot {
    pub fn from_raw(raw: u16) -> Self {
        SharpSlot { offset_dw: (raw & 0x7FFF) as u32, size_flag: raw & 0x8000 != 0 }
    }

    /// Dwords of the descriptor in this slot (texture slots are 8 unless
    /// flagged).
    pub fn dwords(&self, class: usize) -> u32 {
        if class == 0 && !self.size_flag {
            8
        } else {
            4
        }
    }
}

/// The user-data layout of one shader object.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderUserDataLayout {
    pub direct_offsets: [Option<u16>; DIRECT_RESOURCE_COUNT],
    pub sharp_slots: [Vec<SharpSlot>; 4],
    pub eud_size_dw: u32,
    pub srt_size_dw: u32,
}

impl ShaderUserDataLayout {
    pub fn new(direct_offsets: &[u16], sharp_offsets: [&[u16]; 4], eud_size_dw: u32, srt_size_dw: u32) -> Self {
        let mut layout = ShaderUserDataLayout { eud_size_dw, srt_size_dw, ..Default::default() };
        for (i, &o) in direct_offsets.iter().take(DIRECT_RESOURCE_COUNT).enumerate() {
            layout.direct_offsets[i] = (o != ILLEGAL_DIRECT_OFFSET).then_some(o);
        }
        for (class, offsets) in sharp_offsets.iter().enumerate() {
            layout.sharp_slots[class] = offsets.iter().map(|&o| SharpSlot::from_raw(o)).collect();
        }
        layout
    }

    /// The user-data SGPR offset (relative to the stage's user-data base) of
    /// a direct resource.
    pub fn direct(&self, kind: DirectResource) -> Option<u32> {
        self.direct_offsets[kind as usize].map(u32::from)
    }

    /// The class (0..4) of the sharp slot at flat dword `offset_dw`.
    pub fn sharp_class_at(&self, offset_dw: u32) -> Option<usize> {
        (0..4).find(|&c| self.sharp_slots[c].iter().any(|s| s.offset_dw == offset_dw))
    }
}

/// Shader-object stage types (`sceAgcCreateShader` type byte).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AgcShaderType {
    Compute,
    Pixel,
    Geometry,
    Hull,
    GeometryFront,
    HullFront,
    GeometryBack,
    HullBack,
    Function,
}

impl AgcShaderType {
    pub fn from_raw(raw: u8) -> Option<Self> {
        Some(match raw {
            0 => AgcShaderType::Compute,
            1 => AgcShaderType::Pixel,
            2 => AgcShaderType::Geometry,
            3 => AgcShaderType::Hull,
            4 => AgcShaderType::GeometryFront,
            5 => AgcShaderType::HullFront,
            6 => AgcShaderType::GeometryBack,
            7 => AgcShaderType::HullBack,
            8 => AgcShaderType::Function,
            _ => return None,
        })
    }

    /// The planner stage: geometry/hull stages run the vertex work.
    pub fn stage(self) -> super::ShaderStage {
        match self {
            AgcShaderType::Compute => super::ShaderStage::Compute,
            AgcShaderType::Pixel => super::ShaderStage::Pixel,
            AgcShaderType::Function => super::ShaderStage::Unknown,
            _ => super::ShaderStage::Vertex,
        }
    }

    /// The first user-data SGPR: s8 for the GS/HS (vertex) stage, s0
    /// otherwise.
    pub fn user_data_base(self) -> u32 {
        match self.stage() {
            super::ShaderStage::Vertex => 8,
            _ => 0,
        }
    }

    /// Graphics stages compile as wave32; compute follows the dispatch
    /// (DISPATCH_INITIATOR.CS_W32_EN, bit 15 of the dispatch modifier).
    pub fn wave_size(self, dispatch_modifier: u32) -> u32 {
        match self {
            AgcShaderType::Compute => {
                if dispatch_modifier & (1 << 15) != 0 {
                    32
                } else {
                    64
                }
            }
            _ => 32,
        }
    }

    /// The SH register offset (from 0x2C00) of this stage's PGM_RSRC2.
    pub fn rsrc2_register(self) -> Option<u32> {
        match self {
            AgcShaderType::Compute => Some(0x213),
            AgcShaderType::Pixel => Some(0x0B),
            AgcShaderType::Geometry | AgcShaderType::GeometryFront | AgcShaderType::GeometryBack => Some(0x8B),
            AgcShaderType::Hull | AgcShaderType::HullFront | AgcShaderType::HullBack => Some(0x10B),
            AgcShaderType::Function => None,
        }
    }
}

/// USER_SGPR of a PGM_RSRC2 value (bits 5:1, plus USER_SGPR_MSB bit 27 on
/// graphics stages).
pub fn user_sgpr_count(shader: AgcShaderType, rsrc2: u32) -> u32 {
    let low = (rsrc2 >> 1) & 0x1F;
    match shader {
        AgcShaderType::Compute => low,
        _ => low | ((rsrc2 >> 27) & 1) << 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout() {
        let mut direct = [ILLEGAL_DIRECT_OFFSET; 11];
        direct[5] = 12;
        let layout = ShaderUserDataLayout::new(&direct, [&[0, 0x8008, 32], &[168], &[0x80B8], &[]], 196, 0);
        assert_eq!(layout.direct(DirectResource::PtrExtendedUserData), Some(12));
        assert_eq!(layout.direct(DirectResource::PtrVertexBufferTable), None);
        assert_eq!(layout.sharp_slots[0][1], SharpSlot { offset_dw: 8, size_flag: true });
        assert_eq!(layout.sharp_slots[0][1].dwords(0), 4);
        assert_eq!(layout.sharp_slots[0][0].dwords(0), 8);
        assert_eq!(layout.sharp_class_at(0xB8), Some(2));
        assert_eq!(layout.sharp_class_at(168), Some(1));
        assert_eq!(user_sgpr_count(AgcShaderType::Compute, 68508), 14);
        assert_eq!(user_sgpr_count(AgcShaderType::Pixel, 60), 30);
        assert_eq!(user_sgpr_count(AgcShaderType::Geometry, 196624), 8);
        assert_eq!(AgcShaderType::from_raw(2).unwrap().user_data_base(), 8);
    }
}
