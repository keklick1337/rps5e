// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest resource descriptors (V# buffer, T# image, S# sampler) decoded
//! into the host terms the pipeline provider binds.

use super::descriptors::DummyResources;
use crate::backend::DrawContext;
use crate::device::GpuDevice;
use crate::images::{ImageDim, ImageKey, Plane, ViewRange};
use ash::vk;
use ps5shader::resources::{ImageDimension, ImageNumericClass};

/// A 4-dword buffer descriptor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferSharp(pub [u32; 4]);

impl BufferSharp {
    pub fn from_words(w: &[u32]) -> BufferSharp {
        let mut d = [0u32; 4];
        for (o, i) in d.iter_mut().zip(w) {
            *o = *i;
        }
        BufferSharp(d)
    }
    pub fn address(&self) -> u64 {
        (self.0[0] as u64 | ((self.0[1] as u64) << 32)) & 0xFFFF_FFFF_FFFF
    }
    pub fn stride(&self) -> u32 {
        (self.0[1] >> 16) & 0x3FFF
    }
    pub fn swizzle_enabled(&self) -> bool {
        self.0[1] >> 31 != 0
    }
    pub fn records(&self) -> u32 {
        self.0[2]
    }
    pub fn dst_sel(&self) -> u32 {
        self.0[3] & 0xFFF
    }
    /// Unified FORMAT field.
    pub fn format(&self) -> u32 {
        (self.0[3] >> 12) & 0x7F
    }
    pub fn add_tid(&self) -> bool {
        (self.0[3] >> 23) & 1 != 0
    }
    /// OOB_SELECT (dword 3 bits 28..29).
    pub fn oob_select(&self) -> u32 {
        (self.0[3] >> 28) & 3
    }
    /// Bytes one element of FORMAT spans (a dword for the invalid format
    /// that raw accesses use).
    pub fn element_bytes(&self) -> u32 {
        use ps5shader::resources::format::unified_format;
        unified_format::decode(self.format()).map(|(data, _)| unified_format::access_byte_size(data, 4)).filter(|&b| b != 0).unwrap_or(4)
    }
    /// Bytes the descriptor's accesses can reach from its base: the records
    /// (stride 0: records are bytes). A zero-stride descriptor with
    /// OOB_SELECT 2 checks no range while NUM_RECORDS is nonzero, so its
    /// fetch reads a whole element of its format however few bytes the
    /// records declare (a one-record "null" vertex stream of 32_32_32 reads
    /// 12 bytes): the range covers at least that element.
    pub fn footprint(&self) -> u64 {
        let r = self.records() as u64;
        match self.stride() {
            0 if r != 0 && self.oob_select() == 2 && zero_stride_extent_enabled() => r.max(self.element_bytes() as u64),
            0 => r,
            s => r * s as u64,
        }
    }
    /// Bytes from the base the program's accesses can reach, given the
    /// highest byte (`access_bytes`, immediate offset plus access size) its
    /// instructions read: the records' footprint, except for a stride-0
    /// descriptor without range checking (OOB_SELECT 2, NUM_RECORDS not 0),
    /// whose every access is in range: a 12-byte fetch through a one-record
    /// constant attribute reads 12 bytes of memory.
    pub fn extent(&self, access_bytes: u32) -> u64 {
        let footprint = self.footprint();
        if self.stride() == 0 && self.oob_select() == 2 && self.records() != 0 {
            footprint.max(access_bytes as u64)
        } else {
            footprint
        }
    }
}

/// `RPS5E_ZERO_STRIDE_EXTENT=0`: a zero-stride OOB_SELECT 2 descriptor
/// covers only its NUM_RECORDS bytes (the bytes its fetch reads past them
/// are not copied).
fn zero_stride_extent_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_ZERO_STRIDE_EXTENT").is_ok_and(|v| v == "0"))
}

/// GFX10 image types (T# word 3 bits 31:28).
pub mod image_type {
    pub const TYPE_1D: u32 = 8;
    pub const TYPE_2D: u32 = 9;
    pub const TYPE_3D: u32 = 10;
    pub const CUBE: u32 = 11;
    pub const TYPE_1D_ARRAY: u32 = 12;
    pub const TYPE_2D_ARRAY: u32 = 13;
    pub const TYPE_2D_MSAA: u32 = 14;
    pub const TYPE_2D_MSAA_ARRAY: u32 = 15;
}

/// An 8-dword image descriptor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureSharp(pub [u32; 8]);

impl TextureSharp {
    pub fn from_words(w: &[u32]) -> TextureSharp {
        let mut d = [0u32; 8];
        for (o, i) in d.iter_mut().zip(w) {
            *o = *i;
        }
        TextureSharp(d)
    }
    pub fn address(&self) -> u64 {
        ((self.0[0] as u64 | ((self.0[1] as u64) << 32)) & 0xFF_FFFF_FFFF) << 8
    }
    pub fn format(&self) -> u32 {
        (self.0[1] >> 20) & 0x1FF
    }
    /// MIN_LOD clamp (unsigned 4.8 fixed point).
    pub fn min_lod(&self) -> u32 {
        (self.0[1] >> 8) & 0xFFF
    }
    pub fn width(&self) -> u32 {
        (((self.0[1] >> 30) & 3) | ((self.0[2] & 0xFFF) << 2)) + 1
    }
    pub fn height(&self) -> u32 {
        ((self.0[2] >> 14) & 0x3FFF) + 1
    }
    pub fn dst_sel(&self) -> u32 {
        self.0[3] & 0xFFF
    }
    pub fn base_level(&self) -> u32 {
        (self.0[3] >> 12) & 0xF
    }
    pub fn last_level(&self) -> u32 {
        (self.0[3] >> 16) & 0xF
    }
    /// GFX10 swizzle mode.
    pub fn swizzle(&self) -> u32 {
        (self.0[3] >> 20) & 0x1F
    }
    pub fn ty(&self) -> u32 {
        (self.0[3] >> 28) & 0xF
    }
    /// Depth - 1 (3D) or last array slice.
    pub fn depth(&self) -> u32 {
        (self.0[4] & 0x1FFF) + 1
    }
    pub fn base_array(&self) -> u32 {
        (self.0[4] >> 16) & 0x1FFF
    }
    /// MAX_MIP: the last level the surface holds.
    pub fn max_mip(&self) -> u32 {
        (self.0[5] >> 4) & 0xF
    }
    /// PRT_DEFAULT_COLOR (word 5 bit 26): texels of uncommitted pages of
    /// a partially resident texture read the default colour instead of
    /// zeros. Uncommitted pages load as zeros here, which is the result
    /// with the bit clear.
    pub fn prt_default_color(&self) -> bool {
        (self.0[5] >> 26) & 1 != 0
    }
    pub fn is_msaa(&self) -> bool {
        matches!(self.ty(), image_type::TYPE_2D_MSAA | image_type::TYPE_2D_MSAA_ARRAY)
    }
    /// Samples of a multisampled image (LAST_LEVEL holds log2 of the
    /// count), 1 otherwise.
    pub fn samples(&self) -> u32 {
        if self.is_msaa() {
            1 << self.last_level().min(4)
        } else {
            1
        }
    }
    /// Mip levels the surface holds: MAX_MIP + 1, or through LAST_LEVEL
    /// for views reaching past MAX_MIP.
    pub fn levels(&self) -> u32 {
        if self.is_msaa() {
            return 1;
        }
        if self.base_level() > self.max_mip() {
            self.last_level() + 1
        } else {
            self.max_mip() + 1
        }
    }
    pub fn is_null(&self) -> bool {
        self.address() == 0
    }
    /// Array layers of the surface (arrays and cubes: DEPTH holds the
    /// last slice, a cube's six faces included).
    pub fn layers(&self) -> u32 {
        match self.ty() {
            // A cube whose BASE_ARRAY is its last slice (DEPTH) is the one
            // cube starting there: its six faces end five slices later.
            image_type::CUBE if self.base_array() + 1 == self.depth() => self.depth() + 5,
            image_type::TYPE_1D_ARRAY | image_type::TYPE_2D_ARRAY | image_type::TYPE_2D_MSAA_ARRAY | image_type::CUBE => self.depth(),
            _ => 1,
        }
    }

    /// The host component swizzle of DST_SEL.
    pub fn components(&self) -> vk::ComponentMapping {
        components_of(self.dst_sel())
    }
}

/// Component mapping of a 12-bit DST_SEL (0 = zero, 1 = one, 4..7 = XYZW).
pub fn components_of(dst_sel: u32) -> vk::ComponentMapping {
    let c = |s: u32| match s & 7 {
        0 => vk::ComponentSwizzle::ZERO,
        1 => vk::ComponentSwizzle::ONE,
        4 => vk::ComponentSwizzle::R,
        5 => vk::ComponentSwizzle::G,
        6 => vk::ComponentSwizzle::B,
        7 => vk::ComponentSwizzle::A,
        _ => vk::ComponentSwizzle::ZERO,
    };
    vk::ComponentMapping { r: c(dst_sel), g: c(dst_sel >> 3), b: c(dst_sel >> 6), a: c(dst_sel >> 9) }
}

/// Host format facts of a guest image/texel format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexelFormat {
    pub host: vk::Format,
    /// Bytes per element (per 4x4 block for block-compressed formats).
    pub bytes_per_element: u32,
    pub numeric: ImageNumericClass,
    pub block_compressed: bool,
    /// Bit rearrangement between the stored elements and `host`.
    pub transform: crate::tiler::ColorTransform,
    /// The host channel holding each of the element's X, Y, Z and W.
    pub channels: Channels,
}

/// Where a guest element's X, Y, Z and W land among the host format's R, G,
/// B and A (AMD stores X in the lowest bits of packed 16-bit formats, the
/// Vulkan formats name the highest field first).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Channels {
    #[default]
    Identity,
    /// X in B, Z in R (1_5_5_5 as A1R5G5B5, transformed 10_11_11).
    Bgra,
    /// X in A, Y in B, Z in G, W in R (4_4_4_4 as R4G4B4A4).
    Abgr,
    /// X in A, Y in R, Z in G, W in B (4_4_4_4 as B4G4R4A4).
    Argb,
    /// 5_6_5: W reads X (measured on gfx1035).
    RgbR,
}

impl Channels {
    /// The host channel of guest component `c` (R..A select X..W; constant
    /// selectors pass through).
    pub fn of(self, c: vk::ComponentSwizzle) -> vk::ComponentSwizzle {
        use vk::ComponentSwizzle as S;
        let table = match self {
            Channels::Identity => return c,
            Channels::Bgra => [S::B, S::G, S::R, S::A],
            Channels::Abgr => [S::A, S::B, S::G, S::R],
            Channels::Argb => [S::A, S::R, S::G, S::B],
            Channels::RgbR => [S::R, S::G, S::B, S::R],
        };
        match c {
            S::R => table[0],
            S::G => table[1],
            S::B => table[2],
            S::A => table[3],
            other => other,
        }
    }
}

/// `RPS5E_PACKED_CHANNELS=0`: 5_6_5, 1_5_5_5 and 4_4_4_4 textures are read
/// with the Vulkan channel order (X from the highest field), as before.
fn packed_channels_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_PACKED_CHANNELS").is_ok_and(|v| v == "0"))
}

impl TexelFormat {
    /// The view component mapping of the T# swizzle `dst` (identity when
    /// the shader swizzles itself) for this format: each selected guest
    /// component is read from the host channel holding it (a transformed
    /// 10_11_11 element holds the stored X and Z in host B and R).
    pub fn view_components(&self, dst: vk::ComponentMapping) -> vk::ComponentMapping {
        if self.channels == Channels::Identity {
            return dst;
        }
        let resolve = |c: vk::ComponentSwizzle, own: vk::ComponentSwizzle| self.channels.of(if c == vk::ComponentSwizzle::IDENTITY { own } else { c });
        vk::ComponentMapping {
            r: resolve(dst.r, vk::ComponentSwizzle::R),
            g: resolve(dst.g, vk::ComponentSwizzle::G),
            b: resolve(dst.b, vk::ComponentSwizzle::B),
            a: resolve(dst.a, vk::ComponentSwizzle::A),
        }
    }
}

/// The host format of a unified image FORMAT value.
pub fn texel_format(unified: u32) -> Option<TexelFormat> {
    use vk::Format as F;
    use ImageNumericClass::{Float as Fl, Sint as Si, Uint as Ui};
    let (host, bpe, numeric) = match unified {
        1 => (F::R8_UNORM, 1, Fl),
        2 => (F::R8_SNORM, 1, Fl),
        3 => (F::R8_USCALED, 1, Fl),
        4 => (F::R8_SSCALED, 1, Fl),
        5 => (F::R8_UINT, 1, Ui),
        6 => (F::R8_SINT, 1, Si),
        7 => (F::R16_UNORM, 2, Fl),
        8 => (F::R16_SNORM, 2, Fl),
        9 => (F::R16_USCALED, 2, Fl),
        10 => (F::R16_SSCALED, 2, Fl),
        11 => (F::R16_UINT, 2, Ui),
        12 => (F::R16_SINT, 2, Si),
        13 => (F::R16_SFLOAT, 2, Fl),
        14 => (F::R8G8_UNORM, 2, Fl),
        15 => (F::R8G8_SNORM, 2, Fl),
        16 => (F::R8G8_USCALED, 2, Fl),
        17 => (F::R8G8_SSCALED, 2, Fl),
        18 => (F::R8G8_UINT, 2, Ui),
        19 => (F::R8G8_SINT, 2, Si),
        20 => (F::R32_UINT, 4, Ui),
        21 => (F::R32_SINT, 4, Si),
        22 => (F::R32_SFLOAT, 4, Fl),
        23 => (F::R16G16_UNORM, 4, Fl),
        24 => (F::R16G16_SNORM, 4, Fl),
        25 => (F::R16G16_USCALED, 4, Fl),
        26 => (F::R16G16_SSCALED, 4, Fl),
        27 => (F::R16G16_UINT, 4, Ui),
        28 => (F::R16G16_SINT, 4, Si),
        29 => (F::R16G16_SFLOAT, 4, Fl),
        // 11_11_10 UNORM and UINT: one 32-bit word the shader unpacks and
        // packs (no host format).
        30 | 34 => (F::R32_UINT, 4, Ui),
        36 => (F::B10G11R11_UFLOAT_PACK32, 4, Fl),
        // 10_11_11 float: stored with the 10-bit field low (transformed).
        43 => {
            return Some(TexelFormat {
                host: F::B10G11R11_UFLOAT_PACK32,
                bytes_per_element: 4,
                numeric: Fl,
                block_compressed: false,
                transform: crate::tiler::ColorTransform::Reverse10_11_11,
                channels: Channels::Bgra,
            })
        }
        50 => (F::A2B10G10R10_UNORM_PACK32, 4, Fl),
        51 => (F::A2B10G10R10_SNORM_PACK32, 4, Fl),
        54 => (F::A2B10G10R10_UINT_PACK32, 4, Ui),
        55 => (F::A2B10G10R10_SINT_PACK32, 4, Si),
        56 => (F::R8G8B8A8_UNORM, 4, Fl),
        57 => (F::R8G8B8A8_SNORM, 4, Fl),
        58 => (F::R8G8B8A8_USCALED, 4, Fl),
        59 => (F::R8G8B8A8_SSCALED, 4, Fl),
        60 => (F::R8G8B8A8_UINT, 4, Ui),
        61 => (F::R8G8B8A8_SINT, 4, Si),
        62 => (F::R32G32_UINT, 8, Ui),
        63 => (F::R32G32_SINT, 8, Si),
        64 => (F::R32G32_SFLOAT, 8, Fl),
        65 => (F::R16G16B16A16_UNORM, 8, Fl),
        66 => (F::R16G16B16A16_SNORM, 8, Fl),
        67 => (F::R16G16B16A16_USCALED, 8, Fl),
        68 => (F::R16G16B16A16_SSCALED, 8, Fl),
        69 => (F::R16G16B16A16_UINT, 8, Ui),
        70 => (F::R16G16B16A16_SINT, 8, Si),
        71 => (F::R16G16B16A16_SFLOAT, 8, Fl),
        75 => (F::R32G32B32A32_UINT, 16, Ui),
        76 => (F::R32G32B32A32_SINT, 16, Si),
        77 => (F::R32G32B32A32_SFLOAT, 16, Fl),
        128 => (F::R8_SRGB, 1, Fl),
        129 => (F::R8G8_SRGB, 2, Fl),
        130 => (F::R8G8B8A8_SRGB, 4, Fl),
        132 => (F::E5B9G9R9_UFLOAT_PACK32, 4, Fl),
        // X in the lowest field, as AMD samples them.
        133 | 134 | 136 if packed_channels_enabled() => return Some(packed16(unified)),
        133 => (F::B5G6R5_UNORM_PACK16, 2, Fl),
        134 => (F::R5G5B5A1_UNORM_PACK16, 2, Fl),
        136 => (F::R4G4B4A4_UNORM_PACK16, 2, Fl),
        169 => return bc(F::BC1_RGBA_UNORM_BLOCK, 8),
        170 => return bc(F::BC1_RGBA_SRGB_BLOCK, 8),
        171 => return bc(F::BC2_UNORM_BLOCK, 16),
        172 => return bc(F::BC2_SRGB_BLOCK, 16),
        173 => return bc(F::BC3_UNORM_BLOCK, 16),
        174 => return bc(F::BC3_SRGB_BLOCK, 16),
        175 => return bc(F::BC4_UNORM_BLOCK, 8),
        176 => return bc(F::BC4_SNORM_BLOCK, 8),
        177 => return bc(F::BC5_UNORM_BLOCK, 16),
        178 => return bc(F::BC5_SNORM_BLOCK, 16),
        179 => return bc(F::BC6H_UFLOAT_BLOCK, 16),
        180 => return bc(F::BC6H_SFLOAT_BLOCK, 16),
        181 => return bc(F::BC7_UNORM_BLOCK, 16),
        182 => return bc(F::BC7_SRGB_BLOCK, 16),
        _ => return None,
    };
    Some(TexelFormat { host, bytes_per_element: bpe, numeric, block_compressed: false, transform: Default::default(), channels: Channels::Identity })
}

/// 5_6_5 (133), 1_5_5_5 (134) and 4_4_4_4 (136) with X in the lowest
/// field: the host formats colour targets of the same memory use (4_4_4_4
/// as B4G4R4A4, the 16-bit layout every Vulkan device samples; views of a
/// R4G4B4A4 colour target reinterpret the same bits), and the channels
/// holding X..W in them.
fn packed16(unified: u32) -> TexelFormat {
    let (host, channels) = match unified {
        133 => (vk::Format::B5G6R5_UNORM_PACK16, Channels::RgbR),
        134 => (vk::Format::A1R5G5B5_UNORM_PACK16, Channels::Bgra),
        _ => (vk::Format::B4G4R4A4_UNORM_PACK16, Channels::Argb),
    };
    TexelFormat { host, bytes_per_element: 2, numeric: ImageNumericClass::Float, block_compressed: false, transform: Default::default(), channels }
}

fn bc(host: vk::Format, bpe: u32) -> Option<TexelFormat> {
    Some(TexelFormat { host, bytes_per_element: bpe, numeric: ImageNumericClass::Float, block_compressed: true, transform: Default::default(), channels: Channels::Identity })
}

/// The storage-compatible view format of `f` (sRGB images are written
/// through their UNORM alias).
/// The unsigned view of the same bits a storage image of `f` is accessed
/// through when the program uses it as raw words.
pub fn raw_uint_view(f: vk::Format) -> Option<vk::Format> {
    match f {
        vk::Format::R32_SINT | vk::Format::R32_SFLOAT => Some(vk::Format::R32_UINT),
        vk::Format::R16_SINT | vk::Format::R16_SFLOAT => Some(vk::Format::R16_UINT),
        vk::Format::R8_SINT => Some(vk::Format::R8_UINT),
        vk::Format::R8G8_SINT => Some(vk::Format::R8G8_UINT),
        vk::Format::R16G16_SINT => Some(vk::Format::R16G16_UINT),
        vk::Format::R8G8B8A8_SINT => Some(vk::Format::R8G8B8A8_UINT),
        vk::Format::R16G16B16A16_SINT => Some(vk::Format::R16G16B16A16_UINT),
        vk::Format::R32G32_SINT => Some(vk::Format::R32G32_UINT),
        vk::Format::R32G32B32A32_SINT => Some(vk::Format::R32G32B32A32_UINT),
        _ => None,
    }
}

pub fn storage_view_format(f: vk::Format) -> vk::Format {
    match f {
        vk::Format::R8_SRGB => vk::Format::R8_UNORM,
        vk::Format::R8G8_SRGB => vk::Format::R8G8_UNORM,
        vk::Format::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_UNORM,
        vk::Format::B8G8R8A8_SRGB => vk::Format::B8G8R8A8_UNORM,
        other => other,
    }
}

/// The vertex attribute format of a fixed-function fetch: unified buffer
/// FORMAT narrowed to the `components` the program reads.
pub fn vertex_format(unified: u32, components: u32) -> Option<vk::Format> {
    use vk::Format as F;
    let (data, number) = ps5shader::resources::format::unified_format::decode(unified)?;
    let n = components.clamp(1, 4);
    // (data format, component count of the data format, format per used count)
    let pick = |formats: [F; 4]| -> Option<F> {
        let f = formats[(n - 1) as usize];
        (f != F::UNDEFINED).then_some(f)
    };
    let by_number = |unorm: [F; 4], snorm: [F; 4], uscaled: [F; 4], sscaled: [F; 4], uint: [F; 4], sint: [F; 4], float: [F; 4]| match number {
        0 => pick(unorm),
        1 => pick(snorm),
        2 => pick(uscaled),
        3 => pick(sscaled),
        4 => pick(uint),
        5 => pick(sint),
        7 => pick(float),
        _ => None,
    };
    const U: F = F::UNDEFINED;
    match data {
        // 8
        1 => by_number(
            [F::R8_UNORM, U, U, U],
            [F::R8_SNORM, U, U, U],
            [F::R8_USCALED, U, U, U],
            [F::R8_SSCALED, U, U, U],
            [F::R8_UINT, U, U, U],
            [F::R8_SINT, U, U, U],
            [U; 4],
        ),
        // 16
        2 => by_number(
            [F::R16_UNORM, U, U, U],
            [F::R16_SNORM, U, U, U],
            [F::R16_USCALED, U, U, U],
            [F::R16_SSCALED, U, U, U],
            [F::R16_UINT, U, U, U],
            [F::R16_SINT, U, U, U],
            [F::R16_SFLOAT, U, U, U],
        ),
        // 8_8
        3 => by_number(
            [F::R8_UNORM, F::R8G8_UNORM, U, U],
            [F::R8_SNORM, F::R8G8_SNORM, U, U],
            [F::R8_USCALED, F::R8G8_USCALED, U, U],
            [F::R8_SSCALED, F::R8G8_SSCALED, U, U],
            [F::R8_UINT, F::R8G8_UINT, U, U],
            [F::R8_SINT, F::R8G8_SINT, U, U],
            [U; 4],
        ),
        // 32
        4 => by_number([U; 4], [U; 4], [U; 4], [U; 4], [F::R32_UINT, U, U, U], [F::R32_SINT, U, U, U], [F::R32_SFLOAT, U, U, U]),
        // 16_16
        5 => by_number(
            [F::R16_UNORM, F::R16G16_UNORM, U, U],
            [F::R16_SNORM, F::R16G16_SNORM, U, U],
            [F::R16_USCALED, F::R16G16_USCALED, U, U],
            [F::R16_SSCALED, F::R16G16_SSCALED, U, U],
            [F::R16_UINT, F::R16G16_UINT, U, U],
            [F::R16_SINT, F::R16G16_SINT, U, U],
            [F::R16_SFLOAT, F::R16G16_SFLOAT, U, U],
        ),
        // 10_11_11 / 11_11_10 float
        6 | 7 if number == 7 => Some(F::B10G11R11_UFLOAT_PACK32),
        // 2_10_10_10
        9 => by_number(
            [F::A2B10G10R10_UNORM_PACK32; 4],
            [F::A2B10G10R10_SNORM_PACK32; 4],
            [F::A2B10G10R10_USCALED_PACK32; 4],
            [F::A2B10G10R10_SSCALED_PACK32; 4],
            [F::A2B10G10R10_UINT_PACK32; 4],
            [F::A2B10G10R10_SINT_PACK32; 4],
            [U; 4],
        ),
        // 8_8_8_8
        10 => by_number(
            [F::R8_UNORM, F::R8G8_UNORM, F::R8G8B8A8_UNORM, F::R8G8B8A8_UNORM],
            [F::R8_SNORM, F::R8G8_SNORM, F::R8G8B8A8_SNORM, F::R8G8B8A8_SNORM],
            [F::R8_USCALED, F::R8G8_USCALED, F::R8G8B8A8_USCALED, F::R8G8B8A8_USCALED],
            [F::R8_SSCALED, F::R8G8_SSCALED, F::R8G8B8A8_SSCALED, F::R8G8B8A8_SSCALED],
            [F::R8_UINT, F::R8G8_UINT, F::R8G8B8A8_UINT, F::R8G8B8A8_UINT],
            [F::R8_SINT, F::R8G8_SINT, F::R8G8B8A8_SINT, F::R8G8B8A8_SINT],
            [U; 4],
        ),
        // 32_32
        11 => by_number([U; 4], [U; 4], [U; 4], [U; 4], [F::R32_UINT, F::R32G32_UINT, U, U], [F::R32_SINT, F::R32G32_SINT, U, U], [
            F::R32_SFLOAT,
            F::R32G32_SFLOAT,
            U,
            U,
        ]),
        // 16_16_16_16
        12 => by_number(
            [F::R16_UNORM, F::R16G16_UNORM, F::R16G16B16A16_UNORM, F::R16G16B16A16_UNORM],
            [F::R16_SNORM, F::R16G16_SNORM, F::R16G16B16A16_SNORM, F::R16G16B16A16_SNORM],
            [F::R16_USCALED, F::R16G16_USCALED, F::R16G16B16A16_USCALED, F::R16G16B16A16_USCALED],
            [F::R16_SSCALED, F::R16G16_SSCALED, F::R16G16B16A16_SSCALED, F::R16G16B16A16_SSCALED],
            [F::R16_UINT, F::R16G16_UINT, F::R16G16B16A16_UINT, F::R16G16B16A16_UINT],
            [F::R16_SINT, F::R16G16_SINT, F::R16G16B16A16_SINT, F::R16G16B16A16_SINT],
            [F::R16_SFLOAT, F::R16G16_SFLOAT, F::R16G16B16A16_SFLOAT, F::R16G16B16A16_SFLOAT],
        ),
        // 32_32_32
        13 => by_number([U; 4], [U; 4], [U; 4], [U; 4], [F::R32_UINT, F::R32G32_UINT, F::R32G32B32_UINT, U], [
            F::R32_SINT,
            F::R32G32_SINT,
            F::R32G32B32_SINT,
            U,
        ], [F::R32_SFLOAT, F::R32G32_SFLOAT, F::R32G32B32_SFLOAT, U]),
        // 32_32_32_32
        14 => by_number([U; 4], [U; 4], [U; 4], [U; 4], [F::R32_UINT, F::R32G32_UINT, F::R32G32B32_UINT, F::R32G32B32A32_UINT], [
            F::R32_SINT,
            F::R32G32_SINT,
            F::R32G32B32_SINT,
            F::R32G32B32A32_SINT,
        ], [F::R32_SFLOAT, F::R32G32_SFLOAT, F::R32G32B32_SFLOAT, F::R32G32B32A32_SFLOAT]),
        _ => None,
    }
}

/// A 4-dword sampler descriptor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SamplerSharp(pub [u32; 4]);

/// The host sampler a S# describes (hashable cache key).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SamplerKey {
    pub address: [i32; 3],
    pub mag_linear: bool,
    pub min_linear: bool,
    pub mip: i32,
    pub anisotropy: u32,
    pub compare: Option<i32>,
    pub unnormalized: bool,
    pub min_lod: u32,
    pub max_lod: u32,
    pub lod_bias: i32,
    pub border: i32,
    /// FILTER_MODE (word 0 bits 30:29): 0 blend, 1 minimum, 2 maximum.
    pub reduction: u8,
}

impl SamplerSharp {
    pub fn from_words(w: &[u32]) -> SamplerSharp {
        let mut d = [0u32; 4];
        for (o, i) in d.iter_mut().zip(w) {
            *o = *i;
        }
        SamplerSharp(d)
    }

    /// The S# as used with images of one mip level: ANISO_OVERRIDE (word 2
    /// bit 29) turns its anisotropic XY filters into the plain point or
    /// bilinear ones for such images.
    pub fn for_single_level(self) -> SamplerSharp {
        let mut w = self.0;
        let aniso = |f: u32| f & 2 != 0;
        if w[2] & (1 << 29) != 0 && (aniso((w[2] >> 20) & 3) || aniso((w[2] >> 22) & 3)) {
            w[2] &= !((2 << 20) | (2 << 22));
        }
        SamplerSharp(w)
    }

    /// The host sampler; `depth_compare` keeps the compare function,
    /// `integer` (an integer view is sampled) forces nearest filtering.
    pub fn key(&self, depth_compare: bool, integer: bool, force_point: bool) -> SamplerKey {
        let w = self.0;
        let wrap = |v: u32| -> i32 {
            use vk::SamplerAddressMode as A;
            match v & 7 {
                0 => A::REPEAT,
                1 => A::MIRRORED_REPEAT,
                2 => A::CLAMP_TO_EDGE,
                3 => A::MIRROR_CLAMP_TO_EDGE,
                4 | 6 => A::CLAMP_TO_BORDER,
                // Mirror-once to half/full border: closest host mode.
                _ => A::MIRROR_CLAMP_TO_EDGE,
            }
            .as_raw()
        };
        let point = integer || force_point;
        let xy_mag = (w[2] >> 20) & 3;
        let xy_min = (w[2] >> 22) & 3;
        let mip_filter = (w[2] >> 26) & 3;
        let aniso_ratio = (w[0] >> 9) & 7;
        let anisotropic = !point && aniso_ratio != 0 && (xy_mag >= 2 || xy_min >= 2);
        let unnormalized = (w[0] >> 15) & 1 != 0;
        let border = match (w[3] >> 30) & 3 {
            0 => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            1 => vk::BorderColor::FLOAT_OPAQUE_BLACK,
            2 => vk::BorderColor::FLOAT_OPAQUE_WHITE,
            // Register border colours (palette) are not modelled.
            _ => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
        };
        let border = if integer {
            match border {
                vk::BorderColor::FLOAT_OPAQUE_BLACK => vk::BorderColor::INT_OPAQUE_BLACK,
                vk::BorderColor::FLOAT_OPAQUE_WHITE => vk::BorderColor::INT_OPAQUE_WHITE,
                _ => vk::BorderColor::INT_TRANSPARENT_BLACK,
            }
        } else {
            border
        };
        // DEPTH_COMPARE_FUNC NEVER (0) is a compare too: the shader's
        // depth-compare sample needs a compare sampler.
        let compare = depth_compare.then(|| ((w[0] >> 12) & 7) as i32);
        SamplerKey {
            address: [wrap(w[0]), wrap(w[0] >> 3), wrap(w[0] >> 6)],
            mag_linear: !point && matches!(xy_mag, 1 | 3),
            min_linear: !point && matches!(xy_min, 1 | 3),
            // Point sampling (integer views) keeps the mip selection, with
            // nearest mip filtering.
            mip: if unnormalized || (point && mip_filter == 0) {
                -1
            } else if point {
                vk::SamplerMipmapMode::NEAREST.as_raw()
            } else {
                match mip_filter {
                    1 => vk::SamplerMipmapMode::NEAREST.as_raw(),
                    2 => vk::SamplerMipmapMode::LINEAR.as_raw(),
                    _ => -1,
                }
            },
            anisotropy: if anisotropic { 1 << aniso_ratio.min(4) } else { 0 },
            compare,
            unnormalized,
            min_lod: w[1] & 0xFFF,
            max_lod: (w[1] >> 12) & 0xFFF,
            lod_bias: (((w[2] & 0x3FFF) << 18) as i32) >> 18,
            border: border.as_raw(),
            reduction: ((w[0] >> 29) & 3) as u8,
        }
    }
}

impl SamplerKey {
    /// The create info (`max_anisotropy` clamped by the caller's limit).
    pub fn create_info(&self, max_anisotropy: f32, anisotropy_supported: bool) -> vk::SamplerCreateInfo<'static> {
        let filter = |linear: bool| if linear { vk::Filter::LINEAR } else { vk::Filter::NEAREST };
        let (mip_mode, max_lod) = if self.mip < 0 {
            // No mip filtering: sample the base level only.
            (vk::SamplerMipmapMode::NEAREST, 0.25)
        } else {
            (vk::SamplerMipmapMode::from_raw(self.mip), self.max_lod as f32 / 256.0)
        };
        let min_lod = if self.mip < 0 { 0.0 } else { (self.min_lod as f32 / 256.0).min(max_lod) };
        let mut info = vk::SamplerCreateInfo::default()
            .mag_filter(filter(self.mag_linear))
            .min_filter(filter(self.min_linear))
            .mipmap_mode(mip_mode)
            .address_mode_u(vk::SamplerAddressMode::from_raw(self.address[0]))
            .address_mode_v(vk::SamplerAddressMode::from_raw(self.address[1]))
            .address_mode_w(vk::SamplerAddressMode::from_raw(self.address[2]))
            .mip_lod_bias((self.lod_bias as f32 / 256.0).clamp(-15.0, 15.0))
            .min_lod(min_lod)
            .max_lod(max_lod)
            .border_color(vk::BorderColor::from_raw(self.border));
        if self.anisotropy > 1 && anisotropy_supported && !self.unnormalized {
            info = info.anisotropy_enable(true).max_anisotropy((self.anisotropy as f32).min(max_anisotropy));
        }
        if let Some(c) = self.compare {
            info = info.compare_enable(true).compare_op(vk::CompareOp::from_raw(c));
        }
        if self.unnormalized {
            // Unnormalized coordinates: nearest/linear only, clamp modes, one level.
            let clamp = |m: i32| {
                if m == vk::SamplerAddressMode::CLAMP_TO_BORDER.as_raw() {
                    vk::SamplerAddressMode::CLAMP_TO_BORDER
                } else {
                    vk::SamplerAddressMode::CLAMP_TO_EDGE
                }
            };
            info = info
                .unnormalized_coordinates(true)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .min_filter(info.mag_filter)
                .address_mode_u(clamp(self.address[0]))
                .address_mode_v(clamp(self.address[1]))
                .address_mode_w(clamp(self.address[2]))
                .min_lod(0.0)
                .max_lod(0.0)
                .mip_lod_bias(0.0)
                .compare_enable(false)
                .anisotropy_enable(false);
        }
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FORCE_DEGAMMA leaves formats without an sRGB decoding as they are
    /// (reference execution test ImageSampleForceDegamma: 32_FLOAT and
    /// BC4_UNORM sample the same with and without it).
    #[test]
    fn force_degamma_keeps_formats_without_srgb() {
        for f in [vk::Format::R32_SFLOAT, vk::Format::BC4_UNORM_BLOCK, vk::Format::BC5_UNORM_BLOCK, vk::Format::R16G16B16A16_SFLOAT, vk::Format::R8G8B8A8_UINT] {
            assert_eq!(srgb_twin(f), None, "{f:?}");
        }
        assert_eq!(srgb_twin(vk::Format::R8G8B8A8_UNORM), Some(vk::Format::R8G8B8A8_SRGB));
    }

    /// A sampled read of an array builds the slices it views at their own
    /// address; a 2D read of an array builds its base slice (reference
    /// TextureTiling.cpp ViewedArrayWindow, ShaderResources.cpp
    /// firstLayerWords).
    #[test]
    fn sampled_array_views_build_their_viewed_slices() {
        let (w, h, addr) = (256u32, 128u32, 0x10_0000_0000u64);
        let words = |ty: u32, last: u32, base: u32| {
            [
                (addr >> 8) as u32,
                ((addr >> 40) as u32 & 0xFF) | (56 << 20) | (((w - 1) & 3) << 30),
                ((w - 1) >> 2) | ((h - 1) << 14),
                0xFAC | (9 << 20) | (ty << 28),
                last | (base << 16),
                0,
                0,
                0,
            ]
        };
        let fmt = texel_format(56).unwrap();
        let req = |dimension: ImageDimension, cube: bool| TextureRequest {
            numeric: fmt.numeric,
            dimension,
            storage: false,
            cube,
            host_swizzle: true,
            mip_offset: 0,
            single_level: false,
            atomic64: false,
            srgb: false,
        };
        let window = |t: &TextureSharp, r: &TextureRequest| {
            let key = texture_key(t, &fmt).unwrap();
            sampled_array_window(t, &key, r, false, view_type(r.dimension, r.cube)).map(|k| (k.main.addr - addr, k.layers, key.main.slice_size))
        };
        let array = req(ImageDimension::Dim2DArray, false);
        // Slices 5..15 of a 16-slice array: 11 slices from slice 5.
        let t = TextureSharp::from_words(&words(image_type::TYPE_2D_ARRAY, 15, 5));
        let (offset, layers, slice) = window(&t, &array).expect("windowed");
        assert_eq!((offset, layers), (5 * slice, 11));
        // The whole array from slice 0 stays whole.
        assert_eq!(window(&TextureSharp::from_words(&words(image_type::TYPE_2D_ARRAY, 15, 0)), &array), None);
        // A 2D read of the array: its base slice alone.
        let (offset, layers, slice) = window(&t, &req(ImageDimension::Dim2D, false)).expect("base slice");
        assert_eq!((offset, layers), (5 * slice, 1));
        // A cube at slice 6 of a cube array (DEPTH = its first slice).
        let cube = TextureSharp::from_words(&words(image_type::CUBE, 6, 6));
        let (offset, layers, slice) = window(&cube, &req(ImageDimension::Dim2DArray, true)).expect("one cube");
        assert_eq!((offset, layers), (6 * slice, 6));
        // Storage, DCC and single-slice surfaces are not windowed.
        assert_eq!(window(&t, &TextureRequest { storage: true, ..array }), None);
        let key = texture_key(&t, &fmt).unwrap();
        assert_eq!(sampled_array_window(&t, &key, &array, true, vk::ImageViewType::TYPE_2D_ARRAY), None);
        assert_eq!(window(&TextureSharp::from_words(&words(image_type::TYPE_2D, 0, 0)), &req(ImageDimension::Dim2D, false)), None);
    }

    /// 32-bit signed stores and float atomics reach their image through a
    /// UINT view of the same bits (they were bound to a dummy image and
    /// lost); other formats have none.
    #[test]
    fn raw_word_storage_views_are_uint() {
        assert_eq!(raw_uint_view(vk::Format::R32_SINT), Some(vk::Format::R32_UINT));
        assert_eq!(raw_uint_view(vk::Format::R32_SFLOAT), Some(vk::Format::R32_UINT));
        assert_eq!(raw_uint_view(vk::Format::R16_SINT), Some(vk::Format::R16_UINT));
        assert_eq!(raw_uint_view(vk::Format::R32G32B32A32_SINT), Some(vk::Format::R32G32B32A32_UINT));
        assert_eq!(raw_uint_view(vk::Format::R8G8B8A8_UNORM), None);
        assert_eq!(raw_uint_view(vk::Format::R32G32_SFLOAT), None);
    }

    #[test]
    fn texture_fields() {
        // 2D RGBA8 300x200 at 0x12_3456_7800, 64KB_R_X.
        let addr = 0x12_3456_7800u64;
        let w = 300u32 - 1;
        let h = 200u32 - 1;
        let t = TextureSharp([
            (addr >> 8) as u32,
            ((addr >> 40) as u32 & 0xFF) | (56 << 20) | ((w & 3) << 30),
            (w >> 2) | (h << 14),
            0xFAC | (27 << 20) | (image_type::TYPE_2D << 28),
            0,
            0,
            0,
            0,
        ]);
        assert_eq!(t.address(), addr);
        assert_eq!((t.width(), t.height()), (300, 200));
        assert_eq!(t.format(), 56);
        assert_eq!(t.swizzle(), 27);
        assert_eq!(t.ty(), image_type::TYPE_2D);
        let c = t.components();
        assert_eq!([c.r, c.g, c.b, c.a], [vk::ComponentSwizzle::R, vk::ComponentSwizzle::G, vk::ComponentSwizzle::B, vk::ComponentSwizzle::A]);
        assert_eq!(texel_format(56).unwrap().host, vk::Format::R8G8B8A8_UNORM);
        // 10_11_11 float is transformed and viewed with R and B swapped.
        let f = texel_format(43).unwrap();
        assert_eq!((f.host, f.transform), (vk::Format::B10G11R11_UFLOAT_PACK32, crate::tiler::ColorTransform::Reverse10_11_11));
        assert_eq!(texel_format(36).unwrap().transform, crate::tiler::ColorTransform::None);
        use vk::ComponentSwizzle as S;
        let id = f.view_components(vk::ComponentMapping::default());
        assert_eq!((id.r, id.g, id.b, id.a), (S::B, S::G, S::R, S::A));
        let m = f.view_components(vk::ComponentMapping { r: S::B, g: S::ONE, b: S::R, a: S::IDENTITY });
        assert_eq!((m.r, m.g, m.b, m.a), (S::R, S::ONE, S::B, S::A));
        let plain = texel_format(56).unwrap().view_components(vk::ComponentMapping { r: S::B, g: S::G, b: S::R, a: S::A });
        assert_eq!((plain.r, plain.b), (S::B, S::R));

        assert!(texel_format(181).unwrap().block_compressed);
    }

    /// 5_6_5, 1_5_5_5 and 4_4_4_4 read X, Y, Z and W from the bits AMD
    /// stores them in (X lowest; W of 5_6_5 reads X, as measured on
    /// gfx1035), through the host format's channel layout.
    #[test]
    fn packed_16_bit_channels_follow_amd_order() {
        use vk::ComponentSwizzle as S;
        // Lowest bit of each host channel (None: the format has none).
        let low_bit = |f: vk::Format, c: S| -> Option<u32> {
            let i = [S::R, S::G, S::B, S::A].iter().position(|&x| x == c)?;
            match f {
                vk::Format::B5G6R5_UNORM_PACK16 => [Some(0), Some(5), Some(11), None][i],
                vk::Format::A1R5G5B5_UNORM_PACK16 => [Some(10), Some(5), Some(0), Some(15)][i],
                vk::Format::R4G4B4A4_UNORM_PACK16 => [Some(12), Some(8), Some(4), Some(0)][i],
                vk::Format::B4G4R4A4_UNORM_PACK16 => [Some(4), Some(8), Some(12), Some(0)][i],
                other => panic!("no layout for {other:?}"),
            }
        };
        for (format, bits) in [(133u32, [0, 5, 11, 0]), (134, [0, 5, 10, 15]), (136, [0, 4, 8, 12])] {
            let f = packed16(format);
            let m = f.view_components(vk::ComponentMapping::default());
            for (component, (c, bit)) in [m.r, m.g, m.b, m.a].into_iter().zip(bits).enumerate() {
                assert_eq!(low_bit(f.host, c), Some(bit), "format {format} component {component}");
            }
            // A T# swizzle selects guest components, which move with them.
            let swapped = f.view_components(vk::ComponentMapping { r: S::B, g: S::ONE, b: S::R, a: S::ZERO });
            assert_eq!((low_bit(f.host, swapped.r), swapped.g, low_bit(f.host, swapped.b), swapped.a), (Some(bits[2]), S::ONE, Some(bits[0]), S::ZERO), "format {format}");
        }
        assert_eq!(texel_format(56).unwrap().channels, Channels::Identity);
        assert_eq!(packed16(136).host, vk::Format::B4G4R4A4_UNORM_PACK16, "4_4_4_4 samples as B4G4R4A4");
    }

    /// The materializer swaps X and Z of storage accesses exactly for the
    /// formats whose host image reverses them; 11_11_10 UNORM and UINT
    /// are one 32-bit word.
    #[test]
    fn storage_channel_order_matches_the_host_formats() {
        use ps5shader::resources::format::image_format;
        for f in 0..=image_format::MAX_FORMAT {
            if f == 134 {
                // 5_5_5_1 reverses only with the packed-channel views.
                continue;
            }
            let reversed = texel_format(f).is_some_and(|t| t.channels == Channels::Bgra);
            assert_eq!(reversed, image_format::host_reverses_channels(f), "format {f}");
        }
        for f in [image_format::FORMAT_11X2X10_UNORM, image_format::FORMAT_11X2X10_UINT] {
            let t = texel_format(f).unwrap();
            assert_eq!((t.host, t.numeric), (vk::Format::R32_UINT, ImageNumericClass::Uint));
        }
    }

    #[test]
    fn buffer_fields_and_vertex_formats() {
        let v = BufferSharp([0x1000, 0x12 | (16 << 16), 10, (77 << 12) | 0xFAC]);
        assert_eq!(v.address(), 0x12_0000_1000);
        assert_eq!(v.stride(), 16);
        assert_eq!(v.footprint(), 160);
        assert_eq!(v.format(), 77);
        assert_eq!(vertex_format(77, 4), Some(vk::Format::R32G32B32A32_SFLOAT));
        assert_eq!(vertex_format(77, 2), Some(vk::Format::R32G32_SFLOAT));
        assert_eq!(vertex_format(56, 4), Some(vk::Format::R8G8B8A8_UNORM));
        assert_eq!(vertex_format(29, 2), Some(vk::Format::R16G16_SFLOAT));
    }

    /// The one-byte "null" vertex stream skinned meshes bind for absent
    /// offset attributes (SILENT HILL menu): stride 0, NUM_RECORDS 1,
    /// 32_32_32_FLOAT, OOB_SELECT 2. The hardware reads the whole 12-byte
    /// element (OOB_SELECT 2 checks only that NUM_RECORDS is nonzero), so
    /// the bound range must cover it; covering one byte left the other 11
    /// to whatever the copy's staging slot held before.
    #[test]
    fn zero_stride_unchecked_descriptor_covers_its_element() {
        let null_stream = BufferSharp([0x00e3_8db0, 0x0000_0010, 1, 0x2004_a3ac]);
        assert_eq!((null_stream.stride(), null_stream.records(), null_stream.oob_select(), null_stream.format()), (0, 1, 2, 74));
        assert_eq!(null_stream.element_bytes(), 12);
        assert_eq!(null_stream.footprint(), 12);
        // 32_32_32_32: the whole 16-byte element.
        assert_eq!(BufferSharp([0x1000, 0, 1, (77 << 12) | (2 << 28)]).footprint(), 16);
        // More records than one element: the records.
        assert_eq!(BufferSharp([0x1000, 0, 64, (74 << 12) | (2 << 28)]).footprint(), 64);
        // No records: every access reads 0, nothing to cover.
        assert_eq!(BufferSharp([0x1000, 0, 0, (74 << 12) | (2 << 28)]).footprint(), 0);
        // The raw (OOB_SELECT 3) and structured modes check their bytes
        // against NUM_RECORDS: only those bytes are reachable.
        assert_eq!(BufferSharp([0x1000, 0, 1, (74 << 12) | (3 << 28)]).footprint(), 1);
        assert_eq!(BufferSharp([0x1000, 0, 1, 74 << 12]).footprint(), 1);
        // The invalid format of raw accesses: one dword.
        assert_eq!(BufferSharp([0x1000, 0, 1, 2 << 28]).footprint(), 4);
        // Strided descriptors are unchanged.
        assert_eq!(BufferSharp([0x1000, 12 << 16, 3, (74 << 12) | (2 << 28)]).footprint(), 36);
    }

    /// A stride-0, one-record V# without range checking (a constant vertex
    /// attribute of a SILENT HILL skinned mesh, 32_32_32_FLOAT xyz1): every
    /// access is in range, so a 12-byte fetch reaches 12 bytes past the
    /// base, not the one byte NUM_RECORDS names.
    #[test]
    fn unchecked_zero_stride_extent_covers_the_access() {
        let constant = BufferSharp([0x00e3_8db0, 0x0000_0010, 1, 0x2004_a3ac]);
        assert_eq!((constant.stride(), constant.records(), constant.oob_select(), constant.format()), (0, 1, 2, 74));
        assert_eq!(constant.footprint(), 12);
        assert_eq!(constant.extent(12), 12);
        // NUM_RECORDS 0 is out of range under every OOB_SELECT.
        assert_eq!(BufferSharp([0x00e3_8db0, 0x10, 0, 0x2004_a3ac]).extent(12), 0);
        // Checked selects and strided descriptors keep their footprint.
        for oob in [0u32, 1, 3] {
            assert_eq!(BufferSharp([0x00e3_8db0, 0x10, 1, (oob << 28) | 0x4a3ac]).extent(12), 1, "OOB_SELECT {oob}");
        }
        let strided = BufferSharp([0x7795_0000, 0x0008_0020, 0x3d80b, 0x2001_d22c]);
        assert_eq!(strided.extent(4), 0x3d80b * 8);
        // A raw buffer read past its records keeps its records.
        assert_eq!(BufferSharp([0x1000, 0x10, 64, 0x2001_4204]).extent(16), 64);
    }

    #[test]
    fn sampler_keys() {
        // Clamp-to-edge, bilinear, linear mips.
        let s = SamplerSharp([2 | (2 << 3) | (2 << 6), 0xFFF << 12, (1 << 20) | (1 << 22) | (2 << 26), 0]);
        let k = s.key(false, false, false);
        assert!(k.mag_linear && k.min_linear);
        assert_eq!(k.mip, vk::SamplerMipmapMode::LINEAR.as_raw());
        assert_eq!(k.address[0], vk::SamplerAddressMode::CLAMP_TO_EDGE.as_raw());
        let p = s.key(false, true, false);
        assert!(!p.mag_linear && !p.min_linear);
        assert_eq!(p.border, vk::BorderColor::INT_TRANSPARENT_BLACK.as_raw());
        // Negative LOD bias sign-extends.
        let b = SamplerSharp([0, 0, 0x3F00, 0]).key(false, false, false);
        assert_eq!(b.lod_bias, -256);
    }

    #[test]
    fn aniso_override_on_single_level_images() {
        // Clamp, 16x anisotropic linear filters, all LODs (the reference
        // driver's aniso override test sampler), with ANISO_OVERRIDE.
        let words = [0x92 | (4 << 9), 0x00FF_F000, (3 << 20) | (3 << 22), 0];
        let plain = SamplerSharp(words);
        assert_eq!(plain.key(false, false, false).anisotropy, 16);
        assert_eq!(plain.for_single_level(), plain, "no override: the filters stay anisotropic");
        let overridden = SamplerSharp([words[0], words[1], words[2] | (1 << 29), words[3]]);
        assert_eq!(overridden.key(false, false, false).anisotropy, 16, "mipmapped images keep anisotropy");
        let single = overridden.for_single_level().key(false, false, false);
        assert_eq!(single.anisotropy, 0, "single-level images drop anisotropy");
        assert!(single.mag_linear && single.min_linear, "anisotropic linear becomes bilinear");
        // Anisotropic point (2) becomes point (0).
        let point = SamplerSharp([words[0], words[1], (2 << 20) | (2 << 22) | (1 << 29), 0]).for_single_level();
        assert_eq!((point.0[2] >> 20) & 0xF, 0);
    }
}

// ----------------------------------------------------------------------
// T# -> image view
// ----------------------------------------------------------------------

/// FMASK formats: `(samples, fragments)` of the sample-to-fragment map.
pub fn fmask_shape(unified: u32) -> Option<(u32, u32)> {
    Some(match unified {
        156 => (2, 1),
        157 => (4, 1),
        158 => (8, 1),
        159 => (2, 2),
        160 => (4, 2),
        161 => (4, 4),
        162 => (16, 1),
        163 => (8, 2),
        164 => (16, 2),
        165 => (8, 4),
        166 => (8, 8),
        167 => (16, 4),
        168 => (16, 8),
        _ => return None,
    })
}

/// The FMASK value of an expanded (fully decompressed) surface as the
/// texture unit returns it: 4 bits per sample, sample `i` in fragment
/// `i mod fragments`.
pub fn fmask_identity(samples: u32, fragments: u32) -> u32 {
    (0..samples.min(8)).fold(0, |v, i| v | ((i % fragments.max(1)) << (4 * i)))
}

fn creation_error(key: &ImageKey, e: vk::Result) -> String {
    format!("image creation {e:?} ({:?} {:?} {}x{}x{} [{}] {} levels, {} samples)", key.format, key.dim, key.width, key.height, key.depth, key.layers, key.levels, key.samples)
}

/// How a shader binding views a T#.
#[derive(Clone, Copy, Debug)]
pub struct TextureRequest {
    pub numeric: ImageNumericClass,
    pub dimension: ImageDimension,
    pub storage: bool,
    /// The binding is a cube (array) sampled through a direction.
    pub cube: bool,
    /// Apply DST_SEL through the view's component mapping.
    pub host_swizzle: bool,
    /// Level offset of a dynamic-mip element.
    pub mip_offset: u32,
    /// One level per view, no MIN_LOD clamp (the shader selects the mip
    /// and applies the minimum LOD itself).
    pub single_level: bool,
    /// An R64 unsigned view of a 32_32 unsigned image (64-bit atomics).
    pub atomic64: bool,
    /// A sampler with FORCE_DEGAMMA filters the image: a non-sRGB format
    /// with an sRGB twin is read through the twin.
    pub srgb: bool,
}

/// The sRGB format of the same bits as `f` (None when there is none).
pub fn srgb_twin(f: vk::Format) -> Option<vk::Format> {
    use vk::Format as F;
    Some(match f {
        F::R8_UNORM => F::R8_SRGB,
        F::R8G8_UNORM => F::R8G8_SRGB,
        F::R8G8B8A8_UNORM => F::R8G8B8A8_SRGB,
        F::B8G8R8A8_UNORM => F::B8G8R8A8_SRGB,
        F::A8B8G8R8_UNORM_PACK32 => F::A8B8G8R8_SRGB_PACK32,
        F::BC1_RGB_UNORM_BLOCK => F::BC1_RGB_SRGB_BLOCK,
        F::BC1_RGBA_UNORM_BLOCK => F::BC1_RGBA_SRGB_BLOCK,
        F::BC2_UNORM_BLOCK => F::BC2_SRGB_BLOCK,
        F::BC3_UNORM_BLOCK => F::BC3_SRGB_BLOCK,
        F::BC7_UNORM_BLOCK => F::BC7_SRGB_BLOCK,
        _ => return None,
    })
}

/// The view type a binding of `dimension` declares.
pub fn view_type(dimension: ImageDimension, cube: bool) -> vk::ImageViewType {
    match dimension {
        ImageDimension::Dim1D => vk::ImageViewType::TYPE_1D,
        ImageDimension::Dim1DArray => vk::ImageViewType::TYPE_1D_ARRAY,
        ImageDimension::Dim3D => vk::ImageViewType::TYPE_3D,
        ImageDimension::Dim2DArray if cube => vk::ImageViewType::CUBE_ARRAY,
        ImageDimension::Dim2DArray | ImageDimension::Dim2DMsaaArray => vk::ImageViewType::TYPE_2D_ARRAY,
        _ => vk::ImageViewType::TYPE_2D,
    }
}

/// The cached-image key of the surface a T# describes (its whole extent,
/// layers and mip chain).
pub fn texture_key(t: &TextureSharp, fmt: &TexelFormat) -> Result<ImageKey, String> {
    let dim = match t.ty() {
        image_type::TYPE_1D | image_type::TYPE_1D_ARRAY => ImageDim::D1,
        image_type::TYPE_3D => ImageDim::D3,
        image_type::TYPE_2D | image_type::TYPE_2D_ARRAY | image_type::CUBE | image_type::TYPE_2D_MSAA | image_type::TYPE_2D_MSAA_ARRAY => ImageDim::D2,
        other => return Err(format!("image type {other}")),
    };
    let width = t.width();
    let height = if dim == ImageDim::D1 { 1 } else { t.height() };
    let depth = if dim == ImageDim::D3 { t.depth() } else { 1 };
    let mut key = ImageKey {
        format: fmt.host,
        width,
        height,
        depth,
        layers: t.layers().max(1),
        levels: t.levels().clamp(1, crate::images::max_levels(width, height, depth)),
        samples: t.samples(),
        dim,
        main: Plane { addr: 0, bytes_per_element: 0, swizzle: 0, pitch: 0, slice_size: 0, bytes: 0, aspect: vk::ImageAspectFlags::COLOR },
        stencil: None,
        transform: fmt.transform,
    };
    if crate::tiling::pipe_bank_xor(t.address(), t.swizzle()) != 0 {
        // The low address bits of an XOR swizzle surface inside its block
        // are a pipe/bank XOR applied to every in-block offset; the
        // detiler takes the address as the surface start.
        crate::tiling::report_pipe_bank_xor("T#", t.address(), t.swizzle());
    }
    key.main = crate::images::plane(&key, t.address(), fmt.bytes_per_element, t.swizzle(), vk::ImageAspectFlags::COLOR);
    if key.samples == 1 && key.layout(&key.main).is_none() {
        return Err(format!("image layout (swizzle mode {}, {:?}, {} bytes per element)", t.swizzle(), dim, fmt.bytes_per_element));
    }
    Ok(key)
}

/// `RPS5E_ARRAY_WINDOW=0`: a sampled view of an array T# is built from the
/// whole array, slice 0 on (the earlier behaviour).
fn array_window_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_ARRAY_WINDOW").is_ok_and(|v| v == "0"))
}

/// The image a sampled read of an array T# needs: the slices it views,
/// as an array of their own at the first viewed slice, instead of the
/// whole array from slice 0 (a view of six faces at slice 7227 of a cube
/// array, or one slice at 1536 of a 3328x1872 array, would otherwise build
/// gigabytes for megabytes). A read declared 2D (or 1D) views the base
/// slice alone. Only where the viewed slices do not depend on their index:
/// thin single-sample layers of one size without DCC metadata (the thin
/// swizzles are laid out per slice alike here). None: the whole surface.
pub fn sampled_array_window(t: &TextureSharp, key: &ImageKey, req: &TextureRequest, dcc: bool, ty: vk::ImageViewType) -> Option<ImageKey> {
    if req.storage || dcc || key.dim == ImageDim::D3 || key.samples != 1 || key.layers <= 1 || !array_window_enabled() {
        return None;
    }
    let first = t.base_array();
    if first >= key.layers {
        return None;
    }
    let count = if matches!(ty, vk::ImageViewType::TYPE_2D | vk::ImageViewType::TYPE_1D) { 1 } else { key.layers - first };
    if first == 0 && count == key.layers {
        return None;
    }
    let mut w = *key;
    w.layers = count;
    w.main = crate::images::plane(&w, key.main.addr + first as u64 * key.main.slice_size, key.main.bytes_per_element, key.main.swizzle, key.main.aspect);
    (w.main.slice_size == key.main.slice_size && w.main.pitch == key.main.pitch).then_some(w)
}

/// The host view of a T# for `req`: the cached image is found (an alias
/// of a render target or another view), created and uploaded, or
/// converted (a storage view of a depth surface works on a colour copy).
/// FMASK descriptors read as an expanded surface. `Err` says why a dummy
/// must be bound instead.
pub fn texture_view(ctx: &mut DrawContext, dummies: Option<&mut DummyResources>, dev: &GpuDevice, words: &[u32], req: &TextureRequest) -> Result<(vk::ImageView, Option<u64>), String> {
    let t = TextureSharp::from_words(words);
    if t.is_null() {
        return Err("null T#".into());
    }
    if t.prt_default_color() {
        crate::log_once(&format!("ps5gpu: T# {:#x} asks for the PRT default colour (uncommitted pages read as zeros)", t.address()));
    }
    let ty = view_type(req.dimension, req.cube);
    if let Some((samples, fragments)) = fmask_shape(t.format()) {
        if req.storage || req.numeric != ImageNumericClass::Uint {
            return Err(format!("FMASK bound as {:?}", req.numeric));
        }
        let layers = if matches!(ty, vk::ImageViewType::TYPE_2D_ARRAY) { t.layers() } else { 1 };
        let Some(dummies) = dummies else {
            // A slice thread: constant views are made by the committing thread.
            ctx.bail(crate::slices::spec::Hole::Pipeline);
            return Err("FMASK view on a slice thread".into());
        };
        let v = dummies.constant_view(ctx.sched, ty, t.width(), t.height(), layers, fmask_identity(samples, fragments)).ok_or("FMASK view")?;
        return Ok((v, None));
    }
    let Some(fmt) = texel_format(t.format()) else {
        return Err(format!("image format {}", t.format()));
    };
    if fmt.block_compressed && !dev.caps.texture_compression_bc {
        return Err(format!("block-compressed image format {:?}", fmt.host));
    }
    // A storage image the program accesses as raw unsigned words (32-bit
    // signed stores, float atomics on their bits) is viewed as UINT.
    // 64-bit atomics work on the bits of any 32_32 format.
    let atomic_bits = req.storage && req.atomic64 && matches!(fmt.host, vk::Format::R32G32_SINT | vk::Format::R32G32_SFLOAT);
    let raw_uint = (req.storage && req.numeric == ImageNumericClass::Uint && fmt.numeric != ImageNumericClass::Uint)
        .then(|| if atomic_bits { Some(vk::Format::R32G32_UINT) } else { raw_uint_view(fmt.host) })
        .flatten();
    if fmt.numeric != req.numeric && raw_uint.is_none() {
        return Err(format!("image {:?} bound as {:?}", fmt.host, req.numeric));
    }
    let key = crate::backend::profile::time(17, || texture_key(&t, &fmt))?;
    let window = sampled_array_window(&t, &key, req, crate::images::sampled_dcc::descriptor_dcc(words).is_some(), ty);
    let key = window.unwrap_or(key);
    let msaa = key.samples > 1;
    if msaa != matches!(req.dimension, ImageDimension::Dim2DMsaa | ImageDimension::Dim2DMsaaArray) {
        return Err(format!("image type {} bound as {:?}", t.ty(), req.dimension));
    }
    // Levels the view covers.
    let last = t.last_level().min(key.levels - 1);
    let base = (t.base_level() + req.mip_offset).min(last);
    let (base, levels) = if msaa { (0, 1) } else if req.storage || req.single_level { (base, 1) } else { (base, last - base + 1) };
    // An 8-bit texture over the stencil plane of a depth/stencil target
    // samples that target's stencil aspect.
    let stencil_owner = if req.storage || msaa { None } else { ctx.images.stencil_owner(&key) };
    let mut id = match stencil_owner {
        Some(id) => id,
        None => match crate::backend::profile::time(18, || ctx.images.find_alias(&key, base + levels)) {
            Some(id) => id,
            None => ctx.images.find_or_create(ctx.sched, &mut ctx.buffers, &key).map_err(|e| creation_error(&key, e))?,
        },
    };
    if req.storage && ctx.images.get(id).is_some_and(|i| i.key.is_depth()) {
        // Depth formats cannot be storage images: work on a colour image of
        // the same surface (contents converted on the GPU both ways).
        id = ctx.images.find_or_create(ctx.sched, &mut ctx.buffers, &key).map_err(|e| creation_error(&key, e))?;
    }
    // The DCC keys of a compressed surface decide what it reads.
    if let Some(dcc) = crate::images::sampled_dcc::descriptor_dcc(words).filter(|_| stencil_owner.is_none() && crate::images::sampled_dcc::enabled()) {
        match (ctx.images.exact(), ctx.buffers.exact()) {
            (Some(images), Some(buffers)) => {
                images.sampled_dcc(ctx.sched, buffers, id, dcc);
            }
            _ => {
                let cache = ctx.images.cache();
                if cache.has_fill(dcc.address) || cache.needs_upload(id) {
                    ctx.bail(crate::slices::spec::Hole::ClearFill);
                }
            }
        }
    }
    ctx.images.prepare(ctx.sched, &mut ctx.buffers, id, false);
    let Some(img) = ctx.images.get(id) else { return Err("image vanished".into()) };
    let (image_format, aspect, usage, image_key) = (img.key.format, img.image.aspect, img.image.usage, img.key);
    let depth = aspect.contains(vk::ImageAspectFlags::DEPTH);
    let (view_format, view_aspect) = if stencil_owner.is_some() {
        (image_format, vk::ImageAspectFlags::STENCIL)
    } else if depth {
        (image_format, vk::ImageAspectFlags::DEPTH)
    } else if req.storage && req.atomic64 {
        // The texel bits of any 32_32 format.
        if !matches!(fmt.host, vk::Format::R32G32_UINT | vk::Format::R32G32_SINT | vk::Format::R32G32_SFLOAT) {
            return Err(format!("64-bit atomics on {:?}", fmt.host));
        }
        (vk::Format::R64_UINT, vk::ImageAspectFlags::COLOR)
    } else if req.storage {
        (raw_uint.unwrap_or_else(|| storage_view_format(fmt.host)), vk::ImageAspectFlags::COLOR)
    } else if req.srgb {
        (srgb_twin(fmt.host).unwrap_or(fmt.host), vk::ImageAspectFlags::COLOR)
    } else {
        (fmt.host, vk::ImageAspectFlags::COLOR)
    };
    if req.storage && (depth || !usage.contains(vk::ImageUsageFlags::STORAGE)) {
        return Err(format!("storage view of {image_format:?}"));
    }
    if !req.storage && !usage.contains(vk::ImageUsageFlags::SAMPLED) {
        return Err(format!("sampled view of {image_format:?}"));
    }
    let components = if req.storage {
        vk::ComponentMapping::default()
    } else {
        fmt.view_components(if req.host_swizzle { t.components() } else { vk::ComponentMapping::default() })
    };
    // MIN_LOD below the base level does not clamp anything.
    let min_lod = if req.storage || req.single_level || t.min_lod() <= t.base_level() * 256 { 0.0 } else { (t.min_lod() as f32 / 256.0).min(last as f32) };
    let base_layer = if image_key.dim == ImageDim::D3 || window.is_some() { 0 } else { t.base_array() };
    let range = ViewRange { ty, base_level: base, levels, base_layer, layers: image_key.layers.saturating_sub(base_layer).max(1), min_lod };
    match crate::backend::profile::time(19, || ctx.images.view_range(id, view_format, view_aspect, range, components)) {
        Some(v) => Ok((v, Some(id))),
        None => Err(format!("{ty:?} view {view_format:?} of {image_format:?} {:?} image", image_key.dim)),
    }
}

/// Sampled cube bindings are declared as cube arrays by the shader
/// back-end and take cube (array) views.
pub const CUBE_VIEWS: bool = true;
