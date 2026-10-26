// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest color/depth target formats and their host (Vulkan) equivalents.
//!
//! A color target is described by CB_COLOR_INFO's channel layout (FORMAT),
//! number type and component order (COMP_SWAP). The layout and number type
//! select a guest pixel format; the order may select a BGRA host format or
//! is otherwise expressed as an export component mapping the shader
//! back-end applies to the pixel shader outputs.

use ash::vk;

/// Guest pixel formats (the subset render targets use).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GuestFormat {
    Invalid = 0,
    R8Unorm = 1,
    R8Snorm = 2,
    R8Uint = 5,
    R8Sint = 6,
    R16Unorm = 7,
    R16Snorm = 8,
    R16Uint = 11,
    R16Sint = 12,
    R16Float = 13,
    Rg8Unorm = 14,
    Rg8Snorm = 15,
    Rg8Uint = 18,
    Rg8Sint = 19,
    R32Uint = 20,
    R32Sint = 21,
    R32Float = 22,
    Rg16Unorm = 23,
    Rg16Snorm = 24,
    Rg16Uint = 27,
    Rg16Sint = 28,
    Rg16Float = 29,
    R11G11B10Unorm = 30,
    R11G11B10Snorm = 31,
    R11G11B10Uint = 34,
    R11G11B10Sint = 35,
    R11G11B10Float = 36,
    R10G11B11Unorm = 37,
    R10G11B11Float = 43,
    A2B10G10R10Unorm = 44,
    Rgb10A2Unorm = 50,
    Rgb10A2Snorm = 51,
    Rgb10A2Uint = 54,
    Rgb10A2Sint = 55,
    Rgba8Unorm = 56,
    Rgba8Snorm = 57,
    Rgba8Uint = 60,
    Rgba8Sint = 61,
    Rg32Uint = 62,
    Rg32Sint = 63,
    Rg32Float = 64,
    Rgba16Unorm = 65,
    Rgba16Snorm = 66,
    Rgba16Uint = 69,
    Rgba16Sint = 70,
    Rgba16Float = 71,
    Rgba32Uint = 75,
    Rgba32Sint = 76,
    Rgba32Float = 77,
    R8Srgb = 128,
    Rg8Srgb = 129,
    Rgba8Srgb = 130,
    Rgb10A2Float = 131,
    B5G6R5Unorm = 133,
    Rgb5A1Unorm = 134,
    A1Rgb5Unorm = 135,
    Rgba4Unorm = 136,
}

/// Channel number types of CB_COLOR_INFO.NUMBER_TYPE.
pub mod number_type {
    pub const UNORM: u32 = 0;
    pub const SNORM: u32 = 1;
    pub const UINT: u32 = 4;
    pub const SINT: u32 = 5;
    pub const SRGB: u32 = 6;
    pub const FLOAT: u32 = 7;
}

/// Component order (COMP_SWAP): standard, alternate, reversed, alternate-reversed.
pub mod order {
    pub const STANDARD: u32 = 0;
    pub const ALTERNATE: u32 = 1;
    pub const REVERSED: u32 = 2;
    pub const ALTERNATE_REVERSED: u32 = 3;
}

/// Which physical attachment component receives which logical shader
/// output component; two bits per physical component (0xE4 = identity).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComponentMap(pub u8);

impl ComponentMap {
    pub const IDENTITY: ComponentMap = ComponentMap(0xE4);
    pub const GR: ComponentMap = ComponentMap(0xE1);
    pub const RABG: ComponentMap = ComponentMap(0x6C);
    pub const RGAB: ComponentMap = ComponentMap(0xB4);
    pub const BGRA: ComponentMap = ComponentMap(0xC6);
    pub const ABGR: ComponentMap = ComponentMap(0x1B);
    pub const AGBA: ComponentMap = ComponentMap(0x27);
    pub const ARBG: ComponentMap = ComponentMap(0x63);
    pub const AGBR: ComponentMap = ComponentMap(0x87);
    pub const ARGB: ComponentMap = ComponentMap(0x93);

    pub fn map(self, component: u32) -> u32 {
        if component < 4 {
            (self.0 as u32 >> (component * 2)) & 3
        } else {
            component
        }
    }

    /// The physical write mask of a logical component mask.
    pub fn apply_mask(self, logical: u32) -> u32 {
        (0..4).fold(0, |m, p| m | (((logical >> self.map(p)) & 1) << p))
    }

    /// This map, then `next`.
    pub fn then(self, next: ComponentMap) -> ComponentMap {
        ComponentMap((0..4).fold(0u8, |acc, p| acc | ((next.map(self.map(p)) as u8) << (p * 2))))
    }
}

/// Order mappings by component count (rows: order, columns: components-1).
const ORDER_MAPPINGS: [[ComponentMap; 4]; 4] = [
    [ComponentMap::IDENTITY, ComponentMap::IDENTITY, ComponentMap::IDENTITY, ComponentMap::IDENTITY],
    [ComponentMap::GR, ComponentMap::RABG, ComponentMap::RGAB, ComponentMap::BGRA],
    [ComponentMap::BGRA, ComponentMap::GR, ComponentMap::BGRA, ComponentMap::ABGR],
    [ComponentMap::AGBA, ComponentMap::ARBG, ComponentMap::AGBR, ComponentMap::ARGB],
];

struct LayoutFacts {
    layout: u32,
    first_linear: u32,
    linear_types: u8,
    components: u8,
    extra: Option<(GuestFormat, u32)>,
    all_orders: bool,
}

const fn tb(t: u32) -> u8 {
    1 << t
}
const UNORM_T: u8 = tb(number_type::UNORM);
const INT_T: u8 = tb(number_type::UINT) | tb(number_type::SINT);
const NORM_INT_T: u8 = UNORM_T | tb(number_type::SNORM) | INT_T;

const LAYOUTS: [LayoutFacts; 18] = [
    LayoutFacts { layout: 1, first_linear: 1, linear_types: NORM_INT_T, components: 1, extra: Some((GuestFormat::R8Srgb, number_type::SRGB)), all_orders: true },
    LayoutFacts { layout: 2, first_linear: 7, linear_types: NORM_INT_T, components: 1, extra: Some((GuestFormat::R16Float, number_type::FLOAT)), all_orders: true },
    LayoutFacts { layout: 3, first_linear: 14, linear_types: NORM_INT_T, components: 2, extra: Some((GuestFormat::Rg8Srgb, number_type::SRGB)), all_orders: true },
    LayoutFacts { layout: 4, first_linear: 20, linear_types: INT_T, components: 1, extra: Some((GuestFormat::R32Float, number_type::FLOAT)), all_orders: true },
    LayoutFacts { layout: 5, first_linear: 23, linear_types: NORM_INT_T, components: 2, extra: Some((GuestFormat::Rg16Float, number_type::FLOAT)), all_orders: true },
    LayoutFacts { layout: 6, first_linear: 30, linear_types: NORM_INT_T, components: 3, extra: Some((GuestFormat::R11G11B10Float, number_type::FLOAT)), all_orders: true },
    LayoutFacts { layout: 7, first_linear: 37, linear_types: NORM_INT_T, components: 3, extra: Some((GuestFormat::R10G11B11Float, number_type::FLOAT)), all_orders: true },
    LayoutFacts { layout: 8, first_linear: 44, linear_types: NORM_INT_T, components: 4, extra: None, all_orders: true },
    LayoutFacts { layout: 9, first_linear: 50, linear_types: NORM_INT_T, components: 4, extra: None, all_orders: true },
    LayoutFacts { layout: 10, first_linear: 56, linear_types: NORM_INT_T, components: 4, extra: Some((GuestFormat::Rgba8Srgb, number_type::SRGB)), all_orders: true },
    LayoutFacts { layout: 11, first_linear: 62, linear_types: INT_T, components: 2, extra: Some((GuestFormat::Rg32Float, number_type::FLOAT)), all_orders: true },
    LayoutFacts { layout: 12, first_linear: 65, linear_types: NORM_INT_T, components: 4, extra: Some((GuestFormat::Rgba16Float, number_type::FLOAT)), all_orders: true },
    LayoutFacts { layout: 14, first_linear: 75, linear_types: INT_T, components: 4, extra: Some((GuestFormat::Rgba32Float, number_type::FLOAT)), all_orders: true },
    LayoutFacts { layout: 16, first_linear: 133, linear_types: UNORM_T, components: 3, extra: None, all_orders: true },
    LayoutFacts { layout: 17, first_linear: 134, linear_types: UNORM_T, components: 4, extra: None, all_orders: true },
    LayoutFacts { layout: 18, first_linear: 135, linear_types: UNORM_T, components: 4, extra: None, all_orders: true },
    LayoutFacts { layout: 19, first_linear: 136, linear_types: UNORM_T, components: 4, extra: None, all_orders: true },
    LayoutFacts { layout: 31, first_linear: 0, linear_types: 0, components: 4, extra: Some((GuestFormat::Rgb10A2Float, number_type::FLOAT)), all_orders: false },
];

fn guest_format_from_raw(v: u32) -> GuestFormat {
    use GuestFormat::*;
    const ALL: [GuestFormat; 60] = [
        R8Unorm, R8Snorm, R8Uint, R8Sint, R16Unorm, R16Snorm, R16Uint, R16Sint, R16Float, Rg8Unorm, Rg8Snorm, Rg8Uint, Rg8Sint,
        R32Uint, R32Sint, R32Float, Rg16Unorm, Rg16Snorm, Rg16Uint, Rg16Sint, Rg16Float, R11G11B10Unorm, R11G11B10Snorm,
        R11G11B10Uint, R11G11B10Sint, R11G11B10Float, R10G11B11Unorm, R10G11B11Float, A2B10G10R10Unorm, Rgb10A2Unorm,
        Rgb10A2Snorm, Rgb10A2Uint, Rgb10A2Sint, Rgba8Unorm, Rgba8Snorm, Rgba8Uint, Rgba8Sint, Rg32Uint, Rg32Sint, Rg32Float,
        Rgba16Unorm, Rgba16Snorm, Rgba16Uint, Rgba16Sint, Rgba16Float, Rgba32Uint, Rgba32Sint, Rgba32Float, R8Srgb, Rg8Srgb,
        Rgba8Srgb, Rgb10A2Float, B5G6R5Unorm, Rgb5A1Unorm, A1Rgb5Unorm, Rgba4Unorm, Invalid, Invalid, Invalid, Invalid,
    ];
    ALL.iter().copied().find(|f| *f as u32 == v && *f != Invalid).unwrap_or(Invalid)
}

/// Guest pixel format and component count of a color target encoding.
pub fn render_target_encoding(layout: u32, ntype: u32) -> Option<(GuestFormat, u8, bool)> {
    let f = LAYOUTS.iter().find(|f| f.layout == layout)?;
    if ntype >= 8 {
        return None;
    }
    let fmt = match f.extra {
        Some((e, t)) if t == ntype => e,
        _ if f.linear_types & (1 << ntype) != 0 => {
            let first_type = f.linear_types.trailing_zeros();
            guest_format_from_raw(f.first_linear + ntype - first_type)
        }
        _ => GuestFormat::Invalid,
    };
    (fmt != GuestFormat::Invalid).then_some((fmt, f.components, f.all_orders))
}

/// Host format of a guest format (Undefined when there is none).
pub fn host_format(f: GuestFormat) -> vk::Format {
    use vk::Format as V;
    use GuestFormat::*;
    match f {
        R8Unorm | R8Srgb => V::R8_UNORM,
        R8Uint => V::R8_UINT,
        R8Snorm => V::R8_SNORM,
        R8Sint => V::R8_SINT,
        R16Unorm => V::R16_UNORM,
        R16Snorm => V::R16_SNORM,
        R16Uint => V::R16_UINT,
        R16Sint => V::R16_SINT,
        R16Float => V::R16_SFLOAT,
        Rg8Unorm | Rg8Srgb => V::R8G8_UNORM,
        Rg8Snorm => V::R8G8_SNORM,
        Rg8Uint => V::R8G8_UINT,
        Rg8Sint => V::R8G8_SINT,
        R32Uint => V::R32_UINT,
        R32Sint => V::R32_SINT,
        R32Float => V::R32_SFLOAT,
        Rg16Unorm => V::R16G16_UNORM,
        Rg16Snorm => V::R16G16_SNORM,
        Rg16Uint => V::R16G16_UINT,
        Rg16Sint => V::R16G16_SINT,
        Rg16Float => V::R16G16_SFLOAT,
        R11G11B10Float => V::B10G11R11_UFLOAT_PACK32,
        // Packed by the pixel program (`PixelOutputBinding::packed_unorm_11_11_10`).
        R11G11B10Unorm => V::R32_UINT,
        Rgb10A2Unorm => V::A2B10G10R10_UNORM_PACK32,
        Rgb10A2Uint => V::A2B10G10R10_UINT_PACK32,
        Rgba8Unorm => V::R8G8B8A8_UNORM,
        Rgba8Snorm => V::R8G8B8A8_SNORM,
        Rgba8Uint => V::R8G8B8A8_UINT,
        Rgba8Sint => V::R8G8B8A8_SINT,
        Rgba8Srgb => V::R8G8B8A8_SRGB,
        Rg32Uint => V::R32G32_UINT,
        Rg32Sint => V::R32G32_SINT,
        Rg32Float => V::R32G32_SFLOAT,
        Rgba16Unorm => V::R16G16B16A16_UNORM,
        Rgba16Snorm => V::R16G16B16A16_SNORM,
        Rgba16Uint => V::R16G16B16A16_UINT,
        Rgba16Sint => V::R16G16B16A16_SINT,
        Rgba16Float => V::R16G16B16A16_SFLOAT,
        Rgba32Uint => V::R32G32B32A32_UINT,
        Rgba32Sint => V::R32G32B32A32_SINT,
        Rgba32Float => V::R32G32B32A32_SFLOAT,
        B5G6R5Unorm => V::B5G6R5_UNORM_PACK16,
        Rgb5A1Unorm => V::R5G5B5A1_UNORM_PACK16,
        Rgba4Unorm => V::R4G4B4A4_UNORM_PACK16,
        _ => V::UNDEFINED,
    }
}

/// Render-target bytes per element of a guest format.
pub fn bytes_per_element(f: GuestFormat) -> u32 {
    use GuestFormat::*;
    match f {
        R8Unorm | R8Snorm | R8Uint | R8Sint | R8Srgb => 1,
        R16Unorm | R16Snorm | R16Uint | R16Sint | R16Float | Rg8Unorm | Rg8Snorm | Rg8Uint | Rg8Sint | Rg8Srgb | B5G6R5Unorm
        | Rgb5A1Unorm | A1Rgb5Unorm | Rgba4Unorm => 2,
        Rg32Uint | Rg32Sint | Rg32Float | Rgba16Unorm | Rgba16Snorm | Rgba16Uint | Rgba16Sint | Rgba16Float => 8,
        Rgba32Uint | Rgba32Sint | Rgba32Float => 16,
        Invalid => 0,
        _ => 4,
    }
}

/// A resolved color target format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderTargetFormat {
    pub guest: GuestFormat,
    pub host: vk::Format,
    pub bytes_per_element: u32,
    /// Logical shader output component → physical attachment component.
    pub export_mapping: ComponentMap,
}

impl RenderTargetFormat {
    /// Bit rearrangement between the stored elements and the host format.
    pub fn transform(&self) -> crate::tiler::ColorTransform {
        if self.guest == GuestFormat::R10G11B11Float {
            crate::tiler::ColorTransform::Reverse10_11_11
        } else {
            crate::tiler::ColorTransform::None
        }
    }
}

/// Format of a color target from CB_COLOR_INFO fields.
pub fn render_target_format(layout: u32, ntype: u32, ord: u32) -> Option<RenderTargetFormat> {
    let (guest, components, all_orders) = render_target_encoding(layout, ntype)?;
    if !all_orders && ord != order::STANDARD {
        return None;
    }
    let (host, host_to_storage) = if guest == GuestFormat::R10G11B11Float {
        // The 10-bit field is stored low: transformed into a B10G11R11
        // host value whose R and B are the stored Z and X.
        (vk::Format::B10G11R11_UFLOAT_PACK32, ComponentMap::BGRA)
    } else if ord == order::ALTERNATE {
        match guest {
            GuestFormat::Rgba8Unorm => (vk::Format::B8G8R8A8_UNORM, ComponentMap::BGRA),
            GuestFormat::Rgba8Snorm => (vk::Format::B8G8R8A8_SNORM, ComponentMap::BGRA),
            GuestFormat::Rgba8Srgb => (vk::Format::B8G8R8A8_SRGB, ComponentMap::BGRA),
            GuestFormat::Rgb10A2Unorm => (vk::Format::A2R10G10B10_UNORM_PACK32, ComponentMap::BGRA),
            // Packed by the pixel program (see the standard-order arm).
            GuestFormat::Rgba4Unorm => (vk::Format::R16_UINT, ComponentMap::IDENTITY),
            _ => (host_format(guest), ComponentMap::IDENTITY),
        }
    } else {
        match guest {
            GuestFormat::Rgb5A1Unorm => (vk::Format::A1R5G5B5_UNORM_PACK16, ComponentMap::BGRA),
            GuestFormat::A1Rgb5Unorm => (vk::Format::R5G5B5A1_UNORM_PACK16, ComponentMap::ABGR),
            // Packed by the pixel program
            // (`PixelOutputBinding::packed_unorm_4_4_4_4`): the export
            // writes the element's word (X in the lowest field) into a
            // 16-bit unsigned attachment; the mapping still places each
            // guest channel where the packed-16 host format held it.
            GuestFormat::Rgba4Unorm => (vk::Format::R16_UINT, ComponentMap::ABGR),
            _ => (host_format(guest), ComponentMap::IDENTITY),
        }
    };
    let bpe = bytes_per_element(guest);
    if host == vk::Format::UNDEFINED || bpe == 0 || ord > 3 {
        return None;
    }
    let mapping = host_to_storage.then(ORDER_MAPPINGS[ord as usize][components as usize - 1]);
    Some(RenderTargetFormat { guest, host, bytes_per_element: bpe, export_mapping: mapping })
}

/// An unsigned small float of `mantissa_bits` mantissa bits and a 5-bit
/// exponent (the 11- and 10-bit channels of B10G11R11).
fn small_ufloat(v: u32, mantissa_bits: u32) -> f32 {
    let mantissa = v & ((1 << mantissa_bits) - 1);
    let exponent = (v >> mantissa_bits) & 0x1F;
    match exponent {
        0 => mantissa as f32 * 2f32.powi(-14 - mantissa_bits as i32),
        31 if mantissa == 0 => f32::INFINITY,
        31 => f32::NAN,
        e => f32::from_bits((e + 112) << 23 | mantissa << (23 - mantissa_bits)),
    }
}

/// Decode the fast-clear words (CB_COLOR_CLEAR_WORD0/1) of `format`. The
/// words hold the clear texel in the stored layout: its first 64 bits for
/// 8-byte formats; 16-byte formats keep R = G = B in WORD0 and A in WORD1.
pub fn decode_clear_color(format: vk::Format, words: [u32; 2]) -> Option<vk::ClearColorValue> {
    let packed = words[0];
    let unorm8 = |v: u32| (v & 0xFF) as f32 / 255.0;
    let snorm8 = |v: u32| ((v as u8 as i8) as f32 / 127.0).max(-1.0);
    let unorm16 = |v: u32| (v & 0xFFFF) as f32 / 65535.0;
    let snorm16 = |v: u32| ((v as u16 as i16) as f32 / 32767.0).max(-1.0);
    let srgb8 = |v: u32| {
        let e = (v & 0xFF) as f32 / 255.0;
        if e <= 0.04045 {
            e / 12.92
        } else {
            ((e + 0.055) / 1.055).powf(2.4)
        }
    };
    let half = |v: u32| {
        let sign = (v & 0x8000) << 16;
        let exponent = (v >> 10) & 0x1F;
        let mantissa = v & 0x3FF;
        if exponent == 0 {
            (mantissa as f32 * 2f32.powi(-24)).copysign(f32::from_bits(sign | 0x3F80_0000))
        } else {
            f32::from_bits(sign | (if exponent == 31 { 255 } else { exponent + 112 }) << 23 | mantissa << 13)
        }
    };
    let (lo, hi) = (words[0], words[1]);
    let halves = |f: &dyn Fn(u32) -> f32| [f(lo & 0xFFFF), f(lo >> 16), f(hi & 0xFFFF), f(hi >> 16)];
    let uint = |v: [u32; 4]| Some(vk::ClearColorValue { uint32: v });
    let sint = |v: [u32; 4]| Some(vk::ClearColorValue { int32: v.map(|x| x as i32) });
    use vk::Format as F;
    let f = match format {
        F::R16G16_SFLOAT => [half(packed & 0xFFFF), half(packed >> 16), 0.0, 0.0],
        F::R16_SFLOAT => [half(packed & 0xFFFF), 0.0, 0.0, 0.0],
        F::R16G16_UNORM => [unorm16(packed), unorm16(packed >> 16), 0.0, 0.0],
        F::R16_UNORM => [unorm16(packed), 0.0, 0.0, 0.0],
        F::R16G16_SNORM => [snorm16(packed), snorm16(packed >> 16), 0.0, 0.0],
        F::R16_SNORM => [snorm16(packed), 0.0, 0.0, 0.0],
        F::R16_UINT => return uint([packed & 0xFFFF, 0, 0, 0]),
        F::R16_SINT => return sint([packed as i16 as u32, 0, 0, 0]),
        F::R16G16_UINT => return uint([packed & 0xFFFF, packed >> 16, 0, 0]),
        F::R16G16_SINT => return sint([packed as i16 as u32, (packed >> 16) as i16 as u32, 0, 0]),
        F::R8_UNORM => [unorm8(packed), 0.0, 0.0, 0.0],
        F::R8_SNORM => [snorm8(packed), 0.0, 0.0, 0.0],
        F::R8_UINT => return uint([packed & 0xFF, 0, 0, 0]),
        F::R8_SINT => return sint([packed as i8 as u32, 0, 0, 0]),
        F::R8G8_UNORM => [unorm8(packed), unorm8(packed >> 8), 0.0, 0.0],
        F::R8G8_SNORM => [snorm8(packed), snorm8(packed >> 8), 0.0, 0.0],
        F::R8G8_UINT => return uint([packed & 0xFF, (packed >> 8) & 0xFF, 0, 0]),
        F::R8G8_SINT => return sint([packed as i8 as u32, (packed >> 8) as i8 as u32, 0, 0]),
        F::R8G8B8A8_SNORM => [snorm8(packed), snorm8(packed >> 8), snorm8(packed >> 16), snorm8(packed >> 24)],
        F::R8G8B8A8_UINT => return uint([packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, packed >> 24]),
        F::R8G8B8A8_SINT => return sint([packed as i8 as u32, (packed >> 8) as i8 as u32, (packed >> 16) as i8 as u32, (packed >> 24) as i8 as u32]),
        F::R32_SFLOAT => [f32::from_bits(packed), 0.0, 0.0, 0.0],
        F::R32_UINT => return uint([packed, 0, 0, 0]),
        F::R32_SINT => return sint([packed, 0, 0, 0]),
        F::R8G8B8A8_SRGB => [srgb8(packed), srgb8(packed >> 8), srgb8(packed >> 16), unorm8(packed >> 24)],
        F::B8G8R8A8_SRGB => [srgb8(packed >> 16), srgb8(packed >> 8), srgb8(packed), unorm8(packed >> 24)],
        F::R8G8B8A8_UNORM => [unorm8(packed), unorm8(packed >> 8), unorm8(packed >> 16), unorm8(packed >> 24)],
        F::B8G8R8A8_UNORM => [unorm8(packed >> 16), unorm8(packed >> 8), unorm8(packed), unorm8(packed >> 24)],
        F::B8G8R8A8_SNORM => [snorm8(packed >> 16), snorm8(packed >> 8), snorm8(packed), snorm8(packed >> 24)],
        F::A2B10G10R10_UNORM_PACK32 => [
            (packed & 0x3FF) as f32 / 1023.0,
            ((packed >> 10) & 0x3FF) as f32 / 1023.0,
            ((packed >> 20) & 0x3FF) as f32 / 1023.0,
            ((packed >> 30) & 3) as f32 / 3.0,
        ],
        F::A2R10G10B10_UNORM_PACK32 => [
            ((packed >> 20) & 0x3FF) as f32 / 1023.0,
            ((packed >> 10) & 0x3FF) as f32 / 1023.0,
            (packed & 0x3FF) as f32 / 1023.0,
            ((packed >> 30) & 3) as f32 / 3.0,
        ],
        F::A2B10G10R10_UINT_PACK32 => return uint([packed & 0x3FF, (packed >> 10) & 0x3FF, (packed >> 20) & 0x3FF, packed >> 30]),
        F::B10G11R11_UFLOAT_PACK32 => [small_ufloat(packed, 6), small_ufloat(packed >> 11, 6), small_ufloat(packed >> 22, 5), 1.0],
        F::R16G16B16A16_SFLOAT => halves(&half),
        F::R16G16B16A16_UNORM => halves(&unorm16),
        F::R16G16B16A16_SNORM => halves(&snorm16),
        F::R16G16B16A16_UINT => return uint([lo & 0xFFFF, lo >> 16, hi & 0xFFFF, hi >> 16]),
        F::R16G16B16A16_SINT => return sint([lo as i16 as u32, (lo >> 16) as i16 as u32, hi as i16 as u32, (hi >> 16) as i16 as u32]),
        F::R32G32_SFLOAT => [f32::from_bits(lo), f32::from_bits(hi), 0.0, 0.0],
        F::R32G32_UINT => return uint([lo, hi, 0, 0]),
        F::R32G32_SINT => return sint([lo, hi, 0, 0]),
        F::R32G32B32A32_SFLOAT => [f32::from_bits(lo), f32::from_bits(lo), f32::from_bits(lo), f32::from_bits(hi)],
        F::R32G32B32A32_UINT => return uint([lo, lo, lo, hi]),
        F::R32G32B32A32_SINT => return sint([lo, lo, lo, hi]),
        _ => return None,
    };
    Some(vk::ClearColorValue { float32: f })
}

/// Formats whose DCC fixed clear codes (0x40/0x80/0xC0) are valid.
pub fn supports_dcc_fixed_clear(format: vk::Format) -> bool {
    use vk::Format as F;
    matches!(
        format,
        F::R8_UNORM
            | F::R8G8_UNORM
            | F::R8G8B8A8_UNORM
            | F::R8G8B8A8_SRGB
            | F::B8G8R8A8_UNORM
            | F::B8G8R8A8_SRGB
            | F::A2B10G10R10_UNORM_PACK32
            | F::A2R10G10B10_UNORM_PACK32
            | F::R5G6B5_UNORM_PACK16
            | F::A1R5G5B5_UNORM_PACK16
            | F::R4G4B4A4_UNORM_PACK16
            | F::R16_UNORM
            | F::R16G16_UNORM
            | F::R16G16B16A16_UNORM
            | F::R16_SFLOAT
            | F::R16G16_SFLOAT
            | F::R16G16B16A16_SFLOAT
            | F::R32_SFLOAT
            | F::R32G32_SFLOAT
            | F::R32G32B32A32_SFLOAT
            | F::B10G11R11_UFLOAT_PACK32
    )
}

/// True for the byte codes a DCC fast clear leaves in metadata.
pub fn is_dcc_clear_code(code: u8) -> bool {
    matches!(code, 0x00 | 0x20 | 0x40 | 0x80 | 0xC0)
}

/// The DCC clear code layout of a color target: true when the alpha flag
/// of a fixed clear code (bit 6) lands in the component stored last
/// (alpha on the most significant bits), false when it lands in the first.
pub fn dcc_alpha_on_msb(layout: u32, ntype: u32, ord: u32) -> bool {
    match layout {
        // 10_10_10_2, 10_10_10_2 float, 5_5_5_1.
        9 | 31 | 17 => return true,
        // 2_10_10_10, 1_5_5_5.
        8 | 18 => return false,
        _ => {}
    }
    let components = render_target_encoding(layout, ntype).map_or(4, |(_, c, _)| c);
    if components == 1 {
        return ord != order::STANDARD;
    }
    components == 3 || ord == order::STANDARD || ord == order::ALTERNATE
}

/// Components of the host formats a DCC fixed clear code can describe.
fn fixed_clear_components(format: vk::Format) -> usize {
    use vk::Format as F;
    match format {
        F::R8_UNORM | F::R16_UNORM | F::R16_SFLOAT | F::R32_SFLOAT => 1,
        F::R8G8_UNORM | F::R16G16_UNORM | F::R16G16_SFLOAT | F::R32G32_SFLOAT => 2,
        F::R5G6B5_UNORM_PACK16 | F::B10G11R11_UFLOAT_PACK32 => 3,
        _ => 4,
    }
}

/// Clear value of a DCC clear code: 0x00 transparent black, 0x20 the
/// register clear color, 0x40/0x80/0xC0 the fixed clears (bit 7 sets the
/// color channels, bit 6 alpha). The alpha flag belongs to the component
/// `alpha_on_msb` selects ([`dcc_alpha_on_msb`]); one- and two-component
/// formats take their missing lanes from it before the host swizzle. None
/// when the format cannot express the code.
pub fn decode_dcc_clear(code: u8, format: vk::Format, register: Option<vk::ClearColorValue>, alpha_on_msb: bool) -> Option<vk::ClearColorValue> {
    use vk::Format as F;
    match code {
        0x00 => return Some(vk::ClearColorValue { float32: [0.0; 4] }),
        0x20 => return register,
        0x40 | 0x80 | 0xC0 if supports_dcc_fixed_clear(format) => {}
        _ => return None,
    }
    let rgb = if code & 0x80 != 0 { 1.0 } else { 0.0 };
    let alpha = if code & 0x40 != 0 { 1.0 } else { 0.0 };
    let mut c = [rgb, rgb, rgb, alpha];
    if !alpha_on_msb {
        c.swap(0, 3);
    }
    match fixed_clear_components(format) {
        1 => c[0] = c[3],
        2 => c[1] = c[3],
        _ => {}
    }
    match format {
        F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB | F::A2R10G10B10_UNORM_PACK32 | F::A1R5G5B5_UNORM_PACK16 | F::R5G6B5_UNORM_PACK16 => c.swap(0, 2),
        F::R4G4B4A4_UNORM_PACK16 => c.reverse(),
        _ => {}
    }
    Some(vk::ClearColorValue { float32: c })
}

/// A dword fill of a color surface as a clear value: the word repeats
/// across each pixel, so 16-bit single-channel formats need equal halves
/// and 64-bit RGBA16F only expresses zero; 1- and 2-byte formats need
/// every pixel of the word equal. None when the fill is not one
/// color.
pub fn decode_dword_fill(format: vk::Format, packed: u32) -> Option<vk::ClearColorValue> {
    use vk::Format as F;
    let two_byte = matches!(format, F::R16_SFLOAT | F::R16_UNORM | F::R16_SNORM | F::R16_UINT | F::R16_SINT | F::R8G8_UNORM | F::R8G8_SNORM | F::R8G8_UINT | F::R8G8_SINT);
    if two_byte && packed & 0xFFFF != packed >> 16 {
        return None;
    }
    let one_byte = matches!(format, F::R8_UNORM | F::R8_SNORM | F::R8_UINT | F::R8_SINT);
    if one_byte && packed != (packed & 0xFF) * 0x0101_0101 {
        return None;
    }
    if format == F::R16G16B16A16_SFLOAT {
        return (packed == 0).then_some(vk::ClearColorValue { float32: [0.0; 4] });
    }
    decode_clear_color(format, [packed, packed])
}

/// A dword fill of a stencil plane: one byte repeated.
pub fn decode_stencil_fill(packed: u32) -> Option<u8> {
    let v = packed as u8;
    (packed == v as u32 * 0x0101_0101).then_some(v)
}

/// A dword fill of a 32-bit float depth plane: a depth in [0, 1].
pub fn decode_depth_fill(format: vk::Format, packed: u32) -> Option<f32> {
    if !matches!(format, vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT) {
        return None;
    }
    let v = f32::from_bits(packed);
    (v.is_finite() && (0.0..=1.0).contains(&v)).then_some(v)
}

/// Depth formats of DB_Z_INFO.FORMAT.
pub mod depth {
    pub const INVALID: u32 = 0;
    pub const Z16: u32 = 1;
    pub const Z32_FLOAT: u32 = 3;
}

/// The device renders to D16_UNORM_S8_UINT (cleared by a device that
/// cannot: Z16 + stencil then lives in D32_SFLOAT_S8_UINT, the images widen
/// the guest's 16-bit depth on upload and narrow it on write-back).
pub static DEPTH16_STENCIL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Host attachment format and guest element size of a depth target.
pub fn depth_format(z_format: u32, has_stencil: bool) -> Option<(ash::vk::Format, u32)> {
    use ash::vk::Format as F;
    match (z_format, has_stencil) {
        (depth::Z16, false) => Some((F::D16_UNORM, 2)),
        (depth::Z16, true) if !DEPTH16_STENCIL.load(std::sync::atomic::Ordering::Relaxed) => Some((F::D32_SFLOAT_S8_UINT, 2)),
        (depth::Z16, true) => Some((F::D16_UNORM_S8_UINT, 2)),
        (depth::Z32_FLOAT, false) => Some((F::D32_SFLOAT, 4)),
        (depth::Z32_FLOAT, true) => Some((F::D32_SFLOAT_S8_UINT, 4)),
        (depth::INVALID, true) => Some((F::S8_UINT, 1)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_render_target_formats() {
        // FORMAT 10 (8_8_8_8), UNORM, standard order.
        let f = render_target_format(10, number_type::UNORM, order::STANDARD).unwrap();
        assert_eq!(f.host, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(f.bytes_per_element, 4);
        assert_eq!(f.export_mapping, ComponentMap::IDENTITY);
        // Alternate order selects the BGRA host format with an identity export.
        let f = render_target_format(10, number_type::SRGB, order::ALTERNATE).unwrap();
        assert_eq!(f.host, vk::Format::B8G8R8A8_SRGB);
        assert_eq!(f.export_mapping, ComponentMap::IDENTITY);
        // 16_16_16_16 float, 11_11_10 float, 32 float, 2_10_10_10 unorm.
        assert_eq!(render_target_format(12, number_type::FLOAT, 0).unwrap().host, vk::Format::R16G16B16A16_SFLOAT);
        assert_eq!(render_target_format(6, number_type::FLOAT, 0).unwrap().host, vk::Format::B10G11R11_UFLOAT_PACK32);
        // 10_11_11 float: the same host format, transformed, with R and B swapped.
        let f = render_target_format(7, number_type::FLOAT, 0).unwrap();
        assert_eq!((f.host, f.export_mapping, f.transform()), (vk::Format::B10G11R11_UFLOAT_PACK32, ComponentMap::BGRA, crate::tiler::ColorTransform::Reverse10_11_11));
        assert_eq!(render_target_format(6, number_type::FLOAT, 0).unwrap().transform(), crate::tiler::ColorTransform::None);
        // 11_11_10 UNORM: packed by the pixel program into a 32-bit word.
        assert_eq!(render_target_format(6, number_type::UNORM, 0).unwrap().host, vk::Format::R32_UINT);
        assert_eq!(render_target_format(4, number_type::FLOAT, 0).unwrap().host, vk::Format::R32_SFLOAT);
        assert_eq!(render_target_format(9, number_type::UNORM, 0).unwrap().host, vk::Format::A2B10G10R10_UNORM_PACK32);
        assert_eq!(render_target_format(4, number_type::UINT, 0).unwrap().host, vk::Format::R32_UINT);
        // Reversed order of a 2-channel target swaps the outputs.
        let f = render_target_format(5, number_type::FLOAT, order::REVERSED).unwrap();
        assert_eq!(f.export_mapping, ComponentMap::GR);
        assert!(render_target_format(0, 0, 0).is_none());
    }

    #[test]
    fn clear_values_decode() {
        let c = decode_clear_color(vk::Format::R8G8B8A8_UNORM, [0xFF00_80FF, 0]).unwrap();
        // SAFETY: float variant written above.
        let f = unsafe { c.float32 };
        assert_eq!(f[0], 1.0);
        assert!((f[1] - 128.0 / 255.0).abs() < 1e-6);
        assert_eq!(f[2], 0.0);
        assert_eq!(f[3], 1.0);
        let w = decode_dcc_clear(0xC0, vk::Format::R16G16B16A16_SFLOAT, None, true).unwrap();
        // SAFETY: float variant.
        assert_eq!(unsafe { w.float32 }, [1.0; 4]);
        assert!(decode_dcc_clear(0x20, vk::Format::R8G8B8A8_UNORM, None, true).is_none());
        assert!(decode_dcc_clear(0x40, vk::Format::R32G32B32A32_UINT, None, true).is_none());
        assert_eq!(ComponentMap::BGRA.apply_mask(0b0001), 0b0100);
    }

    #[test]
    fn wide_and_packed_float_clear_words_decode() {
        // 8-byte texels take both words: RGBA16F (1, 0.5, -2, 0.25).
        let c = unsafe { decode_clear_color(vk::Format::R16G16B16A16_SFLOAT, [0x3800_3C00, 0x3400_C000]).unwrap().float32 };
        assert_eq!(c, [1.0, 0.5, -2.0, 0.25]);
        let c = unsafe { decode_clear_color(vk::Format::R32G32_SFLOAT, [1.5f32.to_bits(), 3.0f32.to_bits()]).unwrap().float32 };
        assert_eq!(c, [1.5, 3.0, 0.0, 0.0]);
        let c = unsafe { decode_clear_color(vk::Format::R16G16B16A16_UINT, [0x0002_0001, 0x0004_0003]).unwrap().uint32 };
        assert_eq!(c, [1, 2, 3, 4]);
        // 16-byte texels: WORD0 is R = G = B, WORD1 is A.
        let c = unsafe { decode_clear_color(vk::Format::R32G32B32A32_SFLOAT, [0.25f32.to_bits(), 1.0f32.to_bits()]).unwrap().float32 };
        assert_eq!(c, [0.25, 0.25, 0.25, 1.0]);
        // 11_11_10 float: R 1.0 (0x3C0), G 0.5 (0x380), B 2.0 (0x200 of 10 bits).
        let packed = 0x3C0 | 0x380 << 11 | 0x200 << 22;
        let c = unsafe { decode_clear_color(vk::Format::B10G11R11_UFLOAT_PACK32, [packed, 0]).unwrap().float32 };
        assert_eq!(c, [1.0, 0.5, 2.0, 1.0]);
        // 10_11_11 float targets store the fields reversed; the register
        // word is moved to the host layout before it is decoded.
        let f = render_target_format(7, number_type::FLOAT, 0).unwrap();
        let mut w = [0x200 | 0x380 << 10 | 0x3C0 << 21, 0];
        f.transform().apply(&mut w, true);
        let c = unsafe { decode_clear_color(f.host, w).unwrap().float32 };
        assert_eq!(c, [1.0, 0.5, 2.0, 1.0]);
        // A dword fill of a 16-byte target is that word in every channel.
        let c = unsafe { decode_dword_fill(vk::Format::R32G32B32A32_UINT, 7).unwrap().uint32 };
        assert_eq!(c, [7; 4]);
        // 8-byte integer and UNORM16 clears keep each channel and its sign.
        let words = |w: u64| [w as u32, (w >> 32) as u32];
        let c = unsafe { decode_clear_color(vk::Format::R16G16B16A16_UNORM, words(0x0001_8000_0000_FFFF)).unwrap().float32 };
        assert_eq!(c, [1.0, 0.0, 32768.0 / 65535.0, 1.0 / 65535.0]);
        let c = unsafe { decode_clear_color(vk::Format::R16G16B16A16_SINT, words(0x7FFF_0001_FFFF_8000)).unwrap().uint32 };
        assert_eq!(c, [0xFFFF_8000, 0xFFFF_FFFF, 1, 0x7FFF]);
        let c = unsafe { decode_clear_color(vk::Format::R32G32_UINT, words(0x89AB_CDEF_0123_4567)).unwrap().uint32 };
        assert_eq!(c, [0x0123_4567, 0x89AB_CDEF, 0, 0]);
        let c = unsafe { decode_clear_color(vk::Format::R32G32_SINT, words(0x8000_0000_FFFF_FFFF)).unwrap().uint32 };
        assert_eq!(c, [0xFFFF_FFFF, 0x8000_0000, 0, 0]);
        // A dword fill repeats across a 64-bit pixel.
        let c = unsafe { decode_dword_fill(vk::Format::R16G16B16A16_UINT, 0x2222_1111).unwrap().uint32 };
        assert_eq!(c, [0x1111, 0x2222, 0x1111, 0x2222]);
    }

    #[test]
    fn dcc_fixed_clears_follow_the_alpha_position() {
        let f = |code, format, msb| unsafe { decode_dcc_clear(code, format, None, msb).unwrap().float32 };
        // Opaque black / transparent white of RGBA8, alpha last.
        assert!(dcc_alpha_on_msb(10, number_type::UNORM, order::STANDARD));
        assert_eq!(f(0x40, vk::Format::R8G8B8A8_UNORM, true), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(f(0x80, vk::Format::R8G8B8A8_UNORM, true), [1.0, 1.0, 1.0, 0.0]);
        // Reversed order (ABGR in memory): alpha is the first component.
        assert!(!dcc_alpha_on_msb(10, number_type::UNORM, order::REVERSED));
        assert_eq!(f(0x40, vk::Format::R8G8B8A8_UNORM, false), [1.0, 0.0, 0.0, 0.0]);
        // Two channels: the second lane takes the alpha flag.
        assert!(dcc_alpha_on_msb(5, number_type::FLOAT, order::STANDARD));
        assert_eq!(f(0x40, vk::Format::R16G16_SFLOAT, true)[..2], [0.0, 1.0]);
        // One channel, standard order: the color flag.
        assert!(!dcc_alpha_on_msb(1, number_type::UNORM, order::STANDARD));
        assert_eq!(f(0x80, vk::Format::R8_UNORM, false)[0], 1.0);
        assert_eq!(f(0x40, vk::Format::R8_UNORM, false)[0], 0.0);
        // BGRA host layout swaps red and blue; 2_10_10_10 keeps alpha first.
        assert!(!dcc_alpha_on_msb(8, number_type::UNORM, order::STANDARD));
        assert_eq!(f(0x40, vk::Format::B8G8R8A8_UNORM, false), [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(f(0x40, vk::Format::R4G4B4A4_UNORM_PACK16, true), [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn dword_fills_decode() {
        let half = unsafe { decode_clear_color(vk::Format::R16G16_SFLOAT, [0x3C00_3800, 0]).unwrap().float32 };
        assert_eq!(half, [0.5, 1.0, 0.0, 0.0]);
        assert!(decode_dword_fill(vk::Format::R16_UNORM, 0x0001_0002).is_none());
        assert_eq!(unsafe { decode_dword_fill(vk::Format::R16_UNORM, 0xFFFF_FFFF).unwrap().float32 }[0], 1.0);
        assert!(decode_dword_fill(vk::Format::R16G16B16A16_SFLOAT, 1).is_none());
        assert_eq!(decode_stencil_fill(0x8080_8080), Some(0x80));
        assert_eq!(decode_stencil_fill(0x0000_0080), None);
        assert_eq!(decode_depth_fill(vk::Format::D32_SFLOAT, 1.0f32.to_bits()), Some(1.0));
        assert_eq!(decode_depth_fill(vk::Format::D32_SFLOAT, 2.0f32.to_bits()), None);
        assert_eq!(decode_depth_fill(vk::Format::D16_UNORM, 0), None);
    }

    #[test]
    fn narrow_dword_fills_need_equal_pixels() {
        use vk::Format as F;
        // Narrow clears keep only their own channel bits.
        let u = |f, w| unsafe { decode_clear_color(f, [w, 0]).unwrap().uint32 };
        assert_eq!(u(F::R8_UINT, 0xFFFF_FF05), [5, 0, 0, 0]);
        assert_eq!(u(F::R8G8_UINT, 0xFFFF_0A05), [5, 10, 0, 0]);
        assert_eq!(u(F::R16_UINT, 0xFFFF_1234), [0x1234, 0, 0, 0]);
        // A dword fill over 1- and 2-byte pixels is one color only when
        // every pixel in the word holds the same value.
        assert!(decode_dword_fill(F::R8_UINT, 0x0102_0304).is_none());
        assert_eq!(unsafe { decode_dword_fill(F::R8_UINT, 0x0707_0707).unwrap().uint32 }[0], 7);
        assert!(decode_dword_fill(F::R8_UNORM, 0x0000_00FF).is_none());
        assert!(decode_dword_fill(F::R8G8_UNORM, 0x00FF_00FE).is_none());
        assert_eq!(unsafe { decode_dword_fill(F::R8G8_UNORM, 0x00FF_00FF).unwrap().float32 }, [1.0, 0.0, 0.0, 0.0]);
        assert!(decode_dword_fill(F::R8G8_UINT, 0x0001_0002).is_none());
        assert!(decode_dword_fill(F::R16_UINT, 0x0001_0002).is_none());
        assert!(decode_dword_fill(F::R16_SINT, 0x0001_0002).is_none());
        assert!(decode_dword_fill(F::R16_SNORM, 0x0001_0002).is_none());
        assert_eq!(unsafe { decode_dword_fill(F::R16_UINT, 0x1234_1234).unwrap().uint32 }[0], 0x1234);
    }
}
