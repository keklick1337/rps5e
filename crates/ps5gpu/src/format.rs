// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! VideoOut display pixel formats and their conversion to host BGRA8.
//!
//! The presenter shows every display buffer through a `B8G8R8A8_UNORM`
//! image. Display formats keep their encoded (gamma) values, so 8-bit and
//! 10-bit formats are passed through as stored; FP16 scan-out buffers hold
//! linear values and are sRGB-encoded here. BT.2100 PQ buffers are shown
//! with their code values (no tone mapping).

/// Legacy 32-bit display format codes.
pub const A8R8G8B8_SRGB: u64 = 0x8000_0000;
pub const A8B8G8R8_SRGB: u64 = 0x8000_2200;
pub const A2R10G10B10: u64 = 0x8806_0000;
pub const A2R10G10B10_SRGB: u64 = 0x8800_0000;
pub const A2R10G10B10_BT2020_PQ: u64 = 0x8874_0000;

/// 64-bit "format2" display codes. The 0x22000000 field selects R-first
/// component order; 0x81... is packed 10:10:10:2.
pub const FMT2_R8G8B8A8_SRGB: u64 = 0x8000_0000_2200_0000;
pub const FMT2_B8G8R8A8_SRGB: u64 = 0x8000_0000_0000_0000;
pub const FMT2_R10G10B10A2: u64 = 0x8100_0006_2200_0000;
pub const FMT2_B10G10R10A2: u64 = 0x8100_0006_0000_0000;
pub const FMT2_R10G10B10A2_SRGB: u64 = 0x8100_0000_2200_0000;
pub const FMT2_B10G10R10A2_SRGB: u64 = 0x8100_0000_0000_0000;
pub const FMT2_R10G10B10A2_BT2100_PQ: u64 = 0x8100_0704_2200_0000;
pub const FMT2_B10G10R10A2_BT2100_PQ: u64 = 0x8100_0704_0000_0000;
pub const FMT2_R16G16B16A16_FLOAT: u64 = 0xc001_0006_2200_0000;
pub const FMT2_B16G16R16A16_FLOAT: u64 = 0xc001_0006_0000_0000;

/// Bytes per pixel of the 32-bit formats known to the VideoOut layer
/// (FP16 codes are not part of this set; see [`display_bytes_per_pixel`]).
pub fn bytes_per_pixel_legacy(fmt: u64) -> u32 {
    match fmt {
        A8R8G8B8_SRGB | A8B8G8R8_SRGB | A2R10G10B10 | A2R10G10B10_SRGB | A2R10G10B10_BT2020_PQ
        | FMT2_R8G8B8A8_SRGB | FMT2_B8G8R8A8_SRGB | FMT2_R10G10B10A2 | FMT2_B10G10R10A2
        | FMT2_R10G10B10A2_SRGB | FMT2_B10G10R10A2_SRGB | FMT2_R10G10B10A2_BT2100_PQ
        | FMT2_B10G10R10A2_BT2100_PQ => 4,
        _ => 0,
    }
}

/// Canonicalise a pixel format: known codes stay, otherwise the low or high
/// 32 bits, or a repacked `high | low >> 16`, are tried.
pub fn normalize(fmt: u64) -> u64 {
    if bytes_per_pixel_legacy(fmt) != 0 {
        return fmt;
    }
    let low = fmt & 0xFFFF_FFFF;
    if bytes_per_pixel_legacy(low) != 0 {
        return low;
    }
    let high = fmt >> 32;
    if bytes_per_pixel_legacy(high) != 0 {
        return high;
    }
    let packed = high | (low >> 16);
    if bytes_per_pixel_legacy(packed) != 0 {
        packed
    } else {
        fmt
    }
}

/// Decoded layout of a display format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// 8-bit per channel, memory order R,G,B,A.
    Rgba8,
    /// 8-bit per channel, memory order B,G,R,A.
    Bgra8,
    /// 10:10:10:2 with red in the least significant bits.
    Packed10RedLow,
    /// 10:10:10:2 with blue in the least significant bits.
    Packed10BlueLow,
    /// Half floats, memory order R,G,B,A (linear).
    Rgba16F,
    /// Half floats, memory order B,G,R,A (linear).
    Bgra16F,
}

impl Layout {
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Layout::Rgba16F | Layout::Bgra16F => 8,
            _ => 4,
        }
    }
}

/// Layout of a VideoOut pixel format, or None when unsupported.
pub fn decode(fmt: u64) -> Option<Layout> {
    match fmt {
        FMT2_R16G16B16A16_FLOAT => return Some(Layout::Rgba16F),
        FMT2_B16G16R16A16_FLOAT => return Some(Layout::Bgra16F),
        _ => {}
    }
    Some(match normalize(fmt) {
        A8B8G8R8_SRGB | FMT2_R8G8B8A8_SRGB => Layout::Rgba8,
        A8R8G8B8_SRGB | FMT2_B8G8R8A8_SRGB => Layout::Bgra8,
        FMT2_R10G10B10A2 | FMT2_R10G10B10A2_SRGB | FMT2_R10G10B10A2_BT2100_PQ => Layout::Packed10RedLow,
        A2R10G10B10 | A2R10G10B10_SRGB | A2R10G10B10_BT2020_PQ | FMT2_B10G10R10A2
        | FMT2_B10G10R10A2_SRGB | FMT2_B10G10R10A2_BT2100_PQ => Layout::Packed10BlueLow,
        _ => return None,
    })
}

/// Bytes per pixel of a display format (0 when unsupported).
pub fn display_bytes_per_pixel(fmt: u64) -> usize {
    decode(fmt).map_or(0, Layout::bytes_per_pixel)
}

/// True for HDR (PQ) formats.
pub fn is_hdr(fmt: u64) -> bool {
    matches!(
        normalize(fmt),
        A2R10G10B10_BT2020_PQ | FMT2_R10G10B10A2_BT2100_PQ | FMT2_B10G10R10A2_BT2100_PQ
    )
}

/// 10-bit UNORM to 8-bit, round to nearest preserving both endpoints.
#[inline]
pub fn reduce_unorm10_to_8(v: u32) -> u8 {
    ((v * 255 + 511) / 1023) as u8
}

/// 8-bit UNORM to 10-bit.
#[inline]
pub fn expand_unorm8_to_10(v: u8) -> u32 {
    (v as u32 * 1023 + 127) / 255
}

/// Pack RGBA8 into a 10:10:10:2 word of `layout`.
pub fn pack_rgba8_to_10(layout: Layout, r: u8, g: u8, b: u8, a: u8) -> u32 {
    let (r, g, b) = (expand_unorm8_to_10(r), expand_unorm8_to_10(g), expand_unorm8_to_10(b));
    let a2 = (a as u32 * 3 + 127) / 255;
    match layout {
        Layout::Packed10RedLow => r | (g << 10) | (b << 20) | (a2 << 30),
        _ => b | (g << 10) | (r << 20) | (a2 << 30),
    }
}

/// Unpack a 10:10:10:2 word into RGBA8.
pub fn unpack_10_to_rgba8(layout: Layout, packed: u32) -> [u8; 4] {
    let least = packed & 0x3FF;
    let green = (packed >> 10) & 0x3FF;
    let most = (packed >> 20) & 0x3FF;
    let (r, b) = if layout == Layout::Packed10RedLow { (least, most) } else { (most, least) };
    [
        reduce_unorm10_to_8(r),
        reduce_unorm10_to_8(green),
        reduce_unorm10_to_8(b),
        ((((packed >> 30) & 3) * 255 + 1) / 3) as u8,
    ]
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal half: value = mant * 2^-24.
            let v = mant as f32 * (1.0 / 16_777_216.0);
            return if sign != 0 { -v } else { v };
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn linear_to_srgb8(v: f32) -> u8 {
    let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    let s = if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

/// Convert tightly packed `src` pixels of `layout` into BGRA8 (`dst`, 4
/// bytes per pixel). Alpha is forced opaque: scan-out ignores it.
pub fn convert_to_bgra8(layout: Layout, src: &[u8], dst: &mut [u8]) {
    let bpp = layout.bytes_per_pixel();
    let n = (src.len() / bpp).min(dst.len() / 4);
    for i in 0..n {
        let s = &src[i * bpp..i * bpp + bpp];
        let d = &mut dst[i * 4..i * 4 + 4];
        match layout {
            Layout::Bgra8 => {
                d[0] = s[0];
                d[1] = s[1];
                d[2] = s[2];
            }
            Layout::Rgba8 => {
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
            }
            Layout::Packed10RedLow | Layout::Packed10BlueLow => {
                let p = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
                let [r, g, b, _] = unpack_10_to_rgba8(layout, p);
                d[0] = b;
                d[1] = g;
                d[2] = r;
            }
            Layout::Rgba16F | Layout::Bgra16F => {
                let c = |k: usize| linear_to_srgb8(half_to_f32(u16::from_le_bytes([s[k * 2], s[k * 2 + 1]])));
                let (c0, c1, c2) = (c(0), c(1), c(2));
                if layout == Layout::Rgba16F {
                    d[0] = c2;
                    d[1] = c1;
                    d[2] = c0;
                } else {
                    d[0] = c0;
                    d[1] = c1;
                    d[2] = c2;
                }
            }
        }
        d[3] = 0xFF;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed10_self_checks() {
        let r = decode(FMT2_R10G10B10A2_SRGB).unwrap();
        assert_eq!(pack_rgba8_to_10(r, 255, 0, 0, 255), 0xC000_03FF);
        assert_eq!(unpack_10_to_rgba8(r, 0xC000_03FF), [255, 0, 0, 255]);
        let b = decode(FMT2_B10G10R10A2_SRGB).unwrap();
        assert_eq!(pack_rgba8_to_10(b, 255, 0, 0, 255), 0xFFF0_0000);
        assert_eq!(unpack_10_to_rgba8(b, 0xFFF0_0000), [255, 0, 0, 255]);
        assert_eq!(reduce_unorm10_to_8(0), 0);
        assert_eq!(reduce_unorm10_to_8(512), 128);
        assert_eq!(reduce_unorm10_to_8(1023), 255);
    }

    #[test]
    fn normalize_and_decode() {
        assert_eq!(decode(FMT2_R8G8B8A8_SRGB), Some(Layout::Rgba8));
        assert_eq!(decode(FMT2_B8G8R8A8_SRGB), Some(Layout::Bgra8));
        assert_eq!(decode(A8R8G8B8_SRGB), Some(Layout::Bgra8));
        assert_eq!(decode(A8B8G8R8_SRGB), Some(Layout::Rgba8));
        assert_eq!(decode(A2R10G10B10), Some(Layout::Packed10BlueLow));
        assert_eq!(decode(FMT2_R10G10B10A2_BT2100_PQ), Some(Layout::Packed10RedLow));
        assert_eq!(decode(FMT2_R16G16B16A16_FLOAT), Some(Layout::Rgba16F));
        assert_eq!(decode(FMT2_B16G16R16A16_FLOAT), Some(Layout::Bgra16F));
        // Legacy code stored in the high half of a 64-bit slot.
        assert_eq!(normalize(A8B8G8R8_SRGB << 32), A8B8G8R8_SRGB);
        assert_eq!(decode(0x1234), None);
        assert!(is_hdr(FMT2_B10G10R10A2_BT2100_PQ));
        assert!(!is_hdr(FMT2_B10G10R10A2_SRGB));
    }

    #[test]
    fn convert_pixels() {
        let mut out = [0u8; 4];
        convert_to_bgra8(Layout::Rgba8, &[10, 20, 30, 40], &mut out);
        assert_eq!(out, [30, 20, 10, 255]);
        convert_to_bgra8(Layout::Bgra8, &[10, 20, 30, 40], &mut out);
        assert_eq!(out, [10, 20, 30, 255]);
        convert_to_bgra8(Layout::Packed10RedLow, &0xC000_03FFu32.to_le_bytes(), &mut out);
        assert_eq!(out, [0, 0, 255, 255]);
        convert_to_bgra8(Layout::Packed10BlueLow, &0xFFF0_0000u32.to_le_bytes(), &mut out);
        assert_eq!(out, [0, 0, 255, 255]);
        // 1.0, 0.5, 0.0 halves.
        let px: Vec<u8> = [0x3C00u16, 0x3800, 0x0000, 0x3C00].iter().flat_map(|h| h.to_le_bytes()).collect();
        convert_to_bgra8(Layout::Rgba16F, &px, &mut out);
        assert_eq!(out, [0, 188, 255, 255]);
        convert_to_bgra8(Layout::Bgra16F, &px, &mut out);
        assert_eq!(out, [255, 188, 0, 255]);
    }
}
