// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! HDR10 output: the swapchain format choice and the SDR-to-HDR10 colour
//! conversion of everything drawn over an HDR game image.
//!
//! With `RPS5E_HDR=1` ([`crate::display::hdr_enabled`]) the window looks
//! for an HDR10 surface format: a 10:10:10:2 UNORM format in the
//! `HDR10_ST2084` colour space (BT.2020 primaries, SMPTE ST.2084 "PQ"
//! encoding). It is used while the guest flips BT.2100 PQ display buffers;
//! their code values are already HDR10 and are copied to the swapchain
//! unchanged. SDR flips, and every frame when HDR is off or the display
//! has no HDR10 format, use the SDR swapchain as before.
//!
//! The emulator's own overlays (on-screen hints, performance overlay,
//! in-game menu) are sRGB. Over an HDR10 swapchain they are converted with
//! [`sdr_to_hdr10`]: sRGB decode, BT.709 to BT.2020 primaries, SDR white at
//! [`REFERENCE_WHITE_NITS`], PQ encode. The shaders emit the same math
//! ([`emit_sdr_to_hdr10`]).

use ash::vk;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::GlslStd450Op;

/// Luminance SDR white is shown at, in nits.
pub const REFERENCE_WHITE_NITS: f32 = 250.0;
/// Luminance of PQ code value 1.0, in nits.
pub const PQ_PEAK_NITS: f32 = 10_000.0;

/// Linear BT.709 RGB to linear BT.2020 RGB (rows: output R, G, B).
pub const BT709_TO_BT2020: [[f32; 3]; 3] = [
    [0.627404, 0.329283, 0.043313],
    [0.069097, 0.919540, 0.011362],
    [0.016391, 0.088013, 0.895595],
];

/// SMPTE ST.2084 constants.
const PQ_M1: f32 = 2610.0 / 16384.0;
const PQ_M2: f32 = 2523.0 / 32.0;
const PQ_C1: f32 = 3424.0 / 4096.0;
const PQ_C2: f32 = 2413.0 / 128.0;
const PQ_C3: f32 = 2392.0 / 128.0;

/// sRGB transfer function, decode (`v` in 0..=1).
pub fn srgb_to_linear(v: f32) -> f32 {
    if v > 0.04045 {
        ((v + 0.055) / 1.055).powf(2.4)
    } else {
        v / 12.92
    }
}

/// SMPTE ST.2084 (PQ) encode: `l` is linear light relative to
/// [`PQ_PEAK_NITS`] (1.0 = 10,000 nits); the result is the 0..=1 signal.
pub fn pq_encode(l: f32) -> f32 {
    let p = l.max(0.0).powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * p) / (1.0 + PQ_C3 * p)).powf(PQ_M2)
}

/// Linear BT.709 to linear BT.2020.
pub fn bt709_to_bt2020(rgb: [f32; 3]) -> [f32; 3] {
    let m = &BT709_TO_BT2020;
    [0, 1, 2].map(|i| m[i][0] * rgb[0] + m[i][1] * rgb[1] + m[i][2] * rgb[2])
}

/// An SDR colour (sRGB-encoded, or linear with `source_linear`) as HDR10
/// signal values: BT.2020 primaries, PQ, SDR white at
/// [`REFERENCE_WHITE_NITS`]. Inputs are clamped to 0..=1.
pub fn sdr_to_hdr10(rgb: [f32; 3], source_linear: bool) -> [f32; 3] {
    let rgb = rgb.map(|v| if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) });
    let linear = if source_linear { rgb } else { rgb.map(srgb_to_linear) };
    let scale = REFERENCE_WHITE_NITS / PQ_PEAK_NITS;
    bt709_to_bt2020(linear).map(|v| pq_encode(v * scale))
}

/// The 10:10:10:2 UNORM formats an HDR10 swapchain may use.
pub fn is_hdr10_format(format: vk::Format) -> bool {
    matches!(format, vk::Format::A2B10G10R10_UNORM_PACK32 | vk::Format::A2R10G10B10_UNORM_PACK32)
}

/// Pack 0..=1 colour channels and an 8-bit alpha into a 10:10:10:2 texel of
/// `format` (None for other formats).
pub fn pack_unorm10(format: vk::Format, rgb: [f32; 3], alpha: u8) -> Option<u32> {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 1023.0 + 0.5) as u32;
    let (r, g, b) = (q(rgb[0]), q(rgb[1]), q(rgb[2]));
    let a = (alpha as u32 * 3 + 127) / 255;
    match format {
        vk::Format::A2B10G10R10_UNORM_PACK32 => Some(r | (g << 10) | (b << 20) | (a << 30)),
        vk::Format::A2R10G10B10_UNORM_PACK32 => Some(b | (g << 10) | (r << 20) | (a << 30)),
        _ => None,
    }
}

/// The swapchain surface format and HDR state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceChoice {
    pub format: vk::SurfaceFormatKHR,
    /// HDR output is enabled and the surface offers a usable HDR10 format.
    pub hdr_supported: bool,
    /// `format` is that HDR10 format (an HDR frame asked for it).
    pub hdr_active: bool,
}

/// SDR formats in order of preference.
const SDR_FORMATS: [vk::Format; 3] = [vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM, vk::Format::A2B10G10R10_UNORM_PACK32];

/// The SDR surface format: the preferred formats in the sRGB colour space,
/// otherwise in any colour space, otherwise the first one offered.
fn sdr_surface_format(formats: &[vk::SurfaceFormatKHR]) -> Option<vk::SurfaceFormatKHR> {
    let find = |srgb_only: bool| {
        SDR_FORMATS.iter().find_map(|want| {
            formats
                .iter()
                .find(|f| f.format == *want && (!srgb_only || f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR))
                .copied()
        })
    };
    find(true).or_else(|| find(false)).or_else(|| formats.first().copied())
}

/// Choose the swapchain format. `hdr_enabled`: the HDR setting;
/// `hdr_requested`: the frames shown are HDR (PQ); `usable`: the format can
/// be a blit destination of the game image. HDR10 is the first offered
/// 10:10:10:2 UNORM format in the `HDR10_ST2084` colour space.
pub fn choose_surface_format(
    formats: &[vk::SurfaceFormatKHR],
    hdr_enabled: bool,
    hdr_requested: bool,
    usable: impl Fn(vk::Format) -> bool,
) -> Option<SurfaceChoice> {
    let hdr = formats
        .iter()
        .find(|f| f.color_space == vk::ColorSpaceKHR::HDR10_ST2084_EXT && is_hdr10_format(f.format) && usable(f.format))
        .copied()
        .filter(|_| hdr_enabled);
    let hdr_supported = hdr.is_some();
    match hdr {
        Some(format) if hdr_requested => Some(SurfaceChoice { format, hdr_supported, hdr_active: true }),
        _ => sdr_surface_format(formats).map(|format| SurfaceChoice { format, hdr_supported, hdr_active: false }),
    }
}

type R<T> = Result<T, rspirv::dr::Error>;

fn splat(b: &mut Builder, v: f32) -> u32 {
    let f32t = b.type_float(32, None);
    let vec3 = b.type_vector(f32t, 3);
    let c = b.constant_bit32(f32t, v.to_bits());
    b.constant_composite(vec3, [c, c, c])
}

fn vec3_const(b: &mut Builder, v: [f32; 3]) -> u32 {
    let f32t = b.type_float(32, None);
    let vec3 = b.type_vector(f32t, 3);
    let c = v.map(|x| b.constant_bit32(f32t, x.to_bits()));
    b.constant_composite(vec3, c)
}

fn glsl3(b: &mut Builder, glsl: u32, op: GlslStd450Op, args: &[u32]) -> R<u32> {
    let f32t = b.type_float(32, None);
    let vec3 = b.type_vector(f32t, 3);
    b.ext_inst(vec3, None, glsl, op as u32, args.iter().map(|&a| Operand::IdRef(a)))
}

/// Emit [`sdr_to_hdr10`] for the `vec3` value `rgb` inside the current
/// block; `glsl` is the module's GLSL.std.450 import. Returns the `vec3`.
pub(crate) fn emit_sdr_to_hdr10(b: &mut Builder, glsl: u32, rgb: u32, source_linear: bool) -> R<u32> {
    let f32t = b.type_float(32, None);
    let vec3 = b.type_vector(f32t, 3);
    let zero = splat(b, 0.0);
    let one = splat(b, 1.0);
    let rgb = glsl3(b, glsl, GlslStd450Op::FClamp, &[rgb, zero, one])?;
    let linear = if source_linear {
        rgb
    } else {
        let k = splat(b, 12.92);
        let low = b.f_div(vec3, None, rgb, k)?;
        let k = splat(b, 0.055);
        let t = b.f_add(vec3, None, rgb, k)?;
        let k = splat(b, 1.055);
        let t = b.f_div(vec3, None, t, k)?;
        let k = splat(b, 2.4);
        let high = glsl3(b, glsl, GlslStd450Op::Pow, &[t, k])?;
        let boolt = b.type_bool();
        let bvec3 = b.type_vector(boolt, 3);
        let k = splat(b, 0.04045);
        let cond = b.f_ord_greater_than(bvec3, None, rgb, k)?;
        b.select(vec3, None, cond, high, low)?
    };
    // Primaries and the reference white relative to the PQ peak.
    let scale = REFERENCE_WHITE_NITS / PQ_PEAK_NITS;
    let mut comps = [0u32; 3];
    for (i, row) in BT709_TO_BT2020.iter().enumerate() {
        let r = vec3_const(b, row.map(|v| v * scale));
        comps[i] = b.dot(f32t, None, r, linear)?;
    }
    let v = b.composite_construct(vec3, None, comps)?;
    let k = splat(b, PQ_M1);
    let powered = glsl3(b, glsl, GlslStd450Op::Pow, &[v, k])?;
    let k = splat(b, PQ_C2);
    let num = b.f_mul(vec3, None, k, powered)?;
    let k = splat(b, PQ_C1);
    let num = b.f_add(vec3, None, k, num)?;
    let k = splat(b, PQ_C3);
    let den = b.f_mul(vec3, None, k, powered)?;
    let den = b.f_add(vec3, None, one, den)?;
    let q = b.f_div(vec3, None, num, den)?;
    let k = splat(b, PQ_M2);
    glsl3(b, glsl, GlslStd450Op::Pow, &[q, k])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn pq_matches_st2084_reference_values() {
        // Signal values of the ST.2084 / BT.2100 PQ curve.
        for (nits, signal) in [
            (0.0, 0.000_000_73),
            (100.0, 0.508_078),
            (203.0, 0.580_689),
            (250.0, 0.602_559),
            (1000.0, 0.751_827),
            (10_000.0, 1.0),
        ] {
            let got = pq_encode(nits / PQ_PEAK_NITS);
            assert!(close(got, signal, 2e-5), "{nits} nits: {got} != {signal}");
        }
        // 10-bit code values of the same points.
        let code = |nits: f32| (pq_encode(nits / PQ_PEAK_NITS) * 1023.0).round() as u32;
        assert_eq!([code(100.0), code(203.0), code(1000.0), code(10_000.0)], [520, 594, 769, 1023]);
        // Monotonic, negative input clamps to black.
        assert!(pq_encode(0.02) > pq_encode(0.01));
        assert_eq!(pq_encode(-1.0), pq_encode(0.0));
    }

    #[test]
    fn bt709_to_bt2020_matrix() {
        // White stays white (rows sum to 1).
        for row in BT709_TO_BT2020 {
            assert!(close(row.iter().sum::<f32>(), 1.0, 1e-5), "{row:?}");
        }
        // Pure BT.709 primaries in BT.2020.
        assert_eq!(bt709_to_bt2020([1.0, 0.0, 0.0]), [0.627404, 0.069097, 0.016391]);
        assert_eq!(bt709_to_bt2020([0.0, 1.0, 0.0]), [0.329283, 0.919540, 0.088013]);
        assert_eq!(bt709_to_bt2020([0.0, 0.0, 1.0]), [0.043313, 0.011362, 0.895595]);
    }

    #[test]
    fn sdr_white_lands_at_the_reference_luminance() {
        let white = sdr_to_hdr10([1.0, 1.0, 1.0], false);
        for c in white {
            assert!(close(c, pq_encode(REFERENCE_WHITE_NITS / PQ_PEAK_NITS), 1e-5));
            assert!(close(c, 0.602_559, 2e-5));
        }
        assert!(sdr_to_hdr10([0.0; 3], false).iter().all(|&c| c < 1e-5));
        // Out-of-range and NaN inputs clamp.
        assert_eq!(sdr_to_hdr10([2.0, 2.0, 2.0], true), sdr_to_hdr10([1.0; 3], true));
        assert_eq!(sdr_to_hdr10([f32::NAN, -1.0, 0.0], true), sdr_to_hdr10([0.0; 3], true));
    }

    #[test]
    fn sdr_primaries_and_grey_in_hdr10() {
        // 10-bit codes (BT.2020 PQ) of sRGB red, green, blue and mid grey.
        let codes = |rgb: [f32; 3]| sdr_to_hdr10(rgb, false).map(|v| (v * 1023.0).round() as u32);
        assert_eq!(codes([1.0, 0.0, 0.0]), [567, 352, 239]);
        assert_eq!(codes([0.0, 1.0, 0.0]), [500, 607, 374]);
        assert_eq!(codes([0.0, 0.0, 1.0]), [313, 215, 605]);
        assert_eq!(codes([0.5, 0.5, 0.5]), [457, 457, 457]);
        // A linear source skips the sRGB decode.
        let lin = srgb_to_linear(0.5);
        assert_eq!(sdr_to_hdr10([lin; 3], true).map(|v| (v * 1023.0).round() as u32), [457, 457, 457]);
        assert!(close(srgb_to_linear(0.04045), 0.04045 / 12.92, 1e-7));
        assert!(close(srgb_to_linear(1.0), 1.0, 1e-6));
    }

    #[test]
    fn unorm10_packing() {
        let a2b = vk::Format::A2B10G10R10_UNORM_PACK32;
        let a2r = vk::Format::A2R10G10B10_UNORM_PACK32;
        assert_eq!(pack_unorm10(a2b, [1.0, 0.0, 0.0], 255), Some(0xC000_03FF));
        assert_eq!(pack_unorm10(a2r, [1.0, 0.0, 0.0], 255), Some(0xFFF0_0000));
        assert_eq!(pack_unorm10(a2b, [0.0, 0.0, 0.0], 0), Some(0));
        assert_eq!(pack_unorm10(vk::Format::B8G8R8A8_UNORM, [1.0; 3], 255), None);
        assert!(is_hdr10_format(a2b) && is_hdr10_format(a2r) && !is_hdr10_format(vk::Format::B8G8R8A8_UNORM));
    }

    fn sf(format: vk::Format, color_space: vk::ColorSpaceKHR) -> vk::SurfaceFormatKHR {
        vk::SurfaceFormatKHR { format, color_space }
    }

    const SRGB: vk::ColorSpaceKHR = vk::ColorSpaceKHR::SRGB_NONLINEAR;
    const HDR10: vk::ColorSpaceKHR = vk::ColorSpaceKHR::HDR10_ST2084_EXT;

    #[test]
    fn sdr_surface_without_hdr_support() {
        let formats = [sf(vk::Format::R8G8B8A8_UNORM, SRGB), sf(vk::Format::B8G8R8A8_UNORM, SRGB)];
        for (enabled, requested) in [(false, false), (false, true), (true, false), (true, true)] {
            let c = choose_surface_format(&formats, enabled, requested, |_| true).unwrap();
            // B8G8R8A8 is preferred, as before HDR existed.
            assert_eq!(c.format, sf(vk::Format::B8G8R8A8_UNORM, SRGB));
            assert!(!c.hdr_supported && !c.hdr_active);
        }
        assert!(choose_surface_format(&[], true, true, |_| true).is_none());
    }

    #[test]
    fn hdr10_surface_when_enabled_and_requested() {
        let formats = [
            sf(vk::Format::B8G8R8A8_UNORM, SRGB),
            sf(vk::Format::A2B10G10R10_UNORM_PACK32, SRGB),
            sf(vk::Format::B8G8R8A8_UNORM, HDR10),
            sf(vk::Format::A2B10G10R10_UNORM_PACK32, HDR10),
            sf(vk::Format::A2R10G10B10_UNORM_PACK32, HDR10),
        ];
        let c = choose_surface_format(&formats, true, true, |_| true).unwrap();
        assert_eq!(c, SurfaceChoice { format: sf(vk::Format::A2B10G10R10_UNORM_PACK32, HDR10), hdr_supported: true, hdr_active: true });
        // An SDR frame: SDR swapchain, HDR still reported as supported.
        let c = choose_surface_format(&formats, true, false, |_| true).unwrap();
        assert_eq!(c, SurfaceChoice { format: sf(vk::Format::B8G8R8A8_UNORM, SRGB), hdr_supported: true, hdr_active: false });
        // HDR off: the HDR10 formats are never chosen nor reported.
        let c = choose_surface_format(&formats, false, true, |_| true).unwrap();
        assert_eq!(c, SurfaceChoice { format: sf(vk::Format::B8G8R8A8_UNORM, SRGB), hdr_supported: false, hdr_active: false });
        // The first usable HDR10 format wins.
        let c = choose_surface_format(&formats, true, true, |f| f != vk::Format::A2B10G10R10_UNORM_PACK32).unwrap();
        assert_eq!(c.format, sf(vk::Format::A2R10G10B10_UNORM_PACK32, HDR10));
    }

    #[test]
    fn hdr_falls_back_to_sdr_without_a_usable_hdr10_format() {
        // 8-bit HDR10 entries and a format that cannot be blitted to do not count.
        let formats = [sf(vk::Format::B8G8R8A8_UNORM, HDR10), sf(vk::Format::A2B10G10R10_UNORM_PACK32, HDR10), sf(vk::Format::R8G8B8A8_UNORM, SRGB)];
        let c = choose_surface_format(&formats, true, true, |f| f != vk::Format::A2B10G10R10_UNORM_PACK32).unwrap();
        assert_eq!(c, SurfaceChoice { format: sf(vk::Format::R8G8B8A8_UNORM, SRGB), hdr_supported: false, hdr_active: false });
        // Other HDR colour spaces are not HDR10.
        let formats = [sf(vk::Format::A2B10G10R10_UNORM_PACK32, vk::ColorSpaceKHR::HDR10_HLG_EXT), sf(vk::Format::B8G8R8A8_UNORM, SRGB)];
        let c = choose_surface_format(&formats, true, true, |_| true).unwrap();
        assert!(!c.hdr_supported && !c.hdr_active);
        assert_eq!(c.format.color_space, SRGB);
    }

    #[test]
    fn sdr_choice_prefers_srgb_colour_space() {
        // With the colour-space extension a preferred format may also be
        // listed in other colour spaces first: SDR keeps sRGB.
        let formats = [sf(vk::Format::B8G8R8A8_UNORM, vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT), sf(vk::Format::B8G8R8A8_UNORM, SRGB)];
        assert_eq!(choose_surface_format(&formats, true, false, |_| true).unwrap().format, formats[1]);
        // Only non-sRGB entries: the old order-of-preference choice.
        let formats = [sf(vk::Format::R16G16B16A16_SFLOAT, vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT), sf(vk::Format::R8G8B8A8_UNORM, vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT)];
        assert_eq!(choose_surface_format(&formats, false, false, |_| true).unwrap().format, formats[1]);
        // Nothing preferred: the first format.
        let formats = [sf(vk::Format::R16G16B16A16_SFLOAT, SRGB)];
        assert_eq!(choose_surface_format(&formats, false, false, |_| true).unwrap().format, formats[0]);
    }

    #[test]
    fn spirv_conversion_assembles() {
        use rspirv::binary::Assemble;
        use rspirv::spirv;
        for linear in [false, true] {
            let mut b = Builder::new();
            b.set_version(1, 3);
            b.capability(spirv::Capability::Shader);
            b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
            let glsl = b.ext_inst_import("GLSL.std.450");
            let void = b.type_void();
            let f32t = b.type_float(32, None);
            let vec3 = b.type_vector(f32t, 3);
            let fnty = b.type_function(void, []);
            let main = b.begin_function(void, None, spirv::FunctionControl::NONE, fnty).unwrap();
            b.begin_block(None).unwrap();
            let input = vec3_const(&mut b, [1.0, 0.5, 0.0]);
            let out = emit_sdr_to_hdr10(&mut b, glsl, input, linear).unwrap();
            let _ = (out, vec3);
            b.ret().unwrap();
            b.end_function().unwrap();
            b.entry_point(spirv::ExecutionModel::GLCompute, main, "main", []);
            b.execution_mode(main, spirv::ExecutionMode::LocalSize, [1, 1, 1]);
            let words = b.module().assemble();
            let module = rspirv::dr::load_words(&words).unwrap();
            assert_eq!(module.entry_points.len(), 1);
        }
    }
}
