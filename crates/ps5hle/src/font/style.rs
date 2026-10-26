// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Font and renderer style queries, scale in points and resolution, and
//! style frames (guest-owned FontStyleFrame blocks attached to surfaces).

use super::*;

impl Style {
    fn set_scale_point(&mut self, w: f32, h: f32) {
        self.scale_unit = 1;
        self.scale_w = w;
        self.scale_h = h;
    }

    /// A zero resolution selects the default 72 dpi.
    fn set_dpi(&mut self, h: u32, v: u32) {
        self.dpi_x = if h == 0 { DEFAULT_DPI } else { h };
        self.dpi_y = if v == 0 { DEFAULT_DPI } else { v };
    }

    /// Scale in points: pixel scales convert through the resolution.
    fn points(&self) -> (f32, f32) {
        if self.scale_unit == 0 {
            let fx = if self.dpi_x != 0 { POINTS_PER_INCH / self.dpi_x as f32 } else { 1.0 };
            let fy = if self.dpi_y != 0 { POINTS_PER_INCH / self.dpi_y as f32 } else { 1.0 };
            (self.scale_w * fx, self.scale_h * fy)
        } else {
            (self.scale_w, self.scale_h)
        }
    }
}

fn out_f32(ctx: &CpuContext, addr: u64, v: f32) {
    if addr != 0 {
        ctx.write_u32(addr, v.to_bits());
    }
}

fn out_u32(ctx: &CpuContext, addr: u64, v: u32) {
    if addr != 0 {
        ctx.write_u32(addr, v);
    }
}

/// Run `f` on the font's own style, or with `render` on the style of its
/// bound renderer.
fn with_style<R>(h: u64, render: bool, f: impl FnOnce(&mut Style) -> Result<R, HleResult>) -> Result<R, HleResult> {
    let mut s = state();
    let font = s.fonts.get_mut(&h).ok_or(ERROR_INVALID_FONT_HANDLE)?;
    if !render {
        return f(&mut font.style);
    }
    if font.renderer == 0 {
        return Err(ERROR_NOT_BOUND_RENDERER);
    }
    f(&mut font.render_style)
}

fn status(r: Result<(), HleResult>) -> HleResult {
    r.err().unwrap_or(0)
}

/// sceFontSetScalePoint(fontHandle, float w, float h)
pub(super) fn cb_set_scale_point(ctx: &mut CpuContext) -> HleResult {
    let (w, h) = (float_arg(0), float_arg(1));
    update_style(ctx.rdi(), |s| s.set_scale_point(w, h))
}

/// sceFontSetupRenderScalePoint(fontHandle, float w, float h)
pub(super) fn cb_setup_render_scale_point(ctx: &mut CpuContext) -> HleResult {
    let (w, h) = (float_arg(0), float_arg(1));
    update_render_style(ctx.rdi(), |s| {
        s.set_scale_point(w, h);
        0
    })
}

/// sceFontSetResolutionDpi(fontHandle, uint32 hDpi, uint32 vDpi)
pub(super) fn cb_set_resolution_dpi(ctx: &mut CpuContext) -> HleResult {
    let (hd, vd) = (ctx.rsi() as u32, ctx.rdx() as u32);
    update_style(ctx.rdi(), |s| s.set_dpi(hd, vd))
}

/// (w, h) scale query; both outputs read 0 on failure.
fn get_scale(ctx: &CpuContext, render: bool, point: bool) -> HleResult {
    let (h, w_out, h_out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let r = with_style(h, render, |s| {
        if w_out == 0 && h_out == 0 {
            return Err(ERROR_INVALID_PARAMETER);
        }
        Ok(if point { s.points() } else { s.pixels() })
    });
    let (w, hh) = r.unwrap_or((0.0, 0.0));
    out_f32(ctx, w_out, w);
    out_f32(ctx, h_out, hh);
    status(r.map(|_| ()))
}

/// sceFontGetScalePixel(fontHandle, float* w, float* h)
pub(super) fn cb_get_scale_pixel(ctx: &mut CpuContext) -> HleResult {
    get_scale(ctx, false, false)
}

/// sceFontGetScalePoint(fontHandle, float* w, float* h)
pub(super) fn cb_get_scale_point(ctx: &mut CpuContext) -> HleResult {
    get_scale(ctx, false, true)
}

/// sceFontGetRenderScalePixel(fontHandle, float* w, float* h)
pub(super) fn cb_get_render_scale_pixel(ctx: &mut CpuContext) -> HleResult {
    get_scale(ctx, true, false)
}

/// sceFontGetRenderScalePoint(fontHandle, float* w, float* h)
pub(super) fn cb_get_render_scale_point(ctx: &mut CpuContext) -> HleResult {
    get_scale(ctx, true, true)
}

/// sceFontGetResolutionDpi(fontHandle, uint32* hDpi, uint32* vDpi); both
/// outputs read 0 on failure.
pub(super) fn cb_get_resolution_dpi(ctx: &mut CpuContext) -> HleResult {
    let (h, h_out, v_out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let r = with_style(h, false, |s| {
        if h_out == 0 && v_out == 0 {
            return Err(ERROR_INVALID_PARAMETER);
        }
        Ok((s.dpi_x, s.dpi_y))
    });
    let (hd, vd) = r.unwrap_or((0, 0));
    out_u32(ctx, h_out, hd);
    out_u32(ctx, v_out, vd);
    status(r.map(|_| ()))
}

/// Slant query; the output reads 0 on failure.
fn get_slant(ctx: &CpuContext, render: bool) -> HleResult {
    let (h, out) = (ctx.rdi(), ctx.rsi());
    let r = with_style(h, render, |s| if out == 0 { Err(ERROR_INVALID_PARAMETER) } else { Ok(s.slant) });
    out_f32(ctx, out, r.unwrap_or(0.0));
    status(r.map(|_| ()))
}

/// sceFontGetEffectSlant(fontHandle, float* slantRatio)
pub(super) fn cb_get_effect_slant(ctx: &mut CpuContext) -> HleResult {
    get_slant(ctx, false)
}

/// sceFontGetRenderEffectSlant(fontHandle, float* slantRatio)
pub(super) fn cb_get_render_effect_slant(ctx: &mut CpuContext) -> HleResult {
    get_slant(ctx, true)
}

/// Weight query (scales are 1 + the stored delta, mode 0); the outputs
/// read 1, 1, 0 on failure.
fn get_weight(ctx: &CpuContext, render: bool) -> HleResult {
    let (h, wx_out, wy_out, mode_out) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    let r = with_style(h, render, |s| {
        if wx_out == 0 && wy_out == 0 && mode_out == 0 {
            return Err(ERROR_INVALID_PARAMETER);
        }
        Ok((s.weight_x + 1.0, s.weight_y + 1.0))
    });
    let (wx, wy) = r.unwrap_or((1.0, 1.0));
    out_f32(ctx, wx_out, wx);
    out_f32(ctx, wy_out, wy);
    out_u32(ctx, mode_out, 0);
    status(r.map(|_| ()))
}

/// sceFontGetEffectWeight(fontHandle, float* wx, float* wy, uint32* mode)
pub(super) fn cb_get_effect_weight(ctx: &mut CpuContext) -> HleResult {
    get_weight(ctx, false)
}

/// sceFontGetRenderEffectWeight(fontHandle, float* wx, float* wy, uint32* mode)
pub(super) fn cb_get_render_effect_weight(ctx: &mut CpuContext) -> HleResult {
    get_weight(ctx, true)
}

// ---------------------------------------------------------------------------
// Style frames
// ---------------------------------------------------------------------------

/// FontStyleFrame (0x60): magic u16, flags1 u8, flags2 u8, hDpi, vDpi,
/// scaleUnit, baseScale, scale w/h, weight deltas x/y, slant, then
/// library-private cache state.
const FRAME_SIZE: usize = 0x60;
const FRAME_FLAGS: u64 = 2;
const FRAME_HDPI: u64 = 4;
const FRAME_VDPI: u64 = 8;
const FRAME_SCALE_UNIT: u64 = 0xC;
const FRAME_SCALE_W: u64 = 0x14;
const FRAME_SCALE_H: u64 = 0x18;
const FRAME_WEIGHT_X: u64 = 0x1C;
const FRAME_WEIGHT_Y: u64 = 0x20;
const FRAME_SLANT: u64 = 0x24;

fn frame_ok(ctx: &CpuContext, f: u64) -> bool {
    f != 0 && read_u16(ctx, f) == Some(STYLE_FRAME_MAGIC)
}

fn frame_flags(ctx: &CpuContext, f: u64) -> u8 {
    ctx.read_bytes(f + FRAME_FLAGS, 1).map_or(0, |b| b[0])
}

fn set_frame_flags(ctx: &CpuContext, f: u64, set: u8, clear: u8) {
    let flags = (frame_flags(ctx, f) & !clear) | set;
    ctx.write_bytes(f + FRAME_FLAGS, &[flags]);
}

fn frame_f32(ctx: &CpuContext, f: u64, offset: u64) -> f32 {
    f32::from_bits(ctx.read_u32(f + offset).unwrap_or(0))
}

fn frame_u32(ctx: &CpuContext, f: u64, offset: u64) -> u32 {
    ctx.read_u32(f + offset).unwrap_or(0)
}

/// sceFontStyleFrameInit(FontStyleFrame*)
pub(super) fn cb_style_frame_init(ctx: &mut CpuContext) -> HleResult {
    let f = ctx.rdi();
    if f == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let mut b = [0u8; FRAME_SIZE];
    b[0..2].copy_from_slice(&STYLE_FRAME_MAGIC.to_le_bytes());
    b[4..8].copy_from_slice(&DEFAULT_DPI.to_le_bytes());
    b[8..12].copy_from_slice(&DEFAULT_DPI.to_le_bytes());
    if !ctx.write_bytes(f, &b) {
        return ERROR_INVALID_PARAMETER;
    }
    0
}

fn frame_set_scale(ctx: &CpuContext, unit: u32) -> HleResult {
    let f = ctx.rdi();
    if !frame_ok(ctx, f) {
        return ERROR_INVALID_PARAMETER;
    }
    ctx.write_u32(f + FRAME_SCALE_UNIT, unit);
    ctx.write_u32(f + FRAME_SCALE_W, float_arg(0).to_bits());
    ctx.write_u32(f + FRAME_SCALE_H, float_arg(1).to_bits());
    set_frame_flags(ctx, f, STYLE_FRAME_FLAG_SCALE, 0);
    0
}

/// sceFontStyleFrameSetScalePixel(FontStyleFrame*, float w, float h)
pub(super) fn cb_style_frame_set_scale_pixel(ctx: &mut CpuContext) -> HleResult {
    frame_set_scale(ctx, 0)
}

/// sceFontStyleFrameSetScalePoint(FontStyleFrame*, float w, float h)
pub(super) fn cb_style_frame_set_scale_point(ctx: &mut CpuContext) -> HleResult {
    frame_set_scale(ctx, 1)
}

/// sceFontStyleFrameSetResolutionDpi(FontStyleFrame*, uint32 hDpi, uint32 vDpi)
pub(super) fn cb_style_frame_set_resolution_dpi(ctx: &mut CpuContext) -> HleResult {
    let (f, hd, vd) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx() as u32);
    if !frame_ok(ctx, f) {
        return ERROR_INVALID_PARAMETER;
    }
    ctx.write_u32(f + FRAME_HDPI, if hd == 0 { DEFAULT_DPI } else { hd });
    ctx.write_u32(f + FRAME_VDPI, if vd == 0 { DEFAULT_DPI } else { vd });
    0
}

/// sceFontStyleFrameSetEffectSlant(FontStyleFrame*, float slantRatio)
pub(super) fn cb_style_frame_set_effect_slant(ctx: &mut CpuContext) -> HleResult {
    let f = ctx.rdi();
    if !frame_ok(ctx, f) {
        return ERROR_INVALID_PARAMETER;
    }
    ctx.write_u32(f + FRAME_SLANT, float_arg(0).clamp(-1.0, 1.0).to_bits());
    set_frame_flags(ctx, f, STYLE_FRAME_FLAG_SLANT, 0);
    0
}

/// sceFontStyleFrameSetEffectWeight(FontStyleFrame*, float wx, float wy, uint32 mode):
/// the scales are stored as deltas from 1 limited to +-0.04.
pub(super) fn cb_style_frame_set_effect_weight(ctx: &mut CpuContext) -> HleResult {
    let f = ctx.rdi();
    if !frame_ok(ctx, f) || ctx.rsi() as u32 != 0 {
        return ERROR_INVALID_PARAMETER;
    }
    ctx.write_u32(f + FRAME_WEIGHT_X, (float_arg(0) - 1.0).clamp(-0.04, 0.04).to_bits());
    ctx.write_u32(f + FRAME_WEIGHT_Y, (float_arg(1) - 1.0).clamp(-0.04, 0.04).to_bits());
    set_frame_flags(ctx, f, STYLE_FRAME_FLAG_WEIGHT, 0);
    0
}

fn frame_unset(ctx: &CpuContext, flag: u8) -> HleResult {
    let f = ctx.rdi();
    if !frame_ok(ctx, f) {
        return ERROR_INVALID_PARAMETER;
    }
    set_frame_flags(ctx, f, 0, flag);
    0
}

/// sceFontStyleFrameUnsetScale(FontStyleFrame*)
pub(super) fn cb_style_frame_unset_scale(ctx: &mut CpuContext) -> HleResult {
    frame_unset(ctx, STYLE_FRAME_FLAG_SCALE)
}

/// sceFontStyleFrameUnsetEffectSlant(FontStyleFrame*)
pub(super) fn cb_style_frame_unset_effect_slant(ctx: &mut CpuContext) -> HleResult {
    frame_unset(ctx, STYLE_FRAME_FLAG_SLANT)
}

/// sceFontStyleFrameUnsetEffectWeight(FontStyleFrame*)
pub(super) fn cb_style_frame_unset_effect_weight(ctx: &mut CpuContext) -> HleResult {
    frame_unset(ctx, STYLE_FRAME_FLAG_WEIGHT)
}

/// A frame valid with `flag` set (UNSET_PARAMETER otherwise).
fn frame_with(ctx: &CpuContext, f: u64, flag: u8) -> Result<(), HleResult> {
    if !frame_ok(ctx, f) {
        return Err(ERROR_INVALID_PARAMETER);
    }
    if frame_flags(ctx, f) & flag == 0 {
        return Err(ERROR_UNSET_PARAMETER);
    }
    Ok(())
}

fn frame_get_scale(ctx: &CpuContext, point: bool) -> HleResult {
    let (f, w_out, h_out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if let Err(e) = frame_with(ctx, f, STYLE_FRAME_FLAG_SCALE) {
        return e;
    }
    if w_out == 0 && h_out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let style = Style {
        dpi_x: frame_u32(ctx, f, FRAME_HDPI),
        dpi_y: frame_u32(ctx, f, FRAME_VDPI),
        scale_unit: frame_u32(ctx, f, FRAME_SCALE_UNIT),
        scale_w: frame_f32(ctx, f, FRAME_SCALE_W),
        scale_h: frame_f32(ctx, f, FRAME_SCALE_H),
        ..Style::new(0.0)
    };
    let (w, h) = if point { style.points() } else { style.pixels() };
    out_f32(ctx, w_out, w);
    out_f32(ctx, h_out, h);
    0
}

/// sceFontStyleFrameGetScalePixel(const FontStyleFrame*, float* w, float* h)
pub(super) fn cb_style_frame_get_scale_pixel(ctx: &mut CpuContext) -> HleResult {
    frame_get_scale(ctx, false)
}

/// sceFontStyleFrameGetScalePoint(const FontStyleFrame*, float* w, float* h)
pub(super) fn cb_style_frame_get_scale_point(ctx: &mut CpuContext) -> HleResult {
    frame_get_scale(ctx, true)
}

/// sceFontStyleFrameGetResolutionDpi(const FontStyleFrame*, uint32* hDpi, uint32* vDpi)
pub(super) fn cb_style_frame_get_resolution_dpi(ctx: &mut CpuContext) -> HleResult {
    let (f, h_out, v_out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if !frame_ok(ctx, f) || (h_out == 0 && v_out == 0) {
        return ERROR_INVALID_PARAMETER;
    }
    out_u32(ctx, h_out, frame_u32(ctx, f, FRAME_HDPI));
    out_u32(ctx, v_out, frame_u32(ctx, f, FRAME_VDPI));
    0
}

/// sceFontStyleFrameGetEffectSlant(const FontStyleFrame*, float* slantRatio)
pub(super) fn cb_style_frame_get_effect_slant(ctx: &mut CpuContext) -> HleResult {
    let (f, out) = (ctx.rdi(), ctx.rsi());
    if let Err(e) = frame_with(ctx, f, STYLE_FRAME_FLAG_SLANT) {
        return e;
    }
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    out_f32(ctx, out, frame_f32(ctx, f, FRAME_SLANT));
    0
}

/// sceFontStyleFrameGetEffectWeight(const FontStyleFrame*, float* wx, float* wy, uint32* mode)
pub(super) fn cb_style_frame_get_effect_weight(ctx: &mut CpuContext) -> HleResult {
    let (f, wx_out, wy_out, mode_out) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    if let Err(e) = frame_with(ctx, f, STYLE_FRAME_FLAG_WEIGHT) {
        return e;
    }
    if wx_out == 0 && wy_out == 0 && mode_out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    out_f32(ctx, wx_out, frame_f32(ctx, f, FRAME_WEIGHT_X) + 1.0);
    out_f32(ctx, wy_out, frame_f32(ctx, f, FRAME_WEIGHT_Y) + 1.0);
    out_u32(ctx, mode_out, 0);
    0
}

/// Surface style flag bit: a style frame is attached.
const SURFACE_STYLE_FRAME: u8 = 0x01;
const SURFACE_STYLE_FLAG: u64 = 0xE;
const SURFACE_SYSTEM_USE: u64 = 0x28;

/// sceFontRenderSurfaceSetStyleFrame(FontRenderSurface*, FontStyleFrame*):
/// attaches the frame (null detaches it).
pub(super) fn cb_render_surface_set_style_frame(ctx: &mut CpuContext) -> HleResult {
    let (s, f) = (ctx.rdi(), ctx.rsi());
    if s == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    if f != 0 && !frame_ok(ctx, f) {
        return ERROR_INVALID_PARAMETER;
    }
    let flag = ctx.read_bytes(s + SURFACE_STYLE_FLAG, 1).map_or(0, |b| b[0]);
    let flag = if f != 0 { flag | SURFACE_STYLE_FRAME } else { flag & !SURFACE_STYLE_FRAME };
    ctx.write_bytes(s + SURFACE_STYLE_FLAG, &[flag]);
    ctx.write_u64(s + SURFACE_SYSTEM_USE, f);
    ctx.write_u64(s + SURFACE_SYSTEM_USE + 8, 0);
    0
}

pub(super) const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("sw65+7wXCKE", "sceFontSetScalePoint", cb_set_scale_point),
    ("nMZid4oDfi4", "sceFontSetupRenderScalePoint", cb_setup_render_scale_point),
    ("I1acwR7Qp8E", "sceFontSetResolutionDpi", cb_set_resolution_dpi),
    ("CkVmLoCNN-8", "sceFontGetScalePixel", cb_get_scale_pixel),
    ("GoF2bhB7LYk", "sceFontGetScalePoint", cb_get_scale_point),
    ("EY38A01lq2k", "sceFontGetRenderScalePixel", cb_get_render_scale_pixel),
    ("FEafYUcxEGo", "sceFontGetRenderScalePoint", cb_get_render_scale_point),
    ("8REoLjNGCpM", "sceFontGetResolutionDpi", cb_get_resolution_dpi),
    ("ynSqYL8VpoA", "sceFontGetEffectSlant", cb_get_effect_slant),
    ("Gqa5Pp7y4MU", "sceFontGetRenderEffectSlant", cb_get_render_effect_slant),
    ("d7dDgRY+Bzw", "sceFontGetEffectWeight", cb_get_effect_weight),
    ("woOjHrkjIYg", "sceFontGetRenderEffectWeight", cb_get_render_effect_weight),
    ("la2AOWnHEAc", "sceFontStyleFrameInit", cb_style_frame_init),
    ("da4rQ4-+p-4", "sceFontStyleFrameSetScalePixel", cb_style_frame_set_scale_pixel),
    ("O997laxY-Ys", "sceFontStyleFrameSetScalePoint", cb_style_frame_set_scale_point),
    ("dB4-3Wdwls8", "sceFontStyleFrameSetResolutionDpi", cb_style_frame_set_resolution_dpi),
    ("394sckksiCU", "sceFontStyleFrameSetEffectSlant", cb_style_frame_set_effect_slant),
    ("faw77-pEBmU", "sceFontStyleFrameSetEffectWeight", cb_style_frame_set_effect_weight),
    ("bePC0L0vQWY", "sceFontStyleFrameUnsetScale", cb_style_frame_unset_scale),
    ("dUmABkAnVgk", "sceFontStyleFrameUnsetEffectSlant", cb_style_frame_unset_effect_slant),
    ("hwsuXgmKdaw", "sceFontStyleFrameUnsetEffectWeight", cb_style_frame_unset_effect_weight),
    ("2QfqfeLblbg", "sceFontStyleFrameGetScalePixel", cb_style_frame_get_scale_pixel),
    ("7x2xKiiB7MA", "sceFontStyleFrameGetScalePoint", cb_style_frame_get_scale_point),
    ("VSw18Aqzl0U", "sceFontStyleFrameGetResolutionDpi", cb_style_frame_get_resolution_dpi),
    ("lOfduYnjgbo", "sceFontStyleFrameGetEffectSlant", cb_style_frame_get_effect_slant),
    ("HIUdjR-+Wl8", "sceFontStyleFrameGetEffectWeight", cb_style_frame_get_effect_weight),
    ("0hr-w30SjiI", "sceFontRenderSurfaceSetStyleFrame", cb_render_surface_set_style_frame),
];
