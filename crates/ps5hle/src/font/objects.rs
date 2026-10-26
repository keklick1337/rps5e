// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Library device cache and pixel resolution, renderer outline buffers,
//! font file opening and font queries, render-scaled kerning and the
//! metrics / outline views of generated glyphs.

use super::*;

// ---------------------------------------------------------------------------
// Library
// ---------------------------------------------------------------------------

/// Sub-pixel positions per pixel of the font engine.
const PIXEL_RESOLUTION: u32 = 64;

/// sceFontGetPixelResolution(library, uint32* subPixelCount)
pub(super) fn cb_get_pixel_resolution(ctx: &mut CpuContext) -> HleResult {
    let (lib, out) = (ctx.rdi(), ctx.rsi());
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    ctx.write_u32(out, 0);
    if !library_ok(ctx, lib) || ctx.read_u64(lib + 0x80).unwrap_or(0) == 0 {
        return ERROR_INVALID_LIBRARY;
    }
    ctx.write_u32(out, PIXEL_RESOLUTION);
    0
}

/// sceFontSetFontsOpenMode(library, uint32 openMode)
pub(super) fn cb_set_fonts_open_mode(ctx: &mut CpuContext) -> HleResult {
    if !library_ok(ctx, ctx.rdi()) {
        return ERROR_INVALID_LIBRARY;
    }
    if ctx.rsi() as u32 > 2 {
        return ERROR_INVALID_PARAMETER;
    }
    0
}

/// sceFontClearDeviceCache(library): marks every cache page free.
pub(super) fn cb_clear_device_cache(ctx: &mut CpuContext) -> HleResult {
    let lib = ctx.rdi();
    if !library_ok(ctx, lib) {
        return ERROR_INVALID_LIBRARY;
    }
    let header = state().libraries.get(&lib).map_or(0, |l| l.device_cache);
    if header == 0 {
        return ERROR_NOT_ATTACHED_CACHE_BUFFER;
    }
    // { size, pageCount, usedPages, freePages, ... }
    let pages = ctx.read_u32(header + 4).unwrap_or(0);
    ctx.write_u32(header + 12, pages);
    ctx.write_u32(header + 8, 0);
    0
}

/// sceFontDettachDeviceCacheBuffer(library, void** buffer, uint32* size):
/// hands the buffer back (a library-allocated one is released when the
/// caller does not take it).
pub(super) fn cb_dettach_device_cache_buffer(ctx: &mut CpuContext) -> HleResult {
    let (lib, buffer_out, size_out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if !library_ok(ctx, lib) {
        return ERROR_INVALID_LIBRARY;
    }
    let (header, owned) = {
        let mut s = state();
        let l = s.libraries.get_mut(&lib).unwrap();
        let taken = (l.device_cache, l.device_cache_owned);
        l.device_cache = 0;
        l.device_cache_owned = false;
        taken
    };
    if header == 0 {
        return ERROR_NOT_ATTACHED_CACHE_BUFFER;
    }
    if size_out != 0 {
        ctx.write_u32(size_out, ctx.read_u32(header).unwrap_or(0));
    }
    if buffer_out != 0 {
        ctx.write_u64(buffer_out, header);
    } else if owned {
        gfree(header);
    }
    0
}

// ---------------------------------------------------------------------------
// Renderer outline buffer
// ---------------------------------------------------------------------------

/// sceFontRendererGetOutlineBufferSize(renderer, uint32* size)
pub(super) fn cb_renderer_get_outline_buffer_size(ctx: &mut CpuContext) -> HleResult {
    let (r, out) = (ctx.rdi(), ctx.rsi());
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    ctx.write_u32(out, 0);
    if !renderer_ok(ctx, r) {
        return ERROR_INVALID_RENDERER;
    }
    let size = state().renderers.get(&r).map_or(0, |w| w.workspace_size);
    ctx.write_u32(out, size);
    0
}

/// sceFontRendererResetOutlineBuffer(renderer): clears the workspace.
pub(super) fn cb_renderer_reset_outline_buffer(ctx: &mut CpuContext) -> HleResult {
    let r = ctx.rdi();
    if !renderer_ok(ctx, r) {
        return ERROR_INVALID_RENDERER;
    }
    let (workspace, size) = state().renderers.get(&r).map_or((0, 0), |w| (w.workspace, w.workspace_size));
    if workspace != 0 && size != 0 {
        ctx.write_bytes(workspace, &vec![0u8; size as usize]);
    }
    0
}

/// sceFontRendererSetOutlineBufferPolicy(renderer, bufferPolicy, uint32 basalSize, uint32 limitSize):
/// grows the workspace to `basalSize`, never beyond a non-zero `limitSize`.
pub(super) fn cb_renderer_set_outline_buffer_policy(ctx: &mut CpuContext) -> HleResult {
    let (r, basal, limit) = (ctx.rdi(), ctx.rdx() as u32, ctx.rcx() as u32);
    if !renderer_ok(ctx, r) {
        return ERROR_INVALID_RENDERER;
    }
    if limit != 0 && basal > limit {
        return ERROR_INVALID_PARAMETER;
    }
    let (workspace, size) = state().renderers.get(&r).map_or((0, 0), |w| (w.workspace, w.workspace_size));
    let mut desired = size.max(basal);
    if limit != 0 {
        desired = desired.min(limit);
    }
    if desired == 0 {
        desired = RENDERER_WORKSPACE_SIZE;
    }
    if workspace == 0 || size != desired {
        let Some(new) = galloc(ctx, desired as u64) else {
            return ERROR_ALLOCATION_FAILED;
        };
        if workspace != 0 {
            gfree(workspace);
        }
        if let Some(w) = state().renderers.get_mut(&r) {
            w.workspace = new;
            w.workspace_size = desired;
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Fonts
// ---------------------------------------------------------------------------

/// sceFontOpenFontFile(library, const char* path, uint32 openMode,
/// const FontOpenDetail*, FontHandle*)
pub(super) fn cb_open_font_file(ctx: &mut CpuContext) -> HleResult {
    let (lib, path, mode, detail, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u32, ctx.rcx(), ctx.r8());
    let fail = |ctx: &CpuContext, rc: HleResult| {
        if out != 0 {
            ctx.write_u64(out, 0);
        }
        rc
    };
    if lib == 0 {
        return fail(ctx, ERROR_INVALID_LIBRARY);
    }
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let guest_path = if path != 0 { ctx.read_cstring(path, 1024) } else { None };
    let host = guest_path.as_deref().and_then(crate::filesystem::resolve_guest_path);
    if path != 0 && host.is_none() {
        return fail(ctx, ERROR_FS_OPEN_FAILED);
    }
    if !library_ok(ctx, lib) {
        return fail(ctx, ERROR_INVALID_LIBRARY);
    }
    if !state().libraries[&lib].external_fonts {
        return fail(ctx, ERROR_NO_SUPPORT_FUNCTION);
    }
    let (sub_font, unique_id) = if detail != 0 {
        (ctx.read_u32(detail + 8).unwrap_or(0), ctx.read_u32(detail + 12).unwrap_or(0) as i32)
    } else {
        (0, -1)
    };
    let Some(host) = host.filter(|h| !h.as_os_str().is_empty()) else {
        return fail(ctx, ERROR_INVALID_PARAMETER);
    };
    if unique_id < -1 || !(1..=3).contains(&(mode & 0xF)) {
        return fail(ctx, ERROR_INVALID_PARAMETER);
    }
    let Ok(bytes) = std::fs::read(&host) else {
        return fail(ctx, ERROR_FS_OPEN_FAILED);
    };
    let Some(face) = FaceData::parse(bytes, sub_font) else {
        return fail(ctx, ERROR_NO_SUPPORT_FORMAT);
    };
    match install_font(ctx, lib, 0, face, 0) {
        Ok(h) => {
            ctx.write_u64(out, h);
            0
        }
        Err(e) => fail(ctx, e),
    }
}

/// sceFontGetLibrary(fontHandle, FontLibrary*)
pub(super) fn cb_get_library(ctx: &mut CpuContext) -> HleResult {
    let (h, out) = (ctx.rdi(), ctx.rsi());
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    match font(h) {
        Some(f) => {
            ctx.write_u64(out, f.library);
            0
        }
        None => {
            ctx.write_u64(out, 0);
            ERROR_INVALID_FONT_HANDLE
        }
    }
}

/// sceFontGetFontGlyphsCount(fontHandle, uint32* glyphsCount)
pub(super) fn cb_get_font_glyphs_count(ctx: &mut CpuContext) -> HleResult {
    let (h, out) = (ctx.rdi(), ctx.rsi());
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    ctx.write_u32(out, 0);
    let Some(f) = font(h) else {
        return ERROR_INVALID_FONT_HANDLE;
    };
    ctx.write_u32(out, f.face.glyph_count() as u32);
    0
}

/// sceFontGetCharGlyphCode(fontHandle, code, uint32* glyphCode)
pub(super) fn cb_get_char_glyph_code(ctx: &mut CpuContext) -> HleResult {
    let (h, code, out) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx());
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    ctx.write_u32(out, 0);
    let Some(f) = font(h) else {
        return ERROR_INVALID_FONT_HANDLE;
    };
    if code == 0 {
        return ERROR_NO_SUPPORT_CODE;
    }
    match f.face.glyph_index(code) {
        Some(g) => {
            ctx.write_u32(out, g as u32);
            0
        }
        None => ERROR_NO_SUPPORT_GLYPH,
    }
}

/// sceFontGetFontResolution(fontHandle, uint32* unitsPerEm, float* scalePixel):
/// the face's design units and its default pixel scale (units / 64).
pub(super) fn cb_get_font_resolution(ctx: &mut CpuContext) -> HleResult {
    let (h, res_out, scale_out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let r = match font(h) {
        None => Err(ERROR_INVALID_FONT_HANDLE),
        Some(_) if res_out == 0 && scale_out == 0 => Err(ERROR_INVALID_PARAMETER),
        Some(f) => Ok(f.face.units_per_em),
    };
    let units = *r.as_ref().unwrap_or(&0);
    if res_out != 0 {
        ctx.write_u32(res_out, units as u32);
    }
    if scale_out != 0 {
        ctx.write_u32(scale_out, (units as f32 / 64.0).to_bits());
    }
    r.err().unwrap_or(0)
}

/// sceFontGetRenderScaledKerning(fontHandle, preCode, code, FontKerning*):
/// kerning at the bound renderer's scale; characters without a glyph have
/// none.
pub(super) fn cb_get_render_scaled_kerning(ctx: &mut CpuContext) -> HleResult {
    let (h, pre_code, code, out) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx() as u32, ctx.rcx());
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    write_floats(ctx, out, &[0.0; 4]);
    let Some(f) = font(h) else {
        return ERROR_INVALID_FONT_HANDLE;
    };
    if f.renderer == 0 {
        return ERROR_NOT_BOUND_RENDERER;
    }
    if let (Some(left), Some(right)) = (f.face.glyph_index(pre_code), f.face.glyph_index(code)) {
        let (w, _) = f.render_style.pixels();
        let units = f.face.kerning(left, right);
        write_floats(ctx, out, &[units as f32 * w / f.face.units_per_em as f32, 0.0, 0.0, 0.0]);
    }
    0
}

// ---------------------------------------------------------------------------
// Glyph views
// ---------------------------------------------------------------------------

// FontGlyph block layout past the header and FontGlyphMetrics (0x20).
const GLYPH_METRICS_HORIZONTAL: u64 = 0x40;
const GLYPH_METRICS_HORIZONTAL_X: u64 = 0x54;
const GLYPH_METRICS_HORIZONTAL_ADVANCE: u64 = 0x60;
/// FontGlyphOutline { i16 contours, i16 points, u32 flags, points*,
/// tags*, contourEnds* }.
const GLYPH_OUTLINE: u64 = 0x68;
const GLYPH_FORM: u64 = 4;
const GLYPH_METRICS_FORM: u64 = 5;
const GLYPH_FORM_OUTLINE: u8 = 1;

/// The reduced metric views of a generated glyph: horizontal
/// { width, height, bearingX, bearingY, advance }, horizontal X
/// { width, bearingX, advance } and horizontal advance { advance }.
pub(super) fn write_glyph_metric_variants(ctx: &CpuContext, g: u64, m: &GlyphMetrics) {
    write_floats(ctx, g + GLYPH_METRICS_HORIZONTAL, &m[..5]);
    write_floats(ctx, g + GLYPH_METRICS_HORIZONTAL_X, &[m[0], m[2], m[4]]);
    write_floats(ctx, g + GLYPH_METRICS_HORIZONTAL_ADVANCE, &[m[4]]);
}

fn glyph_byte(ctx: &CpuContext, g: u64, offset: u64) -> Result<u8, HleResult> {
    if g == 0 || read_u16(ctx, g) != Some(GLYPH_MAGIC) {
        return Err(ERROR_INVALID_GLYPH);
    }
    Ok(ctx.read_bytes(g + offset, 1).map_or(0, |b| b[0]))
}

/// int sceFontGlyphGetGlyphForm(FontGlyph)
pub(super) fn cb_glyph_get_glyph_form(ctx: &mut CpuContext) -> HleResult {
    glyph_byte(ctx, ctx.rdi(), GLYPH_FORM).map_or_else(|e| e, |b| b as HleResult)
}

/// int sceFontGlyphGetMetricsForm(FontGlyph)
pub(super) fn cb_glyph_get_metrics_form(ctx: &mut CpuContext) -> HleResult {
    glyph_byte(ctx, ctx.rdi(), GLYPH_METRICS_FORM).map_or_else(|e| e, |b| b as HleResult)
}

/// A glyph generated by this library (magic intact).
fn generated(ctx: &CpuContext, g: u64) -> bool {
    g != 0 && read_u16(ctx, g) == Some(GLYPH_MAGIC) && state().glyphs.contains_key(&g)
}

fn refers(ctx: &CpuContext, offset: u64) -> HleResult {
    let g = ctx.rdi();
    if generated(ctx, g) { (g + offset) as HleResult } else { 0 }
}

/// const FontGlyphMetrics* sceFontGlyphRefersMetrics(FontGlyph)
pub(super) fn cb_glyph_refers_metrics(ctx: &mut CpuContext) -> HleResult {
    refers(ctx, GLYPH_METRICS)
}

/// const FontGlyphMetricsHorizontal* sceFontGlyphRefersMetricsHorizontal(FontGlyph)
pub(super) fn cb_glyph_refers_metrics_horizontal(ctx: &mut CpuContext) -> HleResult {
    refers(ctx, GLYPH_METRICS_HORIZONTAL)
}

/// const FontGlyphMetricsHorizontalX* sceFontGlyphRefersMetricsHorizontalX(FontGlyph)
pub(super) fn cb_glyph_refers_metrics_horizontal_x(ctx: &mut CpuContext) -> HleResult {
    refers(ctx, GLYPH_METRICS_HORIZONTAL_X)
}

/// const FontGlyphMetricsHorizontalAdvance* sceFontGlyphRefersMetricsHorizontalAdvance(FontGlyph)
pub(super) fn cb_glyph_refers_metrics_horizontal_advance(ctx: &mut CpuContext) -> HleResult {
    refers(ctx, GLYPH_METRICS_HORIZONTAL_ADVANCE)
}

/// Outline of a glyph in pixels (y up) at the glyph's scale: the font's
/// contours, or the metrics box when the font is closed or the glyph has
/// no contours.
fn glyph_outline(ctx: &CpuContext, g: u64, owner: u64, code: u32) -> Vec<Vec<face::OutlinePoint>> {
    let scale = |o: u64| f32::from_bits(ctx.read_u32(g + o).unwrap_or(0));
    let (sw, sh) = (scale(16), scale(20));
    if let Some(f) = font(owner) {
        if let Some(gid) = f.face.glyph_index(code) {
            let upem = f.face.units_per_em as f32;
            // 26.6 character size, then each coordinate rounded to 1/64 px.
            let (cw, ch) = ((sw * 64.0) as i32 as f32, (sh * 64.0) as i32 as f32);
            let contours: Vec<Vec<face::OutlinePoint>> = f
                .face
                .contours(gid)
                .into_iter()
                .map(|c| c.into_iter().map(|(x, y, on)| ((x * cw / upem).round() / 64.0, (y * ch / upem).round() / 64.0, on)).collect())
                .collect();
            if !contours.is_empty() {
                return contours;
            }
        }
    }
    let m: Vec<f32> = (0..5).map(|i| f32::from_bits(ctx.read_u32(g + GLYPH_METRICS + i * 4).unwrap_or(0))).collect();
    let (left, top) = (m[2], m[3]);
    let (right, bottom) = (left + m[0], top - m[1]);
    vec![vec![(left, top, true), (right, top, true), (right, bottom, true), (left, bottom, true)]]
}

/// FontGlyphOutline* sceFontGlyphRefersOutline(FontGlyph): outline-form
/// glyphs only; points, tags and contour ends live in a library block
/// released with the glyph.
pub(super) fn cb_glyph_refers_outline(ctx: &mut CpuContext) -> HleResult {
    let g = ctx.rdi();
    if !matches!(glyph_byte(ctx, g, GLYPH_FORM), Ok(GLYPH_FORM_OUTLINE)) {
        return 0;
    }
    let Some((owner, code, built)) = state().glyphs.get(&g).map(|gl| (gl.owner, gl.code, gl.outline)) else {
        return 0;
    };
    if built != 0 {
        return (g + GLYPH_OUTLINE) as HleResult;
    }
    let contours = glyph_outline(ctx, g, owner, code);
    let points: Vec<face::OutlinePoint> = contours.iter().flatten().copied().collect();
    let (n, c) = (points.len() as u64, contours.len() as u64);
    // points (8 bytes each), contour ends (u16), tags (u8).
    let ends_at = n * 8;
    let tags_at = ends_at + c * 2;
    let Some(block) = galloc(ctx, tags_at + n) else {
        return 0;
    };
    let mut b = vec![0u8; (tags_at + n) as usize];
    let mut end = 0u16;
    for (i, contour) in contours.iter().enumerate() {
        end += contour.len() as u16;
        b[(ends_at as usize + i * 2)..][..2].copy_from_slice(&(end - 1).to_le_bytes());
    }
    for (i, &(x, y, on)) in points.iter().enumerate() {
        b[i * 8..i * 8 + 4].copy_from_slice(&x.to_le_bytes());
        b[i * 8 + 4..i * 8 + 8].copy_from_slice(&y.to_le_bytes());
        b[tags_at as usize + i] = on as u8;
    }
    ctx.write_bytes(block, &b);
    let flags = read_u16(ctx, g + 2).unwrap_or(0) as u32;
    let mut o = [0u8; 0x20];
    o[0..2].copy_from_slice(&(c as i16).to_le_bytes());
    o[2..4].copy_from_slice(&(n as i16).to_le_bytes());
    o[4..8].copy_from_slice(&flags.to_le_bytes());
    o[8..16].copy_from_slice(&block.to_le_bytes());
    o[16..24].copy_from_slice(&(block + tags_at).to_le_bytes());
    o[24..32].copy_from_slice(&(block + ends_at).to_le_bytes());
    ctx.write_bytes(g + GLYPH_OUTLINE, &o);
    match state().glyphs.get_mut(&g) {
        Some(gl) => gl.outline = block,
        None => {
            gfree(block);
            return 0;
        }
    }
    (g + GLYPH_OUTLINE) as HleResult
}

pub(super) const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("u5fZd3KZcs0", "sceFontCreateRenderer", cb_create_renderer),
    ("BozJej5T6fs", "sceFontGetPixelResolution", cb_get_pixel_resolution),
    ("kihFGYJee7o", "sceFontSetFontsOpenMode", cb_set_fonts_open_mode),
    ("I9R5VC6eZWo", "sceFontClearDeviceCache", cb_clear_device_cache),
    ("UuY-OJF+f0k", "sceFontDettachDeviceCacheBuffer", cb_dettach_device_cache_buffer),
    ("amcmrY62BD4", "sceFontRendererGetOutlineBufferSize", cb_renderer_get_outline_buffer_size),
    ("ai6AfGrBs4o", "sceFontRendererResetOutlineBuffer", cb_renderer_reset_outline_buffer),
    ("ydF+WuH0fAk", "sceFontRendererSetOutlineBufferPolicy", cb_renderer_set_outline_buffer_policy),
    ("RvXyHMUiLhE", "sceFontOpenFontFile", cb_open_font_file),
    ("LzmHDnlcwfQ", "sceFontGetLibrary", cb_get_library),
    ("ZB8xRemRRG8", "sceFontGetFontGlyphsCount", cb_get_font_glyphs_count),
    ("OINC0X9HGBY", "sceFontGetCharGlyphCode", cb_get_char_glyph_code),
    ("tiIlroGki+g", "sceFontGetFontResolution", cb_get_font_resolution),
    ("ryPlnDDI3rU", "sceFontGetRenderScaledKerning", cb_get_render_scaled_kerning),
    ("PXlA0M8ax40", "sceFontGlyphGetGlyphForm", cb_glyph_get_glyph_form),
    ("XUfSWpLhrUw", "sceFontGlyphGetMetricsForm", cb_glyph_get_metrics_form),
    ("ntrc3bEWlvQ", "sceFontGlyphRefersMetrics", cb_glyph_refers_metrics),
    ("9kTbF59TjLs", "sceFontGlyphRefersMetricsHorizontal", cb_glyph_refers_metrics_horizontal),
    ("JCnVgZgcucs", "sceFontGlyphRefersMetricsHorizontalX", cb_glyph_refers_metrics_horizontal_x),
    ("nJavPEdMDvM", "sceFontGlyphRefersMetricsHorizontalAdvance", cb_glyph_refers_metrics_horizontal_advance),
    ("R1T4i+DOhNY", "sceFontGlyphRefersOutline", cb_glyph_refers_outline),
];
