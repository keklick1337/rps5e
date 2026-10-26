// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Library, renderer, style, style-frame and glyph-view exports, with the
//! guest font test sequence ported.

use super::objects::*;
use super::style::*;
use super::*;
use crate::context::CpuRegister::*;

const BASE: u64 = 0x5E37_0000_0000;
const SIZE: u64 = 0x40000;
const STACK: u64 = BASE + 0x200;
const OUT: u64 = BASE + 0x400;
const FONT_DATA: u64 = BASE + 0x4000;

type Cb = fn(&mut CpuContext) -> HleResult;

/// Call `f` with up to seven integer arguments (the seventh on the stack)
/// and the given float arguments.
fn call(ctx: &mut CpuContext, f: Cb, args: &[u64], floats: &[f32]) -> HleResult {
    for (&r, &v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
        ctx.set(r, v);
    }
    if let Some(&v) = args.get(6) {
        ctx.write_u64(STACK + 8, v);
    }
    ctx.set(Rsp, STACK);
    TEST_XMM.with(|x| x.borrow_mut()[..floats.len()].copy_from_slice(floats));
    f(ctx)
}

fn floats(ctx: &CpuContext, addr: u64, n: u64) -> Vec<f32> {
    (0..n).map(|i| ctx.read_f32(addr + i * 4).unwrap()).collect()
}

fn u32s(ctx: &CpuContext, addr: u64, n: u64) -> Vec<u32> {
    (0..n).map(|i| ctx.read_u32(addr + i * 4).unwrap()).collect()
}

/// sceFontMemoryInit with a non-null alloc/free interface.
fn memory(ctx: &mut CpuContext) -> u64 {
    ctx.write_u64(BASE + 0x100, 0x1111);
    ctx.write_u64(BASE + 0x108, 0x2222);
    assert_eq!(call(ctx, cb_memory_init, &[BASE, 0, 0, BASE + 0x100, 0, 0], &[]), 0);
    BASE
}

fn create_library(ctx: &mut CpuContext, mem: u64) -> u64 {
    let driver = call(ctx, cb_select_library_ft, &[0], &[]) as u64;
    assert_eq!(call(ctx, cb_create_library, &[mem, driver, OUT], &[]), 0);
    ctx.read_u64(OUT).unwrap()
}

fn create_renderer(ctx: &mut CpuContext, mem: u64) -> u64 {
    let selection = call(ctx, cb_select_renderer_ft, &[0], &[]) as u64;
    ctx.write_u64(OUT, 0x77);
    assert_eq!(call(ctx, cb_create_renderer, &[mem, 0, OUT], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(ctx.read_u64(OUT), Some(0));
    assert_eq!(call(ctx, cb_create_renderer, &[mem, selection, OUT], &[]), 0);
    ctx.read_u64(OUT).unwrap()
}

fn open_memory(ctx: &mut CpuContext, lib: u64, data: &[u8]) -> Result<u64, HleResult> {
    ctx.write_bytes(FONT_DATA, data);
    match call(ctx, cb_open_font_memory, &[lib, FONT_DATA, data.len() as u64, 0, OUT], &[]) {
        0 => Ok(ctx.read_u64(OUT).unwrap()),
        e => Err(e),
    }
}

fn words(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|&v| (v as u16).to_be_bytes()).collect()
}

/// A TrueType font of `glyph_count` empty glyphs (1000 units per em) plus
/// the given tables.
fn build_font(glyph_count: usize, mut tables: BTreeMap<&'static str, Vec<u8>>) -> Vec<u8> {
    tables.insert("glyf", words(&[0, 0]));
    tables.insert(
        "head",
        words(&[1, 0, 1, 0, 0, 0, 0x5F0F, 0x3CF5, 0, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 2, 0, 0]),
    );
    tables.insert("hhea", words(&[1, 0, 800, -200, 0, 500, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]));
    let mut hmtx = words(&[500]);
    hmtx.resize(2 + 2 * glyph_count, 0);
    tables.insert("hmtx", hmtx);
    tables.insert("loca", vec![0; 2 * (glyph_count + 1)]);
    tables.insert("maxp", words(&[1, 0, glyph_count as i32, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]));
    let count = tables.len() as i32;
    let mut selector = 0;
    while (2 << selector) <= count {
        selector += 1;
    }
    let mut font = words(&[1, 0, count, 16 << selector, selector, 16 * count - (16 << selector)]);
    let mut offset = 12 + 16 * count;
    for (tag, data) in &tables {
        font.extend_from_slice(tag.as_bytes());
        font.extend(words(&[0, 0, offset >> 16, offset & 0xFFFF, 0, data.len() as i32]));
        offset += ((data.len() + 3) & !3) as i32;
    }
    for data in tables.values() {
        font.extend_from_slice(data);
        font.resize((font.len() + 3) & !3, 0);
    }
    font
}

fn empty_glyph_font() -> Vec<u8> {
    build_font(1, BTreeMap::new())
}

/// 'A' -> glyph 1, 'V' -> glyph 2, kerning A,V = -200 units.
fn kerning_font() -> Vec<u8> {
    let (a, v) = (b'A' as i32, b'V' as i32);
    let mut tables = BTreeMap::new();
    tables.insert("cmap", words(&[0, 1, 3, 1, 0, 12, 4, 40, 0, 6, 4, 1, 2, a, v, 0xFFFF, 0, a, v, 0xFFFF, 1 - a, 2 - v, 1, 0, 0, 0]));
    tables.insert("kern", words(&[0, 1, 0, 20, 1, 1, 6, 0, 0, 1, 2, -200]));
    build_font(3, tables)
}

fn kerning(ctx: &mut CpuContext, f: Cb, font: u64, pre: u8, code: u8) -> (HleResult, Vec<f32>) {
    ctx.write_bytes(OUT, &[0x3F; 16]);
    let rc = call(ctx, f, &[font, pre as u64, code as u64, OUT], &[]);
    (rc, floats(ctx, OUT, 4))
}

/// The guest font test: pixel resolution, device cache, font open errors,
/// resolution, glyph queries and render-scaled kerning.
fn guest_sequence(ctx: &mut CpuContext) {
    let mem = memory(ctx);
    let lib = create_library(ctx, mem);

    assert_eq!(call(ctx, cb_get_pixel_resolution, &[lib, OUT], &[]), 0);
    assert_eq!(ctx.read_u32(OUT), Some(64));
    assert_eq!(call(ctx, cb_get_pixel_resolution, &[lib, 0], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_get_pixel_resolution, &[0, 0], &[]), ERROR_INVALID_PARAMETER);
    ctx.write_u32(OUT, 1);
    assert_eq!(call(ctx, cb_get_pixel_resolution, &[0, OUT], &[]), ERROR_INVALID_LIBRARY);
    assert_eq!(ctx.read_u32(OUT), Some(0));
    let not_a_library = BASE + 0x1000;
    ctx.write_u32(OUT, 1);
    assert_eq!(call(ctx, cb_get_pixel_resolution, &[not_a_library, OUT], &[]), ERROR_INVALID_LIBRARY);
    assert_eq!(ctx.read_u32(OUT), Some(0));

    // Device cache.
    let (buf_out, size_out) = (OUT, OUT + 8);
    assert_eq!(call(ctx, cb_dettach_device_cache_buffer, &[0, buf_out, size_out], &[]), ERROR_INVALID_LIBRARY);
    assert_eq!(call(ctx, cb_dettach_device_cache_buffer, &[not_a_library, buf_out, size_out], &[]), ERROR_INVALID_LIBRARY);
    assert_eq!(call(ctx, cb_clear_device_cache, &[lib], &[]), ERROR_NOT_ATTACHED_CACHE_BUFFER);
    assert_eq!(call(ctx, cb_dettach_device_cache_buffer, &[lib, buf_out, size_out], &[]), ERROR_NOT_ATTACHED_CACHE_BUFFER);
    let cache = BASE + 0x10000;
    assert_eq!(call(ctx, cb_attach_device_cache_buffer, &[lib, cache, 0x1000], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_attach_device_cache_buffer, &[lib, cache, 0x2000], &[]), 0);
    assert_eq!(call(ctx, cb_attach_device_cache_buffer, &[lib, cache, 0x2000], &[]), ERROR_ALREADY_ATTACHED);
    // Using pages, then clearing: every page free again.
    ctx.write_u32(cache + 8, 1);
    ctx.write_u32(cache + 12, 0);
    assert_eq!(call(ctx, cb_clear_device_cache, &[lib], &[]), 0);
    assert_eq!(u32s(ctx, cache, 4), [0x2000, 1, 0, 1]);
    assert_eq!(call(ctx, cb_dettach_device_cache_buffer, &[lib, buf_out, size_out], &[]), 0);
    assert_eq!((ctx.read_u64(buf_out), ctx.read_u32(size_out)), (Some(cache), Some(0x2000)));
    assert_eq!(call(ctx, cb_dettach_device_cache_buffer, &[lib, buf_out, size_out], &[]), ERROR_NOT_ATTACHED_CACHE_BUFFER);
    assert_eq!(call(ctx, cb_attach_device_cache_buffer, &[lib, 0, 0x2000], &[]), 0);
    ctx.write_u64(buf_out, 0);
    assert_eq!(call(ctx, cb_dettach_device_cache_buffer, &[lib, buf_out, size_out], &[]), 0);
    assert!(ctx.read_u64(buf_out).unwrap() != 0 && ctx.read_u32(size_out) == Some(0x2000));
    assert_eq!(call(ctx, cb_dettach_device_cache_buffer, &[lib, buf_out, size_out], &[]), ERROR_NOT_ATTACHED_CACHE_BUFFER);
    assert_eq!(call(ctx, cb_attach_device_cache_buffer, &[lib, 0, 0x2000], &[]), 0);
    assert_eq!(call(ctx, cb_dettach_device_cache_buffer, &[lib, 0, 0], &[]), 0);
    assert_eq!(call(ctx, cb_dettach_device_cache_buffer, &[lib, 0, 0], &[]), ERROR_NOT_ATTACHED_CACHE_BUFFER);

    // Opening needs external font support.
    assert_eq!(open_memory(ctx, lib, &[1, 2, 3, 4]), Err(ERROR_NO_SUPPORT_FUNCTION));
    assert_eq!(call(ctx, cb_support_external_fonts, &[lib, 4, 0x52], &[]), 0);
    let missing = BASE + 0x1100;
    ctx.write_bytes(missing, b"rps5e-missing-font-parent/../font.ttf\0");
    ctx.write_u64(OUT, 0x55);
    assert_eq!(call(ctx, cb_open_font_file, &[lib, missing, 1, 0, OUT], &[]), ERROR_FS_OPEN_FAILED);
    assert_eq!(ctx.read_u64(OUT), Some(0));
    assert_eq!(open_memory(ctx, lib, &[1, 2, 3, 4]), Err(ERROR_NO_SUPPORT_FORMAT));

    let font = open_memory(ctx, lib, &empty_glyph_font()).unwrap();
    let (hd, vd) = (OUT, OUT + 4);
    assert_eq!(call(ctx, cb_get_resolution_dpi, &[font, hd, vd], &[]), 0);
    assert_eq!(u32s(ctx, OUT, 2), [72, 72]);
    assert_eq!(call(ctx, cb_set_resolution_dpi, &[font, 96, 144], &[]), 0);
    assert_eq!(call(ctx, cb_get_resolution_dpi, &[font, hd, vd], &[]), 0);
    assert_eq!(u32s(ctx, OUT, 2), [96, 144]);
    ctx.write_bytes(OUT, &[1; 8]);
    assert_eq!(call(ctx, cb_get_resolution_dpi, &[font, hd, 0], &[]), 0);
    assert_eq!(ctx.read_u32(hd), Some(96));
    assert_eq!(call(ctx, cb_get_resolution_dpi, &[font, 0, vd], &[]), 0);
    assert_eq!(ctx.read_u32(vd), Some(144));
    assert_eq!(call(ctx, cb_get_resolution_dpi, &[font, 0, 0], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_set_resolution_dpi, &[font, 0, 300], &[]), 0);
    assert_eq!(call(ctx, cb_get_resolution_dpi, &[font, hd, vd], &[]), 0);
    assert_eq!(u32s(ctx, OUT, 2), [72, 300]);
    assert_eq!(call(ctx, cb_get_resolution_dpi, &[0, hd, vd], &[]), ERROR_INVALID_FONT_HANDLE);
    assert_eq!(u32s(ctx, OUT, 2), [0, 0]);
    let unopened = BASE + 0x1200;
    ctx.write_bytes(OUT, &[1; 8]);
    assert_eq!(call(ctx, cb_get_resolution_dpi, &[unopened, hd, vd], &[]), ERROR_INVALID_FONT_HANDLE);
    assert_eq!(u32s(ctx, OUT, 2), [0, 0]);

    assert_eq!(call(ctx, cb_get_font_glyphs_count, &[font, OUT], &[]), 0);
    assert_eq!(ctx.read_u32(OUT), Some(1));
    assert_eq!(call(ctx, cb_get_font_glyphs_count, &[font, 0], &[]), ERROR_INVALID_PARAMETER);
    for handle in [0, unopened] {
        ctx.write_u32(OUT, 1);
        assert_eq!(call(ctx, cb_get_font_glyphs_count, &[handle, OUT], &[]), ERROR_INVALID_FONT_HANDLE);
        assert_eq!(ctx.read_u32(OUT), Some(0));
    }
    ctx.write_u32(OUT, 1);
    assert_eq!(call(ctx, cb_get_char_glyph_code, &[font, b'A' as u64, OUT], &[]), ERROR_NO_SUPPORT_GLYPH);
    assert_eq!(ctx.read_u32(OUT), Some(0));
    assert_eq!(call(ctx, cb_get_char_glyph_code, &[font, b'A' as u64, 0], &[]), ERROR_INVALID_PARAMETER);
    for handle in [0, unopened] {
        ctx.write_u32(OUT, 1);
        assert_eq!(call(ctx, cb_get_char_glyph_code, &[handle, b'A' as u64, OUT], &[]), ERROR_INVALID_FONT_HANDLE);
        assert_eq!(ctx.read_u32(OUT), Some(0));
    }
    let (res, scale) = (OUT, OUT + 4);
    assert_eq!(call(ctx, cb_get_font_resolution, &[font, res, scale], &[]), 0);
    assert_eq!((ctx.read_u32(res), ctx.read_f32(scale)), (Some(1000), Some(15.625)));
    ctx.write_bytes(OUT, &[0; 8]);
    assert_eq!(call(ctx, cb_get_font_resolution, &[font, res, 0], &[]), 0);
    assert_eq!(ctx.read_u32(res), Some(1000));
    assert_eq!(call(ctx, cb_get_font_resolution, &[font, 0, scale], &[]), 0);
    assert_eq!(ctx.read_f32(scale), Some(15.625));
    assert_eq!(call(ctx, cb_get_font_resolution, &[font, 0, 0], &[]), ERROR_INVALID_PARAMETER);
    for handle in [0, unopened] {
        ctx.write_bytes(OUT, &[1; 8]);
        assert_eq!(call(ctx, cb_get_font_resolution, &[handle, res, scale], &[]), ERROR_INVALID_FONT_HANDLE);
        assert_eq!((ctx.read_u32(res), ctx.read_f32(scale)), (Some(0), Some(0.0)));
    }
    assert_eq!(call(ctx, cb_close_font, &[font], &[]), 0);

    let renderer = create_renderer(ctx, mem);
    let font = open_memory(ctx, lib, &kerning_font()).unwrap();
    assert_eq!(call(ctx, cb_get_font_glyphs_count, &[font, OUT], &[]), 0);
    assert_eq!(ctx.read_u32(OUT), Some(3));
    assert_eq!(call(ctx, cb_get_char_glyph_code, &[font, b'A' as u64, OUT], &[]), 0);
    assert_eq!(ctx.read_u32(OUT), Some(1));
    assert_eq!(call(ctx, cb_get_char_glyph_code, &[font, b'V' as u64, OUT], &[]), 0);
    assert_eq!(ctx.read_u32(OUT), Some(2));
    ctx.write_u32(OUT, 1);
    assert_eq!(call(ctx, cb_get_char_glyph_code, &[font, b'B' as u64, OUT], &[]), ERROR_NO_SUPPORT_GLYPH);
    assert_eq!(ctx.read_u32(OUT), Some(0));
    ctx.write_u32(OUT, 1);
    assert_eq!(call(ctx, cb_get_char_glyph_code, &[font, 0, OUT], &[]), ERROR_NO_SUPPORT_CODE);
    assert_eq!(ctx.read_u32(OUT), Some(0));
    assert_eq!(call(ctx, cb_set_resolution_dpi, &[font, 144, 144], &[]), 0);
    assert_eq!(call(ctx, cb_set_scale_pixel, &[font], &[100.0, 100.0]), 0);

    let k = |v: f32| vec![v, 0.0, 0.0, 0.0];
    assert_eq!(kerning(ctx, cb_get_kerning, font, b'A', b'V'), (0, k(-20.0)));
    assert_eq!(kerning(ctx, cb_get_render_scaled_kerning, font, b'A', b'V'), (ERROR_NOT_BOUND_RENDERER, k(0.0)));
    assert_eq!(call(ctx, cb_bind_renderer, &[font, renderer], &[]), 0);
    assert_eq!(kerning(ctx, cb_get_render_scaled_kerning, font, b'A', b'V'), (0, k(-20.0)));
    assert_eq!(call(ctx, cb_setup_render_scale_pixel, &[font], &[50.0, 50.0]), 0);
    assert_eq!(kerning(ctx, cb_get_render_scaled_kerning, font, b'A', b'V'), (0, k(-10.0)));
    assert_eq!(kerning(ctx, cb_get_kerning, font, b'A', b'V'), (0, k(-20.0)));
    // 20 pt at 144 dpi = 40 px.
    assert_eq!(call(ctx, cb_setup_render_scale_point, &[font], &[20.0, 20.0]), 0);
    assert_eq!(kerning(ctx, cb_get_render_scaled_kerning, font, b'A', b'V'), (0, k(-8.0)));
    assert_eq!(kerning(ctx, cb_get_render_scaled_kerning, font, b'V', b'A'), (0, k(0.0)));
    assert_eq!(kerning(ctx, cb_get_render_scaled_kerning, font, b'A', b'B'), (0, k(0.0)));
    assert_eq!(call(ctx, cb_get_render_scaled_kerning, &[font, b'A' as u64, b'V' as u64, 0], &[]), ERROR_INVALID_PARAMETER);
    for handle in [0, unopened] {
        assert_eq!(kerning(ctx, cb_get_render_scaled_kerning, handle, b'A', b'V'), (ERROR_INVALID_FONT_HANDLE, k(0.0)));
    }
    assert_eq!(call(ctx, cb_unbind_renderer, &[font], &[]), 0);
    assert_eq!(kerning(ctx, cb_get_render_scaled_kerning, font, b'A', b'V').0, ERROR_NOT_BOUND_RENDERER);
    assert_eq!(call(ctx, cb_close_font, &[font], &[]), 0);

    ctx.write_u64(OUT, renderer);
    assert_eq!(call(ctx, cb_destroy_renderer, &[OUT], &[]), 0);
    assert_eq!(ctx.read_u64(OUT), Some(0));
    assert_eq!(call(ctx, cb_destroy_renderer, &[OUT], &[]), ERROR_INVALID_RENDERER);
    ctx.write_u64(OUT, lib);
    assert_eq!(call(ctx, cb_destroy_library, &[OUT], &[]), 0);
    assert_eq!(call(ctx, cb_destroy_library, &[OUT], &[]), ERROR_INVALID_LIBRARY);
}

/// Font and bound-renderer style queries.
fn font_styles(ctx: &mut CpuContext, lib: u64, renderer: u64) {
    let font = open_memory(ctx, lib, &tests::test_font()).unwrap();
    let (w, h) = (OUT, OUT + 4);
    let scale = |ctx: &mut CpuContext, f: Cb, font: u64| -> (HleResult, Vec<f32>) {
        ctx.write_bytes(OUT, &[0x3F; 8]);
        let rc = call(ctx, f, &[font, w, h], &[]);
        (rc, floats(ctx, OUT, 2))
    };
    // Opened at units per em / 64 pixels, 72 dpi.
    assert_eq!(scale(ctx, cb_get_scale_pixel, font), (0, vec![15.625, 15.625]));
    assert_eq!(call(ctx, cb_set_scale_pixel, &[font], &[20.0, 20.0]), 0);
    assert_eq!(scale(ctx, cb_get_scale_point, font), (0, vec![20.0, 20.0]));
    assert_eq!(call(ctx, cb_set_resolution_dpi, &[font, 144, 0], &[]), 0);
    assert_eq!(scale(ctx, cb_get_scale_point, font), (0, vec![10.0, 20.0]));
    assert_eq!(scale(ctx, cb_get_scale_pixel, font), (0, vec![20.0, 20.0]));
    assert_eq!(call(ctx, cb_set_scale_point, &[font], &[12.0, 12.0]), 0);
    assert_eq!(scale(ctx, cb_get_scale_pixel, font), (0, vec![24.0, 12.0]));
    assert_eq!(scale(ctx, cb_get_scale_point, font), (0, vec![12.0, 12.0]));
    assert_eq!(call(ctx, cb_get_scale_pixel, &[font, 0, 0], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(scale(ctx, cb_get_scale_pixel, 0), (ERROR_INVALID_FONT_HANDLE, vec![0.0, 0.0]));
    assert_eq!(scale(ctx, cb_get_scale_point, BASE + 0x1200), (ERROR_INVALID_FONT_HANDLE, vec![0.0, 0.0]));

    let slant = |ctx: &mut CpuContext, f: Cb, font: u64| -> (HleResult, f32) {
        ctx.write_u32(OUT, 0x3F00_0000);
        let rc = call(ctx, f, &[font, OUT], &[]);
        (rc, ctx.read_f32(OUT).unwrap())
    };
    let weight = |ctx: &mut CpuContext, f: Cb, font: u64| -> (HleResult, f32, f32, u32) {
        ctx.write_bytes(OUT, &[0x3F; 12]);
        let rc = call(ctx, f, &[font, OUT, OUT + 4, OUT + 8], &[]);
        (rc, ctx.read_f32(OUT).unwrap(), ctx.read_f32(OUT + 4).unwrap(), ctx.read_u32(OUT + 8).unwrap())
    };
    assert_eq!(slant(ctx, cb_get_effect_slant, font), (0, 0.0));
    assert_eq!(call(ctx, cb_set_effect_slant, &[font], &[0.25]), 0);
    assert_eq!(slant(ctx, cb_get_effect_slant, font), (0, 0.25));
    assert_eq!(call(ctx, cb_set_effect_slant, &[font], &[-3.0]), 0);
    assert_eq!(slant(ctx, cb_get_effect_slant, font), (0, -1.0));
    assert_eq!(call(ctx, cb_get_effect_slant, &[font, 0], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(slant(ctx, cb_get_effect_slant, 0), (ERROR_INVALID_FONT_HANDLE, 0.0));
    assert_eq!(weight(ctx, cb_get_effect_weight, font), (0, 1.0, 1.0, 0));
    assert_eq!(call(ctx, cb_set_effect_weight, &[font, 0], &[1.02, 0.5]), 0);
    let (rc, wx, wy, mode) = weight(ctx, cb_get_effect_weight, font);
    assert!(rc == 0 && (wx - 1.02).abs() < 1e-6 && (wy - 0.96).abs() < 1e-6 && mode == 0);
    assert_eq!(call(ctx, cb_get_effect_weight, &[font, 0, 0, 0], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(weight(ctx, cb_get_effect_weight, 0), (ERROR_INVALID_FONT_HANDLE, 1.0, 1.0, 0));

    // Render style: copied from the font style when the renderer binds.
    assert_eq!(scale(ctx, cb_get_render_scale_pixel, font), (ERROR_NOT_BOUND_RENDERER, vec![0.0, 0.0]));
    assert_eq!(slant(ctx, cb_get_render_effect_slant, font), (ERROR_NOT_BOUND_RENDERER, 0.0));
    assert_eq!(weight(ctx, cb_get_render_effect_weight, font), (ERROR_NOT_BOUND_RENDERER, 1.0, 1.0, 0));
    assert_eq!(call(ctx, cb_setup_render_scale_point, &[font], &[1.0, 1.0]), ERROR_NOT_BOUND_RENDERER);
    assert_eq!(call(ctx, cb_bind_renderer, &[font, renderer], &[]), 0);
    assert_eq!(scale(ctx, cb_get_render_scale_pixel, font), (0, vec![24.0, 12.0]));
    assert_eq!(slant(ctx, cb_get_render_effect_slant, font), (0, -1.0));
    assert_eq!(call(ctx, cb_setup_render_scale_pixel, &[font], &[36.0, 18.0]), 0);
    assert_eq!(scale(ctx, cb_get_render_scale_point, font), (0, vec![18.0, 18.0]));
    assert_eq!(call(ctx, cb_setup_render_scale_point, &[font], &[6.0, 6.0]), 0);
    assert_eq!(scale(ctx, cb_get_render_scale_pixel, font), (0, vec![12.0, 6.0]));
    assert_eq!(scale(ctx, cb_get_scale_pixel, font), (0, vec![24.0, 12.0]));
    assert_eq!(call(ctx, cb_setup_render_effect_slant, &[font], &[0.5]), 0);
    assert_eq!(slant(ctx, cb_get_render_effect_slant, font), (0, 0.5));
    assert_eq!(slant(ctx, cb_get_effect_slant, font), (0, -1.0));
    let (rc, wx, ..) = weight(ctx, cb_get_render_effect_weight, font);
    assert!(rc == 0 && (wx - 1.02).abs() < 1e-6);
    assert_eq!(call(ctx, cb_get_render_scale_point, &[font, 0, 0], &[]), ERROR_INVALID_PARAMETER);

    ctx.write_u64(OUT + 0x40, 0);
    assert_eq!(call(ctx, cb_get_library, &[font, OUT + 0x40], &[]), 0);
    assert_eq!(ctx.read_u64(OUT + 0x40), Some(lib));
    assert_eq!(call(ctx, cb_get_library, &[font, 0], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_get_library, &[0, OUT + 0x40], &[]), ERROR_INVALID_FONT_HANDLE);
    assert_eq!(ctx.read_u64(OUT + 0x40), Some(0));
    assert_eq!(call(ctx, cb_close_font, &[font], &[]), 0);
}

/// Style frames, alone and attached to a render surface.
fn style_frames(ctx: &mut CpuContext, lib: u64, renderer: u64) {
    let frame = BASE + 0x2000;
    let (w, h) = (OUT, OUT + 4);
    ctx.write_bytes(frame, &[0xAA; 0x60]);
    assert_eq!(call(ctx, cb_style_frame_init, &[0], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_style_frame_init, &[frame], &[]), 0);
    let raw = ctx.read_bytes(frame, 0x60).unwrap();
    assert_eq!(&raw[..12], &[0x09, 0x0F, 0, 0, 72, 0, 0, 0, 72, 0, 0, 0]);
    assert!(raw[12..].iter().all(|&b| b == 0));

    assert_eq!(call(ctx, cb_style_frame_get_scale_pixel, &[frame, w, h], &[]), ERROR_UNSET_PARAMETER);
    assert_eq!(call(ctx, cb_style_frame_get_effect_slant, &[frame, w], &[]), ERROR_UNSET_PARAMETER);
    assert_eq!(call(ctx, cb_style_frame_get_effect_weight, &[frame, w, h, 0], &[]), ERROR_UNSET_PARAMETER);
    assert_eq!(call(ctx, cb_style_frame_set_scale_pixel, &[frame], &[30.0, 40.0]), 0);
    assert_eq!(call(ctx, cb_style_frame_get_scale_pixel, &[frame, w, h], &[]), 0);
    assert_eq!(floats(ctx, OUT, 2), [30.0, 40.0]);
    assert_eq!(call(ctx, cb_style_frame_get_scale_pixel, &[frame, 0, 0], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_style_frame_set_resolution_dpi, &[frame, 144, 0], &[]), 0);
    assert_eq!(call(ctx, cb_style_frame_get_resolution_dpi, &[frame, w, h], &[]), 0);
    assert_eq!(u32s(ctx, OUT, 2), [144, 72]);
    assert_eq!(call(ctx, cb_style_frame_get_resolution_dpi, &[frame, 0, 0], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_style_frame_get_scale_point, &[frame, w, h], &[]), 0);
    assert_eq!(floats(ctx, OUT, 2), [15.0, 40.0]);
    assert_eq!(call(ctx, cb_style_frame_set_scale_point, &[frame], &[10.0, 10.0]), 0);
    assert_eq!(call(ctx, cb_style_frame_get_scale_pixel, &[frame, w, h], &[]), 0);
    assert_eq!(floats(ctx, OUT, 2), [20.0, 10.0]);
    assert_eq!(call(ctx, cb_style_frame_unset_scale, &[frame], &[]), 0);
    assert_eq!(call(ctx, cb_style_frame_get_scale_point, &[frame, w, h], &[]), ERROR_UNSET_PARAMETER);

    assert_eq!(call(ctx, cb_style_frame_set_effect_slant, &[frame], &[2.0]), 0);
    assert_eq!(call(ctx, cb_style_frame_get_effect_slant, &[frame, w], &[]), 0);
    assert_eq!(ctx.read_f32(w), Some(1.0));
    assert_eq!(call(ctx, cb_style_frame_get_effect_slant, &[frame, 0], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_style_frame_unset_effect_slant, &[frame], &[]), 0);
    assert_eq!(call(ctx, cb_style_frame_get_effect_slant, &[frame, w], &[]), ERROR_UNSET_PARAMETER);
    assert_eq!(call(ctx, cb_style_frame_set_effect_weight, &[frame, 1], &[1.5, 1.0]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_style_frame_set_effect_weight, &[frame, 0], &[1.5, 0.99]), 0);
    assert_eq!(call(ctx, cb_style_frame_get_effect_weight, &[frame, w, h, OUT + 8], &[]), 0);
    let wv = floats(ctx, OUT, 2);
    assert!((wv[0] - 1.04).abs() < 1e-6 && (wv[1] - 0.99).abs() < 1e-6 && ctx.read_u32(OUT + 8) == Some(0));
    assert_eq!(call(ctx, cb_style_frame_get_effect_weight, &[frame, 0, 0, 0], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_style_frame_unset_effect_weight, &[frame], &[]), 0);
    assert_eq!(call(ctx, cb_style_frame_get_effect_weight, &[frame, w, h, 0], &[]), ERROR_UNSET_PARAMETER);

    // Not a style frame.
    let bogus = BASE + 0x2100;
    for f in [cb_style_frame_unset_scale as Cb, cb_style_frame_unset_effect_slant, cb_style_frame_unset_effect_weight] {
        assert_eq!(call(ctx, f, &[bogus], &[]), ERROR_INVALID_PARAMETER);
        assert_eq!(call(ctx, f, &[0], &[]), ERROR_INVALID_PARAMETER);
    }
    assert_eq!(call(ctx, cb_style_frame_set_scale_pixel, &[bogus], &[1.0, 1.0]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_style_frame_get_scale_pixel, &[bogus, w, h], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_style_frame_set_resolution_dpi, &[bogus, 1, 1], &[]), ERROR_INVALID_PARAMETER);

    // A frame attached to a surface overrides the render scale.
    let font = open_memory(ctx, lib, &tests::test_font()).unwrap();
    assert_eq!(call(ctx, cb_set_scale_pixel, &[font], &[20.0, 20.0]), 0);
    assert_eq!(call(ctx, cb_bind_renderer, &[font, renderer], &[]), 0);
    let (surf, buf) = (BASE + 0x2200, BASE + 0x8000);
    call(ctx, cb_render_surface_init, &[surf, buf, 64, 1, 64, 64], &[]);
    let rendered_width = |ctx: &mut CpuContext| {
        ctx.write_bytes(buf, &[0; 64 * 64]);
        assert_eq!(call(ctx, cb_render_char_glyph_image_horizontal, &[font, b'A' as u64, surf, OUT, OUT + 0x40], &[2.0, 50.0]), 0);
        ctx.read_u32(OUT + 0x40 + 0x38).unwrap()
    };
    assert_eq!(rendered_width(ctx), 10);
    assert_eq!(call(ctx, cb_style_frame_set_scale_pixel, &[frame], &[40.0, 40.0]), 0);
    assert_eq!(call(ctx, cb_render_surface_set_style_frame, &[0, frame], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_render_surface_set_style_frame, &[surf, bogus], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_render_surface_set_style_frame, &[surf, frame], &[]), 0);
    assert_eq!(ctx.read_bytes(surf + 0xE, 1).unwrap()[0] & 1, 1);
    assert_eq!(ctx.read_u64(surf + 0x28), Some(frame));
    assert_eq!(rendered_width(ctx), 20);
    assert_eq!(call(ctx, cb_render_surface_set_style_frame, &[surf, 0], &[]), 0);
    assert_eq!(ctx.read_bytes(surf + 0xE, 1).unwrap()[0] & 1, 0);
    assert_eq!(ctx.read_u64(surf + 0x28), Some(0));
    assert_eq!(rendered_width(ctx), 10);
    assert_eq!(call(ctx, cb_close_font, &[font], &[]), 0);
}

/// Renderer outline workspace size, policy and reset.
fn renderer_outline_buffer(ctx: &mut CpuContext, renderer: u64) {
    let size = |ctx: &mut CpuContext| {
        assert_eq!(call(ctx, cb_renderer_get_outline_buffer_size, &[renderer, OUT], &[]), 0);
        ctx.read_u32(OUT).unwrap()
    };
    assert_eq!(size(ctx), 0x4000);
    assert_eq!(call(ctx, cb_renderer_get_outline_buffer_size, &[renderer, 0], &[]), ERROR_INVALID_PARAMETER);
    ctx.write_u32(OUT, 1);
    assert_eq!(call(ctx, cb_renderer_get_outline_buffer_size, &[BASE + 0x1000, OUT], &[]), ERROR_INVALID_RENDERER);
    assert_eq!(ctx.read_u32(OUT), Some(0));
    assert_eq!(call(ctx, cb_renderer_set_outline_buffer_policy, &[renderer, 0, 0x8000, 0], &[]), 0);
    assert_eq!(size(ctx), 0x8000);
    assert_eq!(call(ctx, cb_renderer_set_outline_buffer_policy, &[renderer, 0, 0x1000, 0], &[]), 0);
    assert_eq!(size(ctx), 0x8000);
    assert_eq!(call(ctx, cb_renderer_set_outline_buffer_policy, &[renderer, 0, 0x10000, 0x9000], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_renderer_set_outline_buffer_policy, &[renderer, 0, 0x2000, 0x6000], &[]), 0);
    assert_eq!(size(ctx), 0x6000);
    assert_eq!(call(ctx, cb_renderer_set_outline_buffer_policy, &[0, 0, 0, 0], &[]), ERROR_INVALID_RENDERER);
    let workspace = state().renderers[&renderer].workspace;
    ctx.write_bytes(workspace, &[0x5A; 0x6000]);
    assert_eq!(call(ctx, cb_renderer_reset_outline_buffer, &[renderer], &[]), 0);
    assert!(ctx.read_bytes(workspace, 0x6000).unwrap().iter().all(|&b| b == 0));
    assert_eq!(call(ctx, cb_renderer_reset_outline_buffer, &[0], &[]), ERROR_INVALID_RENDERER);
}

/// Metrics views and outlines of generated glyphs.
fn glyph_views(ctx: &mut CpuContext, lib: u64) {
    let font = open_memory(ctx, lib, &tests::test_font()).unwrap();
    assert_eq!(call(ctx, cb_set_scale_pixel, &[font], &[20.0, 20.0]), 0);
    // FontGenerateGlyphDetail: id, options, glyph form 1 (outline), metrics form 1.
    let detail = BASE + 0x2400;
    let mut d = [0u8; 0x10];
    d[0..2].copy_from_slice(&GENERATE_GLYPH_DETAIL_ID.to_le_bytes());
    d[6] = 1;
    d[7] = 1;
    ctx.write_bytes(detail, &d);
    let generate = |ctx: &mut CpuContext, code: u8, detail: u64| {
        assert_eq!(call(ctx, cb_generate_char_glyph, &[font, code as u64, detail, OUT + 0x80], &[]), 0);
        ctx.read_u64(OUT + 0x80).unwrap()
    };
    let g = generate(ctx, b'A', detail);
    assert_eq!(call(ctx, cb_glyph_get_glyph_form, &[g], &[]), 1);
    assert_eq!(call(ctx, cb_glyph_get_metrics_form, &[g], &[]), 1);
    assert_eq!(call(ctx, cb_glyph_get_glyph_form, &[0], &[]), ERROR_INVALID_GLYPH);
    assert_eq!(call(ctx, cb_glyph_get_metrics_form, &[BASE + 0x1000], &[]), ERROR_INVALID_GLYPH);

    let m = call(ctx, cb_glyph_refers_metrics, &[g], &[]) as u64;
    assert_eq!(floats(ctx, m, 5), [10.0, 14.0, 0.0, 14.0, 12.0]);
    let mh = call(ctx, cb_glyph_refers_metrics_horizontal, &[g], &[]) as u64;
    assert_eq!(floats(ctx, mh, 5), [10.0, 14.0, 0.0, 14.0, 12.0]);
    let mx = call(ctx, cb_glyph_refers_metrics_horizontal_x, &[g], &[]) as u64;
    assert_eq!(floats(ctx, mx, 3), [10.0, 0.0, 12.0]);
    let ma = call(ctx, cb_glyph_refers_metrics_horizontal_advance, &[g], &[]) as u64;
    assert_eq!(floats(ctx, ma, 1), [12.0]);
    for f in [cb_glyph_refers_metrics as Cb, cb_glyph_refers_metrics_horizontal, cb_glyph_refers_metrics_horizontal_x] {
        assert_eq!(call(ctx, f, &[0], &[]), 0);
        assert_eq!(call(ctx, f, &[BASE + 0x1000], &[]), 0);
    }

    // The 500 x 700 unit square at 20 px/em.
    let o = call(ctx, cb_glyph_refers_outline, &[g], &[]) as u64;
    assert_ne!(o, 0);
    assert_eq!(call(ctx, cb_glyph_refers_outline, &[g], &[]) as u64, o);
    let (contours, points) = (read_u16(ctx, o).unwrap(), read_u16(ctx, o + 2).unwrap());
    assert_eq!((contours, points), (1, 4));
    let (pts, tags, ends) = (ctx.read_u64(o + 8).unwrap(), ctx.read_u64(o + 16).unwrap(), ctx.read_u64(o + 24).unwrap());
    assert_eq!(floats(ctx, pts, 8), [0.0, 0.0, 10.0, 0.0, 10.0, 14.0, 0.0, 14.0]);
    assert_eq!(ctx.read_bytes(tags, 4).unwrap(), [1, 1, 1, 1]);
    assert_eq!(read_u16(ctx, ends), Some(3));

    // An empty glyph outlines its metrics box.
    let space = generate(ctx, b' ', detail);
    let o = call(ctx, cb_glyph_refers_outline, &[space], &[]) as u64;
    assert_eq!((read_u16(ctx, o), read_u16(ctx, o + 2)), (Some(1), Some(4)));
    assert_eq!(floats(ctx, ctx.read_u64(o + 8).unwrap(), 8), [0.0; 8]);

    // Only outline-form glyphs have outlines.
    let plain = generate(ctx, b'A', 0);
    assert_eq!(call(ctx, cb_glyph_get_glyph_form, &[plain], &[]), 0);
    assert_eq!(call(ctx, cb_glyph_refers_outline, &[plain], &[]), 0);
    assert_eq!(call(ctx, cb_glyph_refers_outline, &[0], &[]), 0);

    for glyph in [g, space, plain] {
        ctx.write_u64(OUT + 0x80, glyph);
        assert_eq!(call(ctx, cb_delete_glyph, &[BASE, OUT + 0x80], &[]), 0);
    }
    assert_eq!(call(ctx, cb_glyph_refers_metrics, &[g], &[]), 0);
    assert_eq!(call(ctx, cb_close_font, &[font], &[]), 0);
}

/// sceFontOpenFontFile through a guest mount; sceFontSetFontsOpenMode.
fn open_file(ctx: &mut CpuContext, lib: u64) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../local/tmp/font-open-file-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("kern.ttf"), kerning_font()).unwrap();
    std::fs::write(dir.join("bad.ttf"), [1u8, 2, 3, 4]).unwrap();
    assert!(crate::filesystem::register_guest_path_mount("/rps5e_font_test", &dir));
    let path = |ctx: &CpuContext, s: &str| {
        let p = BASE + 0x2800;
        ctx.write_bytes(p, format!("{s}\0").as_bytes());
        p
    };
    let p = path(ctx, "/rps5e_font_test/kern.ttf");
    assert_eq!(call(ctx, cb_open_font_file, &[0, p, 1, 0, OUT], &[]), ERROR_INVALID_LIBRARY);
    assert_eq!(call(ctx, cb_open_font_file, &[lib, p, 1, 0, 0], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_open_font_file, &[lib, 0, 1, 0, OUT], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_open_font_file, &[lib, p, 0, 0, OUT], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_open_font_file, &[lib, p, 4, 0, OUT], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_open_font_file, &[lib, p, 1, 0, OUT], &[]), 0);
    let font = ctx.read_u64(OUT).unwrap();
    assert_eq!(call(ctx, cb_get_font_glyphs_count, &[font, OUT], &[]), 0);
    assert_eq!(ctx.read_u32(OUT), Some(3));
    assert_eq!(call(ctx, cb_close_font, &[font], &[]), 0);
    let p = path(ctx, "/rps5e_font_test/missing.ttf");
    assert_eq!(call(ctx, cb_open_font_file, &[lib, p, 1, 0, OUT], &[]), ERROR_FS_OPEN_FAILED);
    let p = path(ctx, "/rps5e_font_test/bad.ttf");
    assert_eq!(call(ctx, cb_open_font_file, &[lib, p, 1, 0, OUT], &[]), ERROR_NO_SUPPORT_FORMAT);
    crate::filesystem::unregister_guest_path_mount("/rps5e_font_test");
    std::fs::remove_dir_all(&dir).unwrap();

    assert_eq!(call(ctx, cb_set_fonts_open_mode, &[lib, 2], &[]), 0);
    assert_eq!(call(ctx, cb_set_fonts_open_mode, &[lib, 3], &[]), ERROR_INVALID_PARAMETER);
    assert_eq!(call(ctx, cb_set_fonts_open_mode, &[0, 1], &[]), ERROR_INVALID_LIBRARY);
}

// Font state is process-global and the steps share one mapping, so they
// run as one test.
#[test]
fn guest_font_exports() {
    let vm = ps5vm::VirtualMemory::new();
    vm.map(BASE, SIZE, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);
    guest_sequence(&mut ctx);

    let mem = memory(&mut ctx);
    let lib = create_library(&mut ctx, mem);
    assert_eq!(call(&mut ctx, cb_support_external_fonts, &[lib, 4, 0x52], &[]), 0);
    let renderer = create_renderer(&mut ctx, mem);
    font_styles(&mut ctx, lib, renderer);
    style_frames(&mut ctx, lib, renderer);
    renderer_outline_buffer(&mut ctx, renderer);
    glyph_views(&mut ctx, lib);
    open_file(&mut ctx, lib);
}
