// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only
//
// Unit tests of the `ps5hle::font` module.

use super::*;
use crate::context::CpuRegister::*;

#[test]
fn nids_match_names() {
    for &(nid, name, _) in EXPORTS.iter().chain(EXPORTS_FT).chain(text::EXPORTS).chain(style::EXPORTS).chain(objects::EXPORTS) {
        assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
    }
}

/// Minimal TrueType font: units per em 1000; 'A' is a 500x700 square
/// (advance 600), ' ' is empty (advance 250).
pub(super) fn test_font() -> Vec<u8> {
    fn be16(v: &mut Vec<u8>, x: i32) {
        v.extend_from_slice(&(x as u16).to_be_bytes());
    }
    fn be32(v: &mut Vec<u8>, x: u32) {
        v.extend_from_slice(&x.to_be_bytes());
    }
    let mut head = Vec::new();
    be32(&mut head, 0x0001_0000);
    be32(&mut head, 0x0001_0000);
    be32(&mut head, 0);
    be32(&mut head, 0x5F0F_3CF5);
    be16(&mut head, 0);
    be16(&mut head, 1000);
    head.extend_from_slice(&[0; 16]);
    for v in [0, -200, 500, 800] {
        be16(&mut head, v);
    }
    for v in [0, 8, 2, 0, 0] {
        be16(&mut head, v);
    }
    let mut hhea = Vec::new();
    be32(&mut hhea, 0x0001_0000);
    for v in [800, -200, 0, 600, 0, 0, 500, 1, 0, 0, 0, 0, 0, 0, 0, 3] {
        be16(&mut hhea, v);
    }
    let mut maxp = Vec::new();
    be32(&mut maxp, 0x0000_5000);
    be16(&mut maxp, 3);
    let mut hmtx = Vec::new();
    for (adv, lsb) in [(500, 0), (600, 0), (250, 0)] {
        be16(&mut hmtx, adv);
        be16(&mut hmtx, lsb);
    }
    let mut glyf = Vec::new();
    be16(&mut glyf, 1);
    for v in [0, 0, 500, 700, 3, 0] {
        be16(&mut glyf, v);
    }
    glyf.extend_from_slice(&[1, 1, 1, 1]);
    for v in [0, 500, 0, -500, 0, 0, 700, 0] {
        be16(&mut glyf, v);
    }
    let glyph_len = glyf.len();
    let mut loca = Vec::new();
    for v in [0, 0, glyph_len / 2, glyph_len / 2] {
        be16(&mut loca, v as i32);
    }
    let mut cmap = Vec::new();
    be16(&mut cmap, 0);
    be16(&mut cmap, 1);
    be16(&mut cmap, 3);
    be16(&mut cmap, 10);
    be32(&mut cmap, 12);
    be16(&mut cmap, 12);
    be16(&mut cmap, 0);
    be32(&mut cmap, 16 + 2 * 12);
    be32(&mut cmap, 0);
    be32(&mut cmap, 2);
    for (s, e, g) in [(0x20, 0x20, 2), (0x41, 0x41, 1)] {
        be32(&mut cmap, s);
        be32(&mut cmap, e);
        be32(&mut cmap, g);
    }
    let tables: [(&[u8; 4], Vec<u8>); 7] =
        [(b"cmap", cmap), (b"glyf", glyf), (b"head", head), (b"hhea", hhea), (b"hmtx", hmtx), (b"loca", loca), (b"maxp", maxp)];
    let mut out = Vec::new();
    be32(&mut out, 0x0001_0000);
    for v in [7, 64, 2, 48] {
        be16(&mut out, v);
    }
    let mut offset = 12 + 16 * tables.len();
    let mut body = Vec::new();
    for (tag, data) in &tables {
        out.extend_from_slice(*tag);
        be32(&mut out, 0);
        be32(&mut out, offset as u32);
        be32(&mut out, data.len() as u32);
        body.extend_from_slice(data);
        while body.len() % 4 != 0 {
            body.push(0);
        }
        offset = 12 + 16 * tables.len() + body.len();
    }
    out.extend_from_slice(&body);
    out
}

const BASE: u64 = 0x5C09_0000_0000;

fn set_xmm(v: &[f32]) {
    TEST_XMM.with(|x| {
        let mut x = x.borrow_mut();
        x[..v.len()].copy_from_slice(v);
    });
}

/// Memory + library + renderer + a bound font opened from memory.
fn open_font(ctx: &mut CpuContext) -> (u64, u64) {
    // Memory interface with non-null alloc/free entries.
    ctx.write_u64(BASE + 0x100, 0x1111);
    ctx.write_u64(BASE + 0x108, 0x2222);
    ctx.set(Rdi, BASE);
    ctx.set(Rsi, 0);
    ctx.set(Rdx, 0);
    ctx.set(Rcx, BASE + 0x100);
    ctx.set(R8, 0);
    ctx.set(R9, 0);
    ctx.set(Rsp, BASE + 0x200);
    assert_eq!(cb_memory_init(ctx), 0);

    ctx.set(Rdi, 0);
    let driver = cb_select_library_ft(ctx) as u64;
    assert_ne!(driver, 0);
    let selection = cb_select_renderer_ft(ctx) as u64;
    assert_ne!(selection, 0);

    ctx.set(Rdi, BASE);
    ctx.set(Rsi, driver);
    ctx.set(Rdx, 0);
    ctx.set(Rcx, BASE + 0x300);
    assert_eq!(cb_create_library_with_edition(ctx), 0);
    let lib = ctx.read_u64(BASE + 0x300).unwrap();
    ctx.set(Rdi, lib);
    assert_eq!(cb_support_external_fonts(ctx), 0);
    assert_eq!(cb_support_external_fonts(ctx), ERROR_ALREADY_SPECIFIED);

    let ttf = test_font();
    ctx.write_bytes(BASE + 0x4000, &ttf);
    ctx.set(Rdi, lib);
    ctx.set(Rsi, BASE + 0x4000);
    ctx.set(Rdx, ttf.len() as u64);
    ctx.set(Rcx, 0);
    ctx.set(R8, BASE + 0x308);
    assert_eq!(cb_open_font_memory(ctx), 0);
    let font = ctx.read_u64(BASE + 0x308).unwrap();
    assert_eq!(read_u16(ctx, font), Some(HANDLE_MAGIC));

    ctx.set(Rdi, BASE);
    ctx.set(Rsi, selection);
    ctx.set(Rcx, BASE + 0x310);
    assert_eq!(cb_create_renderer_with_edition(ctx), 0);
    let renderer = ctx.read_u64(BASE + 0x310).unwrap();
    ctx.set(Rdi, font);
    ctx.set(Rsi, renderer);
    assert_eq!(cb_bind_renderer(ctx), 0);
    assert_eq!(cb_bind_renderer(ctx), ERROR_ALREADY_BOUND_RENDERER);
    (font, renderer)
}

#[test]
fn metrics_layout_and_render() {
    let vm = ps5vm::VirtualMemory::new();
    vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);
    let (font, _renderer) = open_font(&mut ctx);

    // Scale 20 px/em: the square is 10x14 px, advance 12 px.
    set_xmm(&[20.0, 20.0]);
    ctx.set(Rdi, font);
    assert_eq!(cb_set_scale_pixel(&mut ctx), 0);
    assert_eq!(cb_setup_render_scale_pixel(&mut ctx), 0);

    ctx.set(Rsi, BASE + 0x400);
    assert_eq!(cb_get_horizontal_layout(&mut ctx), 0);
    // bbox yMin -200, yMax 800 at 20 px/em.
    assert_eq!(ctx.read_f32(BASE + 0x400), Some(16.0));
    assert_eq!(ctx.read_f32(BASE + 0x404), Some(20.0));
    assert_eq!(cb_get_vertical_layout(&mut ctx), 0);
    assert_eq!(ctx.read_f32(BASE + 0x400), Some(-10.0));
    assert_eq!(ctx.read_f32(BASE + 0x404), Some(20.0));

    ctx.set(Rsi, b'A' as u64);
    ctx.set(Rdx, BASE + 0x500);
    assert_eq!(cb_get_render_char_glyph_metrics(&mut ctx), 0);
    let m: Vec<f32> = (0..8).map(|i| ctx.read_f32(BASE + 0x500 + i * 4).unwrap()).collect();
    // Vertical: no vertical tables, so the advance is the hhea line height
    // (800 - -200 units = 20 px), the 14 px glyph is centred in it and the
    // bearing x is half the 12 px advance left of the horizontal bearing.
    assert_eq!(m, vec![10.0, 14.0, 0.0, 14.0, 12.0, -6.0, 3.0, 20.0]);
    ctx.set(Rsi, b'Z' as u64);
    assert_eq!(cb_get_render_char_glyph_metrics(&mut ctx), ERROR_NO_SUPPORT_GLYPH);

    // 32x32, 1 byte per pixel surface.
    const SURF: u64 = BASE + 0x600;
    const BUF: u64 = BASE + 0x1000;
    ctx.set(Rdi, SURF);
    ctx.set(Rsi, BUF);
    ctx.set(Rdx, 32);
    ctx.set(Rcx, 1);
    ctx.set(R8, 32);
    ctx.set(R9, 32);
    cb_render_surface_init(&mut ctx);
    set_xmm(&[2.0, 20.0]);
    ctx.set(Rdi, font);
    ctx.set(Rsi, b'A' as u64);
    ctx.set(Rdx, SURF);
    ctx.set(Rcx, BASE + 0x700);
    ctx.set(R8, BASE + 0x800);
    assert_eq!(cb_render_char_glyph_image_horizontal(&mut ctx), 0);
    let px = ctx.read_bytes(BUF, 32 * 32).unwrap();
    for y in 0..32 {
        for x in 0..32 {
            let inside = (2..12).contains(&x) && (6..20).contains(&y);
            assert_eq!(px[y * 32 + x], if inside { 255 } else { 0 }, "pixel {x},{y}");
        }
    }
    // UpdateRect and image metrics.
    let rect: Vec<u32> = (0..4).map(|i| ctx.read_u32(BASE + 0x818 + i * 4).unwrap()).collect();
    assert_eq!(rect, vec![2, 6, 10, 14]);
    assert_eq!(ctx.read_u64(BASE + 0x808), Some(BUF + 6 * 32 + 2));
    assert_eq!(ctx.read_u32(BASE + 0x838), Some(10));
    assert_eq!(ctx.read_u32(BASE + 0x83C), Some(14));

    // Unbound renderer -> render refused.
    ctx.set(Rdi, font);
    assert_eq!(cb_unbind_renderer(&mut ctx), 0);
    ctx.set(Rsi, b'A' as u64);
    assert_eq!(cb_render_char_glyph_image_horizontal(&mut ctx), ERROR_NOT_BOUND_RENDERER);

    // Glyph objects.
    ctx.set(Rdi, font);
    ctx.set(Rsi, b'A' as u64);
    ctx.set(Rdx, 0);
    ctx.set(Rcx, BASE + 0x900);
    assert_eq!(cb_generate_char_glyph(&mut ctx), 0);
    let g = ctx.read_u64(BASE + 0x900).unwrap();
    ctx.set(Rdi, g);
    ctx.set(Rsi, 5);
    ctx.set(Rdx, BASE + 0x908);
    assert_eq!(cb_glyph_define_attribute(&mut ctx), 0);
    ctx.set(Rsi, 6);
    assert_eq!(cb_glyph_define_attribute(&mut ctx), 0);
    assert_eq!(ctx.read_u32(BASE + 0x908), Some(5));
    ctx.set(Rdi, BASE);
    ctx.set(Rsi, BASE + 0x900);
    assert_eq!(cb_delete_glyph(&mut ctx), 0);
    assert_eq!(cb_delete_glyph(&mut ctx), ERROR_INVALID_GLYPH);

    ctx.set(Rdi, font);
    assert_eq!(cb_close_font(&mut ctx), 0);
    assert_eq!(cb_close_font(&mut ctx), ERROR_INVALID_FONT_HANDLE);
}

/// Text parser used by the string test: one byte per character from
/// textSource->current until textSource->end.
fn byte_parser(_func: u64, args: [u64; 6]) -> u64 {
    let (ts, _order, result) = (args[0], args[1], args[2]);
    // SAFETY: the test maps these guest addresses as host memory.
    unsafe {
        let cur = *((ts + 0x18) as *const u64);
        let end = *((ts + 0x10) as *const u64);
        if cur >= end {
            *((result + 8) as *mut u32) = 0;
            return 0;
        }
        let byte = *(cur as *const u8);
        *((ts + 0x18) as *mut u64) = cur + 1;
        *(result as *mut u64) = 0;
        *((result + 8) as *mut u32) = byte as u32;
        1
    }
}

#[test]
fn strings_and_writing() {
    let vm = ps5vm::VirtualMemory::new();
    const B2: u64 = BASE + 0x10_0000;
    vm.map(B2, 0x10000, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);
    // Reuse the font setup at a separate base.
    ctx.write_u64(B2 + 0x100, 0x1111);
    ctx.write_u64(B2 + 0x108, 0x2222);
    ctx.set(Rdi, B2);
    ctx.set(Rcx, B2 + 0x100);
    ctx.set(Rsp, B2 + 0x200);
    assert_eq!(cb_memory_init(&mut ctx), 0);
    ctx.set(Rdi, B2);
    ctx.set(Rsi, 1);
    ctx.set(Rdx, 0);
    ctx.set(Rcx, B2 + 0x300);
    assert_eq!(cb_create_library_with_edition(&mut ctx), 0);
    let lib = ctx.read_u64(B2 + 0x300).unwrap();
    ctx.set(Rdi, lib);
    assert_eq!(cb_support_external_fonts(&mut ctx), 0);
    let ttf = test_font();
    ctx.write_bytes(B2 + 0x4000, &ttf);
    ctx.set(Rsi, B2 + 0x4000);
    ctx.set(Rdx, ttf.len() as u64);
    ctx.set(Rcx, 0);
    ctx.set(R8, B2 + 0x308);
    assert_eq!(cb_open_font_memory(&mut ctx), 0);
    let font = ctx.read_u64(B2 + 0x308).unwrap();
    // sceFontGetAttribute reads back what DefineAttribute set; the output
    // is checked before the handle.
    ctx.set(Rdi, font);
    ctx.set(Rsi, 5);
    ctx.set(Rdx, 0);
    assert_eq!(cb_define_attribute(&mut ctx), 0);
    assert_eq!(cb_get_attribute(&mut ctx), ERROR_INVALID_PARAMETER);
    ctx.set(Rdx, B2 + 0x310);
    assert_eq!(cb_get_attribute(&mut ctx), 0);
    assert_eq!(ctx.read_u32(B2 + 0x310), Some(5));
    ctx.set(Rdi, 0x1234);
    assert_eq!(cb_get_attribute(&mut ctx), ERROR_INVALID_FONT_HANDLE);
    assert_eq!(cb_glyph_get_attribute(&mut ctx), ERROR_INVALID_GLYPH);
    ctx.set(Rdx, 0);
    assert_eq!(cb_glyph_get_attribute(&mut ctx), ERROR_INVALID_PARAMETER);
    set_xmm(&[10.0, 10.0]);
    ctx.set(Rdi, font);
    assert_eq!(cb_set_scale_pixel(&mut ctx), 0);

    // Text source over "A A".
    set_guest_call(byte_parser);
    ctx.write_bytes(B2 + 0x800, b"A A");
    const TS: u64 = B2 + 0x900;
    ctx.set(Rdi, TS);
    ctx.set(Rsi, B2 + 0x800);
    ctx.set(Rdx, 3);
    ctx.set(Rcx, 0xDEAD);
    ctx.set(R8, 0);
    assert_eq!(text::cb_text_source_init(&mut ctx), 0);
    ctx.set(Rsi, font);
    assert_eq!(text::cb_text_source_set_default_font(&mut ctx), 0);

    ctx.set(Rdi, B2);
    ctx.set(Rsi, TS);
    ctx.set(Rdx, 0);
    ctx.set(Rcx, B2 + 0xA00);
    assert_eq!(text::cb_create_string(&mut ctx), 0);
    let s = ctx.read_u64(B2 + 0xA00).unwrap();
    ctx.set(Rdi, s);
    ctx.set(Rsi, B2 + 0xA08);
    let chars = text::cb_string_refers_text_characters(&mut ctx) as u64;
    assert_eq!(ctx.read_u32(B2 + 0xA08), Some(3));
    // "A" is a strong left-to-right letter: the string picks the LTR form
    // when the source leaves the default horizontal one (measured on
    // hardware).
    assert_eq!(text::cb_string_get_writing_form(&mut ctx), 0x12);

    // Render characters: whole string from the first character.
    ctx.set(Rdi, s);
    ctx.set(Rsi, 0);
    ctx.set(Rdx, 0);
    ctx.set(Rcx, B2 + 0xA10);
    assert_eq!(text::cb_string_refers_render_characters(&mut ctx) as u64, chars);
    assert_eq!(ctx.read_u32(B2 + 0xA10), Some(3));

    // Second character is the space.
    let space = ctx.read_u64(chars + 8).unwrap();
    ctx.set(Rdi, space);
    assert_eq!(text::cb_character_looks_white_space(&mut ctx), 0x20);
    ctx.set(Rdi, chars);
    assert_eq!(text::cb_character_refers_text_next(&mut ctx) as u64, space);
    ctx.set(Rsi, B2 + 0xA20);
    ctx.set(Rdx, B2 + 0xA28);
    assert_eq!(text::cb_character_get_text_font_code(&mut ctx), 0);
    assert_eq!(ctx.read_u64(B2 + 0xA20), Some(font));
    assert_eq!(ctx.read_u32(B2 + 0xA28), Some(b'A' as u32));

    // Writing: advances 6 (A), 2.5 (space), 6 (A) at 10 px/em.
    const W: u64 = B2 + 0xB00;
    ctx.set(Rdi, W);
    ctx.set(Rsi, s);
    ctx.set(Rdx, chars);
    assert_eq!(text::cb_writing_init(&mut ctx), 0);
    // The letter step comes back through a pointer slot; nothing past the
    // slot is written, and there is no step before the first render step.
    const GUARD: u64 = 0x5A5A_5A5A_5A5A_5A5A;
    ctx.write_u64(B2 + 0xC40, B2);
    ctx.write_u64(B2 + 0xC48, GUARD);
    ctx.set(Rsi, B2 + 0xC40);
    assert_eq!(text::cb_writing_refers_render_step_character(&mut ctx), 0);
    assert_eq!((ctx.read_u64(B2 + 0xC40), ctx.read_u64(B2 + 0xC48)), (Some(0), Some(GUARD)));
    let mut xs = Vec::new();
    loop {
        ctx.set(Rdi, W);
        let step = text::cb_writing_refers_render_step(&mut ctx) as u64;
        if step == 0 {
            break;
        }
        xs.push((ctx.read_f32(step).unwrap(), ctx.read_f32(step + 8).unwrap()));
    }
    assert_eq!(xs, vec![(0.0, 6.0), (6.0, 2.5), (8.5, 6.0)]);
    ctx.set(Rdi, W);
    ctx.set(Rsi, B2 + 0xC00);
    assert_eq!(text::cb_writing_get_render_metrics(&mut ctx), 0);
    assert_eq!(ctx.read_f32(B2 + 0xC00), Some(14.5));
    // Extent: top -7 (cap height 7 above the baseline), left 0, right 13.5.
    assert_eq!(ctx.read_f32(B2 + 0xC08), Some(-7.0));
    assert_eq!(ctx.read_f32(B2 + 0xC14), Some(13.5));
    ctx.set(Rsi, B2 + 0xC40);
    ctx.write_u64(B2 + 0xC48, GUARD);
    assert_eq!(text::cb_writing_refers_render_step_character(&mut ctx) as u64, chars + 2 * 0xC8);
    let letter = ctx.read_u64(B2 + 0xC40).unwrap();
    assert_eq!(ctx.read_u64(B2 + 0xC48), Some(GUARD));
    assert!((W..W + 0x100).contains(&letter), "the letter step lives in the writing");
    assert_eq!((ctx.read_f32(letter), ctx.read_f32(letter + 8)), (Some(8.5), Some(6.0)));
    assert_eq!((ctx.read_u32(letter + 0x14), ctx.read_u32(letter + 0x1C)), (Some(2), Some(2)), "texts and glyphs index");
    // The step's character is handed out once; the letter step keeps its
    // contents.
    ctx.set(Rsi, B2 + 0xC40);
    assert_eq!(text::cb_writing_refers_render_step_character(&mut ctx), 0);
    assert_eq!(ctx.read_u64(B2 + 0xC40), Some(0));
    assert_eq!((ctx.read_f32(letter), ctx.read_f32(letter + 8)), (Some(8.5), Some(6.0)));
    ctx.set(Rsi, 0);
    assert_eq!(text::cb_writing_refers_render_step_character(&mut ctx), 0);
    // A new writing: no character before its first step, then each step's
    // character once.
    ctx.set(Rdi, W);
    ctx.set(Rsi, s);
    ctx.set(Rdx, chars);
    assert_eq!(text::cb_writing_init(&mut ctx), 0);
    ctx.set(Rsi, 0);
    assert_eq!(text::cb_writing_refers_render_step_character(&mut ctx), 0);
    for i in 0..2u64 {
        assert_ne!(text::cb_writing_refers_render_step(&mut ctx), 0);
        ctx.set(Rsi, 0);
        assert_eq!(text::cb_writing_refers_render_step_character(&mut ctx) as u64, chars + i * 0xC8);
        assert_eq!(text::cb_writing_refers_render_step_character(&mut ctx), 0);
    }

    // Writing line with two orders.
    ctx.set(Rdi, B2);
    ctx.set(Rsi, 0x10);
    ctx.set(Rdx, 0);
    ctx.set(Rcx, B2 + 0xD00);
    assert_eq!(text::cb_create_writing_line(&mut ctx), 0);
    let line = ctx.read_u64(B2 + 0xD00).unwrap();
    for (i, adv) in [10.0f32, 4.0].iter().enumerate() {
        write_floats(&ctx, B2 + 0xE00, &[*adv, 0.0, -7.0, 0.0, 0.0, *adv]);
        ctx.set(Rdi, line);
        ctx.set(Rsi, 0);
        ctx.set(Rdx, B2 + 0xE00);
        ctx.set(Rcx, 0x100 + i as u64);
        assert_eq!(text::cb_writing_line_writes_order(&mut ctx), 0);
    }
    ctx.set(Rdi, line);
    let s1 = text::cb_writing_line_refers_render_step(&mut ctx) as u64;
    assert_eq!(ctx.read_f32(s1), Some(0.0));
    let s2 = text::cb_writing_line_refers_render_step(&mut ctx) as u64;
    assert_eq!(s1, s2, "one step object per line");
    assert_eq!(ctx.read_f32(s2), Some(10.0));
    assert_eq!(ctx.read_u64(s2 + 0x18), Some(0x101));
    assert_eq!(text::cb_writing_line_refers_render_step(&mut ctx), 0);
    ctx.set(Rsi, B2 + 0xF00);
    assert_eq!(text::cb_writing_line_get_render_metrics(&mut ctx), 0);
    assert_eq!(ctx.read_f32(B2 + 0xF00), Some(14.0));
    assert_eq!(ctx.read_f32(B2 + 0xF14), Some(14.0));
    ctx.set(Rdi, B2 + 0xD00);
    assert_eq!(text::cb_destroy_writing_line(&mut ctx), 0);

    ctx.set(Rdi, B2 + 0xA00);
    assert_eq!(text::cb_destroy_string(&mut ctx), 0);
    assert_eq!(text::cb_destroy_string(&mut ctx), ERROR_INVALID_STRING);
}

/// Exports added alongside the core set, large glyphs and scissoring.
mod extended {
    use super::*;

    const BASE: u64 = 0x5D91_0000_0000;
    const SIZE: u64 = 0x80000;

    /// Memory + library (via sceFontCreateLibrary) + renderer + bound font.
    fn setup(ctx: &mut CpuContext) -> (u64, u64) {
        ctx.write_u64(BASE + 0x100, 0x1111);
        ctx.write_u64(BASE + 0x108, 0x2222);
        ctx.set(Rdi, BASE);
        ctx.set(Rsi, 0);
        ctx.set(Rdx, 0);
        ctx.set(Rcx, BASE + 0x100);
        ctx.set(R8, 0);
        ctx.set(R9, 0);
        ctx.set(Rsp, BASE + 0x200);
        assert_eq!(cb_memory_init(ctx), 0);
        ctx.set(Rdi, 0);
        let selection = cb_select_renderer_ft(ctx) as u64;
        ctx.set(Rdi, BASE);
        ctx.set(Rsi, 1);
        ctx.set(Rdx, BASE + 0x300);
        assert_eq!(cb_create_library(ctx), 0);
        let lib = ctx.read_u64(BASE + 0x300).unwrap();
        ctx.set(Rdi, lib);
        assert_eq!(cb_support_external_fonts(ctx), 0);
        let ttf = test_font();
        ctx.write_bytes(BASE + 0x4000, &ttf);
        ctx.set(Rsi, BASE + 0x4000);
        ctx.set(Rdx, ttf.len() as u64);
        ctx.set(Rcx, 0);
        ctx.set(R8, BASE + 0x308);
        assert_eq!(cb_open_font_memory(ctx), 0);
        let font = ctx.read_u64(BASE + 0x308).unwrap();
        ctx.set(Rdi, BASE);
        ctx.set(Rsi, selection);
        ctx.set(Rcx, BASE + 0x310);
        assert_eq!(cb_create_renderer_with_edition(ctx), 0);
        let renderer = ctx.read_u64(BASE + 0x310).unwrap();
        ctx.set(Rdi, font);
        ctx.set(Rsi, renderer);
        assert_eq!(cb_bind_renderer(ctx), 0);
        (font, lib)
    }

    fn surface(ctx: &mut CpuContext, surf: u64, buf: u64, w: u64, h: u64, bpp: u64) {
        ctx.set(Rdi, surf);
        ctx.set(Rsi, buf);
        ctx.set(Rdx, w * bpp);
        ctx.set(Rcx, bpp);
        ctx.set(R8, w);
        ctx.set(R9, h);
        cb_render_surface_init(ctx);
    }

    fn render(ctx: &mut CpuContext, font: u64, code: u32, surf: u64, x: f32, y: f32) -> HleResult {
        set_xmm(&[x, y]);
        ctx.set(Rdi, font);
        ctx.set(Rsi, code as u64);
        ctx.set(Rdx, surf);
        ctx.set(Rcx, BASE + 0x700);
        ctx.set(R8, BASE + 0x800);
        cb_render_char_glyph_image_horizontal(ctx)
    }

    fn read_rect(ctx: &CpuContext) -> Vec<u32> {
        (0..4).map(|i| ctx.read_u32(BASE + 0x818 + i * 4).unwrap()).collect()
    }

    // Font state is process-global and these steps share one mapping, so
    // they run as one test.
    #[test]
    fn extended_exports() {
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, SIZE, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (font, lib) = setup(&mut ctx);
        large_glyph(&mut ctx, font);
        scissor(&mut ctx, font);
        metrics_and_kerning(&mut ctx, font);
        attributes_and_lifetime(&mut ctx, font, lib);
    }

    /// 400 px/em: 'A' is a 200x280 square, far above 128 px.
    fn large_glyph(ctx: &mut CpuContext, font: u64) {
        set_xmm(&[400.0, 400.0]);
        ctx.set(Rdi, font);
        assert_eq!(cb_setup_render_scale_pixel(ctx), 0);
        const SURF: u64 = BASE + 0x600;
        const BUF: u64 = BASE + 0x10000;
        let (w, h) = (256usize, 320usize);
        for bpp in [1usize, 4] {
            ctx.write_bytes(BUF, &vec![0u8; w * h * bpp]);
            surface(ctx, SURF, BUF, w as u64, h as u64, bpp as u64);
            assert_eq!(render(ctx, font, b'A' as u32, SURF, 10.0, 300.0), 0);
            let px = ctx.read_bytes(BUF, w * h * bpp).unwrap();
            for y in 0..h {
                for x in 0..w {
                    let inside = (10..210).contains(&x) && (20..300).contains(&y);
                    for b in 0..bpp {
                        assert_eq!(px[(y * w + x) * bpp + b], if inside { 255 } else { 0 }, "bpp {bpp} pixel {x},{y}");
                    }
                }
            }
            assert_eq!(read_rect(ctx), vec![10, 20, 200, 280]);
            assert_eq!(ctx.read_u32(BASE + 0x838), Some(200));
            assert_eq!(ctx.read_u32(BASE + 0x83C), Some(280));
            let m: Vec<f32> = (0..5).map(|i| ctx.read_f32(BASE + 0x700 + i * 4).unwrap()).collect();
            assert_eq!(m, vec![200.0, 280.0, 0.0, 280.0, 240.0]);
        }
        // Clipped by the surface: only the visible part is drawn.
        ctx.write_bytes(BUF, &vec![0u8; w * h]);
        surface(ctx, SURF, BUF, w as u64, h as u64, 1);
        assert_eq!(render(ctx, font, b'A' as u32, SURF, 100.0, 200.0), 0);
        assert_eq!(read_rect(ctx), vec![100, 0, 156, 200]);
        let px = ctx.read_bytes(BUF, w * h).unwrap();
        assert_eq!(px.iter().filter(|&&v| v == 255).count(), 156 * 200);
    }

    fn scissor(ctx: &mut CpuContext, font: u64) {
        set_xmm(&[20.0, 20.0]);
        ctx.set(Rdi, font);
        assert_eq!(cb_setup_render_scale_pixel(ctx), 0);
        const SURF: u64 = BASE + 0x680;
        const BUF: u64 = BASE + 0x70000;
        ctx.write_bytes(BUF, &[0u8; 32 * 32]);
        surface(ctx, SURF, BUF, 32, 32, 1);
        ctx.set(Rdi, SURF);
        ctx.set(Rsi, 4);
        ctx.set(Rdx, 8);
        ctx.set(Rcx, 6);
        ctx.set(R8, 10);
        cb_render_surface_set_scissor(ctx);
        let sc: Vec<u32> = (0..4).map(|i| ctx.read_u32(SURF + 0x18 + i * 4).unwrap()).collect();
        assert_eq!(sc, vec![4, 8, 6, 10]);
        // The glyph covers x 2..12, y 6..20; the scissor keeps x 4..6, y 8..10.
        assert_eq!(render(ctx, font, b'A' as u32, SURF, 2.0, 20.0), 0);
        let px = ctx.read_bytes(BUF, 32 * 32).unwrap();
        for y in 0..32 {
            for x in 0..32 {
                let inside = (4..6).contains(&x) && (8..10).contains(&y);
                assert_eq!(px[y * 32 + x], if inside { 255 } else { 0 }, "pixel {x},{y}");
            }
        }
    }

    fn metrics_and_kerning(ctx: &mut CpuContext, font: u64) {
        // Font style 10 px/em, independent of the render style.
        set_xmm(&[10.0, 10.0]);
        ctx.set(Rdi, font);
        assert_eq!(cb_set_scale_pixel(ctx), 0);
        ctx.set(Rsi, b'A' as u64);
        ctx.set(Rdx, BASE + 0x500);
        assert_eq!(cb_get_char_glyph_metrics(ctx), 0);
        let m: Vec<f32> = (0..8).map(|i| ctx.read_f32(BASE + 0x500 + i * 4).unwrap()).collect();
        // Vertical: bearing x half the 6 px advance left, the 7 px glyph
        // centred in the 10 px line height.
        assert_eq!(m, vec![5.0, 7.0, 0.0, 7.0, 6.0, -3.0, 1.5, 10.0]);
        ctx.set(Rsi, 0);
        assert_eq!(cb_get_char_glyph_metrics(ctx), ERROR_NO_SUPPORT_CODE);
        ctx.set(Rsi, b'Z' as u64);
        assert_eq!(cb_get_char_glyph_metrics(ctx), ERROR_NO_SUPPORT_GLYPH);
        ctx.set(Rdi, 0);
        assert_eq!(cb_get_char_glyph_metrics(ctx), ERROR_INVALID_FONT_HANDLE);

        // Rebinding copies the font style to the renderer.
        ctx.set(Rdi, font);
        assert_eq!(cb_rebind_renderer(ctx), 0);
        ctx.set(Rsi, b'A' as u64);
        ctx.set(Rdx, BASE + 0x500);
        assert_eq!(cb_get_render_char_glyph_metrics(ctx), 0);
        assert_eq!(ctx.read_f32(BASE + 0x500), Some(5.0));

        // Kerning: the test font has no kerning data.
        const K: u64 = BASE + 0x540;
        ctx.write_bytes(K, &[0xFF; 16]);
        let kern = |ctx: &mut CpuContext, h: u64, pre: u32, code: u32, out: u64| {
            ctx.set(Rdi, h);
            ctx.set(Rsi, pre as u64);
            ctx.set(Rdx, code as u64);
            ctx.set(Rcx, out);
            cb_get_kerning(ctx)
        };
        assert_eq!(kern(ctx, font, b'A' as u32, b'A' as u32, K), 0);
        assert_eq!(ctx.read_bytes(K, 16).unwrap(), vec![0u8; 16]);
        assert_eq!(kern(ctx, font, 0, b'A' as u32, K), 0);
        assert_eq!(kern(ctx, font, 0x8000_0001, 0x8000_0002, K), 0);
        assert_eq!(kern(ctx, font, b'A' as u32, b'Z' as u32, K), ERROR_NO_SUPPORT_CODE);
        assert_eq!(kern(ctx, font, 0x8000_0001, 0x8000_0003, K), ERROR_NO_SUPPORT_CODE);
        assert_eq!(kern(ctx, font, b'A' as u32, b'A' as u32, 0), ERROR_INVALID_PARAMETER);
        assert_eq!(kern(ctx, 0, b'A' as u32, b'A' as u32, K), ERROR_INVALID_FONT_HANDLE);
    }

    /// Script languages and typographic features are kept per handle;
    /// unset entries read 0.
    fn script_and_typographic_settings(ctx: &mut CpuContext, font: u64) {
        let value = BASE + 0x5A0;
        let call = |ctx: &mut CpuContext, cb: HleCallback, h: u64, key: u64, arg: u64| {
            ctx.set(Rdi, h);
            ctx.set(Rsi, key);
            ctx.set(Rdx, arg);
            cb(ctx)
        };
        let read = |ctx: &CpuContext| ctx.read_u32(value).unwrap() as i32;
        ctx.write_u32(value, u32::MAX);
        assert_eq!(call(ctx, cb_get_script_language, font, 3, value), 0);
        assert_eq!(read(ctx), 0);
        assert_eq!(call(ctx, cb_set_script_language, font, 3, 7), 0);
        assert_eq!(call(ctx, cb_set_script_language, font, 4, 9), 0);
        assert_eq!(call(ctx, cb_get_script_language, font, 3, value), 0);
        assert_eq!(read(ctx), 7);
        assert_eq!(call(ctx, cb_get_script_language, font, 4, value), 0);
        assert_eq!(read(ctx), 9);
        assert_eq!(call(ctx, cb_get_script_language, font, 3, 0), ERROR_INVALID_PARAMETER);
        assert_eq!(call(ctx, cb_set_script_language, 0, 3, 7), ERROR_INVALID_FONT_HANDLE);
        assert_eq!(call(ctx, cb_set_typographic_design, font, 1, 2), 0);
        assert_eq!(call(ctx, cb_get_typographic_design, font, 1, value), 0);
        assert_eq!(read(ctx), 2);
        assert_eq!(call(ctx, cb_get_typographic_design, font, 5, value), 0);
        assert_eq!(read(ctx), 0);
        assert_eq!(call(ctx, cb_get_typographic_design, 0, 1, value), ERROR_INVALID_FONT_HANDLE);
    }

    fn attributes_and_lifetime(ctx: &mut CpuContext, font: u64, lib: u64) {
        script_and_typographic_settings(ctx, font);
        ctx.set(Rdi, font);
        ctx.set(Rsi, 3);
        ctx.set(Rdx, BASE + 0x580);
        assert_eq!(cb_define_attribute(ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x580), Some(0));
        ctx.set(Rsi, 7);
        assert_eq!(cb_define_attribute(ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x580), Some(3));
        ctx.set(Rdi, 0);
        assert_eq!(cb_define_attribute(ctx), ERROR_INVALID_FONT_HANDLE);

        ctx.set(Rdi, font);
        assert_eq!(cb_unbind_renderer(ctx), 0);
        assert_eq!(cb_rebind_renderer(ctx), ERROR_NOT_BOUND_RENDERER);
        assert_eq!(cb_close_font(ctx), 0);

        ctx.write_u64(BASE + 0x590, lib);
        ctx.set(Rdi, BASE + 0x590);
        assert_eq!(cb_destroy_library(ctx), 0);
        assert_eq!(ctx.read_u64(BASE + 0x590), Some(0));
        ctx.write_u64(BASE + 0x590, lib);
        assert_eq!(cb_destroy_library(ctx), ERROR_INVALID_LIBRARY);
        ctx.set(Rdi, 0);
        assert_eq!(cb_destroy_library(ctx), ERROR_INVALID_PARAMETER);

        ctx.set(Rdi, BASE);
        assert_eq!(cb_memory_term(ctx), 0);
        assert_eq!(cb_memory_term(ctx), ERROR_INVALID_MEMORY);
        ctx.set(Rdi, 0);
        assert_eq!(cb_memory_term(ctx), ERROR_INVALID_PARAMETER);
    }

    #[test]
    fn text_codes_and_characters() {
        let vm = ps5vm::VirtualMemory::new();
        const B: u64 = 0x5D92_0000_0000;
        vm.map(B, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_u64(B + 0x100, 0x1111);
        ctx.write_u64(B + 0x108, 0x2222);
        ctx.set(Rdi, B);
        ctx.set(Rsi, 0);
        ctx.set(Rdx, 0);
        ctx.set(Rcx, B + 0x100);
        ctx.set(R8, 0);
        ctx.set(R9, 0);
        ctx.set(Rsp, B + 0x200);
        assert_eq!(cb_memory_init(&mut ctx), 0);

        set_guest_call(byte_parser);
        ctx.write_bytes(B + 0x800, b"ABC");
        const TS: u64 = B + 0x900;
        ctx.set(Rdi, TS);
        ctx.set(Rsi, B + 0x800);
        ctx.set(Rdx, 3);
        ctx.set(Rcx, 0xDEAD);
        ctx.set(R8, 0);
        assert_eq!(text::cb_text_source_init(&mut ctx), 0);
        ctx.set(Rsi, 0x4242);
        assert_eq!(text::cb_text_source_set_default_font(&mut ctx), 0);
        ctx.set(Rdi, B);
        ctx.set(Rsi, TS);
        ctx.set(Rdx, 0);
        ctx.set(Rcx, B + 0xA00);
        assert_eq!(text::cb_create_string(&mut ctx), 0);
        let s = ctx.read_u64(B + 0xA00).unwrap();

        // The parser consumed the text; rewinding restarts it.
        assert_eq!(ctx.read_u64(TS + 0x18), Some(B + 0x803));
        ctx.set(Rdi, TS);
        assert_eq!(text::cb_text_source_rewind(&mut ctx), 0);
        assert_eq!(ctx.read_u64(TS + 0x18), Some(B + 0x800));
        ctx.set(Rdi, 0);
        assert_eq!(text::cb_text_source_rewind(&mut ctx), ERROR_INVALID_PARAMETER);

        ctx.set(Rdi, s);
        assert_eq!(text::cb_string_get_terminate_order(&mut ctx), 0);
        ctx.set(Rsi, B + 0xA08);
        let a = text::cb_string_refers_text_characters(&mut ctx) as u64;
        let b = a + 0xC8;
        let c = b + 0xC8;

        ctx.set(Rdi, c);
        assert_eq!(text::cb_character_refers_text_back(&mut ctx) as u64, b);
        ctx.set(Rdi, a);
        assert_eq!(text::cb_character_refers_text_back(&mut ctx), 0);
        assert_eq!(text::cb_character_looks_format_characters(&mut ctx), 0);
        ctx.set(Rsi, B + 0xA10);
        ctx.write_u32(B + 0xA10, 0xFFFF);
        assert_eq!(text::cb_character_get_syllable_string_state(&mut ctx), 0);
        assert_eq!(ctx.read_u32(B + 0xA10), Some(0));
        ctx.set(Rdi, 0);
        assert_eq!(text::cb_character_get_syllable_string_state(&mut ctx), ERROR_INVALID_PARAMETER);

        // Text codes: walk A -> B, stopping at the terminating character C.
        const CODES: u64 = B + 0xB00;
        ctx.set(Rdi, a);
        ctx.set(Rsi, c);
        ctx.set(Rdx, CODES);
        assert_eq!(text::cb_characters_refers_text_codes(&mut ctx) as u64, CODES);
        assert_eq!(ctx.read_u32(CODES + 8), Some(b'A' as u32));
        ctx.set(Rdi, CODES);
        assert_eq!(text::cb_text_codes_step_next(&mut ctx) as u64, CODES);
        assert_eq!(ctx.read_u32(CODES + 8), Some(b'B' as u32));
        assert_eq!(text::cb_text_codes_step_next(&mut ctx), 0);
        assert_eq!(text::cb_text_codes_step_back(&mut ctx) as u64, CODES);
        assert_eq!(ctx.read_u32(CODES + 8), Some(b'A' as u32));
        assert_eq!(text::cb_text_codes_step_back(&mut ctx), 0);
        ctx.set(Rdi, 0);
        assert_eq!(text::cb_text_codes_step_next(&mut ctx), 0);

        // Invisible mask on a live writing; unknown writings are refused.
        const W: u64 = B + 0xC00;
        ctx.set(Rdi, W);
        ctx.set(Rsi, s);
        ctx.set(Rdx, a);
        assert_eq!(text::cb_writing_init(&mut ctx), 0);
        ctx.set(Rsi, 0);
        assert_eq!(text::cb_writing_set_mask_invisible(&mut ctx), 0);
        ctx.set(Rdi, B + 0xE00);
        assert_eq!(text::cb_writing_set_mask_invisible(&mut ctx), ERROR_INVALID_WRITING);

        ctx.set(Rdi, B + 0xA00);
        assert_eq!(text::cb_destroy_string(&mut ctx), 0);
    }
}

/// Writing lines: visual ordering across embedding levels, spacing
/// attributes and the render-step walk.
mod writing_line {
    use super::*;

    const BASE: u64 = 0x7C63_0000_0000;
    const OUT: u64 = BASE + 0x100;
    const METRICS: u64 = BASE + 0x200;
    const DETAIL: u64 = BASE + 0x300;
    const HANDLES: u64 = BASE + 0x500;
    const SPACE_BEFORE: u64 = 1 << 30;
    const SPACE_AFTER: u64 = 1 << 31;

    type M = [f32; 6];
    const ZERO: M = [0.0; 6];
    const RUN_A: M = [10.0, 0.0, -8.0, 2.0, 1.0, 9.0];
    const RUN_B: M = [6.0, 0.0, -12.0, 3.0, -1.0, 5.0];
    const RUN_C: M = [8.0, 0.0, -4.0, 1.0, 2.0, 5.0];
    const RUN_D: M = [4.0, 0.0, -4.0, 1.0, 0.0, 20.0];
    const A: u64 = 0xA0;
    const B: u64 = 0xB0;
    const C: u64 = 0xC0;
    const D: u64 = 0xD0;

    fn attribute(level: u64, flags: u64) -> u64 {
        0xFA | (level << 8) | flags
    }

    fn create(ctx: &mut CpuContext, form: u64, detail: u64, out: u64) -> (HleResult, u64) {
        ctx.set(Rdi, BASE);
        ctx.set(Rsi, form);
        ctx.set(Rdx, detail);
        ctx.set(Rcx, out);
        let rc = text::cb_create_writing_line(ctx);
        (rc, if out != 0 { ctx.read_u64(out).unwrap() } else { 0 })
    }

    fn order(ctx: &mut CpuContext, line: u64, attr: u64, run: Option<&M>, orderer: u64) -> HleResult {
        if let Some(run) = run {
            write_floats(ctx, METRICS, run);
        }
        ctx.set(Rdi, line);
        ctx.set(Rsi, attr);
        ctx.set(Rdx, if run.is_some() { METRICS } else { 0 });
        ctx.set(Rcx, orderer);
        text::cb_writing_line_writes_order(ctx)
    }

    fn clear(ctx: &mut CpuContext, line: u64) -> HleResult {
        ctx.set(Rdi, line);
        text::cb_writing_line_clear(ctx)
    }

    fn step(ctx: &mut CpuContext, line: u64) -> u64 {
        ctx.set(Rdi, line);
        text::cb_writing_line_refers_render_step(ctx) as u64
    }

    fn floats(ctx: &CpuContext, addr: u64) -> M {
        std::array::from_fn(|i| ctx.read_f32(addr + i as u64 * 4).unwrap())
    }

    fn render(ctx: &mut CpuContext, line: u64) -> M {
        write_floats(ctx, OUT, &[1.0; 6]);
        ctx.set(Rdi, line);
        ctx.set(Rsi, OUT);
        assert_eq!(text::cb_writing_line_get_render_metrics(ctx), 0);
        floats(ctx, OUT)
    }

    /// (advance, head, inline, tail)
    fn space(ctx: &mut CpuContext, line: u64) -> [f32; 4] {
        write_floats(ctx, OUT, &[-1.0; 4]);
        ctx.set(Rdi, line);
        ctx.set(Rsi, OUT);
        ctx.set(Rdx, OUT + 4);
        ctx.set(Rcx, OUT + 8);
        ctx.set(R8, OUT + 12);
        assert_eq!(text::cb_writing_line_get_ordering_space(ctx), 0);
        let f = floats(ctx, OUT);
        [f[0], f[1], f[2], f[3]]
    }

    #[track_caller]
    fn step_is(ctx: &mut CpuContext, line: u64, x: f32, progress: f32, run: &M, orderer: u64) -> u64 {
        let s = step(ctx, line);
        assert_ne!(s, 0, "expected a step");
        let head = floats(ctx, s);
        assert_eq!((head[0], head[1], head[2], head[3], head[4]), (x, 0.0, 0.0, 0.0, progress));
        assert_eq!(ctx.read_u64(s + 0x18), Some(orderer));
        assert_eq!((ctx.read_f32(s + 0x20), ctx.read_f32(s + 0x24)), (Some(0.0), Some(0.0)));
        assert_eq!(&floats(ctx, s + 0x28), run);
        s
    }

    /// Write `levels` with one orderer each, then walk the line and return
    /// the orderer indices in reading order.
    fn visual_order(ctx: &mut CpuContext, line: u64, levels: &[u64]) -> Vec<usize> {
        let run: M = [1.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        assert_eq!(clear(ctx, line), 0);
        for (i, &l) in levels.iter().enumerate() {
            assert_eq!(order(ctx, line, attribute(l, 0), Some(&run), 0x1000 + i as u64), 0);
        }
        let mut seen = Vec::new();
        loop {
            let s = step(ctx, line);
            if s == 0 {
                break;
            }
            seen.push((ctx.read_u64(s + 0x18).unwrap() - 0x1000) as usize);
            assert!(seen.len() <= levels.len());
        }
        seen
    }

    #[test]
    fn writing_line_orders_levels_and_spacing() {
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let ctx = &mut ctx;

        // Creation: output pointer, writing form 0x10..=0x12, detail id.
        assert_eq!(create(ctx, 0x10, 0, 0).0, ERROR_INVALID_PARAMETER);
        assert_eq!(create(ctx, 0x13, 0, OUT), (ERROR_INVALID_PARAMETER, 0));
        assert_eq!(create(ctx, 0, 0, OUT), (ERROR_INVALID_PARAMETER, 0));
        ctx.write_bytes(DETAIL, &[0u8; 0x20]);
        ctx.write_bytes(DETAIL, &0x0FD4u16.to_le_bytes());
        assert_eq!(create(ctx, 0x10, DETAIL, OUT), (ERROR_INVALID_PARAMETER, 0));
        ctx.write_bytes(DETAIL, &0x0FD5u16.to_le_bytes());
        let (rc, line) = create(ctx, 0x10, DETAIL, HANDLES);
        assert_eq!(rc, 0);
        assert_ne!(line, 0);
        let (rc, ltr) = create(ctx, 0x12, 0, HANDLES + 8);
        assert!(rc == 0 && ltr != 0 && ltr != line);
        let (rc, rtl) = create(ctx, 0x11, 0, HANDLES + 0x10);
        assert!(rc == 0 && rtl != 0);

        assert_eq!(space(ctx, line), [0.0; 4]);
        assert_eq!(step(ctx, line), 0);
        assert_eq!(render(ctx, line), ZERO);

        // Rejected orders leave the line untouched.
        assert_eq!(order(ctx, 0, attribute(0, 0), Some(&RUN_A), A), ERROR_INVALID_PARAMETER);
        assert_eq!(order(ctx, line, 1, Some(&RUN_A), A), ERROR_INVALID_PARAMETER);
        assert_eq!(order(ctx, line, 0x1FB, Some(&RUN_A), A), ERROR_INVALID_PARAMETER);
        assert_eq!(order(ctx, line, attribute(0, 0), Some(&RUN_A), 0), ERROR_INVALID_PARAMETER);
        assert_eq!(order(ctx, line, attribute(0x7F, 0), Some(&RUN_A), A), ERROR_INVALID_PARAMETER);
        assert_eq!(step(ctx, line), 0);

        // Two plain orders; render metrics grow one step behind the walk.
        assert_eq!(order(ctx, line, 0, Some(&RUN_A), A), 0);
        assert_eq!(order(ctx, line, attribute(0, 0), Some(&RUN_B), B), 0);
        assert_eq!(space(ctx, line), [16.0, 1.0, 0.0, 1.0]);
        assert_eq!(render(ctx, line), ZERO);
        step_is(ctx, line, 0.0, 0.0, &RUN_A, A);
        assert_eq!(render(ctx, line), ZERO);
        step_is(ctx, line, 10.0, 0.0, &RUN_B, B);
        assert_eq!(render(ctx, line), [10.0, 0.0, -8.0, 2.0, 1.0, 9.0]);
        assert_eq!(step(ctx, line), 0);
        assert_eq!(render(ctx, line), [16.0, 0.0, -12.0, 3.0, 1.0, 15.0]);
        // The walk starts over and resets the render metrics.
        step_is(ctx, line, 0.0, 0.0, &RUN_A, A);
        assert_eq!(render(ctx, line), ZERO);

        // Writing during a walk starts a new line; metrics are optional.
        assert_eq!(order(ctx, line, attribute(0x7E, 0), None, D), 0);
        assert_eq!(space(ctx, line), [0.0; 4]);
        step_is(ctx, line, 0.0, 0.0, &ZERO, D);
        assert_eq!(step(ctx, line), 0);

        // A level 1 run inside level 0 reads right to left.
        assert_eq!(clear(ctx, line), 0);
        assert_eq!(step(ctx, line), 0);
        assert_eq!(order(ctx, line, attribute(0, 0), Some(&RUN_A), A), 0);
        assert_eq!(order(ctx, line, attribute(1, 0), Some(&RUN_B), B), 0);
        assert_eq!(order(ctx, line, attribute(1, 0), Some(&RUN_C), C), 0);
        assert_eq!(order(ctx, line, attribute(0, 0), Some(&RUN_D), D), 0);
        assert_eq!(space(ctx, line), [28.0, 1.0, 0.0, -16.0]);
        step_is(ctx, line, 0.0, 0.0, &RUN_A, A);
        step_is(ctx, line, 10.0, 0.0, &RUN_C, C);
        step_is(ctx, line, 18.0, 0.0, &RUN_B, B);
        step_is(ctx, line, 24.0, 0.0, &RUN_D, D);
        assert_eq!(step(ctx, line), 0);

        // Level 2 nested in level 1 nested in level 0.
        assert_eq!(clear(ctx, line), 0);
        assert_eq!(order(ctx, line, attribute(0, 0), Some(&RUN_A), A), 0);
        assert_eq!(order(ctx, line, attribute(1, 0), Some(&RUN_B), B), 0);
        assert_eq!(order(ctx, line, attribute(2, 0), Some(&RUN_C), C), 0);
        assert_eq!(order(ctx, line, attribute(2, 0), Some(&RUN_D), D), 0);
        assert_eq!(order(ctx, line, attribute(1, 0), Some(&RUN_A), B), 0);
        assert_eq!(order(ctx, line, attribute(0, 0), Some(&RUN_B), A), 0);
        step_is(ctx, line, 0.0, 0.0, &RUN_A, A);
        step_is(ctx, line, 10.0, 0.0, &RUN_A, B);
        step_is(ctx, line, 20.0, 0.0, &RUN_C, C);
        step_is(ctx, line, 28.0, 0.0, &RUN_D, D);
        step_is(ctx, line, 32.0, 0.0, &RUN_B, B);
        step_is(ctx, line, 38.0, 0.0, &RUN_B, A);
        assert_eq!(step(ctx, line), 0);

        // Spacing attributes: inline space and per-step spacing progress.
        assert_eq!(order(ctx, ltr, attribute(0, SPACE_AFTER), Some(&RUN_A), A), 0);
        assert_eq!(space(ctx, ltr), [10.0, 1.0, 0.0, 1.0]);
        assert_eq!(order(ctx, ltr, attribute(0, 0), Some(&RUN_C), C), 0);
        assert_eq!(space(ctx, ltr), [18.0, 1.0, 3.0, 3.0]);
        assert_eq!(order(ctx, ltr, attribute(0, SPACE_BEFORE), Some(&RUN_B), B), 0);
        assert_eq!(space(ctx, ltr), [24.0, 1.0, 5.0, 1.0]);
        step_is(ctx, ltr, 0.0, 0.0, &RUN_A, A);
        step_is(ctx, ltr, 10.0, 3.0 / 5.0, &RUN_C, C);
        step_is(ctx, ltr, 18.0, 1.0, &RUN_B, B);
        assert_eq!(step(ctx, ltr), 0);

        // Spaced orders inside a right-to-left run.
        assert_eq!(clear(ctx, ltr), 0);
        assert_eq!(order(ctx, ltr, attribute(0, SPACE_AFTER), Some(&RUN_A), A), 0);
        assert_eq!(order(ctx, ltr, attribute(1, 0), Some(&RUN_C), C), 0);
        assert_eq!(space(ctx, ltr), [18.0, 1.0, 3.0, 3.0]);
        assert_eq!(order(ctx, ltr, attribute(1, 0), Some(&RUN_B), B), 0);
        assert_eq!(space(ctx, ltr), [24.0, 1.0, 0.0, 3.0]);
        assert_eq!(order(ctx, ltr, attribute(1, SPACE_AFTER), Some(&RUN_D), D), 0);
        assert_eq!(space(ctx, ltr), [28.0, 1.0, -16.0, 3.0]);
        step_is(ctx, ltr, 0.0, 0.0, &RUN_A, A);
        step_is(ctx, ltr, 10.0, 1.0 / -16.0, &RUN_D, D);
        step_is(ctx, ltr, 14.0, 1.0, &RUN_B, B);
        step_is(ctx, ltr, 20.0, 1.0, &RUN_C, C);
        assert_eq!(step(ctx, ltr), 0);

        // Right-to-left line form walks from the tail.
        assert_eq!(order(ctx, rtl, attribute(0, SPACE_AFTER), Some(&RUN_A), A), 0);
        assert_eq!(order(ctx, rtl, attribute(0, 0), Some(&RUN_C), C), 0);
        assert_eq!(space(ctx, rtl), [18.0, 5.0, 15.0, 9.0]);
        step_is(ctx, rtl, 0.0, 0.0, &RUN_C, C);
        step_is(ctx, rtl, 8.0, 1.0, &RUN_A, A);
        assert_eq!(step(ctx, rtl), 0);
        assert_eq!(render(ctx, rtl), [18.0, 0.0, -8.0, 2.0, 2.0, 17.0]);

        assert_eq!(visual_order(ctx, ltr, &[0, 1, 2, 3, 1, 1, 0]), [0, 5, 4, 2, 3, 1, 6]);
        assert_eq!(visual_order(ctx, ltr, &[0, 1, 1, 2, 2, 0, 1, 2, 2, 0]), [0, 3, 4, 2, 1, 5, 7, 8, 6, 9]);
        assert_eq!(visual_order(ctx, ltr, &[0, 2, 3, 3, 2, 1, 0]), [0, 5, 1, 3, 2, 4, 6]);

        // Adjusting written by the caller moves the render metrics.
        let run_e: M = [10.0, 3.0, -8.0, 2.0, 1.0, 9.0];
        let run_f: M = [6.0, -1.0, -12.0, 3.0, -1.0, 5.0];
        assert_eq!(clear(ctx, ltr), 0);
        assert_eq!(order(ctx, ltr, attribute(0, 0), Some(&run_e), A), 0);
        assert_eq!(order(ctx, ltr, attribute(0, 0), Some(&run_f), B), 0);
        let adjusted = step(ctx, ltr);
        assert_ne!(adjusted, 0);
        assert_eq!((ctx.read_f32(adjusted), ctx.read_f32(adjusted + 4)), (Some(0.0), Some(0.0)));
        assert_eq!(ctx.read_u64(adjusted + 0x18), Some(A));
        write_floats(ctx, adjusted + 0x20, &[2.0, 5.0]);
        let second = step(ctx, ltr);
        assert_eq!(second, adjusted);
        assert_eq!(floats(ctx, second)[..2], [10.0, 3.0]);
        assert_eq!(floats(ctx, second + 0x20)[..2], [0.0, 0.0]);
        assert_eq!(render(ctx, ltr), [12.0, 8.0, -3.0, 7.0, 3.0, 11.0]);
        assert_eq!(step(ctx, ltr), 0);
        assert_eq!(render(ctx, ltr), [16.0, 2.0, -9.0, 7.0, 3.0, 15.0]);

        // Ordering space: any pointer may be NULL, but head needs tail.
        ctx.write_f32(OUT, 7.0);
        ctx.set(Rdi, rtl);
        ctx.set(Rsi, 0);
        ctx.set(Rdx, 0);
        ctx.set(Rcx, OUT);
        ctx.set(R8, 0);
        assert_eq!(text::cb_writing_line_get_ordering_space(ctx), 0);
        assert_eq!(ctx.read_f32(OUT), Some(15.0));
        ctx.set(Rsi, OUT);
        ctx.set(Rdx, OUT);
        assert_eq!(text::cb_writing_line_get_ordering_space(ctx), ERROR_INVALID_PARAMETER);
        ctx.set(Rdi, 0);
        ctx.set(Rdx, 0);
        ctx.set(Rcx, 0);
        assert_eq!(text::cb_writing_line_get_ordering_space(ctx), ERROR_INVALID_PARAMETER);
        ctx.set(Rsi, OUT);
        assert_eq!(text::cb_writing_line_get_render_metrics(ctx), ERROR_INVALID_PARAMETER);
        ctx.set(Rdi, rtl);
        ctx.set(Rsi, 0);
        assert_eq!(text::cb_writing_line_get_render_metrics(ctx), ERROR_INVALID_PARAMETER);
        assert_eq!(clear(ctx, 0), ERROR_INVALID_PARAMETER);

        // Unknown handles.
        let not_a_line = BASE + 0x400;
        assert_eq!(step(ctx, 0), 0);
        assert_eq!(clear(ctx, not_a_line), ERROR_INVALID_PARAMETER);
        assert_eq!(step(ctx, not_a_line), 0);
        ctx.set(Rdi, not_a_line);
        ctx.set(Rsi, OUT);
        ctx.set(Rdx, OUT);
        ctx.set(Rcx, OUT);
        ctx.set(R8, OUT);
        assert_eq!(text::cb_writing_line_get_ordering_space(ctx), ERROR_INVALID_PARAMETER);
        ctx.write_u64(HANDLES + 0x18, not_a_line);
        ctx.set(Rdi, HANDLES + 0x18);
        assert_eq!(text::cb_destroy_writing_line(ctx), ERROR_INVALID_PARAMETER);
        assert_eq!(ctx.read_u64(HANDLES + 0x18), Some(not_a_line));

        // Destroy clears the caller's pointer and is idempotent.
        ctx.set(Rdi, HANDLES);
        assert_eq!(text::cb_destroy_writing_line(ctx), 0);
        assert_eq!(ctx.read_u64(HANDLES), Some(0));
        assert_eq!(text::cb_destroy_writing_line(ctx), 0);
        ctx.set(Rdi, 0);
        assert_eq!(text::cb_destroy_writing_line(ctx), 0);
        assert_eq!(clear(ctx, line), ERROR_INVALID_PARAMETER);
        assert_eq!(order(ctx, line, attribute(0, 0), Some(&RUN_A), A), ERROR_INVALID_PARAMETER);
        for p in [HANDLES + 8, HANDLES + 0x10] {
            ctx.set(Rdi, p);
            assert_eq!(text::cb_destroy_writing_line(ctx), 0);
            assert_eq!(ctx.read_u64(p), Some(0));
        }
    }
}
