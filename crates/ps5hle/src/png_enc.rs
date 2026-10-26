// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libScePngEnc — encodes RGBA/BGRA 8-bit images into PNG streams.
//!
//! The encoder context lives in caller-provided memory: a 64-bit magic
//! followed by the maximum image width and maximum filter count. Output is a single-IDAT PNG with
//! 8-bit RGB or RGBA samples; per-row filters are chosen among the ones the
//! caller allows by the smallest sum of absolute residuals.

use crate::context::CpuContext;
use crate::png_dec::codec;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libScePngEnc";

const fn err(code: u32) -> HleResult {
    code as i32 as i64
}

const ERROR_INVALID_ADDR: HleResult = err(0x8069_0101);
const ERROR_INVALID_SIZE: HleResult = err(0x8069_0102);
const ERROR_INVALID_PARAM: HleResult = err(0x8069_0103);
const ERROR_INVALID_HANDLE: HleResult = err(0x8069_0104);
const ERROR_DATA_OVERFLOW: HleResult = err(0x8069_0110);
const ERROR_FATAL: HleResult = err(0x8069_0120);

const MAX_IMAGE_WIDTH: u32 = 1_000_000;
const MAX_IMAGE_HEIGHT: u32 = 1_000_000;
const MAX_FILTER_NUMBER: u32 = 4;
const CREATE_PARAM_SIZE: u32 = 16;
/// Largest filtered image (rows plus filter bytes) accepted for encoding.
const MAX_FILTERED_SIZE: u64 = i32::MAX as u64 / 2;
const COLOR_SPACE_RGB: u16 = 3;
const COLOR_SPACE_RGBA: u16 = 19;
const PIXEL_FORMAT_R8G8B8A8: u16 = 0;
const PIXEL_FORMAT_B8G8R8A8: u16 = 1;
const FILTER_SUB: u16 = 1;
const FILTER_UP: u16 = 2;
const FILTER_AVERAGE: u16 = 4;
const FILTER_PAETH: u16 = 8;
const FILTER_ALL: u16 = FILTER_SUB | FILTER_UP | FILTER_AVERAGE | FILTER_PAETH;
const CONTEXT_MAGIC: u64 = 0x434E_4547_4E50_5341;
const CONTEXT_SIZE: u32 = 16;

/// ScePngEncCreateParam { thisSize, attribute, maxImageWidth, maxFilterNumber } (u32 each)
/// Returns (maxImageWidth, maxFilterNumber).
fn validate_create_param(ctx: &CpuContext, addr: u64) -> Result<(u32, u32), HleResult> {
    if addr == 0 {
        return Err(ERROR_INVALID_ADDR);
    }
    let Some(raw) = ctx.read_bytes(addr, CREATE_PARAM_SIZE as usize) else {
        return Err(ERROR_INVALID_ADDR);
    };
    let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
    let (this_size, attribute, max_width, max_filters) = (u32_at(0), u32_at(4), u32_at(8), u32_at(12));
    if this_size != CREATE_PARAM_SIZE || attribute != 0 || max_filters > MAX_FILTER_NUMBER {
        return Err(ERROR_INVALID_PARAM);
    }
    if max_width == 0 || max_width > MAX_IMAGE_WIDTH {
        return Err(ERROR_INVALID_SIZE);
    }
    Ok((max_width, max_filters))
}

/// (max image width, max filter number) of a live context, None for an
/// invalid handle.
fn context(ctx: &CpuContext, handle: u64) -> Option<(u32, u32)> {
    if handle == 0 || ctx.read_u64(handle)? != CONTEXT_MAGIC {
        return None;
    }
    Some((ctx.read_u32(handle + 8)?, ctx.read_u32(handle + 12)?))
}

fn filter_set(filter_type: u16) -> u8 {
    if filter_type == FILTER_ALL {
        return codec::FILTER_ALL;
    }
    if filter_type == 0 {
        return codec::FILTER_NONE;
    }
    let mut f = 0;
    if filter_type & FILTER_SUB != 0 {
        f |= codec::FILTER_SUB;
    }
    if filter_type & FILTER_UP != 0 {
        f |= codec::FILTER_UP;
    }
    if filter_type & FILTER_AVERAGE != 0 {
        f |= codec::FILTER_AVERAGE;
    }
    if filter_type & FILTER_PAETH != 0 {
        f |= codec::FILTER_PAETH;
    }
    f
}

/// scePngEncQueryMemorySize(const ScePngEncCreateParam*)
fn cb_query_memory_size(ctx: &mut CpuContext) -> HleResult {
    match validate_create_param(ctx, ctx.rdi()) {
        Ok(_) => CONTEXT_SIZE as HleResult,
        Err(e) => e,
    }
}

/// scePngEncCreate(param, memoryAddress, memorySize, ScePngEncHandle*)
fn cb_create(ctx: &mut CpuContext) -> HleResult {
    let (max_width, max_filters) = match validate_create_param(ctx, ctx.rdi()) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let (mem, size, out) = (ctx.rsi(), ctx.rdx() as u32, ctx.rcx());
    if mem == 0 || out == 0 {
        return ERROR_INVALID_ADDR;
    }
    if size < CONTEXT_SIZE {
        return ERROR_INVALID_SIZE;
    }
    let mut b = [0u8; 16];
    b[0..8].copy_from_slice(&CONTEXT_MAGIC.to_le_bytes());
    b[8..12].copy_from_slice(&max_width.to_le_bytes());
    b[12..16].copy_from_slice(&max_filters.to_le_bytes());
    if !ctx.write_bytes(mem, &b) || !ctx.write_u64(out, mem) {
        return ERROR_INVALID_ADDR;
    }
    0
}

/// scePngEncDelete(handle)
fn cb_delete(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    if context(ctx, handle).is_none() {
        return ERROR_INVALID_HANDLE;
    }
    ctx.write_u64(handle, 0);
    0
}

/// scePngEncEncode(handle, const ScePngEncEncodeParam*, ScePngEncOutputInfo*)
/// EncodeParam { imageMemAddr ptr, pngMemAddr ptr, imageMemSize u32,
/// pngMemSize u32, imageWidth u32, imageHeight u32, imagePitch u32,
/// pixelFormat u16, colorSpace u16, bitDepth u16, clutNumber u16,
/// filterType u16, compressionLevel u16 }; OutputInfo { dataSize, processedHeight }.
/// Returns the PNG size in bytes.
fn cb_encode(ctx: &mut CpuContext) -> HleResult {
    let (handle, param, out_info) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let Some((max_width, max_filters)) = context(ctx, handle) else {
        return ERROR_INVALID_HANDLE;
    };
    if param == 0 {
        return ERROR_INVALID_PARAM;
    }
    let Some(raw) = ctx.read_bytes(param, 48) else {
        return ERROR_INVALID_ADDR;
    };
    let u64_at = |o: usize| u64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
    let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
    let u16_at = |o: usize| u16::from_le_bytes(raw[o..o + 2].try_into().unwrap());
    let (image_addr, png_addr) = (u64_at(0), u64_at(8));
    let (image_size, png_size) = (u32_at(16), u32_at(20));
    let (width, height, pitch) = (u32_at(24), u32_at(28), u32_at(32));
    let (pixel_format, color_space, bit_depth) = (u16_at(36), u16_at(38), u16_at(40));
    let (clut_number, filter_type, level) = (u16_at(42), u16_at(44), u16_at(46));
    if image_addr == 0 || png_addr == 0 || image_addr & 3 != 0 {
        return ERROR_INVALID_ADDR;
    }
    if (pixel_format != PIXEL_FORMAT_R8G8B8A8 && pixel_format != PIXEL_FORMAT_B8G8R8A8)
        || (color_space != COLOR_SPACE_RGB && color_space != COLOR_SPACE_RGBA)
        || bit_depth != 8
        || clut_number != 0
        || level > 9
        || filter_type & !FILTER_ALL != 0
        || filter_type.count_ones() > max_filters
    {
        return ERROR_INVALID_PARAM;
    }
    if width == 0 || height == 0 || width > max_width || width > MAX_IMAGE_WIDTH || height > MAX_IMAGE_HEIGHT {
        return ERROR_INVALID_SIZE;
    }
    let channels = if color_space == COLOR_SPACE_RGBA { 4 } else { 3 };
    let row_size = width as u64 * channels as u64;
    if (row_size + 1) * height as u64 > MAX_FILTERED_SIZE
        || png_size == 0
        || (pitch as u64) < width as u64 * 4
        || pitch & 3 != 0
        || pitch as u64 * height as u64 > image_size as u64
    {
        return ERROR_INVALID_SIZE;
    }

    let bgr = pixel_format == PIXEL_FORMAT_B8G8R8A8;
    let mut pixels = Vec::with_capacity(width as usize * height as usize * channels);
    for y in 0..height as u64 {
        let Some(row) = ctx.read_bytes(image_addr + y * pitch as u64, width as usize * 4) else {
            return ERROR_INVALID_ADDR;
        };
        for px in row.chunks_exact(4) {
            pixels.push(px[if bgr { 2 } else { 0 }]);
            pixels.push(px[1]);
            pixels.push(px[if bgr { 0 } else { 2 }]);
            if channels == 4 {
                pixels.push(px[3]);
            }
        }
    }
    let Some(png) = codec::encode(&pixels, width, height, channels as u32, level as u8, filter_set(filter_type)) else {
        return ERROR_FATAL;
    };
    let overflow = png.len() > png_size as usize;
    if out_info != 0 {
        let (size, rows) = if overflow { (0, 0) } else { (png.len() as u32, height) };
        if !ctx.write_u32(out_info, size) || !ctx.write_u32(out_info + 4, rows) {
            return ERROR_INVALID_ADDR;
        }
    }
    if overflow {
        return ERROR_DATA_OVERFLOW;
    }
    if !ctx.write_bytes(png_addr, &png) {
        return ERROR_INVALID_ADDR;
    }
    png.len() as HleResult
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("7aGTPfrqT9s", "scePngEncCreate", cb_create),
    ("RUrWdwTWZy8", "scePngEncDelete", cb_delete),
    ("xgDjJKpcyHo", "scePngEncEncode", cb_encode),
    ("9030RnBDoh4", "scePngEncQueryMemorySize", cb_query_memory_size),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn encode_then_decode_round_trip() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C03_0000_0000;
        vm.map(BASE, 0x20000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let mut cp = Vec::new();
        for v in [16u32, 0, 256, 4] {
            cp.extend_from_slice(&v.to_le_bytes());
        }
        ctx.write_bytes(BASE, &cp);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_query_memory_size(&mut ctx), CONTEXT_SIZE as i64);
        ctx.set(Rsi, BASE + 0x100);
        ctx.set(Rdx, 0x10);
        ctx.set(Rcx, BASE + 0x200);
        assert_eq!(cb_create(&mut ctx), 0);
        let handle = ctx.read_u64(BASE + 0x200).unwrap();

        // 5x3 BGRA image with a 24-byte pitch.
        let (w, h, pitch) = (5u32, 3u32, 24u32);
        let mut bgra = vec![0u8; (pitch * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let o = (y * pitch + x * 4) as usize;
                bgra[o..o + 4].copy_from_slice(&[x as u8 * 40, y as u8 * 60, 7, 100 + x as u8]);
            }
        }
        ctx.write_bytes(BASE + 0x1000, &bgra);
        let mut p = Vec::new();
        p.extend_from_slice(&(BASE + 0x1000).to_le_bytes());
        p.extend_from_slice(&(BASE + 0x2000).to_le_bytes());
        p.extend_from_slice(&(bgra.len() as u32).to_le_bytes());
        p.extend_from_slice(&0x4000u32.to_le_bytes());
        for v in [w, h, pitch] {
            p.extend_from_slice(&v.to_le_bytes());
        }
        for v in [PIXEL_FORMAT_B8G8R8A8, COLOR_SPACE_RGBA, 8, 0, FILTER_ALL, 6] {
            p.extend_from_slice(&v.to_le_bytes());
        }
        ctx.write_bytes(BASE + 0x300, &p);
        ctx.set(Rdi, handle);
        ctx.set(Rsi, BASE + 0x300);
        ctx.set(Rdx, BASE + 0x400);
        let size = cb_encode(&mut ctx);
        assert!(size > 0, "encode failed: {size:#x}");
        assert_eq!(ctx.read_u32(BASE + 0x400), Some(size as u32));
        assert_eq!(ctx.read_u32(BASE + 0x404), Some(h));

        let png = ctx.read_bytes(BASE + 0x2000, size as usize).unwrap();
        let img = codec::decode(&png).unwrap();
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                assert_eq!(&img.pixels[o..o + 4], &[7, y as u8 * 60, x as u8 * 40, 100 + x as u8]);
            }
        }

        // Too small an output buffer reports overflow.
        ctx.write_u32(BASE + 0x300 + 20, 8);
        assert_eq!(cb_encode(&mut ctx), ERROR_DATA_OVERFLOW);
        assert_eq!(ctx.read_u32(BASE + 0x400), Some(0));

        ctx.set(Rdi, handle);
        assert_eq!(cb_delete(&mut ctx), 0);
        assert_eq!(cb_delete(&mut ctx), ERROR_INVALID_HANDLE);
    }

    // -- Encoder unit tests (same expectations as the reference suite) -----

    const BASE: u64 = 0x5D90_0000_0000;
    const PARAM: u64 = BASE + 0x100;
    const INFO: u64 = BASE + 0x180;
    const SRC: u64 = BASE + 0x1000;
    const PNG: u64 = BASE + 0x2000;
    const PNG_CAP: u32 = 1024;

    struct Encoder<'a> {
        ctx: CpuContext<'a>,
        handle: u64,
    }

    #[derive(Clone, Copy)]
    struct Param {
        image: u64,
        png: u64,
        image_size: u32,
        png_size: u32,
        width: u32,
        height: u32,
        pitch: u32,
        pixel_format: u16,
        color_space: u16,
        bit_depth: u16,
        clut_number: u16,
        filter_type: u16,
        level: u16,
    }

    fn encode_param(image: u64, image_size: u32, png: u64, png_size: u32) -> Param {
        Param {
            image,
            png,
            image_size,
            png_size,
            width: 2,
            height: 2,
            pitch: 8,
            pixel_format: PIXEL_FORMAT_R8G8B8A8,
            color_space: COLOR_SPACE_RGBA,
            bit_depth: 8,
            clut_number: 0,
            filter_type: 0,
            level: 6,
        }
    }

    fn write_create(ctx: &CpuContext, addr: u64, v: [u32; 4]) {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        ctx.write_bytes(addr, &b);
    }

    impl<'a> Encoder<'a> {
        /// Encoder context at `slot` (16 bytes inside the scratch page).
        fn new(vm: &'a ps5vm::VirtualMemory, max_filters: u32, slot: u64) -> Self {
            let mut ctx = CpuContext::new(vm);
            write_create(&ctx, BASE + 0x80, [16, 0, MAX_IMAGE_WIDTH, max_filters]);
            ctx.set(Rdi, BASE + 0x80);
            assert_eq!(cb_query_memory_size(&mut ctx), CONTEXT_SIZE as i64);
            ctx.set(Rsi, slot);
            ctx.set(Rdx, CONTEXT_SIZE as u64);
            ctx.set(Rcx, BASE + 0x90);
            assert_eq!(cb_create(&mut ctx), 0);
            let handle = ctx.read_u64(BASE + 0x90).unwrap();
            assert_eq!(handle, slot);
            Encoder { ctx, handle }
        }

        fn encode_with(&mut self, handle: u64, p: Option<&Param>, info: u64) -> i64 {
            if let Some(p) = p {
                let mut b = Vec::new();
                b.extend_from_slice(&p.image.to_le_bytes());
                b.extend_from_slice(&p.png.to_le_bytes());
                for v in [p.image_size, p.png_size, p.width, p.height, p.pitch] {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                for v in [p.pixel_format, p.color_space, p.bit_depth, p.clut_number, p.filter_type, p.level] {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                self.ctx.write_bytes(PARAM, &b);
            }
            self.ctx.set(Rdi, handle);
            self.ctx.set(Rsi, if p.is_some() { PARAM } else { 0 });
            self.ctx.set(Rdx, info);
            cb_encode(&mut self.ctx)
        }

        fn encode(&mut self, p: &Param, info: u64) -> i64 {
            self.encode_with(self.handle, Some(p), info)
        }

        fn delete(&mut self) {
            self.ctx.set(Rdi, self.handle);
            assert_eq!(cb_delete(&mut self.ctx), 0);
            assert_eq!(cb_delete(&mut self.ctx), ERROR_INVALID_HANDLE);
        }
    }

    /// Decode a PNG keeping its own channel count (3 or 4).
    fn decode_native(png: &[u8]) -> (u32, u32, usize, Vec<u8>) {
        let h = codec::parse_header(png).unwrap();
        let img = codec::decode(png).unwrap();
        let ch = if h.color_type == codec::ColorType::Rgba { 4 } else { 3 };
        let px = img.pixels.chunks_exact(4).flat_map(|p| p[..ch].to_vec()).collect();
        (img.width, img.height, ch, px)
    }

    /// The encoder tests share one scratch mapping, so they run as one test.
    #[test]
    fn encoder_reference_suite() {
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        encoder_pixels(&vm);
        encoder_filter_masks(&vm);
        encoder_output_capacity(&vm);
        encoder_validation(&vm);
    }

    fn encoder_pixels(vm: &ps5vm::VirtualMemory) {
        let mut e = Encoder::new(vm, 4, BASE);
        let rgba: [u8; 16] = [255, 0, 13, 0, 4, 210, 6, 255, 7, 8, 220, 64, 111, 12, 113, 128];
        for bgr in [false, true] {
            for alpha in [false, true] {
                for padded in [false, true] {
                    let pitch = if padded { 12u32 } else { 8 };
                    let mut source = vec![0xEEu8; (pitch * 2) as usize];
                    let mut expected = Vec::new();
                    for y in 0..2u32 {
                        for x in 0..2u32 {
                            let px = &rgba[((y * 2 + x) * 4) as usize..][..4];
                            let o = (y * pitch + x * 4) as usize;
                            source[o] = px[if bgr { 2 } else { 0 }];
                            source[o + 1] = px[1];
                            source[o + 2] = px[if bgr { 0 } else { 2 }];
                            source[o + 3] = px[3];
                            expected.extend_from_slice(&px[..if alpha { 4 } else { 3 }]);
                        }
                    }
                    e.ctx.write_bytes(SRC, &source);
                    let mut p = encode_param(SRC, source.len() as u32, PNG, PNG_CAP);
                    p.pixel_format = if bgr { PIXEL_FORMAT_B8G8R8A8 } else { PIXEL_FORMAT_R8G8B8A8 };
                    p.color_space = if alpha { COLOR_SPACE_RGBA } else { COLOR_SPACE_RGB };
                    p.pitch = pitch;
                    let size = e.encode(&p, INFO);
                    assert!(size > 0 && e.ctx.read_u32(INFO) == Some(size as u32));
                    assert_eq!(e.ctx.read_u32(INFO + 4), Some(2));
                    let png = e.ctx.read_bytes(PNG, size as usize).unwrap();
                    let (w, h, ch, px) = decode_native(&png);
                    assert_eq!((w, h, ch), (2, 2, if alpha { 4 } else { 3 }));
                    assert_eq!(px, expected, "bgr={bgr} alpha={alpha} padded={padded}");
                }
            }
        }
        e.delete();
    }

    fn encoder_filter_masks(vm: &ps5vm::VirtualMemory) {
        let mut e = Encoder::new(vm, 4, BASE);
        e.ctx.write_bytes(SRC, &[0u8; 16]);
        let mut p = encode_param(SRC, 16, PNG, PNG_CAP);
        for mask in 0..=15u16 {
            p.filter_type = mask;
            let size = e.encode(&p, 0);
            assert!(size > 0);
            let png = e.ctx.read_bytes(PNG, size as usize).unwrap();
            assert_eq!(&png[37..41], b"IDAT");
            let len = u32::from_be_bytes(png[33..37].try_into().unwrap()) as usize;
            let filtered = miniz_oxide::inflate::decompress_to_vec_zlib(&png[41..41 + len]).unwrap();
            assert_eq!(filtered.len(), 18);
            for row in 0..2 {
                let filter = filtered[row * 9];
                assert!(filter <= 4);
                if filter == 0 {
                    assert!(mask == 0 || mask == 15, "mask {mask}");
                } else {
                    assert!(mask & (1 << (filter - 1)) != 0, "mask {mask} filter {filter}");
                }
            }
        }
        p.filter_type = 16;
        assert_eq!(e.encode(&p, 0), ERROR_INVALID_PARAM);
        for count in 0..=4u32 {
            let mut limited = Encoder::new(vm, count, BASE + 0x20);
            for mask in 0..=15u16 {
                p.filter_type = mask;
                let r = limited.encode(&p, 0);
                if mask.count_ones() <= count {
                    assert!(r > 0, "count {count} mask {mask}");
                } else {
                    assert_eq!(r, ERROR_INVALID_PARAM, "count {count} mask {mask}");
                }
            }
            limited.delete();
        }
        e.delete();
    }

    fn encoder_output_capacity(vm: &ps5vm::VirtualMemory) {
        let mut e = Encoder::new(vm, 4, BASE);
        e.ctx.write_bytes(SRC, &[0u8; 16]);
        let mut p = encode_param(SRC, 16, PNG, PNG_CAP);
        let size = e.encode(&p, 0);
        assert!(size > 0);
        let reference = e.ctx.read_bytes(PNG, size as usize).unwrap();
        const GUARDED: u64 = BASE + 0x3000;
        let guard = vec![0xA5u8; size as usize + 2];
        e.ctx.write_bytes(GUARDED, &guard);
        p.png = GUARDED + 1;
        p.png_size = size as u32;
        assert_eq!(e.encode(&p, INFO), size);
        let g = e.ctx.read_bytes(GUARDED, size as usize + 2).unwrap();
        assert_eq!((g[0], g[size as usize + 1]), (0xA5, 0xA5));
        assert_eq!(&g[1..=size as usize], &reference[..]);
        e.ctx.write_bytes(GUARDED, &guard);
        p.png_size = size as u32 - 1;
        assert_eq!(e.encode(&p, INFO), ERROR_DATA_OVERFLOW);
        assert_eq!((e.ctx.read_u32(INFO), e.ctx.read_u32(INFO + 4)), (Some(0), Some(0)));
        assert_eq!(e.ctx.read_bytes(GUARDED, size as usize + 2).unwrap(), guard);
        e.delete();
    }

    fn encoder_validation(vm: &ps5vm::VirtualMemory) {
        let mut e = Encoder::new(vm, 4, BASE);
        {
            let ctx = &mut e.ctx;
            const CP: u64 = BASE + 0x40;
            write_create(ctx, CP, [16, 0, 2, 4]);
            ctx.set(Rdi, 0);
            assert_eq!(cb_query_memory_size(ctx), ERROR_INVALID_ADDR);
            ctx.set(Rdi, CP);
            ctx.set(Rsi, 0);
            ctx.set(Rdx, CONTEXT_SIZE as u64);
            ctx.set(Rcx, BASE + 0x90);
            assert_eq!(cb_create(ctx), ERROR_INVALID_ADDR);
            ctx.set(Rsi, BASE + 0x60);
            ctx.set(Rdx, 0);
            assert_eq!(cb_create(ctx), ERROR_INVALID_SIZE);
            for (v, rc) in [
                ([16, 0, 0, 4], ERROR_INVALID_SIZE),
                ([16, 0, MAX_IMAGE_WIDTH + 1, 4], ERROR_INVALID_SIZE),
                ([0, 0, 2, 4], ERROR_INVALID_PARAM),
                ([16, 0, 2, 5], ERROR_INVALID_PARAM),
            ] {
                write_create(ctx, CP, v);
                assert_eq!(cb_query_memory_size(ctx), rc, "{v:?}");
            }
        }

        e.ctx.write_bytes(SRC, &[0u8; 16]);
        let valid = encode_param(SRC, 16, PNG, PNG_CAP);
        assert_eq!(e.encode_with(0, Some(&valid), 0), ERROR_INVALID_HANDLE);
        assert_eq!(e.encode_with(e.handle, None, 0), ERROR_INVALID_PARAM);
        for width in [0, MAX_IMAGE_WIDTH + 1, 0x4000_0000, u32::MAX] {
            let mut p = valid;
            p.width = width;
            assert_eq!(e.encode(&p, 0), ERROR_INVALID_SIZE, "width {width}");
        }
        for height in [0, MAX_IMAGE_HEIGHT + 1, u32::MAX] {
            let mut p = valid;
            p.width = 1;
            p.height = height;
            p.image_size = u32::MAX;
            assert_eq!(e.encode(&p, 0), ERROR_INVALID_SIZE, "height {height}");
        }
        for height in [262_144, 400_000, 0x7FFFF] {
            let mut p = valid;
            p.width = 1024;
            p.height = height;
            p.pitch = 4096;
            p.image_size = u32::MAX;
            assert_eq!(e.encode(&p, 0), ERROR_INVALID_SIZE, "height {height}");
        }
        for pitch in [0, 7, 9, 0xFFFF_FFFC] {
            let mut p = valid;
            p.pitch = pitch;
            assert_eq!(e.encode(&p, 0), ERROR_INVALID_SIZE, "pitch {pitch}");
        }
        let mut p = valid;
        p.pitch = 12;
        p.image_size = 20;
        assert_eq!(e.encode(&p, 0), ERROR_INVALID_SIZE);
        let mut p = valid;
        p.image_size -= 1;
        assert_eq!(e.encode(&p, 0), ERROR_INVALID_SIZE);
        let mut p = valid;
        p.image = 0;
        assert_eq!(e.encode(&p, 0), ERROR_INVALID_ADDR);
        p.image = SRC + 1;
        assert_eq!(e.encode(&p, 0), ERROR_INVALID_ADDR);
        let mut p = valid;
        p.bit_depth = 16;
        assert_eq!(e.encode(&p, 0), ERROR_INVALID_PARAM);
        let mut p = valid;
        p.clut_number = 1;
        assert_eq!(e.encode(&p, 0), ERROR_INVALID_PARAM);
        let mut p = valid;
        p.level = 10;
        assert_eq!(e.encode(&p, 0), ERROR_INVALID_PARAM);
        e.delete();
    }
}
