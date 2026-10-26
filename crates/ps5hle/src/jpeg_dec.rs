// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceJpegDec — baseline/progressive JPEG decoding into RGBA/BGRA 8-bit
//! images.
//!
//! The decoder handle lives in caller-provided work memory (0x20 bytes): a
//! self pointer (validity check) and the create attribute. The host decoder
//! reports grayscale images as one channel and everything else as RGB; the
//! output always has four bytes per pixel with the caller's alpha value.
//! Only the normal decode mode without down-scaling is implemented; other
//! modes fail with the not-implemented code and a log line.

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceJpegDec";

const fn err(code: u32) -> HleResult {
    code as i32 as i64
}

const ERROR_INVALID_ADDR: HleResult = err(0x8069_0001);
const ERROR_INVALID_SIZE: HleResult = err(0x8069_0002);
const ERROR_INVALID_PARAM: HleResult = err(0x8069_0003);
const ERROR_INVALID_HANDLE: HleResult = err(0x8069_0004);
const ERROR_INVALID_WORK_MEMORY: HleResult = err(0x8069_0005);
const ERROR_INVALID_DATA: HleResult = err(0x8069_0010);
const ERROR_DECODE_ERROR: HleResult = err(0x8069_0012);
/// Behaviour of the console for this option is unknown.
const ERROR_NOT_IMPLEMENTED: HleResult = err(0x8002_0002);

const ATTRIBUTE_CSC_ORDINARY: u32 = 1;
const ATTRIBUTE_CSC_FULL: u32 = 2;
const MEMORY_SIZE: u32 = 0x20;
const HANDLE_ALIGNMENT: u64 = 8;
const CREATE_PARAM_SIZE: u32 = 0xC;

const DECODE_MODE_NORMAL: u16 = 0;
const DOWN_SCALE_NONE: u16 = 1;

const PIXEL_FORMAT_R8G8B8A8: u16 = 0;
const PIXEL_FORMAT_B8G8R8A8: u16 = 1;

const COLOR_SPACE_RGB: u16 = 3;
const BYTES_PER_PIXEL: u64 = 4;
const MAX_PACKED_DIMENSION: u32 = 32767;
/// Largest JPEG stream accepted from the guest.
const MAX_JPEG_BYTES: u32 = 256 << 20;

const IMAGE_INFO_SIZE: usize = 0x24;
const DECODE_PARAM_SIZE: usize = 0x30;

// ---------------------------------------------------------------------------
// Host codec
// ---------------------------------------------------------------------------

pub(crate) mod codec {
    use zune_core::bytestream::ZCursor;
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Header {
        pub width: u32,
        pub height: u32,
        /// 1 for grayscale, 3 otherwise.
        pub channels: u32,
    }

    pub struct Image {
        pub header: Header,
        /// `channels` bytes per pixel, tightly packed.
        pub pixels: Vec<u8>,
    }

    fn options(channels: u32) -> DecoderOptions {
        let space = if channels == 1 { ColorSpace::Luma } else { ColorSpace::RGB };
        DecoderOptions::default()
            .set_max_width(u16::MAX as usize)
            .set_max_height(u16::MAX as usize)
            .jpeg_set_out_colorspace(space)
    }

    pub fn parse_header(jpeg: &[u8]) -> Option<Header> {
        if jpeg.is_empty() {
            return None;
        }
        let mut d = JpegDecoder::new_with_options(ZCursor::new(jpeg), options(3));
        d.decode_headers().ok()?;
        let info = d.info()?;
        if info.width == 0 || info.height == 0 || info.components == 0 {
            return None;
        }
        let channels = if info.components >= 3 { 3 } else { 1 };
        Some(Header { width: info.width as u32, height: info.height as u32, channels })
    }

    pub fn decode(jpeg: &[u8]) -> Option<Image> {
        let header = parse_header(jpeg)?;
        let mut d = JpegDecoder::new_with_options(ZCursor::new(jpeg), options(header.channels));
        let pixels = d.decode().ok()?;
        let expected = header.width as usize * header.height as usize * header.channels as usize;
        (pixels.len() == expected).then_some(Image { header, pixels })
    }
}

use codec::Header;

// ---------------------------------------------------------------------------
// Guest structures
// ---------------------------------------------------------------------------

/// SceJpegDecCreateParam { thisSize u32, attribute u32, maxImageWidth u32 }
fn validate_create_param(ctx: &CpuContext, addr: u64) -> Result<u32, HleResult> {
    if addr == 0 {
        return Err(ERROR_INVALID_PARAM);
    }
    let (Some(size), Some(attribute)) = (ctx.read_u32(addr), ctx.read_u32(addr + 4)) else {
        return Err(ERROR_INVALID_ADDR);
    };
    if size != CREATE_PARAM_SIZE {
        return Err(ERROR_INVALID_SIZE);
    }
    if attribute != ATTRIBUTE_CSC_ORDINARY && attribute != ATTRIBUTE_CSC_FULL {
        return Err(ERROR_INVALID_PARAM);
    }
    Ok(attribute)
}

const CTX_SELF: u64 = 0;
const CTX_ATTRIBUTE: u64 = 8;

fn context_valid(ctx: &CpuContext, handle: u64) -> bool {
    handle != 0 && handle % HANDLE_ALIGNMENT == 0 && ctx.read_u64(handle + CTX_SELF) == Some(handle)
}

fn check_mode(function: &str, decode_mode: u16, down_scale: u16) -> Result<(), HleResult> {
    if decode_mode != DECODE_MODE_NORMAL {
        log::error!("{function}: decode mode {decode_mode} is not implemented");
        return Err(ERROR_NOT_IMPLEMENTED);
    }
    if down_scale != DOWN_SCALE_NONE {
        log::error!("{function}: down scale {down_scale} is not implemented");
        return Err(ERROR_NOT_IMPLEMENTED);
    }
    Ok(())
}

/// SceJpegDecImageInfo { imageWidth u32, imageHeight u32, colorSpace u16,
/// componentNumber u16, componentId[4], samplingFactor[4],
/// coefficientMemSize u32, suitableCscAttribute u32, outputImageWidth u32,
/// outputImageHeight u32 }
fn image_info(h: &Header) -> [u8; IMAGE_INFO_SIZE] {
    let mut b = [0u8; IMAGE_INFO_SIZE];
    b[0..4].copy_from_slice(&h.width.to_le_bytes());
    b[4..8].copy_from_slice(&h.height.to_le_bytes());
    b[8..10].copy_from_slice(&COLOR_SPACE_RGB.to_le_bytes());
    b[10..12].copy_from_slice(&(h.channels as u16).to_le_bytes());
    b[0x18..0x1C].copy_from_slice(&ATTRIBUTE_CSC_ORDINARY.to_le_bytes());
    b[0x1C..0x20].copy_from_slice(&h.width.to_le_bytes());
    b[0x20..0x24].copy_from_slice(&h.height.to_le_bytes());
    b
}

fn read_jpeg(ctx: &CpuContext, addr: u64, size: u32) -> Option<Vec<u8>> {
    if size > MAX_JPEG_BYTES {
        return None;
    }
    ctx.read_bytes(addr, size as usize)
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

/// sceJpegDecQueryMemorySize(const SceJpegDecCreateParam*)
fn cb_query_memory_size(ctx: &mut CpuContext) -> HleResult {
    match validate_create_param(ctx, ctx.rdi()) {
        Ok(_) => MEMORY_SIZE as HleResult,
        Err(e) => e,
    }
}

/// sceJpegDecCreate(param, memoryAddress, memorySize, SceJpegDecHandle*)
fn cb_create(ctx: &mut CpuContext) -> HleResult {
    let (mem, mem_size, out) = (ctx.rsi(), ctx.rdx() as u32, ctx.rcx());
    let attribute = match validate_create_param(ctx, ctx.rdi()) {
        Ok(a) => a,
        Err(e) => return e,
    };
    if mem == 0 || out == 0 {
        return ERROR_INVALID_ADDR;
    }
    if mem_size < MEMORY_SIZE {
        return ERROR_INVALID_WORK_MEMORY;
    }
    let handle = (mem + HANDLE_ALIGNMENT - 1) & !(HANDLE_ALIGNMENT - 1);
    if !ctx.write_u64(handle + CTX_SELF, handle)
        || !ctx.write_u32(handle + CTX_ATTRIBUTE, attribute)
        || !ctx.write_u64(out, handle)
    {
        return ERROR_INVALID_ADDR;
    }
    0
}

/// sceJpegDecDelete(handle)
fn cb_delete(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    if !context_valid(ctx, handle) {
        return ERROR_INVALID_HANDLE;
    }
    ctx.write_u64(handle + CTX_SELF, 0);
    0
}

/// sceJpegDecParseHeader(const SceJpegDecParseParam*, SceJpegDecImageInfo*)
/// ParseParam { jpegMemAddr ptr, jpegMemSize u32, decodeMode u16, downScale u16 }
fn cb_parse_header(ctx: &mut CpuContext) -> HleResult {
    let (param, info) = (ctx.rdi(), ctx.rsi());
    if param == 0 {
        return ERROR_INVALID_PARAM;
    }
    let Some(raw) = ctx.read_bytes(param, 0x10) else {
        return ERROR_INVALID_ADDR;
    };
    let jpeg_addr = u64::from_le_bytes(raw[0..8].try_into().unwrap());
    let jpeg_size = u32::from_le_bytes(raw[8..12].try_into().unwrap());
    let decode_mode = u16::from_le_bytes([raw[12], raw[13]]);
    let down_scale = u16::from_le_bytes([raw[14], raw[15]]);
    if jpeg_addr == 0 || info == 0 {
        return ERROR_INVALID_ADDR;
    }
    if jpeg_size == 0 {
        return ERROR_INVALID_SIZE;
    }
    if let Err(e) = check_mode("sceJpegDecParseHeader", decode_mode, down_scale) {
        return e;
    }
    let Some(jpeg) = read_jpeg(ctx, jpeg_addr, jpeg_size) else {
        return ERROR_INVALID_ADDR;
    };
    let Some(h) = codec::parse_header(&jpeg) else {
        return ERROR_INVALID_DATA;
    };
    if !ctx.write_bytes(info, &image_info(&h)) {
        return ERROR_INVALID_ADDR;
    }
    0
}

/// sceJpegDecDecode(handle, const SceJpegDecDecodeParam*, SceJpegDecImageInfo*)
/// DecodeParam { jpegMemAddr ptr, imageMemAddr ptr, coefficientMemAddr ptr,
/// jpegMemSize u32, imageMemSize u32, coefficientMemSize u32, decodeMode u16,
/// downScale u16, pixelFormat u16, alphaValue u16, imagePitch u32 }
///
/// Returns (width << 16 | height) on success when both fit in 15 bits.
fn cb_decode(ctx: &mut CpuContext) -> HleResult {
    let (handle, param, info) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if !context_valid(ctx, handle) {
        return ERROR_INVALID_HANDLE;
    }
    if param == 0 {
        return ERROR_INVALID_PARAM;
    }
    let Some(raw) = ctx.read_bytes(param, DECODE_PARAM_SIZE) else {
        return ERROR_INVALID_ADDR;
    };
    let u64_at = |o: usize| u64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
    let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
    let u16_at = |o: usize| u16::from_le_bytes(raw[o..o + 2].try_into().unwrap());
    let (jpeg_addr, image_addr) = (u64_at(0), u64_at(8));
    let (jpeg_size, image_size) = (u32_at(0x18), u32_at(0x1C));
    let (decode_mode, down_scale) = (u16_at(0x24), u16_at(0x26));
    let (pixel_format, alpha_value, pitch) = (u16_at(0x28), u16_at(0x2A), u32_at(0x2C));
    // The image must be 4-byte aligned; the alpha value is a byte
    // (measured on a PS5).
    if jpeg_addr == 0 || image_addr == 0 || image_addr & 3 != 0 {
        return ERROR_INVALID_ADDR;
    }
    if jpeg_size == 0 || image_size == 0 {
        return ERROR_INVALID_SIZE;
    }
    if pixel_format != PIXEL_FORMAT_R8G8B8A8 && pixel_format != PIXEL_FORMAT_B8G8R8A8 {
        return ERROR_INVALID_PARAM;
    }
    if alpha_value > 0xFF {
        return ERROR_INVALID_PARAM;
    }
    if let Err(e) = check_mode("sceJpegDecDecode", decode_mode, down_scale) {
        return e;
    }
    let Some(jpeg) = read_jpeg(ctx, jpeg_addr, jpeg_size) else {
        return ERROR_INVALID_ADDR;
    };
    let Some(h) = codec::parse_header(&jpeg) else {
        return ERROR_INVALID_DATA;
    };
    let row_size = h.width as u64 * BYTES_PER_PIXEL;
    let pitch = if pitch == 0 { row_size } else { pitch as u64 };
    if pitch < row_size {
        return ERROR_INVALID_PARAM;
    }
    if (h.height as u64 - 1) * pitch + row_size > image_size as u64 {
        return ERROR_INVALID_SIZE;
    }
    let image = match codec::decode(&jpeg) {
        Some(i) if i.header.width == h.width && i.header.height == h.height => i,
        _ => return ERROR_DECODE_ERROR,
    };
    let swap = pixel_format == PIXEL_FORMAT_B8G8R8A8;
    let alpha = alpha_value as u8;
    let channels = image.header.channels as usize;
    let mut line = vec![0u8; row_size as usize];
    for y in 0..h.height as usize {
        let src = &image.pixels[y * h.width as usize * channels..(y + 1) * h.width as usize * channels];
        for (px, s) in line.chunks_exact_mut(4).zip(src.chunks_exact(channels)) {
            if channels == 1 {
                px[..3].fill(s[0]);
            } else {
                px[..3].copy_from_slice(&s[..3]);
            }
            if swap {
                px.swap(0, 2);
            }
            px[3] = alpha;
        }
        if !ctx.write_bytes(image_addr + y as u64 * pitch, &line) {
            return ERROR_INVALID_ADDR;
        }
    }
    if info != 0 && !ctx.write_bytes(info, &image_info(&h)) {
        return ERROR_INVALID_ADDR;
    }
    if h.width > MAX_PACKED_DIMENSION || h.height > MAX_PACKED_DIMENSION {
        return 0;
    }
    ((h.width << 16) | h.height) as HleResult
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("JPh3Zgg0Zwc", "sceJpegDecCreate", cb_create),
    ("1kzQRoWEgSA", "sceJpegDecDecode", cb_decode),
    ("Hwh11+m5KoI", "sceJpegDecDelete", cb_delete),
    ("LSinoSQH790", "sceJpegDecParseHeader", cb_parse_header),
    ("uNAUmANZMEw", "sceJpegDecQueryMemorySize", cb_query_memory_size),
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
    use crate::jpeg_enc;

    const BASE: u64 = 0x5D94_0000_0000;
    const JPEG: u64 = BASE + 0x1000;
    const PARAM: u64 = BASE + 0x300;
    const INFO: u64 = BASE + 0x400;
    const IMAGE: u64 = BASE + 0x4000;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// Test image with the guest test's byte pattern (10 + i * 13) as
    /// full-resolution YCbCr (or luma) planes.
    fn test_jpeg(width: usize, height: usize, channels: usize) -> Vec<u8> {
        let mut planes = vec![Vec::with_capacity(width * height); channels];
        for i in 0..width * height {
            for (c, plane) in planes.iter_mut().enumerate() {
                plane.push((10 + (i * channels + c) * 13) as u8);
            }
        }
        let sampling = if channels == 1 { jpeg_enc::Sampling::Gray } else { jpeg_enc::Sampling::Yuv444 };
        jpeg_enc::encode(&jpeg_enc::Image { width, height, planes }, 90, sampling)
    }

    /// The tests map the same guest range; run them one at a time.
    static VM_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn setup(vm: &ps5vm::VirtualMemory) -> CpuContext<'_> {
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        CpuContext::new(vm)
    }

    fn parse(ctx: &mut CpuContext, addr: u64, size: u32, mode: u16, scale: u16, info: u64) -> HleResult {
        let mut p = addr.to_le_bytes().to_vec();
        p.extend_from_slice(&size.to_le_bytes());
        p.extend_from_slice(&mode.to_le_bytes());
        p.extend_from_slice(&scale.to_le_bytes());
        ctx.write_bytes(PARAM, &p);
        ctx.set(Rdi, PARAM);
        ctx.set(Rsi, info);
        cb_parse_header(ctx)
    }

    fn create_param(ctx: &mut CpuContext, size: u32, attribute: u32) -> u64 {
        let mut p = size.to_le_bytes().to_vec();
        p.extend_from_slice(&attribute.to_le_bytes());
        p.extend_from_slice(&7u32.to_le_bytes());
        ctx.write_bytes(BASE + 0x100, &p);
        BASE + 0x100
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(
        ctx: &mut CpuContext,
        handle: u64,
        jpeg_size: u32,
        image_size: u32,
        mode: u16,
        scale: u16,
        format: u16,
        alpha: u16,
        pitch: u32,
        info: u64,
    ) -> HleResult {
        let mut p = JPEG.to_le_bytes().to_vec();
        p.extend_from_slice(&IMAGE.to_le_bytes());
        p.extend_from_slice(&0u64.to_le_bytes());
        p.extend_from_slice(&jpeg_size.to_le_bytes());
        p.extend_from_slice(&image_size.to_le_bytes());
        p.extend_from_slice(&0u32.to_le_bytes());
        for v in [mode, scale, format, alpha] {
            p.extend_from_slice(&v.to_le_bytes());
        }
        p.extend_from_slice(&pitch.to_le_bytes());
        ctx.write_bytes(BASE + 0x500, &p);
        ctx.set(Rdi, handle);
        ctx.set(Rsi, BASE + 0x500);
        ctx.set(Rdx, info);
        cb_decode(ctx)
    }

    #[test]
    fn parse_header_validation_and_info() {
        let _guard = VM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = setup(&vm);
        let jpeg = test_jpeg(7, 5, 3);
        ctx.write_bytes(JPEG, &jpeg);
        let n = jpeg.len() as u32;
        ctx.set(Rdi, 0);
        assert_eq!(cb_parse_header(&mut ctx), ERROR_INVALID_PARAM);
        assert_eq!(parse(&mut ctx, JPEG, n, 0, 1, 0), ERROR_INVALID_ADDR);
        assert_eq!(parse(&mut ctx, 0, 0, 0, 1, INFO), ERROR_INVALID_ADDR);
        assert_eq!(parse(&mut ctx, JPEG, 0, 0, 1, INFO), ERROR_INVALID_SIZE);
        // Down-scaling is not implemented: an error, not a fake success.
        assert_eq!(parse(&mut ctx, JPEG, n, 0, 2, INFO), ERROR_NOT_IMPLEMENTED);
        assert_eq!(parse(&mut ctx, JPEG, n, 1, 1, INFO), ERROR_NOT_IMPLEMENTED);

        let gray = test_jpeg(3, 2, 1);
        ctx.write_bytes(BASE + 0x8000, &gray);
        assert_eq!(parse(&mut ctx, BASE + 0x8000, gray.len() as u32, 0, 1, INFO), 0);
        let info = ctx.read_bytes(INFO, IMAGE_INFO_SIZE).unwrap();
        assert_eq!(&info[0..12], &[3, 0, 0, 0, 2, 0, 0, 0, 3, 0, 1, 0]);

        ctx.write_bytes(INFO, &[0xAA; IMAGE_INFO_SIZE]);
        assert_eq!(parse(&mut ctx, JPEG, n, 0, 1, INFO), 0);
        let info = ctx.read_bytes(INFO, IMAGE_INFO_SIZE).unwrap();
        let mut expected = vec![7, 0, 0, 0, 5, 0, 0, 0, 3, 0, 3, 0];
        expected.extend_from_slice(&[0; 12]); // component ids, sampling, coefficient size
        expected.extend_from_slice(&[1, 0, 0, 0, 7, 0, 0, 0, 5, 0, 0, 0]);
        assert_eq!(info, expected);

        ctx.write_bytes(JPEG, &[0, 1, 2, 3]);
        assert_eq!(parse(&mut ctx, JPEG, 4, 0, 1, INFO), ERROR_INVALID_DATA);
    }

    #[test]
    fn query_create_delete() {
        let _guard = VM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = setup(&vm);
        ctx.set(Rdi, 0);
        assert_eq!(cb_query_memory_size(&mut ctx), ERROR_INVALID_PARAM);
        let p = create_param(&mut ctx, 8, 1);
        ctx.set(Rdi, p);
        assert_eq!(cb_query_memory_size(&mut ctx), ERROR_INVALID_SIZE);
        for attribute in [0, 3] {
            let p = create_param(&mut ctx, 12, attribute);
            ctx.set(Rdi, p);
            assert_eq!(cb_query_memory_size(&mut ctx), ERROR_INVALID_PARAM);
        }
        let p = create_param(&mut ctx, 12, 2);
        ctx.set(Rdi, p);
        assert_eq!(cb_query_memory_size(&mut ctx), MEMORY_SIZE as i64);
        let p = create_param(&mut ctx, 12, 1);
        ctx.set(Rdi, p);
        assert_eq!(cb_query_memory_size(&mut ctx), MEMORY_SIZE as i64);

        ctx.set(Rsi, BASE + 0x201);
        ctx.set(Rdx, 0);
        ctx.set(Rcx, BASE + 0x280);
        assert_eq!(cb_create(&mut ctx), ERROR_INVALID_WORK_MEMORY);
        ctx.set(Rdx, MEMORY_SIZE as u64);
        ctx.set(Rcx, 0);
        assert_eq!(cb_create(&mut ctx), ERROR_INVALID_ADDR);
        ctx.set(Rcx, BASE + 0x280);
        assert_eq!(cb_create(&mut ctx), 0);
        let handle = ctx.read_u64(BASE + 0x280).unwrap();
        assert_eq!(handle, BASE + 0x208);
        ctx.set(Rdi, 0);
        assert_eq!(cb_delete(&mut ctx), ERROR_INVALID_HANDLE);
        ctx.set(Rdi, handle);
        assert_eq!(cb_delete(&mut ctx), 0);
        assert_eq!(cb_delete(&mut ctx), ERROR_INVALID_HANDLE);
    }

    #[test]
    fn decode_rgba_bgra_pitch_and_errors() {
        let _guard = VM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = setup(&vm);
        let p = create_param(&mut ctx, 12, 1);
        ctx.set(Rdi, p);
        ctx.set(Rsi, BASE + 0x200);
        ctx.set(Rdx, MEMORY_SIZE as u64);
        ctx.set(Rcx, BASE + 0x280);
        assert_eq!(cb_create(&mut ctx), 0);
        let handle = ctx.read_u64(BASE + 0x280).unwrap();

        let jpeg = test_jpeg(7, 5, 3);
        ctx.write_bytes(JPEG, &jpeg);
        let n = jpeg.len() as u32;
        let reference = codec::decode(&jpeg).unwrap();
        assert_eq!(reference.header, codec::Header { width: 7, height: 5, channels: 3 });
        let size = 7 * 5 * 4;

        assert_eq!(decode(&mut ctx, 0, n, size, 0, 1, 0, 0xFF, 28, INFO), ERROR_INVALID_HANDLE);
        ctx.set(Rdi, handle);
        ctx.set(Rsi, 0);
        assert_eq!(cb_decode(&mut ctx), ERROR_INVALID_PARAM);
        assert_eq!(decode(&mut ctx, handle, n, size, 0, 1, 2, 0xFF, 28, INFO), ERROR_INVALID_PARAM);
        assert_eq!(decode(&mut ctx, handle, n, size, 0, 2, 0, 0xFF, 28, INFO), ERROR_NOT_IMPLEMENTED);
        assert_eq!(decode(&mut ctx, handle, n, size, 1, 1, 0, 0xFF, 28, INFO), ERROR_NOT_IMPLEMENTED);
        assert_eq!(decode(&mut ctx, handle, 0, size, 0, 1, 0, 0xFF, 28, INFO), ERROR_INVALID_SIZE);
        assert_eq!(decode(&mut ctx, handle, n, size, 0, 1, 0, 0xFF, 27, INFO), ERROR_INVALID_PARAM);
        assert_eq!(decode(&mut ctx, handle, n, size - 1, 0, 1, 0, 0xFF, 28, INFO), ERROR_INVALID_SIZE);
        // An alpha value above a byte is PARAM (measured on a PS5).
        assert_eq!(decode(&mut ctx, handle, n, size, 0, 1, 0, 0x100, 28, INFO), ERROR_INVALID_PARAM);

        ctx.write_bytes(INFO, &[0; IMAGE_INFO_SIZE]);
        assert_eq!(decode(&mut ctx, handle, n, size, 0, 1, 0, 0xFF, 28, INFO), (7 << 16) | 5);
        assert_eq!(ctx.read_u32(INFO), Some(7));
        assert_eq!(ctx.read_u32(INFO + 0x20), Some(5));
        let rgba = ctx.read_bytes(IMAGE, size as usize).unwrap();
        for p in 0..35 {
            assert_eq!(&rgba[p * 4..p * 4 + 3], &reference.pixels[p * 3..p * 3 + 3]);
            assert_eq!(rgba[p * 4 + 3], 0xFF);
        }

        // BGRA with a wider pitch; padding bytes are left alone.
        let pitch = 7 * 4 + 8;
        ctx.write_bytes(IMAGE, &vec![0xEE; (4 * pitch + 28) as usize]);
        assert_eq!(decode(&mut ctx, handle, n, 4 * pitch + 28, 0, 1, 1, 0x77, pitch, 0), (7 << 16) | 5);
        let bgra = ctx.read_bytes(IMAGE, (4 * pitch + 28) as usize).unwrap();
        for y in 0..5 {
            for x in 0..7 {
                let px = &bgra[y * pitch as usize + x * 4..][..4];
                let s = &rgba[(y * 7 + x) * 4..][..4];
                assert_eq!(px, &[s[2], s[1], s[0], 0x77]);
            }
            if y < 4 {
                assert!(bgra[y * pitch as usize + 28..(y + 1) * pitch as usize].iter().all(|&b| b == 0xEE));
            }
        }

        // Grayscale replicates luma into RGB.
        let gray = test_jpeg(3, 2, 1);
        ctx.write_bytes(JPEG, &gray);
        assert_eq!(decode(&mut ctx, handle, gray.len() as u32, 24, 0, 1, 0, 9, 0, 0), (3 << 16) | 2);
        let reference = codec::decode(&gray).unwrap();
        let out = ctx.read_bytes(IMAGE, 24).unwrap();
        for p in 0..6 {
            let g = reference.pixels[p];
            assert_eq!(&out[p * 4..p * 4 + 4], &[g, g, g, 9]);
        }

        ctx.write_bytes(JPEG, &[0, 1, 2, 3]);
        assert_eq!(decode(&mut ctx, handle, 4, 64, 0, 1, 0, 0, 16, INFO), ERROR_INVALID_DATA);
        ctx.set(Rdi, handle);
        assert_eq!(cb_delete(&mut ctx), 0);
    }
}
