// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceJpegEnc HLE: baseline JPEG encoding of 8-bit images.
//!
//! A handle is the 32-byte aligned address inside the memory block given to
//! Create. The live handles are kept here, so the encoder keeps no state in
//! guest memory. Streams are JFIF: the Annex K tables scaled to the quality
//! (IJG scaling), 4:4:4, 4:2:2 or 4:2:0 chroma, or a single grayscale
//! component, without restart markers.
//!
//! Not implemented: MJPEG mode and restart intervals. Both report
//! INVALID_PARAM here.

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};
use std::f32::consts::FRAC_1_SQRT_2;
use std::sync::{LazyLock, Mutex, MutexGuard};

const LIB: &str = "libSceJpegEnc";

const ERR_INVALID_ADDR: HleResult = 0x8065_0101u32 as i32 as i64;
const ERR_INVALID_SIZE: HleResult = 0x8065_0102u32 as i32 as i64;
const ERR_INVALID_PARAM: HleResult = 0x8065_0103u32 as i32 as i64;
const ERR_INVALID_HANDLE: HleResult = 0x8065_0104u32 as i32 as i64;

/// Memory an encoder needs (what Query reports).
const MEMORY_SIZE: u32 = 0x800;
const HANDLE_ALIGNMENT: u64 = 0x20;
/// JpegEncCreateParam { u32 size; u32 attr; }
const CREATE_PARAM_SIZE: u32 = 8;
const ATTRIBUTE_NONE: u32 = 0;
/// JpegEncEncodeParam, 0x30 bytes.
const ENCODE_PARAM_SIZE: usize = 0x30;
/// JpegEncOutputInfo { u32 size; u32 height; }
const OUTPUT_INFO_SIZE: usize = 8;

const PIXEL_R8G8B8A8: u16 = 0;
const PIXEL_B8G8R8A8: u16 = 1;
const PIXEL_Y8U8Y8V8: u16 = 10;
const PIXEL_Y8: u16 = 11;
const ENCODE_MODE_NORMAL: u16 = 0;
const ENCODE_MODE_MJPEG: u16 = 1;
const COLOR_SPACE_YCC: u16 = 1;
const COLOR_SPACE_GRAYSCALE: u16 = 2;
const SAMPLING_FULL: u8 = 0;
const SAMPLING_422: u8 = 1;
const SAMPLING_420: u8 = 2;

const MAX_IMAGE_DIMENSION: u32 = 0xFFFF;
const MAX_IMAGE_PITCH: u32 = 0x0FFF_FFFF;
const MAX_IMAGE_SIZE: u64 = 0x7FFF_FFFF;

/// Handles of encoders created and not yet deleted.
static LIVE: Mutex<Vec<u64>> = Mutex::new(Vec::new());

fn live() -> MutexGuard<'static, Vec<u64>> {
    LIVE.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// JpegEncEncodeParam as the title lays it out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EncodeParam {
    image: u64,
    jpeg: u64,
    image_size: u32,
    jpeg_size: u32,
    width: u32,
    height: u32,
    pitch: u32,
    pixel_format: u16,
    encode_mode: u16,
    color_space: u16,
    sampling: u8,
    compression_ratio: u8,
    restart_interval: i32,
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn u16_at(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(b[off..off + 2].try_into().unwrap())
}

impl EncodeParam {
    fn parse(b: &[u8]) -> Self {
        Self {
            image: u64::from_le_bytes(b[0x00..0x08].try_into().unwrap()),
            jpeg: u64::from_le_bytes(b[0x08..0x10].try_into().unwrap()),
            image_size: u32_at(b, 0x10),
            jpeg_size: u32_at(b, 0x14),
            width: u32_at(b, 0x18),
            height: u32_at(b, 0x1c),
            pitch: u32_at(b, 0x20),
            pixel_format: u16_at(b, 0x24),
            encode_mode: u16_at(b, 0x26),
            color_space: u16_at(b, 0x28),
            sampling: b[0x2a],
            compression_ratio: b[0x2b],
            restart_interval: u32_at(b, 0x2c) as i32,
        }
    }

    /// Bytes per row the encoder reads from the image (its pixel data).
    fn row_bytes(&self) -> usize {
        let w = self.width as usize;
        match self.pixel_format {
            PIXEL_R8G8B8A8 | PIXEL_B8G8R8A8 => w * 4,
            PIXEL_Y8U8Y8V8 => w.div_ceil(2) * 4,
            _ => w,
        }
    }
}

/// Checks in the order of the reference model; null pointers are checked
/// by the caller before the structure is read.
fn validate(p: &EncodeParam) -> Result<(), HleResult> {
    let gray_input = p.pixel_format == PIXEL_Y8;
    if p.image == 0 {
        return Err(ERR_INVALID_ADDR);
    }
    if !gray_input && p.image % 4 != 0 {
        return Err(ERR_INVALID_ADDR);
    }
    if p.jpeg == 0 {
        return Err(ERR_INVALID_ADDR);
    }
    if p.image_size == 0 || p.jpeg_size == 0 {
        return Err(ERR_INVALID_SIZE);
    }
    if p.width == 0 || p.height == 0 {
        return Err(ERR_INVALID_PARAM);
    }
    if p.width > MAX_IMAGE_DIMENSION || p.height > MAX_IMAGE_DIMENSION {
        return Err(ERR_INVALID_PARAM);
    }
    if p.pitch == 0 || p.pitch > MAX_IMAGE_PITCH {
        return Err(ERR_INVALID_PARAM);
    }
    if !gray_input && p.pitch % 4 != 0 {
        return Err(ERR_INVALID_PARAM);
    }
    let required = p.height as u64 * p.pitch as u64;
    if required > MAX_IMAGE_SIZE || required > p.image_size as u64 {
        return Err(ERR_INVALID_PARAM);
    }
    if p.encode_mode != ENCODE_MODE_NORMAL && p.encode_mode != ENCODE_MODE_MJPEG {
        return Err(ERR_INVALID_PARAM);
    }
    if p.color_space != COLOR_SPACE_YCC && p.color_space != COLOR_SPACE_GRAYSCALE {
        return Err(ERR_INVALID_PARAM);
    }
    if p.sampling != SAMPLING_FULL && p.sampling != SAMPLING_422 && p.sampling != SAMPLING_420 {
        return Err(ERR_INVALID_PARAM);
    }
    if p.restart_interval > MAX_IMAGE_DIMENSION as i32 {
        return Err(ERR_INVALID_PARAM);
    }
    match p.pixel_format {
        PIXEL_R8G8B8A8 | PIXEL_B8G8R8A8 => {
            if p.pitch / 4 < p.width || p.color_space != COLOR_SPACE_YCC || p.sampling == SAMPLING_FULL {
                return Err(ERR_INVALID_PARAM);
            }
            Ok(())
        }
        PIXEL_Y8U8Y8V8 => {
            if p.pitch / 2 < ((p.width + 1) & !1) || p.color_space != COLOR_SPACE_YCC || p.sampling == SAMPLING_FULL {
                return Err(ERR_INVALID_PARAM);
            }
            Ok(())
        }
        PIXEL_Y8 => {
            if p.pitch < p.width || p.color_space != COLOR_SPACE_GRAYSCALE || p.sampling != SAMPLING_FULL {
                return Err(ERR_INVALID_PARAM);
            }
            Ok(())
        }
        _ => Err(ERR_INVALID_PARAM),
    }
}

/// JPEG quality from the compression ratio (the reference mapping, not
/// checked on a console).
fn quality_of(ratio: u8) -> u8 {
    (100 - ratio as i32 * 99 / 255).clamp(1, 100) as u8
}

/// Reads a JpegEncCreateParam at `addr`.
fn create_param(ctx: &CpuContext, addr: u64) -> Result<(), HleResult> {
    if addr == 0 {
        return Err(ERR_INVALID_ADDR);
    }
    let (Some(size), Some(attr)) = (ctx.read_u32(addr), ctx.read_u32(addr + 4)) else {
        return Err(ERR_INVALID_ADDR);
    };
    if size != CREATE_PARAM_SIZE {
        return Err(ERR_INVALID_SIZE);
    }
    if attr != ATTRIBUTE_NONE {
        return Err(ERR_INVALID_PARAM);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Guest entry points
// ---------------------------------------------------------------------------

/// sceJpegEncQueryMemorySize(const JpegEncCreateParam* param)
fn cb_query_memory_size(ctx: &mut CpuContext) -> HleResult {
    match create_param(ctx, ctx.rdi()) {
        Ok(()) => MEMORY_SIZE as HleResult,
        Err(e) => e,
    }
}

/// sceJpegEncCreate(param, memory, memorySize, void** handle)
fn cb_create(ctx: &mut CpuContext) -> HleResult {
    let (param, memory, memory_size, handle_out) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u32, ctx.rcx());
    if let Err(e) = create_param(ctx, param) {
        return e;
    }
    if memory == 0 {
        return ERR_INVALID_ADDR;
    }
    if memory_size < MEMORY_SIZE {
        return ERR_INVALID_SIZE;
    }
    if handle_out == 0 {
        return ERR_INVALID_ADDR;
    }
    let handle = (memory + HANDLE_ALIGNMENT - 1) & !(HANDLE_ALIGNMENT - 1);
    if !ctx.write_u64(handle_out, handle) {
        return ERR_INVALID_ADDR;
    }
    let mut l = live();
    if !l.contains(&handle) {
        l.push(handle);
    }
    0
}

/// sceJpegEncDelete(handle)
fn cb_delete(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    let mut l = live();
    match l.iter().position(|&h| h == handle) {
        Some(i) => {
            l.swap_remove(i);
            0
        }
        None => ERR_INVALID_HANDLE,
    }
}

/// sceJpegEncEncode(handle, const JpegEncEncodeParam* param, JpegEncOutputInfo* info)
///
/// Returns the JPEG size in bytes: the title's wrapper takes a non-zero
/// result as success and uses it as the length.
fn cb_encode(ctx: &mut CpuContext) -> HleResult {
    let (handle, param_addr, info_addr) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if !live().contains(&handle) {
        return ERR_INVALID_HANDLE;
    }
    if param_addr == 0 {
        return ERR_INVALID_ADDR;
    }
    let Some(raw) = ctx.read_bytes(param_addr, ENCODE_PARAM_SIZE) else {
        return ERR_INVALID_ADDR;
    };
    let p = EncodeParam::parse(&raw);
    if let Err(e) = validate(&p) {
        return e;
    }
    // Not implemented (see the module comment).
    if p.encode_mode == ENCODE_MODE_MJPEG || p.restart_interval != 0 {
        return ERR_INVALID_PARAM;
    }

    let row_bytes = p.row_bytes();
    let mut rows = Vec::with_capacity(row_bytes * p.height as usize);
    for y in 0..p.height as u64 {
        let Some(row) = ctx.read_bytes(p.image + y * p.pitch as u64, row_bytes) else {
            return ERR_INVALID_ADDR;
        };
        rows.extend_from_slice(&row);
    }
    let image = image_from_pixels(&p, &rows);
    let sampling = match (p.pixel_format, p.sampling) {
        (PIXEL_Y8, _) => Sampling::Gray,
        (_, SAMPLING_422) => Sampling::Yuv422,
        (_, SAMPLING_420) => Sampling::Yuv420,
        _ => Sampling::Yuv444,
    };
    let jpeg = encode(&image, quality_of(p.compression_ratio), sampling);
    if jpeg.len() > p.jpeg_size as usize {
        return ERR_INVALID_SIZE;
    }
    if !ctx.write_bytes(p.jpeg, &jpeg) {
        return ERR_INVALID_ADDR;
    }
    if info_addr != 0 {
        let mut info = [0u8; OUTPUT_INFO_SIZE];
        info[0..4].copy_from_slice(&(jpeg.len() as u32).to_le_bytes());
        info[4..8].copy_from_slice(&p.height.to_le_bytes());
        if !ctx.write_bytes(info_addr, &info) {
            return ERR_INVALID_ADDR;
        }
    }
    jpeg.len() as HleResult
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("o6ZgXfFdWXQ", "sceJpegEncQueryMemorySize", cb_query_memory_size),
    ("K+rocojkr-I", "sceJpegEncCreate", cb_create),
    ("QbrU0cUghEM", "sceJpegEncEncode", cb_encode),
    ("j1LyMdaM+C0", "sceJpegEncDelete", cb_delete),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

// ---------------------------------------------------------------------------
// Pixels
// ---------------------------------------------------------------------------

/// Component planes at full resolution (Y, Cb, Cr; Y only for grayscale).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub planes: Vec<Vec<u8>>,
}

/// Source pixels of the title as full-resolution planes. `rows` holds
/// `height` rows of `p.row_bytes()` bytes each.
fn image_from_pixels(p: &EncodeParam, rows: &[u8]) -> Image {
    let (w, h, row_bytes) = (p.width as usize, p.height as usize, p.row_bytes());
    let gray = p.pixel_format == PIXEL_Y8;
    let mut y_plane = vec![0u8; w * h];
    let mut cb = vec![128u8; if gray { 0 } else { w * h }];
    let mut cr = vec![128u8; if gray { 0 } else { w * h }];
    for r in 0..h {
        let row = &rows[r * row_bytes..(r + 1) * row_bytes];
        for x in 0..w {
            let i = r * w + x;
            match p.pixel_format {
                PIXEL_R8G8B8A8 => {
                    let (yy, u, v) = rgb_to_ycc(row[x * 4], row[x * 4 + 1], row[x * 4 + 2]);
                    y_plane[i] = yy;
                    cb[i] = u;
                    cr[i] = v;
                }
                PIXEL_B8G8R8A8 => {
                    let (yy, u, v) = rgb_to_ycc(row[x * 4 + 2], row[x * 4 + 1], row[x * 4]);
                    y_plane[i] = yy;
                    cb[i] = u;
                    cr[i] = v;
                }
                PIXEL_Y8U8Y8V8 => {
                    let pair = (x / 2) * 4;
                    y_plane[i] = row[pair + (x % 2) * 2];
                    cb[i] = row[pair + 1];
                    cr[i] = row[pair + 3];
                }
                _ => y_plane[i] = row[x],
            }
        }
    }
    let planes = if gray { vec![y_plane] } else { vec![y_plane, cb, cr] };
    Image { width: w, height: h, planes }
}

/// JFIF full-range RGB to YCbCr.
fn rgb_to_ycc(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let (r, g, b) = (r as f32, g as f32, b as f32);
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = -0.168_736 * r - 0.331_264 * g + 0.5 * b + 128.0;
    let cr = 0.5 * r - 0.418_688 * g - 0.081_312 * b + 128.0;
    (clamp_byte(y), clamp_byte(cb), clamp_byte(cr))
}

fn clamp_byte(v: f32) -> u8 {
    (v + 0.5).clamp(0.0, 255.0) as u8
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Output chroma layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampling {
    Yuv444,
    Yuv422,
    Yuv420,
    Gray,
}

/// (horizontal max, vertical max, per-component (h, v) sampling factors).
fn layout(sampling: Sampling) -> (usize, usize, Vec<(usize, usize)>) {
    match sampling {
        Sampling::Yuv444 => (1, 1, vec![(1, 1), (1, 1), (1, 1)]),
        Sampling::Yuv422 => (2, 1, vec![(2, 1), (1, 1), (1, 1)]),
        Sampling::Yuv420 => (2, 2, vec![(2, 2), (1, 1), (1, 1)]),
        Sampling::Gray => (1, 1, vec![(1, 1)]),
    }
}

/// Zigzag scan position -> natural (row-major) index.
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47,
    55, 62, 63,
];

/// Annex K.1 luminance quantization table, natural order.
const LUMA_Q: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, //
    12, 12, 14, 19, 26, 58, 60, 55, //
    14, 13, 16, 24, 40, 57, 69, 56, //
    14, 17, 22, 29, 51, 87, 80, 62, //
    18, 22, 37, 56, 68, 109, 103, 77, //
    24, 35, 55, 64, 81, 104, 113, 92, //
    49, 64, 78, 87, 103, 121, 120, 101, //
    72, 92, 95, 98, 112, 100, 103, 99,
];

/// Annex K.2 chrominance quantization table, natural order.
const CHROMA_Q: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, //
    18, 21, 26, 66, 99, 99, 99, 99, //
    24, 26, 56, 99, 99, 99, 99, 99, //
    47, 66, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99,
];

/// IJG quality scaling of a base table.
fn scaled_table(base: &[u16; 64], quality: u8) -> [u16; 64] {
    let q = quality.clamp(1, 100) as u32;
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let mut out = [0u16; 64];
    for (o, &b) in out.iter_mut().zip(base) {
        *o = ((b as u32 * scale + 50) / 100).clamp(1, 255) as u16;
    }
    out
}

/// Annex K.3 Huffman tables: code counts per length, then symbol values.
const DC_LUMA_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_CHROMA_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const DC_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const AC_LUMA_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
const AC_CHROMA_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const AC_LUMA_VALS: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14,
    0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09,
    0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a,
    0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65,
    0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
    0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9,
    0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca,
    0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
];
const AC_CHROMA_VALS: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32,
    0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16,
    0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64,
    0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86,
    0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8,
    0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
];

/// Canonical Huffman code per symbol.
struct Huff {
    code: [u16; 256],
    len: [u8; 256],
}

impl Huff {
    fn new(bits: &[u8; 16], vals: &[u8]) -> Self {
        let mut h = Huff { code: [0; 256], len: [0; 256] };
        let mut code: u16 = 0;
        let mut k = 0;
        for (l, &n) in bits.iter().enumerate() {
            for _ in 0..n {
                let sym = vals[k] as usize;
                h.code[sym] = code;
                h.len[sym] = (l + 1) as u8;
                code += 1;
                k += 1;
            }
            code <<= 1;
        }
        h
    }
}

/// Entropy-coded bits with 0xFF byte stuffing.
#[derive(Default)]
struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    filled: u32,
}

impl BitWriter {
    fn put(&mut self, value: u32, len: u32) {
        for i in (0..len).rev() {
            self.acc = (self.acc << 1) | ((value >> i) & 1);
            self.filled += 1;
            if self.filled == 8 {
                let byte = self.acc as u8;
                self.out.push(byte);
                if byte == 0xFF {
                    self.out.push(0x00);
                }
                self.acc = 0;
                self.filled = 0;
            }
        }
    }

    fn put_huff(&mut self, h: &Huff, sym: u8) {
        self.put(h.code[sym as usize] as u32, h.len[sym as usize] as u32);
    }

    /// Pads the last byte with 1 bits and returns the stream.
    fn finish(mut self) -> Vec<u8> {
        while self.filled != 0 {
            self.put(1, 1);
        }
        self.out
    }
}

/// Magnitude category and the value bits of a coefficient.
fn magnitude(v: i32) -> (u8, u32) {
    if v == 0 {
        return (0, 0);
    }
    let a = v.unsigned_abs();
    let cat = 32 - a.leading_zeros();
    let bits = if v < 0 { (v - 1) as u32 & ((1u32 << cat) - 1) } else { v as u32 };
    (cat as u8, bits)
}

/// Forward DCT-II of a level-shifted 8x8 block (natural order).
static DCT_T: LazyLock<[[f32; 8]; 8]> = LazyLock::new(|| {
    let mut t = [[0f32; 8]; 8];
    for (u, row) in t.iter_mut().enumerate() {
        let cu = if u == 0 { FRAC_1_SQRT_2 } else { 1.0 };
        for (x, v) in row.iter_mut().enumerate() {
            *v = cu / 2.0 * (((2 * x + 1) as f32) * (u as f32) * std::f32::consts::PI / 16.0).cos();
        }
    }
    t
});

fn fdct(block: &[f32; 64]) -> [f32; 64] {
    let t = &*DCT_T;
    let mut tmp = [0f32; 64];
    for y in 0..8 {
        for u in 0..8 {
            let mut s = 0f32;
            for x in 0..8 {
                s += t[u][x] * block[y * 8 + x];
            }
            tmp[y * 8 + u] = s;
        }
    }
    let mut out = [0f32; 64];
    for v in 0..8 {
        for u in 0..8 {
            let mut s = 0f32;
            for y in 0..8 {
                s += t[v][y] * tmp[y * 8 + u];
            }
            out[v * 8 + u] = s;
        }
    }
    out
}

/// Quantized coefficients (natural order), rounded half away from zero.
fn quantize(coef: &[f32; 64], q: &[u16; 64]) -> [i32; 64] {
    let mut out = [0i32; 64];
    for i in 0..64 {
        out[i] = (coef[i] / q[i] as f32).round() as i32;
    }
    out
}

/// Entropy-codes one block in zigzag order; `dc_pred` is the component's DC predictor.
fn encode_block(w: &mut BitWriter, q: &[i32; 64], dc_pred: &mut i32, dc: &Huff, ac: &Huff) {
    let diff = q[0] - *dc_pred;
    *dc_pred = q[0];
    let (cat, bits) = magnitude(diff);
    w.put_huff(dc, cat);
    w.put(bits, cat as u32);

    let mut run = 0u32;
    for &v in q.iter().skip(1) {
        if v == 0 {
            run += 1;
            continue;
        }
        while run >= 16 {
            w.put_huff(ac, 0xF0);
            run -= 16;
        }
        let (cat, bits) = magnitude(v);
        w.put_huff(ac, ((run << 4) as u8) | cat);
        w.put(bits, cat as u32);
        run = 0;
    }
    if run > 0 {
        w.put_huff(ac, 0x00);
    }
}

/// Plane of `cw` x `ch` samples: each is the mean of an `fx` x `fy` block of
/// `src` (edge samples repeated past the image).
fn downsample(src: &[u8], w: usize, h: usize, fx: usize, fy: usize, cw: usize, ch: usize) -> Vec<u8> {
    let n = (fx * fy) as u32;
    let mut out = vec![0u8; cw * ch];
    for y in 0..ch {
        for x in 0..cw {
            let mut sum = 0u32;
            for dy in 0..fy {
                let sy = (y * fy + dy).min(h - 1);
                for dx in 0..fx {
                    let sx = (x * fx + dx).min(w - 1);
                    sum += src[sy * w + sx] as u32;
                }
            }
            out[y * cw + x] = ((sum + n / 2) / n) as u8;
        }
    }
    out
}

/// One 8x8 block at (x0, y0) of a plane with row stride `stride`, level-shifted.
fn block_at(plane: &[u8], stride: usize, x0: usize, y0: usize) -> [f32; 64] {
    let mut b = [0f32; 64];
    for y in 0..8 {
        for x in 0..8 {
            b[y * 8 + x] = plane[(y0 + y) * stride + x0 + x] as f32 - 128.0;
        }
    }
    b
}

/// Encodes `img` as a baseline JFIF stream.
pub fn encode(img: &Image, quality: u8, sampling: Sampling) -> Vec<u8> {
    let (hmax, vmax, comps) = layout(sampling);
    let (mcu_w, mcu_h) = (8 * hmax, 8 * vmax);
    let mcux = img.width.div_ceil(mcu_w);
    let mcuy = img.height.div_ceil(mcu_h);
    let (pad_w, pad_h) = (mcux * mcu_w, mcuy * mcu_h);

    // Component planes at their own resolution, padded to whole MCUs.
    let planes: Vec<(usize, Vec<u8>)> = comps
        .iter()
        .enumerate()
        .map(|(c, &(h, v))| {
            let (fx, fy) = (hmax / h, vmax / v);
            let (cw, ch) = (pad_w / fx, pad_h / fy);
            (cw, downsample(&img.planes[c], img.width, img.height, fx, fy, cw, ch))
        })
        .collect();

    let luma_q = scaled_table(&LUMA_Q, quality);
    let chroma_q = scaled_table(&CHROMA_Q, quality);
    let color = comps.len() == 3;
    let qtab = |c: usize| if c == 0 { &luma_q } else { &chroma_q };

    let dc_l = Huff::new(&DC_LUMA_BITS, &DC_VALS);
    let ac_l = Huff::new(&AC_LUMA_BITS, &AC_LUMA_VALS);
    let dc_c = Huff::new(&DC_CHROMA_BITS, &DC_VALS);
    let ac_c = Huff::new(&AC_CHROMA_BITS, &AC_CHROMA_VALS);

    let mut w = BitWriter::default();
    let mut dc_pred = vec![0i32; comps.len()];
    for my in 0..mcuy {
        for mx in 0..mcux {
            for (c, &(h, v)) in comps.iter().enumerate() {
                let (stride, plane) = (&planes[c].0, &planes[c].1);
                let (dc, ac) = if c == 0 { (&dc_l, &ac_l) } else { (&dc_c, &ac_c) };
                for by in 0..v {
                    for bx in 0..h {
                        let x0 = (mx * h + bx) * 8;
                        let y0 = (my * v + by) * 8;
                        let coef = fdct(&block_at(plane, *stride, x0, y0));
                        let q = quantize(&coef, qtab(c));
                        let mut zz = [0i32; 64];
                        for (k, &nat) in ZIGZAG.iter().enumerate() {
                            zz[k] = q[nat];
                        }
                        encode_block(&mut w, &zz, &mut dc_pred[c], dc, ac);
                    }
                }
            }
        }
    }
    let scan = w.finish();

    let mut out = Vec::with_capacity(scan.len() + 1024);
    out.extend_from_slice(&[0xFF, 0xD8]); // SOI
    out.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x10]); // APP0 JFIF
    out.extend_from_slice(b"JFIF\0");
    out.extend_from_slice(&[1, 1, 0, 0, 1, 0, 1, 0, 0]);
    let tables: &[&[u16; 64]] = if color { &[&luma_q, &chroma_q] } else { &[&luma_q] };
    for (id, table) in tables.iter().enumerate() {
        out.extend_from_slice(&[0xFF, 0xDB, 0x00, 67, id as u8]); // DQT
        for &nat in ZIGZAG.iter() {
            out.push(table[nat] as u8);
        }
    }
    let nc = comps.len();
    let sof_len = 8 + 3 * nc as u16;
    out.extend_from_slice(&[0xFF, 0xC0]); // SOF0
    out.extend_from_slice(&sof_len.to_be_bytes());
    out.push(8);
    out.extend_from_slice(&(img.height as u16).to_be_bytes());
    out.extend_from_slice(&(img.width as u16).to_be_bytes());
    out.push(nc as u8);
    for (c, &(h, v)) in comps.iter().enumerate() {
        out.extend_from_slice(&[c as u8 + 1, ((h << 4) | v) as u8, if c == 0 { 0 } else { 1 }]);
    }
    // DHT: DC0, AC0, then DC1, AC1 for color.
    let mut dht = |class: u8, id: u8, bits: &[u8; 16], vals: &[u8]| {
        let len = 2 + 1 + 16 + vals.len() as u16;
        out.extend_from_slice(&[0xFF, 0xC4]);
        out.extend_from_slice(&len.to_be_bytes());
        out.push((class << 4) | id);
        out.extend_from_slice(bits);
        out.extend_from_slice(vals);
    };
    dht(0, 0, &DC_LUMA_BITS, &DC_VALS);
    dht(1, 0, &AC_LUMA_BITS, &AC_LUMA_VALS);
    if color {
        dht(0, 1, &DC_CHROMA_BITS, &DC_VALS);
        dht(1, 1, &AC_CHROMA_BITS, &AC_CHROMA_VALS);
    }
    let sos_len = 6 + 2 * nc as u16;
    out.extend_from_slice(&[0xFF, 0xDA]); // SOS
    out.extend_from_slice(&sos_len.to_be_bytes());
    out.push(nc as u8);
    for c in 0..nc {
        let t = if c == 0 { 0 } else { 1 };
        out.extend_from_slice(&[c as u8 + 1, (t << 4) | t]);
    }
    out.extend_from_slice(&[0, 63, 0]);
    out.extend_from_slice(&scan);
    out.extend_from_slice(&[0xFF, 0xD9]); // EOI
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;
    use ps5vm::{MemProtect, VirtualMemory};

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn all_four_are_implementations() {
        let reg = HleRegistry::with_defaults();
        for &(nid, name, _) in EXPORTS {
            assert!(reg.is_implemented(nid), "{name} is still the catalog stub");
        }
    }

    #[test]
    fn zigzag_is_a_permutation_and_tables_are_complete() {
        let mut seen = [false; 64];
        for &z in ZIGZAG.iter() {
            assert!(!seen[z]);
            seen[z] = true;
        }
        assert_eq!(DC_LUMA_BITS.iter().map(|&n| n as usize).sum::<usize>(), DC_VALS.len());
        assert_eq!(DC_CHROMA_BITS.iter().map(|&n| n as usize).sum::<usize>(), DC_VALS.len());
        assert_eq!(AC_LUMA_BITS.iter().map(|&n| n as usize).sum::<usize>(), AC_LUMA_VALS.len());
        assert_eq!(AC_CHROMA_BITS.iter().map(|&n| n as usize).sum::<usize>(), AC_CHROMA_VALS.len());
    }

    /// Annex K codes that the streams depend on: luma DC category 0 is `00`,
    /// category 4 is `101`; the luma AC end-of-block is `1010`; the chroma
    /// AC end-of-block is `00`.
    #[test]
    fn huffman_codes_match_annex_k() {
        let dc = Huff::new(&DC_LUMA_BITS, &DC_VALS);
        assert_eq!((dc.code[0], dc.len[0]), (0b00, 2));
        assert_eq!((dc.code[4], dc.len[4]), (0b101, 3));
        let ac = Huff::new(&AC_LUMA_BITS, &AC_LUMA_VALS);
        assert_eq!((ac.code[0x00], ac.len[0x00]), (0b1010, 4));
        assert_eq!((ac.code[0xF0], ac.len[0xF0]), (0b11111111001, 11));
        let acc = Huff::new(&AC_CHROMA_BITS, &AC_CHROMA_VALS);
        assert_eq!((acc.code[0x00], acc.len[0x00]), (0b00, 2));
    }

    #[test]
    fn dct_of_a_flat_block_is_its_dc_only() {
        let coef = fdct(&[16.0; 64]);
        assert!((coef[0] - 128.0).abs() < 1e-3, "DC was {}", coef[0]);
        assert!(coef[1..].iter().all(|c| c.abs() < 1e-3));
    }

    #[test]
    fn quality_scaling_follows_ijg() {
        assert_eq!(scaled_table(&LUMA_Q, 50), LUMA_Q, "quality 50 is the base table");
        assert!(scaled_table(&LUMA_Q, 100).iter().all(|&q| q == 1));
        assert_eq!(quality_of(64), 76);
        assert_eq!(quality_of(0), 100);
        assert_eq!(quality_of(255), 1);
    }

    #[test]
    fn magnitude_categories() {
        assert_eq!(magnitude(0), (0, 0));
        assert_eq!(magnitude(1), (1, 1));
        assert_eq!(magnitude(-1), (1, 0));
        assert_eq!(magnitude(8), (4, 8));
        assert_eq!(magnitude(-8), (4, 7));
    }

    #[test]
    fn bit_writer_stuffs_ff() {
        let mut w = BitWriter::default();
        w.put(0xFF, 8);
        w.put(0b101, 3);
        assert_eq!(w.finish(), vec![0xFF, 0x00, 0b1011_1111]);
    }

    /// The flat 128 block: DC difference 0 (`00`), end of block (`1010`),
    /// padded with 1 bits.
    #[test]
    fn flat_grey_block_stream() {
        let img = Image { width: 8, height: 8, planes: vec![vec![128u8; 64]] };
        let jpeg = encode(&img, 50, Sampling::Gray);
        assert_eq!(scan_of(&jpeg), &[0x2B]);
    }

    /// Flat 144 block: the DC coefficient is 128, quantized to 8 with the
    /// luminance step 16: category 4 (`101`) and the value `1000`, then
    /// end of block (`1010`).
    #[test]
    fn flat_bright_grey_block_stream() {
        let img = Image { width: 8, height: 8, planes: vec![vec![144u8; 64]] };
        let jpeg = encode(&img, 50, Sampling::Gray);
        assert_eq!(scan_of(&jpeg), &[0xB1, 0x5F]);
    }

    /// The scan data of a stream: after SOS, before EOI.
    fn scan_of(jpeg: &[u8]) -> &[u8] {
        let sos = jpeg.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
        let len = u16::from_be_bytes([jpeg[sos + 2], jpeg[sos + 3]]) as usize;
        &jpeg[sos + 2 + len..jpeg.len() - 2]
    }

    #[test]
    fn quantization_table_is_written_in_zigzag_order() {
        let img = Image { width: 8, height: 8, planes: vec![vec![128u8; 64]] };
        let jpeg = encode(&img, 50, Sampling::Gray);
        let dqt = jpeg.windows(2).position(|w| w == [0xFF, 0xDB]).unwrap();
        assert_eq!(&jpeg[dqt + 5..dqt + 11], &[16, 11, 12, 14, 12, 10]);
    }

    #[test]
    fn color_stream_headers_match_the_layout() {
        let img = Image { width: 20, height: 12, planes: vec![vec![100u8; 240], vec![128u8; 240], vec![128u8; 240]] };
        let jpeg = encode(&img, 76, Sampling::Yuv422);
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]);
        let sof = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        assert_eq!(u16::from_be_bytes([jpeg[sof + 5], jpeg[sof + 6]]), 12, "height");
        assert_eq!(u16::from_be_bytes([jpeg[sof + 7], jpeg[sof + 8]]), 20, "width");
        assert_eq!(jpeg[sof + 9], 3, "components");
        assert_eq!(jpeg[sof + 11], 0x21, "4:2:2 luma factors");
        let jpeg420 = encode(&img, 76, Sampling::Yuv420);
        let sof = jpeg420.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        assert_eq!(jpeg420[sof + 11], 0x22, "4:2:0 luma factors");
    }

    #[test]
    fn pixel_conversion_of_the_title_formats() {
        let mut p = EncodeParam {
            image: 0x1000,
            jpeg: 0x2000,
            image_size: 0,
            jpeg_size: 0,
            width: 2,
            height: 1,
            pitch: 8,
            pixel_format: PIXEL_R8G8B8A8,
            encode_mode: 0,
            color_space: COLOR_SPACE_YCC,
            sampling: SAMPLING_422,
            compression_ratio: 0,
            restart_interval: 0,
        };
        // Pure white and pure black pixels in RGBA order.
        let rgba = [255, 255, 255, 255, 0, 0, 0, 255];
        let img = image_from_pixels(&p, &rgba);
        assert_eq!(img.planes[0], vec![255, 0]);
        assert_eq!(img.planes[1], vec![128, 128]);
        // The same pixels in BGRA order describe the same colors.
        p.pixel_format = PIXEL_B8G8R8A8;
        let bgra = [255, 255, 255, 255, 0, 0, 0, 255];
        assert_eq!(image_from_pixels(&p, &bgra).planes[0], vec![255, 0]);
        // YUYV: Y0 U Y1 V.
        p.pixel_format = PIXEL_Y8U8Y8V8;
        p.pitch = 4;
        p.width = 2;
        let yuyv = [10, 20, 30, 40];
        let img = image_from_pixels(&p, &yuyv);
        assert_eq!(img.planes[0], vec![10, 30]);
        assert_eq!(img.planes[1], vec![20, 20]);
        assert_eq!(img.planes[2], vec![40, 40]);
    }

    #[test]
    fn validation_follows_the_title_rules() {
        let base = EncodeParam {
            image: 0x1000,
            jpeg: 0x2000,
            image_size: 64,
            jpeg_size: 0x100,
            width: 4,
            height: 4,
            pitch: 16,
            pixel_format: PIXEL_R8G8B8A8,
            encode_mode: 0,
            color_space: COLOR_SPACE_YCC,
            sampling: SAMPLING_420,
            compression_ratio: 64,
            restart_interval: 0,
        };
        assert_eq!(validate(&base), Ok(()));
        let mut p = base;
        p.sampling = SAMPLING_FULL;
        assert_eq!(validate(&p), Err(ERR_INVALID_PARAM), "RGBA with 4:4:4 is rejected");
        let mut p = base;
        p.image = 0;
        assert_eq!(validate(&p), Err(ERR_INVALID_ADDR));
        let mut p = base;
        p.image_size = 15;
        assert_eq!(validate(&p), Err(ERR_INVALID_PARAM), "image shorter than height * pitch");
        let mut p = base;
        p.jpeg_size = 0;
        assert_eq!(validate(&p), Err(ERR_INVALID_SIZE));
        let mut p = base;
        p.pixel_format = 99;
        assert_eq!(validate(&p), Err(ERR_INVALID_PARAM));
        let mut p = base;
        p.pitch = 15;
        assert_eq!(validate(&p), Err(ERR_INVALID_PARAM), "pitch is not a multiple of 4");
    }

    const BASE: u64 = 0x7D20_0000_0000;

    /// Encode parameter bytes in the title's layout.
    fn encode_param(p: &EncodeParam) -> Vec<u8> {
        let mut b = vec![0u8; ENCODE_PARAM_SIZE];
        b[0x00..0x08].copy_from_slice(&p.image.to_le_bytes());
        b[0x08..0x10].copy_from_slice(&p.jpeg.to_le_bytes());
        b[0x10..0x14].copy_from_slice(&p.image_size.to_le_bytes());
        b[0x14..0x18].copy_from_slice(&p.jpeg_size.to_le_bytes());
        b[0x18..0x1c].copy_from_slice(&p.width.to_le_bytes());
        b[0x1c..0x20].copy_from_slice(&p.height.to_le_bytes());
        b[0x20..0x24].copy_from_slice(&p.pitch.to_le_bytes());
        b[0x24..0x26].copy_from_slice(&p.pixel_format.to_le_bytes());
        b[0x26..0x28].copy_from_slice(&p.encode_mode.to_le_bytes());
        b[0x28..0x2a].copy_from_slice(&p.color_space.to_le_bytes());
        b[0x2a] = p.sampling;
        b[0x2b] = p.compression_ratio;
        b[0x2c..0x30].copy_from_slice(&p.restart_interval.to_le_bytes());
        b
    }

    #[test]
    fn query_and_create_report_their_errors() {
        let vm = VirtualMemory::new();
        vm.map(BASE, 0x10000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let param = BASE + 0x100;
        ctx.set(Rdi, param);
        assert_eq!(cb_query_memory_size(&mut ctx), ERR_INVALID_SIZE, "a zeroed parameter has size 0");
        assert!(ctx.write_u32(param, 8));
        assert!(ctx.write_u32(param + 4, 0));
        assert_eq!(cb_query_memory_size(&mut ctx), MEMORY_SIZE as HleResult);
        assert!(ctx.write_u32(param, 4));
        assert_eq!(cb_query_memory_size(&mut ctx), ERR_INVALID_SIZE);
        assert!(ctx.write_u32(param, 8));
        assert!(ctx.write_u32(param + 4, 1));
        assert_eq!(cb_query_memory_size(&mut ctx), ERR_INVALID_PARAM);

        assert!(ctx.write_u32(param + 4, 0));
        ctx.set(Rdi, param);
        ctx.set(Rsi, BASE + 0x1000);
        ctx.set(Rdx, 0x7FF);
        ctx.set(Rcx, BASE + 0x200);
        assert_eq!(cb_create(&mut ctx), ERR_INVALID_SIZE, "memory below 0x800");
        ctx.set(Rdx, 0x800);
        ctx.set(Rcx, 0);
        assert_eq!(cb_create(&mut ctx), ERR_INVALID_ADDR, "no handle pointer");
        ctx.set(Rcx, BASE + 0x200);
        assert_eq!(cb_create(&mut ctx), 0);
        let handle = ctx.read_u64(BASE + 0x200).unwrap();
        assert_eq!(handle, BASE + 0x1000, "handle is the memory address, 32-byte aligned");
        ctx.set(Rdi, handle);
        assert_eq!(cb_delete(&mut ctx), 0);
        assert_eq!(cb_delete(&mut ctx), ERR_INVALID_HANDLE);
        ctx.set(Rdi, 0);
        assert_eq!(cb_delete(&mut ctx), ERR_INVALID_HANDLE);
        let _ = vm.unmap(BASE, 0x10000);
    }

    /// A 16x16 RGBA gradient through Create, Encode and Delete in guest memory.
    #[test]
    fn encode_through_guest_memory() {
        let vm = VirtualMemory::new();
        vm.map(BASE, 0x10000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (param, handle_out, image, jpeg, info, enc) =
            (BASE + 0x100, BASE + 0x200, BASE + 0x4000, BASE + 0x8000, BASE + 0x400, BASE + 0x300);
        assert!(ctx.write_u32(param, 8));
        assert!(ctx.write_u32(param + 4, 0));
        ctx.set(Rdi, param);
        ctx.set(Rsi, BASE + 0x1000);
        ctx.set(Rdx, MEMORY_SIZE as u64);
        ctx.set(Rcx, handle_out);
        assert_eq!(cb_create(&mut ctx), 0);
        let handle = ctx.read_u64(handle_out).unwrap();

        let pixels: Vec<u8> = (0..16 * 16)
            .flat_map(|i| {
                let (x, y) = ((i % 16) as u8, (i / 16) as u8);
                [x * 16, y * 16, 128, 255]
            })
            .collect();
        assert!(ctx.write_bytes(image, &pixels));
        let p = EncodeParam {
            image,
            jpeg,
            image_size: pixels.len() as u32,
            jpeg_size: 0x2000,
            width: 16,
            height: 16,
            pitch: 64,
            pixel_format: PIXEL_R8G8B8A8,
            encode_mode: ENCODE_MODE_NORMAL,
            color_space: COLOR_SPACE_YCC,
            sampling: SAMPLING_422,
            compression_ratio: 64,
            restart_interval: 0,
        };
        assert!(ctx.write_bytes(enc, &encode_param(&p)));
        ctx.set(Rdi, handle);
        ctx.set(Rsi, enc);
        ctx.set(Rdx, info);
        let n = cb_encode(&mut ctx);
        assert!(n > 0, "encode failed with {n:#x}");
        let stream = ctx.read_bytes(jpeg, n as usize).unwrap();
        assert_eq!(&stream[..2], &[0xFF, 0xD8]);
        assert_eq!(&stream[stream.len() - 2..], &[0xFF, 0xD9]);
        assert_eq!(ctx.read_u32(info), Some(n as u32), "output size");
        assert_eq!(ctx.read_u32(info + 4), Some(16), "output height");

        // A stream that does not fit the title's buffer.
        let small = EncodeParam { jpeg_size: 8, ..p };
        assert!(ctx.write_bytes(enc, &encode_param(&small)));
        assert_eq!(cb_encode(&mut ctx), ERR_INVALID_SIZE);
        // The title's parameters with 4:4:4 for RGBA are rejected.
        let bad = EncodeParam { sampling: SAMPLING_FULL, ..p };
        assert!(ctx.write_bytes(enc, &encode_param(&bad)));
        assert_eq!(cb_encode(&mut ctx), ERR_INVALID_PARAM);
        // MJPEG is not implemented.
        let mjpeg = EncodeParam { encode_mode: ENCODE_MODE_MJPEG, ..p };
        assert!(ctx.write_bytes(enc, &encode_param(&mjpeg)));
        assert_eq!(cb_encode(&mut ctx), ERR_INVALID_PARAM);

        ctx.set(Rdi, handle);
        assert_eq!(cb_delete(&mut ctx), 0);
        ctx.set(Rsi, enc);
        assert_eq!(cb_encode(&mut ctx), ERR_INVALID_HANDLE, "encode after delete");
        let _ = vm.unmap(BASE, 0x10000);
    }
}
