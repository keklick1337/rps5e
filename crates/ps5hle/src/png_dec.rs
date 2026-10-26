// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libScePngDec — PNG decoding into RGBA/BGRA 8-bit images.
//!
//! The decoder handle lives in caller-provided work memory (0x20 bytes):
//! a self pointer (validity check), the create attribute and the maximum
//! image width. Decoding supports every PNG colour type and bit depth,
//! tRNS transparency and Adam7 interlacing; output is always 8 bits per
//! channel. 16-bit images need a decoder created with the 16-bit
//! attribute; their samples keep the high byte.
//!
//! The codec helpers are shared with libScePngEnc.

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libScePngDec";

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

const ATTRIBUTE_NONE: u32 = 0;
const ATTRIBUTE_BIT_DEPTH_16: u32 = 1;
const MAX_IMAGE_WIDTH: u32 = 1_000_000;
const MEMORY_SIZE: u32 = 0x20;
const HANDLE_ALIGNMENT: u64 = 8;

const PIXEL_FORMAT_R8G8B8A8: u16 = 0;
const PIXEL_FORMAT_B8G8R8A8: u16 = 1;

const COLOR_SPACE_GRAYSCALE: u16 = 2;
const COLOR_SPACE_RGB: u16 = 3;
const COLOR_SPACE_CLUT: u16 = 4;
const COLOR_SPACE_GRAYSCALE_ALPHA: u16 = 18;
const COLOR_SPACE_RGBA: u16 = 19;

const IMAGE_FLAG_ADAM7_INTERLACE: u32 = 1;
const IMAGE_FLAG_TRNS_CHUNK_EXIST: u32 = 2;

const MAX_PACKED_DIMENSION: u32 = 32767;
/// Largest PNG stream accepted from the guest.
const MAX_PNG_BYTES: u32 = 256 << 20;

// ---------------------------------------------------------------------------
// PNG codec
// ---------------------------------------------------------------------------

pub(crate) mod codec {
    pub const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    const CHUNK_OVERHEAD: usize = 12;
    const IHDR_SIZE: usize = 13;
    const MAX_DIMENSION: u32 = 0x7FFF_FFFF;
    /// Refuse to inflate images whose RGBA output would exceed this.
    const MAX_DECODED_BYTES: u64 = 1 << 31;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ColorType {
        Grayscale = 0,
        Rgb = 2,
        Palette = 3,
        GrayscaleAlpha = 4,
        Rgba = 6,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct Header {
        pub width: u32,
        pub height: u32,
        pub bit_depth: u8,
        pub color_type: ColorType,
        pub interlaced: bool,
        pub has_transparency: bool,
    }

    pub struct Image {
        pub width: u32,
        pub height: u32,
        /// RGBA8, tightly packed.
        pub pixels: Vec<u8>,
    }

    fn be32(b: &[u8]) -> u32 {
        u32::from_be_bytes([b[0], b[1], b[2], b[3]])
    }

    fn color_type(color_type: u8) -> Option<ColorType> {
        Some(match color_type {
            0 => ColorType::Grayscale,
            2 => ColorType::Rgb,
            3 => ColorType::Palette,
            4 => ColorType::GrayscaleAlpha,
            6 => ColorType::Rgba,
            _ => return None,
        })
    }

    fn valid_bit_depth(ct: ColorType, bit_depth: u8) -> bool {
        match ct {
            ColorType::Grayscale => matches!(bit_depth, 1 | 2 | 4 | 8 | 16),
            ColorType::Palette => matches!(bit_depth, 1 | 2 | 4 | 8),
            ColorType::Rgb | ColorType::GrayscaleAlpha | ColorType::Rgba => matches!(bit_depth, 8 | 16),
        }
    }

    /// Parse the IHDR chunk and walk the chunks before the image data for
    /// tRNS. Only the signature, IHDR, colour type and non-zero dimensions
    /// are checked here; the remaining IHDR fields are validated when
    /// decoding. The walk stops at the first chunk that does not fit in the
    /// input, so only a complete tRNS chunk reports transparency.
    pub fn parse_header(png: &[u8]) -> Option<Header> {
        let first = SIGNATURE.len();
        if png.len() < first + CHUNK_OVERHEAD + IHDR_SIZE || png[..first] != SIGNATURE {
            return None;
        }
        if be32(&png[first..]) as usize != IHDR_SIZE || &png[first + 4..first + 8] != b"IHDR" {
            return None;
        }
        let ihdr = &png[first + 8..first + 8 + IHDR_SIZE];
        let width = be32(ihdr);
        let height = be32(&ihdr[4..]);
        let color_type = color_type(ihdr[9])?;
        if width == 0 || height == 0 {
            return None;
        }
        let mut header = Header {
            width,
            height,
            bit_depth: ihdr[8],
            color_type,
            interlaced: ihdr[12] == 1,
            has_transparency: false,
        };
        let mut off = first + CHUNK_OVERHEAD + IHDR_SIZE;
        while png.len() - off >= CHUNK_OVERHEAD {
            let len = be32(&png[off..]) as usize;
            if len > png.len() - off - CHUNK_OVERHEAD {
                break;
            }
            let ty = &png[off + 4..off + 8];
            if ty == b"tRNS" {
                header.has_transparency = true;
            }
            if ty == b"tRNS" || ty == b"IDAT" || ty == b"IEND" {
                break;
            }
            off += CHUNK_OVERHEAD + len;
        }
        Some(header)
    }

    fn channels(ct: ColorType) -> usize {
        match ct {
            ColorType::Grayscale | ColorType::Palette => 1,
            ColorType::GrayscaleAlpha => 2,
            ColorType::Rgb => 3,
            ColorType::Rgba => 4,
        }
    }

    fn paeth(a: u8, b: u8, c: u8) -> u8 {
        let (ia, ib, ic) = (a as i16, b as i16, c as i16);
        let p = ia + ib - ic;
        let (pa, pb, pc) = ((p - ia).abs(), (p - ib).abs(), (p - ic).abs());
        if pa <= pb && pa <= pc {
            a
        } else if pb <= pc {
            b
        } else {
            c
        }
    }

    /// Undo the per-scanline filters of one (sub)image in place.
    fn unfilter(data: &mut [u8], rows: usize, row_bytes: usize, bpp: usize) -> Option<()> {
        let stride = row_bytes + 1;
        for y in 0..rows {
            let (prev_part, cur_part) = data.split_at_mut(y * stride);
            let prev: Option<&[u8]> = if y == 0 { None } else { Some(&prev_part[(y - 1) * stride + 1..y * stride]) };
            let filter = cur_part[0];
            let cur = &mut cur_part[1..stride];
            match filter {
                0 => {}
                1 => {
                    for i in bpp..row_bytes {
                        cur[i] = cur[i].wrapping_add(cur[i - bpp]);
                    }
                }
                2 => {
                    if let Some(p) = prev {
                        for i in 0..row_bytes {
                            cur[i] = cur[i].wrapping_add(p[i]);
                        }
                    }
                }
                3 => {
                    for i in 0..row_bytes {
                        let left = if i >= bpp { cur[i - bpp] as u16 } else { 0 };
                        let up = prev.map_or(0, |p| p[i] as u16);
                        cur[i] = cur[i].wrapping_add(((left + up) >> 1) as u8);
                    }
                }
                4 => {
                    for i in 0..row_bytes {
                        let left = if i >= bpp { cur[i - bpp] } else { 0 };
                        let up = prev.map_or(0, |p| p[i]);
                        let ul = if i >= bpp { prev.map_or(0, |p| p[i - bpp]) } else { 0 };
                        cur[i] = cur[i].wrapping_add(paeth(left, up, ul));
                    }
                }
                _ => return None,
            }
        }
        Some(())
    }

    struct Palette {
        rgb: Vec<[u8; 3]>,
        alpha: Vec<u8>,
    }

    /// Read sample `index` of a row with the given bit depth (raw value).
    fn sample(row: &[u8], index: usize, depth: u8) -> u16 {
        match depth {
            16 => u16::from_be_bytes([row[index * 2], row[index * 2 + 1]]),
            8 => row[index] as u16,
            d => {
                let per_byte = 8 / d as usize;
                let byte = row[index / per_byte];
                let shift = 8 - d as usize * (index % per_byte + 1);
                ((byte >> shift) & ((1u8 << d) - 1)) as u16
            }
        }
    }

    fn to8(v: u16, depth: u8) -> u8 {
        match depth {
            16 => (v >> 8) as u8,
            8 => v as u8,
            4 => (v * 0x11) as u8,
            2 => (v * 0x55) as u8,
            1 => (v * 0xFF) as u8,
            _ => 0,
        }
    }

    /// Expand one unfiltered scanline into RGBA8 pixels.
    #[allow(clippy::too_many_arguments)]
    fn expand_row(
        h: &Header,
        row: &[u8],
        width: usize,
        pal: &Palette,
        trns_key: Option<[u16; 3]>,
        out: &mut [u8],
        out_x0: usize,
        out_dx: usize,
    ) -> Option<()> {
        let d = h.bit_depth;
        let ch = channels(h.color_type);
        for x in 0..width {
            let px = &mut out[(out_x0 + x * out_dx) * 4..(out_x0 + x * out_dx) * 4 + 4];
            match h.color_type {
                ColorType::Grayscale => {
                    let g = sample(row, x, d);
                    let v = to8(g, d);
                    let a = if trns_key.is_some_and(|k| k[0] == g) { 0 } else { 255 };
                    px.copy_from_slice(&[v, v, v, a]);
                }
                ColorType::GrayscaleAlpha => {
                    let v = to8(sample(row, x * 2, d), d);
                    let a = to8(sample(row, x * 2 + 1, d), d);
                    px.copy_from_slice(&[v, v, v, a]);
                }
                ColorType::Rgb => {
                    let r = sample(row, x * 3, d);
                    let g = sample(row, x * 3 + 1, d);
                    let b = sample(row, x * 3 + 2, d);
                    let a = if trns_key == Some([r, g, b]) { 0 } else { 255 };
                    px.copy_from_slice(&[to8(r, d), to8(g, d), to8(b, d), a]);
                }
                ColorType::Rgba => {
                    for c in 0..ch {
                        px[c] = to8(sample(row, x * 4 + c, d), d);
                    }
                }
                ColorType::Palette => {
                    let i = sample(row, x, d) as usize;
                    let rgb = pal.rgb.get(i)?;
                    let a = pal.alpha.get(i).copied().unwrap_or(255);
                    px.copy_from_slice(&[rgb[0], rgb[1], rgb[2], a]);
                }
            }
        }
        Some(())
    }

    /// Decode a complete PNG stream into RGBA8.
    pub fn decode(png: &[u8]) -> Option<Image> {
        let h = parse_header(png)?;
        let ihdr = &png[SIGNATURE.len() + 8..];
        if !valid_bit_depth(h.color_type, h.bit_depth)
            || ihdr[10] != 0
            || ihdr[11] != 0
            || ihdr[12] > 1
            || h.width > MAX_DIMENSION
            || h.height > MAX_DIMENSION
        {
            return None;
        }
        let mut idat = Vec::new();
        let mut pal = Palette { rgb: Vec::new(), alpha: Vec::new() };
        let mut trns_raw: Option<Vec<u8>> = None;
        let mut off = SIGNATURE.len();
        while png.len().saturating_sub(off) >= CHUNK_OVERHEAD {
            let len = be32(&png[off..]) as usize;
            if len > png.len() - off - CHUNK_OVERHEAD {
                return None;
            }
            let ty = &png[off + 4..off + 8];
            let body = &png[off + 8..off + 8 + len];
            match ty {
                b"PLTE" => {
                    if len % 3 != 0 || len / 3 > 256 {
                        return None;
                    }
                    pal.rgb = body.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
                }
                b"tRNS" => trns_raw = Some(body.to_vec()),
                b"IDAT" => idat.extend_from_slice(body),
                b"IEND" => break,
                _ => {}
            }
            off += CHUNK_OVERHEAD + len;
        }
        if idat.is_empty() {
            return None;
        }
        if h.color_type == ColorType::Palette && pal.rgb.is_empty() {
            return None;
        }
        let mut trns_key = None;
        if let Some(t) = trns_raw {
            match h.color_type {
                ColorType::Palette => pal.alpha = t,
                ColorType::Grayscale if t.len() >= 2 => trns_key = Some([be16(&t), 0, 0]),
                ColorType::Rgb if t.len() >= 6 => trns_key = Some([be16(&t), be16(&t[2..]), be16(&t[4..])]),
                _ => {}
            }
        }

        let (w, ht) = (h.width as usize, h.height as usize);
        if (w as u64) * (ht as u64) * 4 > MAX_DECODED_BYTES {
            return None;
        }
        let raw = miniz_oxide::inflate::decompress_to_vec_zlib(&idat).ok()?;
        let ch = channels(h.color_type);
        let bits_pp = ch * h.bit_depth as usize;
        let bpp = bits_pp.div_ceil(8).max(1);
        let row_bytes = |pw: usize| (pw * bits_pp).div_ceil(8);
        let mut pixels = vec![0u8; w * ht * 4];

        let passes: &[(usize, usize, usize, usize)] = if h.interlaced {
            &[(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
        } else {
            &[(0, 0, 1, 1)]
        };
        let mut pos = 0usize;
        for &(x0, y0, dx, dy) in passes {
            if x0 >= w || y0 >= ht {
                continue;
            }
            let pw = (w - x0).div_ceil(dx);
            let ph = (ht - y0).div_ceil(dy);
            let rb = row_bytes(pw);
            let size = (rb + 1) * ph;
            if raw.len() < pos + size {
                return None;
            }
            let mut sub = raw[pos..pos + size].to_vec();
            pos += size;
            unfilter(&mut sub, ph, rb, bpp)?;
            for py in 0..ph {
                let row = &sub[py * (rb + 1) + 1..(py + 1) * (rb + 1)];
                let y = y0 + py * dy;
                let line = &mut pixels[y * w * 4..(y + 1) * w * 4];
                expand_row(&h, row, pw, &pal, trns_key, line, x0, dx)?;
            }
        }
        Some(Image { width: h.width, height: h.height, pixels })
    }

    fn be16(b: &[u8]) -> u16 {
        u16::from_be_bytes([b[0], b[1]])
    }

    // -- encoding ----------------------------------------------------------

    pub const FILTER_NONE: u8 = 1 << 0;
    pub const FILTER_SUB: u8 = 1 << 1;
    pub const FILTER_UP: u8 = 1 << 2;
    pub const FILTER_AVERAGE: u8 = 1 << 3;
    pub const FILTER_PAETH: u8 = 1 << 4;
    pub const FILTER_ALL: u8 = FILTER_NONE | FILTER_SUB | FILTER_UP | FILTER_AVERAGE | FILTER_PAETH;

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = (crc >> 1) ^ if crc & 1 != 0 { 0xEDB8_8320 } else { 0 };
            }
        }
        !crc
    }

    fn append_chunk(out: &mut Vec<u8>, ty: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let start = out.len();
        out.extend_from_slice(ty);
        out.extend_from_slice(data);
        let crc = crc32(&out[start..]);
        out.extend_from_slice(&crc.to_be_bytes());
    }

    /// Filter one row with `filter` into `dst`, returning its cost (sum of
    /// absolute signed residuals).
    fn filter_row(row: &[u8], prev: Option<&[u8]>, bpp: usize, filter: u8, dst: &mut [u8]) -> u64 {
        let mut cost = 0u64;
        for i in 0..row.len() {
            let left = if i >= bpp { row[i - bpp] } else { 0 };
            let up = prev.map_or(0, |p| p[i]);
            let ul = if i >= bpp { prev.map_or(0, |p| p[i - bpp]) } else { 0 };
            let pred = match filter {
                1 => left,
                2 => up,
                3 => ((left as u16 + up as u16) >> 1) as u8,
                4 => paeth(left, up, ul),
                _ => 0,
            };
            let v = row[i].wrapping_sub(pred);
            dst[i] = v;
            cost += if v < 0x80 { v as u64 } else { 0x100 - v as u64 };
        }
        cost
    }

    /// Encode 8-bit pixels with 1..=4 channels. `filters` is a mask of the
    /// FILTER_* bits; each row uses the allowed filter with the lowest cost.
    pub fn encode(pixels: &[u8], width: u32, height: u32, channels: u32, level: u8, filters: u8) -> Option<Vec<u8>> {
        if !(1..=4).contains(&channels) || level > 9 || filters == 0 || filters & !FILTER_ALL != 0 {
            return None;
        }
        let row = width as usize * channels as usize;
        if width == 0 || height == 0 || pixels.len() < row * height as usize {
            return None;
        }
        let mut filtered = vec![0u8; (row + 1) * height as usize];
        let mut cand = vec![0u8; row];
        for y in 0..height as usize {
            let cur = &pixels[row * y..row * (y + 1)];
            let prev = if y == 0 { None } else { Some(&pixels[row * (y - 1)..row * y]) };
            let dst = &mut filtered[(row + 1) * y..(row + 1) * (y + 1)];
            let mut best = u64::MAX;
            for f in 0..5u8 {
                if filters & (1 << f) == 0 {
                    continue;
                }
                let cost = filter_row(cur, prev, channels as usize, f, &mut cand);
                if cost >= best {
                    continue;
                }
                best = cost;
                dst[0] = f;
                dst[1..].copy_from_slice(&cand);
            }
        }
        let compressed = miniz_oxide::deflate::compress_to_vec_zlib(&filtered, level);
        let color_type = [0u8, 0, 4, 2, 6][channels as usize];
        let mut ihdr = Vec::with_capacity(IHDR_SIZE);
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.extend_from_slice(&[8, color_type, 0, 0, 0]);
        let mut out = SIGNATURE.to_vec();
        append_chunk(&mut out, b"IHDR", &ihdr);
        append_chunk(&mut out, b"IDAT", &compressed);
        append_chunk(&mut out, b"IEND", &[]);
        Some(out)
    }
}

use codec::{ColorType, Header};

// ---------------------------------------------------------------------------
// Guest structures
// ---------------------------------------------------------------------------

/// ScePngDecCreateParam { thisSize u32, attribute u32, maxImageWidth u32 }
struct CreateParam {
    attribute: u32,
    max_image_width: u32,
}

fn read_create_param(ctx: &CpuContext, addr: u64) -> Option<CreateParam> {
    Some(CreateParam {
        attribute: ctx.read_u32(addr + 4)?,
        max_image_width: ctx.read_u32(addr + 8)?,
    })
}

fn validate_create_param(ctx: &CpuContext, addr: u64) -> Result<CreateParam, HleResult> {
    if addr == 0 {
        return Err(ERROR_INVALID_PARAM);
    }
    let p = read_create_param(ctx, addr).ok_or(ERROR_INVALID_ADDR)?;
    if p.attribute != ATTRIBUTE_NONE && p.attribute != ATTRIBUTE_BIT_DEPTH_16 {
        return Err(ERROR_INVALID_PARAM);
    }
    // Widths 1 to 1000000 (measured on a PS5).
    if p.max_image_width.wrapping_sub(1) >= MAX_IMAGE_WIDTH {
        return Err(ERROR_INVALID_PARAM);
    }
    Ok(p)
}

/// Context stored in the caller's work memory (8-byte aligned).
const CTX_SELF: u64 = 0;
const CTX_ATTRIBUTE: u64 = 8;
const CTX_MAX_WIDTH: u64 = 12;

fn context(ctx: &CpuContext, handle: u64) -> Option<(u32, u32)> {
    if handle == 0 || handle % HANDLE_ALIGNMENT != 0 {
        return None;
    }
    if ctx.read_u64(handle + CTX_SELF)? != handle {
        return None;
    }
    Some((ctx.read_u32(handle + CTX_ATTRIBUTE)?, ctx.read_u32(handle + CTX_MAX_WIDTH)?))
}

fn color_space(ct: ColorType) -> u16 {
    match ct {
        ColorType::Grayscale => COLOR_SPACE_GRAYSCALE,
        ColorType::Rgb => COLOR_SPACE_RGB,
        ColorType::Palette => COLOR_SPACE_CLUT,
        ColorType::GrayscaleAlpha => COLOR_SPACE_GRAYSCALE_ALPHA,
        ColorType::Rgba => COLOR_SPACE_RGBA,
    }
}

/// ScePngDecImageInfo { imageWidth u32, imageHeight u32, colorSpace u16,
/// bitDepth u16, imageFlag u32 }
fn write_image_info(ctx: &CpuContext, addr: u64, h: &Header) -> bool {
    let mut b = [0u8; 16];
    b[0..4].copy_from_slice(&h.width.to_le_bytes());
    b[4..8].copy_from_slice(&h.height.to_le_bytes());
    b[8..10].copy_from_slice(&color_space(h.color_type).to_le_bytes());
    b[10..12].copy_from_slice(&(h.bit_depth as u16).to_le_bytes());
    let flags = if h.interlaced { IMAGE_FLAG_ADAM7_INTERLACE } else { 0 }
        | if h.has_transparency { IMAGE_FLAG_TRNS_CHUNK_EXIST } else { 0 };
    b[12..16].copy_from_slice(&flags.to_le_bytes());
    ctx.write_bytes(addr, &b)
}

fn has_alpha(h: &Header) -> bool {
    h.has_transparency || matches!(h.color_type, ColorType::GrayscaleAlpha | ColorType::Rgba)
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

/// scePngDecQueryMemorySize(const ScePngDecCreateParam*)
fn cb_query_memory_size(ctx: &mut CpuContext) -> HleResult {
    match validate_create_param(ctx, ctx.rdi()) {
        Ok(_) => MEMORY_SIZE as HleResult,
        Err(e) => e,
    }
}

/// scePngDecCreate(param, memoryAddress, memorySize, ScePngDecHandle*)
fn cb_create(ctx: &mut CpuContext) -> HleResult {
    let (mem, mem_size, out) = (ctx.rsi(), ctx.rdx() as u32, ctx.rcx());
    if out == 0 {
        return ERROR_INVALID_PARAM;
    }
    let p = match validate_create_param(ctx, ctx.rdi()) {
        Ok(p) => p,
        Err(e) => return e,
    };
    if mem == 0 {
        return ERROR_INVALID_ADDR;
    }
    if mem_size < MEMORY_SIZE {
        return ERROR_INVALID_WORK_MEMORY;
    }
    let handle = (mem + HANDLE_ALIGNMENT - 1) & !(HANDLE_ALIGNMENT - 1);
    let mut b = [0u8; 16];
    b[0..8].copy_from_slice(&handle.to_le_bytes());
    b[8..12].copy_from_slice(&p.attribute.to_le_bytes());
    b[12..16].copy_from_slice(&p.max_image_width.to_le_bytes());
    if !ctx.write_bytes(handle, &b) || !ctx.write_u64(out, handle) {
        return ERROR_INVALID_ADDR;
    }
    0
}

/// scePngDecDelete(handle)
fn cb_delete(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    if context(ctx, handle).is_none() {
        return ERROR_INVALID_HANDLE;
    }
    ctx.write_u64(handle + CTX_SELF, 0);
    0
}

fn read_png(ctx: &CpuContext, addr: u64, size: u32) -> Option<Vec<u8>> {
    if size > MAX_PNG_BYTES {
        return None;
    }
    ctx.read_bytes(addr, size as usize)
}

/// scePngDecParseHeader(const ScePngDecParseParam*, ScePngDecImageInfo*)
/// ParseParam { pngMemAddr ptr, pngMemSize u32, reserved u32 }
fn cb_parse_header(ctx: &mut CpuContext) -> HleResult {
    let (param, info) = (ctx.rdi(), ctx.rsi());
    if param == 0 || info == 0 {
        return ERROR_INVALID_PARAM;
    }
    let (Some(png_addr), Some(png_size), Some(reserved)) =
        (ctx.read_u64(param), ctx.read_u32(param + 8), ctx.read_u32(param + 12))
    else {
        return ERROR_INVALID_ADDR;
    };
    if png_addr == 0 {
        return ERROR_INVALID_ADDR;
    }
    if reserved != 0 {
        return ERROR_INVALID_PARAM;
    }
    let Some(png) = read_png(ctx, png_addr, png_size) else {
        return ERROR_INVALID_ADDR;
    };
    let Some(h) = codec::parse_header(&png) else {
        return ERROR_INVALID_DATA;
    };
    if !write_image_info(ctx, info, &h) {
        return ERROR_INVALID_ADDR;
    }
    0
}

/// scePngDecDecode(handle, const ScePngDecDecodeParam*, ScePngDecImageInfo*)
/// DecodeParam { pngMemAddr ptr, imageMemAddr ptr, pngMemSize u32,
/// imageMemSize u32, pixelFormat u16, alphaValue u16, imagePitch u32 }
///
/// Returns (width << 16 | height) on success when both fit in 15 bits.
fn cb_decode(ctx: &mut CpuContext) -> HleResult {
    let (handle, param, info) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let Some((attribute, max_width)) = context(ctx, handle) else {
        return ERROR_INVALID_HANDLE;
    };
    if param == 0 {
        return ERROR_INVALID_PARAM;
    }
    let Some(raw) = ctx.read_bytes(param, 0x20) else {
        return ERROR_INVALID_ADDR;
    };
    let u64_at = |o: usize| u64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
    let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
    let u16_at = |o: usize| u16::from_le_bytes(raw[o..o + 2].try_into().unwrap());
    let (png_addr, image_addr) = (u64_at(0), u64_at(8));
    let (png_size, image_size) = (u32_at(16), u32_at(20));
    let (pixel_format, alpha_value, pitch) = (u16_at(24), u16_at(26), u32_at(28));
    // The console's order: addresses (the image 4-byte aligned), sizes,
    // pixel format, alpha value, the header, the pitch, the context's width.
    if png_addr == 0 || image_addr == 0 || image_addr & 3 != 0 {
        return ERROR_INVALID_ADDR;
    }
    if png_size == 0 || image_size == 0 {
        return ERROR_INVALID_SIZE;
    }
    if pixel_format != PIXEL_FORMAT_R8G8B8A8 && pixel_format != PIXEL_FORMAT_B8G8R8A8 {
        return ERROR_INVALID_PARAM;
    }
    if alpha_value > 0xFF {
        return ERROR_INVALID_PARAM;
    }
    let Some(png) = read_png(ctx, png_addr, png_size) else {
        return ERROR_INVALID_ADDR;
    };
    let Some(h) = codec::parse_header(&png) else {
        return ERROR_INVALID_DATA;
    };
    // A size that does not fit the packed result can only be reported
    // through the image info, so a call without one is refused.
    let packed = h.width <= MAX_PACKED_DIMENSION && h.height <= MAX_PACKED_DIMENSION;
    if info == 0 && !packed {
        return ERROR_INVALID_ADDR;
    }
    if info != 0 && !write_image_info(ctx, info, &h) {
        return ERROR_INVALID_ADDR;
    }
    let row_size = h.width as u64 * 4;
    let pitch = if pitch == 0 {
        if h.height as u64 * row_size > image_size as u64 {
            return ERROR_INVALID_SIZE;
        }
        row_size
    } else {
        let pitch = pitch as u64;
        if pitch < row_size || pitch & 3 != 0 {
            return ERROR_INVALID_PARAM;
        }
        let total = h.height as u64 * pitch;
        if total >= 0x8000_0000 || total > image_size as u64 {
            return ERROR_INVALID_PARAM;
        }
        pitch
    };
    let align8 = |w: u64| (w + 7) & !7;
    if align8(h.width as u64) > align8(max_width as u64) {
        return ERROR_INVALID_WORK_MEMORY;
    }
    // The work memory holds 8-bit samples unless the decoder was created
    // with the 16-bit attribute.
    if h.bit_depth as u64 > 8u64 << (attribute & 31) {
        return ERROR_INVALID_WORK_MEMORY;
    }
    let image = match codec::decode(&png) {
        Some(i) if i.width == h.width && i.height == h.height => i,
        _ => return ERROR_DECODE_ERROR,
    };
    let swap = pixel_format == PIXEL_FORMAT_B8G8R8A8;
    let fill_alpha = !has_alpha(&h);
    let alpha = alpha_value.min(255) as u8;
    let mut line = vec![0u8; row_size as usize];
    for y in 0..h.height as u64 {
        let src = &image.pixels[(y * row_size) as usize..((y + 1) * row_size) as usize];
        line.copy_from_slice(src);
        for px in line.chunks_exact_mut(4) {
            if swap {
                px.swap(0, 2);
            }
            if fill_alpha {
                px[3] = alpha;
            }
        }
        if !ctx.write_bytes(image_addr + y * pitch, &line) {
            return ERROR_INVALID_ADDR;
        }
    }
    if !packed {
        return 0;
    }
    ((h.width << 16) | h.height) as HleResult
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("m0uW+8pFyaw", "scePngDecCreate", cb_create),
    ("WC216DD3El4", "scePngDecDecode", cb_decode),
    ("QbD+eENEwo8", "scePngDecDelete", cb_delete),
    ("U6h4e5JRPaQ", "scePngDecParseHeader", cb_parse_header),
    ("-6srIGbLTIU", "scePngDecQueryMemorySize", cb_query_memory_size),
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

    /// Build a PNG by hand: 2x2 palette image with tRNS, bit depth 2.
    fn palette_png() -> Vec<u8> {
        // Rows: [idx0 idx1] [idx2 idx0], 2 bits each, filter byte 0.
        let raw = [0u8, 0b0001_0000, 0, 0b1000_0000];
        let z = miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6);
        let mut out = codec::SIGNATURE.to_vec();
        let chunk = |out: &mut Vec<u8>, ty: &[u8; 4], d: &[u8]| {
            out.extend_from_slice(&(d.len() as u32).to_be_bytes());
            out.extend_from_slice(ty);
            out.extend_from_slice(d);
            out.extend_from_slice(&[0, 0, 0, 0]); // CRC is not checked
        };
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&2u32.to_be_bytes());
        ihdr.extend_from_slice(&2u32.to_be_bytes());
        ihdr.extend_from_slice(&[2, 3, 0, 0, 0]);
        chunk(&mut out, b"IHDR", &ihdr);
        chunk(&mut out, b"PLTE", &[255, 0, 0, 0, 255, 0, 0, 0, 255]);
        chunk(&mut out, b"tRNS", &[128]);
        chunk(&mut out, b"IDAT", &z);
        chunk(&mut out, b"IEND", &[]);
        out
    }

    #[test]
    fn codec_palette_trns() {
        let png = palette_png();
        let h = codec::parse_header(&png).unwrap();
        assert!(h.has_transparency);
        assert_eq!(h.color_type, ColorType::Palette);
        let img = codec::decode(&png).unwrap();
        assert_eq!(
            img.pixels,
            vec![255, 0, 0, 128, 0, 255, 0, 255, 0, 0, 255, 255, 255, 0, 0, 128]
        );
    }

    #[test]
    fn codec_round_trip_all_filters() {
        let (w, h) = (13u32, 7u32);
        let mut px = Vec::new();
        for y in 0..h {
            for x in 0..w {
                px.extend_from_slice(&[(x * 19) as u8, (y * 31) as u8, (x * y) as u8, 200 - x as u8]);
            }
        }
        for filters in [codec::FILTER_NONE, codec::FILTER_SUB, codec::FILTER_PAETH, codec::FILTER_ALL] {
            let png = codec::encode(&px, w, h, 4, 6, filters).unwrap();
            let img = codec::decode(&png).unwrap();
            assert_eq!((img.width, img.height), (w, h));
            assert_eq!(img.pixels, px);
        }
    }

    #[test]
    fn codec_interlaced_gray16() {
        // 3x3 grayscale 16-bit, Adam7; build the passes by hand.
        let val = |x: u32, y: u32| ((x * 3 + y) * 0x1111) as u16;
        let passes = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)];
        let mut raw = Vec::new();
        for (x0, y0, dx, dy) in passes {
            if x0 >= 3 || y0 >= 3 {
                continue;
            }
            let mut y = y0;
            while y < 3 {
                raw.push(0);
                let mut x = x0;
                while x < 3 {
                    raw.extend_from_slice(&val(x, y).to_be_bytes());
                    x += dx;
                }
                y += dy;
            }
        }
        let z = miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6);
        let mut png = codec::SIGNATURE.to_vec();
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&3u32.to_be_bytes());
        ihdr.extend_from_slice(&3u32.to_be_bytes());
        ihdr.extend_from_slice(&[16, 0, 0, 0, 1]);
        for (ty, d) in [(b"IHDR", ihdr.as_slice()), (b"IDAT", z.as_slice()), (b"IEND", &[][..])] {
            png.extend_from_slice(&(d.len() as u32).to_be_bytes());
            png.extend_from_slice(ty);
            png.extend_from_slice(d);
            png.extend_from_slice(&[0; 4]);
        }
        let img = codec::decode(&png).unwrap();
        for y in 0..3 {
            for x in 0..3 {
                let v = (val(x, y) >> 8) as u8;
                let i = ((y * 3 + x) * 4) as usize;
                assert_eq!(&img.pixels[i..i + 4], &[v, v, v, 255], "pixel {x},{y}");
            }
        }
    }

    /// A width or height above 32767 does not fit the packed result: the
    /// image decodes (returning 0) only with an image info to report it,
    /// and without one the call is INVALID_ADDR before anything is written.
    #[test]
    fn large_dimensions_need_the_image_info() {
        const BASE: u64 = 0x6D10_0000_0000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x80_0000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        for (w, h) in [(32767u32, 1u32), (32768, 1), (1, 32768)] {
            let mut create = vec![12, 0, 0, 0, 0, 0, 0, 0];
            create.extend_from_slice(&w.to_le_bytes());
            ctx.write_bytes(BASE, &create);
            ctx.set(Rdi, BASE);
            let size = cb_query_memory_size(&mut ctx) as u64;
            assert!(size > 0 && size < 0x70_0000, "{size:#x}");
            ctx.set(Rsi, BASE + 0x10_0000);
            ctx.set(Rdx, size);
            ctx.set(Rcx, BASE + 0x200);
            assert_eq!(cb_create(&mut ctx), 0);
            let handle = ctx.read_u64(BASE + 0x200).unwrap();
            let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 13 + 7) as u8).collect();
            let png = codec::encode(&rgb, w, h, 3, 6, codec::FILTER_NONE).unwrap();
            let (png_at, image_at) = (BASE + 0x10000, BASE + 0x40000);
            ctx.write_bytes(png_at, &png);
            let image_size = w * h * 4;
            let mut p = Vec::new();
            p.extend_from_slice(&png_at.to_le_bytes());
            p.extend_from_slice(&image_at.to_le_bytes());
            p.extend_from_slice(&(png.len() as u32).to_le_bytes());
            p.extend_from_slice(&image_size.to_le_bytes());
            p.extend_from_slice(&PIXEL_FORMAT_R8G8B8A8.to_le_bytes());
            p.extend_from_slice(&0x77u16.to_le_bytes());
            p.extend_from_slice(&0u32.to_le_bytes());
            ctx.write_bytes(BASE + 0x300, &p);
            ctx.write_bytes(image_at, &vec![0xCC; image_size as usize]);
            ctx.set(Rdi, handle);
            ctx.set(Rsi, BASE + 0x300);
            ctx.set(Rdx, 0);
            let packed = w <= 32767 && h <= 32767;
            let expected = if packed { ((w << 16) | h) as HleResult } else { ERROR_INVALID_ADDR };
            assert_eq!(cb_decode(&mut ctx), expected, "{w}x{h} without info");
            if !packed {
                assert!(ctx.read_bytes(image_at, image_size as usize).unwrap().iter().all(|&b| b == 0xCC));
            }
            ctx.set(Rdx, BASE + 0x400);
            assert_eq!(cb_decode(&mut ctx), if packed { expected } else { 0 }, "{w}x{h} with info");
            assert_eq!(ctx.read_u32(BASE + 0x400), Some(w));
            assert_eq!(ctx.read_u32(BASE + 0x404), Some(h));
            assert_eq!(ctx.read_bytes(image_at, 4).unwrap(), vec![7, 20, 33, 0x77]);
            ctx.set(Rdi, handle);
            assert_eq!(cb_delete(&mut ctx), 0);
        }
    }

    #[test]
    fn guest_decode_bgra_with_pitch() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A61_0000_0000;
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        // Create param.
        ctx.write_bytes(BASE, &[12, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0]);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_query_memory_size(&mut ctx), MEMORY_SIZE as i64);
        ctx.set(Rsi, BASE + 0x101);
        ctx.set(Rdx, 0x40);
        ctx.set(Rcx, BASE + 0x200);
        assert_eq!(cb_create(&mut ctx), 0);
        let handle = ctx.read_u64(BASE + 0x200).unwrap();
        assert_eq!(handle, BASE + 0x108);

        // RGB 2x1 image, no alpha -> alpha_value fills.
        let png = codec::encode(&[10, 20, 30, 40, 50, 60], 2, 1, 3, 9, codec::FILTER_ALL).unwrap();
        ctx.write_bytes(BASE + 0x1000, &png);
        let mut p = Vec::new();
        p.extend_from_slice(&(BASE + 0x1000).to_le_bytes());
        p.extend_from_slice(&(BASE + 0x3000).to_le_bytes());
        p.extend_from_slice(&(png.len() as u32).to_le_bytes());
        p.extend_from_slice(&0x100u32.to_le_bytes());
        p.extend_from_slice(&PIXEL_FORMAT_B8G8R8A8.to_le_bytes());
        p.extend_from_slice(&0x77u16.to_le_bytes());
        p.extend_from_slice(&16u32.to_le_bytes());
        ctx.write_bytes(BASE + 0x300, &p);
        ctx.set(Rdi, handle);
        ctx.set(Rsi, BASE + 0x300);
        ctx.set(Rdx, BASE + 0x400);
        assert_eq!(cb_decode(&mut ctx), (2 << 16) | 1);
        assert_eq!(ctx.read_bytes(BASE + 0x3000, 8).unwrap(), vec![30, 20, 10, 0x77, 60, 50, 40, 0x77]);
        let info = ctx.read_bytes(BASE + 0x400, 16).unwrap();
        assert_eq!(&info[0..12], &[2, 0, 0, 0, 1, 0, 0, 0, 3, 0, 8, 0]);

        ctx.set(Rdi, handle);
        assert_eq!(cb_delete(&mut ctx), 0);
        assert_eq!(cb_delete(&mut ctx), ERROR_INVALID_HANDLE);
    }

    /// Assemble a PNG from IHDR fields, extra chunks and raw scanlines.
    fn build_png(w: u32, h: u32, depth: u8, color: u8, interlace: u8, extra: &[(&[u8; 4], Vec<u8>)], raw: &[u8]) -> Vec<u8> {
        let mut out = codec::SIGNATURE.to_vec();
        let chunk = |out: &mut Vec<u8>, ty: &[u8; 4], d: &[u8]| {
            out.extend_from_slice(&(d.len() as u32).to_be_bytes());
            out.extend_from_slice(ty);
            out.extend_from_slice(d);
            out.extend_from_slice(&[0; 4]);
        };
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&w.to_be_bytes());
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[depth, color, 0, 0, interlace]);
        chunk(&mut out, b"IHDR", &ihdr);
        for (ty, d) in extra {
            chunk(&mut out, ty, d);
        }
        chunk(&mut out, b"IDAT", &miniz_oxide::deflate::compress_to_vec_zlib(raw, 6));
        chunk(&mut out, b"IEND", &[]);
        out
    }

    #[test]
    fn codec_low_bit_depth_grayscale() {
        // 3x1 grayscale at 1, 2 and 4 bits: samples max, 0, max.
        for (depth, byte, full) in [(1u8, 0b1010_0000u8, 255u8), (2, 0b1100_1100, 255), (4, 0xF0, 255)] {
            let raw = if depth == 4 { vec![0, byte, 0xF0] } else { vec![0, byte] };
            let png = build_png(3, 1, depth, 0, 0, &[], &raw);
            let img = codec::decode(&png).unwrap();
            let g: Vec<u8> = img.pixels.chunks_exact(4).map(|p| p[0]).collect();
            assert_eq!(g, vec![full, 0, full], "depth {depth}");
        }
        // Grayscale tRNS key at 8 bits makes matching samples transparent.
        let png = build_png(2, 1, 8, 0, 0, &[(b"tRNS", vec![0, 7])], &[0, 7, 9]);
        assert!(codec::parse_header(&png).unwrap().has_transparency);
        assert_eq!(codec::decode(&png).unwrap().pixels, vec![7, 7, 7, 0, 9, 9, 9, 255]);
    }

    #[test]
    fn codec_gray_alpha_16_and_rgb_trns() {
        let png = build_png(1, 1, 16, 4, 0, &[], &[0, 0x12, 0x34, 0x80, 0xFF]);
        assert_eq!(codec::decode(&png).unwrap().pixels, vec![0x12, 0x12, 0x12, 0x80]);
        let png = build_png(2, 1, 8, 2, 0, &[(b"tRNS", vec![0, 1, 0, 2, 0, 3])], &[0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(codec::decode(&png).unwrap().pixels, vec![1, 2, 3, 0, 4, 5, 6, 255]);
    }

    #[test]
    fn codec_header_leniency_and_truncation() {
        // Parsing accepts an invalid bit depth (rejected only when decoding).
        let png = build_png(1, 1, 3, 2, 0, &[], &[0, 0, 0, 0]);
        assert_eq!(codec::parse_header(&png).unwrap().bit_depth, 3);
        assert!(codec::decode(&png).is_none());
        // Bad colour type, zero size, short data and a bad signature fail.
        assert!(codec::parse_header(&build_png(1, 1, 8, 5, 0, &[], &[0, 0])).is_none());
        assert!(codec::parse_header(&build_png(0, 1, 8, 0, 0, &[], &[0])).is_none());
        let good = build_png(1, 1, 8, 0, 0, &[], &[0, 0]);
        assert!(codec::parse_header(&good[..32]).is_none());
        assert!(codec::parse_header(&good[..33]).is_some());
        let mut bad = good.clone();
        bad[1] = b'X';
        assert!(codec::parse_header(&bad).is_none());
        // A chunk running past the end of the data ends the walk; the
        // header itself is still valid.
        let mut trunc = good[..33].to_vec();
        trunc.extend_from_slice(&100u32.to_be_bytes());
        trunc.extend_from_slice(b"tEXt");
        trunc.extend_from_slice(&[0; 8]);
        assert!(!codec::parse_header(&trunc).unwrap().has_transparency);
        // A tRNS chunk cut short is not reported; a complete one is.
        let png = palette_png();
        let trns = png.windows(4).position(|w| w == b"tRNS").unwrap() - 4;
        let trns_end = trns + 12 + 1;
        for size in [trns + 8, trns_end - 1] {
            let h = codec::parse_header(&png[..size]).unwrap();
            assert_eq!(h.color_type, ColorType::Palette);
            assert!(!h.has_transparency, "tRNS truncated at {size}");
        }
        assert!(codec::parse_header(&png[..trns_end]).unwrap().has_transparency);
        // tRNS after PLTE is still found (the walk continues to IDAT).
        let png = build_png(1, 1, 8, 3, 0, &[(b"PLTE", vec![1, 2, 3]), (b"tRNS", vec![9])], &[0, 0]);
        assert!(codec::parse_header(&png).unwrap().has_transparency);
        assert_eq!(codec::decode(&png).unwrap().pixels, vec![1, 2, 3, 9]);
    }

    #[test]
    fn guest_validation_and_options() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D93_0000_0000;
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        // Create with the 16-bit attribute; a NULL handle pointer is a
        // parameter error.
        ctx.write_bytes(BASE, &[12, 0, 0, 0, 1, 0, 0, 0, 64, 0, 0, 0]);
        ctx.set(Rdi, BASE);
        ctx.set(Rsi, BASE + 0x100);
        ctx.set(Rdx, 0x40);
        ctx.set(Rcx, 0);
        assert_eq!(cb_create(&mut ctx), ERROR_INVALID_PARAM);
        ctx.set(Rsi, 0);
        ctx.set(Rcx, BASE + 0x200);
        assert_eq!(cb_create(&mut ctx), ERROR_INVALID_ADDR);
        ctx.set(Rsi, BASE + 0x100);
        ctx.set(Rdx, 8);
        assert_eq!(cb_create(&mut ctx), ERROR_INVALID_WORK_MEMORY);
        ctx.set(Rdx, 0x40);
        assert_eq!(cb_create(&mut ctx), 0);
        let handle = ctx.read_u64(BASE + 0x200).unwrap();

        // ParseHeader.
        let png = build_png(2, 1, 16, 0, 1, &[], &[0, 0x10, 0, 0, 0x30, 0]);
        ctx.write_bytes(BASE + 0x1000, &png);
        let parse = |ctx: &mut CpuContext, addr: u64, size: u32, reserved: u32, info: u64| {
            let mut p = Vec::new();
            p.extend_from_slice(&addr.to_le_bytes());
            p.extend_from_slice(&size.to_le_bytes());
            p.extend_from_slice(&reserved.to_le_bytes());
            ctx.write_bytes(BASE + 0x300, &p);
            ctx.set(Rdi, BASE + 0x300);
            ctx.set(Rsi, info);
            cb_parse_header(ctx)
        };
        let n = png.len() as u32;
        assert_eq!(parse(&mut ctx, BASE + 0x1000, n, 0, BASE + 0x400), 0);
        let info = ctx.read_bytes(BASE + 0x400, 16).unwrap();
        assert_eq!(&info[..], &[2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 16, 0, 1, 0, 0, 0]);
        assert_eq!(parse(&mut ctx, BASE + 0x1000, n, 0, 0), ERROR_INVALID_PARAM);
        assert_eq!(parse(&mut ctx, 0, n, 0, BASE + 0x400), ERROR_INVALID_ADDR);
        assert_eq!(parse(&mut ctx, BASE + 0x1000, n, 1, BASE + 0x400), ERROR_INVALID_PARAM);
        assert_eq!(parse(&mut ctx, BASE + 0x1000, 0, 0, BASE + 0x400), ERROR_INVALID_DATA);
        assert_eq!(parse(&mut ctx, BASE + 0x1000, 20, 0, BASE + 0x400), ERROR_INVALID_DATA);
        ctx.set(Rdi, 0);
        assert_eq!(cb_parse_header(&mut ctx), ERROR_INVALID_PARAM);

        // Decode: 16-bit interlaced grayscale to 8-bit RGBA even with the
        // 16-bit attribute.
        let decode = |ctx: &mut CpuContext, png_size: u32, image_size: u32, format: u16, alpha: u16, pitch: u32| {
            let mut p = Vec::new();
            p.extend_from_slice(&(BASE + 0x1000).to_le_bytes());
            p.extend_from_slice(&(BASE + 0x3000).to_le_bytes());
            p.extend_from_slice(&png_size.to_le_bytes());
            p.extend_from_slice(&image_size.to_le_bytes());
            p.extend_from_slice(&format.to_le_bytes());
            p.extend_from_slice(&alpha.to_le_bytes());
            p.extend_from_slice(&pitch.to_le_bytes());
            ctx.write_bytes(BASE + 0x500, &p);
            ctx.write_bytes(BASE + 0x600, &[0u8; 16]);
            ctx.set(Rdi, handle);
            ctx.set(Rsi, BASE + 0x500);
            ctx.set(Rdx, BASE + 0x600);
            cb_decode(ctx)
        };
        assert_eq!(decode(&mut ctx, n, 8, PIXEL_FORMAT_R8G8B8A8, 255, 0), (2 << 16) | 1);
        assert_eq!(ctx.read_bytes(BASE + 0x3000, 8).unwrap(), vec![0x10, 0x10, 0x10, 255, 0x30, 0x30, 0x30, 255]);
        // Size errors still report the image information.
        assert_eq!(decode(&mut ctx, n, 7, PIXEL_FORMAT_R8G8B8A8, 0, 0), ERROR_INVALID_SIZE);
        assert_eq!(ctx.read_u32(BASE + 0x600), Some(2));
        // Measured on a PS5: an alpha above 255, a pitch below the row or
        // not a multiple of 4, or one that overruns the image is PARAM; a
        // zero size is SIZE before the header is read; an image wider than
        // the context's maximum is INVALID_WORK_MEMORY.
        assert_eq!(decode(&mut ctx, n, 8, PIXEL_FORMAT_R8G8B8A8, 300, 0), ERROR_INVALID_PARAM);
        assert_eq!(decode(&mut ctx, n, 0x100, PIXEL_FORMAT_R8G8B8A8, 0, 4), ERROR_INVALID_PARAM);
        assert_eq!(decode(&mut ctx, n, 0x100, PIXEL_FORMAT_R8G8B8A8, 0, 10), ERROR_INVALID_PARAM);
        assert_eq!(decode(&mut ctx, n, 8, PIXEL_FORMAT_R8G8B8A8, 0, 12), ERROR_INVALID_PARAM);
        assert_eq!(decode(&mut ctx, n, 12, PIXEL_FORMAT_R8G8B8A8, 0, 12), (2 << 16) | 1);
        assert_eq!(decode(&mut ctx, n, 0x100, 2, 0, 0), ERROR_INVALID_PARAM);
        assert_eq!(decode(&mut ctx, 0, 0x100, PIXEL_FORMAT_R8G8B8A8, 0, 0), ERROR_INVALID_SIZE);
        assert_eq!(decode(&mut ctx, n, 0, PIXEL_FORMAT_R8G8B8A8, 0, 0), ERROR_INVALID_SIZE);
        let wide = build_png(65, 1, 8, 0, 0, &[], &[vec![0u8], vec![7u8; 65]].concat());
        ctx.write_bytes(BASE + 0x1000, &wide);
        assert_eq!(decode(&mut ctx, wide.len() as u32, 0x1000, PIXEL_FORMAT_R8G8B8A8, 0, 0), ERROR_INVALID_WORK_MEMORY);
        // A header the decoder rejects.
        let bad = build_png(1, 1, 3, 0, 0, &[], &[0, 0]);
        ctx.write_bytes(BASE + 0x1000, &bad);
        assert_eq!(decode(&mut ctx, bad.len() as u32, 4, PIXEL_FORMAT_R8G8B8A8, 0, 0), ERROR_DECODE_ERROR);

        ctx.set(Rdi, 0);
        assert_eq!(cb_decode(&mut ctx), ERROR_INVALID_HANDLE);
        ctx.set(Rdi, handle);
        assert_eq!(cb_delete(&mut ctx), 0);
    }

    const GRAY16: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x01, 0x10, 0x00, 0x00, 0x00, 0x00, 0x81, 0xD9, 0xFC, 0x15, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x10, 0x32,
        0x59, 0x7D, 0x16, 0x00, 0x03, 0x0C, 0x01, 0xBF, 0x6E, 0xB9, 0xC6, 0x5D, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60,
        0x82,
    ];
    const GRAY16_TRNS: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x02, 0x10, 0x00, 0x00, 0x00, 0x00, 0x07, 0x4D, 0x8E, 0xBB, 0x00, 0x00, 0x00, 0x02, 0x74, 0x52, 0x4E, 0x53, 0x12, 0x34, 0x2F, 0xD3, 0x49,
        0x5E, 0x00, 0x00, 0x00, 0x12, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x10, 0x32, 0x11, 0x5A, 0xCD, 0xB0, 0xFA, 0x2C, 0x03, 0x03, 0x00,
        0x0B, 0xDF, 0x02, 0x7C, 0x4E, 0x3C, 0x25, 0x55, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];
    const RGB16_TRNS: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x02, 0x10, 0x02, 0x00, 0x00, 0x00, 0xAD, 0x44, 0x46, 0x30, 0x00, 0x00, 0x00, 0x06, 0x74, 0x52, 0x4E, 0x53, 0x12, 0x34, 0x56, 0x78, 0x9A,
        0xBC, 0x89, 0xE4, 0x4E, 0xE6, 0x00, 0x00, 0x00, 0x23, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x10, 0x32, 0x09, 0xAB, 0x98, 0xB5, 0x47,
        0xE8, 0x7F, 0xD8, 0xFF, 0x59, 0xFF, 0x19, 0xFE, 0x33, 0x30, 0x30, 0x36, 0x30, 0x32, 0x30, 0x34, 0x30, 0xFC, 0xFF, 0x0F, 0x00, 0x88, 0xFB,
        0x0A, 0x69, 0xC3, 0x3F, 0x47, 0x19, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];
    const GRAY_ALPHA16: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x02, 0x10, 0x04, 0x00, 0x00, 0x00, 0x88, 0x2F, 0x19, 0xEC, 0x00, 0x00, 0x00, 0x19, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x10, 0x32,
        0xB9, 0xF7, 0x61, 0xF5, 0x59, 0x06, 0x20, 0xF8, 0xF7, 0xFF, 0xFF, 0x7F, 0xC6, 0x86, 0x46, 0x00, 0x48, 0x58, 0x08, 0x8A, 0x8F, 0xC8, 0x5C,
        0x92, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];
    const RGBA16: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x02, 0x10, 0x06, 0x00, 0x00, 0x00, 0x22, 0x26, 0xD1, 0x67, 0x00, 0x00, 0x00, 0x2D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x01, 0x22, 0x00,
        0xDD, 0xFF, 0x00, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0x0D, 0xEF, 0xAB, 0xCD, 0x20, 0x01, 0xEE, 0xFE, 0xFF, 0xFF, 0x00, 0x00, 0x01, 0x80,
        0x40, 0xFF, 0xAB, 0x00, 0x00, 0xFF, 0xFF, 0x01, 0x02, 0x7F, 0xFE, 0x80, 0xFF, 0x07, 0xD0, 0x10, 0x52, 0x90, 0xA2, 0xAC, 0x25, 0x00, 0x00,
        0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    /// 16-bit images: a decoder without the 16-bit attribute refuses them
    /// with INVALID_WORK_MEMORY after reporting the image information and
    /// writes nothing; with it they decode to the high bytes (tRNS keys and
    /// alpha channels included) in either pixel format and with a pitch;
    /// size and pitch errors come first on both decoders.
    #[test]
    fn guest_decode_16_bit_needs_the_attribute() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D93_0100_0000;
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let create = |ctx: &mut CpuContext, attribute: u8, at: u64| {
            ctx.write_bytes(BASE, &[12, 0, 0, 0, attribute, 0, 0, 0, 64, 0, 0, 0]);
            ctx.set(Rdi, BASE);
            ctx.set(Rsi, at);
            ctx.set(Rdx, 0x40);
            ctx.set(Rcx, BASE + 0x80);
            assert_eq!(cb_create(ctx), 0);
            ctx.read_u64(BASE + 0x80).unwrap()
        };
        let limited = create(&mut ctx, 0, BASE + 0x100);
        let enabled = create(&mut ctx, ATTRIBUTE_BIT_DEPTH_16 as u8, BASE + 0x200);
        const OUT: u64 = BASE + 0x3000;
        let decode = |ctx: &mut CpuContext, handle: u64, png: &[u8], image_size: u32, format: u16, pitch: u32| {
            ctx.write_bytes(BASE + 0x1000, png);
            let mut p = Vec::new();
            p.extend_from_slice(&(BASE + 0x1000).to_le_bytes());
            p.extend_from_slice(&OUT.to_le_bytes());
            p.extend_from_slice(&(png.len() as u32).to_le_bytes());
            p.extend_from_slice(&image_size.to_le_bytes());
            p.extend_from_slice(&format.to_le_bytes());
            p.extend_from_slice(&0x77u16.to_le_bytes());
            p.extend_from_slice(&pitch.to_le_bytes());
            ctx.write_bytes(BASE + 0x500, &p);
            ctx.write_bytes(BASE + 0x600, &[0u8; 16]);
            ctx.set(Rdi, handle);
            ctx.set(Rsi, BASE + 0x500);
            ctx.set(Rdx, BASE + 0x600);
            cb_decode(ctx)
        };
        let untouched = |ctx: &CpuContext| ctx.read_bytes(OUT, 0x40).unwrap().iter().all(|&b| b == 0xCC);
        let cases: [(&[u8], &[u8], u16, u32, u32); 5] = [
            (GRAY16, &[0x12, 0x12, 0x12, 0x77, 0xAB, 0xAB, 0xAB, 0x77], COLOR_SPACE_GRAYSCALE, 0, 1),
            (GRAY16_TRNS, &[0x12, 0x12, 0x12, 0, 0x12, 0x12, 0x12, 0xFF, 0xAB, 0xAB, 0xAB, 0xFF, 0, 0, 0, 0xFF], COLOR_SPACE_GRAYSCALE, 2, 2),
            (RGB16_TRNS, &[0x12, 0x56, 0x9A, 0, 0x12, 0x56, 0x9A, 0xFF, 0xFF, 0, 0x80, 0xFF, 0, 0x80, 0xFF, 0xFF], COLOR_SPACE_RGB, 2, 2),
            (
                GRAY_ALPHA16,
                &[0x12, 0x12, 0x12, 0xDE, 0xAB, 0xAB, 0xAB, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0x80],
                COLOR_SPACE_GRAYSCALE_ALPHA,
                0,
                2,
            ),
            (RGBA16, &[0x12, 0x56, 0x9A, 0x0D, 0xAB, 0x20, 0xEE, 0xFF, 0, 0x80, 0xFF, 0, 0xFF, 1, 0x7F, 0x80], COLOR_SPACE_RGBA, 0, 2),
        ];
        for (png, expected, space, flags, height) in cases {
            let mut info = vec![2, 0, 0, 0];
            info.extend_from_slice(&height.to_le_bytes());
            info.extend_from_slice(&space.to_le_bytes());
            info.extend_from_slice(&16u16.to_le_bytes());
            info.extend_from_slice(&flags.to_le_bytes());
            for format in [PIXEL_FORMAT_R8G8B8A8, PIXEL_FORMAT_B8G8R8A8] {
                for padding in [0u32, 8] {
                    let pitch = 8 + padding;
                    let size = pitch * height;
                    let image_pitch = if padding == 0 { 0 } else { pitch };
                    ctx.write_bytes(OUT, &[0xCC; 0x40]);
                    assert_eq!(decode(&mut ctx, limited, png, size, format, image_pitch), ERROR_INVALID_WORK_MEMORY);
                    assert_eq!(ctx.read_bytes(BASE + 0x600, 16).unwrap(), info);
                    assert!(untouched(&ctx));
                    let packed = (2 << 16) | height as HleResult;
                    assert_eq!(decode(&mut ctx, enabled, png, size, format, image_pitch), packed);
                    assert_eq!(ctx.read_bytes(BASE + 0x600, 16).unwrap(), info);
                    let out = ctx.read_bytes(OUT, 0x40).unwrap();
                    for y in 0..height as usize {
                        for x in 0..2 {
                            for ch in 0..4 {
                                let src = if format == PIXEL_FORMAT_B8G8R8A8 && ch < 3 { 2 - ch } else { ch };
                                assert_eq!(out[y * pitch as usize + x * 4 + ch], expected[y * 8 + x * 4 + src], "y {y} x {x} ch {ch}");
                            }
                        }
                        assert!(out[y * pitch as usize + 8..(y + 1) * pitch as usize].iter().all(|&b| b == 0xCC));
                    }
                    assert!(out[size as usize..].iter().all(|&b| b == 0xCC));
                    for handle in [limited, enabled] {
                        ctx.write_bytes(OUT, &[0xCC; 0x40]);
                        let short = if padding == 0 { ERROR_INVALID_SIZE } else { ERROR_INVALID_PARAM };
                        assert_eq!(decode(&mut ctx, handle, png, size - 1, format, image_pitch), short);
                        assert_eq!(decode(&mut ctx, handle, png, size - 1, format, 7), ERROR_INVALID_PARAM);
                        assert!(untouched(&ctx));
                    }
                }
            }
        }
    }
}
