// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceVideodec2: access-unit video decoding (H.264, HEVC, VP9) on the
//! host decoder.
//!
//! Decoding is synchronous: `sceVideodec2Decode` feeds one access unit and,
//! when the decoder has a picture, writes it into the caller's frame buffer
//! as NV12 (pitch = width aligned to 256, chroma right after `height` luma
//! rows). Per-picture metadata (timestamps, SPS fields) is remembered by
//! frame buffer address for `sceVideodec2Get*PictureInfo`.

use crate::avplayer::host::{RawCodec, RawDecoder, RawError, RawPicture};
use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

const LIB: &str = "libSceVideodec2";

const OK: HleResult = 0;
const ERROR_API_FAIL: u32 = 0x811D_0100;
const ERROR_STRUCT_SIZE: u32 = 0x811D_0101;
const ERROR_ARGUMENT_POINTER: u32 = 0x811D_0102;
const ERROR_DECODER_INSTANCE: u32 = 0x811D_0103;
const ERROR_MEMORY_SIZE: u32 = 0x811D_0104;
const ERROR_MEMORY_POINTER: u32 = 0x811D_0105;
const ERROR_FRAME_BUFFER_SIZE: u32 = 0x811D_0106;
const ERROR_FRAME_BUFFER_POINTER: u32 = 0x811D_0107;
const ERROR_ACCESS_UNIT_SIZE: u32 = 0x811D_010D;
const ERROR_ACCESS_UNIT_POINTER: u32 = 0x811D_010E;
const ERROR_OUTPUT_INFO: u32 = 0x811D_010F;
const ERROR_COMPUTE_QUEUE: u32 = 0x811D_0110;
const ERROR_CONFIG_INFO: u32 = 0x811D_0200;
const ERROR_COMPUTE_PIPE_ID: u32 = 0x811D_0201;
const ERROR_COMPUTE_QUEUE_ID: u32 = 0x811D_0202;
const ERROR_RESOURCE_TYPE: u32 = 0x811D_0203;
const ERROR_CODEC_TYPE: u32 = 0x811D_0204;
const ERROR_INPUT_QUEUE_DEPTH: u32 = 0x811D_0206;
const ERROR_DPB_FRAME_COUNT: u32 = 0x811D_0209;
const ERROR_FRAME_WIDTH_HEIGHT: u32 = 0x811D_020A;
const ERROR_ACCESS_UNIT: u32 = 0x811D_0301;
const ERROR_OVERSIZE_DECODE: u32 = 0x811D_0302;

const RESOURCE_TYPE_COMPUTE: u32 = 1;
const MIN_MEMORY_SIZE: u64 = 16 * 1024 * 1024;
const CODEC_AVC: u32 = 1;
const CODEC_HEVC: u32 = 974_921;
const CODEC_VP9: u32 = 2_382_845;

const COMPUTE_MEMORY_INFO_SIZE: u64 = 24;
const COMPUTE_CONFIG_INFO_SIZE: u64 = 16;
const DECODER_CONFIG_INFO_SIZE: u64 = 72;
const DECODER_MEMORY_INFO_SIZE: u64 = 72;
const INPUT_DATA_SIZE: u64 = 48;
const OUTPUT_INFO_SIZE: u64 = 56;
const FRAME_BUFFER_SIZE: u64 = 32;
const AVC_PICTURE_INFO_SIZE: u64 = 120;

fn err(code: u32) -> HleResult {
    code as i32 as HleResult
}

fn raw_codec(codec_type: u32) -> Option<RawCodec> {
    match codec_type {
        CODEC_AVC => Some(RawCodec::H264),
        CODEC_HEVC => Some(RawCodec::Hevc),
        CODEC_VP9 => Some(RawCodec::Vp9),
        _ => None,
    }
}

struct Decoder {
    codec_type: u32,
    max_width: i32,
    max_height: i32,
    raw: RawDecoder,
    /// VP9: header parser state and the shown frames still to be output,
    /// oldest first (VP9 pictures leave the decoder in order).
    vp9: vp9::State,
    vp9_pending: std::collections::VecDeque<vp9::Info>,
    /// H.264/HEVC access unit framing seen so far.
    framing: Framing,
}

/// VP9 frames waiting for their picture at most.
const VP9_MAX_PENDING: usize = 16;

struct Registry {
    decoders: HashMap<u64, Arc<Mutex<Decoder>>>,
    /// Picture metadata by frame buffer address, tagged with its decoder.
    pictures: HashMap<u64, (u64, RawPicture, u32, Option<vp9::Info>)>,
}

static STATE: Mutex<Option<Registry>> = Mutex::new(None);

fn with_state<R>(f: impl FnOnce(&mut Registry) -> R) -> R {
    let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(|| Registry { decoders: HashMap::new(), pictures: HashMap::new() }))
}

fn decoder(handle: u64) -> Option<Arc<Mutex<Decoder>>> {
    with_state(|s| s.decoders.get(&handle).cloned())
}

fn u8_at(ctx: &CpuContext, a: u64) -> u8 {
    ctx.read_bytes(a, 1).map_or(0, |b| b[0])
}

fn u16_at(ctx: &CpuContext, a: u64) -> u16 {
    ctx.read_bytes(a, 2).map_or(0, |b| u16::from_le_bytes([b[0], b[1]]))
}

struct Config {
    resource_type: u32,
    codec_type: u32,
    max_width: i32,
    max_height: i32,
    max_dpb: i32,
    queue_depth: u32,
    compute_queue: u64,
    reserved0: u8,
    reserved1: u8,
}

fn read_config(ctx: &CpuContext, a: u64) -> Config {
    Config {
        resource_type: ctx.read_u32(a + 8).unwrap_or(0),
        codec_type: ctx.read_u32(a + 12).unwrap_or(0),
        max_width: ctx.read_u32(a + 24).unwrap_or(0) as i32,
        max_height: ctx.read_u32(a + 28).unwrap_or(0) as i32,
        max_dpb: ctx.read_u32(a + 32).unwrap_or(0) as i32,
        queue_depth: ctx.read_u32(a + 36).unwrap_or(0),
        compute_queue: ctx.read_u64(a + 40).unwrap_or(0),
        reserved0: u8_at(ctx, a + 62),
        reserved1: u8_at(ctx, a + 63),
    }
}

fn validate_config(c: &Config, require_queue: bool) -> HleResult {
    if c.resource_type != RESOURCE_TYPE_COMPUTE {
        return err(ERROR_RESOURCE_TYPE);
    }
    if raw_codec(c.codec_type).is_none() {
        return err(ERROR_CODEC_TYPE);
    }
    if c.reserved0 != 0 || c.reserved1 != 0 {
        return err(ERROR_CONFIG_INFO);
    }
    if c.queue_depth == 0 {
        return err(ERROR_INPUT_QUEUE_DEPTH);
    }
    if c.max_dpb < -1 || c.max_dpb == 0 {
        return err(ERROR_DPB_FRAME_COUNT);
    }
    if c.max_width < -1 || c.max_height < -1 || c.max_width == 0 || c.max_height == 0 {
        return err(ERROR_FRAME_WIDTH_HEIGHT);
    }
    if require_queue && c.compute_queue == 0 {
        return err(ERROR_COMPUTE_QUEUE);
    }
    OK
}

/// sceVideodec2QueryComputeMemoryInfo(SceVideodec2ComputeMemoryInfo*)
fn cb_query_compute_memory_info(ctx: &mut CpuContext) -> HleResult {
    let info = ctx.rdi();
    if info == 0 {
        return err(ERROR_ARGUMENT_POINTER);
    }
    if ctx.read_u64(info) != Some(COMPUTE_MEMORY_INFO_SIZE) {
        return err(ERROR_STRUCT_SIZE);
    }
    ctx.write_u64(info + 8, MIN_MEMORY_SIZE);
    ctx.write_u64(info + 16, 0);
    OK
}

/// sceVideodec2AllocateComputeQueue(const ConfigInfo*, const MemoryInfo*, ComputeQueue*)
fn cb_allocate_compute_queue(ctx: &mut CpuContext) -> HleResult {
    let (config, memory, queue) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if config == 0 || memory == 0 || queue == 0 {
        return err(ERROR_ARGUMENT_POINTER);
    }
    if ctx.read_u64(config) != Some(COMPUTE_CONFIG_INFO_SIZE) || ctx.read_u64(memory) != Some(COMPUTE_MEMORY_INFO_SIZE) {
        return err(ERROR_STRUCT_SIZE);
    }
    if u8_at(ctx, config + 13) != 0 || u16_at(ctx, config + 14) != 0 {
        return err(ERROR_CONFIG_INFO);
    }
    if u16_at(ctx, config + 8) > 4 {
        return err(ERROR_COMPUTE_PIPE_ID);
    }
    if u16_at(ctx, config + 10) > 7 {
        return err(ERROR_COMPUTE_QUEUE_ID);
    }
    if ctx.read_u64(memory + 8).unwrap_or(0) < MIN_MEMORY_SIZE {
        return err(ERROR_MEMORY_SIZE);
    }
    let mem = ctx.read_u64(memory + 16).unwrap_or(0);
    if mem == 0 {
        return err(ERROR_MEMORY_POINTER);
    }
    ctx.write_u64(queue, mem);
    OK
}

/// sceVideodec2ReleaseComputeQueue(ComputeQueue)
fn cb_release_compute_queue(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() != 0 {
        OK
    } else {
        err(ERROR_COMPUTE_QUEUE_ID)
    }
}

/// sceVideodec2QueryDecoderMemoryInfo(const DecoderConfigInfo*, DecoderMemoryInfo*)
fn cb_query_decoder_memory_info(ctx: &mut CpuContext) -> HleResult {
    query_decoder_memory_info(ctx, ctx.rdi(), ctx.rsi(), DECODER_CONFIG_INFO_SIZE)
}

/// Decoder memory query for a configuration of `config_size` bytes (its
/// leading fields are those of SceVideodec2DecoderConfigInfo).
fn query_decoder_memory_info(ctx: &CpuContext, config: u64, memory: u64, config_size: u64) -> HleResult {
    if config == 0 || memory == 0 {
        return err(ERROR_ARGUMENT_POINTER);
    }
    if ctx.read_u64(config) != Some(config_size) || ctx.read_u64(memory) != Some(DECODER_MEMORY_INFO_SIZE) {
        return err(ERROR_STRUCT_SIZE);
    }
    let r = validate_config(&read_config(ctx, config), false);
    if r != OK {
        return r;
    }
    for (off, v) in [(8, MIN_MEMORY_SIZE), (16, 0), (24, MIN_MEMORY_SIZE), (32, 0), (40, MIN_MEMORY_SIZE), (48, 0), (56, MIN_MEMORY_SIZE)] {
        ctx.write_u64(memory + off, v);
    }
    ctx.write_u32(memory + 64, 0x100);
    ctx.write_u32(memory + 68, 0);
    OK
}

/// sceVideodec2CreateDecoder(const DecoderConfigInfo*, const DecoderMemoryInfo*, Decoder*)
fn cb_create_decoder(ctx: &mut CpuContext) -> HleResult {
    create_decoder(ctx, ctx.rdi(), ctx.rsi(), ctx.rdx(), DECODER_CONFIG_INFO_SIZE)
}

/// Decoder creation for a configuration of `config_size` bytes.
fn create_decoder(ctx: &CpuContext, config: u64, memory: u64, out: u64, config_size: u64) -> HleResult {
    if config == 0 || memory == 0 || out == 0 {
        return err(ERROR_ARGUMENT_POINTER);
    }
    if ctx.read_u64(config) != Some(config_size) || ctx.read_u64(memory) != Some(DECODER_MEMORY_INFO_SIZE) {
        return err(ERROR_STRUCT_SIZE);
    }
    let c = read_config(ctx, config);
    let r = validate_config(&c, true);
    if r != OK {
        return r;
    }
    let m = |o: u64| ctx.read_u64(memory + o).unwrap_or(0);
    if m(8) < MIN_MEMORY_SIZE || m(24) < MIN_MEMORY_SIZE || m(40) < MIN_MEMORY_SIZE || m(56) < MIN_MEMORY_SIZE {
        return err(ERROR_MEMORY_SIZE);
    }
    if m(16) == 0 || m(32) == 0 || m(48) == 0 {
        return err(ERROR_MEMORY_POINTER);
    }
    let Some(raw) = raw_codec(c.codec_type).and_then(RawDecoder::new) else {
        log::warn!("[videodec2] no host decoder for codec type {}", c.codec_type);
        return err(ERROR_API_FAIL);
    };
    let Some(handle) = crate::pthread::alloc_guest_block(ctx, 0x100) else { return err(ERROR_API_FAIL) };
    let d = Decoder {
        codec_type: c.codec_type,
        max_width: c.max_width,
        max_height: c.max_height,
        raw,
        vp9: vp9::State::default(),
        vp9_pending: Default::default(),
        framing: Framing::default(),
    };
    with_state(|s| s.decoders.insert(handle, Arc::new(Mutex::new(d))));
    ctx.write_u64(out, handle);
    log::info!("[videodec2] decoder {handle:#x} codec={} max={}x{}", c.codec_type, c.max_width, c.max_height);
    OK
}

/// sceVideodec2DeleteDecoder(Decoder)
fn cb_delete_decoder(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    let removed = with_state(|s| {
        s.pictures.retain(|_, (owner, ..)| *owner != handle);
        s.decoders.remove(&handle)
    });
    if removed.is_some() {
        OK
    } else {
        err(ERROR_DECODER_INSTANCE)
    }
}

fn output_size_valid(size: u64) -> bool {
    size == OUTPUT_INFO_SIZE || (size | 8) == OUTPUT_INFO_SIZE
}

struct FrameBuffer {
    addr: u64,
    data: u64,
    size: u64,
}

fn read_frame_buffer(ctx: &CpuContext, a: u64) -> FrameBuffer {
    FrameBuffer { addr: a, data: ctx.read_u64(a + 8).unwrap_or(0), size: ctx.read_u64(a + 16).unwrap_or(0) }
}

fn write_no_picture(ctx: &CpuContext, out: u64, fb: &FrameBuffer, codec_type: u32) {
    let this_size = ctx.read_u64(out).unwrap_or(0);
    ctx.write_bytes(out + 8, &[0, 0, 0, 0]);
    ctx.write_u32(out + 12, codec_type);
    ctx.write_bytes(out + 16, &[0; 12]);
    ctx.write_u64(out + 32, fb.data);
    ctx.write_u64(out + 40, fb.size);
    if this_size == OUTPUT_INFO_SIZE {
        ctx.write_u64(out + 48, 0);
    }
    ctx.write_bytes(fb.addr + 24, &[0]);
}

/// Store a decoded picture in the guest frame buffer and fill the output info.
fn publish(ctx: &CpuContext, handle: u64, d: &mut Decoder, pic: RawPicture, fb: &FrameBuffer, out: u64) -> HleResult {
    if (d.max_width > 0 && pic.width > d.max_width as u32) || (d.max_height > 0 && pic.height > d.max_height as u32) {
        return err(ERROR_OVERSIZE_DECODE);
    }
    let pitch = pic.pitch;
    let need = pitch as u64 * pic.height as u64 + pitch as u64 * pic.height.div_ceil(2) as u64;
    if need > fb.size {
        return err(ERROR_FRAME_BUFFER_SIZE);
    }
    if !ps5vm::ranges::accessible(fb.data, need, true) {
        return err(ERROR_FRAME_BUFFER_POINTER);
    }
    // SAFETY: the frame buffer is mapped writable for `need <= fb.size` bytes.
    match unsafe { d.raw.copy_nv12(fb.data as *mut u8, need) } {
        Ok(()) => {}
        Err(RawError::FrameBufferSize) => return err(ERROR_FRAME_BUFFER_SIZE),
        Err(_) => return err(ERROR_API_FAIL),
    }
    let this_size = ctx.read_u64(out).unwrap_or(0);
    ctx.write_bytes(out + 8, &[1, pic.corrupt as u8, 1, 0]);
    ctx.write_u32(out + 12, d.codec_type);
    ctx.write_u32(out + 16, pic.width);
    ctx.write_u32(out + 20, pitch);
    ctx.write_u32(out + 24, pic.height);
    ctx.write_u64(out + 32, fb.data);
    ctx.write_u64(out + 40, fb.size);
    if this_size == OUTPUT_INFO_SIZE {
        ctx.write_u32(out + 48, 0);
        ctx.write_u32(out + 52, pitch);
    }
    ctx.write_bytes(fb.addr + 24, &[1]);
    // VP9: the oldest shown frame is this picture.
    let vp9 = if d.codec_type == CODEC_VP9 { d.vp9_pending.pop_front() } else { None };
    with_state(|s| s.pictures.insert(fb.data, (handle, pic, d.codec_type, vp9)));
    OK
}

fn validate_output(ctx: &CpuContext, fb: u64, out: u64) -> Result<FrameBuffer, HleResult> {
    if ctx.read_u64(fb) != Some(FRAME_BUFFER_SIZE) || !output_size_valid(ctx.read_u64(out).unwrap_or(0)) {
        return Err(err(ERROR_STRUCT_SIZE));
    }
    let f = read_frame_buffer(ctx, fb);
    if f.size == 0 {
        return Err(err(ERROR_FRAME_BUFFER_SIZE));
    }
    if f.data == 0 {
        return Err(err(ERROR_FRAME_BUFFER_POINTER));
    }
    Ok(f)
}

/// Parse the frame headers of a VP9 access unit and queue the information
/// of its shown frames (their pictures leave the decoder in order).
fn note_vp9_frames(d: &mut Decoder, raw: &[u8], pts: u64, dts: u64, attached: u64) -> Result<(), String> {
    let mut state = d.vp9;
    let frames = vp9::parse_packet(raw, &mut state).map_err(|e| e.to_string())?;
    d.vp9 = state;
    for (mut info, _) in frames.into_iter().filter(|f| f.1) {
        (info.pts, info.dts, info.attached) = (pts, dts, attached);
        if d.vp9_pending.len() == VP9_MAX_PENDING {
            d.vp9_pending.pop_front();
        }
        d.vp9_pending.push_back(info);
    }
    Ok(())
}

/// How a decoder's earlier access units were framed, for a unit that
/// reads both ways.
#[derive(Default, Clone, Copy)]
pub(crate) struct Framing {
    length_prefixed: bool,
    short_start_codes: bool,
}

/// Whether `data` is entirely 4-byte big-endian lengths, each followed by a
/// non-empty NAL unit that fits.
fn tiles_as_avcc(data: &[u8]) -> bool {
    if data.len() < 4 {
        return false;
    }
    let mut index = 0;
    while index < data.len() {
        let Some(header) = data.get(index..index + 4) else { return false };
        let length = u32::from_be_bytes(header.try_into().unwrap()) as usize;
        if length == 0 || length > data.len() - index - 4 {
            return false;
        }
        index += 4 + length;
    }
    true
}

/// An H.264/HEVC access unit in Annex B form. A unit that reads as both a
/// start-code stream and a run of length-prefixed NAL units is taken the
/// way the decoder's earlier units were (a unit opening with a 3-byte
/// start code is length-prefixed unless 3-byte start codes were seen);
/// otherwise its own form decides and is remembered. None when it is
/// neither.
fn to_annex_b<'a>(data: &'a [u8], framing: &mut Framing) -> Option<std::borrow::Cow<'a, [u8]>> {
    let tiles = tiles_as_avcc(data);
    let opens = data.len() >= 3 && data[0] == 0 && data[1] == 0 && (data[2] == 1 || (data.len() >= 4 && data[2] == 0 && data[3] == 1));
    if !tiles && !opens {
        return None;
    }
    let short = opens && data[2] == 1;
    let length_prefixed = if tiles && opens {
        if short { !framing.short_start_codes } else { framing.length_prefixed }
    } else {
        framing.length_prefixed = tiles;
        if short {
            framing.short_start_codes = true;
        }
        tiles
    };
    if !length_prefixed {
        return Some(std::borrow::Cow::Borrowed(data));
    }
    let mut out = Vec::with_capacity(data.len() + 16);
    let mut index = 0;
    while index < data.len() {
        let length = u32::from_be_bytes(data[index..index + 4].try_into().unwrap()) as usize;
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&data[index + 4..index + 4 + length]);
        index += 4 + length;
    }
    Some(std::borrow::Cow::Owned(out))
}

/// A decoded picture held in host memory as NV12.
struct HostPicture {
    pic: RawPicture,
    pixels: Vec<u8>,
}

/// NV12 size of a picture: `height` luma rows and half as many chroma rows.
fn nv12_size(pic: &RawPicture) -> u64 {
    pic.pitch as u64 * pic.height as u64 + pic.pitch as u64 * pic.height.div_ceil(2) as u64
}

/// Copy the decoder's current picture into host memory.
fn take_host_picture(d: &mut Decoder, pic: RawPicture) -> Result<HostPicture, HleResult> {
    if (d.max_width > 0 && pic.width > d.max_width as u32) || (d.max_height > 0 && pic.height > d.max_height as u32) {
        return Err(err(ERROR_OVERSIZE_DECODE));
    }
    let need = nv12_size(&pic);
    let mut pixels = vec![0u8; need as usize];
    // SAFETY: `pixels` is writable for `need` bytes.
    match unsafe { d.raw.copy_nv12(pixels.as_mut_ptr(), need) } {
        Ok(()) => Ok(HostPicture { pic, pixels }),
        Err(RawError::FrameBufferSize) => Err(err(ERROR_FRAME_BUFFER_SIZE)),
        Err(_) => Err(err(ERROR_API_FAIL)),
    }
}

/// Decode one access unit into host memory (Annex B conversion as for
/// sceVideodec2Decode).
fn decode_to_host(d: &mut Decoder, raw: &[u8], pts: u64, dts: u64, attached: u64) -> Result<Option<HostPicture>, HleResult> {
    let converted;
    let data = if d.codec_type == CODEC_VP9 {
        raw
    } else {
        match to_annex_b(raw, &mut d.framing) {
            Some(std::borrow::Cow::Borrowed(b)) => b,
            Some(std::borrow::Cow::Owned(v)) => {
                converted = v;
                &converted[..]
            }
            None => return Err(err(ERROR_ACCESS_UNIT)),
        }
    };
    match d.raw.decode(data, pts, dts, attached) {
        Ok(Some(pic)) => take_host_picture(d, pic).map(Some),
        Ok(None) => Ok(None),
        Err(RawError::AccessUnit) => Err(err(ERROR_ACCESS_UNIT)),
        Err(RawError::FrameBufferSize) => Err(err(ERROR_FRAME_BUFFER_SIZE)),
        Err(RawError::ApiFail) => Err(err(ERROR_API_FAIL)),
    }
}

/// Drain one remaining picture into host memory.
fn flush_to_host(d: &mut Decoder) -> Result<Option<HostPicture>, HleResult> {
    match d.raw.flush() {
        Ok(Some(pic)) => take_host_picture(d, pic).map(Some),
        Ok(None) => Ok(None),
        Err(_) => Err(err(ERROR_API_FAIL)),
    }
}

/// sceVideodec2Decode(Decoder, const InputData*, FrameBuffer*, OutputInfo*)
fn cb_decode(ctx: &mut CpuContext) -> HleResult {
    let (handle, input, fb, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    let Some(d) = decoder(handle) else { return err(ERROR_DECODER_INSTANCE) };
    if input == 0 || fb == 0 || out == 0 {
        return err(ERROR_ARGUMENT_POINTER);
    }
    if ctx.read_u64(input) != Some(INPUT_DATA_SIZE)
        || ctx.read_u64(fb) != Some(FRAME_BUFFER_SIZE)
        || !output_size_valid(ctx.read_u64(out).unwrap_or(0))
    {
        return err(ERROR_STRUCT_SIZE);
    }
    let au = ctx.read_u64(input + 8).unwrap_or(0);
    let au_size = ctx.read_u64(input + 16).unwrap_or(0);
    if au_size == 0 {
        return err(ERROR_ACCESS_UNIT_SIZE);
    }
    if au == 0 {
        return err(ERROR_ACCESS_UNIT_POINTER);
    }
    let fbuf = match validate_output(ctx, fb, out) {
        Ok(f) => f,
        Err(e) => return e,
    };
    if au_size > i32::MAX as u64 || !ps5vm::ranges::accessible(au, au_size, false) {
        return err(ERROR_ACCESS_UNIT_POINTER);
    }
    // SAFETY: the access unit is mapped readable (checked above).
    let raw = unsafe { std::slice::from_raw_parts(au as *const u8, au_size as usize) };
    let pts = ctx.read_u64(input + 24).unwrap_or(u64::MAX);
    let dts = ctx.read_u64(input + 32).unwrap_or(u64::MAX);
    let attached = ctx.read_u64(input + 40).unwrap_or(0);
    let mut d = d.lock().unwrap_or_else(|e| e.into_inner());
    write_no_picture(ctx, out, &fbuf, d.codec_type);
    // VP9 frames go to the decoder as they are; H.264 and HEVC units may
    // carry 4-byte NAL lengths instead of start codes.
    let converted;
    let data = if d.codec_type == CODEC_VP9 {
        if let Err(e) = note_vp9_frames(&mut d, raw, pts, dts, attached) {
            log::error!("[videodec2] decoder {handle:#x}: VP9 access unit rejected: {e}");
            return err(ERROR_ACCESS_UNIT);
        }
        raw
    } else {
        match to_annex_b(raw, &mut d.framing) {
            Some(std::borrow::Cow::Borrowed(b)) => b,
            Some(std::borrow::Cow::Owned(v)) => {
                converted = v;
                &converted[..]
            }
            None => return err(ERROR_ACCESS_UNIT),
        }
    };
    match d.raw.decode(data, pts, dts, attached) {
        Ok(Some(pic)) => publish(ctx, handle, &mut d, pic, &fbuf, out),
        Ok(None) => OK,
        Err(RawError::AccessUnit) => err(ERROR_ACCESS_UNIT),
        Err(RawError::FrameBufferSize) => err(ERROR_FRAME_BUFFER_SIZE),
        Err(RawError::ApiFail) => err(ERROR_API_FAIL),
    }
}

/// sceVideodec2Flush(Decoder, FrameBuffer*, OutputInfo*)
fn cb_flush(ctx: &mut CpuContext) -> HleResult {
    let (handle, fb, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let Some(d) = decoder(handle) else { return err(ERROR_DECODER_INSTANCE) };
    if fb == 0 || out == 0 {
        return err(ERROR_ARGUMENT_POINTER);
    }
    let fbuf = match validate_output(ctx, fb, out) {
        Ok(f) => f,
        Err(e) => return e,
    };
    let mut d = d.lock().unwrap_or_else(|e| e.into_inner());
    write_no_picture(ctx, out, &fbuf, d.codec_type);
    match d.raw.flush() {
        Ok(Some(pic)) => publish(ctx, handle, &mut d, pic, &fbuf, out),
        Ok(None) => OK,
        Err(_) => err(ERROR_API_FAIL),
    }
}

/// sceVideodec2Reset(Decoder)
fn cb_reset(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    let Some(d) = decoder(handle) else { return err(ERROR_DECODER_INSTANCE) };
    {
        let mut d = d.lock().unwrap_or_else(|e| e.into_inner());
        d.raw.reset();
        d.vp9 = vp9::State::default();
        d.vp9_pending.clear();
    }
    with_state(|s| s.pictures.retain(|_, (owner, ..)| *owner != handle));
    OK
}

fn encode_avc_picture(pic: &RawPicture, this_size: u64) -> [u8; AVC_PICTURE_INFO_SIZE as usize] {
    let mut b = [0u8; AVC_PICTURE_INFO_SIZE as usize];
    b[0..8].copy_from_slice(&this_size.to_le_bytes());
    b[8] = 1;
    b[16..24].copy_from_slice(&pic.pts.to_le_bytes());
    b[24..32].copy_from_slice(&pic.dts.to_le_bytes());
    b[32..40].copy_from_slice(&pic.attached.to_le_bytes());
    b[40] = pic.key_frame as u8;
    b[41] = pic.profile as u8;
    b[42] = pic.level as u8;
    b[44..48].copy_from_slice(&(pic.width.div_ceil(16).max(1) - 1).to_le_bytes());
    b[48..52].copy_from_slice(&(pic.height.div_ceil(16).max(1) - 1).to_le_bytes());
    b[52] = 1; // frame_mbs_only_flag
    b[53] = pic.crop.iter().any(|&c| c != 0) as u8;
    for (i, c) in pic.crop.iter().enumerate() {
        b[56 + i * 4..60 + i * 4].copy_from_slice(&c.to_le_bytes());
    }
    let sar_present = pic.sar.0 != 0 && pic.sar.1 != 0;
    b[72] = sar_present as u8;
    b[73] = if sar_present { 255 } else { 0 };
    b[74..76].copy_from_slice(&(pic.sar.0.min(65535) as u16).to_le_bytes());
    b[76..78].copy_from_slice(&(pic.sar.1.min(65535) as u16).to_le_bytes());
    b[78] = 1; // video_signal_type_present_flag
    b[79] = 5; // unspecified video format
    b[80] = (pic.color_range == 2) as u8;
    b[81] = (pic.color_primaries != 0 || pic.color_trc != 0 || pic.color_space != 0) as u8;
    b[82] = pic.color_primaries;
    b[83] = pic.color_trc;
    b[84] = pic.color_space;
    b
}

/// sceVideodec2GetPictureInfo(const OutputInfo*, void* first, void* second)
fn cb_get_picture_info(ctx: &mut CpuContext) -> HleResult {
    let (out, first, second) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if out == 0 || first == 0 {
        return err(ERROR_ARGUMENT_POINTER);
    }
    if !output_size_valid(ctx.read_u64(out).unwrap_or(0)) {
        return err(ERROR_STRUCT_SIZE);
    }
    let valid = u8_at(ctx, out + 8) != 0;
    let count = u8_at(ctx, out + 10);
    let codec_type = ctx.read_u32(out + 12).unwrap_or(0);
    let fb = ctx.read_u64(out + 32).unwrap_or(0);
    if !valid || count == 0 || fb == 0 {
        return err(ERROR_OUTPUT_INFO);
    }
    let Some((_, pic, codec, vp9_info)) = with_state(|s| s.pictures.get(&fb).copied()) else { return err(ERROR_OUTPUT_INFO) };
    if codec != codec_type {
        return err(ERROR_OUTPUT_INFO);
    }
    let fill_common = |ctx: &CpuContext, dst: u64, valid: bool| -> HleResult {
        let size = ctx.read_u64(dst).unwrap_or(0);
        if !(40..=256).contains(&size) {
            return err(ERROR_STRUCT_SIZE);
        }
        let mut b = vec![0u8; size as usize - 8];
        b[0] = valid as u8;
        if valid {
            b[8..16].copy_from_slice(&pic.pts.to_le_bytes());
            b[16..24].copy_from_slice(&pic.dts.to_le_bytes());
            b[24..32].copy_from_slice(&pic.attached.to_le_bytes());
        }
        ctx.write_bytes(dst + 8, &b);
        OK
    };
    if codec_type == CODEC_AVC {
        let size = ctx.read_u64(first).unwrap_or(0);
        if size != AVC_PICTURE_INFO_SIZE && (size | 16) != AVC_PICTURE_INFO_SIZE {
            return err(ERROR_STRUCT_SIZE);
        }
        let b = encode_avc_picture(&pic, size);
        ctx.write_bytes(first, &b[..size as usize]);
    } else if codec_type == CODEC_VP9 {
        // The VP9 picture information (as sceVideodec2GetVp9PictureInfo).
        if ctx.read_u64(first).unwrap_or(0) < vp9::PICTURE_INFO_SIZE {
            return err(ERROR_STRUCT_SIZE);
        }
        let Some(info) = vp9_info else { return err(ERROR_OUTPUT_INFO) };
        if !ctx.write_bytes(first + 8, &info.to_bytes()) {
            return err(ERROR_ARGUMENT_POINTER);
        }
    } else {
        let r = fill_common(ctx, first, true);
        if r != OK {
            return r;
        }
    }
    if second != 0 {
        let r = fill_common(ctx, second, false);
        if r != OK {
            return r;
        }
    }
    OK
}

pub(crate) type Export = (&'static str, &'static str, &'static str, HleCallback);

/// sceVideodec2GetVp9PictureInfo(const OutputInfo*, SceVideodec2Vp9PictureInfo*):
/// the header fields of the VP9 frame shown in the output's frame buffer;
/// not valid for a frame buffer no decoder delivered a picture into.
fn cb_get_vp9_picture_info(ctx: &mut CpuContext) -> HleResult {
    let (out, info) = (ctx.rdi(), ctx.rsi());
    if out == 0 || info == 0 {
        return err(ERROR_ARGUMENT_POINTER);
    }
    if !output_size_valid(ctx.read_u64(out).unwrap_or(0)) || ctx.read_u64(info).unwrap_or(0) < vp9::PICTURE_INFO_SIZE {
        return err(ERROR_STRUCT_SIZE);
    }
    let fb = ctx.read_u64(out + 32).unwrap_or(0);
    let found = with_state(|s| s.pictures.get(&fb).map(|p| (p.2, p.3)));
    let picture = match found {
        None => vp9::Info::default(),
        Some((CODEC_VP9, Some(picture))) => picture,
        Some((CODEC_VP9, None)) => {
            log::error!("[videodec2] VP9 picture at {fb:#x} has no header information");
            return err(ERROR_OUTPUT_INFO);
        }
        Some((codec, _)) => {
            log::error!("[videodec2] VP9 picture information asked of a codec {codec} picture");
            return err(ERROR_OUTPUT_INFO);
        }
    };
    if !ctx.write_bytes(info + 8, &picture.to_bytes()) {
        return err(ERROR_ARGUMENT_POINTER);
    }
    OK
}

pub(crate) const EXPORTS: &[Export] = &[
    ("69fVNDtk2Zk", "sceVideodec2GetVp9PictureInfo", LIB, cb_get_vp9_picture_info),
    ("RnDibcGCPKw", "sceVideodec2QueryComputeMemoryInfo", LIB, cb_query_compute_memory_info),
    ("eD+X2SmxUt4", "sceVideodec2AllocateComputeQueue", LIB, cb_allocate_compute_queue),
    ("UvtA3FAiF4Y", "sceVideodec2ReleaseComputeQueue", LIB, cb_release_compute_queue),
    ("qqMCwlULR+E", "sceVideodec2QueryDecoderMemoryInfo", LIB, cb_query_decoder_memory_info),
    ("CNNRoRYd8XI", "sceVideodec2CreateDecoder", LIB, cb_create_decoder),
    ("jwImxXRGSKA", "sceVideodec2DeleteDecoder", LIB, cb_delete_decoder),
    ("852F5+q6+iM", "sceVideodec2Decode", LIB, cb_decode),
    ("l1hXwscLuCY", "sceVideodec2Flush", LIB, cb_flush),
    ("wJXikG6QFN8", "sceVideodec2Reset", LIB, cb_reset),
    ("NtXRa3dRzU0", "sceVideodec2GetPictureInfo", LIB, cb_get_picture_info),
    ("kjrLbcyhEiw", "sceVideodec2GetAvcPictureInfo", LIB, cb_get_picture_info),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, lib, cb) in EXPORTS {
        reg.register_fn(nid, name, lib, cb);
    }
    vdecsw::register(reg);
}

mod vdecsw;
mod vp9;

#[cfg(test)]
mod clips;
#[cfg(test)]
mod tests;
