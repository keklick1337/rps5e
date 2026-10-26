// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceVdecsw: the queued software decoder interface, on top of the
//! libSceVideodec2 decoders.
//!
//! The compute queue, memory and decoder configuration structures are those
//! of libSceVideodec2 (the configuration is 0x50 bytes; its leading fields
//! are shared). Access units are queued with sceVdecswSetDecodeInput and
//! decoded right away into host memory while fewer than four pictures wait
//! for an output buffer; sceVdecswTrySyncDecodeInput reports each consumed
//! unit and how many pictures it produced. A picture is copied into the
//! frame buffer queued with sceVdecswSetDecodeOutput by
//! sceVdecswTrySyncDecodeOutput. The last picture of a sequence is held
//! until sceVdecswFinalizeDecodeSequence, so that it can be marked as the
//! last frame; after it an output without a picture ends the sequence.

use super::*;
use std::collections::VecDeque;

const LIB: &str = "libSceVdecsw";

/// SceVdecswDecoderConfigInfo.
const CONFIG_INFO_SIZE: u64 = 0x50;
/// SceVdecswInputResult: thisSize, decodedAu, outputFrameCount, reserved.
const INPUT_RESULT_SIZE: u64 = 0x18;
/// SceVdecswFrameBuffer: thisSize, frameBuffer, frameBufferSize.
const VDECSW_FRAME_BUFFER_SIZE: u64 = 0x18;
/// SceVdecswOutputInfo: thisSize, isValid @8, isLastFrame @9, isErrorFrame
/// @0xA, pictureCount @0xB, codecType @0xC, frameWidth @0x10, framePitch
/// @0x14, frameHeight @0x18, isDiscardedFrame @0x1C, frameBuffer @0x20,
/// frameBufferSize @0x28, frameFormat @0x30, framePitchInBytes @0x34.
const VDECSW_OUTPUT_INFO_SIZE: u64 = 0x38;
/// Decoded pictures waiting for an output buffer at most.
const MAX_QUEUED_PICTURES: usize = 4;

const ERROR_OUTPUT_PENDING: u32 = 0x8151_0115;
const ERROR_INPUT_QUEUE_EMPTY: u32 = 0x8151_0116;
const ERROR_DECODE_PENDING: u32 = 0x8151_0117;

struct Input {
    au: u64,
    data: Vec<u8>,
    pts: u64,
    dts: u64,
    attached: u64,
}

#[derive(Default)]
struct Stream {
    inputs: VecDeque<Input>,
    /// Consumed access units and the pictures each produced.
    consumed: VecDeque<(u64, u32)>,
    /// Output frame buffers (address, size).
    outputs: VecDeque<(u64, u64)>,
    pictures: VecDeque<HostPicture>,
    /// VP9 frame information of the queued pictures (in the same order).
    picture_vp9: VecDeque<Option<vp9::Info>>,
    /// Picture metadata by the frame buffer it was delivered in.
    delivered: HashMap<u64, RawPicture>,
    /// VP9 frame information by the frame buffer it was delivered in.
    delivered_vp9: HashMap<u64, vp9::Info>,
    finalized: bool,
    drained: bool,
}

static STREAMS: Mutex<Option<HashMap<u64, Arc<Mutex<Stream>>>>> = Mutex::new(None);

fn with_streams<R>(f: impl FnOnce(&mut HashMap<u64, Arc<Mutex<Stream>>>) -> R) -> R {
    let mut g = STREAMS.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(HashMap::new))
}

fn stream(handle: u64) -> Option<Arc<Mutex<Stream>>> {
    with_streams(|s| s.get(&handle).cloned())
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Decode queued input (or drain a finalized sequence) while fewer than
/// four pictures wait.
fn advance(handle: u64, s: &mut Stream) {
    let Some(d) = decoder(handle) else { return };
    let mut d = lock(&d);
    while s.pictures.len() < MAX_QUEUED_PICTURES {
        let result = if let Some(input) = s.inputs.pop_front() {
            if d.codec_type == CODEC_VP9 {
                if let Err(e) = note_vp9_frames(&mut d, &input.data, input.pts, input.dts, input.attached) {
                    log::warn!("[vdecsw] decoder {handle:#x}: VP9 frame headers not read ({e})");
                }
            }
            let r = decode_to_host(&mut d, &input.data, input.pts, input.dts, input.attached);
            s.consumed.push_back((input.au, matches!(r, Ok(Some(_))) as u32));
            r
        } else if s.finalized && !s.drained {
            let r = flush_to_host(&mut d);
            s.drained = !matches!(r, Ok(Some(_)));
            r
        } else {
            return;
        };
        match result {
            Ok(Some(picture)) => {
                s.pictures.push_back(picture);
                let info = if d.codec_type == CODEC_VP9 { d.vp9_pending.pop_front() } else { None };
                s.picture_vp9.push_back(info);
            }
            Ok(None) => {}
            Err(e) => log::warn!("[vdecsw] decoder {handle:#x}: decode failed ({:#x})", e as u32),
        }
    }
}

/// sceVdecswQueryDecoderMemoryInfo(const SceVdecswDecoderConfigInfo*, SceVdecswDecoderMemoryInfo*)
fn cb_query_decoder_memory_info(ctx: &mut CpuContext) -> HleResult {
    query_decoder_memory_info(ctx, ctx.rdi(), ctx.rsi(), CONFIG_INFO_SIZE)
}

/// sceVdecswCreateDecoder(const ConfigInfo*, const DecoderMemoryInfo*, Decoder*)
fn cb_create_decoder(ctx: &mut CpuContext) -> HleResult {
    let (config, memory, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if memory == 0 || out == 0 {
        return err(ERROR_ARGUMENT_POINTER);
    }
    if ctx.read_u64(memory + 56).unwrap_or(0) == 0 {
        return err(ERROR_MEMORY_SIZE);
    }
    let r = create_decoder(ctx, config, memory, out, CONFIG_INFO_SIZE);
    if r != OK {
        return r;
    }
    let handle = ctx.read_u64(out).unwrap_or(0);
    with_streams(|s| s.insert(handle, Arc::new(Mutex::new(Stream::default()))));
    OK
}

/// sceVdecswDeleteDecoder(Decoder)
fn cb_delete_decoder(ctx: &mut CpuContext) -> HleResult {
    if with_streams(|s| s.remove(&ctx.rdi())).is_none() {
        return err(ERROR_DECODER_INSTANCE);
    }
    super::cb_delete_decoder(ctx)
}

/// sceVdecswSetDecodeInput(Decoder, const SceVdecswInputData*): the access
/// unit is copied, so the caller may reuse its memory at once.
fn cb_set_decode_input(ctx: &mut CpuContext) -> HleResult {
    let (handle, input) = (ctx.rdi(), ctx.rsi());
    if input == 0 {
        return err(ERROR_ARGUMENT_POINTER);
    }
    if ctx.read_u64(input).unwrap_or(0) < INPUT_DATA_SIZE {
        return err(ERROR_STRUCT_SIZE);
    }
    let Some(st) = stream(handle) else { return err(ERROR_DECODER_INSTANCE) };
    let mut s = lock(&st);
    if s.finalized {
        log::warn!("[vdecsw] decoder {handle:#x}: input after the sequence was finalized");
        return err(ERROR_API_FAIL);
    }
    let au = ctx.read_u64(input + 8).unwrap_or(0);
    let size = ctx.read_u64(input + 16).unwrap_or(0);
    if size == 0 || size > i32::MAX as u64 {
        return err(ERROR_ACCESS_UNIT_SIZE);
    }
    let Some(data) = (au != 0).then(|| ctx.read_bytes(au, size as usize)).flatten() else {
        return err(ERROR_ACCESS_UNIT_POINTER);
    };
    let field = |o: u64| ctx.read_u64(input + o).unwrap_or(0);
    s.inputs.push_back(Input { au, data, pts: field(24), dts: field(32), attached: field(40) });
    advance(handle, &mut s);
    OK
}

/// sceVdecswTrySyncDecodeInput(Decoder, SceVdecswInputResult*): the next
/// consumed access unit, INPUT_QUEUE_EMPTY when nothing was queued, or
/// DECODE_PENDING while queued units wait for picture space.
fn cb_try_sync_decode_input(ctx: &mut CpuContext) -> HleResult {
    let (handle, result) = (ctx.rdi(), ctx.rsi());
    if result == 0 {
        return err(ERROR_ARGUMENT_POINTER);
    }
    if ctx.read_u64(result).unwrap_or(0) < INPUT_RESULT_SIZE {
        return err(ERROR_STRUCT_SIZE);
    }
    let Some(st) = stream(handle) else { return err(ERROR_DECODER_INSTANCE) };
    let mut s = lock(&st);
    advance(handle, &mut s);
    let (au, frames) = s.consumed.front().copied().unwrap_or((0, 0));
    ctx.write_u64(result + 8, au);
    ctx.write_u32(result + 16, frames);
    ctx.write_u32(result + 20, 0);
    if s.consumed.pop_front().is_some() {
        OK
    } else if s.inputs.is_empty() {
        err(ERROR_INPUT_QUEUE_EMPTY)
    } else {
        err(ERROR_DECODE_PENDING)
    }
}

/// Poll period of the blocking syncs.
const SYNC_POLL: std::time::Duration = std::time::Duration::from_micros(500);

/// sceVdecswSyncDecodeInput(Decoder, SceVdecswInputResult*): as the Try
/// variant, waiting while queued units wait for picture space.
fn cb_sync_decode_input(ctx: &mut CpuContext) -> HleResult {
    let (handle, result) = (ctx.rdi(), ctx.rsi());
    loop {
        ctx.set(crate::context::CpuRegister::Rdi, handle);
        ctx.set(crate::context::CpuRegister::Rsi, result);
        let r = cb_try_sync_decode_input(ctx);
        if r != err(ERROR_DECODE_PENDING) {
            return r;
        }
        crate::signals::sleep(SYNC_POLL);
    }
}

/// sceVdecswSyncDecodeOutput(Decoder, SceVdecswOutputInfo*): as the Try
/// variant, waiting for the next picture. Without a queued frame buffer
/// nothing could end the wait: that is not implemented (OUTPUT_PENDING and
/// a log line).
fn cb_sync_decode_output(ctx: &mut CpuContext) -> HleResult {
    let (handle, out) = (ctx.rdi(), ctx.rsi());
    loop {
        if stream(handle).is_some_and(|st| lock(&st).outputs.is_empty()) {
            log::error!("[vdecsw] decoder {handle:#x}: output sync without an output buffer is not implemented");
            return err(ERROR_OUTPUT_PENDING);
        }
        ctx.set(crate::context::CpuRegister::Rdi, handle);
        ctx.set(crate::context::CpuRegister::Rsi, out);
        let r = cb_try_sync_decode_output(ctx);
        if r != err(ERROR_OUTPUT_PENDING) {
            return r;
        }
        crate::signals::sleep(SYNC_POLL);
    }
}

/// sceVdecswSetDecodeOutput(Decoder, const SceVdecswFrameBuffer*): queue a
/// frame buffer for the next picture.
fn cb_set_decode_output(ctx: &mut CpuContext) -> HleResult {
    let (handle, fb) = (ctx.rdi(), ctx.rsi());
    if fb == 0 {
        return err(ERROR_ARGUMENT_POINTER);
    }
    if ctx.read_u64(fb).unwrap_or(0) < VDECSW_FRAME_BUFFER_SIZE {
        return err(ERROR_STRUCT_SIZE);
    }
    let (data, size) = (ctx.read_u64(fb + 8).unwrap_or(0), ctx.read_u64(fb + 16).unwrap_or(0));
    if data == 0 {
        return err(ERROR_FRAME_BUFFER_POINTER);
    }
    if size == 0 {
        return err(ERROR_FRAME_BUFFER_SIZE);
    }
    let Some(st) = stream(handle) else { return err(ERROR_DECODER_INSTANCE) };
    lock(&st).outputs.push_back((data, size));
    OK
}

/// sceVdecswTrySyncDecodeOutput(Decoder, SceVdecswOutputInfo*)
fn cb_try_sync_decode_output(ctx: &mut CpuContext) -> HleResult {
    let (handle, out) = (ctx.rdi(), ctx.rsi());
    if out == 0 {
        return err(ERROR_ARGUMENT_POINTER);
    }
    if ctx.read_u64(out).unwrap_or(0) < VDECSW_OUTPUT_INFO_SIZE {
        return err(ERROR_STRUCT_SIZE);
    }
    let Some(st) = stream(handle) else { return err(ERROR_DECODER_INSTANCE) };
    let Some(codec_type) = decoder(handle).map(|d| lock(&d).codec_type) else { return err(ERROR_DECODER_INSTANCE) };
    let mut s = lock(&st);
    advance(handle, &mut s);
    // No frame buffer to deliver into: nothing can be output yet.
    let Some(&(target, target_size)) = s.outputs.front() else { return err(ERROR_OUTPUT_PENDING) };
    let ended = s.finalized && s.drained;
    if s.pictures.len() < if ended { 1 } else { 2 } {
        if !ended {
            return err(ERROR_OUTPUT_PENDING);
        }
        let mut b = [0u8; VDECSW_OUTPUT_INFO_SIZE as usize - 8];
        b[1] = 1; // isLastFrame
        ctx.write_bytes(out + 8, &b);
        s.outputs.pop_front();
        return OK;
    }
    let picture = s.pictures.pop_front().expect("a queued picture");
    let picture_vp9 = s.picture_vp9.pop_front().flatten();
    let size = picture.pixels.len() as u64;
    if target_size < size || !ctx.write_bytes(target, &picture.pixels) {
        let e = if target_size < size { ERROR_FRAME_BUFFER_SIZE } else { ERROR_FRAME_BUFFER_POINTER };
        s.pictures.push_front(picture);
        s.picture_vp9.push_front(picture_vp9);
        return err(e);
    }
    s.outputs.pop_front();
    let pic = picture.pic;
    let last = ended && s.pictures.is_empty();
    let mut b = [0u8; VDECSW_OUTPUT_INFO_SIZE as usize - 8];
    b[0] = 1;
    b[1] = last as u8;
    b[2] = pic.corrupt as u8;
    b[3] = 1; // pictureCount
    b[4..8].copy_from_slice(&codec_type.to_le_bytes());
    b[8..12].copy_from_slice(&pic.width.to_le_bytes());
    b[12..16].copy_from_slice(&pic.pitch.to_le_bytes());
    b[16..20].copy_from_slice(&pic.height.to_le_bytes());
    b[24..32].copy_from_slice(&target.to_le_bytes());
    b[32..40].copy_from_slice(&target_size.to_le_bytes());
    b[44..48].copy_from_slice(&pic.pitch.to_le_bytes());
    ctx.write_bytes(out + 8, &b);
    s.delivered.insert(target, pic);
    match picture_vp9 {
        Some(info) => {
            s.delivered_vp9.insert(target, info);
        }
        None => {
            s.delivered_vp9.remove(&target);
        }
    }
    OK
}

/// sceVdecswGetAvcPictureInfo(const SceVdecswOutputInfo*, SceVdecswAvcPictureInfo*
/// first, second): the picture info of a delivered frame. Only frame
/// pictures are produced, so the second field is never filled.
fn cb_get_avc_picture_info(ctx: &mut CpuContext) -> HleResult {
    let (out, first) = (ctx.rdi(), ctx.rsi());
    if out == 0 || first == 0 {
        return err(ERROR_ARGUMENT_POINTER);
    }
    let size = ctx.read_u64(first).unwrap_or(0);
    if size < 8 {
        return err(ERROR_STRUCT_SIZE);
    }
    let count = u8_at(ctx, out + 11);
    if count == 0 {
        return OK;
    }
    if count > 1 {
        return err(ERROR_OUTPUT_INFO);
    }
    let fb = ctx.read_u64(out + 32).unwrap_or(0);
    let streams: Vec<_> = with_streams(|s| s.values().cloned().collect());
    let Some(pic) = streams.iter().find_map(|st| lock(st).delivered.get(&fb).copied()) else {
        return err(ERROR_OUTPUT_INFO);
    };
    let b = encode_avc_picture(&pic, AVC_PICTURE_INFO_SIZE);
    let n = size.min(AVC_PICTURE_INFO_SIZE) as usize;
    ctx.write_bytes(first + 8, &b[8..n]);
    OK
}

/// sceVdecswGetVp9PictureInfo(const SceVdecswOutputInfo*,
/// SceVdecswVp9PictureInfo*): the frame header information of the VP9
/// picture delivered in the output's frame buffer, cut to the caller's
/// size; nothing for an output without a picture.
fn cb_get_vp9_picture_info(ctx: &mut CpuContext) -> HleResult {
    let (out, info) = (ctx.rdi(), ctx.rsi());
    if out == 0 || info == 0 {
        return err(ERROR_ARGUMENT_POINTER);
    }
    let size = ctx.read_u64(info).unwrap_or(0);
    if size < 8 {
        return err(ERROR_STRUCT_SIZE);
    }
    let count = u8_at(ctx, out + 11);
    if count == 0 {
        return OK;
    }
    if count > 1 {
        log::error!("[vdecsw] sceVdecswGetVp9PictureInfo: more than one picture is not supported");
        return NOT_IMPLEMENTED;
    }
    let fb = ctx.read_u64(out + 32).unwrap_or(0);
    let streams: Vec<_> = with_streams(|s| s.values().cloned().collect());
    let Some(picture) = streams.iter().find_map(|st| lock(st).delivered_vp9.get(&fb).copied()) else {
        return err(ERROR_OUTPUT_INFO);
    };
    let n = size.min(vp9::PICTURE_INFO_SIZE) as usize - 8;
    if !ctx.write_bytes(info + 8, &picture.to_bytes()[..n]) {
        return err(ERROR_ARGUMENT_POINTER);
    }
    OK
}

/// sceVdecswGetPictureInfo(const SceVdecswOutputInfo*, void* first, void*
/// second): the codec's picture info (AVC or VP9) of the picture delivered
/// in the output's frame buffer, cut to the size in the first word of
/// `first`, whose size word is kept. The output descriptor is 0x30 or 0x38
/// bytes. An output without a picture, or a null `first`, reports nothing;
/// `second` (the second field of a field pair) is never written, as only
/// frame pictures are produced.
fn cb_get_picture_info(ctx: &mut CpuContext) -> HleResult {
    let (out, first) = (ctx.rdi(), ctx.rsi());
    if out == 0 {
        return err(ERROR_ARGUMENT_POINTER);
    }
    if ctx.read_u64(out).unwrap_or(0) | 8 != VDECSW_OUTPUT_INFO_SIZE {
        return err(ERROR_STRUCT_SIZE);
    }
    let count = u8_at(ctx, out + 11);
    if count == 0 {
        return OK;
    }
    if count > 1 {
        log::error!("[vdecsw] sceVdecswGetPictureInfo: more than one picture is not supported");
        return NOT_IMPLEMENTED;
    }
    if first == 0 {
        return OK;
    }
    let size = ctx.read_u64(first).unwrap_or(0);
    if size < 8 {
        return err(ERROR_STRUCT_SIZE);
    }
    let fb = ctx.read_u64(out + 32).unwrap_or(0);
    let streams: Vec<_> = with_streams(|s| s.values().cloned().collect());
    let bytes = streams.iter().find_map(|st| {
        let s = lock(st);
        if let Some(pic) = s.delivered.get(&fb) {
            return Some(encode_avc_picture(pic, AVC_PICTURE_INFO_SIZE).to_vec());
        }
        s.delivered_vp9.get(&fb).map(|info| [&[0u8; 8][..], &info.to_bytes()[..]].concat())
    });
    let Some(bytes) = bytes else { return err(ERROR_OUTPUT_INFO) };
    let n = (size.min(bytes.len() as u64) as usize).max(8);
    if !ctx.write_bytes(first + 8, &bytes[8..n]) {
        return err(ERROR_ARGUMENT_POINTER);
    }
    OK
}

/// Answer for the cases the console's answer to is not known.
const NOT_IMPLEMENTED: HleResult = 0x8002_0002u32 as i32 as i64;

/// sceVdecswFinalizeDecodeSequence(Decoder): no more input; the remaining
/// pictures are drained.
fn cb_finalize_decode_sequence(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    let Some(st) = stream(handle) else { return err(ERROR_DECODER_INSTANCE) };
    let mut s = lock(&st);
    s.finalized = true;
    advance(handle, &mut s);
    OK
}

/// sceVdecswResetDecoder(Decoder): drops queued input, pictures and output
/// buffers and starts a new sequence.
fn cb_reset_decoder(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    let Some(st) = stream(handle) else { return err(ERROR_DECODER_INSTANCE) };
    *lock(&st) = Stream::default();
    super::cb_reset(ctx)
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("0moTubWCsTM", "sceVdecswQueryComputeMemoryInfo", super::cb_query_compute_memory_info),
    ("hIgrg5h4V6s", "sceVdecswAllocateComputeQueue", super::cb_allocate_compute_queue),
    ("fX-zOOefbbs", "sceVdecswReleaseComputeQueue", super::cb_release_compute_queue),
    ("A+2M7EivuOU", "sceVdecswQueryDecoderMemoryInfo", cb_query_decoder_memory_info),
    ("+L5ArV1tPGA", "sceVdecswCreateDecoder", cb_create_decoder),
    ("ecUtPX+dBYk", "sceVdecswDeleteDecoder", cb_delete_decoder),
    ("aqMiF0AgUYI", "sceVdecswSetDecodeInput", cb_set_decode_input),
    ("l4sQYy5wPkc", "sceVdecswTrySyncDecodeInput", cb_try_sync_decode_input),
    ("AAMM-Q1X0g0", "sceVdecswSyncDecodeInput", cb_sync_decode_input),
    ("tWiSgXov8GM", "sceVdecswSyncDecodeOutput", cb_sync_decode_output),
    ("rgtMCOpyBSc", "sceVdecswSetDecodeOutput", cb_set_decode_output),
    ("kMBw37oH8nI", "sceVdecswTrySyncDecodeOutput", cb_try_sync_decode_output),
    ("ihNT-uuEAr4", "sceVdecswGetAvcPictureInfo", cb_get_avc_picture_info),
    ("LEqH+dkMszo", "sceVdecswGetVp9PictureInfo", cb_get_vp9_picture_info),
    ("FzECy3Wxxas", "sceVdecswGetPictureInfo", cb_get_picture_info),
    ("5Y6nZqIZvBg", "sceVdecswFinalizeDecodeSequence", cb_finalize_decode_sequence),
    ("veb-YBrOqo0", "sceVdecswResetDecoder", cb_reset_decoder),
];

pub(super) fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::super::clips::AVC_HASHES;
    use super::super::tests::{avc_units, LOCK};
    use super::*;
    use crate::avplayer::{arena_alloc, arena_free, host};
    use crate::context::CpuRegister;

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 40;
    const FRAME_BYTES: u64 = 1 << 20;
    /// Display order of the clip's access units.
    const DISPLAY_ORDER: [u64; 6] = [0, 2, 3, 1, 5, 4];

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    struct Session {
        vm: ps5vm::VirtualMemory,
        scratch: u64,
        au: u64,
        frames: [u64; 2],
        handle: u64,
        queue: u64,
    }

    impl Session {
        fn call(&self, f: HleCallback, args: &[u64]) -> HleResult {
            let mut ctx = CpuContext::new(&self.vm);
            const REGS: [CpuRegister; 3] = [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx];
            for (r, v) in REGS.iter().zip(args) {
                ctx.set(*r, *v);
            }
            f(&mut ctx)
        }

        fn ctx(&self) -> CpuContext<'_> {
            CpuContext::new(&self.vm)
        }

        /// Compute queue and a 64x40 H.264 decoder, as a title sets them up.
        fn new() -> Self {
            let mut s = Self {
                vm: ps5vm::VirtualMemory::new(),
                scratch: arena_alloc(0x1000).unwrap(),
                au: arena_alloc(0x10000).unwrap(),
                frames: [arena_alloc(FRAME_BYTES).unwrap(), arena_alloc(FRAME_BYTES).unwrap()],
                handle: 0,
                queue: 0,
            };
            (s.queue, s.handle) = s.setup();
            s
        }

        /// (compute queue, decoder)
        fn setup(&self) -> (u64, u64) {
            let s = self;
            let ctx = s.ctx();
            let (compute, compute_config, queue) = (s.scratch, s.scratch + 0x20, s.scratch + 0x40);
            let (config, memory, handle) = (s.scratch + 0x100, s.scratch + 0x180, s.scratch + 0x1E0);
            ctx.write_bytes(s.scratch, &[0u8; 0x400]);
            ctx.write_u64(compute, COMPUTE_MEMORY_INFO_SIZE);
            assert_eq!(s.call(super::super::cb_query_compute_memory_info, &[compute]), OK);
            assert_ne!(ctx.read_u64(compute + 8), Some(0));
            ctx.write_u64(compute + 16, 0x1000);
            ctx.write_u64(compute_config, COMPUTE_CONFIG_INFO_SIZE);
            ctx.write_bytes(compute_config + 8, &[3, 0, 3, 0, 1]);
            assert_eq!(s.call(super::super::cb_allocate_compute_queue, &[compute_config, compute, queue]), OK);
            let queue = ctx.read_u64(queue).unwrap();
            ctx.write_u64(config, CONFIG_INFO_SIZE);
            for (off, v) in [(8, 1), (12, CODEC_AVC), (16, 100), (20, 42), (24, WIDTH), (28, HEIGHT), (32, 4), (36, 3)] {
                ctx.write_u32(config + off, v);
            }
            ctx.write_u64(config + 40, queue);
            ctx.write_u64(config + 48, 0x3F);
            ctx.write_bytes(config + 60, &[1]);
            ctx.write_u64(memory, DECODER_MEMORY_INFO_SIZE);
            assert_eq!(s.call(cb_query_decoder_memory_info, &[config, memory]), OK);
            assert_ne!(ctx.read_u64(memory + 56), Some(0));
            for off in [16, 32, 48] {
                ctx.write_u64(memory + off, 0x1000);
            }
            assert_eq!(s.call(cb_create_decoder, &[config, memory, handle]), OK);
            (queue, ctx.read_u64(handle).unwrap())
        }

        fn input(&self, unit: &[u8], n: u64) -> HleResult {
            let ctx = self.ctx();
            ctx.write_bytes(self.au, unit);
            let input = self.scratch + 0x200;
            for (off, v) in [(0, INPUT_DATA_SIZE), (8, self.au), (16, unit.len() as u64), (24, 1000 + n), (32, n), (40, 0xA0 + n)] {
                ctx.write_u64(input + off, v);
            }
            self.call(cb_set_decode_input, &[self.handle, input])
        }

        /// (status, decodedAu, outputFrameCount)
        fn sync_input(&self) -> (HleResult, u64, u32) {
            let ctx = self.ctx();
            let result = self.scratch + 0x240;
            ctx.write_bytes(result, &[0xEE; 0x18]);
            ctx.write_u64(result, INPUT_RESULT_SIZE);
            let r = self.call(cb_try_sync_decode_input, &[self.handle, result]);
            (r, ctx.read_u64(result + 8).unwrap(), ctx.read_u32(result + 16).unwrap())
        }

        fn set_output(&self, frame: u64) -> HleResult {
            let ctx = self.ctx();
            let fb = self.scratch + 0x260;
            for (off, v) in [(0, VDECSW_FRAME_BUFFER_SIZE), (8, frame), (16, FRAME_BYTES)] {
                ctx.write_u64(fb + off, v);
            }
            self.call(cb_set_decode_output, &[self.handle, fb])
        }

        fn sync_output(&self) -> (HleResult, u64) {
            let ctx = self.ctx();
            let out = self.scratch + 0x280;
            ctx.write_bytes(out, &[0u8; VDECSW_OUTPUT_INFO_SIZE as usize]);
            ctx.write_u64(out, VDECSW_OUTPUT_INFO_SIZE);
            (self.call(cb_try_sync_decode_output, &[self.handle, out]), out)
        }

        /// FNV-1a over the visible NV12 rows of `frame`.
        fn hash_nv12(&self, frame: u64, pitch: u64) -> u64 {
            let ctx = self.ctx();
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            let rows = (0..HEIGHT as u64).map(|r| r * pitch).chain((0..HEIGHT as u64 / 2).map(|r| (HEIGHT as u64 + r) * pitch));
            for row in rows {
                for b in ctx.read_bytes(frame + row, WIDTH as usize).unwrap() {
                    hash = (hash ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            hash
        }
    }

    impl Drop for Session {
        fn drop(&mut self) {
            let _ = self.call(cb_delete_decoder, &[self.handle]);
            let _ = self.call(super::super::cb_release_compute_queue, &[self.queue]);
            arena_free(self.scratch);
            arena_free(self.au);
            self.frames.iter().for_each(|&f| arena_free(f));
        }
    }

    /// The clip through the queues: every unit is consumed at once, pictures
    /// come in display order with their timestamps and reference pixels, and
    /// only the last picture of the finalized sequence is marked last.
    #[test]
    fn decodes_avc_clip_through_the_queues() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if !host::available() {
            return;
        }
        let s = Session::new();
        let units = avc_units();
        assert_eq!(units.len(), AVC_HASHES.len());
        assert_eq!(s.sync_input().0, err(ERROR_INPUT_QUEUE_EMPTY), "idle decoder");
        let mut pictures = 0usize;
        let mut output_set = false;
        let mut ended = false;
        let mut drain = |pictures: &mut usize, ended: &mut bool| loop {
            let frame = s.frames[*pictures % 2];
            if !output_set {
                assert_eq!(s.set_output(frame), OK);
                output_set = true;
            }
            let (r, out) = s.sync_output();
            if r == err(ERROR_OUTPUT_PENDING) {
                return;
            }
            assert_eq!(r, OK, "output sync failed");
            output_set = false;
            let ctx = s.ctx();
            let b = ctx.read_bytes(out, VDECSW_OUTPUT_INFO_SIZE as usize).unwrap();
            let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
            if b[8] == 0 {
                assert_eq!((b[9], b[11]), (1, 0), "an output without a picture ends the sequence");
                *ended = true;
                return;
            }
            assert!(*pictures < AVC_HASHES.len(), "more pictures than access units");
            assert_eq!(b[9] != 0, *pictures + 1 == AVC_HASHES.len(), "only the last picture is marked last");
            assert_eq!((b[10], b[11], b[0x1C]), (0, 1, 0), "error / count / discarded");
            assert_eq!((u32_at(0x10), u32_at(0x18)), (WIDTH, HEIGHT));
            assert_eq!(ctx.read_u64(out + 0x20), Some(frame));
            let info = s.scratch + 0x300;
            ctx.write_bytes(info, &[0u8; AVC_PICTURE_INFO_SIZE as usize]);
            ctx.write_u64(info, AVC_PICTURE_INFO_SIZE);
            assert_eq!(s.call(cb_get_avc_picture_info, &[out, info, 0]), OK);
            let p = ctx.read_bytes(info, AVC_PICTURE_INFO_SIZE as usize).unwrap();
            let u64_at = |o: usize| u64::from_le_bytes(p[o..o + 8].try_into().unwrap());
            let unit = DISPLAY_ORDER[*pictures];
            assert_eq!(p[8], 1, "picture info missing");
            assert_eq!((u64_at(16), u64_at(24), u64_at(32)), (1000 + unit, unit, 0xA0 + unit), "display order");
            assert_eq!((p[40], p[41]), ((*pictures == 0) as u8, 100), "IDR flag and profile");
            assert_eq!(s.hash_nv12(frame, u32_at(0x14) as u64), AVC_HASHES[*pictures], "picture {pictures}");
            // The generic query copies the same record, cut to its size
            // word, which it keeps; the second output stays untouched.
            let (generic, second) = (s.scratch + 0x400, s.scratch + 0x500);
            for size in [8u64, 9, 0x28, AVC_PICTURE_INFO_SIZE - 1, AVC_PICTURE_INFO_SIZE, AVC_PICTURE_INFO_SIZE + 16, u64::MAX] {
                ctx.write_bytes(generic, &[0xCCu8; AVC_PICTURE_INFO_SIZE as usize + 32]);
                ctx.write_u64(generic, size);
                ctx.write_bytes(second, &[0xCCu8; 16]);
                assert_eq!(s.call(cb_get_picture_info, &[out, generic, second]), OK);
                let g = ctx.read_bytes(generic, AVC_PICTURE_INFO_SIZE as usize + 32).unwrap();
                let copied = size.min(AVC_PICTURE_INFO_SIZE) as usize;
                assert_eq!(g[..8], size.to_le_bytes(), "size word changed");
                assert_eq!(g[8..copied], p[8..copied], "generic picture info of size {size:#x}");
                assert!(g[copied.max(8)..].iter().all(|&b| b == 0xCC), "generic picture info wrote past size {size:#x}");
                assert_eq!(ctx.read_bytes(second, 16).unwrap(), vec![0xCC; 16], "second picture written");
            }
            assert_eq!(s.call(cb_get_picture_info, &[out, 0, second]), OK, "a null first output is nothing to fill");
            *pictures += 1;
        };
        let mut produced = 0;
        for (n, unit) in units.iter().enumerate() {
            assert_eq!(s.input(unit, n as u64), OK);
            let (r, au, frames) = s.sync_input();
            assert_eq!((r, au), (OK, s.au), "access unit {n} not consumed");
            produced += frames;
            drain(&mut pictures, &mut ended);
        }
        assert!(pictures < AVC_HASHES.len(), "the last picture came before the sequence was finalized");
        assert_eq!(s.call(cb_finalize_decode_sequence, &[s.handle]), OK);
        while !ended {
            drain(&mut pictures, &mut ended);
        }
        assert_eq!(pictures, AVC_HASHES.len());
        assert!(produced as usize <= AVC_HASHES.len());
        assert_eq!(s.sync_input().0, err(ERROR_INPUT_QUEUE_EMPTY), "finalized decoder");
        assert_eq!(s.input(&units[0], 0), err(ERROR_API_FAIL), "input after finalize");
    }

    /// An input is decoded before any output buffer exists; the only picture
    /// of an open sequence is held until the sequence is finalized.
    #[test]
    fn single_picture_waits_for_finalize() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if !host::available() {
            return;
        }
        let s = Session::new();
        let units = avc_units();
        // The blocking input sync on an idle decoder returns at once.
        let result = s.scratch + 0x240;
        s.ctx().write_u64(result, INPUT_RESULT_SIZE);
        assert_eq!(s.call(cb_sync_decode_input, &[s.handle, result]), err(ERROR_INPUT_QUEUE_EMPTY));
        assert_eq!(s.input(&units[0], 0), OK);
        assert_eq!(s.call(cb_sync_decode_input, &[s.handle, result]), OK);
        assert_eq!(s.ctx().read_u64(result + 8), Some(s.au));
        assert_eq!(s.sync_output().0, err(ERROR_OUTPUT_PENDING), "no output buffer");
        let out = s.scratch + 0x280;
        assert_eq!(s.call(cb_sync_decode_output, &[s.handle, out]), err(ERROR_OUTPUT_PENDING), "blocking sync without a buffer");
        assert_eq!(s.set_output(s.frames[0]), OK);
        assert_eq!(s.sync_output().0, err(ERROR_OUTPUT_PENDING), "open sequence");
        assert_eq!(s.call(cb_finalize_decode_sequence, &[s.handle]), OK);
        s.ctx().write_bytes(out, &[0u8; VDECSW_OUTPUT_INFO_SIZE as usize]);
        s.ctx().write_u64(out, VDECSW_OUTPUT_INFO_SIZE);
        assert_eq!(s.call(cb_sync_decode_output, &[s.handle, out]), OK);
        let b = s.ctx().read_bytes(out + 8, 2).unwrap();
        assert_eq!(b, vec![1, 1], "the only picture of a finalized sequence is valid and last");
        assert_eq!(s.call(cb_reset_decoder, &[s.handle]), OK);
        assert_eq!(s.sync_input().0, err(ERROR_INPUT_QUEUE_EMPTY), "reset dropped the inputs");
        assert_eq!(s.input(&units[0], 0), OK, "a reset decoder takes input again");
    }

    /// The generic query copies the AVC or VP9 record of the picture
    /// delivered in the output's frame buffer, cut to the size word, which
    /// it keeps; the second output and the bytes past the size stay as
    /// they were.
    #[test]
    fn generic_picture_info_copies_delivered_records() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5F23_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (avc_fb, vp9_fb, handle) = (0x4400_0000u64, 0x4500_0000u64, 0xD0D0_0001u64);
        let pic = RawPicture { width: 64, height: 40, profile: 100, level: 31, key_frame: true, pts: 1234, dts: 5, attached: 0xA7, ..Default::default() };
        let info = vp9::Info { valid: true, pts: 77, dts: 6, attached: 0xB1, profile: 1, key_frame: true, bit_depth: 8, frame_width: 64, frame_height: 40, ..Default::default() };
        let mut st = Stream::default();
        st.delivered.insert(avc_fb, pic);
        st.delivered_vp9.insert(vp9_fb, info);
        with_streams(|s| s.insert(handle, Arc::new(Mutex::new(st))));
        let (out, first, second) = (BASE, BASE + 0x100, BASE + 0x300);
        vm.write(out, &[0u8; VDECSW_OUTPUT_INFO_SIZE as usize]).unwrap();
        vm.write(out, &VDECSW_OUTPUT_INFO_SIZE.to_le_bytes()).unwrap();
        vm.write(out + 11, &[1]).unwrap();
        let avc = encode_avc_picture(&pic, AVC_PICTURE_INFO_SIZE).to_vec();
        let vp9 = [&[0u8; 8][..], &info.to_bytes()[..]].concat();
        for (fb, record) in [(avc_fb, &avc), (vp9_fb, &vp9)] {
            vm.write(out + 32, &fb.to_le_bytes()).unwrap();
            let len = record.len() as u64;
            for size in [8u64, 9, 0x28, len - 1, len, len + 16, u64::MAX] {
                vm.write(first, &[0xCCu8; 0x180]).unwrap();
                vm.write(first, &size.to_le_bytes()).unwrap();
                vm.write(second, &[0xCCu8; 16]).unwrap();
                ctx.set(CpuRegister::Rdi, out);
                ctx.set(CpuRegister::Rsi, first);
                ctx.set(CpuRegister::Rdx, second);
                assert_eq!(cb_get_picture_info(&mut ctx), OK, "size {size:#x}");
                let g = ctx.read_bytes(first, 0x180).unwrap();
                let copied = size.min(len) as usize;
                assert_eq!(g[..8], size.to_le_bytes(), "size word changed");
                assert_eq!(g[8..copied], record[8..copied], "picture info of size {size:#x}");
                assert!(g[copied.max(8)..].iter().all(|&b| b == 0xCC), "picture info written past size {size:#x}");
                assert_eq!(ctx.read_bytes(second, 16).unwrap(), vec![0xCC; 16], "second picture written");
            }
        }
        with_streams(|s| s.remove(&handle));
        vm.write(first, &AVC_PICTURE_INFO_SIZE.to_le_bytes()).unwrap();
        assert_eq!(cb_get_picture_info(&mut ctx), err(ERROR_OUTPUT_INFO), "a deleted decoder's picture");
    }

    #[test]
    fn argument_checks() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5F22_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let mut call = |f: HleCallback, a: u64, b: u64| {
            ctx.set(CpuRegister::Rdi, a);
            ctx.set(CpuRegister::Rsi, b);
            f(&mut ctx)
        };
        let unknown = 0x7777_0000;
        assert_eq!(call(cb_finalize_decode_sequence, unknown, 0), err(ERROR_DECODER_INSTANCE));
        assert_eq!(call(cb_reset_decoder, unknown, 0), err(ERROR_DECODER_INSTANCE));
        assert_eq!(call(cb_delete_decoder, unknown, 0), err(ERROR_DECODER_INSTANCE));
        assert_eq!(call(cb_set_decode_input, unknown, 0), err(ERROR_ARGUMENT_POINTER));
        assert_eq!(call(cb_try_sync_decode_output, unknown, 0), err(ERROR_ARGUMENT_POINTER));
        vm.write(BASE, &4u64.to_le_bytes()).unwrap();
        assert_eq!(call(cb_try_sync_decode_input, unknown, BASE), err(ERROR_STRUCT_SIZE));
        vm.write(BASE, &INPUT_RESULT_SIZE.to_le_bytes()).unwrap();
        assert_eq!(call(cb_try_sync_decode_input, unknown, BASE), err(ERROR_DECODER_INSTANCE));
        // Picture info: no picture is nothing to report; an unknown frame
        // buffer is an output info error.
        let (out, info) = (BASE + 0x100, BASE + 0x200);
        vm.write(info, &AVC_PICTURE_INFO_SIZE.to_le_bytes()).unwrap();
        vm.write(out + 11, &[0]).unwrap();
        assert_eq!(call(cb_get_avc_picture_info, out, info), OK);
        vm.write(out + 11, &[1]).unwrap();
        vm.write(out + 32, &0x1234_5000u64.to_le_bytes()).unwrap();
        assert_eq!(call(cb_get_avc_picture_info, out, info), err(ERROR_OUTPUT_INFO));
        assert_eq!(call(cb_get_avc_picture_info, out, 0), err(ERROR_ARGUMENT_POINTER));
        // VP9 picture info: the same checks; more than one picture is not
        // supported.
        vm.write(info, &vp9::PICTURE_INFO_SIZE.to_le_bytes()).unwrap();
        assert_eq!(call(cb_get_vp9_picture_info, out, info), err(ERROR_OUTPUT_INFO));
        assert_eq!(call(cb_get_vp9_picture_info, 0, info), err(ERROR_ARGUMENT_POINTER));
        vm.write(out + 11, &[2]).unwrap();
        assert_eq!(call(cb_get_vp9_picture_info, out, info), NOT_IMPLEMENTED);
        vm.write(out + 11, &[0]).unwrap();
        assert_eq!(call(cb_get_vp9_picture_info, out, info), OK);
        vm.write(info, &4u64.to_le_bytes()).unwrap();
        assert_eq!(call(cb_get_vp9_picture_info, out, info), err(ERROR_STRUCT_SIZE));
        // Generic picture info: the output descriptor is checked first (0x30
        // or 0x38 bytes); no picture or no first output is nothing to fill;
        // a size word below 8 or an unknown frame buffer is an error.
        assert_eq!(call(cb_get_picture_info, 0, info), err(ERROR_ARGUMENT_POINTER));
        for size in [0u64, 0x28, 0x31, 0x40, 0x1_0000_0038] {
            vm.write(out, &size.to_le_bytes()).unwrap();
            assert_eq!(call(cb_get_picture_info, out, info), err(ERROR_STRUCT_SIZE), "descriptor size {size:#x}");
        }
        for size in [0x30u64, VDECSW_OUTPUT_INFO_SIZE] {
            vm.write(out, &size.to_le_bytes()).unwrap();
            vm.write(out + 11, &[0]).unwrap();
            assert_eq!(call(cb_get_picture_info, out, info), OK);
            assert_eq!(call(cb_get_picture_info, out, 0), OK);
            vm.write(out + 11, &[1]).unwrap();
            assert_eq!(call(cb_get_picture_info, out, 0), OK);
            vm.write(info, &4u64.to_le_bytes()).unwrap();
            assert_eq!(call(cb_get_picture_info, out, info), err(ERROR_STRUCT_SIZE));
            vm.write(info, &AVC_PICTURE_INFO_SIZE.to_le_bytes()).unwrap();
            assert_eq!(call(cb_get_picture_info, out, info), err(ERROR_OUTPUT_INFO));
            vm.write(out + 11, &[2]).unwrap();
            assert_eq!(call(cb_get_picture_info, out, info), NOT_IMPLEMENTED);
        }
    }
}
