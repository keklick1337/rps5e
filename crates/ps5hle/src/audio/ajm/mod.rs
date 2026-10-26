// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceAjm: codec instances and batch jobs.
//!
//! Jobs run when they are added to the batch (decoded PCM and sideband
//! results are in guest memory when the job builder returns), so
//! `sceAjmBatchStart` / `sceAjmBatchWait` complete at once. Decoders:
//! MP3 (0), ATRAC9 (1), MPEG-4 AAC (2) and Opus (24); the AAC and Opus
//! decoders run on the host FFmpeg shim.
//!
//! Instance ids pack the codec above a 14-bit slot.

pub mod aac;
pub mod at9;
pub mod decoder;
pub mod ffmpeg;
pub mod mp3;
pub mod mp3_frame;
pub mod opus;

use self::aac::AjmAacDecoder;
use self::at9::AjmAt9Decoder;
use self::decoder::*;
use self::mp3::AjmMp3Decoder;
use self::opus::AjmOpusDecoder;
use super::Export;
use crate::audio::atrac9::Atrac9Config;
use crate::context::{CpuContext, CpuRegister};
use crate::registry::HleResult;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

pub const AJM_ERROR_INVALID_CONTEXT: HleResult = 0x8093_0002u32 as i32 as i64;
pub const AJM_ERROR_INVALID_BATCH: HleResult = 0x8093_0004u32 as i32 as i64;
pub const AJM_ERROR_INVALID_PARAMETER: HleResult = 0x8093_0005u32 as i32 as i64;
pub const AJM_ERROR_CODEC_NOT_SUPPORTED: HleResult = 0x8093_0008u32 as i32 as i64;
pub const AJM_ERROR_JOB_CREATION: HleResult = 0x8093_0012u32 as i32 as i64;

pub const CODEC_DEC_MP3: u32 = 0;
pub const CODEC_DEC_AT9: u32 = 1;
pub const CODEC_DEC_M4AAC: u32 = 2;
pub const CODEC_DEC_UNKNOWN_NEW_ABI: u32 = 14;
pub const CODEC_DEC_OPUS: u32 = 24;

/// The only context id handed out by sceAjmInitialize.
const CONTEXT_ID: u32 = 1;
const INSTANCE_SLOT_BITS: u32 = 14;
const INSTANCE_SLOT_MASK: u32 = (1 << INSTANCE_SLOT_BITS) - 1;

const JOB_CONTROL_SIZE: u64 = 48;
const JOB_RUN_SIZE: u64 = 64;
/// interval (float) + 3 floats + 8 + a run job.
const JOB_GET_STATISTICS_SIZE: u64 = 4 + 4 * 3 + 8 + JOB_RUN_SIZE;
/// two floats + flags + 12 + a control job.
const JOB_SET_RESAMPLE_PARAMETERS_SIZE: u64 = 4 * 2 + 4 + 12 + JOB_CONTROL_SIZE;
/// SCE_AJM_JOB_RUN_SPLIT_SIZE(n) = 32 + 16 per descriptor.
const JOB_SPLIT_BASE_SIZE: u64 = 32;
const BUFFER_DESCRIPTOR_BYTES: u64 = 16;

const SIDEBAND_RESULT_SIZE: usize = 8;
const SIDEBAND_STREAM_SIZE: usize = 16;
const SIDEBAND_MFRAME_SIZE: usize = 8;
const SIDEBAND_RESAMPLE_INFO_SIZE: usize = 40;
/// AjmGetStatisticsResult: result, batch usage, 3 interval usages, 6 memory words.
const STATISTICS_RESULT_SIZE: usize = SIDEBAND_RESULT_SIZE + 4 + 12 + 24;
/// AjmBatchError { int code; const void *job; u32 cmdOffset; const void *jobRa; }.
const BATCH_ERROR_SIZE: usize = 32;

/// AjmBatchInfo: buffer, offset, size, last_good_job, last_good_job_ra.
const BATCH_INFO_OFFSET: u64 = 8;
const BATCH_INFO_SIZE: u64 = 16;
const BATCH_INFO_LAST_GOOD_JOB: u64 = 24;
const BATCH_INFO_LAST_GOOD_JOB_RA: u64 = 32;

pub const FLAG_RUN_GET_CODEC_INFO: u64 = 1 << 11;
pub const FLAG_RUN_MULTIPLE_FRAMES: u64 = 1 << 12;
pub const FLAG_CONTROL_RESET: u64 = 1 << 13;
pub const FLAG_CONTROL_INITIALIZE: u64 = 1 << 14;
pub const FLAG_SIDEBAND_RESAMPLE_INFO: u64 = 1 << 44;
pub const FLAG_SIDEBAND_GAPLESS: u64 = 1 << 45;
pub const FLAG_SIDEBAND_FORMAT: u64 = 1 << 46;
pub const FLAG_SIDEBAND_STREAM: u64 = 1 << 47;

/// Upper bound for one job's gathered input / output.
const MAX_JOB_BYTES: u64 = 64 * 1024 * 1024;
const MAX_BUFFER_DESCRIPTORS: u64 = 256;

static NEXT_INSTANCE: AtomicU32 = AtomicU32::new(1);
static NEXT_BATCH: AtomicU32 = AtomicU32::new(1);
static DECODE_COUNT: AtomicU64 = AtomicU64::new(0);

struct InstanceState {
    #[allow(dead_code)]
    context: u32,
    #[allow(dead_code)]
    codec: u32,
    #[allow(dead_code)]
    flags: u64,
    gapless: GaplessState,
    /// Resampling ratio set by sceAjmBatchJobSetResampleParameters
    /// (1.0 until the job sets one; 0 remaining samples: nothing is
    /// resampled, the decoders already output the stream rate).
    resample_ratio: f32,
    decoder: Option<Box<dyn AjmDecoder>>,
}

static INSTANCES: Mutex<Option<HashMap<u32, InstanceState>>> = Mutex::new(None);

fn with_instances<R>(f: impl FnOnce(&mut HashMap<u32, InstanceState>) -> R) -> R {
    let mut g = INSTANCES.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(HashMap::new))
}

/// Runs `f` on the instance's decoder; INVALID_PARAMETER when the instance
/// does not exist or its codec has no decoder.
fn with_decoder(instance: u32, f: impl FnOnce(&mut dyn AjmDecoder, &mut GaplessState) -> DecodeResult) -> DecodeResult {
    with_instances(|m| match m.get_mut(&instance) {
        Some(InstanceState { decoder: Some(d), gapless, .. }) => f(d.as_mut(), gapless),
        _ => DecodeResult::with_result(RESULT_INVALID_PARAMETER),
    })
}

pub fn codec_name(codec: u32) -> &'static str {
    match codec {
        CODEC_DEC_MP3 => "MP3 decoder",
        CODEC_DEC_AT9 => "ATRAC9 decoder",
        CODEC_DEC_M4AAC => "MPEG4-AAC decoder",
        3 => "CELP8 decoder",
        4 => "CELP8 encoder",
        12 => "CELP16 decoder",
        13 => "CELP16 encoder",
        22 => "HE-VAG decoder",
        23 => "LPCM decoder",
        CODEC_DEC_OPUS => "Opus decoder",
        _ => "unknown",
    }
}

fn codec_is_supported(codec: u32) -> bool {
    matches!(codec, CODEC_DEC_MP3 | CODEC_DEC_AT9 | CODEC_DEC_M4AAC | CODEC_DEC_OPUS | CODEC_DEC_UNKNOWN_NEW_ABI)
}

fn codec_is_valid(codec: u32) -> bool {
    matches!(codec, 0..=4 | 12 | 13 | 14 | 22 | 23 | 24)
}

fn log_codec_support(codec: u32) -> bool {
    let supported = codec_is_supported(codec);
    if supported {
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if !LOGGED.swap(true, Ordering::Relaxed) {
            log::info!("[AUDIO] AJM codec={codec}, name={}", codec_name(codec));
        }
    } else {
        log::warn!("[AUDIO] AJM unsupported codec={codec} ({})", codec_name(codec));
    }
    supported
}

fn create_decoder(codec: u32, flags: u64) -> Option<Box<dyn AjmDecoder>> {
    if !codec_is_supported(codec) {
        return None;
    }
    let channels = match flag_channels(flags) {
        0 => 2,
        n => n,
    };
    let Some(encoding) = flag_encoding(flags) else {
        log::error!("[AUDIO] AJM: unsupported PCM sample encoding {}", (flags >> 7) & 7);
        return None;
    };
    match codec {
        CODEC_DEC_MP3 => Some(Box::new(AjmMp3Decoder::new(channels, 48000, encoding))),
        CODEC_DEC_AT9 => Some(Box::new(AjmAt9Decoder::new(channels, 48000, encoding, flags))),
        CODEC_DEC_M4AAC => Some(Box::new(AjmAacDecoder::new(channels, 48000, encoding, flags))),
        CODEC_DEC_OPUS => Some(Box::new(AjmOpusDecoder::new(channels, encoding))),
        _ => None,
    }
}

fn stack_arg(ctx: &CpuContext, index: u64) -> u64 {
    ctx.read_u64(ctx.get(CpuRegister::Rsp) + 8 + index * 8).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Batch buffer and sideband writers
// ---------------------------------------------------------------------------

/// Reserves `job_size` bytes in the batch described by `info` (zeroed).
fn append_job(ctx: &CpuContext, info: u64, job_size: u64, name: &str) -> HleResult {
    let (Some(buffer), Some(offset), Some(size)) =
        (ctx.read_u64(info), ctx.read_u64(info + BATCH_INFO_OFFSET), ctx.read_u64(info + BATCH_INFO_SIZE))
    else {
        return AJM_ERROR_JOB_CREATION;
    };
    if offset.checked_add(job_size).is_none_or(|end| end > size) {
        log::debug!("[AUDIO] ajm {name}: not enough batch space, offset={offset:#x} size={size:#x} job={job_size:#x}");
        return AJM_ERROR_JOB_CREATION;
    }
    let job = buffer + offset;
    ctx.write_u64(info + BATCH_INFO_LAST_GOOD_JOB, job);
    ctx.write_u64(info + BATCH_INFO_LAST_GOOD_JOB_RA, 0);
    ctx.write_u64(info + BATCH_INFO_OFFSET, offset + job_size);
    zero_guest(ctx, job, job_size);
    0
}

fn zero_guest(ctx: &CpuContext, addr: u64, len: u64) {
    if addr == 0 || len == 0 {
        return;
    }
    const ZERO: [u8; 4096] = [0; 4096];
    let mut off = 0;
    while off < len {
        let n = (len - off).min(ZERO.len() as u64);
        if !ctx.write_bytes(addr + off, &ZERO[..n as usize]) {
            return;
        }
        off += n;
    }
}

fn clamp_i32(v: usize) -> i32 {
    v.min(i32::MAX as usize) as i32
}

fn stream_bytes(r: &DecodeResult) -> [u8; SIDEBAND_STREAM_SIZE] {
    let mut b = [0u8; SIDEBAND_STREAM_SIZE];
    b[0..4].copy_from_slice(&clamp_i32(r.input_consumed).to_le_bytes());
    b[4..8].copy_from_slice(&clamp_i32(r.output_written).to_le_bytes());
    b[8..16].copy_from_slice(&r.total_decoded_samples.to_le_bytes());
    b
}

/// Sideband output assembled on the host and written in one copy.
struct Sideband {
    bytes: Vec<u8>,
}

impl Sideband {
    /// Zeroed block of `size` bytes with the result header (when it fits).
    fn new(size: usize, r: &DecodeResult) -> Self {
        let mut bytes = vec![0u8; size];
        if size >= SIDEBAND_RESULT_SIZE {
            bytes[0..4].copy_from_slice(&r.result.to_le_bytes());
            bytes[4..8].copy_from_slice(&r.internal_result.to_le_bytes());
        }
        Self { bytes }
    }

    fn put(&mut self, at: usize, data: &[u8]) {
        if at < self.bytes.len() {
            let n = data.len().min(self.bytes.len() - at);
            self.bytes[at..at + n].copy_from_slice(&data[..n]);
        }
    }

    fn write(&self, ctx: &CpuContext, addr: u64) {
        if addr != 0 && !self.bytes.is_empty() {
            ctx.write_bytes(addr, &self.bytes);
        }
    }
}

fn write_result(ctx: &CpuContext, addr: u64, size: usize, r: &DecodeResult) {
    if addr != 0 && size != 0 {
        Sideband::new(size, r).write(ctx, addr);
    }
}

/// Result + Stream (+ MFrame) of the decode job builders.
fn write_stream_result(ctx: &CpuContext, addr: u64, size: usize, r: &DecodeResult, multiple_frames: bool) {
    if addr == 0 {
        return;
    }
    let mut sb = Sideband::new(size, r);
    if size >= SIDEBAND_RESULT_SIZE + SIDEBAND_STREAM_SIZE {
        sb.put(SIDEBAND_RESULT_SIZE, &stream_bytes(r));
    }
    if multiple_frames && size >= SIDEBAND_RESULT_SIZE + SIDEBAND_STREAM_SIZE + SIDEBAND_MFRAME_SIZE {
        sb.put(SIDEBAND_RESULT_SIZE + SIDEBAND_STREAM_SIZE, &r.frames.to_le_bytes());
    }
    sb.write(ctx, addr);
}

/// Sideband of run / control jobs: Result, then Stream, Format,
/// GaplessDecode, ResampleInfo, MFrame and CodecInfo, each only when
/// requested and when it fits (a block that does not fit is skipped).
///
/// The positional order is the one libSceAjm users parse: the stream block
/// (input consumed / output written) always comes right after the result,
/// so a codec that reads `consumed` from Result+8 advances its input even
/// when it also asked for the format or gapless blocks.
fn write_sideband(
    ctx: &CpuContext,
    flags: u64,
    addr: u64,
    size: usize,
    codec_info: Option<(usize, &[u8])>,
    gapless: Option<GaplessDecode>,
    r: &DecodeResult,
) {
    if addr == 0 || size == 0 {
        return;
    }
    let mut sb = Sideband::new(size, r);
    let mut offset = SIDEBAND_RESULT_SIZE;
    if flags & FLAG_SIDEBAND_STREAM != 0 && offset + SIDEBAND_STREAM_SIZE <= size {
        sb.put(offset, &stream_bytes(r));
        offset += SIDEBAND_STREAM_SIZE;
    }
    if flags & FLAG_SIDEBAND_FORMAT != 0 && offset + SidebandFormat::BYTES <= size {
        sb.put(offset, &r.format.to_bytes());
        offset += SidebandFormat::BYTES;
    }
    if flags & FLAG_SIDEBAND_GAPLESS != 0 && offset + GaplessDecode::BYTES <= size {
        if let Some(g) = gapless {
            sb.put(offset, &g.to_bytes());
        }
        offset += GaplessDecode::BYTES;
    }
    if flags & FLAG_SIDEBAND_RESAMPLE_INFO != 0 && offset + SIDEBAND_RESAMPLE_INFO_SIZE <= size {
        sb.put(offset, &1.0f32.to_le_bytes());
        offset += SIDEBAND_RESAMPLE_INFO_SIZE;
    }
    if flags & FLAG_RUN_MULTIPLE_FRAMES != 0 && offset + SIDEBAND_MFRAME_SIZE <= size {
        sb.put(offset, &r.frames.to_le_bytes());
        offset += SIDEBAND_MFRAME_SIZE;
    }
    if flags & FLAG_RUN_GET_CODEC_INFO != 0 {
        if let Some((info_size, info)) = codec_info {
            if offset + info_size <= size {
                sb.put(offset, &info[..info.len().min(info_size)]);
            }
        }
    }
    sb.write(ctx, addr);
}

/// Codec info block and gapless state of an instance, for run/control sidebands.
fn instance_sideband_state(instance: u32) -> (Option<(usize, Vec<u8>)>, Option<GaplessDecode>) {
    with_instances(|m| match m.get(&instance) {
        Some(InstanceState { decoder: Some(d), gapless, .. }) => {
            (Some((d.codec_info_size(), d.codec_info())), Some(gapless.current))
        }
        _ => (None, None),
    })
}

/// Result (with the current format) of an instance, or INVALID_PARAMETER
/// with the default format.
fn instance_format(instance: u32) -> DecodeResult {
    with_instances(|m| match m.get(&instance) {
        Some(InstanceState { decoder: Some(d), .. }) => {
            let mut r = d.make_result();
            r.format = d.format();
            r
        }
        _ => {
            let mut r = DecodeResult::with_result(RESULT_INVALID_PARAMETER);
            r.format = SidebandFormat::new(2, 48000, PcmEncoding::S16);
            r
        }
    })
}

// ---------------------------------------------------------------------------
// Decoding over guest buffers
// ---------------------------------------------------------------------------

thread_local! {
    /// Reused host copies of a job's input and output.
    static SCRATCH: RefCell<(Vec<u8>, Vec<u8>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

#[derive(Clone, Copy)]
struct GuestBuffer {
    addr: u64,
    len: u64,
}

/// One (addr, size) pair, or for the split form an array of AjmBuffer
/// descriptors. None when the descriptors cannot be read or are too large.
/// An input buffer whose stated size is larger than a job can hold (a
/// title's remaining-size count that went below zero states about 4 GiB
/// ahead of one packet) is read only as far as guest memory is readable
/// from its start, up to the job limit; the decoder takes the packets it
/// needs from there.
fn clamp_input(addr: u64, len: u64) -> u64 {
    if len <= MAX_JOB_BYTES {
        return len;
    }
    match ps5vm::ranges::accessible_runs(addr, MAX_JOB_BYTES, false).first() {
        Some(&(start, end)) if start == addr => end - start,
        _ => 0,
    }
}

fn collect_buffers(ctx: &CpuContext, split: bool, addr: u64, count_or_size: u64) -> Option<Vec<GuestBuffer>> {
    collect(ctx, split, addr, count_or_size, false)
}

/// Input buffers of a job: a single oversized buffer is clamped (see
/// [`clamp_input`]) instead of refusing the job.
fn collect_inputs(ctx: &CpuContext, split: bool, addr: u64, count_or_size: u64) -> Option<Vec<GuestBuffer>> {
    collect(ctx, split, addr, count_or_size, true)
}

fn collect(ctx: &CpuContext, split: bool, addr: u64, count_or_size: u64, input: bool) -> Option<Vec<GuestBuffer>> {
    if !split {
        let len = if input && addr != 0 { clamp_input(addr, count_or_size) } else { count_or_size };
        if len > MAX_JOB_BYTES {
            return None;
        }
        return Some(if len != 0 && addr != 0 { vec![GuestBuffer { addr, len }] } else { Vec::new() });
    }
    if input && count_or_size == 1 && addr != 0 {
        let (Some(a), Some(len)) = (ctx.read_u64(addr), ctx.read_u64(addr + 8)) else { return None };
        let len = if a != 0 { clamp_input(a, len) } else { len };
        if len > MAX_JOB_BYTES {
            return None;
        }
        return Some(vec![GuestBuffer { addr: a, len }]);
    }
    if count_or_size > MAX_BUFFER_DESCRIPTORS {
        return None;
    }
    let mut v = Vec::with_capacity(count_or_size as usize);
    let mut total = 0u64;
    for i in 0..count_or_size {
        if addr == 0 {
            break;
        }
        let d = addr + i * BUFFER_DESCRIPTOR_BYTES;
        let (Some(a), Some(len)) = (ctx.read_u64(d), ctx.read_u64(d + 8)) else { return None };
        total = total.checked_add(len)?;
        if total > MAX_JOB_BYTES {
            return None;
        }
        v.push(GuestBuffer { addr: a, len });
    }
    Some(v)
}

/// Gathers the inputs, decodes into a host buffer the size of all outputs,
/// and scatters the produced bytes back over the output buffers.
fn decode_buffers(ctx: &CpuContext, instance: u32, inputs: &[GuestBuffer], outputs: &[GuestBuffer], multiple_frames: bool) -> DecodeResult {
    SCRATCH.with(|s| {
        let (input, output) = &mut *s.borrow_mut();
        input.clear();
        for b in inputs {
            let start = input.len();
            input.resize(start + b.len as usize, 0);
            if b.addr != 0 && b.len != 0 && ctx.vm.read(b.addr, &mut input[start..]).is_err() {
                return DecodeResult::with_result(RESULT_INVALID_PARAMETER);
            }
        }
        let out_len: u64 = outputs.iter().map(|b| b.len).sum();
        output.clear();
        output.resize(out_len as usize, 0);
        let r = with_decoder(instance, |d, g| d.decode(input, output, multiple_frames, g));
        let mut off = 0usize;
        for b in outputs {
            let n = (b.len as usize).min(r.output_written.saturating_sub(off));
            if b.addr != 0 && n != 0 {
                ctx.write_bytes(b.addr, &output[off..off + n]);
            }
            off += b.len as usize;
        }
        if DECODE_COUNT.fetch_add(1, Ordering::Relaxed) == 0 {
            log::info!(
                "[AUDIO] first AJM decode: instance={instance:#x} in={} out={} -> consumed={} produced={} frames={} result={:#x}",
                input.len(),
                out_len,
                r.input_consumed,
                r.output_written,
                r.frames,
                r.result
            );
        }
        r
    })
}

fn decode_job(ctx: &CpuContext, split: bool, instance: u32, input: u64, in_cs: u64, output: u64, out_cs: u64, multiple: bool) -> DecodeResult {
    match (collect_inputs(ctx, split, input, in_cs), collect_buffers(ctx, split, output, out_cs)) {
        (Some(i), Some(o)) => decode_buffers(ctx, instance, &i, &o, multiple),
        _ => DecodeResult::with_result(RESULT_INVALID_PARAMETER),
    }
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

/// sceAjmDecAt9ParseConfigData(const u8 config[4], AjmDecAt9ConfigDataInfo *out)
fn cb_dec_at9_parse_config_data(ctx: &mut CpuContext) -> HleResult {
    let (cfg, out) = (ctx.rdi(), ctx.rsi());
    if cfg == 0 || out == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let Some(data) = ctx.read_bytes(cfg, 4) else { return AJM_ERROR_INVALID_PARAMETER };
    let Ok(c) = Atrac9Config::new(&data) else { return AJM_ERROR_INVALID_PARAMETER };
    if c.channel_count <= 0 || c.sample_rate <= 0 || c.frame_samples <= 0 || c.frames_per_superframe <= 0 || c.superframe_bytes <= 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let mut info = [0u8; 20];
    for (i, v) in [c.channel_count, c.sample_rate, c.frame_samples, c.frame_samples * c.frames_per_superframe, c.superframe_bytes]
        .iter()
        .enumerate()
    {
        info[i * 4..i * 4 + 4].copy_from_slice(&(*v as u32).to_le_bytes());
    }
    if ctx.write_bytes(out, &info) {
        0
    } else {
        AJM_ERROR_INVALID_PARAMETER
    }
}

/// sceAjmInitialize(i64 reserved, u32 *context)
fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    let (reserved, out) = (ctx.rdi(), ctx.rsi());
    if out == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    if reserved != 0 {
        log::debug!("[AUDIO] sceAjmInitialize: ignoring reserved = {reserved:#x}");
    }
    ctx.write_u32(out, CONTEXT_ID);
    0
}

fn cb_finalize(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// sceAjmModuleRegister(context, codec, i64 reserved)
fn cb_module_register(ctx: &mut CpuContext) -> HleResult {
    let (context, codec) = (ctx.rdi() as u32, ctx.rsi() as u32);
    log::info!("[AUDIO] sceAjmModuleRegister codec={codec} ({})", codec_name(codec));
    let valid = codec_is_valid(codec);
    let supported = valid && log_codec_support(codec);
    if context != CONTEXT_ID {
        AJM_ERROR_INVALID_CONTEXT
    } else if !valid {
        AJM_ERROR_INVALID_PARAMETER
    } else if !supported {
        AJM_ERROR_CODEC_NOT_SUPPORTED
    } else {
        0
    }
}

fn cb_module_unregister(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// sceAjmInstanceCreate(context, codec, flags, u32 *instance)
fn cb_instance_create(ctx: &mut CpuContext) -> HleResult {
    let (context, codec, flags, out) = (ctx.rdi() as u32, ctx.rsi() as u32, ctx.rdx(), ctx.rcx());
    if out == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let supported = log_codec_support(codec);
    let slot = NEXT_INSTANCE.fetch_add(1, Ordering::Relaxed) & INSTANCE_SLOT_MASK;
    let instance = (codec << INSTANCE_SLOT_BITS) | slot;
    ctx.write_u32(out, instance);
    let state = InstanceState { context, codec, flags, gapless: GaplessState::default(), resample_ratio: 1.0, decoder: create_decoder(codec, flags) };
    with_instances(|m| m.insert(instance, state));
    log::debug!("[AUDIO] sceAjmInstanceCreate codec={codec} flags={flags:#x} instance={instance:#010x}");
    if !supported {
        log::warn!("[AUDIO] AJM: unsupported codec instance created without decoder");
    }
    0
}

fn cb_instance_destroy(ctx: &mut CpuContext) -> HleResult {
    let instance = ctx.rsi() as u32;
    with_instances(|m| m.remove(&instance));
    0
}

fn cb_memory_register(_ctx: &mut CpuContext) -> HleResult {
    0
}

fn cb_memory_unregister(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// sceAjmBatchInitialize(void *buffer, size_t size, AjmBatchInfo *info)
fn cb_batch_initialize(ctx: &mut CpuContext) -> HleResult {
    let (buf, size, info) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if buf == 0 || info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let mut b = [0u8; 40];
    b[0..8].copy_from_slice(&buf.to_le_bytes());
    b[16..24].copy_from_slice(&size.to_le_bytes());
    ctx.write_bytes(info, &b);
    0
}

fn clear_batch_error(ctx: &CpuContext, addr: u64) {
    if addr != 0 {
        ctx.write_bytes(addr, &[0u8; BATCH_ERROR_SIZE]);
    }
}

/// Started batches not yet waited for, with the context that started them.
static BATCHES: Mutex<BTreeMap<u32, u32>> = Mutex::new(BTreeMap::new());

fn batches() -> std::sync::MutexGuard<'static, BTreeMap<u32, u32>> {
    BATCHES.lock().unwrap_or_else(|p| p.into_inner())
}

/// sceAjmBatchStart(context, info, priority, AjmBatchError *error, u32 *batch)
fn cb_batch_start(ctx: &mut CpuContext) -> HleResult {
    let (context, info, error, batch) = (ctx.rdi() as u32, ctx.rsi(), ctx.rcx(), ctx.r8());
    if context != CONTEXT_ID {
        return AJM_ERROR_INVALID_CONTEXT;
    }
    if info == 0 || batch == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    clear_batch_error(ctx, error);
    let id = NEXT_BATCH.fetch_add(1, Ordering::Relaxed);
    batches().insert(id, context);
    ctx.write_u32(batch, id);
    0
}

/// sceAjmBatchWait(context, batch, timeout, AjmBatchError *error): every
/// batch has already completed; waiting releases it, so a batch of
/// another context or one already waited for is INVALID_BATCH.
fn cb_batch_wait(ctx: &mut CpuContext) -> HleResult {
    let (context, batch) = (ctx.rdi() as u32, ctx.rsi() as u32);
    if context != CONTEXT_ID {
        return AJM_ERROR_INVALID_CONTEXT;
    }
    let mut b = batches();
    if b.get(&batch) != Some(&context) {
        return AJM_ERROR_INVALID_BATCH;
    }
    b.remove(&batch);
    drop(b);
    clear_batch_error(ctx, ctx.rcx());
    0
}

/// sceAjmBatchCancel(context, batch): a batch of this context.
fn cb_batch_cancel(ctx: &mut CpuContext) -> HleResult {
    let (context, batch) = (ctx.rdi() as u32, ctx.rsi() as u32);
    if context != CONTEXT_ID {
        return AJM_ERROR_INVALID_CONTEXT;
    }
    if batches().get(&batch) == Some(&context) { 0 } else { AJM_ERROR_INVALID_BATCH }
}

/// sceAjmBatchErrorDump(const AjmBatchInfo *info, AjmBatchError *error)
fn cb_batch_error_dump(ctx: &mut CpuContext) -> HleResult {
    clear_batch_error(ctx, ctx.rsi());
    0
}

/// sceAjmStrError(int error) -> const char *
fn cb_str_error(ctx: &mut CpuContext) -> HleResult {
    static TEXT: OnceLock<u64> = OnceLock::new();
    let addr = *TEXT.get_or_init(|| {
        let Some(p) = crate::pthread::alloc_guest_block(ctx, 4) else { return 0 };
        ctx.write_bytes(p, b"AJM\0");
        p
    });
    addr as HleResult
}

/// sceAjmBatchJobInitialize(info, instance, const void *params, size_t size, result *)
fn cb_batch_job_initialize(ctx: &mut CpuContext) -> HleResult {
    let (info, instance, params, psize, result) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx(), ctx.r8());
    if info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let bytes = if params != 0 && psize != 0 { ctx.read_bytes(params, psize.min(4096) as usize).unwrap_or_default() } else { Vec::new() };
    let r = with_decoder(instance, |d, g| {
        g.reset();
        d.initialize(&bytes)
    });
    write_result(ctx, result, SIDEBAND_RESULT_SIZE, &r);
    append_job(ctx, info, JOB_CONTROL_SIZE, "initialize")
}

/// Control: gapless update, reset and/or initialize, per the flags.
fn control_instance(instance: u32, flags: u64, input: &[u8]) -> DecodeResult {
    with_decoder(instance, |d, g| {
        let mut r = d.make_result();
        let mut input = input;
        if flags & FLAG_SIDEBAND_GAPLESS != 0 {
            if input.len() < GaplessDecode::BYTES {
                r.result = RESULT_INVALID_PARAMETER;
                return r;
            }
            g.set(GaplessDecode::from_bytes(input), flags & FLAG_CONTROL_RESET != 0);
            input = &input[GaplessDecode::BYTES..];
        }
        if flags & FLAG_CONTROL_RESET != 0 {
            g.reset();
            d.reset();
            r = d.make_result();
        }
        if flags & FLAG_CONTROL_INITIALIZE != 0 {
            g.reset();
            r = d.initialize(input);
        }
        r
    })
}

/// sceAjmBatchJobClearContext(info, instance, result *)
fn cb_batch_job_clear_context(ctx: &mut CpuContext) -> HleResult {
    let (info, instance, result) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx());
    if info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let r = control_instance(instance, FLAG_CONTROL_RESET, &[]);
    write_result(ctx, result, SIDEBAND_RESULT_SIZE, &r);
    append_job(ctx, info, JOB_CONTROL_SIZE, "clear-context")
}

/// sceAjmBatchJobDecode[Single](info, instance, in, inSize, out, outSize, result *)
fn decode_core(ctx: &CpuContext, multiple_frames: bool) -> HleResult {
    let (info, instance, input, in_size, output, out_size) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx(), ctx.r8(), ctx.r9());
    let result = stack_arg(ctx, 0);
    if info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let r = decode_job(ctx, false, instance, input, in_size, output, out_size, multiple_frames);
    let size = SIDEBAND_RESULT_SIZE + SIDEBAND_STREAM_SIZE + if multiple_frames { SIDEBAND_MFRAME_SIZE } else { 0 };
    write_stream_result(ctx, result, size, &r, multiple_frames);
    append_job(ctx, info, JOB_RUN_SIZE, if multiple_frames { "decode" } else { "decode-single" })
}

fn cb_batch_job_decode(ctx: &mut CpuContext) -> HleResult {
    decode_core(ctx, true)
}

fn cb_batch_job_decode_single(ctx: &mut CpuContext) -> HleResult {
    decode_core(ctx, false)
}

/// sceAjmBatchJobDecodeSplit(info, instance, const AjmBuffer *in, nIn, const AjmBuffer *out, nOut, result *)
fn cb_batch_job_decode_split(ctx: &mut CpuContext) -> HleResult {
    let (info, instance, inputs, n_in, outputs, n_out) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx(), ctx.r8(), ctx.r9());
    let result = stack_arg(ctx, 0);
    if info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let r = decode_job(ctx, true, instance, inputs, n_in, outputs, n_out, true);
    write_stream_result(ctx, result, SIDEBAND_RESULT_SIZE + SIDEBAND_STREAM_SIZE + SIDEBAND_MFRAME_SIZE, &r, true);
    let size = BUFFER_DESCRIPTOR_BYTES.saturating_mul(n_in.saturating_add(n_out)).saturating_add(JOB_SPLIT_BASE_SIZE);
    append_job(ctx, info, size, "decode-split")
}

/// sceAjmBatchJobEncode(info, instance, pcm, pcmSize, out, outSize, result *): no
/// encoder; the output is zero-filled and reported as produced.
fn cb_batch_job_encode(ctx: &mut CpuContext) -> HleResult {
    let (info, in_size, output, out_size) = (ctx.rdi(), ctx.rcx(), ctx.r8(), ctx.r9());
    let result = stack_arg(ctx, 0);
    if info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let r = DecodeResult {
        input_consumed: in_size as usize,
        output_written: out_size as usize,
        frames: (in_size != 0 || out_size != 0) as u32,
        total_decoded_samples: out_size,
        ..Default::default()
    };
    if output != 0 && out_size <= MAX_JOB_BYTES {
        zero_guest(ctx, output, out_size);
    }
    write_stream_result(ctx, result, SIDEBAND_RESULT_SIZE + SIDEBAND_STREAM_SIZE + SIDEBAND_MFRAME_SIZE, &r, true);
    append_job(ctx, info, JOB_RUN_SIZE, "encode")
}

/// sceAjmBatchJobGetInfo(info, instance, result *): Result + Format.
fn cb_batch_job_get_info(ctx: &mut CpuContext) -> HleResult {
    let (info, instance, result) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx());
    if info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let r = instance_format(instance);
    if result != 0 {
        let mut sb = Sideband::new(SIDEBAND_RESULT_SIZE + SidebandFormat::BYTES, &r);
        sb.put(SIDEBAND_RESULT_SIZE, &r.format.to_bytes());
        sb.write(ctx, result);
    }
    append_job(ctx, info, JOB_RUN_SIZE, "get-info")
}

/// sceAjmBatchJobGetCodecInfo(info, instance, result *, size_t resultSize)
fn cb_batch_job_get_codec_info(ctx: &mut CpuContext) -> HleResult {
    let (info, instance, result, size) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx());
    if info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let r = instance_format(instance);
    let size = size.min(4096) as usize;
    if result != 0 && size != 0 {
        let mut sb = Sideband::new(size, &r);
        if size > SIDEBAND_RESULT_SIZE {
            if let (Some((info_size, bytes)), _) = instance_sideband_state(instance) {
                if size - SIDEBAND_RESULT_SIZE >= info_size {
                    sb.put(SIDEBAND_RESULT_SIZE, &bytes);
                }
            }
        }
        sb.write(ctx, result);
    }
    append_job(ctx, info, JOB_RUN_SIZE, "get-codec-info")
}

/// sceAjmBatchJobSetGaplessDecode(info, instance, const gapless *, int reset, result *)
fn cb_batch_job_set_gapless_decode(ctx: &mut CpuContext) -> HleResult {
    let (info, instance, gapless, reset, result) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx() as i32, ctx.r8());
    if info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let flags = FLAG_SIDEBAND_GAPLESS | if reset != 0 { FLAG_CONTROL_RESET } else { 0 };
    let bytes = if gapless != 0 { ctx.read_bytes(gapless, GaplessDecode::BYTES).unwrap_or_default() } else { Vec::new() };
    let r = control_instance(instance, flags, &bytes);
    write_result(ctx, result, SIDEBAND_RESULT_SIZE, &r);
    append_job(ctx, info, JOB_CONTROL_SIZE, "set-gapless")
}

/// sceAjmBatchJobGetGaplessDecode(info, instance, result *): Result + GaplessDecode.
fn cb_batch_job_get_gapless_decode(ctx: &mut CpuContext) -> HleResult {
    let (info, instance, result) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx());
    if info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let mut current = GaplessDecode::default();
    let r = with_decoder(instance, |d, g| {
        current = g.current;
        d.make_result()
    });
    if result != 0 {
        let mut sb = Sideband::new(SIDEBAND_RESULT_SIZE + GaplessDecode::BYTES, &r);
        sb.put(SIDEBAND_RESULT_SIZE, &current.to_bytes());
        sb.write(ctx, result);
    }
    append_job(ctx, info, JOB_RUN_SIZE, "get-gapless")
}

/// sceAjmBatchJobSetResampleParameters(info, instance, float ratio, u32 flags,
/// result *): the ratio arrives in XMM0; flag 3 says the job ignores it.
/// A finite 0..4 ratio is recorded on the instance (reported back by
/// sceAjmBatchJobGetResampleInfo); anything else fails the job's result.
fn cb_batch_job_set_resample_parameters(ctx: &mut CpuContext) -> HleResult {
    let (info, instance, flags, result) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx());
    if info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let ratio = ps5cpu::exec::trap_xmm_low(0).map_or(0.0f32, |v| f32::from_bits(v as u32));
    let valid_ratio = flags & 0x8 != 0 || (ratio.is_finite() && (0.0..=4.0).contains(&ratio));
    let status = if valid_ratio {
        with_instances(|m| match m.get_mut(&instance) {
            Some(InstanceState { resample_ratio: r, .. }) => {
                if flags & 0x8 == 0 {
                    *r = ratio;
                }
                0
            }
            None => RESULT_INVALID_PARAMETER,
        })
    } else {
        RESULT_INVALID_PARAMETER
    };
    write_result(ctx, result, SIDEBAND_RESULT_SIZE, &DecodeResult::with_result(status));
    append_job(ctx, info, JOB_SET_RESAMPLE_PARAMETERS_SIZE, "set-resample")
}

/// sceAjmBatchJobSetResampleParametersEx(info, instance, float start, float change, u32 flags, result *)
fn cb_batch_job_set_resample_parameters_ex(ctx: &mut CpuContext) -> HleResult {
    let (info, result) = (ctx.rdi(), ctx.rcx());
    if info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    write_result(ctx, result, SIDEBAND_RESULT_SIZE, &DecodeResult::default());
    append_job(ctx, info, JOB_SET_RESAMPLE_PARAMETERS_SIZE, "set-resample-ex")
}

/// sceAjmBatchJobGetResampleInfo(info, instance, result *): Result +
/// ResampleInfo — the instance's ratio and the samples left to resample
/// (none: nothing resamples). An unknown instance reports the error in the
/// result word.
fn cb_batch_job_get_resample_info(ctx: &mut CpuContext) -> HleResult {
    let (info, instance, result) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx());
    if info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let ratio = with_instances(|m| m.get(&instance).map_or(None, |s| Some(s.resample_ratio)));
    if result != 0 {
        let r = match ratio {
            Some(ratio) => {
                let mut sb = Sideband::new(SIDEBAND_RESULT_SIZE + SIDEBAND_RESAMPLE_INFO_SIZE, &DecodeResult::default());
                sb.put(SIDEBAND_RESULT_SIZE, &ratio.to_le_bytes());
                sb.put(SIDEBAND_RESULT_SIZE + 4, &0i32.to_le_bytes());
                sb
            }
            None => Sideband::new(SIDEBAND_RESULT_SIZE + SIDEBAND_RESAMPLE_INFO_SIZE, &DecodeResult::with_result(RESULT_INVALID_PARAMETER)),
        };
        r.write(ctx, result);
    }
    append_job(ctx, info, JOB_RUN_SIZE, "get-resample-info")
}

/// sceAjmBatchJobGetStatistics(info, float interval, result *)
fn cb_batch_job_get_statistics(ctx: &mut CpuContext) -> HleResult {
    let (info, result) = (ctx.rdi(), ctx.rsi());
    if info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    write_result(ctx, result, STATISTICS_RESULT_SIZE, &DecodeResult::default());
    append_job(ctx, info, JOB_GET_STATISTICS_SIZE, "get-statistics")
}

/// sceAjmBatchJobControl(info, instance, flags, sbIn, sbInSize, sbOut, sbOutSize)
fn cb_batch_job_control(ctx: &mut CpuContext) -> HleResult {
    let (info, instance, flags, sb_in, sb_in_size, sb_out) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx(), ctx.r8(), ctx.r9());
    let sb_out_size = stack_arg(ctx, 0);
    if info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let input = if sb_in != 0 && sb_in_size != 0 { ctx.read_bytes(sb_in, sb_in_size.min(4096) as usize).unwrap_or_default() } else { Vec::new() };
    let r = control_instance(instance, flags, &input);
    let (codec_info, gapless) = instance_sideband_state(instance);
    let ci = codec_info.as_ref().map(|(n, b)| (*n, b.as_slice()));
    write_sideband(ctx, flags, sb_out, sb_out_size.min(4096) as usize, ci, gapless, &r);
    append_job(ctx, info, JOB_CONTROL_SIZE, "control")
}

/// sceAjmBatchJobRun[Split][BufferRa](info, instance, flags, in, inCountOrSize,
/// out, outCountOrSize, sideband, sidebandSize[, ra])
fn run_core(ctx: &CpuContext, split: bool) -> HleResult {
    let (info, instance, flags, input, in_cs, output) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx(), ctx.r8(), ctx.r9());
    let (out_cs, sideband, sideband_size) = (stack_arg(ctx, 0), stack_arg(ctx, 1), stack_arg(ctx, 2));
    if info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let r = decode_job(ctx, split, instance, input, in_cs, output, out_cs, flags & FLAG_RUN_MULTIPLE_FRAMES != 0);
    let (codec_info, gapless) = instance_sideband_state(instance);
    let ci = codec_info.as_ref().map(|(n, b)| (*n, b.as_slice()));
    write_sideband(ctx, flags, sideband, sideband_size.min(4096) as usize, ci, gapless, &r);
    if split {
        let size = BUFFER_DESCRIPTOR_BYTES.saturating_mul(in_cs.saturating_add(out_cs)).saturating_add(JOB_SPLIT_BASE_SIZE);
        append_job(ctx, info, size, "run-split")
    } else {
        append_job(ctx, info, JOB_RUN_SIZE, "run")
    }
}

fn cb_batch_job_run(ctx: &mut CpuContext) -> HleResult {
    run_core(ctx, false)
}

fn cb_batch_job_run_split(ctx: &mut CpuContext) -> HleResult {
    run_core(ctx, true)
}

const LIB: &str = "libSceAjm";

pub(crate) static EXPORTS: &[Export] = &[
    ("dl+4eHSzUu4", "sceAjmInitialize", LIB, cb_initialize),
    ("Q3dyFuwGn64", "sceAjmModuleRegister", LIB, cb_module_register),
    ("Wi7DtlLV+KI", "sceAjmModuleUnregister", LIB, cb_module_unregister),
    ("MHur6qCsUus", "sceAjmFinalize", LIB, cb_finalize),
    ("AxoDrINp4J8", "sceAjmInstanceCreate", LIB, cb_instance_create),
    ("RbLbuKv8zho", "sceAjmInstanceDestroy", LIB, cb_instance_destroy),
    ("bkRHEYG6lEM", "sceAjmMemoryRegister", LIB, cb_memory_register),
    ("pIpGiaYkHkM", "sceAjmMemoryUnregister", LIB, cb_memory_unregister),
    ("MmpF1XsQiHw", "sceAjmBatchInitialize", LIB, cb_batch_initialize),
    ("5tOfnaClcqM", "sceAjmBatchStart", LIB, cb_batch_start),
    ("-qLsfDAywIY", "sceAjmBatchWait", LIB, cb_batch_wait),
    ("NVDXiUesSbA", "sceAjmBatchCancel", LIB, cb_batch_cancel),
    ("WfAiBW8Wcek", "sceAjmBatchErrorDump", LIB, cb_batch_error_dump),
    ("AxhcqVv5AYU", "sceAjmStrError", LIB, cb_str_error),
    ("ezM2OhNxzck", "sceAjmBatchJobInitialize", LIB, cb_batch_job_initialize),
    ("uJ3m8INuikg", "sceAjmBatchJobClearContext", LIB, cb_batch_job_clear_context),
    ("39WxhR-ePew", "sceAjmBatchJobDecode", LIB, cb_batch_job_decode),
    ("5LLWbpP5xi8", "sceAjmBatchJobDecodeSingle", LIB, cb_batch_job_decode_single),
    ("SJ3i0DXP8vg", "sceAjmBatchJobDecodeSplit", LIB, cb_batch_job_decode_split),
    ("SlVIGK1Kl38", "sceAjmBatchJobEncode", LIB, cb_batch_job_encode),
    ("TBWW4aPfWcA", "sceAjmBatchJobGetInfo", LIB, cb_batch_job_get_info),
    ("uSrXaxT+oPQ", "sceAjmBatchJobGetCodecInfo", LIB, cb_batch_job_get_codec_info),
    ("SkEwpiu3tZg", "sceAjmBatchJobSetGaplessDecode", LIB, cb_batch_job_set_gapless_decode),
    ("Esr9db8S1S0", "sceAjmBatchJobGetGaplessDecode", LIB, cb_batch_job_get_gapless_decode),
    ("81HsnXFbWS4", "sceAjmBatchJobSetResampleParameters", LIB, cb_batch_job_set_resample_parameters),
    ("5ldnD16rYZw", "sceAjmBatchJobSetResampleParametersEx", LIB, cb_batch_job_set_resample_parameters_ex),
    ("JkdNCocpu1M", "sceAjmBatchJobGetResampleInfo", LIB, cb_batch_job_get_resample_info),
    ("3cAg7xN995U", "sceAjmBatchJobGetStatistics", LIB, cb_batch_job_get_statistics),
    ("7FZsbyVRM4U", "sceAjmBatchJobControl", LIB, cb_batch_job_control),
    ("jVCWcthifr8", "sceAjmBatchJobRun", LIB, cb_batch_job_run),
    ("Z9NVCesiP0Q", "sceAjmBatchJobRunSplit", LIB, cb_batch_job_run_split),
    ("ElslOCpOIns", "sceAjmBatchJobRunBufferRa", LIB, cb_batch_job_run),
    ("7jdAXK+2fMo", "sceAjmBatchJobRunSplitBufferRa", LIB, cb_batch_job_run_split),
    ("1t3ixYNXyuc", "sceAjmDecAt9ParseConfigData", LIB, cb_dec_at9_parse_config_data),
    ("eDFeTyi+G3Y", "sceAjmDecMp3ParseFrame", LIB, mp3_frame::cb_parse_frame),
];


#[cfg(test)]
mod tests;
