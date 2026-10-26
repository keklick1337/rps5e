// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceOpusDec and libSceOpusCeltDec: raw Opus packets to signed 16-bit
//! PCM, mono or stereo, decoded on the host.
//!
//! Both libraries share one implementation with their own contexts and
//! decoders; libSceOpusCeltDec only takes CELT-only packets (TOC config 16
//! and up). Contexts are ids handed out by `Initialize` (one per guest
//! variable); each decoder is keyed by the 640-byte state block the title
//! allocates. The host decodes at 48 kHz and resamples to the decoder's rate
//! (8, 12, 16, 24 or 48 kHz); the resampler's delay is filled with leading
//! silence so every call returns exactly the packet's duration. `Decode`
//! returns the bytes of PCM written, one packet per call.
//!
//! Not implemented (an error and a log line): terminating a context with
//! live decoders, SILK/hybrid packets in libSceOpusCeltDec. The console's
//! error codes are not known, so invalid arguments return EINVAL.

use super::ajm::decoder::PcmEncoding;
use super::ajm::ffmpeg::{HostAudioDecoder, HostCodec};
use super::ajm::opus::packet_layout;
use super::Export;
use crate::context::CpuContext;
use crate::registry::HleResult;
use std::collections::HashMap;
use std::sync::Mutex;

pub const OPUSDEC_ERROR_INVALID_ARGUMENT: HleResult = 0x8002_0016u32 as i32 as i64;
pub const OPUSDEC_ERROR_NOT_IMPLEMENTED: HleResult = 0x8002_0002u32 as i32 as i64;

/// Bytes of the state block a decoder needs.
const STATE_SIZE: HleResult = 640;
/// Rate the host decoder produces.
const HOST_RATE: u32 = 48000;
const SAMPLE_RATES: [u32; 5] = [8000, 12000, 16000, 24000, 48000];
/// The longest Opus packet: 120 ms.
const MAX_PACKET_SAMPLES: u32 = 5760;
/// Largest packet read.
const MAX_PACKET_BYTES: usize = 1 << 20;
/// First TOC configuration of the CELT-only mode.
const FIRST_CELT_CONFIG: u8 = 16;
/// Decode priorities the WithPriority forms accept.
const MIN_DECODE_PRIORITY: i32 = 20;
const MAX_DECODE_PRIORITY: i32 = 100;
/// The timeout that waits for the decode to finish.
const INFINITE_TIMEOUT: u32 = 0xFFFF_FFFF;

/// Sample format a Decode variant writes.
#[derive(Clone, Copy)]
enum PcmOut {
    S16,
    F32,
}

struct Decoder {
    context: u32,
    channels: u32,
    sample_rate: u32,
    host: HostAudioDecoder,
    /// Resampled PCM not returned yet (S16 bytes).
    pending: Vec<u8>,
}

#[derive(Default)]
struct State {
    /// Context id -> the guest variable that holds it.
    contexts: HashMap<u32, u64>,
    decoders: HashMap<u64, Decoder>,
    next_context: u32,
}

pub(crate) struct OpusLibrary {
    name: &'static str,
    celt_only: bool,
    state: Mutex<Option<State>>,
}

static OPUS: OpusLibrary = OpusLibrary::new("sceOpusDec", false);
static OPUS_CELT: OpusLibrary = OpusLibrary::new("sceOpusCeltDec", true);

/// Samples per channel of an Opus packet at 48 kHz; None when the TOC is
/// malformed.
fn packet_samples(packet: &[u8]) -> Option<u32> {
    if packet.is_empty() || (packet[0] & 3 == 3 && packet.len() < 2) {
        return None;
    }
    let (frames, frame_samples) = packet_layout(packet);
    let samples = frames * frame_samples;
    (frames != 0 && samples <= MAX_PACKET_SAMPLES).then_some(samples)
}

fn valid_channels(channels: i32) -> bool {
    matches!(channels, 1 | 2)
}

impl OpusLibrary {
    const fn new(name: &'static str, celt_only: bool) -> Self {
        Self { name, celt_only, state: Mutex::new(None) }
    }

    fn with_state<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        f(g.get_or_insert_with(|| State { next_context: 1, ..State::default() }))
    }

    /// Initialize(u32 *context)
    fn initialize(&self, ctx: &mut CpuContext) -> HleResult {
        let out = ctx.rdi();
        if out == 0 {
            return OPUSDEC_ERROR_INVALID_ARGUMENT;
        }
        let id = self.with_state(|s| {
            if s.contexts.values().any(|&v| v == out) || s.next_context == 0 {
                return None;
            }
            let id = s.next_context;
            s.next_context = s.next_context.wrapping_add(1);
            s.contexts.insert(id, out);
            Some(id)
        });
        let Some(id) = id else { return OPUSDEC_ERROR_INVALID_ARGUMENT };
        if !ctx.write_u32(out, id) {
            self.with_state(|s| s.contexts.remove(&id));
            return OPUSDEC_ERROR_INVALID_ARGUMENT;
        }
        0
    }

    /// Terminate(u32 *context)
    fn terminate(&self, ctx: &mut CpuContext) -> HleResult {
        let addr = ctx.rdi();
        let Some(id) = (addr != 0).then(|| ctx.read_u32(addr)).flatten() else {
            return OPUSDEC_ERROR_INVALID_ARGUMENT;
        };
        self.with_state(|s| {
            if !s.contexts.contains_key(&id) {
                return OPUSDEC_ERROR_INVALID_ARGUMENT;
            }
            if s.decoders.values().any(|d| d.context == id) {
                log::error!("[AUDIO] {}Terminate with live decoders is not implemented", self.name);
                return OPUSDEC_ERROR_NOT_IMPLEMENTED;
            }
            s.contexts.remove(&id);
            0
        })
    }

    /// GetSize(int channels)
    fn get_size(&self, ctx: &mut CpuContext) -> HleResult {
        if valid_channels(ctx.rdi() as i32) {
            STATE_SIZE
        } else {
            OPUSDEC_ERROR_INVALID_ARGUMENT
        }
    }

    /// CreateEx(u32 *context, void *state, int sampleRate, int channels)
    fn create_ex(&self, ctx: &mut CpuContext) -> HleResult {
        let (context, state, rate, channels) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as i32, ctx.rcx() as i32);
        if context == 0 || state == 0 || !valid_channels(channels) {
            return OPUSDEC_ERROR_INVALID_ARGUMENT;
        }
        let Some(rate) = SAMPLE_RATES.into_iter().find(|&r| r as i32 == rate) else {
            return OPUSDEC_ERROR_INVALID_ARGUMENT;
        };
        let Some(id) = ctx.read_u32(context) else { return OPUSDEC_ERROR_INVALID_ARGUMENT };
        self.with_state(|s| {
            if !s.contexts.contains_key(&id) || s.decoders.contains_key(&state) {
                return OPUSDEC_ERROR_INVALID_ARGUMENT;
            }
            let Some(mut host) = HostAudioDecoder::open(HostCodec::Opus, channels as u32, HOST_RATE) else {
                log::error!("[AUDIO] {}CreateEx: no host Opus decoder", self.name);
                return OPUSDEC_ERROR_NOT_IMPLEMENTED;
            };
            host.set_output_rate(rate);
            let decoder = Decoder { context: id, channels: channels as u32, sample_rate: rate, host, pending: Vec::new() };
            s.decoders.insert(state, decoder);
            0
        })
    }

    /// Decode(void *state, const u8 *data, int bytes, s16 *pcm, int capacity):
    /// bytes of PCM written. Nothing is written for a rejected packet.
    fn decode(&self, ctx: &mut CpuContext) -> HleResult {
        self.decode_variant(ctx, None, PcmOut::S16)
    }

    /// The Decode variants: `priority` (20 to 100) of the WithPriority forms
    /// is checked but schedules nothing; the Float forms write 32-bit float
    /// samples (a `capacity` in bytes) and return the bytes written.
    fn decode_variant(&self, ctx: &mut CpuContext, priority: Option<i32>, out: PcmOut) -> HleResult {
        let (state, data, bytes, pcm, capacity) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as i32, ctx.rcx(), ctx.r8() as i32);
        if data == 0 || bytes <= 0 || bytes as usize > MAX_PACKET_BYTES || pcm == 0 || capacity < 0 {
            // Packet loss concealment (no packet) is not implemented.
            return OPUSDEC_ERROR_INVALID_ARGUMENT;
        }
        if priority.is_some_and(|p| !(MIN_DECODE_PRIORITY..=MAX_DECODE_PRIORITY).contains(&p)) {
            return OPUSDEC_ERROR_INVALID_ARGUMENT;
        }
        let s16_capacity = match out {
            PcmOut::S16 => capacity,
            PcmOut::F32 => capacity / 4 * 2,
        };
        let pcm_bytes = match self.decode_packet(ctx, state, data, bytes, s16_capacity) {
            Ok(b) => b,
            Err(e) => return e,
        };
        let written = match out {
            PcmOut::S16 => pcm_bytes,
            PcmOut::F32 => pcm_bytes
                .chunks_exact(2)
                .flat_map(|c| (i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).to_le_bytes())
                .collect(),
        };
        if !ctx.write_bytes(pcm, &written) {
            return OPUSDEC_ERROR_INVALID_ARGUMENT;
        }
        written.len() as HleResult
    }

    /// One packet as S16 PCM bytes, at most `capacity` of them.
    fn decode_packet(&self, ctx: &CpuContext, state: u64, data: u64, bytes: i32, capacity: i32) -> Result<Vec<u8>, HleResult> {
        let Some(packet) = ctx.read_bytes(data, bytes as usize) else { return Err(OPUSDEC_ERROR_INVALID_ARGUMENT) };
        self.with_state(|s| {
            let Some(d) = s.decoders.get_mut(&state) else { return Err(OPUSDEC_ERROR_INVALID_ARGUMENT) };
            if self.celt_only && packet[0] >> 3 < FIRST_CELT_CONFIG {
                log::error!("[AUDIO] {}Decode: SILK and hybrid packets are not implemented", self.name);
                return Err(OPUSDEC_ERROR_NOT_IMPLEMENTED);
            }
            let Some(samples) = packet_samples(&packet) else { return Err(OPUSDEC_ERROR_INVALID_ARGUMENT) };
            let samples = samples / (HOST_RATE / d.sample_rate);
            let output_bytes = samples as usize * d.channels as usize * 2;
            if (capacity / 2 * 2) < output_bytes as i32 {
                return Err(OPUSDEC_ERROR_INVALID_ARGUMENT);
            }
            if d.host.send(&packet, None).is_err() {
                return Err(OPUSDEC_ERROR_INVALID_ARGUMENT);
            }
            let frame = match d.host.receive(PcmEncoding::S16) {
                Ok(Some(f)) if f.channels == d.channels && f.sample_rate == d.sample_rate => f,
                _ => return Err(OPUSDEC_ERROR_INVALID_ARGUMENT),
            };
            let decoded = d.host.data(&frame, PcmEncoding::S16).to_vec();
            d.pending.extend_from_slice(&decoded);
            if d.pending.len() < output_bytes {
                // The resampler's delay: lead with silence.
                let missing = output_bytes - d.pending.len();
                d.pending.splice(0..0, std::iter::repeat_n(0u8, missing));
            }
            Ok(d.pending.drain(..output_bytes).collect())
        })
    }

    /// The AndTimeout forms: (state, data, bytes, pcm, capacity, priority,
    /// u32 timeout, u32 *batch). Decoding is synchronous, which is what the
    /// infinite timeout (0xFFFFFFFF, the one Decode uses) asks for; a
    /// finite one would leave a batch to wait for and is not implemented.
    fn decode_with_timeout(&self, ctx: &mut CpuContext, out: PcmOut) -> HleResult {
        let rsp = ctx.get(crate::context::CpuRegister::Rsp);
        let timeout = ctx.read_u64(rsp + 8).unwrap_or(0) as u32;
        if timeout != INFINITE_TIMEOUT {
            log::error!("[AUDIO] {}Decode with a timeout of {timeout} is not implemented", self.name);
            return OPUSDEC_ERROR_NOT_IMPLEMENTED;
        }
        let priority = ctx.r9() as i32;
        self.decode_variant(ctx, Some(priority), out)
    }

    /// Destroy(void *state)
    fn destroy(&self, ctx: &mut CpuContext) -> HleResult {
        let state = ctx.rdi();
        self.with_state(|s| if s.decoders.remove(&state).is_some() { 0 } else { OPUSDEC_ERROR_INVALID_ARGUMENT })
    }
}

fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    OPUS.initialize(ctx)
}
fn cb_terminate(ctx: &mut CpuContext) -> HleResult {
    OPUS.terminate(ctx)
}
fn cb_get_size(ctx: &mut CpuContext) -> HleResult {
    OPUS.get_size(ctx)
}
fn cb_create_ex(ctx: &mut CpuContext) -> HleResult {
    OPUS.create_ex(ctx)
}
fn cb_decode(ctx: &mut CpuContext) -> HleResult {
    OPUS.decode(ctx)
}
fn cb_destroy(ctx: &mut CpuContext) -> HleResult {
    OPUS.destroy(ctx)
}

fn celt_initialize(ctx: &mut CpuContext) -> HleResult {
    OPUS_CELT.initialize(ctx)
}
fn celt_terminate(ctx: &mut CpuContext) -> HleResult {
    OPUS_CELT.terminate(ctx)
}
fn celt_get_size(ctx: &mut CpuContext) -> HleResult {
    OPUS_CELT.get_size(ctx)
}
fn celt_create_ex(ctx: &mut CpuContext) -> HleResult {
    OPUS_CELT.create_ex(ctx)
}
fn celt_decode(ctx: &mut CpuContext) -> HleResult {
    OPUS_CELT.decode(ctx)
}
fn cb_decode_with_priority(ctx: &mut CpuContext) -> HleResult {
    let p = ctx.r9() as i32;
    OPUS.decode_variant(ctx, Some(p), PcmOut::S16)
}
fn cb_decode_float(ctx: &mut CpuContext) -> HleResult {
    OPUS.decode_variant(ctx, None, PcmOut::F32)
}
fn cb_decode_float_with_priority(ctx: &mut CpuContext) -> HleResult {
    let p = ctx.r9() as i32;
    OPUS.decode_variant(ctx, Some(p), PcmOut::F32)
}
fn cb_decode_with_priority_and_timeout(ctx: &mut CpuContext) -> HleResult {
    OPUS.decode_with_timeout(ctx, PcmOut::S16)
}
fn cb_decode_float_with_priority_and_timeout(ctx: &mut CpuContext) -> HleResult {
    OPUS.decode_with_timeout(ctx, PcmOut::F32)
}
fn celt_decode_with_priority(ctx: &mut CpuContext) -> HleResult {
    let p = ctx.r9() as i32;
    OPUS_CELT.decode_variant(ctx, Some(p), PcmOut::S16)
}
fn celt_decode_float(ctx: &mut CpuContext) -> HleResult {
    OPUS_CELT.decode_variant(ctx, None, PcmOut::F32)
}
fn celt_decode_float_with_priority(ctx: &mut CpuContext) -> HleResult {
    let p = ctx.r9() as i32;
    OPUS_CELT.decode_variant(ctx, Some(p), PcmOut::F32)
}
fn celt_destroy(ctx: &mut CpuContext) -> HleResult {
    OPUS_CELT.destroy(ctx)
}

const LIB: &str = "libSceOpusDec";
const LIB_CELT: &str = "libSceOpusCeltDec";

pub(crate) static EXPORTS: &[Export] = &[
    ("2dA75YfXNrQ", "sceOpusDecInitialize", LIB, cb_initialize),
    ("qOpZPAFEvIs", "sceOpusDecTerminate", LIB, cb_terminate),
    ("knmEFpLg-Rk", "sceOpusDecGetSize", LIB, cb_get_size),
    ("KAIGiqYrtI4", "sceOpusDecCreateEx", LIB, cb_create_ex),
    ("+CSUx-budSk", "sceOpusDecDecode", LIB, cb_decode),
    ("290AWkA-lck", "sceOpusDecDestroy", LIB, cb_destroy),
    ("vkBmwJuOKgY", "sceOpusDecDecodeWithPriority", LIB, cb_decode_with_priority),
    ("2U9KB5G-W2w", "sceOpusDecDecodeFloat", LIB, cb_decode_float),
    ("DjibZOK00TU", "sceOpusDecDecodeFloatWithPriority", LIB, cb_decode_float_with_priority),
    ("eFhsGSf+Hbw", "sceOpusDecDecodeWithPriorityAndTimeout", LIB, cb_decode_with_priority_and_timeout),
    ("x2nJkPwKx04", "sceOpusDecDecodeFloatWithPriorityAndTimeout", LIB, cb_decode_float_with_priority_and_timeout),
    ("VMOyIG83Jtg", "sceOpusCeltDecInitialize", LIB_CELT, celt_initialize),
    ("pkri5OAPsJk", "sceOpusCeltDecTerminate", LIB_CELT, celt_terminate),
    ("xUp4-mftj5I", "sceOpusCeltDecGetSize", LIB_CELT, celt_get_size),
    ("z6eJnrUEaVY", "sceOpusCeltDecCreateEx", LIB_CELT, celt_create_ex),
    ("6fxZHeuNhIA", "sceOpusCeltDecDecode", LIB_CELT, celt_decode),
    ("Sn-MpfGTN9w", "sceOpusCeltDecDestroy", LIB_CELT, celt_destroy),
    ("q76RU0c+L+4", "sceOpusCeltDecDecodeWithPriority", LIB_CELT, celt_decode_with_priority),
    ("zKQEeM0cBVs", "sceOpusCeltDecDecodeFloat", LIB_CELT, celt_decode_float),
    ("GF5cVgFKg4g", "sceOpusCeltDecDecodeFloatWithPriority", LIB_CELT, celt_decode_float_with_priority),
];

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::context::CpuRegister;

    const BASE: u64 = 0x5EA0_0000_0000;

    fn call(ctx: &mut CpuContext, f: fn(&mut CpuContext) -> HleResult, args: &[u64]) -> HleResult {
        let regs = [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx, CpuRegister::R8];
        for (r, v) in regs.iter().zip(args) {
            ctx.set(*r, *v);
        }
        f(ctx)
    }

    #[test]
    fn packet_durations() {
        // SILK 10 ms, CELT 2.5 ms, 2 frames, code 3 with a frame count.
        assert_eq!(packet_samples(&[0x00]), Some(480));
        assert_eq!(packet_samples(&[0x80]), Some(120));
        assert_eq!(packet_samples(&[0x81]), Some(240));
        assert_eq!(packet_samples(&[0x7C]), Some(960));
        assert_eq!(packet_samples(&[0x7F, 0x03]), Some(3 * 960));
        assert_eq!(packet_samples(&[0x7F]), None, "code 3 without a frame count");
        assert_eq!(packet_samples(&[0x7F, 0]), None, "no frames");
        assert_eq!(packet_samples(&[0x7F, 7]), None, "more than 120 ms");
        assert_eq!(packet_samples(&[]), None);
    }

    #[test]
    fn contexts_and_decoders() {
        let mut ctx = CpuContext::new(crate::sync::test_util::vm_arena(BASE));
        let (context, other, state, state2) = (BASE, BASE + 4, BASE + 0x1000, BASE + 0x2000);
        assert_eq!(call(&mut ctx, cb_initialize, &[0]), OPUSDEC_ERROR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_initialize, &[context]), 0);
        assert_eq!(call(&mut ctx, cb_initialize, &[other]), 0);
        assert_ne!(ctx.read_u32(context), ctx.read_u32(other), "contexts share an id");
        assert_eq!(call(&mut ctx, cb_initialize, &[context]), OPUSDEC_ERROR_INVALID_ARGUMENT, "context initialized twice");
        for (channels, size) in [(1, STATE_SIZE), (2, STATE_SIZE), (3, OPUSDEC_ERROR_INVALID_ARGUMENT), (0, OPUSDEC_ERROR_INVALID_ARGUMENT)] {
            assert_eq!(call(&mut ctx, cb_get_size, &[channels]), size);
        }
        for rate in [0, 22050, 44100, 96000] {
            assert_eq!(call(&mut ctx, cb_create_ex, &[context, state, rate, 2]), OPUSDEC_ERROR_INVALID_ARGUMENT);
        }
        assert_eq!(call(&mut ctx, cb_create_ex, &[context, 0, 48000, 2]), OPUSDEC_ERROR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_create_ex, &[context, state, 48000, 3]), OPUSDEC_ERROR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_destroy, &[state]), OPUSDEC_ERROR_INVALID_ARGUMENT);
        if cfg!(ps5_ffmpeg) {
            assert_eq!(call(&mut ctx, cb_create_ex, &[context, state, 48000, 2]), 0);
            assert_eq!(call(&mut ctx, cb_create_ex, &[context, state, 48000, 2]), OPUSDEC_ERROR_INVALID_ARGUMENT, "state reused");
            assert_eq!(call(&mut ctx, cb_terminate, &[context]), OPUSDEC_ERROR_NOT_IMPLEMENTED, "terminated with a live decoder");
            assert_eq!(call(&mut ctx, cb_create_ex, &[other, state2, 48000, 1]), 0);
            assert_eq!(call(&mut ctx, cb_destroy, &[state]), 0);
            assert_eq!(call(&mut ctx, cb_destroy, &[state]), OPUSDEC_ERROR_INVALID_ARGUMENT);
            assert_eq!(call(&mut ctx, cb_destroy, &[state2]), 0);
        }
        assert_eq!(call(&mut ctx, cb_terminate, &[context]), 0);
        assert_eq!(call(&mut ctx, cb_terminate, &[context]), OPUSDEC_ERROR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_create_ex, &[context, state, 48000, 2]), OPUSDEC_ERROR_INVALID_ARGUMENT, "terminated context");
        assert_eq!(call(&mut ctx, cb_terminate, &[other]), 0);
    }

    pub(crate) const OPUS_STEREO: [u8; 276] = [
        0x46, 0x00, 0x7C, 0x87, 0xFC, 0xB1, 0x1D, 0xC0, 0xE3, 0x07, 0xD5, 0x9C, 0x4D, 0x91, 0x62, 0x6B,
        0x73, 0x86, 0x05, 0xE8, 0xDB, 0xC6, 0x23, 0x6D, 0x5E, 0x6F, 0x73, 0xF8, 0xF2, 0x47, 0xD9, 0x5E,
        0xEA, 0xB3, 0xD2, 0x74, 0x6C, 0xE0, 0xCE, 0xE2, 0x3C, 0xFA, 0x59, 0x4A, 0xBA, 0x8F, 0x76, 0x0F,
        0x15, 0xE1, 0x3E, 0x60, 0xDF, 0x80, 0xC5, 0x44, 0xB1, 0x1B, 0xEF, 0x43, 0x03, 0x4F, 0xF2, 0xDE,
        0xE3, 0xAD, 0x8B, 0x20, 0xFF, 0x68, 0x0C, 0x0A, 0x3F, 0x00, 0x7C, 0x87, 0xFD, 0x45, 0xBD, 0x12,
        0x00, 0xA5, 0xB1, 0xA0, 0x0A, 0x62, 0x24, 0x6F, 0x63, 0x70, 0xA5, 0x35, 0x13, 0x03, 0x74, 0x0D,
        0x83, 0xC9, 0x74, 0x1B, 0x79, 0xED, 0xA3, 0x09, 0x83, 0xA5, 0x02, 0xC4, 0x60, 0x49, 0xC9, 0xB6,
        0x9A, 0x60, 0xAD, 0x6C, 0x77, 0x84, 0xE5, 0xC2, 0xCE, 0x2E, 0x71, 0x07, 0x7E, 0x40, 0xCB, 0xDC,
        0x13, 0xDF, 0xCF, 0x84, 0x5D, 0x55, 0x3B, 0x6B, 0x6F, 0x44, 0x00, 0x7C, 0x88, 0x01, 0xE8, 0xA1,
        0x2A, 0x12, 0x7D, 0x5E, 0xF6, 0x74, 0x78, 0xB1, 0x27, 0x2B, 0x8C, 0xF0, 0x7F, 0x95, 0x51, 0x71,
        0x28, 0x18, 0xEA, 0x66, 0xEB, 0xCA, 0xAC, 0xDF, 0x6C, 0x9C, 0xFD, 0xED, 0x88, 0x8E, 0x66, 0xD7,
        0xDA, 0x36, 0xD7, 0xEB, 0x5F, 0xA6, 0x05, 0x72, 0xDA, 0x52, 0x14, 0x7F, 0xF5, 0x84, 0x54, 0xA1,
        0xA9, 0xF1, 0xAA, 0xA8, 0x73, 0x9A, 0xCC, 0x91, 0x83, 0x92, 0xD0, 0x7F, 0x3B, 0x6B, 0x65, 0x43,
        0x00, 0x7C, 0x88, 0x01, 0xE8, 0xA1, 0x2A, 0x12, 0x7D, 0x5F, 0x04, 0xF8, 0xF5, 0x1F, 0xA0, 0x85,
        0x89, 0xED, 0xBA, 0x7C, 0x5F, 0x63, 0x2A, 0x9E, 0x05, 0xE8, 0x63, 0xD1, 0x73, 0x0C, 0xAF, 0xC8,
        0xC9, 0x22, 0xF7, 0x11, 0x1D, 0x8B, 0xA9, 0x9A, 0x90, 0x77, 0xF2, 0x86, 0x49, 0x15, 0x6E, 0xD6,
        0x34, 0x1F, 0x50, 0x15, 0xEC, 0x85, 0xC7, 0xF5, 0xA7, 0xFA, 0x99, 0xF9, 0x47, 0x32, 0x07, 0xAF,
        0x24, 0xBF, 0x14, 0xC5,
    ];

    /// The priority and float forms: a priority outside 20..=100 is refused
    /// before anything is decoded, a finite timeout is not implemented, and
    /// the float output is the S16 samples over 32768 (four bytes each).
    #[test]
    fn priority_and_float_variants() {
        let mut ctx = CpuContext::new(crate::sync::test_util::vm_arena(BASE + 0x50_0000));
        let b = BASE + 0x50_0000;
        let (context, state, reference_state, data, pcm, reference) = (b, b + 0x100, b + 0x400, b + 0x1000, b + 0x2000, b + 0x6000);
        ctx.write_bytes(data, &OPUS_STEREO);
        let bytes = OPUS_STEREO[0] as u64 | (OPUS_STEREO[1] as u64) << 8;
        let with_priority = |ctx: &mut CpuContext, f: fn(&mut CpuContext) -> HleResult, s: u64, out: u64, cap: u64, p: i32| {
            ctx.set(CpuRegister::R9, p as u32 as u64);
            call(ctx, f, &[s, data + 2, bytes, out, cap])
        };
        for p in [19, 101, -1] {
            assert_eq!(with_priority(&mut ctx, cb_decode_with_priority, state, pcm, 3840, p), OPUSDEC_ERROR_INVALID_ARGUMENT, "{p}");
            assert_eq!(with_priority(&mut ctx, cb_decode_float_with_priority, state, pcm, 7680, p), OPUSDEC_ERROR_INVALID_ARGUMENT, "{p}");
        }
        // A finite timeout (the first stack argument).
        let rsp = b + 0x9000;
        ctx.set(CpuRegister::Rsp, rsp);
        ctx.write_u64(rsp + 8, 100);
        assert_eq!(with_priority(&mut ctx, cb_decode_with_priority_and_timeout, state, pcm, 3840, 40), OPUSDEC_ERROR_NOT_IMPLEMENTED);
        if !cfg!(ps5_ffmpeg) {
            return;
        }
        assert_eq!(call(&mut ctx, cb_initialize, &[context]), 0);
        for s in [state, reference_state] {
            assert_eq!(call(&mut ctx, cb_create_ex, &[context, s, 48000, 2]), 0);
        }
        // Too small for 960 stereo float samples.
        assert_eq!(with_priority(&mut ctx, cb_decode_float_with_priority, state, pcm, 7679, 40), OPUSDEC_ERROR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_decode_float, &[state, data + 2, bytes, pcm, 7680]), 7680);
        assert_eq!(with_priority(&mut ctx, cb_decode_with_priority, reference_state, reference, 3840, 20), 3840);
        let floats = ctx.read_bytes(pcm, 7680).unwrap();
        let shorts = ctx.read_bytes(reference, 3840).unwrap();
        for (f, s) in floats.chunks_exact(4).zip(shorts.chunks_exact(2)) {
            assert_eq!(f32::from_le_bytes(f.try_into().unwrap()), i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0);
        }
        ctx.write_u64(rsp + 8, u32::MAX as u64);
        let next = data + 2 + bytes;
        let next_bytes = ctx.read_bytes(next, 2).map(|v| v[0] as u64 | (v[1] as u64) << 8).unwrap();
        ctx.set(CpuRegister::R9, 100);
        assert_eq!(call(&mut ctx, cb_decode_float_with_priority_and_timeout, &[state, next + 2, next_bytes, pcm, 7680]), 7680);
        for s in [state, reference_state] {
            assert_eq!(call(&mut ctx, cb_destroy, &[s]), 0);
        }
        assert_eq!(call(&mut ctx, cb_terminate, &[context]), 0);
    }

    /// Three stereo 20 ms packets, each prefixed with its 16-bit length.
    #[test]
    fn decodes_stereo_and_mono_packets() {
        if !cfg!(ps5_ffmpeg) {
            return;
        }
        let mut ctx = CpuContext::new(crate::sync::test_util::vm_arena(BASE + 0x10_0000));
        let b = BASE + 0x10_0000;
        let (context, state, reference_state, mono_state, data) = (b, b + 0x100, b + 0x400, b + 0x700, b + 0x1000);
        let (pcm, reference, mono) = (b + 0x2000, b + 0x4000, b + 0x6000);
        ctx.write_bytes(data, &OPUS_STEREO);
        assert_eq!(call(&mut ctx, cb_initialize, &[context]), 0);
        for (s, channels) in [(state, 2), (reference_state, 2), (mono_state, 1)] {
            assert_eq!(call(&mut ctx, cb_create_ex, &[context, s, 48000, channels]), 0);
        }
        // Rejected packets write nothing.
        ctx.write_bytes(pcm, &[0x39; 3844]);
        assert_eq!(call(&mut ctx, cb_decode, &[state, data + 2, 70, pcm + 2, 3839]), OPUSDEC_ERROR_INVALID_ARGUMENT, "buffer too small");
        ctx.write_bytes(b + 0x1800, &[0x7F]);
        assert_eq!(call(&mut ctx, cb_decode, &[state, b + 0x1800, 1, pcm + 2, 3840]), OPUSDEC_ERROR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_decode, &[state, 0, 0, pcm + 2, 3840]), OPUSDEC_ERROR_INVALID_ARGUMENT, "packet loss concealment");
        assert!(ctx.read_bytes(pcm, 3844).unwrap().iter().all(|&v| v == 0x39), "a rejected packet wrote PCM");
        let mut offset = 0usize;
        for frame in 0..3 {
            let bytes = OPUS_STEREO[offset] as u64 | (OPUS_STEREO[offset + 1] as u64) << 8;
            let packet = data + offset as u64 + 2;
            assert_eq!(call(&mut ctx, cb_decode, &[state, packet, bytes, pcm + 2, 3841]), 3840);
            assert_eq!(ctx.read_bytes(pcm, 2).unwrap(), [0x39, 0x39]);
            assert_eq!(ctx.read_bytes(pcm + 3842, 2).unwrap(), [0x39, 0x39], "wrote past the packet's PCM");
            assert_eq!(call(&mut ctx, cb_decode, &[reference_state, packet, bytes, reference, 3840]), 3840);
            assert_eq!(ctx.read_bytes(pcm + 2, 3840), ctx.read_bytes(reference, 3840), "decoders share state");
            if frame > 0 {
                let peak = ctx
                    .read_bytes(pcm + 2, 3840)
                    .unwrap()
                    .chunks_exact(2)
                    .map(|c| (i16::from_le_bytes([c[0], c[1]]) as i32).abs())
                    .max()
                    .unwrap();
                assert!(peak > 900 && peak < 1200, "frame {frame} peak {peak}");
            }
            assert_eq!(call(&mut ctx, cb_decode, &[mono_state, packet, bytes, mono, 1921]), 1920);
            assert!(ctx.read_bytes(mono, 1920).unwrap().iter().any(|&v| v != 0));
            offset += bytes as usize + 2;
        }
        // Two 20 ms frames whose framing the decoder rejects.
        ctx.write_bytes(b + 0x1810, &[0x7D, 1]);
        ctx.write_bytes(pcm, &[0x39; 7684]);
        assert_eq!(call(&mut ctx, cb_decode, &[state, b + 0x1810, 2, pcm + 2, 7680]), OPUSDEC_ERROR_INVALID_ARGUMENT);
        assert!(ctx.read_bytes(pcm, 7684).unwrap().iter().all(|&v| v == 0x39), "a rejected packet wrote PCM");
        for s in [state, reference_state, mono_state] {
            assert_eq!(call(&mut ctx, cb_destroy, &[s]), 0);
        }
        assert_eq!(call(&mut ctx, cb_decode, &[state, data + 2, 70, pcm, 3840]), OPUSDEC_ERROR_INVALID_ARGUMENT, "destroyed decoder");
        assert_eq!(call(&mut ctx, cb_terminate, &[context]), 0);
    }
    const CELT_STEREO: [u8; 318] = [
        0x75, 0x00, 0xFC, 0xB5, 0x1F, 0xBC, 0xC8, 0xCA, 0x0B, 0xC2, 0x18, 0x1A, 0xFB, 0xDB, 0x3E, 0x2B,
        0x6B, 0x0A, 0x67, 0x71, 0x25, 0x2F, 0x51, 0x57, 0x4F, 0x78, 0xB3, 0x1F, 0xE9, 0xF3, 0x99, 0x51,
        0xE9, 0x70, 0x72, 0x23, 0x99, 0x3E, 0x26, 0xA3, 0xF1, 0x8F, 0x02, 0xB3, 0xAA, 0xBA, 0xAB, 0x7E,
        0xA5, 0x6A, 0xBA, 0x33, 0xCA, 0xE9, 0x43, 0x4D, 0x41, 0xC9, 0x90, 0x3C, 0xD0, 0x5F, 0x55, 0xC1,
        0x54, 0xA8, 0xD0, 0x6F, 0x71, 0x4F, 0xCB, 0x95, 0xDF, 0x09, 0xC9, 0xB6, 0xAA, 0xA1, 0x8D, 0xD4,
        0x2B, 0xC0, 0xF7, 0x23, 0xDD, 0xF5, 0x54, 0xEF, 0x90, 0xC4, 0xD3, 0x47, 0xC1, 0x91, 0x2D, 0xA8,
        0x3D, 0x53, 0xF4, 0x7D, 0xFF, 0xBD, 0x47, 0x0E, 0xD2, 0x95, 0x25, 0x42, 0x66, 0x02, 0xD7, 0x80,
        0x00, 0x67, 0x59, 0x13, 0x60, 0x2F, 0x6C, 0x60, 0x00, 0xFC, 0xB3, 0xF1, 0x8F, 0xA9, 0xA8, 0x43,
        0x9D, 0xC0, 0x23, 0xBE, 0x5B, 0x38, 0xE8, 0xF8, 0xB8, 0x85, 0xBE, 0xD2, 0x2F, 0x40, 0x53, 0x40,
        0xEF, 0x24, 0xE5, 0xA6, 0x91, 0xF4, 0x9B, 0x73, 0xC5, 0x61, 0x61, 0x08, 0x3F, 0x56, 0x7D, 0x7A,
        0x2A, 0xBC, 0xAF, 0x3A, 0x70, 0xDD, 0x5E, 0x0B, 0x12, 0xBD, 0xC5, 0xEA, 0x69, 0x63, 0x65, 0x36,
        0x5E, 0x27, 0x15, 0xBA, 0xB0, 0xE3, 0x62, 0xE7, 0x4F, 0x54, 0xFA, 0xD2, 0x49, 0x7E, 0xBD, 0xE4,
        0xDA, 0x9E, 0x2F, 0x90, 0x9E, 0x91, 0x84, 0x6F, 0x95, 0xCE, 0x2D, 0x25, 0x63, 0x23, 0x57, 0x1C,
        0x00, 0x20, 0x79, 0xE7, 0xEE, 0xBB, 0x91, 0x7F, 0xEE, 0x63, 0x00, 0xFC, 0xB2, 0x91, 0x0B, 0x44,
        0x10, 0xE2, 0xD2, 0x82, 0x28, 0x1C, 0x96, 0x98, 0x61, 0xB7, 0x87, 0x46, 0x4A, 0x96, 0xA1, 0x41,
        0xE7, 0x5F, 0x84, 0xA1, 0x73, 0x0E, 0x28, 0x29, 0xD0, 0x65, 0x9C, 0x99, 0x33, 0xA0, 0x7E, 0x63,
        0xB9, 0x51, 0xED, 0x67, 0xCB, 0x5D, 0xD2, 0x22, 0xC8, 0x0C, 0x59, 0x72, 0xB5, 0x37, 0xA9, 0xB5,
        0x24, 0x25, 0xB3, 0xA1, 0x09, 0x8B, 0xBA, 0x9E, 0x6C, 0x5C, 0x23, 0x7C, 0x7B, 0x19, 0xDB, 0x45,
        0xCF, 0x19, 0xC5, 0x1F, 0x79, 0xCC, 0x16, 0x07, 0xA5, 0x4C, 0xD8, 0xA0, 0x4B, 0x48, 0x8F, 0xAE,
        0x17, 0x7C, 0x28, 0xF0, 0x03, 0xFF, 0xF9, 0xE1, 0x80, 0x55, 0x44, 0x44, 0x7F, 0xEE,
    ];

    const CELT_TONE: [[u8; 60]; 4] = [
        [
            0xFC, 0x9F, 0xF9, 0x33, 0x84, 0xFB, 0x41, 0x76, 0xA8, 0x21, 0xCE, 0x57, 0xBB, 0x16, 0x49, 0x05,
            0x21, 0x3F, 0x2A, 0xC4, 0x79, 0xD0, 0x73, 0x62, 0x89, 0x7E, 0xF9, 0x29, 0x75, 0x1E, 0xAC, 0x80,
            0xAE, 0x30, 0xF1, 0x46, 0x18, 0x1F, 0xF8, 0x8B, 0xD0, 0xC4, 0x37, 0xF1, 0x12, 0x47, 0x01, 0xF9,
            0x43, 0xA4, 0x1A, 0x8C, 0x80, 0x40, 0xF0, 0x0A, 0xC3, 0xFD, 0x4C, 0xB1,
        ],
        [
            0xFC, 0x9E, 0xA4, 0xC8, 0xDF, 0x47, 0x28, 0xAE, 0x34, 0xFD, 0x32, 0x11, 0xB9, 0x69, 0x99, 0x7B,
            0x61, 0xC6, 0x21, 0x94, 0x0E, 0x20, 0xED, 0xC2, 0x95, 0x07, 0x78, 0x0D, 0x00, 0x92, 0x42, 0x12,
            0xD4, 0xE9, 0xEC, 0x5E, 0xDC, 0xF4, 0xB9, 0x7A, 0x2E, 0x23, 0xA1, 0xC6, 0x4C, 0xD7, 0x69, 0xEC,
            0x34, 0x8F, 0x78, 0x21, 0x35, 0xF1, 0x1B, 0xB0, 0xF0, 0xF5, 0x5C, 0xD1,
        ],
        [
            0xFC, 0x9E, 0xA6, 0x12, 0xDC, 0x36, 0xC3, 0x4B, 0x2C, 0x3D, 0x85, 0x17, 0x32, 0xE1, 0x56, 0x62,
            0xFE, 0x39, 0xDB, 0x85, 0xFD, 0x83, 0x0D, 0x90, 0xF9, 0x32, 0xEB, 0xEB, 0x38, 0xFD, 0x3D, 0x96,
            0x26, 0x21, 0xE6, 0x38, 0x6A, 0x8F, 0xDD, 0x1F, 0x68, 0xBD, 0xAF, 0x4F, 0x9D, 0x08, 0x16, 0xED,
            0x7B, 0x2A, 0x62, 0x4C, 0x7A, 0x85, 0xAB, 0x7B, 0x0F, 0x3D, 0x43, 0xD1,
        ],
        [
            0xFC, 0x7F, 0xDF, 0x09, 0x18, 0xD2, 0x71, 0xA1, 0xE6, 0x6C, 0x70, 0xED, 0xCA, 0x2F, 0x47, 0x79,
            0xA7, 0xEA, 0xE3, 0xD1, 0x9B, 0xCD, 0x9E, 0x87, 0xE5, 0xE0, 0x0A, 0x86, 0x0A, 0xEE, 0x8E, 0x63,
            0xE4, 0x77, 0x45, 0xE9, 0xF0, 0x82, 0xF8, 0x2A, 0x13, 0xD2, 0xDB, 0x17, 0x11, 0x2E, 0x36, 0x10,
            0x03, 0x30, 0x20, 0x00, 0xF7, 0x42, 0xD8, 0x71, 0x7E, 0xB0, 0xF3, 0x3C,
        ],
    ];

    fn celt_call(ctx: &mut CpuContext, f: fn(&mut CpuContext) -> HleResult, args: &[u64]) -> HleResult {
        call(ctx, f, args)
    }

    fn read_s16(ctx: &CpuContext, addr: u64, count: usize) -> Vec<i16> {
        ctx.read_bytes(addr, count * 2).unwrap().chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
    }

    /// A 1 kHz tone at a low level: peak and zero crossings per channel.
    fn assert_tone(pcm: &[i16], samples: usize, channels: usize, rate: usize) {
        for channel in 0..channels {
            let (mut peak, mut crossings) = (0i32, 0usize);
            for s in 0..samples {
                let v = pcm[s * channels + channel];
                peak = peak.max((v as i32).abs());
                if s > 0 && (v < 0) != (pcm[(s - 1) * channels + channel] < 0) {
                    crossings += 1;
                }
            }
            assert!(peak > 100 && peak < 200, "peak {peak} at {rate} Hz");
            let expected = samples * 2000 / rate;
            assert!(crossings + 4 >= expected && crossings <= expected + 4, "{crossings} crossings at {rate} Hz");
        }
    }

    #[test]
    fn celt_contexts_and_validation() {
        let b = BASE + 0x20_0000;
        let mut ctx = CpuContext::new(crate::sync::test_util::vm_arena(b));
        let (context, state) = (b, b + 0x100);
        assert_eq!(celt_call(&mut ctx, celt_initialize, &[context]), 0);
        assert_eq!(celt_call(&mut ctx, celt_initialize, &[context]), OPUSDEC_ERROR_INVALID_ARGUMENT, "initialized twice");
        assert_eq!(celt_call(&mut ctx, celt_get_size, &[1]), STATE_SIZE);
        assert_eq!(celt_call(&mut ctx, celt_get_size, &[2]), STATE_SIZE);
        assert_eq!(celt_call(&mut ctx, celt_get_size, &[3]), OPUSDEC_ERROR_INVALID_ARGUMENT);
        for rate in [0, 22050, 44100, 96000] {
            assert_eq!(celt_call(&mut ctx, celt_create_ex, &[context, state, rate, 2]), OPUSDEC_ERROR_INVALID_ARGUMENT);
        }
        assert_eq!(celt_call(&mut ctx, celt_destroy, &[state]), OPUSDEC_ERROR_INVALID_ARGUMENT);
        assert_eq!(celt_call(&mut ctx, celt_terminate, &[context]), 0);
        assert_eq!(celt_call(&mut ctx, celt_terminate, &[context]), OPUSDEC_ERROR_INVALID_ARGUMENT);
        assert_eq!(celt_call(&mut ctx, celt_create_ex, &[context, state, 48000, 2]), OPUSDEC_ERROR_INVALID_ARGUMENT);
    }

    #[test]
    fn celt_decodes_stereo_mono_and_tones_at_every_rate() {
        if !cfg!(ps5_ffmpeg) {
            return;
        }
        let b = BASE + 0x30_0000;
        let mut ctx = CpuContext::new(crate::sync::test_util::vm_arena(b));
        let (context, state, mono_state, data, pcm, mono) = (b, b + 0x100, b + 0x400, b + 0x1000, b + 0x2000, b + 0x4000);
        let (hybrid, tone_data, tone_state) = (b + 0x1800, b + 0x1900, b + 0x700);
        ctx.write_bytes(data, &CELT_STEREO);
        ctx.write_bytes(hybrid, &[0x7C, 0x87, 0xFC, 0xB1]);
        assert_eq!(celt_call(&mut ctx, celt_initialize, &[context]), 0);
        assert_eq!(celt_call(&mut ctx, celt_create_ex, &[context, state, 48000, 2]), 0);
        assert_eq!(celt_call(&mut ctx, celt_create_ex, &[context, mono_state, 48000, 1]), 0);
        assert_eq!(celt_call(&mut ctx, celt_create_ex, &[context, state, 48000, 2]), OPUSDEC_ERROR_INVALID_ARGUMENT, "state reused");
        assert_eq!(celt_call(&mut ctx, celt_terminate, &[context]), OPUSDEC_ERROR_NOT_IMPLEMENTED, "live decoders");

        // A hybrid packet is not CELT-only: nothing is written.
        ctx.write_bytes(pcm, &[0x39; 3844]);
        assert_eq!(celt_call(&mut ctx, celt_decode, &[state, hybrid, 4, pcm + 2, 3840]), OPUSDEC_ERROR_NOT_IMPLEMENTED);
        assert!(ctx.read_bytes(pcm, 3844).unwrap().iter().all(|&v| v == 0x39));

        let mut offset = 0usize;
        let mut audible = false;
        for _ in 0..3 {
            let bytes = CELT_STEREO[offset] as u64 | (CELT_STEREO[offset + 1] as u64) << 8;
            let packet = data + offset as u64 + 2;
            ctx.write_bytes(pcm, &[0x39; 3844]);
            assert_eq!(celt_call(&mut ctx, celt_decode, &[state, packet, bytes, pcm + 2, 3839]), OPUSDEC_ERROR_INVALID_ARGUMENT);
            assert_eq!(celt_call(&mut ctx, celt_decode, &[state, packet, bytes, pcm + 2, 3840]), 3840);
            assert_eq!(ctx.read_bytes(pcm, 2).unwrap(), [0x39, 0x39]);
            assert_eq!(ctx.read_bytes(pcm + 3842, 2).unwrap(), [0x39, 0x39]);
            audible |= read_s16(&ctx, pcm + 2, 1920).iter().any(|&v| v != 0);
            ctx.write_bytes(mono, &[0x39; 1924]);
            assert_eq!(celt_call(&mut ctx, celt_decode, &[mono_state, packet, bytes, mono + 2, 1920]), 1920);
            assert_eq!(ctx.read_bytes(mono, 2).unwrap(), [0x39, 0x39]);
            assert_eq!(ctx.read_bytes(mono + 1922, 2).unwrap(), [0x39, 0x39]);
            offset += bytes as usize + 2;
        }
        assert!(audible);

        for rate in [8000u64, 12000, 16000, 24000, 48000] {
            assert_eq!(celt_call(&mut ctx, celt_create_ex, &[context, tone_state, rate, 2]), 0);
            let samples = (960 * rate / 48000) as usize;
            for (frame, tone) in CELT_TONE.iter().enumerate() {
                ctx.write_bytes(tone_data, tone);
                ctx.write_bytes(pcm, &vec![0x39; samples * 4 + 4]);
                let cap = (samples * 4) as u64;
                assert_eq!(celt_call(&mut ctx, celt_decode, &[tone_state, tone_data, 60, pcm + 2, cap - 1]), OPUSDEC_ERROR_INVALID_ARGUMENT);
                assert_eq!(celt_call(&mut ctx, celt_decode, &[tone_state, tone_data, 60, pcm + 2, cap]), cap as i64);
                assert_eq!(ctx.read_bytes(pcm, 2).unwrap(), [0x39, 0x39]);
                assert_eq!(ctx.read_bytes(pcm + 2 + cap, 2).unwrap(), [0x39, 0x39]);
                if frame == 1 || frame == 2 {
                    assert_tone(&read_s16(&ctx, pcm + 2, samples * 2), samples, 2, rate as usize);
                }
            }
            assert_eq!(celt_call(&mut ctx, celt_destroy, &[tone_state]), 0);
        }

        assert_eq!(celt_call(&mut ctx, celt_destroy, &[state]), 0);
        assert_eq!(celt_call(&mut ctx, celt_destroy, &[state]), OPUSDEC_ERROR_INVALID_ARGUMENT);
        assert_eq!(celt_call(&mut ctx, celt_decode, &[state, data + 2, CELT_STEREO[0] as u64, pcm, 3840]), OPUSDEC_ERROR_INVALID_ARGUMENT);
        assert_eq!(celt_call(&mut ctx, celt_destroy, &[mono_state]), 0);
        assert_eq!(celt_call(&mut ctx, celt_terminate, &[context]), 0);
    }

    /// The OpusDec stream resampled to the lower rates keeps its level.
    #[test]
    fn decodes_resampled_rates() {
        if !cfg!(ps5_ffmpeg) {
            return;
        }
        let b = BASE + 0x40_0000;
        let mut ctx = CpuContext::new(crate::sync::test_util::vm_arena(b));
        let (context, state, data, pcm) = (b, b + 0x100, b + 0x1000, b + 0x2000);
        ctx.write_bytes(data, &OPUS_STEREO);
        assert_eq!(call(&mut ctx, cb_initialize, &[context]), 0);
        for rate in [8000u64, 12000, 16000, 24000] {
            assert_eq!(call(&mut ctx, cb_create_ex, &[context, state, rate, 2]), 0);
            let bytes_out = (960 * rate / 48000) * 4;
            let (mut at, mut peak) = (0usize, 0i32);
            for _ in 0..3 {
                let bytes = OPUS_STEREO[at] as u64 | (OPUS_STEREO[at + 1] as u64) << 8;
                let packet = data + at as u64 + 2;
                ctx.write_bytes(pcm, &vec![0x39; bytes_out as usize + 4]);
                assert_eq!(call(&mut ctx, cb_decode, &[state, packet, bytes, pcm + 2, bytes_out - 1]), OPUSDEC_ERROR_INVALID_ARGUMENT);
                assert_eq!(call(&mut ctx, cb_decode, &[state, packet, bytes, pcm + 2, bytes_out]), bytes_out as i64);
                assert_eq!(ctx.read_bytes(pcm + 2 + bytes_out, 2).unwrap(), [0x39, 0x39]);
                for c in ctx.read_bytes(pcm + 2, bytes_out as usize).unwrap().chunks_exact(2) {
                    peak = peak.max((i16::from_le_bytes([c[0], c[1]]) as i32).abs());
                }
                at += bytes as usize + 2;
            }
            assert!(peak > 900 && peak < 1200, "peak {peak} at {rate} Hz");
            assert_eq!(call(&mut ctx, cb_destroy, &[state]), 0);
        }
        assert_eq!(call(&mut ctx, cb_terminate, &[context]), 0);
    }
}
