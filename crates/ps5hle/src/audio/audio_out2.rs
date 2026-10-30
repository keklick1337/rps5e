// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceAudioOut2: contexts, ports, users and speaker arrays.
//!
//! A context renders grains of `numGrains` samples at 48 kHz. Every bed
//! port (MAIN, BGM, VOICE, PADSPK, ...) owns an AudioOut port in its own
//! format; object ports (0x0100, mono) are mixed into one stereo float
//! AudioOut port per context, centred or decoded as first-order
//! ambisonics, scaled by the object's gain. Bed ports forward their
//! per-channel gains to AudioOut, which applies them on output.
//!
//! PCM attached with `PortSetAttributes(PCM)` is copied and queued: every
//! buffer set between two pushes plays, one per grain, in order (at most 16
//! wait). A push without a new buffer repeats the last one only for a port
//! that has always been given the same buffer (rewritten in place); a port
//! whose buffers move on plays silence. Once an object port has PCM, its
//! other attributes are frozen until the next `ContextAdvance`. A push copies the beds and
//! the object mix into buffers the context keeps, so steady-state pushes do
//! not allocate and the output is paced without the state lock.
//!
//! Queue model: a context holds `queueDepth` grains that drain one grain
//! period at a time on the process clock. `Push` queues one grain: when the
//! queue is full a non-blocking push fails with NOT_READY. A blocking push
//! returns once the queue has room for another grain after its own (it
//! waits for the grain playing to finish when its own fills the queue), so
//! `GetQueueLevel` right after it reports a free slot unless advanced
//! grains still wait for their push. Middleware that renders exactly the
//! available grains when its output thread signals a push (Wwise: the
//! output thread waits on a semaphore the render thread posts per grain,
//! pushes, and wakes the render thread) relies on that; a push returning
//! with the queue full left both threads waiting on each other. A push
//! carrying PCM to a real device is also paced by AudioOut's device queue.

use super::audio_out::{self, Format, OutputParam};
use super::Export;
use crate::context::CpuContext;
use crate::registry::HleResult;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

const ERR_MEMORY_FAULT: HleResult = 0x8002_000Eu32 as i32 as i64;

pub const AUDIO_OUT2_ERROR_INVALID_PARAM: HleResult = 0x8026_8001u32 as i32 as i64;
pub const AUDIO_OUT2_ERROR_OUT_OF_RESOURCE: HleResult = 0x8026_8002u32 as i32 as i64;
pub const AUDIO_OUT2_ERROR_OUT_OF_MEMORY: HleResult = 0x8026_8003u32 as i32 as i64;
pub const AUDIO_OUT2_ERROR_BUSY: HleResult = 0x8026_8007u32 as i32 as i64;
pub const AUDIO_OUT2_ERROR_NOT_READY: HleResult = 0x8026_8008u32 as i32 as i64;
pub const AUDIO_OUT2_ERROR_INVALID_PORT: HleResult = 0x8026_8009u32 as i32 as i64;
pub const AUDIO_OUT2_ERROR_INVALID_POINTER: HleResult = 0x8026_800Cu32 as i32 as i64;
/// A MAIN port data format with no supported channel layout or sample type.
pub const AUDIO_OUT2_ERROR_UNSUPPORTED_FORMAT: HleResult = 0x8026_800Eu32 as i32 as i64;
pub const AUDIO_OUT2_ERROR_NOT_INITIALIZED: HleResult = 0x8026_8006u32 as i32 as i64;
pub const AUDIO_OUT2_ERROR_INVALID_USER: HleResult = 0x8026_8010u32 as i32 as i64;
pub const AUDIO_OUT2_ERROR_PORT_FULL: HleResult = 0x8026_8012u32 as i32 as i64;
pub const AUDIO_OUT2_ERROR_MASTERING_INVALID_API_PARAM: HleResult = 0x8026_8201u32 as i32 as i64;
pub const AUDIO_OUT2_ERROR_MASTERING_INVALID_STATES_ID: HleResult = 0x8026_8204u32 as i32 as i64;

const CONTEXTS_MAX: usize = 16;
const PORTS_MAX: usize = 256;
const LATENCIES_MAX: usize = 16;
const DEFAULT_QUEUE_DEPTH: u32 = 4;
const DEFAULT_NUM_GRAINS: u32 = 512;
/// PCM buffers a port keeps between pushes.
/// PCM buffers a port keeps between pushes; a title that advances in a
/// burst before pushing keeps every grain (the oldest goes only past this).
const QUEUED_PCM_MAX: usize = 256;

pub const PORT_ATTRIBUTE_ID_PCM: u32 = 0;
pub const PORT_ATTRIBUTE_ID_GAIN: u32 = 1;
pub const PORT_ATTRIBUTE_ID_AMBISONICS: u32 = 8;
pub const AMBISONICS_NONE: u32 = u32::MAX;
const MASTERING_OUTPUT_RECORDING: u32 = 2;

/// SceAudioOut2ContextParam: maxPorts, maxObjectPorts, guaranteeObjectPorts,
/// queueDepth, numGrains, flags, u32 reserved[10].
const CONTEXT_PARAM_SIZE: usize = 0x40;
/// SceAudioOut2PortParam: u16 portType, u16 pad, u32 dataFormat,
/// u32 samplingFreq, u32 flags, userHandle (0x10), u32 reserved[10].
const PORT_PARAM_SIZE: usize = 0x40;
/// SceAudioOut2Attribute: u32 attributeId, pad, const void *value, size_t valueSize.
const ATTRIBUTE_ENTRY_SIZE: u64 = 0x18;
/// Largest attribute value read (16 channel gains).
const ATTRIBUTE_VALUE_MAX: usize = 64;
/// SceAudioOut2PortState: u16 output, u8 numChannels, u8 pad, s16 volume,
/// u16 rerouteCounter, u32 flags, u32 pad, u64 reserved[6].
const PORT_STATE_SIZE: usize = 0x40;
/// SceAudioOut2SystemState: float loudness, u32 pad, u64 reserved[7].
const SYSTEM_STATE_SIZE: usize = 0x40;
/// SceAudioOut2SpeakerInfo: u8 type + pad, u32 availableBits, u32 flags,
/// u32 pad, SceAudioOut2SpeakerAngle angles[16].
const SPEAKER_INFO_SIZE: usize = 0x50;
const SPEAKER_INFO_ANGLE_OFFSET: usize = 0x10;
const SPEAKER_TYPE_TV: u8 = 0;
const SPEAKER_ARRAYS_MAX: usize = 32;
pub(super) const SPEAKER_ARRAY_MAX_SPEAKERS: u32 = 32;

/// Mastering states: header (u32 statesId, u32 reserved[3]), then the
/// compressor states ({u32 id, u32 size}, u32 reserved[2], float
/// inputRms[bands][8], float compressionCoeff[bands][8]) and the limiter
/// states ({id, size}, reserved[2], float inputPeak/outputPeak/gainPeak[8]).
const MASTERING_STATES_ID_DEFAULT: u32 = 1;
const MASTERING_STATES_ID_V2: u32 = 2;
const MASTERING_HEADER_SIZE: usize = 0x10;
const MASTERING_COMPRESSOR_SIZE: usize = 16 + 3 * 8 * 4 * 2;
const MASTERING_COMPRESSOR_V2_SIZE: usize = 16 + 4 * 8 * 4 * 2;
const MASTERING_LIMITER_SIZE: usize = 16 + 3 * 8 * 4;
const MASTERING_STRUCT_ID_COMPRESSOR_DEFAULT: u32 = 0x0101_0001;
const MASTERING_STRUCT_ID_COMPRESSOR_V2: u32 = 0x0102_0001;
const MASTERING_STRUCT_ID_LIMITER_DEFAULT: u32 = 0x0101_0003;

const HALF_POWER: f32 = 0.707_106_78;

// ---------------------------------------------------------------------------
// Data formats and port types
// ---------------------------------------------------------------------------

/// Channels in bits 8..15 of a data format (0 = stereo, at most 16).
pub fn data_format_channels(data_format: u32) -> usize {
    match ((data_format >> 8) & 0xFF) as usize {
        0 => 2,
        c => c.min(16),
    }
}

/// AudioOut format of a data format: type in bits 0..6 (0 float, 1 s16),
/// bit 7 = standard 7.1 order.
pub fn data_format_to_audio_format(data_format: u32) -> Format {
    let channels = data_format_channels(data_format);
    let is_std = data_format & 0x80 != 0;
    match (data_format & 0x7F, channels) {
        (0, 1) => Format::FloatMono,
        (0, 2) => Format::FloatStereo,
        (0, 8) if is_std => Format::Float8ChStd,
        (0, 8) => Format::Float8Ch,
        (0, 12) if !is_std => Format::Float12Ch,
        (1, 1) => Format::S16Mono,
        (1, 2) => Format::S16Stereo,
        (1, 8) if is_std => Format::S16_8ChStd,
        (1, 8) => Format::S16_8Ch,
        _ => Format::Unknown,
    }
}

/// Port type of the main (bed) output.
const PORT_TYPE_MAIN: u16 = 0;

/// Data format check of a MAIN port, as the console reports it: 1, 2, 8 or
/// 12 channels (bits 8-11) with sample type 0 or 1 (bits 0-6). Bit 7 (the
/// standard 8-channel order) without 8 channels is an invalid parameter;
/// bits 12-31 are ignored.
pub fn validate_main_port_format(data_format: u32) -> Result<(), HleResult> {
    let channels = (data_format >> 8) & 0xF;
    if data_format & 0x80 != 0 && channels != 8 {
        return Err(AUDIO_OUT2_ERROR_INVALID_PARAM);
    }
    if !matches!(channels, 1 | 2 | 8 | 12) || data_format & 0x7F > 1 {
        return Err(AUDIO_OUT2_ERROR_UNSUPPORTED_FORMAT);
    }
    Ok(())
}

fn port_type_to_audio_out_type(port_type: u16) -> i32 {
    match port_type & 0xFF {
        1 => audio_out::PORT_TYPE_BGM,
        2 => audio_out::PORT_TYPE_VOICE,
        3 => audio_out::PORT_TYPE_PADSPK,
        4 => audio_out::PORT_TYPE_PERSONAL,
        5 => audio_out::PORT_TYPE_AUX,
        6 => audio_out::PORT_TYPE_VIBRATION,
        _ => audio_out::PORT_TYPE_MAIN,
    }
}

fn is_object_port(port_type: u16) -> bool {
    port_type & 0xFF00 == 0x0100
}

fn grain_micros(grains: u32) -> u64 {
    let samples = if grains == 0 { DEFAULT_NUM_GRAINS } else { grains } as u64;
    (samples * 1_000_000 / 48_000).max(1000)
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// AudioOut ports, time and sleeping.
pub trait Backend: Send + Sync {
    fn open(&self, port_type: i32, samples_num: u32, freq: u32, format: Format) -> i32;
    fn close(&self, handle: i32);
    fn has_device(&self, handle: i32) -> bool;
    fn outputs(&self, params: &[OutputParam], blocking: bool) -> u32;
    /// Process time in microseconds.
    fn now_us(&self) -> u64;
    /// Sleep with no lock held.
    fn sleep_us(&self, us: u64);
}

/// One bed's copy for a push.
#[derive(Default, Clone)]
struct BedCopy {
    handle: i32,
    pcm: Vec<u8>,
    gains: [f32; 16],
}

/// Buffers a push fills; kept by the context between pushes.
#[derive(Default, Clone)]
struct PushScratch {
    beds: Vec<BedCopy>,
    mix: Vec<f32>,
    mix_bytes: Vec<u8>,
}

#[derive(Default, Clone)]
struct Context {
    used: bool,
    handle: u64,
    queue_depth: u32,
    queued: u32,
    /// Grains completed by ContextAdvance on a context with an output device
    /// and not pushed yet; they count toward the queue level.
    advanced: u32,
    num_grains: u32,
    last_update: u64,
    object_audio_handle: i32,
    /// Taken by a push while it outputs.
    scratch: Option<Box<PushScratch>>,
}

#[derive(Clone)]
struct Port {
    used: bool,
    handle: u64,
    context: u64,
    user: u64,
    port_type: u16,
    data_format: u32,
    samples_num: u32,
    audio_format: Format,
    audio_handle: i32,
    gains: [f32; 16],
    ambisonics: u32,
    attributes_locked: bool,
    /// PCM of the grain being pushed (empty: silence).
    pcm_data: Vec<u8>,
    /// PCM set since the last push, oldest first (an empty buffer is a
    /// null PCM pointer).
    pcm_queue: VecDeque<Vec<u8>>,
    /// Buffers for reuse.
    pcm_pool: Vec<Vec<u8>>,
    /// Last non-null PCM address set.
    pcm_last_address: u64,
    /// Every PCM set used the same address: a push without a new buffer
    /// plays it again.
    pcm_repeat: bool,
}

impl Default for Port {
    fn default() -> Self {
        Self {
            used: false,
            handle: 0,
            context: 0,
            user: 0,
            port_type: 0,
            data_format: 0,
            samples_num: DEFAULT_NUM_GRAINS,
            audio_format: Format::Unknown,
            audio_handle: 0,
            gains: [0.0; 16],
            ambisonics: AMBISONICS_NONE,
            attributes_locked: false,
            pcm_data: Vec::new(),
            pcm_queue: VecDeque::new(),
            pcm_pool: Vec::new(),
            pcm_last_address: 0,
            pcm_repeat: false,
        }
    }
}

impl Port {
    fn pcm_size(&self) -> usize {
        let bps = if self.data_format & 0x7F == 1 { 2 } else { 4 };
        self.samples_num as usize * data_format_channels(self.data_format) * bps
    }

    /// Move to the PCM of the next grain: the oldest queued buffer, else
    /// the last one again if the port repeats it, else silence.
    fn advance_pcm(&mut self) {
        let next = match self.pcm_queue.pop_front() {
            Some(next) => next,
            None if self.pcm_repeat => return,
            None => Vec::new(),
        };
        let old = std::mem::replace(&mut self.pcm_data, next);
        self.recycle(old);
    }

    fn recycle(&mut self, mut buffer: Vec<u8>) {
        if buffer.capacity() != 0 && self.pcm_pool.len() < QUEUED_PCM_MAX + 2 {
            buffer.clear();
            self.pcm_pool.push(buffer);
        }
    }

    /// Queue PCM set at `address` (`pcm` empty: none).
    fn queue_pcm(&mut self, address: u64, pcm: Vec<u8>) {
        self.pcm_queue.push_back(pcm);
        if self.pcm_queue.len() > QUEUED_PCM_MAX {
            if let Some(dropped) = self.pcm_queue.pop_front() {
                self.recycle(dropped);
            }
        }
        self.pcm_repeat = address != 0 && (self.pcm_last_address == 0 || self.pcm_last_address == address);
        if address != 0 {
            self.pcm_last_address = address;
        }
    }
}

#[derive(Default, Clone, Copy)]
struct Latency {
    used: bool,
    user_id: u32,
    output: u32,
    latency_us: u32,
}

struct State {
    contexts: Vec<Context>,
    ports: Vec<Port>,
    users: Vec<u64>,
    next_user: u64,
}

impl State {
    fn context(&mut self, ctx: u64) -> Option<&mut Context> {
        self.contexts.iter_mut().find(|c| c.used && c.handle == ctx)
    }

    fn port(&mut self, port: u64) -> Option<&mut Port> {
        self.ports.iter_mut().find(|p| p.used && p.handle == port)
    }
}

/// Fields of SceAudioOut2ContextParam that matter.
#[derive(Clone, Copy, Default)]
pub struct ContextParam {
    pub max_object_ports: u32,
    pub queue_depth: u32,
    pub num_grains: u32,
}

/// Fields of SceAudioOut2PortParam that matter.
#[derive(Clone, Copy, Default)]
pub struct PortParam {
    pub port_type: u16,
    pub data_format: u32,
    pub sampling_freq: u32,
    pub user_handle: u64,
}

/// One SceAudioOut2Attribute; `bytes` holds the start of the value.
#[derive(Clone, Copy)]
pub struct Attribute {
    pub id: u32,
    pub value: u64,
    pub value_size: u64,
    bytes: [u8; ATTRIBUTE_VALUE_MAX],
    len: usize,
}

impl Default for Attribute {
    fn default() -> Self {
        Self { id: 0, value: 0, value_size: 0, bytes: [0; ATTRIBUTE_VALUE_MAX], len: 0 }
    }
}

impl Attribute {
    /// `bytes`: the start of the value (at most `ATTRIBUTE_VALUE_MAX` bytes are kept).
    pub fn new(id: u32, value: u64, value_size: u64, bytes: &[u8]) -> Self {
        let mut a = Self { id, value, value_size, ..Self::default() };
        a.len = bytes.len().min(ATTRIBUTE_VALUE_MAX);
        a.bytes[..a.len].copy_from_slice(&bytes[..a.len]);
        a
    }

    fn bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

pub struct AudioOut2<B: Backend> {
    backend: B,
    state: Mutex<State>,
    next_context: AtomicU64,
    next_port: AtomicU64,
    latencies: Mutex<[Latency; LATENCIES_MAX]>,
}

fn locked<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Mix one object's grain into the context's stereo bus.
fn mix_object(mix: &mut Vec<f32>, num_grains: u32, object: &Port) {
    if mix.is_empty() {
        mix.resize(num_grains as usize * 2, 0.0);
    }
    // Plain objects are centred. First-order ambisonics plays W and the
    // left/right component as two cardioids; FuMa W has its own scale.
    let (mut left, mut right) = (HALF_POWER, HALF_POWER);
    match object.ambisonics {
        AMBISONICS_NONE | 64 => {}
        0 => (left, right) = (1.0, 1.0),
        2 | 65 => right = -HALF_POWER,
        _ => return,
    }
    left *= object.gains[0];
    right *= object.gains[0];
    let frames = mix.chunks_exact_mut(2);
    if object.audio_format == Format::FloatMono {
        for (m, s) in frames.zip(object.pcm_data.chunks_exact(4)) {
            let sample = f32::from_le_bytes([s[0], s[1], s[2], s[3]]);
            m[0] += sample * left;
            m[1] += sample * right;
        }
    } else {
        for (m, s) in frames.zip(object.pcm_data.chunks_exact(2)) {
            let sample = i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0;
            m[0] += sample * left;
            m[1] += sample * right;
        }
    }
}

impl<B: Backend> AudioOut2<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            state: Mutex::new(State {
                contexts: vec![Context::default(); CONTEXTS_MAX],
                ports: vec![Port::default(); PORTS_MAX],
                users: Vec::new(),
                next_user: 1,
            }),
            next_context: AtomicU64::new(1),
            next_port: AtomicU64::new(1),
            latencies: Mutex::new([Latency::default(); LATENCIES_MAX]),
        }
    }

    /// Drain the grains that finished playing since the last update.
    fn update_context(&self, c: &mut Context) {
        let now = self.backend.now_us();
        if c.last_update == 0 || c.queued == 0 {
            c.last_update = now;
            return;
        }
        let grain = grain_micros(c.num_grains);
        if now <= c.last_update {
            return;
        }
        let drained = (c.queued as u64).min((now - c.last_update) / grain);
        if drained == 0 {
            return;
        }
        c.queued -= drained as u32;
        c.last_update += drained * grain;
        if c.queued == 0 {
            c.last_update = now;
        }
    }

    /// Submit every port of the context: beds as they are, objects as one mix.
    fn queue_context_audio(&self, ctx: u64, blocking: bool) {
        let (mut scratch, beds, object_handle) = {
            let mut s = locked(&self.state);
            let State { contexts, ports, .. } = &mut *s;
            let Some(c) = contexts.iter_mut().find(|c| c.used && c.handle == ctx) else { return };
            let mut scratch = c.scratch.take().unwrap_or_default();
            let PushScratch { beds, mix, mix_bytes } = &mut *scratch;
            mix.clear();
            let mut n = 0;
            for p in ports.iter_mut().filter(|p| p.used && p.context == ctx) {
                p.advance_pcm();
            }
            for p in ports.iter().filter(|p| p.used && p.context == ctx && !p.pcm_data.is_empty()) {
                if is_object_port(p.port_type) {
                    mix_object(mix, c.num_grains, p);
                } else if p.audio_handle > 0 && n < audio_out::OUT_PORTS_MAX {
                    if beds.len() == n {
                        beds.push(BedCopy::default());
                    }
                    let bed = &mut beds[n];
                    bed.handle = p.audio_handle;
                    bed.pcm.clear();
                    bed.pcm.extend_from_slice(&p.pcm_data);
                    bed.gains = p.gains;
                    n += 1;
                }
            }
            let object_handle = if !mix.is_empty() && n < audio_out::OUT_PORTS_MAX {
                mix_bytes.clear();
                mix_bytes.extend(mix.iter().flat_map(|v| v.to_le_bytes()));
                c.object_audio_handle
            } else {
                0
            };
            (scratch, n, object_handle)
        };
        // Submitted without the state lock: the output paces the caller.
        let mut params = [OutputParam::NONE; audio_out::OUT_PORTS_MAX];
        for (param, bed) in params.iter_mut().zip(&scratch.beds[..beds]) {
            *param = OutputParam { handle: bed.handle, data: Some(&bed.pcm), gains: Some(&bed.gains) };
        }
        let mut n = beds;
        if object_handle != 0 {
            params[n] = OutputParam { handle: object_handle, data: Some(&scratch.mix_bytes), gains: None };
            n += 1;
        }
        if n != 0 {
            self.backend.outputs(&params[..n], blocking);
        }
        // Hand the buffers back for the next push.
        scratch.mix.clear();
        let mut s = locked(&self.state);
        if let Some(c) = s.context(ctx) {
            if c.scratch.is_none() {
                c.scratch = Some(scratch);
            }
        }
    }

    pub fn context_create(&self, param: &ContextParam) -> Result<u64, HleResult> {
        let num_grains = if param.num_grains == 0 { DEFAULT_NUM_GRAINS } else { param.num_grains };
        let object_audio_handle = if param.max_object_ports == 0 {
            0
        } else {
            self.backend.open(audio_out::PORT_TYPE_MAIN, num_grains, 48000, Format::FloatStereo)
        };
        if param.max_object_ports != 0 && object_audio_handle == 0 {
            return Err(AUDIO_OUT2_ERROR_OUT_OF_RESOURCE);
        }
        let mut s = locked(&self.state);
        let Some(c) = s.contexts.iter_mut().find(|c| !c.used) else {
            drop(s);
            self.close_audio_handle(object_audio_handle);
            return Err(AUDIO_OUT2_ERROR_OUT_OF_RESOURCE);
        };
        let handle = self.next_context.fetch_add(1, Ordering::Relaxed);
        *c = Context {
            used: true,
            handle,
            queue_depth: if param.queue_depth == 0 { DEFAULT_QUEUE_DEPTH } else { param.queue_depth },
            queued: 0,
            advanced: 0,
            num_grains,
            last_update: self.backend.now_us(),
            object_audio_handle,
            scratch: None,
        };
        Ok(handle)
    }

    fn close_audio_handle(&self, handle: i32) {
        if handle > 0 {
            self.backend.close(handle);
        }
    }

    pub fn context_destroy(&self, ctx: u64) {
        let mut handles = Vec::new();
        {
            let mut s = locked(&self.state);
            if let Some(c) = s.context(ctx) {
                handles.push(c.object_audio_handle);
                *c = Context::default();
            }
            for p in s.ports.iter_mut().filter(|p| p.used && p.context == ctx) {
                handles.push(p.audio_handle);
                *p = Port::default();
            }
        }
        for h in handles {
            self.close_audio_handle(h);
        }
    }

    pub fn context_advance(&self, ctx: u64) {
        let mut s = locked(&self.state);
        let State { contexts, ports, .. } = &mut *s;
        let Some(c) = contexts.iter_mut().find(|c| c.used && c.handle == ctx) else { return };
        self.update_context(c);
        let has_device = self.backend.has_device(c.object_audio_handle)
            || ports.iter().any(|p| p.used && p.context == ctx && self.backend.has_device(p.audio_handle));
        if has_device {
            c.advanced = c.advanced.saturating_add(1);
        }
        for p in ports.iter_mut().filter(|p| p.used && p.context == ctx) {
            p.attributes_locked = false;
        }
    }

    pub fn context_push(&self, ctx: u64, blocking: bool) -> HleResult {
        loop {
            let sleep_micros;
            {
                let mut s = locked(&self.state);
                let Some(c) = s.contexts.iter_mut().find(|c| c.used && c.handle == ctx) else {
                    return AUDIO_OUT2_ERROR_INVALID_PARAM;
                };
                self.update_context(c);
                // Until the grain playing ends (the next slot frees).
                let grain = grain_micros(c.num_grains);
                sleep_micros = (c.last_update + grain).saturating_sub(self.backend.now_us()).clamp(100, grain);
                // A blocking push leaves a free slot behind (see the module
                // documentation); a non-blocking one only needs one.
                let accepted = if blocking { c.queued.saturating_add(1) < c.queue_depth.max(2) } else { c.queued < c.queue_depth };
                if accepted {
                    if c.queued == 0 {
                        c.last_update = self.backend.now_us();
                    }
                    if c.queued < c.queue_depth {
                        c.queued += 1;
                    }
                    c.advanced = c.advanced.saturating_sub(1);
                }
                if accepted {
                    drop(s);
                    self.queue_context_audio(ctx, blocking);
                    return 0;
                }
            }
            if !blocking {
                return AUDIO_OUT2_ERROR_NOT_READY;
            }
            self.backend.sleep_us(sleep_micros);
        }
    }

    pub fn context_get_queue_level(&self, ctx: u64, level: Option<&mut u32>, available: Option<&mut u32>) {
        let (mut l, mut a) = (0, 4);
        let mut s = locked(&self.state);
        if let Some(c) = s.context(ctx) {
            self.update_context(c);
            // Pushed grains still playing and advanced grains waiting for
            // their push both fill the queue.
            l = c.queue_depth.min(c.queued.saturating_add(c.advanced));
            a = c.queue_depth - l;
        }
        if let Some(level) = level {
            *level = l;
        }
        if let Some(available) = available {
            *available = a;
        }
    }

    pub fn port_create(&self, ctx: u64, param: &PortParam) -> Result<u64, HleResult> {
        if param.port_type == PORT_TYPE_MAIN {
            validate_main_port_format(param.data_format)?;
        }
        let handle = self.next_port.fetch_add(1, Ordering::Relaxed);
        let audio_format = data_format_to_audio_format(param.data_format);
        let audio_type = port_type_to_audio_out_type(param.port_type);
        let object = is_object_port(param.port_type);

        let (slot, samples_num) = {
            let mut s = locked(&self.state);
            let Some(&mut Context { num_grains, object_audio_handle, .. }) = s.context(ctx) else {
                return Err(AUDIO_OUT2_ERROR_INVALID_PARAM);
            };
            if object
                && (param.port_type != 0x100
                    || object_audio_handle == 0
                    || param.sampling_freq != 48000
                    || !matches!(audio_format, Format::FloatMono | Format::S16Mono))
            {
                return Err(AUDIO_OUT2_ERROR_INVALID_PARAM);
            }
            if !s.users.contains(&param.user_handle) {
                return Err(AUDIO_OUT2_ERROR_INVALID_PARAM);
            }
            let Some(slot) = s.ports.iter().position(|p| !p.used) else {
                return Err(AUDIO_OUT2_ERROR_PORT_FULL);
            };
            s.ports[slot] = Port {
                used: true,
                handle,
                context: ctx,
                user: param.user_handle,
                port_type: param.port_type,
                data_format: param.data_format,
                samples_num: num_grains,
                audio_format,
                gains: [1.0; 16],
                ..Port::default()
            };
            (slot, num_grains)
        };

        // Opening the output can take a while; the slot stays reserved.
        let audio_handle = if audio_format != Format::Unknown && !object {
            self.backend.open(audio_type, samples_num, param.sampling_freq, audio_format)
        } else {
            0
        };
        let reserved = {
            let mut s = locked(&self.state);
            let p = &mut s.ports[slot];
            let reserved = p.used && p.handle == handle;
            if reserved {
                p.audio_handle = audio_handle;
            }
            reserved
        };
        if !reserved {
            // The context went away meanwhile.
            self.close_audio_handle(audio_handle);
            return Err(AUDIO_OUT2_ERROR_INVALID_PARAM);
        }
        Ok(handle)
    }

    pub fn port_destroy(&self, port: u64) {
        let audio_handle = {
            let mut s = locked(&self.state);
            match s.port(port) {
                Some(p) => std::mem::take(p).audio_handle,
                None => 0,
            }
        };
        self.close_audio_handle(audio_handle);
    }

    /// Apply attributes atomically. `read_pcm(address, buffer)` fills the
    /// buffer with PCM from guest memory; false when it cannot.
    pub fn port_set_attributes(
        &self,
        port: u64,
        attributes: &[Attribute],
        read_pcm: &mut dyn FnMut(u64, &mut [u8]) -> bool,
    ) -> HleResult {
        let mut s = locked(&self.state);
        let Some(p) = s.port(port) else {
            return AUDIO_OUT2_ERROR_INVALID_PORT;
        };
        let mut pcm_address = 0u64;
        let mut gains = p.gains;
        let mut ambisonics = p.ambisonics;
        let mut seen = 0u32;
        for a in attributes {
            let id = a.id;
            if id != PORT_ATTRIBUTE_ID_PCM && id != PORT_ATTRIBUTE_ID_GAIN && id != PORT_ATTRIBUTE_ID_AMBISONICS {
                continue;
            }
            if seen & (1 << id) != 0 {
                return AUDIO_OUT2_ERROR_INVALID_PARAM;
            }
            seen |= 1 << id;
            if id != PORT_ATTRIBUTE_ID_PCM && p.attributes_locked {
                continue;
            }
            if a.value == 0 {
                return AUDIO_OUT2_ERROR_INVALID_PARAM;
            }
            match id {
                PORT_ATTRIBUTE_ID_PCM => {
                    if a.value_size < 8 || a.bytes().len() < 8 {
                        return AUDIO_OUT2_ERROR_INVALID_PARAM;
                    }
                    pcm_address = u64::from_le_bytes(a.bytes()[0..8].try_into().unwrap());
                }
                PORT_ATTRIBUTE_ID_GAIN => {
                    let channels = data_format_channels(p.data_format);
                    // Middleware passes a full speaker-layout gain array even
                    // for a mono port (Wwise: 8 gains on a 1-channel port);
                    // the port's channels take the leading entries.
                    if a.value_size < channels as u64 * 4 || a.bytes().len() < channels * 4 {
                        return AUDIO_OUT2_ERROR_INVALID_PARAM;
                    }
                    for (ch, g) in gains.iter_mut().enumerate().take(channels) {
                        *g = f32::from_le_bytes(a.bytes()[ch * 4..ch * 4 + 4].try_into().unwrap());
                        if !g.is_finite() || *g < 0.0 {
                            return AUDIO_OUT2_ERROR_INVALID_PARAM;
                        }
                    }
                }
                _ => {
                    if !is_object_port(p.port_type) || a.value_size != 4 || a.bytes().len() < 4 {
                        return AUDIO_OUT2_ERROR_INVALID_PARAM;
                    }
                    ambisonics = u32::from_le_bytes(a.bytes()[0..4].try_into().unwrap());
                    if ambisonics != AMBISONICS_NONE && ambisonics > 15 && !(64..=99).contains(&ambisonics) {
                        return AUDIO_OUT2_ERROR_INVALID_PARAM;
                    }
                }
            }
        }

        let has_pcm = seen & (1 << PORT_ATTRIBUTE_ID_PCM) != 0;
        let load_pcm = has_pcm && pcm_address != 0 && p.audio_format != Format::Unknown;
        let mut pcm = Vec::new();
        if load_pcm {
            pcm = p.pcm_pool.pop().unwrap_or_default();
            pcm.resize(p.pcm_size(), 0);
            if !read_pcm(pcm_address, &mut pcm) {
                p.recycle(pcm);
                return ERR_MEMORY_FAULT;
            }
        }
        p.gains = gains;
        p.ambisonics = ambisonics;
        if has_pcm {
            p.queue_pcm(pcm_address, pcm);
            p.attributes_locked = is_object_port(p.port_type);
        }
        0
    }

    /// Guest channel count of a port.
    pub fn port_channels(&self, port: u64) -> Option<usize> {
        locked(&self.state).port(port).map(|p| data_format_channels(p.data_format))
    }

    pub fn user_create(&self, handle: Option<&mut u64>) -> HleResult {
        let Some(handle) = handle else {
            return AUDIO_OUT2_ERROR_INVALID_PARAM;
        };
        let mut s = locked(&self.state);
        *handle = s.next_user;
        s.next_user += 1;
        s.users.push(*handle);
        0
    }

    pub fn user_destroy(&self, handle: u64) -> HleResult {
        let mut s = locked(&self.state);
        let Some(i) = s.users.iter().position(|&u| u == handle) else {
            return AUDIO_OUT2_ERROR_INVALID_PARAM;
        };
        if s.ports.iter().any(|p| p.used && p.user == handle) {
            return AUDIO_OUT2_ERROR_BUSY;
        }
        s.users.remove(i);
        0
    }

    pub fn user_get_supported_attributes(
        &self,
        handle: u64,
        context_attributes: Option<&mut u32>,
        port_attributes: Option<&mut u32>,
    ) -> HleResult {
        let (Some(context_attributes), Some(port_attributes)) = (context_attributes, port_attributes) else {
            return AUDIO_OUT2_ERROR_INVALID_PARAM;
        };
        if !locked(&self.state).users.contains(&handle) {
            return AUDIO_OUT2_ERROR_INVALID_PARAM;
        }
        *context_attributes = 0;
        *port_attributes =
            (1 << PORT_ATTRIBUTE_ID_PCM) | (1 << PORT_ATTRIBUTE_ID_GAIN) | (1 << PORT_ATTRIBUTE_ID_AMBISONICS);
        0
    }

    pub fn set_3d_latency(&self, user_id: u32, output: u32, latency_us: u32) -> HleResult {
        if output > MASTERING_OUTPUT_RECORDING {
            return AUDIO_OUT2_ERROR_INVALID_PARAM;
        }
        let mut table = locked(&self.latencies);
        if let Some(l) = table.iter_mut().find(|l| l.used && l.user_id == user_id && l.output == output) {
            l.latency_us = latency_us;
            return 0;
        }
        match table.iter_mut().find(|l| !l.used) {
            Some(l) => {
                *l = Latency { used: true, user_id, output, latency_us };
                0
            }
            None => AUDIO_OUT2_ERROR_PORT_FULL,
        }
    }
}

// ---------------------------------------------------------------------------
// The emulator's instance
// ---------------------------------------------------------------------------

struct HostBackend;

impl Backend for HostBackend {
    fn open(&self, port_type: i32, samples_num: u32, freq: u32, format: Format) -> i32 {
        audio_out::internal_open(port_type, samples_num, freq, format)
    }

    fn close(&self, handle: i32) {
        audio_out::internal_close(handle);
    }

    fn has_device(&self, handle: i32) -> bool {
        audio_out::internal_has_device(handle)
    }

    fn outputs(&self, params: &[OutputParam], blocking: bool) -> u32 {
        audio_out::internal_outputs(params, blocking)
    }

    fn now_us(&self) -> u64 {
        crate::time::process_time_usec()
    }

    fn sleep_us(&self, us: u64) {
        crate::signals::sleep(Duration::from_micros(us));
    }
}

static OUT2: LazyLock<AudioOut2<HostBackend>> = LazyLock::new(|| AudioOut2::new(HostBackend));
static PUSH_COUNT: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

fn is_writable_out_buffer(v: u64) -> bool {
    (0x1000_0000..0x8000_0000_0000).contains(&v)
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn cb_ok(_ctx: &mut CpuContext) -> HleResult {
    0
}

fn cb_context_reset_param(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return AUDIO_OUT2_ERROR_INVALID_POINTER;
    }
    let mut p = [0u8; CONTEXT_PARAM_SIZE];
    for (i, v) in [256u32, 256, 0, DEFAULT_QUEUE_DEPTH, DEFAULT_NUM_GRAINS, 1].iter().enumerate() {
        p[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    if ctx.write_bytes(addr, &p) {
        0
    } else {
        AUDIO_OUT2_ERROR_INVALID_POINTER
    }
}

fn cb_context_query_memory(ctx: &mut CpuContext) -> HleResult {
    let param = ctx.rdi();
    let out = ctx.rsi();
    if param == 0 || out == 0 {
        return AUDIO_OUT2_ERROR_INVALID_POINTER;
    }
    let Some(qd) = ctx.read_u32(param + 0x0C) else {
        return AUDIO_OUT2_ERROR_INVALID_POINTER;
    };
    let qd = if qd == 0 { DEFAULT_QUEUE_DEPTH } else { qd } as u64;
    if ctx.write_u64(out, 0x10000 + qd * 0x590) {
        0
    } else {
        AUDIO_OUT2_ERROR_INVALID_POINTER
    }
}

/// sceAudioOut2ContextCreate(param*, buffer, size, SceAudioOut2ContextHandle *out)
fn cb_context_create(ctx: &mut CpuContext) -> HleResult {
    let (param, out) = (ctx.rdi(), ctx.rcx());
    if param == 0 || out == 0 {
        return AUDIO_OUT2_ERROR_INVALID_POINTER;
    }
    let Some(p) = ctx.read_bytes(param, CONTEXT_PARAM_SIZE) else {
        return AUDIO_OUT2_ERROR_INVALID_POINTER;
    };
    let param = ContextParam { max_object_ports: u32_at(&p, 4), queue_depth: u32_at(&p, 0x0C), num_grains: u32_at(&p, 0x10) };
    let handle = match OUT2.context_create(&param) {
        Ok(h) => h,
        Err(e) => return e,
    };
    log::info!(
        "[AUDIO] AudioOut2 context 0x{handle:X}: grains={} queue={} objects={}",
        param.num_grains,
        param.queue_depth,
        param.max_object_ports
    );
    if ctx.write_u64(out, handle) {
        0
    } else {
        AUDIO_OUT2_ERROR_INVALID_POINTER
    }
}

fn cb_context_destroy(ctx: &mut CpuContext) -> HleResult {
    OUT2.context_destroy(ctx.rdi());
    0
}

fn cb_context_bed_write(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// sceAudioOut2ContextPush(ctx, blocking)
fn cb_context_push(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    let blocking = ctx.rsi() as u32 != 0;
    if PUSH_COUNT.fetch_add(1, Ordering::Relaxed) == 0 {
        log::info!("[AUDIO] sceAudioOut2ContextPush first call: context=0x{handle:X} blocking={blocking}");
    }
    OUT2.context_push(handle, blocking)
}

fn cb_context_advance(ctx: &mut CpuContext) -> HleResult {
    OUT2.context_advance(ctx.rdi());
    0
}

/// sceAudioOut2ContextGetQueueLevel(ctx, u32 *level, u32 *available)
fn cb_context_get_queue_level(ctx: &mut CpuContext) -> HleResult {
    let out_level = ctx.rsi();
    let out_avail = ctx.rdx();
    let (mut level, mut available) = (0, 0);
    OUT2.context_get_queue_level(ctx.rdi(), Some(&mut level), Some(&mut available));
    if out_level != 0 && !ctx.write_u32(out_level, level) {
        return ERR_MEMORY_FAULT;
    }
    if out_avail != 0 && out_avail != out_level && is_writable_out_buffer(out_avail) && !ctx.write_u32(out_avail, available) {
        return ERR_MEMORY_FAULT;
    }
    0
}

/// Attributes read without allocating; longer lists go to the heap.
const ATTRIBUTES_INLINE: usize = 16;

/// sceAudioOut2PortSetAttributes(port, const SceAudioOut2Attribute *attrs, u32 num)
fn cb_port_set_attributes(ctx: &mut CpuContext) -> HleResult {
    let port = ctx.rdi();
    let attrs = ctx.rsi();
    let count = ctx.rdx() as u32 as usize;
    if count != 0 && attrs == 0 {
        return AUDIO_OUT2_ERROR_INVALID_PARAM;
    }
    let mut inline = [Attribute::default(); ATTRIBUTES_INLINE];
    let mut heap = Vec::new();
    for i in 0..count {
        let mut e = [0u8; ATTRIBUTE_ENTRY_SIZE as usize];
        if ctx.vm.read(attrs + i as u64 * ATTRIBUTE_ENTRY_SIZE, &mut e).is_err() {
            return ERR_MEMORY_FAULT;
        }
        let value = u64::from_le_bytes(e[8..16].try_into().unwrap());
        let value_size = u64::from_le_bytes(e[16..24].try_into().unwrap());
        let mut bytes = [0u8; ATTRIBUTE_VALUE_MAX];
        let n = if value != 0 { value_size.min(ATTRIBUTE_VALUE_MAX as u64) as usize } else { 0 };
        if n != 0 && ctx.vm.read(value, &mut bytes[..n]).is_err() {
            return ERR_MEMORY_FAULT;
        }
        let a = Attribute::new(u32_at(&e, 0), value, value_size, &bytes[..n]);
        if count <= ATTRIBUTES_INLINE {
            inline[i] = a;
        } else {
            heap.push(a);
        }
    }
    let list = if count <= ATTRIBUTES_INLINE { &inline[..count] } else { &heap[..] };
    let mut peak = 0.0f32;
    let r = OUT2.port_set_attributes(port, list, &mut |addr, buffer| {
        let ok = ctx.vm.read(addr, buffer).is_ok();
        if ok && super::host::stats_enabled() {
            peak = buffer.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).abs()).fold(0.0, f32::max);
        }
        ok
    });
    if super::host::stats_enabled() {
        trace_set_attributes(port, list, r, peak);
    }
    r
}

/// `RPS5E_AUDIO_STATS`: failed attribute sets (first few) and, every
/// 2000 calls, how many carried PCM and the loudest float sample seen.
fn trace_set_attributes(port: u64, list: &[Attribute], r: HleResult, peak: f32) {
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static FAILS: AtomicU64 = AtomicU64::new(0);
    static PCM: AtomicU64 = AtomicU64::new(0);
    static PEAK: Mutex<f32> = Mutex::new(0.0);
    let n = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if list.iter().any(|a| a.id == PORT_ATTRIBUTE_ID_PCM) {
        PCM.fetch_add(1, Ordering::Relaxed);
    }
    {
        let mut p = locked(&PEAK);
        *p = p.max(peak);
    }
    if r != 0 && FAILS.fetch_add(1, Ordering::Relaxed) < 16 {
        let attrs: Vec<String> = list.iter().map(|a| format!("id={} size={}", a.id, a.value_size)).collect();
        log::warn!("[AUDIO] sceAudioOut2PortSetAttributes port={port:#x} [{}] -> {r:#x}", attrs.join(", "));
    }
    if n % 2000 == 0 {
        let peak = std::mem::take(&mut *locked(&PEAK));
        log::info!(
            "[AUDIO] AudioOut2 attributes: {n} calls, {} with PCM, {} failed, float peak {peak:.4}",
            PCM.load(Ordering::Relaxed),
            FAILS.load(Ordering::Relaxed)
        );
    }
}

/// sceAudioOut2PortCreate(ctx, const SceAudioOut2PortParam *param, SceAudioOut2PortHandle *out)
fn cb_port_create(ctx: &mut CpuContext) -> HleResult {
    let (context, param, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if param == 0 || out == 0 {
        return AUDIO_OUT2_ERROR_INVALID_POINTER;
    }
    let Some(p) = ctx.read_bytes(param, PORT_PARAM_SIZE) else {
        return AUDIO_OUT2_ERROR_INVALID_POINTER;
    };
    let param = PortParam {
        port_type: u16::from_le_bytes([p[0], p[1]]),
        data_format: u32_at(&p, 4),
        sampling_freq: u32_at(&p, 8),
        user_handle: u64::from_le_bytes(p[0x10..0x18].try_into().unwrap()),
    };
    let handle = match OUT2.port_create(context, &param) {
        Ok(h) => h,
        Err(e) => {
            // A rejected MAIN format also stores an invalid handle.
            if param.port_type == PORT_TYPE_MAIN && validate_main_port_format(param.data_format).is_err() {
                ctx.write_u64(out, u64::MAX);
            }
            static FAILS: AtomicU64 = AtomicU64::new(0);
            if FAILS.fetch_add(1, Ordering::Relaxed) < 16 {
                log::warn!(
                    "[AUDIO] sceAudioOut2PortCreate context=0x{context:X} type=0x{:X} format=0x{:X} freq={} user={:#x} -> {e:#x}",
                    param.port_type,
                    param.data_format,
                    param.sampling_freq,
                    param.user_handle
                );
            }
            return e;
        }
    };
    if handle <= 16 || handle % 600 == 0 {
        log::info!(
            "[AUDIO] AudioOut2 port 0x{handle:X}: context=0x{context:X} type=0x{:X} format=0x{:X} freq={}",
            param.port_type,
            param.data_format,
            param.sampling_freq
        );
    }
    if ctx.write_u64(out, handle) {
        0
    } else {
        AUDIO_OUT2_ERROR_INVALID_POINTER
    }
}

fn cb_port_destroy(ctx: &mut CpuContext) -> HleResult {
    OUT2.port_destroy(ctx.rdi());
    0
}

fn cb_port_get_state(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rsi();
    if addr == 0 {
        return AUDIO_OUT2_ERROR_INVALID_POINTER;
    }
    let channels = OUT2.port_channels(ctx.rdi()).unwrap_or(2);
    let mut st = [0u8; PORT_STATE_SIZE];
    st[0..2].copy_from_slice(&1u16.to_le_bytes());
    st[2] = channels as u8;
    st[4..6].copy_from_slice(&127i16.to_le_bytes());
    if !ctx.write_bytes(addr, &st) {
        return AUDIO_OUT2_ERROR_INVALID_POINTER;
    }
    0
}

fn cb_context_set_attributes(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdx() as u32 != 0 && ctx.rsi() == 0 {
        AUDIO_OUT2_ERROR_INVALID_POINTER
    } else {
        0
    }
}

fn cb_get_system_state(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 || !ctx.write_bytes(addr, &[0u8; SYSTEM_STATE_SIZE]) {
        return AUDIO_OUT2_ERROR_INVALID_POINTER;
    }
    0
}

/// sceAudioOut2Initialize(): marks the audio output system ready.
fn cb_audio_out2_init(_ctx: &mut CpuContext) -> HleResult {
    crate::audio::AUDIO_OUT_INITIALIZED.store(true, std::sync::atomic::Ordering::Release);
    0
}

/// sceAudioOut2GetSpeakerInfo(SceAudioOut2SpeakerInfo *out, u32 flags)
fn cb_get_speaker_info(ctx: &mut CpuContext) -> HleResult {
    // The initialisation is checked before anything else, then the
    // pointer, then the flags (measured on hardware).
    if !crate::audio::AUDIO_OUT_INITIALIZED.load(std::sync::atomic::Ordering::Acquire) {
        return AUDIO_OUT2_ERROR_NOT_INITIALIZED;
    }
    let addr = ctx.rdi();
    if addr == 0 {
        return AUDIO_OUT2_ERROR_INVALID_POINTER;
    }
    if ctx.rsi() as u32 > 1 {
        return AUDIO_OUT2_ERROR_INVALID_PARAM;
    }
    let mut info = [0u8; SPEAKER_INFO_SIZE];
    info[0] = SPEAKER_TYPE_TV;
    info[4..8].copy_from_slice(&0b11u32.to_le_bytes()); // FRONT_LEFT | FRONT_RIGHT
    // Sixteen {i16 azimuth, i16 elevation}: front left at +30 degrees,
    // front right at -30, no speaker {-360, -360} (PS5 FW 13.60).
    let a = SPEAKER_INFO_ANGLE_OFFSET;
    for i in 0..16 {
        let angle: (i16, i16) = match i {
            0 => (30, 0),
            1 => (-30, 0),
            _ => (-360, -360),
        };
        info[a + i * 4..a + i * 4 + 2].copy_from_slice(&angle.0.to_le_bytes());
        info[a + i * 4 + 2..a + i * 4 + 4].copy_from_slice(&angle.1.to_le_bytes());
    }
    if !ctx.write_bytes(addr, &info) {
        return AUDIO_OUT2_ERROR_INVALID_POINTER;
    }
    0
}

/// sceAudioOut2MasteringSetParam(const params*, output, flags): the host
/// mix has no mastering to configure; a null list is refused.
fn cb_mastering_set_param(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 { AUDIO_OUT2_ERROR_MASTERING_INVALID_API_PARAM } else { 0 }
}

/// sceAudioOut2SetSystemDebugState(const param*): nothing to apply; a null
/// parameter is refused.
fn cb_set_system_debug_state(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 { 0x8026_0502u32 as i32 as i64 } else { 0 }
}

/// size_t sceAudioOut2GetSpeakerArrayMemorySize(u32 numSpeakers, u8 is3d, u8 isAmbisonics)
pub fn speaker_array_memory_size(num_speakers: u32, is_3d: bool, is_ambisonics: bool) -> u64 {
    let speakers = num_speakers.clamp(1, SPEAKER_ARRAY_MAX_SPEAKERS) as u64;
    0x400 + speakers * if is_ambisonics { 0x100 } else { 0x40 } + if is_3d { 0x200 } else { 0 }
}

fn cb_get_speaker_array_memory_size(ctx: &mut CpuContext) -> HleResult {
    speaker_array_memory_size(ctx.rdi() as u32, ctx.rsi() as u8 != 0, ctx.rdx() as u8 != 0) as HleResult
}

/// Stereo output: a source plays on both front speakers; ambisonics W uses
/// its normalisation, every other component passes at unity.
pub fn speaker_coefficient(ambisonics_channel: Option<u32>, index: usize) -> f32 {
    match (ambisonics_channel, index) {
        (None, 0 | 1) => 1.0,
        (Some(0 | 64), 0) => 0.707_106_77,
        (Some(_), 0) => 1.0,
        _ => 0.0,
    }
}

fn write_coefficients(ctx: &CpuContext, dst: u64, n: u32, ambisonics_channel: Option<u32>) -> HleResult {
    if n == 0 {
        return 0;
    }
    if dst == 0 {
        return AUDIO_OUT2_ERROR_INVALID_POINTER;
    }
    const CHUNK: usize = 64;
    let mut buf = [0u8; CHUNK * 4];
    let mut done = 0usize;
    while done < n as usize {
        let k = (n as usize - done).min(CHUNK);
        for (i, b) in buf.chunks_exact_mut(4).take(k).enumerate() {
            b.copy_from_slice(&speaker_coefficient(ambisonics_channel, done + i).to_le_bytes());
        }
        if !ctx.write_bytes(dst + done as u64 * 4, &buf[..k * 4]) {
            return AUDIO_OUT2_ERROR_INVALID_POINTER;
        }
        done += k;
    }
    0
}

/// Write `n` mix coefficients of a speaker array (`write_coefficients`
/// for the Audio3d forwarding calls).
pub(super) fn speaker_array_coefficients(ctx: &CpuContext, dst: u64, n: u32) -> HleResult {
    write_coefficients(ctx, dst, n, None)
}

/// (handle, pos [xmm], spread [xmm], float *coeffs, u32 num, u8 heightAware, radius [xmm])
fn cb_get_speaker_array_coefficients(ctx: &mut CpuContext) -> HleResult {
    write_coefficients(ctx, ctx.rsi(), ctx.rdx() as u32, None)
}

/// (handle, ambiChannel, float *coeffs, u32 num)
fn cb_get_speaker_array_ambisonics_coefficients(ctx: &mut CpuContext) -> HleResult {
    write_coefficients(ctx, ctx.rdx(), ctx.rcx() as u32, Some(ctx.rsi() as u32))
}

/// One speaker array; its handle is the slot's address.
#[derive(Clone, Copy, Default)]
struct SpeakerArray {
    used: bool,
    /// Speakers the array was created for (the coefficient count a mix
    /// query must name).
    speakers: u32,
}

static SPEAKER_ARRAYS: Mutex<[SpeakerArray; SPEAKER_ARRAYS_MAX]> =
    Mutex::new([SpeakerArray { used: false, speakers: 0 }; SPEAKER_ARRAYS_MAX]);

fn speaker_array_handle(a: &SpeakerArray) -> u64 {
    a as *const SpeakerArray as u64
}

/// A slot of the speaker array table, or None when all are in use.
pub fn speaker_array_create() -> Option<u64> {
    speaker_array_create_sized(2)
}

/// A slot of the speaker array table for `speakers` speakers, or None
/// when all are in use.
pub fn speaker_array_create_sized(speakers: u32) -> Option<u64> {
    let mut table = locked(&SPEAKER_ARRAYS);
    let a = table.iter_mut().find(|a| !a.used)?;
    *a = SpeakerArray { used: true, speakers };
    Some(speaker_array_handle(a))
}

/// The speakers of a live speaker array.
pub fn speaker_array_speakers(handle: u64) -> Option<u32> {
    let table = locked(&SPEAKER_ARRAYS);
    table.iter().find(|a| a.used && speaker_array_handle(a) == handle).map(|a| a.speakers)
}

/// Destroy a live speaker array; false when `handle` names none.
pub fn speaker_array_destroy(handle: u64) -> bool {
    let mut table = locked(&SPEAKER_ARRAYS);
    match table.iter_mut().find(|a| a.used && speaker_array_handle(a) == handle) {
        Some(a) => {
            *a = SpeakerArray::default();
            true
        }
        None => false,
    }
}

/// sceAudioOut2SpeakerArrayCreate(handle *out, const VbapParam *, const AmbiParam *)
fn cb_speaker_array_create(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return AUDIO_OUT2_ERROR_INVALID_POINTER;
    }
    let handle = speaker_array_create();
    if !ctx.write_u64(out, handle.unwrap_or(0)) {
        return AUDIO_OUT2_ERROR_INVALID_POINTER;
    }
    if handle.is_some() {
        0
    } else {
        AUDIO_OUT2_ERROR_PORT_FULL
    }
}

fn cb_speaker_array_destroy(ctx: &mut CpuContext) -> HleResult {
    speaker_array_destroy(ctx.rdi());
    0
}

/// sceAudioOut2UserCreate(u32 userId, SceAudioOut2UserHandle *out)
fn cb_user_create(ctx: &mut CpuContext) -> HleResult {
    let user_id = ctx.rdi() as i32;
    let out = ctx.rsi();
    if out == 0 {
        return AUDIO_OUT2_ERROR_INVALID_PARAM;
    }
    if user_id == -1 {
        return AUDIO_OUT2_ERROR_INVALID_USER;
    }
    let mut handle = 0;
    let r = OUT2.user_create(Some(&mut handle));
    if r != 0 {
        return r;
    }
    if ctx.write_u64(out, handle) {
        0
    } else {
        AUDIO_OUT2_ERROR_INVALID_POINTER
    }
}

fn cb_user_destroy(ctx: &mut CpuContext) -> HleResult {
    OUT2.user_destroy(ctx.rdi())
}

/// (user, u32 *contextSupported, u32 *portSupported)
fn cb_user_get_supported_attributes(ctx: &mut CpuContext) -> HleResult {
    let (c, p) = (ctx.rsi(), ctx.rdx());
    let (mut ca, mut pa) = (0, 0);
    let r = OUT2.user_get_supported_attributes(
        ctx.rdi(),
        (c != 0).then_some(&mut ca),
        (p != 0).then_some(&mut pa),
    );
    if r != 0 {
        return r;
    }
    if !ctx.write_u32(c, ca) || !ctx.write_u32(p, pa) {
        return AUDIO_OUT2_ERROR_INVALID_POINTER;
    }
    0
}

/// sceAudioOut2Set3DLatency(u32 userId, u32 output, u32 latencyUs)
fn cb_set_3d_latency(ctx: &mut CpuContext) -> HleResult {
    OUT2.set_3d_latency(ctx.rdi() as u32, ctx.rsi() as u32, ctx.rdx() as u32)
}

/// Mastering states for `states_id`, or None for an unknown id.
fn mastering_states(states_id: u32) -> Option<Vec<u8>> {
    let (compressor_id, compressor_size) = match states_id {
        MASTERING_STATES_ID_DEFAULT => (MASTERING_STRUCT_ID_COMPRESSOR_DEFAULT, MASTERING_COMPRESSOR_SIZE),
        MASTERING_STATES_ID_V2 => (MASTERING_STRUCT_ID_COMPRESSOR_V2, MASTERING_COMPRESSOR_V2_SIZE),
        _ => return None,
    };
    let mut s = vec![0u8; MASTERING_HEADER_SIZE + compressor_size + MASTERING_LIMITER_SIZE];
    let mut put = |o: usize, v: u32| s[o..o + 4].copy_from_slice(&v.to_le_bytes());
    put(0, states_id);
    let c = MASTERING_HEADER_SIZE;
    put(c, compressor_id);
    put(c + 4, compressor_size as u32);
    let l = c + compressor_size;
    put(l, MASTERING_STRUCT_ID_LIMITER_DEFAULT);
    put(l + 4, MASTERING_LIMITER_SIZE as u32);
    Some(s)
}

/// sceAudioOut2MasteringGetState(SceAudioOut2MasteringStatesHeader *state, u32 output, user)
fn cb_mastering_get_state(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return AUDIO_OUT2_ERROR_MASTERING_INVALID_API_PARAM;
    }
    let Some(states_id) = ctx.read_u32(addr) else {
        return AUDIO_OUT2_ERROR_MASTERING_INVALID_API_PARAM;
    };
    let Some(states) = mastering_states(states_id) else {
        return AUDIO_OUT2_ERROR_MASTERING_INVALID_STATES_ID;
    };
    if !ctx.write_bytes(addr, &states) {
        return AUDIO_OUT2_ERROR_MASTERING_INVALID_API_PARAM;
    }
    0
}

const LIB: &str = "libSceAudioOut2";

/// Chat audio is allowed alongside the title's (sceAudioOut2EnableChat);
/// the host has no chat audio to mix.
static CHAT_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// sceAudioOut2EnableChat()
fn cb_enable_chat(_ctx: &mut CpuContext) -> HleResult {
    if !CHAT_ENABLED.swap(true, Ordering::SeqCst) {
        log::debug!("[AUDIO] AudioOut2 chat enabled (no chat audio on the host)");
    }
    0
}

/// sceAudioOut2DisableChat()
fn cb_disable_chat(_ctx: &mut CpuContext) -> HleResult {
    if CHAT_ENABLED.swap(false, Ordering::SeqCst) {
        log::debug!("[AUDIO] AudioOut2 chat disabled");
    }
    0
}

pub(crate) static EXPORTS: &[Export] = &[
    ("g2tViFIohHE", "sceAudioOut2Initialize", LIB, cb_audio_out2_init),
    ("oiOi7vgPfAY", "sceAudioOut2EnableChat", LIB, cb_enable_chat),
    ("vzvznsnpjKM", "sceAudioOut2DisableChat", LIB, cb_disable_chat),
    ("XHl38ZNknbs", "sceAudioOut2MasteringInit", LIB, cb_ok),
    ("v8iOE+j8a5o", "sceAudioOut2MasteringSetParam", LIB, cb_mastering_set_param),
    ("VZidxi2cYh0", "sceAudioOut2MasteringGetState", LIB, cb_mastering_get_state),
    ("2bbBBOkH4CY", "sceAudioOut2MasteringTerm", LIB, cb_ok),
    ("b1HWDUC8zaE", "sceAudioOut2SetSystemDebugState", LIB, cb_set_system_debug_state),
    ("TViD1EZXkNI", "sceAudioOut2Set3DLatency", LIB, cb_set_3d_latency),
    ("t5YrizufpQc", "sceAudioOut2ContextResetParam", LIB, cb_context_reset_param),
    ("pDmme7Bgm6E", "sceAudioOut2ContextQueryMemory", LIB, cb_context_query_memory),
    ("0x6o1VVAYSY", "sceAudioOut2ContextCreate", LIB, cb_context_create),
    ("on6ZH7Abo10", "sceAudioOut2ContextDestroy", LIB, cb_context_destroy),
    ("DxGyV8dtOR8", "sceAudioOut2ContextBedWrite", LIB, cb_context_bed_write),
    ("aII9h5nli9U", "sceAudioOut2ContextPush", LIB, cb_context_push),
    ("PE2zHMqLSHs", "sceAudioOut2ContextAdvance", LIB, cb_context_advance),
    ("R7d0F1g2qsU", "sceAudioOut2ContextGetQueueLevel", LIB, cb_context_get_queue_level),
    ("Q8DZkKQ-SYc", "sceAudioOut2LoContextGetQueueLevel", LIB, cb_context_get_queue_level),
    // The low-latency (Lo) calls take the standard calls' arguments and are
    // served by them.
    ("9kkkQp8Xz1Y", "sceAudioOut2LoInit", LIB, cb_ok),
    ("ui6EuBxIaMM", "sceAudioOut2LoContextQueryMemory", LIB, cb_context_query_memory),
    ("ECcNKhS8wvg", "sceAudioOut2LoContextCreate", LIB, cb_context_create),
    ("eSuD0DPJpvE", "sceAudioOut2LoContextDestroy", LIB, cb_context_destroy),
    ("NZu1Z2k14DM", "sceAudioOut2LoContextSetAttributes", LIB, cb_context_set_attributes),
    ("aa2y9JoZwDc", "sceAudioOut2LoContextAdvance", LIB, cb_context_advance),
    ("Hq2VRfX3OOk", "sceAudioOut2LoContextPush", LIB, cb_context_push),
    ("wgaUyb3bh0o", "sceAudioOut2LoPortCreate", LIB, cb_port_create),
    ("9bpjp8T0nRg", "sceAudioOut2LoPortDestroy", LIB, cb_port_destroy),
    ("yiOhxHhSzC0", "sceAudioOut2LoPortSetAttributes", LIB, cb_port_set_attributes),
    ("xaZ3K60Wwz0", "sceAudioOut2LoPortGetState", LIB, cb_port_get_state),
    ("8XTArSPyWHk", "sceAudioOut2PortSetAttributes", LIB, cb_port_set_attributes),
    ("JK2wamZPzwM", "sceAudioOut2PortCreate", LIB, cb_port_create),
    ("gatEUKG+Ea4", "sceAudioOut2PortGetState", LIB, cb_port_get_state),
    ("4dq2rblWlg0", "sceAudioOut2ContextSetAttributes", LIB, cb_context_set_attributes),
    ("bkBN+CMLwRc", "sceAudioOut2GetSystemState", LIB, cb_get_system_state),
    ("DImz2Ft9E2g", "sceAudioOut2GetSpeakerInfo", LIB, cb_get_speaker_info),
    ("G1YOKDJYX2Y", "sceAudioOut2GetSpeakerArrayMemorySize", LIB, cb_get_speaker_array_memory_size),
    ("4BlZurolOAo", "sceAudioOut2GetSpeakerArrayCoefficients", LIB, cb_get_speaker_array_coefficients),
    (
        "28QqMnuuJ9Y",
        "sceAudioOut2GetSpeakerArrayAmbisonicsCoefficients",
        LIB,
        cb_get_speaker_array_ambisonics_coefficients,
    ),
    ("+k91hoTuoA8", "sceAudioOut2SpeakerArrayCreate", LIB, cb_speaker_array_create),
    ("erCWQR5eKiQ", "sceAudioOut2SpeakerArrayDestroy", LIB, cb_speaker_array_destroy),
    ("cd+Rtw+D1x8", "sceAudioOut2PortDestroy", LIB, cb_port_destroy),
    ("IaZXJ9M79uo", "sceAudioOut2UserDestroy", LIB, cb_user_destroy),
    ("xywYcRB7nbQ", "sceAudioOut2UserCreate", LIB, cb_user_create),
    ("iE8trxPKnAg", "sceAudioOut2UserGetSupportedAttributes", LIB, cb_user_get_supported_attributes),
];

#[cfg(test)]
mod tests;
