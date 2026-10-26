// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceAudio3d: 3D audio ports.
//!
//! A port is a ring of `queueDepth` buffers, each Empty, Ready (advanced,
//! not yet pushed) or Play (pushed, playing). `PortAdvance` closes the
//! current buffer and moves on; `PortPush` starts every Ready buffer.
//! Pushed buffers play one after another, `granularity / 48 kHz` each, on
//! the process clock; no PCM reaches the host (objects and beds are not
//! rendered). A blocking push waits until the buffer after the pushed ones
//! has finished playing, which keeps exactly one pushed buffer ahead of
//! playback. `PortFlush` drops every queued and playing buffer.
//!
//! Objects are reserved ids (at most `maxObjects`, never 0 or 0xFFFFFFFF);
//! their attributes are validated but render nothing.

use super::Export;
use crate::context::CpuContext;
use crate::registry::HleResult;
use std::collections::BTreeSet;
use std::sync::{LazyLock, Mutex, MutexGuard};
use std::time::Duration;

pub const AUDIO3D_ERROR_INVALID_PORT: HleResult = 0x80EA_0002u32 as i32 as i64;
pub const AUDIO3D_ERROR_INVALID_OBJECT: HleResult = 0x80EA_0003u32 as i32 as i64;
pub const AUDIO3D_ERROR_INVALID_PARAMETER: HleResult = 0x80EA_0004u32 as i32 as i64;
pub const AUDIO3D_ERROR_OUT_OF_RESOURCES: HleResult = 0x80EA_0006u32 as i32 as i64;
pub const AUDIO3D_ERROR_NOT_READY: HleResult = 0x80EA_0007u32 as i32 as i64;
pub const AUDIO3D_ERROR_NOT_SUPPORTED: HleResult = 0x80EA_0008u32 as i32 as i64;
/// SCE_AUDIO3D_BUFFER_ADVANCE_AND_PUSH: the only buffer mode PortAdvance
/// drives.
const BUFFER_ADVANCE_AND_PUSH: u32 = 2;

pub const MAX_PORTS: usize = 4;
const RATE: u64 = 48_000;

pub const ATTRIBUTE_LATE_REVERB_LEVEL: u32 = 0x10001;
pub const ATTRIBUTE_DOWNMIX_SPREAD_RADIUS: u32 = 0x10002;
pub const ATTRIBUTE_DOWNMIX_SPREAD_HEIGHT_AWARE: u32 = 0x10003;
/// Object attribute that takes no value.
pub const ATTRIBUTE_RESET_STATE: u32 = 0x20000;
pub const OBJECT_INVALID: u32 = u32::MAX;
/// SceAudio3dAttribute: u32 attributeId, u32 pad, const void *value, size_t valueSize.
const ATTRIBUTE_SIZE: u64 = 0x18;

/// SceAudio3dOpenParameters: size_t size, u32 granularity, u32 rate,
/// u32 maxObjects, u32 queueDepth, u32 bufferMode, u32 pad.
pub const OPEN_PARAMETERS_SIZE: usize = 0x20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenParameters {
    pub size: u64,
    pub granularity: u32,
    pub rate: u32,
    pub max_objects: u32,
    pub queue_depth: u32,
    pub buffer_mode: u32,
}

impl Default for OpenParameters {
    fn default() -> Self {
        Self { size: OPEN_PARAMETERS_SIZE as u64, granularity: 256, rate: 0, max_objects: 512, queue_depth: 2, buffer_mode: 2 }
    }
}

impl OpenParameters {
    pub fn to_bytes(self) -> [u8; OPEN_PARAMETERS_SIZE] {
        let mut b = [0u8; OPEN_PARAMETERS_SIZE];
        b[0..8].copy_from_slice(&self.size.to_le_bytes());
        for (i, v) in [self.granularity, self.rate, self.max_objects, self.queue_depth, self.buffer_mode].iter().enumerate() {
            b[8 + i * 4..12 + i * 4].copy_from_slice(&v.to_le_bytes());
        }
        b
    }

    pub fn from_bytes(b: &[u8; OPEN_PARAMETERS_SIZE]) -> Self {
        let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        Self {
            size: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            granularity: u(8),
            rate: u(0x0C),
            max_objects: u(0x10),
            queue_depth: u(0x14),
            buffer_mode: u(0x18),
        }
    }
}

/// Time and sleeping.
pub trait Clock: Send + Sync {
    /// Process time in microseconds.
    fn now_us(&self) -> u64;
    /// Sleep with no lock held (guest signals may be delivered).
    fn sleep_us(&self, us: u64);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum BufferState {
    #[default]
    Empty,
    Ready,
    Play,
}

#[derive(Clone, Copy, Default)]
struct Buffer {
    state: BufferState,
    /// When a playing buffer has finished.
    done_at: u64,
}

#[derive(Default)]
struct Port {
    used: bool,
    buffers: Vec<Buffer>,
    index: usize,
    /// Playing time of one buffer.
    delay_us: u64,
    /// When the last pushed buffer finishes.
    play_tail: u64,
    late_reverb_level: f32,
    downmix_spread_radius: f32,
    downmix_spread_height_aware: i32,
    max_objects: u32,
    objects: BTreeSet<u32>,
    next_object_id: u32,
    buffer_mode: u32,
}

impl Port {
    fn depth(&self) -> usize {
        self.buffers.len()
    }

    /// Retire the buffers that finished playing.
    fn refresh(&mut self, now: u64) {
        for b in &mut self.buffers {
            if b.state == BufferState::Play && b.done_at <= now {
                b.state = BufferState::Empty;
            }
        }
    }
}

pub struct Audio3d<C: Clock> {
    clock: C,
    ports: Mutex<[Port; MAX_PORTS]>,
    /// sceAudio3dInitialize was called (and not terminated).
    initialized: std::sync::atomic::AtomicBool,
}

/// Port attributes sceAudio3dPortGetAttributesSupported lists.
pub const SUPPORTED_PORT_ATTRIBUTES: [u32; 3] =
    [ATTRIBUTE_LATE_REVERB_LEVEL, ATTRIBUTE_DOWNMIX_SPREAD_RADIUS, ATTRIBUTE_DOWNMIX_SPREAD_HEIGHT_AWARE];

impl<C: Clock> Audio3d<C> {
    pub fn new(clock: C) -> Self {
        Self { clock, ports: Mutex::new(Default::default()), initialized: Default::default() }
    }

    pub fn initialize(&self) {
        self.initialized.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Not ready unless initialized with every port closed.
    pub fn terminate(&self) -> HleResult {
        let ports = self.lock();
        if !self.initialized.load(std::sync::atomic::Ordering::SeqCst) || ports.iter().any(|p| p.used) {
            return AUDIO3D_ERROR_NOT_READY;
        }
        self.initialized.store(false, std::sync::atomic::Ordering::SeqCst);
        0
    }

    pub fn port_close(&self, id: u32) -> HleResult {
        let mut ports = self.lock();
        match ports.get_mut(id as usize) {
            Some(p) if p.used => {
                *p = Port::default();
                0
            }
            _ => AUDIO3D_ERROR_INVALID_PORT,
        }
    }

    pub fn port_is_open(&self, id: u32) -> bool {
        self.lock().get(id as usize).is_some_and(|p| p.used)
    }

    fn lock(&self) -> MutexGuard<'_, [Port; MAX_PORTS]> {
        self.ports.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn with_port<R>(&self, id: u32, f: impl FnOnce(&mut Port, u64) -> R) -> Result<R, HleResult> {
        let now = self.clock.now_us();
        let mut ports = self.lock();
        match ports.get_mut(id as usize) {
            Some(p) if p.used => {
                p.refresh(now);
                Ok(f(p, now))
            }
            _ => Err(AUDIO3D_ERROR_INVALID_PORT),
        }
    }

    pub fn port_open(&self, params: &OpenParameters) -> Result<u32, HleResult> {
        if (params.size as usize) < OPEN_PARAMETERS_SIZE || params.queue_depth == 0 {
            return Err(AUDIO3D_ERROR_INVALID_PARAMETER);
        }
        let mut ports = self.lock();
        let Some(id) = ports.iter().position(|p| !p.used) else {
            return Err(AUDIO3D_ERROR_OUT_OF_RESOURCES);
        };
        ports[id] = Port {
            used: true,
            buffers: vec![Buffer::default(); params.queue_depth as usize],
            index: 0,
            delay_us: 1_000_000 * params.granularity as u64 / RATE,
            play_tail: 0,
            late_reverb_level: 0.0,
            downmix_spread_radius: 2.0,
            downmix_spread_height_aware: 0,
            max_objects: params.max_objects,
            objects: BTreeSet::new(),
            next_object_id: 0,
            buffer_mode: params.buffer_mode,
        };
        Ok(id as u32)
    }

    /// The next free object id after the last one handed out.
    pub fn object_reserve(&self, id: u32) -> Result<u32, HleResult> {
        self.with_port(id, |p, _| {
            if p.objects.len() >= p.max_objects as usize {
                return Err(AUDIO3D_ERROR_OUT_OF_RESOURCES);
            }
            loop {
                p.next_object_id = p.next_object_id.wrapping_add(1);
                let o = p.next_object_id;
                if o != 0 && o != OBJECT_INVALID && !p.objects.contains(&o) {
                    break;
                }
            }
            p.objects.insert(p.next_object_id);
            Ok(p.next_object_id)
        })?
    }

    pub fn object_unreserve(&self, id: u32, object: u32) -> HleResult {
        self.with_port(id, |p, _| if p.objects.remove(&object) { 0 } else { AUDIO3D_ERROR_INVALID_OBJECT })
            .unwrap_or_else(|e| e)
    }

    /// Validate a reserved object's attributes, `(id, value address)` each:
    /// every attribute but RESET_STATE needs a value.
    pub fn object_set_attributes(&self, id: u32, object: u32, attributes: Option<&[(u32, u64)]>) -> HleResult {
        self.with_port(id, |p, _| {
            let Some(attributes) = attributes.filter(|a| !a.is_empty()) else {
                return AUDIO3D_ERROR_INVALID_PARAMETER;
            };
            if !p.objects.contains(&object) {
                return AUDIO3D_ERROR_INVALID_OBJECT;
            }
            if attributes.iter().any(|&(a, value)| a != ATTRIBUTE_RESET_STATE && value == 0) {
                return AUDIO3D_ERROR_INVALID_PARAMETER;
            }
            0
        })
        .unwrap_or_else(|e| e)
    }

    /// Drop every advanced and playing buffer.
    pub fn port_flush(&self, id: u32) -> HleResult {
        self.with_port(id, |p, _| {
            p.buffers.fill(Buffer::default());
            p.play_tail = 0;
            0
        })
        .unwrap_or_else(|e| e)
    }

    /// `value` holds the attribute's bytes.
    pub fn port_set_attribute(&self, id: u32, attribute_id: u32, value: &[u8]) -> HleResult {
        let r = self.with_port(id, |p, _| {
            let word = |v: &[u8]| (v.len() == 4).then(|| [v[0], v[1], v[2], v[3]]);
            match attribute_id {
                ATTRIBUTE_LATE_REVERB_LEVEL => p.late_reverb_level = f32::from_le_bytes(word(value)?),
                ATTRIBUTE_DOWNMIX_SPREAD_RADIUS => p.downmix_spread_radius = f32::from_le_bytes(word(value)?),
                ATTRIBUTE_DOWNMIX_SPREAD_HEIGHT_AWARE => {
                    p.downmix_spread_height_aware = i32::from_le_bytes(word(value)?)
                }
                other => log::debug!("[AUDIO] Audio3d port {id}: attribute 0x{other:X} ignored"),
            }
            Some(())
        });
        match r {
            Ok(Some(())) => 0,
            Ok(None) => AUDIO3D_ERROR_INVALID_PARAMETER,
            Err(e) => e,
        }
    }

    /// (queue level, available): the Empty buffers from the current one on are available.
    pub fn port_get_queue_level(&self, id: u32) -> Result<(u32, u32), HleResult> {
        self.with_port(id, |p, _| {
            let depth = p.depth();
            let empty = (0..depth).take_while(|i| p.buffers[(p.index + i) % depth].state == BufferState::Empty).count();
            ((depth - empty) as u32, empty as u32)
        })
    }

    pub fn port_advance(&self, id: u32) -> HleResult {
        self.with_port(id, |p, _| {
            if p.buffer_mode != BUFFER_ADVANCE_AND_PUSH {
                log::warn!("[AUDIO] sceAudio3dPortAdvance: buffer mode {} is not supported", p.buffer_mode);
                return AUDIO3D_ERROR_NOT_SUPPORTED;
            }
            let current = &mut p.buffers[p.index];
            if current.state == BufferState::Empty {
                current.state = BufferState::Ready;
            }
            if current.state != BufferState::Ready {
                // The buffer is still playing: the queue is full.
                return AUDIO3D_ERROR_NOT_READY;
            }
            p.index = (p.index + 1) % p.depth();
            0
        })
        .unwrap_or_else(|e| e)
    }

    pub fn port_push(&self, id: u32, blocking: bool) -> HleResult {
        let pushed = self.with_port(id, |p, now| {
            let depth = p.depth();
            let first = p.index;
            let mut pushed = 0;
            for i in 0..depth {
                let k = (p.index + i) % depth;
                if p.buffers[k].state == BufferState::Ready {
                    p.play_tail = p.play_tail.max(now) + p.delay_us;
                    p.buffers[k] = Buffer { state: BufferState::Play, done_at: p.play_tail };
                    pushed += 1;
                }
            }
            (pushed, first)
        });
        let (pushed, first) = match pushed {
            Ok(v) => v,
            Err(e) => return e,
        };
        if pushed == 0 || !blocking {
            return 0;
        }
        // Wait for the buffer the next advance fills.
        loop {
            let wait = self.with_port(id, |p, now| {
                let b = p.buffers[first];
                match b.state {
                    BufferState::Empty => 0,
                    BufferState::Play => b.done_at.saturating_sub(now).max(1),
                    // Advanced again by another thread: nothing left to wait for.
                    BufferState::Ready => 0,
                }
            });
            match wait {
                Ok(0) | Err(_) => return 0,
                Ok(us) => self.clock.sleep_us(us),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The emulator's instance
// ---------------------------------------------------------------------------

struct HostClock;

impl Clock for HostClock {
    fn now_us(&self) -> u64 {
        crate::time::process_time_usec()
    }

    fn sleep_us(&self, us: u64) {
        crate::signals::sleep(Duration::from_micros(us));
    }
}

static AUDIO3D: LazyLock<Audio3d<HostClock>> = LazyLock::new(|| Audio3d::new(HostClock));

/// sceAudio3dInitialize(int64 reserved)
fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() != 0 {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    AUDIO3D.initialize();
    0
}

/// sceAudio3dTerminate()
fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    AUDIO3D.terminate()
}

/// sceAudio3dPortClose(port)
fn cb_port_close(ctx: &mut CpuContext) -> HleResult {
    AUDIO3D.port_close(ctx.rdi() as u32)
}

/// sceAudio3dPortGetAttributesSupported(port, SceAudio3dAttributeId *attributes, u32 *numAttributes):
/// without an array only the count is written; otherwise at most
/// *numAttributes ids are copied and the count written back.
fn cb_port_get_attributes_supported(ctx: &mut CpuContext) -> HleResult {
    let (id, out, count_addr) = (ctx.rdi() as u32, ctx.rsi(), ctx.rdx());
    if count_addr == 0 {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    if !AUDIO3D.port_is_open(id) {
        return AUDIO3D_ERROR_INVALID_PORT;
    }
    let count = if out == 0 {
        SUPPORTED_PORT_ATTRIBUTES.len()
    } else {
        let Some(room) = ctx.read_u32(count_addr) else { return AUDIO3D_ERROR_INVALID_PARAMETER };
        let n = (room as usize).min(SUPPORTED_PORT_ATTRIBUTES.len());
        let bytes: Vec<u8> = SUPPORTED_PORT_ATTRIBUTES[..n].iter().flat_map(|a| a.to_le_bytes()).collect();
        if !ctx.write_bytes(out, &bytes) {
            return AUDIO3D_ERROR_INVALID_PARAMETER;
        }
        n
    };
    if !ctx.write_u32(count_addr, count as u32) {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    0
}

/// void sceAudio3dGetDefaultOpenParameters(SceAudio3dOpenParameters *p)
fn cb_get_default_open_parameters(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr != 0 {
        ctx.write_bytes(addr, &OpenParameters::default().to_bytes());
    }
    0
}

/// sceAudio3dPortOpen(SceUserServiceUserId, const SceAudio3dOpenParameters *, SceAudio3dPortId *)
fn cb_port_open(ctx: &mut CpuContext) -> HleResult {
    let (user_id, param, out) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    if param == 0 || out == 0 {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    let mut raw = [0u8; OPEN_PARAMETERS_SIZE];
    if ctx.vm.read(param, &mut raw).is_err() {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    let params = OpenParameters::from_bytes(&raw);
    let id = match AUDIO3D.port_open(&params) {
        Ok(id) => id,
        Err(e) => return e,
    };
    log::info!(
        "[AUDIO] Audio3d port {id}: user {user_id}, granularity {}, rate {}, objects {}, queue {}, buffer mode {}",
        params.granularity,
        params.rate,
        params.max_objects,
        params.queue_depth,
        params.buffer_mode
    );
    if !ctx.write_u32(out, id) {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    0
}

/// sceAudio3dPortSetAttribute(port, SceAudio3dAttributeId, const void *attr, size_t size)
fn cb_port_set_attribute(ctx: &mut CpuContext) -> HleResult {
    let (id, attribute_id, value, size) = (ctx.rdi() as u32, ctx.rsi() as u32, ctx.rdx(), ctx.rcx());
    if value == 0 {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    let mut raw = [0u8; 8];
    let n = (size as usize).min(raw.len());
    if ctx.vm.read(value, &mut raw[..n]).is_err() {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    // Sizes the known attributes cannot have are rejected as such.
    let bytes = if size as usize > raw.len() { &raw[..0] } else { &raw[..n] };
    AUDIO3D.port_set_attribute(id, attribute_id, bytes)
}

/// sceAudio3dPortGetQueueLevel(port, u32 *queueLevel, u32 *queueAvailable)
fn cb_port_get_queue_level(ctx: &mut CpuContext) -> HleResult {
    let (id, level_out, available_out) = (ctx.rdi() as u32, ctx.rsi(), ctx.rdx());
    if level_out == 0 && available_out == 0 {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    let (level, available) = match AUDIO3D.port_get_queue_level(id) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if level_out != 0 && !ctx.write_u32(level_out, level) {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    if available_out != 0 && !ctx.write_u32(available_out, available) {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    0
}

fn cb_port_advance(ctx: &mut CpuContext) -> HleResult {
    AUDIO3D.port_advance(ctx.rdi() as u32)
}

/// sceAudio3dPortPush(port, SceAudio3dBlocking)
fn cb_port_push(ctx: &mut CpuContext) -> HleResult {
    AUDIO3D.port_push(ctx.rdi() as u32, ctx.rsi() as u32 != 0)
}

/// sceAudio3dObjectReserve(port, SceAudio3dObjectId *out)
fn cb_object_reserve(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    let (r, object) = match AUDIO3D.object_reserve(ctx.rdi() as u32) {
        Ok(o) => (0, o),
        Err(e) => (e, OBJECT_INVALID),
    };
    if !ctx.write_u32(out, object) {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    r
}

/// sceAudio3dObjectUnreserve(port, object)
fn cb_object_unreserve(ctx: &mut CpuContext) -> HleResult {
    AUDIO3D.object_unreserve(ctx.rdi() as u32, ctx.rsi() as u32)
}

/// Attributes read without allocating; longer lists go to the heap.
const ATTRIBUTES_INLINE: usize = 16;

/// sceAudio3dObjectSetAttributes(port, object, size_t num, const SceAudio3dAttribute *attrs)
fn cb_object_set_attributes(ctx: &mut CpuContext) -> HleResult {
    let (id, object, count, array) = (ctx.rdi() as u32, ctx.rsi() as u32, ctx.rdx(), ctx.rcx());
    if count == 0 || array == 0 {
        return AUDIO3D.object_set_attributes(id, object, None);
    }
    let mut inline = [(0u32, 0u64); ATTRIBUTES_INLINE];
    let mut heap = Vec::new();
    for i in 0..count {
        let mut e = [0u8; ATTRIBUTE_SIZE as usize];
        let Some(at) = i.checked_mul(ATTRIBUTE_SIZE).and_then(|o| array.checked_add(o)) else {
            return AUDIO3D_ERROR_INVALID_PARAMETER;
        };
        if ctx.vm.read(at, &mut e).is_err() {
            return AUDIO3D_ERROR_INVALID_PARAMETER;
        }
        let a = (u32::from_le_bytes(e[0..4].try_into().unwrap()), u64::from_le_bytes(e[8..16].try_into().unwrap()));
        if count as usize <= ATTRIBUTES_INLINE {
            inline[i as usize] = a;
        } else {
            heap.push(a);
        }
    }
    let list = if count as usize <= ATTRIBUTES_INLINE { &inline[..count as usize] } else { &heap[..] };
    AUDIO3D.object_set_attributes(id, object, Some(list))
}

/// sceAudio3dPortFlush(port)
fn cb_port_flush(ctx: &mut CpuContext) -> HleResult {
    AUDIO3D.port_flush(ctx.rdi() as u32)
}

// ---------------------------------------------------------------------------
// Speaker arrays (forwarded to the AudioOut2 ones)
// ---------------------------------------------------------------------------

/// sceAudio3dGetSpeakerArrayMemorySize(u32 numSpeakers, u8 is3d): the
/// AudioOut2 size without ambisonics; zero speakers are invalid.
fn cb_get_speaker_array_memory_size(ctx: &mut CpuContext) -> HleResult {
    let speakers = ctx.rdi() as u32;
    if speakers == 0 {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    super::audio_out2::speaker_array_memory_size(speakers, ctx.rsi() as u8 != 0, false) as HleResult
}

/// sceAudio3dCreateSpeakerArray(handle*, const Audio3dSpeakerArrayParameters*):
/// Audio3dSpeakerArrayParameters {positions@0, numSpeakers@8, is3d@0xC,
/// memory@0x10, memorySize@0x18}.
fn cb_create_speaker_array(ctx: &mut CpuContext) -> HleResult {
    let (out, params) = (ctx.rdi(), ctx.rsi());
    if out == 0 || params == 0 {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    ctx.write_u64(out, 0);
    let Some(b) = ctx.read_bytes(params, 0x20) else {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    };
    let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    let (positions, speakers, is_3d, memory, memory_size) = (u64_at(0), u32_at(8), b[0xC] != 0, u64_at(0x10), u64_at(0x18));
    let need = super::audio_out2::speaker_array_memory_size(speakers, is_3d, false);
    if positions == 0 || speakers == 0 || speakers > super::audio_out2::SPEAKER_ARRAY_MAX_SPEAKERS || memory == 0 || memory % 16 != 0 || memory_size < need {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    match super::audio_out2::speaker_array_create_sized(speakers) {
        Some(h) => {
            if ctx.write_u64(out, h) {
                0
            } else {
                AUDIO3D_ERROR_INVALID_PARAMETER
            }
        }
        None => AUDIO3D_ERROR_INVALID_PARAMETER,
    }
}

/// sceAudio3dDeleteSpeakerArray(handle)
fn cb_delete_speaker_array(ctx: &mut CpuContext) -> HleResult {
    if !super::audio_out2::speaker_array_destroy(ctx.rdi()) {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    0
}

/// sceAudio3dGetSpeakerArrayMixCoefficients(handle, pos, spread,
/// float* coefficients, u32 numCoefficients): the AudioOut2 coefficients
/// (position and spread feed a spatial mix this model has no part in); the
/// count must name the array's speakers and the pointer be readable.
fn cb_get_speaker_array_mix_coefficients(ctx: &mut CpuContext) -> HleResult {
    let (handle, coefficients, num) = (ctx.rdi(), ctx.rdx(), ctx.rcx() as u32);
    let Some(speakers) = super::audio_out2::speaker_array_speakers(handle) else {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    };
    if num != speakers || coefficients == 0 {
        return AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    super::audio_out2::speaker_array_coefficients(ctx, coefficients, num)
}

/// sceAudio3dGetSpeakerArrayMixCoefficients2(handle, pos, spread,
/// float* coefficients, u32 numCoefficients, u8 heightAware,
/// float downmixSpreadRadius): MixCoefficients with the extra parameters.
fn cb_get_speaker_array_mix_coefficients2(ctx: &mut CpuContext) -> HleResult {
    cb_get_speaker_array_mix_coefficients(ctx)
}

const LIB: &str = "libSceAudio3d";

pub(crate) static EXPORTS: &[Export] = &[
    ("UmCvjSmuZIw", "sceAudio3dInitialize", LIB, cb_initialize),
    ("WW1TS2iz5yc", "sceAudio3dTerminate", LIB, cb_terminate),
    ("OyVqOeVNtSk", "sceAudio3dPortClose", LIB, cb_port_close),
    ("9ZA23Ia46Po", "sceAudio3dPortGetAttributesSupported", LIB, cb_port_get_attributes_supported),
    ("Im+jOoa5WAI", "sceAudio3dGetDefaultOpenParameters", LIB, cb_get_default_open_parameters),
    ("XeDDK0xJWQA", "sceAudio3dPortOpen", LIB, cb_port_open),
    ("Yq9bfUQ0uJg", "sceAudio3dPortSetAttribute", LIB, cb_port_set_attribute),
    ("YaaDbDwKpFM", "sceAudio3dPortGetQueueLevel", LIB, cb_port_get_queue_level),
    ("lw0qrdSjZt8", "sceAudio3dPortAdvance", LIB, cb_port_advance),
    ("VEVhZ9qd4ZY", "sceAudio3dPortPush", LIB, cb_port_push),
    ("ZOGrxWLgQzE", "sceAudio3dPortFlush", LIB, cb_port_flush),
    ("jO2tec4dJ2M", "sceAudio3dObjectReserve", LIB, cb_object_reserve),
    ("1HXxo-+1qCw", "sceAudio3dObjectUnreserve", LIB, cb_object_unreserve),
    ("4uyHN9q4ZeU", "sceAudio3dObjectSetAttributes", LIB, cb_object_set_attributes),
    ("kEqqyDkmgdI", "sceAudio3dGetSpeakerArrayMemorySize", LIB, cb_get_speaker_array_memory_size),
    ("lvWMW6vEqFU", "sceAudio3dCreateSpeakerArray", LIB, cb_create_speaker_array),
    ("8hm6YdoQgwg", "sceAudio3dDeleteSpeakerArray", LIB, cb_delete_speaker_array),
    ("-R1DukFq7Dk", "sceAudio3dGetSpeakerArrayMixCoefficients", LIB, cb_get_speaker_array_mix_coefficients),
    ("-Re+pCWvwjQ", "sceAudio3dGetSpeakerArrayMixCoefficients2", LIB, cb_get_speaker_array_mix_coefficients2),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister;
    use std::sync::Arc;

    const BASE: u64 = 0x5CE4_0000_0000;

    #[derive(Clone, Default)]
    struct FakeClock(Arc<Mutex<(u64, Vec<u64>)>>);

    impl Clock for FakeClock {
        fn now_us(&self) -> u64 {
            self.0.lock().unwrap().0
        }
        fn sleep_us(&self, us: u64) {
            let mut c = self.0.lock().unwrap();
            c.0 += us;
            c.1.push(us);
        }
    }

    fn fixture() -> (Audio3d<FakeClock>, FakeClock) {
        let clock = FakeClock::default();
        clock.0.lock().unwrap().0 = 1_000_000;
        (Audio3d::new(clock.clone()), clock)
    }

    #[test]
    fn default_parameters_round_trip() {
        let d = OpenParameters::default();
        assert_eq!(d.granularity, 256);
        assert_eq!(d.max_objects, 512);
        assert_eq!(d.queue_depth, 2);
        assert_eq!(d.buffer_mode, 2);
        assert_eq!(OpenParameters::from_bytes(&d.to_bytes()), d);
    }

    #[test]
    fn port_table_and_validation() {
        let (a, _) = fixture();
        let d = OpenParameters::default();
        for id in 0..MAX_PORTS as u32 {
            assert_eq!(a.port_open(&d), Ok(id));
        }
        assert_eq!(a.port_open(&d), Err(AUDIO3D_ERROR_OUT_OF_RESOURCES));
        let (a, _) = fixture();
        assert_eq!(a.port_open(&OpenParameters { queue_depth: 0, ..d }), Err(AUDIO3D_ERROR_INVALID_PARAMETER));
        assert_eq!(a.port_open(&OpenParameters { size: 0x10, ..d }), Err(AUDIO3D_ERROR_INVALID_PARAMETER));
        assert_eq!(a.port_advance(0), AUDIO3D_ERROR_INVALID_PORT);
        assert_eq!(a.port_push(9, true), AUDIO3D_ERROR_INVALID_PORT);
        assert_eq!(a.port_get_queue_level(3), Err(AUDIO3D_ERROR_INVALID_PORT));
        let id = a.port_open(&d).unwrap();
        assert_eq!(a.port_set_attribute(id, ATTRIBUTE_LATE_REVERB_LEVEL, &0.5f32.to_le_bytes()), 0);
        assert_eq!(a.port_set_attribute(id, ATTRIBUTE_DOWNMIX_SPREAD_RADIUS, &[0; 2]), AUDIO3D_ERROR_INVALID_PARAMETER);
        assert_eq!(a.port_set_attribute(id, ATTRIBUTE_DOWNMIX_SPREAD_HEIGHT_AWARE, &1i32.to_le_bytes()), 0);
        {
            let p = &a.lock()[id as usize];
            assert_eq!((p.late_reverb_level, p.downmix_spread_radius, p.downmix_spread_height_aware), (0.5, 2.0, 1));
        }
        // A port in another buffer mode is not advanced (the module's
        // NOT_SUPPORTED).
        let manual = a.port_open(&OpenParameters { buffer_mode: 1, ..d }).unwrap();
        assert_eq!(a.port_advance(manual), AUDIO3D_ERROR_NOT_SUPPORTED);
    }

    #[test]
    fn advance_push_and_queue_level() {
        let (a, clock) = fixture();
        let id = a.port_open(&OpenParameters::default()).unwrap();
        // 256 samples at 48 kHz.
        let delay = 5333;
        assert_eq!(a.port_get_queue_level(id), Ok((0, 2)));
        assert_eq!(a.port_advance(id), 0);
        assert_eq!(a.port_get_queue_level(id), Ok((1, 1)), "only the empty buffers from the current one on are available");
        // The buffer after the pushed one is empty: no wait.
        assert_eq!(a.port_push(id, true), 0);
        assert!(clock.0.lock().unwrap().1.is_empty());
        assert_eq!(a.port_get_queue_level(id), Ok((1, 1)));
        assert_eq!(a.port_advance(id), 0);
        assert_eq!(a.port_get_queue_level(id), Ok((2, 0)));
        // The queue is full until the first buffer has played.
        assert_eq!(a.port_advance(id), AUDIO3D_ERROR_NOT_READY);
        let start = clock.now_us();
        assert_eq!(a.port_push(id, true), 0);
        assert_eq!(clock.now_us() - start, delay, "blocking push waits for the first buffer only");
        assert_eq!(a.port_get_queue_level(id), Ok((1, 1)));
        // Steady state: one blocking push per buffer period.
        let start = clock.now_us();
        for _ in 0..100 {
            assert_eq!(a.port_advance(id), 0);
            assert_eq!(a.port_push(id, true), 0);
        }
        let el = clock.now_us() - start;
        assert!((100 * delay - delay..=100 * delay).contains(&el), "{el}");
        // A non-blocking push never sleeps; nothing pushed never waits.
        let sleeps = clock.0.lock().unwrap().1.len();
        assert_eq!(a.port_push(id, true), 0);
        assert_eq!(a.port_push(id, false), 0);
        assert_eq!(clock.0.lock().unwrap().1.len(), sleeps);
    }

    #[test]
    fn flush_drops_queued_buffers() {
        let (a, _) = fixture();
        assert_eq!(a.port_flush(0), AUDIO3D_ERROR_INVALID_PORT, "flush needs an open port");
        let id = a.port_open(&OpenParameters::default()).unwrap();
        assert_eq!(a.port_advance(id), 0);
        assert_eq!(a.port_get_queue_level(id), Ok((1, 1)), "one buffer queued");
        assert_eq!(a.port_flush(id + 1), AUDIO3D_ERROR_INVALID_PORT);
        assert_eq!(a.port_flush(id), 0);
        assert_eq!(a.port_get_queue_level(id), Ok((0, 2)), "flush drops the queued buffers");
        // Playing buffers go too.
        assert_eq!(a.port_advance(id), 0);
        assert_eq!(a.port_push(id, false), 0);
        assert_eq!(a.port_advance(id), 0);
        assert_eq!(a.port_get_queue_level(id), Ok((2, 0)));
        assert_eq!(a.port_flush(id), 0);
        assert_eq!(a.port_get_queue_level(id), Ok((0, 2)));
    }

    #[test]
    fn objects_and_attributes() {
        let (a, _) = fixture();
        let value = 0x1000u64;
        let one = [(ATTRIBUTE_LATE_REVERB_LEVEL, value)];
        assert_eq!(a.object_set_attributes(0, 1, Some(&one)), AUDIO3D_ERROR_INVALID_PORT, "attributes need an open port");
        assert_eq!(a.object_reserve(0), Err(AUDIO3D_ERROR_INVALID_PORT));
        let id = a.port_open(&OpenParameters { max_objects: 2, ..OpenParameters::default() }).unwrap();
        assert_eq!(a.object_reserve(id), Ok(1));
        assert_eq!(a.object_set_attributes(id + 1, 1, Some(&one)), AUDIO3D_ERROR_INVALID_PORT);
        assert_eq!(a.object_set_attributes(id, 1, Some(&[])), AUDIO3D_ERROR_INVALID_PARAMETER, "zero attributes");
        assert_eq!(a.object_set_attributes(id, 1, None), AUDIO3D_ERROR_INVALID_PARAMETER, "null array");
        assert_eq!(a.object_set_attributes(id, 2, Some(&one)), AUDIO3D_ERROR_INVALID_OBJECT, "unreserved object");
        assert_eq!(a.object_set_attributes(id, 1, Some(&one)), 0);
        assert_eq!(a.object_set_attributes(id, 1, Some(&[(ATTRIBUTE_RESET_STATE, 0)])), 0, "reset state takes no value");
        assert_eq!(
            a.object_set_attributes(id, 1, Some(&[(ATTRIBUTE_RESET_STATE, 0), (ATTRIBUTE_LATE_REVERB_LEVEL, 0)])),
            AUDIO3D_ERROR_INVALID_PARAMETER,
            "a value is required"
        );
        assert_eq!(a.object_reserve(id), Ok(2));
        assert_eq!(a.object_reserve(id), Err(AUDIO3D_ERROR_OUT_OF_RESOURCES), "more objects than maxObjects");
        assert_eq!(a.object_unreserve(id, 1), 0);
        assert_eq!(a.object_unreserve(id, 1), AUDIO3D_ERROR_INVALID_OBJECT);
        assert_eq!(a.object_reserve(id), Ok(3), "ids move on");
    }

    #[test]
    fn exports_through_guest_memory() {
        let vm = crate::sync::test_util::vm_arena(BASE);
        let mut ctx = CpuContext::new(vm);
        ctx.set(CpuRegister::Rdi, 1);
        assert_eq!(cb_initialize(&mut ctx), AUDIO3D_ERROR_INVALID_PARAMETER);
        ctx.set(CpuRegister::Rdi, 0);
        assert_eq!(cb_initialize(&mut ctx), 0);
        ctx.set(CpuRegister::Rdi, BASE);
        assert_eq!(cb_get_default_open_parameters(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE), Some(0x20));
        assert_eq!(ctx.read_u32(BASE + 8), Some(256));
        assert_eq!(ctx.read_u32(BASE + 0x14), Some(2));
        ctx.set(CpuRegister::Rdi, 255);
        ctx.set(CpuRegister::Rsi, BASE);
        ctx.set(CpuRegister::Rdx, 0);
        assert_eq!(cb_port_open(&mut ctx), AUDIO3D_ERROR_INVALID_PARAMETER);
        ctx.set(CpuRegister::Rdx, BASE + 0x100);
        assert_eq!(cb_port_open(&mut ctx), 0);
        let id = ctx.read_u32(BASE + 0x100).unwrap();
        ctx.set(CpuRegister::Rdi, id as u64);
        ctx.set(CpuRegister::Rsi, 0);
        ctx.set(CpuRegister::Rdx, 0);
        assert_eq!(cb_port_get_queue_level(&mut ctx), AUDIO3D_ERROR_INVALID_PARAMETER);
        ctx.set(CpuRegister::Rsi, BASE + 0x200);
        ctx.set(CpuRegister::Rdx, BASE + 0x204);
        assert_eq!(cb_port_get_queue_level(&mut ctx), 0);
        assert_eq!((ctx.read_u32(BASE + 0x200), ctx.read_u32(BASE + 0x204)), (Some(0), Some(2)));
        ctx.write_u32(BASE + 0x300, 3f32.to_bits());
        ctx.set(CpuRegister::Rsi, ATTRIBUTE_DOWNMIX_SPREAD_RADIUS as u64);
        ctx.set(CpuRegister::Rdx, BASE + 0x300);
        ctx.set(CpuRegister::Rcx, 4);
        assert_eq!(cb_port_set_attribute(&mut ctx), 0);
        ctx.set(CpuRegister::Rcx, 16);
        assert_eq!(cb_port_set_attribute(&mut ctx), AUDIO3D_ERROR_INVALID_PARAMETER);
        assert_eq!(cb_port_advance(&mut ctx), 0);
        ctx.set(CpuRegister::Rsi, 0);
        assert_eq!(cb_port_push(&mut ctx), 0);

        // Objects and their attributes through guest memory.
        assert_eq!(cb_object_reserve(&mut ctx), AUDIO3D_ERROR_INVALID_PARAMETER, "null object output");
        ctx.set(CpuRegister::Rsi, BASE + 0x400);
        assert_eq!(cb_object_reserve(&mut ctx), 0);
        let object = ctx.read_u32(BASE + 0x400).unwrap();
        assert_ne!(object, OBJECT_INVALID);
        // { LATE_REVERB_LEVEL, &value }, { RESET_STATE, null }
        ctx.write_u32(BASE + 0x500, ATTRIBUTE_LATE_REVERB_LEVEL);
        ctx.write_u64(BASE + 0x508, BASE + 0x300);
        ctx.write_u64(BASE + 0x510, 4);
        ctx.write_u32(BASE + 0x518, ATTRIBUTE_RESET_STATE);
        ctx.write_u64(BASE + 0x520, 0);
        ctx.write_u64(BASE + 0x528, 0);
        ctx.set(CpuRegister::Rsi, object as u64);
        ctx.set(CpuRegister::Rdx, 2);
        ctx.set(CpuRegister::Rcx, BASE + 0x500);
        assert_eq!(cb_object_set_attributes(&mut ctx), 0);
        ctx.set(CpuRegister::Rcx, 0);
        assert_eq!(cb_object_set_attributes(&mut ctx), AUDIO3D_ERROR_INVALID_PARAMETER);
        ctx.set(CpuRegister::Rcx, BASE + 0x500);
        ctx.write_u64(BASE + 0x508, 0);
        assert_eq!(cb_object_set_attributes(&mut ctx), AUDIO3D_ERROR_INVALID_PARAMETER);
        assert_eq!(cb_object_unreserve(&mut ctx), 0);
        assert_eq!(cb_object_unreserve(&mut ctx), AUDIO3D_ERROR_INVALID_OBJECT);
        assert_eq!(cb_port_flush(&mut ctx), 0);

        // Supported port attributes: the count alone, then a short array.
        let (attrs, count) = (BASE + 0x600, BASE + 0x700);
        ctx.set(CpuRegister::Rsi, attrs);
        ctx.set(CpuRegister::Rdx, 0);
        assert_eq!(cb_port_get_attributes_supported(&mut ctx), AUDIO3D_ERROR_INVALID_PARAMETER);
        ctx.set(CpuRegister::Rsi, 0);
        ctx.set(CpuRegister::Rdx, count);
        assert_eq!(cb_port_get_attributes_supported(&mut ctx), 0);
        assert_eq!(ctx.read_u32(count), Some(3));
        ctx.write_bytes(attrs, &[0xEE; 12]);
        ctx.write_u32(count, 2);
        ctx.set(CpuRegister::Rsi, attrs);
        assert_eq!(cb_port_get_attributes_supported(&mut ctx), 0);
        assert_eq!(ctx.read_u32(count), Some(2));
        assert_eq!((ctx.read_u32(attrs), ctx.read_u32(attrs + 4)), (Some(0x10001), Some(0x10002)));
        assert_eq!(ctx.read_u32(attrs + 8), Some(0xEEEE_EEEE));
        ctx.write_u32(count, 8);
        assert_eq!(cb_port_get_attributes_supported(&mut ctx), 0);
        assert_eq!(ctx.read_u32(count), Some(3));
        assert_eq!(ctx.read_u32(attrs + 8), Some(0x10003));

        // Terminate needs every port closed.
        assert_eq!(cb_terminate(&mut ctx), AUDIO3D_ERROR_NOT_READY, "a port is open");
        assert_eq!(cb_port_close(&mut ctx), 0);
        assert_eq!(cb_port_close(&mut ctx), AUDIO3D_ERROR_INVALID_PORT);
        assert_eq!(cb_port_get_attributes_supported(&mut ctx), AUDIO3D_ERROR_INVALID_PORT);
        assert_eq!(cb_port_advance(&mut ctx), AUDIO3D_ERROR_INVALID_PORT);
        assert_eq!(cb_terminate(&mut ctx), 0);
        assert_eq!(cb_terminate(&mut ctx), AUDIO3D_ERROR_NOT_READY, "not initialized");
    }
}
