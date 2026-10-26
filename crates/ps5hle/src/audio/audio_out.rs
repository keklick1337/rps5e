// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceAudioOut: output ports, shared with libSceAudioOut2.
//!
//! An `Audio` holds 32 output ports. Each port (except vibration ports)
//! owns a host stream in the guest's layout, at most 8 channels: 7.1 PCM in
//! the non-standard order is reordered, and 12-channel PCM folds its four
//! height channels into the front and back pairs. Port volume, the pad
//! speaker mix level and optional per-channel gains scale the samples on the
//! way out.
//!
//! Pacing (one batch of ports is one guest scheduling point):
//! - a blocking submission first waits while the host queue holds more than
//!   the 40 ms target (at least two buffers), with no port locked; a queue
//!   that does not drain for 200 ms is cleared;
//! - while the queue fills up to the target ("priming") nothing sleeps;
//!   once primed, the batch sleeps until the deadline of each port's sample
//!   clock, which keeps the exact long-term rate;
//! - a batch with no device behind any port (no host device, a vibration
//!   port, no PCM) is paced by the sample clock alone.
//!
//! `AudioOut2` opens its ports through `internal_open` and submits whole
//! contexts through `internal_outputs`.

use super::host::{self, HostAudioStream, PcmFormat};
use super::Export;
use crate::context::CpuContext;
use crate::registry::HleResult;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::time::Duration;

pub const AUDIO_OUT_ERROR_INVALID_PORT: HleResult = 0x8026_0003u32 as i32 as i64;
pub const AUDIO_OUT_ERROR_INVALID_POINTER: HleResult = 0x8026_0004u32 as i32 as i64;
pub const AUDIO_OUT_ERROR_PORT_FULL: HleResult = 0x8026_0005u32 as i32 as i64;
pub const AUDIO_OUT_ERROR_INVALID_SIZE: HleResult = 0x8026_0006u32 as i32 as i64;
pub const AUDIO_OUT_ERROR_INVALID_FORMAT: HleResult = 0x8026_0007u32 as i32 as i64;
pub const AUDIO_OUT_ERROR_INVALID_SAMPLE_FREQ: HleResult = 0x8026_0008u32 as i32 as i64;
pub const AUDIO_OUT_ERROR_INVALID_PORT_TYPE: HleResult = 0x8026_000Au32 as i32 as i64;
pub const AUDIO_OUT_ERROR_INVALID_MIX_LEVEL: HleResult = 0x8026_0014u32 as i32 as i64;
pub const AUDIO_OUT_ERROR_EX_INVALID_MODE: HleResult = 0x8026_0015u32 as i32 as i64;
pub const AUDIO_OUT_ERROR_EX_INVALID_CONTROL: HleResult = 0x8026_8009u32 as i32 as i64;
/// The console's answer is not known (kernel ENOENT, as an unimplemented import).
const ERROR_NOT_IMPLEMENTED: HleResult = 0x8002_0002u32 as i32 as i64;

pub const PORT_TYPE_MAIN: i32 = 0;
pub const PORT_TYPE_BGM: i32 = 1;
pub const PORT_TYPE_VOICE: i32 = 2;
pub const PORT_TYPE_PERSONAL: i32 = 3;
pub const PORT_TYPE_PADSPK: i32 = 4;
/// Bitstream (passthrough) ports of sceAudioOutExOpen; nothing reaches the host.
pub const PORT_TYPE_EX: i32 = 6;
pub const PORT_TYPE_VIBRATION: i32 = 10;

/// sceAudioOutExOpen modes: at most 10, those of mask 0x61F (0-4, 9, 10).
const EX_OPEN_MODE_MAX: u32 = 10;
const EX_OPEN_MODES: u32 = 0x61F;
const EX_CONFIGURE_DEFAULT: u32 = 0xFF;
const MONITOR_TYPE_HDMI: i32 = 1;
const HDMI_MONITOR_INFO_SIZE: u32 = 0x180;
const EX_MONITOR_INFO_MIN_SIZE: u32 = 0xA0;
const EX_MONITOR_INFO_MAX_SIZE: u32 = 0x10000;
pub const PORT_TYPE_AUDIO3D: i32 = 126;
pub const PORT_TYPE_AUX: i32 = 127;

pub const OUT_PORTS_MAX: usize = 32;
/// Queue depth a blocking submission keeps on the host stream.
const TARGET_LATENCY_US: u64 = 40_000;
/// A host queue that does not drain for this long is cleared.
const STALL_US: u64 = 200_000;
const UNITY_VOLUME: i32 = 32768;
const CHANNELS_MAX: usize = 12;

/// SceAudioOutOutputParam { int32 handle; int32 pad; void *ptr; }.
const OUTPUT_PARAM_SIZE: usize = 16;
/// SceAudioOutPortState { u16 output; u8 channel; u8 reserved; s16 volume;
/// u16 rerouteCounter; u64 flag; u64 reserved[2]; }.
const PORT_STATE_SIZE: usize = 0x20;

/// PCM layout of an output port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Unknown,
    S16Mono,
    S16Stereo,
    S16_8Ch,
    FloatMono,
    FloatStereo,
    Float8Ch,
    S16_8ChStd,
    Float8ChStd,
    Float12Ch,
}

impl Format {
    /// SceAudioOutParamFormat (low byte of the open parameter).
    pub fn from_param(param: u32) -> Format {
        match param & 0xFF {
            0 => Format::S16Mono,
            1 => Format::S16Stereo,
            2 => Format::S16_8Ch,
            3 => Format::FloatMono,
            4 => Format::FloatStereo,
            5 => Format::Float8Ch,
            6 => Format::S16_8ChStd,
            7 => Format::Float8ChStd,
            _ => Format::Unknown,
        }
    }

    pub fn channels(self) -> usize {
        match self {
            Format::Unknown => 0,
            Format::S16Mono | Format::FloatMono => 1,
            Format::S16Stereo | Format::FloatStereo => 2,
            Format::S16_8Ch | Format::Float8Ch | Format::S16_8ChStd | Format::Float8ChStd => 8,
            Format::Float12Ch => 12,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(
            self,
            Format::FloatMono | Format::FloatStereo | Format::Float8Ch | Format::Float8ChStd | Format::Float12Ch
        )
    }

    fn is_std(self) -> bool {
        matches!(self, Format::S16_8ChStd | Format::Float8ChStd)
    }

    pub fn bytes_per_sample(self) -> usize {
        if self.is_float() {
            4
        } else {
            2
        }
    }
}

fn port_type_is_valid(t: i32) -> bool {
    (PORT_TYPE_MAIN..=PORT_TYPE_PADSPK).contains(&t)
        || t == PORT_TYPE_VIBRATION
        || t == PORT_TYPE_AUDIO3D
        || t == PORT_TYPE_AUX
}

/// Sample clock: deadlines of consecutive buffers, without drift.
#[derive(Clone, Copy, Default, Debug)]
struct OutputClock {
    next_time: u64,
    remainder: u64,
}

impl OutputClock {
    /// Deadline of this buffer; the clock moves on by `frames`.
    fn advance(&mut self, now: u64, frames: u32, rate: u32) -> u64 {
        // Keep small scheduling errors; drop timing debt after a discontinuity.
        if self.next_time == 0 || (now > self.next_time && now - self.next_time > 100_000) {
            self.next_time = now;
            self.remainder = 0;
        }
        let deadline = self.next_time;
        let duration = 1_000_000 * frames as u64 + self.remainder;
        self.next_time += duration / rate as u64;
        self.remainder = duration % rate as u64;
        deadline
    }
}

/// Time, sleeping and host streams.
pub trait Env: Send + Sync {
    /// Process time in microseconds.
    fn now_us(&self) -> u64;
    /// Sleep with no lock held (guest signals may be delivered).
    fn sleep_us(&self, us: u64);
    fn open_stream(&self, rate: u32, channels: usize, format: PcmFormat) -> Option<Arc<dyn HostAudioStream>>;
}

#[derive(Default)]
struct PortOut {
    used: bool,
    generation: u64,
    port_type: i32,
    samples_num: u32,
    freq: u32,
    format: Option<Format>,
    clock: OutputClock,
    queue_primed: bool,
    channels_num: usize,
    volume: [i32; CHANNELS_MAX],
    /// Pad speaker mix level (sceAudioOutSetMixLevelPadSpk), 32768 = unity.
    mix_level: i32,
    stream: Option<Arc<dyn HostAudioStream>>,
    /// Pad speaker / vibration ports: the controller's stream (pad_audio.rs).
    pad_stream: Option<Box<dyn crate::pad_audio::PadAudioStream>>,
    /// Reused for PCM that needs scaling or reordering.
    scratch: Vec<u8>,
    /// Process time (us) the last buffer with PCM was submitted; 0 before.
    last_output_time: u64,
}

/// Outcome of waiting for a host queue to drain, before the port is locked.
#[derive(Clone, Copy, Default)]
struct Drained {
    generation: u64,
    /// The queue holds less than one buffer (or was cleared after a stall):
    /// priming starts over.
    unprimed: bool,
}

impl PortOut {
    fn format(&self) -> Format {
        self.format.unwrap_or(Format::Unknown)
    }

    /// The pad speaker mix level as a linear gain; set to unity for every port.
    fn mix_gain(&self) -> f32 {
        self.mix_level as f32 / UNITY_VOLUME as f32
    }

    /// Host streams take at most 8 channels; the guest buffer keeps its own count.
    fn output_channels(&self) -> usize {
        self.channels_num.min(8)
    }

    fn buffer_bytes(&self) -> usize {
        self.samples_num as usize * self.channels_num * self.format().bytes_per_sample()
    }

    /// Bytes of one buffer in the host layout.
    fn prepared_size(&self) -> usize {
        self.format().bytes_per_sample() * self.output_channels() * self.samples_num as usize
    }

    /// Queue depth a blocking submission keeps: the target latency, at least two buffers.
    fn min_queued_size(&self) -> usize {
        let buffer_us = if self.freq != 0 { 1_000_000 * self.samples_num as u64 / self.freq as u64 } else { 0 };
        let buffers = if buffer_us != 0 { TARGET_LATENCY_US.div_ceil(buffer_us).clamp(2, 16) as usize } else { 2 };
        self.prepared_size() * buffers
    }

    /// Scale by volume and gains and fit the host layout into `out`; false
    /// when `data` can be queued as it is.
    fn prepare_output(&self, data: &[u8], gain: f32, gains: Option<&[f32]>, out: &mut Vec<u8>) -> bool {
        let format = self.format();
        let frames = self.samples_num as usize;
        let channels = self.channels_num;
        let output_channels = self.output_channels();
        let bps = format.bytes_per_sample();
        let reorder = channels >= 8 && !format.is_std();

        let mut scales = [0.0f64; CHANNELS_MAX];
        let mut volume_changed = false;
        for ch in 0..channels {
            let g = gains.and_then(|g| g.get(ch).copied()).unwrap_or(1.0);
            scales[ch] = (self.volume[ch] as f64 / 32768.0) * gain as f64 * g as f64;
            volume_changed |= scales[ch] != 1.0;
        }
        if !volume_changed && !reorder {
            return false;
        }

        // Host order puts the back pair before the side pair; non-STD PCM has them reversed.
        const HOST_8CH_MAP: [usize; 8] = [0, 1, 2, 3, 6, 7, 4, 5];
        const IDENTITY: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        // The four height channels fold into the front and back pairs.
        const HEIGHT_DST: [usize; 4] = [0, 1, 4, 5];
        let map = if reorder { HOST_8CH_MAP } else { IDENTITY };
        let mut out_scales = [0.0f64; 8];
        for (ch, s) in out_scales.iter_mut().enumerate().take(output_channels) {
            *s = scales[map[ch]];
        }
        out.clear();
        out.resize(frames * output_channels * bps, 0);
        let frames_in = data.chunks_exact(channels * bps);
        let frames_out = out.chunks_exact_mut(output_channels * bps);
        if format.is_float() {
            for (src, dst) in frames_in.zip(frames_out) {
                let s = |c: usize| f32::from_le_bytes([src[c * 4], src[c * 4 + 1], src[c * 4 + 2], src[c * 4 + 3]]) as f64;
                let mut row = [0.0f32; 8];
                for ch in 0..output_channels {
                    row[ch] = (s(map[ch]) * out_scales[ch]) as f32;
                }
                if channels == 12 {
                    for (h, &d) in HEIGHT_DST.iter().enumerate() {
                        row[d] += (s(8 + h) * scales[8 + h]) as f32;
                    }
                }
                for (d, v) in dst.chunks_exact_mut(4).zip(row) {
                    d.copy_from_slice(&v.to_le_bytes());
                }
            }
        } else {
            for (src, dst) in frames_in.zip(frames_out) {
                for (ch, d) in dst.chunks_exact_mut(2).enumerate() {
                    let c = map[ch] * 2;
                    let sample = i16::from_le_bytes([src[c], src[c + 1]]) as f64 * out_scales[ch];
                    d.copy_from_slice(&(sample.clamp(-32768.0, 32767.0) as i16).to_le_bytes());
                }
            }
        }
        true
    }

    /// Queue one buffer on the host stream; false when there is no device to
    /// take it. A blocking submission has already waited for the queue to
    /// drain (`drained`).
    fn queue_audio(&mut self, data: Option<&[u8]>, blocking: bool, gain: f32, gains: Option<&[f32]>, drained: Drained) -> bool {
        let (Some(stream), Some(data)) = (self.stream.clone(), data) else { return false };
        if data.len() < self.buffer_bytes() {
            return false;
        }
        let prepared_size = self.prepared_size();
        let min_queued_size = if blocking { self.min_queued_size() } else { 0 };
        if blocking && drained.generation == self.generation && drained.unprimed {
            self.queue_primed = false;
        }

        let mut scratch = std::mem::take(&mut self.scratch);
        let put = {
            let prepared = if self.prepare_output(data, gain, gains, &mut scratch) { &scratch[..] } else { data };
            stream.put(&prepared[..prepared_size.min(prepared.len())])
        };
        self.scratch = scratch;
        if !put {
            self.queue_primed = false;
            return false;
        }
        // Unpaced priming must not move the playback deadline into the future.
        if !self.queue_primed {
            self.clock = OutputClock::default();
        }
        if blocking && !self.queue_primed && stream.queued_bytes() >= min_queued_size {
            self.queue_primed = true;
        }
        true
    }
}

// --- controller audio hook (pad_audio.rs) -----------------------------------

/// Offer a block to the port's controller stream: (us still queued there,
/// Bluetooth link); 0 when no controller took it.
fn queue_controller(port: &mut PortOut, p: &OutputParam, gain: f32) -> (u64, bool) {
    let (frames, channels, is_float, bytes) =
        (port.samples_num, port.channels_num, port.format().is_float(), port.buffer_bytes());
    let (Some(stream), Some(data)) = (port.pad_stream.as_mut(), p.data) else { return (0, false) };
    if data.len() < bytes {
        return (0, false);
    }
    let block = crate::pad_audio::PadAudioBlock {
        data,
        frames,
        channels: channels as u32,
        is_float,
        volume: &port.volume[..channels],
        gain,
        gains: p.gains,
    };
    (stream.queue(&block), stream.uses_bluetooth())
}

/// Rumble motor levels of a vibration port block (left, right channel)
/// with the port volume and gains applied; None without PCM.
fn vibration_levels(port: &PortOut, p: &OutputParam, gain: f32) -> Option<(u8, u8)> {
    let data = p.data?;
    let (frames, channels) = (port.samples_num as usize, port.channels_num);
    if !(1..=2).contains(&channels) || data.len() < port.buffer_bytes() {
        return None;
    }
    let block = crate::pad_audio::PadAudioBlock {
        data,
        frames: frames as u32,
        channels: channels as u32,
        is_float: port.format().is_float(),
        volume: &port.volume[..channels],
        gain,
        gains: p.gains,
    };
    let level = |c: usize| {
        let g = gain * port.volume[c] as f32 / UNITY_VOLUME as f32 * p.gains.and_then(|g| g.get(c).copied()).unwrap_or(1.0);
        (0..frames).map(|f| block.sample(f * channels + c) * g).collect::<Vec<f32>>()
    };
    let left = level(0);
    let right = if channels == 2 { level(1) } else { left.clone() };
    Some(crate::pad_controller::audio_haptics_levels(&left, &right))
}

/// One port's buffer in a batch.
#[derive(Clone, Copy)]
pub struct OutputParam<'a> {
    pub handle: i32,
    pub data: Option<&'a [u8]>,
    pub gains: Option<&'a [f32]>,
}

impl OutputParam<'_> {
    pub const NONE: OutputParam<'static> = OutputParam { handle: 0, data: None, gains: None };
}

fn slot_index(handle: i32) -> Option<usize> {
    let i = usize::try_from(handle.checked_sub(1)?).ok()?;
    (i < OUT_PORTS_MAX).then_some(i)
}

pub struct Audio<E: Env> {
    env: E,
    ports: Vec<Mutex<PortOut>>,
    generation: AtomicU64,
}

impl<E: Env> Audio<E> {
    pub fn new(env: E) -> Self {
        Self { env, ports: (0..OUT_PORTS_MAX).map(|_| Mutex::default()).collect(), generation: AtomicU64::new(1) }
    }

    fn lock_slot(&self, i: usize) -> MutexGuard<'_, PortOut> {
        self.ports[i].lock().unwrap_or_else(|e| e.into_inner())
    }

    fn slot(&self, handle: i32) -> Option<MutexGuard<'_, PortOut>> {
        let g = self.lock_slot(slot_index(handle)?);
        g.used.then_some(g)
    }

    /// Open a port; returns its handle (1-based).
    pub fn open(&self, port_type: i32, samples_num: u32, freq: u32, format: Format) -> Option<i32> {
        if freq == 0 || format == Format::Unknown {
            return None;
        }
        for (i, slot) in self.ports.iter().enumerate() {
            let mut port = slot.lock().unwrap_or_else(|e| e.into_inner());
            if port.used {
                continue;
            }
            let generation = self.generation.fetch_add(1, Ordering::Relaxed);
            *port = PortOut {
                used: true,
                generation,
                port_type,
                samples_num,
                freq,
                format: Some(format),
                channels_num: format.channels(),
                volume: [UNITY_VOLUME; CHANNELS_MAX],
                mix_level: UNITY_VOLUME,
                ..PortOut::default()
            };
            let output_channels = port.output_channels();
            drop(port);
            // Vibration and bitstream ports have no speaker output.
            if port_type != PORT_TYPE_VIBRATION && port_type != PORT_TYPE_EX {
                let pcm = if format.is_float() { PcmFormat::F32 } else { PcmFormat::S16 };
                let stream = self.env.open_stream(freq, output_channels, pcm);
                let mut port = slot.lock().unwrap_or_else(|e| e.into_inner());
                if port.used && port.generation == generation {
                    port.stream = stream;
                }
            }
            // Controller audio hook (pad_audio.rs): pad speaker / vibration
            // ports also get a controller stream; pad speaker ports keep the
            // main output, which plays them whenever no controller does.
            if let Some(pad_stream) = crate::pad_audio::open_port(port_type, freq) {
                let mut port = slot.lock().unwrap_or_else(|e| e.into_inner());
                if port.used && port.generation == generation {
                    port.pad_stream = Some(pad_stream);
                }
            }
            return Some(i as i32 + 1);
        }
        None
    }

    pub fn close(&self, handle: i32) -> bool {
        match self.slot(handle) {
            Some(mut port) => {
                *port = PortOut::default();
                true
            }
            None => false,
        }
    }

    pub fn valid(&self, handle: i32) -> bool {
        self.slot(handle).is_some()
    }

    pub fn port_type(&self, handle: i32) -> Option<i32> {
        self.slot(handle).map(|p| p.port_type)
    }

    pub fn last_output_time(&self, handle: i32) -> Option<u64> {
        self.slot(handle).map(|p| p.last_output_time)
    }

    pub fn has_device(&self, handle: i32) -> bool {
        self.slot(handle).is_some_and(|p| p.stream.is_some())
    }

    /// (port type, channels, buffer bytes, buffer frames).
    fn status(&self, handle: i32) -> Option<(i32, usize, usize, u32)> {
        self.slot(handle).map(|p| (p.port_type, p.channels_num, p.buffer_bytes(), p.samples_num))
    }

    pub fn set_volume(&self, handle: i32, mut bitflag: u32, volume: &[i32]) -> bool {
        let Some(mut port) = self.slot(handle) else { return false };
        for i in 0..port.channels_num {
            if bitflag & 1 == 1 {
                if let Some(&v) = volume.get(i) {
                    port.volume[i] = v;
                }
            }
            bitflag >>= 1;
        }
        true
    }

    /// Set the mix level of a pad speaker port. Checked in this order: the
    /// handle, the port type, then the range 0 to unity (a negative level is
    /// outside it, and the console's answer for it is unverified).
    pub fn set_mix_level_pad_spk(&self, handle: i32, level: i32) -> HleResult {
        let Some(mut port) = self.slot(handle) else { return AUDIO_OUT_ERROR_INVALID_PORT };
        if port.port_type != PORT_TYPE_PADSPK {
            return AUDIO_OUT_ERROR_INVALID_PORT_TYPE;
        }
        if !(0..=UNITY_VOLUME).contains(&level) {
            return AUDIO_OUT_ERROR_INVALID_MIX_LEVEL;
        }
        port.mix_level = level;
        0
    }

    /// Blocking submission, with no port locked: wait while a device queue
    /// holds more than its target; a queue that does not drain for the stall
    /// time is cleared.
    fn drain(&self, p: &OutputParam) -> Drained {
        let Some(i) = slot_index(p.handle) else { return Drained::default() };
        let (stream, generation, prepared_size, min_queued_size) = {
            let port = self.lock_slot(i);
            match (&port.stream, p.data) {
                (Some(s), Some(d)) if port.used && d.len() >= port.buffer_bytes() => {
                    (s.clone(), port.generation, port.prepared_size(), port.min_queued_size())
                }
                _ => return Drained::default(),
            }
        };
        let wait_start = self.env.now_us();
        let mut queued = stream.queued_bytes();
        let mut unprimed = false;
        while queued > min_queued_size {
            if self.env.now_us().saturating_sub(wait_start) > STALL_US {
                stream.clear();
                unprimed = true;
                break;
            }
            self.env.sleep_us(1000);
            queued = stream.queued_bytes();
        }
        Drained { generation, unprimed: unprimed || queued < prepared_size }
    }

    /// Submit one buffer per port and pace the batch; returns the first
    /// port's buffer length. Invalid handles are skipped.
    pub fn outputs(&self, params: &[OutputParam], blocking: bool) -> u32 {
        let params = &params[..params.len().min(OUT_PORTS_MAX)];
        let mut drained = [Drained::default(); OUT_PORTS_MAX];
        if blocking {
            for (k, p) in params.iter().enumerate() {
                drained[k] = self.drain(p);
            }
        }

        // Lock every port of the batch, in slot order.
        let mut mask = 0u64;
        for p in params {
            if let Some(i) = slot_index(p.handle) {
                mask |= 1 << i;
            }
        }
        let mut guards: [Option<MutexGuard<'_, PortOut>>; OUT_PORTS_MAX] = std::array::from_fn(|_| None);
        for (i, g) in guards.iter_mut().enumerate() {
            if mask & (1 << i) != 0 {
                let port = self.lock_slot(i);
                if port.used {
                    *g = Some(port);
                }
            }
        }
        let mut first_samples = None;
        let mut paced = [false; OUT_PORTS_MAX];
        let mut any_device = false;
        let mut deadline = 0;
        for (k, p) in params.iter().enumerate() {
            let Some(port) = slot_index(p.handle).and_then(|i| guards[i].as_deref_mut()) else { continue };
            first_samples.get_or_insert(port.samples_num);
            // Controller audio hook (pad_audio.rs). The pad speaker mix level
            // is a further gain on the port, on both the controller and the
            // main output.
            let gain = crate::pad_audio::port_gain(port.port_type) * port.mix_gain();
            let (controller_us, bluetooth) = queue_controller(port, p, gain);
            if controller_us == 0 && port.port_type == PORT_TYPE_VIBRATION {
                // No haptic actuator played it: the rumble motors follow it.
                if let Some((large, small)) = vibration_levels(port, p, gain) {
                    crate::pad::submit_audio_haptics(large, small);
                }
            }
            if controller_us == 0 {
                // No controller took it (e.g. it was unplugged): a pad speaker
                // port plays on the main output, a vibration port nowhere.
                if port.queue_audio(p.data, blocking, gain, p.gains, drained[k]) {
                    any_device = true;
                    paced[k] = port.queue_primed;
                }
            } else {
                port.queue_primed = false;
                // Bluetooth has its own sender; the sample clock paces it.
                if port.port_type == PORT_TYPE_PADSPK && !bluetooth {
                    any_device = true;
                    if blocking && controller_us > TARGET_LATENCY_US && port.freq != 0 {
                        // A stalled USB endpoint waits at most one block.
                        let block_us = 1_000_000 * port.samples_num as u64 / port.freq as u64;
                        let wait = (controller_us - TARGET_LATENCY_US).min(block_us);
                        deadline = deadline.max(self.env.now_us() + wait);
                    }
                }
            }
        }

        // The whole batch is queued before it is paced. Device-backed ports
        // own the cadence while priming; a batch without a device uses the
        // sample clock.
        let now = self.env.now_us();
        for (k, p) in params.iter().enumerate() {
            let Some(port) = slot_index(p.handle).and_then(|i| guards[i].as_deref_mut()) else { continue };
            if blocking && (paced[k] || !any_device) {
                deadline = deadline.max(port.clock.advance(now, port.samples_num, port.freq));
            } else {
                port.clock = OutputClock::default();
            }
            if !blocking {
                port.queue_primed = false;
            }
        }
        drop(guards);
        if deadline > now {
            self.env.sleep_us(deadline - now);
        }
        let done = self.env.now_us();
        for p in params.iter().filter(|p| p.data.is_some()) {
            if let Some(mut port) = self.slot(p.handle) {
                port.last_output_time = done;
            }
        }
        first_samples.unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// The emulator's ports
// ---------------------------------------------------------------------------

struct HostEnv;

impl Env for HostEnv {
    fn now_us(&self) -> u64 {
        crate::time::process_time_usec()
    }

    fn sleep_us(&self, us: u64) {
        crate::signals::sleep(Duration::from_micros(us));
    }

    fn open_stream(&self, rate: u32, channels: usize, format: PcmFormat) -> Option<Arc<dyn HostAudioStream>> {
        let s = host::open_stream(rate, channels, format);
        log::info!(
            "[AUDIO] AudioOut stream: {rate} Hz, {channels} ch, {format:?}, backend={}",
            s.as_ref().map_or("none", |s| s.backend_name())
        );
        s
    }
}

static AUDIO: LazyLock<Audio<HostEnv>> = LazyLock::new(|| Audio::new(HostEnv));

/// Open a port for AudioOut2; 0 when none is available.
pub fn internal_open(port_type: i32, samples_num: u32, freq: u32, format: Format) -> i32 {
    AUDIO.open(port_type, samples_num, freq, format).unwrap_or(0)
}

pub fn internal_close(handle: i32) {
    if handle > 0 {
        AUDIO.close(handle);
    }
}

pub fn internal_has_device(handle: i32) -> bool {
    handle > 0 && AUDIO.has_device(handle)
}

/// Submit a batch for AudioOut2; entries without a port or PCM are dropped.
pub fn internal_outputs(params: &[OutputParam], blocking: bool) -> u32 {
    let mut batch = [OutputParam::NONE; OUT_PORTS_MAX];
    let mut n = 0;
    for p in params.iter().filter(|p| p.handle > 0 && p.data.is_some()).take(OUT_PORTS_MAX) {
        batch[n] = *p;
        n += 1;
    }
    if n == 0 {
        return 0;
    }
    AUDIO.outputs(&batch[..n], blocking)
}

static OUTPUT_COUNT: AtomicU64 = AtomicU64::new(0);

fn trace_output(handle: i32, bytes: usize) {
    if OUTPUT_COUNT.fetch_add(1, Ordering::Relaxed) == 0 {
        log::info!("[AUDIO] sceAudioOutOutput first grain: handle={handle} bytes={bytes}");
    }
}

fn cb_init(_ctx: &mut CpuContext) -> HleResult {
    crate::audio::AUDIO_OUT_INITIALIZED.store(true, std::sync::atomic::Ordering::Release);
    0
}

/// sceAudioOutOpen(userId, type, index, bufferLength, frequency, format)
fn cb_open(ctx: &mut CpuContext) -> HleResult {
    let port_type = ctx.rsi() as i32;
    let buffer_length = ctx.rcx() as u32;
    let frequency = ctx.r8() as u32;
    let param = ctx.r9() as u32;
    if !port_type_is_valid(port_type) {
        return AUDIO_OUT_ERROR_INVALID_PORT_TYPE;
    }
    if frequency == 0 {
        return AUDIO_OUT_ERROR_INVALID_SAMPLE_FREQ;
    }
    let format = Format::from_param(param);
    if format == Format::Unknown {
        return AUDIO_OUT_ERROR_INVALID_FORMAT;
    }
    let Some(handle) = AUDIO.open(port_type, buffer_length, frequency, format) else {
        return AUDIO_OUT_ERROR_PORT_FULL;
    };
    log::info!("[AUDIO] AudioOut port {handle}: type {port_type}, {frequency} Hz, {format:?}, {buffer_length} frames");
    handle as HleResult
}

fn cb_close(ctx: &mut CpuContext) -> HleResult {
    if AUDIO.close(ctx.rdi() as i32) {
        0
    } else {
        AUDIO_OUT_ERROR_INVALID_PORT
    }
}

/// (output, channel) reported for a port type.
fn port_state_output(port_type: i32, channels: usize) -> (u16, u8) {
    match port_type {
        PORT_TYPE_MAIN | PORT_TYPE_BGM | PORT_TYPE_AUDIO3D => (1, channels.min(2) as u8),
        PORT_TYPE_VOICE | PORT_TYPE_PERSONAL => (0x40, 1),
        PORT_TYPE_PADSPK | PORT_TYPE_VIBRATION => (4, 1),
        _ => (0x80, 0),
    }
}

fn cb_get_port_state(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rsi();
    let Some((port_type, channels, _, _)) = AUDIO.status(ctx.rdi() as i32) else {
        return AUDIO_OUT_ERROR_INVALID_PORT;
    };
    let (output, channel) = port_state_output(port_type, channels);
    let mut st = [0u8; PORT_STATE_SIZE];
    st[0..2].copy_from_slice(&output.to_le_bytes());
    st[2] = channel;
    st[4..6].copy_from_slice(&127i16.to_le_bytes());
    if addr == 0 || !ctx.write_bytes(addr, &st) {
        return AUDIO_OUT_ERROR_INVALID_POINTER;
    }
    0
}

/// `SceAudioOutSystemState`: loudness (f32), 4 reserved bytes, 3 reserved u64.
const SYSTEM_STATE_SIZE: usize = 32;

/// sceAudioOutGetSystemState(state): loudness is undefined until a
/// measurement starts and none is emulated, so the whole state is zero.
fn cb_get_system_state(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 || !ctx.write_bytes(addr, &[0u8; SYSTEM_STATE_SIZE]) {
        return AUDIO_OUT_ERROR_INVALID_POINTER;
    }
    0
}

thread_local! {
    /// Per-thread copies of the guest's PCM, reused from call to call.
    static GUEST_PCM: RefCell<Vec<Vec<u8>>> = const { RefCell::new(Vec::new()) };
}

/// Read each port's buffer out of guest memory and submit the batch.
fn submit_outputs(ctx: &CpuContext, descriptors: &[(i32, u64)]) -> HleResult {
    GUEST_PCM.with(|cell| match cell.try_borrow_mut() {
        Ok(mut buffers) => submit_outputs_with(ctx, descriptors, &mut buffers),
        // A guest signal handler submitting while this thread is being paced.
        Err(_) => submit_outputs_with(ctx, descriptors, &mut Vec::new()),
    })
}

fn submit_outputs_with(ctx: &CpuContext, descriptors: &[(i32, u64)], buffers: &mut Vec<Vec<u8>>) -> HleResult {
    let descriptors = &descriptors[..descriptors.len().min(OUT_PORTS_MAX)];
    if buffers.len() < descriptors.len() {
        buffers.resize_with(descriptors.len(), Vec::new);
    }
    let mut has_pcm = [false; OUT_PORTS_MAX];
    for (k, &(handle, src)) in descriptors.iter().enumerate() {
        let Some((_, _, bytes, _)) = AUDIO.status(handle) else {
            return AUDIO_OUT_ERROR_INVALID_PORT;
        };
        if src == 0 {
            continue;
        }
        let buffer = &mut buffers[k];
        buffer.resize(bytes, 0);
        if bytes != 0 && ctx.vm.read(src, buffer).is_err() {
            return AUDIO_OUT_ERROR_INVALID_POINTER;
        }
        trace_output(handle, bytes);
        has_pcm[k] = true;
    }
    let mut params = [OutputParam::NONE; OUT_PORTS_MAX];
    for (k, &(handle, _)) in descriptors.iter().enumerate() {
        params[k] = OutputParam { handle, data: has_pcm[k].then_some(&buffers[k][..]), gains: None };
    }
    AUDIO.outputs(&params[..descriptors.len()], true) as HleResult
}

/// sceAudioOutOutput(handle, ptr): one buffer, blocking.
fn cb_output(ctx: &mut CpuContext) -> HleResult {
    submit_outputs(ctx, &[(ctx.rdi() as i32, ctx.rsi())])
}

/// sceAudioOutOutputs(SceAudioOutOutputParam *param, uint32 num)
fn cb_outputs(ctx: &mut CpuContext) -> HleResult {
    let param = ctx.rdi();
    let count = ctx.rsi() as u32 as usize;
    if count == 0 || count > OUT_PORTS_MAX {
        return AUDIO_OUT_ERROR_INVALID_SIZE;
    }
    if param == 0 {
        return AUDIO_OUT_ERROR_INVALID_POINTER;
    }
    let mut raw = [0u8; OUT_PORTS_MAX * OUTPUT_PARAM_SIZE];
    let raw = &mut raw[..count * OUTPUT_PARAM_SIZE];
    if ctx.vm.read(param, raw).is_err() {
        return AUDIO_OUT_ERROR_INVALID_POINTER;
    }
    let mut descriptors = [(0i32, 0u64); OUT_PORTS_MAX];
    for (d, e) in descriptors.iter_mut().zip(raw.chunks_exact(OUTPUT_PARAM_SIZE)) {
        *d = (i32::from_le_bytes(e[0..4].try_into().unwrap()), u64::from_le_bytes(e[8..16].try_into().unwrap()));
    }
    submit_outputs(ctx, &descriptors[..count])
}

/// sceAudioOutSetVolume(handle, channelFlags, int32 vol[])
fn cb_set_volume(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi() as i32;
    let flags = ctx.rsi() as u32;
    let vol_addr = ctx.rdx();
    let Some((_, channels, _, _)) = AUDIO.status(handle) else {
        return AUDIO_OUT_ERROR_INVALID_PORT;
    };
    if vol_addr == 0 {
        return AUDIO_OUT_ERROR_INVALID_POINTER;
    }
    let mut volume = [UNITY_VOLUME; CHANNELS_MAX];
    for (ch, v) in volume.iter_mut().enumerate().take(channels) {
        if flags & (1 << ch) != 0 {
            let Some(raw) = ctx.read_u32(vol_addr + ch as u64 * 4) else {
                return AUDIO_OUT_ERROR_INVALID_POINTER;
            };
            *v = raw as i32;
        }
    }
    if AUDIO.set_volume(handle, flags, &volume) {
        0
    } else {
        AUDIO_OUT_ERROR_INVALID_PORT
    }
}

/// sceAudioOutSetMixLevelPadSpk(handle, mixLevel)
fn cb_set_mix_level_pad_spk(ctx: &mut CpuContext) -> HleResult {
    AUDIO.set_mix_level_pad_spk(ctx.rdi() as i32, ctx.rsi() as i32)
}

/// sceAudioOutGetLastOutputTime(handle, uint64_t *outputTime): when the
/// last buffer with PCM was submitted (process time, us).
fn cb_get_last_output_time(ctx: &mut CpuContext) -> HleResult {
    let (handle, out) = (ctx.rdi() as i32, ctx.rsi());
    if out == 0 {
        return AUDIO_OUT_ERROR_INVALID_POINTER;
    }
    let Some(time) = AUDIO.last_output_time(handle) else { return AUDIO_OUT_ERROR_INVALID_PORT };
    if !ctx.write_u64(out, time) {
        return AUDIO_OUT_ERROR_INVALID_POINTER;
    }
    0
}

/// sceAudioOutExOpen(userId, mode): a bitstream port of 256 stereo s16
/// frames at 48 kHz (1024 at 192 kHz for modes 3, 4 and 10), paced by its
/// sample clock.
fn cb_ex_open(ctx: &mut CpuContext) -> HleResult {
    let mode = ctx.rsi() as u32;
    if mode > EX_OPEN_MODE_MAX || (EX_OPEN_MODES >> mode) & 1 == 0 {
        return AUDIO_OUT_ERROR_EX_INVALID_MODE;
    }
    let high_rate = matches!(mode, 3 | 4 | 10);
    let (frames, rate) = if high_rate { (1024, 192_000) } else { (256, 48_000) };
    let Some(handle) = AUDIO.open(PORT_TYPE_EX, frames, rate, Format::S16Stereo) else {
        return AUDIO_OUT_ERROR_PORT_FULL;
    };
    log::info!("[AUDIO] AudioOutEx port {handle}: mode {mode}, {rate} Hz, {frames} frames (not played)");
    handle as HleResult
}

/// sceAudioOutExClose(handle)
fn cb_ex_close(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi() as i32;
    if AUDIO.port_type(handle) != Some(PORT_TYPE_EX) {
        log::error!("[AUDIO] sceAudioOutExClose: handle {handle} is not an open Ex port");
        return AUDIO_OUT_ERROR_INVALID_PORT;
    }
    AUDIO.close(handle);
    0
}

/// sceAudioOutExConfigureOutput(control, flags, mode, target, option):
/// accepted for the known modes and the HDMI/optical targets (1..3, 0xFF
/// default); nothing changes on the host.
fn cb_ex_configure_output(ctx: &mut CpuContext) -> HleResult {
    let (control, flags, mode, target, option) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as u32, ctx.rcx() as u32, ctx.r8());
    if control != 0 {
        return AUDIO_OUT_ERROR_EX_INVALID_CONTROL;
    }
    let known_mode = mode <= EX_OPEN_MODE_MAX || mode == EX_CONFIGURE_DEFAULT;
    let known_target = (1..=3).contains(&target) || target == EX_CONFIGURE_DEFAULT;
    if flags != 0 || option != 0 || !known_mode || !known_target {
        log::error!("[AUDIO] sceAudioOutExConfigureOutput flags {flags:#x} mode {mode} target {target} option {option:#x} is not implemented");
        return ERROR_NOT_IMPLEMENTED;
    }
    0
}

/// sceAudioOutExGetMonitorInfo(type, index, void *info, size): the monitor
/// lists no audio descriptors (all zero).
fn cb_ex_get_monitor_info(ctx: &mut CpuContext) -> HleResult {
    let (monitor, index, info, size) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx(), ctx.rcx() as u32);
    if monitor != 0 || index != 0 || !(EX_MONITOR_INFO_MIN_SIZE..=EX_MONITOR_INFO_MAX_SIZE).contains(&size) {
        log::error!("[AUDIO] sceAudioOutExGetMonitorInfo monitor {monitor}, {index}, size {size:#x} is not implemented");
        return ERROR_NOT_IMPLEMENTED;
    }
    if info == 0 || !ctx.write_bytes(info, &vec![0u8; size as usize]) {
        return AUDIO_OUT_ERROR_INVALID_POINTER;
    }
    0
}

/// sceAudioOutSysGetHdmiMonitorInfo(type, void *info, size): an HDMI
/// monitor without audio capabilities (all zero).
fn cb_sys_get_hdmi_monitor_info(ctx: &mut CpuContext) -> HleResult {
    let (monitor, info, size) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as u32);
    if monitor != MONITOR_TYPE_HDMI || size != HDMI_MONITOR_INFO_SIZE {
        log::error!("[AUDIO] sceAudioOutSysGetHdmiMonitorInfo type {monitor}, size {size:#x} is not implemented");
        return ERROR_NOT_IMPLEMENTED;
    }
    if info == 0 || !ctx.write_bytes(info, &[0u8; HDMI_MONITOR_INFO_SIZE as usize]) {
        return AUDIO_OUT_ERROR_INVALID_POINTER;
    }
    0
}

/// sceAudioOutSetMainOutput(): returns 0 and reads no argument, as the
/// console's export (`xor eax, eax; ret`, firmware 12.70) does.
fn cb_set_main_output(_ctx: &mut CpuContext) -> HleResult {
    0
}

pub(crate) static EXPORTS: &[Export] = &[
    ("JfEPXVxhFqA", "sceAudioOutInit", "libSceAudioOut", cb_init),
    ("KI9cl22to7E", "sceAudioOutSetMainOutput", "libSceAudioOut", cb_set_main_output),
    ("Ptlts326pds", "sceAudioOutGetLastOutputTime", "libSceAudioOut", cb_get_last_output_time),
    ("6X6dp+07h4U", "sceAudioOutExOpen", "libSceAudioOut", cb_ex_open),
    ("0TfjSulCV2A", "sceAudioOutExClose", "libSceAudioOut", cb_ex_close),
    ("VcE+gXSwFXI", "sceAudioOutExConfigureOutput", "libSceAudioOut", cb_ex_configure_output),
    ("PhnNN0UEV6Y", "sceAudioOutExGetMonitorInfo", "libSceAudioOut", cb_ex_get_monitor_info),
    ("Tf9-yOJwF-A", "sceAudioOutSysGetHdmiMonitorInfo", "libSceAudioOut", cb_sys_get_hdmi_monitor_info),
    ("ekNvsT22rsY", "sceAudioOutOpen", "libSceAudioOut", cb_open),
    ("s1--uE9mBFw", "sceAudioOutClose", "libSceAudioOut", cb_close),
    ("GrQ9s4IrNaQ", "sceAudioOutGetPortState", "libSceAudioOut", cb_get_port_state),
    ("R5hemoKKID8", "sceAudioOutGetSystemState", "libSceAudioOut", cb_get_system_state),
    ("w3PdaSTSwGE", "sceAudioOutOutputs", "libSceAudioOut", cb_outputs),
    ("QOQtbeDqsT4", "sceAudioOutOutput", "libSceAudioOut", cb_output),
    ("b+uAV89IlxE", "sceAudioOutSetVolume", "libSceAudioOut", cb_set_volume),
    ("wVwPU50pS1c", "sceAudioOutSetMixLevelPadSpk", "libSceAudioOut", cb_set_mix_level_pad_spk),
];

#[cfg(test)]
mod tests;
