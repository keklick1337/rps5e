// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Host audio output.
//!
//! Streams take interleaved PCM in the guest's channel layout and sample
//! rate. The preferred backend is SDL3 (`sdl_host.rs`): it plays through
//! the desktop sound server (PipeWire / PulseAudio) and converts channels
//! and rate itself. Without SDL3, cpal is used: on Linux that is raw ALSA,
//! whose `default` PCM is only the sound server when the ALSA plugin
//! configuration is installed — otherwise it is card 0, which may be a
//! loopback or HDMI device. `RPS5E_AUDIO=sdl|cpal|null` forces a backend.
//!
//! A cpal stream is a ring buffer of f32 frames that the device callback
//! drains, converting channels and resampling to what the device accepted.
//!
//! A stream never blocks: `put` enqueues, `queued_bytes` reports the
//! backlog and `clear` drops it. Pacing the guest is the caller's job
//! (AudioOut waits while more than its target latency is queued). When no
//! device is available, or `RPS5E_AUDIO=null`, no stream is opened and the
//! caller paces on its own clock. `RPS5E_AUDIO_STATS=1` logs the RMS of
//! what every stream is given (and, for cpal, what it played) every 5 s.
//!
//! cpal streams are not `Send`, so all of them are created and owned by one
//! dedicated host thread; guest threads only touch the shared ring.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

/// Sample format of the PCM handed to `put`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcmFormat {
    S16,
    F32,
}

impl PcmFormat {
    pub fn bytes_per_sample(self) -> usize {
        match self {
            PcmFormat::S16 => 2,
            PcmFormat::F32 => 4,
        }
    }
}

/// One open host output stream.
pub trait HostAudioStream: Send + Sync {
    /// Enqueue one buffer of interleaved PCM in the stream's format; false
    /// when the stream can no longer accept audio.
    fn put(&self, pcm: &[u8]) -> bool;
    /// Audio queued and not yet played, in bytes of the stream's format.
    fn queued_bytes(&self) -> usize;
    /// Drop everything queued.
    fn clear(&self);
    /// Backend name for logs.
    fn backend_name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Shared ring
// ---------------------------------------------------------------------------

struct Ring {
    samples: VecDeque<f32>,
    closed: bool,
    stats: PcmStats,
}

/// PCM energy counters of one stream, reset at every report: what the
/// guest queued and what the device callback actually played.
#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub struct PcmStats {
    pub put_samples: u64,
    pub put_energy: f64,
    pub played_samples: u64,
    pub played_energy: f64,
    pub peak: f32,
}

impl PcmStats {
    /// Account one buffer of interleaved PCM.
    pub fn add_pcm(&mut self, pcm: &[u8], format: PcmFormat) {
        let mut add = |v: f32| {
            self.put_energy += (v as f64) * (v as f64);
            self.peak = self.peak.max(v.abs());
            self.put_samples += 1;
        };
        match format {
            PcmFormat::F32 => pcm.chunks_exact(4).for_each(|c| {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                add(if v.is_finite() { v } else { 0.0 })
            }),
            PcmFormat::S16 => pcm.chunks_exact(2).for_each(|c| add(i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)),
        }
    }

    fn rms(energy: f64, n: u64) -> f64 {
        if n == 0 { 0.0 } else { (energy / n as f64).sqrt() }
    }
    pub fn put_rms(&self) -> f64 {
        Self::rms(self.put_energy, self.put_samples)
    }
    pub fn played_rms(&self) -> f64 {
        Self::rms(self.played_energy, self.played_samples)
    }
}

struct RingShared {
    ring: Mutex<Ring>,
    channels: usize,
    rate: u32,
    format: PcmFormat,
    underruns: AtomicU64,
    id: u64,
}

struct DeviceStream {
    shared: Arc<RingShared>,
    id: u64,
    name: &'static str,
}

impl HostAudioStream for DeviceStream {
    fn put(&self, pcm: &[u8]) -> bool {
        // Decoded straight into the ring: no allocation once it has grown.
        let mut ring = self.shared.ring.lock().unwrap_or_else(|e| e.into_inner());
        if ring.closed {
            return false;
        }
        let start = ring.samples.len();
        super::pcm::extend_f32(pcm, self.shared.format == PcmFormat::F32, &mut ring.samples);
        let Ring { samples, stats, .. } = &mut *ring;
        for &v in samples.range(start..) {
            stats.put_energy += (v as f64) * (v as f64);
            stats.peak = stats.peak.max(v.abs());
        }
        stats.put_samples += (samples.len() - start) as u64;
        true
    }

    fn queued_bytes(&self) -> usize {
        let ring = self.shared.ring.lock().unwrap_or_else(|e| e.into_inner());
        ring.samples.len() * self.shared.format.bytes_per_sample()
    }

    fn clear(&self) {
        self.shared.ring.lock().unwrap_or_else(|e| e.into_inner()).samples.clear();
    }

    fn backend_name(&self) -> &'static str {
        self.name
    }
}

impl Drop for DeviceStream {
    fn drop(&mut self) {
        self.shared.ring.lock().unwrap_or_else(|e| e.into_inner()).closed = true;
        if let Some(tx) = HOST_THREAD.get().and_then(|h| h.as_ref()) {
            let _ = tx.lock().unwrap_or_else(|e| e.into_inner()).send(HostCommand::Close(self.id));
        }
    }
}

// ---------------------------------------------------------------------------
// Device thread (owns every cpal stream)
// ---------------------------------------------------------------------------

enum HostCommand {
    Open(Arc<RingShared>, u64, mpsc::Sender<Result<&'static str, String>>),
    Close(u64),
    /// Process shutdown: stop every stream and end the thread.
    Shutdown(mpsc::Sender<()>),
}

/// How long shutdown waits for the device thread to stop its streams.
const SHUTDOWN_WAIT: Duration = Duration::from_millis(300);

static HOST_THREAD: OnceLock<Option<Mutex<mpsc::Sender<HostCommand>>>> = OnceLock::new();
static NEXT_STREAM_ID: AtomicU64 = AtomicU64::new(1);

/// Host backend chosen by `RPS5E_AUDIO`: `null` (no device), `cpal`
/// (cpal only) or `sdl` (SDL3 only); unset tries SDL3, then cpal.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Backend {
    Null,
    Auto,
    Sdl,
    Cpal,
}

fn backend() -> Backend {
    if cfg!(test) && std::env::var("RPS5E_AUDIO_TEST_DEVICE").is_err() {
        return Backend::Null;
    }
    static V: OnceLock<Backend> = OnceLock::new();
    *V.get_or_init(|| match std::env::var("RPS5E_AUDIO").map(|v| v.to_ascii_lowercase()).as_deref() {
        Ok("null") | Ok("none") => Backend::Null,
        Ok("cpal") | Ok("alsa") => Backend::Cpal,
        Ok("sdl") | Ok("sdl3") => Backend::Sdl,
        _ => Backend::Auto,
    })
}

// ---------------------------------------------------------------------------
// Master volume
// ---------------------------------------------------------------------------

/// Master output volume, f32 bits (`u32::MAX`: not read from
/// `RPS5E_VOLUME` yet).
static VOLUME: AtomicU32 = AtomicU32::new(u32::MAX);
/// Output silenced (the emulation is paused).
static MUTED: AtomicBool = AtomicBool::new(false);

/// `RPS5E_VOLUME=<0..100>` (percent).
fn parse_volume(v: &str) -> Option<f32> {
    v.trim().parse::<f32>().ok().filter(|p| p.is_finite()).map(|p| (p / 100.0).clamp(0.0, 1.0))
}

/// Master volume of every host stream, 0.0..=1.0 (`RPS5E_VOLUME` at start).
pub fn master_volume() -> f32 {
    match VOLUME.load(Ordering::Relaxed) {
        u32::MAX => {
            let v = std::env::var("RPS5E_VOLUME").ok().and_then(|s| parse_volume(&s)).unwrap_or(1.0);
            let _ = VOLUME.compare_exchange(u32::MAX, v.to_bits(), Ordering::Relaxed, Ordering::Relaxed);
            f32::from_bits(VOLUME.load(Ordering::Relaxed))
        }
        bits => f32::from_bits(bits),
    }
}

/// Set the master volume (applies to audio queued from now on).
pub fn set_master_volume(v: f32) {
    let v = if v.is_finite() { v.clamp(0.0, 1.0) } else { 1.0 };
    VOLUME.store(v.to_bits(), Ordering::Relaxed);
}

/// Silence (or restore) every host stream.
pub fn set_muted(on: bool) {
    MUTED.store(on, Ordering::Relaxed);
}

pub fn muted() -> bool {
    MUTED.load(Ordering::Relaxed)
}

/// Sound turned off by the player (the controller's mute button), apart
/// from the pause silence.
static USER_MUTED: AtomicBool = AtomicBool::new(false);

pub fn set_user_muted(on: bool) {
    USER_MUTED.store(on, Ordering::Relaxed);
}

pub fn user_muted() -> bool {
    USER_MUTED.load(Ordering::Relaxed)
}

/// Flip the player's mute; returns the new state.
pub fn toggle_user_muted() -> bool {
    !USER_MUTED.fetch_xor(true, Ordering::Relaxed)
}

/// The factor host output is scaled by.
pub(crate) fn output_gain() -> f32 {
    gain_of(muted(), user_muted(), master_volume())
}

/// Output gain: silent while paused or muted by the player, else the
/// master volume.
fn gain_of(paused: bool, user_muted: bool, volume: f32) -> f32 {
    if paused || user_muted {
        0.0
    } else {
        volume
    }
}

/// `pcm` (interleaved, `format`) scaled by `gain` into `out`.
pub(crate) fn scale_pcm(pcm: &[u8], format: PcmFormat, gain: f32, out: &mut Vec<u8>) {
    out.clear();
    out.reserve(pcm.len());
    match format {
        PcmFormat::S16 => {
            for c in pcm.chunks_exact(2) {
                let v = i16::from_le_bytes([c[0], c[1]]) as f32 * gain;
                out.extend_from_slice(&(v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16).to_le_bytes());
            }
        }
        PcmFormat::F32 => {
            for c in pcm.chunks_exact(4) {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * gain;
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
}

/// `RPS5E_AUDIO_STATS=1`: every 5 s each device stream logs the RMS of
/// the PCM queued by the guest and of what the device played.
pub(crate) fn stats_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("RPS5E_AUDIO_STATS").is_ok_and(|v| v != "0"))
}

fn host_thread() -> Option<&'static Mutex<mpsc::Sender<HostCommand>>> {
    HOST_THREAD
        .get_or_init(|| {
            let (tx, rx) = mpsc::channel::<HostCommand>();
            let spawned = std::thread::Builder::new()
                .name("rps5e-audio".into())
                .spawn(move || device_thread(rx));
            match spawned {
                Ok(_) => {
                    let stop = tx.clone();
                    ps5vm::shutdown::on_shutdown(ps5vm::shutdown::ORDER_AUDIO, "audio", move || {
                        let (done_tx, done_rx) = mpsc::channel();
                        if stop.send(HostCommand::Shutdown(done_tx)).is_ok() {
                            let _ = done_rx.recv_timeout(SHUTDOWN_WAIT);
                        }
                    });
                    Some(Mutex::new(tx))
                }
                Err(e) => {
                    log::warn!("[AUDIO] cannot start host audio thread: {e}");
                    None
                }
            }
        })
        .as_ref()
}

fn device_thread(rx: mpsc::Receiver<HostCommand>) {
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = cpal::default_host();
    let device = host.default_output_device();
    match &device {
        Some(d) => log::info!(
            "[AUDIO] host backend {:?}, device {:?}",
            host.id(),
            d.name().unwrap_or_else(|_| "?".into())
        ),
        None => log::warn!("[AUDIO] no host output device; output is paced without one"),
    }
    let mut streams: Vec<(u64, cpal::Stream)> = Vec::new();
    while let Ok(cmd) = rx.recv() {
        match cmd {
            HostCommand::Open(shared, id, reply) => {
                let res = match &device {
                    Some(d) => open_device_stream(d, shared).map(|s| {
                        streams.push((id, s));
                        "cpal"
                    }),
                    None => Err("no output device".into()),
                };
                let _ = reply.send(res);
            }
            HostCommand::Close(id) => streams.retain(|(sid, _)| *sid != id),
            HostCommand::Shutdown(done) => {
                streams.clear();
                let _ = done.send(());
                return;
            }
        }
    }
}

fn pick_config(
    device: &cpal::Device,
    channels: u16,
    rate: u32,
) -> Option<(cpal::StreamConfig, cpal::SampleFormat)> {
    use cpal::traits::DeviceTrait;
    let ranges: Vec<_> = device.supported_output_configs().ok()?.collect();
    let formats = [cpal::SampleFormat::F32, cpal::SampleFormat::I16];
    // Prefer the guest's exact layout, then stereo, at the guest rate.
    for want_ch in [channels, 2] {
        for fmt in formats {
            for r in &ranges {
                if r.channels() == want_ch
                    && r.sample_format() == fmt
                    && r.min_sample_rate().0 <= rate
                    && rate <= r.max_sample_rate().0
                {
                    let cfg = cpal::StreamConfig {
                        channels: want_ch,
                        sample_rate: cpal::SampleRate(rate),
                        buffer_size: cpal::BufferSize::Default,
                    };
                    return Some((cfg, fmt));
                }
            }
        }
    }
    // Device default (resampled / remapped in the callback).
    let def = device.default_output_config().ok()?;
    if formats.contains(&def.sample_format()) {
        return Some((def.config(), def.sample_format()));
    }
    None
}

/// Write one source frame to a device frame. 7.1 sources (FL FR C LFE BL
/// BR SL SR) fold into a stereo device; otherwise channels map one to one,
/// mono is duplicated and missing channels are silent.
fn map_frame(src: &[f32], dst: &mut [f32]) {
    const HALF_POWER: f32 = std::f32::consts::FRAC_1_SQRT_2;
    if dst.len() == 2 && src.len() >= 8 {
        dst[0] = src[0] + (src[2] + src[4] + src[6]) * HALF_POWER;
        dst[1] = src[1] + (src[2] + src[5] + src[7]) * HALF_POWER;
        return;
    }
    for (c, slot) in dst.iter_mut().enumerate() {
        *slot = if c < src.len() {
            src[c]
        } else if src.len() == 1 {
            src[0]
        } else {
            0.0
        };
    }
}

/// Pulls frames from the ring, remapping channels and resampling linearly.
struct Puller {
    shared: Arc<RingShared>,
    dev_channels: usize,
    step: f64,
    frac: f64,
    prev: Vec<f32>,
    next: Vec<f32>,
    lerp: Vec<f32>,
    primed: bool,
    /// Device samples written since the last stats report.
    report_samples: u64,
    /// Device samples between two reports.
    report_every: u64,
}

impl Puller {
    fn pop_frame(ring: &mut Ring, channels: usize, out: &mut [f32]) -> bool {
        if ring.samples.len() < channels {
            out.iter_mut().for_each(|s| *s = 0.0);
            return false;
        }
        for s in out.iter_mut().take(channels) {
            *s = ring.samples.pop_front().unwrap_or(0.0);
        }
        true
    }

    fn fill(&mut self, out: &mut [f32]) {
        let src_ch = self.shared.channels;
        let dev_ch = self.dev_channels;
        let mut underrun = false;
        {
            let mut ring = self.shared.ring.lock().unwrap_or_else(|e| e.into_inner());
            if !self.primed {
                underrun |= !Self::pop_frame(&mut ring, src_ch, &mut self.prev);
                underrun |= !Self::pop_frame(&mut ring, src_ch, &mut self.next);
                self.primed = true;
            }
            for frame in out.chunks_mut(dev_ch) {
                while self.frac >= 1.0 {
                    std::mem::swap(&mut self.prev, &mut self.next);
                    underrun |= !Self::pop_frame(&mut ring, src_ch, &mut self.next);
                    self.frac -= 1.0;
                }
                let t = self.frac as f32;
                for c in 0..src_ch {
                    let a = self.prev[c];
                    self.lerp[c] = a + (self.next[c] - a) * t;
                }
                map_frame(&self.lerp, frame);
                self.frac += self.step;
            }
            let gain = output_gain();
            if gain != 1.0 {
                out.iter_mut().for_each(|v| *v *= gain);
            }
            for &v in out.iter() {
                ring.stats.played_energy += (v as f64) * (v as f64);
            }
            ring.stats.played_samples += out.len() as u64;
            self.report_samples += out.len() as u64;
            if self.report_every != 0 && self.report_samples >= self.report_every {
                self.report_samples = 0;
                let st = std::mem::take(&mut ring.stats);
                let queued = ring.samples.len();
                drop(ring);
                log::info!(
                    "[AUDIO] stream {}: queued rms {:.5} peak {:.4} ({} samples), played rms {:.5} ({} samples), backlog {} samples, underruns {}",
                    self.shared.id,
                    st.put_rms(),
                    st.peak,
                    st.put_samples,
                    st.played_rms(),
                    st.played_samples,
                    queued,
                    self.shared.underruns.load(Ordering::Relaxed)
                );
            }
        }
        if underrun {
            self.shared.underruns.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn open_device_stream(device: &cpal::Device, shared: Arc<RingShared>) -> Result<cpal::Stream, String> {
    use cpal::traits::{DeviceTrait, StreamTrait};
    let (cfg, fmt) = pick_config(device, shared.channels as u16, shared.rate)
        .ok_or_else(|| "no usable output configuration".to_string())?;
    let mut puller = Puller {
        dev_channels: cfg.channels as usize,
        step: shared.rate as f64 / cfg.sample_rate.0 as f64,
        frac: 0.0,
        prev: vec![0.0; shared.channels],
        next: vec![0.0; shared.channels],
        lerp: vec![0.0; shared.channels],
        primed: false,
        report_samples: 0,
        report_every: if stats_enabled() { cfg.sample_rate.0 as u64 * cfg.channels as u64 * 5 } else { 0 },
        shared: shared.clone(),
    };
    // A broken device can report an error per period: log the first few,
    // then only at powers of two so the log cannot grow without bound.
    let err_fn = |e: cpal::StreamError| {
        static ERRORS: AtomicU64 = AtomicU64::new(0);
        let n = ERRORS.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 5 || n.is_power_of_two() {
            log::warn!("[AUDIO] stream error #{n}: {e}");
        }
    };
    let stream = match fmt {
        cpal::SampleFormat::I16 => {
            let mut scratch = Vec::new();
            device.build_output_stream(
                &cfg,
                move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                    scratch.resize(data.len(), 0.0f32);
                    puller.fill(&mut scratch);
                    for (d, s) in data.iter_mut().zip(&scratch) {
                        *d = super::pcm::float_to_pcm16(s.clamp(-1.0, 1.0));
                    }
                },
                err_fn,
                None,
            )
        }
        _ => device.build_output_stream(
            &cfg,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| puller.fill(data),
            err_fn,
            None,
        ),
    }
    .map_err(|e| e.to_string())?;
    stream.play().map_err(|e| e.to_string())?;
    log::info!(
        "[AUDIO] device stream: guest {} Hz {} ch -> device {} Hz {} ch {:?}",
        shared.rate,
        shared.channels,
        cfg.sample_rate.0,
        cfg.channels,
        fmt
    );
    Ok(stream)
}

/// Open a host device stream for interleaved PCM in the guest's layout.
/// None when there is no usable device (or audio is disabled).
pub fn open_stream(rate: u32, channels: usize, format: PcmFormat) -> Option<Arc<dyn HostAudioStream>> {
    let channels = channels.clamp(1, 16);
    let backend = backend();
    if backend == Backend::Null || !(8_000..=384_000).contains(&rate) {
        return None;
    }
    let id = NEXT_STREAM_ID.fetch_add(1, Ordering::Relaxed);
    if matches!(backend, Backend::Auto | Backend::Sdl) {
        let every = if stats_enabled() { 5 } else { 0 };
        if let Some(s) = super::sdl_host::open_stream(rate, channels, format, id, every) {
            return Some(Arc::from(s));
        }
        if backend == Backend::Sdl {
            return None;
        }
    }
    open_cpal_stream(rate, channels, format, id)
}

fn open_cpal_stream(rate: u32, channels: usize, format: PcmFormat, id: u64) -> Option<Arc<dyn HostAudioStream>> {
    let tx = host_thread()?;
    let shared = Arc::new(RingShared {
        ring: Mutex::new(Ring { samples: VecDeque::new(), closed: false, stats: PcmStats::default() }),
        channels,
        rate,
        format,
        underruns: AtomicU64::new(0),
        id,
    });
    let (rtx, rrx) = mpsc::channel();
    tx.lock()
        .unwrap_or_else(|e| e.into_inner())
        .send(HostCommand::Open(shared.clone(), id, rtx))
        .ok()?;
    match rrx.recv_timeout(Duration::from_secs(5)) {
        Ok(Ok(name)) => Some(Arc::new(DeviceStream { shared, id, name })),
        Ok(Err(e)) => {
            log::warn!("[AUDIO] device stream unavailable ({e})");
            None
        }
        Err(_) => {
            log::warn!("[AUDIO] device stream open timed out");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_mute_silences_like_pause() {
        assert_eq!(gain_of(false, false, 0.7), 0.7);
        assert_eq!(gain_of(false, true, 0.7), 0.0);
        assert_eq!(gain_of(true, false, 0.7), 0.0);
    }

    fn ring(channels: usize, rate: u32, samples: &[f32]) -> Arc<RingShared> {
        Arc::new(RingShared {
            ring: Mutex::new(Ring { samples: samples.iter().copied().collect(), closed: false, stats: PcmStats::default() }),
            channels,
            rate,
            format: PcmFormat::F32,
            underruns: AtomicU64::new(0),
            id: 0,
        })
    }

    fn puller(shared: Arc<RingShared>, dev_channels: usize, step: f64) -> Puller {
        let ch = shared.channels;
        Puller {
            shared,
            dev_channels,
            step,
            frac: 0.0,
            prev: vec![0.0; ch],
            next: vec![0.0; ch],
            lerp: vec![0.0; ch],
            primed: false,
            report_samples: 0,
            report_every: 0,
        }
    }

    /// Plays 0.5 s of a 440 Hz tone on the real host device.
    /// Run with `RPS5E_AUDIO_TEST_DEVICE=1 cargo test -- --ignored device_tone`.
    #[test]
    #[ignore]
    fn device_tone() {
        let s = open_stream(48_000, 2, PcmFormat::F32).expect("no device");
        eprintln!("backend: {}", s.backend_name());
        let mut phase = 0.0f32;
        for _ in 0..(48_000 / 256 / 2) {
            let mut grain = Vec::with_capacity(256 * 8);
            for _ in 0..256 {
                let v = (phase * std::f32::consts::TAU).sin() * 0.1;
                phase = (phase + 440.0 / 48_000.0).fract();
                grain.extend_from_slice(&v.to_le_bytes());
                grain.extend_from_slice(&v.to_le_bytes());
            }
            assert!(s.put(&grain));
        }
        while s.queued_bytes() > 0 {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn puller_remaps_and_resamples() {
        let mut p = puller(ring(1, 24_000, &[0.0, 1.0, 1.0, 1.0]), 2, 0.5);
        let mut out = vec![0.0f32; 6];
        p.fill(&mut out);
        // Mono duplicated to both channels; 2x upsampled with interpolation.
        assert_eq!(out, vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn surround_folds_into_stereo_device() {
        let frame = [0.1f32, 0.2, 0.3, 0.9, 0.4, 0.5, 0.6, 0.7];
        let mut out = [0.0f32; 2];
        map_frame(&frame, &mut out);
        let h = std::f32::consts::FRAC_1_SQRT_2;
        assert!((out[0] - (0.1 + (0.3 + 0.4 + 0.6) * h)).abs() < 1e-6);
        assert!((out[1] - (0.2 + (0.3 + 0.5 + 0.7) * h)).abs() < 1e-6);
        let mut wide = [1.0f32; 4];
        map_frame(&[0.25, 0.5], &mut wide);
        assert_eq!(wide, [0.25, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn master_volume_scales_pcm() {
        assert_eq!(parse_volume("50"), Some(0.5));
        assert_eq!(parse_volume("250"), Some(1.0));
        assert_eq!(parse_volume("x"), None);
        let s16: Vec<u8> = [1000i16, -32768, 32767].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut out = Vec::new();
        scale_pcm(&s16, PcmFormat::S16, 0.5, &mut out);
        let got: Vec<i16> = out.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        assert_eq!(got, [500, -16384, 16384]);
        let f: Vec<u8> = [0.5f32, -1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        scale_pcm(&f, PcmFormat::F32, 0.0, &mut out);
        assert!(out.chunks_exact(4).all(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) == 0.0));
        // (The global volume is not changed here: other tests play audio.)
    }

    #[test]
    fn stream_queue_accounting() {
        let shared = ring(2, 48_000, &[]);
        let s = DeviceStream { shared: shared.clone(), id: 0, name: "test" };
        let grain: Vec<u8> = [0.5f32; 8].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert!(s.put(&grain));
        assert_eq!(s.queued_bytes(), 32);
        {
            let st = shared.ring.lock().unwrap().stats;
            assert_eq!(st.put_samples, 8);
            assert!((st.put_rms() - 0.5).abs() < 1e-9);
            assert_eq!(st.peak, 0.5);
        }
        s.clear();
        assert_eq!(s.queued_bytes(), 0);
        shared.ring.lock().unwrap().closed = true;
        assert!(!s.put(&grain));
        // Not registered with the host thread: skip the close message.
        std::mem::forget(s);
    }
}
