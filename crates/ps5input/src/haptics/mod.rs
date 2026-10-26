// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! DualSense haptics and speaker: AudioOut vibration ports drive the
//! controller's voice-coil actuators and pad speaker ports its speaker,
//! instead of the emulated rumble motors.
//!
//! USB: the controller's 4-channel audio endpoint (front pair = speaker,
//! back pair = actuators). Bluetooth: raw HID reports ([`bluetooth`]).
//! While haptics are audible the motor rumble is muted (a lease that lasts
//! as long as the queued audio, at least 250 ms); the speaker is routed
//! with the same output report values the Linux driver uses without
//! headphones. Ambiguous USB or Bluetooth endpoints are never selected.
//!
//! The device side goes through [`Backend`] (SDL3 in [`sdl_backend`]).

pub mod bluetooth;
pub mod opus;
pub mod resample;
pub mod sdl_backend;

use ps5hle::pad_audio::PadAudioBlock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::thread::JoinHandle;

/// How a controller is linked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Connection {
    Wired,
    Wireless,
}

/// A playback device.
#[derive(Clone, Debug)]
pub struct AudioDevice {
    pub id: u32,
    pub name: String,
    pub channels: Option<i32>,
}

/// A HID device of the enumeration.
#[derive(Clone, Debug, Default)]
pub struct HidInfo {
    pub path: String,
    pub serial: String,
    pub bluetooth: bool,
    pub usage_page: u16,
    pub usage: u16,
}

pub trait HidDevice: Send + Sync {
    /// Write an output report; returns the bytes written or -1.
    fn write(&self, data: &[u8]) -> i32;
}

/// A 4-channel f32 stream on an audio device.
pub trait DeviceStream: Send {
    /// Channel count of the device the stream plays on.
    fn device_channels(&self) -> Option<i32>;
    fn resume(&mut self) -> bool;
    /// Queued input bytes.
    fn queued(&self) -> i32;
    fn put(&mut self, samples: &[f32]) -> bool;
    /// Take queued (converted) audio into `buf`; returns the bytes read.
    fn get(&mut self, buf: &mut [u8]) -> i32;
    /// Device-side frame size and rate.
    fn target(&self) -> Option<(i32, i32)>;
    fn clear(&mut self);
}

pub type PullCallback = Arc<dyn Fn() + Send + Sync>;

/// Device access (gamepads, audio devices, HID, Opus).
pub trait Backend: Send + Sync + 'static {
    /// Milliseconds / nanoseconds since an arbitrary start.
    fn ticks(&self) -> u64;
    fn ticks_ns(&self) -> u64;
    fn init_audio(&self) -> bool;
    fn quit_audio(&self);
    fn is_ps5(&self, controller: i32) -> bool;
    /// `None` when the controller is not open.
    fn connection(&self, controller: i32) -> Option<Connection>;
    /// Open controllers whose type is PS5, with their link.
    fn gamepads(&self) -> Vec<(i32, bool, Connection)>;
    fn rumble(&self, controller: i32, large: u16, small: u16, duration_ms: u32) -> bool;
    fn send_effect(&self, controller: i32, data: &[u8]) -> bool;
    fn playback_devices(&self) -> Vec<AudioDevice>;
    /// Open a 4-channel f32 stream at `freq` on `device`; `on_pull` runs
    /// whenever the device takes audio.
    fn open_device(&self, device: u32, freq: u32, on_pull: PullCallback) -> Option<Box<dyn DeviceStream>>;
    /// (vendor, product, serial) of a wireless PS5 controller.
    fn wireless_identity(&self, controller: i32) -> Option<(u16, u16, String)>;
    fn hid_enumerate(&self, vendor: u16, product: u16) -> Vec<HidInfo>;
    fn hid_open(&self, path: &str) -> Option<Box<dyn HidDevice>>;
    fn create_encoder(&self) -> Option<Box<dyn opus::Encoder>>;
    /// Run the Bluetooth sender: `None` on failure, `Some(None)` when the
    /// backend does not run threads.
    fn spawn_sender(&self, run: Box<dyn FnOnce() + Send>) -> Option<Option<JoinHandle<()>>>;
}

const FRAME_BYTES: i32 = 4 * 4;
const MAX_QUEUE_MS: u64 = 80;

#[derive(Default)]
struct State {
    controller: i32,
    haptics_streams: i32,
    large: u8,
    small: u8,
    rumble_until: u64,
    haptics_until: u64,
    speaker_controller: i32,
    speaker_streams: HashMap<i32, i32>,
}

/// A port's stream: a pad speaker port routes the speaker to itself, the
/// others play on the actuators.
pub struct Stream {
    freq: u32,
    speaker: bool,
    controller: i32,
    usb: Option<Box<dyn DeviceStream>>,
    bluetooth: Option<Arc<bluetooth::Stream>>,
    device: u32,
    next_check: u64,
    buffer: Vec<f32>,
}

pub struct Haptics<B: Backend> {
    backend: Arc<B>,
    state: Mutex<State>,
    bt: Arc<bluetooth::Bluetooth<B>>,
    me: Weak<Haptics<B>>,
}

impl<B: Backend> Haptics<B> {
    pub fn new(backend: Arc<B>) -> Arc<Self> {
        let bt = bluetooth::Bluetooth::new(backend.clone());
        Arc::new_cyclic(|me| Self {
            backend,
            state: Mutex::new(State { controller: -1, speaker_controller: -1, ..State::default() }),
            bt,
            me: me.clone(),
        })
    }

    pub fn bluetooth(&self) -> &Arc<bluetooth::Bluetooth<B>> {
        &self.bt
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Motors per the lease; caller holds the state.
    fn apply_rumble(&self, s: &State, force: bool) {
        let now = self.backend.ticks();
        let muted = s.haptics_until > now || s.rumble_until <= now;
        if force && !muted {
            // The driver caches motor strengths; force a report after raw
            // speaker routing changed the controller's motor mode.
            self.backend.rumble(s.controller, 0, 0, 0);
        }
        let duration = if s.rumble_until > now { (s.rumble_until - now) as u32 } else { 0 };
        let (l, m) = if muted { (0, 0) } else { (s.large as u16 * 0x101, s.small as u16 * 0x101) };
        self.backend.rumble(s.controller, l, m, duration);
    }

    fn select_controller(&self, s: &mut State, controller: i32) {
        if s.controller != controller {
            s.haptics_until = 0;
            s.rumble_until = 0;
            self.apply_rumble(s, false);
            s.controller = controller;
            s.large = 0;
            s.small = 0;
        }
    }

    /// Route the right channel to the speaker and mute the headphones
    /// (output report 0x02 values); plugged headphones are not checked.
    fn route_speaker(&self, s: &State, controller: i32, speaker: bool) -> bool {
        let report = speaker_report(speaker);
        let routed = self.backend.send_effect(controller, &report);
        if routed && controller == s.controller {
            self.apply_rumble(s, true);
        }
        routed
    }

    /// -1 restores the headphone routing.
    fn select_speaker(&self, s: &mut State, controller: i32) -> bool {
        if s.speaker_controller == controller {
            return true;
        }
        if s.speaker_controller != -1 {
            self.route_speaker(s, s.speaker_controller, false);
        }
        s.speaker_controller = -1;
        if controller != -1 && !self.route_speaker(s, controller, true) {
            return false;
        }
        s.speaker_controller = controller;
        true
    }

    /// Restore the motors once the haptics lease expired.
    pub fn update_rumble(&self) {
        let mut s = self.state();
        if s.haptics_until != 0 && self.backend.ticks() >= s.haptics_until {
            s.haptics_until = 0;
            self.apply_rumble(&s, false);
        }
    }

    fn pull_callback(&self) -> PullCallback {
        let me = self.me.clone();
        Arc::new(move || {
            if let Some(h) = me.upgrade() {
                h.update_rumble();
            }
        })
    }

    fn close_device(&self, stream: &mut Stream) {
        let had_device = stream.usb.is_some() || stream.bluetooth.is_some();
        if let Some(b) = stream.bluetooth.take() {
            self.bt.close(&b);
        }
        // Destroyed without the state lock: the device may be pulling.
        drop(stream.usb.take());
        if had_device {
            let mut s = self.state();
            if !stream.speaker {
                s.haptics_streams -= 1;
                if s.haptics_streams == 0 && s.haptics_until != 0 {
                    s.haptics_until = 0;
                    self.apply_rumble(&s, false);
                }
            }
            // Back to the headphones once no speaker port plays there.
            if stream.speaker {
                let n = s.speaker_streams.entry(stream.controller).or_insert(1);
                *n -= 1;
                if *n == 0 {
                    s.speaker_streams.remove(&stream.controller);
                    if s.speaker_controller == stream.controller {
                        self.select_speaker(&mut s, -1);
                    }
                }
            }
        }
    }

    fn is_wireless(&self, controller: i32) -> bool {
        self.backend.connection(controller) == Some(Connection::Wireless)
    }

    /// No portable link between a gamepad and its USB audio endpoint: only
    /// a single wired DualSense is used.
    fn can_use_device(&self, controller: i32) -> bool {
        let mut wired = 0;
        let mut found = false;
        for (id, ps5, conn) in self.backend.gamepads() {
            if ps5 && conn == Connection::Wired {
                wired += 1;
                found |= id == controller;
            }
        }
        found && wired == 1
    }

    fn find_device(&self) -> u32 {
        let mut device = 0;
        for d in self.backend.playback_devices() {
            let lower = d.name.to_lowercase();
            let dualsense = lower.contains("dualsense") || lower.contains("wireless controller");
            if dualsense && d.channels == Some(4) {
                if device != 0 {
                    return 0; // Several endpoints cannot be associated either.
                }
                device = d.id;
            }
        }
        device
    }

    fn refresh_device(&self, stream: &mut Stream, controller: i32, wireless: bool) {
        if stream.controller != controller
            || (wireless && stream.usb.is_some())
            || (!wireless && stream.bluetooth.is_some())
        {
            self.close_device(stream);
            stream.controller = controller;
            stream.next_check = 0;
        }
        let now = self.backend.ticks();
        if stream.bluetooth.is_some() || now < stream.next_check {
            return;
        }
        stream.next_check = now + 2000;
        let device = if wireless { 0 } else { self.find_device() };
        if stream.device == device && stream.usb.is_some() {
            return;
        }
        self.close_device(stream);
        stream.device = device;
        if !wireless && device == 0 {
            return;
        }
        let mut usb = None;
        let mut bt = None;
        let mut ready;
        if wireless {
            let me = self.me.clone();
            let tick: bluetooth::Tick = Arc::new(move || {
                if let Some(h) = me.upgrade() {
                    h.update_rumble();
                }
            });
            bt = self.bt.open(stream.freq, stream.speaker, controller, tick);
            ready = bt.is_some();
        } else {
            usb = self.backend.open_device(device, stream.freq, self.pull_callback());
            ready = match usb.as_mut() {
                Some(u) => u.device_channels() == Some(4) && u.resume(),
                None => false,
            };
        }
        if ready {
            let mut s = self.state();
            if stream.speaker {
                ready = self.select_speaker(&mut s, controller);
                if ready {
                    *s.speaker_streams.entry(controller).or_insert(0) += 1;
                }
            } else {
                s.haptics_streams += 1;
            }
        }
        if ready {
            stream.usb = usb;
            stream.bluetooth = bt;
            if !wireless {
                log::info!("haptics: playing on the DualSense audio device {device}");
            }
        } else {
            if let Some(b) = bt {
                self.bt.close(&b);
            }
            drop(usb);
            if !wireless {
                log::info!("haptics: cannot open the DualSense quad output");
            }
        }
    }

    fn queue_usb(&self, stream: &mut Stream, frames: u32) -> u64 {
        let freq = stream.freq as u64;
        let Some(usb) = stream.usb.as_mut() else { return 0 };
        let mut queued = usb.queued().max(0);
        let bytes = frames as i32 * FRAME_BYTES;
        let max_bytes = (freq * FRAME_BYTES as u64 * MAX_QUEUE_MS / 1000) as i32;
        if queued + bytes > max_bytes {
            // Keep the newest sound after a brief device stall; clearing the
            // whole stream would leave a gap much longer than one block.
            let mut discarded = [0u8; 16384];
            if let Some((frame_bytes, target_freq)) = usb.target() {
                if frame_bytes > 0 {
                    let chunk = ((discarded.len() as i32) / frame_bytes)
                        .min(((frames as u64 * target_freq as u64).div_ceil(freq)) as i32);
                    while queued + bytes > max_bytes
                        && chunk > 0
                        && usb.get(&mut discarded[..(chunk * frame_bytes) as usize]) > 0
                    {
                        queued = usb.queued().max(0);
                    }
                }
            }
            if queued + bytes > max_bytes {
                usb.clear();
                queued = 0;
            }
        }
        if !usb.put(&stream.buffer) {
            return 0;
        }
        (queued + bytes) as u64 * 1_000_000 / (freq * FRAME_BYTES as u64)
    }

    /// New port stream; `None` for a zero rate or without audio.
    pub fn open(&self, freq: u32, speaker: bool) -> Option<Stream> {
        if freq == 0 || !self.backend.init_audio() {
            return None;
        }
        Some(Stream { freq, speaker, controller: -1, usb: None, bluetooth: None, device: 0, next_check: 0, buffer: Vec::new() })
    }

    pub fn close(&self, mut stream: Stream) {
        self.close_device(&mut stream);
        self.backend.quit_audio();
    }

    /// The stream goes through the Bluetooth sender.
    pub fn uses_bluetooth(stream: &Stream) -> bool {
        stream.bluetooth.is_some()
    }

    /// Play a block on `controller`; returns how long the queued audio
    /// still plays (us), 0 when no DualSense took it.
    pub fn queue(&self, stream: &mut Stream, controller: i32, block: &PadAudioBlock) -> u64 {
        let channels = block.channels;
        if block.frames == 0 || channels == 0 || block.volume.is_empty() {
            return 0;
        }
        let max_frames = (stream.freq as u64 * MAX_QUEUE_MS / 1000) as u32;
        let first_frame = block.frames.saturating_sub(max_frames);
        let frames = block.frames - first_frame;
        if frames == 0 {
            return 0;
        }
        let wireless = self.is_wireless(controller);
        if !wireless && !self.can_use_device(controller) {
            self.close_device(stream);
            stream.next_check = 0; // Look again as soon as a DualSense is active.
            return 0;
        }
        self.refresh_device(stream, controller, wireless);
        if stream.usb.is_none() && stream.bluetooth.is_none() {
            return 0;
        }
        if stream.speaker {
            let routed = {
                let mut s = self.state();
                self.select_speaker(&mut s, controller)
            };
            if !routed {
                self.close_device(stream);
                return 0;
            }
        }
        // USB puts the speaker in front and the actuators in back;
        // Bluetooth takes stereo.
        let output_channels: usize = if wireless { 2 } else { 4 };
        let offset = if wireless || stream.speaker { 0 } else { 2 };
        stream.buffer.clear();
        stream.buffer.resize(frames as usize * output_channels, 0.0);
        let mut audible = false;
        for frame in 0..frames as usize {
            for ch in 0..2usize {
                let src = if channels == 1 { 0 } else { ch };
                let index = (first_frame as usize + frame) * channels as usize + src;
                let volume = block.volume.get(src).copied().unwrap_or(32768);
                let g = block.gains.and_then(|g| g.get(src).copied()).unwrap_or(1.0);
                let v = block.sample(index) * (volume as f32 / 32768.0 * block.gain * g);
                stream.buffer[frame * output_channels + ch + offset] = v;
                audible |= v.is_finite() && v.abs() > 1.0 / 1024.0;
            }
        }
        let queued_us = if wireless {
            let b = stream.bluetooth.clone().expect("bluetooth stream");
            self.bt.queue(&b, &stream.buffer, frames)
        } else {
            self.queue_usb(stream, frames)
        };
        if queued_us == 0 {
            self.close_device(stream);
            stream.next_check = self.backend.ticks() + 2000;
            return 0;
        }
        if audible && !stream.speaker {
            let mut s = self.state();
            self.select_controller(&mut s, controller);
            let now = self.backend.ticks();
            let starting = s.haptics_until <= now;
            s.haptics_until = s.haptics_until.max(now + (queued_us / 1000).max(250));
            if starting {
                self.apply_rumble(&s, false);
            }
        }
        queued_us
    }

    /// Motor rumble of a DualSense under the haptics lease; false for other
    /// controllers, which keep the plain rumble path.
    pub fn set_vibration(&self, controller: i32, large: u8, small: u8, duration_ms: u32) -> bool {
        if !self.backend.is_ps5(controller) {
            return false;
        }
        let mut s = self.state();
        self.select_controller(&mut s, controller);
        s.large = large;
        s.small = small;
        s.rumble_until = self.backend.ticks() + duration_ms as u64;
        self.apply_rumble(&s, false);
        true
    }

    /// Stop everything and restore the headphone routing.
    pub fn shutdown(&self) {
        self.bt.shutdown();
        let mut s = self.state();
        self.select_controller(&mut s, -1);
        self.select_speaker(&mut s, -1);
    }
}

/// Audio control output report: speaker routing on or off.
pub fn speaker_report(speaker: bool) -> [u8; 38] {
    let mut r = [0u8; 38];
    r[0] = 0x80; // Audio control valid
    if speaker {
        r[0] |= 0x20; // Speaker volume valid
        r[1] = 0x80; // Audio control 2 valid
        r[5] = 0x64; // Speaker volume
        r[7] = 0x30; // Output path: right channel to the speaker, headphones muted
        r[37] = 0x02; // Speaker preamp +6 dB
    }
    r
}

// --- the emulator's instance ----------------------------------------------------

static HAPTICS: OnceLock<Arc<Haptics<sdl_backend::SdlBackend>>> = OnceLock::new();
/// SDL id of player 1's controller (-1: none).
static ACTIVE_CONTROLLER: AtomicI32 = AtomicI32::new(-1);

/// Bring up the SDL backend (after SDL's gamepad subsystem started).
pub fn start() {
    if HAPTICS.get().is_some() {
        return;
    }
    match sdl_backend::SdlBackend::new() {
        Ok(b) => {
            let _ = HAPTICS.set(Haptics::new(Arc::new(b)));
        }
        Err(e) => log::info!("haptics: DualSense audio unavailable ({e})"),
    }
}

pub fn instance() -> Option<&'static Arc<Haptics<sdl_backend::SdlBackend>>> {
    HAPTICS.get()
}

pub fn set_active_controller(id: Option<u32>) {
    ACTIVE_CONTROLLER.store(id.map_or(-1, |i| i as i32), Ordering::Release);
}

pub fn active_controller() -> i32 {
    ACTIVE_CONTROLLER.load(Ordering::Acquire)
}

/// A port's stream in the emulator: binds to the backend once it exists.
struct PortStream {
    freq: u32,
    speaker: bool,
    stream: Option<Stream>,
}

impl ps5hle::pad_audio::PadAudioStream for PortStream {
    fn queue(&mut self, block: &PadAudioBlock) -> u64 {
        let Some(h) = instance() else { return 0 };
        if self.stream.is_none() {
            self.stream = h.open(self.freq, self.speaker);
        }
        match self.stream.as_mut() {
            Some(s) => h.queue(s, active_controller(), block),
            None => 0,
        }
    }

    fn uses_bluetooth(&self) -> bool {
        self.stream.as_ref().is_some_and(Haptics::<sdl_backend::SdlBackend>::uses_bluetooth)
    }
}

impl Drop for PortStream {
    fn drop(&mut self) {
        if let (Some(h), Some(s)) = (instance(), self.stream.take()) {
            h.close(s);
        }
    }
}

/// `ps5hle::pad_audio` opener.
pub fn open_port(freq: u32, speaker: bool) -> Option<Box<dyn ps5hle::pad_audio::PadAudioStream>> {
    (freq != 0).then(|| Box::new(PortStream { freq, speaker, stream: None }) as Box<dyn ps5hle::pad_audio::PadAudioStream>)
}

#[cfg(test)]
mod tests;
