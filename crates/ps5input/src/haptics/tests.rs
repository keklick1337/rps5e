// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The haptics module against fake gamepads, USB audio endpoints and HID
//! devices (pads 1 and 3 are DualSenses).

use super::bluetooth::{self, PERIOD_NS, REPORT_BYTES};
use super::opus::{Encoder, OpusEncoder};
use super::*;
use std::collections::VecDeque;

#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Rumble {
    large: u16,
    small: u16,
    duration: u32,
}

#[derive(Clone)]
struct World {
    devices: Vec<AudioDevice>,
    now: u64,
    audio_refs: i32,
    opens: i32,
    rumble_calls: i32,
    fail_open: bool,
    fail_resume: bool,
    actual_channels: i32,
    opened_device: u32,
    opened_freq: u32,
    rumble: Rumble,
    pad_rumble: [Rumble; 4],
    cached_rumble: [Rumble; 4],
    effects: [Vec<u8>; 4],
    connected_pads: Vec<i32>,
    fail_effect: bool,
    wireless: bool,
    wireless_pad3: bool,
    hid_available: bool,
    fail_hid_write: bool,
    duplicate_hid: bool,
    hid_opens: i32,
    hid_closes: i32,
    opened_hid_pad: i32,
    hid_reports: Vec<Vec<u8>>,
    streams: Vec<Arc<Mutex<FakeQueue>>>,
}

#[derive(Default)]
struct FakeQueue {
    samples: VecDeque<f32>,
    on_pull: Option<PullCallback>,
}

impl Default for World {
    fn default() -> Self {
        Self {
            devices: vec![dev(10, "Speakers (DualSense Wireless Controller)", 4)],
            now: 1000,
            audio_refs: 0,
            opens: 0,
            rumble_calls: 0,
            fail_open: false,
            fail_resume: false,
            actual_channels: 4,
            opened_device: 0,
            opened_freq: 0,
            rumble: Rumble::default(),
            pad_rumble: [Rumble::default(); 4],
            cached_rumble: [Rumble::default(); 4],
            effects: Default::default(),
            connected_pads: vec![1],
            fail_effect: false,
            wireless: false,
            wireless_pad3: false,
            hid_available: false,
            fail_hid_write: false,
            duplicate_hid: false,
            hid_opens: 0,
            hid_closes: 0,
            opened_hid_pad: 0,
            hid_reports: Vec::new(),
            streams: Vec::new(),
        }
    }
}

fn dev(id: u32, name: &str, channels: i32) -> AudioDevice {
    AudioDevice { id, name: name.into(), channels: Some(channels) }
}

#[derive(Clone, Default)]
struct Fake(Arc<Mutex<World>>);

impl Fake {
    fn w(&self) -> MutexGuard<'_, World> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

fn exists(c: i32) -> bool {
    c == 1 || c == 3
}

struct FakeStream {
    world: Fake,
    queue: Arc<Mutex<FakeQueue>>,
    freq: u32,
}

impl DeviceStream for FakeStream {
    fn device_channels(&self) -> Option<i32> {
        Some(self.world.w().actual_channels)
    }
    fn resume(&mut self) -> bool {
        !self.world.w().fail_resume
    }
    fn queued(&self) -> i32 {
        (self.queue.lock().unwrap().samples.len() * 4) as i32
    }
    fn put(&mut self, samples: &[f32]) -> bool {
        self.queue.lock().unwrap().samples.extend(samples);
        true
    }
    fn get(&mut self, buf: &mut [u8]) -> i32 {
        let mut q = self.queue.lock().unwrap();
        let n = (buf.len() / 4).min(q.samples.len());
        for i in 0..n {
            let v = q.samples.pop_front().unwrap();
            buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        (n * 4) as i32
    }
    fn target(&self) -> Option<(i32, i32)> {
        Some((16, self.freq as i32))
    }
    fn clear(&mut self) {
        self.queue.lock().unwrap().samples.clear();
    }
}

impl Drop for FakeStream {
    fn drop(&mut self) {
        self.world.w().streams.retain(|s| !Arc::ptr_eq(s, &self.queue));
    }
}

struct FakeHid(Fake);

impl HidDevice for FakeHid {
    fn write(&self, data: &[u8]) -> i32 {
        let mut w = self.0.w();
        if w.fail_hid_write || data.len() != REPORT_BYTES {
            return -1;
        }
        w.hid_reports.push(data.to_vec());
        data.len() as i32
    }
}

impl Drop for FakeHid {
    fn drop(&mut self) {
        self.0.w().hid_closes += 1;
    }
}

impl Backend for Fake {
    fn ticks(&self) -> u64 {
        self.w().now
    }
    fn ticks_ns(&self) -> u64 {
        self.w().now * 1_000_000
    }
    fn init_audio(&self) -> bool {
        self.w().audio_refs += 1;
        true
    }
    fn quit_audio(&self) {
        self.w().audio_refs -= 1;
    }
    fn is_ps5(&self, c: i32) -> bool {
        exists(c)
    }
    fn connection(&self, c: i32) -> Option<Connection> {
        let w = self.w();
        exists(c).then(|| {
            if w.wireless || (w.wireless_pad3 && c == 3) { Connection::Wireless } else { Connection::Wired }
        })
    }
    fn gamepads(&self) -> Vec<(i32, bool, Connection)> {
        let pads = self.w().connected_pads.clone();
        pads.into_iter().filter(|&c| exists(c)).map(|c| (c, true, self.connection(c).unwrap())).collect()
    }
    fn rumble(&self, c: i32, large: u16, small: u16, duration: u32) -> bool {
        if !exists(c) {
            return false;
        }
        let mut w = self.w();
        let i = c as usize;
        // The driver only sends a new report when the strengths change.
        if large != w.cached_rumble[i].large || small != w.cached_rumble[i].small {
            w.pad_rumble[i] = Rumble { large, small, duration };
        }
        w.cached_rumble[i] = Rumble { large, small, duration };
        w.rumble = Rumble { duration, ..w.pad_rumble[i] };
        w.rumble_calls += 1;
        true
    }
    fn send_effect(&self, c: i32, data: &[u8]) -> bool {
        let mut w = self.w();
        if !exists(c) || w.fail_effect {
            return false;
        }
        w.effects[c as usize] = data.to_vec();
        // Raw audio-routing reports clear the emulated-rumble mode without
        // updating the driver's cached motor strengths.
        if data[0] & 3 == 0 {
            w.pad_rumble[c as usize] = Rumble::default();
            w.rumble = Rumble::default();
        }
        true
    }
    fn playback_devices(&self) -> Vec<AudioDevice> {
        self.w().devices.clone()
    }
    fn open_device(&self, device: u32, freq: u32, on_pull: PullCallback) -> Option<Box<dyn DeviceStream>> {
        let mut w = self.w();
        w.opens += 1;
        w.opened_device = device;
        w.opened_freq = freq;
        if w.fail_open {
            return None;
        }
        let queue = Arc::new(Mutex::new(FakeQueue { samples: VecDeque::new(), on_pull: Some(on_pull) }));
        w.streams.push(queue.clone());
        Some(Box::new(FakeStream { world: self.clone(), queue, freq }))
    }
    fn wireless_identity(&self, c: i32) -> Option<(u16, u16, String)> {
        (self.connection(c)? == Connection::Wireless).then(|| {
            (0x054c, 0x0ce6, if c == 3 { "aa-bb-cc-dd-ee-03" } else { "aa-bb-cc-dd-ee-01" }.to_string())
        })
    }
    fn hid_enumerate(&self, _vendor: u16, _product: u16) -> Vec<HidInfo> {
        let w = self.w();
        if !w.hid_available {
            return Vec::new();
        }
        let info = |path: &str, serial: &str| HidInfo { path: path.into(), serial: serial.into(), bluetooth: true, ..Default::default() };
        let serial3 = if w.duplicate_hid { "AA:BB:CC:DD:EE:01" } else { "AA:BB:CC:DD:EE:03" };
        vec![info("pad-1", "AA:BB:CC:DD:EE:01"), info("pad-3", serial3)]
    }
    fn hid_open(&self, path: &str) -> Option<Box<dyn HidDevice>> {
        let mut w = self.w();
        w.hid_opens += 1;
        w.opened_hid_pad = if path.as_bytes()[4] == b'1' { 1 } else { 3 };
        Some(Box::new(FakeHid(self.clone())))
    }
    fn create_encoder(&self) -> Option<Box<dyn Encoder>> {
        OpusEncoder::new().map(|e| Box::new(e) as Box<dyn Encoder>)
    }
    fn spawn_sender(&self, _run: Box<dyn FnOnce() + Send>) -> Option<Option<JoinHandle<()>>> {
        Some(None)
    }
}

// --- fixture -----------------------------------------------------------------------

const UNITY: [i32; 2] = [32768, 32768];
const PCM: [f32; 4] = [0.5; 4];

struct Fixture {
    fake: Fake,
    h: Arc<Haptics<Fake>>,
}

/// A port stream closed when dropped.
struct Port<'a> {
    f: &'a Fixture,
    s: Option<Stream>,
}

impl Port<'_> {
    fn s(&mut self) -> &mut Stream {
        self.s.as_mut().unwrap()
    }
    fn reset(&mut self) {
        if let Some(s) = self.s.take() {
            self.f.h.close(s);
        }
    }
}

impl Drop for Port<'_> {
    fn drop(&mut self) {
        self.reset();
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn i16_bytes(v: &[i16]) -> Vec<u8> {
    v.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn opus_available() -> bool {
    OpusEncoder::new().is_some()
}

impl Fixture {
    fn new() -> Self {
        let fake = Fake::default();
        let h = Haptics::new(Arc::new(fake.clone()));
        Self { fake, h }
    }

    fn w(&self) -> MutexGuard<'_, World> {
        self.fake.w()
    }

    /// Snapshot of the world (no lock held while asserting).
    fn s(&self) -> World {
        self.fake.w().clone()
    }

    fn open(&self, speaker: bool) -> Port<'_> {
        self.open_rate(speaker, 48000)
    }

    fn open_rate(&self, speaker: bool, freq: u32) -> Port<'_> {
        Port { f: self, s: Some(self.h.open(freq, speaker).expect("port open failed")) }
    }

    #[allow(clippy::too_many_arguments)]
    fn queue_full(
        &self,
        port: &mut Port,
        controller: i32,
        data: &[u8],
        frames: u32,
        channels: u32,
        is_float: bool,
        volume: &[i32],
        gain: f32,
        gains: Option<&[f32]>,
    ) -> u64 {
        let block = PadAudioBlock { data, frames, channels, is_float, volume, gain, gains };
        self.h.queue(port.s(), controller, &block)
    }

    fn queue_on(&self, port: &mut Port, controller: i32, pcm: &[f32], frames: u32) -> u64 {
        self.queue_full(port, controller, &f32_bytes(pcm), frames, 2, true, &UNITY, 1.0, None)
    }

    fn queue(&self, port: &mut Port, pcm: &[f32], frames: u32) -> u64 {
        self.queue_on(port, 1, pcm, frames)
    }

    fn pull(&self) {
        let q = self.w().streams.last().cloned().expect("audio device pull failed");
        let cb = q.lock().unwrap().on_pull.clone().unwrap();
        cb();
        let mut q = q.lock().unwrap();
        for _ in 0..64 {
            q.samples.pop_front();
        }
    }

    fn read(&self, n: usize) -> Vec<f32> {
        let q = self.w().streams.last().cloned().expect("no stream");
        let mut q = q.lock().unwrap();
        (0..n).map(|_| q.samples.pop_front().expect("wrong output byte count")).collect()
    }

    fn expect_pcm(&self, expected: [f32; 8]) {
        let actual = self.read(8);
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-6, "haptics mapping or volume is wrong: {actual:?} vs {expected:?}");
        }
    }

    fn bt(&self) -> &Arc<bluetooth::Bluetooth<Fake>> {
        self.h.bluetooth()
    }

    fn now_ns(&self) -> u64 {
        self.w().now * 1_000_000
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.h.shutdown();
    }
}

// --- USB ---------------------------------------------------------------------------

#[test]
fn formats_and_volume() {
    let f = Fixture::new();
    let mut port = f.open(false);
    let volume = [32768, 16384];
    let stereo = f32_bytes(&[0.5, -0.25, 1.0, 0.0]);
    f.queue_full(&mut port, 1, &stereo, 2, 2, true, &volume, 1.0, None);
    assert_eq!((f.s().opened_device, f.s().opened_freq), (10, 48000), "did not select quad DualSense PCM");
    f.expect_pcm([0.0, 0.0, 0.5, -0.125, 0.0, 0.0, 1.0, 0.0]);
    let gains = [0.25f32, 2.0];
    f.queue_full(&mut port, 1, &stereo, 2, 2, true, &volume, 1.0, Some(&gains));
    f.expect_pcm([0.0, 0.0, 0.125, -0.25, 0.0, 0.0, 0.25, 0.0]);
    let mono = i16_bytes(&[16384, -32768]);
    f.queue_full(&mut port, 1, &mono, 2, 1, false, &volume[1..], 1.0, None);
    f.expect_pcm([0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5, -0.5]);
    f.queue_full(&mut port, 1, &mono, 2, 1, false, &volume[1..], 0.5, None);
    f.expect_pcm([0.0, 0.0, 0.125, 0.125, 0.0, 0.0, -0.25, -0.25]);
    f.queue_full(&mut port, 1, &mono, 2, 1, false, &volume[1..], 1.0, Some(&gains));
    f.expect_pcm([0.0, 0.0, 0.0625, 0.0625, 0.0, 0.0, -0.125, -0.125]);
    let mut multichannel = [0.9f32; 24];
    multichannel[0] = 0.1;
    multichannel[1] = 0.2;
    multichannel[12] = 0.3;
    multichannel[13] = 0.4;
    f.queue_full(&mut port, 1, &f32_bytes(&multichannel), 2, 12, true, &UNITY, 1.0, None);
    f.expect_pcm([0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.3, 0.4]);
}

#[test]
fn discovery_and_hotplug() {
    let f = Fixture::new();
    f.w().devices = vec![dev(10, "DualSense Wireless Controller", 2), dev(20, "Other speakers", 4)];
    let mut port = f.open(false);
    f.queue(&mut port, &PCM, 2);
    assert!(f.s().streams.is_empty() && f.s().opens == 0, "stereo controller or unrelated speakers selected");
    f.w().devices[0].channels = Some(4);
    f.w().now += 2001;
    f.queue(&mut port, &PCM, 2);
    f.w().devices.clear();
    f.w().now += 2001;
    f.queue(&mut port, &PCM, 2);
    assert!(f.s().streams.is_empty(), "disconnected endpoint retained its stream");
    f.w().devices = vec![dev(10, "Wireless Controller", 4)];
    f.w().now += 2001;
    f.queue(&mut port, &PCM, 2);
    assert!(f.s().streams.len() == 1 && f.s().opened_device == 10, "Linux endpoint name not discovered");
    f.w().devices.clear();
    f.w().now += 2001;
    f.queue(&mut port, &PCM, 2);
    assert!(f.s().streams.is_empty(), "disconnected endpoint retained its stream");
}

#[test]
fn speaker() {
    let f = Fixture::new();
    f.w().devices = vec![dev(10, "DualSense Wireless Controller", 2)];
    let mut speaker = f.open(true);
    assert!(f.queue(&mut speaker, &PCM, 2) == 0 && f.s().streams.is_empty(), "speaker played without a quad device");
    f.w().devices = vec![dev(10, "Speakers (DualSense Wireless Controller)", 4)];
    f.w().now += 2001;
    assert!(f.h.set_vibration(1, 100, 50, 0xffff), "set rumble failed");
    let stereo = [0.5, -0.25, 1.0, 0.0];
    f.queue(&mut speaker, &stereo, 2);
    f.expect_pcm([0.5, -0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    assert_eq!(f.s().rumble.large, 100 * 0x101, "speaker audio stopped the rumble");
    let routed = f.s().effects[1].clone();
    assert_eq!(routed, speaker_report(true).to_vec());
    assert!(routed.len() == 38 && routed[0] == 0xa0 && routed[1] == 0x80 && routed[5] == 0x64 && routed[7] == 0x30 && routed[37] == 2);
    assert_eq!(f.queue(&mut speaker, &stereo, 2), 2 * 1_000_000 / 48000, "queued speaker playback time is wrong");
    let mut other = f.open(true);
    f.w().connected_pads = vec![3];
    f.queue_on(&mut other, 3, &PCM, 2);
    {
        let w = f.w();
        assert!(w.effects[1][0] == 0x80 && w.effects[1][1] == 0 && w.effects[1][7] == 0 && w.effects[3] == routed, "switching pads left the old speaker routed");
    }
    speaker.reset();
    assert_eq!(f.s().effects[3], routed, "closing one speaker port unrouted the other");
    other.reset();
    assert!(f.s().effects[3][0] == 0x80 && f.s().effects[3][7] == 0, "closing the last speaker port left it routed");
    let mut last = f.open(true);
    f.queue_on(&mut last, 3, &PCM, 2);
    assert_eq!(f.s().effects[3], routed, "reopened speaker port was not routed");
    f.h.shutdown();
    assert!(f.s().effects[3][0] == 0x80 && f.s().effects[3][7] == 0, "shutdown left the speaker routed");
}

#[test]
fn speaker_unplug() {
    let f = Fixture::new();
    let mut speaker = f.open(true);
    assert_ne!(f.queue(&mut speaker, &PCM, 2), 0, "speaker did not play on the DualSense");
    let routed = f.s().effects[1].clone();
    // Another controller becomes active: the port falls back to the main output.
    assert!(f.queue_on(&mut speaker, 2, &PCM, 2) == 0 && f.s().streams.is_empty(), "unplugged pad kept the speaker port");
    assert!(f.s().effects[1][0] == 0x80 && f.s().effects[1][7] == 0, "unplugging left the speaker routed");
    f.w().effects = Default::default();
    assert!(f.queue(&mut speaker, &PCM, 2) != 0 && f.s().streams.len() == 1 && f.s().effects[1] == routed, "replugged pad did not take the speaker back");
    f.w().devices.clear();
    f.w().now += 2001;
    assert!(f.queue(&mut speaker, &PCM, 2) == 0 && f.s().streams.is_empty(), "lost endpoint kept the speaker port");
    assert!(f.s().effects[1][0] == 0x80 && f.s().effects[1][7] == 0, "lost endpoint left the speaker routed");
}

#[test]
fn failures_and_bounded_queue() {
    let f = Fixture::new();
    let mut port = f.open(false);
    let block = [0.5f32; 128];
    f.w().fail_open = true;
    f.queue(&mut port, &block, 64);
    assert!(f.s().streams.is_empty() && f.s().audio_refs == 1, "failed open leaked a stream/reference");
    f.w().fail_open = false;
    f.w().fail_resume = true;
    f.w().now += 2001;
    f.queue(&mut port, &block, 64);
    assert!(f.s().streams.is_empty() && f.s().audio_refs == 1, "failed resume leaked a stream/reference");
    f.w().fail_resume = false;
    f.w().actual_channels = 2;
    f.w().now += 2001;
    f.queue(&mut port, &block, 64);
    assert!(f.s().streams.is_empty(), "post-open stereo downgrade accepted");
    f.w().actual_channels = 4;
    f.w().now += 2001;
    for _ in 0..70 {
        f.queue(&mut port, &block, 64);
    }
    let block_bytes = 64 * 16;
    let max_queue = 48000 * 16 * 80 / 1000;
    let q = f.s().streams.last().unwrap().lock().unwrap().samples.len() * 4;
    assert!(f.s().streams.len() == 1 && q <= max_queue + block_bytes, "queue grows indefinitely ({q})");
    drop(port);
    assert_eq!(f.s().audio_refs, 0, "audio subsystem reference leaked");
}

#[test]
fn rumble_lease_and_duration() {
    let f = Fixture::new();
    let mut port = f.open(false);
    assert!(f.h.set_vibration(1, 100, 50, 0xffff));
    assert!(f.s().rumble.large == 100 * 0x101 && f.s().rumble.duration == 65535, "ordinary rumble changed");
    f.queue(&mut port, &PCM, 2);
    assert!(f.s().rumble.large == 0 && f.s().rumble.small == 0, "rumble was not stopped before haptics");
    f.w().now = 1100;
    f.queue(&mut port, &[0.0; 4], 2);
    f.w().now = 1249;
    f.pull();
    assert_eq!(f.s().rumble.large, 0, "rumble resumed before lease expiration");
    f.w().now = 1250;
    f.pull();
    assert_eq!(f.s().rumble, Rumble { large: 100 * 0x101, small: 50 * 0x101, duration: 65285 }, "idle haptics did not restore rumble");
    f.w().now = 66500;
    f.queue(&mut port, &PCM, 2);
    f.w().now = 66751;
    f.pull();
    assert!(f.s().rumble.large == 0 && f.s().rumble.duration == 0, "expired rumble restarted after haptics");
    f.h.shutdown();
    let calls = f.s().rumble_calls;
    f.pull();
    assert_eq!(f.s().rumble_calls, calls, "audio pull touched rumble after shutdown");
}

#[test]
fn scaled_rumble_duration() {
    for bluetooth in [false, true] {
        let f = Fixture::new();
        f.w().wireless = bluetooth;
        f.w().hid_available = bluetooth;
        let mut port = f.open(false);
        let sound = f32_bytes(&[0.5; 2048]);
        f.h.set_vibration(1, 100, 50, 1000);
        f.queue_full(&mut port, 1, &sound, 1024, 2, true, &UNITY, 0.0, None);
        assert_eq!(f.s().rumble.large, 100 * 0x101, "muted haptics suppressed motor rumble");
        f.queue_full(&mut port, 1, &sound, 1024, 2, true, &UNITY, 0.5, None);
        assert_eq!(f.s().rumble.large, 0, "scaled haptics did not suppress motor rumble");
        f.w().now += 100;
        f.h.set_vibration(1, 50, 25, 900);
        f.w().now += 200;
        if bluetooth {
            f.h.update_rumble();
        } else {
            f.pull();
        }
        assert_eq!(f.s().rumble, Rumble { large: 50 * 0x101, small: 25 * 0x101, duration: 700 }, "bluetooth={bluetooth}");
        f.h.set_vibration(1, 50, 25, 0);
        assert_eq!(f.s().rumble, Rumble::default(), "expired rumble was restarted");
    }
}

#[test]
fn switch_stops_old_rumble() {
    let f = Fixture::new();
    let mut port = f.open(false);
    f.h.set_vibration(1, 100, 50, 0xffff);
    f.queue(&mut port, &PCM, 2);
    assert_eq!(f.s().pad_rumble[1].large, 0, "haptics did not suppress rumble");
    f.h.set_vibration(3, 10, 10, 0xffff);
    assert!(f.s().pad_rumble[1].large == 0 && f.s().pad_rumble[1].small == 0, "switching restarted the old pad's rumble");
    assert_eq!(f.s().pad_rumble[3].large, 10 * 0x101, "new pad did not rumble");
    assert!(!f.h.set_vibration(2, 10, 10, 0xffff), "a non-DualSense took the haptics rumble path");
}

#[test]
fn close_restores_rumble() {
    let f = Fixture::new();
    let mut speaker = f.open(true);
    f.queue(&mut speaker, &PCM, 2);
    let (mut first, mut second) = (f.open(false), f.open(false));
    f.h.set_vibration(1, 100, 50, 0xffff);
    f.queue(&mut first, &PCM, 2);
    f.queue(&mut second, &PCM, 2);
    assert_eq!(f.s().rumble.large, 0, "haptics did not suppress rumble");
    first.reset();
    assert_eq!(f.s().rumble.large, 0, "closing one of two streams restored rumble early");
    second.reset();
    assert!(f.s().rumble.large == 100 * 0x101 && f.s().rumble.small == 50 * 0x101, "closing the final stream left rumble suppressed");
}

#[test]
fn speaker_switch_and_failures() {
    let f = Fixture::new();
    let mut speaker = f.open(true);
    f.w().fail_effect = true;
    assert!(f.queue(&mut speaker, &PCM, 2) == 0 && f.s().streams.is_empty(), "failed routing suppressed fallback");
    f.w().fail_effect = false;
    f.w().now += 2001;
    assert_ne!(f.queue(&mut speaker, &PCM, 2), 0, "speaker did not recover after routing failed");
    f.w().connected_pads = vec![3];
    f.w().devices = vec![dev(20, "Speakers (DualSense Wireless Controller)", 4)];
    assert!(f.queue_on(&mut speaker, 3, &PCM, 2) != 0 && f.s().opened_device == 20);
    assert!(f.s().effects[1][7] == 0 && f.s().effects[3][7] == 0x30, "speaker port did not follow the new controller");
    f.w().connected_pads = vec![1];
    let mut other = f.open(true);
    assert_ne!(f.queue(&mut other, &PCM, 2), 0, "second speaker port failed");
    other.reset();
    assert_eq!(f.s().effects[1][7], 0, "old controller stream kept the new speaker routed");
}

#[test]
fn speaker_requires_unambiguous_usb_device() {
    let f = Fixture::new();
    let mut speaker = f.open(true);
    f.w().wireless = true;
    assert!(f.queue(&mut speaker, &PCM, 2) == 0 && f.s().streams.is_empty(), "Bluetooth pad stole USB audio");
    f.w().wireless = false;
    f.w().connected_pads = vec![1, 3];
    assert!(f.queue(&mut speaker, &PCM, 2) == 0 && f.s().streams.is_empty(), "several DualSenses chose one");
    f.w().connected_pads = vec![1];
    f.w().devices.push(dev(20, "DualSense Wireless Controller", 4));
    assert!(f.queue(&mut speaker, &PCM, 2) == 0 && f.s().streams.is_empty(), "several endpoints chose one");
    f.w().devices.pop();
    f.w().now += 2001;
    assert_ne!(f.queue(&mut speaker, &PCM, 2), 0, "unambiguous USB device was not recovered");
    f.w().wireless = true;
    assert!(f.queue(&mut speaker, &PCM, 2) == 0 && f.s().streams.is_empty() && f.s().effects[1][7] == 0, "USB to Bluetooth kept the USB speaker");
}

#[test]
fn usb_overflow_keeps_recent_audio() {
    let f = Fixture::new();
    let mut speaker = f.open(true);
    for block in 1..=4 {
        assert_ne!(f.queue(&mut speaker, &[block as f32 * 0.1; 2048], 1024), 0, "USB overflow queue failed");
    }
    let first = f.read(4);
    assert!(first[0] > 0.15 && first[0] < 0.25, "USB overflow did not keep the newest audio");
    // A single oversized input keeps only its newest 80 ms.
    let mut sound = vec![0.1f32; 4096 * 2];
    sound[256 * 2..].fill(0.5);
    assert_eq!(f.queue(&mut speaker, &sound, 4096), 80000);
    let q = f.s().streams.last().unwrap().lock().unwrap().samples.len();
    assert_eq!(q, 3840 * 4, "oversized block did not retain only the newest 80 ms");
    let first = f.read(4);
    assert!(first[0] == 0.5 && first[1] == 0.5);
}

// --- Bluetooth -------------------------------------------------------------------

fn bt_fixture() -> Fixture {
    let f = Fixture::new();
    f.w().wireless = true;
    f.w().hid_available = true;
    f
}

#[test]
fn bluetooth_audio_and_identity() {
    if !opus_available() {
        return;
    }
    let f = bt_fixture();
    f.w().connected_pads = vec![1, 3];
    f.w().devices.clear();
    let mut vibration = f.open(false);
    let mut speaker = f.open(true);
    let haptic = [0.5f32; 2048];
    let sound = [0.25f32; 2048];
    assert!(f.h.set_vibration(1, 100, 80, 0xffff) && f.s().pad_rumble[1].large == 100 * 0x101);
    assert_ne!(f.queue(&mut vibration, &haptic, 1024), 0, "Bluetooth haptics did not open");
    assert_ne!(f.queue(&mut speaker, &sound, 1024), 0, "Bluetooth speaker did not open");
    assert!(f.s().hid_opens == 1 && f.s().opened_hid_pad == 1 && f.s().effects[1][7] == 0x30, "not matched to the active pad");
    assert_eq!(f.s().pad_rumble[1].large, 0, "Bluetooth haptics did not suppress motor rumble");
    let pending = f.bt().prepare(f.now_ns());
    assert_eq!(pending.len(), 1, "Bluetooth streams were not combined in one report");
    f.bt().send(&pending);
    assert_eq!(f.s().hid_reports.len(), 1, "Bluetooth audio report was not sent");
    let report = f.s().hid_reports[0].clone();
    assert!(report[0] == 0x39 && report[2] == 0x91 && report[3] == 6 && report[4] == 0x7e);
    assert!(report[10] == 0xd2 && report[11] == 64 && report[140] == 0xd3 && report[141] == 200, "malformed sub-packets");
    assert!(report[12] > 0 && report[13] > 0, "haptic waveform was not downsampled");
    let mut whole = vec![0xa2];
    whole.extend_from_slice(&report[..REPORT_BYTES - 4]);
    assert_eq!(&report[REPORT_BYTES - 4..], &bluetooth::crc32(0, &whole).to_le_bytes(), "invalid CRC");
    let decoded = opus::decode(&report[142..342], 480).unwrap().expect("speaker packet could not be decoded");
    let mean = decoded.iter().sum::<f32>() / decoded.len() as f32;
    assert!(mean > 0.05 && mean < 0.35, "speaker audio was not preserved ({mean})");
    assert!(f.bt().prepare(f.now_ns()).is_empty(), "sender ignored report pacing");
    f.w().now += 300;
    f.h.update_rumble();
    assert_eq!(f.s().pad_rumble[1].large, 100 * 0x101, "haptics did not restore normal rumble");

    // Another active DualSense has another HID address.
    assert!(f.queue_on(&mut vibration, 3, &haptic, 1024) != 0 && f.s().opened_hid_pad == 3, "followed the wrong pad");
    assert_ne!(f.queue_on(&mut speaker, 3, &sound, 1024), 0);
    assert_eq!(f.s().effects[1][7], 0, "old Bluetooth controller remained speaker-routed");
    f.w().wireless = false;
    f.w().connected_pads = vec![3];
    f.w().devices = vec![dev(20, "Speakers (DualSense Wireless Controller)", 4)];
    assert!(f.queue_on(&mut speaker, 3, &sound, 1024) != 0 && f.s().opened_device == 20 && f.s().effects[3][7] == 0x30, "Bluetooth to USB did not reopen USB");
}

#[test]
fn bluetooth_failure_falls_back() {
    if !opus_available() {
        return;
    }
    let f = bt_fixture();
    let mut speaker = f.open(true);
    let sound = [0.25f32; 2048];
    assert_ne!(f.queue(&mut speaker, &sound, 1024), 0, "Bluetooth port did not open");
    f.w().fail_hid_write = true;
    let pending = f.bt().prepare(f.now_ns());
    f.bt().send(&pending);
    assert!(f.queue(&mut speaker, &sound, 1024) == 0 && f.s().effects[1][7] == 0, "failed write kept the controller");
}

#[test]
fn bluetooth_formats_and_routing_failure() {
    if !opus_available() {
        return;
    }
    let f = bt_fixture();
    f.w().fail_effect = true;
    let mut speaker = f.open(true);
    assert_eq!(f.queue(&mut speaker, &PCM, 2), 0);
    assert!(!Haptics::<Fake>::uses_bluetooth(speaker.s()) && f.s().hid_closes == 1 && f.s().effects[1].is_empty(), "failed routing leaked a session");
    f.w().fail_effect = false;
    f.w().now += 2001;
    let mono = i16_bytes(&[16384; 1024]);
    f.queue_full(&mut speaker, 1, &mono, 1024, 1, false, &[16384], 0.5, None);
    let pending = f.bt().prepare(f.now_ns());
    assert!(pending.len() == 1 && f.s().hid_opens == 2 && f.s().effects[1][7] == 0x30);
    assert!((pending[0].audio[200] - 0.125).abs() < 1e-4 && (pending[0].audio[201] - 0.125).abs() < 1e-4, "mono int16 mapping failed");
}

#[test]
fn bluetooth_close_cancels_prepared_report() {
    if !opus_available() {
        return;
    }
    let f = bt_fixture();
    let mut speaker = f.open(true);
    let mut vibration = f.open(false);
    let sound = [0.25f32; 2048];
    f.queue(&mut speaker, &sound, 1024);
    f.queue(&mut vibration, &sound, 1024);
    let base = f.now_ns();
    let pending = f.bt().prepare(base);
    assert_eq!(pending.len(), 1, "report was not prepared before stream close");
    speaker.reset();
    f.bt().send(&pending);
    assert!(f.s().hid_reports.is_empty(), "closed speaker's prepared report was sent");
    f.queue(&mut vibration, &sound, 1024);
    f.bt().send(&f.bt().prepare(base + PERIOD_NS));
    assert!(f.s().hid_reports.len() == 1 && f.s().hid_reports[0][140] == 0, "remaining haptic stream did not send");
}

#[test]
fn bluetooth_speaker_waits_for_its_audio_block() {
    if !opus_available() {
        return;
    }
    let f = bt_fixture();
    let mut vibration = f.open(false);
    let mut speaker = f.open(true);
    assert_ne!(f.queue(&mut vibration, &[0.5; 2048], 1024), 0);
    assert!(f.bt().prepare(f.now_ns()).is_empty(), "haptics sent a silent speaker packet early");
    assert_ne!(f.queue(&mut speaker, &[0.25; 2048], 256), 0);
    assert!(f.bt().prepare(f.now_ns()).is_empty(), "short speaker block was padded immediately");
    assert_ne!(f.queue(&mut speaker, &[0.25; 2048], 768), 0);
    let reports = f.bt().prepare(f.now_ns());
    assert!(reports.len() == 1 && reports[0].audio[100] > 0.2 && reports[0].data[12] > 0, "not combined");
}

#[test]
fn bluetooth_short_haptic_block() {
    let f = bt_fixture();
    let mut vibration = f.open(false);
    f.queue(&mut vibration, &[0.25; 2048], 256);
    let base = f.now_ns();
    assert!(f.bt().prepare(base).is_empty() && f.bt().prepare(base + 1_000_000).is_empty(), "short block padded");
    f.queue(&mut vibration, &[0.25; 2048], 768);
    assert!(f.bt().prepare(base + 1_000_000).is_empty(), "completed block skipped its coalescing window");
    let pending = f.bt().prepare(base + 2_000_000);
    assert!(pending.len() == 1 && pending[0].data[12 + 100] > 0, "complete block did not play");
}

#[test]
fn bluetooth_incremental_batch_coalesces() {
    if !opus_available() {
        return;
    }
    let f = bt_fixture();
    let mut vibration = f.open(false);
    let mut speaker = f.open(true);
    let sound = [0.25f32; 512];
    for _ in 0..3 {
        f.queue(&mut vibration, &sound, 256);
        f.queue(&mut speaker, &sound, 256);
        assert!(f.bt().prepare(f.now_ns()).is_empty(), "incremental batch was padded");
        f.w().now += 5;
    }
    f.queue(&mut vibration, &sound, 256);
    assert!(f.bt().prepare(f.now_ns()).is_empty(), "haptics sent before the batch's last speaker block");
    f.queue(&mut speaker, &sound, 256);
    let pending = f.bt().prepare(f.now_ns());
    assert!(pending.len() == 1 && pending[0].audio[1800] > 0.2 && pending[0].data[12 + 100] > 0);
}

#[test]
fn bluetooth_low_rate_complete_block() {
    if !opus_available() {
        return;
    }
    let f = bt_fixture();
    let mut speaker = f.open_rate(true, 8000);
    f.queue(&mut speaker, &[0.25; 171 * 2], 171);
    let pending = f.bt().prepare(f.now_ns());
    assert!(pending.len() == 1 && pending[0].audio[1000] > 0.2, "low-rate block was delayed");
}

#[test]
fn bluetooth_complete_stream_does_not_wait_for_partial_peer() {
    if !opus_available() {
        return;
    }
    let f = bt_fixture();
    let mut complete = f.open(true);
    let mut partial = f.open(true);
    f.queue(&mut complete, &[0.25; 2048], 1024);
    f.queue(&mut partial, &[0.25; 2048], 256);
    let pending = f.bt().prepare(f.now_ns());
    assert!(pending.len() == 1 && pending[0].audio[100] > 0.4 && pending[0].audio[1000] > 0.2);
}

#[test]
fn bluetooth_short_final_block() {
    for speaker in [false, true] {
        if speaker && !opus_available() {
            continue;
        }
        let f = bt_fixture();
        let mut port = f.open(speaker);
        f.queue(&mut port, &[0.25; 2048], 256);
        let base = f.now_ns();
        assert!(f.bt().prepare(base).is_empty(), "short final block sent too early");
        assert_eq!(f.bt().prepare(base + PERIOD_NS).len(), 1, "short final block was never played");
    }
}

#[test]
fn bluetooth_overflow_keeps_recent_audio() {
    if !opus_available() {
        return;
    }
    let f = bt_fixture();
    let mut speaker = f.open(true);
    for block in 1..=4 {
        assert_ne!(f.queue(&mut speaker, &[block as f32 * 0.1; 2048], 1024), 0);
    }
    let base = f.now_ns();
    let first = f.bt().prepare(base);
    assert!(first.len() == 1 && first[0].audio[200] > 0.15 && first[0].audio[200] < 0.25, "overflow lost all audio");
    let second = f.bt().prepare(base + PERIOD_NS);
    assert!(second.len() == 1 && second[0].audio[200] > 0.25 && second[0].audio[200] < 0.35, "next block lost");
}

#[test]
fn bluetooth_haptic_overflow_keeps_recent_audio() {
    let f = bt_fixture();
    let mut vibration = f.open(false);
    for block in 1..=4 {
        assert_ne!(f.queue(&mut vibration, &[block as f32 * 0.1; 2048], 1024), 0);
    }
    let base = f.now_ns();
    assert!(f.bt().prepare(base).is_empty(), "haptic-only report skipped its coalescing window");
    let first = f.bt().prepare(base + 1_000_000);
    assert!(first.len() == 1 && first[0].data[12 + 32] > 18 && first[0].data[12 + 32] < 32, "discarded too much");
}

#[test]
fn bluetooth_recovers_after_late_write() {
    if !opus_available() {
        return;
    }
    let f = bt_fixture();
    let mut speaker = f.open(true);
    for _ in 0..3 {
        assert_ne!(f.queue(&mut speaker, &[0.25; 2048], 1024), 0);
    }
    let base = f.now_ns();
    assert_eq!(f.bt().prepare(base).len(), 1);
    assert_eq!(f.bt().prepare(base + PERIOD_NS + 10_000_000).len(), 1, "late report was not prepared");
    assert_eq!(f.bt().prepare(base + 2 * PERIOD_NS).len(), 1, "sender did not recover its schedule");
}

#[test]
fn bluetooth_ambiguous_device() {
    let f = bt_fixture();
    f.w().duplicate_hid = true;
    let mut speaker = f.open(true);
    assert!(f.queue(&mut speaker, &PCM, 2) == 0 && f.s().hid_opens == 0 && f.s().effects[1].is_empty(), "ambiguous device routed");
}

#[test]
fn usb_and_bluetooth_feature_parity() {
    if !opus_available() {
        return;
    }
    let f = Fixture::new();
    f.w().connected_pads = vec![1, 3];
    f.w().wireless_pad3 = true;
    f.w().hid_available = true;
    let mut wired = f.open(true);
    let mut wireless = f.open(true);
    let sound = [0.25f32; 2048];
    assert!(f.queue_on(&mut wired, 1, &sound, 1024) != 0 && f.s().opened_device == 10, "Bluetooth pad disabled the wired speaker");
    assert!(f.queue_on(&mut wireless, 3, &sound, 1024) != 0 && f.s().opened_hid_pad == 3, "wired pad disabled Bluetooth");
    assert!(f.queue_on(&mut wired, 1, &sound, 1024) != 0 && f.s().effects[1][7] == 0x30 && f.s().effects[3][7] == 0);
}

#[test]
fn vibration_gain() {
    for bluetooth in [false, true] {
        let f = Fixture::new();
        f.w().wireless = bluetooth;
        f.w().hid_available = bluetooth;
        let mut port = f.open(false);
        f.queue_full(&mut port, 1, &f32_bytes(&[0.5; 2048]), 1024, 2, true, &UNITY, 0.5, None);
        if bluetooth {
            assert!(f.bt().prepare(f.now_ns()).is_empty(), "vibration skipped its coalescing window");
            let pending = f.bt().prepare((f.s().now + 1) * 1_000_000);
            assert!(pending.len() == 1 && pending[0].data[76] == 32, "vibration gain not applied");
        } else {
            f.expect_pcm([0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.25, 0.25]);
        }
    }
}
