// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Output scheduling with a deterministic clock and host queues that drain
//! at the sample rate, plus the gain/volume path.

use super::*;
use crate::context::CpuRegister;
use std::sync::{Arc, Mutex};

const BASE: u64 = 0x5C40_0000_0000;

#[derive(Default)]
struct World {
    now: u64,
    oversleep: u64,
    processing: u64,
    sleeps: Vec<u64>,
    fail_open: bool,
    fail_put: bool,
    stalled: bool,
    clears: u32,
    streams: Vec<Arc<Mutex<StreamState>>>,
}

struct StreamState {
    freq: u32,
    frame_bytes: usize,
    /// Millionths of a frame keep simulated consumption exact.
    frames: u64,
    updated: u64,
    pcm: Vec<u8>,
}

fn drain(w: &World, s: &mut StreamState) {
    if !w.stalled {
        let consumed = (w.now - s.updated) * s.freq as u64;
        s.frames -= s.frames.min(consumed);
    }
    s.updated = w.now;
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

struct FakeStream {
    world: Arc<Mutex<World>>,
    state: Arc<Mutex<StreamState>>,
}

impl HostAudioStream for FakeStream {
    fn put(&self, pcm: &[u8]) -> bool {
        let mut w = lock(&self.world);
        w.now += w.processing;
        if w.fail_put {
            return false;
        }
        let mut s = lock(&self.state);
        s.pcm = pcm.to_vec();
        drain(&w, &mut s);
        s.frames += pcm.len() as u64 * 1_000_000 / s.frame_bytes as u64;
        true
    }

    fn queued_bytes(&self) -> usize {
        let w = lock(&self.world);
        let mut s = lock(&self.state);
        drain(&w, &mut s);
        (s.frames / 1_000_000) as usize * s.frame_bytes
    }

    fn clear(&self) {
        let mut w = lock(&self.world);
        let mut s = lock(&self.state);
        s.frames = 0;
        s.updated = w.now;
        w.clears += 1;
    }

    fn backend_name(&self) -> &'static str {
        "fake"
    }
}

impl Drop for FakeStream {
    fn drop(&mut self) {
        lock(&self.world).streams.retain(|s| !Arc::ptr_eq(s, &self.state));
    }
}

struct Fake(Arc<Mutex<World>>);

impl Env for Fake {
    fn now_us(&self) -> u64 {
        lock(&self.0).now
    }

    fn sleep_us(&self, us: u64) {
        let mut w = lock(&self.0);
        w.sleeps.push(us);
        w.now += us + w.oversleep;
    }

    fn open_stream(&self, rate: u32, channels: usize, format: PcmFormat) -> Option<Arc<dyn HostAudioStream>> {
        let mut w = lock(&self.0);
        if w.fail_open {
            return None;
        }
        let state = Arc::new(Mutex::new(StreamState {
            freq: rate,
            frame_bytes: channels * format.bytes_per_sample(),
            frames: 0,
            updated: w.now,
            pcm: Vec::new(),
        }));
        w.streams.push(state.clone());
        Some(Arc::new(FakeStream { world: self.0.clone(), state }))
    }
}

struct Fixture {
    audio: Audio<Fake>,
    world: Arc<Mutex<World>>,
    pcm: Vec<u8>,
}

impl Fixture {
    fn new() -> Self {
        let world = Arc::new(Mutex::new(World { now: 1_000_000, ..World::default() }));
        Self { audio: Audio::new(Fake(world.clone())), world, pcm: vec![0u8; 2048 * 4] }
    }

    fn w(&self) -> std::sync::MutexGuard<'_, World> {
        lock(&self.world)
    }

    fn now(&self) -> u64 {
        self.w().now
    }

    fn open(&self, frames: u32, port_type: i32) -> i32 {
        self.audio.open(port_type, frames, 48000, Format::FloatStereo).expect("port open failed")
    }

    fn output(&self, port: i32, blocking: bool) {
        self.audio.outputs(&[OutputParam { handle: port, data: Some(&self.pcm), gains: None }], blocking);
    }

    fn batch(&self, ports: &[(i32, bool)]) {
        let params: Vec<OutputParam> = ports
            .iter()
            .map(|&(handle, has_pcm)| OutputParam { handle, data: has_pcm.then_some(&self.pcm[..]), gains: None })
            .collect();
        self.audio.outputs(&params, true);
    }

    fn front_frames(&self) -> u64 {
        let w = self.w();
        let mut s = lock(&w.streams[0]);
        drain(&w, &mut s);
        s.frames
    }

    fn last_pcm(&self) -> Vec<u8> {
        let w = self.w();
        let pcm = lock(w.streams.last().unwrap()).pcm.clone();
        pcm
    }

    fn prime(&self, port: i32) {
        let start = self.now();
        for _ in 0..8 {
            self.output(port, true);
        }
        assert_eq!(self.now(), start, "priming was paced before the 40 ms cushion filled");
        self.output(port, true);
        let el = self.now() - start;
        assert!((5333..=5334).contains(&el), "priming accumulated a multi-block deadline ({el})");
        assert!(self.front_frames() >= 7 * 256 * 1_000_000, "first paced output consumed the priming cushion");
    }

    fn finish(self) {
        let world = self.world.clone();
        drop(self);
        assert!(lock(&world).streams.is_empty(), "output stream leaked");
    }
}

#[test]
fn priming_and_steady_cadence() {
    let f = Fixture::new();
    let port = f.open(256, 0);
    f.prime(port);
    {
        let mut w = f.w();
        w.processing = 125;
        w.oversleep = 175;
    }
    for _ in 0..10 {
        f.output(port, true);
    }
    let start = f.now();
    for _ in 0..3000 {
        f.output(port, true);
    }
    let el = f.now() - start;
    assert!(
        (15_999_999..=16_000_001).contains(&el),
        "processing, sleep overshoot, or fractional periods accumulated timing drift ({el})"
    );
    f.finish();
}

#[test]
fn underrun_and_stalled_queue_recovery() {
    let f = Fixture::new();
    let port = f.open(256, 0);
    f.prime(port);
    f.w().now += 1_000_000;
    f.prime(port);

    // A stopped device clears its stale queue and restarts priming instead of catching up.
    f.w().stalled = true;
    for _ in 0..3 {
        if f.w().clears != 0 {
            break;
        }
        f.output(port, true);
    }
    assert_eq!(f.w().clears, 1, "stalled playback queue was not cleared");
    f.w().stalled = false;
    let start = f.now();
    for _ in 0..7 {
        f.output(port, true);
    }
    assert_eq!(f.now(), start, "queue clear did not restart unpaced priming");
    f.output(port, true);
    let el = f.now() - start;
    assert!((5333..=5334).contains(&el), "queue clear retained an old pacing deadline ({el})");
    f.finish();
}

#[test]
fn fallback_clock_and_long_gap() {
    let f = Fixture::new();
    f.w().fail_open = true;
    let port = f.open(256, 0);
    f.output(port, true);
    let start = f.now();
    f.w().oversleep = 175;
    for _ in 0..3000 {
        f.w().now += 125;
        f.output(port, true);
    }
    let el = f.now() - start;
    assert!((16_000_174..=16_000_176).contains(&el), "device-free output accumulated timing drift ({el})");
    f.w().now += 1_000_000;
    f.output(port, true);
    let resumed = f.now();
    f.output(port, true);
    let el = f.now() - resumed;
    assert!((5333..=5509).contains(&el), "long interruption caused a burst of catch-up submissions ({el})");
    f.finish();
}

#[test]
fn async_does_not_accumulate_deadlines() {
    let f = Fixture::new();
    let port = f.open(256, 0);
    let start = f.now();
    for _ in 0..8 {
        f.output(port, false);
    }
    assert_eq!(f.now(), start, "asynchronous output slept");
    f.output(port, true);
    assert!(f.now() - start <= 5334, "asynchronous submissions accumulated a blocking deadline");
    for _ in 0..3 {
        f.output(port, true);
    }
    f.w().now += 1_000_000;
    f.output(port, false);
    let resumed = f.now();
    for _ in 0..7 {
        f.output(port, true);
    }
    assert_eq!(f.now(), resumed, "async-to-blocking transition did not reset an emptied queue");
    f.finish();
}

#[test]
fn synchronized_batch_and_inactive_ports() {
    let f = Fixture::new();
    let vibration = f.open(256, PORT_TYPE_VIBRATION);
    let main = f.open(256, 0);
    let background = f.open(256, 0);
    let batch = [(vibration, true), (main, true), (background, true)];
    let start = f.now();
    for _ in 0..8 {
        f.batch(&batch);
    }
    assert_eq!(f.now(), start, "vibration fallback slowed a device-backed batch's priming");
    for _ in 0..3 {
        f.w().sleeps.clear();
        f.batch(&batch);
        let sleeps = f.w().sleeps.clone();
        assert!(
            sleeps.len() == 1 && (5333..=5334).contains(&sleeps[0]),
            "synchronized streams accumulated separate pacing waits ({sleeps:?})"
        );
    }

    let inactive = [(main, false), (vibration, true)];
    f.w().now += 1_000_000;
    f.batch(&inactive);
    let resumed = f.now();
    f.batch(&inactive);
    let el = f.now() - resumed;
    assert!((5333..=5334).contains(&el), "a device without PCM disabled fallback pacing ({el})");
    f.finish();
}

#[test]
fn fallback_uses_each_ports_period() {
    let f = Fixture::new();
    f.w().fail_open = true;
    let short_port = f.open(128, 0);
    let long_port = f.open(256, 0);
    let batch = [(short_port, true), (long_port, true)];
    f.batch(&batch);
    let start = f.now();
    f.batch(&batch);
    let el = f.now() - start;
    assert!((5333..=5334).contains(&el), "fallback batch used the first port's period for every port ({el})");
    f.finish();
}

#[test]
fn failed_queue_uses_fallback_clock() {
    let f = Fixture::new();
    let port = f.open(256, 0);
    f.w().fail_put = true;
    f.output(port, true);
    let start = f.now();
    for _ in 0..3 {
        f.output(port, true);
    }
    assert_eq!(f.now() - start, 16000, "repeated queue failures reset fallback pacing");
    f.finish();
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn i16s(v: &[i16]) -> Vec<u8> {
    v.iter().flat_map(|s| s.to_le_bytes()).collect()
}

#[test]
fn channel_gains() {
    let f = Fixture::new();
    let pcm = f32s(&[0.25, -0.5, 0.75, -1.0]);
    let stereo = f.audio.open(0, 2, 48000, Format::FloatStereo).unwrap();
    let mut gains = [2.0f32, 0.5];
    let out = |handle: i32, data: &[u8], gains: Option<&[f32]>| {
        f.audio.outputs(&[OutputParam { handle, data: Some(data), gains }], false);
    };
    for _ in 0..2 {
        out(stereo, &pcm, Some(&gains));
        assert_eq!(f.last_pcm(), f32s(&[0.5, -0.25, 1.5, -0.5]), "float channel gains were not applied independently");
    }
    assert!(f.audio.set_volume(stereo, 3, &[16384, 32768]));
    out(stereo, &pcm, Some(&gains));
    assert_eq!(f.last_pcm(), f32s(&[0.25, -0.25, 0.75, -0.5]), "channel gain did not combine with port volume");
    gains = [0.0, 0.0];
    out(stereo, &pcm, Some(&gains));
    assert_eq!(f.last_pcm(), f32s(&[0.0, -0.0, 0.0, -0.0]), "zero channel gains did not mute");
    out(stereo, &pcm, None);
    assert_eq!(f.last_pcm(), f32s(&[0.125, -0.5, 0.375, -1.0]), "channel scaling changed original float PCM");

    let integer_pcm = i16s(&[20000, -20000]);
    let integer = f.audio.open(0, 1, 48000, Format::S16Stereo).unwrap();
    for gain in [2.0f32, f32::MAX] {
        out(integer, &integer_pcm, Some(&[gain, gain]));
        assert_eq!(f.last_pcm(), i16s(&[32767, -32768]), "integer gain overflowed instead of saturating");
    }
    out(integer, &integer_pcm, Some(&[0.5, 0.25]));
    assert_eq!(f.last_pcm(), i16s(&[10000, -5000]), "integer channel gains changed original PCM");

    let surround: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let surround_pcm = f32s(&surround);
    for format in [Format::Float8Ch, Format::Float12Ch] {
        let port = f.audio.open(0, 1, 48000, format).unwrap();
        out(port, &surround_pcm, Some(&surround));
        if format == Format::Float8Ch {
            assert_eq!(
                f.last_pcm(),
                f32s(&[1.0, 4.0, 9.0, 16.0, 49.0, 64.0, 25.0, 36.0]),
                "channel reordering used destination channel gains"
            );
        } else {
            assert_eq!(
                f.last_pcm(),
                f32s(&[82.0, 104.0, 9.0, 16.0, 170.0, 208.0, 25.0, 36.0]),
                "height downmix lost per-channel gains"
            );
        }
    }
    f.finish();
}

#[test]
fn zero_output_frequency() {
    let vm = crate::sync::test_util::vm_arena(BASE);
    let mut ctx = CpuContext::new(vm);
    // sceAudioOutOpen(user 0, type 0, index 0, 256 frames, 0 Hz, float stereo)
    ctx.set(CpuRegister::Rsi, 0);
    ctx.set(CpuRegister::Rcx, 256);
    ctx.set(CpuRegister::R8, 0);
    ctx.set(CpuRegister::R9, 4);
    assert_eq!(cb_open(&mut ctx), AUDIO_OUT_ERROR_INVALID_SAMPLE_FREQ, "public output open accepted zero frequency");
    let f = Fixture::new();
    assert!(f.audio.open(10, 256, 0, Format::FloatStereo).is_none(), "internal output open accepted zero frequency");
    assert_eq!(f.open(256, 0), 1, "rejected output port occupied a handle");
    f.finish();
}

#[test]
fn invalid_batch_size() {
    let vm = crate::sync::test_util::vm_arena(BASE + 0x10_0000);
    let mut ctx = CpuContext::new(vm);
    ctx.set(CpuRegister::Rdi, BASE + 0x10_0000);
    for count in [0u64, 33] {
        ctx.set(CpuRegister::Rsi, count);
        assert_eq!(cb_outputs(&mut ctx), AUDIO_OUT_ERROR_INVALID_SIZE, "invalid batch size was not rejected");
    }
}

#[test]
fn public_port_lifecycle() {
    let vm = crate::sync::test_util::vm_arena(BASE + 0x20_0000);
    let base = BASE + 0x20_0000;
    let mut ctx = CpuContext::new(vm);
    ctx.set(CpuRegister::Rsi, 77);
    ctx.set(CpuRegister::R8, 48_000);
    assert_eq!(cb_open(&mut ctx), AUDIO_OUT_ERROR_INVALID_PORT_TYPE);
    ctx.set(CpuRegister::Rsi, 0);
    ctx.set(CpuRegister::R9, 9);
    assert_eq!(cb_open(&mut ctx), AUDIO_OUT_ERROR_INVALID_FORMAT);
    ctx.set(CpuRegister::Rcx, 256);
    ctx.set(CpuRegister::R9, 1); // stereo s16
    let h = cb_open(&mut ctx);
    assert!(h > 0);
    ctx.set(CpuRegister::Rdi, h as u64);
    ctx.set(CpuRegister::Rsi, base);
    assert_eq!(cb_get_port_state(&mut ctx), 0);
    assert_eq!(&ctx.read_bytes(base, 6).unwrap(), &[1, 0, 2, 0, 127, 0]);
    ctx.write_u32(base + 0x100, 16384);
    ctx.set(CpuRegister::Rsi, 1);
    ctx.set(CpuRegister::Rdx, base + 0x100);
    assert_eq!(cb_set_volume(&mut ctx), 0);
    // No host device under test: the sample clock paces 10 buffers of 5.33 ms.
    ctx.set(CpuRegister::Rsi, base + 0x1000);
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        assert_eq!(cb_output(&mut ctx), 256);
    }
    assert!(t0.elapsed() >= Duration::from_millis(45), "{:?}", t0.elapsed());
    ctx.write_u64(base + 0x200, h as u64);
    ctx.write_u64(base + 0x208, base + 0x1000);
    ctx.set(CpuRegister::Rdi, base + 0x200);
    ctx.set(CpuRegister::Rsi, 1);
    assert_eq!(cb_outputs(&mut ctx), 256);
    ctx.set(CpuRegister::Rdi, h as u64);
    assert_eq!(cb_close(&mut ctx), 0);
    assert_eq!(cb_close(&mut ctx), AUDIO_OUT_ERROR_INVALID_PORT);
    ctx.set(CpuRegister::Rsi, base + 0x1000);
    assert_eq!(cb_output(&mut ctx), AUDIO_OUT_ERROR_INVALID_PORT);
}

/// Controller stream that takes blocks while `taken`.
struct FakePad {
    taken: Arc<Mutex<bool>>,
    blocks: Arc<Mutex<Vec<(Vec<f32>, f32)>>>,
}

impl crate::pad_audio::PadAudioStream for FakePad {
    fn queue(&mut self, b: &crate::pad_audio::PadAudioBlock) -> u64 {
        if !*lock(&self.taken) {
            return 0;
        }
        let pcm = (0..(b.frames * b.channels) as usize).map(|i| b.sample(i)).collect();
        lock(&self.blocks).push((pcm, b.gain));
        1000
    }

    fn uses_bluetooth(&self) -> bool {
        false
    }
}

#[test]
fn pad_speaker_routing_and_fallback() {
    let f = Fixture::new();
    let pcm = f32s(&[0.5, 0.5, 0.5, 0.5]);
    let port = f.audio.open(PORT_TYPE_PADSPK, 2, 48000, Format::FloatStereo).unwrap();
    let taken = Arc::new(Mutex::new(true));
    let blocks = Arc::new(Mutex::new(Vec::new()));
    f.audio.ports[slot_index(port).unwrap()].lock().unwrap().pad_stream =
        Some(Box::new(FakePad { taken: taken.clone(), blocks: blocks.clone() }));
    let streams_before = f.w().streams.len();
    let pcm_before = f.last_pcm();
    f.audio.outputs(&[OutputParam { handle: port, data: Some(&pcm), gains: None }], false);
    assert_eq!(lock(&blocks).len(), 1, "controller did not take the pad speaker block");
    assert_eq!(lock(&blocks)[0].0, vec![0.5; 4]);
    assert_eq!(f.w().streams.len(), streams_before);
    assert_eq!(f.last_pcm(), pcm_before, "controller playback also queued on the main output");
    // Unplugged: the main output plays the block instead.
    *lock(&taken) = false;
    f.audio.outputs(&[OutputParam { handle: port, data: Some(&pcm), gains: None }], false);
    assert_eq!(f.last_pcm(), pcm, "pad speaker did not fall back to the main output");
    f.audio.close(port);
    f.finish();
}

/// Without a controller, a vibration port is never mixed into the speakers
/// (its waveform drives the actuators); a pad speaker port still plays on
/// the main output.
#[test]
fn pad_ports_without_pad_device() {
    let f = Fixture::new();
    let vibration = f.audio.open(PORT_TYPE_VIBRATION, 2, 48000, Format::FloatStereo).unwrap();
    let speaker = f.audio.open(PORT_TYPE_PADSPK, 2, 48000, Format::FloatMono).unwrap();
    for port in [vibration, speaker] {
        f.audio.ports[slot_index(port).unwrap()].lock().unwrap().pad_stream = None;
    }
    assert_eq!(f.w().streams.len(), 1, "a vibration port opened a speaker stream");
    assert!(!f.audio.has_device(vibration));
    let rumble = f32s(&[0.75; 4]);
    let voice = f32s(&[0.25, -0.25]);
    let batch = [
        OutputParam { handle: vibration, data: Some(&rumble), gains: None },
        OutputParam { handle: speaker, data: Some(&voice), gains: None },
    ];
    f.audio.outputs(&batch, false);
    assert_eq!(f.last_pcm(), voice, "only the pad speaker belongs in the main mix");
    f.audio.outputs(&batch[..1], false);
    assert_eq!(f.last_pcm(), voice, "a vibration port alone reached the speakers");
    f.audio.close(vibration);
    f.audio.close(speaker);
    f.finish();
}

/// sceAudioOutSetMixLevelPadSpk: checks in the order handle, port type,
/// range. The level is a linear gain on the pad speaker port, on the main
/// output and on the controller block alike.
#[test]
fn pad_speaker_mix_level() {
    let f = Fixture::new();
    let main = f.open(256, PORT_TYPE_MAIN);
    let pad = f.open(256, PORT_TYPE_PADSPK);
    assert_eq!(f.audio.set_mix_level_pad_spk(0, UNITY_VOLUME), AUDIO_OUT_ERROR_INVALID_PORT);
    assert_eq!(f.audio.set_mix_level_pad_spk(pad + 1, UNITY_VOLUME), AUDIO_OUT_ERROR_INVALID_PORT);
    assert_eq!(f.audio.set_mix_level_pad_spk(main, UNITY_VOLUME), AUDIO_OUT_ERROR_INVALID_PORT_TYPE);
    assert_eq!(
        f.audio.set_mix_level_pad_spk(main, UNITY_VOLUME + 1),
        AUDIO_OUT_ERROR_INVALID_PORT_TYPE,
        "port type was checked after the level"
    );
    assert_eq!(f.audio.set_mix_level_pad_spk(pad, UNITY_VOLUME + 1), AUDIO_OUT_ERROR_INVALID_MIX_LEVEL);
    assert_eq!(f.audio.set_mix_level_pad_spk(pad, -1), AUDIO_OUT_ERROR_INVALID_MIX_LEVEL);

    // One buffer of 256 stereo float frames, the port's buffer size.
    let samples = 256 * 2;
    let pcm = f32s(&vec![0.5f32; samples]);
    let play = |level: i32| {
        assert_eq!(f.audio.set_mix_level_pad_spk(pad, level), 0);
        f.audio.outputs(&[OutputParam { handle: pad, data: Some(&pcm), gains: None }], false);
        f.last_pcm()
    };
    assert_eq!(play(UNITY_VOLUME), f32s(&vec![0.5f32; samples]), "unity mix level changed the output");
    assert_eq!(play(UNITY_VOLUME / 2), f32s(&vec![0.25f32; samples]), "mix level did not scale the main output");
    assert_eq!(play(0), f32s(&vec![0.0f32; samples]), "zero mix level did not mute the main output");

    let taken = Arc::new(Mutex::new(true));
    let blocks = Arc::new(Mutex::new(Vec::new()));
    f.audio.ports[slot_index(pad).unwrap()].lock().unwrap().pad_stream =
        Some(Box::new(FakePad { taken, blocks: blocks.clone() }));
    let base = crate::pad_audio::port_gain(PORT_TYPE_PADSPK);
    assert_eq!(f.audio.set_mix_level_pad_spk(pad, UNITY_VOLUME / 4), 0);
    f.audio.outputs(&[OutputParam { handle: pad, data: Some(&pcm), gains: None }], false);
    assert_eq!(lock(&blocks).last().map(|b| b.1), Some(base * 0.25), "controller block missed the mix level");
    f.audio.close(main);
    f.audio.close(pad);
    f.finish();
}

#[test]
fn set_mix_level_pad_spk_is_registered() {
    assert_eq!(ps5nid::compute_nid("sceAudioOutSetMixLevelPadSpk"), "wVwPU50pS1c");
    let reg = crate::registry::HleRegistry::with_defaults();
    assert!(reg.is_implemented("wVwPU50pS1c"), "sceAudioOutSetMixLevelPadSpk is still the catalog stub");
}

struct Gate {
    /// 0 idle, 1 an output sleeps, 2 released.
    state: Mutex<u8>,
    cv: std::sync::Condvar,
}

/// [`Fake`] whose first pacing sleep waits for the test to release it.
struct GatedEnv {
    fake: Fake,
    gate: Arc<Gate>,
}

impl Env for GatedEnv {
    fn now_us(&self) -> u64 {
        self.fake.now_us()
    }

    fn sleep_us(&self, us: u64) {
        let mut s = lock(&self.gate.state);
        if *s != 2 {
            *s = 1;
            self.gate.cv.notify_all();
            while *s != 2 {
                s = self.gate.cv.wait(s).unwrap_or_else(|e| e.into_inner());
            }
        }
        drop(s);
        self.fake.sleep_us(us);
    }

    fn open_stream(&self, rate: u32, channels: usize, format: PcmFormat) -> Option<Arc<dyn HostAudioStream>> {
        self.fake.open_stream(rate, channels, format)
    }
}

/// A blocking output holds no port lock while it is paced: state queries,
/// volume changes and close go through meanwhile.
#[test]
fn queries_during_output_wait() {
    let world = Arc::new(Mutex::new(World { now: 1_000_000, fail_open: true, ..World::default() }));
    let gate = Arc::new(Gate { state: Mutex::new(0), cv: std::sync::Condvar::new() });
    let audio = Audio::new(GatedEnv { fake: Fake(world), gate: gate.clone() });
    let handle = audio.open(PORT_TYPE_MAIN, 256, 48000, Format::FloatStereo).unwrap();
    let pcm = vec![0u8; 256 * 2 * 4];
    let audio = &audio;
    std::thread::scope(|s| {
        let output = s.spawn(|| {
            for _ in 0..2 {
                assert_eq!(audio.outputs(&[OutputParam { handle, data: Some(&pcm), gains: None }], true), 256);
            }
        });
        {
            let mut st = lock(&gate.state);
            while *st != 1 {
                st = gate.cv.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        }
        let (tx, rx) = std::sync::mpsc::channel();
        s.spawn(move || {
            let r = (
                audio.valid(handle),
                audio.status(handle).map(|s| s.2),
                audio.set_volume(handle, 1, &[16384]),
                audio.close(handle),
            );
            let _ = tx.send(r);
        });
        let r = rx.recv_timeout(Duration::from_secs(5));
        *lock(&gate.state) = 2;
        gate.cv.notify_all();
        assert_eq!(r.expect("port queries waited for output pacing"), (true, Some(256 * 2 * 4), true, true));
        output.join().unwrap();
    });
    assert!(!audio.valid(handle), "close during an output wait did not take effect");
}

/// The last output time is when the last buffer with PCM was submitted.
#[test]
fn last_output_time() {
    let f = Fixture::new();
    let port = f.open(256, 0);
    assert_eq!(f.audio.last_output_time(port), Some(0), "nothing submitted yet");
    f.output(port, false);
    let t = f.now();
    assert_eq!(f.audio.last_output_time(port), Some(t));
    f.w().now += 1000;
    f.audio.outputs(&[OutputParam { handle: port, data: None, gains: None }], false);
    assert_eq!(f.audio.last_output_time(port), Some(t), "a drain does not count");
    f.batch(&[(port, true)]);
    assert_eq!(f.audio.last_output_time(port), Some(f.now()), "after pacing");
    assert_eq!(f.audio.last_output_time(port + 1), None);
    f.audio.close(port);
    f.finish();
}

#[test]
fn ex_ports_and_monitor_info() {
    let base = BASE + 0x30_0000;
    let vm = crate::sync::test_util::vm_arena(base);
    let mut ctx = CpuContext::new(vm);
    let call = |ctx: &mut CpuContext, f: fn(&mut CpuContext) -> HleResult, args: &[u64]| {
        let regs = [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx, CpuRegister::R8];
        for (r, v) in regs.iter().zip(args) {
            ctx.set(*r, *v);
        }
        f(ctx)
    };
    for mode in [5, 7, 11] {
        assert_eq!(call(&mut ctx, cb_ex_open, &[0xFF, mode]), AUDIO_OUT_ERROR_EX_INVALID_MODE, "mode {mode}");
    }
    let ac3 = call(&mut ctx, cb_ex_open, &[0xFF, 0]);
    assert!(ac3 > 0);
    let eac3 = call(&mut ctx, cb_ex_open, &[0xFF, 4]);
    assert!(eac3 > 0 && eac3 != ac3);
    assert_eq!(AUDIO.status(ac3 as i32).map(|s| s.3), Some(256));
    assert_eq!(AUDIO.status(eac3 as i32).map(|s| s.3), Some(1024));
    assert!(!AUDIO.has_device(ac3 as i32), "bitstream ports have no host stream");
    // A burst is taken (paced by the sample clock) and recorded.
    let before = crate::time::process_time_usec();
    assert_eq!(call(&mut ctx, cb_output, &[ac3 as u64, base + 0x1000]), 256);
    assert_eq!(call(&mut ctx, cb_get_last_output_time, &[ac3 as u64, 0]), AUDIO_OUT_ERROR_INVALID_POINTER);
    assert_eq!(call(&mut ctx, cb_get_last_output_time, &[ac3 as u64, base]), 0);
    let at = ctx.read_u64(base).unwrap();
    assert!(at >= before && at <= crate::time::process_time_usec(), "{before} <= {at}");
    assert_eq!(call(&mut ctx, cb_get_last_output_time, &[0, base]), AUDIO_OUT_ERROR_INVALID_PORT);

    assert_eq!(call(&mut ctx, cb_ex_configure_output, &[1, 0, 0, 1, 0]), AUDIO_OUT_ERROR_EX_INVALID_CONTROL);
    for (mode, target) in [(0, 1), (0, 0xFF), (0xFF, 0xFF), (10, 3)] {
        assert_eq!(call(&mut ctx, cb_ex_configure_output, &[0, 0, mode, target, 0]), 0, "mode {mode} target {target}");
    }
    assert_eq!(call(&mut ctx, cb_ex_configure_output, &[0, 0, 11, 1, 0]), ERROR_NOT_IMPLEMENTED);
    assert_eq!(call(&mut ctx, cb_ex_configure_output, &[0, 0, 0, 4, 0]), ERROR_NOT_IMPLEMENTED);
    assert_eq!(call(&mut ctx, cb_ex_configure_output, &[0, 1, 0, 1, 0]), ERROR_NOT_IMPLEMENTED);

    // System state: zero loudness and reserved words, exactly 32 bytes.
    ctx.write_bytes(base + 0x2000, &[0xEE; 0x40]);
    assert_eq!(call(&mut ctx, cb_get_system_state, &[base + 0x2000]), 0);
    assert!(ctx.read_bytes(base + 0x2000, 0x20).unwrap().iter().all(|&b| b == 0));
    assert_eq!(ctx.read_bytes(base + 0x2020, 1).unwrap(), [0xEE]);
    assert_eq!(call(&mut ctx, cb_get_system_state, &[0]), AUDIO_OUT_ERROR_INVALID_POINTER);

    // Monitor information: no audio descriptors.
    ctx.write_bytes(base + 0x2000, &[0xEE; 0x200]);
    assert_eq!(call(&mut ctx, cb_ex_get_monitor_info, &[0, 0, base + 0x2000, 0xA0]), 0);
    assert!(ctx.read_bytes(base + 0x2000, 0xA0).unwrap().iter().all(|&b| b == 0));
    assert_eq!(ctx.read_bytes(base + 0x20A0, 1).unwrap(), [0xEE]);
    assert_eq!(call(&mut ctx, cb_ex_get_monitor_info, &[0, 0, 0, 0xA0]), AUDIO_OUT_ERROR_INVALID_POINTER);
    assert_eq!(call(&mut ctx, cb_ex_get_monitor_info, &[0, 0, base + 0x2000, 0x9F]), ERROR_NOT_IMPLEMENTED);
    assert_eq!(call(&mut ctx, cb_ex_get_monitor_info, &[1, 0, base + 0x2000, 0xA0]), ERROR_NOT_IMPLEMENTED);
    ctx.write_bytes(base + 0x2000, &[0xEE; 0x200]);
    assert_eq!(call(&mut ctx, cb_sys_get_hdmi_monitor_info, &[1, base + 0x2000, 0x180]), 0);
    assert!(ctx.read_bytes(base + 0x2000, 0x180).unwrap().iter().all(|&b| b == 0));
    assert_eq!(call(&mut ctx, cb_sys_get_hdmi_monitor_info, &[1, 0, 0x180]), AUDIO_OUT_ERROR_INVALID_POINTER);
    assert_eq!(call(&mut ctx, cb_sys_get_hdmi_monitor_info, &[0, base + 0x2000, 0x180]), ERROR_NOT_IMPLEMENTED);
    assert_eq!(call(&mut ctx, cb_sys_get_hdmi_monitor_info, &[1, base + 0x2000, 0x100]), ERROR_NOT_IMPLEMENTED);

    assert_eq!(call(&mut ctx, cb_ex_close, &[ac3 as u64]), 0);
    assert_eq!(call(&mut ctx, cb_ex_close, &[ac3 as u64]), AUDIO_OUT_ERROR_INVALID_PORT, "closed");
    let main = AUDIO.open(PORT_TYPE_MAIN, 256, 48000, Format::S16Stereo).unwrap();
    assert_eq!(call(&mut ctx, cb_ex_close, &[main as u64]), AUDIO_OUT_ERROR_INVALID_PORT, "not an Ex port");
    assert!(AUDIO.close(main));
    assert_eq!(call(&mut ctx, cb_ex_close, &[eac3 as u64]), 0);
}

/// A vibration block no actuator played becomes rumble levels: left and
/// right channel RMS with the port volume, gains and setting gain.
#[test]
fn vibration_block_rumble_levels() {
    let mut port = PortOut {
        used: true,
        port_type: PORT_TYPE_VIBRATION,
        samples_num: 4,
        freq: 48000,
        format: Some(Format::S16Stereo),
        channels_num: 2,
        volume: [UNITY_VOLUME; CHANNELS_MAX],
        ..PortOut::default()
    };
    let pcm: Vec<u8> = [8192i16, 0, -8192, 0, 8192, 0, -8192, 0].iter().flat_map(|s| s.to_le_bytes()).collect();
    let param = OutputParam { handle: 1, data: Some(&pcm), gains: None };
    assert_eq!(vibration_levels(&port, &param, 1.0), Some((128, 0)), "0.25 RMS on the left");
    assert_eq!(vibration_levels(&port, &param, 2.0), Some((255, 0)), "setting gain");
    port.volume[0] = UNITY_VOLUME / 2;
    assert_eq!(vibration_levels(&port, &param, 1.0), Some((64, 0)), "port volume");
    let gains = [2.0f32, 1.0];
    let with_gains = OutputParam { gains: Some(&gains), ..param };
    assert_eq!(vibration_levels(&port, &with_gains, 1.0), Some((128, 0)), "AudioOut2 gains");
    assert_eq!(vibration_levels(&port, &OutputParam { data: None, ..param }, 1.0), None);
    assert_eq!(vibration_levels(&port, &OutputParam { data: Some(&pcm[..8]), ..param }, 1.0), None, "short block");
    // A mono port drives both motors.
    port.format = Some(Format::S16Mono);
    port.channels_num = 1;
    port.volume[0] = UNITY_VOLUME;
    assert_eq!(vibration_levels(&port, &OutputParam { data: Some(&pcm[..8]), ..param }, 1.0), Some((90, 90)));
}

/// sceAudioOutSetMainOutput is exported and answers 0 without arguments.
#[test]
fn set_main_output_answers_zero() {
    let entry = EXPORTS.iter().find(|e| e.1 == "sceAudioOutSetMainOutput").expect("exported");
    assert_eq!(entry.0, ps5nid::compute_nid(entry.1));
    let vm = ps5vm::VirtualMemory::new();
    let mut ctx = CpuContext::new(&vm);
    assert_eq!((entry.3)(&mut ctx), 0);
}
