// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Sampler controls and rendering exercised directly on the engine, with
//! host memory standing in for guest memory (addresses are host pointers).

use super::engine::*;
use super::geom;
use crate::audio::atrac9::Atrac9Config;
use std::cell::RefCell;
use std::rc::Rc;

type Callback = Box<dyn FnMut(&mut Engine, &CallbackInfo)>;

#[derive(Default)]
struct HostEnv {
    callback: Option<Callback>,
}

impl Env for HostEnv {
    fn read(&mut self, addr: u64, out: &mut [u8]) -> bool {
        if addr == 0 {
            return false;
        }
        // SAFETY: tests only pass addresses of live host buffers.
        unsafe { std::ptr::copy_nonoverlapping(addr as *const u8, out.as_mut_ptr(), out.len()) };
        true
    }

    fn write(&mut self, addr: u64, data: &[u8]) -> bool {
        if addr == 0 {
            return false;
        }
        // SAFETY: as above.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), addr as *mut u8, data.len()) };
        true
    }

    fn call(&mut self, _func: u64, _args: [u64; 6]) -> u64 {
        0
    }

    fn voice_callback(&mut self, eng: &mut Engine, _func: u64, info: &CallbackInfo) {
        if let Some(cb) = self.callback.as_mut() {
            cb(eng, info);
        }
    }
}

const SYSTEM: u64 = 0x1000_0000;
const ATRAC9_FORMAT: WaveformFormat = WaveformFormat::new(0x40, 1, 24000, 0xfe40_05f0);

fn addr<T>(p: *const T) -> u64 {
    p as u64
}

/// Parameter image: {u16 size, i16 next = 0, u32 id} + body.
fn param(id: u32, body: &[u8]) -> Vec<u8> {
    let mut p = Vec::with_capacity(8 + body.len());
    p.extend_from_slice(&((8 + body.len()) as u16).to_le_bytes());
    p.extend_from_slice(&0i16.to_le_bytes());
    p.extend_from_slice(&id.to_le_bytes());
    p.extend_from_slice(body);
    p
}

fn format_bytes(f: &WaveformFormat) -> Vec<u8> {
    [f.waveform_type, f.num_channels, f.sample_rate, f.config_data, f.frame_offset, f.frame_margin]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

struct Fixture {
    eng: Engine,
    env: HostEnv,
    vid: Vid,
    next_rack: u64,
}

impl Fixture {
    fn new(rate: u32, channels: u32) -> Self {
        Self::with_type(rate, channels, RACK_SAMPLER)
    }

    fn with_type(rate: u32, channels: u32, rack_id: u32) -> Self {
        let mut eng = Engine::new();
        let option = SystemOption::default();
        eng.system_create(SYSTEM, option, ContextBuffer { host_buffer: SYSTEM, ..Default::default() }, Allocator::default());
        let mut f = Self { eng, env: HostEnv::default(), vid: Vid { slot: 0, idx: 0 }, next_rack: 0x2000_0000 };
        f.vid = f.add_rack(rack_id, 1, 0, 0, 0);
        let mut body = format_bytes(&WaveformFormat::new(0x12, channels, rate, 0));
        body.extend_from_slice(&[0u8; 8]);
        let id = if rack_id == RACK_SAMPLER { 0x1000_0000 } else { 0x4001_0000 };
        assert_eq!(f.control(&param(id, &body)), OK, "sampler setup failed");
        f.v().state = PlayState::Playing;
        f
    }

    fn add_rack(&mut self, rack_id: u32, voices: u32, ports: u32, matrices: u32, max_channels: u32) -> Vid {
        let mut option = RackOption::from_bytes(&[]);
        option.set_max_voices(voices);
        option.set_max_ports(ports);
        option.bytes[88..92].copy_from_slice(&matrices.to_le_bytes());
        if rack_id == RACK_SAMPLER {
            option.set_sampler_max_filters(0);
        } else {
            option.set_max_channels(max_channels);
        }
        let handle = self.next_rack;
        self.next_rack += 0x10_0000;
        let slot = self
            .eng
            .rack_create(&mut self.env, SYSTEM, rack_id, option, ContextBuffer { host_buffer: handle, size: 0x1000, user_data: 0 }, handle)
            .unwrap();
        Vid { slot, idx: 0 }
    }

    fn mastering(&mut self, channels: u32) -> Vid {
        let m = self.add_rack(RACK_MASTERING, 1, 0, 0, 8);
        let v = self.eng.voice_mut(m).unwrap();
        v.channels = channels;
        v.state = PlayState::Playing;
        m
    }

    fn v(&mut self) -> &mut Voice {
        self.eng.voice_mut(self.vid).unwrap()
    }

    fn voice(&self) -> &Voice {
        self.eng.voice(self.vid).unwrap()
    }

    fn handle(&self, v: Vid) -> u64 {
        self.eng.voice_handle(v)
    }

    fn control(&mut self, p: &[u8]) -> i32 {
        self.control_on(self.vid, p)
    }

    fn control_on(&mut self, vid: Vid, p: &[u8]) -> i32 {
        self.eng.voice_control(&mut self.env, vid, addr(p.as_ptr()))
    }

    fn queue(&mut self, data: *const i16, frames: u32, flags: u32, repeats: u32, skip: u32) {
        let channels = self.voice().channels as u64;
        let block = WaveformBlock {
            data_offset: 0,
            data_size: (frames + skip) as u64 * channels * 2,
            num_repeats: repeats,
            num_skip_samples: skip,
            num_samples: frames,
            reserved: 0,
            user_data: 123,
        };
        self.queue_raw(addr(data), flags, &[block]);
    }

    fn queue_raw(&mut self, data: u64, flags: u32, blocks: &[WaveformBlock]) {
        let images: Vec<u8> = blocks.iter().flat_map(|b| b.to_bytes()).collect();
        let mut body = Vec::new();
        body.extend_from_slice(&data.to_le_bytes());
        body.extend_from_slice(&flags.to_le_bytes());
        body.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
        body.extend_from_slice(&addr(images.as_ptr()).to_le_bytes());
        let id = if self.eng.rack(self.vid.slot).unwrap().ty == RackType::Sampler { 0x1000_0001 } else { 0x4001_0001 };
        assert_eq!(self.control(&param(id, &body)), OK, "sampler queue failed");
    }

    fn render(&mut self, frames: u32) {
        self.v().rendered = false;
        self.eng.render_voice_from(&mut self.env, self.vid, &[], frames);
    }

    fn render_target(&mut self, target: Vid, sources: &[Vid], frames: u32) {
        self.eng.render_voice_from(&mut self.env, target, sources, frames);
    }

    fn reset_rendered(&mut self, voices: &[Vid]) {
        for &v in voices {
            self.eng.voice_mut(v).unwrap().rendered = false;
        }
    }

    fn pitch(&mut self, id: u32, ratio: f32) {
        let mut body = ratio.to_le_bytes().to_vec();
        body.extend_from_slice(&[0u8; 4]);
        assert_eq!(self.control(&param(id, &body)), OK);
    }

    fn event(&mut self, event: u32) {
        assert_eq!(self.control(&param(6, &event.to_le_bytes())), OK);
    }

    fn sampler_state(&mut self) -> [u8; SAMPLER_VOICE_STATE_SIZE] {
        let mut s = [0u8; SAMPLER_VOICE_STATE_SIZE];
        self.eng.voice_state(self.vid, &mut s);
        s
    }

    fn set_callback(&mut self, flags: u32, cb: Callback) {
        let v = self.v();
        v.callback = 1;
        v.callback_data = 0;
        v.callback_flags = flags;
        self.env.callback = Some(cb);
    }

    fn set_low_pass(&mut self, frequency: f32, mask: u64) {
        self.eng.rack_mut(self.vid.slot).unwrap().option.set_sampler_max_filters(8);
        let mut body = [0u8; 48];
        body[4..8].copy_from_slice(&1u32.to_le_bytes());
        body[8..12].copy_from_slice(&1u32.to_le_bytes());
        body[16..24].copy_from_slice(&mask.to_le_bytes());
        body[24..28].copy_from_slice(&frequency.to_le_bytes());
        body[28..32].copy_from_slice(&0.707_106_78f32.to_le_bytes());
        body[32..36].copy_from_slice(&1.0f32.to_le_bytes());
        assert_eq!(self.control(&param(0x1000_000a, &body)), OK, "filter ABI command failed");
    }
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn approx(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

#[test]
fn ngs2_sampler_streaming_resample() {
    let mut f = Fixture::new(44100, 2);
    let mut pcm = vec![0i16; 441 * 2];
    for i in 0..441 {
        pcm[i * 2] = (i * 32) as i16;
        pcm[i * 2 + 1] = -((i * 32) as i16);
    }
    f.queue(pcm.as_ptr(), 200, 1, 0, 0);
    f.queue(pcm[400..].as_ptr(), 241, 0, 0, 0);
    for grain in 0..3u32 {
        f.render(160);
        for i in 0..160usize {
            let expected = (((grain * 160 + i as u32) as f64 * 44100.0 / 48000.0).min(440.0) / 1024.0) as f32;
            assert!(approx(f.voice().samples[i], expected, 1e-5), "resampling discontinuity across grain or block boundary");
            assert!(approx(f.voice().samples[160 + i], -expected, 1e-5), "stereo channels were mixed");
        }
    }
    assert!(f.voice().blocks.is_empty(), "final streaming block did not finish");
    assert_eq!(f.voice().state, PlayState::Empty, "finished voice remained active");
    let s = f.sampler_state();
    assert_eq!(u64_at(&s, 24), 441);
    assert_eq!(u64_at(&s, 32), pcm.len() as u64 * 2, "streaming progress does not match consumed source data");
}

#[test]
fn ngs2_sampler_atrac9_vibration_geometry() {
    for index in [7u8, 6] {
        let channels = index as i32 - 5;
        let config = [0xfe, 0x70 | (index << 1), 0x03, 0xf0];
        let Ok(c) = Atrac9Config::new(&config) else {
            panic!("ATRAC9 vibration config {config:02x?} rejected");
        };
        assert_eq!(
            (c.channel_count, c.sample_rate, c.frame_samples, c.frames_per_superframe, c.superframe_bytes),
            (channels, 48000, 256, 4, 128),
            "ATRAC9 vibration config returned incorrect geometry"
        );
        let mut invalid = config;
        invalid[0] = 0;
        assert!(Atrac9Config::new(&invalid).is_err(), "ATRAC9 vibration accepted an invalid header");
        let mut invalid = config;
        invalid[1] |= 1;
        assert!(Atrac9Config::new(&invalid).is_err(), "ATRAC9 vibration accepted an unsupported extension");
    }
}

fn atrac9_two_frame_voice(next: bool) -> Fixture {
    let mut f = Fixture::new(24000, 1);
    f.v().setup_sampler(&ATRAC9_FORMAT);
    let v = f.v();
    v.state = PlayState::Playing;
    v.decoded_frame[0] = 0.0;
    v.decoded_frame[1] = 0.5;
    v.frame_samples = 2;
    if next {
        let n = v.decoded_frame.len();
        v.next_decoded_frame.resize(n, 0.0);
        v.next_decoded_frame[0] = 1.0;
        v.next_decoded_frame[1] = 1.5;
        v.next_frame_samples = 2;
        v.accepts_blocks = false;
    }
    f
}

#[test]
fn ngs2_sampler_atrac9_resample_across_frames() {
    let expected = [0.0f32, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.5];
    let mut whole = atrac9_two_frame_voice(true);
    whole.render(8);
    assert_eq!(whole.voice().samples, expected);
    assert_eq!(whole.voice().decoded_samples, 4);
    assert_eq!(whole.voice().state, PlayState::Empty, "24 kHz ATRAC9 interpolation or source progress failed at frame boundary");

    let mut split = atrac9_two_frame_voice(true);
    split.render(3);
    let mut joined = split.voice().samples.clone();
    split.render(5);
    joined.extend_from_slice(&split.voice().samples);
    assert_eq!(joined, expected, "ATRAC9 resampling changed across render grains");
    assert_eq!(split.voice().decoded_samples, 4);

    let mut pitched = atrac9_two_frame_voice(true);
    pitched.pitch(0x1000_0005, 2.0);
    pitched.render(4);
    assert_eq!(pitched.voice().samples, [0.0f32, 0.5, 1.0, 1.5], "ATRAC9 pitch did not use source-rate phase advancement");
    assert_eq!(pitched.voice().decoded_samples, 4);
}

#[test]
fn ngs2_sampler_atrac9_resample_after_starvation() {
    let mut f = atrac9_two_frame_voice(false);
    f.v().accepts_blocks = true;
    f.render(4);
    assert_eq!(f.voice().samples, [0.0f32, 0.25, 0.5, 0.5], "ATRAC9 resampling lost the final sample before starvation");
    assert_eq!(f.voice().decoded_samples, 2);
    f.render(2);
    assert!(!f.voice().has_samples && f.voice().state == PlayState::Playing && f.voice().decoded_samples == 2,
        "starved ATRAC9 stream consumed source samples");
    let v = f.v();
    let n = v.decoded_frame.len();
    v.next_decoded_frame.resize(n, 0.0);
    v.next_decoded_frame[0] = 1.0;
    v.next_decoded_frame[1] = 1.5;
    v.next_frame_samples = 2;
    f.render(4);
    assert_eq!(f.voice().samples, [1.0f32, 1.25, 1.5, 1.5], "ATRAC9 refill lost source-rate phase");
    assert_eq!(f.voice().decoded_samples, 4);
}

#[test]
fn ngs2_sampler_waveform_block_geometry() {
    let (r, b) = geom::calc_waveform_block(Some(&ATRAC9_FORMAT), 0, 34534);
    assert_eq!((r, b.data_offset, b.data_size, b.num_skip_samples, b.num_samples), (OK, 0, 13056, 0, 34534),
        "ATRAC9 block calculation omitted encoded superframes");
    let (r, b) = geom::calc_waveform_block(Some(&ATRAC9_FORMAT), 513, 1);
    assert_eq!((r, b.data_offset, b.data_size, b.num_skip_samples, b.num_samples), (OK, 192, 192, 1, 1),
        "ATRAC9 block calculation lost the source offset");
    let (r, b) = geom::calc_waveform_block(Some(&ATRAC9_FORMAT), 0, 0);
    assert_eq!((r, b.data_size, b.num_samples), (OK, 0, 0), "empty ATRAC9 block acquired encoded data");
    assert_eq!(geom::calc_waveform_block(None, 0, 1).0, ERR_INVALID_WAVEFORM_FORMAT);

    let mut f = Fixture::new(24000, 1);
    let silence = 0i16;
    f.queue(&silence, 0, 0, u32::MAX, 0);
    f.render(1);
    assert!(f.voice().blocks.is_empty() && f.voice().state == PlayState::Empty, "empty repeated block kept the sampler active");
}

#[test]
fn ngs2_sampler_submixer_setup_and_routing() {
    let mut f = Fixture::new(48000, 2);
    let submixer = f.add_rack(RACK_SUBMIXER, 1, 1, 1, 8);
    let mastering = f.mastering(2);
    let (sub_h, master_h) = (f.handle(submixer), f.handle(mastering));
    {
        let s = f.eng.voice_mut(submixer).unwrap();
        s.state = PlayState::Playing;
        s.ports[0].dest = master_h;
        s.matrices[0] = vec![1.0];
    }
    f.v().ports.push(Port { dest: sub_h, input: 0, volume: 1.0, matrix: -1 });
    let pcm = [8192i16, -8192, 16384, -16384];
    f.queue(pcm.as_ptr(), 2, 0, 0, 0);
    let voices = [f.vid, submixer, mastering];
    f.render_target(mastering, &voices, 1);
    assert!(!f.eng.voice(submixer).unwrap().has_samples && !f.eng.voice(mastering).unwrap().has_samples
        && f.voice().decoded_samples == 0, "unconfigured zero-channel submixer claimed to produce samples");

    let mut body = 2u32.to_le_bytes().to_vec();
    body.extend_from_slice(&0u32.to_le_bytes());
    assert_eq!(f.control_on(submixer, &param(0x2000_0000, &body)), OK);
    {
        let s = f.eng.voice(submixer).unwrap();
        assert!(s.channels == 2 && s.state == PlayState::Empty && s.ports[0].dest == 0 && s.matrices[0].is_empty(),
            "submixer setup did not establish channels and clear old routing");
    }
    {
        let s = f.eng.voice_mut(submixer).unwrap();
        s.state = PlayState::Playing;
        s.ports[0] = Port { dest: master_h, input: 0, volume: 1.0, matrix: 0 };
        s.matrices[0] = vec![1.0, 0.0, 0.0, 1.0];
    }
    for expected in [0.25f32, 0.5] {
        f.reset_rendered(&voices);
        f.render_target(mastering, &voices, 1);
        let m = f.eng.voice(mastering).unwrap();
        assert!(m.has_samples && m.samples[0] == expected && m.samples[1] == -expected,
            "submixer lost routed samples across render grains");
    }
}

#[test]
fn ngs2_sampler_pcm16_render_output() {
    let mut eng = Engine::new();
    let mut env = HostEnv::default();
    let option = SystemOption { num_grain_samples: 4, ..SystemOption::default() };
    eng.system_create(SYSTEM, option, ContextBuffer::default(), Allocator::default());
    let mut sampler_option = RackOption::default_for(RACK_SAMPLER).unwrap();
    sampler_option.set_max_voices(2);
    sampler_option.set_max_ports(1);
    let mastering_option = RackOption::default_for(RACK_MASTERING).unwrap();
    let (sampler_h, mastering_h) = (0x3000_0000u64, 0x3010_0000u64);
    let s_slot = eng.rack_create(&mut env, SYSTEM, RACK_SAMPLER, sampler_option, ContextBuffer::default(), sampler_h).unwrap();
    let m_slot = eng.rack_create(&mut env, SYSTEM, RACK_MASTERING, mastering_option, ContextBuffer::default(), mastering_h).unwrap();
    let mastering = Vid { slot: m_slot, idx: 0 };
    let mastering_handle = eng.voice_handle(mastering);
    {
        let m = eng.voice_mut(mastering).unwrap();
        m.channels = 1;
        m.state = PlayState::Playing;
    }
    let source: [[i16; 4]; 2] = [[16384, -16384, 32767, -32768], [8192, -8192, 32767, -32768]];
    for (i, src) in source.iter().enumerate() {
        let v = eng.voice_mut(Vid { slot: s_slot, idx: i }).unwrap();
        v.setup_sampler(&WaveformFormat::new(0x12, 1, 48000, 0));
        v.blocks.push_back(Block::new(
            addr(src.as_ptr()),
            WaveformBlock { data_size: 8, num_samples: 4, ..WaveformBlock::default() },
        ));
        v.ports[0].dest = mastering_handle;
        v.set_event(1);
    }
    let mut output = [0i16; 4];
    let out = [RenderBuffer { buffer: addr(output.as_mut_ptr()), size: 8, waveform_type: 0x12, num_channels: 1 }];
    assert_eq!(eng.system_render(&mut env, SYSTEM, &out), OK);
    assert_eq!(output, [24576, -24576, 32767, -32768], "PCM16 output failed to mix or saturate mastering voices");
    assert_eq!(eng.system_render(&mut env, SYSTEM, &out), OK);
    assert_eq!(output, [0; 4], "PCM16 output retained samples from the previous render");
    assert!(eng.rack_destroy(&mut env, mastering_h).is_some() && eng.rack_destroy(&mut env, sampler_h).is_some(),
        "PCM16 render rack cleanup failed");
}

#[test]
fn ngs2_sampler_pitch_loop_and_skip() {
    let mut f = Fixture::new(24000, 1);
    let pcm = [-30000i16, 8192, 16384];
    f.queue(pcm.as_ptr(), 2, 0, 1, 1);
    f.pitch(0x1000_0005, 2.0);
    f.render(5);
    let expected = [0.25f32, 0.5, 0.25, 0.5, 0.0];
    for (i, e) in expected.iter().enumerate() {
        assert!(approx(f.voice().samples[i], *e, 1e-5), "pitch, repeat count, or skipped samples incorrect");
    }
}

#[test]
fn ngs2_sampler_pitch_clamps() {
    let mut f = Fixture::new(48000, 1);
    let pcm = [8192i16, 0, 0, 0, -8192];
    f.queue(pcm.as_ptr(), 5, 0, 0, 0);
    f.pitch(0x1000_0005, 5.0);
    f.render(2);
    assert!(f.voice().samples[0] == 0.25 && f.voice().samples[1] == -0.25, "pitch was not clamped to four times playback speed");
}

#[test]
fn ngs2_sampler_starvation_and_pause() {
    let mut f = Fixture::new(48000, 1);
    let pcm = [16384i16, 8192];
    f.queue(pcm.as_ptr(), 2, 1, 0, 0);
    f.render(4);
    assert_eq!(f.voice().state, PlayState::Playing, "stream starvation stopped voice");
    assert_eq!(f.voice().samples[2], 0.0, "starved output was not silent");
    f.queue(pcm.as_ptr(), 2, 0, 0, 0);
    f.v().set_event(16);
    f.render(2);
    assert!(!f.voice().has_samples && f.voice().blocks[0].cursor == 0, "paused voice consumed data");
    f.v().set_event(32);
    f.render(2);
    assert!(f.voice().samples[0] == 0.5 && f.voice().samples[1] == 0.25, "stream failed to resume after starvation/pause");
}

#[test]
fn ngs2_sampler_mono_rate_and_routing() {
    let mut f = Fixture::new(22050, 1);
    let pcm = vec![16384i16; 441];
    f.queue(pcm.as_ptr(), 441, 0, 0, 0);
    let mastering = f.mastering(2);
    let mh = f.handle(mastering);
    f.v().matrices = vec![vec![0.5, 0.25]];
    f.v().ports.push(Port { dest: mh, input: 0, volume: 1.0, matrix: 0 });
    for _ in 0..3 {
        f.reset_rendered(&[mastering, f.vid]);
        let sources = [f.vid, mastering];
        f.render_target(mastering, &sources, 320);
        let m = f.eng.voice(mastering).unwrap();
        assert!(m.has_samples, "standard sampler did not reach mastering voice");
        for j in 0..320 {
            assert!(m.samples[j] == 0.25 && m.samples[320 + j] == 0.125, "mono PCM routing matrix was not applied");
        }
    }
    assert!(f.voice().decoded_samples == 441 && f.voice().blocks.is_empty(), "22.05 kHz effect duration was incorrect");
}

#[test]
fn ngs2_sampler_matrix_output_channels() {
    let mut f = Fixture::new(48000, 2);
    let mastering = f.mastering(8);
    let mh = f.handle(mastering);
    f.v().matrices = vec![vec![1.0, 0.0, 0.0, 1.0]];
    f.v().ports.push(Port { dest: mh, input: 0, volume: 1.0, matrix: 0 });
    let pcm = [8192i16, -8192];
    f.queue(pcm.as_ptr(), 1, 0, 0, 0);
    let sources = [f.vid, mastering];
    f.render_target(mastering, &sources, 1);
    let m = f.eng.voice(mastering).unwrap();
    assert!(m.samples[0] == 0.25 && m.samples[1] == -0.25, "stereo matrix did not route its output channels");
    assert!(m.samples[2..8].iter().all(|&s| s == 0.0), "matrix routed audio to an unspecified output channel");
}

#[test]
fn ngs2_sampler_custom_pcm_still_plays() {
    let mut f = Fixture::with_type(48000, 1, RACK_CUSTOM_SAMPLER);
    let pcm = [8192i16, -16384];
    f.queue(pcm.as_ptr(), 2, 0, 0, 0);
    f.render(4);
    let v = f.voice();
    assert!(v.samples[0] == 0.25 && v.samples[1] == -0.5 && v.samples[2] == 0.0 && v.blocks.is_empty(), "custom PCM sampler regressed");
}

#[test]
fn ngs2_sampler_reuse_resets_routing() {
    for rack in [RACK_SAMPLER, RACK_CUSTOM_SAMPLER] {
        let mut f = Fixture::with_type(48000, 1, rack);
        let mastering = f.mastering(2);
        let mh = f.handle(mastering);
        f.v().ports.push(Port { dest: mh, input: 1, volume: 0.25, matrix: 0 });
        f.v().matrices = vec![vec![0.5, 0.25]];
        f.v().setup_sampler(&WaveformFormat::new(0x12, 2, 48000, 0));
        {
            let v = f.voice();
            assert!(v.ports.len() == 1 && v.ports[0] == Port::default() && v.matrices.len() == 1 && v.matrices[0].is_empty(),
                "sampler setup retained routing from the previous channel layout");
        }
        let pcm = [8192i16, 16384, -8192, -16384];
        f.queue(pcm.as_ptr(), 2, 0, 0, 0);
        f.render(1);
        assert!(!f.voice().has_samples && f.voice().blocks[0].cursor == 0, "sampler setup started the new waveform without a play event");
        f.v().ports[0].dest = mh;
        f.v().set_event(1);
        f.v().rendered = false;
        let sources = [f.vid, mastering];
        f.render_target(mastering, &sources, 2);
        assert_eq!(f.eng.voice(mastering).unwrap().samples, [0.25f32, -0.25, 0.5, -0.5],
            "reused stereo sampler did not restore default channel routing");
    }
}

#[test]
fn ngs2_sampler_deferred_setup() {
    let mut f = Fixture::new(48000, 1);
    // An all-zero setup leaves an earlier setup untouched (measured on
    // firmware); only a voice that was never configured is cleared by it.
    let mut body = vec![0u8; 32];
    assert_eq!(f.control(&param(0x1000_0000, &body)), OK, "deferred stream setup failed");
    {
        let v = f.voice();
        assert!(v.state == PlayState::Playing && v.channels == 1 && v.sample_rate == 48000,
            "deferred stream setup must not clear a configured sampler");
    }
    body[..24].copy_from_slice(&format_bytes(&WaveformFormat::new(0x12, 1, 48000, 0)));
    assert_eq!(f.control(&param(0x1000_0000, &body)), OK, "sampler setup after deferred stream failed");
    let pcm = 8192i16;
    f.queue(&pcm, 1, 0, 0, 0);
    f.v().set_event(1);
    f.render(1);
    assert_eq!(f.voice().samples[0], 0.25, "sampler did not play after deferred stream setup");
}

/// A play event on a voice without channels (never set up) is refused for
/// the whole control walk; other events pass (measured on firmware).
#[test]
fn ngs2_play_event_needs_a_configured_voice() {
    let mut f = Fixture::new(48000, 1);
    f.v().channels = 0;
    assert_eq!(f.control(&param(6, &1u32.to_le_bytes())), 0x804A_8351u32 as i32, "play on an unconfigured voice");
    assert_eq!(f.control(&param(6, &2u32.to_le_bytes())), OK, "stop on an unconfigured voice");
}

#[test]
fn ngs2_sampler_play_state_before_render() {
    let mut f = Fixture::new(48000, 1);
    let pcm = [8192i16, -16384];
    f.v().state = PlayState::Empty;
    f.queue(pcm.as_ptr(), 2, 0, 0, 0);
    f.event(1);
    let flags = f.voice().state_flags;
    assert_eq!(flags, 1, "play must reserve the voice before rendering starts playback");
    let s = f.sampler_state();
    assert!(u32_at(&s, 0) == flags && u64_at(&s, 24) == 0, "state queries disagreed or play consumed samples before render");
    f.render(4);
    assert!(f.voice().samples[0] == 0.25 && f.voice().samples[1] == -0.5, "one-shot effect was dropped before its first render");
}

#[test]
fn ngs2_sampler_state_publication() {
    let mut eng = Engine::new();
    let mut env = HostEnv::default();
    eng.system_create(SYSTEM, SystemOption::default(), ContextBuffer::default(), Allocator::default());
    let mut option = RackOption::default_for(RACK_SAMPLER).unwrap();
    option.set_max_voices(1);
    let rack = 0x4000_0000;
    let slot = eng.rack_create(&mut env, SYSTEM, RACK_SAMPLER, option, ContextBuffer::default(), rack).unwrap();
    let vid = Vid { slot, idx: 0 };
    eng.voice_mut(vid).unwrap().setup_sampler(&WaveformFormat::new(0x12, 1, 48000, 0));
    let check = |eng: &Engine, expected: u32| {
        let mut s = [0u8; SAMPLER_VOICE_STATE_SIZE];
        eng.voice_state(vid, &mut s);
        assert!(eng.voice(vid).unwrap().state_flags == expected && u32_at(&s, 0) == expected,
            "voice state was published at the wrong render boundary");
    };
    // The voice never receives a block, so a playing voice also reports EMPTY.
    const P: u32 = 3 | VOICE_STATE_FLAG_EMPTY;
    let transitions = [(1, 1, P), (16, P, 5), (32, 5, P), (2, P, 0), (1, 1, P), (16, P, 5), (2, 5, 0), (1, 1, P), (4, P, 0), (1, 1, P), (8, P, 0)];
    let output = [RenderBuffer::default()];
    for (event, before, after) in transitions {
        let p = param(6, &(event as u32).to_le_bytes());
        eng.voice_control(&mut env, vid, addr(p.as_ptr()));
        check(&eng, before);
        eng.system_render(&mut env, SYSTEM, &output);
        check(&eng, after);
    }
    assert!(eng.rack_destroy(&mut env, rack).is_some(), "state publication rack cleanup failed");
}

#[test]
fn ngs2_sampler_stop_then_play_before_render() {
    let mut f = Fixture::new(48000, 1);
    let pcm = [16384i16, -8192];
    f.queue(pcm.as_ptr(), 2, 0, 0, 0);
    f.v().set_event(2);
    f.v().set_event(1);
    assert_eq!(f.voice().state_flags, 1, "stopped voice could not restart");
    f.v().set_event(8);
    f.v().set_event(1);
    assert_eq!(f.voice().state_flags, 1, "kill/play commands lost their ordering");
    f.render(2);
    assert!(f.voice().samples[0] == 0.5 && f.voice().samples[1] == -0.25, "restarted voice produced no audio");
}

#[test]
fn ngs2_sampler_low_pass_response() {
    // A second-order Butterworth LPF has unity DC gain and -3 dB at cutoff.
    for frequency in [100.0f64, 1000.0, 10000.0] {
        let mut f = Fixture::new(48000, 1);
        f.set_low_pass(1000.0, 0);
        let pcm: Vec<i16> = (0..48000)
            .map(|i| (8192.0 * ((2.0 * std::f64::consts::PI) * frequency * i as f64 / 48000.0).sin()) as i16)
            .collect();
        f.queue(pcm.as_ptr(), pcm.len() as u32, 0, 0, 0);
        f.render(pcm.len() as u32);
        let (mut input, mut output) = (0f64, 0f64);
        for i in 24000..pcm.len() {
            input += (pcm[i] as f64 / 32768.0).powi(2);
            output += (f.voice().samples[i] as f64).powi(2);
        }
        let ratio = (output / input).sqrt();
        if frequency == 100.0 {
            assert!((ratio - 1.0).abs() < 0.002, "low-pass lost bass");
        }
        if frequency == 1000.0 {
            assert!((ratio - 0.5f64.sqrt()).abs() < 0.002, "cutoff gain incorrect");
        }
        if frequency == 10000.0 {
            assert!(ratio < 0.01, "low-pass did not reject high frequencies");
        }
    }
}

#[test]
fn ngs2_sampler_filter_history_and_channels() {
    let mut whole = Fixture::new(48000, 2);
    let mut split = Fixture::new(48000, 2);
    whole.set_low_pass(1000.0, 1);
    split.set_low_pass(1000.0, 1);
    let mut pcm = vec![0i16; 256 * 2];
    pcm[0] = 16384;
    pcm[1] = 16384;
    whole.queue(pcm.as_ptr(), 256, 0, 0, 0);
    split.queue(pcm.as_ptr(), 32, 1, 0, 0);
    split.queue(pcm[64..].as_ptr(), 224, 0, 0, 0);
    whole.render(256);
    for g in 0..8usize {
        // Titles resend unchanged parameters; this must not reset the delay line.
        split.set_low_pass(1000.0, 1);
        split.render(32);
        for c in 0..2usize {
            for i in 0..32usize {
                assert!(
                    (split.voice().samples[c * 32 + i] - whole.voice().samples[c * 256 + g * 32 + i]).abs() < 1e-7,
                    "filter history changed at a grain/block/control boundary"
                );
            }
        }
    }
    let w = &whole.voice().samples;
    assert!(w[0] == 0.5 && w[1] == 0.0 && w[256] > 0.0 && w[256] < 0.5, "channel mask must bypass selected channels and filter the others");
    let mut stereo = Fixture::new(48000, 2);
    stereo.set_low_pass(1000.0, 0);
    pcm[1] = 0;
    stereo.queue(pcm.as_ptr(), 256, 0, 0, 0);
    stereo.render(256);
    assert!(stereo.voice().samples[256..512].iter().all(|&s| s == 0.0), "filter leaked history between channels");
    stereo.v().setup_sampler(&WaveformFormat::new(0x12, 2, 48000, 0));
    assert!(stereo.voice().filters.is_empty(), "reused voice retained old filters");
}

#[test]
fn ngs2_sampler_filter_uses_output_rate() {
    let mut f = Fixture::new(22050, 1);
    f.set_low_pass(6833.0, 0);
    // Rear filtering runs at the 48 kHz system rate, not the PCM source rate.
    let w = (2.0 * std::f64::consts::PI) * 6833.0 / 48000.0;
    let expected = (1.0 - w.cos()) / (2.0 * (1.0 + w.sin() / (2.0 * 0.707_106_78f32 as f64)));
    assert!((f.voice().filters[0].b0 - expected).abs() < 1e-10, "rear filter used source sample rate");
    f.set_low_pass(48000.0, 0);
    let pcm = [16384i16, 16384];
    f.queue(pcm.as_ptr(), 2, 0, 0, 0);
    f.render(2);
    assert!(f.voice().samples[0] == 0.5 && f.voice().samples[1] == 0.5, "above-Nyquist cutoff was unstable");
}

#[test]
fn ngs2_sampler_filter_control_layout() {
    let mut f = Fixture::new(48000, 2);
    f.eng.rack_mut(f.vid.slot).unwrap().option.set_sampler_max_filters(8);
    // A 56-byte control command with nonzero trailing padding.
    let mut words: [u32; 14] = [56, 0x1000_000a, 0, 1, 1, 0, 0, 0, 0x45d5_8800, 0x3fd5_5555, 0x3f80_0000, 0, 0, 0x13f];
    let control = |f: &mut Fixture, words: &[u32; 14]| {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        f.control(&bytes);
    };
    control(&mut f, &words);
    assert!(f.voice().filters.len() == 1 && f.voice().filters[0].enabled, "FCQ filter command was not recognized");
    let omega = (2.0 * std::f64::consts::PI) * 6833.0 / 48000.0;
    let q = f32::from_bits(0x3fd5_5555) as f64;
    let expected = (1.0 - omega.cos()) / (2.0 * (1.0 + omega.sin() / (2.0 * q)));
    assert!((f.voice().filters[0].b0 - expected).abs() < 1e-10, "FCQ fields were read at the wrong offsets");
    words[6] = 1; // Bypass channel 0 in the low half of the 64-bit mask.
    words[7] = 1; // Bypass channel 32 in the high half.
    let mut wide = Fixture::new(48000, 33);
    wide.eng.rack_mut(wide.vid.slot).unwrap().option.set_sampler_max_filters(8);
    control(&mut wide, &words);
    let pcm = vec![16384i16; 33];
    wide.queue(pcm.as_ptr(), 1, 1, 0, 0);
    wide.render(1);
    let s = &wide.voice().samples;
    assert!(s[0] == 0.5 && s[32] == 0.5 && (s[1] as f64 - expected * 0.5).abs() < 1e-7, "channel mask was not decoded as a 64-bit bypass mask");
    words[4] = 0; // OFF ignores unused FCQ values and clears the old delay line.
    words[8] = 0x7fc0_0000;
    control(&mut wide, &words);
    {
        let flt = &wide.voice().filters[0];
        assert!(!flt.enabled && flt.history.is_empty() && !flt.warned, "filter OFF was treated as an unsupported FCQ configuration");
    }
    wide.queue(pcm.as_ptr(), 1, 1, 0, 0);
    wide.render(1);
    assert_eq!(wide.voice().samples[1], 0.5, "filter OFF did not bypass audio");
    wide.set_low_pass(0.0, 0);
    wide.queue(pcm.as_ptr(), 1, 1, 0, 0);
    wide.render(1);
    assert_eq!(wide.voice().samples[1], 0.0, "zero-Hz low-pass leaked source audio");
}

#[derive(Default)]
struct LoopLog {
    events: Vec<(u32, u32)>,
    pause: bool,
}

fn record_loops(log: Rc<RefCell<LoopLog>>) -> Callback {
    Box::new(move |eng: &mut Engine, info: &CallbackInfo| {
        let mut l = log.borrow_mut();
        l.events.push((info.flag, info.repeats));
        assert!(info.user == 123 && info.attributes == 0, "loop callback metadata changed");
        if l.pause && info.flag == 2 {
            let vid = eng.find_voice(info.voice).unwrap();
            eng.voice_mut(vid).unwrap().set_event(16);
        }
    })
}

#[test]
fn ngs2_sampler_loop_callbacks_and_exit() {
    for (rack, exit_id) in [(RACK_SAMPLER, 0x1000_0004), (RACK_CUSTOM_SAMPLER, 0x4001_0004)] {
        loop_callbacks_and_exit(rack, exit_id);
    }
}

fn loop_callbacks_and_exit(rack: u32, exit_id: u32) {
    let pcm = [8192i16, 16384, 24576];
    let mut f = Fixture::with_type(48000, 1, rack);
    let log = Rc::new(RefCell::new(LoopLog::default()));
    f.set_callback(3, record_loops(log.clone()));
    f.queue(pcm.as_ptr(), 2, 0, 2, 0);
    f.render(6);
    assert_eq!(log.borrow().events, [(2, 1), (2, 2), (1, 2)], "loop and completion callbacks lost the completed repeat count");
    assert!(f.voice().blocks.is_empty(), "finite loop did not complete");

    let mut active = Fixture::with_type(48000, 1, rack);
    let log = Rc::new(RefCell::new(LoopLog { events: Vec::new(), pause: true }));
    active.set_callback(2, record_loops(log.clone()));
    active.queue(pcm.as_ptr(), 2, 1, u32::MAX, 0);
    active.queue(pcm[2..].as_ptr(), 1, 0, 0, 0);
    active.render(6);
    assert!(log.borrow().events.len() == 1 && active.voice().decoded_samples == 2 && active.voice().blocks[0].cursor == 0
        && active.voice().samples[2] == 0.0, "loop callback pause consumed another iteration");
    let exit_loop = param(exit_id, &[]);
    active.control(&exit_loop);
    active.v().set_event(32);
    active.render(3);
    assert!(active.voice().samples == [0.25f32, 0.5, 0.75] && active.voice().blocks.is_empty() && log.borrow().events.len() == 1,
        "exiting infinite loop lost its final iteration or ignored callback flags");

    let mut pending = Fixture::with_type(48000, 1, rack);
    pending.queue(pcm.as_ptr(), 1, 1, 0, 0);
    pending.queue(pcm[1..].as_ptr(), 1, 1, u32::MAX, 0);
    pending.queue(pcm[2..].as_ptr(), 1, 0, 0, 0);
    pending.control(&exit_loop);
    pending.render(3);
    assert!(pending.voice().samples == [0.25f32, 0.5, 0.75] && pending.voice().blocks.is_empty(),
        "exiting a pending loop skipped its first iteration");
    pending.control(&exit_loop);
}

/// One encoded mono ATRAC9 frame with one nonzero spectral coefficient.
const ATRAC9_MONO_BLOCK: [u8; 36] = {
    let mut b = [0u8; 36];
    b[2] = 0x04;
    b[3] = 0x20;
    b[4] = 0x04;
    b[5] = 0xc0;
    b[8] = 0x1f;
    b[9] = 0xc0;
    b
};

#[test]
fn ngs2_sampler_atrac9_looping() {
    // Four encoded mono frames then superframe padding. The block skips one
    // frame plus one sample and ends inside the third frame.
    let mut encoded = [0u8; 192];
    for i in 0..4 {
        encoded[i * 36..i * 36 + 36].copy_from_slice(&ATRAC9_MONO_BLOCK);
        if i != 0 {
            encoded[i * 36] |= 0x80;
        }
    }
    let make = |repeats: u32, log: Rc<RefCell<LoopLog>>| {
        let mut f = Fixture::new(24000, 1);
        f.v().setup_sampler(&ATRAC9_FORMAT);
        f.v().state = PlayState::Playing;
        f.set_callback(3, record_loops(log));
        let block = WaveformBlock { data_size: 192, num_repeats: repeats, num_skip_samples: 129, num_samples: 137, user_data: 123, ..Default::default() };
        f.queue_raw(addr(encoded.as_ptr()), 0, &[block]);
        f
    };
    let logs: Vec<_> = (0..4).map(|_| Rc::new(RefCell::new(LoopLog::default()))).collect();
    let mut baseline = make(0, logs[0].clone());
    baseline.render(274);
    let reference = baseline.voice().samples.clone();
    assert!(reference.iter().any(|s| s.abs() > 0.00001), "synthetic compressed fixture produced no decoded signal");

    let mut finite = make(2, logs[1].clone());
    finite.render(822);
    assert!(finite.voice().state == PlayState::Empty && finite.voice().decoded_samples == 411 && finite.voice().decoded_bytes == 576,
        "compressed finite loop lost samples or input progress");
    assert_eq!(logs[1].borrow().events, [(2, 1), (2, 2), (1, 2)], "compressed finite loop lost completion callbacks");
    let fs = &finite.voice().samples;
    for i in 0..3 {
        assert_eq!(reference[..273], fs[i * 274..i * 274 + 273], "compressed loop did not restore decoder history and skip samples");
        let tail = if i == 2 { reference[273] } else { lerp(reference[272], reference[0], 0.5) };
        assert_eq!(fs[i * 274 + 273], tail, "compressed loop dropped the final frame or broke boundary interpolation");
    }

    let mut split = make(2, logs[2].clone());
    split.render(273);
    let mut joined = split.voice().samples.clone();
    split.render(549);
    joined.extend_from_slice(&split.voice().samples);
    assert_eq!(&joined, &finite.voice().samples, "compressed loop changed across render grains");
    assert_eq!(logs[2].borrow().events, logs[1].borrow().events);

    let mut infinite = make(u32::MAX, logs[3].clone());
    infinite.render(300);
    assert!(infinite.voice().state == PlayState::Playing && infinite.voice().blocks[0].info.num_repeats == u32::MAX,
        "compressed infinite loop did not rewind");
    infinite.control(&param(0x1000_0004, &[]));
    infinite.render(248);
    assert!(infinite.voice().state == PlayState::Empty && infinite.voice().decoded_samples == 274,
        "compressed loop exit failed to finish the active iteration");
    assert_eq!(logs[3].borrow().events, [(2, 1), (1, 1)]);
}

#[test]
fn ngs2_sampler_waveform_read_address() {
    let mut f = Fixture::new(48000, 2);
    let pcm = [-100i16, -100, 8192, 4096, 16384, 8192];
    let at = |i: usize| addr(pcm.as_ptr()) + i as u64 * 2;
    f.queue(pcm.as_ptr(), 2, 1, 1, 1);
    let next_data = |f: &mut Fixture| u64_at(&f.sampler_state(), 48);
    assert_eq!(next_data(&mut f), at(2), "read address ignored skipped source samples");
    f.render(1);
    assert_eq!(next_data(&mut f), at(4), "read address did not advance by a stereo frame");
    f.render(1);
    assert_eq!(next_data(&mut f), at(2), "read address did not wrap at a loop boundary");
    f.render(2);
    assert!(f.voice().blocks.is_empty() && next_data(&mut f) == at(6), "starved read address did not preserve the consumed waveform end");
    f.queue(pcm[2..].as_ptr(), 2, 0, 0, 0);
    assert_eq!(next_data(&mut f), at(2), "refilled read address retained the previous waveform end");
    f.render(2);
    assert_eq!(next_data(&mut f), at(6), "completed read address did not preserve waveform end");
    f.v().setup_sampler(&WaveformFormat::new(0x12, 2, 48000, 0));
    assert_eq!(next_data(&mut f), 0, "setup retained an old waveform address");
}

fn apply_event_once(event: u32, calls: Rc<RefCell<u32>>) -> Callback {
    Box::new(move |eng: &mut Engine, info: &CallbackInfo| {
        let mut c = calls.borrow_mut();
        *c += 1;
        if *c == 1 {
            let vid = eng.find_voice(info.voice).unwrap();
            eng.voice_mut(vid).unwrap().set_event(event);
        }
    })
}

#[test]
fn ngs2_sampler_callback_stops_consumption() {
    for rack in [RACK_SAMPLER, RACK_CUSTOM_SAMPLER] {
        for event in [2u32, 4, 8, 16] {
            let mut f = Fixture::with_type(48000, 1, rack);
            let pcm = [8192i16, 16384];
            f.queue(pcm.as_ptr(), 1, 1, 0, 0);
            f.queue(pcm[1..].as_ptr(), 1, 1, 0, 0);
            let calls = Rc::new(RefCell::new(0));
            f.set_callback(1, apply_event_once(event, calls.clone()));
            f.render(2);
            assert!(f.voice().state != PlayState::Playing && *calls.borrow() == 1, "completion callback failed to interrupt playback");
            let v = f.voice();
            assert!(v.samples[0] == 0.25 && v.samples[1] == 0.0 && v.decoded_samples == 1 && v.blocks.len() == 1 && v.blocks[0].cursor == 0,
                "callback pause/stop consumed the next queued block");
            if event == 16 {
                f.v().set_event(32);
                f.render(1);
                assert!(f.voice().samples[0] == 0.5 && f.voice().decoded_samples == 2, "callback pause failed to preserve queued audio for resume");
            }
        }
    }
}

#[test]
fn ngs2_sampler_callback_pause_during_large_step() {
    let mut f = Fixture::new(48000, 1);
    let pcm = [8192i16, 16384, 24576, 4096];
    f.queue(pcm.as_ptr(), 1, 1, 0, 0);
    f.queue(pcm[1..].as_ptr(), 1, 1, 0, 0);
    f.queue(pcm[2..].as_ptr(), 2, 1, 0, 0);
    f.v().sample_step = 120000u64 << 32;
    let calls = Rc::new(RefCell::new(0));
    f.set_callback(1, apply_event_once(16, calls.clone()));
    f.render(2);
    assert!(*calls.borrow() == 1 && f.voice().decoded_samples == 1 && f.voice().sample_phase == 72000u64 << 32,
        "pause advanced across additional blocks in a large resampling step");
    f.v().set_event(32);
    f.render(1);
    assert!(approx(f.voice().samples[0], 0.4375, 1e-7) && *calls.borrow() == 3, "resume lost fractional position or read past the next short block");
}

#[test]
fn ngs2_sampler_custom_sampler_controls() {
    let mut f = Fixture::with_type(24000, 1, RACK_CUSTOM_SAMPLER);
    let pcm = [8192i16, 16384, 24576, 4096];
    f.queue(pcm.as_ptr(), 4, 1, 0, 0);
    f.render(1);
    assert_eq!(f.voice().samples[0], 0.25, "custom sampler source rate was rejected");
    f.queue(pcm[2..].as_ptr(), 2, 4, 0, 0);
    f.render(1);
    assert!(f.voice().samples[0] == 0.75 && f.voice().blocks.len() == 1, "replacement blocks retained old data or fractional position");
    f.pitch(0x4001_0005, 0.0);
    f.render(2);
    assert!(f.voice().samples[0] == 0.4375 && f.voice().samples[1] == 0.4375, "zero custom sampler pitch advanced playback");
    f.pitch(0x4001_0005, 4.0);
    f.render(1);
    assert!(f.voice().blocks.is_empty(), "custom sampler pitch did not advance playback");
    f.eng.rack_mut(f.vid.slot).unwrap().option.set_custom_state_size(CUSTOM_SAMPLER_VOICE_STATE_SIZE as u32);
    assert_eq!(f.eng.rack(f.vid.slot).unwrap().option.custom_state_size(), 48);
    let mut s = [0u8; CUSTOM_SAMPLER_VOICE_STATE_SIZE];
    f.eng.voice_state(f.vid, &mut s);
    assert_eq!(u32_at(&s, 0), f.voice().state_flags, "custom sampler state layout regressed");
}

#[test]
fn ngs2_sampler_filter_tail_during_starvation() {
    let mut reference = Fixture::new(48000, 1);
    let mut stream = Fixture::new(48000, 1);
    reference.set_low_pass(1000.0, 0);
    stream.set_low_pass(1000.0, 0);
    let mut pcm = vec![0i16; 128];
    pcm[0] = 16384;
    pcm[64] = 16384;
    reference.queue(pcm.as_ptr(), 128, 1, 0, 0);
    reference.render(128);
    let r = reference.voice().samples.clone();
    stream.queue(pcm.as_ptr(), 1, 1, 0, 0);
    stream.render(1);
    assert_eq!(stream.voice().samples[0], r[0], "initial filter sample differs");
    stream.render(63);
    assert!(stream.voice().has_samples, "starved filter tail was not available for routing");
    for i in 0..63 {
        assert!((stream.voice().samples[i] - r[i + 1]).abs() < 1e-7, "filter tail changed at a starvation/grain boundary");
    }
    stream.queue(pcm[64..].as_ptr(), 64, 1, 0, 0);
    stream.render(64);
    for i in 0..64 {
        assert!((stream.voice().samples[i] - r[i + 64]).abs() < 1e-7, "refilled stream reused stale filter history");
    }
    stream.render(8192);
    stream.render(32);
    assert!(!stream.voice().has_samples && !stream.voice().filters[0].has_history(), "fully decayed filter kept a silent stream active");
}

#[test]
fn ngs2_sampler_resampling_phase_after_starvation() {
    for rack in [RACK_SAMPLER, RACK_CUSTOM_SAMPLER] {
        for rate in [44100u32, 96000] {
            let mut reference = Fixture::with_type(rate, 1, rack);
            let mut stream = Fixture::with_type(rate, 1, rack);
            let pcm = [0i16, 0, 16384, 8192];
            let frames = if rate == 44100 { 2 } else { 1 };
            reference.queue(pcm.as_ptr(), 4, 0, 0, 0);
            stream.queue(pcm.as_ptr(), 1, 1, 0, 0);
            reference.render(frames);
            stream.render(frames);
            assert!(stream.voice().blocks.is_empty() && stream.voice().state == PlayState::Playing, "open stream did not enter starvation");
            stream.render(8);
            assert!(!stream.voice().has_samples, "starved stream produced audio without a filter");
            stream.queue(pcm[1..].as_ptr(), 3, 0, 0, 0);
            reference.render(1);
            stream.render(1);
            assert!((stream.voice().samples[0] - reference.voice().samples[0]).abs() < 1e-7, "stream refill lost resampling phase");
            assert_eq!(stream.voice().decoded_samples, reference.voice().decoded_samples, "stream refill lost pending source advancement");
            stream.v().setup_sampler(&WaveformFormat::new(0x12, 1, rate, 0));
            assert_eq!(stream.voice().sample_phase, 0, "sampler setup retained old resampling phase");
        }
    }
}

#[test]
fn ngs2_sampler_finite_filter_tail() {
    let mut whole = Fixture::new(48000, 1);
    let mut split = Fixture::new(48000, 1);
    whole.set_low_pass(1000.0, 0);
    split.set_low_pass(1000.0, 0);
    let impulse = 16384i16;
    whole.queue(&impulse, 1, 0, 0, 0);
    split.queue(&impulse, 1, 0, 0, 0);
    whole.render(256);
    for i in 0..whole.voice().samples.len() {
        split.render(1);
        assert!((split.voice().samples[0] - whole.voice().samples[i]).abs() < 1e-7, "finite filter tail changed with render grain size");
    }
    assert_eq!(split.voice().state, PlayState::Playing, "finite voice stopped before its filter tail decayed");
    split.render(8192);
    assert_eq!(split.voice().state, PlayState::Empty, "finite voice remained active after its filter tail decayed");
    split.render(32);
    assert!(!split.voice().has_samples, "completed finite filter tail still produced samples");
}

/// An ATRAC9 superframe split at any byte between a waveform block and a
/// data-only (APPEND) continuation: nothing plays while the superframe is
/// incomplete, the continuation's own counts are ignored, and the stream
/// then plays exactly as the whole superframe does.
#[test]
fn ngs2_sampler_atrac9_append_boundaries() {
    let mut encoded = [0u8; 192];
    for i in 0..4 {
        encoded[i * 36..i * 36 + 36].copy_from_slice(&ATRAC9_MONO_BLOCK);
        if i != 0 {
            encoded[i * 36] |= 0x80;
        }
    }
    let make = || {
        let mut f = Fixture::new(24000, 1);
        f.v().setup_sampler(&ATRAC9_FORMAT);
        f.v().state = PlayState::Playing;
        f
    };
    let waveform = |size: usize| WaveformBlock { data_size: size as u64, num_skip_samples: 129, num_samples: 137, ..Default::default() };
    let mut reference = make();
    reference.queue_raw(addr(encoded.as_ptr()), 0, &[waveform(192)]);
    reference.render(274);
    let expected = reference.voice().samples.clone();
    assert!(expected.iter().any(|s| s.abs() > 0.00001), "synthetic compressed fixture produced no decoded signal");
    for split in [1usize, 17, 35, 36, 37, 100, 144, 191] {
        let mut f = make();
        f.queue_raw(addr(encoded.as_ptr()), 1, &[waveform(split)]);
        f.render(16);
        assert!(!f.voice().has_samples && f.voice().state == PlayState::Playing, "split {split}: a partial superframe played");
        let rest = WaveformBlock {
            data_offset: split as u64,
            data_size: (192 - split) as u64,
            num_repeats: u32::MAX,
            num_skip_samples: u32::MAX,
            num_samples: u32::MAX,
            ..Default::default()
        };
        f.queue_raw(addr(encoded.as_ptr()), 2, &[rest]);
        f.render(274);
        assert_eq!(f.voice().samples, expected, "split {split}: the appended stream differs from the whole superframe");
        let v = f.voice();
        assert!(
            v.decoded_samples == 137 && v.decoded_bytes == 192 && v.state == reference.voice().state,
            "split {split}: progress {} samples, {} bytes",
            v.decoded_samples,
            v.decoded_bytes
        );
    }
}

use super::reverb::{I3dl2Param, I3DL2_PARAM_SIZE};

#[allow(clippy::too_many_arguments)]
fn i3dl2(
    wet: f32,
    dry: f32,
    room: i32,
    room_hf: i32,
    reflection_pattern: u32,
    decay_time: f32,
    decay_hf_ratio: f32,
    reflections: i32,
    reflections_delay: f32,
    reverb: i32,
    reverb_delay: f32,
    diffusion: f32,
    density: f32,
    hf_reference: f32,
) -> I3dl2Param {
    I3dl2Param {
        wet,
        dry,
        room,
        room_hf,
        reflection_pattern,
        decay_time,
        decay_hf_ratio,
        reflections,
        reflections_delay,
        reverb,
        reverb_delay,
        diffusion,
        density,
        hf_reference,
    }
}

fn reverb_setup(channels: u32) -> Vec<u8> {
    let body: Vec<u8> = [channels, channels, 0, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
    param(0x2001_0000, &body)
}

fn reverb_params(p: &I3dl2Param) -> Vec<u8> {
    let bytes = p.to_bytes();
    assert_eq!(bytes.len(), I3DL2_PARAM_SIZE);
    param(0x2001_0001, &bytes)
}

struct ReverbCase {
    channels: u32,
    amplitude: f32,
    noise: bool,
    params: I3dl2Param,
    change_grain: i32,
    change: I3dl2Param,
    expected: Vec<(usize, Vec<f32>)>,
}

/// A `channels` PCM16 sampler (an impulse at frame 3, or noise) through a
/// reverb voice into a mastering voice, 256-sample grains; the output
/// frames interleaved.
fn render_reverb(case: &ReverbCase, frames: usize) -> Vec<f32> {
    const GRAIN: usize = 256;
    let channels = case.channels as usize;
    let mut f = Fixture::new(48000, case.channels);
    let reverb = f.add_rack(RACK_REVERB, 1, 1, 1, 8);
    let mastering = f.mastering(case.channels);
    let (reverb_h, master_h) = (f.handle(reverb), f.handle(mastering));
    assert_eq!(f.control_on(reverb, &reverb_setup(case.channels)), OK);
    assert_eq!(f.control_on(reverb, &reverb_params(&case.params)), OK);
    assert_eq!(f.eng.voice(reverb).unwrap().channels, case.channels, "reverb setup rejected");
    {
        let r = f.eng.voice_mut(reverb).unwrap();
        r.ports[0] = Port { dest: master_h, input: 0, volume: 1.0, matrix: -1 };
        r.set_event(1);
    }
    let mut pcm = vec![0i16; 4096 * channels];
    let mut seed = 1u32;
    if case.noise {
        let mut i = 0;
        while i + 3 * channels < pcm.len() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            if i < 3000 * channels {
                pcm[i + 3 * channels] = (((seed >> 16) & 0x3fff) as i32 - 0x2000) as i16;
            }
            i += 1;
        }
    } else {
        for c in 0..channels {
            pcm[3 * channels + c] = ((16384 - 4000 * c as i32) as f32 * case.amplitude) as i16;
        }
    }
    let mut identity = vec![0.0f32; channels * channels];
    for c in 0..channels {
        identity[c * channels + c] = 1.0;
    }
    f.v().matrices = vec![identity];
    f.v().ports.push(Port { dest: reverb_h, input: 0, volume: 1.0, matrix: 0 });
    f.queue(pcm.as_ptr(), 4096, 0, 0, 0);
    let voices = [f.vid, reverb, mastering];
    let mut out = Vec::with_capacity(frames * channels);
    for grain in 0..frames.div_ceil(GRAIN) {
        if grain as i32 == case.change_grain {
            assert_eq!(f.control_on(reverb, &reverb_params(&case.change)), OK);
        }
        f.reset_rendered(&voices);
        f.render_target(mastering, &voices, GRAIN as u32);
        let m = f.eng.voice(mastering).unwrap();
        for i in 0..GRAIN {
            for c in 0..channels {
                out.push(if m.has_samples { m.samples[c * GRAIN + i] } else { 0.0 });
            }
        }
    }
    out
}

/// Reverb output of the console's libSceNgs2 for impulses and noise,
/// through several parameter sets, a parameter change and a density change.
#[test]
fn ngs2_reverb_matches_console() {
    let cave = i3dl2(1.0, 0.0, -1000, 0, 0, 2.91, 1.3, -602, 0.015, -302, 0.022, 100.0, 100.0, 5000.0);
    let changed = I3dl2Param {
        reflection_pattern: 5,
        decay_time: 0.7,
        decay_hf_ratio: 0.6,
        reflections_delay: 0.004,
        reverb_delay: 0.03,
        room_hf: -900,
        wet: 0.6,
        dry: 0.4,
        ..cave
    };
    let denser = I3dl2Param { density: 99.999, ..cave };
    let none = I3dl2Param::default();
    let e = |frames: &[(usize, &[f32])]| -> Vec<(usize, Vec<f32>)> { frames.iter().map(|(f, s)| (*f, s.to_vec())).collect() };
    let cases = vec![
        ReverbCase { channels: 2, amplitude: 1.0, noise: false, params: cave, change_grain: -1, change: none, expected: e(&[
            (100, &[0.0, 0.0]), (723, &[0.0687032491, 0.0687032491]), (1055, &[0.0392590016, 0.0392590016]), (1417, &[0.0131517649, 0.0131517649]),
            (1779, &[0.0098147504, 0.0098147504]), (2500, &[0.0, 0.0]), (4000, &[1.45101382e-07, -1.15403088e-06]), (6000, &[0.000655628915, 3.03641864e-05]),
            (9000, &[0.000400578661, -0.000684358703]), (12000, &[0.000191548112, 0.000220489033]), (16000, &[0.000210488462, 0.000146183695]),
            (20000, &[2.41150738e-05, -0.000864534522]), (26000, &[0.000125532999, -0.000234790568]), (32000, &[-0.000462158496, 4.56995949e-05]),
        ]) },
        ReverbCase { channels: 2, amplitude: 1.0, noise: false,
            params: i3dl2(1.0, 0.0, -500, -800, 3, 1.2, 0.5, -300, 0.01, 0, 0.03, 80.0, 60.0, 3000.0), change_grain: -1, change: none, expected: e(&[
            (100, &[0.0, 0.0]), (723, &[0.000166983227, 0.000166983227]), (1055, &[-3.41919585e-05, -3.41919585e-05]), (1417, &[0.0, 0.0]), (1779, &[0.0, 0.0]),
            (2500, &[0.0, 1.55222626e-08]), (4000, &[-0.000106364969, 9.90879998e-05]), (6000, &[0.00013573296, 7.22560799e-06]),
            (9000, &[-7.91876082e-05, 0.000225067721]), (12000, &[0.000150469248, 0.00036288987]), (16000, &[-7.86288219e-05, -0.000195921995]),
            (20000, &[1.01910318e-05, 5.79599109e-05]), (26000, &[2.14232614e-06, -2.05506512e-05]), (32000, &[-4.31467561e-06, 2.34100444e-05]),
        ]) },
        ReverbCase { channels: 6, amplitude: 1.0, noise: false,
            params: i3dl2(1.0, 0.2, -200, -300, 12, 1.0, 0.9, -100, 0.03, -100, 0.04, 70.0, 90.0, 4000.0), change_grain: -1, change: none, expected: e(&[
            (6000, &[-0.000167909253, -0.000179917421, -0.000647808949, 0.0, 3.22720189e-05, 0.000213803389]),
            (9000, &[8.47857882e-05, 8.62322413e-05, -0.000100753634, 0.0, 0.000109430795, -0.000122895101]),
            (12000, &[4.37544441e-05, 3.58103216e-06, -0.00013746327, 0.0, 9.3237948e-05, -0.00013529828]),
            (16000, &[8.63046807e-08, 1.62214419e-05, 8.89167641e-05, 0.0, 1.02178528e-05, 0.000201705057]),
            (20000, &[-9.6526619e-06, -2.73812566e-05, -1.92096536e-07, 0.0, -0.000116049101, 6.94858772e-06]),
            (26000, &[1.27750207e-06, -1.59169917e-06, 5.30674515e-05, 0.0, -3.59285696e-05, 5.16466662e-06]),
            (32000, &[3.5737969e-06, 4.19655044e-06, -2.30460955e-07, 0.0, 1.16596939e-05, 1.14889262e-05]),
        ]) },
        ReverbCase { channels: 2, amplitude: 1.0, noise: false, params: cave, change_grain: 6, change: changed, expected: e(&[
            (723, &[0.0687032491, 0.0687032491]), (1055, &[0.0392590016, 0.0392590016]), (1417, &[0.0131517649, 0.0131517649]),
            (1779, &[4.38156894e-05, 4.38156894e-05]), (2500, &[0.0, 0.0]), (4000, &[-1.60370582e-05, -4.44955149e-05]), (6000, &[1.02755057e-05, 8.34046841e-06]),
            (9000, &[1.1641363e-05, -4.69685165e-06]), (12000, &[3.57917179e-06, -2.34550089e-06]), (16000, &[-8.00212234e-08, 3.19533115e-06]),
            (20000, &[5.66619519e-07, 4.26969848e-07]), (26000, &[1.46052287e-07, -2.44572629e-08]), (32000, &[-4.45668853e-08, 1.51349102e-08]),
        ]) },
        ReverbCase { channels: 2, amplitude: 0.02, noise: false,
            params: i3dl2(40.0, -3.0, 5000, 500, 20, 50.0, 5.0, 3000, 0.9, 5000, 0.9, 300.0, 300.0, 50000.0), change_grain: -1, change: none, expected: e(&[
            (14403, &[0.877262115, 0.877262115]), (14500, &[0.0, 0.0]), (15000, &[0.0, 0.0]), (15500, &[0.0, 7.50572099e-07]), (16000, &[4.95207075e-07, 6.37215771e-06]),
            (20000, &[0.00482429564, -0.00149071764]), (26000, &[-0.0096597122, 0.00718426611]), (32000, &[0.0107425917, -0.000738018774]),
        ]) },
        ReverbCase { channels: 2, amplitude: 1.0, noise: false,
            params: i3dl2(1.0, 0.3, 0, -200, 15, 0.8, 1.0, 0, 0.002, -500, 0.004, 60.0, 80.0, 6000.0), change_grain: -1, change: none, expected: e(&[
            (60, &[0.0, 0.0]), (99, &[0.0947210863, 0.0947210863]), (100, &[0.0472047739, 0.0472047739]), (150, &[0.0, 0.0]), (200, &[0.0, 0.0]), (300, &[0.0, 0.0]),
            (2000, &[-9.86956729e-06, 7.96923473e-07]), (3000, &[-0.000152820328, -4.59704825e-06]), (5000, &[0.000203833857, -0.000208990226]),
            (8000, &[-3.69777154e-05, -0.000259475899]),
        ]) },
        ReverbCase { channels: 2, amplitude: 1.0, noise: true, params: cave, change_grain: 6, change: changed, expected: e(&[
            (1540, &[0.0222691782, 0.0242232569]), (1600, &[0.0201471522, 0.0186609477]), (1700, &[-0.0211425349, -0.0235042125]),
            (1780, &[-0.0168117173, 0.00417749817]), (1800, &[-0.0866222829, -0.0945812687]), (2000, &[0.0131965755, -0.0707511827]),
            (2500, &[0.0942443311, -0.100531064]), (3500, &[0.000985965831, 0.000946713379]), (8000, &[-0.000968122098, 0.000556072395]),
            (15000, &[0.000227724231, 0.000151095577]),
        ]) },
        ReverbCase { channels: 2, amplitude: 1.0, noise: true, params: cave, change_grain: 20, change: denser, expected: e(&[
            (5200, &[-0.00942458585, -0.0064059021]), (6000, &[0.00248076022, 0.00317249796]),
            (8000, &[0.00102439919, 0.00105210941]), (12000, &[1.63059376e-05, 6.10309735e-06]), (20000, &[-1.08853078e-06, -2.99162899e-08]),
        ]) },
    ];
    for (k, case) in cases.iter().enumerate() {
        let out = render_reverb(case, 32256);
        for (frame, samples) in &case.expected {
            for c in 0..case.channels as usize {
                if c == 3 {
                    continue;
                }
                let value = out[frame * case.channels as usize + c];
                let expected = samples[c];
                assert!(
                    (value - expected).abs() <= 2e-7 + expected.abs() * 1e-5,
                    "case {k} frame {frame} channel {c}: {value} != {expected}"
                );
            }
        }
    }
}

/// Reverb setup checks; the setup parameters leave only a faint tail.
#[test]
fn ngs2_reverb_setup() {
    let mut f = Fixture::new(48000, 1);
    let reverb = f.add_rack(RACK_REVERB, 1, 1, 1, 8);
    let mastering = f.mastering(2);
    for channels in [3, 4, 5, 7] {
        assert_eq!(f.control_on(reverb, &reverb_setup(channels)), OK);
        assert!(f.eng.voice(reverb).unwrap().reverb.is_none(), "{channels}-channel reverb accepted");
    }
    let conversion: Vec<u8> = [1u32, 2, 0, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
    f.control_on(reverb, &param(0x2001_0000, &conversion));
    let flags: Vec<u8> = [2u32, 2, 1, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
    f.control_on(reverb, &param(0x2001_0000, &flags));
    assert!(f.eng.voice(reverb).unwrap().reverb.is_none(), "reverb channel conversion or setup flags accepted");
    assert_eq!(f.control_on(reverb, &reverb_setup(2)), OK);
    assert_eq!(f.eng.voice(reverb).unwrap().channels, 2);
    assert_eq!(*f.eng.voice(reverb).unwrap().reverb.as_ref().unwrap().params(), I3dl2Param::SETUP);
    let nan = I3dl2Param { decay_time: f32::NAN, ..I3dl2Param::SETUP };
    f.control_on(reverb, &reverb_params(&nan));
    assert_eq!(*f.eng.voice(reverb).unwrap().reverb.as_ref().unwrap().params(), I3dl2Param::SETUP, "NaN parameters stored");

    let master_h = f.handle(mastering);
    let reverb_h = f.handle(reverb);
    {
        let r = f.eng.voice_mut(reverb).unwrap();
        r.ports[0] = Port { dest: master_h, input: 0, volume: 1.0, matrix: -1 };
        r.set_event(1);
    }
    let pcm = vec![16384i16; 256 * 64];
    f.v().ports.push(Port { dest: reverb_h, input: 0, volume: 1.0, matrix: -1 });
    f.queue(pcm.as_ptr(), pcm.len() as u32, 0, 0, 0);
    let voices = [f.vid, reverb, mastering];
    let mut loudest = 0.0f32;
    for _ in 0..600 {
        f.reset_rendered(&voices);
        f.render_target(mastering, &voices, 256);
        let m = f.eng.voice(mastering).unwrap();
        if m.has_samples {
            loudest = m.samples.iter().fold(loudest, |l, s| l.max(s.abs()));
        }
    }
    assert!(loudest > 0.0 && loudest < 1e-8, "setup reverb level {loudest}");
}

/// One frame of a `src_channels` sampler with 0.5 on channel `src` mixed
/// into a `dst_channels` mastering voice through a port; `prepare` sets up
/// the sampler's port and matrices.
fn mix_frame_with(src_channels: u32, pcm: &[i16], dst_channels: u32, prepare: impl FnOnce(&mut Fixture, u64)) -> Vec<f32> {
    let mut f = Fixture::new(48000, src_channels);
    let mastering = f.mastering(dst_channels);
    let mh = f.handle(mastering);
    prepare(&mut f, mh);
    f.queue(pcm.as_ptr(), 1, 0, 0, 0);
    let sources = [f.vid, mastering];
    f.render_target(mastering, &sources, 1);
    let m = f.eng.voice(mastering).unwrap();
    assert!(m.has_samples, "nothing reached the mastering voice");
    m.samples[..dst_channels as usize].to_vec()
}

fn mix_frame(src_channels: u32, src: usize, dst_channels: u32) -> Vec<f32> {
    let mut pcm = vec![0i16; src_channels as usize];
    pcm[src] = 16384;
    mix_frame_with(src_channels, &pcm, dst_channels, |f, mh| {
        f.v().ports.push(Port { dest: mh, input: 0, volume: 1.0, matrix: -1 });
    })
}

/// Ports without matrix levels mix through the console's default channel
/// map (measured for every source width into 1, 2, 6 and 8 channels).
#[test]
fn ngs2_default_channel_map() {
    let h = std::f32::consts::FRAC_1_SQRT_2;
    let half = 0.5 * h;
    let quarter = 0.5 * (std::f32::consts::SQRT_2 / 4.0);
    let cases: &[(u32, usize, &[f32])] = &[
        (1, 0, &[0.5]),
        (1, 0, &[half, half]),
        (1, 0, &[0., 0., 0.5, 0., 0., 0.]),
        (1, 0, &[0., 0., 0.5, 0., 0., 0., 0., 0.]),
        (2, 0, &[half]),
        (2, 1, &[half]),
        (2, 1, &[0., 0.5]),
        (2, 1, &[0., 0.5, 0., 0., 0., 0.]),
        (3, 2, &[0., 0.]),
        (3, 2, &[0., 0., 0., 0.5, 0., 0.]),
        (4, 2, &[0.25]),
        (4, 2, &[half, 0.]),
        (4, 3, &[0., 0., 0., 0., 0., 0.5]),
        (5, 2, &[half, half]),
        (5, 3, &[0.25]),
        (5, 3, &[0., 0.]),
        (6, 3, &[0.]),
        (6, 4, &[0.25, 0.]),
        (6, 5, &[0., 0., 0., 0., 0., 0.5, 0., 0.]),
        (7, 6, &[quarter]),
        (7, 6, &[quarter, quarter]),
        (7, 6, &[0., 0., 0., 0., half, half]),
        (7, 6, &[0., 0., 0., 0., 0., 0., half, half]),
        (8, 6, &[0.25, 0.]),
        (8, 7, &[0., 0., 0., 0., 0., 0.5]),
        (8, 7, &[0., 0., 0., 0., 0., 0., 0., 0.5]),
    ];
    for &(src_channels, src, expected) in cases {
        assert_eq!(
            mix_frame(src_channels, src, expected.len() as u32),
            expected,
            "default map of channel {src} of {src_channels} into {} channels",
            expected.len()
        );
    }
    // Destination widths the map does not know keep channel n on n.
    assert_eq!(default_level(1, 0, 4, 0), None);
    assert_eq!(mix_frame(2, 1, 4), [0., 0.5, 0., 0.]);
}

/// A port that selects a matrix whose levels were never set mixes like a
/// port without a matrix.
#[test]
fn ngs2_unset_matrix_uses_default_map() {
    let spread = 0.25 * std::f32::consts::FRAC_1_SQRT_2;
    let out = mix_frame_with(1, &[16384], 2, |f, mh| {
        f.v().matrices = vec![Vec::new()];
        f.v().ports.push(Port { dest: mh, input: 0, volume: 0.5, matrix: 0 });
    });
    assert_eq!(out, [spread, spread], "mono through an unset matrix");
    let out = mix_frame_with(2, &[16384, 8192], 2, |f, mh| {
        f.v().matrices = vec![Vec::new()];
        f.v().ports.push(Port { dest: mh, input: 0, volume: 1.0, matrix: 0 });
    });
    assert_eq!(out, [0.5, 0.25], "stereo through an unset matrix");
}

/// Matrix levels are stored clamped to -4..4 (NaN kept); no levels are
/// rejected, through voice control and directly.
#[test]
fn ngs2_matrix_level_clamp() {
    let frame = |levels: &[f32], control: bool| {
        let levels = levels.to_vec();
        mix_frame_with(1, &[16384], 2, move |f, mh| {
            f.v().matrices = vec![Vec::new()];
            f.v().ports.push(Port { dest: mh, input: 0, volume: 1.0, matrix: 0 });
            if control {
                let mut body = 0u32.to_le_bytes().to_vec();
                body.extend_from_slice(&(levels.len() as u32).to_le_bytes());
                body.extend_from_slice(&addr(levels.as_ptr()).to_le_bytes());
                assert_eq!(f.control(&param(1, &body)), OK);
            } else {
                f.v().set_matrix(0, &levels);
            }
        })
    };
    for control in [false, true] {
        assert_eq!(frame(&[4.0, -4.0], control), [2.0, -2.0]);
        assert_eq!(frame(&[4.0001, -4.5], control), [2.0, -2.0]);
        assert_eq!(frame(&[100.0, -100.0], control), [2.0, -2.0]);
        assert_eq!(frame(&[f32::INFINITY, f32::NEG_INFINITY], control), [2.0, -2.0]);
        assert_eq!(frame(&[3.5, -0.25], control), [1.75, -0.125]);
        let nan = frame(&[f32::NAN, 0.5], control);
        assert!(nan[0].is_nan() && nan[1] == 0.25, "{nan:?}");
    }
    let mut f = Fixture::new(48000, 1);
    f.v().matrices = vec![vec![1.0, 1.0]];
    f.v().set_matrix(0, &[]);
    assert_eq!(f.voice().matrices[0], [1.0, 1.0], "an empty level list changed the matrix");
    f.v().set_matrix(0, &[0.5; 65]);
    assert_eq!(f.voice().matrices[0], [1.0, 1.0], "more than 8x8 levels changed the matrix");
}

/// SILENCE (0x10): a PCM block without data plays its samples as silence
/// and advances the decode counters; with data the flag changes nothing
///.
#[test]
fn ngs2_sampler_silence_flag_blocks() {
    const GRAIN: u32 = 256;
    let unused = 0i16;
    let silent = WaveformBlock { data_size: 0, num_samples: GRAIN, ..WaveformBlock::default() };
    let mut f = Fixture::new(48000, 1);
    f.queue_raw(addr(&unused), WAVEFORM_BLOCKS_FLAG_SILENCE, &[silent]);
    assert_eq!(f.voice().blocks.len(), 1, "the silent block was not queued");
    f.v().samples.fill(1.0);
    f.render(GRAIN);
    assert!(f.voice().samples[..GRAIN as usize].iter().all(|&s| s == 0.0));
    let s = f.sampler_state();
    assert_eq!((u64_at(&s, 24), u64_at(&s, 32)), (GRAIN as u64, GRAIN as u64 * 2));

    let pcm = vec![1234i16; GRAIN as usize];
    let mut f = Fixture::new(48000, 1);
    f.queue(pcm.as_ptr(), GRAIN, WAVEFORM_BLOCKS_FLAG_SILENCE, 0, 0);
    f.render(GRAIN);
    assert!(f.voice().samples[..GRAIN as usize].iter().all(|&s| s == 1234.0 / 32768.0));

    // Without the flag a dataless block is not queued.
    let mut f = Fixture::new(48000, 1);
    f.queue_raw(addr(&unused), 0, &[silent]);
    assert!(f.voice().blocks.is_empty());
}


/// A render writes one grain into each buffer and leaves the rest of a
/// larger buffer alone; a buffer that cannot take a grain is not touched.
#[test]
fn ngs2_render_writes_one_grain() {
    let mut eng = Engine::new();
    let mut env = HostEnv::default();
    let option = SystemOption::default();
    let grain = option.num_grain_samples as usize;
    eng.system_create(SYSTEM, option, ContextBuffer::default(), Allocator::default());
    for (waveform_type, bps) in [(0x18u32, 4usize), (0x12, 2)] {
        for channels in [1usize, 2] {
            let bytes = grain * channels * bps;
            let out = vec![0x7Bu8; bytes * 3];
            let info = RenderBuffer { buffer: addr(out.as_ptr()), size: out.len() as u64, waveform_type, num_channels: channels as u32 };
            assert_eq!(eng.system_render(&mut env, SYSTEM, &[info]), OK);
            assert!(out[..bytes].iter().all(|&b| b == 0), "the first grain is silence");
            assert!(out[bytes..].iter().all(|&b| b == 0x7B), "bytes past the grain were written");
        }
    }
    let mut out = vec![0x7Bu8; grain * 4];
    let short = RenderBuffer { buffer: addr(out.as_ptr()), size: (grain * 4 - 1) as u64, waveform_type: 0x18, num_channels: 1 };
    let wide = RenderBuffer { size: (grain * 4 * 9) as u64, num_channels: 9, ..short };
    let none = RenderBuffer { num_channels: 0, ..wide };
    assert_eq!(eng.system_render(&mut env, SYSTEM, &[short, wide, none, RenderBuffer::default()]), OK);
    assert!(out.iter().all(|&b| b == 0x7B), "an unusable buffer was written");
    out.clear();
}

/// SceNgs2CustomVoiceFilterParam body: type, channelMask, frequency, q, level, reserved.
fn custom_filter(ty: u32, frequency: f32) -> Vec<u8> {
    let mut b = Vec::new();
    for v in [ty, 0, frequency.to_bits(), 0.707_106_78f32.to_bits(), 1.0f32.to_bits(), 0] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

/// Give a custom rack one module of `module_id`.
fn set_custom_module(f: &mut Fixture, module_id: u32) {
    let rack = f.eng.rack_mut(f.vid.slot).unwrap();
    rack.option.bytes[184..188].copy_from_slice(&1u32.to_le_bytes());
    rack.option.bytes[200..204].copy_from_slice(&module_id.to_le_bytes());
    for v in rack.voices.iter_mut() {
        v.modules.resize(1, Module { flags: 1, ..Module::default() });
    }
}

#[test]
fn ngs2_custom_filter_module() {
    const GRAIN: u32 = 8;
    let mut f = Fixture::with_type(48000, 1, RACK_CUSTOM_SAMPLER);
    set_custom_module(&mut f, 0x13);
    let pcm = vec![16384i16; GRAIN as usize];
    f.queue(pcm.as_ptr(), GRAIN, 0, 0, 0);
    assert_eq!(f.control(&param(0x4000_1300, &custom_filter(1, 10000.0))), OK);
    // Index 1 is not a filter module: rejected.
    assert_eq!(f.control(&param(0x4000_1301, &custom_filter(1, 10000.0))), OK);
    assert_eq!(f.voice().filters.len(), 1, "a filter param for another module was applied");
    f.render(GRAIN);
    let s = f.voice().samples.clone();
    assert!(s[0] > 0.0 && s[0] < 0.2 && s[GRAIN as usize - 1] > s[0], "low-pass step response {s:?}");

    // Type 2 is not implemented: the filter stays as it was.
    assert_eq!(f.control(&param(0x4000_1300, &custom_filter(2, 10000.0))), OK);
    assert!(f.voice().filters[0].enabled);
    // Type 0 turns it off.
    assert_eq!(f.control(&param(0x4000_1300, &custom_filter(0, 0.0))), OK);
    assert!(!f.voice().filters[0].enabled);
    f.queue(pcm.as_ptr(), GRAIN, 4, 0, 0);
    f.render(GRAIN);
    assert!(f.voice().samples[0] > 0.4, "the disabled filter still ran");
}

/// Runs SceNgs2UserFxProcessContext handlers: fills channel 0 and 1 with
/// two levels and records the calls.
#[derive(Default)]
struct UserFxEnv {
    calls: u32,
    flags: u32,
    data: [u64; 3],
    levels: [f32; 2],
    result: u64,
}

impl Env for UserFxEnv {
    fn read(&mut self, addr: u64, out: &mut [u8]) -> bool {
        HostEnv::default().read(addr, out)
    }

    fn write(&mut self, addr: u64, data: &[u8]) -> bool {
        HostEnv::default().write(addr, data)
    }

    fn call(&mut self, func: u64, args: [u64; 6]) -> u64 {
        assert_eq!(func, 0x5555, "unexpected handler");
        let mut ctx = [0u8; 48];
        assert!(self.read(args[0], &mut ctx));
        let at32 = |o: usize| u32::from_le_bytes(ctx[o..o + 4].try_into().unwrap());
        let at64 = |o: usize| u64::from_le_bytes(ctx[o..o + 8].try_into().unwrap());
        self.calls += 1;
        self.flags = at32(32);
        self.data = [at64(8), at64(16), at64(24)];
        assert_eq!((at32(36), at32(40), at32(44)), (2, 8, 48000), "channels, grain, rate");
        let mut pointers = [0u8; 16];
        assert!(self.read(at64(0), &mut pointers));
        for c in 0..2 {
            let ptr = u64::from_le_bytes(pointers[c * 8..c * 8 + 8].try_into().unwrap());
            let level = self.levels[c].to_le_bytes();
            for i in 0..8 {
                assert!(self.write(ptr + i * 4, &level));
            }
        }
        self.result
    }

    fn voice_callback(&mut self, _eng: &mut Engine, _func: u64, _info: &CallbackInfo) {}
}

#[test]
fn ngs2_custom_user_fx_handler_module() {
    let mut eng = Engine::new();
    let mut env = UserFxEnv { levels: [0.25, 0.5], ..Default::default() };
    eng.system_create(SYSTEM, SystemOption::default(), ContextBuffer { host_buffer: SYSTEM, ..Default::default() }, Allocator::default());
    let mut option = RackOption::default_for(RACK_CUSTOM_SUBMIXER).unwrap();
    option.set_max_voices(2);
    option.bytes[184..188].copy_from_slice(&1u32.to_le_bytes());
    option.bytes[200..204].copy_from_slice(&0x1du32.to_le_bytes());
    option.bytes[220..224].copy_from_slice(&16u32.to_le_bytes());
    let handle = 0x2000_0000;
    let buffer = ContextBuffer { host_buffer: handle, size: 0x1000, user_data: 0 };
    let slot = eng.rack_create(&mut env, SYSTEM, RACK_CUSTOM_SUBMIXER, option, buffer, handle).unwrap();
    let vid = Vid { slot, idx: 0 };
    let control = |eng: &mut Engine, env: &mut UserFxEnv, p: &[u8]| eng.voice_control(env, vid, addr(p.as_ptr()));
    let mut setup = Vec::new();
    for v in [2u32, 2, 0, 0] {
        setup.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(control(&mut eng, &mut env, &param(0x4002_0000, &setup)), OK);
    assert_eq!(control(&mut eng, &mut env, &param(6, &1u32.to_le_bytes())), OK);
    let mut fx = Vec::new();
    for v in [0x5555u64, 0x1111, 123, 456] {
        fx.extend_from_slice(&v.to_le_bytes());
    }
    // A handler for a module index that is not 0x1d is rejected.
    assert_eq!(control(&mut eng, &mut env, &param(0x4000_1d02, &fx)), OK);
    let render = |eng: &mut Engine, env: &mut UserFxEnv| {
        eng.voice_mut(vid).unwrap().rendered = false;
        eng.render_voice_from(env, vid, &[], 8);
    };
    render(&mut eng, &mut env);
    assert_eq!(env.calls, 0, "a rejected handler ran");

    // The handler runs without any input and its output is the voice's.
    assert_eq!(control(&mut eng, &mut env, &param(0x4000_1d00, &fx)), OK);
    render(&mut eng, &mut env);
    assert_eq!((env.calls, env.flags, env.data), (1, 1, [0x1111, 123, 456]));
    let v = eng.voice(vid).unwrap();
    assert!(v.has_samples);
    assert!(v.samples[..8].iter().all(|&x| x == 0.25) && v.samples[8..].iter().all(|&x| x == 0.5));
    render(&mut eng, &mut env);
    assert_eq!((env.calls, env.flags), (2, 0), "flags clear after the first call");

    // Silence from the handler leaves the voice without samples.
    env.levels = [0.0, 0.0];
    render(&mut eng, &mut env);
    assert!(!eng.voice(vid).unwrap().has_samples);
}

/// A custom mastering voice takes its channel count (setup, id 0) and
/// output (id 1) from its parameters and mixes into that render buffer.
#[test]
fn ngs2_custom_mastering_voice_renders_to_its_output() {
    let mut eng = Engine::new();
    let mut env = HostEnv::default();
    let option = SystemOption { num_grain_samples: 4, ..SystemOption::default() };
    eng.system_create(SYSTEM, option, ContextBuffer::default(), Allocator::default());
    let mut sampler_option = RackOption::default_for(RACK_SAMPLER).unwrap();
    sampler_option.set_max_voices(1);
    sampler_option.set_max_ports(1);
    let mut mastering_option = RackOption::default_for(RACK_CUSTOM_MASTERING).unwrap();
    mastering_option.set_max_voice_channels(2);
    let (sampler_h, mastering_h) = (0x3000_0000u64, 0x3010_0000u64);
    let s_slot = eng.rack_create(&mut env, SYSTEM, RACK_SAMPLER, sampler_option, ContextBuffer::default(), sampler_h).unwrap();
    let m_slot = eng.rack_create(&mut env, SYSTEM, RACK_CUSTOM_MASTERING, mastering_option, ContextBuffer::default(), mastering_h).unwrap();
    let mastering = Vid { slot: m_slot, idx: 0 };
    let control = |eng: &mut Engine, env: &mut HostEnv, p: &[u8]| eng.voice_control(env, mastering, addr(p.as_ptr()));
    let words = |v: [u32; 2]| v.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
    // Setups outside 1..=maxVoiceChannels or with flags are refused.
    assert_eq!(control(&mut eng, &mut env, &param(0x4003_0000, &words([1, 0]))), OK);
    for bad in [[0, 0], [3, 0], [2, 1]] {
        assert_eq!(control(&mut eng, &mut env, &param(0x4003_0000, &words(bad))), OK);
        assert_eq!(eng.voice(mastering).unwrap().channels, 1, "{bad:?}");
    }
    assert_eq!(control(&mut eng, &mut env, &param(0x4003_0000, &words([2, 0]))), OK);
    assert_eq!(eng.voice(mastering).unwrap().channels, 2);
    assert_eq!(control(&mut eng, &mut env, &param(0x4003_0001, &words([1, 0]))), OK);
    assert_eq!(eng.voice(mastering).unwrap().output_id, 1);
    assert_eq!(control(&mut eng, &mut env, &param(6, &1u32.to_le_bytes())), OK);

    let source: [i16; 4] = [16384, -16384, 8192, -8192];
    let mastering_handle = eng.voice_handle(mastering);
    let v = eng.voice_mut(Vid { slot: s_slot, idx: 0 }).unwrap();
    v.setup_sampler(&WaveformFormat::new(0x12, 1, 48000, 0));
    v.blocks.push_back(Block::new(addr(source.as_ptr()), WaveformBlock { data_size: 8, num_samples: 4, ..WaveformBlock::default() }));
    v.ports[0].dest = mastering_handle;
    v.set_event(1);
    let (mut first, mut second) = ([1.0f32; 8], [1.0f32; 8]);
    let out = [
        RenderBuffer { buffer: addr(first.as_mut_ptr()), size: 32, waveform_type: 0x18, num_channels: 2 },
        RenderBuffer { buffer: addr(second.as_mut_ptr()), size: 32, waveform_type: 0x18, num_channels: 2 },
    ];
    assert_eq!(eng.system_render(&mut env, SYSTEM, &out), OK);
    assert_eq!(first, [0.0; 8], "output 0 is silent");
    // Mono into stereo through the default channel map: both channels at -3 dB.
    let h = std::f32::consts::FRAC_1_SQRT_2;
    for (frame, x) in second.chunks_exact(2).zip([0.5f32, -0.5, 0.25, -0.25]) {
        assert!(approx(frame[0], x * h, 1e-6) && approx(frame[1], x * h, 1e-6), "{second:?}");
    }
}

/// A user effect handler that multiplies channel 0 by channel 1 (the
/// side-chain buffer's first channel) and records its flags.
#[derive(Default)]
struct SideChainEnv {
    flags: u32,
    calls: u32,
}

impl Env for SideChainEnv {
    fn read(&mut self, addr: u64, out: &mut [u8]) -> bool {
        HostEnv::default().read(addr, out)
    }

    fn write(&mut self, addr: u64, data: &[u8]) -> bool {
        HostEnv::default().write(addr, data)
    }

    fn call(&mut self, func: u64, args: [u64; 6]) -> u64 {
        assert_eq!(func, 0x7777, "unexpected handler");
        let mut ctx = [0u8; 48];
        assert!(self.read(args[0], &mut ctx));
        let at32 = |o: usize| u32::from_le_bytes(ctx[o..o + 4].try_into().unwrap());
        let at64 = |o: usize| u64::from_le_bytes(ctx[o..o + 8].try_into().unwrap());
        assert_eq!((at32(36), at32(40), at64(8)), (1, 1, 7), "channels, grain, user data");
        self.calls += 1;
        self.flags = at32(32);
        let mut pointers = [0u8; 16];
        assert!(self.read(at64(0), &mut pointers));
        let (main, side) = (u64::from_le_bytes(pointers[..8].try_into().unwrap()), u64::from_le_bytes(pointers[8..].try_into().unwrap()));
        let mut a = [0u8; 4];
        let mut b = [0u8; 4];
        assert!(self.read(main, &mut a) && self.read(side, &mut b));
        let product = f32::from_le_bytes(a) * f32::from_le_bytes(b);
        assert!(self.write(main, &product.to_le_bytes()));
        0
    }

    fn voice_callback(&mut self, _eng: &mut Engine, _func: u64, _info: &CallbackInfo) {}
}

/// A custom submixer with two buffers: ports feed the buffer of their input
/// and the user effect gets the source buffer's channels followed by the
/// extra (side-chain) buffer's.
#[test]
fn ngs2_custom_side_chain_user_fx() {
    let mut eng = Engine::new();
    let mut env = SideChainEnv::default();
    eng.system_create(SYSTEM, SystemOption::default(), ContextBuffer { host_buffer: SYSTEM, ..Default::default() }, Allocator::default());
    let mut sampler_option = RackOption::default_for(RACK_SAMPLER).unwrap();
    sampler_option.set_max_voices(2);
    sampler_option.set_max_ports(1);
    let s_slot = eng.rack_create(&mut env, SYSTEM, RACK_SAMPLER, sampler_option, ContextBuffer::default(), 0x3000_0000).unwrap();
    let mut option = RackOption::default_for(RACK_CUSTOM_SUBMIXER).unwrap();
    option.set_max_voices(1);
    option.set_custom_num_buffers(2);
    option.bytes[184..188].copy_from_slice(&1u32.to_le_bytes()); // numModules
    option.bytes[200..204].copy_from_slice(&0x1du32.to_le_bytes()); // USER_FX
    option.bytes[208..212].copy_from_slice(&1u32.to_le_bytes()); // extraBufferId
    let c_handle = 0x3010_0000;
    let c_slot = eng.rack_create(&mut env, SYSTEM, RACK_CUSTOM_SUBMIXER, option, ContextBuffer { host_buffer: c_handle, size: 0x1000, user_data: 0 }, c_handle).unwrap();
    let custom = Vid { slot: c_slot, idx: 0 };
    let control = |eng: &mut Engine, env: &mut SideChainEnv, p: &[u8]| eng.voice_control(env, custom, addr(p.as_ptr()));
    let setup: Vec<u8> = [1u32, 1, 0, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(control(&mut eng, &mut env, &param(0x4002_0000, &setup)), OK);
    let fx: Vec<u8> = [0x7777u64, 7, 0, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(control(&mut eng, &mut env, &param(0x4000_1d00, &fx)), OK);
    assert_eq!(control(&mut eng, &mut env, &param(6, &1u32.to_le_bytes())), OK);

    let custom_handle = eng.voice_handle(custom);
    let pcm = [[8192i16], [16384]];
    let mut sources = Vec::new();
    for (i, src) in pcm.iter().enumerate() {
        let vid = Vid { slot: s_slot, idx: i };
        let v = eng.voice_mut(vid).unwrap();
        v.setup_sampler(&WaveformFormat::new(0x12, 1, 48000, 0));
        v.blocks.push_back(Block::new(addr(src.as_ptr()), WaveformBlock { data_size: 2, num_samples: 1, ..WaveformBlock::default() }));
        v.ports[0].dest = custom_handle;
        v.ports[0].input = i as u32;
        v.set_event(1);
        sources.push(vid);
    }
    eng.render_voice_from(&mut env, custom, &sources, 1);
    let v = eng.voice(custom).unwrap();
    assert_eq!(env.calls, 1);
    assert!(v.has_samples && v.samples == [0.125, 0.5] && env.flags == 1, "user FX did not receive the side-chain buffer after the source channels: {:?}", v.samples);
}

/// The mastering gain param (0x30000004) scales the full-band channels and
/// the LFE channel (3) of a 6-channel voice from the next render; each
/// factor is clamped to [-32, 32] with a NaN kept, and a setup resets both
/// to 1 (measured on a PS5, firmware 3.20 and 4.03).
#[test]
fn ngs2_mastering_gain_is_clamped_and_reset_by_setup() {
    let mut eng = Engine::new();
    let mut env = HostEnv::default();
    let option = SystemOption { num_grain_samples: 4, ..SystemOption::default() };
    eng.system_create(SYSTEM, option, ContextBuffer::default(), Allocator::default());
    let mut sampler_option = RackOption::default_for(RACK_SAMPLER).unwrap();
    sampler_option.set_max_voices(1);
    sampler_option.set_max_ports(1);
    let mut mastering_option = RackOption::default_for(RACK_MASTERING).unwrap();
    mastering_option.set_max_voice_channels(8);
    let (sampler_h, mastering_h) = (0x3100_0000u64, 0x3110_0000u64);
    let s_slot = eng.rack_create(&mut env, SYSTEM, RACK_SAMPLER, sampler_option, ContextBuffer::default(), sampler_h).unwrap();
    let m_slot = eng.rack_create(&mut env, SYSTEM, RACK_MASTERING, mastering_option, ContextBuffer::default(), mastering_h).unwrap();
    let mastering = Vid { slot: m_slot, idx: 0 };
    let control = |eng: &mut Engine, env: &mut HostEnv, p: &[u8]| eng.voice_control(env, mastering, addr(p.as_ptr()));
    assert_eq!(control(&mut eng, &mut env, &param(0x3000_0000, &6u32.to_le_bytes())), OK);
    assert_eq!(control(&mut eng, &mut env, &param(6, &1u32.to_le_bytes())), OK);

    // A 6-channel stream at 0.5 on every channel, long enough for every render.
    let source = vec![16384i16; 6 * 4 * 16];
    let mastering_handle = eng.voice_handle(mastering);
    let v = eng.voice_mut(Vid { slot: s_slot, idx: 0 }).unwrap();
    v.setup_sampler(&WaveformFormat::new(0x12, 6, 48000, 0));
    let block = WaveformBlock { data_size: source.len() as u64 * 2, num_samples: 4 * 16, ..WaveformBlock::default() };
    v.blocks.push_back(Block::new(addr(source.as_ptr()), block));
    v.ports[0].dest = mastering_handle;
    v.set_event(1);
    let render = |eng: &mut Engine, env: &mut HostEnv| {
        let mut out = [-1.0f32; 24];
        let buffers = [RenderBuffer { buffer: addr(out.as_mut_ptr()), size: 96, waveform_type: 0x18, num_channels: 6 }];
        assert_eq!(eng.system_render(env, SYSTEM, &buffers), OK);
        out
    };
    let expect = |s: [f32; 24], fbw: f32, lfe: f32| {
        for (i, &x) in s.iter().enumerate() {
            let want = 0.5 * if i % 6 == 3 { lfe } else { fbw };
            assert!(if want.is_nan() { x.is_nan() } else { x == want }, "sample {i}: {x} != {want}");
        }
    };
    expect(render(&mut eng, &mut env), 1.0, 1.0);
    let gain = |eng: &mut Engine, env: &mut HostEnv, fbw: f32, lfe: f32| {
        let body: Vec<u8> = [fbw.to_bits(), lfe.to_bits()].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(control(eng, env, &param(0x3000_0004, &body)), OK);
    };
    for (fbw, lfe, fbw_out, lfe_out) in [
        (0.5, 0.25, 0.5, 0.25),
        (33.0, -1e30, 32.0, -32.0),
        (f32::NEG_INFINITY, f32::INFINITY, -32.0, 32.0),
        (f32::NAN, 2.0, f32::NAN, 2.0),
    ] {
        gain(&mut eng, &mut env, fbw, lfe);
        expect(render(&mut eng, &mut env), fbw_out, lfe_out);
    }
    gain(&mut eng, &mut env, 0.5, 0.25);
    assert_eq!(control(&mut eng, &mut env, &param(0x3000_0000, &6u32.to_le_bytes())), OK);
    expect(render(&mut eng, &mut env), 1.0, 1.0);
}
