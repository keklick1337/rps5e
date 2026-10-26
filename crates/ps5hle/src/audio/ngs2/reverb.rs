// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Reverb rack voices (I3DL2 parameters), as the console's libSceNgs2
//! renders them.
//!
//! The input is mixed to mono with fixed gains and written into a
//! 14400-sample ring; four early reflection taps (from the reflection
//! pattern tables) and one late tap read it. The late feed runs through
//! sixteen damped comb filters with a pre-emphasis stage; every output
//! channel takes a signed selection of the combs through four allpasses,
//! adds the early reflections and passes a two-pole room HF filter. The LFE
//! channel passes dry. Parameter changes crossfade the taps over one grain;
//! wet and dry levels ramp over one grain.

/// Samples of the input ring the taps read.
const TAP_RING_SAMPLES: usize = 14400;
const EARLY_TAPS: usize = 4;
const COMBS: usize = 16;
const ALLPASSES: usize = 4;
const LFE_CHANNEL: usize = 3;
const MAX_CHANNELS: usize = 8;

/// Reflection pattern tap times (ms) and gains.
const TAP_TIMES: [[f32; EARLY_TAPS]; 8] = [
    [0.0, 11.0, 23.0, 35.0],
    [0.0, 14.75, 26.5499992, 38.0299988],
    [0.0, 6.0, 17.0, 29.0],
    [0.0, 8.0, 20.0, 53.0],
    [0.0, 11.0, 17.0, 30.0],
    [0.0, 16.0, 30.0, 55.0],
    [0.0, 20.0, 29.0, 40.0],
    [0.0, 37.0, 47.0, 62.0],
];
const TAP_GAINS: [[f32; EARLY_TAPS]; 8] = [
    [0.7, 0.4, 0.134, 0.1],
    [0.7, 0.4, 0.3, 0.1],
    [0.7, 0.45, -0.6, 0.2],
    [0.7, 0.55, -0.6, 0.2],
    [0.7, -0.45, 0.6, 0.3],
    [0.7, 0.7, 0.7, 0.4],
    [0.7, 0.7, -0.4, 0.3],
    [0.7, 0.7, -0.4, 0.4],
];
/// Comb delays (ms) at density 100.
const COMB_TIMES: [f32; COMBS] = [
    22.0100002, 23.25, 24.75, 26.6000004, 28.25, 29.6299992, 31.0599995, 32.4399986, 33.6899986, 34.9500008, 36.2900009,
    37.6500015, 38.9900017, 40.3499985, 41.6699982, 43.0499992,
];
const ALLPASS_TIMES: [f32; ALLPASSES] = [11.5799999, 9.18999958, 7.0999999, 4.69000006];
/// Per-channel allpass length offsets (samples).
const ALLPASS_OFFSETS: [[i32; ALLPASSES]; MAX_CHANNELS] = [
    [0, -32, -6, -11],
    [-10, -5, 0, -34],
    [-8, -3, -33, -5],
    [-7, -9, -5, -8],
    [-33, -8, -8, 0],
    [-11, -21, -10, -7],
    [-9, -10, -27, -3],
    [-6, -11, -9, -24],
];
/// Combs each output channel takes (bit n = comb n).
const CHANNEL_COMBS: [u16; MAX_CHANNELS] = [0xaaaa, 0x5555, 0xcccc, 0x3333, 0x0ff0, 0x9ab2, 0x3c3c, 0x399c];

/// sizeof(SceNgs2ReverbI3DL2Param).
pub(crate) const I3DL2_PARAM_SIZE: usize = 88;

/// SceNgs2ReverbI3DL2Param.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct I3dl2Param {
    pub wet: f32,
    pub dry: f32,
    pub room: i32,
    pub room_hf: i32,
    pub reflection_pattern: u32,
    pub decay_time: f32,
    pub decay_hf_ratio: f32,
    pub reflections: i32,
    pub reflections_delay: f32,
    pub reverb: i32,
    pub reverb_delay: f32,
    pub diffusion: f32,
    pub density: f32,
    pub hf_reference: f32,
}

impl I3dl2Param {
    /// The parameters a reverb voice gets at setup.
    pub(crate) const SETUP: Self = Self {
        wet: 1.0,
        dry: 0.0,
        room: -10000,
        room_hf: 0,
        reflection_pattern: 0,
        decay_time: 1.0,
        decay_hf_ratio: 0.5,
        reflections: -10000,
        reflections_delay: 0.02,
        reverb: -10000,
        reverb_delay: 0.04,
        diffusion: 100.0,
        density: 100.0,
        hf_reference: 5000.0,
    };

    pub(crate) fn parse(b: &[u8]) -> Self {
        let f = |o: usize| b.get(o..o + 4).map_or(0.0, |s| f32::from_le_bytes(s.try_into().unwrap()));
        let i = |o: usize| b.get(o..o + 4).map_or(0, |s| i32::from_le_bytes(s.try_into().unwrap()));
        Self {
            wet: f(0),
            dry: f(4),
            room: i(8),
            room_hf: i(12),
            reflection_pattern: i(16) as u32,
            decay_time: f(20),
            decay_hf_ratio: f(24),
            reflections: i(28),
            reflections_delay: f(32),
            reverb: i(36),
            reverb_delay: f(40),
            diffusion: f(44),
            density: f(48),
            hf_reference: f(52),
        }
    }

    #[cfg(test)]
    pub(crate) fn to_bytes(self) -> [u8; I3DL2_PARAM_SIZE] {
        let mut b = [0u8; I3DL2_PARAM_SIZE];
        let words = [
            self.wet.to_bits(),
            self.dry.to_bits(),
            self.room as u32,
            self.room_hf as u32,
            self.reflection_pattern,
            self.decay_time.to_bits(),
            self.decay_hf_ratio.to_bits(),
            self.reflections as u32,
            self.reflections_delay.to_bits(),
            self.reverb as u32,
            self.reverb_delay.to_bits(),
            self.diffusion.to_bits(),
            self.density.to_bits(),
            self.hf_reference.to_bits(),
        ];
        for (k, w) in words.iter().enumerate() {
            b[k * 4..k * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        b
    }
}

/// Why the reverb cannot run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReverbError {
    /// A NaN parameter.
    InvalidParameter,
    /// A sample rate so low an allpass would have no length.
    UnsupportedSampleRate(u32),
}

#[derive(Clone, Copy, Default)]
struct Taps {
    delays: [u32; EARLY_TAPS + 1],
    gains: [f32; EARLY_TAPS + 1],
}

#[derive(Clone, Default)]
struct Comb {
    buffer: Vec<f32>,
    cursor: usize,
    feedback: f32,
    lowpass_in: f32,
    lowpass_back: f32,
    direct: f32,
    filtered: f32,
    emphasis: f32,
    emphasis_back: f32,
    previous: f32,
    lowpass: f32,
}

#[derive(Clone, Default)]
struct Allpass {
    buffer: Vec<f32>,
    cursor: usize,
}

#[derive(Clone, Default)]
struct Channel {
    allpasses: [Allpass; ALLPASSES],
    y1: f32,
    y2: f32,
    wet: f32,
    dry: f32,
}

/// Buffers one grain needs, kept between grains.
#[derive(Default)]
struct Scratch {
    input: Vec<f32>,
    mono: Vec<f32>,
    early: Vec<f32>,
    late: Vec<f32>,
    old_early: Vec<f32>,
    old_late: Vec<f32>,
    combs: Vec<f32>,
    tail: Vec<f32>,
}

pub(crate) struct ReverbState {
    params: I3dl2Param,
    reset: bool,
    changed: bool,
    sample_rate: u32,
    ring: Vec<f32>,
    ring_cursor: usize,
    taps: Taps,
    early_gain: f32,
    late_gain: f32,
    density: f32,
    combs: Vec<Comb>,
    channels: Vec<Channel>,
    room_filter: bool,
    room_b0: f32,
    room_a1: f32,
    room_a2: f32,
    scratch: Scratch,
}

/// A level in millibels as a gain; below -100 dB nothing.
fn millibels(mb: f32) -> f32 {
    if mb < -10000.0 {
        return 0.0;
    }
    10f64.powf((mb * 0.0005) as f64) as f32
}

/// `value` into `low..=high` the way the library does (NaN stays NaN).
fn clamp(value: f32, low: f32, high: f32) -> f32 {
    if value < low {
        low
    } else if high < value {
        high
    } else {
        value
    }
}

fn clamp_millibels(value: i32, high: i32) -> i32 {
    value.min(high).max(-10000)
}

/// `a` unless `b` is smaller (NaN `b` gives `a`).
fn min_of(a: f32, b: f32) -> f32 {
    if b < a {
        b
    } else {
        a
    }
}

fn samples_for(milliseconds: f32, rate: f32) -> u32 {
    (rate * milliseconds * 0.001 + 0.5) as u32
}

/// samples[i] = samples[i] * gain, the gain ramping from `from` to `to`.
fn ramp_in_place(samples: &mut [f32], from: f32, to: f32) {
    let step = (to - from) / samples.len() as f32;
    let mut gain = from;
    for s in samples.iter_mut() {
        *s *= if from == to { to } else { gain };
        gain += step;
    }
}

/// samples[i] += source[i] * gain, the gain ramping from `from` to `to`.
fn ramp_add(samples: &mut [f32], from: f32, to: f32, source: &[f32]) {
    let step = (to - from) / samples.len() as f32;
    let mut gain = from;
    for (s, &x) in samples.iter_mut().zip(source) {
        *s += x * if from == to { to } else { gain };
        gain += step;
    }
}

fn taps_for(params: &I3dl2Param, size: u32, rate: f32) -> Taps {
    let mut taps = Taps::default();
    let pattern = if params.reflection_pattern < 8 { params.reflection_pattern } else { params.reflection_pattern - 8 } as usize;
    let early = ((params.reflections_delay * rate * 1000.0 * 0.001 + 0.5) as u32).min(size);
    let late = (((params.reflections_delay + params.reverb_delay) * rate * 1000.0 * 0.001 + 0.5) as u32).min(size);
    let last = (rate * TAP_TIMES[pattern][EARLY_TAPS - 1] * 0.001 + 0.5) as u32;
    let limit = if params.reflection_pattern > 7 { size } else { late };
    let scale = if early.wrapping_add(last) > limit { limit.wrapping_sub(early) as f32 / last as f32 } else { 1.0 };
    taps.delays[0] = early;
    taps.gains[0] = TAP_GAINS[pattern][0];
    for tap in 1..EARLY_TAPS {
        taps.delays[tap] = ((scale * TAP_TIMES[pattern][tap] * rate * 0.001 + 0.5) as u32).wrapping_add(early);
        taps.gains[tap] = if taps.delays[tap] == taps.delays[tap - 1] { 0.0 } else { TAP_GAINS[pattern][tap] };
    }
    taps.delays[EARLY_TAPS] = early.wrapping_add((params.reverb_delay * rate * 1000.0 * 0.001 + 0.5) as u32).min(size);
    taps.gains[EARLY_TAPS] = 1.0;
    taps
}

fn setup_comb(comb: &mut Comb, length: usize, params: &I3dl2Param, rate: f32, clear: bool) {
    if clear || comb.buffer.len() != length {
        comb.buffer.clear();
        comb.buffer.resize(length, 0.0);
        if comb.cursor >= length {
            comb.cursor = 0;
        }
    }
    comb.feedback = 10f32.powf(length as f32 / rate / params.decay_time * -3.0);
    let ratio = params.decay_hf_ratio;
    if ratio > 1.0 {
        let high = 10f32.powf((ratio - 1.0) * comb.feedback.log10());
        comb.lowpass_in = 0.0506209731;
        comb.lowpass_back = 0.949379027;
        comb.direct = 1.0;
        comb.filtered = 1.0 - high;
    } else {
        let mut shaped = ratio * ratio;
        if 0.8 > ratio {
            shaped = (0.8 - ratio) * -0.857142925 + 0.8;
            shaped *= shaped;
        }
        let back = min_of(0.99977994, comb.feedback.log10() * 2.30258512 * 0.25 * (1.0 - 1.0 / shaped));
        comb.lowpass_in = 1.0 - back;
        comb.lowpass_back = back;
        comb.direct = 0.0;
        comb.filtered = 1.0;
    }
    let root = ratio.sqrt();
    let tilt = (1.0 - root) / (root + 1.0);
    comb.emphasis = tilt / (1.0 - tilt) + 1.0;
    comb.emphasis_back = -tilt / (1.0 - tilt);
}

fn run_comb(comb: &mut Comb, input: &[f32], output: &mut [f32]) {
    let length = comb.buffer.len();
    for (o, &x) in output.iter_mut().zip(input) {
        let emphasized = x * comb.emphasis + comb.previous * comb.emphasis_back;
        comb.previous = x;
        let delayed = comb.buffer[comb.cursor];
        comb.lowpass = delayed * comb.lowpass_in + comb.lowpass * comb.lowpass_back;
        let back = (delayed * comb.direct + comb.lowpass * comb.filtered) * comb.feedback;
        comb.buffer[comb.cursor] = emphasized + back;
        *o = back;
        comb.cursor = if comb.cursor + 1 == length { 0 } else { comb.cursor + 1 };
    }
}

fn run_allpass(allpass: &mut Allpass, gain: f32, samples: &mut [f32]) {
    let length = allpass.buffer.len();
    for s in samples.iter_mut() {
        let delayed = allpass.buffer[allpass.cursor];
        let stored = delayed * gain + *s;
        allpass.buffer[allpass.cursor] = stored;
        *s = delayed - gain * stored;
        allpass.cursor = if allpass.cursor + 1 == length { 0 } else { allpass.cursor + 1 };
    }
}

/// Allpass length of `stage` on `channel`; None when the rate is too low.
fn allpass_length(channel: usize, stage: usize, rate: f32) -> Option<usize> {
    let signed = samples_for(ALLPASS_TIMES[stage], rate) as i32 + ALLPASS_OFFSETS[channel][stage];
    (signed >= 1).then_some(signed as usize)
}

/// Every channel's input mixed into `mono` with the library's gains (the
/// LFE channel left out).
fn mix_down(input: &[f32], channels: usize, grain: usize, mono: &mut [f32]) {
    if channels == 1 {
        mono.copy_from_slice(&input[..grain]);
        return;
    }
    let gain = match channels {
        2 => 0.707000017,
        6 => 0.316000015,
        _ => 0.26699999,
    };
    for (m, &x) in mono.iter_mut().zip(&input[..grain]) {
        *m = x * gain;
    }
    for c in 1..channels {
        if c == LFE_CHANNEL {
            continue;
        }
        for (m, &x) in mono.iter_mut().zip(&input[c * grain..(c + 1) * grain]) {
            *m += x * gain;
        }
    }
}

impl ReverbState {
    /// A voice's reverb right after setup.
    pub(crate) fn new() -> Self {
        let mut r = Self {
            params: I3dl2Param::default(),
            reset: true,
            changed: true,
            sample_rate: 0,
            ring: Vec::new(),
            ring_cursor: 0,
            taps: Taps::default(),
            early_gain: 0.0,
            late_gain: 0.0,
            density: f32::NAN,
            combs: vec![Comb::default(); COMBS],
            channels: vec![Channel::default(); MAX_CHANNELS],
            room_filter: false,
            room_b0: 0.0,
            room_a1: 0.0,
            room_a2: 0.0,
            scratch: Scratch::default(),
        };
        // The setup parameters hold no NaN.
        let _ = r.set_params(&I3dl2Param::SETUP);
        r
    }

    #[cfg(test)]
    pub(crate) fn params(&self) -> &I3dl2Param {
        &self.params
    }

    /// Store new parameters, clamped as the library clamps them; NaN
    /// parameters are rejected.
    pub(crate) fn set_params(&mut self, p: &I3dl2Param) -> Result<(), ReverbError> {
        let floats = [
            p.wet,
            p.dry,
            p.decay_time,
            p.decay_hf_ratio,
            p.reflections_delay,
            p.reverb_delay,
            p.diffusion,
            p.density,
            p.hf_reference,
        ];
        if floats.iter().any(|v| v.is_nan()) {
            return Err(ReverbError::InvalidParameter);
        }
        self.params = I3dl2Param {
            wet: clamp(p.wet, 0.0, 32.0),
            dry: clamp(p.dry, 0.0, 32.0),
            reflection_pattern: if p.reflection_pattern < 16 { p.reflection_pattern } else { 0 },
            decay_time: clamp(p.decay_time, 0.1, 20.0),
            decay_hf_ratio: clamp(p.decay_hf_ratio, 0.00999999978, 2.0),
            reflections_delay: clamp(p.reflections_delay, 0.0, 0.300000012),
            reverb_delay: clamp(p.reverb_delay, 0.0, 0.100000001),
            diffusion: clamp(p.diffusion, 0.0, 100.0),
            density: clamp(p.density, 0.0, 100.0),
            hf_reference: clamp(p.hf_reference, 20.0, 20000.0),
            ..*p
        };
        self.changed = true;
        Ok(())
    }

    /// Stop: the next grain starts from silence.
    pub(crate) fn clear(&mut self) {
        self.reset = true;
    }

    fn reset_state(&mut self) {
        self.ring.clear();
        self.ring.resize(TAP_RING_SAMPLES, 0.0);
        self.ring_cursor = 0;
        self.taps = Taps::default();
        self.early_gain = 1.0;
        self.late_gain = 1.0;
        self.density = f32::NAN;
        self.combs.iter_mut().for_each(|c| *c = Comb::default());
        self.channels.iter_mut().for_each(|c| *c = Channel::default());
    }

    fn apply_params(&mut self, rate: f32) {
        let params = self.params;
        self.taps = taps_for(&params, self.ring.len() as u32, rate);
        let room = min_of(1.0, millibels(params.room as f32));
        self.early_gain = millibels(clamp_millibels(params.reflections, 1000) as f32) * room;
        self.late_gain = millibels((clamp_millibels(params.reverb, 2000) - 2400) as f32) * room;
        let density_changed = self.density != params.density;
        self.density = params.density;
        let density_scale = params.density * -3.0 / 100.0 + 4.0;
        for (i, comb) in self.combs.iter_mut().enumerate() {
            let mut length = (rate * COMB_TIMES[i] / density_scale * 0.001 + 0.5) as u32;
            length += !length & 1;
            setup_comb(comb, length as usize, &params, rate, density_changed);
        }
        for (channel, state) in self.channels.iter_mut().enumerate() {
            for (stage, allpass) in state.allpasses.iter_mut().enumerate() {
                // The rate was checked before any state changed.
                let length = allpass_length(channel, stage, rate).unwrap_or(1);
                if allpass.buffer.len() != length {
                    allpass.buffer.clear();
                    allpass.buffer.resize(length, 0.0);
                }
                if allpass.cursor >= length {
                    allpass.cursor = 0;
                }
            }
        }
    }

    fn update_room_filter(&mut self, rate: f32) {
        let p = &self.params;
        let room_hf = millibels(clamp_millibels(p.room_hf, 0) as f32);
        self.room_filter = 1.0 > room_hf;
        if !self.room_filter {
            return;
        }
        let cosine = (p.hf_reference * 6.28318548 / rate).cos();
        let root = ((room_hf + room_hf) * (1.0 - cosine) - room_hf * room_hf * (1.0 - cosine * cosine)).sqrt();
        let pole = ((1.0 - room_hf * cosine) - root) / (1.0 - room_hf);
        self.room_b0 = (1.0 - pole) * (1.0 - pole);
        self.room_a1 = pole + pole;
        self.room_a2 = pole * -pole;
    }

    /// The early (`early`) and late (`late`) taps of one grain of `input`.
    fn run_taps(&self, taps: &Taps, input: &[f32], early: &mut [f32], late: &mut [f32]) {
        let size = self.ring.len();
        early.fill(0.0);
        late.fill(0.0);
        for tap in 0..=EARLY_TAPS {
            let out: &mut [f32] = if tap < EARLY_TAPS { &mut *early } else { &mut *late };
            let gain = taps.gains[tap];
            let delay = taps.delays[tap] as usize;
            let mut read = self.ring_cursor.wrapping_add(size).wrapping_sub(delay) % size;
            for (i, o) in out.iter_mut().enumerate() {
                if i < delay {
                    *o += self.ring[read] * gain;
                    read = if read + 1 == size { 0 } else { read + 1 };
                } else {
                    *o += input[i - delay] * gain;
                }
            }
        }
    }

    /// Run one grain over `samples` (`channels` runs of `grain` samples) in
    /// place; whether any output sample is not zero.
    pub(crate) fn process(&mut self, samples: &mut [f32], channels: usize, grain: usize, sample_rate: u32) -> Result<bool, ReverbError> {
        let rate = sample_rate as f32;
        let channels = channels.min(MAX_CHANNELS);
        if grain == 0 || channels == 0 || samples.len() < channels * grain {
            return Ok(false);
        }
        if (0..MAX_CHANNELS).any(|c| (0..ALLPASSES).any(|s| allpass_length(c, s, rate).is_none())) {
            return Err(ReverbError::UnsupportedSampleRate(sample_rate));
        }
        if self.sample_rate != sample_rate {
            self.sample_rate = sample_rate;
            self.reset = true;
        }
        let reset = self.reset;
        if reset {
            self.reset_state();
            self.changed = true;
        }
        let mut s = std::mem::take(&mut self.scratch);
        s.input.clear();
        s.input.extend_from_slice(&samples[..channels * grain]);
        for v in [&mut s.mono, &mut s.early, &mut s.late, &mut s.old_early, &mut s.old_late, &mut s.tail] {
            v.clear();
            v.resize(grain, 0.0);
        }
        mix_down(&s.input, channels, grain, &mut s.mono);
        if self.changed {
            let taps = self.taps;
            self.run_taps(&taps, &s.mono, &mut s.old_early, &mut s.old_late);
            self.apply_params(rate);
            let taps = self.taps;
            self.run_taps(&taps, &s.mono, &mut s.early, &mut s.late);
            ramp_in_place(&mut s.early, 0.0, 1.0);
            ramp_add(&mut s.early, 1.0, 0.0, &s.old_early);
            ramp_in_place(&mut s.late, 0.0, 1.0);
            ramp_add(&mut s.late, 1.0, 0.0, &s.old_late);
        } else {
            let taps = self.taps;
            self.run_taps(&taps, &s.mono, &mut s.early, &mut s.late);
        }
        let size = self.ring.len();
        for &m in &s.mono {
            self.ring[self.ring_cursor] = m;
            self.ring_cursor = if self.ring_cursor + 1 == size { 0 } else { self.ring_cursor + 1 };
        }
        s.combs.clear();
        s.combs.resize(COMBS * grain, 0.0);
        for (comb, out) in self.combs.iter_mut().zip(s.combs.chunks_exact_mut(grain)) {
            run_comb(comb, &s.late, out);
        }
        self.update_room_filter(rate);
        let params = self.params;
        let mut audible = false;
        for channel in 0..channels {
            let out = &mut samples[channel * grain..(channel + 1) * grain];
            let input = &s.input[channel * grain..(channel + 1) * grain];
            let state = &mut self.channels[channel];
            if channel == LFE_CHANNEL {
                out.copy_from_slice(input);
            } else {
                s.tail.fill(0.0);
                let mut selected = 0u32;
                for (i, comb) in s.combs.chunks_exact(grain).enumerate() {
                    if (CHANNEL_COMBS[channel] >> i) & 1 == 0 {
                        continue;
                    }
                    let weight = if selected & 1 != 0 {
                        1.0
                    } else {
                        1.0 / (selected as f32 * 0.5 * 20.0 + 40.0) * (params.density + -100.0) + 1.0
                    };
                    for (t, &c) in s.tail.iter_mut().zip(comb) {
                        *t += c * weight;
                    }
                    selected += 1;
                }
                for allpass in state.allpasses.iter_mut() {
                    run_allpass(allpass, params.diffusion * 0.005, &mut s.tail);
                }
                for ((o, &e), &t) in out.iter_mut().zip(&s.early).zip(&s.tail) {
                    *o = e * self.early_gain + t * self.late_gain;
                }
                for o in out.iter_mut() {
                    let y = if self.room_filter {
                        self.room_b0 * *o + self.room_a1 * state.y1 + self.room_a2 * state.y2
                    } else {
                        *o
                    };
                    state.y2 = state.y1;
                    state.y1 = y;
                    *o = y;
                }
            }
            let wet = if channel == LFE_CHANNEL { 0.0 } else { params.wet };
            ramp_in_place(out, if reset { wet } else { state.wet }, wet);
            ramp_add(out, if reset { params.dry } else { state.dry }, params.dry, input);
            state.wet = wet;
            state.dry = params.dry;
            audible |= out.iter().any(|&v| v != 0.0);
        }
        self.scratch = s;
        self.reset = false;
        self.changed = false;
        Ok(audible)
    }
}

impl std::fmt::Debug for ReverbState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReverbState").field("params", &self.params).field("sample_rate", &self.sample_rate).finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameters_are_clamped_and_nan_rejected() {
        let mut r = ReverbState::new();
        assert_eq!(r.params, I3dl2Param::SETUP);
        let wild = I3dl2Param {
            wet: 40.0,
            dry: -3.0,
            reflection_pattern: 20,
            decay_time: 50.0,
            decay_hf_ratio: 5.0,
            reflections_delay: 0.9,
            reverb_delay: 0.9,
            diffusion: 300.0,
            density: -1.0,
            hf_reference: 50000.0,
            room: 5000,
            ..I3dl2Param::SETUP
        };
        r.set_params(&wild).unwrap();
        let p = r.params;
        assert_eq!((p.wet, p.dry, p.reflection_pattern, p.decay_time, p.decay_hf_ratio), (32.0, 0.0, 0, 20.0, 2.0));
        assert_eq!((p.reflections_delay, p.reverb_delay, p.diffusion, p.density, p.hf_reference), (0.3, 0.1, 100.0, 0.0, 20000.0));
        assert_eq!(p.room, 5000, "millibel levels are stored as given");
        let low = I3dl2Param { decay_time: 0.0, decay_hf_ratio: 0.0, hf_reference: 0.0, reflection_pattern: 15, ..I3dl2Param::SETUP };
        r.set_params(&low).unwrap();
        assert_eq!((r.params.decay_time, r.params.decay_hf_ratio, r.params.hf_reference, r.params.reflection_pattern), (0.1, 0.00999999978, 20.0, 15));
        for nan in [
            I3dl2Param { wet: f32::NAN, ..I3dl2Param::SETUP },
            I3dl2Param { density: f32::NAN, ..I3dl2Param::SETUP },
            I3dl2Param { hf_reference: f32::NAN, ..I3dl2Param::SETUP },
        ] {
            assert_eq!(r.set_params(&nan), Err(ReverbError::InvalidParameter));
        }
        assert_eq!(r.params.reflection_pattern, 15, "a rejected set changed the parameters");
    }

    #[test]
    fn parameter_image_round_trip() {
        let p = I3dl2Param { room: -1000, reflections: -602, reverb: -302, reflection_pattern: 5, ..I3dl2Param::SETUP };
        assert_eq!(I3dl2Param::parse(&p.to_bytes()), p);
    }

    /// Below about 7.5 kHz an allpass would have no length.
    #[test]
    fn low_sample_rates_are_rejected() {
        let mut r = ReverbState::new();
        let mut grain = vec![0.25f32; 2 * 64];
        assert_eq!(r.process(&mut grain, 2, 64, 7000), Err(ReverbError::UnsupportedSampleRate(7000)));
        assert_eq!(grain, vec![0.25f32; 2 * 64], "a rejected grain was changed");
        assert!(r.process(&mut grain, 2, 64, 48000).is_ok());
    }

    /// Wet 0 and dry 1 pass the input; the LFE channel always passes dry.
    #[test]
    fn dry_path_and_lfe() {
        let mut r = ReverbState::new();
        r.set_params(&I3dl2Param { wet: 0.0, dry: 1.0, ..I3dl2Param::SETUP }).unwrap();
        let input: Vec<f32> = (0..6 * 32).map(|i| (i % 7) as f32 * 0.1).collect();
        let mut grain = input.clone();
        r.process(&mut grain, 6, 32, 48000).unwrap();
        assert_eq!(grain, input);
        let mut r = ReverbState::new();
        r.set_params(&I3dl2Param { wet: 1.0, dry: 0.5, room: 0, reverb: 0, reflections: 0, ..I3dl2Param::SETUP }).unwrap();
        let mut grain = input.clone();
        r.process(&mut grain, 6, 32, 48000).unwrap();
        let lfe: Vec<f32> = input[3 * 32..4 * 32].iter().map(|x| x * 0.5).collect();
        assert_eq!(&grain[3 * 32..4 * 32], &lfe[..], "the LFE channel did not take only the dry level");
    }
}
