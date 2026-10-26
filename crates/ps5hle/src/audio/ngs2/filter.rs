// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Sampler voice filter (SceNgs2SamplerFilterParam): an RBJ low-pass biquad
//! applied to the planar voice grain after resampling, at the system rate.

/// SceNgs2SamplerFilterParam (48 bytes): index@0, location@4, type@8,
/// channelMask (u64)@16, frequency@24, q@28, level@32, reserved[3]@36.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct FilterParam {
    pub index: u32,
    pub location: u32,
    pub ty: u32,
    pub channel_mask: u64,
    pub frequency: f32,
    pub q: f32,
    pub level: f32,
}

pub(crate) const FILTER_PARAM_SIZE: usize = 48;

impl FilterParam {
    pub fn parse(b: &[u8]) -> Self {
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        Self {
            index: u32_at(0),
            location: u32_at(4),
            ty: u32_at(8),
            channel_mask: u64::from_le_bytes(b[16..24].try_into().unwrap()),
            frequency: f32::from_bits(u32_at(24)),
            q: f32::from_bits(u32_at(28)),
            level: f32::from_bits(u32_at(32)),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct History {
    pub x1: f64,
    pub x2: f64,
    pub y1: f64,
    pub y2: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct Filter {
    pub enabled: bool,
    pub warned: bool,
    pub channel_mask: u64,
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
    pub history: Vec<History>,
}

impl Default for Filter {
    fn default() -> Self {
        Self { enabled: false, warned: false, channel_mask: 0, b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0, history: Vec::new() }
    }
}

impl Filter {
    pub fn configure(&mut self, p: &FilterParam, rate: u32, channels: u32) {
        if p.ty == 0 {
            self.enabled = false;
            self.history.clear();
            return;
        }
        if p.location != 1
            || p.ty != 1
            || !p.frequency.is_finite()
            || p.frequency < 0.0
            || !p.q.is_finite()
            || p.q <= 0.0
            || !p.level.is_finite()
            || p.level < 0.0
            || rate == 0
        {
            self.enabled = false;
            self.history.clear();
            if !self.warned {
                log::warn!("[ngs2] unsupported sampler filter (location={}, type={})", p.location, p.ty);
                self.warned = true;
            }
            return;
        }
        self.channel_mask = p.channel_mask;
        self.history.resize(channels as usize, History::default());
        self.enabled = true;
        let frequency = p.frequency as f64;
        if p.frequency == 0.0 || frequency >= rate as f64 * 0.5 {
            self.b0 = if p.frequency == 0.0 { 0.0 } else { p.level as f64 };
            self.b1 = 0.0;
            self.b2 = 0.0;
            self.a1 = 0.0;
            self.a2 = 0.0;
            return;
        }
        // RBJ low-pass biquad normalised by a0 (Audio EQ Cookbook).
        let omega = (2.0 * std::f64::consts::PI) * frequency / rate as f64;
        let cosine = omega.cos();
        let alpha = omega.sin() / (2.0 * p.q as f64);
        let inverse_a0 = 1.0 / (1.0 + alpha);
        self.b0 = (1.0 - cosine) * 0.5 * inverse_a0 * p.level as f64;
        self.b1 = 2.0 * self.b0;
        self.b2 = self.b0;
        self.a1 = -2.0 * cosine * inverse_a0;
        self.a2 = (1.0 - alpha) * inverse_a0;
    }

    pub fn bypasses(&self, channel: usize) -> bool {
        channel < 64 && (self.channel_mask & (1u64 << channel)) != 0
    }

    pub fn has_history(&self) -> bool {
        self.enabled
            && self
                .history
                .iter()
                .enumerate()
                .any(|(c, h)| !self.bypasses(c) && (h.x1 != 0.0 || h.x2 != 0.0 || h.y1 != 0.0 || h.y2 != 0.0))
    }

    /// Filter planar `samples` (`channels` runs of `grain` floats) in place.
    pub fn process(&mut self, samples: &mut [f32], channels: u32, grain: u32) {
        if !self.enabled {
            return;
        }
        let grain = grain as usize;
        let (b0, b1, b2, a1, a2) = (self.b0, self.b1, self.b2, self.a1, self.a2);
        for c in 0..channels as usize {
            if self.bypasses(c) {
                continue;
            }
            let Some(h) = self.history.get_mut(c) else { continue };
            let Some(run) = samples.get_mut(c * grain..(c + 1) * grain) else { continue };
            let mut s = *h;
            for v in run.iter_mut() {
                let x = *v as f64;
                let mut y = b0 * x + b1 * s.x1 + b2 * s.x2 - a1 * s.y1 - a2 * s.y2;
                if y.abs() < 1e-20 {
                    y = 0.0;
                }
                s.x2 = s.x1;
                s.x1 = x;
                s.y2 = s.y1;
                s.y1 = y;
                *v = y as f32;
            }
            *h = s;
        }
    }
}
