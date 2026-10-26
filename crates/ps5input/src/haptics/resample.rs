// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Stereo f32 rate converter (linear interpolation, exact rational
//! position) for the Bluetooth audio streams.

use std::collections::VecDeque;

pub struct Resampler {
    in_rate: u64,
    out_rate: u64,
    frames: VecDeque<[f32; 2]>,
    /// Read position in input frames times `out_rate`, relative to the
    /// first buffered frame.
    pos: u64,
}

impl Resampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        Self { in_rate: in_rate.max(1) as u64, out_rate: out_rate.max(1) as u64, frames: VecDeque::new(), pos: 0 }
    }

    /// Queue interleaved stereo input.
    pub fn put(&mut self, stereo: &[f32]) {
        self.frames.extend(stereo.chunks_exact(2).map(|f| [f[0], f[1]]));
    }

    /// Input bytes still queued.
    pub fn queued_bytes(&self) -> usize {
        self.frames.len() * 8
    }

    /// Output frames ready (each needs its neighbouring input frames).
    pub fn available_frames(&self) -> usize {
        let Some(last) = (self.frames.len() as u64).checked_sub(1) else { return 0 };
        let end = last * self.out_rate;
        if self.pos > end { 0 } else { ((end - self.pos) / self.in_rate + 1) as usize }
    }

    /// Fill `out` (interleaved stereo) with up to its length; returns the
    /// number of floats written.
    pub fn get(&mut self, out: &mut [f32]) -> usize {
        let n = (out.len() / 2).min(self.available_frames());
        for frame in out.chunks_exact_mut(2).take(n) {
            let i = (self.pos / self.out_rate) as usize;
            let frac = (self.pos % self.out_rate) as f32 / self.out_rate as f32;
            let a = self.frames[i];
            let v = if frac > 0.0 {
                let b = self.frames[i + 1];
                [a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac]
            } else {
                a
            };
            frame.copy_from_slice(&v);
            self.pos += self.in_rate;
        }
        let consumed = ((self.pos / self.out_rate) as usize).min(self.frames.len());
        self.frames.drain(..consumed);
        self.pos -= consumed as u64 * self.out_rate;
        n * 2
    }

    pub fn clear(&mut self) {
        self.frames.clear();
        self.pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downsample_constant_and_ramp() {
        let mut r = Resampler::new(48000, 3000);
        r.put(&[0.5; 2048]);
        assert_eq!(r.queued_bytes(), 1024 * 8);
        assert_eq!(r.available_frames(), 64);
        let mut out = [0.0; 128];
        assert_eq!(r.get(&mut out), 128);
        assert!(out.iter().all(|&v| v == 0.5));
        // 64 output frames consume exactly 1024 input frames.
        assert_eq!(r.queued_bytes(), 0);

        let mut r = Resampler::new(1000, 4000);
        r.put(&[0.0, 0.0, 1.0, -1.0]);
        let mut out = [9.0; 10];
        assert_eq!(r.get(&mut out), 10);
        assert_eq!(&out[..10], &[0.0, 0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0]);
    }

    #[test]
    fn speaker_rate_is_exact() {
        let mut r = Resampler::new(48000, 45000);
        r.put(&vec![0.25; 2 * 2048]);
        let mut out = vec![0.0; 1920];
        assert_eq!(r.get(&mut out), 1920);
        assert_eq!(r.queued_bytes(), 1024 * 8);
        r.clear();
        assert_eq!((r.queued_bytes(), r.available_frames()), (0, 0));
    }
}
