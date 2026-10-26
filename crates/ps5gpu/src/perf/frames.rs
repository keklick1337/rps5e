// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Frame-time history of the presented frames: a fixed ring (no
//! allocation per frame), summaries (FPS, average, 1% low) and the
//! per-column maxima the frame-time graph draws.

use std::time::Instant;

/// Frames kept: covers the graph window at up to ~250 fps.
pub const CAPACITY: usize = 1024;

/// Summary of a window of frames.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameSummary {
    pub frames: usize,
    pub fps: f32,
    pub avg_ms: f32,
    /// FPS over the slowest 1% of the frames.
    pub low1_fps: f32,
    pub max_ms: f32,
}

pub struct FrameTimes {
    /// Frame time (ms) and its end time (s since `origin`), oldest first
    /// from `head`.
    ms: Box<[f32; CAPACITY]>,
    at: Box<[f64; CAPACITY]>,
    head: usize,
    len: usize,
    origin: Instant,
    last: Option<f64>,
    scratch: Box<[f32; CAPACITY]>,
}

impl Default for FrameTimes {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameTimes {
    pub fn new() -> Self {
        Self {
            ms: Box::new([0.0; CAPACITY]),
            at: Box::new([0.0; CAPACITY]),
            head: 0,
            len: 0,
            origin: Instant::now(),
            last: None,
            scratch: Box::new([0.0; CAPACITY]),
        }
    }

    fn secs(&self, t: Instant) -> f64 {
        t.saturating_duration_since(self.origin).as_secs_f64()
    }

    /// A frame was shown at `now`.
    pub fn record(&mut self, now: Instant) {
        let t = self.secs(now);
        if let Some(last) = self.last.replace(t) {
            self.push(((t - last) * 1e3) as f32, t);
        }
    }

    /// Add one frame time ending at `t` seconds (tests drive this directly).
    pub fn push(&mut self, ms: f32, t: f64) {
        let i = (self.head + self.len) % CAPACITY;
        self.ms[i] = ms;
        self.at[i] = t;
        if self.len == CAPACITY {
            self.head = (self.head + 1) % CAPACITY;
        } else {
            self.len += 1;
        }
    }

    /// Frames newer than `window` seconds before `now`, newest first.
    fn recent(&self, now: f64, window: f64) -> impl Iterator<Item = (f32, f64)> + '_ {
        (0..self.len)
            .rev()
            .map(move |k| {
                let i = (self.head + k) % CAPACITY;
                (self.ms[i], self.at[i])
            })
            .take_while(move |&(_, at)| now - at <= window)
    }

    /// Summary of the frames of the last `window` seconds before `now`.
    pub fn summary(&mut self, now: Instant, window: f64) -> FrameSummary {
        self.summary_at(self.secs(now), window)
    }

    pub fn summary_at(&mut self, now: f64, window: f64) -> FrameSummary {
        let mut n = 0;
        let (mut sum, mut max) = (0.0f64, 0.0f32);
        for k in (0..self.len).rev() {
            let i = (self.head + k) % CAPACITY;
            if now - self.at[i] > window {
                break;
            }
            let ms = self.ms[i];
            self.scratch[n] = ms;
            n += 1;
            sum += ms as f64;
            max = max.max(ms);
        }
        let mut s = FrameSummary { frames: n, max_ms: max, ..Default::default() };
        if n > 0 && sum > 0.0 {
            s.avg_ms = (sum / n as f64) as f32;
            s.fps = (n as f64 * 1e3 / sum) as f32;
            // The slowest 1% of the frames (at least one), averaged.
            let m = n.div_ceil(100);
            let slice = &mut self.scratch[..n];
            if m < n {
                slice.select_nth_unstable_by(n - m, f32::total_cmp);
            }
            let worst = slice[n - m..].iter().sum::<f32>() / m as f32;
            s.low1_fps = if worst > 0.0 { 1e3 / worst } else { 0.0 };
        }
        s
    }

    /// Fill `columns` with the largest frame time (ms) of each slice of the
    /// last `window` seconds before `now` (oldest column first; 0 where no
    /// frame ended).
    pub fn graph(&self, now: Instant, window: f64, columns: &mut [f32]) {
        self.graph_at(self.secs(now), window, columns)
    }

    pub fn graph_at(&self, now: f64, window: f64, columns: &mut [f32]) {
        columns.fill(0.0);
        let n = columns.len();
        if n == 0 || window <= 0.0 {
            return;
        }
        for (ms, at) in self.recent(now, window) {
            let age = (now - at).max(0.0);
            let col = n - 1 - (((age / window) * n as f64) as usize).min(n - 1);
            columns[col] = columns[col].max(ms);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steady_frames() {
        let mut f = FrameTimes::new();
        let mut t = 0.0;
        for _ in 0..240 {
            t += 1.0 / 60.0;
            f.push(1000.0 / 60.0, t);
        }
        let s = f.summary_at(t, 1.0);
        assert!((s.fps - 60.0).abs() < 0.01, "{s:?}");
        assert!((s.avg_ms - 16.667).abs() < 0.01);
        assert!((s.low1_fps - 60.0).abs() < 0.01);
        assert!(s.frames >= 59 && s.frames <= 61);
    }

    #[test]
    fn one_percent_low_tracks_the_slowest_frames() {
        let mut f = FrameTimes::new();
        let mut t = 0.0;
        // 198 frames of 10 ms, 2 hitches of 50 ms.
        for i in 0..200 {
            let ms = if i == 50 || i == 150 { 50.0 } else { 10.0 };
            t += ms as f64 / 1e3;
            f.push(ms, t);
        }
        let s = f.summary_at(t, 10.0);
        assert_eq!(s.frames, 200);
        assert_eq!(s.max_ms, 50.0);
        assert!((s.low1_fps - 20.0).abs() < 0.01, "{s:?}");
        assert!((s.avg_ms - 10.4).abs() < 0.01);
    }

    #[test]
    fn empty_and_window() {
        let mut f = FrameTimes::new();
        assert_eq!(f.summary_at(5.0, 1.0), FrameSummary::default());
        f.push(16.0, 1.0);
        // The only frame is older than the window.
        assert_eq!(f.summary_at(5.0, 1.0).frames, 0);
        assert_eq!(f.summary_at(1.5, 1.0).frames, 1);
    }

    #[test]
    fn ring_wraps() {
        let mut f = FrameTimes::new();
        for i in 0..CAPACITY * 3 {
            f.push(1.0, i as f64 * 1e-3);
        }
        let now = (CAPACITY * 3 - 1) as f64 * 1e-3;
        assert_eq!(f.summary_at(now, 100.0).frames, CAPACITY);
    }

    #[test]
    fn graph_columns() {
        let mut f = FrameTimes::new();
        f.push(10.0, 0.5);
        f.push(30.0, 3.9);
        f.push(20.0, 3.95);
        let mut cols = [0.0f32; 4];
        f.graph_at(4.0, 4.0, &mut cols);
        // Ages 3.5 s (first column), 0.1 and 0.05 s (last column, max).
        assert_eq!(cols, [10.0, 0.0, 0.0, 30.0]);
        f.graph_at(4.0, 1.0, &mut cols);
        assert_eq!(cols, [0.0, 0.0, 0.0, 30.0]);
    }

    #[test]
    fn record_measures_intervals() {
        let mut f = FrameTimes::new();
        let t0 = Instant::now();
        f.record(t0);
        f.record(t0 + std::time::Duration::from_millis(20));
        let s = f.summary(t0 + std::time::Duration::from_millis(20), 1.0);
        assert_eq!(s.frames, 1);
        assert!((s.avg_ms - 20.0).abs() < 0.01);
    }
}
