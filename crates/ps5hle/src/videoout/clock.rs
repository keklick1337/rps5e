// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Emulated display clock: vblank counting and flip pacing.
//!
//! Timestamps are nanoseconds on the video-out monotonic timeline
//! ([`super::now_ts`]). Callers hold the video-out state lock while they
//! advance or read a clock.

pub const FLIP_MODE_VSYNC: i32 = 1;
pub const FLIP_MODE_ASAP: i32 = 2;
pub const FLIP_MODE_WINDOW: i32 = 3;
pub const FLIP_MODE_VSYNC_MULTIPLE: i32 = 4;

/// Ticks per second of the timeline.
pub const TIMELINE_HZ: u64 = 1_000_000_000;

#[derive(Debug, Clone)]
pub struct DisplayClock {
    opened_at: i64,
    process_counter_at_open: u64,
    tsc_at_open: u64,
    tsc_hz: u64,
    pub count: u64,
    pub last_timestamp: i64,
    pub process_counter: u64,
    pub timestamp_counter: u64,
}

/// Length of one refresh at `refresh_rate` Hz. The NTSC-style rates 60 and
/// 120 run at 59.94 / 119.88 Hz like the real scan-out.
pub fn refresh_interval(refresh_rate: u32) -> i64 {
    let rate = refresh_rate.max(1) as i64;
    let ns = if rate == 60 || rate == 120 {
        TIMELINE_HZ as i64 * 1001 / (rate * 1000)
    } else {
        TIMELINE_HZ as i64 / rate
    };
    ns.max(1)
}

impl DisplayClock {
    pub fn new(opened_at: i64, process_counter: u64, tsc: u64, tsc_hz: u64) -> Self {
        Self {
            opened_at,
            process_counter_at_open: process_counter,
            tsc_at_open: tsc,
            tsc_hz,
            count: 0,
            last_timestamp: opened_at,
            process_counter: process_counter,
            timestamp_counter: tsc,
        }
    }

    /// Process time (µs) of the latest vblank.
    pub fn process_microseconds(&self) -> u64 {
        self.process_counter / 1000
    }

    /// Advance the vblank count to `timestamp`.
    pub fn advance(&mut self, timestamp: i64, refresh_rate: u32) {
        let interval = refresh_interval(refresh_rate);
        let count = ((timestamp - self.opened_at).max(0) / interval) as u64;
        if count <= self.count {
            return;
        }
        self.count = count;
        let elapsed = interval as u64 * count;
        self.last_timestamp = self.opened_at + elapsed as i64;
        self.process_counter = self.process_counter_at_open + elapsed;
        self.timestamp_counter =
            self.tsc_at_open + ((elapsed as u128 * self.tsc_hz as u128) / TIMELINE_HZ as u128) as u64;
    }

    /// Timestamp of the next vblank.
    pub fn next_timestamp(&self, refresh_rate: u32) -> i64 {
        self.opened_at + (self.count as i64 + 1) * refresh_interval(refresh_rate)
    }
}

/// Earliest time a flip may be shown. `last_presented_at` < 0 means no flip
/// has been shown yet; `ready_at` is when the flip became ready.
#[allow(clippy::too_many_arguments)]
pub fn next_flip_timestamp(
    opened_at: i64,
    last_presented_at: i64,
    timestamp: i64,
    refresh_rate: u32,
    flip_rate: i32,
    flip_mode: i32,
    height: u32,
    window_top: i32,
    window_bottom: i32,
    ready_at: i64,
) -> i64 {
    if flip_mode == FLIP_MODE_ASAP {
        return timestamp;
    }
    if flip_mode == FLIP_MODE_VSYNC_MULTIPLE {
        // All flips ready within one interval share its next boundary.
        let interval = refresh_interval(refresh_rate);
        let ready_interval = (ready_at - opened_at).max(0) / interval;
        return opened_at + (ready_interval + 1) * interval;
    }
    if last_presented_at < 0 && flip_mode != FLIP_MODE_WINDOW {
        return timestamp;
    }
    let interval = refresh_interval(refresh_rate);
    let flip_interval = interval * (flip_rate as i64 + 1).max(1);
    if flip_mode != FLIP_MODE_WINDOW || height == 0 {
        // A flip waits `flip_rate + 1` vblanks counted from the vblank of
        // the previous flip (a swap interval), not a fixed grid: at 30 Hz
        // two flips are never one refresh apart.
        let previous_vblank = (last_presented_at - opened_at).max(0) / interval;
        let next = opened_at + (previous_vblank + (flip_rate as i64).max(0) + 1) * interval;
        return timestamp.max(next);
    }
    let previous_interval = (last_presented_at - opened_at).max(0) / flip_interval;
    let next = opened_at + (previous_interval + 1) * flip_interval;

    // Window margins are scanline positions at the top and bottom of the image.
    let h = height as i64;
    let top_end = interval * (window_top as i64).clamp(0, h) / h;
    let bottom_start = interval * (window_bottom as i64).clamp(0, h) / h;
    if top_end == 0 && bottom_start == interval {
        return if last_presented_at < 0 { timestamp } else { timestamp.max(next) };
    }
    let window_lead = interval - bottom_start;
    let previous_window = if last_presented_at < 0 {
        -1
    } else {
        ((last_presented_at - opened_at).max(0) + window_lead) / flip_interval
    };
    let window_start = opened_at + (previous_window + 1) * flip_interval - window_lead;
    if timestamp < window_start {
        return window_start;
    }
    let phase = (timestamp - opened_at).max(0) % interval;
    if phase <= top_end || phase >= bottom_start {
        timestamp
    } else {
        timestamp - phase + bottom_start
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_is_59_94() {
        assert_eq!(refresh_interval(60), 16_683_333);
        assert_eq!(refresh_interval(30), 33_333_333);
    }

    #[test]
    fn clock_advances_in_whole_refreshes() {
        let mut c = DisplayClock::new(1000, 0, 0, TIMELINE_HZ);
        let i = refresh_interval(60);
        c.advance(1000 + i - 1, 60);
        assert_eq!(c.count, 0);
        c.advance(1000 + 3 * i + 5, 60);
        assert_eq!(c.count, 3);
        assert_eq!(c.last_timestamp, 1000 + 3 * i);
        assert_eq!(c.next_timestamp(60), 1000 + 4 * i);
        assert_eq!(c.timestamp_counter, 3 * i as u64);
        // Never goes backwards.
        c.advance(0, 60);
        assert_eq!(c.count, 3);
    }

    #[test]
    fn flip_pacing() {
        let i = refresh_interval(60);
        // First vsync flip is immediate, the next waits for the boundary.
        assert_eq!(next_flip_timestamp(0, -1, 5, 60, 0, FLIP_MODE_VSYNC, 1080, 0, 1080, 5), 5);
        assert_eq!(next_flip_timestamp(0, 5, 6, 60, 0, FLIP_MODE_VSYNC, 1080, 0, 1080, 6), i);
        // Flip rate 1 (30 Hz) doubles the interval.
        assert_eq!(next_flip_timestamp(0, 5, 6, 60, 1, FLIP_MODE_VSYNC, 1080, 0, 1080, 6), 2 * i);
        assert_eq!(next_flip_timestamp(0, 5, 7, 60, 0, FLIP_MODE_ASAP, 1080, 0, 1080, 7), 7);
        // Vsync-multiple: boundary after readiness.
        assert_eq!(next_flip_timestamp(0, -1, 9, 60, 0, FLIP_MODE_VSYNC_MULTIPLE, 1080, 0, 1080, i + 3), 2 * i);
    }

    #[test]
    fn flip_rate_counts_from_the_previous_flip() {
        let i = refresh_interval(60);
        // Previous flip shown at vblank 5: at 30 Hz the next waits for
        // vblank 7, at 20 Hz for vblank 8, at 60 Hz for vblank 6.
        assert_eq!(next_flip_timestamp(0, 5 * i, 5 * i + 1, 60, 1, FLIP_MODE_VSYNC, 1080, 0, 1080, 5 * i + 1), 7 * i);
        assert_eq!(next_flip_timestamp(0, 5 * i, 5 * i + 1, 60, 2, FLIP_MODE_VSYNC, 1080, 0, 1080, 5 * i + 1), 8 * i);
        assert_eq!(next_flip_timestamp(0, 5 * i, 5 * i + 1, 60, 0, FLIP_MODE_VSYNC, 1080, 0, 1080, 5 * i + 1), 6 * i);
        // A late flip is shown at once.
        assert_eq!(next_flip_timestamp(0, 5 * i, 9 * i + 3, 60, 1, FLIP_MODE_VSYNC, 1080, 0, 1080, 9 * i + 3), 9 * i + 3);
        // The first flip was shown between vblanks (at once): counted from
        // the vblank before it.
        assert_eq!(next_flip_timestamp(0, 4 * i + 7, 4 * i + 9, 60, 1, FLIP_MODE_VSYNC, 1080, 0, 1080, 4 * i + 9), 6 * i);
    }
}
