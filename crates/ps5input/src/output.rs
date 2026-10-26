// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Force feedback / light bar requests from the guest, collected per pad
//! slot for the input thread (all device calls happen there).

use crate::mapping::MAX_PLAYERS;
use crate::sdl::OFF_TRIGGER;
use ps5hle::pad::{DualSenseEffects, HostPadOutput};
use ps5hle::pad_controller::{DS5_ENABLE_LEFT_TRIGGER, DS5_ENABLE_RIGHT_TRIGGER};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Requested output state of one pad; `seq` changes on every request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputState {
    pub seq: u64,
    /// Motor strengths (already scaled by the vibration intensity).
    pub large: u8,
    pub small: u8,
    /// End of the vibration request (`None`: stopped).
    pub rumble_until: Option<Instant>,
    /// `None`: no title colour (reset).
    pub lightbar: Option<[u8; 3]>,
    /// DualSense trigger effects of L2, R2.
    pub triggers: [[u8; 11]; 2],
    /// A trigger effect was ever requested.
    pub triggers_touched: bool,
}

impl Default for OutputState {
    fn default() -> Self {
        Self {
            seq: 0,
            large: 0,
            small: 0,
            rumble_until: None,
            lightbar: None,
            triggers: [OFF_TRIGGER; 2],
            triggers_touched: false,
        }
    }
}

pub type SharedOutput = Arc<Mutex<[OutputState; MAX_PLAYERS]>>;

/// The `HostPadOutput` installed into libScePad.
pub struct OutputSink(pub SharedOutput);

impl OutputSink {
    fn update(&self, slot: usize, f: impl FnOnce(&mut OutputState)) {
        let mut all = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let Some(s) = all.get_mut(slot) else { return };
        let before = *s;
        f(s);
        if *s != before {
            s.seq = s.seq.wrapping_add(1);
        }
    }
}

impl HostPadOutput for OutputSink {
    fn set_vibration(&mut self, slot: usize, large: u8, small: u8, duration_ms: u32) {
        self.update(slot, |s| {
            s.large = large;
            s.small = small;
            s.rumble_until = (duration_ms != 0).then(|| Instant::now() + Duration::from_millis(duration_ms as u64));
        });
    }

    fn set_lightbar(&mut self, slot: usize, r: u8, g: u8, b: u8) {
        self.update(slot, |s| s.lightbar = Some([r, g, b]));
    }

    fn reset_lightbar(&mut self, slot: usize) {
        self.update(slot, |s| s.lightbar = None);
    }

    fn set_trigger_effect(&mut self, slot: usize, effect: &DualSenseEffects) {
        self.update(slot, |s| {
            if effect.enable_bits() & DS5_ENABLE_LEFT_TRIGGER != 0 {
                s.triggers[0].copy_from_slice(effect.left_trigger());
            }
            if effect.enable_bits() & DS5_ENABLE_RIGHT_TRIGGER != 0 {
                s.triggers[1].copy_from_slice(effect.right_trigger());
            }
            s.triggers_touched = true;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdl::ds5_trigger_report;

    #[test]
    fn requests_bump_sequence_only_on_change() {
        let shared = SharedOutput::default();
        let mut sink = OutputSink(shared.clone());
        sink.set_vibration(0, 10, 20, 0);
        sink.set_vibration(0, 10, 20, 0);
        assert_eq!(shared.lock().unwrap()[0].seq, 1);
        sink.set_lightbar(0, 1, 2, 3);
        sink.reset_lightbar(0);
        let s = shared.lock().unwrap()[0];
        assert_eq!((s.seq, s.large, s.small, s.lightbar), (3, 10, 20, None));
        // Other slots are independent; out-of-range slots are ignored.
        sink.set_vibration(2, 99, 0, 1000);
        sink.set_vibration(9, 1, 1, 1000);
        let all = *shared.lock().unwrap();
        assert_eq!((all[1].seq, all[2].large), (0, 99));
        assert!(all[2].rumble_until.is_some());
    }

    #[test]
    fn trigger_effects_reach_the_dualsense_report() {
        let shared = SharedOutput::default();
        let mut sink = OutputSink(shared.clone());
        let mut e = DualSenseEffects::default();
        e.0[0] = DS5_ENABLE_RIGHT_TRIGGER;
        e.0[10..21].copy_from_slice(&[0x21, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        e.0[21] = 0x26; // Left trigger not enabled: ignored.
        sink.set_trigger_effect(0, &e);
        let s = shared.lock().unwrap()[0];
        assert!(s.triggers_touched);
        let r = ds5_trigger_report(&s.triggers);
        assert_eq!(r.len(), 47);
        assert_eq!(r[0], 0x0C);
        assert_eq!(&r[10..21], &[0x21, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(r[21], 0x05); // left untouched -> off
    }
}
