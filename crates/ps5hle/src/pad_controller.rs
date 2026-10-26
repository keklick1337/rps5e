// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Controller output behind libScePad: the user's controller settings (pad
//! speaker volume, vibration and trigger effect intensity), each pad's
//! cached vibration and adaptive-trigger requests (re-sent when a setting
//! changes), the pad API trigger command -> DualSense effect encoding and
//! the trigger effect state machine reported by
//! `scePadGetTriggerEffectState`.
//!
//! Everything here is pure: callers pass the time and forward the returned
//! requests to the host output.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

/// User controller settings, cycled by hotkeys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Setting {
    SpeakerVolume,
    VibrationIntensity,
    TriggerEffectIntensity,
}

// In cycle order, starting at the console default.
const SPEAKER_VOLUME: [f32; 5] = [1.0, 0.0, 0.02, 0.12, 0.42];
const INTENSITY: [f32; 4] = [1.0, 0.0, 0.33, 0.66];

/// Global controller levels (0..=100) and the cycled steps.
pub struct Settings {
    steps: [AtomicU32; 3],
    speaker_volume: AtomicU32,
    vibration_intensity: AtomicU32,
    color: Mutex<Option<[u8; 3]>>,
}

/// Default global pad speaker level: the midpoint, unity gain.
pub const DEFAULT_SPEAKER_VOLUME: u32 = 50;
/// Default global vibration level: full strength.
pub const DEFAULT_VIBRATION_INTENSITY: u32 = 100;

impl Settings {
    pub const fn new() -> Self {
        Self {
            steps: [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)],
            speaker_volume: AtomicU32::new(DEFAULT_SPEAKER_VOLUME),
            vibration_intensity: AtomicU32::new(DEFAULT_VIBRATION_INTENSITY),
            color: Mutex::new(None),
        }
    }

    fn index(setting: Setting) -> usize {
        match setting {
            Setting::SpeakerVolume => 0,
            Setting::VibrationIntensity => 1,
            Setting::TriggerEffectIntensity => 2,
        }
    }

    /// Advance a setting to its next level (wrapping).
    pub fn cycle(&self, setting: Setting) {
        let count = if setting == Setting::SpeakerVolume { SPEAKER_VOLUME.len() } else { INTENSITY.len() } as u32;
        let step = &self.steps[Self::index(setting)];
        step.store((step.load(Ordering::Relaxed) + 1) % count, Ordering::Relaxed);
    }

    /// Output scale of a setting, including the global level.
    pub fn scale(&self, setting: Setting) -> f32 {
        let step = self.steps[Self::index(setting)].load(Ordering::Relaxed) as usize;
        match setting {
            // The midpoint of the global level is unity; its upper half adds
            // up to 6 dB of software gain.
            Setting::SpeakerVolume => {
                SPEAKER_VOLUME[step] * (self.speaker_volume.load(Ordering::Relaxed) as f32 / 50.0)
            }
            Setting::VibrationIntensity => {
                INTENSITY[step] * (self.vibration_intensity.load(Ordering::Relaxed) as f32 / 100.0)
            }
            Setting::TriggerEffectIntensity => INTENSITY[step],
        }
    }

    /// Global levels (0..=100); the cycled steps return to their defaults.
    pub fn configure(&self, speaker_volume: u32, vibration_intensity: u32) {
        self.speaker_volume.store(speaker_volume.min(100), Ordering::Relaxed);
        self.vibration_intensity.store(vibration_intensity.min(100), Ordering::Relaxed);
        for s in &self.steps {
            s.store(0, Ordering::Relaxed);
        }
    }

    /// Fixed light bar colour overriding the title's requests.
    pub fn set_color(&self, color: Option<[u8; 3]>) {
        *self.color.lock().unwrap_or_else(|e| e.into_inner()) = color;
    }

    pub fn color(&self) -> Option<[u8; 3]> {
        *self.color.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self::new()
    }
}

/// The emulator's controller settings.
pub static SETTINGS: Settings = Settings::new();

/// Scale an output strength; a non-zero value stays at least 1.
pub fn scale(value: u8, scale: f32) -> u8 {
    if value == 0 || scale == 0.0 {
        return 0;
    }
    (value as f32 * scale).round().clamp(1.0, 255.0) as u8
}

// --- trigger effects ---------------------------------------------------------

/// ScePadTriggerEffectParam size: u8 triggerMask, reserved[7], two commands.
pub const TRIGGER_EFFECT_PARAM_SIZE: usize = 120;
/// ScePadTriggerEffectCommand size: u32 mode, reserved[4], u8 data[48].
const TRIGGER_EFFECT_COMMAND_SIZE: usize = 56;

/// One ScePadTriggerEffectCommand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TriggerCommand {
    pub mode: u32,
    pub data: [u8; 48],
}

impl Default for TriggerCommand {
    fn default() -> Self {
        Self { mode: 0, data: [0; 48] }
    }
}

impl TriggerCommand {
    fn parse(b: &[u8]) -> Self {
        let mut data = [0u8; 48];
        data.copy_from_slice(&b[8..56]);
        Self { mode: u32::from_le_bytes([b[0], b[1], b[2], b[3]]), data }
    }
}

/// ScePadTriggerEffectParam: bit 0 of the mask selects L2 (command 0),
/// bit 1 R2 (command 1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TriggerEffectParam {
    pub trigger_mask: u8,
    pub command: [TriggerCommand; 2],
}

impl TriggerEffectParam {
    /// Decode the guest structure (at least 120 bytes).
    pub fn parse(b: &[u8]) -> Self {
        let c = TRIGGER_EFFECT_COMMAND_SIZE;
        Self { trigger_mask: b[0], command: [TriggerCommand::parse(&b[8..8 + c]), TriggerCommand::parse(&b[8 + c..8 + 2 * c])] }
    }
}

/// DualSense effect report as taken by the controller driver: enable bits,
/// reserved, right trigger (11 bytes at 10), left trigger (11 bytes at 21).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DualSenseEffects(pub [u8; 32]);

pub const DS5_ENABLE_RIGHT_TRIGGER: u8 = 0x04;
pub const DS5_ENABLE_LEFT_TRIGGER: u8 = 0x08;
const RIGHT_TRIGGER: usize = 10;
const LEFT_TRIGGER: usize = 21;
/// DualSense trigger mode "off".
pub const TRIGGER_OFF: u8 = 0x05;
const TRIGGER_FEEDBACK: u8 = 0x21;
const TRIGGER_WEAPON: u8 = 0x25;
const TRIGGER_VIBRATION: u8 = 0x26;

impl Default for DualSenseEffects {
    fn default() -> Self {
        Self([0; 32])
    }
}

impl DualSenseEffects {
    pub fn enable_bits(&self) -> u8 {
        self.0[0]
    }

    pub fn right_trigger(&self) -> &[u8] {
        &self.0[RIGHT_TRIGGER..RIGHT_TRIGGER + 11]
    }

    pub fn left_trigger(&self) -> &[u8] {
        &self.0[LEFT_TRIGGER..LEFT_TRIGGER + 11]
    }

    /// Both triggers off (sent when the controller is released).
    pub fn off() -> Self {
        let mut e = Self::default();
        e.0[0] = DS5_ENABLE_RIGHT_TRIGGER | DS5_ENABLE_LEFT_TRIGGER;
        e.0[RIGHT_TRIGGER] = TRIGGER_OFF;
        e.0[LEFT_TRIGGER] = TRIGGER_OFF;
        e
    }
}

/// Rumble strength (0..255) approximating a native trigger effect on
/// controllers with plain trigger motors.
pub fn trigger_rumble_strength(native: &[u8]) -> u8 {
    let level = match native[0] {
        TRIGGER_FEEDBACK | TRIGGER_VIBRATION => {
            let active = u16::from_le_bytes([native[1], native[2]]);
            let packed = u32::from_le_bytes([native[3], native[4], native[5], native[6]]);
            (0..10).filter(|i| active & (1 << i) != 0).map(|i| 1 + ((packed >> (3 * i)) & 7)).max().unwrap_or(0)
        }
        TRIGGER_WEAPON => native[3] as u32 + 1,
        _ => 0,
    };
    (level.min(8) * 255 / 8) as u8
}

/// Encode per-zone strengths (0..=8, ten zones) into a zoned DualSense mode.
fn trigger_effect_zones(effect: &mut [u8], strengths: &[u8], scale_by: f32, ty: u8, frequency: u8) -> bool {
    let mut active: u16 = 0;
    let mut packed: u32 = 0;
    for (i, &s) in strengths.iter().take(10).enumerate() {
        if s > 8 {
            return false;
        }
        let strength = scale(s, scale_by);
        if strength != 0 {
            active |= 1 << i;
            packed |= ((strength - 1) as u32) << (3 * i);
        }
    }
    effect[0] = if active != 0 && (ty != TRIGGER_VIBRATION || frequency != 0) { ty } else { TRIGGER_OFF };
    effect[1..3].copy_from_slice(&active.to_le_bytes());
    effect[3..7].copy_from_slice(&packed.to_le_bytes());
    effect[9] = frequency;
    true
}

/// Encode one pad API trigger command as an 11-byte DualSense trigger
/// effect; false for an invalid command.
pub fn trigger_effect_to_dualsense(command: &TriggerCommand, effect: &mut [u8], scale_by: f32) -> bool {
    effect[..11].fill(0);
    effect[0] = TRIGGER_OFF;
    let d = &command.data;
    let mut strengths = [0u8; 10];
    match command.mode {
        0 => true,
        // Feedback: resistance from position d[0] with strength d[1].
        1 => {
            if d[0] > 9 || d[1] > 8 {
                return false;
            }
            strengths[d[0] as usize..].fill(d[1]);
            trigger_effect_zones(effect, &strengths, scale_by, TRIGGER_FEEDBACK, 0)
        }
        // Weapon: resistance from d[0], released at d[1], strength d[2].
        2 => {
            let (start, end, strength) = (d[0], d[1], d[2]);
            if !(2..=7).contains(&start) || end <= start || end > 8 || strength > 8 {
                return false;
            }
            let scaled = scale(strength, scale_by);
            if scaled == 0 {
                return true;
            }
            let zones: u16 = (1 << start) | (1 << end);
            effect[0] = TRIGGER_WEAPON;
            effect[1..3].copy_from_slice(&zones.to_le_bytes());
            effect[3] = scaled - 1;
            true
        }
        // Vibration from position d[0], amplitude d[1], frequency d[2].
        3 => {
            if d[0] > 9 || d[1] > 8 {
                return false;
            }
            strengths[d[0] as usize..].fill(d[1]);
            trigger_effect_zones(effect, &strengths, scale_by, TRIGGER_VIBRATION, d[2])
        }
        // Multiple-position feedback: strength per position.
        4 => trigger_effect_zones(effect, &d[..10], scale_by, TRIGGER_FEEDBACK, 0),
        // Slope feedback: d[2] at position d[0] rising/falling to d[3] at d[1].
        5 => {
            let (start, end, start_strength, end_strength) = (d[0] as i32, d[1] as i32, d[2] as i32, d[3] as i32);
            if start > 8
                || end <= start
                || end > 9
                || !(1..=8).contains(&start_strength)
                || !(1..=8).contains(&end_strength)
            {
                return false;
            }
            let length = end - start;
            for i in start..10 {
                let delta = (end_strength - start_strength) * (i - start);
                // Integer division rounding half away from zero.
                let half = if delta < 0 { -length / 2 } else { length / 2 };
                strengths[i as usize] =
                    if i >= end { end_strength } else { start_strength + (delta + half) / length } as u8;
            }
            trigger_effect_zones(effect, &strengths, scale_by, TRIGGER_FEEDBACK, 0)
        }
        // Multiple-position vibration: frequency d[0], amplitude per position.
        6 => trigger_effect_zones(effect, &d[1..11], scale_by, TRIGGER_VIBRATION, d[0]),
        _ => false,
    }
}

/// SCE_PAD_TRIGGER_EFFECT_TRIGGER_STATE_* values.
pub mod trigger_state {
    pub const OFF: i32 = 0;
    pub const FEEDBACK_STANDBY: i32 = 1;
    pub const FEEDBACK_ACTIVE: i32 = 2;
    pub const WEAPON_STANDBY: i32 = 3;
    pub const WEAPON_PULLING: i32 = 4;
    pub const WEAPON_FIRED: i32 = 5;
    pub const VIBRATION_STANDBY: i32 = 6;
    pub const VIBRATION_ACTIVE: i32 = 7;
}

/// State of one trigger under `command` at analog `value` (0..255).
pub fn trigger_effect_state(command: &TriggerCommand, value: u8, previous: i32) -> i32 {
    use trigger_state::*;
    let value = value as i32;
    let d = &command.data;
    // Trigger positions use nonuniform travel ranges, with endpoints at 0 and 255.
    let pos = if value == 0 {
        0
    } else if value == 255 {
        9
    } else {
        1 + value / 32
    };
    match command.mode {
        1 => {
            if d[1] != 0 && pos >= d[0] as i32 {
                FEEDBACK_ACTIVE
            } else {
                FEEDBACK_STANDBY
            }
        }
        2 => {
            if value < d[0] as i32 * 32 {
                WEAPON_STANDBY
            } else if previous == WEAPON_FIRED || value >= (d[1] as i32 * 32).min(254) {
                WEAPON_FIRED
            } else {
                WEAPON_PULLING
            }
        }
        3 => {
            if d[1] != 0 && d[2] != 0 && pos >= d[0] as i32 {
                VIBRATION_ACTIVE
            } else {
                VIBRATION_STANDBY
            }
        }
        4 => {
            if d[pos as usize] != 0 {
                FEEDBACK_ACTIVE
            } else {
                FEEDBACK_STANDBY
            }
        }
        5 => {
            if pos >= d[0] as i32 && pos < d[1] as i32 {
                FEEDBACK_ACTIVE
            } else {
                FEEDBACK_STANDBY
            }
        }
        6 => {
            if d[0] != 0 && d[1 + pos as usize] != 0 {
                VIBRATION_ACTIVE
            } else {
                VIBRATION_STANDBY
            }
        }
        _ => OFF,
    }
}

// --- per-pad output cache --------------------------------------------------------

/// A vibration request lasts this long unless replaced.
pub const RUMBLE_DURATION_MS: u64 = 0xffff;
/// How long vibration port audio keeps the motors at its level.
pub const AUDIO_HAPTICS_MS: u64 = 80;

/// Motor levels (large = left, small = right) of a vibration block: the
/// RMS of each of the two channels, from 1% up, at twice its level.
pub fn audio_haptics_levels(left: &[f32], right: &[f32]) -> (u8, u8) {
    let level = |samples: &[f32]| -> u8 {
        if samples.is_empty() {
            return 0;
        }
        let energy: f64 = samples.iter().filter(|s| s.is_finite()).map(|&s| (s.clamp(-1.0, 1.0) as f64).powi(2)).sum();
        let rms = (energy / samples.len() as f64).sqrt();
        if rms < 0.01 {
            0
        } else {
            ((rms * 2.0).min(1.0) * 255.0).round() as u8
        }
    };
    (level(left), level(right))
}

/// Scaled motor strengths and the remaining lifetime of a vibration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vibration {
    pub large: u8,
    pub small: u8,
    pub duration_ms: u32,
}

/// The title's last vibration and trigger requests of one pad, re-sent when
/// their intensity setting changes, plus the trigger effect states.
#[derive(Clone, Copy, Debug, Default)]
pub struct ControllerOutput {
    vibration: [u8; 2],
    vibration_until: u64,
    /// Motor levels from vibration port audio (no haptic actuator took it),
    /// held for [`AUDIO_HAPTICS_MS`] after the last audible block.
    audio: [u8; 2],
    audio_until: u64,
    /// Until when the motor levels last sent for audio haptics run.
    audio_sent_until: u64,
    trigger_effect: TriggerEffectParam,
    trigger_state: [i32; 2],
}

impl ControllerOutput {
    pub const fn new() -> Self {
        const OFF: TriggerCommand = TriggerCommand { mode: 0, data: [0; 48] };
        Self {
            vibration: [0; 2],
            vibration_until: 0,
            audio: [0; 2],
            audio_until: 0,
            audio_sent_until: 0,
            trigger_effect: TriggerEffectParam { trigger_mask: 0, command: [OFF; 2] },
            trigger_state: [0; 2],
        }
    }

    /// Forget every request (another controller became active).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// `scePadSetVibration` at `now_ms`; returns what to send.
    pub fn set_vibration(&mut self, large: u8, small: u8, now_ms: u64, settings: &Settings) -> Vibration {
        self.vibration = [large, small];
        self.vibration_until = now_ms + RUMBLE_DURATION_MS;
        self.vibration(now_ms, settings)
    }

    /// The cached vibration scaled by the current intensity, without
    /// changing its requested lifetime. Audio haptics raise the motors
    /// while they last; a level they raise lasts only as long as they do.
    pub fn vibration(&self, now_ms: u64, settings: &Settings) -> Vibration {
        let duration = self.vibration_until.saturating_sub(now_ms) as u32;
        let s = if duration != 0 { settings.scale(Setting::VibrationIntensity) } else { 0.0 };
        let title = Vibration { large: scale(self.vibration[0], s), small: scale(self.vibration[1], s), duration_ms: duration };
        let audio_ms = self.audio_until.saturating_sub(now_ms) as u32;
        if audio_ms == 0 || (self.audio[0] <= title.large && self.audio[1] <= title.small) {
            return title;
        }
        Vibration { large: title.large.max(self.audio[0]), small: title.small.max(self.audio[1]), duration_ms: audio_ms }
    }

    /// Motor levels of a vibration port block that no haptic actuator
    /// played (already scaled by the vibration intensity). Audible levels
    /// merge with those of the last [`AUDIO_HAPTICS_MS`]; silence lets
    /// them expire. Returns what to send when the motors change.
    pub fn set_audio_haptics(&mut self, large: u8, small: u8, now_ms: u64, settings: &Settings) -> Option<Vibration> {
        let expired = now_ms >= self.audio_until;
        if large == 0 && small == 0 {
            return self.expire_audio_haptics(now_ms, settings);
        }
        if expired {
            self.audio = [0; 2];
        }
        let before = self.audio;
        self.audio = [self.audio[0].max(large), self.audio[1].max(small)];
        self.audio_until = now_ms + AUDIO_HAPTICS_MS;
        // Re-send when the levels change, the window had run out or what
        // was sent is about to stop.
        if before == self.audio && !expired && now_ms + AUDIO_HAPTICS_MS / 2 < self.audio_sent_until {
            return None;
        }
        let v = self.vibration(now_ms, settings);
        self.audio_sent_until = now_ms + v.duration_ms as u64;
        Some(v)
    }

    /// Drop audio haptics whose window has passed; returns the title's
    /// vibration to restore.
    pub fn expire_audio_haptics(&mut self, now_ms: u64, settings: &Settings) -> Option<Vibration> {
        if self.audio == [0; 2] || now_ms < self.audio_until {
            return None;
        }
        self.audio = [0; 2];
        Some(self.vibration(now_ms, settings))
    }

    /// Encode `param` with the current trigger intensity: `Err` for an
    /// invalid request, `Ok(None)` when no trigger is selected.
    pub fn encode_trigger_effect(param: &TriggerEffectParam, settings: &Settings) -> Result<Option<DualSenseEffects>, ()> {
        if param.trigger_mask & !0x03 != 0 {
            return Err(());
        }
        let mut effect = DualSenseEffects::default();
        let s = settings.scale(Setting::TriggerEffectIntensity);
        if param.trigger_mask & 0x01 != 0 {
            effect.0[0] |= DS5_ENABLE_LEFT_TRIGGER;
            if !trigger_effect_to_dualsense(&param.command[0], &mut effect.0[LEFT_TRIGGER..], s) {
                return Err(());
            }
        }
        if param.trigger_mask & 0x02 != 0 {
            effect.0[0] |= DS5_ENABLE_RIGHT_TRIGGER;
            if !trigger_effect_to_dualsense(&param.command[1], &mut effect.0[RIGHT_TRIGGER..], s) {
                return Err(());
            }
        }
        Ok((effect.0[0] != 0).then_some(effect))
    }

    /// `scePadSetTriggerEffect` with the current trigger values (L2, R2).
    /// A rejected request leaves the cache and states unchanged.
    pub fn set_trigger_effect(
        &mut self,
        param: &TriggerEffectParam,
        triggers: [u8; 2],
        settings: &Settings,
    ) -> Result<Option<DualSenseEffects>, ()> {
        let effect = Self::encode_trigger_effect(param, settings)?;
        for i in 0..2 {
            if param.trigger_mask & (1 << i) != 0 {
                self.trigger_effect.trigger_mask |= 1 << i;
                self.trigger_effect.command[i] = param.command[i];
            }
        }
        self.update_trigger_state(triggers);
        Ok(effect)
    }

    /// The cached trigger effects re-encoded (after an intensity change).
    pub fn trigger_effect(&self, settings: &Settings) -> Option<DualSenseEffects> {
        Self::encode_trigger_effect(&self.trigger_effect, settings).ok().flatten()
    }

    /// Track every trigger value so a weapon stays fired even when the
    /// title polls after the release started.
    pub fn update_trigger_state(&mut self, triggers: [u8; 2]) {
        for (i, &value) in triggers.iter().enumerate() {
            self.trigger_state[i] = trigger_effect_state(&self.trigger_effect.command[i], value, self.trigger_state[i]);
        }
    }

    pub fn trigger_state(&self) -> [i32; 2] {
        self.trigger_state
    }
}

#[cfg(test)]
mod tests {
    use super::trigger_state::*;
    use super::*;

    fn cmd(mode: u32, params: &[u8]) -> TriggerCommand {
        let mut c = TriggerCommand { mode, ..TriggerCommand::default() };
        c.data[..params.len()].copy_from_slice(params);
        c
    }

    fn zone_strength(effect: &[u8], zone: u32) -> u32 {
        let active = effect[1] as u32 | (effect[2] as u32) << 8;
        let packed = u32::from_le_bytes([effect[3], effect[4], effect[5], effect[6]]);
        if active & (1 << zone) != 0 { 1 + ((packed >> (3 * zone)) & 7) } else { 0 }
    }

    fn encode(c: &TriggerCommand, s: f32) -> Option<[u8; 11]> {
        let mut e = [0u8; 11];
        trigger_effect_to_dualsense(c, &mut e, s).then_some(e)
    }

    #[test]
    fn trigger_command_encoding() {
        assert_eq!(encode(&cmd(0, &[]), 1.0).unwrap()[0], TRIGGER_OFF);
        // Feedback: position 5, strength 8 -> zones 5..9 at level 8.
        let e = encode(&cmd(1, &[5, 8]), 1.0).unwrap();
        assert_eq!(e[0], TRIGGER_FEEDBACK);
        assert_eq!(u16::from_le_bytes([e[1], e[2]]), 0x3E0);
        assert!((5..10).all(|z| zone_strength(&e, z) == 8));
        assert_eq!(trigger_rumble_strength(&e), 255);
        // Zero strength is a valid "off".
        assert_eq!(encode(&cmd(1, &[5, 0]), 1.0).unwrap()[0], TRIGGER_OFF);
        assert!(encode(&cmd(1, &[5, 9]), 1.0).is_none());
        assert!(encode(&cmd(1, &[10, 1]), 1.0).is_none());
        // Weapon.
        let e = encode(&cmd(2, &[2, 5, 4]), 1.0).unwrap();
        assert_eq!(&e[..4], &[TRIGGER_WEAPON, 0x24, 0x00, 3]);
        assert_eq!(trigger_rumble_strength(&e), 127);
        assert!(encode(&cmd(2, &[1, 5, 4]), 1.0).is_none());
        assert!(encode(&cmd(2, &[2, 9, 4]), 1.0).is_none());
        assert_eq!(encode(&cmd(2, &[2, 5, 0]), 1.0).unwrap()[0], TRIGGER_OFF);
        // Vibration needs a frequency.
        assert_eq!(encode(&cmd(3, &[0, 4, 0]), 1.0).unwrap()[0], TRIGGER_OFF);
        let e = encode(&cmd(3, &[8, 4, 30]), 1.0).unwrap();
        assert_eq!(e[0], TRIGGER_VIBRATION);
        assert_eq!(u16::from_le_bytes([e[1], e[2]]), 0x300);
        assert_eq!(e[9], 30);
        // Slope 0..2 from 1 to 2: 1, 1.5 -> 2 (half away from zero), then 2.
        let e = encode(&cmd(5, &[0, 2, 1, 2]), 1.0).unwrap();
        let s: Vec<u32> = (0..10).map(|z| zone_strength(&e, z)).collect();
        assert_eq!(s, [1, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
        // Falling slope 0..4 from 8 to 1: the step rounds half away from zero
        // (-3.5 -> -4), so 8, 6, 4, 3, 1.
        let e = encode(&cmd(5, &[0, 4, 8, 1]), 1.0).unwrap();
        let s: Vec<u32> = (0..5).map(|z| zone_strength(&e, z)).collect();
        assert_eq!(s, [8, 6, 4, 3, 1]);
        assert!(encode(&cmd(5, &[0, 2, 0, 2]), 1.0).is_none());
        // Multi-position vibration without frequency is off.
        assert_eq!(encode(&cmd(6, &[0, 8, 8]), 1.0).unwrap()[0], TRIGGER_OFF);
        assert!(encode(&cmd(7, &[]), 1.0).is_none());
    }

    // --- controller settings ------------------------------------------------

    struct Rumble {
        large: u16,
        small: u16,
        duration: u32,
    }

    /// The host side of a DualSense: vibration requests become motor
    /// strengths, trigger requests effect reports.
    struct Controller {
        settings: Settings,
        out: ControllerOutput,
        now: u64,
        triggers: [u8; 2],
        rumble: Vec<Rumble>,
        effects: Vec<DualSenseEffects>,
    }

    impl Controller {
        fn new() -> Self {
            Self::with_levels(DEFAULT_SPEAKER_VOLUME, DEFAULT_VIBRATION_INTENSITY)
        }

        fn with_levels(speaker: u32, vibration: u32) -> Self {
            let settings = Settings::new();
            settings.configure(speaker, vibration);
            assert_eq!(settings.scale(Setting::SpeakerVolume), speaker as f32 / 50.0);
            assert_eq!(settings.scale(Setting::VibrationIntensity), vibration as f32 / 100.0);
            assert_eq!(settings.scale(Setting::TriggerEffectIntensity), 1.0);
            Self {
                settings,
                out: ControllerOutput::default(),
                now: 1000,
                triggers: [0; 2],
                rumble: Vec::new(),
                effects: Vec::new(),
            }
        }

        fn send_vibration(&mut self, v: Vibration) {
            self.rumble.push(Rumble { large: v.large as u16 * 257, small: v.small as u16 * 257, duration: v.duration_ms });
        }

        fn set_rumble(&mut self, large: u8, small: u8) {
            let v = self.out.set_vibration(large, small, self.now, &self.settings);
            self.send_vibration(v);
        }

        fn cycle(&mut self, setting: Setting) {
            self.settings.cycle(setting);
            match setting {
                Setting::VibrationIntensity => {
                    let v = self.out.vibration(self.now, &self.settings);
                    self.send_vibration(v);
                }
                Setting::TriggerEffectIntensity => {
                    if let Some(e) = self.out.trigger_effect(&self.settings) {
                        self.effects.push(e);
                    }
                }
                Setting::SpeakerVolume => {}
            }
        }

        fn set_trigger_effect(&mut self, p: &TriggerEffectParam) -> Result<(), ()> {
            let e = self.out.set_trigger_effect(p, self.triggers, &self.settings)?;
            self.effects.extend(e);
            Ok(())
        }

        fn axis(&mut self, i: usize, value: u8) {
            self.triggers[i] = value;
            self.out.update_trigger_state(self.triggers);
        }

        fn last_effect(&self) -> DualSenseEffects {
            *self.effects.last().expect("missing trigger output")
        }
    }

    #[test]
    fn setting_cycles() {
        let c = Controller::new();
        for setting in [Setting::SpeakerVolume, Setting::VibrationIntensity, Setting::TriggerEffectIntensity] {
            assert_eq!(c.settings.scale(setting), 1.0, "initial setting is not strong");
            let levels: &[f32] =
                if setting == Setting::SpeakerVolume { &[0.0, 0.02, 0.12, 0.42, 1.0] } else { &[0.0, 0.33, 0.66, 1.0] };
            for _ in 0..2 {
                for &level in levels {
                    c.settings.cycle(setting);
                    assert_eq!(c.settings.scale(setting), level, "setting did not cycle or wrap");
                }
            }
        }
    }

    #[test]
    fn global_controller_levels() {
        let mut c = Controller::with_levels(25, 25);
        assert_eq!(c.settings.scale(Setting::SpeakerVolume), 0.5);
        assert_eq!(c.settings.scale(Setting::VibrationIntensity), 0.25);
        c.set_rumble(200, 100);
        let r = c.rumble.last().unwrap();
        assert_eq!((r.large, r.small), (50 * 257, 25 * 257), "global vibration level did not scale rumble");
        c.cycle(Setting::SpeakerVolume);
        c.cycle(Setting::VibrationIntensity);
        assert_eq!(c.settings.scale(Setting::SpeakerVolume), 0.0);
        assert_eq!(c.settings.scale(Setting::VibrationIntensity), 0.0);
        c.cycle(Setting::SpeakerVolume);
        c.cycle(Setting::VibrationIntensity);
        assert_eq!(c.settings.scale(Setting::SpeakerVolume), 0.01);
        assert_eq!(c.settings.scale(Setting::VibrationIntensity), 0.0825);
        let r = c.rumble.last().unwrap();
        assert_eq!((r.large, r.small), (17 * 257, 8 * 257), "cycling did not apply the combined intensity");

        let c = Controller::with_levels(50, 25);
        assert_eq!(c.settings.scale(Setting::SpeakerVolume), 1.0, "speaker midpoint is not unity");
        let c = Controller::with_levels(100, 25);
        assert_eq!(c.settings.scale(Setting::SpeakerVolume), 2.0, "speaker maximum did not add 6 dB");
        c.settings.cycle(Setting::SpeakerVolume);
        assert_eq!(c.settings.scale(Setting::SpeakerVolume), 0.0, "speaker maximum prevented muting");
        let mut c = Controller::with_levels(0, 0);
        assert_eq!(c.settings.scale(Setting::SpeakerVolume), 0.0);
        c.set_rumble(200, 100);
        let r = c.rumble.last().unwrap();
        assert_eq!((r.large, r.small), (0, 0), "zero global vibration level did not mute motors");
    }

    #[test]
    fn vibration_lifetime() {
        let mut c = Controller::new();
        c.set_rumble(255, 1);
        let r = c.rumble.last().unwrap();
        assert_eq!((c.rumble.len(), r.large, r.small, r.duration), (1, 65535, 257, 65535));
        c.now += 100;
        c.cycle(Setting::VibrationIntensity);
        let r = c.rumble.last().unwrap();
        assert_eq!((r.large, r.small), (0, 0), "muting did not stop vibration");
        c.now += 100;
        c.cycle(Setting::VibrationIntensity);
        let r = c.rumble.last().unwrap();
        assert_eq!((r.large, r.small, r.duration), (84 * 257, 257, 65335), "intensity lost a motor or the deadline");
        assert!(c.effects.is_empty(), "vibration intensity resent triggers");
        c.now = 1000 + 65535;
        c.cycle(Setting::VibrationIntensity);
        c.cycle(Setting::VibrationIntensity);
        let r = c.rumble.last().unwrap();
        assert_eq!((r.large, r.small), (0, 0), "expired vibration was resurrected");
        c.set_rumble(0, 0);
        c.cycle(Setting::VibrationIntensity);
        c.cycle(Setting::VibrationIntensity);
        let r = c.rumble.last().unwrap();
        assert_eq!((r.large, r.small), (0, 0), "explicitly stopped vibration was resurrected");
    }

    fn weapon_and_feedback() -> TriggerEffectParam {
        let mut p = TriggerEffectParam { trigger_mask: 3, ..Default::default() };
        p.command[0] = cmd(1, &[2, 8]);
        p.command[1] = cmd(2, &[2, 7, 6]);
        p
    }

    #[test]
    fn masked_triggers_and_validation() {
        let mut c = Controller::new();
        let mut param = weapon_and_feedback();
        c.set_trigger_effect(&param).expect("initial trigger request failed");
        c.cycle(Setting::TriggerEffectIntensity);
        let e = c.last_effect();
        assert_eq!((e.left_trigger()[0], e.right_trigger()[0]), (5, 5), "muting did not disable both triggers");
        for mode in 1..=7 {
            let mut invalid = param;
            invalid.command[1] = TriggerCommand { mode, data: [255; 48] };
            let calls = c.effects.len();
            assert!(c.set_trigger_effect(&invalid).is_err() && c.effects.len() == calls, "mode {mode} accepted");
        }
        param.trigger_mask = 4;
        assert!(c.set_trigger_effect(&param).is_err(), "invalid trigger mask accepted");
        param.trigger_mask = 1;
        param.command[0].data[1] = 6;
        c.set_trigger_effect(&param).unwrap();
        assert_eq!(c.last_effect().enable_bits(), 8, "left-only update touched the right trigger");
        c.cycle(Setting::TriggerEffectIntensity);
        let e = c.last_effect();
        assert_eq!(e.enable_bits(), 12);
        assert_eq!(e.left_trigger()[0], 0x21);
        assert_eq!(zone_strength(e.left_trigger(), 1), 0);
        assert_eq!(zone_strength(e.left_trigger(), 2), 2);
        assert_eq!((e.right_trigger()[0], e.right_trigger()[1], e.right_trigger()[3]), (0x25, 0x84, 1));
        c.cycle(Setting::TriggerEffectIntensity);
        c.cycle(Setting::TriggerEffectIntensity);
        let e = c.last_effect();
        assert_eq!(zone_strength(e.left_trigger(), 2), 6);
        assert_eq!(e.right_trigger()[3], 5, "strong intensity did not recover the original strengths");
        assert!(c.rumble.is_empty(), "trigger intensity resent vibration");
    }

    #[test]
    fn independent_outputs_and_controller_switch() {
        let mut c = Controller::new();
        c.set_rumble(200, 100);
        c.now += 100;
        c.cycle(Setting::VibrationIntensity);
        c.cycle(Setting::VibrationIntensity);
        let r = c.rumble.last().unwrap();
        assert_eq!((r.large / 257, r.small / 257, r.duration), (66, 33, 65435));
        let mut param = TriggerEffectParam { trigger_mask: 1, ..Default::default() };
        param.command[0] = cmd(1, &[0, 8]);
        c.set_trigger_effect(&param).unwrap();
        let (vibration_calls, trigger_calls) = (c.rumble.len(), c.effects.len());
        c.cycle(Setting::SpeakerVolume);
        assert_eq!((c.rumble.len(), c.effects.len()), (vibration_calls, trigger_calls), "speaker volume resent effects");
        // Another controller becomes active: nothing of the old one leaks.
        c.out.reset();
        c.rumble.clear();
        c.effects.clear();
        c.cycle(Setting::VibrationIntensity);
        c.cycle(Setting::TriggerEffectIntensity);
        c.cycle(Setting::TriggerEffectIntensity);
        assert!(c.effects.is_empty(), "previous controller's trigger effects leaked");
        assert!(c.rumble.iter().all(|r| r.large == 0 && r.small == 0), "previous vibration leaked");
    }

    #[test]
    fn trigger_effect_states() {
        let mut c = Controller::new();
        let check = |c: &Controller, left, right, msg: &str| assert_eq!(c.out.trigger_state(), [left, right], "{msg}");
        check(&c, OFF, OFF, "unset trigger effects did not report off");
        let mut param = TriggerEffectParam { trigger_mask: 3, ..Default::default() };
        param.command[0] = cmd(1, &[7, 8]);
        param.command[1] = cmd(2, &[2, 7, 6]);
        c.set_trigger_effect(&param).unwrap();
        let axes = |c: &mut Controller, l, r| {
            c.axis(0, l);
            c.axis(1, r);
        };
        axes(&mut c, 191, 63);
        check(&c, 1, 3, "triggers activated before the start positions");
        axes(&mut c, 192, 64);
        check(&c, 2, 4, "triggers did not activate at the start positions");
        axes(&mut c, 255, 223);
        check(&c, 2, 4, "weapon fired before its end position");
        c.axis(1, 224);
        c.axis(1, 100);
        check(&c, 2, 5, "unpolled weapon firing was lost on partial release");
        c.set_trigger_effect(&param).unwrap();
        check(&c, 2, 5, "repeating a weapon request rearmed a held trigger");
        c.cycle(Setting::TriggerEffectIntensity);
        check(&c, 2, 5, "muting trigger effects changed their states");

        // A rejected two-trigger request keeps both cached commands and states.
        let mut invalid = param;
        invalid.command[0].mode = 0;
        invalid.command[1].data[1] = 2;
        assert!(c.set_trigger_effect(&invalid).is_err(), "invalid weapon end position accepted");
        c.axis(0, 255);
        check(&c, 2, 5, "an invalid request changed cached commands or states");
        c.axis(1, 63);
        c.axis(1, 100);
        check(&c, 2, 4, "unpolled release did not rearm the weapon");
        param.trigger_mask = 2;
        param.command[1].data[1] = 8;
        c.set_trigger_effect(&param).unwrap();
        axes(&mut c, 255, 253);
        check(&c, 2, 4, "weapon end position 8 fired before 254");
        c.axis(1, 254);
        check(&c, 2, 5, "weapon end position 8 did not fire at 254");

        // Masked updates preserve the other trigger, including its fired state.
        param.trigger_mask = 1;
        param.command[0].data[0] = 9;
        c.set_trigger_effect(&param).unwrap();
        c.axis(0, 254);
        check(&c, 1, 5, "feedback position 9 activated before full travel");
        c.axis(0, 255);
        check(&c, 2, 5, "full trigger travel did not activate feedback");
        param.command[0].data[0] = 0;
        c.set_trigger_effect(&param).unwrap();
        c.axis(0, 0);
        check(&c, 2, 5, "feedback at position 0 was suppressed by a dead zone");
        param.command[0].data[1] = 0;
        c.set_trigger_effect(&param).unwrap();
        check(&c, 1, 5, "zero-strength feedback reported active");

        param.command[0] = cmd(4, &[4, 0, 4, 0, 0, 0, 0, 0, 4]);
        c.set_trigger_effect(&param).unwrap();
        for (value, state) in
            [(0, 2), (1, 1), (31, 1), (32, 2), (63, 2), (64, 1), (223, 1), (224, 2), (254, 2), (255, 1)]
        {
            c.axis(0, value);
            check(&c, state, 5, "multi-position feedback used the wrong position");
        }

        param.command[0] = cmd(5, &[3, 7, 2, 6]);
        c.set_trigger_effect(&param).unwrap();
        for (value, state) in [(63, 1), (64, 2), (191, 2), (192, 1), (255, 1)] {
            c.axis(0, value);
            check(&c, state, 5, "slope feedback did not respect its start and end");
        }
        param.command[0].data[0] = 0;
        c.set_trigger_effect(&param).unwrap();
        c.axis(0, 0);
        check(&c, 2, 5, "slope feedback at position 0 was suppressed");

        param.command[0] = cmd(3, &[6, 4, 20]);
        c.set_trigger_effect(&param).unwrap();
        c.axis(0, 159);
        check(&c, 6, 5, "vibration activated before position 6");
        c.axis(0, 160);
        check(&c, 7, 5, "vibration did not activate at position 6");
        param.command[0].data[0] = 0;
        c.set_trigger_effect(&param).unwrap();
        c.axis(0, 0);
        check(&c, 7, 5, "vibration at position 0 was suppressed");
        param.command[0].data[2] = 0;
        c.set_trigger_effect(&param).unwrap();
        check(&c, 6, 5, "zero-frequency vibration reported active");

        param.command[0] = TriggerCommand { mode: 6, ..Default::default() };
        param.command[0].data[0] = 20;
        param.command[0].data[10] = 4;
        c.set_trigger_effect(&param).unwrap();
        c.axis(0, 254);
        check(&c, 6, 5, "multi-position vibration activated before position 9");
        c.axis(0, 255);
        check(&c, 7, 5, "multi-position vibration did not activate at position 9");
        param.command[0].data[0] = 0;
        c.set_trigger_effect(&param).unwrap();
        check(&c, 6, 5, "zero-frequency multi-position vibration reported active");

        // Input reset (released triggers).
        axes(&mut c, 0, 0);
        check(&c, 6, 3, "input reset retained a fired weapon");
        param.command[0].mode = 0;
        c.set_trigger_effect(&param).unwrap();
        check(&c, 0, 3, "off mode retained an effect state");
        c.out.reset();
        check(&c, 0, 0, "switching controllers retained effect states");
    }

    #[test]
    fn param_layout() {
        let mut b = [0u8; TRIGGER_EFFECT_PARAM_SIZE];
        b[0] = 3;
        b[8..12].copy_from_slice(&2u32.to_le_bytes());
        b[16] = 7;
        b[64..68].copy_from_slice(&6u32.to_le_bytes());
        b[72 + 47] = 9;
        let p = TriggerEffectParam::parse(&b);
        assert_eq!(p.trigger_mask, 3);
        assert_eq!((p.command[0].mode, p.command[0].data[0]), (2, 7));
        assert_eq!((p.command[1].mode, p.command[1].data[47]), (6, 9));
    }

    #[test]
    fn audio_haptics_levels_from_rms() {
        assert_eq!(audio_haptics_levels(&[], &[]), (0, 0));
        assert_eq!(audio_haptics_levels(&[0.009; 64], &[0.0; 64]), (0, 0), "under 1%");
        // A square wave of 0.25: RMS 0.25, twice that is half the motor.
        let square: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 0.25 } else { -0.25 }).collect();
        assert_eq!(audio_haptics_levels(&square, &[1.0; 64]), (128, 255));
        assert_eq!(audio_haptics_levels(&[0.5, f32::NAN], &[4.0; 2]), (180, 255), "non-finite skipped, clamped");
    }

    /// Audio haptics raise the motors for 80 ms past the last audible block
    /// and give the title's request back when they run out.
    #[test]
    fn audio_haptics_merge_and_expire() {
        let settings = Settings::new();
        settings.configure(DEFAULT_SPEAKER_VOLUME, 100);
        let mut out = ControllerOutput::default();
        let now = 1000;
        let title = out.set_vibration(100, 50, now, &settings);
        assert_eq!((title.large, title.small), (100, 50));
        // Silence changes nothing.
        assert!(out.set_audio_haptics(0, 0, now, &settings).is_none());
        // Louder audio on the large motor: sent for the audio window.
        let v = out.set_audio_haptics(200, 10, now, &settings).unwrap();
        assert_eq!((v.large, v.small, v.duration_ms), (200, 50, AUDIO_HAPTICS_MS as u32));
        // The same level within the window is not re-sent until what was
        // sent is about to stop; a higher one is.
        assert!(out.set_audio_haptics(200, 10, now + 10, &settings).is_none());
        let v = out.set_audio_haptics(150, 60, now + 20, &settings).unwrap();
        assert_eq!((v.large, v.small), (200, 60), "levels merge within the window");
        assert!(out.set_audio_haptics(150, 60, now + 30, &settings).is_none());
        assert!(out.set_audio_haptics(150, 60, now + 70, &settings).is_some(), "refreshed before it stops");
        // Silence lets it expire; the title's request comes back.
        assert!(out.set_audio_haptics(0, 0, now + 100, &settings).is_none(), "window still open");
        let v = out.set_audio_haptics(0, 0, now + 160, &settings).unwrap();
        assert_eq!((v.large, v.small), (100, 50));
        assert!(v.duration_ms > 60_000, "the title's lifetime");
        assert!(out.expire_audio_haptics(now + 200, &settings).is_none(), "nothing left to expire");
        // Weaker audio than the title's request keeps the title's lifetime.
        let v = out.set_audio_haptics(20, 20, now + 300, &settings).unwrap();
        assert_eq!((v.large, v.small), (100, 50));
        assert!(v.duration_ms > 60_000);
    }
}
