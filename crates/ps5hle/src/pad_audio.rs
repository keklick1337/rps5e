// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Controller audio: AudioOut pad speaker (PADSPK) and vibration ports
//! are offered to the host controller layer, which plays them on the
//! active DualSense (speaker / haptic actuators). A port whose block no
//! controller took plays on the main output (pad speaker) or drives the
//! first pad's rumble motors by its channel levels (vibration).

use crate::audio::audio_out::{PORT_TYPE_PADSPK, PORT_TYPE_VIBRATION};
use std::sync::RwLock;

/// One block of a port, as the guest submitted it.
#[derive(Clone, Copy)]
pub struct PadAudioBlock<'a> {
    /// Interleaved PCM: f32 or s16, little endian.
    pub data: &'a [u8],
    pub frames: u32,
    pub channels: u32,
    pub is_float: bool,
    /// Per-channel port volume (32768 = unity).
    pub volume: &'a [i32],
    /// Controller setting gain.
    pub gain: f32,
    /// Optional per-channel gains of AudioOut2.
    pub gains: Option<&'a [f32]>,
}

impl PadAudioBlock<'_> {
    /// Sample `index` (frame * channels + channel) as f32.
    pub fn sample(&self, index: usize) -> f32 {
        if self.is_float {
            let o = index * 4;
            f32::from_le_bytes([self.data[o], self.data[o + 1], self.data[o + 2], self.data[o + 3]])
        } else {
            let o = index * 2;
            i16::from_le_bytes([self.data[o], self.data[o + 1]]) as f32 / 32768.0
        }
    }
}

/// A port's stream to the controller.
pub trait PadAudioStream: Send {
    /// Queue a block on the active controller; returns how long the queued
    /// audio still plays (us), 0 when no controller took it.
    fn queue(&mut self, block: &PadAudioBlock) -> u64;
    /// The stream goes through the controller's Bluetooth link (paced by
    /// its own sender).
    fn uses_bluetooth(&self) -> bool;
}

/// Opens a controller stream for a port: (sample rate, pad speaker port).
pub type PadAudioOpener = fn(u32, bool) -> Option<Box<dyn PadAudioStream>>;

static OPENER: RwLock<Option<PadAudioOpener>> = RwLock::new(None);

/// Host controller layer hook.
pub fn set_pad_audio_opener(opener: Option<PadAudioOpener>) {
    *OPENER.write().unwrap_or_else(|e| e.into_inner()) = opener;
}

/// Controller stream for a new pad speaker / vibration port.
pub fn open(freq: u32, speaker: bool) -> Option<Box<dyn PadAudioStream>> {
    let opener = *OPENER.read().unwrap_or_else(|e| e.into_inner());
    opener.and_then(|open| open(freq, speaker))
}

// --- AudioOut port policy ---------------------------------------------------

/// Controller stream for a new AudioOut port; None for ports that do not
/// feed the controller.
pub fn open_port(port_type: i32, freq: u32) -> Option<Box<dyn PadAudioStream>> {
    match port_type {
        PORT_TYPE_PADSPK => open(freq, true),
        PORT_TYPE_VIBRATION => open(freq, false),
        _ => None,
    }
}

/// Gain the controller settings apply to a port's blocks.
pub fn port_gain(port_type: i32) -> f32 {
    match port_type {
        PORT_TYPE_PADSPK => crate::pad::setting_scale(crate::pad::Setting::SpeakerVolume),
        PORT_TYPE_VIBRATION => crate::pad::setting_scale(crate::pad::Setting::VibrationIntensity),
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_samples() {
        let f: Vec<u8> = [0.5f32, -0.25].iter().flat_map(|v| v.to_le_bytes()).collect();
        let b = PadAudioBlock { data: &f, frames: 1, channels: 2, is_float: true, volume: &[32768; 2], gain: 1.0, gains: None };
        assert_eq!((b.sample(0), b.sample(1)), (0.5, -0.25));
        let i: Vec<u8> = [16384i16, -32768].iter().flat_map(|v| v.to_le_bytes()).collect();
        let b = PadAudioBlock { data: &i, is_float: false, ..b };
        assert_eq!((b.sample(0), b.sample(1)), (0.5, -1.0));
    }
}
