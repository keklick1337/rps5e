// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Codec-independent pieces of the AJM decoders: result flags, sideband
//! structures, gapless state and the decoder trait.

pub use crate::audio::atrac9::PcmEncoding;

pub const RESULT_NOT_INITIALIZED: i32 = 0x0000_0001;
pub const RESULT_INVALID_DATA: i32 = 0x0000_0002;
pub const RESULT_INVALID_PARAMETER: i32 = 0x0000_0004;
pub const RESULT_PARTIAL_INPUT: i32 = 0x0000_0008;
pub const RESULT_NOT_ENOUGH_ROOM: i32 = 0x0000_0010;
pub const RESULT_TOO_MANY_CHANNELS: i32 = 0x0000_0040;
pub const RESULT_CODEC_ERROR: i32 = 0x4000_0000;
pub const RESULT_FATAL: i32 = 0x8000_0000u32 as i32;

/// Codec-specific instance flags start at bit 32.
pub const INSTANCE_FLAG_CODEC_OFFSET: u32 = 32;

/// AjmSidebandFormat (24 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SidebandFormat {
    pub channels: u32,
    pub channel_mask: u32,
    pub sample_rate: u32,
    pub encoding: u32,
    pub bitrate: u32,
}

impl SidebandFormat {
    pub const BYTES: usize = 24;

    pub fn new(channels: u32, sample_rate: u32, encoding: PcmEncoding) -> Self {
        Self { channels, channel_mask: channel_mask(channels), sample_rate, encoding: encoding as u32, bitrate: 0 }
    }

    pub fn to_bytes(self) -> [u8; Self::BYTES] {
        let mut b = [0u8; Self::BYTES];
        for (i, v) in [self.channels, self.channel_mask, self.sample_rate, self.encoding, self.bitrate].iter().enumerate() {
            b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        b
    }
}

pub fn channel_mask(channels: u32) -> u32 {
    match channels {
        1 => 0x4,
        2 => 0x3,
        3 => 0x7,
        4 => 0x33,
        5 => 0x607,
        6 => 0x60F,
        7 => 0x70F,
        8 => 0x63F,
        _ => 0,
    }
}

/// AjmSidebandGaplessDecode (8 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GaplessDecode {
    pub total_samples: u32,
    pub skip_samples: u16,
    pub skipped_samples: u16,
}

impl GaplessDecode {
    pub const BYTES: usize = 8;

    pub fn from_bytes(b: &[u8]) -> Self {
        Self {
            total_samples: u32::from_le_bytes(b[0..4].try_into().unwrap()),
            skip_samples: u16::from_le_bytes([b[4], b[5]]),
            skipped_samples: u16::from_le_bytes([b[6], b[7]]),
        }
    }

    pub fn to_bytes(self) -> [u8; Self::BYTES] {
        let mut b = [0u8; Self::BYTES];
        b[0..4].copy_from_slice(&self.total_samples.to_le_bytes());
        b[4..6].copy_from_slice(&self.skip_samples.to_le_bytes());
        b[6..8].copy_from_slice(&self.skipped_samples.to_le_bytes());
        b
    }
}

/// Gapless trimming: the configured values and the live countdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GaplessState {
    pub init: GaplessDecode,
    pub current: GaplessDecode,
}

impl GaplessState {
    pub fn has_sample_limit(&self) -> bool {
        !matches!(self.init.total_samples, 0 | u32::MAX)
    }

    pub fn is_end(&self) -> bool {
        self.has_sample_limit() && self.current.total_samples == 0
    }

    pub fn reset(&mut self) {
        self.current = self.init;
        self.current.skipped_samples = 0;
    }

    pub fn set(&mut self, params: GaplessDecode, reset: bool) {
        self.init.total_samples = params.total_samples;
        self.init.skip_samples = params.skip_samples;
        if reset || self.current == GaplessDecode::default() {
            self.reset();
        }
    }

    /// Splits a decoded block of `samples` into (skip, write) and advances
    /// the skip countdown; the total limit is applied by the caller through
    /// [`GaplessState::commit`] once the written count is known.
    pub fn plan(&mut self, samples: u32) -> (u32, u32) {
        let (skip, write) = self.preview(samples);
        self.current.skip_samples -= skip as u16;
        (skip, write)
    }

    /// (skip, write) for a block of `samples` without changing the state.
    pub fn preview(&self, samples: u32) -> (u32, u32) {
        let skip = samples.min(self.current.skip_samples as u32);
        let mut write = samples - skip;
        if self.has_sample_limit() {
            write = write.min(self.current.total_samples);
        }
        (skip, write)
    }

    /// Accounts `written` output samples out of a block of `samples` of
    /// which `skip` were skipped.
    pub fn commit(&mut self, samples: u32, written: u32) {
        if self.has_sample_limit() {
            self.current.total_samples -= self.current.total_samples.min(written);
        }
        let skipped = self.current.skipped_samples as u32 + (samples - written);
        self.current.skipped_samples = skipped.min(u16::MAX as u32) as u16;
    }
}

/// Outcome of one decoder operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecodeResult {
    pub result: i32,
    pub internal_result: i32,
    pub input_consumed: usize,
    pub output_written: usize,
    pub total_decoded_samples: u64,
    pub frames: u32,
    pub format: SidebandFormat,
}

impl DecodeResult {
    pub fn with_result(result: i32) -> Self {
        Self { result, ..Default::default() }
    }
}

/// One AJM codec instance.
pub trait AjmDecoder: Send {
    /// Codec initialization parameters (sceAjmBatchJobInitialize / control
    /// jobs with the initialize flag).
    fn initialize(&mut self, _params: &[u8]) -> DecodeResult {
        self.make_result()
    }

    /// Restart the stream (clear-context / control reset).
    fn reset(&mut self);

    fn decode(&mut self, input: &[u8], output: &mut [u8], multiple_frames: bool, gapless: &mut GaplessState)
        -> DecodeResult;

    /// Codec-specific sideband block written by get-codec-info requests
    /// (shorter than `codec_info_size` when nothing is known yet; the rest
    /// stays zero).
    fn codec_info(&self) -> Vec<u8> {
        Vec::new()
    }

    /// Size of the codec-info sideband block.
    fn codec_info_size(&self) -> usize {
        0
    }

    fn format(&self) -> SidebandFormat;

    fn total_decoded_samples(&self) -> u64;

    fn make_result(&self) -> DecodeResult {
        DecodeResult { total_decoded_samples: self.total_decoded_samples(), format: self.format(), ..Default::default() }
    }
}

/// Instance flags bits 7..9: PCM sample encoding (S16 / S32 / Float).
pub fn flag_encoding(flags: u64) -> Option<PcmEncoding> {
    match (flags >> 7) & 7 {
        0 => Some(PcmEncoding::S16),
        1 => Some(PcmEncoding::S32),
        2 => Some(PcmEncoding::Float),
        _ => None,
    }
}

/// Instance flags bits 0..6: maximum channel count.
pub fn flag_channels(flags: u64) -> u32 {
    (flags & 0x7F) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gapless_set_and_reset() {
        let mut g = GaplessState::default();
        g.set(GaplessDecode { total_samples: 1000, skip_samples: 100, skipped_samples: 7 }, false);
        assert_eq!(g.current, GaplessDecode { total_samples: 1000, skip_samples: 100, skipped_samples: 0 });
        // Once configured, a non-reset set keeps the live state.
        g.current.skip_samples = 3;
        g.set(GaplessDecode { total_samples: 5, skip_samples: 5, skipped_samples: 0 }, false);
        assert_eq!(g.current.skip_samples, 3);
        assert_eq!(g.init.total_samples, 5);
        g.set(GaplessDecode { total_samples: 5, skip_samples: 5, skipped_samples: 0 }, true);
        assert_eq!(g.current.skip_samples, 5);
        assert!(g.has_sample_limit());
        g.init.total_samples = u32::MAX;
        assert!(!g.has_sample_limit());
    }

    #[test]
    fn gapless_plan_commit() {
        let mut g = GaplessState::default();
        g.set(GaplessDecode { total_samples: 300, skip_samples: 100, skipped_samples: 0 }, true);
        let (skip, write) = g.plan(256);
        assert_eq!((skip, write), (100, 156));
        g.commit(256, write);
        assert_eq!(g.current, GaplessDecode { total_samples: 144, skip_samples: 0, skipped_samples: 100 });
        let (skip, write) = g.plan(256);
        assert_eq!((skip, write), (0, 144));
        g.commit(256, write);
        assert!(g.is_end());
        assert_eq!(g.current.skipped_samples, 100 + 112);
    }

    #[test]
    fn format_and_masks() {
        let f = SidebandFormat::new(8, 48000, PcmEncoding::Float);
        let b = f.to_bytes();
        assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), 0x63F);
        assert_eq!(u32::from_le_bytes(b[12..16].try_into().unwrap()), 2);
        assert_eq!(channel_mask(5), 0x607);
        assert_eq!(flag_encoding(2 << 7), Some(PcmEncoding::Float));
        assert_eq!(flag_encoding(3 << 7), None);
        assert_eq!(flag_channels(0x182), 2);
    }
}
