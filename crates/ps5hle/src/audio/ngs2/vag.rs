// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! PS-ADPCM ("VAG") decoder.
//!
//! NGS2 sampler voices point at waveforms wrapped in the Sony "VAGp"
//! container: a 48-byte big-endian header followed by 16-byte ADPCM frames
//! (1 byte predictor/shift, 1 byte flags, then 14 bytes = 28 nibbles = 28
//! samples). The predictor table and nibble decode are the standard SPU
//! ADPCM algorithm.

/// Predictor filters (scaled by 1/64).
const COEFF0: [i32; 5] = [0, 60, 115, 98, 122];
const COEFF1: [i32; 5] = [0, 0, -52, -55, -60];

pub const VAG_HEADER_SIZE: usize = 0x30;
/// Bytes and samples of one ADPCM unit (per channel).
pub const UNIT_BYTES: u64 = 16;
pub const UNIT_SAMPLES: u64 = 28;
const VAG_MAGIC: u32 = 0x5641_4770; // "VAGp"

/// Decoded mono PCM16 waveform.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Waveform {
    pub samples: Vec<i16>,
    pub sample_rate: i32,
    /// -1 when the waveform does not loop.
    pub loop_start: i32,
    pub loop_end: i32,
}

pub(crate) fn be32(data: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

/// True when the buffer begins with a "VAGp" container header.
pub fn is_vag(data: &[u8]) -> bool {
    data.len() >= VAG_HEADER_SIZE && be32(data, 0) == VAG_MAGIC
}

/// Decode a full "VAGp" container into mono PCM16. `None` when the header is
/// missing/short or nothing could be decoded.
pub fn try_decode(data: &[u8]) -> Option<Waveform> {
    if !is_vag(data) {
        return None;
    }
    // Header (big-endian): +0x0C data size, +0x10 sample rate.
    let declared = be32(data, 0x0C) as i32;
    let mut sample_rate = be32(data, 0x10) as i32;
    if sample_rate <= 0 {
        sample_rate = 48000;
    }
    let body = &data[VAG_HEADER_SIZE..];
    // Trust the declared payload size when it fits; otherwise decode what is
    // actually present (some tools pad or under-report).
    let available = (body.len() - body.len() % 16) as i32;
    let frame_bytes = if declared > 0 && declared <= available {
        declared - declared % 16
    } else {
        available
    };
    if frame_bytes <= 0 {
        return None;
    }
    let w = decode(&body[..frame_bytes as usize], sample_rate);
    (!w.samples.is_empty()).then_some(w)
}

/// Decode raw 16-byte PS-ADPCM units of one channel into PCM16.
pub fn decode_units(frames: &[u8]) -> Vec<i16> {
    decode(frames, 0).samples
}

/// Decode raw 16-byte PS-ADPCM frames (no container header) into PCM16 and
/// resolve loop points from the per-frame flag bytes.
pub fn decode(frames: &[u8], sample_rate: i32) -> Waveform {
    let frame_count = frames.len() / 16;
    let mut samples = vec![0i16; frame_count * 28];
    let mut loop_start: i32 = -1;
    let mut loop_end: i32 = -1;
    let (mut hist1, mut hist2) = (0i32, 0i32);
    let mut out = 0usize;

    for frame in 0..frame_count {
        let off = frame * 16;
        let header = frames[off];
        let shift = (header & 0x0F) as u32;
        let mut filter = ((header >> 4) & 0x0F) as usize;
        if filter > 4 {
            filter = 0;
        }
        // Loop marker (exact values): 3 = loop start, 6 = loop end + jump
        // back, 1/7 = one-shot end.
        let flags = frames[off + 1];
        if flags == 0x03 {
            loop_start = out as i32;
        }
        let (f0, f1) = (COEFF0[filter], COEFF1[filter]);
        for i in 0..14 {
            let d = frames[off + 2 + i];
            for nibble in 0..2 {
                let raw = if nibble == 0 { d & 0x0F } else { d >> 4 } as u16;
                // Sign-extend the 4-bit sample from the top nibble, then scale.
                let s = (((raw << 12) as i16) as i32) >> shift;
                let predicted = (hist1 * f0 + hist2 * f1) >> 6;
                let sample = (s + predicted).clamp(i16::MIN as i32, i16::MAX as i32);
                samples[out] = sample as i16;
                out += 1;
                hist2 = hist1;
                hist1 = sample;
            }
        }
        if flags == 0x06 {
            loop_end = out as i32;
        } else if flags == 0x01 || flags == 0x07 {
            break;
        }
    }

    // A one-shot end marker can stop before the declared frame count.
    samples.truncate(out);
    if loop_start >= 0 && loop_end <= loop_start {
        loop_end = out as i32;
    }
    Waveform { samples, sample_rate, loop_start, loop_end }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(header: u8, flags: u8, nibbles: [u8; 28]) -> [u8; 16] {
        let mut f = [0u8; 16];
        f[0] = header;
        f[1] = flags;
        for i in 0..14 {
            f[2 + i] = (nibbles[2 * i] & 0xF) | (nibbles[2 * i + 1] << 4);
        }
        f
    }

    #[test]
    fn ngs2_vag_decodes_predictor_and_shift() {
        // Filter 0, shift 12: nibble n -> sign-extended n (raw<<12>>12).
        let mut n = [0u8; 28];
        n[0] = 1;
        n[1] = 0xF; // -1
        n[2] = 7;
        n[3] = 8; // -8
        let w = decode(&frame(0x0C, 0, n), 44100);
        assert_eq!(w.samples.len(), 28);
        assert_eq!(&w.samples[..5], &[1, -1, 7, -8, 0]);
        assert_eq!(w.loop_start, -1);
        assert_eq!(w.sample_rate, 44100);

        // Filter 1 (60/64 * hist1), shift 0: first nibble 1 -> 4096, then
        // zeros decay: 4096*60>>6 = 3840, 3840*60>>6 = 3600.
        let mut n = [0u8; 28];
        n[0] = 1;
        let w = decode(&frame(0x10, 0, n), 48000);
        assert_eq!(&w.samples[..3], &[4096, 3840, 3600]);
    }

    #[test]
    fn ngs2_vag_end_and_loop_flags() {
        let z = [0u8; 28];
        // One-shot end on frame 1 stops before frame 2.
        let mut data = Vec::new();
        data.extend_from_slice(&frame(0, 0, z));
        data.extend_from_slice(&frame(0, 1, z));
        data.extend_from_slice(&frame(0, 0, z));
        let w = decode(&data, 48000);
        assert_eq!(w.samples.len(), 56);
        assert_eq!((w.loop_start, w.loop_end), (-1, -1));

        // Loop start on frame 1, loop end on frame 2.
        let mut data = Vec::new();
        data.extend_from_slice(&frame(0, 0, z));
        data.extend_from_slice(&frame(0, 3, z));
        data.extend_from_slice(&frame(0, 6, z));
        let w = decode(&data, 48000);
        assert_eq!((w.loop_start, w.loop_end), (28, 84));

        // Loop start without an end loops to the decoded length.
        let mut data = Vec::new();
        data.extend_from_slice(&frame(0, 3, z));
        data.extend_from_slice(&frame(0, 0, z));
        let w = decode(&data, 48000);
        assert_eq!((w.loop_start, w.loop_end), (0, 56));
    }

    #[test]
    fn ngs2_vag_container() {
        let mut data = vec![0u8; VAG_HEADER_SIZE];
        data[..4].copy_from_slice(b"VAGp");
        data[0x0C..0x10].copy_from_slice(&32u32.to_be_bytes());
        data[0x10..0x14].copy_from_slice(&22050u32.to_be_bytes());
        let mut n = [0u8; 28];
        n[0] = 2;
        data.extend_from_slice(&frame(0x0C, 0, n));
        data.extend_from_slice(&frame(0x0C, 0, n));
        data.extend_from_slice(&frame(0x0C, 0, n)); // beyond declared size
        let w = try_decode(&data).unwrap();
        assert_eq!(w.samples.len(), 56);
        assert_eq!(w.samples[0], 2);
        assert_eq!(w.sample_rate, 22050);
        assert!(try_decode(&data[..0x20]).is_none());
    }
}
