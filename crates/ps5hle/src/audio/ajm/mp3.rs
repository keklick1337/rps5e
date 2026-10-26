// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Stateful MPEG audio decoder for libSceAjm codec 0 (MP3).
//!
//! Titles feed small packets (often less than a frame), so unconsumed bytes
//! are kept between calls. Each call scans the pending + new bytes for
//! complete MPEG audio frames and decodes them one by one with a persistent
//! decoder (the layer 3 bit reservoir spans frames), writing interleaved
//! s16 or f32 PCM until the next frame would not fit the output.

use symphonia_core::audio::{SampleBuffer, SignalSpec};
use symphonia_core::codecs::{CodecParameters, Decoder, DecoderOptions, CODEC_TYPE_MP1, CODEC_TYPE_MP2, CODEC_TYPE_MP3};
use symphonia_core::formats::Packet;

use super::decoder::*;

/// Parsed MPEG audio frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    /// 1 = MPEG1, 2 = MPEG2, 25 = MPEG2.5.
    pub version: u8,
    pub layer: u8,
    pub bitrate_kbps: u32,
    pub sample_rate: u32,
    pub padding: bool,
    pub channels: usize,
    /// Whole frame length including the 4-byte header.
    pub frame_length: usize,
    pub samples_per_frame: usize,
}

const BITRATES_V1: [[u32; 15]; 3] = [
    [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448],
    [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384],
    [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
];
const BITRATES_V2: [[u32; 15]; 2] = [
    [0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256],
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
];

/// Parse a 4-byte MPEG audio frame header (free-format frames are rejected).
pub fn parse_header(h: &[u8]) -> Option<FrameHeader> {
    if h.len() < 4 || h[0] != 0xFF || h[1] & 0xE0 != 0xE0 {
        return None;
    }
    let version = match (h[1] >> 3) & 3 {
        0 => 25,
        2 => 2,
        3 => 1,
        _ => return None,
    };
    let layer = match (h[1] >> 1) & 3 {
        1 => 3,
        2 => 2,
        3 => 1,
        _ => return None,
    };
    let br_idx = (h[2] >> 4) as usize;
    let sr_idx = ((h[2] >> 2) & 3) as usize;
    if br_idx == 0 || br_idx == 15 || sr_idx == 3 {
        return None;
    }
    let bitrate_kbps = if version == 1 {
        BITRATES_V1[layer as usize - 1][br_idx]
    } else {
        BITRATES_V2[if layer == 1 { 0 } else { 1 }][br_idx]
    };
    let base_rate = [44100, 48000, 32000][sr_idx];
    let sample_rate = match version {
        1 => base_rate,
        2 => base_rate / 2,
        _ => base_rate / 4,
    };
    let padding = h[2] & 2 != 0;
    let channels = if (h[3] >> 6) == 3 { 1 } else { 2 };
    let br = bitrate_kbps * 1000;
    let (frame_length, samples_per_frame) = match layer {
        1 => (((12 * br / sample_rate) + padding as u32) as usize * 4, 384),
        2 => ((144 * br / sample_rate + padding as u32) as usize, 1152),
        _ if version == 1 => ((144 * br / sample_rate + padding as u32) as usize, 1152),
        _ => ((72 * br / sample_rate + padding as u32) as usize, 576),
    };
    if frame_length < 4 {
        return None;
    }
    Some(FrameHeader { version, layer, bitrate_kbps, sample_rate, padding, channels, frame_length, samples_per_frame })
}

/// Next complete frame at or after `from`: (offset, header). None when no
/// complete frame is available (the tail is kept for the next call).
fn next_frame(data: &[u8], from: usize) -> Option<(usize, FrameHeader)> {
    let mut i = from;
    while i + 4 <= data.len() {
        if let Some(h) = parse_header(&data[i..]) {
            if i + h.frame_length > data.len() {
                return None;
            }
            return Some((i, h));
        }
        i += 1;
    }
    None
}

/// AjmSidebandDecMp3CodecInfo (16 bytes) from a raw frame header.
fn codec_info_bytes(h: &[u8; 4]) -> [u8; 16] {
    let header = u32::from_be_bytes(*h);
    let mut b = [0u8; 16];
    b[0..4].copy_from_slice(&header.to_le_bytes());
    b[4] = ((header >> 16) & 1) as u8; // protection bit
    b[5] = ((header >> 6) & 3) as u8; // channel mode
    b[6] = ((header >> 4) & 3) as u8; // mode extension
    b[7] = ((header >> 3) & 1) as u8; // copyright
    b[8] = ((header >> 2) & 1) as u8; // original
    b[9] = (header & 3) as u8; // emphasis
    b
}

/// Largest run of bytes kept between jobs while no frame is complete.
const MAX_PENDING: usize = 4096;

pub struct AjmMp3Decoder {
    decoder: Option<(u8, Box<dyn Decoder>)>,
    /// Input not yet consumed by a decoded frame (kept between jobs: titles
    /// feed packets smaller than a frame).
    pending: Vec<u8>,
    /// Reused interleaved f32 scratch.
    samples: Option<SampleBuffer<f32>>,
    encoding: PcmEncoding,
    channels: u32,
    sample_rate: u32,
    bitrate: u32,
    last_header: Option<[u8; 4]>,
    total_decoded_samples: u64,
}

impl AjmMp3Decoder {
    pub fn new(channels: u32, sample_rate: u32, encoding: PcmEncoding) -> Self {
        Self {
            decoder: None,
            pending: Vec::new(),
            samples: None,
            encoding,
            channels,
            sample_rate,
            bitrate: 0,
            last_header: None,
            total_decoded_samples: 0,
        }
    }

    fn decoder_for(&mut self, layer: u8) -> Option<&mut Box<dyn Decoder>> {
        if self.decoder.as_ref().map(|d| d.0) != Some(layer) {
            let codec = match layer {
                1 => CODEC_TYPE_MP1,
                2 => CODEC_TYPE_MP2,
                _ => CODEC_TYPE_MP3,
            };
            let mut params = CodecParameters::new();
            params.for_codec(codec);
            let d = symphonia_bundle_mp3::MpaDecoder::try_new(&params, &DecoderOptions::default()).ok()?;
            self.decoder = Some((layer, Box::new(d)));
        }
        self.decoder.as_mut().map(|d| &mut d.1)
    }

    /// Decode the frame at pending[off..end] into the f32 scratch; returns
    /// the interleaved sample count, or None on a decode error.
    fn decode_frame(&mut self, off: usize, end: usize, layer: u8) -> Option<usize> {
        self.decoder_for(layer)?;
        let dec = &mut self.decoder.as_mut()?.1;
        let packet = Packet::new_from_slice(0, 0, 0, &self.pending[off..end]);
        let buf = dec.decode(&packet).ok()?;
        let spec: SignalSpec = *buf.spec();
        let frames = buf.frames();
        let fits = self
            .samples
            .as_ref()
            .is_some_and(|s| s.capacity() >= frames * spec.channels.count());
        if !fits {
            self.samples = Some(SampleBuffer::<f32>::new(frames.max(1152) as u64, spec));
        }
        let sb = self.samples.as_mut().unwrap();
        sb.copy_interleaved_ref(buf);
        Some(sb.len())
    }
}

fn write_samples(src: &[f32], dst: &mut [u8], encoding: PcmEncoding) {
    match encoding {
        PcmEncoding::S16 => {
            for (d, &s) in dst.chunks_exact_mut(2).zip(src) {
                let v = (s * 32768.0).round_ties_even().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                d.copy_from_slice(&v.to_le_bytes());
            }
        }
        PcmEncoding::S32 => {
            for (d, &s) in dst.chunks_exact_mut(4).zip(src) {
                let v = (s as f64 * 2147483648.0).round_ties_even().clamp(i32::MIN as f64, i32::MAX as f64) as i32;
                d.copy_from_slice(&v.to_le_bytes());
            }
        }
        PcmEncoding::Float => {
            for (d, &s) in dst.chunks_exact_mut(4).zip(src) {
                d.copy_from_slice(&s.to_le_bytes());
            }
        }
    }
}

impl AjmDecoder for AjmMp3Decoder {
    fn reset(&mut self) {
        self.decoder = None;
        self.pending.clear();
        self.total_decoded_samples = 0;
        self.last_header = None;
        self.bitrate = 0;
    }

    /// Decodes whole frames from the pending + new bytes. A frame that does
    /// not fit the output stays pending for the next job.
    fn decode(&mut self, input: &[u8], output: &mut [u8], multiple_frames: bool, gapless: &mut GaplessState) -> DecodeResult {
        let mut result = self.make_result();
        if input.is_empty() && self.pending.is_empty() {
            result.result = RESULT_PARTIAL_INPUT;
            return result;
        }
        let pending_before = self.pending.len();
        self.pending.extend_from_slice(input);

        let bps = self.encoding.bytes_per_sample();
        let mut cursor = 0usize;
        let mut consumed = 0usize;
        let mut decoded = false;
        while let Some((off, h)) = next_frame(&self.pending, cursor) {
            if gapless.is_end() {
                break;
            }
            let end = off + h.frame_length;
            let (_, out_samples) = gapless.preview(h.samples_per_frame as u32);
            if result.output_written + out_samples as usize * h.channels * bps > output.len() {
                result.result |= RESULT_NOT_ENOUGH_ROOM;
                break;
            }
            self.last_header = Some(self.pending[off..off + 4].try_into().unwrap());
            let Some(n) = self.decode_frame(off, end, h.layer) else {
                // Corrupt frame: restart the decoder and drop the frame.
                self.decoder = None;
                consumed = end;
                cursor = end;
                continue;
            };
            consumed = end;
            cursor = end;
            if n == 0 {
                continue;
            }
            self.channels = h.channels as u32;
            self.sample_rate = h.sample_rate;
            self.bitrate = h.bitrate_kbps * 1000;
            let per_channel = (n / h.channels.max(1)) as u32;
            let (skip, write) = gapless.plan(per_channel);
            let bytes = write as usize * h.channels * bps;
            let first = skip as usize * h.channels;
            let src = &self.samples.as_ref().unwrap().samples()[first..first + write as usize * h.channels];
            write_samples(src, &mut output[result.output_written..result.output_written + bytes], self.encoding);
            gapless.commit(skip + write, write);
            result.output_written += bytes;
            result.frames += 1;
            self.total_decoded_samples += write as u64;
            decoded = true;
            if !multiple_frames {
                break;
            }
        }
        // Bytes that cannot start a frame within the largest frame size are
        // dropped so junk cannot accumulate.
        let keep_from = consumed.max(self.pending.len().saturating_sub(MAX_PENDING));
        self.pending.drain(..keep_from.min(self.pending.len()));
        result.input_consumed = keep_from.saturating_sub(pending_before).min(input.len());
        if !decoded && result.result == 0 {
            result.result = RESULT_PARTIAL_INPUT;
        }
        result.total_decoded_samples = self.total_decoded_samples;
        result.format = self.format();
        result
    }

    fn codec_info(&self) -> Vec<u8> {
        self.last_header.map(|h| codec_info_bytes(&h).to_vec()).unwrap_or_default()
    }

    fn codec_info_size(&self) -> usize {
        16
    }

    fn format(&self) -> SidebandFormat {
        let mut f = SidebandFormat::new(self.channels, self.sample_rate, self.encoding);
        f.bitrate = self.bitrate;
        f
    }

    fn total_decoded_samples(&self) -> u64 {
        self.total_decoded_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A silent MPEG1 layer 3 frame: header + all-zero side info and data
    /// (part2_3_length 0 => every granule decodes to zero).
    fn silent_frame(bitrate_idx: u8, mono: bool) -> Vec<u8> {
        let mut h = [0xFF, 0xFB, (bitrate_idx << 4) | (1 << 2), if mono { 0xC0 } else { 0x00 }];
        h[2] |= 0; // 48 kHz, no padding
        let hdr = parse_header(&h).unwrap();
        let mut f = vec![0u8; hdr.frame_length];
        f[..4].copy_from_slice(&h);
        f
    }

    #[test]
    fn header_parsing() {
        let h = parse_header(&[0xFF, 0xFB, 0x90, 0x44]).unwrap(); // 128 kbps 44.1 kHz joint stereo
        assert_eq!((h.version, h.layer, h.bitrate_kbps, h.sample_rate), (1, 3, 128, 44100));
        assert_eq!(h.frame_length, 417);
        assert_eq!(h.samples_per_frame, 1152);
        assert_eq!(h.channels, 2);
        let h = parse_header(&[0xFF, 0xFB, 0x92, 0xC4]).unwrap(); // padded, mono
        assert_eq!(h.frame_length, 418);
        assert_eq!(h.channels, 1);
        // MPEG2 layer 3, 64 kbps, 24 kHz.
        let h = parse_header(&[0xFF, 0xF3, 0x84, 0x00]).unwrap();
        assert_eq!((h.version, h.sample_rate, h.samples_per_frame), (2, 24000, 576));
        assert_eq!(h.frame_length, 72 * 64000 / 24000);
        assert!(parse_header(&[0xFF, 0xFB, 0xF0, 0x00]).is_none());
        assert!(parse_header(&[0xFF, 0xFB, 0x0C, 0x00]).is_none());
    }

    fn decoder(encoding: PcmEncoding) -> AjmMp3Decoder {
        AjmMp3Decoder::new(2, 48000, encoding)
    }

    #[test]
    fn decodes_silent_frames_with_packet_splitting() {
        let frame = silent_frame(9, false); // 128 kbps 48 kHz stereo: 384 bytes
        assert_eq!(frame.len(), 384);
        let stream: Vec<u8> = frame.iter().cycle().take(frame.len() * 3).copied().collect();
        let mut d = decoder(PcmEncoding::S16);
        let mut g = GaplessState::default();
        let mut out = vec![0x55u8; 1152 * 2 * 2 * 4];
        // First packet: 1.5 frames -> one frame decoded, half kept pending.
        let r = d.decode(&stream[..576], &mut out, true, &mut g);
        assert_eq!(r.result, 0);
        assert_eq!(r.frames, 1);
        assert_eq!(r.output_written, 1152 * 2 * 2);
        assert_eq!(r.input_consumed, 384);
        assert!(out[..r.output_written].iter().all(|&b| b == 0));
        assert_eq!(r.format.bitrate, 128000);
        assert_eq!(r.format.channels, 2);
        // Second packet: the rest -> two more frames.
        let r = d.decode(&stream[576..], &mut out, true, &mut g);
        assert_eq!(r.frames, 2);
        assert_eq!(r.output_written, 2 * 1152 * 2 * 2);
        assert_eq!(r.input_consumed, 1152 - 576);
        assert_eq!(d.total_decoded_samples(), 3 * 1152);
        // Codec info: the last frame header.
        let info = d.codec_info();
        assert_eq!(u32::from_le_bytes(info[0..4].try_into().unwrap()), 0xFFFB_9400);
        assert_eq!(info[4], 1); // no CRC: protection bit set
    }

    #[test]
    fn single_frame_mode_and_encodings() {
        let frame = silent_frame(9, false);
        let two: Vec<u8> = frame.iter().cycle().take(frame.len() * 2).copied().collect();
        for (enc, bps) in [(PcmEncoding::S16, 2), (PcmEncoding::S32, 4), (PcmEncoding::Float, 4)] {
            let mut d = decoder(enc);
            let mut out = vec![0x55u8; 1152 * 2 * 4 * 2];
            let r = d.decode(&two, &mut out, false, &mut GaplessState::default());
            assert_eq!(r.frames, 1);
            assert_eq!(r.output_written, 1152 * 2 * bps);
            assert_eq!(r.format.encoding, enc as u32);
            // The second frame is decoded by the next job.
            let r = d.decode(&[], &mut out, false, &mut GaplessState::default());
            assert_eq!(r.frames, 1);
        }
    }

    #[test]
    fn gapless_trims_mp3_output() {
        let frame = silent_frame(9, true);
        let mut d = decoder(PcmEncoding::S16);
        let mut g = GaplessState::default();
        g.set(GaplessDecode { total_samples: 1000, skip_samples: 200, skipped_samples: 0 }, true);
        let mut out = vec![0u8; 1152 * 2];
        let r = d.decode(&frame, &mut out, true, &mut g);
        assert_eq!(r.output_written, (1152 - 200) * 2);
        assert_eq!(g.current, GaplessDecode { total_samples: 48, skip_samples: 0, skipped_samples: 200 });
    }

    #[test]
    fn output_too_small_keeps_frame() {
        let frame = silent_frame(9, true);
        let mut d = decoder(PcmEncoding::S16);
        let mut g = GaplessState::default();
        let mut out = vec![0u8; 100];
        let r = d.decode(&frame, &mut out, true, &mut g);
        assert_eq!(r.result, RESULT_NOT_ENOUGH_ROOM);
        assert_eq!(r.frames, 0);
        assert_eq!(r.output_written, 0);
        assert_eq!(r.input_consumed, 0);
        let mut out = vec![0u8; 1152 * 2];
        let r = d.decode(&[], &mut out, true, &mut g);
        assert_eq!(r.frames, 1);
        assert_eq!(r.output_written, 1152 * 2);
        // Nothing pending and no input: partial input.
        assert_eq!(d.decode(&[], &mut out, true, &mut g).result, RESULT_PARTIAL_INPUT);
    }

    #[test]
    fn garbage_does_not_panic() {
        let mut d = decoder(PcmEncoding::S16);
        let mut seed = 0x1234_5678u32;
        let junk: Vec<u8> = (0..20000)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed as u8
            })
            .collect();
        let mut out = vec![0u8; 65536];
        for chunk in junk.chunks(997) {
            let _ = d.decode(chunk, &mut out, true, &mut GaplessState::default());
        }
    }
}
