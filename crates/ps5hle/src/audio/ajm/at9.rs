// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! AJM ATRAC9 decoder (codec 1): frame-by-frame decoding of superframes,
//! optional RIFF/WAVE header parsing, gapless trimming.

use super::decoder::*;
use crate::audio::atrac9::{Atrac9Config, Atrac9Decoder};

/// Instance flag: the input may start with a RIFF/WAVE (.at9) header.
pub const FLAG_PARSE_RIFF_HEADER: u64 = 1 << INSTANCE_FLAG_CODEC_OFFSET;

/// Internal result reported for a malformed frame.
const INTERNAL_UNPACK_ERROR: i32 = 0x8200_0000u32 as i32;

const AT9_GUID: [u8; 16] = [0xD2, 0x42, 0xE1, 0x47, 0xBA, 0x36, 0x8D, 0x4D, 0x88, 0xFC, 0x61, 0x65, 0x4F, 0x8C, 0x83, 0x6C];

/// AjmSidebandDecAt9CodecInfo (16 bytes).
pub const CODEC_INFO_BYTES: usize = 16;

pub struct AjmAt9Decoder {
    parse_riff_header: bool,
    channels: u32,
    sample_rate: u32,
    encoding: PcmEncoding,
    decoder: Option<Atrac9Decoder>,
    config_data: Option<[u8; 4]>,
    superframe_bytes_remain: u32,
    num_frames: u32,
    total_decoded_samples: u64,
}

impl AjmAt9Decoder {
    pub fn new(channels: u32, sample_rate: u32, encoding: PcmEncoding, flags: u64) -> Self {
        Self {
            parse_riff_header: flags & FLAG_PARSE_RIFF_HEADER != 0,
            channels,
            sample_rate,
            encoding,
            decoder: None,
            config_data: None,
            superframe_bytes_remain: 0,
            num_frames: 0,
            total_decoded_samples: 0,
        }
    }

    pub fn config(&self) -> Option<&Atrac9Config> {
        self.decoder.as_ref().map(|d| d.config())
    }

    fn min_input(&self) -> usize {
        if self.decoder.is_some() {
            self.superframe_bytes_remain as usize
        } else {
            0
        }
    }

    fn frame_samples(&self) -> u32 {
        self.config().map_or(0, |c| c.frame_samples.max(0) as u32)
    }

    fn frame_bytes_out(&self, samples: u32) -> usize {
        self.config().map_or(0, |c| c.channel_count.max(0) as usize) * samples as usize * self.encoding.bytes_per_sample()
    }

    fn initialize_config(&mut self, config: &[u8], result: &mut DecodeResult) -> bool {
        let config: [u8; 4] = config[..4].try_into().unwrap();
        self.config_data = Some(config);
        match Atrac9Decoder::new(&config) {
            Ok(d) => {
                let c = d.config();
                self.superframe_bytes_remain = c.superframe_bytes.max(0) as u32;
                log::debug!(
                    "[AUDIO] AJM ATRAC9 initialized: {} Hz, {} ch, frame_samples={}, superframe={} bytes/{} frames",
                    c.sample_rate,
                    c.channel_count,
                    c.frame_samples,
                    c.superframe_bytes,
                    c.frames_per_superframe
                );
                self.channels = c.channel_count as u32;
                self.sample_rate = c.sample_rate as u32;
                self.decoder = Some(d);
                self.num_frames = 0;
                self.total_decoded_samples = 0;
                true
            }
            Err(_) => {
                self.decoder = None;
                result.result = RESULT_CODEC_ERROR | RESULT_INVALID_DATA;
                result.internal_result = 0x8100_0000u32 as i32;
                false
            }
        }
    }

    /// Bytes of the current superframe left after a frame that used
    /// `used` bytes; at the end of the superframe the padding is skipped.
    fn advance_superframe(&mut self, used: u32, input_len: usize, offset: &mut usize, result: &mut DecodeResult) {
        self.superframe_bytes_remain -= used;
        self.num_frames += 1;
        let Some(c) = self.config().cloned() else { return };
        if self.num_frames < c.frames_per_superframe as u32 {
            return;
        }
        if self.superframe_bytes_remain != 0 {
            let remaining = input_len.saturating_sub(*offset);
            if remaining < self.superframe_bytes_remain as usize {
                result.result = RESULT_PARTIAL_INPUT;
                return;
            }
            *offset += self.superframe_bytes_remain as usize;
            result.input_consumed = *offset;
        }
        self.superframe_bytes_remain = c.superframe_bytes as u32;
        self.num_frames = 0;
        if let Some(d) = self.decoder.as_mut() {
            d.restart_superframe();
        }
    }

    fn decode_frame(&mut self, input: &[u8], remaining: usize, result: &mut DecodeResult) -> Option<u32> {
        let d = self.decoder.as_mut()?;
        match d.decode_next_frame(input) {
            Err(_) => {
                result.result = RESULT_CODEC_ERROR | RESULT_FATAL;
                result.internal_result = INTERNAL_UNPACK_ERROR;
                None
            }
            Ok(used) if used == 0 || used > remaining || used as u32 > self.superframe_bytes_remain => {
                result.result = RESULT_CODEC_ERROR | RESULT_INVALID_DATA;
                None
            }
            Ok(used) => Some(used as u32),
        }
    }

    /// Decodes and discards the rest of the superframe once the gapless
    /// sample limit is reached, so the stream stays superframe aligned.
    fn drain_superframe(&mut self, input: &[u8], offset: &mut usize, result: &mut DecodeResult) {
        while self.num_frames != 0 {
            let remaining = input.len().saturating_sub(*offset);
            if remaining < self.min_input() {
                result.result = RESULT_PARTIAL_INPUT;
                return;
            }
            let Some(used) = self.decode_frame(&input[*offset..], remaining, result) else { return };
            *offset += used as usize;
            result.input_consumed = *offset;
            result.frames += 1;
            self.advance_superframe(used, input.len(), offset, result);
            if result.result != 0 {
                return;
            }
        }
    }

    fn parse_riff_header(&mut self, input: &[u8], gapless: &mut GaplessState, result: &mut DecodeResult) -> Option<usize> {
        if input.len() < 12 || &input[0..4] != b"RIFF" || &input[8..12] != b"WAVE" {
            result.result = RESULT_INVALID_DATA;
            return None;
        }
        let mut offset = 12usize;
        while offset + 8 <= input.len() {
            let id = &input[offset..offset + 4];
            let size = u32::from_le_bytes(input[offset + 4..offset + 8].try_into().unwrap()) as usize;
            let payload = offset + 8;
            // The data chunk describes the whole stream; the job may hold only its start.
            if id == b"data" {
                result.input_consumed = payload;
                result.format = self.format();
                result.total_decoded_samples = self.total_decoded_samples;
                return Some(payload);
            }
            if payload > input.len() || size > input.len() - payload {
                result.result = RESULT_PARTIAL_INPUT;
                return None;
            }
            if id == b"fmt " {
                let fmt = &input[payload..payload + size];
                if size < 48 || u16::from_le_bytes([fmt[0], fmt[1]]) != 0xFFFE || fmt[24..40] != AT9_GUID {
                    result.result = RESULT_INVALID_DATA;
                    return None;
                }
                if !self.initialize_config(&fmt[44..48], result) {
                    return None;
                }
            } else if id == b"fact" && size >= 12 {
                let fact = &input[payload..payload + size];
                let total = u32::from_le_bytes(fact[0..4].try_into().unwrap());
                let skip = u32::from_le_bytes(fact[4..8].try_into().unwrap()).min(u16::MAX as u32) as u16;
                gapless.set(GaplessDecode { total_samples: total, skip_samples: skip, skipped_samples: 0 }, true);
                log::debug!("[AUDIO] AJM ATRAC9 gapless: total={total}, skip={skip}");
            }
            offset = payload + size + (size & 1);
        }
        result.result = RESULT_PARTIAL_INPUT;
        None
    }
}

impl AjmDecoder for AjmAt9Decoder {
    fn initialize(&mut self, params: &[u8]) -> DecodeResult {
        let mut result = self.make_result();
        if params.len() < 4 {
            result.result = RESULT_INVALID_PARAMETER;
            return result;
        }
        if !self.initialize_config(params, &mut result) {
            return result;
        }
        self.make_result()
    }

    fn reset(&mut self) {
        self.decoder = None;
        self.num_frames = 0;
        self.total_decoded_samples = 0;
        self.superframe_bytes_remain = 0;
        if let Some(cfg) = self.config_data {
            let mut r = DecodeResult::default();
            self.initialize_config(&cfg, &mut r);
        }
    }

    fn decode(&mut self, input: &[u8], output: &mut [u8], multiple_frames: bool, gapless: &mut GaplessState) -> DecodeResult {
        let mut result = self.make_result();
        if input.is_empty() {
            result.result = RESULT_PARTIAL_INPUT;
            return result;
        }

        let mut offset = 0usize;
        if self.parse_riff_header && input.len() >= 12 && &input[0..4] == b"RIFF" {
            match self.parse_riff_header(input, gapless, &mut result) {
                None => return result,
                Some(o) => offset = o,
            }
            if offset >= input.len() {
                result.input_consumed = offset;
                result.result = RESULT_PARTIAL_INPUT;
                return result;
            }
        }

        if self.decoder.is_none() {
            result.result = RESULT_NOT_INITIALIZED;
            return result;
        }

        // The run result names the side that ran out (measured on a PS4
        // through the console's builders): PARTIAL_INPUT when the input ran
        // out, on a superframe boundary or not; NOT_ENOUGH_ROOM when the
        // room ran out, after any number of frames; both when the input
        // ended just as the output filled. The end of a gapless segment
        // reports neither.
        let frame_samples = self.frame_samples();
        loop {
            if gapless.is_end() {
                break;
            }
            let (_, out_samples) = gapless.preview(frame_samples);
            let out_bytes = self.frame_bytes_out(out_samples);
            let can_be_empty = gapless.current.skip_samples != 0 || gapless.is_end();
            let room = !(out_bytes == 0 && !can_be_empty) && output.len() - result.output_written >= out_bytes;
            if offset >= input.len() || input.len() - offset < self.min_input() {
                result.result |= RESULT_PARTIAL_INPUT;
                if !room {
                    result.result |= RESULT_NOT_ENOUGH_ROOM;
                }
                break;
            }
            let remaining = input.len() - offset;
            if !room {
                result.result |= RESULT_NOT_ENOUGH_ROOM;
                break;
            }

            let Some(used) = self.decode_frame(&input[offset..], remaining, &mut result) else { break };

            let (skip, write) = gapless.plan(frame_samples);
            let bytes = self.frame_bytes_out(write);
            let dst = &mut output[result.output_written..result.output_written + bytes];
            self.decoder.as_ref().unwrap().write_frame_pcm(dst, skip as usize, write as usize, self.encoding);
            result.output_written += bytes;
            gapless.commit(frame_samples, write);

            offset += used as usize;
            result.input_consumed = offset;
            result.frames += 1;
            self.total_decoded_samples += write as u64;

            self.advance_superframe(used, input.len(), &mut offset, &mut result);
            if gapless.is_end() {
                self.drain_superframe(input, &mut offset, &mut result);
            }
            if result.result != 0 || !multiple_frames {
                break;
            }
        }

        result.total_decoded_samples = self.total_decoded_samples;
        result.format = self.format();
        result
    }

    fn codec_info(&self) -> Vec<u8> {
        let c = self.config();
        let mut b = Vec::with_capacity(CODEC_INFO_BYTES);
        for v in [
            c.map_or(0, |c| c.superframe_bytes as u32),
            c.map_or(0, |c| c.frames_per_superframe as u32),
            self.min_input() as u32,
            c.map_or(0, |c| c.frame_samples as u32),
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    fn codec_info_size(&self) -> usize {
        CODEC_INFO_BYTES
    }

    fn format(&self) -> SidebandFormat {
        match self.config() {
            Some(c) => {
                let mut f = SidebandFormat::new(c.channel_count as u32, c.sample_rate as u32, self.encoding);
                if c.frames_per_superframe > 0 && c.frame_samples > 0 {
                    f.bitrate = (c.sample_rate as u64 * c.superframe_bytes as u64 * 8
                        / (c.frames_per_superframe as u64 * c.frame_samples as u64)) as u32;
                }
                f
            }
            None => SidebandFormat::new(self.channels, self.sample_rate, self.encoding),
        }
    }

    fn total_decoded_samples(&self) -> u64 {
        self.total_decoded_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::atrac9::tests::{build_mono_superframe, config_word, mono_config, TestRng};
    use crate::audio::atrac9::Atrac9Decoder;

    fn mono(encoding: PcmEncoding) -> (AjmAt9Decoder, Vec<u8>) {
        let mut d = AjmAt9Decoder::new(2, 48000, encoding, 0);
        let mut cfg = mono_config().to_vec();
        cfg.extend_from_slice(&[0; 4]);
        assert_eq!(d.initialize(&cfg).result, 0);
        let mut coefficients = [0i32; 24];
        coefficients[3] = 20000;
        (d, build_mono_superframe(&coefficients))
    }

    fn decode(d: &mut AjmAt9Decoder, input: &[u8], out: &mut [u8], multiple: bool, g: &mut GaplessState) -> DecodeResult {
        d.decode(input, out, multiple, g)
    }

    #[test]
    fn uninitialized_and_bad_params() {
        let mut d = AjmAt9Decoder::new(2, 48000, PcmEncoding::S16, 0);
        let mut g = GaplessState::default();
        let mut out = [0u8; 16];
        assert_eq!(d.decode(&[0u8; 16], &mut out, true, &mut g).result, RESULT_NOT_INITIALIZED);
        assert_eq!(d.decode(&[], &mut out, true, &mut g).result, RESULT_PARTIAL_INPUT);
        assert_eq!(d.initialize(&[0xFE, 0x74]).result, RESULT_INVALID_PARAMETER);
        assert_eq!(d.initialize(&[0x00, 0x74, 0x0F, 0xF0]).result, RESULT_CODEC_ERROR | RESULT_INVALID_DATA);
        // Uninitialized format: the instance defaults.
        assert_eq!(d.format(), SidebandFormat::new(2, 48000, PcmEncoding::S16));
    }

    #[test]
    fn single_frame_per_job_without_multiple_frames() {
        let (mut d, data) = mono(PcmEncoding::S16);
        let mut g = GaplessState::default();
        let mut out = vec![0u8; 4096];
        // Frame 0 needs the whole superframe in the input.
        let r = decode(&mut d, &data[..100], &mut out, false, &mut g);
        assert_eq!((r.result, r.input_consumed, r.frames), (RESULT_PARTIAL_INPUT, 0, 0));
        let mut consumed = 0;
        for i in 0..4 {
            let r = decode(&mut d, &data[consumed..], &mut out, false, &mut g);
            assert_eq!(r.result, 0, "frame {i}");
            assert_eq!(r.frames, 1);
            assert_eq!(r.output_written, 256 * 2);
            consumed += r.input_consumed;
        }
        // The last frame also consumed the superframe padding.
        assert_eq!(consumed, 512);
        assert_eq!(d.total_decoded_samples(), 1024);
        let info = d.codec_info();
        assert_eq!(u32::from_le_bytes(info[0..4].try_into().unwrap()), 512);
        assert_eq!(u32::from_le_bytes(info[4..8].try_into().unwrap()), 4);
        assert_eq!(u32::from_le_bytes(info[8..12].try_into().unwrap()), 512);
        assert_eq!(u32::from_le_bytes(info[12..16].try_into().unwrap()), 256);
    }

    #[test]
    fn multiple_frames_and_encodings() {
        let (mut d, data) = mono(PcmEncoding::S16);
        let mut g = GaplessState::default();
        let mut input = data.clone();
        input.extend_from_slice(&data);
        let mut out16 = vec![0u8; 8192];
        let r = decode(&mut d, &input, &mut out16, true, &mut g);
        assert_eq!((r.result, r.input_consumed, r.output_written, r.frames), (RESULT_PARTIAL_INPUT, 1024, 4096, 8), "input ran out with room left");
        assert_eq!(r.total_decoded_samples, 2048);
        assert_eq!(r.format.channels, 1);
        assert_eq!(r.format.bitrate, 48000 * 512 * 8 / 1024);
        let s16: Vec<i16> = out16.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        assert!(s16.iter().any(|&v| v != 0));

        let (mut d, _) = mono(PcmEncoding::S32);
        let mut out32 = vec![0u8; 8192];
        decode(&mut d, &data, &mut out32, true, &mut g);
        for (i, c) in out32[..4096].chunks_exact(4).enumerate() {
            assert_eq!(i32::from_le_bytes(c.try_into().unwrap()), s16[i] as i32);
        }
        let (mut d, _) = mono(PcmEncoding::Float);
        let mut outf = vec![0u8; 8192];
        decode(&mut d, &data, &mut outf, true, &mut g);
        for (i, c) in outf[..4096].chunks_exact(4).enumerate() {
            let f = f32::from_le_bytes(c.try_into().unwrap());
            assert!((f * 32767.0 - s16[i] as f32).abs() <= 0.5 + 1e-3, "{i}: {f}");
        }
    }

    #[test]
    fn not_enough_room_stops_before_the_frame() {
        let (mut d, data) = mono(PcmEncoding::S16);
        let mut g = GaplessState::default();
        let mut out = vec![0u8; 600];
        let r = decode(&mut d, &data, &mut out, true, &mut g);
        assert_eq!(r.result, RESULT_NOT_ENOUGH_ROOM);
        assert_eq!((r.frames, r.output_written), (1, 512));
        let consumed = r.input_consumed;
        let mut out = vec![0u8; 4096];
        let r = decode(&mut d, &data[consumed..], &mut out, true, &mut g);
        assert_eq!((r.result, r.frames, r.output_written), (RESULT_PARTIAL_INPUT, 3, 3 * 512));
        assert_eq!(consumed + r.input_consumed, 512);
    }

    #[test]
    fn gapless_skip_limit_and_drain() {
        let (mut reference, data) = mono(PcmEncoding::S16);
        let mut full = vec![0u8; 2048];
        reference.decode(&data, &mut full, true, &mut GaplessState::default());

        let (mut d, _) = mono(PcmEncoding::S16);
        let mut g = GaplessState::default();
        g.set(GaplessDecode { total_samples: 1000, skip_samples: 100, skipped_samples: 0 }, true);
        let mut out = vec![0u8; 2048];
        let r = decode(&mut d, &data, &mut out, true, &mut g);
        assert_eq!(r.result, RESULT_PARTIAL_INPUT, "the input ran out before the gapless total");
        assert_eq!(r.output_written, 924 * 2);
        assert_eq!(&out[..924 * 2], &full[200..]);
        assert_eq!(g.current, GaplessDecode { total_samples: 76, skip_samples: 0, skipped_samples: 100 });

        // The limit ends the stream after 76 samples; the rest of the
        // superframe is drained so the input stays aligned.
        let r = decode(&mut d, &data, &mut out, true, &mut g);
        assert_eq!(r.output_written, 76 * 2);
        assert_eq!(r.input_consumed, 512);
        assert_eq!(r.frames, 4);
        assert!(g.is_end());
        // Drained frames are not counted as skipped.
        assert_eq!(g.current.skipped_samples, 100 + 256 - 76);
        // At the end nothing more is decoded or consumed.
        let r = decode(&mut d, &data, &mut out, true, &mut g);
        assert_eq!((r.input_consumed, r.output_written, r.frames), (0, 0, 0));
    }

    fn riff(data: &[u8], config: [u8; 4], fact: Option<(u32, u32)>) -> Vec<u8> {
        let mut f = Vec::new();
        f.extend_from_slice(b"RIFF\0\0\0\0WAVE");
        f.extend_from_slice(b"fmt ");
        f.extend_from_slice(&52u32.to_le_bytes());
        let mut fmt = [0u8; 52];
        fmt[0..2].copy_from_slice(&0xFFFEu16.to_le_bytes());
        fmt[24..40].copy_from_slice(&AT9_GUID);
        fmt[44..48].copy_from_slice(&config);
        f.extend_from_slice(&fmt);
        if let Some((total, skip)) = fact {
            f.extend_from_slice(b"fact");
            f.extend_from_slice(&12u32.to_le_bytes());
            f.extend_from_slice(&total.to_le_bytes());
            f.extend_from_slice(&skip.to_le_bytes());
            f.extend_from_slice(&0u32.to_le_bytes());
        }
        f.extend_from_slice(b"data");
        f.extend_from_slice(&(data.len() as u32).to_le_bytes());
        f.extend_from_slice(data);
        f
    }

    #[test]
    fn riff_header_initializes_and_sets_gapless() {
        let (mut reference, data) = mono(PcmEncoding::S16);
        let mut expected = vec![0u8; 2048];
        reference.decode(&data, &mut expected, true, &mut GaplessState::default());

        let file = riff(&data, mono_config(), Some((2000, 0)));
        let mut d = AjmAt9Decoder::new(0, 48000, PcmEncoding::S16, FLAG_PARSE_RIFF_HEADER);
        let mut g = GaplessState::default();
        let mut out = vec![0u8; 2048];
        let r = d.decode(&file, &mut out, true, &mut g);
        assert_eq!(r.result, RESULT_PARTIAL_INPUT | RESULT_NOT_ENOUGH_ROOM, "input ended as the output filled");
        assert_eq!(r.input_consumed, file.len());
        assert_eq!(out, expected);
        assert_eq!(g.init.total_samples, 2000);
        assert_eq!(g.current.total_samples, 2000 - 1024);

        // Header only: the data offset is consumed, partial input.
        let mut d = AjmAt9Decoder::new(0, 48000, PcmEncoding::S16, FLAG_PARSE_RIFF_HEADER);
        let hdr = &file[..file.len() - data.len()];
        let r = d.decode(hdr, &mut out, true, &mut g);
        assert_eq!((r.result, r.input_consumed), (RESULT_PARTIAL_INPUT, hdr.len()));
        // Truncated header.
        let r = d.decode(&file[..30], &mut out, true, &mut g);
        assert_eq!(r.result, RESULT_PARTIAL_INPUT);
        // Without the flag the header is not parsed.
        let mut d = AjmAt9Decoder::new(0, 48000, PcmEncoding::S16, 0);
        assert_eq!(d.decode(&file, &mut out, true, &mut g).result, RESULT_NOT_INITIALIZED);
    }

    #[test]
    fn reset_restarts_the_stream() {
        let (mut d, data) = mono(PcmEncoding::S16);
        let mut g = GaplessState::default();
        let mut out = vec![0u8; 4096];
        let first = d.decode(&data, &mut out, false, &mut g);
        assert_eq!(first.frames, 1);
        d.reset();
        assert_eq!(d.total_decoded_samples(), 0);
        // After a reset frame 0 again needs a whole superframe.
        let r = d.decode(&data[first.input_consumed..], &mut out, false, &mut g);
        assert_eq!(r.result, RESULT_PARTIAL_INPUT);
    }

    #[test]
    fn invalid_data_is_a_codec_error() {
        let (mut d, _) = mono(PcmEncoding::S16);
        let mut out = vec![0u8; 2048];
        let r = d.decode(&[0u8; 512], &mut out, true, &mut GaplessState::default());
        assert_eq!(r.result & RESULT_CODEC_ERROR, RESULT_CODEC_ERROR);
        assert_eq!(r.output_written, 0);
    }

    #[test]
    fn vibration_layout_initializes() {
        let mut d = AjmAt9Decoder::new(2, 48000, PcmEncoding::S16, 0);
        assert_eq!(d.initialize(&config_word(7, 6, 128, 2)).result, 0);
        assert_eq!(d.format().channels, 1);
    }

    #[test]
    fn random_input_never_panics() {
        let mut rng = TestRng(777);
        let mut d = AjmAt9Decoder::new(2, 48000, PcmEncoding::S16, FLAG_PARSE_RIFF_HEADER);
        assert_eq!(d.initialize(&[0xFE, 0x74, 0x0F, 0xF0]).result, 0);
        let mut out = vec![0u8; 1 << 16];
        let mut g = GaplessState::default();
        for _ in 0..500 {
            let len = (rng.next() % 3000) as usize;
            let input: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let out_len = (rng.next() % out.len() as u64) as usize;
            let r = d.decode(&input, &mut out[..out_len], rng.next() & 1 == 0, &mut g);
            assert!(r.input_consumed <= input.len());
            assert!(r.output_written <= out_len);
            if rng.next() % 7 == 0 {
                d.reset();
            }
        }
    }

    fn decode_superframe(config: [u8; 4], superframe: &[u8], channels: usize) -> Vec<i16> {
        let mut d = AjmAt9Decoder::new(0, 48000, PcmEncoding::S16, 0);
        assert_eq!(d.initialize(&config).result, 0);
        let mut out = vec![0u8; 1024 * channels * 2];
        let r = d.decode(superframe, &mut out, true, &mut GaplessState::default());
        assert_eq!((r.result, r.input_consumed), (RESULT_PARTIAL_INPUT | RESULT_NOT_ENOUGH_ROOM, superframe.len()), "input ended as the output filled");
        assert_eq!((r.output_written, r.total_decoded_samples, r.frames), (out.len(), 1024, 4));
        out.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
    }

    /// Bytes each frame of a mono superframe uses.
    fn mono_block_sizes(superframe: &[u8]) -> Vec<usize> {
        let mut d = Atrac9Decoder::new(&mono_config()).unwrap();
        let mut offset = 0;
        (0..4)
            .map(|_| {
                let used = d.decode_next_frame(&superframe[offset..]).unwrap();
                offset += used;
                used
            })
            .collect()
    }

    /// Config header 0x30 (ambisonic, 4..16 channels): every channel is a mono
    /// stream and the blocks are interleaved frame by frame.
    #[test]
    fn multichannel_config_decodes_each_channel_as_mono() {
        let c = Atrac9Config::new(&[0x30, 0x73, 0xC1, 0x7E]).unwrap();
        assert_eq!((c.channel_count, c.sample_rate, c.frame_samples), (16, 48000, 256));
        assert_eq!((c.superframe_samples, c.superframe_bytes), (1024, 6144));
        // Validation bit set.
        assert!(Atrac9Config::new(&[0x30, 0x73, 0xE1, 0x7E]).is_err());

        // Two mono streams with different content, 128-byte frames.
        let mut coefficients = [[0i32; 24]; 2];
        coefficients[0][3] = 20000;
        coefficients[1][7] = -12000;
        let mono_streams: Vec<Vec<u8>> = coefficients.iter().map(build_mono_superframe).collect();
        let reference: Vec<Vec<i16>> = mono_streams.iter().map(|s| decode_superframe(mono_config(), s, 1)).collect();

        // Same rate and frame size, a 6-bit channel count of 2.
        let budget = 512;
        let config = [0x30, 0x70, 0x41, 0xFE];
        let multi = Atrac9Config::new(&config).unwrap();
        assert_eq!((multi.channel_count, multi.superframe_bytes), (2, 2 * budget as i32));
        let sizes: Vec<Vec<usize>> = mono_streams.iter().map(|s| mono_block_sizes(s)).collect();
        let mut superframe = Vec::new();
        let mut offsets = [0usize; 2];
        for frame in 0..4 {
            for channel in 0..2 {
                // The last frame's block takes the rest of the channel budget.
                let bytes = if frame == 3 { budget - offsets[channel] } else { sizes[channel][frame] };
                superframe.extend_from_slice(&mono_streams[channel][offsets[channel]..offsets[channel] + bytes]);
                offsets[channel] += bytes;
            }
        }
        assert_eq!(superframe.len(), 2 * budget);
        let decoded = decode_superframe(config, &superframe, 2);
        let mut audible = [false; 2];
        for sample in 0..1024 {
            assert_eq!(decoded[sample * 2], reference[0][sample], "channel 0 sample {sample}");
            assert_eq!(decoded[sample * 2 + 1], reference[1][sample], "channel 1 sample {sample}");
            audible[0] |= reference[0][sample] != 0;
            audible[1] |= reference[1][sample] != 0;
        }
        assert_eq!(audible, [true, true]);
        assert_ne!(reference[0], reference[1]);
    }
}
