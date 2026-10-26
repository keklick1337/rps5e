// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! AJM Opus decoder (codec 24), decoded on the host. The input is a run of
//! packets, each prefixed with its 16-bit little-endian size.
//!
//! Channel mapping family 0 is mono/stereo; family 1 (Vorbis order, up to
//! 8 channels) uses the RFC 7845 stream layout; family 255 takes its
//! stream layout from the first packet (self-delimited streams, one coupled
//! stream per stereo pair).

use super::decoder::*;
use super::ffmpeg::{HostAudioDecoder, HostCodec};

/// AjmDecOpusInitializeParameters (12 bytes; the mapping family may be left out).
pub const INITIALIZE_PARAMETERS_BYTES: usize = 12;
const MIN_INITIALIZE_PARAMETERS_BYTES: usize = 8;
const MAX_CHANNELS: u32 = 8;
/// AjmSidebandDecOpusCodecInfo (4 bytes).
pub const CODEC_INFO_BYTES: usize = 4;

const INTERNAL_INVALID_PACKET: i32 = 0x1043;
const INTERNAL_BAD_FRAME_LAYOUT: i32 = 0xFFFF_8800u32 as i32;

pub struct AjmOpusDecoder {
    host: Option<HostAudioDecoder>,
    max_channels: u32,
    channels: u32,
    mapping_family: u32,
    /// Family 255: the stream layout comes with the first packet.
    layout_pending: bool,
    sample_rate: u32,
    encoding: PcmEncoding,
    frames_per_packet: u32,
    total_decoded_samples: u64,
}

/// (frames, samples per frame) from the TOC byte (and the frame-count byte
/// of code 3 packets).
pub fn packet_layout(packet: &[u8]) -> (u32, u32) {
    let toc = packet[0];
    let config = (toc >> 3) as u32;
    let frames = match toc & 3 {
        0 => 1,
        3 => (packet.get(1).copied().unwrap_or(0) & 0x3F) as u32,
        _ => 2,
    };
    let frame_samples = if config >= 16 {
        120 << (config & 3)
    } else if config >= 12 {
        480 << (config & 1)
    } else if config & 3 == 3 {
        2880
    } else {
        480 << (config & 3)
    };
    (frames, frame_samples)
}

impl AjmOpusDecoder {
    pub fn new(max_channels: u32, encoding: PcmEncoding) -> Self {
        Self {
            host: None,
            max_channels,
            channels: 0,
            mapping_family: 0,
            layout_pending: false,
            sample_rate: 48000,
            encoding,
            frames_per_packet: 0,
            total_decoded_samples: 0,
        }
    }
}

/// OpusHead of a multistream decoder (RFC 7845 5.1).
fn opus_head(channels: u32, family: u8, streams: u8, coupled: u8, mapping: &[u8]) -> Vec<u8> {
    let mut h = b"OpusHead".to_vec();
    h.extend_from_slice(&[1, channels as u8, 0, 0]);
    h.extend_from_slice(&48000u32.to_le_bytes());
    h.extend_from_slice(&[0, 0, family, streams, coupled]);
    h.extend_from_slice(&mapping[..channels as usize]);
    h
}

/// Family 1 layouts of 1..8 channels: (streams, coupled streams, mapping).
const VORBIS_STREAMS: [u8; 8] = [1, 1, 2, 2, 3, 4, 4, 5];
const VORBIS_COUPLED: [u8; 8] = [0, 1, 1, 2, 2, 2, 3, 3];
const VORBIS_MAPPING: [[u8; 8]; 8] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 2, 1, 0, 0, 0, 0, 0],
    [0, 1, 2, 3, 0, 0, 0, 0],
    [0, 4, 1, 2, 3, 0, 0, 0],
    [0, 4, 1, 2, 3, 5, 0, 0],
    [0, 4, 1, 2, 3, 5, 6, 0],
    [0, 6, 1, 2, 3, 4, 5, 7],
];

/// A frame length of a self-delimited packet (RFC 6716 3.1); None past the end.
fn frame_length(data: &[u8], at: &mut usize) -> Option<usize> {
    let first = *data.get(*at)? as usize;
    *at += 1;
    if first < 252 {
        return Some(first);
    }
    let second = *data.get(*at)? as usize;
    *at += 1;
    Some(first + 4 * second)
}

/// Bytes of the self-delimited Opus packet at the start of `data`
/// (RFC 6716 appendix B); None when it is malformed or cut.
fn self_delimited_bytes(data: &[u8]) -> Option<usize> {
    let mut at = 1;
    match data.first()? & 3 {
        0 => {
            let frame = frame_length(data, &mut at)?;
            Some(at + frame)
        }
        1 => {
            let frame = frame_length(data, &mut at)?;
            Some(at + 2 * frame)
        }
        2 => {
            let first = frame_length(data, &mut at)?;
            let second = frame_length(data, &mut at)?;
            Some(at + first + second)
        }
        _ => {
            let count = *data.get(at)?;
            at += 1;
            let frames = (count & 0x3F) as usize;
            if frames == 0 {
                return None;
            }
            let mut padding = 0;
            if count & 0x40 != 0 {
                loop {
                    let value = *data.get(at)? as usize;
                    at += 1;
                    padding += if value == 255 { 254 } else { value };
                    if value != 255 {
                        break;
                    }
                }
            }
            let mut total = 0;
            if count & 0x80 != 0 {
                for _ in 0..frames {
                    total += frame_length(data, &mut at)?;
                }
            } else {
                total = frame_length(data, &mut at)? * frames;
            }
            Some(at + total + padding)
        }
    }
}

/// (streams, coupled streams) of a family 255 packet of `channels`
/// channels: streams are counted until they make the channels; a stream
/// whose TOC has the stereo bit is coupled.
fn packet_streams(packet: &[u8], channels: u32) -> Option<(u32, u32)> {
    let (mut streams, mut coupled, mut offset) = (0u32, 0u32, 0usize);
    while offset < packet.len() {
        if packet[offset] & 4 != 0 {
            coupled += 1;
        }
        streams += 1;
        if coupled * 2 + (streams - coupled) >= channels {
            break;
        }
        match self_delimited_bytes(&packet[offset..]) {
            Some(bytes) if bytes < packet.len() - offset => offset += bytes,
            _ => break,
        }
    }
    (coupled * 2 + (streams - coupled) == channels).then_some((streams, coupled))
}

impl AjmOpusDecoder {
    fn open_host(&mut self, streams: u8, coupled: u8) -> bool {
        let c = self.channels;
        self.host = match self.mapping_family {
            1 => {
                let i = c as usize - 1;
                let head = opus_head(c, 1, VORBIS_STREAMS[i], VORBIS_COUPLED[i], &VORBIS_MAPPING[i]);
                HostAudioDecoder::open_with_header(HostCodec::Opus, c, 48000, &head)
            }
            255 if c > 1 => {
                let mapping: Vec<u8> = (0..c as u8).collect();
                HostAudioDecoder::open_with_header(HostCodec::Opus, c, 48000, &opus_head(c, 255, streams, coupled, &mapping))
            }
            _ => HostAudioDecoder::open(HostCodec::Opus, c, 48000),
        };
        self.host.is_some()
    }
}

impl AjmDecoder for AjmOpusDecoder {
    fn initialize(&mut self, params: &[u8]) -> DecodeResult {
        self.host = None;
        self.layout_pending = false;
        self.reset();
        let mut result = self.make_result();
        if params.len() < MIN_INITIALIZE_PARAMETERS_BYTES {
            result.result = RESULT_INVALID_PARAMETER | RESULT_FATAL;
            return result;
        }
        let word = |i: usize| u32::from_le_bytes(params[i * 4..i * 4 + 4].try_into().unwrap());
        let (channels, sample_rate) = (word(0), word(1));
        let mapping_family = if params.len() >= INITIALIZE_PARAMETERS_BYTES { word(2) } else { 0 };
        if channels == 0 || channels > MAX_CHANNELS || sample_rate == 0 {
            result.result = RESULT_INVALID_PARAMETER | RESULT_FATAL;
            return result;
        }
        if sample_rate != 48000 || (mapping_family > 1 && mapping_family != 255) {
            log::error!("[AUDIO] AJM Opus at {sample_rate} Hz, mapping family {mapping_family} is not implemented");
            result.result = RESULT_INVALID_PARAMETER | RESULT_FATAL;
            return result;
        }
        if mapping_family == 0 && channels > 2 {
            result.result = RESULT_INVALID_PARAMETER | RESULT_FATAL;
            return result;
        }
        if channels > self.max_channels {
            result.result = RESULT_TOO_MANY_CHANNELS | RESULT_FATAL;
            return result;
        }
        self.channels = channels;
        self.sample_rate = sample_rate;
        self.mapping_family = mapping_family;
        self.layout_pending = mapping_family == 255 && channels > 1;
        if !self.layout_pending && !self.open_host(0, 0) {
            result.result = RESULT_CODEC_ERROR | RESULT_FATAL;
            return result;
        }
        self.make_result()
    }

    fn reset(&mut self) {
        if let Some(h) = self.host.as_mut() {
            h.flush();
        }
        self.total_decoded_samples = 0;
        self.frames_per_packet = 0;
    }

    fn decode(&mut self, input: &[u8], output: &mut [u8], multiple_frames: bool, gapless: &mut GaplessState) -> DecodeResult {
        let mut result = self.make_result();
        if self.host.is_none() && !self.layout_pending {
            result.result = RESULT_NOT_INITIALIZED;
            return result;
        }
        if input.is_empty() {
            result.result = RESULT_PARTIAL_INPUT;
            return result;
        }

        while !gapless.is_end() {
            let remaining = input.len() - result.input_consumed;
            if remaining < 2 {
                result.result = RESULT_PARTIAL_INPUT;
                break;
            }
            let at = result.input_consumed;
            let packet_size = u16::from_le_bytes([input[at], input[at + 1]]) as usize;
            if packet_size > remaining - 2 {
                result.result = RESULT_PARTIAL_INPUT;
                break;
            }
            let packet = &input[at + 2..at + 2 + packet_size];
            if packet_size == 0 || (packet[0] & 3 == 3 && packet_size < 2) {
                result.result = RESULT_CODEC_ERROR | RESULT_INVALID_DATA;
                result.internal_result = INTERNAL_INVALID_PACKET;
                break;
            }
            let (frames, frame_samples) = packet_layout(packet);
            let samples = frame_samples * frames;
            if frames == 0 || frames > 6 || frame_samples > 960 {
                result.result = RESULT_CODEC_ERROR | RESULT_INVALID_DATA;
                result.internal_result = INTERNAL_BAD_FRAME_LAYOUT;
                break;
            }
            let (skip, write) = gapless.preview(samples);
            let sample_bytes = self.channels as usize * self.encoding.bytes_per_sample();
            let output_bytes = write as usize * sample_bytes;
            if output_bytes > output.len() - result.output_written {
                // A run that has decoded frames stops before the packet
                // whose frame no longer fits; with nothing decoded yet the
                // output is too small.
                if result.frames == 0 {
                    result.result = RESULT_NOT_ENOUGH_ROOM;
                }
                break;
            }

            if self.layout_pending {
                let Some((streams, coupled)) = packet_streams(packet, self.channels) else {
                    log::error!("[AUDIO] AJM Opus: a mapping family 255 packet's streams do not make {} channels", self.channels);
                    result.result = RESULT_CODEC_ERROR | RESULT_INVALID_DATA;
                    result.internal_result = INTERNAL_INVALID_PACKET;
                    break;
                };
                if !self.open_host(streams as u8, coupled as u8) {
                    result.result = RESULT_CODEC_ERROR | RESULT_FATAL;
                    break;
                }
                self.layout_pending = false;
            }
            let host = self.host.as_mut().unwrap();
            let frame = match host.send(packet, None).and_then(|_| host.receive(self.encoding)) {
                Ok(Some(f)) => f,
                _ => {
                    result.result = RESULT_CODEC_ERROR | RESULT_INVALID_DATA;
                    result.internal_result = INTERNAL_INVALID_PACKET;
                    break;
                }
            };
            if frame.samples != samples || frame.channels != self.channels {
                result.result = RESULT_CODEC_ERROR | RESULT_INVALID_DATA;
                result.internal_result = INTERNAL_INVALID_PACKET;
                break;
            }
            if output_bytes != 0 {
                let src = host.data(&frame, self.encoding);
                let start = skip as usize * sample_bytes;
                output[result.output_written..result.output_written + output_bytes]
                    .copy_from_slice(&src[start..start + output_bytes]);
            }
            gapless.plan(samples);
            gapless.commit(samples, write);
            result.input_consumed += packet_size + 2;
            result.output_written += output_bytes;
            result.frames += frames;
            self.frames_per_packet = frames;
            self.total_decoded_samples += write as u64;
            if !multiple_frames {
                break;
            }
        }
        result.total_decoded_samples = self.total_decoded_samples;
        result
    }

    fn codec_info(&self) -> Vec<u8> {
        self.frames_per_packet.to_le_bytes().to_vec()
    }

    fn codec_info_size(&self) -> usize {
        CODEC_INFO_BYTES
    }

    fn format(&self) -> SidebandFormat {
        SidebandFormat::new(self.channels, self.sample_rate, self.encoding)
    }

    fn total_decoded_samples(&self) -> u64 {
        self.total_decoded_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(channels: u32, rate: u32, family: u32) -> Vec<u8> {
        [channels, rate, family].iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn toc_layouts() {
        // SILK 10 ms, one frame.
        assert_eq!(packet_layout(&[0 << 3]), (1, 480));
        // SILK 60 ms.
        assert_eq!(packet_layout(&[3 << 3]), (1, 2880));
        // Hybrid 20 ms, two frames.
        assert_eq!(packet_layout(&[(13 << 3) | 1]), (2, 960));
        // CELT 2.5 ms, code 3 with 5 frames.
        assert_eq!(packet_layout(&[(16 << 3) | 3, 5]), (5, 120));
        assert_eq!(packet_layout(&[(31 << 3)]), (1, 960));
    }

    #[test]
    fn initialize_validation() {
        let mut d = AjmOpusDecoder::new(2, PcmEncoding::S16);
        let mut g = GaplessState::default();
        assert_eq!(d.decode(&[1, 0, 0], &mut [0u8; 16], true, &mut g).result, RESULT_NOT_INITIALIZED);
        assert_eq!(d.initialize(&[0; 8]).result, RESULT_INVALID_PARAMETER | RESULT_FATAL);
        assert_eq!(d.initialize(&params(3, 48000, 0)).result, RESULT_INVALID_PARAMETER | RESULT_FATAL);
        assert_eq!(d.initialize(&params(2, 44100, 0)).result, RESULT_INVALID_PARAMETER | RESULT_FATAL);
        assert_eq!(d.initialize(&params(2, 48000, 2)).result, RESULT_INVALID_PARAMETER | RESULT_FATAL, "family 2");
        assert_eq!(d.initialize(&params(3, 48000, 0)).result, RESULT_INVALID_PARAMETER | RESULT_FATAL, "family 0 surround");
        assert_eq!(d.initialize(&params(9, 48000, 1)).result, RESULT_INVALID_PARAMETER | RESULT_FATAL, "9 channels");
        let mut stereo = AjmOpusDecoder::new(2, PcmEncoding::S16);
        assert_eq!(stereo.initialize(&params(6, 48000, 1)).result, RESULT_TOO_MANY_CHANNELS | RESULT_FATAL);
        // Eight bytes: mapping family 0.
        let r = d.initialize(&params(2, 48000, 0)[..8]);
        assert_eq!(r.result, if cfg!(ps5_ffmpeg) { 0 } else { RESULT_CODEC_ERROR | RESULT_FATAL });
        let mut mono = AjmOpusDecoder::new(1, PcmEncoding::S16);
        assert_eq!(mono.initialize(&params(2, 48000, 0)).result, RESULT_TOO_MANY_CHANNELS | RESULT_FATAL);
        let r = d.initialize(&params(2, 48000, 0));
        if cfg!(ps5_ffmpeg) {
            assert_eq!(r.result, 0);
            assert_eq!(r.format.channels, 2);
        } else {
            assert_eq!(r.result, RESULT_CODEC_ERROR | RESULT_FATAL);
        }
    }

    #[test]
    fn packet_framing_errors() {
        if !cfg!(ps5_ffmpeg) {
            return;
        }
        let mut d = AjmOpusDecoder::new(2, PcmEncoding::S16);
        assert_eq!(d.initialize(&params(2, 48000, 0)).result, 0);
        let mut g = GaplessState::default();
        let mut out = vec![0u8; 1 << 16];
        // Size prefix larger than the input.
        let r = d.decode(&[10, 0, 1, 2], &mut out, true, &mut g);
        assert_eq!((r.result, r.input_consumed), (RESULT_PARTIAL_INPUT, 0));
        // Zero-size packet.
        let r = d.decode(&[0, 0, 0], &mut out, true, &mut g);
        assert_eq!((r.result, r.internal_result), (RESULT_CODEC_ERROR | RESULT_INVALID_DATA, INTERNAL_INVALID_PACKET));
        // Code 3 with zero frames.
        let r = d.decode(&[2, 0, 3, 0], &mut out, true, &mut g);
        assert_eq!(r.internal_result, INTERNAL_BAD_FRAME_LAYOUT);
        // Not enough room for a 20 ms stereo frame.
        let r = d.decode(&[1, 0, 31 << 3], &mut out[..100], true, &mut g);
        assert_eq!(r.result, RESULT_NOT_ENOUGH_ROOM);
    }

    #[test]
    fn decodes_celt_silence_packets() {
        if !cfg!(ps5_ffmpeg) {
            return;
        }
        let mut d = AjmOpusDecoder::new(2, PcmEncoding::Float);
        assert_eq!(d.initialize(&params(2, 48000, 0)).result, 0);
        // CELT fullband 20 ms stereo (config 31, s=1), a 1-byte packet is
        // decoded as packet loss concealment / silence.
        let pkt = [1u8, 0, (31 << 3) | 4];
        let mut input = Vec::new();
        for _ in 0..3 {
            input.extend_from_slice(&pkt);
        }
        let mut out = vec![0u8; 3 * 960 * 2 * 4];
        let mut g = GaplessState::default();
        g.set(GaplessDecode { total_samples: 0, skip_samples: 312, skipped_samples: 0 }, true);
        let r = d.decode(&input, &mut out, true, &mut g);
        // The input ran out: partial input after the three packets.
        assert_eq!(r.result, RESULT_PARTIAL_INPUT, "{r:?}");
        assert_eq!(r.frames, 3);
        assert_eq!(r.input_consumed, 9);
        assert_eq!(r.output_written, (3 * 960 - 312) * 2 * 4);
        assert_eq!(r.total_decoded_samples, 3 * 960 - 312);
        assert_eq!(d.codec_info(), 1u32.to_le_bytes().to_vec());
        // Single-frame mode stops after one packet.
        let r = d.decode(&input, &mut out, false, &mut g);
        assert_eq!((r.frames, r.input_consumed), (1, 3));
    }

    /// A run stops before the packet whose frame has no room left, without
    /// an error, once it has decoded a frame; the next run continues there
    /// (reference FMOD Opus run case). With no frame decoded the output is
    /// too small.
    #[test]
    fn run_stops_before_a_frame_without_room() {
        if !cfg!(ps5_ffmpeg) {
            return;
        }
        let mut d = AjmOpusDecoder::new(2, PcmEncoding::S16);
        assert_eq!(d.initialize(&params(2, 48000, 0)).result, 0);
        let pkt = [1u8, 0, (31 << 3) | 4];
        let input: Vec<u8> = pkt.iter().cycle().take(9).copied().collect();
        let frame = 960 * 2 * 2;
        let mut out = vec![0u8; frame + frame / 2];
        let mut g = GaplessState::default();
        let r = d.decode(&input, &mut out, true, &mut g);
        assert_eq!((r.result, r.frames, r.input_consumed, r.output_written), (0, 1, 3, frame), "{r:?}");
        let r = d.decode(&input[3..], &mut out[..frame / 2], true, &mut g);
        assert_eq!((r.result, r.frames), (RESULT_NOT_ENOUGH_ROOM, 0));
    }

    use crate::audio::opus_dec::tests::OPUS_STEREO;

    /// The three 20 ms stereo packets of OPUS_STEREO (code 0, under 252 bytes).
    fn stereo_packets() -> Vec<&'static [u8]> {
        let (mut v, mut at) = (Vec::new(), 0);
        while at < OPUS_STEREO.len() {
            let n = OPUS_STEREO[at] as usize | (OPUS_STEREO[at + 1] as usize) << 8;
            v.push(&OPUS_STEREO[at + 2..at + 2 + n]);
            at += 2 + n;
        }
        v
    }

    fn framed(packet: &[u8]) -> Vec<u8> {
        let mut f = (packet.len() as u16).to_le_bytes().to_vec();
        f.extend_from_slice(packet);
        f
    }

    fn s16(b: &[u8]) -> Vec<i16> {
        b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
    }

    #[test]
    fn self_delimited_framing() {
        assert_eq!(self_delimited_bytes(&[0x7C, 3, 1, 2, 3]), Some(5));
        assert_eq!(self_delimited_bytes(&[0x7D, 2, 1, 2, 3, 4]), Some(6), "two equal frames");
        assert_eq!(self_delimited_bytes(&[0x7E, 1, 2, 9, 9, 9]), Some(6), "two frames");
        assert_eq!(self_delimited_bytes(&[0x7C, 253, 1]), Some(3 + 257), "two-byte length");
        assert_eq!(self_delimited_bytes(&[0x7F, 0x83, 1, 2, 3]), Some(5 + 6), "code 3 VBR: every frame length");
        assert_eq!(self_delimited_bytes(&[0x7F, 0x42, 255, 3, 4]), Some(5 + 8 + 257), "code 3 CBR with padding");
        assert_eq!(self_delimited_bytes(&[0x7F, 0]), None, "no frames");
        assert_eq!(self_delimited_bytes(&[0x7C]), None, "cut");
        // Two stereo streams make four channels; a mono and a stereo three.
        assert_eq!(packet_streams(&[0x7C, 1, 0, 0x7C, 0], 4), Some((2, 2)));
        assert_eq!(packet_streams(&[0x78, 1, 0, 0x7C, 0], 3), Some((2, 1)));
        assert_eq!(packet_streams(&[0x7C, 1, 0, 0x7C, 0], 3), None);
    }

    /// Family 255: two self-delimited copies of a stereo stream decode to
    /// four channels, each pair equal to the stereo decode.
    #[test]
    fn family_255_multistream() {
        if !cfg!(ps5_ffmpeg) {
            return;
        }
        let mut stereo = AjmOpusDecoder::new(2, PcmEncoding::S16);
        let mut multi = AjmOpusDecoder::new(4, PcmEncoding::S16);
        assert_eq!(stereo.initialize(&params(2, 48000, 0)).result, 0);
        assert_eq!(multi.initialize(&params(4, 48000, 255)).result, 0);
        let (mut g1, mut g2) = (GaplessState::default(), GaplessState::default());
        for p in stereo_packets() {
            assert_eq!(p[0] & 3, 0);
            let mut joined = vec![p[0], (p.len() - 1) as u8];
            joined.extend_from_slice(&p[1..]);
            joined.extend_from_slice(p);
            let mut single = vec![0u8; 960 * 2 * 2];
            let mut both = vec![0u8; 960 * 4 * 2];
            assert_eq!(stereo.decode(&framed(p), &mut single, false, &mut g1).result, 0);
            let input = framed(&joined);
            let r = multi.decode(&input, &mut both, false, &mut g2);
            assert_eq!((r.result, r.input_consumed, r.output_written), (0, input.len(), both.len()), "{r:?}");
            let (single, both) = (s16(&single), s16(&both));
            for s in 0..960 {
                for c in 0..2 {
                    assert_eq!(both[s * 4 + c], single[s * 2 + c]);
                    assert_eq!(both[s * 4 + 2 + c], single[s * 2 + c]);
                }
            }
        }
        // A packet whose streams do not make the channels.
        let mut d = AjmOpusDecoder::new(6, PcmEncoding::S16);
        assert_eq!(d.initialize(&params(6, 48000, 255)).result, 0);
        let p = stereo_packets()[0];
        let r = d.decode(&framed(p), &mut vec![0u8; 960 * 12], false, &mut GaplessState::default());
        assert_eq!(r.result, RESULT_CODEC_ERROR | RESULT_INVALID_DATA);
    }

    /// Family 1 surround: the stereo stream in the first coupled stream and
    /// silent CELT streams elsewhere is heard on the front pair only.
    #[test]
    fn family_1_surround() {
        if !cfg!(ps5_ffmpeg) {
            return;
        }
        for channels in [3u32, 4, 5, 6, 7, 8] {
            let streams = VORBIS_STREAMS[channels as usize - 1] as usize;
            let coupled = VORBIS_COUPLED[channels as usize - 1] as usize;
            let mut d = AjmOpusDecoder::new(8, PcmEncoding::S16);
            assert_eq!(d.initialize(&params(channels, 48000, 1)).result, 0, "{channels} channels");
            let mut g = GaplessState::default();
            for (index, p) in stereo_packets().into_iter().enumerate() {
                let mut packet = Vec::new();
                for stream in 0..streams {
                    let last = stream + 1 == streams;
                    if stream == 0 {
                        packet.push(p[0]);
                        if !last {
                            packet.push((p.len() - 1) as u8);
                        }
                        packet.extend_from_slice(&p[1..]);
                    } else {
                        packet.push(if stream < coupled { 0xFC } else { 0xF8 });
                        if !last {
                            packet.push(2);
                        }
                        packet.extend_from_slice(&[0xFF, 0xFE]);
                    }
                }
                let input = framed(&packet);
                let mut out = vec![0u8; 960 * channels as usize * 2];
                let r = d.decode(&input, &mut out, false, &mut g);
                assert_eq!((r.result, r.input_consumed, r.output_written), (0, input.len(), out.len()), "{channels}: {r:?}");
                let pcm = s16(&out);
                for c in 0..channels as usize {
                    let peak = (c..pcm.len()).step_by(channels as usize).map(|i| (pcm[i] as i32).abs()).max().unwrap();
                    // The host reorders Vorbis order into front left/right first.
                    if c < 2 {
                        assert!(index == 0 || peak > 900, "{channels} channels: channel {c} peak {peak}");
                    } else {
                        assert!(peak <= 2, "{channels} channels: channel {c} peak {peak}");
                    }
                }
            }
        }
    }
}
