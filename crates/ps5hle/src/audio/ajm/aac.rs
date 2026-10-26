// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! AJM MPEG-4 AAC decoder (codec 2), decoded on the host. Also used by
//! libSceAudiodec.

use super::decoder::*;
use super::ffmpeg::{HostAudioDecoder, HostCodec, HostError, HostFrame};

pub const FLAG_ENABLE_SBR_DECODE: u64 = 1 << INSTANCE_FLAG_CODEC_OFFSET;
pub const FLAG_ENABLE_NONDELAY_OUTPUT: u64 = 1 << (INSTANCE_FLAG_CODEC_OFFSET + 1);

pub const CONFIG_NUMBER_ADTS: u32 = 1;
pub const CONFIG_NUMBER_RAW: u32 = 2;
pub const CONFIG_NUMBER_SAF: u32 = 3;
const MAX_CHANNELS: u32 = 8;
const MAX_SAMPLING_FREQ_INDEX: u32 = 11;
const ADTS_HEADER_BYTES: usize = 7;
/// Host decoder profiles of HE-AAC (SBR) and HE-AAC v2 (SBR + PS).
const PROFILE_AAC_HE: i32 = 4;
const PROFILE_AAC_HE_V2: i32 = 28;

/// Bytes of the ADTS frame at the start of `data` (header included); None
/// when `data` does not start with an ADTS header.
pub(crate) fn adts_frame_bytes(data: &[u8]) -> Option<usize> {
    if data.len() < ADTS_HEADER_BYTES || data[0] != 0xFF || data[1] & 0xF6 != 0xF0 {
        return None;
    }
    let bytes = ((data[3] as usize & 3) << 11) | ((data[4] as usize) << 3) | (data[5] as usize >> 5);
    let header = if data[1] & 1 != 0 { ADTS_HEADER_BYTES } else { ADTS_HEADER_BYTES + 2 };
    (bytes >= header).then_some(bytes)
}

/// AjmDecM4aacInitializeParameters (8 bytes).
pub const INITIALIZE_PARAMETERS_BYTES: usize = 8;
/// AjmSidebandDecM4aacCodecInfo (8 bytes).
pub const CODEC_INFO_BYTES: usize = 8;

const SAMPLE_RATES: [u32; 12] = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000];

fn sampling_rate_from_index(index: u32) -> u32 {
    SAMPLE_RATES.get(index as usize).copied().unwrap_or(48000)
}

/// Channel configurations a raw stream of at most `max_channels` may use,
/// in the order they are tried: mono or stereo, then 3, 4, 5, 5.1 and 7.1
/// (configuration 7, 8 channels) while they fit.
pub(crate) fn raw_channel_configurations(max_channels: u32) -> Vec<u32> {
    let max = max_channels.min(MAX_CHANNELS);
    let mut v = vec![if max < 2 { 1 } else { 2 }];
    v.extend((3..=7).filter(|&c| if c == 7 { 8 } else { c } <= max));
    v
}

/// AudioSpecificConfig of an AAC LC stream.
fn audio_specific_config(sampling_freq_index: u32, channel_configuration: u32) -> [u8; 2] {
    const AUDIO_OBJECT_TYPE_LC: u32 = 2;
    [
        ((AUDIO_OBJECT_TYPE_LC << 3) | (sampling_freq_index >> 1)) as u8,
        (((sampling_freq_index & 1) << 7) | (channel_configuration << 3)) as u8,
    ]
}

pub struct AjmAacDecoder {
    host: Option<HostAudioDecoder>,
    max_channels: u32,
    flags: u64,
    channels: u32,
    sample_rate: u32,
    encoding: PcmEncoding,
    config_number: u32,
    sampling_freq_index: u32,
    /// Leading frames dropped for the decoder delay.
    skip_frames: u32,
    bitrate: u32,
    initialized: bool,
    raw_configured: bool,
    total_decoded_samples: u64,
    /// The stream decodes as HE-AAC (reported by the codec info).
    heaac: bool,
    /// Samples per channel of the last decoded frame (0 until one decodes).
    frame_samples: u32,
    /// Bytes of the access unit being decoded (for the bit rate).
    packet_bytes: usize,
}

impl AjmAacDecoder {
    pub fn new(max_channels: u32, sample_rate: u32, encoding: PcmEncoding, flags: u64) -> Self {
        Self {
            host: None,
            max_channels,
            flags,
            channels: 0,
            sample_rate,
            encoding,
            config_number: CONFIG_NUMBER_ADTS,
            sampling_freq_index: 3,
            skip_frames: 2,
            bitrate: 0,
            initialized: false,
            raw_configured: false,
            total_decoded_samples: 0,
            heaac: false,
            frame_samples: 0,
            packet_bytes: 0,
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Writes one decoded frame; false stops the job (result set).
    fn write_frame(
        &mut self,
        frame: &HostFrame,
        output: &mut [u8],
        offset: &mut usize,
        gapless: &mut GaplessState,
        result: &mut DecodeResult,
    ) -> bool {
        let channels = frame.channels.max(1);
        if channels > self.max_channels {
            result.result = RESULT_TOO_MANY_CHANNELS;
            return false;
        }
        let heaac = matches!(frame.profile, PROFILE_AAC_HE | PROFILE_AAC_HE_V2);
        if heaac && self.flags & FLAG_ENABLE_SBR_DECODE == 0 {
            log::error!("[AUDIO] AJM AAC: an HE-AAC stream on an instance without ENABLE_SBR_DECODE is not implemented");
            result.result = RESULT_CODEC_ERROR | RESULT_FATAL;
            return false;
        }
        let bpf = channels as usize * self.encoding.bytes_per_sample();
        self.channels = channels;
        self.sample_rate = frame.sample_rate.max(1);
        self.heaac = heaac;
        self.frame_samples = frame.samples;
        if frame.samples != 0 {
            self.bitrate = (self.packet_bytes as u64 * 8 * self.sample_rate as u64 / frame.samples as u64) as u32;
        }

        result.frames += 1;
        if self.skip_frames > 0 {
            self.skip_frames -= 1;
            result.format = self.format();
            return true;
        }

        let (skip, to_write) = gapless.plan(frame.samples);
        let requested = to_write as usize * bpf;
        let mut writable = if *offset >= output.len() { 0 } else { requested.min(output.len() - *offset) };
        writable -= writable % bpf;
        if writable != 0 {
            let src = self.host.as_ref().unwrap().data(frame, self.encoding);
            let start = skip as usize * bpf;
            output[*offset..*offset + writable].copy_from_slice(&src[start..start + writable]);
            *offset += writable;
        }
        let written = (writable / bpf) as u32;
        // Samples planned but not written count as skipped.
        gapless.commit(skip + to_write, written);

        self.total_decoded_samples += written as u64;
        result.total_decoded_samples = self.total_decoded_samples;
        result.format = self.format();
        if writable != requested {
            result.result |= RESULT_NOT_ENOUGH_ROOM;
            return false;
        }
        true
    }

    /// Decode one access unit. A raw stream carries no configuration and the
    /// instance's channel count only bounds it: until an access unit
    /// decodes, every channel configuration up to that bound is tried, from
    /// mono/stereo up, and the first that decodes is kept.
    fn decode_packet(
        &mut self,
        packet: &[u8],
        output: &mut [u8],
        offset: &mut usize,
        gapless: &mut GaplessState,
        result: &mut DecodeResult,
    ) -> bool {
        if self.config_number != CONFIG_NUMBER_RAW || self.raw_configured {
            return self.run_packet(packet, None, output, offset, gapless, result);
        }
        let saved = (self.skip_frames, self.total_decoded_samples, self.channels, self.sample_rate, self.bitrate);
        let configurations = raw_channel_configurations(self.max_channels);
        for (k, &configuration) in configurations.iter().enumerate() {
            if k != 0 {
                self.host = HostAudioDecoder::open(HostCodec::Aac, 0, 0);
                if self.host.is_none() {
                    result.result = RESULT_CODEC_ERROR | RESULT_FATAL;
                    return false;
                }
            }
            let config = audio_specific_config(self.sampling_freq_index, configuration);
            let (mut attempt, mut at, mut g) = (*result, *offset, *gapless);
            let ok = self.run_packet(packet, Some(&config), output, &mut at, &mut g, &mut attempt);
            if attempt.result & RESULT_INVALID_DATA == 0 {
                self.raw_configured = true;
                (*result, *offset, *gapless) = (attempt, at, g);
                return ok;
            }
            (self.skip_frames, self.total_decoded_samples, self.channels, self.sample_rate, self.bitrate) = saved;
        }
        // No configuration decodes this access unit: the next one searches
        // again on a fresh decoder.
        self.host = HostAudioDecoder::open(HostCodec::Aac, 0, 0);
        result.result = RESULT_CODEC_ERROR | RESULT_INVALID_DATA;
        false
    }

    /// ADTS input: one frame per run (every complete frame with
    /// `multiple_frames`), split by the header's frame length. Returns
    /// whether the run ended without an error and the bytes consumed.
    fn decode_adts(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        offset: &mut usize,
        multiple_frames: bool,
        gapless: &mut GaplessState,
        result: &mut DecodeResult,
    ) -> (bool, usize) {
        let mut consumed = 0usize;
        let mut decoded = 0u32;
        while consumed < input.len() {
            let data = &input[consumed..];
            if data.len() < ADTS_HEADER_BYTES {
                result.result |= RESULT_PARTIAL_INPUT;
                break;
            }
            let Some(frame_bytes) = adts_frame_bytes(data) else {
                log::error!("[AUDIO] AJM AAC: no ADTS header at the consumed offset (resynchronization is not implemented)");
                result.result = RESULT_CODEC_ERROR | RESULT_INVALID_DATA;
                return (false, consumed);
            };
            if frame_bytes > data.len() {
                result.result |= RESULT_PARTIAL_INPUT;
                break;
            }
            // A frame needs room for its samples (known once one decoded).
            let bpf = self.channels as usize * self.encoding.bytes_per_sample();
            let (_, emitted) = gapless.preview(self.frame_samples);
            if output.len() - *offset < emitted as usize * bpf {
                if decoded == 0 && *offset == 0 {
                    result.result |= RESULT_NOT_ENOUGH_ROOM;
                }
                break;
            }
            if !self.run_packet(&data[..frame_bytes], None, output, offset, gapless, result) {
                return (false, consumed);
            }
            consumed += frame_bytes;
            decoded += 1;
            if !multiple_frames {
                break;
            }
        }
        (true, consumed)
    }

    /// Send one access unit (with `extradata` the first time) and write the
    /// frames it yields.
    fn run_packet(
        &mut self,
        packet: &[u8],
        extradata: Option<&[u8]>,
        output: &mut [u8],
        offset: &mut usize,
        gapless: &mut GaplessState,
        result: &mut DecodeResult,
    ) -> bool {
        self.packet_bytes = packet.len();
        let Some(host) = self.host.as_mut() else {
            result.result = RESULT_CODEC_ERROR | RESULT_FATAL;
            return false;
        };
        match host.send(packet, extradata) {
            Ok(()) => {}
            Err(HostError::InvalidData) => {
                result.result = RESULT_CODEC_ERROR | RESULT_INVALID_DATA;
                return false;
            }
            Err(_) => {
                result.result = RESULT_PARTIAL_INPUT;
                return false;
            }
        }
        loop {
            let frame = match self.host.as_mut().unwrap().receive(self.encoding) {
                Ok(None) => return true,
                Ok(Some(f)) => f,
                Err(HostError::Api) => {
                    result.result = RESULT_CODEC_ERROR | RESULT_FATAL;
                    return false;
                }
                Err(_) => {
                    result.result = RESULT_CODEC_ERROR | RESULT_INVALID_DATA;
                    return false;
                }
            };
            if !self.write_frame(&frame, output, offset, gapless, result) {
                return false;
            }
        }
    }
}

impl AjmDecoder for AjmAacDecoder {
    fn initialize(&mut self, params: &[u8]) -> DecodeResult {
        let mut result = self.make_result();
        if params.len() < INITIALIZE_PARAMETERS_BYTES {
            result.result = RESULT_INVALID_PARAMETER;
            return result;
        }
        let config_number = u32::from_le_bytes(params[0..4].try_into().unwrap());
        let freq_index = u32::from_le_bytes(params[4..8].try_into().unwrap());
        if !matches!(config_number, CONFIG_NUMBER_ADTS | CONFIG_NUMBER_RAW | CONFIG_NUMBER_SAF) {
            result.result = RESULT_INVALID_PARAMETER;
            return result;
        }
        if config_number == CONFIG_NUMBER_RAW && freq_index > MAX_SAMPLING_FREQ_INDEX {
            result.result = RESULT_INVALID_PARAMETER;
            return result;
        }
        self.config_number = config_number;
        self.sampling_freq_index = freq_index;
        self.initialized = true;
        self.reset();
        if self.host.is_none() {
            self.initialized = false;
            result.result = RESULT_CODEC_ERROR | RESULT_FATAL;
            return result;
        }
        log::debug!("[AUDIO] AJM AAC initialized: config={config_number}, freq_index={freq_index}");
        result
    }

    fn reset(&mut self) {
        self.host = None;
        self.skip_frames = if self.flags & FLAG_ENABLE_NONDELAY_OUTPUT != 0 { 0 } else { 2 };
        self.total_decoded_samples = 0;
        self.bitrate = 0;
        self.channels = 0;
        self.raw_configured = false;
        self.heaac = false;
        self.frame_samples = 0;
        if self.config_number == CONFIG_NUMBER_RAW {
            self.sample_rate = sampling_rate_from_index(self.sampling_freq_index);
        }
        if self.initialized {
            self.host = HostAudioDecoder::open(HostCodec::Aac, 0, 0);
            if self.host.is_none() {
                log::error!("[AUDIO] AJM AAC: decoder initialization failed");
            }
        }
    }

    fn decode(&mut self, input: &[u8], output: &mut [u8], multiple_frames: bool, gapless: &mut GaplessState) -> DecodeResult {
        let mut result = self.make_result();
        if !self.initialized {
            result.result = RESULT_NOT_INITIALIZED;
            return result;
        }
        if self.host.is_none() {
            result.result = RESULT_CODEC_ERROR | RESULT_FATAL;
            return result;
        }
        if input.is_empty() {
            result.result = RESULT_PARTIAL_INPUT;
            return result;
        }
        if input.len() > i32::MAX as usize {
            result.result = RESULT_INVALID_PARAMETER;
            return result;
        }
        let mut offset = 0usize;
        let ok = if self.config_number == CONFIG_NUMBER_ADTS {
            let (ok, consumed) = self.decode_adts(input, output, &mut offset, multiple_frames, gapless, &mut result);
            result.input_consumed = consumed;
            ok
        } else {
            result.input_consumed = input.len();
            self.decode_packet(input, output, &mut offset, gapless, &mut result)
        };
        if ok && result.frames == 0 && offset == 0 && result.result == 0 {
            result.result = RESULT_PARTIAL_INPUT;
        }
        result.output_written = offset;
        result.total_decoded_samples = self.total_decoded_samples;
        result.format = self.format();
        result
    }

    fn codec_info(&self) -> Vec<u8> {
        let mut b = (self.heaac as u32).to_le_bytes().to_vec();
        b.extend_from_slice(&0u32.to_le_bytes());
        b
    }

    fn codec_info_size(&self) -> usize {
        CODEC_INFO_BYTES
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

    fn params(config: u32, index: u32) -> Vec<u8> {
        let mut p = config.to_le_bytes().to_vec();
        p.extend_from_slice(&index.to_le_bytes());
        p
    }

    #[test]
    fn initialize_validates_parameters() {
        let mut d = AjmAacDecoder::new(2, 48000, PcmEncoding::S16, 0);
        let mut g = GaplessState::default();
        assert_eq!(d.decode(&[1, 2, 3], &mut [0u8; 16], true, &mut g).result, RESULT_NOT_INITIALIZED);
        assert_eq!(d.initialize(&[1, 0, 0]).result, RESULT_INVALID_PARAMETER);
        assert_eq!(d.initialize(&params(4, 0)).result, RESULT_INVALID_PARAMETER);
        assert_eq!(d.initialize(&params(CONFIG_NUMBER_RAW, 12)).result, RESULT_INVALID_PARAMETER);
        let r = d.initialize(&params(CONFIG_NUMBER_RAW, 4));
        if cfg!(ps5_ffmpeg) {
            assert_eq!(r.result, 0);
            assert_eq!(d.format().sample_rate, 44100);
            assert_eq!(d.decode(&[], &mut [0u8; 16], true, &mut g).result, RESULT_PARTIAL_INPUT);
        } else {
            assert_eq!(r.result, RESULT_CODEC_ERROR | RESULT_FATAL);
        }
        // ADTS keeps the instance sample rate until a frame is decoded.
        let mut d = AjmAacDecoder::new(2, 48000, PcmEncoding::S16, 0);
        d.initialize(&params(CONFIG_NUMBER_ADTS, 9));
        assert_eq!(d.format().sample_rate, 48000);
    }

    /// The HE-AAC flag follows the decoded stream, not the instance flag.
    #[test]
    fn codec_info_reports_the_stream_profile() {
        let d = AjmAacDecoder::new(2, 48000, PcmEncoding::S16, FLAG_ENABLE_SBR_DECODE);
        assert_eq!(d.codec_info(), vec![0; 8], "no stream decoded yet");
        assert_eq!(d.codec_info_size(), 8);
        if !cfg!(ps5_ffmpeg) {
            return;
        }
        let mut d = AjmAacDecoder::new(8, 48000, PcmEncoding::S16, FLAG_ENABLE_SBR_DECODE | FLAG_ENABLE_NONDELAY_OUTPUT);
        assert_eq!(d.initialize(&params(CONFIG_NUMBER_ADTS, 3)).result, 0);
        let mut out = vec![0u8; 1024 * 6 * 2];
        assert_eq!(d.decode(&adts_51(), &mut out, false, &mut GaplessState::default()).result, 0);
        assert_eq!(d.codec_info(), vec![0; 8], "an AAC LC stream");
    }

    #[test]
    fn adts_header_lengths() {
        let frame = adts_51();
        assert_eq!(adts_frame_bytes(&frame), Some(frame.len()));
        assert_eq!(adts_frame_bytes(&frame[..6]), None, "short header");
        let mut bad = frame.clone();
        bad[1] = 0xE1;
        assert_eq!(adts_frame_bytes(&bad), None, "no sync word");
        // A frame shorter than its header.
        let mut short = frame.clone();
        (short[3], short[4], short[5]) = (0x80, 0, 6 << 5);
        assert_eq!(adts_frame_bytes(&short), None);
        // With a CRC the header is 9 bytes.
        let mut crc = frame;
        crc[1] = 0xF0;
        (crc[3], crc[4], crc[5]) = (0x80, 1, 0);
        assert_eq!(adts_frame_bytes(&crc), None, "8 bytes do not hold a 9-byte header");
        (crc[4], crc[5]) = (1, 1 << 5);
        assert_eq!(adts_frame_bytes(&crc), Some(9));
    }

    /// ADTS input is split by frame: one frame per run, every complete
    /// frame with multiple frames; a cut frame is partial input. The
    /// first two frames are the decoder delay unless NONDELAY_OUTPUT.
    #[test]
    fn adts_frames_per_run() {
        if !cfg!(ps5_ffmpeg) {
            return;
        }
        let frame = adts_51();
        let n = frame.len();
        let mut input = frame.clone();
        input.extend_from_slice(&frame);
        input.extend_from_slice(&frame[..100]);
        let pcm = 1024 * 6 * 2;
        let mut out = vec![0u8; 4 * pcm];

        let mut d = AjmAacDecoder::new(8, 48000, PcmEncoding::S16, FLAG_ENABLE_NONDELAY_OUTPUT);
        assert_eq!(d.initialize(&params(CONFIG_NUMBER_ADTS, 3)).result, 0);
        let mut g = GaplessState::default();
        let r = d.decode(&input, &mut out, false, &mut g);
        assert_eq!((r.result, r.frames, r.input_consumed, r.output_written), (0, 1, n, pcm), "{r:?}");
        assert_eq!(d.format().bitrate, (n * 8 * 48000 / 1024) as u32);
        let r = d.decode(&input, &mut out, true, &mut g);
        assert_eq!((r.result, r.frames, r.input_consumed, r.output_written), (RESULT_PARTIAL_INPUT, 2, 2 * n, 2 * pcm), "{r:?}");
        // No room for a frame of the known length.
        let r = d.decode(&input, &mut out[..pcm - 2], false, &mut g);
        assert_eq!((r.result, r.input_consumed, r.output_written), (RESULT_NOT_ENOUGH_ROOM, 0, 0));
        // Garbage at the consumed offset is not resynchronized.
        let r = d.decode(&[0x12; 64], &mut out, true, &mut g);
        assert_eq!((r.result, r.input_consumed, r.output_written), (RESULT_CODEC_ERROR | RESULT_INVALID_DATA, 0, 0));

        let mut d = AjmAacDecoder::new(8, 48000, PcmEncoding::S16, 0);
        assert_eq!(d.initialize(&params(CONFIG_NUMBER_ADTS, 3)).result, 0);
        let mut g = GaplessState::default();
        let r = d.decode(&input, &mut out, true, &mut g);
        assert_eq!((r.frames, r.input_consumed, r.output_written), (2, 2 * n, 0), "delay frames: {r:?}");
    }

    #[test]
    fn garbage_is_rejected_without_output() {
        if !cfg!(ps5_ffmpeg) {
            return;
        }
        let mut d = AjmAacDecoder::new(2, 48000, PcmEncoding::S16, 0);
        assert_eq!(d.initialize(&params(CONFIG_NUMBER_ADTS, 3)).result, 0);
        let mut out = vec![0u8; 8192];
        let r = d.decode(&[0x12; 64], &mut out, true, &mut GaplessState::default());
        assert_ne!(r.result, 0);
        assert_eq!(r.output_written, 0);
        assert_eq!(r.input_consumed, 0);
    }
    #[test]
    fn raw_configuration_search_order() {
        assert_eq!(raw_channel_configurations(0), [1]);
        assert_eq!(raw_channel_configurations(1), [1]);
        assert_eq!(raw_channel_configurations(2), [2]);
        assert_eq!(raw_channel_configurations(6), [2, 3, 4, 5, 6]);
        assert_eq!(raw_channel_configurations(7), [2, 3, 4, 5, 6], "configuration 7 carries 8 channels");
        assert_eq!(raw_channel_configurations(8), [2, 3, 4, 5, 6, 7]);
        assert_eq!(audio_specific_config(3, 6), [0x11, 0xB0]);
    }

    /// One raw AAC LC access unit of a 48 kHz 5.1 stream.
    const AAC_RAW_51: [u8; 773] = [
        0x00, 0xDE, 0x34, 0xAC, 0xF1, 0x1D, 0x0C, 0x05, 0x89, 0xA2, 0x61, 0x40, 0xB4, 0x4A, 0x37, 0x8D, 0xFF, 0x7F, 0xFD, 0x3F,
        0xCF, 0xFB, 0x79, 0xE7, 0xE3, 0xCE, 0x4C, 0x8B, 0x5D, 0xD5, 0xD5, 0x3F, 0xE9, 0x47, 0x6A, 0x5D, 0x69, 0xBA, 0xB1, 0x9A,
        0x78, 0xAA, 0x3C, 0xD4, 0xF7, 0xA6, 0xD3, 0x13, 0xCC, 0x21, 0x93, 0x2B, 0x3E, 0x67, 0xDE, 0x74, 0x1A, 0xA3, 0xE7, 0x31,
        0xE2, 0x47, 0xE8, 0x68, 0xFB, 0xD5, 0xF6, 0xB5, 0xFC, 0x57, 0xDE, 0x9E, 0x43, 0x6F, 0x3E, 0xF5, 0xF2, 0x1F, 0xBC, 0xFB,
        0xC5, 0xBC, 0x53, 0x16, 0xE4, 0x57, 0xAC, 0x64, 0x14, 0xE6, 0x88, 0xAF, 0x66, 0xBA, 0x8C, 0x7A, 0x86, 0xF5, 0x8D, 0xC4,
        0xA2, 0x36, 0x89, 0xE3, 0x21, 0xF9, 0xA5, 0x8A, 0xAF, 0xF0, 0xE5, 0x91, 0x2C, 0x15, 0xA2, 0x67, 0xF4, 0x97, 0xDA, 0x40,
        0x0A, 0x00, 0x8A, 0x45, 0x7C, 0x92, 0xEA, 0x8F, 0x87, 0x56, 0xC2, 0x40, 0xE1, 0xA2, 0x23, 0x1E, 0x89, 0xE7, 0xBA, 0x61,
        0xA3, 0xD5, 0xB9, 0x77, 0x27, 0x13, 0x73, 0x49, 0xE4, 0x98, 0x83, 0x3A, 0x25, 0x86, 0x3A, 0x10, 0x3D, 0x5C, 0xC8, 0xC0,
        0x8F, 0xDB, 0x58, 0x50, 0xB0, 0xFB, 0x3F, 0x33, 0xAD, 0xE1, 0xD7, 0xC2, 0xF7, 0xB3, 0x98, 0xDF, 0x39, 0x87, 0xE2, 0x97,
        0x34, 0xB4, 0x2A, 0x3A, 0xA0, 0x64, 0x0A, 0x80, 0x76, 0x84, 0xA6, 0x1D, 0x43, 0xF4, 0x0E, 0x12, 0x3E, 0xF2, 0xE2, 0x6A,
        0x1F, 0x30, 0xA1, 0xD5, 0xE8, 0x0A, 0x22, 0x39, 0x9E, 0x44, 0xF3, 0xD1, 0x81, 0x6F, 0x49, 0x5A, 0x9C, 0x1F, 0x54, 0xB9,
        0x2B, 0xB4, 0x73, 0xA3, 0x8D, 0x74, 0x6F, 0x87, 0x2D, 0x51, 0xCE, 0x3A, 0x39, 0xED, 0x38, 0x49, 0x55, 0x7C, 0x7D, 0xBE,
        0xED, 0x65, 0x29, 0xAE, 0x3E, 0xBE, 0x70, 0xD1, 0x08, 0xD2, 0x1B, 0xEC, 0xF0, 0x66, 0x56, 0x8A, 0x03, 0xA9, 0x80, 0xE9,
        0x5F, 0x2F, 0xFA, 0x7F, 0xF1, 0xFE, 0xFF, 0xED, 0xE7, 0x9F, 0x8F, 0x3C, 0xEB, 0x37, 0xAC, 0xEA, 0xD9, 0x75, 0x02, 0xD4,
        0xDC, 0xDD, 0x45, 0xED, 0x9C, 0x2A, 0x83, 0x34, 0xF1, 0x54, 0xD2, 0x3F, 0xBD, 0x36, 0x98, 0x69, 0x87, 0x04, 0xAA, 0x5F,
        0x99, 0xF7, 0x9D, 0x06, 0xA8, 0xF9, 0xCC, 0x78, 0x91, 0xFA, 0x1A, 0x3E, 0xF5, 0x7C, 0xD8, 0xF7, 0x9F, 0x7A, 0x79, 0x0D,
        0xD5, 0x45, 0x5C, 0x55, 0x2A, 0x4A, 0xA0, 0x66, 0x59, 0xC6, 0x82, 0x89, 0x66, 0x0C, 0x65, 0xF5, 0xE1, 0xEA, 0xEE, 0xD9,
        0x9E, 0x5F, 0x5B, 0xB4, 0xE0, 0x93, 0x0F, 0x11, 0xA2, 0x49, 0xC3, 0x2F, 0xC7, 0xA1, 0x6E, 0x1A, 0x81, 0xFF, 0x57, 0x7D,
        0x6F, 0xC3, 0xAB, 0x61, 0x20, 0x70, 0xD1, 0x11, 0x8F, 0x44, 0xEB, 0x3C, 0xF4, 0x60, 0x59, 0xA4, 0xAD, 0x4E, 0x0F, 0x54,
        0xB4, 0x5C, 0xDF, 0x67, 0x88, 0x68, 0xA0, 0x3A, 0x9F, 0x0E, 0x95, 0xF2, 0xFF, 0xB7, 0xFE, 0x3F, 0xDF, 0xFD, 0xBC, 0xF3,
        0xF1, 0xE7, 0xBE, 0xB2, 0xA6, 0xFA, 0xBA, 0x49, 0x02, 0xD5, 0x6E, 0x6E, 0xA2, 0x61, 0xC2, 0xA8, 0x33, 0x4F, 0x15, 0x4D,
        0x23, 0xFB, 0xD3, 0x69, 0x86, 0x6C, 0x70, 0x4A, 0xA5, 0xF9, 0x9F, 0x79, 0xD0, 0x6A, 0x8F, 0x9C, 0xC7, 0x89, 0x1F, 0xA1,
        0xA3, 0xEF, 0x57, 0xCD, 0x2D, 0xE7, 0xEF, 0x4F, 0x21, 0xB7, 0x9F, 0x7A, 0xF9, 0x0F, 0xDE, 0x3B, 0xC5, 0xBC, 0x50, 0x0B,
        0x33, 0xEB, 0x19, 0x05, 0x39, 0xA2, 0x2B, 0xD9, 0xAE, 0xA3, 0x1D, 0x81, 0xCC, 0x1B, 0xCC, 0x9C, 0xA3, 0x5F, 0x03, 0xE4,
        0x96, 0xE6, 0xF8, 0xAC, 0xC0, 0x98, 0x22, 0xA8, 0x3E, 0x29, 0x7A, 0xC8, 0x30, 0xEA, 0xD8, 0x48, 0x1C, 0x34, 0x44, 0x63,
        0xD1, 0x39, 0x18, 0xD2, 0x1B, 0xEC, 0xF1, 0x0D, 0x14, 0x07, 0x52, 0xE1, 0xD2, 0xBE, 0x5F, 0xF6, 0xFF, 0xC7, 0xFB, 0xFF,
        0xB7, 0x9E, 0x7E, 0x3C, 0xF7, 0xD6, 0x54, 0xDF, 0x57, 0x49, 0x20, 0x5A, 0xAD, 0xCD, 0xD4, 0x4C, 0x6F, 0x85, 0x50, 0x66,
        0x9E, 0x2A, 0x9A, 0x47, 0xF7, 0xA6, 0xD3, 0x0C, 0xD8, 0xE0, 0x95, 0x4B, 0xF3, 0x3E, 0xF3, 0xA0, 0xD5, 0x1F, 0x39, 0x8F,
        0x12, 0x3F, 0x43, 0x47, 0xDE, 0xAF, 0x9A, 0x5B, 0xCF, 0xDE, 0x9E, 0x43, 0x6F, 0x3E, 0xF5, 0xF2, 0x1F, 0xBC, 0x77, 0x8B,
        0x78, 0xA0, 0x16, 0x67, 0xD6, 0x32, 0x0A, 0x73, 0x44, 0x57, 0xB3, 0x5D, 0x46, 0x3B, 0x03, 0x98, 0x37, 0x99, 0x39, 0x46,
        0xBE, 0x07, 0xC9, 0x2D, 0xCD, 0xF1, 0x59, 0x81, 0x30, 0x45, 0x50, 0x7C, 0x52, 0xF5, 0x90, 0x61, 0xD5, 0xB0, 0x90, 0x38,
        0x68, 0x88, 0xC7, 0xA2, 0x7B, 0x9D, 0x3C, 0x6A, 0xE3, 0x56, 0xE9, 0x8F, 0x73, 0x3A, 0xA6, 0xFB, 0x3C, 0x43, 0x45, 0x01,
        0xD4, 0xF8, 0x74, 0xAF, 0x97, 0xFD, 0xBF, 0xF1, 0xFE, 0xFF, 0xED, 0xE7, 0x9F, 0x8F, 0x3D, 0xF5, 0x95, 0x37, 0xD5, 0xD2,
        0x48, 0x16, 0xAB, 0x73, 0x75, 0x13, 0x0E, 0x15, 0x41, 0x9A, 0x78, 0xAA, 0x69, 0x1F, 0xDE, 0x9B, 0x4C, 0x33, 0x63, 0x82,
        0x55, 0x2F, 0xCC, 0xFB, 0xCE, 0x83, 0x54, 0x7C, 0xE6, 0x3C, 0x48, 0xFD, 0x0D, 0x1F, 0x7A, 0xBE, 0x69, 0x6F, 0x3F, 0x7A,
        0x79, 0x0D, 0xBC, 0xFB, 0xD7, 0xC8, 0x7E, 0xF1, 0xDE, 0x2D, 0xE2, 0x80, 0x59, 0x9F, 0x58, 0xC8, 0x29, 0xCD, 0x11, 0x5E,
        0xCD, 0x75, 0x18, 0xEC, 0x0E, 0x60, 0xDE, 0x64, 0xE5, 0x1A, 0xF8, 0x1F, 0x24, 0xB7, 0x37, 0xC5, 0x66, 0x04, 0xC1, 0x15,
        0x41, 0xF1, 0x4B, 0xD6, 0x41, 0x87, 0x56, 0xC2, 0x40, 0xE1, 0xA2, 0x23, 0x1E, 0x89, 0xD8, 0x3B, 0x00, 0x6B, 0x1B, 0xED,
        0x80, 0x8B, 0x8E, 0x4A, 0x30, 0x78, 0x0D, 0xAE, 0x19, 0x60, 0x33, 0x3A, 0x38,
    ];

    /// AAC_RAW_51 in an ADTS frame: LC, 48 kHz, configuration 6, no CRC.
    fn adts_51() -> Vec<u8> {
        let len = AAC_RAW_51.len() + 7;
        let mut f = vec![0xFF, 0xF1, 0x4D, 0x80 | (len >> 11) as u8, (len >> 3) as u8, ((len & 7) << 5) as u8 | 0x1F, 0xFC];
        f.extend_from_slice(&AAC_RAW_51);
        f
    }

    /// A raw 5.1 access unit under a limit of 6 or 8 channels is only
    /// decoded by moving past the mono/stereo configuration.
    #[test]
    fn raw_5_1_stream_finds_its_configuration() {
        if !cfg!(ps5_ffmpeg) {
            return;
        }
        for max_channels in [6, 8] {
            let mut d = AjmAacDecoder::new(max_channels, 48000, PcmEncoding::S16, FLAG_ENABLE_NONDELAY_OUTPUT);
            assert_eq!(d.initialize(&params(CONFIG_NUMBER_RAW, 3)).result, 0);
            let mut out = vec![0u8; 1024 * 8 * 2];
            let r = d.decode(&AAC_RAW_51, &mut out, false, &mut GaplessState::default());
            assert_eq!(r.result, 0, "max {max_channels}: {r:?}");
            assert_eq!(r.output_written, 1024 * 6 * 2, "max {max_channels}");
            assert_eq!((d.format().channels, d.format().sample_rate), (6, 48000));
            assert_eq!(r.input_consumed, AAC_RAW_51.len());
        }
    }
}
