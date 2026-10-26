// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! SceNgs2WaveformInfo construction from a raw waveform image (the work of
//! sceNgs2ParseWaveformData/File). Layout:
//!
//! ```text
//! SceNgs2WaveformFormat (0x18): waveformType, numChannels, sampleRate,
//!                               configData, frameOffset, frameMargin
//! +0x18 dataOffset   +0x1C dataSize      +0x20 loopBeginPosition
//! +0x24 loopEndPosition +0x28 numSamples +0x2C audioUnitSize
//! +0x30 numAudioUnitSamples +0x34 numAudioUnitPerFrame
//! +0x38 audioFrameSize +0x3C numAudioFrameSamples
//! +0x40 numDelaySamples +0x44 numBlocks
//! +0x48 aBlock[4] (0x28 each): u64 dataOffset, u64 dataSize,
//!       u32 numRepeats, u32 numSkipSamples, u32 numSamples, u32 reserved,
//!       u64 userData
//! ```

use super::vag;
use crate::audio::atrac9::Atrac9Config;

pub const WAVEFORM_TYPE_PCM_I8: u32 = 0x10;
pub const WAVEFORM_TYPE_PCM_U8: u32 = 0x11;
pub const WAVEFORM_TYPE_PCM_I16L: u32 = 0x12;
pub const WAVEFORM_TYPE_PCM_I24L: u32 = 0x14;
pub const WAVEFORM_TYPE_PCM_I32L: u32 = 0x16;
pub const WAVEFORM_TYPE_PCM_F32L: u32 = 0x18;
pub const WAVEFORM_TYPE_PCM_F64L: u32 = 0x1A;
pub const WAVEFORM_TYPE_VAG: u32 = 0x1C;
pub const WAVEFORM_TYPE_ATRAC9: u32 = 0x40;

/// sizeof(SceNgs2WaveformFormat).
pub const FORMAT_SIZE: usize = 0x18;
/// sizeof(SceNgs2WaveformBlock).
pub const BLOCK_SIZE: usize = 0x28;
/// SCE_NGS2_WAVEFORM_INFO_MAX_BLOCKS.
pub const MAX_BLOCKS: usize = 4;
pub const BLOCK_ARRAY_OFFSET: usize = 0x48;
/// sizeof(SceNgs2WaveformInfo).
pub const WAVEFORM_INFO_SIZE: usize = BLOCK_ARRAY_OFFSET + MAX_BLOCKS * BLOCK_SIZE;

/// One HE-VAG unit: 16 bytes per channel, 28 samples.
const VAG_UNIT_BYTES_PER_CHANNEL: u32 = 16;
const VAG_UNIT_SAMPLES: u32 = 28;

/// WAVE_FORMAT_EXTENSIBLE SubFormat GUID offset inside the "fmt " chunk.
const SUB_FORMAT_OFFSET: usize = 0x18;
/// {47E142D2-36BA-4D8D-88FC-61654F8C836C}: Sony ATRAC9.
const ATRAC9_SUB_FORMAT: [u8; 16] = [
    0xD2, 0x42, 0xE1, 0x47, 0xBA, 0x36, 0x8D, 0x4D, 0x88, 0xFC, 0x61, 0x65, 0x4F, 0x8C, 0x83, 0x6C,
];
/// KSDATAFORMAT_SUBTYPE_PCM.
const PCM_SUB_FORMAT: [u8; 16] = [
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];
/// KSDATAFORMAT_SUBTYPE_IEEE_FLOAT.
const FLOAT_SUB_FORMAT: [u8; 16] = [
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WaveformInfo {
    pub waveform_type: u32,
    pub num_channels: u32,
    pub sample_rate: u32,
    pub config_data: u32,
    pub frame_margin: u32,
    pub frame_offset: u32,

    pub data_offset: u32,
    pub data_size: u32,
    pub loop_begin_position: u32,
    pub loop_end_position: u32,
    pub num_samples: u32,

    pub audio_unit_size: u32,
    pub num_audio_unit_samples: u32,
    pub num_audio_unit_per_frame: u32,
    pub audio_frame_size: u32,
    pub num_audio_frame_samples: u32,
    pub num_delay_samples: u32,
    pub num_blocks: u32,
    /// numSkipSamples of the single block (ATRAC9 "fact" encoder delay).
    pub block_skip_samples: u32,
}

fn le16(d: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([d[o], d[o + 1]])
}

fn le32(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
}

fn put32(d: &mut [u8], o: usize, v: u32) {
    d[o..o + 4].copy_from_slice(&v.to_le_bytes());
}

/// Serialize into a guest SceNgs2WaveformInfo image (exactly
/// `WAVEFORM_INFO_SIZE` bytes). The single block mirrors the whole data
/// chunk, so a title can feed it straight back into sceNgs2VoiceControl.
pub fn write(info: &WaveformInfo) -> [u8; WAVEFORM_INFO_SIZE] {
    let mut d = [0u8; WAVEFORM_INFO_SIZE];
    put32(&mut d, 0x00, info.waveform_type);
    put32(&mut d, 0x04, info.num_channels);
    put32(&mut d, 0x08, info.sample_rate);
    put32(&mut d, 0x0C, info.config_data);
    put32(&mut d, 0x10, info.frame_offset);
    put32(&mut d, 0x14, info.frame_margin);
    put32(&mut d, 0x18, info.data_offset);
    put32(&mut d, 0x1C, info.data_size);
    put32(&mut d, 0x20, info.loop_begin_position);
    put32(&mut d, 0x24, info.loop_end_position);
    put32(&mut d, 0x28, info.num_samples);
    put32(&mut d, 0x2C, info.audio_unit_size);
    put32(&mut d, 0x30, info.num_audio_unit_samples);
    put32(&mut d, 0x34, info.num_audio_unit_per_frame);
    put32(&mut d, 0x38, info.audio_frame_size);
    put32(&mut d, 0x3C, info.num_audio_frame_samples);
    put32(&mut d, 0x40, info.num_delay_samples);
    put32(&mut d, 0x44, info.num_blocks);
    if info.num_blocks != 0 {
        let b = BLOCK_ARRAY_OFFSET;
        d[b..b + 8].copy_from_slice(&(info.data_offset as u64).to_le_bytes());
        d[b + 8..b + 16].copy_from_slice(&(info.data_size as u64).to_le_bytes());
        put32(&mut d, b + 0x10, 0);
        put32(&mut d, b + 0x14, info.block_skip_samples);
        put32(&mut d, b + 0x18, info.num_samples);
    }
    d
}

fn is_riff_wave(data: &[u8]) -> bool {
    data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WAVE"
}

/// True when the image starts with a container this parser understands.
pub fn is_supported_container(data: &[u8]) -> bool {
    vag::is_vag(data) || is_riff_wave(data)
}

pub fn try_parse(data: &[u8]) -> Option<WaveformInfo> {
    let mut info = WaveformInfo::default();
    let ok = if vag::is_vag(data) {
        try_parse_vag(data, &mut info)
    } else {
        try_parse_riff(data, &mut info)
    };
    ok.then_some(info)
}

/// "VAGp" container: 0x30-byte big-endian header, then 16-byte PS-ADPCM
/// units, 28 samples each.
fn try_parse_vag(data: &[u8], info: &mut WaveformInfo) -> bool {
    if data.len() < vag::VAG_HEADER_SIZE {
        return false;
    }
    let declared = vag::be32(data, 0x0C);
    let sample_rate = vag::be32(data, 0x10);
    let channels = if data[0x1E] == 0 { 1 } else { data[0x1E] as u32 };
    let available = (data.len() - vag::VAG_HEADER_SIZE) as u32;
    let data_size = if declared == 0 || declared > available { available } else { declared };

    info.waveform_type = WAVEFORM_TYPE_VAG;
    info.num_channels = channels;
    info.sample_rate = if sample_rate == 0 { 48000 } else { sample_rate };
    info.data_offset = vag::VAG_HEADER_SIZE as u32;
    info.data_size = data_size;
    info.audio_unit_size = VAG_UNIT_BYTES_PER_CHANNEL.wrapping_mul(channels);
    info.num_audio_unit_samples = VAG_UNIT_SAMPLES;
    info.num_audio_unit_per_frame = 1;
    info.num_samples = if info.audio_unit_size == 0 {
        0
    } else {
        (data_size / info.audio_unit_size).wrapping_mul(VAG_UNIT_SAMPLES)
    };
    finish_derived_fields(info);
    true
}

fn try_parse_riff(data: &[u8], info: &mut WaveformInfo) -> bool {
    if !is_riff_wave(data) {
        return false;
    }
    let mut have_format = false;
    let (mut data_offset, mut data_size) = (0u32, 0u32);
    let (mut loop_begin, mut loop_end) = (0u32, 0u32);
    let mut declared_samples = 0u32;
    let mut fact: Option<(u32, u32)> = None;
    let mut offset = 12usize;
    while offset + 8 <= data.len() {
        let id = &data[offset..offset + 4];
        let chunk_size = le32(data, offset + 4);
        let body = offset + 8;
        let body_size = (chunk_size as usize).min(data.len() - body);
        let chunk = &data[body..body + body_size];
        match id {
            b"fmt " => {
                if !try_parse_format_chunk(chunk, info) {
                    return false;
                }
                have_format = true;
            }
            b"data" => {
                data_offset = body as u32;
                data_size = body_size as u32;
            }
            // "fact" carries the decoded sample count for compressed formats.
            b"fact" if body_size >= 4 => {
                declared_samples = le32(chunk, 0);
                // ATRAC9 "fact": numSamples, encoder delay, block skip samples.
                if body_size >= 12 {
                    fact = Some((le32(chunk, 4), le32(chunk, 8)));
                }
            }
            // Sampler chunk: numSampleLoops at +0x1C, first loop start/end at
            // +0x2C/+0x30.
            b"smpl" if body_size >= 0x3C => {
                if le32(chunk, 0x1C) != 0 {
                    loop_begin = le32(chunk, 0x2C);
                    loop_end = le32(chunk, 0x30);
                }
            }
            _ => {}
        }
        // RIFF chunks are word aligned.
        let advance = chunk_size as u64 + (chunk_size & 1) as u64;
        if advance > (i32::MAX as u64).saturating_sub(body as u64) {
            break;
        }
        offset = body + advance as usize;
    }

    if !have_format || data_size == 0 {
        return false;
    }
    info.data_offset = data_offset;
    info.data_size = data_size;
    info.loop_begin_position = loop_begin;
    info.loop_end_position = loop_end;

    if info.waveform_type == WAVEFORM_TYPE_ATRAC9 {
        // The superframe is the NGS2 "audio frame"; one ATRAC9 frame is the
        // audio unit. The encoder delay is one frame.
        info.num_samples = if declared_samples != 0 {
            declared_samples
        } else if info.audio_unit_size == 0 || info.num_audio_unit_per_frame == 0 {
            0
        } else {
            let frame = info.audio_unit_size.wrapping_mul(info.num_audio_unit_per_frame);
            if frame == 0 {
                0
            } else {
                (data_size / frame)
                    .wrapping_mul(info.num_audio_unit_samples.wrapping_mul(info.num_audio_unit_per_frame))
            }
        };
        match fact {
            Some((delay, skip)) => {
                info.num_delay_samples = delay;
                info.block_skip_samples = skip;
            }
            None => info.num_delay_samples = info.num_audio_unit_samples,
        }
        finish_derived_fields(info);
        // The loop end is left unset for ATRAC9 streams.
        info.loop_end_position = loop_end;
        return true;
    } else if info.audio_unit_size != 0 {
        info.num_samples = data_size / info.audio_unit_size;
    }
    finish_derived_fields(info);
    true
}

/// WAVE "fmt " chunk: wFormatTag, nChannels, nSamplesPerSec,
/// nAvgBytesPerSec, nBlockAlign, wBitsPerSample, [cbSize, extension...].
fn try_parse_format_chunk(chunk: &[u8], info: &mut WaveformInfo) -> bool {
    if chunk.len() < 16 {
        return false;
    }
    let mut format_tag = le16(chunk, 0);
    let channels = le16(chunk, 2);
    let sample_rate = le32(chunk, 4);
    let block_align = le16(chunk, 12);
    let bits = le16(chunk, 14);

    info.num_channels = if channels == 0 { 1 } else { channels as u32 };
    info.sample_rate = if sample_rate == 0 { 48000 } else { sample_rate };
    info.num_audio_unit_per_frame = 1;
    info.num_audio_unit_samples = 1;

    let extension_size = if chunk.len() >= 18 { le16(chunk, 16) } else { 0 };
    if format_tag == 0xFFFE {
        if extension_size < 22 || chunk.len() < SUB_FORMAT_OFFSET + 16 {
            return false;
        }
        let sub = &chunk[SUB_FORMAT_OFFSET..SUB_FORMAT_OFFSET + 16];
        if sub == ATRAC9_SUB_FORMAT {
            return try_parse_atrac9_extension(chunk, info);
        }
        if sub == PCM_SUB_FORMAT {
            format_tag = 0x0001;
        } else if sub == FLOAT_SUB_FORMAT {
            format_tag = 0x0003;
        } else {
            return false;
        }
    }

    info.waveform_type = match (format_tag, bits) {
        (0x0001, 8) => WAVEFORM_TYPE_PCM_U8,
        (0x0001, 16) => WAVEFORM_TYPE_PCM_I16L,
        (0x0001, 24) => WAVEFORM_TYPE_PCM_I24L,
        (0x0001, 32) => WAVEFORM_TYPE_PCM_I32L,
        (0x0003, 32) => WAVEFORM_TYPE_PCM_F32L,
        (0x0003, 64) => WAVEFORM_TYPE_PCM_F64L,
        _ => 0,
    };
    if info.waveform_type == 0 {
        return false;
    }
    // PCM: the audio unit is one interleaved frame carrying one sample.
    let bytes_per_sample = bits as u32 / 8;
    info.audio_unit_size = if block_align != 0 {
        block_align as u32
    } else {
        bytes_per_sample.wrapping_mul(info.num_channels)
    };
    info.audio_unit_size != 0
}

/// AT9 fmt extension: SubFormat GUID (+0x18), u32 versionInfo (+0x28),
/// u8 configData[4] (+0x2C).
fn try_parse_atrac9_extension(chunk: &[u8], info: &mut WaveformInfo) -> bool {
    const CONFIG_OFFSET: usize = SUB_FORMAT_OFFSET + 16 + 4;
    if chunk.len() < CONFIG_OFFSET + 4 {
        return false;
    }
    let raw = &chunk[CONFIG_OFFSET..CONFIG_OFFSET + 4];
    let Ok(config) = Atrac9Config::new(raw) else {
        return false;
    };
    info.waveform_type = WAVEFORM_TYPE_ATRAC9;
    info.config_data = le32(raw, 0);
    info.num_channels = config.channel_count as u32;
    info.sample_rate = config.sample_rate as u32;
    info.audio_unit_size = config.frame_bytes as u32;
    info.num_audio_unit_samples = config.frame_samples as u32;
    info.num_audio_unit_per_frame = config.frames_per_superframe as u32;
    true
}

/// audioFrameSize = audioUnitSize * numAudioUnitPerFrame;
/// numAudioFrameSamples = numAudioUnitSamples * numAudioUnitPerFrame.
fn finish_derived_fields(info: &mut WaveformInfo) {
    if info.num_audio_unit_per_frame == 0 {
        info.num_audio_unit_per_frame = 1;
    }
    info.audio_frame_size = info.audio_unit_size.wrapping_mul(info.num_audio_unit_per_frame);
    info.num_audio_frame_samples =
        info.num_audio_unit_samples.wrapping_mul(info.num_audio_unit_per_frame);
    if info.loop_end_position == 0 && info.loop_begin_position == 0 && info.num_samples != 0 {
        info.loop_end_position = info.num_samples - 1;
    }
    info.num_blocks = 1;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn riff(fmt: &[u8], extra: &[(&[u8; 4], Vec<u8>)], data: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"RIFF\0\0\0\0WAVE");
        let mut chunk = |id: &[u8; 4], body: &[u8]| {
            v.extend_from_slice(id);
            v.extend_from_slice(&(body.len() as u32).to_le_bytes());
            v.extend_from_slice(body);
            if body.len() & 1 == 1 {
                v.push(0);
            }
        };
        chunk(b"fmt ", fmt);
        for (id, body) in extra {
            chunk(id, body);
        }
        chunk(b"data", data);
        v
    }

    pub fn pcm_fmt(tag: u16, channels: u16, rate: u32, bits: u16) -> Vec<u8> {
        let align = channels * bits / 8;
        let mut f = Vec::new();
        f.extend_from_slice(&tag.to_le_bytes());
        f.extend_from_slice(&channels.to_le_bytes());
        f.extend_from_slice(&rate.to_le_bytes());
        f.extend_from_slice(&(rate * align as u32).to_le_bytes());
        f.extend_from_slice(&align.to_le_bytes());
        f.extend_from_slice(&bits.to_le_bytes());
        f
    }

    #[test]
    fn ngs2_waveform_parse_riff_pcm() {
        let wav = riff(&pcm_fmt(1, 2, 44100, 16), &[], &[0u8; 400]);
        let info = try_parse(&wav).unwrap();
        assert_eq!(info.waveform_type, WAVEFORM_TYPE_PCM_I16L);
        assert_eq!(info.num_channels, 2);
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.data_offset, 12 + 8 + 16 + 8);
        assert_eq!(info.data_size, 400);
        assert_eq!(info.audio_unit_size, 4);
        assert_eq!(info.num_audio_unit_samples, 1);
        assert_eq!(info.num_audio_unit_per_frame, 1);
        assert_eq!(info.audio_frame_size, 4);
        assert_eq!(info.num_audio_frame_samples, 1);
        assert_eq!(info.num_samples, 100);
        assert_eq!(info.loop_begin_position, 0);
        assert_eq!(info.loop_end_position, 99);
        assert_eq!(info.num_blocks, 1);

        let img = write(&info);
        assert_eq!(le32(&img, 0x00), 0x12);
        assert_eq!(le32(&img, 0x04), 2);
        assert_eq!(le32(&img, 0x08), 44100);
        assert_eq!(le32(&img, 0x28), 100);
        assert_eq!(le32(&img, 0x44), 1);
        let b = BLOCK_ARRAY_OFFSET;
        assert_eq!(u64::from_le_bytes(img[b..b + 8].try_into().unwrap()), 44);
        assert_eq!(u64::from_le_bytes(img[b + 8..b + 16].try_into().unwrap()), 400);
        assert_eq!(le32(&img, b + 0x18), 100);
        assert_eq!(WAVEFORM_INFO_SIZE, 0xE8);

        // Float + smpl loop.
        let mut smpl = vec![0u8; 0x3C];
        smpl[0x1C..0x20].copy_from_slice(&1u32.to_le_bytes());
        smpl[0x2C..0x30].copy_from_slice(&10u32.to_le_bytes());
        smpl[0x30..0x34].copy_from_slice(&20u32.to_le_bytes());
        let wav = riff(&pcm_fmt(3, 1, 48000, 32), &[(b"smpl", smpl)], &[0u8; 64]);
        let info = try_parse(&wav).unwrap();
        assert_eq!(info.waveform_type, WAVEFORM_TYPE_PCM_F32L);
        assert_eq!((info.loop_begin_position, info.loop_end_position), (10, 20));
        assert_eq!(info.num_samples, 16);

        // Unknown format tag / missing data -> failure (but a known container).
        let wav = riff(&pcm_fmt(2, 1, 48000, 4), &[], &[0u8; 8]);
        assert!(try_parse(&wav).is_none());
        assert!(is_supported_container(&wav));
        assert!(!is_supported_container(b"OggS........"));
    }

    #[test]
    fn ngs2_waveform_parse_atrac9() {
        let word: u32 = (0xFE << 24) | (7 << 20) | (2 << 17) | ((0x100 - 1) << 5) | (2 << 3);
        let mut f = pcm_fmt(0xFFFE, 2, 48000, 0);
        f.extend_from_slice(&34u16.to_le_bytes()); // cbSize
        f.extend_from_slice(&[0u8; 6]); // samples, channel mask
        f.extend_from_slice(&ATRAC9_SUB_FORMAT);
        f.extend_from_slice(&1u32.to_le_bytes()); // versionInfo
        f.extend_from_slice(&word.to_be_bytes());
        f.extend_from_slice(&[0u8; 4]);
        let fact = 5000u32.to_le_bytes().to_vec();
        let wav = riff(&f, &[(b"fact", fact)], &[0u8; 0x400 * 2]);
        let info = try_parse(&wav).unwrap();
        assert_eq!(info.waveform_type, WAVEFORM_TYPE_ATRAC9);
        assert_eq!(info.config_data, u32::from_le_bytes(word.to_be_bytes()));
        assert_eq!(info.num_channels, 2);
        assert_eq!(info.audio_unit_size, 0x100);
        assert_eq!(info.num_audio_unit_samples, 256);
        assert_eq!(info.num_audio_unit_per_frame, 4);
        assert_eq!(info.audio_frame_size, 0x400);
        assert_eq!(info.num_audio_frame_samples, 1024);
        assert_eq!(info.num_samples, 5000);
        assert_eq!(info.num_delay_samples, 256);
    }

    #[test]
    fn ngs2_waveform_parse_vag() {
        let mut d = vec![0u8; vag::VAG_HEADER_SIZE + 64];
        d[..4].copy_from_slice(b"VAGp");
        d[0x0C..0x10].copy_from_slice(&48u32.to_be_bytes());
        d[0x10..0x14].copy_from_slice(&32000u32.to_be_bytes());
        d[0x1E] = 2;
        let info = try_parse(&d).unwrap();
        assert_eq!(info.waveform_type, WAVEFORM_TYPE_VAG);
        assert_eq!(info.num_channels, 2);
        assert_eq!(info.sample_rate, 32000);
        assert_eq!(info.data_offset, 0x30);
        assert_eq!(info.data_size, 48);
        assert_eq!(info.audio_unit_size, 32);
        assert_eq!(info.num_audio_unit_samples, 28);
        assert_eq!(info.num_samples, 28);
        assert_eq!(info.loop_end_position, 27);
        assert_eq!(info.audio_frame_size, 32);
        assert_eq!(info.num_audio_frame_samples, 28);
        assert_eq!(info.num_blocks, 1);

        // Mono default, declared size larger than available.
        d[0x1E] = 0;
        d[0x0C..0x10].copy_from_slice(&0x1000u32.to_be_bytes());
        d[0x10..0x14].copy_from_slice(&0u32.to_be_bytes());
        let info = try_parse(&d).unwrap();
        assert_eq!((info.num_channels, info.sample_rate, info.data_size), (1, 48000, 64));
        assert_eq!(info.num_samples, 4 * 28);
    }
}
