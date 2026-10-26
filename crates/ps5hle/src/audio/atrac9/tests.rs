// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only
//
// Unit tests of the `ps5hle::audio::atrac9` module.

use super::bit_reader::{bit_reverse32, sign_extend32, BitReader};
use super::mdct::Mdct;
use super::tables;
use super::*;

/// MSB-first bit writer for building test bitstreams.
#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    bits: usize,
}

impl BitWriter {
    fn write(&mut self, value: u32, count: u32) {
        for i in (0..count).rev() {
            if self.bits % 8 == 0 {
                self.bytes.push(0);
            }
            if (value >> i) & 1 != 0 {
                let last = self.bytes.len() - 1;
                self.bytes[last] |= 0x80 >> (self.bits % 8);
            }
            self.bits += 1;
        }
    }

    fn align(&mut self) {
        self.bits = self.bytes.len() * 8;
    }
}

/// Builds a 4-byte standard config word.
pub(crate) fn config_word(sample_rate_index: u32, channel_config: u32, frame_bytes: u32, superframe_index: u32) -> [u8; 4] {
    let mut w = BitWriter::default();
    w.write(0xFE, 8);
    w.write(sample_rate_index, 4);
    w.write(channel_config, 3);
    w.write(0, 1);
    w.write(frame_bytes - 1, 11);
    w.write(superframe_index, 2);
    w.write(0, 3);
    [w.bytes[0], w.bytes[1], w.bytes[2], w.bytes[3]]
}

/// Scale factor used by the synthetic test blocks: precision 15 (16-bit raw
/// coefficients) plus 10 fine-precision bits (11-bit fine values).
const TEST_SCALE_FACTOR: u32 = 25;
/// Bytes per frame of the synthetic mono test stream.
const TEST_FRAME_BYTES: usize = 128;

/// Writes one 48 kHz mono block whose 10 quant units (24 coefficients) are
/// coded raw with flat scale factors and a zero gradient.
fn write_mono_block(w: &mut BitWriter, first: bool, coefficients: &[i32; 24]) {
    w.write(if first { 0 } else { 1 }, 1); // !first_in_superframe
    w.write(0, 1); // reuse_band_params
    w.write(0, 4); // band_count = 3 -> 10 quant units
    w.write(0, 1); // band extension disabled
    w.write(0, 2); // gradient mode 0
    w.write(1, 6); // gradient start unit
    w.write(0, 6); // gradient end unit - 1
    w.write(0, 5); // gradient start value
    w.write(0, 5); // gradient end value
    w.write(0, 4); // gradient boundary
    w.write(0, 1); // no extension data
    w.write(1, 2); // scale factor mode 1 (CLC offset)
    w.write(3, 2); // bit length 5 (no base value)
    for _ in 0..10 {
        w.write(TEST_SCALE_FACTOR, 5);
    }
    for &c in coefficients {
        w.write((c as u32) & 0xFFFF, 16);
    }
    for _ in coefficients {
        w.write(0, 11); // fine values
    }
    w.align();
}

pub(crate) fn build_mono_superframe(coefficients: &[i32; 24]) -> Vec<u8> {
    let mut w = BitWriter::default();
    for f in 0..4 {
        write_mono_block(&mut w, f == 0, coefficients);
    }
    let mut data = w.bytes;
    assert!(data.len() <= 4 * TEST_FRAME_BYTES, "test superframe does not fit");
    data.resize(4 * TEST_FRAME_BYTES, 0);
    data
}

pub(crate) fn mono_config() -> [u8; 4] {
    config_word(7, 0, TEST_FRAME_BYTES as u32, 2)
}

fn pcm_buffers(config: &Atrac9Config) -> Vec<Vec<i16>> {
    vec![vec![0i16; config.superframe_samples as usize]; config.channel_count as usize]
}

/// Tiny deterministic xorshift for fuzz-style tests.
pub(crate) struct TestRng(pub(crate) u64);
impl TestRng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[test]
fn config_48k_stereo() {
    let config = Atrac9Config::new(&[0xFE, 0x74, 0x0F, 0xF0]).unwrap();
    assert_eq!(config.sample_rate_index, 7);
    assert_eq!(config.channel_config_index, 2);
    assert_eq!(config.channel_count, 2);
    assert_eq!(config.sample_rate, 48000);
    assert!(!config.high_sample_rate);
    assert_eq!(config.frame_bytes, 128);
    assert_eq!(config.superframe_index, 2);
    assert_eq!(config.frames_per_superframe, 4);
    assert_eq!(config.frame_samples_power, 8);
    assert_eq!(config.frame_samples, 256);
    assert_eq!(config.superframe_samples, 1024);
    assert_eq!(config.superframe_bytes, 512);
    assert_eq!(config.independent_block_superframe_bytes, 0);
    assert_eq!(config.channel_config.block_types, vec![BlockType::Stereo]);
    assert_eq!(config.channel_config.block_count, 1);
}

#[test]
fn config_builder_matches_known_word() {
    assert_eq!(config_word(7, 2, 128, 2), [0xFE, 0x74, 0x0F, 0xF0]);
}

#[test]
fn config_surround_and_low_rate() {
    // 5.1: Stereo + Mono + LFE + Stereo.
    let config = Atrac9Config::new(&config_word(7, 3, 256, 2)).unwrap();
    assert_eq!(config.channel_count, 6);
    assert_eq!(config.channel_config.block_count, 4);
    assert_eq!(config.superframe_bytes, 1024);

    let config = Atrac9Config::new(&config_word(0, 0, 64, 0)).unwrap();
    assert_eq!(config.sample_rate, 11025);
    assert_eq!(config.frame_samples, 64);
    assert_eq!(config.superframe_samples, 64);
    assert_eq!(config.channel_count, 1);
}

#[test]
fn config_independent_mono() {
    // Header 0x30, rate index 7, 6-bit channel count (4 -> 4 channels),
    // validation 0, per-channel frame bytes 100, superframe index 2.
    let mut w = BitWriter::default();
    w.write(0x30, 8);
    w.write(7, 4);
    w.write(3, 6);
    w.write(0, 1);
    w.write(99, 11);
    w.write(2, 2);
    let config = Atrac9Config::new(&w.bytes[..4]).unwrap();
    assert_eq!(config.channel_count, 4);
    assert_eq!(config.channel_config.block_count, 4);
    assert_eq!(config.frame_bytes, 400);
    assert_eq!(config.independent_block_superframe_bytes, 400);
    assert_eq!(config.superframe_bytes, 1600);
}

#[test]
fn config_rejects_invalid() {
    assert!(matches!(Atrac9Config::new(&[0xFE, 0x74, 0x0F]), Err(Atrac9Error::InvalidConfig(_))));
    assert!(matches!(Atrac9Config::new(&[0xFD, 0x74, 0x0F, 0xF0]), Err(Atrac9Error::InvalidConfig(_))));
    // Validation bit set.
    assert!(matches!(Atrac9Config::new(&[0xFE, 0x75, 0x0F, 0xF0]), Err(Atrac9Error::InvalidConfig(_))));
}

#[test]
fn config_vibration_layouts_decode_as_mono() {
    // Mappings 6 and 7 carry mono / dual-mono blocks; the raw word is kept.
    let c = Atrac9Config::new(&config_word(7, 6, 128, 2)).unwrap();
    assert_eq!((c.channel_config_index, c.channel_count), (6, 1));
    assert_eq!(c.config_data, config_word(7, 6, 128, 2));
    let c = Atrac9Config::new(&config_word(7, 7, 128, 2)).unwrap();
    assert_eq!((c.channel_config_index, c.channel_count), (7, 2));
}

#[test]
fn decode_rejects_small_buffers() {
    let mut decoder = Atrac9Decoder::new(&[0xFE, 0x74, 0x0F, 0xF0]).unwrap();
    let mut pcm = pcm_buffers(decoder.config());
    assert_eq!(decoder.decode(&[0u8; 511], &mut pcm), Err(Atrac9Error::BufferTooSmall));
    let mut short = vec![vec![0i16; 1023]; 2];
    assert_eq!(decoder.decode(&[0u8; 512], &mut short), Err(Atrac9Error::BufferTooSmall));
    let mut one = vec![vec![0i16; 1024]; 1];
    assert_eq!(decoder.decode(&[0u8; 512], &mut one), Err(Atrac9Error::BufferTooSmall));
}

#[test]
fn decode_all_zero_superframe_is_invalid() {
    // An all-zero block has gradient start unit 0, which is rejected.
    let mut decoder = Atrac9Decoder::new(&[0xFE, 0x74, 0x0F, 0xF0]).unwrap();
    let mut pcm = pcm_buffers(decoder.config());
    let result = decoder.decode(&[0u8; 512], &mut pcm);
    assert!(matches!(result, Err(Atrac9Error::InvalidData(_))));
    assert_eq!(pcm.len(), 2);
    assert!(pcm.iter().all(|c| c.len() == 1024));
}

#[test]
fn decode_silent_superframe() {
    let config = mono_config();
    let mut decoder = Atrac9Decoder::new(&config).unwrap();
    assert_eq!(decoder.config().superframe_samples, 1024);
    let data = build_mono_superframe(&[0; 24]);
    let mut pcm = vec![vec![0x55i16; 1024]];
    decoder.decode(&data, &mut pcm).unwrap();
    assert!(pcm[0].iter().all(|&s| s == 0));
}

#[test]
fn decode_tone_is_deterministic_and_nonzero() {
    let config = mono_config();
    let mut coefficients = [0i32; 24];
    coefficients[3] = 20000;
    coefficients[20] = -9000;
    let data = build_mono_superframe(&coefficients);

    let mut first = vec![vec![0i16; 1024]];
    let mut second = vec![vec![0i16; 1024]];
    Atrac9Decoder::new(&config).unwrap().decode(&data, &mut first).unwrap();
    Atrac9Decoder::new(&config).unwrap().decode(&data, &mut second).unwrap();
    assert_eq!(first, second);
    assert!(first[0].iter().any(|&s| s != 0));

    // Overlap-add: the decoder carries state between superframes.
    let mut decoder = Atrac9Decoder::new(&config).unwrap();
    let mut a = vec![vec![0i16; 1024]];
    let mut b = vec![vec![0i16; 1024]];
    decoder.decode(&data, &mut a).unwrap();
    decoder.decode(&data, &mut b).unwrap();
    assert_eq!(a, first);
}

#[test]
fn random_bytes_never_panic() {
    let configs = [
        [0xFE, 0x74, 0x0F, 0xF0],
        mono_config(),
        config_word(7, 3, 256, 2),
        config_word(12, 2, 200, 1),
        config_word(15, 5, 300, 3),
        config_word(0, 1, 40, 0),
    ];
    let mut rng = TestRng(0x9E37_79B9_7F4A_7C15);
    for config in configs {
        let mut decoder = Atrac9Decoder::new(&config).unwrap();
        let mut pcm = pcm_buffers(decoder.config());
        let len = decoder.config().superframe_bytes as usize;
        for round in 0..300 {
            let mut data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            // Make the block header plausible half of the time so decoding
            // gets past the first checks.
            if round % 2 == 0 {
                data[0] &= 0x3F;
            }
            let _ = decoder.decode(&data, &mut pcm);
        }
    }

    // Independent mono layout.
    let mut w = BitWriter::default();
    w.write(0x30, 8);
    w.write(7, 4);
    w.write(3, 6);
    w.write(0, 1);
    w.write(99, 11);
    w.write(2, 2);
    let mut decoder = Atrac9Decoder::new(&w.bytes[..4]).unwrap();
    let mut pcm = pcm_buffers(decoder.config());
    for _ in 0..300 {
        let data: Vec<u8> = (0..1600).map(|_| rng.next() as u8 & 0x3F).collect();
        let _ = decoder.decode(&data, &mut pcm);
    }
}

#[test]
fn bit_flipped_valid_stream_never_panics() {
    let config = mono_config();
    let mut coefficients = [0i32; 24];
    coefficients[5] = 1234;
    let clean = build_mono_superframe(&coefficients);
    let mut decoder = Atrac9Decoder::new(&config).unwrap();
    let mut pcm = vec![vec![0i16; 1024]];
    let mut rng = TestRng(12345);
    for _ in 0..2000 {
        let mut data = clean.clone();
        for _ in 0..(rng.next() % 4 + 1) {
            let bit = (rng.next() % (data.len() as u64 * 8)) as usize;
            data[bit / 8] ^= 0x80 >> (bit % 8);
        }
        let _ = decoder.decode(&data, &mut pcm);
    }
}

#[test]
fn bit_reader_semantics() {
    let data = [0b1010_1100, 0xFF, 0x00, 0x80];
    let mut r = BitReader::new(&data);
    assert_eq!(r.read_int(1), 1);
    assert_eq!(r.read_int(3), 0b010);
    assert_eq!(r.read_signed_int(4), sign_extend32(0b1100, 4));
    assert_eq!(r.read_int(8), 0xFF);
    // Reading past the end pads with zeros and never panics.
    let mut r = BitReader::new(&data);
    r.position = 30;
    assert_eq!(r.read_int(8), 0);
    assert_eq!(r.read_int(8), 0);
    assert_eq!(sign_extend32(0b11, 2), -1);
    assert_eq!(sign_extend32(5, 0), 5);
    assert_eq!(bit_reverse32(1, 8), 0x80);
    assert_eq!(bit_reverse32(0x1234_5678, 0), 0, "zero bits reverse to zero");
    assert_eq!(bit_reverse32(1, 32), i32::MIN);
    assert_eq!(sign_extend32(i32::MAX, 1), -1);
}

#[test]
fn imdct_of_zero_spectrum_is_zero() {
    for bits in 6..=8usize {
        let window: &'static [f64] = &tables::tables().imdct_window[bits - 6];
        let mut mdct = Mdct::new(bits, window, 1.0);
        let input = vec![0.0f64; 1 << bits];
        let mut output = vec![1.0f64; 1 << bits];
        assert!(mdct.run_imdct(&input, &mut output));
        assert!(output.iter().all(|&v| v == 0.0));
        assert!(!mdct.run_imdct(&input[..1], &mut output));
    }
}

#[test]
fn imdct_impulse_has_energy() {
    let window: &'static [f64] = &tables::tables().imdct_window[2];
    let mut mdct = Mdct::new(8, window, 1.0);
    let mut input = vec![0.0f64; 256];
    input[4] = 1000.0;
    let mut output = vec![0.0f64; 256];
    mdct.run_imdct(&input, &mut output);
    assert!(output.iter().any(|v| v.abs() > 1.0));
    assert!(output.iter().all(|v| v.is_finite()));
}

/// Reference comparison harness: decodes a raw dump (4 config bytes followed
/// by superframes) and writes, per superframe, a status byte (0 = ok,
/// 1 = error) followed by planar little-endian i16 PCM.
/// Run with `RPS5E_AT9_RAW=<in> RPS5E_AT9_OUT=<out> cargo test -- --ignored`.
#[test]
#[ignore]
fn reference_dump() {
    let (Ok(input), Ok(output)) = (std::env::var("RPS5E_AT9_RAW"), std::env::var("RPS5E_AT9_OUT")) else {
        return;
    };
    let raw = std::fs::read(input).unwrap();
    let mut decoder = Atrac9Decoder::new(&raw[..4]).unwrap();
    let mut pcm = pcm_buffers(decoder.config());
    let sf_bytes = decoder.config().superframe_bytes as usize;
    let mut out = Vec::new();
    for chunk in raw[4..].chunks_exact(sf_bytes) {
        let status = decoder.decode(chunk, &mut pcm).is_err() as u8;
        out.push(status);
        for channel in &pcm {
            for s in channel {
                out.extend_from_slice(&s.to_le_bytes());
            }
        }
    }
    std::fs::write(output, out).unwrap();
}
