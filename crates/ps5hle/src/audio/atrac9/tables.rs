// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Static ATRAC9 tables and generated lookup tables.

use std::f64::consts::PI;
use std::sync::OnceLock;

use super::bit_allocation::generate_gradient_curves;
use super::config::BlockType;

pub(super) fn max_huff_precision(high_sample_rate: bool) -> i32 {
    if high_sample_rate {
        1
    } else {
        7
    }
}

pub(super) fn min_band_count(high_sample_rate: bool) -> i32 {
    if high_sample_rate {
        1
    } else {
        3
    }
}

pub(super) fn max_extension_band(high_sample_rate: bool) -> i32 {
    if high_sample_rate {
        16
    } else {
        18
    }
}

pub(super) const SAMPLE_RATES: [i32; 16] = [
    11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 44100, 48000, 64000, 88200, 96000,
    128000, 176400, 192000,
];

pub(super) const SAMPLING_RATE_INDEX_TO_FRAME_SAMPLES_POWER: [u8; 16] =
    [6, 6, 7, 7, 7, 8, 8, 8, 6, 6, 7, 7, 7, 8, 8, 8];

/// Indexed by sampling rate index.
pub(super) const MAX_BAND_COUNT: [u8; 16] = [8, 8, 12, 12, 12, 18, 18, 18, 8, 8, 12, 12, 12, 16, 16, 16];

pub(super) const BAND_TO_QUANT_UNIT_COUNT: [u8; 19] =
    [0, 4, 8, 10, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30];

pub(super) const QUANT_UNIT_TO_COEFF_COUNT: [u8; 30] = [
    2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16,
];

pub(super) const QUANT_UNIT_TO_COEFF_INDEX: [i16; 31] = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 112, 128, 144,
    160, 176, 192, 208, 224, 240, 256,
];

pub(super) const QUANT_UNIT_TO_CODEBOOK_INDEX: [u8; 30] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
];

/// Block layouts selected by the 3-bit channel configuration index.
pub(super) const CHANNEL_CONFIGS: [&[BlockType]; 6] = [
    &[BlockType::Mono],
    &[BlockType::Mono, BlockType::Mono],
    &[BlockType::Stereo],
    &[BlockType::Stereo, BlockType::Mono, BlockType::Lfe, BlockType::Stereo],
    &[BlockType::Stereo, BlockType::Mono, BlockType::Lfe, BlockType::Stereo, BlockType::Stereo],
    &[BlockType::Stereo, BlockType::Stereo],
];

pub(super) struct GeneratedTables {
    /// IMDCT windows for frame sizes 2^6, 2^7 and 2^8.
    pub imdct_window: [Vec<f64>; 3],
    pub spectrum_scale: [f64; 32],
    pub quantizer_step_size: [f64; 16],
    pub quantizer_fine_step_size: [f64; 16],
    pub gradient_curves: Vec<Vec<u8>>,
}

pub(super) fn tables() -> &'static GeneratedTables {
    static TABLES: OnceLock<GeneratedTables> = OnceLock::new();
    TABLES.get_or_init(|| GeneratedTables {
        imdct_window: [generate_imdct_window(6), generate_imdct_window(7), generate_imdct_window(8)],
        spectrum_scale: std::array::from_fn(|x| spectrum_scale_function(x as i32)),
        quantizer_step_size: std::array::from_fn(|x| quantizer_step_size_function(x as i32)),
        quantizer_fine_step_size: std::array::from_fn(|x| quantizer_fine_step_size_function(x as i32)),
        gradient_curves: generate_gradient_curves(),
    })
}

fn quantizer_step_size_function(x: i32) -> f64 {
    2.0 / ((1i32 << (x + 1)) - 1) as f64
}

fn quantizer_fine_step_size_function(x: i32) -> f64 {
    quantizer_step_size_function(x) / u16::MAX as f64
}

fn spectrum_scale_function(x: i32) -> f64 {
    2.0f64.powf((x - 15) as f64)
}

fn generate_imdct_window(frame_size_power: i32) -> Vec<f64> {
    let frame_size = 1usize << frame_size_power;
    let a1 = generate_mdct_window(frame_size_power);
    (0..frame_size)
        .map(|i| {
            let j = frame_size - 1 - i;
            a1[i] / (a1[j] * a1[j] + a1[i] * a1[i])
        })
        .collect()
}

fn generate_mdct_window(frame_size_power: i32) -> Vec<f64> {
    let frame_size = 1usize << frame_size_power;
    (0..frame_size)
        .map(|i| ((((i as f64 + 0.5) / frame_size as f64 - 0.5) * PI).sin() + 1.0) * 0.5)
        .collect()
}
