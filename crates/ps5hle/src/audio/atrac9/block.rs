// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Per-frame decoder state: frames, blocks (substreams) and channels.

use super::config::{Atrac9Config, BlockType};
use super::mdct::Mdct;
use super::rng::Atrac9Rng;
use super::tables;

pub(super) struct Channel {
    pub channel_index: i32,
    pub mdct: Mdct,

    pub pcm: [f64; 256],
    pub spectra: [f64; 256],

    pub coded_quant_units: i32,
    pub scale_factor_coding_mode: i32,
    pub scale_factors: [i32; 31],
    pub scale_factors_prev: [i32; 31],

    pub precisions: [i32; 30],
    pub precisions_fine: [i32; 30],
    pub precision_mask: [i32; 30],

    pub spectra_values_buffer: [i32; 16],
    pub codebook_set: [i32; 30],

    pub quantized_spectra: [i32; 256],
    pub quantized_spectra_fine: [i32; 256],

    pub bex_mode: i32,
    pub bex_value_count: i32,
    pub bex_values: [i32; 4],
    pub bex_scales: [f64; 6],
    pub rng: Option<Atrac9Rng>,
}

impl Channel {
    fn new(config: &Atrac9Config, channel_index: i32) -> Self {
        let power = config.frame_samples_power as usize;
        let window: &'static [f64] = &tables::tables().imdct_window[power - 6];
        Self {
            channel_index,
            mdct: Mdct::new(power, window, 1.0),
            pcm: [0.0; 256],
            spectra: [0.0; 256],
            coded_quant_units: 0,
            scale_factor_coding_mode: 0,
            scale_factors: [0; 31],
            scale_factors_prev: [0; 31],
            precisions: [0; 30],
            precisions_fine: [0; 30],
            precision_mask: [0; 30],
            spectra_values_buffer: [0; 16],
            codebook_set: [0; 30],
            quantized_spectra: [0; 256],
            quantized_spectra_fine: [0; 256],
            bex_mode: 0,
            bex_value_count: 0,
            bex_values: [0; 4],
            bex_scales: [0.0; 6],
            rng: None,
        }
    }
}

pub(super) struct Block {
    pub block_type: BlockType,
    pub channels: Vec<Channel>,

    pub first_in_superframe: bool,
    pub reuse_band_params: bool,
    pub superframe_bytes_used: i32,

    pub band_count: i32,
    pub stereo_band: i32,
    pub extension_band: i32,
    pub quantization_unit_count: i32,
    pub stereo_quantization_unit: i32,
    pub extension_unit: i32,
    pub quantization_units_prev: i32,

    pub gradient: [i32; 31],
    pub gradient_mode: i32,
    pub gradient_start_unit: i32,
    pub gradient_start_value: i32,
    pub gradient_end_unit: i32,
    pub gradient_end_value: i32,
    pub gradient_boundary: i32,

    pub primary_channel_index: i32,
    pub joint_stereo_signs: [i32; 30],
    pub has_joint_stereo_signs: bool,

    pub band_extension_enabled: bool,
    pub has_extension_data: bool,
    pub bex_data_length: i32,
    pub bex_mode: i32,
}

impl Block {
    fn new(config: &Atrac9Config, block_type: BlockType) -> Self {
        let channels = (0..block_type.channel_count()).map(|i| Channel::new(config, i)).collect();
        Self {
            block_type,
            channels,
            first_in_superframe: false,
            reuse_band_params: false,
            superframe_bytes_used: 0,
            band_count: 0,
            stereo_band: 0,
            extension_band: 0,
            quantization_unit_count: 0,
            stereo_quantization_unit: 0,
            extension_unit: 0,
            quantization_units_prev: 0,
            gradient: [0; 31],
            gradient_mode: 0,
            gradient_start_unit: 0,
            gradient_start_value: 0,
            gradient_end_unit: 0,
            gradient_end_value: 0,
            gradient_boundary: 0,
            primary_channel_index: 0,
            joint_stereo_signs: [0; 30],
            has_joint_stereo_signs: false,
            band_extension_enabled: false,
            has_extension_data: false,
            bex_data_length: 0,
            bex_mode: 0,
        }
    }

    pub fn is_primary(&self, channel_index: i32) -> bool {
        self.primary_channel_index == channel_index
    }

    pub fn primary_channel(&self) -> usize {
        if self.primary_channel_index == 0 {
            0
        } else {
            1
        }
    }

    pub fn secondary_channel(&self) -> usize {
        if self.primary_channel_index == 0 {
            1
        } else {
            0
        }
    }

    pub fn update_coded_units(&mut self, ch: usize) {
        let index = self.channels[ch].channel_index;
        self.channels[ch].coded_quant_units = if self.is_primary(index) {
            self.quantization_unit_count
        } else {
            self.stereo_quantization_unit
        };
    }
}

pub(super) struct Frame {
    pub frame_index: i32,
    pub blocks: Vec<Block>,
}

impl Frame {
    pub fn new(config: &Atrac9Config) -> Self {
        let blocks = config
            .channel_config
            .block_types
            .iter()
            .map(|&t| Block::new(config, t))
            .collect();
        Self { frame_index: 0, blocks }
    }
}
