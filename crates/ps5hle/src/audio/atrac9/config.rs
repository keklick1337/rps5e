// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! ATRAC9 stream configuration (the 4-byte config word) and channel layout.

use super::bit_reader::BitReader;
use super::tables;
use super::Atrac9Error;

/// An ATRAC9 block (substream) type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockType {
    /// Mono block.
    Mono = 0,
    /// Stereo block.
    Stereo = 1,
    /// Low-frequency effects block.
    Lfe = 2,
}

impl BlockType {
    pub fn channel_count(self) -> i32 {
        match self {
            BlockType::Mono => 1,
            BlockType::Stereo => 2,
            BlockType::Lfe => 1,
        }
    }
}

/// Channel mapping of an ATRAC9 stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelConfig {
    /// Number of blocks (substreams).
    pub block_count: i32,
    /// Type of each block.
    pub block_types: Vec<BlockType>,
    /// Total number of channels.
    pub channel_count: i32,
}

impl ChannelConfig {
    pub fn new(block_types: &[BlockType]) -> Self {
        Self {
            block_count: block_types.len() as i32,
            block_types: block_types.to_vec(),
            channel_count: block_types.iter().map(|t| t.channel_count()).sum(),
        }
    }

    pub fn independent_mono(channel_count: i32) -> Self {
        Self::new(&vec![BlockType::Mono; channel_count.max(0) as usize])
    }
}

const STANDARD_HEADER: i32 = 0xFE;
const INDEPENDENT_MONO_HEADER: i32 = 0x30;

/// Parameters derived from the 4-byte ATRAC9 configuration data.
#[derive(Clone, Debug)]
pub struct Atrac9Config {
    /// The raw 4-byte configuration data.
    pub config_data: [u8; 4],
    /// 4-bit sample rate index.
    pub sample_rate_index: i32,
    /// 3-bit channel mapping index (0 for independent-mono streams).
    pub channel_config_index: i32,
    /// Average size of one frame in bytes.
    pub frame_bytes: i32,
    /// 2-bit value: log2 of the frames per superframe.
    pub superframe_index: i32,
    pub channel_config: ChannelConfig,
    pub channel_count: i32,
    pub sample_rate: i32,
    /// `sample_rate_index >= 8`.
    pub high_sample_rate: bool,
    pub frames_per_superframe: i32,
    /// `frame_samples == 1 << frame_samples_power`.
    pub frame_samples_power: i32,
    pub frame_samples: i32,
    pub superframe_bytes: i32,
    pub superframe_samples: i32,
    /// Non-zero for streams whose config header is 0x30: every channel is an
    /// independent mono stream with this many bytes per superframe, and the
    /// channels are interleaved frame by frame.
    pub independent_block_superframe_bytes: i32,
}

impl Atrac9Config {
    /// Parses ATRAC9 configuration data and derives the stream parameters.
    pub fn new(config_data: &[u8]) -> Result<Atrac9Config, Atrac9Error> {
        let config_data: [u8; 4] = config_data
            .try_into()
            .map_err(|_| Atrac9Error::InvalidConfig("Config data must be 4 bytes long".into()))?;

        let mut reader = BitReader::new(&config_data);
        let header = reader.read_int(8);
        let sample_rate_index = reader.read_int(4);
        let mut channel_config_index = 0;
        let mut independent_channels = 0;
        if header == INDEPENDENT_MONO_HEADER {
            // Multichannel (ambisonic) streams replace the 3-bit channel
            // mapping with a 6-bit channel count.
            independent_channels = reader.read_int(6) + 1;
        } else {
            channel_config_index = reader.read_int(3);
        }
        let validation_bit = reader.read_int(1);
        let c = reader.read_int(11) + 1;
        let superframe_index = reader.read_int(2);

        if (header != STANDARD_HEADER && header != INDEPENDENT_MONO_HEADER) || validation_bit != 0 {
            return Err(Atrac9Error::InvalidConfig("ATRAC9 Config Data is invalid".into()));
        }

        let (channel_config, independent_block_superframe_bytes, frame_bytes) = if independent_channels > 0 {
            // The frame size field holds one channel's share; the frame as a
            // whole carries one mono block per channel.
            (
                ChannelConfig::independent_mono(independent_channels),
                c << superframe_index,
                c * independent_channels,
            )
        } else {
            // Mappings 6 and 7 are the vibration layouts; their blocks are
            // coded as ordinary mono / dual mono.
            let layout_index = if channel_config_index >= 6 { channel_config_index - 6 } else { channel_config_index };
            let layout = tables::CHANNEL_CONFIGS
                .get(layout_index as usize)
                .ok_or_else(|| Atrac9Error::InvalidConfig("ATRAC9 Config Data is invalid".into()))?;
            (ChannelConfig::new(layout), 0, c)
        };

        let frames_per_superframe = 1 << superframe_index;
        let superframe_bytes = frame_bytes << superframe_index;
        let channel_count = channel_config.channel_count;
        let sample_rate = tables::SAMPLE_RATES[sample_rate_index as usize];
        let high_sample_rate = sample_rate_index > 7;
        let frame_samples_power =
            tables::SAMPLING_RATE_INDEX_TO_FRAME_SAMPLES_POWER[sample_rate_index as usize] as i32;
        let frame_samples = 1 << frame_samples_power;
        let superframe_samples = frame_samples * frames_per_superframe;

        Ok(Atrac9Config {
            config_data,
            sample_rate_index,
            channel_config_index,
            frame_bytes,
            superframe_index,
            channel_config,
            channel_count,
            sample_rate,
            high_sample_rate,
            frames_per_superframe,
            frame_samples_power,
            frame_samples,
            superframe_bytes,
            superframe_samples,
            independent_block_superframe_bytes,
        })
    }
}
