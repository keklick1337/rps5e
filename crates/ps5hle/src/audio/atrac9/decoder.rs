// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Top-level ATRAC9 superframe decoder producing planar 16-bit PCM.

use super::bit_reader::{clamp16, BitReader};
use super::block::{Block, Frame};
use super::config::Atrac9Config;
use super::{band_extension, quantization, unpack, Atrac9Error};

/// Decodes an ATRAC9 stream into 16-bit PCM.
pub struct Atrac9Decoder {
    config: Atrac9Config,
    frame: Frame,
    /// Index of the next frame within the superframe (per-frame decoding).
    frame_cursor: i32,
}

/// Output PCM sample encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmEncoding {
    S16 = 0,
    S32 = 1,
    Float = 2,
}

impl PcmEncoding {
    /// Bytes per sample per channel.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            PcmEncoding::S16 => 2,
            PcmEncoding::S32 | PcmEncoding::Float => 4,
        }
    }
}

impl std::fmt::Debug for Atrac9Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Atrac9Decoder").field("config", &self.config).finish_non_exhaustive()
    }
}

impl Atrac9Decoder {
    /// Creates a decoder for the stream described by the 4-byte `config_data`.
    pub fn new(config_data: &[u8]) -> Result<Self, Atrac9Error> {
        let config = Atrac9Config::new(config_data)?;
        let frame = Frame::new(&config);
        Ok(Self { config, frame, frame_cursor: 0 })
    }

    /// The configuration of the current stream.
    pub fn config(&self) -> &Atrac9Config {
        &self.config
    }

    /// Decode one superframe into planar PCM: `pcm_out[channel][sample]`, each
    /// channel buffer at least `superframe_samples` long. `atrac9_data` must
    /// hold at least `superframe_bytes` bytes.
    ///
    /// On `Err(InvalidData)` the PCM of frames decoded before the error has
    /// already been written, matching the reference behaviour.
    pub fn decode(&mut self, atrac9_data: &[u8], pcm_out: &mut [Vec<i16>]) -> Result<(), Atrac9Error> {
        self.validate_decode_buffers(atrac9_data, pcm_out)?;
        let mut reader = BitReader::new(atrac9_data);
        self.decode_superframe(&mut reader, pcm_out)
    }

    /// Decode the next frame of the current superframe from `data`, which
    /// starts at the frame's first byte. Returns the bytes the frame used
    /// (byte aligned). The PCM stays in the decoder until the next frame;
    /// read it with [`Atrac9Decoder::write_frame_pcm`].
    pub fn decode_next_frame(&mut self, data: &[u8]) -> Result<usize, Atrac9Error> {
        let mut reader = BitReader::new(data);
        let index = self.frame_cursor;
        self.frame.frame_index = index;
        let r = decode_frame(&mut reader, &mut self.frame, &self.config);
        // The frame counter advances even on errors, as the bitstream does.
        self.frame_cursor = (index + 1) % self.config.frames_per_superframe.max(1);
        r?;
        reader.align_position(8);
        Ok((reader.position.max(0) / 8) as usize)
    }

    /// Restart frame numbering at the start of a superframe.
    pub fn restart_superframe(&mut self) {
        self.frame_cursor = 0;
    }

    /// Write `count` samples of the last decoded frame, starting at sample
    /// `skip`, as interleaved PCM in `encoding` (bytes per sample:
    /// 2 / 4 / 4). Signed 16-bit and 32-bit samples are `floor(x + 0.5)`
    /// (16-bit saturated), float samples are `x / 32767`.
    pub fn write_frame_pcm(&self, out: &mut [u8], skip: usize, count: usize, encoding: PcmEncoding) {
        let bps = encoding.bytes_per_sample();
        let nch = self.config.channel_count.max(0) as usize;
        let stride = nch * bps;
        let end = (skip + count).min(self.config.frame_samples.max(0) as usize).min(256);
        let channels = self.frame.blocks.iter().flat_map(|b| b.channels.iter().map(|c| &c.pcm));
        for (ch, pcm) in channels.enumerate().take(nch) {
            let src = &pcm[skip.min(end)..end];
            let dst = out.chunks_exact_mut(stride).take(src.len());
            match encoding {
                PcmEncoding::S16 => {
                    for (d, &x) in dst.zip(src) {
                        let v = clamp16((x + 0.5).floor() as i32);
                        d[ch * 2..ch * 2 + 2].copy_from_slice(&v.to_le_bytes());
                    }
                }
                PcmEncoding::S32 => {
                    for (d, &x) in dst.zip(src) {
                        let v = (x + 0.5).floor() as i32;
                        d[ch * 4..ch * 4 + 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
                PcmEncoding::Float => {
                    for (d, &x) in dst.zip(src) {
                        let v = (x / i16::MAX as f64) as f32;
                        d[ch * 4..ch * 4 + 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
    }

    fn validate_decode_buffers(&self, atrac9_data: &[u8], pcm_out: &[Vec<i16>]) -> Result<(), Atrac9Error> {
        let config = &self.config;
        if atrac9_data.len() < config.superframe_bytes as usize {
            return Err(Atrac9Error::BufferTooSmall);
        }
        let channels = config.channel_count as usize;
        if pcm_out.len() < channels {
            return Err(Atrac9Error::BufferTooSmall);
        }
        if pcm_out[..channels]
            .iter()
            .any(|c| c.len() < config.superframe_samples as usize)
        {
            return Err(Atrac9Error::BufferTooSmall);
        }
        Ok(())
    }

    fn decode_superframe(&mut self, reader: &mut BitReader<'_>, pcm_out: &mut [Vec<i16>]) -> Result<(), Atrac9Error> {
        for i in 0..self.config.frames_per_superframe {
            self.frame.frame_index = i;
            decode_frame(reader, &mut self.frame, &self.config)?;
            self.pcm_float_to_short(pcm_out, (i * self.config.frame_samples) as usize);
            reader.align_position(8);
        }
        Ok(())
    }

    fn pcm_float_to_short(&self, pcm_out: &mut [Vec<i16>], start: usize) {
        let frame_samples = self.config.frame_samples as usize;
        let mut channel_num = 0;
        for block in &self.frame.blocks {
            for channel in &block.channels {
                let dest = &mut pcm_out[channel_num][start..start + frame_samples];
                channel_num += 1;
                for (d, &sample) in dest.iter_mut().zip(channel.pcm.iter()) {
                    // floor(x + 0.5), then saturate to 32 and 16 bits.
                    let rounded = (sample + 0.5).floor() as i32;
                    *d = clamp16(rounded);
                }
            }
        }
    }
}

fn decode_frame(reader: &mut BitReader<'_>, frame: &mut Frame, config: &Atrac9Config) -> Result<(), Atrac9Error> {
    unpack::unpack_frame(reader, frame, config)?;

    for block in &mut frame.blocks {
        quantization::dequantize_spectra(block)?;
        quantization::apply_intensity_stereo(block)?;
        quantization::scale_spectrum(block)?;
        band_extension::apply_band_extension(block)?;
        imdct_block(block);
    }
    Ok(())
}

fn imdct_block(block: &mut Block) {
    for channel in &mut block.channels {
        let ok = channel.mdct.run_imdct(&channel.spectra, &mut channel.pcm);
        debug_assert!(ok, "IMDCT buffers are always 256 values");
    }
}
