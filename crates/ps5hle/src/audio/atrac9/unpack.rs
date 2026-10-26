// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Bitstream unpacking: block headers, band/gradient/stereo/extension params,
//! scale factors and quantized spectra.

use super::band_extension::{get_bex_band_info, BEX_DATA_LENGTHS, BEX_ENCODED_VALUE_COUNTS};
use super::bit_allocation;
use super::bit_reader::{sign_extend32, BitReader};
use super::block::{Block, Frame};
use super::config::{Atrac9Config, BlockType};
use super::huffman::{huffman_tables, HuffmanCodebook};
use super::scale_factors;
use super::tables::{self, QUANT_UNIT_TO_CODEBOOK_INDEX, QUANT_UNIT_TO_COEFF_COUNT, QUANT_UNIT_TO_COEFF_INDEX};
use super::{invalid, Atrac9Error};

pub(super) fn unpack_frame(
    reader: &mut BitReader<'_>,
    frame: &mut Frame,
    config: &Atrac9Config,
) -> Result<(), Atrac9Error> {
    if config.independent_block_superframe_bytes != 0 {
        return unpack_independent_mono_frame(reader, frame, config);
    }

    let frame_index = frame.frame_index;
    for block in &mut frame.blocks {
        unpack_block(reader, block, config, frame_index)?;
    }
    Ok(())
}

/// Each channel is a standalone mono superframe with a fixed byte budget,
/// interleaved frame by frame. Blocks of the earlier frames are packed back to
/// back; in the last frame every block gets whatever is left of its channel's
/// budget, so those slots end in padding.
fn unpack_independent_mono_frame(
    reader: &mut BitReader<'_>,
    frame: &mut Frame,
    config: &Atrac9Config,
) -> Result<(), Atrac9Error> {
    let budget = config.independent_block_superframe_bytes;
    let frame_index = frame.frame_index;
    let last_frame = frame_index == config.frames_per_superframe - 1;
    let mut slot_start = reader.position / 8;

    for block in &mut frame.blocks {
        if frame_index == 0 {
            block.superframe_bytes_used = 0;
        }

        let used_before = block.superframe_bytes_used;
        if last_frame {
            reader.position = slot_start.wrapping_mul(8);
        }

        let start = reader.position;
        unpack_block(reader, block, config, frame_index)?;
        block.superframe_bytes_used += (reader.position - start) / 8;
        if block.superframe_bytes_used > budget {
            return Err(invalid("independent block exceeds its byte budget"));
        }

        if last_frame {
            slot_start += budget - used_before;
            reader.position = slot_start.wrapping_mul(8);
        }
    }
    Ok(())
}

fn unpack_block(
    reader: &mut BitReader<'_>,
    block: &mut Block,
    config: &Atrac9Config,
    frame_index: i32,
) -> Result<(), Atrac9Error> {
    read_block_header(reader, block, frame_index)?;

    if block.block_type == BlockType::Lfe {
        unpack_lfe_block(reader, block);
    } else {
        unpack_standard_block(reader, block, config)?;
    }

    reader.align_position(8);
    Ok(())
}

fn read_block_header(reader: &mut BitReader<'_>, block: &mut Block, frame_index: i32) -> Result<(), Atrac9Error> {
    let first_in_superframe = frame_index == 0;
    block.first_in_superframe = !reader.read_bool();
    block.reuse_band_params = reader.read_bool();

    if block.first_in_superframe != first_in_superframe {
        return Err(invalid("block superframe position mismatch"));
    }

    if first_in_superframe && block.reuse_band_params && block.block_type != BlockType::Lfe {
        return Err(invalid("band parameters reused in first frame"));
    }
    Ok(())
}

fn unpack_standard_block(
    reader: &mut BitReader<'_>,
    block: &mut Block,
    config: &Atrac9Config,
) -> Result<(), Atrac9Error> {
    if !block.reuse_band_params {
        read_band_params(reader, block, config)?;
    }

    read_gradient_params(reader, block)?;
    bit_allocation::create_gradient(block)?;
    read_stereo_params(reader, block)?;
    read_extension_params(reader, block)?;

    for ch in 0..block.channels.len() {
        block.update_coded_units(ch);

        scale_factors::read(reader, block, ch)?;
        bit_allocation::calculate_mask(block, ch)?;
        bit_allocation::calculate_precisions(block, ch)?;
        calculate_spectrum_codebook_index(block, ch, config)?;

        read_spectra(reader, block, ch, config)?;
        read_spectra_fine(reader, block, ch)?;
    }

    block.quantization_units_prev = if block.band_extension_enabled {
        block.extension_unit
    } else {
        block.quantization_unit_count
    };
    Ok(())
}

fn band_to_quant_units(band: i32) -> Result<i32, Atrac9Error> {
    usize::try_from(band)
        .ok()
        .and_then(|b| tables::BAND_TO_QUANT_UNIT_COUNT.get(b))
        .map(|&v| v as i32)
        .ok_or_else(|| invalid("band count out of range"))
}

fn read_band_params(reader: &mut BitReader<'_>, block: &mut Block, config: &Atrac9Config) -> Result<(), Atrac9Error> {
    let min_band_count = tables::min_band_count(config.high_sample_rate);
    let max_extension_band = tables::max_extension_band(config.high_sample_rate);
    block.band_count = reader.read_int(4) + min_band_count;
    block.quantization_unit_count = band_to_quant_units(block.band_count)?;
    if block.band_count < min_band_count
        || block.band_count > tables::MAX_BAND_COUNT[config.sample_rate_index as usize] as i32
    {
        return Ok(());
    }

    if block.block_type == BlockType::Stereo {
        block.stereo_band = reader.read_int(4) + min_band_count;
        block.stereo_quantization_unit = band_to_quant_units(block.stereo_band)?;
    } else {
        block.stereo_band = block.band_count;
    }

    block.band_extension_enabled = reader.read_bool();
    if block.band_extension_enabled {
        block.extension_band = reader.read_int(4) + min_band_count;
        if block.extension_band < block.band_count || block.extension_band > max_extension_band {
            return Err(invalid("extension band out of range"));
        }
        block.extension_unit = band_to_quant_units(block.extension_band)?;
    } else {
        block.extension_band = block.band_count;
        block.extension_unit = block.quantization_unit_count;
    }
    Ok(())
}

fn read_gradient_params(reader: &mut BitReader<'_>, block: &mut Block) -> Result<(), Atrac9Error> {
    block.gradient_mode = reader.read_int(2);
    if block.gradient_mode > 0 {
        block.gradient_end_unit = 31;
        block.gradient_end_value = 31;
        block.gradient_start_unit = reader.read_int(5);
        block.gradient_start_value = reader.read_int(5);
    } else {
        block.gradient_start_unit = reader.read_int(6);
        block.gradient_end_unit = reader.read_int(6) + 1;
        block.gradient_start_value = reader.read_int(5);
        block.gradient_end_value = reader.read_int(5);
    }
    block.gradient_boundary = reader.read_int(4);

    // The gradient holds 31 units (one past the last quantisation unit), so
    // an end unit past 31 is outside the decoder's bounds.
    if block.gradient_boundary > block.quantization_unit_count
        || !(0..48).contains(&block.gradient_start_unit)
        || !(0..=31).contains(&block.gradient_end_unit)
        || block.gradient_start_unit > block.gradient_end_unit
        || !(0..32).contains(&block.gradient_start_value)
        || !(0..32).contains(&block.gradient_end_value)
    {
        return Err(invalid("invalid gradient parameters"));
    }
    Ok(())
}

fn read_stereo_params(reader: &mut BitReader<'_>, block: &mut Block) -> Result<(), Atrac9Error> {
    if block.block_type != BlockType::Stereo {
        return Ok(());
    }

    block.primary_channel_index = reader.read_int(1);
    block.has_joint_stereo_signs = reader.read_bool();
    if block.has_joint_stereo_signs {
        for i in block.stereo_quantization_unit.max(0)..block.quantization_unit_count {
            *block
                .joint_stereo_signs
                .get_mut(i as usize)
                .ok_or_else(|| invalid("stereo unit out of range"))? = reader.read_int(1);
        }
    } else {
        block.joint_stereo_signs = [0; 30];
    }
    Ok(())
}

fn read_bex_header(reader: &mut BitReader<'_>, block: &mut Block, ch: usize, bex_band: i32) -> Result<(), Atrac9Error> {
    let bex_mode = reader.read_int(2);
    let channel = block
        .channels
        .get_mut(ch)
        .ok_or_else(|| invalid("band extension channel missing"))?;
    channel.bex_mode = if bex_band > 2 { bex_mode } else { 4 };
    channel.bex_value_count = BEX_ENCODED_VALUE_COUNTS
        .get(channel.bex_mode as usize)
        .and_then(|row| row.get(bex_band as usize))
        .copied()
        .ok_or_else(|| invalid("band extension mode out of range"))? as i32;
    Ok(())
}

fn read_bex_data(reader: &mut BitReader<'_>, block: &mut Block, ch: usize, bex_band: i32) -> Result<(), Atrac9Error> {
    let channel = &mut block.channels[ch];
    for i in 0..channel.bex_value_count.max(0) as usize {
        let data_length = BEX_DATA_LENGTHS
            .get(channel.bex_mode as usize)
            .and_then(|m| m.get(bex_band as usize))
            .and_then(|b| b.get(i))
            .copied()
            .ok_or_else(|| invalid("band extension data length out of range"))?;
        *channel
            .bex_values
            .get_mut(i)
            .ok_or_else(|| invalid("band extension value out of range"))? = reader.read_int(data_length as i32);
    }
    Ok(())
}

fn read_extension_params(reader: &mut BitReader<'_>, block: &mut Block) -> Result<(), Atrac9Error> {
    let mut bex_band = 0;
    if block.band_extension_enabled {
        bex_band = get_bex_band_info(block.quantization_unit_count)?.0;
        if block.block_type == BlockType::Stereo {
            read_bex_header(reader, block, 1, bex_band)?;
        } else {
            reader.position += 1;
        }
    }
    block.has_extension_data = reader.read_bool();

    if !block.has_extension_data {
        return Ok(());
    }
    if !block.band_extension_enabled {
        block.bex_mode = reader.read_int(2);
        block.bex_data_length = reader.read_int(5);
        reader.position += block.bex_data_length;
        return Ok(());
    }

    read_bex_header(reader, block, 0, bex_band)?;

    block.bex_data_length = reader.read_int(5);
    if block.bex_data_length <= 0 {
        return Ok(());
    }
    let bex_data_end = reader.position + block.bex_data_length;

    read_bex_data(reader, block, 0, bex_band)?;

    if block.block_type == BlockType::Stereo {
        read_bex_data(reader, block, 1, bex_band)?;
    }

    // Make sure we didn't read too many bits.
    if reader.position > bex_data_end {
        return Err(invalid("band extension data overrun"));
    }
    Ok(())
}

fn calculate_spectrum_codebook_index(block: &mut Block, ch: usize, config: &Atrac9Config) -> Result<(), Atrac9Error> {
    let channel = &mut block.channels[ch];
    channel.codebook_set = [0; 30];
    let quant_units = channel.coded_quant_units;

    if quant_units <= 1 || config.high_sample_rate {
        return Ok(());
    }
    if quant_units > 30 {
        return Err(invalid("coded quant units out of range"));
    }
    let quant_units = quant_units as usize;

    // Temporarily duplicating the last value makes the final unit a
    // non-special case.
    let mut sf = channel.scale_factors;
    sf[quant_units] = sf[quant_units - 1];

    let mut avg = 0;
    if quant_units > 12 {
        avg = sf[..12].iter().sum::<i32>();
        avg = (avg + 6) / 12;
    }

    for i in 8..quant_units {
        let prev_sf = sf[i - 1];
        let next_sf = sf[i + 1];
        let min_sf = prev_sf.min(next_sf);
        if sf[i] - min_sf >= 3 || sf[i] - prev_sf + sf[i] - next_sf >= 3 {
            channel.codebook_set[i] = 1;
        }
    }

    for i in 12..quant_units {
        if channel.codebook_set[i] == 0 {
            let min_sf = sf[i - 1].min(sf[i + 1]);
            let bias = if QUANT_UNIT_TO_COEFF_COUNT[i] == 16 { 1 } else { 0 };
            if sf[i] - min_sf >= 2 && sf[i] >= avg - bias {
                channel.codebook_set[i] = 1;
            }
        }
    }
    Ok(())
}

fn read_spectra(reader: &mut BitReader<'_>, block: &mut Block, ch: usize, config: &Atrac9Config) -> Result<(), Atrac9Error> {
    let channel = &mut block.channels[ch];
    channel.quantized_spectra = [0; 256];
    let max_huff_precision = tables::max_huff_precision(config.high_sample_rate);
    let huff_tables = huffman_tables();

    for i in 0..channel.coded_quant_units.max(0) as usize {
        if i >= 30 {
            return Err(invalid("coded quant units out of range"));
        }
        let subband_count = QUANT_UNIT_TO_COEFF_COUNT[i] as i32;
        let subband_index = QUANT_UNIT_TO_COEFF_INDEX[i] as usize;
        let precision = channel.precisions[i] + 1;
        if precision <= max_huff_precision {
            let huff = huff_tables
                .spectrum_codebook(channel.codebook_set[i], precision, QUANT_UNIT_TO_CODEBOOK_INDEX[i] as i32)
                .ok_or_else(|| invalid("missing spectrum codebook"))?;
            let group_count = (subband_count >> huff.value_count_power) as usize;
            for j in 0..group_count {
                channel.spectra_values_buffer[j] = huff.read_value(reader, false);
            }

            decode_huffman_values(
                &mut channel.quantized_spectra,
                subband_index,
                subband_count,
                huff,
                &channel.spectra_values_buffer,
            );
        } else {
            let end = QUANT_UNIT_TO_COEFF_INDEX[i + 1] as usize;
            for j in subband_index..end {
                channel.quantized_spectra[j] = reader.read_signed_int(precision);
            }
        }
    }
    Ok(())
}

fn read_spectra_fine(reader: &mut BitReader<'_>, block: &mut Block, ch: usize) -> Result<(), Atrac9Error> {
    let channel = &mut block.channels[ch];
    channel.quantized_spectra_fine = [0; 256];

    for i in 0..channel.coded_quant_units.max(0) as usize {
        if i >= 30 {
            return Err(invalid("coded quant units out of range"));
        }
        if channel.precisions_fine[i] > 0 {
            let overflow_bits = channel.precisions_fine[i] + 1;
            let start_subband = QUANT_UNIT_TO_COEFF_INDEX[i] as usize;
            let end_subband = QUANT_UNIT_TO_COEFF_INDEX[i + 1] as usize;
            for j in start_subband..end_subband {
                channel.quantized_spectra_fine[j] = reader.read_signed_int(overflow_bits);
            }
        }
    }
    Ok(())
}

fn decode_huffman_values(
    spectrum: &mut [i32; 256],
    mut index: usize,
    band_count: i32,
    huff: &HuffmanCodebook,
    values: &[i32; 16],
) {
    let value_count = (band_count >> huff.value_count_power) as usize;
    let mask = (1i32 << huff.value_bits) - 1;

    for &packed in values.iter().take(value_count) {
        let mut value = packed;
        for _ in 0..huff.value_count {
            if let Some(slot) = spectrum.get_mut(index) {
                *slot = sign_extend32(value & mask, huff.value_bits);
            }
            index += 1;
            value >>= huff.value_bits;
        }
    }
}

fn unpack_lfe_block(reader: &mut BitReader<'_>, block: &mut Block) {
    block.quantization_unit_count = 2;
    let reuse = block.reuse_band_params;
    let channel = &mut block.channels[0];

    // Scale factors.
    channel.scale_factors = [0; 31];
    for i in 0..2 {
        channel.scale_factors[i] = reader.read_int(5);
    }

    // Precision.
    let precision = if reuse { 8 } else { 4 };
    for i in 0..2 {
        channel.precisions[i] = precision;
        channel.precisions_fine[i] = 0;
    }
    channel.coded_quant_units = 2;

    // Spectra.
    channel.quantized_spectra = [0; 256];
    for i in 0..2 {
        if channel.precisions[i] <= 0 {
            continue;
        }
        let precision = channel.precisions[i] + 1;
        for j in QUANT_UNIT_TO_COEFF_INDEX[i] as usize..QUANT_UNIT_TO_COEFF_INDEX[i + 1] as usize {
            channel.quantized_spectra[j] = reader.read_signed_int(precision);
        }
    }
}

#[cfg(test)]
mod gradient_tests {
    use super::*;

    /// Bits of a mode-0 gradient header: start unit, end unit - 1, values 0,
    /// boundary 0.
    fn header(start: u32, end_minus_one: u32) -> Vec<u8> {
        let fields = [(0u32, 2u32), (start, 6), (end_minus_one, 6), (0, 5), (0, 5), (0, 4)];
        let mut bits = Vec::new();
        for (v, n) in fields {
            for i in (0..n).rev() {
                bits.push((v >> i) & 1 != 0);
            }
        }
        let mut out = vec![0u8; bits.len().div_ceil(8) + 4];
        for (i, b) in bits.iter().enumerate() {
            if *b {
                out[i / 8] |= 0x80 >> (i % 8);
            }
        }
        out
    }

    /// A gradient end unit past 31 is refused before it reaches the 31-unit
    /// gradient; 31 itself is accepted (reference LibAtrac9 bounds fix).
    #[test]
    fn gradient_end_unit_past_31_is_refused() {
        let config = Atrac9Config::new(&super::super::tests::mono_config()).unwrap();
        let mut block = Frame::new(&config).blocks.remove(0);
        let ok = header(1, 30);
        assert!(read_gradient_params(&mut BitReader::new(&ok), &mut block).is_ok());
        assert_eq!(block.gradient_end_unit, 31);
        for end_minus_one in [31u32, 40, 46] {
            let bad = header(1, end_minus_one);
            assert!(read_gradient_params(&mut BitReader::new(&bad), &mut block).is_err(), "end unit {}", end_minus_one + 1);
        }
    }
}
