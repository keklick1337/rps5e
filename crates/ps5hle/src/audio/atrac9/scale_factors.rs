// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Scale factor decoding (four coding modes per channel).

use super::bit_reader::BitReader;
use super::block::Block;
use super::huffman::huffman_tables;
use super::{invalid, Atrac9Error};

pub(super) fn read(reader: &mut BitReader<'_>, block: &mut Block, ch: usize) -> Result<(), Atrac9Error> {
    let extension_unit = block.extension_unit;
    if !(0..=30).contains(&extension_unit) {
        return Err(invalid("extension unit out of range"));
    }
    let first = block.first_in_superframe;
    let prev_units = block.quantization_units_prev;
    let channel_index = block.channels[ch].channel_index;
    let channel0_sf = block.channels[0].scale_factors;
    let channel = &mut block.channels[ch];

    channel.scale_factors = [0; 31];
    channel.scale_factor_coding_mode = reader.read_int(2);
    let prev_sf = channel.scale_factors_prev;
    let sf = &mut channel.scale_factors;

    if channel_index == 0 {
        match channel.scale_factor_coding_mode {
            0 => read_vlc_delta_offset(reader, sf, extension_unit)?,
            1 => read_clc_offset(reader, sf, extension_unit),
            2 => {
                if first {
                    return Err(invalid("scale factor mode 2 in first frame"));
                }
                read_vlc_distance_to_baseline(reader, sf, extension_unit, &prev_sf, prev_units)?;
            }
            _ => {
                if first {
                    return Err(invalid("scale factor mode 3 in first frame"));
                }
                read_vlc_delta_offset_with_baseline(reader, sf, extension_unit, &prev_sf, prev_units)?;
            }
        }
    } else {
        match channel.scale_factor_coding_mode {
            0 => read_vlc_delta_offset(reader, sf, extension_unit)?,
            1 => read_vlc_distance_to_baseline(reader, sf, extension_unit, &channel0_sf, extension_unit)?,
            2 => read_vlc_delta_offset_with_baseline(reader, sf, extension_unit, &channel0_sf, extension_unit)?,
            _ => {
                if first {
                    return Err(invalid("scale factor mode 3 in first frame"));
                }
                read_vlc_distance_to_baseline(reader, sf, extension_unit, &prev_sf, prev_units)?;
            }
        }
    }

    for &value in &sf[..extension_unit as usize] {
        if !(0..=31).contains(&value) {
            return Err(invalid("Scale factor values are out of range."));
        }
    }

    channel.scale_factors_prev = channel.scale_factors;
    Ok(())
}

fn read_clc_offset(reader: &mut BitReader<'_>, sf: &mut [i32; 31], extension_unit: i32) {
    const MAX_BITS: i32 = 5;
    let bit_length = reader.read_int(2) + 2;
    let base_value = if bit_length < MAX_BITS { reader.read_int(MAX_BITS) } else { 0 };
    for value in sf.iter_mut().take(extension_unit as usize) {
        *value = reader.read_int(bit_length) + base_value;
    }
}

fn read_vlc_delta_offset(reader: &mut BitReader<'_>, sf: &mut [i32; 31], extension_unit: i32) -> Result<(), Atrac9Error> {
    let weight_index = reader.read_int(3) as usize;
    let weights = &SCALE_FACTOR_WEIGHTS[weight_index];

    let base_value = reader.read_int(5);
    let bit_length = reader.read_int(2) + 3;
    let codebook = huffman_tables()
        .scale_factors_unsigned(bit_length)
        .ok_or_else(|| invalid("missing scale factor codebook"))?;

    sf[0] = reader.read_int(bit_length);

    for i in 1..extension_unit as usize {
        let delta = codebook.read_value(reader, false);
        sf[i] = (sf[i - 1] + delta) & (codebook.value_max - 1);
    }

    for i in 0..extension_unit as usize {
        sf[i] += base_value - weights[i] as i32;
    }
    Ok(())
}

fn read_vlc_distance_to_baseline(
    reader: &mut BitReader<'_>,
    sf: &mut [i32; 31],
    extension_unit: i32,
    baseline: &[i32; 31],
    baseline_length: i32,
) -> Result<(), Atrac9Error> {
    let bit_length = reader.read_int(2) + 2;
    let codebook = huffman_tables()
        .scale_factors_signed(bit_length)
        .ok_or_else(|| invalid("missing scale factor codebook"))?;
    let unit_count = extension_unit.min(baseline_length).max(0) as usize;

    for i in 0..unit_count {
        let distance = codebook.read_value(reader, true);
        sf[i] = (baseline[i] + distance) & 31;
    }

    for value in sf.iter_mut().take(extension_unit as usize).skip(unit_count) {
        *value = reader.read_int(5);
    }
    Ok(())
}

fn read_vlc_delta_offset_with_baseline(
    reader: &mut BitReader<'_>,
    sf: &mut [i32; 31],
    extension_unit: i32,
    baseline: &[i32; 31],
    baseline_length: i32,
) -> Result<(), Atrac9Error> {
    let base_value = reader.read_offset_binary_negative(5);
    let bit_length = reader.read_int(2) + 1;
    let codebook = huffman_tables()
        .scale_factors_unsigned(bit_length)
        .ok_or_else(|| invalid("missing scale factor codebook"))?;
    let unit_count = extension_unit.min(baseline_length).max(0) as usize;

    sf[0] = reader.read_int(bit_length);

    for i in 1..unit_count {
        let delta = codebook.read_value(reader, false);
        sf[i] = (sf[i - 1] + delta) & (codebook.value_max - 1);
    }

    for i in 0..unit_count {
        sf[i] += base_value + baseline[i];
    }

    for value in sf.iter_mut().take(extension_unit as usize).skip(unit_count) {
        *value = reader.read_int(5);
    }
    Ok(())
}

static SCALE_FACTOR_WEIGHTS: [[u8; 32]; 8] = [
    [0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 2, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 10, 12, 12, 12],
    [3, 2, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 2, 3, 3, 4, 5, 7, 10, 10, 10],
    [0, 2, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 9, 12, 12, 12],
    [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 8, 8, 10, 11, 11, 12, 13, 13, 13, 13],
    [0, 2, 2, 3, 3, 4, 4, 5, 4, 5, 5, 5, 5, 6, 7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 11, 12, 12, 13, 13, 14, 14, 14],
    [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 7, 9, 11, 11, 11],
    [0, 5, 8, 10, 11, 11, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 13, 15, 15, 15],
    [0, 2, 3, 4, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 15, 15, 15],
];
