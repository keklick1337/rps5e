// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Bit allocation: gradient curves, precision masks and per-unit precisions.

use super::block::Block;
use super::tables;
use super::{invalid, Atrac9Error};

pub(super) fn create_gradient(block: &mut Block) -> Result<(), Atrac9Error> {
    let value_count = block.gradient_end_value - block.gradient_start_value;
    let unit_count = block.gradient_end_unit - block.gradient_start_unit;

    for i in 0..block.gradient_end_unit.max(0) {
        *block
            .gradient
            .get_mut(i as usize)
            .ok_or_else(|| invalid("gradient end unit out of range"))? = block.gradient_start_value;
    }

    for i in block.gradient_end_unit.max(0)..=block.quantization_unit_count {
        *block
            .gradient
            .get_mut(i as usize)
            .ok_or_else(|| invalid("gradient index out of range"))? = block.gradient_end_value;
    }

    if unit_count <= 0 || value_count == 0 {
        return Ok(());
    }

    let curve = tables::tables()
        .gradient_curves
        .get((unit_count - 1) as usize)
        .ok_or_else(|| invalid("gradient curve out of range"))?;
    let start = block.gradient_start_unit;
    let (scale, base_val, sign) = if value_count <= 0 {
        ((-value_count - 1) as f64 / 31.0, block.gradient_start_value - 1, -1)
    } else {
        ((value_count - 1) as f64 / 31.0, block.gradient_start_value + 1, 1)
    };

    for i in start..block.gradient_end_unit {
        let c = *curve
            .get((i - start) as usize)
            .ok_or_else(|| invalid("gradient curve index out of range"))?;
        let delta = (c as f64 * scale) as i32;
        let value = if sign < 0 { base_val - delta } else { base_val + delta };
        *block
            .gradient
            .get_mut(i as usize)
            .ok_or_else(|| invalid("gradient index out of range"))? = value;
    }
    Ok(())
}

pub(super) fn calculate_mask(block: &mut Block, ch: usize) -> Result<(), Atrac9Error> {
    let qu = block.quantization_unit_count;
    let channel = &mut block.channels[ch];
    channel.precision_mask = [0; 30];
    for i in 1..qu {
        let i = i as usize;
        let cur = *channel.scale_factors.get(i).ok_or_else(|| invalid("quant unit out of range"))?;
        let delta = cur - channel.scale_factors[i - 1];
        if delta > 1 {
            *channel.precision_mask.get_mut(i).ok_or_else(|| invalid("quant unit out of range"))? +=
                (delta - 1).min(5);
        } else if delta < -1 {
            *channel
                .precision_mask
                .get_mut(i - 1)
                .ok_or_else(|| invalid("quant unit out of range"))? += (-delta - 1).min(5);
        }
    }
    Ok(())
}

pub(super) fn calculate_precisions(block: &mut Block, ch: usize) -> Result<(), Atrac9Error> {
    let qu = block.quantization_unit_count.max(0) as usize;
    if qu > 30 {
        return Err(invalid("quantization unit count out of range"));
    }
    let gradient_mode = block.gradient_mode;
    let gradient_boundary = block.gradient_boundary.max(0) as usize;
    let gradient = block.gradient;
    let channel = &mut block.channels[ch];

    if gradient_mode != 0 {
        for i in 0..qu {
            let mut p = channel.scale_factors[i] + channel.precision_mask[i] - gradient[i];
            if p > 0 {
                match gradient_mode {
                    1 => p /= 2,
                    2 => p = 3 * p / 8,
                    3 => p /= 4,
                    _ => {}
                }
            }
            channel.precisions[i] = p;
        }
    } else {
        for i in 0..qu {
            channel.precisions[i] = channel.scale_factors[i] - gradient[i];
        }
    }

    for i in 0..qu {
        if channel.precisions[i] < 1 {
            channel.precisions[i] = 1;
        }
    }

    for i in 0..gradient_boundary {
        *channel.precisions.get_mut(i).ok_or_else(|| invalid("gradient boundary out of range"))? += 1;
    }

    for i in 0..qu {
        channel.precisions_fine[i] = 0;
        if channel.precisions[i] > 15 {
            channel.precisions_fine[i] = channel.precisions[i] - 15;
            channel.precisions[i] = 15;
        }
    }
    Ok(())
}

pub(super) fn generate_gradient_curves() -> Vec<Vec<u8>> {
    const MAIN: [u8; 48] = [
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 26, 27, 27, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
    ];
    (1..=MAIN.len())
        .map(|length| (0..length).map(|i| MAIN[i * MAIN.len() / length]).collect())
        .collect()
}
