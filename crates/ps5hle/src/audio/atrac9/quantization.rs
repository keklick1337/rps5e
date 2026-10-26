// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Dequantization, spectrum scaling and intensity stereo.

use super::block::Block;
use super::config::BlockType;
use super::tables::{self, QUANT_UNIT_TO_COEFF_COUNT, QUANT_UNIT_TO_COEFF_INDEX};
use super::{invalid, Atrac9Error};

fn unit_range(unit: i32) -> Result<(usize, usize), Atrac9Error> {
    let u = usize::try_from(unit).map_err(|_| invalid("quant unit out of range"))?;
    if u + 1 >= QUANT_UNIT_TO_COEFF_INDEX.len() {
        return Err(invalid("quant unit out of range"));
    }
    Ok((QUANT_UNIT_TO_COEFF_INDEX[u] as usize, QUANT_UNIT_TO_COEFF_INDEX[u + 1] as usize))
}

pub(super) fn dequantize_spectra(block: &mut Block) -> Result<(), Atrac9Error> {
    let t = tables::tables();
    for channel in &mut block.channels {
        channel.spectra = [0.0; 256];

        for band in 0..channel.coded_quant_units.max(0) as usize {
            let sub_band_index = *QUANT_UNIT_TO_COEFF_INDEX
                .get(band)
                .ok_or_else(|| invalid("quant unit out of range"))? as usize;
            let sub_band_count = *QUANT_UNIT_TO_COEFF_COUNT
                .get(band)
                .ok_or_else(|| invalid("quant unit out of range"))? as usize;
            let step_size = *usize::try_from(channel.precisions[band])
                .ok()
                .and_then(|p| t.quantizer_step_size.get(p))
                .ok_or_else(|| invalid("precision out of range"))?;
            let step_size_fine = *usize::try_from(channel.precisions_fine[band])
                .ok()
                .and_then(|p| t.quantizer_fine_step_size.get(p))
                .ok_or_else(|| invalid("fine precision out of range"))?;

            for sb in 0..sub_band_count {
                let idx = sub_band_index + sb;
                let coarse = channel.quantized_spectra[idx] as f64 * step_size;
                let fine = channel.quantized_spectra_fine[idx] as f64 * step_size_fine;
                channel.spectra[idx] = coarse + fine;
            }
        }
    }
    Ok(())
}

pub(super) fn scale_spectrum(block: &mut Block) -> Result<(), Atrac9Error> {
    let t = tables::tables();
    let quant_unit_count = block.quantization_unit_count;
    for channel in &mut block.channels {
        for i in 0..quant_unit_count.max(0) {
            let (start, end) = unit_range(i)?;
            let scale = *usize::try_from(channel.scale_factors[i as usize])
                .ok()
                .and_then(|s| t.spectrum_scale.get(s))
                .ok_or_else(|| invalid("scale factor out of range"))?;
            for sb in start..end {
                channel.spectra[sb] *= scale;
            }
        }
    }
    Ok(())
}

pub(super) fn apply_intensity_stereo(block: &mut Block) -> Result<(), Atrac9Error> {
    if block.block_type != BlockType::Stereo {
        return Ok(());
    }

    let total_units = block.quantization_unit_count;
    let stereo_units = block.stereo_quantization_unit;
    if stereo_units >= total_units {
        return Ok(());
    }

    let source = block.primary_channel();
    let dest = block.secondary_channel();
    let source_spectra = block.channels[source].spectra;

    for i in stereo_units.max(0)..total_units {
        let sign = *block
            .joint_stereo_signs
            .get(i as usize)
            .ok_or_else(|| invalid("stereo unit out of range"))?;
        let (start, end) = unit_range(i)?;
        let dest_spectra = &mut block.channels[dest].spectra;
        for sb in start..end {
            dest_spectra[sb] = if sign > 0 { -source_spectra[sb] } else { source_spectra[sb] };
        }
    }
    Ok(())
}
