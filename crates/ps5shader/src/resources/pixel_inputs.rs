// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Pixel-shader input locations from SPI_PS_INPUT_CNTL.

use super::ResourcePlanError;

/// The host location of each active pixel input: the OFFSET field (bits
/// 4:0) of its SPI_PS_INPUT_CNTL entry, or, when two inputs collide, the
/// first free location at or after the input's own index.
pub fn resolve_pixel_input_locations(controls: &[u32], active_inputs: &[u32]) -> Result<Vec<u32>, ResourcePlanError> {
    let mut used = [false; 32];
    let mut locations = Vec::with_capacity(active_inputs.len());
    for &input in active_inputs {
        let mut location = controls.get(input as usize).map_or(input, |c| c & 0x1F);
        if location < 32 && used[location as usize] {
            location = input;
            while location < 32 && used[location as usize] {
                location += 1;
            }
            if location >= 32 {
                return Err(ResourcePlanError::new("pixel input locations exceed the hardware interface limit"));
            }
        }
        locations.push(location);
        if location < 32 {
            used[location as usize] = true;
        }
    }
    Ok(locations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locations() {
        assert_eq!(resolve_pixel_input_locations(&[0, 1, 2], &[0, 1, 2]).unwrap(), vec![0, 1, 2]);
        // Remapped offsets.
        assert_eq!(resolve_pixel_input_locations(&[5, 0x23, 1], &[0, 1, 2]).unwrap(), vec![5, 3, 1]);
        // A collision moves to the first free location from the input index.
        assert_eq!(resolve_pixel_input_locations(&[2, 2, 2], &[0, 1, 2]).unwrap(), vec![2, 1, 3]);
        // Inputs past the control table keep their index.
        assert_eq!(resolve_pixel_input_locations(&[], &[4]).unwrap(), vec![4]);
        let all: Vec<u32> = (0..33).collect();
        assert!(resolve_pixel_input_locations(&[0; 33], &all).is_err());
    }
}
