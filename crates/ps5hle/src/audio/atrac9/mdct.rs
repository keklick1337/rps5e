// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Inverse MDCT (via a type-4 DCT) in double precision.

use std::f64::consts::PI;
use std::sync::OnceLock;

use super::bit_reader::bit_reverse32;

/// Largest transform size (in bits) the trig/shuffle tables are built for.
pub(super) const MAX_TABLE_BITS: usize = 8;

struct TrigTables {
    sin: Vec<Vec<f64>>,
    cos: Vec<Vec<f64>>,
    shuffle: Vec<Vec<usize>>,
}

fn trig_tables() -> &'static TrigTables {
    static TABLES: OnceLock<TrigTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut sin = Vec::with_capacity(MAX_TABLE_BITS + 1);
        let mut cos = Vec::with_capacity(MAX_TABLE_BITS + 1);
        let mut shuffle = Vec::with_capacity(MAX_TABLE_BITS + 1);
        for bits in 0..=MAX_TABLE_BITS as i32 {
            let (s, c) = generate_trig_tables(bits);
            sin.push(s);
            cos.push(c);
            shuffle.push(generate_shuffle_table(bits));
        }
        TrigTables { sin, cos, shuffle }
    })
}

pub(super) fn generate_trig_tables(size_bits: i32) -> (Vec<f64>, Vec<f64>) {
    let size = 1usize << size_bits;
    let mut sin = vec![0.0; size];
    let mut cos = vec![0.0; size];
    for i in 0..size {
        let value = PI * (4 * i + 1) as f64 / (4 * size) as f64;
        sin[i] = value.sin();
        cos[i] = value.cos();
    }
    (sin, cos)
}

pub(super) fn generate_shuffle_table(size_bits: i32) -> Vec<usize> {
    let size = 1i32 << size_bits;
    (0..size)
        .map(|i| bit_reverse32(i ^ (i / 2), size_bits) as u32 as usize)
        .collect()
}

pub(super) struct Mdct {
    mdct_bits: usize,
    mdct_size: usize,
    scale: f64,
    imdct_previous: Vec<f64>,
    imdct_window: &'static [f64],
    scratch_mdct: Vec<f64>,
    scratch_dct: Vec<f64>,
}

impl Mdct {
    /// `mdct_bits` must be in 1..=8 and `window` at least `1 << mdct_bits` long.
    pub fn new(mdct_bits: usize, window: &'static [f64], scale: f64) -> Self {
        assert!((1..=MAX_TABLE_BITS).contains(&mdct_bits), "unsupported MDCT size");
        let mdct_size = 1usize << mdct_bits;
        assert!(window.len() >= mdct_size, "Window must be as long as the MDCT size.");
        Self {
            mdct_bits,
            mdct_size,
            scale,
            imdct_previous: vec![0.0; mdct_size],
            imdct_window: window,
            scratch_mdct: vec![0.0; mdct_size],
            scratch_dct: vec![0.0; mdct_size],
        }
    }

    /// Runs the inverse MDCT with overlap-add. Both slices must hold at least
    /// `size()` values; shorter slices are rejected without touching state.
    pub fn run_imdct(&mut self, input: &[f64], output: &mut [f64]) -> bool {
        if input.len() < self.mdct_size || output.len() < self.mdct_size {
            return false;
        }

        let size = self.mdct_size;
        let half = size / 2;
        let mut dct_out = std::mem::take(&mut self.scratch_mdct);
        self.dct4(input, &mut dct_out);

        let window = self.imdct_window;
        let prev = &mut self.imdct_previous;
        for i in 0..half {
            output[i] = window[i] * dct_out[i + half] + prev[i];
            output[i + half] = window[i + half] * -dct_out[size - 1 - i] - prev[i + half];
            prev[i] = window[size - 1 - i] * -dct_out[half - i - 1];
            prev[i + half] = window[half - i - 1] * dct_out[i];
        }

        self.scratch_mdct = dct_out;
        true
    }

    fn dct4(&mut self, input: &[f64], output: &mut [f64]) {
        let tables = trig_tables();
        let shuffle_table = &tables.shuffle[self.mdct_bits];
        let mut sin_table = &tables.sin[self.mdct_bits];
        let mut cos_table = &tables.cos[self.mdct_bits];
        let dct_temp = &mut self.scratch_dct;

        let size = self.mdct_size;
        let last_index = size - 1;
        let half_size = size / 2;

        for i in 0..half_size {
            let i2 = i * 2;
            let a = input[i2];
            let b = input[last_index - i2];
            let sin = sin_table[i];
            let cos = cos_table[i];
            dct_temp[i2] = a * cos + b * sin;
            dct_temp[i2 + 1] = a * sin - b * cos;
        }

        let stage_count = self.mdct_bits - 1;
        for stage in 0..stage_count {
            let block_count = 1usize << stage;
            let block_size_bits = stage_count - stage;
            let block_half_size_bits = block_size_bits - 1;
            let block_size = 1usize << block_size_bits;
            let block_half_size = 1usize << block_half_size_bits;
            sin_table = &tables.sin[block_half_size_bits];
            cos_table = &tables.cos[block_half_size_bits];

            for block in 0..block_count {
                for i in 0..block_half_size {
                    let front_pos = (block * block_size + i) * 2;
                    let back_pos = front_pos + block_size;
                    let a = dct_temp[front_pos] - dct_temp[back_pos];
                    let b = dct_temp[front_pos + 1] - dct_temp[back_pos + 1];
                    let sin = sin_table[i];
                    let cos = cos_table[i];
                    dct_temp[front_pos] += dct_temp[back_pos];
                    dct_temp[front_pos + 1] += dct_temp[back_pos + 1];
                    dct_temp[back_pos] = a * cos + b * sin;
                    dct_temp[back_pos + 1] = a * sin - b * cos;
                }
            }
        }

        for i in 0..size {
            output[i] = dct_temp[shuffle_table[i]] * self.scale;
        }
    }
}
