// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Huffman codebooks (lookup-table decoding) for scale factors and spectra.

use std::sync::OnceLock;

use super::bit_reader::{log2, sign_extend32, BitReader};
use super::huffman_codebooks::*;

pub(super) struct HuffmanCodebook {
    pub bits: &'static [u8],
    pub lookup: Vec<u8>,
    pub value_count: i32,
    pub value_count_power: i32,
    pub value_bits: i32,
    pub value_max: i32,
    pub max_bit_size: i32,
}

impl HuffmanCodebook {
    fn new(codes: &'static [i16], bits: &'static [u8], value_count_power: u8) -> Self {
        let value_count_power = value_count_power as i32;
        let value_bits = log2(codes.len() as i32) >> value_count_power;
        let max_bit_size = bits.iter().copied().max().unwrap_or(0) as i32;

        let table_size = 1usize << max_bit_size;
        let mut lookup = vec![0u8; table_size];
        for (i, &bit_size) in bits.iter().enumerate() {
            if bit_size == 0 {
                continue;
            }
            let unused_bits = max_bit_size - bit_size as i32;
            let start = (codes[i] as i32) << unused_bits;
            let length = 1i32 << unused_bits;
            for j in start..start + length {
                if let Some(slot) = lookup.get_mut(j as usize) {
                    *slot = i as u8;
                }
            }
        }

        Self {
            bits,
            lookup,
            value_count: 1 << value_count_power,
            value_count_power,
            value_bits,
            value_max: 1 << value_bits,
            max_bit_size,
        }
    }

    /// Reads one Huffman-coded value.
    pub fn read_value(&self, reader: &mut BitReader<'_>, signed: bool) -> i32 {
        let code = reader.peek_int(self.max_bit_size);
        let value = self.lookup.get(code as u32 as usize).copied().unwrap_or(0);
        let bits = self.bits.get(value as usize).copied().unwrap_or(0) as i32;
        reader.position = reader.position.wrapping_add(bits);
        if signed {
            sign_extend32(value as i32, self.value_bits)
        } else {
            value as i32
        }
    }
}

pub(super) struct HuffmanTables {
    /// Indexed by bit length (1..=6); index 0 is empty.
    pub scale_factors_unsigned: Vec<Option<HuffmanCodebook>>,
    /// Indexed by bit length (2..=5); indices 0 and 1 are empty.
    pub scale_factors_signed: Vec<Option<HuffmanCodebook>>,
    /// `[codebook set][precision][codebook index]`.
    pub spectrum: [Vec<Vec<Option<HuffmanCodebook>>>; 2],
}

type CodebookSource = Option<(&'static [i16], &'static [u8])>;

fn generate(sources: &[CodebookSource], group_sizes: &[u8]) -> Vec<Option<HuffmanCodebook>> {
    sources
        .iter()
        .enumerate()
        .map(|(i, src)| src.map(|(codes, bits)| HuffmanCodebook::new(codes, bits, group_sizes[i])))
        .collect()
}

macro_rules! cb {
    ($codes:ident, $bits:ident) => {
        Some((&$codes[..], &$bits[..]))
    };
}

const SCALE_FACTORS_GROUP_SIZES: [u8; 7] = [0, 0, 0, 0, 0, 0, 0];

const SPECTRUM_A_GROUP_SIZES: [[u8; 4]; 8] = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 2, 2, 2],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
];

const SPECTRUM_B_GROUP_SIZES: [[u8; 4]; 8] = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 2, 2, 2],
    [0, 1, 1, 1],
    [0, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
];

pub(super) fn huffman_tables() -> &'static HuffmanTables {
    static TABLES: OnceLock<HuffmanTables> = OnceLock::new();
    TABLES.get_or_init(build_tables)
}

fn build_tables() -> HuffmanTables {
    let sf_unsigned: [CodebookSource; 7] = [
        None,
        cb!(SCALE_FACTORS_A_1_CODES, SCALE_FACTORS_A_1_BITS),
        cb!(SCALE_FACTORS_A_2_CODES, SCALE_FACTORS_A_2_BITS),
        cb!(SCALE_FACTORS_A_3_CODES, SCALE_FACTORS_A_3_BITS),
        cb!(SCALE_FACTORS_A_4_CODES, SCALE_FACTORS_A_4_BITS),
        cb!(SCALE_FACTORS_A_5_CODES, SCALE_FACTORS_A_5_BITS),
        cb!(SCALE_FACTORS_A_6_CODES, SCALE_FACTORS_A_6_BITS),
    ];
    let sf_signed: [CodebookSource; 6] = [
        None,
        None,
        cb!(SCALE_FACTORS_B_2_CODES, SCALE_FACTORS_B_2_BITS),
        cb!(SCALE_FACTORS_B_3_CODES, SCALE_FACTORS_B_3_BITS),
        cb!(SCALE_FACTORS_B_4_CODES, SCALE_FACTORS_B_4_BITS),
        cb!(SCALE_FACTORS_B_5_CODES, SCALE_FACTORS_B_5_BITS),
    ];

    let spectrum_a: [Option<[CodebookSource; 4]>; 8] = [
        None,
        None,
        Some([
            cb!(SPECTRUM_A_21_CODES, SPECTRUM_A_21_BITS),
            cb!(SPECTRUM_A_22_CODES, SPECTRUM_A_22_BITS),
            cb!(SPECTRUM_A_23_CODES, SPECTRUM_A_23_BITS),
            cb!(SPECTRUM_A_24_CODES, SPECTRUM_A_24_BITS),
        ]),
        Some([
            cb!(SPECTRUM_A_31_CODES, SPECTRUM_A_31_BITS),
            cb!(SPECTRUM_A_32_CODES, SPECTRUM_A_32_BITS),
            cb!(SPECTRUM_A_33_CODES, SPECTRUM_A_33_BITS),
            cb!(SPECTRUM_A_34_CODES, SPECTRUM_A_34_BITS),
        ]),
        Some([
            cb!(SPECTRUM_A_41_CODES, SPECTRUM_A_41_BITS),
            cb!(SPECTRUM_A_42_CODES, SPECTRUM_A_42_BITS),
            cb!(SPECTRUM_A_43_CODES, SPECTRUM_A_43_BITS),
            cb!(SPECTRUM_A_44_CODES, SPECTRUM_A_44_BITS),
        ]),
        Some([
            cb!(SPECTRUM_A_51_CODES, SPECTRUM_A_51_BITS),
            cb!(SPECTRUM_A_52_CODES, SPECTRUM_A_52_BITS),
            cb!(SPECTRUM_A_53_CODES, SPECTRUM_A_53_BITS),
            cb!(SPECTRUM_A_54_CODES, SPECTRUM_A_54_BITS),
        ]),
        Some([
            cb!(SPECTRUM_A_61_CODES, SPECTRUM_A_61_BITS),
            cb!(SPECTRUM_A_62_CODES, SPECTRUM_A_62_BITS),
            cb!(SPECTRUM_A_63_CODES, SPECTRUM_A_63_BITS),
            cb!(SPECTRUM_A_64_CODES, SPECTRUM_A_64_BITS),
        ]),
        Some([
            cb!(SPECTRUM_A_71_CODES, SPECTRUM_A_71_BITS),
            cb!(SPECTRUM_A_72_CODES, SPECTRUM_A_72_BITS),
            cb!(SPECTRUM_A_73_CODES, SPECTRUM_A_73_BITS),
            cb!(SPECTRUM_A_74_CODES, SPECTRUM_A_74_BITS),
        ]),
    ];

    let spectrum_b: [Option<[CodebookSource; 4]>; 8] = [
        None,
        None,
        Some([
            None,
            cb!(SPECTRUM_B_22_CODES, SPECTRUM_B_22_BITS),
            cb!(SPECTRUM_B_23_CODES, SPECTRUM_B_23_BITS),
            cb!(SPECTRUM_B_24_CODES, SPECTRUM_B_24_BITS),
        ]),
        Some([
            None,
            cb!(SPECTRUM_B_32_CODES, SPECTRUM_B_32_BITS),
            cb!(SPECTRUM_B_33_CODES, SPECTRUM_B_33_BITS),
            cb!(SPECTRUM_B_34_CODES, SPECTRUM_B_34_BITS),
        ]),
        Some([
            None,
            cb!(SPECTRUM_B_42_CODES, SPECTRUM_B_42_BITS),
            cb!(SPECTRUM_B_43_CODES, SPECTRUM_B_43_BITS),
            cb!(SPECTRUM_B_44_CODES, SPECTRUM_B_44_BITS),
        ]),
        Some([
            None,
            cb!(SPECTRUM_B_52_CODES, SPECTRUM_B_52_BITS),
            cb!(SPECTRUM_B_53_CODES, SPECTRUM_B_53_BITS),
            cb!(SPECTRUM_B_54_CODES, SPECTRUM_B_54_BITS),
        ]),
        Some([
            None,
            cb!(SPECTRUM_B_62_CODES, SPECTRUM_B_62_BITS),
            cb!(SPECTRUM_B_63_CODES, SPECTRUM_B_63_BITS),
            cb!(SPECTRUM_B_64_CODES, SPECTRUM_B_64_BITS),
        ]),
        Some([
            None,
            cb!(SPECTRUM_B_72_CODES, SPECTRUM_B_72_BITS),
            cb!(SPECTRUM_B_73_CODES, SPECTRUM_B_73_BITS),
            cb!(SPECTRUM_B_74_CODES, SPECTRUM_B_74_BITS),
        ]),
    ];

    let spectrum_set = |sources: &[Option<[CodebookSource; 4]>; 8], groups: &[[u8; 4]; 8]| {
        sources
            .iter()
            .enumerate()
            .map(|(p, src)| match src {
                Some(row) => generate(row, &groups[p]),
                None => Vec::new(),
            })
            .collect::<Vec<_>>()
    };

    HuffmanTables {
        scale_factors_unsigned: generate(&sf_unsigned, &SCALE_FACTORS_GROUP_SIZES),
        scale_factors_signed: generate(&sf_signed, &SCALE_FACTORS_GROUP_SIZES),
        spectrum: [
            spectrum_set(&spectrum_a, &SPECTRUM_A_GROUP_SIZES),
            spectrum_set(&spectrum_b, &SPECTRUM_B_GROUP_SIZES),
        ],
    }
}

impl HuffmanTables {
    pub fn spectrum_codebook(&self, set: i32, precision: i32, index: i32) -> Option<&HuffmanCodebook> {
        self.spectrum
            .get(usize::try_from(set).ok()?)?
            .get(usize::try_from(precision).ok()?)?
            .get(usize::try_from(index).ok()?)?
            .as_ref()
    }

    pub fn scale_factors_unsigned(&self, bit_length: i32) -> Option<&HuffmanCodebook> {
        self.scale_factors_unsigned.get(usize::try_from(bit_length).ok()?)?.as_ref()
    }

    pub fn scale_factors_signed(&self, bit_length: i32) -> Option<&HuffmanCodebook> {
        self.scale_factors_signed.get(usize::try_from(bit_length).ok()?)?.as_ref()
    }
}
