// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Bit-level helpers and the MSB-first bit reader used by the ATRAC9 decoder.
//!
//! All shifts follow 32-bit two's-complement semantics with the shift count
//! masked to 5 bits, so malformed streams produce the same values as the
//! reference implementation instead of panicking.

/// Reverses the lowest `bit_count` bits of `value`; zero bits reverse to 0
/// (not a 32-bit shift that leaves the whole word).
pub(super) fn bit_reverse32(value: i32, bit_count: i32) -> i32 {
    if bit_count == 0 {
        return 0;
    }
    let reversed = (value as u32).reverse_bits();
    reversed.wrapping_shr((32 - bit_count) as u32) as i32
}

/// Sign-extends the lowest `bits` bits of `value`.
#[inline]
pub(super) fn sign_extend32(value: i32, bits: i32) -> i32 {
    let shift = (32 - bits) as u32;
    value.wrapping_shl(shift).wrapping_shr(shift)
}

#[inline]
pub(super) fn clamp16(value: i32) -> i16 {
    value.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

pub(super) fn get_next_multiple(value: i32, multiple: i32) -> i32 {
    if multiple <= 0 {
        return value;
    }
    if value % multiple == 0 {
        return value;
    }
    value + multiple - value % multiple
}

const MULTIPLY_DE_BRUIJN_BIT_POSITION: [i32; 32] = [
    0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30, 8, 12, 20, 28, 15, 17, 24, 7, 19,
    27, 23, 6, 26, 5, 4, 31,
];

/// Floor of the base-2 logarithm.
pub(super) fn log2(mut value: i32) -> i32 {
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    MULTIPLY_DE_BRUIJN_BIT_POSITION[((value as u32).wrapping_mul(0x07C4_ACDD) >> 27) as usize]
}

/// MSB-first bit reader over a borrowed byte buffer.
pub(super) struct BitReader<'a> {
    buffer: &'a [u8],
    length_bits: i32,
    pub position: i32,
}

impl<'a> BitReader<'a> {
    pub fn new(buffer: &'a [u8]) -> Self {
        let length_bits = (buffer.len().min(i32::MAX as usize / 8) * 8) as i32;
        Self { buffer, length_bits, position: 0 }
    }

    #[inline]
    fn remaining(&self) -> i32 {
        self.length_bits - self.position
    }

    #[inline]
    fn byte(&self, index: i32) -> i32 {
        if index < 0 {
            return 0;
        }
        self.buffer.get(index as usize).copied().unwrap_or(0) as i32
    }

    pub fn read_int(&mut self, bit_count: i32) -> i32 {
        let value = self.peek_int(bit_count);
        self.position = self.position.wrapping_add(bit_count);
        value
    }

    pub fn read_signed_int(&mut self, bit_count: i32) -> i32 {
        let value = self.peek_int(bit_count);
        self.position = self.position.wrapping_add(bit_count);
        sign_extend32(value, bit_count)
    }

    pub fn read_bool(&mut self) -> bool {
        self.read_int(1) == 1
    }

    /// Offset-binary read with a negative bias (the only bias the codec uses).
    pub fn read_offset_binary_negative(&mut self, bit_count: i32) -> i32 {
        let offset = 1i32.wrapping_shl((bit_count - 1) as u32);
        let value = self.peek_int(bit_count).wrapping_sub(offset);
        self.position = self.position.wrapping_add(bit_count);
        value
    }

    pub fn align_position(&mut self, multiple: i32) {
        self.position = get_next_multiple(self.position, multiple);
    }

    pub fn peek_int(&self, bit_count: i32) -> i32 {
        let remaining = self.remaining();
        if bit_count > remaining {
            if self.position >= self.length_bits {
                return 0;
            }
            let extra_bits = bit_count - remaining;
            return self.peek_int_fallback(remaining).wrapping_shl(extra_bits as u32);
        }

        let byte_index = self.position / 8;
        let bit_index = self.position % 8;

        if bit_count <= 9 && remaining >= 16 {
            let mut value = (self.byte(byte_index) << 8) | self.byte(byte_index + 1);
            value &= 0xFFFF >> bit_index;
            value = value.wrapping_shr((16 - bit_count - bit_index) as u32);
            return value;
        }

        if bit_count <= 17 && remaining >= 24 {
            let mut value = (self.byte(byte_index) << 16)
                | (self.byte(byte_index + 1) << 8)
                | self.byte(byte_index + 2);
            value &= 0xFF_FFFF >> bit_index;
            value = value.wrapping_shr((24 - bit_count - bit_index) as u32);
            return value;
        }

        if bit_count <= 25 && remaining >= 32 {
            let mut value = (self.byte(byte_index) << 24)
                | (self.byte(byte_index + 1) << 16)
                | (self.byte(byte_index + 2) << 8)
                | self.byte(byte_index + 3);
            value &= (0xFFFF_FFFFu32 >> bit_index) as i32;
            // Arithmetic shift, as in the reference.
            value = value.wrapping_shr((32 - bit_count - bit_index) as u32);
            return value;
        }

        self.peek_int_fallback(bit_count)
    }

    fn peek_int_fallback(&self, mut bit_count: i32) -> i32 {
        let mut value: i32 = 0;
        let mut byte_index = self.position / 8;
        let mut bit_index = self.position % 8;

        while bit_count > 0 {
            if bit_index >= 8 {
                bit_index = 0;
                byte_index += 1;
            }

            let bits_to_read = bit_count.min(8 - bit_index);
            let mask = 0xFF >> bit_index;
            let current_byte = (mask & self.byte(byte_index)) >> (8 - bit_index - bits_to_read);

            value = value.wrapping_shl(bits_to_read as u32) | current_byte;
            bit_index += bits_to_read;
            bit_count -= bits_to_read;
        }
        value
    }
}
