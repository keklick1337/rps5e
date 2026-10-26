// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The gfx10 inline-constant operand table, shared by every consumer so
//! back-ends cannot drift on constant semantics.

pub const SHARED_BASE: u32 = 235;
pub const SHARED_LIMIT: u32 = 236;
pub const PRIVATE_BASE: u32 = 237;
pub const PRIVATE_LIMIT: u32 = 238;
pub const POPS_EXITING_WAVE_ID: u32 = 239;

/// PS5 FLAT apertures (SH_MEM_BASES): a FLAT address whose high dword is
/// 0x80000000 reaches LDS and one whose high dword is 0x70000000 scratch,
/// at the byte offset in the low dword. Each aperture is a 4 GiB window:
/// an address that carries past its end (high dword 0x80000001) or any
/// other high dword with the same top nibble is an ordinary device address.
/// A title shader builds its LDS flat pointer with `s_brev_b32 s, 1`
/// (0x80000000) as the high dword; none of the shaders dumped from the test
/// titles uses FLAT or reads an aperture.
/// Guest device addresses fit in 40 bits and never reach them.
pub const SHARED_APERTURE_HIGH: u32 = 0x8000_0000;
pub const PRIVATE_APERTURE_HIGH: u32 = 0x7000_0000;

pub fn is_shared_aperture_high(high: u32) -> bool {
    high == SHARED_APERTURE_HIGH
}

pub fn is_private_aperture_high(high: u32) -> bool {
    high == PRIVATE_APERTURE_HIGH
}

pub fn is_aperture(encoded: u32) -> bool {
    (SHARED_BASE..=PRIVATE_LIMIT).contains(&encoded)
}

pub fn is_shared_aperture(encoded: u32) -> bool {
    matches!(encoded, SHARED_BASE | SHARED_LIMIT)
}

/// The 64-bit aperture value: a base has a zero low dword, a limit spans the
/// whole 32-bit offset range. A 32-bit read returns the low dword (0 for a
/// base): only a 64-bit read yields the aperture.
pub fn decode_aperture64(encoded: u32) -> u64 {
    let high = if is_shared_aperture(encoded) { SHARED_APERTURE_HIGH } else { PRIVATE_APERTURE_HIGH };
    let low = if matches!(encoded, SHARED_LIMIT | PRIVATE_LIMIT) { u32::MAX } else { 0 };
    ((high as u64) << 32) | low as u64
}

/// 32-bit value of an encoded constant source, if it has one.
pub fn decode(encoded: u32) -> Option<u32> {
    if is_aperture(encoded) {
        return Some(decode_aperture64(encoded) as u32);
    }
    match encoded {
        // No primitive-ordered pixel shading is modelled.
        POPS_EXITING_WAVE_ID => Some(0),
        // NULL source.
        125 => Some(0),
        128..=192 => Some(encoded - 128),
        193..=208 => Some((-((encoded - 192) as i32)) as u32),
        240 => Some(0.5f32.to_bits()),
        241 => Some((-0.5f32).to_bits()),
        242 => Some(1.0f32.to_bits()),
        243 => Some((-1.0f32).to_bits()),
        244 => Some(2.0f32.to_bits()),
        245 => Some((-2.0f32).to_bits()),
        246 => Some(4.0f32.to_bits()),
        247 => Some((-4.0f32).to_bits()),
        248 => Some((1.0f32 / (2.0 * std::f32::consts::PI)).to_bits()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table() {
        assert_eq!(decode(128), Some(0));
        assert_eq!(decode(192), Some(64));
        assert_eq!(decode(193), Some(u32::MAX));
        assert_eq!(decode(208), Some((-16i32) as u32));
        assert_eq!(decode(242), Some(0x3F80_0000));
        assert_eq!(decode(248), Some(0x3E22_F983));
        assert_eq!(decode(235), Some(0));
        assert_eq!(decode(238), Some(0xFFFF_FFFF));
        assert_eq!(decode_aperture64(236), 0x8000_0000_FFFF_FFFF);
        assert_eq!(decode(251), None);
        assert!(is_shared_aperture_high(0x8000_0000));
        assert!(is_private_aperture_high(0x7000_0000));
        // 4 GiB windows: a carry out of the low dword leaves the aperture.
        assert!(!is_shared_aperture_high(0x8000_0001));
        assert!(!is_shared_aperture_high(0x8123_4567));
        assert!(!is_private_aperture_high(0x7000_0001));
        assert!(!is_private_aperture_high(0x7800_0000));
    }
}
