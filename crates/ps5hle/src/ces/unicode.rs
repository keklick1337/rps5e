// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! One-character conversions between the Unicode encodings: UTF-8, UCS-2,
//! UTF-16 and UTF-32 (native order, which is little-endian, or an explicit
//! byte order), and EUC-JP to UTF-8.

use super::{
    decode_mbcs, profile_kind, read_units, stack_arg0, ERROR_DST_BUFFER_END, ERROR_INVALID_DST_BUFFER,
    ERROR_INVALID_ENCODE, ERROR_INVALID_PARAMETER, ERROR_INVALID_SRC_BUFFER, ERROR_SRC_BUFFER_END,
    ERROR_UNASSIGNED_CODE,
};
use crate::context::CpuContext;
use crate::registry::{HleCallback, HleResult};

fn is_scalar(code: u32) -> bool {
    code <= 0x10FFFF && !(0xD800..=0xDFFF).contains(&code)
}

fn read_byte(ctx: &CpuContext, addr: u64) -> Result<u8, HleResult> {
    ctx.read_bytes(addr, 1).map(|b| b[0]).ok_or(ERROR_INVALID_SRC_BUFFER)
}

/// Decode one UTF-8 character of at most `available` bytes: (code,
/// length). The second byte is range-checked against overlong forms and
/// surrogates as it is read.
fn decode_utf8(ctx: &CpuContext, source: u64, available: u32) -> Result<(u32, u32), HleResult> {
    if source == 0 {
        return Err(ERROR_INVALID_SRC_BUFFER);
    }
    if available == 0 {
        return Err(ERROR_SRC_BUFFER_END);
    }
    let lead = read_byte(ctx, source)?;
    let (mut code, length) = match lead {
        0..=0x7F => return Ok((lead as u32, 1)),
        0xC2..=0xDF => (lead as u32 & 0x1F, 2),
        0xE0..=0xEF => (lead as u32 & 0x0F, 3),
        0xF0..=0xF4 => (lead as u32 & 0x07, 4),
        _ => return Err(ERROR_INVALID_ENCODE),
    };
    for i in 1..length {
        if i >= available {
            return Err(ERROR_SRC_BUFFER_END);
        }
        let next = read_byte(ctx, source + i as u64)?;
        if !(0x80..=0xBF).contains(&next) {
            return Err(ERROR_INVALID_ENCODE);
        }
        if i == 1
            && ((lead == 0xE0 && next < 0xA0)
                || (lead == 0xED && next > 0x9F)
                || (lead == 0xF0 && next < 0x90)
                || (lead == 0xF4 && next > 0x8F))
        {
            return Err(ERROR_INVALID_ENCODE);
        }
        code = code << 6 | (next as u32 & 0x3F);
    }
    Ok((code, length))
}

/// Encode a scalar value as UTF-8 into `target` (room for `capacity`
/// bytes), storing the length at `target_length` when given.
fn encode_utf8(ctx: &CpuContext, code: u32, target: u64, capacity: u32, target_length: u64) -> HleResult {
    if !is_scalar(code) {
        return ERROR_INVALID_ENCODE;
    }
    if target == 0 {
        return ERROR_INVALID_DST_BUFFER;
    }
    let bytes = super::encode_utf8(code);
    if (capacity as usize) < bytes.len() {
        return ERROR_DST_BUFFER_END;
    }
    if !ctx.write_bytes(target, &bytes) {
        return ERROR_INVALID_DST_BUFFER;
    }
    if target_length != 0 {
        ctx.write_u32(target_length, bytes.len() as u32);
    }
    0
}

fn unit_bytes<const N: usize>(code: u32, big_endian: bool) -> [u8; N] {
    let mut b = [0u8; N];
    for (i, byte) in b.iter_mut().enumerate() {
        let shift = 8 * if big_endian { N - 1 - i } else { i };
        *byte = (code >> shift) as u8;
    }
    b
}

fn load_unit<const N: usize>(ctx: &CpuContext, addr: u64, big_endian: bool) -> Result<u32, HleResult> {
    let b = ctx.read_bytes(addr, N).ok_or(ERROR_INVALID_SRC_BUFFER)?;
    let mut code = 0u32;
    for (i, &byte) in b.iter().enumerate() {
        code |= (byte as u32) << (8 * if big_endian { N - 1 - i } else { i });
    }
    Ok(code)
}

fn status(r: Result<(), HleResult>) -> HleResult {
    r.err().unwrap_or(0)
}

/// (const u8* source, u32 available, u32* sourceLength, u32* target)
fn utf8_to_utf32(ctx: &CpuContext, big_endian: bool) -> HleResult {
    status((|| {
        let (source, available, source_length, target) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx());
        let (code, length) = decode_utf8(ctx, source, available)?;
        if target == 0 || !ctx.write_bytes(target, &unit_bytes::<4>(code, big_endian)) {
            return Err(ERROR_INVALID_DST_BUFFER);
        }
        if source_length != 0 {
            ctx.write_u32(source_length, length);
        }
        Ok(())
    })())
}

/// (const u8* source, u32 available, u32* sourceLength, u16* target,
/// u32 capacity, u32* targetLength): one or two UTF-16 units.
fn utf8_to_utf16(ctx: &CpuContext, big_endian: bool) -> HleResult {
    status((|| {
        let (source, available, source_length, target) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx());
        let (capacity, target_length) = (ctx.r8() as u32, ctx.r9());
        let (code, length) = decode_utf8(ctx, source, available)?;
        if target == 0 {
            return Err(ERROR_INVALID_DST_BUFFER);
        }
        let units: Vec<u32> = if code < 0x10000 {
            vec![code]
        } else {
            let c = code - 0x10000;
            vec![0xD800 | c >> 10, 0xDC00 | (c & 0x3FF)]
        };
        if (capacity as usize) < units.len() {
            return Err(ERROR_DST_BUFFER_END);
        }
        let bytes: Vec<u8> = units.iter().flat_map(|&u| unit_bytes::<2>(u, big_endian)).collect();
        if !ctx.write_bytes(target, &bytes) {
            return Err(ERROR_INVALID_DST_BUFFER);
        }
        if source_length != 0 {
            ctx.write_u32(source_length, length);
        }
        if target_length != 0 {
            ctx.write_u32(target_length, units.len() as u32);
        }
        Ok(())
    })())
}

/// (const u16* source, u32 available, u32* sourceLength, u8* target,
/// u32 capacity, u32* targetLength): one character of one unit or a
/// surrogate pair.
fn utf16_to_utf8(ctx: &CpuContext, big_endian: bool) -> HleResult {
    status((|| {
        let (source, available, source_length, target) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx());
        let (capacity, target_length) = (ctx.r8() as u32, ctx.r9());
        if source == 0 {
            return Err(ERROR_INVALID_SRC_BUFFER);
        }
        if available == 0 {
            return Err(ERROR_SRC_BUFFER_END);
        }
        let mut code = load_unit::<2>(ctx, source, big_endian)?;
        let mut units = 1;
        if (0xDC00..=0xDFFF).contains(&code) {
            return Err(ERROR_INVALID_ENCODE);
        }
        if (0xD800..=0xDBFF).contains(&code) {
            if available < 2 {
                return Err(ERROR_SRC_BUFFER_END);
            }
            let low = load_unit::<2>(ctx, source + 2, big_endian)?;
            if !(0xDC00..=0xDFFF).contains(&low) {
                return Err(ERROR_INVALID_ENCODE);
            }
            code = 0x10000 + ((code - 0xD800) << 10) + low - 0xDC00;
            units = 2;
        }
        match encode_utf8(ctx, code, target, capacity, target_length) {
            0 => {}
            e => return Err(e),
        }
        if source_length != 0 {
            ctx.write_u32(source_length, units);
        }
        Ok(())
    })())
}

/// (const u32* source, u8* target, u32 capacity, u32* targetLength)
fn utf32_to_utf8(ctx: &CpuContext, big_endian: bool) -> HleResult {
    let source = ctx.rdi();
    if source == 0 {
        return ERROR_INVALID_SRC_BUFFER;
    }
    match load_unit::<4>(ctx, source, big_endian) {
        Ok(code) => encode_utf8(ctx, code, ctx.rsi(), ctx.rdx() as u32, ctx.rcx()),
        Err(e) => e,
    }
}

/// sceCesUcs2ToUtf8(u16 source, u8* target, u32 capacity, u32* targetLength)
fn cb_ucs2_to_utf8(ctx: &mut CpuContext) -> HleResult {
    encode_utf8(ctx, ctx.rdi() as u16 as u32, ctx.rsi(), ctx.rdx() as u32, ctx.rcx())
}

/// sceCesUtf32ToUtf8(u32 source, u8* target, u32 capacity, u32* targetLength)
fn cb_utf32_to_utf8(ctx: &mut CpuContext) -> HleResult {
    encode_utf8(ctx, ctx.rdi() as u32, ctx.rsi(), ctx.rdx() as u32, ctx.rcx())
}

/// sceCesUtf8ToUcs2(const u8* source, u32 available, u32* sourceLength,
/// u16* target): a character beyond the BMP is UNASSIGNED_CODE.
fn cb_utf8_to_ucs2(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        let (source, available, source_length, target) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx());
        let (code, length) = decode_utf8(ctx, source, available)?;
        if target == 0 {
            return Err(ERROR_INVALID_DST_BUFFER);
        }
        if code > 0xFFFF {
            return Err(ERROR_UNASSIGNED_CODE);
        }
        if !ctx.write_bytes(target, &(code as u16).to_le_bytes()) {
            return Err(ERROR_INVALID_DST_BUFFER);
        }
        if source_length != 0 {
            ctx.write_u32(source_length, length);
        }
        Ok(())
    })())
}

fn cb_utf8_to_utf32(ctx: &mut CpuContext) -> HleResult {
    utf8_to_utf32(ctx, false)
}

fn cb_utf8_to_utf32be(ctx: &mut CpuContext) -> HleResult {
    utf8_to_utf32(ctx, true)
}

fn cb_utf8_to_utf32le(ctx: &mut CpuContext) -> HleResult {
    utf8_to_utf32(ctx, false)
}

fn cb_utf8_to_utf16(ctx: &mut CpuContext) -> HleResult {
    utf8_to_utf16(ctx, false)
}

fn cb_utf8_to_utf16be(ctx: &mut CpuContext) -> HleResult {
    utf8_to_utf16(ctx, true)
}

fn cb_utf8_to_utf16le(ctx: &mut CpuContext) -> HleResult {
    utf8_to_utf16(ctx, false)
}

fn cb_utf16_to_utf8(ctx: &mut CpuContext) -> HleResult {
    utf16_to_utf8(ctx, false)
}

fn cb_utf16be_to_utf8(ctx: &mut CpuContext) -> HleResult {
    utf16_to_utf8(ctx, true)
}

fn cb_utf16le_to_utf8(ctx: &mut CpuContext) -> HleResult {
    utf16_to_utf8(ctx, false)
}

fn cb_utf32be_to_utf8(ctx: &mut CpuContext) -> HleResult {
    utf32_to_utf8(ctx, true)
}

fn cb_utf32le_to_utf8(ctx: &mut CpuContext) -> HleResult {
    utf32_to_utf8(ctx, false)
}

/// sceCesEucJpToUtf8(profile, const u8* euc, u32 eucMax, u32* eucLength,
/// u8* utf8, u32 utf8Max, u32* utf8Length): one character.
fn cb_euc_jp_to_utf8(ctx: &mut CpuContext) -> HleResult {
    let (profile, euc, max, euc_length) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u32, ctx.rcx());
    let (utf8, utf8_max, utf8_length) = (ctx.r8(), ctx.r9() as u32, stack_arg0(ctx));
    let Some(profile) = profile_kind(ctx, profile).filter(|p| p.euc_jp()) else {
        return ERROR_INVALID_PARAMETER;
    };
    if euc == 0 {
        return ERROR_INVALID_SRC_BUFFER;
    }
    if max == 0 {
        return ERROR_SRC_BUFFER_END;
    }
    if utf8 == 0 {
        return ERROR_INVALID_DST_BUFFER;
    }
    let bytes = read_units(ctx, euc, 1, max.min(3)).unwrap_or_default();
    let (length, code) = match decode_mbcs(profile, |i| bytes.get(i).copied().unwrap_or(0) as u8, max) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let encoded = super::encode_utf8(code);
    if (utf8_max as usize) < encoded.len() {
        return ERROR_DST_BUFFER_END;
    }
    if !ctx.write_bytes(utf8, &encoded) {
        return ERROR_INVALID_DST_BUFFER;
    }
    if euc_length != 0 {
        ctx.write_u32(euc_length, length);
    }
    if utf8_length != 0 {
        ctx.write_u32(utf8_length, encoded.len() as u32);
    }
    0
}

pub(super) const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("5k6oMFx-ihY", "sceCesUcs2ToUtf8", cb_ucs2_to_utf8),
    ("2OeA+kH1s1g", "sceCesUtf8ToUcs2", cb_utf8_to_ucs2),
    ("qICXjhb1exk", "sceCesUtf8ToUtf32", cb_utf8_to_utf32),
    ("lQJnJoUTY4w", "sceCesUtf8ToUtf32be", cb_utf8_to_utf32be),
    ("1spdlCFoiww", "sceCesUtf8ToUtf32le", cb_utf8_to_utf32le),
    ("sRYcNgFGq+0", "sceCesUtf8ToUtf16", cb_utf8_to_utf16),
    ("wFbJ1paXfPg", "sceCesUtf8ToUtf16be", cb_utf8_to_utf16be),
    ("9DeMz0wOHWc", "sceCesUtf8ToUtf16le", cb_utf8_to_utf16le),
    ("rVZ2PrNNKJg", "sceCesUtf16ToUtf8", cb_utf16_to_utf8),
    ("dDoDH0qtexE", "sceCesUtf16beToUtf8", cb_utf16be_to_utf8),
    ("b8upshFAom0", "sceCesUtf16leToUtf8", cb_utf16le_to_utf8),
    ("4cBWKt8i5c4", "sceCesUtf32ToUtf8", cb_utf32_to_utf8),
    ("yMsd3ZjniL0", "sceCesUtf32beToUtf8", cb_utf32be_to_utf8),
    ("cKjaZxqfgQY", "sceCesUtf32leToUtf8", cb_utf32le_to_utf8),
    ("zPdlo9OiO0g", "sceCesEucJpToUtf8", cb_euc_jp_to_utf8),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    const BASE: u64 = 0x5D5A_0000_0000;
    const SRC: u64 = BASE;
    const DST: u64 = BASE + 0x100;
    const CONSUMED: u64 = BASE + 0x200;
    const PRODUCED: u64 = BASE + 0x208;
    const SCALAR: u64 = BASE + 0x210;
    const STACK: u64 = BASE + 0x300;
    const SHEET: u64 = BASE + 0x1000;
    const SENTINEL: u32 = 0xA5A5_A5A5;

    fn call(ctx: &mut CpuContext, f: HleCallback, args: &[u64]) -> HleResult {
        for (&r, &v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
            ctx.set(r, v);
        }
        if let Some(&v) = args.get(6) {
            ctx.write_u64(STACK + 8, v);
        }
        ctx.set(Rsp, STACK);
        f(ctx)
    }

    fn reset(ctx: &CpuContext) {
        ctx.write_bytes(DST, &[0xA5; 16]);
        for a in [CONSUMED, PRODUCED, SCALAR] {
            ctx.write_u32(a, SENTINEL);
        }
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// Reference GuestCesUnicode cases: byte orders, round trips over the
    /// Unicode scalars, and the error order of every conversion.
    #[test]
    fn unicode_scalar_conversions() {
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x2000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let emoji = [0xF0, 0x9F, 0x98, 0x80, b'X'];
        ctx.write_bytes(SRC, &emoji);
        // Byte orders.
        for (f, expected) in [
            (cb_utf8_to_utf16be as HleCallback, vec![0xD8, 0x3D, 0xDE, 0x00]),
            (cb_utf8_to_utf16le, vec![0x3D, 0xD8, 0x00, 0xDE]),
            (cb_utf8_to_utf16, vec![0x3D, 0xD8, 0x00, 0xDE]),
        ] {
            reset(&ctx);
            assert_eq!(call(&mut ctx, f, &[SRC, 4, CONSUMED, DST, 2, PRODUCED]), 0);
            assert_eq!(ctx.read_bytes(DST, 4).unwrap(), expected);
            assert_eq!((ctx.read_u32(CONSUMED), ctx.read_u32(PRODUCED)), (Some(4), Some(2)));
        }
        for (f, expected) in [
            (cb_utf8_to_utf32be as HleCallback, vec![0x00, 0x01, 0xF6, 0x00]),
            (cb_utf8_to_utf32le, vec![0x00, 0xF6, 0x01, 0x00]),
            (cb_utf8_to_utf32, vec![0x00, 0xF6, 0x01, 0x00]),
        ] {
            assert_eq!(call(&mut ctx, f, &[SRC, 5, 0, DST]), 0);
            assert_eq!(ctx.read_bytes(DST, 4).unwrap(), expected);
        }
        // Round trips through every form, scalars sampled across the range.
        let mut code = 0u32;
        while code <= 0x10FFFF {
            if !(0xD800..=0xDFFF).contains(&code) {
                reset(&ctx);
                assert_eq!(call(&mut ctx, cb_utf32_to_utf8, &[code as u64, SRC, 4, PRODUCED]), 0);
                let n = ctx.read_u32(PRODUCED).unwrap();
                let expected = char::from_u32(code).unwrap().to_string().into_bytes();
                assert_eq!(ctx.read_bytes(SRC, n as usize).unwrap(), expected);
                assert_eq!(call(&mut ctx, cb_utf8_to_utf32, &[SRC, n as u64, CONSUMED, SCALAR]), 0);
                assert_eq!((ctx.read_u32(SCALAR), ctx.read_u32(CONSUMED)), (Some(code), Some(n)));
                let ucs = call(&mut ctx, cb_utf8_to_ucs2, &[SRC, n as u64, CONSUMED, DST]);
                if code <= 0xFFFF {
                    assert_eq!(ucs, 0);
                    assert_eq!(call(&mut ctx, cb_ucs2_to_utf8, &[code as u64, DST + 8, 4, PRODUCED]), 0);
                    assert_eq!(ctx.read_bytes(DST + 8, n as usize).unwrap(), expected);
                } else {
                    assert_eq!(ucs, ERROR_UNASSIGNED_CODE);
                }
                let pairs16: [(HleCallback, HleCallback); 2] =
                    [(cb_utf8_to_utf16be, cb_utf16be_to_utf8), (cb_utf8_to_utf16le, cb_utf16le_to_utf8)];
                for (to16, from16) in pairs16 {
                    assert_eq!(call(&mut ctx, to16, &[SRC, n as u64, CONSUMED, DST, 2, PRODUCED]), 0);
                    let units = ctx.read_u32(PRODUCED).unwrap();
                    assert_eq!(units, if code <= 0xFFFF { 1 } else { 2 });
                    assert_eq!(call(&mut ctx, from16, &[DST, units as u64, CONSUMED, DST + 8, 4, PRODUCED]), 0);
                    assert_eq!((ctx.read_u32(CONSUMED), ctx.read_u32(PRODUCED)), (Some(units), Some(n)));
                    assert_eq!(ctx.read_bytes(DST + 8, n as usize).unwrap(), expected);
                }
                let pairs32: [(HleCallback, HleCallback); 2] =
                    [(cb_utf8_to_utf32be, cb_utf32be_to_utf8), (cb_utf8_to_utf32le, cb_utf32le_to_utf8)];
                for (to32, from32) in pairs32 {
                    assert_eq!(call(&mut ctx, to32, &[SRC, n as u64, CONSUMED, DST]), 0);
                    assert_eq!(call(&mut ctx, from32, &[DST, DST + 8, 4, PRODUCED]), 0);
                    assert_eq!(ctx.read_bytes(DST + 8, n as usize).unwrap(), expected);
                }
            }
            code += if code < 0x800 { 1 } else { 97 };
        }
        // Too small a destination or source changes nothing.
        ctx.write_bytes(SRC, &emoji);
        for cap in 0..4u64 {
            reset(&ctx);
            assert_eq!(call(&mut ctx, cb_utf32_to_utf8, &[0x1F600, DST, cap, PRODUCED]), ERROR_DST_BUFFER_END);
            assert_eq!(call(&mut ctx, cb_utf8_to_utf32, &[SRC, cap, CONSUMED, SCALAR]), ERROR_SRC_BUFFER_END);
            assert_eq!([ctx.read_u32(PRODUCED), ctx.read_u32(CONSUMED), ctx.read_u32(SCALAR)], [Some(SENTINEL); 3]);
            assert_eq!(ctx.read_bytes(DST, 4).unwrap(), vec![0xA5; 4]);
        }
        for cap in 0..2u64 {
            assert_eq!(call(&mut ctx, cb_utf8_to_utf16, &[SRC, 4, CONSUMED, DST, cap, PRODUCED]), ERROR_DST_BUFFER_END);
            assert_eq!(ctx.read_u32(CONSUMED), Some(SENTINEL));
        }
        let invalid: [&[u8]; 12] = [
            &[0x80],
            &[0xC0, 0x80],
            &[0xC1, 0xBF],
            &[0xC2, 0x7F],
            &[0xE0, 0x9F],
            &[0xED, 0xA0],
            &[0xF0, 0x8F],
            &[0xF4, 0x90],
            &[0xF5],
            &[0xFF],
            &[0xE2, 0x82, 0x41],
            &[0xF0, 0x9F, 0x98, 0x41],
        ];
        for bytes in invalid {
            ctx.write_bytes(SRC, bytes);
            let n = bytes.len() as u64;
            assert_eq!(call(&mut ctx, cb_utf8_to_utf32, &[SRC, n, CONSUMED, SCALAR]), ERROR_INVALID_ENCODE, "{bytes:x?}");
            assert_eq!(call(&mut ctx, cb_utf8_to_utf16be, &[SRC, n, CONSUMED, DST, 2, PRODUCED]), ERROR_INVALID_ENCODE);
            assert_eq!([ctx.read_u32(PRODUCED), ctx.read_u32(CONSUMED), ctx.read_u32(SCALAR)], [Some(SENTINEL); 3]);
        }
        for code in [0xD800u64, 0xDBFF, 0xDC00, 0xDFFF, 0x110000, 0xFFFF_FFFF] {
            assert_eq!(call(&mut ctx, cb_utf32_to_utf8, &[code, DST, 4, PRODUCED]), ERROR_INVALID_ENCODE);
            if code <= 0xFFFF {
                assert_eq!(call(&mut ctx, cb_ucs2_to_utf8, &[code, DST, 4, PRODUCED]), ERROR_INVALID_ENCODE);
            }
        }
        ctx.write_bytes(SRC, &[0x00, 0xD8, 0x41, 0x00]);
        assert_eq!(call(&mut ctx, cb_utf16_to_utf8, &[SRC, 1, CONSUMED, DST, 4, PRODUCED]), ERROR_SRC_BUFFER_END);
        assert_eq!(call(&mut ctx, cb_utf16_to_utf8, &[SRC, 2, CONSUMED, DST, 4, PRODUCED]), ERROR_INVALID_ENCODE);
        assert_eq!(call(&mut ctx, cb_utf16_to_utf8, &[SRC, 0, CONSUMED, DST, 4, PRODUCED]), ERROR_SRC_BUFFER_END);
        ctx.write_bytes(SRC + 8, &[0x00, 0xDC]);
        assert_eq!(call(&mut ctx, cb_utf16_to_utf8, &[SRC + 8, 1, CONSUMED, DST, 4, PRODUCED]), ERROR_INVALID_ENCODE);
        assert_eq!([ctx.read_u32(PRODUCED), ctx.read_u32(CONSUMED)], [Some(SENTINEL); 2]);
        assert_eq!(call(&mut ctx, cb_utf16_to_utf8, &[0, 1, CONSUMED, DST, 4, PRODUCED]), ERROR_INVALID_SRC_BUFFER);
        assert_eq!(call(&mut ctx, cb_utf8_to_utf32, &[0, 4, CONSUMED, SCALAR]), ERROR_INVALID_SRC_BUFFER);
        ctx.write_bytes(SRC, &emoji);
        assert_eq!(call(&mut ctx, cb_utf8_to_utf32, &[SRC, 4, CONSUMED, 0]), ERROR_INVALID_DST_BUFFER);
        assert_eq!(call(&mut ctx, cb_utf8_to_ucs2, &[SRC, 4, CONSUMED, 0]), ERROR_INVALID_DST_BUFFER);
        assert_eq!(call(&mut ctx, cb_utf8_to_utf16, &[SRC, 4, CONSUMED, 0, 2, PRODUCED]), ERROR_INVALID_DST_BUFFER);
        assert_eq!(call(&mut ctx, cb_utf32be_to_utf8, &[0, DST, 4, PRODUCED]), ERROR_INVALID_SRC_BUFFER);
        assert_eq!(call(&mut ctx, cb_utf32_to_utf8, &[0, 0, 4, PRODUCED]), ERROR_INVALID_DST_BUFFER);
        assert_eq!(call(&mut ctx, cb_ucs2_to_utf8, &[0, DST, 1, 0]), 0);
        assert_eq!(ctx.read_bytes(DST, 1).unwrap(), vec![0]);
    }

    /// Reference GuestCesUnicode EUC-JP cases (JIS X 0208 and half-width
    /// katakana; the three-byte JIS X 0212 form is not decoded here).
    #[test]
    fn euc_jp_to_utf8() {
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x2000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let profile = call(&mut ctx, super::super::cb_ucs_profile_init_euc_jp_cp51932, &[SHEET]) as u64;
        assert_eq!(profile, SHEET);
        // U+3042 HIRAGANA LETTER A, half-width katakana U+FF71, ASCII.
        for (bytes, expected) in [(&b"\xA4\xA2"[..], "\u{3042}"), (&b"\x8E\xB1"[..], "\u{FF71}"), (&b"A"[..], "A")] {
            let expected = expected.as_bytes();
            let n = bytes.len() as u64;
            ctx.write_bytes(SRC, bytes);
            for cap in 0..expected.len() as u64 {
                reset(&ctx);
                let r = call(&mut ctx, cb_euc_jp_to_utf8, &[profile, SRC, n, CONSUMED, DST, cap, PRODUCED]);
                assert_eq!(r, ERROR_DST_BUFFER_END);
                assert_eq!([ctx.read_u32(PRODUCED), ctx.read_u32(CONSUMED)], [Some(SENTINEL); 2]);
            }
            reset(&ctx);
            let r = call(&mut ctx, cb_euc_jp_to_utf8, &[profile, SRC, n, CONSUMED, DST, expected.len() as u64, PRODUCED]);
            assert_eq!(r, 0);
            assert_eq!(ctx.read_u32(CONSUMED), Some(n as u32));
            assert_eq!(ctx.read_u32(PRODUCED), Some(expected.len() as u32));
            assert_eq!(ctx.read_bytes(DST, expected.len() + 1).unwrap(), [expected, &[0xA5]].concat());
        }
        ctx.write_bytes(SRC, b"\xA4\xA2X");
        assert_eq!(call(&mut ctx, cb_euc_jp_to_utf8, &[0, SRC, 3, CONSUMED, DST, 4, PRODUCED]), ERROR_INVALID_PARAMETER);
        assert_eq!(call(&mut ctx, cb_euc_jp_to_utf8, &[profile, 0, 3, CONSUMED, DST, 4, PRODUCED]), ERROR_INVALID_SRC_BUFFER);
        assert_eq!(call(&mut ctx, cb_euc_jp_to_utf8, &[profile, SRC, 0, CONSUMED, DST, 4, PRODUCED]), ERROR_SRC_BUFFER_END);
        assert_eq!(call(&mut ctx, cb_euc_jp_to_utf8, &[profile, SRC, 3, CONSUMED, 0, 4, PRODUCED]), ERROR_INVALID_DST_BUFFER);
        assert_eq!(call(&mut ctx, cb_euc_jp_to_utf8, &[profile, SRC, 1, CONSUMED, DST, 4, PRODUCED]), ERROR_SRC_BUFFER_END);
        ctx.write_bytes(SRC, b"\x8E\x41");
        reset(&ctx);
        assert_eq!(call(&mut ctx, cb_euc_jp_to_utf8, &[profile, SRC, 2, CONSUMED, DST, 4, PRODUCED]), ERROR_INVALID_ENCODE);
        assert_eq!([ctx.read_u32(PRODUCED), ctx.read_u32(CONSUMED)], [Some(SENTINEL); 2]);
        ctx.write_bytes(SRC, b"\xA4\xA2");
        assert_eq!(call(&mut ctx, cb_euc_jp_to_utf8, &[profile, SRC, 2, 0, DST, 4, 0]), 0);
    }
}
