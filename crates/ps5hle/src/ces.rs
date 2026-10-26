// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceCesCs — character encoding conversion.
//!
//! Profiles: Shift-JIS 1997 (CP932, through the host iconv), EUC-JP
//! (CP51932, JIS X 0208 table) and the single-byte Windows-1252 / 1251
//! profiles. Strings convert between a profile's multibyte encoding and
//! UTF-8 or UCS-2; single characters between UTF-8 and a single-byte
//! profile.
//!
//! Profile sheets and contexts are 256-byte guest objects (32 pointers of
//! library-private state). A profile sheet records its encoding as a pointer
//! to a library-owned name string; the single-byte profiles are
//! library-owned bytes; a context records its profile.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
mod jisx0208;
mod unicode;

use std::sync::{Mutex, OnceLock};

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceCesCs";

const fn err(code: u32) -> HleResult {
    code as i32 as i64
}

pub const ERROR_INVALID_PARAMETER: HleResult = err(0x805C_0001);
pub const ERROR_INVALID_PROFILE: HleResult = err(0x805C_0004);
pub const ERROR_INVALID_SRC_BUFFER: HleResult = err(0x805C_0010);
pub const ERROR_SRC_BUFFER_END: HleResult = err(0x805C_0011);
pub const ERROR_INVALID_ENCODE: HleResult = err(0x805C_0014);
pub const ERROR_UNASSIGNED_CODE: HleResult = err(0x805C_0020);
/// An overlong UTF-8 encoding or a surrogate.
pub const ERROR_ILLEGAL_CODE: HleResult = err(0x805C_0015);
/// An encoding the console refuses outright (its Shift-JIS C1 range).
pub const ERROR_UNSUPPORTED_CODE: HleResult = err(0x805C_0024);
pub const ERROR_INVALID_DST_BUFFER: HleResult = err(0x805C_0030);
pub const ERROR_DST_BUFFER_END: HleResult = err(0x805C_0031);

/// Size of a profile sheet and of a context.
const OBJECT_SIZE: usize = 256;

/// Library-owned guest data: the CP932 name string at 0, the CP1252 and
/// CP1251 profile bytes at 8 and 9, the CP51932 name string at 16, the
/// strict-profile name strings from 24.
const STATICS_SIZE: usize = 64;
const CP932_NAME: u64 = 0;
const CP1252_PROFILE: u64 = 8;
const CP1251_PROFILE: u64 = 9;
const CP51932_NAME: u64 = 16;
const EUCJPX208_NAME: u64 = 24;
const EUCJPXSS2_NAME: u64 = 34;
const EUCJPXSS3_NAME: u64 = 44;
const SJISX208_NAME: u64 = 54;

/// Base of the library-owned guest data, allocated on first use.
static STATICS: Mutex<Option<u64>> = Mutex::new(None);

fn statics(ctx: &CpuContext) -> Option<u64> {
    let mut g = STATICS.lock().unwrap_or_else(|p| p.into_inner());
    if g.is_none() {
        let block = crate::pthread::alloc_guest_block(ctx, STATICS_SIZE as u64)?;
        let mut b = [0u8; STATICS_SIZE];
        b[..5].copy_from_slice(b"CP932");
        b[CP51932_NAME as usize..CP51932_NAME as usize + 7].copy_from_slice(b"CP51932");
        b[EUCJPX208_NAME as usize..EUCJPX208_NAME as usize + 9].copy_from_slice(b"EUCJPX208");
        b[EUCJPXSS2_NAME as usize..EUCJPXSS2_NAME as usize + 9].copy_from_slice(b"EUCJPXSS2");
        b[EUCJPXSS3_NAME as usize..EUCJPXSS3_NAME as usize + 9].copy_from_slice(b"EUCJPXSS3");
        b[SJISX208_NAME as usize..SJISX208_NAME as usize + 8].copy_from_slice(b"SJISX208");
        ctx.write_bytes(block, &b);
        *g = Some(block);
    }
    *g
}

fn static_addr(ctx: &CpuContext, offset: u64) -> u64 {
    statics(ctx).map_or(0, |s| s + offset)
}

fn cp932_name(ctx: &CpuContext) -> u64 {
    static_addr(ctx, CP932_NAME)
}

/// Encoding of a profile.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Profile {
    Cp932,
    EucJp,
    /// EUC-JP over the plain JIS X 0208 plane: no CP932 extension rows,
    /// no half-width katakana (SS2) and no JIS X 0212 (SS3).
    EucJpX208,
    /// The same with SS2 (half-width katakana) back.
    EucJpX208Ss2,
    /// The same with SS2 and SS3 (SS3 reads stay unsupported: no JIS X
    /// 0212 table).
    EucJpX208Ss2Ss3,
    /// Shift-JIS over the plain JIS X 0208 plane: no CP932 extension
    /// rows (lead bytes from 0xFA).
    SjisX208,
    Cp1252,
    Cp1251,
}

impl Profile {
    fn single_byte(self) -> bool {
        matches!(self, Profile::Cp1252 | Profile::Cp1251)
    }

    fn euc_jp(self) -> bool {
        matches!(self, Profile::EucJp | Profile::EucJpX208 | Profile::EucJpX208Ss2 | Profile::EucJpX208Ss2Ss3)
    }

    /// The JIS X 0208 plane without the CP932 extensions and with the
    /// strict row mappings.
    fn strict_jis(self) -> bool {
        matches!(self, Profile::EucJpX208 | Profile::EucJpX208Ss2 | Profile::EucJpX208Ss2Ss3 | Profile::SjisX208)
    }

    /// The JIS X 0208 plane the profile maps through.
    fn jis_plane(self) -> JisStrict {
        match self {
            Profile::SjisX208 => JisStrict::Sjis,
            p if p.strict_jis() => JisStrict::Euc,
            _ => JisStrict::None,
        }
    }

    /// Half-width katakana (SS2, lead 0x8E): every EUC-JP profile but the
    /// plain X0208 one.
    fn has_ss2(self) -> bool {
        !matches!(self, Profile::EucJpX208)
    }

    /// JIS X 0212 (SS3, lead 0x8F): CP51932 and the SS2+SS3 profile.
    fn has_ss3(self) -> bool {
        matches!(self, Profile::EucJp | Profile::EucJpX208Ss2Ss3)
    }
}

/// Identify the profile at guest address `profile`.
fn profile_kind(ctx: &CpuContext, profile: u64) -> Option<Profile> {
    let base = statics(ctx)?;
    if profile == 0 {
        return None;
    }
    if profile == base + CP1252_PROFILE {
        return Some(Profile::Cp1252);
    }
    if profile == base + CP1251_PROFILE {
        return Some(Profile::Cp1251);
    }
    match ctx.read_u64(profile)? {
        n if n == base + CP932_NAME => Some(Profile::Cp932),
        n if n == base + CP51932_NAME => Some(Profile::EucJp),
        n if n == base + EUCJPX208_NAME => Some(Profile::EucJpX208),
        n if n == base + EUCJPXSS2_NAME => Some(Profile::EucJpX208Ss2),
        n if n == base + EUCJPXSS3_NAME => Some(Profile::EucJpX208Ss2Ss3),
        n if n == base + SJISX208_NAME => Some(Profile::SjisX208),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Character codecs
// ---------------------------------------------------------------------------

/// Windows-1252 bytes 0x80..0x9F (the undefined bytes are 0: an
/// unassigned code, measured on hardware).
const CP1252_HIGH: [u16; 32] = [
    0x20AC, 0x0000, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0x0000, 0x017D, 0x0000,
    0x0000, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x0000, 0x017E, 0x0178,
];

/// Windows-1251 bytes 0x80..0xFF (the undefined 0x98 is 0).
const CP1251_HIGH: [u16; 128] = [
    0x0402, 0x0403, 0x201A, 0x0453, 0x201E, 0x2026, 0x2020, 0x2021, 0x20AC, 0x2030, 0x0409, 0x2039, 0x040A, 0x040C, 0x040B, 0x040F,
    0x0452, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, 0x0000, 0x2122, 0x0459, 0x203A, 0x045A, 0x045C, 0x045B, 0x045F,
    0x00A0, 0x040E, 0x045E, 0x0408, 0x00A4, 0x0490, 0x00A6, 0x00A7, 0x0401, 0x00A9, 0x0404, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x0407,
    0x00B0, 0x00B1, 0x0406, 0x0456, 0x0491, 0x00B5, 0x00B6, 0x00B7, 0x0451, 0x2116, 0x0454, 0x00BB, 0x0458, 0x0405, 0x0455, 0x0457,
    0x0410, 0x0411, 0x0412, 0x0413, 0x0414, 0x0415, 0x0416, 0x0417, 0x0418, 0x0419, 0x041A, 0x041B, 0x041C, 0x041D, 0x041E, 0x041F,
    0x0420, 0x0421, 0x0422, 0x0423, 0x0424, 0x0425, 0x0426, 0x0427, 0x0428, 0x0429, 0x042A, 0x042B, 0x042C, 0x042D, 0x042E, 0x042F,
    0x0430, 0x0431, 0x0432, 0x0433, 0x0434, 0x0435, 0x0436, 0x0437, 0x0438, 0x0439, 0x043A, 0x043B, 0x043C, 0x043D, 0x043E, 0x043F,
    0x0440, 0x0441, 0x0442, 0x0443, 0x0444, 0x0445, 0x0446, 0x0447, 0x0448, 0x0449, 0x044A, 0x044B, 0x044C, 0x044D, 0x044E, 0x044F,
];

/// A single-byte profile's byte to Unicode; None when unassigned.
fn sbc_to_unicode(profile: Profile, sbc: u8) -> Option<u32> {
    match profile {
        Profile::Cp1251 if sbc >= 0x80 => (CP1251_HIGH[sbc as usize - 0x80] != 0).then(|| CP1251_HIGH[sbc as usize - 0x80] as u32),
        Profile::Cp1252 if (0x80..0xA0).contains(&sbc) => (CP1252_HIGH[sbc as usize - 0x80] != 0).then(|| CP1252_HIGH[sbc as usize - 0x80] as u32),
        _ => Some(sbc as u32),
    }
}

/// Unicode to a single-byte profile's byte; None when unassigned.
fn unicode_to_sbc(profile: Profile, code: u32) -> Option<u8> {
    let (table, identity): (&[u16], bool) = match profile {
        Profile::Cp1251 => (&CP1251_HIGH, code < 0x80),
        _ => (&CP1252_HIGH, code < 0x80 || (0xA0..=0xFF).contains(&code)),
    };
    if identity {
        return Some(code as u8);
    }
    table.iter().position(|&c| c as u32 == code).map(|i| 0x80 + i as u8)
}

/// Decode one UTF-8 character; `byte(i)` reads past the buffer as 0 and
/// `max` bounds the sequence: Ok((length, code)). An overlong encoding
/// or a surrogate is an illegal code (measured on hardware); five- and
/// six-byte leads decode (their codes then fail to encode).
fn decode_utf8(byte: impl Fn(usize) -> u8, max: u32) -> Result<(u32, u32), HleResult> {
    let lead = byte(0) as u32;
    let (length, mut code, minimum) = match lead {
        0..=0x7F => return Ok((1, lead)),
        0xC0..=0xDF => (2, lead & 0x1F, 0x80),
        0xE0..=0xEF => (3, lead & 0x0F, 0x800),
        0xF0..=0xF7 => (4, lead & 0x07, 0x10000),
        0xF8..=0xFB => (5, lead & 0x03, 0x200000),
        0xFC..=0xFD => (6, lead & 0x01, 0x4000000),
        _ => return Err(ERROR_INVALID_ENCODE),
    };
    for i in 1..length {
        if i >= max {
            return Err(ERROR_SRC_BUFFER_END);
        }
        let b = byte(i as usize) as u32;
        if b & 0xC0 != 0x80 {
            return Err(ERROR_INVALID_ENCODE);
        }
        code = code << 6 | (b & 0x3F);
    }
    if code < minimum || (0xD800..=0xDFFF).contains(&code) {
        return Err(ERROR_ILLEGAL_CODE);
    }
    Ok((length, code))
}

/// UTF-8 encoding of `code` (1..4 bytes).
fn encode_utf8(code: u32) -> Vec<u8> {
    if code <= 0x7F {
        vec![code as u8]
    } else if code <= 0x7FF {
        vec![0xC0 | ((code >> 6) & 0x1F) as u8, 0x80 | (code & 0x3F) as u8]
    } else if code <= 0xFFFF {
        vec![0xE0 | ((code >> 12) & 0x0F) as u8, 0x80 | ((code >> 6) & 0x3F) as u8, 0x80 | (code & 0x3F) as u8]
    } else {
        vec![
            0xF0 | ((code >> 18) & 0x07) as u8,
            0x80 | ((code >> 12) & 0x3F) as u8,
            0x80 | ((code >> 6) & 0x3F) as u8,
            0x80 | (code & 0x3F) as u8,
        ]
    }
}

/// The strict JIS X 0208 plane's differences from CP932's mapping of the
/// same cells (measured on hardware over every cell).
const JIS_X0208_CHANGES: [(u16, u16); 7] =
    [(0x213D, 0x2014), (0x2141, 0x301C), (0x2142, 0x2016), (0x215D, 0x2212), (0x2171, 0x00A2), (0x2172, 0x00A3), (0x224C, 0x00AC)];
/// The backslash cell: U+005C in strict Shift-JIS, U+FF3C in strict EUC-JP.
const JIS_X0208_BACKSLASH_CELL: u16 = 0x2140;
/// Characters strict EUC-JP also encodes into the cell of their full-width
/// form (measured on hardware): (cell, character).
const JIS_X0208_EUC_FALLBACKS: [(u16, u16); 2] = [(0x216F, 0x00A5), (0x2131, 0x203E)];

/// Which JIS X 0208 plane a profile uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum JisStrict {
    /// CP932's plane, with the NEC and IBM extension rows.
    None,
    /// The strict plane of `sceCesUcsProfileInitSJis1997X0208`.
    Sjis,
    /// The strict plane of the `EucJpX0208` profiles.
    Euc,
}

const fn jis_index(jis: u16) -> usize {
    ((jis >> 8) as usize - 0x21) * 94 + (jis & 0xFF) as usize - 0x21
}

fn is_cp932_extension_row(row: usize) -> bool {
    row == 13 || (89..=92).contains(&row)
}

/// Unicode of JIS X 0208 cell `index` in `strict`'s plane. The strict
/// planes are CP932's without the extension rows, with the changed cells
/// and the backslash cell taking their strict mapping.
fn jisx0208_to_unicode(strict: JisStrict, index: usize) -> u16 {
    if strict == JisStrict::None {
        return jisx0208::TO_UCS2[index];
    }
    if is_cp932_extension_row(index / 94 + 1) {
        return 0;
    }
    if index == jis_index(JIS_X0208_BACKSLASH_CELL) {
        return if strict == JisStrict::Sjis { 0x005C } else { 0xFF3C };
    }
    for &(jis, code) in &JIS_X0208_CHANGES {
        if jis_index(jis) == index {
            return code;
        }
    }
    jisx0208::TO_UCS2[index]
}

/// Unicode of one strict Shift-JIS byte below 0x80: the JIS X 0201 Roman
/// set, 0x5C the yen sign and 0x7E the overline (measured on hardware).
fn sjis_x0208_single(byte: u8) -> u32 {
    match byte {
        0x5C => 0x00A5,
        0x7E => 0x203E,
        _ => byte as u32,
    }
}

fn build_jis_reverse(strict: JisStrict) -> Vec<(u16, u16)> {
    let mut v: Vec<(u16, u16)> =
        (0..94 * 94).map(|cell| (jisx0208_to_unicode(strict, cell), cell as u16)).filter(|&(u, _)| u != 0).collect();
    if strict == JisStrict::Euc {
        v.extend(JIS_X0208_EUC_FALLBACKS.iter().map(|&(jis, code)| (code, jis_index(jis) as u16)));
    }
    // A stable sort: the first (lowest) cell of a character stays first.
    v.sort_by_key(|e| e.0);
    v.dedup_by_key(|e| e.0);
    v
}

/// UCS-2 to JIS X 0208 cell index of `strict`'s plane, first (lowest) cell
/// per character.
fn jis_reverse(strict: JisStrict) -> &'static [(u16, u16)] {
    static REVERSE: OnceLock<Vec<(u16, u16)>> = OnceLock::new();
    static SJIS_REVERSE: OnceLock<Vec<(u16, u16)>> = OnceLock::new();
    static EUC_REVERSE: OnceLock<Vec<(u16, u16)>> = OnceLock::new();
    let table = match strict {
        JisStrict::None => &REVERSE,
        JisStrict::Sjis => &SJIS_REVERSE,
        JisStrict::Euc => &EUC_REVERSE,
    };
    table.get_or_init(|| build_jis_reverse(strict))
}

/// One CP932 character through the host iconv.
fn cp932_iconv(to: &std::ffi::CStr, from: &std::ffi::CStr, input: &[u8]) -> Option<Vec<u8>> {
    let mut conv = Converter::open(to, from)?;
    let mut out = [0u8; 8];
    match conv.convert(input, &mut out) {
        (Iconv::Done, used, made) if used == input.len() => Some(out[..made].to_vec()),
        _ => None,
    }
}

/// Decode one character of a profile's multibyte encoding; `byte(i)` reads
/// past the buffer as 0 and `max` bounds the sequence: Ok((length, code)).
fn decode_mbcs(profile: Profile, byte: impl Fn(usize) -> u8, max: u32) -> Result<(u32, u32), HleResult> {
    let lead = byte(0);
    if profile.single_byte() {
        return sbc_to_unicode(profile, lead).map(|code| (1, code)).ok_or(ERROR_UNASSIGNED_CODE);
    }
    if lead < 0x80 {
        return Ok((1, if profile == Profile::SjisX208 { sjis_x0208_single(lead) } else { lead as u32 }));
    }
    match profile {
        Profile::Cp932 => {
            let length = if is_lead_byte(lead) { 2 } else { 1 };
            if length > max {
                return Err(ERROR_SRC_BUFFER_END);
            }
            let input: Vec<u8> = (0..length as usize).map(&byte).collect();
            let decoded = cp932_iconv(c"UTF-8", c"CP932", &input).filter(|d| !d.is_empty());
            match decoded {
                Some(d) => decode_utf8(|i| d.get(i).copied().unwrap_or(0), d.len() as u32).map(|(_, c)| (length, c)),
                None if length == 2 && is_trail_byte(input[1]) => Err(ERROR_UNASSIGNED_CODE),
                None => Err(ERROR_INVALID_ENCODE),
            }
        }
        Profile::SjisX208 => {
            // The strict JIS X 0208 plane: a two-byte code maps through it
            // (no CP932 extension rows); one byte from 0xA1 to 0xDF is
            // half-width katakana.
            if (0xA1..=0xDF).contains(&lead) {
                return Ok((1, 0xFF61 + (lead - 0xA1) as u32));
            }
            if !is_lead_byte(lead) {
                return Err(ERROR_INVALID_ENCODE);
            }
            if max < 2 {
                return Err(ERROR_SRC_BUFFER_END);
            }
            let trail = byte(1);
            if !is_trail_byte(trail) {
                return Err(ERROR_INVALID_ENCODE);
            }
            let (row, cell) = sjis_to_jis(lead, trail);
            if !(1..=94).contains(&row) || !(1..=94).contains(&cell) {
                return Err(ERROR_UNASSIGNED_CODE);
            }
            let code = jisx0208_to_unicode(JisStrict::Sjis, (row - 1) as usize * 94 + (cell - 1) as usize);
            return if code != 0 { Ok((2, code as u32)) } else { Err(ERROR_UNASSIGNED_CODE) };
        }
        _ => {
            if lead == 0x8E && profile.has_ss2() {
                if max < 2 {
                    return Err(ERROR_SRC_BUFFER_END);
                }
                let trail = byte(1);
                if !(0xA1..=0xDF).contains(&trail) {
                    return Err(ERROR_INVALID_ENCODE);
                }
                return Ok((2, 0xFF61 + (trail - 0xA1) as u32));
            }
            if (0xA1..=0xFE).contains(&lead) {
                if max < 2 {
                    return Err(ERROR_SRC_BUFFER_END);
                }
                let trail = byte(1);
                if !(0xA1..=0xFE).contains(&trail) {
                    return Err(ERROR_INVALID_ENCODE);
                }
                let code = jisx0208_to_unicode(profile.jis_plane(), (lead - 0xA1) as usize * 94 + (trail - 0xA1) as usize);
                return if code != 0 { Ok((2, code as u32)) } else { Err(ERROR_UNASSIGNED_CODE) };
            }
            if lead == 0x8F && profile.has_ss3() {
                // JIS X 0212 (three-byte) characters: no mapping table.
                log::warn!("[ces] EUC-JP JIS X 0212 character {lead:#04x} {:#04x} {:#04x} not supported", byte(1), byte(2));
            }
            Err(ERROR_INVALID_ENCODE)
        }
    }
}

/// Code points the console's non-strict JIS encodings take through a
/// full-width or repeated alias before the table lookup (behaviour
/// measured on hardware).
fn jis_alias(code: u32) -> u32 {
    match code {
        0x00A2 => 0xFFE0,
        0x00A3 => 0xFFE1,
        0x00A5 => 0xFFE5,
        0x00AC => 0xFFE2,
        0x2014 => 0x2015,
        0x2016 => 0x2225,
        0x203E => 0xFFE3,
        0x2212 => 0xFF0D,
        0x301C => 0xFF5E,
        _ => code,
    }
}

/// Encode one character in a profile's multibyte encoding.
fn encode_mbcs(profile: Profile, code: u32) -> Result<Vec<u8>, HleResult> {
    if profile.single_byte() {
        return unicode_to_sbc(profile, code).map(|b| vec![b]).ok_or(ERROR_UNASSIGNED_CODE);
    }
    if code < 0x80 && !(profile == Profile::SjisX208 && (code == 0x5C || code == 0x7E)) {
        return Ok(vec![code as u8]);
    }
    match profile {
        Profile::Cp932 => {
            // The console's Shift-JIS encoders refuse the C1 range and
            // drop a byte-order mark (measured on hardware).
            if code <= 0x9F {
                return Err(ERROR_UNSUPPORTED_CODE);
            }
            if code == 0xFEFF {
                return Ok(Vec::new());
            }
            cp932_iconv(c"CP932", c"UTF-8", &encode_utf8(code)).filter(|e| !e.is_empty()).ok_or(ERROR_UNASSIGNED_CODE)
        }
        Profile::SjisX208 => {
            // The strict profile's bytes 0x5C and 0x7E are the yen sign
            // and the overline; ASCII backslash and tilde have no single
            // byte, and no character takes an alias (measured on hardware).
            match code {
                0x00A5 => return Ok(vec![0x5C]),
                0x203E => return Ok(vec![0x7E]),
                _ => {}
            }
            // Half-width katakana is its single Shift-JIS byte.
            if (0xFF61..=0xFF9F).contains(&code) {
                return Ok(vec![(0xA1 + (code - 0xFF61)) as u8]);
            }
            if code > 0xFFFF {
                return Err(ERROR_UNASSIGNED_CODE);
            }
            let reverse = jis_reverse(JisStrict::Sjis);
            let i = reverse.binary_search_by_key(&(code as u16), |e| e.0).map_err(|_| ERROR_UNASSIGNED_CODE)?;
            let cell = reverse[i].1;
            let (row, column) = ((cell / 94 + 1) as i32, (cell % 94 + 1) as i32);
            let (lead, trail) = if row % 2 == 1 {
                let lead = if row <= 61 { (row - 1) / 2 + 0x81 } else { (row - 63) / 2 + 0xE0 };
                (lead, if column < 64 { column + 0x3F } else { column + 0x40 })
            } else {
                let base = row - 1;
                let lead = if base <= 61 { (base - 1) / 2 + 0x81 } else { (base - 63) / 2 + 0xE0 };
                (lead, column + 0x9E)
            };
            Ok(vec![lead as u8, trail as u8])
        }
        _ => {
            let strict = profile.strict_jis();
            // The console's plain EUC-JP refuses one control, passes the
            // C1 bytes (but the SS2 / SS3 leads) through and drops a byte
            // order mark (measured on hardware).
            if !strict && code == 0x0E {
                return Err(ERROR_UNASSIGNED_CODE);
            }
            if code < 0x80 || (!strict && code < 0xA0 && code != 0x8E && code != 0x8F) {
                return Ok(vec![code as u8]);
            }
            if !strict && code == 0xFEFF {
                return Ok(Vec::new());
            }
            let code = if strict { code } else { jis_alias(code) };
            if profile.euc_jp() && profile.has_ss2() && (0xFF61..=0xFF9F).contains(&code) {
                return Ok(vec![0x8E, (0xA1 + (code - 0xFF61)) as u8]);
            }
            if code > 0xFFFF {
                return Err(ERROR_UNASSIGNED_CODE);
            }
            let reverse = jis_reverse(profile.jis_plane());
            let i = reverse.binary_search_by_key(&(code as u16), |e| e.0).map_err(|_| ERROR_UNASSIGNED_CODE)?;
            let cell = reverse[i].1;
            Ok(vec![(0xA1 + cell / 94) as u8, (0xA1 + cell % 94) as u8])
        }
    }
}

// ---------------------------------------------------------------------------
// Single-byte profiles
// ---------------------------------------------------------------------------

/// sceCesRefersUcsProfileCp1252() -> const profile*
fn cb_refers_ucs_profile_cp1252(ctx: &mut CpuContext) -> HleResult {
    static_addr(ctx, CP1252_PROFILE) as HleResult
}

/// sceCesRefersUcsProfileCp1251() -> const profile*
fn cb_refers_ucs_profile_cp1251(ctx: &mut CpuContext) -> HleResult {
    static_addr(ctx, CP1251_PROFILE) as HleResult
}

fn sbc_profile(ctx: &CpuContext, profile: u64) -> Option<Profile> {
    profile_kind(ctx, profile).filter(|p| p.single_byte())
}

/// sceCesUtf8ToSbc(utf8, utf8Max, uint32* utf8Len, profile, uint8* sbc)
fn cb_utf8_to_sbc(ctx: &mut CpuContext) -> HleResult {
    let (utf8, max, len_out, profile, sbc) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx(), ctx.r8());
    let Some(profile) = sbc_profile(ctx, profile) else {
        return ERROR_INVALID_PARAMETER;
    };
    if utf8 == 0 {
        return ERROR_INVALID_SRC_BUFFER;
    }
    if max == 0 {
        return ERROR_SRC_BUFFER_END;
    }
    if sbc == 0 {
        return ERROR_INVALID_DST_BUFFER;
    }
    let bytes = read_units(ctx, utf8, 1, max.min(4)).unwrap_or_default();
    let (len, code) = match decode_utf8(|i| bytes.get(i).copied().unwrap_or(0) as u8, max) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let Some(b) = unicode_to_sbc(profile, code) else {
        return ERROR_UNASSIGNED_CODE;
    };
    ctx.write_bytes(sbc, &[b]);
    if len_out != 0 {
        ctx.write_u32(len_out, len);
    }
    0
}

/// sceCesSbcToUtf8(profile, uint8 sbc, utf8, utf8Max, uint32* utf8Len)
fn cb_sbc_to_utf8(ctx: &mut CpuContext) -> HleResult {
    let (profile, sbc, utf8, max, len_out) = (ctx.rdi(), ctx.rsi() as u8, ctx.rdx(), ctx.rcx() as u32, ctx.r8());
    let Some(profile) = sbc_profile(ctx, profile) else {
        return ERROR_INVALID_PARAMETER;
    };
    if utf8 == 0 {
        return ERROR_INVALID_DST_BUFFER;
    }
    // An undefined byte of the profile is an unassigned code (measured on
    // hardware).
    let Some(code) = sbc_to_unicode(profile, sbc) else {
        return ERROR_UNASSIGNED_CODE;
    };
    let encoded = encode_utf8(code);
    if max < encoded.len() as u32 {
        return ERROR_DST_BUFFER_END;
    }
    ctx.write_bytes(utf8, &encoded);
    if len_out != 0 {
        ctx.write_u32(len_out, encoded.len() as u32);
    }
    0
}

// ---------------------------------------------------------------------------
// Strings through any profile
// ---------------------------------------------------------------------------

/// Read up to `max` units of `unit` bytes from `addr`, stopping after a zero
/// unit (not included). Reads stay within pages so a string ending just
/// before unmapped memory reads correctly.
fn read_units(ctx: &CpuContext, addr: u64, unit: u64, max: u32) -> Option<Vec<u32>> {
    const PAGE: u64 = 0x1000;
    let mut out = Vec::new();
    let mut at = addr;
    while (out.len() as u32) < max {
        let left = (max - out.len() as u32) as u64 * unit;
        let chunk = left.min(PAGE - at % PAGE).max(unit);
        let chunk = chunk - chunk % unit;
        let bytes = ctx.read_bytes(at, chunk as usize)?;
        for u in bytes.chunks_exact(unit as usize) {
            let v = if unit == 1 { u[0] as u32 } else { u16::from_le_bytes([u[0], u[1]]) as u32 };
            if v == 0 {
                return Some(out);
            }
            out.push(v);
        }
        at += chunk;
    }
    Some(out)
}

/// The profile a context was initialised with.
fn context_profile(ctx: &CpuContext, context: u64) -> Result<Profile, HleResult> {
    if context == 0 {
        return Err(ERROR_INVALID_PARAMETER);
    }
    profile_kind(ctx, ctx.read_u64(context).unwrap_or(0)).ok_or(ERROR_INVALID_PROFILE)
}

/// Source string encoding of a conversion.
#[derive(Clone, Copy, PartialEq)]
enum Unicode {
    Utf8,
    Ucs2,
}

impl Unicode {
    fn unit(self) -> u64 {
        if self == Unicode::Utf8 { 1 } else { 2 }
    }

    fn decode(self, s: &[u32], max: u32) -> Result<(u32, u32), HleResult> {
        match self {
            Unicode::Utf8 => decode_utf8(|i| s.get(i).copied().unwrap_or(0) as u8, max),
            Unicode::Ucs2 if (0xD800..=0xDFFF).contains(&s[0]) => Err(ERROR_INVALID_ENCODE),
            Unicode::Ucs2 => Ok((1, s[0])),
        }
    }

    fn encode(self, code: u32) -> Vec<u32> {
        match self {
            Unicode::Utf8 => encode_utf8(code).into_iter().map(u32::from).collect(),
            Unicode::Ucs2 => vec![code & 0xFFFF],
        }
    }

    fn write(self, ctx: &CpuContext, addr: u64, units: &[u32]) {
        let bytes: Vec<u8> = match self {
            Unicode::Utf8 => units.iter().map(|&u| u as u8).collect(),
            Unicode::Ucs2 => units.iter().flat_map(|&u| (u as u16).to_le_bytes()).collect(),
        };
        ctx.write_bytes(addr, &bytes);
    }
}

/// Convert a UTF-8 / UCS-2 string to the context's multibyte encoding
/// (`mbcs` 0 only measures). Stops at `source_max` units or a zero unit;
/// the lengths are written on success.
#[allow(clippy::too_many_arguments)]
fn unicode_to_mbcs(
    ctx: &CpuContext,
    kind: Unicode,
    context: u64,
    source: u64,
    source_max: u32,
    source_len: u64,
    mbcs: u64,
    mbcs_max: u32,
    mbcs_len: u64,
) -> HleResult {
    let profile = match context_profile(ctx, context) {
        Ok(p) => p,
        Err(e) => return e,
    };
    if source == 0 {
        return ERROR_INVALID_SRC_BUFFER;
    }
    let text = read_units(ctx, source, kind.unit(), source_max).unwrap_or_default();
    let (mut read, mut written) = (0u32, 0u32);
    while read < source_max && (read as usize) < text.len() {
        let (length, code) = match kind.decode(&text[read as usize..], source_max - read) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let encoded = match encode_mbcs(profile, code) {
            Ok(e) => e,
            Err(e) => return e,
        };
        if mbcs != 0 {
            if written + encoded.len() as u32 > mbcs_max {
                return ERROR_DST_BUFFER_END;
            }
            ctx.write_bytes(mbcs + written as u64, &encoded);
        }
        read += length;
        written += encoded.len() as u32;
    }
    if source_len != 0 {
        ctx.write_u32(source_len, read);
    }
    if mbcs_len != 0 {
        ctx.write_u32(mbcs_len, written);
    }
    0
}

/// Convert a string in the context's multibyte encoding to UTF-8 / UCS-2
/// (`target` 0 only measures). Stops at `mbcs_max` bytes or a zero byte;
/// the lengths are written on success.
#[allow(clippy::too_many_arguments)]
fn mbcs_to_unicode(
    ctx: &CpuContext,
    kind: Unicode,
    profile: Profile,
    mbcs: u64,
    mbcs_max: u32,
    mbcs_len: u64,
    target: u64,
    target_max: u32,
    target_len: u64,
) -> HleResult {
    if mbcs == 0 {
        return ERROR_INVALID_SRC_BUFFER;
    }
    let text = read_units(ctx, mbcs, 1, mbcs_max).unwrap_or_default();
    let (mut read, mut written) = (0u32, 0u32);
    while read < mbcs_max && (read as usize) < text.len() {
        let rest = &text[read as usize..];
        let (length, code) = match decode_mbcs(profile, |i| rest.get(i).copied().unwrap_or(0) as u8, mbcs_max - read) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let encoded = kind.encode(code);
        if target != 0 {
            if written + encoded.len() as u32 > target_max {
                return ERROR_DST_BUFFER_END;
            }
            kind.write(ctx, target + written as u64 * kind.unit(), &encoded);
        }
        read += length;
        written += encoded.len() as u32;
    }
    if mbcs_len != 0 {
        ctx.write_u32(mbcs_len, read);
    }
    if target_len != 0 {
        ctx.write_u32(target_len, written);
    }
    0
}

fn stack_arg0(ctx: &CpuContext) -> u64 {
    ctx.read_u64(ctx.get(crate::context::CpuRegister::Rsp) + 8).unwrap_or(0)
}

/// sceCesUtf8StrToMbcsStr(context, utf8, utf8Max, uint32* utf8Len, mbcs, mbcsMax, uint32* mbcsLen)
fn cb_utf8_str_to_mbcs_str(ctx: &mut CpuContext) -> HleResult {
    if ctx.r8() == 0 {
        return ERROR_INVALID_DST_BUFFER;
    }
    let len = stack_arg0(ctx);
    unicode_to_mbcs(ctx, Unicode::Utf8, ctx.rdi(), ctx.rsi(), ctx.rdx() as u32, ctx.rcx(), ctx.r8(), ctx.r9() as u32, len)
}

/// sceCesUtf8StrGetMbcsLen(context, utf8, utf8Max, uint32* utf8Len, uint32* mbcsLen)
fn cb_utf8_str_get_mbcs_len(ctx: &mut CpuContext) -> HleResult {
    unicode_to_mbcs(ctx, Unicode::Utf8, ctx.rdi(), ctx.rsi(), ctx.rdx() as u32, ctx.rcx(), 0, 0, ctx.r8())
}

/// sceCesUcs2StrToMbcsStr(context, ucs2, ucs2Max, uint32* ucs2Len, mbcs, mbcsMax, uint32* mbcsLen)
fn cb_ucs2_str_to_mbcs_str(ctx: &mut CpuContext) -> HleResult {
    if ctx.r8() == 0 {
        return ERROR_INVALID_DST_BUFFER;
    }
    let len = stack_arg0(ctx);
    unicode_to_mbcs(ctx, Unicode::Ucs2, ctx.rdi(), ctx.rsi(), ctx.rdx() as u32, ctx.rcx(), ctx.r8(), ctx.r9() as u32, len)
}

/// sceCesUcs2StrGetMbcsLen(context, ucs2, ucs2Max, uint32* ucs2Len, uint32* mbcsLen)
fn cb_ucs2_str_get_mbcs_len(ctx: &mut CpuContext) -> HleResult {
    unicode_to_mbcs(ctx, Unicode::Ucs2, ctx.rdi(), ctx.rsi(), ctx.rdx() as u32, ctx.rcx(), 0, 0, ctx.r8())
}

/// sceCesMbcsStrToUcs2Str(context, mbcs, mbcsMax, uint32* mbcsLen, ucs2, ucs2Max, uint32* ucs2Len)
fn cb_mbcs_str_to_ucs2_str(ctx: &mut CpuContext) -> HleResult {
    if ctx.r8() == 0 {
        return ERROR_INVALID_DST_BUFFER;
    }
    let profile = match context_profile(ctx, ctx.rdi()) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let len = stack_arg0(ctx);
    mbcs_to_unicode(ctx, Unicode::Ucs2, profile, ctx.rsi(), ctx.rdx() as u32, ctx.rcx(), ctx.r8(), ctx.r9() as u32, len)
}

/// sceCesMbcsStrGetUcs2Len(context, mbcs, mbcsMax, uint32* mbcsLen, uint32* ucs2Len)
fn cb_mbcs_str_get_ucs2_len(ctx: &mut CpuContext) -> HleResult {
    let profile = match context_profile(ctx, ctx.rdi()) {
        Ok(p) => p,
        Err(e) => return e,
    };
    mbcs_to_unicode(ctx, Unicode::Ucs2, profile, ctx.rsi(), ctx.rdx() as u32, ctx.rcx(), 0, 0, ctx.r8())
}

/// sceCesEucJpToUcs2(profile, euc, eucMax, uint32* eucSize, uint16* ucs2):
/// one character.
fn cb_euc_jp_to_ucs2(ctx: &mut CpuContext) -> HleResult {
    let (profile, euc, max, size_out, ucs2) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u32, ctx.rcx(), ctx.r8());
    let Some(profile) = profile_kind(ctx, profile).filter(|p| p.euc_jp()) else {
        return ERROR_INVALID_PARAMETER;
    };
    if euc == 0 {
        return ERROR_INVALID_SRC_BUFFER;
    }
    if max == 0 {
        return ERROR_SRC_BUFFER_END;
    }
    if ucs2 == 0 {
        return ERROR_INVALID_DST_BUFFER;
    }
    let bytes = read_units(ctx, euc, 1, max.min(3)).unwrap_or_default();
    let (length, code) = match decode_mbcs(profile, |i| bytes.get(i).copied().unwrap_or(0) as u8, max) {
        Ok(v) => v,
        Err(e) => return e,
    };
    ctx.write_bytes(ucs2, &(code as u16).to_le_bytes());
    if size_out != 0 {
        ctx.write_u32(size_out, length);
    }
    0
}

/// sceCesUcsProfileInitEucJpCp51932(profile* sheet) -> sheet
fn cb_ucs_profile_init_euc_jp_cp51932(ctx: &mut CpuContext) -> HleResult {
    init_profile(ctx, CP51932_NAME)
}

/// sceCesUcsProfileInitEucJpX0208(profile* sheet) -> sheet
fn cb_ucs_profile_init_euc_jp_x0208(ctx: &mut CpuContext) -> HleResult {
    init_profile(ctx, EUCJPX208_NAME)
}

/// sceCesUcsProfileInitEucJpX0208Ss2(profile* sheet) -> sheet
fn cb_ucs_profile_init_euc_jp_x0208_ss2(ctx: &mut CpuContext) -> HleResult {
    init_profile(ctx, EUCJPXSS2_NAME)
}

/// sceCesUcsProfileInitEucJpX0208Ss2Ss3(profile* sheet) -> sheet
fn cb_ucs_profile_init_euc_jp_x0208_ss2_ss3(ctx: &mut CpuContext) -> HleResult {
    init_profile(ctx, EUCJPXSS3_NAME)
}

/// sceCesUcsProfileInitSJis1997X0208(profile* sheet) -> sheet
fn cb_ucs_profile_init_sjis1997_x0208(ctx: &mut CpuContext) -> HleResult {
    init_profile(ctx, SJISX208_NAME)
}

/// Write a profile sheet naming `name` into it; the sheet pointer returns.
fn init_profile(ctx: &CpuContext, name: u64) -> HleResult {
    let sheet = ctx.rdi();
    if sheet != 0 {
        let mut obj = vec![0u8; OBJECT_SIZE];
        obj[..8].copy_from_slice(&static_addr(ctx, name).to_le_bytes());
        ctx.write_bytes(sheet, &obj);
    }
    sheet as HleResult
}

// ---------------------------------------------------------------------------
// Single characters into any MBCS profile
// ---------------------------------------------------------------------------

/// A scalar above the Unicode range is an invalid encode and a surrogate
/// out of the code space an illegal code (measured on hardware).
fn scalar_error(code: u32) -> HleResult {
    if code > 0x10FFFF {
        return ERROR_INVALID_ENCODE;
    }
    if (0xD800..=0xDFFF).contains(&code) {
        return ERROR_ILLEGAL_CODE;
    }
    0
}

/// Encode one scalar into a profile's multibyte encoding (the UCS-2 and
/// UTF-32 ToMbc forms): the needed length is reported with
/// DST_BUFFER_END, any other refusal reports zero (measured on hardware).
fn scalar_to_mbcs(ctx: &CpuContext, code: u32, profile: u64, mbc: u64, mbc_max: u32, mbc_len: u64) -> HleResult {
    let Some(profile) = profile_kind(ctx, profile) else {
        return ERROR_INVALID_PARAMETER;
    };
    if mbc == 0 {
        return ERROR_INVALID_DST_BUFFER;
    }
    let mut result = scalar_error(code);
    let encoded = if result == 0 {
        match encode_mbcs(profile, code) {
            Ok(e) => e,
            Err(e) => {
                result = e;
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };
    if result == 0 && mbc_max < encoded.len() as u32 {
        result = ERROR_DST_BUFFER_END;
    }
    if result == 0 {
        ctx.write_bytes(mbc, &encoded);
    }
    if mbc_len != 0 {
        let reported = if result == 0 || result == ERROR_DST_BUFFER_END { encoded.len() as u32 } else { 0 };
        ctx.write_u32(mbc_len, reported);
    }
    result
}

/// Decode one UTF-16 character of at most `max` units: a lone low
/// surrogate or a broken pair is an invalid encode, a high surrogate at
/// the end of the buffer leaves it short.
fn decode_utf16_units(unit: impl Fn(usize) -> u16, max: u32) -> Result<(u32, u32), HleResult> {
    let first = unit(0);
    if (0xDC00..=0xDFFF).contains(&first) {
        return Err(ERROR_INVALID_ENCODE);
    }
    if (0xD800..=0xDBFF).contains(&first) {
        if max < 2 {
            return Err(ERROR_SRC_BUFFER_END);
        }
        let second = unit(1);
        if !(0xDC00..=0xDFFF).contains(&second) {
            return Err(ERROR_INVALID_ENCODE);
        }
        return Ok((2, 0x10000 + ((first as u32 - 0xD800) << 10) + (second as u32 - 0xDC00)));
    }
    Ok((1, first as u32))
}

/// Store the first character of a UTF-8 / UTF-16 buffer: the lengths stay
/// untouched on every failure (measured on hardware).
fn finish_buffer_to_mbcs(ctx: &CpuContext, code: u32, units: u32, profile: Profile, src_len: u64, mbc: u64, mbc_max: u32, mbc_len: u64) -> HleResult {
    let encoded = match encode_mbcs(profile, code) {
        Ok(e) => e,
        Err(e) => return e,
    };
    if mbc_max < encoded.len() as u32 {
        return ERROR_DST_BUFFER_END;
    }
    ctx.write_bytes(mbc, &encoded);
    if src_len != 0 {
        ctx.write_u32(src_len, units);
    }
    if mbc_len != 0 {
        ctx.write_u32(mbc_len, encoded.len() as u32);
    }
    0
}

/// sceCesUcs2ToMbc(ucs2, profile*, mbc*, mbcMax, mbcLen*)
fn cb_ucs2_to_mbc(ctx: &mut CpuContext) -> HleResult {
    scalar_to_mbcs(ctx, ctx.rdi() as u16 as u32, ctx.rsi(), ctx.rdx(), ctx.rcx() as u32, ctx.r8())
}

/// sceCesUtf8ToMbc(utf8*, utf8Max, utf8Len*, profile*, mbc*, mbcMax, mbcLen*)
fn cb_utf8_to_mbc(ctx: &mut CpuContext) -> HleResult {
    let (src, max, src_len, profile, mbc, mbc_max) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx(), ctx.r8(), ctx.r9() as u32);
    let mbc_len = stack_arg0(ctx);
    let Some(p) = profile_kind(ctx, profile) else {
        return ERROR_INVALID_PARAMETER;
    };
    if src == 0 {
        return ERROR_INVALID_SRC_BUFFER;
    }
    if max == 0 {
        return ERROR_SRC_BUFFER_END;
    }
    if mbc == 0 {
        return ERROR_INVALID_DST_BUFFER;
    }
    let bytes = ctx.read_bytes(src, max.min(6) as usize).unwrap_or_default();
    let (units, code) = match decode_utf8(|i| bytes.get(i).copied().unwrap_or(0), max) {
        Ok(v) => v,
        Err(e) => return e,
    };
    finish_buffer_to_mbcs(ctx, code, units, p, src_len, mbc, mbc_max, mbc_len)
}

/// sceCesUtf16ToMbc / be / le(utf16*, utf16Max, utf16Len*, profile*, mbc*, mbcMax, mbcLen*):
/// the unsuffixed form reads the console's native byte order.
fn utf16_to_mbc(ctx: &mut CpuContext, big_endian: bool) -> HleResult {
    let (src, max, src_len, profile, mbc, mbc_max) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx(), ctx.r8(), ctx.r9() as u32);
    let mbc_len = stack_arg0(ctx);
    let Some(p) = profile_kind(ctx, profile) else {
        return ERROR_INVALID_PARAMETER;
    };
    if src == 0 {
        return ERROR_INVALID_SRC_BUFFER;
    }
    if max == 0 {
        return ERROR_SRC_BUFFER_END;
    }
    if mbc == 0 {
        return ERROR_INVALID_DST_BUFFER;
    }
    let bytes = ctx.read_bytes(src, max.min(2) as usize * 2).unwrap_or_default();
    let unit = |i: usize| -> u16 {
        let b = [bytes.get(i * 2).copied().unwrap_or(0), bytes.get(i * 2 + 1).copied().unwrap_or(0)];
        if big_endian { u16::from_be_bytes(b) } else { u16::from_le_bytes(b) }
    };
    let (units, code) = match decode_utf16_units(unit, max) {
        Ok(v) => v,
        Err(e) => return e,
    };
    finish_buffer_to_mbcs(ctx, code, units, p, src_len, mbc, mbc_max, mbc_len)
}

fn cb_utf16_to_mbc(ctx: &mut CpuContext) -> HleResult {
    utf16_to_mbc(ctx, false)
}

fn cb_utf16be_to_mbc(ctx: &mut CpuContext) -> HleResult {
    utf16_to_mbc(ctx, true)
}

fn cb_utf16le_to_mbc(ctx: &mut CpuContext) -> HleResult {
    utf16_to_mbc(ctx, false)
}

/// sceCesUtf32ToMbc(utf32, profile*, mbc*, mbcMax, mbcLen*)
fn cb_utf32_to_mbc(ctx: &mut CpuContext) -> HleResult {
    scalar_to_mbcs(ctx, ctx.rdi() as u32, ctx.rsi(), ctx.rdx(), ctx.rcx() as u32, ctx.r8())
}

/// sceCesUtf32beToMbc / sceCesUtf32leToMbc(utf32*, profile*, mbc*, mbcMax, mbcLen*)
fn utf32_ptr_to_mbc(ctx: &mut CpuContext, big_endian: bool) -> HleResult {
    let (src, profile, mbc, mbc_max, mbc_len) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx() as u32, ctx.r8());
    if profile_kind(ctx, profile).is_none() {
        return ERROR_INVALID_PARAMETER;
    }
    let value = match ctx.read_bytes(src, 4) {
        Some(b) => u32::from_le_bytes(b.try_into().unwrap()),
        None => return ERROR_INVALID_SRC_BUFFER,
    };
    let value = if big_endian { value.swap_bytes() } else { value };
    scalar_to_mbcs(ctx, value, profile, mbc, mbc_max, mbc_len)
}

fn cb_utf32be_to_mbc(ctx: &mut CpuContext) -> HleResult {
    utf32_ptr_to_mbc(ctx, true)
}

fn cb_utf32le_to_mbc(ctx: &mut CpuContext) -> HleResult {
    utf32_ptr_to_mbc(ctx, false)
}

// ---------------------------------------------------------------------------
// Multibyte (CP932) strings to UTF-8
// ---------------------------------------------------------------------------

/// sceCesUcsProfileInitSJis1997Cp932(profile* sheet) -> sheet
fn cb_ucs_profile_init_sjis1997_cp932(ctx: &mut CpuContext) -> HleResult {
    let sheet = ctx.rdi();
    if sheet != 0 {
        let mut obj = vec![0u8; OBJECT_SIZE];
        obj[..8].copy_from_slice(&cp932_name(ctx).to_le_bytes());
        ctx.write_bytes(sheet, &obj);
    }
    sheet as HleResult
}

/// sceCesMbcsUcsContextInit(context*, const profile*)
fn cb_mbcs_ucs_context_init(ctx: &mut CpuContext) -> HleResult {
    let (context, profile) = (ctx.rdi(), ctx.rsi());
    if context == 0 || profile == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let mut obj = vec![0u8; OBJECT_SIZE];
    obj[..8].copy_from_slice(&profile.to_le_bytes());
    ctx.write_bytes(context, &obj);
    0
}

fn is_lead_byte(b: u8) -> bool {
    (0x81..=0x9F).contains(&b) || (0xE0..=0xFC).contains(&b)
}

/// Shift-JIS bytes to a JIS X 0208 row and cell (1-based; the caller
/// checks the plane's range).
fn sjis_to_jis(lead: u8, trail: u8) -> (i32, i32) {
    let mut row = if lead <= 0x9F { (lead as i32 - 0x81) * 2 + 1 } else { (lead as i32 - 0xE0) * 2 + 63 };
    let cell = if trail >= 0x9F {
        row += 1;
        trail as i32 - 0x9E
    } else if trail > 0x7E {
        trail as i32 - 0x40
    } else {
        trail as i32 - 0x3F
    };
    (row, cell)
}

fn is_trail_byte(b: u8) -> bool {
    (0x40..=0x7E).contains(&b) || (0x80..=0xFC).contains(&b)
}

/// Outcome of one host iconv call.
enum Iconv {
    Done,
    OutputFull,
    Incomplete,
    Invalid,
}

/// A host iconv conversion.
#[cfg(unix)]
struct Converter(libc::iconv_t);

/// CP932 <-> UTF-8 (the only conversions used) with the WHATWG
/// Shift_JIS codec, which is Windows-31J.
#[cfg(windows)]
enum Converter {
    Decode(encoding_rs::Decoder),
    Encode(encoding_rs::Encoder),
}

#[cfg(windows)]
impl Converter {
    fn open(to: &std::ffi::CStr, from: &std::ffi::CStr) -> Option<Self> {
        let sjis = encoding_rs::SHIFT_JIS;
        match (to.to_bytes(), from.to_bytes()) {
            (b"UTF-8", b"CP932") => Some(Self::Decode(sjis.new_decoder_without_bom_handling())),
            (b"CP932", b"UTF-8") => Some(Self::Encode(sjis.new_encoder())),
            _ => None,
        }
    }

    /// Convert from `input` into `output`; returns (status, consumed, produced).
    fn convert(&mut self, input: &[u8], output: &mut [u8]) -> (Iconv, usize, usize) {
        match self {
            Self::Decode(d) => {
                let (r, read, written) = d.decode_to_utf8_without_replacement(input, output, true);
                let status = match r {
                    encoding_rs::DecoderResult::InputEmpty => Iconv::Done,
                    encoding_rs::DecoderResult::OutputFull => Iconv::OutputFull,
                    encoding_rs::DecoderResult::Malformed(..) => Iconv::Invalid,
                };
                (status, read, written)
            }
            Self::Encode(e) => {
                let text = match std::str::from_utf8(input) {
                    Ok(t) => t,
                    Err(err) if err.error_len().is_none() => {
                        let (s, used, made) = Self::encode(e, &input[..err.valid_up_to()], output);
                        return (if matches!(s, Iconv::Done) { Iconv::Incomplete } else { s }, used, made);
                    }
                    Err(err) => {
                        let (s, used, made) = Self::encode(e, &input[..err.valid_up_to()], output);
                        return (if matches!(s, Iconv::Done) { Iconv::Invalid } else { s }, used, made);
                    }
                };
                Self::encode(e, text.as_bytes(), output)
            }
        }
    }

    fn encode(e: &mut encoding_rs::Encoder, input: &[u8], output: &mut [u8]) -> (Iconv, usize, usize) {
        // SAFETY: the caller passes valid UTF-8 only.
        let text = unsafe { std::str::from_utf8_unchecked(input) };
        let (r, read, written) = e.encode_from_utf8_without_replacement(text, output, true);
        let status = match r {
            encoding_rs::EncoderResult::InputEmpty => Iconv::Done,
            encoding_rs::EncoderResult::OutputFull => Iconv::OutputFull,
            encoding_rs::EncoderResult::Unmappable(_) => Iconv::Invalid,
        };
        (status, read, written)
    }
}

#[cfg(unix)]
impl Converter {
    fn open(to: &std::ffi::CStr, from: &std::ffi::CStr) -> Option<Self> {
        let cd = unsafe { libc::iconv_open(to.as_ptr(), from.as_ptr()) };
        (cd as isize != -1).then_some(Self(cd))
    }

    /// Convert from `input` into `output`; returns (status, consumed, produced).
    fn convert(&mut self, input: &[u8], output: &mut [u8]) -> (Iconv, usize, usize) {
        let mut inp = input.as_ptr() as *mut libc::c_char;
        let mut in_left = input.len();
        let mut outp = output.as_mut_ptr() as *mut libc::c_char;
        let mut out_left = output.len();
        let r = unsafe { libc::iconv(self.0, &mut inp, &mut in_left, &mut outp, &mut out_left) };
        let status = if r == usize::MAX {
            match std::io::Error::last_os_error().raw_os_error() {
                Some(libc::E2BIG) => Iconv::OutputFull,
                Some(libc::EINVAL) => Iconv::Incomplete,
                _ => Iconv::Invalid,
            }
        } else {
            Iconv::Done
        };
        (status, input.len() - in_left, output.len() - out_left)
    }
}

#[cfg(unix)]
impl Drop for Converter {
    fn drop(&mut self) {
        unsafe { libc::iconv_close(self.0) };
    }
}

/// Longest source string scanned for its terminator.
const MAX_SOURCE: usize = 1 << 24;

#[allow(clippy::too_many_arguments)]
fn convert_mbcs_to_utf8(
    ctx: &CpuContext,
    context: u64,
    source: u64,
    source_max: u32,
    source_len: u64,
    destination: u64,
    destination_max: u32,
    destination_len: u64,
    measure: bool,
) -> HleResult {
    // Profiles other than CP932 convert character by character.
    if let Some(profile) = context_profile(ctx, context).ok().filter(|&p| p != Profile::Cp932) {
        if !measure && destination == 0 {
            return ERROR_INVALID_DST_BUFFER;
        }
        let (dst, dst_max) = if measure { (0, 0) } else { (destination, destination_max) };
        return mbcs_to_unicode(ctx, Unicode::Utf8, profile, source, source_max, source_len, dst, dst_max, destination_len);
    }
    if source_len != 0 {
        ctx.write_u32(source_len, 0);
    }
    if destination_len != 0 {
        ctx.write_u32(destination_len, 0);
    }
    if destination != 0 && destination_max != 0 {
        ctx.write_bytes(destination, &[0]);
    }
    if context == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let profile = ctx.read_u64(context).unwrap_or(0);
    let encoding = if profile == 0 { 0 } else { ctx.read_u64(profile).unwrap_or(0) };
    if profile == 0 || encoding == 0 || encoding != cp932_name(ctx) {
        return ERROR_INVALID_PROFILE;
    }
    if source == 0 {
        return ERROR_INVALID_SRC_BUFFER;
    }
    if !measure && destination == 0 {
        return ERROR_INVALID_DST_BUFFER;
    }
    if !measure && destination_max == 0 {
        return ERROR_DST_BUFFER_END;
    }

    let limit = if source_max == 0 { MAX_SOURCE } else { source_max as usize };
    let text = ctx.read_cstr_bytes(source, limit).unwrap_or_default();
    let bounded_end = source_max != 0 && text.len() == source_max as usize;
    let Some(mut conv) = Converter::open(c"UTF-8", c"CP932") else { return ERROR_INVALID_PROFILE };

    let mut consumed = 0usize;
    let mut produced: Vec<u8> = Vec::new();
    let mut result = 0;
    let capacity = if measure { 256 } else { destination_max as usize - 1 };
    loop {
        let mut scratch = vec![0u8; if measure { 256 } else { capacity - produced.len() }];
        let input = &text[consumed..];
        let (status, used, made) = conv.convert(input, &mut scratch);
        consumed += used;
        produced.extend_from_slice(&scratch[..made]);
        match status {
            Iconv::Done => {}
            Iconv::OutputFull => {
                if measure {
                    if consumed < text.len() {
                        continue;
                    }
                } else {
                    result = ERROR_DST_BUFFER_END;
                }
            }
            Iconv::Incomplete => {
                result = if bounded_end { ERROR_SRC_BUFFER_END } else { ERROR_INVALID_ENCODE };
            }
            Iconv::Invalid => {
                let rest = &text[consumed..];
                let lead = rest.first().is_some_and(|&b| is_lead_byte(b));
                let trail = rest.len() >= 2 && is_trail_byte(rest[1]);
                result = if lead && trail { ERROR_UNASSIGNED_CODE } else { ERROR_INVALID_ENCODE };
            }
        }
        break;
    }

    if !measure {
        let mut out = produced.clone();
        out.push(0);
        ctx.write_bytes(destination, &out);
    }
    if source_len != 0 {
        ctx.write_u32(source_len, consumed as u32);
    }
    if destination_len != 0 {
        ctx.write_u32(destination_len, produced.len() as u32);
    }
    result
}

/// sceCesMbcsStrGetUtf8Len(context, src, srcMax, uint32* srcLen, uint32* utf8Len)
fn cb_mbcs_str_get_utf8_len(ctx: &mut CpuContext) -> HleResult {
    let (c, src, src_max, src_len, dst_len) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u32, ctx.rcx(), ctx.r8());
    convert_mbcs_to_utf8(ctx, c, src, src_max, src_len, 0, 0, dst_len, true)
}

/// sceCesMbcsStrToUtf8Str(context, src, srcMax, uint32* srcLen, utf8, utf8Max, uint32* utf8Len)
fn cb_mbcs_str_to_utf8_str(ctx: &mut CpuContext) -> HleResult {
    let (c, src, src_max, src_len, dst, dst_max) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u32, ctx.rcx(), ctx.r8(), ctx.r9() as u32);
    let dst_len = ctx.read_u64(ctx.get(crate::context::CpuRegister::Rsp) + 8).unwrap_or(0);
    convert_mbcs_to_utf8(ctx, c, src, src_max, src_len, dst, dst_max, dst_len, false)
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("ZiDCxUUGbec", "sceCesUcsProfileInitSJis1997Cp932", cb_ucs_profile_init_sjis1997_cp932),
    ("538bRGc6Zo8", "sceCesMbcsUcsContextInit", cb_mbcs_ucs_context_init),
    ("r7Sr1i7KLus", "sceCesMbcsStrGetUtf8Len", cb_mbcs_str_get_utf8_len),
    ("yGKn6vdYInc", "sceCesMbcsStrToUtf8Str", cb_mbcs_str_to_utf8_str),
    ("LPzYZ+FR0BI", "sceCesRefersUcsProfileCp1252", cb_refers_ucs_profile_cp1252),
    ("3Q1gOWWarcw", "sceCesUtf8ToSbc", cb_utf8_to_sbc),
    ("xTd54EEL1Ao", "sceCesSbcToUtf8", cb_sbc_to_utf8),
    ("99HNmYMUJZQ", "sceCesRefersUcsProfileCp1251", cb_refers_ucs_profile_cp1251),
    ("N3dPQZbNQPI", "sceCesUcsProfileInitEucJpCp51932", cb_ucs_profile_init_euc_jp_cp51932),
    ("oH6EMqs-jUM", "sceCesUcsProfileInitEucJpX0208", cb_ucs_profile_init_euc_jp_x0208),
    ("QIcDUVSdXw0", "sceCesUcsProfileInitEucJpX0208Ss2", cb_ucs_profile_init_euc_jp_x0208_ss2),
    ("DnG4HGO0QZo", "sceCesUcsProfileInitEucJpX0208Ss2Ss3", cb_ucs_profile_init_euc_jp_x0208_ss2_ss3),
    ("5USjtgtV6Cg", "sceCesUcsProfileInitSJis1997X0208", cb_ucs_profile_init_sjis1997_x0208),
    ("pJ1jG+C1oRM", "sceCesEucJpToUcs2", cb_euc_jp_to_ucs2),
    ("RZ8ra69Qkxw", "sceCesUtf8StrToMbcsStr", cb_utf8_str_to_mbcs_str),
    ("IljzwFmlBKg", "sceCesUtf8StrGetMbcsLen", cb_utf8_str_get_mbcs_len),
    ("OfNANruoaxs", "sceCesUcs2StrToMbcsStr", cb_ucs2_str_to_mbcs_str),
    ("kz9bGYFvopY", "sceCesUcs2StrGetMbcsLen", cb_ucs2_str_get_mbcs_len),
    ("q9ONK5ZpMVQ", "sceCesMbcsStrToUcs2Str", cb_mbcs_str_to_ucs2_str),
    ("WAIqT36MwJU", "sceCesMbcsStrGetUcs2Len", cb_mbcs_str_get_ucs2_len),
    ("hi75DVfII3Y", "sceCesUcs2ToMbc", cb_ucs2_to_mbc),
    ("EamBMojFPls", "sceCesUtf8ToMbc", cb_utf8_to_mbc),
    ("6EG6Zzcmrbg", "sceCesUtf16ToMbc", cb_utf16_to_mbc),
    ("NGjHS-+WpA4", "sceCesUtf16beToMbc", cb_utf16be_to_mbc),
    ("V+P9rsOiLmM", "sceCesUtf16leToMbc", cb_utf16le_to_mbc),
    ("zO7NlO2unTI", "sceCesUtf32ToMbc", cb_utf32_to_mbc),
    ("LDzoVPBgzc0", "sceCesUtf32beToMbc", cb_utf32be_to_mbc),
    ("W+QrDTbhLkg", "sceCesUtf32leToMbc", cb_utf32le_to_mbc),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS.iter().chain(unicode::EXPORTS) {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    const BASE: u64 = 0x5D52_0000_0000;
    const PROFILE: u64 = BASE;
    const CONTEXT: u64 = BASE + 0x100;
    const SRC: u64 = BASE + 0x1000;
    const DST: u64 = BASE + 0x2000;
    const CONSUMED: u64 = BASE + 0x200;
    const PRODUCED: u64 = BASE + 0x208;
    const STACK: u64 = BASE + 0x300;

    /// The tests share one scratch mapping, so they run one at a time.
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    fn setup(vm: &ps5vm::VirtualMemory) -> CpuContext<'_> {
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(vm);
        ctx.set(Rdi, PROFILE);
        assert_eq!(cb_ucs_profile_init_sjis1997_cp932(&mut ctx), PROFILE as HleResult);
        ctx.set(Rdi, CONTEXT);
        ctx.set(Rsi, PROFILE);
        assert_eq!(cb_mbcs_ucs_context_init(&mut ctx), 0);
        ctx
    }

    fn to_utf8(ctx: &mut CpuContext, src: &[u8], limit: u32, capacity: u32) -> HleResult {
        ctx.write_bytes(SRC, src);
        ctx.write_u64(STACK + 8, PRODUCED);
        ctx.set(Rsp, STACK);
        for (r, v) in [(Rdi, CONTEXT), (Rsi, SRC), (Rdx, limit as u64), (Rcx, CONSUMED), (R8, DST), (R9, capacity as u64)] {
            ctx.set(r, v);
        }
        cb_mbcs_str_to_utf8_str(ctx)
    }

    fn utf8_len(ctx: &mut CpuContext, src: &[u8], limit: u32) -> HleResult {
        ctx.write_bytes(SRC, src);
        for (r, v) in [(Rdi, CONTEXT), (Rsi, SRC), (Rdx, limit as u64), (Rcx, CONSUMED), (R8, PRODUCED)] {
            ctx.set(r, v);
        }
        cb_mbcs_str_get_utf8_len(ctx)
    }

    fn counts(ctx: &CpuContext) -> (u32, u32) {
        (ctx.read_u32(CONSUMED).unwrap(), ctx.read_u32(PRODUCED).unwrap())
    }

    fn check_conversion(ctx: &mut CpuContext, source: &[u8], expected: &[u8]) {
        ctx.write_bytes(DST, &vec![0xCC; expected.len() + 2]);
        assert_eq!(to_utf8(ctx, source, 0, expected.len() as u32 + 1), 0);
        assert_eq!(counts(ctx), (source.len() as u32 - 1, expected.len() as u32));
        let out = ctx.read_bytes(DST, expected.len() + 2).unwrap();
        assert_eq!(&out[..expected.len()], expected);
        assert_eq!(out[expected.len()], 0);
        assert_eq!(out[expected.len() + 1], 0xCC);
        assert_eq!(utf8_len(ctx, source, 0), 0);
        assert_eq!(counts(ctx), (source.len() as u32 - 1, expected.len() as u32));
    }

    #[test]
    fn ces_mappings_and_measurement() {
        let _serial = serial();
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = setup(&vm);
        check_conversion(&mut ctx, &[0], b"");
        check_conversion(
            &mut ctx,
            &[b'S', b'a', b'v', b'e', b' ', 0x93, 0xFA, 0x96, 0x7B, 0xB6, 0],
            b"Save \xe6\x97\xa5\xe6\x9c\xac\xef\xbd\xb6",
        );
        // CP932 maps this character to U+FF5E, unlike strict Shift-JIS (U+301C).
        check_conversion(&mut ctx, &[0x81, 0x60, 0], b"\xef\xbd\x9e");
        let mut source = Vec::new();
        let mut expected = Vec::new();
        for _ in 0..200 {
            source.extend_from_slice(&[0x82, 0xA0]);
            expected.extend_from_slice(b"\xe3\x81\x82");
        }
        source.push(0);
        check_conversion(&mut ctx, &source, &expected);
    }

    #[test]
    fn ces_partial_output() {
        let _serial = serial();
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = setup(&vm);
        let source = [b'A', 0x82, 0xA0, b'B', 0];
        let full = b"A\xe3\x81\x82B";
        for capacity in 1u32..=7 {
            ctx.write_bytes(DST, &[0xCC; 8]);
            ctx.write_u32(CONSUMED, 99);
            ctx.write_u32(PRODUCED, 99);
            let status = to_utf8(&mut ctx, &source, 0, capacity);
            let expected_consumed = match capacity {
                1 => 0,
                2..=4 => 1,
                5 => 3,
                _ => 4,
            };
            let expected_produced = match capacity {
                1 => 0,
                2..=4 => 1,
                5 => 4,
                _ => 5,
            };
            assert_eq!(status, if capacity < 6 { ERROR_DST_BUFFER_END } else { 0 }, "capacity {capacity}");
            assert_eq!(counts(&ctx), (expected_consumed, expected_produced), "capacity {capacity}");
            let out = ctx.read_bytes(DST, 8).unwrap();
            assert_eq!(&out[..expected_produced as usize], &full[..expected_produced as usize]);
            assert_eq!(out[expected_produced as usize], 0);
            assert_eq!(out[capacity as usize], 0xCC);
        }
    }

    #[test]
    fn ces_errors() {
        let _serial = serial();
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = setup(&vm);
        let cases: [(&[u8], u32, HleResult); 5] = [
            (&[b'A', 0x80, 0], 0, ERROR_INVALID_ENCODE),
            (&[b'A', 0x82, 0], 0, ERROR_INVALID_ENCODE),
            (&[b'A', 0x82, 0xA0, 0], 2, ERROR_SRC_BUFFER_END),
            (&[b'A', 0x82, 0x20, 0], 0, ERROR_INVALID_ENCODE),
            (&[b'A', 0x81, 0xAD, 0], 0, ERROR_UNASSIGNED_CODE),
        ];
        for (source, limit, error) in cases {
            ctx.write_bytes(DST, &[0; 16]);
            ctx.write_u32(CONSUMED, 99);
            ctx.write_u32(PRODUCED, 99);
            assert_eq!(to_utf8(&mut ctx, source, limit, 16), error, "{source:x?}");
            assert_eq!(counts(&ctx), (1, 1), "{source:x?}");
            let out = ctx.read_bytes(DST, 2).unwrap();
            assert_eq!(out, [b'A', 0]);
            assert_eq!(utf8_len(&mut ctx, source, limit), error, "{source:x?}");
            assert_eq!(counts(&ctx), (1, 1), "{source:x?}");
        }
    }

    #[test]
    fn ces_argument_checks() {
        let _serial = serial();
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = setup(&vm);
        ctx.set(Rdi, 0);
        assert_eq!(cb_mbcs_ucs_context_init(&mut ctx), ERROR_INVALID_PARAMETER);
        // A context bound to a zeroed (uninitialised) profile.
        ctx.write_bytes(BASE + 0x400, &[0; 256]);
        ctx.set(Rdi, BASE + 0x500);
        ctx.set(Rsi, BASE + 0x400);
        assert_eq!(cb_mbcs_ucs_context_init(&mut ctx), 0);
        ctx.write_bytes(SRC, b"A\0");
        for (r, v) in [(Rdi, BASE + 0x500), (Rsi, SRC), (Rdx, 0), (Rcx, CONSUMED), (R8, PRODUCED)] {
            ctx.set(r, v);
        }
        assert_eq!(cb_mbcs_str_get_utf8_len(&mut ctx), ERROR_INVALID_PROFILE);
        ctx.set(Rdi, CONTEXT);
        ctx.set(Rsi, 0);
        assert_eq!(cb_mbcs_str_get_utf8_len(&mut ctx), ERROR_INVALID_SRC_BUFFER);
        assert_eq!(to_utf8(&mut ctx, b"A\0", 0, 0), ERROR_DST_BUFFER_END);
        ctx.set(R8, 0);
        ctx.set(R9, 4);
        assert_eq!(cb_mbcs_str_to_utf8_str(&mut ctx), ERROR_INVALID_DST_BUFFER);
    }

    type Cb = fn(&mut CpuContext) -> HleResult;

    /// The ToMbc family (measured on hardware): U+3042 encodes to
    /// Shift-JIS 82 A0 from every source form; the scalar forms report
    /// the needed length with DST_BUFFER_END and zero on other refusals,
    /// the buffer forms leave the lengths untouched.
    #[test]
    fn to_mbc_measured_cases() {
        let _serial = serial();
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = setup(&vm);
        ctx.write_bytes(SRC, &[0xE3, 0x81, 0x82]);
        assert_eq!(call(&mut ctx, cb_ucs2_to_mbc, &[0x3042, PROFILE, DST, 4, PRODUCED]), 0);
        assert_eq!(ctx.read_u32(PRODUCED), Some(2));
        assert_eq!(ctx.read_bytes(DST, 2).as_deref(), Some(&[0x82, 0xA0][..]));
        // The needed length is reported with DST_BUFFER_END.
        assert_eq!(call(&mut ctx, cb_ucs2_to_mbc, &[0x3042, PROFILE, DST, 1, PRODUCED]), ERROR_DST_BUFFER_END);
        assert_eq!(ctx.read_u32(PRODUCED), Some(2));
        // A surrogate is out of the code space, the length zeroed.
        assert_eq!(call(&mut ctx, cb_ucs2_to_mbc, &[0xD800, PROFILE, DST, 4, PRODUCED]), ERROR_ILLEGAL_CODE);
        assert_eq!(ctx.read_u32(PRODUCED), Some(0));
        // A profile mismatch is checked before a null output.
        assert_eq!(call(&mut ctx, cb_ucs2_to_mbc, &[0x3042, 0, 0, 0, 0]), ERROR_INVALID_PARAMETER);
        // The UTF-8 buffer form: the source length is set on success only.
        assert_eq!(call(&mut ctx, cb_utf8_to_mbc, &[SRC, 3, CONSUMED, PROFILE, DST, 4, PRODUCED]), 0);
        assert_eq!(ctx.read_u32(CONSUMED), Some(3));
        assert_eq!(ctx.read_u32(PRODUCED), Some(2));
        // A lone low surrogate is an invalid encode; the lengths stay.
        ctx.write_bytes(SRC, &[0x00, 0xDC]);
        ctx.write_u32(CONSUMED, 0xEEEE_EEEE);
        assert_eq!(call(&mut ctx, cb_utf16le_to_mbc, &[SRC, 1, CONSUMED, PROFILE, DST, 4, PRODUCED]), ERROR_INVALID_ENCODE);
        assert_eq!(ctx.read_u32(CONSUMED), Some(0xEEEE_EEEE));
        // The BE form of the same buffer combines into U+3042.
        ctx.write_bytes(SRC, &[0x30, 0x42]);
        assert_eq!(call(&mut ctx, cb_utf16be_to_mbc, &[SRC, 1, CONSUMED, PROFILE, DST, 4, PRODUCED]), 0);
        assert_eq!(ctx.read_bytes(DST, 2).as_deref(), Some(&[0x82, 0xA0][..]));
        // The UTF-32 pointer forms read their unit's byte order.
        ctx.write_bytes(SRC, &0x3042u32.to_le_bytes());
        assert_eq!(call(&mut ctx, cb_utf32le_to_mbc, &[SRC, PROFILE, DST, 4, PRODUCED]), 0);
        ctx.write_bytes(SRC, &0x3042u32.to_be_bytes());
        assert_eq!(call(&mut ctx, cb_utf32be_to_mbc, &[SRC, PROFILE, DST, 4, PRODUCED]), 0);
        assert_eq!(ctx.read_bytes(DST, 2).as_deref(), Some(&[0x82, 0xA0][..]));
        // A scalar above the Unicode range is an invalid encode.
        assert_eq!(call(&mut ctx, cb_utf32_to_mbc, &[0x110000, PROFILE, DST, 4, PRODUCED]), ERROR_INVALID_ENCODE);
        assert_eq!(ctx.read_u32(PRODUCED), Some(0));
    }

    /// Call `f` with up to seven integer arguments (the seventh on the stack).
    fn call(ctx: &mut CpuContext, f: Cb, args: &[u64]) -> HleResult {
        for (&r, &v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
            ctx.set(r, v);
        }
        if let Some(&v) = args.get(6) {
            ctx.write_u64(STACK + 8, v);
        }
        ctx.set(Rsp, STACK);
        f(ctx)
    }

    /// Write `bytes` then a zero to SRC.
    fn source(ctx: &CpuContext, bytes: &[u8]) {
        ctx.write_bytes(SRC, bytes);
        ctx.write_bytes(SRC + bytes.len() as u64, &[0, 0]);
    }

    fn sbc_to_utf8(ctx: &mut CpuContext, profile: u64, sbc: u8) -> (HleResult, Vec<u8>) {
        let rc = call(ctx, cb_sbc_to_utf8, &[profile, sbc as u64, DST, 4, PRODUCED]);
        let len = ctx.read_u32(PRODUCED).unwrap() as usize;
        (rc, ctx.read_bytes(DST, len).unwrap())
    }

    fn utf8_to_sbc(ctx: &mut CpuContext, profile: u64, utf8: &[u8], max: u32) -> (HleResult, u32, u8) {
        source(ctx, utf8);
        ctx.write_u32(CONSUMED, 0);
        let rc = call(ctx, cb_utf8_to_sbc, &[SRC, max as u64, CONSUMED, profile, DST]);
        (rc, ctx.read_u32(CONSUMED).unwrap(), ctx.read_bytes(DST, 1).unwrap()[0])
    }

    #[test]
    fn ces_single_byte_profiles() {
        let _serial = serial();
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = setup(&vm);
        let profile = cb_refers_ucs_profile_cp1252(&mut ctx) as u64;
        assert!(profile != 0 && profile == cb_refers_ucs_profile_cp1252(&mut ctx) as u64);

        for (sbc, utf8) in [(b'A', "A"), (0x80, "\u{20AC}"), (0x9F, "\u{178}"), (0xE9, "\u{E9}"), (0xFF, "\u{FF}")] {
            assert_eq!(sbc_to_utf8(&mut ctx, profile, sbc), (0, utf8.as_bytes().to_vec()), "{sbc:#x}");
        }
        // The undefined bytes of the profile are unassigned (measured on
        // hardware), not their own control characters.
        for sbc in [0x81u8, 0x8D, 0x8F, 0x90, 0x9D] {
            assert_eq!(sbc_to_utf8(&mut ctx, profile, sbc).0, ERROR_UNASSIGNED_CODE, "{sbc:#x}");
        }
        for (utf8, sbc) in [("A", b'A'), ("\u{20AC}", 0x80), ("\u{2122}", 0x99), ("\u{E9}", 0xE9)] {
            let b = utf8.as_bytes();
            assert_eq!(utf8_to_sbc(&mut ctx, profile, b, b.len() as u32 + 4), (0, b.len() as u32, sbc), "{utf8}");
        }
        assert_eq!(utf8_to_sbc(&mut ctx, profile, "\u{8D}".as_bytes(), 5).0, ERROR_UNASSIGNED_CODE);
        for sbc in 1..=255u8 {
            let (rc, utf8) = sbc_to_utf8(&mut ctx, profile, sbc);
            if matches!(sbc, 0x81 | 0x8D | 0x8F | 0x90 | 0x9D) {
                assert_eq!(rc, ERROR_UNASSIGNED_CODE, "{sbc:#x}");
                continue;
            }
            assert_eq!(rc, 0);
            assert_eq!(utf8_to_sbc(&mut ctx, profile, &utf8, utf8.len() as u32).2, sbc);
        }

        assert_eq!(call(&mut ctx, cb_sbc_to_utf8, &[0, b'A' as u64, DST, 4, PRODUCED]), ERROR_INVALID_PARAMETER);
        assert_eq!(call(&mut ctx, cb_sbc_to_utf8, &[profile, b'A' as u64, 0, 4, PRODUCED]), ERROR_INVALID_DST_BUFFER);
        assert_eq!(call(&mut ctx, cb_sbc_to_utf8, &[profile, 0x80, DST, 2, PRODUCED]), ERROR_DST_BUFFER_END);
        assert_eq!(call(&mut ctx, cb_utf8_to_sbc, &[SRC, 1, CONSUMED, 0, DST]), ERROR_INVALID_PARAMETER);
        assert_eq!(call(&mut ctx, cb_utf8_to_sbc, &[0, 1, CONSUMED, profile, DST]), ERROR_INVALID_SRC_BUFFER);
        assert_eq!(call(&mut ctx, cb_utf8_to_sbc, &[SRC, 1, CONSUMED, profile, 0]), ERROR_INVALID_DST_BUFFER);
        let cases: [(&[u8], u32, HleResult); 7] = [
            (b"A", 0, ERROR_SRC_BUFFER_END),
            (b"\xE2\x82\xAC", 2, ERROR_SRC_BUFFER_END),
            (b"\x80", 1, ERROR_INVALID_ENCODE),
            // An overlong encoding and a surrogate are illegal codes
            // (measured on hardware).
            (b"\xC0\xAF", 2, ERROR_ILLEGAL_CODE),
            (b"\xED\xA0\x80", 3, ERROR_ILLEGAL_CODE),
            (b"\xE3\x81\x82", 3, ERROR_UNASSIGNED_CODE),
            (b"\xF0\x9F\x98\x80", 4, ERROR_UNASSIGNED_CODE),
        ];
        for (utf8, max, error) in cases {
            assert_eq!(utf8_to_sbc(&mut ctx, profile, utf8, max).0, error, "{utf8:x?}");
        }

        let cyrillic = cb_refers_ucs_profile_cp1251(&mut ctx) as u64;
        assert!(cyrillic != 0 && cyrillic == cb_refers_ucs_profile_cp1251(&mut ctx) as u64 && cyrillic != profile);
        for (sbc, utf8) in [(b'A', "A"), (0x88, "\u{20AC}"), (0xC0, "\u{410}"), (0xFF, "\u{44F}"), (0xA8, "\u{401}"), (0xB9, "\u{2116}")] {
            assert_eq!(sbc_to_utf8(&mut ctx, cyrillic, sbc), (0, utf8.as_bytes().to_vec()), "{sbc:#x}");
        }
        for (utf8, sbc) in [("\u{41F}", 0xCF), ("\u{451}", 0xB8)] {
            let b = utf8.as_bytes();
            assert_eq!(utf8_to_sbc(&mut ctx, cyrillic, b, b.len() as u32 + 4), (0, b.len() as u32, sbc), "{utf8}");
        }
        for sbc in 1..=255u8 {
            let (rc, utf8) = sbc_to_utf8(&mut ctx, cyrillic, sbc);
            // 0x98 is undefined in the profile: unassigned, not a control
            // character (measured on hardware).
            if sbc == 0x98 {
                assert_eq!(rc, ERROR_UNASSIGNED_CODE);
                continue;
            }
            assert_eq!(rc, 0);
            assert_eq!(utf8_to_sbc(&mut ctx, cyrillic, &utf8, utf8.len() as u32).2, sbc);
        }
        assert_eq!(utf8_to_sbc(&mut ctx, cyrillic, b"\xC3\xA9", 2).0, ERROR_UNASSIGNED_CODE);
        assert_eq!(utf8_to_sbc(&mut ctx, profile, b"\xD0\x90", 2).0, ERROR_UNASSIGNED_CODE);
        // A CP932 profile sheet is not a single-byte profile.
        assert_eq!(call(&mut ctx, cb_sbc_to_utf8, &[PROFILE, b'A' as u64, DST, 4, PRODUCED]), ERROR_INVALID_PARAMETER);
    }

    const EUC_SHEET: u64 = BASE + 0x400;
    const EUC_CONTEXT: u64 = BASE + 0x500;
    const CP1251_CONTEXT: u64 = BASE + 0x600;
    const MBCS: u64 = BASE + 0x3000;

    /// (status, consumed, produced) of a string conversion of `src` (written
    /// to SRC) through `f` into DST of `dst_max` units.
    fn convert(ctx: &mut CpuContext, f: Cb, context: u64, src: &[u8], src_max: u32, dst_max: u32) -> (HleResult, u32, u32) {
        source(ctx, src);
        ctx.write_u32(CONSUMED, 0xEEEE);
        ctx.write_u32(PRODUCED, 0xEEEE);
        let rc = call(ctx, f, &[context, SRC, src_max as u64, CONSUMED, DST, dst_max as u64, PRODUCED]);
        let (a, b) = counts(ctx);
        (rc, a, b)
    }

    /// Like [`convert`] for the length-only variants.
    fn measure(ctx: &mut CpuContext, f: Cb, context: u64, src: &[u8], src_max: u32) -> (HleResult, u32, u32) {
        source(ctx, src);
        ctx.write_u32(CONSUMED, 0xEEEE);
        ctx.write_u32(PRODUCED, 0xEEEE);
        let rc = call(ctx, f, &[context, SRC, src_max as u64, CONSUMED, PRODUCED]);
        let (a, b) = counts(ctx);
        (rc, a, b)
    }

    fn ucs2(units: &[u16]) -> Vec<u8> {
        units.iter().flat_map(|u| u.to_le_bytes()).collect()
    }

    fn dst_ucs2(ctx: &CpuContext, n: usize) -> Vec<u16> {
        ctx.read_bytes(DST, n * 2).unwrap().chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }

    fn euc_to_ucs2(ctx: &mut CpuContext, profile: u64, bytes: &[u8], max: u32) -> (HleResult, u32, u16) {
        ctx.write_bytes(MBCS, bytes);
        ctx.write_u32(CONSUMED, 0);
        let rc = call(ctx, cb_euc_jp_to_ucs2, &[profile, MBCS, max as u64, CONSUMED, DST]);
        (rc, ctx.read_u32(CONSUMED).unwrap(), dst_ucs2(ctx, 1)[0])
    }

    #[test]
    fn ces_euc_jp_and_string_conversions() {
        let _serial = serial();
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = setup(&vm);
        let cyrillic = cb_refers_ucs_profile_cp1251(&mut ctx) as u64;
        assert_eq!(call(&mut ctx, cb_mbcs_ucs_context_init, &[CP1251_CONTEXT, cyrillic]), 0);
        assert_eq!(call(&mut ctx, cb_mbcs_ucs_context_init, &[0, cyrillic]), ERROR_INVALID_PARAMETER);
        assert_eq!(call(&mut ctx, cb_ucs_profile_init_sjis1997_cp932, &[0]), 0);

        assert_eq!(call(&mut ctx, cb_ucs_profile_init_euc_jp_cp51932, &[EUC_SHEET]), EUC_SHEET as HleResult);
        assert_eq!(call(&mut ctx, cb_ucs_profile_init_euc_jp_cp51932, &[0]), 0);
        assert_eq!(call(&mut ctx, cb_mbcs_ucs_context_init, &[EUC_CONTEXT, EUC_SHEET]), 0);

        let euc_source = b"Go \xEF\xBD\xB1!";
        assert_eq!(convert(&mut ctx, cb_utf8_str_to_mbcs_str, EUC_CONTEXT, euc_source, 16, 16), (0, 7, 6));
        assert_eq!(ctx.read_bytes(DST, 6).unwrap(), b"Go \x8E\xB1!");
        assert_eq!(measure(&mut ctx, cb_utf8_str_get_mbcs_len, EUC_CONTEXT, euc_source, 16), (0, 7, 6));
        assert_eq!(convert(&mut ctx, cb_utf8_str_to_mbcs_str, EUC_CONTEXT, euc_source, 16, 4).0, ERROR_DST_BUFFER_END);
        assert_eq!(convert(&mut ctx, cb_utf8_str_to_mbcs_str, 0, euc_source, 16, 16).0, ERROR_INVALID_PARAMETER);
        assert_eq!(convert(&mut ctx, cb_utf8_str_to_mbcs_str, EUC_CONTEXT, b"\xF0\x9F\x98\x80", 4, 16).0, ERROR_UNASSIGNED_CODE);
        assert_eq!(call(&mut ctx, cb_utf8_str_to_mbcs_str, &[EUC_CONTEXT, SRC, 4, CONSUMED, 0, 16, PRODUCED]), ERROR_INVALID_DST_BUFFER);

        assert_eq!(convert(&mut ctx, cb_mbcs_str_to_utf8_str, EUC_CONTEXT, b"Go \x8E\xB1!", 6, 16), (0, 6, 7));
        assert_eq!(ctx.read_bytes(DST, 7).unwrap(), euc_source);
        assert_eq!(measure(&mut ctx, cb_mbcs_str_get_utf8_len, EUC_CONTEXT, b"Go \x8E\xB1!", 6), (0, 6, 7));
        assert_eq!(convert(&mut ctx, cb_mbcs_str_to_utf8_str, EUC_CONTEXT, b"\x8E", 1, 16).0, ERROR_SRC_BUFFER_END);
        assert_eq!(convert(&mut ctx, cb_mbcs_str_to_utf8_str, EUC_CONTEXT, b"\x8E\x41", 2, 16).0, ERROR_INVALID_ENCODE);

        let ok = ucs2(&[b'O' as u16, b'K' as u16, 0xFF9F]);
        assert_eq!(convert(&mut ctx, cb_ucs2_str_to_mbcs_str, EUC_CONTEXT, &ok, 4, 16), (0, 3, 4));
        assert_eq!(ctx.read_bytes(DST, 4).unwrap(), b"OK\x8E\xDF");
        assert_eq!(measure(&mut ctx, cb_ucs2_str_get_mbcs_len, EUC_CONTEXT, &ok, 4), (0, 3, 4));
        assert_eq!(convert(&mut ctx, cb_mbcs_str_to_ucs2_str, EUC_CONTEXT, b"OK\x8E\xDF", 4, 8), (0, 4, 3));
        assert_eq!(dst_ucs2(&ctx, 3), [b'O' as u16, b'K' as u16, 0xFF9F]);
        assert_eq!(measure(&mut ctx, cb_mbcs_str_get_ucs2_len, EUC_CONTEXT, b"OK\x8E\xDF", 4), (0, 4, 3));
        assert_eq!(euc_to_ucs2(&mut ctx, EUC_SHEET, b"\x8E\xB1", 2), (0, 2, 0xFF71));
        assert_eq!(euc_to_ucs2(&mut ctx, cyrillic, b"A", 1).0, ERROR_INVALID_PARAMETER);

        let japanese = "こんにちは日本語".as_bytes();
        let euc = b"\xA4\xB3\xA4\xF3\xA4\xCB\xA4\xC1\xA4\xCF\xC6\xFC\xCB\xDC\xB8\xEC";
        assert_eq!(convert(&mut ctx, cb_utf8_str_to_mbcs_str, EUC_CONTEXT, japanese, 24, 16), (0, 24, 16));
        assert_eq!(ctx.read_bytes(DST, 16).unwrap(), euc);
        assert_eq!(convert(&mut ctx, cb_mbcs_str_to_utf8_str, EUC_CONTEXT, euc, 16, 32), (0, 16, 24));
        assert_eq!(ctx.read_bytes(DST, 24).unwrap(), japanese);
        assert_eq!(convert(&mut ctx, cb_mbcs_str_to_ucs2_str, EUC_CONTEXT, euc, 16, 8), (0, 16, 8));
        let wide = dst_ucs2(&ctx, 8);
        assert_eq!((wide[0], wide[5], wide[7]), (0x3053, 0x65E5, 0x8A9E));
        assert_eq!(convert(&mut ctx, cb_mbcs_str_to_ucs2_str, EUC_CONTEXT, euc, 15, 8).0, ERROR_SRC_BUFFER_END);
        let extensions = ucs2(&[0x2460, 0xFF5E, 0x2252]);
        assert_eq!(convert(&mut ctx, cb_ucs2_str_to_mbcs_str, EUC_CONTEXT, &extensions, 4, 16), (0, 3, 6));
        assert_eq!(ctx.read_bytes(DST, 6).unwrap(), b"\xAD\xA1\xA1\xC1\xA2\xE2");
        assert_eq!(convert(&mut ctx, cb_mbcs_str_to_ucs2_str, EUC_CONTEXT, b"\xA9\xA1", 2, 8).0, ERROR_UNASSIGNED_CODE);
        assert_eq!(convert(&mut ctx, cb_mbcs_str_to_ucs2_str, EUC_CONTEXT, b"\xA4\x41", 2, 8).0, ERROR_INVALID_ENCODE);
        assert_eq!(convert(&mut ctx, cb_mbcs_str_to_ucs2_str, EUC_CONTEXT, b"\xA4", 1, 8).0, ERROR_SRC_BUFFER_END);
        // The console's plain EUC-JP encodes the yen sign through its
        // full-width alias (measured on hardware).
        assert_eq!(convert(&mut ctx, cb_ucs2_str_to_mbcs_str, EUC_CONTEXT, &ucs2(&[0xA5]), 2, 16), (0, 1, 2));
        assert_eq!(ctx.read_bytes(DST, 2).unwrap(), b"\xA1\xEF");

        // Every assigned JIS X 0208 cell round-trips to the first cell of
        // its character.
        let mut assigned = 0;
        for lead in 0xA1..=0xFEu8 {
            for trail in 0xA1..=0xFEu8 {
                let (rc, _, code) = euc_to_ucs2(&mut ctx, EUC_SHEET, &[lead, trail], 2);
                if rc != 0 {
                    continue;
                }
                assigned += 1;
                assert_eq!(convert(&mut ctx, cb_ucs2_str_to_mbcs_str, EUC_CONTEXT, &ucs2(&[code]), 2, 4), (0, 1, 2));
                let again = ctx.read_bytes(DST, 2).unwrap();
                assert_eq!(euc_to_ucs2(&mut ctx, EUC_SHEET, &again, 2), (0, 2, code));
                assert!((again[0], again[1]) <= (lead, trail));
            }
        }
        assert_eq!(assigned, 7336);

        assert_eq!(convert(&mut ctx, cb_mbcs_str_to_utf8_str, CP1251_CONTEXT, b"\xC0\xFF", 2, 16), (0, 2, 4));
        assert_eq!(ctx.read_bytes(DST, 4).unwrap(), b"\xD0\x90\xD1\x8F");
        assert_eq!(convert(&mut ctx, cb_utf8_str_to_mbcs_str, CP1251_CONTEXT, b"\xD0\x90\xD1\x8F", 4, 16), (0, 4, 2));
        assert_eq!(ctx.read_bytes(DST, 2).unwrap(), b"\xC0\xFF");
    }

    #[test]
    fn ces_cp932_string_conversions() {
        let _serial = serial();
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = setup(&vm);
        let text = "Save 日本ｶ".as_bytes();
        let sjis = b"Save \x93\xFA\x96\x7B\xB6";
        assert_eq!(convert(&mut ctx, cb_utf8_str_to_mbcs_str, CONTEXT, text, 64, 16), (0, text.len() as u32, 10));
        assert_eq!(ctx.read_bytes(DST, 10).unwrap(), sjis);
        assert_eq!(measure(&mut ctx, cb_utf8_str_get_mbcs_len, CONTEXT, text, 64), (0, text.len() as u32, 10));
        assert_eq!(convert(&mut ctx, cb_mbcs_str_to_ucs2_str, CONTEXT, sjis, 10, 16), (0, 10, 8));
        assert_eq!(dst_ucs2(&ctx, 8)[5..], [0x65E5, 0x672C, 0xFF76]);
        assert_eq!(measure(&mut ctx, cb_mbcs_str_get_ucs2_len, CONTEXT, sjis, 10), (0, 10, 8));
        assert_eq!(convert(&mut ctx, cb_mbcs_str_to_ucs2_str, CONTEXT, b"\x81\xAD", 2, 16).0, ERROR_UNASSIGNED_CODE);
        assert_eq!(convert(&mut ctx, cb_mbcs_str_to_ucs2_str, CONTEXT, b"\x82", 1, 16).0, ERROR_SRC_BUFFER_END);
        assert_eq!(convert(&mut ctx, cb_ucs2_str_to_mbcs_str, CONTEXT, &ucs2(&[0x0410]), 2, 16), (0, 1, 2));
        assert_eq!(ctx.read_bytes(DST, 2).unwrap(), b"\x84\x40");
        // A context bound to an unknown profile.
        ctx.write_bytes(BASE + 0x700, &[0; 256]);
        assert_eq!(call(&mut ctx, cb_mbcs_ucs_context_init, &[BASE + 0x800, BASE + 0x700]), 0);
        assert_eq!(measure(&mut ctx, cb_utf8_str_get_mbcs_len, BASE + 0x800, b"A", 1).0, ERROR_INVALID_PROFILE);
    }

    /// FNV-1a over the low then the high byte of `value`.
    fn digest(hash: u32, value: u16) -> u32 {
        let hash = (hash ^ (value & 0xFF) as u32).wrapping_mul(0x0100_0193);
        (hash ^ (value >> 8) as u32).wrapping_mul(0x0100_0193)
    }

    /// Shift-JIS bytes of JIS X 0208 row and column (1-based).
    fn jis_to_sjis(row: u32, column: u32) -> [u8; 2] {
        let lead = if row <= 62 { (row + 1) / 2 + 0x80 } else { (row - 63) / 2 + 0xE0 };
        let trail = if row % 2 == 0 {
            column + 0x9E
        } else if column < 64 {
            column + 0x3F
        } else {
            column + 0x40
        };
        [lead as u8, trail as u8]
    }

    fn decode(profile: Profile, bytes: &[u8]) -> Result<(u32, u32), HleResult> {
        decode_mbcs(profile, |i| bytes.get(i).copied().unwrap_or(0), bytes.len() as u32)
    }

    /// Sweep `cells` through `profile`: every assigned cell encodes back to
    /// itself; (assigned, digest of the decoded characters, 0 for a hole).
    fn sweep(profile: Profile, cells: impl Iterator<Item = [u8; 2]>) -> (u32, u32) {
        let (mut assigned, mut hash) = (0, 0x811C_9DC5u32);
        for bytes in cells {
            let value = match decode(profile, &bytes) {
                Ok((2, code)) => {
                    assigned += 1;
                    assert_eq!(encode_mbcs(profile, code), Ok(bytes.to_vec()), "{bytes:x?}");
                    code as u16
                }
                Ok(other) => panic!("{bytes:x?}: {other:?}"),
                Err(e) => {
                    assert_eq!(e, ERROR_UNASSIGNED_CODE, "{bytes:x?}");
                    0
                }
            };
            hash = digest(hash, value);
        }
        (assigned, hash)
    }

    /// The strict JIS X 0208 profiles over every two-byte cell: the count
    /// and digest of the decoded characters are the console's (measured on
    /// hardware), and every character encodes back to its cell.
    #[test]
    fn ces_strict_jisx0208_planes_match_the_console() {
        for profile in [Profile::EucJpX208, Profile::EucJpX208Ss2, Profile::EucJpX208Ss2Ss3] {
            let cells = (0xA1..=0xFEu8).flat_map(|lead| (0xA1..=0xFEu8).map(move |trail| [lead, trail]));
            assert_eq!(sweep(profile, cells), (6879, 0xC1C9_8702), "{profile:?}");
        }
        let cells = (1..=94).flat_map(|row| (1..=94).map(move |column| jis_to_sjis(row, column)));
        assert_eq!(sweep(Profile::SjisX208, cells), (6879, 0x0CE4_6B27));
    }

    /// The cells and bytes whose strict mapping differs between the
    /// Shift-JIS and EUC-JP profiles (measured on hardware).
    #[test]
    fn ces_strict_jisx0208_backslash_yen_and_dash() {
        for profile in [Profile::EucJpX208, Profile::EucJpX208Ss2, Profile::EucJpX208Ss2Ss3] {
            assert_eq!(decode(profile, b"\xA1\xC0"), Ok((2, 0xFF3C)));
            assert_eq!(encode_mbcs(profile, 0xFF3C), Ok(b"\xA1\xC0".to_vec()));
            assert_eq!(decode(profile, b"\xA1\xBD"), Ok((2, 0x2014)));
            assert_eq!(encode_mbcs(profile, 0x2014), Ok(b"\xA1\xBD".to_vec()));
            assert_eq!(encode_mbcs(profile, 0x2015), Err(ERROR_UNASSIGNED_CODE));
            // The yen sign and the overline encode into their full-width
            // cells, which decode to the full-width forms.
            assert_eq!(encode_mbcs(profile, 0x00A5), Ok(b"\xA1\xEF".to_vec()));
            assert_eq!(encode_mbcs(profile, 0x203E), Ok(b"\xA1\xB1".to_vec()));
            assert_eq!(decode(profile, b"\xA1\xEF"), Ok((2, 0xFFE5)));
            assert_eq!(decode(profile, b"\xA1\xB1"), Ok((2, 0xFFE3)));
            // ASCII stays ASCII.
            assert_eq!(decode(profile, b"\\"), Ok((1, 0x5C)));
            assert_eq!(encode_mbcs(profile, 0x5C), Ok(b"\\".to_vec()));
            for cp932_form in [0xFF5E, 0x2225, 0xFF0D, 0xFFE0, 0xFFE1, 0xFFE2, 0x2460, 0x2170] {
                assert_eq!(encode_mbcs(profile, cp932_form), Err(ERROR_UNASSIGNED_CODE), "{cp932_form:#x}");
            }
        }

        let sjis = Profile::SjisX208;
        for byte in 0..0x80u8 {
            let roman = match byte {
                0x5C => 0x00A5,
                0x7E => 0x203E,
                b => b as u32,
            };
            assert_eq!(decode(sjis, &[byte]), Ok((1, roman)), "{byte:#x}");
            assert_eq!(encode_mbcs(sjis, roman), Ok(vec![byte]), "{byte:#x}");
        }
        for byte in 0xA1..=0xDFu8 {
            assert_eq!(decode(sjis, &[byte]), Ok((1, 0xFF61 + (byte - 0xA1) as u32)));
            assert_eq!(encode_mbcs(sjis, 0xFF61 + (byte - 0xA1) as u32), Ok(vec![byte]));
        }
        assert_eq!(decode(sjis, b"\x81\x5F"), Ok((2, 0x005C)));
        assert_eq!(encode_mbcs(sjis, 0x005C), Ok(b"\x81\x5F".to_vec()));
        assert_eq!(decode(sjis, b"\x81\x5C"), Ok((2, 0x2014)));
        assert_eq!(encode_mbcs(sjis, 0x2014), Ok(b"\x81\x5C".to_vec()));
        for unmapped in [0x007E, 0xFF3C, 0x2015] {
            assert_eq!(encode_mbcs(sjis, unmapped), Err(ERROR_UNASSIGNED_CODE), "{unmapped:#x}");
        }
        assert_eq!(encode_mbcs(sjis, 0x3042), Ok(b"\x82\xA0".to_vec()));
        assert_eq!(encode_mbcs(sjis, 0x301C), Ok(b"\x81\x60".to_vec()));
    }
}
