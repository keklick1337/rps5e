// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! MUBUF / MTBUF semantics shared by the interpreter: V# address and range
//! checks, and the buffer format conversions (loads, stores and their D16
//! forms). Every conversion is exact integer arithmetic so a SPIR-V lowering
//! can reproduce it bit for bit.

use crate::resources::format::unified_format;
use super::eval::Val;
use super::InterpError;
use crate::ssa::{Access, BufferAttr};

// ---------------------------------------------------------------------------
// Addressing and range checking
// ---------------------------------------------------------------------------

/// The V# fields a buffer access depends on.
#[derive(Clone, Copy, Debug)]
pub struct BufferDescriptor {
    pub base: u64,
    pub stride: u32,
    pub records: u32,
    /// Unified format (dword3 bits 12..18).
    pub format: u32,
    /// DST_SEL_X..W (dword3 bits 0..11, three bits each).
    pub dst_sel: [u32; 4],
    /// OOB_SELECT (dword3 bits 28..29).
    pub oob_select: u32,
    /// SWIZZLE_EN (dword1 bit 31).
    pub swizzle: bool,
    /// INDEX_STRIDE (dword3 bits 21..22): 8 << n records per swizzle group.
    pub index_stride: u32,
    /// ADD_TID_ENABLE (dword3 bit 23): the lane is added to the index.
    pub add_tid: bool,
}

impl BufferDescriptor {
    pub fn new(desc: [u32; 4]) -> Self {
        BufferDescriptor {
            base: desc[0] as u64 | (((desc[1] & 0xFFFF) as u64) << 32),
            stride: (desc[1] >> 16) & 0x3FFF,
            records: desc[2],
            format: (desc[3] >> 12) & 0x7F,
            dst_sel: [desc[3] & 7, (desc[3] >> 3) & 7, (desc[3] >> 6) & 7, (desc[3] >> 9) & 7],
            oob_select: (desc[3] >> 28) & 3,
            swizzle: desc[1] >> 31 != 0,
            index_stride: (desc[3] >> 21) & 3,
            add_tid: (desc[3] >> 23) & 1 != 0,
        }
    }
}

/// One invocation's buffer address operands.
#[derive(Clone, Copy, Debug)]
pub struct BufferLaneAddress {
    /// Record index (0 without `idxen`).
    pub index: u32,
    /// `voffset` (0 without `offen`) plus the instruction offset, wrapping.
    pub offset: u32,
    pub soffset: u32,
}

impl BufferLaneAddress {
    /// `lane`: the invocation's lane (added to the index with ADD_TID).
    pub fn new(attr: &BufferAttr, d: &BufferDescriptor, lane: u32, vindex: u32, voffset: u32, soffset: u32) -> Self {
        let index = if attr.idxen { vindex } else { 0 };
        BufferLaneAddress {
            index: if d.add_tid { index.wrapping_add(lane & 63) } else { index },
            offset: (if attr.offen { voffset } else { 0 }).wrapping_add(attr.offset),
            soffset,
        }
    }
}

/// Guest address of byte `extra` of an access. Without swizzling:
/// `base + soffset + index * stride + offset + extra`; with SWIZZLE_EN
/// (dword1 bit 31, stride != 0) records interleave in groups of
/// `2^(INDEX_STRIDE + 3)` with dword elements:
/// `(index_msb * stride + offset_msb) * n + index_lsb * 4 + offset_lsb`.
/// `extra` (the byte of a multi-dword or formatted access) advances from
/// that one address. Offset arithmetic wraps at 32 bits.
pub fn buffer_byte_address(d: &BufferDescriptor, a: &BufferLaneAddress, extra: u32) -> u64 {
    let offset = a.offset;
    let rel = if d.swizzle && d.stride != 0 {
        let shift = d.index_stride + 3;
        let n = 1u32 << shift;
        let msb = (a.index >> shift).wrapping_mul(d.stride).wrapping_add(offset & !3).wrapping_mul(n);
        let lsb = ((a.index & (n - 1)) << 2).wrapping_add(offset & 3);
        msb.wrapping_add(lsb)
    } else {
        a.index.wrapping_mul(d.stride).wrapping_add(offset)
    };
    d.base.wrapping_add(rel.wrapping_add(a.soffset).wrapping_add(extra) as u64)
}

/// Range check of `bytes` bytes at byte `extra` of an access (each dword
/// of a multi-dword access is checked on its own, with `extra = 4 * k`;
/// formatted accesses and atomics are checked all or nothing by their
/// callers). With `offset' = offset + extra` (not including the index or
/// SOFFSET), by OOB_SELECT (dword3 bits 28..29; the names are the gfx10
/// register field's: STRUCTURED_WITH_OFFSET, STRUCTURED, DISABLED, RAW):
/// * 0 (structured with offset): `index < records` and `offset' + bytes
///   <= stride`. A zero stride puts every access out of range (a dummy V#
///   of mode 0 and stride 0 reads 0).
/// * 1 (structured): `index < records`.
/// * 2: `records != 0`.
/// * 3 (raw): `soffset <= records` and, with `raw = records - soffset`,
///   `offset' + bytes <= raw`; swizzled buffers check `index < raw` and
///   `offset' + bytes <= stride` instead.
pub fn buffer_in_bounds(d: &BufferDescriptor, a: &BufferLaneAddress, extra: u32, bytes: u32) -> bool {
    let offset = a.offset.wrapping_add(extra) as u64;
    let bytes = bytes as u64;
    let payload = |size: u64| size >= bytes && offset <= size - bytes;
    let stride = d.stride as u64;
    match d.oob_select {
        0 => a.index < d.records && payload(stride),
        1 => a.index < d.records,
        2 => d.records != 0,
        _ => {
            let records = d.records as u64;
            let soffset = a.soffset as u64;
            if soffset > records {
                return false;
            }
            let raw = records - soffset;
            if d.swizzle && d.stride != 0 {
                (a.index as u64) < raw && payload(stride)
            } else {
                payload(raw)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Formats
// ---------------------------------------------------------------------------

/// Number format class of a buffer format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormatKind {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
}

impl FormatKind {
    fn signed(self) -> bool {
        matches!(self, FormatKind::Snorm | FormatKind::Sscaled | FormatKind::Sint)
    }

    fn integer(self) -> bool {
        matches!(self, FormatKind::Uint | FormatKind::Sint)
    }
}

/// Layout of a buffer format element.
#[derive(Clone, Copy, Debug)]
pub struct FormatInfo {
    pub kind: FormatKind,
    pub count: u32,
    pub bits: [u32; 4],
    /// Byte offset of each component (0 for packed fields).
    pub byte_offset: [u32; 4],
    /// Bit offset of each component inside its dword (packed fields only).
    pub bit_offset: [u32; 4],
    /// Components share one dword (10_11_11, 2_10_10_10, ...).
    pub packed: bool,
}

impl FormatInfo {
    /// Layout of a unified format; `None` for the invalid format and the
    /// encodings without a buffer layout.
    pub fn new(unified: u32) -> Option<Self> {
        let (dfmt, nfmt) = unified_format::decode(unified)?;
        let count = unified_format::component_count(dfmt);
        if count == 0 {
            return None;
        }
        let kind = match nfmt {
            0 | 9 => FormatKind::Unorm,
            1 => FormatKind::Snorm,
            2 => FormatKind::Uscaled,
            3 => FormatKind::Sscaled,
            4 => FormatKind::Uint,
            5 => FormatKind::Sint,
            7 => FormatKind::Float,
            _ => return None,
        };
        let mut info = FormatInfo { kind, count, bits: [0; 4], byte_offset: [0; 4], bit_offset: [0; 4], packed: matches!(dfmt, 6..=9) };
        for c in 0..count {
            let (byte, bit, bits) = unified_format::component_layout(dfmt, c)?;
            info.bits[c as usize] = bits;
            info.byte_offset[c as usize] = byte;
            info.bit_offset[c as usize] = bit;
        }
        Some(info)
    }

    /// `(byte offset, bytes)` of the memory a component lives in.
    pub fn component_span(&self, c: u32) -> (u32, u32) {
        if self.packed {
            (0, 4)
        } else {
            (self.byte_offset[c as usize], self.bits[c as usize] / 8)
        }
    }

    /// Raw field of component `c` out of its memory (`component_span`),
    /// sign-extended for signed number formats.
    pub fn extract(&self, c: u32, memory: u32) -> u32 {
        let bits = self.bits[c as usize];
        let shift = if self.packed { self.bit_offset[c as usize] } else { 0 };
        let field = if bits >= 32 { memory } else { (memory >> shift) & ((1u32 << bits) - 1) };
        if self.kind.signed() && bits < 32 {
            (((field << (32 - bits)) as i32) >> (32 - bits)) as u32
        } else {
            field
        }
    }
}

/// Where an output component of a formatted load comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormatSource {
    Memory(u32),
    Zero,
    One,
}

/// Source of output component `out`: typed (MTBUF) loads read components
/// in order (missing ones are 0); untyped loads follow the descriptor's
/// DST_SEL (0 = 0, 1 = 1, 4..7 = X..W modulo the format's component count;
/// the reserved selectors 2 and 3 read 0).
pub fn format_source(info: &FormatInfo, typed: bool, dst_sel: [u32; 4], out: u32) -> FormatSource {
    if typed {
        return if out < info.count { FormatSource::Memory(out) } else { FormatSource::Zero };
    }
    match dst_sel[out as usize & 3] {
        1 => FormatSource::One,
        s @ 4..=7 => FormatSource::Memory((s - 4) % info.count),
        _ => FormatSource::Zero,
    }
}

/// Register value of a constant source.
pub fn format_constant(info: &FormatInfo, source: FormatSource, d16: bool) -> u32 {
    match source {
        FormatSource::One if info.kind.integer() => 1,
        FormatSource::One if d16 => 0x3C00,
        FormatSource::One => 0x3F80_0000,
        _ => 0,
    }
}

fn find_msb(v: u32) -> u32 {
    31 - v.leading_zeros()
}

/// `value / max` (exact) as f32 bits, rounded to nearest.
fn unorm_to_f32(value: u32, max: u32) -> u32 {
    if value == 0 {
        return 0;
    }
    // Both operands are exact in f64 and the quotient is never a rounding
    // midpoint of f32, so the double rounding is exact.
    ((value as f64 / max as f64) as f32).to_bits()
}

/// Unsigned 11- or 10-bit float (5-bit exponent, no sign) to f32 bits.
pub fn ufloat_to_f32(raw: u32, bits: u32) -> u32 {
    let mantissa_bits: u32 = if bits == 11 { 6 } else { 5 };
    let mantissa = raw & ((1 << mantissa_bits) - 1);
    let exponent = (raw >> mantissa_bits) & 0x1F;
    let wide = mantissa << (23 - mantissa_bits);
    match exponent {
        0 => (mantissa as f32 * (2.0f32).powi(-14 - mantissa_bits as i32)).to_bits(),
        31 => 0x7F80_0000 | wide,
        e => ((e + 127 - 15) << 23) | wide,
    }
}

/// Load conversion of one extracted component to a 32-bit register value.
pub fn load_component(info: &FormatInfo, c: u32, raw: u32) -> u32 {
    let bits = info.bits[c as usize];
    match info.kind {
        FormatKind::Uint | FormatKind::Sint => raw,
        FormatKind::Uscaled => (raw as f32).to_bits(),
        FormatKind::Sscaled => (raw as i32 as f32).to_bits(),
        FormatKind::Unorm => unorm_to_f32(raw, if bits >= 32 { u32::MAX } else { (1 << bits) - 1 }),
        FormatKind::Snorm => {
            let max = (1u32 << (bits - 1)) - 1;
            let negative = (raw as i32) < 0;
            let magnitude = if negative { raw.wrapping_neg() } else { raw }.min(max);
            (if negative { 0x8000_0000 } else { 0 }) | unorm_to_f32(magnitude, max)
        }
        FormatKind::Float => match bits {
            32 => raw,
            16 => crate::interp::eval::f16_to_f32(raw as u16).to_bits(),
            _ => ufloat_to_f32(raw, bits),
        },
    }
}

/// `value / maximum` as f16 bits, rounded to nearest.
fn norm_to_f16(value: u32, maximum: u32) -> u32 {
    if value == 0 {
        return 0;
    }
    let width = 32 - maximum.leading_zeros();
    let first = (9 + width).wrapping_sub(find_msb(value));
    let adjusted = if (value << first) < 1024 * maximum { first + 1 } else { first };
    let shift = adjusted.min(24);
    let scaled = value << shift;
    let quotient = scaled / maximum;
    let up = scaled % maximum > maximum >> 1;
    let exponent = (25 - shift) << 10;
    (exponent + quotient + up as u32).wrapping_sub(1024)
}

/// Unsigned integer as f16 bits, rounded to nearest even, clamped to the
/// largest finite value.
fn integer_to_f16(value: u32) -> u32 {
    if value == 0 {
        return 0;
    }
    let msb = find_msb(value);
    let wide = msb > 10;
    let (left, right) = if wide { (0, msb - 10) } else { (10 - msb, 0) };
    let quotient = (value << left) >> right;
    let unit = 1u32 << right;
    let remainder = value & (unit - 1);
    let half = unit >> 1;
    let tie = remainder == half && quotient & 1 != 0;
    let up = wide && (remainder > half || tie);
    let exponent = (msb + 15) << 10;
    (exponent + quotient + up as u32 - 1024).min(0x7BFF)
}

/// f32 bits to a small float with a 5-bit exponent and `mantissa_bits`
/// mantissa bits, rounding toward zero: finite overflow gives the largest
/// finite value, infinities stay infinite, NaN keeps the top payload bits
/// (or becomes the quiet pattern with only bit 0 when they are all zero).
/// Unsigned formats (`signed == false`) store negative non-NaN values as 0.
pub fn f32_to_small_float_rtz(bits: u32, mantissa_bits: u32, signed: bool) -> u32 {
    let infinity = 0x1F << mantissa_bits;
    let negative = bits & 0x8000_0000 != 0;
    let sign = if signed && negative { 1 << (5 + mantissa_bits) } else { 0 };
    let exponent = (bits >> 23) & 0xFF;
    let mantissa = bits & 0x7F_FFFF;
    let top = mantissa >> (23 - mantissa_bits);
    let biased = exponent as i32 - 112;
    let mut magnitude = if biased >= 31 {
        infinity - 1
    } else if biased >= 1 {
        ((biased as u32) << mantissa_bits) | top
    } else if biased >= -(mantissa_bits as i32) {
        (mantissa | 0x80_0000) >> (24 - mantissa_bits as i32 - biased) as u32
    } else {
        0
    };
    if exponent == 0xFF {
        magnitude = infinity | if top == 0 && mantissa != 0 { 1 } else { top };
    }
    if !signed && negative && !(exponent == 0xFF && mantissa != 0) {
        magnitude = 0;
    }
    sign | magnitude
}

/// f32 bits to a UNORM (`signed == false`) or SNORM field of magnitude
/// `maximum`: the exact product `|value| * maximum` rounded to nearest
/// even, saturated at `maximum`; NaN and non-positive UNORM inputs give 0.
fn norm_store(bits: u32, maximum: u32, signed: bool) -> u32 {
    let value = f32::from_bits(bits);
    let magnitude = f32::from_bits(bits & 0x7FFF_FFFF);
    let exponent = (bits >> 23) & 0xFF;
    let denormal = exponent == 0;
    let mantissa = (bits & 0x7F_FFFF) | if denormal { 0 } else { 0x80_0000 };
    // value = mantissa * 2^-shift
    let shift: u32 = if denormal { 149 } else { 150u32.wrapping_sub(exponent) };
    let extra = shift.wrapping_sub(24);
    let mut result = if extra > 16 {
        0
    } else {
        let product = mantissa as u64 * maximum as u64;
        let integer = (product >> 24) as u32;
        let fraction = product as u32 & 0xFF_FFFF;
        let fixed = (integer << 16) | (fraction >> 8);
        let sticky = fraction & 0xFF != 0;
        let quotient = (fixed >> 16) >> extra;
        let remainder = fixed.wrapping_sub((quotient << 16) << extra);
        let half = (1u32 << 15) << extra;
        let tie = remainder == half && (sticky || quotient & 1 != 0);
        quotient + (remainder > half || tie) as u32
    };
    if magnitude >= 1.0 {
        result = maximum;
    }
    // NaN, zero and (UNORM) negative inputs store 0.
    let positive = if signed { magnitude } else { value } > 0.0;
    if !positive {
        result = 0;
    }
    if signed && bits & 0x8000_0000 != 0 {
        result.wrapping_neg()
    } else {
        result
    }
}

/// Store conversion of a 32-bit register value to component `c`'s field
/// (unshifted, masked to the field width).
pub fn store_component(info: &FormatInfo, c: u32, data: u32) -> u32 {
    let bits = info.bits[c as usize];
    let mask = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
    match info.kind {
        FormatKind::Uint => data.min(mask),
        FormatKind::Sint if bits >= 32 => data,
        FormatKind::Sint => {
            let high = (mask >> 1) as i32;
            ((data as i32).clamp(!high, high) as u32) & mask
        }
        FormatKind::Unorm | FormatKind::Uscaled => norm_store(data, mask, false),
        FormatKind::Snorm | FormatKind::Sscaled => norm_store(data, mask >> 1, true) & mask,
        FormatKind::Float => match bits {
            32 => data,
            16 => f32_to_small_float_rtz(data, 10, true),
            _ => f32_to_small_float_rtz(data, bits - 5, false),
        },
    }
}

/// D16 load conversion of one extracted component to a 16-bit value.
pub fn load_component_d16(info: &FormatInfo, c: u32, raw: u32) -> u32 {
    let bits = info.bits[c as usize];
    match info.kind {
        FormatKind::Uint if bits > 16 => raw.min(0xFFFF),
        FormatKind::Sint if bits > 16 => ((raw as i32).clamp(-0x8000, 0x7FFF) as u32) & 0xFFFF,
        FormatKind::Uint | FormatKind::Sint => raw & 0xFFFF,
        FormatKind::Unorm => norm_to_f16(raw, if bits >= 32 { u32::MAX } else { (1 << bits) - 1 }),
        FormatKind::Snorm => {
            let max = (1u32 << (bits - 1)) - 1;
            let negative = (raw as i32) < 0;
            let magnitude = if negative { raw.wrapping_neg() } else { raw }.min(max);
            (if negative { 0x8000 } else { 0 }) | norm_to_f16(magnitude, max)
        }
        FormatKind::Uscaled => integer_to_f16(raw),
        FormatKind::Sscaled => {
            let negative = (raw as i32) < 0;
            (if negative { 0x8000 } else { 0 }) | integer_to_f16(if negative { raw.wrapping_neg() } else { raw })
        }
        FormatKind::Float if bits == 16 => raw & 0xFFFF,
        FormatKind::Float => f32_to_small_float_rtz(load_component(info, c, raw), 10, true),
    }
}

/// D16 store conversion of a 16-bit register half to component `c`'s field.
pub fn store_component_d16(info: &FormatInfo, c: u32, half: u32) -> u32 {
    let half = half & 0xFFFF;
    match info.kind {
        FormatKind::Uint => store_component(info, c, half),
        FormatKind::Sint => store_component(info, c, half as u16 as i16 as i32 as u32),
        FormatKind::Float if info.bits[c as usize] == 16 => half,
        _ => store_component(info, c, f16_bits_to_f32(half)),
    }
}

/// Exact f16 to f32 bit conversion (NaN payloads kept).
fn f16_bits_to_f32(half: u32) -> u32 {
    let sign = (half & 0x8000) << 16;
    let exponent = (half >> 10) & 0x1F;
    let mantissa = half & 0x3FF;
    let magnitude = match exponent {
        0 if mantissa == 0 => 0,
        0 => {
            let msb = find_msb(mantissa);
            ((msb + 103) << 23) | ((mantissa << (23 - msb)) & 0x7F_FFFF)
        }
        31 => 0x7F80_0000 | (mantissa << 13),
        e => ((e + 112) << 23) | (mantissa << 13),
    };
    sign | magnitude
}

// ---------------------------------------------------------------------------
// Interpreter accesses
// ---------------------------------------------------------------------------

impl super::Run<'_, '_, '_> {
    /// One invocation's raw access; `store` is `Some(exec bit)` for stores.
    pub(super) fn buffer_raw(&mut self, d: &BufferDescriptor, a: &BufferLaneAddress, access: Access, store: Option<bool>, data: Option<Val>) -> Val {
        let mut result = [0u64; 4];
        let (units, bytes) = match access {
            Access::Dwords(k) => (k as u32, 4),
            _ => (1, access.bytes()),
        };
        for k in 0..units {
            let extra = 4 * k;
            if !buffer_in_bounds(d, a, extra, bytes) {
                continue;
            }
            let mut addr = buffer_byte_address(d, a, extra);
            if bytes == 4 {
                // Dword accesses drop the low two bits of the offset from
                // the base; a base off a dword boundary stays as it is.
                addr = d.base.wrapping_add(addr.wrapping_sub(d.base) & !3);
            }
            match store {
                Some(true) => self.m.mem_write(addr, bytes, data.unwrap()[k as usize]),
                Some(false) => {}
                None => {
                    let x = self.m.mem_read(addr, bytes);
                    result[k as usize] = match access {
                        Access::I8 => x as u8 as i8 as i32 as u32 as u64,
                        Access::I16 => x as u16 as i16 as i32 as u32 as u64,
                        _ => x,
                    };
                }
            }
        }
        result
    }

    /// One invocation's formatted access; `store` is `Some(exec bit)` for
    /// stores.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn buffer_format(
        &mut self,
        d: &BufferDescriptor,
        a: &BufferLaneAddress,
        components: u32,
        typed_format: Option<u32>,
        d16: bool,
        store: Option<bool>,
        data: Option<Val>,
    ) -> Result<Val, InterpError> {
        let mut d = *d;
        if let Some(f) = typed_format {
            d.format = f;
        }
        let mut result = [0u64; 4];
        let Some(info) = FormatInfo::new(d.format) else {
            // An invalid format (0 among them) accesses raw dwords.
            return Ok(self.buffer_raw(&d, a, Access::Dwords(components as u8), store, data));
        };
        let in_bounds = |c: u32| {
            let (byte, bytes) = info.component_span(c);
            buffer_in_bounds(&d, a, byte, bytes)
        };
        match store {
            None => {
                let sources: Vec<FormatSource> = (0..components).map(|c| format_source(&info, typed_format.is_some(), d.dst_sel, c)).collect();
                // All or nothing over the whole element, also the
                // components the access does not return.
                let ok = (0..info.count).all(in_bounds);
                for (out, source) in sources.into_iter().enumerate() {
                    result[out] = match source {
                        FormatSource::Memory(c) if ok => {
                            let (byte, bytes) = info.component_span(c);
                            let raw = info.extract(c, self.m.mem_read(buffer_byte_address(&d, a, byte), bytes) as u32);
                            (if d16 { load_component_d16(&info, c, raw) } else { load_component(&info, c, raw) }) as u64
                        }
                        FormatSource::Memory(_) => 0,
                        constant => format_constant(&info, constant, d16) as u64,
                    };
                }
            }
            Some(true) if (0..info.count).all(in_bounds) => {
                let data = data.unwrap();
                let encoded: Vec<u32> = (0..info.count)
                    .map(|c| match c < components {
                        false => 0,
                        true if d16 => store_component_d16(&info, c, data[c as usize] as u32),
                        true => store_component(&info, c, data[c as usize] as u32),
                    })
                    .collect();
                if info.packed {
                    let word = (0..info.count).fold(0u32, |w, c| {
                        let bits = info.bits[c as usize];
                        w | ((encoded[c as usize] & ((1u32 << bits) - 1)) << info.bit_offset[c as usize])
                    });
                    self.m.mem_write(buffer_byte_address(&d, a, 0), 4, word as u64);
                } else {
                    for c in 0..info.count {
                        let (byte, bytes) = info.component_span(c);
                        self.m.mem_write(buffer_byte_address(&d, a, byte), bytes, encoded[c as usize] as u64);
                    }
                }
            }
            Some(_) => {}
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_store_rounds_the_exact_product() {
        // 0.50000003 * 255 = 127.500008 -> 128; 0.5 * 255 = 127.5 -> 128 (even).
        assert_eq!(norm_store(0x3F00_0001, 255, false), 128);
        assert_eq!(norm_store(0.5f32.to_bits(), 255, false), 128);
        assert_eq!(norm_store((1.5f32 / 255.0).to_bits(), 255, false), 2);
        assert_eq!(norm_store(1.0f32.to_bits(), 255, false), 255);
        assert_eq!(norm_store((-1.0f32).to_bits(), 127, true), (-127i32) as u32);
        assert_eq!(norm_store(f32::NAN.to_bits(), 255, false), 0);
    }

    #[test]
    fn small_floats() {
        assert_eq!(f32_to_small_float_rtz(65520.0f32.to_bits(), 10, true), 0x7BFF);
        assert_eq!(f32_to_small_float_rtz(1.0f32.to_bits(), 10, true), 0x3C00);
        assert_eq!(f32_to_small_float_rtz(f32::INFINITY.to_bits(), 10, true), 0x7C00);
        assert_eq!(f32_to_small_float_rtz(1.0f32.to_bits(), 6, false), 0x3C0);
        assert_eq!(ufloat_to_f32(0x3C0, 11), 1.0f32.to_bits());
        assert_eq!(ufloat_to_f32(0x1E0, 10), 1.0f32.to_bits());
        assert_eq!(f16_bits_to_f32(0x0001), (2.0f32).powi(-24).to_bits());
        assert_eq!(integer_to_f16(2049), 0x6800);
        assert_eq!(norm_to_f16(255, 255), 0x3C00);
        assert_eq!(norm_to_f16(1, 65535), 0x0100);
    }
}
