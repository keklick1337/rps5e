// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Decoder for the Kraken ("newLZ") blocks of PS5 package compression.
//!
//! A compressed package span holds one block of at most 256 KiB of output,
//! split into two 128 KiB sub-chunks. The usual per-chunk headers of the
//! Kraken stream are absent: the package metadata carries them as flags
//! (see [`decode_block`]) plus the compressed size of the first sub-chunk.
//!
//! A sub-chunk is one of:
//! - stored: its compressed length equals its output length;
//! - a bare entropy array that expands to the whole sub-chunk;
//! - an LZ chunk: an optional 8-byte raw seed, then entropy arrays for the
//!   literals, the command bytes, the packed offsets (optionally scaled) and
//!   the packed lengths, followed by a two-ended bit stream with the offset
//!   and long-length extra bits (and an optional "excess" area at the very
//!   end holding the long lengths).
//!
//! Entropy arrays: raw (type 0), Huffman with one three-stream split (type 2)
//! or two halves of three streams each (type 4), with code lengths sent either
//! as adaptive gamma codes or as Golomb-Rice codes with symbol ranges, and
//! the recursive form (type 5) that concatenates sub-arrays.
//!
//! All reads and writes are bounds-checked: corrupt input yields
//! [`KrakenError::Malformed`], never a panic. Work buffers live in a
//! per-thread scratch area so a decode does not allocate.

use std::cell::RefCell;
use std::fmt;

/// Why a block could not be decoded.
#[derive(Debug)]
pub enum KrakenError {
    /// Inconsistent sizes, bad codes or out-of-range references.
    Malformed(&'static str),
    /// A valid encoding this decoder does not implement.
    Unsupported(&'static str),
}

impl fmt::Display for KrakenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KrakenError::Malformed(what) => write!(f, "malformed Kraken block: {what}"),
            KrakenError::Unsupported(what) => write!(f, "unsupported Kraken encoding: {what}"),
        }
    }
}

impl std::error::Error for KrakenError {}

type Result<T> = std::result::Result<T, KrakenError>;

#[cold]
#[inline(never)]
fn malformed<T>(what: &'static str) -> Result<T> {
    Err(KrakenError::Malformed(what))
}

/// Output size of one sub-chunk.
const CHUNK_SIZE: usize = 0x20000;
/// Largest block: two sub-chunks.
const BLOCK_MAX: usize = 2 * CHUNK_SIZE;
/// Slack after the output so copies may run up to 16 bytes past their end.
const PAD: usize = 64;
/// Nesting limit of recursive entropy arrays.
const MAX_DEPTH: u32 = 8;

/// Flag: chunk 0 literals are deltas against the last match source.
pub const FLAG_CHUNK0_SUB_LITERALS: u32 = 0x01;
/// Flag: chunk 0 is an LZ chunk (clear: one bare entropy array).
pub const FLAG_CHUNK0_LZ: u32 = 0x02;
/// Flag: chunk 1 literals are deltas.
pub const FLAG_CHUNK1_SUB_LITERALS: u32 = 0x10;
/// Flag: chunk 1 is an LZ chunk.
pub const FLAG_CHUNK1_LZ: u32 = 0x20;
/// Flag: chunk 1 starts with a fresh window (and its own 8-byte seed).
pub const FLAG_CHUNK1_RESTART: u32 = 0x40;

/// Decode one PS5 package compression block (<= 256 KiB output) into `out`.
/// `flags`: bit 0x01 chunk0 literal mode (set = sub/delta literals, clear = raw), 0x02 chunk0 is LZ (clear = bare entropy array),
/// 0x10 / 0x20 the same for chunk1, 0x40 chunk1 restarts with a fresh window (else chunk1 may reference chunk0).
/// `first_chunk_len`: compressed bytes of the first 128 KiB sub-chunk when out.len() > 128 KiB (ignored otherwise).
pub fn decode_block(
    payload: &[u8],
    flags: u32,
    first_chunk_len: usize,
    out: &mut [u8],
) -> std::result::Result<(), KrakenError> {
    if out.is_empty() {
        return if payload.is_empty() { Ok(()) } else { malformed("payload for an empty block") };
    }
    if out.len() > BLOCK_MAX {
        return Err(KrakenError::Unsupported("block larger than 256 KiB"));
    }
    thread_local! {
        static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
    }
    SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut scratch) => scratch.decode_block(payload, flags, first_chunk_len, out),
        Err(_) => Scratch::default().decode_block(payload, flags, first_chunk_len, out),
    })
}

/// Work buffers reused between blocks.
#[derive(Default)]
struct Scratch {
    /// The block being decoded, with `PAD` bytes of slack.
    buf: Vec<u8>,
    streams: Streams,
}

/// Decoded arrays of one LZ chunk.
#[derive(Default)]
struct Streams {
    lit: Vec<u8>,
    cmd: Vec<u8>,
    offs_packed: Vec<u8>,
    offs_low: Vec<u8>,
    lens_packed: Vec<u8>,
    /// Match distances (positive).
    offs: Vec<u32>,
    /// Long literal and match lengths.
    lens: Vec<u32>,
}

impl Scratch {
    fn decode_block(&mut self, payload: &[u8], flags: u32, first_chunk_len: usize, out: &mut [u8]) -> Result<()> {
        let len0 = out.len().min(CHUNK_SIZE);
        let len1 = out.len() - len0;
        let src0_len = if len1 == 0 { payload.len() } else { first_chunk_len };
        if src0_len == 0 || src0_len > payload.len() {
            return malformed("first sub-chunk size");
        }
        let (src0, src1) = payload.split_at(src0_len);
        if self.buf.len() < BLOCK_MAX + PAD {
            self.buf = vec![0; BLOCK_MAX + PAD];
        }
        let buf = &mut self.buf[..out.len() + PAD];
        decode_sub_chunk(
            buf,
            &mut self.streams,
            src0,
            0,
            len0,
            flags & FLAG_CHUNK0_LZ != 0,
            true,
            flags & FLAG_CHUNK0_SUB_LITERALS != 0,
        )?;
        if len1 != 0 {
            decode_sub_chunk(
                buf,
                &mut self.streams,
                src1,
                len0,
                len1,
                flags & FLAG_CHUNK1_LZ != 0,
                flags & FLAG_CHUNK1_RESTART != 0,
                flags & FLAG_CHUNK1_SUB_LITERALS != 0,
            )?;
        }
        out.copy_from_slice(&buf[..out.len()]);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_sub_chunk(
    buf: &mut [u8],
    st: &mut Streams,
    src: &[u8],
    start: usize,
    len: usize,
    lz: bool,
    restart: bool,
    sub_literals: bool,
) -> Result<()> {
    if src.len() == len {
        buf[start..start + len].copy_from_slice(src);
        return Ok(());
    }
    if !lz {
        let (data, used) = decode_bytes(src, len, &mut st.lit, 0)?;
        if used != src.len() || data.len() != len {
            return malformed("bare entropy chunk size");
        }
        buf[start..start + len].copy_from_slice(data);
        return Ok(());
    }
    decode_lz_chunk(buf, st, src, start, len, restart, sub_literals)
}

// ---------------------------------------------------------------------------
// Entropy arrays
// ---------------------------------------------------------------------------

/// Header of one entropy array.
struct ArrayHeader {
    kind: u8,
    header_len: usize,
    src_size: usize,
    dst_size: usize,
}

fn array_header(src: &[u8]) -> Result<ArrayHeader> {
    if src.len() < 2 {
        return malformed("array header");
    }
    let kind = (src[0] >> 4) & 7;
    let (header_len, src_size, dst_size);
    if kind == 0 {
        if src[0] >= 0x80 {
            header_len = 2;
            src_size = ((src[0] as usize) << 8 | src[1] as usize) & 0xFFF;
        } else {
            if src.len() < 3 {
                return malformed("array header");
            }
            let v = (src[0] as usize) << 16 | (src[1] as usize) << 8 | src[2] as usize;
            if v & !0x3FFFF != 0 {
                return malformed("raw array size");
            }
            header_len = 3;
            src_size = v;
        }
        dst_size = src_size;
    } else if src[0] >= 0x80 {
        if src.len() < 3 {
            return malformed("array header");
        }
        let v = (src[0] as usize) << 16 | (src[1] as usize) << 8 | src[2] as usize;
        header_len = 3;
        src_size = v & 0x3FF;
        dst_size = src_size + ((v >> 10) & 0x3FF) + 1;
    } else {
        if src.len() < 5 {
            return malformed("array header");
        }
        let v = u32::from_be_bytes([src[1], src[2], src[3], src[4]]);
        header_len = 5;
        src_size = (v & 0x3FFFF) as usize;
        dst_size = ((((v >> 18) | (src[0] as u32) << 14) & 0x3FFFF) + 1) as usize;
        if src_size >= dst_size {
            return malformed("entropy array sizes");
        }
    }
    if src.len() - header_len < src_size {
        return malformed("array exceeds its chunk");
    }
    Ok(ArrayHeader { kind, header_len, src_size, dst_size })
}

/// Decodes the entropy array at the start of `src` (at most `cap` bytes of
/// output). Returns the decoded bytes (borrowed from `src` for a raw array,
/// else from `scratch`) and the number of source bytes used.
fn decode_bytes<'a>(src: &'a [u8], cap: usize, scratch: &'a mut Vec<u8>, depth: u32) -> Result<(&'a [u8], usize)> {
    let h = array_header(src)?;
    if h.dst_size > cap {
        return malformed("array larger than its limit");
    }
    let body = &src[h.header_len..h.header_len + h.src_size];
    let used = h.header_len + h.src_size;
    if h.kind == 0 {
        return Ok((body, used));
    }
    if scratch.len() < h.dst_size {
        scratch.resize(h.dst_size.max(CHUNK_SIZE), 0);
    }
    let dst = &mut scratch[..h.dst_size];
    decode_entropy(h.kind, body, dst, depth)?;
    Ok((dst, used))
}

/// Decodes the body of a non-raw array into exactly `dst`.
fn decode_entropy(kind: u8, body: &[u8], dst: &mut [u8], depth: u32) -> Result<()> {
    match kind {
        2 => decode_huffman(body, dst, false),
        4 => decode_huffman(body, dst, true),
        5 => decode_recursive(body, dst, depth),
        1 => Err(KrakenError::Unsupported("tANS entropy array")),
        3 => Err(KrakenError::Unsupported("RLE entropy array")),
        _ => malformed("unknown entropy array type"),
    }
}

/// Type 5: a list of arrays whose outputs are concatenated.
fn decode_recursive(body: &[u8], dst: &mut [u8], depth: u32) -> Result<()> {
    if depth >= MAX_DEPTH {
        return malformed("entropy arrays nested too deep");
    }
    if body.len() < 6 {
        return malformed("recursive array size");
    }
    let mut count = body[0] & 0x7F;
    if count < 2 {
        return malformed("recursive array count");
    }
    if body[0] & 0x80 != 0 {
        return Err(KrakenError::Unsupported("multi-array entropy array"));
    }
    let mut sp = 1;
    let mut op = 0;
    while count != 0 {
        let h = array_header(&body[sp..])?;
        if h.dst_size > dst.len() - op {
            return malformed("recursive array overflows");
        }
        let sub = &body[sp + h.header_len..sp + h.header_len + h.src_size];
        let out = &mut dst[op..op + h.dst_size];
        if h.kind == 0 {
            out.copy_from_slice(sub);
        } else {
            decode_entropy(h.kind, sub, out, depth + 1)?;
        }
        op += h.dst_size;
        sp += h.header_len + h.src_size;
        count -= 1;
    }
    if op != dst.len() || sp != body.len() {
        return malformed("recursive array size mismatch");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Bit reader (MSB first, 24..32 bits buffered)
// ---------------------------------------------------------------------------

/// Reads a byte range forward or backward; past the range it reads zeros.
/// `bitpos` counts the bits shifted out since the register held 24 fresh
/// bits; refilling adds whole bytes while it is positive.
struct BitReader<'a> {
    src: &'a [u8],
    p: isize,
    lo: isize,
    hi: isize,
    bits: u32,
    bitpos: i32,
}

impl<'a> BitReader<'a> {
    fn forward(src: &'a [u8], lo: usize, hi: usize) -> Self {
        let mut r = BitReader { src, p: lo as isize, lo: lo as isize, hi: hi as isize, bits: 0, bitpos: 24 };
        r.refill_f();
        r
    }

    fn backward(src: &'a [u8], lo: usize, hi: usize) -> Self {
        let mut r = BitReader { src, p: hi as isize, lo: lo as isize, hi: hi as isize, bits: 0, bitpos: 24 };
        r.refill_b();
        r
    }

    #[inline]
    fn refill_f(&mut self) {
        while self.bitpos > 0 {
            let b = if self.p >= self.lo && self.p < self.hi { self.src[self.p as usize] } else { 0 };
            self.bits |= (b as u32) << self.bitpos;
            self.bitpos -= 8;
            self.p += 1;
        }
    }

    #[inline]
    fn refill_b(&mut self) {
        while self.bitpos > 0 {
            self.p -= 1;
            let b = if self.p >= self.lo && self.p < self.hi { self.src[self.p as usize] } else { 0 };
            self.bits |= (b as u32) << self.bitpos;
            self.bitpos -= 8;
        }
    }

    #[inline]
    fn consume(&mut self, n: u32) {
        self.bits = ((self.bits as u64) << n) as u32;
        self.bitpos += n as i32;
    }

    /// Reads `n` (0..=32) bits without refilling.
    #[inline]
    fn bits_nr(&mut self, n: u32) -> u32 {
        let r = ((self.bits as u64) >> (32 - n.min(32))) as u32;
        self.consume(n.min(32));
        r
    }

    #[inline]
    fn bit_nr(&mut self) -> u32 {
        self.bits_nr(1)
    }

    #[inline]
    fn bit(&mut self) -> u32 {
        self.refill_f();
        self.bits_nr(1)
    }

    /// Byte position after the consumed bits, partial bytes counted as used.
    fn byte_pos(&self) -> isize {
        self.p - ((24 - self.bitpos) / 8) as isize
    }

    /// `n` (0..=26) bits; refills afterwards.
    fn bits_long(&mut self, n: u32, fwd: bool) -> u32 {
        let r = if n <= 24 {
            self.bits_nr(n)
        } else {
            let hi = self.bits_nr(24) << (n - 24);
            if fwd { self.refill_f() } else { self.refill_b() }
            hi.wrapping_add(self.bits_nr(n - 24))
        };
        if fwd { self.refill_f() } else { self.refill_b() }
        r
    }

    /// Match distance for packed offset symbol `v`.
    fn distance(&mut self, v: u32, fwd: bool) -> u32 {
        let r = if v < 0xF0 {
            let n = (v >> 4) + 4;
            let top = self.bits_nr(n) | 1 << n;
            (top << 4).wrapping_add(v & 0xF).wrapping_sub(248)
        } else {
            let n = v - 0xF0 + 4;
            let top = self.bits_nr(n) | 1 << n;
            let x = 0x7F_0000u32.wrapping_add(top << 12);
            if fwd { self.refill_f() } else { self.refill_b() }
            x.wrapping_add(self.bits_nr(12))
        };
        if fwd { self.refill_f() } else { self.refill_b() }
        r
    }

    /// Extra length (Elias-gamma-like, at most 12 leading zeros).
    fn length(&mut self, fwd: bool) -> Option<u32> {
        let n = self.bits.leading_zeros();
        if n > 12 {
            return None;
        }
        self.consume(n);
        if fwd { self.refill_f() } else { self.refill_b() }
        let v = self.bits_nr(n + 7).wrapping_sub(64);
        if fwd { self.refill_f() } else { self.refill_b() }
        Some(v)
    }
}

// ---------------------------------------------------------------------------
// Huffman arrays
// ---------------------------------------------------------------------------

/// First slot in the symbol table for each code length (1..=11).
const CODE_PREFIX_ORG: [u32; 12] = [0, 0, 2, 6, 14, 30, 62, 126, 254, 510, 766, 1022];

/// 11-bit bit reversal.
static REV11: [u16; 2048] = {
    let mut t = [0u16; 2048];
    let mut i = 0;
    while i < 2048 {
        let mut r = 0u16;
        let mut b = 0;
        while b < 11 {
            r = (r << 1) | ((i >> b) & 1) as u16;
            b += 1;
        }
        t[i] = r;
        i += 1;
    }
    t
};

/// Decoding table indexed by the next 11 bits (LSB first): symbol in the
/// low byte, code length in the high byte.
type HuffLut = [u16; 2048];

fn decode_huffman(a: &[u8], dst: &mut [u8], halves: bool) -> Result<()> {
    let mut br = BitReader::forward(a, 0, a.len());
    let mut syms = [0u8; 1280];
    let mut prefix = CODE_PREFIX_ORG;
    let num_syms = if br.bit_nr() == 0 {
        code_lengths_gamma(&mut br, &mut syms, &mut prefix)?
    } else if br.bit_nr() == 0 {
        code_lengths_rice(&mut br, &mut syms, &mut prefix)?
    } else {
        return malformed("Huffman code length transmission");
    };
    let pos = br.byte_pos();
    if num_syms == 1 {
        dst.fill(syms[0]);
        return Ok(());
    }
    let lut = make_lut(&prefix, &syms)?;
    if pos < 0 {
        return malformed("Huffman header");
    }
    let mut pos = pos as usize;
    let end = a.len();
    if !halves {
        if pos + 3 > end {
            return malformed("Huffman split");
        }
        let mid = le16(a, pos);
        pos += 2;
        huff_streams(a, pos, pos + mid, end, dst, &lut)
    } else {
        if pos + 6 > end {
            return malformed("Huffman split");
        }
        let half = dst.len().div_ceil(2);
        let first_len = le16(a, pos) | (a[pos + 2] as usize) << 16;
        pos += 3;
        if first_len > end - pos {
            return malformed("Huffman split");
        }
        let e1 = pos + first_len;
        let mid1 = le16(a, pos);
        pos += 2;
        if e1 < pos || e1 - pos < mid1 + 2 || end - e1 < 3 {
            return malformed("Huffman split");
        }
        let mid2 = le16(a, e1);
        if end - (e1 + 2) < mid2 + 2 {
            return malformed("Huffman split");
        }
        let (d1, d2) = dst.split_at_mut(half);
        huff_streams(a, pos, pos + mid1, e1, d1, &lut)?;
        huff_streams(a, e1 + 2, e1 + 2 + mid2, end, d2, &lut)
    }
}

#[inline]
fn le16(a: &[u8], p: usize) -> usize {
    a[p] as usize | (a[p + 1] as usize) << 8
}

#[inline(always)]
fn rd32(a: &[u8], p: usize) -> u32 {
    u32::from_le_bytes([a[p], a[p + 1], a[p + 2], a[p + 3]])
}

/// Decodes `dst` from three interleaved streams in `a`: forward from
/// `src`, backward from `end` and forward from `mid` (symbol i goes to
/// stream i % 3). All three must end exactly where the next one begins.
fn huff_streams(a: &[u8], src: usize, mid: usize, end: usize, dst: &mut [u8], lut: &HuffLut) -> Result<()> {
    if src > mid || mid > end || end > a.len() {
        return malformed("Huffman streams");
    }
    let mid_org = mid;
    let (mut s0, mut s1, mut s2) = (src as isize, end as isize, mid as isize);
    let (mut b0, mut b1, mut b2) = (0u32, 0u32, 0u32);
    let (mut p0, mut p1, mut p2) = (0i32, 0i32, 0i32);
    let n = dst.len();
    let mut d = 0usize;

    if end - mid >= 4 && n >= 6 {
        s1 -= 4;
        while d + 6 <= n && s0 <= s2 && s2 <= s1 {
            b0 |= rd32(a, s0 as usize) << p0;
            s0 += ((31 - p0) >> 3) as isize;
            p0 |= 24;
            b1 |= rd32(a, s1 as usize).swap_bytes() << p1;
            s1 -= ((31 - p1) >> 3) as isize;
            p1 |= 24;
            b2 |= rd32(a, s2 as usize) << p2;
            s2 += ((31 - p2) >> 3) as isize;
            p2 |= 24;
            let out = &mut dst[d..d + 6];
            macro_rules! sym {
                ($b:ident, $p:ident, $i:expr) => {{
                    let e = lut[($b & 0x7FF) as usize];
                    let l = (e >> 8) as u32;
                    $b >>= l;
                    $p -= l as i32;
                    out[$i] = e as u8;
                }};
            }
            sym!(b0, p0, 0);
            sym!(b1, p1, 1);
            sym!(b2, p2, 2);
            sym!(b0, p0, 3);
            sym!(b1, p1, 4);
            sym!(b2, p2, 5);
            d += 6;
        }
        s0 -= (p0 >> 3) as isize;
        p0 &= 7;
        s1 += 4 + (p1 >> 3) as isize;
        p1 &= 7;
        s2 -= (p2 >> 3) as isize;
        p2 &= 7;
        if s0 > s2 || s2 > s1 {
            return malformed("Huffman streams overlap");
        }
    }

    let (mut s0, mut s1, mut s2) = (s0 as usize, s1 as usize, s2 as usize);
    while d < n {
        if s2 - s0 >= 2 {
            b0 |= (a[s0] as u32 | (a[s0 + 1] as u32) << 8) << p0;
        } else if s2 - s0 == 1 {
            b0 |= (a[s0] as u32) << p0;
        }
        let e = lut[(b0 & 0x7FF) as usize];
        let l = (e >> 8) as i32;
        p0 -= l;
        b0 >>= l;
        dst[d] = e as u8;
        d += 1;
        s0 += ((7 - p0) >> 3) as usize;
        p0 &= 7;
        if d < n {
            if s1 - s2 >= 2 {
                b1 |= (a[s1 - 1] as u32 | (a[s1 - 2] as u32) << 8) << p1;
                b2 |= (a[s2] as u32 | (a[s2 + 1] as u32) << 8) << p2;
            } else if s1 - s2 == 1 {
                b1 |= (a[s2] as u32) << p1;
                b2 |= (a[s2] as u32) << p2;
            }
            let e = lut[(b1 & 0x7FF) as usize];
            let l = (e >> 8) as i32;
            dst[d] = e as u8;
            d += 1;
            p1 -= l;
            b1 >>= l;
            let back = ((7 - p1) >> 3) as usize;
            if back > s1 {
                return malformed("Huffman streams overlap");
            }
            s1 -= back;
            p1 &= 7;
            if d < n {
                let e = lut[(b2 & 0x7FF) as usize];
                let l = (e >> 8) as i32;
                dst[d] = e as u8;
                d += 1;
                p2 -= l;
                b2 >>= l;
                s2 += ((7 - p2) >> 3) as usize;
                p2 &= 7;
            }
        }
        if s0 > s2 || s2 > s1 {
            return malformed("Huffman streams overlap");
        }
    }
    if s0 != mid_org || s1 != s2 {
        return malformed("Huffman streams do not meet");
    }
    Ok(())
}

/// Code lengths sent as adaptive Elias-gamma deltas (or a short explicit
/// list of symbols).
fn code_lengths_gamma(br: &mut BitReader, syms: &mut [u8; 1280], prefix: &mut [u32; 12]) -> Result<usize> {
    if br.bit_nr() != 0 {
        let mut sym: i32 = 0;
        let mut total: i32 = 0;
        let mut avg: i32 = 32;
        let k = br.bits_nr(2) as i32;
        let thresh: u32 = 1 << (31 - (20 >> k));
        let mut skip_first = br.bit() != 0;
        loop {
            if !skip_first {
                if br.bits & 0xFF00_0000 == 0 {
                    return malformed("Huffman gamma code");
                }
                let c = br.bits.leading_zeros();
                sym += br.bits_nr(2 * (c + 1)) as i32 - 2 + 1;
                if sym >= 256 {
                    break;
                }
            }
            skip_first = false;
            br.refill_f();
            if br.bits & 0xFF00_0000 == 0 {
                return malformed("Huffman gamma code");
            }
            let c = br.bits.leading_zeros();
            let mut run = br.bits_nr(2 * (c + 1)) as i32 - 2 + 1;
            if sym + run > 256 {
                return malformed("Huffman symbol run");
            }
            br.refill_f();
            total += run;
            loop {
                if br.bits < thresh {
                    return malformed("Huffman length code");
                }
                let c = br.bits.leading_zeros() as i32;
                let v = br.bits_nr((c + k + 1) as u32) as i32 + ((c - 1) << k);
                let len = (-(v & 1) ^ (v >> 1)) + ((avg + 2) >> 2);
                if !(1..=11).contains(&len) {
                    return malformed("Huffman code length");
                }
                avg = len + ((3 * avg + 2) >> 2);
                br.refill_f();
                let slot = &mut prefix[len as usize];
                syms[*slot as usize] = sym as u8;
                *slot += 1;
                sym += 1;
                run -= 1;
                if run == 0 {
                    break;
                }
            }
            if sym == 256 {
                break;
            }
        }
        if sym != 256 || total < 2 {
            return malformed("Huffman symbol count");
        }
        return Ok(total as usize);
    }
    let num = br.bits_nr(8) as usize;
    match num {
        0 => return malformed("Huffman without symbols"),
        1 => syms[0] = br.bits_nr(8) as u8,
        _ => {
            let width = br.bits_nr(3);
            if width > 4 {
                return malformed("Huffman length width");
            }
            for _ in 0..num {
                br.refill_f();
                let sym = br.bits_nr(8) as u8;
                let len = br.bits_nr(width) as usize + 1;
                if len > 11 {
                    return malformed("Huffman code length");
                }
                let slot = &mut prefix[len];
                syms[*slot as usize] = sym;
                *slot += 1;
            }
        }
    }
    Ok(num)
}

/// Plain MSB-first reader used for the Golomb-Rice part.
struct ByteBits<'a> {
    src: &'a [u8],
    p: usize,
    /// Bits of `src[p]` already used (0..=7).
    bit: u32,
    end: usize,
}

/// Code lengths sent as Golomb-Rice deltas against a running average, with
/// the used symbols given as ranges ("fluff" entries).
fn code_lengths_rice(br: &mut BitReader, syms: &mut [u8; 1280], prefix: &mut [u32; 12]) -> Result<usize> {
    let rice_k = br.bits_nr(2);
    let num = br.bits_nr(8) as usize + 1;
    let fluff = read_fluff(br, num);
    let total = num + fluff;
    let mut vals = [0u8; 528];
    if total + 16 > vals.len() {
        return malformed("Huffman range count");
    }
    let used_bytes = (24 - br.bitpos + 7) >> 3;
    let start = br.p - used_bytes as isize;
    if start < br.lo {
        return malformed("Huffman header");
    }
    let mut bb = ByteBits { src: br.src, p: start as usize, bit: ((br.bitpos - 24) & 7) as u32, end: br.hi as usize };

    // Unary parts: zeros terminated by a one.
    let mut i = 0;
    let mut zeros = 0u32;
    while i < total {
        if bb.p >= bb.end {
            return malformed("Huffman Rice codes");
        }
        let byte = bb.src[bb.p] << bb.bit;
        let avail = 8 - bb.bit;
        if byte == 0 {
            zeros += avail;
            bb.p += 1;
            bb.bit = 0;
            continue;
        }
        let z = byte.leading_zeros();
        vals[i] = (zeros + z) as u8;
        i += 1;
        zeros = 0;
        bb.bit += z + 1;
        if bb.bit == 8 {
            bb.bit = 0;
            bb.p += 1;
        }
    }
    // Low bits of the code length deltas (not of the ranges).
    if rice_k != 0 {
        let nbits = bb.bit as usize + rice_k as usize * num;
        if (nbits + 7) >> 3 > bb.end - bb.p {
            return malformed("Huffman Rice bits");
        }
        let mut pos = bb.p * 8 + bb.bit as usize;
        for v in vals.iter_mut().take(num) {
            let mut x = 0u32;
            for _ in 0..rice_k {
                x = (x << 1) | ((bb.src[pos >> 3] >> (7 - (pos & 7))) & 1) as u32;
                pos += 1;
            }
            *v = (((*v as u32) << rice_k) | x) as u8;
        }
        bb.p = pos >> 3;
        bb.bit = (pos & 7) as u32;
    }
    // Back to the main reader.
    *br = BitReader::forward(br.src, bb.p, bb.end);
    br.lo = br.lo.min(bb.p as isize);
    br.consume(bb.bit);

    let mut avg: u32 = 30;
    for v in vals.iter_mut().take(num) {
        let x = *v as i32;
        let delta = -(x & 1) ^ (x >> 1);
        let len = delta + (avg >> 2) as i32 + 1;
        if !(1..=11).contains(&len) {
            return malformed("Huffman code length");
        }
        *v = len as u8;
        avg = avg.wrapping_add(delta as u32);
    }

    // Symbol ranges.
    let pairs = fluff >> 1;
    let mut lens_at = num;
    let mut sym = 0usize;
    if fluff & 1 != 0 {
        br.refill_f();
        let n = vals[lens_at] as u32;
        lens_at += 1;
        if n >= 8 {
            return malformed("Huffman range");
        }
        sym = br.bits_nr(n + 1) as usize + (1 << (n + 1)) - 1;
    }
    let mut ranges = [(0usize, 0usize); 129];
    let mut counted = 0usize;
    for r in ranges.iter_mut().take(pairs) {
        br.refill_f();
        let n = vals[lens_at] as u32;
        if n >= 9 {
            return malformed("Huffman range");
        }
        let count = br.bits_nr(n) as usize + (1 << n);
        let n = vals[lens_at + 1] as u32;
        if n >= 8 {
            return malformed("Huffman range");
        }
        let gap = br.bits_nr(n + 1) as usize + (1 << (n + 1)) - 1;
        *r = (sym, count);
        counted += count;
        sym += count + gap;
        lens_at += 2;
    }
    if sym >= 256 || counted >= num || sym + num - counted > 256 {
        return malformed("Huffman ranges");
    }
    ranges[pairs] = (sym, num - counted);

    let mut li = 0;
    for &(first, count) in ranges.iter().take(pairs + 1) {
        for s in first..first + count {
            let slot = &mut prefix[vals[li] as usize];
            li += 1;
            syms[*slot as usize] = s as u8;
            *slot += 1;
        }
    }
    Ok(num)
}

/// Number of range entries that follow the code lengths.
fn read_fluff(br: &mut BitReader, num: usize) -> usize {
    if num == 256 {
        return 0;
    }
    let x = (257 - num).min(num) * 2;
    let width = usize::BITS - (x - 1).leading_zeros();
    let v = ((br.bits as u64) >> (32 - width)) as usize;
    let lim = (1usize << width) - x;
    if v >> 1 >= lim {
        br.consume(width);
        v - lim
    } else {
        br.consume(width - 1);
        v >> 1
    }
}

/// Builds the LSB-first decoding table of a canonical code; the code must
/// be complete.
fn make_lut(prefix: &[u32; 12], syms: &[u8; 1280]) -> Result<HuffLut> {
    let mut msb = [0u16; 2048];
    let mut at = 0usize;
    for len in 1..=11usize {
        let first = CODE_PREFIX_ORG[len] as usize;
        let count = prefix[len] as usize - first;
        let step = 1usize << (11 - len);
        if count == 0 {
            continue;
        }
        if count * step > 2048 - at {
            return malformed("Huffman code oversubscribed");
        }
        for &s in &syms[first..first + count] {
            msb[at..at + step].fill((len as u16) << 8 | s as u16);
            at += step;
        }
    }
    if at != 2048 {
        return malformed("Huffman code incomplete");
    }
    let mut lut = [0u16; 2048];
    for (i, e) in lut.iter_mut().enumerate() {
        *e = msb[REV11[i] as usize];
    }
    Ok(lut)
}

// ---------------------------------------------------------------------------
// LZ chunks
// ---------------------------------------------------------------------------

fn decode_lz_chunk(
    buf: &mut [u8],
    st: &mut Streams,
    src: &[u8],
    start: usize,
    len: usize,
    restart: bool,
    sub_literals: bool,
) -> Result<()> {
    let mut sp = 0;
    let mut end = src.len();
    if restart {
        if src.len() < 8 || len < 8 {
            return malformed("LZ chunk too short");
        }
        buf[start..start + 8].copy_from_slice(&src[..8]);
        sp = 8;
    }
    let mut excess = None;
    if sp < end && src[sp] & 0x80 != 0 {
        let b = src[sp];
        sp += 1;
        if b & 0xC0 != 0x80 {
            return malformed("LZ chunk flags");
        }
        let mut n = (b & 0x3F) as usize;
        if n > 31 {
            if sp >= end {
                return malformed("LZ excess size");
            }
            n += src[sp] as usize * 32;
            sp += 1;
        }
        if end - sp < n {
            return malformed("LZ excess size");
        }
        end -= n;
        excess = Some(n);
    }

    let Streams { lit, cmd, offs_packed, offs_low, lens_packed, offs, lens } = st;
    let (lit, used) = decode_bytes(&src[sp..end], len, lit, 0)?;
    sp += used;
    let (cmd, used) = decode_bytes(&src[sp..end], len, cmd, 0)?;
    sp += used;
    if end - sp < 3 {
        return malformed("LZ chunk truncated");
    }
    let mut scaling = 0u32;
    let mut low: &[u8] = &[];
    let packed;
    if src[sp] & 0x80 != 0 {
        scaling = src[sp] as u32 - 127;
        sp += 1;
        let (p, used) = decode_bytes(&src[sp..end], cmd.len(), offs_packed, 0)?;
        sp += used;
        packed = p;
        if scaling != 1 {
            let (l, used) = decode_bytes(&src[sp..end], packed.len(), offs_low, 0)?;
            if l.len() != packed.len() {
                return malformed("scaled offset arrays differ");
            }
            sp += used;
            low = l;
        }
    } else {
        let (p, used) = decode_bytes(&src[sp..end], cmd.len(), offs_packed, 0)?;
        sp += used;
        packed = p;
    }
    let (lens_packed, used) = decode_bytes(&src[sp..end], len >> 2, lens_packed, 0)?;
    sp += used;

    unpack_offsets(src, sp, end, excess, packed, scaling, low, lens_packed, offs, lens)?;
    let window = if restart { start } else { 0 };
    run_commands(buf, start, len, window, restart, sub_literals, lit, cmd, offs, lens)
}

#[allow(clippy::too_many_arguments)]
fn unpack_offsets(
    src: &[u8],
    begin: usize,
    end: usize,
    excess: Option<usize>,
    packed: &[u8],
    scaling: u32,
    low: &[u8],
    lens_packed: &[u8],
    offs: &mut Vec<u32>,
    lens: &mut Vec<u32>,
) -> Result<()> {
    let mut fa = BitReader::forward(src, begin, end);
    let mut fb = BitReader::backward(src, begin, end);
    let num_long;
    if excess.is_none() {
        if fb.bits < 0x2000 {
            return malformed("LZ long length count");
        }
        let n = fb.bits.leading_zeros();
        fb.consume(n);
        fb.refill_b();
        let n = n + 1;
        num_long = (fb.bits_nr(n) - 1) as usize;
        fb.refill_b();
    } else {
        num_long = lens_packed.iter().filter(|&&b| b == 255).count();
    }

    offs.clear();
    offs.reserve(packed.len());
    if scaling == 0 {
        for (i, &v) in packed.iter().enumerate() {
            let d = if i & 1 == 0 { fa.distance(v as u32, true) } else { fb.distance(v as u32, false) };
            if d == 0 || d > i32::MAX as u32 {
                return malformed("LZ offset");
            }
            offs.push(d);
        }
    } else {
        for (i, &v) in packed.iter().enumerate() {
            let v = v as u32;
            let n = v >> 3;
            if n > 26 {
                return malformed("LZ offset");
            }
            let extra = if i & 1 == 0 { fa.bits_long(n, true) } else { fb.bits_long(n, false) };
            let raw = ((8 + (v & 7)) << n) | extra;
            let mut d = raw as i64 - 8;
            if scaling != 1 {
                d = d * scaling as i64 + low[i] as i64;
            }
            if d <= 0 || d > i32::MAX as i64 {
                return malformed("LZ offset");
            }
            offs.push(d as u32);
        }
    }

    if num_long > 512 {
        return malformed("LZ long length count");
    }
    let mut long = [0u32; 512];
    {
        let (mut ra, mut rb, lim_check) = match excess {
            Some(n) => (BitReader::forward(src, end, end + n), BitReader::backward(src, end, end + n), false),
            None => (fa, fb, true),
        };
        let mut j = 0;
        while j + 1 < num_long {
            long[j] = ra.length(true).ok_or(KrakenError::Malformed("LZ long length"))?;
            long[j + 1] = rb.length(false).ok_or(KrakenError::Malformed("LZ long length"))?;
            j += 2;
        }
        if j < num_long {
            long[j] = ra.length(true).ok_or(KrakenError::Malformed("LZ long length"))?;
        }
        if lim_check {
            let a_end = ra.p - ((24 - ra.bitpos) >> 3) as isize;
            let b_end = rb.p + ((24 - rb.bitpos) >> 3) as isize;
            if a_end != b_end {
                return malformed("LZ offset bit streams do not meet");
            }
        }
    }

    lens.clear();
    lens.reserve(lens_packed.len());
    let mut li = 0;
    for &b in lens_packed {
        let mut v = b as u32;
        if v == 255 {
            if li >= num_long {
                return malformed("LZ long lengths");
            }
            v = long[li].wrapping_add(255);
            li += 1;
        }
        lens.push(v.wrapping_add(3));
    }
    if li != num_long {
        return malformed("LZ long lengths unused");
    }
    Ok(())
}

#[inline(always)]
fn copy8(buf: &mut [u8], from: usize, to: usize) {
    let v: [u8; 8] = buf[from..from + 8].try_into().unwrap();
    buf[to..to + 8].copy_from_slice(&v);
}

#[inline(always)]
fn copy16(buf: &mut [u8], from: usize, to: usize) {
    let v: [u8; 16] = buf[from..from + 16].try_into().unwrap();
    buf[to..to + 16].copy_from_slice(&v);
}

/// Copies `n` literals to `buf[pos..]`, adding the byte `last` positions
/// back for delta literals. `buf` has `PAD` slack past every chunk.
#[inline(always)]
fn put_literals(buf: &mut [u8], pos: usize, lit: &[u8], n: usize, sub: bool, last: usize) {
    if !sub {
        if lit.len() >= 16 && n <= 16 {
            buf[pos..pos + 16].copy_from_slice(&lit[..16]);
        } else {
            buf[pos..pos + n].copy_from_slice(&lit[..n]);
        }
    } else if last >= 8 {
        let mut i = 0;
        while i + 8 <= n {
            let a = u64::from_le_bytes(lit[i..i + 8].try_into().unwrap());
            let b = u64::from_le_bytes(buf[pos + i - last..pos + i - last + 8].try_into().unwrap());
            const H: u64 = 0x8080_8080_8080_8080;
            let s = ((a & !H) + (b & !H)) ^ ((a ^ b) & H);
            buf[pos + i..pos + i + 8].copy_from_slice(&s.to_le_bytes());
            i += 8;
        }
        while i < n {
            buf[pos + i] = lit[i].wrapping_add(buf[pos + i - last]);
            i += 1;
        }
    } else {
        for i in 0..n {
            buf[pos + i] = lit[i].wrapping_add(buf[pos + i - last]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_commands(
    buf: &mut [u8],
    start: usize,
    len: usize,
    window: usize,
    seeded: bool,
    sub: bool,
    lit: &[u8],
    cmd: &[u8],
    offs: &[u32],
    lens: &[u32],
) -> Result<()> {
    let end = start + len;
    let mut pos = if seeded { start + 8 } else { start };
    if pos > end || end + PAD > buf.len() {
        return malformed("LZ chunk size");
    }
    let (mut lp, mut oi, mut li) = (0usize, 0usize, 0usize);
    let mut recent = [8u32; 7];
    let mut last = 8usize;
    for &c in cmd {
        let mut lit_len = (c & 3) as usize;
        let idx = (c >> 6) as usize;
        let short_ml = ((c >> 2) & 15) as usize;
        if lit_len == 3 {
            lit_len = match lens.get(li) {
                Some(&v) => v as usize,
                None => return malformed("LZ lengths exhausted"),
            };
            li += 1;
        }
        recent[6] = offs.get(oi).copied().unwrap_or(0);
        if lit_len != 0 {
            if lit.len() - lp < lit_len || end - pos < lit_len {
                return malformed("LZ literals overflow");
            }
            put_literals(buf, pos, &lit[lp..], lit_len, sub, last);
            pos += lit_len;
            lp += lit_len;
        }
        let dist = recent[idx + 3];
        recent[idx + 3] = recent[idx + 2];
        recent[idx + 2] = recent[idx + 1];
        recent[idx + 1] = recent[idx];
        recent[3] = dist;
        oi += (idx + 1) >> 2;
        let dist = dist as usize;
        if dist == 0 || dist > pos - window {
            return malformed("LZ match before the window");
        }
        last = dist;
        let ml = if short_ml != 15 {
            short_ml + 2
        } else {
            let v = match lens.get(li) {
                Some(&v) => v as usize,
                None => return malformed("LZ lengths exhausted"),
            };
            li += 1;
            14 + v
        };
        if end - pos < ml {
            return malformed("LZ match overflow");
        }
        let from = pos - dist;
        if dist >= 16 {
            let mut i = 0;
            while i < ml {
                copy16(buf, from + i, pos + i);
                i += 16;
            }
        } else if dist >= 8 {
            let mut i = 0;
            while i < ml {
                copy8(buf, from + i, pos + i);
                i += 8;
            }
        } else {
            for i in 0..ml {
                buf[pos + i] = buf[from + i];
            }
        }
        pos += ml;
    }
    if oi != offs.len() || li != lens.len() {
        return malformed("LZ streams not fully used");
    }
    let tail = end - pos;
    if tail != lit.len() - lp {
        return malformed("LZ literal tail size");
    }
    // The tail is copied exactly: nothing is written past the chunk end.
    if !sub {
        buf[pos..end].copy_from_slice(&lit[lp..]);
    } else {
        put_literals(buf, pos, &lit[lp..], tail, true, last);
    }
    Ok(())
}

#[cfg(test)]
#[path = "kraken_tests.rs"]
pub(crate) mod tests;
