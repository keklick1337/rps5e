// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! NAPS: the block map of a PS5 package's compressed inner image.
//!
//! The outer PFS of a PS5 package holds `pfs_image.dat` (the packed,
//! mostly Kraken-compressed inner image) and `naps_pkg_layout.dat`, which
//! maps the logical inner image onto it. [`NapsImage`] presents the logical
//! image as a [`ReadAt`], decoding spans on demand.
//!
//! Layout file (little-endian):
//! ```text
//! 0x00 u64: bits 0..23 files-1, 24..25 codec, 26..27 keys-1,
//!           28..31 shuffle patterns, 32..54 last ublock index
//! 0x08 u64: bits 0..23 outer blocks, 24..47 span records-2
//! then: outer-block digests (8 B each), shuffle patterns (8 B each),
//!       file starts (6 B: 40-bit logical offset, type byte; type bit 6
//!       marks a continuation of the previous range),
//!       ublock index (10 B per 8 ublocks of 256 KiB: u24 first record,
//!       seven u8 deltas for the next ublocks), padding to 8,
//!       span records (9 B each, 72-bit little-endian words).
//! ```
//! A span record has two forms, told apart by bit 18:
//! - span (bit 18 clear): bits 0..17 stored offset mod 256 KiB, 19 end
//!   marker, 20..37 logical offset mod 256 KiB, 38..54 first sub-chunk
//!   stored length - 1, 55..57 / 58..60 coding modes of the two 128 KiB
//!   sub-chunks, 61..66 predictor, 68..71 shuffle pattern;
//! - run base (bit 18 set): bits 0..17 stored end offset mod 256 KiB of the
//!   previous span, 50..71 stored offset / 256 KiB of the next span.
//!
//! A span's stored length is the distance to the next record's stored
//! offset (mod 256 KiB), its logical length the distance to the next span's
//! logical offset; a span not preceded by a run base starts where the
//! previous one ended. Equal stored and logical lengths mean stored bytes;
//! a few tiny spans stand for blocks of zeros.

use crate::io::{invalid, le_u64, ReadAt, Source};
use crate::kraken;
use std::io;
use std::sync::{Arc, Mutex};

/// File name of the layout in the outer PFS.
pub const LAYOUT_FILE: &str = "naps_pkg_layout.dat";
/// File name of the packed image in the outer PFS.
pub const IMAGE_FILE: &str = "pfs_image.dat";

/// Logical block of the span index (and the largest span).
const UBLOCK: u64 = 0x4_0000;
const HALF: u32 = 0x2_0000;
/// Decoded spans kept per image (256 KiB each).
const CACHE_SPANS: usize = 48;

#[derive(Clone, Copy, Debug)]
struct Span {
    /// Offset in the packed image.
    stored: u64,
    stored_len: u32,
    first_len: u32,
    logical: u64,
    len: u32,
    even: u8,
    odd: u8,
    predictor: u8,
    shuffle: u8,
}

impl Span {
    fn is_stored(&self) -> bool {
        self.stored_len == self.len
    }

    /// A tiny span standing for a block of zeros.
    fn is_zero(&self) -> bool {
        let long = self.odd == 1 && self.stored_len == 16 && self.len > 16;
        let short = self.odd == 0 && self.stored_len == 8 && self.len > 8 && self.len <= HALF;
        (long || short) && self.first_len == 8 && self.even == 1 && self.predictor == 0 && self.shuffle == 0
    }

    /// Kraken flags of the two sub-chunks.
    fn kraken_flags(&self) -> u32 {
        let mut f = 0;
        if self.even & 4 != 0 {
            f |= kraken::FLAG_CHUNK0_LZ;
        }
        if self.even & 2 != 0 {
            f |= kraken::FLAG_CHUNK0_SUB_LITERALS;
        }
        if self.odd & 4 != 0 {
            f |= kraken::FLAG_CHUNK1_LZ;
        }
        if self.odd & 2 != 0 {
            f |= kraken::FLAG_CHUNK1_SUB_LITERALS;
        }
        f
    }
}

/// One 9-byte span record.
#[derive(Clone, Copy, Debug)]
enum Record {
    Span { stored_mod: u32, end: bool, logical_mod: u32, first_len: u32, even: u8, odd: u8, predictor: u8, shuffle: u8 },
    RunBase { stored_end_mod: u32, stored_base: u64 },
}

impl Record {
    fn parse(b: &[u8]) -> Self {
        let v = u128::from(le_u64(b, 0)) | (u128::from(b[8]) << 64);
        let bits = |at: u32, n: u32| ((v >> at) & ((1u128 << n) - 1)) as u64;
        if bits(18, 1) == 0 {
            Record::Span {
                stored_mod: bits(0, 18) as u32,
                end: bits(19, 1) != 0,
                logical_mod: bits(20, 18) as u32,
                first_len: bits(38, 17) as u32 + 1,
                even: bits(55, 3) as u8,
                odd: bits(58, 3) as u8,
                predictor: bits(61, 6) as u8,
                shuffle: bits(68, 4) as u8,
            }
        } else {
            Record::RunBase { stored_end_mod: bits(0, 18) as u32, stored_base: bits(50, 22) }
        }
    }
    fn is_data(&self) -> bool {
        matches!(self, Record::Span { end: false, .. })
    }
}

/// Distance from `prev` to `next` on the 256 KiB ring (a full ring when equal).
fn ring_delta(next: u32, prev: u32) -> u32 {
    let d = next as i64 - prev as i64;
    (if d <= 0 { d + UBLOCK as i64 } else { d }) as u32
}

/// The parsed layout: logical file ranges and the span list.
#[derive(Clone, Debug)]
pub struct Layout {
    spans: Vec<Span>,
    /// Logical start of every file range (not continuations), ascending.
    file_starts: Vec<u64>,
    len: u64,
}

impl Layout {
    pub fn parse(b: &[u8]) -> io::Result<Self> {
        if b.len() < 16 {
            return Err(invalid("NAPS layout shorter than its header"));
        }
        let (w0, w1) = (le_u64(b, 0), le_u64(b, 8));
        if w0 >> 55 != 0 || w1 >> 48 != 0 {
            return Err(invalid("NAPS layout header has reserved bits set"));
        }
        let files = (w0 & 0xFF_FFFF) as usize + 1;
        let shuffles = ((w0 >> 28) & 0xF) as usize;
        let ublocks = ((w0 >> 32) & 0x7F_FFFF) as usize + 1;
        let outer = (w1 & 0xFF_FFFF) as usize;
        let records = ((w1 >> 24) & 0xFF_FFFF) as usize + 2;
        let fidx_off = 16 + outer * 8 + shuffles * 8;
        let u2c_off = fidx_off + files * 6;
        let u2c_count = (ublocks - 1 + 8) >> 3;
        let rec_off = (u2c_off + u2c_count * 10).next_multiple_of(8);
        if rec_off + records * 9 > b.len() {
            return Err(invalid(format!(
                "NAPS layout of {} bytes is shorter than its tables ({} bytes)",
                b.len(),
                rec_off + records * 9
            )));
        }
        let fidx: Vec<(u8, u64)> = (0..files)
            .map(|i| {
                let e = &b[fidx_off + i * 6..fidx_off + i * 6 + 6];
                (e[5], u64::from_le_bytes([e[0], e[1], e[2], e[3], e[4], 0, 0, 0]))
            })
            .collect();
        let first_record = |ub: usize| -> usize {
            let e = &b[u2c_off + (ub >> 3) * 10..u2c_off + (ub >> 3) * 10 + 10];
            let base = e[0] as usize | (e[1] as usize) << 8 | (e[2] as usize) << 16;
            if ub & 7 == 0 { base } else { base + e[2 + (ub & 7)] as usize }
        };
        let recs: Vec<Record> = (0..records).map(|i| Record::parse(&b[rec_off + i * 9..rec_off + i * 9 + 9])).collect();
        // The record whose span starts at logical offset `pos`.
        let find = |pos: u64| -> io::Result<usize> {
            let ub = (pos / UBLOCK) as usize;
            if ub >= ublocks {
                return Err(invalid(format!("NAPS logical offset {pos:#x} is past the ublock index")));
            }
            let (lo, hi) = (first_record(ub), if ub + 1 < ublocks { first_record(ub + 1) } else { recs.len() });
            (lo..hi.min(recs.len()))
                .find(|&k| match recs[k] {
                    Record::Span { end: false, logical_mod, .. } => pos == ub as u64 * UBLOCK + logical_mod as u64,
                    _ => false,
                })
                .ok_or_else(|| invalid(format!("no NAPS span starts at logical offset {pos:#x}")))
        };
        let mut spans: Vec<Span> = Vec::new();
        let mut file_starts = Vec::new();
        for w in fidx.windows(2) {
            let ((kind, start), (_, next)) = (w[0], w[1]);
            if next < start {
                return Err(invalid("NAPS file ranges go backwards"));
            }
            if kind & 0x40 != 0 {
                if spans.is_empty() {
                    return Err(invalid("NAPS continuation range without a preceding span"));
                }
                continue;
            }
            file_starts.push(start);
            let (mut pos, mut left) = (start, next - start);
            if left == 0 {
                continue;
            }
            let mut k = find(pos)?;
            while left != 0 {
                let rec = *recs.get(k).ok_or_else(|| invalid("NAPS file runs past the span records"))?;
                let Record::Span { stored_mod, end, logical_mod, first_len, even, odd, predictor, shuffle } = rec else {
                    k += 1;
                    continue;
                };
                if end {
                    return Err(invalid("NAPS file reaches the end marker early"));
                }
                let n1 = *recs.get(k + 1).ok_or_else(|| invalid("NAPS span without a following record"))?;
                let k2 = if matches!(n1, Record::RunBase { .. }) { k + 2 } else { k + 1 };
                let n2 = *recs.get(k2).ok_or_else(|| invalid("NAPS run base without a following span"))?;
                let next_stored = match n1 {
                    Record::RunBase { stored_end_mod, .. } => stored_end_mod,
                    Record::Span { stored_mod, .. } => stored_mod,
                };
                let Record::Span { logical_mod: next_logical, .. } = n2 else {
                    return Err(invalid("NAPS has two run bases in a row"));
                };
                let stored_len = ring_delta(next_stored, stored_mod);
                let len = ring_delta(next_logical, logical_mod);
                if len as u64 > UBLOCK || len as u64 > left {
                    return Err(invalid(format!("NAPS span at logical {pos:#x} has an invalid length {len:#x}")));
                }
                let stored = match k.checked_sub(1).map(|p| recs[p]) {
                    Some(Record::RunBase { stored_base, .. }) => stored_base << 18 | stored_mod as u64,
                    _ => match spans.last() {
                        Some(p) => p.stored + p.stored_len as u64,
                        None => return Err(invalid("the first NAPS span has no run base")),
                    },
                };
                spans.push(Span { stored, stored_len, first_len, logical: pos, len, even, odd, predictor, shuffle });
                pos += len as u64;
                left -= len as u64;
                k += 1;
            }
        }
        let data_records = recs.iter().filter(|r| r.is_data()).count();
        if spans.len() != data_records {
            return Err(invalid(format!("NAPS ranges use {} of {} spans", spans.len(), data_records)));
        }
        spans.sort_by_key(|s| s.logical);
        if spans.windows(2).any(|w| w[0].logical + w[0].len as u64 > w[1].logical) {
            return Err(invalid("NAPS spans overlap"));
        }
        let len = fidx.last().map_or(0, |f| f.1);
        Ok(Self { spans, file_starts, len })
    }

    /// Size of the logical image.
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Logical start offsets of the file ranges.
    pub fn file_starts(&self) -> &[u64] {
        &self.file_starts
    }
    /// End of the packed data the spans use.
    pub fn stored_end(&self) -> u64 {
        self.spans.iter().map(|s| s.stored + s.stored_len as u64).max().unwrap_or(0)
    }
}

/// The logical inner image of a PS5 package.
pub struct NapsImage {
    data: Source,
    layout: Layout,
    /// Most recently used span last.
    cache: Mutex<Vec<(usize, Arc<Vec<u8>>)>>,
    stats: Option<Stats>,
}

/// Decode counters printed every 5 s when `RPS5E_NAPS_STATS` is set.
struct Stats {
    decoded: std::sync::atomic::AtomicU64,
    repeated: std::sync::atomic::AtomicU64,
    hits: std::sync::atomic::AtomicU64,
    nanos: std::sync::atomic::AtomicU64,
    seen: Mutex<(Vec<u64>, std::time::Instant)>,
}

impl Stats {
    fn record(&self, idx: usize, spans: usize, took: std::time::Duration) {
        use std::sync::atomic::Ordering::Relaxed;
        self.decoded.fetch_add(1, Relaxed);
        self.nanos.fetch_add(took.as_nanos() as u64, Relaxed);
        let mut g = self.seen.lock().unwrap_or_else(|p| p.into_inner());
        if g.0.is_empty() {
            g.0 = vec![0; spans.div_ceil(64)];
        }
        let (w, b) = (idx / 64, 1u64 << (idx % 64));
        if g.0[w] & b != 0 {
            self.repeated.fetch_add(1, Relaxed);
        }
        g.0[w] |= b;
        if g.1.elapsed().as_secs() >= 5 {
            g.1 = std::time::Instant::now();
            eprintln!(
                "[naps] spans decoded {} (again {}), cache hits {}, decode time {} ms",
                self.decoded.load(Relaxed),
                self.repeated.load(Relaxed),
                self.hits.load(Relaxed),
                self.nanos.load(Relaxed) / 1_000_000
            );
        }
    }
}

impl NapsImage {
    /// `data` is `pfs_image.dat`, `layout` the bytes of `naps_pkg_layout.dat`.
    pub fn new(data: Source, layout: &[u8]) -> io::Result<Self> {
        let layout = Layout::parse(layout)?;
        if layout.stored_end() > data.len() {
            return Err(invalid(format!(
                "NAPS spans reach {:#x} but pfs_image.dat has {:#x} bytes",
                layout.stored_end(),
                data.len()
            )));
        }
        let stats = std::env::var_os("RPS5E_NAPS_STATS").map(|_| Stats {
            decoded: Default::default(),
            repeated: Default::default(),
            hits: Default::default(),
            nanos: Default::default(),
            seen: Mutex::new((Vec::new(), std::time::Instant::now())),
        });
        Ok(Self { data, layout, cache: Mutex::new(Vec::with_capacity(CACHE_SPANS)), stats })
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    fn decoded(&self, idx: usize) -> io::Result<Arc<Vec<u8>>> {
        {
            let mut c = self.cache.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(pos) = c.iter().position(|(i, _)| *i == idx) {
                let e = c.remove(pos);
                let data = e.1.clone();
                c.push(e);
                if let Some(st) = &self.stats {
                    st.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                return Ok(data);
            }
        }
        // Decoded without the lock: readers of other spans do not wait.
        let started = std::time::Instant::now();
        let s = self.layout.spans[idx];
        let mut out = vec![0u8; s.len as usize];
        if !s.is_zero() {
            if s.predictor != 0 || s.shuffle != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!(
                        "NAPS span at logical {:#x} uses predictor {} / shuffle {} (not supported)",
                        s.logical, s.predictor, s.shuffle
                    ),
                ));
            }
            let mut payload = vec![0u8; s.stored_len as usize];
            self.data.read_exact_at(&mut payload, s.stored)?;
            if s.is_stored() {
                out.copy_from_slice(&payload);
            } else {
                let first = if s.len > HALF { s.first_len as usize } else { 0 };
                kraken::decode_block(&payload, s.kraken_flags(), first, &mut out)
                    .map_err(|e| invalid(format!("NAPS span at logical {:#x}: {e}", s.logical)))?;
            }
        }
        if let Some(st) = &self.stats {
            st.record(idx, self.layout.spans.len(), started.elapsed());
        }
        let data = Arc::new(out);
        let mut c = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(pos) = c.iter().position(|(i, _)| *i == idx) {
            // Another reader decoded it meanwhile.
            c.remove(pos);
        }
        if c.len() >= CACHE_SPANS {
            c.remove(0);
        }
        c.push((idx, data.clone()));
        Ok(data)
    }
}

impl ReadAt for NapsImage {
    fn len(&self) -> u64 {
        self.layout.len
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        if offset.checked_add(buf.len() as u64).is_none_or(|e| e > self.layout.len) {
            return Err(crate::io::eof());
        }
        let spans = &self.layout.spans;
        let mut idx = spans.partition_point(|s| s.logical + s.len as u64 <= offset);
        let mut done = 0usize;
        while done < buf.len() {
            let pos = offset + done as u64;
            let Some(s) = spans.get(idx).filter(|s| s.logical <= pos) else {
                // A hole between spans (or after the last one) reads as zeros.
                let end = spans.get(idx).map_or(self.layout.len, |s| s.logical);
                let n = ((end - pos) as usize).min(buf.len() - done);
                buf[done..done + n].fill(0);
                done += n;
                continue;
            };
            let within = (pos - s.logical) as usize;
            let n = (s.len as usize - within).min(buf.len() - done);
            if s.is_stored() && !s.is_zero() {
                self.data.read_exact_at(&mut buf[done..done + n], s.stored + within as u64)?;
            } else {
                let block = self.decoded(idx)?;
                buf[done..done + n].copy_from_slice(&block[within..within + n]);
            }
            done += n;
            idx += 1;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// How [`build_naps`] packs one span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanForm {
    Stored,
    /// The zero-block encoding (the data must be zeros).
    Zero,
}

fn put_bits(v: &mut u128, at: u32, n: u32, val: u64) {
    assert!(n == 64 || val < 1u64 << n, "field overflow");
    *v |= u128::from(val) << at;
}

fn span_record(stored_mod: u64, end: bool, logical_mod: u64, first_len: u64, even: u64, odd: u64) -> [u8; 9] {
    let mut v = 0u128;
    put_bits(&mut v, 0, 18, stored_mod);
    put_bits(&mut v, 19, 1, end as u64);
    put_bits(&mut v, 20, 18, logical_mod);
    put_bits(&mut v, 38, 17, first_len - 1);
    put_bits(&mut v, 55, 3, even);
    put_bits(&mut v, 58, 3, odd);
    v.to_le_bytes()[..9].try_into().unwrap()
}

fn run_base(stored_end_mod: u64, stored_base: u64) -> [u8; 9] {
    let mut v = 0u128;
    put_bits(&mut v, 0, 18, stored_end_mod);
    put_bits(&mut v, 18, 1, 1);
    put_bits(&mut v, 50, 22, stored_base);
    v.to_le_bytes()[..9].try_into().unwrap()
}

/// Write `image` as (pfs_image.dat, naps_pkg_layout.dat) with stored or
/// zero spans only (tests and tools; no Kraken encoder here). `files` are the
/// logical starts of the file ranges (first 0). `pack(i)` chooses the form of
/// span `i`; `gap(i)` inserts that many unused packed bytes (and a run base)
/// before span `i`.
pub fn build_naps(
    image: &[u8],
    files: &[u64],
    pack: impl Fn(usize) -> SpanForm,
    gap: impl Fn(usize) -> u64,
) -> (Vec<u8>, Vec<u8>) {
    let len = image.len() as u64;
    let mut bounds: Vec<u64> = files.to_vec();
    bounds.push(len);
    let mut packed: Vec<u8> = Vec::new();
    let mut recs: Vec<[u8; 9]> = Vec::new();
    let mut ublock_first: Vec<u32> = Vec::new();
    let mut pos = 0u64;
    let mut i = 0usize;
    let mut f = 1usize;
    while pos < len {
        while f < bounds.len() && bounds[f] <= pos {
            f += 1;
        }
        let n = (UBLOCK - pos % UBLOCK).min(bounds[f] - pos);
        let data = &image[pos as usize..(pos + n) as usize];
        let g = gap(i);
        if i == 0 || g != 0 {
            packed.resize(packed.len() + g as usize, 0xEE);
            let end = packed.len() as u64 - g;
            recs.push(run_base(end % UBLOCK, packed.len() as u64 / UBLOCK));
        }
        if pos % UBLOCK == 0 {
            // The ublock index points at the span itself, or at the run base
            // right before it.
            let k = if i == 0 || g != 0 { recs.len() - 1 } else { recs.len() };
            ublock_first.push(k as u32);
        }
        let start = packed.len() as u64;
        let (stored, first, even, odd) = match pack(i) {
            SpanForm::Stored => (data.to_vec(), n.min(HALF as u64), 1, (n > HALF as u64) as u64),
            SpanForm::Zero => {
                assert!(data.iter().all(|&b| b == 0), "a zero span must hold zeros");
                let (bytes, odd) = if n > HALF as u64 { (16, 1) } else { (8, 0) };
                (vec![0u8; bytes], 8, 1, odd)
            }
        };
        packed.extend_from_slice(&stored);
        recs.push(span_record(start % UBLOCK, false, pos % UBLOCK, first, even, odd));
        pos += n;
        i += 1;
    }
    let end_rec = recs.len() as u32;
    recs.push(span_record(packed.len() as u64 % UBLOCK, true, len % UBLOCK, 1, 0, 0));
    let ublocks = len.div_ceil(UBLOCK) as usize;
    assert_eq!(ublock_first.len(), ublocks);
    let mut b = Vec::new();
    let w0 = (files.len() as u64) | 2 << 24 | ((ublocks as u64 - 1) << 32);
    let outer = (packed.len() as u64).div_ceil(0x1_0000);
    let w1 = outer | ((recs.len() as u64 - 2) << 24);
    b.extend_from_slice(&w0.to_le_bytes());
    b.extend_from_slice(&w1.to_le_bytes());
    b.resize(b.len() + outer as usize * 8, 0);
    for (k, &s) in bounds.iter().enumerate() {
        let kind: u8 = if k + 1 == bounds.len() { 0x40 } else { 0 };
        b.extend_from_slice(&s.to_le_bytes()[..5]);
        b.push(kind);
    }
    for g in 0..ublocks.div_ceil(8) {
        let base = ublock_first[g * 8];
        b.extend_from_slice(&base.to_le_bytes()[..3]);
        for j in 1..8 {
            let v = ublock_first.get(g * 8 + j).copied().unwrap_or(end_rec);
            b.push((v - base) as u8);
        }
    }
    b.resize(b.len().next_multiple_of(8), 0);
    for r in &recs {
        b.extend_from_slice(r);
    }
    b.resize(b.len().next_multiple_of(8) + 8, 0);
    (packed, b)
}

#[cfg(test)]
#[path = "naps_tests.rs"]
mod tests;
