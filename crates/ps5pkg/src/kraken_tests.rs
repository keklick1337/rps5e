// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Tests of the Kraken block decoder. A small test-only encoder writes
//! every block form the decoder reads (stored, bare entropy arrays, LZ
//! chunks with raw or delta literals, one or two sub-chunks with or without
//! a window restart, raw and Huffman arrays); its output is decoded and
//! compared with the input, then corrupted to check that bad input never
//! panics. A gated test decodes a few real package spans and compares
//! hashes recorded from an independent decoder.

use super::*;
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// Test inputs
// ---------------------------------------------------------------------------

/// xorshift64* generator for test inputs.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        (self.next() >> 33) % n
    }
}

/// Text from a small vocabulary: matches and literals.
fn text(n: usize, seed: u64) -> Vec<u8> {
    const WORDS: [&str; 24] = [
        "the", "fog", "town", "silent", "lake", "road", "hotel", "letter", "mirror", "room", "door", "radio", "static",
        "street", "night", "park", "bridge", "water", "light", "stairs", "window", "key", "map", "rust",
    ];
    let mut r = Rng(seed);
    let mut v = Vec::with_capacity(n + 16);
    while v.len() < n {
        match r.below(16) {
            0 => v.push(b'\n'),
            1 => v.extend_from_slice(format!("{} ", r.below(100_000)).as_bytes()),
            _ => {
                v.extend_from_slice(WORDS[r.below(WORDS.len() as u64) as usize].as_bytes());
                v.push(b' ');
            }
        }
    }
    v.truncate(n);
    v
}

/// 16-byte records with slowly changing fields: suits delta literals.
fn records(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed);
    let mut v = Vec::with_capacity(n + 16);
    let (mut a, mut b, mut c) = (1000u32, 50_000u32, 7u32);
    while v.len() < n {
        a = a.wrapping_add(1 + r.below(3) as u32);
        b = b.wrapping_add(r.below(40) as u32);
        c = c.wrapping_add(r.below(2) as u32);
        v.extend_from_slice(&a.to_le_bytes());
        v.extend_from_slice(&b.to_le_bytes());
        v.extend_from_slice(&c.to_le_bytes());
        v.extend_from_slice(&((a ^ b) & 0xFF).to_le_bytes());
    }
    v.truncate(n);
    v
}

/// Skewed byte distribution without repeats worth matching.
fn skewed(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed);
    (0..n)
        .map(|_| {
            let x = r.next();
            (x.trailing_zeros().min(12) as u8) * 17 + ((x >> 60) as u8 & 1)
        })
        .collect()
}

/// Uniform random bytes (all 256 symbols, incompressible).
fn random(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed);
    (0..n).map(|_| r.next() as u8).collect()
}

/// Long runs and very long matches (long length codes).
fn runs(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed);
    let mut v = Vec::with_capacity(n + 4096);
    while v.len() < n {
        let b = r.next() as u8;
        let len = 1 + r.below(3000) as usize;
        v.extend(std::iter::repeat_n(b, len));
        v.extend_from_slice(&random(1 + r.below(40) as usize, r.next()));
    }
    v.truncate(n);
    v
}

// ---------------------------------------------------------------------------
// Test encoder
// ---------------------------------------------------------------------------

/// MSB-first bit writer.
#[derive(Default)]
struct MsbWriter {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl MsbWriter {
    fn put(&mut self, v: u32, n: u32) {
        if n == 0 {
            return;
        }
        self.acc = (self.acc << n) | (v as u64 & ((1u64 << n) - 1));
        self.n += n;
        while self.n >= 8 {
            self.n -= 8;
            self.out.push((self.acc >> self.n) as u8);
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push((self.acc << (8 - self.n)) as u8);
        }
        self.out
    }
}

/// LSB-first bit writer (Huffman streams).
#[derive(Default)]
struct LsbWriter {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl LsbWriter {
    fn put(&mut self, v: u32, n: u32) {
        self.acc |= (v as u64) << self.n;
        self.n += n;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

fn bit_len(x: u32) -> u32 {
    32 - x.leading_zeros()
}

/// How an array is written.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Ent {
    Raw,
    /// Huffman; `halves` selects the two-half layout, `gamma` the gamma
    /// code-length form (else the explicit list, used up to 255 symbols).
    Huff { halves: bool, gamma: bool },
}

/// Huffman code lengths limited to 11 bits (frequencies are flattened
/// until the tree fits).
fn code_lengths(counts: &[u64; 256]) -> [u32; 256] {
    let mut w = *counts;
    loop {
        let mut heap = BinaryHeap::new();
        let mut parent = vec![usize::MAX; 512];
        let mut next = 256;
        for (s, &c) in w.iter().enumerate() {
            if c > 0 {
                heap.push(std::cmp::Reverse((c, s)));
            }
        }
        while heap.len() > 1 {
            let std::cmp::Reverse((c1, a)) = heap.pop().unwrap();
            let std::cmp::Reverse((c2, b)) = heap.pop().unwrap();
            parent[a] = next;
            parent[b] = next;
            heap.push(std::cmp::Reverse((c1 + c2, next)));
            next += 1;
        }
        let mut lens = [0u32; 256];
        let mut max = 0;
        for s in 0..256 {
            if w[s] > 0 {
                let mut d = 0;
                let mut n = s;
                while parent[n] != usize::MAX {
                    n = parent[n];
                    d += 1;
                }
                lens[s] = d;
                max = max.max(d);
            }
        }
        if max <= 11 {
            return lens;
        }
        for c in w.iter_mut().filter(|c| **c > 0) {
            *c = c.div_ceil(2);
        }
    }
}

/// Body of a Huffman array (type 2 or 4), or `None` when it cannot be
/// written (a stream longer than 64 KiB).
fn huffman_body(data: &[u8], halves: bool, gamma: bool) -> Option<Vec<u8>> {
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let used: Vec<usize> = (0..256).filter(|&s| counts[s] > 0).collect();
    let mut w = MsbWriter::default();
    w.put(0, 1); // gamma / explicit transmission
    if used.len() == 1 {
        w.put(0, 1);
        w.put(1, 8);
        w.put(used[0] as u32, 8);
        return Some(w.finish());
    }
    let lens = code_lengths(&counts);
    if gamma || used.len() > 255 {
        w.put(1, 1);
        let k = 0u32;
        w.put(k, 2);
        let mut avg: i32 = 32;
        let mut i = 0;
        w.put((used[0] == 0) as u32, 1);
        let gamma_code = |w: &mut MsbWriter, x: u32| {
            let c = bit_len(x) - 2;
            w.put(0, c);
            w.put(x, c + 2);
        };
        let mut sym = 0usize;
        while i < used.len() {
            if used[i] != sym {
                gamma_code(&mut w, (used[i] - sym + 1) as u32);
                sym = used[i];
            }
            let mut run = 1;
            while i + run < used.len() && used[i + run] == sym + run {
                run += 1;
            }
            gamma_code(&mut w, run as u32 + 1);
            for &len in &lens[sym..sym + run] {
                let len = len as i32;
                let delta = len - ((avg + 2) >> 2);
                let v = if delta >= 0 { 2 * delta } else { -2 * delta - 1 } as u32;
                let c = v >> k;
                w.put(0, c);
                w.put(1, 1);
                w.put(v, k);
                avg = len + ((3 * avg + 2) >> 2);
            }
            sym += run;
            i += run;
        }
        if sym != 256 {
            gamma_code(&mut w, (256 - sym + 1) as u32);
        }
    } else {
        w.put(0, 1);
        w.put(used.len() as u32, 8);
        let width = used.iter().map(|&s| bit_len(lens[s] - 1)).max().unwrap();
        w.put(width, 3);
        for &s in &used {
            w.put(s as u32, 8);
            w.put(lens[s] - 1, width);
        }
    }
    let mut body = w.finish();

    // Canonical codes in the decoder's order: by length, then by symbol.
    let mut code = [(0u32, 0u32); 256];
    let mut at = 0u32;
    for len in 1..=11 {
        for &s in used.iter().filter(|&&s| lens[s] == len) {
            let c = at >> (11 - len);
            let rev = c.reverse_bits() >> (32 - len);
            code[s] = (rev, len);
            at += 1 << (11 - len);
        }
    }
    assert_eq!(at, 2048);

    let part = |d: &[u8]| -> Option<Vec<u8>> {
        let mut s = [LsbWriter::default(), LsbWriter::default(), LsbWriter::default()];
        for (i, &b) in d.iter().enumerate() {
            let (c, l) = code[b as usize];
            s[i % 3].put(c, l);
        }
        let [s0, s1, s2] = s.map(|w| w.finish());
        if s0.len() > 0xFFFF {
            return None;
        }
        let mut p = (s0.len() as u16).to_le_bytes().to_vec();
        p.extend_from_slice(&s0);
        p.extend_from_slice(&s2);
        p.extend(s1.iter().rev());
        Some(p)
    };
    if halves {
        let (a, b) = data.split_at(data.len().div_ceil(2));
        let pa = part(a)?;
        body.extend_from_slice(&(pa.len() as u32).to_le_bytes()[..3]);
        body.extend_from_slice(&pa);
        body.extend_from_slice(&part(b)?);
    } else {
        body.extend_from_slice(&part(data)?);
    }
    Some(body)
}

/// Appends an array; `short_raw` allows the 2-byte raw header (whose first
/// byte has the top bit set, which some positions in a chunk must avoid).
fn put_array(out: &mut Vec<u8>, data: &[u8], ent: Ent, short_raw: bool) {
    if let Ent::Huff { halves, gamma } = ent {
        if data.len() >= 16 {
            if let Some(body) = huffman_body(data, halves, gamma) {
                if body.len() < data.len() {
                    let d = data.len() as u32 - 1;
                    let kind = if halves { 4 } else { 2 };
                    out.push((kind << 4 | d >> 14) as u8);
                    out.extend_from_slice(&((d & 0x3FFF) << 18 | body.len() as u32).to_be_bytes());
                    out.extend_from_slice(&body);
                    return;
                }
            }
        }
    }
    let n = data.len();
    if short_raw && n < 0x1000 {
        out.extend_from_slice(&[0x80 | (n >> 8) as u8, n as u8]);
    } else {
        out.extend_from_slice(&[(n >> 16) as u8, (n >> 8) as u8, n as u8]);
    }
    out.extend_from_slice(data);
}

/// Form of a sub-chunk.
#[derive(Clone, Copy, Debug)]
enum Form {
    Stored,
    Bare(Ent),
    Lz { sub: bool, ent: Ent, excess: bool },
}

fn hash4(d: &[u8], p: usize) -> usize {
    (u32::from_le_bytes(d[p..p + 4].try_into().unwrap()).wrapping_mul(0x9E37_79B1) >> 16) as usize
}

fn match_len(d: &[u8], a: usize, b: usize, end: usize) -> usize {
    let mut n = 0;
    while b + n < end && d[a + n] == d[b + n] {
        n += 1;
    }
    n
}

/// Packed offset symbol and extra bits of a distance (>= 8).
fn distance_code(d: u32) -> (u8, u32, u32) {
    let x = d + 248;
    let y = x >> 4;
    let n = bit_len(y) - 1;
    assert!((4..=18).contains(&n));
    ((((n - 4) << 4) | (x & 15)) as u8, y - (1 << n), n)
}

fn put_long_length(w: &mut MsbWriter, v: u32) {
    let x = v + 64;
    let nb = bit_len(x);
    w.put(0, nb - 7);
    w.put(x, nb);
}

/// Greedy LZ encoding of `d[start..end]`; matches reach back to the
/// chunk start with a seed (fresh window), else to the block start. Returns `None` when a limit of the format is exceeded.
fn encode_lz(d: &[u8], start: usize, end: usize, seed: bool, sub: bool, ent: Ent, excess: bool) -> Option<Vec<u8>> {
    let window = if seed { start } else { 0 };
    let mut out = Vec::new();
    let mut pos = start;
    if seed {
        out.extend_from_slice(&d[start..start + 8]);
        pos += 8;
    }
    // Parse: (literal start, literal count, distance, match length).
    let mut table = vec![usize::MAX; 1 << 16];
    for p in window..pos.saturating_sub(3) {
        table[hash4(d, p)] = p;
    }
    let mut recent = [8usize; 3];
    let mut cmds = Vec::new();
    let mut anchor = pos;
    while pos + 4 <= end {
        let mut best = (0usize, 0usize);
        for &r in &recent {
            if pos - window >= r {
                let l = match_len(d, pos - r, pos, end);
                if l >= 3 && l > best.1 {
                    best = (r, l);
                }
            }
        }
        let h = hash4(d, pos);
        let cand = table[h];
        table[h] = pos;
        if best.1 < 8 && cand != usize::MAX && cand >= window && pos - cand >= 8 {
            let l = match_len(d, cand, pos, end);
            if l >= 4 && l > best.1 + 1 {
                best = (pos - cand, l);
            }
        }
        if best.1 == 0 {
            pos += 1;
            continue;
        }
        cmds.push((anchor, pos - anchor, best.0, best.1));
        if let Some(i) = recent.iter().position(|&r| r == best.0) {
            recent[..=i].rotate_right(1);
        } else {
            recent.rotate_right(1);
            recent[0] = best.0;
        }
        for p in pos + 1..(pos + best.1).min(end.saturating_sub(3)) {
            table[hash4(d, p)] = p;
        }
        pos += best.1;
        anchor = pos;
    }

    // Streams.
    let (mut lit, mut cmd, mut offs, mut lens, mut long) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut recent = [8usize; 3];
    let mut last = 8usize;
    let put_lits = |lit: &mut Vec<u8>, from: usize, n: usize, last: usize| {
        for p in from..from + n {
            lit.push(if sub { d[p].wrapping_sub(d[p - last]) } else { d[p] });
        }
    };
    let mut put_len = |lens: &mut Vec<u8>, v: usize| {
        if v >= 255 {
            lens.push(255);
            long.push((v - 255) as u32);
        } else {
            lens.push(v as u8);
        }
    };
    for &(from, n, dist, ml) in &cmds {
        put_lits(&mut lit, from, n, last);
        let idx = match recent.iter().position(|&r| r == dist) {
            Some(i) => {
                recent[..=i].rotate_right(1);
                i
            }
            None => {
                recent.rotate_right(1);
                recent[0] = dist;
                offs.push(dist as u32);
                3
            }
        };
        last = dist;
        let lit_field = if n < 3 {
            n
        } else {
            put_len(&mut lens, n - 3);
            3
        };
        let ml_field = if ml - 2 < 15 {
            ml - 2
        } else {
            put_len(&mut lens, ml - 17);
            15
        };
        cmd.push((idx << 6 | ml_field << 2 | lit_field) as u8);
    }
    put_lits(&mut lit, anchor, end - anchor, last);
    if lens.len() > (end - start) >> 2 || long.len() > 512 {
        return None;
    }

    // Two-ended bit stream with the offset bits (and the long lengths).
    let (mut fa, mut fb) = (MsbWriter::default(), MsbWriter::default());
    let (mut ea, mut eb) = (MsbWriter::default(), MsbWriter::default());
    if !excess {
        let x = long.len() as u32 + 1;
        fb.put(0, bit_len(x) - 1);
        fb.put(x, bit_len(x));
    }
    let mut packed = Vec::new();
    for (i, &dist) in offs.iter().enumerate() {
        let (v, extra, n) = distance_code(dist);
        packed.push(v);
        if i & 1 == 0 { fa.put(extra, n) } else { fb.put(extra, n) }
    }
    for (i, &v) in long.iter().enumerate() {
        let w = match (excess, i & 1 == 0) {
            (false, true) => &mut fa,
            (false, false) => &mut fb,
            (true, true) => &mut ea,
            (true, false) => &mut eb,
        };
        put_long_length(w, v);
    }
    let mut excess_area = ea.finish();
    excess_area.extend(eb.finish().iter().rev());

    if excess {
        let n = excess_area.len();
        if n <= 31 {
            out.push(0x80 | n as u8);
        } else {
            let c = 32 + n % 32;
            out.push(0x80 | c as u8);
            out.push(((n - c) / 32) as u8);
        }
    }
    // The literal and offset arrays must not start with the top bit set
    // (that would read as the excess flag / offset scaling byte).
    put_array(&mut out, &lit, ent, false);
    put_array(&mut out, &cmd, ent, true);
    put_array(&mut out, &packed, ent, false);
    put_array(&mut out, &lens, ent, true);
    out.extend_from_slice(&fa.finish());
    out.extend(fb.finish().iter().rev());
    out.extend_from_slice(&excess_area);
    Some(out)
}

/// Encodes one sub-chunk; returns its bytes and its flag nibble.
fn encode_sub(d: &[u8], start: usize, end: usize, restart: bool, form: Form) -> (Vec<u8>, u32) {
    let len = end - start;
    let res = match form {
        Form::Stored => None,
        Form::Bare(ent) => {
            let mut v = Vec::new();
            put_array(&mut v, &d[start..end], ent, true);
            Some((v, 0))
        }
        Form::Lz { sub, ent, excess } => {
            encode_lz(d, start, end, restart, sub, ent, excess).map(|v| (v, 2 | sub as u32))
        }
    };
    match res {
        Some((v, nibble)) if v.len() != len => (v, nibble),
        _ => (d[start..end].to_vec(), 2),
    }
}

/// Encodes a block; returns payload, flags and first sub-chunk size.
fn encode_block(d: &[u8], f0: Form, f1: Form, restart: bool) -> (Vec<u8>, u32, usize) {
    let len0 = d.len().min(CHUNK_SIZE);
    let (mut payload, mut flags) = encode_sub(d, 0, len0, true, f0);
    let first = payload.len();
    if d.len() > len0 {
        let (c1, n1) = encode_sub(d, len0, d.len(), restart, f1);
        payload.extend_from_slice(&c1);
        flags |= n1 << 4 | if restart { FLAG_CHUNK1_RESTART } else { 0 };
    }
    (payload, flags, first)
}

fn roundtrip(name: &str, d: &[u8], f0: Form, f1: Form, restart: bool) -> (Vec<u8>, u32, usize) {
    let (payload, flags, first) = encode_block(d, f0, f1, restart);
    let mut out = vec![0u8; d.len()];
    decode_block(&payload, flags, first, &mut out).unwrap_or_else(|e| panic!("{name} {f0:?}/{f1:?}: {e}"));
    assert!(out == d, "{name} {f0:?}/{f1:?} restart {restart}: output differs");
    (payload, flags, first)
}

const RAW: Ent = Ent::Raw;
const HUFF: Ent = Ent::Huff { halves: false, gamma: false };
const HUFF_HALVES: Ent = Ent::Huff { halves: true, gamma: false };
const HUFF_GAMMA: Ent = Ent::Huff { halves: false, gamma: true };

fn lz(sub: bool, ent: Ent) -> Form {
    Form::Lz { sub, ent, excess: false }
}

// ---------------------------------------------------------------------------
// Round trips
// ---------------------------------------------------------------------------

#[test]
fn kraken_stored_and_empty() {
    let d = text(1000, 11);
    let mut out = vec![0u8; d.len()];
    decode_block(&d, FLAG_CHUNK0_LZ, 0, &mut out).unwrap();
    assert_eq!(out, d);

    let d = records(CHUNK_SIZE + 5000, 12);
    let mut out = vec![0u8; d.len()];
    decode_block(&d, FLAG_CHUNK0_LZ | FLAG_CHUNK1_LZ, CHUNK_SIZE, &mut out).unwrap();
    assert_eq!(out, d);

    decode_block(&[], 0, 0, &mut []).unwrap();
    assert!(decode_block(&[1], 0, 0, &mut []).is_err());
    assert!(decode_block(&d, 0x22, 0, &mut out).is_err());
    assert!(decode_block(&d, 0x22, d.len() + 1, &mut out).is_err());
    let mut big = vec![0u8; BLOCK_MAX + 1];
    assert!(decode_block(&d, 0x22, CHUNK_SIZE, &mut big).is_err());
}

#[test]
fn kraken_bare_entropy_chunks() {
    for (name, d) in [("skewed", skewed(20_000, 1)), ("text", text(3000, 2)), ("random", random(5000, 3)), ("one", vec![7u8; 900])] {
        for ent in [RAW, HUFF, HUFF_HALVES, HUFF_GAMMA] {
            let (payload, flags, _) = roundtrip(name, &d, Form::Bare(ent), Form::Stored, false);
            assert_eq!(flags & FLAG_CHUNK0_LZ, 0, "{name}");
            if ent != RAW && name != "random" {
                let kind = (payload[0] >> 4) & 7;
                assert!(kind == 2 || kind == 4, "{name} {ent:?}: array type {kind}");
            }
        }
    }
    // A whole sub-chunk (18-bit sizes) and all 256 symbols (gamma lengths).
    roundtrip("skewed-chunk", &skewed(CHUNK_SIZE, 4), Form::Bare(HUFF_HALVES), Form::Stored, false);
    let mut d = skewed(60_000, 5);
    d.extend((0..=255u8).cycle().take(512));
    roundtrip("all-symbols", &d, Form::Bare(HUFF), Form::Stored, false);
}

#[test]
fn kraken_lz_chunks() {
    let inputs = [
        ("text", text(60_000, 21)),
        ("records", records(40_000, 22)),
        ("runs", runs(90_000, 23)),
        ("small", text(200, 24)),
        ("chunk", text(CHUNK_SIZE, 25)),
    ];
    for (name, d) in &inputs {
        for sub in [false, true] {
            for ent in [RAW, HUFF, HUFF_HALVES, HUFF_GAMMA] {
                for excess in [false, true] {
                    let (payload, flags, _) = roundtrip(name, d, Form::Lz { sub, ent, excess }, Form::Stored, false);
                    assert_ne!(payload.len(), d.len(), "{name}: stored instead of LZ");
                    assert_eq!(flags & 3, FLAG_CHUNK0_LZ | sub as u32, "{name}");
                    assert_eq!(payload[8] & 0x80 != 0, excess, "{name}: excess flag");
                }
            }
        }
    }
}

#[test]
fn kraken_two_sub_chunks() {
    let mut mixed = text(150_000, 31);
    mixed.extend_from_slice(&records(70_000, 32));
    let inputs = [
        ("text", text(BLOCK_MAX, 33)),
        ("mixed", mixed),
        ("runs", runs(CHUNK_SIZE + 9, 34)),
        ("repeat", {
            // The second sub-chunk repeats the first: long back-references.
            let mut v = text(CHUNK_SIZE, 35);
            v.extend_from_within(..100_000);
            v
        }),
    ];
    let forms = [
        Form::Stored,
        Form::Bare(RAW),
        Form::Bare(HUFF),
        lz(false, RAW),
        lz(true, HUFF),
        Form::Lz { sub: false, ent: HUFF_HALVES, excess: true },
    ];
    for (name, d) in &inputs {
        for &f0 in &forms {
            for &f1 in &forms {
                for restart in [false, true] {
                    let (_, flags, first) = roundtrip(name, d, f0, f1, restart);
                    assert!(first < d.len());
                    assert_eq!(flags & FLAG_CHUNK1_RESTART != 0, restart);
                }
            }
        }
    }
    // Without a restart chunk 1 copies chunk 0 with distances past its
    // own start; the restart version must not.
    let d = &inputs[3].1;
    let (a, _, fa) = encode_block(d, lz(false, RAW), lz(false, RAW), false);
    let (b, _, fb) = encode_block(d, lz(false, RAW), lz(false, RAW), true);
    assert!(a.len() - fa < (b.len() - fb) / 4, "chunk 1 does not reference chunk 0");
}

// ---------------------------------------------------------------------------
// Corrupt input
// ---------------------------------------------------------------------------

/// Corrupt inputs must give errors (or some output), never a panic.
/// `PS5PKG_KRAKEN_FUZZ=<rounds>` runs a longer fuzz with a random seed.
#[test]
fn kraken_corruption_fuzz() {
    let long: Option<u64> = std::env::var("PS5PKG_KRAKEN_FUZZ").ok().and_then(|v| v.parse().ok());
    let rounds = long.unwrap_or(1000);
    let seed = if long.is_some() {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64 | 1
    } else {
        0x1234_5678
    };
    let mut mixed = text(140_000, 41);
    mixed.extend_from_slice(&records(40_000, 42));
    let inputs: [(Vec<u8>, Form, Form, bool); 7] = [
        (text(30_000, 43), lz(false, RAW), Form::Stored, false),
        (records(30_000, 44), lz(true, HUFF), Form::Stored, false),
        (skewed(9000, 45), Form::Bare(HUFF_HALVES), Form::Stored, false),
        (skewed(9000, 46), Form::Bare(HUFF_GAMMA), Form::Stored, false),
        (runs(40_000, 47), Form::Lz { sub: false, ent: HUFF, excess: true }, Form::Stored, false),
        (mixed.clone(), lz(false, HUFF_HALVES), lz(true, RAW), false),
        (mixed, lz(true, RAW), lz(false, HUFF), true),
    ];
    let mut r = Rng(seed);
    for (data, f0, f1, restart) in &inputs {
        let (payload, flags, first) = encode_block(data, *f0, *f1, *restart);
        let ulen = data.len();
        let mut out = vec![0u8; ulen];
        decode_block(&payload, flags, first, &mut out).unwrap();
        for cut in (0..payload.len()).step_by(payload.len() / 200 + 1) {
            let _ = decode_block(&payload[..cut], flags, first.min(cut), &mut out);
        }
        for _ in 0..rounds {
            let mut p = payload.clone();
            for _ in 0..1 + r.below(4) {
                let i = r.below(p.len() as u64) as usize;
                if r.below(2) == 0 {
                    p[i] ^= 1 << r.below(8);
                } else {
                    p[i] = r.next() as u8;
                }
            }
            let fl = if r.below(8) == 0 { r.next() as u32 & 0x73 } else { flags };
            let _ = decode_block(&p, fl, first, &mut out);
        }
        for d in [1usize, 7, 4096] {
            let mut short = vec![0u8; ulen.saturating_sub(d).max(1)];
            let _ = decode_block(&payload, flags, first, &mut short);
            let mut longer = vec![0u8; (ulen + d).min(BLOCK_MAX)];
            let _ = decode_block(&payload, flags, first, &mut longer);
        }
    }
    let mut out = vec![0u8; BLOCK_MAX];
    for _ in 0..rounds * 2 {
        let n = 1 + r.below(600) as usize;
        let p: Vec<u8> = (0..n).map(|_| r.next() as u8).collect();
        let len = 1 + r.below(BLOCK_MAX as u64) as usize;
        let _ = decode_block(&p, r.next() as u32 & 0x73, r.below(n as u64 + 1) as usize, &mut out[..len]);
    }
}

// ---------------------------------------------------------------------------
// Real package spans
// ---------------------------------------------------------------------------

/// Spans of the Silent Hill 2 (PPSA08709 v1.007) package: file offset,
/// compressed size, output size, first sub-chunk size, flags, SHA-256 of
/// the output (from an independent decoder). One per flag mix, plus the
/// span at the end of the image (inner file system metadata).
const REAL_SPANS: [(u64, usize, usize, usize, u32, &str); 10] = [
    (0x2e2418eef, 151377, 262144, 76918, 0x00, "3d27ded77922311e93ec7bcfa1cf60fa10ffeb528b9492e05343d3310e0cf7b4"),
    (0x41cff472, 243343, 262144, 112271, 0x02, "5715fc1e8ea4fc5b44636d2e023ef9ee7e7b80f7b99c027499c75ea391d7f5f5"),
    (0x15d8e80a, 237261, 262144, 106189, 0x03, "6ca23d9b7b088d6a8e476b81df84a3f6a2e4722d83791ef3bd2134d1eeff87d6"),
    (0x109747f98, 230929, 262144, 131072, 0x20, "c057d96d6ddb9cde4c058fccb17a07a87fa36a491589d9cfb6d701c3da1e8947"),
    (0x2b5a236b6, 52307, 262144, 23766, 0x22, "5109b29d72a1d2b1b620ccdb4df40c99455b54ab798e054895833f6c447620bc"),
    (0x705dc3d64, 202432, 262144, 112672, 0x23, "28ce2f3de81c9a382e5e58202a932673ba609b1932035ec797e7919d95459e97"),
    (0x3b61b345, 183685, 262144, 131072, 0x30, "77799c4743f9a1e77ee928584a1b9ce02930572a8780fbbabfe465dd71e3cdad"),
    (0x724e506ef, 31405, 262144, 12711, 0x32, "a34e8a59b29d3b84e3ec81420371cfd123a8f3aaf9b44262e8da83f5542dad7e"),
    (0x522f3257d, 112614, 262144, 76597, 0x33, "48e7a64dc774b5392bd791b4bf0b572e12f1a2eede25eb34424cfdbf7425d0b6"),
    (0xb49b8126b, 399, 262144, 84, 0x22, "9e32e81909b766b623d576942c1f0acf028c6eef109d6825a0108e166c7c9c68"),
];

pub(crate) const REAL_PKG: &str = "EP0101-PPSA08709_00-MAINGAME00000000.pkg";

/// The package file under `GAMES_DIR` (up to three folder levels deep).
pub(crate) fn find_file(dir: &std::path::Path, name: &str, depth: u32) -> Option<std::path::PathBuf> {
    let p = dir.join(name);
    if p.is_file() {
        return Some(p);
    }
    if depth == 0 {
        return None;
    }
    let mut subdirs: Vec<_> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    subdirs.sort();
    subdirs.into_iter().find_map(|d| find_file(&d, name, depth - 1))
}

#[test]
fn kraken_real_package_spans() {
    use sha2::{Digest, Sha256};
    use std::io::{Read, Seek, SeekFrom};
    let Some(root) = std::env::var_os("GAMES_DIR") else { return };
    let Some(path) = find_file(std::path::Path::new(&root), REAL_PKG, 3) else { return };
    let mut file = std::fs::File::open(path).unwrap();
    for (off, clen, ulen, first, flags, sha) in REAL_SPANS {
        let mut src = vec![0u8; clen];
        file.seek(SeekFrom::Start(off)).unwrap();
        file.read_exact(&mut src).unwrap();
        let mut out = vec![0u8; ulen];
        decode_block(&src, flags, first, &mut out).unwrap_or_else(|e| panic!("span at {off:#x}: {e}"));
        let got: String = Sha256::digest(&out).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(got, sha, "span at {off:#x}");
    }
}

/// Byte-exact check of many spans (local validation). Needs
/// `PS5PKG_SH2_PKG` (the package), `PS5PKG_KRAKEN_SPANS` (records of u64
/// file offset, u32 compressed size, u32 output size, u32 flags, u32 first
/// chunk size) and optionally `PS5PKG_KRAKEN_REF` (per span: i32 status,
/// SHA-256 of the expected output).
#[test]
#[ignore]
fn kraken_real_spans() {
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::io::{Read, Seek, SeekFrom};
    let (Ok(pkg), Ok(list)) = (std::env::var("PS5PKG_SH2_PKG"), std::env::var("PS5PKG_KRAKEN_SPANS")) else {
        eprintln!("PS5PKG_SH2_PKG / PS5PKG_KRAKEN_SPANS not set: skipped");
        return;
    };
    let mut file = std::fs::File::open(pkg).unwrap();
    let list = std::fs::read(list).unwrap();
    let reference = std::env::var("PS5PKG_KRAKEN_REF").ok().map(|p| std::fs::read(p).unwrap());
    let mut stats: BTreeMap<u32, (usize, usize, usize, usize)> = BTreeMap::new();
    let mut out = vec![0u8; BLOCK_MAX];
    let mut src = Vec::new();
    let (mut total, mut time) = (0usize, std::time::Duration::ZERO);
    let mut failures = 0;
    for (i, e) in list.as_chunks::<24>().0.iter().enumerate() {
        let off = u64::from_le_bytes(e[0..8].try_into().unwrap());
        let w = |j: usize| u32::from_le_bytes(e[j..j + 4].try_into().unwrap()) as usize;
        let (clen, ulen, flags, first) = (w(8), w(12), w(16) as u32, w(20));
        src.resize(clen, 0);
        file.seek(SeekFrom::Start(off)).unwrap();
        file.read_exact(&mut src).unwrap();
        let t = std::time::Instant::now();
        let res = decode_block(&src, flags, first, &mut out[..ulen]);
        time += t.elapsed();
        total += ulen;
        let s = stats.entry(flags).or_default();
        s.0 += 1;
        match res {
            Err(err) => {
                s.1 += 1;
                if failures < 20 {
                    eprintln!("span {i} off {off:#x} flags {flags:#x}: {err}");
                }
                failures += 1;
            }
            Ok(()) => {
                if let Some(r) = &reference {
                    let rec = &r[i * 36..i * 36 + 36];
                    if rec[..4] == [0, 0, 0, 0] {
                        if Sha256::digest(&out[..ulen])[..] == rec[4..] {
                            s.2 += 1;
                        } else {
                            s.3 += 1;
                            if failures < 20 {
                                eprintln!("span {i} off {off:#x} flags {flags:#x}: differs from the reference");
                            }
                            failures += 1;
                        }
                    }
                }
            }
        }
    }
    for (flags, (n, err, same, diff)) in &stats {
        eprintln!("flags {flags:#04x}: {n} spans, {err} errors, {same} byte-exact, {diff} different");
    }
    eprintln!(
        "{} MiB in {:.3} s: {:.0} MB/s",
        total >> 20,
        time.as_secs_f64(),
        total as f64 / time.as_secs_f64() / 1e6
    );
    assert_eq!(failures, 0);
}
