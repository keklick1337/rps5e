// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! PFSC: the block-compressed container used for `pfs_image.dat` (the
//! inner image of a package) and for individually compressed PFS files.
//!
//! ```text
//! 0x00 u32 magic "PFSC"      0x04 u32 0 (non-zero: the PS5 Kraken format)
//! 0x08 u32 (6)               0x0C u32 block size (usually 0x10000)
//! 0x10 u64 block size again  0x18 u64 offset of the block offset table
//! 0x20 u64 data start        0x28 u64 decompressed length
//! ```
//! The table holds `blocks + 1` little-endian u64 offsets; block `i` is
//! stored in `[off[i], off[i+1])`. A stored size equal to the block size
//! means a raw block, a smaller one a zlib (or raw deflate) stream, a
//! larger one or zero a block of zeros.
//!
//! Files of PS5 game images use format version 2 instead ([`PfscKraken`],
//! opened through [`open`]):
//! ```text
//! 0x00 u32 magic "PFSC"   0x04 u32 version (2)   0x08 u32 codec (2: Kraken)
//! 0x0C u32 block size (0x20000)                  0x10 u32 (32)
//! 0x14 u32 block count    0x18 u64 decompressed length
//! 0x20 u64 end of data    0x30 u64 table offset  0x40 u64 data offset
//! ```
//! The table holds `blocks + 1` u64 words: bits 0..47 the stored offset,
//! bits 48..63 flags. 0x8000 starts a group; a group start with 0x4000 set
//! (0xC000) is Kraken-coded and covers this block and, when the next word's
//! flags are exactly 0x4000, the next one too (256 KiB, two sub-chunks).
//! Other group starts are stored blocks. The last word is 0x8000 << 48 | end.

use crate::io::{invalid, le_u32, le_u64, read_vec, ReadAt, Source};
use std::io::{self, Read};
use std::sync::{Arc, Mutex};

pub const PFSC_MAGIC: u32 = 0x4353_4650; // "PFSC"
pub const PFSC_HEADER_SIZE: usize = 0x30;
/// Decompressed blocks kept per stream (random small reads hit the cache).
const CACHE_BLOCKS: usize = 8;

/// True when `head` starts with the PFSC magic.
pub fn is_pfsc(head: &[u8]) -> bool {
    le_u32(head, 0) == PFSC_MAGIC
}

pub struct Pfsc {
    inner: Source,
    block_size: u64,
    offsets: Vec<u64>,
    len: u64,
    cache: Mutex<Vec<(u64, Arc<Vec<u8>>)>>,
}

impl Pfsc {
    pub fn new(inner: Source) -> io::Result<Self> {
        if inner.len() < PFSC_HEADER_SIZE as u64 {
            return Err(invalid("PFSC stream shorter than its header"));
        }
        let h = read_vec(&*inner, 0, PFSC_HEADER_SIZE)?;
        if !is_pfsc(&h) {
            return Err(invalid("missing PFSC magic"));
        }
        if le_u32(&h, 4) != 0 {
            // The PS5 block format (versions 2 and 3, Kraken-coded blocks)
            // keeps a version here; this stream format stores 0.
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("Kraken-compressed PFSC (format version {}) is not supported", le_u32(&h, 4)),
            ));
        }
        let block_size = le_u32(&h, 0x0C) as u64;
        let block_size2 = le_u64(&h, 0x10);
        let table = le_u64(&h, 0x18);
        let len = le_u64(&h, 0x28);
        if block_size == 0 || block_size != block_size2 || block_size > 16 << 20 {
            return Err(invalid(format!("PFSC block size {block_size:#x}/{block_size2:#x} is invalid")));
        }
        let blocks = len.div_ceil(block_size);
        let table_len = (blocks + 1).checked_mul(8).ok_or_else(|| invalid("PFSC too large"))?;
        if table.checked_add(table_len).is_none_or(|e| e > inner.len()) {
            return Err(invalid("PFSC block table lies outside the stream"));
        }
        let raw = read_vec(&*inner, table, table_len as usize)?;
        let offsets: Vec<u64> = raw.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        if offsets.windows(2).any(|w| w[1] < w[0]) || offsets.last().is_some_and(|&e| e > inner.len()) {
            return Err(invalid("PFSC block table is corrupt"));
        }
        Ok(Self { inner, block_size, offsets, len, cache: Mutex::new(Vec::new()) })
    }

    pub fn block_size(&self) -> u64 {
        self.block_size
    }

    /// True when block `idx` is stored uncompressed.
    fn is_raw(&self, idx: usize) -> bool {
        self.offsets[idx + 1] - self.offsets[idx] == self.block_size
    }

    fn decode_block(&self, idx: usize) -> io::Result<Arc<Vec<u8>>> {
        {
            let mut c = self.cache.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(pos) = c.iter().position(|(i, _)| *i == idx as u64) {
                let e = c.remove(pos);
                let data = e.1.clone();
                c.push(e);
                return Ok(data);
            }
        }
        let (start, end) = (self.offsets[idx], self.offsets[idx + 1]);
        let stored = end - start;
        let bs = self.block_size as usize;
        let mut out = vec![0u8; bs];
        if stored == self.block_size {
            self.inner.read_exact_at(&mut out, start)?;
        } else if stored != 0 && stored < self.block_size {
            let comp = read_vec(&*self.inner, start, stored as usize)?;
            inflate_block(&comp, &mut out).map_err(|e| invalid(format!("PFSC block {idx}: {e}")))?;
        }
        let data = Arc::new(out);
        let mut c = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        if c.len() >= CACHE_BLOCKS {
            c.remove(0);
        }
        c.push((idx as u64, data.clone()));
        Ok(data)
    }
}

/// Decompress one block: zlib first, raw deflate as the fallback. Output
/// shorter than the block is zero-padded.
fn inflate_block(comp: &[u8], out: &mut [u8]) -> io::Result<()> {
    fn fill(mut r: impl Read, out: &mut [u8]) -> io::Result<()> {
        let mut done = 0;
        while done < out.len() {
            match r.read(&mut out[done..])? {
                0 => break,
                n => done += n,
            }
        }
        out[done..].fill(0);
        Ok(())
    }
    let zlib_header = comp.len() >= 2 && comp[0] & 0x0F == 8 && (u16::from(comp[0]) << 8 | u16::from(comp[1])) % 31 == 0;
    if zlib_header && fill(flate2::read::ZlibDecoder::new(comp), out).is_ok() {
        return Ok(());
    }
    fill(flate2::read::DeflateDecoder::new(comp), out)
}

impl ReadAt for Pfsc {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        if offset.checked_add(buf.len() as u64).is_none_or(|e| e > self.len) {
            return Err(crate::io::eof());
        }
        let mut done = 0usize;
        while done < buf.len() {
            let pos = offset + done as u64;
            let idx = (pos / self.block_size) as usize;
            let within = pos % self.block_size;
            let n = ((self.block_size - within) as usize).min(buf.len() - done);
            if self.is_raw(idx) {
                self.inner.read_exact_at(&mut buf[done..done + n], self.offsets[idx] + within)?;
            } else {
                let block = self.decode_block(idx)?;
                buf[done..done + n].copy_from_slice(&block[within as usize..within as usize + n]);
            }
            done += n;
        }
        Ok(())
    }
}

/// Open a PFSC stream of either format.
pub fn open(inner: Source) -> io::Result<Source> {
    let mut head = [0u8; 8];
    inner.read_exact_at(&mut head, 0)?;
    if le_u32(&head, 4) == KRAKEN_VERSION {
        Ok(Arc::new(PfscKraken::new(inner)?))
    } else {
        Ok(Arc::new(Pfsc::new(inner)?))
    }
}

/// Format version of the Kraken-coded file format.
pub const KRAKEN_VERSION: u32 = 2;
const KRAKEN_BLOCK: u64 = 0x2_0000;
const OFFSET_MASK: u64 = (1 << 48) - 1;
const GROUP_START: u16 = 0x8000;
const GROUP_KRAKEN: u16 = 0x4000;

/// A group of blocks decoded together.
#[derive(Clone, Copy)]
struct Group {
    stored: u64,
    stored_len: u64,
    /// Stored length of the first sub-chunk (two-block groups).
    first_len: u64,
    blocks: u64,
    kraken: bool,
}

/// A Kraken-coded PFSC file (format version 2).
pub struct PfscKraken {
    inner: Source,
    /// Group of every block (index into `groups`).
    block_group: Vec<u32>,
    groups: Vec<Group>,
    /// First block of every group.
    group_first: Vec<u64>,
    len: u64,
    cache: Mutex<Vec<(u32, Arc<Vec<u8>>)>>,
}

impl PfscKraken {
    pub fn new(inner: Source) -> io::Result<Self> {
        let h = read_vec(&*inner, 0, 0x48.min(inner.len() as usize))?;
        if h.len() < 0x48 || !is_pfsc(&h) || le_u32(&h, 4) != KRAKEN_VERSION {
            return Err(invalid("not a Kraken PFSC file"));
        }
        let (codec, block, count, len) = (le_u32(&h, 8), le_u32(&h, 0x0C) as u64, le_u32(&h, 0x14) as u64, le_u64(&h, 0x18));
        let (end, table, data) = (le_u64(&h, 0x20), le_u64(&h, 0x30), le_u64(&h, 0x40));
        if codec != 2 || block != KRAKEN_BLOCK {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("PFSC version 2 with codec {codec} and block size {block:#x} is not supported"),
            ));
        }
        if count != len.div_ceil(KRAKEN_BLOCK).max(1) || count > 1 << 26 {
            return Err(invalid("PFSC block count does not match its length"));
        }
        let table_len = (count + 1) * 8;
        if table.checked_add(table_len).is_none_or(|e| e > data) || data > end || end > inner.len() {
            return Err(invalid("PFSC table or data range is invalid"));
        }
        let raw = read_vec(&*inner, table, table_len as usize)?;
        let word = |i: u64| le_u64(&raw, i as usize * 8);
        let flags = |i: u64| (word(i) >> 48) as u16;
        let off = |i: u64| word(i) & OFFSET_MASK;
        if word(count) != (u64::from(GROUP_START) << 48 | end) {
            return Err(invalid("PFSC final table word is invalid"));
        }
        let mut groups = Vec::new();
        let mut group_first = Vec::new();
        let mut block_group = Vec::with_capacity(count as usize);
        let mut i = 0;
        while i < count {
            let f = flags(i);
            if f & GROUP_START == 0 {
                return Err(invalid(format!("PFSC block {i} is a continuation without a group")));
            }
            let kraken = f == GROUP_START | GROUP_KRAKEN;
            if !kraken && f != GROUP_START {
                return Err(invalid(format!("PFSC block {i} has unknown flags {f:#x}")));
            }
            let blocks = if kraken && i + 1 < count && flags(i + 1) == GROUP_KRAKEN { 2 } else { 1 };
            let (start, next, stop) = (off(i), off(i + 1), off(i + blocks));
            if start < data || next < start || stop < next || stop > end {
                return Err(invalid(format!("PFSC block {i} has an invalid stored range")));
            }
            let logical = (len.saturating_sub(i * KRAKEN_BLOCK)).min(blocks * KRAKEN_BLOCK);
            if !kraken && len != 0 && stop - start != logical {
                return Err(invalid(format!("PFSC stored block {i} has the wrong size")));
            }
            group_first.push(i);
            for _ in 0..blocks {
                block_group.push(groups.len() as u32);
            }
            groups.push(Group { stored: start, stored_len: stop - start, first_len: next - start, blocks, kraken });
            i += blocks;
        }
        Ok(Self { inner, block_group, groups, group_first, len, cache: Mutex::new(Vec::new()) })
    }

    fn group_data(&self, g: u32) -> io::Result<Arc<Vec<u8>>> {
        {
            let mut c = self.cache.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(pos) = c.iter().position(|(i, _)| *i == g) {
                let e = c.remove(pos);
                let data = e.1.clone();
                c.push(e);
                return Ok(data);
            }
        }
        let grp = self.groups[g as usize];
        let first = self.group_first[g as usize];
        let n = (self.len - first * KRAKEN_BLOCK).min(grp.blocks * KRAKEN_BLOCK) as usize;
        let payload = read_vec(&*self.inner, grp.stored, grp.stored_len as usize)?;
        let mut out = vec![0u8; n];
        let (flags, first_len) = if grp.blocks == 2 {
            (crate::kraken::FLAG_CHUNK0_LZ | crate::kraken::FLAG_CHUNK1_LZ, grp.first_len as usize)
        } else {
            (crate::kraken::FLAG_CHUNK0_LZ, 0)
        };
        crate::kraken::decode_block(&payload, flags, first_len, &mut out)
            .map_err(|e| invalid(format!("PFSC block {first}: {e}")))?;
        let data = Arc::new(out);
        let mut c = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(pos) = c.iter().position(|(i, _)| *i == g) {
            // Another reader decoded it meanwhile.
            c.remove(pos);
        }
        if c.len() >= CACHE_BLOCKS {
            c.remove(0);
        }
        c.push((g, data.clone()));
        Ok(data)
    }
}

impl ReadAt for PfscKraken {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        if offset.checked_add(buf.len() as u64).is_none_or(|e| e > self.len) {
            return Err(crate::io::eof());
        }
        let mut done = 0usize;
        while done < buf.len() {
            let pos = offset + done as u64;
            let g = self.block_group[(pos / KRAKEN_BLOCK) as usize];
            let grp = self.groups[g as usize];
            let start = self.group_first[g as usize] * KRAKEN_BLOCK;
            let within = pos - start;
            let n = ((grp.blocks * KRAKEN_BLOCK - within) as usize).min(buf.len() - done);
            if grp.kraken {
                let data = self.group_data(g)?;
                buf[done..done + n].copy_from_slice(&data[within as usize..within as usize + n]);
            } else {
                self.inner.read_exact_at(&mut buf[done..done + n], grp.stored + within)?;
            }
            done += n;
        }
        Ok(())
    }
}

/// Build a Kraken-format PFSC file with stored blocks only (tests).
pub fn build_pfsc_stored_v2(data: &[u8]) -> Vec<u8> {
    let count = (data.len() as u64).div_ceil(KRAKEN_BLOCK).max(1);
    let table = 0x400u64;
    let start = (table + (count + 1) * 8).next_multiple_of(32);
    let mut out = vec![0u8; start as usize];
    out[0..4].copy_from_slice(&PFSC_MAGIC.to_le_bytes());
    out[4..8].copy_from_slice(&KRAKEN_VERSION.to_le_bytes());
    out[8..12].copy_from_slice(&2u32.to_le_bytes());
    out[0x0C..0x10].copy_from_slice(&(KRAKEN_BLOCK as u32).to_le_bytes());
    out[0x10..0x14].copy_from_slice(&32u32.to_le_bytes());
    out[0x14..0x18].copy_from_slice(&(count as u32).to_le_bytes());
    out[0x18..0x20].copy_from_slice(&(data.len() as u64).to_le_bytes());
    out[0x30..0x38].copy_from_slice(&table.to_le_bytes());
    out[0x40..0x48].copy_from_slice(&start.to_le_bytes());
    let mut words = Vec::new();
    for i in 0..count as usize {
        words.push(u64::from(GROUP_START) << 48 | out.len() as u64);
        let chunk = &data[(i * KRAKEN_BLOCK as usize).min(data.len())..((i + 1) * KRAKEN_BLOCK as usize).min(data.len())];
        if data.is_empty() {
            out.resize(out.len() + KRAKEN_BLOCK as usize, 0);
        } else {
            out.extend_from_slice(chunk);
        }
    }
    let end = out.len() as u64;
    words.push(u64::from(GROUP_START) << 48 | end);
    out[0x20..0x28].copy_from_slice(&end.to_le_bytes());
    for (i, w) in words.iter().enumerate() {
        out[table as usize + i * 8..table as usize + i * 8 + 8].copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// Build a PFSC stream from `data` (tests, and tools that want to write
/// one). Blocks that shrink under zlib are stored compressed.
pub fn build_pfsc(data: &[u8], block_size: usize, compress: bool) -> Vec<u8> {
    use std::io::Write;
    let blocks = data.len().div_ceil(block_size);
    let table = 0x400usize;
    let data_start = (table + (blocks + 1) * 8).next_multiple_of(block_size);
    let mut out = vec![0u8; data_start];
    out[0..4].copy_from_slice(&PFSC_MAGIC.to_le_bytes());
    out[8..12].copy_from_slice(&6u32.to_le_bytes());
    out[0x0C..0x10].copy_from_slice(&(block_size as u32).to_le_bytes());
    out[0x10..0x18].copy_from_slice(&(block_size as u64).to_le_bytes());
    out[0x18..0x20].copy_from_slice(&(table as u64).to_le_bytes());
    out[0x20..0x28].copy_from_slice(&(data_start as u64).to_le_bytes());
    out[0x28..0x30].copy_from_slice(&(data.len() as u64).to_le_bytes());
    let mut offsets = vec![data_start as u64];
    for i in 0..blocks {
        let mut block = data[i * block_size..((i + 1) * block_size).min(data.len())].to_vec();
        block.resize(block_size, 0);
        let mut stored = block.clone();
        if compress {
            let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
            enc.write_all(&block).unwrap();
            let z = enc.finish().unwrap();
            if z.len() < block_size {
                stored = z;
            }
        }
        out.extend_from_slice(&stored);
        offsets.push(out.len() as u64);
    }
    for (i, o) in offsets.iter().enumerate() {
        out[table + i * 8..table + i * 8 + 8].copy_from_slice(&o.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Memory;

    #[test]
    fn roundtrip_raw_and_compressed() {
        let mut data = vec![0u8; 0x5000 + 123];
        for (i, b) in data.iter_mut().enumerate() {
            // Compressible in the first blocks, noisy afterwards.
            *b = if i < 0x2000 { (i / 512) as u8 } else { (i.wrapping_mul(2654435761) >> 13) as u8 };
        }
        for compress in [false, true] {
            let img = build_pfsc(&data, 0x1000, compress);
            let p = Pfsc::new(Arc::new(Memory(img))).unwrap();
            assert_eq!(p.len(), data.len() as u64);
            let mut all = vec![0u8; data.len()];
            p.read_exact_at(&mut all, 0).unwrap();
            assert_eq!(all, data);
            let mut part = vec![0u8; 0x1800];
            p.read_exact_at(&mut part, 0xF00).unwrap();
            assert_eq!(part, data[0xF00..0x2700]);
            assert!(p.read_exact_at(&mut part, data.len() as u64 - 4).is_err());
        }
    }

    #[test]
    fn kraken_format_stored_blocks() {
        let data: Vec<u8> = (0..0x5_1234u32).map(|i| (i.wrapping_mul(2654435761) >> 9) as u8).collect();
        let f = build_pfsc_stored_v2(&data);
        let p = open(Arc::new(Memory(f.clone()))).unwrap();
        assert_eq!(p.len(), data.len() as u64);
        let mut all = vec![0u8; data.len()];
        p.read_exact_at(&mut all, 0).unwrap();
        assert_eq!(all, data);
        let mut part = vec![0u8; 0x3_0000];
        p.read_exact_at(&mut part, 0x1_FFF0).unwrap();
        assert_eq!(part, data[0x1_FFF0..0x4_FFF0]);
        // Empty file: one stored zero block.
        let e = open(Arc::new(Memory(build_pfsc_stored_v2(&[])))).unwrap();
        assert_eq!(e.len(), 0);
        // A broken final word or flags are rejected.
        let mut bad = f.clone();
        bad[0x400 + 3 * 8 + 7] = 0x12;
        assert!(open(Arc::new(Memory(bad))).is_err());
        let mut bad = f;
        bad[0x400 + 8 + 7] = 0x40;
        assert!(open(Arc::new(Memory(bad))).is_err());
    }
}
