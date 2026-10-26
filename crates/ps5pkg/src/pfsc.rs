// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! PFSC: the block-compressed container used for `pfs_image.dat` (the
//! inner image of a package) and for individually compressed PFS files.
//!
//! ```text
//! 0x00 u32 magic "PFSC"      0x04 u32 0
//! 0x08 u32 (6)               0x0C u32 block size (usually 0x10000)
//! 0x10 u64 block size again  0x18 u64 offset of the block offset table
//! 0x20 u64 data start        0x28 u64 decompressed length
//! ```
//! The table holds `blocks + 1` little-endian u64 offsets; block `i` is
//! stored in `[off[i], off[i+1])`. A stored size equal to the block size
//! means a raw block, a smaller one a zlib (or raw deflate) stream, a
//! larger one or zero a block of zeros.

use crate::io::{invalid, le_u32, le_u64, read_vec, ReadAt, Source};
use std::io::{self, Read};
use std::sync::Mutex;

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
    cache: Mutex<Vec<(u64, std::sync::Arc<Vec<u8>>)>>,
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

    fn decode_block(&self, idx: usize) -> io::Result<std::sync::Arc<Vec<u8>>> {
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
        let data = std::sync::Arc::new(out);
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
    use std::sync::Arc;

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
}
