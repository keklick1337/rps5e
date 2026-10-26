// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Random-access byte sources.
//!
//! Every layer of a package (the file on disk, an encrypted PFS, a PFSC
//! stream, a file inside a PFS) is a [`ReadAt`]: positioned, thread-safe
//! reads with a known length. Layers wrap each other, so a file inside the
//! inner image of a fake PKG reads through `Slice -> Pfsc -> Slice -> Xts ->
//! Slice -> HostFile` without ever copying the whole image.

use std::fs::File;
use std::io;
use std::sync::Arc;

/// Fill `buf` from `offset` of a host file without moving its cursor.
#[cfg(unix)]
pub fn file_read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

/// Fill `buf` from `offset` of a host file (Windows moves the cursor; every
/// read here is positioned, so that is harmless).
#[cfg(windows)]
pub fn file_read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// A positioned, thread-safe byte source of fixed length.
pub trait ReadAt: Send + Sync {
    /// Total length in bytes.
    fn len(&self) -> u64;

    /// Fill `buf` from `offset`. Reading past the end is an error
    /// ([`io::ErrorKind::UnexpectedEof`]); use [`read_at`] for short reads.
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Shared handle to a byte source.
pub type Source = Arc<dyn ReadAt>;

pub(crate) fn eof() -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "read past the end of the data")
}

pub(crate) fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Read up to `buf.len()` bytes at `offset`, stopping at the end of the
/// source. Returns the number of bytes read (0 at or past the end).
pub fn read_at(src: &dyn ReadAt, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    let len = src.len();
    if offset >= len {
        return Ok(0);
    }
    let n = (len - offset).min(buf.len() as u64) as usize;
    src.read_exact_at(&mut buf[..n], offset)?;
    Ok(n)
}

/// Read `len` bytes at `offset` into a new vector.
pub fn read_vec(src: &dyn ReadAt, offset: u64, len: usize) -> io::Result<Vec<u8>> {
    let mut v = vec![0u8; len];
    src.read_exact_at(&mut v, offset)?;
    Ok(v)
}

fn check_range(len: u64, offset: u64, n: usize) -> io::Result<()> {
    match offset.checked_add(n as u64) {
        Some(end) if end <= len => Ok(()),
        _ => Err(eof()),
    }
}

/// A host file.
pub struct HostFile {
    file: File,
    len: u64,
}

impl HostFile {
    pub fn new(file: File) -> io::Result<Self> {
        let len = file.metadata()?.len();
        Ok(Self { file, len })
    }
}

impl ReadAt for HostFile {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        check_range(self.len, offset, buf.len())?;
        file_read_exact_at(&self.file, buf, offset)
    }
}

/// An in-memory buffer.
pub struct Memory(pub Vec<u8>);

impl ReadAt for Memory {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        check_range(self.len(), offset, buf.len())?;
        let o = offset as usize;
        buf.copy_from_slice(&self.0[o..o + buf.len()]);
        Ok(())
    }
}

/// A window `[base, base + len)` of another source.
pub struct Slice {
    inner: Source,
    base: u64,
    len: u64,
}

impl Slice {
    /// The window must lie inside `inner`.
    pub fn new(inner: Source, base: u64, len: u64) -> io::Result<Self> {
        match base.checked_add(len) {
            Some(end) if end <= inner.len() => Ok(Self { inner, base, len }),
            _ => Err(invalid(format!(
                "range {base:#x}+{len:#x} exceeds the containing data ({:#x} bytes)",
                inner.len()
            ))),
        }
    }
}

impl ReadAt for Slice {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        check_range(self.len, offset, buf.len())?;
        self.inner.read_exact_at(buf, self.base + offset)
    }
}

/// A file stored in non-contiguous fixed-size blocks of another source.
pub struct Blocks {
    inner: Source,
    block_size: u64,
    /// Block numbers in file order.
    blocks: Vec<u64>,
    len: u64,
}

impl Blocks {
    pub fn new(inner: Source, block_size: u64, blocks: Vec<u64>, len: u64) -> io::Result<Self> {
        if block_size == 0 || len > blocks.len() as u64 * block_size {
            return Err(invalid("block list shorter than the file"));
        }
        for &b in &blocks {
            if b.checked_add(1).and_then(|e| e.checked_mul(block_size)).is_none_or(|e| e > inner.len()) {
                return Err(invalid(format!("block {b} lies outside the image")));
            }
        }
        Ok(Self { inner, block_size, blocks, len })
    }
}

impl ReadAt for Blocks {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        check_range(self.len, offset, buf.len())?;
        let mut done = 0usize;
        while done < buf.len() {
            let pos = offset + done as u64;
            let (idx, within) = (pos / self.block_size, pos % self.block_size);
            let n = ((self.block_size - within) as usize).min(buf.len() - done);
            let phys = self.blocks[idx as usize] * self.block_size + within;
            self.inner.read_exact_at(&mut buf[done..done + n], phys)?;
            done += n;
        }
        Ok(())
    }
}

/// Little-endian field readers over a byte slice; out-of-range reads give 0
/// (callers validate sizes first).
pub(crate) fn le_u16(b: &[u8], o: usize) -> u16 {
    b.get(o..o + 2).map_or(0, |s| u16::from_le_bytes([s[0], s[1]]))
}
pub(crate) fn le_u32(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4).map_or(0, |s| u32::from_le_bytes(s.try_into().unwrap()))
}
pub(crate) fn le_u64(b: &[u8], o: usize) -> u64 {
    b.get(o..o + 8).map_or(0, |s| u64::from_le_bytes(s.try_into().unwrap()))
}
pub(crate) fn be_u32(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4).map_or(0, |s| u32::from_be_bytes(s.try_into().unwrap()))
}
pub(crate) fn be_u64(b: &[u8], o: usize) -> u64 {
    b.get(o..o + 8).map_or(0, |s| u64::from_be_bytes(s.try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_and_blocks() {
        let data: Vec<u8> = (0..64u8).collect();
        let mem: Source = Arc::new(Memory(data));
        let s = Slice::new(mem.clone(), 8, 16).unwrap();
        let mut b = [0u8; 4];
        s.read_exact_at(&mut b, 2).unwrap();
        assert_eq!(b, [10, 11, 12, 13]);
        assert!(s.read_exact_at(&mut b, 14).is_err());
        assert_eq!(read_at(&s, &mut b, 14).unwrap(), 2);
        assert!(Slice::new(mem.clone(), 60, 8).is_err());

        // Blocks of 8 bytes in the order 3, 0, 5; file of 20 bytes.
        let f = Blocks::new(mem, 8, vec![3, 0, 5], 20).unwrap();
        let mut all = [0u8; 20];
        f.read_exact_at(&mut all, 0).unwrap();
        let want: Vec<u8> = (24..32).chain(0..8).chain(40..44).collect();
        assert_eq!(all.to_vec(), want);
        let mut mid = [0u8; 4];
        f.read_exact_at(&mut mid, 6).unwrap();
        assert_eq!(mid, [30, 31, 0, 1]);
    }
}
