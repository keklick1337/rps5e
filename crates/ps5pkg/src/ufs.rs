// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! UFS2: read-only access to FreeBSD UFS2 file systems (game dumps made on
//! the console as UFS2 disk images).
//!
//! The superblock sits at byte 65536 (little-endian `struct fs`):
//! ```text
//! 0x10 i32 iblkno: inode blocks, in fragments from a cylinder group start
//! 0x2C u32 ncg     0x30 i32 bsize   0x34 i32 fsize   0x38 i32 frag
//! 0x74 i32 nindir  0x78 u32 inopb   0xB8 u32 ipg     0xBC i32 fpg
//! 0x55C i32 magic 0x19540119
//! ```
//! Cylinder group `c` starts at fragment `c * fpg` and keeps its inodes
//! (`ipg` of them, 256 bytes each) from fragment `iblkno` on, so inode `i`
//! lives at byte `((i / ipg) * fpg + iblkno) * fsize + (i % ipg) * 256`.
//!
//! UFS2 inode: `u16 mode` (0o040000 directory, 0o100000 regular file),
//! `u64 size` at 0x10, `i64 mtime` at 0x28, 12 direct block pointers at
//! 0x70 and the single/double/triple indirect pointers at 0xD0. Pointers
//! are 64-bit fragment numbers; 0 is a hole. A logical block is `bsize`
//! bytes, the last one of a small file may be a run of fragments.
//!
//! Directories are 512-byte chunks of records: `u32 ino, u16 reclen,
//! u8 type, u8 namlen, name` (ino 0: unused record). The root is inode 2.

use crate::io::{le_u16, le_u32, le_u64, read_vec, ExtentList, ReadAt, Source};
use crate::pfs::{FileOpener, FsKind, Node, NodeId, Pfs};
use crate::Error;
use std::collections::HashSet;
use std::sync::Arc;

pub const SUPERBLOCK_OFFSET: u64 = 0x1_0000;
pub const MAGIC_OFFSET: u64 = 0x55C;
pub const UFS2_MAGIC: u32 = 0x1954_0119;
pub const ROOT_INO: u32 = 2;
pub const INODE_SIZE: u64 = 256;
pub const IFMT: u16 = 0o170000;
pub const IFDIR: u16 = 0o040000;
pub const IFREG: u16 = 0o100000;
const DIRBLKSIZ: usize = 512;
const NDADDR: usize = 12;
const MAX_DIR_BYTES: u64 = 256 << 20;
const MAX_NODES: usize = 8_000_000;

/// True when `src` holds a UFS2 superblock at its standard place.
pub fn is_ufs2(src: &dyn ReadAt) -> bool {
    let mut m = [0u8; 4];
    crate::io::read_at(src, &mut m, SUPERBLOCK_OFFSET + MAGIC_OFFSET).is_ok_and(|n| n == 4)
        && u32::from_le_bytes(m) == UFS2_MAGIC
}

fn format(msg: impl Into<String>) -> Error {
    Error::Format(format!("UFS2: {}", msg.into()))
}

/// A decoded UFS2 inode (the fields a read-only reader needs).
#[derive(Clone, Debug, Default)]
pub struct Inode {
    pub mode: u16,
    pub size: u64,
    pub mtime: i64,
    pub direct: [u64; NDADDR],
    pub indirect: [u64; 3],
}

impl Inode {
    pub fn is_dir(&self) -> bool {
        self.mode & IFMT == IFDIR
    }
    pub fn is_file(&self) -> bool {
        self.mode & IFMT == IFREG
    }
}

/// An opened UFS2 file system.
pub struct Ufs {
    src: Source,
    bsize: u64,
    fsize: u64,
    ipg: u64,
    fpg: u64,
    iblkno: u64,
    ncg: u64,
}

impl Ufs {
    pub fn open(src: Source) -> Result<Self, Error> {
        if !is_ufs2(&*src) {
            return Err(format("no superblock"));
        }
        let sb = read_vec(&*src, SUPERBLOCK_OFFSET, 0x560)?;
        let field = |o: usize| le_u32(&sb, o) as u64;
        let (iblkno, ncg, bsize, fsize, frag) = (field(0x10), field(0x2C), field(0x30), field(0x34), field(0x38));
        let (ipg, fpg) = (field(0xB8), field(0xBC));
        let ok = bsize.is_power_of_two()
            && (4096..=65536).contains(&bsize)
            && fsize.is_power_of_two()
            && fsize >= 512
            && fsize <= bsize
            && frag == bsize / fsize
            && ipg > 0
            && fpg > 0
            && ncg > 0
            && ipg * INODE_SIZE <= fpg * fsize;
        if !ok {
            return Err(format("invalid superblock geometry"));
        }
        Ok(Ufs { src, bsize, fsize, ipg, fpg, iblkno, ncg })
    }

    pub fn block_size(&self) -> u64 {
        self.bsize
    }

    /// Read inode `ino`.
    pub fn inode(&self, ino: u32) -> Result<Inode, Error> {
        let ino = ino as u64;
        if ino >= self.ipg * self.ncg {
            return Err(format(format!("inode {ino} out of range")));
        }
        let off = ((ino / self.ipg) * self.fpg + self.iblkno) * self.fsize + (ino % self.ipg) * INODE_SIZE;
        let b = read_vec(&*self.src, off, INODE_SIZE as usize)?;
        let mut i = Inode { mode: le_u16(&b, 0), size: le_u64(&b, 0x10), mtime: le_u64(&b, 0x28) as i64, ..Default::default() };
        for k in 0..NDADDR {
            i.direct[k] = le_u64(&b, 0x70 + k * 8);
        }
        for k in 0..3 {
            i.indirect[k] = le_u64(&b, 0xD0 + k * 8);
        }
        Ok(i)
    }

    fn frag_offset(&self, ptr: u64) -> Result<u64, Error> {
        ptr.checked_mul(self.fsize).filter(|&o| o < self.src.len()).ok_or_else(|| format(format!("block pointer {ptr:#x} lies outside the image")))
    }

    /// Append the data blocks under indirect block `ptr` (`level` 0: its
    /// pointers are data blocks) until `left` logical blocks are mapped.
    fn walk_indirect(&self, ptr: u64, level: u32, left: &mut u64, size_left: &mut u64, out: &mut ExtentList) -> Result<(), Error> {
        let per = self.bsize / 8;
        let span = per.pow(level + 1);
        if ptr == 0 {
            // A hole covering everything this pointer would map.
            let blocks = span.min(*left);
            let bytes = (blocks * self.bsize).min(*size_left);
            out.push(bytes, None);
            *left -= blocks;
            *size_left -= bytes;
            return Ok(());
        }
        let block = read_vec(&*self.src, self.frag_offset(ptr)?, self.bsize as usize)?;
        for k in 0..per as usize {
            if *left == 0 {
                break;
            }
            let p = le_u64(&block, k * 8);
            if level == 0 {
                self.push_block(p, left, size_left, out)?;
            } else {
                self.walk_indirect(p, level - 1, left, size_left, out)?;
            }
        }
        Ok(())
    }

    fn push_block(&self, p: u64, left: &mut u64, size_left: &mut u64, out: &mut ExtentList) -> Result<(), Error> {
        let bytes = self.bsize.min(*size_left);
        out.push(bytes, if p == 0 { None } else { Some(self.frag_offset(p)?) });
        *left -= 1;
        *size_left -= bytes;
        Ok(())
    }

    /// Readable contents of an inode.
    pub fn open_inode(&self, ino: &Inode) -> Result<Source, Error> {
        let mut out = ExtentList::new();
        let mut left = ino.size.div_ceil(self.bsize);
        let mut size_left = ino.size;
        for &p in &ino.direct {
            if left == 0 {
                break;
            }
            self.push_block(p, &mut left, &mut size_left, &mut out)?;
        }
        for (level, &p) in ino.indirect.iter().enumerate() {
            if left == 0 {
                break;
            }
            self.walk_indirect(p, level as u32, &mut left, &mut size_left, &mut out)?;
        }
        if left != 0 {
            return Err(format("file larger than its block map"));
        }
        Ok(out.build(self.src.clone())?)
    }

    /// The records of a directory: (name, inode).
    pub fn read_dir(&self, dir: &Inode) -> Result<Vec<(String, u32)>, Error> {
        if !dir.is_dir() || dir.size > MAX_DIR_BYTES {
            return Err(format("not a directory"));
        }
        let data = self.open_inode(dir)?;
        let raw = read_vec(&*data, 0, data.len() as usize)?;
        let mut out = Vec::new();
        for chunk in raw.chunks(DIRBLKSIZ) {
            let mut pos = 0usize;
            while pos + 8 <= chunk.len() {
                let (ino, reclen, namlen) = (le_u32(chunk, pos), le_u16(chunk, pos + 4) as usize, chunk[pos + 7] as usize);
                if reclen < 8 || pos + reclen > chunk.len() {
                    break;
                }
                if ino != 0 && 8 + namlen <= reclen {
                    let name = String::from_utf8_lossy(&chunk[pos + 8..pos + 8 + namlen]).into_owned();
                    if name != "." && name != ".." {
                        out.push((name, ino));
                    }
                }
                pos += reclen;
            }
        }
        Ok(out)
    }

    /// Index the whole tree from the root inode. Entries that are neither
    /// directories nor regular files (links, devices) are left out.
    pub fn into_tree(self) -> Result<Pfs, Error> {
        let src = self.src.clone();
        let root = self.inode(ROOT_INO)?;
        if !root.is_dir() {
            return Err(format("the root inode is not a directory"));
        }
        let fs = Arc::new(self);
        let mut nodes = vec![Node {
            name: String::new(),
            parent: 0,
            ino: ROOT_INO,
            is_dir: true,
            size: 0,
            mtime: root.mtime,
            children: Vec::new(),
        }];
        let mut visited: HashSet<u32> = HashSet::from([ROOT_INO]);
        let mut stack: Vec<(NodeId, Inode)> = vec![(0, root)];
        while let Some((dir, dino)) = stack.pop() {
            let mut kids = Vec::new();
            for (name, ino) in fs.read_dir(&dino)? {
                if name.is_empty() || name.contains(['/', '\\']) {
                    continue;
                }
                let inode = fs.inode(ino)?;
                let is_dir = inode.is_dir();
                if !is_dir && !inode.is_file() {
                    continue;
                }
                if is_dir && !visited.insert(ino) {
                    return Err(format("directory cycle"));
                }
                if nodes.len() >= MAX_NODES {
                    return Err(format("too many files"));
                }
                let id = nodes.len() as NodeId;
                nodes.push(Node {
                    name,
                    parent: dir,
                    ino,
                    is_dir,
                    size: if is_dir { 0 } else { inode.size },
                    mtime: inode.mtime,
                    children: Vec::new(),
                });
                kids.push(id);
                if is_dir {
                    stack.push((id, inode));
                }
            }
            nodes[dir as usize].children = kids;
        }
        Ok(Pfs::from_nodes(src, FsKind::Ufs2, nodes, Box::new(UfsFiles { fs })))
    }
}

struct UfsFiles {
    fs: Arc<Ufs>,
}

impl FileOpener for UfsFiles {
    fn open(&self, ino: u32) -> Result<Source, Error> {
        let inode = self.fs.inode(ino)?;
        self.fs.open_inode(&inode)
    }
}
