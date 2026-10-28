// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! PFS: the read-only file system inside packages.
//!
//! The tree type [`Pfs`] also serves the game folder of exFAT and UFS2 disk
//! images (see [`crate::exfat`], [`crate::ufs`]).
//!
//! Superblock (little-endian, block 0):
//! ```text
//! 0x00 u64 version (1, PS5: 2) 0x08 u64 magic (20130315)
//! 0x10 u64 id                 0x18 u8 fmode, clean, ronly, rsv
//! 0x1C u16 mode               0x1E u16 -
//! 0x20 u32 block size         0x24 u32 nbackup
//! 0x28 u64 nblock             0x30 u64 inode count
//! 0x38 u64 data block count   0x40 u64 inode block count
//! 0x48 u64 super-root inode   0x50 inode describing the inode blocks
//! 0x370 u8[16] seed (encrypted / signed images)
//! ```
//! Mode bits: 1 signed, 2 64-bit block numbers, 4 encrypted (AES-XTS from
//! the second block on, see [`crate::crypto`]), 8 always set, 0x10 direct
//! offsets (PS5 game images, see below).
//!
//! PS5 packages put the superblock of their images anywhere (the package
//! header or a search finds it); block numbers still count from the start of
//! the image, so block 0 holds data there. An outer image marked encrypted
//! whose seed is `PPRPLAIN-NOAUTH!` is stored in plain text.
//!
//! Inodes are packed into the inode blocks (never straddling a block).
//! Four layouts exist: unsigned/signed x 32/64-bit block pointers; a
//! signed pointer is a 32-byte block hash followed by the block number.
//! Common inode header: `u16 mode, u16 nlink, u32 flags, u64 size,
//! u64 size_compressed, i64 atime, mtime, ctime, birthtime, u32 nsec[4],
//! u32 uid, gid, u64 x2, blocks` then 12 direct and 5 indirect pointers
//! (indirect 0 single, 1 double). Flag 1 marks a PFSC-compressed file.
//!
//! Directories are blocks of dirents: `u32 ino, u32 type (2 file, 3 dir,
//! 4 ".", 5 ".."), u32 name length, u32 entry size, name`; an entry size
//! of 0 ends the block. The super-root (normally inode 0) holds
//! `flat_path_table` and `uroot`, the root of the visible tree.
//!
//! Direct-offset images use 0xA8-byte inodes: the common header, then at
//! 0x60 the u64 byte offset of the contiguous data (files and directories),
//! then 64 bytes this reader does not need.

use crate::crypto::{pfs_xts_keys, Xts, XtsReader, XTS_SECTOR_SIZE};
use crate::io::{invalid, le_u16, le_u32, le_u64, read_vec, Blocks, Slice, Source};
use crate::pfsc::is_pfsc;
use crate::Error;
use std::collections::HashSet;
use std::io;
use std::sync::Arc;

pub const PFS_VERSION: u64 = 1;
/// Superblock version of PS5 images (same layout).
pub const PFS_VERSION_PS5: u64 = 2;
pub const PFS_MAGIC: u64 = 20130315;
pub const MODE_SIGNED: u16 = 0x1;
pub const MODE_64BIT: u16 = 0x2;
pub const MODE_ENCRYPTED: u16 = 0x4;
pub const MODE_DIRECT_OFFSETS: u16 = 0x10;
/// Seed of an image marked encrypted that is stored in plain text.
pub const PLAINTEXT_SEED: &[u8; 16] = b"PPRPLAIN-NOAUTH!";
pub const INODE_FLAG_COMPRESSED: u32 = 0x1;
pub const INODE_MODE_DIR: u16 = 0x4000;
pub const INODE_MODE_FILE: u16 = 0x8000;
pub const DIRENT_FILE: u32 = 2;
pub const DIRENT_DIR: u32 = 3;
pub const DIRENT_DOT: u32 = 4;
pub const DIRENT_DOTDOT: u32 = 5;
pub const SUPERBLOCK_SIZE: usize = 0x400;
const SEED_OFFSET: usize = 0x370;
const INODE_BLOCK_SIG_OFFSET: usize = 0x50;
/// Upper bound on tree nodes (a corrupt image must not exhaust memory).
const MAX_NODES: usize = 8_000_000;

/// True when `head` (at least 0x10 bytes) starts with a PFS superblock.
pub fn is_pfs(head: &[u8]) -> bool {
    matches!(le_u64(head, 0), PFS_VERSION | PFS_VERSION_PS5) && le_u64(head, 8) == PFS_MAGIC
}

/// One of the four on-disk inode layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InodeLayout {
    D32,
    S32,
    D64,
    S64,
    /// Byte offset of contiguous data instead of block pointers.
    Direct,
}

impl InodeLayout {
    fn from_mode(mode: u16) -> Self {
        if mode & MODE_DIRECT_OFFSETS != 0 {
            return Self::Direct;
        }
        match (mode & MODE_SIGNED != 0, mode & MODE_64BIT != 0) {
            (false, false) => Self::D32,
            (true, false) => Self::S32,
            (false, true) => Self::D64,
            (true, true) => Self::S64,
        }
    }
    /// Bytes of one inode.
    pub fn size(self) -> usize {
        match self {
            Self::Direct => 0xA8,
            _ => self.ptr_base() + 17 * self.ptr_size(),
        }
    }
    /// Offset of the first direct pointer.
    fn ptr_base(self) -> usize {
        match self {
            Self::D32 | Self::S32 | Self::Direct => 0x64,
            Self::D64 | Self::S64 => 0x68,
        }
    }
    /// Bytes of one block pointer (hash included for signed images).
    pub fn ptr_size(self) -> usize {
        match self {
            Self::D32 | Self::Direct => 4,
            Self::S32 => 36,
            Self::D64 => 8,
            Self::S64 => 40,
        }
    }
    /// Offset of the block number inside a pointer.
    fn ptr_value_offset(self) -> usize {
        match self {
            Self::D32 | Self::D64 | Self::Direct => 0,
            Self::S32 | Self::S64 => 32,
        }
    }
    fn ptr_is_64(self) -> bool {
        matches!(self, Self::D64 | Self::S64)
    }
    /// Block number of the pointer at `off` in `b`; negative = unused.
    fn read_ptr(self, b: &[u8], off: usize) -> i64 {
        let o = off + self.ptr_value_offset();
        if self.ptr_is_64() {
            le_u64(b, o) as i64
        } else {
            le_u32(b, o) as i32 as i64
        }
    }
}

/// A decoded inode.
#[derive(Clone, Debug, Default)]
pub struct Inode {
    pub mode: u16,
    pub nlink: u16,
    pub flags: u32,
    pub size: u64,
    pub size_compressed: u64,
    pub mtime: i64,
    pub blocks: u64,
    pub direct: [i64; 12],
    pub indirect: [i64; 5],
    /// Byte offset of the contiguous data (direct-offset images).
    pub data_offset: Option<u64>,
}

impl Inode {
    fn parse(b: &[u8], layout: InodeLayout) -> Self {
        if layout == InodeLayout::Direct {
            return Inode {
                mode: le_u16(b, 0),
                nlink: le_u16(b, 2),
                flags: le_u32(b, 4),
                size: le_u64(b, 8),
                size_compressed: le_u64(b, 0x10),
                mtime: le_u64(b, 0x20) as i64,
                data_offset: Some(le_u64(b, 0x60)),
                ..Default::default()
            };
        }
        let blocks = if layout.ptr_is_64() { le_u64(b, 0x60) } else { le_u32(b, 0x60) as u64 };
        let mut ino = Inode {
            mode: le_u16(b, 0),
            nlink: le_u16(b, 2),
            flags: le_u32(b, 4),
            size: le_u64(b, 8),
            size_compressed: le_u64(b, 0x10),
            mtime: le_u64(b, 0x20) as i64,
            blocks,
            ..Default::default()
        };
        let (base, ps) = (layout.ptr_base(), layout.ptr_size());
        for i in 0..12 {
            ino.direct[i] = layout.read_ptr(b, base + i * ps);
        }
        for i in 0..5 {
            ino.indirect[i] = layout.read_ptr(b, base + (12 + i) * ps);
        }
        ino
    }

    pub fn is_dir(&self) -> bool {
        self.mode & INODE_MODE_DIR != 0
    }
    pub fn is_compressed(&self) -> bool {
        self.flags & INODE_FLAG_COMPRESSED != 0
    }
}

/// Parsed superblock.
#[derive(Clone, Debug)]
pub struct Superblock {
    pub version: u64,
    pub mode: u16,
    pub block_size: u64,
    pub inode_count: u64,
    pub inode_block_count: u64,
    pub superroot_ino: u64,
    pub inode_blocks: Inode,
    pub seed: [u8; 16],
}

impl Superblock {
    pub fn parse(b: &[u8]) -> Result<Self, Error> {
        if b.len() < SUPERBLOCK_SIZE || !is_pfs(b) {
            return Err(Error::Format("not a PFS image (bad superblock version or magic)".into()));
        }
        let mode = le_u16(b, 0x1C);
        let block_size = le_u32(b, 0x20) as u64;
        if !(0x1000..=0x10_0000).contains(&block_size) || !block_size.is_power_of_two() {
            return Err(Error::Format(format!("PFS block size {block_size:#x} is invalid")));
        }
        let mut seed = [0u8; 16];
        seed.copy_from_slice(&b[SEED_OFFSET..SEED_OFFSET + 16]);
        // The inode-block descriptor always uses the signed 64-bit layout.
        let inode_blocks = Inode::parse(&b[INODE_BLOCK_SIG_OFFSET..], InodeLayout::S64);
        Ok(Self {
            version: le_u64(b, 0),
            mode,
            block_size,
            inode_count: le_u64(b, 0x30),
            inode_block_count: le_u64(b, 0x40),
            superroot_ino: le_u64(b, 0x48),
            inode_blocks,
            seed,
        })
    }

    pub fn layout(&self) -> InodeLayout {
        InodeLayout::from_mode(self.mode)
    }
    /// True when the data is encrypted (marked so, and not the plain-text seed).
    pub fn is_encrypted(&self) -> bool {
        self.mode & MODE_ENCRYPTED != 0 && &self.seed != PLAINTEXT_SEED
    }
}

/// Index of a node in a [`Pfs`] tree.
pub type NodeId = u32;

/// A file or directory of the visible tree.
#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    pub parent: NodeId,
    pub ino: u32,
    pub is_dir: bool,
    /// Logical (decompressed) size; 0 for directories.
    pub size: u64,
    pub mtime: i64,
    /// Children sorted by lower-case name (directories only).
    pub children: Vec<NodeId>,
}

/// Key material for an encrypted PFS.
#[derive(Clone, Copy)]
pub struct PfsKey {
    pub ekpfs: [u8; 32],
    /// The newer key schedule (package PFS flag bit 61).
    pub new_scheme: bool,
}

/// Which file system a [`Pfs`] tree was read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsKind {
    Pfs,
    ExFat,
    Ufs2,
}

impl FsKind {
    pub fn name(self) -> &'static str {
        match self {
            FsKind::Pfs => "PFS",
            FsKind::ExFat => "exFAT",
            FsKind::Ufs2 => "UFS2",
        }
    }
}

/// File contents of a tree read from a disk image (exFAT, UFS2): opens the
/// file whose node carries `key` in [`Node::ino`].
pub trait FileOpener: Send + Sync {
    fn open(&self, key: u32) -> Result<Source, Error>;
}

enum Backing {
    Pfs(PfsVolume),
    Image { kind: FsKind, files: Box<dyn FileOpener> },
}

/// A read-only game file tree with every directory indexed: a PFS image,
/// or the game folder of an exFAT / UFS2 disk image.
pub struct Pfs {
    src: Source,
    backing: Backing,
    nodes: Vec<Node>,
    /// Files added on top of the image ([`Pfs::add_file`]); their nodes
    /// carry `EXTRA_INO | index`.
    extra: Vec<Source>,
}

/// Inode-number flag of a file added with [`Pfs::add_file`].
const EXTRA_INO: u32 = 0x8000_0000;

/// Inode-level access to a PFS image.
struct PfsVolume {
    src: Source,
    sb: Superblock,
    layout: InodeLayout,
    inodes: Vec<Inode>,
    /// Block 0 holds data (the superblock lives elsewhere).
    block0_is_data: bool,
}

impl Pfs {
    /// Open a PFS image. Encrypted images need `key`.
    pub fn open(src: Source, key: Option<PfsKey>) -> Result<Self, Error> {
        Self::open_at(src, 0, key)
    }

    /// Open a PFS image whose superblock is at byte `sb_offset` (block
    /// numbers count from the start of `src`).
    pub fn open_at(src: Source, sb_offset: u64, key: Option<PfsKey>) -> Result<Self, Error> {
        if sb_offset.checked_add(SUPERBLOCK_SIZE as u64).is_none_or(|e| e > src.len()) {
            return Err(Error::Format("PFS image shorter than its superblock".into()));
        }
        let sb = Superblock::parse(&read_vec(&*src, sb_offset, SUPERBLOCK_SIZE)?)?;
        if sb_offset != 0 && sb.is_encrypted() {
            return Err(Error::Encrypted("the PS5 image is encrypted".into()));
        }
        let src: Source = if sb.is_encrypted() {
            let Some(k) = key else {
                return Err(Error::Encrypted("the PFS image is encrypted and no key is available".into()));
            };
            let (tweak, data) = pfs_xts_keys(&k.ekpfs, &sb.seed, k.new_scheme);
            Arc::new(XtsReader::new(src, Xts::new(&tweak, &data), sb.block_size / XTS_SECTOR_SIZE))
        } else {
            src
        };
        let layout = sb.layout();
        let mut vol = PfsVolume { src: src.clone(), sb, layout, inodes: Vec::new(), block0_is_data: sb_offset != 0 };
        let encrypted = vol.sb.is_encrypted();
        let bad_key = |e: Error| {
            if encrypted {
                Error::Encrypted(format!("the PFS image does not decrypt with the package key ({e})"))
            } else {
                e
            }
        };
        vol.read_inodes().map_err(bad_key)?;
        let nodes = vol.build_tree().map_err(bad_key)?;
        Ok(Pfs { src, backing: Backing::Pfs(vol), nodes, extra: Vec::new() })
    }

    /// A tree read from a disk image. `nodes[0]` is the root; every node's
    /// `parent` and `children` are set (children in any order: they are
    /// sorted here) and a file's `ino` is the key `files` opens it by.
    pub fn from_nodes(src: Source, kind: FsKind, mut nodes: Vec<Node>, files: Box<dyn FileOpener>) -> Self {
        sort_children(&mut nodes);
        Pfs { src, backing: Backing::Image { kind, files }, nodes, extra: Vec::new() }
    }

    /// The file system the tree comes from.
    pub fn kind(&self) -> FsKind {
        match &self.backing {
            Backing::Pfs(_) => FsKind::Pfs,
            Backing::Image { kind, .. } => *kind,
        }
    }

    /// The PFS superblock (PFS images only).
    pub fn superblock(&self) -> Option<&Superblock> {
        match &self.backing {
            Backing::Pfs(v) => Some(&v.sb),
            Backing::Image { .. } => None,
        }
    }

    /// The decrypted image (for nested images and diagnostics).
    pub fn source(&self) -> &Source {
        &self.src
    }

    /// The same tree re-rooted at directory `dir` (a game folder inside a
    /// disk image); nodes outside it are dropped.
    pub fn subtree(self, dir: NodeId) -> Self {
        if dir == self.root() || !self.nodes[dir as usize].is_dir {
            return self;
        }
        let mut nodes = Vec::new();
        let mut map: Vec<(NodeId, NodeId)> = vec![(dir, 0)];
        let mut root = self.nodes[dir as usize].clone();
        root.name = String::new();
        root.parent = 0;
        nodes.push(root);
        let mut i = 0;
        while i < map.len() {
            let (old, new) = map[i];
            i += 1;
            let mut kids = Vec::new();
            for &c in &self.nodes[old as usize].children {
                let id = nodes.len() as NodeId;
                let mut n = self.nodes[c as usize].clone();
                n.parent = new;
                nodes.push(n);
                kids.push(id);
                map.push((c, id));
            }
            nodes[new as usize].children = kids;
        }
        Pfs { src: self.src, backing: self.backing, nodes, extra: self.extra }
    }
}

/// Sort every directory's children by lower-case name (the order
/// [`Pfs::child`] searches).
fn sort_children(nodes: &mut [Node]) {
    for d in 0..nodes.len() {
        if nodes[d].children.len() < 2 {
            continue;
        }
        let mut kids = std::mem::take(&mut nodes[d].children);
        kids.sort_by(|&a, &b| {
            let (na, nb) = (&nodes[a as usize].name, &nodes[b as usize].name);
            na.to_ascii_lowercase().cmp(&nb.to_ascii_lowercase()).then_with(|| na.cmp(nb))
        });
        nodes[d].children = kids;
    }
}

impl PfsVolume {
    fn read_inodes(&mut self) -> Result<(), Error> {
        let bs = self.sb.block_size;
        let isz = self.layout.size() as u64;
        let per_block = bs / isz;
        let count = self.sb.inode_count;
        let nblocks = self.sb.inode_block_count;
        if count == 0 || nblocks == 0 || count > per_block * nblocks || nblocks * bs > self.src.len() {
            return Err(Error::Format(format!("inode table ({count} inodes in {nblocks} blocks) is invalid")));
        }
        let desc = &self.sb.inode_blocks;
        let list = match self.block_list(desc, nblocks)? {
            Some(l) => l,
            None => {
                let start = if desc.direct[0] > 0 { desc.direct[0] as u64 } else { 1 };
                (start..start + nblocks).collect()
            }
        };
        let mut inodes = Vec::with_capacity(count as usize);
        let mut buf = vec![0u8; bs as usize];
        for &blk in &list {
            if inodes.len() as u64 >= count {
                break;
            }
            self.read_block(blk, &mut buf)?;
            for j in 0..per_block {
                if inodes.len() as u64 >= count {
                    break;
                }
                let o = (j * isz) as usize;
                inodes.push(Inode::parse(&buf[o..o + isz as usize], self.layout));
            }
        }
        self.inodes = inodes;
        Ok(())
    }

    /// True when `b` is a usable block pointer.
    fn ptr_ok(&self, b: i64) -> bool {
        b > 0 || (b == 0 && self.block0_is_data)
    }

    fn read_block(&self, block: u64, buf: &mut [u8]) -> io::Result<()> {
        let off = block.checked_mul(self.sb.block_size).ok_or_else(|| invalid("block number overflow"))?;
        self.src.read_exact_at(buf, off)
    }

    /// Physical block numbers of an inode's first `count` blocks, or None
    /// when they are contiguous from the first direct pointer.
    fn block_list(&self, ino: &Inode, count: u64) -> Result<Option<Vec<u64>>, Error> {
        if count <= 1 || ino.direct[1] <= 0 {
            return Ok(None);
        }
        let bs = self.sb.block_size;
        let total_blocks = self.src.len() / bs;
        let mut out: Vec<u64> = Vec::with_capacity(count.min(1 << 24) as usize);
        let push = |out: &mut Vec<u64>, b: i64| -> Result<(), Error> {
            if !self.ptr_ok(b) || b as u64 >= total_blocks {
                return Err(Error::Format(format!("block pointer {b} lies outside the image")));
            }
            out.push(b as u64);
            Ok(())
        };
        for &b in ino.direct.iter().take(count.min(12) as usize) {
            push(&mut out, b)?;
        }
        let ps = self.layout.ptr_size();
        let per = bs as usize / ps;
        let mut buf = vec![0u8; bs as usize];
        let mut inner = vec![0u8; bs as usize];
        if (out.len() as u64) < count {
            let ib = ino.indirect[0];
            if !self.ptr_ok(ib) {
                return Err(Error::Format("missing single-indirect block".into()));
            }
            self.read_block(ib as u64, &mut buf)?;
            for i in 0..per {
                if out.len() as u64 >= count {
                    break;
                }
                push(&mut out, self.layout.read_ptr(&buf, i * ps))?;
            }
        }
        if (out.len() as u64) < count {
            let ib = ino.indirect[1];
            if !self.ptr_ok(ib) {
                return Err(Error::Format("missing double-indirect block".into()));
            }
            self.read_block(ib as u64, &mut buf)?;
            for i in 0..per {
                if out.len() as u64 >= count {
                    break;
                }
                let blk = self.layout.read_ptr(&buf, i * ps);
                if !self.ptr_ok(blk) {
                    return Err(Error::Format("bad double-indirect pointer".into()));
                }
                self.read_block(blk as u64, &mut inner)?;
                for k in 0..per {
                    if out.len() as u64 >= count {
                        break;
                    }
                    push(&mut out, self.layout.read_ptr(&inner, k * ps))?;
                }
            }
        }
        if (out.len() as u64) < count {
            return Err(Error::Unsupported("files needing triple-indirect blocks".into()));
        }
        if out.windows(2).all(|w| w[1] == w[0] + 1) {
            return Ok(None);
        }
        Ok(Some(out))
    }

    fn inode(&self, ino: u64) -> Result<&Inode, Error> {
        self.inodes.get(ino as usize).ok_or_else(|| Error::Format(format!("inode {ino} out of range")))
    }

    /// Raw on-disk bytes of an inode (`len` bytes from its first block).
    fn raw_data(&self, ino: &Inode, len: u64) -> Result<Source, Error> {
        let bs = self.sb.block_size;
        let nblocks = len.div_ceil(bs);
        if len == 0 {
            return Ok(Arc::new(crate::io::Memory(Vec::new())));
        }
        if let Some(off) = ino.data_offset {
            if off >= self.src.len() {
                return Err(Error::Format(format!("file data at {off:#x} lies outside the image")));
            }
            return Ok(Arc::new(Slice::new(self.src.clone(), off, len.min(self.src.len() - off))?));
        }
        match self.block_list(ino, nblocks)? {
            Some(list) => Ok(Arc::new(Blocks::new(self.src.clone(), bs, list, len)?)),
            None => {
                if !self.ptr_ok(ino.direct[0]) {
                    return Err(Error::Format("file without a data block".into()));
                }
                Ok(Arc::new(Slice::new(self.src.clone(), ino.direct[0] as u64 * bs, len)?))
            }
        }
    }

    /// Readable contents of an inode: the logical bytes, decompressed.
    fn inode_data(&self, ino: &Inode) -> Result<Source, Error> {
        if !ino.is_compressed() {
            return self.raw_data(ino, ino.size);
        }
        // The PFSC stream occupies whichever size field is the stored one;
        // never read past the blocks the inode owns.
        let cap = ino.blocks.saturating_mul(self.sb.block_size);
        let stored = if ino.data_offset.is_some() {
            // Direct-offset images keep the stored length in `size`.
            ino.size
        } else {
            ino.size.max(ino.size_compressed).min(if cap == 0 { u64::MAX } else { cap })
        };
        let raw = self.raw_data(ino, stored)?;
        let mut head = [0u8; 4];
        if raw.len() < 0x30 || raw.read_exact_at(&mut head, 0).is_err() || !is_pfsc(&head) {
            return self.raw_data(ino, ino.size);
        }
        // The PFSC header carries the exact decompressed length.
        Ok(crate::pfsc::open(raw)?)
    }

    fn read_dirents(&self, ino: &Inode) -> Result<Vec<(String, u32, u32)>, Error> {
        let bs = self.sb.block_size;
        let len = ino.blocks.max(ino.size.div_ceil(bs)).saturating_mul(bs).min(self.src.len());
        let data = self.raw_data(ino, len)?;
        let mut buf = vec![0u8; bs as usize];
        let mut out = Vec::new();
        for blk in 0..data.len().div_ceil(bs) {
            let n = (data.len() - blk * bs).min(bs) as usize;
            buf[n..].fill(0);
            data.read_exact_at(&mut buf[..n], blk * bs)?;
            let mut pos = 0usize;
            while pos + 16 <= buf.len() {
                let (dino, kind, nlen, esize) =
                    (le_u32(&buf, pos), le_u32(&buf, pos + 4), le_u32(&buf, pos + 8) as usize, le_u32(&buf, pos + 12) as usize);
                if esize == 0 {
                    break;
                }
                if esize < 16 || pos + esize > buf.len() || 16 + nlen > esize {
                    return Err(Error::Format("corrupt directory entry".into()));
                }
                let raw = &buf[pos + 16..pos + 16 + nlen];
                let raw = raw.split(|&c| c == 0).next().unwrap_or(raw);
                if kind == DIRENT_FILE || kind == DIRENT_DIR {
                    out.push((String::from_utf8_lossy(raw).into_owned(), dino, kind));
                }
                pos += esize;
            }
        }
        Ok(out)
    }

    fn node_size(&self, ino: &Inode) -> Result<u64, Error> {
        if ino.is_compressed() {
            Ok(self.inode_data(ino)?.len())
        } else {
            Ok(ino.size)
        }
    }

    fn build_tree(&self) -> Result<Vec<Node>, Error> {
        let sr = self.sb.superroot_ino;
        let sr_ino = self.inode(sr)?.clone();
        let has_data = sr_ino.data_offset.is_some() || (sr_ino.blocks != 0 && self.ptr_ok(sr_ino.direct[0]));
        if !sr_ino.is_dir() || !has_data {
            return Err(Error::Format("the super-root is not a directory".into()));
        }
        // The visible root is "uroot" inside the super-root (or the
        // super-root itself when an image has no uroot).
        let top = self.read_dirents(&sr_ino)?;
        let root_ino = top
            .iter()
            .find(|(n, _, k)| *k == DIRENT_DIR && n == "uroot")
            .map(|(_, i, _)| *i as u64)
            .unwrap_or(sr);
        let root = self.inode(root_ino)?.clone();
        let mut nodes = vec![Node {
            name: String::new(),
            parent: 0,
            ino: root_ino as u32,
            is_dir: true,
            size: 0,
            mtime: root.mtime,
            children: Vec::new(),
        }];
        let mut visited: HashSet<u64> = HashSet::from([root_ino]);
        let mut stack: Vec<NodeId> = vec![0];
        while let Some(dir) = stack.pop() {
            let dino = self.inode(nodes[dir as usize].ino as u64)?.clone();
            let mut kids = Vec::new();
            for (name, ino, kind) in self.read_dirents(&dino)? {
                if name.is_empty() || name == "." || name == ".." || name.contains(['/', '\\']) {
                    continue;
                }
                let inode = self.inode(ino as u64)?.clone();
                let is_dir = kind == DIRENT_DIR;
                if is_dir && !visited.insert(ino as u64) {
                    return Err(Error::Format("directory cycle".into()));
                }
                if nodes.len() >= MAX_NODES {
                    return Err(Error::Format("too many files".into()));
                }
                let id = nodes.len() as NodeId;
                let size = if is_dir { 0 } else { self.node_size(&inode)? };
                nodes.push(Node { name, parent: dir, ino, is_dir, size, mtime: inode.mtime, children: Vec::new() });
                if is_dir {
                    stack.push(id);
                }
                kids.push(id);
            }
            nodes[dir as usize].children = kids;
        }
        sort_children(&mut nodes);
        Ok(nodes)
    }
}

impl Pfs {
    // ---- tree access ----

    pub fn root(&self) -> NodeId {
        0
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id as usize]
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The child named `name`: exact match first, else the only child
    /// whose name matches ignoring ASCII case.
    pub fn child(&self, dir: NodeId, name: &str) -> Option<NodeId> {
        let kids = &self.nodes[dir as usize].children;
        let lower = name.to_ascii_lowercase();
        let start = kids.partition_point(|&k| self.nodes[k as usize].name.to_ascii_lowercase() < lower);
        let run: Vec<NodeId> = kids[start..]
            .iter()
            .copied()
            .take_while(|&k| self.nodes[k as usize].name.eq_ignore_ascii_case(name))
            .collect();
        match run.as_slice() {
            [] => None,
            [one] => Some(*one),
            many => many.iter().copied().find(|&k| self.nodes[k as usize].name == name),
        }
    }

    /// Resolve a path relative to the root ('/' or '\' separated, case-
    /// insensitive, "." and ".." honoured and clamped at the root).
    pub fn lookup(&self, path: &str) -> Option<NodeId> {
        let mut cur = self.root();
        for seg in path.split(['/', '\\']) {
            match seg {
                "" | "." => {}
                ".." => cur = self.nodes[cur as usize].parent,
                s => {
                    if !self.nodes[cur as usize].is_dir {
                        return None;
                    }
                    cur = self.child(cur, s)?;
                }
            }
        }
        Some(cur)
    }

    /// Path of a node relative to the root, '/' separated.
    pub fn path(&self, id: NodeId) -> String {
        let mut parts = Vec::new();
        let mut cur = id;
        while cur != self.root() {
            parts.push(self.nodes[cur as usize].name.as_str());
            cur = self.nodes[cur as usize].parent;
        }
        parts.reverse();
        parts.join("/")
    }

    /// Every node below the root in depth-first order (parents first).
    pub fn walk(&self) -> Vec<NodeId> {
        let mut out = Vec::with_capacity(self.nodes.len());
        let mut stack: Vec<NodeId> = self.nodes[0].children.iter().rev().copied().collect();
        while let Some(id) = stack.pop() {
            out.push(id);
            stack.extend(self.nodes[id as usize].children.iter().rev().copied());
        }
        out
    }

    /// Add a file at `path` (relative to the root, '/' separated; missing
    /// folders are created). An existing file or folder there is kept and
    /// false returned.
    pub fn add_file(&mut self, path: &str, data: Source) -> bool {
        let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
        let Some((name, dirs)) = parts.split_last() else { return false };
        let mut cur = self.root();
        for d in dirs {
            cur = match self.child(cur, d) {
                Some(c) if self.nodes[c as usize].is_dir => c,
                Some(_) => return false,
                None => self.push_node(cur, d, true, 0, 0),
            };
        }
        if self.child(cur, name).is_some() {
            return false;
        }
        let ino = EXTRA_INO | self.extra.len() as u32;
        let id = self.push_node(cur, name, false, data.len(), ino);
        self.nodes[id as usize].ino = ino;
        self.extra.push(data);
        true
    }

    fn push_node(&mut self, parent: NodeId, name: &str, is_dir: bool, size: u64, ino: u32) -> NodeId {
        let id = self.nodes.len() as NodeId;
        let mtime = self.nodes[parent as usize].mtime;
        self.nodes.push(Node { name: name.to_string(), parent, ino, is_dir, size, mtime, children: Vec::new() });
        self.nodes[parent as usize].children.push(id);
        sort_children(&mut self.nodes[..]);
        id
    }

    /// Readable contents of a file node.
    pub fn open_file(&self, id: NodeId) -> Result<FileData, Error> {
        let n = &self.nodes[id as usize];
        if n.is_dir {
            return Err(Error::Format(format!("{} is a directory", self.path(id))));
        }
        if n.ino & EXTRA_INO != 0 {
            if let Some(src) = self.extra.get((n.ino & !EXTRA_INO) as usize) {
                return Ok(FileData { src: src.clone() });
            }
        }
        let src = match &self.backing {
            Backing::Pfs(v) => v.inode_data(v.inode(n.ino as u64)?)?,
            Backing::Image { files, .. } => files.open(n.ino)?,
        };
        Ok(FileData { src })
    }

    /// Open the file at `path`.
    pub fn open_path(&self, path: &str) -> Result<FileData, Error> {
        let id = self.lookup(path).ok_or_else(|| Error::NotFound(path.to_string()))?;
        self.open_file(id)
    }

    /// Whole contents of the file at `path`.
    pub fn read(&self, path: &str) -> Result<Vec<u8>, Error> {
        Ok(self.open_path(path)?.read_all()?)
    }
}

/// The contents of one file inside a PFS image.
#[derive(Clone)]
pub struct FileData {
    src: Source,
}

impl FileData {
    pub fn len(&self) -> u64 {
        self.src.len()
    }
    pub fn is_empty(&self) -> bool {
        self.src.len() == 0
    }
    /// Read up to `buf.len()` bytes at `offset`; 0 at or past the end.
    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        crate::io::read_at(&*self.src, buf, offset)
    }
    pub fn read_all(&self) -> io::Result<Vec<u8>> {
        read_vec(&*self.src, 0, self.src.len() as usize)
    }
    /// The underlying byte source.
    pub fn source(&self) -> Source {
        self.src.clone()
    }
}
