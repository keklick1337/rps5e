// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Minimal writers for PFS images, fake PKG containers and exFAT volumes.
//!
//! They exist so tests (here and in the file-system HLE) can build small
//! images in memory; they follow the layouts documented in [`crate::pfs`]
//! and [`crate::pkg`], not any packaging tool's exact block placement.

use crate::crypto::{derive_key, key_digest, pfs_xts_keys, Xts, XTS_SECTOR_SIZE};
use crate::pfs::{
    InodeLayout, DIRENT_DIR, DIRENT_DOT, DIRENT_DOTDOT, DIRENT_FILE, INODE_FLAG_COMPRESSED, INODE_MODE_DIR,
    INODE_MODE_FILE, MODE_DIRECT_OFFSETS, MODE_ENCRYPTED, MODE_SIGNED, PFS_MAGIC, PFS_VERSION, PFS_VERSION_PS5,
    PLAINTEXT_SEED,
};
use crate::io::{HostFile, Memory, ReadAt, Slice, Source};
use crate::naps;
use crate::pfsc::{build_pfsc, build_pfsc_stored_v2};
use crate::pkg;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// A PFS image to build.
#[derive(Clone)]
pub struct PfsBuilder {
    pub block_size: u64,
    /// Signed 32-bit inodes (block pointers carry a hash slot).
    pub signed: bool,
    /// Place file blocks out of order (exercises block lists and the
    /// indirect blocks; implies `signed`-style per-block pointers).
    pub scatter: bool,
    /// Encrypt with this EKPFS (and key schedule) and seed.
    pub encrypt: Option<([u8; 32], bool, [u8; 16])>,
    /// Superblock version (1; PS5 images use 2).
    pub version: u64,
    /// PS5 placement: data from block 0, the superblock and the inode blocks
    /// after it ([`PfsBuilder::build_with_superblock`] tells where).
    pub ps5_layout: bool,
    /// Direct-offset inodes (PS5 game images): contiguous data at a byte
    /// offset; compressed files are Kraken-format PFSC files (stored blocks).
    pub direct_offsets: bool,
    /// Mark the image encrypted with the plain-text seed (stored in clear).
    pub plaintext_marker: bool,
    files: Vec<(String, Vec<u8>, bool)>,
    dirs: Vec<String>,
}

impl Default for PfsBuilder {
    fn default() -> Self {
        Self {
            block_size: 0x1000,
            signed: false,
            scatter: false,
            encrypt: None,
            version: PFS_VERSION,
            ps5_layout: false,
            direct_offsets: false,
            plaintext_marker: false,
            files: Vec::new(),
            dirs: Vec::new(),
        }
    }
}

struct BNode {
    is_dir: bool,
    data: Vec<u8>,
    compress: bool,
    children: BTreeMap<String, usize>,
    parent: usize,
}

struct Plan {
    inode: InodeRec,
    blocks: Vec<u64>,
}

#[derive(Clone, Default)]
struct InodeRec {
    mode: u16,
    nlink: u16,
    flags: u32,
    size: u64,
    size_compressed: u64,
}

impl PfsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a file (`a/b/c.bin`); parents are created.
    pub fn file(mut self, path: &str, data: &[u8]) -> Self {
        self.files.push((path.to_string(), data.to_vec(), false));
        self
    }

    /// Add a file stored as a PFSC stream (inode flag "compressed").
    pub fn compressed_file(mut self, path: &str, data: &[u8]) -> Self {
        self.files.push((path.to_string(), data.to_vec(), true));
        self
    }

    /// Add an empty directory.
    pub fn dir(mut self, path: &str) -> Self {
        self.dirs.push(path.to_string());
        self
    }

    fn tree(&self) -> Vec<BNode> {
        let mut nodes = vec![BNode {
            is_dir: true,
            data: Vec::new(),
            compress: false,
            children: BTreeMap::new(),
            parent: 0,
        }];
        let ensure_dir = |nodes: &mut Vec<BNode>, parts: &[&str]| -> usize {
            let mut cur = 0;
            for p in parts {
                cur = match nodes[cur].children.get(*p) {
                    Some(&c) => c,
                    None => {
                        let id = nodes.len();
                        nodes.push(BNode {
                            is_dir: true,
                            data: Vec::new(),
                            compress: false,
                            children: BTreeMap::new(),
                            parent: cur,
                        });
                        nodes[cur].children.insert(p.to_string(), id);
                        id
                    }
                };
            }
            cur
        };
        for d in &self.dirs {
            let parts: Vec<&str> = d.split('/').filter(|s| !s.is_empty()).collect();
            ensure_dir(&mut nodes, &parts);
        }
        for (path, data, compress) in &self.files {
            let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
            let (name, dirs) = parts.split_last().expect("file path");
            let parent = ensure_dir(&mut nodes, dirs);
            let id = nodes.len();
            nodes.push(BNode {
                is_dir: false,
                data: data.clone(),
                compress: *compress,
                children: BTreeMap::new(),
                parent,
            });
            nodes[parent].children.insert(name.to_string(), id);
        }
        nodes
    }

    /// Serialize the image.
    pub fn build(&self) -> Vec<u8> {
        self.build_with_superblock().0
    }

    /// Serialize the image; also returns the byte offset of its superblock.
    pub fn build_with_superblock(&self) -> (Vec<u8>, u64) {
        let bs = self.block_size;
        let signed = (self.signed || self.scatter) && !self.direct_offsets;
        let layout = if self.direct_offsets {
            InodeLayout::Direct
        } else if signed {
            InodeLayout::S32
        } else {
            InodeLayout::D32
        };
        let isz = layout.size() as u64;
        let tree = self.tree();

        // Inode numbers: 0 super-root, 1 flat_path_table, 2 uroot, then the
        // tree in index order (tree index i > 0 -> inode 2 + i).
        let ino_of = |tree_idx: usize| -> u32 { 2 + tree_idx as u32 };
        let inode_count = 2 + tree.len() as u64;
        let inode_blocks = inode_count.div_ceil(bs / isz);

        // Directory contents.
        let dirents = |entries: &[(u32, u32, &str)]| -> Vec<u8> {
            let mut out: Vec<u8> = Vec::new();
            for &(ino, kind, name) in entries {
                let esize = (17 + name.len()).next_multiple_of(8);
                let block_left = bs as usize - out.len() % bs as usize;
                if esize > block_left {
                    out.resize(out.len() + block_left, 0);
                }
                let start = out.len();
                out.extend_from_slice(&ino.to_le_bytes());
                out.extend_from_slice(&kind.to_le_bytes());
                out.extend_from_slice(&(name.len() as u32).to_le_bytes());
                out.extend_from_slice(&(esize as u32).to_le_bytes());
                out.extend_from_slice(name.as_bytes());
                out.resize(start + esize, 0);
            }
            out
        };
        let superroot = dirents(&[(0, DIRENT_DOT, "."), (0, DIRENT_DOTDOT, ".."), (1, DIRENT_FILE, "flat_path_table"), (2, DIRENT_DIR, "uroot")]);
        let fpt = vec![0u8; 16];

        // Contents per inode.
        let mut contents: Vec<(InodeRec, Vec<u8>)> = Vec::new();
        let dir_rec = |len: usize, nlink: u16| InodeRec {
            mode: INODE_MODE_DIR | 0o555,
            nlink,
            flags: 0,
            size: (len as u64).div_ceil(bs).max(1) * bs,
            size_compressed: (len as u64).div_ceil(bs).max(1) * bs,
        };
        contents.push((dir_rec(superroot.len(), 3), superroot));
        contents.push((
            InodeRec { mode: INODE_MODE_FILE | 0o555, nlink: 1, flags: 0, size: 16, size_compressed: 16 },
            fpt,
        ));
        for (i, n) in tree.iter().enumerate() {
            if n.is_dir {
                let me = if i == 0 { 2 } else { ino_of(i) };
                let parent = if i == 0 { 0 } else { ino_of(n.parent) };
                let mut entries: Vec<(u32, u32, &str)> = vec![(me, DIRENT_DOT, "."), (parent, DIRENT_DOTDOT, "..")];
                for (name, &c) in &n.children {
                    entries.push((ino_of(c), if tree[c].is_dir { DIRENT_DIR } else { DIRENT_FILE }, name.as_str()));
                }
                let subdirs = n.children.values().filter(|&&c| tree[c].is_dir).count() as u16;
                let d = dirents(&entries);
                contents.push((dir_rec(d.len(), 2 + subdirs), d));
            } else if n.compress && self.direct_offsets {
                // PS5 convention: `size` is the stored length.
                let stream = build_pfsc_stored_v2(&n.data);
                contents.push((
                    InodeRec {
                        mode: INODE_MODE_FILE | 0o555,
                        nlink: 1,
                        flags: INODE_FLAG_COMPRESSED,
                        size: stream.len() as u64,
                        size_compressed: n.data.len() as u64,
                    },
                    stream,
                ));
            } else if n.compress {
                let stream = build_pfsc(&n.data, bs as usize, true);
                contents.push((
                    InodeRec {
                        mode: INODE_MODE_FILE | 0o555,
                        nlink: 1,
                        flags: INODE_FLAG_COMPRESSED,
                        size: n.data.len() as u64,
                        size_compressed: stream.len() as u64,
                    },
                    stream,
                ));
            } else {
                let len = n.data.len() as u64;
                contents.push((
                    InodeRec { mode: INODE_MODE_FILE | 0o555, nlink: 1, flags: 0, size: len, size_compressed: len },
                    n.data.clone(),
                ));
            }
        }

        // Block allocation: data blocks after the inode blocks; scattered
        // files get their blocks in reverse order, then indirect blocks.
        let mut next = if self.ps5_layout { 0 } else { 1 + inode_blocks };
        let mut plans: Vec<Plan> = Vec::new();
        let mut indirect_needed: Vec<usize> = Vec::new();
        for (idx, (rec, data)) in contents.iter().enumerate() {
            let n = (data.len() as u64).div_ceil(bs).max(if rec.mode & INODE_MODE_DIR != 0 { 1 } else { 0 });
            let mut blocks: Vec<u64> = (next..next + n).collect();
            next += n;
            if self.scatter && !self.direct_offsets && n > 1 && rec.mode & INODE_MODE_FILE != 0 {
                blocks.reverse();
            }
            if signed && n > 12 {
                indirect_needed.push(idx);
            }
            plans.push(Plan { inode: rec.clone(), blocks });
        }
        let mut indirect_block: BTreeMap<usize, u64> = BTreeMap::new();
        for idx in indirect_needed {
            indirect_block.insert(idx, next);
            next += 1;
        }
        let (sb_block, inode_start) = if self.ps5_layout { (next, next + 1) } else { (0, 1) };
        let total_blocks = if self.ps5_layout { inode_start + inode_blocks } else { next };
        let sb_off = (sb_block * bs) as usize;
        let mut img = vec![0u8; (total_blocks * bs) as usize];

        // Superblock.
        let mut mode: u16 = 0x8;
        if signed {
            mode |= MODE_SIGNED;
        }
        if self.encrypt.is_some() || self.plaintext_marker {
            mode |= MODE_ENCRYPTED;
        }
        if self.direct_offsets {
            mode |= MODE_DIRECT_OFFSETS;
        }
        let put = |img: &mut [u8], off: usize, b: &[u8]| img[off..off + b.len()].copy_from_slice(b);
        let sb = sb_off;
        put(&mut img, sb, &self.version.to_le_bytes());
        put(&mut img, sb + 8, &PFS_MAGIC.to_le_bytes());
        put(&mut img, sb + 0x1C, &mode.to_le_bytes());
        put(&mut img, sb + 0x20, &(bs as u32).to_le_bytes());
        put(&mut img, sb + 0x28, &1u64.to_le_bytes());
        put(&mut img, sb + 0x30, &inode_count.to_le_bytes());
        put(&mut img, sb + 0x38, &total_blocks.to_le_bytes());
        put(&mut img, sb + 0x40, &inode_blocks.to_le_bytes());
        put(&mut img, sb + 0x48, &0u64.to_le_bytes());
        // Inode-block descriptor (signed 64-bit layout) at 0x50.
        let d = sb + 0x50;
        put(&mut img, d + 8, &(inode_blocks * bs).to_le_bytes());
        put(&mut img, d + 0x10, &(inode_blocks * bs).to_le_bytes());
        put(&mut img, d + 0x60, &inode_blocks.to_le_bytes());
        for i in 0..inode_blocks.min(12) {
            put(&mut img, d + 0x68 + i as usize * 40 + 32, &(inode_start + i).to_le_bytes());
        }
        if let Some((_, _, seed)) = &self.encrypt {
            put(&mut img, sb + 0x370, seed);
        } else if self.plaintext_marker {
            put(&mut img, sb + 0x370, PLAINTEXT_SEED);
        }

        // Inodes.
        let per_block = bs / isz;
        let ps = layout.ptr_size();
        let base = 0x64usize;
        let vo = if signed { 32 } else { 0 };
        for (i, p) in plans.iter().enumerate() {
            let blk = inode_start + i as u64 / per_block;
            let off = (blk * bs + (i as u64 % per_block) * isz) as usize;
            let r = &p.inode;
            put(&mut img, off, &r.mode.to_le_bytes());
            put(&mut img, off + 2, &r.nlink.to_le_bytes());
            put(&mut img, off + 4, &r.flags.to_le_bytes());
            put(&mut img, off + 8, &r.size.to_le_bytes());
            put(&mut img, off + 0x10, &r.size_compressed.to_le_bytes());
            put(&mut img, off + 0x20, &1_600_000_000i64.to_le_bytes());
            if self.direct_offsets {
                let at = p.blocks.first().map_or(0, |&b| b * bs);
                put(&mut img, off + 0x60, &at.to_le_bytes());
                continue;
            }
            put(&mut img, off + 0x60, &(p.blocks.len() as u32).to_le_bytes());
            let direct = if signed { p.blocks.len().min(12) } else { p.blocks.len().min(1) };
            for (k, &b) in p.blocks.iter().take(direct).enumerate() {
                put(&mut img, off + base + k * ps + vo, &(b as i32).to_le_bytes());
            }
            if let Some(&ib) = indirect_block.get(&i) {
                put(&mut img, off + base + 12 * ps + vo, &(ib as i32).to_le_bytes());
                for (k, &b) in p.blocks.iter().skip(12).enumerate() {
                    put(&mut img, (ib * bs) as usize + k * ps + vo, &(b as i32).to_le_bytes());
                }
            }
        }

        // Data.
        for (p, (_, data)) in plans.iter().zip(&contents) {
            for (k, chunk) in data.chunks(bs as usize).enumerate() {
                let at = (p.blocks[k] * bs) as usize;
                img[at..at + chunk.len()].copy_from_slice(chunk);
            }
        }

        if let Some((ekpfs, new_scheme, seed)) = &self.encrypt {
            let (tweak, data) = pfs_xts_keys(ekpfs, seed, *new_scheme);
            let xts = Xts::new(&tweak, &data);
            let start = bs / XTS_SECTOR_SIZE;
            for s in start..img.len() as u64 / XTS_SECTOR_SIZE {
                let o = (s * XTS_SECTOR_SIZE) as usize;
                xts.encrypt_sector(&mut img[o..o + XTS_SECTOR_SIZE as usize], s);
            }
        }
        (img, sb_off as u64)
    }
}

/// A fake PKG to build.
pub struct PkgBuilder {
    pub content_id: String,
    pub content_type: u32,
    pub content_flags: u32,
    pub pfs_flags: u64,
    /// Passcode whose key digests are stored (None: no entry keys).
    pub passcode: Option<String>,
    pub param_sfo: Option<Vec<u8>>,
    pub icon0: Option<Vec<u8>>,
    /// More plain entries (id, bytes), e.g. a PS5 param.json.
    pub extra_entries: Vec<(u32, Vec<u8>)>,
    pub pfs_image: Vec<u8>,
}

impl PkgBuilder {
    pub fn new(content_id: &str, pfs_image: Vec<u8>) -> Self {
        Self {
            content_id: content_id.to_string(),
            content_type: pkg::CONTENT_TYPE_GD,
            content_flags: 0,
            pfs_flags: 0,
            passcode: Some(crate::crypto::DEFAULT_PASSCODE.to_string()),
            param_sfo: None,
            icon0: None,
            extra_entries: Vec::new(),
            pfs_image,
        }
    }

    pub fn build(&self) -> Vec<u8> {
        let mut entries: Vec<(u32, Vec<u8>)> = Vec::new();
        if let Some(pc) = &self.passcode {
            let mut keys = Vec::with_capacity(2048);
            let mut cid = self.content_id.as_bytes().to_vec();
            cid.resize(48, 0);
            keys.extend_from_slice(&<sha2::Sha256 as sha2::Digest>::digest(&cid));
            for i in 0..7 {
                keys.extend_from_slice(&key_digest(&derive_key(&self.content_id, pc, i)));
            }
            keys.resize(2048, 0);
            entries.push((pkg::ENTRY_KEYS, keys));
        }
        entries.push((pkg::ENTRY_NAMES, vec![0u8]));
        if let Some(s) = &self.param_sfo {
            entries.push((pkg::ENTRY_PARAM_SFO, s.clone()));
        }
        if let Some(i) = &self.icon0 {
            entries.push((pkg::ENTRY_ICON0_PNG, i.clone()));
        }
        entries.extend(self.extra_entries.iter().cloned());
        let table = 0x1000usize;
        let mut data_off = table + entries.len() * pkg::ENTRY_SIZE;
        let mut out = vec![0u8; data_off];
        let mut rows = Vec::new();
        for (id, bytes) in &entries {
            data_off = data_off.next_multiple_of(16);
            out.resize(data_off, 0);
            out.extend_from_slice(bytes);
            rows.push((*id, data_off as u32, bytes.len() as u32));
            data_off = out.len();
        }
        let pfs_off = out.len().next_multiple_of(0x10000);
        out.resize(pfs_off, 0);
        out.extend_from_slice(&self.pfs_image);
        let be32 = |out: &mut Vec<u8>, o: usize, v: u32| out[o..o + 4].copy_from_slice(&v.to_be_bytes());
        let be64 = |out: &mut Vec<u8>, o: usize, v: u64| out[o..o + 8].copy_from_slice(&v.to_be_bytes());
        out[..4].copy_from_slice(&pkg::PKG_MAGIC);
        be32(&mut out, 0x04, 0x8000_0001);
        be32(&mut out, 0x10, entries.len() as u32);
        out[0x14..0x16].copy_from_slice(&(entries.len() as u16).to_be_bytes());
        be32(&mut out, 0x18, table as u32);
        be64(&mut out, 0x20, table as u64);
        be64(&mut out, 0x28, (pfs_off - table) as u64);
        let cid = self.content_id.as_bytes();
        out[0x40..0x40 + cid.len().min(0x30)].copy_from_slice(&cid[..cid.len().min(0x30)]);
        be32(&mut out, 0x70, 0xF);
        be32(&mut out, 0x74, self.content_type);
        be32(&mut out, 0x78, self.content_flags);
        be32(&mut out, 0x404, 1);
        be64(&mut out, 0x408, self.pfs_flags);
        be64(&mut out, 0x410, pfs_off as u64);
        be64(&mut out, 0x418, self.pfs_image.len() as u64);
        let total = out.len() as u64;
        be64(&mut out, 0x430, total);
        for (i, (id, off, size)) in rows.iter().enumerate() {
            let e = table + i * pkg::ENTRY_SIZE;
            be32(&mut out, e, *id);
            be32(&mut out, e + 16, *off);
            be32(&mut out, e + 20, *size);
        }
        out
    }
}

/// A PS5 fake package ("\x7FFIH", plain-text mode) holding `inner`: the
/// game tree as a direct-offset image, packed through NAPS (stored and zero
/// spans), inside a plain outer PFS, with a CNT container carrying
/// `param.json` (and `icon0`) after it. `retail` sets the retail byte.
pub fn ps5_fake_pkg(inner: &PfsBuilder, content_id: &str, param_json: &[u8], icon0: Option<&[u8]>, retail: bool) -> Vec<u8> {
    const BLOCK: u64 = 0x1_0000;
    let mut inner = inner.clone();
    inner.block_size = BLOCK;
    inner.version = PFS_VERSION_PS5;
    inner.ps5_layout = true;
    inner.direct_offsets = true;
    let (img, sb) = inner.build_with_superblock();
    // Zero ublocks use the zero-span form; the superblock opens the last
    // file range, as in console packages.
    let zero = |i: usize| img.chunks(0x4_0000).nth(i).is_some_and(|c| c.len() > 16 && c.iter().all(|&b| b == 0));
    let ublock_aligned = sb % 0x4_0000 == 0;
    let form = |i: usize| if ublock_aligned && zero(i) { naps::SpanForm::Zero } else { naps::SpanForm::Stored };
    let (packed, layout) = naps::build_naps(&img, &[0, sb], form, |i| if i == 1 { 0x40 } else { 0 });
    let mut outer = PfsBuilder::new();
    outer.block_size = BLOCK;
    outer.version = PFS_VERSION_PS5;
    outer.signed = true;
    outer.ps5_layout = true;
    outer.plaintext_marker = true;
    outer.files.push((naps::IMAGE_FILE.to_string(), packed, false));
    outer.files.push((naps::LAYOUT_FILE.to_string(), layout, false));
    let (mut outer_img, outer_sb) = outer.build_with_superblock();
    mark_compressed_at(&mut outer_img, &outer, naps::IMAGE_FILE, outer_sb);
    let mut cnt = PkgBuilder::new(content_id, Vec::new());
    cnt.content_type = pkg::CONTENT_TYPE_PS5_GD;
    cnt.passcode = None;
    cnt.icon0 = icon0.map(|b| b.to_vec());
    cnt.extra_entries.push((pkg::ENTRY_PS5_PARAM_JSON, param_json.to_vec()));
    let cnt = cnt.build();
    let image_off = BLOCK;
    let cnt_off = (image_off + outer_img.len() as u64).next_multiple_of(BLOCK);
    let mut out = vec![0u8; image_off as usize];
    out[..4].copy_from_slice(&pkg::FIH_MAGIC);
    out[4] = 1;
    out[5] = if retail { 0x80 } else { 0 };
    let le = |out: &mut Vec<u8>, o: usize, v: u64| out[o..o + 8].copy_from_slice(&v.to_le_bytes());
    le(&mut out, 0x10, image_off);
    le(&mut out, 0x18, outer_img.len() as u64);
    le(&mut out, 0x20, image_off + outer_sb);
    le(&mut out, 0x58, cnt_off);
    out.extend_from_slice(&outer_img);
    out.resize(cnt_off as usize, 0);
    out.extend_from_slice(&cnt);
    out
}

/// Wrap a game tree the way a fake package does: an encrypted outer PFS
/// whose `uroot/pfs_image.dat` is the PFSC-compressed inner image.
pub fn fake_pkg_image(inner: &PfsBuilder, content_id: &str, passcode: &str, new_scheme: bool) -> Vec<u8> {
    let inner_img = inner.build();
    let pfsc = build_pfsc(&inner_img, 0x1000, true);
    let ekpfs = derive_key(content_id, passcode, 1);
    let mut outer = PfsBuilder::new();
    outer.signed = true;
    outer.encrypt = Some((ekpfs, new_scheme, [0x5A; 16]));
    outer.files.push(("pfs_image.dat".to_string(), pfsc, false));
    let mut img = outer.build();
    // The outer image marks pfs_image.dat as compressed (it is a PFSC).
    mark_compressed(&mut img, &outer, "pfs_image.dat");
    img
}

/// Set the compressed flag on a top-level file of a built image.
fn mark_compressed(img: &mut [u8], b: &PfsBuilder, name: &str) {
    mark_compressed_at(img, b, name, 0)
}

/// [`mark_compressed`] for an image whose superblock is at `sb_off` (the
/// inode blocks follow it).
fn mark_compressed_at(img: &mut [u8], b: &PfsBuilder, name: &str, sb_off: u64) {
    let tree = b.tree();
    let idx = tree[0].children[name];
    let ino = 2 + idx as u64;
    let layout = if b.signed || b.scatter { InodeLayout::S32 } else { InodeLayout::D32 };
    let isz = layout.size() as u64;
    let per = b.block_size / isz;
    let first = sb_off / b.block_size + 1;
    let off = ((first + ino / per) * b.block_size + (ino % per) * isz) as usize;
    if let Some((ekpfs, new_scheme, seed)) = &b.encrypt {
        // Decrypt the sector, patch, re-encrypt.
        let (tweak, data) = pfs_xts_keys(ekpfs, seed, *new_scheme);
        let xts = Xts::new(&tweak, &data);
        let s = off as u64 / XTS_SECTOR_SIZE;
        let o = (s * XTS_SECTOR_SIZE) as usize;
        let sector = &mut img[o..o + XTS_SECTOR_SIZE as usize];
        xts.decrypt_sector(sector, s);
        let rel = off - o + 4;
        let flags = u32::from_le_bytes(sector[rel..rel + 4].try_into().unwrap()) | INODE_FLAG_COMPRESSED;
        sector[rel..rel + 4].copy_from_slice(&flags.to_le_bytes());
        xts.encrypt_sector(sector, s);
    } else {
        let flags = u32::from_le_bytes(img[off + 4..off + 8].try_into().unwrap()) | INODE_FLAG_COMPRESSED;
        img[off + 4..off + 8].copy_from_slice(&flags.to_le_bytes());
    }
}

// ---------------------------------------------------------------------------
// exFAT images
// ---------------------------------------------------------------------------

/// How the clusters of one file of an [`ExFatBuilder`] are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExFatStore {
    /// Consecutive clusters with the NoFatChain flag (FAT entries unused).
    Contiguous,
    /// Consecutive clusters linked through the FAT.
    Chain,
    /// Clusters interleaved with the other fragmented files, linked
    /// through the FAT.
    Fragmented,
}

/// An exFAT volume to build: a writer for tests and for making sample
/// disk images from a game folder. File contents are byte sources read
/// only when the image is read or written, so very large (or sparse)
/// files cost nothing up front.
pub struct ExFatBuilder {
    /// Bytes per cluster (a power of two, 512 B .. 32 MiB).
    pub cluster_size: u64,
    files: Vec<(String, Source, ExFatStore)>,
    dirs: Vec<String>,
}

/// A built disk image: pieces at byte offsets, zeros in between.
pub struct DiskImage {
    len: u64,
    pieces: Vec<(u64, Source)>,
}

impl Default for ExFatBuilder {
    fn default() -> Self {
        Self { cluster_size: 0x1000, files: Vec::new(), dirs: Vec::new() }
    }
}

const EXFAT_SECTOR: u64 = 512;
/// FAT offset in sectors (after the main and backup boot regions).
const EXFAT_FAT_OFFSET: u64 = 128;

struct XNode {
    name: String,
    is_dir: bool,
    data: Option<(Source, ExFatStore)>,
    children: Vec<usize>,
    /// Clusters of the file or directory, in chain order.
    clusters: Vec<u32>,
    /// Directory entry bytes (directories only).
    entries: Vec<u8>,
}

impl ExFatBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a file (`a/b/c.bin`) stored contiguously; parents are created.
    pub fn file(self, path: &str, data: &[u8]) -> Self {
        self.file_source(path, Arc::new(Memory(data.to_vec())), ExFatStore::Contiguous)
    }

    /// Add a file with its contents as a byte source and a storage mode.
    pub fn file_source(mut self, path: &str, data: Source, store: ExFatStore) -> Self {
        self.files.push((path.to_string(), data, store));
        self
    }

    /// Add an empty directory.
    pub fn dir(mut self, path: &str) -> Self {
        self.dirs.push(path.to_string());
        self
    }

    /// Every file and directory below `root` on the host, stored
    /// contiguously under `prefix` (file contents are read when the image
    /// is written).
    pub fn host_dir(mut self, root: &Path, prefix: &str) -> std::io::Result<Self> {
        let mut stack = vec![(root.to_path_buf(), prefix.trim_matches('/').to_string())];
        while let Some((dir, rel)) = stack.pop() {
            if !rel.is_empty() {
                self.dirs.push(rel.clone());
            }
            for e in std::fs::read_dir(&dir)? {
                let e = e?;
                let name = e.file_name().to_string_lossy().into_owned();
                let path = if rel.is_empty() { name } else { format!("{rel}/{name}") };
                let ft = e.file_type()?;
                if ft.is_dir() {
                    stack.push((e.path(), path));
                } else if ft.is_file() {
                    let src: Source = Arc::new(HostFile::new(std::fs::File::open(e.path())?)?);
                    self.files.push((path, src, ExFatStore::Contiguous));
                }
            }
        }
        Ok(self)
    }

    fn tree(&self) -> Vec<XNode> {
        let mut nodes =
            vec![XNode { name: String::new(), is_dir: true, data: None, children: Vec::new(), clusters: Vec::new(), entries: Vec::new() }];
        fn ensure_dir(nodes: &mut Vec<XNode>, parts: &[&str]) -> usize {
            let mut cur = 0;
            for p in parts {
                cur = match nodes[cur].children.iter().copied().find(|&c| nodes[c].name == *p) {
                    Some(c) => c,
                    None => {
                        let id = nodes.len();
                        nodes.push(XNode {
                            name: p.to_string(),
                            is_dir: true,
                            data: None,
                            children: Vec::new(),
                            clusters: Vec::new(),
                            entries: Vec::new(),
                        });
                        nodes[cur].children.push(id);
                        id
                    }
                };
            }
            cur
        }
        for d in &self.dirs {
            let parts: Vec<&str> = d.split('/').filter(|s| !s.is_empty()).collect();
            ensure_dir(&mut nodes, &parts);
        }
        for (path, data, store) in &self.files {
            let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
            let (name, dirs) = parts.split_last().expect("file path");
            let parent = ensure_dir(&mut nodes, dirs);
            let id = nodes.len();
            nodes.push(XNode {
                name: name.to_string(),
                is_dir: false,
                data: Some((data.clone(), *store)),
                children: Vec::new(),
                clusters: Vec::new(),
                entries: Vec::new(),
            });
            nodes[parent].children.push(id);
        }
        nodes
    }

    /// Lay the volume out.
    pub fn build(&self) -> DiskImage {
        let cs = self.cluster_size;
        assert!(cs.is_power_of_two() && (EXFAT_SECTOR..=32 << 20).contains(&cs), "invalid cluster size");
        let mut nodes = self.tree();
        let upcase = exfat_upcase_table();
        let clusters_of = |bytes: u64| bytes.div_ceil(cs).max(1);
        // Directory sizes: a file set is 3 entries + one per 15 name units;
        // the root also holds the bitmap and up-case entries.
        let dir_bytes = |nodes: &Vec<XNode>, d: usize| -> u64 {
            let sets: u64 = nodes[d].children.iter().map(|&c| 32 * (2 + nodes[c].name.encode_utf16().count().div_ceil(15) as u64)).sum();
            sets + if d == 0 { 64 } else { 0 } + 32
        };
        let mut data_clusters = clusters_of(upcase.len() as u64);
        for d in 0..nodes.len() {
            data_clusters += if nodes[d].is_dir {
                clusters_of(dir_bytes(&nodes, d))
            } else {
                nodes[d].data.as_ref().map_or(0, |(s, _)| s.len().div_ceil(cs))
            };
        }
        let mut count = data_clusters + 1;
        loop {
            let need = data_clusters + clusters_of(count.div_ceil(8));
            if need <= count {
                break;
            }
            count = need;
        }
        let count = count as u32;
        let spc = cs / EXFAT_SECTOR;
        let fat_len = ((count as u64 + 2) * 4).div_ceil(EXFAT_SECTOR);
        let heap = (EXFAT_FAT_OFFSET + fat_len).next_multiple_of(spc);
        let volume_sectors = heap + count as u64 * spc;
        let mut fat = vec![0u32; count as usize + 2];
        fat[0] = 0xFFFF_FFF8;
        fat[1] = 0xFFFF_FFFF;
        let mut next = 2u32;
        let mut alloc = |n: u64| -> Vec<u32> {
            let v: Vec<u32> = (next..next + n as u32).collect();
            next += n as u32;
            v
        };
        let bitmap_clusters = alloc(clusters_of((count as u64).div_ceil(8)));
        let upcase_clusters = alloc(clusters_of(upcase.len() as u64));
        let mut chained: Vec<Vec<u32>> = vec![bitmap_clusters.clone(), upcase_clusters.clone()];
        for d in 0..nodes.len() {
            if nodes[d].is_dir {
                nodes[d].clusters = alloc(clusters_of(dir_bytes(&nodes, d)));
                chained.push(nodes[d].clusters.clone());
            }
        }
        let mut fragmented: Vec<(usize, u64)> = Vec::new();
        for (i, n) in nodes.iter_mut().enumerate().filter(|(_, n)| !n.is_dir) {
            let (src, store) = n.data.as_ref().unwrap();
            let k = src.len().div_ceil(cs);
            match store {
                ExFatStore::Fragmented => fragmented.push((i, k)),
                ExFatStore::Chain => {
                    n.clusters = alloc(k);
                    chained.push(n.clusters.clone());
                }
                ExFatStore::Contiguous => n.clusters = alloc(k),
            }
        }
        // Round-robin over the fragmented files.
        while fragmented.iter().any(|f| f.1 > 0) {
            for f in fragmented.iter_mut().filter(|f| f.1 > 0) {
                f.1 -= 1;
                nodes[f.0].clusters.push(alloc(1)[0]);
            }
        }
        for n in nodes.iter().filter(|n| matches!(n.data, Some((_, ExFatStore::Fragmented)))) {
            chained.push(n.clusters.clone());
        }
        let used = next - 2;
        assert!(used <= count);
        for chain in &chained {
            for w in chain.windows(2) {
                fat[w[0] as usize] = w[1];
            }
            if let Some(&last) = chain.last() {
                fat[last as usize] = 0xFFFF_FFFF;
            }
        }
        // Directory entries (the clusters of every node are known now).
        let ts = exfat_timestamp(2026, 1, 1, 12, 0, 0);
        for d in 0..nodes.len() {
            if !nodes[d].is_dir {
                continue;
            }
            let mut out = Vec::new();
            if d == 0 {
                let mut e = [0u8; 32];
                e[0] = crate::exfat::ENTRY_BITMAP;
                e[20..24].copy_from_slice(&bitmap_clusters[0].to_le_bytes());
                e[24..32].copy_from_slice(&(count as u64).div_ceil(8).to_le_bytes());
                out.extend_from_slice(&e);
                let mut e = [0u8; 32];
                e[0] = crate::exfat::ENTRY_UPCASE;
                e[4..8].copy_from_slice(&crate::exfat::table_checksum(&upcase).to_le_bytes());
                e[20..24].copy_from_slice(&upcase_clusters[0].to_le_bytes());
                e[24..32].copy_from_slice(&(upcase.len() as u64).to_le_bytes());
                out.extend_from_slice(&e);
            }
            for &c in &nodes[d].children {
                let n = &nodes[c];
                let (len, contiguous) = match &n.data {
                    Some((s, store)) => (s.len(), *store == ExFatStore::Contiguous),
                    None => (n.clusters.len() as u64 * cs, false),
                };
                let first = if len == 0 { 0 } else { n.clusters[0] };
                out.extend_from_slice(&exfat_entry_set(&n.name, n.is_dir, len, first, contiguous, ts));
            }
            nodes[d].entries = out;
        }
        // Serialize.
        let mut sys = vec![0u8; (heap * EXFAT_SECTOR) as usize];
        let boot = exfat_boot_region(volume_sectors, fat_len, heap, count, nodes[0].clusters[0], spc);
        sys[..boot.len()].copy_from_slice(&boot);
        sys[boot.len()..boot.len() * 2].copy_from_slice(&boot);
        let fo = (EXFAT_FAT_OFFSET * EXFAT_SECTOR) as usize;
        for (i, v) in fat.iter().enumerate() {
            sys[fo + i * 4..fo + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let coff = |c: u32| (heap + (c as u64 - 2) * spc) * EXFAT_SECTOR;
        let mut pieces: Vec<(u64, Source)> = vec![(0, Arc::new(Memory(sys)))];
        let mut bitmap = vec![0u8; (count as usize).div_ceil(8)];
        for i in 0..used as usize {
            bitmap[i / 8] |= 1 << (i % 8);
        }
        let mut put = |clusters: &[u32], src: Source| {
            // One piece per run of consecutive clusters.
            let mut i = 0;
            while i < clusters.len() {
                let mut j = i + 1;
                while j < clusters.len() && clusters[j] == clusters[j - 1] + 1 {
                    j += 1;
                }
                let start = i as u64 * cs;
                let len = ((j - i) as u64 * cs).min(src.len().saturating_sub(start));
                if len > 0 {
                    pieces.push((coff(clusters[i]), Arc::new(Slice::new(src.clone(), start, len).unwrap())));
                }
                i = j;
            }
        };
        put(&bitmap_clusters, Arc::new(Memory(bitmap)));
        put(&upcase_clusters, Arc::new(Memory(upcase)));
        for n in &nodes {
            if n.is_dir {
                put(&n.clusters, Arc::new(Memory(n.entries.clone())));
            } else if let Some((s, _)) = &n.data {
                put(&n.clusters, s.clone());
            }
        }
        pieces.sort_by_key(|p| p.0);
        DiskImage { len: volume_sectors * EXFAT_SECTOR, pieces }
    }
}

impl DiskImage {
    /// Write the image to a host file (gaps are left sparse).
    pub fn write_to(&self, path: &Path) -> std::io::Result<()> {
        use std::io::{Seek, SeekFrom, Write};
        let mut f = std::fs::File::create(path)?;
        let mut buf = vec![0u8; 4 << 20];
        for (off, src) in &self.pieces {
            f.seek(SeekFrom::Start(*off))?;
            let mut pos = 0u64;
            while pos < src.len() {
                let n = (src.len() - pos).min(buf.len() as u64) as usize;
                src.read_exact_at(&mut buf[..n], pos)?;
                f.write_all(&buf[..n])?;
                pos += n as u64;
            }
        }
        f.set_len(self.len)
    }

    /// The image as one byte vector (small images).
    pub fn to_vec(&self) -> Vec<u8> {
        let mut v = vec![0u8; self.len as usize];
        self.read_exact_at(&mut v, 0).unwrap();
        v
    }
}

impl ReadAt for DiskImage {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        if offset.checked_add(buf.len() as u64).is_none_or(|e| e > self.len) {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        buf.fill(0);
        let end = offset + buf.len() as u64;
        let first = self.pieces.partition_point(|p| p.0 + p.1.len() <= offset);
        for (start, src) in &self.pieces[first..] {
            if *start >= end {
                break;
            }
            let lo = offset.max(*start);
            let hi = end.min(start + src.len());
            if lo < hi {
                src.read_exact_at(&mut buf[(lo - offset) as usize..(hi - offset) as usize], lo - start)?;
            }
        }
        Ok(())
    }
}

/// An exFAT timestamp (no UTC offset).
fn exfat_timestamp(year: u32, month: u32, day: u32, hour: u32, min: u32, sec: u32) -> u32 {
    (year - 1980) << 25 | month << 21 | day << 16 | hour << 11 | min << 5 | sec / 2
}

/// The up-case table this writer stores: ASCII a-z map to A-Z, every other
/// code unit to itself (compressed with identity runs).
fn exfat_upcase_table() -> Vec<u8> {
    let mut units: Vec<u16> = vec![0xFFFF, 0x61];
    units.extend(0x41..=0x5A);
    units.extend([0xFFFF, (0x1_0000 - 0x7B) as u16]);
    units.iter().flat_map(|u| u.to_le_bytes()).collect()
}

/// Name hash of the stream extension entry (over the up-cased name).
fn exfat_name_hash(name: &str) -> u16 {
    let mut h = 0u16;
    for u in name.encode_utf16() {
        let up = if (0x61..=0x7A).contains(&u) { u - 0x20 } else { u };
        for b in up.to_le_bytes() {
            h = h.rotate_right(1).wrapping_add(b as u16);
        }
    }
    h
}

/// File, stream extension and file name entries of one file or directory.
fn exfat_entry_set(name: &str, is_dir: bool, len: u64, first: u32, contiguous: bool, ts: u32) -> Vec<u8> {
    let units: Vec<u16> = name.encode_utf16().collect();
    assert!(!units.is_empty() && units.len() <= 255, "exFAT names are 1..255 UTF-16 units");
    let names = units.len().div_ceil(15);
    let mut set = vec![0u8; 32 * (2 + names)];
    set[0] = crate::exfat::ENTRY_FILE;
    set[1] = (1 + names) as u8;
    let attrs: u16 = if is_dir { crate::exfat::ATTR_DIRECTORY } else { 0x20 };
    set[4..6].copy_from_slice(&attrs.to_le_bytes());
    for o in [8, 12, 16] {
        set[o..o + 4].copy_from_slice(&ts.to_le_bytes());
    }
    let s = &mut set[32..64];
    s[0] = crate::exfat::ENTRY_STREAM;
    s[1] = if len == 0 { 0 } else { 0x01 | if contiguous { crate::exfat::FLAG_NO_FAT_CHAIN } else { 0 } };
    s[3] = units.len() as u8;
    s[4..6].copy_from_slice(&exfat_name_hash(name).to_le_bytes());
    s[8..16].copy_from_slice(&len.to_le_bytes());
    s[20..24].copy_from_slice(&first.to_le_bytes());
    s[24..32].copy_from_slice(&len.to_le_bytes());
    for (k, chunk) in units.chunks(15).enumerate() {
        let e = &mut set[64 + k * 32..96 + k * 32];
        e[0] = crate::exfat::ENTRY_NAME;
        for (j, u) in chunk.iter().enumerate() {
            e[2 + j * 2..4 + j * 2].copy_from_slice(&u.to_le_bytes());
        }
    }
    let sum = crate::exfat::set_checksum(&set);
    set[2..4].copy_from_slice(&sum.to_le_bytes());
    set
}

/// The 12 sectors of a boot region.
fn exfat_boot_region(volume_sectors: u64, fat_len: u64, heap: u64, count: u32, root: u32, spc: u64) -> Vec<u8> {
    let sec = EXFAT_SECTOR as usize;
    let mut r = vec![0u8; sec * 12];
    r[0..3].copy_from_slice(&[0xEB, 0x76, 0x90]);
    r[3..11].copy_from_slice(crate::exfat::EXFAT_SIGNATURE);
    r[0x48..0x50].copy_from_slice(&volume_sectors.to_le_bytes());
    r[0x50..0x54].copy_from_slice(&(EXFAT_FAT_OFFSET as u32).to_le_bytes());
    r[0x54..0x58].copy_from_slice(&(fat_len as u32).to_le_bytes());
    r[0x58..0x5C].copy_from_slice(&(heap as u32).to_le_bytes());
    r[0x5C..0x60].copy_from_slice(&count.to_le_bytes());
    r[0x60..0x64].copy_from_slice(&root.to_le_bytes());
    r[0x64..0x68].copy_from_slice(&0x5053_3545u32.to_le_bytes());
    r[0x68..0x6A].copy_from_slice(&0x0100u16.to_le_bytes());
    r[0x6C] = 9;
    r[0x6D] = spc.trailing_zeros() as u8;
    r[0x6E] = 1;
    r[0x6F] = 0x80;
    r[0x70] = 0xFF;
    r[0x1FE] = 0x55;
    r[0x1FF] = 0xAA;
    // Extended boot sectors 1..8 end with their signature.
    for s in 1..9 {
        r[s * sec + sec - 4..s * sec + sec].copy_from_slice(&0xAA55_0000u32.to_le_bytes());
    }
    let sum = crate::exfat::boot_checksum(&r[..sec * 11]);
    for k in 0..sec / 4 {
        r[11 * sec + k * 4..11 * sec + k * 4 + 4].copy_from_slice(&sum.to_le_bytes());
    }
    r
}
