// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Minimal writers for PFS images and fake PKG containers.
//!
//! They exist so tests (here and in the file-system HLE) can build small
//! images in memory; they follow the layouts documented in [`crate::pfs`]
//! and [`crate::pkg`], not any packaging tool's exact block placement.

use crate::crypto::{derive_key, key_digest, pfs_xts_keys, Xts, XTS_SECTOR_SIZE};
use crate::pfs::{
    InodeLayout, DIRENT_DIR, DIRENT_DOT, DIRENT_DOTDOT, DIRENT_FILE, INODE_FLAG_COMPRESSED, INODE_MODE_DIR,
    INODE_MODE_FILE, MODE_ENCRYPTED, MODE_SIGNED, PFS_MAGIC, PFS_VERSION,
};
use crate::pfsc::build_pfsc;
use crate::pkg;
use std::collections::BTreeMap;

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
    files: Vec<(String, Vec<u8>, bool)>,
    dirs: Vec<String>,
}

impl Default for PfsBuilder {
    fn default() -> Self {
        Self { block_size: 0x1000, signed: false, scatter: false, encrypt: None, files: Vec::new(), dirs: Vec::new() }
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
        let bs = self.block_size;
        let signed = self.signed || self.scatter;
        let layout = if signed { InodeLayout::S32 } else { InodeLayout::D32 };
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
        let mut next = 1 + inode_blocks;
        let mut plans: Vec<Plan> = Vec::new();
        let mut indirect_needed: Vec<usize> = Vec::new();
        for (idx, (rec, data)) in contents.iter().enumerate() {
            let n = (data.len() as u64).div_ceil(bs).max(if rec.mode & INODE_MODE_DIR != 0 { 1 } else { 0 });
            let mut blocks: Vec<u64> = (next..next + n).collect();
            next += n;
            if self.scatter && n > 1 && rec.mode & INODE_MODE_FILE != 0 {
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
        let total_blocks = next;
        let mut img = vec![0u8; (total_blocks * bs) as usize];

        // Superblock.
        let mut mode: u16 = 0x8;
        if signed {
            mode |= MODE_SIGNED;
        }
        if self.encrypt.is_some() {
            mode |= MODE_ENCRYPTED;
        }
        let put = |img: &mut [u8], off: usize, b: &[u8]| img[off..off + b.len()].copy_from_slice(b);
        put(&mut img, 0, &PFS_VERSION.to_le_bytes());
        put(&mut img, 8, &PFS_MAGIC.to_le_bytes());
        put(&mut img, 0x1C, &mode.to_le_bytes());
        put(&mut img, 0x20, &(bs as u32).to_le_bytes());
        put(&mut img, 0x28, &1u64.to_le_bytes());
        put(&mut img, 0x30, &inode_count.to_le_bytes());
        put(&mut img, 0x38, &total_blocks.to_le_bytes());
        put(&mut img, 0x40, &inode_blocks.to_le_bytes());
        put(&mut img, 0x48, &0u64.to_le_bytes());
        // Inode-block descriptor (signed 64-bit layout) at 0x50.
        let d = 0x50;
        put(&mut img, d + 8, &(inode_blocks * bs).to_le_bytes());
        put(&mut img, d + 0x10, &(inode_blocks * bs).to_le_bytes());
        put(&mut img, d + 0x60, &inode_blocks.to_le_bytes());
        for i in 0..inode_blocks.min(12) {
            put(&mut img, d + 0x68 + i as usize * 40 + 32, &(1 + i).to_le_bytes());
        }
        if let Some((_, _, seed)) = &self.encrypt {
            put(&mut img, 0x370, seed);
        }

        // Inodes.
        let per_block = bs / isz;
        let ps = layout.ptr_size();
        let base = 0x64usize;
        let vo = if signed { 32 } else { 0 };
        for (i, p) in plans.iter().enumerate() {
            let blk = 1 + i as u64 / per_block;
            let off = (blk * bs + (i as u64 % per_block) * isz) as usize;
            let r = &p.inode;
            put(&mut img, off, &r.mode.to_le_bytes());
            put(&mut img, off + 2, &r.nlink.to_le_bytes());
            put(&mut img, off + 4, &r.flags.to_le_bytes());
            put(&mut img, off + 8, &r.size.to_le_bytes());
            put(&mut img, off + 0x10, &r.size_compressed.to_le_bytes());
            put(&mut img, off + 0x20, &1_600_000_000i64.to_le_bytes());
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
        img
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
    let tree = b.tree();
    let idx = tree[0].children[name];
    let ino = 2 + idx as u64;
    let layout = if b.signed || b.scatter { InodeLayout::S32 } else { InodeLayout::D32 };
    let isz = layout.size() as u64;
    let per = b.block_size / isz;
    let off = ((1 + ino / per) * b.block_size + (ino % per) * isz) as usize;
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
