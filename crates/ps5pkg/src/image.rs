// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Disk images of game dumps: recognising an exFAT or UFS2 file system in
//! a byte source, opening it as a [`Pfs`] tree and re-rooting that tree at
//! the game folder (the image may keep the game in a subfolder).

use crate::exfat::{self, ExFat};
use crate::io::{Slice, Source};
use crate::pfs::{NodeId, Pfs};
use crate::ufs::{self, Ufs};
use crate::Error;
use std::sync::Arc;

/// A file system found in a disk image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiskKind {
    /// exFAT volume starting at this byte offset (0, or a partition).
    ExFat(u64),
    Ufs2,
}

/// Recognise a disk image.
pub fn detect(src: &Source) -> Option<DiskKind> {
    if let Some(off) = exfat::find_volume(&**src) {
        return Some(DiskKind::ExFat(off));
    }
    ufs::is_ufs2(&**src).then_some(DiskKind::Ufs2)
}

/// Open the file system of a disk image, rooted at its game folder when
/// one is found.
pub fn open(src: Source, kind: DiskKind) -> Result<Pfs, Error> {
    let tree = match kind {
        DiskKind::ExFat(0) => ExFat::open(src)?.into_tree()?,
        DiskKind::ExFat(off) => {
            let len = src.len() - off;
            ExFat::open(Arc::new(Slice::new(src, off, len)?))?.into_tree()?
        }
        DiskKind::Ufs2 => Ufs::open(src)?.into_tree()?,
    };
    Ok(match find_game_root(&tree) {
        Some(dir) => tree.subtree(dir),
        None => tree,
    })
}

/// The game folder of a tree: the shallowest directory holding `eboot.bin`
/// and `sce_sys/param.json` (failing that, `sce_sys/param.sfo`). Among
/// folders at the same depth the first in name order wins.
pub fn find_game_root(pfs: &Pfs) -> Option<NodeId> {
    let is_root = |dir: NodeId, param: &str| {
        let eboot = pfs.child(dir, "eboot.bin").is_some_and(|e| !pfs.node(e).is_dir);
        let sys = pfs.child(dir, "sce_sys").filter(|&s| pfs.node(s).is_dir);
        eboot && sys.and_then(|s| pfs.child(s, param)).is_some_and(|p| !pfs.node(p).is_dir)
    };
    for param in ["param.json", "param.sfo"] {
        let mut level = vec![pfs.root()];
        while !level.is_empty() {
            if let Some(&d) = level.iter().find(|&&d| is_root(d, param)) {
                return Some(d);
            }
            level = level
                .iter()
                .flat_map(|&d| pfs.node(d).children.iter().copied())
                .filter(|&c| pfs.node(c).is_dir)
                .collect();
        }
    }
    None
}
