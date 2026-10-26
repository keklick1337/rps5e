// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Mount slots: up to 16 save directories mounted at `/savedata0..15`.

use std::path::{Path, PathBuf};

pub(super) const COUNT: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Available {
    /// First free slot.
    Slot(usize),
    /// The directory is already mounted.
    Busy,
    /// Every slot is in use.
    Full,
}

pub(super) struct MountSlots {
    dirs: [Option<PathBuf>; COUNT],
}

impl MountSlots {
    pub const fn new() -> Self {
        Self { dirs: [const { None }; COUNT] }
    }

    /// `Busy` if `directory` (or a path naming the same directory) is
    /// mounted, else the first free slot, else `Full`.
    pub fn find_available(&self, directory: &Path) -> Available {
        let mut available = Available::Full;
        for (index, mounted) in self.dirs.iter().enumerate() {
            match mounted {
                Some(m) if m == directory || equivalent(m, directory) => return Available::Busy,
                None if available == Available::Full => available = Available::Slot(index),
                _ => {}
            }
        }
        available
    }

    pub fn mount(&mut self, slot: usize, directory: &Path) {
        self.dirs[slot] = Some(directory.to_path_buf());
    }

    pub fn directory(&self, slot: usize) -> &Path {
        self.dirs[slot].as_deref().expect("mounted slot")
    }

    pub fn release(&mut self, slot: usize) {
        if let Some(d) = self.dirs.get_mut(slot) {
            *d = None;
        }
    }

    /// Slot mounted at `mount_point`.
    pub fn find(&self, mount_point: &str) -> Option<usize> {
        (0..COUNT).find(|&i| self.dirs[i].is_some() && Self::mount_point(i) == mount_point)
    }

    pub fn mount_point(slot: usize) -> String {
        format!("/savedata{slot}")
    }

    pub fn is_empty(&self) -> bool {
        self.dirs.iter().all(Option::is_none)
    }

    pub fn count(&self) -> usize {
        self.dirs.iter().filter(|d| d.is_some()).count()
    }

    pub fn mounted(&self) -> impl Iterator<Item = (usize, &Path)> {
        self.dirs.iter().enumerate().filter_map(|(i, d)| d.as_deref().map(|d| (i, d)))
    }
}

/// Both paths exist and name the same file system object.
fn equivalent(a: &Path, b: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match (std::fs::metadata(a), std::fs::metadata(b)) {
            (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
            _ => false,
        }
    }
    #[cfg(not(unix))]
    {
        match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        }
    }
}
