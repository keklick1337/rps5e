// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest views by the backing bytes they map, to find views that share
//! backing bytes (the page tracker and the GPU caches work on guest
//! addresses: a CPU write through one view is not seen by what the caches
//! hold for another view of the same bytes).

use std::collections::BTreeMap;

/// Views of the backing: `(backing start, guest address) -> backing end`.
/// The entries follow the guest spans one to one (split where they split).
#[derive(Default)]
pub struct BackingViews {
    views: BTreeMap<(u64, u64), u64>,
    /// Longest view ever entered (bounds the overlap search).
    longest: u64,
}

/// Bytes two views share: `(guest address in the other view, guest address
/// in the new view, bytes)`.
pub type SharedBytes = (u64, u64, u64);

impl BackingViews {
    /// The guest view at `addr` maps `size` backing bytes from `backing`.
    pub fn insert(&mut self, addr: u64, size: u64, backing: u64) {
        if size == 0 {
            return;
        }
        self.views.insert((backing, addr), backing + size);
        self.longest = self.longest.max(size);
    }

    /// The guest span at `addr` (mapping `backing`, ending at guest `end`)
    /// loses `[start, stop)`: its entry is cut as the span is.
    pub fn cut(&mut self, addr: u64, end: u64, backing: u64, start: u64, stop: u64) {
        if self.views.remove(&(backing, addr)).is_none() {
            return;
        }
        if addr < start {
            self.views.insert((backing, addr), backing + (start - addr));
        }
        if end > stop {
            self.views.insert((backing + (stop - addr), stop), backing + (end - addr));
        }
    }

    /// Another view sharing backing bytes with `[backing, backing + size)`,
    /// the new view being at guest `addr` (not entered yet).
    pub fn shared(&self, addr: u64, size: u64, backing: u64) -> Option<SharedBytes> {
        let low = backing.saturating_sub(self.longest);
        let end = backing + size;
        self.views.range((low, 0)..(end, 0)).find_map(|(&(b, there), &e)| {
            (e > backing && there != addr).then(|| {
                let first = b.max(backing);
                (there + (first - b), addr + (first - backing), e.min(end) - first)
            })
        })
    }

    pub fn len(&self) -> usize {
        self.views.len()
    }

    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::BackingViews;

    #[test]
    fn views_sharing_backing_bytes_are_found() {
        let mut v = BackingViews::default();
        v.insert(0x10_0000, 0x4000, 0x8000);
        // Disjoint backing: nothing shared.
        assert_eq!(v.shared(0x20_0000, 0x4000, 0x0), None);
        assert_eq!(v.shared(0x20_0000, 0x4000, 0xC000), None);
        // A second view of the same bytes, partly.
        assert_eq!(v.shared(0x20_0000, 0x4000, 0xA000), Some((0x10_2000, 0x20_0000, 0x2000)));
        // A view over the start of the first one.
        assert_eq!(v.shared(0x30_0000, 0x4000, 0x6000), Some((0x10_0000, 0x30_2000, 0x2000)));
    }

    #[test]
    fn a_cut_view_no_longer_shares_its_removed_part() {
        let mut v = BackingViews::default();
        v.insert(0x10_0000, 0x8000, 0x8000);
        // The guest unmaps the middle of the view.
        v.cut(0x10_0000, 0x10_8000, 0x8000, 0x10_2000, 0x10_6000);
        assert_eq!(v.len(), 2);
        assert_eq!(v.shared(0x20_0000, 0x4000, 0xA000), None);
        assert_eq!(v.shared(0x20_0000, 0x1000, 0x9000), Some((0x10_1000, 0x20_0000, 0x1000)));
        assert_eq!(v.shared(0x20_0000, 0x1000, 0xE000), Some((0x10_6000, 0x20_0000, 0x1000)));
        // The whole view goes.
        v.cut(0x10_0000, 0x10_2000, 0x8000, 0x10_0000, 0x10_2000);
        v.cut(0x10_6000, 0x10_8000, 0xE000, 0x10_6000, 0x10_8000);
        assert!(v.is_empty());
    }
}
