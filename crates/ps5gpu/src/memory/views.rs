// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest views by the backing bytes they map, to find views that share
//! backing bytes (the page tracker and the GPU caches work on guest
//! addresses: a CPU write through one view is not seen by what the caches
//! hold for another view of the same bytes).

use std::collections::BTreeMap;

const CLASSES: usize = 64;

/// Views of the backing: `(backing start, guest address) -> backing end`.
/// The entries follow the guest spans one to one (split where they split).
///
/// Views may share backing bytes, so the one before a backing offset says
/// nothing about the others that reach it. The entries are kept by length
/// class (lengths of `[2^k, 2^(k+1))`): a view of class `k` that reaches
/// an offset starts less than `2^(k+1)` below it, which bounds the search
/// in every class by the views near the offset.
pub struct BackingViews {
    classes: [BTreeMap<(u64, u64), u64>; CLASSES],
    /// Bit `k`: class `k` holds a view.
    used: u64,
}

impl Default for BackingViews {
    fn default() -> Self {
        Self { classes: std::array::from_fn(|_| BTreeMap::new()), used: 0 }
    }
}

/// Bytes two views share: `(guest address in the other view, guest address
/// in the new view, bytes)`.
pub type SharedBytes = (u64, u64, u64);

fn class_of(size: u64) -> usize {
    63 - size.max(1).leading_zeros() as usize
}

impl BackingViews {
    fn enter(&mut self, backing: u64, addr: u64, size: u64) {
        let k = class_of(size);
        self.classes[k].insert((backing, addr), backing + size);
        self.used |= 1 << k;
    }

    fn take(&mut self, backing: u64, addr: u64, size: u64) -> bool {
        // The entry has the length of its span; a class it is not in is
        // tried only when it is not where the span says.
        let (first, used) = (class_of(size), self.used);
        let mut classes = std::iter::once(first).chain((0..CLASSES).filter(|&k| k != first && used & (1 << k) != 0));
        let Some(k) = classes.find(|&k| self.classes[k].remove(&(backing, addr)).is_some()) else { return false };
        if self.classes[k].is_empty() {
            self.used &= !(1 << k);
        }
        true
    }

    /// The guest view at `addr` maps `size` backing bytes from `backing`.
    pub fn insert(&mut self, addr: u64, size: u64, backing: u64) {
        if size == 0 {
            return;
        }
        self.take(backing, addr, size);
        self.enter(backing, addr, size);
    }

    /// The guest span at `addr` (mapping `backing`, ending at guest `end`)
    /// loses `[start, stop)`: its entry is cut as the span is.
    pub fn cut(&mut self, addr: u64, end: u64, backing: u64, start: u64, stop: u64) {
        if !self.take(backing, addr, end - addr) {
            return;
        }
        if addr < start {
            self.enter(backing, addr, start - addr);
        }
        if end > stop {
            self.enter(backing + (stop - addr), stop, end - stop);
        }
    }

    /// Another view sharing backing bytes with `[backing, backing + size)`,
    /// the new view being at guest `addr` (not entered yet): of those, the
    /// first by backing start, then guest address.
    pub fn shared(&self, addr: u64, size: u64, backing: u64) -> Option<SharedBytes> {
        let end = backing + size;
        let mut best: Option<((u64, u64), u64)> = None;
        let mut used = self.used;
        while used != 0 {
            let k = used.trailing_zeros() as usize;
            used &= used - 1;
            let reach = if k == 63 { u64::MAX } else { (2u64 << k) - 1 };
            let found = self.classes[k].range((backing.saturating_sub(reach), 0)..(end, 0)).find(|(&(_, there), &e)| e > backing && there != addr);
            if let Some((&key, &e)) = found {
                if best.is_none_or(|(b, _)| key < b) {
                    best = Some((key, e));
                }
            }
        }
        best.map(|((b, there), e)| {
            let first = b.max(backing);
            (there + (first - b), addr + (first - backing), e.min(end) - first)
        })
    }

    pub fn len(&self) -> usize {
        self.classes.iter().map(BTreeMap::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.used == 0
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
