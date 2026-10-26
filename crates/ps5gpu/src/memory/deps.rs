// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! [`PageDeps`]: the guest pages a result was derived from, with what it
//! takes to know their bytes are still the ones read.

use super::tracker::GuestPageTracker;
use super::{GuestGpuMemory, PAGE_BYTES};

/// Pages remembered at most (a result reading more is not memoised).
const MAX_PAGES: usize = 64;

/// The guest pages a result read, each with its content generation (see
/// [`super::region::PageGenerations`]) taken before the read. The bytes
/// read are still the bytes memory holds while every page's generation is
/// unchanged, no mapping came or went and, when a page could be written
/// without a fault (CPU-dirty, or untracked), no CPU synchronisation point
/// passed (the CPU writes data the GPU reads before the synchronisation
/// point of the commands reading it, as for every CPU-dirty page).
#[derive(Clone, Default)]
pub struct PageDeps {
    pages: smallvec::SmallVec<[(u64, Option<u32>); 8]>,
    volatile: bool,
    overflow: bool,
    sync: u64,
    maps: u64,
    /// Pages kept at most (0: [`MAX_PAGES`]).
    limit: u16,
}

impl PageDeps {
    /// Start recording the reads of a result.
    pub fn begin(memory: &GuestGpuMemory) -> PageDeps {
        PageDeps { sync: super::sync_generation(), maps: memory.map_generation(), ..PageDeps::default() }
    }

    /// [`Self::begin`] keeping up to `limit` pages (a slice thread's read
    /// set, R7b: a result read from many pages is still checked at commit).
    pub fn begin_with_limit(memory: &GuestGpuMemory, limit: u16) -> PageDeps {
        PageDeps { limit, ..Self::begin(memory) }
    }

    /// `[addr, addr+size)` is about to be read.
    pub fn note(&mut self, tracker: &GuestPageTracker, addr: u64, size: u64) {
        if self.overflow {
            return;
        }
        let first = addr & !(PAGE_BYTES - 1);
        let end = addr.saturating_add(size.max(1));
        let mut page = first;
        let mut block = None;
        while page < end {
            // Most reads fall in the page read last.
            if self.pages.last().is_none_or(|&(p, _)| p != page) && !self.pages.iter().any(|&(p, _)| p == page) {
                if self.pages.len() >= if self.limit == 0 { MAX_PAGES } else { self.limit as usize } {
                    self.overflow = true;
                    return;
                }
                // The generation first: a write after it moves it.
                let generation = tracker.page_generation_in(&mut block, page);
                // Once volatile, the page's own dirty state changes nothing
                // (it takes the region lock).
                self.volatile = self.volatile || generation.is_none() || tracker.page_maybe_cpu_dirty(page);
                self.pages.push((page, generation));
            }
            page += PAGE_BYTES;
        }
    }

    /// The pages read still hold the bytes read.
    pub fn current(&self, tracker: &GuestPageTracker, memory: &GuestGpuMemory) -> bool {
        !self.overflow
            && self.maps == memory.map_generation()
            && (!self.volatile || self.sync == super::sync_generation())
            && {
                let mut block = None;
                self.pages.iter().all(|&(p, g)| tracker.page_generation_in(&mut block, p) == g)
            }
    }

    /// The pages recorded (their addresses).
    pub fn page_list(&self) -> impl Iterator<Item = u64> + '_ {
        self.pages.iter().map(|&(p, _)| p)
    }

    /// Pages recorded (diagnostics, tests).
    pub fn pages(&self) -> usize {
        self.pages.len()
    }

    /// Too many pages to remember (the result is never memoised).
    pub fn overflowed(&self) -> bool {
        self.overflow
    }

    /// Some page could be written without a fault.
    pub fn volatile(&self) -> bool {
        self.volatile
    }
}
