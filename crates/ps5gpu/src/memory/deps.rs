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
    /// What the result was derived from, read by read (see
    /// [`Self::revalidate`]); None when not recorded or too many.
    contents: Option<smallvec::SmallVec<[Content; 8]>>,
}

/// One input of a result: bytes read from guest memory (their hash), or
/// whether a range held GPU results then.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Content {
    Bytes { addr: u64, len: u32, hash: u64 },
    GpuDirty { addr: u64, len: u32, dirty: bool },
}

/// Inputs recorded at most per result.
const MAX_CONTENTS: usize = 128;
/// Bytes hashed at most per result.
const MAX_CONTENT_BYTES: usize = 16 << 10;

fn hash_bytes(b: &[u8]) -> u64 {
    // FNV-1a over 8-byte words: the inputs are a few descriptor words.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ b.len() as u64;
    let mut chunks = b.chunks_exact(8);
    for c in &mut chunks {
        h = (h ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(0x0100_0000_01b3);
    }
    for &x in chunks.remainder() {
        h = (h ^ x as u64).wrapping_mul(0x0100_0000_01b3);
    }
    h
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

    /// Record the inputs of the result too, not only their pages (see
    /// [`Self::revalidate`]).
    pub fn record_contents(&mut self) {
        self.contents = Some(smallvec::SmallVec::new());
    }

    fn push_content(&mut self, c: Content) {
        if let Some(list) = &mut self.contents {
            let bytes: usize = list.iter().map(|c| if let Content::Bytes { len, .. } = c { *len as usize } else { 0 }).sum();
            if list.len() >= MAX_CONTENTS || bytes > MAX_CONTENT_BYTES {
                self.contents = None;
            } else if list.last() != Some(&c) {
                list.push(c);
            }
        }
    }

    /// The result used `bytes`, read at `addr` (after [`Self::note`]).
    pub fn read(&mut self, addr: u64, bytes: &[u8]) {
        if self.contents.is_some() {
            self.push_content(Content::Bytes { addr, len: bytes.len() as u32, hash: hash_bytes(bytes) });
        }
    }

    /// The result depended on whether `[addr, addr+len)` held GPU results.
    pub fn gpu_state(&mut self, addr: u64, len: u64, dirty: bool) {
        if self.contents.is_some() {
            self.push_content(Content::GpuDirty { addr, len: len as u32, dirty });
        }
    }

    /// The result still stands although a page it read may have changed
    /// ([`Self::current`] failed): its pages are mapped as before, every
    /// input reads the same bytes (and no GPU results are pending over
    /// them) and every GPU-state input is unchanged. The materialisation is
    /// a function of its program, its user data and these inputs, so it
    /// would give the same result now. On success the pages' generations
    /// and the synchronisation point are taken anew (before the bytes are
    /// compared), so [`Self::current`] holds again until they move.
    pub fn revalidate(&mut self, tracker: &GuestPageTracker, memory: &GuestGpuMemory) -> bool {
        let Some(contents) = &self.contents else { return false };
        if self.overflow || self.maps != memory.map_generation() {
            return false;
        }
        // Generations first: a write after them moves them.
        let sync = super::sync_generation();
        let mut block = None;
        let mut volatile = false;
        let gens: smallvec::SmallVec<[Option<u32>; 8]> = self
            .pages
            .iter()
            .map(|&(p, _)| {
                let g = tracker.page_generation_in(&mut block, p);
                volatile = volatile || g.is_none() || tracker.page_maybe_cpu_dirty(p);
                g
            })
            .collect();
        let mut buf: smallvec::SmallVec<[u8; 256]> = smallvec::SmallVec::new();
        for c in contents {
            match *c {
                Content::Bytes { addr, len, hash } => {
                    if tracker.has_gpu_dirty_pages(addr, len as u64) {
                        return false;
                    }
                    buf.clear();
                    buf.resize(len as usize, 0);
                    if !memory.read_backing(addr, &mut buf) || hash_bytes(&buf) != hash {
                        return false;
                    }
                }
                Content::GpuDirty { addr, len, dirty } => {
                    if tracker.has_gpu_dirty_pages(addr, len as u64) != dirty {
                        return false;
                    }
                }
            }
        }
        for (p, g) in self.pages.iter_mut().zip(gens) {
            p.1 = g;
        }
        self.volatile = volatile;
        self.sync = sync;
        true
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
