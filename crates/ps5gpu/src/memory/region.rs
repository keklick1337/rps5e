// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Dirty state of one 4 MiB block: every page is CPU-dirty, GPU-dirty or
//! clean, never both dirty.
//!
//! * CPU-dirty pages are writable; clean pages are write-watched, so the
//!   first CPU write faults and makes them CPU-dirty again.
//! * GPU-dirty pages are read-watched (inaccessible) until the GPU data is
//!   downloaded, except *read-through* pages: their guest memory holds the
//!   GPU contents as of an earlier point and every GPU write to them is
//!   copied into guest memory after it (write-through), so the CPU reads
//!   them like the GPU's memory on the console (the last completed write).
//!   A label the command processor publishes ends read-through for pages
//!   whose copy has not landed yet, so a read synchronised with the label
//!   faults and sees the new data (`BufferCache::publish_point`).
//!   CPU writes to them still fault.
//! * A GPU-dirty page whose current contents are being copied into guest
//!   memory (write-through, settling, a download into the guest pages'
//!   import) carries the tick of that copy (*landing*): once the tick
//!   completed, a CPU access makes it clean from the fault handler without
//!   the buffer cache. A GPU write of the page forgets the tick.
//!
//! Every upload write-watches the pages it cleaned, so a page is uploaded
//! again only after the CPU wrote it since.
//!
//! Pages the CPU rewrites between uses again and again (constant and
//! dynamic vertex rings) become *hot*: they stay CPU-dirty and unwatched,
//! so they cost neither a write fault nor two protection changes per use.
//! The first use after a CPU synchronisation point
//! ([`super::sync_generation`]) compares a hot page with the copy last
//! uploaded and uploads it only when the bytes changed. A GPU write cools it, and so does a run of
//! [`HOT_IDLE_USES`] uses that found it unchanged.
//!
//! Hot pages are off unless `RPS5E_HOT_PAGES=1`: the guest's CPU threads
//! keep writing these pages while the command processor works through a
//! submission (SILENT HILL: ~1.5M changes of already-compared hot pages
//! inside one synchronisation interval in 4 minutes), so the copy taken at
//! the first use of an interval handed later draws stale instance and
//! constant data. Watched pages fault on every such write and the next use
//! uploads the bytes current at that point, as the GPU reads memory when it
//! executes the draw.

use super::guard::PageGuard;
use super::lock::RegionLock;
use super::mask::PageMask;
use super::{WriteOrigin, BLOCK_BYTES, PAGES_PER_BLOCK, PAGE_BYTES};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

/// Per page of a block: bumped whenever the page's guest bytes may change
/// without a fault the tracker would still see: the page became CPU-dirty
/// (writable), or its GPU-dirty state changed (the GPU writes it, or its
/// GPU contents reached guest memory). Read without the region lock.
pub type PageGenerations = Arc<Generations>;

/// Pages per group counter of [`Generations`].
const GROUP_PAGES: usize = 64;

/// The content generations of a block's pages. A page's generation is its
/// own counter plus the counter of its group of [`GROUP_PAGES`] pages, so
/// a change of a whole group moves one counter instead of 64 (large GPU
/// writes cover hundreds of pages per use). Both counters only grow while
/// the region lock is held, so the sum moves with every change; a reader
/// that meets a change half done sees a value that differs from both the
/// one before and (or equals) the one after, never an old value again.
pub struct Generations {
    pages: [AtomicU32; PAGES_PER_BLOCK],
    groups: [AtomicU32; PAGES_PER_BLOCK / GROUP_PAGES],
}

impl Generations {
    fn new() -> Generations {
        Generations { pages: [const { AtomicU32::new(0) }; PAGES_PER_BLOCK], groups: [const { AtomicU32::new(0) }; PAGES_PER_BLOCK / GROUP_PAGES] }
    }

    /// The generation of page `p` of the block. Lock-free.
    #[inline]
    pub fn get(&self, p: usize) -> u32 {
        self.pages[p].load(Ordering::Acquire).wrapping_add(self.groups[p / GROUP_PAGES].load(Ordering::Acquire))
    }

    /// Every caller holds the region lock, the only way to move them, so
    /// a plain load and store does (a locked add per page made marking a
    /// large GPU write cost hundreds of microseconds).
    #[inline]
    fn step(g: &AtomicU32) {
        g.store(g.load(Ordering::Relaxed).wrapping_add(1), Ordering::Release);
    }

    /// Move the generations of pages `[s, e)`: whole groups by their group
    /// counter (`RPS5E_PAGE_GROUPS=0`: page by page).
    fn bump_range(&self, s: usize, e: usize) {
        self.bump_range_with(s, e, group_bumps_enabled());
    }

    fn bump_range_with(&self, s: usize, e: usize, groups: bool) {
        let mut p = s;
        while p < e {
            if groups && p % GROUP_PAGES == 0 && p + GROUP_PAGES <= e {
                Self::step(&self.groups[p / GROUP_PAGES]);
                p += GROUP_PAGES;
            } else {
                Self::step(&self.pages[p]);
                p += 1;
            }
        }
    }
}

/// `RPS5E_PAGE_GROUPS=0`: page generations move page by page (A/B scope
/// `release` toggles it; both ways move every page's sum).
fn group_bumps_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_PAGE_GROUPS")) && crate::backend::ab::release_fast("RPS5E_PAGE_GROUPS")
}

/// One bit per block: set while the block has a CPU-dirty page.
pub struct CpuDirtySummary {
    words: Box<[AtomicU64]>,
    epoch: AtomicU64,
}

impl CpuDirtySummary {
    pub fn new(blocks: usize) -> Self {
        Self { words: super::zeroed_slice(blocks.div_ceil(64)), epoch: AtomicU64::new(0) }
    }

    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    pub fn note_dirtied(&self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    pub fn set(&self, block: u64, dirty: bool) {
        let bit = 1u64 << (block % 64);
        let w = &self.words[(block / 64) as usize];
        if dirty {
            w.fetch_or(bit, Ordering::AcqRel);
        } else {
            w.fetch_and(!bit, Ordering::AcqRel);
        }
    }

    pub fn is_dirty(&self, block: u64) -> bool {
        self.words[(block / 64) as usize].load(Ordering::Acquire) & (1 << (block % 64)) != 0
    }

    pub fn word(&self, index: usize) -> u64 {
        self.words[index].load(Ordering::Acquire)
    }
}

/// Pages of a range that were not handled (diagnostics).
pub static STATE_CONFLICTS: AtomicU64 = AtomicU64::new(0);

/// GPU-dirty pages a CPU access found already copied into guest memory
/// (made clean without the back-end, see [`RegionState::take_landed`]).
pub static LANDED_PAGES: AtomicU64 = AtomicU64::new(0);

/// `RPS5E_HOT_PAGES=1`: pages rewritten between uses turn hot (see the
/// module notes; off by default, every page stays write-watched).
pub fn hot_pages_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::backend::ab::safe() && std::env::var("RPS5E_HOT_PAGES").is_ok_and(|v| v == "1"))
}

/// Uploads of CPU-written contents after which a page becomes hot.
pub const HOT_UPLOADS: u8 = 4;
/// Uses finding a hot page unchanged after which it is watched again.
pub const HOT_IDLE_USES: u8 = 64;

/// `RPS5E_FAULT_RUNS=1` (opt-in; A/B name `RPS5E_FAULT_RUNS` in scope
/// `release`): a CPU write fault on a page the CPU has rewritten
/// between uploads at least [`HOT_UPLOADS`] times makes the run of such
/// pages around it CPU-dirty and writable in one protection change
/// ([`RegionState::fault_run`]), so a constant or vertex ring the guest
/// rewrites every frame costs one fault per run instead of one per page,
/// in whatever order its pages are written. Exact: an opened page is
/// CPU-dirty, so its next use uploads it and watches it again, as for a
/// page the guest wrote; one the guest did not write costs one upload of
/// unchanged bytes. GPU-dirty pages end a run. Opt-in: measured slower in
/// SILENT HILL 2 and GTA V (write faults -40..-75 %, but the per-draw
/// memos over opened pages the guest did not write miss ~10x as often,
/// and the command thread pays more than the faults saved).
pub fn fault_runs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::backend::ab::safe() && std::env::var("RPS5E_FAULT_RUNS").is_ok_and(|v| v == "1"))
}

/// Pages a fault run spans at most (the faulting page included).
pub const FAULT_RUN_PAGES: usize = 32;

/// Diagnostics: write faults on pages rewritten at least [`HOT_UPLOADS`]
/// times (the faults a run could have saved), faults that opened a run,
/// and pages the runs opened.
pub static HEATED_WRITE_FAULTS: AtomicU64 = AtomicU64::new(0);
pub static FAULT_RUNS: AtomicU64 = AtomicU64::new(0);
pub static FAULT_RUN_OPENED: AtomicU64 = AtomicU64::new(0);

/// Diagnostics: hot pages now, and hot page uses that compared / found
/// changed bytes.
pub static HOT_PAGES: AtomicU64 = AtomicU64::new(0);
pub static HOT_COMPARES: AtomicU64 = AtomicU64::new(0);
pub static HOT_CHANGED: AtomicU64 = AtomicU64::new(0);

/// Bytes of the last upload of each hot page (by page index).
type Shadow = Box<[u8; PAGE_BYTES as usize]>;

pub struct RegionState {
    base: u64,
    cpu_dirty: PageMask,
    gpu_dirty: PageMask,
    /// GPU-dirty pages readable by the CPU (subset of `gpu_dirty`).
    read_through: PageMask,
    writable: PageMask,
    readable: PageMask,
    /// Hot pages (always CPU-dirty and writable).
    hot: PageMask,
    /// Uploads of CPU-written contents per page (saturating). A GPU write
    /// of the page resets it.
    heat: [u8; PAGES_PER_BLOCK],
    /// Pages a fault run made CPU-dirty that were not uploaded since: their
    /// next upload does not count as a rewrite.
    opened: PageMask,
    /// Consecutive uses that found a hot page unchanged.
    idle: [u8; PAGES_PER_BLOCK],
    shadows: std::collections::HashMap<u16, Shadow>,
    /// Hot pages compared in generation `checked_generation`.
    checked: PageMask,
    checked_generation: u64,
    /// Per GPU-dirty page: the timeline tick whose completion lands the
    /// page's current GPU contents in guest memory (a copy into guest
    /// memory recorded after the page's last GPU write), 0 for none.
    /// Allocated by the first copy recorded in the block.
    landing: Option<Box<[u64; PAGES_PER_BLOCK]>>,
    /// See [`PageGenerations`] (shared with the region).
    generations: PageGenerations,
}

/// What [`RegionState::take_landed`] found for a page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Landing {
    /// The page is not GPU-dirty.
    Clean,
    /// Its copy completed: the page is clean now.
    Landed,
    /// Its copy completes with this tick.
    InFlight(u64),
    /// No copy of its current contents is recorded.
    Missing,
}

pub struct TrackedRegion {
    pub lock: RegionLock,
    pub base: u64,
    state: UnsafeCell<RegionState>,
    /// The pages' content generations (lock-free reads).
    pub generations: PageGenerations,
}

// SAFETY: `state` is only accessed with `lock` held.
unsafe impl Sync for TrackedRegion {}
unsafe impl Send for TrackedRegion {}

/// Context passed to state changes: the page guard, the summary, and
/// whether we run inside the fault handler.
pub struct Ctx<'a> {
    pub pages: &'a PageGuard,
    pub summary: &'a CpuDirtySummary,
    pub fault: bool,
}

impl TrackedRegion {
    pub fn new(base: u64, summary: &CpuDirtySummary) -> Self {
        debug_assert_eq!(base % BLOCK_BYTES, 0);
        summary.set(base / BLOCK_BYTES, true);
        summary.note_dirtied();
        let generations: PageGenerations = Arc::new(Generations::new());
        Self {
            lock: RegionLock::new(),
            base,
            generations: generations.clone(),
            state: UnsafeCell::new(RegionState {
                base,
                cpu_dirty: PageMask::full(),
                gpu_dirty: PageMask::empty(),
                read_through: PageMask::empty(),
                writable: PageMask::full(),
                readable: PageMask::full(),
                hot: PageMask::empty(),
                heat: [0; PAGES_PER_BLOCK],
                opened: PageMask::empty(),
                idle: [0; PAGES_PER_BLOCK],
                shadows: std::collections::HashMap::new(),
                checked: PageMask::empty(),
                checked_generation: 0,
                landing: None,
                generations,
            }),
        }
    }

    /// The state. The caller must hold `self.lock`.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn state(&self) -> &mut RegionState {
        &mut *self.state.get()
    }
}

impl RegionState {
    fn pages(&self, addr: u64, size: u64) -> (usize, usize) {
        let off = addr.saturating_sub(self.base);
        let end = (off + size).min(BLOCK_BYTES);
        ((off / PAGE_BYTES) as usize, end.div_ceil(PAGE_BYTES) as usize)
    }

    /// The guest bytes of `pages` may change: their generations move (the
    /// caller holds the region lock).
    fn bump(&self, pages: &PageMask) {
        for (s, e) in pages.runs() {
            self.generations.bump_range(s, e);
        }
    }

    /// Host code wrote the guest bytes of the range (without a fault).
    pub fn note_written(&self, addr: u64, size: u64) {
        let (s, e) = self.pages(addr, size);
        let mut range = PageMask::empty();
        range.set_range(s, e);
        self.bump(&range);
    }

    fn mask(&self, side: WriteOrigin) -> &PageMask {
        match side {
            WriteOrigin::Cpu => &self.cpu_dirty,
            WriteOrigin::Gpu => &self.gpu_dirty,
        }
    }

    /// Any page of `[base+offset, +size)` dirty on `side`.
    pub fn is_modified(&self, side: WriteOrigin, offset: u64, size: u64) -> bool {
        let (s, e) = self.pages(self.base + offset, size);
        self.mask(side).any_in_range(s, e)
    }

    /// A CPU write: the pages become CPU-dirty (and writable).
    pub fn mark_cpu_write(&mut self, ctx: &Ctx, addr: u64, size: u64) -> bool {
        self.change_state(ctx, WriteOrigin::Cpu, true, addr, size)
    }

    /// The range a CPU write fault at `[addr, +size)` (one block) makes
    /// CPU-dirty with [`fault_runs_enabled`]: the range itself, grown over
    /// the following pages the CPU rewrote between uploads at least
    /// [`HOT_UPLOADS`] times that are not GPU-dirty, up to
    /// [`FAULT_RUN_PAGES`] pages, when the faulting page is such a page.
    /// Only forward: the pages before it in a ring were written and
    /// uploaded already in this pass, and each opened page that is not
    /// written makes the per-draw memos over it miss.
    /// Allocation-free (fault handler). The pages it adds that were clean
    /// are remembered as opened.
    pub fn fault_run(&mut self, addr: u64, size: u64) -> (u64, u64) {
        let (s, e) = self.pages(addr, size);
        if s >= e {
            return (addr, size);
        }
        let heated = |st: &Self, p: usize| st.heat[p] >= HOT_UPLOADS && !st.gpu_dirty.get(p);
        if self.heat[s] < HOT_UPLOADS {
            return (addr, size);
        }
        HEATED_WRITE_FAULTS.fetch_add(1, Ordering::Relaxed);
        let (rs, mut re) = (s, e);
        while re < PAGES_PER_BLOCK && re - rs < FAULT_RUN_PAGES && heated(self, re) {
            re += 1;
        }
        if (rs, re) == (s, e) {
            return (addr, size);
        }
        let mut added = PageMask::empty();
        added.set_range(rs, s);
        added.set_range(e, re);
        let added = added.and(&self.cpu_dirty.not());
        if added.any() {
            FAULT_RUNS.fetch_add(1, Ordering::Relaxed);
            FAULT_RUN_OPENED.fetch_add(added.count() as u64, Ordering::Relaxed);
            self.opened = self.opened.or(&added);
        }
        (self.base + rs as u64 * PAGE_BYTES, (re - rs) as u64 * PAGE_BYTES)
    }

    /// Set or clear `side`'s dirty bits over the range and update the page
    /// protection. Pages dirty on the other side are left alone (counted in
    /// [`STATE_CONFLICTS`]). Returns false when the fault-path protection
    /// update could not take its lock.
    pub fn change_state(&mut self, ctx: &Ctx, side: WriteOrigin, enable: bool, addr: u64, size: u64) -> bool {
        let (s, e) = self.pages(addr, size);
        let mut range = PageMask::empty();
        range.set_range(s, e);
        if enable {
            let other = match side {
                WriteOrigin::Cpu => self.gpu_dirty,
                WriteOrigin::Gpu => self.cpu_dirty,
            };
            let conflicts = range.and(&other);
            if conflicts.any() {
                STATE_CONFLICTS.fetch_add(1, Ordering::Relaxed);
                range = range.and(&conflicts.not());
            }
        }
        if side == WriteOrigin::Gpu {
            if enable {
                self.cool_written(&range);
            }
            // New GPU contents (or none): earlier copies do not describe them.
            self.clear_landing(s, e);
            // Written by the GPU now, or holding its contents in guest
            // memory now.
            self.bump(&if enable { range } else { range.and(&self.gpu_dirty) });
        }
        let bits = match side {
            WriteOrigin::Cpu => &mut self.cpu_dirty,
            WriteOrigin::Gpu => &mut self.gpu_dirty,
        };
        if enable {
            *bits = bits.or(&range);
            if side == WriteOrigin::Gpu {
                self.read_through = self.read_through.and(&range.not());
            }
        } else {
            *bits = bits.and(&range.not());
            self.read_through = self.read_through.and(&self.gpu_dirty);
        }
        match side {
            WriteOrigin::Cpu => self.update_cpu_protection(ctx),
            WriteOrigin::Gpu => self.update_gpu_protection(ctx),
        }
    }

    /// The GPU writes the range: the pages become GPU-dirty. Pages for which
    /// `through` holds (their GPU writes are written through to guest
    /// memory) stay CPU-readable when they were readable before, i.e. guest
    /// memory holds a coherent earlier state of them.
    pub fn mark_gpu_written(&mut self, ctx: &Ctx, addr: u64, size: u64, through: &dyn Fn(u64) -> bool) -> bool {
        let (s, e) = self.pages(addr, size);
        let readable_before = self.gpu_dirty.not().or(&self.read_through);
        let mut range = PageMask::empty();
        range.set_range(s, e);
        let conflicts = range.and(&self.cpu_dirty);
        if conflicts.any() {
            STATE_CONFLICTS.fetch_add(1, Ordering::Relaxed);
            range = range.and(&conflicts.not());
        }
        let mut keep = PageMask::empty();
        for (rs, re) in range.and(&readable_before).runs() {
            for p in rs..re {
                if through(self.base + p as u64 * PAGE_BYTES) {
                    keep.set(p);
                }
            }
        }
        self.clear_landing(s, e);
        self.bump(&range);
        self.cool_written(&range);
        self.gpu_dirty = self.gpu_dirty.or(&range);
        self.read_through = self.read_through.and(&range.not()).or(&keep);
        self.update_gpu_protection(ctx)
    }

    /// GPU-written pages are not CPU rings: their rewrite count restarts
    /// (hot pages cool through [`Self::cool`]).
    fn cool_written(&mut self, range: &PageMask) {
        for (rs, re) in range.and(&self.hot.not()).runs() {
            self.heat[rs..re].fill(0);
        }
        self.opened = self.opened.and(&range.not());
    }

    fn clear_landing(&mut self, s: usize, e: usize) {
        if let Some(l) = self.landing.as_mut() {
            l[s..e].fill(0);
        }
    }

    /// A copy of the current GPU contents of the range's GPU-dirty pages
    /// into guest memory is recorded in `tick` (after their last GPU
    /// write): once the tick completes, guest memory holds them (see
    /// [`Self::take_landed`]). A later GPU write of a page forgets it.
    pub fn set_landing(&mut self, addr: u64, size: u64, tick: u64) {
        let (s, e) = self.pages(addr, size);
        if !self.gpu_dirty.any_in_range(s, e) {
            return;
        }
        let l = self.landing.get_or_insert_with(|| Box::new([0; PAGES_PER_BLOCK]));
        for (rs, re) in PageMask::slice(&self.gpu_dirty, s, e).runs() {
            l[rs..re].fill(tick);
        }
    }

    /// A copy of the GPU-dirty pages of the range into guest memory is
    /// about to be recorded in `tick` (0: their bytes reach guest memory
    /// later, through the back-end): they land with it, not with an older
    /// copy, and `visit` sees them (the pages to copy). Pages made clean
    /// before (a CPU access took an older copy) are not visited: the CPU
    /// may have written them since, a copy of the mirror would undo that.
    pub fn claim_landing(&mut self, addr: u64, size: u64, tick: u64, visit: &mut dyn FnMut(u64, u64)) {
        let (s, e) = self.pages(addr, size);
        if !self.gpu_dirty.any_in_range(s, e) {
            return;
        }
        let l = self.landing.get_or_insert_with(|| Box::new([0; PAGES_PER_BLOCK]));
        for (rs, re) in PageMask::slice(&self.gpu_dirty, s, e).runs() {
            l[rs..re].fill(tick);
            visit(self.base + rs as u64 * PAGE_BYTES, (re - rs) as u64 * PAGE_BYTES);
        }
    }

    /// The landing tick of the page holding `addr` (0: none).
    pub fn landing(&self, addr: u64) -> u64 {
        let (i, _) = self.pages(addr, 1);
        match &self.landing {
            Some(l) if self.gpu_dirty.get(i) => l[i],
            _ => 0,
        }
    }

    /// Fault path, allocation-free: the page holding `addr` is GPU-dirty
    /// and its copy into guest memory completed by tick `completed`: guest
    /// memory holds the GPU contents, so the page becomes clean (readable;
    /// write-watched) without any GPU work. `Landed` only when its
    /// protection could be updated.
    pub fn take_landed(&mut self, ctx: &Ctx, addr: u64, completed: u64) -> Landing {
        let (i, _) = self.pages(addr, 1);
        if !self.gpu_dirty.get(i) {
            return Landing::Clean;
        }
        let tick = self.landing.as_ref().map_or(0, |l| l[i]);
        if tick == 0 {
            return Landing::Missing;
        }
        if tick > completed {
            return Landing::InFlight(tick);
        }
        if let Some(l) = self.landing.as_mut() {
            l[i] = 0;
        }
        self.gpu_dirty.unset(i);
        self.read_through = self.read_through.and(&self.gpu_dirty);
        self.generations.bump_range(i, i + 1);
        LANDED_PAGES.fetch_add(1, Ordering::Relaxed);
        if self.update_gpu_protection(ctx) {
            Landing::Landed
        } else {
            Landing::Missing
        }
    }

    /// GPU writes to the range are no longer written through: its
    /// GPU-dirty pages block CPU reads again.
    pub fn end_read_through(&mut self, ctx: &Ctx, addr: u64, size: u64) -> bool {
        let (s, e) = self.pages(addr, size);
        if !self.read_through.any_in_range(s, e) {
            return true;
        }
        self.read_through.unset_range(s, e);
        self.update_gpu_protection(ctx)
    }

    /// Clear the CPU-dirty pages of the range for upload (they become
    /// write-watched); `visit` sees every run to upload. With `read` (the
    /// use only reads the range; it gives a page's guest bytes without
    /// tripping the watches) pages rewritten again and again turn hot and stay dirty and
    /// unwatched, and a hot page is visited only when its bytes differ from
    /// its last upload. Without it (the GPU writes the range) hot pages cool
    /// down and are cleared like any other page.
    pub fn for_each_cpu_upload_range(&mut self, ctx: &Ctx, addr: u64, size: u64, read: Option<&dyn Fn(u64) -> Option<*const u8>>, visit: &mut dyn FnMut(u64, u64)) {
        let (s, e) = self.pages(addr, size);
        let mut range = PageMask::empty();
        range.set_range(s, e);
        if read.is_none() {
            let cooled = self.hot.and(&range);
            if cooled.any() {
                self.cool(&cooled, 0);
            }
        }
        let dirty = self.cpu_dirty.and(&range);
        if dirty.none() {
            return;
        }
        let mut upload = PageMask::empty();
        let hot = dirty.and(&self.hot);
        let generation = super::sync_generation();
        if self.checked_generation != generation {
            self.checked = PageMask::empty();
            self.checked_generation = generation;
        }
        let unchecked = hot.and(&self.checked.not());
        if let (true, Some(read)) = (unchecked.any(), read) {
            self.checked = self.checked.or(&unchecked);
            let mut idle = PageMask::empty();
            for (rs, re) in unchecked.runs() {
                for i in rs..re {
                    let page = self.base + i as u64 * PAGE_BYTES;
                    HOT_COMPARES.fetch_add(1, Ordering::Relaxed);
                    let same = match (read(page), self.shadows.get_mut(&(i as u16))) {
                        (Some(ptr), Some(shadow)) => {
                            // SAFETY: `read` returns the page's mapped backing.
                            let current = unsafe { std::slice::from_raw_parts(ptr, PAGE_BYTES as usize) };
                            let same = shadow[..] == current[..];
                            if !same {
                                shadow.copy_from_slice(current);
                            }
                            same
                        }
                        _ => false,
                    };
                    if same {
                        self.idle[i] = self.idle[i].saturating_add(1);
                        if self.idle[i] >= HOT_IDLE_USES {
                            idle.set(i);
                        }
                    } else {
                        self.idle[i] = 0;
                        upload.set(i);
                        HOT_CHANGED.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            if idle.any() {
                // Unchanged for a while: watch it again (it is current on
                // the GPU, so it becomes clean).
                self.cool(&idle, 0);
                self.cpu_dirty = self.cpu_dirty.and(&idle.not());
                self.update_cpu_protection(ctx);
            }
        }
        let cold = dirty.and(&hot.not());
        // A page a fault run opened was not necessarily rewritten: its
        // upload leaves its count alone.
        let opened = cold.and(&self.opened);
        self.opened = self.opened.and(&dirty.not());
        let mut newly_hot = PageMask::empty();
        if cold.any() {
            for (rs, re) in cold.runs() {
                for i in rs..re {
                    if opened.get(i) {
                        continue;
                    }
                    let h = self.heat[i].saturating_add(1);
                    self.heat[i] = h;
                    if h >= HOT_UPLOADS {
                        if let Some(ptr) = read.and_then(|r| r(self.base + i as u64 * PAGE_BYTES)) {
                            let mut bytes = Box::new([0u8; PAGE_BYTES as usize]);
                            // SAFETY: `read` returns the page's mapped backing.
                            bytes.copy_from_slice(unsafe { std::slice::from_raw_parts(ptr, PAGE_BYTES as usize) });
                            self.shadows.insert(i as u16, bytes);
                            self.idle[i] = 0;
                            newly_hot.set(i);
                        }
                    }
                }
            }
        }
        let clear = cold.and(&newly_hot.not());
        if clear.any() {
            self.cpu_dirty = self.cpu_dirty.and(&clear.not());
            self.update_cpu_protection(ctx);
        }
        if newly_hot.any() {
            HOT_PAGES.fetch_add(newly_hot.count() as u64, Ordering::Relaxed);
            self.hot = self.hot.or(&newly_hot);
        }
        for (rs, re) in clear.or(&newly_hot).or(&upload).runs() {
            visit(self.base + rs as u64 * PAGE_BYTES, (re - rs) as u64 * PAGE_BYTES);
        }
    }

    /// Hot pages of `pages` return to the watched state with `heat`.
    fn cool(&mut self, pages: &PageMask, heat: u8) {
        HOT_PAGES.fetch_sub(pages.and(&self.hot).count() as u64, Ordering::Relaxed);
        for (rs, re) in pages.runs() {
            for i in rs..re {
                self.heat[i] = heat;
                self.shadows.remove(&(i as u16));
            }
        }
        self.hot = self.hot.and(&pages.not());
    }

    /// Forget the uploaded copies of hot pages (a failed upload): their
    /// next use uploads them.
    pub fn forget_shadows(&mut self) {
        self.shadows.clear();
        self.checked = PageMask::empty();
    }

    /// Hot page count (diagnostics).
    pub fn hot_pages(&self) -> u32 {
        self.hot.count()
    }

    /// Visit maximal dirty runs of `side` (after clearing them when `clear`).
    pub fn for_each_modified_range(&mut self, ctx: &Ctx, side: WriteOrigin, clear: bool, addr: u64, size: u64, visit: &mut dyn FnMut(u64, u64)) {
        let (s, e) = self.pages(addr, size);
        let m = PageMask::slice(self.mask(side), s, e);
        if clear {
            match side {
                WriteOrigin::Cpu => {
                    self.cpu_dirty.unset_range(s, e);
                    self.update_cpu_protection(ctx);
                }
                WriteOrigin::Gpu => {
                    self.bump(&m);
                    self.gpu_dirty.unset_range(s, e);
                    self.read_through.unset_range(s, e);
                    self.clear_landing(s, e);
                    self.update_gpu_protection(ctx);
                }
            }
        }
        for (rs, re) in m.runs() {
            visit(self.base + rs as u64 * PAGE_BYTES, (re - rs) as u64 * PAGE_BYTES);
        }
    }

    /// Writable pages follow the CPU-dirty set: clean pages get a write watch.
    fn update_cpu_protection(&mut self, ctx: &Ctx) -> bool {
        ctx.summary.set(self.base / BLOCK_BYTES, self.cpu_dirty.any());
        let dirtied = self.cpu_dirty.and(&self.writable.not());
        if dirtied.any() {
            self.bump(&dirtied);
            ctx.summary.note_dirtied();
        }
        let changed = self.cpu_dirty.xor(&self.writable);
        self.writable = self.cpu_dirty;
        if changed.none() {
            return true;
        }
        // Pages that became clean gain a write watch; pages that became
        // dirty lose theirs.
        let gained = changed.and(&self.cpu_dirty.not());
        let lost = changed.and(&self.cpu_dirty);
        if gained.any() {
            ctx.pages.add_watch_mask(self.base, &gained, false);
        }
        if lost.any() {
            return ctx.pages.remove_watch_mask(self.base, &lost, false, ctx.fault);
        }
        true
    }

    /// Readable pages are the complement of the GPU-dirty set, plus the
    /// read-through pages.
    fn update_gpu_protection(&mut self, ctx: &Ctx) -> bool {
        let readable = self.gpu_dirty.not().or(&self.read_through);
        let changed = readable.xor(&self.readable);
        self.readable = readable;
        if changed.none() {
            return true;
        }
        let blocked = changed.and(&readable.not());
        let released = changed.and(&readable);
        if blocked.any() {
            ctx.pages.add_watch_mask(self.base, &blocked, true);
        }
        if released.any() {
            return ctx.pages.remove_watch_mask(self.base, &released, true, ctx.fault);
        }
        true
    }
}

#[cfg(test)]
mod generation_tests {
    use super::*;

    /// A bump moves the generation of every page of its range, by group
    /// counters or page by page, and of no other page.
    #[test]
    fn bumps_move_exactly_the_pages_of_the_range() {
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        for groups in [true, false] {
            let g = Generations::new();
            for _ in 0..3000 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let s = (rng % PAGES_PER_BLOCK as u64) as usize;
                let len = [1, 3, 63, 64, 65, 129, 640, PAGES_PER_BLOCK][(rng >> 32) as usize % 8];
                let e = (s + len).min(PAGES_PER_BLOCK);
                let before: Vec<u32> = (0..PAGES_PER_BLOCK).map(|p| g.get(p)).collect();
                g.bump_range_with(s, e, groups);
                for (p, &b) in before.iter().enumerate() {
                    assert_eq!(g.get(p) != b, (s..e).contains(&p), "page {p} after a bump of {s}..{e} (groups {groups})");
                }
            }
        }
    }
}
