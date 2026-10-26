// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest image cache.
//!
//! Guest surfaces (render targets, depth buffers, display buffers) are
//! mirrored in Vulkan images. An image is created from an [`ImageKey`]
//! (address, host format, extent, layers, tiling), filled from guest memory
//! on first use (read through the backing alias, detiled on the GPU by
//! [`GpuTiler`]) and its pages are write-watched afterwards: a CPU write
//! into the surface marks the image CPU-written (from the fault handler,
//! through [`WatchTable`]) and the next use uploads it again.
//!
//! Images the GPU wrote (`gpu_modified`) hold data newer than guest memory.
//! They stay on the GPU. A CPU-visible consumer gets them through
//! [`ImageCache::writeback`]: the GPU tiles the image into the buffer
//! cache's mirror of its pages, which become GPU-dirty, so the CPU
//! receives them on its first access (through the fault hook) and nothing
//! waits. Uploads read that mirror when it holds GPU-written pages.
//! Display flips take GPU-modified images directly
//! ([`ImageCache::find_display`]).
//!
//! Render-target memory is reused by surfaces of other formats and sizes.
//! Images over the same memory coexist; each carries a content version
//! (`seq`, bumped by every upload, clear and draw). When an image is used
//! while an overlapping image holds newer GPU data, it takes that data on
//! the GPU: an image copy when the surface geometry matches (another
//! format of the same surface), otherwise the newer images are tiled into
//! a scratch buffer in their guest layout and the image is detiled from it
//! (a reinterpretation of the bytes, as the hardware would see them). GPU
//! data never goes through guest memory because another image reuses the
//! range. An image fully overwritten by a newer one and idle for a while
//! is dropped; retired host images go to an [`pool::ImagePool`], so
//! recreating one is cheap.
//!
//! Images carry the guest surface's full shape: 1D / 2D / 3D, array
//! layers, mip levels and sample count. Uploads and write-backs move every
//! level of every layer (or depth slice) through the GPU tiler, following
//! the GFX10 mip chain layout of [`crate::tiling_layout`] (mip tail,
//! thick volumes, thin-slice volumes, linear chains). Square 2D arrays of
//! at least six layers are cube compatible; volumes are 2D-array
//! compatible, so their slices can be rendered to. Uploads stage guest
//! bytes in a persistent ring (or take them from the buffer cache when the
//! GPU wrote the range through a buffer).
//!
//! Every image tracks the layout and access of its last use (for the whole
//! image, or per level and layer once a use covered part of it; see
//! [`layout`]). Operations take images into the layout they need (transfer
//! source or destination, attachment, sampled, storage) and record a barrier
//! only when the layout or the access changes, or after a write. Transfers
//! through buffers order the buffers with memory barriers of their own, so
//! image work needs no full barrier around it.

pub mod blit;
pub mod layout;
pub mod pool;
pub mod sampled_dcc;

use crate::buffers::BufferCache;
use crate::device::{Buffer, GpuDevice, Image, ImageView, MemoryUse};
use crate::memory::guard::PageGuard;
use crate::memory::lock::RegionLock;
use crate::memory::{GuestGpuMemory, PAGE_BYTES};
use crate::scheduler::{Scheduler, StagingRing};
use crate::tiler::{ColorTransform, GpuTiler, TileRegion, TilerBuffer, TransformRegion};
use crate::tiling_layout::SurfaceLayout;
use ash::vk;
use layout::{stages_for, LayoutTracker, SubRange};
use pool::{ImagePool, PoolKey, Scratch};
use std::cell::UnsafeCell;
use crate::fxhash::HashMap;
use std::collections::BTreeMap;
use std::sync::Arc;

/// One plane of a guest surface (color, depth or stencil).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Plane {
    pub addr: u64,
    pub bytes_per_element: u32,
    /// GFX10 swizzle mode.
    pub swizzle: u32,
    /// Tiled row pitch of level 0 in elements.
    pub pitch: u32,
    /// Bytes of one layer (with its whole mip chain), or of one slab of a
    /// thick volume.
    pub slice_size: u64,
    /// Bytes of the whole plane in guest memory.
    pub bytes: u64,
    pub aspect: vk::ImageAspectFlags,
}

impl Plane {
    /// Guest bytes of the plane.
    pub fn size(&self) -> u64 {
        self.bytes
    }
}

/// Host image type of a guest surface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ImageDim {
    D1,
    #[default]
    D2,
    D3,
}

/// Identity of a cached image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageKey {
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    /// Depth of a volume (1 otherwise).
    pub depth: u32,
    /// Array layers (1 for volumes).
    pub layers: u32,
    pub levels: u32,
    pub samples: u32,
    pub dim: ImageDim,
    /// Color or depth plane.
    pub main: Plane,
    /// Stencil plane of a depth/stencil image.
    pub stencil: Option<Plane>,
    /// Bit rearrangement between the guest storage and the host format,
    /// applied by uploads and write-backs.
    pub transform: ColorTransform,
}

impl ImageKey {
    pub fn addr(&self) -> u64 {
        self.main.addr
    }
    /// Guest bytes of the main plane.
    pub fn size(&self) -> u64 {
        self.main.size()
    }
    pub fn is_depth(&self) -> bool {
        self.main.aspect != vk::ImageAspectFlags::COLOR
    }
    fn planes(&self) -> impl Iterator<Item = Plane> {
        std::iter::once(self.main).chain(self.stencil)
    }
    /// Array layers, or depth slices of a volume.
    pub fn slices(&self) -> u32 {
        if self.dim == ImageDim::D3 {
            self.depth
        } else {
            self.layers
        }
    }
    /// The guest layout of `plane` (None when it cannot be described).
    pub fn layout(&self, plane: &Plane) -> Option<SurfaceLayout> {
        let scale = if plane.aspect == vk::ImageAspectFlags::COLOR { block_dim(self.format) } else { 1 };
        crate::tiling_layout::describe(plane.swizzle, plane.bytes_per_element, scale, self.width, self.height, self.slices(), self.levels, self.dim == ImageDim::D3)
    }
    /// Texel extent of `level`.
    pub fn level_extent(&self, level: u32) -> vk::Extent3D {
        vk::Extent3D {
            width: (self.width >> level).max(1),
            height: (self.height >> level).max(1),
            depth: if self.dim == ImageDim::D3 { (self.depth >> level).max(1) } else { 1 },
        }
    }
    /// `o` is this array surface with at least as many layers: the same
    /// format, size, levels, samples and per-layer layout at the same
    /// address. Its first layers are this image (a view of fewer layers,
    /// as a T# or render target whose last slice is lower addresses).
    fn layers_within(&self, o: &ImageKey) -> bool {
        let plane = |a: &Plane, b: &Plane| {
            a.addr == b.addr && a.bytes_per_element == b.bytes_per_element && a.swizzle == b.swizzle && a.pitch == b.pitch && a.slice_size == b.slice_size && a.aspect == b.aspect
        };
        self.dim != ImageDim::D3
            && self.dim == o.dim
            && self.layers <= o.layers
            && self.format == o.format
            && self.width == o.width
            && self.height == o.height
            && self.depth == o.depth
            && self.levels == o.levels
            && self.samples == o.samples
            && self.transform == o.transform
            && plane(&self.main, &o.main)
            && match (&self.stencil, &o.stencil) {
                (None, None) => true,
                (Some(a), Some(b)) => plane(a, b),
                _ => false,
            }
    }

    /// Same guest surface geometry as `o` (contents can be copied between
    /// the two images element for element).
    fn same_geometry(&self, o: &ImageKey) -> bool {
        self.width == o.width
            && self.height == o.height
            && self.depth == o.depth
            && self.layers == o.layers
            && self.levels == o.levels
            && self.samples == o.samples
            && self.dim == o.dim
            && self.main.addr == o.main.addr
            && self.main.bytes_per_element == o.main.bytes_per_element
            && self.main.swizzle == o.main.swizzle
            && block_dim(self.format) == block_dim(o.format)
    }
}

/// Plane fields of a surface of `key` shape: pitch, layer/slab stride and
/// guest bytes from the GFX10 layout (`samples` multiplies the bytes of
/// multisampled surfaces, which are not laid out here).
pub fn plane(key_shape: &ImageKey, addr: u64, bytes_per_element: u32, swizzle: u32, aspect: vk::ImageAspectFlags) -> Plane {
    let mut p = Plane { addr, bytes_per_element, swizzle, pitch: key_shape.width, slice_size: 0, bytes: 0, aspect };
    match key_shape.layout(&p) {
        Some(l) => {
            p.pitch = l.mips[0].pitch;
            p.slice_size = l.layer_bytes * key_shape.samples.max(1) as u64;
            p.bytes = l.guest_bytes * key_shape.samples.max(1) as u64;
        }
        None => {
            let bd = if aspect == vk::ImageAspectFlags::COLOR { block_dim(key_shape.format) } else { 1 };
            let row = key_shape.width.div_ceil(bd) as u64 * bytes_per_element as u64;
            p.slice_size = row * key_shape.height.div_ceil(bd) as u64 * key_shape.samples.max(1) as u64;
            p.bytes = p.slice_size * key_shape.slices() as u64;
        }
    }
    p
}

/// One bit per page of an image's page range.
#[derive(Clone, Debug, Default)]
struct PageBits(Vec<u64>);

impl PageBits {
    fn new(pages: u64) -> Self {
        PageBits(vec![0; pages.div_ceil(64) as usize])
    }
    fn get(&self, i: u64) -> bool {
        self.0.get((i / 64) as usize).is_some_and(|w| w & (1 << (i % 64)) != 0)
    }
    /// Set bit `i`; true when it was clear.
    fn set(&mut self, i: u64) -> bool {
        match self.0.get_mut((i / 64) as usize) {
            Some(w) if *w & (1 << (i % 64)) == 0 => {
                *w |= 1 << (i % 64);
                true
            }
            _ => false,
        }
    }
    fn any(&self) -> bool {
        self.0.iter().any(|&w| w != 0)
    }
    fn clear(&mut self) {
        self.0.iter_mut().for_each(|w| *w = 0);
    }
    /// Runs of set (`set`) or clear bits among the first `pages`:
    /// `(first, end)` page indices.
    fn runs(&self, pages: u64, set: bool) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        let mut start = None;
        for i in 0..=pages {
            let on = i < pages && self.get(i) == set;
            match (on, start) {
                (true, None) => start = Some(i),
                (false, Some(s)) => {
                    out.push((s, i));
                    start = None;
                }
                _ => {}
            }
        }
        out
    }
}

/// Fault-side record of one image's watched range.
///
/// An image the GPU modified keeps its contents across CPU writes into its
/// memory (`partial`): a write fault unwatches only the pages it hits and
/// records them, the rest stays watched, and the next use takes those
/// pages from guest memory over the image's own GPU contents (as when the
/// image had been written back before the CPU wrote). Other images stop
/// being watched as a whole on the first write and are loaded again.
#[derive(Clone, Debug)]
struct WatchEntry {
    id: u64,
    begin: u64,
    end: u64,
    /// The image's range is watched (in partial mode: except the pages in
    /// `unwatched`).
    watched: bool,
    cpu_written: bool,
    /// `cpu_written` came from GPU work through a buffer (diagnostics).
    gpu_written: bool,
    /// The image holds GPU contents newer than guest memory: CPU writes
    /// are tracked per page.
    partial: bool,
    /// The image's range was not wholly committed when it was last armed
    /// (a partially resident texture): its committed pages are watched,
    /// and CPU writes and mapping changes are tracked per page, like
    /// `partial`, so committing a page refreshes just that page.
    sparse: bool,
    /// `cpu_written` covers the whole image (a write in whole-image mode,
    /// or the image was marked stale): nothing to merge.
    whole: bool,
    /// The CPU (or a buffer write) wrote into the image's bytes since it
    /// was last loaded whole from guest memory. Its contents stay usable on
    /// the GPU (merged), but its memory partly belongs to the CPU now, and
    /// not every guest write passes the tracker: it is never written back
    /// (a write-back would put GPU bytes over the CPU's).
    touched: bool,
    /// Pages whose watch a CPU write removed (partial mode).
    unwatched: PageBits,
    /// Pages whose image bytes were written since the last load (partial
    /// mode): taken from guest memory at the next use.
    dirty: PageBits,
    /// Edge pages (the first and last, the only ones an image shares with
    /// neighbouring allocations) unwatched by a write next to the image:
    /// later writes to the image's bytes there no longer fault, so the
    /// bytes are compared with `maybe_hash` at the next use (partial mode).
    maybe: PageBits,
    /// Hash of the image's bytes on the first ([0]) and last ([1]) page
    /// when the page was unwatched (see `maybe`).
    maybe_hash: [u64; 2],
    /// The first write that set `cpu_written` (diagnostics): address,
    /// size, guest instruction pointer of a CPU write.
    hit: (u64, u64, u64),
    /// Writes that marked the image (bytes or dirty pages) so far: a
    /// load reads guest memory before the image is armed again, and a
    /// write landing in between must not be cleared with the old marks.
    writes: u64,
    /// A bindless table keeps a view of the image (see
    /// [`ImageCache::pin`]): marks move [`WatchTable::pinned_changes`].
    pinned: bool,
}

impl WatchEntry {
    fn pages(&self) -> (u64, u64) {
        WatchTable::page_range(self.begin, self.end)
    }

    fn page_count(&self) -> u64 {
        let (b, e) = self.pages();
        (e - b) / PAGE_BYTES
    }

    /// Byte runs of the watched pages (the whole range unless partially
    /// unwatched).
    fn watched_runs(&self) -> Vec<(u64, u64)> {
        let (b, _) = self.pages();
        if !self.unwatched.any() {
            return vec![(self.begin, self.end)];
        }
        self.unwatched.runs(self.page_count(), false).into_iter().map(|(f, e)| (b + f * PAGE_BYTES, b + e * PAGE_BYTES)).collect()
    }
}

/// Bucket size of the watch table's index (1 MiB).
const WATCH_BUCKET_BITS: u32 = 20;
/// Ranges over this many buckets or more look at every entry instead.
const WATCH_INDEX_MAX_BUCKETS: u64 = 64;

/// Changes of pinned images the watch table remembers by image.
const PINNED_RING: usize = 4096;

/// `RPS5E_PINNED_LOG=0`: a bindless table looks at every image it holds
/// whenever any pinned image's memory changed, instead of at the images
/// whose memory changed (A/B scope `release` toggles it).
pub fn pinned_log_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_PINNED_LOG")) && crate::backend::ab::release_fast("RPS5E_PINNED_LOG")
}

/// `RPS5E_WATCH_INDEX=0`: a CPU write (a fault) or a GPU buffer write
/// looks at every image entry to find the ones it hits instead of only the
/// entries over its 1 MiB of memory (with thousands of images, the scan
/// under the table lock made every write fault cost tens of microseconds
/// and serialised the threads that fault). A/B scope `release` toggles it.
#[inline]
fn watch_index_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_WATCH_INDEX")) && crate::backend::ab::release_fast("RPS5E_WATCH_INDEX")
}

/// `RPS5E_WATCH_INDEX_VERIFY=1` (diagnostic): every lookup through the
/// index also counts the overlapping entries by looking at all of them;
/// lookups that disagree are counted in [`WATCH_INDEX_MISSES`].
fn watch_index_verify() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_WATCH_INDEX_VERIFY").is_ok_and(|v| v == "1"))
}

pub static WATCH_INDEX_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `RPS5E_MAYBE_DIRTY=0`: edge pages unwatched by a neighbour's write are
/// not compared at the next use (writes to the image's bytes there are
/// missed, as before the comparison existed).
fn maybe_dirty_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_MAYBE_DIRTY").is_ok_and(|v| v == "0"))
}

/// FNV-1a hash of the image bytes `[begin, end)` on the page at `page`,
/// read through the memfd alias (never protected, so this is safe on the
/// fault path and allocates nothing). None without an alias.
fn edge_page_hash(memory: Option<&crate::memory::GuestGpuMemory>, page: u64, begin: u64, end: u64) -> Option<u64> {
    let (lo, hi) = (page.max(begin), (page + PAGE_BYTES).min(end));
    if lo >= hi || !maybe_dirty_enabled() {
        return None;
    }
    let p = memory?.backing_ptr(lo, hi - lo)?;
    // SAFETY: the alias maps the memfd range backing these guest bytes.
    let bytes = unsafe { std::slice::from_raw_parts(p as *const u8, (hi - lo) as usize) };
    Some(fnv1a(bytes))
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The watched ranges of all images, consulted by the fault handler
/// (allocation-free, spin-locked).
pub struct WatchTable {
    lock: RegionLock,
    entries: UnsafeCell<Vec<WatchEntry>>,
    /// Position of each image's entry (not used by the fault handler).
    index: UnsafeCell<HashMap<u64, usize>>,
    /// Positions of the entries over each 1 MiB of guest memory
    /// ([`WATCH_BUCKET_BITS`]): a write finds the images it hits without
    /// looking at every entry (the fault handler reads it, never changes it).
    buckets: UnsafeCell<HashMap<u64, smallvec::SmallVec<[u32; 6]>>>,
    pages: Arc<PageGuard>,
    /// Bumped whenever an entry may have changed state (CPU write, GPU
    /// buffer write): lookups memoised by [`ImageCache::lookup_epoch`].
    changes: std::sync::atomic::AtomicU64,
    /// Changes of pinned entries logged so far: the image of change `n` is
    /// `pinned_ring[n % PINNED_RING]` (see [`Self::pinned_changed_since`]).
    pinned_changes: std::sync::atomic::AtomicU64,
    /// Changes whose ring slot is being (or was) written: moved before the
    /// slot, so a reader that may have read an overwritten slot sees it.
    pinned_claimed: std::sync::atomic::AtomicU64,
    pinned_ring: Box<[std::sync::atomic::AtomicU64]>,
    /// GPU-written linear images whose pages block CPU reads until a CPU
    /// access writes them back: (first page, end page, image).
    read_backs: std::sync::Mutex<Vec<(u64, u64, u64)>>,
    /// Guest memory whose memfd alias the edge-page hashes read (none: no
    /// maybe-written pages are tracked).
    memory: Option<&'static crate::memory::GuestGpuMemory>,
}

// SAFETY: `entries` is only accessed with `lock` held.
unsafe impl Sync for WatchTable {}
unsafe impl Send for WatchTable {}

impl WatchTable {
    pub fn new(pages: Arc<PageGuard>) -> Self {
        // Read the switch here: the fault handler must not be the first
        // to ask (reading the environment allocates).
        let _ = (watch_index_enabled(), watch_index_verify());
        Self {
            lock: RegionLock::new(),
            entries: UnsafeCell::new(Vec::with_capacity(1024)),
            index: UnsafeCell::new(HashMap::default()),
            buckets: UnsafeCell::new(HashMap::default()),
            pages,
            changes: std::sync::atomic::AtomicU64::new(0),
            pinned_changes: std::sync::atomic::AtomicU64::new(0),
            pinned_claimed: std::sync::atomic::AtomicU64::new(0),
            pinned_ring: (0..PINNED_RING).map(|_| std::sync::atomic::AtomicU64::new(0)).collect(),
            read_backs: std::sync::Mutex::new(Vec::new()),
            memory: None,
        }
    }

    /// A table over `memory`'s pages that hashes edge pages through its
    /// memfd alias (see `WatchEntry::maybe`).
    pub fn with_memory(memory: &'static crate::memory::GuestGpuMemory) -> Self {
        Self { memory: Some(memory), ..Self::new(memory.pages.clone()) }
    }

    /// Image `id` at `[begin, end)` holds GPU results the CPU has not seen:
    /// read-watch its pages until a CPU access asks for them
    /// ([`Self::take_read_backs`]). Pages a buffer already read-watches keep
    /// that watch; the fault handler asks the images first either way.
    pub fn watch_reads(&self, id: u64, begin: u64, end: u64) {
        let mut v = self.read_backs.lock().unwrap_or_else(|e| e.into_inner());
        let (pb, pe) = Self::page_range(begin, end);
        // Also when the entry exists: a buffer download may have cleared a
        // read watch the image relied on since.
        // One protection change per run of unwatched pages.
        let mut run: Option<u64> = None;
        for page in (pb..pe).step_by(PAGE_BYTES as usize) {
            match (self.pages.watchers(page).0, run) {
                (false, None) => run = Some(page),
                (true, Some(start)) => {
                    self.pages.add_watch(start, page - start, true);
                    run = None;
                }
                _ => {}
            }
        }
        if let Some(start) = run {
            self.pages.add_watch(start, pe - start, true);
        }
        if !v.iter().any(|e| e.2 == id) {
            v.push((pb, pe, id));
        }
    }

    /// The read-watched images over the page of `addr`, removed from the
    /// table (fault path: None when the table is busy).
    pub fn take_read_backs(&self, addr: u64) -> Option<Vec<(u64, u64, u64)>> {
        let mut v = self.read_backs.try_lock().ok()?;
        let page = addr & !(PAGE_BYTES - 1);
        let mut hits = Vec::new();
        v.retain(|&e| {
            let hit = e.0 <= page && page < e.1;
            if hit {
                hits.push(e);
            }
            !hit
        });
        Some(hits)
    }

    pub fn pages(&self) -> &Arc<PageGuard> {
        &self.pages
    }

    fn page_range(begin: u64, end: u64) -> (u64, u64) {
        (begin & !(PAGE_BYTES - 1), end.div_ceil(PAGE_BYTES) * PAGE_BYTES)
    }

    /// The buckets of `[pb, pe)` (page-aligned, not empty), first and last.
    fn bucket_span(pb: u64, pe: u64) -> (u64, u64) {
        (pb >> WATCH_BUCKET_BITS, (pe - 1) >> WATCH_BUCKET_BITS)
    }

    /// Call `f` once for every entry whose pages overlap `[pb, pe)`
    /// (page-aligned), through the bucket index when it is on and the
    /// range spans few buckets, otherwise by looking at every entry. No
    /// allocation: the fault handler calls it.
    fn for_overlapping(entries: &mut [WatchEntry], buckets: &HashMap<u64, smallvec::SmallVec<[u32; 6]>>, pb: u64, pe: u64, mut f: impl FnMut(&mut WatchEntry)) {
        if pb >= pe {
            return;
        }
        let (qb, qe) = Self::bucket_span(pb, pe);
        if !watch_index_enabled() || qe - qb >= WATCH_INDEX_MAX_BUCKETS {
            for e in entries.iter_mut() {
                let (eb, ee) = e.pages();
                if eb < pe && pb < ee {
                    f(e);
                }
            }
            return;
        }
        let mut found = 0usize;
        for b in qb..=qe {
            let Some(list) = buckets.get(&b) else { continue };
            for &pos in list.iter() {
                let Some(e) = entries.get_mut(pos as usize) else { continue };
                let (eb, ee) = e.pages();
                // Each entry once: at the first bucket both spans share.
                if !(eb < pe && pb < ee) || (eb >> WATCH_BUCKET_BITS).max(qb) != b {
                    continue;
                }
                found += 1;
                f(e);
            }
        }
        if watch_index_verify() {
            let all = entries.iter().filter(|e| {
                let (eb, ee) = e.pages();
                eb < pe && pb < ee
            });
            if all.count() != found {
                WATCH_INDEX_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// A CPU write fault at `[addr, addr+size)`: images whose bytes it hit
    /// become CPU-written; every image sharing the pages stops being
    /// watched (a GPU-modified image only on the pages written). True when
    /// an image page covers the range.
    pub fn mark_cpu_write(&self, addr: u64, size: u64, fault: bool) -> bool {
        let _l = if fault {
            match self.lock.lock_for_fault() {
                Some(l) => l,
                None => return false,
            }
        } else {
            self.lock.lock()
        };
        // SAFETY: lock held.
        let entries = unsafe { &mut *self.entries.get() };
        // SAFETY: lock held.
        let buckets = unsafe { &*self.buckets.get() };
        let (pb, pe) = Self::page_range(addr, addr + size);
        let mut covered = false;
        Self::for_overlapping(entries, buckets, pb, pe, |e| {
            let (eb, ee) = e.pages();
            covered = true;
            if e.pinned {
                self.note_pinned_change(e.id);
            }
            let bytes = addr < e.end && e.begin < addr + size;
            if bytes && !e.cpu_written {
                e.hit = (addr, size, if fault { ps5vm::fault_site::get() } else { 0 });
            }
            if (e.partial || e.sparse) && e.watched && !e.whole {
                // Only the pages written: no allocation on this path. The
                // watch goes from every page the write touches, also one the
                // image only shares with a neighbouring allocation: page
                // protection is per page, and the faulting store can only
                // complete once the page is writable. Such a page is
                // unwatched but not dirty.
                for page in (pb.max(eb)..pe.min(ee)).step_by(PAGE_BYTES as usize) {
                    let i = (page - eb) / PAGE_BYTES;
                    let lo = page.max(addr).max(e.begin);
                    let hi = (page + PAGE_BYTES).min(addr + size).min(e.end);
                    if lo < hi {
                        e.dirty.set(i);
                        e.touched = true;
                        e.writes += 1;
                    } else if !e.dirty.get(i) && !e.maybe.get(i) {
                        // A neighbour's write: remember the image's bytes
                        // on this edge page to see later writes to them.
                        if let Some(h) = edge_page_hash(self.memory, page, e.begin, e.end) {
                            e.maybe.set(i);
                            e.maybe_hash[(i != 0) as usize] = h;
                        } else {
                            // Later stores to the image's bytes here can
                            // neither fault nor be compared: the memory may
                            // be the CPU's now, never written back over.
                            e.touched = true;
                        }
                    }
                    if e.unwatched.set(i) {
                        if fault {
                            self.pages.remove_watch_in_fault(page, PAGE_BYTES, false);
                        } else {
                            self.pages.remove_watch(page, PAGE_BYTES, false);
                        }
                    }
                }
                e.cpu_written |= bytes;
                return;
            }
            if bytes {
                e.cpu_written = true;
                e.whole = true;
                e.writes += 1;
            }
            if e.watched {
                e.watched = false;
                let (b, en) = (e.begin, e.end);
                if !e.unwatched.any() {
                    if fault {
                        self.pages.remove_watch_in_fault(b, en - b, false);
                    } else {
                        self.pages.remove_watch(b, en - b, false);
                    }
                } else {
                    // Partially unwatched already: the remaining pages.
                    for i in 0..e.page_count() {
                        if e.unwatched.set(i) {
                            let page = eb + i * PAGE_BYTES;
                            if fault {
                                self.pages.remove_watch_in_fault(page, PAGE_BYTES, false);
                            } else {
                                self.pages.remove_watch(page, PAGE_BYTES, false);
                            }
                        }
                    }
                }
                e.unwatched.clear();
            }
        });
        if covered {
            self.changes.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
        covered
    }

    /// GPU work wrote `[addr, addr+size)` through a buffer: images over it
    /// must be loaded again (from the buffer data, downloaded first); a
    /// GPU-modified image takes only the pages written.
    pub fn invalidate_range(&self, addr: u64, size: u64) -> usize {
        // The page of `addr` too when the range is empty: the hit test
        // (`invalidate_entry`) also counts an image around `addr` then.
        let end = addr.saturating_add(size).max(addr.saturating_add(1));
        let (pb, pe) = Self::page_range(addr, end);
        self.with_buckets(|v, buckets| {
            let mut hit = 0;
            Self::for_overlapping(v, buckets, pb, pe, |e| {
                let h = Self::invalidate_entry(e, addr, size);
                hit += h as usize;
                if h && e.pinned {
                    self.note_pinned_change(e.id);
                }
            });
            if hit > 0 {
                self.changes.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            }
            hit
        })
    }

    /// [`Self::invalidate_range`] when only the images `ids` can overlap
    /// the range (every other entry is left alone without being looked at).
    pub fn invalidate_range_of(&self, ids: &[u64], addr: u64, size: u64) -> usize {
        self.with_index(|v, index| {
            let mut hit = 0;
            for id in ids {
                if let Some(e) = index.get(id).and_then(|&i| v.get_mut(i)) {
                    let h = Self::invalidate_entry(e, addr, size);
                    hit += h as usize;
                    if h && e.pinned {
                        self.note_pinned_change(e.id);
                    }
                }
            }
            if hit > 0 {
                self.changes.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            }
            hit
        })
    }

    /// One entry of [`Self::invalidate_range`]: true when the range hit it.
    fn invalidate_entry(e: &mut WatchEntry, addr: u64, size: u64) -> bool {
        if !(addr < e.end && e.begin < addr + size) {
            return false;
        }
        if !e.cpu_written {
            e.hit = (addr, size, 0);
        }
        e.cpu_written = true;
        e.writes += 1;
        e.gpu_written = true;
        if (e.partial || e.sparse) && !e.whole {
            let (eb, _) = e.pages();
            let (b, en) = (addr.max(e.begin), (addr + size).min(e.end));
            for i in (b - eb) / PAGE_BYTES..(en - eb).div_ceil(PAGE_BYTES) {
                e.dirty.set(i);
            }
            e.touched = true;
        } else {
            e.whole = true;
        }
        true
    }

    /// Guest memory was mapped (or unmapped) at `[addr, addr+size)`: the
    /// images over it hold stale bytes there. A watched image tracked per
    /// page (sparse or GPU-modified) takes exactly those pages at its next
    /// use (uncommitted ones read as zeros); others are loaded again. The
    /// pages' new contents came from the mapping, not from a CPU store, so
    /// the image's memory stays its own (`touched` is left alone).
    pub fn mark_mapped(&self, addr: u64, size: u64) -> usize {
        let Some(end) = addr.checked_add(size) else { return 0 };
        let (pb, pe) = Self::page_range(addr, end);
        let mut unwatch: Vec<(u64, u64)> = Vec::new();
        let hit = self.with_buckets(|v, buckets| {
            let mut hit = 0;
            Self::for_overlapping(v, buckets, pb, pe, |e| {
                let (eb, ee) = e.pages();
                if !(addr < e.end && e.begin < end) {
                    return;
                }
                hit += 1;
                if e.pinned {
                    self.note_pinned_change(e.id);
                }
                if !e.cpu_written {
                    e.hit = (addr, size, 0);
                }
                e.cpu_written = true;
                e.writes += 1;
                if (e.partial || e.sparse) && e.watched && !e.whole {
                    for page in (pb.max(eb)..pe.min(ee)).step_by(PAGE_BYTES as usize) {
                        let i = (page - eb) / PAGE_BYTES;
                        e.dirty.set(i);
                        if e.unwatched.set(i) {
                            unwatch.push((page, PAGE_BYTES));
                        }
                    }
                    return;
                }
                e.whole = true;
                if e.watched {
                    e.watched = false;
                    unwatch.extend(e.watched_runs().into_iter().map(|(b, en)| (b, en - b)));
                    e.unwatched.clear();
                }
            });
            if hit > 0 {
                self.changes.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            }
            hit
        });
        for (b, n) in unwatch {
            self.pages.remove_watch(b, n, false);
        }
        hit
    }

    fn with<R>(&self, f: impl FnOnce(&mut Vec<WatchEntry>) -> R) -> R {
        let _l = self.lock.lock();
        // SAFETY: lock held.
        f(unsafe { &mut *self.entries.get() })
    }

    /// The entries and the index (lock held for the closure).
    fn with_index<R>(&self, f: impl FnOnce(&mut Vec<WatchEntry>, &mut HashMap<u64, usize>) -> R) -> R {
        let _l = self.lock.lock();
        // SAFETY: lock held.
        unsafe { f(&mut *self.entries.get(), &mut *self.index.get()) }
    }

    /// The entries and the bucket index (lock held for the closure).
    fn with_buckets<R>(&self, f: impl FnOnce(&mut Vec<WatchEntry>, &HashMap<u64, smallvec::SmallVec<[u32; 6]>>) -> R) -> R {
        let _l = self.lock.lock();
        // SAFETY: lock held.
        unsafe { f(&mut *self.entries.get(), &*self.buckets.get()) }
    }

    fn insert(&self, id: u64, begin: u64, end: u64) {
        let (pb, pe) = Self::page_range(begin, end);
        let pages = (pe - pb) / PAGE_BYTES;
        let entry = WatchEntry {
            id,
            begin,
            end,
            watched: false,
            cpu_written: false,
            gpu_written: false,
            partial: false,
            sparse: false,
            whole: false,
            touched: false,
            unwatched: PageBits::new(pages),
            dirty: PageBits::new(pages),
            maybe: PageBits::new(pages),
            maybe_hash: [0; 2],
            hit: (0, 0, 0),
            writes: 0,
            pinned: false,
        };
        let (qb, qe) = Self::bucket_span(pb, pe.max(pb + PAGE_BYTES));
        self.with_index(|v, index| {
            // SAFETY: lock held (with_index).
            let buckets = unsafe { &mut *self.buckets.get() };
            let pos = v.len() as u32;
            for b in qb..=qe {
                buckets.entry(b).or_default().push(pos);
            }
            index.insert(id, v.len());
            v.push(entry);
        });
    }

    fn remove(&self, id: u64) {
        let unwatch = self.with_index(|v, index| {
            let i = index.remove(&id)?;
            // SAFETY: lock held (with_index).
            let buckets = unsafe { &mut *self.buckets.get() };
            let last = v.len() - 1;
            let span = |e: &WatchEntry| {
                let (pb, pe) = e.pages();
                Self::bucket_span(pb, pe.max(pb + PAGE_BYTES))
            };
            let (qb, qe) = span(&v[i]);
            for b in qb..=qe {
                if let Some(list) = buckets.get_mut(&b) {
                    list.retain(|p| *p as usize != i);
                    if list.is_empty() {
                        buckets.remove(&b);
                    }
                }
            }
            if i != last {
                let (mb, me) = span(&v[last]);
                for b in mb..=me {
                    if let Some(list) = buckets.get_mut(&b) {
                        for p in list.iter_mut().filter(|p| **p as usize == last) {
                            *p = i as u32;
                        }
                    }
                }
            }
            let e = v.swap_remove(i);
            if let Some(moved) = v.get(i) {
                index.insert(moved.id, i);
            }
            e.watched.then(|| e.watched_runs())
        });
        for (b, e) in unwatch.unwrap_or_default() {
            self.pages.remove_watch(b, e - b, false);
        }
    }

    /// The image's contents are older than guest memory (a newer alias
    /// was written back and dropped): load it again before use.
    fn mark_stale(&self, id: u64) {
        self.with_index(|v, index| {
            if let Some(&i) = index.get(&id) {
                v[i].cpu_written = true;
                v[i].gpu_written = true;
                v[i].whole = true;
                v[i].writes += 1;
                if v[i].pinned {
                    self.note_pinned_change(id);
                }
            }
        });
    }

    /// Image `id` is (or no longer is) kept in a bindless table's slots.
    fn set_pinned(&self, id: u64, pinned: bool) {
        self.with_index(|v, index| {
            if let Some(&i) = index.get(&id) {
                v[i].pinned = pinned;
            }
        });
    }

    /// Bumped whenever an entry of a pinned image may have changed state
    /// (CPU writes, GPU buffer writes, mappings, a stale mark): its next
    /// lookup may load it again.
    pub fn pinned_changes(&self) -> u64 {
        self.pinned_changes.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Log a change of pinned image `id` (the table lock is held: one
    /// writer at a time; allocation-free, for the fault handler).
    fn note_pinned_change(&self, id: u64) {
        use std::sync::atomic::Ordering;
        let n = self.pinned_changes.load(Ordering::Relaxed);
        self.pinned_claimed.store(n + 1, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::Release);
        self.pinned_ring[(n % PINNED_RING as u64) as usize].store(id, Ordering::Relaxed);
        self.pinned_changes.store(n + 1, Ordering::Release);
    }

    /// The pinned images changed since change count `since` (an image may
    /// repeat), and the count they reach. None when the ring no longer
    /// holds them all (any pinned image may have changed).
    pub fn pinned_changed_since(&self, since: u64, out: &mut Vec<u64>) -> Option<u64> {
        use std::sync::atomic::Ordering;
        let end = self.pinned_changes.load(Ordering::Acquire);
        if end.wrapping_sub(since) > PINNED_RING as u64 || since > end {
            return None;
        }
        for n in since..end {
            out.push(self.pinned_ring[(n % PINNED_RING as u64) as usize].load(Ordering::Relaxed));
        }
        // A writer that overwrote a slot read above claimed its change
        // first: seen here, the ids may be wrong.
        std::sync::atomic::fence(Ordering::Acquire);
        if self.pinned_claimed.load(Ordering::Relaxed).saturating_sub(since) > PINNED_RING as u64 {
            return None;
        }
        Some(end)
    }

    /// The image's write count (see `WatchEntry::writes`).
    fn write_count(&self, id: u64) -> u64 {
        self.with_index(|v, index| index.get(&id).map_or(0, |&i| v[i].writes))
    }

    /// [`Self::dirty_runs`], clearing the dirty pages taken: pages written
    /// after this are dirty again when the merge re-arms the image.
    /// Returns the runs and the write count at the snapshot.
    fn take_dirty_runs(&self, id: u64) -> Option<(Vec<(u64, u64)>, u64)> {
        self.with_index(|v, index| {
            let e = &mut v[*index.get(&id)?];
            if !e.cpu_written || e.whole || !(e.partial || e.sparse) || !e.dirty.any() {
                return None;
            }
            let (b, _) = e.pages();
            let runs = e.dirty.runs(e.page_count(), true).into_iter().map(|(f, en)| ((b + f * PAGE_BYTES).max(e.begin), (b + en * PAGE_BYTES).min(e.end))).collect();
            e.dirty.clear();
            Some((runs, e.writes))
        })
    }

    /// Track CPU writes into the image per page (it holds GPU contents
    /// newer than guest memory) or as a whole.
    fn set_partial(&self, id: u64, partial: bool) {
        self.with_index(|v, index| {
            if let Some(&i) = index.get(&id) {
                v[i].partial = partial;
            }
        });
    }

    /// The first write that made an image stale and whether it came from
    /// GPU work through a buffer (diagnostics).
    fn hit(&self, id: u64) -> Option<((u64, u64, u64), bool)> {
        self.with_index(|v, index| index.get(&id).map(|&i| (v[i].hit, v[i].gpu_written)))
    }

    /// The image's range was partly uncommitted when it was last armed.
    fn is_sparse(&self, id: u64) -> bool {
        self.with_index(|v, index| index.get(&id).is_some_and(|&i| v[i].sparse))
    }

    /// Compare the image's bytes on its maybe-written edge pages with the
    /// hashes taken when a neighbour's write unwatched them: a change is a
    /// CPU write to the image (the page becomes dirty and is merged from
    /// guest memory at this use).
    #[cfg(test)]
    fn resolve_maybe(&self, id: u64) {
        let memory = self.memory;
        self.with_index(|v, index| {
            let Some(&i) = index.get(&id) else { return };
            Self::resolve_entry(memory, &mut v[i]);
        });
    }

    /// [`Self::resolve_maybe`] then [`Self::state`] under one lock.
    fn resolved_state(&self, id: u64) -> (bool, bool) {
        let memory = self.memory;
        self.with_index(|v, index| {
            let Some(&i) = index.get(&id) else { return (false, true) };
            Self::resolve_entry(memory, &mut v[i]);
            (v[i].watched, v[i].cpu_written)
        })
    }

    /// [`Self::write_count`], then [`Self::resolve_maybe`] and
    /// [`Self::state`], under one lock.
    fn counted_resolved_state(&self, id: u64) -> (u64, (bool, bool)) {
        let memory = self.memory;
        self.with_index(|v, index| {
            let Some(&i) = index.get(&id) else { return (0, (false, true)) };
            let writes = v[i].writes;
            Self::resolve_entry(memory, &mut v[i]);
            (writes, (v[i].watched, v[i].cpu_written))
        })
    }

    /// The edge-page comparison of [`Self::resolve_maybe`] for one entry
    /// (lock held).
    fn resolve_entry(memory: Option<&'static crate::memory::GuestGpuMemory>, e: &mut WatchEntry) {
        if !e.maybe.any() || !maybe_dirty_enabled() {
            return;
        }
        let (eb, _) = e.pages();
        let last = e.page_count() - 1;
        // A one-page image has a single edge page, hashed in slot 0.
        let edges: &[(usize, u64)] = if last == 0 { &[(0, 0)] } else { &[(0, 0), (1, last)] };
        for &(slot, page_index) in edges {
            if !e.maybe.get(page_index) || e.dirty.get(page_index) {
                continue;
            }
            let page = eb + page_index * PAGE_BYTES;
            let Some(h) = edge_page_hash(memory, page, e.begin, e.end) else { continue };
            if h != e.maybe_hash[slot] {
                e.maybe_hash[slot] = h;
                e.dirty.set(page_index);
                e.touched = true;
                e.cpu_written = true;
                e.writes += 1;
            }
        }
    }

    /// `(watched, cpu_written)` of an image.
    fn state(&self, id: u64) -> (bool, bool) {
        self.with_index(|v, index| index.get(&id).map_or((false, true), |&i| (v[i].watched, v[i].cpu_written)))
    }

    /// The image's contents are stale as a whole: CPU (or buffer) writes
    /// it cannot merge, or unknown. An image with only some pages written
    /// in partial mode still holds the newest bytes of the others.
    fn stale_whole(&self, id: u64) -> bool {
        self.with_index(|v, index| Self::stale_whole_in(v, index, id))
    }

    fn stale_whole_in(v: &[WatchEntry], index: &HashMap<u64, usize>, id: u64) -> bool {
        index.get(&id).is_none_or(|&i| v[i].cpu_written && (v[i].whole || !(v[i].partial || v[i].sparse) || !v[i].dirty.any()))
    }

    /// `f` given [`Self::stale_whole`] of any image, all asked under one
    /// lock (`f` must not use the table otherwise).
    fn with_stale_whole<R>(&self, f: impl FnOnce(&dyn Fn(u64) -> bool) -> R) -> R {
        self.with_index(|v, index| f(&|id| Self::stale_whole_in(v, index, id)))
    }

    /// The image's GPU contents may be written back to guest memory: not
    /// stale, and nothing wrote into its memory since it was last loaded
    /// whole (those pages belong to the CPU).
    /// Edge pages a neighbour's write unwatched are compared first: a store
    /// into the image's bytes there no longer faults, and a write-back
    /// would put the GPU bytes over it (GTA V reuses the memory of small
    /// GPU-written images for CPU objects next to live allocations: an
    /// eviction wrote a stale texel over an object's vtable).
    fn may_write_back(&self, id: u64) -> bool {
        let memory = self.memory;
        self.with_index(|v, index| {
            index.get(&id).is_some_and(|&i| {
                Self::resolve_entry(memory, &mut v[i]);
                !v[i].cpu_written && !v[i].touched
            })
        })
    }

    /// Diagnostics: every entry over the page of `addr` (looked up without
    /// the bucket index) with its watch state there.
    pub fn describe_page(&self, addr: u64) -> String {
        let page = addr & !(PAGE_BYTES - 1);
        let in_index = self.with_buckets(|_, buckets| {
            buckets.get(&(page >> WATCH_BUCKET_BITS)).map(|l| l.iter().map(|&p| p as usize).collect::<Vec<_>>()).unwrap_or_default()
        });
        self.with(|v| {
            v.iter()
                .enumerate()
                .filter(|(_, e)| {
                    let (eb, ee) = e.pages();
                    eb <= page && page < ee
                })
                .map(|(pos, e)| {
                    let (eb, _) = e.pages();
                    let i = (page - eb) / PAGE_BYTES;
                    format!(
                        "#{} {:#x}..{:#x} watched {} unwatched-here {} partial {} sparse {} whole {} cpu-written {} indexed {}",
                        e.id,
                        e.begin,
                        e.end,
                        e.watched,
                        e.unwatched.get(i),
                        e.partial,
                        e.sparse,
                        e.whole,
                        e.cpu_written,
                        in_index.contains(&pos)
                    )
                })
                .collect::<Vec<_>>()
                .join("; ")
        })
    }

    /// The CPU wrote into the image's memory since its load (see
    /// `WatchEntry::touched`).
    fn touched(&self, id: u64) -> bool {
        self.with_index(|v, index| index.get(&id).is_some_and(|&i| v[i].touched))
    }

    /// Keep the CPU's claim on the image's memory across a refill that
    /// rearmed it (the refill kept the CPU's pages, see `merge_cpu_pages`).
    fn touch(&self, id: u64) {
        self.with_index(|v, index| {
            if let Some(&i) = index.get(&id) {
                v[i].touched = true;
            }
        });
    }

    /// The whole image is about to be overwritten by the GPU.
    fn untouch(&self, id: u64) {
        self.with_index(|v, index| {
            if let Some(&i) = index.get(&id) {
                v[i].touched = false;
            }
        });
    }

    /// Byte runs of the pages written into a partially tracked image since
    /// its last load, clipped to the image (None when there are none or
    /// the image is stale as a whole).
    fn dirty_runs(&self, id: u64) -> Option<Vec<(u64, u64)>> {
        self.with_index(|v, index| {
            let e = &v[*index.get(&id)?];
            if !e.cpu_written || e.whole || !(e.partial || e.sparse) || !e.dirty.any() {
                return None;
            }
            let (b, _) = e.pages();
            Some(e.dirty.runs(e.page_count(), true).into_iter().map(|(f, en)| ((b + f * PAGE_BYTES).max(e.begin), (b + en * PAGE_BYTES).min(e.end))).collect())
        })
    }

    /// The image was just loaded: watch it again (when `watch`) and clear
    /// its CPU-written state.
    fn rearm(&self, id: u64, watch: bool) {
        self.rearm_with(id, watch, false, None);
    }

    /// [`WatchTable::rearm`] after a load that read guest memory when the
    /// image's write count was `since`: a write counted after it (it may
    /// have landed after the bytes were read) keeps the image stale, whole
    /// or with its dirty pages.
    fn rearm_since(&self, id: u64, watch: bool, since: u64) {
        self.rearm_with(id, watch, false, Some(since));
    }

    /// [`WatchTable::rearm`]; with `merged` the image took only the pages
    /// written and stays touched; `since` as in [`Self::rearm_since`].
    fn rearm_with(&self, id: u64, watch: bool, merged: bool, since: Option<u64>) {
        // Whether the range is wholly registered, looked at outside the
        // table lock: a watchable range that is not is partially resident.
        let Some((begin, end)) = self.with_index(|v, index| index.get(&id).map(|&i| (v[i].begin, v[i].end))) else { return };
        let sparse = watch && !self.pages.is_registered(begin, end - begin);
        self.with_index(|v, index| {
            let e = &mut v[*index.get(&id)?];
            let written_since = since.is_some_and(|s| s != e.writes) && read_race_check_enabled();
            e.sparse = sparse;
            if !merged {
                e.touched = false;
            }
            if written_since {
                // Still stale: as a whole, or in the pages written (the
                // merge took the earlier ones out of `dirty`).
                e.cpu_written = true;
                if !(e.partial || e.sparse) || !e.dirty.any() {
                    e.whole = true;
                }
            } else {
                e.cpu_written = false;
                e.gpu_written = false;
                e.whole = false;
                e.hit = (0, 0, 0);
                e.dirty.clear();
            }
            let add = if !watch {
                None
            } else if !e.watched {
                e.watched = true;
                Some(vec![(e.begin, e.end)])
            } else if e.unwatched.any() {
                // Watch the pages written again.
                let (b, _) = e.pages();
                Some(e.unwatched.runs(e.page_count(), true).into_iter().map(|(f, en)| (b + f * PAGE_BYTES, b + en * PAGE_BYTES)).collect())
            } else {
                None
            };
            if watch {
                e.unwatched.clear();
                e.maybe.clear();
            }
            // The watches go on before the table lock is released: a CPU
            // write faulting in between would otherwise find the page
            // watched by the entry, remove a watch that is not there yet,
            // and leave the one added afterwards to nobody (the store then
            // faults for ever).
            for (b, e) in add.unwrap_or_default() {
                self.pages.add_watch(b, e - b, false);
            }
            Some(())
        });
    }

    /// Why an image needs an upload (diagnostics): 1 new / unknown, 2 CPU
    /// write, 3 GPU write through a buffer, 4 not watched, 0 current.
    fn upload_reason(&self, id: u64, gpu_modified: bool) -> usize {
        self.with_index(|v, index| match index.get(&id) {
            None => 1,
            Some(&i) if v[i].gpu_written => 3,
            Some(&i) if v[i].cpu_written => 2,
            Some(&i) if !v[i].watched && !gpu_modified => 4,
            Some(_) => 0,
        })
    }

    pub fn len(&self) -> usize {
        self.with(|v| v.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The subresources and type of an image view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewRange {
    pub ty: vk::ImageViewType,
    pub base_level: u32,
    pub levels: u32,
    /// Array layers, or depth slices of a 2D (array) view of a volume.
    pub base_layer: u32,
    pub layers: u32,
    /// Minimum LOD clamp (0 = none).
    pub min_lod: f32,
}

/// Identity of a cached view: format, aspect, level / layer range, view
/// type, component swizzle and LOD clamp.
type ViewKey = (vk::Format, vk::ImageAspectFlags, [u32; 4], vk::ImageViewType, [vk::ComponentSwizzle; 4], u32);

/// A view an image can provide, clamped to it ([`ImageCache::view_plan`]).
#[derive(Clone, Copy, Debug)]
pub struct ViewPlan {
    pub id: u64,
    key: ViewKey,
    image: vk::Image,
    ty: vk::ImageViewType,
    format: vk::Format,
    components: vk::ComponentMapping,
    aspect: vk::ImageAspectFlags,
    min_lod: f32,
    /// The subresources the view covers.
    pub range: SubRange,
}

impl ViewPlan {
    /// The same view of the same image.
    pub fn same_key(&self, other: &ViewPlan) -> bool {
        self.id == other.id && self.key == other.key
    }

    /// A view of the stencil aspect.
    pub fn is_stencil(&self) -> bool {
        self.aspect == vk::ImageAspectFlags::STENCIL
    }
}

/// A cached image.
pub struct CachedImage {
    pub id: u64,
    pub key: ImageKey,
    pub image: Image,
    views: HashMap<ViewKey, ImageView>,
    /// Views of keys `views` already held, made by slice threads
    /// concurrently (R7b): they live as long as the image.
    extra_views: Vec<ImageView>,
    /// The view `views` handed out last (views live as long as the image):
    /// most uses of an image ask for the same view again.
    last_view: Option<(ViewKey, vk::ImageView)>,
    /// Contents were filled at least once (upload, clear or render).
    pub initialized: bool,
    /// The GPU wrote data newer than guest memory.
    pub gpu_modified: bool,
    pub last_use: u64,
    /// Content version: the cache's write sequence number of the last
    /// upload, clear, draw or copy that changed the contents. Of two
    /// overlapping images the one with the higher number holds the current
    /// bytes of the shared memory.
    pub seq: u64,
    /// Images whose guest planes overlap this one's.
    aliases: Vec<u64>,
    pool_key: PoolKey,
    /// Layout and access of the last use.
    pub layout: LayoutTracker,
    /// Hash of the guest bytes when the image was last loaded, for images
    /// within two pages (see [`ImageCache::revalidate`]).
    edge_hash: Option<u64>,
    /// Pages written into the image since its last whole load were merged
    /// (its memory may belong to the CPU: see `WatchEntry::touched`) and no
    /// GPU work wrote it since.
    merged: bool,
    /// Lookup generation (see [`ImageCache::image_generation`]).
    generation: u64,
    /// Bindless table slots holding a view of the image (see
    /// [`ImageCache::pin`]).
    pins: u32,
    /// `seq` at the last write-back (0: never): the buffer mirror of the
    /// image's memory holds these contents.
    written_back_seq: u64,
}

impl CachedImage {
    pub fn full_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.image.aspect,
            base_mip_level: 0,
            level_count: self.image.mip_levels,
            base_array_layer: 0,
            // All layers (and every depth slice of a 2D-array-compatible volume).
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        }
    }

    /// Guest range of the main plane.
    fn main_range(&self) -> (u64, u64) {
        (self.key.main.addr, self.key.main.addr + self.key.size())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ImageStats {
    pub images: usize,
    pub created: u64,
    pub uploads: u64,
    pub uploaded_bytes: u64,
    pub writebacks: u64,
    pub clears: u64,
    pub deleted: u64,
    /// Contents moved between images of the same surface on the GPU.
    pub conversions: u64,
    /// Images refilled from newer overlapping images through their guest
    /// layout on the GPU.
    pub retiles: u64,
    /// Conversions and retiles done as one image copy.
    pub direct_copies: u64,
    /// GPU-modified images that took the pages written into them since
    /// they were loaded, keeping their GPU contents elsewhere.
    pub merges: u64,
    /// Host images taken from the pool instead of being created.
    pub reused: u64,
    /// Uploads by reason: new image, CPU write, GPU write through a
    /// buffer, unwatched range.
    pub upload_reasons: [u64; 5],
    /// Refreshes skipped because the caller overwrites the whole image.
    pub discards: u64,
    /// Images of fewer layers of an array surface merged into a new image
    /// of more layers.
    pub absorbed: u64,
    /// Multisampled depth images rebuilt from colour images holding their
    /// samples.
    pub reinterpretations: u64,
    /// Memory-pressure collection runs, and the images they dropped.
    pub pressure_runs: u64,
    /// Write-backs skipped because a newer image holds the memory.
    pub superseded_writebacks: u64,
    /// Stale GPU-modified images that merged their CPU-written pages and
    /// kept their content version before a refill from newer aliases.
    pub kept_merges: u64,
    /// Depth/stencil or multisampled images written back before their
    /// refill through guest memory.
    pub own_writebacks: u64,
    /// Write-backs before GPU work wrote the image's memory as a buffer.
    pub buffer_write_writebacks: u64,
    pub pressure_drops: u64,
    /// Linear images copied back to guest memory.
    pub readbacks: u64,
    /// Unwatched images found unchanged by their edge hash (not reloaded).
    pub revalidations: u64,
    /// Images dropped by reason: idle clean alias under a new image,
    /// superseded, over budget (clean), unmapped, over budget after a
    /// write-back of GPU-modified contents; then GPU-modified images over
    /// budget that could not be written back (kept).
    pub drops: [u64; 6],
    /// Guest bytes of all images / of GPU-modified images (last collect).
    pub bytes: u64,
    pub modified_bytes: u64,
}

/// Hash of the bytes of a small image (64-bit words, multiply-rotate).
fn edge_hash_of(bytes: &[u8]) -> u64 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64 ^ bytes.len() as u64;
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        let w = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
        h = (h ^ w).wrapping_mul(0xFF51_AFD7_ED55_8CCD).rotate_left(29);
    }
    for &b in chunks.remainder() {
        h = (h ^ b as u64).wrapping_mul(0xC4CE_B9FE_1A85_EC53).rotate_left(17);
    }
    h ^ (h >> 33)
}

/// Image owners of each 1 MiB page of the 44-bit guest space.

type ImagePageTable = crate::page_table::MultiLevelPageTable<crate::page_table::InlinePageOwnerList<u64, 16>, 20, 44, 14>;

/// Recorded constant fill of a metadata surface (HTILE/CMASK/DCC).
#[derive(Clone, Copy, Debug)]
pub struct MetadataFill {
    pub size: u64,
    pub value: u32,
}

/// The pending metadata fills by address, with the largest fill size seen:
/// the fills overlapping a range start at most that far before it, so a
/// write finds them without looking at every fill.
#[derive(Default)]
struct PendingFills {
    map: BTreeMap<u64, MetadataFill>,
    max_size: u64,
}

impl PendingFills {
    fn insert(&mut self, addr: u64, fill: MetadataFill) {
        self.max_size = self.max_size.max(fill.size);
        self.map.insert(addr, fill);
    }

    fn remove(&mut self, addr: &u64) -> Option<MetadataFill> {
        self.map.remove(addr)
    }

    fn get(&self, addr: &u64) -> Option<&MetadataFill> {
        self.map.get(addr)
    }

    fn contains_key(&self, addr: &u64) -> bool {
        self.map.contains_key(addr)
    }

    fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Drop the fills starting in `[begin, end)`.
    fn remove_starting_in(&mut self, begin: u64, end: u64) {
        let gone: smallvec::SmallVec<[u64; 8]> = self.map.range(begin..end.max(begin)).map(|(&a, _)| a).collect();
        for a in gone {
            self.map.remove(&a);
        }
    }

    /// Drop the fills overlapping `[begin, end)`; true when one was.
    /// `indexed`: only the fills that can reach the range are looked at
    /// (else every fill).
    fn drop_overlapping(&mut self, begin: u64, end: u64, indexed: bool) -> bool {
        let before = self.map.len();
        if indexed {
            let lo = begin.saturating_sub(self.max_size);
            let gone: smallvec::SmallVec<[u64; 8]> =
                self.map.range(lo..end.max(lo)).filter(|(&a, f)| a.saturating_add(f.size) > begin).map(|(&a, _)| a).collect();
            for a in gone {
                self.map.remove(&a);
            }
        } else {
            self.map.retain(|&a, f| a.saturating_add(f.size) <= begin || a >= end);
        }
        self.map.len() != before
    }
}

/// Scheduler ticks without use after which an image that a newer image
/// fully overwrote, or a clean image a new image overlaps, is dropped. A
/// frame takes a few hundred ticks (one per EOP): images whose memory two
/// keys share within a frame (a texture viewed two ways, transient render
/// target pools) must outlive a frame, or each frame drops and re-uploads
/// them.
pub(crate) const IDLE_TICKS: u64 = 2048;
/// Presented frames an image must stay unused before the budget may
/// evict it: everything a frame (or a few alternating frames) binds stays
/// resident, so textures are not reloaded every frame when the working
/// set exceeds the budget.
const EVICT_FRAMES: usize = 16;
/// Frames of idleness that suffice once the cache exceeds its budget by a
/// quarter (the working set of the last [`EVICT_FRAMES`] frames is larger
/// than the budget): only images of the current frames stay.
const EVICT_FRAMES_OVER: usize = 4;
/// Ticks standing in for [`EVICT_FRAMES`] while no frames are presented
/// through the cache.
const EVICT_TICKS: u64 = 4096;
/// Upload staging ring.
const STAGING_BYTES: u64 = 64 << 20;

pub struct ImageCache {
    /// `RPS5E_GPU_PROFILE`: uploads per (address, format, width, height).
    upload_counts: HashMap<(u64, i32, u32, u32), u64>,
    /// `RPS5E_GPU_PROFILE`: refreshes from newer aliases per (image, newest
    /// source) — `(addr, format, w, h, layers)` each.
    retile_counts: HashMap<((u64, i32, u32, u32, u32), (u64, i32, u32, u32, u32), usize), u64>,
    /// `RPS5E_GPU_PROFILE`: stale prepares per (address, reason).
    stale_reasons: HashMap<(u64, &'static str), u64>,
    /// `RPS5E_GPU_PROFILE`: invalidations per GPU-written range.
    invalidate_counts: HashMap<(u64, u64), u64>,
    dev: Arc<GpuDevice>,
    memory: &'static GuestGpuMemory,
    pub tiler: GpuTiler,
    images: HashMap<u64, CachedImage>,
    /// Image of each key.
    by_key: HashMap<ImageKey, u64>,
    /// Every plane of every image: (address, id) -> end.
    planes: BTreeMap<(u64, u64), u64>,
    /// Images owning each 1 MiB guest page (overlap queries).
    page_owners: ImagePageTable,
    next_id: u64,
    /// Write sequence (see [`CachedImage::seq`]).
    seq: u64,
    /// Bumped by every change that can alter what a texture lookup
    /// (find_alias, prepare) returns or does (see [`Self::lookup_epoch`]).
    epoch: u64,
    /// Lookup generations handed out (see [`Self::image_generation`]).
    generations: u64,
    /// Bumped by [`Self::note_fill`].
    metadata_generation: u64,
    pub table: Arc<WatchTable>,
    /// Pending metadata fills by address (consumed by the next bind).
    metadata: PendingFills,
    /// The image each DCC key address read through a descriptor belongs
    /// to ([`sampled_dcc`]): a fill of those keys moves its generation, so
    /// memoised lookups of the image look at the keys again.
    sampled_dcc_owners: HashMap<u64, u64>,
    /// Images fully overwritten by a newer image: id -> version then.
    superseded: HashMap<u64, u64>,
    /// Scheduler tick at the start of each of the last presented frames
    /// (oldest first): the age of an image in frames for eviction.
    frame_ticks: std::collections::VecDeque<u64>,
    pool: ImagePool,
    staging: StagingRing,
    /// Packed linear data of the tiler.
    linear: Scratch,
    /// Guest-layout bytes of retiles.
    tiled: Scratch,
    /// 32-bit float depth of 16-bit guest depth (D16S8 fallback).
    wide: Scratch,
    stats: ImageStats,
    /// Subresources of every view of a cached image.
    view_ranges: HashMap<vk::ImageView, SubRange>,
    /// The views of the stencil aspect of a depth/stencil image.
    stencil_views: std::collections::HashSet<vk::ImageView>,
    /// Views made for images deleted before the view was cached (R7b).
    orphan_views: Vec<ImageView>,
    /// A cutoff before which an eviction pass found no idle image (see
    /// `evict_idle`).
    no_idle_below: u64,
    /// Barriers being collected (reused).
    barriers: Vec<vk::ImageMemoryBarrier2<'static>>,
    /// Colour to multisampled depth reinterpretation (created on first
    /// use; None inside when the device cannot run it).
    blit: Option<Option<blit::BlitHelper>>,
    /// Linear storage images shaders write are how results reach the CPU,
    /// which reads them without any GPU-visible request (a game spins until
    /// a read-back surface loses its sentinel): their pages block CPU
    /// accesses until one writes them back ([`LinearReadback`]).
    pub(crate) readback_linear: LinearReadback,
    /// Linear storage images written since the last collection.
    downloads: Vec<u64>,
    /// Tick of the last shader store to each read-watched linear image.
    storage_ticks: HashMap<u64, u64>,
    /// Pinned images whose use by a bindless table may need work again
    /// (see [`Self::pinned_since`]), from position `pinned_base` on.
    pinned_log: Vec<u64>,
    pinned_base: u64,
}

/// How GPU-written linear storage images reach guest memory
/// (`RPS5E_READBACK_LINEAR_IMAGES`): at every collection (default), on
/// the first CPU access to their pages (`access`), or not (`0`).
///
/// On access is not the default: guest memory read through the backing
/// alias (HLE services, the command processor, uploads of other
/// resources) does not fault, so such readers see stale bytes (ASTRO BOT
/// does not leave its title menu).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LinearReadback {
    Off,
    Eager,
    OnAccess,
}

/// Uses of an image: the layout it is in and the access.
pub type ImageUse = (vk::ImageLayout, vk::AccessFlags2);

/// Copy source.
pub const TRANSFER_SRC: ImageUse = (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags2::TRANSFER_READ);
/// Copy or clear destination.
pub const TRANSFER_DST: ImageUse = (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags2::TRANSFER_WRITE);
/// Colour attachment of a rendering scope.
pub const COLOR_ATTACHMENT: ImageUse = (
    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    vk::AccessFlags2::from_raw(vk::AccessFlags2::COLOR_ATTACHMENT_READ.as_raw() | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw()),
);
/// Depth/stencil attachment of a rendering scope (both aspects writable:
/// the attachment store writes even when the tests do not).
pub const DEPTH_ATTACHMENT: ImageUse = (
    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    vk::AccessFlags2::from_raw(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ.as_raw() | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()),
);
/// Depth/stencil attachment that is only read (tested), possibly also
/// sampled by the same draws.
pub const DEPTH_READ_ONLY: ImageUse = (
    vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
    vk::AccessFlags2::from_raw(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ.as_raw() | vk::AccessFlags2::SHADER_READ.as_raw()),
);
/// Depth/stencil attachment whose depth aspect is only read (tested and
/// sampled by the same draws) while the stencil aspect is written.
pub const DEPTH_SAMPLED_STENCIL_WRITE: ImageUse = (
    vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
    vk::AccessFlags2::from_raw(
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ.as_raw() | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw() | vk::AccessFlags2::SHADER_READ.as_raw(),
    ),
);
/// Sampled colour image.
pub const SAMPLED: ImageUse = (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::AccessFlags2::SHADER_READ);
/// Sampled depth/stencil image.
pub const SAMPLED_DEPTH: ImageUse = (vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL, vk::AccessFlags2::SHADER_READ);
/// Storage image (read and written by shaders).
pub const STORAGE: ImageUse =
    (vk::ImageLayout::GENERAL, vk::AccessFlags2::from_raw(vk::AccessFlags2::SHADER_READ.as_raw() | vk::AccessFlags2::SHADER_WRITE.as_raw()));

/// Order buffer accesses of a transfer (copies and the tiler) after every
/// earlier command: reads of data written before, and writes over data
/// read or written before.
fn before_buffer_work(sched: &mut Scheduler) {
    sched.memory_barrier(
        (vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_WRITE),
        (
            vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        ),
    );
}

/// Make buffer writes of a transfer visible to every later command.
fn after_buffer_writes(sched: &mut Scheduler) {
    sched.memory_barrier(
        (vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_WRITE),
        (vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE),
    );
}

/// Make tiler (compute) buffer writes visible to copies.
fn tiler_to_copy(sched: &mut Scheduler) {
    sched.memory_barrier((vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_WRITE), (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ));
}

/// Make tiler (compute) buffer writes visible to the next tiler pass.
fn compute_to_compute(sched: &mut Scheduler) {
    sched.memory_barrier(
        (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_WRITE),
        (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE),
    );
}

/// The colour transform of `plane` of `key` (none for depth and stencil
/// planes, and for elements the transform does not fit).
fn plane_transform(key: &ImageKey, plane: &Plane) -> ColorTransform {
    let t = key.transform;
    if plane.aspect != vk::ImageAspectFlags::COLOR || t.element_bytes() != plane.bytes_per_element || block_dim(key.format) != 1 {
        return ColorTransform::None;
    }
    t
}

/// The transform box of one linear-plan copy (rows `buffer_row_length`
/// elements apart, the same positions on both sides) and the bytes it
/// spans.
fn linear_box(c: &vk::BufferImageCopy, bytes_per_element: u32) -> (TransformRegion, u64) {
    let row = c.buffer_row_length.max(c.image_extent.width) * bytes_per_element;
    let slice = row * c.buffer_image_height.max(c.image_extent.height);
    let region = TransformRegion {
        width: c.image_extent.width,
        height: c.image_extent.height,
        slices: c.image_extent.depth.max(1),
        src_row: row,
        src_slice: slice,
        dst_row: row,
        dst_slice: slice,
    };
    let span = slice as u64 * (region.slices as u64 - 1) + row as u64 * (region.height as u64 - 1) + region.width as u64 * bytes_per_element as u64;
    (region, span)
}

/// Make copy buffer writes visible to the tiler (compute).

fn copy_to_tiler(sched: &mut Scheduler) {
    sched.memory_barrier(
        (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE),
        (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE),
    );
}

/// Texels per side of one element of `format` (4 for BC block formats).
pub fn block_dim(format: vk::Format) -> u32 {
    let r = format.as_raw();
    if (vk::Format::BC1_RGB_UNORM_BLOCK.as_raw()..=vk::Format::BC7_SRGB_BLOCK.as_raw()).contains(&r) {
        4
    } else {
        1
    }
}

/// The usage of an image of `format` and the create flags that usage needs.
/// sRGB formats have no storage support, but their storage views use the
/// UNORM format of the same layout (`storage_view_format`): such an image
/// gets STORAGE usage with EXTENDED_USAGE (the usage is checked against the
/// view formats) when the UNORM format supports it.
fn usage_and_flags_for(dev: &GpuDevice, format: vk::Format, depth: bool, samples: u32) -> (vk::ImageUsageFlags, vk::ImageCreateFlags) {
    let mut usage = usage_for(dev, format, depth, samples);
    let alias = crate::pipeline::sharps::storage_view_format(format);
    if !depth && samples <= 1 && alias != format && !usage.contains(vk::ImageUsageFlags::STORAGE) {
        // SAFETY: plain query.
        let props = unsafe { dev.instance.get_physical_device_format_properties(dev.phys, alias) }.optimal_tiling_features;
        if props.contains(vk::FormatFeatureFlags::STORAGE_IMAGE) {
            usage |= vk::ImageUsageFlags::STORAGE;
            return (usage, vk::ImageCreateFlags::EXTENDED_USAGE);
        }
    }
    (usage, vk::ImageCreateFlags::empty())
}

fn usage_for(dev: &GpuDevice, format: vk::Format, depth: bool, samples: u32) -> vk::ImageUsageFlags {
    // SAFETY: plain query.
    let props = unsafe { dev.instance.get_physical_device_format_properties(dev.phys, format) }.optimal_tiling_features;
    let mut u = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    if props.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE) {
        u |= vk::ImageUsageFlags::SAMPLED;
    }
    if depth {
        if props.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT) {
            u |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
    } else {
        if props.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT) {
            u |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        // Multisampled storage images need a feature the device does not enable.
        if props.contains(vk::FormatFeatureFlags::STORAGE_IMAGE) && samples <= 1 {
            u |= vk::ImageUsageFlags::STORAGE;
        }
    }
    u
}

/// True when the device can create an image of this shape.
fn image_supported(dev: &GpuDevice, ty: vk::ImageType, format: vk::Format, usage: vk::ImageUsageFlags, flags: vk::ImageCreateFlags, key: &ImageKey) -> bool {
    // SAFETY: plain query.
    let Ok(p) = (unsafe { dev.instance.get_physical_device_image_format_properties(dev.phys, format, ty, vk::ImageTiling::OPTIMAL, usage, flags) }) else {
        return false;
    };
    p.max_extent.width >= key.width
        && p.max_extent.height >= key.height
        && p.max_extent.depth >= key.depth
        && p.max_array_layers >= key.layers
        && p.max_mip_levels >= key.levels
        && p.sample_counts.contains(samples_flag(key.samples))
}

/// Mip levels a host image of this extent can hold.
pub fn max_levels(width: u32, height: u32, depth: u32) -> u32 {
    32 - width.max(height).max(depth).max(1).leading_zeros()
}

fn samples_flag(n: u32) -> vk::SampleCountFlags {
    match n {
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        _ => vk::SampleCountFlags::TYPE_1,
    }
}

/// `[b, e)` lies inside `[ob, oe)`.
/// `RPS5E_ALIAS_MERGE=0`: a GPU-modified image the CPU wrote any page of
/// is no source for refilling its aliases (they keep older contents).
fn alias_merge_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_ALIAS_MERGE"))
}

/// `RPS5E_LOAD_RACE_CHECK=0`: a load clears the image's CPU-written marks
/// even when a write was counted while guest memory was being read.
fn read_race_check_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_LOAD_RACE_CHECK"))
}

/// `RPS5E_SHADER_WRITE_INVALIDATE=base` (see
/// `ImageCache::invalidate_shader_write`).
fn shader_write_rule_base() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_SHADER_WRITE_INVALIDATE").is_ok_and(|v| v == "base"))
}

/// `RPS5E_IMAGE_GENERATIONS=0`: a change to one image (a render target
/// written, an image refreshed, created or cleared) invalidates every
/// memoised texture lookup, not only those of the image and its aliases.
pub fn image_generations_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_IMAGE_GENERATIONS"))
}

/// `RPS5E_INDEXED_INVALIDATE=0`: a GPU write through a buffer looks at
/// every image's watch entry instead of those the page index names.
fn indexed_invalidate_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_INDEXED_INVALIDATE"))
}

/// `x` stores the texel blocks of block-compressed `y` as uncompressed
/// texels of the same size in the same guest layout (a compute pass
/// writing compressed blocks through a storage view): one level, one
/// sample, no stencil, no colour transform, the same memory, swizzle,
/// layers and element grid, and an equal surface layout.
fn block_view_of(x: &ImageKey, y: &ImageKey) -> bool {
    let bd = block_dim(y.format);
    bd > 1
        && block_dim(x.format) == 1
        && x.levels == 1
        && y.levels == 1
        && x.samples == 1
        && y.samples == 1
        && x.dim == ImageDim::D2
        && y.dim == ImageDim::D2
        && x.stencil.is_none()
        && y.stencil.is_none()
        && x.transform == ColorTransform::None
        && y.transform == ColorTransform::None
        && x.main.aspect == vk::ImageAspectFlags::COLOR
        && x.main.addr == y.main.addr
        && x.main.swizzle == y.main.swizzle
        && x.main.bytes_per_element == y.main.bytes_per_element
        && x.layers == y.layers
        && x.depth == 1
        && y.depth == 1
        && y.width % bd == 0
        && y.height % bd == 0
        && x.width == y.width / bd
        && x.height == y.height / bd
        && x.layout(&x.main).is_some_and(|l| y.layout(&y.main) == Some(l))
}

/// `RPS5E_IMAGE_DIRECT_COPY=0`: format conversions between images of one
/// memory go through a buffer (image to buffer to image) and an image
/// refreshed from a newer one storing its blocks as uncompressed texels
/// goes through the guest layout, instead of one image copy.
fn direct_copy_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_IMAGE_DIRECT_COPY"))
}

/// vkCmdCopyImage copies the texels of a `src` image into a `dst` image
/// bit for bit (texel blocks of equal size are the caller's part): two
/// colour images, one format and its own aspect, or with `maintenance8` a
/// depth image and a colour format the specification pairs it with.
fn copy_compatible(maintenance8: bool, (sf, sa): (vk::Format, vk::ImageAspectFlags), (df, da): (vk::Format, vk::ImageAspectFlags)) -> bool {
    use vk::Format as F;
    let color = vk::ImageAspectFlags::COLOR;
    let depth_pairs = |d: vk::Format, c: vk::Format| match d {
        F::D16_UNORM => matches!(c, F::R16_SFLOAT | F::R16_UNORM | F::R16_SNORM | F::R16_UINT | F::R16_SINT),
        F::D32_SFLOAT => matches!(c, F::R32_SFLOAT | F::R32_SINT | F::R32_UINT),
        _ => false,
    };
    match (sa == color, da == color) {
        (true, true) => true,
        _ if sf == df && sa == da => true,
        (false, true) => maintenance8 && sa == vk::ImageAspectFlags::DEPTH && depth_pairs(sf, df),
        (true, false) => maintenance8 && da == vk::ImageAspectFlags::DEPTH && depth_pairs(df, sf),
        (false, false) => false,
    }
}

fn inside(b: u64, e: u64, ob: u64, oe: u64) -> bool {
    b >= ob && e <= oe
}

impl ImageCache {
    pub fn new(dev: &Arc<GpuDevice>, memory: &'static GuestGpuMemory) -> Result<ImageCache, vk::Result> {
        let scratch = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        Ok(ImageCache {
            upload_counts: HashMap::default(),
            retile_counts: HashMap::default(),
            invalidate_counts: HashMap::default(),
            stale_reasons: HashMap::default(),
            dev: dev.clone(),
            memory,
            tiler: GpuTiler::new(dev)?,
            images: HashMap::default(),
            by_key: HashMap::default(),
            planes: BTreeMap::new(),
            page_owners: ImagePageTable::new(),
            next_id: 1,
            seq: 0,
            epoch: 0,
            generations: 0,
            metadata_generation: 0,
            table: Arc::new(WatchTable::with_memory(memory)),
            metadata: PendingFills::default(),
            sampled_dcc_owners: HashMap::default(),
            superseded: HashMap::default(),
            frame_ticks: std::collections::VecDeque::with_capacity(EVICT_FRAMES + 1),
            pool: ImagePool::default(),
            staging: StagingRing::new(dev, STAGING_BYTES, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER, MemoryUse::Upload)?,
            linear: Scratch::new(scratch),
            tiled: Scratch::new(scratch),
            wide: Scratch::new(scratch),
            stats: ImageStats::default(),
            view_ranges: HashMap::default(),
            stencil_views: Default::default(),
            orphan_views: Vec::new(),
            no_idle_below: 0,
            pinned_log: Vec::new(),
            pinned_base: 0,
            barriers: Vec::new(),
            blit: None,
            // RPS5E_SAFE: every linear storage image is read back at
            // every collection, whatever the setting.
            readback_linear: match std::env::var("RPS5E_READBACK_LINEAR_IMAGES").as_deref() {
                _ if crate::backend::ab::safe() => LinearReadback::Eager,
                Ok("0") | Ok("off") => LinearReadback::Off,
                Ok("access") => LinearReadback::OnAccess,
                _ => LinearReadback::Eager,
            },
            downloads: Vec::new(),
            storage_ticks: HashMap::default(),
        })
    }

    pub fn stats(&self) -> ImageStats {
        ImageStats { images: self.images.len(), reused: self.pool.reused, ..self.stats }
    }

    /// Changes when anything a texture lookup depends on may have changed:
    /// images created, deleted, filled, converted or cleared, GPU writes
    /// that change which alias holds the newest contents, CPU writes and
    /// buffer writes over images. A lookup made at one value (and that
    /// needed no upload) gives the same image and view at the same value.
    pub fn lookup_epoch(&self) -> u64 {
        self.epoch.wrapping_add(self.table.changes.load(std::sync::atomic::Ordering::Acquire))
    }

    /// The lookup generation of image `id` (None: no such image). A texture
    /// lookup that found `id` (and needed no upload) at one generation and
    /// [`Self::lookup_epoch`] finds it again, doing nothing, while both are
    /// unchanged: it moves whenever the image or an image overlapping it is
    /// created, deleted, filled, refreshed, cleared or written by GPU work
    /// that changes which of them holds the newest contents. Changes that
    /// can affect any lookup (CPU writes and buffer writes over images,
    /// write-backs, unmaps) move the lookup epoch instead.
    pub fn image_generation(&self, id: u64) -> Option<u64> {
        self.images.get(&id).map(|i| i.generation)
    }

    /// Changes whenever any lookup result may change ([`Self::lookup_epoch`]
    /// or any image generation): a lookup that left it unchanged changed
    /// nothing.
    pub fn lookup_stamp(&self) -> (u64, u64) {
        (self.lookup_epoch(), self.generations)
    }

    /// The global image generation (see [`Self::lookup_stamp`]).
    pub fn generations_value(&self) -> u64 {
        self.generations
    }

    /// Lookups that find `id` or an image overlapping it may give another
    /// result now: their generations move. Without per-image generations
    /// (`RPS5E_IMAGE_GENERATIONS=0`) every lookup is affected.
    fn changed(&mut self, id: u64) {
        if crate::slice_shadow::enabled() {
            crate::slice_shadow::hole(crate::slice_shadow::Hole::ImageGeneration);
            crate::slice_shadow::changed_image(id, self.guest_range(id));
        }
        self.generations += 1;
        let g = self.generations;
        if !image_generations_enabled() {
            self.epoch += 1;
        }
        let Some(i) = self.images.get_mut(&id) else { return };
        i.generation = g;
        let mut pinned = i.pins != 0;
        let aliases = std::mem::take(&mut i.aliases);
        for a in &aliases {
            if let Some(o) = self.images.get_mut(a) {
                o.generation = g;
                pinned |= o.pins != 0;
            }
        }
        if pinned {
            self.note_pinned(id);
            for &a in &aliases {
                if self.images.get(&a).is_some_and(|o| o.pins != 0) {
                    self.note_pinned(a);
                }
            }
        }
        if let Some(i) = self.images.get_mut(&id) {
            i.aliases = aliases;
        }
    }

    /// Bindless table slots keep a view of image `id` across draws: what
    /// may change the work a use of it needs (another layout or access, a
    /// new generation, a deletion) is logged for them ([`Self::pinned_since`]),
    /// and eviction leaves it alone. Every pin is undone by [`Self::unpin`].
    pub fn pin(&mut self, id: u64) {
        if let Some(i) = self.images.get_mut(&id) {
            i.pins += 1;
            if i.pins == 1 {
                self.table.set_pinned(id, true);
            }
        }
    }

    pub fn unpin(&mut self, id: u64) {
        if let Some(i) = self.images.get_mut(&id) {
            i.pins = i.pins.saturating_sub(1);
            if i.pins == 0 {
                self.table.set_pinned(id, false);
            }
        }
    }

    /// The log position after the last change of a pinned image.
    pub fn pinned_cursor(&self) -> u64 {
        self.pinned_base + self.pinned_log.len() as u64
    }

    /// Pinned images changed since log position `cursor` (an image may
    /// repeat). None when the log no longer reaches back that far: any
    /// pinned image may have changed.
    pub fn pinned_since(&self, cursor: u64) -> Option<&[u64]> {
        let start = cursor.checked_sub(self.pinned_base)?;
        self.pinned_log.get(start as usize..)
    }

    /// Changes of pinned images' guest memory seen by the watch table
    /// (CPU writes, GPU buffer writes, mappings): a lookup may load such an
    /// image again.
    pub fn pinned_memory_changes(&self) -> u64 {
        self.table.pinned_changes()
    }

    /// The pinned images whose guest memory changed since
    /// [`Self::pinned_memory_changes`] was `since` (appended to `out`, an
    /// image may repeat), and the count they reach; None when they are no
    /// longer known (any pinned image may have changed).
    pub fn pinned_memory_changed_since(&self, since: u64, out: &mut Vec<u64>) -> Option<u64> {
        self.table.pinned_changed_since(since, out)
    }

    fn note_pinned(&mut self, id: u64) {
        const LOG: usize = 1 << 14;
        if self.pinned_log.len() >= LOG {
            self.pinned_log.drain(..LOG / 2);
            self.pinned_base += (LOG / 2) as u64;
        }
        self.pinned_log.push(id);
    }

    /// Mark an image used by `tick` (a memoised lookup).
    pub fn touch(&mut self, id: u64, tick: u64) {
        if let Some(i) = self.images.get_mut(&id) {
            i.last_use = tick;
        }
    }

    pub fn get(&self, id: u64) -> Option<&CachedImage> {
        self.images.get(&id)
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut CachedImage> {
        self.images.get_mut(&id)
    }

    /// The guest range `[begin, end)` of image `id`'s main plane.
    pub fn guest_range(&self, id: u64) -> Option<(u64, u64)> {
        self.images.get(&id).map(|i| i.main_range())
    }

    /// Append the barriers taking `range` (None: all) of image `id` into
    /// `layout` for `access` (see [`layout::LayoutTracker::barriers`]).
    pub fn transition(&mut self, id: u64, (layout, access): ImageUse, range: Option<SubRange>, out: &mut Vec<vk::ImageMemoryBarrier2<'static>>) {
        if let Some(i) = self.images.get_mut(&id) {
            let (image, aspect) = (i.image.image, i.image.aspect);
            i.layout.barriers(image, aspect, layout, access, stages_for(access), range, out);
            // A use other than sampling leaves the state a bindless slot
            // expects.
            if i.pins != 0 && !(access == vk::AccessFlags2::SHADER_READ && matches!(layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL | vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)) {
                self.note_pinned(id);
            }
        }
    }

    /// The layout every subresource of `range` of image `id` is in (None
    /// when they differ or the image is unknown).
    pub fn uniform_layout(&self, id: u64, range: SubRange) -> Option<vk::ImageLayout> {
        let t = &self.images.get(&id)?.layout;
        if t.is_uniform() {
            return Some(t.state().layout);
        }
        let first = t.subresource_state(range.base_level, range.base_layer).layout;
        for level in range.base_level..range.base_level + range.levels.max(1) {
            for layer in range.base_layer..range.base_layer + range.layers.max(1) {
                if t.subresource_state(level, layer).layout != first {
                    return None;
                }
            }
        }
        Some(first)
    }

    /// Take `range` (None: all) of image `id` into `layout` for `access`
    /// now (outside any rendering scope).
    pub fn transit(&mut self, sched: &mut Scheduler, id: u64, usage: ImageUse, range: Option<SubRange>) {
        let mut v = std::mem::take(&mut self.barriers);
        v.clear();
        self.transition(id, usage, range, &mut v);
        sched.image_barriers(&v);
        self.barriers = v;
    }

    /// Two images at once (copies between them).
    fn transit_pair(&mut self, sched: &mut Scheduler, src: (u64, Option<SubRange>), dst: (u64, Option<SubRange>)) {
        let mut v = std::mem::take(&mut self.barriers);
        v.clear();
        self.transition(src.0, TRANSFER_SRC, src.1, &mut v);
        self.transition(dst.0, TRANSFER_DST, dst.1, &mut v);
        sched.image_barriers(&v);
        self.barriers = v;
    }

    /// The levels and layers of a view of a cached image (depth slices
    /// for 2D views of a volume).
    pub fn is_stencil_view(&self, view: vk::ImageView) -> bool {
        self.stencil_views.contains(&view)
    }

    /// The levels and layers of a view of a cached image (depth slices
    /// for 2D views of a volume).
    pub fn view_subresources(&self, view: vk::ImageView) -> Option<SubRange> {
        self.view_ranges.get(&view).copied()
    }

    /// Images with a plane overlapping `[begin, end)`, except `except`.
    fn overlapping(&self, begin: u64, end: u64, except: u64) -> Vec<u64> {
        let mut v = Vec::new();
        let Some(pages) = ImagePageTable::page_range(begin, end.saturating_sub(begin)) else { return v };
        for page in pages {
            let Some(owners) = self.page_owners.find(page) else { continue };
            for id in owners.iter() {
                if id == except || v.contains(&id) {
                    continue;
                }
                if self.images.get(&id).is_some_and(|i| i.key.planes().any(|p| p.addr < end && begin < p.addr + p.size())) {
                    v.push(id);
                }
            }
        }
        v.sort_unstable();
        v
    }

    /// Write the GPU-modified images over `[addr, addr+size)` back to guest
    /// memory (oldest first, so newer contents land last), for transfers
    /// that read the range as memory (command-processor copies). Returns
    /// how many were written back.
    pub fn writeback_range(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, addr: u64, size: u64) -> usize {
        let mut ids: Vec<(u64, u64)> = self
            .overlapping(addr, addr + size, u64::MAX)
            .into_iter()
            .filter_map(|id| self.images.get(&id))
            .filter(|i| i.gpu_modified && i.initialized && self.table.may_write_back(i.id))
            .map(|i| (i.seq, i.id))
            .collect();
        ids.sort_unstable();
        let mut n = 0;
        for (_, id) in ids {
            if self.writeback(sched, buffers, id) {
                n += 1;
            }
        }
        n
    }

    /// Before GPU work writes `[addr, addr+size)` through a buffer: the
    /// images over the range whose contents the GPU changed since their
    /// last write-back are written back first, on the GPU, so the buffer's
    /// stores land over their bytes. The write then makes the images stale
    /// (`invalidate_shader_write`) and they take the memory back, keeping
    /// both the image's results and the buffer's stores (a partial store
    /// would otherwise drop the rest of the image). Returns how many were
    /// written back. `RPS5E_WRITEBACK_BEFORE_BUFFER_WRITE=0` turns it off.
    pub fn writeback_before_buffer_write(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, addr: u64, size: u64) -> usize {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| !std::env::var("RPS5E_WRITEBACK_BEFORE_BUFFER_WRITE").is_ok_and(|v| v == "0")) || size == 0 {
            return 0;
        }
        let mut ids: Vec<(u64, u64)> = self
            .overlapping(addr, addr.saturating_add(size), u64::MAX)
            .into_iter()
            .filter_map(|id| self.images.get(&id))
            .filter(|i| i.gpu_modified && i.initialized && i.seq != i.written_back_seq && self.table.may_write_back(i.id))
            .map(|i| (i.seq, i.id))
            .collect();
        if ids.is_empty() {
            return 0;
        }
        ids.sort_unstable();
        let mut n = 0;
        for (_, id) in ids {
            if self.writeback(sched, buffers, id) {
                n += 1;
            }
        }
        self.stats.buffer_write_writebacks += n as u64;
        n
    }

    /// An image over `[addr, addr+size)` would be written back by
    /// [`Self::writeback_before_buffer_write`].
    pub fn has_pending_writeback(&self, addr: u64, size: u64) -> bool {
        self.overlapping(addr, addr.saturating_add(size), u64::MAX)
            .into_iter()
            .filter_map(|id| self.images.get(&id))
            .any(|i| i.gpu_modified && i.initialized && i.seq != i.written_back_seq && self.table.may_write_back(i.id))
    }

    fn register_pages(&mut self, id: u64, p: &Plane) {
        if let Some(pages) = ImagePageTable::page_range(p.addr, p.size()) {
            for page in pages {
                if let Some(owners) = self.page_owners.get_or_create(page) {
                    if !owners.contains(id) {
                        owners.push(id);
                    }
                }
            }
        }
    }

    fn unregister_pages(&mut self, id: u64, p: &Plane) {
        if let Some(pages) = ImagePageTable::page_range(p.addr, p.size()) {
            for page in pages {
                if let Some(owners) = self.page_owners.find_mut(page) {
                    owners.erase(id);
                }
            }
        }
    }

    /// Images whose main plane starts at `addr`.
    fn at(&self, addr: u64) -> impl Iterator<Item = &CachedImage> {
        self.planes.range((addr, 0)..(addr + 1, 0)).filter_map(move |(&(_, id), _)| self.images.get(&id).filter(|i| i.key.main.addr == addr))
    }

    /// A cached image the texture `key` can be viewed through without a
    /// copy: same address, extent, shape, element size, tiling and sample
    /// count, at least the layers and the levels `levels_needed` (the view
    /// reinterprets the format; depth images are sampled through their
    /// depth aspect). Of several (the image of the exact key included), the
    /// one with the newest contents: a texture view of a depth buffer the
    /// GPU keeps rendering to reads the depth image itself instead of
    /// refilling its own copy after every draw.
    pub fn find_alias(&self, key: &ImageKey, levels_needed: u32) -> Option<u64> {
        let exact = self.by_key.get(key).and_then(|id| self.images.get(id));
        let mut candidates = self
            .at(key.main.addr)
            .filter(|i| {
                i.key.width == key.width
                    && i.key.height == key.height
                    && i.key.depth == key.depth
                    && i.key.dim == key.dim
                    && i.key.samples == key.samples
                    && i.key.layers >= key.layers
                    && i.image.mip_levels >= levels_needed.max(1)
                    && i.key.main.bytes_per_element == key.main.bytes_per_element
                    && i.key.main.swizzle == key.main.swizzle
                    && block_dim(i.key.format) == block_dim(key.format)
                    && i.key.transform == key.transform
            })
            .chain(exact)
            .peekable();
        candidates.peek()?;
        // Contents the CPU overwrote since are not current (the watch
        // table is locked once for all candidates).
        let alias = self.table.with_stale_whole(|stale_whole| candidates.max_by_key(|i| (i.initialized && !stale_whole(i.id), i.seq, i.gpu_modified)));
        alias.map(|i| i.id)
    }

    /// A depth/stencil image whose stencil plane is the surface of `key`
    /// (an 8-bit texture over a stencil buffer: deferred renderers read
    /// the material bits the geometry passes wrote into stencil), when its
    /// contents are newer than every image at that address. The stencil
    /// lives only in that image's stencil aspect; an image loaded from the
    /// plane's guest memory would hold whatever was there before.
    pub fn stencil_owner(&self, key: &ImageKey) -> Option<u64> {
        if key.is_depth() || key.main.bytes_per_element != 1 || key.dim != ImageDim::D2 {
            return None;
        }
        let addr = key.main.addr;
        let owner = self
            .planes
            .range((addr, 0)..(addr + 1, 0))
            .filter_map(|(&(_, id), _)| self.images.get(&id))
            .filter(|i| {
                i.key.stencil.is_some_and(|s| s.addr == addr && s.swizzle == key.main.swizzle)
                    && i.key.width == key.width
                    && i.key.height == key.height
                    && i.key.samples == key.samples
                    && i.key.layers >= key.layers
                    && i.initialized
                    && !self.table.stale_whole(i.id)
            })
            .max_by_key(|i| i.seq)?;
        let newer_here = self.at(addr).any(|i| i.initialized && !self.table.stale_whole(i.id) && i.seq > owner.seq);
        (!newer_here).then_some(owner.id)
    }

    /// The cached image for `key` (created when missing). Overlapping
    /// images stay; the new image takes their newer GPU contents when it is
    /// first used ([`ImageCache::prepare`]). Idle clean images under it are
    /// dropped (they reload from guest memory if used again).
    pub fn find_or_create(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, key: &ImageKey) -> Result<u64, vk::Result> {
        if let Some(&id) = self.by_key.get(key) {
            return Ok(id);
        }
        // An image of the same array surface with more layers holds this
        // one: one image per surface, not one per layer count (views of 1,
        // 2, 3 and 4 layers of one array were refilled from each other at
        // every use).
        if let Some(id) = self.containing_layers(key) {
            return Ok(id);
        }
        let id = self.create(sched, key)?;
        crate::slice_shadow::hole(crate::slice_shadow::Hole::ImageCreate);
        self.changed(id);
        // Images of fewer layers of this surface: their contents move into
        // the new image (from their GPU contents when newer than guest
        // memory) and they go.
        let within: Vec<u64> = self.images[&id].aliases.iter().copied().filter(|a| self.images.get(a).is_some_and(|o| o.key.layers_within(key))).collect();
        if !within.is_empty() {
            self.prepare(sched, buffers, id, false);
            for o in within {
                self.delete(sched, o);
                self.stats.absorbed += 1;
            }
        }
        let tick = sched.current_tick();
        let idle: Vec<u64> = self.images[&id]
            .aliases
            .iter()
            .filter(|a| self.images.get(a).is_some_and(|o| !o.gpu_modified && o.last_use + IDLE_TICKS < tick))
            .copied()
            .collect();
        for o in idle {
            self.delete(sched, o);
            self.stats.drops[0] += 1;
        }
        Ok(id)
    }

    /// An image whose first layers are the array surface of `key`.
    fn containing_layers(&self, key: &ImageKey) -> Option<u64> {
        if key.dim == ImageDim::D3 {
            return None;
        }
        let end = key.main.addr + key.size();
        self.overlapping(key.main.addr, end, u64::MAX).into_iter().find(|id| self.images.get(id).is_some_and(|i| key.layers_within(&i.key)))
    }

    /// Create the image of `key` (not yet filled).
    fn create(&mut self, sched: &mut Scheduler, key: &ImageKey) -> Result<u64, vk::Result> {
        let depth = key.is_depth();
        let (mut usage, extended) = usage_and_flags_for(&self.dev, key.format, depth, key.samples);
        let ty = match key.dim {
            ImageDim::D1 => vk::ImageType::TYPE_1D,
            ImageDim::D2 => vk::ImageType::TYPE_2D,
            ImageDim::D3 => vk::ImageType::TYPE_3D,
        };
        let mut flags = if depth { vk::ImageCreateFlags::empty() } else { vk::ImageCreateFlags::MUTABLE_FORMAT | extended };
        if key.dim == ImageDim::D2 && key.width == key.height && key.layers >= 6 && key.samples == 1 {
            flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }
        if key.dim == ImageDim::D3 && block_dim(key.format) == 1 {
            flags |= vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE;
        }
        if !image_supported(&self.dev, ty, key.format, usage, flags, key) {
            // Storage and attachment uses are optional; retry without them.
            usage &= !(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT);
            if !image_supported(&self.dev, ty, key.format, usage, flags, key) {
                // So are the optional create flags: a driver may reject the
                // combination even though it accepts each flag alone.
                flags &= !(vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::CUBE_COMPATIBLE | vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE | vk::ImageCreateFlags::EXTENDED_USAGE);
                if !image_supported(&self.dev, ty, key.format, usage, flags, key) {
                    return Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED);
                }
            }
        }
        let extent = vk::Extent3D { width: key.width, height: if key.dim == ImageDim::D1 { 1 } else { key.height }, depth: key.depth.max(1) };
        let pool_key = PoolKey {
            ty,
            format: key.format,
            extent: (extent.width, extent.height, extent.depth),
            levels: key.levels.max(1),
            layers: key.layers,
            samples: samples_flag(key.samples),
            usage,
            flags,
        };
        let image = match self.pool.take(sched, &pool_key) {
            Some(i) => i,
            None => {
                let i = Image::new_typed(&self.dev, ty, key.format, extent, key.levels.max(1), key.layers, samples_flag(key.samples), usage, flags)?;
                self.stats.created += 1;
                i
            }
        };
        let id = self.next_id;
        self.next_id += 1;
        // Contents are undefined (a pooled image's are discarded): the first
        // use transitions it from UNDEFINED.
        let layout = LayoutTracker::new(image.mip_levels, image.layers, key.dim == ImageDim::D3);
        self.table.insert(id, key.main.addr, key.main.addr + key.size());
        let mut aliases = Vec::new();
        for p in key.planes() {
            aliases.extend(self.overlapping(p.addr, p.addr + p.size(), id));
        }
        aliases.sort_unstable();
        aliases.dedup();
        for a in &aliases {
            if let Some(o) = self.images.get_mut(a) {
                o.aliases.push(id);
            }
        }
        for p in key.planes() {
            self.planes.insert((p.addr, id), p.addr + p.size());
            self.register_pages(id, &p);
        }
        self.by_key.insert(*key, id);
        self.images.insert(
            id,
            CachedImage {
                id,
                key: *key,
                image,
                views: HashMap::default(),
                extra_views: Vec::new(),
                last_view: None,
                initialized: false,
                gpu_modified: false,
                last_use: sched.current_tick(),
                seq: 0,
                aliases,
                pool_key,
                layout,
                edge_hash: None,
                merged: false,
                generation: 0,
                pins: 0,
                written_back_seq: 0,
            },
        );
        Ok(id)
    }

    /// Copy the contents of `src` into `dst` (same geometry, other host
    /// format) through a buffer; `dst` takes over the GPU-modified state
    /// and the content version.
    fn convert(&mut self, sched: &mut Scheduler, src: u64, dst: u64) -> bool {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::IMG_CONVERT);
        let r = self.convert_timed(sched, src, dst);
        sched.time_item_end(t);
        r
    }

    fn convert_timed(&mut self, sched: &mut Scheduler, src: u64, dst: u64) -> bool {
        let (Some(s), Some(d)) = (self.images.get(&src), self.images.get(&dst)) else { return false };
        let key = s.key;
        let (si, di) = (s.image.image, d.image.image);
        let (s_aspect, d_aspect) = (key.main.aspect, d.key.main.aspect);
        let levels = s.image.mip_levels.min(d.image.mip_levels);
        if key.samples > 1 {
            return false;
        }
        let bpe = key.main.bytes_per_element as u64;
        let bd = block_dim(key.format);
        // Elements are re-encoded when the two images store them through
        // different transforms (from the source's host format to the
        // stored bits, then into the destination's).
        let transforms = [(key.transform, false), (d.key.transform, true)];
        let recode = key.transform != d.key.transform;
        if recode && (bd != 1 || transforms.iter().any(|&(t, _)| t != ColorTransform::None && t.element_bytes() as u64 != bpe)) {
            return false;
        }
        let mut regions = Vec::new();
        let mut boxes = Vec::new();
        let mut offset = 0u64;
        for level in 0..levels {
            let e = key.level_extent(level);
            let layers = if key.dim == ImageDim::D3 { 1 } else { key.layers };
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(offset)
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: s_aspect, mip_level: level, base_array_layer: 0, layer_count: layers })
                    .image_extent(e),
            );
            boxes.push((offset, TransformRegion::packed(e.width, e.height, e.depth * layers, bpe as u32)));
            offset += e.width.div_ceil(bd) as u64 * e.height.div_ceil(bd) as u64 * e.depth as u64 * layers as u64 * bpe;
            offset = offset.next_multiple_of(16);
        }
        if !recode && key.transform == ColorTransform::None && direct_copy_enabled() && copy_compatible(self.dev.caps.maintenance8, (key.format, s_aspect), (d.key.format, d_aspect)) {
            // The same bytes as through the buffer, in one copy.
            let copies: Vec<vk::ImageCopy> = regions
                .iter()
                .map(|r| {
                    vk::ImageCopy::default()
                        .src_subresource(r.image_subresource)
                        .dst_subresource(vk::ImageSubresourceLayers { aspect_mask: d_aspect, ..r.image_subresource })
                        .extent(r.image_extent)
                })
                .collect();
            self.transit_pair(sched, (src, None), (dst, None));
            // SAFETY: recording buffer; source in TRANSFER_SRC, destination in TRANSFER_DST layout; the formats are size-compatible.
            unsafe { self.dev.device.cmd_copy_image(sched.ordered_cmd(), si, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, di, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &copies) };
            self.stats.direct_copies += 1;
        } else {
            let Some(tmp) = self.linear.get(&self.dev, sched, offset) else { return false };
            self.convert_through(sched, (src, si), (dst, di, d_aspect), tmp, &regions, &boxes, transforms, recode);
        }
        self.converted(sched, src, dst, key);
        true
    }

    /// The buffer path of [`Self::convert`].
    #[allow(clippy::too_many_arguments)]
    fn convert_through(
        &mut self,
        sched: &mut Scheduler,
        (src, si): (u64, vk::Image),
        (dst, di, d_aspect): (u64, vk::Image, vk::ImageAspectFlags),
        tmp: vk::Buffer,
        regions: &[vk::BufferImageCopy],
        boxes: &[(u64, TransformRegion)],
        transforms: [(ColorTransform, bool); 2],
        recode: bool,
    ) {
        let back: Vec<vk::BufferImageCopy> = regions
            .iter()
            .map(|r| {
                let mut r = *r;
                r.image_subresource.aspect_mask = d_aspect;
                r
            })
            .collect();
        before_buffer_work(sched);
        self.transit_pair(sched, (src, None), (dst, None));
        let cb = sched.ordered_cmd();
        // SAFETY: recording buffer; source in TRANSFER_SRC, destination in TRANSFER_DST layout.
        unsafe { self.dev.device.cmd_copy_image_to_buffer(cb, si, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, tmp, regions) };
        if recode {
            copy_to_tiler(sched);
            for (i, &(t, to_host)) in transforms.iter().filter(|(t, _)| *t != ColorTransform::None).enumerate() {
                if i > 0 {
                    compute_to_compute(sched);
                }
                let cb = sched.ordered_cmd();
                for &(at, region) in boxes {
                    let b = TilerBuffer { buffer: tmp, offset: at, size: region.src_slice as u64 * region.slices as u64 };
                    self.tiler.record_transform(cb, t, to_host, b, b, &region);
                }
            }
            tiler_to_copy(sched);
        } else {
            sched.memory_barrier((vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE), (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ));
        }
        // SAFETY: as above.
        unsafe { self.dev.device.cmd_copy_buffer_to_image(sched.ordered_cmd(), tmp, di, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &back) };
    }

    /// `dst` took the contents of `src` (of shape `key`): it takes over the
    /// GPU-modified state, the content version and the guest-side state.
    fn converted(&mut self, sched: &mut Scheduler, src: u64, dst: u64, key: ImageKey) {
        let (modified, seq, written) = (self.images[&src].gpu_modified, self.images[&src].seq, self.table.state(src).1);
        let tick = sched.current_tick();
        if let Some(d) = self.images.get_mut(&dst) {
            d.initialized = true;
            d.gpu_modified = modified;
            d.seq = seq;
            d.last_use = tick;
        }
        // Guest-side state carries over: CPU writes since the source was
        // filled still require an upload.
        let tracked = self.memory.is_watchable(key.main.addr, key.size());
        self.table.set_partial(dst, modified);
        self.table.rearm(dst, tracked && !written);
        if written {
            self.table.invalidate_range(key.main.addr, key.size());
        }
        self.stats.conversions += 1;
        crate::slice_shadow::hole(crate::slice_shadow::Hole::ImageRetile);
    }

    /// Drop an image (its contents are discarded; the host image goes to
    /// the pool).
    pub fn delete(&mut self, sched: &mut Scheduler, id: u64) {
        self.changed(id);
        let Some(img) = self.images.remove(&id) else { return };
        self.table.remove(id);
        if self.by_key.get(&img.key) == Some(&id) {
            self.by_key.remove(&img.key);
        }
        for p in img.key.planes() {
            self.planes.remove(&(p.addr, id));
            self.unregister_pages(id, &p);
        }
        for a in &img.aliases {
            if let Some(o) = self.images.get_mut(a) {
                o.aliases.retain(|&x| x != id);
            }
        }
        self.superseded.remove(&id);
        for v in img.views.values().chain(img.extra_views.iter()) {
            self.view_ranges.remove(&v.view);
            self.stencil_views.remove(&v.view);
        }
        let CachedImage { image, views, extra_views, pool_key, .. } = img;
        sched.retire(views);
        if !extra_views.is_empty() {
            sched.retire(extra_views);
        }
        self.pool.put(sched, pool_key, image);
        self.stats.deleted += 1;
    }

    /// `RPS5E_GPU_PROFILE`: the images most often refilled from newer
    /// aliases (`image <- newest source (+ other sources)`).
    pub fn retile_top(&self, n: usize) -> String {
        let mut v: Vec<_> = self.retile_counts.iter().map(|(k, c)| (*c, *k)).collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        let f = |k: &(u64, i32, u32, u32, u32)| format!("{:#x}:{:?}:{}x{}x{}", k.0, vk::Format::from_raw(k.1), k.2, k.3, k.4);
        v.iter().take(n).map(|(c, (y, x, others))| format!("{}<-{}(+{others})={c}", f(y), f(x))).collect::<Vec<_>>().join(" ")
    }

    /// `RPS5E_GPU_PROFILE`: the most uploaded images.
    pub fn upload_top(&self, n: usize) -> String {
        let mut v: Vec<_> = self.upload_counts.iter().map(|(k, c)| (*c, *k)).collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        v.iter()
            .take(n)
            .map(|(c, (a, f, w, h))| format!("{a:#x}:{:?}:{w}x{h}={c}", vk::Format::from_raw(*f)))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// True when the image must be (re)loaded from guest memory before use.
    pub fn needs_upload(&self, id: u64) -> bool {
        let Some(img) = self.images.get(&id) else { return false };
        if !img.initialized {
            return true;
        }
        let (watched, written) = self.table.resolved_state(id);
        written || (!watched && !img.gpu_modified)
    }

    /// [`Self::needs_upload`] cannot turn true without the lookup epoch
    /// moving: the image is initialised, and its range is watched (a CPU
    /// write moves the epoch) or it holds data newer than guest memory
    /// (GPU_PLAN §56: the store condition of the target memo — an
    /// unwatched image keeps its per-draw upload check).
    pub fn upload_covered(&self, id: u64) -> bool {
        let Some(img) = self.images.get(&id) else { return false };
        if !img.initialized {
            return false;
        }
        let modified = img.gpu_modified;
        let (watched, _) = self.table.resolved_state(id);
        watched || modified
    }

    /// The image [`Self::find_or_create`] finds for `key` without creating
    /// one (None: it would create it).
    pub fn find_existing(&self, key: &ImageKey) -> Option<u64> {
        if let Some(&id) = self.by_key.get(key) {
            return Some(id);
        }
        self.containing_layers(key)
    }

    /// [`Self::prepare`] would only mark the image used (no upload, refresh,
    /// revalidation or discard): read-only, for slice threads (R7b).
    pub fn prepare_noop(&self, id: u64, discard: bool) -> bool {
        if discard {
            return false;
        }
        let Some(i) = self.images.get(&id) else { return false };
        if !i.initialized {
            return false;
        }
        if i.aliases.is_empty() && (i.edge_hash.is_none() || i.gpu_modified) {
            let (watched, written) = self.table.resolved_state(id);
            return !(written || (!watched && !i.gpu_modified));
        }
        if i.edge_hash.is_some() && !i.gpu_modified && self.table.state(id) == (false, false) {
            return false;
        }
        !self.needs_upload(id) && self.newer_aliases(id).is_empty()
    }

    /// [`Self::mark_gpu_written`] would change lookups (the image does not
    /// hold the newest contents of its memory yet, or was merged).
    pub fn gpu_write_changes(&self, id: u64) -> bool {
        let Some(i) = self.images.get(&id) else { return true };
        let newest = i.gpu_modified && i.initialized && i.aliases.iter().all(|a| self.images.get(a).is_none_or(|x| x.seq < i.seq));
        !newest || i.merged
    }

    /// A fill of metadata at `addr` waits for its consumer ([`Self::take_fill`]).
    pub fn has_fill(&self, addr: u64) -> bool {
        self.metadata.contains_key(&addr)
    }

    /// The layout state of image `id` and what barriers on it name.
    pub fn layout_of(&self, id: u64) -> Option<(&layout::LayoutTracker, vk::Image, vk::ImageAspectFlags)> {
        self.images.get(&id).map(|i| (&i.layout, i.image.image, i.image.aspect))
    }

    /// Bring image `id` from its layout state into the requirements a
    /// slice segment recorded against (the barriers go to `out`), then take
    /// the state the segment left (see [`layout::LayoutTracker::adopt`]).
    pub fn apply_layouts(&mut self, id: u64, requirements: &[(Option<SubRange>, ImageUse, vk::PipelineStageFlags2)], last: &layout::LayoutTracker, out: &mut Vec<vk::ImageMemoryBarrier2<'static>>) {
        let Some(i) = self.images.get_mut(&id) else { return };
        let (image, aspect) = (i.image.image, i.image.aspect);
        for &(range, (layout, access), stage) in requirements {
            i.layout.barriers(image, aspect, layout, access, stage, range, out);
        }
        i.layout.adopt(last);
    }

    /// Overlapping images holding GPU data newer than the image's contents
    /// (and than guest memory), oldest first.
    fn newer_aliases(&self, id: u64) -> Vec<u64> {
        let Some(y) = self.images.get(&id) else { return Vec::new() };
        if y.aliases.is_empty() {
            return Vec::new();
        }
        let (b, e) = y.main_range();
        // An ordinary image over the depth/stencil target's stencil plane
        // also holds newer bytes the target's S8 aspect must adopt: the
        // alias walk below matched only the main plane, so an image over
        // just the stencil range was missed and the stencil plane loaded
        // stale guest bytes. Only when the target's own contents are not
        // newer than memory (the refresh of a split target falls back to
        // write-back plus upload, which would drop a gpu-modified main
        // plane).
        let stencil = (!y.gpu_modified && y.key.stencil.is_some()).then(|| {
            let p = y.key.stencil.as_ref().unwrap();
            (p.addr, p.addr + p.size())
        });
        let mut v: Vec<(u64, u64)> = y
            .aliases
            .iter()
            .filter_map(|a| self.images.get(a))
            .filter(|x| {
                let (xb, xe) = x.main_range();
                let main = xb < e && b < xe;
                let over_stencil = stencil.is_some_and(|(sb, se)| xb < se && sb < xe) && !x.key.is_depth();
                x.initialized && x.gpu_modified && x.seq > y.seq && (main || over_stencil)
            })
            // An image the CPU wrote only some pages of still holds the
            // newest bytes everywhere else (`prepare` merges those pages
            // into it before it is used as a source); one stale as a whole
            // is not newer than guest memory.
            .filter(|x| if alias_merge_enabled() { !self.table.stale_whole(x.id) } else { !self.table.state(x.id).1 })
            .map(|x| (x.seq, x.id))
            .collect();
        v.sort_unstable();
        v.into_iter().map(|(_, id)| id).collect()
    }

    /// True when the image must be refilled before use: from guest memory
    /// or from newer overlapping images.
    /// (`alias_merge_enabled` decides whether a newer alias with some CPU-
    /// written pages still counts, see `newer_aliases`.)
    pub fn needs_refresh(&self, id: u64) -> bool {
        self.needs_upload(id) || !self.newer_aliases(id).is_empty()
    }

    /// Make the image current (with guest memory, or with the newer GPU
    /// contents of overlapping images) and mark it used by the current
    /// tick. With `discard` the caller will overwrite the whole image, so
    /// nothing is loaded.
    pub fn prepare(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, id: u64, discard: bool) -> bool {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::IMG_PREPARE);
        let r = self.prepare_timed(sched, buffers, id, discard);
        sched.time_item_end(t);
        r
    }

    fn prepare_timed(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, id: u64, discard: bool) -> bool {
        crate::backend::profile::time(14, || self.prepare_inner(sched, buffers, id, discard))
    }

    fn prepare_inner(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, id: u64, discard: bool) -> bool {
        let tick = sched.current_tick();
        // The common use: an initialised image without aliases that
        // revalidation leaves alone (no edge hash, or GPU-written). Its
        // write count and upload state are then read under one lock (the
        // steps below in the same order, without the alias walk).
        let plain = !discard && self.images.get(&id).is_some_and(|i| i.initialized && i.aliases.is_empty() && (i.edge_hash.is_none() || i.gpu_modified));
        let (since, newer, mut stale) = if plain {
            let (since, (watched, written)) = self.table.counted_resolved_state(id);
            let modified = self.images.get(&id).is_some_and(|i| i.gpu_modified);
            (since, Vec::new(), written || (!watched && !modified))
        } else {
            // Writes counted after this may land after guest memory is read.
            let since = self.table.write_count(id);
            if !discard {
                self.revalidate(buffers, id);
            } else {
                // The caller overwrites the whole image: its memory is the GPU's.
                self.table.untouch(id);
            }
            let mut newer = self.newer_aliases(id);
            if newer.iter().any(|&x| self.table.state(x).1) {
                // A newer alias with CPU-written pages takes them first (a
                // merge raises its sequence; a full reload leaves it no longer
                // newer), so the refill reads the bytes memory holds now.
                for x in newer.clone() {
                    if self.table.state(x).1 {
                        self.prepare_inner(sched, buffers, x, false);
                    }
                }
                newer = self.newer_aliases(id);
            }
            (since, newer, self.needs_upload(id))
        };
        if stale && crate::backend::profile::on() {
            if let Some(img) = self.images.get(&id) {
                let (watched, written) = self.table.state(id);
                let reason = if !img.initialized { "new" } else if written { "written" } else if !watched { "unwatched" } else { "?" };
                *self.stale_reasons.entry((img.key.main.addr, reason)).or_default() += 1;
            }
        }
        if !stale && newer.is_empty() {
            if let Some(i) = self.images.get_mut(&id) {
                i.last_use = tick;
            }
            return true;
        }
        self.changed(id);
        crate::slice_shadow::hole(if discard {
            crate::slice_shadow::Hole::ImageDiscard
        } else if newer.is_empty() {
            crate::slice_shadow::Hole::ImageUpload
        } else {
            crate::slice_shadow::Hole::ImageRefresh
        });
        let mut merged_ok = true;
        let mut keep_touched = false;
        if stale && !discard {
            let modified = self.images.get(&id).is_some_and(|i| i.initialized && i.gpu_modified);
            if newer.is_empty() {
                if let Some(ok) = self.merge_cpu_pages(sched, buffers, id, false) {
                    if let Some(i) = self.images.get_mut(&id) {
                        i.last_use = tick;
                    }
                    return ok;
                }
            } else if modified {
                // Its own GPU contents with the CPU's pages merged are
                // current but for the newer aliases, which go on top below.
                if let Some(ok) = self.merge_cpu_pages(sched, buffers, id, true) {
                    merged_ok = ok;
                    stale = false;
                    keep_touched = self.table.touched(id);
                    self.stats.kept_merges += 1;
                }
            }
        }
        if !newer.is_empty() && !discard {
            // Diagnostics: an image refilled from a newer image of another
            // shape (reinterpreted memory), a few per run.
            static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if let (Some(y), Some(x)) = (self.images.get(&id), newer.last().and_then(|n| self.images.get(n))) {
                if !x.key.same_geometry(&y.key) && (y.key.layers >= 6 || x.key.layers >= 6) && LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 64 {
                    log::info!(
                        "ps5gpu: image #{id} {:#x} {:?} {}x{}x{} L{} (initialized {}, stale {stale}) refilled from newer #{} {:#x} {:?} {}x{}x{} (tick {})",
                        y.key.main.addr, y.key.format, y.key.width, y.key.height, y.key.layers.max(y.key.depth), y.key.levels, y.initialized,
                        x.id, x.key.main.addr, x.key.format, x.key.width, x.key.height, x.key.layers.max(x.key.depth), tick
                    );
                }
            }
        }
        if newer.is_empty() && stale && !discard {
            static LOGGED_UP: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if let Some(y) = self.images.get(&id) {
                if y.key.layers >= 6 && LOGGED_UP.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 64 {
                    let (watched, written) = self.table.state(id);
                    let mut head = [0u8; 64];
                    let mid = y.key.main.addr + (y.key.size() / 2) & !0xFF;
                    let mut middle = [0u8; 64];
                    let _ = self.memory.read_backing(y.key.main.addr, &mut head);
                    let _ = self.memory.read_backing(mid, &mut middle);
                    let words = |b: &[u8]| b.chunks(4).map(|c| format!("{:08x}", u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect::<Vec<_>>().join(" ");
                    let (hit, by_gpu) = self.table.hit(id).unwrap_or(((0, 0, 0), false));
                    log::info!(
                        "ps5gpu: image #{id} {:#x} {:?} {}x{}x{} L{} sw{} size {:#x} loaded from guest memory (initialized {}, gpu-modified {}, watched {watched}, cpu-written {written} at {:#x}+{:#x} by {}, gpu-dirty {}, tick {tick}) head [{}] middle [{}]",
                        y.key.main.addr, y.key.format, y.key.width, y.key.height, y.key.layers, y.key.levels, y.key.main.swizzle, y.key.size(), y.initialized, y.gpu_modified,
                        hit.0, hit.1, if by_gpu { "gpu buffer write".to_string() } else { format!("cpu rip {:#x}", hit.2) },
                        buffers.has_gpu_dirty(y.key.main.addr, y.key.size()), words(&head), words(&middle)
                    );
                    if hit.1 != 0 {
                        let mut at = [0u8; 64];
                        let _ = self.memory.read_backing(hit.0 & !0x3F, &mut at);
                        let others: Vec<String> = self
                            .overlapping(hit.0, hit.0 + 8, id)
                            .into_iter()
                            .filter_map(|o| self.images.get(&o))
                            .map(|o| format!("#{} {:#x} {:?} {}x{}x{} seq {} mod {}", o.id, o.key.main.addr, o.key.format, o.key.width, o.key.height, o.key.layers.max(o.key.depth), o.seq, o.gpu_modified))
                            .collect();
                        log::info!("ps5gpu:   hit bytes [{}], other images there: [{}]", words(&at), others.join(", "));
                    }
                }
            }
        }
        if crate::pipeline::trace::active() && !newer.is_empty() {
            let describe = |i: u64| self.images.get(&i).map(|c| format!("#{i} {:#x} {}x{} {:?} sw{}", c.key.main.addr, c.key.width, c.key.height, c.key.format, c.key.main.swizzle)).unwrap_or_default();
            let from: Vec<String> = newer.iter().map(|&x| describe(x)).collect();
            crate::pipeline::trace::log(&format!("    ({} {} from newer [{}]{})", if discard { "discard" } else { "refresh" }, describe(id), from.join(", "), if stale { ", stale" } else { "" }));
        }
        if discard {
            self.stats.discards += 1;
        } else if newer.is_empty() {
            let reason = match self.images.get(&id) {
                Some(i) if !i.initialized => 1,
                Some(i) => self.table.upload_reason(id, i.gpu_modified),
                None => 0,
            };
            self.stats.upload_reasons[reason] += 1;
        }
        let ok = if discard {
            true
        } else if !newer.is_empty() {
            match self.refresh_from_aliases(sched, buffers, id, &newer, stale) {
                Some(ok) => {
                    if keep_touched {
                        self.table.touch(id);
                    }
                    if let Some(i) = self.images.get_mut(&id) {
                        i.last_use = tick;
                    }
                    return ok && merged_ok;
                }
                None => {
                    // Shapes the GPU path does not handle (stencil planes,
                    // multisampling): through guest memory. The image's own
                    // GPU contents go there first (the newer images then
                    // land over them), or the reload would lose them where
                    // no newer image lies.
                    if !stale && self.images.get(&id).is_some_and(|i| i.initialized && i.gpu_modified) {
                        self.writeback(sched, buffers, id);
                        self.stats.own_writebacks += 1;
                    }
                    for &x in &newer {
                        if self.images.get(&x).is_some_and(|i| i.gpu_modified) {
                            self.writeback(sched, buffers, x);
                        }
                    }
                    self.upload(sched, buffers, id)
                }
            }
        } else {
            self.upload(sched, buffers, id)
        };
        let tracked = self.images.get(&id).is_some_and(|i| self.memory.is_watchable(i.key.main.addr, i.key.size()));
        self.table.set_partial(id, false);
        if discard {
            self.table.rearm(id, tracked);
        } else {
            self.table.rearm_since(id, tracked, since);
        }
        self.seq += 1;
        // A loaded image matches guest memory (a discarded one does not).
        let edge_hash = if discard { None } else { self.images.get(&id).and_then(|i| self.edge_hash(buffers, &i.key)) };
        if let Some(i) = self.images.get_mut(&id) {
            i.initialized = true;
            i.gpu_modified = false;
            i.last_use = tick;
            i.seq = self.seq;
            i.edge_hash = edge_hash;
        }
        ok && merged_ok
    }

    /// The hash of the guest bytes of an image lying within two pages (its
    /// edge pages are all of it), when guest memory holds them.
    fn edge_hash(&self, buffers: &BufferCache, key: &ImageKey) -> Option<u64> {
        let (b, e) = (key.main.addr, key.main.addr + key.size());
        let pages = e.div_ceil(PAGE_BYTES) - b / PAGE_BYTES;
        if key.stencil.is_some() || key.samples > 1 || pages > 2 || buffers.has_gpu_dirty(b, e - b) {
            return None;
        }
        let mut bytes = [0u8; 2 * PAGE_BYTES as usize];
        let bytes = &mut bytes[..(e - b) as usize];
        self.memory.read_backing(b, bytes).then(|| edge_hash_of(bytes))
    }

    /// An image that stopped being watched because a CPU write hit a page
    /// it shares (not its bytes) may still match guest memory: one within
    /// two pages compares the hash of its bytes with the one taken when it
    /// was loaded, and is watched again when they match instead of being
    /// reloaded.
    fn revalidate(&mut self, buffers: &BufferCache, id: u64) {
        let Some(img) = self.images.get(&id) else { return };
        let Some(hash) = img.edge_hash else { return };
        if !img.initialized || img.gpu_modified || self.table.state(id) != (false, false) {
            return;
        }
        let key = img.key;
        let since = self.table.write_count(id);
        if self.edge_hash(buffers, &key) == Some(hash) {
            let tracked = self.memory.is_watchable(key.main.addr, key.size());
            self.table.rearm_since(id, tracked, since);
            crate::slice_shadow::hole(crate::slice_shadow::Hole::ImageRevalidate);
            self.changed(id);
            self.stats.revalidations += 1;
        } else if let Some(i) = self.images.get_mut(&id) {
            i.edge_hash = None;
        }
    }

    /// Take the pages the CPU (or GPU work through a buffer) wrote into a
    /// GPU-modified image since its last load from guest memory, keeping
    /// the image's own contents everywhere else: the image is tiled into a
    /// scratch buffer in its guest layout, the written pages are copied
    /// over it from guest memory (or the buffer mirror) and the image is
    /// detiled from it. The result is what the hardware holds after the GPU
    /// results and then the CPU's bytes landed in memory. None when the
    /// image has no such pages or its shape needs a full load.
    ///
    /// `keep_version`: the image keeps its content version (sequence)
    /// instead of becoming the newest: newer aliases still go over it.
    fn merge_cpu_pages(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, id: u64, keep_version: bool) -> Option<bool> {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::IMG_UPLOAD);
        let r = self.merge_cpu_pages_timed(sched, buffers, id, keep_version);
        sched.time_item_end(t);
        r
    }

    fn merge_cpu_pages_timed(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, id: u64, keep_version: bool) -> Option<bool> {
        let img = self.images.get(&id)?;
        // A partially resident image takes the pages committed or written
        // since its load the same way: its other pages still match guest
        // memory.
        if !img.initialized || (!img.gpu_modified && !self.table.is_sparse(id)) {
            return None;
        }
        let key = img.key;
        if key.samples > 1 || key.stencil.is_some() {
            return None;
        }
        let runs = self.table.dirty_runs(id)?;
        let layout = key.layout(&key.main)?;
        let (yb, ye) = (key.main.addr, key.main.addr + layout.guest_bytes);
        if runs.len() == 1 && runs[0].0 <= yb && runs[0].1 >= ye {
            // All of it written: a plain load.
            return None;
        }
        let levels = img.image.mip_levels;
        let tiled = self.tiled.get(&self.dev, sched, ye - yb)?;
        // The pages taken now are read below; pages written from here on
        // stay dirty for the next merge.
        let (runs, since) = if read_race_check_enabled() { self.table.take_dirty_runs(id)? } else { (runs, 0) };
        // Only the layers (or block rows) holding the written pages go
        // through the scratch buffer: a page committed into a large
        // partially resident array does not tile the whole array out and in.
        let (wb, we) = runs.iter().fold((u64::MAX, 0u64), |(b, e), &(rb, re)| (b.min(rb.max(yb)), e.max(re.min(ye))));
        let mut ok = self.tile_out_window(sched, &key, id, levels, &layout, wb, we, tiled, 0);
        let mut bytes = 0;
        for &(b, e) in &runs {
            let (b, e) = (b.max(yb), e.min(ye));
            if b >= e {
                continue;
            }
            let Some((src, off)) = self.upload_source(sched, buffers, b, e - b) else {
                ok = false;
                continue;
            };
            before_buffer_work(sched);
            let region = vk::BufferCopy { src_offset: off, dst_offset: b - yb, size: e - b };
            // SAFETY: recording buffer; both buffers alive for the tick.
            unsafe { self.dev.device.cmd_copy_buffer(sched.ordered_cmd(), src, tiled, &[region]) };
            bytes += e - b;
        }
        ok &= self.tile_in_window(sched, &key, id, levels, &layout, wb, we, tiled, 0);
        static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 32 {
            log::info!(
                "ps5gpu: image #{id} {:#x} {:?} {}x{}x{} L{}: {} page run(s), {:#x} bytes written into its GPU contents merged (tick {})",
                key.main.addr, key.format, key.width, key.height, key.layers.max(key.depth), key.levels, runs.len(), bytes, sched.current_tick()
            );
        }
        let tracked = self.memory.is_watchable(yb, key.size());
        self.table.rearm_with(id, tracked, true, read_race_check_enabled().then_some(since));
        if !keep_version {
            self.seq += 1;
        }
        let seq = self.seq;
        if let Some(i) = self.images.get_mut(&id) {
            if !keep_version {
                i.seq = seq;
            }
            i.edge_hash = None;
            i.merged = true;
        }
        self.stats.uploads += 1;
        self.stats.uploaded_bytes += bytes;
        self.stats.merges += 1;
        Some(ok)
    }

    /// Refill `id` from the newer GPU contents of overlapping images
    /// (`newer`, oldest first) without a trip through guest memory: an image
    /// copy when the newest one is the same surface in another format,
    /// otherwise every newer image is tiled into a scratch buffer in its
    /// guest layout (over the image's own current bytes, or guest memory
    /// when those are stale) and the image is detiled from it. None when
    /// the shapes need the guest-memory path.
    fn refresh_from_aliases(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, id: u64, newer: &[u64], stale: bool) -> Option<bool> {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::IMG_RETILE);
        let r = self.refresh_from_aliases_timed(sched, buffers, id, newer, stale);
        sched.time_item_end(t);
        r
    }

    fn refresh_from_aliases_timed(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, id: u64, newer: &[u64], stale: bool) -> Option<bool> {
        let ykey = self.images.get(&id)?.key;
        let &last = newer.last()?;
        let xkey = self.images.get(&last)?.key;
        if xkey.same_geometry(&ykey) && xkey.stencil.is_none() && ykey.stencil.is_none() && xkey.samples == 1 && self.convert(sched, last, id) {
            return Some(true);
        }
        if blit::can_reinterpret(&xkey, &ykey) && self.reinterpret_as_multisampled_depth(sched, last, id) {
            return Some(true);
        }
        if direct_copy_enabled() && block_view_of(&xkey, &ykey) && self.copy_block_view(sched, last, id) {
            return Some(true);
        }
        if ykey.samples > 1 || ykey.stencil.is_some() {
            return None;
        }
        let ylayout = ykey.layout(&ykey.main)?;
        let mut sources = Vec::with_capacity(newer.len());
        for &x in newer {
            let k = self.images.get(&x)?.key;
            if k.samples > 1 {
                return None;
            }
            sources.push((x, k, k.layout(&k.main)?));
        }
        let (yb, ye) = (ykey.main.addr, ykey.main.addr + ylayout.guest_bytes);
        let base = sources.iter().map(|s| s.1.main.addr).fold(yb, u64::min) & !0xFF;
        let end = sources.iter().map(|s| s.1.main.addr + s.2.guest_bytes).fold(ye, u64::max);
        // A newer image whose elements fill its whole range (no padding)
        // and contains the image's range leaves no byte to keep.
        let dense = |k: &ImageKey, l: &SurfaceLayout| {
            let bd = block_dim(k.format) as u64;
            k.levels == 1 && (k.width as u64).div_ceil(bd) * (k.height as u64).div_ceil(bd) * k.main.bytes_per_element as u64 * k.slices() as u64 == l.guest_bytes
        };
        let covered = sources.iter().any(|s| dense(&s.1, &s.2) && inside(yb, ye, s.1.main.addr, s.1.main.addr + s.2.guest_bytes));
        let tiled = self.tiled.get(&self.dev, sched, end - base)?;
        let mut ok = true;
        // An image holding its own current GPU contents changes only where
        // the newer images lie: that part of it goes through the scratch
        // buffer, the rest stays (a large target refilled from small ones
        // reusing its memory is not tiled out and in whole).
        let (wb, we) = sources.iter().fold((u64::MAX, 0u64), |(b, e), s| (b.min(s.1.main.addr), e.max(s.1.main.addr + s.2.guest_bytes)));
        let (wb, we) = (wb.max(yb), we.min(ye));
        let partial = !covered && !stale && self.images[&id].gpu_modified && wb < we;
        if !covered {
            // The bytes no newer image covers: the image's own GPU data, or
            // guest memory (equal to its contents, or newer when stale).
            let y = &self.images[&id];
            if partial {
                let levels = y.image.mip_levels;
                ok &= self.tile_out_window(sched, &ykey, id, levels, &ylayout, wb, we, tiled, yb - base);
            } else if !stale && y.gpu_modified {
                let levels = y.image.mip_levels;
                ok &= self.tile_out(sched, &ykey, id, levels, &ykey.main, &ylayout, tiled, yb - base);
            } else {
                let (src, off) = self.upload_source(sched, buffers, yb, ye - yb)?;
                before_buffer_work(sched);
                let region = vk::BufferCopy { src_offset: off, dst_offset: yb - base, size: ye - yb };
                // SAFETY: recording buffer; both buffers alive for the tick.
                unsafe { self.dev.device.cmd_copy_buffer(sched.ordered_cmd(), src, tiled, &[region]) };
            }
        }
        let mut seq = 0;
        for (x, k, l) in &sources {
            let xi = &self.images[x];
            let levels = xi.image.mip_levels;
            seq = seq.max(xi.seq);
            // Only the source's bytes over this image are needed.
            ok &= self.tile_out_window(sched, k, *x, levels, l, yb, ye, tiled, k.main.addr - base);
        }
        let levels = self.images[&id].image.mip_levels;
        if partial {
            ok &= self.tile_in_window(sched, &ykey, id, levels, &ylayout, wb, we, tiled, yb - base);
        } else {
            ok &= self.tile_in(sched, &ykey, id, levels, &ykey.main, &ylayout, tiled, yb - base);
        }
        let tracked = self.memory.is_watchable(yb, ykey.size());
        self.table.set_partial(id, true);
        self.table.rearm(id, tracked);
        if let Some(i) = self.images.get_mut(&id) {
            i.initialized = true;
            i.gpu_modified = true;
            i.seq = seq;
        }
        self.stats.retiles += 1;
        crate::slice_shadow::hole(crate::slice_shadow::Hole::ImageRetile);
        if crate::backend::profile::on() {
            let k = |k: &ImageKey| (k.main.addr, k.format.as_raw(), k.width, k.height, k.layers.max(k.depth));
            *self.retile_counts.entry((k(&ykey), k(&xkey), newer.len() - 1)).or_default() += 1;
        }
        Some(ok)
    }

    /// Refill block-compressed image `id` from the newer image `src` that
    /// stores its blocks as uncompressed texels in the same guest layout
    /// ([`block_view_of`]): texel (x, y) of `src` is block (x, y) of `id`,
    /// so one image copy gives what tiling `src` out and `id` in through
    /// the guest layout gives.
    fn copy_block_view(&mut self, sched: &mut Scheduler, src: u64, id: u64) -> bool {
        let (Some(s), Some(d)) = (self.images.get(&src), self.images.get(&id)) else { return false };
        if s.image.mip_levels != 1 || d.image.mip_levels != 1 || s.image.aspect != vk::ImageAspectFlags::COLOR || d.image.aspect != vk::ImageAspectFlags::COLOR {
            return false;
        }
        let (xkey, ykey) = (s.key, d.key);
        let (si, di, seq) = (s.image.image, d.image.image, s.seq);
        let sub = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: xkey.layers };
        // Between an uncompressed and a compressed format the extent counts
        // texels of the source.
        let copy = vk::ImageCopy::default().src_subresource(sub).dst_subresource(sub).extent(vk::Extent3D { width: xkey.width, height: xkey.height, depth: 1 });
        self.transit_pair(sched, (src, None), (id, None));
        // SAFETY: recording buffer; source in TRANSFER_SRC, destination in TRANSFER_DST layout; the texel and block sizes are equal.
        unsafe { self.dev.device.cmd_copy_image(sched.ordered_cmd(), si, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, di, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy]) };
        let tracked = self.memory.is_watchable(ykey.main.addr, ykey.size());
        self.table.set_partial(id, true);
        self.table.rearm(id, tracked);
        if let Some(i) = self.images.get_mut(&id) {
            i.initialized = true;
            i.gpu_modified = true;
            i.seq = seq;
        }
        self.stats.retiles += 1;
        self.stats.direct_copies += 1;
        if crate::backend::profile::on() {
            let k = |k: &ImageKey| (k.main.addr, k.format.as_raw(), k.width, k.height, k.layers.max(k.depth));
            *self.retile_counts.entry((k(&ykey), k(&xkey), 0)).or_default() += 1;
        }
        true
    }

    /// Rebuild the multisampled depth image `dst` from the single-sample
    /// colour image `src` over the same memory, whose components hold the
    /// samples' depths ([`blit`]). `dst` takes the source's content version.
    fn reinterpret_as_multisampled_depth(&mut self, sched: &mut Scheduler, src: u64, dst: u64) -> bool {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::IMG_CONVERT);
        let r = self.reinterpret_as_multisampled_depth_timed(sched, src, dst);
        sched.time_item_end(t);
        r
    }

    fn reinterpret_as_multisampled_depth_timed(&mut self, sched: &mut Scheduler, src: u64, dst: u64) -> bool {
        let (Some(s), Some(d)) = (self.images.get(&src), self.images.get(&dst)) else { return false };
        let (skey, dkey) = (s.key, d.key);
        if !s.image.usage.contains(vk::ImageUsageFlags::SAMPLED) || !d.image.usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
            return false;
        }
        if self.blit.is_none() {
            self.blit = Some(blit::BlitHelper::new(&self.dev));
        }
        if !matches!(self.blit, Some(Some(_))) {
            return false;
        }
        let Some(source) = self.view(src, skey.format, vk::ImageAspectFlags::COLOR, 0, 1) else { return false };
        let Some(destination) = self.view(dst, dkey.format, vk::ImageAspectFlags::DEPTH, 0, 1) else { return false };
        let mut v = std::mem::take(&mut self.barriers);
        v.clear();
        self.transition(src, SAMPLED, None, &mut v);
        self.transition(dst, (blit::DESTINATION_LAYOUT, DEPTH_ATTACHMENT.1), None, &mut v);
        sched.image_barriers(&v);
        self.barriers = v;
        let cb = sched.ordered_cmd();
        let Some(Some(helper)) = self.blit.as_mut() else { return false };
        if !helper.record(cb, source, destination, dkey.width, dkey.height, dkey.samples, dkey.format) {
            return false;
        }
        let (seq, tick) = (self.images[&src].seq, sched.current_tick());
        let tracked = self.memory.is_watchable(dkey.main.addr, dkey.size());
        self.table.set_partial(dst, true);
        self.table.rearm(dst, tracked);
        if let Some(i) = self.images.get_mut(&dst) {
            i.initialized = true;
            i.gpu_modified = true;
            i.seq = seq;
            i.last_use = tick;
        }
        self.stats.reinterpretations += 1;
        crate::slice_shadow::hole(crate::slice_shadow::Hole::ImageRetile);
        true
    }

    /// Where an upload reads `[addr, addr+size)` from: the buffer cache's

    /// device mirror when it holds GPU-written pages there (no download
    /// first), else a copy of guest memory in the staging ring (a temporary
    /// buffer when the ring is full). A mirror that merely covers the range
    /// is not used: in SILENT HILL's brightness menu it held stale texture
    /// bytes, so some guest writes still reach texture memory without
    /// passing the page tracker. `(buffer, offset)`.
    fn upload_source(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, addr: u64, size: u64) -> Option<(vk::Buffer, u64)> {
        if buffers.has_gpu_dirty(addr, size) {
            if let Ok(s) = buffers.obtain(sched, addr, size, false) {
                return Some((s.buffer, s.offset));
            }
            buffers.download(sched, addr, size);
        }
        if let Some(off) = self.staging.alloc(sched, size, 256) {
            if !self.memory.read_committed(addr, self.staging.bytes(off, size)) {
                crate::log_once(&format!("ps5gpu: image source {addr:#x}+{size:#x} unreadable"));
                return None;
            }
            self.staging.buffer.flush(off, size);
            return Some((self.staging.buffer.buffer, off));
        }
        let tmp = Buffer::new(&self.dev, size, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER, MemoryUse::Upload).ok()?;
        let readable = self.memory.read_committed(addr, &mut tmp.bytes_mut()[..size as usize]);
        tmp.flush(0, size);
        let handle = tmp.buffer;
        sched.retire_sized(tmp, size);
        if !readable {
            crate::log_once(&format!("ps5gpu: image source {addr:#x}+{size:#x} unreadable"));
            return None;
        }
        Some((handle, 0))
    }

    /// Record the transfer of `plane` from its guest layout at
    /// `src`+`offset` into image `id` (GPU detile, or buffer copies when
    /// linear), with the colour transform of the key into the host format.
    #[allow(clippy::too_many_arguments)]
    fn tile_in(&mut self, sched: &mut Scheduler, key: &ImageKey, id: u64, levels: u32, plane: &Plane, layout: &SurfaceLayout, src: vk::Buffer, offset: u64) -> bool {
        let plan = TransferPlan::new(key, plane, layout, levels);
        self.tile_in_plan(sched, key, id, plane, layout, plan, src, offset)
    }

    /// [`Self::tile_in`] of the part of the main plane holding the bytes
    /// `[begin, end)` (see [`Self::tile_out_window`]): the rest of the
    /// image keeps its contents.
    #[allow(clippy::too_many_arguments)]
    fn tile_in_window(&mut self, sched: &mut Scheduler, key: &ImageKey, id: u64, levels: u32, layout: &SurfaceLayout, begin: u64, end: u64, src: vk::Buffer, offset: u64) -> bool {
        let plane = key.main;
        let plan = TransferPlan::window(key, &plane, layout, levels, begin.saturating_sub(plane.addr), end.saturating_sub(plane.addr))
            .unwrap_or_else(|| TransferPlan::new(key, &plane, layout, levels));
        self.tile_in_plan(sched, key, id, &plane, layout, plan, src, offset)
    }

    #[allow(clippy::too_many_arguments)]
    fn tile_in_plan(&mut self, sched: &mut Scheduler, key: &ImageKey, id: u64, plane: &Plane, layout: &SurfaceLayout, plan: TransferPlan, src: vk::Buffer, offset: u64) -> bool {
        let Some(image) = self.images.get(&id).map(|i| i.image.image) else { return false };
        let transform = plane_transform(key, plane);
        let bpe = plane.bytes_per_element;
        before_buffer_work(sched);
        let widen = widened_depth(key, plane);
        if plan.linear {
            if offset % (bpe.max(4) as u64) != 0 {
                return false;
            }
            if widen {
                // 16-bit guest depth into the 32-bit float depth aspect: the
                // elements widened in place order, offsets doubled.
                let count = (layout.guest_bytes / 2) as u32;
                let Some(wide) = self.wide.get(&self.dev, sched, layout.guest_bytes * 2) else { return false };
                let from = TilerBuffer { buffer: src, offset, size: layout.guest_bytes };
                let to = TilerBuffer { buffer: wide, offset: 0, size: layout.guest_bytes * 2 };
                if !self.tiler.record_depth16(sched.ordered_cmd(), true, from, to, count) {
                    return false;
                }
                tiler_to_copy(sched);
                let copies: Vec<vk::BufferImageCopy> = plan.copies.iter().map(|c| vk::BufferImageCopy { buffer_offset: c.buffer_offset * 2, ..*c }).collect();
                self.transit(sched, id, TRANSFER_DST, None);
                // SAFETY: recording buffer; image in TRANSFER_DST layout.
                unsafe { self.dev.device.cmd_copy_buffer_to_image(sched.ordered_cmd(), wide, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &copies) };
                return true;
            }
            if transform == ColorTransform::None {
                let copies: Vec<vk::BufferImageCopy> = plan.copies.iter().map(|c| vk::BufferImageCopy { buffer_offset: c.buffer_offset + offset, ..*c }).collect();
                self.transit(sched, id, TRANSFER_DST, None);
                // SAFETY: recording buffer; image in TRANSFER_DST layout.
                unsafe { self.dev.device.cmd_copy_buffer_to_image(sched.ordered_cmd(), src, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &copies) };
                return true;
            }
            // Transformed elements at the same positions of a scratch copy.
            let Some(lin) = self.linear.get(&self.dev, sched, layout.guest_bytes) else { return false };
            let cb = sched.ordered_cmd();
            for c in &plan.copies {
                let (region, span) = linear_box(c, bpe);
                let from = TilerBuffer { buffer: src, offset: offset + c.buffer_offset, size: span };
                let to = TilerBuffer { buffer: lin, offset: c.buffer_offset, size: span };
                self.tiler.record_transform(cb, transform, true, from, to, &region);
            }
            tiler_to_copy(sched);
            self.transit(sched, id, TRANSFER_DST, None);
            // SAFETY: recording buffer; image in TRANSFER_DST layout.
            unsafe { self.dev.device.cmd_copy_buffer_to_image(sched.ordered_cmd(), lin, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &plan.copies) };
            return true;
        }
        let Some(lin) = self.linear.get(&self.dev, sched, plan.linear_bytes) else { return false };
        let cb = sched.ordered_cmd();
        let tiled = TilerBuffer { buffer: src, offset, size: layout.guest_bytes };
        let mut recorded = true;
        for r in &plan.tiles {
            let dst = TilerBuffer { buffer: lin, offset: r.0, size: plan.linear_bytes - r.0 };
            recorded &= self.tiler.record_region(cb, plane.swizzle, bpe, layout.thick, false, tiled, dst, &r.1);
        }
        if !recorded {
            crate::log_once(&format!("ps5gpu: no GPU tiler for swizzle {} bpe {} thick {}", plane.swizzle, bpe, layout.thick));
            return false;
        }
        if transform != ColorTransform::None {
            compute_to_compute(sched);
            let cb = sched.ordered_cmd();
            for r in &plan.tiles {
                let region = TransformRegion::packed(r.1.width, r.1.height, r.1.slices, bpe);
                let b = TilerBuffer { buffer: lin, offset: r.0, size: region.src_slice as u64 * region.slices as u64 };
                self.tiler.record_transform(cb, transform, true, b, b, &region);
            }
        }
        if widen {
            compute_to_compute(sched);
            let count = (plan.linear_bytes / 2) as u32;
            let Some(wide) = self.wide.get(&self.dev, sched, plan.linear_bytes * 2) else { return false };
            let from = TilerBuffer { buffer: lin, offset: 0, size: plan.linear_bytes };
            let to = TilerBuffer { buffer: wide, offset: 0, size: plan.linear_bytes * 2 };
            if !self.tiler.record_depth16(sched.ordered_cmd(), true, from, to, count) {
                return false;
            }
            tiler_to_copy(sched);
            let copies: Vec<vk::BufferImageCopy> = plan.copies.iter().map(|c| vk::BufferImageCopy { buffer_offset: c.buffer_offset * 2, ..*c }).collect();
            self.transit(sched, id, TRANSFER_DST, None);
            // SAFETY: recording buffer; image in TRANSFER_DST layout.
            unsafe { self.dev.device.cmd_copy_buffer_to_image(sched.ordered_cmd(), wide, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &copies) };
            return true;
        }
        tiler_to_copy(sched);
        self.transit(sched, id, TRANSFER_DST, None);
        // SAFETY: recording buffer; image in TRANSFER_DST layout.
        unsafe { self.dev.device.cmd_copy_buffer_to_image(sched.ordered_cmd(), lin, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &plan.copies) };
        true
    }

    /// Record the transfer of `plane` of image `id` into its guest layout
    /// at `dst`+`offset` (GPU tiling, or buffer copies when linear), with
    /// the colour transform of the key back to the stored bits. Bytes the
    /// layout does not cover keep their contents.
    #[allow(clippy::too_many_arguments)]
    fn tile_out(&mut self, sched: &mut Scheduler, key: &ImageKey, id: u64, levels: u32, plane: &Plane, layout: &SurfaceLayout, dst: vk::Buffer, offset: u64) -> bool {
        let plan = TransferPlan::new(key, plane, layout, levels);
        self.tile_out_plan(sched, key, id, plane, layout, plan, dst, offset)
    }

    /// [`Self::tile_out`] of the part of the main plane holding the bytes
    /// `[begin, end)` (guest addresses): whole layers, or the block rows of
    /// one layer, of a thin single-level surface (all of it otherwise).
    /// A small image refilled from a large overlapping one needs only the
    /// large one's bytes over it.
    #[allow(clippy::too_many_arguments)]
    fn tile_out_window(&mut self, sched: &mut Scheduler, key: &ImageKey, id: u64, levels: u32, layout: &SurfaceLayout, begin: u64, end: u64, dst: vk::Buffer, offset: u64) -> bool {
        let plane = key.main;
        let plan = TransferPlan::window(key, &plane, layout, levels, begin.saturating_sub(plane.addr), end.saturating_sub(plane.addr))
            .unwrap_or_else(|| TransferPlan::new(key, &plane, layout, levels));
        self.tile_out_plan(sched, key, id, &plane, layout, plan, dst, offset)
    }

    #[allow(clippy::too_many_arguments)]
    fn tile_out_plan(&mut self, sched: &mut Scheduler, key: &ImageKey, id: u64, plane: &Plane, layout: &SurfaceLayout, plan: TransferPlan, dst: vk::Buffer, offset: u64) -> bool {
        let Some(image) = self.images.get(&id).map(|i| i.image.image) else { return false };
        let transform = plane_transform(key, plane);
        let bpe = plane.bytes_per_element;
        before_buffer_work(sched);
        let widen = widened_depth(key, plane);
        if plan.linear && widen {
            if offset % 4 != 0 {
                return false;
            }
            let Some(wide) = self.wide.get(&self.dev, sched, layout.guest_bytes * 2) else { return false };
            let copies: Vec<vk::BufferImageCopy> = plan.copies.iter().map(|c| vk::BufferImageCopy { buffer_offset: c.buffer_offset * 2, ..*c }).collect();
            self.transit(sched, id, TRANSFER_SRC, None);
            // SAFETY: recording buffer; image in TRANSFER_SRC layout.
            unsafe { self.dev.device.cmd_copy_image_to_buffer(sched.ordered_cmd(), image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, wide, &copies) };
            copy_to_tiler(sched);
            let from = TilerBuffer { buffer: wide, offset: 0, size: layout.guest_bytes * 2 };
            let to = TilerBuffer { buffer: dst, offset, size: layout.guest_bytes };
            if !self.tiler.record_depth16(sched.ordered_cmd(), false, from, to, (layout.guest_bytes / 2) as u32) {
                return false;
            }
            after_buffer_writes(sched);
            return true;
        }
        if plan.linear {
            if offset % (bpe.max(4) as u64) != 0 {
                return false;
            }
            if transform == ColorTransform::None {
                let copies: Vec<vk::BufferImageCopy> = plan.copies.iter().map(|c| vk::BufferImageCopy { buffer_offset: c.buffer_offset + offset, ..*c }).collect();
                self.transit(sched, id, TRANSFER_SRC, None);
                // SAFETY: recording buffer; image in TRANSFER_SRC layout.
                unsafe { self.dev.device.cmd_copy_image_to_buffer(sched.ordered_cmd(), image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, dst, &copies) };
            } else {
                // Through a scratch copy at the same positions: only the
                // elements are transformed into the destination.
                let Some(lin) = self.linear.get(&self.dev, sched, layout.guest_bytes) else { return false };
                self.transit(sched, id, TRANSFER_SRC, None);
                // SAFETY: recording buffer; image in TRANSFER_SRC layout.
                unsafe { self.dev.device.cmd_copy_image_to_buffer(sched.ordered_cmd(), image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, lin, &plan.copies) };
                copy_to_tiler(sched);
                let cb = sched.ordered_cmd();
                for c in &plan.copies {
                    let (region, span) = linear_box(c, bpe);
                    let from = TilerBuffer { buffer: lin, offset: c.buffer_offset, size: span };
                    let to = TilerBuffer { buffer: dst, offset: offset + c.buffer_offset, size: span };
                    self.tiler.record_transform(cb, transform, false, from, to, &region);
                }
            }
            after_buffer_writes(sched);
            return true;
        }
        let Some(lin) = self.linear.get(&self.dev, sched, plan.linear_bytes) else { return false };
        self.transit(sched, id, TRANSFER_SRC, None);
        if widen {
            let Some(wide) = self.wide.get(&self.dev, sched, plan.linear_bytes * 2) else { return false };
            let copies: Vec<vk::BufferImageCopy> = plan.copies.iter().map(|c| vk::BufferImageCopy { buffer_offset: c.buffer_offset * 2, ..*c }).collect();
            // SAFETY: recording buffer; image in TRANSFER_SRC layout.
            unsafe { self.dev.device.cmd_copy_image_to_buffer(sched.ordered_cmd(), image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, wide, &copies) };
            copy_to_tiler(sched);
            let from = TilerBuffer { buffer: wide, offset: 0, size: plan.linear_bytes * 2 };
            let to = TilerBuffer { buffer: lin, offset: 0, size: plan.linear_bytes };
            if !self.tiler.record_depth16(sched.ordered_cmd(), false, from, to, (plan.linear_bytes / 2) as u32) {
                return false;
            }
            compute_to_compute(sched);
        } else {
            // SAFETY: recording buffer; image in TRANSFER_SRC layout.
            unsafe { self.dev.device.cmd_copy_image_to_buffer(sched.ordered_cmd(), image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, lin, &plan.copies) };
            copy_to_tiler(sched);
        }
        if transform != ColorTransform::None {
            let cb = sched.ordered_cmd();
            for r in &plan.tiles {
                let region = TransformRegion::packed(r.1.width, r.1.height, r.1.slices, bpe);
                let b = TilerBuffer { buffer: lin, offset: r.0, size: region.src_slice as u64 * region.slices as u64 };
                self.tiler.record_transform(cb, transform, false, b, b, &region);
            }
            compute_to_compute(sched);
        }
        let cb = sched.ordered_cmd();
        let tiled = TilerBuffer { buffer: dst, offset, size: layout.guest_bytes };
        let mut recorded = true;
        for r in &plan.tiles {
            let src = TilerBuffer { buffer: lin, offset: r.0, size: plan.linear_bytes - r.0 };
            recorded &= self.tiler.record_region(cb, plane.swizzle, bpe, layout.thick, true, src, tiled, &r.1);
        }
        if !recorded {
            return false;
        }
        after_buffer_writes(sched);
        true
    }

    /// Fill the image from guest memory (GPU detile of every plane, level
    /// and layer).
    fn upload(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, id: u64) -> bool {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::IMG_UPLOAD);
        let r = self.upload_timed(sched, buffers, id);
        sched.time_item_end(t);
        r
    }

    fn upload_timed(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, id: u64) -> bool {
        let Some(img) = self.images.get(&id) else { return false };
        let key = img.key;
        if key.samples > 1 {
            // Multisampled guest data is not loaded (only rendered): start
            // from zero rather than undefined contents.
            crate::log_once(&format!("ps5gpu: multisampled image upload not supported ({:?})", key.format));
            let (image, range, depth) = (img.image.image, img.full_range(), key.is_depth());
            self.transit(sched, id, TRANSFER_DST, None);
            let cb = sched.ordered_cmd();
            // SAFETY: recording buffer; image in TRANSFER_DST layout.
            unsafe {
                if depth {
                    let v = vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 };
                    self.dev.device.cmd_clear_depth_stencil_image(cb, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &v, &[range]);
                } else {
                    let v = vk::ClearColorValue { uint32: [0; 4] };
                    self.dev.device.cmd_clear_color_image(cb, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &v, &[range]);
                }
            }
            return false;
        }
        let levels = img.image.mip_levels;
        let mut ok = true;
        for plane in key.planes() {
            let Some(layout) = key.layout(&plane) else {
                crate::log_once(&format!("ps5gpu: no layout for swizzle {} bpe {} ({:?} {:?})", plane.swizzle, plane.bytes_per_element, key.dim, key.format));
                ok = false;
                continue;
            };
            let size = layout.guest_bytes;
            let Some((src, offset)) = self.upload_source(sched, buffers, plane.addr, size) else {
                ok = false;
                continue;
            };
            ok &= self.tile_in(sched, &key, id, levels, &plane, &layout, src, offset);
            self.stats.uploads += 1;
            if crate::backend::profile::on() {
                *self.upload_counts.entry((plane.addr, key.format.as_raw(), key.width, key.height)).or_default() += 1;
            }
            self.stats.uploaded_bytes += size;
        }
        ok
    }

    /// Write a GPU-modified image back to guest memory: the GPU tiles every
    /// level and layer into the buffer cache's mirror of its pages, which
    /// become GPU-dirty. Nothing waits; the CPU receives the bytes on its
    /// first access (fault) and later GPU users read the mirror. The image
    /// keeps its state: it still holds the newest bytes of its memory, so
    /// aliases keep taking them from it on the GPU.
    pub fn writeback(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, id: u64) -> bool {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::IMG_WRITEBACK);
        let r = self.writeback_timed(sched, buffers, id);
        sched.time_item_end(t);
        r
    }

    fn writeback_timed(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, id: u64) -> bool {
        self.epoch += 1;
        // Memory the CPU wrote into since the image was loaded belongs to
        // the CPU: GPU bytes never go over it.
        if !self.table.may_write_back(id) {
            static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if let Some(i) = self.images.get(&id) {
                if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
                    log::info!("ps5gpu: image #{id} {:#x} {:?} {}x{} not written back: the CPU wrote into its memory", i.key.main.addr, i.key.format, i.key.width, i.key.height);
                }
            }
            return false;
        }
        let Some(img) = self.images.get(&id) else { return false };
        let key = img.key;
        if key.samples > 1 {
            return false;
        }
        // An image whose whole range a newer image holds (loaded from guest
        // memory or written later: the memory was reused) has nothing left
        // to write: its bytes would overwrite the newer contents. (SILENT
        // HILL's reflection capture array shares memory with an older
        // R32G32B32A32 scene-data texture; writing that one back on
        // eviction put float rows into the cube array.)
        let (b, e) = img.main_range();
        let superseded = img.aliases.iter().filter_map(|a| self.images.get(a)).find(|o| {
            let (ob, oe) = o.main_range();
            o.initialized && o.seq > img.seq && inside(b, e, ob, oe)
        });
        if let Some(o) = superseded {
            static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
                log::info!(
                    "ps5gpu: image #{id} {:#x} {:?} {}x{} not written back: newer #{} {:#x} {:?} {}x{}x{} holds its memory",
                    key.main.addr, key.format, key.width, key.height, o.id, o.key.main.addr, o.key.format, o.key.width, o.key.height, o.key.layers.max(o.key.depth)
                );
            }
            if let Some(i) = self.images.get_mut(&id) {
                i.gpu_modified = false;
            }
            self.stats.superseded_writebacks += 1;
            return true;
        }
        let levels = img.image.mip_levels;
        let mut ok = true;
        for plane in key.planes() {
            let Some(layout) = key.layout(&plane) else { continue };
            let size = layout.guest_bytes;
            // The mirror holds current guest bytes first: the padding the
            // tiling does not cover is kept.
            let Ok(out) = buffers.obtain(sched, plane.addr, size, true) else {
                ok = false;
                continue;
            };
            ok &= self.tile_out(sched, &key, id, levels, &plane, &layout, out.buffer, out.offset);
        }
        self.stats.writebacks += 1;
        if ok {
            if let Some(i) = self.images.get_mut(&id) {
                i.written_back_seq = i.seq;
            }
        }
        ok
    }

    /// Copy the top-left `width` x `height` elements of level 0, layer 0 of
    /// colour image `id` into `target` (GENERAL layout) through a buffer,
    /// with the colour transform `transform` towards the host format on
    /// the way (BGRA16 display buffers rendered in storage order).
    pub fn copy_transformed(&mut self, sched: &mut Scheduler, id: u64, target: vk::Image, width: u32, height: u32, transform: ColorTransform) -> bool {
        let bpe = transform.element_bytes();
        let Some(img) = self.images.get(&id) else { return false };
        if bpe == 0 || img.key.main.bytes_per_element != bpe || img.key.is_depth() || img.key.width < width || img.key.height < height {
            return false;
        }
        let image = img.image.image;
        let bytes = width as u64 * height as u64 * bpe as u64;
        let Some(lin) = self.linear.get(&self.dev, sched, bytes) else { return false };
        let copy = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
            .image_extent(vk::Extent3D { width, height, depth: 1 });
        before_buffer_work(sched);
        self.transit(sched, id, TRANSFER_SRC, None);
        // SAFETY: recording buffer; image in TRANSFER_SRC layout.
        unsafe { self.dev.device.cmd_copy_image_to_buffer(sched.ordered_cmd(), image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, lin, &[copy]) };
        copy_to_tiler(sched);
        let region = TransformRegion::packed(width, height, 1, bpe);
        let b = TilerBuffer { buffer: lin, offset: 0, size: bytes };
        let cb = sched.ordered_cmd();
        if !self.tiler.record_transform(cb, transform, true, b, b, &region) {
            return false;
        }
        tiler_to_copy(sched);
        // SAFETY: recording buffer; the target is in GENERAL layout.
        unsafe { self.dev.device.cmd_copy_buffer_to_image(sched.ordered_cmd(), lin, target, vk::ImageLayout::GENERAL, &[copy]) };
        true
    }

    /// An image view of level 0 (cached per format/aspect/layer range).

    pub fn view(&mut self, id: u64, format: vk::Format, aspect: vk::ImageAspectFlags, base_layer: u32, layers: u32) -> Option<vk::ImageView> {
        self.view_level(id, format, aspect, 0, base_layer, layers)
    }

    /// A 2D (array) view of one level, for rendering: the layers of a
    /// volume are its depth slices at that level.
    pub fn view_level(&mut self, id: u64, format: vk::Format, aspect: vk::ImageAspectFlags, level: u32, base_layer: u32, layers: u32) -> Option<vk::ImageView> {
        let ty = if layers > 1 { vk::ImageViewType::TYPE_2D_ARRAY } else { vk::ImageViewType::TYPE_2D };
        let range = ViewRange { ty, base_level: level, levels: 1, base_layer, layers, min_lod: 0.0 };
        self.view_range(id, format, aspect, range, vk::ComponentMapping::default())
    }

    /// An image view of an explicit view type and component swizzle of
    /// level 0 (cached).
    #[allow(clippy::too_many_arguments)]
    pub fn view_with(
        &mut self,
        id: u64,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
        base_layer: u32,
        layers: u32,
        ty: vk::ImageViewType,
        components: vk::ComponentMapping,
    ) -> Option<vk::ImageView> {
        self.view_range(id, format, aspect, ViewRange { ty, base_level: 0, levels: 1, base_layer, layers, min_lod: 0.0 }, components)
    }

    /// An image view of `range` (cached). Ranges past the image are
    /// clamped; a view the image cannot provide is None.
    pub fn view_range(&mut self, id: u64, format: vk::Format, aspect: vk::ImageAspectFlags, range: ViewRange, components: vk::ComponentMapping) -> Option<vk::ImageView> {
        let plan = self.view_plan(id, format, aspect, range, components)?;
        let img = self.images.get_mut(&id)?;
        if let Some((k, v)) = img.last_view {
            if k == plan.key {
                return Some(v);
            }
        }
        if let Some(v) = img.views.get(&plan.key) {
            img.last_view = Some((plan.key, v.view));
            return Some(v.view);
        }
        let v = self.create_view(&plan)?;
        let handle = v.view;
        self.adopt_view(id, plan, v);
        Some(handle)
    }

    /// The clamped view of `range` an image can provide (None: it cannot,
    /// see [`Self::view_range`]).
    pub fn view_plan(&self, id: u64, format: vk::Format, aspect: vk::ImageAspectFlags, range: ViewRange, components: vk::ComponentMapping) -> Option<ViewPlan> {
        let min_lod_ok = self.dev.caps.image_view_min_lod;
        let img = self.images.get(&id)?;
        let image_levels = img.image.mip_levels;
        let base_level = range.base_level.min(image_levels - 1);
        let levels = range.levels.clamp(1, image_levels - base_level);
        let volume = img.key.dim == ImageDim::D3;
        let slices = if volume && range.ty != vk::ImageViewType::TYPE_3D { img.key.level_extent(base_level).depth } else { img.image.layers };
        let base_layer = range.base_layer.min(slices - 1);
        let mut layers = range.layers.clamp(1, slices - base_layer);
        let ty = range.ty;
        match ty {
            vk::ImageViewType::TYPE_3D | vk::ImageViewType::TYPE_1D | vk::ImageViewType::TYPE_2D => layers = 1,
            vk::ImageViewType::CUBE => layers = 6,
            vk::ImageViewType::CUBE_ARRAY => layers -= layers % 6,
            _ => {}
        }
        if matches!(ty, vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY)
            && (base_layer + layers > img.image.layers || layers < 6 || img.key.width != img.key.height || img.key.dim != ImageDim::D2)
        {
            return None;
        }
        // Views of another dimension than the image cannot be made (a 3D
        // image only offers 2D (array) views of single levels).
        let compatible = match img.key.dim {
            ImageDim::D1 => matches!(ty, vk::ImageViewType::TYPE_1D | vk::ImageViewType::TYPE_1D_ARRAY),
            ImageDim::D2 => !matches!(ty, vk::ImageViewType::TYPE_1D | vk::ImageViewType::TYPE_1D_ARRAY | vk::ImageViewType::TYPE_3D),
            ImageDim::D3 => ty == vk::ImageViewType::TYPE_3D || (matches!(ty, vk::ImageViewType::TYPE_2D | vk::ImageViewType::TYPE_2D_ARRAY) && levels == 1 && block_dim(img.key.format) == 1),
        };
        if !compatible {
            return None;
        }
        // The clamp is an absolute level, at most the view's last one.
        let min_lod = if min_lod_ok && range.min_lod > 0.0 { range.min_lod.min((base_level + levels - 1) as f32) } else { 0.0 };
        let key = (format, aspect, [base_level, levels, base_layer, layers], ty, [components.r, components.g, components.b, components.a], min_lod.to_bits());
        Some(ViewPlan { id, key, image: img.image.image, ty, format, components, aspect, min_lod, range: SubRange { base_level, levels, base_layer, layers } })
    }

    /// The cached view of `plan` (read-only: the most recent view is not
    /// remembered).
    pub fn find_view(&self, plan: &ViewPlan) -> Option<vk::ImageView> {
        let img = self.images.get(&plan.id)?;
        if let Some((k, v)) = img.last_view {
            if k == plan.key {
                return Some(v);
            }
        }
        img.views.get(&plan.key).map(|v| v.view)
    }

    /// Create the view of `plan` (not cached: see [`Self::adopt_view`]).
    pub fn create_view(&self, plan: &ViewPlan) -> Option<ImageView> {
        let mut lod = vk::ImageViewMinLodCreateInfoEXT::default().min_lod(plan.min_lod);
        let r = plan.range;
        let mut info = vk::ImageViewCreateInfo::default()
            .image(plan.image)
            .view_type(plan.ty)
            .format(plan.format)
            .components(plan.components)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: plan.aspect,
                base_mip_level: r.base_level,
                level_count: r.levels,
                base_array_layer: r.base_layer,
                layer_count: r.layers,
            });
        if plan.min_lod > 0.0 {
            info = info.push_next(&mut lod);
        }
        ImageView::new(&self.dev, &info).ok()
    }

    /// Cache view `v` of `plan` (made by [`Self::create_view`]) with its
    /// image; a second view of the same key (made concurrently by another
    /// recording thread) lives as long as the image too. A view of an image
    /// deleted meanwhile is retired.
    pub fn adopt_view(&mut self, id: u64, plan: ViewPlan, v: ImageView) {
        let handle = v.view;
        let Some(img) = self.images.get_mut(&id) else {
            self.orphan_views.push(v);
            return;
        };
        if img.image.image != plan.image {
            self.orphan_views.push(v);
            return;
        }
        if img.views.contains_key(&plan.key) {
            img.extra_views.push(v);
        } else {
            img.views.insert(plan.key, v);
            img.last_view = Some((plan.key, handle));
        }
        self.view_ranges.insert(handle, plan.range);
        if plan.aspect == vk::ImageAspectFlags::STENCIL {
            self.stencil_views.insert(handle);
        }
    }

    /// Views of images deleted before their view was cached (retired by
    /// the caller with the open tick).
    pub fn take_orphan_views(&mut self) -> Vec<ImageView> {
        std::mem::take(&mut self.orphan_views)
    }

    /// Clear a color image (whole image, or `rect` of every layer via
    /// dynamic rendering when it does not cover the image).
    pub fn clear_color(&mut self, sched: &mut Scheduler, id: u64, value: vk::ClearColorValue, rect: Option<vk::Rect2D>) {
        crate::slice_shadow::hole(crate::slice_shadow::Hole::ImageClear);
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::IMG_CLEAR);
        let r = self.clear_color_timed(sched, id, value, rect);
        sched.time_item_end(t);
        r
    }

    fn clear_color_timed(&mut self, sched: &mut Scheduler, id: u64, value: vk::ClearColorValue, rect: Option<vk::Rect2D>) {
        self.changed(id);
        let Some(img) = self.images.get(&id) else { return };
        let (w, h, layers, format, usage) = (img.key.width, img.key.height, img.key.slices(), img.key.format, img.image.usage);
        let full = img.image.mip_levels > 1
            || rect.is_none_or(|r| r.offset.x <= 0 && r.offset.y <= 0 && r.extent.width as i32 + r.offset.x >= w as i32 && r.extent.height as i32 + r.offset.y >= h as i32);
        let range = img.full_range();
        let image = img.image.image;
        if full || !usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
            self.transit(sched, id, TRANSFER_DST, None);
            let cb = sched.ordered_cmd();
            // SAFETY: recording buffer; image in TRANSFER_DST layout.
            unsafe { self.dev.device.cmd_clear_color_image(cb, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &value, &[range]) };
        } else if let Some(view) = self.view(id, format, vk::ImageAspectFlags::COLOR, 0, layers) {
            let r = rect.unwrap();
            self.transit(sched, id, COLOR_ATTACHMENT, Some(SubRange { base_level: 0, levels: 1, base_layer: 0, layers }));
            let attachment = vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(COLOR_ATTACHMENT.0)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE);
            let info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: w, height: h } })
                .layer_count(layers)
                .color_attachments(std::slice::from_ref(&attachment));
            let clear = vk::ClearAttachment { aspect_mask: vk::ImageAspectFlags::COLOR, color_attachment: 0, clear_value: vk::ClearValue { color: value } };
            let cr = vk::ClearRect { rect: clamp_rect(r, w, h), base_array_layer: 0, layer_count: layers };
            let cb = sched.ordered_cmd();
            // SAFETY: recording buffer; attachment view of an image in COLOR_ATTACHMENT layout.
            unsafe {
                self.dev.device.cmd_begin_rendering(cb, &info);
                self.dev.device.cmd_clear_attachments(cb, &[clear], &[cr]);
                self.dev.device.cmd_end_rendering(cb);
            }
        }
        self.mark_gpu_written(id, sched.current_tick());
        self.stats.clears += 1;
    }

    /// Clear every layer of level 0 through a colour attachment
    /// (`vkCmdClearAttachments`, which conditional rendering governs). The
    /// caller marks the image written. False when the image is no colour
    /// attachment.
    pub fn clear_color_attachment(&mut self, sched: &mut Scheduler, id: u64, value: vk::ClearColorValue, base_layer: u32, layers: u32) -> bool {
        crate::slice_shadow::hole(crate::slice_shadow::Hole::ImageClear);
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::IMG_CLEAR);
        let r = self.clear_color_attachment_timed(sched, id, value, base_layer, layers);
        sched.time_item_end(t);
        r
    }

    fn clear_color_attachment_timed(&mut self, sched: &mut Scheduler, id: u64, value: vk::ClearColorValue, base_layer: u32, layers: u32) -> bool {
        self.changed(id);
        let Some(img) = self.images.get(&id) else { return false };
        let (w, h, slices, format, usage) = (img.key.width, img.key.height, img.key.slices(), img.key.format, img.image.usage);
        if !usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) || base_layer >= slices {
            return false;
        }
        let layers = layers.clamp(1, slices - base_layer);
        let Some(view) = self.view(id, format, vk::ImageAspectFlags::COLOR, base_layer, layers) else { return false };
        self.transit(sched, id, COLOR_ATTACHMENT, Some(SubRange { base_level: 0, levels: 1, base_layer, layers }));
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(COLOR_ATTACHMENT.0)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);
        let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: w, height: h } };
        let info = vk::RenderingInfo::default().render_area(area).layer_count(layers).color_attachments(std::slice::from_ref(&attachment));
        let clear = vk::ClearAttachment { aspect_mask: vk::ImageAspectFlags::COLOR, color_attachment: 0, clear_value: vk::ClearValue { color: value } };
        let cr = vk::ClearRect { rect: area, base_array_layer: 0, layer_count: layers };
        let cb = sched.ordered_cmd();
        // SAFETY: recording buffer; attachment view of an image in COLOR_ATTACHMENT layout.
        unsafe {
            self.dev.device.cmd_begin_rendering(cb, &info);
            self.dev.device.cmd_clear_attachments(cb, &[clear], &[cr]);
            self.dev.device.cmd_end_rendering(cb);
        }
        self.stats.clears += 1;
        true
    }

    /// Conditional clears of single layers of level 0 (depth slices of a
    /// volume) in one rendering scope over `[base_layer, base_layer +
    /// layers)`: each `(layer, value, buffer, offset)` clears its layer to
    /// `value` when the predicate word at `offset` of `buffer` is non-zero
    /// (`vkCmdClearAttachments` under conditional rendering). The caller
    /// orders the predicates before and marks the image written. False,
    /// with nothing recorded, when the image has no attachment view of the
    /// layers.
    pub fn conditional_layer_clears(
        &mut self,
        sched: &mut Scheduler,
        cond: &ash::ext::conditional_rendering::Device,
        id: u64,
        base_layer: u32,
        layers: u32,
        clears: &[(u32, vk::ClearColorValue, vk::Buffer, u64)],
    ) -> bool {
        let Some(img) = self.images.get(&id) else { return false };
        let (w, h, slices, format, usage) = (img.key.width, img.key.height, img.key.slices(), img.key.format, img.image.usage);
        if !usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) || layers == 0 || base_layer + layers > slices || clears.iter().any(|c| c.0 < base_layer || c.0 >= base_layer + layers) {
            return false;
        }
        let Some(view) = self.view(id, format, vk::ImageAspectFlags::COLOR, base_layer, layers) else { return false };
        if self.view_subresources(view).is_none_or(|r| r.base_layer != base_layer || r.layers != layers) {
            return false;
        }
        crate::slice_shadow::hole(crate::slice_shadow::Hole::ImageClear);
        self.changed(id);
        self.transit(sched, id, COLOR_ATTACHMENT, Some(SubRange { base_level: 0, levels: 1, base_layer, layers }));
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(COLOR_ATTACHMENT.0)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);
        let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: w, height: h } };
        let info = vk::RenderingInfo::default().render_area(area).layer_count(layers).color_attachments(std::slice::from_ref(&attachment));
        let cb = sched.ordered_cmd();
        // SAFETY: recording buffer; attachment view of layers in
        // COLOR_ATTACHMENT layout; each conditional block begins and ends
        // inside the rendering scope; predicate words are 4-byte aligned.
        unsafe {
            self.dev.device.cmd_begin_rendering(cb, &info);
            for &(layer, value, buffer, offset) in clears {
                let begin = vk::ConditionalRenderingBeginInfoEXT::default().buffer(buffer).offset(offset);
                (cond.fp().cmd_begin_conditional_rendering_ext)(cb, &begin);
                let clear = vk::ClearAttachment { aspect_mask: vk::ImageAspectFlags::COLOR, color_attachment: 0, clear_value: vk::ClearValue { color: value } };
                let cr = vk::ClearRect { rect: area, base_array_layer: layer - base_layer, layer_count: 1 };
                self.dev.device.cmd_clear_attachments(cb, &[clear], &[cr]);
                (cond.fp().cmd_end_conditional_rendering_ext)(cb);
            }
            self.dev.device.cmd_end_rendering(cb);
        }
        self.stats.clears += 1;
        true
    }

    /// Clear `layers` layers from `base_layer` of every level of colour
    /// image `id` (depth slices of a volume, through an attachment view).
    pub fn clear_color_layers(&mut self, sched: &mut Scheduler, id: u64, value: vk::ClearColorValue, base_layer: u32, layers: u32) {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::IMG_CLEAR);
        let r = self.clear_color_layers_timed(sched, id, value, base_layer, layers);
        sched.time_item_end(t);
        r
    }

    fn clear_color_layers_timed(&mut self, sched: &mut Scheduler, id: u64, value: vk::ClearColorValue, base_layer: u32, layers: u32) {
        let Some(img) = self.images.get(&id) else { return };
        let (slices, levels, volume, format, usage, image) = (img.key.slices(), img.image.mip_levels, img.key.dim == ImageDim::D3, img.key.format, img.image.usage, img.image.image);
        let (w, h) = (img.key.width, img.key.height);
        if base_layer == 0 && layers >= slices {
            self.clear_color(sched, id, value, None);
            return;
        }
        if base_layer >= slices {
            return;
        }
        let layers = layers.min(slices - base_layer);
        crate::slice_shadow::hole(crate::slice_shadow::Hole::ImageClear);
        self.changed(id);
        if !volume {
            let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: levels, base_array_layer: base_layer, layer_count: layers };
            self.transit(sched, id, TRANSFER_DST, Some(SubRange { base_level: 0, levels, base_layer, layers }));
            let cb = sched.ordered_cmd();
            // SAFETY: recording buffer; the layers are in TRANSFER_DST layout.
            unsafe { self.dev.device.cmd_clear_color_image(cb, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &value, &[range]) };
        } else if usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
            // Depth slices of a volume: a clearing load of a 2D-array view
            // of them at level 0.
            let Some(view) = self.view(id, format, vk::ImageAspectFlags::COLOR, base_layer, layers) else { return };
            self.transit(sched, id, COLOR_ATTACHMENT, Some(SubRange { base_level: 0, levels: 1, base_layer, layers }));
            let attachment = vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(COLOR_ATTACHMENT.0)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: value });
            let info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: w, height: h } })
                .layer_count(layers)
                .color_attachments(std::slice::from_ref(&attachment));
            let cb = sched.ordered_cmd();
            // SAFETY: recording buffer; attachment view of slices in COLOR_ATTACHMENT layout.
            unsafe {
                self.dev.device.cmd_begin_rendering(cb, &info);
                self.dev.device.cmd_end_rendering(cb);
            }
        } else {
            self.clear_color(sched, id, value, None);
            return;
        }
        self.mark_gpu_written(id, sched.current_tick());
        self.stats.clears += 1;
    }

    /// Clear depth and/or stencil of a depth image: `layers` ((first,
    /// count); None: all) of every level, or `rect` of them through an
    /// attachment.
    pub fn clear_depth_stencil(&mut self, sched: &mut Scheduler, id: u64, depth: Option<f32>, stencil: Option<u8>, rect: Option<vk::Rect2D>, layers: Option<(u32, u32)>) {
        crate::slice_shadow::hole(crate::slice_shadow::Hole::ImageClear);
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::IMG_CLEAR);
        let r = self.clear_depth_stencil_timed(sched, id, depth, stencil, rect, layers);
        sched.time_item_end(t);
        r
    }

    fn clear_depth_stencil_timed(&mut self, sched: &mut Scheduler, id: u64, depth: Option<f32>, stencil: Option<u8>, rect: Option<vk::Rect2D>, layers: Option<(u32, u32)>) {
        self.changed(id);
        let Some(img) = self.images.get(&id) else { return };
        let (w, h, slices, format, usage) = (img.key.width, img.key.height, img.key.slices(), img.key.format, img.image.usage);
        let (base_layer, layers) = match layers {
            Some((b, n)) if b < slices => (b, n.clamp(1, slices - b)),
            Some(_) => return,
            None => (0, slices),
        };
        let levels = img.image.mip_levels;
        let mut aspect = vk::ImageAspectFlags::empty();
        if depth.is_some() && img.image.aspect.contains(vk::ImageAspectFlags::DEPTH) {
            aspect |= vk::ImageAspectFlags::DEPTH;
        }
        if stencil.is_some() && img.image.aspect.contains(vk::ImageAspectFlags::STENCIL) {
            aspect |= vk::ImageAspectFlags::STENCIL;
        }
        if aspect.is_empty() {
            return;
        }
        let value = vk::ClearDepthStencilValue { depth: depth.unwrap_or(0.0).clamp(0.0, 1.0), stencil: stencil.unwrap_or(0) as u32 };
        let full = levels > 1
            || rect.is_none_or(|r| r.offset.x <= 0 && r.offset.y <= 0 && r.extent.width as i32 + r.offset.x >= w as i32 && r.extent.height as i32 + r.offset.y >= h as i32);
        let image = img.image.image;
        if full || !usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
            let range = vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: levels, base_array_layer: base_layer, layer_count: layers };
            self.transit(sched, id, TRANSFER_DST, Some(SubRange { base_level: 0, levels, base_layer, layers }));
            let cb = sched.ordered_cmd();
            // SAFETY: recording buffer; the layers are in TRANSFER_DST layout.
            unsafe { self.dev.device.cmd_clear_depth_stencil_image(cb, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &value, &[range]) };
        } else if let Some(view) = self.view(id, format, crate::device::aspect_of(format), base_layer, layers) {
            self.transit(sched, id, DEPTH_ATTACHMENT, Some(SubRange { base_level: 0, levels: 1, base_layer, layers }));

            let attachment = vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(DEPTH_ATTACHMENT.0)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE);
            let has_depth = crate::device::aspect_of(format).contains(vk::ImageAspectFlags::DEPTH);
            let has_stencil = crate::device::aspect_of(format).contains(vk::ImageAspectFlags::STENCIL);
            let mut info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: w, height: h } })
                .layer_count(layers);
            if has_depth {
                info = info.depth_attachment(&attachment);
            }
            if has_stencil {
                info = info.stencil_attachment(&attachment);
            }
            let clear = vk::ClearAttachment { aspect_mask: aspect, color_attachment: 0, clear_value: vk::ClearValue { depth_stencil: value } };
            let cr = vk::ClearRect { rect: clamp_rect(rect.unwrap(), w, h), base_array_layer: 0, layer_count: layers };
            let cb = sched.ordered_cmd();
            // SAFETY: recording buffer; attachment view of an image in DEPTH_ATTACHMENT layout.
            unsafe {
                self.dev.device.cmd_begin_rendering(cb, &info);
                self.dev.device.cmd_clear_attachments(cb, &[clear], &[cr]);
                self.dev.device.cmd_end_rendering(cb);
            }
        }
        self.mark_gpu_written(id, sched.current_tick());
        self.stats.clears += 1;
    }

    /// Copy (or resolve) `layers` layers of level/first layer `from` of
    /// `src` into level/first layer `to` of `dst` (compatible formats; the
    /// smaller extent of the two levels).
    pub fn resolve(&mut self, sched: &mut Scheduler, src: u64, from: (u32, u32), dst: u64, to: (u32, u32), layers: u32) -> bool {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::IMG_COPY);
        let r = self.resolve_timed(sched, src, from, dst, to, layers);
        sched.time_item_end(t);
        r
    }

    fn resolve_timed(&mut self, sched: &mut Scheduler, src: u64, from: (u32, u32), dst: u64, to: (u32, u32), layers: u32) -> bool {
        self.changed(dst);
        let (Some(s), Some(d)) = (self.images.get(&src), self.images.get(&dst)) else { return false };
        let layers = layers.max(1);
        if from.0 >= s.image.mip_levels || to.0 >= d.image.mip_levels || from.1 + layers > s.key.layers || to.1 + layers > d.key.layers || (src == dst && from == to) {
            return false;
        }
        let w = (s.key.width >> from.0).max(1).min((d.key.width >> to.0).max(1));
        let h = (s.key.height >> from.0).max(1).min((d.key.height >> to.0).max(1));
        let sub = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: from.0, base_array_layer: from.1, layer_count: layers };
        let dsub = vk::ImageSubresourceLayers { mip_level: to.0, base_array_layer: to.1, ..sub };
        let (si, di, multisampled) = (s.image.image, d.image.image, s.key.samples > 1);
        let from_range = SubRange { base_level: from.0, levels: 1, base_layer: from.1, layers };
        let to_range = SubRange { base_level: to.0, levels: 1, base_layer: to.1, layers };
        self.transit_pair(sched, (src, Some(from_range)), (dst, Some(to_range)));
        let cb = sched.ordered_cmd();
        let (src_layout, dst_layout) = (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        // SAFETY: recording buffer; source subresources in TRANSFER_SRC, destination ones in TRANSFER_DST layout.
        unsafe {
            if multisampled {
                let r = vk::ImageResolve::default().src_subresource(sub).dst_subresource(dsub).extent(vk::Extent3D { width: w, height: h, depth: 1 });
                self.dev.device.cmd_resolve_image(cb, si, src_layout, di, dst_layout, &[r]);
            } else {
                let r = vk::ImageCopy::default().src_subresource(sub).dst_subresource(dsub).extent(vk::Extent3D { width: w, height: h, depth: 1 });
                self.dev.device.cmd_copy_image(cb, si, src_layout, di, dst_layout, &[r]);
            }
        }
        self.mark_gpu_written(dst, sched.current_tick());
        true
    }

    /// Copy the `extent` texels at the origin of level/layer `from` of
    /// single-sample colour image `src` into level/layer `to` of `dst`
    /// (size-compatible formats; the extent lies inside both levels), as a
    /// shader storing into `dst` would: `dst` becomes GPU-written. False
    /// when the subresources or the extent do not fit.
    pub fn copy_region(&mut self, sched: &mut Scheduler, src: u64, from: (u32, u32), dst: u64, to: (u32, u32), extent: (u32, u32)) -> bool {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::IMG_COPY);
        let r = self.copy_region_timed(sched, src, from, dst, to, extent);
        sched.time_item_end(t);
        r
    }

    fn copy_region_timed(&mut self, sched: &mut Scheduler, src: u64, from: (u32, u32), dst: u64, to: (u32, u32), extent: (u32, u32)) -> bool {
        let (Some(s), Some(d)) = (self.images.get(&src), self.images.get(&dst)) else { return false };
        let fits = |i: &CachedImage, (level, layer): (u32, u32)| {
            level < i.image.mip_levels
                && layer < i.key.slices()
                && extent.0 <= (i.key.width >> level).max(1)
                && extent.1 <= (i.key.height >> level).max(1)
                && i.key.samples == 1
                && i.image.aspect == vk::ImageAspectFlags::COLOR
        };
        if src == dst || !fits(s, from) || !fits(d, to) || extent.0 == 0 || extent.1 == 0 {
            return false;
        }
        let sub = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: from.0, base_array_layer: from.1, layer_count: 1 };
        let dsub = vk::ImageSubresourceLayers { mip_level: to.0, base_array_layer: to.1, ..sub };
        let (si, di) = (s.image.image, d.image.image);
        let from_range = SubRange { base_level: from.0, levels: 1, base_layer: from.1, layers: 1 };
        let to_range = SubRange { base_level: to.0, levels: 1, base_layer: to.1, layers: 1 };
        self.transit_pair(sched, (src, Some(from_range)), (dst, Some(to_range)));
        let r = vk::ImageCopy::default().src_subresource(sub).dst_subresource(dsub).extent(vk::Extent3D { width: extent.0, height: extent.1, depth: 1 });
        // SAFETY: recording buffer; the source subresource is in TRANSFER_SRC, the destination one in TRANSFER_DST layout.
        unsafe { self.dev.device.cmd_copy_image(sched.ordered_cmd(), si, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, di, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[r]) };
        self.mark_gpu_written(dst, sched.current_tick());
        true
    }

    /// Copy the `aspect` planes of every layer of level 0 from `src` into
    /// `dst` (same format and extent).
    pub fn copy_depth_stencil(&mut self, sched: &mut Scheduler, src: u64, dst: u64, aspect: vk::ImageAspectFlags) -> bool {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::IMG_COPY);
        let r = self.copy_depth_stencil_timed(sched, src, dst, aspect);
        sched.time_item_end(t);
        r
    }

    fn copy_depth_stencil_timed(&mut self, sched: &mut Scheduler, src: u64, dst: u64, aspect: vk::ImageAspectFlags) -> bool {
        self.changed(dst);
        let (Some(s), Some(d)) = (self.images.get(&src), self.images.get(&dst)) else { return false };
        if src == dst || s.key.format != d.key.format || s.key.width != d.key.width || s.key.height != d.key.height || s.key.samples != d.key.samples {
            return false;
        }
        let aspect = aspect & s.image.aspect;
        if aspect.is_empty() {
            return false;
        }
        let layers = s.key.layers.min(d.key.layers);
        let extent = vk::Extent3D { width: s.key.width, height: s.key.height, depth: 1 };
        let mut regions = smallvec::SmallVec::<[vk::ImageCopy; 2]>::new();
        for a in [vk::ImageAspectFlags::DEPTH, vk::ImageAspectFlags::STENCIL] {
            if aspect.contains(a) {
                let sub = vk::ImageSubresourceLayers { aspect_mask: a, mip_level: 0, base_array_layer: 0, layer_count: layers };
                regions.push(vk::ImageCopy::default().src_subresource(sub).dst_subresource(sub).extent(extent));
            }
        }
        let (si, di) = (s.image.image, d.image.image);
        self.transit_pair(sched, (src, None), (dst, None));
        let cb = sched.ordered_cmd();
        // SAFETY: recording buffer; source in TRANSFER_SRC, destination in TRANSFER_DST layout.
        unsafe { self.dev.device.cmd_copy_image(cb, si, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, di, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions) };
        self.mark_gpu_written(dst, sched.current_tick());

        true
    }

    /// Copy the 32-bit float depth of layer `src_layer` (level 0) of the
    /// single-sample depth image `src` into layer `dst_layer` of the R32
    /// float colour image `dst` of the same extent (the DB's depth copy into
    /// colour target 0). A depth aspect cannot be copied into a colour image
    /// directly: the texels pass through a device buffer.
    pub fn copy_depth_to_color(&mut self, sched: &mut Scheduler, src: u64, src_layer: u32, dst: u64, dst_layer: u32) -> bool {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::IMG_COPY);
        let r = self.copy_depth_to_color_timed(sched, src, src_layer, dst, dst_layer);
        sched.time_item_end(t);
        r
    }

    fn copy_depth_to_color_timed(&mut self, sched: &mut Scheduler, src: u64, src_layer: u32, dst: u64, dst_layer: u32) -> bool {
        let (Some(s), Some(d)) = (self.images.get(&src), self.images.get(&dst)) else { return false };
        let float_depth = matches!(s.image.format, vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT);
        if src == dst
            || !float_depth
            || d.image.format != vk::Format::R32_SFLOAT
            || s.key.samples != 1
            || d.key.samples != 1
            || (s.key.width, s.key.height) != (d.key.width, d.key.height)
            || src_layer >= s.image.layers
            || dst_layer >= d.image.layers
        {
            return false;
        }
        let (si, di, width, height) = (s.image.image, d.image.image, s.key.width, s.key.height);
        let bytes = width as u64 * height as u64 * 4;
        let Ok(tmp) = Buffer::new(&self.dev, bytes, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, MemoryUse::Device) else { return false };
        let buffer = tmp.buffer;
        sched.retire_sized(tmp, bytes);
        self.changed(dst);
        let one = |layer: u32| Some(SubRange { base_level: 0, levels: 1, base_layer: layer, layers: 1 });
        self.transit_pair(sched, (src, one(src_layer)), (dst, one(dst_layer)));
        let extent = vk::Extent3D { width, height, depth: 1 };
        let out = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::DEPTH, mip_level: 0, base_array_layer: src_layer, layer_count: 1 })
            .image_extent(extent);
        let cb = sched.ordered_cmd();
        // SAFETY: recording buffer; source in TRANSFER_SRC layout, buffer large enough for the plane.
        unsafe { self.dev.device.cmd_copy_image_to_buffer(cb, si, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buffer, &[out]) };
        sched.memory_barrier((vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE), (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ));
        let back = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: dst_layer, layer_count: 1 })
            .image_extent(extent);
        let cb = sched.ordered_cmd();
        // SAFETY: recording buffer; destination in TRANSFER_DST layout.
        unsafe { self.dev.device.cmd_copy_buffer_to_image(cb, buffer, di, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[back]) };
        self.mark_gpu_written(dst, sched.current_tick());
        true
    }

    /// The watch entries a GPU write through a buffer over `[addr,
    /// addr+size)` makes stale: only the images the page index names over
    /// the range are looked at (`RPS5E_INDEXED_INVALIDATE=0`: every entry).
    fn invalidate_table(&self, addr: u64, size: u64) -> usize {
        if indexed_invalidate_enabled() {
            let ids = self.overlapping(addr, addr.saturating_add(size), u64::MAX);
            if ids.is_empty() {
                return 0;
            }
            self.table.invalidate_range_of(&ids, addr, size)
        } else {
            self.table.invalidate_range(addr, size)
        }
    }

    /// GPU writes through a buffer made the images over a range stale.
    pub fn invalidate_range(&mut self, addr: u64, size: u64) {
        let hit = self.invalidate_table(addr, size);
        if hit > 0 && crate::backend::profile::on() {
            *self.invalidate_counts.entry((addr, size)).or_default() += 1;
        }
    }

    /// A draw or dispatch wrote `[addr, addr+size)` through a buffer (the
    /// range is the V#'s whole footprint, clamped to readable memory, not
    /// the bytes the shader stored). By default every image over the range
    /// takes it from the buffer at its next use, as for a command-processor
    /// write. `RPS5E_SHADER_WRITE_INVALIDATE=base` only invalidates images
    /// that start at the buffer's address (the reference renderer's rule),
    /// for A/B checks of render targets lost to oversized footprints. The
    /// first GPU-modified images hit are logged.
    pub fn invalidate_shader_write(&mut self, addr: u64, size: u64) {
        self.drop_fills(addr, size);
        static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        static SCANS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let base_only = shader_write_rule_base();
        let log = LOGGED.load(std::sync::atomic::Ordering::Relaxed) < 32;
        if !base_only {
            // Looked at (for the log) only when the write hit an image.
            let hit = self.invalidate_table(addr, size);
            if hit > 0 && crate::backend::profile::on() {
                *self.invalidate_counts.entry((addr, size)).or_default() += 1;
            }
            if hit == 0 || !log || SCANS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 100_000 {
                return;
            }
        }
        let end = addr.saturating_add(size);
        let mut at_base = Vec::new();
        for id in self.overlapping(addr, end, u64::MAX) {
            let Some(i) = self.images.get(&id) else { continue };
            let (b, e) = i.main_range();
            if i.gpu_modified && b != addr && LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 32 {
                log::info!(
                    "ps5gpu: shader buffer write {addr:#x}+{size:#x} covers GPU-modified image #{id} {b:#x}..{e:#x} {:?} {}x{}{}",
                    i.key.format,
                    i.key.width,
                    i.key.height,
                    if base_only { " (left alone: RPS5E_SHADER_WRITE_INVALIDATE=base)" } else { ": its pages are taken from the buffer at its next use" }
                );
            }
            if b == addr {
                at_base.push((b.max(addr), e.min(end)));
            }
        }
        if !base_only {
            return;
        }
        for (b, e) in at_base {
            if b < e {
                self.invalidate_range(b, e - b);
            }
        }
    }

    /// The GPU filled `[begin, end)` with a 32-bit pattern (the SDK fill
    /// kernel, run as a buffer fill). A zero fill wholly covering an image
    /// clears the image instead of reloading it; other overlapping images
    /// are loaded again from the filled memory.
    pub fn filled(&mut self, sched: &mut Scheduler, begin: u64, end: u64, pattern: u32) {
        self.epoch += 1;
        for id in self.overlapping(begin, end, u64::MAX) {
            let Some(img) = self.images.get(&id) else { continue };
            let key = img.key;
            let inside = key.planes().all(|p| p.addr >= begin && p.addr + p.size() <= end);
            if pattern == 0 && inside && key.samples == 1 {
                if key.is_depth() {
                    self.clear_depth_stencil(sched, id, Some(0.0), key.stencil.map(|_| 0), None, None);
                } else {
                    self.clear_color(sched, id, vk::ClearColorValue { uint32: [0; 4] }, None);
                }
                // The image now matches the filled memory.
                let tracked = self.memory.is_watchable(key.main.addr, key.size());
                self.table.rearm(id, tracked);
                continue;
            }
            // A fill of exactly one plane with a value the plane's format
            // expresses is a clear of that plane.
            if key.samples == 1 && self.clear_from_fill(sched, id, &key, begin, end, pattern) {
                continue;
            }
            for p in key.planes() {
                let (b, e) = (p.addr.max(begin), (p.addr + p.size()).min(end));
                if b < e {
                    self.table.invalidate_range(b, e - b);
                }
            }
        }
    }

    /// Clear the plane of image `id` that `[begin, end)` covers exactly
    /// with the dword `pattern` decoded in its format: color, 32-bit float
    /// depth, or a stencil byte. False when no plane matches or the value
    /// is not one the format stores. A depth/stencil image whose other
    /// plane is not refreshed keeps it as it is.
    fn clear_from_fill(&mut self, sched: &mut Scheduler, id: u64, key: &ImageKey, begin: u64, end: u64, pattern: u32) -> bool {
        let exact = |p: &Plane| p.addr == begin && p.addr + p.size() == end;
        let main = exact(&key.main);
        let stencil = key.stencil.as_ref().is_some_and(exact);
        if !main && !stencil {
            return false;
        }
        // The plane left alone must already be current in the image.
        let other_stale = |this: &Self| !this.images.get(&id).is_some_and(|i| i.initialized) || this.table.state(id).1;
        if !key.is_depth() {
            let Some(value) = crate::render::formats::decode_dword_fill(key.format, pattern) else { return false };
            self.clear_color(sched, id, value, None);
        } else if main {
            let Some(depth) = crate::render::formats::decode_depth_fill(key.format, pattern) else { return false };
            if key.stencil.is_some() && other_stale(self) {
                return false;
            }
            self.clear_depth_stencil(sched, id, Some(depth), None, None, None);
        } else {
            let Some(value) = crate::render::formats::decode_stencil_fill(pattern) else { return false };
            if other_stale(self) {
                return false;
            }
            self.clear_depth_stencil(sched, id, None, Some(value), None, None);
        }
        // The cleared plane matches the filled memory; the image keeps
        // its watch over the rest.
        let tracked = self.memory.is_watchable(key.main.addr, key.size());
        self.table.rearm(id, tracked);
        true
    }

    /// `RPS5E_GPU_PROFILE`: GPU-written ranges that invalidated images most often.
    pub fn stale_top(&self, n: usize) -> String {
        let mut v: Vec<_> = self.stale_reasons.iter().map(|(k, c)| (*c, *k)).collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        v.iter().take(n).map(|(c, (a, r))| format!("{a:#x}:{r}={c}")).collect::<Vec<_>>().join(" ")
    }

    pub fn invalidate_top(&self, n: usize) -> String {
        let mut v: Vec<_> = self.invalidate_counts.iter().map(|(k, c)| (*c, *k)).collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        v.iter().take(n).map(|(c, (a, s))| format!("{a:#x}+{s:#x}={c}")).collect::<Vec<_>>().join(" ")
    }

    /// Mark an image written by the GPU (a draw rendered into it): its
    /// contents become the newest of its memory. Overlapping images it
    /// fully covers are dropped once idle ([`ImageCache::collect`]).
    pub fn mark_gpu_written(&mut self, id: u64, tick: u64) {
        if let Some(i) = self.images.get(&id) {
            static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if i.key.layers >= 6 && !i.gpu_modified && LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 32 {
                log::info!("ps5gpu: image #{id} {:#x} {:?} {}x{}x{} L{} written by the GPU (tick {tick})", i.key.main.addr, i.key.format, i.key.width, i.key.height, i.key.layers, i.key.levels);
            }
        }
        // Writing the image that already holds the newest contents of its
        // memory changes no lookup.
        if let Some(i) = self.images.get(&id) {
            let newest = i.gpu_modified && i.initialized && i.aliases.iter().all(|a| self.images.get(a).is_none_or(|x| x.seq < i.seq));
            if !newest || i.merged {
                crate::slice_shadow::hole(crate::slice_shadow::Hole::TargetChange);
                self.changed(id);
            }
        }
        self.seq += 1;
        let Some(i) = self.images.get_mut(&id) else { return };
        if !i.gpu_modified {
            self.table.set_partial(id, true);
        }
        if i.merged {
            // GPU work renders into it after the CPU wrote: the memory is
            // still used as this image (memory reused for other data is not
            // rendered into), so it may be written back again.
            i.merged = false;
            self.table.untouch(id);
        }
        i.gpu_modified = true;
        i.initialized = true;
        i.last_use = tick;
        i.seq = self.seq;
        if i.aliases.is_empty() {
            return;
        }
        let (b, e) = i.main_range();
        let i = &self.images[&id];
        for a in &i.aliases {
            let Some(x) = self.images.get(a) else { continue };
            if x.key.planes().all(|p| inside(p.addr, p.addr + p.size(), b, e)) {
                self.superseded.insert(x.id, x.seq);
            }
        }
    }

    /// A CPU access to read-watched pages (`hits` from
    /// [`WatchTable::take_read_backs`]): write the images back, download
    /// their pages into guest memory, and drop the watches no buffer needs
    /// (the download cleared the pages a buffer held GPU data on).
    pub fn resolve_read_backs(
        &mut self,
        sched: &mut Scheduler,
        buffers: &mut BufferCache,
        tracker: &crate::memory::tracker::GuestPageTracker,
        hits: &[(u64, u64, u64)],
        fault: bool,
    ) {
        let pages = self.table.pages().clone();
        for &(begin, end, id) in hits {
            if self.read_back_linear(sched, id) {
                tracker.note_written(begin, end - begin);
            } else {
                // Not a plain linear image: through its buffer mirror.
                self.writeback(sched, buffers, id);
                buffers.download(sched, begin, end - begin);
            }
            for page in (begin..end).step_by(PAGE_BYTES as usize) {
                if pages.watchers(page).0 && !tracker.has_gpu_dirty_pages(page, PAGE_BYTES) {
                    if fault {
                        pages.remove_watch_in_fault(page, PAGE_BYTES, true);
                    } else {
                        pages.remove_watch(page, PAGE_BYTES, true);
                    }
                }
            }
        }
    }

    /// Copy the texels of linear image `id` straight into guest memory
    /// (waiting for the GPU), exactly the bytes the image covers: other
    /// data on its first and last pages (labels the command processor
    /// writes, say) stays as it is. False when the image is not a plain
    /// single-sample linear colour image or may not be written back.
    fn read_back_linear(&mut self, sched: &mut Scheduler, id: u64) -> bool {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::IMG_READBACK);
        let r = self.read_back_linear_timed(sched, id);
        sched.time_item_end(t);
        r
    }

    fn read_back_linear_timed(&mut self, sched: &mut Scheduler, id: u64) -> bool {
        if !self.table.may_write_back(id) {
            return false;
        }
        let Some(img) = self.images.get(&id) else { return false };
        let key = img.key;
        let levels = img.image.mip_levels;
        let image = img.image.image;
        if key.samples != 1 || key.stencil.is_some() || key.main.swizzle != crate::tiling::SWIZZLE_LINEAR || key.main.aspect != vk::ImageAspectFlags::COLOR {
            return false;
        }
        let plane = key.main;
        if plane_transform(&key, &plane) != ColorTransform::None {
            return false;
        }
        let Some(layout) = key.layout(&plane) else { return false };
        let plan = TransferPlan::new(&key, &plane, &layout, levels);
        if !plan.linear {
            return false;
        }
        let Ok(out) = Buffer::new(&self.dev, layout.guest_bytes, vk::BufferUsageFlags::TRANSFER_DST, MemoryUse::Download) else { return false };
        before_buffer_work(sched);
        self.transit(sched, id, TRANSFER_SRC, None);
        // SAFETY: recording buffer; image in TRANSFER_SRC layout.
        unsafe { self.dev.device.cmd_copy_image_to_buffer(sched.ordered_cmd(), image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, out.buffer, &plan.copies) };
        after_buffer_writes(sched);
        sched.flush_and_wait();
        out.invalidate(0, layout.guest_bytes);
        let bytes = out.bytes_mut();
        let bpe = plane.bytes_per_element as u64;
        let mut ok = true;
        for c in &plan.copies {
            let row_len = if c.buffer_row_length == 0 { c.image_extent.width } else { c.buffer_row_length } as u64 * bpe;
            let rows_per_slice = if c.buffer_image_height == 0 { c.image_extent.height } else { c.buffer_image_height } as u64;
            let slices = c.image_extent.depth.max(1) as u64 * c.image_subresource.layer_count.max(1) as u64;
            let row_bytes = c.image_extent.width as u64 * bpe;
            for slice in 0..slices {
                for row in 0..c.image_extent.height as u64 {
                    let off = c.buffer_offset + (slice * rows_per_slice + row) * row_len;
                    if off + row_bytes > bytes.len() as u64 {
                        continue;
                    }
                    ok &= self.memory.write_backing(plane.addr + off, &bytes[off as usize..(off + row_bytes) as usize]);
                }
            }
        }
        sched.retire(out);
        self.stats.readbacks += 1;
        ok
    }

    /// A shader stored to image `id` (after [`Self::mark_gpu_written`]): a
    /// linear one reaches guest memory when the CPU touches it.
    pub fn note_storage_write(&mut self, id: u64) {
        crate::slice_shadow::hole(crate::slice_shadow::Hole::StorageImage);
        let Some(i) = self.images.get(&id) else { return };
        if i.key.main.swizzle != crate::tiling::SWIZZLE_LINEAR || i.key.samples != 1 {
            return;
        }
        match self.readback_linear {
            LinearReadback::Off => {}
            LinearReadback::Eager => {
                if !self.downloads.contains(&id) {
                    self.downloads.push(id);
                }
            }
            // Written back when the CPU touches its pages (fault): images the
            // CPU never reads cost nothing.
            LinearReadback::OnAccess => {
                self.storage_ticks.insert(id, i.last_use);
                let planes: Vec<(u64, u64)> = i.key.planes().map(|p| (p.addr, p.addr + p.size())).collect();
                for (b, e) in planes {
                    self.table.watch_reads(id, b, e);
                }
            }
        }
    }

    /// The GPU-modified color image whose surface starts at `addr` and is
    /// at least `width` x `height` (display flips); the newest of several.
    pub fn find_display(&self, addr: u64, width: u32, height: u32) -> Option<u64> {
        self.at(addr)
            .filter(|i| i.gpu_modified && !i.key.is_depth() && i.key.width >= width && i.key.height >= height)
            .max_by_key(|i| i.seq)
            .map(|i| i.id)
    }

    /// The guest unmapped `[addr, addr+size)`: drop every image in it.
    pub fn unmap(&mut self, sched: &mut Scheduler, addr: u64, size: u64) {
        self.epoch += 1;
        let end = addr + size;
        let mut partly = false;
        for id in self.overlapping(addr, end, 0) {
            // An image keeping committed pages outside the range (a
            // partially resident texture losing some pages) stays: the
            // pages unmapped read as zeros from now on, which its next use
            // takes like a mapping change.
            let keeps = self.images.get(&id).is_some_and(|i| {
                let (b, e) = i.main_range();
                i.key.stencil.is_none() && (b < addr || e > end)
            });
            if keeps {
                partly = true;
                continue;
            }
            self.delete(sched, id);
            self.stats.drops[3] += 1;
        }
        if partly {
            self.table.mark_mapped(addr, size);
        }
        self.metadata.remove_starting_in(addr, addr.saturating_add(size));
    }

    /// A constant fill of guest memory (metadata clears are remembered).
    pub fn note_fill(&mut self, addr: u64, size: u64, value: u32) {
        self.metadata.insert(addr, MetadataFill { size, value });
        self.metadata_generation += 1;
        if let Some(&id) = self.sampled_dcc_owners.get(&addr) {
            self.changed(id);
        }
    }

    /// Writes over `[addr, addr+size)` (shader stores, command-processor
    /// copies) replace the bytes of pending metadata fills there: those
    /// fills no longer describe the memory (a later surface whose metadata
    /// lands there must not take them as its clear).
    pub fn drop_fills(&mut self, addr: u64, size: u64) {
        if self.metadata.is_empty() || !fill_invalidation_enabled() {
            return;
        }
        if self.metadata.drop_overlapping(addr, addr.saturating_add(size), fill_index_enabled()) {
            self.metadata_generation += 1;
        }
    }

    /// Changes whenever a metadata fill is recorded.
    pub fn metadata_generation(&self) -> u64 {
        self.metadata_generation
    }

    /// Take the pending metadata fill recorded at `addr`.
    pub fn take_fill(&mut self, addr: u64) -> Option<MetadataFill> {
        let fill = self.metadata.remove(&addr);
        if let Some(f) = &fill {
            crate::slice_shadow::hole(crate::slice_shadow::Hole::TakeFill);
            crate::slice_shadow::changed(addr, f.size);
        }
        fill
    }

    /// A frame was presented (eviction ages are counted in frames).
    pub fn end_frame(&mut self, tick: u64) {
        if self.frame_ticks.len() > EVICT_FRAMES {
            self.frame_ticks.pop_front();
        }
        self.frame_ticks.push_back(tick);
    }

    /// Images last used before this tick have been idle for at least
    /// [`EVICT_FRAMES`] frames.
    fn eviction_cutoff(&self, tick: u64) -> u64 {
        self.cutoff_after(tick, EVICT_FRAMES)
    }

    /// Images last used before this tick have been idle for `frames`
    /// presented frames.
    pub fn cutoff_after(&self, tick: u64, frames: usize) -> u64 {
        if self.frame_ticks.len() > frames {
            let f = self.frame_ticks[self.frame_ticks.len() - 1 - frames];
            // Frames may stop (no presents through the cache): ticks bound
            // the age then.
            f.max(tick.saturating_sub(EVICT_TICKS * 4))
        } else {
            tick.saturating_sub(EVICT_TICKS)
        }
    }

    /// Drop idle images a newer image fully overwrote since (their bytes
    /// live on in it), then, above the budget, images idle for
    /// [`EVICT_FRAMES`] frames, least recently used first. Clean (or
    /// CPU-overwritten) images just go; GPU-modified ones are written back
    /// to guest memory first ([`Self::evict_modified`]), and their buffer
    /// mirror is settled so it can be released too. Still a quarter above
    /// the budget, images idle for [`EVICT_FRAMES_OVER`] frames go too.
    /// Images used in the last frames are never evicted, so a working set
    /// above the budget is not reloaded every frame.
    pub fn collect(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, budget: u64) {
        // What it changes moves the lookups itself (deletions, write-backs).
        if !image_generations_enabled() {
            self.epoch += 1;
        }
        let tick = sched.current_tick();
        let mut dead = Vec::new();
        self.superseded.retain(|&id, &mut seq| match self.images.get(&id) {
            Some(x) if x.seq != seq => false,
            Some(x) if x.last_use + IDLE_TICKS < tick => {
                dead.push(id);
                false
            }
            Some(_) => true,
            None => false,
        });
        for id in dead {
            self.delete(sched, id);
            self.stats.drops[1] += 1;
        }
        let total: u64 = self.images.values().map(|i| i.key.size()).sum();
        self.stats.bytes = total;
        self.stats.modified_bytes = self.images.values().filter(|i| i.gpu_modified).map(|i| i.key.size()).sum();
        if total <= budget {
            return;
        }
        if crate::backend::profile::on() {
            self.report_over_budget(tick, total);
        }
        let done = sched.timeline.refresh();
        let target = budget - budget / 4;
        let mut remaining = total;
        let cutoff = self.idle_cutoff(tick, done);
        self.evict_idle(sched, buffers, cutoff, target, &mut remaining);
        if remaining > budget + budget / 4 {
            let cutoff = self.cutoff_after(tick, EVICT_FRAMES_OVER).min(done);
            self.evict_idle(sched, buffers, cutoff, budget, &mut remaining);
        }
    }

    /// `RPS5E_GPU_PROFILE`: now and then, the largest images with their
    /// idle age in presented frames and the bytes per age.
    fn report_over_budget(&mut self, tick: u64, total: u64) {
        static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 512 != 0 {
            return;
        }
        let frames = |last: u64| self.frame_ticks.iter().rev().take_while(|&&f| f > last).count();
        let mut by_age = [0u64; 18];
        let mut all: Vec<(u64, &CachedImage)> = self.images.values().map(|i| (i.key.size(), i)).collect();
        for (n, i) in &all {
            by_age[frames(i.last_use).min(17)] += n;
        }
        all.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        let top: Vec<String> = all
            .iter()
            .take(16)
            .map(|(n, i)| {
                format!(
                    "{:#x} {:?} {}x{}x{} L{} {}MiB age{}{}{}",
                    i.key.main.addr,
                    i.key.format,
                    i.key.width,
                    i.key.height,
                    i.key.layers.max(i.key.depth),
                    i.key.levels,
                    n >> 20,
                    frames(i.last_use),
                    if i.gpu_modified { " mod" } else { "" },
                    if i.aliases.is_empty() { String::new() } else { format!(" aliases{}", i.aliases.len()) }
                )
            })
            .collect();
        let ages: Vec<String> = by_age.iter().map(|b| format!("{}", b >> 20)).collect();
        log::info!(
            "ps5gpu: images over budget: {} MiB in {} images (tick {tick}); MiB by idle frames 0..17+: [{}]; largest: {}",
            total >> 20,
            self.images.len(),
            ages.join(" "),
            top.join(", ")
        );
    }

    /// Evict images last used before `cutoff`, least recently used first,
    /// until `remaining` guest bytes are at most `target`.
    fn evict_idle(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, cutoff: u64, target: u64, remaining: &mut u64) {
        // Uses mark images with the current tick, which never falls below
        // a cutoff taken earlier: once no image was idle before a cutoff,
        // none is before it (or an earlier one) again.
        if cutoff <= self.no_idle_below {
            return;
        }
        // Images a bindless table holds are in use while it is.
        let mut idle: Vec<(u64, u64, u64)> = self.images.values().filter(|i| i.last_use < cutoff && i.pins == 0).map(|i| (i.last_use, i.id, i.key.size())).collect();
        if idle.is_empty() {
            self.no_idle_below = cutoff;
            return;
        }
        idle.sort();
        for (_, id, n) in idle {
            if *remaining <= target {
                break;
            }
            let Some(img) = self.images.get(&id) else { continue };
            // Contents the CPU rewrote since are stale and just go.
            if img.gpu_modified && self.table.may_write_back(id) {
                if !self.evict_modified(sched, buffers, id) {
                    self.stats.drops[5] += 1;
                    continue;
                }
                self.stats.drops[4] += 1;
            } else {
                self.delete(sched, id);
                self.stats.drops[2] += 1;
            }
            *remaining -= n;
        }
    }

    /// Images (and buffers) last used before this tick are idle: unused for
    /// [`EVICT_FRAMES`] presented frames and done on the GPU.
    pub fn idle_cutoff(&self, tick: u64, done: u64) -> u64 {
        self.eviction_cutoff(tick).min(done)
    }

    /// Evict a GPU-modified image: write its contents back into guest
    /// memory (tiled on the GPU into the buffer mirror, which is then
    /// settled into guest memory) and drop it. Overlapping images keep the
    /// newest-wins order of the guest bytes: older GPU-modified aliases are
    /// written back before it (and become plain, reloadable copies), newer
    /// ones after it; older aliases reload from guest memory on their next
    /// use. False (nothing dropped) for shapes that cannot be written back.
    fn evict_modified(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, id: u64) -> bool {
        let Some(x) = self.images.get(&id) else { return false };
        let seq = x.seq;
        let mut older: Vec<(u64, u64)> = Vec::new();
        let mut newer: Vec<(u64, u64)> = Vec::new();
        for &a in &x.aliases {
            let Some(o) = self.images.get(&a) else { continue };
            if o.seq < seq {
                older.push((o.seq, a));
            } else if o.gpu_modified && o.initialized && self.table.may_write_back(a) {
                newer.push((o.seq, a));
            }
        }
        older.sort_unstable();
        newer.sort_unstable();
        let modified = |c: &ImageCache, a: u64| c.images.get(&a).is_some_and(|o| o.gpu_modified && o.initialized && c.table.may_write_back(a));
        let older_modified: Vec<u64> = older.iter().map(|o| o.1).filter(|&a| modified(self, a)).collect();
        let order: Vec<u64> = older_modified.iter().copied().chain(std::iter::once(id)).chain(newer.iter().map(|n| n.1)).collect();
        // Multisampled contents have no guest layout here.
        if order.iter().any(|a| self.images.get(a).is_none_or(|o| o.key.samples > 1)) {
            return false;
        }
        let mut written = Vec::with_capacity(order.len());
        for &a in &order {
            if self.writeback(sched, buffers, a) {
                written.push(a);
            } else if a == id {
                return false;
            }
        }
        for &a in &written {
            let key = self.images[&a].key;
            for plane in key.planes() {
                if let Some(layout) = key.layout(&plane) {
                    buffers.settle(sched, plane.addr, layout.guest_bytes);
                }
            }
        }
        for &(_, a) in &older {
            if written.contains(&a) {
                if let Some(o) = self.images.get_mut(&a) {
                    o.gpu_modified = false;
                }
            }
            self.table.mark_stale(a);
        }
        self.delete(sched, id);
        true
    }

    pub fn ids(&self) -> Vec<u64> {
        self.images.keys().copied().collect()
    }

    /// Before a frame capture: the images the GPU
    /// modified are written back into their buffer mirrors, oldest first
    /// (as [`Self::writeback_range`]), then every image goes except the
    /// ones whose write-back failed (multisampled images). Returns
    /// (dropped, kept).
    pub fn capture_reset(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache) -> (usize, usize) {
        let mut all: Vec<(u64, u64)> = self.images.values().map(|i| (i.seq, i.id)).collect();
        all.sort_unstable();
        let mut keep = crate::fxhash::HashSet::default();
        for &(_, id) in &all {
            let Some(i) = self.images.get(&id) else { continue };
            if i.gpu_modified && i.initialized && self.table.may_write_back(id) && !self.writeback(sched, buffers, id) {
                keep.insert(id);
            }
        }
        let mut dropped = 0;
        for (_, id) in all {
            if !keep.contains(&id) {
                self.delete(sched, id);
                dropped += 1;
            }
        }
        (dropped, keep.len())
    }

    /// Bytes an image counts for in the collection's own accounting.
    fn accounted(i: &CachedImage) -> u64 {
        i.key.planes().map(|p| p.size()).sum::<u64>().next_multiple_of(1024)
    }

    /// Copy the linear images the GPU wrote since the last call back to
    /// guest memory (unless `RPS5E_READBACK_LINEAR_IMAGES=0`): tiled into
    /// their buffer mirrors, then settled into guest memory once the GPU
    /// is done, without waiting.
    pub fn process_downloads(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache) {
        for id in std::mem::take(&mut self.downloads) {
            let Some(img) = self.images.get(&id) else { continue };
            // Contents the CPU (or a buffer) overwrote since are stale.
            if !img.gpu_modified || !self.table.may_write_back(id) {
                continue;
            }
            let key = img.key;
            if !self.writeback(sched, buffers, id) {
                continue;
            }
            for plane in key.planes() {
                if let Some(layout) = key.layout(&plane) {
                    buffers.settle(sched, plane.addr, layout.guest_bytes);
                }
            }
            self.stats.readbacks += 1;
        }
    }

    /// Memory-pressure collection ([`crate::gc`]): with the process using
    /// `usage` bytes of device memory (None: the bytes of the cached
    /// images), above the trigger mark the least recently used images
    /// idle long enough go; GPU-modified ones are written back first under
    /// pressure (tiled ones stay), and ones the CPU overwrote just go.
    pub fn collect_pressure(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, usage: Option<u64>, t: &crate::gc::Thresholds, ticks: &crate::gc::CollectionTicks) {
        let mut usage = usage.unwrap_or_else(|| self.images.values().map(Self::accounted).sum());
        if usage < t.trigger {
            return;
        }
        self.pressure_run(sched, buffers, &mut usage, t, ticks, false);
        if usage >= t.critical {
            self.pressure_run(sched, buffers, &mut usage, t, ticks, true);
        }
    }

    fn pressure_run(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, usage: &mut u64, t: &crate::gc::Thresholds, ticks: &crate::gc::CollectionTicks, allow_aggressive: bool) {
        self.stats.pressure_runs += 1;
        let (mut pressured, mut aggressive, age, mut deletions) = crate::gc::image_run(*usage, t, allow_aggressive);
        let cutoff = ticks.cutoff(age).min(self.cutoff_after(sched.current_tick(), crate::gc::idle_frames(pressured, aggressive)));
        let mut candidates: Vec<(u64, u64)> = self.images.values().filter(|i| i.last_use < cutoff).map(|i| (i.last_use, i.id)).collect();
        candidates.sort_unstable();
        candidates.truncate(deletions);
        for (_, id) in candidates {
            if deletions == 0 {
                break;
            }
            deletions -= 1;
            let Some(img) = self.images.get(&id) else { continue };
            let size = Self::accounted(img);
            if img.gpu_modified {
                // Contents the CPU overwrote since are stale and just go.
                let safe = self.table.may_write_back(id);
                let tiled = img.key.main.swizzle != crate::tiling::SWIZZLE_LINEAR;
                if safe && (tiled || !pressured) {
                    continue;
                }
                if safe {
                    if !self.evict_modified(sched, buffers, id) {
                        continue;
                    }
                    self.stats.drops[4] += 1;
                } else {
                    self.delete(sched, id);
                    self.stats.drops[2] += 1;
                }
            } else {
                self.delete(sched, id);
                self.stats.drops[2] += 1;
            }
            self.stats.pressure_drops += 1;
            *usage = usage.saturating_sub(size);
            if *usage < t.critical && aggressive {
                deletions >>= 2;
                aggressive = false;
            }
            if *usage < t.pressure && pressured {
                deletions >>= 1;
                pressured = false;
            }
        }
    }
}


fn clamp_rect(r: vk::Rect2D, w: u32, h: u32) -> vk::Rect2D {
    let x0 = r.offset.x.clamp(0, w as i32);
    let y0 = r.offset.y.clamp(0, h as i32);
    let x1 = (r.offset.x + r.extent.width as i32).clamp(x0, w as i32);
    let y1 = (r.offset.y + r.extent.height as i32).clamp(y0, h as i32);
    vk::Rect2D { offset: vk::Offset2D { x: x0, y: y0 }, extent: vk::Extent2D { width: (x1 - x0) as u32, height: (y1 - y0) as u32 } }
}

/// Copies of one plane between its guest layout and the image: buffer
/// copies (from the guest bytes directly when linear, else from a packed
/// linear buffer the GPU tiler fills per level).
struct TransferPlan {
    linear: bool,
    /// Tiler regions with their byte offset in the packed linear buffer.
    tiles: Vec<(u64, TileRegion)>,
    linear_bytes: u64,
    copies: Vec<vk::BufferImageCopy>,
}

impl TransferPlan {
    /// The plan of the bytes `[begin, end)` (offsets into the plane) of a
    /// thin, single-level, tiled 2D (array) surface: the layers they touch,
    /// and within a single layer the block rows they touch. None when the
    /// surface has another shape or the window is all of it.
    fn window(key: &ImageKey, plane: &Plane, layout: &SurfaceLayout, levels: u32, begin: u64, end: u64) -> Option<TransferPlan> {
        let linear = plane.swizzle == crate::tiling::SWIZZLE_LINEAR;
        if layout.thick || layout.volume || key.dim == ImageDim::D3 || end <= begin || layout.layer_bytes == 0 {
            return None;
        }
        let first = (begin / layout.layer_bytes).min(layout.slices as u64) as u32;
        let last = end.div_ceil(layout.layer_bytes).min(layout.slices as u64) as u32;
        if first >= last {
            return None;
        }
        let m = layout.mips[0];
        if linear || levels.min(layout.mips.len() as u32) != 1 || m.tail || last - first > 1 {
            // Whole layers (every level of each).
            return (first != 0 || last != layout.slices).then(|| Self::new_layers(key, plane, layout, levels, first, last - first));
        }
        let bd = if plane.aspect == vk::ImageAspectFlags::COLOR { block_dim(key.format) } else { 1 };
        let bpe = plane.bytes_per_element as u64;
        let (mut y0, mut y1) = (0u32, m.height);
        if last - first == 1 {
            let (_, bh, block_bytes) = crate::tiling::thin_block_extent(plane.swizzle, plane.bytes_per_element as usize)?;
            let row = m.blocks_per_row as u64 * block_bytes as u64;
            let layer = first as u64 * layout.layer_bytes + m.offset;
            if row != 0 && begin >= layer {
                let r0 = (begin - layer) / row;
                let r1 = (end.min(layer + m.size) - layer).div_ceil(row);
                y0 = (r0 * bh as u64).min(m.height as u64) as u32;
                y1 = (r1 * bh as u64).min(m.height as u64) as u32;
            }
        }
        if y0 >= y1 || (first == 0 && last == layout.slices && y0 == 0 && y1 == m.height) {
            return None;
        }
        let (slices, rows) = (last - first, y1 - y0);
        let e = key.level_extent(0);
        let texel_y0 = y0 * bd;
        let texel_rows = (rows * bd).min(e.height.saturating_sub(texel_y0));
        if texel_rows == 0 {
            return None;
        }
        let mut plan = TransferPlan { linear: false, tiles: Vec::new(), linear_bytes: 0, copies: Vec::new() };
        plan.tiles.push((
            0,
            TileRegion {
                width: m.width,
                height: rows,
                slices,
                first_slice: first,
                row_bytes: m.width * bpe as u32,
                slice_bytes: m.width * rows * bpe as u32,
                blocks_per_row: m.blocks_per_row,
                layer_bytes: layout.layer_bytes as u32,
                tail_x: 0,
                tail_y: y0,
                slice_shift: 0,
                mip_offset: m.offset as u32,
            },
        ));
        plan.copies.push(
            vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(m.width * bd)
                .buffer_image_height(rows * bd)
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: plane.aspect, mip_level: 0, base_array_layer: first, layer_count: slices })
                .image_offset(vk::Offset3D { x: 0, y: texel_y0 as i32, z: 0 })
                .image_extent(vk::Extent3D { width: e.width, height: texel_rows, depth: 1 }),
        );
        plan.linear_bytes = (m.width as u64 * rows as u64 * bpe * slices as u64).next_multiple_of(16);
        Some(plan)
    }

    fn new(key: &ImageKey, plane: &Plane, layout: &SurfaceLayout, levels: u32) -> TransferPlan {
        Self::new_layers(key, plane, layout, levels, 0, u32::MAX)
    }

    /// The plan of layers `first..first + count` (every slice of a volume).
    fn new_layers(key: &ImageKey, plane: &Plane, layout: &SurfaceLayout, levels: u32, first: u32, count: u32) -> TransferPlan {
        let bd = if plane.aspect == vk::ImageAspectFlags::COLOR { block_dim(key.format) } else { 1 };
        let bpe = plane.bytes_per_element as u64;
        let volume = key.dim == ImageDim::D3;
        let linear = plane.swizzle == crate::tiling::SWIZZLE_LINEAR;
        let levels = levels.min(layout.mips.len() as u32);
        let mut plan = TransferPlan { linear, tiles: Vec::new(), linear_bytes: 0, copies: Vec::new() };
        let shift = if layout.thick { layout.block_depth.trailing_zeros() } else { 0 };
        for level in 0..levels {
            let m = layout.mips[level as usize];
            let e = key.level_extent(level);
            let all = layout.level_slices(level);
            let (first, slices) = if volume { (0, all) } else { (first.min(all), count.min(all - first.min(all))) };
            if slices == 0 {
                continue;
            }
            if linear {
                for slice in first..first + slices {
                    plan.copies.push(
                        vk::BufferImageCopy::default()
                            .buffer_offset(layout.slice_offset(slice) + m.offset)
                            .buffer_row_length(m.pitch * bd)
                            .buffer_image_height(m.height * bd)
                            .image_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: plane.aspect,
                                mip_level: level,
                                base_array_layer: if volume { 0 } else { slice },
                                layer_count: 1,
                            })
                            .image_offset(vk::Offset3D { x: 0, y: 0, z: if volume { slice as i32 } else { 0 } })
                            .image_extent(vk::Extent3D { depth: 1, ..e }),
                    );
                }
                continue;
            }
            let offset = plan.linear_bytes;
            plan.tiles.push((
                offset,
                TileRegion {
                    width: m.width,
                    height: m.height,
                    slices,
                    first_slice: first,
                    row_bytes: m.width * bpe as u32,
                    slice_bytes: m.width * m.height * bpe as u32,
                    blocks_per_row: m.blocks_per_row,
                    layer_bytes: layout.layer_bytes as u32,
                    tail_x: m.tail_x,
                    tail_y: m.tail_y,
                    slice_shift: shift,
                    mip_offset: m.offset as u32,
                },
            ));
            plan.copies.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(offset)
                    .buffer_row_length(m.width * bd)
                    .buffer_image_height(m.height * bd)
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: plane.aspect, mip_level: level, base_array_layer: first, layer_count: if volume { 1 } else { slices } })
                    .image_extent(vk::Extent3D { depth: if volume { slices } else { 1 }, ..e }),
            );
            plan.linear_bytes = (offset + m.width as u64 * m.height as u64 * bpe * slices as u64).next_multiple_of(16);
        }
        plan
    }
}

/// Image key of a resolved color target (a volume when ATTRIB3 says 3D:
/// its slices are the depth of the image).
/// `RPS5E_FILL_INVALIDATE=0`: a pending metadata fill stays until a
/// bind takes it, whatever writes its memory meanwhile (the earlier
/// behaviour).
/// `RPS5E_FILL_INDEX=0`: a write looks at every pending metadata fill to
/// drop the ones it overlaps, instead of only the fills starting within
/// the largest fill size before it (every shader write of every draw and
/// dispatch asks; the scan of a map grown to thousands of fills was most
/// of the command thread's time in some games). A/B scope `release`.
fn fill_index_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_FILL_INDEX")) && crate::backend::ab::release_fast("RPS5E_FILL_INDEX")
}

fn fill_invalidation_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_FILL_INVALIDATE").is_ok_and(|v| v == "0"))
}

pub fn color_key(t: &crate::render::ColorTarget) -> ImageKey {
    let volume = (t.attrib3 >> 24) & 3 == 2;
    let depth = if volume { (t.attrib3 & 0x1FFF) + 1 } else { 1 };
    let mut key = ImageKey {
        format: t.format.host,
        width: t.width,
        height: t.height,
        depth,
        layers: if volume { 1 } else { t.base_layer + t.layers },
        levels: t.mip_levels.clamp(1, max_levels(t.width, t.height, depth)),
        samples: t.samples,
        dim: if volume { ImageDim::D3 } else { ImageDim::D2 },
        main: Plane { addr: t.base, bytes_per_element: t.format.bytes_per_element, swizzle: t.swizzle, pitch: t.pitch, slice_size: t.slice_size, bytes: t.size(), aspect: vk::ImageAspectFlags::COLOR },
        stencil: None,
        transform: t.format.transform(),

    };
    if t.samples == 1 {
        key.main = plane(&key, t.base, t.format.bytes_per_element, t.swizzle, vk::ImageAspectFlags::COLOR);
    }
    key
}

/// A depth plane of 16-bit guest depth held in a 32-bit float host depth
/// aspect (the D16_UNORM_S8_UINT fallback): transfers widen / narrow it.
fn widened_depth(key: &ImageKey, plane: &Plane) -> bool {
    plane.aspect == vk::ImageAspectFlags::DEPTH && plane.bytes_per_element == 2 && matches!(key.format, vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D32_SFLOAT)
}

/// Image key of a resolved depth target.
pub fn depth_key(t: &crate::render::DepthTarget) -> ImageKey {
    let aspect = crate::device::aspect_of(t.host_format);
    let main_aspect = if aspect.contains(vk::ImageAspectFlags::DEPTH) { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::STENCIL };
    let mut key = ImageKey {
        format: t.host_format,
        width: t.width,
        height: t.height,
        depth: 1,
        layers: t.base_layer + t.layers,
        levels: 1,
        samples: t.samples,
        dim: ImageDim::D2,
        main: Plane { addr: 0, bytes_per_element: 0, swizzle: 0, pitch: 0, slice_size: 0, bytes: 0, aspect: main_aspect },
        stencil: None,
        transform: Default::default(),
    };
    let stencil = (t.has_stencil && aspect.contains(vk::ImageAspectFlags::DEPTH) && t.stencil_read_base != 0)
        .then(|| plane(&key, t.stencil_read_base, 1, t.swizzle, vk::ImageAspectFlags::STENCIL));
    key.main = if main_aspect == vk::ImageAspectFlags::DEPTH {
        plane(&key, t.z_read_base, t.bytes_per_element, t.swizzle, main_aspect)
    } else {
        plane(&key, t.stencil_read_base, 1, t.swizzle, main_aspect)
    };
    key.stencil = stencil;
    key
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod watch_neighbour_tests {
    use super::*;

    /// A partial image covering the first 2 KiB of a page, watched.
    fn armed(table: &WatchTable, id: u64, begin: u64, end: u64) {
        table.insert(id, begin, end);
        table.with(|v| {
            let e = v.iter_mut().find(|e| e.id == id).unwrap();
            e.partial = true;
            e.watched = true;
        });
        let page = begin & !(PAGE_BYTES - 1);
        table.pages.add_watch(page, PAGE_BYTES, false);
    }

    #[test]
    fn neighbour_write_on_shared_page_unwatches_without_dirtying() {
        let mem = crate::testutil::test_memory();
        let (base, _) = crate::testutil::guest_mapping(mem, 1 << 20);
        let table = WatchTable { pages: Arc::new(PageGuard::new()), ..WatchTable::with_memory(mem) };
        armed(&table, 1, base, base + 0x800);
        // A CPU write to the next allocation on the same page: the page
        // must become writable (else the faulting store retries for ever),
        // but the image's bytes were not written.
        table.mark_cpu_write(base + 0x800, 0x100, false);
        let (_, writes) = table.pages.watchers(base);
        assert_eq!(writes, 0, "the shared page must lose its watch");
        table.with(|v| {
            let e = &v[0];
            assert!(e.unwatched.get(0));
            assert!(!e.dirty.get(0));
            assert!(!e.touched);
        });
    }

    /// The bucket index finds exactly the images a write hits, across
    /// bucket edges, for images over many buckets, and after removals have
    /// moved entries (each probe compared with the overlap computed by
    /// hand).
    /// A write drops exactly the pending fills it overlaps, through the
    /// size bound as by looking at every fill.
    #[test]
    fn pending_fills_drop_the_ones_a_write_overlaps() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for round in 0..200 {
            let (mut a, mut b) = (PendingFills::default(), PendingFills::default());
            for _ in 0..64 {
                let addr = (next() % 0x10_0000) & !0xF;
                let size = if next() % 8 == 0 { 0x2_0000 + next() % 0x4_0000 } else { 0x10 + next() % 0x1000 };
                let fill = MetadataFill { size, value: 0 };
                a.insert(addr, fill);
                b.insert(addr, fill);
            }
            let begin = next() % 0x12_0000;
            let end = begin + if round % 5 == 0 { 0 } else { 1 + next() % 0x8000 };
            assert_eq!(a.drop_overlapping(begin, end, true), b.drop_overlapping(begin, end, false), "round {round}");
            let ka: Vec<u64> = a.map.keys().copied().collect();
            let kb: Vec<u64> = b.map.keys().copied().collect();
            assert_eq!(ka, kb, "round {round}: write {begin:#x}..{end:#x}");
        }
    }

    #[test]
    fn bucket_index_finds_the_images_a_write_hits() {
        let table = WatchTable::new(Arc::new(PageGuard::new()));
        let base = 0x5000_0000u64;
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut live: Vec<(u64, u64, u64)> = Vec::new();
        for id in 1..=400u64 {
            let begin = base + (next() % (64 << 20)) / 0x100 * 0x100;
            let size = match id % 5 {
                0 => 0x10_0000 + next() % (6 << 20),
                1 => 0x40,
                _ => 0x100 + next() % 0x4_0000,
            };
            table.insert(id, begin, begin + size);
            live.push((id, begin, begin + size));
        }
        // Removals move the last entry into the hole.
        for k in 0..150 {
            let i = (next() as usize + k) % live.len();
            let (id, _, _) = live.swap_remove(i);
            table.remove(id);
        }
        for probe in 0..2000u64 {
            let addr = base + next() % (72 << 20);
            let size = if probe % 7 == 0 { 0x3000 + next() % (3 << 20) } else { 8 };
            for &(id, _, _) in &live {
                table.rearm(id, false);
            }
            table.mark_cpu_write(addr, size, false);
            for &(id, b, e) in &live {
                let hit = addr < e && b < addr + size;
                assert_eq!(table.state(id).1, hit, "probe {probe}: write {addr:#x}+{size:#x}, image {id} {b:#x}..{e:#x}");
            }
        }
        // GPU buffer writes go through the same index.
        for &(id, _, _) in &live {
            table.rearm(id, false);
        }
        let (addr, size) = (base + (5 << 20) + 0x123, 0x2_0000);
        let hits = live.iter().filter(|&&(_, b, e)| addr < e && b < addr + size).count();
        assert_eq!(table.invalidate_range(addr, size), hits);
        // An empty write at a page start inside an image still hits it
        // (the byte test counts the images around the address).
        let (_, b, e) = *live.iter().find(|&&(_, b, e)| e - b > 0x2000).expect("a large image");
        let at = (b + 0x1000) & !0xFFF;
        let around = live.iter().filter(|&&(_, ib, ie)| ib < at && at < ie).count();
        assert!(around >= 1 && at < e);
        assert_eq!(table.invalidate_range(at, 0), around);
    }

    /// The change log of pinned images names exactly the pinned images a
    /// write, a GPU buffer write or a mapping hit, and gives up once more
    /// changes than it holds were logged.
    #[test]
    fn pinned_change_log_names_the_pinned_images_hit() {
        let table = WatchTable::new(Arc::new(PageGuard::new()));
        let base = 0x6100_0000u64;
        for id in 1..=8u64 {
            let b = base + id * 0x10_0000;
            table.insert(id, b, b + 0x8000);
        }
        for id in [2u64, 3, 5] {
            table.set_pinned(id, true);
        }
        let image = |id: u64| base + id * 0x10_0000;
        let since = table.pinned_changes();
        table.mark_cpu_write(image(1) + 8, 4, false);
        table.mark_cpu_write(image(2) + 8, 4, false);
        table.invalidate_range(image(3), 0x100);
        table.invalidate_range(image(4), 0x100);
        table.mark_mapped(image(5), 0x1000);
        table.mark_stale(6);
        table.mark_stale(2);
        let mut ids = Vec::new();
        let end = table.pinned_changed_since(since, &mut ids).expect("logged");
        assert_eq!(end, table.pinned_changes());
        assert_eq!(ids, vec![2, 3, 5, 2]);
        let mut none = Vec::new();
        assert_eq!(table.pinned_changed_since(end, &mut none), Some(end));
        assert!(none.is_empty());
        // More changes than the ring holds: unknown.
        for _ in 0..=PINNED_RING {
            table.invalidate_range(image(3), 0x100);
        }
        assert_eq!(table.pinned_changed_since(end, &mut Vec::new()), None);
        let mut last = Vec::new();
        let now = table.pinned_changes();
        assert_eq!(table.pinned_changed_since(now - 4, &mut last), Some(now));
        assert_eq!(last, vec![3; 4]);
    }

    /// A write counted while a load read guest memory keeps the image
    /// stale after the load re-arms it (the bytes read may predate it).
    #[test]
    fn write_during_a_load_keeps_the_image_stale() {
        let table = WatchTable::new(Arc::new(PageGuard::new()));
        let base = 0x4100_0000u64;
        table.insert(1, base, base + 0x800);
        table.rearm(1, true);
        assert_eq!(table.state(1), (true, false));
        // The image is loaded: no write while reading, clean afterwards.
        let since = table.write_count(1);
        table.rearm_since(1, true, since);
        assert_eq!(table.state(1), (true, false));
        // A write lands between the read and the re-arm.
        let since = table.write_count(1);
        table.mark_cpu_write(base + 0x10, 4, false);
        table.rearm_since(1, true, since);
        assert_eq!(table.state(1), (true, true), "the write after the read must not be cleared");
        // The next load without a write in between makes it clean.
        let since = table.write_count(1);
        table.rearm_since(1, true, since);
        assert_eq!(table.state(1), (true, false));
    }

    /// A merge takes the dirty pages it reads; a page written while it runs
    /// stays dirty (and only that page) for the next merge.
    #[test]
    fn page_written_during_a_merge_stays_dirty() {
        let table = WatchTable::new(Arc::new(PageGuard::new()));
        let base = 0x4200_0000u64;
        table.insert(1, base, base + 2 * PAGE_BYTES);
        table.with(|v| {
            v[0].partial = true;
            v[0].watched = true;
        });
        table.pages.add_watch(base, 2 * PAGE_BYTES, false);
        table.mark_cpu_write(base + 0x10, 4, false);
        let (runs, since) = table.take_dirty_runs(1).expect("page 0 is dirty");
        assert_eq!(runs, vec![(base, base + PAGE_BYTES)]);
        // Written while the merge reads page 0.
        table.mark_cpu_write(base + PAGE_BYTES + 0x10, 4, false);
        table.rearm_with(1, true, true, Some(since));
        assert!(table.state(1).1, "still written");
        assert!(!table.stale_whole(1), "only a page: still mergeable");
        assert_eq!(table.dirty_runs(1), Some(vec![(base + PAGE_BYTES, base + 2 * PAGE_BYTES)]));
        // Without a write during the merge the image is clean.
        let (_, since) = table.take_dirty_runs(1).unwrap();
        table.rearm_with(1, true, true, Some(since));
        assert_eq!(table.state(1), (true, false));
    }

    /// After a neighbour's write unwatched an edge page, a CPU write to
    /// the image's bytes there (it no longer faults) is found at the next
    /// use by comparing the bytes, and the page becomes dirty.
    #[test]
    fn unwatched_edge_page_write_is_found_at_the_next_use() {
        let mem = crate::testutil::test_memory();
        let (view, alias) = crate::testutil::guest_mapping(mem, 1 << 20);
        let table = WatchTable { pages: Arc::new(PageGuard::new()), ..WatchTable::with_memory(mem) };
        armed(&table, 1, view, view + 0x800);
        table.mark_cpu_write(view + 0x800, 0x100, false);
        table.resolve_maybe(1);
        table.with(|v| assert!(!v[0].dirty.get(0), "nothing changed in the image's bytes"));
        // SAFETY: the alias maps the test view's memfd; byte 0x10 of the view.
        unsafe { std::ptr::write_volatile((alias + 0x10) as *mut u8, 0x5A) };
        table.resolve_maybe(1);
        table.with(|v| {
            let e = &v[0];
            assert!(e.dirty.get(0), "the write to the image's bytes is found");
            assert!(e.cpu_written);
        });
    }

    /// A GPU-written image sharing its page with a CPU allocation: after the
    /// neighbour's write unwatched the page, the CPU reuses the image's
    /// memory without a fault. The write-back (an eviction) must find that
    /// store and leave the memory to the CPU.
    #[test]
    fn write_after_a_neighbour_write_blocks_the_write_back() {
        let mem = crate::testutil::test_memory();
        let (view, alias) = crate::testutil::guest_mapping(mem, 1 << 20);
        let table = WatchTable { pages: Arc::new(PageGuard::new()), ..WatchTable::with_memory(mem) };
        armed(&table, 1, view + 0x500, view + 0x600);
        table.mark_cpu_write(view + 0xC00, 8, false);
        assert!(table.may_write_back(1), "the image's bytes are unchanged");
        // SAFETY: the alias maps the test view's memfd; bytes 0x500.. of the view.
        unsafe { std::ptr::write_volatile((alias + 0x500) as *mut u64, 0x0000_0008_03BA_33A8) };
        assert!(!table.may_write_back(1), "a store into the image's bytes after the page lost its watch");
        assert!(table.state(1).1, "the image is CPU-written");
    }

    /// Without the memory alias the image's bytes on a page a neighbour's
    /// write unwatched cannot be compared: the image is not written back.
    #[test]
    fn neighbour_write_without_an_alias_blocks_the_write_back() {
        let table = WatchTable::new(Arc::new(PageGuard::new()));
        let base = 0x4000_0000u64;
        armed(&table, 1, base, base + 0x800);
        assert!(table.may_write_back(1));
        table.mark_cpu_write(base + 0x800, 0x100, false);
        assert!(!table.may_write_back(1));
    }

    #[test]
    fn write_to_image_bytes_marks_the_page() {
        let table = WatchTable::new(Arc::new(PageGuard::new()));
        let base = 0x4000_0000u64;
        armed(&table, 1, base, base + 0x800);
        table.mark_cpu_write(base + 0x100, 0x10, false);
        let (_, writes) = table.pages.watchers(base);
        assert_eq!(writes, 0, "the image page's watch is removed after an image write");
        table.with(|v| {
            let e = &v[0];
            assert!(e.unwatched.get(0));
            assert!(e.dirty.get(0));
            assert!(e.touched);
        });
    }
}
