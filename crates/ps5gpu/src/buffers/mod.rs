// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest memory mirrored in device buffers: upload on use, download on CPU
//! fault, page table for buffer-device-address access.
//!
//! Buffers cover disjoint, 16 KiB-aligned guest ranges; a request that
//! overlaps several buffers merges them into one (the old contents are
//! copied over on the GPU). Because every guest page is mirrored by at most
//! one buffer, the page tracker's per-page state describes that buffer's
//! copy:
//!
//! * CPU-dirty pages are uploaded (through the backing alias and a staging
//!   ring) the next time the buffer is used; the upload write-watches them.
//! * A use that writes the range makes it GPU-dirty (read-watched); a CPU
//!   access then downloads it ([`BufferCache::download`]).
//!
//! Guest memory outside registered (memfd-backed) mappings cannot be
//! watched; such ranges are re-uploaded on every use.
//!
//! The BDA page table holds, for every 16 KiB guest page, the device
//! address of the buffer byte that mirrors it (0 when unmapped); shaders
//! translate guest pointers through it.

pub mod async_copy;
pub mod guest;
mod through;
mod written_words;

use crate::device::{Buffer, GpuDevice, MemoryUse};
use crate::memory::tracker::GuestPageTracker;
use crate::memory::GuestGpuMemory;
use crate::scheduler::{Scheduler, StagingRing};
use ash::vk;
use std::collections::BTreeMap;
use std::sync::Arc;

pub const CACHING_PAGE_BITS: u32 = 14;
pub const CACHING_PAGE: u64 = 1 << CACHING_PAGE_BITS;
/// Owner of each caching page over the 1 TiB guest space.
type BufferPageTable = crate::page_table::MultiLevelPageTable<u64, CACHING_PAGE_BITS, 40, 14>;
/// Entries of the BDA page table (1 TiB / 16 KiB).
pub const BDA_PAGE_COUNT: u64 = 1 << (40 - CACHING_PAGE_BITS);
const STAGING_BYTES: u64 = 64 << 20;
/// Merge score above which a growing range is a stream, and the room
/// (128 caching pages) a stream merge reserves ahead of it.
const STREAM_LEAP_THRESHOLD: u32 = 16;
const STREAM_LEAP_BYTES: u64 = CACHING_PAGE * 128;
const DOWNLOAD_BYTES: u64 = 16 << 20;
/// Registered bytes above which idle clean buffers are collected.
const COLLECT_THRESHOLD: u64 = 3 << 30;
/// Registered bytes above which clean buffers unused for a while (the
/// caller's idle tick) are released, least recently used first.
const IDLE_COLLECT_THRESHOLD: u64 = 512 << 20;
/// Bytes of idle GPU-written buffers one collect copies out to guest memory.
const SETTLE_PER_COLLECT: u64 = 256 << 20;

pub const BUFFER_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::VERTEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw()
        | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER.as_raw()
        | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw(),
);

/// One device buffer mirroring `[addr, addr + size)`.
pub struct GpuBuffer {
    pub addr: u64,
    pub size: u64,
    pub buf: Buffer,
    pub last_use: u64,
    /// Unique per buffer ever created (memo validation).
    pub serial: u64,
    /// Merges this buffer grew through (a range that keeps growing is a
    /// stream: the next merge reserves room ahead of it).
    stream_score: u32,
    /// The guest pages themselves imported as a buffer (made on the first
    /// download): downloads copy straight into guest memory.
    host: HostView,
    /// Tick the buffer was created in (a merge creates a new one).
    created: u64,
    /// GPU write epoch at which a collect found nothing to settle in this
    /// buffer (it is looked at again once the GPU wrote anything since).
    unsettled_at: Option<u64>,
}

/// The copies of a synchronous download, recorded and not yet waited for:
/// (guest, size) copied into guest memory directly, and (guest, size,
/// ring offset) staged in the download ring.
struct RecordedDownload {
    direct: Vec<(u64, u64)>,
    pending: Vec<(u64, u64, u64)>,
    ok: bool,
}

enum HostView {
    Unknown,
    /// The import and the backing-alias pointer it imports (the guest
    /// range's bytes when it was made).
    Imported(Buffer, u64),
    /// A slice of a guest-memory chunk import ([`guest::GuestImports`]):
    /// its buffer, the offset of the buffer's first byte in it, and the
    /// backing-alias pointer the slice stands for.
    Chunk(vk::Buffer, u64, u64),
    Unavailable,
}

/// `RPS5E_CHUNK_HOST_VIEWS=0`: every mirror whose GPU results are copied
/// into guest memory imports its own guest range. By default a mirror
/// whose range lies in one chunk the guest-memory import table already
/// imported (unranged reads, `RPS5E_BDA`) copies into that import: no
/// import (pinning pages) per mirror and none to release when the mirror
/// goes (mirrors come and go by the hundred per frame in GTA V).
fn chunk_host_views_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_CHUNK_HOST_VIEWS")) && crate::backend::ab::release_fast("chunk-host-views")
}

/// Where a mirror's bytes go in guest memory: the import buffer and the
/// offset there of the mirror's first byte (`g - addr + base` for guest
/// address `g`).
#[derive(Clone, Copy)]
struct HostDst {
    buffer: vk::Buffer,
    base: u64,
}

impl GpuBuffer {
    fn retire(self, sched: &mut Scheduler) {
        sched.retire_sized(self.buf, self.size);
        if let HostView::Imported(h, _) = self.host {
            sched.retire(h);
        }
    }
}

impl GpuBuffer {
    pub fn contains(&self, addr: u64, size: u64) -> bool {
        addr >= self.addr && addr + size <= self.addr + self.size
    }
}

/// A buffer binding: the Vulkan buffer, the offset of the requested guest
/// address in it and its device address.
#[derive(Clone, Copy, Debug)]
pub struct BufferSlice {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub size: u64,
    pub device_address: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BufferStats {
    pub buffers: usize,
    pub bytes: u64,
    pub uploads: u64,
    pub uploaded_bytes: u64,
    pub downloads: u64,
    pub downloaded_bytes: u64,
    pub merges: u64,
    pub collected: u64,
    /// Bytes uploaded from guest memory the tracker cannot watch.
    pub untracked_bytes: u64,
    /// Uses whose range lay in blocks without CPU-dirty pages.
    pub clean_skips: u64,
    /// Time spent synchronising (uploads) and downloading, in microseconds.
    pub upload_us: u64,
    pub download_us: u64,
    /// Command-processor copies / fills executed on the GPU.
    pub gpu_copies: u64,
    pub gpu_fills: u64,
    pub gpu_writes: u64,
    /// Guest ranges imported as buffers, and downloads copied straight
    /// into them.
    pub host_imports: u64,
    pub direct_downloads: u64,
    /// Downloads satisfied by waiting for a command-processor write-through.
    pub written_through_waits: u64,
    /// Descriptor reads satisfied by the write-through that landed the word
    /// (an older one than its page's newest).
    pub word_landings: u64,
    pub word_landing_us: u64,
    /// Ranges copied out to guest memory to release their mirrors
    /// ([`BufferCache::settle`]), and buffers / bytes collected.
    pub settled: u64,
    /// Mirrors whose guest-memory copies go into a chunk import.
    pub chunk_host_views: u64,
    pub settled_bytes: u64,
    pub collected_bytes: u64,
    /// Buffers released by memory-pressure collections.
    pub pressure_releases: u64,
    /// GPU-written buffers a memory-pressure collection copied out to
    /// guest memory without waiting (released once they landed).
    pub pressure_settles: u64,
    /// Read-only uses answered by the memo; uses by kind.
    pub memo_reads: u64,
    pub read_uses: u64,
    pub written_uses: u64,
    /// Read-only uses that missed the memo with a valid entry whose
    /// synchronisation point (hot pages) or dirty epoch had passed.
    pub memo_sync_misses: u64,
    pub memo_dirty_misses: u64,
    /// Read-only ranges bound from a copy of their guest bytes.
    pub streamed: u64,
    /// Synchronisations that uploaded: by requested size (<= 16 KiB,
    /// <= 64 KiB, <= 1 MiB, larger), into a buffer the open tick already
    /// used, for a written use.
    pub upload_sizes: [u64; 4],
    pub upload_used_in_tick: u64,
    pub upload_written: u64,
    /// Upload runs recorded into the tick's upload prologue, and those that
    /// could not be (pages already touched by work of the tick / buffer
    /// created in the tick).
    pub prologue_runs: u64,
    pub prologue_touched: u64,
    pub prologue_new: u64,
    /// Import-table updates pointing pages at their (current) mirror, and
    /// pointing them back at guest memory after CPU writes; dirty logs
    /// that overflowed (every resident page checked).
    pub resident_flips: u64,
    /// GPU-written runs written through because a stage reading guest
    /// memory used them.
    pub lazy_throughs: u64,
    pub resident_evictions: u64,
    pub dirty_log_lost: u64,
    /// Stages reading memory through device addresses the host cannot
    /// bound that found CPU-dirty mirrors, and the mirrors synchronised for
    /// them ([`BufferCache::prepare_unranged_reads`]).
    pub unranged_syncs: u64,
    pub unranged_synced_buffers: u64,
}

pub struct BufferCache {
    dev: Arc<GpuDevice>,
    memory: &'static GuestGpuMemory,
    pub tracker: Arc<GuestPageTracker>,
    buffers: BTreeMap<u64, GpuBuffer>,
    staging: StagingRing,
    download: StagingRing,
    bda: Option<Buffer>,
    /// Guest memory imported for read-only stages (`RPS5E_BDA=1`), made
    /// on first use.
    guest: Option<guest::GuestImports>,
    guest_tried: bool,
    /// Import guest memory at the first [`Self::bda_imports`]
    /// (`RPS5E_BDA=1`, or [`Self::set_bda`]).
    guest_wanted: bool,
    /// Write-throughs are being recorded as one batch (one barrier after).
    batch_through: bool,
    /// Ranges made current for stages reading guest memory at a GPU write
    /// epoch.
    current_memo: (u64, crate::fxhash::HashSet<(u64, u64)>),
    /// Pages current in a mirror are read from it (see
    /// [`guest::residency_enabled`]).
    residency: bool,
    stats: BufferStats,
    zero_copy: bool,
    /// Page spans whose current contents command-processor work also wrote
    /// to guest memory (through the import) at the tick: a CPU access waits
    /// for that tick only instead of downloading behind all queued work.
    written_through: through::ThroughSpans,
    /// The exact ranges GPU work wrote and the write-throughs landing them
    /// (descriptor reads next to newer writes, see [`Self::download_window`]).
    written_words: written_words::WrittenWords,
    /// Ranges whose mirror bytes the last GPU write of them set to one
    /// 32-bit value: a fill ([`Self::fill`]) no other GPU write overlapped
    /// since (start -> (end, value), see [`Self::known_fill`]).
    known_fills: BTreeMap<u64, (u64, u32)>,
    /// Tick of the last write-through recorded.
    last_through_tick: u64,
    /// GPU-written ranges to write through at the next submit (or CPU
    /// access), after the work writing them.
    pending_through: Vec<(u64, u64)>,
    /// A command-processor copy, fill or write is obtaining its
    /// destination: it is written through right after the operation, so
    /// its pages stay readable whatever `RPS5E_READ_THROUGHS` decides.
    cp_write: bool,
    /// Write-throughs recorded since the last publish point, as (start,
    /// end, tick): their read-through pages show the guest the bytes of an
    /// earlier GPU write until the copy lands ([`Self::publish_point`]).
    unpublished_through: Vec<(u64, u64, u64)>,
    /// Pages the CPU read back (downloaded on access) before: when a large
    /// GPU-written range covers them, just these pages are written through.
    readback: crate::scheduler::hazard::RangeSet,
    /// GPU-dirty page spans being copied into guest memory at a tick so
    /// that their mirror becomes clean (and collectable) once it completes
    /// ([`BufferCache::settle`]): (start, end, tick).
    settling: Vec<(u64, u64, u64)>,
    /// Registered bytes above which idle clean buffers are released.
    idle_threshold: u64,
    /// Pages read-only uses synchronised (see [`PageMemo`]).
    page_memo: PageMemos,
    /// The buffer of every caching page (key = buffer start, 0: none).
    page_index: BufferPageTable,
    /// Larger ranges by their exact extent.
    range_memo: crate::fxhash::HashMap<(u64, u64), PageMemo>,
    /// Serial of the next buffer created.
    next_serial: u64,
    /// Bumped by every use that lets the GPU write guest memory.
    gpu_write_epoch: u64,
    /// Bumped whenever a buffer is created or released: slices handed out
    /// before stay valid while it is unchanged.
    layout_epoch: u64,
    /// Guest bytes whose mirrors work of the open tick uses (see
    /// [`Touched`]).
    touched: Touched,
    /// `(addr, size, watchable)` of the range the last synchronisation of
    /// the current `obtain` looked at (None: it skipped the question).
    watchable_seen: Option<(u64, u64, bool)>,
    /// Host-visible buffers of staged downloads.
    download_pool: Arc<DownloadPool>,
    /// `(CPU-dirty epoch, layout epoch)` at which every mirror was last
    /// made current ([`Self::prepare_unranged_reads`]).
    mirrors_current: Option<(u64, u64)>,
    /// Guest memory imported for work reading through unbounded device
    /// addresses while read-only stages use mirrors
    /// ([`Self::prepare_unranged_reads`]; made on first use).
    unranged: Option<guest::GuestImports>,
    unranged_tried: bool,
}

/// Guest ranges (tracker-page aligned) whose mirrors work recorded in the
/// open tick reads or writes. An upload of CPU-dirty pages outside them
/// can run before all of the tick's work (in the scheduler's upload
/// prologue): no earlier command of the tick sees the new bytes, and the
/// use that needs them follows. Uploads into touched pages are recorded in
/// order instead (they end an open rendering scope).
#[derive(Default)]
struct Touched {
    tick: u64,
    /// (start, end), sorted, disjoint and merged (never adjacent).
    spans: Vec<(u64, u64)>,
    /// The span the last use fell in (empty: none).
    last: (u64, u64),
}

impl Touched {
    fn at(&mut self, tick: u64) {
        if self.tick != tick {
            self.tick = tick;
            self.spans.clear();
            self.last = (0, 0);
        }
    }

    fn add(&mut self, tick: u64, addr: u64, size: u64) {
        self.at(tick);
        let page = crate::memory::PAGE_BYTES;
        let (start, end) = (addr & !(page - 1), (addr + size.max(1)).div_ceil(page) * page);
        // Most uses fall in a span already touched (a constant ring's pages).
        if self.last.0 <= start && end <= self.last.1 {
            return;
        }
        // Spans starting at or before `end`; of them, the ones ending at or
        // after `start` touch [start, end] (ends grow with starts).
        let hi = self.spans.partition_point(|&(s, _)| s <= end);
        if hi > 0 {
            let (s, e) = self.spans[hi - 1];
            if s <= start && end <= e {
                self.last = (s, e);
                return;
            }
        }
        let lo = self.spans[..hi].partition_point(|&(_, e)| e < start);
        let merged = if lo < hi { (start.min(self.spans[lo].0), end.max(self.spans[hi - 1].1)) } else { (start, end) };
        self.spans.splice(lo..hi, std::iter::once(merged));
        self.last = merged;
    }

    fn overlaps(&self, tick: u64, addr: u64, size: u64) -> bool {
        let i = self.spans.partition_point(|&(s, _)| s < addr + size);
        self.tick == tick && i > 0 && self.spans[i - 1].1 > addr
    }
}

/// `RPS5E_UPLOAD_PROLOGUE=0`: every upload is recorded in order with the
/// tick's work (ending an open rendering scope).
fn prologue_enabled() -> bool {
    crate::backend::ab::new_fast() && prologue_configured()
}

/// The prologue is not switched off: touched ranges are recorded in every
/// mode (an A/B switch within a tick must not forget earlier uses).
fn prologue_configured() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_UPLOAD_PROLOGUE").is_ok_and(|v| v == "0"))
}

/// A tracked guest page a read-only use synchronised into buffer `key`
/// (`serial`): its mirror stays current while that buffer stands, no page
/// became CPU-dirty (the tracker's dirty epoch) and, for pages of blocks
/// that still hold CPU-dirty (hot) pages, no CPU synchronisation point
/// passed (hot pages are compared once per point). A later read-only use
/// that lies in such pages needs no synchronisation: constant rings hand
/// out a new range per draw inside the same pages.
#[derive(Clone, Copy, Default)]
struct PageMemo {
    key: u64,
    serial: u64,
    dirty_epoch: u64,
    /// None: every block of the use was CPU-clean after it.
    sync: Option<u64>,
}

/// Pages a memoised read-only use may span.
const MEMO_MAX_PAGES: u64 = 16;

/// Slots of the [`PageMemos`] front.
const PAGE_MEMO_FRONT: usize = 1024;

/// [`PageMemo`]s by page: a hash map (up to 64 Ki pages, its lookups miss
/// the processor caches) behind a small direct-mapped front holding
/// copies of the entries used last (uses cluster on a few ring pages).
/// A front slot naming a page always holds that page's map entry.
struct PageMemos {
    map: crate::fxhash::HashMap<u64, PageMemo>,
    front: Box<[(u64, PageMemo)]>,
}

impl Default for PageMemos {
    fn default() -> Self {
        Self { map: Default::default(), front: vec![(u64::MAX, PageMemo::default()); PAGE_MEMO_FRONT].into_boxed_slice() }
    }
}

impl PageMemos {
    fn get(&mut self, page: u64) -> Option<PageMemo> {
        let slot = &mut self.front[page as usize % PAGE_MEMO_FRONT];
        if slot.0 == page {
            return Some(slot.1);
        }
        let m = *self.map.get(&page)?;
        *slot = (page, m);
        Some(m)
    }

    fn insert(&mut self, page: u64, m: PageMemo) {
        self.map.insert(page, m);
        self.front[page as usize % PAGE_MEMO_FRONT] = (page, m);
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn clear(&mut self) {
        self.map.clear();
        self.front.fill((u64::MAX, PageMemo::default()));
    }
}

/// Guest pages whose GPU data is (being) copied out at `tick`: straight
/// into guest memory through the import (command-processor write-through,
/// direct downloads), or into a host-visible buffer (`staged`, at the
/// span's offset) that is copied into guest memory once the tick
/// completed. The pages stay GPU-dirty (protected) until then.
#[derive(Clone)]
struct ThroughSpan {
    start: u64,
    end: u64,
    tick: u64,
    staged: Option<(Arc<PooledBuffer>, u64)>,
}

/// `RPS5E_DOWNLOAD_POOL=0`: every staged download (pages without an
/// import) allocates a host-visible buffer of its own and frees it after.
fn download_pool_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_DOWNLOAD_POOL"))
}

/// Host-visible buffers of staged downloads, reused once no span or tick
/// holds them any more (sizes are powers of two from [`POOL_MIN_BYTES`]).
#[derive(Default)]
pub(crate) struct DownloadPool {
    free: std::sync::Mutex<Vec<Buffer>>,
}

/// Smallest pooled buffer, largest one, and bytes kept free at most.
const POOL_MIN_BYTES: u64 = 64 << 10;
const POOL_MAX_BYTES: u64 = 32 << 20;
const POOL_KEEP_BYTES: u64 = 64 << 20;

impl DownloadPool {
    /// A buffer of at least `bytes`: a free one of the size class, else a
    /// new one.
    pub(crate) fn take(self: &Arc<Self>, dev: &Arc<GpuDevice>, bytes: u64) -> Result<Arc<PooledBuffer>, vk::Result> {
        let pooled = download_pool_enabled() && bytes <= POOL_MAX_BYTES;
        let size = if pooled { bytes.max(POOL_MIN_BYTES).next_power_of_two() } else { bytes };
        if pooled {
            let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(i) = free.iter().position(|b| b.size == size) {
                let buf = free.swap_remove(i);
                return Ok(Arc::new(PooledBuffer { buf: Some(buf), pool: pooled.then(|| self.clone()) }));
            }
        }
        let buf = Buffer::new(dev, size, vk::BufferUsageFlags::TRANSFER_DST, MemoryUse::Download)?;
        Ok(Arc::new(PooledBuffer { buf: Some(buf), pool: pooled.then(|| self.clone()) }))
    }

    /// Buffers kept for reuse.
    #[cfg(test)]
    pub(crate) fn free_count(&self) -> usize {
        self.free.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

/// A staged-download buffer that goes back to its pool when dropped (after
/// the tick that writes it completed and its spans were resolved).
pub(crate) struct PooledBuffer {
    buf: Option<Buffer>,
    pool: Option<Arc<DownloadPool>>,
}

impl std::ops::Deref for PooledBuffer {
    type Target = Buffer;
    fn deref(&self) -> &Buffer {
        self.buf.as_ref().expect("pooled buffer present until dropped")
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        let (Some(buf), Some(pool)) = (self.buf.take(), self.pool.take()) else { return };
        let mut free = pool.free.lock().unwrap_or_else(|e| e.into_inner());
        if free.iter().map(|b| b.size).sum::<u64>() + buf.size <= POOL_KEEP_BYTES {
            free.push(buf);
        }
    }
}

impl ThroughSpan {
    /// The part `[s, e)` of the span.
    fn part(&self, s: u64, e: u64) -> ThroughSpan {
        let staged = self.staged.as_ref().map(|(b, off)| (b.clone(), off + (s - self.start)));
        ThroughSpan { start: s, end: e, tick: self.tick, staged }
    }
}

/// One step of a download that does not hold the caller while the GPU
/// works ([`BufferCache::download_async`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DownloadStep {
    /// Guest memory holds the GPU data (false: a page could not be written).
    Done(bool),
    /// The copies are submitted: wait for this tick (without the cache),
    /// then call again to finish.
    Wait(u64),
}

/// Read-back bytes remembered at most (all are forgotten past it).
const READBACK_BYTES_MAX: u64 = 64 << 20;
/// Read-back runs up to this size are remembered from any download; larger
/// ones only from CPU accesses ([`readback_any_size_enabled`]).
const READBACK_SMALL_PAGES: u64 = 16;

/// Written ranges up to this size stay written through whole with
/// `RPS5E_READ_THROUGHS` (GTA V: copying every small record out keeps the
/// CPU's reads of them from faulting; the ranges this skips are the large
/// vertex and BVH buffers its refit writes).
const READ_THROUGHS_MIN_BYTES: u64 = 16 << 10;

/// `RPS5E_READ_THROUGHS=1` (opt-in): by default every GPU-written range
/// of up to 1 MiB is copied out to guest memory after the work writing it
/// (the CPU may poll it). With it, of such a range over [`READ_THROUGHS_MIN_BYTES`], only
/// the pages the CPU read back (or faulted on) before are, as for larger
/// ranges: the other pages stay GPU-dirty and protected, and the first CPU
/// access downloads them and adds them to the read-back set. Smaller
/// ranges (job records, counters, descriptor words the CPU polls) and
/// ranges whose pages cannot be watched are always written through. A/B
/// scope release `read-throughs`. Opt-in: GTA V's GPU time drops ~10 %
/// with it, but ASTRO BOT stalls for 1-3 s several times a run (its CPU
/// reads of the skipped pages fault and wait).
pub(crate) fn read_throughs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::backend::ab::safe() && std::env::var("RPS5E_READ_THROUGHS").is_ok_and(|v| v == "1")) && crate::backend::ab::release_fast("read-throughs")
}

/// `RPS5E_READBACK_ANY_SIZE=0`: only read-back runs of up to 16 pages are
/// remembered, so a large GPU-written range the CPU copies out every frame
/// faults (and waits for the GPU) again every frame. Otherwise every run a
/// CPU access downloaded is written through after later GPU writes.
fn readback_any_size_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_READBACK_ANY_SIZE"))
}

/// `RPS5E_UNMAP_WHOLE_BUFFERS=0`: an unmap downloads the GPU-dirty pages of
/// the unmapped range only, not of the whole buffers it drops.
fn unmap_whole_buffers_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_UNMAP_WHOLE_BUFFERS").is_ok_and(|v| v == "0"))
}

/// Bytes of the aligned window whose GPU-written pages a CPU read of a
/// GPU-written descriptor word downloads (`RPS5E_DESC_WINDOW_KIB`, default
/// 1024; 0: the word's page only).
fn descriptor_window() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let w = *V.get_or_init(|| parse_descriptor_window(std::env::var("RPS5E_DESC_WINDOW_KIB").ok().as_deref()));
    if crate::backend::ab::new_fast() { w } else { 0 }
}

/// `RPS5E_KNOWN_FILLS=0`: the value of a GPU fill is not remembered, so a
/// CPU decision on the filled bytes (sampled DCC keys) downloads them. On
/// by default; A/B in one run with `RPS5E_AB_SCOPE=release
/// RPS5E_AB_ONLY=known-fills`.
fn known_fills_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_KNOWN_FILLS")) && crate::backend::ab::release_fast("known-fills")
}

/// `RPS5E_WRITTEN_WORDS=0`: a descriptor read waits for the newest
/// write-through of the word's page, not the one that landed the word (see
/// [`BufferCache::download_window`]). A/B scope `release`.
fn written_words_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_WRITTEN_WORDS")) && crate::backend::ab::release_fast("RPS5E_WRITTEN_WORDS")
}

/// `RPS5E_WORD_LANDED_ALONE=0`: a descriptor word whose last GPU write
/// landed with a direct write-through is read after that write-through
/// only while its page's write-through spans are still remembered; when
/// they were dropped (GTA V records ~3000 write-throughs per frame, the
/// list keeps 1024) the page is downloaded again, a submit and a GPU wait
/// for every such word. On, the exact write record decides alone: every
/// GPU write of a buffer range is recorded there before its pages turn
/// GPU-dirty, so a word covered by landed ranges only is in guest memory
/// once the newest of their ticks completed. A/B scope `release`
/// (`word-landed-alone`).
fn word_landed_alone_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_WORD_LANDED_ALONE")) && crate::backend::ab::release_fast("word-landed-alone")
}

/// `RPS5E_DESC_WINDOW_TICKS=0`: a descriptor read downloads the whole
/// window whatever the other pages wait for (see
/// [`BufferCache::download_window`]). A/B scope `release`.
fn window_ticks_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_DESC_WINDOW_TICKS")) && crate::backend::ab::release_fast("RPS5E_DESC_WINDOW_TICKS")
}

pub(crate) fn parse_descriptor_window(v: Option<&str>) -> u64 {
    match v.map(|v| v.trim().parse::<u64>()) {
        Some(Ok(k)) => (k << 10).next_power_of_two().min(1 << 30) * (k != 0) as u64,
        _ => 1 << 20,
    }
}

/// GPU-written ranges up to this size are written through to guest memory
/// once per submit (counters, readback slots, small results the CPU polls).
pub const WRITE_THROUGH_MAX_BYTES: u64 = 1 << 20;
/// Write-throughs remembered for the next publish point; past this, the
/// ones that already landed are dropped.
const UNPUBLISHED_THROUGH_MAX: usize = 4096;

/// Read-only ranges up to this size bound from a copy of their guest
/// bytes when those are newer than the mirror ([`BufferCache::prefers_stream`]).
pub const STREAM_MAX_BYTES: u64 = 16 << 10;

/// `RPS5E_BUFFER_STREAM=0`: every range is bound from its mirror.
fn stream_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_BUFFER_STREAM").is_ok_and(|v| v == "0"))
}

/// `RPS5E_SETTLE_SKIP=0`: every collect walks every idle GPU-written
/// buffer again, even those that had nothing to settle.
fn settle_skip_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| !std::env::var("RPS5E_SETTLE_SKIP").is_ok_and(|v| v == "0"))
}

/// `RPS5E_PRESSURE_SETTLE=0`: memory-pressure collections download
/// GPU-written buffers before releasing them (one GPU wait each) instead of
/// settling them and releasing them at a later collection.
fn pressure_settle_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_PRESSURE_SETTLE"))
}

/// `RPS5E_PAGE_CLEAN_MEMO=0`: a memoised read-only use misses whenever any
/// page became CPU-dirty since, not only its own.
fn page_clean_memo_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| !std::env::var("RPS5E_PAGE_CLEAN_MEMO").is_ok_and(|v| v == "0"))
}

/// `RPS5E_UNRANGED_SYNC=0`: work reading memory through device addresses
/// the host cannot bound reads mirrors as they are (pages the CPU wrote
/// since their last use stay old) and nothing where no mirror is (see
/// [`BufferCache::prepare_unranged_reads`]).
pub fn unranged_sync_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_UNRANGED_SYNC").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
}

/// `RPS5E_UNRANGED_IMPORTS=0`: work reading through unbounded device
/// addresses reads zeros where no mirror is (no import of guest memory).
fn unranged_imports_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_UNRANGED_IMPORTS").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
}

/// Repeated read-only uses skip the synchronisation unless
/// `RPS5E_BUFFER_MEMO=0`.
fn read_memo_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_BUFFER_MEMO"))
}

/// Downloads copy into imported guest pages unless `RPS5E_GPU_ZERO_COPY=0`.
fn zero_copy_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_GPU_ZERO_COPY").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
}

/// Transfer writes recorded so far become visible to host reads after the
/// submission completes.
fn host_read_barrier(dev: &GpuDevice, sched: &mut Scheduler) {
    let mb = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COPY)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::HOST)
        .dst_access_mask(vk::AccessFlags2::HOST_READ);
    // SAFETY: recording buffer.
    unsafe { dev.device.cmd_pipeline_barrier2(sched.cmd(), &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mb))) };
}

fn align_range(addr: u64, size: u64) -> (u64, u64) {
    let start = addr & !(CACHING_PAGE - 1);
    let end = (addr + size.max(1)).div_ceil(CACHING_PAGE) * CACHING_PAGE;
    (start, end)
}

impl BufferCache {
    pub fn new(dev: &Arc<GpuDevice>, memory: &'static GuestGpuMemory, tracker: Arc<GuestPageTracker>) -> Result<BufferCache, vk::Result> {
        Ok(BufferCache {
            dev: dev.clone(),
            memory,
            tracker,
            buffers: BTreeMap::new(),
            staging: StagingRing::new(dev, STAGING_BYTES, vk::BufferUsageFlags::TRANSFER_SRC, MemoryUse::Upload)?,
            download: StagingRing::new(dev, DOWNLOAD_BYTES, vk::BufferUsageFlags::TRANSFER_DST, MemoryUse::Download)?,
            bda: None,
            guest: None,
            guest_tried: false,
            guest_wanted: guest::bda_configured(),
            batch_through: false,
            current_memo: (u64::MAX, Default::default()),
            residency: guest::residency_enabled(),
            stats: BufferStats::default(),
            zero_copy: zero_copy_enabled(),
            written_through: through::ThroughSpans::default(),
            written_words: Default::default(),
            known_fills: BTreeMap::new(),
            last_through_tick: 0,
            pending_through: Vec::new(),
            cp_write: false,
            unpublished_through: Vec::new(),
            readback: Default::default(),
            settling: Vec::new(),
            idle_threshold: IDLE_COLLECT_THRESHOLD,
            page_memo: Default::default(),
            page_index: BufferPageTable::new(),
            range_memo: Default::default(),
            next_serial: 0,
            gpu_write_epoch: 0,
            layout_epoch: 0,
            touched: Touched::default(),
            watchable_seen: None,
            download_pool: Arc::default(),
            mirrors_current: None,
            unranged: None,
            unranged_tried: false,
        })
    }

    /// Registered bytes above which idle clean buffers are released
    /// (default [`IDLE_COLLECT_THRESHOLD`]).
    pub fn set_idle_threshold(&mut self, bytes: u64) {
        self.idle_threshold = bytes;
    }

    /// Download into imported guest pages (default: on unless
    /// `RPS5E_GPU_ZERO_COPY=0`); off, downloads go through the staging ring.
    pub fn set_zero_copy(&mut self, on: bool) {
        self.zero_copy = on;
    }

    pub fn stats(&self) -> BufferStats {
        BufferStats { buffers: self.buffers.len(), bytes: self.buffers.values().map(|b| b.size).sum(), ..self.stats }
    }

    /// Changes whenever GPU work may write guest memory (a use with
    /// `written`).
    pub fn gpu_write_epoch(&self) -> u64 {
        self.gpu_write_epoch
    }

    /// What guest bytes the GPU reads depend on between two looks at them:
    /// the CPU synchronisation point (CPU-dirty pages are compared once per
    /// point), the tracker's CPU-dirty epoch (a watched page was written),
    /// the GPU write epoch (work or a command-processor write may change
    /// guest memory) and the unmap generation. While the stamp is equal,
    /// guest words read before are still the words a new read would see.
    pub fn content_stamp(&self) -> [u64; 4] {
        [crate::memory::sync_generation(), self.tracker.cpu_dirty_epoch(), self.gpu_write_epoch, self.memory.unmap_generation()]
    }

    /// Changes whenever a buffer is created or released.
    pub fn layout_epoch(&self) -> u64 {
        self.layout_epoch
    }

    /// Key of the buffer containing the whole range, if any.
    pub fn find_containing(&self, addr: u64, size: u64) -> Option<u64> {
        // Buffers are disjoint and cover whole caching pages: the page of
        // `addr` names the only candidate.
        let k = *self.page_index.find(addr >> CACHING_PAGE_BITS)?;
        (k != 0 && self.buffers.get(&k).is_some_and(|b| b.contains(addr, size))).then_some(k)
    }

    /// The end of the run of buffers that mirror guest memory from `addr`
    /// on without a gap (the buffer holding `addr`, then each buffer that
    /// starts where the previous one ends), at most `limit` bytes past
    /// `addr`. None when no buffer holds `addr`.
    pub fn mirrored_end(&self, addr: u64, limit: u64) -> Option<u64> {
        let b = self.get(self.find_containing(addr, 4)?)?;
        let cap = addr.saturating_add(limit);
        let mut end = b.addr + b.size;
        while end < cap {
            match self.buffers.range(end..).next() {
                Some((_, n)) if n.addr == end => end = n.addr + n.size,
                _ => break,
            }
        }
        Some(end.min(cap))
    }

    /// Point the caching pages of `[addr, addr+size)` at buffer `key`
    /// (None: at no buffer).
    fn index_pages(&mut self, addr: u64, size: u64, key: Option<u64>) {
        let Some(pages) = BufferPageTable::page_range(addr, size) else { return };
        for p in pages {
            match key {
                Some(k) => {
                    if let Some(e) = self.page_index.get_or_create(p) {
                        *e = k;
                    }
                }
                None => {
                    if self.page_index.find(p).is_some() {
                        if let Some(e) = self.page_index.get_or_create(p) {
                            *e = 0;
                        }
                    }
                }
            }
        }
    }

    pub fn get(&self, key: u64) -> Option<&GpuBuffer> {
        self.buffers.get(&key)
    }

    fn overlapping(&self, start: u64, end: u64) -> Vec<u64> {
        let first = self.buffers.range(..start).next_back().filter(|(_, b)| b.addr + b.size > start).map(|(k, _)| *k);
        first.into_iter().chain(self.buffers.range(start..end).map(|(k, _)| *k)).collect()
    }

    /// Find or create the buffer that will contain `[addr, addr+size)`,
    /// merging overlapping buffers. Returns its key.
    pub fn find_or_create(&mut self, sched: &mut Scheduler, addr: u64, size: u64) -> Result<u64, vk::Result> {
        if let Some(k) = self.find_containing(addr, size) {
            return Ok(k);
        }
        let (mut start, mut end) = align_range(addr, size);
        crate::slice_shadow::hole(crate::slice_shadow::Hole::BufferMiss);
        let mut old = self.overlapping(start, end);
        let mut score = 0u32;
        let mut leap = false;
        let (request_start, request_end) = (start, end);
        for k in &old {
            let b = &self.buffers[k];
            let (expands_left, expands_right) = (b.addr < request_start, b.addr + b.size > request_end);
            start = start.min(b.addr);
            end = end.max(b.addr + b.size);
            score += b.stream_score;
            if !leap && score > STREAM_LEAP_THRESHOLD {
                leap = true;
                // Reserve room in the direction the stream grows: an old
                // buffer reaching left of the request predicts growth to the
                // right, and the other way round.
                if expands_left {
                    // Never below the request (ranges past the guest
                    // space reach here too).
                    end = (end + STREAM_LEAP_BYTES).min(crate::memory::SPACE_BYTES).max(end);
                }
                if expands_right {
                    start = start.saturating_sub(STREAM_LEAP_BYTES).max(2 * CACHING_PAGE).min(start);
                }
            }
        }
        if leap {
            // The reserved room may reach further buffers.
            old = self.overlapping(start, end);
            for k in &old {
                let b = &self.buffers[k];
                start = start.min(b.addr);
                end = end.max(b.addr + b.size);
            }
        }
        let stream_score = if leap { 0 } else { old.iter().map(|k| self.buffers[k].stream_score + 1).sum() };
        let buf = Buffer::new(&self.dev, end - start, BUFFER_USAGE, MemoryUse::Device)?;
        let cb = sched.cmd();
        if !old.is_empty() {
            self.stats.merges += 1;
            for k in &old {
                let o = self.buffers.remove(k).unwrap();
                self.layout_epoch += 1;
                let region = vk::BufferCopy { src_offset: 0, dst_offset: o.addr - start, size: o.size };
                // SAFETY: recording buffer; both buffers are alive.
                unsafe { self.dev.device.cmd_copy_buffer(cb, o.buf.buffer, buf.buffer, &[region]) };
                o.retire(sched);
            }
            sched.full_barrier();
        }
        self.next_serial += 1;
        let b = GpuBuffer { addr: start, size: end - start, buf, last_use: sched.current_tick(), host: HostView::Unknown, serial: self.next_serial, stream_score, unsettled_at: None, created: sched.current_tick() };
        if self.bda.is_some() || self.guest.is_some() || self.unranged.is_some() {
            self.write_bda_entries(sched, b.addr, b.size, b.buf.address);
        }
        self.buffers.insert(start, b);
        self.layout_epoch += 1;
        crate::slice_shadow::changed(start, end - start);
        // Pages of merged buffers are taken over.
        self.index_pages(start, end - start, Some(start));
        Ok(start)
    }

    /// Upload the CPU-dirty pages of `[addr, addr+size)` into buffer `key`;
    /// with `written` the range becomes GPU-dirty afterwards (the pages it
    /// selects are written through and may stay CPU-readable).
    fn synchronize(&mut self, sched: &mut Scheduler, key: u64, addr: u64, size: u64, written: Option<&dyn Fn(u64) -> bool>) -> bool {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::BUF_UPLOAD);
        let r = self.synchronize_timed(sched, key, addr, size, written);
        sched.time_item_end(t);
        r
    }

    fn synchronize_timed(&mut self, sched: &mut Scheduler, key: u64, addr: u64, size: u64, written: Option<&dyn Fn(u64) -> bool>) -> bool {
        // Fast path: nothing to upload in blocks without CPU-dirty pages.
        if written.is_none() && self.tracker.known_cpu_clean(addr, size) {
            self.stats.clean_skips += 1;
            return true;
        }
        let tracked = self.memory.is_watchable(addr, size);
        // The caller's read memo asks the same question of the same range.
        self.watchable_seen = Some((addr, size, tracked));
        let runs: std::cell::RefCell<Vec<(u64, u64)>> = std::cell::RefCell::new(Vec::new());
        let dev = &self.dev;
        let memory = self.memory;
        let (dst, base, len, created) = {
            let b = &self.buffers[&key];
            (b.buf.buffer, b.addr, b.size, b.created)
        };
        let tick = sched.current_tick();
        // Read-only uses of a buffer older than the tick may upload ahead
        // of the tick's work (see `Touched`).
        let early = written.is_none() && prologue_enabled();
        let fresh = created >= tick;
        let touched = &self.touched;
        let in_order = std::cell::Cell::new(false);
        // Copies from the backing alias into coherent staging may run on
        // the upload helpers (see `async_copy`).
        let helpers = async_copy::enabled() && self.staging.buffer.coherent && !sched.is_slice() && !crate::capture::active();
        let staging = &mut self.staging;
        let stats = &mut self.stats;
        let read = |page: u64| memory.backing_ptr(page, crate::memory::PAGE_BYTES).map(|p| p as *const u8);
        let read: Option<&dyn Fn(u64) -> Option<*const u8>> = if crate::memory::region::hot_pages_enabled() { Some(&read) } else { None };
        let ok = self.tracker.for_each_upload_range_through(
            addr,
            size,
            written.filter(|_| tracked),
            read,
            &mut |a, n| runs.borrow_mut().push((a, n)),
            &mut || {
                let runs = runs.borrow();
                if runs.is_empty() {
                    return true;
                }
                for &(a, n) in runs.iter() {
                    // Runs are whole tracker pages inside the request, which
                    // the 16 KiB-aligned buffer covers.
                    let (a, n) = (a.max(base), (a + n).min(base + len).saturating_sub(a.max(base)));
                    let ahead = early && !fresh && !touched.overlaps(tick, a, n);
                    if early {
                        if ahead {
                            stats.prologue_runs += 1;
                        } else if fresh {
                            stats.prologue_new += 1;
                        } else {
                            stats.prologue_touched += 1;
                        }
                    }
                    if !ahead {
                        in_order.set(true);
                    }
                    let mut done = 0u64;
                    while done < n {
                        let chunk = (n - done).min(STAGING_BYTES / 2);
                        let src = a + done;
                        // A ring full of this tick's uploads: submit them
                        // so their space can be reclaimed.
                        let off = match staging.alloc(sched, chunk, 16) {
                            Some(o) => o,
                            None => {
                                sched.flush();
                                match staging.alloc(sched, chunk, 16) {
                                    Some(o) => o,
                                    None => return false,
                                }
                            }
                        };
                        let queued = helpers && chunk >= async_copy::MIN_BYTES && {
                            match memory.backing_ptr(src, chunk) {
                                Some(p) => {
                                    // SAFETY: the alias stays mapped; the ring region
                                    // belongs to the open tick, submitted only after a drain.
                                    unsafe { async_copy::queue(p, staging.bytes(off, chunk).as_mut_ptr(), chunk as usize) };
                                    true
                                }
                                None => false,
                            }
                        };
                        if !queued && !memory.read_committed(src, staging.bytes(off, chunk)) {
                            return false;
                        }
                        staging.buffer.flush(off, chunk);
                        let copy = vk::BufferCopy { src_offset: off, dst_offset: src - base, size: chunk };
                        let cb = if ahead {
                            sched.note_prologue_copy();
                            sched.prologue_cmd()
                        } else {
                            sched.cmd()
                        };
                        // SAFETY: recording buffer; ring and destination alive until the tick completes.
                        unsafe { dev.device.cmd_copy_buffer(cb, staging.buffer.buffer, dst, &[copy]) };
                        stats.uploads += 1;
                        stats.uploaded_bytes += chunk;
                        done += chunk;
                    }
                }
                true
            },
        );
        if !runs.borrow().is_empty() {
            if crate::slice_shadow::enabled() {
                crate::slice_shadow::hole(crate::slice_shadow::Hole::BufferUpload);
                for &(a, n) in runs.borrow().iter() {
                    crate::slice_shadow::changed(a, n);
                }
            }
            let class = match size {
                0..=0x4000 => 0,
                0x4001..=0x10000 => 1,
                0x10001..=0x100000 => 2,
                _ => 3,
            };
            self.stats.upload_sizes[class] += 1;
            self.stats.upload_used_in_tick += (self.buffers.get(&key).is_some_and(|b| b.last_use == sched.current_tick())) as u64;
            self.stats.upload_written += written.is_some() as u64;
            if in_order.get() {
                sched.full_barrier();
            }
        }
        if !tracked {
            // Untracked memory cannot be watched: keep it dirty.
            self.stats.untracked_bytes += runs.borrow().iter().map(|r| r.1).sum::<u64>();
            self.tracker.mark_cpu_dirty_pages(addr, size);
        }
        ok
    }

    /// The device buffer slice mirroring `[addr, addr+size)`, up to date
    /// with guest memory. With `written` the GPU will write it: the range
    /// becomes GPU-dirty.
    pub fn obtain(&mut self, sched: &mut Scheduler, addr: u64, size: u64, written: bool) -> Result<BufferSlice, vk::Result> {
        let size = size.max(1);
        if written {
            crate::slice_shadow::hole(crate::slice_shadow::Hole::WrittenRange);
            crate::slice_shadow::changed(addr, size);
            self.stats.written_uses += 1;
        } else {
            self.stats.read_uses += 1;
        }
        if !written {
            if let Some(slice) = self.memoized_read(sched, addr, size) {
                self.touch(sched, addr, size);
                return Ok(slice);
            }
        }
        // Stamps taken before the synchronisation: a page dirtied during
        // it is looked at again.
        let dirty_epoch = self.tracker.cpu_dirty_epoch();
        let sync_point = crate::memory::sync_generation();
        self.watchable_seen = None;
        let (slice, key, serial) = self.obtain_uncached(sched, addr, size, written)?;
        // After the synchronisation: its own uploads may run ahead.
        self.touch(sched, addr, size);
        let page = crate::memory::PAGE_BYTES;
        let (first, last) = (addr / page, (addr + size - 1) / page);
        // The synchronisation looked at the range's watchability already
        // (unless it skipped a range known clean).
        let seen = self.watchable_seen.take();
        let watchable = || match seen {
            Some((a, n, w)) if (a, n) == (addr, size) => w,
            _ => self.memory.is_watchable(addr, size),
        };
        if !written && read_memo_enabled() && watchable() {
            // The buffer the slice came from is the one containing the
            // range (buffers are disjoint).
            {
                let sync = (!self.tracker.known_cpu_clean(addr, size)).then_some(sync_point);
                let m = PageMemo { key, serial, dirty_epoch, sync };
                if last - first < MEMO_MAX_PAGES {
                    if self.page_memo.len() >= 1 << 16 {
                        self.page_memo.clear();
                    }
                    for p in first..=last {
                        self.page_memo.insert(p, m);
                    }
                } else {
                    if self.range_memo.len() >= 1 << 12 {
                        self.range_memo.clear();
                    }
                    self.range_memo.insert((addr, size), m);
                }
            }
        }
        Ok(slice)
    }

    /// Work of the open tick uses the mirror of `[addr, addr+size)`.
    fn touch(&mut self, sched: &Scheduler, addr: u64, size: u64) {
        if prologue_configured() {
            self.touched.add(sched.current_tick(), addr, size);
        }
    }

    /// A read-only use repeated while nothing it depends on changed (see
    /// [`ReadMemo`]).
    fn memoized_read(&mut self, sched: &mut Scheduler, addr: u64, size: u64) -> Option<BufferSlice> {
        let page = crate::memory::PAGE_BYTES;
        let (first, last) = (addr / page, (addr + size - 1) / page);
        let dirty_epoch = self.tracker.cpu_dirty_epoch();
        let sync_point = crate::memory::sync_generation();
        let large = last - first >= MEMO_MAX_PAGES;
        let head = if large { *self.range_memo.get(&(addr, size))? } else { self.page_memo.get(first)? };
        let pages = if large { first..=first } else { first..=last };
        // A CPU write elsewhere moved the dirty epoch: the mirror of this
        // range is still current while none of its pages is CPU-dirty
        // (checked once, only when needed).
        let mut clean_checked: Option<bool> = None;
        for p in pages {
            let m = if p == first { head } else { self.page_memo.get(p)? };
            if m.key != head.key || m.serial != head.serial {
                return None;
            }
            if m.dirty_epoch != dirty_epoch {
                let clean = *clean_checked.get_or_insert_with(|| page_clean_memo_enabled() && !self.tracker.has_cpu_dirty_pages(addr, size));
                if !clean {
                    self.stats.memo_dirty_misses += 1;
                    return None;
                }
            }
            if m.sync.is_some_and(|s| s != sync_point) {
                self.stats.memo_sync_misses += 1;
                return None;
            }
        }
        let b = self.buffers.get_mut(&head.key)?;
        if b.serial != head.serial || !b.contains(addr, size) {
            return None;
        }
        b.last_use = sched.current_tick();
        self.stats.memo_reads += 1;
        let offset = addr - b.addr;
        Some(BufferSlice { buffer: b.buf.buffer, offset, size, device_address: b.buf.address + offset })
    }

    /// The slice, the key of the buffer holding it and that buffer's serial.
    fn obtain_uncached(&mut self, sched: &mut Scheduler, addr: u64, size: u64, written: bool) -> Result<(BufferSlice, u64, u64), vk::Result> {
        if written {
            self.gpu_write_epoch += 1;
        }
        let t0 = crate::backend::profile::on().then(std::time::Instant::now);
        // Stale GPU data of pages a CPU write touched must not be lost: the
        // tracker never has both, so uploading is enough.
        let key = self.find_or_create(sched, addr, size)?;
        if written {
            // GPU work may change these pages after any write-through.
            self.forget_written_through(addr, size);
            self.written_words.written(addr, size);
            self.forget_fills(addr, size);
        }
        // Written ranges the CPU may poll are written through to guest
        // memory after the work: small ranges whole, of a large range only
        // the pages the CPU read back before. Their pages stay CPU-readable.
        let page = crate::memory::PAGE_BYTES;
        // With guest memory imported, stages reading through the import
        // table read written pages from the mirror (they stay current
        // there); pages that cannot be watched never point at the mirror,
        // so GPU writes to them must reach guest memory.
        // Stages reading through the import table get larger GPU-written
        // ranges written through when they read them (`bda_make_current`);
        // with residency they read the mirror, except unwatchable ranges.
        // With `RPS5E_READ_THROUGHS` a small watchable range over 16 KiB is
        // treated as a large one: only its pages the CPU read back before.
        let small = size <= WRITE_THROUGH_MAX_BYTES && !(size > READ_THROUGHS_MIN_BYTES && read_throughs_enabled() && !self.cp_write && self.memory.is_watchable(addr, size));
        let through_whole = written && (small || (self.guest.is_some() && self.residency && !self.memory.is_watchable(addr, size)));
        let first_page = addr & !(page - 1);
        let hot: Vec<(u64, u64)> = if written && !through_whole { self.readback.clipped(first_page, addr + size - first_page).collect() } else { Vec::new() };
        let can_through = written && (through_whole || !hot.is_empty()) && (self.guest.is_some() || (self.zero_copy && self.host_view(sched, key).is_some()));
        let in_hot = |p: u64| {
            let i = hot.partition_point(|&(s, _)| s <= p);
            i > 0 && p < hot[i - 1].0 + hot[i - 1].1
        };
        let selects = |p: u64| can_through && if through_whole { p < addr + size && addr < p + page } else { in_hot(p) };
        let through: &dyn Fn(u64) -> bool = &selects;
        if !self.synchronize(sched, key, addr, size, written.then_some(through)) {
            log::warn!("ps5gpu: buffer upload of {addr:#x}+{size:#x} failed");
        }
        if self.guest.is_some() && self.residency {
            self.make_resident(sched, key, addr, size);
            // Written pages that stay in guest memory (their 16 KiB entry
            // holds CPU-dirty pages too) get the GPU results copied there.
            if written && !through_whole && !self.guest.as_ref().is_some_and(|g| g.all_resident(addr, size)) && !self.pending_through.contains(&(addr, size)) {
                self.pending_through.push((addr, size));
            }
        }
        if can_through {
            if through_whole {
                if !self.pending_through.contains(&(addr, size)) {
                    self.pending_through.push((addr, size));
                }
            } else {
                for r in hot {
                    if !self.pending_through.contains(&r) {
                        self.pending_through.push(r);
                    }
                }
            }
        }
        if let Some(t0) = t0 {
            self.stats.upload_us += t0.elapsed().as_micros() as u64;
        }
        let tick = sched.current_tick();
        let b = self.buffers.get_mut(&key).unwrap();
        b.last_use = tick;
        let offset = addr - b.addr;
        Ok((BufferSlice { buffer: b.buf.buffer, offset, size, device_address: b.buf.address + offset }, key, b.serial))
    }

    /// Prepare guest memory for work that reads through device addresses
    /// the host cannot bound (a ray query's BVH nodes, a V# built in the
    /// program): such work may read any page, so its page table
    /// ([`Self::unranged_table`]) points every mirrored page at a current
    /// mirror and every other page at guest memory itself.
    ///
    /// * CPU-dirty pages of every mirror are uploaded (a mirror holds GPU
    ///   results guest memory does not have, and must also hold the CPU's
    ///   latest bytes);
    /// * mirrored pages read the mirror, pages without one read their
    ///   import of the guest memory backing (made on first use), released
    ///   mirrors fall back to it.
    ///
    /// Skipped while no page became CPU-dirty, no mirror was created or
    /// released and no guest mapping changed since the last call. Returns
    /// the mirrors synchronised.
    pub fn prepare_unranged_reads(&mut self, sched: &mut Scheduler) -> usize {
        if !self.unranged_tried {
            self.unranged_tried = true;
            if self.guest.is_none() && unranged_imports_enabled() {
                match guest::GuestImports::new(&self.dev, self.memory, sched) {
                    Ok(g) => self.unranged = g,
                    Err(e) => log::warn!("ps5gpu: guest memory import table for unranged reads: {e:?}"),
                }
            }
        }
        let mut changed = false;
        if let Some(g) = self.unranged.as_mut() {
            let first = !g.started();
            let cleared = g.sync(sched);
            if first {
                // Mirrors made before serve the pages without an import.
                let list: Vec<(u64, u64, u64)> = self.buffers.values().map(|b| (b.addr, b.size, b.buf.address)).collect();
                for (a, n, d) in list {
                    g.mirror_entries(sched, a, n, d);
                }
                changed = true;
            }
            for (a, n) in cleared {
                // Mirrors over memory that lost its import serve it again.
                let (start, end) = align_range(a, n);
                let list: Vec<(u64, u64, u64)> =
                    self.buffers.values().filter(|b| b.addr < end && start < b.addr + b.size).map(|b| (b.addr, b.size, b.buf.address)).collect();
                for (ba, bn, d) in list {
                    g.mirror_entries(sched, ba, bn, d);
                }
                changed = true;
            }
        }
        if !changed && self.mirrors_current == Some(self.mirror_stamp()) {
            return 0;
        }
        let tracker = &self.tracker;
        let dirty: Vec<(u64, u64, u64)> = self.buffers.iter().filter(|(_, b)| tracker.has_cpu_dirty_pages(b.addr, b.size)).map(|(&k, b)| (k, b.addr, b.size)).collect();
        for &(key, addr, size) in &dirty {
            if !self.synchronize(sched, key, addr, size, None) {
                log::warn!("ps5gpu: buffer upload of {addr:#x}+{size:#x} failed");
            }
        }
        if !dirty.is_empty() {
            self.stats.unranged_syncs += 1;
            self.stats.unranged_synced_buffers += dirty.len() as u64;
        }
        // Every mirrored page with an import entry reads its mirror (now
        // current): it holds GPU results guest memory may not have.
        if let Some(g) = self.unranged.as_mut() {
            let tracker = &self.tracker;
            let mut wrote = false;
            for b in self.buffers.values() {
                let (base, device) = (b.addr, b.buf.address);
                wrote |= g.make_resident(sched, b.addr, b.addr + b.size, |page| device + (page - base), |page| !tracker.has_cpu_dirty_pages(page, CACHING_PAGE));
            }
            if wrote {
                sched.full_barrier();
            }
        }
        self.mirrors_current = Some(self.mirror_stamp());
        dirty.len()
    }

    fn mirror_stamp(&self) -> (u64, u64) {
        (self.tracker.cpu_dirty_epoch(), self.layout_epoch)
    }

    /// [`Self::prepare_unranged_reads`] has nothing to do.
    pub fn unranged_reads_ready(&self) -> bool {
        self.unranged_tried && self.mirrors_current == Some(self.mirror_stamp()) && (self.unranged.is_none() || self.memory.pending_map_events() == 0)
    }

    /// The page table of work reading through unbounded device addresses:
    /// mirrors, and guest memory where no mirror is (None: the mirror
    /// table, without imports of guest memory).
    pub fn unranged_table(&self) -> Option<vk::Buffer> {
        self.unranged.as_ref().map(|g| g.table.buffer)
    }

    /// Slices for two ranges used together: obtaining the second may merge
    /// (and retire) the buffer of the first, so both are obtained until
    /// their buffers stand.
    fn obtain_pair(&mut self, sched: &mut Scheduler, a: (u64, u64, bool), b: (u64, u64, bool)) -> Result<(BufferSlice, BufferSlice), vk::Result> {
        let mut first = self.obtain(sched, a.0, a.1, a.2)?;
        loop {
            let second = self.obtain(sched, b.0, b.1, b.2)?;
            if self.find_containing(a.0, a.1).is_some_and(|k| self.buffers[&k].buf.buffer == first.buffer) {
                return Ok((first, second));
            }
            first = self.obtain(sched, a.0, a.1, a.2)?;
        }
    }

    /// A command-processor copy of guest memory executed on the GPU, in
    /// order with the work recorded before it: the destination becomes
    /// GPU-dirty. False (nothing recorded) for overlapping ranges.
    pub fn copy(&mut self, sched: &mut Scheduler, dst: u64, src: u64, size: u64) -> Result<bool, vk::Result> {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::BUF_COPY);
        let r = self.copy_timed(sched, dst, src, size);
        sched.time_item_end(t);
        r
    }

    fn copy_timed(&mut self, sched: &mut Scheduler, dst: u64, src: u64, size: u64) -> Result<bool, vk::Result> {
        if size == 0 || (src < dst + size && dst < src + size) {
            return Ok(false);
        }
        self.cp_write = true;
        let pair = self.obtain_pair(sched, (dst, size, true), (src, size, false));
        self.cp_write = false;
        let (d, s) = pair?;
        sched.full_barrier();
        let region = vk::BufferCopy { src_offset: s.offset, dst_offset: d.offset, size };
        let cb = sched.cmd();
        // SAFETY: recording buffer; both buffers alive until the tick completes; distinct ranges.
        unsafe { self.dev.device.cmd_copy_buffer(cb, s.buffer, d.buffer, &[region]) };
        sched.full_barrier();
        self.write_through(sched, dst, size);
        self.stats.gpu_copies += 1;
        Ok(true)
    }

    /// A command-processor fill of guest memory executed on the GPU (dword
    /// aligned ranges only): the destination becomes GPU-dirty.
    pub fn fill(&mut self, sched: &mut Scheduler, dst: u64, size: u64, value: u32) -> Result<bool, vk::Result> {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::FILL);
        let r = self.fill_timed(sched, dst, size, value);
        sched.time_item_end(t);
        r
    }

    fn fill_timed(&mut self, sched: &mut Scheduler, dst: u64, size: u64, value: u32) -> Result<bool, vk::Result> {
        if size == 0 || dst % 4 != 0 || size % 4 != 0 {
            return Ok(false);
        }
        self.cp_write = true;
        let d = self.obtain(sched, dst, size, true);
        self.cp_write = false;
        let d = d?;
        sched.full_barrier();
        let cb = sched.cmd();
        // SAFETY: recording buffer; offset and size are multiples of 4.
        unsafe { self.dev.device.cmd_fill_buffer(cb, d.buffer, d.offset, size, value) };
        sched.full_barrier();
        self.write_through(sched, dst, size);
        self.stats.gpu_fills += 1;
        if known_fills_enabled() {
            if self.known_fills.len() >= 4096 {
                self.known_fills.clear();
            }
            self.known_fills.insert(dst, (dst + size, value));
        }
        Ok(true)
    }

    /// The 32-bit value every dword of `[addr, addr+size)` holds in the
    /// mirror when the last GPU write over the range was one fill covering
    /// all of it (`RPS5E_KNOWN_FILLS`): a CPU decision on those bytes (DCC
    /// keys of a sampled image) needs no download then. None when unknown.
    pub fn known_fill(&self, addr: u64, size: u64) -> Option<u32> {
        if !known_fills_enabled() {
            return None;
        }
        let (_, &(end, value)) = self.known_fills.range(..=addr).next_back()?;
        (end >= addr + size).then_some(value)
    }

    /// GPU work writes `[addr, addr+size)`: fills over it are no longer
    /// what the mirror holds.
    fn forget_fills(&mut self, addr: u64, size: u64) {
        if self.known_fills.is_empty() {
            return;
        }
        let end = addr + size;
        let first = self.known_fills.range(..addr).next_back().filter(|(_, &(e, _))| e > addr).map(|(&k, _)| k);
        let keys: Vec<u64> = first.into_iter().chain(self.known_fills.range(addr..end).map(|(&k, _)| k)).collect();
        for k in keys {
            self.known_fills.remove(&k);
        }
    }

    /// A command-processor write of `data` to guest memory executed on the
    /// GPU (dword aligned, at most 64 KiB): the destination becomes
    /// GPU-dirty.
    pub fn write(&mut self, sched: &mut Scheduler, dst: u64, data: &[u8]) -> Result<bool, vk::Result> {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::BUF_COPY);
        let r = self.write_timed(sched, dst, data);
        sched.time_item_end(t);
        r
    }

    fn write_timed(&mut self, sched: &mut Scheduler, dst: u64, data: &[u8]) -> Result<bool, vk::Result> {
        let size = data.len() as u64;
        if size == 0 || size > 65536 || dst % 4 != 0 || size % 4 != 0 {
            return Ok(false);
        }
        self.cp_write = true;
        let d = self.obtain(sched, dst, size, true);
        self.cp_write = false;
        let d = d?;
        sched.full_barrier();
        let cb = sched.cmd();
        // SAFETY: recording buffer; aligned, at most 64 KiB.
        unsafe { self.dev.device.cmd_update_buffer(cb, d.buffer, d.offset, data) };
        sched.full_barrier();
        self.write_through(sched, dst, size);
        self.stats.gpu_writes += 1;
        Ok(true)
    }

    /// Download the GPU-dirty pages of `[addr, addr+size)` into guest
    /// memory (waits for the GPU). Returns false when a page could not be
    /// written back.
    #[track_caller]
    pub fn download(&mut self, sched: &mut Scheduler, addr: u64, size: u64) -> bool {
        let mut runs: Vec<(u64, u64)> = Vec::new();
        self.tracker.for_each_download_range(addr, size, false, &mut |a, n| runs.push((a, n)));
        if runs.is_empty() {
            return true;
        }
        crate::slice_shadow::hole(crate::slice_shadow::Hole::Download);
        crate::slice_shadow::changed(addr, size);
        let _site = crate::backend::profile::DownloadSite::new(std::panic::Location::caller(), addr);
        self.remember_readback(&runs, false);
        self.flush_write_through(sched);
        let t0 = std::time::Instant::now();
        log::trace!("ps5gpu: download {addr:#x}+{size:#x} on {:?}: {runs:x?}", std::thread::current().name());
        // Runs a command-processor write already wrote through: wait for it.
        let mut through: Vec<(u64, u64)> = Vec::new();
        let mut tick = 0;
        runs.retain(|&(a, n)| match self.written_through_tick(a, n) {
            Some(t) => {
                tick = tick.max(t);
                through.push((a, n));
                false
            }
            None => true,
        });
        if !through.is_empty() {
            sched.wait(tick);
            for &(a, n) in &through {
                self.resolve_through(a, n);
                self.stats.written_through_waits += 1;
            }
        }
        let (staged, through_n, through_tick, current) = (runs.len(), through.len(), tick, sched.current_tick());
        let ok = runs.is_empty() || self.download_runs(sched, runs);
        self.stats.download_us += t0.elapsed().as_micros() as u64;
        log::trace!("ps5gpu: sync download {addr:#x}+{size:#x}: {staged} run(s) copied, {through_n} written through (tick {through_tick}, current {current}), {} us", t0.elapsed().as_micros());
        ok
    }

    /// Write through the GPU-written ranges handed out since the last call:
    /// the work writing them is recorded by now. The backend calls this
    /// before every submit; downloads call it first.
    pub fn flush_write_through(&mut self, sched: &mut Scheduler) {
        let pending = std::mem::take(&mut self.pending_through);
        self.flush_write_through_list(sched, pending);
    }

    /// Pending write-throughs over ranges the CPU read back (or faulted
    /// on) before: a CPU access waits for their next copy, so the copy
    /// should follow the writing dispatch instead of the next submit.
    pub fn has_hot_pending_through(&self) -> bool {
        self.pending_through.iter().any(|&(a, n)| self.readback.clipped(a, n).next().is_some())
    }

    /// [`Self::flush_write_through`] over only the pending ranges with a
    /// CPU-read history, right after the dispatch that wrote them; the
    /// rest wait for the submit. True when a copy was recorded.
    pub fn flush_write_through_hot(&mut self, sched: &mut Scheduler) -> bool {
        let mut hot = Vec::new();
        let mut rest = Vec::new();
        for r in std::mem::take(&mut self.pending_through) {
            if self.readback.clipped(r.0, r.1).next().is_some() {
                hot.push(r);
            } else {
                rest.push(r);
            }
        }
        self.pending_through = rest;
        self.flush_write_through_list(sched, hot)
    }

    /// The write-throughs of `ranges`: barriers, the copies and the
    /// host-read ordering (see [`Self::write_through`]). True when a copy
    /// was recorded.
    fn flush_write_through_list(&mut self, sched: &mut Scheduler, ranges: Vec<(u64, u64)>) -> bool {
        if ranges.is_empty() {
            return false;
        }
        // Copies out of the mirrors (into their imports) wait only for the
        // compute writes of their ranges (see `write_through`).
        let ranged = self.guest.is_none() && crate::scheduler::hazard::range_barriers_enabled();
        if !ranged {
            sched.full_barrier();
        }
        self.batch_through = true;
        let mut copied = false;
        for (a, n) in ranges {
            copied |= self.write_through(sched, a, n);
        }
        self.batch_through = false;
        if self.guest.is_some() {
            host_read_barrier(&self.dev, sched);
            sched.full_barrier();
        } else if ranged && copied {
            sched.host_read_barrier();
        }
        copied
    }

    /// Also write the pages holding `[addr, addr+size)` (just changed by a
    /// command-processor operation recorded before) to guest memory through
    /// the buffer's import, and remember the tick. The device buffer is
    /// current for whole pages: the operation's own bytes, and the rest
    /// synchronised by its `obtain` or GPU-written before.
    /// True when a copy was recorded.
    pub(crate) fn write_through(&mut self, sched: &mut Scheduler, addr: u64, size: u64) -> bool {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::BUF_THROUGH);
        let r = self.write_through_timed(sched, addr, size);
        sched.time_item_end(t);
        r
    }

    fn write_through_timed(&mut self, sched: &mut Scheduler, addr: u64, size: u64) -> bool {
        let page = crate::memory::PAGE_BYTES;
        let (start, end) = (addr & !(page - 1), (addr + size).div_ceil(page) * page);
        if self.guest.is_some() {
            let Some(key) = self.find_containing(start, end - start) else {
                self.tracker.end_read_through(start, end - start);
                return false;
            };
            let (src, base) = (self.buffers[&key].buf.buffer, self.buffers[&key].addr);
            let complete = self.guest.as_mut().is_some_and(|g| g.write_through(sched, src, base, start, end));
            crate::memory::history::note(crate::memory::history::Event::Through, start, end - start, sched.current_tick());
            if !self.batch_through {
                host_read_barrier(&self.dev, sched);
                // Shader reads through the imports come after the copy.
                sched.full_barrier();
            }
            if !complete {
                // Some pages have no import: guest memory keeps older data
                // there, CPU reads must download.
                self.tracker.end_read_through(start, end - start);
                return true;
            }
            self.written_through.push(ThroughSpan { start, end, tick: sched.current_tick(), staged: None });
            self.written_words.written_through(start, end, addr, size, sched.current_tick());
            self.note_landing(start, end - start, sched.current_tick());
            self.last_through_tick = sched.current_tick();
            self.note_unpublished_through(sched, start, end);
            return true;
        }
        let Some(host) = self.find_containing(start, end - start).and_then(|key| self.host_view(sched, key).map(|h| (key, h))) else {
            // Guest memory keeps older data: CPU reads must download.
            self.tracker.end_read_through(start, end - start);
            return false;
        };
        let (key, host) = host;
        let b = &self.buffers[&key];
        let region = vk::BufferCopy { src_offset: start - b.addr, dst_offset: start - b.addr + host.base, size: end - start };
        let host = host.buffer;
        if crate::scheduler::hazard::range_barriers_enabled() {
            // The copy only reads the mirror (nothing on the GPU reads the
            // import): it waits for the compute writes of the range, later
            // writers of the range wait for it.
            let cb = sched.transfer_read_cmd(&[(start, end - start)]);
            // SAFETY: recording buffer; the device buffer and its import are alive until the tick completes.
            unsafe { self.dev.device.cmd_copy_buffer(cb, b.buf.buffer, host, &[region]) };
            if !self.batch_through {
                sched.host_read_barrier();
            }
        } else {
            // SAFETY: recording buffer; the device buffer and its import are alive until the tick completes.
            unsafe { self.dev.device.cmd_copy_buffer(sched.cmd(), b.buf.buffer, host, &[region]) };
            host_read_barrier(&self.dev, sched);
        }
        crate::memory::history::note(crate::memory::history::Event::Through, start, end - start, sched.current_tick());
        self.written_through.push(ThroughSpan { start, end, tick: sched.current_tick(), staged: None });
        self.written_words.written_through(start, end, addr, size, sched.current_tick());
        self.note_landing(start, end - start, sched.current_tick());
        self.last_through_tick = sched.current_tick();
        self.note_unpublished_through(sched, start, end);
        true
    }

    /// A write-through of `[start, end)` was recorded in the current tick.
    fn note_unpublished_through(&mut self, sched: &Scheduler, start: u64, end: u64) {
        if self.unpublished_through.len() >= UNPUBLISHED_THROUGH_MAX {
            let done = sched.timeline.completed_tick();
            self.unpublished_through.retain(|&(_, _, t)| t > done);
        }
        self.unpublished_through.push((start, end, sched.current_tick()));
    }

    /// The command processor published a write the guest can observe (a
    /// label, a fence value): for the guest the work recorded before it is
    /// done, and a CPU read synchronised with it must see that work's
    /// results, as on the console. Read-through pages whose write-through
    /// has not landed by tick `completed` (still pending, or recorded in a
    /// later tick) hold the bytes of an earlier GPU write in guest memory:
    /// they stop being readable, so such a read faults and waits for the
    /// copy. Reads not ordered by a label still see the earlier bytes
    /// without a fault.
    pub fn publish_point(&mut self, completed: u64) {
        for &(a, n) in &self.pending_through {
            self.tracker.end_read_through(a, n);
        }
        for (s, e, t) in self.unpublished_through.drain(..) {
            if t > completed {
                self.tracker.end_read_through(s, e - s);
            }
        }
    }

    /// A copy of the GPU-dirty pages of the range straight into guest
    /// memory is recorded in `tick`: a CPU access after the tick completed
    /// finds the pages current without the back-end
    /// ([`crate::memory::landing_enabled`]).
    fn note_landing(&self, addr: u64, size: u64, tick: u64) {
        if crate::memory::landing_enabled() {
            self.tracker.set_landing(addr, size, tick);
        }
    }

    /// The pages of `[addr, addr+size)` are covered by completed spans:
    /// copy staged bytes into guest memory (oldest span first), clear the
    /// GPU-dirty state and drop the spans inside. False when staged bytes
    /// could not be written.
    fn resolve_through(&mut self, addr: u64, size: u64) -> bool {
        let end = addr + size;
        let mut ok = true;
        for sp in self.written_through.overlapping_oldest_first(addr, end) {
            let Some((buf, off)) = &sp.staged else { continue };
            let (s, e) = (sp.start.max(addr), sp.end.min(end));
            let o = off + (s - sp.start);
            buf.invalidate(o, e - s);
            ok &= self.memory.write_committed(s, &buf.bytes_mut()[o as usize..(o + e - s) as usize]);
        }
        self.tracker.clear_gpu_dirty_pages(addr, size);
        self.forget_written_through_inside(addr, size);
        ok
    }

    /// Download without holding the caller while the GPU works: the copies
    /// of the GPU-dirty runs of `[addr, addr+size)` are recorded (into the
    /// guest pages' import, or a host-visible buffer) and submitted, and
    /// `Wait(tick)` is returned; once the tick completed, the next call
    /// copies staged bytes into guest memory and clears the pages. GPU work
    /// recorded meanwhile that writes the pages again cancels their copies
    /// (the call records new ones). Pages stay protected until done.
    pub fn download_async(&mut self, sched: &mut Scheduler, addr: u64, size: u64) -> DownloadStep {
        self.download_async_as(sched, addr, size, true)
    }

    /// [`Self::download_async`]; `cpu`: the CPU itself reads the pages
    /// (remembered as read back at any size with
    /// `RPS5E_READBACK_ANY_SIZE`), false: on behalf of an image read-back
    /// ([`Self::download`]'s bookkeeping).
    pub fn download_async_as(&mut self, sched: &mut Scheduler, addr: u64, size: u64, cpu: bool) -> DownloadStep {
        let mut runs: Vec<(u64, u64)> = Vec::new();
        self.tracker.for_each_download_range(addr, size, false, &mut |a, n| runs.push((a, n)));
        if runs.is_empty() {
            return DownloadStep::Done(true);
        }
        crate::slice_shadow::hole(crate::slice_shadow::Hole::Download);
        crate::slice_shadow::changed(addr, size);
        let t0 = std::time::Instant::now();
        self.remember_readback(&runs, cpu);
        self.flush_write_through(sched);
        let done = sched.timeline.refresh();
        let (mut wait, mut ok) = (0u64, true);
        let mut fresh: Vec<(u64, u64)> = Vec::new();
        for (a, n) in runs {
            match self.written_through_tick(a, n) {
                Some(t) if t <= done => {
                    ok &= self.resolve_through(a, n);
                    self.stats.downloads += 1;
                    self.stats.downloaded_bytes += n;
                }
                Some(t) => wait = wait.max(t),
                None => fresh.push((a, n)),
            }
        }
        if !fresh.is_empty() {
            match self.record_download(sched, &fresh) {
                Some(t) => wait = wait.max(t),
                None => ok &= self.download_runs(sched, fresh),
            }
        }
        if wait >= sched.current_tick() {
            // Spans recorded into the open command buffer: submit them.
            sched.flush();
        }
        self.stats.download_us += t0.elapsed().as_micros() as u64;
        if wait > done {
            DownloadStep::Wait(wait)
        } else {
            DownloadStep::Done(ok)
        }
    }

    /// Record the copies of `runs` out of their device buffers (into the
    /// import, else into one host-visible buffer) as spans of the current
    /// tick, and submit. The tick, or None when nothing could be recorded.
    fn record_download(&mut self, sched: &mut Scheduler, runs: &[(u64, u64)]) -> Option<u64> {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::BUF_DOWNLOAD);
        let r = self.record_download_timed(sched, runs);
        sched.time_item_end(t);
        r
    }

    fn record_download_timed(&mut self, sched: &mut Scheduler, runs: &[(u64, u64)]) -> Option<u64> {
        let page = crate::memory::PAGE_BYTES;
        // (guest, size, buffer key, direct destination)
        let mut pieces: Vec<(u64, u64, u64, Option<HostDst>)> = Vec::new();
        let mut staged_bytes = 0u64;
        for &(a, n) in runs {
            let mut cur = a;
            while cur < a + n {
                let Some((&key, b)) = self.buffers.range(..=cur).next_back() else { break };
                if b.addr + b.size <= cur {
                    // No buffer mirrors this page: nothing newer than guest memory.
                    self.tracker.clear_gpu_dirty_pages(cur, (a + n - cur).min(page));
                    cur += page;
                    continue;
                }
                let run_end = (a + n).min(b.addr + b.size);
                let direct = self.host_view(sched, key);
                if direct.is_none() {
                    staged_bytes += run_end - cur;
                }
                pieces.push((cur, run_end - cur, key, direct));
                cur = run_end;
            }
        }
        if pieces.is_empty() {
            return None;
        }
        let staging = if staged_bytes > 0 { Some(self.download_pool.take(&self.dev, staged_bytes).ok()?) } else { None };
        let before = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
        // SAFETY: recording buffer.
        unsafe { self.dev.device.cmd_pipeline_barrier2(sched.cmd(), &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&before))) };
        let tick = sched.current_tick();
        let mut off = 0u64;
        for (piece, len, key, direct) in pieces {
            // Only the pages still GPU-dirty, landing with this copy (or,
            // staged, when the back-end resolves the span).
            let mut claimed: smallvec::SmallVec<[(u64, u64); 4]> = smallvec::SmallVec::new();
            self.tracker.claim_landing(piece, len, if direct.is_some() { tick } else { 0 }, &mut |a, n| claimed.push((a, n)));
            for (g, n) in claimed {
                let b = &self.buffers[&key];
                let (src, base) = (b.buf.buffer, b.addr);
                let (dst, dst_offset, staged) = match (direct, &staging) {
                    (Some(h), _) => (h.buffer, g - base + h.base, None),
                    (_, Some(s)) => {
                        let o = off;
                        off += n;
                        (s.buffer, o, Some((s.clone(), o)))
                    }
                    _ => continue,
                };
                let region = vk::BufferCopy { src_offset: g - base, dst_offset, size: n };
                // SAFETY: recording buffer; source, import and staging buffer stay alive until the tick completes.
                unsafe { self.dev.device.cmd_copy_buffer(sched.cmd(), src, dst, &[region]) };
                crate::memory::history::note(if direct.is_some() { crate::memory::history::Event::DownloadDirect } else { crate::memory::history::Event::DownloadStaged }, g, n, tick);
                if direct.is_some() {
                    self.stats.direct_downloads += 1;
                }
                self.written_through.push(ThroughSpan { start: g, end: g + n, tick, staged });
            }
        }
        host_read_barrier(&self.dev, sched);
        if let Some(s) = staging {
            // Back to the download pool as soon as the copies completed.
            sched.retire_local(s);
        }
        Some(sched.flush())
    }

    /// The CPU accessed the GPU-dirty pages of `[addr, addr+size)` (a
    /// fault): later GPU writes over them are written through after the
    /// work, whatever the size of the written range, so the next access
    /// finds their bytes in guest memory ([`crate::memory::landing_enabled`]).
    pub fn note_cpu_access(&mut self, addr: u64, size: u64) {
        let page = crate::memory::PAGE_BYTES;
        let first = addr & !(page - 1);
        self.remember_readback(&[(first, (addr + size.max(1)).div_ceil(page) * page - first)], true);
    }

    /// Remember read-back runs: later GPU writes over them are written
    /// through. Small runs from any download; with `cpu` (a CPU access
    /// downloads them) runs of any size.
    fn remember_readback(&mut self, runs: &[(u64, u64)], cpu: bool) {
        let page = crate::memory::PAGE_BYTES;
        let any = cpu && readback_any_size_enabled();
        for &(a, n) in runs {
            if n <= READBACK_SMALL_PAGES * page || any {
                let (s, e) = (a & !(page - 1), (a + n).div_ceil(page) * page);
                if self.readback.bytes() + (e - s) > READBACK_BYTES_MAX {
                    self.readback.clear();
                }
                self.readback.insert(s, e - s);
            }
        }
    }

    /// The guest-memory import holding `[addr, addr+size)` as a transfer
    /// source (see [`guest::GuestImports::source_slice`]), when guest
    /// memory is imported (resident guest memory, or the imports of
    /// unranged reads) and one import covers the range.
    pub fn guest_source(&mut self, addr: u64, size: u64) -> Option<(vk::Buffer, u64)> {
        self.guest.as_mut().or(self.unranged.as_mut())?.source_slice(addr, size)
    }

    /// A word at `addr` GPU work wrote is read on the CPU (a descriptor the
    /// host needs): the GPU-written pages of the aligned window around it
    /// are downloaded at once, so the other words of its table cost no
    /// further wait (`RPS5E_DESC_WINDOW_KIB`).
    ///
    /// A word GPU work wrote long ago whose page got a newer write-through
    /// (a small write next to it) is read once its own write-through
    /// completed ([`written_words`], `RPS5E_WRITTEN_WORDS=0`: the page's).
    ///
    /// When the word's page only waits for its write-through (the copy of
    /// an earlier tick), the window takes only the pages whose data is in
    /// guest memory by that tick too: a page GPU work recorded since (the
    /// dispatch just before, writing next to the table) would make the read
    /// wait for all the work recorded so far (`RPS5E_DESC_WINDOW_TICKS=0`:
    /// the whole window).
    pub fn download_window(&mut self, sched: &mut Scheduler, addr: u64) -> bool {
        let w = descriptor_window().max(crate::memory::PAGE_BYTES);
        let base = addr & !(w - 1);
        let page = crate::memory::PAGE_BYTES;
        let first = addr & !(page - 1);
        if written_words_enabled() {
            // The word itself landed with an older write-through than its
            // page's newest: wait for that one only, and leave the page.
            if let Some(word_tick) = self.written_words.landed(addr & !3, 4) {
                let page_tick = self.written_through_tick(first, page);
                if page_tick.map_or(word_landed_alone_enabled(), |p| word_tick <= p) {
                    let t0 = std::time::Instant::now();
                    sched.wait(word_tick);
                    self.stats.word_landings += 1;
                    self.stats.word_landing_us += t0.elapsed().as_micros() as u64;
                    return true;
                }
            }
        }
        if !window_ticks_enabled() {
            return self.download(sched, base, w);
        }
        let mut own = false;
        self.tracker.for_each_download_range(first, page, false, &mut |_, _| own = true);
        let Some(need) = own.then(|| self.written_through_tick(first, page)).flatten() else {
            // A copy of the word's page waits for all the work recorded so
            // far: the rest of the window comes at no extra wait.
            return self.download(sched, base, w);
        };
        let mut runs: Vec<(u64, u64)> = Vec::new();
        self.tracker.for_each_download_range(base, w, false, &mut |a, n| runs.push((a, n)));
        let mut ok = self.download(sched, first, page);
        for (a, n) in runs {
            if a + n <= first || a >= first + page {
                if self.written_through_tick(a, n).is_some_and(|t| t <= need) {
                    ok &= self.download(sched, a, n);
                }
            } else {
                // The run around the word's page: its other pages.
                for (s, e) in [(a, first), (first + page, a + n)] {
                    if s < e && self.written_through_tick(s, e - s).is_some_and(|t| t <= need) {
                        ok &= self.download(sched, s, e - s);
                    }
                }
            }
        }
        ok
    }

    /// The guest memory the cache mirrors.
    pub fn memory(&self) -> &'static GuestGpuMemory {
        self.memory
    }

    /// Diagnostics (`RPS5E_WORD_WAIT_LOG`): what the exact write record
    /// and the page's write-throughs say about the word at `addr`.
    pub fn describe_word(&self, addr: u64, current: u64) -> String {
        let page = crate::memory::PAGE_BYTES;
        let first = addr & !(page - 1);
        let ranges: Vec<String> = self.written_words.describe(first, first + page);
        format!(
            "word {:?} page-through {:?} now {current} ranges [{}]",
            self.written_words.landed(addr & !3, 4),
            self.written_through_tick(first, page),
            ranges.join(" ")
        )
    }

    /// The tick after which guest memory holds the GPU data of every page
    /// of the run, when write-throughs cover it.
    fn written_through_tick(&self, addr: u64, size: u64) -> Option<u64> {
        let page = crate::memory::PAGE_BYTES;
        let mut tick = 0;
        let mut cur = addr;
        while cur < addr + size {
            let sp = self.written_through.newest_containing(cur)?;
            tick = tick.max(sp.tick);
            cur = sp.end.min(addr + size).max(cur + page);
        }
        Some(tick)
    }

    /// GPU work changes the pages of the range: their write-throughs no
    /// longer describe them.
    fn forget_written_through(&mut self, addr: u64, size: u64) {
        let page = crate::memory::PAGE_BYTES;
        let (start, end) = (addr & !(page - 1), (addr + size).div_ceil(page) * page);
        if self.settling.iter().any(|&(s, e, _)| s < end && start < e) {
            let spans = std::mem::take(&mut self.settling);
            for (s, e, t) in spans {
                if e <= start || s >= end {
                    self.settling.push((s, e, t));
                    continue;
                }
                if s < start {
                    self.settling.push((s, start, t));
                }
                if e > end {
                    self.settling.push((end, e, t));
                }
            }
        }
        // Keep the parts of overlapping spans outside the range.
        if self.written_through.overlaps(start, end) {
            self.written_through.cut(start, end);
        }
    }

    /// The pages of a run were downloaded: drop the spans inside it.
    fn forget_written_through_inside(&mut self, addr: u64, size: u64) {
        self.written_through.remove_inside(addr, addr + size);
    }

    /// The import of buffer `key`'s guest pages (made on first use): the
    /// backing alias of the range, when it is one contiguous memfd range,
    /// imported with VK_EXT_external_memory_host. GPU copies into it land
    /// in guest memory without a staging copy.
    ///
    /// None while a frame capture records: GPU copies into guest memory
    /// would change pages behind the capture's back (it learns of GPU
    /// results the back-end writes on the CPU, not of copies the GPU makes
    /// into the import), and the next read of such a page would record the
    /// GPU's bytes as a CPU write. The replay then wrote this run's GPU
    /// results over its own, which differ wherever GPU work orders atomics
    /// differently (GTA V's BVH build: a parent walk looped forever).
    /// Downloads are staged and nothing is written through meanwhile.
    fn host_view(&mut self, sched: &mut Scheduler, key: u64) -> Option<HostDst> {
        if crate::capture::recording(self.memory) {
            return None;
        }
        let b = self.buffers.get_mut(&key)?;
        if let HostView::Chunk(_, _, ptr) = b.host {
            // The range may map other backing bytes by now: look again.
            if self.memory.backing_ptr(b.addr, b.size).map(|p| p as u64) != Some(ptr) {
                b.host = HostView::Unknown;
            }
        }
        if let HostView::Imported(_, ptr) = b.host {
            // The guest range may map other backing bytes since the import
            // was made (a view replaced in place): copies into the import
            // would land in the old bytes, which the guest may use
            // elsewhere by now.
            let now = self.memory.backing_ptr(b.addr, b.size).map(|p| p as u64);
            if now != Some(ptr) {
                let alias = self.memory.backing_alias().0;
                crate::memory::note_stale_import(crate::memory::StaleImport { addr: b.addr, size: b.size, imported: ptr - alias, current: now.map(|p| p - alias) });
                if let HostView::Imported(h, _) = std::mem::replace(&mut b.host, HostView::Unknown) {
                    sched.retire(h);
                }
            }
        }
        if let HostView::Unknown = b.host {
            b.host = HostView::Unavailable;
            let (addr, size) = (b.addr, b.size);
            let ptr = self.memory.backing_ptr(addr, size);
            let chunk = match ptr {
                Some(p) if self.zero_copy && chunk_host_views_enabled() => self.unranged.as_mut().or(self.guest.as_mut()).and_then(|g| g.imported_slice(addr, size)).map(|(buffer, base)| HostView::Chunk(buffer, base, p as u64)),
                _ => None,
            };
            let view = match (chunk, ptr) {
                (Some(c), _) => {
                    self.stats.chunk_host_views += 1;
                    c
                }
                // SAFETY: the backing alias stays mapped for the whole
                // process; the import is retired with the buffer.
                (None, Some(ptr)) if self.zero_copy => match unsafe { Buffer::import_host(&self.dev, ptr, size, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC) } {
                    Ok(h) if h.coherent => {
                        self.stats.host_imports += 1;
                        HostView::Imported(h, ptr as u64)
                    }
                    Ok(h) => {
                        drop(h);
                        HostView::Unavailable
                    }
                    Err(e) => {
                        crate::log_once(&format!("ps5gpu: guest memory import failed ({e:?}); downloads are staged"));
                        HostView::Unavailable
                    }
                },
                _ => HostView::Unavailable,
            };
            self.buffers.get_mut(&key)?.host = view;
        }
        let b = self.buffers.get(&key)?;
        match &b.host {
            HostView::Imported(h, _) => Some(HostDst { buffer: h.buffer, base: 0 }),
            &HostView::Chunk(buffer, base, _) => Some(HostDst { buffer, base }),
            _ => None,
        }
    }

    fn download_runs(&mut self, sched: &mut Scheduler, runs: Vec<(u64, u64)>) -> bool {
        let t = sched.time_item_begin(crate::scheduler::gpu_timer::label::BUF_DOWNLOAD);
        let r = self.download_runs_timed(sched, runs);
        sched.time_item_end(t);
        r
    }

    fn download_runs_timed(&mut self, sched: &mut Scheduler, runs: Vec<(u64, u64)>) -> bool {
        match self.record_download_runs(sched, runs) {
            Some(rec) => self.finish_download_runs(sched, rec),
            None => false,
        }
    }

    /// Record the copies of a synchronous download (see
    /// [`Self::download_runs`]); None when the download ring had no room.
    ///
    /// The pages stay GPU-dirty until [`Self::finish_download_runs`], and
    /// the fault path must not make them clean meanwhile on an older
    /// landing copy: a CPU write after that would be overwritten by these
    /// copies (they copy the mirror as it is now). Pages copied straight
    /// into guest memory land with the copies' tick; staged pages are
    /// written by the CPU after the wait, so they get no landing at all
    /// (a CPU access waits for the back-end, which this download holds).
    fn record_download_runs(&mut self, sched: &mut Scheduler, runs: Vec<(u64, u64)>) -> Option<RecordedDownload> {
        let mut rec = RecordedDownload { direct: Vec::new(), pending: Vec::new(), ok: true };
        let before = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
        // SAFETY: recording buffer.
        unsafe { self.dev.device.cmd_pipeline_barrier2(sched.cmd(), &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&before))) };
        for (a, n) in runs {
            let mut cur = a;
            while cur < a + n {
                let Some((&key, b)) = self.buffers.range(..=cur).next_back() else { break };
                if b.addr + b.size <= cur {
                    // No buffer mirrors this page: nothing newer than guest memory.
                    self.tracker.clear_gpu_dirty_pages(cur, (a + n - cur).min(crate::memory::PAGE_BYTES));
                    cur += crate::memory::PAGE_BYTES;
                    continue;
                }
                let run_end = (a + n).min(b.addr + b.size);
                let (src, base) = (b.buf.buffer, b.addr);
                let host = self.host_view(sched, key);
                let tick = sched.current_tick();
                // Only the pages still GPU-dirty, landing with this copy
                // (or, staged, after the wait).
                let mut claimed: smallvec::SmallVec<[(u64, u64); 4]> = smallvec::SmallVec::new();
                self.tracker.claim_landing(cur, run_end - cur, if host.is_some() { tick } else { 0 }, &mut |a, n| claimed.push((a, n)));
                cur = run_end;
                for (g, len) in claimed {
                    if let Some(host) = host {
                        let region = vk::BufferCopy { src_offset: g - base, dst_offset: g - base + host.base, size: len };
                        // SAFETY: recording buffer; the device buffer and the import are alive until the tick completes.
                        unsafe { self.dev.device.cmd_copy_buffer(sched.cmd(), src, host.buffer, &[region]) };
                        crate::memory::history::note(crate::memory::history::Event::DownloadDirect, g, len, tick);
                        rec.direct.push((g, len));
                        continue;
                    }
                    let mut done = 0;
                    while done < len {
                        let chunk = (len - done).min(DOWNLOAD_BYTES / 2);
                        let off = match self.download.alloc(sched, chunk, 16) {
                            Some(o) => o,
                            None => {
                                rec.ok &= self.flush_download_batch(sched, &mut rec.pending);
                                self.download.alloc(sched, chunk, 16)?
                            }
                        };
                        let region = vk::BufferCopy { src_offset: g + done - base, dst_offset: off, size: chunk };
                        // SAFETY: recording buffer; both buffers alive.
                        unsafe { self.dev.device.cmd_copy_buffer(sched.cmd(), src, self.download.buffer.buffer, &[region]) };
                        rec.pending.push((g + done, chunk, off));
                        done += chunk;
                    }
                }
            }
        }
        Some(rec)
    }

    /// Wait for the copies [`Self::record_download_runs`] recorded, write
    /// the staged bytes and make the pages clean.
    fn finish_download_runs(&mut self, sched: &mut Scheduler, mut rec: RecordedDownload) -> bool {
        let mut ok = rec.ok;
        if !rec.direct.is_empty() && rec.pending.is_empty() {
            host_read_barrier(&self.dev, sched);
            sched.flush_and_wait();
        }
        ok &= self.flush_download_batch(sched, &mut rec.pending);
        for (g, n) in rec.direct {
            self.stats.downloads += 1;
            self.stats.downloaded_bytes += n;
            self.stats.direct_downloads += 1;
            self.tracker.clear_gpu_dirty_pages(g, n);
        }
        ok
    }

    /// Wait for the staged download copies in `pending`, write their bytes
    /// into guest memory and make the pages clean.
    fn flush_download_batch(&mut self, sched: &mut Scheduler, pending: &mut Vec<(u64, u64, u64)>) -> bool {
        if pending.is_empty() {
            return true;
        }
        host_read_barrier(&self.dev, sched);
        sched.flush_and_wait();
        let mut ok = true;
        for &(g, n, off) in pending.iter() {
            self.download.buffer.invalidate(off, n);
            let bytes = self.download.bytes(off, n);
            ok &= self.memory.write_committed(g, bytes);
            self.stats.downloads += 1;
            self.stats.downloaded_bytes += n;
        }
        for &(g, n, _) in pending.iter() {
            self.tracker.clear_gpu_dirty_pages(g, n);
        }
        pending.clear();
        ok
    }

    /// Before a frame capture: every buffer's GPU
    /// results reach guest memory and the buffer goes. Returns the
    /// buffers dropped.
    pub fn capture_reset(&mut self, sched: &mut Scheduler) -> usize {
        let ranges: Vec<(u64, u64)> = self.buffers.values().map(|b| (b.addr, b.size)).collect();
        for &(addr, size) in &ranges {
            self.unmap(sched, addr, size);
        }
        ranges.len()
    }

    /// Download every GPU-dirty page of the buffers overlapping
    /// `[addr, addr+size)` and forget them (the guest unmapped the range).
    pub fn unmap(&mut self, sched: &mut Scheduler, addr: u64, size: u64) {
        // The buffers overlapping the range go whole (they are 16 KiB
        // aligned and merged across neighbouring allocations), so the GPU
        // data they hold outside the range is downloaded too: the mirror is
        // its only copy (`RPS5E_UNMAP_WHOLE_BUFFERS=0`: the range only).
        let (lo, hi) = if unmap_whole_buffers_enabled() {
            self.overlapping(addr, addr + size).iter().filter_map(|k| self.buffers.get(k)).fold((addr, addr + size), |(lo, hi), b| (lo.min(b.addr), hi.max(b.addr + b.size)))
        } else {
            (addr, addr + size)
        };
        self.download(sched, lo, hi - lo);
        self.forget_written_through(addr, size);
        self.written_words.forget(addr, size);
        self.forget_fills(addr, size);
        self.settling.retain(|&(s, e, _)| e <= addr || s >= addr + size);
        self.pending_through.retain(|&(a, n)| a + n <= addr || a >= addr + size);
        self.readback.remove(addr, size);
        for k in self.overlapping(addr, addr + size) {
            let b = self.buffers.remove(&k).unwrap();
            self.layout_epoch += 1;
            self.index_pages(b.addr, b.size, None);
            self.tracker.untrack_memory(b.addr, b.size);
            if self.bda.is_some() || self.guest.is_some() || self.unranged.is_some() {
                self.write_bda_entries(sched, b.addr, b.size, 0);
            }
            b.retire(sched);
        }
    }

    /// Copy the GPU-dirty pages of `[addr, addr+size)` into guest memory
    /// (through the import of their buffer) after the work recorded so
    /// far, without waiting: once the tick completes, [`Self::collect`]
    /// marks them clean, so the mirror no longer holds the only copy and
    /// can be released when idle. Until then the pages stay protected and a
    /// CPU access waits for that tick only. GPU work that writes the pages
    /// again before cancels their settling. Returns the bytes recorded
    /// (0 without an import: the mirror then stays GPU-dirty).
    pub fn settle(&mut self, sched: &mut Scheduler, addr: u64, size: u64) -> u64 {
        let mut runs: Vec<(u64, u64)> = Vec::new();
        self.tracker.for_each_download_range(addr, size, false, &mut |a, n| runs.push((a, n)));
        if runs.is_empty() {
            return 0;
        }
        let page = crate::memory::PAGE_BYTES;
        let mut pieces: Vec<(u64, u64, u64)> = Vec::new();
        for (a, n) in runs {
            let mut cur = a;
            while cur < a + n {
                let Some((&key, b)) = self.buffers.range(..=cur).next_back() else { break };
                if b.addr + b.size <= cur {
                    // No buffer mirrors this page: nothing newer than guest memory.
                    self.tracker.clear_gpu_dirty_pages(cur, (a + n - cur).min(page));
                    cur += page;
                    continue;
                }
                let run_end = (a + n).min(b.addr + b.size);
                pieces.push((cur, run_end - cur, key));
                cur = run_end;
            }
        }
        let mut recorded = 0;
        let mut barrier = false;
        let tick = sched.current_tick();
        for (piece, len, key) in pieces {
            let Some(host) = self.host_view(sched, key) else { continue };
            // Only the pages still GPU-dirty, landing with this copy.
            let mut claimed: smallvec::SmallVec<[(u64, u64); 4]> = smallvec::SmallVec::new();
            self.tracker.claim_landing(piece, len, tick, &mut |a, n| claimed.push((a, n)));
            for (g, n) in claimed {
                if !barrier {
                    let before = vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
                    // SAFETY: recording buffer.
                    unsafe { self.dev.device.cmd_pipeline_barrier2(sched.cmd(), &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&before))) };
                    barrier = true;
                }
                let b = &self.buffers[&key];
                let region = vk::BufferCopy { src_offset: g - b.addr, dst_offset: g - b.addr + host.base, size: n };
                // SAFETY: recording buffer; the device buffer and its import are alive until the tick completes.
                unsafe { self.dev.device.cmd_copy_buffer(sched.cmd(), b.buf.buffer, host.buffer, &[region]) };
                crate::memory::history::note(crate::memory::history::Event::Settle, g, n, tick);
                self.written_through.push(ThroughSpan { start: g, end: g + n, tick, staged: None });
                self.settling.push((g, g + n, tick));
                recorded += n;
            }
        }
        if barrier {
            host_read_barrier(&self.dev, sched);
            self.stats.settled += 1;
            self.stats.settled_bytes += recorded;
        }
        recorded
    }

    /// Settling spans whose copies completed: guest memory holds their
    /// bytes, the pages become clean.
    fn finish_settled(&mut self, done: u64) {
        if !self.settling.iter().any(|&(_, _, t)| t <= done) {
            return;
        }
        let spans = std::mem::take(&mut self.settling);
        for (s, e, t) in spans {
            if t > done {
                self.settling.push((s, e, t));
                continue;
            }
            self.tracker.clear_gpu_dirty_pages(s, e - s);
            self.forget_written_through_inside(s, e - s);
        }
    }

    /// Finish completed settles, then release idle clean buffers, least
    /// recently used first: above [`COLLECT_THRESHOLD`] any buffer the GPU
    /// is done with (down to half of it), above
    /// the idle threshold ([`IDLE_COLLECT_THRESHOLD`]) those last used before `idle_before`.
    /// Idle buffers holding GPU-written pages are settled first
    /// ([`Self::settle`]) and released once guest memory holds their bytes.
    pub fn collect(&mut self, sched: &mut Scheduler, idle_before: u64) {
        let done = sched.timeline.refresh();
        self.finish_settled(done);
        let total: u64 = self.buffers.values().map(|b| b.size).sum();
        if total <= self.idle_threshold {
            return;
        }
        // A buffer last used in a completed tick is no longer in use.
        // Below the urgent mark only aged buffers can go: gather just those
        // (the cache holds thousands of recently used mirrors).
        let urgent_possible = total > COLLECT_THRESHOLD;
        let mut idle: Vec<(u64, u64)> = self.buffers.iter().filter(|(_, b)| b.last_use <= done && (urgent_possible || b.last_use < idle_before)).map(|(k, b)| (b.last_use, *k)).collect();
        idle.sort();
        let mut freed = 0;
        let mut settled = 0;
        for (last_use, k) in idle {
            let remaining = total - freed;
            let urgent = total > COLLECT_THRESHOLD && remaining > COLLECT_THRESHOLD / 2;
            let aged = last_use < idle_before && remaining > self.idle_threshold;
            if !urgent && !aged {
                break;
            }
            let (a, n) = (self.buffers[&k].addr, self.buffers[&k].size);
            if self.tracker.has_gpu_dirty_pages(a, n) {
                // GPU-written pages nobody read back: copy them out so the
                // idle mirror can go at a later collect.
                // A buffer that had nothing to settle (its pages cannot be
                // copied out, or are being written through) is not walked
                // again until the GPU wrote something since.
                let epoch = self.gpu_write_epoch;
                if aged && settled < SETTLE_PER_COLLECT && self.buffers[&k].unsettled_at != Some(epoch) && !self.settling.iter().any(|&(s, e, _)| s < a + n && a < e) {
                    // A buffer known to have no host import cannot copy
                    // anything out: its settle would walk the pages and
                    // record nothing (the pages it finds all lie in it).
                    let done = if matches!(self.buffers[&k].host, HostView::Unavailable) { 0 } else { self.settle(sched, a, n) };
                    settled += done;
                    if done == 0 && settle_skip_enabled() {
                        if let Some(b) = self.buffers.get_mut(&k) {
                            b.unsettled_at = Some(epoch);
                        }
                    }
                }
                continue;
            }
            freed += self.release(sched, k);
        }
    }

    /// Drop buffer `k` (its pages are no longer mirrored). Returns its size.
    fn release(&mut self, sched: &mut Scheduler, k: u64) -> u64 {
        let Some(b) = self.buffers.remove(&k) else { return 0 };
        self.layout_epoch += 1;
        self.index_pages(b.addr, b.size, None);
        self.tracker.untrack_memory(b.addr, b.size);
        if self.bda.is_some() || self.guest.is_some() || self.unranged.is_some() {
            self.write_bda_entries(sched, b.addr, b.size, 0);
        }
        let size = b.size;
        self.stats.collected += 1;
        self.stats.collected_bytes += size;
        b.retire(sched);
        size
    }

    /// Memory-pressure collection ([`crate::gc`]): with the process using
    /// `usage` bytes of device memory (None: the bytes of the buffers),
    /// above the trigger mark the least recently used buffers idle long
    /// enough are released; above the critical mark GPU-written ones are
    /// settled first (copied out after the queued work, released by a later
    /// collection; `RPS5E_PRESSURE_SETTLE=0`: downloaded, waiting for the
    /// GPU once per buffer), below it they stay.
    /// `frame_cutoff(frames)`: the scheduler tick before which entries have
    /// been idle for that many presented frames (see [`crate::gc::idle_frames`]).
    pub fn collect_pressure(&mut self, sched: &mut Scheduler, usage: Option<u64>, t: &crate::gc::Thresholds, ticks: &crate::gc::CollectionTicks, frame_cutoff: impl Fn(usize) -> u64) {
        let usage = usage.unwrap_or_else(|| self.buffers.values().map(|b| b.size).sum());
        if usage < t.trigger {
            return;
        }
        let (aggressive, age, limit) = crate::gc::buffer_run(usage, t);
        let done = sched.timeline.refresh();
        let settle = pressure_settle_enabled();
        if settle {
            // Settled earlier: clean now, released below.
            self.finish_settled(done);
        }
        let cutoff = ticks.cutoff(age).min(frame_cutoff(crate::gc::idle_frames(true, aggressive)));
        let mut idle: Vec<(u64, u64)> = self.buffers.iter().filter(|(_, b)| b.last_use < cutoff && b.last_use <= done).map(|(k, b)| (b.last_use, *k)).collect();
        idle.sort_unstable();
        let mut dirty = Vec::new();
        let mut taken = 0;
        for (_, k) in idle {
            if taken == limit {
                break;
            }
            let (a, n) = (self.buffers[&k].addr, self.buffers[&k].size);
            if self.tracker.has_gpu_dirty_pages(a, n) {
                if !aggressive {
                    continue;
                }
                dirty.push((k, a, n));
            } else {
                self.release(sched, k);
                self.stats.pressure_releases += 1;
            }
            taken += 1;
        }
        for (k, a, n) in dirty {
            // GPU-written pages reach guest memory before the mirror goes:
            // copied out after the queued work without waiting for it, the
            // mirror is released by a later collection once they landed
            // (a CPU access meanwhile finds them landed or waits for that
            // copy only). Without an import, downloaded here.
            if settle {
                if self.settling.iter().any(|&(s, e, _)| s < a + n && a < e) {
                    continue;
                }
                if self.settle(sched, a, n) > 0 {
                    self.stats.pressure_settles += 1;
                    continue;
                }
            }
            if self.download(sched, a, n) && !self.tracker.has_gpu_dirty_pages(a, n) && self.buffers.get(&k).is_some_and(|b| b.addr == a && b.size == n) {
                self.release(sched, k);
                self.stats.pressure_releases += 1;
            }
        }
    }


    fn write_bda_entries(&mut self, sched: &mut Scheduler, addr: u64, size: u64, device_address: u64) {
        if let Some(g) = self.guest.as_mut() {
            if g.mirror_entries(sched, addr, size, device_address) {
                sched.full_barrier();
            }
        }
        if let Some(g) = self.unranged.as_mut().filter(|g| g.started()) {
            if g.mirror_entries(sched, addr, size, device_address) {
                sched.full_barrier();
            }
        }
        let Some(table) = &self.bda else { return };
        let first = addr >> CACHING_PAGE_BITS;
        // The table covers the low 1 TiB; buffers above it (guest stacks
        // and TLS high in the address space) have no entries. Writing them
        // landed past the table and faulted the GPU.
        if first >= BDA_PAGE_COUNT {
            return;
        }
        let count = (size >> CACHING_PAGE_BITS).min(BDA_PAGE_COUNT - first);
        let entries: Vec<u64> = (0..count).map(|i| if device_address == 0 { 0 } else { device_address + (i << CACHING_PAGE_BITS) }).collect();
        let bytes: Vec<u8> = entries.iter().flat_map(|e| e.to_le_bytes()).collect();
        let table_buf = table.buffer;
        let cb = sched.cmd();
        let mut done = 0usize;
        while done < bytes.len() {
            let chunk = (bytes.len() - done).min(65536);
            // SAFETY: recording buffer; vkCmdUpdateBuffer takes at most 64 KiB.
            unsafe {
                self.dev.device.cmd_update_buffer(cb, table_buf, first * 8 + done as u64, &bytes[done..done + chunk]);
            }
            done += chunk;
        }
        sched.full_barrier();
    }

    /// The BDA page table (created and filled on first use). Returns the
    /// table buffer and its device address.
    pub fn bda_page_table(&mut self, sched: &mut Scheduler) -> Result<(vk::Buffer, u64), vk::Result> {
        if self.bda.is_none() {
            let table = Buffer::new(&self.dev, BDA_PAGE_COUNT * 8, BUFFER_USAGE, MemoryUse::Device)?;
            let cb = sched.cmd();
            // SAFETY: recording buffer.
            unsafe { self.dev.device.cmd_fill_buffer(cb, table.buffer, 0, vk::WHOLE_SIZE, 0) };
            sched.full_barrier();
            self.bda = Some(table);
            let list: Vec<(u64, u64, u64)> = self.buffers.values().map(|b| (b.addr, b.size, b.buf.address)).collect();
            for (a, n, d) in list {
                self.write_bda_entries(sched, a, n, d);
            }
        }
        let t = self.bda.as_ref().unwrap();
        Ok((t.buffer, t.address))
    }

    /// The import table of guest memory (`RPS5E_BDA=1`; made on first
    /// use), current with the guest mappings: read-only stages translate
    /// guest addresses through it. GPU writes recorded before are written
    /// through to guest memory first, so the stages read the newest bytes.
    pub fn bda_imports(&mut self, sched: &mut Scheduler) -> Option<vk::Buffer> {
        if !self.guest_tried {
            self.guest_tried = true;
            if self.guest_wanted {
                match guest::GuestImports::new(&self.dev, self.memory, sched) {
                    Ok(g) => self.guest = g,
                    Err(e) => log::warn!("ps5gpu: guest memory import table: {e:?}"),
                }
                if self.guest.is_some() {
                    self.tracker.enable_dirty_log();
                    // Mirrors made before point the table at themselves
                    // where guest memory has no import.
                    let list: Vec<(u64, u64, u64)> = self.buffers.values().map(|b| (b.addr, b.size, b.buf.address)).collect();
                    for (a, n, d) in list {
                        if let Some(g) = self.guest.as_mut() {
                            g.mirror_entries(sched, a, n, d);
                        }
                    }
                }
            }
        }
        self.evict_dirty(sched);
        let g = self.guest.as_mut()?;
        let cleared = g.sync(sched);
        for (a, n) in cleared {
            // Mirrors over memory that lost its import serve it again.
            let (start, end) = align_range(a, n);
            let list: Vec<(u64, u64, u64)> =
                self.buffers.values().filter(|b| b.addr < end && start < b.addr + b.size).map(|b| (b.addr, b.size, b.buf.address)).collect();
            for (ba, bn, d) in list {
                if let Some(g) = self.guest.as_mut() {
                    g.mirror_entries(sched, ba, bn, d);
                }
            }
            sched.full_barrier();
        }
        if !self.pending_through.is_empty() {
            self.flush_write_through(sched);
            sched.full_barrier();
        }
        self.guest.as_ref().map(|g| g.table.buffer)
    }

    /// Import guest memory for read-only stages (before the first
    /// [`Self::bda_imports`]; the default is `RPS5E_BDA=1`).
    pub fn set_bda(&mut self, on: bool) {
        self.guest_wanted = on;
    }

    /// The pages of `[addr, addr+size)` in buffer `key` that hold current
    /// bytes (no CPU write since their upload) are read from the mirror by
    /// stages that read guest memory: their import-table entries point at
    /// it until a CPU write (see [`Self::bda_imports`]).
    fn make_resident(&mut self, sched: &mut Scheduler, key: u64, addr: u64, size: u64) {
        let Some(g) = self.guest.as_mut() else { return };
        let b = &self.buffers[&key];
        let (start, end) = align_range(addr, size);
        let (start, end) = (start.max(b.addr), end.min(b.addr + b.size));
        let (base, device) = (b.addr, b.buf.address);
        let tracker = &self.tracker;
        let wrote = g.make_resident(sched, start, end, |page| device + (page - base), |page| !tracker.has_cpu_dirty_pages(page, CACHING_PAGE));
        if wrote {
            self.stats.resident_flips += 1;
            sched.full_barrier();
        }
    }

    /// The import-table pages that CPU writes made stale in their mirrors
    /// read guest memory again.
    fn evict_dirty(&mut self, sched: &mut Scheduler) {
        if !self.tracker.dirty_log_pending() || self.guest.is_none() {
            return;
        }
        // Resident pages that may hold CPU writes now (16 KiB pages).
        let mut stale: Vec<u64> = Vec::new();
        match self.tracker.drain_dirty_log() {
            crate::memory::tracker::DirtyDrain::Ranges(ranges) => {
                let g = self.guest.as_ref().unwrap();
                for (a, n) in ranges {
                    let (start, end) = align_range(a, n);
                    stale.extend((start..end).step_by(CACHING_PAGE as usize).filter(|&p| g.is_resident(p)));
                }
            }
            crate::memory::tracker::DirtyDrain::Lost => {
                self.stats.dirty_log_lost += 1;
                let g = self.guest.as_ref().unwrap();
                for b in self.buffers.values() {
                    stale.extend((b.addr..b.addr + b.size).step_by(CACHING_PAGE as usize).filter(|&p| g.is_resident(p) && self.tracker.has_cpu_dirty_pages(p, CACHING_PAGE)));
                }
            }
        }
        if stale.is_empty() {
            return;
        }
        stale.sort_unstable();
        stale.dedup();
        // A page can mix a CPU write with GPU results its mirror holds
        // (other 4 KiB pages of the 16 KiB entry): those reach guest memory
        // before the entry points there.
        let mut through: Vec<(u64, u64)> = Vec::new();
        for &p in &stale {
            self.tracker.for_each_download_range(p, CACHING_PAGE, false, &mut |a, n| through.push((a, n)));
        }
        if !through.is_empty() {
            sched.full_barrier();
            for (a, n) in through {
                if let Some(key) = self.find_containing(a, n) {
                    let (src, base) = (self.buffers[&key].buf.buffer, self.buffers[&key].addr);
                    if let Some(g) = self.guest.as_mut() {
                        g.write_through(sched, src, base, a, a + n);
                    }
                }
            }
            sched.full_barrier();
        }
        let g = self.guest.as_mut().unwrap();
        let mut wrote = false;
        for p in stale {
            wrote |= g.evict(sched, p, p + CACHING_PAGE, |_| true);
        }
        if wrote {
            self.stats.resident_evictions += 1;
            sched.full_barrier();
        }
    }

    /// Read pages current in a mirror from it (default
    /// `RPS5E_BDA_RESIDENT=1`).
    pub fn set_bda_residency(&mut self, on: bool) {
        self.residency = on;
    }

    /// Pages current in a mirror are read from it.
    pub fn bda_residency(&self) -> bool {
        self.residency
    }

    /// A stage reading guest memory reads `[addr, addr+size)`: GPU results
    /// its mirror holds that guest memory does not have yet are written
    /// through first (in the command stream, before the stage).
    pub fn bda_make_current(&mut self, sched: &mut Scheduler, addr: u64, size: u64) {
        if self.guest.is_none() || self.residency || size == 0 {
            return;
        }
        // Nothing the GPU wrote since the range was made current.
        if self.current_memo.0 != self.gpu_write_epoch {
            self.current_memo.0 = self.gpu_write_epoch;
            self.current_memo.1.clear();
        }
        if !self.current_memo.1.insert((addr, size)) || !self.tracker.has_gpu_dirty_pages(addr, size) {
            return;
        }
        let mut runs: Vec<(u64, u64)> = Vec::new();
        self.tracker.for_each_download_range(addr, size, false, &mut |a, n| runs.push((a, n)));
        runs.retain(|&(a, n)| self.written_through_tick(a, n).is_none());
        if runs.is_empty() {
            return;
        }
        sched.full_barrier();
        self.batch_through = true;
        for (a, n) in runs {
            self.stats.lazy_throughs += 1;
            self.write_through(sched, a, n);
        }
        self.batch_through = false;
        host_read_barrier(&self.dev, sched);
        sched.full_barrier();
    }

    /// Guest memory is imported (`RPS5E_BDA=1` and the import worked).
    pub fn bda_active(&self) -> bool {
        self.guest.is_some()
    }

    /// Every page of the range is read through its import (else a mirror
    /// must serve it: see [`Self::obtain`]).
    pub fn bda_backed(&self, addr: u64, size: u64) -> bool {
        self.guest.as_ref().is_some_and(|g| g.is_backed(addr, size))
    }

    /// The page holding `addr` is read from its mirror by stages reading
    /// guest memory.
    pub fn bda_resident(&self, addr: u64) -> bool {
        self.guest.as_ref().is_some_and(|g| g.is_resident(addr))
    }

    /// Diagnostics: see [`guest::GuestImports::capture`].
    pub fn bda_capture(&mut self, sched: &mut Scheduler, addr: u64, size: u64, done: impl FnOnce(Vec<u8>) + Send + 'static) {
        if let Some(g) = self.guest.as_mut() {
            g.capture(sched, addr, size, done);
        }
    }

    pub fn import_stats(&self) -> Option<guest::ImportStats> {
        self.guest.as_ref().map(|g| g.stats)
    }

    /// True when the GPU holds newer data for some page of the range.
    /// A small read-only range whose guest bytes are newer than its mirror
    /// (CPU-dirty, not GPU-dirty), or that cannot be watched: binding a
    /// copy of the guest bytes made for this use avoids uploading it into
    /// the mirror and watching its pages again (per-frame constants the CPU
    /// rewrites every frame).
    pub fn prefers_stream(&mut self, addr: u64, size: u64) -> bool {
        if size > STREAM_MAX_BYTES || !stream_enabled() {
            return false;
        }
        let r = if self.memory.is_watchable(addr, size) {
            self.tracker.cpu_dirty_only(addr, size)
        } else {
            !self.tracker.has_gpu_dirty_pages(addr, size)
        };
        if r {
            self.stats.streamed += 1;
        }
        r
    }

    /// A range small enough to bind from a copy of its guest bytes.
    pub fn stream_eligible(&self, _addr: u64, size: u64) -> bool {
        size <= STREAM_MAX_BYTES && stream_enabled()
    }

    /// [`Self::prefers_stream`] without counting it (read-only).
    pub fn prefers_stream_query(&self, addr: u64, size: u64) -> bool {
        if size > STREAM_MAX_BYTES || !stream_enabled() {
            return false;
        }
        if self.memory.is_watchable(addr, size) {
            self.tracker.cpu_dirty_only(addr, size)
        } else {
            !self.tracker.has_gpu_dirty_pages(addr, size)
        }
    }

    /// What a read-only [`Self::obtain`] of `[addr, addr+size)` binds, when
    /// it would only bind (the mirror holds the range and is current):
    /// the slice, the buffer's key and serial. Read-only, for slice threads
    /// (R7b); otherwise why not.
    pub fn peek_read(&self, addr: u64, size: u64) -> Result<(BufferSlice, u64, u64), crate::slices::spec::Hole> {
        use crate::slices::spec::Hole;
        let size = size.max(1);
        if self.guest.is_some() {
            return Err(Hole::Special);
        }
        let key = self.find_containing(addr, size).ok_or(Hole::BufferMiss)?;
        if !self.tracker.known_cpu_clean(addr, size) {
            if !self.memory.is_watchable(addr, size) {
                return Err(Hole::Untracked);
            }
            if self.tracker.has_cpu_dirty_pages(addr, size) {
                return Err(Hole::Upload);
            }
        }
        let b = &self.buffers[&key];
        let offset = addr - b.addr;
        Ok((BufferSlice { buffer: b.buf.buffer, offset, size, device_address: b.buf.address + offset }, key, b.serial))
    }

    /// The mirror slice [`Self::obtain`] binds for a read of a range with
    /// CPU-dirty pages (watched memory, the buffer holds the range): its
    /// key and serial. Read-only (R7b stage 3: the upload is made at commit).
    pub fn peek_read_dirty(&self, addr: u64, size: u64) -> Option<(BufferSlice, u64, u64)> {
        let size = size.max(1);
        if self.guest.is_some() || !self.memory.is_watchable(addr, size) {
            return None;
        }
        let key = self.find_containing(addr, size)?;
        let b = &self.buffers[&key];
        let offset = addr - b.addr;
        Some((BufferSlice { buffer: b.buf.buffer, offset, size, device_address: b.buf.address + offset }, key, b.serial))
    }

    /// The device-address page table when it exists (see
    /// [`Self::bda_page_table`]).
    pub fn bda_page_table_existing(&self) -> Option<(vk::Buffer, u64)> {
        self.bda.as_ref().map(|t| (t.buffer, t.address))
    }

    /// The import table was asked for once (see [`Self::bda_imports`]).
    pub fn bda_tried(&self) -> bool {
        self.guest_tried
    }

    /// Mirrors used by a committed slice segment: their buffers count as
    /// used by `tick` (eviction, upload ordering).
    pub fn note_used(&mut self, key: u64, addr: u64, size: u64, tick: u64) {
        if let Some(b) = self.buffers.get_mut(&key) {
            b.last_use = tick;
        }
        if prologue_configured() {
            self.touched.add(tick, addr, size);
        }
    }

    /// GPU results are still on their way to guest memory after tick
    /// `completed`: write-throughs waiting for the next submit, or recorded
    /// in a later tick. Their pages may be CPU-readable (read-through) and
    /// hold the older bytes until then.
    pub fn writes_through_after(&self, completed: u64) -> bool {
        !self.pending_through.is_empty() || self.last_through_tick > completed
    }

    /// GPU-written ranges wait to be written through ([`Self::flush_write_through`]).
    pub fn has_pending_through(&self) -> bool {
        !self.pending_through.is_empty()
    }

    pub fn has_gpu_dirty(&self, addr: u64, size: u64) -> bool {
        self.tracker.has_gpu_dirty_pages(addr, size)
    }
}

#[cfg(test)]
mod tests;
