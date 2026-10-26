// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest memory resident for the GPU (`RPS5E_BDA=1`).
//!
//! The backing memfd (its alias mapping) is imported into Vulkan in
//! [`CHUNK`]-sized pieces with VK_EXT_external_memory_host, on demand as
//! guest mappings reference them. An *import table* holds, for every 16 KiB
//! guest page, the device address of the page's bytes inside its import
//! (0 for pages that are not memfd-backed). Shader stages that only read
//! memory translate guest addresses through it, so their buffers need no
//! per-draw mirror, upload or descriptor: the GPU reads guest memory where
//! the CPU wrote it.
//!
//! Coherency:
//! * CPU writes are visible to the GPU through the host-coherent import.
//! * GPU writes into mirrors (written bindings, command-processor copies,
//!   image write-backs) are copied through to guest memory right after the
//!   work ([`GuestImports::write_through`]), before any later reading stage
//!   runs, so guest memory holds the newest bytes in GPU order.
//! * Pages without a memfd backing keep a mirror: their entries point at it
//!   (written with the mirror's own table entries).
//!
//! The table has the layout of the mirror page table
//! ([`super::BDA_PAGE_COUNT`] entries of [`super::CACHING_PAGE`] bytes), so
//! the emitted translation is the same for both.

use super::{BDA_PAGE_COUNT, BUFFER_USAGE, CACHING_PAGE, CACHING_PAGE_BITS};
use crate::device::{Buffer, GpuDevice, MemoryUse};
use crate::memory::{GuestGpuMemory, MapEvent};
use crate::scheduler::Scheduler;
use ash::vk;
use std::sync::Arc;

/// Bytes of the backing memfd imported as one buffer.
pub const CHUNK_BITS: u32 = 28;
pub const CHUNK: u64 = 1 << CHUNK_BITS;

/// `RPS5E_BDA=1`: read-only shader stages read guest memory through the
/// imports.
pub fn bda_configured() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_BDA").is_ok_and(|v| v == "1"))
}

/// `RPS5E_BDA_MIRROR_MIN` (bytes, default 64 KiB): ranges of at least this
/// size that stages read through the import table are mirrored in device
/// memory while the CPU does not write them; smaller ones are read in guest
/// memory over the bus.
pub fn mirror_min_bytes() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("RPS5E_BDA_MIRROR_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(64 << 10))
}

/// `RPS5E_BDA_RESIDENT=1` (experimental): pages current in a mirror point
/// at it (large ranges read from device memory instead of over the bus).
/// Off (the default), pages always read guest memory and every GPU write
/// into a mirror is copied through to guest memory (GPU_PLAN §27: frames
/// with residency still show stale reads in SILENT HILL's phone sequence).
pub fn residency_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_BDA_RESIDENT").is_ok_and(|v| v == "1"))
}

/// `RPS5E_BDA_PLAN=0`: stages reading guest memory keep the ordinary plan
/// (host-read data and materialised V#s, only the binding differs).
pub fn guest_plan_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_BDA_PLAN").is_ok_and(|v| v == "0"))
}

/// The plan options of stages reading guest memory: scalar data reads stay
/// in the program. `RPS5E_BDA_PLAN=all` also leaves buffer descriptors to
/// the program ([`ps5shader::resources::PlanOptions::guest_memory`]): only
/// correct once every guest mapping the program may address is visible to
/// the GPU (private commits and module segments have no import yet, and the
/// host no longer knows the ranges to mirror for them); `buffers` does that
/// alone (data read by the host).
pub fn guest_plan_options() -> ps5shader::resources::PlanOptions {
    let mut o = ps5shader::resources::PlanOptions::guest_memory();
    match std::env::var("RPS5E_BDA_PLAN").as_deref() {
        Ok("all") => {}
        Ok("buffers") => o.flatten_data_reads = true,
        _ => o.buffers_on_device = false,
    }
    o
}

/// `RPS5E_BDA_PRIVATE=0`: private guest mappings are not imported (mirrors
/// serve them).
pub fn private_imports_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_BDA_PRIVATE").is_ok_and(|v| v == "0"))
}

/// Import counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct ImportStats {
    pub chunks: u64,
    pub chunk_bytes: u64,
    pub failed_chunks: u64,
    pub import_us: u64,
    pub map_events: u64,
    /// Table entries written (pages).
    pub entries: u64,
    /// Mapping edges not on a 16 KiB page boundary (their page reads zero).
    pub partial_pages: u64,
    pub write_throughs: u64,
    pub write_through_bytes: u64,
    /// Pages pointed at their mirror, and back at guest memory.
    pub resident_pages: u64,
    pub evicted_pages: u64,
    /// Private read/write mappings imported in place, their bytes, and
    /// imports the driver refused.
    pub private_imports: u64,
    pub private_bytes: u64,
    pub failed_private: u64,
}

enum Chunk {
    Unknown,
    Imported(Buffer),
    Failed,
}

pub struct GuestImports {
    dev: Arc<GpuDevice>,
    memory: &'static GuestGpuMemory,
    alias_base: u64,
    alias_size: u64,
    chunks: Vec<Chunk>,
    /// The import table (device-local, [`BDA_PAGE_COUNT`] u64 entries).
    pub table: Buffer,
    /// Pages that have an import (one bit per page).
    backed: Vec<u64>,
    /// Private read/write guest mappings imported in place: address →
    /// (size, import).
    private: std::collections::BTreeMap<u64, (u64, Buffer)>,
    /// Backed pages whose entry points at their mirror instead (its bytes
    /// are current: uploaded and not written by the CPU since).
    resident: Vec<u64>,
    started: bool,
    pub stats: ImportStats,
}

fn bit(map: &[u64], page: u64) -> bool {
    map.get((page / 64) as usize).is_some_and(|w| w & (1 << (page % 64)) != 0)
}

fn set_bit(map: &mut [u64], page: u64, on: bool) {
    if let Some(w) = map.get_mut((page / 64) as usize) {
        if on {
            *w |= 1 << (page % 64);
        } else {
            *w &= !(1 << (page % 64));
        }
    }
}

impl GuestImports {
    /// The imports of `memory`'s backing alias; None without an alias or
    /// without VK_EXT_external_memory_host (or a misaligned alias).
    pub fn new(dev: &Arc<GpuDevice>, memory: &'static GuestGpuMemory, sched: &mut Scheduler) -> Result<Option<GuestImports>, vk::Result> {
        let (base, size) = memory.backing_alias();
        let align = dev.caps.min_imported_host_pointer_alignment.max(1);
        if base == 0 || !dev.caps.external_memory_host || base % align != 0 || CHUNK % align != 0 {
            log::warn!("ps5gpu: guest memory cannot be imported (alias {base:#x}+{size:#x}, host import {}, alignment {align:#x})", dev.caps.external_memory_host);
            return Ok(None);
        }
        let table = Buffer::new(dev, BDA_PAGE_COUNT * 8, BUFFER_USAGE, MemoryUse::Device)?;
        let cb = sched.cmd();
        // SAFETY: recording buffer.
        unsafe { dev.device.cmd_fill_buffer(cb, table.buffer, 0, vk::WHOLE_SIZE, 0) };
        sched.full_barrier();
        let chunks = (0..size.div_ceil(CHUNK)).map(|_| Chunk::Unknown).collect();
        log::info!("ps5gpu: guest memory resident for the GPU: alias {base:#x}+{size:#x} in {} MiB imports", CHUNK >> 20);
        Ok(Some(GuestImports {
            dev: dev.clone(),
            memory,
            alias_base: base,
            alias_size: size,
            chunks,
            table,
            backed: vec![0; (BDA_PAGE_COUNT / 64) as usize],
            private: Default::default(),
            resident: vec![0; (BDA_PAGE_COUNT / 64) as usize],
            started: false,
            stats: ImportStats::default(),
        }))
    }

    /// The import holding backing chunk `index` (made on first use): its
    /// buffer, device address and size (less than CHUNK for a rounded-down
    /// last chunk).
    fn chunk(&mut self, index: usize) -> Option<(vk::Buffer, u64, u64)> {
        if let Some(Chunk::Unknown) = self.chunks.get(index) {
            let start = index as u64 * CHUNK;
            let align = self.dev.caps.min_imported_host_pointer_alignment.max(1);
            // The import must not reach past the alias: with an alignment
            // above the alias' own granularity the last chunk is rounded
            // down, and its tail stays without a device address
            // (`backing_address` checks the imported size).
            let len = (self.alias_size - start).min(CHUNK) / align * align;
            if len == 0 {
                self.stats.failed_chunks += 1;
                self.chunks[index] = Chunk::Failed;
                return None;
            }
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
            let t0 = std::time::Instant::now();
            // SAFETY: the alias maps the whole memfd read/write for the
            // life of the process; the import is never released before it.
            let r = unsafe { Buffer::import_host(&self.dev, (self.alias_base + start) as *mut u8, len, usage) };
            self.stats.import_us += t0.elapsed().as_micros() as u64;
            self.chunks[index] = match r {
                Ok(b) if b.coherent && b.address != 0 => {
                    self.stats.chunks += 1;
                    self.stats.chunk_bytes += len;
                    Chunk::Imported(b)
                }
                Ok(_) => {
                    self.stats.failed_chunks += 1;
                    log::warn!("ps5gpu: guest memory import {start:#x}+{len:#x} is not coherent or has no device address");
                    Chunk::Failed
                }
                Err(e) => {
                    self.stats.failed_chunks += 1;
                    log::warn!("ps5gpu: guest memory import {start:#x}+{len:#x} failed: {e:?}");
                    Chunk::Failed
                }
            };
        }
        match self.chunks.get(index) {
            Some(Chunk::Imported(b)) => Some((b.buffer, b.address, b.size)),
            _ => None,
        }
    }

    /// Device address of backing bytes `[b, b+len)` (inside one chunk).
    fn backing_address(&mut self, b: u64, len: u64) -> Option<u64> {
        if len == 0 || b + len > self.alias_size || (b >> CHUNK_BITS) != ((b + len - 1) >> CHUNK_BITS) {
            return None;
        }
        let (_, address, size) = self.chunk((b >> CHUNK_BITS) as usize)?;
        let off = b & (CHUNK - 1);
        // A rounded-down last chunk does not cover its tail.
        (off + len <= size).then_some(address + off)
    }

    /// Apply the guest mapping changes since the last call (the current
    /// mappings on the first) to the table. Returns the ranges that lost
    /// their import entries without gaining others (mapped without a memfd
    /// backing, or unmapped): mirrors over them write their own entries
    /// again.
    pub fn sync(&mut self, sched: &mut Scheduler) -> Vec<(u64, u64)> {
        let events = if !self.started {
            self.started = true;
            self.memory.enable_map_log()
        } else if self.memory.pending_map_events() == 0 {
            return Vec::new();
        } else {
            self.memory.take_map_events()
        };
        if events.is_empty() {
            return Vec::new();
        }
        self.stats.map_events += events.len() as u64;
        let mut cleared = Vec::new();
        for e in events {
            match e {
                MapEvent::Mapped { addr, size, backing: Some(backing) } => {
                    // Edge pages shared with other mappings lose their entry:
                    // a mirror serves them.
                    for page in self.map(sched, addr, size, backing) {
                        cleared.push((page << CACHING_PAGE_BITS, CACHING_PAGE));
                    }
                }
                MapEvent::Mapped { addr, size, backing: None } => {
                    self.drop_private(sched, addr, size, &mut cleared);
                    self.clear(sched, addr, size);
                    match self.import_private(sched, addr, size) {
                        Some(edges) => cleared.extend(edges.into_iter().map(|p| (p << CACHING_PAGE_BITS, CACHING_PAGE))),
                        None => cleared.push((addr, size)),
                    }
                }
                MapEvent::Unmapped { addr, size } => {
                    self.drop_private(sched, addr, size, &mut cleared);
                    self.clear(sched, addr, size);
                    cleared.push((addr, size));
                }
            }
        }
        sched.full_barrier();
        cleared
    }

    /// Private guest memory (no memfd backing) mapped read/write is
    /// imported where it lives (the guest address is the host address);
    /// read-only or inaccessible ranges are not (the NVIDIA driver refuses
    /// them and then fails every later allocation). Returns the cut edge
    /// pages, None when nothing was imported.
    fn import_private(&mut self, sched: &mut Scheduler, addr: u64, size: u64) -> Option<smallvec::SmallVec<[u64; 2]>> {
        if !private_imports_enabled() {
            return None;
        }
        let align = self.dev.caps.min_imported_host_pointer_alignment.max(4096);
        if addr % align != 0 || size % align != 0 || size == 0 || addr.saturating_add(size) > BDA_PAGE_COUNT << CACHING_PAGE_BITS {
            return None;
        }
        if !ps5vm::ranges::accessible(addr, size, true) {
            return None;
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let t0 = std::time::Instant::now();
        // SAFETY: a read/write guest mapping; the import is dropped (after
        // the GPU is done with it) when the mapping changes.
        let r = unsafe { Buffer::import_host(&self.dev, addr as *mut u8, size, usage) };
        self.stats.import_us += t0.elapsed().as_micros() as u64;
        let b = match r {
            Ok(b) if b.coherent && b.address != 0 => b,
            Ok(_) => return None,
            Err(e) => {
                self.stats.failed_private += 1;
                crate::log_once(&format!("ps5gpu: private guest memory import failed ({e:?}); mirrors serve it"));
                return None;
            }
        };
        self.stats.private_imports += 1;
        self.stats.private_bytes += size;
        let (first, last, _) = Self::inner_pages(addr, size);
        let mut edges = smallvec::SmallVec::new();
        for p in [addr >> CACHING_PAGE_BITS, (addr + size - 1) >> CACHING_PAGE_BITS] {
            if p < first || p >= last {
                if !edges.contains(&p) {
                    edges.push(p);
                }
            }
        }
        let entries: Vec<u64> = (first..last).map(|p| b.address + ((p << CACHING_PAGE_BITS) - addr)).collect();
        for p in first..last {
            set_bit(&mut self.backed, p, true);
            set_bit(&mut self.resident, p, false);
        }
        for (i, piece) in entries.chunks(1 << 16).enumerate() {
            self.write_run(sched, first + (i << 16) as u64, piece);
        }
        self.private.insert(addr, (size, b));
        Some(edges)
    }

    /// Drop the private imports overlapping `[addr, addr+size)`; the parts
    /// of them outside the range lose their entries too (`cleared`).
    fn drop_private(&mut self, sched: &mut Scheduler, addr: u64, size: u64, cleared: &mut Vec<(u64, u64)>) {
        let end = addr.saturating_add(size);
        let keys: Vec<u64> = self.private.range(..end).filter(|(&a, (n, _))| a + n > addr).map(|(&a, _)| a).collect();
        for a in keys {
            let (n, b) = self.private.remove(&a).unwrap();
            sched.retire(b);
            self.clear(sched, a, n);
            cleared.push((a, n));
        }
    }

    /// Pages fully inside `[addr, addr+size)`, and whether an edge page was
    /// cut.
    fn inner_pages(addr: u64, size: u64) -> (u64, u64, bool) {
        let end = addr.saturating_add(size).min(BDA_PAGE_COUNT << CACHING_PAGE_BITS);
        let first = addr.div_ceil(CACHING_PAGE);
        let last = end >> CACHING_PAGE_BITS;
        (first, last.max(first), addr % CACHING_PAGE != 0 || end % CACHING_PAGE != 0)
    }

    /// Returns the edge pages left without an entry.
    fn map(&mut self, sched: &mut Scheduler, addr: u64, size: u64, backing: u64) -> smallvec::SmallVec<[u64; 2]> {
        let mut edges = smallvec::SmallVec::new();
        let (first, last, cut) = Self::inner_pages(addr, size);
        if cut {
            self.stats.partial_pages += 1;
            // The cut edge pages may hold another mapping's bytes too.
            let end = addr.saturating_add(size);
            for p in [addr >> CACHING_PAGE_BITS, end.saturating_sub(1) >> CACHING_PAGE_BITS] {
                if p < BDA_PAGE_COUNT && (p < first || p >= last) {
                    self.write_run(sched, p, &[0]);
                    set_bit(&mut self.backed, p, false);
                    set_bit(&mut self.resident, p, false);
                    if !edges.contains(&p) {
                        edges.push(p);
                    }
                }
            }
        }
        let mut run: Vec<u64> = Vec::new();
        let mut run_first = first;
        for page in first..last {
            let b = backing + ((page << CACHING_PAGE_BITS) - addr);
            let entry = self.backing_address(b, CACHING_PAGE).unwrap_or(0);
            set_bit(&mut self.backed, page, entry != 0);
            set_bit(&mut self.resident, page, false);
            if run.is_empty() {
                run_first = page;
            }
            run.push(entry);
            if run.len() == 1 << 16 {
                self.write_run(sched, run_first, &run);
                run.clear();
            }
        }
        if !run.is_empty() {
            self.write_run(sched, run_first, &run);
        }
        edges
    }

    fn clear(&mut self, sched: &mut Scheduler, addr: u64, size: u64) {
        let first = addr >> CACHING_PAGE_BITS;
        let last = (addr.saturating_add(size).div_ceil(CACHING_PAGE)).min(BDA_PAGE_COUNT);
        if first >= last {
            return;
        }
        for p in first..last {
            set_bit(&mut self.backed, p, false);
            set_bit(&mut self.resident, p, false);
        }
        let cb = sched.cmd();
        // SAFETY: recording buffer; offset and size are multiples of 8.
        unsafe { self.dev.device.cmd_fill_buffer(cb, self.table.buffer, first * 8, (last - first) * 8, 0) };
        self.stats.entries += last - first;
    }

    /// Write `entries` from page `first` on (in the command stream).
    fn write_run(&mut self, sched: &mut Scheduler, first: u64, entries: &[u64]) {
        let bytes: &[u8] = bytemuck_u64(entries);
        self.stats.entries += entries.len() as u64;
        if bytes.len() <= 65536 {
            let cb = sched.cmd();
            // SAFETY: recording buffer; at most 64 KiB, multiple of 4.
            unsafe { self.dev.device.cmd_update_buffer(cb, self.table.buffer, first * 8, bytes) };
            return;
        }
        let Ok(staging) = Buffer::new(&self.dev, bytes.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC, MemoryUse::Upload) else {
            for (i, piece) in entries.chunks(8192).enumerate() {
                self.write_run(sched, first + (i * 8192) as u64, piece);
            }
            return;
        };
        staging.bytes_mut()[..bytes.len()].copy_from_slice(bytes);
        staging.flush(0, bytes.len() as u64);
        let cb = sched.cmd();
        let region = vk::BufferCopy { src_offset: 0, dst_offset: first * 8, size: bytes.len() as u64 };
        // SAFETY: recording buffer; the staging buffer is retired after the tick.
        unsafe { self.dev.device.cmd_copy_buffer(cb, staging.buffer, self.table.buffer, &[region]) };
        sched.retire(staging);
    }

    /// The table holds the guest mappings ([`Self::sync`] ran once).
    pub fn started(&self) -> bool {
        self.started
    }

    /// Every page of `[addr, addr+size)` has an import entry.
    pub fn is_backed(&self, addr: u64, size: u64) -> bool {
        if size == 0 {
            return true;
        }
        let first = addr >> CACHING_PAGE_BITS;
        let last = (addr + size - 1) >> CACHING_PAGE_BITS;
        last < BDA_PAGE_COUNT && (first..=last).all(|p| bit(&self.backed, p))
    }

    /// A mirror at `device_address` (0: released) covers
    /// `[addr, addr+size)` (caching-page aligned): the pages without an
    /// import entry point at it; resident pages (which pointed at the
    /// mirrors it replaces, or at it) follow it, or read guest memory again
    /// when it goes.
    pub fn mirror_entries(&mut self, sched: &mut Scheduler, addr: u64, size: u64, device_address: u64) -> bool {
        let first = addr >> CACHING_PAGE_BITS;
        let count = (size >> CACHING_PAGE_BITS).min(BDA_PAGE_COUNT.saturating_sub(first));
        let mut wrote = false;
        let mut p = 0;
        while p < count {
            let backed = bit(&self.backed, first + p);
            if backed && !bit(&self.resident, first + p) {
                p += 1;
                continue;
            }
            let start = p;
            while p < count && bit(&self.backed, first + p) == backed && (!backed || bit(&self.resident, first + p)) {
                p += 1;
            }
            let entries: Vec<u64> = if backed && device_address == 0 {
                (start..p).map(|i| self.import_entry(first + i)).collect()
            } else {
                (start..p).map(|i| if device_address == 0 { 0 } else { device_address + (i << CACHING_PAGE_BITS) }).collect()
            };
            if backed && device_address == 0 {
                for i in start..p {
                    set_bit(&mut self.resident, first + i, false);
                }
            }
            self.write_run(sched, first + start, &entries);
            wrote = true;
        }
        wrote
    }

    /// The import address of backed page `page` (0 when its 16 KiB do not
    /// lie in one import).
    fn import_entry(&mut self, page: u64) -> u64 {
        let a = page << CACHING_PAGE_BITS;
        let runs = self.memory.backing_runs(a, CACHING_PAGE);
        match runs.as_slice() {
            &[(g, b, n)] if g == a && n == CACHING_PAGE => self.backing_address(b, CACHING_PAGE).unwrap_or(0),
            _ => 0,
        }
    }

    /// Every page of `[addr, addr+size)` is read from its mirror.
    pub fn all_resident(&self, addr: u64, size: u64) -> bool {
        let last = (addr + size.max(1) - 1) >> CACHING_PAGE_BITS;
        (addr >> CACHING_PAGE_BITS..=last).all(|p| bit(&self.resident, p))
    }

    /// The page holding `addr` is read from its mirror.
    pub fn is_resident(&self, addr: u64) -> bool {
        bit(&self.resident, addr >> CACHING_PAGE_BITS)
    }

    /// Point the backed, not yet resident pages of `[start, end)` (caching
    /// pages) for which `current(page address)` holds at their mirror
    /// bytes at `mirror(page address)`. True when an entry was written.
    pub fn make_resident(&mut self, sched: &mut Scheduler, start: u64, end: u64, mirror: impl Fn(u64) -> u64, current: impl Fn(u64) -> bool) -> bool {
        let (first, last) = (start >> CACHING_PAGE_BITS, (end >> CACHING_PAGE_BITS).min(BDA_PAGE_COUNT));
        let mut wrote = false;
        let mut run: Vec<u64> = Vec::new();
        let mut run_first = 0;
        for page in first..last {
            let take = bit(&self.backed, page) && !bit(&self.resident, page) && current(page << CACHING_PAGE_BITS);
            if take {
                if run.is_empty() {
                    run_first = page;
                }
                run.push(mirror(page << CACHING_PAGE_BITS));
                set_bit(&mut self.resident, page, true);
                self.stats.resident_pages += 1;
            }
            if (!take || page + 1 == last) && !run.is_empty() {
                self.write_run(sched, run_first, &run);
                run.clear();
                wrote = true;
            }
        }
        wrote
    }

    /// Point the resident pages of `[start, end)` for which
    /// `stale(page address)` holds back at guest memory. True when an entry
    /// was written.
    pub fn evict(&mut self, sched: &mut Scheduler, start: u64, end: u64, stale: impl Fn(u64) -> bool) -> bool {
        let (first, last) = (start >> CACHING_PAGE_BITS, (end.div_ceil(CACHING_PAGE)).min(BDA_PAGE_COUNT));
        let mut wrote = false;
        let mut run: Vec<u64> = Vec::new();
        let mut run_first = 0;
        for page in first..last {
            let take = bit(&self.resident, page) && stale(page << CACHING_PAGE_BITS);
            if take {
                if run.is_empty() {
                    run_first = page;
                }
                let entry = self.import_entry(page);
                run.push(entry);
                set_bit(&mut self.resident, page, false);
                self.stats.evicted_pages += 1;
            }
            if (!take || page + 1 == last) && !run.is_empty() {
                self.write_run(sched, run_first, &run);
                run.clear();
                wrote = true;
            }
        }
        wrote
    }

    /// Copy `[start, end)` of mirror `src` (whose byte 0 mirrors guest
    /// address `src_base`) into guest memory through the imports, in the
    /// command stream (after the barrier the caller recorded). False when
    /// some part has no import (it keeps older bytes).
    pub fn write_through(&mut self, sched: &mut Scheduler, src: vk::Buffer, src_base: u64, start: u64, end: u64) -> bool {
        let mut complete = true;
        let mut covered = 0;
        let mut regions: smallvec::SmallVec<[(vk::Buffer, vk::BufferCopy); 4]> = smallvec::SmallVec::new();
        for (g, b, n) in self.memory.backing_runs(start, end - start) {
            covered += n;
            let mut done = 0;
            while done < n {
                let bo = b + done;
                let piece = (n - done).min(CHUNK - (bo & (CHUNK - 1)));
                match self.chunk((bo >> CHUNK_BITS) as usize) {
                    Some((buffer, _, size)) if (bo & (CHUNK - 1)) + piece <= size => {
                        regions.push((buffer, vk::BufferCopy { src_offset: g + done - src_base, dst_offset: bo & (CHUNK - 1), size: piece }));
                    }
                    _ => complete = false,
                }
                done += piece;
            }
        }
        // Private mappings imported in place.
        for (&a, (n, b)) in self.private.range(..end) {
            let (s0, e0) = (a.max(start), (a + n).min(end));
            if s0 < e0 {
                covered += e0 - s0;
                regions.push((b.buffer, vk::BufferCopy { src_offset: s0 - src_base, dst_offset: s0 - a, size: e0 - s0 }));
            }
        }
        complete &= covered == end - start;
        if regions.is_empty() {
            return false;
        }
        let cb = sched.cmd();
        for (dst, region) in &regions {
            // SAFETY: recording buffer; the mirror is alive until the tick
            // completes and the import for the process.
            unsafe { self.dev.device.cmd_copy_buffer(cb, src, *dst, std::slice::from_ref(region)) };
            self.stats.write_through_bytes += region.size;
        }
        self.stats.write_throughs += 1;
        complete
    }
}

impl GuestImports {
    /// Diagnostics: copy `size` bytes of guest memory at `addr` as the GPU
    /// sees them through the import, at this point of the command stream,
    /// and hand them to `done` once the tick completed.
    pub fn capture(&mut self, sched: &mut Scheduler, addr: u64, size: u64, done: impl FnOnce(Vec<u8>) + Send + 'static) {
        let Some(&(g, b, n)) = self.memory.backing_runs(addr, size).first() else { return };
        if g != addr || n < size {
            return;
        }
        let Some((src, _, chunk_size)) = self.chunk((b >> CHUNK_BITS) as usize) else { return };
        if (b & (CHUNK - 1)) + size > chunk_size {
            return;
        }
        let Ok(out) = Buffer::new(&self.dev, size, vk::BufferUsageFlags::TRANSFER_DST, MemoryUse::Download) else { return };
        sched.full_barrier();
        // SAFETY: recording buffer; both buffers outlive the tick.
        unsafe { self.dev.device.cmd_copy_buffer(sched.cmd(), src, out.buffer, &[vk::BufferCopy { src_offset: b & (CHUNK - 1), dst_offset: 0, size }]) };
        sched.full_barrier();
        sched.queue_priority(move || {
            out.invalidate(0, size);
            done(out.bytes_mut()[..size as usize].to_vec());
        });
    }
}

fn bytemuck_u64(v: &[u64]) -> &[u8] {
    // SAFETY: u64 has no padding; the slice covers exactly the words.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[cfg(test)]
mod tests;
