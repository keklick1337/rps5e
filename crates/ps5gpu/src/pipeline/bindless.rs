// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The bindless set: images selected through host table maps.
//!
//! A bindless table the draw cannot bind as a few candidates (too many
//! records its keys may reach, or records of several classes or shapes)
//! reaches the back-end as its heap V# ([`BindlessTable`]): the program reads
//! the word of its key from a map and samples the view in that slot of one
//! device-wide descriptor array (set [`bindless::SET`]: a storage buffer of
//! maps and a partially bound, update-after-bind sampled-image array). Per
//! table this module keeps the records it read last, one slot per distinct
//! T# and the map from keys to slots; a draw only checks that none of it
//! changed, so its cost does not grow with the table:
//!
//! * the table's memory: unchanged while the buffer cache's content stamp is
//!   (no CPU or GPU write, synchronisation point or unmap since), else while
//!   its pages' content generations are (and, for pages the CPU may write
//!   without a fault, no synchronisation point passed); otherwise the records
//!   are read again and only the changed ones are resolved again;
//! * its images: each is pinned in the image cache, which logs every pinned
//!   image whose use by a slot may need work again (a use other than
//!   sampling, a new lookup generation, a deletion) and counts changes of
//!   pinned images' guest memory; a table looks at the logged images only
//!   (all of them after a memory change), and once per tick at the images
//!   whose CPU writes no watch reports.
//!
//! A record reads the null view (zeros, as a null T# does) when it is null or
//! has no view the program can read: another dimension than the
//! instruction's (a 2D T# is a one-layer array, a cube T# its six faces), a
//! format the program would convert, a colour format under a depth
//! reference, no cached image. Slots and maps are never rewritten while
//! recorded work may read them: a replaced one is reused once its tick
//! completed.

use super::descriptors::DummyResources;
use super::sharps::{self, TextureRequest, TextureSharp};
use crate::backend::DrawContext;
use crate::device::{Buffer, GpuDevice, MemoryUse};
use crate::fxhash::HashMap;
use crate::images::layout::SubRange;
use crate::images::ImageUse;
use crate::memory::{GuestGpuMemory, PAGE_BYTES};
use ash::vk;
use ps5shader::resources::format::image_format;
use ps5shader::resources::{bindless, descriptor_dimension, image_binding, BindlessTable, DescriptorBindingKind, ImageDimension, ImageNumericClass};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Words of the map buffer (every table's map).
const MAP_WORDS: u32 = 1 << 22;
/// Ticks a table is kept unused (its images stay pinned meanwhile).
const IDLE_TICKS: u64 = crate::images::IDLE_TICKS;
/// A key without a distinct T# (null record).
const NONE: u32 = u32::MAX;

/// Counters of the bindless set.
#[derive(Clone, Copy, Debug, Default)]
pub struct BindlessStats {
    /// Draws and dispatches that bound a table.
    pub binds: u64,
    /// Binds whose table memory was read again.
    pub reads: u64,
    /// Records resolved again (changed records).
    pub resolved: u64,
    /// Distinct T#s looked at again (new ones, or their image changed).
    pub revalidated: u64,
    /// Maps written.
    pub maps: u64,
    /// Slots in use and tables kept.
    pub slots: u64,
    pub tables: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TableKey {
    /// Guest address of record 0 and the record count.
    addr: u64,
    entries: u32,
    kind: DescriptorBindingKind,
}

/// A pinned image a slot holds a view of.
#[derive(Clone, Copy)]
struct Held {
    id: u64,
    generation: u64,
    usage: ImageUse,
    range: Option<SubRange>,
}

/// One distinct T# of a table.
struct Distinct {
    words: [u32; 8],
    /// Keys whose record holds it.
    refs: u32,
    /// Its slot (a null slot when it has no view) and the view's class.
    slot: u32,
    mode: u32,
    /// The slot is its own (else a null slot).
    owned: bool,
    image: Option<Held>,
    view: vk::ImageView,
    /// Listed in the table's `failed` / `uncovered` lists.
    failed: bool,
    uncovered: bool,
}

impl Distinct {
    fn word(&self) -> u32 {
        self.slot | self.mode << bindless::MODE_SHIFT
    }
}

struct Table {
    key: TableKey,
    /// The records as read last (8 words each; empty before the first read).
    words: Vec<u32>,
    /// Per key: its distinct T# (NONE: null record).
    keys: Vec<u32>,
    distinct: Vec<Distinct>,
    free_distinct: Vec<u32>,
    by_words: HashMap<[u32; 8], u32>,
    by_image: HashMap<u64, smallvec::SmallVec<[u32; 2]>>,
    /// Distinct T#s not looked up yet.
    pending: Vec<u32>,
    /// Distinct T#s without a view, looked up again after a mapping change
    /// (`failed_maps`: the mapping generation they were tried at).
    failed: Vec<u32>,
    failed_maps: u64,
    /// Distinct T#s whose image's CPU writes no watch reports (looked at
    /// once per tick).
    uncovered: Vec<u32>,
    /// The first word of the map in use (None: not written yet) and whether
    /// it is behind the keys.
    map: Option<u32>,
    map_stale: bool,
    null_slot: u32,
    // Memory stamps of the records read last.
    stamp: [u64; 4],
    generations: Vec<Option<u32>>,
    volatile: bool,
    sync: u64,
    maps: u64,
    // Image stamps of the last look.
    cursor: u64,
    memory_changes: u64,
    tick_checked: u64,
    last_use: u64,
}

/// The device-wide bindless set and the tables bound through it.
pub struct BindlessSet {
    dev: Arc<GpuDevice>,
    pub layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    pub set: vk::DescriptorSet,
    maps: Buffer,
    map_free: BTreeMap<u32, u32>,
    map_retired: Vec<(u64, u32, u32)>,
    capacity: u32,
    next_slot: u32,
    free_slots: Vec<u32>,
    retired_slots: Vec<(u64, u32)>,
    /// The null view's slot per view type.
    /// The null view's slot per view type and storage class.
    null_slots: HashMap<(i32, bool), u32>,
    tables: HashMap<TableKey, Table>,
    collected: u64,
    // Scratch of one bind.
    scratch: Vec<u32>,
    generations: Vec<Option<u32>>,
    barriers: Vec<vk::ImageMemoryBarrier2<'static>>,
    writes: Vec<(u32, vk::DescriptorImageInfo)>,
    pub stats: BindlessStats,
}

impl Drop for BindlessSet {
    fn drop(&mut self) {
        // SAFETY: the owner waits for the device before dropping its caches.
        unsafe {
            self.dev.device.destroy_descriptor_pool(self.pool, None);
            self.dev.device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

/// The view an image selected through binding `kind` reads, its image and
/// its class ([`bindless::MODE_FLOAT`]...), or why there is none (it then
/// reads the null view).
fn table_view(ctx: &mut DrawContext, dummies: &mut DummyResources, dev: &GpuDevice, words: &[u32; 8], kind: DescriptorBindingKind) -> Result<(vk::ImageView, u64, u32), String> {
    let (_, _, dimension, _) = image_binding::describe(kind);
    if TextureSharp::from_words(words).is_null() {
        return Err("null T#".into());
    }
    // The program declares the instruction's view; a T# the hardware reads
    // as another shape has no view of it.
    let bound = descriptor_dimension(words, dimension);
    let same = bound == dimension || (dimension == ImageDimension::Dim2DArray && bound == ImageDimension::Dim2D) || (dimension == ImageDimension::Dim1DArray && bound == ImageDimension::Dim1D);
    if !same {
        return Err(format!("{bound:?} T# read as {dimension:?}"));
    }
    let format = image_format::format_of(words);
    if image_format::remap(format) != format {
        return Err(format!("format {format} converted by the program"));
    }
    let (_, _, _, atomic) = image_binding::describe(kind);
    if image_binding::is_atomic64(kind) {
        // No R64ui view on the table path: the element reads zeros and its
        // atomic faults nowhere (64-bit table atomics stay unsupported).
        return Err("64-bit image atomics through a table".into());
    }
    if atomic {
        // A 32-bit atomic runs on an R32ui storage view of the element
        // (the same view class its map word's mode selects).
        let req = TextureRequest { numeric: ImageNumericClass::Uint, dimension, storage: true, cube: false, host_swizzle: true, mip_offset: 0, single_level: false, atomic64: false, srgb: false };
        return match sharps::texture_view(ctx, Some(dummies), dev, words, &req)? {
            (view, Some(id)) => Ok((view, id, bindless::MODE_UINT)),
            (_, None) => Err("no cached image".into()),
        };
    }
    let compare = image_binding::is_compare(kind);
    if compare && !image_format::has_depth_equivalent(format) {
        return Err(format!("depth compare of colour format {format}"));
    }
    // The program reads float, unsigned and signed views (a depth
    // reference only float ones).
    let numeric = image_format::sampled_numeric_class(format);
    let mode = match numeric {
        ImageNumericClass::Float => bindless::MODE_FLOAT,
        ImageNumericClass::Uint if !compare => bindless::MODE_UINT,
        ImageNumericClass::Sint if !compare => bindless::MODE_SINT,
        _ => return Err(format!("format {format} read as {numeric:?}")),
    };
    let req = TextureRequest { numeric, dimension, storage: false, cube: false, host_swizzle: true, mip_offset: 0, single_level: false, atomic64: false, srgb: false };
    match sharps::texture_view(ctx, Some(dummies), dev, words, &req)? {
        (view, Some(id)) => Ok((view, id, mode)),
        (_, None) => Err("no cached image".into()),
    }
}

/// The base address of a V#.
fn heap_address(heap: &[u32; 4]) -> u64 {
    (heap[0] as u64 | ((heap[1] as u64) << 32)) & 0x0000_FFFF_FFFF_FFFF
}

impl BindlessSet {
    /// The set of a device with [`crate::device::DeviceCaps::bindless_slots`].
    pub fn new(dev: &Arc<GpuDevice>) -> Result<BindlessSet, vk::Result> {
        let capacity = dev.caps.bindless_slots;
        if capacity == 0 {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let d = &dev.device;
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(bindless::MAP_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(bindless::IMAGE_BINDING)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(capacity)
                .stage_flags(vk::ShaderStageFlags::ALL),
        ];
        let flags = [
            vk::DescriptorBindingFlags::empty(),
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING,
        ];
        let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
        let info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .bindings(&bindings)
            .push_next(&mut binding_flags);
        // SAFETY: valid create infos; the objects are destroyed on drop.
        let layout = unsafe { d.create_descriptor_set_layout(&info, None)? };
        let sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: capacity },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default().flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND).max_sets(1).pool_sizes(&sizes);
        let destroy = |pool: Option<vk::DescriptorPool>| {
            // SAFETY: objects of this function no work uses.
            unsafe {
                if let Some(p) = pool {
                    d.destroy_descriptor_pool(p, None);
                }
                d.destroy_descriptor_set_layout(layout, None);
            }
        };
        // SAFETY: as above.
        let pool = unsafe { d.create_descriptor_pool(&pool_info, None) }.inspect_err(|_| destroy(None))?;
        let layouts = [layout];
        // SAFETY: a pool of one set of this layout.
        let set = unsafe { d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)) }.inspect_err(|_| destroy(Some(pool)))?[0];
        let maps = Buffer::new(dev, MAP_WORDS as u64 * 4, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryUse::Upload).inspect_err(|_| destroy(Some(pool)))?;
        let buffer = [vk::DescriptorBufferInfo { buffer: maps.buffer, offset: 0, range: vk::WHOLE_SIZE }];
        let write = vk::WriteDescriptorSet::default().dst_set(set).dst_binding(bindless::MAP_BINDING).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&buffer);
        // SAFETY: the set is not in use yet.
        unsafe { d.update_descriptor_sets(&[write], &[]) };
        log::info!("ps5gpu: bindless set of {capacity} slots");
        let mut map_free = BTreeMap::new();
        map_free.insert(0, MAP_WORDS);
        Ok(BindlessSet {
            dev: dev.clone(),
            layout,
            pool,
            set,
            maps,
            map_free,
            map_retired: Vec::new(),
            capacity,
            next_slot: 0,
            free_slots: Vec::new(),
            retired_slots: Vec::new(),
            null_slots: HashMap::default(),
            tables: HashMap::default(),
            collected: 0,
            scratch: Vec::new(),
            generations: Vec::new(),
            barriers: Vec::new(),
            writes: Vec::new(),
            stats: BindlessStats::default(),
        })
    }

    /// The images the slots of every table hold views of.
    #[cfg(test)]
    pub fn held_images(&self) -> Vec<u64> {
        self.tables.values().flat_map(|t| t.distinct.iter().filter(|d| d.refs != 0).filter_map(|d| d.image.map(|h| h.id))).collect()
    }

    /// Slots and maps whose tick completed become free again.
    fn reclaim(&mut self, completed: u64) {
        let free = &mut self.free_slots;
        self.retired_slots.retain(|&(tick, slot)| {
            let done = tick <= completed;
            if done {
                free.push(slot);
            }
            !done
        });
        let mut done = Vec::new();
        self.map_retired.retain(|&(tick, first, len)| {
            let d = tick <= completed;
            if d {
                done.push((first, len));
            }
            !d
        });
        for (first, len) in done {
            self.free_map(first, len);
        }
    }

    fn free_map(&mut self, mut first: u32, mut len: u32) {
        // Merged with its free neighbours.
        if let Some((&prev, &plen)) = self.map_free.range(..first).next_back() {
            if prev + plen == first {
                self.map_free.remove(&prev);
                first = prev;
                len += plen;
            }
        }
        if let Some(next) = self.map_free.remove(&(first + len)) {
            len += next;
        }
        self.map_free.insert(first, len);
    }

    fn alloc_map(&mut self, len: u32) -> Option<u32> {
        let (&first, &have) = self.map_free.iter().find(|(_, &l)| l >= len)?;
        self.map_free.remove(&first);
        if have > len {
            self.map_free.insert(first + len, have - len);
        }
        Some(first)
    }

    fn alloc_slot(&mut self, sched: &crate::scheduler::Scheduler) -> Result<u32, String> {
        if self.free_slots.is_empty() && self.next_slot == self.capacity {
            self.reclaim(sched.timeline.refresh());
        }
        if let Some(s) = self.free_slots.pop() {
            return Ok(s);
        }
        if self.next_slot < self.capacity {
            self.next_slot += 1;
            return Ok(self.next_slot - 1);
        }
        Err(format!("bindless set full ({} slots)", self.capacity))
    }

    /// The slot of the null view of `kind`'s view type (an atomic kind's
    /// null view is the storage R32ui one).
    fn null_slot(&mut self, ctx: &mut DrawContext, dummies: &mut DummyResources, kind: DescriptorBindingKind) -> Result<u32, String> {
        let (_, numeric, dimension, atomic) = image_binding::describe(kind);
        let ty = sharps::view_type(dimension, false);
        let key = (ty.as_raw(), atomic);
        if let Some(&s) = self.null_slots.get(&key) {
            return Ok(s);
        }
        let view = dummies.image_view(ctx.sched, ty, numeric, false, atomic).ok_or("no null view")?;
        let slot = self.alloc_slot(ctx.sched)?;
        // Dummies stay in GENERAL.
        self.writes.push((slot, vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL }));
        self.null_slots.insert(key, slot);
        Ok(slot)
    }

    /// Distinct T# `d` no longer holds a view: its slot is reused once
    /// `tick` completed, its image unpinned.
    fn drop_view(&mut self, images: &mut crate::images::ImageCache, t: &mut Table, d: u32, tick: u64) {
        let x = &mut t.distinct[d as usize];
        if x.owned {
            self.retired_slots.push((tick, x.slot));
            x.owned = false;
        }
        x.slot = t.null_slot;
        x.mode = bindless::MODE_FLOAT;
        x.view = vk::ImageView::null();
        if let Some(held) = x.image.take() {
            images.unpin(held.id);
            if let Some(list) = t.by_image.get_mut(&held.id) {
                list.retain(|y| *y != d);
                if list.is_empty() {
                    t.by_image.remove(&held.id);
                }
            }
        }
    }

    /// Drop tables unused for [`IDLE_TICKS`].
    fn collect(&mut self, ctx: &mut DrawContext, now: u64) {
        let dead: Vec<TableKey> = self.tables.values().filter(|t| t.last_use + IDLE_TICKS < now).map(|t| t.key).collect();
        for key in dead {
            let Some(mut t) = self.tables.remove(&key) else { continue };
            for d in 0..t.distinct.len() as u32 {
                if let Some(images) = ctx.images.exact() {
                    self.drop_view(images, &mut t, d, now);
                }
            }
            if let Some(first) = t.map {
                self.map_retired.push((now, first, t.key.entries));
            }
        }
    }

    /// The flattened-table words ([`bindless::TABLE_WORDS`]) of `table` read
    /// through binding `kind`: its map's first word, its key count and the
    /// null slot. Changed records and images are resolved again first, and
    /// the barriers taking the table's images into their sampled layouts
    /// are recorded.
    pub fn bind(&mut self, ctx: &mut DrawContext, dummies: &mut DummyResources, memory: &'static GuestGpuMemory, table: &BindlessTable, kind: DescriptorBindingKind) -> Result<[u32; 3], String> {
        let now = ctx.sched.current_tick();
        self.reclaim(ctx.sched.timeline.completed_tick());
        if self.collected != now {
            self.collected = now;
            self.collect(ctx, now);
        }
        self.stats.binds += 1;
        let key = TableKey { addr: heap_address(&table.heap) + table.entry_offset as u64, entries: table.entries, kind };
        let mut t = match self.tables.remove(&key) {
            Some(t) => t,
            None => Table {
                key,
                words: Vec::new(),
                keys: vec![NONE; table.entries as usize],
                distinct: Vec::new(),
                free_distinct: Vec::new(),
                by_words: HashMap::default(),
                by_image: HashMap::default(),
                pending: Vec::new(),
                failed: Vec::new(),
                failed_maps: 0,
                uncovered: Vec::new(),
                map: None,
                map_stale: true,
                null_slot: self.null_slot(ctx, dummies, kind)?,
                stamp: [0; 4],
                generations: Vec::new(),
                volatile: true,
                sync: 0,
                maps: 0,
                cursor: 0,
                memory_changes: 0,
                tick_checked: 0,
                last_use: now,
            },
        };
        let result = self.bind_table(ctx, dummies, memory, &mut t, now);
        self.tables.insert(key, t);
        self.flush(ctx);
        self.stats.slots = (self.next_slot as usize - self.free_slots.len() - self.retired_slots.len()) as u64;
        self.stats.tables = self.tables.len() as u64;
        result
    }

    /// Apply the queued slot writes and record the queued barriers. Done
    /// before anything that may record other work on an image or submit the
    /// open tick (an image lookup: uploads, downloads, deletions): a write
    /// or barrier applied later could name a view or image retired, and
    /// destroyed by a submit, meanwhile, and a barrier recorded after
    /// another transition of its image would leave the wrong layout.
    fn flush(&mut self, ctx: &mut DrawContext) {
        if !self.writes.is_empty() {
            let infos: Vec<[vk::DescriptorImageInfo; 1]> = self.writes.iter().map(|w| [w.1]).collect();
            let writes: Vec<vk::WriteDescriptorSet> = self
                .writes
                .iter()
                .zip(&infos)
                .map(|(w, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.set)
                        .dst_binding(bindless::IMAGE_BINDING)
                        .dst_array_element(w.0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(info)
                })
                .collect();
            // SAFETY: slots no recorded work reads (new or reclaimed) and
            // views alive (no lookup ran since they were made); the binding
            // is update-after-bind and unused-while-pending.
            unsafe { self.dev.device.update_descriptor_sets(&writes, &[]) };
            self.writes.clear();
        }
        if !self.barriers.is_empty() {
            ctx.sched.image_barriers(&self.barriers);
            self.barriers.clear();
        }
    }

    fn bind_table(&mut self, ctx: &mut DrawContext, dummies: &mut DummyResources, memory: &'static GuestGpuMemory, t: &mut Table, now: u64) -> Result<[u32; 3], String> {
        t.last_use = now;
        self.flush(ctx);
        // ---- the records ----
        if let Some(changed) = self.read_records(ctx, memory, t)? {
            for k in changed {
                self.set_record(ctx, t, k as usize, now);
            }
        }
        // ---- the images ----
        let mut check = std::mem::take(&mut t.pending);
        let memory_changes = ctx.images.pinned_memory_changes();
        let cursor = ctx.images.pinned_cursor();
        // The pinned images whose memory changed, when the watch table
        // still knows them (else any held image may need an upload).
        let mut changed: Vec<u64> = Vec::new();
        let memory_known = memory_changes == t.memory_changes
            || (crate::images::pinned_log_enabled() && ctx.images.pinned_memory_changed_since(t.memory_changes, &mut changed).is_some());
        if !memory_known || ctx.images.pinned_since(t.cursor).is_none() {
            check.extend(t.by_image.values().flatten().copied());
        } else {
            if cursor != t.cursor {
                for id in ctx.images.pinned_since(t.cursor).unwrap_or_default() {
                    if let Some(ds) = t.by_image.get(id) {
                        check.extend_from_slice(ds);
                    }
                }
            }
            for id in &changed {
                if let Some(ds) = t.by_image.get(id) {
                    check.extend_from_slice(ds);
                }
            }
        }
        if t.tick_checked != now {
            t.tick_checked = now;
            let distinct = &mut t.distinct;
            t.uncovered.retain(|&d| distinct[d as usize].uncovered);
            check.extend_from_slice(&t.uncovered);
        }
        // Mappings changed: T#s without a view may have one now.
        let maps = memory.map_generation();
        if t.failed_maps != maps {
            t.failed_maps = maps;
            let distinct = &mut t.distinct;
            t.failed.retain(|&d| distinct[d as usize].failed);
            check.extend_from_slice(&t.failed);
        }
        check.sort_unstable();
        check.dedup();
        // A lookup may delete or change an image an earlier T# of this
        // table took (an image of more layers of the same surface absorbs
        // it, a fill replaces it): its T#s are looked at again until no
        // held image changed, so the map names no retired view.
        let mut round = 0;
        loop {
            let since = ctx.images.pinned_cursor();
            for d in check.drain(..) {
                self.revalidate(ctx, dummies, t, d, now)?;
            }
            match ctx.images.pinned_since(since) {
                Some(ids) => {
                    for id in ids {
                        if let Some(ds) = t.by_image.get(id) {
                            check.extend_from_slice(ds);
                        }
                    }
                }
                None => check.extend(t.by_image.values().flatten().copied()),
            }
            let images = &ctx.images;
            let distinct = &t.distinct;
            check.retain(|&d| distinct[d as usize].image.is_some_and(|h| images.image_generation(h.id) != Some(h.generation) || images.needs_upload(h.id)));
            if check.is_empty() {
                break;
            }
            round += 1;
            if round == 8 {
                return Err("bindless table images keep changing while looked up".into());
            }
            check.sort_unstable();
            check.dedup();
        }
        self.flush(ctx);
        // Our own uploads and transitions are seen: the log moves on.
        t.cursor = ctx.images.pinned_cursor();
        t.memory_changes = ctx.images.pinned_memory_changes();
        // ---- the map ----
        if t.map_stale || t.map.is_none() {
            let entries = t.key.entries;
            let first = match self.alloc_map(entries) {
                Some(f) => f,
                None => {
                    self.reclaim(ctx.sched.timeline.refresh());
                    self.alloc_map(entries).ok_or("bindless map buffer full")?
                }
            };
            if let Some(old) = t.map.replace(first) {
                self.map_retired.push((now, old, entries));
            }
            let bytes = self.maps.bytes_mut();
            // SAFETY: the map buffer's mapping holds MAP_WORDS u32s (mapped
            // memory is aligned far beyond 4 bytes).
            let words = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut u32, MAP_WORDS as usize) };
            let null = t.null_slot;
            for (word, &d) in words[first as usize..(first + entries) as usize].iter_mut().zip(&t.keys) {
                *word = if d == NONE { null } else { t.distinct[d as usize].word() };
            }
            self.maps.flush(first as u64 * 4, entries as u64 * 4);
            t.map_stale = false;
            self.stats.maps += 1;
        }
        Ok([t.map.unwrap_or(0), t.key.entries, t.null_slot])
    }

    /// The keys whose records changed since the last read (None: none
    /// did), with the records read into `t.words`.
    fn read_records(&mut self, ctx: &mut DrawContext, memory: &'static GuestGpuMemory, t: &mut Table) -> Result<Option<Vec<u32>>, String> {
        let stamp = ctx.buffers.content_stamp();
        let fresh = t.words.is_empty();
        if !fresh && stamp == t.stamp {
            return Ok(None);
        }
        let (addr, bytes) = (t.key.addr, t.key.entries as u64 * 32);
        if ctx.buffers.tracker().has_gpu_dirty_pages(addr, bytes) {
            // Records GPU work wrote: their bytes are fetched first (as for
            // every descriptor the host reads).
            ctx.buffers.download(ctx.sched, addr, bytes);
        }
        let tracker = ctx.buffers.tracker().clone();
        if tracker.has_gpu_dirty_pages(addr, bytes) {
            return Err(format!("bindless table at {addr:#x} is being written by GPU work"));
        }
        // The generations before the bytes: a write after them moves them.
        self.generations.clear();
        let mut block = None;
        let mut page = addr & !(PAGE_BYTES - 1);
        while page < addr + bytes {
            self.generations.push(tracker.page_generation_in(&mut block, page));
            page += PAGE_BYTES;
        }
        let volatile = self.generations.iter().any(|g| g.is_none()) || !tracker.known_cpu_clean(addr, bytes);
        let (sync, maps) = (crate::memory::sync_generation(), memory.map_generation());
        // A page without a region then (untracked: volatile) that has one
        // now holds the bytes read until the next synchronisation point.
        let same_sync = sync == t.sync;
        let same_pages = self.generations.len() == t.generations.len() && self.generations.iter().zip(&t.generations).all(|(now, then)| now == then || (then.is_none() && same_sync));
        if !fresh && same_pages && maps == t.maps && (!t.volatile || same_sync) {
            if self.generations != t.generations {
                std::mem::swap(&mut t.generations, &mut self.generations);
                t.volatile = volatile;
            }
            t.stamp = stamp;
            return Ok(None);
        }
        self.stats.reads += 1;
        self.scratch.clear();
        self.scratch.resize(t.key.entries as usize * 8, 0);
        // SAFETY: plain u32s viewed as bytes.
        let out = unsafe { std::slice::from_raw_parts_mut(self.scratch.as_mut_ptr() as *mut u8, self.scratch.len() * 4) };
        crate::capture::note_read(memory, addr, bytes);
        if !memory.read_backing(addr, out) {
            return Err(format!("bindless table at {addr:#x} is not mapped"));
        }
        std::mem::swap(&mut t.generations, &mut self.generations);
        t.volatile = volatile;
        t.sync = sync;
        t.maps = maps;
        t.stamp = stamp;
        let changed: Vec<u32> = if fresh {
            (0..t.key.entries).collect()
        } else {
            (0..t.key.entries).filter(|&k| t.words[k as usize * 8..k as usize * 8 + 8] != self.scratch[k as usize * 8..k as usize * 8 + 8]).collect()
        };
        std::mem::swap(&mut t.words, &mut self.scratch);
        Ok((!changed.is_empty()).then_some(changed))
    }

    /// Key `k`'s record is `t.words[k]` now.
    fn set_record(&mut self, ctx: &mut DrawContext, t: &mut Table, k: usize, now: u64) {
        let mut words = [0u32; 8];
        words.copy_from_slice(&t.words[k * 8..k * 8 + 8]);
        let old = t.keys[k];
        if old != NONE && t.distinct[old as usize].words == words {
            return;
        }
        self.stats.resolved += 1;
        t.map_stale = true;
        if old != NONE {
            t.distinct[old as usize].refs -= 1;
            if t.distinct[old as usize].refs == 0 {
                if let Some(images) = ctx.images.exact() {
                    self.drop_view(images, t, old, now);
                }
                let x = &mut t.distinct[old as usize];
                x.failed = false;
                x.uncovered = false;
                t.by_words.remove(&x.words);
                t.free_distinct.push(old);
            }
        }
        if TextureSharp::from_words(&words).is_null() {
            t.keys[k] = NONE;
            return;
        }
        let d = match t.by_words.get(&words) {
            Some(&d) => d,
            None => {
                let fresh = Distinct {
                    words,
                    refs: 0,
                    slot: t.null_slot,
                    mode: bindless::MODE_FLOAT,
                    owned: false,
                    image: None,
                    view: vk::ImageView::null(),
                    failed: false,
                    uncovered: false,
                };
                let d = match t.free_distinct.pop() {
                    Some(d) => {
                        t.distinct[d as usize] = fresh;
                        d
                    }
                    None => {
                        t.distinct.push(fresh);
                        (t.distinct.len() - 1) as u32
                    }
                };
                t.by_words.insert(words, d);
                t.pending.push(d);
                d
            }
        };
        t.distinct[d as usize].refs += 1;
        t.keys[k] = d;
    }

    /// Look at distinct T# `d` again: its view (looked up again when its
    /// image changed, needs an upload or there was none) and the layout its
    /// image is in.
    fn revalidate(&mut self, ctx: &mut DrawContext, dummies: &mut DummyResources, t: &mut Table, d: u32, now: u64) -> Result<(), String> {
        let x = &t.distinct[d as usize];
        if x.refs == 0 {
            return Ok(());
        }
        self.stats.revalidated += 1;
        if let Some(held) = x.image {
            if ctx.images.image_generation(held.id) == Some(held.generation) && !ctx.images.needs_upload(held.id) {
                // Only the layout (another use took the image away).
                ctx.images.transition(held.id, held.usage, held.range, &mut self.barriers);
                return Ok(());
            }
        }
        let words = x.words;
        self.flush(ctx);
        match table_view(ctx, dummies, &self.dev, &words, t.key.kind) {
            Ok((view, id, mode)) => {
                let generation = ctx.images.image_generation(id).unwrap_or(u64::MAX);
                let depth = ctx.images.get(id).is_some_and(|i| i.key.is_depth());
                // An atomic element's view is a storage one: it lives in
                // GENERAL with shader read / write access, not in a
                // sampled layout.
                let atomic = image_binding::describe(t.key.kind).3;
                let usage = if atomic {
                    crate::images::STORAGE
                } else if depth {
                    crate::images::SAMPLED_DEPTH
                } else {
                    crate::images::SAMPLED
                };
                let range = ctx.images.view_subresources(view);
                let covered = ctx.images.upload_covered(id);
                ctx.images.transition(id, usage, range, &mut self.barriers);
                let x = &t.distinct[d as usize];
                if x.view != view || !x.owned || x.mode != mode {
                    // A new view: a new slot (the old one may still be read).
                    let slot = self.alloc_slot(ctx.sched)?;
                    if let Some(images) = ctx.images.exact() {
                    self.drop_view(images, t, d, now);
                }
                    let x = &mut t.distinct[d as usize];
                    x.slot = slot;
                    x.owned = true;
                    x.mode = mode;
                    x.view = view;
                    self.writes.push((slot, vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: usage.0 }));
                    t.map_stale = true;
                }
                let x = &mut t.distinct[d as usize];
                let pinned = x.image.is_some_and(|h| h.id == id);
                x.image = Some(Held { id, generation, usage, range });
                x.failed = false;
                if !pinned {
                    ctx.images.pin(id);
                    t.by_image.entry(id).or_default().push(d);
                }
                if covered != !x.uncovered {
                    x.uncovered = !covered;
                    if !covered {
                        t.uncovered.push(d);
                    }
                }
            }
            Err(why) => {
                // Failed entries are looked at again whenever images change:
                // the message is formatted when the entry first fails.
                if !t.distinct[d as usize].failed {
                    crate::log_once(&format!("ps5gpu: bindless table entry reads zeros: {why}"));
                }
                if t.distinct[d as usize].slot != t.null_slot {
                    t.map_stale = true;
                }
                if let Some(images) = ctx.images.exact() {
                    self.drop_view(images, t, d, now);
                }
                let x = &mut t.distinct[d as usize];
                x.uncovered = false;
                if !x.failed {
                    x.failed = true;
                    t.failed.push(d);
                }
            }
        }
        Ok(())
    }
}
