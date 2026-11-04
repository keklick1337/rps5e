// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The host mirrors of guest BVHs.
//!
//! Life of a mirror: a shader reports the root of a BVH it does not find
//! in the table (feedback ring) → the host reads the BVH's header on the
//! GPU (a probe; the bytes are GPU-written and need not be in guest
//! memory) → a BVH with an instance array is a TLAS and gets a table entry
//! at once; one without is a BLAS, which gets an entry (traced as the top
//! level, one identity instance) only when no mirrored TLAS uses it a few
//! updates later. At every update (before a BVH dispatch, when guest memory
//! may have changed since the last one) the host extracts the triangles of
//! every mirrored BLAS from the guest's nodes, builds or refits the host
//! BLASes, turns each entry's instance records into host instances and
//! builds its TLAS. An instance whose BLAS the host does not mirror yet is
//! reported in the request ring and makes the entry invalid (no pruning)
//! until the host has built that BLAS.
//!
//! `RPS5E_HW_RT_DUMP=<dir>` (diagnostics) writes the guest bytes of every
//! reported BVH there (with `RPS5E_HW_RT_DUMP_AT=<s>`: those reported from
//! then on).

use super::kernels::{paging, Kernel, Push};
use super::table;
use crate::device::{Buffer, GpuDevice, MemoryUse};
use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;

/// Slots of the BLAS map ({key, AS address, capacity, stale}).
const MAP_SLOTS: usize = 4096;
/// Words per map slot of the status the triangle kernel keeps: built entry
/// count, flags of update banks 0..BANKS.
const STATUS_WORDS: usize = 8;
/// Updates write their flags to bank `serial % BANKS`; the host reads a bank
/// once its update completed (the GPU may run a few updates behind).
const BANKS: u64 = 4;
/// Status flags: a triangle changed; the entry count is not the built one.
const CHANGED: u32 = 1;
const REBUILD: u32 = 2;
/// Updates without a guest BVH write at most this far apart (CPU writes
/// and command-processor writes are not watched).
const REFRESH: std::time::Duration = std::time::Duration::from_secs(1);
/// Request ring records ({key, entries}).
const REQUEST_SLOTS: usize = 1024;
/// Slots of an entry's BLAS set.
const SET_SLOTS: usize = 4096;
/// Updates a BLAS root waits before it is mirrored as a top level (it must
/// be reported again after the feedback filter was cleared).
const TOP_DELAY: u64 = 2 * FILTER_PERIOD;
/// Refits of a host BLAS between full builds (`RPS5E_HW_RT_REFITS`,
/// default 32; 0: every change builds it again).
fn refits() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("RPS5E_HW_RT_REFITS").ok().and_then(|v| v.parse().ok()).unwrap_or(32))
}
/// Updates between clears of the feedback filter.
const FILTER_PERIOD: u64 = 64;
/// A table entry or a host BLAS unused this long is freed.
const IDLE: std::time::Duration = std::time::Duration::from_secs(3);

/// `RPS5E_HW_RT_HOST=0` (diagnostics): no mirror is offered to the
/// traversal copies (they stay compiled; the guest's loops run).
fn host_traversals() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_HW_RT_HOST").is_ok_and(|v| v == "0"))
}

/// A guest BLAS mirrored on the host.
struct HostBlas {
    /// Leaf entries the host BLAS has room for (the guest's count rounded
    /// up, so small changes fit).
    entries: u32,
    /// Bytes of the guest block (watched for writes).
    bytes: u64,
    /// The map slot of the last full build (the status words there are its
    /// own).
    slot: Option<u32>,
    vertices: Buffer,
    storage: Buffer,
    accel: vk::AccelerationStructureKHR,
    address: u64,
    build_scratch: u64,
    update_scratch: u64,
    /// Built at least once (refits allowed), refits since the last build.
    built: bool,
    refits: u32,
    /// Last time an instance of a mirrored TLAS used it.
    used: std::time::Instant,
}

/// A table entry: a guest top level mirrored as a host TLAS.
struct Top {
    guest: u64,
    /// A bottom level traced as the top (one identity instance).
    single: bool,
    /// Bytes of the guest block (watched for writes).
    bytes: u64,
    capacity: u32,
    instances: Buffer,
    set: Buffer,
    storage: Buffer,
    accel: vk::AccelerationStructureKHR,
    address: u64,
    scratch: u64,
    /// Last time a shader queried its root.
    used: std::time::Instant,
}

/// A header read in flight.
struct Probe {
    base: u64,
    tick: u64,
    out: Buffer,
}

pub struct Mirror {
    dev: Arc<GpuDevice>,
    scratch_align: u64,
    instances_k: Option<Kernel>,
    triangles_k: Option<Kernel>,
    gather_k: Option<Kernel>,
    /// Feedback entries consumed.
    seen: u32,
    /// Bases probed, registered or rejected (not probed again).
    known: HashMap<u64, Known>,
    probes: Vec<Probe>,
    /// BLAS roots that may be top levels (base → update it was probed at).
    tops_pending: HashMap<u64, u64>,
    tops: Vec<Option<Top>>,
    blas: HashMap<u64, HostBlas>,
    /// The BLAS map: host-visible, written in the command stream.
    map: Buffer,
    /// Per map slot, the device address of its BLAS's list starts.
    starts: Buffer,
    requests: Buffer,
    /// The unit-box BLAS (one procedural AABB) standing in for BLASes not
    /// mirrored yet: its storage and AABB buffers, address, built.
    proxy: (Buffer, Buffer, vk::AccelerationStructureKHR, u64, u64, bool),
    /// A word per map slot the instance kernels set (the host clears it).
    used: Buffer,
    requests_seen: u32,
    /// Tick of the last update (its requests are complete after it).
    update_tick: u64,
    scratch: Option<Buffer>,
    /// Entries replaced since the last update (retired there).
    dropped: Vec<Top>,
    jobs: Option<(Buffer, Buffer, u32)>,
    jobs_dirty: bool,
    /// Map slot of every mirrored BLAS (as last written).
    placed: HashMap<u64, u32>,
    /// Write generation of the watched guest BVHs at the last update (None:
    /// the next BVH dispatch updates).
    stamp: Option<u64>,
    last_update: std::time::Instant,
    /// The watched ranges must be set again.
    watch_dirty: bool,
    /// Per map slot: built count and the flags of two update banks (GPU
    /// written, host read and cleared).
    status: Buffer,
    /// A bit per map slot: BLASes built in full by this update.
    full: Buffer,
    /// Updates whose flag banks are not read yet: (serial, tick).
    unread: std::collections::VecDeque<(u64, u64)>,
    /// Per map slot: the flags of the latest updates read (the next refits
    /// follow them until newer ones are read).
    slot_flags: Vec<u32>,
    serial: u64,
    log: bool,
    dump_dir: Option<std::path::PathBuf>,
    dump_at: Option<f64>,
    dumped: Vec<u64>,
    started: std::time::Instant,
    stats: Stats,
    rejected: u32,
    /// Box pruning is on (`RPS5E_HW_RT_PRUNE`): BLAS roots are reported
    /// too.
    prune: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Known {
    Probing,
    Top,
    Blas,
    /// Not a BVH of the format when probed (probed again after a while:
    /// the memory may hold one later).
    Rejected(std::time::Instant),
}

#[derive(Default, Debug)]
struct Stats {
    updates: u64,
    builds: u64,
    refits: u64,
    /// BLASes an update neither built nor refit.
    unchanged: u64,
    /// BLASes the triangle kernel found changed, with a new entry count.
    changed: u64,
    rebuilt: u64,
    triangles: u64,
}

fn as_usage() -> vk::BufferUsageFlags {
    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
}

fn input_usage() -> vk::BufferUsageFlags {
    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER
}

fn host_usage() -> vk::BufferUsageFlags {
    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::TRANSFER_DST
}

/// The two geometries of a host BLAS of `entries` leaf entries: the opaque
/// triangles, then the others (each entry is active in one of them).
fn triangles_geometry(vertices: u64, entries: u32) -> [vk::AccelerationStructureGeometryKHR<'static>; 2] {
    let one = |at: u64, flags: vk::GeometryFlagsKHR| {
        let tri = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: at })
            .vertex_stride(12)
            .max_vertex(entries * 3 - 1)
            .index_type(vk::IndexType::NONE_KHR);
        vk::AccelerationStructureGeometryKHR::default().geometry_type(vk::GeometryTypeKHR::TRIANGLES).flags(flags).geometry(vk::AccelerationStructureGeometryDataKHR { triangles: tri })
    };
    [one(vertices, vk::GeometryFlagsKHR::OPAQUE), one(vertices + entries as u64 * 36, vk::GeometryFlagsKHR::empty())]
}

fn instances_geometry(instances: u64) -> [vk::AccelerationStructureGeometryKHR<'static>; 1] {
    let data = vk::AccelerationStructureGeometryInstancesDataKHR::default().array_of_pointers(false).data(vk::DeviceOrHostAddressConstKHR { device_address: instances });
    [vk::AccelerationStructureGeometryKHR::default().geometry_type(vk::GeometryTypeKHR::INSTANCES).geometry(vk::AccelerationStructureGeometryDataKHR { instances: data })]
}

const BLAS_FLAGS: vk::BuildAccelerationStructureFlagsKHR = vk::BuildAccelerationStructureFlagsKHR::from_raw(
    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE.as_raw() | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE.as_raw(),
);

impl Mirror {
    pub fn new(dev: &Arc<GpuDevice>) -> Result<Mirror, vk::Result> {
        dev.accel.as_ref().ok_or(vk::Result::ERROR_EXTENSION_NOT_PRESENT)?;
        let mut asp = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        {
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut asp);
            // SAFETY: valid chain on the stack.
            unsafe { dev.instance.get_physical_device_properties2(dev.phys, &mut p2) };
        }
        let map = Buffer::new(dev, (MAP_SLOTS * 32) as u64, host_usage(), MemoryUse::Download)?;
        map.bytes_mut()[..MAP_SLOTS * 32].fill(0);
        map.flush(0, (MAP_SLOTS * 32) as u64);
        let rbytes = (16 + REQUEST_SLOTS * 16) as u64;
        let requests = Buffer::new(dev, rbytes, host_usage(), MemoryUse::Download)?;
        requests.bytes_mut()[..rbytes as usize].fill(0);
        requests.flush(0, rbytes);
        let used = Buffer::new(dev, (MAP_SLOTS * 4) as u64, host_usage(), MemoryUse::Download)?;
        used.bytes_mut()[..MAP_SLOTS * 4].fill(0);
        used.flush(0, (MAP_SLOTS * 4) as u64);
        let sbytes = (MAP_SLOTS * STATUS_WORDS * 4) as u64;
        let status = Buffer::new(dev, sbytes, host_usage(), MemoryUse::Download)?;
        status.bytes_mut()[..sbytes as usize].fill(0);
        status.flush(0, sbytes);
        let full = Buffer::new(dev, (MAP_SLOTS / 8) as u64, host_usage(), MemoryUse::Device)?;
        let starts = Buffer::new(dev, (MAP_SLOTS * 8) as u64, host_usage(), MemoryUse::Device)?;
        let proxy = Self::make_proxy(dev)?;
        let instances_k = Kernel::new(dev, super::kernels::INSTANCES, 2, "instances");
        let triangles_k = Kernel::new(dev, super::kernels::TRIANGLES, 1, "triangles");
        if instances_k.is_none() || triangles_k.is_none() {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        let dump_dir = std::env::var_os("RPS5E_HW_RT_DUMP").filter(|d| !d.is_empty()).map(std::path::PathBuf::from);
        log::info!("ps5gpu: hardware ray tracing for BVH queries (RPS5E_HW_RT={})", super::level());
        Ok(Mirror {
            dev: dev.clone(),
            scratch_align: (asp.min_acceleration_structure_scratch_offset_alignment as u64).max(256),
            instances_k,
            triangles_k,
            gather_k: None,
            seen: 0,
            known: HashMap::new(),
            probes: Vec::new(),
            tops_pending: HashMap::new(),
            tops: (0..table::ENTRY_COUNT).map(|_| None).collect(),
            blas: HashMap::new(),
            map,
            starts,
            requests,
            proxy,
            used,
            requests_seen: 0,
            update_tick: 0,
            scratch: None,
            dropped: Vec::new(),
            jobs: None,
            jobs_dirty: true,
            placed: HashMap::new(),
            stamp: None,
            last_update: std::time::Instant::now(),
            watch_dirty: true,
            status,
            full,
            unread: std::collections::VecDeque::new(),
            slot_flags: vec![CHANGED | REBUILD; MAP_SLOTS],
            serial: 0,
            log: super::level() >= 2,
            dump_dir,
            dump_at: std::env::var("RPS5E_HW_RT_DUMP_AT").ok().and_then(|v| v.parse().ok()),
            dumped: Vec::new(),
            started: std::time::Instant::now(),
            stats: Stats::default(),
            rejected: 0,
            prune: crate::shader::EmitOptions::for_device(dev).hw_rt_prune,
        })
    }

    /// The unit-box BLAS: storage, AABB data, structure, address, build
    /// scratch size, built.
    #[allow(clippy::type_complexity)]
    fn make_proxy(dev: &Arc<GpuDevice>) -> Result<(Buffer, Buffer, vk::AccelerationStructureKHR, u64, u64, bool), vk::Result> {
        let a = dev.accel.as_ref().ok_or(vk::Result::ERROR_EXTENSION_NOT_PRESENT)?;
        let aabb = Buffer::new(dev, 32, input_usage(), MemoryUse::Upload)?;
        let words: [f32; 6] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        aabb.bytes_mut()[..24].copy_from_slice(&bytes);
        aabb.flush(0, 32);
        let geometry = Self::proxy_geometry(aabb.address);
        let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometry);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: valid build info.
        unsafe { a.get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &info, &[1], &mut sizes) };
        let storage = Buffer::new(dev, sizes.acceleration_structure_size, as_usage(), MemoryUse::Device)?;
        let ci = vk::AccelerationStructureCreateInfoKHR::default().buffer(storage.buffer).size(sizes.acceleration_structure_size).ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
        // SAFETY: valid create info on a live buffer.
        let accel = unsafe { a.create_acceleration_structure(&ci, None) }?;
        // SAFETY: live acceleration structure.
        let address = unsafe { a.get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(accel)) };
        Ok((storage, aabb, accel, address, sizes.build_scratch_size, false))
    }

    fn proxy_geometry(aabb: u64) -> [vk::AccelerationStructureGeometryKHR<'static>; 1] {
        let data = vk::AccelerationStructureGeometryAabbsDataKHR::default().data(vk::DeviceOrHostAddressConstKHR { device_address: aabb }).stride(24);
        [vk::AccelerationStructureGeometryKHR::default().geometry_type(vk::GeometryTypeKHR::AABBS).geometry(vk::AccelerationStructureGeometryDataKHR { aabbs: data })]
    }

    fn accel(&self) -> &ash::khr::acceleration_structure::Device {
        self.dev.accel.as_ref().expect("acceleration structures")
    }

    fn word(fault: &Buffer, i: u32) -> u32 {
        let b = fault.bytes_mut();
        let i = (table::BASE + i) as usize * 4;
        u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
    }

    /// New guest BVH bases from the feedback ring (guest byte addresses).
    fn take_feedback(&mut self, fault: &Buffer) -> Vec<u64> {
        fault.invalidate(table::BASE as u64 * 4, table::BYTES);
        let n = Self::word(fault, table::FEEDBACK_COUNT);
        let mut out = Vec::new();
        if n == self.seen {
            return out;
        }
        let first = if n.wrapping_sub(self.seen) > table::FEEDBACK_SLOTS { n.wrapping_sub(table::FEEDBACK_SLOTS) } else { self.seen };
        let mut k = first;
        while k != n {
            let slot = k % table::FEEDBACK_SLOTS;
            let lo = Self::word(fault, table::FEEDBACK + 2 * slot);
            let hi = Self::word(fault, table::FEEDBACK + 2 * slot + 1);
            if hi & 1 << 31 != 0 {
                let base = ((hi as u64 & 0xFF) << 32 | lo as u64) << 8;
                if !out.contains(&base) {
                    out.push(base);
                }
            }
            k = k.wrapping_add(1);
        }
        self.seen = n;
        out
    }

    /// Before a dispatch with BVH queries is recorded (its barriers are
    /// recorded, its pipeline not bound yet): bring the mirrors up to
    /// date. `page_table` is the device-address page table its stage reads
    /// guest memory through; `fault` holds the ray-tracing table.
    pub fn before_dispatch(&mut self, ctx: &mut crate::backend::DrawContext, program: u64, page_table: vk::Buffer, fault: &Buffer) {
        if page_table == vk::Buffer::null() {
            return;
        }
        let fresh = self.take_feedback(fault);
        if self.dump_dir.is_some() && (!fresh.is_empty() || self.dump_at.is_some()) {
            self.dump_reported(ctx, program, page_table, &fresh);
        }
        for base in fresh {
            if let Some(&first) = self.tops_pending.get(&base) {
                // Reported again after a filter clear while no mirrored
                // TLAS uses it: a BLAS traced as the top level.
                if self.serial >= first + TOP_DELAY && !self.blas.contains_key(&base) {
                    self.tops_pending.remove(&base);
                    self.register_top(base, 1, true, 0);
                }
            } else if self.known.get(&base).is_none_or(|k| matches!(k, Known::Rejected(at) if at.elapsed().as_secs() >= 2)) && self.probes.len() < 16 {
                self.probe(ctx, page_table, base);
            }
        }
        self.finish_probes(ctx);
        self.take_requests(ctx);
        self.promote_tops();
        self.note_use(ctx, fault);
        if self.tops.iter().all(Option::is_none) && self.stamp.is_some() {
            return;
        }
        if self.watch_dirty {
            self.watch_dirty = false;
            let ranges = self.tops.iter().flatten().map(|t| (t.guest, t.guest + t.bytes.max(256))).chain(self.blas.iter().map(|(g, b)| (*g, *g + b.bytes.max(256)))).collect();
            crate::rt::watch::set(ranges);
        }
        // Once per write of the guest BVHs (the console's refits and
        // builds), or after a while.
        let generation = crate::rt::watch::generation();
        if self.stamp == Some(generation) && self.last_update.elapsed() < REFRESH {
            return;
        }
        self.update(ctx, page_table, fault);
        self.stamp = Some(generation);
        self.last_update = std::time::Instant::now();
    }

    fn probe(&mut self, ctx: &mut crate::backend::DrawContext, page_table: vk::Buffer, base: u64) {
        if self.gather_k.is_none() {
            self.gather_k = Kernel::new(&self.dev, super::kernels::GATHER, 1, "gather");
        }
        let Some(k) = self.gather_k.as_ref() else { return };
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let Ok(out) = Buffer::new(&self.dev, 256, usage, MemoryUse::Download) else { return };
        let (bits, limit) = paging();
        let push = Push::default().u64(base).u64(out.address).u32(64).u32(bits).u32(limit);
        let cb = ctx.sched.ordered_cmd();
        let pt = vk::DescriptorBufferInfo { buffer: page_table, offset: 0, range: vk::WHOLE_SIZE };
        k.record(cb, &[pt], &push.0, [1, 1, 1]);
        ctx.sched.memory_barrier((vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_WRITE), (vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_READ));
        self.known.insert(base, Known::Probing);
        self.probes.push(Probe { base, tick: ctx.sched.current_tick(), out });
    }

    fn finish_probes(&mut self, ctx: &mut crate::backend::DrawContext) {
        let mut i = 0;
        while i < self.probes.len() {
            if !ctx.sched.is_complete(self.probes[i].tick) {
                i += 1;
                continue;
            }
            let p = self.probes.swap_remove(i);
            p.out.invalidate(0, 256);
            let h = p.out.bytes_mut();
            let w = |i: usize| u32::from_le_bytes(h[i * 4..i * 4 + 4].try_into().unwrap());
            let (magic, version, array, count, bytes) = ((w(0), w(1)), w(63), w(12), w(28), w(4) as u64);
            if magic != (0x5F52_5350, 0x4C48_5642) || version != 9 {
                if self.log || self.rejected < 8 {
                    self.rejected += 1;
                    log::info!("[hw-rt] {:#x}: not a BVH the mirror knows (magic {:08x} {:08x}, version {version})", p.base, magic.0, magic.1);
                }
                self.known.insert(p.base, Known::Rejected(std::time::Instant::now()));
            } else if array != 0 {
                self.known.remove(&p.base);
                self.register_top(p.base, count, false, bytes);
            } else if !self.prune {
                // Only traversals report their top level (no box pruning
                // reports BLAS roots): a bottom level traced on its own.
                self.known.remove(&p.base);
                self.register_top(p.base, 1, true, bytes);
            } else {
                self.known.remove(&p.base);
                self.tops_pending.insert(p.base, self.serial);
            }
            ctx.sched.retire(p.out);
        }
    }

    /// Entries and BLASes the GPU used since the last update stay; those
    /// unused for [`IDLE`] are freed.
    fn note_use(&mut self, ctx: &mut crate::backend::DrawContext, fault: &Buffer) {
        let now = std::time::Instant::now();
        fault.invalidate(table::BASE as u64 * 4, table::BYTES);
        for i in 0..self.tops.len() {
            let w = table::BASE + table::ENTRIES + i as u32 * table::ENTRY_WORDS + table::WORD_USED;
            let used = Self::word(fault, w - table::BASE) != 0;
            if used {
                fault.bytes_mut()[w as usize * 4..w as usize * 4 + 4].fill(0);
                fault.flush(w as u64 * 4, 4);
            }
            let Some(t) = self.tops[i].as_mut() else { continue };
            if used {
                t.used = now;
            } else if now - t.used > IDLE {
                let t = self.tops[i].take().unwrap();
                log::info!("[hw-rt] {:#x}: entry {i} freed (unused)", t.guest);
                self.known.remove(&t.guest);
                self.dropped.push(t);
                // The table entry: written as unknown at the next update.
                self.stamp = None;
                self.watch_dirty = true;
            }
        }
        // The flags are cleared here, in host memory: a later GPU write
        // sets them again (the race only delays a free).
        self.used.invalidate(0, (MAP_SLOTS * 4) as u64);
        let m = self.used.bytes_mut();
        let mut idle = Vec::new();
        for (guest, b) in self.blas.iter_mut() {
            let used = self.placed.get(guest).is_some_and(|&s| {
                let i = s as usize * 4;
                let u = m[i] != 0;
                m[i] = 0;
                u
            });
            if used {
                b.used = now;
            } else if now - b.used > IDLE {
                idle.push(*guest);
            }
        }
        self.used.flush(0, (MAP_SLOTS * 4) as u64);
        for guest in idle {
            if let Some(b) = self.blas.remove(&guest) {
                self.retire_blas(ctx, b);
                self.jobs_dirty = true;
                self.stamp = None;
                self.watch_dirty = true;
            }
        }
        for t in std::mem::take(&mut self.dropped) {
            self.retire_top(ctx, t);
        }
    }

    /// Pending roots a mirrored TLAS uses are its BLASes.
    fn promote_tops(&mut self) {
        let blas: Vec<u64> = self.tops_pending.keys().filter(|b| self.blas.contains_key(b)).copied().collect();
        for base in blas {
            self.tops_pending.remove(&base);
            self.known.insert(base, Known::Blas);
        }
    }

    fn register_top(&mut self, base: u64, count: u32, single: bool, bytes: u64) {
        let free = self.tops.iter().position(Option::is_none);
        // Else the entry unused the longest, when idle for a second.
        let lru = || self.tops.iter().enumerate().filter_map(|(i, t)| t.as_ref().map(|t| (i, t.used))).filter(|(_, u)| u.elapsed().as_secs() >= 1).min_by_key(|(_, u)| *u).map(|(i, _)| i);
        let Some(slot) = free.or_else(lru) else {
            log::warn!("[hw-rt] {base:#x}: every table entry is in use; its traversals keep the guest's for now");
            return;
        };
        if let Some(old) = self.tops[slot].take() {
            log::info!("[hw-rt] {:#x}: entry {slot} freed for {base:#x}", old.guest);
            self.known.remove(&old.guest);
            self.dropped.push(old);
        }
        let capacity = (count.max(16) * 2).next_power_of_two();
        match self.make_top(base, capacity, single, bytes) {
            Ok(t) => {
                log::info!("[hw-rt] {base:#x}: mirrored in entry {slot} ({count} instances, {capacity} slots)");
                self.tops[slot] = Some(t);
                self.known.insert(base, Known::Top);
                self.stamp = None;
                self.watch_dirty = true;
            }
            Err(e) => {
                log::warn!("[hw-rt] {base:#x}: host TLAS: {e:?}");
                self.known.insert(base, Known::Rejected(std::time::Instant::now()));
            }
        }
    }

    fn make_top(&self, guest: u64, capacity: u32, single: bool, bytes: u64) -> Result<Top, vk::Result> {
        let instances = Buffer::new(&self.dev, capacity as u64 * 64, input_usage(), MemoryUse::Device)?;
        let set = Buffer::new(&self.dev, (SET_SLOTS * 8) as u64, host_usage(), MemoryUse::Device)?;
        let geometry = instances_geometry(0);
        let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometry);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: valid build info.
        unsafe { self.accel().get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &info, &[capacity], &mut sizes) };
        let storage = Buffer::new(&self.dev, sizes.acceleration_structure_size, as_usage(), MemoryUse::Device)?;
        let ci = vk::AccelerationStructureCreateInfoKHR::default().buffer(storage.buffer).size(sizes.acceleration_structure_size).ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);
        // SAFETY: valid create info on a live buffer.
        let accel = unsafe { self.accel().create_acceleration_structure(&ci, None) }?;
        // SAFETY: live acceleration structure.
        let address = unsafe { self.accel().get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(accel)) };
        Ok(Top { guest, single, bytes, capacity, instances, set, storage, accel, address, scratch: sizes.build_scratch_size, used: std::time::Instant::now() })
    }

    fn make_blas(&self, entries: u32, bytes: u64) -> Result<HostBlas, vk::Result> {
        // Two geometries' triangles, then the list starts (a word per entry).
        let vertices = Buffer::new(&self.dev, entries as u64 * 76, input_usage(), MemoryUse::Device)?;
        let geometry = triangles_geometry(vertices.address, entries);
        let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(BLAS_FLAGS)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometry);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: valid build info.
        unsafe { self.accel().get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &info, &[entries, entries], &mut sizes) };
        let storage = Buffer::new(&self.dev, sizes.acceleration_structure_size, as_usage(), MemoryUse::Device)?;
        let ci = vk::AccelerationStructureCreateInfoKHR::default().buffer(storage.buffer).size(sizes.acceleration_structure_size).ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
        // SAFETY: valid create info on a live buffer.
        let accel = unsafe { self.accel().create_acceleration_structure(&ci, None) }?;
        // SAFETY: live acceleration structure.
        let address = unsafe { self.accel().get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(accel)) };
        // Unwritten triangles are inactive (NaN) until the triangle kernel
        // fills them.
        Ok(HostBlas { entries, bytes, slot: None, vertices, storage, accel, address, build_scratch: sizes.build_scratch_size, update_scratch: sizes.update_scratch_size, built: false, refits: 0, used: std::time::Instant::now() })
    }

    /// Requests of the last update (complete by now): BLASes to mirror,
    /// TLASes to grow.
    fn take_requests(&mut self, ctx: &mut crate::backend::DrawContext) {
        if self.update_tick == 0 || !ctx.sched.is_complete(self.update_tick) {
            return;
        }
        let rbytes = (16 + REQUEST_SLOTS * 16) as u64;
        self.requests.invalidate(0, rbytes);
        let b = self.requests.bytes_mut();
        let n = u32::from_le_bytes(b[0..4].try_into().unwrap());
        if n == self.requests_seen {
            return;
        }
        let q = |i: usize| u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
        let first = if n.wrapping_sub(self.requests_seen) as usize > REQUEST_SLOTS { n.wrapping_sub(REQUEST_SLOTS as u32) } else { self.requests_seen };
        let mut wanted: HashMap<u64, (u32, u64)> = HashMap::new();
        let mut grow: Vec<(usize, u32)> = Vec::new();
        let mut k = first;
        while k != n {
            let s = 1 + 2 * (k as usize % REQUEST_SLOTS);
            let (key, entries, bytes) = (q(s), q(s + 1) as u32, q(s + 1) >> 32);
            if key >> 63 != 0 {
                grow.push(((key >> 32 & 0xFF) as usize, key as u32));
            } else if key > 1 {
                wanted.insert(key - 1, (entries, bytes));
            }
            k = k.wrapping_add(1);
        }
        self.requests_seen = n;
        for (slot, count) in grow {
            let Some(t) = self.tops.get(slot).and_then(|t| t.as_ref()) else { continue };
            if count > t.capacity && count <= 1 << 20 {
                let (guest, capacity, single, bytes) = (t.guest, (count * 2).next_power_of_two(), t.single, t.bytes);
                if let Ok(new) = self.make_top(guest, capacity, single, bytes) {
                    log::info!("[hw-rt] {guest:#x}: {count} instances, entry {slot} grows to {capacity} slots");
                    if let Some(old) = self.tops[slot].replace(new) {
                        self.retire_top(ctx, old);
                    }
                    self.stamp = None;
                    self.watch_dirty = true;
                }
            }
        }
        for (guest, (entries, bytes)) in wanted {
            if entries == 0 || entries > 1 << 22 {
                continue;
            }
            if self.blas.get(&guest).is_some_and(|b| b.entries >= entries) {
                continue;
            }
            // Room for a few more entries: a rebuilt BLAS of about the same
            // size keeps its host BLAS.
            let capacity = (entries + entries / 8 + 63) & !63;
            match self.make_blas(capacity, bytes) {
                Ok(h) => {
                    if self.log {
                        log::info!("[hw-rt] BLAS {guest:#x}: {entries} triangles mirrored");
                    }
                    if let Some(old) = self.blas.insert(guest, h) {
                        self.retire_blas(ctx, old);
                    }
                    self.jobs_dirty = true;
                    self.stamp = None;
                    self.watch_dirty = true;
                    self.tops_pending.remove(&guest);
                }
                Err(e) => log::warn!("[hw-rt] BLAS {guest:#x} ({entries} triangles): {e:?}"),
            }
        }
    }

    fn retire_blas(&self, ctx: &mut crate::backend::DrawContext, b: HostBlas) {
        ctx.sched.retire(AccelGuard { dev: self.dev.clone(), accel: b.accel, _buffers: vec![b.vertices, b.storage] });
    }

    fn retire_top(&self, ctx: &mut crate::backend::DrawContext, t: Top) {
        ctx.sched.retire(AccelGuard { dev: self.dev.clone(), accel: t.accel, _buffers: vec![t.instances, t.set, t.storage] });
    }

    /// The device address of a scratch area of at least `size` bytes.
    fn scratch(&mut self, ctx: &mut crate::backend::DrawContext, size: u64) -> Option<u64> {
        if self.scratch.as_ref().is_none_or(|s| s.size < size) {
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
            let b = Buffer::new(&self.dev, size.next_power_of_two(), usage, MemoryUse::Device).ok()?;
            if let Some(old) = self.scratch.replace(b) {
                ctx.sched.retire(old);
            }
        }
        self.scratch.as_ref().map(|s| s.address.next_multiple_of(self.scratch_align))
    }

    /// The BLAS map, written in the command stream (every update: the
    /// triangle kernel clears the slots of BLASes it cannot mirror), and
    /// the jobs and chunks of the triangle kernel when the BLAS set changed.
    fn write_map(&mut self, ctx: &mut crate::backend::DrawContext) {
        let mut slots = vec![[0u64; 4]; MAP_SLOTS];
        let mut starts = vec![0u64; MAP_SLOTS];
        let mut placed: Vec<(u64, u32)> = Vec::new();
        let mut list: Vec<(&u64, &HostBlas)> = self.blas.iter().collect();
        list.sort_by_key(|(g, _)| **g);
        for (&guest, b) in list {
            let key = guest + 1;
            let h = (guest as u32).wrapping_mul(0x9E37_79B1) >> 12;
            let Some(s) = (0..32).map(|k| (h as usize + k) & (MAP_SLOTS - 1)).find(|&s| slots[s][0] == 0) else { continue };
            slots[s] = [key, b.address, b.entries as u64, 0];
            starts[s] = b.vertices.address + b.entries as u64 * 72;
            placed.push((guest, s as u32));
        }
        self.placed = placed.iter().copied().collect();
        let bytes: Vec<u8> = slots.iter().flatten().flat_map(|q| q.to_le_bytes()).collect();
        let cb = ctx.sched.ordered_cmd();
        for (i, chunk) in bytes.chunks(65536).enumerate() {
            // SAFETY: recording buffer outside a rendering scope; in range.
            unsafe { self.dev.device.cmd_update_buffer(cb, self.map.buffer, (i * 65536) as u64, chunk) };
        }
        let bytes: Vec<u8> = starts.iter().flat_map(|q| q.to_le_bytes()).collect();
        // SAFETY: recording buffer outside a rendering scope; in range.
        unsafe { self.dev.device.cmd_update_buffer(cb, self.starts.buffer, 0, &bytes) };
        if !self.jobs_dirty {
            return;
        }
        // Jobs {guest, vertices, entries, map slot, 0} and chunks {job, first}.
        let mut jobs: Vec<u8> = Vec::new();
        let mut chunks: Vec<u8> = Vec::new();
        for (j, (guest, slot)) in placed.iter().enumerate() {
            let b = &self.blas[guest];
            jobs.extend_from_slice(&guest.to_le_bytes());
            jobs.extend_from_slice(&b.vertices.address.to_le_bytes());
            jobs.extend_from_slice(&b.entries.to_le_bytes());
            jobs.extend_from_slice(&slot.to_le_bytes());
            jobs.extend_from_slice(&[0u8; 8]);
            for first in (0..b.entries).step_by(64) {
                chunks.extend_from_slice(&(j as u32).to_le_bytes());
                chunks.extend_from_slice(&first.to_le_bytes());
            }
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let upload = |data: &[u8]| -> Option<Buffer> {
            let n = data.len().max(16) as u64;
            let b = Buffer::new(&self.dev, n, usage, MemoryUse::Upload).ok()?;
            b.bytes_mut()[..data.len()].copy_from_slice(data);
            b.flush(0, n);
            Some(b)
        };
        let n = (chunks.len() / 8) as u32;
        if let (Some(jb), Some(cb)) = (upload(&jobs), upload(&chunks)) {
            if let Some((a, b, _)) = self.jobs.replace((jb, cb, n)) {
                ctx.sched.retire(a);
                ctx.sched.retire(b);
            }
            self.jobs_dirty = false;
        }
    }

    fn update(&mut self, ctx: &mut crate::backend::DrawContext, page_table: vk::Buffer, fault: &Buffer) {
        self.serial += 1;
        self.stats.updates += 1;
        let timed = ctx.sched.time_item_begin(crate::scheduler::gpu_timer::label::HW_RT);
        let (bits, limit) = paging();
        let pt = vk::DescriptorBufferInfo { buffer: page_table, offset: 0, range: vk::WHOLE_SIZE };
        let table_info = vk::DescriptorBufferInfo { buffer: fault.buffer, offset: 0, range: vk::WHOLE_SIZE };
        let compute = (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_WRITE);
        let build = (vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR, vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags2::SHADER_READ);
        let built = (vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR, vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR);
        let transfer = (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let readers = (
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
        );
        // The BLAS map, the table entries and their BLAS sets.
        self.write_map(ctx);
        // What the last update's triangle kernel found (its flags bank, read
        // once that update completed; unknown otherwise: every BLAS counts
        // as changed): a BLAS is built in full when new, after its refits,
        // moved in the map or when its entry count changed; refit when a
        // triangle changed; left alone otherwise.
        let bank = (self.serial % BANKS) as u32;
        let mut fresh: Option<Vec<u32>> = None;
        while let Some(&(serial, tick)) = self.unread.front() {
            if !ctx.sched.is_complete(tick) {
                break;
            }
            self.unread.pop_front();
            let n = MAP_SLOTS * STATUS_WORDS;
            self.status.invalidate(0, (n * 4) as u64);
            let b = self.status.bytes_mut();
            let at = |s: usize| (s * STATUS_WORDS + 1 + (serial % BANKS) as usize) * 4;
            let f = fresh.get_or_insert_with(|| vec![0; MAP_SLOTS]);
            for (s, x) in f.iter_mut().enumerate() {
                *x |= u32::from_le_bytes(b[at(s)..at(s) + 4].try_into().unwrap());
                // Written again by update serial + BANKS (recorded later).
                b[at(s)..at(s) + 4].fill(0);
            }
            self.status.flush(0, (n * 4) as u64);
        }
        if let Some(f) = fresh {
            self.slot_flags = f;
        }
        let flags = Some(self.slot_flags.clone());
        if let Some(f) = &flags {
            self.stats.changed += self.placed.values().filter(|&&s| f[s as usize] & CHANGED != 0).count() as u64;
            self.stats.rebuilt += self.placed.values().filter(|&&s| f[s as usize] & REBUILD != 0).count() as u64;
        }
        let mut full_bits = vec![0u32; MAP_SLOTS / 32];
        let mut plan: Vec<(u64, u32, bool)> = Vec::new();
        for (guest, b) in self.blas.iter() {
            let Some(&slot) = self.placed.get(guest) else { continue };
            let f = flags.as_ref().map(|f| f[slot as usize]);
            let changed = f.is_none_or(|f| f & CHANGED != 0);
            let full = !b.built || b.slot != Some(slot) || f.is_some_and(|f| f & REBUILD != 0) || (changed && b.refits >= refits());
            let refit = !full && changed;
            if full {
                full_bits[slot as usize / 32] |= 1 << (slot % 32);
            }
            if full || refit {
                plan.push((*guest, slot, full));
            }
        }
        let cb = ctx.sched.ordered_cmd();
        let full_bytes: Vec<u8> = full_bits.iter().flat_map(|w| w.to_le_bytes()).collect();
        // SAFETY: recording buffer outside a rendering scope; in range.
        unsafe { self.dev.device.cmd_update_buffer(cb, self.full.buffer, 0, &full_bytes) };
        for (i, t) in self.tops.iter().enumerate() {
            let at = (table::BASE + table::ENTRIES + i as u32 * table::ENTRY_WORDS) as u64 * 4;
            let Some(t) = t else {
                // SAFETY: recording buffer outside a rendering scope; in range.
                unsafe { self.dev.device.cmd_fill_buffer(cb, fault.buffer, at, table::WORD_USED as u64 * 4, 0) };
                continue;
            };
            let base = t.guest >> 8;
            let words = [
                base as u32,
                (base >> 32) as u32 & 0xFF | table::KNOWN | table::TLAS | table::VALID | if host_traversals() { table::COMPLETE } else { 0 } | if t.single { table::SINGLE } else { 0 },
                t.address as u32,
                (t.address >> 32) as u32,
                t.set.address as u32,
                (t.set.address >> 32) as u32,
                SET_SLOTS as u32 - 1,
            ];
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            // SAFETY: recording buffer outside a rendering scope; in range.
            unsafe {
                self.dev.device.cmd_update_buffer(cb, fault.buffer, at, &bytes);
                self.dev.device.cmd_fill_buffer(cb, t.set.buffer, 0, vk::WHOLE_SIZE, 0);
            }
        }
        {
            let at = (table::BASE + table::STARTS) as u64 * 4;
            let a = self.starts.address;
            let bytes: Vec<u8> = [a as u32, (a >> 32) as u32].iter().flat_map(|w| w.to_le_bytes()).collect();
            // SAFETY: recording buffer outside a rendering scope; in range.
            unsafe { self.dev.device.cmd_update_buffer(cb, fault.buffer, at, &bytes) };
        }
        if self.serial % FILTER_PERIOD == 0 {
            // Roots reported before may be reported again.
            let at = (table::BASE + table::FILTER) as u64 * 4;
            // SAFETY: recording buffer outside a rendering scope; in range.
            unsafe { self.dev.device.cmd_fill_buffer(cb, fault.buffer, at, table::FILTER_WORDS as u64 * 4, 0) };
        }
        ctx.sched.memory_barrier(transfer, readers);
        // Triangles of every mirrored BLAS (each compared with the last
        // update's: the flags of this update's bank say what changed).
        let mut blas_count = 0;
        if let (Some((jobs, chunks, n)), Some(k)) = (&self.jobs, &self.triangles_k) {
            let push = Push::default()
                .u64(chunks.address)
                .u64(jobs.address)
                .u64(self.map.address)
                .u64(self.status.address)
                .u64(self.full.address)
                .u32(bits)
                .u32(limit)
                .u32(bank);
            let cb = ctx.sched.ordered_cmd();
            k.record(cb, &[pt], &push.0, [*n, 1, 1]);
            blas_count = plan.len();
        }
        ctx.sched.memory_barrier(compute, build);
        if !self.proxy.5 {
            let align = self.scratch_align;
            if let Some(scratch) = self.scratch(ctx, self.proxy.4 + align) {
                let geometry = Self::proxy_geometry(self.proxy.1.address);
                let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .dst_acceleration_structure(self.proxy.2)
                    .geometries(&geometry)
                    .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch });
                let range = [vk::AccelerationStructureBuildRangeInfoKHR { primitive_count: 1, primitive_offset: 0, first_vertex: 0, transform_offset: 0 }];
                let cb = ctx.sched.ordered_cmd();
                // SAFETY: recording buffer; every structure lives through the call.
                unsafe { self.accel().cmd_build_acceleration_structures(cb, &[info], &[&range[..]]) };
                ctx.sched.memory_barrier(built, build);
                self.proxy.5 = true;
            }
        }
        // Host BLAS builds and refits of the BLASes that need them, one
        // command.
        if blas_count > 0 {
            let align = self.scratch_align;
            let mut total = 0u64;
            let mut steps: Vec<(vk::AccelerationStructureKHR, u64, u32, bool, u64)> = Vec::with_capacity(blas_count);
            for &(guest, _, full) in &plan {
                let b = &self.blas[&guest];
                steps.push((b.accel, b.vertices.address, b.entries, !full, total));
                total += if full { b.build_scratch } else { b.update_scratch }.next_multiple_of(align);
                self.stats.triangles += b.entries as u64;
            }
            if let Some(scratch) = self.scratch(ctx, total + align) {
                let geometries: Vec<[vk::AccelerationStructureGeometryKHR; 2]> = steps.iter().map(|p| triangles_geometry(p.1, p.2)).collect();
                let infos: Vec<vk::AccelerationStructureBuildGeometryInfoKHR> = steps
                    .iter()
                    .zip(&geometries)
                    .map(|(p, g)| {
                        let i = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                            .flags(BLAS_FLAGS)
                            .mode(if p.3 { vk::BuildAccelerationStructureModeKHR::UPDATE } else { vk::BuildAccelerationStructureModeKHR::BUILD })
                            .dst_acceleration_structure(p.0)
                            .geometries(g)
                            .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch + p.4 });
                        if p.3 {
                            i.src_acceleration_structure(p.0)
                        } else {
                            i
                        }
                    })
                    .collect();
                let range = |n: u32| vk::AccelerationStructureBuildRangeInfoKHR { primitive_count: n, primitive_offset: 0, first_vertex: 0, transform_offset: 0 };
                let ranges: Vec<[vk::AccelerationStructureBuildRangeInfoKHR; 2]> = steps.iter().map(|p| [range(p.2), range(p.2)]).collect();
                let range_refs: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> = ranges.iter().map(|r| &r[..]).collect();
                let cb = ctx.sched.ordered_cmd();
                // SAFETY: recording buffer; every structure lives through the call.
                unsafe { self.accel().cmd_build_acceleration_structures(cb, &infos, &range_refs) };
                for &(guest, slot, full) in &plan {
                    let Some(b) = self.blas.get_mut(&guest) else { continue };
                    if full {
                        b.built = true;
                        b.refits = 0;
                        b.slot = Some(slot);
                        self.stats.builds += 1;
                    } else {
                        b.refits += 1;
                        self.stats.refits += 1;
                    }
                }
            }
        }
        self.stats.unchanged += (self.blas.len() - plan.len()) as u64;
        ctx.sched.memory_barrier(built, readers);
        // Instances and TLAS of every entry.
        let mut tlas: Vec<(vk::AccelerationStructureKHR, u64, u32, u64)> = Vec::new();
        for (i, t) in self.tops.iter().enumerate() {
            let Some(t) = t else { continue };
            let flags_word = table::BASE + table::ENTRIES + i as u32 * table::ENTRY_WORDS + table::WORD_FLAGS;
            let push = Push::default()
                .u64(t.guest)
                .u64(t.instances.address)
                .u64(self.map.address)
                .u64(t.set.address)
                .u64(self.requests.address)
                .u64(self.used.address)
                .u64(self.proxy.3)
                .u32(t.capacity)
                .u32(MAP_SLOTS as u32 - 1)
                .u32(SET_SLOTS as u32 - 1)
                .u32(REQUEST_SLOTS as u32 - 1)
                .u32(flags_word)
                .u32(i as u32)
                .u32(bits)
                .u32(limit)
                .u32(table::BASE + table::STATS);
            let cb = ctx.sched.ordered_cmd();
            if let Some(k) = &self.instances_k {
                k.record(cb, &[pt, table_info], &push.0, [t.capacity.div_ceil(64), 1, 1]);
            }
            tlas.push((t.accel, t.instances.address, t.capacity, t.scratch));
        }
        ctx.sched.memory_barrier(compute, build);
        if !tlas.is_empty() {
            let align = self.scratch_align;
            let total: u64 = tlas.iter().map(|b| b.3.next_multiple_of(align)).sum();
            if let Some(scratch) = self.scratch(ctx, total + align) {
                let geometries: Vec<[vk::AccelerationStructureGeometryKHR; 1]> = tlas.iter().map(|t| instances_geometry(t.1)).collect();
                let mut off = 0u64;
                let infos: Vec<vk::AccelerationStructureBuildGeometryInfoKHR> = tlas
                    .iter()
                    .zip(&geometries)
                    .map(|(t, g)| {
                        let i = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                            .dst_acceleration_structure(t.0)
                            .geometries(g)
                            .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch + off });
                        off += t.3.next_multiple_of(align);
                        i
                    })
                    .collect();
                let ranges: Vec<[vk::AccelerationStructureBuildRangeInfoKHR; 1]> =
                    tlas.iter().map(|t| [vk::AccelerationStructureBuildRangeInfoKHR { primitive_count: t.2, primitive_offset: 0, first_vertex: 0, transform_offset: 0 }]).collect();
                let range_refs: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> = ranges.iter().map(|r| &r[..]).collect();
                let cb = ctx.sched.ordered_cmd();
                // SAFETY: recording buffer; every structure lives through the call.
                unsafe { self.accel().cmd_build_acceleration_structures(cb, &infos, &range_refs) };
            }
        }
        // The dispatch's ray queries read the TLASes; its BVH queries the
        // table words the instance kernels wrote.
        ctx.sched.memory_barrier(
            (vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags2::SHADER_WRITE),
            (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ),
        );
        ctx.sched.time_item_end(timed);
        self.update_tick = ctx.sched.current_tick();
        self.unread.push_back((self.serial, self.update_tick));
        while self.unread.len() as u64 > BANKS {
            // Its bank is written again before it was read: its flags merge
            // into the later update's.
            self.unread.pop_front();
        }
        if self.stats.updates % if self.log { 100 } else { 200 } == 1 {
            fault.invalidate(table::BASE as u64 * 4, table::BYTES);
            let counts: Vec<u32> = (0..table::STAT_WORDS).map(|i| Self::word(fault, table::STATS + i)).collect();
            let valid: Vec<String> = self
                .tops
                .iter()
                .enumerate()
                .filter_map(|(i, t)| t.as_ref().map(|t| (i, t)))
                .map(|(i, t)| {
                    let f = Self::word(fault, table::ENTRIES + i as u32 * table::ENTRY_WORDS + table::WORD_FLAGS);
                    format!("{:#x}:{}{}", t.guest, if f & table::VALID != 0 { "v" } else { "-" }, if f & table::COMPLETE != 0 { "c" } else { "-" })
                })
                .collect();
            if self.log {
                log::info!("[hw-rt] writes that moved the BVH generation (program, ranged, unranged): {:x?}", crate::rt::watch::take_triggers().iter().take(12).collect::<Vec<_>>());
            }
            log::info!(
                "[hw-rt] update {}: entries (v valid, c complete, as of the last update) {valid:?}, {} BLASes, {:?}; counters {counts:?}",
                self.serial,
                self.blas.len(),
                self.stats
            );
        }
    }

    /// Diagnostics: dump the BVHs reported now.
    fn dump_reported(&mut self, ctx: &mut crate::backend::DrawContext, program: u64, page_table: vk::Buffer, fresh: &[u64]) {
        if self.dump_at.is_some_and(|at| self.started.elapsed().as_secs_f64() < at) {
            return;
        }
        let mut list: Vec<u64> = fresh.to_vec();
        if self.dump_at.take().is_some() {
            // The mirrored ones as they are now.
            list.extend(self.tops.iter().flatten().map(|t| t.guest));
        }
        for base in list {
            if self.dumped.contains(&base) {
                continue;
            }
            self.dumped.push(base);
            log::info!("[hw-rt] program {program:#x}: BVH root {base:#x} reported");
            self.dump(ctx, base, page_table);
        }
    }

    /// Guest bytes `[addr, addr+size)` as the GPU sees them through
    /// `page_table` (waits for the GPU; diagnostics).
    fn gather(&mut self, ctx: &mut crate::backend::DrawContext, page_table: vk::Buffer, addr: u64, size: u64) -> Vec<u8> {
        if self.gather_k.is_none() {
            self.gather_k = Kernel::new(&self.dev, super::kernels::GATHER, 1, "gather");
        }
        let Some(k) = self.gather_k.as_ref() else { return Vec::new() };
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let Ok(out) = Buffer::new(&self.dev, size.max(4), usage, MemoryUse::Download) else { return Vec::new() };
        let count = (size / 4) as u32;
        let (bits, limit) = paging();
        let push = Push::default().u64(addr).u64(out.address).u32(count).u32(bits).u32(limit);
        let cb = ctx.sched.ordered_cmd();
        let pt = vk::DescriptorBufferInfo { buffer: page_table, offset: 0, range: vk::WHOLE_SIZE };
        k.record(cb, &[pt], &push.0, [count.div_ceil(64), 1, 1]);
        ctx.sched.memory_barrier((vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_WRITE), (vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_READ));
        ctx.sched.flush_and_wait();
        out.invalidate(0, size);
        let v = out.bytes_mut()[..size as usize].to_vec();
        ctx.sched.retire(out);
        v
    }

    /// Write the guest bytes of the BVH at `base` (and of the BLASes of its
    /// instance records when it is a TLAS) to the dump directory.
    fn dump(&mut self, ctx: &mut crate::backend::DrawContext, base: u64, page_table: vk::Buffer) {
        let Some(dir) = self.dump_dir.clone() else { return };
        let _ = std::fs::create_dir_all(&dir);
        let header = self.gather(ctx, page_table, base, 256);
        let w = |h: &[u8], i: usize| h.get(i * 4..i * 4 + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
        let size = (w(&header, 4) as u64).clamp(256, 64 << 20).next_multiple_of(4);
        let bytes = self.gather(ctx, page_table, base, size);
        let _ = std::fs::write(dir.join(format!("bvh-{base:x}.bin")), &bytes);
        // Instance records (TLAS): count at +112, array at +48, 128 bytes
        // each, BLAS address in bits 53:0 of the qword at +48 (8-byte units).
        let (inst_off, count) = (w(&header, 12) as usize, w(&header, 28) as usize);
        if inst_off == 0 {
            return;
        }
        let mut blases: Vec<u64> = Vec::new();
        for i in 0..count {
            let at = inst_off + i * 128;
            let Some(q) = bytes.get(at + 48..at + 56) else { break };
            let q = u64::from_le_bytes(q.try_into().unwrap());
            let blas = (q & ((1 << 54) - 1)) << 3;
            if blas != 0 && !blases.contains(&blas) {
                blases.push(blas);
            }
        }
        log::info!("[hw-rt] {base:#x}: {count} instances, {} BLASes", blases.len());
        for b in blases.into_iter().take(48) {
            let h = self.gather(ctx, page_table, b, 256);
            let size = (w(&h, 4) as u64).clamp(256, 64 << 20).next_multiple_of(4);
            let bytes = self.gather(ctx, page_table, b, size);
            let _ = std::fs::write(dir.join(format!("blas-{b:x}.bin")), &bytes);
        }
    }
}

impl Drop for Mirror {
    fn drop(&mut self) {
        // SAFETY: the provider waits for the device before dropping.
        unsafe {
            for b in self.blas.values() {
                self.accel().destroy_acceleration_structure(b.accel, None);
            }
            for t in self.tops.iter().flatten() {
                self.accel().destroy_acceleration_structure(t.accel, None);
            }
            self.accel().destroy_acceleration_structure(self.proxy.2, None);
        }
    }
}

/// An acceleration structure and its buffers, destroyed when the GPU is
/// done with them.
struct AccelGuard {
    dev: Arc<GpuDevice>,
    accel: vk::AccelerationStructureKHR,
    _buffers: Vec<Buffer>,
}

impl Drop for AccelGuard {
    fn drop(&mut self) {
        if let Some(a) = self.dev.accel.as_ref() {
            // SAFETY: retired after the last work that used it completed.
            unsafe { a.destroy_acceleration_structure(self.accel, None) };
        }
    }
}
