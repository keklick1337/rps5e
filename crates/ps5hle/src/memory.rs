// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest memory manager: direct (physical) memory, flexible memory, memory
//! pools and the guest virtual address map.
//!
//! Model:
//!   - One sparse host backing file (memfd) stands in for physical memory:
//!     `[0, direct_size)` is direct memory and the flexible pool follows it.
//!     Mapping memory creates a `MAP_SHARED` view of the backing, so the
//!     same physical range can be aliased at several virtual addresses.
//!   - `PhysicalMemory` tracks allocated direct blocks (direct, pool
//!     expansions, automatic AMM memory) and every virtual view of them.
//!   - `FlexibleMemory` hands out backing blocks of the flexible pool;
//!     `PooledMemory` hands out pool-expansion blocks to committed pool pages.
//!   - `VirtualRanges` is the guest-visible map: reserved, pool-reserved,
//!     direct, flexible, pooled, stack, code and runtime ranges with their
//!     protection, memory type, physical offset and name. Adjacent ranges
//!     with equal attributes merge (never direct ones).
//!   - Host side: every range this manager owns is held on the host
//!     (reservations as `PROT_NONE | MAP_NORESERVE`). Placement searches the
//!     guest regions in kernel order and skips ranges already used by the
//!     host or the loader.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

pub const PAGE_SIZE: u64 = 0x4000;
/// Physical memory a retail console gives the title (13.5 GiB of the 16 GiB
/// GDDR6); the flexible pool is carved out of it, the rest is direct memory.
pub const TOTAL_BYTES: u64 = 13824 * 1024 * 1024;
/// Physical memory the system keeps for itself, after the title's memory (the
/// other 2.5 GiB of the 16 GiB). The native AGC driver's memory lives here:
/// the title's direct memory is `[0, direct_size)` and stays whole.
pub const SYSTEM_BYTES: u64 = 4 * 1024 * 1024;
/// Size of the host backing file: title memory, then system memory.
const BACKING_BYTES: u64 = TOTAL_BYTES + SYSTEM_BYTES;
/// Flexible pool size unless the title's param.json sets
/// `kernel.flexibleMemorySize`.
pub const DEFAULT_FLEXIBLE_BYTES: u64 = 1024 * 1024 * 1024;

/// Page-table entries sceKernelGetPageTableStats reports for the CPU and
/// for the GPU (measured on a PS5, FW 13.60).
const PAGE_TABLE_POOL_ENTRIES: i32 = 8192;
const PAGE_TABLE_GRANULARITY: u64 = 2 * 1024 * 1024;
/// A GPU page directory entry covers 1 GiB.
const PAGE_TABLE_GPU_REGION: u64 = 1024 * 1024 * 1024;

/// Guest address regions.
const HOST_SYSTEM_MANAGED_MIN: u64 = 0x0004_0000;
const HOST_SYSTEM_MANAGED_MAX: u64 = 0x07_ffff_bfff;
const SYSTEM_RESERVED_MAX: u64 = 0x0f_ffff_ffff;
const HOST_USER_MIN: u64 = 0x10_0000_0000;
const HOST_USER_MAX: u64 = 0xfb_ffff_ffff;
/// Hint-less user mappings are placed at or above this.
const GUEST_DEFAULT_MAP_BASE: u64 = 0x2_0000_0000;
/// AMM automatic memory window.
pub const EXTENDED_MEMORY_BASE: u64 = 0x0800_0000_0000;
pub const EXTENDED_MEMORY_SIZE: u64 = 512 * 1024 * 1024 * 1024;

const OK: HleResult = 0;
const fn kerr(errno: u32) -> HleResult {
    (0x8002_0000u32 | errno) as i32 as i64
}
const ENOENT: HleResult = kerr(2);
const ENOMEM: HleResult = kerr(12);
const EACCES: HleResult = kerr(13);
const EFAULT: HleResult = kerr(14);
const EBUSY: HleResult = kerr(16);
const EINVAL: HleResult = kerr(22);
const EAGAIN: HleResult = kerr(35);
const ENAMETOOLONG: HleResult = kerr(63);

pub(crate) const MAP_FIXED: u64 = 0x10;
const MAP_NO_OVERWRITE: u64 = 0x80;
const MAP_DMEM_COMPAT: u64 = 0x400;
const MAP_UNKNOWN_8000: u64 = 0x8000;
const MAP_NO_COALESCE: u64 = 0x40_0000;
const MAP_ALIGNMENT_MASK: u64 = 0xff00_0000;

pub const PROT_CPU_READ: i32 = 0x01;
pub const PROT_CPU_WRITE: i32 = 0x02;
pub const PROT_CPU_EXEC: i32 = 0x04;
pub const PROT_GPU_READ: i32 = 0x10;
pub const PROT_GPU_WRITE: i32 = 0x20;

const NAME_LEN: usize = 32;
const VIRTUAL_QUERY_INFO_SIZE: u64 = 72;
const DIRECT_QUERY_INFO_SIZE: u64 = 24;

const PRT_APERTURE_MAX_INDEX: i32 = 2;
const PRT_PAGE_SIZE: u64 = 0x4000;
const PRT_APERTURE_START: u64 = 0x0f_0000_0000;
const PRT_APERTURE_END: u64 = 0xfc_0000_0000;

static FLEXIBLE_SIZE: AtomicU64 = AtomicU64::new(DEFAULT_FLEXIBLE_BYTES);
static FLEXIBLE_SIZE_FROZEN: AtomicBool = AtomicBool::new(false);

fn flexible_size() -> u64 {
    FLEXIBLE_SIZE.load(Ordering::Relaxed)
}

/// Direct memory size: everything that is not flexible memory.
pub fn direct_size() -> u64 {
    TOTAL_BYTES - flexible_size()
}

/// Set the flexible pool size before the memory manager starts (from the
/// title's param.json). Later calls are ignored.
pub fn set_flexible_memory_size(size: u64) -> bool {
    if size == 0 || size % PAGE_SIZE != 0 || size >= TOTAL_BYTES || FLEXIBLE_SIZE_FROZEN.load(Ordering::Acquire) {
        return false;
    }
    FLEXIBLE_SIZE.store(size, Ordering::Release);
    log::info!("[mem] flexible memory size = {size:#x} ({} MiB)", size >> 20);
    true
}

/// Range callbacks (address-sanitizer style shadow tracking): called for
/// every committed range mapped / unmapped.
/// Every committed range is also reported to the address sanitizer
/// (`crate::dbg_asan::kernel_alloc` / `kernel_free`, no-ops until it runs).
pub type RangeCallback = fn(u64, u64);
static CALLBACKS: Mutex<Option<(RangeCallback, RangeCallback)>> = Mutex::new(None);

fn alloc_callback(addr: u64, len: u64) {
    if let Some(vm) = crate::pthread::guest_vm() {
        crate::dbg_asan::kernel_alloc(vm, addr, len);
    }
    let cb = *CALLBACKS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((alloc, _)) = cb {
        alloc(addr, len);
    }
}

fn free_callback(addr: u64, len: u64) {
    if let Some(vm) = crate::pthread::guest_vm() {
        crate::dbg_asan::kernel_free(vm, addr, len);
    }
    let cb = *CALLBACKS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, free)) = cb {
        free(addr, len);
    }
}

/// Install the map/unmap callbacks; existing mappings are reported to
/// `alloc` right away.
pub fn register_callbacks(alloc: RangeCallback, free: RangeCallback) {
    let st = lock();
    *CALLBACKS.lock().unwrap_or_else(|e| e.into_inner()) = Some((alloc, free));
    for m in &st.physical.mappings {
        if m.map_vaddr != 0 && m.map_size != 0 {
            alloc(m.map_vaddr, m.map_size);
        }
    }
    for b in &st.flexible.blocks {
        alloc(b.map_vaddr, b.map_size);
    }
    for m in &st.pooled.mappings {
        alloc(m.vaddr, m.size);
    }
}

fn end_of(start: u64, size: u64) -> u64 {
    start.checked_add(size).unwrap_or(u64::MAX)
}

fn overlap(a: u64, asz: u64, b: u64, bsz: u64) -> bool {
    asz != 0 && bsz != 0 && a < end_of(b, bsz) && b < end_of(a, asz)
}

fn align_up(v: u64, a: u64) -> Option<u64> {
    if a == 0 {
        return Some(v);
    }
    let r = v % a;
    if r == 0 { Some(v) } else { v.checked_add(a - r) }
}

/// Count a failed memory call by name and return its count when it is to be
/// logged: the first 32 failures of each call, then only at powers of two. A
/// title that retries a failing call (Demon's Souls retries its tracking setup
/// until the stack overflows) must not flood the log.
fn note_failure(call: &'static str) -> Option<u64> {
    static COUNTS: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());
    let mut counts = COUNTS.lock().unwrap_or_else(|e| e.into_inner());
    let n = match counts.iter_mut().find(|(c, _)| *c == call) {
        Some((_, n)) => {
            *n += 1;
            *n
        }
        None => {
            counts.push((call, 1));
            1
        }
    };
    (n <= 32 || n.is_power_of_two()).then_some(n)
}

fn truncate_name(name: &str) -> String {
    let mut n = name.to_string();
    while n.len() > NAME_LEN - 1 {
        n.pop();
    }
    n
}

/// Validate a guest protection: unknown-only bits are rejected.
fn valid_protection(prot: i32) -> bool {
    prot == 0 || prot & (PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC | PROT_GPU_READ | PROT_GPU_WRITE) != 0
}

/// The protection the queries report: every write bit brings its read bit
/// (CPU write 0x2 → 0x1, 0x80 → 0x40, 0x200 → 0x100), as measured on a PS5;
/// GPU write 0x20 does not add GPU read 0x10.
pub(crate) fn reported_protection(prot: i32) -> i32 {
    let mut p = prot;
    if p & 0x2 != 0 {
        p |= 0x1;
    }
    if p & 0x80 != 0 {
        p |= 0x40;
    }
    if p & 0x200 != 0 {
        p |= 0x100;
    }
    p
}

/// Host protection of a guest protection: GPU access needs host access,
/// write implies read.
fn host_prot(prot: i32) -> i32 {
    let read = prot & (PROT_CPU_READ | PROT_GPU_READ) != 0;
    let write = prot & (PROT_CPU_WRITE | PROT_GPU_WRITE) != 0;
    let mut p = libc::PROT_NONE;
    if read || write {
        p |= libc::PROT_READ;
    }
    if write {
        p |= libc::PROT_WRITE;
    }
    if prot & PROT_CPU_EXEC != 0 {
        p |= libc::PROT_EXEC;
    }
    p
}

fn vm_prot(prot: i32) -> ps5vm::MemProtect {
    let p = host_prot(prot);
    ps5vm::MemProtect { read: p & libc::PROT_READ != 0, write: p & libc::PROT_WRITE != 0, exec: p & libc::PROT_EXEC != 0 }
}

// ---------------------------------------------------------------------------
// Virtual ranges
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeType {
    Reserved,
    PoolReserved,
    Direct,
    Flexible,
    Pooled,
    Stack,
    Code,
    Runtime,
}

impl RangeType {
    fn reserved(self) -> bool {
        matches!(self, RangeType::Reserved | RangeType::PoolReserved)
    }

    fn pooled(self) -> bool {
        matches!(self, RangeType::Pooled | RangeType::PoolReserved)
    }

    fn committed(self) -> bool {
        !self.reserved()
    }

    fn private_committed(self) -> bool {
        matches!(self, RangeType::Stack | RangeType::Code | RangeType::Runtime)
    }
}

#[derive(Clone, Debug)]
struct Range {
    start: u64,
    size: u64,
    offset: u64,
    protection: i32,
    memory_type: i32,
    rtype: RangeType,
    disallow_merge: bool,
    name: String,
}

impl Range {
    fn end(&self) -> u64 {
        end_of(self.start, self.size)
    }

    /// The part `[start, end)` of this range.
    fn piece(&self, start: u64, end: u64) -> Range {
        let mut p = self.clone();
        p.start = start;
        p.size = end - start;
        if p.rtype == RangeType::Direct {
            p.offset += start - self.start;
        }
        p
    }
}

#[derive(Default)]
struct VirtualRanges {
    ranges: Vec<Range>,
}

impl VirtualRanges {
    fn same_merge_key(l: &Range, r: &Range) -> bool {
        if l.disallow_merge || r.disallow_merge || l.rtype == RangeType::Direct || r.rtype == RangeType::Direct {
            return false;
        }
        l.rtype == r.rtype && l.protection == r.protection && l.memory_type == r.memory_type && l.name == r.name
    }

    fn lower_bound(&self, start: u64) -> usize {
        self.ranges.partition_point(|r| r.start < start)
    }

    fn merge_around(&mut self, mut index: usize) {
        if index >= self.ranges.len() {
            return;
        }
        if index != 0 {
            let (prev, cur) = (&self.ranges[index - 1], &self.ranges[index]);
            if prev.end() == cur.start && Self::same_merge_key(prev, cur) {
                let add = cur.size;
                self.ranges[index - 1].size += add;
                self.ranges.remove(index);
                index -= 1;
            }
        }
        while index + 1 < self.ranges.len() {
            let (cur, next) = (&self.ranges[index], &self.ranges[index + 1]);
            if cur.end() != next.start || !Self::same_merge_key(cur, next) {
                break;
            }
            let add = next.size;
            self.ranges[index].size += add;
            self.ranges.remove(index + 1);
        }
    }

    fn merge_all(&mut self) {
        if self.ranges.len() < 2 {
            return;
        }
        self.ranges.sort_by_key(|r| r.start);
        let mut merged: Vec<Range> = Vec::with_capacity(self.ranges.len());
        for r in self.ranges.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.end() == r.start && Self::same_merge_key(last, &r) {
                    last.size += r.size;
                    continue;
                }
            }
            merged.push(r);
        }
        self.ranges = merged;
    }

    fn find_overlap(&self, start: u64, size: u64) -> Option<&Range> {
        let pos = self.lower_bound(start);
        if let Some(r) = self.ranges.get(pos) {
            if overlap(start, size, r.start, r.size) {
                return Some(r);
            }
        }
        if pos > 0 {
            let r = &self.ranges[pos - 1];
            if overlap(start, size, r.start, r.size) {
                return Some(r);
            }
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn add(&mut self, start: u64, size: u64, offset: u64, protection: i32, memory_type: i32, rtype: RangeType, name: &str, disallow_merge: bool) -> bool {
        if start == 0 || size == 0 || self.find_overlap(start, size).is_some() {
            return false;
        }
        let pos = self.lower_bound(start);
        self.ranges.insert(
            pos,
            Range { start, size, offset, protection, memory_type, rtype, disallow_merge, name: truncate_name(name) },
        );
        self.merge_around(pos);
        true
    }

    fn remove_unlocked(&mut self, start: u64, size: u64) -> bool {
        if size == 0 {
            return false;
        }
        let rem_end = end_of(start, size);
        let mut out = Vec::with_capacity(self.ranges.len() + 1);
        let mut removed = false;
        for r in self.ranges.drain(..) {
            if !overlap(start, size, r.start, r.size) {
                out.push(r);
                continue;
            }
            removed = true;
            if r.start < start {
                out.push(r.piece(r.start, start));
            }
            if rem_end < r.end() {
                out.push(r.piece(rem_end, r.end()));
            }
        }
        self.ranges = out;
        removed
    }

    fn remove(&mut self, start: u64, size: u64) -> bool {
        let pos = self.lower_bound(start);
        if let Some(r) = self.ranges.get(pos) {
            if r.start == start && r.size == size {
                self.ranges.remove(pos);
                return true;
            }
        }
        let removed = self.remove_unlocked(start, size);
        self.merge_all();
        removed
    }

    fn has_overlap(&self, start: u64, size: u64) -> bool {
        self.find_overlap(start, size).is_some()
    }

    fn query_overlap(&self, start: u64, size: u64) -> Option<Range> {
        self.find_overlap(start, size).cloned()
    }

    #[allow(clippy::too_many_arguments)]
    fn replace_span(&mut self, start: u64, size: u64, expected: RangeType, offset: u64, protection: i32, memory_type: i32, rtype: RangeType, name: &str, disallow_merge: bool) -> bool {
        if start == 0 || size == 0 || size > u64::MAX - start {
            return false;
        }
        let end = start + size;
        let mut current = start;
        while current < end {
            let Some(c) = self.ranges.iter().find(|r| r.rtype == expected && current >= r.start && current < r.end()) else {
                return false;
            };
            current = end.min(c.end());
        }
        self.remove_unlocked(start, size);
        let pos = self.lower_bound(start);
        self.ranges.insert(
            pos,
            Range { start, size, offset, protection, memory_type, rtype, disallow_merge, name: truncate_name(name) },
        );
        self.merge_around(pos);
        true
    }

    fn edit(&mut self, start: u64, size: u64, f: impl Fn(&mut Range)) {
        if size == 0 {
            return;
        }
        let edit_end = end_of(start, size);
        let mut out = Vec::with_capacity(self.ranges.len() + 2);
        for r in self.ranges.drain(..) {
            if !overlap(start, size, r.start, r.size) {
                out.push(r);
                continue;
            }
            let mid_start = start.max(r.start);
            let mid_end = edit_end.min(r.end());
            if r.start < mid_start {
                out.push(r.piece(r.start, mid_start));
            }
            let mut mid = r.piece(mid_start, mid_end);
            f(&mut mid);
            out.push(mid);
            if mid_end < r.end() {
                out.push(r.piece(mid_end, r.end()));
            }
        }
        self.ranges = out;
        self.merge_all();
    }

    fn rename(&mut self, start: u64, size: u64, name: &str) {
        let name = truncate_name(name);
        let pos = self.lower_bound(start);
        if let Some(r) = self.ranges.get_mut(pos) {
            if r.start == start && r.size == size {
                r.name = name;
                self.merge_around(pos);
                return;
            }
        }
        self.edit(start, size, |r| r.name = name.clone());
    }

    fn protect(&mut self, start: u64, size: u64, protection: i32) {
        self.edit(start, size, |r| r.protection = protection);
    }

    /// The ranges overlapping `[start, start+size)`, clipped to it, in
    /// address order (holes are simply absent).
    fn pieces(&self, start: u64, size: u64) -> Vec<Range> {
        let end = start.saturating_add(size);
        let first = self.ranges.partition_point(|r| r.end() <= start);
        self.ranges[first..].iter().take_while(|r| r.start < end).map(|r| r.piece(r.start.max(start), r.end().min(end))).collect()
    }

    /// The range containing `addr`, or with `flags == 1` the next one.
    fn query(&self, addr: u64, flags: i32) -> Option<Range> {
        let next = self.ranges.partition_point(|r| r.start <= addr);
        if next > 0 {
            let cur = &self.ranges[next - 1];
            if addr < cur.end() {
                return Some(cur.clone());
            }
        }
        if flags != 1 {
            return None;
        }
        self.ranges.get(next).cloned()
    }

    /// The ranges exactly covering `[start, start+size)`, clipped; None
    /// when there is a hole.
    fn query_span(&self, start: u64, size: u64) -> Option<Vec<Range>> {
        if start == 0 || size == 0 || size > u64::MAX - start {
            return None;
        }
        let end = start + size;
        let mut current = start;
        let mut out = Vec::new();
        let first = self.ranges.partition_point(|r| r.end() <= current);
        for r in &self.ranges[first..] {
            if r.end() <= current {
                continue;
            }
            if r.start > current {
                return None;
            }
            let part = r.piece(current, end.min(r.end()));
            current += part.size;
            out.push(part);
            if current == end {
                return Some(out);
            }
        }
        None
    }

    /// Size of the committed span starting at `addr` (up to `size`).
    fn clamp_range_size(&self, addr: u64, size: u64) -> u64 {
        if addr == 0 || size == 0 || size > u64::MAX - addr {
            return 0;
        }
        let next = self.ranges.partition_point(|r| r.start <= addr);
        if next == 0 {
            return 0;
        }
        let first = &self.ranges[next - 1];
        if addr >= first.end() || !first.rtype.committed() {
            return 0;
        }
        let mut clamped = size.min(first.end() - addr);
        let mut expected = addr + clamped;
        for r in &self.ranges[next..] {
            if r.start != expected || !r.rtype.committed() || clamped >= size {
                break;
            }
            let chunk = (size - clamped).min(r.size);
            clamped += chunk;
            expected += chunk;
        }
        clamped
    }

    /// Page-table entries in use, counted as the console does:
    /// - CPU: every 2 MiB granule a committed range touches, whatever its
    ///   protection, each granule once;
    /// - GPU: for the GPU-visible ranges, each 1 GiB region they touch,
    ///   plus each 2 MiB granule they touch except a granule a direct
    ///   mapping covers whole with 2 MiB-aligned physical memory (a large
    ///   page needs no table of its own); shared granules and regions once.
    fn count_page_table_entries(&self, gpu: bool) -> u64 {
        const G: u64 = PAGE_TABLE_GRANULARITY;
        let mut used = 0;
        let mut next_granule = 0u64;
        let mut next_region = 0u64;
        for r in &self.ranges {
            if !r.rtype.committed() || r.size == 0 {
                continue;
            }
            if gpu && r.protection & (PROT_GPU_READ | PROT_GPU_WRITE) == 0 {
                continue;
            }
            let (start, end) = (r.start, r.end());
            if gpu {
                let first = (start / PAGE_TABLE_GPU_REGION).max(next_region);
                let last = (end - 1) / PAGE_TABLE_GPU_REGION;
                if first <= last {
                    used += last - first + 1;
                    next_region = last + 1;
                }
            }
            let first = (start / G).max(next_granule);
            let last = (end - 1) / G;
            if first > last {
                continue;
            }
            next_granule = last + 1;
            for g in first..=last {
                let (gs, ge) = (g * G, g * G + G);
                let large = gpu
                    && r.rtype == RangeType::Direct
                    && gs >= start
                    && ge <= end
                    && (r.offset + (gs - start)) % G == 0;
                if !large {
                    used += 1;
                }
            }
        }
        used
    }
}

// ---------------------------------------------------------------------------
// Physical (direct) memory
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AllocationKind {
    Direct,
    Pooled,
    Automatic,
    /// System memory at `[TOTAL_BYTES, TOTAL_BYTES + SYSTEM_BYTES)`: never in
    /// the title's free ranges, skipped by the direct memory query.
    System,
}

#[derive(Clone, Debug)]
struct AllocatedBlock {
    start_addr: u64,
    size: u64,
    map_vaddr: u64,
    map_size: u64,
    prot: i32,
    memory_type: i32,
    kind: AllocationKind,
    name: String,
}

fn remove_free_range(ranges: &mut BTreeMap<u64, u64>, start: u64, size: u64) {
    let end = start + size;
    let first = ranges.range(..=start).next_back().map_or(start, |(&k, _)| k);
    let keys: Vec<u64> = ranges.range(first..end).map(|(&k, _)| k).collect();
    for k in keys {
        let len = ranges[&k];
        let rend = k + len;
        if rend <= start {
            continue;
        }
        ranges.remove(&k);
        if k < start {
            ranges.insert(k, start - k);
        }
        if end < rend {
            ranges.insert(end, rend - end);
        }
    }
}

fn add_free_range(ranges: &mut BTreeMap<u64, u64>, mut start: u64, size: u64) {
    let mut end = start + size;
    if let Some((&ps, &pl)) = ranges.range(..start).next_back() {
        if ps + pl >= start {
            start = ps;
            end = end.max(ps + pl);
            ranges.remove(&ps);
        }
    }
    let following: Vec<(u64, u64)> = ranges.range(start..=end).map(|(&k, &v)| (k, v)).collect();
    for (k, v) in following {
        end = end.max(k + v);
        ranges.remove(&k);
    }
    ranges.insert(start, end - start);
}

#[derive(Default)]
struct PhysicalMemory {
    physical: BTreeMap<u64, AllocatedBlock>,
    free: BTreeMap<u64, u64>,
    automatic_free: BTreeMap<u64, u64>,
    /// Free ranges of system memory (`AllocationKind::System`).
    system_free: BTreeMap<u64, u64>,
    mappings: Vec<AllocatedBlock>,
}

impl PhysicalMemory {
    fn new(size: u64) -> Self {
        let mut p = Self::default();
        p.free.insert(0, size);
        p.system_free.insert(TOTAL_BYTES, SYSTEM_BYTES);
        p
    }

    /// Allocate system memory: first fit in the system free ranges.
    fn alloc_system(&mut self, len: u64, alignment: u64, memory_type: i32) -> Option<u64> {
        let mut chosen = None;
        for (&rs, &rl) in &self.system_free {
            let Some(pos) = align_up(rs, alignment) else { continue };
            if pos < rs + rl && len <= rs + rl - pos {
                chosen = Some(pos);
                break;
            }
        }
        let pos = chosen?;
        remove_free_range(&mut self.system_free, pos, len);
        self.physical.insert(
            pos,
            AllocatedBlock { start_addr: pos, size: len, map_vaddr: 0, map_size: 0, prot: 0, memory_type, kind: AllocationKind::System, name: String::new() },
        );
        Some(pos)
    }

    #[allow(clippy::too_many_arguments)]
    fn alloc(&mut self, search_start: u64, search_end: u64, len: u64, alignment: u64, memory_type: i32, kind: AllocationKind) -> Option<u64> {
        let search_end = search_end.min(direct_size());
        if search_start >= search_end {
            return None;
        }
        let first = self.free.range(..=search_start).next_back().map_or(0, |(&k, _)| k);
        let mut chosen = None;
        for (&rs, &rl) in self.free.range(first..) {
            if rs >= search_end {
                break;
            }
            let range_end = (rs + rl).min(search_end);
            let lower = rs.max(search_start);
            let Some(pos) = align_up(lower, alignment) else { continue };
            if pos > range_end || len > range_end - pos {
                continue;
            }
            chosen = Some(pos);
            break;
        }
        let pos = chosen?;
        remove_free_range(&mut self.free, pos, len);
        self.physical.insert(
            pos,
            AllocatedBlock { start_addr: pos, size: len, map_vaddr: 0, map_size: 0, prot: 0, memory_type, kind, name: String::new() },
        );
        if kind == AllocationKind::Automatic {
            add_free_range(&mut self.automatic_free, pos, len);
        }
        Some(pos)
    }

    /// Largest aligned free span in the window: (start, size).
    fn available(&self, search_start: u64, search_end: u64, alignment: u64) -> Option<(u64, u64)> {
        let search_end = search_end.min(direct_size());
        if search_start >= search_end {
            return None;
        }
        let mut best = (0, 0);
        for (&rs, &rl) in &self.free {
            if rs >= search_end {
                break;
            }
            let range_end = (rs + rl).min(search_end);
            let lower = rs.max(search_start);
            let Some(pos) = align_up(lower, alignment) else { continue };
            if pos >= lower && pos < range_end && range_end - pos > best.1 {
                best = (pos, range_end - pos);
            }
        }
        (best.1 != 0).then_some(best)
    }

    fn reserve_automatic(&mut self, mut size: u64) -> Option<Vec<(u64, u64)>> {
        let mut out = Vec::new();
        for (&start, &avail) in &self.automatic_free {
            let len = size.min(avail);
            out.push((start, len));
            size -= len;
            if size == 0 {
                for &(o, l) in &out {
                    remove_free_range(&mut self.automatic_free, o, l);
                }
                return Some(out);
            }
        }
        None
    }

    fn restore_automatic(&mut self, ranges: &[(u64, u64)]) {
        for &(s, l) in ranges {
            add_free_range(&mut self.automatic_free, s, l);
        }
    }

    fn reclaim_automatic(&mut self, start: u64, size: u64) {
        let end = start + size;
        let first = self.physical.range(..=start).next_back().map_or(start, |(&k, _)| k);
        let mut automatic = false;
        let blocks: Vec<(u64, u64)> = self
            .physical
            .range(first..end)
            .filter(|(_, b)| b.kind == AllocationKind::Automatic && b.start_addr + b.size > start)
            .map(|(_, b)| (start.max(b.start_addr), end.min(b.start_addr + b.size)))
            .collect();
        for (f, l) in blocks {
            add_free_range(&mut self.automatic_free, f, l - f);
            automatic = true;
        }
        if automatic {
            let mapped: Vec<(u64, u64)> = self
                .mappings
                .iter()
                .filter_map(|m| {
                    let f = start.max(m.start_addr);
                    let l = end.min(m.start_addr + m.size);
                    (f < l).then_some((f, l - f))
                })
                .collect();
            for (f, l) in mapped {
                remove_free_range(&mut self.automatic_free, f, l);
            }
        }
    }

    /// Release part of one non-pooled block.
    fn release(&mut self, start: u64, len: u64) -> bool {
        let Some((&k, b)) = self.physical.range(..=start).next_back() else { return false };
        if b.kind == AllocationKind::Pooled || start < b.start_addr || start >= b.start_addr + b.size || len > b.start_addr + b.size - start {
            return false;
        }
        let b = self.physical.remove(&k).unwrap();
        if b.kind == AllocationKind::Automatic {
            remove_free_range(&mut self.automatic_free, start, len);
        }
        let end = start + len;
        let bend = b.start_addr + b.size;
        if b.start_addr < start {
            let mut left = b.clone();
            left.size = start - b.start_addr;
            if left.map_vaddr != 0 {
                left.map_size = left.size;
            }
            self.physical.insert(left.start_addr, left);
        }
        if end < bend {
            let mut right = b.clone();
            right.start_addr = end;
            right.size = bend - end;
            if right.map_vaddr != 0 {
                right.map_vaddr += end - b.start_addr;
                right.map_size = right.size;
            }
            self.physical.insert(right.start_addr, right);
        }
        let free = if b.kind == AllocationKind::System { &mut self.system_free } else { &mut self.free };
        add_free_range(free, start, len);
        true
    }

    /// Whether `[phys, phys+len)` is covered by contiguous non-pooled blocks.
    fn can_map_direct(&self, phys: u64, len: u64) -> bool {
        self.allocated_span(phys, len, false).is_some()
    }

    /// The allocated non-pooled parts of `[phys, phys+len)`. Without
    /// `skip_gaps` the range must be covered contiguously; with it the
    /// unallocated gaps are skipped and only the allocated parts return
    /// (an unchecked release frees those and leaves the gaps alone).
    fn allocated_span(&self, phys: u64, len: u64, skip_gaps: bool) -> Option<Vec<AllocatedBlock>> {
        if len == 0 || u64::MAX - phys < len {
            return None;
        }
        let end = phys + len;
        let mut current = phys;
        let mut out = Vec::new();
        while current < end {
            let (_, b) = self.physical.range(..=current).next_back()?;
            let bend = b.start_addr + b.size;
            if b.kind == AllocationKind::Pooled {
                return None;
            }
            if current >= bend {
                // The range starts (or stepped) into a gap.
                if !skip_gaps {
                    return None;
                }
                current = self.physical.range(current..end).next().map_or(end, |(&k, _)| k);
                continue;
            }
            let mut part = b.clone();
            part.start_addr = current;
            part.size = end.min(bend) - current;
            current += part.size;
            out.push(part);
            if current < end && !self.physical.contains_key(&current) {
                if !skip_gaps {
                    return None;
                }
                current = self.physical.range(current..end).next().map_or(end, |(&k, _)| k);
            }
        }
        Some(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn map(&mut self, vaddr: u64, phys: u64, len: u64, prot: i32, name: &str, memory_type: i32) -> bool {
        let Some(span) = self.allocated_span(phys, len, false) else { return false };
        let has_automatic = span.iter().any(|b| b.kind == AllocationKind::Automatic);
        let first = &span[0];
        self.mappings.push(AllocatedBlock {
            start_addr: phys,
            size: len,
            map_vaddr: vaddr,
            map_size: len,
            prot,
            memory_type,
            kind: first.kind,
            name: truncate_name(name),
        });
        if has_automatic {
            remove_free_range(&mut self.automatic_free, phys, len);
        }
        true
    }

    fn release_pool_expansion(&mut self, phys: u64, len: u64) -> bool {
        match self.physical.get(&phys) {
            Some(b) if b.kind == AllocationKind::Pooled && b.size == len => {}
            _ => return false,
        }
        self.physical.remove(&phys);
        add_free_range(&mut self.free, phys, len);
        true
    }

    /// Remove a view (or part of one) of direct memory.
    fn unmap(&mut self, vaddr: u64, size: u64) -> bool {
        for i in 0..self.mappings.len() {
            let b = &self.mappings[i];
            if vaddr < b.map_vaddr || vaddr >= b.map_vaddr + b.map_size || size > b.map_vaddr + b.map_size - vaddr {
                continue;
            }
            let phys = b.start_addr + vaddr - b.map_vaddr;
            let b = self.mappings[i].clone();
            if vaddr == b.map_vaddr && size == b.map_size {
                self.mappings.remove(i);
            } else if vaddr > b.map_vaddr && vaddr + size < b.map_vaddr + b.map_size {
                let mut right = b.clone();
                right.start_addr += vaddr + size - b.map_vaddr;
                right.size = b.map_vaddr + b.map_size - (vaddr + size);
                right.map_size = right.size;
                right.map_vaddr = vaddr + size;
                let m = &mut self.mappings[i];
                m.size = vaddr - b.map_vaddr;
                m.map_size = m.size;
                self.mappings.push(right);
            } else if vaddr == b.map_vaddr {
                let m = &mut self.mappings[i];
                m.start_addr += size;
                m.size -= size;
                m.map_vaddr += size;
                m.map_size -= size;
            } else {
                let m = &mut self.mappings[i];
                m.size = vaddr - b.map_vaddr;
                m.map_size = m.size;
            }
            self.reclaim_automatic(phys, size);
            return true;
        }
        false
    }

    fn protect_mapping(&mut self, vaddr: u64, size: u64, prot: i32) {
        if size == 0 || u64::MAX - vaddr < size {
            return;
        }
        let end = vaddr + size;
        let mut updated = Vec::with_capacity(self.mappings.len() + 2);
        for b in self.mappings.drain(..) {
            let bend = b.map_vaddr + b.map_size;
            if !overlap(vaddr, size, b.map_vaddr, b.map_size) {
                updated.push(b);
                continue;
            }
            let os = vaddr.max(b.map_vaddr);
            let oe = end.min(bend);
            if b.map_vaddr < os {
                let mut left = b.clone();
                left.size = os - b.map_vaddr;
                left.map_size = left.size;
                updated.push(left);
            }
            let mut mid = b.clone();
            mid.start_addr = b.start_addr + os - b.map_vaddr;
            mid.size = oe - os;
            mid.map_vaddr = os;
            mid.map_size = mid.size;
            mid.prot = prot;
            updated.push(mid);
            if oe < bend {
                let mut right = b.clone();
                right.start_addr = b.start_addr + oe - b.map_vaddr;
                right.size = bend - oe;
                right.map_vaddr = oe;
                right.map_size = right.size;
                updated.push(right);
            }
        }
        self.mappings = updated;
    }

    /// The block containing `phys`, or with `next` the following one.
    fn find(&self, phys: u64, next: bool) -> Option<AllocatedBlock> {
        if let Some((_, b)) = self.physical.range(..=phys).next_back() {
            if phys < b.start_addr + b.size {
                return Some(b.clone());
            }
        }
        if next {
            return self.physical.range(phys..).next().map(|(_, b)| b.clone());
        }
        None
    }

    /// Views of `[phys, phys+len)`, clipped, ordered by address.
    fn find_mappings(&self, phys: u64, len: u64) -> Vec<AllocatedBlock> {
        let mut out: Vec<AllocatedBlock> = self
            .mappings
            .iter()
            .filter(|m| overlap(phys, len, m.start_addr, m.size))
            .map(|m| {
                let os = phys.max(m.start_addr);
                let oe = (phys + len).min(m.start_addr + m.size);
                let mut part = m.clone();
                part.start_addr = os;
                part.size = oe - os;
                part.map_vaddr += os - m.start_addr;
                part.map_size = part.size;
                part
            })
            .collect();
        out.sort_by_key(|m| m.map_vaddr);
        out
    }

    fn set_name(&mut self, vaddr: u64, len: u64, name: &str) {
        for b in &mut self.mappings {
            if overlap(vaddr, len, b.map_vaddr, b.map_size) {
                b.name = truncate_name(name);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Flexible memory
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct FlexBlock {
    map_vaddr: u64,
    map_size: u64,
    backing_offset: u64,
    prot: i32,
    name: String,
}

#[derive(Default)]
struct FlexibleMemory {
    blocks: Vec<FlexBlock>,
    free: BTreeMap<u64, u64>,
    allocated_total: u64,
    /// Guest ranges charged against the flexible budget: a map charges,
    /// munmap and a release drop the charge (the release keeps the mapping
    /// and its contents), a fixed map over charged bytes replaces them.
    charged: BTreeMap<u64, u64>,
}

impl FlexibleMemory {
    fn new() -> Self {
        let mut f = Self::default();
        f.free.insert(direct_size(), flexible_size());
        f
    }

    fn available(&self) -> u64 {
        flexible_size().saturating_sub(self.charged.values().sum())
    }

    fn map(&mut self, host: &HostSpace, vaddr: u64, len: u64, prot: i32, name: &str) -> bool {
        if len == 0 || len > self.available() {
            log::warn!(
                "[mem] flexible memory exhausted: configured={:#x} allocated={:#x} requested={len:#x}",
                flexible_size(),
                self.allocated_total
            );
            return false;
        }
        let mut blocks = Vec::new();
        let mut current = vaddr;
        let mut remaining = len;
        for (&off, &free) in &self.free {
            if remaining == 0 {
                break;
            }
            let chunk = remaining.min(free);
            blocks.push(FlexBlock { map_vaddr: current, map_size: chunk, backing_offset: off, prot, name: truncate_name(name) });
            current += chunk;
            remaining -= chunk;
        }
        if remaining != 0 {
            return false;
        }
        let mut mapped: Vec<&FlexBlock> = Vec::new();
        for b in &blocks {
            if !host.zero_backing(b.backing_offset, b.map_size) || !host.map_backing(b.map_vaddr, b.map_size, b.backing_offset, prot) {
                for m in mapped.iter().rev() {
                    host.unmap_backing(m.map_vaddr, m.map_size);
                }
                return false;
            }
            mapped.push(b);
        }
        for b in blocks {
            remove_free_range(&mut self.free, b.backing_offset, b.map_size);
            self.blocks.push(b);
        }
        self.blocks.sort_by_key(|b| b.map_vaddr);
        self.allocated_total += len;
        add_free_range(&mut self.charged, vaddr, len);
        true
    }

    fn covers(&self, vaddr: u64, size: u64) -> bool {
        let end = vaddr + size;
        let mut current = vaddr;
        for b in &self.blocks {
            if b.map_vaddr + b.map_size <= current {
                continue;
            }
            if b.map_vaddr > current || b.map_vaddr >= end {
                break;
            }
            current = end.min(b.map_vaddr + b.map_size);
            if current == end {
                return true;
            }
        }
        false
    }

    fn unmap(&mut self, host: &HostSpace, vaddr: u64, size: u64) -> bool {
        if size == 0 || u64::MAX - vaddr < size || !self.covers(vaddr, size) {
            return false;
        }
        host.unmap_backing(vaddr, size);
        self.forget(vaddr, size);
        true
    }

    /// Drop the bookkeeping of `[vaddr, vaddr+size)` (fully covered).
    fn forget(&mut self, vaddr: u64, size: u64) {
        let end = vaddr + size;
        let mut remaining = Vec::with_capacity(self.blocks.len() + 1);
        let mut removed = 0;
        for b in self.blocks.drain(..) {
            let bend = b.map_vaddr + b.map_size;
            if !overlap(vaddr, size, b.map_vaddr, b.map_size) {
                remaining.push(b);
                continue;
            }
            let os = vaddr.max(b.map_vaddr);
            let oe = end.min(bend);
            add_free_range(&mut self.free, b.backing_offset + os - b.map_vaddr, oe - os);
            removed += oe - os;
            if b.map_vaddr < os {
                let mut left = b.clone();
                left.map_size = os - b.map_vaddr;
                remaining.push(left);
            }
            if oe < bend {
                let mut right = b.clone();
                right.map_vaddr = oe;
                right.map_size = bend - oe;
                right.backing_offset = b.backing_offset + oe - b.map_vaddr;
                remaining.push(right);
            }
        }
        self.blocks = remaining;
        self.allocated_total -= removed;
        remove_free_range(&mut self.charged, vaddr, size);
    }

    /// Bytes of `[vaddr, vaddr+size)` the budget charges (a fixed map
    /// over them replaces them).
    fn charged_overlap(&self, vaddr: u64, size: u64) -> u64 {
        if size == 0 || u64::MAX - vaddr < size {
            return 0;
        }
        let end = vaddr + size;
        self.charged.iter().map(|(&k, &l)| end.min(k.saturating_add(l)).saturating_sub(vaddr.max(k))).sum()
    }

    /// Give the bytes of `[vaddr, vaddr+size)` back to the flexible budget
    /// while the mapping and its contents stay (measured on a console: the
    /// released range keeps its protection and reads back its bytes, a
    /// fixed no-overwrite map over it fails with ENOMEM and a plain map
    /// over it zeroes it and charges it again; the host pages stay
    /// committed until the range is unmapped). Parts outside every charged
    /// range change nothing.
    fn release(&mut self, vaddr: u64, size: u64) {
        remove_free_range(&mut self.charged, vaddr, size);
    }

    fn snapshot(&self, vaddr: u64, size: u64) -> Option<Vec<FlexBlock>> {
        if size == 0 || u64::MAX - vaddr < size {
            return None;
        }
        let end = vaddr + size;
        let mut current = vaddr;
        let mut out = Vec::new();
        for b in &self.blocks {
            let bend = b.map_vaddr + b.map_size;
            if bend <= current {
                continue;
            }
            if b.map_vaddr > current || b.map_vaddr >= end {
                break;
            }
            let pe = end.min(bend);
            let mut part = b.clone();
            part.backing_offset += current - b.map_vaddr;
            part.map_vaddr = current;
            part.map_size = pe - current;
            out.push(part);
            current = pe;
            if current == end {
                return Some(out);
            }
        }
        None
    }

    fn protect(&mut self, vaddr: u64, size: u64, prot: i32) {
        if size == 0 || u64::MAX - vaddr < size {
            return;
        }
        let end = vaddr + size;
        let mut updated = Vec::with_capacity(self.blocks.len() + 2);
        for b in self.blocks.drain(..) {
            let bend = b.map_vaddr + b.map_size;
            if !overlap(vaddr, size, b.map_vaddr, b.map_size) {
                updated.push(b);
                continue;
            }
            let os = vaddr.max(b.map_vaddr);
            let oe = end.min(bend);
            if b.map_vaddr < os {
                let mut left = b.clone();
                left.map_size = os - b.map_vaddr;
                updated.push(left);
            }
            let mut mid = b.clone();
            mid.map_vaddr = os;
            mid.map_size = oe - os;
            mid.backing_offset = b.backing_offset + os - b.map_vaddr;
            mid.prot = prot;
            updated.push(mid);
            if oe < bend {
                let mut right = b.clone();
                right.map_vaddr = oe;
                right.map_size = bend - oe;
                right.backing_offset = b.backing_offset + oe - b.map_vaddr;
                updated.push(right);
            }
        }
        self.blocks = updated;
    }

    fn set_name(&mut self, vaddr: u64, len: u64, name: &str) {
        for b in &mut self.blocks {
            if overlap(vaddr, len, b.map_vaddr, b.map_size) {
                b.name = truncate_name(name);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pooled memory
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct PoolMapping {
    vaddr: u64,
    size: u64,
    phys: u64,
}

#[derive(Default)]
struct PooledMemory {
    free: Vec<(u64, u64)>,
    expansions: Vec<(u64, u64)>,
    mappings: Vec<PoolMapping>,
    committed: u64,
}

impl PooledMemory {
    fn add_free(&mut self, start: u64, size: u64) {
        if size == 0 {
            return;
        }
        self.free.push((start, size));
        self.free.sort_by_key(|r| r.0);
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for (s, l) in self.free.drain(..) {
            if let Some(last) = merged.last_mut() {
                if s <= last.0 + last.1 {
                    let e = (last.0 + last.1).max(s + l);
                    last.1 = e - last.0;
                    continue;
                }
            }
            merged.push((s, l));
        }
        self.free = merged;
    }

    fn expand(&mut self, phys: u64, size: u64) {
        self.expansions.push((phys, size));
        self.add_free(phys, size);
    }

    fn release_expansion(&mut self, phys: u64, size: u64) -> bool {
        let Some(i) = self.expansions.iter().position(|&(s, l)| s == phys && l == size) else { return false };
        if self.mappings.iter().any(|m| overlap(phys, size, m.phys, m.size)) {
            return false;
        }
        let end = phys + size;
        let Some(fi) = self.free.iter().position(|&(s, l)| phys >= s && end <= s + l) else { return false };
        let (s, l) = self.free.remove(fi);
        if phys > s {
            self.free.push((s, phys - s));
        }
        if s + l > end {
            self.free.push((end, s + l - end));
        }
        self.free.sort_by_key(|r| r.0);
        self.expansions.remove(i);
        true
    }

    fn allocate(&mut self, vaddr: u64, size: u64) -> Option<Vec<PoolMapping>> {
        if vaddr == 0 || size == 0 || u64::MAX - vaddr < size {
            return None;
        }
        let mut free = self.free.clone();
        let mut out = Vec::new();
        let mut current = vaddr;
        let mut remaining = size;
        for r in free.iter_mut() {
            if remaining == 0 || r.1 == 0 {
                continue;
            }
            let part = r.1.min(remaining);
            out.push(PoolMapping { vaddr: current, size: part, phys: r.0 });
            r.0 += part;
            r.1 -= part;
            current += part;
            remaining -= part;
        }
        if remaining != 0 {
            return None;
        }
        free.retain(|r| r.1 != 0);
        self.free = free;
        self.mappings.extend(out.iter().copied());
        Some(out)
    }

    fn query(&self, vaddr: u64, size: u64) -> Option<Vec<PoolMapping>> {
        if vaddr == 0 || size == 0 || u64::MAX - vaddr < size {
            return None;
        }
        let end = vaddr + size;
        let mut current = vaddr;
        let mut out = Vec::new();
        while current < end {
            let m = self.mappings.iter().find(|m| current >= m.vaddr && current < m.vaddr + m.size)?;
            let part = end.min(m.vaddr + m.size) - current;
            out.push(PoolMapping { vaddr: current, size: part, phys: m.phys + (current - m.vaddr) });
            current += part;
        }
        Some(out)
    }

    fn release(&mut self, vaddr: u64, size: u64) -> bool {
        let Some(released) = self.query(vaddr, size) else { return false };
        let end = vaddr + size;
        let mut kept = Vec::with_capacity(self.mappings.len() + 1);
        for m in self.mappings.drain(..) {
            let mend = m.vaddr + m.size;
            let cs = vaddr.max(m.vaddr);
            let ce = end.min(mend);
            if cs >= ce {
                kept.push(m);
                continue;
            }
            if m.vaddr < cs {
                kept.push(PoolMapping { size: cs - m.vaddr, ..m });
            }
            if ce < mend {
                kept.push(PoolMapping { vaddr: ce, size: mend - ce, phys: m.phys + (ce - m.vaddr) });
            }
        }
        self.mappings = kept;
        for m in released {
            self.add_free(m.phys, m.size);
        }
        true
    }

    fn available(&self) -> u64 {
        self.free.iter().map(|r| r.1).sum()
    }

    fn subtract_committed(&mut self, len: u64) {
        self.committed = self.committed.saturating_sub(len);
    }
}

// ---------------------------------------------------------------------------
// Host address space
// ---------------------------------------------------------------------------

/// Host side of the guest address space: the physical backing file and the
/// host mappings of guest ranges.
#[cfg(unix)]
struct HostSpace {
    fd: i32,
}

#[cfg(unix)]
impl HostSpace {
    fn create() -> Option<Self> {
        let name = b"ps5-physical\0";
        // SAFETY: plain syscalls on our own descriptor.
        let fd = unsafe { libc::memfd_create(name.as_ptr() as *const libc::c_char, libc::MFD_CLOEXEC) };
        if fd < 0 {
            return None;
        }
        if unsafe { libc::ftruncate(fd, BACKING_BYTES as libc::off_t) } != 0 {
            unsafe { libc::close(fd) };
            return None;
        }
        // Second view of the whole backing for the GPU side: it reads and
        // writes guest memory through it without tripping the protection
        // the GPU memory tracker installs on the guest views.
        let alias = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                BACKING_BYTES as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_NORESERVE,
                fd,
                0,
            )
        };
        if alias != libc::MAP_FAILED {
            ps5gpu::memory::global().set_backing_alias(alias as u64, BACKING_BYTES);
        } else {
            log::warn!("[mem] cannot map the backing alias; GPU memory tracking disabled");
        }
        Some(Self { fd })
    }

    /// Zero a backing range and give its pages back to the host.
    fn zero_backing(&self, offset: u64, size: u64) -> bool {
        // SAFETY: hole punch inside our backing file.
        unsafe {
            libc::fallocate(self.fd, libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE, offset as libc::off_t, size as libc::off_t) == 0
        }
    }

    /// Map a view of the backing at a held guest range.
    fn map_backing(&self, addr: u64, len: u64, offset: u64, prot: i32) -> bool {
        // SAFETY: the range is owned by the memory manager.
        let r = unsafe {
            libc::mmap(addr as *mut libc::c_void, len as usize, host_prot(prot), libc::MAP_SHARED | libc::MAP_FIXED, self.fd, offset as libc::off_t)
        };
        if r == libc::MAP_FAILED {
            return false;
        }
        log::debug!("[mem] view {addr:#x}+{len:#x} backing {offset:#x} prot {prot:#x}");
        ps5vm::ranges::track(addr, len, vm_prot(prot));
        ps5gpu::memory::global().note_mapped(addr, len, host_prot(prot), Some(offset));
        true
    }

    /// Replace a view with an inaccessible hold of the same range.
    fn unmap_backing(&self, addr: u64, len: u64) -> bool {
        hold_fixed(addr, len)
    }

    /// Private committed memory (stacks, runtime allocations).
    fn commit(&self, addr: u64, len: u64, prot: i32) -> bool {
        // SAFETY: the range is owned by the memory manager.
        let r = unsafe {
            libc::mmap(addr as *mut libc::c_void, len as usize, host_prot(prot), libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED, -1, 0)
        };
        if r == libc::MAP_FAILED {
            return false;
        }
        log::debug!("[mem] private {addr:#x}+{len:#x} prot {prot:#x}");
        ps5vm::ranges::track(addr, len, vm_prot(prot));
        // Untracked memory the GPU caches cannot watch, possibly over pages
        // they hold.
        ps5gpu::memory::global().note_mapped(addr, len, host_prot(prot), None);
        true
    }

    fn protect(&self, addr: u64, len: u64, prot: i32) -> bool {
        // Ranges the GPU memory tracker manages get the guest protection
        // intersected with its page watches.
        let ok = ps5gpu::memory::global().note_protected(addr, len, host_prot(prot))
            // SAFETY: the range is owned by the memory manager.
            || unsafe { libc::mprotect(addr as *mut libc::c_void, len as usize, host_prot(prot)) == 0 };
        if ok {
            ps5vm::ranges::track(addr, len, vm_prot(prot));
        }
        ok
    }
}

/// Hold a host range we own as an inaccessible, non-committed reservation.
/// Whether `[start, start+size)` lies inside one guest-owned region: the
/// system-managed/system-reserved low region, the user region or the
/// automatic-memory window. Fixed mappings elsewhere have no fallback.
fn in_guest_region(start: u64, size: u64) -> bool {
    let Some(end) = start.checked_add(size) else { return false };
    let inside = |lo: u64, hi: u64| start >= lo && end <= hi + 1;
    inside(HOST_SYSTEM_MANAGED_MIN, SYSTEM_RESERVED_MAX)
        || inside(HOST_USER_MIN, HOST_USER_MAX)
        || inside(EXTENDED_MEMORY_BASE, EXTENDED_MEMORY_BASE + EXTENDED_MEMORY_SIZE - 1)
}

#[cfg(unix)]
fn hold_fixed(addr: u64, len: u64) -> bool {
    log::debug!("[mem] hold {addr:#x}+{len:#x}");
    ps5gpu::memory::global().note_unmapped(addr, len);
    // SAFETY: the range is owned by the memory manager.
    let r = unsafe {
        libc::mmap(
            addr as *mut libc::c_void,
            len as usize,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | libc::MAP_FIXED,
            -1,
            0,
        )
    };
    if r == libc::MAP_FAILED {
        return false;
    }
    ps5vm::ranges::track(addr, len, ps5vm::MemProtect::NONE);
    true
}

/// Hold a host range without replacing anything already mapped there.
#[cfg(unix)]
fn hold_noreplace(addr: u64, len: u64) -> bool {
    // SAFETY: MAP_FIXED_NOREPLACE never clobbers existing mappings.
    let r = unsafe {
        libc::mmap(
            addr as *mut libc::c_void,
            len as usize,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | libc::MAP_FIXED_NOREPLACE,
            -1,
            0,
        )
    };
    if r == libc::MAP_FAILED {
        return false;
    }
    if r as u64 != addr {
        // SAFETY: undo a mapping the kernel placed elsewhere.
        unsafe { libc::munmap(r, len as usize) };
        return false;
    }
    ps5vm::ranges::track(addr, len, ps5vm::MemProtect::NONE);
    true
}

/// Give a held range back to the host.
#[cfg(unix)]
fn release_host(addr: u64, len: u64) {
    ps5gpu::memory::global().note_unmapped(addr, len);
    // SAFETY: the range is owned by the memory manager.
    unsafe { libc::munmap(addr as *mut libc::c_void, len as usize) };
    ps5vm::ranges::untrack(addr, len);
}

/// End of the host mapping overlapping `[addr, addr+len)`, if any.
#[cfg(unix)]
fn host_conflict_end(addr: u64, len: u64) -> Option<u64> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    let end = addr + len;
    let mut best: Option<u64> = None;
    for line in maps.lines() {
        let range = line.split_whitespace().next()?;
        let (s, e) = range.split_once('-')?;
        let (s, e) = (u64::from_str_radix(s, 16).ok()?, u64::from_str_radix(e, 16).ok()?);
        if s < end && addr < e {
            best = Some(best.map_or(e, |b: u64| b.max(e)));
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Manager state
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct PrtAperture {
    address: u64,
    size: u64,
}

struct MemState {
    host: Option<HostSpace>,
    physical: PhysicalMemory,
    flexible: FlexibleMemory,
    pooled: PooledMemory,
    ranges: VirtualRanges,
    prt: [PrtAperture; (PRT_APERTURE_MAX_INDEX + 1) as usize],
}

static MEM: Mutex<Option<MemState>> = Mutex::new(None);

/// Module segments loaded before the memory state exists (creating it
/// freezes the flexible memory size, which the program parameters set later).
static PENDING_CODE: Mutex<Vec<(u64, u64, i32, String)>> = Mutex::new(Vec::new());

struct Guard(std::sync::MutexGuard<'static, Option<MemState>>);

impl std::ops::Deref for Guard {
    type Target = MemState;
    fn deref(&self) -> &MemState {
        self.0.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for Guard {
    fn deref_mut(&mut self) -> &mut MemState {
        self.0.as_mut().unwrap()
    }
}

fn lock() -> Guard {
    let mut g = MEM.lock().unwrap_or_else(|e| e.into_inner());
    if g.is_none() {
        FLEXIBLE_SIZE_FROZEN.store(true, Ordering::Release);
        let host = HostSpace::create();
        if host.is_none() {
            log::error!("[mem] failed to create physical memory backing");
        }
        *g = Some(MemState {
            host,
            physical: PhysicalMemory::new(direct_size()),
            flexible: FlexibleMemory::new(),
            pooled: PooledMemory::default(),
            ranges: VirtualRanges::default(),
            prt: [PrtAperture::default(); (PRT_APERTURE_MAX_INDEX + 1) as usize],
        });
        let st = g.as_mut().unwrap();
        for (start, size, prot, name) in PENDING_CODE.lock().unwrap_or_else(|e| e.into_inner()).drain(..) {
            st.ranges.add(start, size, 0, prot, 0, RangeType::Code, &name, false);
        }
    }
    Guard(g)
}

impl MemState {
    fn in_prt_aperture(&self, address: u64) -> bool {
        self.prt.iter().any(|a| address >= a.address && address - a.address < a.size)
    }

    /// Find and hold a free guest range: from `search_addr` up to the end
    /// of the user region, or without a hint from the default base in the
    /// system-managed region and then the user region.
    fn find_free_range(&self, search_addr: u64, size: u64, alignment: u64) -> u64 {
        let alignment = alignment.max(PAGE_SIZE);
        let find_in = |begin: u64, end: u64| -> u64 {
            let mut current = begin;
            let mut tries = 0;
            while current < end && size <= end - current && tries < 8192 {
                tries += 1;
                let Some(cand) = align_up(current, alignment) else { return 0 };
                if cand >= end || size > end - cand {
                    return 0;
                }
                if let Some(o) = self.ranges.query_overlap(cand, size) {
                    current = o.end();
                    continue;
                }
                if hold_noreplace(cand, size) {
                    return cand;
                }
                current = host_conflict_end(cand, size).unwrap_or(cand + alignment).max(cand + PAGE_SIZE);
            }
            0
        };
        if search_addr != 0 {
            return find_in(search_addr, HOST_USER_MAX + 1);
        }
        let addr = find_in(GUEST_DEFAULT_MAP_BASE, HOST_SYSTEM_MANAGED_MAX + 1);
        if addr != 0 {
            return addr;
        }
        find_in(HOST_USER_MIN, HOST_USER_MAX + 1)
    }

    /// Whether a non-fixed address hint names an aligned span that lies
    /// wholly inside reserved ranges (then the mapping goes there).
    fn hint_in_reservation(&self, hint: u64, size: u64, alignment: u64) -> bool {
        hint != 0
            && hint % PAGE_SIZE == 0
            && (alignment == 0 || hint % alignment == 0)
            && self.ranges.query_span(hint, size).is_some_and(|rs| rs.iter().all(|r| r.rtype == RangeType::Reserved))
    }

    /// Whether every page of `[start, start+size)` is guest memory: a range
    /// of the map (with `committed`, a committed one) or, outside the map, a
    /// loader image the guest range registry tracks.
    fn guest_span(&self, start: u64, size: u64, committed: bool) -> bool {
        let Some(end) = start.checked_add(size) else { return false };
        let mut cursor = start;
        while cursor < end {
            match self.ranges.query(cursor, 0) {
                Some(r) => {
                    if committed && !r.rtype.committed() {
                        return false;
                    }
                    cursor = end.min(r.end());
                }
                None => {
                    let next = self.ranges.query(cursor, 1).map_or(end, |r| r.start.min(end));
                    if !ps5vm::ranges::covered(cursor, next - cursor) {
                        return false;
                    }
                    cursor = next;
                }
            }
        }
        true
    }

    /// Turn `[start, start+size)` into one reserved range: existing views
    /// are unmapped (and restored on failure), gaps must be free host
    /// memory.
    fn replace_fixed_range_with_reserved(&mut self, start: u64, size: u64) -> bool {
        if !in_guest_region(start, size) {
            return false;
        }
        let Some(host) = self.host.as_ref() else { return false };
        let end = start + size;
        let mut chunks: Vec<Range> = Vec::new();
        let mut current = start;
        while current < end {
            let Some(r) = self.ranges.query(current, 1) else { break };
            if r.start >= end {
                break;
            }
            if current < r.start {
                current = end.min(r.start);
                continue;
            }
            let piece = r.piece(current, end.min(r.end()));
            if r.rtype == RangeType::Flexible && self.flexible.snapshot(piece.start, piece.size).is_none() {
                return false;
            }
            if r.rtype == RangeType::Pooled {
                log::error!("[mem] fixed reservation over pooled memory at {:#x}+{:#x} is unsupported", piece.start, piece.size);
                return false;
            }
            current += piece.size;
            chunks.push(piece);
        }
        // Gaps between existing ranges must be free on the host.
        let mut held_gaps = Vec::new();
        let mut cur = start;
        for c in &chunks {
            if c.start > cur {
                held_gaps.push((cur, c.start - cur));
            }
            cur = c.end();
        }
        if cur < end {
            held_gaps.push((cur, end - cur));
        }
        let mut held = Vec::new();
        for &(g, gl) in &held_gaps {
            if !hold_noreplace(g, gl) {
                for &(h, hl) in &held {
                    release_host(h, hl);
                }
                return false;
            }
            held.push((g, gl));
        }
        self.ranges.remove(start, size);
        for c in &chunks {
            match c.rtype {
                RangeType::Direct => {
                    host.unmap_backing(c.start, c.size);
                    self.physical.unmap(c.start, c.size);
                }
                RangeType::Flexible => {
                    self.flexible.unmap(host, c.start, c.size);
                }
                t if t.private_committed() => {
                    hold_fixed(c.start, c.size);
                }
                _ => {}
            }
        }
        self.ranges.add(start, size, 0, 0, 0, RangeType::Reserved, "anon", false);
        for c in &chunks {
            if c.rtype.committed() {
                free_callback(c.start, c.size);
            }
        }
        true
    }

    /// Unmap one range piece (recursing across range boundaries).
    fn unmap_range(&mut self, vaddr: u64, len: u64) -> HleResult {
        if len == 0 || u64::MAX - vaddr < len {
            return EINVAL;
        }
        let Some(range) = self.ranges.query(vaddr, 0) else { return EACCES };
        let chunk = len.min(range.size - (vaddr - range.start));
        if chunk < len {
            let r = self.unmap_range(vaddr, chunk);
            return if r == OK { self.unmap_range(vaddr + chunk, len - chunk) } else { r };
        }
        if range.rtype.reserved() {
            release_host(vaddr, len);
            self.ranges.remove(vaddr, len);
            return OK;
        }
        let ok = match range.rtype {
            RangeType::Pooled => self.pooled.query(vaddr, len).is_some() && self.pooled.release(vaddr, len),
            RangeType::Direct => self.physical.unmap(vaddr, len),
            RangeType::Flexible => {
                if self.flexible.covers(vaddr, len) {
                    self.flexible.forget(vaddr, len);
                    true
                } else {
                    false
                }
            }
            _ => true,
        };
        if !ok {
            return EACCES;
        }
        release_host(vaddr, len);
        self.ranges.remove(vaddr, len);
        if range.rtype.committed() {
            free_callback(vaddr, len);
        }
        if range.rtype == RangeType::Pooled {
            self.pooled.subtract_committed(len);
        }
        OK
    }

    /// munmap: every mapping in the range goes; holes are skipped and a
    /// range with nothing mapped is 0, as FreeBSD's vm_map_delete.
    fn unmap_memory(&mut self, vaddr: u64, len: u64) -> HleResult {
        if len == 0 || u64::MAX - vaddr < len {
            return EINVAL;
        }
        for p in self.ranges.pieces(vaddr, len) {
            let r = self.unmap_range(p.start, p.size);
            if r != OK {
                return r;
            }
        }
        OK
    }

    /// Map flexible memory; `*addr_in_out` is updated (0 on failure).
    fn map_flexible(&mut self, addr_in_out: &mut u64, len: u64, prot: i32, flags: u64, name: Option<&str>) -> HleResult {
        let supported = MAP_FIXED | MAP_NO_OVERWRITE | MAP_DMEM_COMPAT | MAP_UNKNOWN_8000 | MAP_NO_COALESCE | MAP_ALIGNMENT_MASK;
        if len == 0 || len % PAGE_SIZE != 0 {
            return EINVAL;
        }
        // A fixed map (no no-overwrite) over bytes the budget already
        // charges replaces them, so they count as available for it.
        let replaced = if flags as u32 as u64 & MAP_FIXED != 0 && flags as u32 as u64 & MAP_NO_OVERWRITE == 0 {
            self.flexible.charged_overlap(*addr_in_out, len)
        } else {
            0
        };
        if len > self.flexible.available() + replaced {
            return ENOMEM;
        }
        let Some(name) = name else { return EFAULT };
        if name.len() >= NAME_LEN {
            return ENAMETOOLONG;
        }
        let flags = flags as u32 as u64;
        let shift = (flags & MAP_ALIGNMENT_MASK) >> 24;
        if flags & !supported != 0 || (shift != 0 && !(14..=31).contains(&shift)) {
            return EINVAL;
        }
        let align = if shift != 0 { 1u64 << shift } else { PAGE_SIZE };
        if !valid_protection(prot) {
            return EINVAL;
        }
        let in_addr = *addr_in_out;
        let mut out = 0u64;
        let mut reserved_target = false;
        if flags & MAP_FIXED != 0 {
            if in_addr == 0 || in_addr % PAGE_SIZE != 0 || in_addr % align != 0 {
                return EINVAL;
            }
            if flags & MAP_NO_OVERWRITE != 0 && self.ranges.has_overlap(in_addr, len) {
                return ENOMEM;
            }
            if self.ranges.query_span(in_addr, len).is_some_and(|rs| rs.iter().all(|r| r.rtype == RangeType::Reserved)) {
                reserved_target = true;
                out = in_addr;
            }
            if !reserved_target && self.replace_fixed_range_with_reserved(in_addr, len) {
                reserved_target = true;
                out = in_addr;
            }
        } else if self.hint_in_reservation(in_addr, len, align) {
            // A hint that lies wholly inside a reservation maps there, as a
            // fixed request would.
            reserved_target = true;
            out = in_addr;
        } else {
            let search = if in_addr != 0 { in_addr } else { GUEST_DEFAULT_MAP_BASE };
            out = self.find_free_range(search, len, align);
        }
        *addr_in_out = out;
        if out == 0 {
            return ENOMEM;
        }
        let Some(host) = self.host.as_ref() else { return ENOMEM };
        if !self.flexible.map(host, out, len, prot, name) {
            if !reserved_target {
                release_host(out, len);
            }
            return ENOMEM;
        }
        let no_coalesce = flags & MAP_NO_COALESCE != 0;
        let published = if reserved_target {
            self.ranges.replace_span(out, len, RangeType::Reserved, 0, prot, 0, RangeType::Flexible, name, no_coalesce)
        } else {
            self.ranges.add(out, len, 0, prot, 0, RangeType::Flexible, name, no_coalesce)
        };
        if !published {
            self.flexible.unmap(host, out, len);
            return EBUSY;
        }
        alloc_callback(out, len);
        OK
    }

    /// sceKernelReleaseFlexibleMemory: the bytes return to the flexible
    /// budget while the range stays mapped with its protection and
    /// contents (see [`FlexibleMemory::release`]).
    fn release_flexible(&mut self, addr: u64, len: u64) -> HleResult {
        if addr == 0 || addr % PAGE_SIZE != 0 || len % PAGE_SIZE != 0 || len > u64::MAX - addr {
            return EINVAL;
        }
        if len != 0 {
            self.flexible.release(addr, len);
        }
        OK
    }

    /// Map direct memory; `*addr` is updated (0 on failure).
    #[allow(clippy::too_many_arguments)]
    fn map_direct(&mut self, addr: &mut u64, len: u64, prot: i32, flags: u64, phys: i64, alignment: u64, name: &str, mtype: Option<i32>) -> HleResult {
        if len == 0
            || len % PAGE_SIZE != 0
            || phys < 0
            || phys as u64 % PAGE_SIZE != 0
            || (alignment != 0 && !alignment.is_power_of_two() && alignment % PAGE_SIZE != 0)
        {
            return EINVAL;
        }
        if prot & PROT_CPU_EXEC != 0 {
            return EACCES;
        }
        if !valid_protection(prot) {
            return EINVAL;
        }
        let phys = phys as u64;
        if !self.physical.can_map_direct(phys, len) {
            return ENOMEM;
        }
        let Some(host) = self.host.as_ref() else { return ENOMEM };
        let fixed = flags & MAP_FIXED != 0;
        let in_addr = *addr;
        let mut out = 0u64;
        let mut reserved_target = false;
        let mut replaced: Vec<AllocatedBlock> = Vec::new();
        let mut held_new = false;
        if fixed {
            if in_addr == 0 || in_addr % PAGE_SIZE != 0 || (alignment != 0 && in_addr % alignment != 0) {
                return EINVAL;
            }
            if flags & MAP_NO_OVERWRITE != 0 && self.ranges.has_overlap(in_addr, len) {
                return ENOMEM;
            }
            let span = self.ranges.query_span(in_addr, len);
            if span.as_ref().is_some_and(|rs| rs.iter().all(|r| r.rtype == RangeType::Direct)) {
                // Replace direct views in place: the new view atomically
                // replaces the old ones.
                let mut covered = 0;
                for m in &self.physical.mappings {
                    let s = in_addr.max(m.map_vaddr);
                    let e = (in_addr + len).min(m.map_vaddr + m.map_size);
                    if s >= e {
                        continue;
                    }
                    let mut old = m.clone();
                    old.start_addr += s - old.map_vaddr;
                    old.map_vaddr = s;
                    old.size = e - s;
                    old.map_size = old.size;
                    covered += old.map_size;
                    replaced.push(old);
                }
                if covered != len {
                    return EBUSY;
                }
                for old in &replaced {
                    self.physical.unmap(old.map_vaddr, old.map_size);
                }
                if !host.map_backing(in_addr, len, phys, prot) {
                    for old in &replaced {
                        host.map_backing(old.map_vaddr, old.map_size, old.start_addr, old.prot);
                        self.physical.map(old.map_vaddr, old.start_addr, old.map_size, old.prot, &old.name, old.memory_type);
                    }
                    return ENOMEM;
                }
                out = in_addr;
            } else {
                if span.is_some_and(|rs| rs.iter().all(|r| r.rtype == RangeType::Reserved)) {
                    reserved_target = true;
                }
                if !reserved_target && self.replace_fixed_range_with_reserved(in_addr, len) {
                    reserved_target = true;
                }
                if !reserved_target {
                    return ENOMEM;
                }
                let host = self.host.as_ref().unwrap();
                if host.map_backing(in_addr, len, phys, prot) {
                    out = in_addr;
                }
            }
        } else {
            let alignment = if alignment != 0 { alignment } else { PAGE_SIZE };
            if self.hint_in_reservation(in_addr, len, alignment) {
                reserved_target = true;
                if host.map_backing(in_addr, len, phys, prot) {
                    out = in_addr;
                }
            }
            if !reserved_target {
                out = self.find_free_range(in_addr, len, alignment);
                if out != 0 {
                    held_new = true;
                    let host = self.host.as_ref().unwrap();
                    if !host.map_backing(out, len, phys, prot) {
                        release_host(out, len);
                        out = 0;
                    }
                }
            }
        }
        *addr = out;
        if out == 0 {
            return ENOMEM;
        }
        let memory_type = mtype.unwrap_or_else(|| self.physical.find(phys, false).map_or(0, |b| b.memory_type));
        self.physical.map(out, phys, len, prot, name, memory_type);
        let published = if !replaced.is_empty() || reserved_target {
            let expected = if replaced.is_empty() { RangeType::Reserved } else { RangeType::Direct };
            self.ranges.replace_span(out, len, expected, phys, prot, memory_type, RangeType::Direct, name, false)
        } else {
            self.ranges.add(out, len, phys, prot, memory_type, RangeType::Direct, name, false)
        };
        if !published {
            self.physical.unmap(out, len);
            if held_new {
                release_host(out, len);
            } else {
                hold_fixed(out, len);
            }
            return EBUSY;
        }
        alloc_callback(out, len);
        OK
    }

    fn protect_memory(&mut self, addr: u64, len: u64, prot: i32) -> HleResult {
        let aligned = addr & !(PAGE_SIZE - 1);
        let page_off = addr - aligned;
        let Some(total) = len.checked_add(page_off).and_then(|l| align_up(l, PAGE_SIZE)) else { return EINVAL };
        if total == 0 {
            return OK;
        }
        // Any combination of the defined bits, 0x40/0x80/0x100/0x200 alone
        // included, is accepted.
        if prot & !PROT_DEFINED != 0 {
            return EINVAL;
        }
        let Some(host) = self.host.as_ref() else { return EACCES };
        let pieces = self.ranges.pieces(aligned, total);
        if pieces.is_empty() {
            // Host memory outside the range map that guest code still
            // protects (e.g. a loader image made GPU-readable).
            if ps5vm::ranges::covered(aligned, total) {
                return if host.protect(aligned, total, prot) { OK } else { EINVAL };
            }
            return EINVAL;
        }
        // An unmapped start is EINVAL. Further on, as FreeBSD's
        // vm_map_protect: holes and reservations (guard entries) are
        // skipped, every committed part is protected.
        if pieces[0].start != aligned {
            return EINVAL;
        }
        let committed: Vec<Range> = pieces.into_iter().filter(|r| r.rtype.committed()).collect();
        for r in &committed {
            if !host.protect(r.start, r.size, prot) {
                return EACCES;
            }
        }
        for r in &committed {
            match r.rtype {
                RangeType::Direct => self.physical.protect_mapping(r.start, r.size, prot),
                RangeType::Flexible => self.flexible.protect(r.start, r.size, prot),
                _ => {}
            }
            self.ranges.protect(r.start, r.size, prot);
        }
        OK
    }

    /// sceKernelMtypeprotect: protect as sceKernelMprotect, then give the
    /// direct mappings in the range the memory type `mtype`.
    fn mtype_protect(&mut self, addr: u64, len: u64, mtype: i32, prot: i32) -> HleResult {
        let r = self.protect_memory(addr, len, prot);
        if r != OK {
            return r;
        }
        let aligned = addr & !(PAGE_SIZE - 1);
        let Some(total) = (len + (addr - aligned)).checked_next_multiple_of(PAGE_SIZE) else { return r };
        for p in self.ranges.pieces(aligned, total).into_iter().filter(|p| matches!(p.rtype, RangeType::Direct | RangeType::Pooled)) {
            self.ranges.edit(p.start, p.size, |r| r.memory_type = mtype);
        }
        r
    }

    fn release_direct_internal(&mut self, start: u64, len: u64, skip_gaps: bool) -> HleResult {
        if self.pooled.release_expansion(start, len) {
            self.physical.release_pool_expansion(start, len);
            return OK;
        }
        let Some(span) = self.physical.allocated_span(start, len, skip_gaps) else { return EACCES };
        let aliases = self.physical.find_mappings(start, len);
        let Some(host) = self.host.as_ref() else { return EACCES };
        for a in &aliases {
            host.unmap_backing(a.map_vaddr, a.map_size);
            self.physical.unmap(a.map_vaddr, a.map_size);
        }
        for a in &aliases {
            release_host(a.map_vaddr, a.map_size);
            self.ranges.remove(a.map_vaddr, a.map_size);
        }
        for b in &span {
            self.physical.release(b.start_addr, b.size);
        }
        host.zero_backing(start, len);
        for a in &aliases {
            free_callback(a.map_vaddr, a.map_size);
        }
        OK
    }

    fn reserve_virtual_range(&mut self, addr: &mut u64, len: u64, flags: u64, alignment: u64) -> HleResult {
        if len == 0 || len % PAGE_SIZE != 0 {
            return EINVAL;
        }
        if alignment != 0 && !alignment.is_power_of_two() && alignment % PAGE_SIZE != 0 {
            return EINVAL;
        }
        let in_addr = *addr;
        let mut out = 0;
        let mut added = false;
        if flags & MAP_FIXED != 0 {
            if in_addr == 0 || in_addr % PAGE_SIZE != 0 {
                return EINVAL;
            }
            if flags & MAP_NO_OVERWRITE != 0 && self.ranges.has_overlap(in_addr, len) {
                return ENOMEM;
            }
            if self.replace_fixed_range_with_reserved(in_addr, len) {
                out = in_addr;
                added = true;
            }
        } else {
            let alignment = if alignment != 0 { alignment } else { PAGE_SIZE };
            out = self.find_free_range(in_addr, len, alignment);
        }
        if out == 0 {
            return ENOMEM;
        }
        if !added && !self.ranges.add(out, len, 0, 0, 0, RangeType::Reserved, "anon", false) {
            release_host(out, len);
            return EBUSY;
        }
        *addr = out;
        OK
    }

    fn pool_commit(&mut self, vaddr: u64, len: u64, mtype: i32, prot: i32) -> HleResult {
        if vaddr == 0 || len == 0 || len % 0x10000 != 0 || prot & PROT_CPU_EXEC != 0 || !valid_protection(prot) {
            return EINVAL;
        }
        let Some(old) = self.ranges.query(vaddr, 0) else { return EACCES };
        if old.rtype != RangeType::PoolReserved || len > old.size - (vaddr - old.start) {
            return EACCES;
        }
        let Some(maps) = self.pooled.allocate(vaddr, len) else { return ENOMEM };
        let Some(host) = self.host.as_ref() else { return ENOMEM };
        for (i, m) in maps.iter().enumerate() {
            if !host.map_backing(m.vaddr, m.size, m.phys, prot) {
                for done in &maps[..i] {
                    host.unmap_backing(done.vaddr, done.size);
                }
                self.pooled.release(vaddr, len);
                return ENOMEM;
            }
        }
        if !self.ranges.replace_span(vaddr, len, RangeType::PoolReserved, 0, prot, mtype, RangeType::Pooled, &old.name, false) {
            let host = self.host.as_ref().unwrap();
            for m in &maps {
                host.unmap_backing(m.vaddr, m.size);
            }
            self.pooled.release(vaddr, len);
            return EBUSY;
        }
        alloc_callback(vaddr, len);
        self.pooled.committed += len;
        OK
    }

    fn pool_decommit_range(&mut self, vaddr: u64, len: u64) -> HleResult {
        let Some(old) = self.ranges.query(vaddr, 0) else { return EACCES };
        let chunk = len.min(old.size - (vaddr - old.start));
        if old.rtype == RangeType::PoolReserved {
            return if chunk < len { self.pool_decommit_range(vaddr + chunk, len - chunk) } else { OK };
        }
        if old.rtype != RangeType::Pooled {
            return EACCES;
        }
        if chunk < len {
            let r = self.pool_decommit_range(vaddr, chunk);
            return if r == OK { self.pool_decommit_range(vaddr + chunk, len - chunk) } else { r };
        }
        if self.pooled.query(vaddr, len).is_none() {
            return EACCES;
        }
        let Some(host) = self.host.as_ref() else { return EACCES };
        host.unmap_backing(vaddr, len);
        self.pooled.release(vaddr, len);
        self.ranges.remove(vaddr, len);
        self.ranges.add(vaddr, len, 0, 0, 0, RangeType::PoolReserved, &old.name, false);
        free_callback(vaddr, len);
        self.pooled.subtract_committed(len);
        OK
    }

    fn pool_decommit(&mut self, vaddr: u64, len: u64) -> HleResult {
        if vaddr == 0 || len == 0 || len % 0x10000 != 0 || u64::MAX - vaddr < len {
            return EINVAL;
        }
        let end = vaddr + len;
        let mut scan = vaddr;
        while scan < end {
            let Some(r) = self.ranges.query(scan, 0).filter(|r| r.rtype.pooled()) else { return EACCES };
            let next = end.min(r.end());
            if next <= scan {
                return EACCES;
            }
            scan = next;
        }
        self.pool_decommit_range(vaddr, len)
    }
}

// ---------------------------------------------------------------------------
// Internal API for other modules
// ---------------------------------------------------------------------------

/// Allocate system memory (the native driver's), outside the title's direct
/// memory: a block of `len` bytes aligned to `alignment`.
pub(crate) fn allocate_system_memory(len: u64, alignment: u64, mtype: i32) -> Result<u64, HleResult> {
    lock().physical.alloc_system(len, alignment, mtype).ok_or(ENOMEM)
}

/// Allocate direct memory; `automatic` marks AMM memory that
/// [`map_automatic_memory`] can hand out.
pub(crate) fn allocate_direct_memory(search_start: i64, search_end: i64, len: u64, alignment: u64, mtype: i32, automatic: bool) -> Result<u64, HleResult> {
    if search_start < 0 || search_end <= search_start || len == 0 || len % PAGE_SIZE != 0 || (alignment != 0 && alignment % PAGE_SIZE != 0) {
        return Err(EINVAL);
    }
    let kind = if automatic { AllocationKind::Automatic } else { AllocationKind::Direct };
    lock().physical.alloc(search_start as u64, search_end as u64, len, alignment, mtype, kind).ok_or(EAGAIN)
}

/// Allocate direct memory in `[start, end)` (internal callers).
pub(crate) fn allocate_direct(search_start: u64, search_end: u64, len: u64, align: u64, mtype: i32) -> Option<u64> {
    allocate_direct_memory(search_start as i64, search_end as i64, len, align, mtype, false).ok()
}

/// Map direct memory at `requested` (exactly there with MAP_FIXED).
pub(crate) fn map_direct_at(requested: u64, len: u64, prot: i32, flags: u64, phys: u64, align: u64) -> Result<u64, HleResult> {
    let mut addr = requested;
    match lock().map_direct(&mut addr, len, prot, flags, phys as i64, align, "", None) {
        OK => Ok(addr),
        e => Err(e),
    }
}

/// Map direct memory under `name` (`requested` is a hint without MAP_FIXED).
pub(crate) fn map_named_direct(requested: u64, len: u64, prot: i32, flags: u64, phys: u64, align: u64, name: &str) -> Result<u64, HleResult> {
    let mut addr = requested;
    match lock().map_direct(&mut addr, len, prot, flags, phys as i64, align, name, None) {
        OK => Ok(addr),
        e => Err(e),
    }
}

/// Release direct memory (and every mapping of it).
pub(crate) fn release_direct(start: u64, len: u64) -> HleResult {
    lock().release_direct_internal(start, len, false)
}

/// Map `size` bytes of automatic (AMM) memory at `vaddr` in the extended
/// memory window. Entry point of the AMM "map auto" command; the AMPR
/// command processor (ampr/amm.rs) still keeps its own automatic pool.
#[allow(dead_code)]
pub(crate) fn map_automatic_memory(vaddr: u64, size: u64, mtype: i32, prot: i32) -> HleResult {
    if size == 0
        || vaddr % PAGE_SIZE != 0
        || size % PAGE_SIZE != 0
        || vaddr < EXTENDED_MEMORY_BASE
        || vaddr >= EXTENDED_MEMORY_BASE + EXTENDED_MEMORY_SIZE
        || size > EXTENDED_MEMORY_BASE + EXTENDED_MEMORY_SIZE - vaddr
    {
        return EINVAL;
    }
    let mut st = lock();
    let Some(ranges) = st.physical.reserve_automatic(size) else { return EAGAIN };
    let mut mapped = 0;
    for &(offset, length) in &ranges {
        let mut a = vaddr + mapped;
        let r = st.map_direct(&mut a, length, prot, MAP_FIXED, offset as i64, PAGE_SIZE, "", Some(mtype));
        if r != OK {
            if mapped != 0 {
                st.unmap_memory(vaddr, mapped);
            }
            st.physical.restore_automatic(&ranges);
            return r;
        }
        mapped += length;
    }
    OK
}

/// munmap.
pub(crate) fn unmap(addr: u64, len: u64) -> HleResult {
    lock().unmap_memory(addr, len)
}

/// A direct mapping clipped to a queried span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DirectPiece {
    pub start: u64,
    pub size: u64,
    /// Direct memory offset backing `start`.
    pub phys: u64,
    pub prot: i32,
    pub mtype: i32,
}

/// The direct mappings overlapping `[addr, addr + len)`, clipped to it, in
/// address order; holes and other mapping kinds are absent.
pub(crate) fn direct_pieces(addr: u64, len: u64) -> Vec<DirectPiece> {
    lock()
        .ranges
        .pieces(addr, len)
        .into_iter()
        .filter(|r| r.rtype == RangeType::Direct)
        .map(|r| DirectPiece { start: r.start, size: r.size, phys: r.offset, prot: r.protection, mtype: r.memory_type })
        .collect()
}

/// Map direct memory exactly at `addr` with memory type `mtype`.
pub(crate) fn map_direct_typed_at(addr: u64, len: u64, prot: i32, phys: u64, align: u64, mtype: i32) -> HleResult {
    let mut a = addr;
    lock().map_direct(&mut a, len, prot, MAP_FIXED, phys as i64, align, "", Some(mtype))
}

/// sceKernelMprotect for internal callers.
pub(crate) fn protect(addr: u64, len: u64, prot: i32) -> HleResult {
    lock().protect_memory(addr, len, prot)
}

/// sceKernelMtypeprotect for internal callers.
pub(crate) fn mtype_protect(addr: u64, len: u64, mtype: i32, prot: i32) -> HleResult {
    lock().mtype_protect(addr, len, mtype, prot)
}

/// Size of the committed span at `addr` (0 when `addr` is not committed).
pub fn clamp_range_size(addr: u64, size: u64) -> u64 {
    lock().ranges.clamp_range_size(addr, size)
}

/// Record a loaded module segment (page-rounded) as a code range, so the
/// protection and range queries answer for addresses inside module images.
pub fn register_module_segment(start: u64, size: u64, prot: i32, name: &str) {
    let (Some(end), s) = (align_up(start.saturating_add(size), PAGE_SIZE), start & !(PAGE_SIZE - 1)) else { return };
    let mut g = MEM.lock().unwrap_or_else(|e| e.into_inner());
    match g.as_mut() {
        Some(st) => {
            st.ranges.add(s, end - s, 0, prot, 0, RangeType::Code, name, false);
        }
        None => PENDING_CODE.lock().unwrap_or_else(|e| e.into_inner()).push((s, end - s, prot, name.to_string())),
    }
}

/// Forget the code ranges of an unloaded module segment.
pub fn unregister_module_segment(start: u64, size: u64) {
    let (Some(end), s) = (align_up(start.saturating_add(size), PAGE_SIZE), start & !(PAGE_SIZE - 1)) else { return };
    let mut g = MEM.lock().unwrap_or_else(|e| e.into_inner());
    match g.as_mut() {
        Some(st) => {
            st.ranges.remove(s, end - s);
        }
        None => PENDING_CODE.lock().unwrap_or_else(|e| e.into_inner()).retain(|c| c.0 != s),
    }
}

/// Record a guest thread stack so queries report it.
/// Record a guest thread stack (mapped by the thread code) as a stack range.
pub(crate) fn register_stack(base: u64, size: u64) -> bool {
    let added = lock().ranges.add(base, size, 0, PROT_CPU_READ | PROT_CPU_WRITE, 0, RangeType::Stack, "stack", false);
    if added {
        alloc_callback(base, size);
    }
    added
}

pub(crate) fn unregister_stack(base: u64, size: u64) {
    lock().ranges.remove(base, size);
    free_callback(base, size);
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

fn read_guest_name(ctx: &CpuContext, ptr: u64) -> Option<String> {
    if ptr == 0 {
        return None;
    }
    let mut out = Vec::new();
    for i in 0..=NAME_LEN as u64 {
        let b = ctx.read_bytes(ptr + i, 1)?[0];
        if b == 0 {
            break;
        }
        out.push(b);
    }
    Some(String::from_utf8_lossy(&out).into_owned())
}

fn cb_get_direct_memory_size(_ctx: &mut CpuContext) -> HleResult {
    direct_size() as i64
}

/// sceKernelAvailableDirectMemorySize(start, end, align, phys_out*, size_out*)
fn cb_available_direct_memory_size(ctx: &mut CpuContext) -> HleResult {
    let (start, end, align, phys_out, size_out) = (ctx.rdi() as i64, ctx.rsi() as i64, ctx.rdx(), ctx.rcx(), ctx.r8());
    // A null output is allowed: the value is just not written back
    // (measured on hardware).
    if phys_out != 0 {
        ctx.write_u64(phys_out, 0);
    }
    if size_out != 0 {
        ctx.write_u64(size_out, 0);
    }
    if start < 0 || end < 0 {
        return EINVAL;
    }
    if end <= start {
        return ENOMEM;
    }
    // A non-power-of-two alignment is refused; zero means the page
    // alignment (measured on hardware).
    if align != 0 && align & (align - 1) != 0 {
        return EINVAL;
    }
    let Some((p, s)) = lock().physical.available(start as u64, end as u64, align) else { return ENOMEM };
    if (phys_out != 0 && !ctx.write_u64(phys_out, p)) || (size_out != 0 && !ctx.write_u64(size_out, s)) {
        return EFAULT;
    }
    OK
}

/// sceKernelGetPageTableStats(cpu_total*, cpu_available*, gpu_total*, gpu_available*)
fn cb_get_page_table_stats(ctx: &mut CpuContext) -> HleResult {
    let (ct, ca, gt, ga) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    if ct == 0 || ca == 0 || gt == 0 || ga == 0 {
        return EFAULT;
    }
    let (cpu_used, gpu_used) = {
        let st = lock();
        (st.ranges.count_page_table_entries(false), st.ranges.count_page_table_entries(true))
    };
    let avail = |used: u64| PAGE_TABLE_POOL_ENTRIES - used.min(PAGE_TABLE_POOL_ENTRIES as u64) as i32;
    ctx.write_u32(ct, PAGE_TABLE_POOL_ENTRIES as u32);
    ctx.write_u32(gt, PAGE_TABLE_POOL_ENTRIES as u32);
    ctx.write_u32(ca, avail(cpu_used) as u32);
    ctx.write_u32(ga, avail(gpu_used) as u32);
    OK
}

/// sceKernelDirectMemoryQuery(offset, flags, info*, info_size): { start, end, type }
fn cb_direct_memory_query(ctx: &mut CpuContext) -> HleResult {
    let (offset, flags, info, info_size) = (ctx.rdi() as i64, ctx.rsi() as i32, ctx.rdx(), ctx.rcx());
    if offset < 0 || (flags != 0 && flags != 1) || info_size != DIRECT_QUERY_INFO_SIZE || info == 0 {
        return EINVAL;
    }
    let offset = offset as u64;
    let result = {
        let st = lock();
        let blocks = &st.physical.physical;
        let mut current = blocks.range(offset..).next().map(|(_, b)| b.clone());
        if let Some((_, prev)) = blocks.range(..=offset).next_back() {
            if offset < prev.start_addr + prev.size {
                current = Some(prev.clone());
            }
        }
        // System memory is not the title's direct memory.
        let current = current.filter(|b| b.start_addr < direct_size());
        match current {
            Some(b) if flags == 1 || (offset >= b.start_addr && offset < b.start_addr + b.size) => {
                let mut end = b.start_addr + b.size;
                for (_, f) in blocks.range(end..) {
                    if f.start_addr != end || f.memory_type != b.memory_type {
                        break;
                    }
                    end += f.size;
                }
                Some((b.start_addr, end, b.memory_type))
            }
            _ if flags == 1 && offset < direct_size() => Some((direct_size(), direct_size(), 0)),
            _ => None,
        }
    };
    let Some((s, e, t)) = result else { return EACCES };
    if !ctx.write_u64(info, s) || !ctx.write_u64(info + 8, e) || !ctx.write_u32(info + 16, t as u32) {
        return EFAULT;
    }
    OK
}

/// sceKernelAllocateDirectMemory(start, end, len, align, type, out*)
fn cb_allocate_direct_memory(ctx: &mut CpuContext) -> HleResult {
    let (start, end, len, align, mtype, out) = (ctx.rdi() as i64, ctx.rsi() as i64, ctx.rdx(), ctx.rcx(), ctx.r8() as i32, ctx.r9());
    if out == 0 {
        return EINVAL;
    }
    match allocate_direct_memory(start, end, len, align, mtype, false) {
        Ok(p) => {
            if ctx.write_u64(out, p) { OK } else { EFAULT }
        }
        Err(e) => {
            if let Some(n) = note_failure("sceKernelAllocateDirectMemory") {
                log::warn!("[mem] sceKernelAllocateDirectMemory({start:#x}..{end:#x}, len {len:#x}, align {align:#x}, type {mtype}) failed: {:#x} (failure #{n})", e as u32);
            }
            e
        }
    }
}

/// sceKernelAllocateMainDirectMemory(len, align, type, out*)
fn cb_allocate_main_direct_memory(ctx: &mut CpuContext) -> HleResult {
    let (len, align, mtype, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as i32, ctx.rcx());
    if out == 0 {
        return EINVAL;
    }
    match allocate_direct_memory(0, direct_size() as i64, len, align, mtype, false) {
        Ok(p) => {
            if ctx.write_u64(out, p) { OK } else { EFAULT }
        }
        Err(e) => e,
    }
}

/// Arguments of the direct memory release calls: a page-aligned range that
/// lies inside direct memory (a zero length is accepted and does nothing).
fn validate_release(start: i64, len: u64) -> HleResult {
    if start < 0 || start as u64 % PAGE_SIZE != 0 || len % PAGE_SIZE != 0 {
        return EINVAL;
    }
    if len != 0 && (len > direct_size() || start as u64 > direct_size() - len) {
        return EINVAL;
    }
    OK
}

/// sceKernelReleaseDirectMemory(start, len)
fn cb_release_direct_memory(ctx: &mut CpuContext) -> HleResult {
    let (start, len) = (ctx.rdi() as i64, ctx.rsi());
    let v = validate_release(start, len);
    if v != OK {
        return v;
    }
    if len != 0 {
        // The unchecked release frees the allocated parts and skips the
        // unallocated gaps between them.
        let _ = lock().release_direct_internal(start as u64, len, true);
    }
    OK
}

/// sceKernelCheckedReleaseDirectMemory(start, len)
fn cb_checked_release_direct_memory(ctx: &mut CpuContext) -> HleResult {
    let (start, len) = (ctx.rdi() as i64, ctx.rsi());
    let v = validate_release(start, len);
    if v != OK || len == 0 {
        return v;
    }
    match lock().release_direct_internal(start as u64, len, false) {
        EACCES => ENOENT,
        r => r,
    }
}

#[allow(clippy::too_many_arguments)]
fn map_direct_cb(ctx: &mut CpuContext, inout: u64, len: u64, prot: i32, flags: u64, phys: i64, align: u64, name: &str, mtype: Option<i32>) -> HleResult {
    if inout == 0 {
        return EFAULT;
    }
    let Some(mut addr) = ctx.read_u64(inout) else { return EFAULT };
    let hint = addr;
    let r = lock().map_direct(&mut addr, len, prot, flags, phys, align, name, mtype);
    if r != OK {
        if let Some(n) = note_failure("sceKernelMapDirectMemory") {
            log::warn!("[mem] direct mapping {name:?} of {len:#x} bytes at phys {phys:#x} (hint {hint:#x}, flags {flags:#x}, prot {prot:#x}) failed: {:#x} (failure #{n})", r as u32);
        }
    }
    if !ctx.write_u64(inout, addr) {
        return EFAULT;
    }
    r
}

/// sceKernelMapDirectMemory(inout*, len, prot, flags, phys, align)
fn cb_map_direct_memory(ctx: &mut CpuContext) -> HleResult {
    let (a, l, p, f, s, al) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as i32, ctx.rcx(), ctx.r8() as i64, ctx.r9());
    map_direct_cb(ctx, a, l, p, f, s, al, "", None)
}

/// sceKernelMapNamedDirectMemory(inout*, len, prot, flags, phys, align, name)
fn cb_map_named_direct_memory(ctx: &mut CpuContext) -> HleResult {
    let (a, l, p, f, s, al) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as i32, ctx.rcx(), ctx.r8() as i64, ctx.r9());
    let rsp = ctx.get(crate::context::CpuRegister::Rsp);
    let name_ptr = ctx.read_u64(rsp + 8).unwrap_or(0);
    let Some(name) = read_guest_name(ctx, name_ptr) else { return EFAULT };
    if name.len() >= NAME_LEN {
        return ENAMETOOLONG;
    }
    map_direct_cb(ctx, a, l, p, f, s, al, &name, None)
}

/// sceKernelMapDirectMemory2(inout*, len, type, prot, flags, phys, align)
fn cb_map_direct_memory2(ctx: &mut CpuContext) -> HleResult {
    let rsp = ctx.get(crate::context::CpuRegister::Rsp);
    let align = ctx.read_u64(rsp + 8).unwrap_or(0);
    let (a, l, t, p, f, s) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as i32, ctx.rcx() as i32, ctx.r8(), ctx.r9() as i64);
    map_direct_cb(ctx, a, l, p, f, s, align, "", (t != -1).then_some(t))
}

fn map_flexible_cb(ctx: &mut CpuContext, name: Option<String>) -> HleResult {
    let (inout, len, prot, flags) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as i32, ctx.rcx());
    if inout == 0 {
        return EFAULT;
    }
    let Some(mut addr) = ctx.read_u64(inout) else { return EFAULT };
    let hint = addr;
    let r = lock().map_flexible(&mut addr, len, prot, flags, name.as_deref());
    if r != OK {
        if let Some(n) = note_failure("sceKernelMapFlexibleMemory") {
            log::warn!("[mem] flexible mapping {name:?} of {len:#x} bytes (hint {hint:#x}, flags {flags:#x}, prot {prot:#x}) failed: {:#x} (failure #{n})", r as u32);
        }
    }
    if !ctx.write_u64(inout, addr) {
        return EFAULT;
    }
    r
}

/// sceKernelMapFlexibleMemory(inout*, len, prot, flags)
fn cb_map_flexible_memory(ctx: &mut CpuContext) -> HleResult {
    map_flexible_cb(ctx, Some(String::new()))
}

/// sceKernelMapNamedFlexibleMemory(inout*, len, prot, flags, name)
fn cb_map_named_flexible_memory(ctx: &mut CpuContext) -> HleResult {
    let name = read_guest_name(ctx, ctx.r8());
    map_flexible_cb(ctx, name)
}

/// sceKernelMunmap(addr, len): the munmap system call with the error as an
/// SCE code, so the range rounds to whole pages like POSIX munmap (the
/// start down, the end up); a zero length or an overflowing range is
/// EINVAL.
fn cb_munmap(ctx: &mut CpuContext) -> HleResult {
    let (addr, len) = (ctx.rdi(), ctx.rsi());
    match page_range(addr, len) {
        Some((start, end)) if len != 0 && end > start => unmap(start, end - start),
        _ => EINVAL,
    }
}

/// sceKernelReleaseFlexibleMemory(addr, len)
fn cb_release_flexible_memory(ctx: &mut CpuContext) -> HleResult {
    lock().release_flexible(ctx.rdi(), ctx.rsi())
}

/// sceKernelReserveVirtualRange(inout*, len, flags, align)
fn cb_reserve_virtual_range(ctx: &mut CpuContext) -> HleResult {
    let (inout, len, flags, align) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u32 as u64, ctx.rcx());
    if inout == 0 {
        return EINVAL;
    }
    let Some(mut addr) = ctx.read_u64(inout) else { return EFAULT };
    let hint = addr;
    let r = lock().reserve_virtual_range(&mut addr, len, flags, align);
    if r != OK {
        if let Some(n) = note_failure("sceKernelReserveVirtualRange") {
            log::warn!("[mem] sceKernelReserveVirtualRange(hint {hint:#x}, len {len:#x}, flags {flags:#x}, align {align:#x}) failed: {:#x} (failure #{n})", r as u32);
        }
    }
    if r == OK && !ctx.write_u64(inout, addr) {
        return EFAULT;
    }
    r
}

// FreeBSD errno values for the POSIX-named exports.
const ERRNO_ENOMEM: i32 = 12;
const ERRNO_EBUSY: i32 = 16;
const ERRNO_EINVAL: i32 = 22;
const ERRNO_ENOTSUP: i32 = 45;

/// Finish a POSIX-named call: the value, or -1 with the thread's errno set.
fn posix_result(ctx: &CpuContext, r: Result<i64, i32>) -> HleResult {
    match r {
        Ok(v) => v,
        Err(errno) => {
            crate::kernel::set_errno(ctx, errno);
            -1
        }
    }
}

/// FreeBSD mmap(addr, len, prot, flags, fd, offset) of anonymous memory,
/// kept as a runtime range. `MAP_ALIGNED(n)` / `MAP_ALIGNED_SUPER` in
/// bits 24..31 of the flags set the placement alignment; a non-fixed
/// address is a search hint. Returns the address or an errno.
fn posix_mmap(requested: u64, len: u64, prot: i32, flags: u64, fd: i32, offset: u64) -> Result<u64, i32> {
    const POSIX_MAP_FIXED: u64 = 0x10;
    const POSIX_MAP_ANON: u64 = 0x1000;
    const MAP_ALIGNMENT_SHIFT: u64 = 24;
    const MAP_ALIGNED_SUPER: u64 = 1;
    const SUPERPAGE_SIZE: u64 = 2 * 1024 * 1024;
    let flags = flags as u32 as u64;
    if len == 0 || prot & !(PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC) != 0 {
        return Err(ERRNO_EINVAL);
    }
    let Some(len) = align_up(len, PAGE_SIZE) else { return Err(ERRNO_EINVAL) };
    if flags & POSIX_MAP_ANON == 0 {
        log::warn!("[mem] mmap of file descriptor {fd} is not supported");
        return Err(ERRNO_ENOTSUP);
    }
    if fd != -1 || offset != 0 {
        return Err(ERRNO_EINVAL);
    }
    let alignment = match (flags >> MAP_ALIGNMENT_SHIFT) & 0xff {
        0 => PAGE_SIZE,
        MAP_ALIGNED_SUPER => SUPERPAGE_SIZE,
        shift if (14..64).contains(&shift) => 1u64 << shift,
        _ => return Err(ERRNO_EINVAL),
    };
    let mut st = lock();
    let addr = if flags & POSIX_MAP_FIXED != 0 {
        if requested == 0 || requested % PAGE_SIZE != 0 || !st.replace_fixed_range_with_reserved(requested, len) {
            return Err(ERRNO_EINVAL);
        }
        st.ranges.remove(requested, len);
        requested
    } else {
        let a = st.find_free_range(requested, len, alignment);
        if a == 0 {
            return Err(ERRNO_ENOMEM);
        }
        a
    };
    let ok = st.host.as_ref().is_some_and(|h| h.commit(addr, len, prot));
    if !ok || !st.ranges.add(addr, len, 0, prot, 0, RangeType::Runtime, "anon", false) {
        release_host(addr, len);
        return Err(ERRNO_ENOMEM);
    }
    drop(st);
    alloc_callback(addr, len);
    Ok(addr)
}

/// POSIX mmap(addr, len, prot, flags, fd, offset): the address, or -1 and
/// errno.
fn cb_mmap(ctx: &mut CpuContext) -> HleResult {
    let r = posix_mmap(ctx.rdi(), ctx.rsi(), ctx.rdx() as i32, ctx.rcx(), ctx.r8() as i32, ctx.r9());
    posix_result(ctx, r.map(|a| a as i64))
}

/// sceKernelMmap(addr, len, prot, flags, fd, offset, res*): the address goes
/// to `*res`; errors are kernel codes.
fn cb_kernel_mmap(ctx: &mut CpuContext) -> HleResult {
    let rsp = ctx.get(crate::context::CpuRegister::Rsp);
    let Some(res) = ctx.read_u64(rsp + 8).filter(|&r| r != 0) else { return EFAULT };
    match posix_mmap(ctx.rdi(), ctx.rsi(), ctx.rdx() as i32, ctx.rcx(), ctx.r8() as i32, ctx.r9()) {
        Ok(a) => {
            if ctx.write_u64(res, a) { OK } else { EFAULT }
        }
        Err(errno) => kerr(errno as u32),
    }
}

/// Protection bits sceKernelMprotect keeps; the others are dropped, as
/// FreeBSD's sys_mprotect masks with VM_PROT_ALL (a title protects with
/// 0xfffffff2).
const PROT_DEFINED: i32 = 0x3F7;

/// sceKernelMprotect(addr, len, prot): a zero length is 0.
fn cb_mprotect(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 {
        return OK;
    }
    lock().protect_memory(ctx.rdi(), ctx.rsi(), ctx.rdx() as i32 & PROT_DEFINED)
}

/// POSIX mprotect(addr, len, prot): 0, or -1 and errno.
fn cb_posix_mprotect(ctx: &mut CpuContext) -> HleResult {
    let (addr, len, prot) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as i32);
    let r = if len == 0 {
        Ok(0)
    } else if addr == 0 || len > u64::MAX - addr {
        Err(ERRNO_EINVAL)
    } else if lock().protect_memory(addr, len, prot & PROT_DEFINED) != OK {
        Err(ERRNO_EINVAL)
    } else {
        Ok(0)
    };
    posix_result(ctx, r)
}

/// POSIX munmap(addr, len): the kernel rounds the start down and the end
/// up to page boundaries instead of rejecting an unaligned start (a zero
/// length is EINVAL), 0, or -1 and errno.
fn cb_posix_munmap(ctx: &mut CpuContext) -> HleResult {
    let (addr, len) = (ctx.rdi(), ctx.rsi());
    let r = match page_range(addr, len) {
        // The kernel returns EINVAL for a zero length before rounding, so
        // an empty result is only reachable through `len != 0`.
        Some((start, end)) if len != 0 && end > start => {
            if unmap(start, end - start) == OK { Ok(0) } else { Err(ERRNO_EINVAL) }
        }
        _ => Err(ERRNO_EINVAL),
    };
    posix_result(ctx, r)
}

/// Page-rounded `[start, end)` of `[addr, addr+len)`; None on overflow.
fn page_range(addr: u64, len: u64) -> Option<(u64, u64)> {
    let mask = PAGE_SIZE - 1;
    let last = addr.checked_add(len).filter(|&l| l <= u64::MAX - mask)?;
    Some((addr & !mask, (last + mask) & !mask))
}

/// sceKernelMlock(addr, len): guest memory is never paged out, so locking
/// only checks that every page is committed memory.
fn kernel_mlock(addr: u64, len: u64) -> HleResult {
    let Some((start, end)) = page_range(addr, len) else { return EINVAL };
    if start == end || lock().guest_span(start, end - start, true) { OK } else { ENOMEM }
}

/// sceKernelMunlock(addr, len): every page must be mapped (reservations
/// included).
fn kernel_munlock(addr: u64, len: u64) -> HleResult {
    let Some((start, end)) = page_range(addr, len) else { return EINVAL };
    if start == end || lock().guest_span(start, end - start, false) { OK } else { ENOMEM }
}

fn cb_kernel_mlock(ctx: &mut CpuContext) -> HleResult {
    kernel_mlock(ctx.rdi(), ctx.rsi())
}

fn cb_kernel_munlock(ctx: &mut CpuContext) -> HleResult {
    kernel_munlock(ctx.rdi(), ctx.rsi())
}

/// Kernel code -> POSIX result (errno is the low 16 bits).
fn kernel_to_posix_result(r: HleResult) -> Result<i64, i32> {
    if r == OK { Ok(0) } else { Err((r as u32 & 0xffff) as i32) }
}

/// POSIX mlock(addr, len)
fn cb_posix_mlock(ctx: &mut CpuContext) -> HleResult {
    let r = kernel_to_posix_result(kernel_mlock(ctx.rdi(), ctx.rsi()));
    posix_result(ctx, r)
}

/// POSIX munlock(addr, len)
fn cb_posix_munlock(ctx: &mut CpuContext) -> HleResult {
    let r = kernel_to_posix_result(kernel_munlock(ctx.rdi(), ctx.rsi()));
    posix_result(ctx, r)
}

/// FreeBSD msync(addr, len, flags): there are no file-backed mappings, so
/// it only validates. MS_INVALIDATE (alone or combined) is EBUSY, checked
/// before the range; a range that is not wholly mapped (reservations count
/// as mapped) is ENOMEM.
fn posix_msync(addr: u64, len: u64, flags: i32) -> Result<i64, i32> {
    const MS_INVALIDATE: i32 = 0x2;
    let mask = PAGE_SIZE - 1;
    let start = addr & !mask;
    let size = len.checked_add(addr - start).and_then(|s| s.checked_add(mask)).map(|s| s & !mask);
    let Some(size) = size.filter(|&s| s <= u64::MAX - start) else { return Err(ERRNO_EINVAL) };
    // The console refuses MS_INVALIDATE with EBUSY before any range check
    // (measured on hardware: it fails on an unmapped range too, and the
    // combined async+invalidate flag fails the same way).
    if flags & MS_INVALIDATE != 0 {
        return Err(ERRNO_EBUSY);
    }
    // A zero-length request still names the page at `addr`.
    if lock().guest_span(start, size.max(1), false) { Ok(0) } else { Err(ERRNO_ENOMEM) }
}

fn cb_msync(ctx: &mut CpuContext) -> HleResult {
    let r = posix_msync(ctx.rdi(), ctx.rsi(), ctx.rdx() as i32);
    posix_result(ctx, r)
}

/// sceKernelMsync(addr, len, flags): msync with kernel error codes.
fn cb_kernel_msync(ctx: &mut CpuContext) -> HleResult {
    match posix_msync(ctx.rdi(), ctx.rsi(), ctx.rdx() as i32) {
        Ok(_) => OK,
        Err(errno) => kerr(errno as u32),
    }
}

/// FreeBSD madvise(addr, len, advice): advice up to MADV_CORE is accepted
/// (it never changes contents here); the range must lie below the end of
/// the user address space.
fn posix_madvise(addr: u64, len: u64, advice: i32) -> Result<i64, i32> {
    const MADV_CORE: i32 = 9;
    const MADV_PROTECT: i32 = 10;
    const USER_ADDRESS_END: u64 = 0x8000_0000_0000;
    // FreeBSD's kern_madvise handles MADV_PROTECT before the address
    // check, ignoring the range: it protects the process from the OOM
    // killer, which is taken to succeed.
    if advice == MADV_PROTECT {
        return Ok(0);
    }
    if !(0..=MADV_CORE).contains(&advice) || addr > USER_ADDRESS_END || len > USER_ADDRESS_END - addr {
        return Err(ERRNO_EINVAL);
    }
    Ok(0)
}

/// mincore(addr, len, vec): FreeBSD 11's kern_mincore order. The address
/// is rounded down to its page; an overflowing range or any unmapped page
/// is ENOMEM (a zero length looks at the start page only), then a null
/// vector is EFAULT. Each page reports MINCORE_INCORE (1) when committed
/// and 0 when only reserved.
fn posix_mincore(ctx: &CpuContext, addr: u64, len: u64, vec: u64) -> Result<i64, i32> {
    const ERRNO_EFAULT: i32 = 14;
    let start = addr & !(PAGE_SIZE - 1);
    let Some(end) = addr.checked_add(len).and_then(|e| align_up(e, PAGE_SIZE)) else { return Err(ERRNO_ENOMEM) };
    let pages = {
        let st = lock();
        if !st.guest_span(start, (end - start).max(PAGE_SIZE), false) {
            return Err(ERRNO_ENOMEM);
        }
        if len == 0 {
            return Ok(0);
        }
        (start..end).step_by(PAGE_SIZE as usize).map(|p| st.ranges.query(p, 0).map_or(1, |r| r.rtype.committed() as u8)).collect::<Vec<u8>>()
    };
    if vec == 0 || !ctx.write_bytes(vec, &pages) {
        return Err(ERRNO_EFAULT);
    }
    Ok(0)
}

fn cb_mincore(ctx: &mut CpuContext) -> HleResult {
    let r = posix_mincore(ctx, ctx.rdi(), ctx.rsi(), ctx.rdx());
    posix_result(ctx, r)
}

/// sceKernelEnableDmemAliasing(): direct memory may always be mapped more
/// than once here.
fn cb_enable_dmem_aliasing(_ctx: &mut CpuContext) -> HleResult {
    OK
}

fn cb_madvise(ctx: &mut CpuContext) -> HleResult {
    let r = posix_madvise(ctx.rdi(), ctx.rsi(), ctx.rdx() as i32);
    posix_result(ctx, r)
}

/// sceKernelMtypeprotect(addr, len, type, prot): a zero length is 0; the
/// direct mappings of the range take the memory type.
fn cb_mtypeprotect(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 {
        return OK;
    }
    lock().mtype_protect(ctx.rdi(), ctx.rsi(), ctx.rdx() as i32, ctx.rcx() as i32)
}

/// sceKernelVirtualQuery(addr, flags, info*, info_size)
fn cb_virtual_query(ctx: &mut CpuContext) -> HleResult {
    let (addr, flags, info, info_size) = (ctx.rdi(), ctx.rsi() as i32, ctx.rdx(), ctx.rcx());
    if info == 0 || info_size != VIRTUAL_QUERY_INFO_SIZE || (flags != 0 && flags != 1) {
        return EINVAL;
    }
    let (r, prt) = {
        let st = lock();
        let Some(r) = st.ranges.query(addr, flags) else { return EACCES };
        let prt = st.in_prt_aperture(addr);
        (r, prt)
    };
    let mut p = [0u8; VIRTUAL_QUERY_INFO_SIZE as usize];
    p[0..8].copy_from_slice(&r.start.to_le_bytes());
    p[8..16].copy_from_slice(&r.end().to_le_bytes());
    p[16..24].copy_from_slice(&r.offset.to_le_bytes());
    p[24..28].copy_from_slice(&reported_protection(r.protection).to_le_bytes());
    p[28..32].copy_from_slice(&r.memory_type.to_le_bytes());
    // The flags are one-bit fields of an unsigned int followed by the
    // name: with the SysV layout the console's compiler uses, the bits
    // take byte 32 and the name starts at byte 33 (the GPU mask id at 65).
    let mut bits = 0u8;
    bits |= (r.rtype == RangeType::Flexible) as u8;
    bits |= ((r.rtype == RangeType::Direct) as u8) << 1;
    bits |= ((r.rtype == RangeType::Stack) as u8) << 2;
    bits |= (r.rtype.pooled() as u8) << 3;
    bits |= (r.rtype.committed() as u8) << 4;
    bits |= (prt as u8) << 5;
    p[32] = bits;
    let nb = r.name.as_bytes();
    let n = nb.len().min(NAME_LEN - 1);
    p[33..33 + n].copy_from_slice(&nb[..n]);
    if !ctx.write_bytes(info, &p) {
        return EFAULT;
    }
    OK
}

/// sceKernelQueryMemoryProtection(addr, start*, end*, prot*)
fn cb_query_memory_protection(ctx: &mut CpuContext) -> HleResult {
    let (addr, start_out, end_out, prot_out) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    if addr == 0 {
        return EINVAL;
    }
    let Some(r) = lock().ranges.query(addr, 0) else { return EACCES };
    if (start_out != 0 && !ctx.write_u64(start_out, r.start))
        || (end_out != 0 && !ctx.write_u64(end_out, r.end()))
        || (prot_out != 0 && !ctx.write_u32(prot_out, reported_protection(r.protection) as u32))
    {
        return EFAULT;
    }
    OK
}

/// sceKernelSetVirtualRangeName(addr, len, name)
fn cb_set_virtual_range_name(ctx: &mut CpuContext) -> HleResult {
    let (addr, len) = (ctx.rdi(), ctx.rsi());
    let Some(name) = read_guest_name(ctx, ctx.rdx()) else { return EFAULT };
    if name.len() >= NAME_LEN {
        return ENAMETOOLONG;
    }
    let mut st = lock();
    st.physical.set_name(addr, len, &name);
    st.flexible.set_name(addr, len, &name);
    st.ranges.rename(addr, len, &name);
    OK
}

/// sceKernelClearVirtualRangeName(addr, len)
fn cb_clear_virtual_range_name(ctx: &mut CpuContext) -> HleResult {
    let (addr, len) = (ctx.rdi(), ctx.rsi());
    let mut st = lock();
    st.physical.set_name(addr, len, "");
    st.flexible.set_name(addr, len, "");
    st.ranges.rename(addr, len, "");
    OK
}

/// sceKernelIsStack(addr, start*, end*)
fn cb_is_stack(ctx: &mut CpuContext) -> HleResult {
    let (addr, start_out, end_out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let Some(r) = lock().ranges.query(addr, 0) else { return EACCES };
    let (s, e) = if r.rtype == RangeType::Stack { (r.start, r.end()) } else { (0, 0) };
    if (start_out != 0 && !ctx.write_u64(start_out, s)) || (end_out != 0 && !ctx.write_u64(end_out, e)) {
        return EFAULT;
    }
    OK
}

fn cb_available_flexible_memory_size(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return EINVAL;
    }
    let avail = lock().flexible.available();
    if ctx.write_u64(out, avail) { OK } else { EFAULT }
}

fn cb_configured_flexible_memory_size(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return EINVAL;
    }
    if ctx.write_u64(out, flexible_size()) { OK } else { EFAULT }
}

fn cb_getpagesize(_ctx: &mut CpuContext) -> HleResult {
    PAGE_SIZE as i64
}

/// sceKernelSetPrtAperture(index, addr, len)
fn cb_set_prt_aperture(ctx: &mut CpuContext) -> HleResult {
    let (index, addr, len) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    if !(0..=PRT_APERTURE_MAX_INDEX).contains(&index) {
        return EINVAL;
    }
    if len != 0
        && (addr == 0 || addr % PRT_PAGE_SIZE != 0 || len % PRT_PAGE_SIZE != 0 || addr < PRT_APERTURE_START || addr > PRT_APERTURE_END || len > PRT_APERTURE_END - addr)
    {
        return EINVAL;
    }
    lock().prt[index as usize] = if len == 0 { PrtAperture::default() } else { PrtAperture { address: addr, size: len } };
    OK
}

/// sceKernelGetPrtAperture(index, addr*, len*)
fn cb_get_prt_aperture(ctx: &mut CpuContext) -> HleResult {
    let (index, addr_out, len_out) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    if !(0..=PRT_APERTURE_MAX_INDEX).contains(&index) {
        return EINVAL;
    }
    if addr_out == 0 || len_out == 0 {
        return EFAULT;
    }
    let a = lock().prt[index as usize];
    if !ctx.write_u64(addr_out, a.address) || !ctx.write_u64(len_out, a.size) {
        return EFAULT;
    }
    OK
}

/// sceKernelBatchMap2(entries, count, processed*, flags). Entry (32 bytes):
/// start u64, offset u64, length u64, prot u8, type u8, reserved i16, op i32.
fn batch_map(ctx: &mut CpuContext, entries: u64, count: i32, processed_out: u64, flags: u64) -> HleResult {
    const OP_MAP_DIRECT: i32 = 0;
    const OP_UNMAP: i32 = 1;
    const OP_PROTECT: i32 = 2;
    const OP_MAP_FLEXIBLE: i32 = 3;
    const OP_TYPE_PROTECT: i32 = 4;
    if entries == 0 || count < 0 {
        return EINVAL;
    }
    let mut processed = 0;
    let mut st = lock();
    while processed < count {
        let e = entries + processed as u64 * 32;
        let (Some(start), Some(offset), Some(length), Some(pt), Some(op)) =
            (ctx.read_u64(e), ctx.read_u64(e + 8), ctx.read_u64(e + 16), ctx.read_u32(e + 24), ctx.read_u32(e + 28))
        else {
            break;
        };
        let op = op as i32;
        let prot = (pt & 0xFF) as i32;
        if length == 0 || !(OP_MAP_DIRECT..=OP_TYPE_PROTECT).contains(&op) {
            if let Some(n) = note_failure("sceKernelBatchMap invalid entry") {
                log::warn!("[mem] sceKernelBatchMap entry {processed} of {count}: op {op}, start {start:#x}, length {length:#x}: invalid, batch stopped (failure #{n})");
            }
            break;
        }
        let r = match op {
            OP_MAP_DIRECT => {
                let mut a = start;
                let r = st.map_direct(&mut a, length, prot, flags, offset as i64, 0, "anon", None);
                ctx.write_u64(e, a);
                r
            }
            OP_UNMAP => st.unmap_memory(start, length),
            OP_PROTECT | OP_TYPE_PROTECT => st.protect_memory(start, length, prot),
            OP_MAP_FLEXIBLE => {
                let mut a = start;
                let r = st.map_flexible(&mut a, length, prot, flags, Some("anon"));
                ctx.write_u64(e, a);
                r
            }
            _ => EINVAL,
        };
        if r != OK {
            if let Some(n) = note_failure("sceKernelBatchMap") {
                log::warn!("[mem] sceKernelBatchMap entry {processed} of {count}: op {op}, start {start:#x}, length {length:#x}, prot {prot:#x}: failed {:#x} (failure #{n})", r as u32);
            }
            if processed_out != 0 {
                ctx.write_u32(processed_out, processed as u32);
            }
            return r;
        }
        processed += 1;
    }
    drop(st);
    if processed_out != 0 {
        ctx.write_u32(processed_out, processed as u32);
    }
    if processed == count { OK } else { EINVAL }
}

/// sceKernelBatchMap(entries, count, processed*)
fn cb_batch_map(ctx: &mut CpuContext) -> HleResult {
    let (e, c, p) = (ctx.rdi(), ctx.rsi() as i32, ctx.rdx());
    batch_map(ctx, e, c, p, MAP_FIXED)
}

/// sceKernelBatchMap2(entries, count, processed*, flags)
fn cb_batch_map2(ctx: &mut CpuContext) -> HleResult {
    let (e, c, p, f) = (ctx.rdi(), ctx.rsi() as i32, ctx.rdx(), ctx.rcx() as u32 as u64);
    batch_map(ctx, e, c, p, f)
}

/// sceKernelMemoryPoolExpand(start, end, len, align, phys_out*)
fn cb_memory_pool_expand(ctx: &mut CpuContext) -> HleResult {
    const POOL_PAGE: u64 = 0x10000;
    let (start, end, len, align, out) = (ctx.rdi() as i64, ctx.rsi() as i64, ctx.rdx(), ctx.rcx(), ctx.r8());
    if start < 0
        || end <= start
        || len == 0
        || len % POOL_PAGE != 0
        || out == 0
        || (align != 0 && (!align.is_power_of_two() || align % POOL_PAGE != 0))
    {
        return EINVAL;
    }
    if ((end - start) as u64) < len {
        return ENOMEM;
    }
    let align = if align != 0 { align } else { POOL_PAGE };
    let phys = {
        let mut st = lock();
        let Some(p) = st.physical.alloc(start as u64, end as u64, len, align, 0, AllocationKind::Pooled) else { return ENOMEM };
        st.pooled.expand(p, len);
        p
    };
    if ctx.write_u64(out, phys) { OK } else { EFAULT }
}

/// sceKernelMemoryPoolReserve(addr_in, len, align, flags, addr_out*)
fn cb_memory_pool_reserve(ctx: &mut CpuContext) -> HleResult {
    const POOL_RESERVE_ALIGN: u64 = 0x20_0000;
    let (addr_in, len, align, flags, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx() as u32 as u64, ctx.r8());
    if out == 0 || len == 0 || len % POOL_RESERVE_ALIGN != 0 {
        return EINVAL;
    }
    if align != 0 && (!align.is_power_of_two() || align % POOL_RESERVE_ALIGN != 0) {
        return EINVAL;
    }
    let mut st = lock();
    let mut addr = addr_in;
    let r = st.reserve_virtual_range(&mut addr, len, flags, if align != 0 { align } else { POOL_RESERVE_ALIGN });
    if r != OK {
        return r;
    }
    let Some(res) = st.ranges.query(addr, 0).filter(|r| r.start == addr && r.size == len && r.rtype.reserved()) else { return EBUSY };
    if !st.ranges.replace_span(addr, len, RangeType::Reserved, 0, 0, 0, RangeType::PoolReserved, &res.name, false) {
        return EBUSY;
    }
    drop(st);
    if ctx.write_u64(out, addr) { OK } else { EFAULT }
}

/// sceKernelMemoryPoolCommit(addr, len, type, prot, flags)
fn cb_memory_pool_commit(ctx: &mut CpuContext) -> HleResult {
    lock().pool_commit(ctx.rdi(), ctx.rsi(), ctx.rdx() as i32, ctx.rcx() as i32)
}

/// sceKernelMemoryPoolDecommit(addr, len, flags)
fn cb_memory_pool_decommit(ctx: &mut CpuContext) -> HleResult {
    lock().pool_decommit(ctx.rdi(), ctx.rsi())
}

/// sceKernelMemoryPoolBatch(entries, count, processed*, flags). Entry (32
/// bytes): op u32, flags u32, then addr u64, len u64, prot u8, type u8.
fn cb_memory_pool_batch(ctx: &mut CpuContext) -> HleResult {
    const OP_COMMIT: u32 = 1;
    const OP_DECOMMIT: u32 = 2;
    const OP_PROTECT: u32 = 3;
    const OP_TYPE_PROTECT: u32 = 4;
    let (entries, count, processed_out) = (ctx.rdi(), ctx.rsi() as i32, ctx.rdx());
    if entries == 0 || count < 0 {
        return EINVAL;
    }
    let mut processed = 0;
    let mut result = OK;
    let mut st = lock();
    while processed < count {
        let e = entries + processed as u64 * 32;
        let (Some(op), Some(addr), Some(len), Some(pt)) = (ctx.read_u32(e), ctx.read_u64(e + 8), ctx.read_u64(e + 16), ctx.read_bytes(e + 24, 2))
        else {
            result = EFAULT;
            break;
        };
        let (prot, mtype) = (pt[0] as i32, pt[1] as i32);
        result = match op {
            OP_COMMIT => st.pool_commit(addr, len, mtype, prot),
            OP_DECOMMIT => st.pool_decommit(addr, len),
            OP_PROTECT => st.protect_memory(addr, len, prot),
            // The entry's memory type is applied with the protection.
            OP_TYPE_PROTECT => st.mtype_protect(addr, len, mtype, prot),
            _ => EINVAL,
        };
        if result != OK {
            break;
        }
        processed += 1;
    }
    drop(st);
    if processed_out != 0 {
        ctx.write_u32(processed_out, processed as u32);
    }
    result
}

/// sceKernelMemoryPoolGetBlockStats(stats*, size): available/allocated
/// 64 KiB blocks (flushed; nothing is cached).
fn cb_memory_pool_get_block_stats(ctx: &mut CpuContext) -> HleResult {
    let (out, size) = (ctx.rdi(), ctx.rsi());
    if out == 0 && size != 0 {
        return EFAULT;
    }
    let (committed, available) = {
        let st = lock();
        (st.pooled.committed, st.pooled.available())
    };
    let mut stats = [0u8; 16];
    stats[0..4].copy_from_slice(&((available / 0x10000) as i32).to_le_bytes());
    stats[8..12].copy_from_slice(&((committed / 0x10000) as i32).to_le_bytes());
    let n = (size as usize).min(16);
    if n != 0 && !ctx.write_bytes(out, &stats[..n]) {
        return EFAULT;
    }
    OK
}

const EXPORTS: &[(&str, &str, crate::registry::HleCallback)] = &[
    ("pO96TwzOm5E", "sceKernelGetDirectMemorySize", cb_get_direct_memory_size),
    ("C0f7TJcbfac", "sceKernelAvailableDirectMemorySize", cb_available_direct_memory_size),
    ("tZ2yplY8MBY", "sceKernelGetPageTableStats", cb_get_page_table_stats),
    ("aNz11fnnzi4", "sceKernelAvailableFlexibleMemorySize", cb_available_flexible_memory_size),
    ("n1-v6FgU7MQ", "sceKernelConfiguredFlexibleMemorySize", cb_configured_flexible_memory_size),
    ("rTXw65xmLIA", "sceKernelAllocateDirectMemory", cb_allocate_direct_memory),
    ("B+vc2AO2Zrc", "sceKernelAllocateMainDirectMemory", cb_allocate_main_direct_memory),
    ("MBuItvba6z8", "sceKernelReleaseDirectMemory", cb_release_direct_memory),
    ("hwVSPCmp5tM", "sceKernelCheckedReleaseDirectMemory", cb_checked_release_direct_memory),
    ("L-Q3LEjIbgA", "sceKernelMapDirectMemory", cb_map_direct_memory),
    ("NcaWUxfMNIQ", "sceKernelMapNamedDirectMemory", cb_map_named_direct_memory),
    ("BQQniolj9tQ", "sceKernelMapDirectMemory2", cb_map_direct_memory2),
    ("mL8NDH86iQI", "sceKernelMapNamedFlexibleMemory", cb_map_named_flexible_memory),
    ("IWIBBdTHit4", "sceKernelMapFlexibleMemory", cb_map_flexible_memory),
    ("4h6F1LLbTiw", "sceKernelMapNamedFlexibleMemoryInternal", cb_map_named_flexible_memory),
    ("teiItL2boFw", "sceKernelReleaseFlexibleMemory", cb_release_flexible_memory),
    ("cQke9UuBQOk", "sceKernelMunmap", cb_munmap),
    ("UqDGjXA5yUM", "munmap", cb_posix_munmap),
    ("7oxv3PPCumo", "sceKernelReserveVirtualRange", cb_reserve_virtual_range),
    ("PGhQHd-dzv8", "sceKernelMmap", cb_kernel_mmap),
    ("BPE9s9vQQXo", "mmap", cb_mmap),
    ("vSMAm3cxYTY", "sceKernelMprotect", cb_mprotect),
    ("YQOfxL4QfeU", "mprotect", cb_posix_mprotect),
    ("3k6kx-zOOSQ", "sceKernelMlock", cb_kernel_mlock),
    ("xQIIfJ860sk", "sceKernelMunlock", cb_kernel_munlock),
    ("mTBZfEal2Bw", "mlock", cb_posix_mlock),
    ("OG4RsDwLguo", "munlock", cb_posix_munlock),
    ("DkNebxmJyVI", "sceKernelMsync", cb_kernel_msync),
    ("tZY4+SZNFhA", "msync", cb_msync),
    ("Jahsnh4KKkg", "madvise", cb_madvise),
    ("+hFjyohmOU0", "mincore", cb_mincore),
    ("usHTMoFoBTM", "sceKernelEnableDmemAliasing", cb_enable_dmem_aliasing),
    ("9bfdLIyuwCY", "sceKernelMtypeprotect", cb_mtypeprotect),
    ("rVjRvHJ0X6c", "sceKernelVirtualQuery", cb_virtual_query),
    ("WFcfL2lzido", "sceKernelQueryMemoryProtection", cb_query_memory_protection),
    ("BHouLQzh0X0", "sceKernelDirectMemoryQuery", cb_direct_memory_query),
    ("DGMG3JshrZU", "sceKernelSetVirtualRangeName", cb_set_virtual_range_name),
    ("mkgXxsoxWHg", "sceKernelClearVirtualRangeName", cb_clear_virtual_range_name),
    ("yDBwVAolDgg", "sceKernelIsStack", cb_is_stack),
    ("k+AXqu2-eBc", "getpagesize", cb_getpagesize),
    ("2SKEx6bSq-4", "sceKernelBatchMap", cb_batch_map),
    ("kBJzF8x4SyE", "sceKernelBatchMap2", cb_batch_map2),
    ("BohYr-F7-is", "sceKernelSetPrtAperture", cb_set_prt_aperture),
    ("L0v2Go5jOuM", "sceKernelGetPrtAperture", cb_get_prt_aperture),
    ("qCSfqDILlns", "sceKernelMemoryPoolExpand", cb_memory_pool_expand),
    ("pU-QydtGcGY", "sceKernelMemoryPoolReserve", cb_memory_pool_reserve),
    ("Vzl66WmfLvk", "sceKernelMemoryPoolCommit", cb_memory_pool_commit),
    ("LXo1tpFqJGs", "sceKernelMemoryPoolDecommit", cb_memory_pool_decommit),
    ("YN878uKRBbE", "sceKernelMemoryPoolBatch", cb_memory_pool_batch),
    ("bvD+95Q6asU", "sceKernelMemoryPoolGetBlockStats", cb_memory_pool_get_block_stats),
    ("BC+OG5m9+bw", "sceKernelGetDirectMemoryType", cb_get_direct_memory_type),
];

/// Map `len` bytes of the host descriptor `fd` (shared, from offset 0) at a
/// free guest range near `hint`, kept as a runtime range named `name`.
/// Returns the address or a kernel error.
#[cfg(unix)]
pub(crate) fn map_shared_host_fd(hint: u64, len: u64, prot: i32, fd: std::os::fd::RawFd, name: &str) -> Result<u64, HleResult> {
    let mut st = lock();
    let addr = st.find_free_range(hint, len, PAGE_SIZE);
    if addr == 0 {
        return Err(ENOMEM);
    }
    // SAFETY: a free range the memory manager owns.
    let r = unsafe { libc::mmap(addr as *mut libc::c_void, len as usize, host_prot(prot), libc::MAP_SHARED | libc::MAP_FIXED, fd, 0) };
    if r == libc::MAP_FAILED {
        return Err(ENOMEM);
    }
    ps5vm::ranges::track(addr, len, vm_prot(prot));
    ps5gpu::memory::global().note_mapped(addr, len, host_prot(prot), None);
    if !st.ranges.add(addr, len, 0, prot, 0, RangeType::Runtime, name, false) {
        release_host(addr, len);
        return Err(ENOMEM);
    }
    drop(st);
    alloc_callback(addr, len);
    Ok(addr)
}

/// sceKernelGetDirectMemoryType(offset, int* type, off_t* start, off_t*
/// end): the allocated direct memory block holding `offset` (adjacent
/// blocks of the same type merged, as sceKernelDirectMemoryQuery reports).
fn cb_get_direct_memory_type(ctx: &mut CpuContext) -> HleResult {
    let (offset, type_out, start_out, end_out) = (ctx.rdi() as i64, ctx.rsi(), ctx.rdx(), ctx.rcx());
    if type_out == 0 || start_out == 0 || end_out == 0 {
        return EINVAL;
    }
    if offset < 0 {
        return ENOENT;
    }
    let offset = offset as u64;
    let found = {
        let st = lock();
        let blocks = &st.physical.physical;
        blocks.range(..=offset).next_back().map(|(_, b)| b.clone()).filter(|b| offset < b.start_addr + b.size && b.start_addr < direct_size()).map(|b| {
            let mut end = b.start_addr + b.size;
            for (_, f) in blocks.range(end..) {
                if f.start_addr != end || f.memory_type != b.memory_type {
                    break;
                }
                end += f.size;
            }
            (b.start_addr, end, b.memory_type)
        })
    };
    let Some((s, e, t)) = found else { return ENOENT };
    if ctx.write_u32(type_out, t as u32) && ctx.write_u64(start_out, s) && ctx.write_u64(end_out, e) { OK } else { EFAULT }
}

pub(crate) mod jit;

pub fn register(reg: &mut HleRegistry) {
    // Guest accesses to pages the GPU memory tracker protects are resolved
    // by the tracker before a fault is reported.
    ps5cpu::exec::set_fault_hook(ps5gpu::memory::handle_fault);
    ps5cpu::exec::set_write_protect_hook(ps5gpu::memory::handle_wp_fault);
    ps5cpu::exec::set_describe_hook(ps5gpu::memory::describe_address);
    for &(nid, name, cb) in EXPORTS.iter().chain(jit::JIT_EXPORTS) {
        reg.register_fn(nid, name, "libKernel", cb);
    }
}

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Host address space on Windows
// ---------------------------------------------------------------------------

/// Windows: the physical backing is a pagefile-backed section; guest views
/// and private memory are placeholder-space mappings (see
/// `ps5vm::os::windows`).
#[cfg(windows)]
struct HostSpace {
    section: std::sync::Arc<ps5vm::os::Section>,
    /// The whole backing, read/write (also the GPU's alias).
    alias: u64,
}

#[cfg(windows)]
impl HostSpace {
    fn create() -> Option<Self> {
        let section = ps5vm::os::Section::create(BACKING_BYTES)?;
        let alias = section.map_anywhere(0, BACKING_BYTES)?;
        ps5gpu::memory::global().set_backing_alias(alias, BACKING_BYTES);
        Some(Self { section, alias })
    }

    /// Zero a backing range (Windows cannot return section pages early).
    fn zero_backing(&self, offset: u64, size: u64) -> bool {
        if offset + size > BACKING_BYTES {
            return false;
        }
        // SAFETY: inside the read/write alias of the backing.
        unsafe { std::ptr::write_bytes((self.alias + offset) as *mut u8, 0, size as usize) };
        true
    }

    fn map_backing(&self, addr: u64, len: u64, offset: u64, prot: i32) -> bool {
        if !ps5vm::os::space::map_view(addr, len, &self.section, offset, vm_prot(prot)) {
            return false;
        }
        log::debug!("[mem] view {addr:#x}+{len:#x} backing {offset:#x} prot {prot:#x}");
        ps5vm::ranges::track(addr, len, vm_prot(prot));
        ps5gpu::memory::global().note_mapped(addr, len, host_prot(prot), Some(offset));
        true
    }

    fn unmap_backing(&self, addr: u64, len: u64) -> bool {
        hold_fixed(addr, len)
    }

    fn commit(&self, addr: u64, len: u64, prot: i32) -> bool {
        if !ps5vm::os::space::commit(addr, len, vm_prot(prot)) {
            return false;
        }
        log::debug!("[mem] private {addr:#x}+{len:#x} prot {prot:#x}");
        ps5vm::ranges::track(addr, len, vm_prot(prot));
        ps5gpu::memory::global().note_mapped(addr, len, host_prot(prot), None);
        true
    }

    fn protect(&self, addr: u64, len: u64, prot: i32) -> bool {
        let ok = ps5gpu::memory::global().note_protected(addr, len, host_prot(prot)) || ps5vm::os::protect(addr, len, vm_prot(prot));
        if ok {
            ps5vm::ranges::track(addr, len, vm_prot(prot));
        }
        ok
    }
}

#[cfg(windows)]
fn hold_fixed(addr: u64, len: u64) -> bool {
    log::debug!("[mem] hold {addr:#x}+{len:#x}");
    ps5gpu::memory::global().note_unmapped(addr, len);
    if !ps5vm::os::space::hold(addr, len) {
        return false;
    }
    ps5vm::ranges::track(addr, len, ps5vm::MemProtect::NONE);
    true
}

#[cfg(windows)]
fn hold_noreplace(addr: u64, len: u64) -> bool {
    if !ps5vm::os::space::hold_noreplace(addr, len) {
        return false;
    }
    ps5vm::ranges::track(addr, len, ps5vm::MemProtect::NONE);
    true
}

/// Give a held range back: on Windows it stays reserved, inaccessible.
#[cfg(windows)]
fn release_host(addr: u64, len: u64) {
    ps5gpu::memory::global().note_unmapped(addr, len);
    ps5vm::os::space::release(addr, len);
    ps5vm::ranges::untrack(addr, len);
}

#[cfg(windows)]
fn host_conflict_end(addr: u64, len: u64) -> Option<u64> {
    ps5vm::os::conflict_end(addr, len)
}

/// Map `len` bytes of the host file `fd` (shared, from offset 0) at a free
/// guest range near `hint`, kept as a runtime range named `name`.
#[cfg(windows)]
pub(crate) fn map_shared_host_fd(hint: u64, len: u64, prot: i32, fd: crate::host::fs::RawFd, name: &str) -> Result<u64, HleResult> {
    let section = ps5vm::os::Section::from_file(fd as isize as _, len).ok_or(ENOMEM)?;
    let mut st = lock();
    let addr = st.find_free_range(hint, len, PAGE_SIZE);
    if addr == 0 {
        return Err(ENOMEM);
    }
    if !ps5vm::os::space::map_view(addr, len, &section, 0, vm_prot(prot)) {
        return Err(ENOMEM);
    }
    ps5vm::ranges::track(addr, len, vm_prot(prot));
    ps5gpu::memory::global().note_mapped(addr, len, host_prot(prot), None);
    if !st.ranges.add(addr, len, 0, prot, 0, RangeType::Runtime, name, false) {
        release_host(addr, len);
        return Err(ENOMEM);
    }
    drop(st);
    alloc_callback(addr, len);
    Ok(addr)
}
