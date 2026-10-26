// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Device-local memory blocks shared by many buffers.
//!
//! Every [`crate::device::Buffer`] used to own a `VkDeviceMemory` of its
//! own: creating or releasing a guest-memory mirror, a pool buffer or a
//! scratch buffer cost a driver allocation (a kernel call) each time, and
//! drivers bound the number of live allocations. Device-local buffers up
//! to [`MAX_SUBALLOCATION`] are instead bound at an offset inside a large
//! block ([`BLOCK_BYTES`]) of their memory type, with device addresses
//! enabled on every block. Free space is kept per block as offset-ordered
//! runs, merged with their neighbours on release (first fit). A block that
//! becomes empty is kept while it is the only empty one of its type, so a
//! burst of releases and creations does not return memory to the driver
//! and take it back again.
//!
//! `RPS5E_BUFFER_SUBALLOC=0` gives every buffer its own allocation.

use ash::vk;
use std::collections::BTreeMap;
use std::sync::Mutex;

/// Bytes of one block.
pub const BLOCK_BYTES: u64 = 256 << 20;
/// Buffers larger than this keep a dedicated allocation.
pub const MAX_SUBALLOCATION: u64 = 64 << 20;

/// Bytes of one image block (images have an arena of their own: only
/// optimal-tiling images share its blocks, so `bufferImageGranularity`
/// never separates neighbours).
pub const IMAGE_BLOCK_BYTES: u64 = 64 << 20;
/// Images larger than this keep a dedicated allocation.
pub const MAX_IMAGE_SUBALLOCATION: u64 = 4 << 20;

/// `RPS5E_IMAGE_SUBALLOC=1`: small images the driver does not want in
/// dedicated memory share blocks (default off). One driver allocation per
/// image costs ~60 us on NVIDIA: a table of thousands of new textures
/// stalled the frame that first bound it.
pub fn images_enabled() -> bool {
    match IMAGES_FORCED.load(std::sync::atomic::Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var("RPS5E_IMAGE_SUBALLOC").is_ok_and(|v| v == "1"))
        }
    }
}

static IMAGES_FORCED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Force image suballocation on or off (tests and benchmarks).
pub fn force_images(on: Option<bool>) {
    IMAGES_FORCED.store(on.map_or(0, |o| if o { 1 } else { 2 }), std::sync::atomic::Ordering::Relaxed);
}

/// `RPS5E_BUFFER_SUBALLOC=0`: one allocation per buffer.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| !std::env::var("RPS5E_BUFFER_SUBALLOC").is_ok_and(|v| v == "0"))
}

/// Where a buffer's memory lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Suballocation {
    pub memory: vk::DeviceMemory,
    pub offset: u64,
    pub size: u64,
    /// Block id (index into the arena's blocks, stable while the block lives).
    block: u64,
}

struct Block {
    id: u64,
    memory_type: u32,
    memory: vk::DeviceMemory,
    size: u64,
    used: u64,
    /// Free runs: offset → length.
    free: BTreeMap<u64, u64>,
}

impl Block {
    fn new(id: u64, memory_type: u32, memory: vk::DeviceMemory, size: u64) -> Block {
        let mut free = BTreeMap::new();
        free.insert(0, size);
        Block { id, memory_type, memory, size, used: 0, free }
    }

    /// First fit of `size` bytes at `align`.
    fn take(&mut self, size: u64, align: u64) -> Option<u64> {
        let (start, len, at) = self.free.iter().find_map(|(&off, &len)| {
            let at = off.next_multiple_of(align);
            (at + size <= off + len).then_some((off, len, at))
        })?;
        self.free.remove(&start);
        if at > start {
            self.free.insert(start, at - start);
        }
        let end = at + size;
        if end < start + len {
            self.free.insert(end, start + len - end);
        }
        self.used += size;
        Some(at)
    }

    /// Return `[offset, offset + size)`, merged with adjacent free runs.
    fn give(&mut self, offset: u64, size: u64) {
        let (mut start, mut end) = (offset, offset + size);
        if let Some((&prev, &len)) = self.free.range(..offset).next_back() {
            if prev + len == offset {
                self.free.remove(&prev);
                start = prev;
            }
        }
        if let Some(&len) = self.free.get(&end) {
            self.free.remove(&end);
            end += len;
        }
        self.free.insert(start, end - start);
        self.used -= size;
    }
}

/// Counters of the arena (summary line).
#[derive(Clone, Copy, Debug, Default)]
pub struct ArenaStats {
    pub blocks: u64,
    pub block_bytes: u64,
    pub used_bytes: u64,
    pub live: u64,
    /// Buffers placed in blocks / blocks allocated / blocks released.
    pub placed: u64,
    pub block_allocations: u64,
    pub block_releases: u64,
}

#[derive(Default)]
struct Inner {
    blocks: Vec<Block>,
    next_id: u64,
    stats: ArenaStats,
}

/// The device's blocks (see the module documentation).
#[derive(Default)]
pub struct Arena {
    inner: Mutex<Inner>,
}

impl Arena {
    /// Place `size` bytes at `align` in a block of `memory_type`, allocating
    /// a new block with `allocate` when none has room.
    pub fn place(&self, memory_type: u32, size: u64, align: u64, allocate: impl FnOnce(u64) -> Result<vk::DeviceMemory, vk::Result>) -> Result<Suballocation, vk::Result> {
        self.place_in(memory_type, size, align, BLOCK_BYTES, allocate)
    }

    /// [`Self::place`] with blocks of `block_bytes`.
    pub fn place_in(&self, memory_type: u32, size: u64, align: u64, block_bytes: u64, allocate: impl FnOnce(u64) -> Result<vk::DeviceMemory, vk::Result>) -> Result<Suballocation, vk::Result> {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let align = align.max(1);
        for b in g.blocks.iter_mut().filter(|b| b.memory_type == memory_type) {
            if b.size - b.used < size {
                continue;
            }
            if let Some(offset) = b.take(size, align) {
                let s = Suballocation { memory: b.memory, offset, size, block: b.id };
                g.stats.placed += 1;
                g.stats.live += 1;
                g.stats.used_bytes += size;
                return Ok(s);
            }
        }
        let memory = allocate(block_bytes)?;
        let id = g.next_id;
        g.next_id += 1;
        let mut b = Block::new(id, memory_type, memory, block_bytes);
        let offset = b.take(size, align).expect("a fresh block holds a suballocation");
        g.blocks.push(b);
        g.stats.blocks += 1;
        g.stats.block_bytes += block_bytes;
        g.stats.block_allocations += 1;
        g.stats.placed += 1;
        g.stats.live += 1;
        g.stats.used_bytes += size;
        Ok(Suballocation { memory, offset, size, block: id })
    }

    /// Release `s`; returns the memory of a block to free when it became
    /// empty while another empty block of its type exists.
    pub fn release(&self, s: Suballocation) -> Option<vk::DeviceMemory> {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let i = g.blocks.iter().position(|b| b.id == s.block)?;
        g.blocks[i].give(s.offset, s.size);
        g.stats.live -= 1;
        g.stats.used_bytes -= s.size;
        if g.blocks[i].used != 0 {
            return None;
        }
        let ty = g.blocks[i].memory_type;
        let other_empty = g.blocks.iter().enumerate().any(|(j, b)| j != i && b.memory_type == ty && b.used == 0);
        if !other_empty {
            return None;
        }
        let b = g.blocks.swap_remove(i);
        g.stats.blocks -= 1;
        g.stats.block_bytes -= b.size;
        g.stats.block_releases += 1;
        Some(b.memory)
    }

    /// Memory of every block (device teardown: no buffer is left).
    pub fn drain(&self) -> Vec<vk::DeviceMemory> {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.blocks.drain(..).map(|b| b.memory).collect()
    }

    pub fn stats(&self) -> ArenaStats {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake(n: u64) -> vk::DeviceMemory {
        use ash::vk::Handle;
        vk::DeviceMemory::from_raw(n)
    }

    #[test]
    fn first_fit_merges_released_runs() {
        let mut b = Block::new(0, 0, fake(1), 1024);
        let a = b.take(100, 64).unwrap();
        let c = b.take(100, 64).unwrap();
        let d = b.take(100, 64).unwrap();
        assert_eq!((a, c, d), (0, 128, 256));
        b.give(c, 100);
        // The gap is reused by a fitting request.
        assert_eq!(b.take(64, 64), Some(128));
        b.give(128, 64);
        b.give(a, 100);
        b.give(d, 100);
        assert_eq!(b.used, 0);
        assert_eq!(b.free.len(), 1);
        assert_eq!(b.free.get(&0), Some(&1024));
    }

    #[test]
    fn arena_keeps_one_empty_block_per_type() {
        let arena = Arena::default();
        let mut next = 10u64;
        let mut alloc = |_: u64| {
            next += 1;
            Ok(fake(next))
        };
        let big = BLOCK_BYTES - 4096;
        let x = arena.place(3, big, 256, &mut alloc).unwrap();
        let y = arena.place(3, big, 256, &mut alloc).unwrap();
        assert_ne!(x.memory, y.memory);
        assert_eq!(arena.stats().blocks, 2);
        // The first block emptied is kept, the second freed.
        assert_eq!(arena.release(x), None);
        assert_eq!(arena.release(y), Some(y.memory));
        assert_eq!(arena.stats().blocks, 1);
        // The kept block serves the next request.
        let z = arena.place(3, 4096, 256, |_| panic!("no new block")).unwrap();
        assert_eq!(z.memory, x.memory);
        // Other memory types get their own blocks.
        let w = arena.place(4, 4096, 256, &mut alloc).unwrap();
        assert_ne!(w.memory, z.memory);
        assert_eq!(arena.drain().len(), 2);
    }

    /// Image blocks: sizes rounded to the alignment (as `Image::new_typed`
    /// places them) leave no padding runs, so first fit looks at one run
    /// however many images a block holds; unrounded sizes leave one
    /// padding run per image.
    #[test]
    fn rounded_sizes_keep_one_free_run() {
        for (size, runs) in [(512u64.next_multiple_of(1024), 1usize), (512, 1000)] {
            let arena = Arena::default();
            let mut placed = Vec::new();
            for _ in 0..1000 {
                placed.push(arena.place_in(7, size, 1024, IMAGE_BLOCK_BYTES, |n| {
                    assert_eq!(n, IMAGE_BLOCK_BYTES);
                    Ok(fake(1))
                }));
            }
            let g = arena.inner.lock().unwrap();
            assert_eq!(g.blocks.len(), 1);
            assert_eq!(g.blocks[0].free.len(), runs, "size {size}");
            drop(g);
            for (k, p) in placed.into_iter().enumerate() {
                assert_eq!(p.unwrap().offset, k as u64 * 1024);
            }
        }
    }
}
