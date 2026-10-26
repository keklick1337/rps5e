// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Reusable resources of the image cache: retired host images kept for
//! images of the same shape, and device scratch buffers for transfers.
//!
//! Render-target memory is reused by surfaces of other formats and sizes
//! every frame; each switch drops one image and creates another. Pooling
//! the retired images makes recreating one a table lookup instead of an
//! image creation plus a dedicated allocation. A pooled image is handed out
//! only once the GPU finished the tick that retired it, and the cache tracks
//! it from `UNDEFINED` again (its first use discards the contents).

use crate::device::{Buffer, GpuDevice, Image, MemoryUse};
use crate::scheduler::Scheduler;
use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;

/// Creation parameters of a host image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PoolKey {
    pub ty: vk::ImageType,
    pub format: vk::Format,
    pub extent: (u32, u32, u32),
    pub levels: u32,
    pub layers: u32,
    pub samples: vk::SampleCountFlags,
    pub usage: vk::ImageUsageFlags,
    pub flags: vk::ImageCreateFlags,
}

/// Retired images kept per key.
const MAX_PER_KEY: usize = 4;
/// Bytes of device memory the pool may hold.
const MAX_POOLED_BYTES: u64 = 512 << 20;

struct Entry {
    /// The tick that last used the image.
    tick: u64,
    bytes: u64,
    /// Insertion order (oldest entries are evicted first).
    order: u64,
    image: Image,
}

#[derive(Default)]
pub struct ImagePool {
    entries: HashMap<PoolKey, Vec<Entry>>,
    bytes: u64,
    order: u64,
    /// Images handed out again.
    pub reused: u64,
}

impl ImagePool {
    /// A retired image of `key` the GPU no longer uses.
    pub fn take(&mut self, sched: &Scheduler, key: &PoolKey) -> Option<Image> {
        let v = self.entries.get_mut(key)?;
        let i = v.iter().position(|e| sched.is_complete(e.tick))?;
        let e = v.swap_remove(i);
        self.bytes -= e.bytes;
        self.reused += 1;
        Some(e.image)
    }

    /// Keep a retired image (the GPU may use it until the current tick
    /// completes). Images beyond the pool's limits are retired.
    pub fn put(&mut self, sched: &mut Scheduler, key: PoolKey, image: Image) {
        // SAFETY: plain query on a live image.
        let bytes = unsafe { image.dev.device.get_image_memory_requirements(image.image) }.size;
        let v = self.entries.entry(key).or_default();
        if bytes > MAX_POOLED_BYTES / 4 || v.len() >= MAX_PER_KEY {
            sched.retire(image);
            return;
        }
        self.order += 1;
        v.push(Entry { tick: sched.current_tick(), bytes, order: self.order, image });
        self.bytes += bytes;
        while self.bytes > MAX_POOLED_BYTES {
            let Some((k, i)) = self
                .entries
                .iter()
                .flat_map(|(k, v)| v.iter().enumerate().map(move |(i, e)| (e.order, *k, i)))
                .min_by_key(|e| e.0)
                .map(|(_, k, i)| (k, i))
            else {
                break;
            };
            let v = self.entries.get_mut(&k).unwrap();
            let e = v.swap_remove(i);
            if v.is_empty() {
                self.entries.remove(&k);
            }
            self.bytes -= e.bytes;
            sched.retire(e.image);
        }
    }

    pub fn len(&self) -> usize {
        self.entries.values().map(|v| v.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A device-local buffer shared by every transfer of the command stream,
/// grown on demand. Every transfer orders its buffer accesses after all
/// earlier commands, so one buffer serves them all; a replaced buffer is
/// retired with the tick.
pub struct Scratch {
    buffer: Option<Buffer>,
    usage: vk::BufferUsageFlags,
}

impl Scratch {
    pub fn new(usage: vk::BufferUsageFlags) -> Scratch {
        Scratch { buffer: None, usage }
    }

    /// The buffer, at least `size` bytes large.
    pub fn get(&mut self, dev: &Arc<GpuDevice>, sched: &mut Scheduler, size: u64) -> Option<vk::Buffer> {
        if self.buffer.as_ref().is_none_or(|b| b.size < size) {
            let bytes = size.max(1 << 20).next_power_of_two();
            let b = Buffer::new(dev, bytes, self.usage, MemoryUse::Device).ok()?;
            if let Some(old) = self.buffer.replace(b) {
                sched.retire(old);
            }
        }
        self.buffer.as_ref().map(|b| b.buffer)
    }
}
