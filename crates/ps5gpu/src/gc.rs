// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Memory-pressure garbage collection of the image and buffer caches.
//!
//! Next to the idle-frame eviction of each cache, a collection runs when
//! the device memory the process uses (VK_EXT_memory_budget; without it,
//! the bytes the cache itself holds) reaches a trigger derived from the
//! memory budget. Images: above the trigger, the least recently used
//! images idle for 16 collection ticks go, 10 per run; above the pressure
//! mark 20 per run idle for 80 ticks, written back first when the GPU
//! modified them (tiled GPU-modified images stay); above the critical mark
//! a second run takes 40 idle for 160 ticks. Buffers: above the trigger 32
//! clean buffers idle for 160 ticks; above the critical mark 64 idle for 80
//! ticks, GPU-written ones downloaded first. A collection tick is one end
//! of a command-stream slice that submitted work; ages are measured
//! against the scheduler tick at each of the last collection ticks.

/// Scheduler ticks at the last collection ticks (oldest first).
#[derive(Default)]
pub struct CollectionTicks {
    ticks: std::collections::VecDeque<u64>,
}

impl CollectionTicks {
    /// Ticks kept: the oldest age asked for.
    const KEPT: usize = 161;

    /// A collection tick at scheduler tick `tick`.
    pub fn push(&mut self, tick: u64) {
        if self.ticks.len() == Self::KEPT {
            self.ticks.pop_front();
        }
        self.ticks.push_back(tick);
    }

    /// Entries last used before this scheduler tick have been idle for
    /// `age` collection ticks (0 while fewer have passed).
    pub fn cutoff(&self, age: u64) -> u64 {
        let n = self.ticks.len();
        if (age as usize) < n {
            self.ticks[n - 1 - age as usize]
        } else {
            0
        }
    }
}

use crate::device::GpuDevice;
use ash::vk;

const GIB: i64 = 1 << 30;

/// Usage marks of one cache (bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Thresholds {
    /// Below it nothing is collected.
    pub trigger: u64,
    /// Above it collections take more, older entries and write back
    /// GPU-modified images (images only).
    pub pressure: u64,
    /// Above it collections are aggressive.
    pub critical: u64,
}

impl Thresholds {
    /// The image cache's marks for a memory budget (None: the device does
    /// not report one; fixed marks over the cache's own bytes).
    pub fn images(budget: Option<u64>) -> Thresholds {
        // Without a reported budget the cache counts its own bytes, and
        // collecting starts at the pressure mark (the idle-frame eviction
        // keeps the cache below its budget otherwise).
        let Some(budget) = budget else {
            return Thresholds { trigger: (GIB + GIB / 2) as u64, pressure: (GIB + GIB / 2) as u64, critical: 3 * GIB as u64 };
        };
        let budget = budget.min(i64::MAX as u64) as i64;
        let threshold = budget.min(8 * GIB);
        Thresholds {
            pressure: (budget - 6 * threshold / 10).min(budget - GIB).max(GIB + GIB / 2) as u64,
            critical: (budget - 2 * threshold / 10).min(budget - GIB / 2).max(3 * GIB) as u64,
            trigger: ((budget - threshold) / 2).max(0) as u64,
        }
    }

    /// The buffer cache's marks (its pressure mark is the trigger).
    pub fn buffers(budget: Option<u64>) -> Thresholds {
        let Some(budget) = budget else {
            return Thresholds { trigger: GIB as u64, pressure: GIB as u64, critical: 2 * GIB as u64 };
        };
        let budget = budget.min(i64::MAX as u64) as i64;
        let threshold = budget.min(8 * GIB);
        let trigger = (budget - 6 * threshold / 10).min(budget - GIB).max(GIB) as u64;
        Thresholds { trigger, pressure: trigger, critical: (budget - 2 * threshold / 10).min(budget - GIB / 2).max(2 * GIB) as u64 }
    }
}

/// One image collection run at `usage`: (pressured, aggressive, minimum
/// idle ticks, images to look at).
pub fn image_run(usage: u64, t: &Thresholds, allow_aggressive: bool) -> (bool, bool, u64, usize) {
    let pressured = usage >= t.pressure;
    let aggressive = allow_aggressive && usage >= t.critical;
    let age = if aggressive {
        160
    } else if pressured {
        80
    } else {
        16
    };
    let deletions = if aggressive {
        40
    } else if pressured {
        20
    } else {
        10
    };
    (pressured, aggressive, age, deletions)
}

/// Presented frames an entry must also have stayed unused before a run
/// takes it: the collection ticks count submissions, and a game that
/// submits a few dozen times per frame would otherwise lose (and reload)
/// textures it binds every frame or two whenever the process sits above
/// the trigger (SILENT HILL with other processes on the GPU: a working set
/// of ~8 GiB over a 6.5 GiB trigger). Every frame's entries stay unless
/// memory is critical.
pub fn idle_frames(pressured: bool, aggressive: bool) -> usize {
    if aggressive {
        1
    } else if pressured {
        4
    } else {
        16
    }
}

/// One buffer collection run at `usage`: (aggressive, minimum idle ticks,
/// buffers to release).
pub fn buffer_run(usage: u64, t: &Thresholds) -> (bool, u64, usize) {
    let aggressive = usage >= t.critical;
    (aggressive, if aggressive { 80 } else { 160 }, if aggressive { 64 } else { 32 })
}

/// Heaps of the device: (size, budget, usage, device local).
pub type Heap = (u64, u64, u64, bool);

/// The memory the process may use: the budgets of the device-local heaps
/// of a discrete device less an eighth (at most 1 GiB), or on a device
/// sharing system memory the larger of its local heaps and what the budget
/// leaves free after 8 GiB for the system.
pub fn total_budget(heaps: &[Heap], discrete: bool) -> u64 {
    let (mut budget, mut local, mut usage) = (0u64, 0u64, 0u64);
    for &(size, b, u, device_local) in heaps {
        if device_local {
            local += size;
        }
        if !discrete || device_local {
            budget += b;
            usage += u;
        }
    }
    if discrete {
        return budget - (budget / 8).min(GIB as u64);
    }
    let reserve = 8 * GIB as u64;
    let available = budget.saturating_sub(usage);
    local.max(available.saturating_sub(reserve))
}

/// Memory the process uses on the heaps [`total_budget`] counts.
pub fn device_usage(heaps: &[Heap], discrete: bool) -> u64 {
    heaps.iter().filter(|h| !discrete || h.3).map(|h| h.2).sum()
}

/// The heaps of `dev` with their budgets (None without
/// VK_EXT_memory_budget), and whether the device is discrete.
pub fn heaps(dev: &GpuDevice) -> Option<(Vec<Heap>, bool)> {
    if !dev.caps.memory_budget {
        return None;
    }
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let props = {
        let mut p2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        // SAFETY: valid chain on the stack; physical device queries need no
        // external synchronisation.
        unsafe { dev.instance.get_physical_device_memory_properties2(dev.phys, &mut p2) };
        p2.memory_properties
    };
    let heaps = (0..props.memory_heap_count as usize)
        .map(|i| {
            let h = props.memory_heaps[i];
            (h.size, budget.heap_budget[i], budget.heap_usage[i], h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        })
        .collect();
    Some((heaps, dev.props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU))
}

/// Device memory reports of one device.
pub struct DeviceMemory {
    pub images: Thresholds,
    pub buffers: Thresholds,
    /// The device reports its usage (else the caches count their bytes).
    pub reported: bool,
}

impl DeviceMemory {
    pub fn new(dev: &GpuDevice) -> DeviceMemory {
        let budget = heaps(dev).map(|(h, discrete)| total_budget(&h, discrete));
        DeviceMemory { images: Thresholds::images(budget), buffers: Thresholds::buffers(budget), reported: budget.is_some() }
    }

    /// The memory the process uses now (None when the device does not
    /// report it).
    pub fn usage(&self, dev: &GpuDevice) -> Option<u64> {
        if !self.reported {
            return None;
        }
        heaps(dev).map(|(h, discrete)| device_usage(&h, discrete))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: u64 = 1 << 30;

    #[test]
    fn image_marks_follow_the_budget() {
        // 16 GiB: threshold 8 GiB.
        let t = Thresholds::images(Some(16 * G));
        assert_eq!(t.pressure, 16 * G - 8 * G * 6 / 10);
        assert_eq!(t.critical, 16 * G - 8 * G * 2 / 10);
        assert_eq!(t.trigger, 4 * G);
        // 4 GiB: threshold = budget, no trigger.
        let t = Thresholds::images(Some(4 * G));
        assert_eq!(t.pressure, 4 * G - 4 * G * 6 / 10);
        assert_eq!(t.critical, 4 * G - 4 * G * 2 / 10);
        assert_eq!(t.trigger, 0);
        // 2 GiB: the floors hold.
        let t = Thresholds::images(Some(2 * G));
        assert_eq!((t.pressure, t.critical), (G + G / 2, 3 * G));
        // No budget: fixed marks over the cache's own bytes.
        assert_eq!(Thresholds::images(None), Thresholds { trigger: G + G / 2, pressure: G + G / 2, critical: 3 * G });
    }

    #[test]
    fn buffer_marks_follow_the_budget() {
        let t = Thresholds::buffers(Some(16 * G));
        assert_eq!(t.trigger, 16 * G - 8 * G * 6 / 10);
        assert_eq!(t.critical, 16 * G - 8 * G * 2 / 10);
        let t = Thresholds::buffers(Some(2 * G));
        assert_eq!((t.trigger, t.critical), (G, 2 * G));
        assert_eq!(Thresholds::buffers(None), Thresholds { trigger: G, pressure: G, critical: 2 * G });
    }

    #[test]
    fn collection_tick_cutoffs() {
        let mut t = CollectionTicks::default();
        assert_eq!(t.cutoff(16), 0);
        for i in 0..200u64 {
            t.push(1000 + i * 10);
        }
        // The newest tick is 1000 + 199 * 10.
        assert_eq!(t.cutoff(0), 2990);
        assert_eq!(t.cutoff(16), 2990 - 160);
        assert_eq!(t.cutoff(160), 2990 - 1600);
        assert_eq!(t.cutoff(161), 0);
    }

    #[test]
    fn runs_grow_with_usage() {
        let t = Thresholds { trigger: G, pressure: 2 * G, critical: 3 * G };
        assert_eq!(image_run(G, &t, true), (false, false, 16, 10));
        assert_eq!(image_run(2 * G, &t, true), (true, false, 80, 20));
        assert_eq!(image_run(3 * G, &t, false), (true, false, 80, 20));
        assert_eq!(image_run(3 * G, &t, true), (true, true, 160, 40));
        assert_eq!(buffer_run(G, &t), (false, 160, 32));
        assert_eq!(buffer_run(3 * G, &t), (true, 80, 64));
    }

    #[test]
    fn budgets_of_discrete_and_shared_devices() {
        // Discrete: device-local budgets less an eighth, at most 1 GiB.
        let heaps = [(24 * G, 22 * G, 3 * G, true), (64 * G, 30 * G, G, false)];
        assert_eq!(total_budget(&heaps, true), 22 * G - G);
        assert_eq!(total_budget(&[(4 * G, 4 * G, 0, true)], true), 4 * G - G / 2);
        assert_eq!(device_usage(&heaps, true), 3 * G);
        // Shared memory: the larger of the local heaps and what is left
        // after 8 GiB for the system.
        let shared = [(2 * G, 2 * G, 0, true), (32 * G, 30 * G, 4 * G, false)];
        assert_eq!(total_budget(&shared, false), (32 * G - 4 * G) - 8 * G);
        assert_eq!(total_budget(&[(2 * G, 2 * G, 0, true), (8 * G, 6 * G, G, false)], false), 2 * G);
        assert_eq!(device_usage(&shared, false), 4 * G);
    }
}
