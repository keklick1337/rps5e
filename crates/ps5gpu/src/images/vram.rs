// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The image cache's rows for the device memory report
//! ([`crate::vram_report`]): gathered only when a report is due.

use super::{ImageCache, SubRange};
use crate::vram_report::{ImageRow, LastUse};

impl ImageCache {
    /// One row per cached image.
    pub fn vram_rows(&self) -> Vec<ImageRow> {
        self.images
            .values()
            .map(|i| {
                let k = &i.key;
                let planes: Vec<(u64, u64)> = k.planes().map(|p| (p.addr, p.addr + p.size())).collect();
                let uncommitted = planes
                    .iter()
                    .map(|&(b, e)| {
                        let committed: u64 = self.memory.committed(b, e - b).runs.iter().map(|r| r.1 - r.0).sum();
                        (e - b).saturating_sub(committed)
                    })
                    .sum();
                let all = SubRange { base_level: 0, levels: i.image.mip_levels, base_layer: 0, layers: i.image.layers };
                ImageRow {
                    addr: k.main.addr,
                    guest_bytes: planes.iter().map(|p| p.1 - p.0).sum(),
                    planes,
                    host_bytes: i.image.bytes,
                    format: k.format,
                    width: k.width,
                    height: k.height,
                    depth: k.depth,
                    layers: k.layers,
                    levels: k.levels,
                    samples: k.samples,
                    tiled: k.main.swizzle != crate::tiling::SWIZZLE_LINEAR,
                    last_use: i.last_use,
                    last: LastUse::of(self.uniform_layout(i.id, all), k.is_depth()),
                    gpu_modified: i.gpu_modified,
                    initialized: i.initialized,
                    pinned: i.pins != 0,
                    superseded: self.superseded.contains_key(&i.id),
                    aliases: i.aliases.len(),
                    uncommitted,
                }
            })
            .collect()
    }

    /// Device memory the cache holds next to its images: (name, bytes).
    pub fn vram_other(&self) -> Vec<(&'static str, u64)> {
        vec![
            ("image pool", self.pool.bytes()),
            ("image staging ring", self.staging.buffer.size),
            ("tiler scratch", self.linear.bytes() + self.tiled.bytes() + self.wide.bytes()),
        ]
    }
}
