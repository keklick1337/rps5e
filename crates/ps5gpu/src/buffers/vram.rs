// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The buffer cache's rows for the device memory report
//! ([`crate::vram_report`]): gathered only when a report is due.

use super::{BufferCache, COLLECT_THRESHOLD};
use crate::vram_report::BufferRow;

impl BufferCache {
    /// One row per buffer mirror.
    pub fn vram_rows(&self) -> Vec<BufferRow> {
        self.buffers
            .values()
            .map(|b| BufferRow {
                addr: b.addr,
                size: b.size,
                last_use: b.last_use,
                committed: self.memory.committed(b.addr, b.size).runs.iter().map(|r| r.1 - r.0).sum(),
                gpu_dirty: self.tracker.has_gpu_dirty_pages(b.addr, b.size),
            })
            .collect()
    }

    /// Memory the cache holds next to its mirrors: (name, bytes). The
    /// rings and the import chunks are host memory.
    pub fn vram_other(&self) -> Vec<(&'static str, u64)> {
        let imports = |g: &Option<super::guest::GuestImports>| g.as_ref().map_or(0, |g| g.stats.chunk_bytes);
        vec![
            ("buffer upload ring (host)", self.staging.buffer.size),
            ("buffer download ring (host)", self.download.buffer.size),
            ("address page table", self.bda.as_ref().map_or(0, |b| b.size)),
            ("guest memory imports (host)", imports(&self.guest) + imports(&self.unranged)),
        ]
    }

    /// Mirror bytes above which idle clean mirrors are released, and above
    /// which any mirror the GPU is done with is.
    pub fn vram_marks(&self) -> (u64, u64) {
        (self.idle_threshold, COLLECT_THRESHOLD)
    }
}
