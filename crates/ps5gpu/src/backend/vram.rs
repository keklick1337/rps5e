// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The back-end's side of the device memory report
//! ([`crate::vram_report`], `RPS5E_VRAM_REPORT=<seconds>`): when a report
//! is due, the thread owning the caches gathers their rows and logs them.

use super::{Backend, IMAGE_BUDGET};
use crate::vram_report as report;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// Report state of a back-end.
#[derive(Default)]
pub(super) struct State {
    clock: report::TickClock,
    last: Option<Instant>,
}

impl Backend {
    /// At a completed submission with the report on: sample the tick
    /// clock, and log the report once `period` passed since the last one.
    pub(super) fn vram_report_tick(&mut self, period: Duration) {
        let now = Instant::now();
        self.vram.clock.sample(now, self.sched.current_tick());
        match self.vram.last {
            Some(t) if now.duration_since(t) < period => return,
            // The first report comes one period after the first tick.
            None => {
                self.vram.last = Some(now);
                return;
            }
            _ => {}
        }
        self.vram.last = Some(now);
        log::info!("[vram] {}", self.vram_report(now));
    }

    fn vram_report(&self, now: Instant) -> String {
        let t0 = Instant::now();
        let mut out = String::new();
        let images = self.images.vram_rows();
        let buffers = self.buffers.vram_rows();
        let heaps = crate::gc::heaps(&self.dev);
        let budget = heaps.as_ref().map(|(h, discrete)| crate::gc::total_budget(h, *discrete));
        let cache_images: u64 = images.iter().map(|r| r.host_bytes).sum();
        let cache_buffers: u64 = buffers.iter().map(|r| r.size).sum();
        let counted_buffers = crate::device::DEVICE_LOCAL_BYTES[0].load(Ordering::Relaxed);
        let counted_images = crate::device::DEVICE_LOCAL_BYTES[1].load(Ordering::Relaxed);
        let (arena, image_arena) = (self.dev.arena.stats(), self.dev.image_arena.stats());
        let slack = (arena.block_bytes - arena.used_bytes.min(arena.block_bytes)) + (image_arena.block_bytes - image_arena.used_bytes.min(image_arena.block_bytes));
        out.push_str(&format!("device memory at tick {}\n", self.sched.current_tick()));
        report::image_lines(&mut out, &images, &self.vram.clock, now);
        report::buffer_lines(&mut out, &buffers, &self.vram.clock, now);
        let mut other = self.images.vram_other();
        other.extend(self.buffers.vram_other());
        let (staging_used, staging_size) = self.pipelines.staging_usage();
        other.push(("draw staging ring (host)", staging_size));
        other.push(("draw staging in flight (host)", staging_used));
        other.push(("display snapshots", self.snapshots.iter().map(|s| s.image.bytes).sum()));
        other.push(("retired in the open tick", self.sched.open_retired_bytes()));
        other.push(("buffer blocks", arena.block_bytes));
        other.push(("buffer blocks in use", arena.used_bytes));
        other.push(("image blocks", image_arena.block_bytes));
        other.push(("image blocks in use", image_arena.used_bytes));
        other.push(("image objects outside the cache", counted_images.saturating_sub(cache_images)));
        other.push(("buffer objects outside the mirrors", counted_buffers.saturating_sub(cache_buffers)));
        report::other_lines(&mut out, &other);
        match self.pipelines.object_counts() {
            Some((modules, graphics, compute)) => out.push_str(&format!("  objects: {modules} shader modules, {graphics} graphics pipelines, {compute} compute pipelines, {} snapshots\n", self.snapshots.len())),
            None => out.push_str("  objects: no pipeline provider counts\n"),
        }
        report::heap_lines(&mut out, heaps.as_ref(), counted_buffers, counted_images, slack);
        let (buffer_idle, buffer_urgent) = self.buffers.vram_marks();
        report::threshold_lines(&mut out, budget, &self.gc.images, &self.gc.buffers, IMAGE_BUDGET, buffer_idle, buffer_urgent);
        out.push_str(&format!("  (gathered in {:.1} ms; the marks were derived at start-up, the budget is the device's now)", t0.elapsed().as_secs_f64() * 1000.0));
        out
    }
}
