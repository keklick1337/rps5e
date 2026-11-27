// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Periodic report of where device memory goes (`RPS5E_VRAM_REPORT=<s>`).
//!
//! Every period the back-end gathers one row per cached image and buffer
//! mirror ([`crate::images::ImageCache::vram_rows`],
//! [`crate::buffers::BufferCache::vram_rows`]) together with the totals
//! of everything else it can name, and this module turns them into a
//! multi-line log entry: bytes per format, size class, last use, guest
//! address region and idle time, the bytes of images over uncommitted
//! guest pages, the bytes several images hold over the same guest
//! memory, the largest entries, and what the device reports beyond the
//! two caches. Nothing is gathered while the switch is off.

use ash::vk;
use std::collections::VecDeque;
use std::fmt::Write;
use std::time::{Duration, Instant};

/// The report period (None: off).
pub fn period() -> Option<Duration> {
    static PERIOD: std::sync::OnceLock<Option<Duration>> = std::sync::OnceLock::new();
    *PERIOD.get_or_init(|| parse_period(std::env::var("RPS5E_VRAM_REPORT").ok().as_deref()))
}

/// Seconds (fractions allowed) above zero turn the report on.
pub fn parse_period(v: Option<&str>) -> Option<Duration> {
    let s: f64 = v?.trim().parse().ok()?;
    (s.is_finite() && s > 0.0).then(|| Duration::from_secs_f64(s.max(0.25)))
}

/// Guest addresses from here on belong to the sparse (partially
/// resident) windows of the memory manager; below lie the direct and
/// flexible memory mappings.
pub const SPARSE_WINDOW_BASE: u64 = 0x0800_0000_0000;
const TIB: u64 = 1 << 40;

/// Region names by [`region`].
pub const REGIONS: [&str; 3] = ["<1TiB", "1TiB..window", "window"];

pub fn region(addr: u64) -> usize {
    if addr >= SPARSE_WINDOW_BASE {
        2
    } else if addr >= TIB {
        1
    } else {
        0
    }
}

/// Size class names by [`size_class`].
pub const SIZE_CLASSES: [&str; 6] = ["<64K", "<1M", "<4M", "<16M", "<64M", ">=64M"];

pub fn size_class(bytes: u64) -> usize {
    const LIMITS: [u64; 5] = [64 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20];
    LIMITS.iter().position(|&l| bytes < l).unwrap_or(LIMITS.len())
}

/// Idle class names by [`age_class`].
pub const AGE_CLASSES: [&str; 4] = ["<1s", "<10s", "<60s", "older"];

/// The class of an idle time (None: older than the clock knows).
pub fn age_class(age: Option<f64>) -> usize {
    match age {
        Some(a) if a < 1.0 => 0,
        Some(a) if a < 10.0 => 1,
        Some(a) if a < 60.0 => 2,
        _ => 3,
    }
}

/// What an image was last used as, by the layout it was left in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LastUse {
    Sampled,
    ColorTarget,
    Depth,
    Storage,
    Transfer,
    /// Never used, or subresources in different layouts.
    Other,
}

pub const LAST_USES: [&str; 6] = ["sampled", "color-target", "depth", "storage", "transfer", "other"];

impl LastUse {
    pub fn of(layout: Option<vk::ImageLayout>, depth_format: bool) -> LastUse {
        match layout {
            Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => LastUse::Sampled,
            Some(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => LastUse::ColorTarget,
            Some(
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                | vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                | vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL
                | vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
            ) => LastUse::Depth,
            Some(vk::ImageLayout::GENERAL) => LastUse::Storage,
            Some(vk::ImageLayout::TRANSFER_SRC_OPTIMAL | vk::ImageLayout::TRANSFER_DST_OPTIMAL) => {
                if depth_format {
                    LastUse::Depth
                } else {
                    LastUse::Transfer
                }
            }
            _ => LastUse::Other,
        }
    }
}

/// Wall-clock times of scheduler ticks, sampled about once a second
/// while the report is on: the idle time of a cache entry is the time
/// since the first sample at or after its last use.
pub struct TickClock {
    samples: VecDeque<(Instant, u64)>,
}

impl Default for TickClock {
    fn default() -> Self {
        TickClock { samples: VecDeque::new() }
    }
}

impl TickClock {
    /// Samples kept (ten minutes).
    const KEPT: usize = 600;

    /// Note `tick` at `now` when a second passed since the last sample.
    pub fn sample(&mut self, now: Instant, tick: u64) {
        if self.samples.back().is_some_and(|&(t, _)| now.duration_since(t) < Duration::from_secs(1)) {
            return;
        }
        if self.samples.len() == Self::KEPT {
            self.samples.pop_front();
        }
        self.samples.push_back((now, tick));
    }

    /// Seconds an entry last used at `tick` has been idle at `now`, at
    /// most a sample interval too short (None: used before the oldest
    /// sample).
    pub fn idle(&self, now: Instant, tick: u64) -> Option<f64> {
        let i = self.samples.partition_point(|&(_, t)| t < tick);
        match self.samples.get(i) {
            Some(&(at, _)) if i > 0 || self.samples[0].1 == tick => Some(now.duration_since(at).as_secs_f64()),
            Some(_) => None,
            // Used after the newest sample.
            None => Some(0.0),
        }
    }

    /// Seconds the clock covers at `now`.
    pub fn span(&self, now: Instant) -> f64 {
        self.samples.front().map_or(0.0, |&(t, _)| now.duration_since(t).as_secs_f64())
    }
}

/// One cached image.
#[derive(Clone, Debug)]
pub struct ImageRow {
    pub addr: u64,
    /// Guest bytes of the main plane and of all planes.
    pub guest_bytes: u64,
    /// Every plane's guest range.
    pub planes: Vec<(u64, u64)>,
    /// Bytes of the host allocation.
    pub host_bytes: u64,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub layers: u32,
    pub levels: u32,
    pub samples: u32,
    pub tiled: bool,
    pub last_use: u64,
    pub last: LastUse,
    pub gpu_modified: bool,
    pub initialized: bool,
    pub pinned: bool,
    pub superseded: bool,
    /// Live images overlapping its guest planes.
    pub aliases: usize,
    /// Guest bytes of its planes without committed memory behind them.
    pub uncommitted: u64,
}

/// One buffer mirror.
#[derive(Clone, Debug)]
pub struct BufferRow {
    pub addr: u64,
    pub size: u64,
    pub last_use: u64,
    /// Bytes of its guest range with committed memory behind them.
    pub committed: u64,
    pub gpu_dirty: bool,
}

/// Bytes covered by the union of `ranges` (`[begin, end)` each).
pub fn union_bytes(ranges: &mut [(u64, u64)]) -> u64 {
    ranges.sort_unstable();
    let (mut total, mut reach) = (0u64, 0u64);
    for &(b, e) in ranges.iter() {
        let b = b.max(reach);
        if e > b {
            total += e - b;
            reach = e;
        }
    }
    total
}

/// Sums per key, largest first.
pub fn top_by<K: Eq + std::hash::Hash + Copy>(items: impl Iterator<Item = (K, u64)>, n: usize) -> Vec<(K, u64, usize)> {
    let mut m: std::collections::HashMap<K, (u64, usize)> = std::collections::HashMap::new();
    for (k, b) in items {
        let e = m.entry(k).or_default();
        e.0 += b;
        e.1 += 1;
    }
    let mut v: Vec<(K, u64, usize)> = m.into_iter().map(|(k, (b, c))| (k, b, c)).collect();
    v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));
    v.truncate(n);
    v
}

fn mib(b: u64) -> String {
    if b >= 10 << 20 || b == 0 {
        format!("{}", b >> 20)
    } else {
        format!("{:.1}", b as f64 / (1u64 << 20) as f64)
    }
}

fn age_text(a: Option<f64>) -> String {
    match a {
        Some(a) => format!("{a:.0}s"),
        None => "old".into(),
    }
}

/// `name bytes/count` per class.
fn classes(names: &[&str], sums: &[(u64, usize)]) -> String {
    names.iter().zip(sums).map(|(n, (b, c))| format!("{n} {}/{c}", mib(*b))).collect::<Vec<_>>().join(", ")
}

/// The image part of the report (host MiB unless said otherwise).
pub fn image_lines(out: &mut String, rows: &[ImageRow], clock: &TickClock, now: Instant) {
    let host: u64 = rows.iter().map(|r| r.host_bytes).sum();
    let guest: u64 = rows.iter().map(|r| r.guest_bytes).sum();
    let _ = writeln!(out, "  images: {} live, host {} MiB, guest {} MiB (MiB/count below are host bytes)", rows.len(), mib(host), mib(guest));
    let formats = top_by(rows.iter().map(|r| (r.format, r.host_bytes)), 10);
    let _ = writeln!(out, "    by format: {}", formats.iter().map(|(f, b, c)| format!("{f:?} {}/{c}", mib(*b))).collect::<Vec<_>>().join(", "));
    let mut size = [(0u64, 0usize); SIZE_CLASSES.len()];
    let mut last = [(0u64, 0usize); LAST_USES.len()];
    let mut reg = [(0u64, 0usize); REGIONS.len()];
    let mut age = [(0u64, 0usize); AGE_CLASSES.len()];
    let add = |slot: &mut (u64, usize), b: u64| {
        slot.0 += b;
        slot.1 += 1;
    };
    // (host bytes, count) of images with the property.
    let (mut modified, mut pinned, mut superseded, mut aliased, mut uninit, mut ms, mut tiled) = ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0));
    let (mut partial, mut partial_guest, mut partial_missing, mut all_missing) = ((0u64, 0usize), 0u64, 0u64, (0u64, 0usize));
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    let mut plane_bytes = 0u64;
    for r in rows {
        add(&mut size[size_class(r.host_bytes)], r.host_bytes);
        add(&mut last[r.last as usize], r.host_bytes);
        add(&mut reg[region(r.addr)], r.host_bytes);
        add(&mut age[age_class(clock.idle(now, r.last_use))], r.host_bytes);
        for (flag, slot) in [
            (r.gpu_modified, &mut modified),
            (r.pinned, &mut pinned),
            (r.superseded, &mut superseded),
            (r.aliases != 0, &mut aliased),
            (!r.initialized, &mut uninit),
            (r.samples > 1, &mut ms),
            (r.tiled, &mut tiled),
        ] {
            if flag {
                add(slot, r.host_bytes);
            }
        }
        if r.uncommitted != 0 {
            add(&mut partial, r.host_bytes);
            partial_guest += r.guest_bytes;
            partial_missing += r.uncommitted;
            if r.uncommitted >= r.guest_bytes {
                add(&mut all_missing, r.host_bytes);
            }
        }
        for &(b, e) in &r.planes {
            plane_bytes += e - b;
            ranges.push((b, e));
        }
    }
    let _ = writeln!(out, "    by host size: {}", classes(&SIZE_CLASSES, &size));
    let _ = writeln!(out, "    by last use: {}", classes(&LAST_USES, &last));
    let _ = writeln!(out, "    by guest region: {}", classes(&REGIONS, &reg));
    let _ = writeln!(out, "    by idle time: {} (clock covers {:.0} s)", classes(&AGE_CLASSES, &age), clock.span(now));
    let p = |v: (u64, usize)| format!("{}/{}", mib(v.0), v.1);
    let _ = writeln!(
        out,
        "    states: gpu-modified {}, pinned by tables {}, superseded {}, never filled {}, multisampled {}, tiled {}",
        p(modified),
        p(pinned),
        p(superseded),
        p(uninit),
        p(ms),
        p(tiled)
    );
    let _ = writeln!(
        out,
        "    uncommitted guest pages: {} in images with any ({} MiB guest, of which {} MiB uncommitted = {} 4K pages); wholly uncommitted {}",
        p(partial),
        mib(partial_guest),
        mib(partial_missing),
        partial_missing >> 12,
        p(all_missing)
    );
    let union = union_bytes(&mut ranges);
    let _ = writeln!(
        out,
        "    overlap: {} in images overlapping another live image; guest plane bytes {} MiB cover {} MiB of guest memory ({} MiB held more than once)",
        p(aliased),
        mib(plane_bytes),
        mib(union),
        mib(plane_bytes - union)
    );
    let mut largest: Vec<&ImageRow> = rows.iter().collect();
    largest.sort_unstable_by(|a, b| b.host_bytes.cmp(&a.host_bytes));
    for r in largest.iter().take(15) {
        let _ = writeln!(
            out,
            "    {:#x} host {} guest {} MiB {:?} {}x{}x{} L{} M{} S{} {:?} idle {}{}{}{}{}{}{}",
            r.addr,
            mib(r.host_bytes),
            mib(r.guest_bytes),
            r.format,
            r.width,
            r.height,
            r.depth,
            r.layers,
            r.levels,
            r.samples,
            r.last,
            age_text(clock.idle(now, r.last_use)),
            if r.gpu_modified { " mod" } else { "" },
            if r.pinned { " pinned" } else { "" },
            if r.superseded { " superseded" } else { "" },
            if r.tiled { "" } else { " linear" },
            if r.aliases != 0 { format!(" aliases {}", r.aliases) } else { String::new() },
            if r.uncommitted != 0 { format!(" uncommitted {} MiB", mib(r.uncommitted)) } else { String::new() },
        );
    }
}

/// The buffer part of the report.
pub fn buffer_lines(out: &mut String, rows: &[BufferRow], clock: &TickClock, now: Instant) {
    let total: u64 = rows.iter().map(|r| r.size).sum();
    let committed: u64 = rows.iter().map(|r| r.committed).sum();
    let _ = writeln!(out, "  buffer mirrors: {} live, {} MiB, {} MiB of their guest ranges committed ({} MiB mirror uncommitted pages)", rows.len(), mib(total), mib(committed), mib(total - committed.min(total)));
    let mut size = [(0u64, 0usize); SIZE_CLASSES.len()];
    let mut reg = [(0u64, 0usize); REGIONS.len()];
    let mut age = [(0u64, 0usize); AGE_CLASSES.len()];
    let mut dirty = (0u64, 0usize);
    for r in rows {
        for slot in [&mut size[size_class(r.size)], &mut reg[region(r.addr)], &mut age[age_class(clock.idle(now, r.last_use))]] {
            slot.0 += r.size;
            slot.1 += 1;
        }
        if r.gpu_dirty {
            dirty.0 += r.size;
            dirty.1 += 1;
        }
    }
    let _ = writeln!(out, "    by size: {}", classes(&SIZE_CLASSES, &size));
    let _ = writeln!(out, "    by guest region: {}", classes(&REGIONS, &reg));
    let _ = writeln!(out, "    by idle time: {}", classes(&AGE_CLASSES, &age));
    let _ = writeln!(out, "    holding GPU-written pages: {}/{}", mib(dirty.0), dirty.1);
    let mut largest: Vec<&BufferRow> = rows.iter().collect();
    largest.sort_unstable_by(|a, b| b.size.cmp(&a.size));
    for r in largest.iter().take(10) {
        let _ = writeln!(
            out,
            "    {:#x}+{:#x} {} MiB idle {} committed {} MiB ({}%){}",
            r.addr,
            r.size,
            mib(r.size),
            age_text(clock.idle(now, r.last_use)),
            mib(r.committed),
            r.committed * 100 / r.size.max(1),
            if r.gpu_dirty { " gpu-dirty" } else { "" }
        );
    }
}

/// Named amounts of device memory outside the two caches, in bytes.
pub fn other_lines(out: &mut String, items: &[(&str, u64)]) {
    let _ = writeln!(out, "  other (MiB): {}", items.iter().map(|(n, b)| format!("{n} {}", mib(*b))).collect::<Vec<_>>().join(", "));
}

/// The heaps as the device reports them and what the report cannot name:
/// `counted` are the device-local bytes of every live buffer and image
/// object of the process.
pub fn heap_lines(out: &mut String, heaps: Option<&(Vec<crate::gc::Heap>, bool)>, counted_buffers: u64, counted_images: u64, arena_slack: u64) {
    let Some((heaps, discrete)) = heaps else {
        let _ = writeln!(out, "  heaps: the device reports no memory budget");
        return;
    };
    for (i, &(size, budget, usage, local)) in heaps.iter().enumerate() {
        let _ = writeln!(out, "  heap {i}{}: size {} budget {} usage {} MiB", if local { " (device local)" } else { "" }, mib(size), mib(budget), mib(usage));
    }
    let usage = crate::gc::device_usage(heaps, *discrete);
    let named = counted_buffers + counted_images + arena_slack;
    let _ = writeln!(
        out,
        "  device usage {} MiB = buffer objects {} + image objects {} + free room in shared blocks {} + unaccounted {} MiB (pipelines, shader modules, descriptor pools, swapchain, driver)",
        mib(usage),
        mib(counted_buffers),
        mib(counted_images),
        mib(arena_slack),
        (usage as i64 - named as i64) >> 20
    );
}

/// The collection marks in force.
pub fn threshold_lines(out: &mut String, budget: Option<u64>, images: &crate::gc::Thresholds, buffers: &crate::gc::Thresholds, image_budget: u64, buffer_idle: u64, buffer_urgent: u64) {
    let _ = writeln!(
        out,
        "  collection: device budget {} MiB; images trigger/pressure/critical {}/{}/{} MiB of device usage, idle eviction above {} MiB of guest image bytes; buffers trigger/critical {}/{} MiB of device usage, idle release above {} MiB and urgent above {} MiB of mirror bytes",
        budget.map_or("none".into(), mib),
        mib(images.trigger),
        mib(images.pressure),
        mib(images.critical),
        mib(image_budget),
        mib(buffers.trigger),
        mib(buffers.critical),
        mib(buffer_idle),
        mib(buffer_urgent)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_period_is_seconds_above_zero() {
        assert_eq!(parse_period(None), None);
        assert_eq!(parse_period(Some("0")), None);
        assert_eq!(parse_period(Some("x")), None);
        assert_eq!(parse_period(Some("-3")), None);
        assert_eq!(parse_period(Some("5")), Some(Duration::from_secs(5)));
        assert_eq!(parse_period(Some(" 2.5 ")), Some(Duration::from_millis(2500)));
        assert_eq!(parse_period(Some("0.01")), Some(Duration::from_millis(250)));
    }

    #[test]
    fn classes_split_at_their_limits() {
        assert_eq!(size_class(0), 0);
        assert_eq!(size_class((64 << 10) - 1), 0);
        assert_eq!(size_class(64 << 10), 1);
        assert_eq!(size_class(4 << 20), 3);
        assert_eq!(size_class(64 << 20), 5);
        assert_eq!(size_class(u64::MAX), 5);
        assert_eq!(region(0x2_0000_0000), 0);
        assert_eq!(region(1 << 40), 1);
        assert_eq!(region(SPARSE_WINDOW_BASE), 2);
        assert_eq!(age_class(Some(0.5)), 0);
        assert_eq!(age_class(Some(1.0)), 1);
        assert_eq!(age_class(Some(59.0)), 2);
        assert_eq!(age_class(Some(60.0)), 3);
        assert_eq!(age_class(None), 3);
    }

    #[test]
    fn union_counts_shared_bytes_once() {
        assert_eq!(union_bytes(&mut []), 0);
        assert_eq!(union_bytes(&mut [(0, 10), (20, 30)]), 20);
        assert_eq!(union_bytes(&mut [(20, 30), (0, 10), (5, 25)]), 30);
        // The same range three times, and one inside another.
        assert_eq!(union_bytes(&mut [(100, 200), (100, 200), (100, 200), (120, 130)]), 100);
    }

    #[test]
    fn top_sums_per_key() {
        let v = top_by([(1, 10u64), (2, 5), (1, 1), (3, 7)].into_iter(), 2);
        assert_eq!(v, vec![(1, 11, 2), (3, 7, 1)]);
    }

    #[test]
    fn idle_time_comes_from_the_tick_samples() {
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        let mut c = TickClock::default();
        assert_eq!(c.idle(at(0), 5), Some(0.0));
        c.sample(at(0), 100);
        // Within a second of the last sample: not taken.
        c.sample(at(0) + Duration::from_millis(500), 150);
        c.sample(at(10), 200);
        c.sample(at(20), 300);
        // Used before the oldest sample: unknown.
        assert_eq!(c.idle(at(30), 50), None);
        assert_eq!(c.idle(at(30), 100), Some(30.0));
        // Between samples: the next sample's time.
        assert_eq!(c.idle(at(30), 150), Some(20.0));
        assert_eq!(c.idle(at(30), 300), Some(10.0));
        assert_eq!(c.idle(at(30), 301), Some(0.0));
        assert_eq!(c.span(at(30)), 30.0);
    }

    #[test]
    fn the_report_names_duplicates_and_uncommitted_bytes() {
        let row = |addr: u64, bytes: u64, uncommitted: u64, aliases: usize| ImageRow {
            addr,
            guest_bytes: bytes,
            planes: vec![(addr, addr + bytes)],
            host_bytes: bytes,
            format: vk::Format::BC7_UNORM_BLOCK,
            width: 1024,
            height: 1024,
            depth: 1,
            layers: 1,
            levels: 1,
            samples: 1,
            tiled: true,
            last_use: 1,
            last: LastUse::Sampled,
            gpu_modified: false,
            initialized: true,
            pinned: aliases != 0,
            superseded: false,
            aliases,
            uncommitted,
        };
        let rows = [row(SPARSE_WINDOW_BASE, 32 << 20, 16 << 20, 1), row(SPARSE_WINDOW_BASE, 32 << 20, 0, 1), row(0x2_0000_0000, 16 << 20, 0, 0)];
        let mut s = String::new();
        image_lines(&mut s, &rows, &TickClock::default(), Instant::now());
        assert!(s.contains("images: 3 live, host 80 MiB, guest 80 MiB"), "{s}");
        assert!(s.contains("by guest region: <1TiB 16/1, 1TiB..window 0/0, window 64/2"), "{s}");
        assert!(s.contains("32/1 in images with any (32 MiB guest, of which 16 MiB uncommitted = 4096 4K pages)"), "{s}");
        assert!(s.contains("64/2 in images overlapping another live image; guest plane bytes 80 MiB cover 48 MiB of guest memory (32 MiB held more than once)"), "{s}");
        assert!(s.contains("pinned by tables 64/2"), "{s}");
        let mut b = String::new();
        buffer_lines(&mut b, &[BufferRow { addr: SPARSE_WINDOW_BASE, size: 64 << 20, last_use: 0, committed: 16 << 20, gpu_dirty: true }], &TickClock::default(), Instant::now());
        assert!(b.contains("1 live, 64 MiB, 16 MiB of their guest ranges committed (48 MiB mirror uncommitted pages)"), "{b}");
        assert!(b.contains("committed 16 MiB (25%) gpu-dirty"), "{b}");
    }
}
