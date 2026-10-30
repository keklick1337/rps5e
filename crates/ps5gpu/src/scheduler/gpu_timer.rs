// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! GPU execution time (`RPS5E_GPU_PROFILE=1`): the recording command
//! buffer is cut into labelled segments and the GPU time of the buffer is
//! split exactly between them.
//!
//! A buffer starts with a top-of-pipe timestamp; every change of the
//! active label ([`GpuTimer::enter`]) after commands were recorded under
//! the old one writes a timestamp that waits for every earlier command
//! (`ALL_COMMANDS`), closing the segment; the end of the buffer closes the
//! last one. Segment `i` is charged `t[i] - t[i-1]`: the time the GPU
//! needed to finish the segment's work after it finished everything before
//! it. The segments of a buffer add up to its busy time, so no work goes
//! unnamed: draws and dispatches are labelled with their shader address,
//! transfers, image preparation, clears and so on with the small labels of
//! [`label`] (see [`label_name`]), and commands recorded outside any of
//! them with [`label::OTHER`]. The upload prologue of a tick (a separate
//! buffer submitted right before it) is timed as [`label::PROLOGUE`].
//!
//! The queries come from one ring, reset on the host after they were read
//! back; a tick's queries are allocated while it is recorded and freed in
//! submission order.

use crate::device::GpuDevice;
use ash::vk;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Queries in the ring.
const QUERIES: u32 = 1 << 20;

/// Labels of the non-shader segments (shader addresses are far above).
pub mod label {
    /// Commands recorded outside every labelled operation (barriers,
    /// rendering-scope ends, state).
    pub const OTHER: u64 = 0;
    pub const TARGETS: u64 = 1;
    pub const HOUSEKEEPING: u64 = 2;
    pub const METADATA: u64 = 3;
    pub const RESOLVE: u64 = 4;
    pub const PRESSURE_GC: u64 = 5;
    pub const CAPTURE: u64 = 6;
    pub const CP_COPY: u64 = 7;
    /// The upload prologue buffer of a tick.
    pub const PROLOGUE: u64 = 8;
    /// Image contents tiled in from guest memory.
    pub const IMG_UPLOAD: u64 = 9;
    /// Image contents tiled out to their buffer mirror.
    pub const IMG_WRITEBACK: u64 = 10;
    /// Linear image copied out for a CPU read.
    pub const IMG_READBACK: u64 = 11;
    /// Image contents converted between formats of one memory.
    pub const IMG_CONVERT: u64 = 12;
    /// Image refreshed from newer aliases (retile through memory).
    pub const IMG_RETILE: u64 = 13;
    /// Image clears.
    pub const IMG_CLEAR: u64 = 14;
    /// Image copies and resolves between cached images.
    pub const IMG_COPY: u64 = 15;
    /// Buffer mirror uploads from guest memory.
    pub const BUF_UPLOAD: u64 = 16;
    /// Buffer mirror downloads to guest memory.
    pub const BUF_DOWNLOAD: u64 = 17;
    /// Fills of guest memory.
    pub const FILL: u64 = 18;
    /// Commands after the last timestamp a full ring could write.
    pub const OVERFLOW: u64 = 19;
    /// Image preparation not covered by a finer label.
    pub const IMG_PREPARE: u64 = 20;
    /// Buffer mirror ranges written through to guest memory.
    pub const BUF_THROUGH: u64 = 21;
    /// Buffer copies, fills and writes of the command processor.
    pub const BUF_COPY: u64 = 22;
    /// Memory barriers between commands of one kind and the next.
    pub const BARRIER: u64 = 23;
    /// Image layout transitions.
    pub const IMAGE_BARRIER: u64 = 24;
    /// Full barriers (every earlier command before every later one).
    pub const FULL_BARRIER: u64 = 25;
    /// Labels below this are categories, not shader addresses.
    pub const CATEGORIES: u64 = 0x100;
    /// `RETILE_AT | address`: a refresh from newer aliases of the image at
    /// that guest address (profile detail, see [`super::retile_label`]).
    pub const RETILE_AT: u64 = 1 << 63;
}

/// The label charging a refresh of the image at `addr` to that image.
pub fn retile_label(addr: u64) -> u64 {
    label::RETILE_AT | addr
}

/// Name of a category label (`None` for a shader address).
pub fn label_name(l: u64) -> Option<&'static str> {
    use label::*;
    Some(match l {
        OTHER => "other",
        TARGETS => "targets",
        HOUSEKEEPING => "housekeeping",
        METADATA => "metadata",
        RESOLVE => "resolve",
        PRESSURE_GC => "pressure_gc",
        CAPTURE => "capture",
        CP_COPY => "cp_copy",
        PROLOGUE => "prologue",
        IMG_UPLOAD => "img_upload",
        IMG_WRITEBACK => "img_writeback",
        IMG_READBACK => "img_readback",
        IMG_CONVERT => "img_convert",
        IMG_RETILE => "img_retile",
        IMG_CLEAR => "img_clear",
        IMG_COPY => "img_copy",
        BUF_UPLOAD => "buf_upload",
        BUF_DOWNLOAD => "buf_download",
        FILL => "fill",
        OVERFLOW => "overflow",
        IMG_PREPARE => "img_prepare",
        BUF_THROUGH => "buf_through",
        BUF_COPY => "buf_copy",
        BARRIER => "barrier",
        IMAGE_BARRIER => "image_barrier",
        FULL_BARRIER => "full_barrier",
        _ => return None,
    })
}

/// `label` as printed: its name or its shader address.
pub fn label_text(l: u64) -> String {
    label_name(l).map_or_else(|| format!("{l:#x}"), str::to_string)
}

/// Totals (ns) of the completed command buffers.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuTime {
    pub busy_ns: u64,
    pub max_ns: u64,
    pub buffers: u64,
    /// Command buffers that ran 16 ms or longer.
    pub long: u64,
}

/// Ring positions handed out to the open tick.
#[derive(Default)]
struct Open {
    /// Query of the recording buffer's start.
    start: Option<u32>,
    /// Closing query and label of each segment of the recording buffer.
    marks: Vec<(u32, u64)>,
    /// Start and end query of the prologue buffer.
    prologue: Option<(u32, Option<u32>)>,
}

struct Inflight {
    tick: u64,
    /// Ring position after the tick's last query.
    end: u32,
    open: Open,
}

/// Exact split of command buffers' GPU time into labelled segments.
#[derive(Default)]
struct Split {
    /// Per label: (ns, segments, longest ns).
    labels: HashMap<u64, (u64, u64, u64)>,
}

impl Split {
    /// Charge the segments closed at `ends` (timestamps, the buffer's start
    /// first): segment `i` gets `ends[i+1] - ends[i]` (a timestamp written
    /// before the one ahead of it — never by the GPU — counts as 0).
    /// Returns the buffer's time and its largest segment.
    fn charge(&mut self, start: u64, marks: &[(u64, u64)], period_ns: f64) -> (u64, (u64, u64)) {
        let mut cursor = start;
        let mut largest = (0u64, 0u64);
        for &(t, l) in marks {
            let ns = ((t.saturating_sub(cursor)) as f64 * period_ns) as u64;
            cursor = cursor.max(t);
            let e = self.labels.entry(l).or_default();
            e.0 += ns;
            e.1 += 1;
            e.2 = e.2.max(ns);
            if ns > largest.0 {
                largest = (ns, l);
            }
        }
        (((cursor.saturating_sub(start)) as f64 * period_ns) as u64, largest)
    }
}

pub struct GpuTimer {
    dev: Arc<GpuDevice>,
    pool: vk::QueryPool,
    period_ns: f64,
    /// Next free ring position and the first one still in flight.
    head: u32,
    tail: u32,
    open: Option<Open>,
    inflight: VecDeque<Inflight>,
    pub time: GpuTime,
    split: Split,
    /// The active label and whether commands were recorded under it since
    /// the last timestamp.
    label: u64,
    touched: bool,
    /// Timestamps a full ring could not write.
    pub dropped: u64,
    logged: u32,
}

impl GpuTimer {
    pub fn new(dev: &Arc<GpuDevice>) -> Option<GpuTimer> {
        if dev.caps.timestamp_period <= 0.0 || !dev.caps.host_query_reset {
            return None;
        }
        let info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(QUERIES);
        // SAFETY: valid create info.
        let pool = unsafe { dev.device.create_query_pool(&info, None) }.ok()?;
        // SAFETY: the pool is not in use; host query reset is enabled.
        unsafe { dev.device.reset_query_pool(pool, 0, QUERIES) };
        Some(GpuTimer {
            dev: dev.clone(),
            pool,
            period_ns: dev.caps.timestamp_period as f64,
            head: 0,
            tail: 0,
            open: None,
            inflight: VecDeque::new(),
            time: GpuTime::default(),
            split: Split::default(),
            label: label::OTHER,
            touched: false,
            dropped: 0,
            logged: 0,
        })
    }

    fn free(&self) -> u32 {
        QUERIES - (self.head.wrapping_sub(self.tail) % QUERIES) - 1
    }

    /// A query of the ring, keeping `reserve` more for the buffers' ends.
    fn take(&mut self, reserve: u32) -> Option<u32> {
        if self.free() <= reserve {
            self.dropped += 1;
            return None;
        }
        let q = self.head;
        self.head = (self.head + 1) % QUERIES;
        Some(q)
    }

    fn write(&self, cb: vk::CommandBuffer, stage: vk::PipelineStageFlags2, q: u32) {
        // SAFETY: recording buffer; the query was reset on the host after
        // its last read and timestamps are allowed in and out of rendering
        // scopes.
        unsafe { self.dev.device.cmd_write_timestamp2(cb, stage, self.pool, q) };
    }

    /// The recording command buffer was begun: its start timestamp.
    pub fn begin(&mut self, cb: vk::CommandBuffer) {
        self.touched = false;
        let Some(q) = self.take(3) else { return };
        self.write(cb, vk::PipelineStageFlags2::TOP_OF_PIPE, q);
        self.open.get_or_insert_with(Open::default).start = Some(q);
    }

    /// The prologue buffer of the open tick was begun.
    pub fn prologue_begin(&mut self, cb: vk::CommandBuffer) {
        let Some(q) = self.take(3) else { return };
        self.write(cb, vk::PipelineStageFlags2::TOP_OF_PIPE, q);
        self.open.get_or_insert_with(Open::default).prologue = Some((q, None));
    }

    /// The prologue buffer is about to end.
    pub fn prologue_end(&mut self, cb: vk::CommandBuffer) {
        if !self.open.as_ref().is_some_and(|o| o.prologue.is_some()) {
            return;
        }
        let Some(q) = self.take(0) else { return };
        self.write(cb, vk::PipelineStageFlags2::ALL_COMMANDS, q);
        if let Some(p) = self.open.as_mut().and_then(|o| o.prologue.as_mut()) {
            p.1 = Some(q);
        }
    }

    /// Commands are being recorded into the recording buffer.
    #[inline]
    pub fn touch(&mut self) {
        self.touched = true;
    }

    /// Make `l` the active label (`cb`: the recording buffer, if begun);
    /// returns the label it replaces. Work recorded under the old label
    /// is closed by a timestamp first.
    pub fn enter(&mut self, cb: Option<vk::CommandBuffer>, l: u64) -> u64 {
        let prev = self.label;
        if prev == l || prev == label::OVERFLOW {
            return prev;
        }
        if let (Some(cb), true) = (cb, self.touched) {
            if self.open.as_ref().is_some_and(|o| o.start.is_some()) {
                match self.take(2) {
                    Some(q) => {
                        self.write(cb, vk::PipelineStageFlags2::ALL_COMMANDS, q);
                        self.open.as_mut().unwrap().marks.push((q, prev));
                    }
                    None => {
                        // The rest of the buffer is charged as one segment.
                        self.label = label::OVERFLOW;
                        return prev;
                    }
                }
            }
        }
        self.touched = false;
        self.label = l;
        prev
    }

    /// The recording command buffer is about to end.
    pub fn end(&mut self, cb: vk::CommandBuffer) {
        let l = self.label;
        if l == label::OVERFLOW {
            self.label = label::OTHER;
        }
        self.touched = false;
        if !self.open.as_ref().is_some_and(|o| o.start.is_some()) {
            return;
        }
        // The two queries `begin` kept back.
        let Some(q) = self.take(0) else { return };
        self.write(cb, vk::PipelineStageFlags2::ALL_COMMANDS, q);
        self.open.as_mut().unwrap().marks.push((q, l));
    }

    /// The command buffers of the open tick were submitted as `tick`.
    pub fn submitted(&mut self, tick: u64) {
        if let Some(open) = self.open.take() {
            self.inflight.push_back(Inflight { tick, end: self.head, open });
        }
    }

    /// Timestamps of ring positions `from..to` (wrapping), 0 where the
    /// query has no result.
    fn read_range(&self, from: u32, to: u32) -> Vec<u64> {
        let mut out = Vec::new();
        let mut get = |first: u32, n: u32| {
            if n == 0 {
                return;
            }
            let mut data = vec![[0u64; 2]; n as usize];
            // SAFETY: the queries were written by completed submissions (or
            // never, which WITH_AVAILABILITY reports as 0).
            let r = unsafe {
                self.dev.device.get_query_pool_results(self.pool, first, &mut data, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY)
            };
            out.extend(data.iter().map(|d| if r.is_ok() && d[1] != 0 { d[0] } else { 0 }));
            // SAFETY: the queries are no longer in use.
            unsafe { self.dev.device.reset_query_pool(self.pool, first, n) };
        };
        if to >= from {
            get(from, to - from);
        } else {
            get(from, QUERIES - from);
            get(0, to);
        }
        out
    }

    /// Read back the command buffers of ticks up to `done`.
    pub fn collect(&mut self, done: u64) {
        while self.inflight.front().is_some_and(|f| f.tick <= done) {
            let f = self.inflight.pop_front().unwrap();
            let base = self.tail;
            let ts = self.read_range(base, f.end);
            self.tail = f.end;
            let at = |q: u32| ts.get((q.wrapping_sub(base) % QUERIES) as usize).copied().unwrap_or(0);
            let mut ns = 0u64;
            let mut largest = (0u64, 0u64);
            if let Some(start) = f.open.start.map(at).filter(|&s| s != 0) {
                let marks: Vec<(u64, u64)> = f.open.marks.iter().map(|&(q, l)| (at(q), l)).filter(|m| m.0 != 0).collect();
                (ns, largest) = self.split.charge(start, &marks, self.period_ns);
            }
            if let Some((s, Some(e))) = f.open.prologue {
                let (s, e) = (at(s), at(e));
                if s != 0 && e != 0 {
                    ns += self.split.charge(s, &[(e, label::PROLOGUE)], self.period_ns).0;
                }
            }
            if ns == 0 {
                continue;
            }
            self.time.busy_ns += ns;
            self.time.max_ns = self.time.max_ns.max(ns);
            self.time.buffers += 1;
            self.time.long += (ns >= 16_000_000) as u64;
            if ns >= 16_000_000 && self.logged < 40 {
                self.logged += 1;
                log::info!(
                    "ps5gpu: command buffer of tick {tick}: {:.1} ms in {} segments, largest {:.1} ms ({})",
                    ns as f64 / 1e6,
                    f.open.marks.len(),
                    largest.0 as f64 / 1e6,
                    label_text(largest.1),
                    tick = f.tick
                );
            }
        }
    }

    /// GPU time per label so far: (label, (ns, segments, longest ns)).
    pub fn labels(&self) -> Vec<(u64, (u64, u64, u64))> {
        self.split.labels.iter().map(|(&l, &v)| (l, v)).collect()
    }

    /// The categories and the shaders with the most GPU time:
    /// `label:ms/segments/longest ms`.
    pub fn top(&self, n: usize) -> String {
        let fmt = |(l, (ns, k, max)): (&u64, &(u64, u64, u64))| format!("{}:{}ms/{k}/{:.1}ms", label_text(*l), ns / 1_000_000, *max as f64 / 1e6);
        let mut cats: Vec<_> = self.split.labels.iter().filter(|(l, _)| **l < label::CATEGORIES).collect();
        cats.sort_unstable_by(|a, b| b.1 .0.cmp(&a.1 .0));
        let mut retiles: Vec<_> = self.split.labels.iter().filter(|(l, _)| **l & label::RETILE_AT != 0).collect();
        retiles.sort_unstable_by(|a, b| b.1 .0.cmp(&a.1 .0));
        let retile_ns: u64 = retiles.iter().map(|r| r.1 .0).sum();
        let mut shaders: Vec<_> = self.split.labels.iter().filter(|(l, _)| **l >= label::CATEGORIES && **l & label::RETILE_AT == 0).collect();
        shaders.sort_unstable_by(|a, b| b.1 .0.cmp(&a.1 .0));
        let total: u64 = shaders.iter().map(|s| s.1 .0).sum();
        let rfmt = |(l, (ns, k, max)): (&u64, &(u64, u64, u64))| format!("{:#x}:{}ms/{k}/{:.1}ms", *l & !label::RETILE_AT, ns / 1_000_000, *max as f64 / 1e6);
        format!(
            "{}; categories {}; draws/dispatches {} ms; retiles by target {} ms: {}{}",
            shaders.into_iter().take(n).map(fmt).collect::<Vec<_>>().join(" "),
            cats.into_iter().map(fmt).collect::<Vec<_>>().join(" "),
            total / 1_000_000,
            retile_ns / 1_000_000,
            retiles.into_iter().take(n * 2).map(rfmt).collect::<Vec<_>>().join(" "),
            if self.dropped != 0 { format!(", {} timestamps dropped", self.dropped) } else { String::new() }
        )
    }
}

impl Drop for GpuTimer {
    fn drop(&mut self) {
        // SAFETY: the owner waited for the device before dropping.
        unsafe { self.dev.device.destroy_query_pool(self.pool, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_split_the_buffer_exactly() {
        let mut s = Split::default();
        // Start 100; segments end at 130 (A), 130 (B, empty), 190 (A), 200 (C).
        let (ns, largest) = s.charge(100, &[(130, 7), (130, 9), (190, 7), (200, 0x2000)], 1.0);
        assert_eq!(ns, 100);
        assert_eq!(largest, (60, 7));
        let total: u64 = s.labels.values().map(|v| v.0).sum();
        assert_eq!(total, ns);
        assert_eq!(s.labels[&7], (90, 2, 60));
        assert_eq!(s.labels[&9], (0, 1, 0));
        assert_eq!(s.labels[&0x2000], (10, 1, 10));
    }

    #[test]
    fn a_timestamp_behind_the_cursor_counts_nothing() {
        let mut s = Split::default();
        let (ns, _) = s.charge(100, &[(150, 1), (140, 2), (160, 3)], 2.0);
        assert_eq!(ns, 120);
        assert_eq!(s.labels[&2].0, 0);
        assert_eq!(s.labels[&3].0, 20);
    }

    #[test]
    fn category_names() {
        assert_eq!(label_text(label::IMG_RETILE), "img_retile");
        assert_eq!(label_text(0x20555c0000), "0x20555c0000");
        assert!(label_name(label::CATEGORIES).is_none());
    }
}
