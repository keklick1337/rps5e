// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Cost probes for recorder-side work that has no counter of its own,
//! printed at the end of the `[gpu-sys]` line as `probes indirect=..
//! layout=..` (cumulative). Counting is on with `RPS5E_GPU_PROFILE` set;
//! otherwise every probe is one relaxed load of a cached flag.
//!
//! - `indirect`: indirect draws whose last argument record the command
//!   processor reads on the CPU for the registers it leaves behind
//!   (calls / ms).
//! - `layout`: image layout requests that walk subresources one by one
//!   (requests / subresources visited / barriers emitted).

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;
use std::time::Instant;

static INDIRECT: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static LAYOUT: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// Probes count only in profiled runs.
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_GPU_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// A start time when probes count.
pub fn start() -> Option<Instant> {
    on().then(Instant::now)
}

fn ns(t: Option<Instant>) -> u64 {
    t.map_or(0, |t| t.elapsed().as_nanos() as u64)
}

/// One CPU read of an indirect argument record started at `t`.
pub fn indirect(t: Option<Instant>) {
    if t.is_some() {
        INDIRECT[0].fetch_add(1, Relaxed);
        INDIRECT[1].fetch_add(ns(t), Relaxed);
    }
}

/// One per-subresource layout request: `visited` subresources, `barriers`
/// barriers emitted.
pub fn layout(visited: u64, barriers: u64) {
    if on() {
        LAYOUT[0].fetch_add(1, Relaxed);
        LAYOUT[1].fetch_add(visited, Relaxed);
        LAYOUT[2].fetch_add(barriers, Relaxed);
    }
}

/// ` probes indirect=calls/ms layout=req/visited/barriers`,
/// or nothing when probes are off.
pub fn summary() -> String {
    if !on() {
        return String::new();
    }
    let l = |a: &AtomicU64| a.load(Relaxed);
    format!(
        " probes indirect={}/{}ms layout={}/{}/{}",
        l(&INDIRECT[0]),
        l(&INDIRECT[1]) / 1_000_000,
        l(&LAYOUT[0]),
        l(&LAYOUT[1]),
        l(&LAYOUT[2])
    )
}
