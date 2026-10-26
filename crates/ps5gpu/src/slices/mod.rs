// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Slice recording, stage 1 (GPU_PLAN §63, design R7b).
//!
//! The PM4 interpreter thread (`gpu-cmd`) becomes a sequencer: it still
//! parses every packet, follows control flow and does the command
//! processor's memory reads in order, but register writes, draws and
//! dispatches go into a compact log instead of being resolved and recorded
//! there. The log is cut into slices (hard cuts at dispatches, end of pipe,
//! flips and submission ends; soft cuts near `RPS5E_SLICE_DRAWS` draws at a
//! render-target change), each with a snapshot of its queue's registers.
//! One slice thread (`gpu-slice-0`) rebuilds the registers from the
//! snapshot, replays the log, resolves the draws and records them through
//! the back-end, in order: the interpretation runs next to the recording.
//!
//! Command-processor writes, fills and copies the sequencer queued behind
//! earlier work are positioned events ([`pending`]): the sequencer's own
//! reads see their known bytes, or wait (drain) for a copy. One-shot
//! control decisions read memory only once the GPU work before them is
//! recorded. `RPS5E_SLICES=1` turns slice mode on (default off).
//!
//! The frame-capture replay drives the same cuts and queue
//! (`rps5e-cli replay --slices 1`).

pub mod cut;
pub mod frozen;
pub mod exec;
pub mod reglog;
pub(crate) mod pending;
pub mod queue;
pub mod run;
pub mod pool;
pub mod spec;
#[cfg(test)]
mod tests;

pub use exec::SliceExecutor;
pub use reglog::{Cut, InstanceSource, Slice};
pub use run::Runner;

use std::sync::OnceLock;

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

/// `RPS5E_SLICES=K`: slice threads (0 = off, the serial recorder). The
/// game's command thread runs one (stage 1); the capture replay records
/// with K threads in waves (stage 2, `rps5e-cli replay --slices K`).
pub fn configured() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| env_u64("RPS5E_SLICES").unwrap_or(0).min(16) as u32)
}

/// `RPS5E_SLICE_DRAWS`: the soft-cut budget (draws per slice).
pub fn budget() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| env_u64("RPS5E_SLICE_DRAWS").filter(|&n| n >= 1).unwrap_or(128) as u32)
}

/// `RPS5E_SLICE_WAVE`: slices per wave and slice thread (stage 2): the
/// threads take a wave's slices as they become free.
pub fn wave_slices(threads: usize) -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    threads * *N.get_or_init(|| env_u64("RPS5E_SLICE_WAVE").filter(|&n| n >= 1).unwrap_or(4) as usize)
}

/// `RPS5E_SLICE_AFFINITY=1`: a wave's slices go to fixed slice threads
/// (contiguous runs, by position in the wave) instead of whichever thread
/// is free: the same draws meet the same thread's memos, recipes and
/// lookups every frame.
pub fn affinity() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_SLICE_AFFINITY").is_ok_and(|v| v == "1"))
}

/// The slice thread of slice `j` of a wave of `n` slices among `k`
/// threads (affinity mode): contiguous runs of about `n / k` slices.
pub fn slice_owner(j: usize, n: usize, k: usize) -> usize {
    if n == 0 || k == 0 {
        0
    } else {
        j * k / n
    }
}

/// `RPS5E_SLICE_QUEUE`: slices handed over and not yet executed, at most.
pub fn depth() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| env_u64("RPS5E_SLICE_QUEUE").filter(|&n| n >= 1).unwrap_or(4) as usize)
}

/// `RPS5E_SLICE_VERIFY=1`: every snapshot is also compared with the
/// registers the slice thread's log replay left (exact mode).
pub fn verify() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_SLICE_VERIFY").is_ok_and(|v| v == "1"))
}

/// Counters (process-wide).
pub mod stats {
    use std::sync::atomic::AtomicU64;

    pub static SLICES: AtomicU64 = AtomicU64::new(0);
    pub static SLICES_RUN: AtomicU64 = AtomicU64::new(0);
    pub static DRAWS: AtomicU64 = AtomicU64::new(0);
    pub static LOG_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static CUTS: [AtomicU64; super::Cut::COUNT] = [const { AtomicU64::new(0) }; super::Cut::COUNT];
    pub const DRAIN_RESULT: usize = 0;
    pub const DRAIN_CALL: usize = 1;
    pub const DRAIN_HAZARD: usize = 2;
    pub const DRAIN_COPY: usize = 3;
    pub const DRAIN_CONTROL: usize = 4;
    pub const DRAIN_NAMES: [&str; 5] = ["result", "call", "hazard", "copy", "control"];
    pub static DRAINS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
    /// Time the sequencer waited in drains / for room in the queue.
    pub static DRAIN_NS: AtomicU64 = AtomicU64::new(0);
    pub static SEQ_WAIT_NS: AtomicU64 = AtomicU64::new(0);
    /// CPU time of the slice thread executing slices.
    pub static BUSY_NS: AtomicU64 = AtomicU64::new(0);
    pub static SNAPSHOTS: AtomicU64 = AtomicU64::new(0);
    pub static FIXUPS: AtomicU64 = AtomicU64::new(0);
    pub static FIXUPS_MISSING: AtomicU64 = AtomicU64::new(0);
    pub static VERIFY_MISMATCHES: AtomicU64 = AtomicU64::new(0);
}

/// The `slices=…` part of the `[gpu]` line (None when slice mode never ran).
pub fn summary() -> Option<String> {
    use std::sync::atomic::Ordering::Relaxed;
    let slices = stats::SLICES.load(Relaxed);
    if slices == 0 {
        return None;
    }
    let cuts: Vec<String> = Cut::NAMES.iter().zip(&stats::CUTS).skip(1).map(|(n, c)| format!("{n}:{}", c.load(Relaxed))).collect();
    let drains: Vec<String> = stats::DRAIN_NAMES.iter().zip(&stats::DRAINS).map(|(n, c)| format!("{n}:{}", c.load(Relaxed))).collect();
    let draws = stats::DRAWS.load(Relaxed);
    Some(format!(
        "slices={} run={} draws={} log={}B/slice cuts=[{}] drains=[{}] drain_ms={} seq_wait_ms={} slice_busy_ms={} snapshots={} fixups={}{}{}",
        slices,
        stats::SLICES_RUN.load(Relaxed),
        draws,
        stats::LOG_BYTES.load(Relaxed) / slices,
        cuts.join(" "),
        drains.join(" "),
        stats::DRAIN_NS.load(Relaxed) / 1_000_000,
        stats::SEQ_WAIT_NS.load(Relaxed) / 1_000_000,
        stats::BUSY_NS.load(Relaxed) / 1_000_000,
        stats::SNAPSHOTS.load(Relaxed),
        stats::FIXUPS.load(Relaxed),
        match stats::FIXUPS_MISSING.load(Relaxed) {
            0 => String::new(),
            n => format!(" fixups_missing={n}"),
        },
        match stats::VERIFY_MISMATCHES.load(Relaxed) {
            0 => String::new(),
            n => format!(" VERIFY_MISMATCHES={n}"),
        },
    ))
}
