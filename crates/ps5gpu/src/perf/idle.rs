// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! GPU idle time by what the recorder did meanwhile (`RPS5E_IDLE_WHY=1`,
//! diagnostics; needs the GPU profile's timer, `RPS5E_GPU_PROFILE=1`).
//!
//! The recording thread logs its state changes with their host time:
//! idle on an empty queue, recording a draw or a dispatch (with its
//! program), waiting for the GPU or for the back-end lock inside one, or
//! other work. The GPU timer reads back each command buffer's GPU span;
//! the idle gap before a buffer ends when that buffer was submitted (a
//! single in-order queue starts a submission at once when it has nothing
//! else), so the gap's host interval is `[submit - gap, submit]`. The
//! recorder's states over that interval are summed per state and program:
//! recorder time spent while the GPU had nothing to run, which is the
//! recorder work on the critical path. The PM4 interpreter logs its own
//! waits (no work queued, every queue head blocked), so time the recorder
//! idled on an empty queue is split by what the interpreter did. A
//! `[idle-why]` line every ten seconds lists the totals and the programs
//! with the most such time.

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, OnceLock};

pub const OTHER: u8 = 0;
pub const IDLE: u8 = 1;
pub const DRAW: u8 = 2;
pub const DISPATCH: u8 = 3;
pub const GPU_WAIT: u8 = 4;
pub const LOCK: u8 = 5;
// `OTHER` work of the recording thread carries a code in place of a
// program: 1 end of pipe, 2 after pipe, 3 memory filled, 4 flush, 5
// submission complete, 6 sync point, 7 call (CP writes, reads, drains).
const NAMES: [&str; 6] = ["other", "idle", "draw", "dispatch", "gpu-wait", "lock"];

/// States of the PM4 interpreter ([`interpreter`]).
pub const CP_RUN: u8 = 0;
pub const CP_NO_WORK: u8 = 1;
pub const CP_BLOCKED: u8 = 2;
const CP_NAMES: [&str; 3] = ["cp-run", "cp-no-work", "cp-blocked"];

std::thread_local! {
    static RECORDER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The calling thread is the one whose states are logged (the recording
/// thread; the interpreter thread when it records itself).
pub fn mark_thread() {
    RECORDER.with(|r| r.set(true));
}

fn is_marked() -> bool {
    RECORDER.try_with(|r| r.get()).unwrap_or(false)
}

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_IDLE_WHY").is_ok_and(|v| v == "1"))
}

fn now_ns() -> u64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64
}

/// How long state entries and submit times are kept (read-backs come
/// within milliseconds of completion).
const KEEP_NS: u64 = 3_000_000_000;

#[derive(Default)]
struct State {
    /// (host ns, state, program), oldest first.
    log: VecDeque<(u64, u8, u64)>,
    /// The interpreter's (host ns, state), oldest first.
    cp: VecDeque<(u64, u8, u64)>,
    /// Recorder-idle ns per interpreter state in the report window.
    idle_cp: [u64; 3],
    /// The recorder's current state (restored by [`leave`]).
    current: (u8, u64),
    /// (tick, host ns of its submit), oldest first.
    submits: VecDeque<(u64, u64)>,
    /// Idle ns per (state, program) in the current report window.
    sums: HashMap<(u8, u64), u64>,
    gap_ns: u64,
    unmatched_ns: u64,
    window_start: u64,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(State::default))
}

fn push(s: &mut State, t: u64, kind: u8, program: u64) {
    s.current = (kind, program);
    trim_push(&mut s.log, (t, kind, program));
}

fn trim_push(log: &mut VecDeque<(u64, u8, u64)>, e: (u64, u8, u64)) {
    log.push_back(e);
    while log.front().is_some_and(|f| f.0 + KEEP_NS < e.0) && log.len() > 1 {
        log.pop_front();
    }
}

/// The PM4 interpreter enters `state` ([`CP_RUN`] after a wait).
pub fn interpreter(state: u8) {
    if enabled() {
        let t = now_ns();
        with(|s| trim_push(&mut s.cp, (t, state, 0)));
    }
}

/// The recording thread enters `kind` (for `program`); returns the state
/// to restore with [`leave`]. A wait inside an item keeps the item's
/// program.
pub fn enter(kind: u8, program: u64) -> Option<(u8, u64)> {
    if !enabled() || !is_marked() {
        return None;
    }
    let t = now_ns();
    Some(with(|s| {
        let prev = s.current;
        let program = if program == 0 && matches!(kind, GPU_WAIT | LOCK) { prev.1 } else { program };
        push(s, t, kind, program);
        prev
    }))
}

/// Back to the state [`enter`] returned.
pub fn leave(prev: Option<(u8, u64)>) {
    if let Some((k, p)) = prev {
        let t = now_ns();
        with(|s| push(s, t, k, p));
    }
}

/// `tick` was submitted now.
pub fn submitted(tick: u64) {
    if !enabled() {
        return;
    }
    let t = now_ns();
    with(|s| {
        s.submits.push_back((tick, t));
        while s.submits.front().is_some_and(|e| e.1 + KEEP_NS < t) {
            s.submits.pop_front();
        }
    });
}

/// Recorder time per state over `[a, b)` from the state log (`log`
/// sorted by time; the state at a time is the latest entry at or before
/// it).
fn overlap(log: &VecDeque<(u64, u8, u64)>, a: u64, b: u64, out: &mut HashMap<(u8, u64), u64>) -> u64 {
    if b <= a || log.is_empty() {
        return 0;
    }
    let mut covered = 0;
    let mut i = log.partition_point(|e| e.0 <= a).saturating_sub(1);
    while i < log.len() && log[i].0 < b {
        let from = log[i].0.max(a);
        let to = log.get(i + 1).map_or(b, |n| n.0.min(b));
        if to > from {
            *out.entry((log[i].1, log[i].2)).or_insert(0) += to - from;
            covered += to - from;
        }
        i += 1;
    }
    covered
}

/// The GPU was idle `gap_ns` before the buffer of `tick` started.
pub fn gap_before(tick: u64, gap_ns: u64) {
    if !enabled() || gap_ns == 0 {
        return;
    }
    let line = with(|s| {
        s.gap_ns += gap_ns;
        let at = s.submits.iter().rev().find(|e| e.0 == tick).map(|e| e.1);
        match at {
            Some(end) => {
                let mut sums = std::mem::take(&mut s.sums);
                let start = end.saturating_sub(gap_ns);
                let covered = overlap(&s.log, start, end, &mut sums);
                s.sums = sums;
                // The recorder's idle segments split by the interpreter's state.
                let mut idle: Vec<(u64, u64)> = Vec::new();
                let mut i = s.log.partition_point(|e| e.0 <= start).saturating_sub(1);
                while i < s.log.len() && s.log[i].0 < end {
                    let (a, b) = (s.log[i].0.max(start), s.log.get(i + 1).map_or(end, |n| n.0.min(end)));
                    if s.log[i].1 == IDLE && b > a {
                        idle.push((a, b));
                    }
                    i += 1;
                }
                for (a, b) in idle {
                    let mut by = HashMap::new();
                    let c = overlap(&s.cp, a, b, &mut by);
                    for ((k, _), ns) in by {
                        s.idle_cp[k as usize] += ns;
                    }
                    s.idle_cp[CP_RUN as usize] += (b - a) - c.min(b - a);
                }
                s.unmatched_ns += gap_ns - covered.min(gap_ns);
            }
            None => s.unmatched_ns += gap_ns,
        }
        let now = now_ns();
        if s.window_start == 0 {
            s.window_start = now;
        }
        if now - s.window_start < 10_000_000_000 {
            return None;
        }
        let secs = (now - s.window_start) as f64 / 1e9;
        s.window_start = now;
        let mut kinds = [0u64; 6];
        for (&(k, _), &ns) in &s.sums {
            kinds[k as usize] += ns;
        }
        let mut top: Vec<(&(u8, u64), &u64)> = s.sums.iter().collect();
        top.sort_by(|a, b| b.1.cmp(a.1));
        let ms = |ns: u64| ns as f64 / 1e6 / secs;
        let mut line = format!("[idle-why] per s: GPU idle {:.1} ms, unmatched {:.1};", ms(s.gap_ns), ms(s.unmatched_ns));
        for (k, n) in kinds.iter().enumerate() {
            line.push_str(&format!(" {} {:.1}", NAMES[k], ms(*n)));
        }
        line.push_str(" | idle by cp:");
        for (k, n) in s.idle_cp.iter().enumerate() {
            line.push_str(&format!(" {} {:.1}", CP_NAMES[k], ms(*n)));
        }
        s.idle_cp = [0; 3];
        line.push_str(" | top:");
        for ((k, p), ns) in top.iter().take(24) {
            line.push_str(&format!(" {}:{p:#x}={:.2}", NAMES[*k as usize], ms(**ns)));
        }
        s.sums.clear();
        s.gap_ns = 0;
        s.unmatched_ns = 0;
        Some(line)
    });
    if let Some(l) = line {
        log::info!("{l}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The recorder's states over an interval are split exactly at the
    /// state changes; time before the first entry is not covered.
    #[test]
    fn overlap_splits_the_interval_at_state_changes() {
        let log: VecDeque<(u64, u8, u64)> = [(10, IDLE, 0), (20, DISPATCH, 0x3c00), (35, GPU_WAIT, 0x3c00), (40, DRAW, 7)].into_iter().collect();
        let mut out = HashMap::new();
        assert_eq!(overlap(&log, 15, 45, &mut out), 30);
        assert_eq!(out[&(IDLE, 0)], 5);
        assert_eq!(out[&(DISPATCH, 0x3c00)], 15);
        assert_eq!(out[&(GPU_WAIT, 0x3c00)], 5);
        assert_eq!(out[&(DRAW, 7)], 5);
        let mut out = HashMap::new();
        assert_eq!(overlap(&log, 0, 12, &mut out), 2, "only from the first entry on");
        assert_eq!(out[&(IDLE, 0)], 2);
        assert_eq!(overlap(&log, 0, 8, &mut out), 0);
    }
}
