// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Compile jobs running now, for the log: a translation or pipeline build
//! that takes far longer than any normal one is named while it still runs
//!. A job registers itself with [`start`] and is removed
//! when the returned guard drops; [`check`] (called once per presented
//! frame) logs each job running longer than [`STUCK_AFTER`] once.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A job running this long is named in the log.
pub const STUCK_AFTER: Duration = Duration::from_secs(10);

struct Job {
    id: u64,
    what: String,
    since: Instant,
    warned: bool,
}

static RUNNING: Mutex<Vec<Job>> = Mutex::new(Vec::new());
static NEXT: AtomicU64 = AtomicU64::new(1);
/// When [`check`] last looked (milliseconds since [`epoch`]).
static LAST_CHECK_MS: AtomicU64 = AtomicU64::new(0);

fn epoch() -> Instant {
    static E: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    *E.get_or_init(Instant::now)
}

/// A registered job; dropping it ends the registration.
pub struct Running(u64);

impl Drop for Running {
    fn drop(&mut self) {
        let mut r = RUNNING.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = r.iter().position(|j| j.id == self.0) {
            let j = r.swap_remove(i);
            if j.warned {
                log::warn!("ps5gpu: compile of {} finished after {:.1} s", j.what, j.since.elapsed().as_secs_f64());
            }
        }
    }
}

/// Register a job described by `what` (e.g. `Pixel program 0x1234`).
pub fn start(what: String) -> Running {
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    RUNNING.lock().unwrap_or_else(|e| e.into_inner()).push(Job { id, what, since: Instant::now(), warned: false });
    Running(id)
}

/// Jobs running longer than `limit`, oldest first: (description, time).
pub fn longer_than(limit: Duration) -> Vec<(String, Duration)> {
    let r = RUNNING.lock().unwrap_or_else(|e| e.into_inner());
    let mut v: Vec<(String, Duration)> = r.iter().map(|j| (j.what.clone(), j.since.elapsed())).filter(|(_, t)| *t >= limit).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    v
}

/// Log the jobs running longer than [`STUCK_AFTER`] (each once); looks at
/// most once a second.
pub fn check() {
    let now = epoch().elapsed().as_millis() as u64;
    let last = LAST_CHECK_MS.load(Ordering::Relaxed);
    if now.saturating_sub(last) < 1000 || LAST_CHECK_MS.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    let mut r = RUNNING.lock().unwrap_or_else(|e| e.into_inner());
    for j in r.iter_mut().filter(|j| !j.warned && j.since.elapsed() >= STUCK_AFTER) {
        j.warned = true;
        log::warn!(
            "ps5gpu: compile of {} running for {:.0} s (draws that need it are skipped, a dispatch waits for it)",
            j.what,
            j.since.elapsed().as_secs_f64()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jobs_are_listed_while_they_run() {
        let a = start("Pixel program 0x1000".into());
        let b = start("Vertex program 0x2000".into());
        let names = |l: Vec<(String, Duration)>| l.into_iter().map(|(w, _)| w).collect::<Vec<_>>();
        let all = names(longer_than(Duration::ZERO));
        assert!(all.contains(&"Pixel program 0x1000".to_string()) && all.contains(&"Vertex program 0x2000".to_string()));
        assert!(!names(longer_than(Duration::from_secs(3600))).iter().any(|w| w.contains("0x1000")));
        drop(a);
        let all = names(longer_than(Duration::ZERO));
        assert!(!all.contains(&"Pixel program 0x1000".to_string()));
        assert!(all.contains(&"Vertex program 0x2000".to_string()));
        drop(b);
        check();
    }
}
