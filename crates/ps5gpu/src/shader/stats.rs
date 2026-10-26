// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Compiler counters (allocation-free, any thread): translations, on-disk
//! cache hits and the time per compiler phase and optimizer pass. Printed
//! with the pipeline summary line (`shaders=[...]`).

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
pub struct Counter {
    pub count: AtomicU64,
    pub nanos: AtomicU64,
    pub max_nanos: AtomicU64,
}

impl Counter {
    pub const fn new() -> Counter {
        Counter { count: AtomicU64::new(0), nanos: AtomicU64::new(0), max_nanos: AtomicU64::new(0) }
    }

    pub fn add(&self, ns: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.nanos.fetch_add(ns, Ordering::Relaxed);
        self.max_nanos.fetch_max(ns, Ordering::Relaxed);
    }

    /// Time `f` into this counter.
    pub fn time<R>(&self, f: impl FnOnce() -> R) -> R {
        let t = std::time::Instant::now();
        let r = f();
        self.add(t.elapsed().as_nanos() as u64);
        r
    }

    fn text(&self, name: &str) -> String {
        format!(
            "{name} {}/{}ms/max {}ms",
            self.count.load(Ordering::Relaxed),
            self.nanos.load(Ordering::Relaxed) / 1_000_000,
            self.max_nanos.load(Ordering::Relaxed) / 1_000_000
        )
    }
}

/// Guest programs translated to SPIR-V (cache misses).
pub static TRANSLATE: Counter = Counter::new();
/// Modules read from the on-disk module cache.
pub static DISK_HITS: Counter = Counter::new();

/// Translation phases: the front end's ([`ps5shader::translate::PHASES`]),
/// then resource-use attachment and SPIR-V emission (ns).
pub const PHASE_NAMES: [&str; 8] = ["cfg", "structurize", "translate", "verify", "ssa", "opt", "uses", "emit"];
pub static PHASE_NS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Wave64 compute programs on a 32-wide host subgroup translated split
/// ([`super::wave_split`]) and paired.
pub static WAVE64_SPLIT: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

pub fn note_split(split: bool) {
    WAVE64_SPLIT[!split as usize].fetch_add(1, Ordering::Relaxed);
}

/// Optimizer passes ([`ps5shader::opt::PASSES`], ns).
pub static PASS_NS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

fn times(names: &[&str], values: &[AtomicU64]) -> String {
    names.iter().zip(values).map(|(n, v)| format!("{n} {}", v.load(Ordering::Relaxed) / 1_000_000)).collect::<Vec<_>>().join(" ")
}

pub fn passes_text() -> String {
    times(&ps5shader::opt::PASSES, &PASS_NS)
}

pub fn phases_text() -> String {
    times(&PHASE_NAMES, &PHASE_NS)
}

pub fn line() -> String {
    let (split, paired) = (WAVE64_SPLIT[0].load(Ordering::Relaxed), WAVE64_SPLIT[1].load(Ordering::Relaxed));
    let waves = if split + paired > 0 { format!("; wave64 split {split}/paired {paired}") } else { String::new() };
    format!(" shaders=[{}, {}; phases ms: {}; passes ms: {}{waves}]", TRANSLATE.text("translate"), DISK_HITS.text("disk"), phases_text(), passes_text())
}

/// The parts of one translation timed for [`last`]: the front end's phases
/// ([`PHASE_NAMES`] up to `opt`), then the wave64 split, the EXEC runs,
/// resource uses and emission.
pub const LAST_NAMES: [&str; 10] = ["cfg", "structurize", "translate", "verify", "ssa", "opt", "split", "exec_runs", "uses", "emit"];

thread_local! {
    static LAST: std::cell::Cell<[u64; 10]> = const { std::cell::Cell::new([0; 10]) };
}

/// Record the times (ns, [`LAST_NAMES`]) of the translation this thread
/// just made.
pub(crate) fn set_last(ns: [u64; 10]) {
    LAST.with(|l| l.set(ns));
}

/// The times (ns, [`LAST_NAMES`]) of the last translation of this thread
/// (`shader-cache translate`: where one program's time goes).
pub fn last() -> [u64; 10] {
    LAST.with(|l| l.get())
}
