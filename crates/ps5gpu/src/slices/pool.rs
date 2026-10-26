// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The slice threads of stage 2 (R7b): K workers that record the slices of
//! one wave concurrently against the frozen back-end, each with its own
//! recorder ([`crate::backend::SliceRecorder`]), pinned to the fast
//! cores. The committing thread hands a wave over while it holds the
//! back-end guard and waits for every slice; workers take the wave's slices
//! in order as they become free (a wave may hold more slices than there are
//! workers). The recorders stay with their workers between waves (the
//! committing thread borrows one to commit its segments).

use crate::backend::{Backend, CpuBytes, SliceItem, SliceRecorder, SliceUnit};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};

/// A slice of a wave: how many of the wave's CPU writes come before it,
/// and its items.
pub type WaveSlice<'a> = (usize, Vec<SliceItem<'a>>);

/// The CPU writes of a wave, in order.
pub type WaveWrites<'a> = Vec<(u64, u64, CpuBytes<'a>)>;

#[derive(Clone, Copy)]
struct BackendPtr(*const Backend);
// SAFETY: the committing thread holds the back-end guard and only waits
// while workers read through the pointer (the frozen discipline).
unsafe impl Send for BackendPtr {}

struct Wave<'a> {
    backend: Option<BackendPtr>,
    /// Serial of the wave (recorders keep the wave's writes they applied).
    serial: u64,
    writes: std::sync::Arc<WaveWrites<'a>>,
    /// Slices not taken yet: (index, slice).
    todo: VecDeque<(usize, WaveSlice<'a>)>,
    /// Slices in the wave (affinity mode: the owner of slice `i` is
    /// `slice_owner(i, slices, threads)`).
    slices: usize,
    affinity: bool,
    /// Units per slice index.
    done: Vec<Option<Vec<SliceUnit>>>,
    /// Slices taken and not finished.
    running: usize,
    stop: bool,
}

// SAFETY: slices hold shared references to resolved draws and capture
// bytes (read-only) and the back-end pointer (see `BackendPtr`).
unsafe impl Send for Wave<'_> {}

pub struct WavePool<'a> {
    wave: Mutex<Wave<'a>>,
    /// Workers wait for slices; the committing thread for the wave's end.
    work: Condvar,
    finished: Condvar,
    recorders: Vec<Mutex<Option<SliceRecorder>>>,
    busy_ns: AtomicU64,
}

impl<'a> WavePool<'a> {
    pub fn new(k: usize) -> WavePool<'a> {
        WavePool {
            wave: Mutex::new(Wave { backend: None, serial: 0, writes: Default::default(), todo: VecDeque::new(), slices: 0, affinity: super::affinity(), done: Vec::new(), running: 0, stop: false }),
            work: Condvar::new(),
            finished: Condvar::new(),
            recorders: (0..k).map(|_| Mutex::new(None)).collect(),
            busy_ns: AtomicU64::new(0),
        }
    }

    pub fn len(&self) -> usize {
        self.recorders.len()
    }

    pub fn is_empty(&self) -> bool {
        self.recorders.is_empty()
    }

    /// Start the workers inside `scope` (they run until [`Self::stop`]).
    pub fn spawn<'s>(&'s self, scope: &'s std::thread::Scope<'s, '_>)
    where
        'a: 's,
    {
        for w in 0..self.recorders.len() {
            std::thread::Builder::new().name(format!("gpu-slice-{w}")).spawn_scoped(scope, move || self.run(w)).expect("spawn a slice thread");
        }
    }

    /// Stop the workers.
    pub fn stop(&self) {
        self.wave.lock().unwrap_or_else(|e| e.into_inner()).stop = true;
        self.work.notify_all();
    }

    /// Drop the recorders (after the workers stopped).
    pub fn drop_recorders(&self) {
        for r in &self.recorders {
            r.lock().unwrap_or_else(|e| e.into_inner()).take();
        }
    }

    /// Record `slices` against `b`, frozen while this runs; units per
    /// slice, in order. None when no recorder can be made (the provider has
    /// no slice instances).
    pub fn record(&self, b: &Backend, writes: WaveWrites<'a>, slices: Vec<WaveSlice<'a>>) -> Option<Vec<Vec<SliceUnit>>> {
        for (w, r) in self.recorders.iter().enumerate() {
            let mut r = r.lock().unwrap_or_else(|e| e.into_inner());
            if r.is_none() {
                *r = Some(b.slice_recorder(w)?);
            }
        }
        let n = slices.len();
        let mut wave = self.wave.lock().unwrap_or_else(|e| e.into_inner());
        wave.backend = Some(BackendPtr(b as *const Backend));
        wave.serial += 1;
        wave.writes = std::sync::Arc::new(writes);
        wave.done = (0..n).map(|_| None).collect();
        wave.todo = slices.into_iter().enumerate().collect();
        wave.slices = n;
        self.work.notify_all();
        while !wave.todo.is_empty() || wave.running != 0 {
            wave = self.finished.wait(wave).unwrap_or_else(|e| e.into_inner());
        }
        wave.backend = None;
        Some(wave.done.drain(..).map(|d| d.unwrap_or_default()).collect())
    }

    /// Affinity mode on or off (tests; the default is `RPS5E_SLICE_AFFINITY`).
    pub fn set_affinity(&self, on: bool) {
        self.wave.lock().unwrap_or_else(|e| e.into_inner()).affinity = on;
    }

    /// Run `f` with worker `w`'s recorder (between waves).
    pub fn with_recorder<R>(&self, w: usize, f: impl FnOnce(&mut SliceRecorder) -> R) -> Option<R> {
        let mut r = self.recorders.get(w)?.lock().unwrap_or_else(|e| e.into_inner());
        r.as_mut().map(f)
    }

    /// CPU time the workers spent recording so far (ns).
    pub fn busy_ns(&self) -> u64 {
        self.busy_ns.load(Ordering::Relaxed)
    }

    fn run(&self, w: usize) {
        super::frozen::mark_slice_thread();
        ps5vm::topology::pin_current_thread_to_fast_cores(&format!("gpu-slice-{w}"));
        let mut wave = self.wave.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if wave.stop {
                return;
            }
            let k = self.recorders.len();
            let next = if wave.affinity {
                let n = wave.slices;
                wave.todo.iter().position(|(i, _)| super::slice_owner(*i, n, k) == w).and_then(|at| wave.todo.remove(at))
            } else {
                wave.todo.pop_front()
            };
            let Some((i, (prefix, items))) = next else {
                wave = self.work.wait(wave).unwrap_or_else(|e| e.into_inner());
                continue;
            };
            let Some(ptr) = wave.backend else { continue };
            let (serial, writes) = (wave.serial, wave.writes.clone());
            wave.running += 1;
            drop(wave);
            let t0 = crate::backend::profile::thread_cpu();
            let units = {
                let _recording = super::frozen::Recording::begin();
                let mut r = self.recorders[w].lock().unwrap_or_else(|e| e.into_inner());
                // SAFETY: see `BackendPtr`.
                let b = unsafe { &*ptr.0 };
                match r.as_mut() {
                    Some(rec) => b.record_slice(rec, serial, &writes[..prefix], &items),
                    None => Vec::new(),
                }
            };
            drop((writes, items));
            self.busy_ns.fetch_add(crate::backend::profile::thread_cpu() - t0, Ordering::Relaxed);
            wave = self.wave.lock().unwrap_or_else(|e| e.into_inner());
            wave.done[i] = Some(units);
            wave.running -= 1;
            if wave.todo.is_empty() && wave.running == 0 {
                self.finished.notify_all();
            }
        }
    }
}
