// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The slice thread's side: per-queue register files rebuilt from each
//! slice's snapshot, the log replayed into them, draws resolved and
//! handed to the executor in order.

use super::reglog::{apply_fixup, guest_read, Entry, FixupValues, InstanceSource, Slice, Snapshot};
use super::stats;
use crate::render::regfile::RegisterFile;
use crate::render::{resolve_dispatch, resolve_draw, resolve_draw_into, DirtyBanks, RenderExecutor, ResolvedDraw};
use std::sync::atomic::Ordering::Relaxed;

/// One queue's state on the slice thread.
struct QueueState {
    regs: RegisterFile,
    /// The last resolved draw (the next one re-resolves what changed).
    cache: Option<ResolvedDraw>,
    /// Values of the fixups executed (by id) that snapshots may carry.
    fixups: Vec<(u64, FixupValues)>,
    /// `RPS5E_SLICE_VERIFY`: the state rebuilt from a snapshot, compared
    /// with the state the log left.
    scratch: Option<RegisterFile>,
}

/// Replays slices in order (one per call).
pub struct Runner {
    queues: Vec<Option<Box<QueueState>>>,
    verify: bool,
    read: fn(u64, &mut [u8]) -> bool,
    /// Snapshots that differed from the state the log left (`verify`).
    mismatches: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl Default for Runner {
    fn default() -> Self {
        Runner::new(super::verify())
    }
}

impl Runner {
    pub fn new(verify: bool) -> Runner {
        Runner { queues: Vec::new(), verify, read: guest_read, mismatches: Default::default() }
    }

    /// The count of snapshots that differed from the state the log left.
    pub fn mismatches(&self) -> std::sync::Arc<std::sync::atomic::AtomicU64> {
        self.mismatches.clone()
    }

    /// Indirect-argument records are read with `read` (tests).
    pub fn with_reader(mut self, read: fn(u64, &mut [u8]) -> bool) -> Runner {
        self.read = read;
        self
    }

    fn state(&mut self, queue: usize) -> &mut QueueState {
        if self.queues.len() <= queue {
            self.queues.resize_with(queue + 1, || None);
        }
        self.queues[queue].get_or_insert_with(|| Box::new(QueueState { regs: RegisterFile::new(), cache: None, fixups: Vec::new(), scratch: None }))
    }

    /// The register file of `queue` as the slices so far left it.
    pub fn registers(&self, queue: usize) -> Option<&RegisterFile> {
        self.queues.get(queue)?.as_ref().map(|s| &s.regs)
    }

    fn load(&mut self, queue: usize, snap: &Snapshot) {
        let verify = self.verify;
        let mismatches = self.mismatches.clone();
        let st = self.state(queue);
        // Fixups the snapshot no longer carries are not needed again.
        st.fixups.retain(|(id, _)| snap.fixups.iter().any(|f| f.id == *id));
        let fixups = std::mem::take(&mut st.fixups);
        let values = |id: u64| fixups.iter().find(|(i, _)| *i == id).map(|(_, v)| *v);
        let complete = if verify {
            let scratch = st.scratch.get_or_insert_with(RegisterFile::new);
            let complete = snap.load_into(scratch, &values);
            let same = scratch.same_state(&st.regs);
            std::mem::swap(&mut st.regs, scratch);
            if !same {
                mismatches.fetch_add(1, Relaxed);
                stats::VERIFY_MISMATCHES.fetch_add(1, Relaxed);
                crate::log_once("ps5gpu: slice snapshot differs from the state the register log left (RPS5E_SLICE_VERIFY)");
            }
            complete
        } else {
            snap.load_into(&mut st.regs, &values)
        };
        st.fixups = fixups;
        if !complete {
            stats::FIXUPS_MISSING.fetch_add(1, Relaxed);
        }
        stats::SNAPSHOTS.fetch_add(1, Relaxed);
    }

    /// Execute `slice` through `ex` (its log entries are consumed; the
    /// allocations stay for reuse).
    pub fn run(&mut self, slice: &mut Slice, ex: &dyn RenderExecutor) {
        if let Some(snap) = slice.snapshot.take() {
            self.load(slice.queue, &snap);
            slice.snapshot = Some(snap);
        }
        let read = self.read;
        let queue = slice.queue;
        let mut entries = std::mem::take(&mut slice.entries);
        for e in entries.drain(..) {
            match e {
                Entry::Run { bank, offset, at, len } => {
                    let st = self.state(queue);
                    st.regs.set_run(bank, offset, &slice.values[at as usize..(at + len) as usize]);
                }
                Entry::Table { bank, at, len } => {
                    let st = self.state(queue);
                    st.regs.table(bank, &slice.values[at as usize..(at + len) as usize]);
                }
                Entry::ContextState(op) => {
                    let _ = self.state(queue).regs.context_state(op);
                }
                Entry::Reset => {
                    let st = self.state(queue);
                    st.regs.reset();
                    st.fixups.clear();
                }
                Entry::NumInstances(n) => self.state(queue).regs.num_instances = n,
                Entry::Draw(i) => {
                    let rec = slice.draws[i as usize];
                    let st = self.state(queue);
                    let mut p = rec.packet;
                    match rec.instances {
                        InstanceSource::Packet => {}
                        InstanceSource::State => p.instance_count = st.regs.num_instances.max(1),
                        InstanceSource::StateRaw => p.instance_count = st.regs.num_instances,
                    }
                    if rec.fixup != 0 {
                        let v = apply_fixup(&p, &mut st.regs, &read);
                        st.fixups.push((rec.fixup, v));
                        stats::FIXUPS.fetch_add(1, Relaxed);
                    }
                    let banks = st.regs.banks();
                    let resolved = match &mut st.cache {
                        Some(d) => {
                            resolve_draw_into(&banks, &p, st.regs.dirty, d);
                            d
                        }
                        cache => cache.insert(resolve_draw(&banks, &p)),
                    };
                    st.regs.dirty = DirtyBanks::NONE;
                    ex.draw(resolved);
                }
                Entry::Dispatch(i) => {
                    let p = slice.dispatches[i as usize];
                    let resolved = resolve_dispatch(&self.state(queue).regs.banks(), &p);
                    ex.dispatch(&resolved);
                }
                Entry::Resolved(d) => ex.draw(&d),
                Entry::ResolvedDispatch(d) => ex.dispatch(&d),
                Entry::EndOfPipe(f) => ex.end_of_pipe(f),
                Entry::AfterPipe(f) => ex.after_pipe(f),
                Entry::MemoryFilled(a, n, v) => ex.memory_filled(a, n, v),
                Entry::Flush => ex.flush(),
                Entry::SubmissionComplete => ex.submission_complete(),
                Entry::SyncPoint => ex.sync_point(),
                Entry::Call(f) => f(),
            }
        }
        slice.entries = entries;
    }
}
