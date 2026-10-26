// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The sequencer side: the executor the PM4 interpreters use in slice
//! mode. Register writes, draws and dispatches go into the open slice's
//! log; the command processor's other work (labels, transfers, flips,
//! completion callbacks) is placed between them in the same log; cuts hand
//! the slice to the slice thread, which replays it in order.

use super::cut::{writes_target_base, Cutter};
use super::reglog::{note_fixup, note_instances, note_shader_writes, CarriedFixup, Cut, DrawRecord, Entry, InstanceSource, Slice};
use super::pending::{EventData, PendingEvents};
use super::queue::SliceQueue;
use super::run::Runner;
use super::stats;
use crate::render::regfile::{Bank, ContextState, RegisterFile, REGISTER_SELECTOR_MASK};
use crate::render::{DispatchPacket, DrawPacket, GdsTransfer, OcclusionWrite, RenderExecutor, ResolvedDispatch, ResolvedDraw};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// Recycled slices kept for reuse.
const FREE_SLICES: usize = 16;

struct Open {
    slice: Box<Slice>,
    /// Sequence number of the last slice handed over.
    seq: u64,
    /// The last slice holding draws or dispatches.
    work_seq: u64,
    open_work: bool,
    cutter: Cutter,
    /// Per queue: indirect-draw writes the sequencer's banks lack.
    fixups: Vec<Vec<CarriedFixup>>,
    next_fixup: u64,
}

/// The executor of slice mode (`RPS5E_SLICES=1`).
pub struct SliceExecutor {
    inner: Arc<dyn RenderExecutor>,
    open: Mutex<Open>,
    queue: Arc<SliceQueue<Box<Slice>>>,
    pending: Arc<PendingEvents>,
    free: Arc<Mutex<Vec<Box<Slice>>>>,
    /// Thread id of the slice thread (its own drains cannot wait).
    thread: Arc<AtomicU64>,
    /// The slice thread left its loop.
    exited: Arc<AtomicBool>,
    join: Mutex<Option<std::thread::JoinHandle<()>>>,
}

fn thread_id() -> u64 {
    ps5vm::os::current_tid()
}

impl SliceExecutor {
    /// Start a slice thread that executes through `inner`, with
    /// `budget` draws per slice and `depth` slices in flight.
    pub fn start(inner: Arc<dyn RenderExecutor>, budget: u32, depth: usize, runner: Runner) -> Arc<SliceExecutor> {
        let queue = Arc::new(SliceQueue::new(depth));
        let free: Arc<Mutex<Vec<Box<Slice>>>> = Arc::default();
        let thread = Arc::new(AtomicU64::new(0));
        let exited = Arc::new(AtomicBool::new(false));
        let ex = Arc::new(SliceExecutor {
            inner: inner.clone(),
            open: Mutex::new(Open { slice: Box::default(), seq: 0, work_seq: 0, open_work: false, cutter: Cutter::new(budget), fixups: Vec::new(), next_fixup: 1 }),
            queue: queue.clone(),
            pending: Arc::default(),
            free: free.clone(),
            thread: thread.clone(),
            exited: exited.clone(),
            join: Mutex::new(None),
        });
        let handle = std::thread::Builder::new()
            .name("gpu-slice-0".into())
            .spawn(move || {
                thread.store(thread_id(), Ordering::Release);
                // The recorder's working set wants the large L3.
                ps5vm::topology::pin_current_thread_to_fast_cores("gpu-slice-0");
                let mut runner = runner;
                while let Some((seq, mut slice)) = queue.pop() {
                    let t = crate::backend::profile::thread_cpu();
                    runner.run(&mut slice, &*inner);
                    stats::BUSY_NS.fetch_add(crate::backend::profile::thread_cpu() - t, Relaxed);
                    stats::SLICES_RUN.fetch_add(1, Relaxed);
                    queue.complete(seq);
                    slice.clear();
                    let mut f = free.lock().unwrap_or_else(|e| e.into_inner());
                    if f.len() < FREE_SLICES {
                        f.push(slice);
                    }
                }
                crate::backend::profile::flush_local();
                exited.store(true, Ordering::Release);
            })
            .expect("spawn the GPU slice thread");
        *ex.join.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle);
        ex
    }

    /// Stop the slice thread: queued slices are dropped, waiters return,
    /// and the thread is waited for (bounded: a shutdown hook must not
    /// hang on a slice stuck in the back-end).
    pub fn stop(&self) {
        self.queue.close();
        self.pending.clear();
        let h = self.join.lock().unwrap_or_else(|e| e.into_inner()).take();
        let Some(h) = h else { return };
        if self.thread.load(Ordering::Acquire) == thread_id() {
            return;
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
        while !self.exited.load(Ordering::Acquire) {
            if std::time::Instant::now() >= deadline {
                ::log::warn!("ps5gpu: the slice thread did not stop in time");
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let _ = h.join();
    }

    fn lock(&self) -> MutexGuard<'_, Open> {
        self.open.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn fresh(&self) -> Box<Slice> {
        self.free.lock().unwrap_or_else(|e| e.into_inner()).pop().unwrap_or_default()
    }

    /// Hand the open slice to the slice thread.
    fn ship(&self, o: &mut Open, cut: Cut) {
        o.cutter.reset();
        if o.slice.is_empty() {
            return;
        }
        o.slice.cut = cut;
        stats::CUTS[cut as usize].fetch_add(1, Relaxed);
        let next = self.fresh();
        let slice = std::mem::replace(&mut o.slice, next);
        o.seq += 1;
        if std::mem::take(&mut o.open_work) {
            o.work_seq = o.seq;
        }
        stats::LOG_BYTES.fetch_add(slice.log_bytes() as u64, Relaxed);
        stats::SLICES.fetch_add(1, Relaxed);
        match self.queue.push(o.seq, slice) {
            Some(waited) => {
                stats::SEQ_WAIT_NS.fetch_add(waited, Relaxed);
            }
            // Shut down: the slice was dropped.
            None => self.pending.clear(),
        }
    }

    /// The open slice, ready for register work or a draw of `queue`: cut
    /// when it belongs to another queue; the snapshot of `regs` (the
    /// sequencer's state before this entry) taken when it has none.
    fn prepare<'a>(&self, o: &'a mut Open, queue: usize, regs: &RegisterFile) -> &'a mut Open {
        if o.slice.snapshot.is_some() && o.slice.queue != queue {
            self.ship(o, Cut::Queue);
        }
        if o.slice.snapshot.is_none() {
            if o.fixups.len() <= queue {
                o.fixups.resize_with(queue + 1, Vec::new);
            }
            let mut snap = o.slice.spare.take().unwrap_or_default();
            snap.take(regs, &o.fixups[queue]);
            o.slice.snapshot = Some(snap);
            o.slice.queue = queue;
        }
        o
    }

    fn fixups(o: &mut Open, queue: usize) -> &mut Vec<CarriedFixup> {
        if o.fixups.len() <= queue {
            o.fixups.resize_with(queue + 1, Vec::new);
        }
        &mut o.fixups[queue]
    }

    // ---- the sequencer's register log ----

    /// SET_*_REG: `values` from register `offset` on (`regs`: the
    /// sequencer's state before the write).
    pub fn log_run(&self, queue: usize, regs: &RegisterFile, bank: Bank, offset: u32, values: &[u32]) {
        let mut g = self.lock();
        let o = self.prepare(&mut *g, queue, regs);
        o.slice.push_run(bank, offset, values);
        let n = values.len() as u32;
        match bank {
            Bank::Shader => note_shader_writes(Self::fixups(o, queue), offset, n),
            Bank::Context if writes_target_base(offset, n) => o.cutter.target_written(),
            _ => {}
        }
    }

    /// A register table of `{offset, value}` pairs.
    pub fn log_table(&self, queue: usize, regs: &RegisterFile, bank: Bank, pairs: &[u32]) {
        let mut g = self.lock();
        let o = self.prepare(&mut *g, queue, regs);
        o.slice.push_table(bank, pairs);
        for e in pairs.chunks_exact(2) {
            if e[0] == u32::MAX {
                continue;
            }
            let offset = e[0] & !REGISTER_SELECTOR_MASK;
            match bank {
                Bank::Shader => note_shader_writes(Self::fixups(o, queue), offset, 1),
                Bank::Context if writes_target_base(offset, 1) => o.cutter.target_written(),
                _ => {}
            }
        }
    }

    pub fn log_context_state(&self, queue: usize, regs: &RegisterFile, op: ContextState) {
        let mut g = self.lock();
        let o = self.prepare(&mut *g, queue, regs);
        o.slice.entries.push(Entry::ContextState(op));
        // A clear or pop may change the targets.
        o.cutter.target_written();
    }

    /// Processor reset (frame boundary, queue reset).
    pub fn log_reset(&self, queue: usize, regs: &RegisterFile) {
        let mut g = self.lock();
        let o = self.prepare(&mut *g, queue, regs);
        o.slice.entries.push(Entry::Reset);
        Self::fixups(o, queue).clear();
        o.cutter.target_written();
    }

    pub fn log_num_instances(&self, queue: usize, regs: &RegisterFile, n: u32) {
        let mut g = self.lock();
        let o = self.prepare(&mut *g, queue, regs);
        o.slice.entries.push(Entry::NumInstances(n));
        note_instances(Self::fixups(o, queue));
    }

    /// A draw: `instances` says where its instance count comes from;
    /// `fixup`: its argument record is applied to the registers on the
    /// slice thread.
    pub fn log_draw(&self, queue: usize, regs: &RegisterFile, packet: &DrawPacket, instances: InstanceSource, fixup: bool) {
        let mut g = self.lock();
        let o = &mut *g;
        if let Some(cut) = o.cutter.before_draw() {
            // Cut in front of the draw (the counter restarts with it).
            self.ship(o, cut);
            o.cutter.before_draw();
        }
        let o = self.prepare(o, queue, regs);
        let id = if fixup {
            o.next_fixup += 1;
            o.next_fixup
        } else {
            0
        };
        o.slice.push_draw(DrawRecord { packet: *packet, instances, fixup: id });
        o.open_work = true;
        if fixup {
            note_fixup(Self::fixups(o, queue), CarriedFixup::new(id, packet));
        }
        stats::DRAWS.fetch_add(1, Relaxed);
    }

    /// A dispatch (resolved on the slice thread; a hard cut).
    pub fn log_dispatch(&self, queue: usize, regs: &RegisterFile, packet: &DispatchPacket) {
        let mut g = self.lock();
        let o = self.prepare(&mut *g, queue, regs);
        o.slice.push_dispatch(*packet);
        o.open_work = true;
        self.ship(o, Cut::Hard);
    }

    // ---- events ----

    fn event(&self, e: Entry, cut: Option<Cut>) {
        let mut o = self.lock();
        o.slice.entries.push(e);
        if let Some(c) = cut {
            self.ship(&mut o, c);
        }
    }

    /// Draws or dispatches the slice thread has not executed yet.
    pub fn work_pending(&self) -> bool {
        let o = self.lock();
        o.open_work || self.queue.done() < o.work_seq
    }

    /// Hand over the open slice and wait until everything queued so far
    /// was executed. `why` counts the cause.
    fn drain_for(&self, why: usize) {
        let seq = {
            let mut o = self.lock();
            if !o.slice.is_empty() {
                self.ship(&mut o, Cut::Drain);
            }
            o.seq
        };
        if self.queue.done() >= seq || self.thread.load(Ordering::Acquire) == thread_id() {
            return;
        }
        stats::DRAINS[why].fetch_add(1, Relaxed);
        let t = std::time::Instant::now();
        self.queue.wait_done(seq);
        stats::DRAIN_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    }

    /// Slices handed over so far / executed (tests, diagnostics).
    pub fn progress(&self) -> (u64, u64) {
        (self.lock().seq, self.queue.done())
    }
}

impl RenderExecutor for SliceExecutor {
    fn slices(&self) -> Option<&SliceExecutor> {
        Some(self)
    }

    fn draw(&self, draw: &ResolvedDraw) {
        let mut g = self.lock();
        let o = &mut *g;
        o.cutter.draw_targets(crate::slice_shadow::target_key(draw));
        if let Some(cut) = o.cutter.before_draw() {
            self.ship(o, cut);
            o.cutter.before_draw();
        }
        o.slice.entries.push(Entry::Resolved(Box::new(draw.clone())));
        o.open_work = true;
    }

    fn dispatch(&self, dispatch: &ResolvedDispatch) {
        let mut o = self.lock();
        o.slice.entries.push(Entry::ResolvedDispatch(Box::new(*dispatch)));
        o.open_work = true;
        self.ship(&mut o, Cut::Hard);
    }

    fn end_of_pipe(&self, done: Box<dyn FnOnce() + Send>) {
        self.event(Entry::EndOfPipe(done), Some(Cut::Hard));
    }

    fn after_pipe(&self, done: Box<dyn FnOnce() + Send>) {
        self.event(Entry::AfterPipe(done), None);
    }

    fn writes_through_pending(&self) -> bool {
        self.drain_for(stats::DRAIN_RESULT);
        self.inner.writes_through_pending()
    }

    fn memory_filled(&self, addr: u64, size: u64, value: u32) {
        self.event(Entry::MemoryFilled(addr, size, value), None);
    }

    fn copy_memory(&self, dst: u64, src: u64, size: u64) -> bool {
        self.drain_for(stats::DRAIN_RESULT);
        self.inner.copy_memory(dst, src, size)
    }

    fn fill_memory(&self, dst: u64, size: u64, value: u32) -> bool {
        self.drain_for(stats::DRAIN_RESULT);
        self.inner.fill_memory(dst, size, value)
    }

    fn write_memory(&self, dst: u64, data: &[u8]) -> bool {
        self.drain_for(stats::DRAIN_RESULT);
        self.inner.write_memory(dst, data)
    }

    fn gds_transfer(&self, op: GdsTransfer) -> bool {
        self.drain_for(stats::DRAIN_RESULT);
        self.inner.gds_transfer(op)
    }

    fn occlusion_counter_dump(&self, addr: u64, write: &OcclusionWrite) -> bool {
        self.drain_for(stats::DRAIN_RESULT);
        self.inner.occlusion_counter_dump(addr, write)
    }

    fn flush(&self) {
        self.event(Entry::Flush, Some(Cut::Flush));
    }

    fn sync_point(&self) {
        self.event(Entry::SyncPoint, None);
    }

    fn drain(&self) {
        self.drain_for(stats::DRAIN_CALL);
    }

    fn run_ordered(&self, f: Box<dyn FnOnce() + Send>) {
        self.event(Entry::Call(f), Some(Cut::Hard));
    }

    fn fill_memory_or(&self, dst: u64, size: u64, value: u32, cpu: Box<dyn FnOnce() -> bool + Send>) -> bool {
        let pending = self.pending.clone();
        let serial = pending.add(dst, size, Some(EventData::Pattern(value)));
        let inner = self.inner.clone();
        self.event(
            Entry::Call(Box::new(move || {
                if !inner.fill_memory_or(dst, size, value, cpu) {
                    crate::log_once(&format!("ps5gpu: command-processor fill of {dst:#x} ({size:#x} bytes) outside guest memory"));
                }
                pending.remove(serial);
            })),
            None,
        );
        true
    }

    fn copy_memory_or(&self, dst: u64, src: u64, size: u64, cpu: Box<dyn FnOnce() -> bool + Send>) -> bool {
        let pending = self.pending.clone();
        let serial = pending.add(dst, size, None);
        let inner = self.inner.clone();
        self.event(
            Entry::Call(Box::new(move || {
                if !inner.copy_memory_or(dst, src, size, cpu) {
                    crate::log_once(&format!("ps5gpu: command-processor copy {src:#x} -> {dst:#x} ({size:#x} bytes) outside guest memory"));
                }
                pending.remove(serial);
            })),
            None,
        );
        true
    }

    fn memory_hazard(&self, addr: u64, len: u64) {
        if self.pending.overlaps(addr, len) {
            self.drain_for(stats::DRAIN_HAZARD);
        }
    }

    fn write_ordered(&self, addr: u64, data: &[u8], write: Box<dyn FnOnce() + Send>) {
        // Behind the queued work: a label the CPU sees comes after the
        // draws and transfers before it were recorded.
        let pending = self.pending.clone();
        let serial = pending.add(addr, data.len() as u64, Some(EventData::Bytes(data.to_vec())));
        self.event(
            Entry::Call(Box::new(move || {
                write();
                pending.remove(serial);
            })),
            None,
        );
    }

    fn read_ordered(&self, addr: u64, out: &mut [u8], read: &dyn Fn(&mut [u8]) -> bool) -> bool {
        match self.pending.compose(addr, out, read) {
            Some(ok) => ok,
            None => {
                // A queued copy over the range: its bytes come from memory.
                self.drain_for(stats::DRAIN_COPY);
                read(out)
            }
        }
    }

    fn read_control(&self, addr: u64, out: &mut [u8], read: &dyn Fn(&mut [u8]) -> bool) -> bool {
        // A decision taken once: the GPU work before it is recorded first
        // (its written pages fault and download on the read).
        if self.work_pending() {
            self.drain_for(stats::DRAIN_CONTROL);
        }
        self.read_ordered(addr, out, read)
    }

    fn submission_complete(&self) {
        self.event(Entry::SubmissionComplete, Some(Cut::Flush));
    }
}

impl SliceExecutor {
    /// The process-wide slice executor over the back-end (one per process:
    /// stopped at shutdown before the back-end goes away).
    pub fn start_global() -> Arc<SliceExecutor> {
        let ex = SliceExecutor::start(Arc::new(crate::backend::GpuExecutor), super::budget(), super::depth(), Runner::default());
        let weak = Arc::downgrade(&ex);
        ps5vm::shutdown::on_shutdown(ps5vm::shutdown::ORDER_GPU, "GPU slice thread", move || {
            if let Some(ex) = weak.upgrade() {
                ex.stop();
            }
        });
        ::log::info!("ps5gpu: slice recording on (one slice thread, {} draws per slice, {} slices in flight)", super::budget(), super::depth());
        ex
    }
}
