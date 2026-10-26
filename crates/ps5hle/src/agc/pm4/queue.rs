// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Submission queues and the GPU command thread.
//!
//! One graphics queue (id 0) and 56 compute queues (ids 1..=56, owners
//! 0x20..=0x57). Guest threads enqueue and return; a dedicated host thread
//! runs one slice at a time, round-robin over queue heads. A slice that
//! suspends (unmet wait) is put back at the head of its queue as "blocked";
//! when every head is blocked the thread waits briefly and retries them all.
//!
//! Frame boundaries: `sceAgcSuspendPoint` queues a boundary marker on the
//! graphics queue and blocks the guest while more than
//! [`boundaries_in_flight`] boundaries are still queued ahead of the GPU
//! thread (default 1; `PS5_AGC_SUSPEND_POINTS_IN_FLIGHT=0` makes it
//! synchronous). Retiring a boundary resets the graphics processor.

use super::interp::{Execution, Interpreter, Progress, COMPUTE_QUEUE_BASE};
use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

pub const COMPUTE_QUEUE_COUNT: usize = 56;
pub const QUEUE_COUNT: usize = 1 + COMPUTE_QUEUE_COUNT;
/// Time the GPU thread waited without submissions / with every queue
/// blocked on a wait (status line).
static IDLE_NO_WORK_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static IDLE_BLOCKED_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How long the GPU thread waits before re-polling blocked submissions.
const RETRY_INTERVAL: Duration = Duration::from_micros(100);

/// `RPS5E_PM4_WAKE=0`: blocked queues are polled again only every
/// [`RETRY_INTERVAL`] (and after a submission completed), even when a
/// label or flip that may release them was written meanwhile.
fn wake_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_PM4_WAKE").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
}

/// Labels written and flips queued so far (a slice that changed them may
/// have released a queue waiting on memory).
fn release_count() -> u64 {
    use std::sync::atomic::Ordering::Relaxed;
    super::interp::LABEL_WRITES.load(Relaxed).wrapping_add(super::interp::FLIPS.load(Relaxed))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Commands,
    FrameBoundary,
}

#[derive(Debug)]
struct Submission {
    kind: Kind,
    queue: usize,
    addr: u64,
    dwords: u32,
    exec: Execution,
    started: bool,
    blocked: bool,
    blocked_since: Option<Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceResult {
    NoWork,
    AllBlocked,
    Progressed,
    BlockedWithoutProgress,
    Completed,
}

struct State {
    queues: Vec<VecDeque<Submission>>,
    count: usize,
    next_queue: usize,
    pending_boundaries: usize,
    processing: bool,
    frames: u64,
    /// The GPU thread stopped for process shutdown.
    stopped: bool,
}

pub struct GpuQueue {
    state: Mutex<State>,
    cv: Condvar,
    interpreters: Mutex<Vec<Option<Interpreter>>>,
    boundaries_in_flight: usize,
    worker: OnceLock<()>,
}

/// Boundaries allowed ahead of the GPU thread.
pub fn boundaries_in_flight() -> usize {
    match std::env::var("PS5_AGC_SUSPEND_POINTS_IN_FLIGHT") {
        Ok(v) => v.trim().parse().unwrap_or(1),
        Err(_) => 1,
    }
}

static GLOBAL: OnceLock<GpuQueue> = OnceLock::new();

/// The process-wide queue set (the GPU thread starts on first submit).
pub fn global() -> &'static GpuQueue {
    GLOBAL.get_or_init(|| GpuQueue::new(boundaries_in_flight()))
}

impl GpuQueue {
    pub fn new(boundaries_in_flight: usize) -> Self {
        Self {
            state: Mutex::new(State {
                queues: (0..QUEUE_COUNT).map(|_| VecDeque::new()).collect(),
                count: 0,
                next_queue: 0,
                pending_boundaries: 0,
                processing: false,
                frames: 0,
                stopped: false,
            }),
            cv: Condvar::new(),
            interpreters: Mutex::new((0..QUEUE_COUNT).map(|_| None).collect()),
            boundaries_in_flight,
            worker: OnceLock::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn enqueue(st: &mut State, s: Submission) {
        st.queues[s.queue].push_back(s);
        st.count += 1;
    }

    /// Queue a graphics command buffer.
    pub fn submit_graphics(&self, addr: u64, dwords: u32) {
        if dwords == 0 {
            return;
        }
        let mut st = self.lock();
        Self::enqueue(
            &mut st,
            Submission { kind: Kind::Commands, queue: 0, addr, dwords, exec: Execution::default(), started: false, blocked: false, blocked_since: None },
        );
        self.cv.notify_all();
    }

    /// Queue a compute command buffer for owner/queue `0x20..=0x57`.
    pub fn submit_compute(&self, owner: u32, addr: u64, dwords: u32) -> bool {
        let base = COMPUTE_QUEUE_BASE as u32;
        if owner < base || owner >= base + COMPUTE_QUEUE_COUNT as u32 || dwords == 0 {
            return false;
        }
        let queue = (owner - base) as usize + 1;
        let mut st = self.lock();
        Self::enqueue(
            &mut st,
            Submission { kind: Kind::Commands, queue, addr, dwords, exec: Execution::default(), started: false, blocked: false, blocked_since: None },
        );
        self.cv.notify_all();
        true
    }

    /// Frame boundary. Queues the marker and, unless called on the GPU
    /// thread itself, blocks while too many boundaries are outstanding.
    pub fn suspend_point(&self) {
        let mut st = self.lock();
        st.pending_boundaries += 1;
        Self::enqueue(
            &mut st,
            Submission {
                kind: Kind::FrameBoundary,
                queue: 0,
                addr: 0,
                dwords: 0,
                exec: Execution::default(),
                started: false,
                blocked: false,
                blocked_since: None,
            },
        );
        self.cv.notify_all();
        if !self.has_worker() {
            return;
        }
        let mut warned = Instant::now();
        while st.pending_boundaries > self.boundaries_in_flight {
            let (g, _) = crate::signals::wait_cv(&self.state, &self.cv, st, None);
            st = g;
            if warned.elapsed() > Duration::from_secs(5) {
                warned = Instant::now();
                let blocked: Vec<String> = st
                    .queues
                    .iter()
                    .filter_map(|q| q.front())
                    .filter(|s| s.blocked)
                    .map(|s| match s.exec.waiting_on {
                        Some((addr, value, reference, mask, f)) => format!(
                            "q{}@0x{:x} wait [0x{addr:x}]=0x{value:x} f{f} ref 0x{reference:x} mask 0x{mask:x}",
                            s.queue,
                            s.exec.current_packet()
                        ),
                        None => format!("q{}@0x{:x}", s.queue, s.exec.current_packet()),
                    })
                    .collect();
                let queued: Vec<String> = st.queues.iter().enumerate().filter(|(_, q)| !q.is_empty()).map(|(i, q)| format!("q{i}:{}", q.len())).collect();
                log::warn!(
                    "[agc] SuspendPoint waiting on the GPU: {} boundaries pending, processing={}, queued {:?}, blocked: {:?}",
                    st.pending_boundaries,
                    st.processing,
                    queued,
                    blocked
                );
            }
        }
    }

    /// Wait until nothing is queued or running (bounded by `timeout`).
    /// Without a GPU thread the caller drains the queues itself.
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        if !self.has_worker() {
            loop {
                self.retry_blocked();
                while !matches!(self.process_one(), SliceResult::NoWork | SliceResult::AllBlocked) {}
                if self.pending() == 0 {
                    return true;
                }
                if Instant::now() >= deadline {
                    return false;
                }
                crate::signals::sleep(RETRY_INTERVAL);
            }
        }
        let mut st = self.lock();
        while st.count != 0 || st.processing {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let (g, _) = crate::signals::wait_cv(&self.state, &self.cv, st, Some(deadline));
            st = g;
        }
        true
    }

    /// Make every blocked queue head runnable again (re-poll its wait).
    pub fn retry_blocked(&self) {
        Self::unblock_all(&mut self.lock());
        self.cv.notify_all();
    }

    /// Frames (boundaries) retired by the GPU thread.
    pub fn frames(&self) -> u64 {
        self.lock().frames
    }

    pub fn pending(&self) -> usize {
        self.lock().count
    }

    fn has_worker(&self) -> bool {
        self.worker.get().is_some()
    }

    /// Start the GPU command thread (idempotent).
    pub fn ensure_worker(&'static self) {
        self.worker.get_or_init(|| {
            std::thread::Builder::new()
                .name("gpu-cmd".into())
                .spawn(move || self.run())
                .expect("spawn GPU command thread");
            ps5vm::shutdown::on_shutdown(ps5vm::shutdown::ORDER_GPU, "GPU command thread", move || self.stop());
        });
    }

    /// Process shutdown: wait (briefly) for the GPU thread to leave its
    /// loop between two slices, so no command is mid-flight on exit.
    fn stop(&self) {
        let deadline = Instant::now() + Duration::from_millis(500);
        let mut st = self.lock();
        self.cv.notify_all();
        while !st.stopped {
            let now = Instant::now();
            if now >= deadline {
                log::warn!("[pm4] GPU command thread did not stop in time");
                return;
            }
            st = self.cv.wait_timeout(st, deadline - now).unwrap_or_else(|e| e.into_inner()).0;
        }
    }

    fn run(&self) {
        // The recorder's working set wants the large L3 (asymmetric CPUs).
        ps5vm::topology::pin_current_thread_to_fast_cores("gpu-cmd");
        let mut last_report = Instant::now();
        loop {
            if ps5vm::shutdown::requested() {
                self.lock().stopped = true;
                self.cv.notify_all();
                return;
            }
            match self.process_one() {
                SliceResult::NoWork => {
                    let t = Instant::now();
                    let st = self.lock();
                    if st.count == 0 && !ps5vm::shutdown::requested() {
                        let _ = self.cv.wait_timeout(st, Duration::from_millis(500));
                    }
                    IDLE_NO_WORK_US.fetch_add(t.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
                }
                SliceResult::AllBlocked => {
                    let t = Instant::now();
                    let st = self.lock();
                    // A label written since the heads blocked (end-of-pipe
                    // work unblocks them, see `retry_blocked`) or a new
                    // submission: run again without waiting.
                    let mut st = if wake_enabled() && Self::select(&st).is_some() {
                        st
                    } else {
                        self.cv.wait_timeout(st, RETRY_INTERVAL).unwrap_or_else(|e| e.into_inner()).0
                    };
                    Self::unblock_all(&mut st);
                    IDLE_BLOCKED_US.fetch_add(t.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
                }
                _ => {}
            }
            if last_report.elapsed() > Duration::from_secs(10) {
                last_report = Instant::now();
                use super::interp::*;
                use std::sync::atomic::Ordering::Relaxed;
                if profiling() {
                    log::info!(
                        "[pm4-profile] cp_ms={} exec_ms={} packets={} draws={} frames={}",
                        CP_NANOS.load(Relaxed) / 1_000_000,
                        EXEC_NANOS.load(Relaxed) / 1_000_000,
                        PACKETS.load(Relaxed),
                        DRAWS.load(Relaxed),
                        self.frames()
                    );
                }
                log::info!(
                    "[pm4] packets={} draws={} dispatches={} flips={} labels={} interrupts={} frames={} idle no-work {} ms, blocked {} ms",
                    PACKETS.load(Relaxed),
                    DRAWS.load(Relaxed),
                    DISPATCHES.load(Relaxed),
                    FLIPS.load(Relaxed),
                    LABEL_WRITES.load(Relaxed),
                    INTERRUPTS.load(Relaxed),
                    self.frames(),
                    IDLE_NO_WORK_US.load(Relaxed) / 1000,
                    IDLE_BLOCKED_US.load(Relaxed) / 1000
                );
                if let Some(s) = ps5gpu::backend::summary() {
                    log::info!("[gpu] {s}");
                }
            }
        }
    }

    fn unblock_all(st: &mut State) {
        for q in st.queues.iter_mut() {
            if let Some(h) = q.front_mut() {
                h.blocked = false;
            }
        }
    }

    fn select(st: &State) -> Option<usize> {
        (0..QUEUE_COUNT)
            .map(|o| (st.next_queue + o) % QUEUE_COUNT)
            .find(|&q| st.queues[q].front().is_some_and(|h| !h.blocked))
    }

    /// Run one slice of the next runnable submission on the calling thread.
    pub fn process_one(&self) -> SliceResult {
        let mut sub = {
            let mut st = self.lock();
            if st.processing {
                return SliceResult::AllBlocked;
            }
            let Some(q) = Self::select(&st) else {
                return if st.count != 0 { SliceResult::AllBlocked } else { SliceResult::NoWork };
            };
            let s = st.queues[q].pop_front().unwrap();
            st.count -= 1;
            st.next_queue = (q + 1) % QUEUE_COUNT;
            st.processing = true;
            s
        };

        let released_before = release_count();
        let complete = match sub.kind {
            Kind::FrameBoundary => {
                // The graphics processor starts each frame from reset state.
                self.with_interpreter(0, |i| i.reset());
                true
            }
            Kind::Commands => self.run_slice(&mut sub),
        };
        if complete {
            // A finished submission (or a frame boundary) is one collection
            // tick of the GPU caches.
            let q = if sub.kind == Kind::FrameBoundary { 0 } else { sub.queue };
            let ex = self.with_interpreter(q, |i| i.executor.clone());
            if let Some(ex) = ex {
                ex.submission_complete();
            }
        }

        let mut st = self.lock();
        let result = if !complete {
            let progressed = sub.exec.made_progress;
            sub.blocked = true;
            if progressed || sub.blocked_since.is_none() {
                sub.blocked_since = Some(Instant::now());
            }
            let q = sub.queue;
            st.queues[q].push_front(sub);
            st.count += 1;
            if wake_enabled() && release_count() != released_before {
                // The slice wrote labels or flipped: queues waiting on
                // memory may run now.
                Self::unblock_all(&mut st);
            }
            if progressed {
                SliceResult::Progressed
            } else {
                SliceResult::BlockedWithoutProgress
            }
        } else {
            if sub.kind == Kind::FrameBoundary {
                st.pending_boundaries = st.pending_boundaries.saturating_sub(1);
                st.frames += 1;
            }
            Self::unblock_all(&mut st);
            SliceResult::Completed
        };
        st.processing = false;
        self.cv.notify_all();
        result
    }

    fn run_slice(&self, sub: &mut Submission) -> bool {
        // A new submission, or one resumed after a wait: what the CPU wrote
        // before it is visible to its commands.
        let mut interps = self.interpreters.lock().unwrap_or_else(|e| e.into_inner());
        let interp = interps[sub.queue].get_or_insert_with(|| Interpreter::new(sub.queue));
        // A new submission, or one resumed after anything but a label the
        // GPU wrote, may read what the CPU wrote meanwhile.
        let released = !sub.started || sub.exec.waiting_on.is_none_or(|w| !super::interp::is_gpu_label(w.0));
        if released {
            match &interp.executor {
                Some(ex) => ex.sync_point(),
                None => ps5gpu::memory::note_sync_point(),
            }
        }
        if !sub.started {
            sub.started = true;
            interp.begin_submission();
        }
        let t = super::interp::profiling().then(Instant::now);
        let r = match interp.process(&mut sub.exec, sub.addr, sub.dwords) {
            Ok(Progress::Complete) => true,
            Ok(Progress::Blocked) => false,
            Err(e) => {
                log::error!("[pm4] queue {} submission 0x{:x} ({} dwords) dropped: {e}", sub.queue, sub.addr, sub.dwords);
                true
            }
        };
        if let Some(t) = t {
            super::interp::CP_NANOS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        // Submit what the slice recorded (end-of-pipe work waits on it).
        if let Some(ex) = &interp.executor {
            ex.flush();
        }
        r
    }

    /// Run `f` with the interpreter of `queue` (tests / diagnostics).
    pub fn with_interpreter<R>(&self, queue: usize, f: impl FnOnce(&mut Interpreter) -> R) -> R {
        let mut interps = self.interpreters.lock().unwrap_or_else(|e| e.into_inner());
        f(interps[queue].get_or_insert_with(|| Interpreter::new(queue)))
    }
}
