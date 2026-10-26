// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Worker threads compiling shader modules and pipelines off the command
//! thread.
//!
//! The provider submits a job per missing module or pipeline; the
//! command thread waits for the jobs its draws need (with
//! `RPS5E_COMPILE_WAIT_MS` set, a draw writing only its attachments is
//! skipped once the frame's wait budget is spent), while the workers
//! compile the other permutations and the prewarm list. Finished jobs are collected
//! by the provider on its own thread ([`CompilePool::take`]), so every
//! cache stays single-threaded. Jobs of the prewarm list run after the
//! jobs draws wait for; a draw that needs one of them moves that job alone
//! in front ([`CompilePool::hurry`]), so work of the running game never
//! queues behind the list. A job the command thread blocks on (a dispatch,
//! a draw that writes memory) that no worker has started yet is taken out
//! of the queue and run by the waiting thread itself
//! ([`CompilePool::take_queued`]), or, when that thread waits with the
//! back-end unlocked, moved to the front of the queue
//! ([`CompilePool::hasten`]): it never waits behind the compiles of
//! skipped draws or of the lookahead. Background jobs (optimised rebuilds of pipelines
//! already usable) run last, never on every worker at once, so a job a
//! draw waits for always finds a free worker soon; they are dropped
//! unrun at shutdown.
//!
//! Work a draw or dispatch asked for (urgent jobs) may take every worker;
//! the prewarm list keeps to the pool's share and background jobs to less,
//! so while nothing waits the game's own threads keep their cores
//!. Urgent and prewarm jobs queued or running are counted
//! in [`foreground_pending`] (the "compiling N shaders" indicator).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Urgent and prewarm jobs queued or running in every pool of the process.
static FOREGROUND: AtomicUsize = AtomicUsize::new(0);

/// Urgent and prewarm jobs finished in every pool of the process.
static FOREGROUND_DONE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Urgent and prewarm compile jobs queued or running now (all pools).
pub fn foreground_pending() -> usize {
    FOREGROUND.load(Ordering::Relaxed)
}

/// Urgent and prewarm compile jobs finished so far (all pools).
pub fn foreground_done() -> u64 {
    FOREGROUND_DONE.load(Ordering::Acquire)
}

/// Signalled whenever an urgent or prewarm job of any pool finishes
/// ([`wait_foreground_progress`]).
static PROGRESS: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Block until an urgent or prewarm job of any pool finished after
/// [`foreground_done`] returned `seen`, or `timeout` passes. Holds no
/// lock of a pool or of the back-end: the command thread waits here for
/// a compile with the back-end free for other threads.
pub fn wait_foreground_progress(seen: u64, timeout: Duration) {
    let g = PROGRESS.0.lock().unwrap_or_else(|e| e.into_inner());
    if FOREGROUND_DONE.load(Ordering::Acquire) != seen {
        return;
    }
    let _ = PROGRESS.1.wait_timeout(g, timeout);
}

fn note_foreground_done() {
    FOREGROUND_DONE.fetch_add(1, Ordering::AcqRel);
    // Taken after the count moved: a waiter that read the old count is
    // inside `wait_timeout` (or has not locked yet and sees the new one).
    let _g = PROGRESS.0.lock().unwrap_or_else(|e| e.into_inner());
    PROGRESS.1.notify_all();
}
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

type Job<T> = Box<dyn FnOnce() -> T + Send>;

/// The tag of jobs nothing hurries or takes back.
const UNTAGGED: u64 = u64::MAX;

struct State<T> {
    /// Jobs of work being recorded with their tags, then prewarm jobs.
    urgent: VecDeque<(u64, Job<T>)>,
    /// Prewarm jobs with their tags ([`CompilePool::hurry`]).
    prewarm: VecDeque<(u64, Job<T>)>,
    background: VecDeque<Job<T>>,
    /// Background jobs running now, and how many may run at once.
    background_running: usize,
    background_limit: usize,
    /// Prewarm jobs running now, and how many may run at once (the share).
    prewarm_running: usize,
    prewarm_limit: usize,
    done: Vec<T>,
    stop: bool,
    /// Workers not yet exited (each counts down on exit).
    running: usize,
}

struct Shared<T> {
    state: Mutex<State<T>>,
    /// Signalled when a job is queued (workers) or done (waiters).
    work: Condvar,
    finished: Condvar,
    has_done: AtomicBool,
    /// Jobs queued or running.
    in_flight: AtomicUsize,
}

/// What a worker runs: urgent jobs may take every worker, prewarm and
/// background jobs keep to their limits.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Urgent,
    Prewarm,
    Background,
}

pub struct CompilePool<T: Send + 'static> {
    shared: Arc<Shared<T>>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl<T: Send + 'static> CompilePool<T> {
    /// A pool of `threads` workers (named `rps5e-compile-N`); background
    /// jobs leave at least one worker (two from four workers up) free.
    pub fn new(threads: usize) -> CompilePool<T> {
        Self::with_share(threads, threads)
    }

    /// A pool of `threads` workers of which prewarm jobs use at most
    /// `share` (background jobs fewer still); urgent jobs use them all.
    pub fn with_share(threads: usize, share: usize) -> CompilePool<T> {
        let threads = threads.max(1);
        let share = share.clamp(1, threads);
        let background_limit = if share >= 4 { share - 2 } else { share.saturating_sub(1).max(1) };
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                urgent: VecDeque::new(),
                prewarm: VecDeque::new(),
                background: VecDeque::new(),
                background_running: 0,
                background_limit,
                prewarm_running: 0,
                prewarm_limit: share,
                done: Vec::new(),
                stop: false,
                running: 0,
            }),
            work: Condvar::new(),
            finished: Condvar::new(),
            has_done: AtomicBool::new(false),
            in_flight: AtomicUsize::new(0),
        });
        let mut spawned = Vec::new();
        for i in 0..threads {
            let s = shared.clone();
            match std::thread::Builder::new().name(format!("rps5e-compile-{i}")).spawn(move || Self::run(s)) {
                Ok(h) => spawned.push(h),
                Err(_) => log::warn!("ps5gpu: compile worker {i} not started"),
            }
        }
        shared.state.lock().unwrap_or_else(|e| e.into_inner()).running = spawned.len();
        CompilePool { shared, threads: spawned }
    }

    fn run(s: Arc<Shared<T>>) {
        loop {
            let (job, kind) = {
                let mut st = s.state.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    if st.stop {
                        st.running -= 1;
                        s.finished.notify_all();
                        return;
                    }
                    if let Some((_, j)) = st.urgent.pop_front() {
                        break (j, Kind::Urgent);
                    }
                    if st.prewarm_running < st.prewarm_limit {
                        if let Some((_, j)) = st.prewarm.pop_front() {
                            st.prewarm_running += 1;
                            break (j, Kind::Prewarm);
                        }
                    }
                    if st.background_running < st.background_limit {
                        if let Some(j) = st.background.pop_front() {
                            st.background_running += 1;
                            break (j, Kind::Background);
                        }
                    }
                    st = s.work.wait(st).unwrap_or_else(|e| e.into_inner());
                }
            };
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
            let mut st = s.state.lock().unwrap_or_else(|e| e.into_inner());
            match kind {
                Kind::Urgent => {}
                Kind::Prewarm => st.prewarm_running -= 1,
                Kind::Background => st.background_running -= 1,
            }
            if kind != Kind::Background {
                FOREGROUND.fetch_sub(1, Ordering::Relaxed);
            }
            if kind != Kind::Urgent {
                // Another limited job may start now.
                s.work.notify_one();
            }
            match r {
                Ok(v) => st.done.push(v),
                Err(_) => log::error!("ps5gpu: a pipeline compile job panicked"),
            }
            s.in_flight.fetch_sub(1, Ordering::AcqRel);
            s.has_done.store(true, Ordering::Release);
            s.finished.notify_all();
            drop(st);
            if kind != Kind::Background {
                // After the result is in `done`: a waiter woken by the count
                // finds it there.
                note_foreground_done();
            }
        }
    }

    /// Queue `job` (behind the urgent jobs when `prewarm`, untagged).
    /// Dropped without running once the pool is shutting down.
    pub fn submit(&self, job: impl FnOnce() -> T + Send + 'static, prewarm: bool) {
        self.enqueue(Box::new(job), prewarm, UNTAGGED);
    }

    /// Queue urgent `job` under `tag` ([`CompilePool::take_queued`]).
    pub fn submit_urgent(&self, job: impl FnOnce() -> T + Send + 'static, tag: u64) {
        self.enqueue(Box::new(job), false, tag);
    }

    /// The urgent or prewarm job tagged `tag` if no worker has started it,
    /// taken out of the pool (no longer in flight) for the caller to run:
    /// a thread about to block on that job runs it at once instead of
    /// waiting for a worker to reach it. None: running, finished or never
    /// queued.
    pub fn take_queued(&self, tag: u64) -> Option<Box<dyn FnOnce() -> T + Send>> {
        if tag == UNTAGGED {
            return None;
        }
        let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        let job = match st.urgent.iter().position(|(t, _)| *t == tag) {
            Some(i) => st.urgent.remove(i).map(|(_, j)| j),
            None => {
                let i = st.prewarm.iter().position(|(t, _)| *t == tag)?;
                st.prewarm.remove(i).map(|(_, j)| j)
            }
        }?;
        drop(st);
        self.shared.in_flight.fetch_sub(1, Ordering::AcqRel);
        FOREGROUND.fetch_sub(1, Ordering::Relaxed);
        note_foreground_done();
        Some(job)
    }

    /// Move the urgent or prewarm job tagged `tag`, if no worker has
    /// started it, to the front of the queue: the next free worker runs it
    /// (for a caller that waits for it without running it itself). False:
    /// running, finished or never queued.
    pub fn hasten(&self, tag: u64) -> bool {
        if tag == UNTAGGED {
            return false;
        }
        let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        let job = match st.urgent.iter().position(|(t, _)| *t == tag) {
            Some(i) => st.urgent.remove(i),
            None => match st.prewarm.iter().position(|(t, _)| *t == tag) {
                Some(i) => st.prewarm.remove(i),
                None => None,
            },
        };
        let Some(job) = job else { return false };
        st.urgent.push_front(job);
        self.shared.work.notify_one();
        true
    }

    /// Queue prewarm `job` under `tag` (behind every urgent job; see
    /// [`CompilePool::hurry`]).
    pub fn submit_prewarm(&self, job: impl FnOnce() -> T + Send + 'static, tag: u64) {
        self.enqueue(Box::new(job), true, tag);
    }

    fn enqueue(&self, job: Job<T>, prewarm: bool, tag: u64) {
        self.shared.in_flight.fetch_add(1, Ordering::AcqRel);
        let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        if st.stop {
            self.shared.in_flight.fetch_sub(1, Ordering::AcqRel);
            return;
        }
        match prewarm {
            true => st.prewarm.push_back((tag, job)),
            false => st.urgent.push_back((tag, job)),
        }
        FOREGROUND.fetch_add(1, Ordering::Relaxed);
        self.shared.work.notify_one();
    }

    /// Queue `job` behind every other job (see the module notes).
    pub fn submit_background(&self, job: impl FnOnce() -> T + Send + 'static) {
        self.shared.in_flight.fetch_add(1, Ordering::AcqRel);
        let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        if st.stop {
            self.shared.in_flight.fetch_sub(1, Ordering::AcqRel);
            return;
        }
        st.background.push_back(Box::new(job));
        self.shared.work.notify_one();
    }

    /// Background jobs queued or running.
    pub fn background_pending(&self) -> usize {
        let st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        st.background.len() + st.background_running
    }

    /// Finished jobs since the last call (cheap when there are none).
    pub fn take(&self) -> Vec<T> {
        if !self.shared.has_done.load(Ordering::Acquire) {
            return Vec::new();
        }
        let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        self.shared.has_done.store(false, Ordering::Release);
        std::mem::take(&mut st.done)
    }

    /// Block until a job finishes (or `timeout` passes) when none is
    /// waiting to be taken.
    pub fn wait(&self, timeout: Duration) {
        let st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        if st.done.is_empty() && self.shared.in_flight.load(Ordering::Acquire) != 0 {
            let _ = self.shared.finished.wait_timeout(st, timeout);
        }
    }

    /// Jobs queued or running.
    pub fn in_flight(&self) -> usize {
        self.shared.in_flight.load(Ordering::Acquire)
    }

    /// Move the queued prewarm job tagged `tag` behind the urgent jobs (a
    /// draw waits for it); the other prewarm jobs stay behind any urgent
    /// work. False when no such job is queued (it runs or ran).
    pub fn hurry(&self, tag: u64) -> bool {
        let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        let Some(i) = st.prewarm.iter().position(|(t, _)| *t == tag) else { return false };
        let job = st.prewarm.remove(i).expect("indexed");
        st.urgent.push_back(job);
        true
    }

    /// Urgent and prewarm jobs queued or running in this pool.
    pub fn foreground_in_flight(&self) -> usize {
        let st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        self.in_flight().saturating_sub(st.background.len() + st.background_running)
    }

    /// Prewarm jobs queued (not yet started).
    pub fn prewarm_queued(&self) -> usize {
        self.shared.state.lock().unwrap_or_else(|e| e.into_inner()).prewarm.len()
    }

    /// Wait for the queued jobs, stop the workers and join them: no new
    /// job is accepted (dropped in [`CompilePool::submit`]); the whole
    /// call takes at most [`TEARDOWN_TIMEOUT`] — a stuck job or worker is
    /// logged and leaked (its handle dropped, detaching it) rather than
    /// blocking teardown forever. Returns the finished results.
    pub fn shutdown(&mut self) -> Vec<T> {
        const TEARDOWN_TIMEOUT: Duration = Duration::from_secs(10);
        let deadline = Instant::now() + TEARDOWN_TIMEOUT;
        let all_exited = {
            let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            // Background jobs not started are not needed any more.
            let dropped = std::mem::take(&mut st.background).len();
            self.shared.in_flight.fetch_sub(dropped, Ordering::AcqRel);
            while self.shared.in_flight.load(Ordering::Acquire) != 0 {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    log::warn!("ps5gpu: pipeline compile jobs still in flight after {TEARDOWN_TIMEOUT:?}; stopping the workers anyway");
                    break;
                }
                let (s, _) = self.shared.finished.wait_timeout(st, left).unwrap_or_else(|e| e.into_inner());
                st = s;
            }
            st.stop = true;
            self.shared.work.notify_all();
            while st.running != 0 {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    break;
                }
                let (s, _) = self.shared.finished.wait_timeout(st, left).unwrap_or_else(|e| e.into_inner());
                st = s;
            }
            if st.running != 0 {
                log::warn!("ps5gpu: {} pipeline compile worker(s) did not stop within {TEARDOWN_TIMEOUT:?}; continuing without them", st.running);
            }
            st.running == 0
        };
        if all_exited {
            for t in self.threads.drain(..) {
                let _ = t.join();
            }
        } else {
            // Joining a running worker would block: leak the handles.
            self.threads.clear();
        }
        let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut st.done)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jobs_run_and_results_are_taken() {
        let mut pool: CompilePool<u32> = CompilePool::new(3);
        for i in 0..20 {
            pool.submit(move || i * 2, i % 2 == 0);
        }
        let mut got = Vec::new();
        while got.len() < 20 {
            got.extend(pool.take());
            pool.wait(Duration::from_millis(5));
        }
        got.sort();
        assert_eq!(got, (0..20).map(|i| i * 2).collect::<Vec<_>>());
        assert_eq!(pool.in_flight(), 0);
        assert!(pool.shutdown().is_empty());
    }

    #[test]
    fn background_jobs_leave_workers_free_and_are_dropped_at_shutdown() {
        use std::sync::atomic::AtomicBool;
        let mut pool: CompilePool<u32> = CompilePool::new(4);
        // Two background jobs may run at once (two of four workers stay
        // free); hold them until released.
        let release = Arc::new(AtomicBool::new(false));
        let started = Arc::new(AtomicUsize::new(0));
        for i in 0..5 {
            let (release, started) = (release.clone(), started.clone());
            pool.submit_background(move || {
                started.fetch_add(1, Ordering::SeqCst);
                while !release.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                100 + i
            });
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while started.load(Ordering::SeqCst) < 2 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(started.load(Ordering::SeqCst), 2, "background jobs limited to two of four workers");
        assert_eq!(pool.background_pending(), 5);
        // Urgent work still runs on the free workers.
        pool.submit(|| 7, false);
        let mut got = Vec::new();
        while !got.contains(&7) && Instant::now() < deadline {
            got.extend(pool.take());
            pool.wait(Duration::from_millis(5));
        }
        assert!(got.contains(&7), "urgent job ran while background jobs held their workers");
        // Shutdown drops the three queued background jobs and waits for the
        // two running ones.
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            release.store(true, Ordering::SeqCst);
        });
        let mut done = pool.shutdown();
        releaser.join().unwrap();
        done.sort();
        assert_eq!(done, vec![100, 101]);
        assert_eq!(pool.in_flight(), 0);
    }

    /// A draw needing one queued prewarm job moves that job alone: urgent
    /// work submitted later still runs before the rest of the list.
    #[test]
    fn hurrying_one_prewarm_job_keeps_urgent_work_first() {
        use std::sync::atomic::AtomicBool;
        let mut pool: CompilePool<u32> = CompilePool::new(1);
        let release = Arc::new(AtomicBool::new(false));
        let r = release.clone();
        pool.submit(
            move || {
                while !r.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                0
            },
            false,
        );
        // Let the single worker take the blocking job.
        let deadline = Instant::now() + Duration::from_secs(5);
        while pool.shared.state.lock().unwrap().urgent.len() != 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        for tag in 1..=50u64 {
            pool.submit_prewarm(move || tag as u32, tag);
        }
        assert!(pool.hurry(30));
        assert!(!pool.hurry(99));
        pool.submit(|| 1000, false);
        assert_eq!(pool.prewarm_queued(), 49);
        release.store(true, Ordering::SeqCst);
        let mut order = Vec::new();
        while order.len() < 52 && Instant::now() < deadline {
            order.extend(pool.take());
            pool.wait(Duration::from_millis(5));
        }
        assert_eq!(&order[..3], &[0, 30, 1000], "the hurried job, then urgent work, before the rest of the list");
        assert_eq!(order.len(), 52);
        assert!(pool.shutdown().is_empty());
    }

    /// A job the caller blocks on is taken back while still queued (behind
    /// other urgent work or in the prewarm list) and run by the caller; a
    /// job a worker started, an untagged one or an unknown tag is not.
    #[test]
    fn a_queued_job_is_taken_back_by_the_thread_that_waits_for_it() {
        use std::sync::atomic::AtomicBool;
        let mut pool: CompilePool<u32> = CompilePool::new(1);
        let release = Arc::new(AtomicBool::new(false));
        let r = release.clone();
        pool.submit_urgent(
            move || {
                while !r.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                0
            },
            7,
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        while pool.shared.state.lock().unwrap().urgent.len() != 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        for tag in 1..=40u64 {
            pool.submit_urgent(move || tag as u32 + 100, tag + 100);
        }
        pool.submit_prewarm(|| 555, 555);
        pool.submit(|| 9, false);
        assert_eq!(pool.in_flight(), 43);
        assert!(pool.take_queued(7).is_none(), "running on the worker");
        assert!(pool.take_queued(999).is_none());
        assert!(pool.take_queued(UNTAGGED).is_none());
        let job = pool.take_queued(130).expect("queued urgent job");
        assert_eq!(job(), 130);
        assert!(pool.take_queued(130).is_none(), "taken once");
        assert_eq!(pool.take_queued(555).expect("queued prewarm job")(), 555);
        assert_eq!(pool.in_flight(), 41);
        release.store(true, Ordering::SeqCst);
        let mut got = Vec::new();
        while got.len() < 41 && Instant::now() < deadline {
            got.extend(pool.take());
            pool.wait(Duration::from_millis(5));
        }
        got.sort();
        let mut want: Vec<u32> = (101..=140).filter(|&v| v != 130).collect();
        want.extend([0, 9]);
        want.sort();
        assert_eq!(got, want);
        assert_eq!(pool.in_flight(), 0);
        assert!(pool.shutdown().is_empty());
    }

    /// A hastened job runs before everything queued ahead of it, from the
    /// urgent queue or the prewarm list.
    #[test]
    fn a_hastened_job_runs_next() {
        use std::sync::atomic::AtomicBool;
        let mut pool: CompilePool<u32> = CompilePool::new(1);
        let release = Arc::new(AtomicBool::new(false));
        let r = release.clone();
        pool.submit_urgent(
            move || {
                while !r.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                0
            },
            7,
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        while pool.shared.state.lock().unwrap().urgent.len() != 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        for tag in 101..=120u64 {
            pool.submit_urgent(move || tag as u32, tag);
        }
        pool.submit_prewarm(|| 555, 555);
        assert!(!pool.hasten(7), "running on the worker");
        assert!(!pool.hasten(999));
        assert!(pool.hasten(115));
        assert!(pool.hasten(555));
        release.store(true, Ordering::SeqCst);
        let mut order = Vec::new();
        while order.len() < 22 && Instant::now() < deadline {
            order.extend(pool.take());
            pool.wait(Duration::from_millis(5));
        }
        assert_eq!(&order[..3], &[0, 555, 115], "the hastened jobs, the last one first");
        assert_eq!(order.len(), 22);
        assert!(pool.shutdown().is_empty());
    }

    /// Urgent jobs take every worker; prewarm jobs keep to the share, so
    /// while nothing waits the game keeps the other cores. Both count as
    /// pending compiles until they finish.
    #[test]
    fn prewarm_jobs_keep_to_the_share_and_urgent_jobs_use_every_worker() {
        use std::sync::atomic::AtomicBool;
        let mut pool: CompilePool<u32> = CompilePool::with_share(4, 1);
        let release = Arc::new(AtomicBool::new(false));
        let (running, peak) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let job = |v: u32| {
            let (release, running, peak) = (release.clone(), running.clone(), peak.clone());
            move || {
                let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                while !release.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                running.fetch_sub(1, Ordering::SeqCst);
                v
            }
        };
        for tag in 0..4u64 {
            pool.submit_prewarm(job(tag as u32), tag);
        }
        assert!(pool.foreground_in_flight() == 4 && foreground_pending() >= 4);
        let deadline = Instant::now() + Duration::from_secs(5);
        while running.load(Ordering::SeqCst) < 1 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(running.load(Ordering::SeqCst), 1, "one prewarm job at a time (the share)");
        // Three urgent jobs start on the three idle workers at once.
        for v in 100..103 {
            pool.submit(job(v), false);
        }
        while running.load(Ordering::SeqCst) < 4 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(running.load(Ordering::SeqCst), 4, "urgent jobs use every worker");
        release.store(true, Ordering::SeqCst);
        let mut got = Vec::new();
        while got.len() < 7 && Instant::now() < deadline {
            got.extend(pool.take());
            pool.wait(Duration::from_millis(5));
        }
        assert_eq!(got.len(), 7);
        assert_eq!(pool.foreground_in_flight(), 0);
        assert!(pool.shutdown().is_empty());
    }

    #[test]
    fn shutdown_joins_workers_and_later_submits_are_dropped() {
        let mut pool: CompilePool<u32> = CompilePool::new(2);
        for i in 0..8 {
            pool.submit(move || i, i >= 4);
        }
        let mut done = pool.shutdown();
        done.sort();
        assert_eq!(done, (0..8).collect::<Vec<_>>());
        assert!(pool.threads.is_empty(), "workers joined");
        // Submits after the shutdown are dropped, not queued forever.
        pool.submit(|| 99, false);
        assert_eq!(pool.in_flight(), 0);
        assert!(pool.take().is_empty());
        assert!(pool.shutdown().is_empty());
    }
}
