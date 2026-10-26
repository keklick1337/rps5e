// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Spin lock usable from the fault handler: it never allocates and never
//! blocks in the kernel. It remembers its owner so a fault taken while the
//! same thread holds the lock is declined instead of deadlocking.

use std::cell::Cell;
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};

thread_local! {
    /// The calling thread's kernel id, once asked (0 = not yet).
    static THREAD_ID: Cell<i32> = const { Cell::new(0) };
}

/// Kernel thread id of the caller (async-signal-safe). Cached per thread:
/// every lock and fault asks for it, and the id never changes.
#[inline]
pub fn thread_id() -> i32 {
    THREAD_ID.with(|id| {
        if id.get() == 0 {
            id.set(ps5vm::os::current_tid() as i32);
        }
        id.get()
    })
}

/// Busy retries a waiter makes before it starts yielding the processor.
const SPINS_BEFORE_YIELD: u32 = 64;

#[derive(Default)]
pub struct RegionLock {
    held: AtomicU32,
    owner: AtomicI32,
}

/// Guard returned by [`RegionLock::lock`].
pub struct RegionGuard<'a> {
    lock: &'a RegionLock,
}

impl RegionLock {
    pub const fn new() -> Self {
        Self { held: AtomicU32::new(0), owner: AtomicI32::new(0) }
    }

    /// True when the calling thread holds the lock.
    pub fn held_by_me(&self) -> bool {
        self.held.load(Ordering::Acquire) != 0 && self.owner.load(Ordering::Acquire) == thread_id()
    }

    fn acquire(&self, me: i32) {
        let mut spins = 0u32;
        while self.held.compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed).is_err() {
            if spins < SPINS_BEFORE_YIELD {
                spins += 1;
                std::hint::spin_loop();
            } else {
                // The owner may be preempted while waiters spin on its
                // CPU: give the processor back so it can run and release.
                // sched_yield: async-signal-safe (the fault path acquires
                // too).
                std::thread::yield_now();
            }
        }
        self.owner.store(me, Ordering::Release);
    }

    /// Take the lock. Re-entry from the owning thread is a bug.
    pub fn lock(&self) -> RegionGuard<'_> {
        let me = thread_id();
        assert!(!(self.held.load(Ordering::Acquire) != 0 && self.owner.load(Ordering::Acquire) == me), "region lock re-entered");
        self.acquire(me);
        RegionGuard { lock: self }
    }

    /// Take the lock from the fault path: None when the calling thread
    /// already holds it (the fault cannot be resolved).
    pub fn lock_for_fault(&self) -> Option<RegionGuard<'_>> {
        let me = thread_id();
        if self.held.load(Ordering::Acquire) != 0 && self.owner.load(Ordering::Acquire) == me {
            return None;
        }
        self.acquire(me);
        Some(RegionGuard { lock: self })
    }

    /// Manual acquire (paired with [`RegionLock::unlock`]).
    pub fn enter(&self) {
        std::mem::forget(self.lock());
    }

    pub fn unlock(&self) {
        self.owner.store(0, Ordering::Release);
        self.held.store(0, Ordering::Release);
    }
}

impl Drop for RegionGuard<'_> {
    fn drop(&mut self) {
        self.lock.unlock();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    /// Upper bound for any wait in these tests: a lock that is never handed
    /// over fails the test instead of hanging the run.
    const TIMEOUT: Duration = Duration::from_secs(20);

    /// A holder keeps the lock while another thread spins on it past the
    /// yield threshold; the waiter gets it once the holder releases it.
    #[test]
    fn waiter_gets_lock_after_long_holder_releases() {
        for fault_path in [false, true] {
            let lock = Arc::new(RegionLock::new());
            let (held_tx, held_rx) = mpsc::channel();
            let holder = {
                let lock = lock.clone();
                std::thread::spawn(move || {
                    let guard = lock.lock();
                    held_tx.send(()).unwrap();
                    std::thread::sleep(Duration::from_millis(200));
                    drop(guard);
                })
            };
            held_rx.recv_timeout(TIMEOUT).expect("holder did not take the lock");
            let (got_tx, got_rx) = mpsc::channel();
            let waiter = {
                let lock = lock.clone();
                std::thread::spawn(move || {
                    let guard = if fault_path { lock.lock_for_fault().expect("not the owner") } else { lock.lock() };
                    assert!(lock.held_by_me());
                    got_tx.send(()).unwrap();
                    drop(guard);
                })
            };
            got_rx.recv_timeout(TIMEOUT).expect("waiter never got the lock after the release");
            holder.join().unwrap();
            waiter.join().unwrap();
            assert!(!lock.held_by_me());
        }
    }

    /// Many threads contend for one lock; a counter inside the critical
    /// section checks mutual exclusion, and every thread finishes. The
    /// thread count is twice the host's CPUs, so waiters and the owner
    /// compete for processors.
    #[test]
    fn oversubscribed_contention_keeps_mutual_exclusion() {
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get() * 2).max(8);
        const ROUNDS: usize = 500;
        let lock = Arc::new(RegionLock::new());
        let inside = Arc::new(AtomicUsize::new(0));
        let total = Arc::new(AtomicUsize::new(0));
        let (done_tx, done_rx) = mpsc::channel();
        for _ in 0..threads {
            let (lock, inside, total, done) = (lock.clone(), inside.clone(), total.clone(), done_tx.clone());
            std::thread::spawn(move || {
                for _ in 0..ROUNDS {
                    let _guard = lock.lock();
                    assert_eq!(inside.fetch_add(1, Ordering::SeqCst), 0, "two owners");
                    total.fetch_add(1, Ordering::SeqCst);
                    inside.fetch_sub(1, Ordering::SeqCst);
                }
                done.send(()).unwrap();
            });
        }
        drop(done_tx);
        for _ in 0..threads {
            done_rx.recv_timeout(TIMEOUT).expect("a contender never finished");
        }
        assert_eq!(total.load(Ordering::SeqCst), threads * ROUNDS);
        assert!(!lock.held_by_me());
    }

    /// The owner's own fault is declined (no self-wait), the fault path
    /// takes the lock once it is released, and manual enter/unlock pairs.
    #[test]
    fn owner_fault_is_declined_and_manual_pairs_work() {
        let lock = RegionLock::new();
        let guard = lock.lock();
        assert!(lock.lock_for_fault().is_none(), "the owner must not wait for itself");
        drop(guard);
        assert!(lock.lock_for_fault().is_some());
        lock.enter();
        assert!(lock.held_by_me());
        lock.unlock();
        assert!(!lock.held_by_me());
    }
}
