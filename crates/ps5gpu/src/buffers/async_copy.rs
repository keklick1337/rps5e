// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Upload copies (guest memory → host-visible staging) run by helper
//! threads instead of the GPU command thread.
//!
//! The command thread copies the CPU-dirty pages a draw reads into the
//! staging ring before recording the GPU copy into the mirror. The guest
//! wrote those lines moments before, often on another core complex, so the
//! copy runs at ~2 GB/s and was up to ~10 % of the thread in GTA V's city.
//! Here the command thread only queues the copy; helper threads (and the
//! command thread itself when it has to wait) run it.
//!
//! Every queued copy is finished ([`drain`]) before:
//! * the command buffer reading the staging bytes is submitted
//!   (`Scheduler::submit_with`);
//! * the command processor makes a write the guest can observe (labels,
//!   WRITE_DATA: `publish_point`, `write_ordered`): after such a write the
//!   guest may reuse the memory a pending copy reads.
//!
//! A copy that runs later than it was queued may see bytes the guest wrote
//! meanwhile. The page was write-watched before it was queued (the tracker
//! protects before the upload callback), so such a write also makes the
//! page CPU-dirty again and the next use uploads it once more; the GPU
//! reads guest memory when the work executes, which is later still.
//!
//! Only copies from the backing alias (one contiguous piece, never
//! unmapped) into coherent staging memory are queued; frame captures and
//! slice threads copy at once. `RPS5E_UPLOAD_THREADS=<n>` (default 3, 0 =
//! copy on the command thread).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

/// Copies are split into pieces of at most this many bytes (helpers and a
/// draining thread share large uploads).
const PIECE: usize = 64 << 10;
/// Smaller copies run at once (queueing costs about as much).
pub const MIN_BYTES: u64 = 8 << 10;

struct Job {
    src: usize,
    dst: usize,
    len: usize,
}

struct Pool {
    jobs: Mutex<VecDeque<Job>>,
    wake: Condvar,
}

/// Copies queued and not finished.
static PENDING: AtomicUsize = AtomicUsize::new(0);
/// Statistics: copies and bytes queued, drains that had to wait.
pub static QUEUED: AtomicU64 = AtomicU64::new(0);
pub static QUEUED_BYTES: AtomicU64 = AtomicU64::new(0);
pub static WAITED: AtomicU64 = AtomicU64::new(0);

fn threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::env::var("RPS5E_UPLOAD_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(3).min(16))
}

/// Upload copies may be queued now (`RPS5E_UPLOAD_THREADS` > 0; A/B
/// scope release `upload-threads`).
pub fn enabled() -> bool {
    threads() > 0 && !crate::backend::ab::safe() && crate::backend::ab::release_fast("upload-threads")
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<&'static Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let pool: &'static Pool = Box::leak(Box::new(Pool { jobs: Mutex::new(VecDeque::new()), wake: Condvar::new() }));
        for i in 0..threads() {
            let _ = std::thread::Builder::new().name(format!("gpu-upload-{i}")).spawn(move || {
                // Started by the command thread: not on its CPU.
                ps5vm::topology::unpin_current_thread_from_recorder_core("gpu-upload");
                worker(pool)
            });
        }
        pool
    })
}

fn run(job: Job) {
    // SAFETY: `src` points into the backing alias (always mapped) and
    // `dst` into a staging ring region reserved for the open tick, which
    // is not submitted before `drain` saw this copy finish.
    unsafe { std::ptr::copy_nonoverlapping(job.src as *const u8, job.dst as *mut u8, job.len) };
    // A locked read-modify-write: the copy's (write-combined) stores are
    // visible before the count drops.
    PENDING.fetch_sub(1, Ordering::Release);
}

fn worker(pool: &'static Pool) {
    loop {
        let job = {
            let mut q = pool.jobs.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(j) = q.pop_front() {
                    break j;
                }
                q = pool.wake.wait(q).unwrap_or_else(|e| e.into_inner());
            }
        };
        run(job);
    }
}

/// Queue a copy of `len` bytes from `src` (the backing alias) to `dst`
/// (coherent staging memory).
///
/// # Safety
/// Both ranges stay valid until the next [`drain`] returns.
pub unsafe fn queue(src: *const u8, dst: *mut u8, len: usize) {
    let pool = pool();
    let pieces = len.div_ceil(PIECE);
    PENDING.fetch_add(pieces, Ordering::Relaxed);
    QUEUED.fetch_add(1, Ordering::Relaxed);
    QUEUED_BYTES.fetch_add(len as u64, Ordering::Relaxed);
    {
        let mut q = pool.jobs.lock().unwrap_or_else(|e| e.into_inner());
        let mut done = 0;
        while done < len {
            let n = (len - done).min(PIECE);
            q.push_back(Job { src: src as usize + done, dst: dst as usize + done, len: n });
            done += n;
        }
    }
    if pieces > 1 {
        pool.wake.notify_all();
    } else {
        pool.wake.notify_one();
    }
}

/// Finish every queued copy (helping with the ones not started yet).
pub fn drain() {
    if PENDING.load(Ordering::Acquire) == 0 {
        return;
    }
    WAITED.fetch_add(1, Ordering::Relaxed);
    let pool = pool();
    loop {
        let job = pool.jobs.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
        match job {
            Some(j) => run(j),
            None => {
                // The rest are running on helpers.
                let mut spins = 0u32;
                while PENDING.load(Ordering::Acquire) != 0 {
                    spins += 1;
                    if spins < 4096 {
                        std::hint::spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                }
                return;
            }
        }
    }
}

/// `queued n/MiB, drains waited n` for the statistics line.
pub fn summary() -> String {
    format!(
        "upload-threads {}: queued {}/{} MiB, drains waited {}",
        threads(),
        QUEUED.load(Ordering::Relaxed),
        QUEUED_BYTES.load(Ordering::Relaxed) >> 20,
        WAITED.load(Ordering::Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Queued copies of every size land complete once `drain` returns,
    /// whether helpers or the draining thread ran them.
    #[test]
    fn drained_copies_are_complete() {
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let src: Vec<u8> = (0..(1usize << 20)).map(|i| (i * 7 + 3) as u8).collect();
        let mut dst = vec![0u8; 1 << 20];
        for round in 0..200 {
            dst.fill(0);
            let mut spans = Vec::new();
            let mut at = 0usize;
            while at < dst.len() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let len = [16 << 10, 1 << 10, 200 << 10, 64 << 10, 3][(rng % 5) as usize].min(dst.len() - at);
                spans.push((at, len));
                at += len + (rng >> 40) as usize % 4096;
            }
            for &(at, len) in &spans {
                // SAFETY: both buffers outlive the drain below.
                unsafe { queue(src.as_ptr().add(at), dst.as_mut_ptr().add(at), len) };
            }
            drain();
            assert_eq!(PENDING.load(Ordering::Acquire), 0);
            for &(at, len) in &spans {
                assert!(dst[at..at + len] == src[at..at + len], "round {round}: span {at}+{len}");
            }
        }
    }
}
