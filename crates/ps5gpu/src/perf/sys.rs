// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Kernel-side cost of the GPU back-end (diagnostics): host protection
//! changes (`mprotect`, by the watch change behind them) and queue
//! submissions, counted and timed apart for
//! the command thread and every other thread, plus the command thread's own
//! resource usage (user/system time, faults, context switches). Printed as
//! the `[gpu-sys]` log line next to the `[gpu]` statistics.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// What is counted.
#[derive(Clone, Copy)]
pub enum Call {
    /// A host protection change (`mprotect` / `VirtualProtect`) adding a
    /// write watch (pages become read-only).
    WatchWrites = 0,
    /// ... removing a write watch.
    UnwatchWrites = 1,
    /// ... adding a read watch (pages become inaccessible).
    WatchReads = 2,
    /// ... removing a read watch.
    UnwatchReads = 3,
    /// Any other protection change (guest protection, registration).
    Protect = 4,
    /// `vkQueueSubmit2`.
    Submit = 5,
}
const CALLS: usize = 6;

/// [count, ns, bytes] per call kind, for the command thread (0) and
/// others (1).
static COUNTS: [[[AtomicU64; 3]; CALLS]; 2] = [const { [const { [const { AtomicU64::new(0) }; 3] }; CALLS] }; 2];

thread_local! {
    static RECORDER: Cell<bool> = const { Cell::new(false) };
}

/// Mark the calling thread as the command thread (the one whose time the
/// frame waits for).
pub fn mark_recorder_thread() {
    RECORDER.with(|r| r.set(true));
}

pub(crate) fn is_recorder() -> bool {
    RECORDER.try_with(|r| r.get()).unwrap_or(false)
}

/// Run `f` (a kernel call of kind `call`) and account its wall time.
/// Async-signal-safe: a thread-local flag, the vDSO clock and relaxed
/// atomics (the fault handler changes protections too).
#[inline]
pub fn timed<R>(call: Call, f: impl FnOnce() -> R) -> R {
    timed_bytes(call, 0, f)
}

/// [`timed`] for a call over `bytes` bytes.
#[inline]
pub fn timed_bytes<R>(call: Call, bytes: u64, f: impl FnOnce() -> R) -> R {
    let t = std::time::Instant::now();
    let r = f();
    let ns = t.elapsed().as_nanos() as u64;
    let who = if is_recorder() { 0 } else { 1 };
    let c = &COUNTS[who][call as usize];
    c[0].fetch_add(1, Relaxed);
    c[1].fetch_add(ns, Relaxed);
    c[2].fetch_add(bytes, Relaxed);
    r
}

/// Protection changes of the command thread so far: (calls, ns).
pub fn recorder_protects() -> (u64, u64) {
    COUNTS[0][..4].iter().fold((0, 0), |(n, t), c| (n + c[0].load(Relaxed), t + c[1].load(Relaxed)))
}

/// A/B accounting of the command thread: its CPU and wall time per A/B
/// mode (fast, slow), charged to the mode current at each call.
static AB_CPU_US: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static AB_WALL_US: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

thread_local! {
    static AB_LAST: Cell<Option<(u64, std::time::Instant)>> = const { Cell::new(None) };
}

/// The calling thread's CPU time (user + system, microseconds).
fn thread_cpu_us() -> Option<u64> {
    #[cfg(unix)]
    {
        let mut t: libc::timespec = unsafe { std::mem::zeroed() };
        // SAFETY: plain out-parameter.
        if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut t) } != 0 {
            return None;
        }
        Some(t.tv_sec as u64 * 1_000_000 + t.tv_nsec as u64 / 1000)
    }
    #[cfg(not(unix))]
    None
}

/// Charge the command thread's CPU and wall time since the previous call
/// to the current A/B mode (only with `RPS5E_AB_PERIOD`). Called by the
/// command thread between command-stream slices.
pub fn account_ab_mode() {
    if !crate::backend::ab::enabled() {
        return;
    }
    let (Some(cpu), now) = (thread_cpu_us(), std::time::Instant::now()) else { return };
    let mode = if crate::backend::ab::mode_fast() { 0 } else { 1 };
    if let Ok(Some((c0, w0))) = AB_LAST.try_with(|l| l.replace(Some((cpu, now)))) {
        AB_CPU_US[mode].fetch_add(cpu.saturating_sub(c0), Relaxed);
        AB_WALL_US[mode].fetch_add(now.duration_since(w0).as_micros() as u64, Relaxed);
    }
}

/// CPU time of the recording thread (`RPS5E_GPU_RECORD_THREAD=1`) per
/// A/B mode.
static AB_REC_CPU_US: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// [`account_ab_mode`] for the recording thread: its CPU since the
/// previous call goes to the current mode (called every few commands).
pub fn account_ab_mode_recorder() {
    if !crate::backend::ab::enabled() {
        return;
    }
    let Some(cpu) = thread_cpu_us() else { return };
    let mode = if crate::backend::ab::mode_fast() { 0 } else { 1 };
    if let Ok(Some((c0, _))) = AB_LAST.try_with(|l| l.replace(Some((cpu, std::time::Instant::now())))) {
        AB_REC_CPU_US[mode].fetch_add(cpu.saturating_sub(c0), Relaxed);
    }
    if let Some(sys) = thread_sys_us() {
        if let Ok(Some(s0)) = AB_LAST_SYS.try_with(|l| l.replace(Some(sys))) {
            AB_REC_SYS_US[mode].fetch_add(sys.saturating_sub(s0), Relaxed);
        }
    }
}

/// Kernel CPU time of the recording thread per A/B mode.
static AB_REC_SYS_US: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

thread_local! {
    static AB_LAST_SYS: Cell<Option<u64>> = const { Cell::new(None) };
}

/// The calling thread's kernel CPU time in microseconds (Linux).
fn thread_sys_us() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: plain out-parameter.
        let mut u: libc::rusage = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut u) } != 0 {
            return None;
        }
        Some(u.ru_stime.tv_sec as u64 * 1_000_000 + u.ru_stime.tv_usec as u64)
    }
    #[cfg(not(target_os = "linux"))]
    None
}

/// The command thread's own resource usage (Linux: `RUSAGE_THREAD`),
/// when called on it.
#[cfg(target_os = "linux")]
fn own_usage() -> Option<String> {
    if !is_recorder() {
        return None;
    }
    // SAFETY: plain out-parameter.
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut u) } != 0 {
        return None;
    }
    let ms = |t: libc::timeval| t.tv_sec as u64 * 1000 + t.tv_usec as u64 / 1000;
    Some(format!(
        " cmd-thread user {} ms sys {} ms minflt {} majflt {} vcsw {} ivcsw {}",
        ms(u.ru_utime),
        ms(u.ru_stime),
        u.ru_minflt,
        u.ru_majflt,
        u.ru_nvcsw,
        u.ru_nivcsw
    ))
}

#[cfg(not(target_os = "linux"))]
fn own_usage() -> Option<String> {
    None
}

/// The `[gpu-sys]` line: `protect cmd n/ms other n/ms submit cmd n/ms
/// other n/ms` (cumulative), plus the command thread's usage when the
/// caller is the command thread.
pub fn line() -> String {
    let g = |who: usize, c: Call| {
        let a = &COUNTS[who][c as usize];
        format!("{}/{}ms/{}KiB", a[0].load(Relaxed), a[1].load(Relaxed) / 1_000_000, a[2].load(Relaxed) >> 10)
    };
    let kinds = [("+w", Call::WatchWrites), ("-w", Call::UnwatchWrites), ("+r", Call::WatchReads), ("-r", Call::UnwatchReads), ("prot", Call::Protect)];
    let side = |who: usize| kinds.iter().map(|(n, c)| format!("{n} {}", g(who, *c))).collect::<Vec<_>>().join(" ");
    let ab = if crate::backend::ab::enabled() {
        format!(
            " ab_cmd_cpu_ms=[{}, {}] ab_cmd_wall_ms=[{}, {}] ab_rec_cpu_ms=[{}, {}] ab_rec_sys_ms=[{}, {}]",
            AB_CPU_US[0].load(Relaxed) / 1000,
            AB_CPU_US[1].load(Relaxed) / 1000,
            AB_WALL_US[0].load(Relaxed) / 1000,
            AB_WALL_US[1].load(Relaxed) / 1000,
            AB_REC_CPU_US[0].load(Relaxed) / 1000,
            AB_REC_CPU_US[1].load(Relaxed) / 1000,
            AB_REC_SYS_US[0].load(Relaxed) / 1000,
            AB_REC_SYS_US[1].load(Relaxed) / 1000
        )
    } else {
        String::new()
    };
    format!(
        "protect cmd [{}] other [{}] submit cmd {} other {}, {}{}{}{}",
        side(0),
        side(1),
        g(0, Call::Submit),
        g(1, Call::Submit),
        crate::buffers::async_copy::summary(),
        own_usage().unwrap_or_default(),
        crate::backend::pipeline_stalls(),
        ab
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timed_calls_count_per_thread_kind() {
        let before = COUNTS[1][Call::Submit as usize][0].load(Relaxed);
        assert_eq!(timed(Call::Submit, || 7), 7);
        assert_eq!(COUNTS[1][Call::Submit as usize][0].load(Relaxed), before + 1);
        let h = std::thread::spawn(|| {
            mark_recorder_thread();
            let before = COUNTS[0][Call::WatchReads as usize][0].load(Relaxed);
            timed(Call::WatchReads, || ());
            assert_eq!(COUNTS[0][Call::WatchReads as usize][0].load(Relaxed), before + 1);
            assert!(line().contains("cmd-thread") == cfg!(target_os = "linux"));
        });
        h.join().unwrap();
    }
}
