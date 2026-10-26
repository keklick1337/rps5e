// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Sync-on-address behaviour (wait/wake contract, timeouts, isolation and
//! the compare/park/wake race).

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use super::*;
use crate::sync::test_util::vm_arena;
use std::sync::atomic::AtomicI32;
use std::sync::Arc;
use std::thread;

const BASE: u64 = 0x5CF0_0000_0000;

static POLLS: AtomicI32 = AtomicI32::new(0);

fn count_poll() {
    POLLS.fetch_add(1, Ordering::Relaxed);
}

fn no_poll() {}

/// The arena is mapped once and shared by every test here.
fn vm() -> &'static ps5vm::VirtualMemory {
    static VM: std::sync::OnceLock<&'static ps5vm::VirtualMemory> = std::sync::OnceLock::new();
    VM.get_or_init(|| vm_arena(BASE))
}

fn arena() -> u64 {
    vm();
    BASE
}

fn store32(addr: u64, v: u32) {
    unsafe { (*(addr as *const AtomicU32)).store(v, Ordering::Release) }
}

fn store64(addr: u64, v: u64) {
    unsafe { (*(addr as *const AtomicU64)).store(v, Ordering::Release) }
}

fn wait32(addr: u64, expected: u32, timeout_us: Option<u32>) -> HleResult {
    wait(addr, expected as u64, 4, Deadline::from_micros(timeout_us), &no_poll)
}

fn wait64(addr: u64, expected: u64, timeout_us: Option<u32>) -> HleResult {
    wait(addr, expected, 8, Deadline::from_micros(timeout_us), &no_poll)
}

/// Each test uses its own words inside the shared arena.
fn word(test: u64, index: u64) -> u64 {
    arena() + test * 0x400 + index * 8
}

#[test]
fn invalid_address() {
    let base = word(0, 0);
    assert_eq!(wait32(0, 0, Some(1)), ERR_INVALID_ARGUMENT);
    assert_eq!(wait64(0, 0, Some(1)), ERR_INVALID_ARGUMENT);
    assert_eq!(wake(0, 1), ERR_INVALID_ARGUMENT);
    assert_eq!(wait32(base + 1, 0, Some(1)), ERR_INVALID_ARGUMENT);
    assert_eq!(wait64(base + 4, 0, Some(1)), ERR_INVALID_ARGUMENT);
    // A wake of any address wakes the aligned word that holds it.
    assert_eq!(wake(base + 1, 1), OK);
    assert_eq!(wait(base + 1, 0, 2, Deadline::from_micros(Some(1)), &no_poll), ERR_INVALID_ARGUMENT);
    assert_eq!(wake(base, -1), ERR_INVALID_ARGUMENT);
    // Not guest memory.
    assert_eq!(wait32(0x1000, 0, Some(1)), ERR_MEMORY_FAULT);
}

#[test]
fn mismatch_returns_immediately() {
    let (w32, w64) = (word(1, 0), word(1, 1));
    store32(w32, 7);
    store64(w64, 0x1_0000_0000);
    POLLS.store(0, Ordering::Relaxed);
    let start = Instant::now();
    assert_eq!(wait(w32, 6, 4, Deadline::from_micros(Some(500_000)), &count_poll), OK);
    // The full 64-bit value is compared, not just the futex half.
    assert_eq!(wait(w64, 0, 8, Deadline::from_micros(Some(500_000)), &count_poll), OK);
    assert!(start.elapsed() < Duration::from_millis(100));
    assert!(POLLS.load(Ordering::Relaxed) >= 2, "mismatch is still a signal safe point");
}

#[test]
fn timeout() {
    let w = word(2, 0);
    store32(w, 0);
    let start = Instant::now();
    assert_eq!(wait32(w, 0, Some(20_000)), ERR_TIMED_OUT);
    let elapsed = start.elapsed();
    assert!(elapsed >= Duration::from_millis(10));
    assert!(elapsed < Duration::from_millis(500));
    assert_eq!(wait32(w, 0, Some(0)), ERR_TIMED_OUT);
    store32(w, 1);
    assert_eq!(wait32(w, 0, Some(0)), OK);
}

#[test]
fn value_change_and_wake() {
    let w = word(3, 0);
    store64(w, 0);
    let ready = Arc::new(AtomicU32::new(0));
    let r = ready.clone();
    let waiter = thread::spawn(move || {
        r.store(1, Ordering::Release);
        wait64(w, 0, Some(1_000_000))
    });
    while ready.load(Ordering::Acquire) == 0 {
        thread::yield_now();
    }
    store64(w, 0x1_0000_0000);
    assert_eq!(wake(w, 1), OK);
    assert_eq!(waiter.join().unwrap(), OK);
}

#[test]
fn nanosecond_timeout() {
    let w = word(4, 0);
    store64(w, 0);
    let start = Instant::now();
    assert_eq!(wait(w, 0, 8, Deadline::after(Duration::from_millis(1)), &no_poll), ERR_TIMED_OUT);
    assert!(start.elapsed() >= Duration::from_millis(1));
    assert_eq!(wait(w, 0, 8, Deadline::after(Duration::ZERO), &no_poll), ERR_TIMED_OUT);
    assert_eq!(wait(w, 1, 8, Deadline::after(Duration::ZERO), &no_poll), OK);

    // A huge timeout does not overflow and keeps polling guest signals.
    static LONG_POLLS: AtomicI32 = AtomicI32::new(0);
    fn long_poll() {
        LONG_POLLS.fetch_add(1, Ordering::Relaxed);
    }
    let returned = Arc::new(AtomicU32::new(0));
    let r = returned.clone();
    let waker = thread::spawn(move || {
        while LONG_POLLS.load(Ordering::Relaxed) < 2 && r.load(Ordering::Acquire) == 0 {
            thread::yield_now();
        }
        store64(w, 0x1_0000_0000);
        assert_eq!(wake(w, 1), OK);
    });
    let result = wait(w, 0, 8, Deadline::after(Duration::MAX), &long_poll);
    returned.store(1, Ordering::Release);
    waker.join().unwrap();
    assert_eq!(result, OK);
    assert!(LONG_POLLS.load(Ordering::Relaxed) > 1);
}

fn spawn_waiters(w: u64, n: usize, ready: &Arc<AtomicI32>, returned: &Arc<AtomicI32>) -> Vec<thread::JoinHandle<HleResult>> {
    (0..n)
        .map(|_| {
            let (ready, returned) = (ready.clone(), returned.clone());
            thread::spawn(move || {
                ready.fetch_add(1, Ordering::Release);
                let r = wait32(w, 0, Some(1_000_000));
                returned.fetch_add(1, Ordering::Release);
                r
            })
        })
        .collect()
}

fn wait_returned(returned: &AtomicI32, at_least: i32) {
    let deadline = Instant::now() + Duration::from_millis(500);
    while returned.load(Ordering::Acquire) < at_least && Instant::now() < deadline {
        thread::yield_now();
    }
}

#[test]
fn wake_one_then_all() {
    let w = word(5, 0);
    store32(w, 0);
    let (ready, returned) = (Arc::new(AtomicI32::new(0)), Arc::new(AtomicI32::new(0)));
    let waiters = spawn_waiters(w, 4, &ready, &returned);
    while ready.load(Ordering::Acquire) != 4 {
        thread::yield_now();
    }
    thread::sleep(Duration::from_millis(50));
    assert_eq!(wake(w, 1), OK);
    wait_returned(&returned, 1);
    assert_eq!(returned.load(Ordering::Acquire), 1, "wake-one releases exactly one waiter");
    assert_eq!(wake(w, 2), OK);
    wait_returned(&returned, 3);
    assert_eq!(returned.load(Ordering::Acquire), 3, "wake-two releases exactly two more");
    assert_eq!(wake(w, i32::MAX), OK);
    for t in waiters {
        assert_eq!(t.join().unwrap(), OK);
    }
    assert_eq!(returned.load(Ordering::Acquire), 4);
}

#[test]
fn addresses_are_isolated() {
    let (first, second) = (word(6, 0), word(6, 1));
    store32(first, 0);
    store32(second, 0);
    let first_done = Arc::new(AtomicU32::new(0));
    let second_done = Arc::new(AtomicU32::new(0));
    let (f, s) = (first_done.clone(), second_done.clone());
    let a = thread::spawn(move || {
        let r = wait32(first, 0, Some(1_000_000));
        f.store(1, Ordering::Release);
        r
    });
    let b = thread::spawn(move || {
        let r = wait32(second, 0, Some(1_000_000));
        s.store(1, Ordering::Release);
        r
    });
    thread::sleep(Duration::from_millis(50));
    assert_eq!(wake(first, 1), OK);
    thread::sleep(Duration::from_millis(50));
    assert_eq!(first_done.load(Ordering::Acquire), 1);
    assert_eq!(second_done.load(Ordering::Acquire), 0, "waking one address leaves the other");
    assert_eq!(wake(second, 1), OK);
    assert_eq!(a.join().unwrap(), OK);
    assert_eq!(b.join().unwrap(), OK);
}

#[test]
fn compare_register_wake_race() {
    let w = word(7, 0);
    for _ in 0..100 {
        store32(w, 0);
        let ready = Arc::new(AtomicU32::new(0));
        let r = ready.clone();
        let waiter = thread::spawn(move || {
            r.store(1, Ordering::Release);
            wait32(w, 0, Some(500_000))
        });
        while ready.load(Ordering::Acquire) == 0 {
            thread::yield_now();
        }
        store32(w, 1);
        let _ = wake(w, 1);
        assert_eq!(waiter.join().unwrap(), OK, "compare/register/wake race never loses progress");
    }
}

#[test]
fn wake_zero_is_noop() {
    let w = word(8, 0);
    store32(w, 0);
    let (ready, returned) = (Arc::new(AtomicI32::new(0)), Arc::new(AtomicI32::new(0)));
    let waiters = spawn_waiters(w, 2, &ready, &returned);
    while ready.load(Ordering::Acquire) != 2 {
        thread::yield_now();
    }
    thread::sleep(Duration::from_millis(50));
    assert_eq!(wake(w, 0), OK);
    thread::sleep(Duration::from_millis(20));
    assert_eq!(returned.load(Ordering::Acquire), 0, "zero-count wake releases nobody");
    assert_eq!(wake(w, i32::MAX), OK);
    for t in waiters {
        assert_eq!(t.join().unwrap(), OK);
    }
}

#[test]
fn guest_entry_points() {
    let w = word(9, 0);
    let t = word(9, 1);
    store32(w, 5);
    let mut ctx = CpuContext::new(vm());
    ctx.write_u32(t, 0);
    ctx.set(crate::context::CpuRegister::Rdi, w);
    ctx.set(crate::context::CpuRegister::Rsi, 5);
    ctx.set(crate::context::CpuRegister::Rdx, t);
    assert_eq!(cb_wait(&mut ctx), ERR_TIMED_OUT);
    ctx.set(crate::context::CpuRegister::Rsi, 4);
    assert_eq!(cb_wait(&mut ctx), OK);
    // _umtx_op: wake ok, bad op -> -1.
    ctx.set(crate::context::CpuRegister::Rsi, UMTX_OP_WAKE as u64);
    ctx.set(crate::context::CpuRegister::Rdx, 1);
    ctx.set(crate::context::CpuRegister::Rcx, 0);
    assert_eq!(cb_umtx_op(&mut ctx), 0);
    ctx.set(crate::context::CpuRegister::Rsi, 99);
    assert_eq!(cb_umtx_op(&mut ctx), -1);
}

/// _umtx_op: WAIT compares the 64-bit word and WAIT_UINT(_PRIVATE) the
/// 32-bit one (a value past 32 bits never matches); a timeout is a
/// timespec or, with a larger size argument, an _umtx_time (relative or
/// UMTX_ABSTIME on its clock); invalid timeouts EINVAL; NULL objects EFAULT
/// except a private wake; wake releases a waiter without a value change;
/// reserved ops EOPNOTSUPP, ops past the last EINVAL (reference GuestUmtxOp
/// cases).
#[test]
fn umtx_op_wait_wake_and_timeouts() {
    use crate::context::CpuRegister::{Rcx, Rdi, Rdx, Rsi, R8};
    const WAIT: u64 = 2;
    const WAKE: u64 = 3;
    const WAIT_UINT: u64 = 11;
    const WAIT_UINT_PRIVATE: u64 = 15;
    const WAKE_PRIVATE: u64 = 16;
    std::thread::spawn(|| {
        let mut ctx = CpuContext::new(vm());
        let errno = crate::kernel::bind_test_errno(&ctx);
        let (value, ts, ut) = (word(12, 0), word(12, 2), word(12, 4));
        let umtx = |ctx: &mut CpuContext, obj: u64, op: u64, val: u64, size: u64, time: u64| -> (HleResult, u32) {
            for (r, v) in [(Rdi, obj), (Rsi, op), (Rdx, val), (Rcx, size), (R8, time)] {
                ctx.set(r, v);
            }
            let r = cb_umtx_op(ctx);
            (r, if r == 0 { 0 } else { ctx.read_u32(errno).unwrap() })
        };
        let timespec = |ctx: &CpuContext, sec: i64, nsec: i64| {
            ctx.write_u64(ts, sec as u64);
            ctx.write_u64(ts + 8, nsec as u64);
            ts
        };
        let umtx_time = |ctx: &CpuContext, sec: i64, nsec: i64, flags: u32, clock: u32| {
            ctx.write_u64(ut, sec as u64);
            ctx.write_u64(ut + 8, nsec as u64);
            ctx.write_u32(ut + 16, flags);
            ctx.write_u32(ut + 20, clock);
            ut
        };
        const ETIMEDOUT: u32 = 60;
        // The value differs: no wait.
        store64(value, 2);
        assert_eq!(umtx(&mut ctx, value, WAIT, 1, 0, 0), (0, 0));
        let none = timespec(&ctx, 0, 0);
        assert_eq!(umtx(&mut ctx, value, WAIT, 1, 0, none), (0, 0));
        assert_eq!(umtx(&mut ctx, value, WAIT_UINT, 1, 0, 0), (0, 0));
        assert_eq!(umtx(&mut ctx, value, WAIT_UINT_PRIVATE, 1, 0, none), (0, 0));
        // WAIT compares 64 bits, WAIT_UINT 32.
        store64(value, 0x1_0000_0001);
        assert_eq!(umtx(&mut ctx, value, WAIT, 1, 0, none), (0, 0));
        assert_eq!(umtx(&mut ctx, value, WAIT, 0x1_0000_0001, 0, none), (-1, ETIMEDOUT));
        assert_eq!(umtx(&mut ctx, value, WAIT_UINT, 1, 0, none), (-1, ETIMEDOUT));
        assert_eq!(umtx(&mut ctx, value, WAIT_UINT_PRIVATE, 0x7_0000_0001, 0, none), (0, 0), "a value past 32 bits never matches");
        // Timeouts while the value matches.
        store32(value, 1);
        let rel = timespec(&ctx, 0, 20_000_000);
        let start = std::time::Instant::now();
        assert_eq!(umtx(&mut ctx, value, WAIT_UINT, 1, 16, rel), (-1, ETIMEDOUT));
        assert!(start.elapsed() >= std::time::Duration::from_millis(15));
        let t = umtx_time(&ctx, 0, 20_000_000, 0, 4);
        let start = std::time::Instant::now();
        assert_eq!(umtx(&mut ctx, value, WAIT_UINT, 1, 24, t), (-1, ETIMEDOUT));
        assert!(start.elapsed() >= std::time::Duration::from_millis(15));
        let mut now = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
        let (mut s, mut ns) = (now.tv_sec as i64, now.tv_nsec as i64 + 20_000_000);
        if ns >= 1_000_000_000 {
            s += 1;
            ns -= 1_000_000_000;
        }
        let abs = umtx_time(&ctx, s, ns, 1, 4);
        let start = std::time::Instant::now();
        assert_eq!(umtx(&mut ctx, value, WAIT_UINT, 1, 24, abs), (-1, ETIMEDOUT));
        assert!(start.elapsed() >= std::time::Duration::from_millis(15));
        let past = umtx_time(&ctx, now.tv_sec as i64, now.tv_nsec as i64, 1, 4);
        let start = std::time::Instant::now();
        assert_eq!(umtx(&mut ctx, value, WAIT_UINT, 1, 24, past), (-1, ETIMEDOUT));
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
        // Invalid timeouts.
        let neg = timespec(&ctx, -1, 0);
        assert_eq!(umtx(&mut ctx, value, WAIT_UINT, 1, 16, neg), (-1, 22));
        assert_eq!(umtx(&mut ctx, value, WAIT, 1, 16, neg), (-1, 22));
        let many = timespec(&ctx, 0, 1_000_000_000);
        assert_eq!(umtx(&mut ctx, value, WAIT_UINT, 1, 16, many), (-1, 22));
        // Size argument 0: a 32-bit microsecond timeout (what libc.prx
        // passes); the bytes after it are not read.
        ctx.write_u32(ts, 20_000);
        ctx.write_u32(ts + 4, 0);
        ctx.write_u64(ts + 8, u64::MAX);
        let start = std::time::Instant::now();
        assert_eq!(umtx(&mut ctx, value, WAIT_UINT, 1, 0, ts), (-1, ETIMEDOUT));
        assert!(start.elapsed() >= std::time::Duration::from_millis(15));
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
        let negns = umtx_time(&ctx, 0, -1, 0, 4);
        assert_eq!(umtx(&mut ctx, value, WAIT_UINT, 1, 24, negns), (-1, 22));
        // NULL objects; reserved and unknown operations.
        assert_eq!(umtx(&mut ctx, 0, WAIT, 0, 0, 0), (-1, 14));
        assert_eq!(umtx(&mut ctx, 0, WAIT_UINT, 0, 0, 0), (-1, 14));
        assert_eq!(umtx(&mut ctx, 0, WAKE, 1, 0, 0), (-1, 14));
        assert_eq!(umtx(&mut ctx, 0, WAKE_PRIVATE, 1, 0, 0), (0, 0));
        assert_eq!(umtx(&mut ctx, value, 0, 0, 0, 0), (-1, 45));
        assert_eq!(umtx(&mut ctx, value, 1, 0, 0, 0), (-1, 45));
        assert_eq!(umtx(&mut ctx, value, 27, 0, 0, 0), (-1, 22));
        // NWAKE_PRIVATE: a count of 0 or less reads nothing, a null list
        // is EFAULT.
        assert_eq!(umtx(&mut ctx, 0, 21, 0, 0, 0), (0, 0));
        assert_eq!(umtx(&mut ctx, 0, 21, (-3i64) as u64, 0, 0), (0, 0));
        assert_eq!(umtx(&mut ctx, 0, 21, 2, 0, 0), (-1, 14));
        assert_eq!(umtx(&mut ctx, value, (-1i64) as u64, 0, 0, 0), (-1, 22));
    })
    .join()
    .unwrap();
    // A wake releases a waiter whose value has not changed (count 0 still
    // wakes one).
    let w = word(13, 0);
    store32(w, 1);
    let waiter = thread::spawn(move || {
        let mut ctx = CpuContext::new(vm());
        crate::kernel::bind_test_errno(&ctx);
        for (r, v) in [(Rdi, w), (Rsi, WAIT_UINT_PRIVATE), (Rdx, 1), (Rcx, 0), (R8, 0)] {
            ctx.set(r, v);
        }
        cb_umtx_op(&mut ctx)
    });
    thread::sleep(Duration::from_millis(50));
    let mut ctx = CpuContext::new(vm());
    for (r, v) in [(Rdi, w), (Rsi, WAKE), (Rdx, 0), (Rcx, 0), (R8, 0)] {
        ctx.set(r, v);
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while !waiter.is_finished() {
        assert!(Instant::now() < deadline, "the waiter was not woken");
        assert_eq!(cb_umtx_op(&mut ctx), 0);
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(waiter.join().unwrap(), 0);
}

/// UMTX_OP_NWAKE_PRIVATE wakes every waiter of each listed address.
#[test]
fn umtx_nwake_private_wakes_each_listed_address() {
    use crate::context::CpuRegister::{Rcx, Rdi, Rdx, Rsi, R8};
    let (a, b, list) = (word(14, 0), word(14, 1), word(14, 4));
    store32(a, 1);
    store32(b, 1);
    let ready = Arc::new(AtomicI32::new(0));
    let returned = Arc::new(AtomicI32::new(0));
    let mut waiters = spawn_waiters(a, 2, &ready, &returned);
    waiters.extend(spawn_waiters(b, 2, &ready, &returned));
    while ready.load(Ordering::SeqCst) < 4 {
        thread::sleep(Duration::from_millis(1));
    }
    thread::sleep(Duration::from_millis(20));
    let mut ctx = CpuContext::new(vm());
    ctx.write_u64(list, a);
    ctx.write_u64(list + 8, b);
    for (r, v) in [(Rdi, list), (Rsi, 21), (Rdx, 2), (Rcx, 0), (R8, 0)] {
        ctx.set(r, v);
    }
    assert_eq!(cb_umtx_op(&mut ctx), 0);
    wait_returned(&returned, 4);
    for w in waiters {
        assert_eq!(w.join().unwrap(), OK);
    }
}

/// sceKernelSyncOnAddressWait8/16 wait while the byte or halfword matches
/// (its neighbours in the word do not matter) and are released by a store
/// plus a wake of their own address.
#[test]
fn byte_and_halfword_waits() {
    let w = word(15, 0);
    store64(w, 0);
    let b = w + 1;
    let h = w + 2;
    // Mismatch and timeout.
    assert_eq!(wait(b, 1, 1, Deadline::from_micros(Some(1000)), &no_poll), OK);
    assert_eq!(wait(b, 0, 1, Deadline::from_micros(Some(1000)), &no_poll), ERR_TIMED_OUT);
    assert_eq!(wait(h, 0, 2, Deadline::from_micros(Some(1000)), &no_poll), ERR_TIMED_OUT);
    assert_eq!(wait(w + 1, 0, 2, Deadline::from_micros(Some(1000)), &no_poll), ERR_INVALID_ARGUMENT);
    // A neighbour changing does not release the byte wait.
    let waiter = thread::spawn(move || wait(b, 0, 1, Deadline::from_micros(Some(2_000_000)), &no_poll));
    thread::sleep(Duration::from_millis(20));
    unsafe { (*(w as *const std::sync::atomic::AtomicU8)).store(7, Ordering::Release) };
    thread::sleep(Duration::from_millis(30));
    assert!(!waiter.is_finished(), "a neighbouring byte released the wait");
    unsafe { (*(b as *const std::sync::atomic::AtomicU8)).store(1, Ordering::Release) };
    assert_eq!(wake(b, 1), OK);
    assert_eq!(waiter.join().unwrap(), OK);
    // The guest entry takes the comparand by value in its size.
    let mut ctx = CpuContext::new(vm());
    ctx.set(crate::context::CpuRegister::Rdi, h);
    ctx.set(crate::context::CpuRegister::Rsi, 0xFFFF_0000);
    ctx.set(crate::context::CpuRegister::Rdx, 0);
    let waiter = thread::spawn(move || {
        let mut ctx = CpuContext::new(vm());
        ctx.set(crate::context::CpuRegister::Rdi, h);
        ctx.set(crate::context::CpuRegister::Rsi, 0xFFFF_0000);
        ctx.set(crate::context::CpuRegister::Rdx, 0);
        cb_wait16(&mut ctx)
    });
    thread::sleep(Duration::from_millis(20));
    assert!(!waiter.is_finished(), "the halfword 0 matches the low 16 bits of the comparand");
    unsafe { (*(h as *const std::sync::atomic::AtomicU16)).store(9, Ordering::Release) };
    assert_eq!(cb_wake_at(&mut ctx, h), OK);
    assert_eq!(waiter.join().unwrap(), OK);
}

fn cb_wake_at(ctx: &mut CpuContext, addr: u64) -> HleResult {
    ctx.set(crate::context::CpuRegister::Rdi, addr);
    ctx.set(crate::context::CpuRegister::Rsi, i32::MAX as u64);
    cb_wake(ctx)
}

#[test]
fn sync_on_address_nids_match_names() {
    for &(nid, name, _, _) in EXPORTS {
        assert_eq!(ps5nid::compute_nid(name), nid, "{name}");
    }
}

/// UMTX_OP_SEM_WAIT / UMTX_OP_SEM_WAKE on a `struct _usem`: a wait returns
/// at once while the count is non-zero and sets `_has_waiters`; with a zero
/// count it times out, or a wake releases it and clears the flag; a wake
/// without waiters succeeds; NULL is EFAULT (reference GuestUmtxOp cases).
#[test]
fn umtx_op_semaphore() {
    use crate::context::CpuRegister::{Rcx, Rdi, Rdx, Rsi, R8};
    const SEM_WAIT: u64 = 19;
    const SEM_WAKE: u64 = 20;
    fn umtx(obj: u64, op: u64, size: u64, time: u64) -> HleResult {
        let mut ctx = CpuContext::new(vm());
        for (r, v) in [(Rdi, obj), (Rsi, op), (Rdx, 0), (Rcx, size), (R8, time)] {
            ctx.set(r, v);
        }
        cb_umtx_op(&mut ctx)
    }
    std::thread::spawn(|| {
        let ctx = CpuContext::new(vm());
        let errno = crate::kernel::bind_test_errno(&ctx);
        let (sem, ts) = (word(20, 0), word(20, 4));
        let (has_waiters, count) = (sem, sem + 4);
        // Count non-zero: at once.
        store32(has_waiters, 0);
        store32(count, 2);
        ctx.write_u64(ts, 0);
        ctx.write_u64(ts + 8, 0);
        assert_eq!(umtx(sem, SEM_WAIT, 16, ts), 0);
        assert_eq!(((read_word(has_waiters, 4) as u32), (read_word(count, 4) as u32)), (1, 2));
        // Count zero: times out after the timeout.
        store32(has_waiters, 0);
        store32(count, 0);
        ctx.write_u64(ts + 8, 20_000_000);
        let start = Instant::now();
        assert_eq!(umtx(sem, SEM_WAIT, 16, ts), -1);
        assert_eq!(ctx.read_u32(errno), Some(60));
        assert!(start.elapsed() >= Duration::from_millis(15));
        assert_eq!((read_word(has_waiters, 4) as u32), 1);
        // A wake releases the sleeper and clears the flag.
        store32(has_waiters, 0);
        let waiter = std::thread::spawn(move || {
            let ctx = CpuContext::new(vm());
            crate::kernel::bind_test_errno(&ctx);
            umtx(sem, SEM_WAIT, 0, 0)
        });
        let end = Instant::now() + Duration::from_secs(5);
        while (read_word(has_waiters, 4) as u32) == 0 || !SEM_QUEUES.lock().unwrap().contains_key(&sem) {
            assert!(Instant::now() < end);
            std::thread::sleep(Duration::from_millis(1));
        }
        std::thread::sleep(Duration::from_millis(30));
        assert!(!waiter.is_finished());
        store32(count, 1);
        assert_eq!(umtx(sem, SEM_WAKE, 0, 0), 0);
        assert_eq!(waiter.join().unwrap(), 0);
        assert_eq!((read_word(has_waiters, 4) as u32), 0);
        // No waiters; NULL objects; an unaligned object.
        assert_eq!(umtx(sem, SEM_WAKE, 0, 0), 0);
        assert_eq!((read_word(has_waiters, 4) as u32), 0);
        assert_eq!(umtx(0, SEM_WAKE, 0, 0), -1);
        assert_eq!(ctx.read_u32(errno), Some(14));
        assert_eq!(umtx(0, SEM_WAIT, 0, 0), -1);
        assert_eq!(ctx.read_u32(errno), Some(14));
        assert_eq!(umtx(sem + 2, SEM_WAKE, 0, 0), -1);
        assert_eq!(ctx.read_u32(errno), Some(22));
    })
    .join()
    .unwrap();
}
