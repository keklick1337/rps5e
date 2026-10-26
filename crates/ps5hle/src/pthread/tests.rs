// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only
//
// Unit tests of the `ps5hle::pthread` module.

use super::*;
use crate::context::CpuRegister;
use ps5vm::{MemProtect, VirtualMemory};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

/// One scratch page per test; every test binds its threads to distinct
/// fake guest thread handles.
struct Scratch {
    vm: &'static VirtualMemory,
    base: u64,
}

impl Scratch {
    fn new(base: u64) -> Self {
        let vm: &'static VirtualMemory = Box::leak(Box::new(VirtualMemory::new()));
        vm.map(base, 0x10000, MemProtect::RW).unwrap();
        Scratch { vm, base }
    }

    fn ctx(&self) -> CpuContext<'static> {
        CpuContext::new(self.vm)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = self.vm.unmap(self.base, 0x10000);
    }
}

fn call(ctx: &mut CpuContext, cb: crate::registry::HleCallback, args: &[u64]) -> HleResult {
    let regs = [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx, CpuRegister::R8];
    for (r, v) in regs.iter().zip(args) {
        ctx.set(*r, *v);
    }
    cb(ctx)
}

static NEXT_HANDLE: AtomicU32 = AtomicU32::new(1);

/// Bind the calling test thread to a fresh fake handle.
fn bind() -> u64 {
    let n = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    let h = 0x7_0000_0000 + n as u64 * 0x100;
    bind_test_thread(h, n);
    h
}

#[test]
fn errorcheck_mutex_semantics() {
    let s = Scratch::new(0x5CE0_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let m = s.base;
    assert_eq!(call(&mut ctx, cb_mutex_init, &[m, 0, 0]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), EDEADLK);
    assert_eq!(call(&mut ctx, cb_mutex_trylock, &[m]), EBUSY);
    // A timed relock by the owner times out instead of deadlocking.
    assert_eq!(call(&mut ctx, cb_mutex_timedlock_usec, &[m, 0]), ETIMEDOUT);
    let t = Instant::now();
    assert_eq!(call(&mut ctx, cb_mutex_timedlock_usec, &[m, 2000]), ETIMEDOUT);
    assert!(t.elapsed() >= Duration::from_micros(2000));
    assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), EBUSY);
    assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), EPERM);
    assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), OK);
    assert_eq!(ctx.read_u64(m), Some(2), "THR_MUTEX_DESTROYED");
    // Destroying a destroyed mutex is invalid; a null pointer too.
    assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), EINVAL);
    assert_eq!(call(&mut ctx, cb_mutex_destroy, &[0]), EINVAL);
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[0]), EINVAL);
}

#[test]
fn recursive_mutex_counts_and_trylock() {
    let s = Scratch::new(0x5CE1_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (attr, m) = (s.base, s.base + 8);
    assert_eq!(call(&mut ctx, cb_mutexattr_init, &[attr]), OK);
    assert_eq!(call(&mut ctx, cb_mutexattr_settype, &[attr, 5]), EINVAL);
    assert_eq!(call(&mut ctx, cb_mutexattr_settype, &[attr, MUTEX_RECURSIVE as u64]), OK);
    assert_eq!(call(&mut ctx, cb_mutexattr_gettype, &[attr, s.base + 0x100]), OK);
    assert_eq!(ctx.read_u32(s.base + 0x100), Some(MUTEX_RECURSIVE as u32));
    assert_eq!(call(&mut ctx, cb_mutexattr_setprotocol, &[attr, 3]), EINVAL);
    assert_eq!(call(&mut ctx, cb_mutex_init, &[m, attr, 0]), OK);
    assert_eq!(call(&mut ctx, cb_mutexattr_destroy, &[attr]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_trylock, &[m]), OK);
    for _ in 0..3 {
        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
    }
    assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), EPERM);
}

#[test]
fn static_initializers_create_objects() {
    let s = Scratch::new(0x5CE2_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (m0, m1, c, l) = (s.base, s.base + 8, s.base + 16, s.base + 24);
    ctx.write_u64(m1, MUTEX_STATIC_ADAPTIVE_INITIALIZER);
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m0]), OK);
    assert!(ctx.read_u64(m0).unwrap() >= OBJECT_REGION);
    assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m0]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m1]), OK);
    let obj = ctx.read_u64(m1).unwrap();
    assert_eq!(ctx.read_u32(obj + 0x20), Some(MUTEX_ADAPTIVE as u32));
    // Adaptive mutexes are not recursive.
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m1]), EDEADLK);
    assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m1]), OK);
    assert_eq!(call(&mut ctx, cb_cond_signal, &[c]), OK);
    assert!(ctx.read_u64(c).unwrap() >= OBJECT_REGION);
    assert_eq!(call(&mut ctx, cb_rwlock_rdlock, &[l]), OK);
    assert_eq!(call(&mut ctx, cb_rwlock_unlock, &[l]), OK);
    // A garbage word is not an object.
    ctx.write_u64(s.base + 32, 0x1234_5678);
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[s.base + 32]), EINVAL);
}

#[test]
fn mutex_blocks_other_threads_until_unlock() {
    let s = Scratch::new(0x5CE3_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let m = s.base;
    assert_eq!(call(&mut ctx, cb_mutex_init, &[m, 0, 0]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
    let vm = s.vm;
    let other = std::thread::spawn(move || {
        let mut ctx = CpuContext::new(vm);
        bind();
        assert_eq!(call(&mut ctx, cb_mutex_trylock, &[m]), EBUSY);
        assert_eq!(call(&mut ctx, cb_mutex_timedlock_usec, &[m, 0]), ETIMEDOUT);
        assert_eq!(call(&mut ctx, cb_mutex_timedlock_usec, &[m, 1000]), ETIMEDOUT);
        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), EPERM);
        let t = Instant::now();
        assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
        let waited = t.elapsed();
        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
        waited
    });
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
    assert!(other.join().unwrap() >= Duration::from_millis(10));
}

#[test]
fn cond_signal_wakes_one_waiter_in_order() {
    let s = Scratch::new(0x5CE4_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (m, c) = (s.base, s.base + 8);
    assert_eq!(call(&mut ctx, cb_mutex_init, &[m, 0, 0]), OK);
    assert_eq!(call(&mut ctx, cb_cond_init, &[c, 0, 0]), OK);
    let vm = s.vm;
    let woken = Arc::new(AtomicU32::new(0));
    let mut threads = Vec::new();
    for _ in 0..2 {
        let woken = woken.clone();
        threads.push(std::thread::spawn(move || {
            let mut ctx = CpuContext::new(vm);
            bind();
            assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
            let r = call(&mut ctx, cb_sce_cond_timedwait, &[c, m, 300_000]);
            if r == OK {
                woken.fetch_add(1, Ordering::SeqCst);
            }
            // The mutex is owned again after the wait, either way.
            assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
            r
        }));
    }
    // Wait until both sleep on the condition.
    let obj = loop {
        let o = ctx.read_u64(c).unwrap();
        if let Some(st) = CONDS.get(o) {
            if st.waiters() == 2 {
                break o;
            }
        }
        std::thread::sleep(Duration::from_millis(1));
    };
    let _ = obj;
    assert_eq!(call(&mut ctx, cb_cond_signal, &[c]), OK);
    let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(woken.load(Ordering::SeqCst), 1);
    assert!(results.contains(&ETIMEDOUT));
}

#[test]
fn cond_broadcast_and_signalto() {
    let s = Scratch::new(0x5CE5_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (m, c) = (s.base, s.base + 8);
    assert_eq!(call(&mut ctx, cb_mutex_init, &[m, 0, 0]), OK);
    assert_eq!(call(&mut ctx, cb_cond_init, &[c, 0, 0]), OK);
    let vm = s.vm;
    let spawn = |n: usize| {
        (0..n)
            .map(|_| {
                let (tx, rx) = std::sync::mpsc::channel();
                let t = std::thread::spawn(move || {
                    let mut ctx = CpuContext::new(vm);
                    tx.send(bind()).unwrap();
                    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
                    let r = call(&mut ctx, cb_sce_cond_timedwait, &[c, m, 2_000_000]);
                    assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
                    r
                });
                (rx.recv().unwrap(), t)
            })
            .collect::<Vec<_>>()
    };
    let wait_for = |ctx: &CpuContext, n: usize| loop {
        let o = ctx.read_u64(c).unwrap();
        if CONDS.get(o).is_some_and(|st| st.waiters() == n) {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    };
    let ts = spawn(3);
    wait_for(&ctx, 3);
    assert_eq!(call(&mut ctx, cb_cond_broadcast, &[c]), OK);
    for (_, t) in ts {
        assert_eq!(t.join().unwrap(), OK);
    }
    let mut ts = spawn(2);
    wait_for(&ctx, 2);
    let (target, t) = ts.remove(1);
    assert_eq!(call(&mut ctx, cb_cond_signalto, &[c, target]), OK);
    assert_eq!(t.join().unwrap(), OK);
    wait_for(&ctx, 1);
    assert_eq!(call(&mut ctx, cb_cond_signalto, &[c, 0]), EINVAL);
    assert_eq!(call(&mut ctx, cb_cond_broadcast, &[c]), OK);
    for (_, t) in ts {
        assert_eq!(t.join().unwrap(), OK);
    }
}

#[test]
fn cond_wait_checks_owner_and_keeps_recursion() {
    let s = Scratch::new(0x5CE6_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (attr, m, c) = (s.base, s.base + 8, s.base + 16);
    assert_eq!(call(&mut ctx, cb_cond_init, &[c, 0, 0]), OK);
    assert_eq!(call(&mut ctx, cb_mutexattr_init, &[attr]), OK);
    assert_eq!(call(&mut ctx, cb_mutexattr_settype, &[attr, MUTEX_RECURSIVE as u64]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_init, &[m, attr, 0]), OK);
    // Not owned: EPERM.
    assert_eq!(call(&mut ctx, cb_sce_cond_timedwait, &[c, m, 10]), EPERM);
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
    // usec 0 times out at once and the recursion depth comes back.
    assert_eq!(call(&mut ctx, cb_sce_cond_timedwait, &[c, m, 0]), ETIMEDOUT);
    assert_eq!(call(&mut ctx, cb_sce_cond_timedwait, &[c, m, 1000]), ETIMEDOUT);
    assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), EPERM);
    // POSIX timedwait with an absolute deadline in the past, and a bad one.
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
    let ts = s.base + 0x100;
    ctx.write_u64(ts, 1);
    ctx.write_u64(ts + 8, 0);
    assert_eq!(call(&mut ctx, cb_posix_cond_timedwait, &[c, m, ts]), ETIMEDOUT);
    ctx.write_u64(ts + 8, 1_000_000_000);
    assert_eq!(call(&mut ctx, cb_posix_cond_timedwait, &[c, m, ts]), EINVAL);
    assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
}

#[test]
fn condattr_clock_selects_deadline_clock() {
    let s = Scratch::new(0x5CE7_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (attr, m, c, ts) = (s.base, s.base + 8, s.base + 16, s.base + 0x100);
    assert_eq!(call(&mut ctx, cb_condattr_init, &[attr]), OK);
    for bad in [-1i64, 3, 5, 12, 13, 14, 15] {
        assert_eq!(call(&mut ctx, cb_condattr_setclock, &[attr, bad as u64]), EINVAL, "clock {bad}");
    }
    for good in [0u64, 1, 2] {
        assert_eq!(call(&mut ctx, cb_condattr_setclock, &[attr, good]), OK, "clock {good}");
    }
    assert_eq!(call(&mut ctx, cb_condattr_setclock, &[attr, crate::time::CLOCK_MONOTONIC as u64]), OK);
    assert_eq!(call(&mut ctx, cb_cond_init, &[c, attr, 0]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_init, &[m, 0, 0]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
    // 20 ms ahead on the monotonic clock.
    let now = crate::time::clock_now(crate::time::CLOCK_MONOTONIC).unwrap() + Duration::from_millis(20);
    ctx.write_u64(ts, now.as_secs());
    ctx.write_u64(ts + 8, now.subsec_nanos() as u64);
    let t = Instant::now();
    assert_eq!(call(&mut ctx, cb_posix_cond_timedwait, &[c, m, ts]), ETIMEDOUT);
    let waited = t.elapsed();
    assert!(waited >= Duration::from_millis(15) && waited < Duration::from_secs(2), "{waited:?}");
    assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
}

/// The POSIX rwlock attribute pair, and a lock initialised from it
/// (pthread_rwlockattr_init/destroy).
#[test]
fn posix_rwlockattr_init_and_destroy() {
    let s = Scratch::new(0x5F8A_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (attr, l) = (s.base, s.base + 8);
    assert_eq!(call(&mut ctx, px_rwlockattr_init, &[attr]), 0);
    assert_ne!(ctx.read_u64(attr), Some(0));
    assert_eq!(call(&mut ctx, px_rwlock_init, &[l, attr]), 0);
    assert_eq!(call(&mut ctx, px_rwlock_wrlock, &[l]), 0);
    assert_eq!(call(&mut ctx, px_rwlock_tryrdlock, &[l]), 16, "EBUSY");
    assert_eq!(call(&mut ctx, px_rwlock_unlock, &[l]), 0);
    assert_eq!(call(&mut ctx, px_rwlock_destroy, &[l]), 0);
    assert_eq!(call(&mut ctx, px_rwlockattr_destroy, &[attr]), 0);
    assert_eq!(call(&mut ctx, px_rwlockattr_init, &[0]), 22, "EINVAL");
}

#[test]
fn rwlock_writer_preference_and_errors() {
    let s = Scratch::new(0x5CE8_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let l = s.base;
    assert_eq!(call(&mut ctx, cb_rwlock_init, &[l, 0, 0]), OK);
    assert_eq!(call(&mut ctx, cb_rwlock_rdlock, &[l]), OK);
    assert_eq!(call(&mut ctx, cb_rwlock_rdlock, &[l]), OK);
    assert_eq!(call(&mut ctx, cb_rwlock_trywrlock, &[l]), EDEADLK);
    assert_eq!(call(&mut ctx, cb_rwlock_wrlock, &[l]), EDEADLK);
    let vm = s.vm;
    let reader_started = Arc::new(AtomicU32::new(0));
    // A writer waits behind our read locks; new readers wait behind it.
    let writer = std::thread::spawn(move || {
        let mut ctx = CpuContext::new(vm);
        bind();
        assert_eq!(call(&mut ctx, cb_rwlock_trywrlock, &[l]), EBUSY);
        assert_eq!(call(&mut ctx, cb_rwlock_timedwrlock, &[l, 0]), ETIMEDOUT);
        assert_eq!(call(&mut ctx, cb_rwlock_wrlock, &[l]), OK);
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(call(&mut ctx, cb_rwlock_unlock, &[l]), OK);
    });
    loop {
        let o = ctx.read_u64(l).unwrap();
        if RWLOCKS.get(o).is_some_and(|st| st.waiting_writers() == 1) {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let rs = reader_started.clone();
    let reader = std::thread::spawn(move || {
        let mut ctx = CpuContext::new(vm);
        bind();
        assert_eq!(call(&mut ctx, cb_rwlock_tryrdlock, &[l]), EBUSY);
        rs.store(1, Ordering::SeqCst);
        assert_eq!(call(&mut ctx, cb_rwlock_rdlock, &[l]), OK);
        assert_eq!(call(&mut ctx, cb_rwlock_unlock, &[l]), OK);
    });
    while reader_started.load(Ordering::SeqCst) == 0 {
        std::thread::sleep(Duration::from_millis(1));
    }
    // An existing reader may still read-lock recursively.
    assert_eq!(call(&mut ctx, cb_rwlock_tryrdlock, &[l]), OK);
    for _ in 0..3 {
        assert_eq!(call(&mut ctx, cb_rwlock_unlock, &[l]), OK);
    }
    assert_eq!(call(&mut ctx, cb_rwlock_unlock, &[l]), EPERM);
    writer.join().unwrap();
    reader.join().unwrap();
    // The writer's own read lock: EDEADLK, or EBUSY for the try variant; the
    // timed one waits its time out and then reports ETIMEDOUT (measured on
    // hardware).
    assert_eq!(call(&mut ctx, cb_rwlock_wrlock, &[l]), OK);
    assert_eq!(call(&mut ctx, cb_rwlock_rdlock, &[l]), EDEADLK);
    assert_eq!(call(&mut ctx, cb_rwlock_timedrdlock, &[l, 1000]), ETIMEDOUT);
    assert_eq!(call(&mut ctx, cb_rwlock_tryrdlock, &[l]), EBUSY);
    assert_eq!(call(&mut ctx, cb_rwlock_unlock, &[l]), OK);
    assert_eq!(call(&mut ctx, cb_rwlock_unlock, &[l]), EPERM);
    assert_eq!(call(&mut ctx, cb_rwlock_destroy, &[l]), OK);
}

#[test]
fn keys_generation_and_limits() {
    let s = Scratch::new(0x5CE9_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let out = s.base;
    assert_eq!(call(&mut ctx, cb_key_create, &[0, 0]), EINVAL);
    assert_eq!(call(&mut ctx, cb_key_create, &[out, 0x1234]), OK);
    let key = ctx.read_u32(out).unwrap() as u64;
    assert!(key >= 1 && key <= KEYS_MAX as u64);
    assert_eq!(call(&mut ctx, cb_getspecific, &[key]), 0);
    assert_eq!(call(&mut ctx, cb_setspecific, &[key, 0x55]), OK);
    assert_eq!(call(&mut ctx, cb_getspecific, &[key]), 0x55);
    // Another thread sees its own value.
    let other = std::thread::spawn(move || {
        let mut ctx = CpuContext::new(s_vm());
        call(&mut ctx, cb_getspecific, &[key])
    });
    assert_eq!(other.join().unwrap(), 0);
    assert_eq!(call(&mut ctx, cb_key_delete, &[key]), OK);
    assert_eq!(call(&mut ctx, cb_key_delete, &[key]), EINVAL);
    assert_eq!(call(&mut ctx, cb_getspecific, &[key]), 0);
    assert_eq!(call(&mut ctx, cb_setspecific, &[key, 1]), EINVAL);
    assert_eq!(call(&mut ctx, cb_setspecific, &[0, 1]), EINVAL);
    assert_eq!(call(&mut ctx, cb_key_delete, &[KEYS_MAX as u64 + 1]), EINVAL);
}

fn s_vm() -> &'static VirtualMemory {
    Box::leak(Box::new(VirtualMemory::new()))
}

#[test]
fn thread_attr_validation() {
    let s = Scratch::new(0x5CEA_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (attr, out) = (s.base, s.base + 0x100);
    assert_eq!(call(&mut ctx, cb_attr_init, &[attr]), OK);
    assert_eq!(call(&mut ctx, cb_attr_setstacksize, &[attr, 0x3FFF]), EINVAL);
    assert_eq!(call(&mut ctx, cb_attr_setstacksize, &[attr, 0x4000]), OK);
    assert_eq!(call(&mut ctx, cb_attr_setstack, &[attr, 0, 0x10000]), EINVAL);
    assert_eq!(call(&mut ctx, cb_attr_setstack, &[attr, 0x1000_0000, 0x100]), EINVAL);
    assert_eq!(call(&mut ctx, cb_attr_setstack, &[attr, 0x1000_0000, 0x10000]), OK);
    assert_eq!(call(&mut ctx, cb_attr_getstack, &[attr, out, out + 8]), OK);
    assert_eq!(ctx.read_u64(out), Some(0x1000_0000));
    assert_eq!(ctx.read_u64(out + 8), Some(0x10000));
    assert_eq!(call(&mut ctx, cb_attr_getstack, &[attr, 0, out]), EINVAL);
    assert_eq!(call(&mut ctx, cb_attr_setstackaddr, &[attr, 0]), EINVAL);
    assert_eq!(call(&mut ctx, cb_attr_setstackaddr, &[attr, 0x2000_0000]), OK);
    assert_eq!(call(&mut ctx, cb_attr_getstackaddr, &[attr, out]), OK);
    assert_eq!(ctx.read_u64(out), Some(0x2000_0000));
    assert_eq!(call(&mut ctx, cb_attr_setdetachstate, &[attr, 2]), EINVAL);
    assert_eq!(call(&mut ctx, cb_attr_setinheritsched, &[attr, 1]), EOPNOTSUPP);
    assert_eq!(call(&mut ctx, cb_attr_setinheritsched, &[attr, 0]), OK);
    assert_eq!(call(&mut ctx, cb_attr_getinheritsched, &[attr, out]), OK);
    assert_eq!(ctx.read_u32(out), Some(0));
    assert_eq!(call(&mut ctx, cb_attr_getschedpolicy, &[attr, out]), OK);
    assert_eq!(ctx.read_u32(out), Some(DEFAULT_POLICY as u32));
    assert_eq!(call(&mut ctx, cb_attr_setschedparam, &[attr, 0]), EOPNOTSUPP);
    assert_eq!(call(&mut ctx, cb_attr_getschedparam, &[attr, out]), OK);
    assert_eq!(ctx.read_u32(out), Some(DEFAULT_PRIORITY as u32));
    assert_eq!(call(&mut ctx, cb_attr_getdetachstate, &[attr, 0]), EINVAL);
    assert_eq!(call(&mut ctx, cb_attr_destroy, &[attr]), OK);
    assert_eq!(call(&mut ctx, cb_attr_destroy, &[attr]), EINVAL);
}

#[test]
fn posix_names_return_errno_values() {
    let s = Scratch::new(0x5CEB_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let m = s.base;
    assert_eq!(call(&mut ctx, px_mutex_init, &[m, 0]), 0);
    assert_eq!(call(&mut ctx, px_mutex_lock, &[m]), 0);
    assert_eq!(call(&mut ctx, px_mutex_lock, &[m]), 11); // EDEADLK
    assert_eq!(call(&mut ctx, px_mutex_trylock, &[m]), 16); // EBUSY
    assert_eq!(call(&mut ctx, px_mutex_unlock, &[m]), 0);
    assert_eq!(call(&mut ctx, px_mutex_unlock, &[m]), 1); // EPERM
    assert_eq!(call(&mut ctx, px_mutex_lock, &[0]), 22); // EINVAL
    assert_eq!(posix_error(0x8002_0000u32 as i32 as i64), 1062);
    assert_eq!(posix_error(OK), 0);
    // pthread_mutex_timedlock with an absolute deadline already past.
    assert_eq!(call(&mut ctx, px_mutex_lock, &[m]), 0);
    let ts = s.base + 0x100;
    ctx.write_u64(ts, 1);
    ctx.write_u64(ts + 8, 0);
    assert_eq!(call(&mut ctx, px_mutex_timedlock, &[m, ts]), 60); // ETIMEDOUT
    ctx.write_u64(ts + 8, u64::MAX);
    assert_eq!(call(&mut ctx, px_mutex_timedlock, &[m, ts]), 22);
    assert_eq!(call(&mut ctx, px_mutex_unlock, &[m]), 0);
}

#[test]
fn export_nids_match_names() {
    for &(nid, name, _) in SCE_EXPORTS.iter().chain(POSIX_EXPORTS).chain(super::posix_ext::POSIX_EXT_EXPORTS) {
        assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
    }
}

/// Uncontended lock/unlock cost through the HLE callbacks.
#[test]
#[ignore]
fn bench_uncontended_mutex() {
    let s = Scratch::new(0x5CEC_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (m, c, key) = (s.base, s.base + 8, s.base + 16);
    assert_eq!(call(&mut ctx, cb_mutex_init, &[m, 0, 0]), OK);
    let n = 2_000_000;
    let t = Instant::now();
    for _ in 0..n {
        call(&mut ctx, cb_mutex_lock, &[m]);
        call(&mut ctx, cb_mutex_unlock, &[m]);
    }
    eprintln!("lock+unlock: {:.1} ns", t.elapsed().as_nanos() as f64 / n as f64);
    let t = Instant::now();
    for _ in 0..n {
        call(&mut ctx, cb_cond_signal, &[c]);
    }
    eprintln!("cond_signal (no waiters): {:.1} ns", t.elapsed().as_nanos() as f64 / n as f64);
    call(&mut ctx, cb_key_create, &[key, 0]);
    let k = ctx.read_u32(key).unwrap() as u64;
    call(&mut ctx, cb_setspecific, &[k, 7]);
    let t = Instant::now();
    for _ in 0..n {
        call(&mut ctx, cb_getspecific, &[k]);
    }
    eprintln!("getspecific: {:.1} ns", t.elapsed().as_nanos() as f64 / n as f64);
    let t = Instant::now();
    for _ in 0..n {
        call(&mut ctx, cb_getthreadid, &[]);
    }
    eprintln!("getthreadid: {:.1} ns", t.elapsed().as_nanos() as f64 / n as f64);
}

/// Held by tests that check thread affinities: a process-wide cpuset change
/// rewrites the affinity of every registered thread.
pub(crate) static AFFINITY_TESTS: Mutex<()> = Mutex::new(());

/// Register a fake live guest thread for the calling test thread and bind
/// to it; returns its handle and id.
pub(crate) fn fake_thread(name: &str, affinity: u64) -> (u64, u32) {
    fake_thread_with(name, affinity, false)
}

/// [`fake_thread`], marked as the process's initial thread or not.
fn fake_thread_with(name: &str, affinity: u64, is_main: bool) -> (u64, u32) {
    let n = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    let handle = 0x7_8000_0000 + n as u64 * 0x100;
    let id = 0x1000_0000 + n;
    let info = Arc::new(ThreadInfo {
        handle,
        id,
        tp: 0,
        name: Mutex::new(name.into()),
        stack_top: 0,
        stack_size: 0,
        attr: Mutex::new(ThreadAttr { affinity, ..ThreadAttr::default() }),
        join: Mutex::new(JoinState::default()),
        exit: Mutex::new(None),
        exited: Condvar::new(),
        is_main,
        user_stack: true,
        cpu_clock: Mutex::new(host_cpu_clock()),
        suspended: Mutex::new(false),
        resumed: Condvar::new(),
    });
    threads(|t| t.insert(handle, info));
    bind_test_thread(handle, id);
    (handle, id)
}

pub(crate) fn forget_fake_thread(handle: u64) {
    threads(|t| t.remove(&handle));
}

/// scePthreadMain is pthread_main_np: 1 on the initial thread, 0 on any
/// other.
#[test]
fn sce_pthread_main_is_pthread_main_np() {
    let export = |name: &str| SCE_EXPORTS.iter().chain(POSIX_EXPORTS).chain(SP_SYSCALL_EXPORTS).find(|e| e.1 == name).map(|e| e.2 as usize);
    assert_eq!(export("scePthreadMain"), export("pthread_main_np"));
    assert_eq!(ps5nid::compute_nid("scePthreadMain"), "55aShElDfY4");
    let s = Scratch::new(0x5ED0_0000_0000);
    let mut ctx = s.ctx();
    let (main, _) = fake_thread_with("main", 0, true);
    assert_eq!(call(&mut ctx, cb_pthread_main_np, &[]), 1);
    let vm = s.vm;
    let worker = std::thread::spawn(move || {
        let mut ctx = CpuContext::new(vm);
        let (handle, _) = fake_thread("worker", 0);
        let r = call(&mut ctx, cb_pthread_main_np, &[]);
        forget_fake_thread(handle);
        r
    });
    assert_eq!(worker.join().unwrap(), 0);
    forget_fake_thread(main);
}

/// A condition wait on a recursive mutex locked twice releases both levels,
/// so another thread can take the mutex and signal, and gets both back.
#[test]
fn cond_wait_releases_every_recursive_level() {
    let s = Scratch::new(0x5EC8_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (attr, m, c, flag, ts) = (s.base, s.base + 8, s.base + 16, s.base + 24, s.base + 0x100);
    assert_eq!(call(&mut ctx, cb_mutexattr_init, &[attr]), OK);
    assert_eq!(call(&mut ctx, cb_mutexattr_settype, &[attr, MUTEX_RECURSIVE as u64]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_init, &[m, attr, 0]), OK);
    assert_eq!(call(&mut ctx, cb_cond_init, &[c, 0, 0]), OK);
    let vm = s.vm;
    for kind in 0..3 {
        ctx.write_u64(flag, 0);
        assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
        assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
        let signaller = std::thread::spawn(move || {
            let mut ctx = CpuContext::new(vm);
            bind();
            assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
            ctx.write_u64(flag, 1);
            assert_eq!(call(&mut ctx, cb_cond_signal, &[c]), OK);
            assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
        });
        while ctx.read_u64(flag) == Some(0) {
            let r = match kind {
                0 => call(&mut ctx, cb_cond_wait, &[c, m]),
                1 => call(&mut ctx, cb_sce_cond_timedwait, &[c, m, 1000]),
                _ => {
                    let at = realtime_now() + Duration::from_millis(1);
                    ctx.write_u64(ts, at.as_secs());
                    ctx.write_u64(ts + 8, at.subsec_nanos() as u64);
                    call(&mut ctx, cb_posix_cond_timedwait, &[c, m, ts])
                }
            };
            assert!(r == OK || r == ETIMEDOUT, "{r:#x}");
        }
        signaller.join().unwrap();
        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), EPERM);
    }
    assert_eq!(call(&mut ctx, cb_cond_destroy, &[c]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), OK);
}

/// Two threads taking a fresh (statically initialized) rwlock for writing
/// at the same time never both hold it.
#[test]
fn concurrent_first_wrlock_excludes() {
    let s = Scratch::new(0x5EC9_0000_0000);
    let vm = s.vm;
    for round in 0..200u64 {
        let l = s.base + round * 8;
        let inside = Arc::new(AtomicU32::new(0));
        let overlaps = Arc::new(AtomicU32::new(0));
        let ready = Arc::new(std::sync::Barrier::new(2));
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let (inside, overlaps, ready) = (inside.clone(), overlaps.clone(), ready.clone());
                std::thread::spawn(move || {
                    let mut ctx = CpuContext::new(vm);
                    bind();
                    ready.wait();
                    assert_eq!(call(&mut ctx, px_rwlock_wrlock, &[l]), 0);
                    if inside.fetch_add(1, Ordering::SeqCst) != 0 {
                        overlaps.fetch_add(1, Ordering::SeqCst);
                    }
                    for _ in 0..200 {
                        std::hint::spin_loop();
                    }
                    inside.fetch_sub(1, Ordering::SeqCst);
                    assert_eq!(call(&mut ctx, px_rwlock_unlock, &[l]), 0);
                })
            })
            .collect();
        for w in workers {
            w.join().unwrap();
        }
        assert_eq!(overlaps.load(Ordering::SeqCst), 0, "round {round}");
        let mut ctx = s.ctx();
        assert_eq!(call(&mut ctx, px_rwlock_destroy, &[l]), 0);
    }
}

const PRIO_PROTECT_FOR_TESTS: u64 = 2;

/// The priority ceiling exists only under PTHREAD_PRIO_PROTECT; a protocol
/// change resets it to the highest SCHED_RR priority.
#[test]
fn mutexattr_prioceiling_follows_protocol() {
    let s = Scratch::new(0x5ECA_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (attr, out, m) = (s.base, s.base + 0x100, s.base + 8);
    assert_eq!(call(&mut ctx, px_mutexattr_init, &[attr]), 0);
    ctx.write_u32(out, u32::MAX);
    assert_eq!(call(&mut ctx, px_mutexattr_getprioceiling, &[attr, out]), 22);
    assert_eq!(ctx.read_u32(out), Some(u32::MAX));
    assert_eq!(call(&mut ctx, px_mutexattr_setprioceiling, &[attr, 300]), 22);
    assert_eq!(call(&mut ctx, px_mutexattr_setprotocol, &[attr, 1]), 0);
    assert_eq!(call(&mut ctx, px_mutexattr_setprioceiling, &[attr, 300]), 22);
    assert_eq!(call(&mut ctx, px_mutexattr_setprotocol, &[attr, PRIO_PROTECT_FOR_TESTS]), 0);
    assert_eq!(call(&mut ctx, px_mutexattr_getprioceiling, &[attr, out]), 0);
    assert_eq!(ctx.read_u32(out), Some(256));
    assert_eq!(call(&mut ctx, px_mutexattr_setprioceiling, &[attr, 300]), 0);
    assert_eq!(call(&mut ctx, cb_mutexattr_getprioceiling, &[attr, out]), OK);
    assert_eq!(ctx.read_u32(out), Some(300));
    assert_eq!(call(&mut ctx, px_mutex_init, &[m, attr]), 0);
    assert_eq!(call(&mut ctx, px_mutex_lock, &[m]), 0);
    assert_eq!(call(&mut ctx, px_mutex_unlock, &[m]), 0);
    assert_eq!(call(&mut ctx, px_mutex_destroy, &[m]), 0);
    assert_eq!(call(&mut ctx, px_mutexattr_setprotocol, &[attr, PRIO_PROTECT_FOR_TESTS]), 0);
    assert_eq!(call(&mut ctx, px_mutexattr_getprioceiling, &[attr, out]), 0);
    assert_eq!(ctx.read_u32(out), Some(256));
    assert_eq!(call(&mut ctx, px_mutexattr_setprotocol, &[attr, 0]), 0);
    assert_eq!(call(&mut ctx, px_mutexattr_getprioceiling, &[attr, out]), 22);
    assert_eq!(call(&mut ctx, cb_mutexattr_setprioceiling, &[attr, 300]), EINVAL);
    assert_eq!(call(&mut ctx, px_mutexattr_destroy, &[attr]), 0);
    assert_eq!(call(&mut ctx, px_mutexattr_setprioceiling, &[attr, 300]), 22);
}

/// A thread gets exactly the stack size it asks for (reported back), with
/// host room below it.
#[test]
fn thread_stack_is_the_requested_size() {
    for requested in [0x4000u64, 0x1_0000, 0x10_0000, 0x100_0000, 0x800_0000] {
        let (mapped, size) = stack_layout(requested);
        assert_eq!(size, requested);
        assert!(mapped >= size + STACK_HEADROOM && mapped <= STACK_SLOT, "{requested:#x}");
        assert_eq!(mapped % 0x4000, 0);
    }
    assert_eq!(stack_layout(0x100).1, STACK_MIN);
    assert_eq!(stack_layout(u64::MAX).1, MAX_STACK_SIZE);
    assert!(stack_layout(u64::MAX).0 <= STACK_SLOT);
}

/// pthread_rename_np keeps names below 32 bytes and rejects longer ones
/// with ENAMETOOLONG, keeping the old name; getname writes exactly the name
/// and its terminator.
#[test]
fn posix_thread_names() {
    let s = Scratch::new(0x5ECB_0000_0000);
    let mut ctx = s.ctx();
    let (me, _) = fake_thread("", DEFAULT_AFFINITY);
    let (name, buf) = (s.base, s.base + 0x100);
    let check = |ctx: &mut CpuContext, expected: &str| {
        ctx.write_bytes(buf, &[0xAA; 64]);
        assert_eq!(call(ctx, px_getname, &[me, buf]), 0);
        let got = ctx.read_bytes(buf, 64).unwrap();
        assert_eq!(&got[..expected.len()], expected.as_bytes());
        assert_eq!(got[expected.len()], 0);
        assert!(got[expected.len() + 1..].iter().all(|&b| b == 0xAA));
    };
    check(&mut ctx, "");
    let longest = "0123456789012345678901234567890";
    for n in ["", "A", "probe", longest] {
        ctx.write_bytes(name, n.as_bytes());
        ctx.write_bytes(name + n.len() as u64, &[0]);
        assert_eq!(call(&mut ctx, px_rename, &[me, name]), 0);
        check(&mut ctx, n);
    }
    for len in [32usize, 40, 200] {
        ctx.write_bytes(name, &vec![b'x'; len]);
        ctx.write_bytes(name + len as u64, &[0]);
        assert_eq!(call(&mut ctx, px_rename, &[me, name]), POSIX_ENAMETOOLONG);
        check(&mut ctx, longest);
    }
    ctx.write_bytes(buf, &[0xAA; 64]);
    assert_eq!(call(&mut ctx, px_getname, &[0, buf]), 3); // ESRCH
    assert!(ctx.read_bytes(buf, 64).unwrap().iter().all(|&b| b == 0xAA));
    assert_eq!(call(&mut ctx, px_getname, &[me, 0]), 14); // EFAULT
    forget_fake_thread(me);
    assert_eq!(call(&mut ctx, px_getname, &[me, buf]), 3);
}

/// pthread_getaffinity_np reports the mask as a 32-byte cpuset_t (rest
/// zero, nothing past it written), with FreeBSD 11's errors (reference
/// GuestAvailableCpumask cases).
#[test]
fn getaffinity_np_writes_a_cpuset() {
    let _g = AFFINITY_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    let s = Scratch::new(0x7C51_0000_0000);
    let mut ctx = s.ctx();
    let (me, _) = fake_thread("cpuset", 0x0F0);
    let set = s.base;
    ctx.write_bytes(set, &[0xAB; 40]);
    assert_eq!(call(&mut ctx, px_getaffinity, &[me, 32, set]), 0);
    assert_eq!(ctx.read_u64(set), Some(0x0F0));
    assert_eq!(ctx.read_bytes(set + 8, 24).unwrap(), vec![0; 24]);
    assert_eq!(ctx.read_bytes(set + 32, 8).unwrap(), vec![0xAB; 8], "past the set");
    for size in [0u64, 8, 31, 33, 64] {
        assert_eq!(call(&mut ctx, px_getaffinity, &[me, size, set]), 34, "size {size}");
    }
    assert_eq!(call(&mut ctx, px_getaffinity, &[0, 32, set]), 22);
    assert_eq!(call(&mut ctx, px_getaffinity, &[0, 0, set]), 22);
    assert_eq!(call(&mut ctx, px_getaffinity, &[me, 32, 0]), 14);
    forget_fake_thread(me);
    assert_eq!(call(&mut ctx, px_getaffinity, &[me, 32, set]), 3);
}

/// A new thread inherits its creator's affinity unless its attributes set
/// one; a thread no guest thread creates takes the process cpuset
/// (reference GuestAvailableCpumask inheritance case).
#[test]
fn new_threads_inherit_affinity_unless_the_attributes_set_one() {
    let _g = AFFINITY_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    let s = Scratch::new(0x7C52_0000_0000);
    let mut ctx = s.ctx();
    let attr = s.base;
    assert_eq!(call(&mut ctx, cb_attr_init, &[attr]), OK);
    let plain = read_attr(&ctx, attr).unwrap();
    assert!(!plain.affinity_explicit);
    // Created by no guest thread: the process cpuset.
    let process = std::thread::spawn(move || inherited_affinity(&plain)).join().unwrap();
    assert_eq!(process, PROCESS_AFFINITY.load(Ordering::Relaxed));
    std::thread::spawn(move || {
        let s = Scratch::new(0x7C53_0000_0000);
        let mut ctx = s.ctx();
        let attr = s.base;
        assert_eq!(call(&mut ctx, cb_attr_init, &[attr]), OK);
        let (me, _) = fake_thread("creator", 0x30);
        assert_eq!(inherited_affinity(&read_attr(&ctx, attr).unwrap()), 0x30, "the creator's affinity");
        assert_eq!(call(&mut ctx, cb_attr_setaffinity, &[attr, 0x3]), OK);
        let explicit = read_attr(&ctx, attr).unwrap();
        assert!(explicit.affinity_explicit);
        assert_eq!(inherited_affinity(&explicit), 0x3, "the attributes' affinity");
        forget_fake_thread(me);
    })
    .join()
    .unwrap();
}

/// pthread_set_name_np renames; a NULL name clears the name and a NULL
/// thread is ignored (reference checks); a long name keeps 31 bytes.
#[test]
fn set_name_np_renames_and_clears() {
    let s = Scratch::new(0x7C47_0000_0000);
    let mut ctx = s.ctx();
    let (me, _) = fake_thread("before", DEFAULT_AFFINITY);
    let (name, buf) = (s.base, s.base + 0x100);
    let get = |ctx: &mut CpuContext| -> Vec<u8> {
        assert_eq!(call(ctx, px_getname, &[me, buf]), 0);
        ctx.read_cstr_bytes(buf, 64).unwrap()
    };
    ctx.write_bytes(name, b"q2-main\0");
    call(&mut ctx, px_set_name_np, &[me, name]);
    assert_eq!(get(&mut ctx), b"q2-main");
    call(&mut ctx, px_set_name_np, &[me, 0]);
    assert_eq!(get(&mut ctx), b"");
    ctx.write_bytes(name, b"ignored\0");
    call(&mut ctx, px_set_name_np, &[0, name]);
    assert_eq!(get(&mut ctx), b"");
    ctx.write_bytes(name, &[vec![b'n'; 40], vec![0]].concat());
    call(&mut ctx, px_set_name_np, &[me, name]);
    assert_eq!(get(&mut ctx), vec![b'n'; 31]);
    forget_fake_thread(me);
}

/// pthread_getcpuclockid hands out CPUCLOCK_BIT | tid; the clock follows the
/// thread's CPU time and goes away with the thread.
#[test]
fn thread_cpu_clocks() {
    let s = Scratch::new(0x5ECC_0000_0000);
    let mut ctx = s.ctx();
    let (me, id) = fake_thread("clock", DEFAULT_AFFINITY);
    let out = s.base;
    assert_eq!(call(&mut ctx, px_getcpuclockid, &[0, out]), 22);
    assert_eq!(call(&mut ctx, px_getcpuclockid, &[me, 0]), 14);
    assert_eq!(call(&mut ctx, px_getcpuclockid, &[me, out]), 0);
    let clock = ctx.read_u32(out).unwrap();
    assert_eq!(clock & 0xC000_0000, CPUCLOCK_BIT);
    assert_eq!(clock & CPUCLOCK_ID_MASK, id);
    let start = crate::time::clock_now(clock as i32).unwrap();
    let t = Instant::now();
    while crate::time::clock_now(clock as i32).unwrap() - start < Duration::from_millis(30) {
        assert!(t.elapsed() < Duration::from_secs(10));
    }
    let own = crate::time::clock_now(crate::time::CLOCK_THREAD_CPUTIME_ID).unwrap();
    assert!(own + Duration::from_millis(20) >= crate::time::clock_now(clock as i32).unwrap());
    // Read from another thread it is still this thread's time.
    let before = crate::time::clock_now(clock as i32).unwrap();
    let other = std::thread::spawn(move || crate::time::clock_now(clock as i32)).join().unwrap().unwrap();
    assert!(other >= before && other < before + Duration::from_millis(20));
    forget_fake_thread(me);
    assert!(crate::time::clock_now(clock as i32).is_none());
    assert_eq!(call(&mut ctx, px_getcpuclockid, &[me, out]), 3);
}

/// The current CPU stays within the calling thread's affinity.
#[test]
fn current_affinity_is_the_thread_affinity() {
    let _g = AFFINITY_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    let (me, id) = fake_thread("affinity", 0x30);
    assert_eq!(current_affinity(), 0x30);
    assert_eq!(affinity_of_thread_id(id as i64), Some(0x30));
    assert_eq!(affinity_of_thread_id(-5), None);
    for host in 0..64 {
        let cpu = crate::kernel_ext::guest_cpu_from_host(host, current_affinity());
        assert!(cpu == 4 || cpu == 5, "{cpu}");
    }
    forget_fake_thread(me);
}

/// A thread holding `mutex` until `release` is set.
fn hold_mutex(vm: &'static VirtualMemory, m: u64, held: Arc<AtomicU32>, release: Arc<AtomicU32>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut ctx = CpuContext::new(vm);
        bind();
        assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
        held.store(1, Ordering::SeqCst);
        while release.load(Ordering::SeqCst) == 0 {
            std::thread::yield_now();
        }
        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
    })
}

/// Every condition wait (sce and POSIX, untimed and timed) refuses a mutex
/// the caller does not hold: unlocked, never initialised, the adaptive
/// static initializer, or held by another thread. The sce forms return
/// EPERM as a kernel code, the POSIX forms the errno value 1.
#[test]
fn cond_wait_with_a_mutex_the_caller_does_not_hold() {
    let s = Scratch::new(0x6A1C_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (c, m, ts) = (s.base, s.base + 0x10, s.base + 0x100);
    let deadline = crate::time::clock_now(crate::time::CLOCK_REALTIME).unwrap() + Duration::from_secs(60);
    ctx.write_u64(ts, deadline.as_secs());
    ctx.write_u64(ts + 8, deadline.subsec_nanos() as u64);
    let waits: [(crate::registry::HleCallback, u64, HleResult); 4] = [
        (cb_cond_wait, 0, EPERM),
        (cb_sce_cond_timedwait, 60_000_000, EPERM),
        (px_cond_wait, 0, 1),
        (px_cond_timedwait, ts, 1),
    ];
    for (wait, third, not_owned) in waits {
        assert_eq!(call(&mut ctx, cb_cond_init, &[c, 0, 0]), OK);
        assert_eq!(call(&mut ctx, cb_mutex_init, &[m, 0, 0]), OK);
        assert_eq!(call(&mut ctx, wait, &[c, m, third]), not_owned, "unlocked");
        assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
        assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), OK);
        ctx.write_u64(m, 0);
        assert_eq!(call(&mut ctx, wait, &[c, m, third]), not_owned, "uninitialised");
        ctx.write_u64(m, MUTEX_STATIC_ADAPTIVE_INITIALIZER);
        assert_eq!(call(&mut ctx, wait, &[c, m, third]), not_owned, "adaptive initializer");
        assert_eq!(call(&mut ctx, cb_mutex_init, &[m, 0, 0]), OK);
        let (held, release) = (Arc::new(AtomicU32::new(0)), Arc::new(AtomicU32::new(0)));
        let holder = hold_mutex(s.vm, m, held.clone(), release.clone());
        while held.load(Ordering::SeqCst) == 0 {
            std::thread::yield_now();
        }
        assert_eq!(call(&mut ctx, wait, &[c, m, third]), not_owned, "held by another thread");
        release.store(1, Ordering::SeqCst);
        holder.join().unwrap();
        assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), OK);
        assert_eq!(call(&mut ctx, cb_cond_destroy, &[c]), OK);
    }
}

/// Destroying a rwlock, as measured on a PS5 (firmware 13.60): 0 whether it
/// is free or held for reading or writing, leaving 1 in the handle; every
/// call on that handle is then EINVAL until it is initialised again, and a
/// zero handle is 0.
#[test]
fn rwlock_destroy_answers_as_the_console() {
    let s = Scratch::new(0x6A1D_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let l = s.base;
    assert_eq!(call(&mut ctx, cb_rwlock_destroy, &[l]), OK, "zero handle");
    for write in [false, true] {
        let lock = if write { px_rwlock_wrlock } else { px_rwlock_rdlock };
        let again = if write { px_rwlock_trywrlock } else { px_rwlock_tryrdlock };
        ctx.write_u64(l, 0);
        assert_eq!(call(&mut ctx, lock, &[l]), 0);
        assert_eq!(call(&mut ctx, px_rwlock_destroy, &[l]), 0, "held");
        assert_eq!(ctx.read_u64(l), Some(RWLOCK_DESTROYED));
        assert_eq!(call(&mut ctx, px_rwlock_unlock, &[l]), 22);
        assert_eq!(call(&mut ctx, again, &[l]), 22);
        assert_eq!(call(&mut ctx, cb_rwlock_rdlock, &[l]), EINVAL);
        assert_eq!(call(&mut ctx, cb_rwlock_timedwrlock, &[l, 0]), EINVAL);
        assert_eq!(call(&mut ctx, cb_rwlock_destroy, &[l]), EINVAL);
        assert_eq!(call(&mut ctx, px_rwlock_destroy, &[l]), 22);
        assert_eq!(call(&mut ctx, cb_rwlock_init, &[l, 0, 0]), OK);
        assert_eq!(call(&mut ctx, again, &[l]), 0);
        assert_eq!(call(&mut ctx, px_rwlock_unlock, &[l]), 0);
        assert_eq!(call(&mut ctx, px_rwlock_destroy, &[l]), 0, "free");
        assert_eq!(ctx.read_u64(l), Some(RWLOCK_DESTROYED));
    }
}

/// The POSIX timed locks take a free lock whatever the deadline says and
/// wait without limit for a null one; a held lock with an invalid deadline
/// is EINVAL.
#[test]
fn posix_timed_locks_try_first_and_take_null_deadlines() {
    let s = Scratch::new(0x6A1F_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (m, l, ts) = (s.base, s.base + 8, s.base + 0x100);
    ctx.write_u64(ts, 0);
    ctx.write_u64(ts + 8, 1_000_000_000);
    assert_eq!(call(&mut ctx, px_mutex_init, &[m, 0]), 0);
    assert_eq!(call(&mut ctx, px_mutex_timedlock, &[m, ts]), 0, "free mutex, invalid deadline");
    assert_eq!(call(&mut ctx, px_mutex_timedlock, &[m, ts]), 22, "held mutex, invalid deadline");
    assert_eq!(call(&mut ctx, px_mutex_unlock, &[m]), 0);
    assert_eq!(call(&mut ctx, px_mutex_timedlock, &[m, 0]), 0, "null deadline");
    assert_eq!(call(&mut ctx, px_mutex_unlock, &[m]), 0);
    assert_eq!(call(&mut ctx, px_rwlock_timedwrlock, &[l, ts]), 0);
    assert_eq!(call(&mut ctx, px_rwlock_unlock, &[l]), 0);
    assert_eq!(call(&mut ctx, px_rwlock_timedrdlock, &[l, ts]), 0);
    assert_eq!(call(&mut ctx, px_rwlock_timedrdlock, &[l, 0]), 0);
    assert_eq!(call(&mut ctx, px_rwlock_unlock, &[l]), 0);
    assert_eq!(call(&mut ctx, px_rwlock_unlock, &[l]), 0);
    let vm = s.vm;
    let other = std::thread::spawn(move || {
        let mut ctx = CpuContext::new(vm);
        bind();
        assert_eq!(call(&mut ctx, px_rwlock_wrlock, &[l]), 0);
        assert_eq!(call(&mut ctx, px_mutex_lock, &[m]), 0);
    });
    other.join().unwrap();
    assert_eq!(call(&mut ctx, px_rwlock_timedrdlock, &[l, ts]), 22, "held rwlock, invalid deadline");
    assert_eq!(call(&mut ctx, px_mutex_timedlock, &[m, ts]), 22);
}

/// pthread_attr_setsolosched_np: EINVAL for a null or uninitialised
/// attribute and after destroy; otherwise the value scePthreadAttrGetsolosched
/// reads back.
#[test]
fn posix_attr_setsolosched_np() {
    let s = Scratch::new(0x6A1E_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (attr, out) = (s.base, s.base + 0x100);
    assert_eq!(call(&mut ctx, px_attr_setsolosched, &[0, 1]), 22);
    ctx.write_u64(attr, 0);
    assert_eq!(call(&mut ctx, px_attr_setsolosched, &[attr, 1]), 22);
    assert_eq!(call(&mut ctx, cb_attr_init, &[attr]), OK);
    ctx.write_u32(out, u32::MAX);
    assert_eq!(call(&mut ctx, cb_attr_getsolosched, &[attr, out]), OK);
    assert_eq!(ctx.read_u32(out), Some(0));
    assert_eq!(call(&mut ctx, px_attr_setsolosched, &[attr, 1]), 45, "ENOTSUP");
    assert_eq!(call(&mut ctx, px_attr_setsolosched, &[attr, 16]), 0);
    assert_eq!(call(&mut ctx, cb_attr_getsolosched, &[attr, out]), OK);
    assert_eq!(ctx.read_u32(out), Some(16));
    assert_eq!(call(&mut ctx, px_attr_setsolosched, &[attr, 0]), 0);
    assert_eq!(call(&mut ctx, cb_attr_getsolosched, &[attr, out]), OK);
    assert_eq!(ctx.read_u32(out), Some(0));
    assert_eq!(call(&mut ctx, cb_attr_destroy, &[attr]), OK);
    assert_eq!(call(&mut ctx, px_attr_setsolosched, &[attr, 1]), 22);
}

// ---------------------------------------------------------------------------
// Lock word, spinning and missed wakeups under load
// ---------------------------------------------------------------------------

/// The mutex implementations: host-lock state, lock word, lock word with a
/// 4 µs adaptive spin.
const MUTEX_MODES: [(bool, u64); 3] = [(false, 0), (true, 0), (true, 4_000)];

/// Create a mutex of `kind` at `m` (attribute object at `m + 8`) in `mode`.
fn make_mutex(ctx: &mut CpuContext, m: u64, kind: i32, mode: (bool, u64)) {
    let attr = m + 8;
    assert_eq!(call(ctx, cb_mutexattr_init, &[attr]), OK);
    assert_eq!(call(ctx, cb_mutexattr_settype, &[attr, kind as u64]), OK);
    TEST_MUTEX_MODE.with(|t| t.set(Some(mode)));
    assert_eq!(call(ctx, cb_mutex_init, &[m, attr, 0]), OK);
    TEST_MUTEX_MODE.with(|t| t.set(None));
    assert_eq!(call(ctx, cb_mutexattr_destroy, &[attr]), OK);
}

/// Waits of the calling thread wake only when notified (or after a minute),
/// so a missed wakeup makes the test slow instead of hiding in the 10 ms
/// slice.
fn long_slices() {
    crate::signals::TEST_SLICE.with(|s| s.set(Some(Duration::from_secs(60))));
}

/// Threads hammering one mutex (lock, timed lock, trylock; recursive ones
/// nest) never overlap inside it, every increment lands, and no sleeper
/// misses its wakeup.
#[test]
fn mutex_stress_excludes_and_wakes() {
    let s = Scratch::new(0x6B40_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let mut slot = 0;
    for (mode, handoff) in MUTEX_MODES.into_iter().flat_map(|m| [(m, false), (m, true)]) {
        for kind in [MUTEX_ERRORCHECK, MUTEX_RECURSIVE, 3, MUTEX_ADAPTIVE] {
            let m = s.base + slot * 0x40;
            slot += 1;
            TEST_MUTEX_HANDOFF.with(|t| t.set(Some(handoff)));
            make_mutex(&mut ctx, m, kind, mode);
            TEST_MUTEX_HANDOFF.with(|t| t.set(None));
            let inside = Arc::new(AtomicU32::new(0));
            let total = Arc::new(std::sync::atomic::AtomicU64::new(0));
            let t0 = Instant::now();
            let threads: Vec<_> = (0..8u64)
                .map(|t| {
                    let (vm, inside, total) = (s.vm, inside.clone(), total.clone());
                    std::thread::spawn(move || {
                        let mut ctx = CpuContext::new(vm);
                        bind();
                        long_slices();
                        let mut done = 0u64;
                        let mut i = 0u64;
                        while done < 5_000 {
                            i += 1;
                            let r = match (i + t) % 4 {
                                0 => call(&mut ctx, cb_mutex_trylock, &[m]),
                                1 => call(&mut ctx, cb_mutex_timedlock_usec, &[m, 50]),
                                _ => call(&mut ctx, cb_mutex_lock, &[m]),
                            };
                            if r == EBUSY || r == ETIMEDOUT {
                                continue;
                            }
                            assert_eq!(r, OK);
                            assert_eq!(inside.fetch_add(1, Ordering::SeqCst), 0, "two owners at once");
                            if kind == MUTEX_RECURSIVE {
                                assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
                                assert_eq!(call(&mut ctx, cb_mutex_trylock, &[m]), OK);
                                assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
                                assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
                            } else if kind == MUTEX_ERRORCHECK {
                                assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), EDEADLK);
                                assert_eq!(call(&mut ctx, cb_mutex_trylock, &[m]), EBUSY);
                            }
                            // A plain read-modify-write: lost updates show
                            // a broken exclusion.
                            let v = total.load(Ordering::Relaxed);
                            std::hint::spin_loop();
                            total.store(v + 1, Ordering::Relaxed);
                            inside.fetch_sub(1, Ordering::SeqCst);
                            assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
                            done += 1;
                        }
                        // Not the owner any more.
                        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), EPERM);
                    })
                })
                .collect();
            for t in threads {
                t.join().unwrap();
            }
            assert_eq!(total.load(Ordering::SeqCst), 8 * 5_000, "mode {mode:?} handoff {handoff} kind {kind}");
            assert!(t0.elapsed() < Duration::from_secs(30), "mode {mode:?} kind {kind}: a sleeper missed its wakeup");
            assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), OK);
        }
    }
}

/// Producers and consumers through one mutex and condition: signal and
/// broadcast lose no wakeup, every item is consumed once, and recursion
/// survives the waits.
#[test]
fn cond_stress_loses_no_wakeup() {
    let s = Scratch::new(0x6B41_0000_0000);
    let mut ctx = s.ctx();
    bind();
    for (n, mode) in MUTEX_MODES.into_iter().enumerate() {
        let base = s.base + n as u64 * 0x100;
        let (m, c, items) = (base, base + 0x40, base + 0x80);
        make_mutex(&mut ctx, m, if n == 2 { MUTEX_ADAPTIVE } else { MUTEX_RECURSIVE }, mode);
        assert_eq!(call(&mut ctx, cb_cond_init, &[c, 0, 0]), OK);
        ctx.write_u64(items, 0);
        const PER_PRODUCER: u64 = 4_000;
        let consumed = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let t0 = Instant::now();
        let vm = s.vm;
        let consumers: Vec<_> = (0..4)
            .map(|_| {
                let consumed = consumed.clone();
                std::thread::spawn(move || {
                    let mut ctx = CpuContext::new(vm);
                    bind();
                    long_slices();
                    loop {
                        assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
                        while ctx.read_u64(items).unwrap() == 0 {
                            if consumed.load(Ordering::SeqCst) == 2 * PER_PRODUCER {
                                assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
                                return;
                            }
                            assert_eq!(call(&mut ctx, cb_cond_wait, &[c, m]), OK);
                        }
                        ctx.write_u64(items, ctx.read_u64(items).unwrap() - 1);
                        consumed.fetch_add(1, Ordering::SeqCst);
                        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
                    }
                })
            })
            .collect();
        let producers: Vec<_> = (0..2u64)
            .map(|p| {
                std::thread::spawn(move || {
                    let mut ctx = CpuContext::new(vm);
                    bind();
                    for i in 0..PER_PRODUCER {
                        assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
                        ctx.write_u64(items, ctx.read_u64(items).unwrap() + 1);
                        let wake: crate::registry::HleCallback = if (i + p) % 3 == 0 { cb_cond_broadcast } else { cb_cond_signal };
                        assert_eq!(call(&mut ctx, wake, &[c]), OK);
                        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
                    }
                })
            })
            .collect();
        for p in producers {
            p.join().unwrap();
        }
        // Release the consumers once everything is taken.
        while consumed.load(Ordering::SeqCst) < 2 * PER_PRODUCER {
            assert!(t0.elapsed() < Duration::from_secs(30), "mode {mode:?}: a consumer missed its wakeup");
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
        assert_eq!(call(&mut ctx, cb_cond_broadcast, &[c]), OK);
        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
        for t in consumers {
            t.join().unwrap();
        }
        assert_eq!(ctx.read_u64(items), Some(0));
        assert!(t0.elapsed() < Duration::from_secs(30), "mode {mode:?}: a consumer missed its wakeup");
    }
}

/// A timed lock that gives up leaves the mutex consistent: the next unlock
/// still wakes the remaining sleeper.
#[test]
fn timed_out_lock_keeps_wakeups() {
    let s = Scratch::new(0x6B42_0000_0000);
    let mut ctx = s.ctx();
    bind();
    for (n, mode) in MUTEX_MODES.into_iter().enumerate() {
        let m = s.base + n as u64 * 0x40;
        make_mutex(&mut ctx, m, MUTEX_ADAPTIVE, mode);
        assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
        let vm = s.vm;
        let timed = std::thread::spawn(move || {
            let mut ctx = CpuContext::new(vm);
            bind();
            long_slices();
            call(&mut ctx, cb_mutex_timedlock_usec, &[m, 20_000])
        });
        let blocked = std::thread::spawn(move || {
            let mut ctx = CpuContext::new(vm);
            bind();
            long_slices();
            assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
            assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
        });
        assert_eq!(timed.join().unwrap(), ETIMEDOUT);
        std::thread::sleep(Duration::from_millis(5));
        let t = Instant::now();
        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
        blocked.join().unwrap();
        assert!(t.elapsed() < Duration::from_secs(5), "mode {mode:?}: the sleeper was not woken");
        assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), OK);
    }
}

/// A host thread without a guest handle owns a lock-word mutex under its
/// own id: another such thread cannot unlock it.
#[test]
fn unbound_threads_have_distinct_owners() {
    let s = Scratch::new(0x6B43_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let m = s.base;
    make_mutex(&mut ctx, m, MUTEX_ERRORCHECK, (true, 0));
    let vm = s.vm;
    std::thread::spawn(move || {
        let mut ctx = CpuContext::new(vm);
        assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
        std::thread::spawn(move || {
            let mut ctx = CpuContext::new(vm);
            assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), EPERM);
            assert_eq!(call(&mut ctx, cb_mutex_trylock, &[m]), EBUSY);
        })
        .join()
        .unwrap();
        assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), EDEADLK);
        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
    })
    .join()
    .unwrap();
    assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), OK);
}

/// Sleepers own a handoff mutex in arrival order: the unlock reserves it
/// for the first one (the unlocker's own trylock is then EBUSY), a timed
/// sleeper that gives up leaves the queue from the head, middle or tail,
/// and a recursive owner keeps the mutex until its last unlock.
#[test]
fn mutex_hands_off_to_sleepers_in_fifo_order() {
    let s = Scratch::new(0x6B45_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let vm = s.vm;
    let mut slot = 0;
    for mode in MUTEX_MODES {
        for timed in 0..3usize {
            let m = s.base + slot * 0x40;
            slot += 1;
            TEST_MUTEX_HANDOFF.with(|t| t.set(Some(true)));
            make_mutex(&mut ctx, m, MUTEX_RECURSIVE, mode);
            TEST_MUTEX_HANDOFF.with(|t| t.set(None));
            assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
            assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
            let host = resolve_mutex(&ctx, m).ok().unwrap();
            let order = Arc::new(std::sync::Mutex::new(Vec::new()));
            // Holds a waiter that got the mutex until the check below that
            // the unlocker could not take it back: on a loaded machine the
            // waiters could otherwise take, release and pass it on before
            // the unlocker's trylock ran.
            let gate = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
            let mut waiters = Vec::new();
            for k in 0..3usize {
                let order = order.clone();
                let gate = gate.clone();
                let queued = lock_state(host).queue.len();
                waiters.push(std::thread::spawn(move || {
                    let mut ctx = CpuContext::new(vm);
                    bind();
                    long_slices();
                    let r = if k == timed { call(&mut ctx, cb_mutex_timedlock_usec, &[m, 100_000]) } else { call(&mut ctx, cb_mutex_lock, &[m]) };
                    if r == OK {
                        order.lock().unwrap().push(k);
                        let mut open = gate.0.lock().unwrap();
                        while !*open {
                            open = gate.1.wait(open).unwrap();
                        }
                        drop(open);
                        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
                    }
                    r
                }));
                // Each sleeper is queued before the next one starts.
                let t0 = Instant::now();
                while lock_state(host).queue.len() == queued {
                    assert!(t0.elapsed() < Duration::from_secs(5), "waiter {k} never queued");
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            let timed_out = waiters.remove(timed);
            assert_eq!(timed_out.join().unwrap(), ETIMEDOUT);
            assert_eq!(lock_state(host).queue.len(), 2, "the timed sleeper left the queue");
            // The first unlock of the recursive owner keeps the mutex.
            assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
            assert_eq!(call(&mut ctx, cb_mutex_trylock, &[m]), OK);
            assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
            // The last one hands it over before anybody runs.
            assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
            assert_eq!(call(&mut ctx, cb_mutex_trylock, &[m]), EBUSY, "mode {mode:?}: the unlocker took the mutex back");
            *gate.0.lock().unwrap() = true;
            gate.1.notify_all();
            for w in waiters {
                assert_eq!(w.join().unwrap(), OK);
            }
            let expected: Vec<usize> = (0..3).filter(|&k| k != timed).collect();
            assert_eq!(*order.lock().unwrap(), expected, "mode {mode:?} timed {timed}");
            assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
            assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
            assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), OK);
        }
    }
}

fn lock_state(m: &HostMutex) -> std::sync::MutexGuard<'_, MutexState> {
    m.state.lock().unwrap_or_else(|e| e.into_inner())
}

/// Contended lock throughput: `threads` threads each take the mutex
/// `iters` times around a short critical section. Returns ns per lock.
fn mutex_handoff_ns(s: &Scratch, m: u64, threads: usize, iters: u64) -> f64 {
    let vm = s.vm;
    let shared = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let start = Arc::new(std::sync::Barrier::new(threads + 1));
    let hs: Vec<_> = (0..threads)
        .map(|_| {
            let (shared, start) = (shared.clone(), start.clone());
            std::thread::spawn(move || {
                let mut ctx = CpuContext::new(vm);
                bind();
                start.wait();
                for _ in 0..iters {
                    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
                    // Critical section and outside work of similar length,
                    // so the mutex changes hands most of the time.
                    for _ in 0..40 {
                        let v = shared.load(Ordering::Relaxed);
                        shared.store(v + 1, Ordering::Relaxed);
                        std::hint::spin_loop();
                    }
                    assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
                    for _ in 0..40 {
                        std::hint::spin_loop();
                    }
                }
            })
        })
        .collect();
    start.wait();
    let t = Instant::now();
    for h in hs {
        h.join().unwrap();
    }
    let ns = t.elapsed().as_nanos() as f64 / (threads as u64 * iters) as f64;
    assert_eq!(shared.load(Ordering::SeqCst), threads as u64 * iters * 40);
    ns
}

/// Median delay from the unlock to a blocked locker owning the mutex, when
/// the holder keeps it `hold` long.
fn mutex_wake_latency_ns(s: &Scratch, m: u64, hold: Duration, rounds: usize) -> u64 {
    let vm = s.vm;
    let base = Instant::now();
    let go = Arc::new(AtomicU32::new(0));
    let got = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let (g2, got2) = (go.clone(), got.clone());
    let locker = std::thread::spawn(move || {
        let mut ctx = CpuContext::new(vm);
        bind();
        for r in 1..=rounds as u32 {
            while g2.load(Ordering::Acquire) != r {
                std::hint::spin_loop();
            }
            assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
            got2.store(base.elapsed().as_nanos() as u64, Ordering::Release);
            assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
        }
    });
    let mut ctx = CpuContext::new(vm);
    let mut lat = Vec::with_capacity(rounds);
    for r in 1..=rounds as u32 {
        got.store(0, Ordering::Relaxed);
        assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
        go.store(r, Ordering::Release);
        let t = Instant::now();
        while t.elapsed() < hold {
            std::hint::spin_loop();
        }
        let t0 = base.elapsed().as_nanos() as u64;
        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
        let t1 = loop {
            let v = got.load(Ordering::Acquire);
            if v != 0 {
                break v;
            }
            std::hint::spin_loop();
        };
        lat.push(t1.saturating_sub(t0));
    }
    locker.join().unwrap();
    lat.sort_unstable();
    lat[lat.len() / 2]
}

/// Contended adaptive-mutex throughput and wake latency per implementation.
#[test]
#[ignore]
fn bench_mutex_handoff() {
    let s = Scratch::new(0x6B44_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let names = ["host lock", "lock word", "lock word + spin 4us"];
    for threads in [2, 4, 8] {
        for (i, mode) in MUTEX_MODES.into_iter().enumerate() {
            let m = s.base + (i as u64) * 0x40;
            make_mutex(&mut ctx, m, MUTEX_ADAPTIVE, mode);
            let ns = mutex_handoff_ns(&s, m, threads, 100_000);
            eprintln!("adaptive mutex, {threads} threads, {:<22}: {ns:7.1} ns/lock", names[i]);
            assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), OK);
        }
    }
    for hold_us in [1, 3, 50] {
        for (i, mode) in MUTEX_MODES.into_iter().enumerate() {
            let m = s.base + 0x800 + (i as u64) * 0x40;
            make_mutex(&mut ctx, m, MUTEX_ADAPTIVE, mode);
            let ns = mutex_wake_latency_ns(&s, m, Duration::from_micros(hold_us), 2_000);
            eprintln!("unlock -> blocked locker owns it, held {hold_us:2} us, {:<22}: {ns:6} ns median", names[i]);
            assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), OK);
        }
    }
    for (i, mode) in MUTEX_MODES.into_iter().enumerate() {
        let m = s.base + 0x400 + (i as u64) * 0x40;
        make_mutex(&mut ctx, m, MUTEX_ADAPTIVE, mode);
        let n = 2_000_000;
        let t = Instant::now();
        for _ in 0..n {
            call(&mut ctx, cb_mutex_lock, &[m]);
            call(&mut ctx, cb_mutex_unlock, &[m]);
        }
        eprintln!("uncontended lock+unlock, {:<22}: {:.1} ns", names[i], t.elapsed().as_nanos() as f64 / n as f64);
    }
}

// ---------------------------------------------------------------------------
// Console answers of the attribute and scheduling calls (PS5 firmware 13.60)
// ---------------------------------------------------------------------------

/// The scePthreadAttr setters and getters answer as the console: a null or
/// uninitialised attribute is EINVAL, a refused value EOPNOTSUPP (policy
/// other than 1 and 3, inheritance other than 0 and 4, priority outside
/// 256..=767 or a null param, solo scheduling other than 0 and 16), a
/// detach state other than 0 and 1 and a stack below 16 KiB EINVAL, any
/// affinity mask is taken.
#[test]
fn attr_calls_return_the_console_errors() {
    let s = Scratch::new(0x6A20_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (attr, param, out, unset) = (s.base, s.base + 0x100, s.base + 0x200, s.base + 0x300);
    assert_eq!(call(&mut ctx, cb_attr_init, &[attr]), OK);
    for size in [0u64, 1, 16383] {
        assert_eq!(call(&mut ctx, cb_attr_setstacksize, &[attr, size]), EINVAL);
    }
    assert_eq!(call(&mut ctx, cb_attr_setstacksize, &[attr, 16384]), OK);
    for state in [-1i64, 2, 3, 100] {
        assert_eq!(call(&mut ctx, cb_attr_setdetachstate, &[attr, state as u64]), EINVAL);
    }
    for policy in [-1i64, 0, 2, 4] {
        assert_eq!(call(&mut ctx, cb_attr_setschedpolicy, &[attr, policy as u64]), EOPNOTSUPP);
    }
    for policy in [1u64, 3] {
        assert_eq!(call(&mut ctx, cb_attr_setschedpolicy, &[attr, policy]), OK);
    }
    for inherit in [-1i64, 1, 2, 8] {
        assert_eq!(call(&mut ctx, cb_attr_setinheritsched, &[attr, inherit as u64]), EOPNOTSUPP);
    }
    for inherit in [0u64, 4] {
        assert_eq!(call(&mut ctx, cb_attr_setinheritsched, &[attr, inherit]), OK);
    }
    for priority in [-1i32, 0, 255, 768, 900, 960] {
        ctx.write_u32(param, priority as u32);
        assert_eq!(call(&mut ctx, cb_attr_setschedparam, &[attr, param]), EOPNOTSUPP, "priority {priority}");
    }
    assert_eq!(call(&mut ctx, cb_attr_setschedparam, &[attr, 0]), EOPNOTSUPP);
    for priority in [256u32, 767] {
        ctx.write_u32(param, priority);
        assert_eq!(call(&mut ctx, cb_attr_setschedparam, &[attr, param]), OK);
        assert_eq!(call(&mut ctx, cb_attr_getschedparam, &[attr, out]), OK);
        assert_eq!(ctx.read_u32(out), Some(priority));
    }
    for solo in [1u64, 2, 15, 17] {
        assert_eq!(call(&mut ctx, cb_attr_setsolosched, &[attr, solo]), EOPNOTSUPP);
    }
    for solo in [0u64, 16] {
        assert_eq!(call(&mut ctx, cb_attr_setsolosched, &[attr, solo]), OK);
    }
    for mask in [0u64, 1, u64::MAX] {
        assert_eq!(call(&mut ctx, cb_attr_setaffinity, &[attr, mask]), OK);
    }
    assert_eq!(call(&mut ctx, cb_attr_getstacksize, &[attr, 0]), EINVAL);
    assert_eq!(call(&mut ctx, cb_attr_getdetachstate, &[attr, 0]), EINVAL);
    assert_eq!(call(&mut ctx, cb_attr_getschedparam, &[attr, 0]), EINVAL);
    // A null or uninitialised attribute is EINVAL before the value.
    ctx.write_u64(unset, 0);
    ctx.write_u32(param, 700);
    for target in [0, unset] {
        assert_eq!(call(&mut ctx, cb_attr_setstacksize, &[target, 16384]), EINVAL);
        assert_eq!(call(&mut ctx, cb_attr_setdetachstate, &[target, 0]), EINVAL);
        assert_eq!(call(&mut ctx, cb_attr_setschedpolicy, &[target, 9]), EINVAL);
        assert_eq!(call(&mut ctx, cb_attr_setinheritsched, &[target, 9]), EINVAL);
        assert_eq!(call(&mut ctx, cb_attr_setschedparam, &[target, param]), EINVAL);
        assert_eq!(call(&mut ctx, cb_attr_setsolosched, &[target, 9]), EINVAL);
        assert_eq!(call(&mut ctx, cb_attr_setaffinity, &[target, 1]), EINVAL);
        assert_eq!(call(&mut ctx, cb_attr_getstacksize, &[target, out]), EINVAL);
        assert_eq!(call(&mut ctx, cb_attr_getschedparam, &[target, out]), EINVAL);
        assert_eq!(call(&mut ctx, cb_attr_getaffinity, &[target, out]), EINVAL);
    }
    assert_eq!(call(&mut ctx, cb_attr_destroy, &[attr]), OK);
}

/// The POSIX attribute setters follow FreeBSD's libthr: inheritance other
/// than 0 and 4 and a policy outside 1..=3 are ENOTSUP (45), a null param
/// too, and under SCHED_FIFO or SCHED_RR a priority outside 256..=767;
/// SCHED_OTHER takes any priority.
#[test]
fn posix_attr_setters_return_enotsup() {
    let s = Scratch::new(0x6A21_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (attr, param, out) = (s.base, s.base + 0x100, s.base + 0x200);
    assert_eq!(call(&mut ctx, px_attr_init, &[attr]), 0);
    assert_eq!(call(&mut ctx, px_attr_setinheritsched, &[attr, 1]), 45);
    assert_eq!(call(&mut ctx, px_attr_setinheritsched, &[attr, 0]), 0);
    assert_eq!(call(&mut ctx, px_attr_setschedpolicy, &[attr, 0]), 45);
    assert_eq!(call(&mut ctx, px_attr_setschedpolicy, &[attr, 4]), 45);
    assert_eq!(call(&mut ctx, px_attr_setschedparam, &[attr, 0]), 45);
    ctx.write_u32(param, 100);
    assert_eq!(call(&mut ctx, px_attr_setschedparam, &[attr, param]), 45, "FIFO priority below 256");
    assert_eq!(call(&mut ctx, px_attr_setschedpolicy, &[attr, 2]), 0);
    assert_eq!(call(&mut ctx, px_attr_setschedparam, &[attr, param]), 0, "SCHED_OTHER takes any priority");
    assert_eq!(call(&mut ctx, px_attr_getschedparam, &[attr, out]), 0);
    assert_eq!(ctx.read_u32(out), Some(100));
    assert_eq!(call(&mut ctx, px_attr_setschedpolicy, &[attr, 3]), 0);
    ctx.write_u32(param, 767);
    assert_eq!(call(&mut ctx, px_attr_setschedparam, &[attr, param]), 0);
    assert_eq!(call(&mut ctx, px_attr_destroy, &[attr]), 0);
    assert_eq!(call(&mut ctx, px_attr_setschedparam, &[attr, param]), 22);
    assert_eq!(call(&mut ctx, px_attr_setschedpolicy, &[attr, 1]), 22);
}

/// A new thread inherits its creator's policy and priority unless its
/// attributes ask for explicit scheduling; with no creating guest thread it
/// takes the attributes'.
#[test]
fn new_thread_scheduling_follows_inheritance() {
    let creator = ThreadAttr { policy: SCHED_RR, priority: 300, ..ThreadAttr::default() };
    let inherit = ThreadAttr { policy: SCHED_FIFO, priority: 500, inherit: DEFAULT_INHERIT, ..ThreadAttr::default() };
    let explicit = ThreadAttr { inherit: EXPLICIT_SCHED, ..inherit };
    assert_eq!(start_sched(&inherit, Some(&creator)), (SCHED_RR, 300));
    assert_eq!(start_sched(&ThreadAttr::default(), Some(&creator)), (SCHED_RR, 300));
    assert_eq!(start_sched(&explicit, Some(&creator)), (SCHED_FIFO, 500));
    assert_eq!(start_sched(&inherit, None), (SCHED_FIFO, 500));
}

/// scePthreadSetschedparam takes SCHED_FIFO or SCHED_RR with a priority of
/// 256..=767 or 896..=959 and keeps the policy; anything else is EINVAL and
/// changes nothing. Null threads are EINVAL for the priority and affinity
/// calls, and an empty affinity mask too.
#[test]
fn schedparam_and_affinity_checks() {
    let s = Scratch::new(0x6A22_0000_0000);
    let mut ctx = s.ctx();
    let (me, _) = fake_thread("sched", DEFAULT_AFFINITY);
    let (param, policy) = (s.base, s.base + 8);
    for (pol, prio) in [(2u64, 700u32), (0, 700), (1, 255), (1, 768), (3, 895), (3, 960)] {
        ctx.write_u32(param, prio);
        assert_eq!(call(&mut ctx, cb_setschedparam, &[me, pol, param]), EINVAL, "policy {pol} priority {prio}");
        assert_eq!(call(&mut ctx, px_setschedparam, &[me, pol, param]), 22);
    }
    assert_eq!(call(&mut ctx, cb_getschedparam, &[me, policy, param]), OK);
    assert_eq!((ctx.read_u32(policy), ctx.read_u32(param)), (Some(DEFAULT_POLICY as u32), Some(DEFAULT_PRIORITY as u32)));
    for (pol, prio) in [(3u64, 256u32), (1, 767), (3, 896), (1, 959)] {
        ctx.write_u32(param, prio);
        assert_eq!(call(&mut ctx, cb_setschedparam, &[me, pol, param]), OK);
        assert_eq!(call(&mut ctx, cb_getschedparam, &[me, policy, param]), OK);
        assert_eq!((ctx.read_u32(policy), ctx.read_u32(param)), (Some(pol as u32), Some(prio)));
    }
    assert_eq!(call(&mut ctx, cb_getprio, &[0, param]), EINVAL);
    assert_eq!(call(&mut ctx, cb_setprio, &[0, 700]), EINVAL);
    assert_eq!(call(&mut ctx, cb_getaffinity, &[0, param]), EINVAL);
    assert_eq!(call(&mut ctx, cb_setaffinity, &[0, 1]), EINVAL);
    assert_eq!(call(&mut ctx, cb_setaffinity, &[me, 0]), EINVAL);
    assert_eq!(call(&mut ctx, cb_setaffinity, &[me, 3]), OK);
    assert_eq!(call(&mut ctx, cb_getaffinity, &[me, param]), OK);
    assert_eq!(ctx.read_u64(param), Some(3));
    forget_fake_thread(me);
}

/// Joining a null thread is EINVAL and the calling thread EDEADLK;
/// detaching a thread twice is EINVAL, and so is joining a detached one.
#[test]
fn join_and_detach_errors() {
    let s = Scratch::new(0x6A23_0000_0000);
    let mut ctx = s.ctx();
    let (other, _) = fake_thread("other", DEFAULT_AFFINITY);
    let (me, _) = fake_thread("joiner", DEFAULT_AFFINITY);
    assert_eq!(call(&mut ctx, cb_join, &[0, 0]), EINVAL);
    assert_eq!(call(&mut ctx, cb_join, &[me, 0]), EDEADLK);
    assert_eq!(call(&mut ctx, cb_detach, &[other]), OK);
    assert_eq!(call(&mut ctx, cb_detach, &[other]), EINVAL);
    assert_eq!(call(&mut ctx, cb_join, &[other, 0]), EINVAL);
    forget_fake_thread(other);
    forget_fake_thread(me);
}

/// A destroyed mutex or condition is not revived as a static initializer:
/// every later operation is EINVAL (libthr THR_MUTEX_DESTROYED).
#[test]
fn destroyed_mutex_and_cond_are_invalid() {
    let s = Scratch::new(0x6A26_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (m, c, attr, out) = (s.base, s.base + 8, s.base + 16, s.base + 24);
    assert_eq!(call(&mut ctx, cb_mutex_init, &[m, 0, 0]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), OK);
    assert_eq!(ctx.read_u64(m), Some(2));
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), EINVAL);
    assert_eq!(call(&mut ctx, cb_mutex_trylock, &[m]), EINVAL);
    assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), EINVAL);
    assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), EINVAL);
    // A static adaptive initializer destroyed before use too.
    ctx.write_u64(m, MUTEX_STATIC_ADAPTIVE_INITIALIZER);
    assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), EINVAL);
    assert_eq!(call(&mut ctx, cb_cond_init, &[c, 0, 0]), OK);
    assert_eq!(call(&mut ctx, cb_cond_destroy, &[c]), OK);
    assert_eq!(ctx.read_u64(c), Some(1), "a destroyed condition holds 1");
    assert_eq!(call(&mut ctx, cb_cond_signal, &[c]), EINVAL);
    assert_eq!(call(&mut ctx, cb_cond_broadcast, &[c]), EINVAL);
    assert_eq!(call(&mut ctx, cb_cond_destroy, &[c]), EINVAL);
    assert_eq!(call(&mut ctx, cb_mutex_init, &[m, 0, 0]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
    assert_eq!(call(&mut ctx, cb_cond_wait, &[c, m]), EINVAL);
    assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), OK);
    // pthread_condattr_setpshared: only PTHREAD_PROCESS_PRIVATE.
    assert_eq!(call(&mut ctx, cb_condattr_init, &[attr]), OK);
    for v in [1u64, 2, u64::MAX] {
        assert_eq!(call(&mut ctx, cb_condattr_setpshared, &[attr, v]), EINVAL);
        assert_eq!(call(&mut ctx, cb_condattr_getpshared, &[attr, out]), OK);
        assert_eq!(ctx.read_u32(out), Some(0));
    }
    assert_eq!(call(&mut ctx, cb_condattr_setpshared, &[attr, 0]), OK);
    assert_eq!(call(&mut ctx, cb_condattr_setpshared, &[0, 0]), EINVAL);
    // Destroying an untouched (zero) condition leaves it zero; a later use
    // creates it as a static initializer.
    ctx.write_u64(c, 0);
    assert_eq!(call(&mut ctx, cb_cond_destroy, &[c]), OK);
    assert_eq!(ctx.read_u64(c), Some(0));
    assert_eq!(call(&mut ctx, cb_cond_signal, &[c]), OK);
    assert_ne!(ctx.read_u64(c), Some(0));
    assert_eq!(call(&mut ctx, cb_cond_destroy, &[c]), OK);
    assert_eq!(ctx.read_u64(c), Some(1));
}

/// A thread being joined can be neither detached nor joined again
/// (thr_detach.c EINVAL, thr_join.c EOPNOTSUPP); the joiner releases it
/// once and a later join finds no thread.
#[test]
fn thread_being_joined_refuses_detach_and_second_join() {
    let s = Scratch::new(0x6A25_0000_0000);
    let mut ctx = s.ctx();
    let (worker, _) = fake_thread("worker", DEFAULT_AFFINITY);
    let (me0, _) = fake_thread("main", DEFAULT_AFFINITY);
    let joiner = std::thread::spawn(move || {
        bind();
        join_thread(worker)
    });
    let info = thread(worker).unwrap();
    while !info.join.lock().unwrap().joining {
        std::thread::yield_now();
    }
    assert_eq!(call(&mut ctx, cb_detach, &[worker]), EINVAL);
    assert_eq!(call(&mut ctx, cb_join, &[worker, 0]), EOPNOTSUPP);
    finish_thread(&info, 7);
    assert_eq!(joiner.join().unwrap(), Ok(7));
    assert!(thread(worker).is_none(), "released by the joiner");
    assert_eq!(call(&mut ctx, cb_join, &[worker, 0]), ESRCH);
    forget_fake_thread(me0);
    // Detaching a running thread, then joining it: EINVAL.
    let (other, _) = fake_thread("other", DEFAULT_AFFINITY);
    let (me, _) = fake_thread("main2", DEFAULT_AFFINITY);
    assert_eq!(call(&mut ctx, cb_detach, &[other]), OK);
    assert_eq!(call(&mut ctx, cb_join, &[other, 0]), EINVAL);
    forget_fake_thread(other);
    forget_fake_thread(me);
}

/// A thread that ends inside a once initializer leaves the control "not
/// run": a waiter wakes and runs the initializer itself instead of waiting
/// forever (FreeBSD's once_cancel_handler).
#[test]
fn once_initializer_that_exits_is_run_again() {
    let s = Scratch::new(0x6A24_0000_0000);
    let ctx = s.ctx();
    let once = s.base;
    ctx.write_u32(once, 0);
    let vm = s.vm;
    let entered = Arc::new(AtomicU32::new(0));
    let release = Arc::new(AtomicU32::new(0));
    let (e, r) = (entered.clone(), release.clone());
    let first = std::thread::spawn(move || {
        let ctx = CpuContext::new(vm);
        run_once(&ctx, once, || {
            e.store(1, Ordering::SeqCst);
            while r.load(Ordering::SeqCst) == 0 {
                std::thread::yield_now();
            }
            // The thread ends here (pthread_exit / cancellation).
            abandon_running_once(&ctx);
        })
    });
    while entered.load(Ordering::SeqCst) == 0 {
        std::thread::yield_now();
    }
    let ran = Arc::new(AtomicU32::new(0));
    let ran2 = ran.clone();
    let waiter = std::thread::spawn(move || {
        let ctx = CpuContext::new(vm);
        run_once(&ctx, once, || {
            ran2.fetch_add(1, Ordering::SeqCst);
        })
    });
    std::thread::sleep(Duration::from_millis(20));
    assert!(!waiter.is_finished(), "the waiter waits for the running initializer");
    release.store(1, Ordering::SeqCst);
    assert_eq!(first.join().unwrap(), OK);
    assert_eq!(waiter.join().unwrap(), OK);
    assert_eq!(ran.load(Ordering::SeqCst), 1, "the waiter ran the initializer");
    assert_eq!(ctx.read_u32(once), Some(1), "done");
    assert_eq!(run_once(&ctx, once, || panic!("runs once")), OK);
}

/// The spin/yield loop accessors, isowned and reltimedlock (reference
/// GuestPthreadMutexErrors cases): an adaptive mutex starts with 2000 spin
/// loops, others with 0; a static initializer is created by the getters
/// but is never owned.
#[test]
fn mutex_loop_counts_and_ownership() {
    let s = Scratch::new(0x6A27_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (attr, m, adaptive, out) = (s.base, s.base + 8, s.base + 16, s.base + 0x100);
    assert_eq!(call(&mut ctx, cb_mutexattr_init, &[attr]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_init, &[m, attr, 0]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_getspinloops, &[m, out]), OK);
    assert_eq!(ctx.read_u32(out), Some(0));
    ctx.write_u64(adaptive, MUTEX_STATIC_ADAPTIVE_INITIALIZER);
    assert_eq!(call(&mut ctx, cb_mutex_isowned, &[adaptive]), 0, "a static initializer is not owned");
    assert_eq!(call(&mut ctx, cb_mutex_getspinloops, &[adaptive, out]), OK);
    assert_eq!(ctx.read_u32(out), Some(2000));
    assert_eq!(call(&mut ctx, cb_mutex_setspinloops, &[m, 17]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_getspinloops, &[m, out]), OK);
    assert_eq!(ctx.read_u32(out), Some(17));
    assert_eq!(call(&mut ctx, cb_mutex_setyieldloops, &[m, 5]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_getyieldloops, &[m, out]), OK);
    assert_eq!(ctx.read_u32(out), Some(5));
    assert_eq!(call(&mut ctx, cb_mutex_setyieldloops, &[0, 5]), OK, "always succeeds");
    assert_eq!(call(&mut ctx, cb_mutex_getspinloops, &[0, out]), EINVAL);
    assert_eq!(call(&mut ctx, cb_mutex_setspinloops, &[0, 1]), EINVAL);
    assert_eq!(call(&mut ctx, cb_mutex_isowned, &[m]), 0);
    assert_eq!(call(&mut ctx, px_mutex_reltimedlock, &[m, 1000]), 0);
    assert_eq!(call(&mut ctx, cb_mutex_isowned, &[m]), 1);
    assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_isowned, &[m]), 0);
    assert_eq!(call(&mut ctx, px_mutex_getspinloops, &[0, out]), 22);
    assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_destroy, &[adaptive]), OK);
}

/// A thread that has ended but is not joined yet: scePthreadAttrGet and
/// pthread_kill (signal 0 too) are ESRCH, pthread_getprio -1 with ESRCH;
/// a live one answers (reference GuestPthread cases measured on a PS5).
#[test]
fn ended_thread_is_esrch_before_join() {
    let s = Scratch::new(0x6A26_0000_0000);
    let mut ctx = s.ctx();
    let errno = crate::kernel::bind_test_errno(&ctx);
    let (worker, _) = fake_thread("ended", DEFAULT_AFFINITY);
    let (me, _) = fake_thread("main3", DEFAULT_AFFINITY);
    assert_eq!(call(&mut ctx, px_getprio, &[worker]), DEFAULT_PRIORITY as HleResult);
    assert_eq!(call(&mut ctx, cb_attr_init, &[s.base]), OK);
    assert_eq!(call(&mut ctx, cb_attr_get, &[worker, s.base]), OK);
    assert_eq!(call(&mut ctx, crate::signals::tests_pthread_kill, &[worker, 0]), 0);
    finish_thread(&thread(worker).unwrap(), 0);
    assert_eq!(call(&mut ctx, cb_attr_get, &[worker, s.base]), ESRCH);
    // The attribute is checked before the thread: NULL or an
    // uninitialized one is EINVAL even for an ended thread.
    assert_eq!(call(&mut ctx, px_attr_get_np, &[worker, s.base]), 3);
    assert_eq!(call(&mut ctx, px_attr_get_np, &[worker, 0]), 22);
    ctx.write_u64(s.base + 0x40, 0);
    assert_eq!(call(&mut ctx, px_attr_get_np, &[worker, s.base + 0x40]), 22);
    assert_eq!(call(&mut ctx, crate::signals::tests_pthread_kill, &[worker, 0]), 3);
    assert_eq!(call(&mut ctx, px_getprio, &[worker]), -1);
    assert_eq!(ctx.read_u32(errno), Some(3));
    assert_eq!(call(&mut ctx, px_getprio, &[0]), -1);
    forget_fake_thread(worker);
    forget_fake_thread(me);
}

mod posix_ext;

/// Threads created suspended: pthread_attr_setcreatesuspend_np marks the
/// attributes (EINVAL for NULL or a destroyed one), the start routine
/// waits until pthread_resume_np / resume_all lets it go, a NULL thread
/// is EINVAL and an ended one ESRCH; pthread_attr_get_np reports the flag.
#[test]
fn create_suspended_waits_for_resume() {
    let s = Scratch::new(0x6A27_0000_0000);
    let mut ctx = s.ctx();
    let (attr, destroyed, got) = (s.base, s.base + 8, s.base + 16);
    assert_eq!(call(&mut ctx, px_attr_setcreatesuspend, &[0]), 22);
    assert_eq!(call(&mut ctx, cb_attr_setcreatesuspend, &[0]), EINVAL);
    assert_eq!(call(&mut ctx, cb_attr_init, &[destroyed]), OK);
    assert_eq!(call(&mut ctx, cb_attr_destroy, &[destroyed]), OK);
    assert_eq!(call(&mut ctx, px_attr_setcreatesuspend, &[destroyed]), 22);
    assert_eq!(call(&mut ctx, cb_attr_init, &[attr]), OK);
    assert_eq!(call(&mut ctx, px_attr_setcreatesuspend, &[attr]), 0);
    assert!(read_attr(&ctx, attr).unwrap().create_suspended);
    assert_eq!(call(&mut ctx, px_resume, &[0]), 22);
    assert_eq!(call(&mut ctx, cb_resume, &[0]), EINVAL);

    let (worker, _) = fake_thread("suspended", DEFAULT_AFFINITY);
    let (me, _) = fake_thread("resumer", DEFAULT_AFFINITY);
    let info = thread(worker).unwrap();
    info.attr.lock().unwrap().create_suspended = true;
    *info.suspended.lock().unwrap() = true;
    assert_eq!(call(&mut ctx, cb_attr_init, &[got]), OK);
    assert_eq!(call(&mut ctx, px_attr_get_np, &[worker, got]), 0);
    assert!(read_attr(&ctx, got).unwrap().create_suspended, "attr_get reports a thread created suspended");
    let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (i2, r2) = (info.clone(), ran.clone());
    let start = std::thread::spawn(move || {
        wait_until_resumed(&i2);
        r2.store(true, Ordering::SeqCst);
    });
    std::thread::sleep(Duration::from_millis(50));
    assert!(!ran.load(Ordering::SeqCst), "a thread created suspended ran before it was resumed");
    // Resuming the caller or a running thread is 0 too.
    assert_eq!(call(&mut ctx, px_resume, &[me]), 0);
    assert_eq!(call(&mut ctx, px_resume, &[worker]), 0);
    start.join().unwrap();
    assert!(ran.load(Ordering::SeqCst));
    // resume_all lets every other thread go.
    *info.suspended.lock().unwrap() = true;
    assert_eq!(call(&mut ctx, cb_resume_all, &[]), OK);
    assert!(!*info.suspended.lock().unwrap());
    finish_thread(&info, 0);
    assert_eq!(call(&mut ctx, px_resume, &[worker]), 3);
    assert_eq!(call(&mut ctx, cb_resume, &[worker]), ESRCH);
    forget_fake_thread(worker);
    forget_fake_thread(me);
}

/// pthread_cond_setname_np keeps the condition variable usable, creates a
/// statically initialised one and refuses a destroyed one with EINVAL.
#[test]
fn cond_setname_np_lifecycle() {
    let s = Scratch::new(0x6D00_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (c, lazy, name) = (s.base, s.base + 8, s.base + 0x100);
    ctx.write_bytes(name, b"cond\0");
    assert_eq!(call(&mut ctx, px_cond_init, &[c, 0]), 0);
    assert_eq!(call(&mut ctx, px_cond_setname_np, &[c, name]), 0);
    assert_eq!(call(&mut ctx, px_cond_setname_np, &[c, 0]), 22);
    assert_eq!(call(&mut ctx, px_cond_signal, &[c]), 0);
    assert_eq!(call(&mut ctx, px_cond_destroy, &[c]), 0);
    assert_eq!(call(&mut ctx, px_cond_setname_np, &[c, name]), 22);
    assert_eq!(call(&mut ctx, px_cond_setname_np, &[0, name]), 22);
    ctx.write_u64(lazy, 0);
    assert_eq!(call(&mut ctx, px_cond_setname_np, &[lazy, name]), 0);
    assert_ne!(ctx.read_u64(lazy), Some(0), "a static condition variable is created");
    assert_eq!(call(&mut ctx, px_cond_destroy, &[lazy]), 0);
}
