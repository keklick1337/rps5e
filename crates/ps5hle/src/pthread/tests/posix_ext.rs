// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! POSIX extras: spin locks, barriers, atfork, scope, concurrency, pshared,
//! relative condition waits.

use super::*;
use crate::pthread::posix_ext::*;
use std::time::Duration;

/// Spin locks: argument errors, ownership (EPERM / EBUSY), relock refused
/// instead of spinning forever, destroy clears the variable.
#[test]
fn spin_lock_single_thread() {
    let s = Scratch::new(0x7E40_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let lock = s.base;
    assert_eq!(call(&mut ctx, px_spin_init, &[0, 0]), 22);
    assert_eq!(call(&mut ctx, px_spin_init, &[lock, 1]), 22);
    assert_eq!(call(&mut ctx, px_spin_init, &[lock, 2]), 22);
    assert_eq!(call(&mut ctx, px_spin_init, &[lock, u32::MAX as u64]), 22);
    assert_eq!(ctx.read_u64(lock), Some(0));
    for cb in [px_spin_lock, px_spin_trylock, px_spin_unlock, px_spin_destroy] {
        assert_eq!(call(&mut ctx, cb, &[0]), 22);
        assert_eq!(call(&mut ctx, cb, &[lock]), 22);
    }
    assert_eq!(call(&mut ctx, px_spin_init, &[lock, 0]), 0);
    assert_ne!(ctx.read_u64(lock), Some(0));
    assert_eq!(call(&mut ctx, px_spin_unlock, &[lock]), 1);
    assert_eq!(call(&mut ctx, px_spin_lock, &[lock]), 0);
    assert_eq!(call(&mut ctx, px_spin_trylock, &[lock]), 16);
    assert_eq!(call(&mut ctx, px_spin_lock, &[lock]), 11);
    assert_eq!(call(&mut ctx, px_spin_unlock, &[lock]), 0);
    assert_eq!(call(&mut ctx, px_spin_unlock, &[lock]), 1);
    assert_eq!(call(&mut ctx, px_spin_trylock, &[lock]), 0);
    assert_eq!(call(&mut ctx, px_spin_unlock, &[lock]), 0);
    assert_eq!(call(&mut ctx, px_spin_destroy, &[lock]), 0);
    assert_eq!(ctx.read_u64(lock), Some(0));
    assert_eq!(call(&mut ctx, px_spin_destroy, &[lock]), 22);
}

/// Another thread sees a held spin lock busy and cannot release it; two
/// threads incrementing under the lock lose no update.
#[test]
fn spin_lock_between_threads() {
    let s = Scratch::new(0x7E41_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (lock, counter) = (s.base, s.base + 0x100);
    assert_eq!(call(&mut ctx, px_spin_init, &[lock, 0]), 0);
    assert_eq!(call(&mut ctx, px_spin_lock, &[lock]), 0);
    let vm = s.vm;
    let (trylock, unlock) = std::thread::spawn(move || {
        let mut ctx = CpuContext::new(vm);
        bind();
        (call(&mut ctx, px_spin_trylock, &[lock]), call(&mut ctx, px_spin_unlock, &[lock]))
    })
    .join()
    .unwrap();
    assert_eq!((trylock, unlock), (16, 1));
    assert_eq!(call(&mut ctx, px_spin_unlock, &[lock]), 0);
    const N: u64 = 20000;
    let workers: Vec<_> = (0..2)
        .map(|_| {
            std::thread::spawn(move || {
                let mut ctx = CpuContext::new(vm);
                bind();
                for _ in 0..N {
                    assert_eq!(call(&mut ctx, px_spin_lock, &[lock]), 0);
                    let v = ctx.read_u64(counter).unwrap();
                    ctx.write_u64(counter, v + 1);
                    assert_eq!(call(&mut ctx, px_spin_unlock, &[lock]), 0);
                }
            })
        })
        .collect();
    for w in workers {
        w.join().unwrap();
    }
    assert_eq!(ctx.read_u64(counter), Some(2 * N));
    assert_eq!(call(&mut ctx, px_spin_destroy, &[lock]), 0);
}

/// Barrier attributes: only process-private, a destroyed or empty
/// attribute is EINVAL, getpshared needs an output.
#[test]
fn barrierattr_calls() {
    let s = Scratch::new(0x7E42_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (attr, missing, out) = (s.base, s.base + 8, s.base + 16);
    ctx.write_u32(out, u32::MAX);
    assert_eq!(call(&mut ctx, px_barrierattr_init, &[0]), 22);
    assert_eq!(call(&mut ctx, px_barrierattr_destroy, &[0]), 22);
    assert_eq!(call(&mut ctx, px_barrierattr_destroy, &[missing]), 22);
    assert_eq!(call(&mut ctx, px_barrierattr_getpshared, &[0, out]), 22);
    assert_eq!(call(&mut ctx, px_barrierattr_getpshared, &[missing, out]), 22);
    assert_eq!(call(&mut ctx, px_barrierattr_setpshared, &[0, 0]), 22);
    assert_eq!(call(&mut ctx, px_barrierattr_setpshared, &[missing, 0]), 22);
    assert_eq!(ctx.read_u32(out), Some(u32::MAX));
    assert_eq!(call(&mut ctx, px_barrierattr_init, &[attr]), 0);
    assert_eq!(call(&mut ctx, px_barrierattr_getpshared, &[attr, out]), 0);
    assert_eq!(ctx.read_u32(out), Some(0));
    for bad in [2u64, u32::MAX as u64, 1] {
        assert_eq!(call(&mut ctx, px_barrierattr_setpshared, &[attr, bad]), 22);
    }
    assert_eq!(call(&mut ctx, px_barrierattr_setpshared, &[attr, 0]), 0);
    assert_eq!(call(&mut ctx, px_barrierattr_getpshared, &[attr, 0]), 22);
    assert_eq!(call(&mut ctx, px_barrierattr_destroy, &[attr]), 0);
    assert_eq!(ctx.read_u64(attr), Some(0));
    assert_eq!(call(&mut ctx, px_barrierattr_destroy, &[attr]), 22);
}

/// Barriers: count 0 refused, a single-thread barrier always returns the
/// serial value, N threads get one serial and N-1 zeros, and the serial
/// thread may destroy the barrier while the others are still leaving.
#[test]
fn barrier_wait_and_destroy() {
    let s = Scratch::new(0x7E43_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (barrier, single) = (s.base, s.base + 8);
    assert_eq!(call(&mut ctx, px_barrier_init, &[0, 0, 1]), 22);
    assert_eq!(call(&mut ctx, px_barrier_init, &[barrier, 0, 0]), 22);
    assert_eq!(ctx.read_u64(barrier), Some(0));
    assert_eq!(call(&mut ctx, px_barrier_wait, &[barrier]), 22);
    assert_eq!(call(&mut ctx, px_barrier_init, &[single, 0, 1]), 0);
    assert_eq!(call(&mut ctx, px_barrier_wait, &[single]), -1);
    assert_eq!(call(&mut ctx, px_barrier_wait, &[single]), -1);
    assert_eq!(call(&mut ctx, px_barrier_destroy, &[single]), 0);
    const WAITERS: usize = 4;
    let vm = s.vm;
    for round in 0..20 {
        assert_eq!(call(&mut ctx, px_barrier_init, &[barrier, 0, WAITERS as u64]), 0);
        let destroy = round % 2 == 1;
        let threads: Vec<_> = (0..WAITERS)
            .map(|_| {
                std::thread::spawn(move || {
                    let mut ctx = CpuContext::new(vm);
                    bind();
                    let r = call(&mut ctx, px_barrier_wait, &[barrier]);
                    if r == -1 && destroy {
                        assert_eq!(call(&mut ctx, px_barrier_destroy, &[barrier]), 0);
                        assert_eq!(ctx.read_u64(barrier), Some(0));
                    }
                    r
                })
            })
            .collect();
        let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(results.iter().filter(|&&r| r == -1).count(), 1);
        assert_eq!(results.iter().filter(|&&r| r == 0).count(), WAITERS - 1);
        if !destroy {
            assert_eq!(call(&mut ctx, px_barrier_destroy, &[barrier]), 0);
        }
    }
}

/// A barrier with a thread waiting on it cannot be destroyed.
#[test]
fn barrier_destroy_while_waiting_is_busy() {
    let s = Scratch::new(0x7E44_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let barrier = s.base;
    assert_eq!(call(&mut ctx, px_barrier_init, &[barrier, 0, 2]), 0);
    let vm = s.vm;
    let waiter = std::thread::spawn(move || {
        let mut ctx = CpuContext::new(vm);
        bind();
        call(&mut ctx, px_barrier_wait, &[barrier])
    });
    // Give the waiter time to arrive (a destroy before that succeeds).
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(call(&mut ctx, px_barrier_destroy, &[barrier]), 16);
    assert_eq!(call(&mut ctx, px_barrier_wait, &[barrier]) + waiter.join().unwrap(), -1);
    assert_eq!(call(&mut ctx, px_barrier_destroy, &[barrier]), 0);
}

/// pthread_atfork keeps every handler triple in order, NULLs included, and
/// calls none of them.
#[test]
fn atfork_keeps_handlers() {
    let s = Scratch::new(0x7E45_0000_0000);
    let mut ctx = s.ctx();
    let triples = [(0x1000u64, 0x2000u64, 0x3000u64), (0, 0x2000, 0), (0, 0, 0)];
    for &(a, b, c) in &triples {
        assert_eq!(call(&mut ctx, px_atfork, &[a, b, c]), 0);
    }
    let h = atfork_handlers();
    let pos = h.iter().position(|t| *t == triples[0]).unwrap();
    assert_eq!(&h[pos..pos + 3], &triples);
}

/// Contention scope: system by default, process or system accepted,
/// anything else EINVAL.
#[test]
fn attr_scope() {
    let s = Scratch::new(0x7E46_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (attr, out) = (s.base, s.base + 8);
    assert_eq!(call(&mut ctx, px_attr_getscope, &[attr, out]), 22);
    assert_eq!(call(&mut ctx, cb_attr_init, &[attr]), OK);
    assert_eq!(call(&mut ctx, px_attr_getscope, &[attr, out]), 0);
    assert_eq!(ctx.read_u32(out), Some(SCOPE_SYSTEM as u32));
    assert_eq!(call(&mut ctx, px_attr_setscope, &[attr, SCOPE_PROCESS as u64]), 0);
    assert_eq!(call(&mut ctx, px_attr_getscope, &[attr, out]), 0);
    assert_eq!(ctx.read_u32(out), Some(SCOPE_PROCESS as u32));
    assert_eq!(call(&mut ctx, px_attr_setscope, &[attr, 1]), 22);
    assert_eq!(call(&mut ctx, px_attr_getscope, &[attr, 0]), 22);
    assert_eq!(call(&mut ctx, cb_attr_destroy, &[attr]), OK);
}

/// The concurrency level is kept; a negative one is EINVAL.
#[test]
fn concurrency_level() {
    let s = Scratch::new(0x7E47_0000_0000);
    let mut ctx = s.ctx();
    assert_eq!(call(&mut ctx, px_setconcurrency, &[4]), 0);
    assert_eq!(call(&mut ctx, px_getconcurrency, &[]), 4);
    assert_eq!(call(&mut ctx, px_setconcurrency, &[u32::MAX as u64]), 22);
    assert_eq!(call(&mut ctx, px_getconcurrency, &[]), 4);
    assert_eq!(call(&mut ctx, px_setconcurrency, &[0]), 0);
    assert_eq!(call(&mut ctx, px_getconcurrency, &[]), 0);
}

/// Mutex and rwlock attributes are process-private only.
#[test]
fn mutex_and_rwlock_attr_pshared() {
    let s = Scratch::new(0x7E48_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (mattr, rattr, out) = (s.base, s.base + 8, s.base + 16);
    assert_eq!(call(&mut ctx, px_mutexattr_getpshared, &[mattr, out]), 22);
    assert_eq!(call(&mut ctx, cb_mutexattr_init, &[mattr]), OK);
    ctx.write_u32(out, 7);
    assert_eq!(call(&mut ctx, px_mutexattr_getpshared, &[mattr, out]), 0);
    assert_eq!(ctx.read_u32(out), Some(0));
    assert_eq!(call(&mut ctx, px_mutexattr_setpshared, &[mattr, 1]), 22);
    assert_eq!(call(&mut ctx, px_mutexattr_setpshared, &[mattr, 0]), 0);
    assert_eq!(call(&mut ctx, px_mutexattr_getpshared, &[mattr, 0]), 22);
    assert_eq!(call(&mut ctx, cb_rwlockattr_init, &[rattr]), OK);
    ctx.write_u32(out, 7);
    assert_eq!(call(&mut ctx, px_rwlockattr_getpshared, &[rattr, out]), 0);
    assert_eq!(ctx.read_u32(out), Some(0));
    assert_eq!(call(&mut ctx, px_rwlockattr_setpshared, &[rattr, 1]), 22);
    assert_eq!(call(&mut ctx, px_rwlockattr_setpshared, &[rattr, 0]), 0);
    assert_eq!(call(&mut ctx, px_rwlockattr_setpshared, &[0, 0]), 22);
    assert_eq!(call(&mut ctx, px_rwlockattr_getpshared, &[0, out]), 22);
}

/// pthread_cond_reltimedwait_np: EPERM without the mutex, ETIMEDOUT after
/// the relative wait (also at 0), as POSIX numbers.
#[test]
fn cond_reltimedwait_np() {
    let s = Scratch::new(0x7E49_0000_0000);
    let mut ctx = s.ctx();
    bind();
    let (cond, m) = (s.base, s.base + 8);
    assert_eq!(call(&mut ctx, cb_cond_init, &[cond, 0]), OK);
    assert_eq!(call(&mut ctx, cb_mutex_init, &[m, 0, 0]), OK);
    assert_eq!(call(&mut ctx, px_cond_reltimedwait_np, &[cond, m, 1000]), 1);
    assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), OK);
    let start = Instant::now();
    assert_eq!(call(&mut ctx, px_cond_reltimedwait_np, &[cond, m, 50_000]), 60);
    assert!(start.elapsed() >= Duration::from_millis(45));
    assert_eq!(call(&mut ctx, px_cond_reltimedwait_np, &[cond, m, 0]), 60);
    assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), OK);
}
