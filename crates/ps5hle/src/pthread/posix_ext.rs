// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! POSIX thread calls without an `scePthread*` twin: spin locks, barriers,
//! `pthread_atfork`, `pthread_cond_reltimedwait_np`, the contention scope,
//! the concurrency level and the process-shared attributes. They return
//! POSIX error numbers (EINVAL = 22), not kernel codes.
//!
//! Spin locks and barriers are opaque objects: the guest variable holds a
//! pointer to a 0x100-byte object from the pthread object region. Only
//! process-private objects exist (PTHREAD_PROCESS_SHARED is refused).

use super::*;
use std::sync::atomic::AtomicI32;

const P_EPERM: HleResult = 1;
const P_ENOMEM: HleResult = 12;
const P_EBUSY: HleResult = 16;
const P_EINVAL: HleResult = 22;
const P_EDEADLK: HleResult = 11;
const PROCESS_PRIVATE: u64 = 0;
/// PTHREAD_BARRIER_SERIAL_THREAD: what the last thread to arrive gets.
const BARRIER_SERIAL_THREAD: HleResult = -1;
pub(super) const SCOPE_PROCESS: i32 = 0;
pub(super) const SCOPE_SYSTEM: i32 = 2;

/// Owner word of a spin lock object: the handle of the holding thread, 0
/// when free. Threads without a guest handle (host callers) use a
/// per-thread id from the top of the address space.
fn spin_owner_id() -> u64 {
    let h = current_handle();
    if h != 0 {
        return h;
    }
    thread_local!(static ID: u64 = {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        u64::MAX - NEXT.fetch_add(1, Ordering::Relaxed)
    });
    ID.with(|i| *i)
}

/// Live spin lock objects (a destroyed or foreign pointer is EINVAL).
static SPINS: Mutex<Vec<u64>> = Mutex::new(Vec::new());

fn spin_object(ctx: &CpuContext, ptr: u64) -> Option<&'static AtomicU64> {
    let obj = attr_object(ctx, ptr)?;
    if !SPINS.lock().unwrap().contains(&obj) {
        return None;
    }
    // SAFETY: a live spin object is a 0x100-byte block of the pthread
    // object region, mapped for the life of the process and 8-aligned.
    Some(unsafe { &*(obj as *const AtomicU64) })
}

/// pthread_spin_init(pthread_spinlock_t*, int pshared)
pub(super) fn px_spin_init(ctx: &mut CpuContext) -> HleResult {
    let (ptr, pshared) = (ctx.rdi(), ctx.rsi() as i32 as i64 as u64);
    if ptr == 0 || pshared != PROCESS_PRIVATE {
        return P_EINVAL;
    }
    let Some(obj) = alloc_object(ctx) else { return P_ENOMEM };
    SPINS.lock().unwrap().push(obj);
    if ctx.write_u64(ptr, obj) {
        0
    } else {
        SPINS.lock().unwrap().retain(|&o| o != obj);
        free_object(obj);
        P_EINVAL
    }
}

/// pthread_spin_destroy(pthread_spinlock_t*)
pub(super) fn px_spin_destroy(ctx: &mut CpuContext) -> HleResult {
    let ptr = ctx.rdi();
    let Some(obj) = attr_object(ctx, ptr) else { return P_EINVAL };
    let mut spins = SPINS.lock().unwrap();
    let Some(i) = spins.iter().position(|&o| o == obj) else { return P_EINVAL };
    spins.swap_remove(i);
    drop(spins);
    free_object(obj);
    ctx.write_u64(ptr, 0);
    0
}

fn spin_try(owner: &AtomicU64, me: u64) -> bool {
    owner.compare_exchange(0, me, Ordering::Acquire, Ordering::Relaxed).is_ok()
}

/// pthread_spin_trylock(pthread_spinlock_t*): EBUSY while held.
pub(super) fn px_spin_trylock(ctx: &mut CpuContext) -> HleResult {
    let Some(owner) = spin_object(ctx, ctx.rdi()) else { return P_EINVAL };
    if spin_try(owner, spin_owner_id()) { 0 } else { P_EBUSY }
}

/// pthread_spin_lock(pthread_spinlock_t*): spins, yielding the CPU every
/// 100000 tries. Relocking from the holder would spin forever; it is
/// refused with EDEADLK.
pub(super) fn px_spin_lock(ctx: &mut CpuContext) -> HleResult {
    const SPIN_COUNT: u32 = 100_000;
    let Some(owner) = spin_object(ctx, ctx.rdi()) else { return P_EINVAL };
    let me = spin_owner_id();
    if owner.load(Ordering::Relaxed) == me {
        log::error!("[pthread] pthread_spin_lock: the calling thread already holds the lock and would spin forever");
        return P_EDEADLK;
    }
    let mut count = SPIN_COUNT;
    while !spin_try(owner, me) {
        while owner.load(Ordering::Relaxed) != 0 {
            std::hint::spin_loop();
            count -= 1;
            if count == 0 {
                count = SPIN_COUNT;
                std::thread::yield_now();
            }
        }
    }
    0
}

/// pthread_spin_unlock(pthread_spinlock_t*): EPERM unless the caller holds
/// the lock.
pub(super) fn px_spin_unlock(ctx: &mut CpuContext) -> HleResult {
    let Some(owner) = spin_object(ctx, ctx.rdi()) else { return P_EINVAL };
    match owner.compare_exchange(spin_owner_id(), 0, Ordering::Release, Ordering::Relaxed) {
        Ok(_) => 0,
        Err(_) => P_EPERM,
    }
}

// ---------------------------------------------------------------------------
// Barriers
// ---------------------------------------------------------------------------

struct BarrierState {
    count: u32,
    arrived: u32,
    /// Threads inside pthread_barrier_wait (destroy waits for them to leave).
    active: u32,
    generation: u64,
}

struct Barrier {
    state: Mutex<BarrierState>,
    changed: Condvar,
}

static BARRIERS: Mutex<Option<HashMap<u64, Arc<Barrier>>>> = Mutex::new(None);

fn barrier(ctx: &CpuContext, ptr: u64) -> Option<(u64, Arc<Barrier>)> {
    let obj = attr_object(ctx, ptr)?;
    let b = BARRIERS.lock().unwrap().as_ref()?.get(&obj).cloned()?;
    Some((obj, b))
}

/// Barrier attribute objects: process-shared at +0 (always private).
static BARRIER_ATTRS: Mutex<Vec<u64>> = Mutex::new(Vec::new());

fn barrierattr_object(ctx: &CpuContext, ptr: u64) -> Option<u64> {
    attr_object(ctx, ptr).filter(|o| BARRIER_ATTRS.lock().unwrap().contains(o))
}

/// pthread_barrierattr_init(pthread_barrierattr_t*)
pub(super) fn px_barrierattr_init(ctx: &mut CpuContext) -> HleResult {
    let ptr = ctx.rdi();
    if ptr == 0 {
        return P_EINVAL;
    }
    let Some(obj) = alloc_object(ctx) else { return P_ENOMEM };
    BARRIER_ATTRS.lock().unwrap().push(obj);
    if ctx.write_u64(ptr, obj) { 0 } else { P_EINVAL }
}

/// pthread_barrierattr_destroy(pthread_barrierattr_t*)
pub(super) fn px_barrierattr_destroy(ctx: &mut CpuContext) -> HleResult {
    let ptr = ctx.rdi();
    let Some(obj) = barrierattr_object(ctx, ptr) else { return P_EINVAL };
    BARRIER_ATTRS.lock().unwrap().retain(|&o| o != obj);
    free_object(obj);
    ctx.write_u64(ptr, 0);
    0
}

/// pthread_barrierattr_getpshared(const pthread_barrierattr_t*, int*)
pub(super) fn px_barrierattr_getpshared(ctx: &mut CpuContext) -> HleResult {
    let (ptr, out) = (ctx.rdi(), ctx.rsi());
    let Some(obj) = barrierattr_object(ctx, ptr) else { return P_EINVAL };
    let v = ctx.read_u32(obj).unwrap_or(0);
    if out == 0 || !ctx.write_u32(out, v) {
        log::error!("[pthread] pthread_barrierattr_getpshared: null or unwritable output");
        return P_EINVAL;
    }
    0
}

/// pthread_barrierattr_setpshared(pthread_barrierattr_t*, int): only
/// PTHREAD_PROCESS_PRIVATE.
pub(super) fn px_barrierattr_setpshared(ctx: &mut CpuContext) -> HleResult {
    let (ptr, pshared) = (ctx.rdi(), ctx.rsi() as i32 as i64 as u64);
    let Some(obj) = barrierattr_object(ctx, ptr) else { return P_EINVAL };
    if pshared != PROCESS_PRIVATE {
        return P_EINVAL;
    }
    ctx.write_u32(obj, pshared as u32);
    0
}

/// pthread_barrier_init(pthread_barrier_t*, const pthread_barrierattr_t*,
/// unsigned count): count 0 is EINVAL; the attributes are not read.
pub(super) fn px_barrier_init(ctx: &mut CpuContext) -> HleResult {
    let (ptr, count) = (ctx.rdi(), ctx.rdx() as u32);
    if ptr == 0 || count == 0 {
        return P_EINVAL;
    }
    let Some(obj) = alloc_object(ctx) else { return P_ENOMEM };
    let b = Arc::new(Barrier {
        state: Mutex::new(BarrierState { count, arrived: 0, active: 0, generation: 0 }),
        changed: Condvar::new(),
    });
    BARRIERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(obj, b);
    if ctx.write_u64(ptr, obj) { 0 } else { P_EINVAL }
}

/// pthread_barrier_wait(pthread_barrier_t*): blocks until `count` threads
/// arrived; the last one gets PTHREAD_BARRIER_SERIAL_THREAD (-1), the
/// others 0.
pub(super) fn px_barrier_wait(ctx: &mut CpuContext) -> HleResult {
    let Some((_, b)) = barrier(ctx, ctx.rdi()) else { return P_EINVAL };
    let mut g = b.state.lock().unwrap();
    let generation = g.generation;
    g.active += 1;
    g.arrived += 1;
    if g.arrived == g.count {
        g.arrived = 0;
        g.generation += 1;
        g.active -= 1;
        b.changed.notify_all();
        return BARRIER_SERIAL_THREAD;
    }
    let (mut g, _) = crate::signals::wait_while(&b.state, &b.changed, g, None, |s| s.generation != generation);
    g.active -= 1;
    if g.active == 0 {
        b.changed.notify_all();
    }
    0
}

/// pthread_barrier_destroy(pthread_barrier_t*): EBUSY while threads wait
/// on it; otherwise waits for the woken threads to leave.
pub(super) fn px_barrier_destroy(ctx: &mut CpuContext) -> HleResult {
    let ptr = ctx.rdi();
    let Some((obj, b)) = barrier(ctx, ptr) else { return P_EINVAL };
    {
        let g = b.state.lock().unwrap();
        if g.arrived != 0 {
            return P_EBUSY;
        }
        drop(crate::signals::wait_while(&b.state, &b.changed, g, None, |s| s.active == 0));
    }
    if let Some(m) = BARRIERS.lock().unwrap().as_mut() {
        m.remove(&obj);
    }
    free_object(obj);
    ctx.write_u64(ptr, 0);
    0
}

// ---------------------------------------------------------------------------
// Fork handlers, relative condition wait, scope, concurrency, pshared
// ---------------------------------------------------------------------------

/// (prepare, parent, child) as registered; a PS5 process never forks, so
/// they are only kept.
static ATFORK: Mutex<Vec<(u64, u64, u64)>> = Mutex::new(Vec::new());

/// pthread_atfork(prepare, parent, child): any of them may be NULL.
pub(super) fn px_atfork(ctx: &mut CpuContext) -> HleResult {
    ATFORK.lock().unwrap().push((ctx.rdi(), ctx.rsi(), ctx.rdx()));
    0
}

#[cfg(test)]
pub(super) fn atfork_handlers() -> Vec<(u64, u64, u64)> {
    ATFORK.lock().unwrap().clone()
}

/// pthread_cond_reltimedwait_np(cond*, mutex*, useconds): the relative wait
/// of scePthreadCondTimedwait with a POSIX error number.
pub(super) fn px_cond_reltimedwait_np(ctx: &mut CpuContext) -> HleResult {
    posix_error(cb_sce_cond_timedwait(ctx))
}

/// pthread_attr_getscope(const pthread_attr_t*, int*)
pub(super) fn px_attr_getscope(ctx: &mut CpuContext) -> HleResult {
    let (ptr, out) = (ctx.rdi(), ctx.rsi());
    let Some(a) = read_attr(ctx, ptr) else { return P_EINVAL };
    if out == 0 || !ctx.write_u32(out, a.scope as u32) {
        return P_EINVAL;
    }
    0
}

/// pthread_attr_setscope(pthread_attr_t*, int): PTHREAD_SCOPE_PROCESS (0)
/// or PTHREAD_SCOPE_SYSTEM (2).
pub(super) fn px_attr_setscope(ctx: &mut CpuContext) -> HleResult {
    let (ptr, scope) = (ctx.rdi(), ctx.rsi() as i32);
    if scope != SCOPE_PROCESS && scope != SCOPE_SYSTEM {
        return P_EINVAL;
    }
    posix_error(update_attr(ctx, ptr, |a| a.scope = scope))
}

static CONCURRENCY: AtomicI32 = AtomicI32::new(0);

/// pthread_setconcurrency(int): a hint, kept for getconcurrency.
pub(super) fn px_setconcurrency(ctx: &mut CpuContext) -> HleResult {
    let level = ctx.rdi() as i32;
    if level < 0 {
        return P_EINVAL;
    }
    CONCURRENCY.store(level, Ordering::Relaxed);
    0
}

/// pthread_getconcurrency(): the last level set (0 at start).
pub(super) fn px_getconcurrency(_ctx: &mut CpuContext) -> HleResult {
    CONCURRENCY.load(Ordering::Relaxed) as HleResult
}

/// Process-shared word of a mutex attribute object (after type, protocol
/// and ceiling).
const MUTEXATTR_PSHARED: u64 = 12;

/// pthread_mutexattr_getpshared(const pthread_mutexattr_t*, int*)
pub(super) fn px_mutexattr_getpshared(ctx: &mut CpuContext) -> HleResult {
    let (ptr, out) = (ctx.rdi(), ctx.rsi());
    let Some(obj) = attr_object(ctx, ptr) else { return P_EINVAL };
    let v = ctx.read_u32(obj + MUTEXATTR_PSHARED).unwrap_or(0);
    if out == 0 || !ctx.write_u32(out, v) {
        log::error!("[pthread] pthread_mutexattr_getpshared: null or unwritable output");
        return P_EINVAL;
    }
    0
}

/// pthread_mutexattr_setpshared(pthread_mutexattr_t*, int): only
/// PTHREAD_PROCESS_PRIVATE.
pub(super) fn px_mutexattr_setpshared(ctx: &mut CpuContext) -> HleResult {
    let (ptr, pshared) = (ctx.rdi(), ctx.rsi() as i32 as i64 as u64);
    let Some(obj) = attr_object(ctx, ptr) else { return P_EINVAL };
    if pshared != PROCESS_PRIVATE {
        return P_EINVAL;
    }
    ctx.write_u32(obj + MUTEXATTR_PSHARED, 0);
    0
}

/// pthread_rwlockattr_getpshared(const pthread_rwlockattr_t*, int*): always
/// PTHREAD_PROCESS_PRIVATE.
pub(super) fn px_rwlockattr_getpshared(ctx: &mut CpuContext) -> HleResult {
    let (ptr, out) = (ctx.rdi(), ctx.rsi());
    if attr_object(ctx, ptr).is_none() || out == 0 || !ctx.write_u32(out, PROCESS_PRIVATE as u32) {
        log::error!("[pthread] pthread_rwlockattr_getpshared({ptr:#x}, {out:#x}): null attribute or output");
        return P_EINVAL;
    }
    0
}

/// pthread_rwlockattr_setpshared(pthread_rwlockattr_t*, int): the value is
/// checked first; only PTHREAD_PROCESS_PRIVATE.
pub(super) fn px_rwlockattr_setpshared(ctx: &mut CpuContext) -> HleResult {
    let (ptr, pshared) = (ctx.rdi(), ctx.rsi() as i32 as i64 as u64);
    if pshared != PROCESS_PRIVATE {
        return P_EINVAL;
    }
    if attr_object(ctx, ptr).is_none() {
        log::error!("[pthread] pthread_rwlockattr_setpshared({ptr:#x}): null attribute");
        return P_EINVAL;
    }
    0
}

pub(super) const POSIX_EXT_EXPORTS: &[(&str, &str, crate::registry::HleCallback)] = &[
    ("ZMn3clnAGBA", "pthread_spin_init", px_spin_init),
    ("IJIggoPZExk", "pthread_spin_destroy", px_spin_destroy),
    ("pw+70ClLYlY", "pthread_spin_lock", px_spin_lock),
    ("rCTGkBIHfPY", "pthread_spin_trylock", px_spin_trylock),
    ("LEfMMCT+SlM", "pthread_spin_unlock", px_spin_unlock),
    ("4nqCnLJSvck", "pthread_barrierattr_init", px_barrierattr_init),
    ("AsCQCYTbe80", "pthread_barrierattr_destroy", px_barrierattr_destroy),
    ("a5JZMyjFV68", "pthread_barrierattr_getpshared", px_barrierattr_getpshared),
    ("jqrGJJxFhmU", "pthread_barrierattr_setpshared", px_barrierattr_setpshared),
    ("ZsXLFtd2jqQ", "pthread_barrier_init", px_barrier_init),
    ("CawZgCYqXWk", "pthread_barrier_wait", px_barrier_wait),
    ("+Pqub9HZCPo", "pthread_barrier_destroy", px_barrier_destroy),
    ("U9t5kJAWPnA", "pthread_atfork", px_atfork),
    ("K953PF5u6Pc", "pthread_cond_reltimedwait_np", px_cond_reltimedwait_np),
    ("e2G+cdEkOmU", "pthread_attr_getscope", px_attr_getscope),
    ("xesmlSI-KCI", "pthread_attr_setscope", px_attr_setscope),
    ("fBG1gHx1RlI", "pthread_setconcurrency", px_setconcurrency),
    ("s+QHU9RLHS4", "pthread_getconcurrency", px_getconcurrency),
    ("PmL-TwKUzXI", "pthread_mutexattr_getpshared", px_mutexattr_getpshared),
    ("EXv3ztGqtDM", "pthread_mutexattr_setpshared", px_mutexattr_setpshared),
    ("VqEMuCv-qHY", "pthread_rwlockattr_getpshared", px_rwlockattr_getpshared),
    ("OuKg+kRDD7U", "pthread_rwlockattr_setpshared", px_rwlockattr_setpshared),
];
