// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Mutexes, condition variables, read/write locks, their attribute
//! objects, thread-specific keys and `pthread_once`.
//!
//! A guest `pthread_mutex_t` / `pthread_cond_t` / `pthread_rwlock_t` holds
//! the address of a 0x100-byte object in the pthread object region. The
//! host state of every object lives in a per-type slot table indexed by
//! that address, so a lookup is two atomic loads: no global lock and no
//! hash map on the lock/unlock paths. Slot states are allocated once and
//! reused when the guest object slot is reused, so a stale guest pointer
//! can never reach freed host memory (it sees a dead object instead).
//!
//! The uncontended paths take only the object's own host mutex (a futex
//! CAS, no system call); waiters are counted so that unlock and signal
//! only wake the kernel when somebody actually sleeps. Mutexes go further:
//! the owner lives in an atomic lock word, so lock and unlock are one
//! atomic operation each, and a contended adaptive mutex spins a few
//! microseconds before it sleeps (`RPS5E_FAST_MUTEX`,
//! `RPS5E_MUTEX_SPIN_US`). With `RPS5E_MUTEX_HANDOFF=1` a contended
//! unlock hands the mutex to the longest sleeper (FIFO) instead.

use super::{alloc_object, attr_object, current_handle, free_object, OBJECT_REGION, OBJECT_REGION_SIZE, OBJECT_SIZE};
use super::{EAGAIN, EBUSY, EDEADLK, EINVAL, ENOMEM, EPERM, ETIMEDOUT, OK};
use crate::context::CpuContext;
use crate::registry::HleResult;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

pub(super) const MUTEX_ERRORCHECK: i32 = 1;
pub(super) const MUTEX_RECURSIVE: i32 = 2;
pub(super) const MUTEX_ADAPTIVE: i32 = 4;
/// `PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP`: a mutex word of 1.
pub(super) const MUTEX_STATIC_ADAPTIVE_INITIALIZER: u64 = 1;

const PRIO_NONE: i32 = 0;
const PRIO_PROTECT: i32 = 2;

/// Kernel clock ids accepted by `pthread_condattr_setclock`.
const CLOCK_REALTIME: i32 = 0;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// Slot tables
// ---------------------------------------------------------------------------

const SLOTS: usize = (OBJECT_REGION_SIZE / OBJECT_SIZE) as usize;
const CHUNK: usize = 4096;
const CHUNKS: usize = SLOTS / CHUNK;

/// Host state of guest objects, indexed by guest object address.
pub(super) struct SlotTable<T: 'static> {
    chunks: [OnceLock<Box<[AtomicPtr<T>]>>; CHUNKS],
}

impl<T: 'static> SlotTable<T> {
    pub(super) const fn new() -> Self {
        Self { chunks: [const { OnceLock::new() }; CHUNKS] }
    }

    fn index(obj: u64) -> Option<usize> {
        let off = obj.checked_sub(OBJECT_REGION)?;
        (off < OBJECT_REGION_SIZE && off % OBJECT_SIZE == 0).then_some((off / OBJECT_SIZE) as usize)
    }

    /// The state stored for `obj`, if any (live or not).
    #[inline]
    pub(super) fn get(&self, obj: u64) -> Option<&'static T> {
        let i = Self::index(obj)?;
        let chunk = self.chunks[i / CHUNK].get()?;
        let p = chunk[i % CHUNK].load(Ordering::Acquire);
        // SAFETY: slot states are leaked boxes that are never freed.
        (!p.is_null()).then(|| unsafe { &*p })
    }

    /// The state for `obj`, allocating it with `make` on first use.
    pub(super) fn get_or_create(&self, obj: u64, make: impl FnOnce() -> T) -> Option<&'static T> {
        let i = Self::index(obj)?;
        let chunk = self.chunks[i / CHUNK].get_or_init(|| (0..CHUNK).map(|_| AtomicPtr::new(std::ptr::null_mut())).collect());
        let slot = &chunk[i % CHUNK];
        let p = slot.load(Ordering::Acquire);
        if !p.is_null() {
            // SAFETY: see `get`.
            return Some(unsafe { &*p });
        }
        let fresh = Box::into_raw(Box::new(make()));
        match slot.compare_exchange(std::ptr::null_mut(), fresh, Ordering::AcqRel, Ordering::Acquire) {
            // SAFETY: just published, never freed.
            Ok(_) => Some(unsafe { &*fresh }),
            Err(existing) => {
                // SAFETY: `fresh` was never shared.
                drop(unsafe { Box::from_raw(fresh) });
                // SAFETY: see `get`.
                Some(unsafe { &*existing })
            }
        }
    }
}

/// What a destroyed mutex or condition variable holds (libthr
/// THR_MUTEX_DESTROYED): no object lives there, so every later operation,
/// a second destroy included, is EINVAL instead of a fresh static object.
pub(super) const DESTROYED: u64 = 2;

/// Serializes lazy creation of statically initialized objects.
static STATIC_INIT_GATE: Mutex<()> = Mutex::new(());

/// Resolve the object behind `ptr` (a `T*` holding the object address),
/// creating it with `create` when the word is a static initializer.
fn resolve<T: 'static>(
    ctx: &CpuContext,
    ptr: u64,
    table: &SlotTable<T>,
    is_static: impl Fn(u64) -> bool,
    live: impl Fn(&T) -> bool,
    create: impl FnOnce(&CpuContext, u64) -> Option<(u64, &'static T)>,
) -> Result<&'static T, HleResult> {
    if ptr == 0 {
        return Err(EINVAL);
    }
    let val = ctx.read_u64(ptr).ok_or(EINVAL)?;
    if !is_static(val) {
        return table.get(val).filter(|t| live(t)).ok_or(EINVAL);
    }
    let _g = lock(&STATIC_INIT_GATE);
    let val = ctx.read_u64(ptr).ok_or(EINVAL)?;
    if !is_static(val) {
        return table.get(val).filter(|t| live(t)).ok_or(EINVAL);
    }
    let (obj, t) = create(ctx, val).ok_or(ENOMEM)?;
    ctx.write_u64(ptr, obj);
    Ok(t)
}

// ---------------------------------------------------------------------------
// Mutexes
// ---------------------------------------------------------------------------

pub(super) struct MutexState {
    /// Owner and depth of a mutex kept under its host lock (`fast` off).
    pub(super) owner: u64,
    pub(super) count: u32,
    protocol: i32,
    /// Threads sleeping on `cv` (both implementations).
    sleepers: u32,
    /// Owner ids of the sleepers in arrival order (handoff mutexes): an
    /// unlock passes the mutex to the first one.
    pub(super) queue: VecDeque<u64>,
}

/// Lock word bit 0: a thread may sleep on the mutex, so the unlock must
/// wake one.
const CONTENDED: u64 = 1;

pub(super) struct HostMutex {
    live: AtomicBool,
    pub(super) kind: AtomicI32,
    /// Lock state held in `word` (true) or in `state` (false); fixed when the
    /// object is created.
    fast: AtomicBool,
    /// Lock word: owner id << 1 | CONTENDED, 0 while free.
    word: AtomicU64,
    /// Recursion depth while `word` is held; only the owner touches it.
    depth: AtomicU32,
    /// How long a contended lock spins before it sleeps, in nanoseconds.
    spin_ns: AtomicU64,
    /// A contended unlock hands ownership to the first queued sleeper.
    handoff: AtomicBool,
    pub(super) state: Mutex<MutexState>,
    cv: Condvar,
    /// The guest-visible spin and yield loop counts
    /// (scePthreadMutexGet/Setspinloops, ...Yieldloops): 2000 spins for an
    /// adaptive mutex, 0 otherwise; they do not change how we wait.
    spinloops: AtomicI32,
    yieldloops: AtomicI32,
}

/// Spin loops of a new adaptive mutex.
const ADAPTIVE_MUTEX_SPINS: i32 = 2000;

pub(super) static MUTEXES: SlotTable<HostMutex> = SlotTable::new();

fn env_on(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|v| !matches!(v.trim(), "0" | "off" | "false" | "no"))
}

/// `RPS5E_FAST_MUTEX=0`: every mutex keeps its whole state under the host
/// lock, as before the lock word.
fn fast_mutex_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_on("RPS5E_FAST_MUTEX").unwrap_or(true))
}

/// Spin before sleeping on a contended mutex, in nanoseconds: adaptive
/// mutexes spin `RPS5E_MUTEX_SPIN_US` (default 4 µs, 0 = never);
/// `RPS5E_MUTEX_SPIN_ALL=1` makes every kind spin.
fn spin_budget_ns(kind: i32) -> u64 {
    static CFG: OnceLock<(u64, bool)> = OnceLock::new();
    let (ns, all) = *CFG.get_or_init(|| {
        let us = std::env::var("RPS5E_MUTEX_SPIN_US").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(4);
        (us.min(1000) * 1000, env_on("RPS5E_MUTEX_SPIN_ALL").unwrap_or(false))
    });
    if kind == MUTEX_ADAPTIVE || all { ns } else { 0 }
}

/// `RPS5E_MUTEX_HANDOFF=1`: a contended unlock hands the mutex to the
/// longest sleeper (FIFO) instead of freeing it for the woken sleepers and
/// newcomers to race for. Off by default: fair, but contended throughput
/// drops 2-4x (lock convoys) in `bench_mutex_handoff`.
fn mutex_handoff_enabled() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_MUTEX_HANDOFF.with(|m| m.get()) {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_on("RPS5E_MUTEX_HANDOFF").unwrap_or(false))
}

#[cfg(test)]
thread_local! {
    /// Tests pick the implementation (lock word, spin ns) of the mutexes
    /// they create.
    pub(super) static TEST_MUTEX_MODE: std::cell::Cell<Option<(bool, u64)>> = const { std::cell::Cell::new(None) };
    /// Tests pick whether the mutexes they create hand off.
    pub(super) static TEST_MUTEX_HANDOFF: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

fn creation_mode(kind: i32) -> (bool, u64) {
    #[cfg(test)]
    if let Some((fast, spin)) = TEST_MUTEX_MODE.with(|m| m.get()) {
        return (fast, if kind == MUTEX_ADAPTIVE { spin } else { 0 });
    }
    let fast = fast_mutex_enabled();
    (fast, if fast { spin_budget_ns(kind) } else { 0 })
}

/// Lock word owner id of the calling thread: its guest thread handle, or a
/// private id for host threads without one.
fn owner_id() -> u64 {
    let h = current_handle();
    if h != 0 {
        return h;
    }
    static NEXT: AtomicU64 = AtomicU64::new(1);
    thread_local! {
        static ID: u64 = (1 << 61) | NEXT.fetch_add(1, Ordering::Relaxed);
    }
    ID.with(|id| *id)
}

fn new_mutex(ctx: &CpuContext, kind: i32, protocol: i32) -> Option<(u64, &'static HostMutex)> {
    let obj = alloc_object(ctx)?;
    // Type and protocol at the offsets the guest-visible mutex layout uses.
    ctx.write_u32(obj + 0x20, kind as u32);
    ctx.write_u32(obj + 0x3C, protocol as u32);
    let m = MUTEXES.get_or_create(obj, || HostMutex {
        live: AtomicBool::new(false),
        kind: AtomicI32::new(kind),
        fast: AtomicBool::new(false),
        word: AtomicU64::new(0),
        depth: AtomicU32::new(0),
        spin_ns: AtomicU64::new(0),
        handoff: AtomicBool::new(false),
        state: Mutex::new(MutexState { owner: 0, count: 0, protocol, sleepers: 0, queue: VecDeque::new() }),
        cv: Condvar::new(),
        spinloops: AtomicI32::new(0),
        yieldloops: AtomicI32::new(0),
    })?;
    let (fast, spin_ns) = creation_mode(kind);
    {
        let mut s = lock(&m.state);
        s.owner = 0;
        s.count = 0;
        s.protocol = protocol;
        s.queue.clear();
        m.handoff.store(mutex_handoff_enabled(), Ordering::Relaxed);
        m.kind.store(kind, Ordering::Relaxed);
        m.fast.store(fast, Ordering::Relaxed);
        m.word.store(0, Ordering::Relaxed);
        m.depth.store(0, Ordering::Relaxed);
        m.spin_ns.store(spin_ns, Ordering::Relaxed);
        m.spinloops.store(if kind == MUTEX_ADAPTIVE { ADAPTIVE_MUTEX_SPINS } else { 0 }, Ordering::Relaxed);
        m.yieldloops.store(0, Ordering::Relaxed);
    }
    m.live.store(true, Ordering::Release);
    Some((obj, m))
}

/// Resolve `mutex*`, creating statically initialized mutexes.
pub(super) fn resolve_mutex(ctx: &CpuContext, ptr: u64) -> Result<&'static HostMutex, HleResult> {
    resolve(
        ctx,
        ptr,
        &MUTEXES,
        |v| v <= MUTEX_STATIC_ADAPTIVE_INITIALIZER,
        |m| m.live.load(Ordering::Acquire),
        |ctx, v| {
            let kind = if v == MUTEX_STATIC_ADAPTIVE_INITIALIZER { MUTEX_ADAPTIVE } else { MUTEX_ERRORCHECK };
            new_mutex(ctx, kind, PRIO_NONE)
        },
    )
}

impl HostMutex {
    #[inline]
    fn is_fast(&self) -> bool {
        self.fast.load(Ordering::Relaxed)
    }

    /// Whether the calling thread holds the mutex.
    fn held_by_current(&self) -> bool {
        if self.is_fast() {
            let w = self.word.load(Ordering::Relaxed);
            w != 0 && w >> 1 == owner_id()
        } else {
            let s = lock(&self.state);
            s.count > 0 && s.owner == current_handle()
        }
    }

    /// Whether any thread holds the mutex (`s` is its locked state).
    fn is_held(&self, s: &MutexState) -> bool {
        if self.is_fast() { self.word.load(Ordering::Acquire) != 0 } else { s.count > 0 }
    }

    /// Take the free lock word.
    #[inline]
    fn try_take(&self, mine: u64) -> bool {
        if self.word.compare_exchange(0, mine, Ordering::Acquire, Ordering::Relaxed).is_ok() {
            self.depth.store(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Spin up to `spin_ns` for the lock word to come free.
    fn spin_take(&self, mine: u64) -> bool {
        let budget = self.spin_ns.load(Ordering::Relaxed);
        if budget == 0 {
            return false;
        }
        let start = Instant::now();
        loop {
            for _ in 0..64 {
                if self.word.load(Ordering::Relaxed) == 0 && self.try_take(mine) {
                    return true;
                }
                std::hint::spin_loop();
            }
            if start.elapsed().as_nanos() as u64 >= budget {
                return false;
            }
        }
    }
}

/// Lock with an optional relative timeout in microseconds.
pub(super) fn mutex_lock(m: &HostMutex, timeout_us: Option<u64>) -> HleResult {
    if m.is_fast() {
        return fast_lock(m, timeout_us);
    }
    let me = current_handle();
    let mut s = lock(&m.state);
    if s.count > 0 && s.owner == me {
        if m.kind.load(Ordering::Relaxed) == MUTEX_RECURSIVE {
            if s.count == u32::MAX {
                return EAGAIN;
            }
            s.count += 1;
            return OK;
        }
        if let Some(us) = timeout_us {
            if us > 0 {
                let deadline = Instant::now() + Duration::from_micros(us);
                while Instant::now() < deadline {
                    s = crate::signals::wait_cv(&m.state, &m.cv, s, Some(deadline)).0;
                }
            }
            return ETIMEDOUT;
        }
        return EDEADLK;
    }
    if m.handoff.load(Ordering::Relaxed) {
        // A held mutex is handed over by its unlock, in queue order.
        if s.count == 0 {
            s.owner = me;
            s.count = 1;
            return OK;
        }
        if timeout_us == Some(0) {
            return ETIMEDOUT;
        }
        let deadline = timeout_us.map(|us| Instant::now() + Duration::from_micros(us));
        s.queue.push_back(me);
        s.sleepers += 1;
        loop {
            if s.count > 0 && s.owner == me {
                s.sleepers -= 1;
                return OK;
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                if let Some(k) = s.queue.iter().position(|&t| t == me) {
                    s.queue.remove(k);
                }
                s.sleepers -= 1;
                return ETIMEDOUT;
            }
            s = crate::signals::wait_cv(&m.state, &m.cv, s, deadline).0;
        }
    }
    match timeout_us {
        None => {
            while s.count > 0 {
                s.sleepers += 1;
                s = crate::signals::wait_cv(&m.state, &m.cv, s, None).0;
                s.sleepers -= 1;
            }
        }
        Some(0) => {
            if s.count > 0 {
                return ETIMEDOUT;
            }
        }
        Some(us) => {
            let deadline = Instant::now() + Duration::from_micros(us);
            while s.count > 0 {
                if Instant::now() >= deadline {
                    return ETIMEDOUT;
                }
                s.sleepers += 1;
                s = crate::signals::wait_cv(&m.state, &m.cv, s, Some(deadline)).0;
                s.sleepers -= 1;
            }
        }
    }
    s.owner = me;
    s.count = 1;
    OK
}

/// Lock-word implementation of [`mutex_lock`]: a free mutex is one CAS;
/// a contended one spins (adaptive) and then sleeps on `cv` with the
/// CONTENDED bit set, so the unlock knows to wake it.
fn fast_lock(m: &HostMutex, timeout_us: Option<u64>) -> HleResult {
    let me = owner_id();
    let mine = me << 1;
    if m.try_take(mine) {
        return OK;
    }
    let w = m.word.load(Ordering::Relaxed);
    if w != 0 && w >> 1 == me {
        if m.kind.load(Ordering::Relaxed) == MUTEX_RECURSIVE {
            let d = m.depth.load(Ordering::Relaxed);
            if d == u32::MAX {
                return EAGAIN;
            }
            m.depth.store(d + 1, Ordering::Relaxed);
            return OK;
        }
        if let Some(us) = timeout_us {
            if us > 0 {
                let deadline = Instant::now() + Duration::from_micros(us);
                let mut s = lock(&m.state);
                while Instant::now() < deadline {
                    s = crate::signals::wait_cv(&m.state, &m.cv, s, Some(deadline)).0;
                }
            }
            return ETIMEDOUT;
        }
        return EDEADLK;
    }
    if timeout_us == Some(0) {
        return if m.try_take(mine) { OK } else { ETIMEDOUT };
    }
    let deadline = timeout_us.map(|us| Instant::now() + Duration::from_micros(us));
    if m.spin_take(mine) {
        return OK;
    }
    let mut s = lock(&m.state);
    if m.handoff.load(Ordering::Relaxed) {
        return fast_lock_queued(m, s, me, deadline);
    }
    loop {
        let w = m.word.load(Ordering::Relaxed);
        if w == 0 {
            // Others still sleep: keep the bit so this unlock wakes the next.
            let word = mine | if s.sleepers > 0 { CONTENDED } else { 0 };
            if m.word.compare_exchange(0, word, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                m.depth.store(1, Ordering::Relaxed);
                return OK;
            }
            continue;
        }
        // Publish the sleeper before sleeping. The bit is set with the host
        // lock held, and the unlock takes that lock before it notifies, so
        // the wakeup cannot fall between this check and the wait.
        if w & CONTENDED == 0 && m.word.compare_exchange(w, w | CONTENDED, Ordering::Relaxed, Ordering::Relaxed).is_err() {
            continue;
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return ETIMEDOUT;
        }
        s.sleepers += 1;
        s = crate::signals::wait_cv(&m.state, &m.cv, s, deadline).0;
        s.sleepers -= 1;
    }
}

/// The sleeping part of [`fast_lock`] for a handoff mutex: the caller
/// queues behind the other sleepers and owns the mutex once an unlock
/// writes its id into the lock word. While anyone is queued the word is
/// never 0 (the unlock passes it on), so newcomers queue as well.
fn fast_lock_queued<'a>(m: &'a HostMutex, mut s: MutexGuard<'a, MutexState>, me: u64, deadline: Option<Instant>) -> HleResult {
    let mine = me << 1;
    let mut queued = false;
    loop {
        let w = m.word.load(Ordering::Acquire);
        if queued {
            if w >> 1 == me {
                s.sleepers -= 1;
                return OK;
            }
        } else if w == 0 {
            if m.word.compare_exchange(0, mine, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                m.depth.store(1, Ordering::Relaxed);
                return OK;
            }
            continue;
        } else {
            // Publish the sleeper with the host lock held: the unlock takes
            // that lock before it hands over, so it finds this entry.
            if w & CONTENDED == 0 && m.word.compare_exchange(w, w | CONTENDED, Ordering::Relaxed, Ordering::Relaxed).is_err() {
                continue;
            }
            s.queue.push_back(me);
            s.sleepers += 1;
            queued = true;
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            // Handovers happen under the host lock: the word is settled.
            if let Some(k) = s.queue.iter().position(|&t| t == me) {
                s.queue.remove(k);
            }
            s.sleepers -= 1;
            return ETIMEDOUT;
        }
        s = crate::signals::wait_cv(&m.state, &m.cv, s, deadline).0;
    }
}

/// Unlock of a handoff lock word: an uncontended word is cleared; with
/// sleepers the first queued one becomes the owner (depth 1) before any
/// thread is woken, so neither the unlocker nor a newcomer can take the
/// mutex in between.
fn fast_unlock_handoff(m: &HostMutex) {
    loop {
        let w = m.word.load(Ordering::Relaxed);
        if w & CONTENDED != 0 {
            break;
        }
        if m.word.compare_exchange(w, 0, Ordering::Release, Ordering::Relaxed).is_ok() {
            return;
        }
    }
    let mut s = lock(&m.state);
    match s.queue.pop_front() {
        Some(next) => {
            let more = if s.queue.is_empty() { 0 } else { CONTENDED };
            m.depth.store(1, Ordering::Relaxed);
            m.word.store(next << 1 | more, Ordering::Release);
            drop(s);
            m.cv.notify_all();
        }
        // Every sleeper gave up (timed out).
        None => m.word.store(0, Ordering::Release),
    }
}

pub(super) fn mutex_trylock(m: &HostMutex) -> HleResult {
    if m.is_fast() {
        let me = owner_id();
        if m.try_take(me << 1) {
            return OK;
        }
        let w = m.word.load(Ordering::Relaxed);
        if w != 0 && w >> 1 == me && m.kind.load(Ordering::Relaxed) == MUTEX_RECURSIVE {
            let d = m.depth.load(Ordering::Relaxed);
            if d == u32::MAX {
                return EAGAIN;
            }
            m.depth.store(d + 1, Ordering::Relaxed);
            return OK;
        }
        return EBUSY;
    }
    let me = current_handle();
    let mut s = lock(&m.state);
    if s.count > 0 {
        if s.owner == me && m.kind.load(Ordering::Relaxed) == MUTEX_RECURSIVE {
            if s.count == u32::MAX {
                return EAGAIN;
            }
            s.count += 1;
            return OK;
        }
        return EBUSY;
    }
    s.owner = me;
    s.count = 1;
    OK
}

/// Unlock; with `recurse`, release completely and report the depth.
pub(super) fn mutex_unlock(m: &HostMutex, recurse: Option<&mut u32>) -> HleResult {
    if m.is_fast() {
        let me = owner_id();
        let w = m.word.load(Ordering::Relaxed);
        if w == 0 || w >> 1 != me {
            return EPERM;
        }
        if let Some(r) = recurse {
            *r = m.depth.load(Ordering::Relaxed);
            m.depth.store(1, Ordering::Relaxed);
        }
        let d = m.depth.load(Ordering::Relaxed);
        if m.kind.load(Ordering::Relaxed) == MUTEX_RECURSIVE && d > 1 {
            m.depth.store(d - 1, Ordering::Relaxed);
            return OK;
        }
        m.depth.store(0, Ordering::Relaxed);
        if m.handoff.load(Ordering::Relaxed) {
            fast_unlock_handoff(m);
            return OK;
        }
        // Only the owner clears the word; others only add CONTENDED.
        if m.word.swap(0, Ordering::Release) & CONTENDED != 0 {
            // A sleeper set the bit holding the host lock and keeps it until
            // it waits: once the lock is ours it is waiting.
            drop(lock(&m.state));
            m.cv.notify_one();
        }
        return OK;
    }
    let me = current_handle();
    let mut s = lock(&m.state);
    if s.count == 0 || s.owner != me {
        return EPERM;
    }
    if let Some(r) = recurse {
        *r = s.count;
        s.count = 1;
    }
    if m.kind.load(Ordering::Relaxed) == MUTEX_RECURSIVE && s.count > 1 {
        s.count -= 1;
        return OK;
    }
    if m.handoff.load(Ordering::Relaxed) {
        // The first sleeper owns the mutex before it wakes.
        match s.queue.pop_front() {
            Some(next) => {
                s.owner = next;
                s.count = 1;
                drop(s);
                m.cv.notify_all();
            }
            None => {
                s.owner = 0;
                s.count = 0;
            }
        }
        return OK;
    }
    s.owner = 0;
    s.count = 0;
    let wake = s.sleepers > 0;
    drop(s);
    if wake {
        m.cv.notify_one();
    }
    OK
}

fn mutex_lock_recurse(m: &HostMutex, recurse: u32) -> HleResult {
    let r = mutex_lock(m, None);
    if r == OK {
        if m.is_fast() {
            m.depth.store(recurse.max(1), Ordering::Relaxed);
        } else {
            lock(&m.state).count = recurse.max(1);
        }
    }
    r
}

/// Priority ceiling a protocol change resets to: the highest SCHED_RR
/// priority.
const DEFAULT_PRIO_CEILING: u32 = crate::kernel_ext::PRIORITY_HIGHEST as u32;

/// Mutex attribute object: type at +0, protocol at +4, priority ceiling at
/// +8.
fn mutexattr_read(ctx: &CpuContext, attr_ptr: u64) -> (i32, i32) {
    match attr_object(ctx, attr_ptr) {
        Some(obj) => (
            ctx.read_u32(obj).map_or(MUTEX_ERRORCHECK, |v| v as i32),
            ctx.read_u32(obj + 4).map_or(PRIO_NONE, |v| v as i32),
        ),
        None => (MUTEX_ERRORCHECK, PRIO_NONE),
    }
}

pub(super) fn cb_mutexattr_init(ctx: &mut CpuContext) -> HleResult {
    let ptr = ctx.rdi();
    if ptr == 0 {
        return EINVAL;
    }
    let Some(obj) = alloc_object(ctx) else { return ENOMEM };
    ctx.write_u32(obj, MUTEX_ERRORCHECK as u32);
    ctx.write_u32(obj + 4, PRIO_NONE as u32);
    ctx.write_u32(obj + 8, 0);
    if ctx.write_u64(ptr, obj) { OK } else { EINVAL }
}

pub(super) fn cb_mutexattr_destroy(ctx: &mut CpuContext) -> HleResult {
    let ptr = ctx.rdi();
    let Some(obj) = attr_object(ctx, ptr) else { return EINVAL };
    free_object(obj);
    ctx.write_u64(ptr, 0);
    OK
}

pub(super) fn cb_mutexattr_settype(ctx: &mut CpuContext) -> HleResult {
    let (ptr, kind) = (ctx.rdi(), ctx.rsi() as i32);
    let Some(obj) = attr_object(ctx, ptr) else { return EINVAL };
    if !(MUTEX_ERRORCHECK..=MUTEX_ADAPTIVE).contains(&kind) {
        return EINVAL;
    }
    ctx.write_u32(obj, kind as u32);
    OK
}

pub(super) fn cb_mutexattr_gettype(ctx: &mut CpuContext) -> HleResult {
    let (ptr, out) = (ctx.rdi(), ctx.rsi());
    let Some(obj) = attr_object(ctx, ptr) else { return EINVAL };
    let v = ctx.read_u32(obj).unwrap_or(MUTEX_ERRORCHECK as u32);
    if out == 0 || !ctx.write_u32(out, v) { EINVAL } else { OK }
}

pub(super) fn cb_mutexattr_setprotocol(ctx: &mut CpuContext) -> HleResult {
    let (ptr, protocol) = (ctx.rdi(), ctx.rsi() as i32);
    let Some(obj) = attr_object(ctx, ptr) else { return EINVAL };
    if !(PRIO_NONE..=PRIO_PROTECT).contains(&protocol) {
        return EINVAL;
    }
    ctx.write_u32(obj + 4, protocol as u32);
    ctx.write_u32(obj + 8, DEFAULT_PRIO_CEILING);
    OK
}

/// pthread_mutexattr_setprioceiling(attr*, ceiling): only under
/// PTHREAD_PRIO_PROTECT. Mutexes do not apply priority protection.
pub(super) fn cb_mutexattr_setprioceiling(ctx: &mut CpuContext) -> HleResult {
    let (ptr, ceiling) = (ctx.rdi(), ctx.rsi() as u32);
    let Some(obj) = attr_object(ctx, ptr) else { return EINVAL };
    if ctx.read_u32(obj + 4) != Some(PRIO_PROTECT as u32) {
        return EINVAL;
    }
    ctx.write_u32(obj + 8, ceiling);
    OK
}

/// pthread_mutexattr_getprioceiling(attr*, ceiling*)
pub(super) fn cb_mutexattr_getprioceiling(ctx: &mut CpuContext) -> HleResult {
    let (ptr, out) = (ctx.rdi(), ctx.rsi());
    let Some(obj) = attr_object(ctx, ptr) else { return EINVAL };
    if ctx.read_u32(obj + 4) != Some(PRIO_PROTECT as u32) {
        return EINVAL;
    }
    let v = ctx.read_u32(obj + 8).unwrap_or(0);
    if out == 0 || !ctx.write_u32(out, v) { EINVAL } else { OK }
}

pub(super) fn cb_mutexattr_getprotocol(ctx: &mut CpuContext) -> HleResult {
    let (ptr, out) = (ctx.rdi(), ctx.rsi());
    let Some(obj) = attr_object(ctx, ptr) else { return EINVAL };
    let v = ctx.read_u32(obj + 4).unwrap_or(0);
    if out == 0 || !ctx.write_u32(out, v) { EINVAL } else { OK }
}

/// pthread_mutex_init(mutex*, attr*) / scePthreadMutexInit(mutex*, attr*, name)
pub(super) fn cb_mutex_init(ctx: &mut CpuContext) -> HleResult {
    let (ptr, attr_ptr) = (ctx.rdi(), ctx.rsi());
    if ptr == 0 {
        return EINVAL;
    }
    let (kind, protocol) = mutexattr_read(ctx, attr_ptr);
    let Some((obj, _)) = new_mutex(ctx, kind, protocol) else { return ENOMEM };
    if ctx.write_u64(ptr, obj) { OK } else { EINVAL }
}

pub(super) fn cb_mutex_destroy(ctx: &mut CpuContext) -> HleResult {
    let ptr = ctx.rdi();
    if ptr == 0 {
        return EINVAL;
    }
    let Some(val) = ctx.read_u64(ptr) else { return EINVAL };
    if val == 0 {
        return EINVAL;
    }
    if val == MUTEX_STATIC_ADAPTIVE_INITIALIZER {
        ctx.write_u64(ptr, DESTROYED);
        return OK;
    }
    let Some(m) = MUTEXES.get(val).filter(|m| m.live.load(Ordering::Acquire)) else { return EINVAL };
    {
        let s = lock(&m.state);
        if m.is_held(&s) {
            return EBUSY;
        }
        m.live.store(false, Ordering::Release);
    }
    free_object(val);
    ctx.write_u64(ptr, DESTROYED);
    OK
}

pub(super) fn cb_mutex_lock(ctx: &mut CpuContext) -> HleResult {
    match resolve_mutex(ctx, ctx.rdi()) {
        Ok(m) => mutex_lock(m, None),
        Err(e) => e,
    }
}

pub(super) fn cb_mutex_trylock(ctx: &mut CpuContext) -> HleResult {
    match resolve_mutex(ctx, ctx.rdi()) {
        Ok(m) => mutex_trylock(m),
        Err(e) => e,
    }
}

/// scePthreadMutexGetspinloops / Getyieldloops(mutex*, int* count): a
/// static initializer is created first; an invalid mutex is EINVAL.
fn mutex_get_loops(ctx: &CpuContext, yield_loops: bool) -> HleResult {
    let m = match resolve_mutex(ctx, ctx.rdi()) {
        Ok(m) => m,
        Err(e) => return e,
    };
    let v = if yield_loops { &m.yieldloops } else { &m.spinloops }.load(Ordering::Relaxed);
    if ctx.rsi() == 0 || !ctx.write_u32(ctx.rsi(), v as u32) {
        return EINVAL;
    }
    OK
}

pub(super) fn cb_mutex_getspinloops(ctx: &mut CpuContext) -> HleResult {
    mutex_get_loops(ctx, false)
}

pub(super) fn cb_mutex_getyieldloops(ctx: &mut CpuContext) -> HleResult {
    mutex_get_loops(ctx, true)
}

/// scePthreadMutexSetspinloops(mutex*, int count): EINVAL for an invalid
/// mutex.
pub(super) fn cb_mutex_setspinloops(ctx: &mut CpuContext) -> HleResult {
    match resolve_mutex(ctx, ctx.rdi()) {
        Ok(m) => {
            m.spinloops.store(ctx.rsi() as i32, Ordering::Relaxed);
            OK
        }
        Err(e) => e,
    }
}

/// scePthreadMutexSetyieldloops(mutex*, int count): always succeeds.
pub(super) fn cb_mutex_setyieldloops(ctx: &mut CpuContext) -> HleResult {
    if let Ok(m) = resolve_mutex(ctx, ctx.rdi()) {
        m.yieldloops.store(ctx.rsi() as i32, Ordering::Relaxed);
    }
    OK
}

/// pthread_mutex_isowned_np(mutex*) -> 1 when the calling thread holds it;
/// a static initializer (or anything that is not a mutex) is not held.
pub(super) fn cb_mutex_isowned(ctx: &mut CpuContext) -> HleResult {
    let ptr = ctx.rdi();
    match ctx.read_u64(ptr) {
        Some(v) if v > MUTEX_STATIC_ADAPTIVE_INITIALIZER => {}
        _ => return 0,
    }
    match resolve_mutex(ctx, ptr) {
        Ok(m) => m.held_by_current() as HleResult,
        Err(_) => 0,
    }
}

/// scePthreadMutexTimedlock(mutex*, usec)
pub(super) fn cb_mutex_timedlock_usec(ctx: &mut CpuContext) -> HleResult {
    let usec = ctx.rsi() as u32 as u64;
    match resolve_mutex(ctx, ctx.rdi()) {
        Ok(m) => mutex_lock(m, Some(usec)),
        Err(e) => e,
    }
}

/// pthread_mutex_timedlock(mutex*, abstime*): the absolute CLOCK_REALTIME
/// deadline becomes a relative timeout, clamped to 32 bits of microseconds.
/// As in FreeBSD's libthr, a free mutex is taken before the deadline is
/// looked at (an invalid one then does not matter), and a null deadline
/// waits without limit.
pub(super) fn cb_mutex_timedlock_abs(ctx: &mut CpuContext) -> HleResult {
    let m = match resolve_mutex(ctx, ctx.rdi()) {
        Ok(m) => m,
        Err(e) => return e,
    };
    if ctx.rsi() == 0 {
        return mutex_lock(m, None);
    }
    if mutex_trylock(m) == OK {
        return OK;
    }
    let Some((sec, nsec)) = read_abs_timespec(ctx, ctx.rsi()) else { return EINVAL };
    let now = super::realtime_now();
    let mut sec_delta = sec - now.as_secs() as i64;
    let mut nsec_delta = nsec - now.subsec_nanos() as i64;
    if nsec_delta < 0 {
        sec_delta -= 1;
        nsec_delta += 1_000_000_000;
    }
    let mut usec = 0u64;
    if sec_delta > 0 || (sec_delta == 0 && nsec_delta > 0) {
        usec = if sec_delta as u64 > u32::MAX as u64 / 1_000_000 { u32::MAX as u64 } else { sec_delta as u64 * 1_000_000 };
        usec = (usec + (nsec_delta / 1000) as u64).min(u32::MAX as u64);
    }
    mutex_lock(m, Some(usec))
}

pub(super) fn cb_mutex_unlock(ctx: &mut CpuContext) -> HleResult {
    match resolve_mutex(ctx, ctx.rdi()) {
        Ok(m) => mutex_unlock(m, None),
        Err(e) => e,
    }
}

/// `(tv_sec, tv_nsec)` of a valid absolute timespec.
fn read_abs_timespec(ctx: &CpuContext, ts: u64) -> Option<(i64, i64)> {
    if ts == 0 {
        return None;
    }
    let sec = ctx.read_u64(ts)? as i64;
    let nsec = ctx.read_u64(ts + 8)? as i64;
    (sec >= 0 && (0..1_000_000_000).contains(&nsec)).then_some((sec, nsec))
}

// ---------------------------------------------------------------------------
// Condition variables
// ---------------------------------------------------------------------------

/// One per guest thread: a condition waiter is woken by bumping its
/// sequence and notifying its own condvar, so signal wakes exactly the
/// thread it picked.
pub(super) struct Waiter {
    handle: AtomicU64,
    seq: AtomicU64,
    cv: Condvar,
}

thread_local! {
    static WAITER: Arc<Waiter> = Arc::new(Waiter { handle: AtomicU64::new(0), seq: AtomicU64::new(0), cv: Condvar::new() });
}

pub(super) struct CondState {
    clock: i32,
    waiters: Vec<Arc<Waiter>>,
}

pub(super) struct HostCond {
    live: AtomicBool,
    state: Mutex<CondState>,
}

pub(super) static CONDS: SlotTable<HostCond> = SlotTable::new();

/// What `scePthreadCondDestroy` leaves in the slot: 1, which no object
/// lives at, so every later condition call is EINVAL (measured on a PS5,
/// firmware 13.60).
pub(super) const COND_DESTROYED: u64 = 1;

fn wake_waiter(s: &mut CondState, handle: Option<u64>) {
    let idx = match handle {
        None => (!s.waiters.is_empty()).then_some(0),
        Some(h) => s.waiters.iter().position(|w| w.handle.load(Ordering::Relaxed) == h),
    };
    if let Some(i) = idx {
        let w = s.waiters.remove(i);
        w.seq.fetch_add(1, Ordering::Release);
        w.cv.notify_one();
    }
}

fn new_cond(ctx: &CpuContext, clock: i32) -> Option<(u64, &'static HostCond)> {
    let obj = alloc_object(ctx)?;
    let c = CONDS.get_or_create(obj, || HostCond { live: AtomicBool::new(false), state: Mutex::new(CondState { clock, waiters: Vec::new() }) })?;
    {
        let mut s = lock(&c.state);
        s.clock = clock;
        s.waiters.clear();
    }
    c.live.store(true, Ordering::Release);
    Some((obj, c))
}

pub(super) fn resolve_cond(ctx: &CpuContext, ptr: u64) -> Result<&'static HostCond, HleResult> {
    resolve(ctx, ptr, &CONDS, |v| v == 0, |c| c.live.load(Ordering::Acquire), |ctx, _| new_cond(ctx, CLOCK_REALTIME))
}

/// Condition attribute object: clock id at +0, process-shared at +4.
pub(super) fn cb_condattr_init(ctx: &mut CpuContext) -> HleResult {
    let ptr = ctx.rdi();
    if ptr == 0 {
        return EINVAL;
    }
    let Some(obj) = alloc_object(ctx) else { return ENOMEM };
    ctx.write_u32(obj, CLOCK_REALTIME as u32);
    ctx.write_u32(obj + 4, 0);
    if ctx.write_u64(ptr, obj) { OK } else { EINVAL }
}

pub(super) fn cb_attr_object_destroy(ctx: &mut CpuContext) -> HleResult {
    let ptr = ctx.rdi();
    let Some(obj) = attr_object(ctx, ptr) else { return EINVAL };
    free_object(obj);
    ctx.write_u64(ptr, 0);
    OK
}

/// Whether a kernel clock id has a host clock behind it.
/// The clocks `pthread_condattr_setclock` accepts: realtime, virtual,
/// profiling and monotonic only; every other id (the uptime, precise, fast
/// and second variants too) is EINVAL (measured on a PS5).
pub(crate) fn clock_is_supported(clock_id: i32) -> bool {
    use crate::time::{CLOCK_MONOTONIC, CLOCK_PROF, CLOCK_REALTIME, CLOCK_VIRTUAL};
    matches!(clock_id, CLOCK_REALTIME | CLOCK_VIRTUAL | CLOCK_PROF | CLOCK_MONOTONIC)
}

pub(super) fn cb_condattr_setclock(ctx: &mut CpuContext) -> HleResult {
    let (ptr, clock) = (ctx.rdi(), ctx.rsi() as i32);
    let Some(obj) = attr_object(ctx, ptr) else { return EINVAL };
    if !clock_is_supported(clock) {
        return EINVAL;
    }
    ctx.write_u32(obj, clock as u32);
    OK
}

pub(super) fn cb_condattr_getclock(ctx: &mut CpuContext) -> HleResult {
    let (ptr, out) = (ctx.rdi(), ctx.rsi());
    let Some(obj) = attr_object(ctx, ptr) else { return EINVAL };
    let v = ctx.read_u32(obj).unwrap_or(0);
    if out == 0 || !ctx.write_u32(out, v) { EINVAL } else { OK }
}

/// Only PTHREAD_PROCESS_PRIVATE (0) is accepted; PTHREAD_PROCESS_SHARED and
/// anything else is EINVAL and leaves the attribute as it was (PS5 FW 12.70).
pub(super) fn cb_condattr_setpshared(ctx: &mut CpuContext) -> HleResult {
    let (ptr, v) = (ctx.rdi(), ctx.rsi() as i32);
    let Some(obj) = attr_object(ctx, ptr) else { return EINVAL };
    if v != 0 {
        return EINVAL;
    }
    ctx.write_u32(obj + 4, 0);
    OK
}

pub(super) fn cb_condattr_getpshared(ctx: &mut CpuContext) -> HleResult {
    let (ptr, out) = (ctx.rdi(), ctx.rsi());
    let Some(obj) = attr_object(ctx, ptr) else { return EINVAL };
    let v = ctx.read_u32(obj + 4).unwrap_or(0);
    if out == 0 || !ctx.write_u32(out, v) { EINVAL } else { OK }
}

pub(super) fn cb_cond_init(ctx: &mut CpuContext) -> HleResult {
    let (ptr, attr_ptr) = (ctx.rdi(), ctx.rsi());
    if ptr == 0 {
        return EINVAL;
    }
    let clock = attr_object(ctx, attr_ptr).and_then(|o| ctx.read_u32(o)).map_or(CLOCK_REALTIME, |v| v as i32);
    let Some((obj, _)) = new_cond(ctx, clock) else { return ENOMEM };
    if ctx.write_u64(ptr, obj) { OK } else { EINVAL }
}

pub(super) fn cb_cond_destroy(ctx: &mut CpuContext) -> HleResult {
    let ptr = ctx.rdi();
    if ptr == 0 {
        return EINVAL;
    }
    let Some(val) = ctx.read_u64(ptr) else { return EINVAL };
    if val == 0 {
        return OK;
    }
    let Some(c) = CONDS.get(val).filter(|c| c.live.load(Ordering::Acquire)) else { return EINVAL };
    c.live.store(false, Ordering::Release);
    free_object(val);
    ctx.write_u64(ptr, COND_DESTROYED);
    OK
}

#[derive(Clone, Copy)]
enum CondTimeout {
    Infinite,
    /// Relative microseconds (`scePthreadCondTimedwait`); 0 times out at once.
    Usec(u64),
    Deadline(Instant),
}

fn cond_wait(ctx: &CpuContext, cond_ptr: u64, mutex_ptr: u64, timeout: impl FnOnce(i32) -> Option<CondTimeout>) -> HleResult {
    let c = match resolve_cond(ctx, cond_ptr) {
        Ok(c) => c,
        Err(e) => return e,
    };
    let m = match resolve_mutex(ctx, mutex_ptr) {
        Ok(m) => m,
        Err(e) => return e,
    };
    let me = current_handle();
    let mut g = lock(&c.state);
    let Some(timeout) = timeout(g.clock) else { return EINVAL };
    if !m.held_by_current() {
        return EPERM;
    }
    let waiter = WAITER.with(|w| w.clone());
    waiter.handle.store(me, Ordering::Relaxed);
    let start = waiter.seq.load(Ordering::Acquire);
    g.waiters.push(waiter.clone());
    let mut recurse = 0u32;
    let r = mutex_unlock(m, Some(&mut recurse));
    if r != OK {
        g.waiters.retain(|w| !Arc::ptr_eq(w, &waiter));
        return r;
    }
    let ready = || waiter.seq.load(Ordering::Acquire) != start;
    let deadline = match timeout {
        CondTimeout::Infinite => None,
        CondTimeout::Usec(us) => Some(Instant::now() + Duration::from_micros(us)),
        CondTimeout::Deadline(d) => Some(d),
    };
    let mut result = OK;
    // A cancellation request ends the wait (a cancellation point).
    let canceled = crate::kernel_ext::cancel_requested;
    if matches!(timeout, CondTimeout::Usec(0)) {
        result = ETIMEDOUT;
    } else {
        while !ready() {
            if canceled() || deadline.is_some_and(|d| Instant::now() >= d) {
                break;
            }
            g = crate::signals::wait_cv(&c.state, &waiter.cv, g, deadline).0;
        }
        if !ready() {
            result = ETIMEDOUT;
        }
    }
    g.waiters.retain(|w| !Arc::ptr_eq(w, &waiter));
    drop(g);
    let lr = mutex_lock_recurse(m, recurse);
    if result == ETIMEDOUT && lr == OK {
        // Not woken: a pending request is acted on with the mutex held
        // again, as the cleanup handlers expect.
        crate::kernel_ext::test_cancel();
    }
    if result == OK {
        result = lr;
    }
    result
}

pub(super) fn cb_cond_wait(ctx: &mut CpuContext) -> HleResult {
    cond_wait(ctx, ctx.rdi(), ctx.rsi(), |_| Some(CondTimeout::Infinite))
}

/// scePthreadCondTimedwait(cond*, mutex*, usec)
pub(super) fn cb_sce_cond_timedwait(ctx: &mut CpuContext) -> HleResult {
    let us = ctx.rdx() as u32 as u64;
    cond_wait(ctx, ctx.rdi(), ctx.rsi(), |_| Some(CondTimeout::Usec(us)))
}

/// pthread_cond_timedwait(cond*, mutex*, abstime*): the deadline is on the
/// condition's clock.
pub(super) fn cb_posix_cond_timedwait(ctx: &mut CpuContext) -> HleResult {
    let abs = read_abs_timespec(ctx, ctx.rdx());
    cond_wait(ctx, ctx.rdi(), ctx.rsi(), |clock| {
        let (sec, nsec) = abs?;
        let now = crate::time::clock_now(clock)?;
        let target = sec as i128 * 1_000_000_000 + nsec as i128;
        let cur = now.as_nanos() as i128;
        let delta = (target - cur).max(0) as u64;
        Some(CondTimeout::Deadline(Instant::now() + Duration::from_nanos(delta)))
    })
}

fn cond_wake(ctx: &CpuContext, target: Option<Option<u64>>) -> HleResult {
    let c = match resolve_cond(ctx, ctx.rdi()) {
        Ok(c) => c,
        Err(e) => return e,
    };
    let mut s = lock(&c.state);
    match target {
        None => {
            for w in s.waiters.drain(..) {
                w.seq.fetch_add(1, Ordering::Release);
                w.cv.notify_one();
            }
        }
        Some(h) => wake_waiter(&mut s, h),
    }
    OK
}

pub(super) fn cb_cond_signal(ctx: &mut CpuContext) -> HleResult {
    cond_wake(ctx, Some(None))
}

/// scePthreadCondSignalto(cond*, thread)
pub(super) fn cb_cond_signalto(ctx: &mut CpuContext) -> HleResult {
    let thread = ctx.rsi();
    if thread == 0 {
        return EINVAL;
    }
    cond_wake(ctx, Some(Some(thread)))
}

pub(super) fn cb_cond_broadcast(ctx: &mut CpuContext) -> HleResult {
    cond_wake(ctx, None)
}

/// pthread_cond_setname_np(cond*, name): the name is not kept, as nothing
/// reads one back; a statically initialised condition variable is created
/// first, a destroyed one (or a null pointer or name) is EINVAL.
pub(super) fn cb_cond_setname_np(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 {
        return EINVAL;
    }
    match resolve_cond(ctx, ctx.rdi()) {
        Ok(_) => OK,
        Err(e) => e,
    }
}

// ---------------------------------------------------------------------------
// Read/write locks
// ---------------------------------------------------------------------------

pub(super) struct RwState {
    writer: u64,
    writer_count: u32,
    reader_count: u32,
    waiting_writers: u32,
    readers: Vec<(u64, u32)>,
    sleepers: u32,
}

impl RwState {
    fn is_reader(&self, t: u64) -> bool {
        self.readers.iter().any(|r| r.0 == t)
    }

    fn add_reader(&mut self, t: u64) {
        match self.readers.iter_mut().find(|r| r.0 == t) {
            Some(r) => r.1 += 1,
            None => self.readers.push((t, 1)),
        }
        self.reader_count += 1;
    }

    fn remove_reader(&mut self, t: u64) -> bool {
        let Some(i) = self.readers.iter().position(|r| r.0 == t) else { return false };
        self.readers[i].1 -= 1;
        self.reader_count -= 1;
        if self.readers[i].1 == 0 {
            self.readers.remove(i);
        }
        true
    }

    /// A reader may enter while no writer holds the lock and none waits
    /// (unless it already reads).
    fn can_read(&self, t: u64) -> bool {
        self.writer_count == 0 && (self.waiting_writers == 0 || self.is_reader(t))
    }

    /// Whether `t` holds the lock for writing.
    fn is_writer(&self, t: u64) -> bool {
        self.writer_count > 0 && self.writer == t
    }
}

pub(super) struct HostRwlock {
    live: AtomicBool,
    state: Mutex<RwState>,
    cv: Condvar,
}

pub(super) static RWLOCKS: SlotTable<HostRwlock> = SlotTable::new();

fn new_rwlock(ctx: &CpuContext) -> Option<(u64, &'static HostRwlock)> {
    let obj = alloc_object(ctx)?;
    let l = RWLOCKS.get_or_create(obj, || HostRwlock {
        live: AtomicBool::new(false),
        state: Mutex::new(RwState { writer: 0, writer_count: 0, reader_count: 0, waiting_writers: 0, readers: Vec::new(), sleepers: 0 }),
        cv: Condvar::new(),
    })?;
    {
        let mut s = lock(&l.state);
        s.writer = 0;
        s.writer_count = 0;
        s.reader_count = 0;
        s.waiting_writers = 0;
        s.readers.clear();
    }
    l.live.store(true, Ordering::Release);
    Some((obj, l))
}

pub(super) fn resolve_rwlock(ctx: &CpuContext, ptr: u64) -> Result<&'static HostRwlock, HleResult> {
    resolve(ctx, ptr, &RWLOCKS, |v| v == 0, |l| l.live.load(Ordering::Acquire), |ctx, _| new_rwlock(ctx))
}

pub(super) fn cb_rwlock_init(ctx: &mut CpuContext) -> HleResult {
    let ptr = ctx.rdi();
    if ptr == 0 {
        return EINVAL;
    }
    let Some((obj, _)) = new_rwlock(ctx) else { return ENOMEM };
    if ctx.write_u64(ptr, obj) { OK } else { EINVAL }
}

/// What a destroyed rwlock leaves in its handle: every call on it is EINVAL
/// until it is initialised again.
pub(super) const RWLOCK_DESTROYED: u64 = 1;

/// scePthreadRwlockDestroy / pthread_rwlock_destroy, as measured on a PS5
/// (firmware 13.60): 0 whether the lock is free or held, leaving
/// `RWLOCK_DESTROYED` in the handle; a zero handle is 0, a destroyed one
/// EINVAL. A lock destroyed while held is not freed (its holders and
/// waiters keep a live object).
pub(super) fn cb_rwlock_destroy(ctx: &mut CpuContext) -> HleResult {
    let ptr = ctx.rdi();
    if ptr == 0 {
        return EINVAL;
    }
    let Some(val) = ctx.read_u64(ptr) else { return EINVAL };
    if val == 0 {
        return OK;
    }
    let Some(l) = RWLOCKS.get(val).filter(|l| l.live.load(Ordering::Acquire)) else { return EINVAL };
    let held = {
        let s = lock(&l.state);
        let held = s.writer_count > 0 || s.reader_count != 0;
        if !held {
            l.live.store(false, Ordering::Release);
        }
        held
    };
    if !held {
        free_object(val);
    }
    ctx.write_u64(ptr, RWLOCK_DESTROYED);
    OK
}

pub(super) fn rw_lock(l: &HostRwlock, write: bool, timeout_us: Option<u64>) -> HleResult {
    let me = current_handle();
    let deadline = timeout_us.map(|us| Instant::now() + Duration::from_micros(us));
    let mut s = lock(&l.state);
    let mut writer_waiting = false;
    loop {
        if write {
            if s.is_writer(me) {
                // The console's timed self-lock waits the whole time out
                // (measured on hardware) instead of failing at once.
                if let Some(d) = deadline {
                    while Instant::now() < d {
                        s = crate::signals::wait_cv(&l.state, &l.cv, s, Some(d)).0;
                    }
                    return ETIMEDOUT;
                }
                return EDEADLK;
            }
            if s.is_reader(me) {
                if writer_waiting {
                    s.waiting_writers -= 1;
                }
                return EDEADLK;
            }
            if s.writer_count == 0 && s.reader_count == 0 {
                if writer_waiting {
                    s.waiting_writers -= 1;
                }
                s.writer = me;
                s.writer_count = 1;
                return OK;
            }
            if !writer_waiting {
                s.waiting_writers += 1;
                writer_waiting = true;
            }
        } else if s.is_writer(me) {
            // The writer's own read lock (blocking or timed) would never be
            // granted; the try variant reports EBUSY instead. The timed one
            // waits its time out, as the console does.
            if let Some(d) = deadline {
                while Instant::now() < d {
                    s = crate::signals::wait_cv(&l.state, &l.cv, s, Some(d)).0;
                }
                return ETIMEDOUT;
            }
            return EDEADLK;
        } else if s.can_read(me) {
            s.add_reader(me);
            return OK;
        }
        if let Some(d) = deadline {
            if timeout_us == Some(0) || Instant::now() >= d {
                if writer_waiting {
                    s.waiting_writers -= 1;
                    // Readers held back by this writer may proceed now.
                    if s.sleepers > 0 {
                        l.cv.notify_all();
                    }
                }
                return ETIMEDOUT;
            }
        }
        s.sleepers += 1;
        s = crate::signals::wait_cv(&l.state, &l.cv, s, deadline).0;
        s.sleepers -= 1;
    }
}

pub(super) fn rw_tryrdlock(l: &HostRwlock) -> HleResult {
    let me = current_handle();
    let mut s = lock(&l.state);
    if s.can_read(me) {
        s.add_reader(me);
        OK
    } else {
        EBUSY
    }
}

pub(super) fn rw_trywrlock(l: &HostRwlock) -> HleResult {
    let me = current_handle();
    let mut s = lock(&l.state);
    if s.is_writer(me) || s.is_reader(me) {
        return EDEADLK;
    }
    if s.writer_count == 0 && s.reader_count == 0 {
        s.writer = me;
        s.writer_count = 1;
        return OK;
    }
    EBUSY
}

pub(super) fn rw_unlock(l: &HostRwlock) -> HleResult {
    let me = current_handle();
    let mut s = lock(&l.state);
    let released = if s.remove_reader(me) {
        true
    } else if s.writer_count > 0 && s.writer == me {
        s.writer_count -= 1;
        if s.writer_count == 0 {
            s.writer = 0;
        }
        true
    } else {
        return EPERM;
    };
    if released && s.sleepers > 0 {
        l.cv.notify_all();
    }
    OK
}

fn with_rwlock(ctx: &CpuContext, f: impl FnOnce(&HostRwlock) -> HleResult) -> HleResult {
    match resolve_rwlock(ctx, ctx.rdi()) {
        Ok(l) => f(l),
        Err(e) => e,
    }
}

pub(super) fn cb_rwlock_rdlock(ctx: &mut CpuContext) -> HleResult {
    with_rwlock(ctx, |l| rw_lock(l, false, None))
}

pub(super) fn cb_rwlock_wrlock(ctx: &mut CpuContext) -> HleResult {
    with_rwlock(ctx, |l| rw_lock(l, true, None))
}

pub(super) fn cb_rwlock_tryrdlock(ctx: &mut CpuContext) -> HleResult {
    with_rwlock(ctx, rw_tryrdlock)
}

pub(super) fn cb_rwlock_trywrlock(ctx: &mut CpuContext) -> HleResult {
    with_rwlock(ctx, rw_trywrlock)
}

/// scePthreadRwlockTimedrdlock(rwlock*, usec)
pub(super) fn cb_rwlock_timedrdlock(ctx: &mut CpuContext) -> HleResult {
    let us = ctx.rsi() as u32 as u64;
    with_rwlock(ctx, |l| rw_lock(l, false, Some(us)))
}

/// scePthreadRwlockTimedwrlock(rwlock*, usec)
pub(super) fn cb_rwlock_timedwrlock(ctx: &mut CpuContext) -> HleResult {
    let us = ctx.rsi() as u32 as u64;
    with_rwlock(ctx, |l| rw_lock(l, true, Some(us)))
}

pub(super) fn cb_rwlock_unlock(ctx: &mut CpuContext) -> HleResult {
    with_rwlock(ctx, rw_unlock)
}

/// Read/write lock attribute object: type at +0 (default 1).
pub(super) fn cb_rwlockattr_init(ctx: &mut CpuContext) -> HleResult {
    let ptr = ctx.rdi();
    if ptr == 0 {
        return EINVAL;
    }
    let Some(obj) = alloc_object(ctx) else { return ENOMEM };
    ctx.write_u32(obj, 1);
    if ctx.write_u64(ptr, obj) { OK } else { EINVAL }
}

pub(super) fn cb_rwlockattr_settype(ctx: &mut CpuContext) -> HleResult {
    let (ptr, kind) = (ctx.rdi(), ctx.rsi());
    let Some(obj) = attr_object(ctx, ptr) else { return EINVAL };
    ctx.write_u32(obj, kind as u32);
    OK
}

pub(super) fn cb_rwlockattr_gettype(ctx: &mut CpuContext) -> HleResult {
    let (ptr, out) = (ctx.rdi(), ctx.rsi());
    let Some(obj) = attr_object(ctx, ptr) else { return EINVAL };
    let v = ctx.read_u32(obj).unwrap_or(1);
    if out == 0 || !ctx.write_u32(out, v) { EINVAL } else { OK }
}

// ---------------------------------------------------------------------------
// Thread-specific keys
// ---------------------------------------------------------------------------

/// Number of keys; keys are handed out as 1..=KEYS_MAX (0 is never valid).
pub(super) const KEYS_MAX: usize = 256;
pub(super) const DESTRUCTOR_ITERATIONS: usize = 4;

/// A key is allocated while its generation is odd.
struct KeyState {
    generation: AtomicU64,
    destructor: AtomicU64,
}

static KEYS: [KeyState; KEYS_MAX] = [const { KeyState { generation: AtomicU64::new(0), destructor: AtomicU64::new(0) } }; KEYS_MAX];
static KEYS_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    /// (generation, value) per key index.
    static SPECIFIC: RefCell<Option<Box<[(u64, u64); KEYS_MAX]>>> = const { RefCell::new(None) };
}

fn key_index(key: u64) -> Option<usize> {
    let k = key as u32 as usize;
    (1..=KEYS_MAX).contains(&k).then(|| k - 1)
}

/// pthread_key_create(key*, destructor)
pub(super) fn cb_key_create(ctx: &mut CpuContext) -> HleResult {
    let (out, dtor) = (ctx.rdi(), ctx.rsi());
    if out == 0 {
        return EINVAL;
    }
    let _g = lock(&KEYS_LOCK);
    for (i, k) in KEYS.iter().enumerate() {
        let gen = k.generation.load(Ordering::Relaxed);
        if gen & 1 == 0 {
            if !ctx.write_u32(out, i as u32 + 1) {
                return EINVAL;
            }
            k.destructor.store(dtor, Ordering::Relaxed);
            k.generation.store(gen + 1, Ordering::Release);
            return OK;
        }
    }
    EAGAIN
}

pub(super) fn cb_key_delete(ctx: &mut CpuContext) -> HleResult {
    let Some(i) = key_index(ctx.rdi()) else { return EINVAL };
    let _g = lock(&KEYS_LOCK);
    let k = &KEYS[i];
    let gen = k.generation.load(Ordering::Relaxed);
    if gen & 1 == 0 {
        return EINVAL;
    }
    k.generation.store(gen + 1, Ordering::Release);
    k.destructor.store(0, Ordering::Relaxed);
    OK
}

/// pthread_setspecific(key, value)
pub(super) fn cb_setspecific(ctx: &mut CpuContext) -> HleResult {
    let value = ctx.rsi();
    let Some(i) = key_index(ctx.rdi()) else { return EINVAL };
    let gen = KEYS[i].generation.load(Ordering::Acquire);
    if gen & 1 == 0 {
        return EINVAL;
    }
    SPECIFIC.with(|s| {
        let mut s = s.borrow_mut();
        if s.is_none() {
            if value == 0 {
                return;
            }
            *s = Some(Box::new([(0, 0); KEYS_MAX]));
        }
        if let Some(t) = s.as_mut() {
            t[i] = (gen, value);
        }
    });
    OK
}

/// pthread_getspecific(key) -> value
pub(super) fn cb_getspecific(ctx: &mut CpuContext) -> HleResult {
    let Some(i) = key_index(ctx.rdi()) else { return 0 };
    let gen = KEYS[i].generation.load(Ordering::Acquire);
    if gen & 1 == 0 {
        return 0;
    }
    SPECIFIC.with(|s| match s.borrow().as_ref() {
        Some(t) if t[i].0 == gen => t[i].1 as i64,
        _ => 0,
    })
}

/// Take one round of `(destructor, value)` pairs for the calling thread's
/// non-null values of live keys; the values are cleared.
pub(super) fn take_key_destructors() -> Vec<(u64, u64)> {
    SPECIFIC.with(|s| {
        let mut s = s.borrow_mut();
        let Some(t) = s.as_mut() else { return Vec::new() };
        let mut out = Vec::new();
        for (i, slot) in t.iter_mut().enumerate() {
            let (gen, val) = std::mem::take(slot);
            if val == 0 {
                continue;
            }
            let k = &KEYS[i];
            let cur = k.generation.load(Ordering::Acquire);
            let dtor = k.destructor.load(Ordering::Relaxed);
            if cur & 1 != 0 && cur == gen && dtor != 0 {
                out.push((dtor, val));
            }
        }
        out
    })
}

/// Drop the calling thread's key values.
pub(super) fn clear_specific() {
    SPECIFIC.with(|s| *s.borrow_mut() = None);
}

// ---------------------------------------------------------------------------
// Once
// ---------------------------------------------------------------------------

const ONCE_NEEDS_INIT: u32 = 0;
const ONCE_DONE: u32 = 1;
const ONCE_IN_PROGRESS: u32 = 2;
const ONCE_WAIT: u32 = 3;

static ONCE_GATE: Mutex<()> = Mutex::new(());
static ONCE_CV: Condvar = Condvar::new();

thread_local! {
    /// Once controls whose initializer the calling thread is running.
    static ONCE_RUNNING: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

/// Give back the once controls whose initializer the calling thread was
/// running when it ends (pthread_exit or cancellation inside the
/// initializer): each goes back to "not run" and its waiters wake, one of
/// them to run it, as FreeBSD's once_cancel_handler does.
pub(crate) fn abandon_running_once(ctx: &CpuContext) {
    let running = ONCE_RUNNING.with(|r| std::mem::take(&mut *r.borrow_mut()));
    if running.is_empty() {
        return;
    }
    let _g = lock(&ONCE_GATE);
    for once in running {
        ctx.write_u32(once, ONCE_NEEDS_INIT);
    }
    ONCE_CV.notify_all();
}

/// pthread_once(once*, init): the control word is `{ int state; mutex }`.
pub(super) fn cb_once(ctx: &mut CpuContext) -> HleResult {
    let (once, init) = (ctx.rdi(), ctx.rsi());
    if once == 0 || init == 0 {
        return EINVAL;
    }
    run_once(ctx, once, || {
        ps5cpu::exec::call_guest_nested(init, [0; 6]);
    })
}

/// Run `init` once for the control word at `once`; other callers wait for
/// it. An initializer whose thread ends inside it leaves the control "not
/// run" (see `abandon_running_once`).
pub(super) fn run_once(ctx: &CpuContext, once: u64, init: impl FnOnce()) -> HleResult {
    let mut g = lock(&ONCE_GATE);
    loop {
        match ctx.read_u32(once) {
            Some(ONCE_DONE) => return OK,
            Some(ONCE_NEEDS_INIT) => break,
            Some(ONCE_IN_PROGRESS) | Some(ONCE_WAIT) => {
                ctx.write_u32(once, ONCE_WAIT);
                while ctx.read_u32(once) == Some(ONCE_WAIT) {
                    g = crate::signals::wait_cv(&ONCE_GATE, &ONCE_CV, g, None).0;
                }
            }
            _ => return EINVAL,
        }
    }
    ctx.write_u32(once, ONCE_IN_PROGRESS);
    drop(g);
    ONCE_RUNNING.with(|r| r.borrow_mut().push(once));
    init();
    let still_running = ONCE_RUNNING.with(|r| {
        let mut r = r.borrow_mut();
        let before = r.len();
        r.retain(|&o| o != once);
        r.len() != before
    });
    if still_running {
        let _g = lock(&ONCE_GATE);
        ctx.write_u32(once, ONCE_DONE);
        ONCE_CV.notify_all();
    }
    OK
}

#[cfg(test)]
impl HostCond {
    pub(super) fn waiters(&self) -> usize {
        lock(&self.state).waiters.len()
    }
}

#[cfg(test)]
impl HostRwlock {
    pub(super) fn waiting_writers(&self) -> u32 {
        lock(&self.state).waiting_writers
    }
}
