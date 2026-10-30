// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Kernel counting semaphores (`sceKernel*Sema`) and POSIX semaphores
//! (`sem_*`, `scePthreadSem*`).
//!
//! Kernel semaphores keep a waiter list, FIFO or by thread priority
//! (attribute 1 / 2). A signal hands tokens to the waiters in list order
//! (each one takes its full need when it fits), so a later arrival never
//! steals from an earlier waiter. Cancel and delete release every waiter
//! with ECANCELED / EACCES. Each blocked waiter sleeps on its own thread's
//! condvar, so a signal wakes exactly the waiters it handed tokens to
//! (`RPS5E_SEMA_TARGETED_WAKE=0`: one shared condvar, every signal wakes
//! all waiters). `sceKernelCreateSema` writes a 32-bit handle
//! into a 64-bit slot; an unknown handle is ESRCH.
//!
//! POSIX semaphores live at the guest `sem_t` address. The guest object
//! mirrors the state (16 bytes: u16 magic 0x09fa, u16 name id, u32
//! has-waiters, u32 count, u32 flags). `sem_*` set errno and return -1;
//! `scePthreadSem*` return 0x80020000 | errno.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use super::{
    read_name, remaining_usec, usec_deadline, wait_until, Export, Waited, ERR_BUSY, ERR_CANCELED,
    ERR_DELETED, ERR_INVALID_ARGUMENT, ERR_MEMORY_FAULT, ERR_TIMED_OUT, OK,
};
use crate::context::CpuContext;
use crate::registry::HleResult;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, RwLock};
use std::time::{Duration, Instant};

const MAX_NAME_LEN: usize = 128;

/// ESRCH: not a live semaphore.
const ERR_NOT_FOUND: HleResult = 0x8002_0003u32 as i32 as i64;

const POSIX_EAGAIN: i32 = 35;
const POSIX_EINVAL: i32 = 22;
const POSIX_ETIMEDOUT: i32 = 60;
const POSIX_EOVERFLOW: i32 = 84;

// ---------------------------------------------------------------------------
// Kernel semaphores
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WaitResult {
    Pending,
    Ok,
    Canceled,
    Deleted,
}

struct Waiter {
    id: u64,
    priority: i32,
    need: i32,
    result: WaitResult,
    /// The waiting thread's own condvar (targeted wakeups).
    cv: Option<Arc<Condvar>>,
}

impl Waiter {
    /// Settle a pending waiter and wake its thread if it sleeps on its own
    /// condvar.
    fn settle(&mut self, result: WaitResult) {
        self.result = result;
        if let Some(cv) = &self.cv {
            cv.notify_one();
        }
    }
}

thread_local! {
    /// Condvar a thread sleeps on while it waits on a kernel semaphore.
    static WAIT_CV: Arc<Condvar> = Arc::new(Condvar::new());
}

/// `RPS5E_SEMA_TARGETED_WAKE=0`: waiters share the semaphore condvar and
/// every signal wakes all of them.
fn targeted_wake_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("RPS5E_SEMA_TARGETED_WAKE").map_or(true, |v| !matches!(v.trim(), "0" | "off" | "false" | "no"))
    })
}

struct Inner {
    count: i32,
    deleted: bool,
    waiters: Vec<Waiter>,
    next_waiter: u64,
    /// Diagnostics: when the semaphore was last signalled.
    last_signal: Option<Instant>,
}

impl Inner {
    /// Hand tokens to the waiters in list order, waking the threads that
    /// sleep on their own condvar; returns how many got theirs.
    fn wake_waiters(&mut self) -> u32 {
        let mut count = self.count;
        let mut n = 0;
        for w in self.waiters.iter_mut() {
            if count == 0 {
                break;
            }
            if w.result == WaitResult::Pending && w.need <= count {
                count -= w.need;
                w.settle(WaitResult::Ok);
                n += 1;
            }
        }
        self.count = count;
        n
    }

    fn release_all(&mut self, result: WaitResult) -> u32 {
        let mut n = 0;
        for w in self.waiters.iter_mut().filter(|w| w.result == WaitResult::Pending) {
            w.settle(result);
            n += 1;
        }
        n
    }
}

struct Sema {
    name: String,
    fifo: bool,
    /// Waiters sleep on their own condvars (fixed at creation); otherwise
    /// all on `cv`.
    targeted: bool,
    initial: i32,
    max: i32,
    inner: Mutex<Inner>,
    cv: Condvar,
}

static SEMAS: RwLock<Option<HashMap<u32, Arc<Sema>>>> = RwLock::new(None);
static SEMAS_GENERATION: super::Generation = super::Generation::new();
static NEXT_HANDLE: AtomicU32 = AtomicU32::new(1);

thread_local! {
    static SEMA_LOOKUPS: super::LocalLookups<Sema> = const { super::LocalLookups::new() };
}

fn get(handle: u32) -> Option<Arc<Sema>> {
    let lookup = || SEMAS.read().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&handle).cloned();
    SEMA_LOOKUPS.try_with(|c| c.get(&SEMAS_GENERATION, handle as u64, lookup)).unwrap_or_else(|_| lookup())
}

fn lock_inner(s: &Sema) -> MutexGuard<'_, Inner> {
    s.inner.lock().unwrap_or_else(|e| e.into_inner())
}

pub(crate) fn create(name: String, fifo: bool, initial: i32, max: i32) -> u32 {
    create_with(name, fifo, initial, max, targeted_wake_enabled())
}

fn create_with(name: String, fifo: bool, initial: i32, max: i32, targeted: bool) -> u32 {
    let mut handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
    if handle == 0 {
        handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
    }
    let s = Arc::new(Sema {
        name,
        fifo,
        targeted,
        initial,
        max,
        inner: Mutex::new(Inner { count: initial, deleted: false, waiters: Vec::new(), next_waiter: 1, last_signal: None }),
        cv: Condvar::new(),
    });
    {
        let mut m = SEMAS.write().unwrap_or_else(|e| e.into_inner());
        m.get_or_insert_with(HashMap::new).insert(handle, s);
        SEMAS_GENERATION.bump();
    }
    handle
}

/// Diagnostics (`RPS5E_HLE_STUCK`): name, count and waiters of a
/// semaphore.
pub fn describe(handle: u32) -> Option<String> {
    let s = get(handle)?;
    let g = lock_inner(&s);
    let waiters: Vec<String> = g.waiters.iter().map(|w| format!("{}:{:?}", w.need, w.result)).collect();
    let last = g.last_signal.map_or("never".to_string(), |t| format!("{:.1} s ago", t.elapsed().as_secs_f64()));
    Some(format!("'{}' count {} max {} [{}] last signal {last}", s.name, g.count, s.max, waiters.join(" ")))
}

pub(crate) fn delete(handle: u32) -> bool {
    let removed = {
        let mut m = SEMAS.write().unwrap_or_else(|e| e.into_inner());
        let s = m.as_mut().and_then(|t| t.remove(&handle));
        SEMAS_GENERATION.bump();
        s
    };
    let Some(s) = removed else {
        return false;
    };
    let mut g = lock_inner(&s);
    g.deleted = true;
    g.release_all(WaitResult::Deleted);
    drop(g);
    if !s.targeted {
        s.cv.notify_all();
    }
    log::debug!("[sema] delete handle={handle:#x} name='{}'", s.name);
    true
}

/// Acquire `need` tokens, blocking until `deadline` (None = forever).
pub(crate) fn wait(handle: u32, need: i32, deadline: Option<Instant>) -> HleResult {
    let Some(s) = get(handle) else { return ERR_NOT_FOUND };
    if need < 1 || need > s.max {
        return ERR_INVALID_ARGUMENT;
    }
    let mut g = lock_inner(&s);
    if g.deleted {
        return ERR_DELETED;
    }
    if g.count >= need {
        g.count -= need;
        return OK;
    }
    if deadline.is_some_and(|d| Instant::now() >= d) {
        return ERR_TIMED_OUT;
    }
    let id = g.next_waiter;
    g.next_waiter += 1;
    let priority = if s.fifo { 0 } else { crate::pthread::current_priority() };
    let own_cv = s.targeted.then(|| WAIT_CV.with(|cv| cv.clone()));
    let cv: &Condvar = own_cv.as_deref().unwrap_or(&s.cv);
    let w = Waiter { id, priority, need, result: WaitResult::Pending, cv: own_cv.clone() };
    let pos = if s.fifo { g.waiters.len() } else { g.waiters.iter().position(|o| o.priority > priority).unwrap_or(g.waiters.len()) };
    g.waiters.insert(pos, w);
    let result = loop {
        let state = g.waiters.iter().find(|w| w.id == id).map_or(WaitResult::Deleted, |w| w.result);
        if state != WaitResult::Pending {
            break state;
        }
        let timed_out = match wait_until(&s.inner, cv, g, deadline, None) {
            Waited::Woken(ng) => {
                g = ng;
                false
            }
            Waited::TimedOut(ng) => {
                g = ng;
                true
            }
        };
        if timed_out {
            let state = g.waiters.iter().find(|w| w.id == id).map_or(WaitResult::Deleted, |w| w.result);
            if state == WaitResult::Pending {
                g.waiters.retain(|w| w.id != id);
                return ERR_TIMED_OUT;
            }
        }
    };
    g.waiters.retain(|w| w.id != id);
    match result {
        WaitResult::Ok => OK,
        WaitResult::Canceled => ERR_CANCELED,
        _ => ERR_DELETED,
    }
}

pub(crate) fn poll(handle: u32, need: i32) -> HleResult {
    let Some(s) = get(handle) else { return ERR_NOT_FOUND };
    if need < 1 || need > s.max {
        return ERR_INVALID_ARGUMENT;
    }
    let mut g = lock_inner(&s);
    if g.deleted || g.count < need {
        return ERR_BUSY;
    }
    g.count -= need;
    OK
}

pub(crate) fn signal(handle: u32, n: i32) -> HleResult {
    let Some(s) = get(handle) else { return ERR_NOT_FOUND };
    if n < 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let mut g = lock_inner(&s);
    if g.deleted || g.count > s.max - n {
        return ERR_INVALID_ARGUMENT;
    }
    g.count += n;
    g.last_signal = Some(Instant::now());
    g.wake_waiters();
    drop(g);
    if !s.targeted {
        s.cv.notify_all();
    }
    OK
}

/// Returns (result, number of threads that were waiting).
pub(crate) fn cancel(handle: u32, set_count: i32) -> (HleResult, u32) {
    let Some(s) = get(handle) else { return (ERR_NOT_FOUND, 0) };
    if set_count > s.max {
        return (ERR_INVALID_ARGUMENT, 0);
    }
    let mut g = lock_inner(&s);
    if g.deleted {
        return (ERR_INVALID_ARGUMENT, 0);
    }
    let waiting = g.release_all(WaitResult::Canceled);
    g.count = if set_count < 0 { s.initial } else { set_count };
    drop(g);
    if !s.targeted {
        s.cv.notify_all();
    }
    (OK, waiting)
}

#[cfg(test)]
fn value(handle: u32) -> Option<i32> {
    get(handle).map(|s| lock_inner(&s).count)
}

/// sceKernelCreateSema(SceKernelSema* sem, const char* name, u32 attr, int init, int max, opt*)
fn cb_create_sema(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    let name_ptr = ctx.rsi();
    let attr = ctx.rdx() as u32;
    let init = ctx.rcx() as i32;
    let max = ctx.r8() as i32;
    let opt = ctx.r9();
    if out == 0 || name_ptr == 0 || attr > 2 || init < 0 || max <= 0 || init > max || opt != 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(name) = read_name(ctx, name_ptr, MAX_NAME_LEN) else { return ERR_MEMORY_FAULT };
    log::debug!("[sema] create name='{name}' attr={attr:#x} init={init} max={max}");
    let handle = create(name, attr == 1, init, max);
    if !ctx.write_u64(out, handle as u64) {
        delete(handle);
        return ERR_MEMORY_FAULT;
    }
    OK
}

/// Wait with an optional in/out `SceKernelUseconds*` timeout: on return it
/// holds the time left (0 on timeout).
fn wait_with_timeout_ptr(ctx: &CpuContext, handle: u32, need: i32, timeout_ptr: u64) -> HleResult {
    let Some(s) = get(handle) else { return ERR_NOT_FOUND };
    if need < 1 || need > s.max {
        return ERR_INVALID_ARGUMENT;
    }
    let timeout = if timeout_ptr != 0 {
        match ctx.read_u32(timeout_ptr) {
            Some(t) => Some(t),
            None => return ERR_MEMORY_FAULT,
        }
    } else {
        None
    };
    // Early returns come before the timer starts: an available token leaves
    // the timeout untouched, and a zero timeout never blocks.
    if poll(handle, need) == OK {
        return OK;
    }
    if timeout == Some(0) {
        return ERR_TIMED_OUT;
    }
    let deadline = timeout.map(usec_deadline);
    let r = wait(handle, need, deadline);
    if let Some(d) = deadline {
        ctx.write_u32(timeout_ptr, if r == ERR_TIMED_OUT { 0 } else { remaining_usec(d) });
    }
    r
}

/// sceKernelWaitSema(sem, int need, SceKernelUseconds* timeout)
fn cb_wait_sema(ctx: &mut CpuContext) -> HleResult {
    wait_with_timeout_ptr(ctx, ctx.rdi() as u32, ctx.rsi() as i32, ctx.rdx())
}

/// sceKernelPollSema(sem, int need)
fn cb_poll_sema(ctx: &mut CpuContext) -> HleResult {
    poll(ctx.rdi() as u32, ctx.rsi() as i32)
}

/// sceKernelSignalSema(sem, int count)
fn cb_signal_sema(ctx: &mut CpuContext) -> HleResult {
    signal(ctx.rdi() as u32, ctx.rsi() as i32)
}

/// sceKernelCancelSema(sem, int set_count, int* waiting_threads)
fn cb_cancel_sema(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi() as u32;
    let out = ctx.rdx();
    if get(handle).is_none() {
        return ERR_NOT_FOUND;
    }
    let (r, waiting) = cancel(handle, ctx.rsi() as i32);
    if r == OK && out != 0 && !ctx.write_u32(out, waiting) {
        return ERR_MEMORY_FAULT;
    }
    r
}

/// sceKernelDeleteSema(sem)
fn cb_delete_sema(ctx: &mut CpuContext) -> HleResult {
    if delete(ctx.rdi() as u32) { OK } else { ERR_NOT_FOUND }
}

// ---------------------------------------------------------------------------
// POSIX semaphores
// ---------------------------------------------------------------------------

const POSIX_SEM_MAGIC: u16 = 0x09fa;
const POSIX_SEM_VALUE_MAX: i32 = 0x7fff_ffff;

struct PosixInner {
    count: i32,
    waiters: u32,
}

struct PosixSem {
    inner: Mutex<PosixInner>,
    cv: Condvar,
}

static POSIX_SEMS: RwLock<Option<HashMap<u64, Arc<PosixSem>>>> = RwLock::new(None);
static POSIX_SEMS_GENERATION: super::Generation = super::Generation::new();

thread_local! {
    static POSIX_SEM_LOOKUPS: super::LocalLookups<PosixSem> = const { super::LocalLookups::new() };
}

fn posix_get(sem: u64) -> Option<Arc<PosixSem>> {
    if sem == 0 {
        return None;
    }
    let lookup = || POSIX_SEMS.read().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&sem).cloned();
    POSIX_SEM_LOOKUPS.try_with(|c| c.get(&POSIX_SEMS_GENERATION, sem, lookup)).unwrap_or_else(|_| lookup())
}

/// Mirror the state into the guest `sem_t`.
fn sync_guest(ctx: &CpuContext, sem: u64, g: &PosixInner, flags: Option<u32>) {
    ctx.write_bytes(sem, &POSIX_SEM_MAGIC.to_le_bytes());
    ctx.write_bytes(sem + 2, &0u16.to_le_bytes());
    ctx.write_u32(sem + 4, (g.waiters > 0) as u32);
    ctx.write_u32(sem + 8, g.count as u32);
    if let Some(f) = flags {
        ctx.write_u32(sem + 12, f);
    }
}

fn posix_init(ctx: &CpuContext, sem: u64, pshared: u32, value: u64) -> i32 {
    if sem == 0 || value > POSIX_SEM_VALUE_MAX as u64 || !ps5vm::ranges::accessible(sem, 16, true) {
        return POSIX_EINVAL;
    }
    let s = Arc::new(PosixSem { inner: Mutex::new(PosixInner { count: value as i32, waiters: 0 }), cv: Condvar::new() });
    sync_guest(ctx, sem, &s.inner.lock().unwrap(), Some(pshared));
    {
        let mut m = POSIX_SEMS.write().unwrap_or_else(|e| e.into_inner());
        m.get_or_insert_with(HashMap::new).insert(sem, s);
        POSIX_SEMS_GENERATION.bump();
    }
    0
}

fn posix_destroy(ctx: &CpuContext, sem: u64) -> i32 {
    if sem == 0 {
        return POSIX_EINVAL;
    }
    let removed = {
        let mut m = POSIX_SEMS.write().unwrap_or_else(|e| e.into_inner());
        let s = m.as_mut().and_then(|m| m.remove(&sem));
        POSIX_SEMS_GENERATION.bump();
        s
    };
    if removed.is_none() {
        return POSIX_EINVAL;
    }
    ctx.write_bytes(sem, &[0u8; 16]);
    0
}

/// `posix_wait_at` result: the thread has a cancellation request to act on
/// and took no token.
const WAIT_CANCELLED: i32 = -1;

/// Wait for one token; `timeout_us` None = forever. The wait is a
/// cancellation point, checked on entry (a pending request is acted on
/// even when a token is available, and the token stays) and while blocked.
fn posix_wait(ctx: &CpuContext, sem: u64, timeout_us: Option<u32>) -> i32 {
    loop {
        let r = posix_wait_at(ctx, sem, timeout_us, &crate::kernel_ext::cancel_requested);
        if r != WAIT_CANCELLED {
            return r;
        }
        // Ends the thread; returns only when the request went away.
        crate::kernel_ext::test_cancel();
    }
}

/// One wait attempt; `cancelled` reports a cancellation request, which
/// ends the attempt with `WAIT_CANCELLED` and no lock held.
fn posix_wait_at(ctx: &CpuContext, sem: u64, timeout_us: Option<u32>, cancelled: &dyn Fn() -> bool) -> i32 {
    let Some(s) = posix_get(sem) else { return POSIX_EINVAL };
    if cancelled() {
        return WAIT_CANCELLED;
    }
    let deadline = timeout_us.map(|us| Instant::now() + Duration::from_micros(us as u64));
    let mut g = s.inner.lock().unwrap_or_else(|e| e.into_inner());
    while g.count <= 0 {
        if cancelled() {
            sync_guest(ctx, sem, &g, None);
            return WAIT_CANCELLED;
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            sync_guest(ctx, sem, &g, None);
            return POSIX_ETIMEDOUT;
        }
        g.waiters += 1;
        sync_guest(ctx, sem, &g, None);
        g = crate::signals::wait_cv(&s.inner, &s.cv, g, deadline).0;
        g.waiters -= 1;
    }
    g.count -= 1;
    sync_guest(ctx, sem, &g, None);
    0
}

fn posix_trywait(ctx: &CpuContext, sem: u64) -> i32 {
    let Some(s) = posix_get(sem) else { return POSIX_EINVAL };
    let mut g = s.inner.lock().unwrap_or_else(|e| e.into_inner());
    if g.count <= 0 {
        sync_guest(ctx, sem, &g, None);
        return POSIX_EAGAIN;
    }
    g.count -= 1;
    sync_guest(ctx, sem, &g, None);
    0
}

fn posix_post(ctx: &CpuContext, sem: u64) -> i32 {
    let Some(s) = posix_get(sem) else { return POSIX_EINVAL };
    let mut g = s.inner.lock().unwrap_or_else(|e| e.into_inner());
    if g.count == POSIX_SEM_VALUE_MAX {
        return POSIX_EOVERFLOW;
    }
    g.count += 1;
    sync_guest(ctx, sem, &g, None);
    let wake = g.waiters > 0;
    drop(g);
    if wake {
        s.cv.notify_one();
    }
    0
}

fn posix_getvalue(sem: u64) -> Option<i32> {
    posix_get(sem).map(|s| s.inner.lock().unwrap_or_else(|e| e.into_inner()).count)
}

/// Absolute CLOCK_REALTIME timespec -> relative microseconds (clamped);
/// None for an invalid one.
fn abs_to_micros(ctx: &CpuContext, ts: u64) -> Option<u32> {
    let sec = ctx.read_u64(ts)? as i64;
    let nsec = ctx.read_u64(ts + 8)? as i64;
    if !(0..1_000_000_000).contains(&nsec) {
        return None;
    }
    // A deadline before the epoch has passed: ETIMEDOUT, not EINVAL (PS5
    // firmware 13.60).
    if sec < 0 {
        return Some(0);
    }
    let now = crate::pthread::realtime_now();
    let delta = (sec - now.as_secs() as i64) * 1_000_000 + (nsec - now.subsec_nanos() as i64) / 1000;
    Some(delta.clamp(0, u32::MAX as i64) as u32)
}

/// POSIX result: 0, or errno and -1.
fn px(ctx: &CpuContext, errno: i32) -> HleResult {
    if errno == 0 {
        0
    } else {
        crate::kernel::set_errno(ctx, errno);
        -1
    }
}

/// scePthreadSem* result: 0x80020000 | errno.
fn kr(errno: i32) -> HleResult {
    if errno == 0 { OK } else { (0x8002_0000u32 as i32 + errno) as i64 }
}

/// sem_init(sem_t*, int pshared, unsigned value)
fn cb_sem_init(ctx: &mut CpuContext) -> HleResult {
    let e = posix_init(ctx, ctx.rdi(), ctx.rsi() as u32, ctx.rdx() as u32 as u64);
    px(ctx, e)
}

/// sem_destroy(sem)
fn cb_sem_destroy(ctx: &mut CpuContext) -> HleResult {
    let e = posix_destroy(ctx, ctx.rdi());
    px(ctx, e)
}

/// sem_wait(sem)
fn cb_sem_wait(ctx: &mut CpuContext) -> HleResult {
    let e = posix_wait(ctx, ctx.rdi(), None);
    px(ctx, e)
}

/// sem_trywait(sem)
fn cb_sem_trywait(ctx: &mut CpuContext) -> HleResult {
    let e = posix_trywait(ctx, ctx.rdi());
    px(ctx, e)
}

/// sem_reltimedwait_np(sem, useconds)
fn cb_sem_reltimedwait_np(ctx: &mut CpuContext) -> HleResult {
    let e = posix_wait(ctx, ctx.rdi(), Some(ctx.rsi() as u32));
    px(ctx, e)
}

/// sem_timedwait(sem, const struct timespec* abstime): a cancellation
/// point first; an available count is taken whatever the deadline says, a
/// null deadline waits without limit, an invalid one is EINVAL.
fn cb_sem_timedwait(ctx: &mut CpuContext) -> HleResult {
    crate::kernel_ext::test_cancel();
    let e = sem_timedwait(ctx, ctx.rdi(), ctx.rsi());
    px(ctx, e)
}

fn sem_timedwait(ctx: &CpuContext, sem: u64, abstime: u64) -> i32 {
    match posix_trywait(ctx, sem) {
        POSIX_EAGAIN => {}
        r => return r,
    }
    if abstime == 0 {
        return posix_wait(ctx, sem, None);
    }
    let Some(us) = abs_to_micros(ctx, abstime) else { return POSIX_EINVAL };
    posix_wait(ctx, sem, Some(us))
}

/// sem_post(sem)
fn cb_sem_post(ctx: &mut CpuContext) -> HleResult {
    let e = posix_post(ctx, ctx.rdi());
    px(ctx, e)
}

/// sem_getvalue(sem, int* value)
fn cb_sem_getvalue(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    match posix_getvalue(ctx.rdi()) {
        Some(v) if out != 0 && ctx.write_u32(out, v as u32) => 0,
        _ => px(ctx, POSIX_EINVAL),
    }
}

/// scePthreadSemInit(sem, int flag, unsigned value, const char* name):
/// only private semaphores (flag 0).
fn cb_pthread_sem_init(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() != 0 {
        return ERR_INVALID_ARGUMENT;
    }
    kr(posix_init(ctx, ctx.rdi(), 0, ctx.rdx() as u32 as u64))
}

fn cb_pthread_sem_destroy(ctx: &mut CpuContext) -> HleResult {
    kr(posix_destroy(ctx, ctx.rdi()))
}

fn cb_pthread_sem_wait(ctx: &mut CpuContext) -> HleResult {
    kr(posix_wait(ctx, ctx.rdi(), None))
}

fn cb_pthread_sem_trywait(ctx: &mut CpuContext) -> HleResult {
    kr(posix_trywait(ctx, ctx.rdi()))
}

/// scePthreadSemTimedwait(sem, SceKernelUseconds usec)
fn cb_pthread_sem_timedwait(ctx: &mut CpuContext) -> HleResult {
    kr(posix_wait(ctx, ctx.rdi(), Some(ctx.rsi() as u32)))
}

fn cb_pthread_sem_post(ctx: &mut CpuContext) -> HleResult {
    kr(posix_post(ctx, ctx.rdi()))
}

/// scePthreadSemGetvalue(sem, int* value)
fn cb_pthread_sem_getvalue(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    match posix_getvalue(ctx.rdi()) {
        Some(v) if out != 0 && ctx.write_u32(out, v as u32) => OK,
        _ => ERR_INVALID_ARGUMENT,
    }
}

pub(crate) static EXPORTS: &[Export] = &[
    ("188x57JYp0g", "sceKernelCreateSema", "libKernel", cb_create_sema),
    ("Zxa0VhQVTsk", "sceKernelWaitSema", "libKernel", cb_wait_sema),
    ("12wOHk8ywb0", "sceKernelPollSema", "libKernel", cb_poll_sema),
    ("4czppHBiriw", "sceKernelSignalSema", "libKernel", cb_signal_sema),
    ("4DM06U2BNEY", "sceKernelCancelSema", "libKernel", cb_cancel_sema),
    ("R1Jvn8bSCW8", "sceKernelDeleteSema", "libKernel", cb_delete_sema),
    ("pDuPEf3m4fI", "sem_init", "libKernel", cb_sem_init),
    ("cDW233RAwWo", "sem_destroy", "libKernel", cb_sem_destroy),
    ("YCV5dGGBcCo", "sem_wait", "libKernel", cb_sem_wait),
    ("WBWzsRifCEA", "sem_trywait", "libKernel", cb_sem_trywait),
    ("4SbrhCozqQU", "sem_reltimedwait_np", "libKernel", cb_sem_reltimedwait_np),
    ("w5IHyvahg-o", "sem_timedwait", "libKernel", cb_sem_timedwait),
    ("IKP8typ0QUk", "sem_post", "libKernel", cb_sem_post),
    ("Bq+LRV-N6Hk", "sem_getvalue", "libKernel", cb_sem_getvalue),
    ("GEnUkDZoUwY", "scePthreadSemInit", "libKernel", cb_pthread_sem_init),
    ("Vwc+L05e6oE", "scePthreadSemDestroy", "libKernel", cb_pthread_sem_destroy),
    ("C36iRE0F5sE", "scePthreadSemWait", "libKernel", cb_pthread_sem_wait),
    ("H2a+IN9TP0E", "scePthreadSemTrywait", "libKernel", cb_pthread_sem_trywait),
    ("fjN6NQHhK8k", "scePthreadSemTimedwait", "libKernel", cb_pthread_sem_timedwait),
    ("aishVAiFaYM", "scePthreadSemPost", "libKernel", cb_pthread_sem_post),
    ("DjpBvGlaWbQ", "scePthreadSemGetvalue", "libKernel", cb_pthread_sem_getvalue),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::test_util::vm_arena;
    use crate::CpuRegister;
    use std::thread;

    #[test]
    fn counts_poll_signal_limits() {
        let h = create("t".into(), true, 2, 3);
        assert_eq!(poll(h, 2), OK);
        assert_eq!(poll(h, 1), ERR_BUSY);
        assert_eq!(poll(h, 4), ERR_INVALID_ARGUMENT);
        assert_eq!(signal(h, 3), OK);
        assert_eq!(signal(h, 1), ERR_INVALID_ARGUMENT); // would exceed max
        assert_eq!(signal(h, -1), ERR_INVALID_ARGUMENT);
        // A zero count adds nothing and succeeds, full or empty.
        assert_eq!(signal(h, 0), OK);
        assert_eq!(value(h), Some(3));
        assert_eq!(wait(h, 3, None), OK);
        assert_eq!(value(h), Some(0));
        assert_eq!(signal(h, 0), OK);
        assert_eq!(poll(h, 1), ERR_BUSY);
        assert!(delete(h));
        assert_eq!(poll(h, 1), ERR_NOT_FOUND);
    }

    #[test]
    fn wait_blocks_until_signal_and_times_out() {
        let h = create("t".into(), true, 0, 10);
        let start = Instant::now();
        assert_eq!(wait(h, 1, Some(usec_deadline(20_000))), ERR_TIMED_OUT);
        assert!(start.elapsed() >= Duration::from_millis(20));
        let t = thread::spawn(move || wait(h, 2, None));
        thread::sleep(Duration::from_millis(20));
        assert_eq!(signal(h, 1), OK);
        thread::sleep(Duration::from_millis(10));
        assert!(!t.is_finished());
        assert_eq!(signal(h, 1), OK);
        assert_eq!(t.join().unwrap(), OK);
        assert_eq!(value(h), Some(0));
    }

    #[test]
    fn signal_hands_tokens_to_waiters_in_order() {
        let h = create("t".into(), true, 0, 10);
        let first = thread::spawn(move || wait(h, 3, None));
        thread::sleep(Duration::from_millis(20));
        let second = thread::spawn(move || wait(h, 1, None));
        thread::sleep(Duration::from_millis(20));
        // Two tokens fit only the second waiter; the first keeps waiting.
        assert_eq!(signal(h, 2), OK);
        assert_eq!(second.join().unwrap(), OK);
        assert_eq!(value(h), Some(1));
        // A poll may still take what is left; the queued waiter needs 3.
        assert_eq!(signal(h, 2), OK);
        assert_eq!(first.join().unwrap(), OK);
        assert_eq!(value(h), Some(0));
    }

    #[test]
    fn cancel_and_delete_release_waiters() {
        let h = create("t".into(), true, 1, 10);
        assert_eq!(poll(h, 1), OK);
        let waiters: Vec<_> = (0..2).map(|_| thread::spawn(move || wait(h, 1, None))).collect();
        thread::sleep(Duration::from_millis(30));
        assert_eq!(cancel(h, 11), (ERR_INVALID_ARGUMENT, 0));
        assert_eq!(cancel(h, -1), (OK, 2));
        for w in waiters {
            assert_eq!(w.join().unwrap(), ERR_CANCELED);
        }
        assert_eq!(value(h), Some(1)); // reset to the initial count
        assert_eq!(poll(h, 1), OK);
        let w = thread::spawn(move || wait(h, 1, None));
        thread::sleep(Duration::from_millis(20));
        assert!(delete(h));
        assert_eq!(w.join().unwrap(), ERR_DELETED);
        assert_eq!(wait(h, 1, None), ERR_NOT_FOUND);
    }

    #[test]
    fn posix_semaphores() {
        const BASE: u64 = 0x5CF2_0000_0000;
        let vm = vm_arena(BASE);
        let mut ctx = CpuContext::new(vm);
        ctx.set(CpuRegister::Rdi, BASE);
        ctx.set(CpuRegister::Rsi, 0);
        ctx.set(CpuRegister::Rdx, 0x8000_0000);
        assert_eq!(cb_sem_init(&mut ctx), -1, "value above SEM_VALUE_MAX");
        ctx.set(CpuRegister::Rdx, 1);
        assert_eq!(cb_sem_init(&mut ctx), 0);
        // Guest mirror: magic, has_waiters, count.
        assert_eq!(ctx.read_bytes(BASE, 2), Some(vec![0xfa, 0x09]));
        assert_eq!(ctx.read_u32(BASE + 8), Some(1));
        assert_eq!(cb_sem_trywait(&mut ctx), 0);
        assert_eq!(cb_sem_trywait(&mut ctx), -1);
        assert_eq!(ctx.read_u32(BASE + 8), Some(0));
        ctx.set(CpuRegister::Rsi, 1000);
        assert_eq!(cb_sem_reltimedwait_np(&mut ctx), -1);
        assert_eq!(cb_pthread_sem_timedwait(&mut ctx), kr(POSIX_ETIMEDOUT));
        assert_eq!(cb_pthread_sem_trywait(&mut ctx), kr(POSIX_EAGAIN));
        // Absolute deadline in the past times out; a bad one is EINVAL.
        ctx.write_u64(BASE + 0x40, 1);
        ctx.write_u64(BASE + 0x48, 0);
        ctx.set(CpuRegister::Rsi, BASE + 0x40);
        assert_eq!(cb_sem_timedwait(&mut ctx), -1);
        ctx.write_u64(BASE + 0x48, 1_000_000_000);
        assert_eq!(cb_sem_timedwait(&mut ctx), -1);
        assert_eq!(sem_timedwait(&ctx, BASE, BASE + 0x40), POSIX_EINVAL);
        ctx.write_u64(BASE + 0x40, (-5i64) as u64);
        ctx.write_u64(BASE + 0x48, 0);
        assert_eq!(sem_timedwait(&ctx, BASE, BASE + 0x40), POSIX_ETIMEDOUT, "negative tv_sec");
        ctx.write_u64(BASE + 0x48, (-1i64) as u64);
        assert_eq!(sem_timedwait(&ctx, BASE, BASE + 0x40), POSIX_EINVAL, "negative tv_nsec");
        ctx.write_u64(BASE + 0x40, 1);
        ctx.write_u64(BASE + 0x48, 1_000_000_000);
        // An available count is taken whatever the deadline says.
        assert_eq!(cb_sem_post(&mut ctx), 0);
        assert_eq!(cb_sem_timedwait(&mut ctx), 0, "available, invalid deadline");
        assert_eq!(cb_sem_post(&mut ctx), 0);
        ctx.set(CpuRegister::Rsi, 0);
        assert_eq!(cb_sem_timedwait(&mut ctx), 0, "available, null deadline");
        // A null deadline waits until a post.
        let t = thread::spawn(move || sem_timedwait(&CpuContext::new(vm), BASE, 0));
        thread::sleep(Duration::from_millis(20));
        assert!(!t.is_finished(), "a null deadline does not time out");
        assert_eq!(cb_sem_post(&mut ctx), 0);
        assert_eq!(t.join().unwrap(), 0);
        assert_eq!(ctx.read_u32(BASE + 8), Some(0));
        // A blocked waiter is released by post and shows in the mirror.
        let t = thread::spawn(move || posix_wait(&CpuContext::new(vm), BASE, None));
        thread::sleep(Duration::from_millis(20));
        assert_eq!(ctx.read_u32(BASE + 4), Some(1), "has_waiters");
        assert_eq!(cb_sem_post(&mut ctx), 0);
        assert_eq!(t.join().unwrap(), 0);
        assert_eq!(cb_pthread_sem_post(&mut ctx), OK);
        ctx.set(CpuRegister::Rsi, BASE + 0x10);
        assert_eq!(cb_sem_getvalue(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x10), Some(1));
        assert_eq!(cb_pthread_sem_getvalue(&mut ctx), OK);
        assert_eq!(cb_sem_wait(&mut ctx), 0);
        assert_eq!(cb_sem_destroy(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 8), Some(0));
        assert_eq!(cb_sem_destroy(&mut ctx), -1);
        assert_eq!(cb_sem_post(&mut ctx), -1);
        assert_eq!(cb_pthread_sem_wait(&mut ctx), kr(POSIX_EINVAL));
        ctx.set(CpuRegister::Rsi, 1);
        assert_eq!(cb_pthread_sem_init(&mut ctx), ERR_INVALID_ARGUMENT);
    }

    /// A cancelled thread leaves sem_wait without taking the token, even
    /// when one is available, and a blocked waiter leaves when the request
    /// arrives.
    #[test]
    fn cancelled_semaphore_wait_takes_no_token() {
        const BASE: u64 = 0x5F82_0000_0000;
        let vm = vm_arena(BASE);
        let ctx = CpuContext::new(vm);
        assert_eq!(posix_init(&ctx, BASE, 0, 1), 0);
        for timeout in [None, Some(60_000_000)] {
            assert_eq!(posix_wait_at(&ctx, BASE, timeout, &|| true), WAIT_CANCELLED);
            assert_eq!(posix_getvalue(BASE), Some(1), "a cancelled wait took the token");
        }
        assert_eq!(posix_wait_at(&ctx, BASE, None, &|| false), 0);
        assert_eq!(posix_getvalue(BASE), Some(0));
        // Blocked with no token: the request ends the wait.
        let requested = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = requested.clone();
        let t = thread::spawn(move || {
            posix_wait_at(&CpuContext::new(vm), BASE, None, &|| flag.load(std::sync::atomic::Ordering::SeqCst))
        });
        thread::sleep(Duration::from_millis(20));
        requested.store(true, std::sync::atomic::Ordering::SeqCst);
        // The waiter re-checks on its slice wakeups (signals::wait_cv).
        assert_eq!(t.join().unwrap(), WAIT_CANCELLED);
        assert_eq!(posix_getvalue(BASE), Some(0));
        assert_eq!(posix_destroy(&ctx, BASE), 0);
    }

    #[test]
    fn kernel_sema_callbacks() {
        const BASE: u64 = 0x5CF3_0000_0000;
        let vm = vm_arena(BASE);
        let mut ctx = CpuContext::new(vm);
        ctx.write_bytes(BASE + 0x200, b"sema\0");
        ctx.set(CpuRegister::Rdi, BASE + 0x100);
        ctx.set(CpuRegister::Rsi, BASE + 0x200);
        ctx.set(CpuRegister::Rdx, 1);
        ctx.set(CpuRegister::Rcx, 0);
        ctx.set(CpuRegister::R8, 4);
        assert_eq!(cb_create_sema(&mut ctx), OK);
        let h = ctx.read_u64(BASE + 0x100).unwrap();
        ctx.write_u32(BASE + 0x20, 5_000);
        ctx.regs = [0; 16];
        ctx.set(CpuRegister::Rdi, h);
        ctx.set(CpuRegister::Rsi, 1);
        ctx.set(CpuRegister::Rdx, BASE + 0x20);
        assert_eq!(cb_wait_sema(&mut ctx), ERR_TIMED_OUT);
        assert_eq!(ctx.read_u32(BASE + 0x20), Some(0));
        assert_eq!(cb_signal_sema(&mut ctx), OK);
        ctx.write_u32(BASE + 0x20, 5_000_000);
        assert_eq!(cb_wait_sema(&mut ctx), OK);
        // A token that is already there leaves the timeout untouched.
        assert_eq!(cb_signal_sema(&mut ctx), OK);
        ctx.write_u32(BASE + 0x20, 1234);
        assert_eq!(cb_wait_sema(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE + 0x20), Some(1234));
        // A zero timeout fails at once without blocking.
        ctx.write_u32(BASE + 0x20, 0);
        let start = Instant::now();
        assert_eq!(cb_wait_sema(&mut ctx), ERR_TIMED_OUT);
        assert!(start.elapsed() < Duration::from_millis(5));
        assert_eq!(ctx.read_u32(BASE + 0x20), Some(0));
        assert_eq!(cb_delete_sema(&mut ctx), OK);
        assert_eq!(cb_delete_sema(&mut ctx), ERR_NOT_FOUND);
    }

    /// Waits of the calling thread wake only when notified, so a missed
    /// wakeup shows as a hang instead of a 10 ms delay.
    fn long_slices() {
        crate::signals::TEST_SLICE.with(|s| s.set(Some(Duration::from_secs(60))));
    }

    fn wait_for_waiters(h: u32, n: usize) {
        let t0 = Instant::now();
        while get(h).map_or(0, |s| lock_inner(&s).waiters.len()) != n {
            assert!(t0.elapsed() < Duration::from_secs(10), "waiters never queued");
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// signal(n) releases exactly the waiters it has tokens for, at once,
    /// in both wake modes; the others keep waiting.
    #[test]
    fn signal_n_releases_exactly_n_waiters() {
        for targeted in [true, false] {
            let h = create_with("t".into(), true, 0, 100, targeted);
            let done = Arc::new(AtomicU32::new(0));
            let ts: Vec<_> = (0..6)
                .map(|_| {
                    let done = done.clone();
                    thread::spawn(move || {
                        long_slices();
                        let r = wait(h, 1, None);
                        done.fetch_add(1, Ordering::SeqCst);
                        r
                    })
                })
                .collect();
            wait_for_waiters(h, 6);
            assert_eq!(signal(h, 4), OK);
            let t0 = Instant::now();
            while done.load(Ordering::SeqCst) < 4 {
                assert!(t0.elapsed() < Duration::from_secs(5), "targeted={targeted}: woken waiters did not run");
                thread::sleep(Duration::from_millis(1));
            }
            thread::sleep(Duration::from_millis(20));
            assert_eq!(done.load(Ordering::SeqCst), 4, "targeted={targeted}");
            assert_eq!(value(h), Some(0));
            assert_eq!(signal(h, 2), OK);
            for t in ts {
                assert_eq!(t.join().unwrap(), OK);
            }
            assert!(t0.elapsed() < Duration::from_secs(5), "targeted={targeted}: a waiter missed its wakeup");
            // Different needs: a signal serves every waiter it fits, in order.
            let big = thread::spawn(move || {
                long_slices();
                wait(h, 3, None)
            });
            wait_for_waiters(h, 1);
            let small = thread::spawn(move || {
                long_slices();
                wait(h, 1, None)
            });
            wait_for_waiters(h, 2);
            assert_eq!(signal(h, 2), OK);
            assert_eq!(small.join().unwrap(), OK);
            assert_eq!(value(h), Some(1));
            assert_eq!(signal(h, 2), OK);
            assert_eq!(big.join().unwrap(), OK);
            assert_eq!(value(h), Some(0));
            assert!(delete(h));
        }
    }

    /// Many waiters with mixed needs and untimed waits against signals of
    /// mixed sizes: every waiter gets its tokens, none misses its wakeup,
    /// and no token is lost or made up.
    #[test]
    fn stress_no_lost_wakeups() {
        for targeted in [true, false] {
            let h = create_with("t".into(), targeted, 0, 1 << 20, targeted);
            const PER_WAITER: u32 = 1_500;
            let t0 = Instant::now();
            let waiters: Vec<_> = (0..8u32)
                .map(|w| {
                    thread::spawn(move || {
                        long_slices();
                        let mut taken = 0u32;
                        for i in 0..PER_WAITER {
                            let need = 1 + ((i + w) % 3) as i32;
                            assert_eq!(wait(h, need, None), OK);
                            taken += need as u32;
                        }
                        taken
                    })
                })
                .collect();
            let total: u32 = (0..8u32).map(|w| (0..PER_WAITER).map(|i| 1 + (i + w) % 3).sum::<u32>()).sum();
            let mut sent = 0u32;
            let mut i = 0u32;
            while sent < total {
                let n = (1 + i % 4).min(total - sent);
                i += 1;
                assert_eq!(signal(h, n as i32), OK);
                sent += n;
                if i % 64 == 0 {
                    thread::yield_now();
                }
            }
            let taken: u32 = waiters.into_iter().map(|t| t.join().unwrap()).sum();
            assert_eq!(taken, total);
            assert_eq!(value(h), Some(0));
            assert!(t0.elapsed() < Duration::from_secs(30), "targeted={targeted}: a waiter missed its wakeup");
            assert!(delete(h));
        }
    }

    /// Timed waits that give up race with signals: a waiter either takes its
    /// whole need or nothing, so the tokens taken plus those left equal the
    /// tokens signalled.
    #[test]
    fn stress_timeouts_keep_every_token() {
        for targeted in [true, false] {
            let h = create_with("t".into(), true, 0, 1 << 20, targeted);
            let waiters: Vec<_> = (0..6u64)
                .map(|w| {
                    thread::spawn(move || {
                        let mut taken = 0u64;
                        let mut timeouts = 0u64;
                        for i in 0..1_500u64 {
                            let need = 1 + ((i + w) % 2) as i32;
                            match wait(h, need, Some(usec_deadline(50 + ((i * 37 + w * 11) % 400) as u32))) {
                                OK => taken += need as u64,
                                ERR_TIMED_OUT => timeouts += 1,
                                r => panic!("unexpected {r:#x}"),
                            }
                        }
                        (taken, timeouts)
                    })
                })
                .collect();
            let mut sent = 0u64;
            let mut i = 0u64;
            while !waiters.iter().all(|t| t.is_finished()) {
                let n = 1 + i % 3;
                i += 1;
                assert_eq!(signal(h, n as i32), OK);
                sent += n;
                thread::sleep(Duration::from_micros(30));
            }
            let (taken, timeouts) =
                waiters.into_iter().map(|t| t.join().unwrap()).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
            assert_eq!(taken + value(h).unwrap() as u64, sent, "targeted={targeted}: tokens lost or made up");
            assert!(taken > 0, "no waiter ever got tokens");
            let _ = timeouts;
            assert!(delete(h));
        }
    }

    /// Cancel and delete release waiters that sleep on their own condvars.
    #[test]
    fn targeted_cancel_and_delete_release_waiters() {
        let h = create_with("t".into(), false, 0, 10, true);
        let ts: Vec<_> = (0..3)
            .map(|_| {
                thread::spawn(move || {
                    long_slices();
                    wait(h, 2, None)
                })
            })
            .collect();
        wait_for_waiters(h, 3);
        assert_eq!(cancel(h, 1), (OK, 3));
        for t in ts {
            assert_eq!(t.join().unwrap(), ERR_CANCELED);
        }
        assert_eq!(value(h), Some(1));
        let t = thread::spawn(move || {
            long_slices();
            wait(h, 2, None)
        });
        wait_for_waiters(h, 1);
        assert!(delete(h));
        assert_eq!(t.join().unwrap(), ERR_DELETED);
    }

    /// One signaller handing single tokens to `waiters` waiting threads.
    /// Returns wall and process CPU ns per token.
    fn sema_handoff_ns(targeted: bool, waiters: u32, per_waiter: u32) -> (f64, f64) {
        let h = create_with("bench".into(), true, 0, 1 << 30, targeted);
        let ts: Vec<_> = (0..waiters)
            .map(|_| {
                thread::spawn(move || {
                    for _ in 0..per_waiter {
                        assert_eq!(wait(h, 1, None), OK);
                    }
                })
            })
            .collect();
        wait_for_waiters(h, waiters as usize);
        let cpu0 = cpu_time();
        let t = Instant::now();
        for _ in 0..waiters * per_waiter {
            assert_eq!(signal(h, 1), OK);
            // Give the woken waiter time to take its token, as a producer
            // doing real work between signals would.
            let spin = Instant::now();
            while spin.elapsed() < Duration::from_micros(2) {
                std::hint::spin_loop();
            }
        }
        for t in ts {
            t.join().unwrap();
        }
        let n = (waiters * per_waiter) as f64;
        let ns = t.elapsed().as_nanos() as f64 / n;
        let cpu = (cpu_time() - cpu0).as_nanos() as f64 / n;
        assert!(delete(h));
        (ns, cpu)
    }

    /// CPU time of the whole process.
    fn cpu_time() -> Duration {
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
        let tv = |t: libc::timeval| Duration::new(t.tv_sec as u64, t.tv_usec as u32 * 1000);
        tv(ru.ru_utime) + tv(ru.ru_stime)
    }

    #[test]
    #[ignore]
    fn bench_sema_signal() {
        for waiters in [1, 4, 8, 16] {
            for targeted in [false, true] {
                let (ns, cpu) = sema_handoff_ns(targeted, waiters, 20_000 / waiters);
                let name = if targeted { "targeted wake" } else { "notify_all" };
                eprintln!("sema signal(1), {waiters:2} waiters, {name:<14}: {ns:7.0} ns/token wall, {cpu:7.0} ns/token CPU");
            }
        }
    }

    #[test]
    fn sema_nids_match_names() {
        for &(nid, name, _, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }
}
