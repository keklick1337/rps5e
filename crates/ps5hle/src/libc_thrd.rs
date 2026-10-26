// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceLibcInternal C++ runtime thread helpers served when no loaded libc
//! module exports them: `_Thrd_join`, `_Thrd_sleep`, the global shared_ptr
//! spin lock and the Dinkumware mutex and condition objects (`_Mtx_*`,
//! `_Cnd_*`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::ThreadId;
use std::time::Duration;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceLibcInternal";

/// `_Thrd_*` / `_Mtx_*` / `_Cnd_*` results (thrd_success, thrd_busy,
/// thrd_error).
const THRD_SUCCESS: HleResult = 0;
const THRD_TIMEDOUT: HleResult = 2;
const THRD_BUSY: HleResult = 3;
const THRD_ERROR: HleResult = 4;

const NANOS_PER_SECOND: i128 = 1_000_000_000;
/// Longest single sleep of `_Thrd_sleep`; longer waits are re-checked.
const LONGEST_SLEEP: Duration = Duration::from_secs(86_400);

/// _Thrd_join(thrd_t thread, int* code): pthread_join; the code is the low
/// 32 bits of the thread's exit value.
fn cb_thrd_join(ctx: &mut CpuContext) -> HleResult {
    let (thread, code) = (ctx.rdi(), ctx.rsi());
    match crate::pthread::join_thread(thread) {
        Ok(value) => {
            if code != 0 {
                ctx.write_u32(code, value as u32);
            }
            THRD_SUCCESS
        }
        Err(_) => THRD_ERROR,
    }
}

/// The `xtime` {i64 sec, i64 nsec} at `addr` as nanoseconds since the epoch
/// (an unnormalised nsec carries into the seconds).
fn xtime_nanos(ctx: &CpuContext, addr: u64) -> Option<i128> {
    let sec = ctx.read_u64(addr)? as i64 as i128;
    let nsec = ctx.read_u64(addr + 8)? as i64 as i128;
    Some(sec * NANOS_PER_SECOND + nsec)
}

fn realtime_nanos() -> i128 {
    crate::pthread::realtime_now().as_nanos() as i128
}

/// _Thrd_sleep(const xtime* target, xtime* remaining): sleep until the
/// realtime clock reaches `target`; a past target returns at once. The
/// sleeps are cancellation points.
fn cb_thrd_sleep(ctx: &mut CpuContext) -> HleResult {
    let Some(deadline) = xtime_nanos(ctx, ctx.rdi()) else {
        log::error!("[libc] _Thrd_sleep: unreadable target {:#x}", ctx.rdi());
        return THRD_ERROR;
    };
    loop {
        let left = deadline - realtime_nanos();
        if left <= 0 {
            return THRD_SUCCESS;
        }
        let nanos = left.min(LONGEST_SLEEP.as_nanos() as i128) as u64;
        crate::kernel_ext::cancellable_sleep(Duration::from_nanos(nanos));
    }
}

static SHARED_PTR_SPIN_LOCK: AtomicBool = AtomicBool::new(false);

/// _Lock_shared_ptr_spin_lock(): the process-wide lock of the atomic
/// shared_ptr functions. Waiters yield, and every 64 attempts sleep
/// briefly so guest signals are delivered.
fn cb_lock_shared_ptr_spin_lock(_ctx: &mut CpuContext) -> HleResult {
    let mut attempts = 0u32;
    while SHARED_PTR_SPIN_LOCK.swap(true, Ordering::Acquire) {
        attempts = attempts.wrapping_add(1);
        if attempts % 64 == 0 {
            crate::signals::sleep(Duration::from_micros(50));
        } else {
            std::thread::yield_now();
        }
    }
    0
}

/// _Unlock_shared_ptr_spin_lock()
fn cb_unlock_shared_ptr_spin_lock(_ctx: &mut CpuContext) -> HleResult {
    SHARED_PTR_SPIN_LOCK.store(false, Ordering::Release);
    0
}

// ---------------------------------------------------------------------------
// Dinkumware mutexes and conditions
// ---------------------------------------------------------------------------

/// `_Mtx_init` type bit of a recursive mutex (the plain, try and timed
/// bits change nothing here).
const MTX_RECURSIVE: i32 = 0x100;

#[derive(Default)]
struct MtxState {
    owner: Option<ThreadId>,
    count: u32,
}

struct DinkMutex {
    kind: i32,
    state: Mutex<MtxState>,
    cv: Condvar,
}

/// A condition: waiters sleep until the broadcast generation moves on.
struct DinkCond {
    generation: Mutex<u64>,
    cv: Condvar,
}

#[derive(Clone)]
enum DinkObject {
    Mutex(Arc<DinkMutex>),
    Cond(Arc<DinkCond>),
}

/// Live objects by the opaque handle stored in the guest's `_Mtx_t` /
/// `_Cnd_t` slot.
static DINK_OBJECTS: Mutex<Option<HashMap<u64, DinkObject>>> = Mutex::new(None);
static NEXT_DINK_HANDLE: AtomicU64 = AtomicU64::new(0x4D54_5800_0000_0010);

fn lock_state<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn dink_insert(object: DinkObject) -> u64 {
    let handle = NEXT_DINK_HANDLE.fetch_add(0x10, Ordering::Relaxed);
    lock_state(&DINK_OBJECTS).get_or_insert_with(HashMap::new).insert(handle, object);
    handle
}

fn dink_lookup(handle: u64) -> Option<DinkObject> {
    lock_state(&DINK_OBJECTS).as_ref()?.get(&handle).cloned()
}

fn dink_remove(handle: u64) {
    if let Some(m) = lock_state(&DINK_OBJECTS).as_mut() {
        m.remove(&handle);
    }
}

/// The handle in the guest slot `slot` (Err: a null slot pointer).
fn read_slot(ctx: &CpuContext, slot: u64, function: &str) -> Result<u64, HleResult> {
    if slot == 0 {
        log::error!("[libc] {function}: null handle pointer");
        return Err(THRD_ERROR);
    }
    ctx.read_u64(slot).ok_or_else(|| {
        log::error!("[libc] {function}: unreadable handle pointer {slot:#x}");
        THRD_ERROR
    })
}

/// The mutex whose handle is in `slot`: null slots, null handles and
/// handles that are not `_Mtx_init` mutexes are errors.
fn resolve_mutex(ctx: &CpuContext, slot: u64, function: &str) -> Result<Arc<DinkMutex>, HleResult> {
    let handle = read_slot(ctx, slot, function)?;
    match dink_lookup(handle) {
        Some(DinkObject::Mutex(m)) => Ok(m),
        _ => {
            log::error!("[libc] {function}: {handle:#x} is not a mutex created by _Mtx_init");
            Err(THRD_ERROR)
        }
    }
}

fn resolve_cond(ctx: &CpuContext, slot: u64, function: &str) -> Result<Arc<DinkCond>, HleResult> {
    let handle = read_slot(ctx, slot, function)?;
    match dink_lookup(handle) {
        Some(DinkObject::Cond(c)) => Ok(c),
        _ => {
            log::error!("[libc] {function}: {handle:#x} is not a condition created by _Cnd_init");
            Err(THRD_ERROR)
        }
    }
}

fn this_thread() -> ThreadId {
    std::thread::current().id()
}

fn owned_by_caller(s: &MtxState) -> bool {
    s.count > 0 && s.owner == Some(this_thread())
}

/// Wait for the mutex to be free and take it once.
fn acquire<'a>(m: &'a DinkMutex, mut s: MutexGuard<'a, MtxState>) {
    while s.count > 0 {
        s = crate::signals::wait_cv(&m.state, &m.cv, s, None).0;
    }
    s.owner = Some(this_thread());
    s.count = 1;
}

/// Give the mutex up completely.
fn release(m: &DinkMutex, s: &mut MtxState) {
    s.count = 0;
    s.owner = None;
    m.cv.notify_one();
}

fn create_object(ctx: &CpuContext, slot: u64, function: &str, object: DinkObject) -> HleResult {
    if slot == 0 {
        log::error!("[libc] {function}: null handle pointer");
        return THRD_ERROR;
    }
    if !ctx.write_u64(slot, 0) {
        log::error!("[libc] {function}: unwritable handle pointer {slot:#x}");
        return THRD_ERROR;
    }
    let handle = dink_insert(object);
    ctx.write_u64(slot, handle);
    THRD_SUCCESS
}

/// _Mtx_init(_Mtx_t* mtx, int type)
fn cb_mtx_init(ctx: &mut CpuContext) -> HleResult {
    let (slot, kind) = (ctx.rdi(), ctx.rsi() as i32);
    let m = DinkMutex { kind, state: Mutex::new(MtxState::default()), cv: Condvar::new() };
    create_object(ctx, slot, "_Mtx_init", DinkObject::Mutex(Arc::new(m)))
}

/// _Mtx_destroy(_Mtx_t* mtx): a null handle is ignored; a locked mutex is
/// not destroyed.
fn cb_mtx_destroy(ctx: &mut CpuContext) -> HleResult {
    let slot = ctx.rdi();
    let Ok(handle) = read_slot(ctx, slot, "_Mtx_destroy") else { return 0 };
    if handle == 0 {
        return 0;
    }
    let Ok(m) = resolve_mutex(ctx, slot, "_Mtx_destroy") else { return 0 };
    if lock_state(&m.state).count != 0 {
        log::error!("[libc] _Mtx_destroy: mutex {handle:#x} destroyed while locked");
        return 0;
    }
    dink_remove(handle);
    ctx.write_u64(slot, 0);
    0
}

/// _Mtx_lock(_Mtx_t* mtx): relocking a non-recursive mutex the caller
/// holds is thrd_busy.
fn cb_mtx_lock(ctx: &mut CpuContext) -> HleResult {
    let m = match resolve_mutex(ctx, ctx.rdi(), "_Mtx_lock") {
        Ok(m) => m,
        Err(e) => return e,
    };
    let mut s = lock_state(&m.state);
    if owned_by_caller(&s) {
        if m.kind & MTX_RECURSIVE == 0 {
            return THRD_BUSY;
        }
        if s.count >= i32::MAX as u32 {
            log::error!("[libc] _Mtx_lock: recursive lock count overflow");
            return THRD_ERROR;
        }
        s.count += 1;
        return THRD_SUCCESS;
    }
    acquire(&m, s);
    THRD_SUCCESS
}

/// _Mtx_timedlock(_Mtx_t* mtx, const xtime* target): _Mtx_lock that gives
/// up with thrd_timedout once the realtime clock reaches `target`. A free
/// mutex is taken even when the target has passed; relocking a held
/// non-recursive mutex is thrd_busy at once.
fn cb_mtx_timedlock(ctx: &mut CpuContext) -> HleResult {
    let (slot, target) = (ctx.rdi(), ctx.rsi());
    if target == 0 {
        log::error!("[libc] _Mtx_timedlock: null target");
        return THRD_ERROR;
    }
    let Some(deadline) = xtime_nanos(ctx, target) else {
        log::error!("[libc] _Mtx_timedlock: unreadable target {target:#x}");
        return THRD_ERROR;
    };
    let m = match resolve_mutex(ctx, slot, "_Mtx_timedlock") {
        Ok(m) => m,
        Err(e) => return e,
    };
    let mut s = lock_state(&m.state);
    if owned_by_caller(&s) {
        if m.kind & MTX_RECURSIVE == 0 {
            return THRD_BUSY;
        }
        if s.count >= i32::MAX as u32 {
            log::error!("[libc] _Mtx_timedlock: recursive lock count overflow");
            return THRD_ERROR;
        }
        s.count += 1;
        return THRD_SUCCESS;
    }
    while s.count > 0 {
        let left = deadline - realtime_nanos();
        if left <= 0 {
            return THRD_TIMEDOUT;
        }
        let wait = Duration::from_nanos(left.min(LONGEST_SLEEP.as_nanos() as i128) as u64);
        s = crate::signals::wait_cv(&m.state, &m.cv, s, Some(std::time::Instant::now() + wait)).0;
    }
    s.owner = Some(this_thread());
    s.count = 1;
    THRD_SUCCESS
}

/// _Thrd_id(): the calling thread's `pthread_t`, unchanged (measured on a
/// PS5 in two threads).
fn cb_thrd_id(_ctx: &mut CpuContext) -> HleResult {
    crate::pthread::current_handle() as HleResult
}

/// _Mtx_unlock(_Mtx_t* mtx): a mutex the caller does not hold is left as
/// it is.
fn cb_mtx_unlock(ctx: &mut CpuContext) -> HleResult {
    let m = match resolve_mutex(ctx, ctx.rdi(), "_Mtx_unlock") {
        Ok(m) => m,
        Err(e) => return e,
    };
    let mut s = lock_state(&m.state);
    if owned_by_caller(&s) {
        s.count -= 1;
        if s.count == 0 {
            release(&m, &mut s);
        }
    }
    THRD_SUCCESS
}

/// _Cnd_init(_Cnd_t* cond)
fn cb_cnd_init(ctx: &mut CpuContext) -> HleResult {
    let c = DinkCond { generation: Mutex::new(0), cv: Condvar::new() };
    create_object(ctx, ctx.rdi(), "_Cnd_init", DinkObject::Cond(Arc::new(c)))
}

/// _Cnd_init_with_default_name_override(_Cnd_t* cond, const char* name):
/// _Cnd_init; the debug name is not kept.
fn cb_cnd_init_named(ctx: &mut CpuContext) -> HleResult {
    let c = DinkCond { generation: Mutex::new(0), cv: Condvar::new() };
    create_object(ctx, ctx.rdi(), "_Cnd_init_with_default_name_override", DinkObject::Cond(Arc::new(c)))
}

/// _Cnd_destroy(_Cnd_t* cond): a null handle is ignored.
fn cb_cnd_destroy(ctx: &mut CpuContext) -> HleResult {
    let slot = ctx.rdi();
    let Ok(handle) = read_slot(ctx, slot, "_Cnd_destroy") else { return 0 };
    if handle == 0 || resolve_cond(ctx, slot, "_Cnd_destroy").is_err() {
        return 0;
    }
    dink_remove(handle);
    ctx.write_u64(slot, 0);
    0
}

/// _Cnd_wait(_Cnd_t* cond, _Mtx_t* mtx): returns at once unless the caller
/// holds the mutex exactly once; otherwise releases it, sleeps until a
/// broadcast and takes it again.
fn cb_cnd_wait(ctx: &mut CpuContext) -> HleResult {
    let c = match resolve_cond(ctx, ctx.rdi(), "_Cnd_wait") {
        Ok(c) => c,
        Err(e) => return e,
    };
    let m = match resolve_mutex(ctx, ctx.rsi(), "_Cnd_wait") {
        Ok(m) => m,
        Err(e) => return e,
    };
    let mut generation = lock_state(&c.generation);
    {
        let mut s = lock_state(&m.state);
        if !owned_by_caller(&s) || s.count != 1 {
            return THRD_SUCCESS;
        }
        release(&m, &mut s);
    }
    let start = *generation;
    while *generation == start {
        generation = crate::signals::wait_cv(&c.generation, &c.cv, generation, None).0;
    }
    drop(generation);
    acquire(&m, lock_state(&m.state));
    THRD_SUCCESS
}

/// _Cnd_broadcast(_Cnd_t* cond)
fn cb_cnd_broadcast(ctx: &mut CpuContext) -> HleResult {
    let c = match resolve_cond(ctx, ctx.rdi(), "_Cnd_broadcast") {
        Ok(c) => c,
        Err(e) => return e,
    };
    *lock_state(&c.generation) += 1;
    c.cv.notify_all();
    THRD_SUCCESS
}

const FALLBACK_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("YvmY5Jf0VYU", "_Thrd_join", cb_thrd_join),
    ("YaHc3GS7y7g", "_Mtx_init", cb_mtx_init),
    ("5Lf51jvohTQ", "_Mtx_destroy", cb_mtx_destroy),
    ("iS4aWbUonl0", "_Mtx_lock", cb_mtx_lock),
    ("gTuXQwP9rrs", "_Mtx_unlock", cb_mtx_unlock),
    ("hPzYSd5Nasc", "_Mtx_timedlock", cb_mtx_timedlock),
    ("np6xXcXEnXE", "_Thrd_id", cb_thrd_id),
    ("SreZybSRWpU", "_Cnd_init", cb_cnd_init),
    ("jBOZAv6CwkM", "_Cnd_init_with_default_name_override", cb_cnd_init_named),
    ("7yMFgcS8EPA", "_Cnd_destroy", cb_cnd_destroy),
    ("vEaqE-7IZYc", "_Cnd_wait", cb_cnd_wait),
    ("VsP3daJgmVA", "_Cnd_broadcast", cb_cnd_broadcast),
    ("jfRI3snge3o", "_Thrd_sleep", cb_thrd_sleep),
    ("fRWufXAccuI", "_Lock_shared_ptr_spin_lock", cb_lock_shared_ptr_spin_lock),
    ("1HYEoANqZ1w", "_Unlock_shared_ptr_spin_lock", cb_unlock_shared_ptr_spin_lock),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in FALLBACK_EXPORTS {
        reg.register_fallback_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::{Rdi, Rsi};
    use std::sync::atomic::AtomicU32;
    use std::sync::Arc;
    use std::time::Instant;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in FALLBACK_EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    fn scratch(base: u64) -> &'static ps5vm::VirtualMemory {
        let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
        vm.map(base, 0x1000, ps5vm::MemProtect::RW).unwrap();
        vm
    }

    /// Joining a thread that cannot be joined is thrd_error and leaves the
    /// code alone.
    #[test]
    fn thrd_join_failures_are_thrd_error() {
        const BASE: u64 = 0x5F87_0000_0000;
        let vm = scratch(BASE);
        let mut ctx = CpuContext::new(vm);
        ctx.write_u32(BASE, 9);
        ctx.set(Rdi, 0x7_7700_0000);
        ctx.set(Rsi, BASE);
        assert_eq!(cb_thrd_join(&mut ctx), THRD_ERROR);
        assert_eq!(ctx.read_u32(BASE), Some(9));
    }

    /// _Thrd_sleep waits for the realtime target, normalising nsec in
    /// both directions; past targets return at once.
    #[test]
    fn thrd_sleep_reaches_the_realtime_target() {
        const BASE: u64 = 0x5F88_0000_0000;
        let vm = scratch(BASE);
        let mut ctx = CpuContext::new(vm);
        let write = |ctx: &CpuContext, sec: i64, nsec: i64| {
            ctx.write_u64(BASE, sec as u64);
            ctx.write_u64(BASE + 8, nsec as u64);
        };
        let split = |nanos: i128| ((nanos / NANOS_PER_SECOND) as i64, (nanos % NANOS_PER_SECOND) as i64);
        ctx.set(Rdi, BASE);
        let target = realtime_nanos() + 30_000_000;
        let (sec, nsec) = split(target);
        write(&ctx, sec, nsec);
        assert_eq!(cb_thrd_sleep(&mut ctx), THRD_SUCCESS);
        assert!(realtime_nanos() >= target);

        let (sec, nsec) = split(realtime_nanos() - NANOS_PER_SECOND);
        write(&ctx, sec, nsec);
        let start = Instant::now();
        assert_eq!(cb_thrd_sleep(&mut ctx), THRD_SUCCESS);
        assert!(start.elapsed() < Duration::from_millis(500));

        let carried = realtime_nanos() + 20_000_000;
        let (sec, nsec) = split(carried);
        write(&ctx, sec - 2, nsec + 2 * NANOS_PER_SECOND as i64);
        assert_eq!(cb_thrd_sleep(&mut ctx), THRD_SUCCESS);
        assert!(realtime_nanos() >= carried);

        let borrowed = realtime_nanos() + 20_000_000;
        let (sec, nsec) = split(borrowed);
        write(&ctx, sec + 1, nsec - NANOS_PER_SECOND as i64);
        assert_eq!(cb_thrd_sleep(&mut ctx), THRD_SUCCESS);
        assert!(realtime_nanos() >= borrowed);

        ctx.set(Rdi, 0);
        assert_eq!(cb_thrd_sleep(&mut ctx), THRD_ERROR);
    }

    fn call(ctx: &mut CpuContext, cb: HleCallback, args: &[u64]) -> HleResult {
        for (r, v) in [Rdi, Rsi].iter().zip(args) {
            ctx.set(*r, *v);
        }
        cb(ctx)
    }

    const PLAIN: u64 = 0x01;
    const TRY: u64 = 0x02;
    const TIMED: u64 = 0x04;
    const RECURSIVE: u64 = 0x100;

    /// A mutex in the guest slot `slot` (pre-filled with junk, which init
    /// replaces).
    fn create_mutex(ctx: &mut CpuContext, slot: u64, kind: u64) {
        ctx.write_u64(slot, 1);
        assert_eq!(call(ctx, cb_mtx_init, &[slot, kind]), THRD_SUCCESS);
        assert_ne!(ctx.read_u64(slot), Some(0));
    }

    /// Mutex basics: relock of a non-recursive mutex is busy for every
    /// type, unlock by a thread that does not hold it is a no-op, destroy of
    /// a locked mutex keeps it, and bad handles are errors.
    #[test]
    fn dinkumware_mutex_basics() {
        const BASE: u64 = 0x6A12_0000_0000;
        let vm = scratch(BASE);
        let mut ctx = CpuContext::new(vm);
        let (m, other) = (BASE, BASE + 0x10);
        create_mutex(&mut ctx, m, TRY);
        assert_eq!(call(&mut ctx, cb_mtx_lock, &[m]), THRD_SUCCESS);
        assert_eq!(call(&mut ctx, cb_mtx_lock, &[m]), THRD_BUSY);
        let handle = ctx.read_u64(m).unwrap();
        call(&mut ctx, cb_mtx_destroy, &[m]);
        assert_eq!(ctx.read_u64(m), Some(handle), "a locked mutex is not destroyed");
        std::thread::spawn(move || {
            let mut ctx = CpuContext::new(vm);
            assert_eq!(call(&mut ctx, cb_mtx_unlock, &[m]), THRD_SUCCESS);
        })
        .join()
        .unwrap();
        assert_eq!(call(&mut ctx, cb_mtx_lock, &[m]), THRD_BUSY, "still held after a foreign unlock");
        assert_eq!(call(&mut ctx, cb_mtx_unlock, &[m]), THRD_SUCCESS);
        assert_eq!(call(&mut ctx, cb_mtx_unlock, &[m]), THRD_SUCCESS);
        ctx.write_u64(other, 0);
        assert_eq!(call(&mut ctx, cb_mtx_lock, &[other]), THRD_ERROR);
        assert_eq!(call(&mut ctx, cb_cnd_init, &[other]), THRD_SUCCESS);
        assert_eq!(call(&mut ctx, cb_mtx_lock, &[other]), THRD_ERROR, "a condition is not a mutex");
        call(&mut ctx, cb_cnd_destroy, &[other]);
        call(&mut ctx, cb_mtx_destroy, &[m]);
        assert_eq!(ctx.read_u64(m), Some(0));
        assert_eq!(call(&mut ctx, cb_mtx_lock, &[m]), THRD_ERROR);
        call(&mut ctx, cb_mtx_destroy, &[m]);
        call(&mut ctx, cb_mtx_destroy, &[0]);
        for kind in [PLAIN, TIMED | TRY, 0, TRY | 0x8] {
            create_mutex(&mut ctx, m, kind);
            assert_eq!(call(&mut ctx, cb_mtx_lock, &[m]), THRD_SUCCESS);
            assert_eq!(call(&mut ctx, cb_mtx_lock, &[m]), THRD_BUSY, "type {kind:#x}");
            assert_eq!(call(&mut ctx, cb_mtx_unlock, &[m]), THRD_SUCCESS);
            call(&mut ctx, cb_mtx_destroy, &[m]);
            assert_eq!(ctx.read_u64(m), Some(0));
        }
        assert_eq!(call(&mut ctx, cb_mtx_init, &[0, TRY]), THRD_ERROR);
        assert_eq!(call(&mut ctx, cb_mtx_lock, &[0]), THRD_ERROR);
        assert_eq!(call(&mut ctx, cb_mtx_unlock, &[0]), THRD_ERROR);
    }

    /// A recursive mutex is released by the last unlock; mutual exclusion
    /// holds across threads.
    #[test]
    fn dinkumware_recursive_mutex_and_exclusion() {
        const BASE: u64 = 0x6A13_0000_0000;
        let vm = scratch(BASE);
        let mut ctx = CpuContext::new(vm);
        let m = BASE;
        create_mutex(&mut ctx, m, TRY | RECURSIVE);
        for _ in 0..3 {
            assert_eq!(call(&mut ctx, cb_mtx_lock, &[m]), THRD_SUCCESS);
        }
        let acquired = Arc::new(AtomicBool::new(false));
        let a = acquired.clone();
        let other = std::thread::spawn(move || {
            let mut ctx = CpuContext::new(vm);
            assert_eq!(call(&mut ctx, cb_mtx_lock, &[m]), THRD_SUCCESS);
            a.store(true, Ordering::SeqCst);
            assert_eq!(call(&mut ctx, cb_mtx_unlock, &[m]), THRD_SUCCESS);
        });
        for _ in 0..2 {
            assert_eq!(call(&mut ctx, cb_mtx_unlock, &[m]), THRD_SUCCESS);
        }
        std::thread::sleep(Duration::from_millis(20));
        assert!(!acquired.load(Ordering::SeqCst));
        assert_eq!(call(&mut ctx, cb_mtx_unlock, &[m]), THRD_SUCCESS);
        other.join().unwrap();
        assert!(acquired.load(Ordering::SeqCst));
        call(&mut ctx, cb_mtx_destroy, &[m]);

        let exclusive = BASE + 0x10;
        create_mutex(&mut ctx, exclusive, TRY);
        let counter = Arc::new(AtomicU32::new(0));
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let counter = counter.clone();
                std::thread::spawn(move || {
                    let mut ctx = CpuContext::new(vm);
                    for _ in 0..20_000 {
                        assert_eq!(call(&mut ctx, cb_mtx_lock, &[exclusive]), THRD_SUCCESS);
                        let v = counter.load(Ordering::Relaxed);
                        counter.store(v + 1, Ordering::Relaxed);
                        assert_eq!(call(&mut ctx, cb_mtx_unlock, &[exclusive]), THRD_SUCCESS);
                    }
                })
            })
            .collect();
        for w in workers {
            w.join().unwrap();
        }
        assert_eq!(counter.load(Ordering::SeqCst), 80_000);
        call(&mut ctx, cb_mtx_destroy, &[exclusive]);
    }

    /// A broadcast wakes every waiter, which returns holding the mutex; a
    /// wait without the mutex held exactly once returns at once.
    #[test]
    fn dinkumware_condition_broadcast() {
        const BASE: u64 = 0x6A14_0000_0000;
        let vm = scratch(BASE);
        let mut ctx = CpuContext::new(vm);
        let (c, m, ready, waiting, woken) = (BASE, BASE + 0x10, BASE + 0x20, BASE + 0x28, BASE + 0x30);
        ctx.write_u64(c, 1);
        assert_eq!(call(&mut ctx, cb_cnd_init, &[c]), THRD_SUCCESS);
        create_mutex(&mut ctx, m, TRY);
        let waiters: Vec<_> = (0..2)
            .map(|_| {
                std::thread::spawn(move || {
                    let mut ctx = CpuContext::new(vm);
                    assert_eq!(call(&mut ctx, cb_mtx_lock, &[m]), THRD_SUCCESS);
                    ctx.write_u32(waiting, ctx.read_u32(waiting).unwrap() + 1);
                    while ctx.read_u32(ready) == Some(0) {
                        assert_eq!(call(&mut ctx, cb_cnd_wait, &[c, m]), THRD_SUCCESS);
                    }
                    ctx.write_u32(woken, ctx.read_u32(woken).unwrap() + 1);
                    assert_eq!(call(&mut ctx, cb_mtx_lock, &[m]), THRD_BUSY, "the wait returns holding the mutex");
                    assert_eq!(call(&mut ctx, cb_mtx_unlock, &[m]), THRD_SUCCESS);
                })
            })
            .collect();
        loop {
            assert_eq!(call(&mut ctx, cb_mtx_lock, &[m]), THRD_SUCCESS);
            let all = ctx.read_u32(waiting) == Some(2);
            assert_eq!(call(&mut ctx, cb_mtx_unlock, &[m]), THRD_SUCCESS);
            if all {
                break;
            }
            std::thread::yield_now();
        }
        assert_eq!(call(&mut ctx, cb_mtx_lock, &[m]), THRD_SUCCESS);
        ctx.write_u32(ready, 1);
        assert_eq!(call(&mut ctx, cb_cnd_broadcast, &[c]), THRD_SUCCESS);
        assert_eq!(call(&mut ctx, cb_mtx_unlock, &[m]), THRD_SUCCESS);
        for w in waiters {
            w.join().unwrap();
        }
        assert_eq!(ctx.read_u32(woken), Some(2));
        assert_eq!(call(&mut ctx, cb_cnd_broadcast, &[c]), THRD_SUCCESS);

        // Not held, or held twice: no wait.
        assert_eq!(call(&mut ctx, cb_cnd_wait, &[c, m]), THRD_SUCCESS);
        let recursive = BASE + 0x40;
        create_mutex(&mut ctx, recursive, TRY | RECURSIVE);
        assert_eq!(call(&mut ctx, cb_mtx_lock, &[recursive]), THRD_SUCCESS);
        assert_eq!(call(&mut ctx, cb_mtx_lock, &[recursive]), THRD_SUCCESS);
        assert_eq!(call(&mut ctx, cb_cnd_wait, &[c, recursive]), THRD_SUCCESS);
        assert_eq!(call(&mut ctx, cb_mtx_unlock, &[recursive]), THRD_SUCCESS);
        assert_eq!(call(&mut ctx, cb_mtx_unlock, &[recursive]), THRD_SUCCESS);
        call(&mut ctx, cb_mtx_destroy, &[recursive]);

        assert_eq!(call(&mut ctx, cb_cnd_wait, &[m, m]), THRD_ERROR, "a mutex is not a condition");
        // The named initializer makes the same condition, the name ignored.
        let named = BASE + 0x60;
        for name in [0, BASE + 0x80] {
            assert_eq!(call(&mut ctx, cb_cnd_init_named, &[named, name]), THRD_SUCCESS);
            assert_ne!(ctx.read_u64(named), Some(0));
            assert_eq!(call(&mut ctx, cb_cnd_broadcast, &[named]), THRD_SUCCESS);
            assert_eq!(call(&mut ctx, cb_cnd_destroy, &[named]), 0);
        }
        assert_eq!(call(&mut ctx, cb_cnd_init_named, &[0, 0]), THRD_ERROR);
        let empty = BASE + 0x50;
        ctx.write_u64(empty, 0);
        assert_eq!(call(&mut ctx, cb_cnd_broadcast, &[empty]), THRD_ERROR);
        assert_eq!(call(&mut ctx, cb_cnd_broadcast, &[0]), THRD_ERROR);
        assert_eq!(call(&mut ctx, cb_cnd_init, &[0]), THRD_ERROR);
        call(&mut ctx, cb_cnd_destroy, &[c]);
        assert_eq!(ctx.read_u64(c), Some(0));
        assert_eq!(call(&mut ctx, cb_cnd_broadcast, &[c]), THRD_ERROR);
        call(&mut ctx, cb_cnd_destroy, &[empty]);
        call(&mut ctx, cb_cnd_destroy, &[0]);
        call(&mut ctx, cb_mtx_destroy, &[m]);
    }

    /// _Mtx_timedlock takes a free mutex even with a past target, times out
    /// on one held by another thread, gets it once it is released before
    /// the target, and follows _Mtx_lock for relocks.
    #[test]
    fn dinkumware_mutex_timedlock() {
        const BASE: u64 = 0x6A15_0000_0000;
        let vm = scratch(BASE);
        let mut ctx = CpuContext::new(vm);
        let (m, target) = (BASE, BASE + 0x10);
        let set_target = |ctx: &CpuContext, nanos: i128| {
            ctx.write_u64(target, (nanos / NANOS_PER_SECOND) as u64);
            ctx.write_u64(target + 8, (nanos % NANOS_PER_SECOND) as u64);
        };
        create_mutex(&mut ctx, m, TIMED);
        set_target(&ctx, realtime_nanos() - NANOS_PER_SECOND);
        assert_eq!(call(&mut ctx, cb_mtx_timedlock, &[m, target]), THRD_SUCCESS);
        assert_eq!(call(&mut ctx, cb_mtx_timedlock, &[m, target]), THRD_BUSY);
        let timed_out = std::thread::spawn(move || {
            let mut ctx = CpuContext::new(vm);
            let deadline = realtime_nanos() + 30_000_000;
            ctx.write_u64(target, (deadline / NANOS_PER_SECOND) as u64);
            ctx.write_u64(target + 8, (deadline % NANOS_PER_SECOND) as u64);
            let r = call(&mut ctx, cb_mtx_timedlock, &[m, target]);
            (r, realtime_nanos() >= deadline)
        })
        .join()
        .unwrap();
        assert_eq!(timed_out, (THRD_TIMEDOUT, true));
        let waiter = std::thread::spawn(move || {
            let mut ctx = CpuContext::new(vm);
            let slot = target + 0x20;
            let deadline = realtime_nanos() + 10 * NANOS_PER_SECOND;
            ctx.write_u64(slot, (deadline / NANOS_PER_SECOND) as u64);
            ctx.write_u64(slot + 8, (deadline % NANOS_PER_SECOND) as u64);
            let r = call(&mut ctx, cb_mtx_timedlock, &[m, slot]);
            assert_eq!(call(&mut ctx, cb_mtx_unlock, &[m]), THRD_SUCCESS);
            r
        });
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(call(&mut ctx, cb_mtx_unlock, &[m]), THRD_SUCCESS);
        assert_eq!(waiter.join().unwrap(), THRD_SUCCESS);
        call(&mut ctx, cb_mtx_destroy, &[m]);

        create_mutex(&mut ctx, m, TIMED | RECURSIVE);
        assert_eq!(call(&mut ctx, cb_mtx_timedlock, &[m, target]), THRD_SUCCESS);
        assert_eq!(call(&mut ctx, cb_mtx_timedlock, &[m, target]), THRD_SUCCESS);
        assert_eq!(call(&mut ctx, cb_mtx_unlock, &[m]), THRD_SUCCESS);
        assert_eq!(call(&mut ctx, cb_mtx_unlock, &[m]), THRD_SUCCESS);
        call(&mut ctx, cb_mtx_destroy, &[m]);
        assert_eq!(call(&mut ctx, cb_mtx_timedlock, &[m, 0]), THRD_ERROR);
    }

    /// _Thrd_id is the caller's pthread handle.
    #[test]
    fn thrd_id_is_pthread_self() {
        let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
        let mut ctx = CpuContext::new(vm);
        assert_eq!(cb_thrd_id(&mut ctx) as u64, crate::pthread::current_handle());
        assert!(FALLBACK_EXPORTS.iter().any(|&(_, name, _)| name == "_Thrd_id"));
    }

    /// The spin lock keeps every critical section exclusive.
    #[test]
    fn shared_ptr_spin_lock_is_exclusive() {
        let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
        let inside = Arc::new(AtomicU32::new(0));
        let counter = Arc::new(AtomicU32::new(0));
        let overlapped = Arc::new(AtomicBool::new(false));
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let (inside, counter, overlapped) = (inside.clone(), counter.clone(), overlapped.clone());
                std::thread::spawn(move || {
                    let mut ctx = CpuContext::new(vm);
                    for _ in 0..20_000 {
                        cb_lock_shared_ptr_spin_lock(&mut ctx);
                        if inside.fetch_add(1, Ordering::SeqCst) != 0 {
                            overlapped.store(true, Ordering::SeqCst);
                        }
                        // A plain read-modify-write, safe only under the lock.
                        let v = counter.load(Ordering::Relaxed);
                        counter.store(v + 1, Ordering::Relaxed);
                        inside.fetch_sub(1, Ordering::SeqCst);
                        cb_unlock_shared_ptr_spin_lock(&mut ctx);
                    }
                })
            })
            .collect();
        for w in workers {
            w.join().unwrap();
        }
        assert!(!overlapped.load(Ordering::SeqCst));
        assert_eq!(counter.load(Ordering::SeqCst), 4 * 20_000);
    }
}
