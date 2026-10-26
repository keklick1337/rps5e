// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceUlt — user-level threads and their synchronisation objects.
//!
//! Ulthreads run as ordinary guest threads (one host thread each), so the
//! ULT synchronisation objects are host objects keyed by the guest object
//! address: blocking waits block the calling host thread. Waiting-queue and
//! queue-data resource pools are tracked so their creation/destroy rules
//! (capacity, busy) behave like the library; queues store their elements in
//! a fixed slot pool shared by every queue created on it.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::context::{CpuContext, CpuRegister};
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceUlt";

pub const ERR_NULL: HleResult = 0x8081_0001u32 as i32 as i64;
pub const ERR_ALIGNMENT: HleResult = 0x8081_0002u32 as i32 as i64;
pub const ERR_RANGE: HleResult = 0x8081_0003u32 as i32 as i64;
pub const ERR_INVALID: HleResult = 0x8081_0004u32 as i32 as i64;
pub const ERR_PERMISSION: HleResult = 0x8081_0005u32 as i32 as i64;
pub const ERR_STATE: HleResult = 0x8081_0006u32 as i32 as i64;
pub const ERR_BUSY: HleResult = 0x8081_0007u32 as i32 as i64;
pub const ERR_AGAIN: HleResult = 0x8081_0008u32 as i32 as i64;


fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

fn stack_arg(ctx: &CpuContext, i: u64) -> u64 {
    ctx.read_u64(ctx.get(CpuRegister::Rsp) + 8 + 8 * i).unwrap_or(0)
}

fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) & !(a - 1)
}

/// Identity of the calling thread for mutex ownership.
fn self_id() -> u64 {
    let h = crate::pthread::current_handle();
    if h != 0 {
        return h;
    }
    thread_local!(static ID: u64 = {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    });
    ID.with(|i| *i)
}

// ---------------------------------------------------------------------------
// Object state
// ---------------------------------------------------------------------------

#[derive(Default)]
struct MutexInner {
    owner: u64,
    count: u32,
    waiters: u32,
}

#[derive(Default)]
struct UltMutex {
    inner: Mutex<MutexInner>,
    cv: Condvar,
}

impl UltMutex {
    fn lock(&self, try_only: bool) -> HleResult {
        let me = self_id();
        let mut g = lock(&self.inner);
        if g.owner == me {
            g.count += 1;
            return 0;
        }
        if g.owner != 0 {
            if try_only {
                return ERR_BUSY;
            }
            g.waiters += 1;
            while g.owner != 0 {
                g = self.cv.wait(g).unwrap_or_else(|p| p.into_inner());
            }
            g.waiters -= 1;
        }
        g.owner = me;
        g.count = 1;
        0
    }

    fn unlock(&self) -> HleResult {
        let mut g = lock(&self.inner);
        if g.owner != self_id() {
            return ERR_PERMISSION;
        }
        g.count -= 1;
        if g.count == 0 {
            g.owner = 0;
            drop(g);
            self.cv.notify_one();
        }
        0
    }

    /// Release fully (for a condition wait); returns the recursion count.
    fn release_all(&self) -> Result<u32, HleResult> {
        let mut g = lock(&self.inner);
        if g.owner != self_id() {
            return Err(ERR_PERMISSION);
        }
        let n = g.count;
        g.owner = 0;
        g.count = 0;
        drop(g);
        self.cv.notify_one();
        Ok(n)
    }

    fn reacquire(&self, count: u32) {
        let me = self_id();
        let mut g = lock(&self.inner);
        while g.owner != 0 {
            g = self.cv.wait(g).unwrap_or_else(|p| p.into_inner());
        }
        g.owner = me;
        g.count = count;
    }
}

struct SemInner {
    resources: i32,
    waiters: u32,
    alive: bool,
}

struct UltSemaphore {
    inner: Mutex<SemInner>,
    cv: Condvar,
}

#[derive(Default)]
struct CondInner {
    seq: u64,
    waiters: u32,
    /// Number of pending wake-ups for current waiters.
    signals: u32,
}

#[derive(Default)]
struct UltCond {
    inner: Mutex<CondInner>,
    cv: Condvar,
}

#[derive(Default)]
struct RwInner {
    readers: u32,
    writer: u64,
    waiters: u32,
}

#[derive(Default)]
struct UltRwLock {
    inner: Mutex<RwInner>,
    cv: Condvar,
}

struct WaitingPool;

struct DataPoolInner {
    /// Data slots shared by the pool's queues (0: unbounded).
    num_data: u32,
    queues: u32,
    used: u32,
    waiting_pool: u64,
}

struct DataPool {
    inner: Mutex<DataPoolInner>,
    /// Signalled when a slot is freed.
    space: Condvar,
}

struct QueueInner {
    items: VecDeque<Vec<u8>>,
    waiters: u32,
    alive: bool,
}

struct UltQueue {
    pool: Arc<DataPool>,
    data_size: u64,
    inner: Mutex<QueueInner>,
    data: Condvar,
}

struct Ulthread {
    handle: u64,
    /// The runtime the ulthread was created on.
    runtime: u64,
}

#[derive(Default)]
struct Tables {
    mutexes: HashMap<u64, Arc<UltMutex>>,
    semaphores: HashMap<u64, Arc<UltSemaphore>>,
    conds: HashMap<u64, Arc<UltCond>>,
    rwlocks: HashMap<u64, Arc<UltRwLock>>,
    waiting_pools: HashMap<u64, WaitingPool>,
    data_pools: HashMap<u64, Arc<DataPool>>,
    queues: HashMap<u64, Arc<UltQueue>>,
    runtimes: HashMap<u64, (u32, u32)>,
    ulthreads: HashMap<u64, Ulthread>,
    /// pthread handle -> ulthread object.
    by_handle: HashMap<u64, u64>,
}

static TABLES: Mutex<Option<Tables>> = Mutex::new(None);

fn tables<R>(f: impl FnOnce(&mut Tables) -> R) -> R {
    let mut g = lock(&TABLES);
    f(g.get_or_insert_with(Tables::default))
}

fn check_waiting_pool(t: &Tables, pool: u64) -> Result<(), HleResult> {
    if pool != 0 && !t.waiting_pools.contains_key(&pool) {
        return Err(ERR_INVALID);
    }
    Ok(())
}

/// Clear the guest-visible part of a newly created object.
fn clear_object(ctx: &CpuContext, obj: u64, size: usize) {
    ctx.write_bytes(obj, &vec![0u8; size]);
}

fn new_object(obj: u64, align: u64) -> Result<(), HleResult> {
    if obj == 0 {
        return Err(ERR_NULL);
    }
    if obj & (align - 1) != 0 {
        return Err(ERR_ALIGNMENT);
    }
    Ok(())
}

macro_rules! lookup {
    ($field:ident, $obj:expr) => {{
        let obj = $obj;
        match tables(|t| t.$field.get(&obj).cloned()) {
            Some(s) => s,
            None => return if obj == 0 { ERR_NULL } else { ERR_STATE },
        }
    }};
}

// ---------------------------------------------------------------------------
// Library
// ---------------------------------------------------------------------------

fn cb_initialize(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// Takes every object for sceUltFinalize, or ERR_BUSY (nothing torn down)
/// while an ulthread has not been joined: its guest thread still uses it.
fn take_tables(slot: &mut Option<Tables>) -> Result<Tables, HleResult> {
    if slot.as_ref().is_some_and(|t| !t.ulthreads.is_empty()) {
        return Err(ERR_BUSY);
    }
    Ok(slot.take().unwrap_or_default())
}

fn cb_finalize(_ctx: &mut CpuContext) -> HleResult {
    let t = match take_tables(&mut lock(&TABLES)) {
        Ok(t) => t,
        Err(e) => return e,
    };
    for s in t.semaphores.values() {
        lock(&s.inner).alive = false;
        s.cv.notify_all();
    }
    for q in t.queues.values() {
        lock(&q.inner).alive = false;
        q.data.notify_all();
        q.pool.space.notify_all();
    }
    0
}

/// Opt-param initializers: (optParam[, buildVersion]) — zero the header.
fn opt_param_init(ctx: &mut CpuContext, size: usize) -> HleResult {
    let opt = ctx.rdi();
    if opt == 0 {
        return ERR_NULL;
    }
    if opt & 7 != 0 {
        return ERR_ALIGNMENT;
    }
    if ctx.write_bytes(opt, &vec![0u8; size]) { 0 } else { ERR_INVALID }
}

fn cb_opt_param_initialize(ctx: &mut CpuContext) -> HleResult {
    opt_param_init(ctx, 16)
}

/// _sceUltMutexOptParamInitialize / _sceUltUlthreadRuntimeOptParamInitialize
/// (optParam, buildVersion): only a null parameter is rejected.
fn plain_opt_param_init(ctx: &mut CpuContext, size: usize) -> HleResult {
    let opt = ctx.rdi();
    if opt == 0 {
        return ERR_NULL;
    }
    clear_object(ctx, opt, size);
    0
}

fn cb_mutex_opt_param_initialize(ctx: &mut CpuContext) -> HleResult {
    plain_opt_param_init(ctx, 16)
}

fn cb_runtime_opt_param_initialize(ctx: &mut CpuContext) -> HleResult {
    plain_opt_param_init(ctx, 128)
}

// --- waiting queue resource pool ---

fn cb_waiting_pool_work_area_size(ctx: &mut CpuContext) -> HleResult {
    let (threads, objs) = (ctx.rdi() as u32 as u64, ctx.rsi() as u32 as u64);
    align_up((threads + objs) * 256, 8) as i64
}

/// _sceUltWaitingQueueResourcePoolCreate(pool, name, numThreads, numSyncObjects, workArea, opt, build)
fn cb_waiting_pool_create(ctx: &mut CpuContext) -> HleResult {
    let pool = ctx.rdi();
    if pool == 0 {
        return ERR_NULL;
    }
    clear_object(ctx, pool, 256);
    tables(|t| t.waiting_pools.insert(pool, WaitingPool));
    0
}

fn cb_waiting_pool_destroy(ctx: &mut CpuContext) -> HleResult {
    let pool = ctx.rdi();
    if pool == 0 {
        return ERR_NULL;
    }
    tables(|t| {
        if !t.waiting_pools.contains_key(&pool) {
            return ERR_STATE;
        }
        if t.data_pools.values().any(|p| lock(&p.inner).waiting_pool == pool) {
            return ERR_BUSY;
        }
        t.waiting_pools.remove(&pool);
        0
    })
}

// --- mutex ---

/// _sceUltMutexCreate(mutex, name, waitingPool, opt, build)
fn cb_mutex_create(ctx: &mut CpuContext) -> HleResult {
    let (m, pool) = (ctx.rdi(), ctx.rdx());
    if m == 0 {
        return ERR_NULL;
    }
    clear_object(ctx, m, 256);
    tables(|t| {
        if let Err(e) = check_waiting_pool(t, pool) {
            return e;
        }
        t.mutexes.insert(m, Arc::new(UltMutex::default()));
        0
    })
}

fn cb_mutex_lock(ctx: &mut CpuContext) -> HleResult {
    lookup!(mutexes, ctx.rdi()).lock(false)
}

fn cb_mutex_try_lock(ctx: &mut CpuContext) -> HleResult {
    lookup!(mutexes, ctx.rdi()).lock(true)
}

fn cb_mutex_unlock(ctx: &mut CpuContext) -> HleResult {
    lookup!(mutexes, ctx.rdi()).unlock()
}

fn cb_mutex_destroy(ctx: &mut CpuContext) -> HleResult {
    let m = ctx.rdi();
    let s = lookup!(mutexes, m);
    let g = lock(&s.inner);
    if g.owner != 0 || g.waiters != 0 {
        return ERR_BUSY;
    }
    drop(g);
    tables(|t| t.mutexes.remove(&m));
    0
}

// --- semaphore ---

/// _sceUltSemaphoreCreate(sem, name, numInitialResource, waitingPool, opt, build)
fn cb_semaphore_create(ctx: &mut CpuContext) -> HleResult {
    let (s, initial, pool, opt) = (ctx.rdi(), ctx.rdx() as i32, ctx.rcx(), ctx.r8());
    if let Err(e) = new_object(s, 8) {
        return e;
    }
    if opt & 7 != 0 {
        return ERR_ALIGNMENT;
    }
    if initial < 0 {
        return ERR_RANGE;
    }
    tables(|t| {
        if let Err(e) = check_waiting_pool(t, pool) {
            return e;
        }
        if t.semaphores.contains_key(&s) {
            return ERR_STATE;
        }
        clear_object(ctx, s, 256);
        let sem = UltSemaphore { inner: Mutex::new(SemInner { resources: initial, waiters: 0, alive: true }), cv: Condvar::new() };
        t.semaphores.insert(s, Arc::new(sem));
        0
    })
}

fn semaphore_acquire(ctx: &mut CpuContext, try_only: bool) -> HleResult {
    let n = ctx.rsi() as i32;
    if n <= 0 {
        return ERR_RANGE;
    }
    let s = lookup!(semaphores, ctx.rdi());
    let mut g = lock(&s.inner);
    if !g.alive {
        return ERR_STATE;
    }
    if g.resources < n {
        if try_only {
            return ERR_AGAIN;
        }
        g.waiters += 1;
        while g.alive && g.resources < n {
            g = s.cv.wait(g).unwrap_or_else(|p| p.into_inner());
        }
        g.waiters -= 1;
        if !g.alive {
            return ERR_STATE;
        }
    }
    g.resources -= n;
    0
}

fn cb_semaphore_acquire(ctx: &mut CpuContext) -> HleResult {
    semaphore_acquire(ctx, false)
}

fn cb_semaphore_try_acquire(ctx: &mut CpuContext) -> HleResult {
    semaphore_acquire(ctx, true)
}

fn cb_semaphore_release(ctx: &mut CpuContext) -> HleResult {
    let n = ctx.rsi() as i32;
    if n <= 0 {
        return ERR_RANGE;
    }
    let s = lookup!(semaphores, ctx.rdi());
    let mut g = lock(&s.inner);
    if !g.alive {
        return ERR_STATE;
    }
    if g.resources > i32::MAX - n {
        return ERR_RANGE;
    }
    g.resources += n;
    drop(g);
    s.cv.notify_all();
    0
}

fn cb_semaphore_destroy(ctx: &mut CpuContext) -> HleResult {
    let sem = ctx.rdi();
    let s = lookup!(semaphores, sem);
    let mut g = lock(&s.inner);
    if g.waiters != 0 {
        return ERR_BUSY;
    }
    g.alive = false;
    drop(g);
    tables(|t| t.semaphores.remove(&sem));
    0
}

// --- condition variable ---

/// _sceUltConditionVariableCreate(cv, name, waitingPool, opt, build)
fn cb_cond_create(ctx: &mut CpuContext) -> HleResult {
    let (c, pool) = (ctx.rdi(), ctx.rdx());
    if let Err(e) = new_object(c, 8) {
        return e;
    }
    tables(|t| {
        if let Err(e) = check_waiting_pool(t, pool) {
            return e;
        }
        t.conds.insert(c, Arc::new(UltCond::default()));
        0
    })
}

/// sceUltConditionVariableWait(cv, mutex)
fn cb_cond_wait(ctx: &mut CpuContext) -> HleResult {
    let c = lookup!(conds, ctx.rdi());
    let m = lookup!(mutexes, ctx.rsi());
    // Register as a waiter before releasing the mutex so a signal sent
    // right after the release is not lost.
    let mut g = lock(&c.inner);
    let seq = g.seq;
    g.waiters += 1;
    let count = match m.release_all() {
        Ok(n) => n,
        Err(e) => {
            g.waiters -= 1;
            return e;
        }
    };
    loop {
        if g.signals > 0 && g.seq != seq {
            g.signals -= 1;
            break;
        }
        g = c.cv.wait(g).unwrap_or_else(|p| p.into_inner());
    }
    g.waiters -= 1;
    drop(g);
    m.reacquire(count);
    0
}

fn cond_signal(ctx: &mut CpuContext, all: bool) -> HleResult {
    let c = lookup!(conds, ctx.rdi());
    let mut g = lock(&c.inner);
    let pending = g.waiters - g.signals.min(g.waiters);
    if pending == 0 {
        return 0;
    }
    g.seq += 1;
    g.signals += if all { pending } else { 1 };
    drop(g);
    c.cv.notify_all();
    0
}

fn cb_cond_signal(ctx: &mut CpuContext) -> HleResult {
    cond_signal(ctx, false)
}

fn cb_cond_signal_all(ctx: &mut CpuContext) -> HleResult {
    cond_signal(ctx, true)
}

fn cb_cond_destroy(ctx: &mut CpuContext) -> HleResult {
    let cv = ctx.rdi();
    let c = lookup!(conds, cv);
    if lock(&c.inner).waiters != 0 {
        return ERR_BUSY;
    }
    tables(|t| t.conds.remove(&cv));
    0
}

// --- reader/writer lock ---

/// _sceUltReaderWriterLockCreate(rwlock, name, waitingPool, opt, build)
fn cb_rw_create(ctx: &mut CpuContext) -> HleResult {
    let (rw, pool) = (ctx.rdi(), ctx.rdx());
    if let Err(e) = new_object(rw, 8) {
        return e;
    }
    tables(|t| {
        if let Err(e) = check_waiting_pool(t, pool) {
            return e;
        }
        t.rwlocks.insert(rw, Arc::new(UltRwLock::default()));
        0
    })
}

fn rw_lock(ctx: &mut CpuContext, write: bool, try_only: bool) -> HleResult {
    let l = lookup!(rwlocks, ctx.rdi());
    let me = self_id();
    let mut g = lock(&l.inner);
    let blocked = |g: &RwInner| if write { g.writer != 0 || g.readers != 0 } else { g.writer != 0 };
    if g.writer == me {
        return ERR_PERMISSION;
    }
    if blocked(&g) {
        if try_only {
            return ERR_BUSY;
        }
        g.waiters += 1;
        while blocked(&g) {
            g = l.cv.wait(g).unwrap_or_else(|p| p.into_inner());
        }
        g.waiters -= 1;
    }
    if write {
        g.writer = me;
    } else {
        g.readers += 1;
    }
    0
}

fn cb_rw_lock_read(ctx: &mut CpuContext) -> HleResult {
    rw_lock(ctx, false, false)
}
fn cb_rw_lock_write(ctx: &mut CpuContext) -> HleResult {
    rw_lock(ctx, true, false)
}
fn cb_rw_try_lock_read(ctx: &mut CpuContext) -> HleResult {
    rw_lock(ctx, false, true)
}
fn cb_rw_try_lock_write(ctx: &mut CpuContext) -> HleResult {
    rw_lock(ctx, true, true)
}

fn rw_unlock(ctx: &mut CpuContext, write: bool) -> HleResult {
    let l = lookup!(rwlocks, ctx.rdi());
    let mut g = lock(&l.inner);
    if write {
        if g.writer != self_id() {
            return ERR_PERMISSION;
        }
        g.writer = 0;
    } else {
        if g.readers == 0 {
            return ERR_PERMISSION;
        }
        g.readers -= 1;
    }
    drop(g);
    l.cv.notify_all();
    0
}

fn cb_rw_unlock_read(ctx: &mut CpuContext) -> HleResult {
    rw_unlock(ctx, false)
}
fn cb_rw_unlock_write(ctx: &mut CpuContext) -> HleResult {
    rw_unlock(ctx, true)
}

fn cb_rw_destroy(ctx: &mut CpuContext) -> HleResult {
    let rw = ctx.rdi();
    let l = lookup!(rwlocks, rw);
    let g = lock(&l.inner);
    if g.writer != 0 || g.readers != 0 || g.waiters != 0 {
        return ERR_BUSY;
    }
    drop(g);
    tables(|t| t.rwlocks.remove(&rw));
    0
}

// --- queue data resource pool & queues ---

fn cb_data_pool_work_area_size(ctx: &mut CpuContext) -> HleResult {
    let (num, size, queues) = (ctx.rdi() as u32 as u64, ctx.rsi(), ctx.rdx() as u32 as u64);
    let stride = align_up(size.min(u64::MAX - 7), 8);
    let total = num.saturating_mul(stride).saturating_add(queues * 512);
    align_up(total.min(u64::MAX - 7), 8) as i64
}

/// _sceUltQueueDataResourcePoolCreate(pool, name, numData, dataSize,
/// numQueueObject, waitingPool, workArea, opt, build)
fn cb_data_pool_create(ctx: &mut CpuContext) -> HleResult {
    let (pool, num, wpool) = (ctx.rdi(), ctx.rdx() as u32, ctx.r9());
    let _work_area = stack_arg(ctx, 0);
    if pool == 0 {
        return ERR_NULL;
    }
    tables(|t| {
        if let Err(e) = check_waiting_pool(t, wpool) {
            return e;
        }
        clear_object(ctx, pool, 512);
        let inner = DataPoolInner { num_data: num, queues: 0, used: 0, waiting_pool: wpool };
        t.data_pools.insert(pool, Arc::new(DataPool { inner: Mutex::new(inner), space: Condvar::new() }));
        0
    })
}

fn cb_data_pool_destroy(ctx: &mut CpuContext) -> HleResult {
    let pool = ctx.rdi();
    if pool == 0 {
        return ERR_NULL;
    }
    tables(|t| {
        let Some(p) = t.data_pools.get(&pool) else { return ERR_STATE };
        if lock(&p.inner).queues != 0 {
            return ERR_BUSY;
        }
        t.data_pools.remove(&pool);
        0
    })
}

/// _sceUltQueueCreate(queue, name, dataSize, waitingPool, dataPool, opt, build)
fn cb_queue_create(ctx: &mut CpuContext) -> HleResult {
    let (q, size, wpool, dpool) = (ctx.rdi(), ctx.rdx(), ctx.rcx(), ctx.r8());
    if q == 0 {
        return ERR_NULL;
    }
    tables(|t| {
        let Some(pool) = t.data_pools.get(&dpool).cloned() else { return ERR_INVALID };
        if let Err(e) = check_waiting_pool(t, wpool) {
            return e;
        }
        clear_object(ctx, q, 512);
        if let Some(old) = t.queues.remove(&q) {
            release_queue(&old);
        }
        lock(&pool.inner).queues += 1;
        let inner = QueueInner { items: VecDeque::new(), waiters: 0, alive: true };
        t.queues.insert(q, Arc::new(UltQueue { pool, data_size: size, inner: Mutex::new(inner), data: Condvar::new() }));
        0
    })
}

fn queue_push(ctx: &mut CpuContext, blocking: bool) -> HleResult {
    let q = lookup!(queues, ctx.rdi());
    let src = ctx.rsi();
    if src == 0 && q.data_size != 0 {
        return ERR_NULL;
    }
    let bytes = if q.data_size == 0 {
        Vec::new()
    } else {
        let Some(b) = ctx.read_bytes(src, q.data_size as usize) else { return ERR_INVALID };
        b
    };
    // Claim a slot of the shared pool (a pool without data slots is unbounded).
    {
        let mut p = lock(&q.pool.inner);
        while p.num_data != 0 && p.used == p.num_data {
            if !blocking {
                return ERR_AGAIN;
            }
            lock(&q.inner).waiters += 1;
            p = q.pool.space.wait(p).unwrap_or_else(|e| e.into_inner());
            let mut qi = lock(&q.inner);
            qi.waiters -= 1;
            if !qi.alive {
                return ERR_STATE;
            }
        }
        if !lock(&q.inner).alive {
            return ERR_STATE;
        }
        p.used += 1;
    }
    lock(&q.inner).items.push_back(bytes);
    q.data.notify_one();
    0
}

fn queue_pop(ctx: &mut CpuContext, blocking: bool) -> HleResult {
    let q = lookup!(queues, ctx.rdi());
    let dst = ctx.rsi();
    if dst == 0 && q.data_size != 0 {
        return ERR_NULL;
    }
    let item = {
        let mut g = lock(&q.inner);
        loop {
            if !g.alive {
                return ERR_STATE;
            }
            if let Some(i) = g.items.pop_front() {
                break i;
            }
            if !blocking {
                return ERR_AGAIN;
            }
            g.waiters += 1;
            g = q.data.wait(g).unwrap_or_else(|p| p.into_inner());
            g.waiters -= 1;
        }
    };
    lock(&q.pool.inner).used -= 1;
    q.pool.space.notify_one();
    if item.is_empty() || ctx.write_bytes(dst, &item) { 0 } else { ERR_INVALID }
}

fn cb_queue_push(ctx: &mut CpuContext) -> HleResult {
    queue_push(ctx, true)
}
fn cb_queue_try_push(ctx: &mut CpuContext) -> HleResult {
    queue_push(ctx, false)
}
fn cb_queue_pop(ctx: &mut CpuContext) -> HleResult {
    queue_pop(ctx, true)
}
fn cb_queue_try_pop(ctx: &mut CpuContext) -> HleResult {
    queue_pop(ctx, false)
}

/// Drop a queue's items and its claim on the data pool.
fn release_queue(q: &UltQueue) {
    let mut g = lock(&q.inner);
    g.alive = false;
    let n = g.items.len() as u32;
    g.items.clear();
    drop(g);
    {
        let mut p = lock(&q.pool.inner);
        p.used -= n;
        p.queues -= 1;
    }
    q.pool.space.notify_all();
    q.data.notify_all();
}

fn cb_queue_destroy(ctx: &mut CpuContext) -> HleResult {
    let qa = ctx.rdi();
    let q = lookup!(queues, qa);
    let g = lock(&q.inner);
    if g.waiters != 0 {
        return ERR_BUSY;
    }
    drop(g);
    release_queue(&q);
    tables(|t| t.queues.remove(&qa));
    0
}

// --- ulthread runtime & ulthreads ---

fn cb_runtime_work_area_size(ctx: &mut CpuContext) -> HleResult {
    let (max, workers) = (ctx.rdi() as u32 as u64, ctx.rsi() as u32 as u64);
    align_up(max * 256 + workers * 16 * 1024, 8) as i64
}

/// _sceUltUlthreadRuntimeCreate(runtime, name, maxNumUlthread, numWorkerThread, workArea, opt, build)
fn cb_runtime_create(ctx: &mut CpuContext) -> HleResult {
    let (rt, max, workers) = (ctx.rdi(), ctx.rdx() as u32, ctx.rcx() as u32);
    if rt == 0 {
        return ERR_NULL;
    }
    clear_object(ctx, rt, 4096);
    tables(|t| t.runtimes.insert(rt, (max, workers)));
    0
}

/// A runtime can be destroyed once every ulthread created on it is joined.
fn destroy_runtime(t: &mut Tables, rt: u64) -> HleResult {
    if !t.runtimes.contains_key(&rt) {
        return ERR_STATE;
    }
    if t.ulthreads.values().any(|u| u.runtime == rt) {
        return ERR_BUSY;
    }
    t.runtimes.remove(&rt);
    0
}

/// sceUltUlthreadRuntimeDestroy(runtime)
fn cb_runtime_destroy(ctx: &mut CpuContext) -> HleResult {
    let rt = ctx.rdi();
    if rt == 0 {
        return ERR_NULL;
    }
    tables(|t| destroy_runtime(t, rt))
}

/// _sceUltUlthreadCreate(ulthread, name, entry, arg, context, sizeContext,
/// runtime, opt, build). The ulthread runs as a guest thread executing
/// `int32_t entry(uint64_t arg)`.
fn cb_ulthread_create(ctx: &mut CpuContext) -> HleResult {
    let (ut, name, entry, arg, context) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8());
    let runtime = stack_arg(ctx, 0);
    if ut == 0 || entry == 0 || context == 0 || runtime == 0 {
        return ERR_NULL;
    }
    let ok = tables(|t| {
        if !t.runtimes.contains_key(&runtime) {
            return ERR_STATE;
        }
        if t.ulthreads.contains_key(&ut) {
            return ERR_STATE;
        }
        clear_object(ctx, ut, 512);
        t.ulthreads.insert(ut, Ulthread { handle: 0, runtime });
        0
    });
    if ok != 0 {
        return ok;
    }
    // The pthread layer writes the new handle into the ulthread object's
    // first word; it is kept host-side as well.
    let r = crate::pthread::create_thread(ctx, ut, 0, entry, arg, name);
    if r != 0 {
        tables(|t| t.ulthreads.remove(&ut));
        return ERR_AGAIN;
    }
    let handle = ctx.read_u64(ut).unwrap_or(0);
    tables(|t| {
        if let Some(u) = t.ulthreads.get_mut(&ut) {
            u.handle = handle;
        }
        t.by_handle.insert(handle, ut);
    });
    0
}

fn ulthread_join(ctx: &mut CpuContext, block: bool) -> HleResult {
    let (ut, status) = (ctx.rdi(), ctx.rsi());
    if ut == 0 {
        return ERR_NULL;
    }
    let Some(handle) = tables(|t| t.ulthreads.get(&ut).map(|u| u.handle)) else { return ERR_STATE };
    match crate::pthread::join_handle(handle, block) {
        Ok(Some(v)) => {
            if status != 0 {
                ctx.write_u32(status, v as u32);
            }
            tables(|t| {
                t.ulthreads.remove(&ut);
                t.by_handle.remove(&handle);
            });
            0
        }
        Ok(None) => ERR_BUSY,
        Err(_) => ERR_STATE,
    }
}

fn cb_ulthread_join(ctx: &mut CpuContext) -> HleResult {
    ulthread_join(ctx, true)
}

fn cb_ulthread_try_join(ctx: &mut CpuContext) -> HleResult {
    ulthread_join(ctx, false)
}

/// sceUltUlthreadExit(status)
fn cb_ulthread_exit(ctx: &mut CpuContext) -> HleResult {
    let me = crate::pthread::current_handle();
    if tables(|t| !t.by_handle.contains_key(&me)) {
        return ERR_PERMISSION;
    }
    ps5cpu::exec::exit_guest_thread(ctx.rdi() as u32 as u64)
}

/// sceUltUlthreadGetSelf(ulthread**)
fn cb_ulthread_get_self(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_NULL;
    }
    let me = crate::pthread::current_handle();
    match tables(|t| t.by_handle.get(&me).copied()) {
        Some(ut) => {
            ctx.write_u64(out, ut);
            0
        }
        None => ERR_PERMISSION,
    }
}

fn cb_ulthread_yield(_ctx: &mut CpuContext) -> HleResult {
    std::thread::yield_now();
    0
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("hZIg1EWGsHM", "sceUltInitialize", cb_initialize),
    ("d-kSG2fLrvI", "sceUltFinalize", cb_finalize),
    ("LuLTRt0rfTw", "_sceUltWaitingQueueResourcePoolOptParamInitialize", cb_opt_param_initialize),
    ("WIWV1Qd7PFU", "sceUltWaitingQueueResourcePoolGetWorkAreaSize", cb_waiting_pool_work_area_size),
    ("YiHujOG9vXY", "_sceUltWaitingQueueResourcePoolCreate", cb_waiting_pool_create),
    ("or55417wcDk", "sceUltWaitingQueueResourcePoolDestroy", cb_waiting_pool_destroy),
    ("1+8t9aHLiz8", "_sceUltMutexOptParamInitialize", cb_mutex_opt_param_initialize),
    ("mmt8Sa6tL6c", "_sceUltMutexCreate", cb_mutex_create),
    ("8hEGkR1pfr8", "sceUltMutexLock", cb_mutex_lock),
    ("jOsUG0BJI-Y", "sceUltMutexTryLock", cb_mutex_try_lock),
    ("h0XebKiMBtk", "sceUltMutexUnlock", cb_mutex_unlock),
    ("jW+HnafeS3Y", "sceUltMutexDestroy", cb_mutex_destroy),
    ("NPRRPNKDBN0", "_sceUltSemaphoreOptParamInitialize", cb_opt_param_initialize),
    ("h5QlIYj+Ro8", "_sceUltSemaphoreCreate", cb_semaphore_create),
    ("QAH1ofI97vU", "sceUltSemaphoreAcquire", cb_semaphore_acquire),
    ("HA1Ldbi3lPY", "sceUltSemaphoreTryAcquire", cb_semaphore_try_acquire),
    ("lbtk5X1mecw", "sceUltSemaphoreRelease", cb_semaphore_release),
    ("izXyehpoZGo", "sceUltSemaphoreDestroy", cb_semaphore_destroy),
    ("RVmEia0vXMI", "_sceUltConditionVariableOptParamInitialize", cb_opt_param_initialize),
    ("jnKaHGkrxZ4", "_sceUltConditionVariableCreate", cb_cond_create),
    ("5xGAHCxA8M0", "sceUltConditionVariableWait", cb_cond_wait),
    ("JTw1cAVkuc0", "sceUltConditionVariableSignal", cb_cond_signal),
    ("byiceqcMvV0", "sceUltConditionVariableSignalAll", cb_cond_signal_all),
    ("xrmmI832R4U", "sceUltConditionVariableDestroy", cb_cond_destroy),
    ("Gw7yn0CEmv8", "_sceUltReaderWriterLockOptParamInitialize", cb_opt_param_initialize),
    ("iIfTXvh1hiM", "_sceUltReaderWriterLockCreate", cb_rw_create),
    ("Hb9HWFKo9F4", "sceUltReaderWriterLockLockRead", cb_rw_lock_read),
    ("RgKmNey20Ns", "sceUltReaderWriterLockLockWrite", cb_rw_lock_write),
    ("J7Xs-UluzGk", "sceUltReaderWriterLockTryLockRead", cb_rw_try_lock_read),
    ("9Sh0Kk7Xf4w", "sceUltReaderWriterLockTryLockWrite", cb_rw_try_lock_write),
    ("8Ssk4OU38vw", "sceUltReaderWriterLockUnlockRead", cb_rw_unlock_read),
    ("gcFCn5J5DXY", "sceUltReaderWriterLockUnlockWrite", cb_rw_unlock_write),
    ("gCA-D2wiiD0", "sceUltReaderWriterLockDestroy", cb_rw_destroy),
    ("6gYjd50q0CE", "_sceUltQueueDataResourcePoolOptParamInitialize", cb_opt_param_initialize),
    ("evj9YPkS8s4", "sceUltQueueDataResourcePoolGetWorkAreaSize", cb_data_pool_work_area_size),
    ("TFHm6-N6vks", "_sceUltQueueDataResourcePoolCreate", cb_data_pool_create),
    ("dh11uAUWNyM", "sceUltQueueDataResourcePoolDestroy", cb_data_pool_destroy),
    ("TkASc9I-xX0", "_sceUltQueueOptParamInitialize", cb_opt_param_initialize),
    ("9Y5keOvb6ok", "_sceUltQueueCreate", cb_queue_create),
    ("dUwpX3e5NDE", "sceUltQueuePush", cb_queue_push),
    ("6Mc2Xs7pI1I", "sceUltQueueTryPush", cb_queue_try_push),
    ("RVSq2tsm2yw", "sceUltQueuePop", cb_queue_pop),
    ("uZz3ci7XYqc", "sceUltQueueTryPop", cb_queue_try_pop),
    ("PP9nZxpSKLY", "sceUltQueueDestroy", cb_queue_destroy),
    ("V2u3WLrwh64", "_sceUltUlthreadRuntimeOptParamInitialize", cb_runtime_opt_param_initialize),
    ("grs2pbc2awM", "sceUltUlthreadRuntimeGetWorkAreaSize", cb_runtime_work_area_size),
    ("jw9FkZBXo-g", "_sceUltUlthreadRuntimeCreate", cb_runtime_create),
    ("-gxcs521SvA", "sceUltUlthreadRuntimeDestroy", cb_runtime_destroy),
    ("vIuen2qlRmo", "_sceUltUlthreadOptParamInitialize", cb_opt_param_initialize),
    ("znI3q8S7KQ4", "_sceUltUlthreadCreate", cb_ulthread_create),
    ("gCeAI57LGgI", "sceUltUlthreadJoin", cb_ulthread_join),
    ("DsW+3FTXL0Q", "sceUltUlthreadTryJoin", cb_ulthread_try_join),
    ("kF--s3xZjco", "sceUltUlthreadExit", cb_ulthread_exit),
    ("xOQ9oACobA4", "sceUltUlthreadGetSelf", cb_ulthread_get_self),
    ("HFd-lpjGxJA", "sceUltUlthreadYield", cb_ulthread_yield),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    fn call(ctx: &mut CpuContext, f: HleCallback, args: &[u64]) -> HleResult {
        for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
            ctx.set(*r, *v);
        }
        f(ctx)
    }

    #[test]
    fn queue_fifo_and_capacity() {
        let vm = ps5vm::VirtualMemory::new();
        const B: u64 = 0x5A62_0000_0000;
        vm.map(B, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (pool, q, val) = (B, B + 0x1000, B + 0x2000);
        // The stack argument area (work area) is read from rsp.
        ctx.set(Rsp, B + 0x3000);
        assert_eq!(call(&mut ctx, cb_data_pool_create, &[0, 0, 2, 8, 1, 0]), ERR_NULL);
        assert_eq!(call(&mut ctx, cb_data_pool_create, &[pool, 0, 2, 8, 1, 0x1234]), ERR_INVALID);
        assert_eq!(call(&mut ctx, cb_data_pool_create, &[pool, 0, 2, 8, 1, 0]), 0);
        assert_eq!(call(&mut ctx, cb_queue_create, &[0, 0, 8, 0, pool]), ERR_NULL);
        assert_eq!(call(&mut ctx, cb_queue_create, &[q, 0, 8, 0, B + 0x8000]), ERR_INVALID);
        assert_eq!(call(&mut ctx, cb_queue_create, &[q, 0, 8, 0x1234, pool]), ERR_INVALID);
        ctx.write_u64(q, u64::MAX);
        assert_eq!(call(&mut ctx, cb_queue_create, &[q, 0, 8, 0, pool]), 0);
        // The queue object is cleared on creation.
        assert_eq!(ctx.read_u64(q), Some(0));
        // A null data pointer is rejected for a queue with element data.
        assert_eq!(call(&mut ctx, cb_queue_try_push, &[q, 0]), ERR_NULL);
        assert_eq!(call(&mut ctx, cb_queue_try_pop, &[q, 0]), ERR_NULL);
        for v in [11u64, 22] {
            ctx.write_u64(val, v);
            assert_eq!(call(&mut ctx, cb_queue_try_push, &[q, val]), 0);
        }
        assert_eq!(call(&mut ctx, cb_queue_try_push, &[q, val]), ERR_AGAIN);
        assert_eq!(call(&mut ctx, cb_queue_try_pop, &[q, val]), 0);
        assert_eq!(ctx.read_u64(val), Some(11));
        assert_eq!(call(&mut ctx, cb_queue_pop, &[q, val]), 0);
        assert_eq!(ctx.read_u64(val), Some(22));
        assert_eq!(call(&mut ctx, cb_queue_try_pop, &[q, val]), ERR_AGAIN);
        assert_eq!(call(&mut ctx, cb_data_pool_destroy, &[pool]), ERR_BUSY);
        assert_eq!(call(&mut ctx, cb_queue_destroy, &[q]), 0);
        assert_eq!(call(&mut ctx, cb_data_pool_destroy, &[pool]), 0);
    }

    #[test]
    fn mutex_semaphore_rwlock() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        let (m, s, rw) = (0x5A63_0000_1000u64, 0x5A63_0000_2000u64, 0x5A63_0000_3000u64);
        assert_eq!(call(&mut ctx, cb_mutex_create, &[m, 0, 0x1234]), ERR_INVALID);
        assert_eq!(call(&mut ctx, cb_mutex_create, &[m, 0, 0]), 0);
        assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), 0);
        assert_eq!(call(&mut ctx, cb_mutex_try_lock, &[m]), 0); // recursive
        let other = std::thread::scope(|sc| sc.spawn(|| { let mut c = CpuContext::new(&vm); call(&mut c, cb_mutex_try_lock, &[m]) }).join().unwrap());
        assert_eq!(other, ERR_BUSY);
        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), 0);
        assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), ERR_BUSY);
        assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), 0);
        assert_eq!(call(&mut ctx, cb_mutex_destroy, &[m]), 0);

        assert_eq!(call(&mut ctx, cb_semaphore_create, &[s + 4, 0, 1, 0]), ERR_ALIGNMENT);
        assert_eq!(call(&mut ctx, cb_semaphore_create, &[s, 0, 1, 0, 4]), ERR_ALIGNMENT);
        assert_eq!(call(&mut ctx, cb_semaphore_create, &[s, 0, -1i64 as u64, 0, 0]), ERR_RANGE);
        assert_eq!(call(&mut ctx, cb_semaphore_create, &[s, 0, 1, 0, 0]), 0);
        assert_eq!(call(&mut ctx, cb_semaphore_create, &[s, 0, 1, 0, 0]), ERR_STATE);
        assert_eq!(call(&mut ctx, cb_semaphore_try_acquire, &[s, 2]), ERR_AGAIN);
        assert_eq!(call(&mut ctx, cb_semaphore_acquire, &[s, 1]), 0);
        assert_eq!(call(&mut ctx, cb_semaphore_release, &[s, 0]), ERR_RANGE);
        assert_eq!(call(&mut ctx, cb_semaphore_release, &[s, 3]), 0);
        assert_eq!(call(&mut ctx, cb_semaphore_try_acquire, &[s, 3]), 0);
        assert_eq!(call(&mut ctx, cb_semaphore_destroy, &[s]), 0);
        assert_eq!(call(&mut ctx, cb_semaphore_acquire, &[s, 1]), ERR_STATE);

        assert_eq!(call(&mut ctx, cb_rw_create, &[rw, 0, 0]), 0);
        assert_eq!(call(&mut ctx, cb_rw_lock_read, &[rw]), 0);
        assert_eq!(call(&mut ctx, cb_rw_try_lock_write, &[rw]), ERR_BUSY);
        assert_eq!(call(&mut ctx, cb_rw_unlock_read, &[rw]), 0);
        assert_eq!(call(&mut ctx, cb_rw_lock_write, &[rw]), 0);
        assert_eq!(call(&mut ctx, cb_rw_try_lock_read, &[rw]), ERR_PERMISSION);
        assert_eq!(call(&mut ctx, cb_rw_unlock_write, &[rw]), 0);
        assert_eq!(call(&mut ctx, cb_rw_destroy, &[rw]), 0);
    }

    #[test]
    fn runtime_and_ulthread_argument_checks() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D51_0000_0000;
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (rt, ut, opt) = (BASE, BASE + 0x1000, BASE + 0x2003);
        ctx.set(Rsp, BASE + 0x3000);
        // Option parameters: only null is rejected, then cleared.
        assert_eq!(call(&mut ctx, cb_runtime_opt_param_initialize, &[0]), ERR_NULL);
        ctx.write_bytes(opt, &[0xAA; 128]);
        assert_eq!(call(&mut ctx, cb_runtime_opt_param_initialize, &[opt]), 0);
        assert_eq!(ctx.read_bytes(opt, 128).unwrap(), vec![0; 128]);
        assert_eq!(call(&mut ctx, cb_mutex_opt_param_initialize, &[opt]), 0);
        // Runtime creation accepts any sizes, clears the object and may be repeated.
        assert_eq!(call(&mut ctx, cb_runtime_create, &[0, 0, 1, 1]), ERR_NULL);
        ctx.write_bytes(rt, &[0xAA; 16]);
        assert_eq!(call(&mut ctx, cb_runtime_create, &[rt, 0, 0, 0]), 0);
        assert_eq!(ctx.read_bytes(rt, 16).unwrap(), vec![0; 16]);
        assert_eq!(call(&mut ctx, cb_runtime_create, &[rt, 0, 4, 1]), 0);
        // Ulthread creation needs the object, entry, context and runtime.
        ctx.write_u64(BASE + 0x3008, rt);
        assert_eq!(call(&mut ctx, cb_ulthread_create, &[ut, 0, 0x1234, 0, 0, 0x100]), ERR_NULL);
        ctx.write_u64(BASE + 0x3008, BASE + 0x100);
        assert_eq!(call(&mut ctx, cb_ulthread_create, &[ut, 0, 0x1234, 0, BASE + 0x2800, 0x100]), ERR_STATE);
        assert_eq!(call(&mut ctx, cb_ulthread_join, &[0, 0]), ERR_NULL);
        assert_eq!(call(&mut ctx, cb_ulthread_join, &[ut, 0]), ERR_STATE);
    }

    /// Finalize refuses to tear the library down while an ulthread is not
    /// joined, and leaves every object in place.
    #[test]
    fn finalize_is_busy_with_outstanding_ulthreads() {
        let mut slot = Some(Tables::default());
        let t = slot.as_mut().unwrap();
        t.runtimes.insert(0x1000, (1, 1));
        t.ulthreads.insert(0x2000, Ulthread { handle: 0x77, runtime: 0x1000 });
        t.by_handle.insert(0x77, 0x2000);
        assert_eq!(take_tables(&mut slot).err(), Some(ERR_BUSY));
        let t = slot.as_mut().expect("busy finalize tore the tables down");
        assert!(t.runtimes.contains_key(&0x1000) && t.ulthreads.contains_key(&0x2000));
        // Joining removes the ulthread; finalize then succeeds.
        t.ulthreads.remove(&0x2000);
        t.by_handle.remove(&0x77);
        let taken = take_tables(&mut slot).unwrap();
        assert!(taken.runtimes.contains_key(&0x1000));
        assert!(slot.is_none());
        assert!(take_tables(&mut slot).is_ok());
    }

    /// Destroying a runtime: unknown runtimes are a state error, a runtime
    /// with an unjoined ulthread is busy, other runtimes are unaffected.
    #[test]
    fn runtime_destroy_waits_for_its_ulthreads() {
        let mut t = Tables::default();
        let (rt, other, ut) = (0x1000, 0x1100, 0x2000);
        assert_eq!(destroy_runtime(&mut t, rt), ERR_STATE);
        t.runtimes.insert(rt, (1, 1));
        t.runtimes.insert(other, (1, 1));
        t.ulthreads.insert(ut, Ulthread { handle: 0x77, runtime: rt });
        assert_eq!(destroy_runtime(&mut t, other), 0);
        assert_eq!(destroy_runtime(&mut t, rt), ERR_BUSY);
        assert!(t.runtimes.contains_key(&rt));
        t.ulthreads.remove(&ut);
        assert_eq!(destroy_runtime(&mut t, rt), 0);
        assert_eq!(destroy_runtime(&mut t, rt), ERR_STATE);
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        assert_eq!(call(&mut ctx, cb_runtime_destroy, &[0]), ERR_NULL);
    }

    #[test]
    fn condition_variable_wakes_waiter() {
        let vm = ps5vm::VirtualMemory::new();
        let (m, c) = (0x5A64_0000_1000u64, 0x5A64_0000_2000u64);
        let mut ctx = CpuContext::new(&vm);
        assert_eq!(call(&mut ctx, cb_mutex_create, &[m, 0, 0]), 0);
        assert_eq!(call(&mut ctx, cb_cond_create, &[c, 0, 0]), 0);
        static FLAG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|sc| {
            let h = sc.spawn(|| {
                let mut c2 = CpuContext::new(&vm);
                assert_eq!(call(&mut c2, cb_mutex_lock, &[m]), 0);
                while !FLAG.load(std::sync::atomic::Ordering::SeqCst) {
                    assert_eq!(call(&mut c2, cb_cond_wait, &[c, m]), 0);
                }
                assert_eq!(call(&mut c2, cb_mutex_unlock, &[m]), 0);
            });
            std::thread::sleep(std::time::Duration::from_millis(20));
            assert_eq!(call(&mut ctx, cb_mutex_lock, &[m]), 0);
            FLAG.store(true, std::sync::atomic::Ordering::SeqCst);
            assert_eq!(call(&mut ctx, cb_cond_signal, &[c]), 0);
            assert_eq!(call(&mut ctx, cb_mutex_unlock, &[m]), 0);
            h.join().unwrap();
        });
        assert_eq!(call(&mut ctx, cb_cond_destroy, &[c]), 0);
    }
}
