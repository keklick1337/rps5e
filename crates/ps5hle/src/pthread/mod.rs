// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libkernel pthreads: one host thread per guest thread.
//!
//! Guest-visible objects:
//!   - `pthread_t` is the address of the thread's pthread object inside its
//!     thread block (`tls::HANDLE_OFFSET`); the object starts with the
//!     64-bit thread id.
//!   - `pthread_mutex_t`, `pthread_cond_t`, `pthread_rwlock_t` and the attr
//!     types hold a pointer to a 0x100-byte opaque guest object; the real
//!     state lives on the host (see `sync`). A zero mutex/cond/rwlock is
//!     created lazily (static initializer); a mutex value of 1 is the
//!     adaptive static initializer.
//!
//! Blocking operations block the calling host thread on host primitives.
//! `scePthread*` exports return 0 or a `0x8002xxxx` kernel error code; the
//! POSIX-named `pthread_*` exports return the POSIX error number instead.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
mod posix_ext;
pub mod sync;

use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};
use crate::tls;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;
use sync::*;
pub(crate) use sync::abandon_running_once;

const OK: HleResult = 0;
const EPERM: HleResult = 0x8002_0001u32 as i32 as i64;
const ESRCH: HleResult = 0x8002_0003u32 as i32 as i64;
const EINVAL: HleResult = 0x8002_0016u32 as i32 as i64;
const EDEADLK: HleResult = 0x8002_000Bu32 as i32 as i64;
const ENOMEM: HleResult = 0x8002_000Cu32 as i32 as i64;
const EFAULT: HleResult = 0x8002_000Eu32 as i32 as i64;
const ENAMETOOLONG: HleResult = 0x8002_003Fu32 as i32 as i64;
const EBUSY: HleResult = 0x8002_0010u32 as i32 as i64;
const EAGAIN: HleResult = 0x8002_0023u32 as i32 as i64;
const ETIMEDOUT: HleResult = 0x8002_003Cu32 as i32 as i64;
const EOPNOTSUPP: HleResult = 0x8002_002Du32 as i32 as i64;

const DEFAULT_STACK_SIZE: u64 = 0x10_0000;
const STACK_HEADROOM: u64 = 0x10_0000;
/// Largest guest stack: a stack slot less the headroom below it.
const MAX_STACK_SIZE: u64 = STACK_SLOT - STACK_HEADROOM;
const DEFAULT_GUARD_SIZE: u64 = 0x1000;
const DEFAULT_PRIORITY: i32 = 700;
const DEFAULT_POLICY: i32 = 1;
const DEFAULT_INHERIT: i32 = 4;
/// PTHREAD_EXPLICIT_SCHED: a new thread takes the attributes' scheduling.
const EXPLICIT_SCHED: i32 = 0;
const SCHED_FIFO: i32 = 1;
const SCHED_RR: i32 = 3;
/// The priority range thread attributes accept (sched_get_priority_min /
/// max of SCHED_FIFO and SCHED_RR).
const ATTR_PRIORITY: std::ops::RangeInclusive<i32> = 256..=767;
/// Solo scheduling values the attributes accept: off and on.
const SOLOSCHED_OFF: i32 = 0;
const SOLOSCHED_ON: i32 = 16;
const DEFAULT_AFFINITY: u64 = 0x7F;
/// The process cpuset (cpuset_setaffinity of the process): the affinity of
/// threads that no guest thread creates.
static PROCESS_AFFINITY: AtomicU64 = AtomicU64::new(DEFAULT_AFFINITY);
const NAME_MAX: usize = 32;
/// Smallest stack a thread attribute accepts.
const STACK_MIN: u64 = 0x4000;

/// Guest stacks: one 256 MiB slot per thread-block slot.
const STACK_REGION: u64 = 0x6E80_0000_0000;
const STACK_SLOT: u64 = 0x1000_0000;

// ---------------------------------------------------------------------------
// Opaque guest objects
// ---------------------------------------------------------------------------

const OBJECT_REGION: u64 = 0x6E00_0000_0000;
const OBJECT_REGION_SIZE: u64 = 0x4000_0000;
const OBJECT_SIZE: u64 = 0x100;

struct ObjectPool {
    mapped: bool,
    next: u64,
    free: Vec<u64>,
}

static OBJECTS: Mutex<ObjectPool> = Mutex::new(ObjectPool { mapped: false, next: OBJECT_REGION, free: Vec::new() });

/// Allocate one zeroed 0x100-byte guest object.
pub(crate) fn alloc_object(ctx: &CpuContext) -> Option<u64> {
    let mut p = OBJECTS.lock().unwrap();
    if !ensure_object_region(&mut p) {
        return None;
    }
    let addr = match p.free.pop() {
        Some(a) => a,
        None => {
            if p.next + OBJECT_SIZE > OBJECT_REGION + OBJECT_REGION_SIZE {
                return None;
            }
            let a = p.next;
            p.next += OBJECT_SIZE;
            a
        }
    };
    drop(p);
    ctx.write_bytes(addr, &[0u8; OBJECT_SIZE as usize]);
    Some(addr)
}

/// Allocate a zeroed, never-freed guest block of `size` bytes (rounded up
/// to whole objects, so it is 0x100-aligned) from the same region.
pub(crate) fn alloc_guest_block(ctx: &CpuContext, size: u64) -> Option<u64> {
    let len = size.max(1).div_ceil(OBJECT_SIZE) * OBJECT_SIZE;
    let mut p = OBJECTS.lock().unwrap();
    if !ensure_object_region(&mut p) || p.next + len > OBJECT_REGION + OBJECT_REGION_SIZE {
        return None;
    }
    let addr = p.next;
    p.next += len;
    drop(p);
    ctx.write_bytes(addr, &vec![0u8; len as usize]);
    Some(addr)
}

fn ensure_object_region(p: &mut ObjectPool) -> bool {
    if !p.mapped {
        let r = unsafe {
            libc::mmap(
                OBJECT_REGION as *mut libc::c_void,
                OBJECT_REGION_SIZE as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | libc::MAP_FIXED_NOREPLACE,
                -1,
                0,
            )
        };
        if r == libc::MAP_FAILED || r as u64 != OBJECT_REGION {
            log::error!("[pthread] cannot map object region");
            return false;
        }
        ps5vm::ranges::track(OBJECT_REGION, OBJECT_REGION_SIZE, ps5vm::MemProtect::RW);
        p.mapped = true;
    }
    true
}

pub(crate) fn free_object(addr: u64) {
    if (OBJECT_REGION..OBJECT_REGION + OBJECT_REGION_SIZE).contains(&addr) {
        OBJECTS.lock().unwrap().free.push(addr);
    }
}

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

struct ThreadInfo {
    handle: u64,
    id: u32,
    tp: u64,
    name: Mutex<String>,
    stack_top: u64,
    stack_size: u64,
    /// The thread's own attributes (priority, policy, affinity, stack...).
    attr: Mutex<ThreadAttr>,
    /// Detach/join state: who may release the thread, and whether it was.
    join: Mutex<JoinState>,
    /// Exit value once the thread has finished.
    exit: Mutex<Option<u64>>,
    exited: Condvar,
    is_main: bool,
    /// The stack came from the creator (attr stack address); not ours to unmap.
    user_stack: bool,
    /// Host CPU-time clock of the thread while it runs.
    cpu_clock: Mutex<Option<libc::clockid_t>>,
    /// Created suspended (pthread_attr_setcreatesuspend_np) and not resumed
    /// yet: the start routine waits for pthread_resume_np / resume_all.
    suspended: Mutex<bool>,
    resumed: Condvar,
}

static THREADS: Mutex<Option<HashMap<u64, Arc<ThreadInfo>>>> = Mutex::new(None);
static NEXT_THREAD_ID: AtomicU32 = AtomicU32::new(1);
static VM: OnceLock<usize> = OnceLock::new();

thread_local! {
    static CURRENT: Cell<u64> = const { Cell::new(0) };
    static CURRENT_ID: Cell<u32> = const { Cell::new(0) };
}

fn vm() -> &'static ps5vm::VirtualMemory {
    unsafe { &*(*VM.get().expect("pthread runtime not initialized") as *const ps5vm::VirtualMemory) }
}

/// The guest address space once the thread runtime is up.
pub(crate) fn guest_vm() -> Option<&'static ps5vm::VirtualMemory> {
    // SAFETY: VM holds a `&'static VirtualMemory` stored by init_main_thread.
    VM.get().map(|&p| unsafe { &*(p as *const ps5vm::VirtualMemory) })
}

/// Bumped after every access that may change the thread table.
static THREADS_GENERATION: crate::sync::Generation = crate::sync::Generation::new();

thread_local! {
    static THREAD_LOOKUPS: crate::sync::LocalLookups<ThreadInfo> = const { crate::sync::LocalLookups::new() };
}

fn threads<R>(f: impl FnOnce(&mut HashMap<u64, Arc<ThreadInfo>>) -> R) -> R {
    let mut g = THREADS.lock().unwrap();
    let r = f(g.get_or_insert_with(HashMap::new));
    THREADS_GENERATION.bump();
    r
}

fn thread(handle: u64) -> Option<Arc<ThreadInfo>> {
    let lookup = || THREADS.lock().unwrap().as_ref().and_then(|t| t.get(&handle).cloned());
    THREAD_LOOKUPS.try_with(|c| c.get(&THREADS_GENERATION, handle, lookup)).unwrap_or_else(|_| lookup())
}

/// Handle of the calling guest thread.
#[inline]
pub fn current_handle() -> u64 {
    CURRENT.with(|c| c.get())
}

/// Scheduling priority of the calling guest thread (700 by default).
pub(crate) fn current_priority() -> i32 {
    current_thread().map_or(DEFAULT_PRIORITY, |t| t.attr.lock().unwrap().priority)
}

/// The calling host thread's CPU-time clock.
fn host_cpu_clock() -> Option<libc::clockid_t> {
    let mut clock: libc::clockid_t = 0;
    // SAFETY: queries the calling thread into a local.
    (unsafe { libc::pthread_getcpuclockid(libc::pthread_self(), &mut clock) } == 0).then_some(clock)
}

/// CPU affinity of the calling guest thread.
pub(crate) fn current_affinity() -> u64 {
    current_thread().map_or(DEFAULT_AFFINITY, |t| t.attr.lock().unwrap().affinity)
}

/// The process cpuset as cpuset_getaffinity reports it: the CPUs of the
/// live guest threads (the process cpuset when there are none).
pub(crate) fn process_affinity() -> u64 {
    let live = threads(|t| t.values().filter(|i| i.exit.lock().unwrap().is_none()).fold(0, |m, i| m | i.attr.lock().unwrap().affinity));
    if live == 0 { PROCESS_AFFINITY.load(Ordering::Relaxed) } else { live }
}

fn thread_by_id(id: i64) -> Option<Arc<ThreadInfo>> {
    threads(|t| t.values().find(|i| i.id as i64 == id).cloned())
}

/// CPU affinity of the live guest thread with thread id `id`.
pub(crate) fn affinity_of_thread_id(id: i64) -> Option<u64> {
    thread_by_id(id).map(|t| t.attr.lock().unwrap().affinity)
}

/// FreeBSD CPU-time clock ids: `CPUCLOCK_BIT | tid` for a thread, with
/// `CPUCLOCK_PROCESS_BIT` as well for a process.
pub(crate) const CPUCLOCK_BIT: u32 = 0x8000_0000;
pub(crate) const CPUCLOCK_PROCESS_BIT: u32 = 0x4000_0000;
pub(crate) const CPUCLOCK_ID_MASK: u32 = !(CPUCLOCK_BIT | CPUCLOCK_PROCESS_BIT);

/// CPU time used so far by the running guest thread `tid`.
pub(crate) fn thread_cpu_time(tid: u32) -> Option<Duration> {
    let info = thread_by_id(tid as i64)?;
    let clock = (*info.cpu_clock.lock().unwrap())?;
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: plain clock read into a local.
    (unsafe { libc::clock_gettime(clock, &mut ts) } == 0).then(|| Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32))
}

/// Bind the calling host thread to a guest thread.
fn bind_current(handle: u64, id: u32) {
    CURRENT.with(|c| c.set(handle));
    CURRENT_ID.with(|c| c.set(id));
}

/// Wall-clock time since the Unix epoch.
pub(crate) fn realtime_now() -> Duration {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default()
}

/// Bind the calling test thread to a fake guest thread handle.
#[cfg(test)]
pub(crate) fn bind_test_thread(handle: u64, id: u32) {
    bind_current(handle, id);
}

/// True while `handle` names a live (not yet released) guest thread.
pub(crate) fn thread_exists(handle: u64) -> bool {
    thread(handle).is_some()
}

/// A thread that exists and has not finished (a finished thread stays in
/// the table until it is joined or detached).
pub(crate) fn thread_is_live(handle: u64) -> bool {
    thread(handle).is_some_and(|t| t.exit.lock().unwrap().is_none())
}

fn current_thread() -> Option<Arc<ThreadInfo>> {
    thread(current_handle())
}

fn stack_top_for(tp: u64) -> u64 {
    let slot = (0x6FFE_0000_0000 - tp) / 0x100_0000;
    STACK_REGION + (slot + 1) * STACK_SLOT
}

/// Create the main thread's block and stack and bind the calling host
/// thread to it. Returns `(tp, stack_top)`.
pub fn init_main_thread(vm: &'static ps5vm::VirtualMemory, stack_size: u64) -> Option<(u64, u64)> {
    let _ = VM.set(vm as *const _ as usize);
    let id = NEXT_THREAD_ID.fetch_add(1, Ordering::SeqCst);
    let tp = tls::allocate_thread_block(vm, id)?;
    let handle = tp + tls::HANDLE_OFFSET;
    let stack_top = stack_top_for(tp);
    ps5cpu::exec::map_guest_stack(stack_top, stack_size).ok()?;
    crate::memory::register_stack(stack_top - stack_size, stack_size);
    let info = Arc::new(ThreadInfo {
        handle,
        id,
        tp,
        name: Mutex::new("MainThread".into()),
        stack_top,
        stack_size,
        attr: Mutex::new(ThreadAttr { stack_addr: stack_top - stack_size, stack_size, ..ThreadAttr::default() }),
        join: Mutex::new(JoinState::default()),
        exit: Mutex::new(None),
        exited: Condvar::new(),
        is_main: true,
        user_stack: false,
        cpu_clock: Mutex::new(host_cpu_clock()),
        suspended: Mutex::new(false),
        resumed: Condvar::new(),
    });
    threads(|t| t.insert(handle, info));
    bind_current(handle, id);
    crate::signals::register_thread(handle);
    ps5cpu::exec::bind_thread_pointer(tp);
    log::info!("[pthread] main thread id={id} handle={handle:#x} tp={tp:#x} stack_top={stack_top:#x}");
    Some((tp, stack_top))
}

/// Run the C runtime's thread destructor and then the key destructors on
/// a thread that has left guest code, using the top of its guest stack.
fn run_thread_cleanup(stack_top: u64) {
    let call = |func: u64, arg: u64| {
        ps5cpu::exec::enter_guest(&ps5cpu::exec::call_regs(func, stack_top - 0x100, [arg, 0, 0, 0, 0, 0]));
    };
    // Per-thread `__cxa_thread_atexit` destructors (NID qBS714-Jr3g) first,
    // last registered first (as `crate::libc::run_thread_atexit_destructors`,
    // which needs a dispatch frame; the thread has left guest code here).
    while let Some((dtor, obj)) = crate::libc::pop_thread_atexit_destructor() {
        if dtor != 0 {
            call(dtor, obj);
        }
    }
    let dtors = crate::modules::thread_dtors_callback();
    if dtors != 0 {
        call(dtors, 0);
    }
    for _ in 0..DESTRUCTOR_ITERATIONS {
        let pending = take_key_destructors();
        if pending.is_empty() {
            break;
        }
        for (dtor, val) in pending {
            call(dtor, val);
        }
    }
    clear_specific();
}

fn finish_thread(info: &Arc<ThreadInfo>, value: u64) {
    *info.cpu_clock.lock().unwrap() = None;
    *info.exit.lock().unwrap() = Some(value);
    info.exited.notify_all();
    let mut j = info.join.lock().unwrap();
    if j.detached && !j.released {
        j.released = true;
        drop(j);
        release_thread(info);
    }
}

/// Join/detach bookkeeping (libthr thr_join.c / thr_detach.c): a thread
/// being joined can be neither detached nor joined a second time, and its
/// resources are released exactly once.
#[derive(Default, Clone, Copy)]
struct JoinState {
    detached: bool,
    /// A joiner is waiting for the thread.
    joining: bool,
    /// release_thread ran (or is about to run).
    released: bool,
}

/// After the thread has exited: the joiner marks it detached and releases it
/// unless someone already did.
fn finish_join(info: &Arc<ThreadInfo>) {
    let mut j = info.join.lock().unwrap();
    j.joining = false;
    j.detached = true;
    let release = !j.released;
    j.released = true;
    drop(j);
    if release {
        release_thread(info);
    }
}

/// Claim the join of `info`: EINVAL for a detached thread, EOPNOTSUPP when
/// another thread is already joining it.
fn begin_join(info: &ThreadInfo) -> Result<(), HleResult> {
    let mut j = info.join.lock().unwrap();
    if j.detached {
        return Err(EINVAL);
    }
    if j.joining {
        return Err(EOPNOTSUPP);
    }
    j.joining = true;
    Ok(())
}

fn release_thread(info: &Arc<ThreadInfo>) {
    if info.is_main {
        return;
    }
    crate::kernel_ext::forget_thread(info.handle);
    threads(|t| t.remove(&info.handle));
    if !info.user_stack {
        crate::memory::unregister_stack(info.stack_top - info.stack_size, info.stack_size);
        ps5cpu::exec::unmap_guest_stack(info.stack_top, info.stack_size);
    }
    if info.tp != 0 {
        tls::free_thread_block(info.tp);
    }
}

#[derive(Clone, Copy)]
struct ThreadAttr {
    detach: i32,
    stack_size: u64,
    stack_addr: u64,
    guard_size: u64,
    priority: i32,
    policy: i32,
    inherit: i32,
    affinity: u64,
    solo: i32,
    /// The affinity was set on the attributes; otherwise a new thread
    /// inherits its creator's (or the process cpuset).
    affinity_explicit: bool,
    /// Contention scope (pthread_attr_setscope); threads are system scope.
    scope: i32,
    /// A thread created with these attributes starts suspended.
    create_suspended: bool,
}

impl Default for ThreadAttr {
    fn default() -> Self {
        Self {
            detach: 0,
            stack_size: DEFAULT_STACK_SIZE,
            stack_addr: 0,
            guard_size: DEFAULT_GUARD_SIZE,
            priority: DEFAULT_PRIORITY,
            policy: DEFAULT_POLICY,
            inherit: DEFAULT_INHERIT,
            affinity: DEFAULT_AFFINITY,
            solo: 0,
            affinity_explicit: false,
            scope: posix_ext::SCOPE_SYSTEM,
            create_suspended: false,
        }
    }
}

static ATTRS: Mutex<Option<HashMap<u64, ThreadAttr>>> = Mutex::new(None);

fn with_attrs<R>(f: impl FnOnce(&mut HashMap<u64, ThreadAttr>) -> R) -> R {
    let mut g = ATTRS.lock().unwrap();
    f(g.get_or_insert_with(HashMap::new))
}

/// Resolve `attr*` (pointer to the attr object pointer).
fn attr_object(ctx: &CpuContext, attr_ptr: u64) -> Option<u64> {
    if attr_ptr == 0 {
        return None;
    }
    ctx.read_u64(attr_ptr).filter(|&o| o != 0)
}

fn read_attr(ctx: &CpuContext, attr_ptr: u64) -> Option<ThreadAttr> {
    let obj = attr_object(ctx, attr_ptr)?;
    with_attrs(|a| a.get(&obj).copied())
}

fn update_attr(ctx: &CpuContext, attr_ptr: u64, f: impl FnOnce(&mut ThreadAttr)) -> HleResult {
    let Some(obj) = attr_object(ctx, attr_ptr) else { return EINVAL };
    with_attrs(|a| match a.get_mut(&obj) {
        Some(at) => {
            f(at);
            OK
        }
        None => EINVAL,
    })
}

/// Stack of a new thread asking for `requested` bytes: `(mapped, size)`.
/// The thread gets exactly the size it asked for (bounded to what a stack
/// slot holds) at the top of the mapping, which also keeps
/// `STACK_HEADROOM` below it for host calls made on the guest stack
/// (nested guest calls, signal handlers).
fn stack_layout(requested: u64) -> (u64, u64) {
    let size = requested.clamp(STACK_MIN, MAX_STACK_SIZE);
    if size != requested.max(STACK_MIN) {
        log::warn!("[pthread] stack of {requested:#x} bytes limited to {size:#x}");
    }
    let mapped = ((size + STACK_HEADROOM + 0x3FFF) & !0x3FFF).max(0x20_0000);
    (mapped, size)
}

/// `(policy, priority)` a new thread starts with: its creator's when it
/// inherits its scheduling (the default, also with no attributes), the
/// attributes' with PTHREAD_EXPLICIT_SCHED or when no guest thread creates
/// it, as on a PS5 (firmware 13.60).
fn start_sched(attr: &ThreadAttr, creator: Option<&ThreadAttr>) -> (i32, i32) {
    match creator {
        Some(c) if attr.inherit != EXPLICIT_SCHED => (c.policy, c.priority),
        _ => (attr.policy, attr.priority),
    }
}

pub(crate) fn create_thread(ctx: &mut CpuContext, out: u64, attr_ptr: u64, entry: u64, arg: u64, name_ptr: u64) -> HleResult {
    if out == 0 || entry == 0 {
        return EINVAL;
    }
    let mut attr = if attr_ptr != 0 { read_attr(ctx, attr_ptr).unwrap_or_default() } else { ThreadAttr::default() };
    (attr.policy, attr.priority) = start_sched(&attr, current_thread().map(|c| *c.attr.lock().unwrap()).as_ref());
    let name = if name_ptr != 0 { ctx.read_cstring(name_ptr, NAME_MAX).unwrap_or_default() } else { String::new() };
    // The console keeps thread names in 32 bytes (measured on hardware):
    // a longer one is ENAMETOOLONG rather than cut.
    if name.len() >= 32 {
        return ENAMETOOLONG;
    }
    let id = NEXT_THREAD_ID.fetch_add(1, Ordering::SeqCst);
    let Some(tp) = tls::allocate_thread_block(vm(), id) else { return EAGAIN };
    let handle = tp + tls::HANDLE_OFFSET;
    let user_stack = attr.stack_addr != 0;
    // (top, mapped size, size the guest sees)
    let (stack_top, stack_size, guest_stack_size) = if user_stack {
        ((attr.stack_addr + attr.stack_size) & !0xF, attr.stack_size, attr.stack_size)
    } else {
        let (size, requested) = stack_layout(attr.stack_size);
        let top = stack_top_for(tp);
        if let Err(e) = ps5cpu::exec::map_guest_stack(top, size) {
            log::error!("[pthread] {e}");
            tls::free_thread_block(tp);
            return ENOMEM;
        }
        crate::memory::register_stack(top - size, size);
        (top, size, requested)
    };
    let info = Arc::new(ThreadInfo {
        handle,
        id,
        tp,
        name: Mutex::new(name.clone()),
        stack_top,
        stack_size,
        attr: Mutex::new(ThreadAttr {
            stack_addr: if user_stack { attr.stack_addr } else { stack_top - guest_stack_size },
            stack_size: guest_stack_size,
            affinity: inherited_affinity(&attr),
            ..attr
        }),
        join: Mutex::new(JoinState { detached: attr.detach != 0, ..JoinState::default() }),
        exit: Mutex::new(None),
        exited: Condvar::new(),
        is_main: false,
        user_stack,
        cpu_clock: Mutex::new(None),
        suspended: Mutex::new(attr.create_suspended),
        resumed: Condvar::new(),
    });
    threads(|t| t.insert(handle, info.clone()));
    if !ctx.write_u64(out, handle) {
        release_thread(&info);
        return EINVAL;
    }
    log::info!("[pthread] create '{name}' id={id} handle={handle:#x} entry={entry:#x} arg={arg:#x} stack={stack_size:#x}");
    let host_name = if name.is_empty() { format!("guest-{id}") } else { name };
    let start_handle = handle;
    let start_info = info.clone();
    let start = ps5cpu::exec::ThreadStart {
        name: host_name,
        entry,
        arg,
        stack_top,
        tp,
        on_start: Some(Box::new(move || {
            bind_current(start_handle, id);
            *start_info.cpu_clock.lock().unwrap() = host_cpu_clock();
            crate::signals::register_thread(start_handle);
            wait_until_resumed(&start_info);
        })),
    };
    let info2 = info.clone();
    let spawned = ps5cpu::exec::spawn_guest_thread(
        start,
        Box::new(move |exit| {
            let value = match exit {
                ps5cpu::exec::GuestExit::Returned(v) | ps5cpu::exec::GuestExit::ThreadExit(v) => v,
            };
            run_thread_cleanup(info2.stack_top);
            log::info!("[pthread] thread id={} exited value={value:#x}", info2.id);
            crate::signals::unregister_thread();
            finish_thread(&info2, value);
        }),
    );
    match spawned {
        Ok(_) => OK,
        Err(e) => {
            log::error!("[pthread] host thread spawn failed: {e}");
            release_thread(&info);
            EAGAIN
        }
    }
}

/// A thread created suspended waits here, before its start routine, until
/// it is resumed; shutdown parks it for good.
fn wait_until_resumed(info: &ThreadInfo) {
    let mut g = info.suspended.lock().unwrap_or_else(|e| e.into_inner());
    while *g {
        ps5vm::shutdown::park_if_requested();
        g = info.resumed.wait_timeout(g, Duration::from_millis(10)).unwrap_or_else(|e| e.into_inner()).0;
    }
}

fn resume(info: &ThreadInfo) {
    *info.suspended.lock().unwrap_or_else(|e| e.into_inner()) = false;
    info.resumed.notify_all();
}

/// pthread_attr_setcreatesuspend_np(attr*): threads created with `attr`
/// start suspended. A NULL or uninitialized attribute is EINVAL.
fn cb_attr_setcreatesuspend(ctx: &mut CpuContext) -> HleResult {
    update_attr(ctx, ctx.rdi(), |a| a.create_suspended = true)
}

/// pthread_resume_np(thread): let a thread created suspended run; a live
/// thread that is not suspended (the caller too) is 0, a NULL thread EINVAL,
/// one that has ended ESRCH (FreeBSD's _thr_find_thread).
fn cb_resume(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    if handle == 0 {
        return EINVAL;
    }
    let Some(info) = thread(handle).filter(|t| t.exit.lock().unwrap().is_none()) else { return ESRCH };
    resume(&info);
    OK
}

/// pthread_resume_all_np / pthread_multi_np: resume every other thread,
/// then a cancellation point.
fn cb_resume_all(_ctx: &mut CpuContext) -> HleResult {
    let me = current_handle();
    let all: Vec<Arc<ThreadInfo>> = threads(|t| t.values().filter(|i| i.handle != me).cloned().collect());
    for info in &all {
        resume(info);
    }
    drop(all);
    crate::kernel_ext::test_cancel();
    OK
}

/// scePthreadCreate(thread*, attr*, entry, arg, name)
fn cb_sce_create(ctx: &mut CpuContext) -> HleResult {
    let (out, attr, entry, arg, name) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8());
    create_thread(ctx, out, attr, entry, arg, name)
}

/// pthread_create(thread*, attr*, entry, arg)
fn cb_posix_create(ctx: &mut CpuContext) -> HleResult {
    let (out, attr, entry, arg) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    create_thread(ctx, out, attr, entry, arg, 0)
}

/// pthread_create_name_np(thread*, attr*, entry, arg, name)
fn cb_create_name_np(ctx: &mut CpuContext) -> HleResult {
    cb_sce_create(ctx)
}

fn cb_self(_ctx: &mut CpuContext) -> HleResult {
    current_handle() as i64
}

fn cb_equal(ctx: &mut CpuContext) -> HleResult {
    (ctx.rdi() == ctx.rsi()) as i64
}

fn cb_getthreadid(_ctx: &mut CpuContext) -> HleResult {
    CURRENT_ID.with(|c| c.get()) as i64
}

/// Wait for (or with `block` false, poll) the exit of thread `handle` and
/// release it. `Ok(None)`: the thread is still running.
pub(crate) fn join_handle(handle: u64, block: bool) -> Result<Option<u64>, HleResult> {
    if handle == current_handle() {
        return Err(EDEADLK);
    }
    let Some(info) = thread(handle) else { return Err(ESRCH) };
    begin_join(&info)?;
    let value = {
        let mut exit = info.exit.lock().unwrap();
        while exit.is_none() {
            if !block {
                drop(exit);
                info.join.lock().unwrap().joining = false;
                return Ok(None);
            }
            exit = info.exited.wait(exit).unwrap();
        }
        exit.unwrap()
    };
    finish_join(&info);
    Ok(Some(value))
}

/// scePthreadJoin(thread, value*)
fn cb_join(ctx: &mut CpuContext) -> HleResult {
    let (handle, value_out) = (ctx.rdi(), ctx.rsi());
    match join_thread(handle) {
        Ok(value) => {
            if value_out != 0 {
                ctx.write_u64(value_out, value);
            }
            OK
        }
        Err(e) => e,
    }
}

/// Wait for `handle` to end and release it: its exit value, or the
/// scePthreadJoin error. A cancellation point, before and while waiting.
pub(crate) fn join_thread(handle: u64) -> Result<u64, HleResult> {
    if handle == 0 {
        return Err(EINVAL);
    }
    if handle == current_handle() {
        return Err(EDEADLK);
    }
    let Some(info) = thread(handle) else { return Err(ESRCH) };
    {
        let j = info.join.lock().unwrap();
        if j.detached {
            return Err(EINVAL);
        }
    }
    crate::kernel_ext::test_cancel();
    begin_join(&info)?;
    let value = loop {
        let exit = info.exit.lock().unwrap();
        let (exit, _) = crate::signals::wait_while(&info.exit, &info.exited, exit, None, |e| {
            e.is_some() || crate::kernel_ext::cancel_requested()
        });
        if let Some(v) = *exit {
            break v;
        }
        drop(exit);
        // A cancelled joiner gives the join up (thr_join.c backout).
        info.join.lock().unwrap().joining = false;
        crate::kernel_ext::test_cancel();
        begin_join(&info)?;
    };
    finish_join(&info);
    Ok(value)
}

fn cb_detach(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return EINVAL;
    }
    let Some(info) = thread(ctx.rdi()) else { return ESRCH };
    let mut j = info.join.lock().unwrap();
    if j.detached || j.joining {
        return EINVAL;
    }
    j.detached = true;
    drop(j);
    if info.exit.lock().unwrap().is_some() {
        let mut j = info.join.lock().unwrap();
        let release = !j.released;
        j.released = true;
        drop(j);
        if release {
            release_thread(&info);
        }
    }
    OK
}

/// scePthreadExit(value) / pthread_exit(value): run the cleanup handlers
/// and end the thread.
fn cb_exit(ctx: &mut CpuContext) -> HleResult {
    let value = ctx.rdi();
    if current_thread().is_some_and(|t| t.is_main) {
        log::warn!("[pthread] main thread called pthread_exit({value:#x})");
    }
    crate::kernel_ext::exit_thread(value)
}

fn cb_yield(_ctx: &mut CpuContext) -> HleResult {
    std::thread::yield_now();
    OK
}

/// Store a thread's name (the WebKit library's pthread_setname_np);
/// false when the thread is unknown.
pub(crate) fn set_thread_name(handle: u64, name: &[u8]) -> bool {
    let Some(info) = thread(handle) else { return false };
    *info.name.lock().unwrap() = String::from_utf8_lossy(name).into_owned();
    true
}

/// scePthreadRename(thread, name) / pthread_rename_np
fn cb_rename(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return EINVAL;
    }
    let Some(info) = thread(ctx.rdi()) else { return ESRCH };
    if ctx.rsi() == 0 {
        return OK;
    }
    let name = ctx.read_cstring(ctx.rsi(), NAME_MAX).unwrap_or_default();
    *info.name.lock().unwrap() = name;
    OK
}

/// scePthreadGetname(thread, buf[32])
fn cb_getname(ctx: &mut CpuContext) -> HleResult {
    let (handle, buf) = (ctx.rdi(), ctx.rsi());
    let Some(info) = thread(handle) else { return ESRCH };
    if buf == 0 {
        return EFAULT;
    }
    let mut bytes = info.name.lock().unwrap().as_bytes().to_vec();
    bytes.truncate(NAME_MAX - 1);
    bytes.push(0);
    if ctx.write_bytes(buf, &bytes) { OK } else { EFAULT }
}

fn cb_getprio(ctx: &mut CpuContext) -> HleResult {
    let (handle, out) = (ctx.rdi(), ctx.rsi());
    if handle == 0 {
        return EINVAL;
    }
    let Some(info) = thread(handle) else { return ESRCH };
    if out == 0 || !ctx.write_u32(out, info.attr.lock().unwrap().priority as u32) {
        return EINVAL;
    }
    OK
}

fn cb_setprio(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return EINVAL;
    }
    let Some(info) = thread(ctx.rdi()) else { return ESRCH };
    info.attr.lock().unwrap().priority = ctx.rsi() as i32;
    OK
}

/// GetSchedparam(thread, policy*, param*)
fn cb_getschedparam(ctx: &mut CpuContext) -> HleResult {
    let (handle, policy, param) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if handle == 0 || policy == 0 || param == 0 {
        return EINVAL;
    }
    let Some(info) = thread(handle) else { return ESRCH };
    let a = *info.attr.lock().unwrap();
    if !ctx.write_u32(policy, a.policy as u32) || !ctx.write_u32(param, a.priority as u32) {
        return EINVAL;
    }
    OK
}

/// Whether a thread may run with `policy` at `priority`, as a PS5
/// (firmware 13.60) answers pthread_setschedparam: SCHED_FIFO or SCHED_RR
/// with a priority of 256 to 767 or 896 to 959.
fn accepts_sched_param(policy: i32, priority: i32) -> bool {
    matches!(policy, SCHED_FIFO | SCHED_RR) && ((256..=767).contains(&priority) || (896..=959).contains(&priority))
}

/// SetSchedparam(thread, policy, param*): EINVAL for a policy or priority
/// the console refuses; the thread keeps both.
fn cb_setschedparam(ctx: &mut CpuContext) -> HleResult {
    let (handle, policy, param) = (ctx.rdi(), ctx.rsi() as i32, ctx.rdx());
    if handle == 0 || param == 0 {
        return EINVAL;
    }
    let Some(p) = ctx.read_u32(param) else { return EINVAL };
    if !accepts_sched_param(policy, p as i32) {
        return EINVAL;
    }
    let Some(info) = thread(handle) else { return ESRCH };
    let mut a = info.attr.lock().unwrap();
    a.policy = policy;
    a.priority = p as i32;
    OK
}

fn cb_getaffinity(ctx: &mut CpuContext) -> HleResult {
    let (handle, out) = (ctx.rdi(), ctx.rsi());
    if handle == 0 {
        return EINVAL;
    }
    let Some(info) = thread(handle) else { return ESRCH };
    if out == 0 || !ctx.write_u64(out, info.attr.lock().unwrap().affinity) {
        return EINVAL;
    }
    OK
}

/// scePthreadSetaffinity(thread, mask): an empty mask is EINVAL.
fn cb_setaffinity(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 || ctx.rsi() == 0 {
        return EINVAL;
    }
    let Some(info) = thread(ctx.rdi()) else { return ESRCH };
    info.attr.lock().unwrap().affinity = ctx.rsi();
    OK
}

// ---------------------------------------------------------------------------
// Thread attributes
// ---------------------------------------------------------------------------

fn cb_attr_init(ctx: &mut CpuContext) -> HleResult {
    let ptr = ctx.rdi();
    if ptr == 0 {
        return EINVAL;
    }
    let Some(obj) = alloc_object(ctx) else { return ENOMEM };
    with_attrs(|a| a.insert(obj, ThreadAttr::default()));
    if ctx.write_u64(ptr, obj) { OK } else { EINVAL }
}

fn cb_attr_destroy(ctx: &mut CpuContext) -> HleResult {
    let ptr = ctx.rdi();
    let Some(obj) = attr_object(ctx, ptr) else { return EINVAL };
    if with_attrs(|a| a.remove(&obj)).is_none() {
        return EINVAL;
    }
    free_object(obj);
    ctx.write_u64(ptr, 0);
    OK
}

/// scePthreadAttrGet(thread, attr*) — copy a running thread's attributes.
fn cb_attr_get(ctx: &mut CpuContext) -> HleResult {
    let (handle, attr_ptr) = (ctx.rdi(), ctx.rsi());
    // A NULL thread, a NULL or an uninitialized attribute is EINVAL before
    // the thread is looked up (as FreeBSD's _pthread_attr_get_np).
    if handle == 0 || read_attr(ctx, attr_ptr).is_none() {
        return EINVAL;
    }
    // A thread that has ended is ESRCH even before it is joined (measured on
    // a PS5).
    let Some(info) = thread(handle).filter(|t| t.exit.lock().unwrap().is_none()) else { return ESRCH };
    let detached = info.join.lock().unwrap().detached;
    let src = *info.attr.lock().unwrap();
    update_attr(ctx, attr_ptr, |a| {
        *a = src;
        a.detach = detached as i32;
    })
}

/// Read one attribute field into `out` (u32 or u64 wide).
fn attr_get_field(ctx: &CpuContext, wide: bool, f: impl FnOnce(&ThreadAttr) -> u64) -> HleResult {
    let (ptr, out) = (ctx.rdi(), ctx.rsi());
    if out == 0 {
        return EINVAL;
    }
    let Some(a) = read_attr(ctx, ptr) else { return EINVAL };
    let v = f(&a);
    let ok = if wide { ctx.write_u64(out, v) } else { ctx.write_u32(out, v as u32) };
    if ok { OK } else { EINVAL }
}

fn cb_attr_setdetachstate(ctx: &mut CpuContext) -> HleResult {
    let v = ctx.rsi() as i32;
    if !(0..=1).contains(&v) {
        return EINVAL;
    }
    update_attr(ctx, ctx.rdi(), |a| a.detach = v)
}

fn cb_attr_setstacksize(ctx: &mut CpuContext) -> HleResult {
    let v = ctx.rsi();
    if v < STACK_MIN {
        return EINVAL;
    }
    update_attr(ctx, ctx.rdi(), |a| a.stack_size = v)
}

fn cb_attr_setguardsize(ctx: &mut CpuContext) -> HleResult {
    let v = ctx.rsi();
    update_attr(ctx, ctx.rdi(), |a| a.guard_size = v)
}

/// Attribute setters check the attribute before the value: a null or
/// uninitialised attribute is EINVAL, a value the console refuses
/// EOPNOTSUPP (PS5 firmware 13.60).
fn update_attr_checked(ctx: &CpuContext, attr_ptr: u64, valid: bool, f: impl FnOnce(&mut ThreadAttr)) -> HleResult {
    if read_attr(ctx, attr_ptr).is_none() {
        return EINVAL;
    }
    if !valid {
        return EOPNOTSUPP;
    }
    update_attr(ctx, attr_ptr, f)
}

/// AttrSetinheritsched(attr*, v): PTHREAD_EXPLICIT_SCHED (0) or
/// PTHREAD_INHERIT_SCHED (4).
fn cb_attr_setinheritsched(ctx: &mut CpuContext) -> HleResult {
    let v = ctx.rsi() as i32;
    update_attr_checked(ctx, ctx.rdi(), v == EXPLICIT_SCHED || v == DEFAULT_INHERIT, |a| a.inherit = v)
}

/// AttrSetschedpolicy(attr*, policy): SCHED_FIFO or SCHED_RR.
fn cb_attr_setschedpolicy(ctx: &mut CpuContext) -> HleResult {
    let v = ctx.rsi() as i32;
    update_attr_checked(ctx, ctx.rdi(), v == SCHED_FIFO || v == SCHED_RR, |a| a.policy = v)
}

fn cb_attr_setaffinity(ctx: &mut CpuContext) -> HleResult {
    let v = ctx.rsi();
    update_attr(ctx, ctx.rdi(), |a| {
        a.affinity = v;
        a.affinity_explicit = true;
    })
}

/// The affinity a new thread starts with: its attributes' when they set
/// one, else its creator's, else (no guest thread creates it) the process
/// cpuset, as a FreeBSD thread joins its creator's cpuset.
fn inherited_affinity(attr: &ThreadAttr) -> u64 {
    if attr.affinity_explicit {
        return attr.affinity;
    }
    current_thread().map_or_else(|| PROCESS_AFFINITY.load(Ordering::Relaxed), |t| t.attr.lock().unwrap().affinity)
}

/// AttrSetsolosched(attr*, v): off (0) or on (16).
fn cb_attr_setsolosched(ctx: &mut CpuContext) -> HleResult {
    let v = ctx.rsi() as i32;
    update_attr_checked(ctx, ctx.rdi(), v == SOLOSCHED_OFF || v == SOLOSCHED_ON, |a| a.solo = v)
}

fn cb_attr_getdetachstate(ctx: &mut CpuContext) -> HleResult {
    attr_get_field(ctx, false, |a| a.detach as u64)
}

fn cb_attr_getsolosched(ctx: &mut CpuContext) -> HleResult {
    attr_get_field(ctx, false, |a| a.solo as u64)
}

fn cb_attr_getinheritsched(ctx: &mut CpuContext) -> HleResult {
    attr_get_field(ctx, false, |a| a.inherit as u64)
}

fn cb_attr_getschedpolicy(ctx: &mut CpuContext) -> HleResult {
    attr_get_field(ctx, false, |a| a.policy as u64)
}

fn cb_attr_getstacksize(ctx: &mut CpuContext) -> HleResult {
    attr_get_field(ctx, true, |a| a.stack_size)
}

fn cb_attr_getguardsize(ctx: &mut CpuContext) -> HleResult {
    attr_get_field(ctx, true, |a| a.guard_size)
}

fn cb_attr_getstackaddr(ctx: &mut CpuContext) -> HleResult {
    attr_get_field(ctx, true, |a| a.stack_addr)
}

fn cb_attr_getaffinity(ctx: &mut CpuContext) -> HleResult {
    attr_get_field(ctx, true, |a| a.affinity)
}

/// AttrSetschedparam(attr*, param*) — param holds the priority, 256 to
/// 767; a null param is EOPNOTSUPP like a priority out of range.
fn cb_attr_setschedparam(ctx: &mut CpuContext) -> HleResult {
    let (ptr, param) = (ctx.rdi(), ctx.rsi());
    let p = if param == 0 { None } else { ctx.read_u32(param).map(|v| v as i32) };
    let valid = p.is_some_and(|p| ATTR_PRIORITY.contains(&p));
    update_attr_checked(ctx, ptr, valid, |a| a.priority = p.unwrap_or(DEFAULT_PRIORITY))
}

/// pthread_attr_setschedparam(attr*, param*) -> error number, as FreeBSD's
/// libthr: ENOTSUP for a null param, and under SCHED_FIFO or SCHED_RR for
/// a priority outside sched_get_priority_min..max.
fn px_attr_setschedparam(ctx: &mut CpuContext) -> HleResult {
    let (ptr, param) = (ctx.rdi(), ctx.rsi());
    let Some(attr) = read_attr(ctx, ptr) else { return posix_error(EINVAL) };
    let Some(p) = (if param == 0 { None } else { ctx.read_u32(param).map(|v| v as i32) }) else {
        return posix_error(EOPNOTSUPP);
    };
    if matches!(attr.policy, SCHED_FIFO | SCHED_RR) && !ATTR_PRIORITY.contains(&p) {
        return posix_error(EOPNOTSUPP);
    }
    posix_error(update_attr(ctx, ptr, |a| a.priority = p))
}

/// pthread_attr_setschedpolicy(attr*, policy) -> error number: SCHED_FIFO,
/// SCHED_OTHER or SCHED_RR, as FreeBSD's libthr; others are ENOTSUP.
fn px_attr_setschedpolicy(ctx: &mut CpuContext) -> HleResult {
    let v = ctx.rsi() as i32;
    posix_error(update_attr_checked(ctx, ctx.rdi(), (SCHED_FIFO..=SCHED_RR).contains(&v), |a| a.policy = v))
}

fn cb_attr_getschedparam(ctx: &mut CpuContext) -> HleResult {
    attr_get_field(ctx, false, |a| a.priority as u32 as u64)
}

/// AttrSetstack(attr*, addr, size)
fn cb_attr_setstack(ctx: &mut CpuContext) -> HleResult {
    let (ptr, addr, size) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if addr == 0 || size < STACK_MIN {
        return EINVAL;
    }
    update_attr(ctx, ptr, |a| {
        a.stack_addr = addr;
        a.stack_size = size;
    })
}

/// AttrSetstackaddr(attr*, addr)
fn cb_attr_setstackaddr(ctx: &mut CpuContext) -> HleResult {
    let (ptr, addr) = (ctx.rdi(), ctx.rsi());
    if addr == 0 {
        return EINVAL;
    }
    update_attr(ctx, ptr, |a| a.stack_addr = addr)
}

/// AttrGetstack(attr*, addr*, size*)
fn cb_attr_getstack(ctx: &mut CpuContext) -> HleResult {
    let (ptr, addr_out, size_out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if addr_out == 0 || size_out == 0 {
        return EINVAL;
    }
    let Some(a) = read_attr(ctx, ptr) else { return EINVAL };
    if !ctx.write_u64(addr_out, a.stack_addr) || !ctx.write_u64(size_out, a.stack_size) {
        return EINVAL;
    }
    OK
}

// ---------------------------------------------------------------------------
// POSIX-named wrappers
// ---------------------------------------------------------------------------

/// pthread_getprio(pthread_t) -> the thread's priority; -1 with errno ESRCH
/// for a NULL thread or one that has ended.
fn px_getprio(ctx: &mut CpuContext) -> HleResult {
    const ESRCH_ERRNO: i32 = 3;
    match thread(ctx.rdi()).filter(|t| t.exit.lock().unwrap().is_none()) {
        Some(info) if ctx.rdi() != 0 => info.attr.lock().unwrap().priority as HleResult,
        _ => {
            crate::kernel::set_errno(ctx, ESRCH_ERRNO);
            -1
        }
    }
}

/// Kernel error code -> POSIX error number (pthread_* return values).
pub(crate) fn posix_error(r: HleResult) -> HleResult {
    crate::kernel::kernel_to_posix(r) as HleResult
}

macro_rules! posix_wrappers {
    ($($posix:ident => $sce:path),* $(,)?) => {
        $(fn $posix(ctx: &mut CpuContext) -> HleResult {
            posix_error($sce(ctx))
        })*
    };
}

posix_wrappers! {
    px_create => cb_posix_create,
    px_create_name_np => cb_create_name_np,
    px_join => cb_join,
    px_detach => cb_detach,
    px_getname => cb_getname,
    px_setprio => cb_setprio,
    px_getschedparam => cb_getschedparam,
    px_setschedparam => cb_setschedparam,
    px_attr_init => cb_attr_init,
    px_attr_destroy => cb_attr_destroy,
    px_attr_get_np => cb_attr_get,
    px_attr_setcreatesuspend => cb_attr_setcreatesuspend,
    px_resume => cb_resume,
    px_attr_setdetachstate => cb_attr_setdetachstate,
    px_attr_getdetachstate => cb_attr_getdetachstate,
    px_attr_setstacksize => cb_attr_setstacksize,
    px_attr_getstacksize => cb_attr_getstacksize,
    px_attr_setguardsize => cb_attr_setguardsize,
    px_attr_getguardsize => cb_attr_getguardsize,
    px_attr_setinheritsched => cb_attr_setinheritsched,
    px_attr_getinheritsched => cb_attr_getinheritsched,
    px_attr_getschedpolicy => cb_attr_getschedpolicy,
    px_attr_getschedparam => cb_attr_getschedparam,
    px_attr_setstack => cb_attr_setstack,
    px_attr_getstack => cb_attr_getstack,
    px_attr_setstackaddr => cb_attr_setstackaddr,
    px_attr_getstackaddr => cb_attr_getstackaddr,
    px_attr_setaffinity => cb_attr_setaffinity,
    px_attr_getaffinity => cb_attr_getaffinity,
    px_attr_setsolosched => cb_attr_setsolosched,
    px_key_create => cb_key_create,
    px_key_delete => cb_key_delete,
    px_setspecific => cb_setspecific,
    px_once => cb_once,
    px_mutexattr_init => cb_mutexattr_init,
    px_mutexattr_destroy => cb_mutexattr_destroy,
    px_mutexattr_settype => cb_mutexattr_settype,
    px_mutexattr_gettype => cb_mutexattr_gettype,
    px_mutexattr_setprotocol => cb_mutexattr_setprotocol,
    px_mutexattr_getprotocol => cb_mutexattr_getprotocol,
    px_mutexattr_setprioceiling => cb_mutexattr_setprioceiling,
    px_mutexattr_getprioceiling => cb_mutexattr_getprioceiling,
    px_mutex_init => cb_mutex_init,
    px_mutex_destroy => cb_mutex_destroy,
    px_mutex_lock => cb_mutex_lock,
    px_mutex_trylock => cb_mutex_trylock,
    px_mutex_timedlock => cb_mutex_timedlock_abs,
    px_mutex_reltimedlock => cb_mutex_timedlock_usec,
    px_mutex_getspinloops => cb_mutex_getspinloops,
    px_mutex_setspinloops => cb_mutex_setspinloops,
    px_mutex_getyieldloops => cb_mutex_getyieldloops,
    px_mutex_setyieldloops => cb_mutex_setyieldloops,
    px_mutex_unlock => cb_mutex_unlock,
    px_condattr_init => cb_condattr_init,
    px_condattr_destroy => cb_attr_object_destroy,
    px_condattr_setclock => cb_condattr_setclock,
    px_condattr_getclock => cb_condattr_getclock,
    px_condattr_setpshared => cb_condattr_setpshared,
    px_condattr_getpshared => cb_condattr_getpshared,
    px_cond_init => cb_cond_init,
    px_cond_destroy => cb_cond_destroy,
    px_cond_wait => cb_cond_wait,
    px_cond_timedwait => cb_posix_cond_timedwait,
    px_cond_signal => cb_cond_signal,
    px_cond_broadcast => cb_cond_broadcast,
    px_cond_setname_np => cb_cond_setname_np,
    px_rwlockattr_init => cb_rwlockattr_init,
    px_rwlockattr_destroy => cb_attr_object_destroy,
    px_rwlockattr_settype => cb_rwlockattr_settype,
    px_rwlockattr_gettype => cb_rwlockattr_gettype,
    px_rwlock_init => cb_rwlock_init,
    px_rwlock_destroy => cb_rwlock_destroy,
    px_rwlock_rdlock => cb_rwlock_rdlock,
    px_rwlock_wrlock => cb_rwlock_wrlock,
    px_rwlock_tryrdlock => cb_rwlock_tryrdlock,
    px_rwlock_trywrlock => cb_rwlock_trywrlock,
    px_rwlock_unlock => cb_rwlock_unlock,
}

/// POSIX ENAMETOOLONG.
const POSIX_ENAMETOOLONG: HleResult = 63;

/// pthread_rename_np(thread, name): a name of NAME_MAX bytes or more is
/// rejected and the old name kept.
fn px_rename(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() != 0 && ctx.rsi() != 0 && ctx.read_cstr_bytes(ctx.rsi(), NAME_MAX).is_some_and(|b| b.len() >= NAME_MAX) {
        return POSIX_ENAMETOOLONG;
    }
    posix_error(cb_rename(ctx))
}

/// Size of a FreeBSD `cpuset_t` (256 CPUs).
const CPUSET_BYTES: u64 = 32;
const POSIX_EINVAL: HleResult = 22;
const POSIX_ESRCH: HleResult = 3;
const POSIX_EFAULT: HleResult = 14;
const POSIX_ERANGE: HleResult = 34;

/// pthread_getaffinity_np(thread, cpusetsize, cpuset_t*) -> 0 or an error
/// number, FreeBSD 11's: EINVAL for a NULL thread, ERANGE for any size but
/// a full cpuset_t (32 bytes), EFAULT for a NULL set, ESRCH for an unknown
/// thread. The mask fills the first 8 bytes, the rest of the set is zero.
fn px_getaffinity(ctx: &mut CpuContext) -> HleResult {
    let (handle, size, set) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if handle == 0 {
        return POSIX_EINVAL;
    }
    if size != CPUSET_BYTES {
        return POSIX_ERANGE;
    }
    if set == 0 {
        return POSIX_EFAULT;
    }
    let Some(info) = thread(handle) else { return POSIX_ESRCH };
    let mut bytes = [0u8; CPUSET_BYTES as usize];
    bytes[..8].copy_from_slice(&info.attr.lock().unwrap().affinity.to_le_bytes());
    if ctx.write_bytes(set, &bytes) { OK } else { POSIX_EFAULT }
}

const POSIX_EDEADLK: HleResult = 11;

/// The target of a cpuset change.
pub(crate) enum CpusetTarget {
    /// A guest thread by handle (0: the calling thread).
    Thread(u64),
    /// A guest thread by thread id.
    ThreadId(i64),
    /// Every live guest thread.
    Process,
}

/// FreeBSD cpuset_setaffinity on a cpuset_t at `set` of `size` bytes ->
/// 0 or an error number: ERANGE for a size outside 8..=32, EFAULT for a
/// NULL set, EINVAL for a CPU beyond the 64 the mask holds, EDEADLK when
/// none of the requested CPUs is available to the game (the bits outside
/// the available mask are dropped), ESRCH for an unknown thread.
pub(crate) fn set_cpuset_affinity(ctx: &CpuContext, target: CpusetTarget, size: u64, set: u64) -> HleResult {
    if !(8..=CPUSET_BYTES).contains(&size) {
        return POSIX_ERANGE;
    }
    if set == 0 {
        return POSIX_EFAULT;
    }
    let Some(bytes) = ctx.read_bytes(set, size as usize) else { return POSIX_EFAULT };
    if bytes[8..].iter().any(|&b| b != 0) {
        return POSIX_EINVAL;
    }
    let affinity = u64::from_le_bytes(bytes[..8].try_into().unwrap()) & crate::kernel_ext::AVAILABLE_CPUMASK;
    if affinity == 0 {
        return POSIX_EDEADLK;
    }
    let info = match target {
        CpusetTarget::Thread(0) => current_thread(),
        CpusetTarget::Thread(h) => thread(h),
        CpusetTarget::ThreadId(id) => thread_by_id(id),
        CpusetTarget::Process => {
            threads(|t| t.values().for_each(|i| i.attr.lock().unwrap().affinity = affinity));
            PROCESS_AFFINITY.store(affinity, Ordering::Relaxed);
            return OK;
        }
    };
    let Some(info) = info else { return POSIX_ESRCH };
    info.attr.lock().unwrap().affinity = affinity;
    OK
}

/// pthread_setaffinity_np(thread, cpusetsize, const cpuset_t*) -> 0 or an
/// error number: cpuset_setaffinity on the thread, as FreeBSD's libthr.
fn px_setaffinity(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    if handle == 0 {
        return POSIX_ESRCH;
    }
    set_cpuset_affinity(ctx, CpusetTarget::Thread(handle), ctx.rsi(), ctx.rdx())
}

/// pthread_set_name_np(thread, name) (void): renames the thread; a NULL
/// name clears it, as in FreeBSD; a NULL or unknown thread is ignored. A
/// name of 32 bytes or more keeps its first 31 bytes (what getname reports).
fn px_set_name_np(ctx: &mut CpuContext) -> HleResult {
    let Some(info) = (ctx.rdi() != 0).then(|| thread(ctx.rdi())).flatten() else { return 0 };
    let mut name = if ctx.rsi() == 0 { Vec::new() } else { ctx.read_cstr_bytes(ctx.rsi(), 1 << 16).unwrap_or_default() };
    name.truncate(NAME_MAX - 1);
    *info.name.lock().unwrap() = String::from_utf8_lossy(&name).into_owned();
    0
}

/// pthread_getcpuclockid(thread, clockid_t*)
fn px_getcpuclockid(ctx: &mut CpuContext) -> HleResult {
    let (handle, out) = (ctx.rdi(), ctx.rsi());
    if handle == 0 {
        return posix_error(EINVAL);
    }
    if out == 0 {
        return posix_error(EFAULT);
    }
    let Some(info) = thread(handle) else { return posix_error(ESRCH) };
    if !ctx.write_u32(out, CPUCLOCK_BIT | info.id) {
        return posix_error(EFAULT);
    }
    OK
}

/// pthread_rwlock_timedrdlock(rwlock*, abstime*)
fn px_rwlock_timedrdlock(ctx: &mut CpuContext) -> HleResult {
    rwlock_timed_abs(ctx, false)
}

/// pthread_rwlock_timedwrlock(rwlock*, abstime*)
fn px_rwlock_timedwrlock(ctx: &mut CpuContext) -> HleResult {
    rwlock_timed_abs(ctx, true)
}

/// The timed rwlock locks, as FreeBSD's libthr: a lock that is free is
/// taken before the deadline is looked at, and a null deadline waits
/// without limit.
fn rwlock_timed_abs(ctx: &mut CpuContext, write: bool) -> HleResult {
    let l = match resolve_rwlock(ctx, ctx.rdi()) {
        Ok(l) => l,
        Err(e) => return posix_error(e),
    };
    let ts = ctx.rsi();
    if ts == 0 {
        return posix_error(rw_lock(l, write, None));
    }
    if (if write { rw_trywrlock(l) } else { rw_tryrdlock(l) }) == OK {
        return OK;
    }
    let (Some(sec), Some(nsec)) = (ctx.read_u64(ts), ctx.read_u64(ts + 8)) else {
        return posix_error(EINVAL);
    };
    if (sec as i64) < 0 || nsec >= 1_000_000_000 {
        return posix_error(EINVAL);
    }
    let target = Duration::new(sec, nsec as u32);
    let us = target.saturating_sub(realtime_now()).as_micros().min(u32::MAX as u128) as u64;
    posix_error(rw_lock(l, write, Some(us)))
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// `scePthread*` exports: kernel error codes.
const SCE_EXPORTS: &[(&str, &str, crate::registry::HleCallback)] = &[
    ("6UgtwV+0zb4", "scePthreadCreate", cb_sce_create),
    ("aI+OeCz8xrQ", "scePthreadSelf", cb_self),
    ("3PtV6p3QNX4", "scePthreadEqual", cb_equal),
    ("EI-5-jlq2dE", "scePthreadGetthreadid", cb_getthreadid),
    ("55aShElDfY4", "scePthreadMain", cb_pthread_main_np),
    ("onNY9Byn-W8", "scePthreadJoin", cb_join),
    ("4qGrR6eoP9Y", "scePthreadDetach", cb_detach),
    ("3kg7rT0NQIs", "scePthreadExit", cb_exit),
    ("T72hz6ffq08", "scePthreadYield", cb_yield),
    ("GBUY7ywdULE", "scePthreadRename", cb_rename),
    ("How7B8Oet6k", "scePthreadGetname", cb_getname),
    ("1tKyG7RlMJo", "scePthreadGetprio", cb_getprio),
    ("W0Hpm2X0uPE", "scePthreadSetprio", cb_setprio),
    ("P41kTWUS3EI", "scePthreadGetschedparam", cb_getschedparam),
    ("oIRFTjoILbg", "scePthreadSetschedparam", cb_setschedparam),
    ("rcrVFJsQWRY", "scePthreadGetaffinity", cb_getaffinity),
    ("bt3CTBKmGyI", "scePthreadSetaffinity", cb_setaffinity),
    ("nsYoNRywwNg", "scePthreadAttrInit", cb_attr_init),
    ("62KCwEMmzcM", "scePthreadAttrDestroy", cb_attr_destroy),
    ("x1X76arYMxU", "scePthreadAttrGet", cb_attr_get),
    ("GZSR0Ooae9Q", "scePthreadAttrSetcreatesuspend", cb_attr_setcreatesuspend),
    ("DB7Mkm+Pqzw", "scePthreadResume", cb_resume),
    ("te+MBYMzDhY", "scePthreadResumeAll", cb_resume_all),
    ("JVc71p0lpFs", "scePthreadMulti", cb_resume_all),
    ("-Wreprtu0Qs", "scePthreadAttrSetdetachstate", cb_attr_setdetachstate),
    ("JaRMy+QcpeU", "scePthreadAttrGetdetachstate", cb_attr_getdetachstate),
    ("UTXzJbWhhTE", "scePthreadAttrSetstacksize", cb_attr_setstacksize),
    ("-fA+7ZlGDQs", "scePthreadAttrGetstacksize", cb_attr_getstacksize),
    ("El+cQ20DynU", "scePthreadAttrSetguardsize", cb_attr_setguardsize),
    ("txHtngJ+eyc", "scePthreadAttrGetguardsize", cb_attr_getguardsize),
    ("eXbUSpEaTsA", "scePthreadAttrSetinheritsched", cb_attr_setinheritsched),
    ("lpMP8HhkBbg", "scePthreadAttrGetinheritsched", cb_attr_getinheritsched),
    ("4+h9EzwKF4I", "scePthreadAttrSetschedpolicy", cb_attr_setschedpolicy),
    ("NMyIQ9WgWbU", "scePthreadAttrGetschedpolicy", cb_attr_getschedpolicy),
    ("DzES9hQF4f4", "scePthreadAttrSetschedparam", cb_attr_setschedparam),
    ("FXPWHNk8Of0", "scePthreadAttrGetschedparam", cb_attr_getschedparam),
    ("3qxgM4ezETA", "scePthreadAttrSetaffinity", cb_attr_setaffinity),
    ("8+s5BzZjxSg", "scePthreadAttrGetaffinity", cb_attr_getaffinity),
    ("Dk6FC-TI+7Q", "scePthreadAttrSetsolosched", cb_attr_setsolosched),
    ("9RnL-m0+diQ", "scePthreadAttrGetsolosched", cb_attr_getsolosched),
    ("Bvn74vj6oLo", "scePthreadAttrSetstack", cb_attr_setstack),
    ("-quPa4SEJUw", "scePthreadAttrGetstack", cb_attr_getstack),
    ("F+yfmduIBB8", "scePthreadAttrSetstackaddr", cb_attr_setstackaddr),
    ("Ru36fiTtJzA", "scePthreadAttrGetstackaddr", cb_attr_getstackaddr),
    ("geDaqgH9lTg", "scePthreadKeyCreate", cb_key_create),
    ("PrdHuuDekhY", "scePthreadKeyDelete", cb_key_delete),
    ("+BzXYkqYeLE", "scePthreadSetspecific", cb_setspecific),
    ("eoht7mQOCmo", "scePthreadGetspecific", cb_getspecific),
    ("14bOACANTBo", "scePthreadOnce", cb_once),
    ("F8bUHwAG284", "scePthreadMutexattrInit", cb_mutexattr_init),
    ("smWEktiyyG0", "scePthreadMutexattrDestroy", cb_mutexattr_destroy),
    ("iMp8QpE+XO4", "scePthreadMutexattrSettype", cb_mutexattr_settype),
    ("gquEhBrS2iw", "scePthreadMutexattrGettype", cb_mutexattr_gettype),
    ("1FGvU0i9saQ", "scePthreadMutexattrSetprotocol", cb_mutexattr_setprotocol),
    ("GoTmFeui+hQ", "scePthreadMutexattrGetprotocol", cb_mutexattr_getprotocol),
    ("532IaQguwMg", "scePthreadMutexattrSetprioceiling", cb_mutexattr_setprioceiling),
    ("SgjMpyH9Z9I", "scePthreadMutexattrGetprioceiling", cb_mutexattr_getprioceiling),
    ("cmo1RIYva9o", "scePthreadMutexInit", cb_mutex_init),
    ("2Of0f+3mhhE", "scePthreadMutexDestroy", cb_mutex_destroy),
    ("9UK1vLZQft4", "scePthreadMutexLock", cb_mutex_lock),
    ("upoVrzMHFeE", "scePthreadMutexTrylock", cb_mutex_trylock),
    ("IafI2PxcPnQ", "scePthreadMutexTimedlock", cb_mutex_timedlock_usec),
    ("pOmNmyRKlIE", "scePthreadMutexGetspinloops", cb_mutex_getspinloops),
    ("42YkUouoMI0", "scePthreadMutexSetspinloops", cb_mutex_setspinloops),
    ("AWS3NyViL9o", "scePthreadMutexGetyieldloops", cb_mutex_getyieldloops),
    ("bP+cqFmBW+A", "scePthreadMutexSetyieldloops", cb_mutex_setyieldloops),
    ("tn3VlD0hG60", "scePthreadMutexUnlock", cb_mutex_unlock),
    ("2Tb92quprl0", "scePthreadCondInit", cb_cond_init),
    ("g+PZd2hiacg", "scePthreadCondDestroy", cb_cond_destroy),
    ("WKAXJ4XBPQ4", "scePthreadCondWait", cb_cond_wait),
    ("BmMjYxmew1w", "scePthreadCondTimedwait", cb_sce_cond_timedwait),
    ("kDh-NfxgMtE", "scePthreadCondSignal", cb_cond_signal),
    ("o69RpYO-Mu0", "scePthreadCondSignalto", cb_cond_signalto),
    ("JGgj7Uvrl+A", "scePthreadCondBroadcast", cb_cond_broadcast),
    ("m5-2bsNfv7s", "scePthreadCondattrInit", cb_condattr_init),
    ("waPcxYiR3WA", "scePthreadCondattrDestroy", cb_attr_object_destroy),
    ("c-bxj027czs", "scePthreadCondattrSetclock", cb_condattr_setclock),
    ("6qM3kO5S3Oo", "scePthreadCondattrGetclock", cb_condattr_getclock),
    ("6ULAa0fq4jA", "scePthreadRwlockInit", cb_rwlock_init),
    ("BB+kb08Tl9A", "scePthreadRwlockDestroy", cb_rwlock_destroy),
    ("Ox9i0c7L5w0", "scePthreadRwlockRdlock", cb_rwlock_rdlock),
    ("mqdNorrB+gI", "scePthreadRwlockWrlock", cb_rwlock_wrlock),
    ("XD3mDeybCnk", "scePthreadRwlockTryrdlock", cb_rwlock_tryrdlock),
    ("bIHoZCTomsI", "scePthreadRwlockTrywrlock", cb_rwlock_trywrlock),
    ("iPtZRWICjrM", "scePthreadRwlockTimedrdlock", cb_rwlock_timedrdlock),
    ("adh--6nIqTk", "scePthreadRwlockTimedwrlock", cb_rwlock_timedwrlock),
    ("+L98PIbGttk", "scePthreadRwlockUnlock", cb_rwlock_unlock),
    ("yOfGg-I1ZII", "scePthreadRwlockattrInit", cb_rwlockattr_init),
    ("i2ifZ3fS2fo", "scePthreadRwlockattrDestroy", cb_attr_object_destroy),
    ("h-OifiouBd8", "scePthreadRwlockattrSettype", cb_rwlockattr_settype),
    ("Kyls1ChFyrc", "scePthreadRwlockattrGettype", cb_rwlockattr_gettype),
];

/// POSIX-named exports: POSIX error numbers (or plain values).
const POSIX_EXPORTS: &[(&str, &str, crate::registry::HleCallback)] = &[
    ("OxhIB8LB-PQ", "pthread_create", px_create),
    ("Jmi+9w9u0E4", "pthread_create_name_np", px_create_name_np),
    ("EotR8a3ASf4", "pthread_self", cb_self),
    ("7Xl257M4VNI", "pthread_equal", cb_equal),
    ("3eqs37G74-s", "pthread_getthreadid_np", cb_getthreadid),
    ("h9CcP3J0oVM", "pthread_join", px_join),
    ("+U1R4WtXvoc", "pthread_detach", px_detach),
    ("FJrT5LuUBAU", "pthread_exit", cb_exit),
    ("B5GmVDKwpn0", "pthread_yield", cb_yield),
    ("6XG4B33N09g", "sched_yield", cb_yield),
    ("9vyP6Z7bqzc", "pthread_rename_np", px_rename),
    ("oxMp8uPqa+U", "pthread_set_name_np", px_set_name_np),
    ("9HzfhdtESio", "pthread_getname_np", px_getname),
    ("nnoxZUHK+SA", "pthread_getcpuclockid", px_getcpuclockid),
    ("aO+WS2xWuo4", "pthread_getprio", px_getprio),
    ("a2P9wYGeZvc", "pthread_setprio", px_setprio),
    ("FIs3-UQT9sg", "pthread_getschedparam", px_getschedparam),
    ("Xs9hdiD7sAA", "pthread_setschedparam", px_setschedparam),
    ("Jb2uGFMr688", "pthread_getaffinity_np", px_getaffinity),
    ("5KWrg7-ZqvE", "pthread_setaffinity_np", px_setaffinity),
    ("wtkt-teR1so", "pthread_attr_init", px_attr_init),
    ("zHchY8ft5pk", "pthread_attr_destroy", px_attr_destroy),
    ("Ucsu-OK+els", "pthread_attr_get_np", px_attr_get_np),
    ("Q2y5IqSDZGs", "pthread_attr_setcreatesuspend_np", px_attr_setcreatesuspend),
    ("BYM3L--ojzI", "pthread_resume_np", px_resume),
    ("3gY5B0FCkNY", "pthread_resume_all_np", cb_resume_all),
    ("vhcrecqZy-k", "pthread_multi_np", cb_resume_all),
    ("E+tyo3lp5Lw", "pthread_attr_setdetachstate", px_attr_setdetachstate),
    ("VUT1ZSrHT0I", "pthread_attr_getdetachstate", px_attr_getdetachstate),
    ("2Q0z6rnBrTE", "pthread_attr_setstacksize", px_attr_setstacksize),
    ("0qOtCR-ZHck", "pthread_attr_getstacksize", px_attr_getstacksize),
    ("JKyG3SWyA10", "pthread_attr_setguardsize", px_attr_setguardsize),
    ("JNkVVsVDmOk", "pthread_attr_getguardsize", px_attr_getguardsize),
    ("7ZlAakEf0Qg", "pthread_attr_setinheritsched", px_attr_setinheritsched),
    ("oLjPqUKhzes", "pthread_attr_getinheritsched", px_attr_getinheritsched),
    ("JarMIy8kKEY", "pthread_attr_setschedpolicy", px_attr_setschedpolicy),
    ("RtLRV-pBTTY", "pthread_attr_getschedpolicy", px_attr_getschedpolicy),
    ("euKRgm0Vn2M", "pthread_attr_setschedparam", px_attr_setschedparam),
    ("qlk9pSLsUmM", "pthread_attr_getschedparam", px_attr_getschedparam),
    ("-SrbXpGR1f0", "pthread_attr_setstack", px_attr_setstack),
    ("vQm4fDEsWi8", "pthread_attr_getstack", px_attr_getstack),
    ("suCrEbr0xIQ", "pthread_attr_setstackaddr", px_attr_setstackaddr),
    ("DxmIMUQ-wXY", "pthread_attr_getstackaddr", px_attr_getstackaddr),
    ("o8pd4juNbgc", "pthread_attr_setaffinity_np", px_attr_setaffinity),
    ("-wzZ7dvA7UU", "pthread_attr_getaffinity_np", px_attr_getaffinity),
    ("2+pVfgiEd7A", "pthread_attr_setsolosched_np", px_attr_setsolosched),
    ("mqULNdimTn0", "pthread_key_create", px_key_create),
    ("6BpEZuDT7YI", "pthread_key_delete", px_key_delete),
    ("WrOLvHU0yQM", "pthread_setspecific", px_setspecific),
    ("0-KXaS70xy4", "pthread_getspecific", cb_getspecific),
    ("Z4QosVuAsA0", "pthread_once", px_once),
    ("dQHWEsJtoE4", "pthread_mutexattr_init", px_mutexattr_init),
    ("HF7lK46xzjY", "pthread_mutexattr_destroy", px_mutexattr_destroy),
    ("mDmgMOGVUqg", "pthread_mutexattr_settype", px_mutexattr_settype),
    ("GZFlI7RhuQo", "pthread_mutexattr_gettype", px_mutexattr_gettype),
    ("5txKfcMUAok", "pthread_mutexattr_setprotocol", px_mutexattr_setprotocol),
    ("yDaWxUE50s0", "pthread_mutexattr_getprotocol", px_mutexattr_getprotocol),
    ("ZLvf6lVAc4M", "pthread_mutexattr_setprioceiling", px_mutexattr_setprioceiling),
    ("+m8+quqOwhM", "pthread_mutexattr_getprioceiling", px_mutexattr_getprioceiling),
    ("ttHNfU+qDBU", "pthread_mutex_init", px_mutex_init),
    ("ltCfaGr2JGE", "pthread_mutex_destroy", px_mutex_destroy),
    ("7H0iTOciTLo", "pthread_mutex_lock", px_mutex_lock),
    ("K-jXhbt2gn4", "pthread_mutex_trylock", px_mutex_trylock),
    ("Io9+nTKXZtA", "pthread_mutex_timedlock", px_mutex_timedlock),
    ("nbQ0bXMRlhk", "pthread_mutex_reltimedlock_np", px_mutex_reltimedlock),
    ("x4vQj3JKKmc", "pthread_mutex_getspinloops_np", px_mutex_getspinloops),
    ("5-ncLMtL5+g", "pthread_mutex_setspinloops_np", px_mutex_setspinloops),
    ("OxEIUqkByy4", "pthread_mutex_getyieldloops_np", px_mutex_getyieldloops),
    ("frFuGprJmPc", "pthread_mutex_setyieldloops_np", px_mutex_setyieldloops),
    ("gKqzW-zWhvY", "pthread_mutex_isowned_np", cb_mutex_isowned),
    ("2Z+PpY6CaJg", "pthread_mutex_unlock", px_mutex_unlock),
    ("mKoTx03HRWA", "pthread_condattr_init", px_condattr_init),
    ("dJcuQVn6-Iw", "pthread_condattr_destroy", px_condattr_destroy),
    ("EjllaAqAPZo", "pthread_condattr_setclock", px_condattr_setclock),
    ("cTDYxTUNPhM", "pthread_condattr_getclock", px_condattr_getclock),
    ("3BpP850hBT4", "pthread_condattr_setpshared", px_condattr_setpshared),
    ("h0qUqSuOmC8", "pthread_condattr_getpshared", px_condattr_getpshared),
    ("0TyVk4MSLt0", "pthread_cond_init", px_cond_init),
    ("RXXqi4CtF8w", "pthread_cond_destroy", px_cond_destroy),
    ("Op8TBGY5KHg", "pthread_cond_wait", px_cond_wait),
    ("27bAgiJmOh0", "pthread_cond_timedwait", px_cond_timedwait),
    ("2MOy+rUfuhQ", "pthread_cond_signal", px_cond_signal),
    ("mkx2fVhNMsg", "pthread_cond_broadcast", px_cond_broadcast),
    ("EZ8h70dtFLg", "pthread_cond_setname_np", px_cond_setname_np),
    ("xFebsA4YsFI", "pthread_rwlockattr_init", px_rwlockattr_init),
    ("qsdmgXjqSgk", "pthread_rwlockattr_destroy", px_rwlockattr_destroy),
    ("8NuOHiTr1Vw", "pthread_rwlockattr_settype_np", px_rwlockattr_settype),
    ("l+bG5fsYkhg", "pthread_rwlockattr_gettype_np", px_rwlockattr_gettype),
    ("ytQULN-nhL4", "pthread_rwlock_init", px_rwlock_init),
    ("1471ajPzxh0", "pthread_rwlock_destroy", px_rwlock_destroy),
    ("iGjsr1WAtI0", "pthread_rwlock_rdlock", px_rwlock_rdlock),
    ("sIlRvQqsN2Y", "pthread_rwlock_wrlock", px_rwlock_wrlock),
    ("SFxTMOfuCkE", "pthread_rwlock_tryrdlock", px_rwlock_tryrdlock),
    ("XhWHn6P5R7U", "pthread_rwlock_trywrlock", px_rwlock_trywrlock),
    ("lb8lnYo-o7k", "pthread_rwlock_timedrdlock", px_rwlock_timedrdlock),
    ("9zklzAl9CGM", "pthread_rwlock_timedwrlock", px_rwlock_timedwrlock),
    ("EgmLo6EWgso", "pthread_rwlock_unlock", px_rwlock_unlock),
];

/// pthread_main_np() and scePthreadMain(): whether the calling thread is
/// the initial one (behaviour measured on hardware: 1 or 0).
fn cb_pthread_main_np(_ctx: &mut CpuContext) -> HleResult {
    current_thread().is_some_and(|t| t.is_main) as i64
}

/// rtprio_thread(function, lwpid, rtprio*): the lookup reports the
/// scheduling class of the calling thread or a live one by its id —
/// type 3 (normal) with priority 0, or 10 / 2 with the priority for a
/// FIFO / round-robin class (behaviour measured on hardware). A change
/// is only the no-op normal-to-normal one; the rest is refused. Returns
/// -1 with errno.
fn cb_rtprio_thread(ctx: &mut CpuContext) -> HleResult {
    const LOOKUP: i32 = 0;
    const SET: i32 = 1;
    const RTPRIO_NORMAL: u16 = 3;
    const RTPRIO_REALTIME: u16 = 2;
    const RTPRIO_FIFO: u16 = 10;
    const ESRCH_ERRNO: i32 = 3;
    const EFAULT_ERRNO: i32 = 14;
    const EPERM_ERRNO: i32 = 1;
    let (function, lwpid, out) = (ctx.rdi() as i32, ctx.rsi() as i64, ctx.rdx());
    let fail = |ctx: &mut CpuContext, errno: i32| -> HleResult {
        crate::kernel::set_errno(ctx, errno);
        -1
    };
    if function != LOOKUP && function != SET {
        return fail(ctx, ESRCH_ERRNO);
    }
    if out == 0 {
        return fail(ctx, EFAULT_ERRNO);
    }
    let self_id = CURRENT_ID.with(|c| c.get()) as i64;
    let other = lwpid != 0 && lwpid != self_id;
    let info = if other {
        let Some(info) = thread_by_id(lwpid) else { return fail(ctx, ESRCH_ERRNO) };
        // A finished thread is gone.
        if info.exit.lock().unwrap().is_some() {
            return fail(ctx, ESRCH_ERRNO);
        }
        info
    } else {
        let Some(info) = current_thread() else { return fail(ctx, ESRCH_ERRNO) };
        info
    };
    let (policy, priority) = {
        let a = info.attr.lock().unwrap();
        (a.policy, a.priority)
    };
    let real_time = matches!(policy, SCHED_FIFO | SCHED_RR);
    if function == LOOKUP {
        let (kind, prio) = if real_time {
            (if policy == SCHED_FIFO { RTPRIO_FIFO } else { RTPRIO_REALTIME }, priority as u16)
        } else {
            (RTPRIO_NORMAL, 0)
        };
        let mut value = [0u8; 4];
        value[..2].copy_from_slice(&kind.to_le_bytes());
        value[2..].copy_from_slice(&prio.to_le_bytes());
        if !ctx.write_bytes(out, &value) {
            return fail(ctx, EFAULT_ERRNO);
        }
        return OK;
    }
    // A change to another thread or to a real-time class is refused.
    if other || real_time {
        log::warn!("rtprio_thread: refusing a real-time priority change");
        return fail(ctx, EPERM_ERRNO);
    }
    let Some(b) = ctx.read_bytes(out, 4) else { return fail(ctx, EFAULT_ERRNO) };
    let kind = u16::from_le_bytes([b[0], b[1]]);
    let prio = u16::from_le_bytes([b[2], b[3]]);
    if kind == RTPRIO_NORMAL && prio == 0 {
        return OK;
    }
    log::warn!("rtprio_thread: refusing a real-time priority change");
    fail(ctx, EPERM_ERRNO)
}

/// sceSpPthreadMutexattrSetprioceiling: not on the console wrapper either,
/// it reports success and does nothing.
fn cb_sp_mutexattr_setprioceiling(_ctx: &mut CpuContext) -> HleResult {
    log::warn!("sceSpPthreadMutexattrSetprioceiling: doing nothing");
    0
}

/// The sceSp* wrappers of libSceSpSysCallWrapper: each forwards to the
/// pthread function of the same shape.
const SP_SYSCALL_EXPORTS: &[(&str, &str, fn(&mut CpuContext) -> HleResult)] = &[
    ("-7ovXt8hp6g", "sceSpPthreadCreate", cb_sce_create),
    ("WwzNbMVtOJA", "sceSpPthreadAttrInit", cb_attr_init),
    ("dVY7nLRddBM", "sceSpPthreadAttrDestroy", cb_attr_destroy),
    ("xS43U4VgIXE", "sceSpPthreadAttrSetinheritsched", cb_attr_setinheritsched),
    ("NwWU0vvhPTc", "sceSpPthreadAttrSetschedparam", cb_attr_setschedparam),
    ("-p2Gvg7QqFo", "sceSpPthreadAttrSetstacksize", cb_attr_setstacksize),
    ("INOiD5QWHNg", "sceSpPthreadCondInit", cb_cond_init),
    ("Jaa16wSEA9U", "sceSpPthreadCondDestroy", cb_cond_destroy),
    ("aODvnvMj8vs", "sceSpPthreadCondSignal", cb_cond_signal),
    ("4hl78F9jMCM", "sceSpPthreadCondWait", cb_cond_wait),
    ("537ZOR325ZI", "sceSpPthreadCondattrInit", cb_condattr_init),
    ("Fql-di-dWu0", "sceSpPthreadCondattrDestroy", cb_attr_object_destroy),
    ("Mlo+OHFpOeY", "sceSpPthreadDetach", cb_detach),
    ("B59oMLpCo6s", "sceSpPthreadExit", cb_exit),
    ("rP40jSNZ-6Y", "sceSpPthreadJoin", cb_join),
    ("S3qFxKMTgKQ", "sceSpPthreadMutexInit", cb_mutex_init),
    ("gMlHVhsO9lk", "sceSpPthreadMutexDestroy", cb_mutex_destroy),
    ("cukQjzNcSFw", "sceSpPthreadMutexLock", cb_mutex_lock),
    ("OecFSm0BxdI", "sceSpPthreadMutexUnlock", cb_mutex_unlock),
    ("a6yeQI2uXGk", "sceSpPthreadMutexattrInit", cb_mutexattr_init),
    ("59JzpduJC-k", "sceSpPthreadMutexattrDestroy", cb_mutexattr_destroy),
    ("4jZLf+MFU8U", "sceSpPthreadMutexattrSetprotocol", cb_mutexattr_setprotocol),
    ("0aJMoN6KBpg", "sceSpPthreadMutexattrSettype", cb_mutexattr_settype),
    ("zsVbNbWPYzg", "sceSpPthreadSetprio", cb_setprio),
    ("l29UO6Oj0ds", "sceSpPthreadMutexattrSetprioceiling", cb_sp_mutexattr_setprioceiling),
    ("2I2RV6LyNng", "rtprio_thread", cb_rtprio_thread),
    ("D1D6GFIYYGs", "pthread_main_np", cb_pthread_main_np),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in SCE_EXPORTS.iter().chain(POSIX_EXPORTS).chain(posix_ext::POSIX_EXT_EXPORTS) {
        reg.register_fn(nid, name, "libKernel", cb);
    }
    for &(nid, name, cb) in SP_SYSCALL_EXPORTS {
        reg.register_fn(nid, name, "libSceSpSysCallWrapper", cb);
    }
}

#[cfg(test)]
pub(crate) mod tests;
#[cfg(test)]
pub(crate) use tests::{fake_thread, forget_fake_thread, AFFINITY_TESTS};

/// scePthreadGetaffinity, for tests of other modules.
#[cfg(test)]
pub(crate) fn test_scegetaffinity(ctx: &mut CpuContext) -> HleResult {
    cb_getaffinity(ctx)
}

/// pthread_setaffinity_np(thread, size, set), for tests of other modules.
#[cfg(test)]
pub(crate) fn test_setaffinity_np(ctx: &mut CpuContext, thread: u64, size: u64, set: u64) -> HleResult {
    ctx.set(crate::context::CpuRegister::Rdi, thread);
    ctx.set(crate::context::CpuRegister::Rsi, size);
    ctx.set(crate::context::CpuRegister::Rdx, set);
    px_setaffinity(ctx)
}
