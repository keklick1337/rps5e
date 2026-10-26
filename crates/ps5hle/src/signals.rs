// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest exceptions/signals: `sceKernelInstallExceptionHandler`,
//! `sceKernelRemoveExceptionHandler` and `sceKernelRaiseException`.
//!
//! Handlers are process-wide, one per guest signal number. A raise marks
//! the signal pending on the target guest thread and delivers it there:
//!
//!   * raising on the calling thread runs the handler synchronously, with
//!     the context of the `sceKernelRaiseException` call;
//!   * raising on another thread queues the signal, interrupts the target
//!     host thread with `ps5cpu::signal::interrupt_signal()` and wakes it
//!     if it is blocked in an HLE wait, then waits briefly for the target
//!     to take it.
//!
//! The target takes the signal at the first of:
//!   * the interrupt landing in guest code — the handler runs on top of the
//!     interrupted guest frame and may change its context;
//!   * an import returning to guest code (`ImportReturn` safe point);
//!   * a blocking HLE wait: every wait goes through [`wait_cv`] / [`sleep`],
//!     which wait in slices of at most [`SLICE`] (or are woken by the
//!     raiser) and deliver with no HLE locks held. The handler then sees
//!     the guest state at the import call; changes to it are ignored.
//!
//! The same waits stop guest threads once the process is shutting down
//! (`ps5vm::shutdown`): the thread parks with its HLE locks released.
//!
//! The context handed to handlers is the Orbis `ucontext_t`
//! (`ps5cpu::signal::OrbisUcontext`), built in guest stack memory below the
//! interrupted frame. A signal is not taken while a handler runs on the
//! same thread (the signal stays pending until it returns).
//!
//! The POSIX signal interface (`sigaction`, masks, `raise`, `kill`,
//! `sigsetjmp`/`siglongjmp`) is in [`posix`].

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};
use ps5cpu::signal::{GuestInterrupt, OrbisUcontext};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

mod posix;

const OK: HleResult = 0;
const ESRCH: HleResult = 0x8002_0003u32 as i32 as i64;
const EINVAL: HleResult = 0x8002_0016u32 as i32 as i64;
const EAGAIN: HleResult = 0x8002_0023u32 as i32 as i64;

/// Guest (FreeBSD) signal numbers accepted for exception handlers.
pub const SIGHUP: i32 = 1;
pub const SIGILL: i32 = 4;
pub const SIGFPE: i32 = 8;
pub const SIGBUS: i32 = 10;
pub const SIGSEGV: i32 = 11;
pub const SIGUSR1: i32 = 30;

/// Longest a blocking HLE wait sleeps before checking for pending signals.
pub const SLICE: Duration = Duration::from_millis(10);
/// How long a raiser waits for the target to take a queued signal.
const DISPATCH_WAIT_MAX: Duration = Duration::from_millis(2);

const MAX_SIGNAL: usize = 128;

fn allowed_signal(signum: i32) -> bool {
    matches!(signum, SIGHUP | SIGILL | SIGFPE | SIGBUS | SIGSEGV | SIGUSR1)
}

static HANDLERS: [AtomicU64; MAX_SIGNAL] = [const { AtomicU64::new(0) }; MAX_SIGNAL];

fn handler_for(signum: i32) -> u64 {
    if (0..MAX_SIGNAL as i32).contains(&signum) {
        HANDLERS[signum as usize].load(Ordering::Acquire)
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Per-thread state
// ---------------------------------------------------------------------------

struct ThreadSignals {
    host: libc::pthread_t,
    /// Bit n = guest signal n pending.
    pending: u128,
    /// Condvar the thread is currently blocked on in `wait_cv` (address),
    /// when waits publish it under the table lock (`RPS5E_FAST_WAIT=0`).
    waiting_on: usize,
    /// The same, published without the table lock.
    slot: Arc<WaitSlot>,
}

/// Wait state a thread publishes without taking the table lock. The
/// thread keeps one reference, its table entry the other.
struct WaitSlot {
    /// Address of the condvar the thread blocks on in `wait_cv` (0 = none),
    /// with [`SLOT_BUSY`] set while a waker notifies it. The thread cannot
    /// clear a busy slot, so the condvar outlives the notify.
    waiting_on: AtomicUsize,
    /// `ThreadSignals::pending != 0`, written under the table lock.
    has_pending: AtomicBool,
}

const SLOT_BUSY: usize = 1;

impl WaitSlot {
    fn publish(&self, cv: usize) {
        self.waiting_on.store(cv, Ordering::Release);
    }

    /// Withdraw `cv`, waiting out a waker that is notifying it.
    fn withdraw(&self, cv: usize) {
        loop {
            match self.waiting_on.compare_exchange(cv, 0, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return,
                Err(v) if v == cv | SLOT_BUSY => std::hint::spin_loop(),
                Err(_) => {
                    self.waiting_on.store(0, Ordering::Release);
                    return;
                }
            }
        }
    }

    /// Notify the condvar the thread blocks on, if any. Wakers call this
    /// with the table lock held, so at most one marks the slot busy.
    fn notify(&self) {
        let cv = self.waiting_on.load(Ordering::Acquire);
        if cv == 0 || cv & SLOT_BUSY != 0 {
            return;
        }
        if self.waiting_on.compare_exchange(cv, cv | SLOT_BUSY, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            // Valid: the thread cannot withdraw it while it is busy.
            unsafe { (*(cv as *const Condvar)).notify_all() };
            self.waiting_on.store(cv, Ordering::Release);
        }
    }
}

/// `RPS5E_FAST_WAIT=0`: blocking waits publish their condvar and read
/// their pending signals under the global table lock (twice per wait).
static FAST_WAIT: AtomicU8 = AtomicU8::new(0);

fn fast_wait() -> bool {
    match FAST_WAIT.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on = std::env::var("RPS5E_FAST_WAIT").map_or(true, |v| !matches!(v.trim(), "0" | "off" | "false" | "no"));
            FAST_WAIT.store(if on { 1 } else { 2 }, Ordering::Relaxed);
            on
        }
    }
}

#[cfg(test)]
pub(crate) fn tests_pthread_kill(ctx: &mut CpuContext) -> crate::registry::HleResult {
    posix::tests_pthread_kill(ctx)
}

#[cfg(test)]
fn set_fast_wait(on: bool) {
    FAST_WAIT.store(if on { 1 } else { 2 }, Ordering::Relaxed);
}

impl ThreadSignals {
    fn set_pending(&mut self, pending: u128) {
        self.pending = pending;
        self.slot.has_pending.store(pending != 0, Ordering::Release);
    }

    /// Wake the thread if it is blocked in an HLE wait. Called with the
    /// table lock held.
    fn notify_wait(&self) {
        if self.waiting_on != 0 {
            // Valid: the waiter clears it (under this lock) before its
            // condvar can go away.
            unsafe { (*(self.waiting_on as *const Condvar)).notify_all() };
        }
        self.slot.notify();
    }
}

static THREADS: Mutex<Option<HashMap<u64, ThreadSignals>>> = Mutex::new(None);
/// Number of threads with a non-empty pending set (fast path check).
static PENDING_THREADS: AtomicUsize = AtomicUsize::new(0);
/// Notified whenever a thread takes a pending signal.
static TAKEN: Condvar = Condvar::new();

thread_local! {
    static HANDLE: Cell<u64> = const { Cell::new(0) };
    static DELIVERING: Cell<bool> = const { Cell::new(false) };
    static SLOT: RefCell<Option<Arc<WaitSlot>>> = const { RefCell::new(None) };
}

fn threads() -> MutexGuard<'static, Option<HashMap<u64, ThreadSignals>>> {
    THREADS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Register the calling host thread as guest thread `handle` (call on the
/// thread itself, before it runs guest code).
pub fn register_thread(handle: u64) {
    HANDLE.with(|h| h.set(handle));
    let host = unsafe { libc::pthread_self() };
    let slot = Arc::new(WaitSlot { waiting_on: AtomicUsize::new(0), has_pending: AtomicBool::new(false) });
    SLOT.with(|s| *s.borrow_mut() = Some(slot.clone()));
    threads().get_or_insert_with(HashMap::new).insert(handle, ThreadSignals { host, pending: 0, waiting_on: 0, slot });
    ps5cpu::signal::set_signal_hook(interrupt_hook);
}

/// Unregister the calling thread (call on the thread itself before its
/// host thread ends: a raise must never signal an exited host thread).
pub fn unregister_thread() {
    let handle = HANDLE.with(|h| h.replace(0));
    if handle == 0 {
        return;
    }
    SLOT.with(|s| *s.borrow_mut() = None);
    let mut g = threads();
    if let Some(t) = g.get_or_insert_with(HashMap::new).remove(&handle) {
        if t.pending != 0 {
            PENDING_THREADS.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

fn current() -> u64 {
    HANDLE.with(|h| h.get())
}

/// Whether the calling thread has a deliverable pending signal.
pub fn pending_for_current() -> bool {
    if PENDING_THREADS.load(Ordering::Acquire) == 0 || DELIVERING.with(|d| d.get()) {
        return false;
    }
    let h = current();
    if h == 0 {
        return false;
    }
    if fast_wait() {
        return SLOT.with(|s| s.borrow().as_ref().is_some_and(|s| s.has_pending.load(Ordering::Acquire)));
    }
    threads().as_ref().and_then(|t| t.get(&h)).is_some_and(|t| t.pending != 0)
}

fn take_pending(handle: u64) -> Option<i32> {
    let mut g = threads();
    let t = g.as_mut()?.get_mut(&handle)?;
    if t.pending == 0 {
        return None;
    }
    let signum = t.pending.trailing_zeros() as i32;
    t.set_pending(t.pending & !(1u128 << signum));
    if t.pending == 0 {
        PENDING_THREADS.fetch_sub(1, Ordering::AcqRel);
    }
    drop(g);
    TAKEN.notify_all();
    Some(signum)
}

// ---------------------------------------------------------------------------
// Delivery
// ---------------------------------------------------------------------------

/// Run the guest handler for `signum` with context `uc`, on the guest stack
/// below the current interrupted frame. Returns the context as left by the
/// handler (None when no guest call is possible here).
fn run_handler(signum: i32, handler: u64, uc: &OrbisUcontext) -> Option<OrbisUcontext> {
    let size = std::mem::size_of::<OrbisUcontext>() as u64;
    let mut uc = *uc;
    uc.uc_sigmask = posix::blocked_mask();
    let was = DELIVERING.with(|d| d.replace(true));
    let r = ps5cpu::signal::with_guest_scratch(size, |addr| {
        unsafe { std::ptr::copy_nonoverlapping(uc.as_bytes().as_ptr(), addr as *mut u8, size as usize) };
        ps5cpu::exec::call_guest_nested(handler, [signum as u64, addr, 0, 0, 0, 0]);
        let bytes = unsafe { std::slice::from_raw_parts(addr as *const u8, size as usize) };
        OrbisUcontext::from_bytes(bytes)
    });
    DELIVERING.with(|d| d.set(was));
    r.flatten()
}

/// Run a POSIX signal handler on the calling thread from inside the
/// current import: `handler(sig, info, uc)` with `info` the given
/// `siginfo_t` (SA_SIGINFO) or 0, and `uc` the context of the import call
/// with `mask` as its signal mask. Changes the handler makes to the context
/// are not applied. Returns false outside an import dispatch.
fn run_handler_args(signum: i32, handler: u64, mask: [u32; 4], info: Option<&[u8]>) -> bool {
    let Some(view) = ps5cpu::signal::current_import_interrupt() else { return false };
    let mut uc = view.ucontext();
    uc.uc_sigmask = mask;
    let uc_size = std::mem::size_of::<OrbisUcontext>() as u64;
    let uc_room = uc_size.div_ceil(0x40) * 0x40;
    let info_size = info.map_or(0, |i| i.len() as u64);
    let was = DELIVERING.with(|d| d.replace(true));
    let r = ps5cpu::signal::with_guest_scratch(uc_room + info_size, |addr| {
        let info_addr = addr + uc_room;
        unsafe { std::ptr::copy_nonoverlapping(uc.as_bytes().as_ptr(), addr as *mut u8, uc_size as usize) };
        let arg1 = match info {
            Some(i) => {
                unsafe { std::ptr::copy_nonoverlapping(i.as_ptr(), info_addr as *mut u8, i.len()) };
                info_addr
            }
            None => 0,
        };
        ps5cpu::exec::call_guest_nested(handler, [signum as u64, arg1, addr, 0, 0, 0]);
    });
    DELIVERING.with(|d| d.set(was));
    r.is_some()
}

/// Deliver every pending signal of the calling thread at `point`.
fn deliver_pending(point: &mut GuestInterrupt) -> bool {
    let handle = current();
    if handle == 0 || DELIVERING.with(|d| d.get()) {
        return false;
    }
    let mut delivered = false;
    while let Some(signum) = take_pending(handle) {
        let handler = handler_for(signum);
        if handler == 0 {
            continue;
        }
        log::trace!("[signal] deliver {signum} to {handle:#x} at {:?}", point.kind());
        let uc = point.ucontext();
        if let Some(out) = run_handler(signum, handler, &uc) {
            point.apply(&out);
        }
        delivered = true;
    }
    delivered
}

/// Hook called by the CPU layer when guest code was interrupted or an
/// import is returning.
fn interrupt_hook(point: &mut GuestInterrupt) {
    if PENDING_THREADS.load(Ordering::Acquire) == 0 {
        return;
    }
    deliver_pending(point);
}

/// Deliver pending signals of the calling thread from inside an HLE call.
/// Must be called with no HLE locks held. Returns true if a handler ran.
pub fn dispatch_pending() -> bool {
    if !pending_for_current() {
        return false;
    }
    let Some(mut view) = ps5cpu::signal::current_import_interrupt() else { return false };
    deliver_pending(&mut view)
}

// ---------------------------------------------------------------------------
// Interruptible waits
// ---------------------------------------------------------------------------

/// Publish (`cv` != 0) or withdraw (`cv` == 0, `was` = the published
/// address) the condvar the calling thread blocks on; `fast` picks the
/// lock-free slot over the table entry.
fn set_waiting_on(cv: usize, was: usize, fast: bool) {
    if fast {
        SLOT.with(|s| {
            if let Some(slot) = s.borrow().as_ref() {
                if cv != 0 {
                    slot.publish(cv);
                } else {
                    slot.withdraw(was);
                }
            }
        });
        return;
    }
    let h = current();
    if h == 0 {
        return;
    }
    if let Some(t) = threads().as_mut().and_then(|t| t.get_mut(&h)) {
        t.waiting_on = cv;
    }
}

/// Wait once on `cv` (whose mutex is `m`, held as `guard`) until notified,
/// `deadline` (None = forever) or one [`SLICE`], whichever comes first;
/// `slice` may shorten the slice. Pending guest signals are delivered with
/// the lock released. Returns the re-acquired guard and whether the
/// deadline has passed. Spurious returns are normal: callers re-check
/// their condition in a loop.
pub fn wait_cv_slice<'a, T>(
    m: &'a Mutex<T>,
    cv: &Condvar,
    guard: MutexGuard<'a, T>,
    deadline: Option<Instant>,
    slice: Option<Duration>,
) -> (MutexGuard<'a, T>, bool) {
    let mut guard = guard;
    if pending_for_current() {
        drop(guard);
        dispatch_pending();
        guard = m.lock().unwrap_or_else(|e| e.into_inner());
    }
    let registered = current() != 0;
    if registered && ps5vm::shutdown::requested() {
        // The process is ending: the guest thread stops here, holding no
        // HLE lock.
        drop(guard);
        ps5vm::shutdown::park_if_requested();
        guard = m.lock().unwrap_or_else(|e| e.into_inner());
    }
    let now = Instant::now();
    let max_slice = slice_limit();
    let mut limit = slice.map_or(max_slice, |s| s.min(max_slice));
    if let Some(d) = deadline {
        if now >= d {
            return (guard, true);
        }
        limit = limit.min(d - now);
    }
    let cv_addr = cv as *const Condvar as usize;
    let fast = fast_wait();
    if registered {
        set_waiting_on(cv_addr, 0, fast);
    }
    let (g, _) = cv.wait_timeout(guard, limit).unwrap_or_else(|e| e.into_inner());
    if registered {
        set_waiting_on(0, cv_addr, fast);
    }
    let mut g = g;
    if pending_for_current() {
        drop(g);
        dispatch_pending();
        g = m.lock().unwrap_or_else(|e| e.into_inner());
    }
    if registered && ps5vm::shutdown::requested() {
        drop(g);
        ps5vm::shutdown::park_if_requested();
        g = m.lock().unwrap_or_else(|e| e.into_inner());
    }
    let timed_out = deadline.is_some_and(|d| Instant::now() >= d);
    (g, timed_out)
}

#[cfg(test)]
thread_local! {
    /// Tests stretch the slice of the calling thread's waits so that a
    /// missed wakeup shows as a hang instead of a 10 ms delay.
    pub(crate) static TEST_SLICE: Cell<Option<Duration>> = const { Cell::new(None) };
}

#[inline]
fn slice_limit() -> Duration {
    #[cfg(test)]
    if let Some(s) = TEST_SLICE.with(|s| s.get()) {
        return s;
    }
    SLICE
}

/// [`wait_cv_slice`] with the default slice.
pub fn wait_cv<'a, T>(
    m: &'a Mutex<T>,
    cv: &Condvar,
    guard: MutexGuard<'a, T>,
    deadline: Option<Instant>,
) -> (MutexGuard<'a, T>, bool) {
    wait_cv_slice(m, cv, guard, deadline, None)
}

/// Wait on `cv` until `done(&state)` holds or `deadline` passes, delivering
/// pending guest signals meanwhile. Returns the guard and whether the
/// condition holds.
pub fn wait_while<'a, T>(
    m: &'a Mutex<T>,
    cv: &Condvar,
    guard: MutexGuard<'a, T>,
    deadline: Option<Instant>,
    mut done: impl FnMut(&T) -> bool,
) -> (MutexGuard<'a, T>, bool) {
    let mut g = guard;
    loop {
        if done(&g) {
            return (g, true);
        }
        let (ng, timed_out) = wait_cv(m, cv, g, deadline);
        g = ng;
        if timed_out {
            let ok = done(&g);
            return (g, ok);
        }
    }
}

thread_local! {
    static SLEEPER: (Mutex<()>, Condvar) = const { (Mutex::new(()), Condvar::new()) };
}

/// Sleep until `deadline`, delivering pending guest signals meanwhile.
pub fn sleep_until(deadline: Instant) {
    sleep_until_unless(deadline, || false);
}

/// Sleep until `deadline` or until `stop()` holds (checked whenever the
/// thread wakes: every [`SLICE`], and on [`wake_thread`]), delivering
/// pending guest signals meanwhile.
pub fn sleep_until_unless(deadline: Instant, stop: impl Fn() -> bool) {
    if current() == 0 {
        let now = Instant::now();
        if deadline > now && !stop() {
            std::thread::sleep(deadline - now);
        }
        return;
    }
    SLEEPER.with(|(m, cv)| {
        // The sleeper pair lives as long as the thread: extend the borrow.
        let (m, cv): (&'static Mutex<()>, &'static Condvar) = unsafe { (&*(m as *const _), &*(cv as *const _)) };
        let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if stop() {
                break;
            }
            let (ng, timed_out) = wait_cv(m, cv, g, Some(deadline));
            g = ng;
            if timed_out {
                break;
            }
        }
    });
}

/// Wake guest thread `handle` if it is blocked in an HLE wait, so that it
/// re-checks its wait condition (used by thread cancellation).
pub fn wake_thread(handle: u64) {
    let g = threads();
    if let Some(t) = g.as_ref().and_then(|t| t.get(&handle)) {
        t.notify_wait();
    }
}

/// Sleep for `d`, delivering pending guest signals meanwhile.
pub fn sleep(d: Duration) {
    sleep_until(Instant::now() + d);
}

// ---------------------------------------------------------------------------
// Raise
// ---------------------------------------------------------------------------

/// Queue `signum` on guest thread `target`, interrupt it, and wait briefly
/// for it to take the signal.
fn raise_remote(target: u64, signum: i32) -> HleResult {
    {
        let mut g = threads();
        let Some(t) = g.as_mut().and_then(|t| t.get_mut(&target)) else { return ESRCH };
        if t.pending == 0 {
            PENDING_THREADS.fetch_add(1, Ordering::AcqRel);
        }
        t.set_pending(t.pending | 1u128 << signum);
        // The thread is registered, so its host thread is alive while the
        // table lock is held.
        if !ps5cpu::signal::kick_thread(t.host) {
            log::warn!("[signal] cannot interrupt thread {target:#x}");
        }
        t.notify_wait();
    }
    let deadline = Instant::now() + DISPATCH_WAIT_MAX;
    let mut g = threads();
    while g.as_ref().and_then(|t| t.get(&target)).is_some_and(|t| t.pending & (1u128 << signum) != 0) {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        g = TAKEN.wait_timeout(g, deadline - now).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(g);
    OK
}

/// Deliver `signum` to the calling thread now, with the context of the
/// current import call.
fn raise_self(signum: i32, handler: u64) -> HleResult {
    let Some(view) = ps5cpu::signal::current_import_interrupt() else { return OK };
    let uc = view.ucontext();
    run_handler(signum, handler, &uc);
    OK
}

/// Raise guest signal `signum` on guest thread `target`.
pub fn raise_exception(target: u64, signum: i32) -> HleResult {
    if signum != SIGUSR1 || target == 0 {
        return EINVAL;
    }
    let handler = handler_for(signum);
    if handler == 0 {
        // Accepted; there is no user callback to deliver.
        return OK;
    }
    if target == current() {
        return raise_self(signum, handler);
    }
    raise_remote(target, signum)
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

/// sceKernelInstallExceptionHandler(signum, handler)
fn cb_install_exception_handler(ctx: &mut CpuContext) -> HleResult {
    let (signum, handler) = (ctx.rdi() as i32, ctx.rsi());
    log::info!("[signal] sceKernelInstallExceptionHandler signum={signum} handler={handler:#x}");
    if !allowed_signal(signum) {
        return EINVAL;
    }
    match HANDLERS[signum as usize].compare_exchange(0, handler, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => OK,
        Err(_) => EAGAIN,
    }
}

/// sceKernelRemoveExceptionHandler(signum)
fn cb_remove_exception_handler(ctx: &mut CpuContext) -> HleResult {
    let signum = ctx.rdi() as i32;
    log::info!("[signal] sceKernelRemoveExceptionHandler signum={signum}");
    if !allowed_signal(signum) {
        return EINVAL;
    }
    HANDLERS[signum as usize].store(0, Ordering::Release);
    OK
}

/// sceKernelRaiseException(thread, signum)
fn cb_raise_exception(ctx: &mut CpuContext) -> HleResult {
    let (target, signum) = (ctx.rdi(), ctx.rsi() as i32);
    log::trace!("[signal] sceKernelRaiseException thread={target:#x} signum={signum}");
    raise_exception(target, signum)
}

pub fn register(reg: &mut HleRegistry) {
    posix::register(reg);
    reg.register_fn("WkwEd3N7w0Y", "sceKernelInstallExceptionHandler", "libKernel", cb_install_exception_handler);
    reg.register_fn("Qhv5ARAoOEc", "sceKernelRemoveExceptionHandler", "libKernel", cb_remove_exception_handler);
    reg.register_fn("il03nluKfMk", "sceKernelRaiseException", "libKernel", cb_raise_exception);
}

#[cfg(test)]
mod tests;
