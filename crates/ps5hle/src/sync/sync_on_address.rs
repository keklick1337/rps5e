// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Sync-on-address (futex-style) waits and `_umtx_op` wait/wake.
//!
//! Guest memory is host memory, so a wait is a host futex wait on the guest
//! word itself and a wake is a host futex wake: no registry, no lock, and
//! guest code that waits or wakes through other paths on the same word
//! interoperates. A wait blocks only while the value equals the expected
//! one; it sleeps in slices of at most 10 ms, delivering pending guest
//! signals between slices, so a store without a wake also releases it.
//! 64-bit waits compare the full word before parking (the futex itself
//! watches the low half); 8- and 16-bit waits park on the aligned 32-bit
//! word that holds them, and a wake of any address wakes that word.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use super::{Export, ERR_INVALID_ARGUMENT, ERR_MEMORY_FAULT, ERR_TIMED_OUT, OK};
use crate::context::CpuContext;
use crate::registry::HleResult;
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant};

const UMTX_OP_WAIT: i32 = 2;
const UMTX_OP_WAKE: i32 = 3;

/// Longest single sleep between guest-signal polls.
const SIGNAL_POLL_MICROS: u32 = 10_000;

/// When a wait gives up: never, or at an instant (`None` inside means the
/// timeout is too large to represent and never expires either, but the
/// wait still counts as timed).
#[derive(Clone, Copy)]
pub(crate) enum Deadline {
    Infinite,
    At(Option<Instant>),
}

impl Deadline {
    pub(crate) fn after(timeout: Duration) -> Self {
        Deadline::At(Instant::now().checked_add(timeout))
    }

    pub(crate) fn from_micros(timeout: Option<u32>) -> Self {
        timeout.map_or(Deadline::Infinite, |us| Self::after(Duration::from_micros(us as u64)))
    }

    fn expired(&self) -> bool {
        matches!(self, Deadline::At(Some(d)) if Instant::now() >= *d)
    }

    /// Microseconds to sleep next; None once the deadline has passed after
    /// the first sleep (the first one still polls with a zero timeout).
    fn slice(&self, first: bool) -> Option<u32> {
        match self {
            Deadline::Infinite | Deadline::At(None) => Some(SIGNAL_POLL_MICROS),
            Deadline::At(Some(end)) => {
                let now = Instant::now();
                if now >= *end {
                    return if first { Some(0) } else { None };
                }
                let left = (*end - now).as_nanos().div_ceil(1000);
                Some(left.min(SIGNAL_POLL_MICROS as u128) as u32)
            }
        }
    }
}

fn read_word(addr: u64, size: u64) -> u64 {
    // SAFETY: the caller checked that the aligned word is tracked guest memory.
    unsafe {
        match size {
            1 => (*(addr as *const AtomicU8)).load(Ordering::Acquire) as u64,
            2 => (*(addr as *const AtomicU16)).load(Ordering::Acquire) as u64,
            4 => (*(addr as *const AtomicU32)).load(Ordering::Acquire) as u64,
            _ => (*(addr as *const AtomicU64)).load(Ordering::Acquire),
        }
    }
}

/// The aligned 32-bit word a wait of `size` bytes at `addr` parks on.
fn futex_word(addr: u64) -> u64 {
    addr & !3
}

#[cfg(windows)]
fn futex_wait(addr: u64, expected: u32, micros: u32) -> Result<(), i32> {
    use windows_sys::Win32::System::Threading::WaitOnAddress;
    let ms = micros.div_ceil(1000);
    // SAFETY: a mapped, aligned guest word; the comparand is a local.
    let ok = unsafe { WaitOnAddress(addr as *const std::ffi::c_void, &expected as *const u32 as *const std::ffi::c_void, 4, ms) };
    if ok != 0 {
        Ok(())
    } else {
        Err(libc::ETIMEDOUT)
    }
}

#[cfg(unix)]
fn futex_wait(addr: u64, expected: u32, micros: u32) -> Result<(), i32> {
    let ts = libc::timespec { tv_sec: (micros / 1_000_000) as libc::time_t, tv_nsec: (micros % 1_000_000) as libc::c_long * 1000 };
    // SAFETY: futex on a mapped, aligned guest word.
    let r = unsafe {
        libc::syscall(libc::SYS_futex, addr as *const u32, libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG, expected, &ts as *const libc::timespec, 0usize, 0u32)
    };
    if r == 0 { Ok(()) } else { Err(std::io::Error::last_os_error().raw_os_error().unwrap_or(0)) }
}

/// Deliver pending guest signals for the calling thread.
fn poll_guest_signals() {
    if crate::signals::pending_for_current() {
        crate::signals::dispatch_pending();
    }
}

/// Wait while the `size`-byte (1, 2, 4 or 8) value at `addr` equals
/// `expected`; the address must be aligned to the size.
pub(crate) fn wait(addr: u64, expected: u64, size: u64, deadline: Deadline, poll: &dyn Fn()) -> HleResult {
    if addr == 0 || addr & (size - 1) != 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if !ps5vm::ranges::accessible(futex_word(addr), 4.max(size), false) {
        return ERR_MEMORY_FAULT;
    }
    let r = wait_word(addr, expected, size, deadline, poll);
    poll();
    r
}

fn wait_word(addr: u64, expected: u64, size: u64, deadline: Deadline, poll: &dyn Fn()) -> HleResult {
    let word = futex_word(addr);
    let mut first = true;
    loop {
        // The futex watches the whole aligned word: the value it expects
        // is the word as read with our part still matching.
        let current = read_word(word, 4) as u32;
        if read_word(addr, size) != expected {
            return OK;
        }
        let Some(slice) = deadline.slice(first) else {
            return if read_word(addr, size) == expected { ERR_TIMED_OUT } else { OK };
        };
        match futex_wait(word, if size >= 4 { expected as u32 } else { current }, slice) {
            Ok(()) => return OK,
            // A wider word may have changed elsewhere: look again.
            Err(libc::EAGAIN) if size < 4 => {}
            Err(libc::EAGAIN) => return OK,
            Err(libc::ETIMEDOUT) | Err(libc::EINTR) => {}
            Err(_) => return ERR_INVALID_ARGUMENT,
        }
        // The process is ending: the guest thread stops here.
        ps5vm::shutdown::park_if_requested();
        poll();
        if deadline.expired() {
            return if read_word(addr, size) == expected { ERR_TIMED_OUT } else { OK };
        }
        first = false;
    }
}

/// Wake up to `count` waiters on `addr` (INT_MAX: all): those parked on
/// the aligned 32-bit word that holds it.
pub(crate) fn wake(addr: u64, count: i32) -> HleResult {
    if addr == 0 || count < 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let addr = futex_word(addr);
    if count == 0 {
        return OK;
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Threading::{WakeByAddressAll, WakeByAddressSingle};
        let p = addr as *const std::ffi::c_void;
        // SAFETY: waking does not access the word's contents.
        unsafe {
            if count == i32::MAX {
                WakeByAddressAll(p);
            } else {
                for _ in 0..count {
                    WakeByAddressSingle(p);
                }
            }
        }
        OK
    }
    #[cfg(unix)]
    {
        // SAFETY: FUTEX_WAKE does not access the word's contents.
        let r = unsafe { libc::syscall(libc::SYS_futex, addr as *const u32, libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG, count, 0usize, 0usize, 0u32) };
        if r < 0 { ERR_INVALID_ARGUMENT } else { OK }
    }
}

/// sceKernelSyncOnAddressWait{,8,16,32,64}(addr, expected,
/// SceKernelUseconds* timeout, const char* name): the comparand is passed by
/// value in the size of the wait; the name is not used.
fn wait_entry(ctx: &mut CpuContext, size: u64) -> HleResult {
    let addr = ctx.rdi();
    let expected = if size == 8 { ctx.rsi() } else { ctx.rsi() & ((1u64 << (size * 8)) - 1) };
    let timeout_ptr = ctx.rdx();
    let timeout = if timeout_ptr != 0 {
        match ctx.read_u32(timeout_ptr) {
            Some(us) => Some(us),
            None => return ERR_MEMORY_FAULT,
        }
    } else {
        None
    };
    wait(addr, expected, size, Deadline::from_micros(timeout), &poll_guest_signals)
}

fn cb_wait(ctx: &mut CpuContext) -> HleResult {
    wait_entry(ctx, 4)
}

fn cb_wait8(ctx: &mut CpuContext) -> HleResult {
    wait_entry(ctx, 1)
}

fn cb_wait16(ctx: &mut CpuContext) -> HleResult {
    wait_entry(ctx, 2)
}

fn cb_wait64(ctx: &mut CpuContext) -> HleResult {
    wait_entry(ctx, 8)
}

/// sceKernelSyncOnAddressWake(addr, int count)
fn cb_wake(ctx: &mut CpuContext) -> HleResult {
    wake(ctx.rdi(), ctx.rsi() as i32)
}

/// The deadline of a `_umtx_op` wait: `uaddr2` NULL waits without limit.
/// A size argument (`uaddr`) of 0 makes the time argument a 32-bit
/// relative timeout in microseconds, as the console's libc.prx passes it
/// (FreeBSD would read a timespec); up to a timespec's size it is a
/// relative `timespec`, otherwise a `struct _umtx_time { timespec; u32
/// flags; u32 clockid; }` whose UMTX_ABSTIME flag makes the timeout an
/// absolute time on that clock. A field out of range is EINVAL.
fn umtx_deadline(ctx: &CpuContext, size: u64, time: u64) -> Result<Deadline, HleResult> {
    const TIMESPEC_SIZE: u64 = 16;
    const UMTX_ABSTIME: u32 = 1;
    /// Waits this long or longer never expire.
    const UNBOUNDED_NANOS: u128 = 1 << 62;
    if time == 0 {
        return Ok(Deadline::Infinite);
    }
    if size == 0 {
        let Some(us) = ctx.read_u32(time) else { return Err(ERR_MEMORY_FAULT) };
        return Ok(Deadline::after(Duration::from_micros(us as u64)));
    }
    let (Some(sec), Some(nsec)) = (ctx.read_u64(time), ctx.read_u64(time + 8)) else { return Err(ERR_MEMORY_FAULT) };
    let (sec, nsec) = (sec as i64, nsec as i64);
    let (flags, clock) = if size > TIMESPEC_SIZE {
        match (ctx.read_u32(time + 16), ctx.read_u32(time + 20)) {
            (Some(f), Some(c)) => (f, c as i32),
            _ => return Err(ERR_MEMORY_FAULT),
        }
    } else {
        (0, crate::time::CLOCK_REALTIME)
    };
    if sec < 0 || !(0..1_000_000_000).contains(&nsec) {
        return Err(ERR_INVALID_ARGUMENT);
    }
    let mut remaining = sec as i128 * 1_000_000_000 + nsec as i128;
    if flags & UMTX_ABSTIME != 0 {
        let Some(host) = crate::time::host_clock(clock) else { return Err(ERR_INVALID_ARGUMENT) };
        let mut now = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: a valid clock id and a valid timespec pointer.
        unsafe { libc::clock_gettime(host, &mut now) };
        remaining -= now.tv_sec as i128 * 1_000_000_000 + now.tv_nsec as i128;
    }
    let remaining = remaining.max(0) as u128;
    if remaining >= UNBOUNDED_NANOS {
        return Ok(Deadline::At(None));
    }
    Ok(Deadline::after(Duration::from_nanos(remaining as u64)))
}

/// _umtx_op(void* obj, int op, u_long val, void* uaddr, void* uaddr2):
/// UMTX_OP_WAIT compares the 64-bit word, UMTX_OP_WAIT_UINT(_PRIVATE) the
/// 32-bit one (a value that does not fit can never match: no wait);
/// UMTX_OP_WAKE(_PRIVATE) wakes up to `val` waiters, at least one;
/// UMTX_OP_NWAKE_PRIVATE wakes every waiter of each of the `val` (an int)
/// addresses listed at the object (none for a count of 0 or less, EFAULT
/// for a null list). A NULL object is EFAULT (a private wake of NULL
/// succeeds), the reserved ops 0 and 1 EOPNOTSUPP,
/// operations outside 0..27 EINVAL; the other operations (mutexes,
/// condition variables) are not implemented: EINVAL, logged.
/// UMTX_OP_SEM_WAIT / UMTX_OP_SEM_WAKE work on a `struct _usem` as FreeBSD
/// 11 does (see [`sem_wait`]).
fn cb_umtx_op(ctx: &mut CpuContext) -> HleResult {
    const UMTX_OP_WAIT_UINT: i32 = 11;
    const UMTX_OP_WAIT_UINT_PRIVATE: i32 = 15;
    const UMTX_OP_WAKE_PRIVATE: i32 = 16;
    const UMTX_OP_SEM_WAIT: i32 = 19;
    const UMTX_OP_SEM_WAKE: i32 = 20;
    const UMTX_OP_NWAKE_PRIVATE: i32 = 21;
    const UMTX_OP_MAX: i32 = 27;
    const ERR_NOT_SUPPORTED: HleResult = 0x8002_002Du32 as i32 as i64;
    let (addr, op, val, size, time) = (ctx.rdi(), ctx.rsi() as i32, ctx.rdx(), ctx.rcx(), ctx.r8());
    let r = match op {
        UMTX_OP_WAIT | UMTX_OP_WAIT_UINT | UMTX_OP_WAIT_UINT_PRIVATE => match umtx_deadline(ctx, size, time) {
            Err(e) => e,
            Ok(_) if addr == 0 => ERR_MEMORY_FAULT,
            Ok(_) if op != UMTX_OP_WAIT && val > u32::MAX as u64 => OK,
            Ok(deadline) => wait(addr, val, if op == UMTX_OP_WAIT { 8 } else { 4 }, deadline, &poll_guest_signals),
        },
        UMTX_OP_NWAKE_PRIVATE => nwake(ctx, addr, val as i32),
        UMTX_OP_SEM_WAIT => match umtx_deadline(ctx, size, time) {
            Err(e) => e,
            Ok(deadline) => sem_wait(addr, deadline),
        },
        UMTX_OP_SEM_WAKE => sem_wake(addr),
        UMTX_OP_WAKE | UMTX_OP_WAKE_PRIVATE => match addr {
            0 if op == UMTX_OP_WAKE_PRIVATE => OK,
            0 => ERR_MEMORY_FAULT,
            _ => wake(addr, (val.min(i32::MAX as u64) as i32).max(1)),
        },
        0 | 1 => ERR_NOT_SUPPORTED,
        _ if !(0..UMTX_OP_MAX).contains(&op) => ERR_INVALID_ARGUMENT,
        _ => {
            log::error!("[umtx] _umtx_op operation {op} is not implemented");
            ERR_INVALID_ARGUMENT
        }
    };
    crate::kernel::posix_call(ctx, r)
}

/// Threads asleep in UMTX_OP_SEM_WAIT, per semaphore address.
#[derive(Default)]
struct SemQueue {
    /// Threads asleep on the semaphore.
    sleeping: u32,
    /// Wakes given out and not yet taken by a sleeper.
    released: u32,
}

static SEM_QUEUES: std::sync::Mutex<std::collections::BTreeMap<u64, SemQueue>> = std::sync::Mutex::new(std::collections::BTreeMap::new());
static SEM_WAKE_CV: std::sync::Condvar = std::sync::Condvar::new();

/// `struct _usem { u32 _has_waiters; u32 _count; u32 _flags; }`: checks the
/// object as FreeBSD's do_sem_wait / do_sem_wake do (NULL EFAULT, an
/// unaligned or untracked word EINVAL / EFAULT).
fn usem_check(addr: u64) -> Result<(), HleResult> {
    if addr == 0 {
        return Err(ERR_MEMORY_FAULT);
    }
    if addr & 3 != 0 {
        return Err(ERR_INVALID_ARGUMENT);
    }
    if !ps5vm::ranges::accessible(addr, 8, true) {
        return Err(ERR_MEMORY_FAULT);
    }
    Ok(())
}

/// UMTX_OP_SEM_WAIT: mark the semaphore as having waiters, return at once
/// while `_count` is non-zero, otherwise sleep until a UMTX_OP_SEM_WAKE
/// releases this thread (0) or the deadline passes (ETIMEDOUT).
fn sem_wait(addr: u64, deadline: Deadline) -> HleResult {
    if let Err(e) = usem_check(addr) {
        return e;
    }
    // SAFETY: usem_check: tracked, writable, aligned guest words.
    let (has_waiters, count) = unsafe { (&*(addr as *const AtomicU32), &*((addr + 4) as *const AtomicU32)) };
    let mut queues = SEM_QUEUES.lock().unwrap_or_else(|e| e.into_inner());
    let _ = has_waiters.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire);
    if count.load(Ordering::Acquire) != 0 {
        return OK;
    }
    queues.entry(addr).or_default().sleeping += 1;
    let until = match deadline {
        Deadline::Infinite | Deadline::At(None) => None,
        Deadline::At(Some(at)) => Some(at),
    };
    let (mut queues, woken) = crate::signals::wait_while(&SEM_QUEUES, &SEM_WAKE_CV, queues, until, |q| q.get(&addr).is_some_and(|s| s.released > 0));
    let queue = queues.get_mut(&addr).expect("a sleeper keeps its queue");
    queue.sleeping -= 1;
    if woken {
        queue.released -= 1;
    }
    if queue.sleeping == 0 {
        queues.remove(&addr);
    }
    if woken { OK } else { ERR_TIMED_OUT }
}

/// UMTX_OP_SEM_WAKE: release one sleeping thread; `_has_waiters` is cleared
/// when it was the last one.
fn sem_wake(addr: u64) -> HleResult {
    if let Err(e) = usem_check(addr) {
        return e;
    }
    let mut queues = SEM_QUEUES.lock().unwrap_or_else(|e| e.into_inner());
    let Some(queue) = queues.get_mut(&addr) else { return OK };
    let waiting = queue.sleeping - queue.released;
    if waiting == 0 {
        return OK;
    }
    if waiting == 1 {
        // SAFETY: usem_check: tracked, writable, aligned guest word.
        unsafe { &*(addr as *const AtomicU32) }.store(0, Ordering::Release);
    }
    queue.released += 1;
    SEM_WAKE_CV.notify_all();
    OK
}

/// UMTX_OP_NWAKE_PRIVATE: wake every waiter on each of `count` addresses
/// listed at `list`.
fn nwake(ctx: &CpuContext, list: u64, count: i32) -> HleResult {
    if count <= 0 {
        return OK;
    }
    if list == 0 {
        return ERR_MEMORY_FAULT;
    }
    for i in 0..count as u64 {
        let Some(addr) = ctx.read_u64(list + i * 8) else { return ERR_MEMORY_FAULT };
        if addr != 0 {
            wake(addr, i32::MAX);
        }
    }
    OK
}

pub(crate) static EXPORTS: &[Export] = &[
    ("Hc4CaR6JBL0", "sceKernelSyncOnAddressWait", "libKernel", cb_wait),
    ("dFym5b41JLU", "sceKernelSyncOnAddressWait8", "libKernel", cb_wait8),
    ("XgVs4jRehcY", "sceKernelSyncOnAddressWait16", "libKernel", cb_wait16),
    ("B2n8aDorSH4", "sceKernelSyncOnAddressWait32", "libKernel", cb_wait),
    ("PZQhiiLXRFs", "sceKernelSyncOnAddressWait64", "libKernel", cb_wait64),
    ("q2y-wDIVWZA", "sceKernelSyncOnAddressWake", "libKernel", cb_wake),
    ("04AjkP0jO9U", "_umtx_op", "libKernel", cb_umtx_op),
];

#[cfg(test)]
mod tests;
