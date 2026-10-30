// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Kernel event flags (`sceKernel*EventFlag`): a 64-bit pattern that
//! threads wait on with AND/OR conditions and optional clear-on-wake.
//!
//! Attributes: queue order `0x01` FIFO / `0x02` priority, waiters `0x10`
//! single / `0x20` multi. Wait modes: `0x01` AND / `0x02` OR, plus
//! `0x10` clear all / `0x20` clear the waited pattern.

use super::{
    kernel_sleep_deadline, read_name, remaining_usec, wait_until, Export, Waited, ERR_BUSY, ERR_CANCELED,
    ERR_DELETED, ERR_INVALID_ARGUMENT, ERR_MEMORY_FAULT, ERR_PERMISSION_DENIED, ERR_TIMED_OUT, OK,
};
use crate::context::CpuContext;
use crate::registry::HleResult;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::Instant;

const MAX_NAME_LEN: usize = 31;

/// ESRCH: not a live event flag.
const ERR_NOT_FOUND: HleResult = 0x8002_0003u32 as i32 as i64;

pub const ATTR_TH_FIFO: u32 = 0x01;
pub const ATTR_TH_PRIO: u32 = 0x02;
pub const ATTR_SINGLE: u32 = 0x10;
pub const ATTR_MULTI: u32 = 0x20;
pub const WAITMODE_AND: u32 = 0x01;
pub const WAITMODE_OR: u32 = 0x02;
pub const WAITMODE_CLEAR_ALL: u32 = 0x10;
pub const WAITMODE_CLEAR_PAT: u32 = 0x20;

struct Inner {
    bits: u64,
    waiting: u32,
    cancel_epoch: u64,
    deleted: bool,
    /// Blocked waiters in arrival order. `set` hands the bits to them in
    /// this order, applying each one's clear mode at once, so a waiter
    /// keeps an event even when the bits are cleared before it runs, and
    /// with clear modes the first waiter wins.
    waiters: Vec<Waiter>,
    next_waiter: u64,
    /// Diagnostics: when bits were last set, and which.
    last_set: Option<(Instant, u64)>,
}

struct Waiter {
    id: u64,
    pattern: u64,
    mode: u32,
    /// The bits that satisfied the waiter, recorded by `set`.
    result: Option<u64>,
}

impl Inner {
    fn take_result(&mut self, id: u64) -> Option<u64> {
        let i = self.waiters.iter().position(|w| w.id == id)?;
        self.waiters.remove(i).result
    }
}

struct EventFlag {
    name: String,
    attr: u32,
    inner: Mutex<Inner>,
    cv: Condvar,
}

static FLAGS: RwLock<Option<HashMap<u64, Arc<EventFlag>>>> = RwLock::new(None);
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

static FLAGS_GENERATION: super::Generation = super::Generation::new();

thread_local! {
    static FLAG_LOOKUPS: super::LocalLookups<EventFlag> = const { super::LocalLookups::new() };
}

fn table<R>(f: impl FnOnce(&mut HashMap<u64, Arc<EventFlag>>) -> R) -> R {
    let mut g = FLAGS.write().unwrap_or_else(|e| e.into_inner());
    let r = f(g.get_or_insert_with(HashMap::new));
    FLAGS_GENERATION.bump();
    r
}

/// Lookups only take the table's read lock.
fn get(handle: u64) -> Option<Arc<EventFlag>> {
    let lookup = || FLAGS.read().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&handle).cloned();
    FLAG_LOOKUPS.try_with(|c| c.get(&FLAGS_GENERATION, handle, lookup)).unwrap_or_else(|_| lookup())
}

impl EventFlag {
    /// Only attribute 0x20 allows several waiters; 0 and 0x10 are single.
    fn single_waiter(&self) -> bool {
        self.attr & 0xF0 != ATTR_MULTI
    }
}

pub(crate) fn valid_attr(attr: u32) -> bool {
    matches!(attr & 0x0F, 0 | ATTR_TH_FIFO | ATTR_TH_PRIO)
        && matches!(attr & 0xF0, 0 | ATTR_SINGLE | ATTR_MULTI)
        && attr & !0x33 == 0
}

pub(crate) fn valid_mode(mode: u32) -> bool {
    matches!(mode & 0x0F, WAITMODE_AND | WAITMODE_OR)
        && matches!(mode & 0xF0, 0 | WAITMODE_CLEAR_ALL | WAITMODE_CLEAR_PAT)
        && mode & !0x33 == 0
}

fn satisfied(bits: u64, pattern: u64, mode: u32) -> bool {
    if mode & 0x0F == WAITMODE_AND {
        bits & pattern == pattern
    } else {
        bits & pattern != 0
    }
}

fn apply_clear(bits: &mut u64, pattern: u64, mode: u32) {
    match mode & 0xF0 {
        WAITMODE_CLEAR_ALL => *bits = 0,
        WAITMODE_CLEAR_PAT => *bits &= !pattern,
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Core operations
// ---------------------------------------------------------------------------

pub(crate) fn create(name: String, attr: u32, init: u64) -> u64 {
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed) + 1;
    let f = Arc::new(EventFlag {
        name,
        attr,
        inner: Mutex::new(Inner { bits: init, waiting: 0, cancel_epoch: 0, deleted: false, waiters: Vec::new(), next_waiter: 0, last_set: None }),
        cv: Condvar::new(),
    });
    table(|t| t.insert(handle, f));
    handle
}

/// Diagnostics (`RPS5E_HLE_STUCK`): name, bits and waiters of a flag.
pub fn describe(handle: u64) -> Option<String> {
    let f = get(handle)?;
    let g = f.inner.lock().unwrap_or_else(|e| e.into_inner());
    let waiters: Vec<String> = g.waiters.iter().map(|w| format!("{:#x}/{:#x}{}", w.pattern, w.mode, if w.result.is_some() { "*" } else { "" })).collect();
    let last = g.last_set.map_or("never".to_string(), |(t, p)| format!("{p:#x} {:.1} s ago", t.elapsed().as_secs_f64()));
    Some(format!("'{}' attr {:#x} bits {:#x} waiting {} [{}] last set {last}", f.name, f.attr, g.bits, g.waiting, waiters.join(" ")))
}

pub(crate) fn delete(handle: u64) -> bool {
    let Some(f) = table(|t| t.remove(&handle)) else { return false };
    f.inner.lock().unwrap().deleted = true;
    f.cv.notify_all();
    log::debug!("[event_flag] delete handle={handle:#x} name='{}'", f.name);
    true
}

pub(crate) fn set(handle: u64, pattern: u64) -> HleResult {
    let Some(f) = get(handle) else { return ERR_NOT_FOUND };
    let mut g = f.inner.lock().unwrap();
    g.bits |= pattern;
    g.last_set = Some((Instant::now(), pattern));
    let mut bits = g.bits;
    let mut woke = false;
    for w in g.waiters.iter_mut() {
        if w.result.is_none() && satisfied(bits, w.pattern, w.mode) {
            w.result = Some(bits);
            apply_clear(&mut bits, w.pattern, w.mode);
            woke = true;
        }
    }
    g.bits = bits;
    drop(g);
    if woke {
        f.cv.notify_all();
    }
    OK
}

pub(crate) fn clear(handle: u64, mask: u64) -> HleResult {
    let Some(f) = get(handle) else { return ERR_NOT_FOUND };
    f.inner.lock().unwrap().bits &= mask;
    OK
}

/// Returns (result, bits at the time of the check).
pub(crate) fn poll(handle: u64, pattern: u64, mode: u32) -> (HleResult, u64) {
    let Some(f) = get(handle) else { return (ERR_NOT_FOUND, 0) };
    if pattern == 0 || !valid_mode(mode) {
        return (ERR_INVALID_ARGUMENT, 0);
    }
    let mut g = f.inner.lock().unwrap();
    let bits = g.bits;
    if f.single_waiter() && g.waiting > 0 {
        return (ERR_PERMISSION_DENIED, bits);
    }
    if !satisfied(bits, pattern, mode) {
        return (ERR_BUSY, bits);
    }
    apply_clear(&mut g.bits, pattern, mode);
    (OK, bits)
}

/// Block until the condition holds, the flag is cancelled/deleted, or the
/// deadline passes. Returns (result, bits observed at wake, before clear).
pub(crate) fn wait(handle: u64, pattern: u64, mode: u32, deadline: Option<Instant>) -> (HleResult, u64) {
    let Some(f) = get(handle) else { return (ERR_NOT_FOUND, 0) };
    if pattern == 0 || !valid_mode(mode) {
        return (ERR_INVALID_ARGUMENT, 0);
    }
    let mut g = f.inner.lock().unwrap();
    if f.single_waiter() && g.waiting > 0 {
        return (ERR_PERMISSION_DENIED, g.bits);
    }
    if satisfied(g.bits, pattern, mode) {
        let bits = g.bits;
        apply_clear(&mut g.bits, pattern, mode);
        return (OK, bits);
    }
    let epoch = g.cancel_epoch;
    g.waiting += 1;
    g.next_waiter += 1;
    let id = g.next_waiter;
    g.waiters.push(Waiter { id, pattern, mode, result: None });
    loop {
        let timed_out = match wait_until(&f.inner, &f.cv, g, deadline, None) {
            Waited::Woken(ng) => {
                g = ng;
                false
            }
            Waited::TimedOut(ng) => {
                g = ng;
                true
            }
        };
        let done = g.deleted
            || g.cancel_epoch != epoch
            || timed_out
            || g.waiters.iter().any(|w| w.id == id && w.result.is_some());
        if !done {
            continue;
        }
        let result = g.take_result(id);
        if g.deleted {
            return (ERR_DELETED, g.bits);
        }
        if g.cancel_epoch != epoch {
            return (ERR_CANCELED, g.bits);
        }
        g.waiting = g.waiting.saturating_sub(1);
        return match result {
            Some(bits) => (OK, bits),
            None => (ERR_TIMED_OUT, g.bits),
        };
    }
}

/// Set the pattern to `set_pattern` and release every waiter with
/// CANCELED. Returns (result, number of waiters released).
pub(crate) fn cancel(handle: u64, set_pattern: u64) -> (HleResult, u32) {
    let Some(f) = get(handle) else { return (ERR_NOT_FOUND, 0) };
    let mut g = f.inner.lock().unwrap();
    let waiting = g.waiting;
    g.bits = set_pattern;
    g.waiting = 0;
    g.cancel_epoch += 1;
    drop(g);
    f.cv.notify_all();
    (OK, waiting)
}

// ---------------------------------------------------------------------------
// HLE callbacks
// ---------------------------------------------------------------------------

/// sceKernelCreateEventFlag(SceKernelEventFlag* ef, const char* name, u32 attr, u64 init, opt*)
fn cb_create(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    let name_ptr = ctx.rsi();
    let attr = ctx.rdx() as u32;
    let init = ctx.rcx();
    let opt = ctx.r8();
    if out == 0 || name_ptr == 0 || opt != 0 || !valid_attr(attr) {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(name) = read_name(ctx, name_ptr, MAX_NAME_LEN + 1) else { return ERR_MEMORY_FAULT };
    log::debug!("[event_flag] create name='{name}' attr={attr:#x} init={init:#x}");
    let handle = create(name, attr, init);
    if !ctx.write_u64(out, handle) {
        delete(handle);
        return ERR_MEMORY_FAULT;
    }
    OK
}

/// sceKernelOpenEventFlag(SceKernelEventFlag* ef, const char* name): the
/// live event flag with that exact name; ENOENT when there is none.
fn cb_open(ctx: &mut CpuContext) -> HleResult {
    let (out, name_ptr) = (ctx.rdi(), ctx.rsi());
    if out == 0 || name_ptr == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(name) = read_name(ctx, name_ptr, MAX_NAME_LEN + 1) else { return ERR_MEMORY_FAULT };
    let found = table(|t| t.iter().filter(|(_, f)| f.name == name).map(|(&h, _)| h).min());
    let Some(handle) = found else { return ERR_NOT_FOUND };
    if ctx.write_u64(out, handle) { OK } else { ERR_MEMORY_FAULT }
}

/// sceKernelDeleteEventFlag(ef)
fn cb_delete(ctx: &mut CpuContext) -> HleResult {
    if delete(ctx.rdi()) { OK } else { ERR_NOT_FOUND }
}

/// sceKernelSetEventFlag(ef, u64 pattern)
fn cb_set(ctx: &mut CpuContext) -> HleResult {
    set(ctx.rdi(), ctx.rsi())
}

/// sceKernelClearEventFlag(ef, u64 mask): bits &= mask
fn cb_clear(ctx: &mut CpuContext) -> HleResult {
    clear(ctx.rdi(), ctx.rsi())
}

fn write_result(ctx: &CpuContext, addr: u64, bits: u64) -> bool {
    addr == 0 || ctx.write_u64(addr, bits)
}

/// sceKernelPollEventFlag(ef, u64 pattern, u32 mode, u64* result)
fn cb_poll(ctx: &mut CpuContext) -> HleResult {
    let result_ptr = ctx.rcx();
    let (r, bits) = poll(ctx.rdi(), ctx.rsi(), ctx.rdx() as u32);
    if r == ERR_NOT_FOUND || r == ERR_INVALID_ARGUMENT {
        return r;
    }
    if !write_result(ctx, result_ptr, bits) {
        return ERR_MEMORY_FAULT;
    }
    r
}

/// sceKernelWaitEventFlag(ef, u64 pattern, u32 mode, u64* result, SceKernelUseconds* timeout)
fn cb_wait(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    let pattern = ctx.rsi();
    let mode = ctx.rdx() as u32;
    let result_ptr = ctx.rcx();
    let timeout_ptr = ctx.r8();
    if get(handle).is_none() {
        return ERR_NOT_FOUND;
    }
    if pattern == 0 || !valid_mode(mode) {
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
    // An unsatisfied wait sleeps for at least one scheduler tick.
    let deadline = timeout.map(kernel_sleep_deadline);
    let (r, bits) = wait(handle, pattern, mode, deadline);
    if r == ERR_NOT_FOUND {
        return r;
    }
    if let (Some(t), Some(d)) = (timeout, deadline) {
        ctx.write_u32(timeout_ptr, if r == ERR_TIMED_OUT { 0 } else { remaining_usec(d).min(t) });
    }
    if !write_result(ctx, result_ptr, bits) {
        return ERR_MEMORY_FAULT;
    }
    r
}

/// sceKernelCancelEventFlag(ef, u64 set_pattern, int* waiting_threads)
fn cb_cancel(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    let out = ctx.rdx();
    if get(handle).is_none() {
        return ERR_NOT_FOUND;
    }
    if out != 0 && !ctx.write_u32(out, 0) {
        return ERR_MEMORY_FAULT;
    }
    let (r, waiting) = cancel(handle, ctx.rsi());
    if r == OK && out != 0 {
        ctx.write_u32(out, waiting);
    }
    r
}

pub(crate) static EXPORTS: &[Export] = &[
    ("BpFoboUJoZU", "sceKernelCreateEventFlag", "libKernel", cb_create),
    ("1vDaenmJtyA", "sceKernelOpenEventFlag", "libKernel", cb_open),
    ("8mql9OcQnd4", "sceKernelDeleteEventFlag", "libKernel", cb_delete),
    ("IOnSvHzqu6A", "sceKernelSetEventFlag", "libKernel", cb_set),
    ("7uhBFWRAS60", "sceKernelClearEventFlag", "libKernel", cb_clear),
    ("9lvj5DjHZiA", "sceKernelPollEventFlag", "libKernel", cb_poll),
    ("JTvBflhYazQ", "sceKernelWaitEventFlag", "libKernel", cb_wait),
    ("PZku4ZrXJqg", "sceKernelCancelEventFlag", "libKernel", cb_cancel),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::test_util::vm_arena;
    use crate::sync::{usec_deadline, KERNEL_TICK_USEC};
    use crate::CpuRegister;
    use std::thread;
    use std::time::Duration;

    /// A zero timeout on an unsatisfied flag still sleeps for one scheduler
    /// tick (a poll loop must not spin), and a set during that tick wakes
    /// the waiter with success; a satisfied flag returns at once.
    /// sceKernelOpenEventFlag finds a live flag by its exact name.
    #[test]
    fn open_by_name() {
        const BASE: u64 = 0x7B3A_5000_0000;
        let vm = vm_arena(BASE);
        let mut ctx = CpuContext::new(vm);
        let h = create("open-by-name-test".into(), ATTR_MULTI, 0);
        ctx.write_bytes(BASE + 0x100, b"open-by-name-test\0");
        ctx.write_bytes(BASE + 0x200, b"open-by-name\0");
        let open = |ctx: &mut CpuContext, out: u64, name: u64| {
            ctx.set(CpuRegister::Rdi, out);
            ctx.set(CpuRegister::Rsi, name);
            cb_open(ctx)
        };
        assert_eq!(open(&mut ctx, BASE, BASE + 0x100), OK);
        assert_eq!(ctx.read_u64(BASE), Some(h));
        assert_eq!(open(&mut ctx, BASE, BASE + 0x200), ERR_NOT_FOUND);
        assert_eq!(open(&mut ctx, 0, BASE + 0x100), ERR_INVALID_ARGUMENT);
        assert_eq!(open(&mut ctx, BASE, 0), ERR_INVALID_ARGUMENT);
        assert!(delete(h));
        assert_eq!(open(&mut ctx, BASE, BASE + 0x100), ERR_NOT_FOUND);
    }

    #[test]
    fn zero_timeout_sleeps_one_tick() {
        const BASE: u64 = 0x7B31_0000_0000;
        let vm = vm_arena(BASE);
        let mut ctx = CpuContext::new(vm);
        let h = create("z".into(), ATTR_MULTI, 0);
        ctx.write_u32(BASE + 0x10, 0);
        ctx.set(CpuRegister::Rdi, h);
        ctx.set(CpuRegister::Rsi, 0x1);
        ctx.set(CpuRegister::Rdx, (WAITMODE_OR | WAITMODE_CLEAR_ALL) as u64);
        ctx.set(CpuRegister::Rcx, 0);
        ctx.set(CpuRegister::R8, BASE + 0x10);
        let start = Instant::now();
        assert_eq!(cb_wait(&mut ctx), ERR_TIMED_OUT);
        assert!(start.elapsed() >= Duration::from_micros(KERNEL_TICK_USEC as u64));
        assert_eq!(ctx.read_u32(BASE + 0x10), Some(0));

        set(h, 0x1);
        assert_eq!(cb_wait(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE + 0x10), Some(0));

        // Set while the waiter sleeps its tick: success.
        let t = thread::spawn(move || {
            thread::sleep(Duration::from_micros(200));
            set(h, 0x1)
        });
        let r = wait(h, 1, WAITMODE_OR, Some(kernel_sleep_deadline(0)));
        t.join().unwrap();
        assert_eq!(r.0, OK);
    }

    /// A set hands the bits to a blocked waiter at once: clearing them
    /// right after the set does not take the event away from it.
    #[test]
    fn set_then_clear_still_releases_the_waiter() {
        let h = create("sc".into(), ATTR_MULTI, 0);
        let t = thread::spawn(move || wait(h, 1, WAITMODE_AND | WAITMODE_CLEAR_PAT, Some(usec_deadline(2_000_000))));
        while get(h).unwrap().inner.lock().unwrap().waiters.is_empty() {
            thread::yield_now();
        }
        set(h, 1);
        clear(h, 0);
        assert_eq!(t.join().unwrap(), (OK, 1));
        assert!(delete(h));
    }

    /// Two waiters with a clear mode: the first to arrive takes the bits,
    /// the second times out, and the bits end cleared.
    #[test]
    fn first_waiter_takes_cleared_bits() {
        let h = create("fw".into(), ATTR_MULTI, 0);
        let n = |h| get(h).unwrap().inner.lock().unwrap().waiters.len();
        let a = thread::spawn(move || wait(h, 1, WAITMODE_AND | WAITMODE_CLEAR_ALL, Some(usec_deadline(200_000))));
        while n(h) < 1 {
            thread::yield_now();
        }
        let b = thread::spawn(move || wait(h, 1, WAITMODE_AND | WAITMODE_CLEAR_ALL, Some(usec_deadline(200_000))));
        while n(h) < 2 {
            thread::yield_now();
        }
        set(h, 1);
        assert_eq!(a.join().unwrap(), (OK, 1));
        assert_eq!(b.join().unwrap().0, ERR_TIMED_OUT);
        assert_eq!(poll(h, 1, WAITMODE_OR).0, ERR_BUSY);
        assert!(delete(h));
    }

    #[test]
    fn validation() {
        assert!(valid_attr(0) && valid_attr(0x11) && valid_attr(0x22));
        assert!(!valid_attr(0x03) && !valid_attr(0x30) && !valid_attr(0x100));
        assert!(valid_mode(0x01) && valid_mode(0x12) && valid_mode(0x21));
        assert!(!valid_mode(0) && !valid_mode(0x03) && !valid_mode(0x31));
    }

    #[test]
    fn and_or_and_clear_modes() {
        let h = create("t".into(), ATTR_MULTI, 0b0101);
        assert_eq!(poll(h, 0b0111, WAITMODE_AND), (ERR_BUSY, 0b0101));
        assert_eq!(poll(h, 0b0110, WAITMODE_OR), (OK, 0b0101));
        assert_eq!(poll(h, 0b0101, WAITMODE_AND | WAITMODE_CLEAR_PAT), (OK, 0b0101));
        assert_eq!(poll(h, 0b1111, WAITMODE_OR), (ERR_BUSY, 0));
        assert_eq!(set(h, 0b1011), OK);
        assert_eq!(poll(h, 0b0001, WAITMODE_OR | WAITMODE_CLEAR_ALL), (OK, 0b1011));
        assert_eq!(poll(h, u64::MAX, WAITMODE_OR), (ERR_BUSY, 0));
        set(h, 0xF0);
        assert_eq!(clear(h, 0x30), OK); // bits &= mask
        assert_eq!(poll(h, 0x30, WAITMODE_AND), (OK, 0x30));
        assert_eq!(poll(h, 0, WAITMODE_AND).0, ERR_INVALID_ARGUMENT);
        assert!(delete(h));
        assert_eq!(set(h, 1), ERR_NOT_FOUND);
    }

    #[test]
    fn wait_across_threads_with_timeout() {
        let h = create("t".into(), ATTR_MULTI, 0);
        let start = Instant::now();
        assert_eq!(wait(h, 1, WAITMODE_OR, Some(usec_deadline(20_000))), (ERR_TIMED_OUT, 0));
        assert!(start.elapsed() >= Duration::from_millis(20));
        let t = thread::spawn(move || wait(h, 0b11, WAITMODE_AND | WAITMODE_CLEAR_PAT, None));
        thread::sleep(Duration::from_millis(20));
        set(h, 0b01);
        thread::sleep(Duration::from_millis(10));
        assert!(!t.is_finished());
        set(h, 0b110);
        assert_eq!(t.join().unwrap(), (OK, 0b111));
        assert_eq!(poll(h, 0b100, WAITMODE_AND), (OK, 0b100));
    }

    #[test]
    fn single_waiter_cancel_delete() {
        // Attribute 0 means a single waiter, like 0x10.
        let h = create("t".into(), 0, 0);
        let t = thread::spawn(move || wait(h, 1, WAITMODE_OR, None));
        thread::sleep(Duration::from_millis(30));
        assert_eq!(wait(h, 1, WAITMODE_OR, None).0, ERR_PERMISSION_DENIED);
        // Even a satisfiable poll is refused while the single waiter waits.
        assert_eq!(poll(h, 1, WAITMODE_OR).0, ERR_PERMISSION_DENIED);
        assert_eq!(cancel(h, 0x8), (OK, 1));
        assert_eq!(t.join().unwrap(), (ERR_CANCELED, 0x8));
        let t = thread::spawn(move || wait(h, 1, WAITMODE_OR, None));
        thread::sleep(Duration::from_millis(20));
        assert!(delete(h));
        assert_eq!(t.join().unwrap().0, ERR_DELETED);
    }

    #[test]
    fn wait_callback_writes_result_and_timeout() {
        const BASE: u64 = 0x7B30_0000_0000;
        let vm = vm_arena(BASE);
        let mut ctx = CpuContext::new(vm);
        ctx.write_bytes(BASE + 0x100, b"flag\0");
        ctx.set(CpuRegister::Rdi, BASE);
        ctx.set(CpuRegister::Rsi, BASE + 0x100);
        ctx.set(CpuRegister::Rdx, (ATTR_MULTI | ATTR_TH_FIFO) as u64);
        ctx.set(CpuRegister::Rcx, 0x3);
        assert_eq!(cb_create(&mut ctx), OK);
        let h = ctx.read_u64(BASE).unwrap();
        ctx.regs = [0; 16];
        ctx.write_u32(BASE + 0x10, 1000);
        ctx.set(CpuRegister::Rdi, h);
        ctx.set(CpuRegister::Rsi, 0x4);
        ctx.set(CpuRegister::Rdx, WAITMODE_AND as u64);
        ctx.set(CpuRegister::Rcx, BASE + 0x8);
        ctx.set(CpuRegister::R8, BASE + 0x10);
        assert_eq!(cb_wait(&mut ctx), ERR_TIMED_OUT);
        assert_eq!(ctx.read_u64(BASE + 0x8), Some(0x3));
        assert_eq!(ctx.read_u32(BASE + 0x10), Some(0));
        ctx.set(CpuRegister::Rsi, 0x2);
        ctx.set(CpuRegister::Rdx, (WAITMODE_OR | WAITMODE_CLEAR_ALL) as u64);
        ctx.set(CpuRegister::R8, 0);
        assert_eq!(cb_wait(&mut ctx), OK);
        assert_eq!(ctx.read_u64(BASE + 0x8), Some(0x3));
        assert_eq!(cb_poll(&mut ctx), ERR_BUSY);
        assert_eq!(ctx.read_u64(BASE + 0x8), Some(0));
        ctx.set(CpuRegister::Rdx, BASE + 0x20);
        assert_eq!(cb_cancel(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE + 0x20), Some(0));
        assert_eq!(cb_delete(&mut ctx), OK);
        assert_eq!(cb_delete(&mut ctx), ERR_NOT_FOUND);
    }
}
