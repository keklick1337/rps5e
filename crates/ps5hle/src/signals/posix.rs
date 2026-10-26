// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! POSIX signal interface of libkernel: signal sets, dispositions
//! (`signal`, `sigaction`), the per-thread signal mask (`sigprocmask`,
//! `pthread_sigmask`), `raise` / `kill` to the own process, the alternate
//! signal stack record and `sigsetjmp` / `siglongjmp`.
//!
//! Layouts and rules follow FreeBSD (amd64):
//!   * `sigset_t` is four u32 words, signal n at bit n - 1 (`_SIG_BIT`);
//!     signals are 1..=128.
//!   * `struct sigaction` is { handler u64 @0, flags i32 @8, mask @12 },
//!     32 bytes.
//!   * SIGKILL and SIGSTOP can never be blocked or caught.
//!   * The mask is per thread. A signal raised while blocked stays pending
//!     on the thread until it is unblocked.
//!
//! Host signals are never forwarded to guest handlers; only signals the
//! guest sends to itself (`raise`, `kill` of its own pid) are delivered,
//! synchronously on the calling thread, from inside the import.

use super::run_handler_args;
use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};
use std::cell::Cell;
use std::sync::Mutex;

const LIB: &str = "libKernel";

pub(super) const SIG_MAX: i32 = 128;
const SIGKILL: i32 = 9;
const SIGSTOP: i32 = 17;

const SIG_DFL: u64 = 0;
const SIG_IGN: u64 = 1;
const SIG_ERR: u64 = u64::MAX;

const SA_RESETHAND: i32 = 0x4;
const SA_RESTART: i32 = 0x2;
const SA_NODEFER: i32 = 0x10;
const SA_SIGINFO: i32 = 0x40;

const SIG_BLOCK: i32 = 1;
const SIG_UNBLOCK: i32 = 2;
const SIG_SETMASK: i32 = 3;

const SS_DISABLE: i32 = 4;
const MINSIGSTKSZ: u64 = 2048;

/// `si_code` of a signal sent by a process (`SI_USER`).
const SI_USER: i32 = 0x10001;
/// FreeBSD `siginfo_t` size.
const SIGINFO_SIZE: usize = 0x50;

const ESRCH: i32 = 3;
const ENOMEM: i32 = 12;
const EFAULT: i32 = 14;
const EINVAL: i32 = 22;

/// FreeBSD `sigset_t`.
pub(crate) type SigSet = [u32; 4];

const EMPTY: SigSet = [0; 4];

fn valid_signal(sig: i32) -> bool {
    (1..=SIG_MAX).contains(&sig)
}

fn word_bit(sig: i32) -> (usize, u32) {
    (((sig - 1) >> 5) as usize, 1u32 << ((sig - 1) & 31))
}

fn has(set: &SigSet, sig: i32) -> bool {
    let (w, b) = word_bit(sig);
    set[w] & b != 0
}

fn add(set: &mut SigSet, sig: i32) {
    let (w, b) = word_bit(sig);
    set[w] |= b;
}

fn del(set: &mut SigSet, sig: i32) {
    let (w, b) = word_bit(sig);
    set[w] &= !b;
}

/// The signals that can never be blocked (`SIG_CANTMASK`).
fn cant_mask(mut set: SigSet) -> SigSet {
    del(&mut set, SIGKILL);
    del(&mut set, SIGSTOP);
    set
}

fn read_set(ctx: &CpuContext, addr: u64) -> Option<SigSet> {
    if addr == 0 {
        return None;
    }
    let mut s = EMPTY;
    for (i, w) in s.iter_mut().enumerate() {
        *w = ctx.read_u32(addr + i as u64 * 4)?;
    }
    Some(s)
}

fn write_set(ctx: &CpuContext, addr: u64, set: &SigSet) -> bool {
    addr != 0 && set.iter().enumerate().all(|(i, &w)| ctx.write_u32(addr + i as u64 * 4, w))
}

fn fail(ctx: &CpuContext, errno: i32) -> HleResult {
    crate::kernel::set_errno(ctx, errno);
    -1
}

// ---------------------------------------------------------------------------
// Per-thread mask and pending set
// ---------------------------------------------------------------------------

thread_local! {
    static BLOCKED: Cell<SigSet> = const { Cell::new(EMPTY) };
    static PENDING: Cell<SigSet> = const { Cell::new(EMPTY) };
    /// (sp, size, enabled) of the alternate signal stack.
    static ALT_STACK: Cell<(u64, u64, bool)> = const { Cell::new((0, 0, false)) };
}

/// The calling thread's signal mask.
pub(crate) fn blocked_mask() -> SigSet {
    BLOCKED.with(|b| b.get())
}

/// Change the calling thread's mask like `sigprocmask(how, set)`; returns
/// the old mask, or None for an invalid `how`. Pending signals that the new
/// mask unblocks are delivered before returning.
fn change_mask(how: i32, set: Option<SigSet>) -> Option<SigSet> {
    let old = blocked_mask();
    if let Some(s) = set {
        let new = match how {
            SIG_BLOCK => std::array::from_fn(|i| old[i] | s[i]),
            SIG_UNBLOCK => std::array::from_fn(|i| old[i] & !s[i]),
            SIG_SETMASK => s,
            _ => return None,
        };
        BLOCKED.with(|b| b.set(cant_mask(new)));
        deliver_unblocked();
    }
    Some(old)
}

/// Deliver the calling thread's pending signals the mask no longer blocks.
fn deliver_unblocked() {
    loop {
        let mask = blocked_mask();
        let pending = PENDING.with(|p| p.get());
        let Some(sig) = (1..=SIG_MAX).find(|&s| has(&pending, s) && !has(&mask, s)) else { return };
        PENDING.with(|p| {
            let mut v = p.get();
            del(&mut v, sig);
            p.set(v);
        });
        deliver(sig);
    }
}

// ---------------------------------------------------------------------------
// Dispositions
// ---------------------------------------------------------------------------

/// Guest `struct sigaction`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SigAction {
    pub handler: u64,
    pub flags: i32,
    pub mask: SigSet,
}

const SIGACTION_SIZE: usize = 32;

impl SigAction {
    fn read(ctx: &CpuContext, addr: u64) -> Option<Self> {
        if addr == 0 {
            return None;
        }
        let b = ctx.read_bytes(addr, SIGACTION_SIZE)?;
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        Some(SigAction {
            handler: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            flags: u32_at(8) as i32,
            mask: [u32_at(12), u32_at(16), u32_at(20), u32_at(24)],
        })
    }

    fn write(&self, ctx: &CpuContext, addr: u64) -> bool {
        let mut b = [0u8; SIGACTION_SIZE];
        b[0..8].copy_from_slice(&self.handler.to_le_bytes());
        b[8..12].copy_from_slice(&self.flags.to_le_bytes());
        for (i, w) in self.mask.iter().enumerate() {
            b[12 + i * 4..16 + i * 4].copy_from_slice(&w.to_le_bytes());
        }
        addr != 0 && ctx.write_bytes(addr, &b)
    }
}

static DISPOSITIONS: Mutex<[SigAction; SIG_MAX as usize + 1]> =
    Mutex::new([SigAction { handler: 0, flags: 0, mask: EMPTY }; SIG_MAX as usize + 1]);

fn dispositions() -> std::sync::MutexGuard<'static, [SigAction; SIG_MAX as usize + 1]> {
    DISPOSITIONS.lock().unwrap_or_else(|e| e.into_inner())
}

/// `kern_sigaction`: install `act` (if any) for `sig`; returns the previous
/// disposition or the errno.
fn set_action(sig: i32, act: Option<SigAction>) -> Result<SigAction, i32> {
    if !valid_signal(sig) {
        return Err(EINVAL);
    }
    if let Some(a) = act {
        if (sig == SIGKILL || sig == SIGSTOP) && a.handler != SIG_DFL {
            return Err(EINVAL);
        }
    }
    let mut d = dispositions();
    let old = d[sig as usize];
    if let Some(mut a) = act {
        a.mask = cant_mask(a.mask);
        d[sig as usize] = a;
    }
    Ok(old)
}

pub(crate) fn action(sig: i32) -> SigAction {
    if valid_signal(sig) { dispositions()[sig as usize] } else { SigAction::default() }
}

/// Signals whose default action is to be discarded.
fn default_ignored(sig: i32) -> bool {
    // SIGURG, SIGCONT, SIGCHLD, SIGIO, SIGWINCH, SIGINFO, SIGTHR, SIGLIBRT.
    matches!(sig, 16 | 19 | 20 | 23 | 28 | 29 | 32 | 33)
}

/// Signals whose default action stops the process.
fn default_stops(sig: i32) -> bool {
    // SIGSTOP, SIGTSTP, SIGTTIN, SIGTTOU.
    matches!(sig, 17 | 18 | 21 | 22)
}

/// Take signal `sig` on the calling thread now (it is not blocked).
fn deliver(sig: i32) {
    let act = action(sig);
    match act.handler {
        SIG_IGN => {}
        SIG_DFL => {
            if default_ignored(sig) {
                return;
            }
            if default_stops(sig) {
                // Stopping needs a job-control parent; nothing can resume
                // the process, so the stop is not modelled.
                log::warn!("[signal] stop signal {sig} with the default action is not modelled");
                return;
            }
            log::error!("[signal] guest signal {sig} with the default action terminates the process");
            ps5vm::shutdown::exit(128 + sig)
        }
        handler => {
            // The handler runs with its mask and the signal itself blocked.
            let saved = blocked_mask();
            let mut during = act.mask;
            if act.flags & SA_NODEFER == 0 {
                add(&mut during, sig);
            }
            BLOCKED.with(|b| b.set(cant_mask(std::array::from_fn(|i| saved[i] | during[i]))));
            if act.flags & SA_RESETHAND != 0 && sig != SIGKILL && sig != SIGSTOP {
                let _ = set_action(sig, Some(SigAction::default()));
            }
            let info = user_siginfo(sig);
            let siginfo = act.flags & SA_SIGINFO != 0;
            if !run_handler_args(sig, handler, saved, siginfo.then_some(&info[..])) {
                log::warn!("[signal] signal {sig}: handler {handler:#x} cannot run outside an import");
            }
            BLOCKED.with(|b| b.set(saved));
        }
    }
}

/// Send `sig` to the calling thread: deliver it now, or keep it pending
/// while the thread blocks it.
fn send_to_self(sig: i32) {
    // An ignored signal is discarded when it is sent, blocked or not (as
    // FreeBSD's tdsendsignal does); SIGCONT with the default action still
    // becomes pending.
    const SIGCONT: i32 = 19;
    let act = action(sig);
    if act.handler == SIG_IGN || (act.handler == SIG_DFL && default_ignored(sig) && sig != SIGCONT) {
        return;
    }
    if has(&blocked_mask(), sig) && sig != SIGKILL && sig != SIGSTOP {
        PENDING.with(|p| {
            let mut v = p.get();
            add(&mut v, sig);
            p.set(v);
        });
        return;
    }
    deliver(sig);
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

/// sigemptyset(set*)
fn cb_sigemptyset(ctx: &mut CpuContext) -> HleResult {
    if !write_set(ctx, ctx.rdi(), &EMPTY) {
        return fail(ctx, EFAULT);
    }
    0
}

/// sigfillset(set*)
fn cb_sigfillset(ctx: &mut CpuContext) -> HleResult {
    if !write_set(ctx, ctx.rdi(), &[u32::MAX; 4]) {
        return fail(ctx, EFAULT);
    }
    0
}

fn edit_set(ctx: &CpuContext, f: impl FnOnce(&mut SigSet, i32)) -> HleResult {
    let (addr, sig) = (ctx.rdi(), ctx.rsi() as i32);
    if !valid_signal(sig) {
        return fail(ctx, EINVAL);
    }
    let Some(mut set) = read_set(ctx, addr) else { return fail(ctx, EFAULT) };
    f(&mut set, sig);
    if !write_set(ctx, addr, &set) {
        return fail(ctx, EFAULT);
    }
    0
}

/// sigaddset(set*, sig)
fn cb_sigaddset(ctx: &mut CpuContext) -> HleResult {
    edit_set(ctx, add)
}

/// sigdelset(set*, sig)
fn cb_sigdelset(ctx: &mut CpuContext) -> HleResult {
    edit_set(ctx, del)
}

/// sigismember(set*, sig) -> 1 / 0
fn cb_sigismember(ctx: &mut CpuContext) -> HleResult {
    let (addr, sig) = (ctx.rdi(), ctx.rsi() as i32);
    if !valid_signal(sig) {
        return fail(ctx, EINVAL);
    }
    let Some(set) = read_set(ctx, addr) else { return fail(ctx, EFAULT) };
    has(&set, sig) as HleResult
}

/// sigprocmask core: returns the errno.
fn procmask(ctx: &CpuContext, how: i32, set_ptr: u64, old_ptr: u64) -> i32 {
    let set = if set_ptr != 0 {
        match read_set(ctx, set_ptr) {
            Some(s) => Some(s),
            None => return EFAULT,
        }
    } else {
        None
    };
    let Some(old) = change_mask(how, set) else { return EINVAL };
    if old_ptr != 0 && !write_set(ctx, old_ptr, &old) {
        return EFAULT;
    }
    0
}

/// sigprocmask / _sigprocmask(how, set*, oset*) -> 0 / -1 with errno
fn cb_sigprocmask(ctx: &mut CpuContext) -> HleResult {
    match procmask(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx()) {
        0 => 0,
        e => fail(ctx, e),
    }
}

/// pthread_sigmask(how, set*, oset*) -> 0 / errno (errno untouched)
fn cb_pthread_sigmask(ctx: &mut CpuContext) -> HleResult {
    procmask(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx()) as HleResult
}

/// signal(sig, handler) -> previous handler / SIG_ERR. Installs the handler
/// with SA_RESTART and an empty mask, as FreeBSD's libc does.
pub(super) fn cb_signal(ctx: &mut CpuContext) -> HleResult {
    let (sig, handler) = (ctx.rdi() as i32, ctx.rsi());
    match set_action(sig, Some(SigAction { handler, flags: SA_RESTART, mask: EMPTY })) {
        Ok(old) => {
            log::debug!("[signal] signal({sig}, {handler:#x}) prev={:#x}", old.handler);
            old.handler as HleResult
        }
        Err(e) => {
            crate::kernel::set_errno(ctx, e);
            SIG_ERR as HleResult
        }
    }
}

/// sigaction / _sigaction(sig, act*, oact*)
fn cb_sigaction(ctx: &mut CpuContext) -> HleResult {
    let (sig, act_ptr, old_ptr) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    let act = if act_ptr != 0 {
        match SigAction::read(ctx, act_ptr) {
            Some(a) => Some(a),
            None => return fail(ctx, EFAULT),
        }
    } else {
        None
    };
    let old = match set_action(sig, act) {
        Ok(o) => o,
        Err(e) => return fail(ctx, e),
    };
    if let Some(a) = act {
        log::debug!("[signal] sigaction({sig}, handler={:#x} flags={:#x})", a.handler, a.flags);
    }
    if old_ptr != 0 && !old.write(ctx, old_ptr) {
        return fail(ctx, EFAULT);
    }
    0
}

/// raise(sig)
fn cb_raise(ctx: &mut CpuContext) -> HleResult {
    let sig = ctx.rdi() as i32;
    if !(0..=SIG_MAX).contains(&sig) {
        return fail(ctx, EINVAL);
    }
    if sig != 0 {
        send_to_self(sig);
    }
    0
}

/// kill(pid, sig): the only process is the guest itself, so its pid, 0 (its
/// process group) and minus its pid reach it; every other target, -1
/// included (it skips the caller), does not exist.
fn cb_kill(ctx: &mut CpuContext) -> HleResult {
    let (pid, sig) = (ctx.rdi() as i32, ctx.rsi() as i32);
    if !(0..=SIG_MAX).contains(&sig) {
        return fail(ctx, EINVAL);
    }
    let me = std::process::id() as i32;
    if pid != me && pid != 0 && pid != -me {
        return fail(ctx, ESRCH);
    }
    if sig != 0 {
        send_to_self(sig);
    }
    0
}

const EAGAIN: i32 = 35;

/// The `siginfo_t` of a signal the process sent itself.
fn user_siginfo(sig: i32) -> [u8; SIGINFO_SIZE] {
    let mut info = [0u8; SIGINFO_SIZE];
    info[0..4].copy_from_slice(&sig.to_le_bytes());
    info[8..12].copy_from_slice(&SI_USER.to_le_bytes());
    info[12..16].copy_from_slice(&(std::process::id() as i32).to_le_bytes());
    info
}

/// Take the lowest pending signal of the calling thread that is in `set`.
fn take_pending(set: &SigSet) -> Option<i32> {
    PENDING.with(|p| {
        let mut v = p.get();
        let sig = (1..=SIG_MAX).find(|&s| has(&v, s) && has(set, s))?;
        del(&mut v, sig);
        p.set(v);
        Some(sig)
    })
}

/// sigtimedwait(set*, info*, timeout*) -> the signal taken, or -1 with
/// errno: a signal of `set` pending on the calling thread is removed and
/// returned (its siginfo stored at `info`) without running its handler;
/// with none, the call waits up to `timeout` (NULL: without limit) and
/// fails with EAGAIN. Signals reach a thread only when it raises them
/// itself, so nothing new arrives during the wait. A NULL set is EFAULT,
/// an invalid timespec EINVAL.
fn cb_sigtimedwait(ctx: &mut CpuContext) -> HleResult {
    let (set_ptr, info, timeout) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let Some(set) = read_set(ctx, set_ptr) else { return fail(ctx, EFAULT) };
    let deadline = if timeout == 0 {
        None
    } else {
        let (Some(sec), Some(nsec)) = (ctx.read_u64(timeout), ctx.read_u64(timeout + 8)) else { return fail(ctx, EFAULT) };
        let (sec, nsec) = (sec as i64, nsec as i64);
        if sec < 0 || !(0..1_000_000_000).contains(&nsec) {
            return fail(ctx, EINVAL);
        }
        std::time::Instant::now().checked_add(std::time::Duration::new(sec as u64, nsec as u32))
    };
    loop {
        if let Some(sig) = take_pending(&set) {
            if info != 0 && !ctx.write_bytes(info, &user_siginfo(sig)) {
                return fail(ctx, EFAULT);
            }
            return sig as HleResult;
        }
        // No deadline: no timeout, or one too large for the clock.
        let left = match deadline {
            Some(d) => match d.saturating_duration_since(std::time::Instant::now()) {
                l if l.is_zero() => return fail(ctx, EAGAIN),
                l => l,
            },
            None => super::SLICE,
        };
        super::sleep(left.min(super::SLICE));
    }
}

/// FreeBSD ENOTSUP (EOPNOTSUPP).
const ENOTSUP: i32 = 45;

/// pthread_kill(thread, sig) -> 0 or an error number (errno untouched), as
/// FreeBSD's: EINVAL for a signal outside 0..=128, ESRCH for a NULL or
/// unknown thread; signal 0 only checks the thread. A signal to the calling
/// thread is raised on it. Signals for another thread are not delivered:
/// POSIX handlers run only synchronously on the thread that sends them, so
/// that is reported as ENOTSUP instead of pretending it was sent.
fn cb_pthread_kill(ctx: &mut CpuContext) -> HleResult {
    let (thread, sig) = (ctx.rdi(), ctx.rsi() as i32);
    if !(0..=SIG_MAX).contains(&sig) {
        return EINVAL as HleResult;
    }
    // A thread that has ended is ESRCH, also for signal 0 (measured on a
    // PS5), until it is joined or detached.
    if thread == 0 || !crate::pthread::thread_exists(thread) || (thread != crate::pthread::current_handle() && !crate::pthread::thread_is_live(thread)) {
        return ESRCH as HleResult;
    }
    if sig == 0 {
        return 0;
    }
    if thread != crate::pthread::current_handle() {
        log::error!("[signal] pthread_kill({thread:#x}, {sig}): signals to another thread are not delivered");
        return ENOTSUP as HleResult;
    }
    send_to_self(sig);
    0
}

#[cfg(test)]
pub(super) fn tests_pthread_kill(ctx: &mut CpuContext) -> HleResult {
    cb_pthread_kill(ctx)
}

/// sigaltstack(ss*, oss*): `stack_t` is { sp u64, size u64, flags i32 }.
/// Handlers never run on the alternate stack, so the thread is never on it
/// (no SS_ONSTACK, no EPERM).
fn cb_sigaltstack(ctx: &mut CpuContext) -> HleResult {
    let (ss, oss) = (ctx.rdi(), ctx.rsi());
    let (sp, size, enabled) = ALT_STACK.with(|a| a.get());
    let mut replacement = (sp, size, enabled);
    if ss != 0 {
        let (Some(nsp), Some(nsize), Some(flags)) = (ctx.read_u64(ss), ctx.read_u64(ss + 8), ctx.read_u32(ss + 16)) else {
            return fail(ctx, EFAULT);
        };
        let flags = flags as i32;
        if flags & !SS_DISABLE != 0 {
            return fail(ctx, EINVAL);
        }
        if flags & SS_DISABLE != 0 {
            replacement.2 = false;
        } else {
            if nsize < MINSIGSTKSZ {
                return fail(ctx, ENOMEM);
            }
            replacement = (nsp, nsize, true);
        }
    }
    if oss != 0 {
        let flags = if enabled { 0 } else { SS_DISABLE };
        if !(ctx.write_u64(oss, sp) && ctx.write_u64(oss + 8, size) && ctx.write_u32(oss + 16, flags as u32)) {
            return fail(ctx, EFAULT);
        }
    }
    ALT_STACK.with(|a| a.set(replacement));
    0
}

// ---------------------------------------------------------------------------
// sigsetjmp / siglongjmp
// ---------------------------------------------------------------------------
//
// `sigjmp_buf`: rip @0, rbx @8, rsp @16 (pointing at the return address),
// rbp @24, r12-r15 @32..56, x87 control word @64, MXCSR @68, saved mask
// @72 (16 bytes), save-mask flag @88.

const JB_RSP: u64 = 16;
const JB_FPCW: u64 = 64;
const JB_MXCSR: u64 = 68;
const JB_MASK: u64 = 72;
const JB_SAVEMASK: u64 = 88;

/// sigsetjmp(env, savemask): record the caller's callee-saved state; 0.
fn cb_sigsetjmp(ctx: &mut CpuContext) -> HleResult {
    let (env, savemask) = (ctx.rdi(), ctx.rsi() as u32);
    let Some(view) = ps5cpu::signal::current_import_interrupt() else {
        log::error!("[signal] sigsetjmp outside an import");
        return 0;
    };
    let m = view.ucontext().uc_mcontext;
    // The view presents the guest as returned from the call; the buffer
    // keeps the stack pointer at the call's return address.
    let fp = &m.mc_fpstate;
    let fpcw = (fp[0] & 0xFFFF) as u32;
    let mxcsr = (fp[3] & 0xFFFF_FFFF) as u32;
    let regs = [m.mc_rip, m.mc_rbx, m.mc_rsp - 8, m.mc_rbp, m.mc_r12, m.mc_r13, m.mc_r14, m.mc_r15];
    let mut ok = regs.iter().enumerate().all(|(i, &v)| ctx.write_u64(env + i as u64 * 8, v));
    ok &= ctx.write_u32(env + JB_FPCW, fpcw) && ctx.write_u32(env + JB_MXCSR, mxcsr);
    ok &= ctx.write_u32(env + JB_SAVEMASK, savemask);
    if savemask != 0 {
        ok &= write_set(ctx, env + JB_MASK, &blocked_mask());
    }
    if !ok {
        log::error!("[signal] sigsetjmp: cannot write the buffer at {env:#x}");
    }
    0
}

/// Restore the mask a `sigsetjmp(env, 1)` saved. Returns false when the
/// buffer cannot be read.
fn longjmp_restore_mask(ctx: &CpuContext, env: u64) -> bool {
    let Some(save) = ctx.read_u32(env + JB_SAVEMASK) else { return false };
    if save != 0 {
        let Some(mask) = read_set(ctx, env + JB_MASK) else { return false };
        change_mask(SIG_SETMASK, Some(mask));
    }
    true
}

std::arch::global_asm!(
    // Entered as the tail of the siglongjmp import with the caller's
    // registers: rdi = env, esi = val.
    ".globl ps5hle_siglongjmp",
    "ps5hle_siglongjmp:",
    "mov rdx, rdi",
    "mov eax, esi",
    "mov rcx, [rdx]",
    "mov rbx, [rdx + 8]",
    "mov rsp, [rdx + 16]",
    "mov rbp, [rdx + 24]",
    "mov r12, [rdx + 32]",
    "mov r13, [rdx + 40]",
    "mov r14, [rdx + 48]",
    "mov r15, [rdx + 56]",
    "fldcw [rdx + 64]",
    "test eax, eax",
    "jnz 2f",
    "inc eax",
    "2:",
    "mov [rsp], rcx",
    "ret",
);

extern "C" {
    fn ps5hle_siglongjmp();
}

/// siglongjmp(env, val): restore the saved mask (when sigsetjmp saved it),
/// then resume at the sigsetjmp call with its registers and `val` (1 for 0).
fn cb_siglongjmp(ctx: &mut CpuContext) -> HleResult {
    let env = ctx.rdi();
    if env == 0 || !longjmp_restore_mask(ctx, env) || ctx.read_u64(env + JB_RSP).is_none() {
        log::error!("[signal] siglongjmp: unreadable buffer {env:#x}");
        ps5vm::shutdown::exit(128 + 11)
    }
    ps5cpu::exec::set_trap_tail_call(ps5hle_siglongjmp as *const () as u64);
    0
}

// ---------------------------------------------------------------------------
// setcontext
// ---------------------------------------------------------------------------

/// FreeBSD ucontext_t: the signal mask at 0, mcontext_t (0x320 bytes) at
/// 0x10; in mcontext_t: rflags 0xB0, flags 0x90, len 0xC8, fpstate (the
/// FXSAVE image) 0xE0.
const UC_MCONTEXT: u64 = 0x10;
const MC_FLAGS: u64 = 0x90;
const MC_LEN: u64 = 0xC8;
const MC_FPSTATE: u64 = 0xE0;
const MC_SIZE: u64 = 0x320;
const MC_FLAG_MASK: u32 = 0x7;
const FXSAVE_MXCSR: u64 = 24;

std::arch::global_asm!(
    // Entered as the tail of the setcontext import: rdi = ucontext. An
    // iretq frame built on the abandoned stack loads rip, rflags (user
    // bits only) and rsp together; every other register comes from the
    // context.
    ".globl ps5hle_setcontext",
    "ps5hle_setcontext:",
    "lea r12, [rdi + 0x10]",
    "xor eax, eax",
    "mov ax, ss",
    "push rax",
    "push qword ptr [r12 + 0xB8]",
    "pushfq",
    "pop rax",
    "and rax, -0x254dd6",
    "mov rcx, [r12 + 0xB0]",
    "and rcx, 0x254dd5",
    "or rax, rcx",
    "push rax",
    "xor eax, eax",
    "mov ax, cs",
    "push rax",
    "push qword ptr [r12 + 0xA0]",
    "fxrstor64 [r12 + 0xE0]",
    "mov rax, [r12 + 0x38]",
    "mov rbx, [r12 + 0x40]",
    "mov rcx, [r12 + 0x20]",
    "mov rdx, [r12 + 0x18]",
    "mov rsi, [r12 + 0x10]",
    "mov rdi, [r12 + 0x08]",
    "mov rbp, [r12 + 0x48]",
    "mov r8, [r12 + 0x28]",
    "mov r9, [r12 + 0x30]",
    "mov r10, [r12 + 0x50]",
    "mov r11, [r12 + 0x58]",
    "mov r13, [r12 + 0x68]",
    "mov r14, [r12 + 0x70]",
    "mov r15, [r12 + 0x78]",
    "mov r12, [r12 + 0x60]",
    "iretq",
);

extern "C" {
    fn ps5hle_setcontext();
}

/// MXCSR bits the host's FXRSTOR accepts.
fn host_mxcsr_mask() -> u32 {
    #[repr(C, align(16))]
    struct Image([u8; 512]);
    let mut image = Image([0; 512]);
    // SAFETY: a 16-byte aligned 512-byte buffer.
    unsafe { std::arch::x86_64::_fxsave64(image.0.as_mut_ptr()) };
    let mask = u32::from_le_bytes(image.0[28..32].try_into().unwrap());
    if mask != 0 { mask } else { 0xFFBF }
}

/// Why `uc` cannot be resumed (EINVAL), or None.
fn setcontext_error(ctx: &CpuContext, uc: u64) -> Option<&'static str> {
    if uc == 0 {
        return Some("null context");
    }
    let mc = uc + UC_MCONTEXT;
    let (Some(len), Some(flags), Some(mxcsr)) = (ctx.read_u64(mc + MC_LEN), ctx.read_u32(mc + MC_FLAGS), ctx.read_u32(mc + MC_FPSTATE + FXSAVE_MXCSR)) else {
        return Some("unreadable context");
    };
    if len != MC_SIZE || flags & !MC_FLAG_MASK != 0 {
        return Some("bad mcontext length or flags");
    }
    if (mc + MC_FPSTATE) % 16 != 0 {
        return Some("FPU state not 16-byte aligned");
    }
    if mxcsr & !host_mxcsr_mask() != 0 {
        return Some("reserved MXCSR bits set");
    }
    None
}

/// setcontext(const ucontext_t*): installs the context's signal mask and
/// resumes at its registers; returns -1 with EINVAL only for a context it
/// cannot resume.
fn cb_setcontext(ctx: &mut CpuContext) -> HleResult {
    let uc = ctx.rdi();
    if let Some(why) = setcontext_error(ctx, uc) {
        log::warn!("[signal] setcontext({uc:#x}): {why}");
        return fail(ctx, EINVAL);
    }
    let Some(mask) = read_set(ctx, uc) else { return fail(ctx, EINVAL) };
    change_mask(SIG_SETMASK, Some(mask));
    ps5cpu::exec::set_trap_tail_call(ps5hle_setcontext as *const () as u64);
    0
}

// ---------------------------------------------------------------------------
// alarm
// ---------------------------------------------------------------------------

const SIGALRM: i32 = 14;

/// The process alarm: its deadline, watched by one host thread started by
/// the first alarm() and kept for the life of the process.
struct AlarmState {
    deadline: Option<std::time::Instant>,
    /// Reload period of ITIMER_REAL (zero: one shot).
    interval: std::time::Duration,
    thread_started: bool,
}

static ALARM: Mutex<AlarmState> =
    Mutex::new(AlarmState { deadline: None, interval: std::time::Duration::ZERO, thread_started: false });
static ALARM_CHANGED: std::sync::Condvar = std::sync::Condvar::new();

fn alarm_state() -> std::sync::MutexGuard<'static, AlarmState> {
    ALARM.lock().unwrap_or_else(|e| e.into_inner())
}

/// SIGALRM of an expired alarm: ignored, or the default action ends the
/// process. A guest handler cannot run on the alarm thread.
fn expire_alarm() {
    match action(SIGALRM).handler {
        SIG_IGN => {}
        SIG_DFL => {
            log::error!("[signal] SIGALRM: the alarm expired with the default disposition; the process terminates");
            ps5vm::shutdown::request(128 + SIGALRM);
        }
        handler => log::error!("[signal] alarm: delivering SIGALRM to the guest handler {handler:#x} is not implemented"),
    }
}

fn run_alarms() {
    let mut s = alarm_state();
    loop {
        let Some(deadline) = s.deadline else {
            s = ALARM_CHANGED.wait(s).unwrap_or_else(|e| e.into_inner());
            continue;
        };
        let now = std::time::Instant::now();
        if now < deadline {
            s = ALARM_CHANGED.wait_timeout(s, deadline - now).unwrap_or_else(|e| e.into_inner()).0;
            continue;
        }
        // A periodic timer reloads from its interval.
        s.deadline = (!s.interval.is_zero()).then(|| deadline + s.interval);
        drop(s);
        expire_alarm();
        s = alarm_state();
    }
}

/// Arm ITIMER_REAL: `value` until the first expiry (None: disarm) and the
/// reload `interval`; the alarm thread starts on first use.
fn arm_alarm(s: &mut AlarmState, value: Option<std::time::Duration>, interval: std::time::Duration) {
    let now = std::time::Instant::now();
    s.deadline = value.map(|v| now + v);
    s.interval = if value.is_some() { interval } else { std::time::Duration::ZERO };
    if !s.thread_started {
        match std::thread::Builder::new().name("guest-alarm".into()).spawn(run_alarms) {
            Ok(_) => s.thread_started = true,
            Err(e) => log::error!("[signal] alarm: cannot start the alarm thread: {e}"),
        }
    }
    ALARM_CHANGED.notify_all();
}

/// alarm(seconds) -> the whole seconds (rounded up) the previous alarm had
/// left; 0 cancels the alarm.
fn alarm(seconds: u32) -> u32 {
    let mut s = alarm_state();
    if seconds > ITIMER_MAX_SECONDS as u32 {
        // Beyond the itimer range: EINVAL, the alarm left as it is.
        return u32::MAX;
    }
    let now = std::time::Instant::now();
    let remaining = s.deadline.filter(|&d| d > now).map_or(0, |d| (d - now).as_micros().div_ceil(1_000_000) as u32);
    arm_alarm(&mut s, (seconds != 0).then(|| std::time::Duration::from_secs(seconds as u64)), std::time::Duration::ZERO);
    remaining
}

// ---------------------------------------------------------------------------
// Interval timers
// ---------------------------------------------------------------------------

const ITIMER_REAL: i32 = 0;
const ITIMER_PROF: i32 = 2;
/// Intervals below this are rounded up by the kernel to a clock tick whose
/// length is not known.
const MIN_EXACT_INTERVAL_US: i64 = 1000;

/// Largest second count an interval timer takes (FreeBSD itimerfix:
/// 100000000 s is the documented bound; the console refuses above
/// 0x3FFFFFFF).
const ITIMER_MAX_SECONDS: i64 = 0x3FFF_FFFF;

/// Guest `struct timeval` (sec, usec as two i64).
fn read_timeval(ctx: &CpuContext, addr: u64) -> Option<(i64, i64)> {
    Some((ctx.read_u64(addr)? as i64, ctx.read_u64(addr + 8)? as i64))
}

fn timeval_valid((sec, usec): (i64, i64)) -> bool {
    sec >= 0 && (0..1_000_000).contains(&usec)
}

fn timeval_duration((sec, usec): (i64, i64)) -> std::time::Duration {
    std::time::Duration::from_secs(sec as u64) + std::time::Duration::from_micros(usec as u64)
}

fn duration_timeval(d: std::time::Duration) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&(d.as_secs() as i64).to_le_bytes());
    b[8..].copy_from_slice(&(d.subsec_micros() as i64).to_le_bytes());
    b
}

/// The current ITIMER_REAL as an `itimerval` (interval, value).
fn real_itimer() -> [u8; 32] {
    let s = alarm_state();
    let now = std::time::Instant::now();
    let value = s.deadline.filter(|&d| d > now).map_or(std::time::Duration::ZERO, |d| d - now);
    let mut out = [0u8; 32];
    out[..16].copy_from_slice(&duration_timeval(s.interval));
    out[16..].copy_from_slice(&duration_timeval(value));
    out
}

/// getitimer(which, itimerval*): ITIMER_REAL as armed; ITIMER_VIRTUAL and
/// ITIMER_PROF read as disarmed.
fn cb_getitimer(ctx: &mut CpuContext) -> HleResult {
    let (which, out) = (ctx.rdi() as i32, ctx.rsi());
    if !(ITIMER_REAL..=ITIMER_PROF).contains(&which) {
        return fail(ctx, EINVAL);
    }
    if out == 0 {
        return fail(ctx, EFAULT);
    }
    let value = if which == ITIMER_REAL { real_itimer() } else { [0u8; 32] };
    if !ctx.write_bytes(out, &value) {
        return fail(ctx, EFAULT);
    }
    0
}

/// setitimer(which, const itimerval* value, itimerval* old): ITIMER_REAL
/// arms the process alarm (SIGALRM); a zero value disarms. Arming
/// ITIMER_VIRTUAL or ITIMER_PROF needs process CPU-time accounting and is
/// refused; an interval below 1 ms is refused too (its rounding is
/// unknown).
fn cb_setitimer(ctx: &mut CpuContext) -> HleResult {
    let (which, value, old) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    if !(ITIMER_REAL..=ITIMER_PROF).contains(&which) {
        return fail(ctx, EINVAL);
    }
    if value == 0 && old == 0 {
        return fail(ctx, EFAULT);
    }
    let mut requested = ((0i64, 0i64), (0i64, 0i64));
    if value != 0 {
        let (Some(interval), Some(val)) = (read_timeval(ctx, value), read_timeval(ctx, value + 16)) else {
            return fail(ctx, EFAULT);
        };
        if !timeval_valid(val) {
            return fail(ctx, EINVAL);
        }
        let set = val != (0, 0);
        let interval = if set { interval } else { (0, 0) };
        if !timeval_valid(interval) || val.0 > ITIMER_MAX_SECONDS || interval.0 > ITIMER_MAX_SECONDS {
            return fail(ctx, EINVAL);
        }
        requested = (interval, val);
    }
    let armed = requested.1 != (0, 0);
    if which != ITIMER_REAL {
        if armed {
            log::error!("[signal] setitimer({which}): needs process CPU time accounting, which is not implemented");
            return fail(ctx, EINVAL);
        }
        if old != 0 && !ctx.write_bytes(old, &[0u8; 32]) {
            return fail(ctx, EFAULT);
        }
        return 0;
    }
    let (isec, iusec) = requested.0;
    if isec == 0 && iusec != 0 && iusec < MIN_EXACT_INTERVAL_US {
        log::error!("[signal] setitimer: an interval of {iusec} us is below 1 ms, where the kernel's rounding is not known");
        return fail(ctx, EINVAL);
    }
    let previous = real_itimer();
    if value != 0 {
        let mut s = alarm_state();
        arm_alarm(&mut s, armed.then(|| timeval_duration(requested.1)), timeval_duration(requested.0));
    }
    if old != 0 && !ctx.write_bytes(old, &previous) {
        return fail(ctx, EFAULT);
    }
    0
}

// ---------------------------------------------------------------------------
// Waiting for signals
// ---------------------------------------------------------------------------

const EINTR: i32 = 4;

/// Sleep until a signal pending on the calling thread is deliverable
/// (unblocked), then deliver it. Signals reach a thread only when it
/// raises them itself, so without one this sleeps until shutdown.
fn wait_deliverable() {
    loop {
        let pending = PENDING.with(|p| p.get());
        let mask = blocked_mask();
        if (1..=SIG_MAX).any(|s| has(&pending, s) && !has(&mask, s)) {
            deliver_unblocked();
            return;
        }
        if ps5vm::shutdown::requested() {
            ps5vm::shutdown::park_if_requested();
            return;
        }
        super::sleep(super::SLICE);
    }
}

/// pause(): waits for a signal; always -1 with EINTR.
fn cb_pause(ctx: &mut CpuContext) -> HleResult {
    wait_deliverable();
    fail(ctx, EINTR)
}

/// sigsuspend(const sigset_t*): waits with the given mask, restores the
/// old one; always -1 with EINTR.
fn cb_sigsuspend(ctx: &mut CpuContext) -> HleResult {
    let Some(set) = read_set(ctx, ctx.rdi()) else {
        log::error!("[signal] sigsuspend: null mask");
        return fail(ctx, EFAULT);
    };
    let previous = blocked_mask();
    BLOCKED.with(|b| b.set(cant_mask(set)));
    wait_deliverable();
    change_mask(SIG_SETMASK, Some(previous));
    fail(ctx, EINTR)
}

/// sigpending(sigset_t*): the calling thread's pending signals that are
/// blocked.
fn cb_sigpending(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    let pending = PENDING.with(|p| p.get());
    let mask = blocked_mask();
    let set: SigSet = std::array::from_fn(|i| pending[i] & mask[i]);
    if out == 0 || !write_set(ctx, out, &set) {
        return fail(ctx, EFAULT);
    }
    0
}

fn set_is_empty(set: &SigSet) -> bool {
    set.iter().all(|&w| w == 0)
}

/// Take a pending signal of `set`, waiting without limit.
fn wait_in_set(set: &SigSet) -> i32 {
    loop {
        if let Some(sig) = take_pending(set) {
            return sig;
        }
        super::sleep(super::SLICE);
    }
}

/// sigwait(const sigset_t*, int* sig) -> 0 or an error number (errno
/// untouched): EINVAL for a NULL or empty set or a NULL output.
fn cb_sigwait(ctx: &mut CpuContext) -> HleResult {
    let (set_ptr, out) = (ctx.rdi(), ctx.rsi());
    let Some(set) = read_set(ctx, set_ptr).filter(|s| !set_is_empty(s)) else { return EINVAL as HleResult };
    if out == 0 {
        return EINVAL as HleResult;
    }
    let sig = wait_in_set(&set);
    if ctx.write_u32(out, sig as u32) { 0 } else { EFAULT as HleResult }
}

/// sigwaitinfo(const sigset_t*, siginfo_t*) -> the signal taken.
fn cb_sigwaitinfo(ctx: &mut CpuContext) -> HleResult {
    let (set_ptr, info) = (ctx.rdi(), ctx.rsi());
    let Some(set) = read_set(ctx, set_ptr).filter(|s| !set_is_empty(s)) else { return fail(ctx, EINVAL) };
    let sig = wait_in_set(&set);
    if info != 0 && !ctx.write_bytes(info, &user_siginfo(sig)) {
        return fail(ctx, EFAULT);
    }
    sig as HleResult
}

fn cb_alarm(ctx: &mut CpuContext) -> HleResult {
    let seconds = ctx.rdi() as u32;
    let r = alarm(seconds);
    if seconds > ITIMER_MAX_SECONDS as u32 {
        crate::kernel::set_errno(ctx, EINVAL);
    }
    r as HleResult
}

/// libc exports served when no loaded libc module has them.
pub(super) const LIBC_FALLBACK_EXPORTS: &[(&str, &str, HleCallback)] = &[("DQXJraCc1rA", "alarm", cb_alarm)];

pub(super) const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("+F7C-hdk7+E", "sigemptyset", cb_sigemptyset),
    ("VkTAsrZDcJ0", "sigfillset", cb_sigfillset),
    ("JUimFtKe0Kc", "sigaddset", cb_sigaddset),
    ("Nd-u09VFSCA", "sigdelset", cb_sigdelset),
    ("JnNl8Xr-z4Y", "sigismember", cb_sigismember),
    ("6xVpy0Fdq+I", "_sigprocmask", cb_sigprocmask),
    ("aPcyptbOiZs", "sigprocmask", cb_sigprocmask),
    ("JZKw5+Wrnaw", "pthread_sigmask", cb_pthread_sigmask),
    ("VADc3MNQ3cM", "signal", cb_signal),
    ("KiJEPEWRyUY", "sigaction", cb_sigaction),
    ("UDCI-WazohQ", "_sigaction", cb_sigaction),
    ("0t0-MxQNwK4", "raise", cb_raise),
    ("W0xkN0+ZkCE", "kill", cb_kill),
    ("yH-uQW3LbX0", "pthread_kill", cb_pthread_kill),
    ("D2-dIoJ0ZtE", "sigtimedwait", cb_sigtimedwait),
    ("2VRU7xiqLO8", "setcontext", cb_setcontext),
    ("J1H1QRCJf2Q", "getitimer", cb_getitimer),
    ("hPWDGx8ioXQ", "setitimer", cb_setitimer),
    ("EDyhWelj7UY", "pause", cb_pause),
    ("KZ-4qlqlpmo", "sigsuspend", cb_sigsuspend),
    ("hpoDTzy9Yy0", "sigpending", cb_sigpending),
    ("mrbHXqK8wkg", "sigwait", cb_sigwait),
    ("Uq5BGthgbl4", "sigwaitinfo", cb_sigwaitinfo),
    ("sHziAegVp74", "sigaltstack", cb_sigaltstack),
    ("aRo9AhFUXcM", "sigsetjmp", cb_sigsetjmp),
    ("pebqbE5ws8s", "siglongjmp", cb_siglongjmp),
];

pub(super) fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
    for &(nid, name, cb) in LIBC_FALLBACK_EXPORTS {
        reg.register_fallback_fn(nid, name, "libSceLibcInternal", cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister;
    use ps5vm::{MemProtect, VirtualMemory};

    struct Mem {
        vm: &'static VirtualMemory,
        base: u64,
    }

    impl Mem {
        fn new(base: u64) -> Self {
            let vm: &'static VirtualMemory = Box::leak(Box::new(VirtualMemory::new()));
            vm.map(base, 0x10000, MemProtect::RW).unwrap();
            Mem { vm, base }
        }
    }

    impl Drop for Mem {
        fn drop(&mut self) {
            let _ = self.vm.unmap(self.base, 0x10000);
        }
    }

    fn call(ctx: &mut CpuContext, cb: HleCallback, args: &[u64]) -> HleResult {
        let regs = [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx];
        for (r, v) in regs.iter().zip(args.iter().chain(std::iter::repeat(&0))) {
            ctx.set(*r, *v);
        }
        cb(ctx)
    }

    fn put_set(ctx: &CpuContext, addr: u64, set: SigSet) {
        assert!(write_set(ctx, addr, &set));
    }

    /// alarm returns the rounded-up seconds the previous alarm had left; an
    /// expired alarm with SIGALRM ignored leaves nothing behind. (SIGALRM is
    /// ignored first, so a failing step cannot end the test process.)
    #[test]
    fn alarm_reports_the_time_left() {
        let _alarm = ALARM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let old = set_action(SIGALRM, Some(SigAction { handler: SIG_IGN, flags: 0, mask: EMPTY })).unwrap();
        assert_eq!(alarm(0), 0);
        assert_eq!(alarm(5), 0);
        assert_eq!(alarm(3), 5);
        assert_eq!(alarm(0), 3);
        assert_eq!(alarm(0), 0);
        assert_eq!(alarm(1), 0);
        std::thread::sleep(std::time::Duration::from_millis(1500));
        assert_eq!(alarm(0), 0);
        set_action(SIGALRM, Some(old)).unwrap();
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS.iter().chain(LIBC_FALLBACK_EXPORTS) {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn sigset_functions_use_bit_n_minus_one() {
        let m = Mem::new(0x5EC0_0000_0000);
        let mut ctx = CpuContext::new(m.vm);
        let set = m.base;
        put_set(&ctx, set, [0xa5a5_a5a5; 4]);
        assert_eq!(call(&mut ctx, cb_sigemptyset, &[set]), 0);
        assert_eq!(read_set(&ctx, set), Some(EMPTY));
        for (sig, word, bit) in [(1, 0, 0), (32, 0, 31), (33, 1, 0), (64, 1, 31), (65, 2, 0), (128, 3, 31)] {
            assert_eq!(call(&mut ctx, cb_sigaddset, &[set, sig]), 0);
            assert_eq!(read_set(&ctx, set).unwrap()[word], 1 << bit);
            assert_eq!(call(&mut ctx, cb_sigismember, &[set, sig]), 1);
            assert_eq!(call(&mut ctx, cb_sigdelset, &[set, sig]), 0);
            assert_eq!(call(&mut ctx, cb_sigismember, &[set, sig]), 0);
        }
        assert_eq!(read_set(&ctx, set), Some(EMPTY));
        assert_eq!(call(&mut ctx, cb_sigfillset, &[set]), 0);
        assert_eq!(read_set(&ctx, set), Some([u32::MAX; 4]));
        for sig in [0u64, 129, (-1i64) as u64] {
            assert_eq!(call(&mut ctx, cb_sigaddset, &[set, sig]), -1);
            assert_eq!(call(&mut ctx, cb_sigdelset, &[set, sig]), -1);
            assert_eq!(call(&mut ctx, cb_sigismember, &[set, sig]), -1);
        }
        assert_eq!(read_set(&ctx, set), Some([u32::MAX; 4]));
        assert_eq!(call(&mut ctx, cb_sigemptyset, &[0]), -1);
        assert_eq!(call(&mut ctx, cb_sigismember, &[0, 1]), -1);
    }

    #[test]
    fn sigprocmask_and_pthread_sigmask_share_the_thread_mask() {
        let m = Mem::new(0x5EC1_0000_0000);
        let mut ctx = CpuContext::new(m.vm);
        let (set, old) = (m.base, m.base + 0x10);
        // SIGTERM is bit 14, SIGURG bit 15.
        put_set(&ctx, set, [1 << 14, 0, 0, 1 << 31]);
        assert_eq!(call(&mut ctx, cb_sigprocmask, &[SIG_BLOCK as u64, set, 0]), 0);
        assert_eq!(call(&mut ctx, cb_sigprocmask, &[SIG_BLOCK as u64, 0, old]), 0);
        assert_eq!(read_set(&ctx, old), Some([1 << 14, 0, 0, 1 << 31]));
        assert!(has(&blocked_mask(), 15) && !has(&blocked_mask(), 16) && has(&blocked_mask(), 128));
        // pthread_sigmask returns the error number and leaves the old set.
        put_set(&ctx, old, [7; 4]);
        assert_eq!(call(&mut ctx, cb_pthread_sigmask, &[0, set, old]), EINVAL as i64);
        assert_eq!(call(&mut ctx, cb_pthread_sigmask, &[4, set, old]), EINVAL as i64);
        assert_eq!(read_set(&ctx, old), Some([7; 4]));
        // With no set the how is not looked at.
        assert_eq!(call(&mut ctx, cb_pthread_sigmask, &[0, 0, old]), 0);
        assert_eq!(read_set(&ctx, old), Some([1 << 14, 0, 0, 1 << 31]));
        assert_eq!(call(&mut ctx, cb_pthread_sigmask, &[SIG_UNBLOCK as u64, set, old]), 0);
        assert_eq!(blocked_mask(), EMPTY);
        // SIGKILL and SIGSTOP never enter the mask.
        put_set(&ctx, set, [u32::MAX; 4]);
        assert_eq!(call(&mut ctx, cb_sigprocmask, &[SIG_SETMASK as u64, set, 0]), 0);
        assert!(!has(&blocked_mask(), SIGKILL) && !has(&blocked_mask(), SIGSTOP) && has(&blocked_mask(), 1));
        // Another thread has its own mask.
        assert_eq!(std::thread::spawn(blocked_mask).join().unwrap(), EMPTY);
        put_set(&ctx, set, EMPTY);
        assert_eq!(call(&mut ctx, cb_sigprocmask, &[SIG_SETMASK as u64, set, 0]), 0);
        assert_eq!(call(&mut ctx, cb_sigprocmask, &[SIG_SETMASK as u64, 0x10, 0]), -1);
    }

    #[test]
    fn sigaction_records_dispositions_like_freebsd() {
        let m = Mem::new(0x5EC2_0000_0000);
        let mut ctx = CpuContext::new(m.vm);
        let (act, old) = (m.base, m.base + 0x40);
        // Signals 1..=128 only; SIGKILL/SIGSTOP accept only SIG_DFL.
        let term = SigAction { handler: 0x1234_5000, flags: 0, mask: [0x4000, 0, 0, 0x8000_0000] };
        assert!(term.write(&ctx, act));
        // A signal no other test touches.
        let sig = 70u64;
        ctx.write_bytes(old, &[0xa5; 32]);
        assert_eq!(call(&mut ctx, cb_sigaction, &[sig, 0, old]), 0);
        assert_eq!(SigAction::read(&ctx, old), Some(SigAction::default()));
        assert_eq!(call(&mut ctx, cb_sigaction, &[sig, act, old]), 0);
        assert_eq!(SigAction::read(&ctx, old).unwrap().handler, 0);
        assert_eq!(call(&mut ctx, cb_sigaction, &[sig, 0, old]), 0);
        assert_eq!(SigAction::read(&ctx, old), Some(term));
        // signal() shares the table and records SA_RESTART.
        assert_eq!(call(&mut ctx, cb_signal, &[sig, 0x2222_0000]), 0x1234_5000);
        assert_eq!(call(&mut ctx, cb_sigaction, &[sig, 0, old]), 0);
        assert_eq!(SigAction::read(&ctx, old), Some(SigAction { handler: 0x2222_0000, flags: SA_RESTART, mask: EMPTY }));
        assert_eq!(call(&mut ctx, cb_signal, &[sig, SIG_DFL]), 0x2222_0000);
        // SA_SIGINFO handlers are recorded as given, on every signal number.
        let info = SigAction { handler: 0x3333_0000, flags: SA_SIGINFO | SA_NODEFER, mask: EMPTY };
        assert!(info.write(&ctx, act));
        for s in [71u64, 1, 64, 128] {
            let prev = action(s as i32);
            assert_eq!(call(&mut ctx, cb_sigaction, &[s, act, 0]), 0);
            assert_eq!(action(s as i32), info);
            set_action(s as i32, Some(prev)).unwrap();
        }
        for bad in [0u64, 129] {
            assert_eq!(call(&mut ctx, cb_sigaction, &[bad, 0, old]), -1);
            assert_eq!(call(&mut ctx, cb_signal, &[bad, 0]), SIG_ERR as i64);
        }
        assert!(term.write(&ctx, act));
        for fixed in [SIGKILL as u64, SIGSTOP as u64] {
            assert_eq!(call(&mut ctx, cb_sigaction, &[fixed, act, old]), -1);
            assert!(SigAction::default().write(&ctx, act));
            assert_eq!(call(&mut ctx, cb_sigaction, &[fixed, act, old]), 0);
            assert!(term.write(&ctx, act));
        }
        assert_eq!(call(&mut ctx, cb_sigaction, &[sig, 0, 0]), 0);
        assert_eq!(call(&mut ctx, cb_sigaction, &[sig, 0x10, 0]), -1);
    }

    #[test]
    fn blocked_or_ignored_signals_are_not_delivered() {
        let m = Mem::new(0x5EC3_0000_0000);
        let mut ctx = CpuContext::new(m.vm);
        let set = m.base;
        // A guest handler would need guest code: keep the signal blocked
        // and check it stays pending until the handler is changed away.
        let sig = 72;
        set_action(sig, Some(SigAction { handler: 0x4444_0000, flags: 0, mask: EMPTY })).unwrap();
        let mut s = EMPTY;
        add(&mut s, sig);
        put_set(&ctx, set, s);
        assert_eq!(call(&mut ctx, cb_sigprocmask, &[SIG_BLOCK as u64, set, 0]), 0);
        assert_eq!(call(&mut ctx, cb_raise, &[sig as u64]), 0);
        assert!(has(&PENDING.with(|p| p.get()), sig));
        // Ignored now: unblocking discards it.
        set_action(sig, Some(SigAction { handler: SIG_IGN, flags: 0, mask: EMPTY })).unwrap();
        assert_eq!(call(&mut ctx, cb_sigprocmask, &[SIG_UNBLOCK as u64, set, 0]), 0);
        assert!(!has(&PENDING.with(|p| p.get()), sig));
        assert_eq!(call(&mut ctx, cb_raise, &[sig as u64]), 0);
        // An ignored signal raised while blocked is never pending.
        set_action(sig, Some(SigAction { handler: SIG_IGN, flags: 0, mask: EMPTY })).unwrap();
        assert_eq!(call(&mut ctx, cb_sigprocmask, &[SIG_BLOCK as u64, set, 0]), 0);
        assert_eq!(call(&mut ctx, cb_raise, &[sig as u64]), 0);
        assert!(!has(&PENDING.with(|p| p.get()), sig));
        let mut chld = EMPTY;
        add(&mut chld, 20);
        put_set(&ctx, set + 0x40, chld);
        assert_eq!(call(&mut ctx, cb_sigprocmask, &[SIG_BLOCK as u64, set + 0x40, 0]), 0);
        assert_eq!(call(&mut ctx, cb_raise, &[20]), 0);
        assert!(!has(&PENDING.with(|p| p.get()), 20), "SIGCHLD's default action discards it");
        assert_eq!(call(&mut ctx, cb_sigprocmask, &[SIG_UNBLOCK as u64, set + 0x40, 0]), 0);
        assert_eq!(call(&mut ctx, cb_sigprocmask, &[SIG_UNBLOCK as u64, set, 0]), 0);
        // SIGCHLD's default action discards it.
        assert_eq!(call(&mut ctx, cb_raise, &[20]), 0);
        assert_eq!(call(&mut ctx, cb_raise, &[0]), 0);
        assert_eq!(call(&mut ctx, cb_raise, &[129]), -1);
        set_action(sig, Some(SigAction::default())).unwrap();
    }

    #[test]
    fn kill_reaches_only_the_own_process() {
        let m = Mem::new(0x5EC4_0000_0000);
        let mut ctx = CpuContext::new(m.vm);
        let me = std::process::id() as i64;
        for target in [me, 0, -me] {
            assert_eq!(call(&mut ctx, cb_kill, &[target as u64, 0]), 0);
            // SIGURG's default action discards it.
            assert_eq!(call(&mut ctx, cb_kill, &[target as u64, 16]), 0);
        }
        for target in [me + 1, -1, -(me + 1)] {
            assert_eq!(call(&mut ctx, cb_kill, &[target as u64, 0]), -1);
            assert_eq!(call(&mut ctx, cb_kill, &[target as u64, 15]), -1);
        }
        for sig in [-1i64, 129] {
            assert_eq!(call(&mut ctx, cb_kill, &[me as u64, sig as u64]), -1);
            assert_eq!(call(&mut ctx, cb_kill, &[(me + 1) as u64, sig as u64]), -1);
        }
    }

    /// pthread_kill returns error numbers (EINVAL outside 0..=128, ESRCH for
    /// a NULL or unknown thread, reference edges 128 / 129), signal 0 only
    /// checks, a signal to the calling thread is raised on it (kept pending
    /// while blocked), and another thread is ENOTSUP.
    #[test]
    fn pthread_kill_errors_and_self_delivery() {
        std::thread::spawn(|| {
            let m = Mem::new(0x7C49_0000_0000);
            let mut ctx = CpuContext::new(m.vm);
            let (me, _) = crate::pthread::fake_thread("killer", 0x7F);
            let errno = crate::kernel::bind_test_errno(&ctx);
            ctx.write_u32(errno, 777);
            assert_eq!(call(&mut ctx, cb_pthread_kill, &[me, 200]), EINVAL as HleResult);
            assert_eq!(call(&mut ctx, cb_pthread_kill, &[me, (-1i64) as u64]), EINVAL as HleResult);
            assert_eq!(call(&mut ctx, cb_pthread_kill, &[0, 15]), ESRCH as HleResult);
            assert_eq!(call(&mut ctx, cb_pthread_kill, &[0, 128]), ESRCH as HleResult);
            assert_eq!(call(&mut ctx, cb_pthread_kill, &[0, 129]), EINVAL as HleResult);
            assert_eq!(call(&mut ctx, cb_pthread_kill, &[0x10, 15]), ESRCH as HleResult, "unknown thread");
            assert_eq!(call(&mut ctx, cb_pthread_kill, &[me, 0]), 0);
            // A blocked signal sent to the calling thread stays pending on it.
            let sig = 73;
            set_action(sig, Some(SigAction { handler: 0x4444_0000, flags: 0, mask: EMPTY })).unwrap();
            let mut s = EMPTY;
            add(&mut s, sig);
            put_set(&ctx, m.base, s);
            assert_eq!(call(&mut ctx, cb_sigprocmask, &[SIG_BLOCK as u64, m.base, 0]), 0);
            assert_eq!(call(&mut ctx, cb_pthread_kill, &[me, sig as u64]), 0);
            assert!(has(&PENDING.with(|p| p.get()), sig));
            set_action(sig, Some(SigAction { handler: SIG_IGN, flags: 0, mask: EMPTY })).unwrap();
            assert_eq!(call(&mut ctx, cb_sigprocmask, &[SIG_UNBLOCK as u64, m.base, 0]), 0);
            set_action(sig, Some(SigAction::default())).unwrap();
            // Another live thread.
            let other = std::thread::spawn(|| crate::pthread::fake_thread("target", 0x7F).0).join().unwrap();
            assert_eq!(call(&mut ctx, cb_pthread_kill, &[other, 15]), ENOTSUP as HleResult);
            assert_eq!(call(&mut ctx, cb_pthread_kill, &[other, 0]), 0);
            assert_eq!(ctx.read_u32(errno), Some(777), "errno untouched");
            crate::pthread::forget_fake_thread(other);
            crate::pthread::forget_fake_thread(me);
        })
        .join()
        .unwrap();
    }

    /// sigtimedwait takes a blocked pending signal of its set (siginfo
    /// stored, the handler not run), times out with EAGAIN after the
    /// timeout, and refuses a NULL set (EFAULT) or a bad timespec (EINVAL).
    /// The reference throws instead (VLC drains SIGPIPE with a zero
    /// timeout).
    #[test]
    fn sigtimedwait_takes_a_pending_signal() {
        std::thread::spawn(|| {
            let m = Mem::new(0x7C5D_0000_0000);
            let mut ctx = CpuContext::new(m.vm);
            let errno = crate::kernel::bind_test_errno(&ctx);
            let (set, info, ts) = (m.base, m.base + 0x40, m.base + 0x100);
            let sig = 13; // SIGPIPE
            set_action(sig, Some(SigAction { handler: 0x4444_0000, flags: 0, mask: EMPTY })).unwrap();
            let mut s = EMPTY;
            add(&mut s, sig);
            put_set(&ctx, set, s);
            assert_eq!(call(&mut ctx, cb_sigprocmask, &[SIG_BLOCK as u64, set, 0]), 0);
            assert_eq!(call(&mut ctx, cb_raise, &[sig as u64]), 0);
            assert!(has(&PENDING.with(|p| p.get()), sig));
            ctx.write_u64(ts, 0);
            ctx.write_u64(ts + 8, 0);
            ctx.write_bytes(info, &[0xAA; SIGINFO_SIZE]);
            assert_eq!(call(&mut ctx, cb_sigtimedwait, &[set, info, ts]), sig as HleResult);
            assert_eq!(ctx.read_u32(info), Some(sig as u32));
            assert_eq!(ctx.read_u32(info + 8), Some(SI_USER as u32));
            assert!(!has(&PENDING.with(|p| p.get()), sig), "taken");
            // Nothing pending: a zero timeout fails at once, a short one
            // after waiting it out.
            assert_eq!(call(&mut ctx, cb_sigtimedwait, &[set, 0, ts]), -1);
            assert_eq!(ctx.read_u32(errno), Some(35));
            ctx.write_u64(ts + 8, 30_000_000);
            let start = std::time::Instant::now();
            assert_eq!(call(&mut ctx, cb_sigtimedwait, &[set, 0, ts]), -1);
            assert!(start.elapsed() >= std::time::Duration::from_millis(30));
            // A pending signal outside the set stays pending.
            let other = 30;
            let mut s2 = EMPTY;
            add(&mut s2, other);
            put_set(&ctx, set + 0x10, s2);
            assert_eq!(call(&mut ctx, cb_sigprocmask, &[SIG_BLOCK as u64, set + 0x10, 0]), 0);
            set_action(other, Some(SigAction { handler: 0x4444_0000, flags: 0, mask: EMPTY })).unwrap();
            assert_eq!(call(&mut ctx, cb_raise, &[other as u64]), 0);
            ctx.write_u64(ts + 8, 0);
            assert_eq!(call(&mut ctx, cb_sigtimedwait, &[set, 0, ts]), -1);
            assert!(has(&PENDING.with(|p| p.get()), other));
            assert_eq!(call(&mut ctx, cb_sigtimedwait, &[set + 0x10, 0, ts]), other as HleResult);
            // Errors.
            assert_eq!(call(&mut ctx, cb_sigtimedwait, &[0, 0, ts]), -1);
            assert_eq!(ctx.read_u32(errno), Some(14));
            ctx.write_u64(ts + 8, 1_000_000_000);
            assert_eq!(call(&mut ctx, cb_sigtimedwait, &[set, 0, ts]), -1);
            assert_eq!(ctx.read_u32(errno), Some(22));
            set_action(sig, Some(SigAction::default())).unwrap();
            set_action(other, Some(SigAction::default())).unwrap();
        })
        .join()
        .unwrap();
    }

    #[test]
    fn sigaltstack_validates_and_reports() {
        let m = Mem::new(0x5EC5_0000_0000);
        let mut ctx = CpuContext::new(m.vm);
        let (ss, oss) = (m.base, m.base + 0x40);
        let put = |ctx: &CpuContext, sp: u64, size: u64, flags: u32| {
            ctx.write_u64(ss, sp);
            ctx.write_u64(ss + 8, size);
            ctx.write_u32(ss + 16, flags);
        };
        assert_eq!(call(&mut ctx, cb_sigaltstack, &[0, oss]), 0);
        assert_eq!(ctx.read_u32(oss + 16), Some(SS_DISABLE as u32));
        put(&ctx, 0x1000_0000, MINSIGSTKSZ - 1, 0);
        assert_eq!(call(&mut ctx, cb_sigaltstack, &[ss, 0]), -1);
        // SS_ONSTACK cannot be set.
        put(&ctx, 0x1000_0000, 0x8000, 1);
        assert_eq!(call(&mut ctx, cb_sigaltstack, &[ss, 0]), -1);
        put(&ctx, 0x1000_0000, 0x8000, 0);
        assert_eq!(call(&mut ctx, cb_sigaltstack, &[ss, oss]), 0);
        assert_eq!(ctx.read_u32(oss + 16), Some(SS_DISABLE as u32));
        assert_eq!(call(&mut ctx, cb_sigaltstack, &[0, oss]), 0);
        assert_eq!((ctx.read_u64(oss), ctx.read_u64(oss + 8), ctx.read_u32(oss + 16)), (Some(0x1000_0000), Some(0x8000), Some(0)));
        // Disabling keeps the recorded range.
        put(&ctx, 0, 0, SS_DISABLE as u32);
        assert_eq!(call(&mut ctx, cb_sigaltstack, &[ss, 0]), 0);
        assert_eq!(call(&mut ctx, cb_sigaltstack, &[0, oss]), 0);
        assert_eq!((ctx.read_u64(oss), ctx.read_u32(oss + 16)), (Some(0x1000_0000), Some(SS_DISABLE as u32)));
        assert_eq!(call(&mut ctx, cb_sigaltstack, &[0x10, 0]), -1);
    }

    /// A sleep that may be stopped returns as soon as its condition holds.
    #[test]
    fn stoppable_sleep_returns_early() {
        let t = std::time::Instant::now();
        super::super::sleep_until_unless(t + std::time::Duration::from_secs(5), || true);
        assert!(t.elapsed() < std::time::Duration::from_secs(1));
        let t = std::time::Instant::now();
        super::super::sleep_until_unless(t + std::time::Duration::from_millis(20), || false);
        assert!(t.elapsed() >= std::time::Duration::from_millis(20));
    }

    #[test]
    fn siglongjmp_restores_the_saved_mask() {
        let m = Mem::new(0x5EC6_0000_0000);
        let ctx = CpuContext::new(m.vm);
        let env = m.base;
        let saved: SigSet = [0x20, 0, 0, 0];
        // Mask not saved: the current mask is kept.
        ctx.write_u32(env + JB_SAVEMASK, 0);
        assert!(write_set(&ctx, env + JB_MASK, &saved));
        change_mask(SIG_SETMASK, Some([0x4, 0, 0, 0]));
        assert!(longjmp_restore_mask(&ctx, env));
        assert_eq!(blocked_mask(), [0x4, 0, 0, 0]);
        ctx.write_u32(env + JB_SAVEMASK, 1);
        assert!(longjmp_restore_mask(&ctx, env));
        assert_eq!(blocked_mask(), saved);
        assert!(!longjmp_restore_mask(&ctx, 0x10));
        change_mask(SIG_SETMASK, Some(EMPTY));
    }

    /// The process alarm is shared by the alarm and itimer tests.
    static ALARM_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// getitimer / setitimer: argument errors, ITIMER_REAL armed and read
    /// back with its interval, the old value returned, VIRTUAL/PROF read as
    /// disarmed and refused when armed, sub-millisecond intervals refused.
    #[test]
    fn interval_timers() {
        let _alarm = ALARM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::thread::spawn(|| {
            let m = Mem::new(0x5EC7_0000_0000);
            let mut ctx = CpuContext::new(m.vm);
            let errno = crate::kernel::bind_test_errno(&ctx);
            let old_action = set_action(SIGALRM, Some(SigAction { handler: SIG_IGN, flags: 0, mask: EMPTY })).unwrap();
            let (value, old) = (m.base, m.base + 0x40);
            let set3 = |ctx: &mut CpuContext, which: u64, v: u64, o: u64| {
                for (r, x) in [(CpuRegister::Rdi, which), (CpuRegister::Rsi, v), (CpuRegister::Rdx, o)] {
                    ctx.set(r, x);
                }
                cb_setitimer(ctx)
            };
            assert_eq!(call(&mut ctx, cb_getitimer, &[3, old]), -1);
            assert_eq!(ctx.read_u32(errno), Some(EINVAL as u32));
            assert_eq!(call(&mut ctx, cb_getitimer, &[0, 0]), -1);
            assert_eq!(ctx.read_u32(errno), Some(EFAULT as u32));
            assert_eq!(set3(&mut ctx, 0, 0, 0), -1);
            assert_eq!(ctx.read_u32(errno), Some(EFAULT as u32));
            // interval 2 s, value 5 s
            for (off, v) in [(0u64, 2u64), (8, 0), (16, 5), (24, 0)] {
                ctx.write_u64(value + off, v);
            }
            assert_eq!(set3(&mut ctx, 0, value, 0), 0);
            assert_eq!(call(&mut ctx, cb_getitimer, &[0, old]), 0);
            assert_eq!(ctx.read_u64(old), Some(2));
            assert!(matches!(ctx.read_u64(old + 16), Some(4) | Some(5)));
            // Disarm, getting the old value back.
            ctx.write_bytes(value, &[0u8; 32]);
            assert_eq!(set3(&mut ctx, 0, value, old), 0);
            assert_eq!(ctx.read_u64(old), Some(2));
            assert_eq!(call(&mut ctx, cb_getitimer, &[0, old]), 0);
            assert_eq!(ctx.read_bytes(old, 32).unwrap(), vec![0u8; 32]);
            // ITIMER_PROF reads disarmed, cannot be armed.
            ctx.write_bytes(old, &[0xFF; 32]);
            assert_eq!(call(&mut ctx, cb_getitimer, &[2, old]), 0);
            assert_eq!(ctx.read_bytes(old, 32).unwrap(), vec![0u8; 32]);
            ctx.write_u64(value + 16, 1);
            assert_eq!(set3(&mut ctx, 2, value, 0), -1);
            assert_eq!(ctx.read_u32(errno), Some(EINVAL as u32));
            // A bad value, a sub-millisecond interval.
            ctx.write_u64(value + 24, 1_000_000);
            assert_eq!(set3(&mut ctx, 0, value, 0), -1);
            ctx.write_u64(value + 24, 0);
            ctx.write_u64(value + 8, 500);
            assert_eq!(set3(&mut ctx, 0, value, 0), -1);
            assert_eq!(ctx.read_u32(errno), Some(EINVAL as u32));
            // Seconds beyond the itimer range.
            ctx.write_u64(value + 8, 0);
            ctx.write_u64(value + 16, 0x4000_0000);
            assert_eq!(set3(&mut ctx, 0, value, 0), -1);
            assert_eq!(ctx.read_u32(errno), Some(EINVAL as u32));
            assert_eq!(alarm(0), 0);
            assert_eq!(alarm(5), 0);
            ctx.write_u32(errno, 0);
            assert_eq!(call(&mut ctx, cb_alarm, &[u32::MAX as u64]), u32::MAX as HleResult);
            assert_eq!(ctx.read_u32(errno), Some(EINVAL as u32));
            assert!(matches!(alarm(0), 4 | 5), "the pending alarm is kept");
            set_action(SIGALRM, Some(old_action)).unwrap();
        })
        .join()
        .unwrap();
    }

    /// sigpending reports blocked pending signals; sigwait / sigwaitinfo
    /// take one of the set; sigsuspend lets a blocked pending signal through
    /// and returns EINTR with the mask restored.
    #[test]
    fn waiting_for_signals() {
        std::thread::spawn(|| {
            let m = Mem::new(0x5EC8_0000_0000);
            let mut ctx = CpuContext::new(m.vm);
            let errno = crate::kernel::bind_test_errno(&ctx);
            // SIGCONT: with the default action it stays pending while blocked
            // and is discarded when delivered (an ignored signal never
            // becomes pending).
            const SIGUSR1: i32 = 19;
            let old = set_action(SIGUSR1, Some(SigAction { handler: SIG_DFL, flags: 0, mask: EMPTY })).unwrap();
            let (set, out, info) = (m.base, m.base + 0x20, m.base + 0x40);
            let mut usr1 = EMPTY;
            add(&mut usr1, SIGUSR1);
            change_mask(SIG_BLOCK, Some(usr1));
            assert_eq!(call(&mut ctx, cb_sigpending, &[0]), -1);
            assert_eq!(call(&mut ctx, cb_sigpending, &[out]), 0);
            assert_eq!(read_set(&ctx, out), Some(EMPTY));
            send_to_self(SIGUSR1);
            assert_eq!(call(&mut ctx, cb_sigpending, &[out]), 0);
            assert_eq!(read_set(&ctx, out), Some(usr1));
            write_set(&ctx, set, &EMPTY);
            assert_eq!(call(&mut ctx, cb_sigwait, &[set, out]), EINVAL as HleResult);
            write_set(&ctx, set, &usr1);
            assert_eq!(call(&mut ctx, cb_sigwait, &[set, 0]), EINVAL as HleResult);
            assert_eq!(call(&mut ctx, cb_sigwait, &[set, out]), 0);
            assert_eq!(ctx.read_u32(out), Some(SIGUSR1 as u32));
            send_to_self(SIGUSR1);
            assert_eq!(call(&mut ctx, cb_sigwaitinfo, &[set, info]), SIGUSR1 as HleResult);
            assert_eq!(ctx.read_u32(info), Some(SIGUSR1 as u32));
            write_set(&ctx, set, &EMPTY);
            assert_eq!(call(&mut ctx, cb_sigwaitinfo, &[set, info]), -1);
            assert_eq!(ctx.read_u32(errno), Some(EINVAL as u32));
            // sigsuspend with an empty mask delivers the pending SIGUSR1.
            send_to_self(SIGUSR1);
            assert_eq!(call(&mut ctx, cb_sigsuspend, &[set]), -1);
            assert_eq!(ctx.read_u32(errno), Some(EINTR as u32));
            assert_eq!(blocked_mask(), usr1);
            assert_eq!(call(&mut ctx, cb_sigpending, &[out]), 0);
            assert_eq!(read_set(&ctx, out), Some(EMPTY));
            change_mask(SIG_SETMASK, Some(EMPTY));
            set_action(SIGUSR1, Some(old)).unwrap();
        })
        .join()
        .unwrap();
    }

    /// setcontext refuses a context it cannot resume (the resume itself
    /// runs as an import tail in guest context).
    #[test]
    fn setcontext_checks_the_context() {
        std::thread::spawn(|| {
            let m = Mem::new(0x5EC9_0000_0000);
            let mut ctx = CpuContext::new(m.vm);
            let errno = crate::kernel::bind_test_errno(&ctx);
            let uc = m.base;
            assert_eq!(call(&mut ctx, cb_setcontext, &[0]), -1);
            assert_eq!(ctx.read_u32(errno), Some(EINVAL as u32));
            assert_eq!(setcontext_error(&ctx, uc), Some("bad mcontext length or flags"));
            ctx.write_u64(uc + UC_MCONTEXT + MC_LEN, MC_SIZE);
            ctx.write_u32(uc + UC_MCONTEXT + MC_FLAGS, 8);
            assert_eq!(setcontext_error(&ctx, uc), Some("bad mcontext length or flags"));
            ctx.write_u32(uc + UC_MCONTEXT + MC_FLAGS, 1);
            ctx.write_u32(uc + UC_MCONTEXT + MC_FPSTATE + FXSAVE_MXCSR, 0xFFFF_0000);
            assert_eq!(setcontext_error(&ctx, uc), Some("reserved MXCSR bits set"));
            ctx.write_u32(uc + UC_MCONTEXT + MC_FPSTATE + FXSAVE_MXCSR, 0x1F80);
            assert_eq!(setcontext_error(&ctx, uc), None);
            assert_eq!(setcontext_error(&ctx, uc + 8), Some("bad mcontext length or flags"));
        })
        .join()
        .unwrap();
    }

    #[test]
    fn new_signal_nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }
}
