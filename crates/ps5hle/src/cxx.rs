// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! C++ ABI runtime: __cxa_guard, atexit, TLS, exception handling stubs.
//!
//! These are called during static initialization and are critical for
//! getting past the very first instructions of a C++ game binary.

use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};
use std::sync::Mutex;
use std::collections::HashMap;

const OK: HleResult = 0;

// ---------------------------------------------------------------------------
// __cxa_guard_acquire / __cxa_guard_release / __cxa_guard_abort
//
// Controls one-time initialization of function-local statics. The guard
// variable is a 64-bit value in guest memory. The first caller to get
// acquire==1 runs the initializer; subsequent callers get 0.
//
// Guard word state (the Itanium C++ ABI):
//   bit 0  (0x01) = initialization complete
//   bit 8  (0x0100) = initialization in progress
//   upper bits preserved
// ---------------------------------------------------------------------------

/// Guard word values.
const GUARD_COMPLETE: u64 = 0x0000_0000_0000_0001;
const GUARD_PENDING: u64 = 0x0000_0000_0000_0100;
const GUARD_STATE_MASK: u64 = 0x0000_0000_0000_FFFF;

/// Guards whose initializer is running, with the owning thread handle.
static GUARDS_IN_PROGRESS: Mutex<Option<HashMap<u64, u64>>> = Mutex::new(None);
static GUARD_DONE: std::sync::Condvar = std::sync::Condvar::new();

const EINVAL: HleResult = 0x8002_0016u32 as i32 as i64;
const EFAULT: HleResult = 0x8002_000Eu32 as i32 as i64;

fn write_guard_state(ctx: &CpuContext, guard: u64, state: u64) -> bool {
    match ctx.read_u64(guard) {
        Some(w) => ctx.write_u64(guard, (w & !GUARD_STATE_MASK) | state),
        None => false,
    }
}

/// __cxa_guard_acquire(guard*) -> 1 when the caller must run the initializer.
/// Other threads wait while an initializer runs; a recursive acquire by the
/// owning thread returns 0.
fn cb_cxa_guard_acquire(ctx: &mut CpuContext) -> HleResult {
    let guard = ctx.rdi();
    if guard == 0 {
        return EINVAL;
    }
    let me = crate::pthread::current_handle();
    let mut g = GUARDS_IN_PROGRESS.lock().unwrap();
    loop {
        let Some(word) = ctx.read_u64(guard) else { return EFAULT };
        if word & GUARD_COMPLETE != 0 {
            return 0;
        }
        let map = g.get_or_insert_with(HashMap::new);
        match map.get(&guard) {
            None => {
                if !write_guard_state(ctx, guard, GUARD_PENDING) {
                    return EFAULT;
                }
                map.insert(guard, me);
                return 1;
            }
            Some(&owner) if owner == me => return 0,
            Some(_) => g = crate::signals::wait_cv(&GUARDS_IN_PROGRESS, &GUARD_DONE, g, None).0,
        }
    }
}

fn finish_guard(ctx: &CpuContext, guard: u64, state: u64) -> HleResult {
    if guard == 0 {
        return EINVAL;
    }
    let me = crate::pthread::current_handle();
    let mut g = GUARDS_IN_PROGRESS.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    if map.get(&guard).is_some_and(|&owner| owner != me) {
        return EINVAL;
    }
    if !write_guard_state(ctx, guard, state) {
        return EFAULT;
    }
    map.remove(&guard);
    GUARD_DONE.notify_all();
    0
}

/// __cxa_guard_release(guard*): mark initialized and wake waiters.
fn cb_cxa_guard_release(ctx: &mut CpuContext) -> HleResult {
    finish_guard(ctx, ctx.rdi(), GUARD_COMPLETE)
}

/// __cxa_guard_abort(guard*): the initializer threw; allow a retry.
fn cb_cxa_guard_abort(ctx: &mut CpuContext) -> HleResult {
    finish_guard(ctx, ctx.rdi(), 0)
}

// ---------------------------------------------------------------------------
// atexit / __cxa_atexit / __cxa_finalize
// ---------------------------------------------------------------------------

/// Exit handlers in registration order: (function, argument, dso handle).
/// atexit and __cxa_atexit share one list, as in the C library, so exit
/// runs them all in reverse order of registration; an atexit handler has
/// no dso handle and runs only for `__cxa_finalize(NULL)`.
static CXA_DESTRUCTORS: Mutex<Vec<(u64, u64, u64)>> = Mutex::new(Vec::new());

/// atexit(func): a null function is ignored.
fn cb_atexit(ctx: &mut CpuContext) -> HleResult {
    let func = ctx.rdi();
    if func != 0 {
        CXA_DESTRUCTORS.lock().unwrap().push((func, 0, 0));
    }
    OK
}

/// __cxa_atexit(func, arg, dso)
fn cb_cxa_atexit(ctx: &mut CpuContext) -> HleResult {
    CXA_DESTRUCTORS.lock().unwrap().push((ctx.rdi(), ctx.rsi(), ctx.rdx()));
    OK
}

/// Next destructor for `__cxa_finalize(dso)`, latest registration first; a
/// destructor runs at most once.
fn take_cxa_destructor(dso: u64) -> Option<(u64, u64)> {
    let mut list = CXA_DESTRUCTORS.lock().unwrap();
    list.iter_mut().rev().find(|e| (dso == 0 || e.2 == dso) && e.0 != 0).map(|e| {
        let f = e.0;
        e.0 = 0;
        (f, e.1)
    })
}

/// __cxa_finalize(dso): runs the destructors registered for `dso` (all of
/// them for a null handle), latest first.
fn cb_cxa_finalize(ctx: &mut CpuContext) -> HleResult {
    run_exit_handlers(ctx.rdi());
    OK
}

/// Run the registered exit handlers of `dso` (all of them for 0), latest
/// first. Must be called on a guest thread, from an HLE dispatch; no lock
/// is held while a handler runs, so a handler may register more.
pub(crate) fn run_exit_handlers(dso: u64) {
    while let Some((func, obj)) = take_cxa_destructor(dso) {
        ps5cpu::exec::call_guest_nested(func, [obj, 0, 0, 0, 0, 0]);
    }
}

// ---------------------------------------------------------------------------
// std::_Execute_once(once_flag&, int (*)(void*, void*, void**), void*)
// ---------------------------------------------------------------------------

const ONCE_INIT: u32 = 0;
const ONCE_DONE: u32 = 1;
const ONCE_RUNNING: u32 = 2;

static ONCE_LOCK: Mutex<()> = Mutex::new(());
static ONCE_CV: std::sync::Condvar = std::sync::Condvar::new();

/// Returns 1 when already done; otherwise runs `func(nullptr, arg, &ctx)`
/// and returns its result (non-zero marks the flag done, zero resets it).
fn cb_std_execute_once(ctx: &mut CpuContext) -> HleResult {
    let (flag, func, arg) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if flag == 0 || func == 0 {
        return 0;
    }
    {
        let mut g = ONCE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        while ctx.read_u32(flag) == Some(ONCE_RUNNING) {
            g = crate::signals::wait_cv(&ONCE_LOCK, &ONCE_CV, g, None).0;
        }
        if ctx.read_u32(flag) == Some(ONCE_DONE) {
            return 1;
        }
        ctx.write_u32(flag, ONCE_RUNNING);
    }
    let context = crate::pthread::alloc_guest_block(ctx, 8).unwrap_or(0);
    if context != 0 {
        ctx.write_u64(context, 0);
    }
    let result = ps5cpu::exec::call_guest_nested(func, [0, arg, context, 0, 0, 0]) as i32;
    {
        let _g = ONCE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        ctx.write_u32(flag, if result != 0 { ONCE_DONE } else { ONCE_INIT });
    }
    ONCE_CV.notify_all();
    result as HleResult
}

// ---------------------------------------------------------------------------
// _init_env — early libc environment initialization
//
// The PS5 entry stub calls _init_env(entry_params) with a pointer to the
// process argument block:
//   +0x00: argc (i32)
//   +0x04: pad
//   +0x08: argv[0]
//   +0x10: argv[1]
//   ...
// We store argc/argv/envp into guest globals so libc's __argc/__argv/
// environ accessors return sane values.
// ---------------------------------------------------------------------------

/// Guest globals for the C runtime environment. We allocate a small block in
/// the HLE data region and expose it through the standard accessor NIDs.
static ENV_BLOCK: Mutex<u64> = Mutex::new(0);

/// argc / argv recorded by `_init_env` (read by libkernel getargc/getargv).
static ARGC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ARGV: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(argc, argv)` of the process as passed to `_init_env`.
pub(crate) fn args() -> (u64, u64) {
    use std::sync::atomic::Ordering::Acquire;
    (ARGC.load(Acquire), ARGV.load(Acquire))
}

fn cb_init_env(ctx: &mut CpuContext) -> HleResult {
    use std::sync::atomic::Ordering::Release;
    let params = ctx.rdi();
    if params == 0 {
        ARGC.store(0, Release);
        ARGV.store(0, Release);
        return OK;
    }
    // Read argc and argv pointer from the entry-params block.
    let argc = ctx.read_u32(params).unwrap_or(0);
    let argv_ptr = params + 8; // argv[] starts at +8 in the entry block
    ARGC.store(argc as u64, Release);
    ARGV.store(argv_ptr, Release);
    log::info!(
        "[cxx] _init_env: params={:#x} argc={} argv={:#x}",
        params, argc, argv_ptr
    );
    // Store into a dedicated guest block so later libc accessors (which read
    // these via data-symbol globals) see the real values:
    // g_argc = argc, g_argv = argv, g_envp = argv + argc + 1.
    let mut block = ENV_BLOCK.lock().unwrap();
    if *block == 0 {
        // Place the env block in the pthread/misc region just above the TCB.
        *block = 0x0F00_8000u64;
        let _ = ctx.vm.map(*block, 0x1000, ps5vm::MemProtect::RW);
    }
    let base = *block;
    let _ = ctx.vm.write(base, &(argc as u64).to_le_bytes());
    let _ = ctx.vm.write(base + 8, &argv_ptr.to_le_bytes());
    let envp = argv_ptr + ((argc as u64) + 1) * 8;
    let _ = ctx.vm.write(base + 16, &envp.to_le_bytes());
    OK
}

fn cb_catch_return_from_main(_ctx: &mut CpuContext) -> HleResult {
    log::info!("[cxx] catchReturnFromMain — guest main() returned");
    OK
}

/// Register C++ ABI HLE exports.
pub fn register(reg: &mut HleRegistry) {
    reg.register_fn("3GPpjQdAMTw", "__cxa_guard_acquire", "libc", cb_cxa_guard_acquire);
    reg.register_fn("9rAeANT2tyE", "__cxa_guard_release", "libc", cb_cxa_guard_release);
    reg.register_fn("2emaaluWzUw", "__cxa_guard_abort", "libc", cb_cxa_guard_abort);
    reg.register_fn("8G2LB+A3rzg", "atexit", "libc", cb_atexit);
    reg.register_fn("tsvEmnenz48", "__cxa_atexit", "libc", cb_cxa_atexit);
    reg.register_fn("H2e8t5ScQGc", "__cxa_finalize", "libc", cb_cxa_finalize);
    reg.register_fn("w32KS2RAD48", "std_execute_once", "libc", cb_std_execute_once);
    reg.register_fallback_fn("DiGVep5yB5w", "_ZSt13_Execute_onceRSt9once_flagPFiPvS1_PS1_ES1_", "libc", cb_std_execute_once);
    reg.register_fn("bzQExy189ZI", "_init_env", "libc", cb_init_env);
    reg.register_fn("XKRegsFpEpk", "catchReturnFromMain", "libc", cb_catch_return_from_main);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::{Rdi, Rdx, Rsi};

    /// atexit and __cxa_atexit handlers form one list: exit runs them in
    /// reverse registration order, interleaved; a dso-specific finalize
    /// leaves the atexit handlers for exit.
    #[test]
    fn exit_handlers_run_latest_first_across_atexit_and_cxa_atexit() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        let atexit = |ctx: &mut CpuContext, f: u64| {
            ctx.set(Rdi, f);
            cb_atexit(ctx)
        };
        let cxa = |ctx: &mut CpuContext, f: u64, obj: u64, dso: u64| {
            ctx.set(Rdi, f);
            ctx.set(Rsi, obj);
            ctx.set(Rdx, dso);
            cb_cxa_atexit(ctx)
        };
        assert_eq!(atexit(&mut ctx, 0xA1), OK);
        assert_eq!(cxa(&mut ctx, 0xB1, 0x11, 0xD50), OK);
        assert_eq!(atexit(&mut ctx, 0xA2), OK);
        assert_eq!(atexit(&mut ctx, 0), OK); // ignored
        assert_eq!(cxa(&mut ctx, 0xB2, 0x22, 0xD51), OK);
        assert_eq!(take_cxa_destructor(0xD50), Some((0xB1, 0x11)));
        assert_eq!(take_cxa_destructor(0xD50), None);
        assert_eq!(take_cxa_destructor(0), Some((0xB2, 0x22)));
        assert_eq!(take_cxa_destructor(0), Some((0xA2, 0)));
        assert_eq!(take_cxa_destructor(0), Some((0xA1, 0)));
        assert_eq!(take_cxa_destructor(0), None);
    }
}
