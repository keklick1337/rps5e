// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceFiber — cooperative fibers with real guest context switches.
//!
//! A fiber switch has to exchange the guest stack pointer and callee-saved
//! registers. The HLE callback validates the request and updates the fiber
//! bookkeeping, then hands the actual switch to `ps5hle_fiber_switch`, a
//! small routine executed in guest context: the import "tail-calls" it
//! (`ps5cpu::exec::set_trap_tail_call`), so it runs after the trap's signal
//! frame has been consumed, exactly like the body of a native library
//! function. RAX carries a per-host-thread [`SwitchDesc`].
//!
//! The routine pushes rbp/rbx/r12–r15 and MXCSR/x87 control word on the
//! departing stack, stores the resulting stack pointer in the departing
//! fiber object (or the thread's root slot), publishes the departing fiber
//! as IDLE, and then either pops the same frame from the target's saved
//! stack (returning 0 from the target's own switch call) or starts a fresh
//! fiber at its entry point. Because the IDLE store happens only after the
//! departing stack is no longer touched, a fiber may be resumed by another
//! thread right away (fibers migrate between threads); resumers wait while
//! a fiber is still SUSPENDING. The guest thread pointer is never changed:
//! a fiber always runs with the TP of the thread currently executing it.
//!
//! Out-arguments (`argOnReturn`, `argOnRun`) of a suspended call are written
//! by whoever resumes it, before the switch, while the target is idle.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::context::{CpuContext, CpuRegister};
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceFiber";

pub const ERR_NULL: HleResult = 0x8059_0001u32 as i32 as i64;
pub const ERR_ALIGNMENT: HleResult = 0x8059_0002u32 as i32 as i64;
pub const ERR_RANGE: HleResult = 0x8059_0003u32 as i32 as i64;
pub const ERR_INVALID: HleResult = 0x8059_0004u32 as i32 as i64;
pub const ERR_PERMISSION: HleResult = 0x8059_0005u32 as i32 as i64;
pub const ERR_STATE: HleResult = 0x8059_0006u32 as i32 as i64;

const MAGIC_START: u32 = 0xDEF1_649C;
const MAGIC_END: u32 = 0xB375_92A0;
const OPT_MAGIC: u32 = 0xBB40_E64D;
const STACK_MAGIC: u64 = 0x7149_F2CA_7149_F2CA;
const SIZE_CHECK_FILL: u64 = 0xDEAD_BEEF_DEAD_BEEF;
const CONTEXT_MIN_SIZE: u64 = 512;
const MAX_NAME: usize = 31;
const INFO_SIZE: u64 = 128;

pub const STATE_RUNNING: u32 = 1;
pub const STATE_IDLE: u32 = 2;
pub const STATE_TERMINATED: u32 = 3;
/// Switching away: the context is not saved yet (internal, transient).
pub const STATE_SUSPENDING: u32 = 4;

const FLAG_CONTEXT_SIZE_CHECK: u32 = 0x10;
const FLAG_SET_FPU_REGS: u32 = 0x100;
const FW_350: u32 = 0x0350_0000;
const INITIAL_FCW: u16 = 0x037F;
const INITIAL_MXCSR: u32 = 0x9FC0;

// Guest fiber object layout (SceFiber, 0x100 bytes reserved by the guest).
const OFF_MAGIC_START: u64 = 0;
const OFF_STATE: u64 = 4;
const OFF_ENTRY: u64 = 8;
const OFF_ARG_INIT: u64 = 16;
const OFF_CTX_ADDR: u64 = 24;
const OFF_CTX_SIZE: u64 = 32;
const OFF_NAME: u64 = 40;
/// Saved stack pointer of a suspended fiber (0 = never run).
const OFF_SAVED_SP: u64 = 72;
const OFF_FLAGS: u64 = 80;
const OFF_CTX_START: u64 = 88;
const OFF_CTX_END: u64 = 96;
const OFF_MAGIC_END: u64 = 104;
/// Where to store `argOnRun` when this suspended fiber is resumed.
const OFF_RESUME_OUT: u64 = 112;
const OBJECT_SIZE: u64 = 0x100;

/// Parameters of one guest-side switch; read by `ps5hle_fiber_switch`.
#[repr(C)]
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwitchDesc {
    /// Where to store the departing stack pointer (0 = discard).
    pub save_slot: u64,
    /// State word set to IDLE once the departing context is saved (0 = none).
    pub state_ptr: u64,
    /// Resume: saved stack pointer. Fresh start: stack top (0 = current).
    pub new_sp: u64,
    /// Fresh start: entry point (0 = resume `new_sp`).
    pub entry: u64,
    pub arg0: u64,
    pub arg1: u64,
    pub mxcsr: u32,
    pub fcw: u16,
    /// Non-zero: load `mxcsr`/`fcw` before starting the entry.
    pub load_fpu: u16,
}

std::arch::global_asm!(
    ".globl ps5hle_fiber_switch",
    "ps5hle_fiber_switch:",
    "push rbp",
    "push rbx",
    "push r12",
    "push r13",
    "push r14",
    "push r15",
    "sub rsp, 8",
    "stmxcsr [rsp]",
    "fnstcw [rsp + 4]",
    "mov rcx, [rax]",
    "test rcx, rcx",
    "jz .Lps5hle_fiber_nosave",
    "mov [rcx], rsp",
    ".Lps5hle_fiber_nosave:",
    "mov rcx, [rax + 8]",
    "test rcx, rcx",
    "jz .Lps5hle_fiber_nostate",
    "mov dword ptr [rcx], 2",
    ".Lps5hle_fiber_nostate:",
    "mov rdx, [rax + 24]",
    "test rdx, rdx",
    "jnz .Lps5hle_fiber_fresh",
    "mov rsp, [rax + 16]",
    "ldmxcsr [rsp]",
    "fldcw [rsp + 4]",
    "add rsp, 8",
    "pop r15",
    "pop r14",
    "pop r13",
    "pop r12",
    "pop rbx",
    "pop rbp",
    "xor eax, eax",
    "ret",
    ".Lps5hle_fiber_fresh:",
    "mov rcx, [rax + 16]",
    "test rcx, rcx",
    "jz .Lps5hle_fiber_curstack",
    "mov rsp, rcx",
    ".Lps5hle_fiber_curstack:",
    "and rsp, -16",
    "cmp word ptr [rax + 54], 0",
    "je .Lps5hle_fiber_nofpu",
    "ldmxcsr [rax + 48]",
    "fldcw [rax + 52]",
    ".Lps5hle_fiber_nofpu:",
    "mov rdi, [rax + 32]",
    "mov rsi, [rax + 40]",
    // The frame chain ends at a zero record {saved rbp 0, return 0}: a
    // frame-pointer walker (Unreal Engine's stack walk) stops there instead
    // of reading address 8.
    "push 0",
    "push 0",
    "mov rbp, rsp",
    "call rdx",
    // A fiber entry function must never return.
    "ud2",
    ".globl ps5hle_fiber_switch_end",
    "ps5hle_fiber_switch_end:",
);

extern "C" {
    fn ps5hle_fiber_switch();
    fn ps5hle_fiber_switch_end();
}

/// Guest address of the switch routine.
pub fn switch_routine() -> u64 {
    ps5hle_fiber_switch as *const () as u64
}

/// Per-host-thread fiber state (one host thread per guest thread).
struct ThreadFibers {
    /// Fiber currently running on this thread (0 = the thread itself).
    current: u64,
    /// Thread stack pointer saved by the switch routine at sceFiberRun.
    thread_sp: u64,
    /// `argOnReturn` out-pointer of the pending sceFiberRun.
    thread_out: u64,
    /// Frame pointer of the sceFiberRun caller.
    thread_fp: u64,
    desc: SwitchDesc,
}

thread_local! {
    static THREAD: UnsafeCell<ThreadFibers> = const {
        UnsafeCell::new(ThreadFibers {
            current: 0,
            thread_sp: 0,
            thread_out: 0,
            thread_fp: 0,
            desc: SwitchDesc { save_slot: 0, state_ptr: 0, new_sp: 0, entry: 0, arg0: 0, arg1: 0, mxcsr: 0, fcw: 0, load_fpu: 0 },
        })
    };
}

fn thread() -> &'static mut ThreadFibers {
    // Only touched by its own thread; the address is stable for its life.
    THREAD.with(|t| unsafe { &mut *t.get() })
}

static CONTEXT_SIZE_CHECK: AtomicBool = AtomicBool::new(false);

fn state_atomic(fiber: u64) -> &'static AtomicU32 {
    unsafe { &*((fiber + OFF_STATE) as *const AtomicU32) }
}

fn validate(ctx: &CpuContext, fiber: u64) -> Result<(), HleResult> {
    if fiber == 0 {
        return Err(ERR_NULL);
    }
    if fiber & 7 != 0 {
        return Err(ERR_ALIGNMENT);
    }
    if !ps5vm::ranges::accessible(fiber, OBJECT_SIZE, true) {
        return Err(ERR_INVALID);
    }
    if ctx.read_u32(fiber + OFF_MAGIC_START) != Some(MAGIC_START) || ctx.read_u32(fiber + OFF_MAGIC_END) != Some(MAGIC_END) {
        return Err(ERR_INVALID);
    }
    Ok(())
}

fn stack_arg(ctx: &CpuContext, i: u64) -> u64 {
    ctx.read_u64(ctx.get(CpuRegister::Rsp) + 8 + 8 * i).unwrap_or(0)
}

fn write_name(ctx: &CpuContext, addr: u64, name: &str) {
    let mut buf = [0u8; MAX_NAME + 1];
    let b = name.as_bytes();
    let n = b.len().min(MAX_NAME);
    buf[..n].copy_from_slice(&b[..n]);
    ctx.write_bytes(addr, &buf);
}

#[allow(clippy::too_many_arguments)]
fn initialize(ctx: &mut CpuContext, fiber: u64, name_ptr: u64, entry: u64, arg_init: u64, ctx_addr: u64, ctx_size: u64, opt: u64, flags: u32, build: u32) -> HleResult {
    if fiber == 0 || name_ptr == 0 || entry == 0 {
        return ERR_NULL;
    }
    if fiber & 7 != 0 || ctx_addr & 15 != 0 || opt & 7 != 0 {
        return ERR_ALIGNMENT;
    }
    if ctx_size != 0 && ctx_size < CONTEXT_MIN_SIZE {
        return ERR_RANGE;
    }
    if ctx_size & 15 != 0 || (ctx_addr == 0) != (ctx_size == 0) {
        return ERR_INVALID;
    }
    if opt != 0 && ctx.read_u32(opt) != Some(OPT_MAGIC) {
        return ERR_INVALID;
    }
    if !ps5vm::ranges::accessible(fiber, OBJECT_SIZE, true) {
        return ERR_INVALID;
    }
    if ctx_addr != 0 && !ps5vm::ranges::accessible(ctx_addr, ctx_size, true) {
        return ERR_INVALID;
    }
    let Some(name) = ctx.read_cstring(name_ptr, MAX_NAME + 1) else { return ERR_INVALID };
    let mut flags = flags;
    if build >= FW_350 {
        flags |= FLAG_SET_FPU_REGS;
    }
    if CONTEXT_SIZE_CHECK.load(Ordering::Relaxed) {
        flags |= FLAG_CONTEXT_SIZE_CHECK;
    }
    ctx.write_bytes(fiber, &[0u8; OBJECT_SIZE as usize]);
    ctx.write_u32(fiber + OFF_MAGIC_START, MAGIC_START);
    ctx.write_u32(fiber + OFF_STATE, STATE_IDLE);
    ctx.write_u64(fiber + OFF_ENTRY, entry);
    ctx.write_u64(fiber + OFF_ARG_INIT, arg_init);
    ctx.write_u64(fiber + OFF_CTX_ADDR, ctx_addr);
    ctx.write_u64(fiber + OFF_CTX_SIZE, ctx_size);
    write_name(ctx, fiber + OFF_NAME, &name);
    ctx.write_u32(fiber + OFF_FLAGS, flags);
    ctx.write_u64(fiber + OFF_CTX_START, ctx_addr);
    ctx.write_u64(fiber + OFF_CTX_END, if ctx_addr == 0 { 0 } else { ctx_addr + ctx_size });
    ctx.write_u32(fiber + OFF_MAGIC_END, MAGIC_END);
    if ctx_addr != 0 {
        prepare_context(ctx, ctx_addr, ctx_size, flags);
    }
    log::debug!("[fiber] init {fiber:#x} '{name}' entry={entry:#x} ctx={ctx_addr:#x}+{ctx_size:#x} flags={flags:#x}");
    0
}

fn prepare_context(ctx: &CpuContext, addr: u64, size: u64, flags: u32) {
    ctx.write_u64(addr, STACK_MAGIC);
    if flags & FLAG_CONTEXT_SIZE_CHECK != 0 {
        let fill: Vec<u8> = std::iter::repeat_n(SIZE_CHECK_FILL.to_le_bytes(), (size / 8 - 1) as usize).flatten().collect();
        ctx.write_bytes(addr + 8, &fill);
    }
}

/// _sceFiberInitializeImpl(fiber, name, entry, argOnInit, addrContext,
/// sizeContext, optParam, buildVersion)
fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    let (opt, build) = (stack_arg(ctx, 0), stack_arg(ctx, 1) as u32);
    let (a, b, c, d, e, f) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8(), ctx.r9());
    initialize(ctx, a, b, c, d, e, f, opt, 0, build)
}

/// _sceFiberInitializeWithInternalOptionImpl(..., optParam, flags, buildVersion)
fn cb_initialize_internal(ctx: &mut CpuContext) -> HleResult {
    let (opt, flags, build) = (stack_arg(ctx, 0), stack_arg(ctx, 1) as u32, stack_arg(ctx, 2) as u32);
    let (a, b, c, d, e, f) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8(), ctx.r9());
    initialize(ctx, a, b, c, d, e, f, opt, flags, build)
}

fn cb_opt_param_initialize(ctx: &mut CpuContext) -> HleResult {
    let opt = ctx.rdi();
    if opt == 0 {
        return ERR_NULL;
    }
    if opt & 7 != 0 {
        return ERR_ALIGNMENT;
    }
    if ctx.write_u32(opt, OPT_MAGIC) { 0 } else { ERR_INVALID }
}

fn cb_finalize(ctx: &mut CpuContext) -> HleResult {
    let fiber = ctx.rdi();
    if let Err(e) = validate(ctx, fiber) {
        return e;
    }
    match state_atomic(fiber).compare_exchange(STATE_IDLE, STATE_TERMINATED, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => 0,
        Err(_) => ERR_STATE,
    }
}

/// Claim an idle fiber for running, waiting out a concurrent suspend.
fn acquire(fiber: u64) -> Result<(), HleResult> {
    let state = state_atomic(fiber);
    loop {
        match state.compare_exchange_weak(STATE_IDLE, STATE_RUNNING, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return Ok(()),
            Err(STATE_SUSPENDING) => std::thread::yield_now(),
            Err(STATE_IDLE) => continue,
            Err(_) => return Err(ERR_STATE),
        }
    }
}

fn attach_context(ctx: &CpuContext, fiber: u64, addr: u64, size: u64) -> Result<(), HleResult> {
    if addr & 15 != 0 {
        return Err(ERR_ALIGNMENT);
    }
    if size != 0 && size < CONTEXT_MIN_SIZE {
        return Err(ERR_RANGE);
    }
    if size & 15 != 0 || addr == 0 || size == 0 || ctx.read_u64(fiber + OFF_CTX_ADDR) != Some(0) {
        return Err(ERR_INVALID);
    }
    if !ps5vm::ranges::accessible(addr, size, true) {
        return Err(ERR_INVALID);
    }
    ctx.write_u64(fiber + OFF_CTX_ADDR, addr);
    ctx.write_u64(fiber + OFF_CTX_SIZE, size);
    ctx.write_u64(fiber + OFF_CTX_START, addr);
    ctx.write_u64(fiber + OFF_CTX_END, addr + size);
    let flags = ctx.read_u32(fiber + OFF_FLAGS).unwrap_or(0);
    prepare_context(ctx, addr, size, flags);
    Ok(())
}

/// Fill `desc` to enter `fiber` (already claimed): resume its saved context
/// or start it fresh.
fn target_desc(ctx: &CpuContext, fiber: u64, arg_on_run: u64, fresh_sp_fallback: u64, desc: &mut SwitchDesc) {
    let saved = ctx.read_u64(fiber + OFF_SAVED_SP).unwrap_or(0);
    if saved != 0 {
        let out = ctx.read_u64(fiber + OFF_RESUME_OUT).unwrap_or(0);
        if out != 0 {
            ctx.write_u64(out, arg_on_run);
        }
        ctx.write_u64(fiber + OFF_SAVED_SP, 0);
        desc.new_sp = saved;
        desc.entry = 0;
        return;
    }
    let addr = ctx.read_u64(fiber + OFF_CTX_ADDR).unwrap_or(0);
    let size = ctx.read_u64(fiber + OFF_CTX_SIZE).unwrap_or(0);
    let flags = ctx.read_u32(fiber + OFF_FLAGS).unwrap_or(0);
    desc.new_sp = if addr != 0 { (addr + size) & !15 } else { fresh_sp_fallback };
    desc.entry = ctx.read_u64(fiber + OFF_ENTRY).unwrap_or(0);
    desc.arg0 = ctx.read_u64(fiber + OFF_ARG_INIT).unwrap_or(0);
    desc.arg1 = arg_on_run;
    desc.load_fpu = (flags & FLAG_SET_FPU_REGS != 0) as u16;
    desc.mxcsr = INITIAL_MXCSR;
    desc.fcw = INITIAL_FCW;
}

/// Hand the prepared switch to the guest-side routine.
fn do_switch(desc: SwitchDesc) -> HleResult {
    let t = thread();
    t.desc = desc;
    ps5cpu::exec::set_trap_tail_call(switch_routine());
    &t.desc as *const SwitchDesc as i64
}

fn run(ctx: &mut CpuContext, fiber: u64, arg_on_run: u64, out: u64, attach: Option<(u64, u64)>) -> HleResult {
    if let Err(e) = validate(ctx, fiber) {
        return e;
    }
    if let Some((addr, size)) = attach {
        if let Err(e) = attach_context(ctx, fiber, addr, size) {
            return e;
        }
    }
    let t = thread();
    if t.current != 0 {
        return ERR_PERMISSION;
    }
    if let Err(e) = acquire(fiber) {
        return e;
    }
    t.current = fiber;
    t.thread_out = out;
    t.thread_fp = ctx.get(CpuRegister::Rbp);
    let mut desc = SwitchDesc { save_slot: &t.thread_sp as *const u64 as u64, ..Default::default() };
    // A contextless fiber runs on the thread stack, below the saved frame.
    target_desc(ctx, fiber, arg_on_run, 0, &mut desc);
    do_switch(desc)
}

fn switch(ctx: &mut CpuContext, fiber: u64, arg_on_run: u64, out: u64, attach: Option<(u64, u64)>) -> HleResult {
    if let Err(e) = validate(ctx, fiber) {
        return e;
    }
    if let Some((addr, size)) = attach {
        if let Err(e) = attach_context(ctx, fiber, addr, size) {
            return e;
        }
    }
    let t = thread();
    let me = t.current;
    if me == 0 {
        return ERR_PERMISSION;
    }
    if me == fiber {
        return ERR_STATE;
    }
    if let Err(e) = acquire(fiber) {
        return e;
    }
    ctx.write_u64(me + OFF_RESUME_OUT, out);
    state_atomic(me).store(STATE_SUSPENDING, Ordering::Release);
    t.current = fiber;
    let mut desc = SwitchDesc { save_slot: me + OFF_SAVED_SP, state_ptr: me + OFF_STATE, ..Default::default() };
    let below_thread = if t.thread_sp != 0 { t.thread_sp } else { 0 };
    target_desc(ctx, fiber, arg_on_run, below_thread, &mut desc);
    do_switch(desc)
}

/// sceFiberRun(fiber, argOnRun, argOnReturn*)
fn cb_run(ctx: &mut CpuContext) -> HleResult {
    let (f, a, o) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    run(ctx, f, a, o, None)
}

/// _sceFiberAttachContextAndRun(fiber, addrContext, sizeContext, argOnRun, argOnReturn*)
fn cb_attach_and_run(ctx: &mut CpuContext) -> HleResult {
    let (f, addr, size, a, o) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8());
    run(ctx, f, a, o, Some((addr, size)))
}

/// sceFiberSwitch(fiber, argOnRun, argOnRun*)
fn cb_switch(ctx: &mut CpuContext) -> HleResult {
    let (f, a, o) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    switch(ctx, f, a, o, None)
}

/// _sceFiberAttachContextAndSwitch(fiber, addrContext, sizeContext, argOnRun, argOnRun*)
fn cb_attach_and_switch(ctx: &mut CpuContext) -> HleResult {
    let (f, addr, size, a, o) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8());
    switch(ctx, f, a, o, Some((addr, size)))
}

/// sceFiberReturnToThread(argOnReturn, argOnRun*)
fn cb_return_to_thread(ctx: &mut CpuContext) -> HleResult {
    let t = thread();
    let me = t.current;
    if me == 0 || t.thread_sp == 0 {
        return ERR_PERMISSION;
    }
    let (arg_on_return, out) = (ctx.rdi(), ctx.rsi());
    if t.thread_out != 0 {
        ctx.write_u64(t.thread_out, arg_on_return);
    }
    ctx.write_u64(me + OFF_RESUME_OUT, out);
    state_atomic(me).store(STATE_SUSPENDING, Ordering::Release);
    t.current = 0;
    let new_sp = std::mem::take(&mut t.thread_sp);
    t.thread_out = 0;
    do_switch(SwitchDesc { save_slot: me + OFF_SAVED_SP, state_ptr: me + OFF_STATE, new_sp, ..Default::default() })
}

/// sceFiberGetSelf(fiber**)
fn cb_get_self(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_NULL;
    }
    let cur = thread().current;
    ctx.write_u64(out, cur);
    if cur == 0 { ERR_PERMISSION } else { 0 }
}

/// sceFiberGetInfo(fiber, info*)
fn cb_get_info(ctx: &mut CpuContext) -> HleResult {
    let (fiber, info) = (ctx.rdi(), ctx.rsi());
    if info == 0 {
        return ERR_NULL;
    }
    if let Err(e) = validate(ctx, fiber) {
        return e;
    }
    if ctx.read_u64(info) != Some(INFO_SIZE) {
        return ERR_INVALID;
    }
    let addr = ctx.read_u64(fiber + OFF_CTX_ADDR).unwrap_or(0);
    let size = ctx.read_u64(fiber + OFF_CTX_SIZE).unwrap_or(0);
    let flags = ctx.read_u32(fiber + OFF_FLAGS).unwrap_or(0);
    let name = ctx.read_bytes(fiber + OFF_NAME, MAX_NAME + 1).unwrap_or_default();
    ctx.write_u64(info + 8, ctx.read_u64(fiber + OFF_ENTRY).unwrap_or(0));
    ctx.write_u64(info + 16, ctx.read_u64(fiber + OFF_ARG_INIT).unwrap_or(0));
    ctx.write_u64(info + 24, addr);
    ctx.write_u64(info + 32, size);
    ctx.write_bytes(info + 40, &name);
    let mut margin = u64::MAX;
    if flags & FLAG_CONTEXT_SIZE_CHECK != 0 && addr != 0 {
        // Untouched fill words from the low end of the stack.
        let words = size / 8;
        let mut n = 1;
        while n < words && ctx.read_u64(addr + n * 8) == Some(SIZE_CHECK_FILL) {
            n += 1;
        }
        margin = (n - 1) * 8;
    }
    ctx.write_u64(info + 72, margin);
    0
}

/// sceFiberRename(fiber, name)
fn cb_rename(ctx: &mut CpuContext) -> HleResult {
    let (fiber, name_ptr) = (ctx.rdi(), ctx.rsi());
    if let Err(e) = validate(ctx, fiber) {
        return e;
    }
    if name_ptr == 0 {
        return ERR_NULL;
    }
    let Some(name) = ctx.read_cstring(name_ptr, MAX_NAME + 1) else { return ERR_INVALID };
    write_name(ctx, fiber + OFF_NAME, &name);
    0
}

fn cb_start_context_size_check(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() as u32 != 0 {
        return ERR_INVALID;
    }
    if CONTEXT_SIZE_CHECK.swap(true, Ordering::AcqRel) { ERR_STATE } else { 0 }
}

fn cb_stop_context_size_check(_ctx: &mut CpuContext) -> HleResult {
    if CONTEXT_SIZE_CHECK.swap(false, Ordering::AcqRel) { 0 } else { ERR_STATE }
}

/// _sceFiberGetThreadFramePointerAddress(addr*)
fn cb_get_thread_frame_pointer_address(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_NULL;
    }
    let t = thread();
    if t.current == 0 {
        return ERR_PERMISSION;
    }
    if ctx.write_u64(out, t.thread_fp) { 0 } else { ERR_INVALID }
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("hVYD7Ou2pCQ", "_sceFiberInitializeImpl", cb_initialize),
    ("7+OJIpko9RY", "_sceFiberInitializeWithInternalOptionImpl", cb_initialize_internal),
    ("asjUJJ+aa8s", "sceFiberOptParamInitialize", cb_opt_param_initialize),
    ("JeNX5F-NzQU", "sceFiberFinalize", cb_finalize),
    ("a0LLrZWac0M", "sceFiberRun", cb_run),
    ("avfGJ94g36Q", "_sceFiberAttachContextAndRun", cb_attach_and_run),
    ("PFT2S-tJ7Uk", "sceFiberSwitch", cb_switch),
    ("ZqhZFuzKT6U", "_sceFiberAttachContextAndSwitch", cb_attach_and_switch),
    ("B0ZX2hx9DMw", "sceFiberReturnToThread", cb_return_to_thread),
    ("p+zLIOg27zU", "sceFiberGetSelf", cb_get_self),
    ("uq2Y5BFz0PE", "sceFiberGetInfo", cb_get_info),
    ("JzyT91ucGDc", "sceFiberRename", cb_rename),
    ("Lcqty+QNWFc", "sceFiberStartContextSizeCheck", cb_start_context_size_check),
    ("Kj4nXMpnM8Y", "sceFiberStopContextSizeCheck", cb_stop_context_size_check),
    ("0dy4JtMUcMQ", "_sceFiberGetThreadFramePointerAddress", cb_get_thread_frame_pointer_address),
];

/// Exported callbacks by name (for tests and tools).
pub fn export(name: &str) -> Option<HleCallback> {
    EXPORTS.iter().find(|e| e.1 == name).map(|e| e.2)
}

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
    let start = ps5hle_fiber_switch as *const () as u64;
    let end = ps5hle_fiber_switch_end as *const () as u64;
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| ps5cpu::exec::add_guest_code_range(start, end - start, "fiber-switch"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn desc_layout_matches_routine() {
        assert_eq!(std::mem::offset_of!(SwitchDesc, new_sp), 16);
        assert_eq!(std::mem::offset_of!(SwitchDesc, entry), 24);
        assert_eq!(std::mem::offset_of!(SwitchDesc, arg0), 32);
        assert_eq!(std::mem::offset_of!(SwitchDesc, arg1), 40);
        assert_eq!(std::mem::offset_of!(SwitchDesc, mxcsr), 48);
        assert_eq!(std::mem::offset_of!(SwitchDesc, fcw), 52);
        assert_eq!(std::mem::offset_of!(SwitchDesc, load_fpu), 54);
    }

    #[test]
    fn initialize_validation_and_info() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C02_0000_0000;
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (fiber, name, stack, info) = (BASE, BASE + 0x200, BASE + 0x1000, BASE + 0x400);
        ctx.write_bytes(name, b"worker\0");
        assert_eq!(initialize(&mut ctx, 0, name, 0x1234, 0, 0, 0, 0, 0, 0), ERR_NULL);
        assert_eq!(initialize(&mut ctx, fiber + 4, name, 0x1234, 0, 0, 0, 0, 0, 0), ERR_ALIGNMENT);
        assert_eq!(initialize(&mut ctx, fiber, name, 0x1234, 0, stack, 0x100, 0, 0, 0), ERR_RANGE);
        assert_eq!(initialize(&mut ctx, fiber, name, 0x1234, 0, stack, 0x1008, 0, 0, 0), ERR_INVALID);
        assert_eq!(initialize(&mut ctx, fiber, name, 0x1234, 7, stack, 0x1000, 0, 0, FW_350), 0);
        assert_eq!(ctx.read_u32(fiber + OFF_STATE), Some(STATE_IDLE));
        assert_eq!(ctx.read_u64(stack), Some(STACK_MAGIC));
        ctx.write_u64(info, INFO_SIZE);
        ctx.set(CpuRegister::Rdi, fiber);
        ctx.set(CpuRegister::Rsi, info);
        assert_eq!(cb_get_info(&mut ctx), 0);
        assert_eq!(ctx.read_u64(info + 8), Some(0x1234));
        assert_eq!(ctx.read_u64(info + 16), Some(7));
        assert_eq!(ctx.read_cstring(info + 40, 32).as_deref(), Some("worker"));
        // Not inside a fiber: thread-only calls are refused.
        ctx.set(CpuRegister::Rdi, BASE + 0x800);
        assert_eq!(cb_get_self(&mut ctx), ERR_PERMISSION);
        assert_eq!(cb_return_to_thread(&mut ctx), ERR_PERMISSION);
        ctx.set(CpuRegister::Rdi, fiber);
        assert_eq!(cb_switch(&mut ctx), ERR_PERMISSION);
        assert_eq!(cb_finalize(&mut ctx), 0);
        assert_eq!(cb_finalize(&mut ctx), ERR_STATE);
    }
}
