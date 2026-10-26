// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Windows host layer of the execution core.
//!
//! * Per-thread state reachable from assembly without a register: two TEB
//!   TLS slots (`gs:[0x1480 + 8*index]`), one for the host stack of the
//!   innermost guest call, one for the guest thread pointer (read by the
//!   rewritten guest thread-pointer accesses, see `tls_patch`).
//! * Guest entry and exit: `setjmp`/`longjmp` of our own (the C runtime's
//!   `longjmp` unwinds through frames, which guest frames do not describe).
//!   A guest function returns to a sentinel page that jumps to
//!   `ps5cpu_guest_return`, which switches to the host stack and leaves the
//!   guest call; no fault is involved.
//! * Faults and traps: a vectored exception handler converts the
//!   `CONTEXT` to the Linux-shaped context of `ctx` and runs the shared
//!   handler, then writes the result back.
//! * Interrupting a guest thread ([`kick_thread`]): suspend it, and if it
//!   executes guest code, redirect it to `ps5cpu_win_interrupt_entry`,
//!   which saves the vector state and runs the shared interrupt handler on
//!   that thread, then resumes the saved context.

use crate::ctx::{self, greg_t, siginfo_t, ucontext_t, FpState};
use std::cell::Cell;
use std::sync::atomic::{AtomicU32, Ordering};
use windows_sys::Win32::Foundation::{CloseHandle, EXCEPTION_ACCESS_VIOLATION, EXCEPTION_BREAKPOINT, EXCEPTION_ILLEGAL_INSTRUCTION, EXCEPTION_INT_DIVIDE_BY_ZERO, EXCEPTION_PRIV_INSTRUCTION, NTSTATUS};
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, GetThreadContext, RtlRestoreContext, SetThreadContext, CONTEXT, CONTEXT_FULL_AMD64, EXCEPTION_CONTINUE_EXECUTION,
    EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
};
use windows_sys::Win32::System::Threading::{OpenThread, ResumeThread, SuspendThread, TlsAlloc, THREAD_GET_CONTEXT, THREAD_SET_CONTEXT, THREAD_SUSPEND_RESUME};

/// TEB offset of the TLS slot holding the host stack of the innermost
/// guest call (read by the import entry).
pub(crate) static mut HOST_STACK_SLOT: u32 = 0;
/// TEB offset of the TLS slot holding the guest thread pointer.
static GUEST_TP_SLOT: AtomicU32 = AtomicU32::new(0);
/// Size of the XSAVE image of the enabled state components (0: no XSAVE).
static XSAVE_SIZE: AtomicU32 = AtomicU32::new(0);

const EXCEPTION_IN_PAGE_ERROR: NTSTATUS = 0xC000_0006_u32 as NTSTATUS;
const EXCEPTION_INT_OVERFLOW: NTSTATUS = 0xC000_0095_u32 as NTSTATUS;

fn alloc_slot() -> u32 {
    // SAFETY: plain TLS index allocation.
    let index = unsafe { TlsAlloc() };
    assert!(index < 64, "no TEB TLS slot left for the guest thread state (index {index})");
    0x1480 + index * 8
}

/// Allocate the TLS slots and install the exception handler (once).
pub(crate) fn init() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: written once, before any guest code or import runs.
        unsafe { HOST_STACK_SLOT = alloc_slot() };
        GUEST_TP_SLOT.store(alloc_slot(), Ordering::Release);
        XSAVE_SIZE.store(xsave_size(), Ordering::Release);
        // SAFETY: registering a process-wide handler.
        unsafe { AddVectoredExceptionHandler(1, Some(veh)) };
    });
}

/// TEB offset of the guest thread-pointer slot.
pub fn guest_tp_slot() -> u32 {
    init();
    GUEST_TP_SLOT.load(Ordering::Acquire)
}

fn xsave_size() -> u32 {
    use std::arch::x86_64::__cpuid_count;
    let l1 = __cpuid_count(1, 0);
    // OSXSAVE: the OS enabled XSAVE.
    if l1.ecx & (1 << 27) == 0 {
        return 0;
    }
    let d = __cpuid_count(0xD, 0);
    d.ebx
}

/// Set the guest thread pointer of the calling thread.
pub(crate) fn set_thread_pointer(tp: u64) -> bool {
    let off = guest_tp_slot();
    // SAFETY: our own TEB slot.
    unsafe { std::arch::asm!("mov qword ptr gs:[{o}], {v}", o = in(reg) off as u64, v = in(reg) tp, options(nostack, preserves_flags)) };
    true
}

/// Guest thread pointer of the calling thread.
pub(crate) fn thread_pointer() -> u64 {
    let off = guest_tp_slot();
    let v: u64;
    // SAFETY: our own TEB slot.
    unsafe { std::arch::asm!("mov {v}, qword ptr gs:[{o}]", o = in(reg) off as u64, v = out(reg) v, options(nostack, preserves_flags, readonly)) };
    v
}

// ---------------------------------------------------------------------------
// setjmp / longjmp and the guest return path
// ---------------------------------------------------------------------------

// Buffer: rbx, rbp, r12-r15, rsp, rip, MXCSR, x87 control word. Only the
// System V callee-saved state: callers use the System V convention, so
// everything else is already saved around the call by the compiler.
std::arch::global_asm!(
    ".globl ps5cpu_setjmp",
    "ps5cpu_setjmp:",
    "mov [rdi], rbx",
    "mov [rdi + 8], rbp",
    "mov [rdi + 16], r12",
    "mov [rdi + 24], r13",
    "mov [rdi + 32], r14",
    "mov [rdi + 40], r15",
    "lea rdx, [rsp + 8]",
    "mov [rdi + 48], rdx",
    "mov rdx, [rsp]",
    "mov [rdi + 56], rdx",
    "stmxcsr [rdi + 64]",
    "fnstcw [rdi + 68]",
    "xor eax, eax",
    "ret",
    ".globl ps5cpu_longjmp",
    "ps5cpu_longjmp:",
    "mov rbx, [rdi]",
    "mov rbp, [rdi + 8]",
    "mov r12, [rdi + 16]",
    "mov r13, [rdi + 24]",
    "mov r14, [rdi + 32]",
    "mov r15, [rdi + 40]",
    "ldmxcsr [rdi + 64]",
    "fldcw [rdi + 68]",
    "mov rsp, [rdi + 48]",
    "mov eax, esi",
    "jmp qword ptr [rdi + 56]",
    // Reached through the sentinel page when a guest call returns: RAX is
    // its result, RSP the guest stack. Leave on the host stack.
    ".globl ps5cpu_guest_return",
    "ps5cpu_guest_return:",
    "mov r10d, dword ptr [rip + {slot}]",
    "mov r10, qword ptr gs:[r10]",
    "mov rsp, r10",
    "and rsp, -16",
    "mov rdi, rax",
    "call {returned}",
    "ud2",
    // Entered by a redirected guest thread (see `kick_thread`): RDI = the
    // saved CONTEXT, RSI = the XSAVE area (or 0); every other register
    // still holds the guest's value.
    ".globl ps5cpu_win_interrupt_entry",
    "ps5cpu_win_interrupt_entry:",
    "test rsi, rsi",
    "jz 1f",
    "mov eax, -1",
    "mov edx, -1",
    "xsave64 [rsi]",
    "1:",
    "and rsp, -16",
    "call {interrupted}",
    "ud2",
    slot = sym HOST_STACK_SLOT,
    returned = sym guest_returned,
    interrupted = sym interrupted,
);

extern "sysv64" {
    fn ps5cpu_setjmp(env: *mut u64) -> i32;
    fn ps5cpu_longjmp(env: *mut u64, val: i32) -> !;
    fn ps5cpu_guest_return();
    fn ps5cpu_win_interrupt_entry();
}

/// `sigsetjmp` replacement (the mask argument is ignored).
///
/// # Safety
/// Same contract as `sigsetjmp`.
pub(crate) unsafe fn setjmp<T>(env: *mut T, _savesigs: i32) -> i32 {
    ps5cpu_setjmp(env as *mut u64)
}

/// `siglongjmp` replacement.
///
/// # Safety
/// `env` must hold a live `setjmp` frame of this thread.
pub(crate) unsafe fn longjmp<T>(env: *mut T, val: i32) -> ! {
    ps5cpu_longjmp(env as *mut u64, val)
}

extern "sysv64" fn guest_returned(rax: u64) -> ! {
    crate::exec::leave_guest(crate::exec::GuestExit::Returned(rax))
}

/// Code of the sentinel page: `jmp [rip+0]` to `ps5cpu_guest_return`.
pub(crate) fn sentinel_code() -> [u8; 16] {
    let mut c = [0xCCu8; 16];
    c[..6].copy_from_slice(&[0xFF, 0x25, 0, 0, 0, 0]);
    c[6..14].copy_from_slice(&(ps5cpu_guest_return as *const () as u64).to_le_bytes());
    c
}

// ---------------------------------------------------------------------------
// Exceptions
// ---------------------------------------------------------------------------

thread_local! {
    static HOST_FAULT: Cell<bool> = const { Cell::new(false) };
    static TRAP_LOW: Cell<u64> = const { Cell::new(0) };
}

/// The shared handler found a host fault: pass the exception on.
pub(crate) fn set_host_fault() {
    HOST_FAULT.with(|c| c.set(true));
}

/// Lowest guest-stack address used by the exception being handled.
pub(crate) fn trap_frame_low() -> u64 {
    TRAP_LOW.with(|c| c.get())
}

#[inline(always)]
fn read_rsp() -> u64 {
    let rsp: u64;
    // SAFETY: reads the stack pointer.
    unsafe { std::arch::asm!("mov {}, rsp", out(reg) rsp, options(nomem, nostack, preserves_flags)) };
    rsp
}

fn to_uc(c: &mut CONTEXT, rip_adjust: u64) -> ucontext_t {
    let mut g: [greg_t; 23] = [0; 23];
    let set = |g: &mut [greg_t; 23], r: ctx::c_int, v: u64| g[r as usize] = v as greg_t;
    set(&mut g, ctx::REG_R8, c.R8);
    set(&mut g, ctx::REG_R9, c.R9);
    set(&mut g, ctx::REG_R10, c.R10);
    set(&mut g, ctx::REG_R11, c.R11);
    set(&mut g, ctx::REG_R12, c.R12);
    set(&mut g, ctx::REG_R13, c.R13);
    set(&mut g, ctx::REG_R14, c.R14);
    set(&mut g, ctx::REG_R15, c.R15);
    set(&mut g, ctx::REG_RDI, c.Rdi);
    set(&mut g, ctx::REG_RSI, c.Rsi);
    set(&mut g, ctx::REG_RBP, c.Rbp);
    set(&mut g, ctx::REG_RBX, c.Rbx);
    set(&mut g, ctx::REG_RDX, c.Rdx);
    set(&mut g, ctx::REG_RAX, c.Rax);
    set(&mut g, ctx::REG_RCX, c.Rcx);
    set(&mut g, ctx::REG_RSP, c.Rsp);
    set(&mut g, ctx::REG_RIP, c.Rip.wrapping_add(rip_adjust));
    set(&mut g, ctx::REG_EFL, c.EFlags as u64);
    set(&mut g, ctx::REG_CSGSFS, c.SegCs as u64 | (c.SegGs as u64) << 16 | (c.SegFs as u64) << 32);
    // FltSave is the FXSAVE image of the context (16-byte aligned).
    let fp = std::ptr::addr_of_mut!(c.Anonymous.FltSave) as *mut FpState;
    ucontext_t { uc_mcontext: ctx::mcontext_t { gregs: g, fpregs: fp }, native: c as *mut CONTEXT as *mut ctx::c_void }
}

fn from_uc(c: &mut CONTEXT, uc: &ucontext_t) {
    let g = &uc.uc_mcontext.gregs;
    let get = |r: ctx::c_int| g[r as usize] as u64;
    c.R8 = get(ctx::REG_R8);
    c.R9 = get(ctx::REG_R9);
    c.R10 = get(ctx::REG_R10);
    c.R11 = get(ctx::REG_R11);
    c.R12 = get(ctx::REG_R12);
    c.R13 = get(ctx::REG_R13);
    c.R14 = get(ctx::REG_R14);
    c.R15 = get(ctx::REG_R15);
    c.Rdi = get(ctx::REG_RDI);
    c.Rsi = get(ctx::REG_RSI);
    c.Rbp = get(ctx::REG_RBP);
    c.Rbx = get(ctx::REG_RBX);
    c.Rdx = get(ctx::REG_RDX);
    c.Rax = get(ctx::REG_RAX);
    c.Rcx = get(ctx::REG_RCX);
    c.Rsp = get(ctx::REG_RSP);
    c.Rip = get(ctx::REG_RIP);
    c.EFlags = get(ctx::REG_EFL) as u32;
    // SAFETY: plain field of the FXSAVE image.
    c.MxCsr = unsafe { c.Anonymous.FltSave.MxCsr };
}

unsafe extern "system" fn veh(ptrs: *mut EXCEPTION_POINTERS) -> i32 {
    let rec = &*(*ptrs).ExceptionRecord;
    let c = &mut *(*ptrs).ContextRecord;
    let signum = match rec.ExceptionCode {
        EXCEPTION_ACCESS_VIOLATION | EXCEPTION_IN_PAGE_ERROR => ctx::SIGSEGV,
        EXCEPTION_BREAKPOINT => ctx::SIGTRAP,
        EXCEPTION_ILLEGAL_INSTRUCTION | EXCEPTION_PRIV_INSTRUCTION => ctx::SIGILL,
        EXCEPTION_INT_DIVIDE_BY_ZERO | EXCEPTION_INT_OVERFLOW => ctx::SIGFPE,
        _ => return EXCEPTION_CONTINUE_SEARCH,
    };
    if signum == ctx::SIGSEGV && rec.NumberParameters >= 2 && ps5vm::os::space::wait_transient(rec.ExceptionInformation[1] as u64) {
        // The page was being re-mapped by another thread: retry.
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    let restore = crate::exec::import_restore_address();
    if signum == ctx::SIGTRAP && c.Rip != restore && !crate::exec::is_guest_code_addr(c.Rip) {
        // A breakpoint in host code belongs to a debugger.
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // Linux reports the address after INT3.
    let mut uc = to_uc(c, (signum == ctx::SIGTRAP) as u64);
    let mut info = siginfo_t { si_code: 2, addr: rec.ExceptionAddress as u64 };
    if signum == ctx::SIGSEGV && rec.NumberParameters >= 2 {
        info.addr = rec.ExceptionInformation[1] as u64;
        // Page-fault error code: bit 1 = write, bit 4 = instruction fetch.
        let err = match rec.ExceptionInformation[0] {
            1 => 2,
            8 => 0x10,
            _ => 0,
        };
        uc.uc_mcontext.gregs[ctx::REG_ERR as usize] = err;
    }
    let saved_low = TRAP_LOW.with(|l| l.replace(read_rsp().saturating_sub(0x1000)));
    HOST_FAULT.with(|h| h.set(false));
    crate::exec::guest_signal_handler(signum, &mut info, &mut uc as *mut ucontext_t as *mut ctx::c_void);
    TRAP_LOW.with(|l| l.set(saved_low));
    if HOST_FAULT.with(|h| h.replace(false)) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    from_uc(c, &uc);
    EXCEPTION_CONTINUE_EXECUTION
}

// ---------------------------------------------------------------------------
// Interrupting a guest thread
// ---------------------------------------------------------------------------

/// Identity of a host thread that can be interrupted (its thread id).
pub type HostThread = u64;

/// The calling thread's [`HostThread`].
pub fn current_host_thread() -> HostThread {
    ps5vm::os::current_tid()
}

/// Interrupt host thread `tid` so it runs the signal hook. A thread that
/// executes host code (an HLE call) is left alone, as on Linux: delivery
/// waits for its next safe point. True when the thread was reached.
pub fn kick_thread(tid: HostThread) -> bool {
    // SAFETY: suspending another thread of this process and editing its
    // context while it is stopped.
    unsafe {
        let h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, 0, tid as u32);
        if h.is_null() {
            return false;
        }
        if SuspendThread(h) == u32::MAX {
            CloseHandle(h);
            return false;
        }
        let mut c: Box<Aligned> = Box::new(std::mem::zeroed());
        c.0.ContextFlags = CONTEXT_FULL_AMD64;
        if GetThreadContext(h, &mut c.0) != 0 && crate::exec::is_guest_code_addr(c.0.Rip) {
            let xs = XSAVE_SIZE.load(Ordering::Acquire) as u64;
            // Below the red zone: the XSAVE area, then the CONTEXT copy.
            let mut sp = c.0.Rsp - 128;
            let mut xsave = 0;
            if xs != 0 {
                sp = (sp - xs) & !63;
                xsave = sp;
                std::ptr::write_bytes(xsave as *mut u8, 0, 64 + 512);
            }
            sp = (sp - std::mem::size_of::<CONTEXT>() as u64) & !63;
            let saved = sp;
            std::ptr::copy_nonoverlapping(&c.0, saved as *mut CONTEXT, 1);
            let mut redirect = c.0;
            redirect.Rsp = sp - 64;
            redirect.Rip = ps5cpu_win_interrupt_entry as *const () as u64;
            redirect.Rdi = saved;
            redirect.Rsi = xsave;
            redirect.ContextFlags = CONTEXT_FULL_AMD64;
            SetThreadContext(h, &redirect);
        }
        ResumeThread(h);
        CloseHandle(h);
        true
    }
}

#[repr(C, align(16))]
struct Aligned(CONTEXT);

/// Zero bits 255:128 of YMM `index` of an interrupted context: in its
/// extended state when the context carries one, else in the live register
/// (which the exception return then leaves alone).
///
/// # Safety
/// `native` must be null or the `CONTEXT` of the exception being handled.
pub(crate) unsafe fn clear_upper_ymm(native: *mut ctx::c_void, index: u8) {
    use windows_sys::Win32::System::Diagnostics::Debug::LocateXStateFeature;
    const XSTATE_AVX: u32 = 2;
    if !native.is_null() {
        let mut len = 0u32;
        let p = LocateXStateFeature(native as *const CONTEXT, XSTATE_AVX, &mut len) as *mut u8;
        if !p.is_null() && len >= 256 {
            std::ptr::write_bytes(p.add(index as usize * 16), 0, 16);
            return;
        }
    }
    macro_rules! clear {
        ($($n:literal),*) => {
            match index {
                $($n => std::arch::asm!(concat!("vmovaps xmm", $n, ", xmm", $n), options(nomem, nostack, preserves_flags)),)*
                _ => {}
            }
        };
    }
    clear!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}

extern "sysv64" fn interrupted(saved: *mut CONTEXT, xsave: *mut u8) -> ! {
    // SAFETY: the context and XSAVE area `kick_thread` placed on this
    // thread's guest stack.
    unsafe {
        let c = &mut *saved;
        let mut uc = to_uc(c, 0);
        let mut info = siginfo_t { si_code: 0, addr: 0 };
        let saved_low = TRAP_LOW.with(|l| l.replace(read_rsp().saturating_sub(0x1000)));
        crate::signal::interrupt_handler(0, &mut info, &mut uc as *mut ucontext_t as *mut ctx::c_void);
        TRAP_LOW.with(|l| l.set(saved_low));
        from_uc(c, &uc);
        if !xsave.is_null() {
            // Vector state beyond the FXSAVE image (upper YMM halves).
            std::arch::asm!("xrstor64 [{}]", in(reg) xsave, in("eax") -1i32, in("edx") -1i32, options(nostack));
        }
        RtlRestoreContext(c, std::ptr::null());
        unreachable!()
    }
}
