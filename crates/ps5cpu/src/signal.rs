// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest signal (exception) delivery support.
//!
//! The policy — handler table, pending signals per guest thread, who gets
//! what — lives in the HLE layer. This module provides the CPU-level
//! mechanics:
//!
//!   * the Orbis `ucontext_t` layout handed to guest exception handlers
//!     ([`OrbisUcontext`]) and its conversion from/to the host signal
//!     context;
//!   * a process-wide host signal ([`interrupt_signal`]) used to interrupt
//!     a guest thread. When it lands while the thread executes guest code,
//!     the handler switches to the host stack (like an import trap) and
//!     calls the registered [`SignalHook`]; the hook may run guest code
//!     with [`crate::exec::call_guest_nested`] and change the interrupted
//!     context. When it lands in host code (an HLE call, possibly blocked
//!     on a host primitive) nothing is run: the HLE layer must deliver at
//!     a safe point instead;
//!   * an import-return safe point: after every HLE import, before the
//!     guest resumes, the hook is called with the guest context at the
//!     return address;
//!   * [`current_import_interrupt`] — a read-only view of the guest state
//!     of the import being dispatched, for delivery from inside HLE waits.

use crate::exec;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Orbis ucontext
// ---------------------------------------------------------------------------

/// amd64 machine context as seen by guest signal/exception handlers.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OrbisMcontext {
    pub mc_onstack: u64,
    pub mc_rdi: u64,
    pub mc_rsi: u64,
    pub mc_rdx: u64,
    pub mc_rcx: u64,
    pub mc_r8: u64,
    pub mc_r9: u64,
    pub mc_rax: u64,
    pub mc_rbx: u64,
    pub mc_rbp: u64,
    pub mc_r10: u64,
    pub mc_r11: u64,
    pub mc_r12: u64,
    pub mc_r13: u64,
    pub mc_r14: u64,
    pub mc_r15: u64,
    pub mc_trapno: u32,
    pub mc_fs: u16,
    pub mc_gs: u16,
    pub mc_addr: u64,
    pub mc_flags: u32,
    pub mc_es: u16,
    pub mc_ds: u16,
    pub mc_err: u64,
    pub mc_rip: u64,
    pub mc_cs: u64,
    pub mc_rflags: u64,
    pub mc_rsp: u64,
    pub mc_ss: u64,
    pub mc_len: u64,
    pub mc_fpformat: u64,
    pub mc_ownedfp: u64,
    pub mc_lbrfrom: u64,
    pub mc_lbrto: u64,
    pub mc_aux1: u64,
    pub mc_aux2: u64,
    /// FXSAVE image (first 512 bytes) followed by extended state.
    pub mc_fpstate: [u64; 104],
    pub mc_fsbase: u64,
    pub mc_gsbase: u64,
    pub mc_spare: [u64; 6],
}

/// Orbis `ucontext_t`: a 16-byte signal mask and 0x30 bytes of private
/// fields precede the machine context, so `mc_rsp` sits at offset 0xF8
/// (the field the platform's GC suspend handlers read).
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct OrbisUcontext {
    pub uc_sigmask: [u32; 4],
    pub uc_reserved: [i32; 12],
    pub uc_mcontext: OrbisMcontext,
    pub uc_link: u64,
    pub uc_stack_sp: u64,
    pub uc_stack_size: u64,
    pub uc_stack_flags: i32,
    pub uc_stack_align: i32,
    pub uc_flags: i32,
    pub uc_spare: [i32; 4],
    pub uc_tail: [i32; 3],
}

const _: () = {
    assert!(std::mem::size_of::<OrbisMcontext>() == 0x480);
    assert!(std::mem::size_of::<OrbisUcontext>() == 0x500);
    assert!(std::mem::offset_of!(OrbisUcontext, uc_mcontext) == 0x40);
    assert!(std::mem::offset_of!(OrbisMcontext, mc_rip) == 0xA0);
    assert!(std::mem::offset_of!(OrbisMcontext, mc_rsp) == 0xB8);
    assert!(std::mem::offset_of!(OrbisMcontext, mc_fpstate) == 0x100);
    assert!(std::mem::offset_of!(OrbisMcontext, mc_fsbase) == 0x440);
};

/// `mc_fpformat`: FXSAVE layout.
pub const MC_FPFMT_XMM: u64 = 0x10002;
/// `mc_ownedfp`: state was saved from the FPU.
pub const MC_FPOWNED_FPU: u64 = 0x20001;
/// `mc_flags`: `mc_fsbase`/`mc_gsbase` are valid.
pub const MC_HASBASES: u32 = 0x2;

impl OrbisUcontext {
    pub fn zeroed() -> Self {
        // All-zero is a valid value of this plain-data struct.
        unsafe { std::mem::zeroed() }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self as *const Self as *const u8, std::mem::size_of::<Self>()) }
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() < std::mem::size_of::<Self>() {
            return None;
        }
        let mut uc = Self::zeroed();
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), &mut uc as *mut Self as *mut u8, std::mem::size_of::<Self>()) };
        Some(uc)
    }
}

// ---------------------------------------------------------------------------
// Interrupted guest state
// ---------------------------------------------------------------------------

/// Where the guest was stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterruptKind {
    /// Interrupted by [`interrupt_signal`] while executing guest code.
    Async,
    /// An HLE import has finished and is about to return to the guest.
    ImportReturn,
    /// Inside an HLE import (read-only view: the import has not returned).
    InImport,
}

/// The guest CPU state at an interruption point (host signal context).
pub struct GuestInterrupt {
    gregs: *mut crate::ctx::greg_t,
    fp: *mut crate::ctx::FpState,
    kind: InterruptKind,
}

/// Byte ranges of the FXSAVE image a handler may change: x87 control
/// words, ST0-7 and XMM0-15. MXCSR/MXCSR_MASK (24..32) and the reserved
/// tail (416..512, which holds the kernel's extended-state marker) are
/// kept from the host frame.
const FX_WRITABLE: [(usize, usize); 2] = [(0, 24), (32, 416)];

impl GuestInterrupt {
    pub fn kind(&self) -> InterruptKind {
        self.kind
    }

    fn g(&self, r: crate::ctx::c_int) -> u64 {
        unsafe { *self.gregs.add(r as usize) as u64 }
    }

    fn set(&mut self, r: crate::ctx::c_int, v: u64) {
        unsafe { *self.gregs.add(r as usize) = v as crate::ctx::greg_t };
    }

    /// Guest RSP and RIP as the handler should see them. For an import
    /// still in progress the guest is presented as having just returned
    /// from the import call.
    fn rsp_rip(&self) -> (u64, u64) {
        let rsp = self.g(crate::ctx::REG_RSP);
        match self.kind {
            InterruptKind::InImport => {
                let ret = if ps5vm::ranges::accessible(rsp, 8, false) { unsafe { std::ptr::read(rsp as *const u64) } } else { 0 };
                (rsp + 8, ret)
            }
            _ => (rsp, self.g(crate::ctx::REG_RIP)),
        }
    }

    /// Build the guest `ucontext_t` for this state.
    pub fn ucontext(&self) -> OrbisUcontext {
        use crate::ctx::*;
        let mut uc = OrbisUcontext::zeroed();
        let m = &mut uc.uc_mcontext;
        m.mc_rdi = self.g(REG_RDI);
        m.mc_rsi = self.g(REG_RSI);
        m.mc_rdx = self.g(REG_RDX);
        m.mc_rcx = self.g(REG_RCX);
        m.mc_r8 = self.g(REG_R8);
        m.mc_r9 = self.g(REG_R9);
        m.mc_rax = self.g(REG_RAX);
        m.mc_rbx = self.g(REG_RBX);
        m.mc_rbp = self.g(REG_RBP);
        m.mc_r10 = self.g(REG_R10);
        m.mc_r11 = self.g(REG_R11);
        m.mc_r12 = self.g(REG_R12);
        m.mc_r13 = self.g(REG_R13);
        m.mc_r14 = self.g(REG_R14);
        m.mc_r15 = self.g(REG_R15);
        let (rsp, rip) = self.rsp_rip();
        m.mc_rsp = rsp;
        m.mc_rip = rip;
        m.mc_rflags = self.g(REG_EFL);
        // Linux packs cs/gs/fs into one greg.
        let csgsfs = self.g(REG_CSGSFS);
        m.mc_cs = csgsfs & 0xFFFF;
        m.mc_gs = ((csgsfs >> 16) & 0xFFFF) as u16;
        m.mc_fs = ((csgsfs >> 32) & 0xFFFF) as u16;
        m.mc_ss = 0x2B;
        m.mc_len = std::mem::size_of::<OrbisMcontext>() as u64;
        m.mc_flags = MC_HASBASES;
        // The guest thread pointer is what the guest reads through %fs.
        m.mc_fsbase = exec::current_thread_pointer();
        if !self.fp.is_null() {
            let dst = m.mc_fpstate.as_mut_ptr() as *mut u8;
            unsafe { std::ptr::copy_nonoverlapping(self.fp as *const u8, dst, 512) };
            // The direct import entry saves only what host code may change;
            // at the import-return safe point (on the guest thread, not in a
            // signal handler) the x87 status, tag, pointers and ST0-7 are
            // still the guest's own in the CPU: take them from there.
            if self.kind == InterruptKind::ImportReturn && exec::exec().trap_fp_partial {
                let live = live_fxsave();
                // SAFETY: both are 512-byte FXSAVE images.
                unsafe {
                    std::ptr::copy_nonoverlapping(live.0.as_ptr().add(2), dst.add(2), 22);
                    std::ptr::copy_nonoverlapping(live.0.as_ptr().add(32), dst.add(32), 128);
                }
            }
            m.mc_fpformat = MC_FPFMT_XMM;
            m.mc_ownedfp = MC_FPOWNED_FPU;
        }
        uc
    }

    /// Apply a (possibly handler-modified) context to the interrupted
    /// state. Returns false for a read-only view ([`InterruptKind::InImport`]).
    pub fn apply(&mut self, uc: &OrbisUcontext) -> bool {
        use crate::ctx::*;
        if self.kind == InterruptKind::InImport {
            return false;
        }
        if self.kind == InterruptKind::ImportReturn {
            exec::exec().context_replaced = true;
        }
        let m = &uc.uc_mcontext;
        self.set(REG_RDI, m.mc_rdi);
        self.set(REG_RSI, m.mc_rsi);
        self.set(REG_RDX, m.mc_rdx);
        self.set(REG_RCX, m.mc_rcx);
        self.set(REG_R8, m.mc_r8);
        self.set(REG_R9, m.mc_r9);
        self.set(REG_RAX, m.mc_rax);
        self.set(REG_RBX, m.mc_rbx);
        self.set(REG_RBP, m.mc_rbp);
        self.set(REG_R10, m.mc_r10);
        self.set(REG_R11, m.mc_r11);
        self.set(REG_R12, m.mc_r12);
        self.set(REG_R13, m.mc_r13);
        self.set(REG_R14, m.mc_r14);
        self.set(REG_R15, m.mc_r15);
        self.set(REG_RIP, m.mc_rip);
        self.set(REG_RSP, m.mc_rsp);
        // Only the user-modifiable arithmetic/control flags.
        const USER_FLAGS: u64 = 0xCD5 | (1 << 10) | (1 << 18) | (1 << 21);
        let fl = (self.g(REG_EFL) & !USER_FLAGS) | (m.mc_rflags & USER_FLAGS);
        self.set(REG_EFL, fl);
        // Segment selectors are left untouched: the kernel validates them.
        if !self.fp.is_null() && m.mc_fpformat == MC_FPFMT_XMM {
            let src = m.mc_fpstate.as_ptr() as *const u8;
            let dst = self.fp as *mut u8;
            for (a, b) in FX_WRITABLE {
                unsafe { std::ptr::copy_nonoverlapping(src.add(a), dst.add(a), b - a) };
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Hook and interrupt signal
// ---------------------------------------------------------------------------

/// Called on a guest thread, on its host stack, at a point where guest
/// signals may be delivered (see [`InterruptKind`]). No host locks are
/// held by the thread's innermost HLE frame. Must be cheap when nothing
/// is pending: it runs after every import.
pub type SignalHook = fn(&mut GuestInterrupt);

static HOOK: OnceLock<SignalHook> = OnceLock::new();

/// Install the process-wide signal-delivery hook (once).
pub fn set_signal_hook(h: SignalHook) {
    let _ = HOOK.set(h);
}

/// Host signal used to interrupt a guest thread.
#[cfg(unix)]
pub fn interrupt_signal() -> crate::ctx::c_int {
    libc::SIGRTMIN() + 3
}

/// Identity of a host thread that can be interrupted.
#[cfg(unix)]
pub type HostThread = libc::pthread_t;

/// The calling thread's [`HostThread`].
#[cfg(unix)]
pub fn current_host_thread() -> HostThread {
    // SAFETY: plain query.
    unsafe { libc::pthread_self() }
}

/// Interrupt host thread `thread` (a guest thread) so it checks for
/// pending guest signals.
#[cfg(unix)]
pub fn kick_thread(thread: libc::pthread_t) -> bool {
    unsafe { libc::pthread_kill(thread, interrupt_signal()) == 0 }
}

#[cfg(windows)]
pub use crate::win::{current_host_thread, kick_thread, HostThread};

/// Read-only view of the guest state at the import currently being
/// dispatched on this thread (None outside an import dispatch).
pub fn current_import_interrupt() -> Option<GuestInterrupt> {
    let t = exec::exec();
    if t.trap_gregs.is_null() || t.trap_rsp == 0 {
        return None;
    }
    Some(GuestInterrupt { gregs: t.trap_gregs, fp: t.trap_fp as *mut _, kind: InterruptKind::InImport })
}

/// Reserve `size` bytes (64-byte aligned) of guest stack below the
/// interrupted frame of the current import/interrupt, run `f` with its
/// address, and release it. Nested guest calls made inside `f` start below
/// the reservation. None outside an import dispatch or interrupt.
pub fn with_guest_scratch<R>(size: u64, f: impl FnOnce(u64) -> R) -> Option<R> {
    let t = exec::exec();
    let rsp = t.trap_rsp;
    if rsp == 0 {
        return None;
    }
    let top = exec::nested_stack_top();
    let addr = (top - size) & !0x3F;
    let saved = t.trap_frame_low;
    t.trap_frame_low = addr + 0x100;
    let r = f(addr);
    exec::exec().trap_frame_low = saved;
    Some(r)
}

/// Install the interrupt-signal handler (called from `exec::init`).
#[cfg(unix)]
pub(crate) fn install() {
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = interrupt_handler as *const () as usize;
        // The signal stays blocked while a delivery runs; a new interrupt
        // is taken after the handler returns.
        sa.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
        libc::sigemptyset(&mut sa.sa_mask);
        libc::sigaction(interrupt_signal(), &sa, std::ptr::null_mut());
    }
}

struct InterruptArgs {
    info: *mut crate::ctx::siginfo_t,
    uc: *mut crate::ctx::ucontext_t,
}

use crate::exec::ps5cpu_call_on_stack;

pub(crate) extern "C" fn interrupt_handler(_signum: crate::ctx::c_int, info: *mut crate::ctx::siginfo_t, uc: *mut crate::ctx::c_void) {
    let uc = uc as *mut crate::ctx::ucontext_t;
    let t = exec::exec();
    if t.depth == 0 || HOOK.get().is_none() {
        return;
    }
    let rip = unsafe { (*uc).uc_mcontext.gregs[crate::ctx::REG_RIP as usize] as u64 };
    // Only guest code can be interrupted: host code (an HLE call) may hold
    // locks, so delivery there waits for a safe point.
    if !exec::is_guest_code_addr(rip) {
        return;
    }
    let mut args = InterruptArgs { info, uc };
    let stack = t.frames[t.depth - 1].host_rsp;
    unsafe { ps5cpu_call_on_stack(&mut args as *mut _ as *mut crate::ctx::c_void, interrupt_body, stack) };
}

extern "sysv64" fn interrupt_body(arg: *mut crate::ctx::c_void) {
    let a = unsafe { &mut *(arg as *mut InterruptArgs) };
    let Some(hook) = HOOK.get() else { return };
    let gregs = unsafe { (*a.uc).uc_mcontext.gregs.as_mut_ptr() };
    let fp = unsafe { (*a.uc).uc_mcontext.fpregs };
    let t = exec::exec();
    let saved = (t.trap_rsp, t.trap_frame_low, t.trap_fp, t.trap_gregs);
    t.trap_rsp = unsafe { *gregs.add(crate::ctx::REG_RSP as usize) as u64 };
    t.trap_frame_low = exec::trap_frame_low(a.uc, a.info);

    t.trap_fp = fp;
    t.trap_gregs = gregs;
    let mut gi = GuestInterrupt { gregs, fp, kind: InterruptKind::Async };
    hook(&mut gi);
    let t = exec::exec();
    (t.trap_rsp, t.trap_frame_low, t.trap_fp, t.trap_gregs) = saved;
}

/// Import-return safe point (called by the import trap handler after the
/// return value and return address have been written to `gregs`).
pub(crate) fn import_safe_point(gregs: *mut crate::ctx::greg_t) {
    let Some(hook) = HOOK.get() else { return };
    let t = exec::exec();
    let saved = (t.trap_rsp, t.trap_gregs);
    t.trap_rsp = unsafe { *gregs.add(crate::ctx::REG_RSP as usize) as u64 };
    t.trap_gregs = gregs;
    let mut gi = GuestInterrupt { gregs, fp: t.trap_fp as *mut _, kind: InterruptKind::ImportReturn };
    hook(&mut gi);
    let t = exec::exec();
    (t.trap_rsp, t.trap_gregs) = saved;
}

/// A 512-byte FXSAVE image of this thread's live FPU state.
#[repr(C, align(16))]
struct FxImage([u8; 512]);

fn live_fxsave() -> FxImage {
    let mut img = FxImage([0; 512]);
    // SAFETY: a 16-byte aligned, 512-byte buffer owned by this frame.
    unsafe { std::arch::asm!("fxsave64 [{}]", in(reg) img.0.as_mut_ptr(), options(nostack, preserves_flags)) };
    img
}
