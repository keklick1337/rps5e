// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest execution core: every guest thread runs on its own host thread.
//!
//! Entering guest code (`enter_guest`) records a resume point with
//! `sigsetjmp`, loads the complete guest register file and jumps to the
//! guest RIP. The guest's return address is always `HOST_RETURN_SENTINEL`;
//! leaving guest code — normal return, `pthread_exit`, or a fatal fault —
//! always goes through the signal handler and `siglongjmp`, so guest code
//! can never corrupt host frames.
//!
//! Import stubs (`lea r11, [rip-7]; jmp [entry]`) jump straight to
//! `ps5cpu_import_entry`, which switches to the host thread's own stack
//! (below the host stack pointer saved by the innermost `enter_guest`),
//! saves the guest registers in a signal-context-shaped frame and
//! dispatches the HLE call there. With `RPS5E_IMPORT_TRAP=1` the stubs are
//! INT3 bytes instead and the SIGTRAP handler does the same; a trap that is
//! not an import stub is a guest fault. Because each nested guest call made
//! by an HLE function saves a lower host stack pointer, nesting is
//! unlimited.
//!
//! The guest thread pointer lives in the host GS base (see `tls_patch`).

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

#[cfg(unix)]
extern "C" {
    #[link_name = "__sigsetjmp"]
    fn sigsetjmp(env: *mut JmpBuf, savesigs: crate::ctx::c_int) -> crate::ctx::c_int;
    fn siglongjmp(env: *mut JmpBuf, val: crate::ctx::c_int) -> !;
}

#[cfg(windows)]
use crate::win::{longjmp as siglongjmp, setjmp as sigsetjmp};

/// glibc `sigjmp_buf` (200 bytes on x86-64), over-allocated.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct JmpBuf([u64; 32]);

/// Guest return address that terminates a guest call. The page holds HLT,
/// which faults in user mode and is recognised by the signal handler.
pub const HOST_RETURN_SENTINEL: u64 = 0x6FFC_0000_0000;
/// Process exit-handler stub passed to the entry point (`xor eax,eax; ret`).
pub const EXIT_HANDLER_STUB: u64 = 0x6FFD_0000_0000;

/// Maximum nesting of guest calls made from HLE callbacks on one thread.
const MAX_NESTING: usize = 32;

/// Full guest register file.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct GuestRegs {
    pub rax: u64,
    pub rbx: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rbp: u64,
    pub rsp: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    pub rip: u64,
}

/// How a guest call ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuestExit {
    /// The guest function returned to the sentinel with this RAX.
    Returned(u64),
    /// The guest thread called its exit routine with this value.
    ThreadExit(u64),
}

/// Trait object the signal handler calls to dispatch an HLE import.
///
/// `regs` holds the 16 GPRs in hardware order (rax, rcx, rdx, rbx, rsp,
/// rbp, rsi, rdi, r8..r15) at the moment of the call; `rsp` points at the
/// guest return address. Returns `Some(rax)` when the stub is known.
pub trait HleDispatcher: Send + Sync {
    fn dispatch_at(&self, stub_addr: u64, regs: &[u64; 16], rsp: u64, rip: u64) -> Option<u64>;
}

static DISPATCHER: OnceLock<Box<dyn HleDispatcher>> = OnceLock::new();

/// Install the process-wide HLE dispatcher (once).
pub fn set_dispatcher(d: Box<dyn HleDispatcher>) {
    let _ = DISPATCHER.set(d);
}

/// Guest code ranges, for telling guest faults from host bugs.
static GUEST_CODE: std::sync::RwLock<Vec<(u64, u64, String)>> = std::sync::RwLock::new(Vec::new());

/// Register an executable guest image range (with a name for diagnostics).
pub fn add_guest_code_range(base: u64, size: u64, name: &str) {
    GUEST_CODE.write().unwrap().push((base, base + size, name.to_string()));
}

/// Forget the executable guest ranges inside `[base, base + size)`.
pub fn remove_guest_code_ranges(base: u64, size: u64) {
    GUEST_CODE.write().unwrap().retain(|(s, _, _)| *s < base || *s >= base + size);
}

fn guest_code_owner(addr: u64) -> Option<(u64, String)> {
    GUEST_CODE
        .read()
        .ok()?
        .iter()
        .find(|(s, e, _)| addr >= *s && addr < *e)
        .map(|(s, _, n)| (*s, n.clone()))
}

#[derive(Clone, Copy)]
pub(crate) struct Frame {
    jmp: JmpBuf,
    /// Host stack pointer at entry; the handler runs below it.
    pub(crate) host_rsp: u64,
    exit: GuestExit,
}

pub(crate) struct ThreadExec {
    pub(crate) frames: [Frame; MAX_NESTING],
    pub(crate) depth: usize,
    tp: u64,
    imports: u64,
    /// Guest RSP at the import currently being dispatched (for nested calls).
    pub(crate) trap_rsp: u64,
    /// Saved FP state of the import trap being dispatched (null otherwise).
    pub(crate) trap_fp: *const crate::ctx::FpState,
    /// `trap_fp` is the direct entry's frame, which holds the x87 control
    /// word, MXCSR and XMM0-15 only (ST0-7 and the x87 status, tag and
    /// pointers are still live in the CPU); false for a kernel signal frame.
    pub(crate) trap_fp_partial: bool,
    /// Lowest address of the kernel signal frame of the trap being
    /// dispatched. The frame sits on the guest stack below `trap_rsp`, so
    /// nested guest calls must start below it.
    pub(crate) trap_frame_low: u64,
    /// Guest address the import being dispatched continues at instead of
    /// returning to its caller (0 = plain return). See `set_trap_tail_call`.
    tail_call: u64,
    /// Host register file (`gregs`) of the trap being dispatched.
    pub(crate) trap_gregs: *mut crate::ctx::greg_t,
    /// Set when the signal hook replaced the guest context at the
    /// import-return safe point (direct entry must restore it exactly).
    pub(crate) context_replaced: bool,
    /// Extra return registers of the import being dispatched (see
    /// `set_return_rdx` / `set_return_xmm0`).
    ret_rdx: Option<u64>,
    ret_xmm0: Option<u128>,
    /// x87 80-bit result pushed as ST0 (`long double` returns).
    ret_st0: Option<[u8; 10]>,
}

const EMPTY_FRAME: Frame = Frame { jmp: JmpBuf([0; 32]), host_rsp: 0, exit: GuestExit::Returned(0) };

thread_local! {
    static EXEC: UnsafeCell<ThreadExec> = const {
        UnsafeCell::new(ThreadExec {
            frames: [EMPTY_FRAME; MAX_NESTING],
            depth: 0,
            tp: 0,
            imports: 0,
            trap_rsp: 0,
            trap_fp: std::ptr::null(),
            trap_fp_partial: false,
            trap_frame_low: 0,
            tail_call: 0,
            trap_gregs: std::ptr::null_mut(),
            context_replaced: false,
            ret_rdx: None,
            ret_xmm0: None,
            ret_st0: None,
        })
    };
}

/// Access this thread's execution record. The record is only touched by
/// its own thread (including from its signal handler).
pub(crate) fn exec() -> &'static mut ThreadExec {
    EXEC.with(|e| unsafe { &mut *e.get() })
}

/// Guest thread pointer of the calling host thread (0 outside guest threads).
pub fn current_thread_pointer() -> u64 {
    exec().tp
}

/// Process-wide import counter, striped per thread so the hot path never
/// shares a cache line with another thread.
#[repr(align(64))]
struct Stripe(AtomicU64);
const STRIPES: usize = 64;
static IMPORT_STRIPES: [Stripe; STRIPES] = [const { Stripe(AtomicU64::new(0)) }; STRIPES];
static NEXT_STRIPE: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static STRIPE: usize = NEXT_STRIPE.fetch_add(1, Ordering::Relaxed) % STRIPES;
}

#[inline]
fn count_import() {
    STRIPE.with(|s| IMPORT_STRIPES[*s].0.fetch_add(1, Ordering::Relaxed));
}

/// Total HLE imports dispatched by all threads since start.
pub fn total_imports() -> u64 {
    IMPORT_STRIPES.iter().map(|s| s.0.load(Ordering::Relaxed)).sum()
}

/// Total HLE imports dispatched on the calling host thread.
pub fn imports_dispatched() -> u64 {
    exec().imports
}

/// Low 64 bits of guest XMM register `index` at the import currently being
/// dispatched on this thread (floating-point arguments arrive in xmm0..7).
/// None outside an import dispatch.
pub fn trap_xmm_low(index: usize) -> Option<u64> {
    let fp = exec().trap_fp;
    if fp.is_null() || index >= 16 {
        return None;
    }
    let e = unsafe { (*fp)._xmm[index].element };
    Some(e[0] as u64 | (e[1] as u64) << 32)
}

/// Run `f` as if it were an import dispatched on this thread with the low
/// 64 bits of xmm0.. set to `xmm` (tests of floating-point HLE functions).
/// Returns `f`'s result and the XMM0 return value it set, if any.
pub fn with_import_xmm<R>(xmm: &[u64], f: impl FnOnce() -> R) -> (R, Option<u128>) {
    // SAFETY: plain-old-data register image.
    let mut fp: crate::ctx::FpState = unsafe { std::mem::zeroed() };
    for (i, &v) in xmm.iter().enumerate().take(16) {
        fp._xmm[i].element[0] = v as u32;
        fp._xmm[i].element[1] = (v >> 32) as u32;
    }
    let t = exec();
    let saved = (t.trap_fp, t.ret_xmm0.take());
    t.trap_fp = &fp;
    let r = f();
    let t = exec();
    let ret = t.ret_xmm0.take();
    (t.trap_fp, t.ret_xmm0) = saved;
    (r, ret)
}

/// Make the import being dispatched on this thread continue at guest
/// address `target` instead of returning to its caller, as if the import had
/// tail-called `target`: the guest resumes there with the import's frame
/// intact (its return address at [rsp]), every register except RAX as the
/// caller left it, and RAX = the HLE result. Lets an HLE function finish its
/// work in guest context — e.g. a fiber switch, which must save and load
/// stack pointers after the trap's signal frame has been consumed. Must be
/// called last in the callback (nested guest calls start fresh).
pub fn set_trap_tail_call(target: u64) {
    exec().tail_call = target;
}

/// Make the import being dispatched on this thread return `value` in RDX
/// as well (the upper half of a 128-bit result: RAX:RDX).
pub fn set_return_rdx(value: u64) {
    exec().ret_rdx = Some(value);
}

/// Make the import being dispatched on this thread return `value` in XMM0
/// (floating-point results; the low lanes hold the value).
pub fn set_return_xmm0(value: u128) {
    exec().ret_xmm0 = Some(value);
}

/// Make the import being dispatched on this thread return the x87 80-bit
/// extended `value` in ST0, pushed onto the guest's x87 stack (`long
/// double` results).
pub fn set_return_st0(value: [u8; 10]) {
    exec().ret_st0 = Some(value);
}

/// Run `f` as if it were an import dispatched on this thread; returns
/// `f`'s result and the ST0 return value it set, if any (tests of `long
/// double` HLE functions).
pub fn with_import_st0<R>(f: impl FnOnce() -> R) -> (R, Option<[u8; 10]>) {
    let saved = exec().ret_st0.take();
    let r = f();
    let t = exec();
    let ret = std::mem::replace(&mut t.ret_st0, saved);
    (r, ret)
}

/// Push `value` onto the live x87 stack. The direct import entry saves and
/// restores only the x87 control word, so the guest finds it in ST0.
fn push_live_st0(value: &[u8; 10]) {
    unsafe { std::arch::asm!("fld tbyte ptr [{}]", in(reg) value.as_ptr(), options(nostack, readonly)) };
}

/// Push `value` onto the x87 stack held in an FXSAVE image (a signal frame
/// restored on return): ST0..ST6 move down one slot, TOP is decremented
/// and the new top's physical register is tagged valid.
pub fn push_saved_st0(fx: &mut [u8], value: &[u8; 10]) {
    fx.copy_within(32..32 + 7 * 16, 48);
    fx[32..42].copy_from_slice(value);
    fx[42..48].fill(0);
    let fsw = u16::from_le_bytes([fx[2], fx[3]]);
    let top = (((fsw >> 11) & 7) + 7) & 7;
    let fsw = (fsw & !(7 << 11)) | (top << 11);
    fx[2..4].copy_from_slice(&fsw.to_le_bytes());
    fx[4] |= 1 << top;
}

/// Apply an ST0 return to the FP state of the import being dispatched.
fn apply_return_st0(fp: *const crate::ctx::FpState, partial: bool, value: &[u8; 10]) {
    if partial {
        push_live_st0(value);
        return;
    }
    if fp.is_null() {
        return;
    }
    let fp = fp as *mut u8;
    unsafe {
        push_saved_st0(std::slice::from_raw_parts_mut(fp, 512), value);
        // An XSAVE signal frame restores x87 state only when its component
        // bit is set.
        let magic = std::ptr::read_unaligned(fp.add(464) as *const u32);
        if magic == 0x4650_5853 {
            let bv = fp.add(512) as *mut u64;
            std::ptr::write_unaligned(bv, std::ptr::read_unaligned(bv) | 1);
        }
    }
}

/// Bind the calling host thread to a guest thread pointer.
pub fn bind_thread_pointer(tp: u64) {
    exec().tp = tp;
    if !crate::tls_patch::set_thread_pointer(tp) {
        log::error!("[exec] cannot set the guest thread pointer {tp:#x}");
    }
}

// Load the full register file from `*rdi` and jump to its RIP; never
// returns. The RIP is pushed just below the guest RSP and popped by `ret`,
// which is safe because every entry is a fresh call whose frame starts at
// that RSP (the slot below it is dead).
std::arch::global_asm!(
    ".globl ps5cpu_exec_switch",
    "ps5cpu_exec_switch:",
    "mov rax, [rdi + 0x38]",
    "sub rax, 8",
    "mov rcx, [rdi + 0x80]",
    "mov [rax], rcx",
    "mov rsp, rax",
    "mov rax, [rdi + 0x00]",
    "mov rbx, [rdi + 0x08]",
    "mov rcx, [rdi + 0x10]",
    "mov rdx, [rdi + 0x18]",
    "mov rsi, [rdi + 0x20]",
    "mov rbp, [rdi + 0x30]",
    "mov r8,  [rdi + 0x40]",
    "mov r9,  [rdi + 0x48]",
    "mov r10, [rdi + 0x50]",
    "mov r11, [rdi + 0x58]",
    "mov r12, [rdi + 0x60]",
    "mov r13, [rdi + 0x68]",
    "mov r14, [rdi + 0x70]",
    "mov r15, [rdi + 0x78]",
    "mov rdi, [rdi + 0x28]",
    "ret",
    // ps5cpu_call_on_stack(arg, f, stack): run f(arg) on `stack`.
    ".globl ps5cpu_call_on_stack",
    "ps5cpu_call_on_stack:",
    "push rbp",
    "mov rbp, rsp",
    "mov rsp, rdx",
    "and rsp, -16",
    "call rsi",
    "mov rsp, rbp",
    "pop rbp",
    "ret",
);

// Direct import entry.
//
// An import stub loads its own address into r11 and jumps here with the
// guest frame exactly as the caller left it ([rsp] = return address). The
// entry switches to the host stack of the innermost guest call (published
// per thread in `ps5cpu_tls_host_stack`), records the complete guest state
// in a frame laid out like a signal context (`gregs` + FXSAVE image), and
// calls `ps5cpu_import_dispatch`. On return the (possibly updated) state is
// loaded back and the guest resumes at the saved RIP/RSP. When the signal
// hook replaced the context at the import-return safe point, every register
// must be restored exactly, so the entry raises a trap at
// `ps5cpu_import_restore` and the signal handler installs the frame.
//
// Frame (host stack, 64-byte aligned): [0, 512) FXSAVE, [512, 696) gregs.
/// Body of the direct import entry after the host stack pointer of the
/// innermost guest call has been loaded into r10.
macro_rules! import_entry_body {
    () => {
        concat!(
            "xchg rsp, r10\n",
            "and rsp, -64\n",
            "sub rsp, 768\n",
            "mov [rsp + 512 + 0], r8\n",
            "mov [rsp + 512 + 8], r9\n",
            "mov qword ptr [rsp + 512 + 16], 0\n",
            "mov [rsp + 512 + 24], r11\n",
            "mov [rsp + 512 + 32], r12\n",
            "mov [rsp + 512 + 40], r13\n",
            "mov [rsp + 512 + 48], r14\n",
            "mov [rsp + 512 + 56], r15\n",
            "mov [rsp + 512 + 64], rdi\n",
            "mov [rsp + 512 + 72], rsi\n",
            "mov [rsp + 512 + 80], rbp\n",
            "mov [rsp + 512 + 88], rbx\n",
            "mov [rsp + 512 + 96], rdx\n",
            "mov [rsp + 512 + 104], rax\n",
            "mov [rsp + 512 + 112], rcx\n",
            "mov [rsp + 512 + 120], r10\n",
            "mov [rsp + 512 + 128], r11\n",
            "pushfq\n",
            "pop qword ptr [rsp + 512 + 136]\n",
            "mov qword ptr [rsp + 512 + 144], 0x33\n",
            // Floating-point state: the parts host code may change (XMM, MXCSR
            // x87 control word) in FXSAVE layout; ST0-7 are left untouched.
            "fnstcw [rsp + 0]\n",
            "mov dword ptr [rsp + 2], 0\n",
            "stmxcsr [rsp + 24]\n",
            "mov dword ptr [rsp + 28], 0xFFFF\n",
            // Not an XSAVE frame (no software-reserved magic).
            "mov dword ptr [rsp + 464], 0\n",
            "movaps [rsp + 160], xmm0\n",
            "movaps [rsp + 176], xmm1\n",
            "movaps [rsp + 192], xmm2\n",
            "movaps [rsp + 208], xmm3\n",
            "movaps [rsp + 224], xmm4\n",
            "movaps [rsp + 240], xmm5\n",
            "movaps [rsp + 256], xmm6\n",
            "movaps [rsp + 272], xmm7\n",
            "movaps [rsp + 288], xmm8\n",
            "movaps [rsp + 304], xmm9\n",
            "movaps [rsp + 320], xmm10\n",
            "movaps [rsp + 336], xmm11\n",
            "movaps [rsp + 352], xmm12\n",
            "movaps [rsp + 368], xmm13\n",
            "movaps [rsp + 384], xmm14\n",
            "movaps [rsp + 400], xmm15\n",
            "cld\n",
            "lea rdi, [rsp + 512]\n",
            "mov rsi, rsp\n",
            "mov rdx, r11\n",
            "call {dispatch}\n",
            "test eax, eax\n",
            "jnz 2f\n",
            "fldcw [rsp + 0]\n",
            "ldmxcsr [rsp + 24]\n",
            "movaps xmm0, [rsp + 160]\n",
            "movaps xmm1, [rsp + 176]\n",
            "movaps xmm2, [rsp + 192]\n",
            "movaps xmm3, [rsp + 208]\n",
            "movaps xmm4, [rsp + 224]\n",
            "movaps xmm5, [rsp + 240]\n",
            "movaps xmm6, [rsp + 256]\n",
            "movaps xmm7, [rsp + 272]\n",
            "movaps xmm8, [rsp + 288]\n",
            "movaps xmm9, [rsp + 304]\n",
            "movaps xmm10, [rsp + 320]\n",
            "movaps xmm11, [rsp + 336]\n",
            "movaps xmm12, [rsp + 352]\n",
            "movaps xmm13, [rsp + 368]\n",
            "movaps xmm14, [rsp + 384]\n",
            "movaps xmm15, [rsp + 400]\n",
            "push qword ptr [rsp + 512 + 136]\n",
            "popfq\n",
            "mov r8, [rsp + 512 + 0]\n",
            "mov r9, [rsp + 512 + 8]\n",
            "mov r10, [rsp + 512 + 16]\n",
            "mov r12, [rsp + 512 + 32]\n",
            "mov r13, [rsp + 512 + 40]\n",
            "mov r14, [rsp + 512 + 48]\n",
            "mov r15, [rsp + 512 + 56]\n",
            "mov rdi, [rsp + 512 + 64]\n",
            "mov rsi, [rsp + 512 + 72]\n",
            "mov rbp, [rsp + 512 + 80]\n",
            "mov rbx, [rsp + 512 + 88]\n",
            "mov rdx, [rsp + 512 + 96]\n",
            "mov rax, [rsp + 512 + 104]\n",
            "mov rcx, [rsp + 512 + 112]\n",
            "mov r11, [rsp + 512 + 128]\n",
            "mov rsp, [rsp + 512 + 120]\n",
            "jmp r11\n",
            "2:\n",
            "fxrstor64 [rsp]\n",
            "lea rdi, [rsp + 512]\n",
            ".globl ps5cpu_import_restore\n",
            "ps5cpu_import_restore:\n",
            "int3\n",
            "ud2\n",
        )
    };
}

#[cfg(unix)]
std::arch::global_asm!(
    ".pushsection .tbss,\"awT\",@nobits",
    ".p2align 3",
    "ps5cpu_tls_host_stack:",
    ".zero 8",
    ".popsection",
    ".globl ps5cpu_host_stack_get",
    "ps5cpu_host_stack_get:",
    "mov rax, qword ptr fs:[ps5cpu_tls_host_stack@tpoff]",
    "ret",
    ".globl ps5cpu_host_stack_set",
    "ps5cpu_host_stack_set:",
    "mov qword ptr fs:[ps5cpu_tls_host_stack@tpoff], rdi",
    "ret",
    ".globl ps5cpu_import_entry",
    ".p2align 4",
    "ps5cpu_import_entry:",
    "mov r10, qword ptr fs:[ps5cpu_tls_host_stack@tpoff]",
    import_entry_body!(),
    dispatch = sym ps5cpu_import_dispatch,
);

// Windows: GS is the TEB, so the host stack of the innermost guest call
// lives in a TEB TLS slot whose offset `win::HOST_STACK_SLOT` holds.
#[cfg(windows)]
std::arch::global_asm!(
    ".globl ps5cpu_host_stack_get",
    "ps5cpu_host_stack_get:",
    "mov eax, dword ptr [rip + {slot}]",
    "mov rax, qword ptr gs:[rax]",
    "ret",
    ".globl ps5cpu_host_stack_set",
    "ps5cpu_host_stack_set:",
    "mov eax, dword ptr [rip + {slot}]",
    "mov qword ptr gs:[rax], rdi",
    "ret",
    ".globl ps5cpu_import_entry",
    ".p2align 4",
    "ps5cpu_import_entry:",
    "mov r10d, dword ptr [rip + {slot}]",
    "mov r10, qword ptr gs:[r10]",
    import_entry_body!(),
    dispatch = sym ps5cpu_import_dispatch,
    slot = sym crate::win::HOST_STACK_SLOT,
);

extern "sysv64" {
    fn ps5cpu_exec_switch(regs: *const GuestRegs) -> !;
    pub(crate) fn ps5cpu_call_on_stack(arg: *mut crate::ctx::c_void, f: extern "sysv64" fn(*mut crate::ctx::c_void), stack: u64);
    fn ps5cpu_host_stack_get() -> u64;
    pub(crate) fn ps5cpu_host_stack_set(stack: u64);
}

extern "C" {
    fn ps5cpu_import_entry();
    fn ps5cpu_import_restore();
}

/// Address of the INT3 of the exact-restore path of the import entry.
#[cfg(windows)]
pub(crate) fn import_restore_address() -> u64 {
    ps5cpu_import_restore as *const () as u64
}

/// Address guest import stubs jump to (see [`import_stub_code`]).

pub fn import_entry_address() -> u64 {
    ps5cpu_import_entry as *const () as u64
}

/// Machine code of a direct import stub placed at `stub`: it loads its own
/// address into r11 and jumps through the 64-bit entry pointer stored at
/// `entry_slot` (which must hold [`import_entry_address`] and lie within
/// ±2 GiB of the stub). 16 bytes, INT3-padded.
pub fn import_stub_code(stub: u64, entry_slot: u64) -> [u8; 16] {
    let mut c = [0xCCu8; 16];
    // lea r11, [rip - 7]
    c[..7].copy_from_slice(&[0x4C, 0x8D, 0x1D, 0xF9, 0xFF, 0xFF, 0xFF]);
    // jmp qword ptr [rip + disp32]
    let disp = entry_slot.wrapping_sub(stub + 13) as i64;
    assert!(i32::try_from(disp).is_ok(), "import entry slot out of range");
    c[7..9].copy_from_slice(&[0xFF, 0x25]);
    c[9..13].copy_from_slice(&(disp as i32).to_le_bytes());
    c
}

/// Rust side of the direct import entry: dispatch the import whose stub is
/// at `stub` with the guest state in `g`/`fp`. Returns 1 when the state was
/// replaced at the safe point and must be restored exactly.
extern "sysv64" fn ps5cpu_import_dispatch(g: *mut crate::ctx::greg_t, fp: *mut crate::ctx::FpState, stub: u64) -> u64 {
    let g = unsafe { &mut *(g as *mut [crate::ctx::greg_t; 23]) };
    let t = exec();
    let saved = (t.trap_fp, t.trap_fp_partial, t.trap_frame_low, t.trap_gregs, t.context_replaced);
    t.trap_fp = fp;
    t.trap_fp_partial = true;
    // Nothing but the return address was pushed on the guest stack.
    t.trap_frame_low = g[crate::ctx::REG_RSP as usize] as u64;
    t.trap_gregs = g.as_mut_ptr();
    t.context_replaced = false;
    dispatch_import(g, stub);
    crate::signal::import_safe_point(g.as_mut_ptr());
    let t = exec();
    let replaced = t.context_replaced;
    (t.trap_fp, t.trap_fp_partial, t.trap_frame_low, t.trap_gregs, t.context_replaced) = saved;
    replaced as u64
}

/// Install a frame saved by the direct import entry into a signal context
/// (the exact-restore path of `ps5cpu_import_entry`).
fn restore_import_frame(uc: *mut crate::ctx::ucontext_t) {
    let g = gregs(uc);
    let frame = g[crate::ctx::REG_RDI as usize] as *const [crate::ctx::greg_t; 23];
    let saved = unsafe { &*frame };
    // Every general register, RIP and RSP; flags limited to the user bits.
    for r in 0..=crate::ctx::REG_RIP as usize {
        g[r] = saved[r];
    }
    const USER_FLAGS: i64 = 0xCD5 | (1 << 10) | (1 << 18) | (1 << 21);
    let efl = crate::ctx::REG_EFL as usize;
    g[efl] = (g[efl] & !USER_FLAGS) | (saved[efl] & USER_FLAGS);
}

#[inline(always)]
fn read_rsp() -> u64 {
    let rsp: u64;
    unsafe { std::arch::asm!("mov {}, rsp", out(reg) rsp, options(nomem, nostack, preserves_flags)) };
    rsp
}

/// Run guest code with `regs` until it leaves through the sentinel or a
/// thread exit. May be called from HLE callbacks (nested guest calls).
#[inline(never)]
pub fn enter_guest(regs: &GuestRegs) -> GuestExit {
    #[cfg(windows)]
    crate::win::init();
    let regs = *regs;
    let t = exec();
    assert!(t.depth < MAX_NESTING, "guest call nesting too deep");
    let idx = t.depth;
    t.depth += 1;
    // Leave room for this frame's locals before the handler's stack area.
    t.frames[idx].host_rsp = read_rsp() - 0x400;
    let frame = std::ptr::addr_of_mut!(t.frames[idx]);
    if unsafe { sigsetjmp(std::ptr::addr_of_mut!((*frame).jmp), 1) } == 0 {
        // Imports made by this guest call run on the host stack below here.
        unsafe { ps5cpu_host_stack_set((*frame).host_rsp) };
        unsafe { ps5cpu_exec_switch(&regs) };
    }
    let t = exec();
    t.depth = idx;
    let outer = if idx > 0 { t.frames[idx - 1].host_rsp } else { 0 };
    unsafe { ps5cpu_host_stack_set(outer) };
    t.frames[idx].exit
}

/// Host stack top used by imports of the innermost guest call on this
/// thread (0 outside guest code).
pub fn import_host_stack() -> u64 {
    unsafe { ps5cpu_host_stack_get() }
}

/// Leave the innermost guest call with `exit` (called on the guest thread,
/// typically from inside an HLE callback).
pub fn leave_guest(exit: GuestExit) -> ! {
    let t = exec();
    assert!(t.depth > 0, "leave_guest outside guest code");
    let f = &mut t.frames[t.depth - 1];
    f.exit = exit;
    unsafe { siglongjmp(&mut f.jmp, 1) }
}

/// Terminate the current guest thread (pthread_exit): unwind every nested
/// guest call back to the thread's outermost entry.
pub fn exit_guest_thread(value: u64) -> ! {
    let t = exec();
    assert!(t.depth > 0, "exit_guest_thread outside guest code");
    t.depth = 1;
    leave_guest(GuestExit::ThreadExit(value))
}

/// Register state for calling `func(args...)` with its stack ending at
/// `stack_top`: rbp points at a zeroed 16-byte root frame (frame-pointer
/// walks stop at its null return address) and the return address is the
/// sentinel, with rsp % 16 == 8 at entry.
pub fn call_regs(func: u64, stack_top: u64, args: [u64; 6]) -> GuestRegs {
    let root = (stack_top & !0xF) - 0x10;
    let sp = root - 8;
    unsafe {
        std::ptr::write(root as *mut u64, 0);
        std::ptr::write((root + 8) as *mut u64, 0);
        std::ptr::write(sp as *mut u64, HOST_RETURN_SENTINEL);
    }
    GuestRegs {
        rip: func,
        rsp: sp,
        rbp: root,
        rdi: args[0],
        rsi: args[1],
        rdx: args[2],
        rcx: args[3],
        r8: args[4],
        r9: args[5],
        ..Default::default()
    }
}

/// Call a guest function from an HLE callback on the current guest thread,
/// using the guest stack below the interrupted frame. Returns RAX.
pub fn call_guest_nested(func: u64, args: [u64; 6]) -> u64 {
    assert!(exec().trap_rsp != 0, "nested guest call outside an HLE dispatch");
    match enter_guest(&call_regs(func, nested_stack_top(), args)) {
        GuestExit::Returned(rax) => rax,
        GuestExit::ThreadExit(v) => exit_guest_thread(v),
    }
}

/// Guest stack top for a nested call from the current HLE dispatch.
pub(crate) fn nested_stack_top() -> u64 {
    let rsp = exec().trap_rsp;
    // Skip the red zone and the signal frame the kernel pushed onto the
    // guest stack for the import trap (ucontext, siginfo, xsave area):
    // writing over it would corrupt the registers restored on return.
    let low = exec().trap_frame_low;
    if low != 0 && low < rsp { (rsp - 0x200).min(low - 0x100) } else { rsp - 0x200 }
}

// ---------------------------------------------------------------------------
// Signal handling
// ---------------------------------------------------------------------------

struct HandlerArgs {
    signum: crate::ctx::c_int,
    info: *mut crate::ctx::siginfo_t,
    uc: *mut crate::ctx::ucontext_t,
}

fn gregs(uc: *mut crate::ctx::ucontext_t) -> &'static mut [crate::ctx::greg_t; 23] {
    unsafe { &mut (*uc).uc_mcontext.gregs }
}

fn read_regs(g: &[crate::ctx::greg_t; 23]) -> [u64; 16] {
    use crate::ctx::*;
    [
        g[REG_RAX as usize] as u64,
        g[REG_RCX as usize] as u64,
        g[REG_RDX as usize] as u64,
        g[REG_RBX as usize] as u64,
        g[REG_RSP as usize] as u64,
        g[REG_RBP as usize] as u64,
        g[REG_RSI as usize] as u64,
        g[REG_RDI as usize] as u64,
        g[REG_R8 as usize] as u64,
        g[REG_R9 as usize] as u64,
        g[REG_R10 as usize] as u64,
        g[REG_R11 as usize] as u64,
        g[REG_R12 as usize] as u64,
        g[REG_R13 as usize] as u64,
        g[REG_R14 as usize] as u64,
        g[REG_R15 as usize] as u64,
    ]
}

/// Consulted before a protection fault is reported: `(fault address,
/// is_write) -> retry`. Used by the GPU memory tracker, which write- or
/// read-protects guest pages it mirrors. Runs inside the signal handler:
/// it must not allocate or take locks the faulting thread may hold.
pub type FaultHook = fn(u64, bool) -> bool;

static FAULT_HOOK: AtomicUsize = AtomicUsize::new(0);

/// Install the protection-fault hook (replaces any previous one).
pub fn set_fault_hook(hook: FaultHook) {
    FAULT_HOOK.store(hook as usize, Ordering::Release);
}

/// Describes a guest address for the report of a fatal guest fault (the
/// memory manager's view of it); None for addresses it does not know.
pub type DescribeHook = fn(u64) -> Option<String>;

static DESCRIBE_HOOK: AtomicUsize = AtomicUsize::new(0);

/// Install the address description used by fatal guest fault reports.
pub fn set_describe_hook(hook: DescribeHook) {
    DESCRIBE_HOOK.store(hook as usize, Ordering::Release);
}

/// Faults resolved by the hook (diagnostics).
pub static HOOK_RESOLVED_FAULTS: AtomicU64 = AtomicU64::new(0);

/// Ask the fault hook about a SIGSEGV/SIGBUS. Only access-rights faults on
/// mapped pages (`SEGV_ACCERR`) are offered to it.
fn try_fault_hook(signum: crate::ctx::c_int, info: *mut crate::ctx::siginfo_t, uc: *mut crate::ctx::ucontext_t) -> bool {
    if signum != crate::ctx::SIGSEGV {
        return false;
    }
    let raw = FAULT_HOOK.load(Ordering::Acquire);
    if raw == 0 {
        return false;
    }
    // SAFETY: kernel-provided signal info/context.
    let (code, addr, err) = unsafe { ((*info).si_code, (*info).si_addr() as u64, (*uc).uc_mcontext.gregs[crate::ctx::REG_ERR as usize] as u64) };
    const SEGV_ACCERR: crate::ctx::c_int = 2;
    if code != SEGV_ACCERR {
        return false;
    }
    // SAFETY: the value was stored from a `FaultHook`.
    let hook: FaultHook = unsafe { std::mem::transmute::<usize, FaultHook>(raw) };
    // SAFETY: kernel-provided signal context.
    ps5vm::fault_site::set(unsafe { (*uc).uc_mcontext.gregs[crate::ctx::REG_RIP as usize] as u64 });
    // Page-fault error code bit 1: the access was a write.
    let resolved = hook(addr, err & 2 != 0);
    if resolved {
        HOOK_RESOLVED_FAULTS.fetch_add(1, Ordering::Relaxed);
    }
    resolved
}

pub(crate) extern "C" fn guest_signal_handler(signum: crate::ctx::c_int, info: *mut crate::ctx::siginfo_t, uc: *mut crate::ctx::c_void) {
    let uc = uc as *mut crate::ctx::ucontext_t;
    #[cfg(windows)]
    if signum == crate::ctx::SIGTRAP {
        // A thread-pointer access too short for a jump was replaced by
        // INT3: continue at its rewritten copy.
        let at = (gregs(uc)[crate::ctx::REG_RIP as usize] as u64).wrapping_sub(1);
        if let Some(target) = crate::tls_patch::trap_target(at) {
            gregs(uc)[crate::ctx::REG_RIP as usize] = target as i64;
            return;
        }
    }

    if signum == crate::ctx::SIGTRAP && gregs(uc)[crate::ctx::REG_RIP as usize] as u64 == ps5cpu_import_restore as *const () as u64 + 1 {
        // Runs on the host stack below the saved frame: install it in place.
        restore_import_frame(uc);
        return;
    }
    if signum == crate::ctx::SIGILL {
        // An instruction of the console CPU the host lacks: emulate it in
        // place and resume after it.
        if let Some(mut ctx) = unsafe { crate::x64emu::SignalContext::new(uc) } {
            if crate::x64emu::try_emulate(&mut ctx) {
                return;
            }
        }
    }
    let t = exec();
    if t.depth == 0 {
        // Not inside guest code on this thread: host code (an emulator
        // thread or HLE code) touched memory the GPU tracker protects, or a
        // host fault.
        if try_fault_hook(signum, info, uc) {
            return;
        }
        host_fault(signum);
        return;
    }
    let stack = t.frames[t.depth - 1].host_rsp;
    let rsp = read_rsp();
    if rsp < stack && stack - rsp < HOST_THREAD_STACK as u64 {
        // Already on this thread's host stack (an HLE callback faulted):
        // switching to `stack` would clobber the live HLE frames.
        if try_fault_hook(signum, info, uc) {
            return;
        }
    }
    let mut args = HandlerArgs { signum, info, uc };
    unsafe { ps5cpu_call_on_stack(&mut args as *mut _ as *mut crate::ctx::c_void, handler_body, stack) };
}

/// Restore the default action so returning re-raises the fault (core dump).
#[cfg(unix)]
fn host_fault(signum: crate::ctx::c_int) {
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = libc::SIG_DFL;
        libc::sigaction(signum, &sa, std::ptr::null_mut());
    }
}

/// Let the exception go on to the next handler (the process crashes with
/// the host's report).
#[cfg(windows)]
fn host_fault(_signum: crate::ctx::c_int) {
    crate::win::set_host_fault();
}

extern "sysv64" fn handler_body(arg: *mut crate::ctx::c_void) {
    let a = unsafe { &mut *(arg as *mut HandlerArgs) };
    let g = gregs(a.uc);
    let rip = g[crate::ctx::REG_RIP as usize] as u64;

    if a.signum == crate::ctx::SIGTRAP {
        let saved_fp = (exec().trap_fp, exec().trap_fp_partial);
        let saved_low = exec().trap_frame_low;
        let saved_gregs = exec().trap_gregs;
        exec().trap_fp = unsafe { (*a.uc).uc_mcontext.fpregs } as *const crate::ctx::FpState;
        exec().trap_fp_partial = false;
        exec().trap_frame_low = trap_frame_low(a.uc, a.info);
        exec().trap_gregs = g.as_mut_ptr();
        let handled = dispatch_import(g, rip.wrapping_sub(1));
        if handled {
            crate::signal::import_safe_point(g.as_mut_ptr());
        }
        (exec().trap_fp, exec().trap_fp_partial) = saved_fp;
        exec().trap_frame_low = saved_low;
        exec().trap_gregs = saved_gregs;
        if !handled {
            // An INT3 that is not an import stub: a guest `int3`, or a jump
            // into the padding of a stub. Its return address is unknown.
            report_guest_fault(a.signum, g, rip);
            ps5vm::os::exit_now(128 + a.signum);
        }
        return;
    }
    if rip == HOST_RETURN_SENTINEL {
        leave_guest(GuestExit::Returned(g[crate::ctx::REG_RAX as usize] as u64));
    }
    // A guest (or HLE) access to a page the GPU memory tracker protects.
    if try_fault_hook(a.signum, a.info, a.uc) {
        return;
    }
    let fault = unsafe { (*a.info).si_addr() as u64 };
    if rip >= 0x10000 && guest_code_owner(rip).is_none() && !is_guest_stub(rip) {
        // The fault happened in host code reached from guest context
        // (an HLE callback): that is an emulator bug, not a guest crash.
        log::error!("[exec] host fault signal {} at rip={rip:#x} addr={fault:#x}", a.signum);
        host_fault(a.signum);
        return;
    }
    report_guest_fault(a.signum, g, fault);
    ps5vm::os::exit_now(128 + a.signum);
}

/// Lowest address of the host's record of a trap on the guest stack.
#[cfg(unix)]
pub(crate) fn trap_frame_low(uc: *mut crate::ctx::ucontext_t, info: *mut crate::ctx::siginfo_t) -> u64 {
    (uc as u64).min(info as u64)
}

/// Windows dispatches exceptions on the faulting stack: the exception
/// record, `CONTEXT` and the handler frames sit below the guest RSP.
#[cfg(windows)]
pub(crate) fn trap_frame_low(_uc: *mut crate::ctx::ucontext_t, _info: *mut crate::ctx::siginfo_t) -> u64 {
    crate::win::trap_frame_low()
}

fn is_guest_stub(rip: u64) -> bool {
    (0x7000_0000_0000..0x7000_1000_0000).contains(&rip)
}

/// Whether `addr` is guest code (an image or an import stub). Does not
/// allocate or block, so it is usable from an asynchronous signal handler;
/// answers false while the code-range table is being updated.
pub(crate) fn is_guest_code_addr(addr: u64) -> bool {
    if is_guest_stub(addr) {
        return true;
    }
    match GUEST_CODE.try_read() {
        Ok(v) => v.iter().any(|(s, e, _)| addr >= *s && addr < *e),
        Err(_) => false,
    }
}

/// Run the HLE callback of the import stub that trapped at `stub`. False
/// when `stub` is not an import stub: the guest state is left unchanged.
fn dispatch_import(g: &mut [crate::ctx::greg_t; 23], stub: u64) -> bool {
    let regs = read_regs(g);
    let rsp = regs[4];
    let t = exec();
    let saved_trap_rsp = t.trap_rsp;
    let saved_tail = t.tail_call;
    let saved_ret = (t.ret_rdx.take(), t.ret_xmm0.take(), t.ret_st0.take());
    t.trap_rsp = rsp;
    t.tail_call = 0;
    t.imports += 1;
    count_import();
    let rax = match DISPATCHER.get().and_then(|d| d.dispatch_at(stub, &regs, rsp, stub)) {
        Some(v) => v,
        None => {
            let t = exec();
            t.trap_rsp = saved_trap_rsp;
            t.tail_call = saved_tail;
            (t.ret_rdx, t.ret_xmm0, t.ret_st0) = saved_ret;
            return false;
        }
    };
    let t = exec();
    t.trap_rsp = saved_trap_rsp;
    let (rdx, xmm0, st0) = std::mem::replace(&mut (t.ret_rdx, t.ret_xmm0, t.ret_st0), saved_ret);
    if let Some(v) = rdx {
        g[crate::ctx::REG_RDX as usize] = v as i64;
    }
    if let Some(v) = xmm0 {
        set_saved_xmm0(t.trap_fp, v);
    }
    if let Some(v) = st0 {
        apply_return_st0(t.trap_fp, t.trap_fp_partial, &v);
    }
    let tail = std::mem::replace(&mut t.tail_call, saved_tail);
    if tail != 0 {
        g[crate::ctx::REG_RIP as usize] = tail as i64;
        g[crate::ctx::REG_RAX as usize] = rax as i64;
        return true;
    }
    // Emulate `ret` from the stub.
    let ret = unsafe { std::ptr::read(rsp as *const u64) };
    g[crate::ctx::REG_RSP as usize] = (rsp + 8) as i64;
    g[crate::ctx::REG_RIP as usize] = ret as i64;
    g[crate::ctx::REG_RAX as usize] = rax as i64;
    true
}

/// Store XMM0 into the saved FP state of the import being dispatched (a
/// signal frame or the direct entry's frame), restored on return.
fn set_saved_xmm0(fp: *const crate::ctx::FpState, value: u128) {
    if fp.is_null() {
        return;
    }
    let fp = fp as *mut u8;
    unsafe {
        std::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), fp.add(160), 16);
        // An XSAVE signal frame restores SSE state only when its component
        // bit is set.
        let magic = std::ptr::read_unaligned(fp.add(464) as *const u32);
        if magic == 0x4650_5853 {
            let bv = fp.add(512) as *mut u64;
            std::ptr::write_unaligned(bv, std::ptr::read_unaligned(bv) | 2);
        }
    }
}

fn report_guest_fault(signum: crate::ctx::c_int, g: &[crate::ctx::greg_t; 23], fault: u64) {
    let regs = read_regs(g);
    let rip = g[crate::ctx::REG_RIP as usize] as u64;
    let owner = |a: u64| match guest_code_owner(a) {
        Some((base, name)) => format!("{a:#x} ({name}+{:#x})", a - base),
        None => format!("{a:#x}"),
    };
    // An address in a red-zone copy also names the function it copies.
    #[cfg(windows)]
    let loc = |a: u64| match crate::redzone::original_of(a) {
        Some(f) => format!("{} in the copy of the function at {}", owner(a), owner(f)),
        None => owner(a),
    };
    #[cfg(unix)]
    let loc = owner;
    log::error!(
        "[exec] GUEST FAULT signal {} at {} addr={fault:#x} tid={:?}",
        signum,
        loc(rip),
        std::thread::current().name()
    );
    if signum == crate::ctx::SIGILL && is_guest_code_addr(rip) {
        // An instruction the host does not implement and nothing emulates.
        let len = if is_guest_code_addr(rip + 14) { 15 } else { (0x1000 - (rip & 0xFFF)).min(15) as usize };
        let code = unsafe { std::slice::from_raw_parts(rip as *const u8, len) };
        let insn = iced_x86::Decoder::with_ip(64, code, rip, iced_x86::DecoderOptions::NONE).decode();
        let bytes: Vec<String> = code[..insn.len().max(1)].iter().map(|b| format!("{b:02x}")).collect();
        log::error!(
            "[exec]   illegal instruction {:?} [{}] (needs {:?}; host: {})",
            insn.mnemonic(),
            bytes.join(" "),
            insn.cpuid_features(),
            crate::host_cpu::host().brand
        );
    }
    let names = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];
    let line: Vec<String> = names.iter().zip(regs.iter()).map(|(n, v)| format!("{n}={v:#x}")).collect();
    log::error!("[exec]   {}", line.join(" "));
    // The memory the registers point at, as the memory manager sees it
    // (the process is going away, so the report may allocate).
    let describe = DESCRIBE_HOOK.load(Ordering::Acquire);
    if describe != 0 {
        // SAFETY: only `set_describe_hook` stores a `DescribeHook` here.
        let hook: DescribeHook = unsafe { std::mem::transmute::<usize, DescribeHook>(describe) };
        for (n, &v) in names.iter().zip(regs.iter()) {
            if let Some(d) = hook(v) {
                log::error!("[exec]   [{n}] {v:#x}: {d}");
            }
        }
    }
    // Frame-pointer backtrace through guest frames.
    let mut rbp = regs[5];
    for depth in 0..32 {
        if !ps5vm::ranges::accessible(rbp, 16, false) {
            break;
        }
        let ret = unsafe { std::ptr::read((rbp + 8) as *const u64) };
        let next = unsafe { std::ptr::read(rbp as *const u64) };
        log::error!("[exec]   #{depth} {}", loc(ret));
        if next <= rbp {
            break;
        }
        rbp = next;
    }
}

#[cfg(windows)]
fn map_fixed_code(addr: u64, bytes: &[u8]) {
    if let Err(e) = ps5vm::os::map_anon_noreplace(addr, 0x1000, ps5vm::MemProtect::RWX) {
        panic!("cannot map guest helper page at {addr:#x}: {e}");
    }
    // SAFETY: the page was just mapped writable.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), addr as *mut u8, bytes.len()) };
    ps5vm::ranges::track(addr, 0x1000, ps5vm::MemProtect::RX);
}

#[cfg(unix)]
fn map_fixed_code(addr: u64, bytes: &[u8]) {
    unsafe {
        let p = libc::mmap(
            addr as *mut crate::ctx::c_void,
            0x1000,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED_NOREPLACE,
            -1,
            0,
        );
        if p == libc::MAP_FAILED || p as u64 != addr {
            panic!("cannot map guest helper page at {addr:#x}");
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), addr as *mut u8, bytes.len());
    }
    ps5vm::ranges::track(addr, 0x1000, ps5vm::MemProtect::RX);
}

/// Install the process-wide signal handlers and map the helper pages.
/// Must run once before any guest code.
pub fn init() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    #[cfg(windows)]
    ONCE.call_once(|| {
        crate::win::init();
        map_fixed_code(HOST_RETURN_SENTINEL, &crate::win::sentinel_code());
        map_fixed_code(EXIT_HANDLER_STUB, &[0x31, 0xC0, 0xC3]); // xor eax,eax; ret
    });
    #[cfg(unix)]
    ONCE.call_once(|| {
        map_fixed_code(HOST_RETURN_SENTINEL, &[0xF4, 0xCC]); // hlt
        map_fixed_code(EXIT_HANDLER_STUB, &[0x31, 0xC0, 0xC3]); // xor eax,eax; ret
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = guest_signal_handler as *const () as usize;
            sa.sa_flags = libc::SA_SIGINFO | libc::SA_NODEFER;
            libc::sigemptyset(&mut sa.sa_mask);
            for sig in [crate::ctx::SIGTRAP, crate::ctx::SIGSEGV, crate::ctx::SIGBUS, crate::ctx::SIGILL, crate::ctx::SIGFPE] {
                libc::sigaction(sig, &sa, std::ptr::null_mut());
            }
        }
        crate::signal::install();
    });
}

/// Map a guest stack of `size` bytes ending at `top` (read/write, tracked).
pub fn map_guest_stack(top: u64, size: u64) -> Result<(), String> {
    let base = top - size;
    #[cfg(windows)]
    {
        ps5vm::os::map_anon_noreplace(base, size, ps5vm::MemProtect::RW).map_err(|e| format!("stack map at {base:#x}+{size:#x}: {e}"))?;
        ps5vm::ranges::track(base, size, ps5vm::MemProtect::RW);
        Ok(())
    }
    #[cfg(unix)]
    {
    let p = unsafe {
        libc::mmap(
            base as *mut crate::ctx::c_void,
            size as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED_NOREPLACE | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    if p == libc::MAP_FAILED || p as u64 != base {
        return Err(format!("stack map at {base:#x}+{size:#x}: {}", std::io::Error::last_os_error()));
    }
    ps5vm::ranges::track(base, size, ps5vm::MemProtect::RW);
    Ok(())
    }
}

/// Release a guest stack mapped with `map_guest_stack`.
pub fn unmap_guest_stack(top: u64, size: u64) {
    let base = top - size;
    ps5vm::os::unmap(base, size);
    ps5vm::ranges::untrack(base, size);
}

/// Description of a new guest thread.
pub struct ThreadStart {
    pub name: String,
    pub entry: u64,
    pub arg: u64,
    pub stack_top: u64,
    pub tp: u64,
    /// Runs on the new host thread before guest code starts.
    pub on_start: Option<Box<dyn FnOnce() + Send>>,
}

/// Host stack for threads running guest code (HLE callbacks run on it).
const HOST_THREAD_STACK: usize = 16 * 1024 * 1024;

/// Start a guest thread on a new host thread. `on_exit` runs on that thread
/// after the guest entry returns or the thread exits.
pub fn spawn_guest_thread(
    start: ThreadStart,
    on_exit: Box<dyn FnOnce(GuestExit) + Send>,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name(start.name.clone())
        .stack_size(HOST_THREAD_STACK)
        .spawn(move || {
            let mut start = start;
            bind_thread_pointer(start.tp);
            if let Some(f) = start.on_start.take() {
                f();
            }
            let exit = enter_guest(&call_regs(start.entry, start.stack_top, [start.arg, 0, 0, 0, 0, 0]));
            on_exit(exit);
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    static PAGE: AtomicU64 = AtomicU64::new(0);
    static CALLS: AtomicU64 = AtomicU64::new(0);
    /// The tests that share [`PAGE`] run one at a time.
    static HOOK_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Unprotects the watched page on a write fault (what the GPU memory
    /// tracker does for a write-watched page).
    fn hook(addr: u64, is_write: bool) -> bool {
        let page = PAGE.load(Ordering::SeqCst);
        if !is_write || addr & !0xFFF != page {
            return false;
        }
        CALLS.fetch_add(1, Ordering::SeqCst);
        ps5vm::os::protect(page, 0x1000, ps5vm::MemProtect::RW)
    }

    fn map(prot: ps5vm::MemProtect) -> u64 {
        let p = ps5vm::os::map_anon(0x1000, ps5vm::MemProtect::RW).expect("map a test page");
        assert!(ps5vm::os::protect(p, 0x1000, prot));
        p
    }

    #[test]
    fn fault_hook_resolves_guest_and_host_write_faults() {
        let _one = HOOK_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        init();
        set_fault_hook(hook);
        let calls = CALLS.load(Ordering::SeqCst);

        // Host code (no guest frame on this thread) writing a protected page.
        let data = map(ps5vm::MemProtect::R);
        PAGE.store(data, Ordering::SeqCst);
        // SAFETY: the write faults and the hook makes the page writable.
        unsafe { std::ptr::write_volatile(data as *mut u32, 0x1234_5678) };
        assert_eq!(unsafe { std::ptr::read_volatile(data as *const u32) }, 0x1234_5678);
        assert_eq!(CALLS.load(Ordering::SeqCst), calls + 1);

        // Guest code: `mov byte [rdi], 0x42; ret` against a read-only page.
        let code = map(ps5vm::MemProtect::RWX);
        // SAFETY: writable code page owned by the test.
        unsafe { std::ptr::copy_nonoverlapping([0xC6u8, 0x07, 0x42, 0xC3].as_ptr(), code as *mut u8, 4) };
        add_guest_code_range(code, 0x1000, "fault-hook-test");
        let data = map(ps5vm::MemProtect::R);
        PAGE.store(data, Ordering::SeqCst);
        let top = 0x5F00_0000_0000u64;
        map_guest_stack(top, 0x10000).unwrap();
        let exit = enter_guest(&call_regs(code, top, [data, 0, 0, 0, 0, 0]));
        assert_eq!(exit, GuestExit::Returned(0));
        assert_eq!(unsafe { std::ptr::read_volatile(data as *const u8) }, 0x42);
        assert_eq!(CALLS.load(Ordering::SeqCst), calls + 2);
        assert!(HOOK_RESOLVED_FAULTS.load(Ordering::SeqCst) >= 2);
        unmap_guest_stack(top, 0x10000);
    }

    /// A leaf function keeps values in its red zone across a write that
    /// faults and is resolved by the hook.
    #[test]
    fn red_zone_survives_a_resolved_write_fault() {
        let _one = HOOK_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        init();
        set_fault_hook(hook);
        #[rustfmt::skip]
        let leaf: [u8; 48] = [
            0x48, 0x89, 0x74, 0x24, 0xF8,       // mov [rsp-0x08], rsi
            0x48, 0x89, 0x74, 0x24, 0xD0,       // mov [rsp-0x30], rsi
            0x48, 0x89, 0x74, 0x24, 0x80,       // mov [rsp-0x80], rsi
            0xC6, 0x07, 0x42,                   // mov byte [rdi], 0x42 (faults)
            0x31, 0xC0,                         // xor eax, eax
            0x48, 0x39, 0x74, 0x24, 0xF8,       // cmp [rsp-0x08], rsi
            0x75, 0x02, 0xFF, 0xC0,             // jne +2; inc eax
            0x48, 0x39, 0x74, 0x24, 0xD0,       // cmp [rsp-0x30], rsi
            0x75, 0x02, 0xFF, 0xC0,             // jne +2; inc eax
            0x48, 0x39, 0x74, 0x24, 0x80,       // cmp [rsp-0x80], rsi
            0x75, 0x02, 0xFF, 0xC0,             // jne +2; inc eax
            0xC3,                               // ret
        ];
        let code = map(ps5vm::MemProtect::RWX);
        // SAFETY: writable code page owned by the test.
        unsafe { std::ptr::copy_nonoverlapping(leaf.as_ptr(), code as *mut u8, leaf.len()) };
        add_guest_code_range(code, 0x1000, "red-zone-test");
        // What module loading does on Windows, where the exception frame is
        // built right below RSP: the function moves to a copy that runs
        // with RSP below its red zone.
        #[cfg(windows)]
        {
            let none = crate::code_patch::Replace { monitorx: false, sse4a: false };
            // SAFETY: the test's own code page; nothing runs it yet.
            let (c, _) = unsafe { crate::redzone::protect_functions(&[(code, leaf.len() as u64)], &[(code, 0x1000)], none) };
            assert_eq!(c.moved, 1);
        }
        let data = map(ps5vm::MemProtect::R);
        PAGE.store(data, Ordering::SeqCst);
        let top = 0x5F10_0000_0000u64;
        map_guest_stack(top, 0x10000).unwrap();
        let magic = 0x0123_4567_89AB_CDEFu64;
        let exit = enter_guest(&call_regs(code, top, [data, magic, 0, 0, 0, 0]));
        unmap_guest_stack(top, 0x10000);
        assert_eq!(unsafe { std::ptr::read_volatile(data as *const u8) }, 0x42);
        assert_eq!(exit, GuestExit::Returned(3), "red-zone slots intact after the fault (of 3)");
    }

    #[test]
    fn trap_outside_an_import_stub_is_not_dispatched() {
        // No dispatcher is installed in unit tests: every address is
        // "not an import stub". The guest state must stay as it was, with
        // no emulated `ret` through an unknown return address.
        let mut g: [crate::ctx::greg_t; 23] = [0; 23];
        g[crate::ctx::REG_RIP as usize] = 0x1234_5679;
        g[crate::ctx::REG_RSP as usize] = 0x7777_0000;
        g[crate::ctx::REG_RAX as usize] = 0x55;
        let before = g;
        set_return_xmm0(0xAB);
        assert!(!dispatch_import(&mut g, 0x1234_5678));
        assert_eq!(g, before);
        assert_eq!(exec().ret_xmm0, Some(0xAB), "the outer call's XMM0 return is kept");
        exec().ret_xmm0 = None;
    }
}
