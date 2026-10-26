// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libkernel process-level exports: process parameters, the runtime heap
//! hook, errno, dynamic symbol lookup, sleeping, process exit and system
//! information.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};
use std::sync::Mutex;
use std::time::Duration;

const OK: HleResult = 0;
const ERR_INVALID_ARGUMENT: HleResult = 0x8002_0016u32 as i32 as i64;
const ERR_MEMORY_FAULT: HleResult = 0x8002_000Eu32 as i32 as i64;

/// errno lives in the thread control block.
const TLS_ERRNO_OFFSET: u64 = 0x40;

/// Address of the main executable's PT_SCE_PROCPARAM segment.
static PROC_PARAM_ADDR: Mutex<u64> = Mutex::new(0);

/// Set the ProcParam address (called by the loader).
pub fn set_proc_param(addr: u64) {
    *PROC_PARAM_ADDR.lock().unwrap() = addr;
}

/// sceKernelGetProcParam() -> ProcParam*
fn cb_get_proc_param(_ctx: &mut CpuContext) -> HleResult {
    *PROC_PARAM_ADDR.lock().unwrap() as i64
}

/// The runtime heap table libc registers (malloc, free, ..., posix_memalign).
static HEAP_API: Mutex<u64> = Mutex::new(0);

/// _sceKernelRtldSetApplicationHeapAPI(void* api[])
fn cb_set_application_heap_api(ctx: &mut CpuContext) -> HleResult {
    *HEAP_API.lock().unwrap() = ctx.rdi();
    OK
}

/// Address of the calling thread's errno slot.
pub fn errno_address() -> u64 {
    let tp = ps5cpu::exec::current_thread_pointer();
    if tp == 0 {
        0
    } else {
        tp + TLS_ERRNO_OFFSET
    }
}

/// Give the calling test thread a guest thread pointer (a zeroed block of
/// the object region) so errno can be set and read; returns the errno
/// address. Call it on a thread of its own.
#[cfg(test)]
pub(crate) fn bind_test_errno(ctx: &CpuContext) -> u64 {
    let tcb = crate::pthread::alloc_guest_block(ctx, 0x100).expect("guest block");
    ps5cpu::exec::bind_thread_pointer(tcb);
    errno_address()
}

/// Set errno for the calling guest thread.
pub fn set_errno(ctx: &CpuContext, value: i32) {
    let addr = errno_address();
    if addr != 0 {
        ctx.write_u32(addr, value as u32);
    }
}

/// POSIX "other error" for codes outside the kernel errno range.
pub const POSIX_EOTHER: i32 = 1062;

/// Kernel error code (0x80020000 | errno) -> POSIX errno; 0 stays 0 and any
/// other value becomes EOTHER.
pub fn kernel_to_posix(r: HleResult) -> i32 {
    if r == 0 {
        return 0;
    }
    let v = r as i32;
    if v > 0x8002_0000u32 as i32 && v <= 0x8002_0065u32 as i32 {
        v - 0x8002_0000u32 as i32
    } else {
        POSIX_EOTHER
    }
}

/// Finish a POSIX-named call: 0 on success; otherwise set errno from the
/// kernel code and return -1.
pub fn posix_call(ctx: &CpuContext, r: HleResult) -> HleResult {
    if r == 0 {
        0
    } else {
        set_errno(ctx, kernel_to_posix(r));
        -1
    }
}

/// Like [`posix_call`] for calls returning a count: negative kernel codes
/// set errno and become -1, other values pass through.
pub fn posix_n_call(ctx: &CpuContext, r: HleResult) -> HleResult {
    if r < 0 {
        set_errno(ctx, kernel_to_posix(r));
        -1
    } else {
        r
    }
}

/// __error() -> int*
fn cb_error(_ctx: &mut CpuContext) -> HleResult {
    errno_address() as i64
}

fn cb_getpid(_ctx: &mut CpuContext) -> HleResult {
    std::process::id() as i64
}

/// Offset and size of the title id in the app info (9 characters and a
/// terminator).
const APP_INFO_TITLE_ID: u64 = 0x10;
const APP_INFO_TITLE_ID_BYTES: usize = 10;
/// The running title's app id (libSceSystemService reports the same).
const RUNNING_APP_ID: u32 = 1;
const KERNEL_EPERM: HleResult = 0x8002_0001u32 as i32 as i64;

/// sceKernelGetAppInfo(pid, info) for the given title id: only the app id
/// (+0x00) and the title id (+0x10) are written, as callers declare the
/// structure with different sizes. Another pid is EPERM; a NULL info is
/// accepted without writing.
fn get_app_info(ctx: &CpuContext, pid: i32, info: u64, title_id: Option<&str>) -> HleResult {
    if pid as u32 != std::process::id() {
        return KERNEL_EPERM;
    }
    if info == 0 {
        return 0;
    }
    let mut field = [0u8; APP_INFO_TITLE_ID_BYTES];
    match title_id {
        Some(t) => {
            let n = t.len().min(APP_INFO_TITLE_ID_BYTES - 1);
            field[..n].copy_from_slice(&t.as_bytes()[..n]);
        }
        None => log::warn!("[kernel] sceKernelGetAppInfo: the title id is unknown (no param.json)"),
    }
    if !ctx.write_u32(info, RUNNING_APP_ID) || !ctx.write_bytes(info + APP_INFO_TITLE_ID, &field) {
        return 0x8002_000Eu32 as i32 as i64;
    }
    0
}

fn cb_get_app_info(ctx: &mut CpuContext) -> HleResult {
    let title = crate::app_info::title_id();
    get_app_info(ctx, ctx.rdi() as i32, ctx.rsi(), title.as_deref())
}

/// System software version reported to titles: 01.000.000 (0xMMmmmppp).
/// The console's firmware version is not modelled.
const SYSTEM_SW_VERSION: u32 = 0x0100_0000;
const SYSTEM_SW_VERSION_STRING: &[u8] = b"01.000.000";
/// `{u64 size; char versionString[0x1C]; u32 version}`.
const SYSTEM_SW_VERSION_SIZE: u64 = 0x28;

/// sceKernelGetSystemSwVersion / sceKernelGetProsperoSystemSwVersion(info):
/// the caller sets `size` to 0x28; another size is EINVAL and nothing is
/// written, NULL is EFAULT. The string is zero-padded; size stays.
fn get_system_sw_version(ctx: &CpuContext, info: u64) -> HleResult {
    const KERNEL_EFAULT: HleResult = 0x8002_000Eu32 as i32 as i64;
    const KERNEL_EINVAL: HleResult = 0x8002_0016u32 as i32 as i64;
    if info == 0 {
        return KERNEL_EFAULT;
    }
    match ctx.read_u64(info) {
        Some(SYSTEM_SW_VERSION_SIZE) => {}
        Some(_) => return KERNEL_EINVAL,
        None => return KERNEL_EFAULT,
    }
    let mut text = [0u8; 0x1C];
    text[..SYSTEM_SW_VERSION_STRING.len()].copy_from_slice(SYSTEM_SW_VERSION_STRING);
    if !ctx.write_bytes(info + 8, &text) || !ctx.write_u32(info + 0x24, SYSTEM_SW_VERSION) {
        return KERNEL_EFAULT;
    }
    0
}

fn cb_get_system_sw_version(ctx: &mut CpuContext) -> HleResult {
    get_system_sw_version(ctx, ctx.rdi())
}

/// sceKernelGetAllowedSdkVersionOnSystem(u32*): the system software
/// version with the 12 patch bits set; NULL is EINVAL.
fn cb_get_allowed_sdk_version_on_system(ctx: &mut CpuContext) -> HleResult {
    const KERNEL_EINVAL: HleResult = 0x8002_0016u32 as i32 as i64;
    let out = ctx.rdi();
    if out == 0 {
        return KERNEL_EINVAL;
    }
    if ctx.write_u32(out, SYSTEM_SW_VERSION | 0xFFF) { 0 } else { 0x8002_000Eu32 as i32 as i64 }
}

/// setsid(): -1 with EPERM, as FreeBSD for a process group leader (the
/// guest is taken to lead its own group, as `kill` treats it).
fn cb_setsid(ctx: &mut CpuContext) -> HleResult {
    set_errno(ctx, 1);
    -1
}

/// The process's file creation mask (FreeBSD starts at 022).
static CREATION_MASK: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0o022);

/// umask(mask) -> the previous mask; the new one keeps the bits up to
/// 07777. Files are created through the host, which applies its own mask.
fn cb_umask(ctx: &mut CpuContext) -> HleResult {
    CREATION_MASK.swap(ctx.rdi() as u32 & 0o7777, std::sync::atomic::Ordering::Relaxed) as HleResult
}

/// getppid(): the parent of the process, as getpid reports the process.
fn cb_getppid(_ctx: &mut CpuContext) -> HleResult {
    // SAFETY: getppid has no preconditions.
    unsafe { libc::getppid() as i64 }
}

/// _exit(status): terminate the process.
fn cb_exit(ctx: &mut CpuContext) -> HleResult {
    let status = ctx.rdi() as i32;
    log::info!("[kernel] _exit({status})");
    ps5vm::shutdown::exit(status)
}

/// sceKernelDlsym(handle, symbol, void** addr): handle 0 is the main
/// program. A NULL output or symbol is EFAULT, an unknown module or symbol
/// ESRCH.
///
/// A failed lookup leaves `*addr` untouched: callers pre-store a fallback
/// there (IL2CPP stores its own `(align, size)` allocator before probing
/// for the optional `scriptingGetMem` hook, which no image exports, and
/// calls whatever the slot holds afterwards), so no name aliasing is done
/// here either.
fn cb_dlsym(ctx: &mut CpuContext) -> HleResult {
    const KERNEL_ESRCH: HleResult = 0x8002_0003u32 as i32 as i64;
    const KERNEL_EFAULT: HleResult = 0x8002_000Eu32 as i32 as i64;
    let (handle, name_ptr, out) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    if out == 0 {
        return KERNEL_EFAULT;
    }
    let Some(name) = (name_ptr != 0).then(|| ctx.read_cstring(name_ptr, 512)).flatten() else {
        return KERNEL_EFAULT;
    };
    let Some(addr) = crate::modules::resolve_symbol(handle, &name) else {
        log::warn!("[kernel] sceKernelDlsym failed: handle={handle:#x} symbol='{name}'");
        return KERNEL_ESRCH;
    };
    if ctx.write_u64(out, addr) { 0 } else { KERNEL_EFAULT }
}

/// Sleeps shorter than this only yield: guest worker pools poll with
/// usleep(1), and a host timed sleep costs tens of microseconds of timer
/// slack whatever is asked.
const YIELD_BELOW_USEC: u64 = 50;

/// Sleep with microsecond precision; very short waits only yield (see
/// [`YIELD_BELOW_USEC`]), longer ones sleep at least the time asked for.
/// Every sleep is a cancellation point.
pub(crate) fn sleep_micros(us: u64) {
    if us == 0 {
        crate::kernel_ext::test_cancel();
        return;
    }
    if us < YIELD_BELOW_USEC {
        crate::kernel_ext::test_cancel();
        std::thread::yield_now();
        crate::kernel_ext::test_cancel();
    } else {
        crate::kernel_ext::cancellable_sleep(Duration::from_micros(us));
    }
}

/// sceKernelUsleep(usec) / usleep
fn cb_usleep(ctx: &mut CpuContext) -> HleResult {
    sleep_micros(ctx.rdi() as u32 as u64);
    OK
}

/// sceKernelSleep(seconds)
fn cb_sleep(ctx: &mut CpuContext) -> HleResult {
    crate::kernel_ext::cancellable_sleep(Duration::from_secs(ctx.rdi() as u32 as u64));
    OK
}

/// sleep(seconds) (POSIX): a cancellation point. The sleep is never cut
/// short (as in `nanosleep` above), so the result is the seconds left
/// unslept, always 0.
fn cb_posix_sleep(ctx: &mut CpuContext) -> HleResult {
    crate::kernel_ext::cancellable_sleep(Duration::from_secs(ctx.rdi() as u32 as u64));
    0
}

/// Sleep for the timespec at `req`. The sleep is never interrupted, and as
/// FreeBSD writes the remaining time only on an interruption, `rem` is left
/// untouched. Errors are errno values.
fn nanosleep_core(ctx: &CpuContext, req: u64) -> Result<(), i32> {
    if req == 0 {
        return Err(libc::EFAULT);
    }
    let (Some(s), Some(ns)) = (ctx.read_u64(req), ctx.read_u64(req + 8)) else { return Err(libc::EFAULT) };
    if (s as i64) < 0 || ns >= 1_000_000_000 {
        return Err(libc::EINVAL);
    }
    let total = s.saturating_mul(1_000_000) + ns / 1000;
    sleep_micros(total.max((ns > 0) as u64));
    Ok(())
}

/// nanosleep(req*, rem*) -> 0, or -1 with errno set
fn cb_nanosleep(ctx: &mut CpuContext) -> HleResult {
    match nanosleep_core(ctx, ctx.rdi()) {
        Ok(()) => OK,
        Err(e) => {
            set_errno(ctx, e);
            -1
        }
    }
}

/// sceKernelGetRandomNumber(buf, size) — size is at most 64 bytes.
fn cb_get_random_number(ctx: &mut CpuContext) -> HleResult {
    let (buf, len) = (ctx.rdi(), ctx.rsi() as usize);
    if buf == 0 || len > 64 {
        return ERR_INVALID_ARGUMENT;
    }
    let mut data = vec![0u8; len];
    let n = unsafe { libc::getrandom(data.as_mut_ptr() as *mut libc::c_void, len, 0) };
    if n != len as isize {
        return ERR_INVALID_ARGUMENT;
    }
    if ctx.write_bytes(buf, &data) { OK } else { ERR_MEMORY_FAULT }
}

/// Retail console.
fn cb_is_development_mode(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// sceKernelDebugRaiseException(error, 0): a fatal, non-returning debug
/// exception; terminate the calling guest thread with the error code.
fn cb_debug_raise_exception(ctx: &mut CpuContext) -> HleResult {
    let (error, unknown) = (ctx.rdi() as u32, ctx.rsi());
    if unknown != 0 {
        return ERR_INVALID_ARGUMENT;
    }
    log::error!("[kernel] sceKernelDebugRaiseException error={error:#010x}");
    ps5cpu::exec::exit_guest_thread(error as u64)
}

/// Guest exception handlers are not entered through a signal-return frame.
fn cb_is_signal_return(_ctx: &mut CpuContext) -> HleResult {
    0
}

fn cb_ok(_ctx: &mut CpuContext) -> HleResult {
    OK
}

/// ProcParam: size u64, magic u32, entry count u32, SDK version u32 @0x10.
const PROC_PARAM_SDK_VERSION: u64 = 0x10;

/// sceKernelGetCompiledSdkVersion(u32* version): the SDK the executable was
/// built with, from its ProcParam.
fn cb_get_compiled_sdk_version(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let pp = *PROC_PARAM_ADDR.lock().unwrap();
    let version = if pp == 0 { 0 } else { ctx.read_u32(pp + PROC_PARAM_SDK_VERSION).unwrap_or(0) };
    if ctx.write_u32(out, version) { OK } else { ERR_MEMORY_FAULT }
}

/// sceKernelIsNeoMode(): 0, a base console (never the enhanced mode).
fn cb_is_neo_mode(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// Raise the host's open-file soft limit to the hard limit before the guest
/// opens anything: guest files, sockets and memory backings are host
/// descriptors, and the common default soft limit (1024) is easily reached.
fn raise_open_file_limit() {
    let mut l = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    // SAFETY: plain limit query/update with a local.
    unsafe {
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut l) != 0 {
            log::warn!("[kernel] cannot read the open file limit");
            return;
        }
        if l.rlim_cur >= l.rlim_max {
            return;
        }
        let old = l.rlim_cur;
        l.rlim_cur = l.rlim_max;
        if libc::setrlimit(libc::RLIMIT_NOFILE, &l) != 0 {
            log::warn!("[kernel] cannot raise the open file limit from {old}");
        }
    }
}

pub fn register(reg: &mut HleRegistry) {
    raise_open_file_limit();
    reg.register_fn("OMDRKKAZ8I4", "sceKernelDebugRaiseException", "libKernel", cb_debug_raise_exception);
    reg.register_fn("zE-wXIZjLoM", "sceKernelDebugRaiseExceptionOnReleaseMode", "libKernel", cb_debug_raise_exception);
    reg.register_fn("crb5j7mkk1c", "_is_signal_return", "libKernel", cb_is_signal_return);
    reg.register_fn("kbw4UHHSYy0", "__pthread_cxa_finalize", "libKernel", cb_ok);
    reg.register_fn("cfwBSQyr5Ys", "sceKernelDebugWriteCppExceptionInfo", "libKernel", cb_ok);
    reg.register_fn("959qrazPIrg", "sceKernelGetProcParam", "libKernel", cb_get_proc_param);
    reg.register_fn("p5EcQeEeJAE", "_sceKernelRtldSetApplicationHeapAPI", "libKernel", cb_set_application_heap_api);
    reg.register_fn("9BcDykPmo1I", "__error", "libKernel", cb_error);
    reg.register_fn("HoLVWNanBBc", "getpid", "libKernel", cb_getpid);
    // The sceSp* wrapper of libSceSpSysCallWrapper forwards to getpid.
    reg.register_fn("jWd9FwvNv4E", "sceSpGetPid", "libSceSpSysCallWrapper", cb_getpid);
    reg.register_fn("e6ovBo9ZvJc", "getppid", "libKernel", cb_getppid);
    reg.register_fn("G-MYv5erXaU", "sceKernelGetAppInfo", "libKernel", cb_get_app_info);
    reg.register_fn("edJoqV2FIWs", "setsid", "libKernel", cb_setsid);
    reg.register_fn("Mv1zUObHvXI", "sceKernelGetSystemSwVersion", "libKernel", cb_get_system_sw_version);
    reg.register_fn("aML18Z0J0t0", "sceKernelGetProsperoSystemSwVersion", "libKernel", cb_get_system_sw_version);
    reg.register_fn("YeU23Szo3BM", "sceKernelGetAllowedSdkVersionOnSystem", "libKernel", cb_get_allowed_sdk_version_on_system);
    reg.register_fn("Rb6ziJKvCCQ", "umask", "libKernel", cb_umask);
    reg.register_fn("6Z83sYWFlA8", "_exit", "libKernel", cb_exit);
    reg.register_fn("LwG8g3niqwA", "sceKernelDlsym", "libKernel", cb_dlsym);
    reg.register_fn("1jfXLRVzisc", "sceKernelUsleep", "libKernel", cb_usleep);
    reg.register_fn("QcteRwbsnV0", "usleep", "libKernel", cb_usleep);
    reg.register_fn("-ZR+hG7aDHw", "sceKernelSleep", "libKernel", cb_sleep);
    reg.register_fn("0wu33hunNdE", "sleep", "libKernel", cb_posix_sleep);
    reg.register_fn("yS8U2TGCe1A", "nanosleep", "libKernel", cb_nanosleep);
    // The sceSp* wrapper of libSceSpSysCallWrapper forwards to nanosleep.
    reg.register_fn("Bs3QDmfuh4w", "sceSpKernelNanosleep", "libSceSpSysCallWrapper", cb_nanosleep);
    reg.register_fn("NhpspxdjEKU", "_nanosleep", "libKernel", cb_nanosleep);
    reg.register_fn("-2lSgH73m4o", "sceKernelGetRandomNumber", "libKernel", cb_get_random_number);
    reg.register_fn("UtO0OHMCgmI", "sceKernelIsDevelopmentMode", "libKernel", cb_is_development_mode);
    reg.register_fn("WB66evu8bsU", "sceKernelGetCompiledSdkVersion", "libKernel", cb_get_compiled_sdk_version);
    reg.register_fn("WslcK1FQcGI", "sceKernelIsNeoMode", "libKernel", cb_is_neo_mode);
    // Leaving the secure block needs no work: nothing entered one.
    reg.register_fn("Ac86z8q7T8A", "sceKernelExitSblock", "libKernel", cb_ok);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister;
    use ps5vm::{MemProtect, VirtualMemory};

    /// POSIX nanosleep reports failures as -1 (errno), never as kernel codes.
    #[test]
    fn posix_nanosleep_returns_minus_one() {
        const BASE: u64 = 0x7A60_0000_0000;
        let vm = VirtualMemory::new();
        vm.map(BASE, 0x10000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(CpuRegister::Rdi, 0);
        assert_eq!(cb_nanosleep(&mut ctx), -1);
        ctx.write_u64(BASE, 0);
        ctx.write_u64(BASE + 8, 1_000_000_000);
        ctx.set(CpuRegister::Rdi, BASE);
        assert_eq!(cb_nanosleep(&mut ctx), -1);
        ctx.write_u64(BASE + 8, 1000);
        ctx.write_u64(BASE + 16, 7);
        ctx.set(CpuRegister::Rsi, BASE + 16);
        assert_eq!(cb_nanosleep(&mut ctx), OK);
        assert_eq!(ctx.read_u64(BASE + 16), Some(7), "a completed sleep leaves rem alone");
        let _ = vm.unmap(BASE, 0x10000);
    }

    /// POSIX sleep(0) returns at once; sleep(1) waits the full second and
    /// returns 0, the seconds left unslept.
    #[test]
    fn posix_sleep_waits_full_seconds_and_returns_zero() {
        assert_eq!(ps5nid::compute_nid("sleep"), "0wu33hunNdE");
        let vm = VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(CpuRegister::Rdi, 0);
        let start = std::time::Instant::now();
        assert_eq!(cb_posix_sleep(&mut ctx), 0);
        assert!(start.elapsed() < Duration::from_millis(500), "sleep(0) waited");
        ctx.set(CpuRegister::Rdi, 1);
        let start = std::time::Instant::now();
        assert_eq!(cb_posix_sleep(&mut ctx), 0);
        assert!(start.elapsed() >= Duration::from_secs(1), "sleep(1) returned early");
    }

    #[test]
    fn posix_sleep_is_registered_as_implementation() {
        let reg = crate::registry::HleRegistry::with_defaults();
        assert!(reg.is_implemented("0wu33hunNdE"), "sleep is still the catalog stub");
    }

    /// getppid is the host parent of the process getpid reports: positive,
    /// not the process itself, stable (reference checks).
    #[test]
    #[cfg(unix)]
    fn getppid_is_the_parent_process() {
        let vm = VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        let parent = cb_getppid(&mut ctx);
        assert!(parent > 0 && parent != cb_getpid(&mut ctx));
        assert_eq!(cb_getppid(&mut ctx), parent);
        assert_eq!(parent, std::os::unix::process::parent_id() as i64);
    }

    /// sceKernelGetAppInfo writes the app id (1) and the title id only, the
    /// field between them and the rest are left alone; another pid is EPERM
    /// and a NULL info is accepted (reference checks).
    #[test]
    fn get_app_info_writes_app_id_and_title_id() {
        const BASE: u64 = 0x7C48_0000_0000;
        let vm = VirtualMemory::new();
        vm.map(BASE, 0x1000, MemProtect::RW).unwrap();
        let ctx = CpuContext::new(&vm);
        let me = std::process::id() as i32;
        ctx.write_bytes(BASE, &[0x5A; 0x58]);
        assert_eq!(get_app_info(&ctx, me, BASE, Some("PPSA97358")), 0);
        assert_eq!(ctx.read_u32(BASE), Some(1));
        assert_eq!(ctx.read_u64(BASE + 4), Some(0x5A5A_5A5A_5A5A_5A5A), "the field after the app id");
        assert_eq!(ctx.read_bytes(BASE + 0x10, 10).unwrap(), b"PPSA97358\0");
        assert_eq!(ctx.read_bytes(BASE + 0x1A, 0x3E).unwrap(), vec![0x5A; 0x3E], "past the title id");
        assert_eq!(get_app_info(&ctx, me + 1, BASE, Some("PPSA97358")), KERNEL_EPERM);
        assert_eq!(get_app_info(&ctx, me, 0, Some("PPSA97358")), 0);
        // A longer id keeps 9 characters and the terminator.
        assert_eq!(get_app_info(&ctx, me, BASE, Some("ABCDEFGHIJKL")), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x10, 10).unwrap(), b"ABCDEFGHI\0");
        assert_eq!(ps5nid::compute_nid("sceKernelGetAppInfo"), "G-MYv5erXaU");
        vm.unmap(BASE, 0x1000).unwrap();
    }

    /// setsid fails with EPERM; umask starts at 022, returns the previous
    /// mask and keeps the bits up to 07777 (reference checks).
    #[test]
    fn setsid_and_umask() {
        std::thread::spawn(|| {
            const BASE: u64 = 0x7C4A_0000_0000;
            let vm: &'static VirtualMemory = Box::leak(Box::new(VirtualMemory::new()));
            vm.map(BASE, 0x1000, MemProtect::RW).unwrap();
            let mut ctx = CpuContext::new(vm);
            let errno = bind_test_errno(&ctx);
            assert_eq!(cb_setsid(&mut ctx), -1);
            assert_eq!(ctx.read_u32(errno), Some(1));
            let mut umask = |m: u64| {
                ctx.set(CpuRegister::Rdi, m);
                cb_umask(&mut ctx)
            };
            assert_eq!(umask(0), 0o022);
            assert_eq!(umask(0o7777), 0);
            assert_eq!(umask(0o022), 0o7777);
            assert_eq!(umask(0o17777), 0o022);
            assert_eq!(umask(0o022), 0o7777, "bits above 07777 dropped");
        })
        .join()
        .unwrap();
    }

    /// The system software version block: 0x28 bytes with the size at 0,
    /// the string at 8 and the version at 0x24; other sizes are EINVAL
    /// without writing, NULL is EFAULT, errno is untouched (reference
    /// GuestSystemConfiguration / GuestSoftwareVersion cases); the allowed
    /// SDK version is the system version with the patch bits set.
    #[test]
    fn system_sw_version_blocks() {
        std::thread::spawn(|| {
            const BASE: u64 = 0x7C55_0000_0000;
            let vm: &'static VirtualMemory = Box::leak(Box::new(VirtualMemory::new()));
            vm.map(BASE, 0x1000, MemProtect::RW).unwrap();
            let mut ctx = CpuContext::new(vm);
            let errno = bind_test_errno(&ctx);
            ctx.write_u32(errno, 13);
            for cb in [cb_get_system_sw_version as crate::registry::HleCallback, cb_get_system_sw_version] {
                ctx.set(CpuRegister::Rdi, 0);
                assert_eq!(cb(&mut ctx), 0x8002_000Eu32 as i32 as i64);
                for size in [0u64, 0x18, 0x27, 0x29, 0x30] {
                    ctx.write_bytes(BASE, &[0xAA; 0x30]);
                    ctx.write_u64(BASE, size);
                    ctx.set(CpuRegister::Rdi, BASE);
                    assert_eq!(cb(&mut ctx), 0x8002_0016u32 as i32 as i64, "size {size:#x}");
                    assert_eq!((ctx.read_u64(BASE), ctx.read_u32(BASE + 0x24), ctx.read_bytes(BASE + 8, 1).unwrap()[0]), (Some(size), Some(0xAAAA_AAAA), 0xAA));
                }
                ctx.write_bytes(BASE, &[0xAA; 0x30]);
                ctx.write_u64(BASE, 0x28);
                assert_eq!(cb(&mut ctx), 0);
                assert_eq!(ctx.read_u64(BASE), Some(0x28));
                let text = ctx.read_bytes(BASE + 8, 0x1C).unwrap();
                assert_eq!(&text[..11], b"01.000.000\0");
                assert!(text[11..].iter().all(|&b| b == 0));
                assert_eq!(ctx.read_u32(BASE + 0x24), Some(0x0100_0000));
                assert_eq!(ctx.read_bytes(BASE + 0x28, 8).unwrap(), vec![0xAA; 8], "past the block");
            }
            ctx.set(CpuRegister::Rdi, 0);
            assert_eq!(cb_get_allowed_sdk_version_on_system(&mut ctx), 0x8002_0016u32 as i32 as i64);
            ctx.set(CpuRegister::Rdi, BASE + 0x40);
            assert_eq!(cb_get_allowed_sdk_version_on_system(&mut ctx), 0);
            let allowed = ctx.read_u32(BASE + 0x40).unwrap();
            assert_eq!((allowed, allowed >> 16), (0x0100_0FFF, 0x0100_0000 >> 16));
            assert_eq!(ctx.read_u32(errno), Some(13));
            for (nid, name) in [
                ("Mv1zUObHvXI", "sceKernelGetSystemSwVersion"),
                ("aML18Z0J0t0", "sceKernelGetProsperoSystemSwVersion"),
                ("YeU23Szo3BM", "sceKernelGetAllowedSdkVersionOnSystem"),
            ] {
                assert_eq!(ps5nid::compute_nid(name), nid);
            }
        })
        .join()
        .unwrap();
    }

    /// sceKernelDlsym: a symbol of a loaded module resolves; an unknown
    /// symbol or module is ESRCH with the output cleared; NULL symbol or
    /// output EFAULT.
    #[test]
    fn dlsym_resolves_and_reports_kernel_errors() {
        const BASE: u64 = 0x7C5F_0000_0000;
        let vm = VirtualMemory::new();
        vm.map(BASE, 0x1000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let img = crate::modules::ModuleImage {
            path: "/app0/dlsym-kernel-test.prx",
            host_path: "",
            so_name: "",
            base: 0x7C60_0000_0000,
            size: 0x10000,
            seg_start: 0x7C60_0000_0000,
            seg_end: 0x7C60_0000_F000,
            entry_point: 0,
            init_entry_point: 0,
            fini_entry_point: 0,
            eh_frame_hdr_segment: None,
            segments: &[],
            is_main: false,
        };
        let h = crate::modules::register_module(&vm, &img);
        let nid = ps5nid::compute_nid("dlsym_kernel_symbol");
        crate::modules::register_module_symbols(h, vec![(nid, crate::modules::SymbolKind::Func, 0x7C60_0000_0040)]);
        let (name, missing, out) = (BASE, BASE + 0x40, BASE + 0x100);
        ctx.write_bytes(name, b"dlsym_kernel_symbol\0");
        ctx.write_bytes(missing, b"dlsym_kernel_missing\0");
        let call = |ctx: &mut CpuContext, handle: i32, sym: u64, out: u64| {
            ctx.set(CpuRegister::Rdi, handle as u32 as u64);
            ctx.set(CpuRegister::Rsi, sym);
            ctx.set(CpuRegister::Rdx, out);
            cb_dlsym(ctx)
        };
        assert_eq!(call(&mut ctx, h, name, out), 0);
        assert_eq!(ctx.read_u64(out), Some(0x7C60_0000_0040));
        ctx.write_u64(out, 7);
        assert_eq!(call(&mut ctx, h, missing, out), 0x8002_0003u32 as i32 as i64);
        assert_eq!(ctx.read_u64(out), Some(7), "the caller's fallback stays");
        assert_eq!(call(&mut ctx, 0x7FFF_0000, name, out), 0x8002_0003u32 as i32 as i64, "unknown module");
        assert_eq!(ctx.read_u64(out), Some(7));
        assert_eq!(call(&mut ctx, h, 0, out), 0x8002_000Eu32 as i32 as i64);
        assert_eq!(ctx.read_u64(out), Some(7));
        assert_eq!(call(&mut ctx, h, name, 0), 0x8002_000Eu32 as i32 as i64);
        crate::modules::unregister_module(h);
        vm.unmap(BASE, 0x1000).unwrap();
    }

    #[test]
    fn process_query_nids_match_names() {
        for (nid, name) in [
            ("WB66evu8bsU", "sceKernelGetCompiledSdkVersion"),
            ("WslcK1FQcGI", "sceKernelIsNeoMode"),
            ("Ac86z8q7T8A", "sceKernelExitSblock"),
        ] {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn compiled_sdk_version_from_proc_param() {
        const BASE: u64 = 0x5DA2_0000_0000;
        let vm = VirtualMemory::new();
        vm.map(BASE, 0x1000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(CpuRegister::Rdi, 0);
        assert_eq!(cb_get_compiled_sdk_version(&mut ctx), ERR_INVALID_ARGUMENT);
        // Only when no other test installed a ProcParam.
        if *PROC_PARAM_ADDR.lock().unwrap() == 0 {
            ctx.write_u32(BASE + PROC_PARAM_SDK_VERSION, 0x0700_0000);
            set_proc_param(BASE);
            ctx.set(CpuRegister::Rdi, BASE + 0x100);
            assert_eq!(cb_get_compiled_sdk_version(&mut ctx), OK);
            assert_eq!(ctx.read_u32(BASE + 0x100), Some(0x0700_0000));
            set_proc_param(0);
        }
        let _ = vm.unmap(BASE, 0x1000);
    }

    /// A sub-millisecond usleep sleeps at least the time asked for; only
    /// the shortest ones just yield.
    #[test]
    fn short_usleeps_sleep_the_time_asked() {
        for us in [YIELD_BELOW_USEC, 200, 900] {
            let t = std::time::Instant::now();
            sleep_micros(us);
            assert!(t.elapsed() >= Duration::from_micros(us), "usleep({us}) took {:?}", t.elapsed());
        }
        let t = std::time::Instant::now();
        for _ in 0..100 {
            sleep_micros(1);
        }
        assert!(t.elapsed() < Duration::from_millis(500));
    }

    /// After start-up the open-file soft limit is the hard limit.
    #[test]
    fn open_file_limit_is_raised() {
        raise_open_file_limit();
        let mut l = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        assert_eq!(unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut l) }, 0);
        // An unlimited hard limit cannot be the soft one (fs.nr_open caps it).
        assert!(l.rlim_cur == l.rlim_max || l.rlim_max == libc::RLIM_INFINITY);
    }

    #[test]
    fn nanosleep_nids_match_names() {
        for (nid, name) in [("yS8U2TGCe1A", "nanosleep"), ("NhpspxdjEKU", "_nanosleep")] {
            assert_eq!(ps5nid::compute_nid(name).replace('_', "-"), nid, "NID mismatch for {name}");
        }
    }
}
