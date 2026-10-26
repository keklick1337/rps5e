// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Small system services: libSceRandom, the game heap error report hook
//! of libSceLibcInternal, the coredump handler registration, the device
//! service (libSceMbus) and the video recording metadata hook. Also registers libSceCesCs, libSceHmd2 and
//! DbgAddressSanitizer.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use std::sync::Mutex;

use crate::context::{CpuContext, CpuRegister};
use crate::registry::{HleCallback, HleRegistry, HleResult};

const RANDOM_ERROR_INVALID: HleResult = 0x817C_0016u32 as i32 as i64;
const MAX_RANDOM_BYTES: u64 = 64;
const ERR_MEMORY_FAULT: HleResult = 0x8002_000Eu32 as i32 as i64;

/// Fill `buf` from the host CSPRNG.
pub(crate) fn fill_random(buf: &mut [u8]) {
    let mut done = 0;
    while done < buf.len() {
        let r = unsafe { libc::getrandom(buf[done..].as_mut_ptr() as *mut libc::c_void, buf.len() - done, 0) };
        if r > 0 {
            done += r as usize;
        } else if std::io::Error::last_os_error().raw_os_error() != Some(libc::EINTR) {
            break;
        }
    }
}

/// sceRandomGetRandomNumber(void* buf, size_t size): at most 64 bytes.
fn cb_random_get_random_number(ctx: &mut CpuContext) -> HleResult {
    let dst = ctx.rdi();
    let size = ctx.rsi();
    if (dst == 0 && size != 0) || size > MAX_RANDOM_BYTES {
        return RANDOM_ERROR_INVALID;
    }
    if size == 0 {
        return 0;
    }
    let mut bytes = vec![0u8; size as usize];
    fill_random(&mut bytes);
    if ctx.write_bytes(dst, &bytes) {
        0
    } else {
        ERR_MEMORY_FAULT
    }
}

/// sceLibcInternalHeapErrorReportForGame: diagnostic sink for the game's
/// own heap checks.
fn cb_heap_error_report_for_game(ctx: &mut CpuContext) -> HleResult {
    log::warn!(
        "[libc] sceLibcInternalHeapErrorReportForGame: rdi={:#x} rsi={:#x} rdx={:#x} rcx={:#x}",
        ctx.rdi(),
        ctx.rsi(),
        ctx.rdx(),
        ctx.rcx()
    );
    0
}

/// Frames the backtrace reports after the import's caller.
const BACKTRACE_DEPTH: usize = 32;
/// Longest name the backtrace reads.
const BACKTRACE_NAME_MAX: usize = 256;

/// Guest return addresses, innermost first: the caller of the running
/// import (`[rsp]`), then the frame-pointer chain from `rbp`. The walk stops
/// at a frame that cannot be read or whose link does not move up the stack,
/// as the guest fault report does.
fn guest_backtrace(ctx: &CpuContext) -> Vec<u64> {
    let mut frames = Vec::new();
    if let Some(ret) = ctx.read_u64(ctx.get(CpuRegister::Rsp)) {
        frames.push(ret);
    }
    let mut rbp = ctx.get(CpuRegister::Rbp);
    for _ in 0..BACKTRACE_DEPTH {
        let (Some(ret), Some(next)) = (ctx.read_u64(rbp.wrapping_add(8)), ctx.read_u64(rbp)) else { break };
        frames.push(ret);
        if next <= rbp {
            break;
        }
        rbp = next;
    }
    frames
}

/// NUL-terminated name of at most `BACKTRACE_NAME_MAX` bytes; None for a
/// null pointer, a fault, or a longer name.
fn read_name(ctx: &CpuContext, addr: u64) -> Option<String> {
    if addr == 0 {
        return None;
    }
    let mut bytes = Vec::new();
    for i in 0..BACKTRACE_NAME_MAX as u64 {
        let b = ctx.read_bytes(addr + i, 1)?[0];
        if b == 0 {
            return Some(String::from_utf8_lossy(&bytes).into_owned());
        }
        bytes.push(b);
    }
    None
}

/// sceLibcInternalBacktraceForGame(const char* name): logs the guest call
/// chain under `name`. It returns nothing and the caller carries on, so the
/// log lines are the whole effect; the value returned here is never read.
fn cb_backtrace_for_game(ctx: &mut CpuContext) -> HleResult {
    let name = read_name(ctx, ctx.rdi()).unwrap_or_else(|| "(unnamed)".into());
    log::warn!("[libc] backtrace for {name}:");
    for (i, addr) in guest_backtrace(ctx).into_iter().enumerate() {
        match crate::modules::module_by_address(addr) {
            Some(m) => log::warn!("[libc]   #{i} {addr:#x} ({}+{:#x})", m.name, addr - m.base),
            None => log::warn!("[libc]   #{i} {addr:#x}"),
        }
    }
    0
}

/// (handler, context) registered for crash reports; never invoked.
static COREDUMP_HANDLER: Mutex<(u64, u64)> = Mutex::new((0, 0));

/// sceCoredumpRegisterCoredumpHandler(handler, stackSize, context)
fn cb_coredump_register_handler(ctx: &mut CpuContext) -> HleResult {
    *COREDUMP_HANDLER.lock().unwrap_or_else(|p| p.into_inner()) = (ctx.rdi(), ctx.rdx());
    0
}

/// sceCoredumpUnregisterCoredumpHandler(): forget the handler.
fn cb_coredump_unregister_handler(_ctx: &mut CpuContext) -> HleResult {
    *COREDUMP_HANDLER.lock().unwrap_or_else(|p| p.into_inner()) = (0, 0);
    0
}

/// sceDeviceServiceInitialize(): nothing to set up without host device
/// passthrough.
fn cb_ok(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// sceDeviceServiceQueryDeviceInfo_(u32 deviceHandle, void* outInfo): no
/// device is known, the output is left untouched.
fn cb_device_service_query_device_info(ctx: &mut CpuContext) -> HleResult {
    log::debug!("[deviceservice] query device info handle={:#x} out={:#x}", ctx.rdi() as u32, ctx.rsi());
    0
}

/// Kernel EINVAL as libraries report it.
const ERR_INVALID: HleResult = 0x8002_0016u32 as i32 as i64;
/// Largest recording info value kept (the biggest documented info, the
/// chapter/permission strings, are far smaller).
const VIDEO_RECORDING_INFO_MAX: u64 = 0x1_0000;

/// Recording metadata by info id, as the guest last set it.
static VIDEO_RECORDING_INFO: Mutex<Vec<(i32, Vec<u8>)>> = Mutex::new(Vec::new());

/// sceVideoRecordingSetInfo(int info, const void* data, size_t size): the
/// value is kept for GetInfo; nothing records gameplay video.
fn cb_video_recording_set_info(ctx: &mut CpuContext) -> HleResult {
    let (info, data, size) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    if (data == 0 && size != 0) || size > VIDEO_RECORDING_INFO_MAX {
        return ERR_INVALID;
    }
    let bytes = if size == 0 { Vec::new() } else { ctx.read_bytes(data, size as usize).unwrap_or_default() };
    if bytes.len() as u64 != size {
        return ERR_MEMORY_FAULT;
    }
    let mut all = VIDEO_RECORDING_INFO.lock().unwrap_or_else(|p| p.into_inner());
    match all.iter_mut().find(|(id, _)| *id == info) {
        Some(entry) => entry.1 = bytes,
        None => all.push((info, bytes)),
    }
    0
}

/// sceVideoRecordingGetInfo(int info, void* data, size_t size): the value
/// set last; a NULL buffer, one smaller than the value or an info that was
/// never set is refused.
fn cb_video_recording_get_info(ctx: &mut CpuContext) -> HleResult {
    let (info, data, size) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    if data == 0 {
        return ERR_INVALID;
    }
    let all = VIDEO_RECORDING_INFO.lock().unwrap_or_else(|p| p.into_inner());
    let Some((_, value)) = all.iter().find(|(id, _)| *id == info) else {
        log::warn!("[videorecording] GetInfo({info:#x}): never set");
        return ERR_INVALID;
    };
    if size < value.len() as u64 {
        return ERR_INVALID;
    }
    if ctx.write_bytes(data, value) { 0 } else { ERR_MEMORY_FAULT }
}

/// EOPNOTSUPP as libraries report it.
const ERR_NOT_SUPPORTED: HleResult = 0x8002_002Du32 as i32 as i64;

/// Gameplay recording into a file is not available on the host: no
/// recording session ever opens. Its calls are refused (EOPNOTSUPP, logged
/// once per call name) and the status is "no recording" (0). A NULL
/// parameter or path is EINVAL.
fn video_recording_refused(what: &str) -> HleResult {
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|p| p.into_inner());
    if !seen.iter().any(|s| s == what) {
        log::warn!("[videorecording] {what}: gameplay recording is not available");
        seen.push(what.to_string());
    }
    ERR_NOT_SUPPORTED
}

/// sceVideoRecordingQueryMemSize(const SceVideoRecordingParam*)
fn cb_video_recording_query_mem_size(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return ERR_INVALID;
    }
    video_recording_refused("sceVideoRecordingQueryMemSize")
}

/// sceVideoRecordingOpen(path, const SceVideoRecordingParam*, heap, heapSize)
fn cb_video_recording_open(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 || ctx.rsi() == 0 {
        return ERR_INVALID;
    }
    video_recording_refused("sceVideoRecordingOpen")
}

/// sceVideoRecordingStart / Stop / Close(discard): no session is open.
fn cb_video_recording_session(_ctx: &mut CpuContext) -> HleResult {
    video_recording_refused("sceVideoRecordingStart/Stop/Close")
}

/// sceVideoRecordingGetStatus(): no recording (0).
fn cb_video_recording_get_status(_ctx: &mut CpuContext) -> HleResult {
    0
}

const EXPORTS: &[(&str, &str, &str, HleCallback)] = &[
    ("ZvWzS2wTIMc", "sceVideoRecordingQueryMemSize", "libSceVideoRecording", cb_video_recording_query_mem_size),
    ("7QoTYzVuzto", "sceVideoRecordingOpen", "libSceVideoRecording", cb_video_recording_open),
    ("tWoe9IlGAhs", "sceVideoRecordingStart", "libSceVideoRecording", cb_video_recording_session),
    ("OOFxrMY+mfI", "sceVideoRecordingStop", "libSceVideoRecording", cb_video_recording_session),
    ("KHvkPQJDMLk", "sceVideoRecordingClose", "libSceVideoRecording", cb_video_recording_session),
    ("fZJQzFK4Gv4", "sceVideoRecordingGetStatus", "libSceVideoRecording", cb_video_recording_get_status),
    // No device events are ever pending.
    ("9ddRUOV8Q5A", "sceDeviceServiceGetEventState", "libSceMbus", cb_ok),
    ("Fc8qxlKINYQ", "sceVideoRecordingSetInfo", "libSceVideoRecording", cb_video_recording_set_info),
    ("sA6+5XdbqMA", "sceVideoRecordingGetInfo", "libSceVideoRecording", cb_video_recording_get_info),
    ("84fDxStrG44", "sceDeviceServiceInitialize", "libSceMbus", cb_ok),
    // The device set never changes: generation 0.
    ("oFon+A5v1z8", "sceDeviceServiceGetGeneration", "libSceMbus", cb_ok),
    ("Uq8uW74rVpU", "sceDeviceServiceTerminate", "libSceMbus", cb_ok),
    ("UNMEa+5lrUA", "sceDeviceServiceQueryDeviceInfo_", "libSceMbus", cb_device_service_query_device_info),
    ("PI7jIZj4pcE", "sceRandomGetRandomNumber", "libSceRandom", cb_random_get_random_number),
    ("al3JzFI9MQ0", "sceLibcInternalHeapErrorReportForGame", "libSceLibcInternal", cb_heap_error_report_for_game),
    ("EHsF2i9FXPM", "sceLibcInternalBacktraceForGame", "libSceLibcInternal", cb_backtrace_for_game),
    ("8zLSfEfW5AU", "sceCoredumpRegisterCoredumpHandler", "libSceCoredump", cb_coredump_register_handler),
    ("fFkhOgztiCA", "sceCoredumpUnregisterCoredumpHandler", "libSceCoredump", cb_coredump_unregister_handler),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, lib, cb) in EXPORTS {
        reg.register_fn(nid, name, lib, cb);
    }
    crate::ces::register(reg);
    crate::hmd2::register(reg);
    crate::dbg_asan::register(reg);
    crate::usbd::register(reg);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// The device service keeps no state: initialize and terminate succeed.
    #[test]
    fn device_service_terminate() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        for name in ["sceDeviceServiceInitialize", "sceDeviceServiceTerminate"] {
            let cb = EXPORTS.iter().find(|e| e.1 == name).unwrap().3;
            assert_eq!(cb(&mut ctx), 0, "{name}");
        }
    }

    /// The backtrace starts at the import's caller and follows the frame
    /// pointers; a link that does not move up, an unreadable frame or a null
    /// rbp ends the walk.
    #[test]
    fn backtrace_follows_frame_pointers() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E80_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (sp, a, b, c) = (BASE + 0x80, BASE + 0x100, BASE + 0x200, BASE + 0x300);
        assert!(ctx.write_u64(sp, 0x1111));
        assert!(ctx.write_u64(a, b) && ctx.write_u64(a + 8, 0x2222));
        assert!(ctx.write_u64(b, c) && ctx.write_u64(b + 8, 0x3333));
        assert!(ctx.write_u64(c, 0) && ctx.write_u64(c + 8, 0x4444));
        ctx.set(Rsp, sp);
        ctx.set(Rbp, a);
        assert_eq!(guest_backtrace(&ctx), vec![0x1111, 0x2222, 0x3333, 0x4444]);

        assert!(ctx.write_u64(b, a));
        assert_eq!(guest_backtrace(&ctx), vec![0x1111, 0x2222, 0x3333], "a link that goes down ends the walk");

        assert!(ctx.write_u64(b, BASE + 0x2000));
        assert_eq!(guest_backtrace(&ctx), vec![0x1111, 0x2222, 0x3333], "an unmapped frame ends the walk");

        ctx.set(Rbp, 0);
        assert_eq!(guest_backtrace(&ctx), vec![0x1111]);
    }

    /// The name is read up to its NUL, at most 256 bytes; the command logs
    /// it and returns.
    #[test]
    fn backtrace_name_is_bounded() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E81_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        assert!(ctx.write_bytes(BASE, b"heap\0"));
        assert_eq!(read_name(&ctx, BASE), Some("heap".to_string()));
        assert_eq!(read_name(&ctx, 0), None);
        assert!(ctx.write_bytes(BASE + 0x100, &[b'a'; 256]));
        assert_eq!(read_name(&ctx, BASE + 0x100), None, "no NUL within 256 bytes");
        ctx.set(Rdi, BASE);
        assert_eq!(cb_backtrace_for_game(&mut ctx), 0);
        ctx.set(Rdi, 0);
        assert_eq!(cb_backtrace_for_game(&mut ctx), 0, "an unnamed backtrace still runs");
    }

    /// GetInfo returns what SetInfo stored for the same id; the size check
    /// and unknown ids are refused.
    #[test]
    fn video_recording_info_round_trip() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E82_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let id = 0x7E57_0001u64;
        ctx.write_bytes(BASE, b"chapter\0");
        for (r, v) in [(Rdi, id), (Rsi, BASE), (Rdx, 8)] {
            ctx.set(r, v);
        }
        assert_eq!(cb_video_recording_set_info(&mut ctx), 0);
        for (r, v) in [(Rdi, id), (Rsi, BASE + 0x100), (Rdx, 16)] {
            ctx.set(r, v);
        }
        assert_eq!(cb_video_recording_get_info(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x100, 8).unwrap(), b"chapter\0");
        ctx.set(Rdx, 7);
        assert_eq!(cb_video_recording_get_info(&mut ctx), ERR_INVALID, "a buffer smaller than the value");
        ctx.set(Rdx, 16);
        ctx.set(Rsi, 0);
        assert_eq!(cb_video_recording_get_info(&mut ctx), ERR_INVALID);
        ctx.set(Rsi, BASE + 0x100);
        ctx.set(Rdi, id + 1);
        assert_eq!(cb_video_recording_get_info(&mut ctx), ERR_INVALID, "never set");
        for (r, v) in [(Rdi, id), (Rsi, 0), (Rdx, 0)] {
            ctx.set(r, v);
        }
        assert_eq!(cb_video_recording_set_info(&mut ctx), 0, "an empty value");
        ctx.set(Rdx, 4);
        assert_eq!(cb_video_recording_set_info(&mut ctx), ERR_INVALID);

        // No recording session opens; the status stays "none".
        for (r, v) in [(Rdi, 0), (Rsi, 0)] {
            ctx.set(r, v);
        }
        assert_eq!(cb_video_recording_query_mem_size(&mut ctx), ERR_INVALID);
        assert_eq!(cb_video_recording_open(&mut ctx), ERR_INVALID);
        for (r, v) in [(Rdi, BASE), (Rsi, BASE + 0x100)] {
            ctx.set(r, v);
        }
        assert_eq!(cb_video_recording_query_mem_size(&mut ctx), ERR_NOT_SUPPORTED);
        assert_eq!(cb_video_recording_open(&mut ctx), ERR_NOT_SUPPORTED);
        assert_eq!(cb_video_recording_session(&mut ctx), ERR_NOT_SUPPORTED);
        assert_eq!(cb_video_recording_get_status(&mut ctx), 0);
    }

    #[test]
    fn random_limits() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A60_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, BASE);
        ctx.set(Rsi, 65);
        assert_eq!(cb_random_get_random_number(&mut ctx), RANDOM_ERROR_INVALID);
        ctx.set(Rsi, 64);
        assert_eq!(cb_random_get_random_number(&mut ctx), 0);
        assert_ne!(ctx.read_bytes(BASE, 64).unwrap(), vec![0u8; 64]);
        ctx.set(Rdi, 0);
        ctx.set(Rsi, 0);
        assert_eq!(cb_random_get_random_number(&mut ctx), 0);
    }
}
