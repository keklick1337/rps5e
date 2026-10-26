// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceShareUtility (imported as libSceShare.native) — capture/share
//! feature control — plus libSceSharePlay. Settings are accepted; nothing
//! is ever captured, and the running status reports no active share
//! features. No Share Play session can start.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceShareUtility";
const OK: HleResult = 0;

const fn err(code: u32) -> HleResult {
    code as i32 as i64
}

pub const ERR_INVALID_PARAM: HleResult = err(0x8196_0002);
pub const ERR_NOT_SUPPORTED: HleResult = err(0x8196_0007);
/// SCE_SHARE_REQUEST_ID_INVALID.
const REQUEST_ID_INVALID: i32 = -1;
/// SceShareCurrentStatus (union of the recording status and 16 reserved bytes).
const CURRENT_STATUS_SIZE: usize = 16;

const MAX_CONTENT_PARAM_BYTES: usize = 4096;

static CONTENT_PARAM: Mutex<String> = Mutex::new(String::new());

/// Feature flag sets must name at least one feature.
fn feature_flags_valid(flags: u32) -> bool {
    flags != 0
}

fn cb_ok(_ctx: &mut CpuContext) -> HleResult {
    OK
}

/// sceShareSetContentParam(const char* param): kept as text.
fn cb_set_content_param(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return ERR_INVALID_PARAM;
    }
    let text = ctx.read_cstring(addr, MAX_CONTENT_PARAM_BYTES).unwrap_or_default();
    log::debug!("[share] content param ({} bytes)", text.len());
    *CONTENT_PARAM.lock().unwrap_or_else(|p| p.into_inner()) = text;
    OK
}

/// sceShareSetContentParamForApplicationTitle(const char* title)
fn cb_set_content_param_for_application_title(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 { ERR_INVALID_PARAM } else { OK }
}

/// sceShareFeaturePermit / sceShareFeatureProhibit(u32 featureFlags)
fn cb_feature_flags(ctx: &mut CpuContext) -> HleResult {
    if feature_flags_valid(ctx.rdi() as u32) { OK } else { ERR_INVALID_PARAM }
}

/// sceShareSetCaptureSource(u32 featureFlags, const void* tapPoint)
fn cb_set_capture_source(ctx: &mut CpuContext) -> HleResult {
    cb_feature_flags(ctx)
}

/// sceShareGetCurrentStatus(u32 featureFlag, SceShareCurrentStatus*): nothing recording.
fn cb_get_current_status(ctx: &mut CpuContext) -> HleResult {
    let (flags, status) = (ctx.rdi() as u32, ctx.rsi());
    if !feature_flags_valid(flags) || status == 0 {
        return ERR_INVALID_PARAM;
    }
    if ctx.write_bytes(status, &[0u8; CURRENT_STATUS_SIZE]) { OK } else { ERR_INVALID_PARAM }
}

/// sceShareGetRunningStatus(uint32_t* featureFlags): nothing running.
fn cb_get_running_status(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return ERR_INVALID_PARAM;
    }
    if ctx.write_u32(addr, 0) { OK } else { ERR_INVALID_PARAM }
}

/// Screenshot and video clip captures (param, int32_t* reqId): no capture
/// service exists, the request id is invalidated.
fn cb_capture(ctx: &mut CpuContext) -> HleResult {
    let req = ctx.rsi();
    if req != 0 && !ctx.write_bytes(req, &REQUEST_ID_INVALID.to_le_bytes()) {
        return ERR_INVALID_PARAM;
    }
    log::debug!("[share] capture is not supported");
    ERR_NOT_SUPPORTED
}

/// sceShareOpenMenuForContent(contentId): no share menu.
fn cb_open_menu_for_content(_ctx: &mut CpuContext) -> HleResult {
    ERR_NOT_SUPPORTED
}

// ---------------------------------------------------------------------------
// libSceSharePlay
// ---------------------------------------------------------------------------

const SHARE_PLAY_LIB: &str = "libSceSharePlay";
pub const SHARE_PLAY_ERR_INVALID_ARGS: HleResult = err(0x810E_0001);
pub const SHARE_PLAY_ERR_ALREADY_INITIALIZED: HleResult = err(0x810E_0003);
pub const SHARE_PLAY_ERR_NOT_INITIALIZED: HleResult = err(0x810E_0004);
/// Smallest heap a caller may pass.
const SHARE_PLAY_MIN_HEAP: u64 = 6 * 1024;
/// SceSharePlayConnectionInfoA: status, mode, two 20-byte online ids, two
/// account ids (0x30, 0x38), host / visitor user ids (0x40, 0x44).
pub const SHARE_PLAY_CONNECTION_INFO_A_SIZE: usize = 72;

static SHARE_PLAY_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// sceSharePlayInitialize(void* heap, size_t heapSize): the heap may be
/// null, but a size is always required; a given heap must hold at least
/// 6 KiB.
fn cb_share_play_initialize(ctx: &mut CpuContext) -> HleResult {
    if SHARE_PLAY_INITIALIZED.load(Ordering::Acquire) {
        return SHARE_PLAY_ERR_ALREADY_INITIALIZED;
    }
    if ctx.rsi() == 0 || (ctx.rdi() != 0 && ctx.rsi() < SHARE_PLAY_MIN_HEAP) {
        return SHARE_PLAY_ERR_INVALID_ARGS;
    }
    if SHARE_PLAY_INITIALIZED.swap(true, Ordering::AcqRel) {
        return SHARE_PLAY_ERR_ALREADY_INITIALIZED;
    }
    OK
}

fn cb_share_play_terminate(_ctx: &mut CpuContext) -> HleResult {
    if SHARE_PLAY_INITIALIZED.swap(false, Ordering::AcqRel) { OK } else { SHARE_PLAY_ERR_NOT_INITIALIZED }
}

/// The connection info of "no session": everything zero, user ids invalid.
pub fn no_connection_info() -> [u8; SHARE_PLAY_CONNECTION_INFO_A_SIZE] {
    let mut info = [0u8; SHARE_PLAY_CONNECTION_INFO_A_SIZE];
    info[0x40..0x44].copy_from_slice(&(-1i32).to_le_bytes());
    info[0x44..0x48].copy_from_slice(&(-1i32).to_le_bytes());
    info
}

/// sceSharePlayGetCurrentConnectionInfoA(SceSharePlayConnectionInfoA*)
fn cb_share_play_get_current_connection_info_a(ctx: &mut CpuContext) -> HleResult {
    if !SHARE_PLAY_INITIALIZED.load(Ordering::Acquire) {
        return SHARE_PLAY_ERR_NOT_INITIALIZED;
    }
    let info = ctx.rdi();
    if info == 0 || !ctx.write_bytes(info, &no_connection_info()) {
        return SHARE_PLAY_ERR_INVALID_ARGS;
    }
    OK
}

/// SceSharePlayConnectionInfo: status, mode, two 20-byte online ids, host /
/// visitor user ids (0x30, 0x34); the A form without the account ids.
pub const SHARE_PLAY_CONNECTION_INFO_SIZE: usize = 56;

/// sceSharePlayGetCurrentConnectionInfo(SceSharePlayConnectionInfo*): the
/// checks and "no session" values of the A form.
fn cb_share_play_get_current_connection_info(ctx: &mut CpuContext) -> HleResult {
    if !SHARE_PLAY_INITIALIZED.load(Ordering::Acquire) {
        return SHARE_PLAY_ERR_NOT_INITIALIZED;
    }
    let mut info = [0u8; SHARE_PLAY_CONNECTION_INFO_SIZE];
    info[0x30..0x34].copy_from_slice(&(-1i32).to_le_bytes());
    info[0x34..0x38].copy_from_slice(&(-1i32).to_le_bytes());
    let out = ctx.rdi();
    if out == 0 || !ctx.write_bytes(out, &info) {
        return SHARE_PLAY_ERR_INVALID_ARGS;
    }
    OK
}

/// Whether the visitor's controller is prohibited (no visitor ever joins).
static SHARE_PLAY_CONTROLLER_PROHIBITED: AtomicBool = AtomicBool::new(false);

/// sceSharePlayProhibitController(bool prohibit)
fn cb_share_play_prohibit_controller(ctx: &mut CpuContext) -> HleResult {
    let prohibit = ctx.rdi();
    if prohibit > 1 {
        return SHARE_PLAY_ERR_INVALID_ARGS;
    }
    SHARE_PLAY_CONTROLLER_PROHIBITED.store(prohibit != 0, Ordering::Release);
    OK
}

const SHARE_PLAY_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("isruqthpYcw", "sceSharePlayInitialize", cb_share_play_initialize),
    ("UaLjloJinow", "sceSharePlayTerminate", cb_share_play_terminate),
    ("+MCXJlWdi+s", "sceSharePlayGetCurrentConnectionInfoA", cb_share_play_get_current_connection_info_a),
    ("OOrLKB0bSDs", "sceSharePlayGetCurrentConnectionInfo", cb_share_play_get_current_connection_info),
    ("FzQS6DREDfk", "sceSharePlayProhibitController", cb_share_play_prohibit_controller),
];

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("nBDD66kiFW8", "sceShareInitialize", cb_ok),
    ("0IL1keINExQ", "sceShareTerminate", cb_ok),
    ("ErH6tKS7fzE", "sceShareCaptureScreenshot", cb_capture),
    ("GQTObcITIXI", "sceShareCaptureScreenshotExtended", cb_capture),
    ("4jt8pMDudgk", "sceShareCaptureVideoClip", cb_capture),
    ("AcDNpEpoT9U", "sceShareCaptureVideoClipExtended", cb_capture),
    ("8qAJ0Jd58-Q", "sceShareOpenMenuForContent", cb_open_menu_for_content),
    ("7QZtURYnXG4", "sceShareSetContentParam", cb_set_content_param),
    ("ORspsWDXPps", "sceShareSetContentParamForApplicationTitle", cb_set_content_param_for_application_title),
    ("Sygnk9dr5WQ", "sceShareRegisterContentEventCallback", cb_ok),
    ("KnsfHKmZqFA", "sceShareUnregisterContentEventCallback", cb_ok),
    ("5wjxESwX68I", "sceShareFeatureProhibit", cb_feature_flags),
    ("YBiIdcDPrxs", "sceShareFeaturePermit", cb_feature_flags),
    ("kCurUZVFqcI", "sceShareSetCaptureSource", cb_set_capture_source),
    ("T64o-315wbg", "sceShareSetScreenshotOverlayImage", cb_ok),
    ("QNop2YAtIDE", "sceShareGetCurrentStatus", cb_get_current_status),
    ("crFxyW3HdK0", "sceShareGetRunningStatus", cb_get_running_status),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
    for &(nid, name, cb) in SHARE_PLAY_EXPORTS {
        reg.register_fn(nid, name, SHARE_PLAY_LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS.iter().chain(SHARE_PLAY_EXPORTS) {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn share_calls() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A5F_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, b"{\"k\":1}\0");
        ctx.set(Rdi, BASE);
        assert_eq!(cb_set_content_param(&mut ctx), OK);
        assert_eq!(CONTENT_PARAM.lock().unwrap().as_str(), "{\"k\":1}");
        ctx.set(Rdi, 0);
        assert_eq!(cb_set_content_param(&mut ctx), ERR_INVALID_PARAM);
        assert_eq!(cb_set_content_param_for_application_title(&mut ctx), ERR_INVALID_PARAM);
        assert_eq!(cb_feature_flags(&mut ctx), ERR_INVALID_PARAM, "no feature named");
        assert_eq!(cb_set_capture_source(&mut ctx), ERR_INVALID_PARAM);
        assert_eq!(cb_get_running_status(&mut ctx), ERR_INVALID_PARAM);
        ctx.set(Rdi, 1);
        assert_eq!(cb_feature_flags(&mut ctx), OK);
        ctx.write_bytes(BASE + 0x40, b"title\0");
        ctx.set(Rdi, BASE + 0x40);
        assert_eq!(cb_set_content_param_for_application_title(&mut ctx), OK);

        let req = BASE + 0x100;
        ctx.write_u32(req, 5);
        ctx.set(Rsi, req);
        assert_eq!(cb_capture(&mut ctx), ERR_NOT_SUPPORTED);
        assert_eq!(ctx.read_u32(req), Some(u32::MAX));

        let status = BASE + 0x200;
        ctx.write_bytes(status, &[0xAA; 20]);
        ctx.set(Rsi, status);
        assert_eq!(cb_get_current_status(&mut ctx), OK);
        assert_eq!(ctx.read_bytes(status, 20).unwrap()[..16], [0u8; 16]);
        assert_eq!(ctx.read_bytes(status + 16, 1).unwrap()[0], 0xAA);
        ctx.set(Rsi, 0);
        assert_eq!(cb_get_current_status(&mut ctx), ERR_INVALID_PARAM);
    }

    #[test]
    fn share_play_lifecycle() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D40_0600_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_share_play_get_current_connection_info_a(&mut ctx), SHARE_PLAY_ERR_NOT_INITIALIZED);
        assert_eq!(cb_share_play_terminate(&mut ctx), SHARE_PLAY_ERR_NOT_INITIALIZED);

        ctx.set(Rdi, BASE);
        ctx.set(Rsi, 6 * 1024 - 1);
        assert_eq!(cb_share_play_initialize(&mut ctx), SHARE_PLAY_ERR_INVALID_ARGS);
        ctx.set(Rsi, 0);
        assert_eq!(cb_share_play_initialize(&mut ctx), SHARE_PLAY_ERR_INVALID_ARGS, "a heap needs a size");
        ctx.set(Rdi, 0);
        assert_eq!(cb_share_play_initialize(&mut ctx), SHARE_PLAY_ERR_INVALID_ARGS, "no heap still needs a size");
        ctx.set(Rsi, 0x1800);
        assert_eq!(cb_share_play_initialize(&mut ctx), OK, "no heap is allowed");
        assert_eq!(cb_share_play_initialize(&mut ctx), SHARE_PLAY_ERR_ALREADY_INITIALIZED);

        ctx.write_bytes(BASE, &[0x11; SHARE_PLAY_CONNECTION_INFO_A_SIZE]);
        ctx.set(Rdi, 0);
        assert_eq!(cb_share_play_get_current_connection_info_a(&mut ctx), SHARE_PLAY_ERR_INVALID_ARGS);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_share_play_get_current_connection_info_a(&mut ctx), OK);
        let info = ctx.read_bytes(BASE, SHARE_PLAY_CONNECTION_INFO_A_SIZE).unwrap();
        assert!(info[..0x40].iter().all(|&b| b == 0));
        assert_eq!(ctx.read_u32(BASE + 0x40), Some(u32::MAX));
        assert_eq!(ctx.read_u32(BASE + 0x44), Some(u32::MAX));

        // The 56-byte form: same checks, user ids at 0x30 / 0x34, nothing
        // past the struct written.
        ctx.write_bytes(BASE, &[0x5A; SHARE_PLAY_CONNECTION_INFO_SIZE + 8]);
        ctx.set(Rdi, 0);
        assert_eq!(cb_share_play_get_current_connection_info(&mut ctx), SHARE_PLAY_ERR_INVALID_ARGS);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_share_play_get_current_connection_info(&mut ctx), OK);
        let info = ctx.read_bytes(BASE, SHARE_PLAY_CONNECTION_INFO_SIZE + 8).unwrap();
        assert!(info[..0x30].iter().all(|&b| b == 0));
        assert_eq!(&info[0x30..0x38], &[0xFF; 8]);
        assert_eq!(&info[0x38..], &[0x5A; 8]);
        assert_eq!(cb_share_play_terminate(&mut ctx), OK);
        assert_eq!(cb_share_play_get_current_connection_info(&mut ctx), SHARE_PLAY_ERR_NOT_INITIALIZED);
    }
}
