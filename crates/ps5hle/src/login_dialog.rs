// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceLoginDialog: user selection dialog. With a single local user the
//! dialog finishes on the first status update with that user selected (or
//! the initial focus the title asked for).

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};
use std::sync::Mutex;

const LIB: &str = "libSceLoginDialog";

pub const STATUS_NONE: i32 = 0;
pub const STATUS_INITIALIZED: i32 = 1;
pub const STATUS_RUNNING: i32 = 2;
pub const STATUS_FINISHED: i32 = 3;
const RESULT_OK: i32 = 0;
/// SceLoginDialogMode.
const MODE_ALL_USERS: i32 = 0;
const MODE_NOT_LOGGED_IN: i32 = 1;
const USER_INVALID: i32 = -1;

const ERROR_NOT_INITIALIZED: i32 = 0x8134_0001u32 as i32;
const ERROR_ALREADY_INITIALIZED: i32 = 0x8134_0002u32 as i32;
const ERROR_PARAM_INVALID: i32 = 0x8134_0003u32 as i32;
const ERROR_INVALID_STATE: i32 = 0x8134_0005u32 as i32;

/// SceLoginDialogParam { size; mode; excludeUsersFromLoginList[4];
/// excludeUsersFromLogoutList[4]; initialFocus @0x28; reserved[5]; }.
pub const PARAM_SIZE: usize = 64;
const PARAM_INITIAL_FOCUS_OFFSET: usize = 0x28;
/// SceLoginDialogResult { result; selectedUser; reserved[2]; }.
pub const RESULT_SIZE: usize = 16;

struct State {
    status: i32,
    selected_user: i32,
}

static STATE: Mutex<State> = Mutex::new(State { status: STATUS_NONE, selected_user: USER_INVALID });

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|p| p.into_inner())
}

fn user_id() -> i32 {
    crate::user_service::PRIMARY_USER_ID
}

pub fn initialize() -> i32 {
    let mut s = state();
    if s.status != STATUS_NONE {
        return ERROR_ALREADY_INITIALIZED;
    }
    *s = State { status: STATUS_INITIALIZED, selected_user: user_id() };
    0
}

pub fn terminate() -> i32 {
    let mut s = state();
    if s.status == STATUS_NONE {
        return ERROR_NOT_INITIALIZED;
    }
    *s = State { status: STATUS_NONE, selected_user: USER_INVALID };
    0
}

pub fn open(ctx: &CpuContext, param: u64) -> i32 {
    let mut s = state();
    if s.status != STATUS_INITIALIZED && s.status != STATUS_FINISHED {
        return ERROR_INVALID_STATE;
    }
    if param == 0 {
        return ERROR_PARAM_INVALID;
    }
    let Some(p) = ctx.read_bytes(param, PARAM_SIZE) else {
        return ERROR_PARAM_INVALID;
    };
    let i32_at = |o: usize| i32::from_le_bytes(p[o..o + 4].try_into().unwrap());
    let (size, mode, focus) = (i32_at(0), i32_at(4), i32_at(PARAM_INITIAL_FOCUS_OFFSET));
    if size != PARAM_SIZE as i32 || (mode != MODE_ALL_USERS && mode != MODE_NOT_LOGGED_IN) {
        return ERROR_PARAM_INVALID;
    }
    s.selected_user = if focus != USER_INVALID { focus } else { user_id() };
    log::debug!("[LOGINDIALOG] open mode={mode} initialFocus={focus} selected={}", s.selected_user);
    s.status = STATUS_RUNNING;
    0
}

pub fn close() -> i32 {
    let mut s = state();
    if s.status == STATUS_NONE {
        return ERROR_NOT_INITIALIZED;
    }
    s.status = STATUS_FINISHED;
    0
}

pub fn update_status() -> i32 {
    let mut s = state();
    if s.status == STATUS_RUNNING {
        s.status = STATUS_FINISHED;
    }
    s.status
}

pub fn get_status() -> i32 {
    state().status
}

pub fn get_result(ctx: &CpuContext, result: u64) -> i32 {
    let s = state();
    if s.status == STATUS_NONE {
        return ERROR_NOT_INITIALIZED;
    }
    if result == 0 {
        return ERROR_PARAM_INVALID;
    }
    let mut r = [0u8; RESULT_SIZE];
    r[0..4].copy_from_slice(&RESULT_OK.to_le_bytes());
    r[4..8].copy_from_slice(&s.selected_user.to_le_bytes());
    if ctx.write_bytes(result, &r) { 0 } else { ERROR_PARAM_INVALID }
}

/// Zeroed param with size set and every user slot / the focus invalid.
pub fn param_initialize(ctx: &CpuContext, param: u64) {
    if param == 0 {
        return;
    }
    let mut p = [0u8; PARAM_SIZE];
    p[0..4].copy_from_slice(&(PARAM_SIZE as i32).to_le_bytes());
    for i in 0..9 {
        let o = 8 + i * 4; // both exclude lists, then initialFocus
        p[o..o + 4].copy_from_slice(&USER_INVALID.to_le_bytes());
    }
    ctx.write_bytes(param, &p);
}

fn cb_initialize(_ctx: &mut CpuContext) -> HleResult {
    initialize() as HleResult
}

fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    terminate() as HleResult
}

fn cb_open(ctx: &mut CpuContext) -> HleResult {
    open(ctx, ctx.rdi()) as HleResult
}

fn cb_close(_ctx: &mut CpuContext) -> HleResult {
    close() as HleResult
}

fn cb_update_status(_ctx: &mut CpuContext) -> HleResult {
    update_status() as HleResult
}

fn cb_get_status(_ctx: &mut CpuContext) -> HleResult {
    get_status() as HleResult
}

fn cb_get_result(ctx: &mut CpuContext) -> HleResult {
    get_result(ctx, ctx.rdi()) as HleResult
}

fn cb_param_initialize(ctx: &mut CpuContext) -> HleResult {
    param_initialize(ctx, ctx.rdi());
    0
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("qP-EvQRl2Hc", "sceLoginDialogInitialize", cb_initialize),
    ("vMQJRUKsf3U", "sceLoginDialogTerminate", cb_terminate),
    ("S56ra1+Tymg", "sceLoginDialogOpen", cb_open),
    ("F0XIzrG5yvw", "sceLoginDialogClose", cb_close),
    ("2rc+egSfb5A", "sceLoginDialogUpdateStatus", cb_update_status),
    ("HAiWUEwEfGo", "sceLoginDialogGetStatus", cb_get_status),
    ("Btkx21f1M8k", "sceLoginDialogGetResult", cb_get_result),
    ("3NPobi5lNmk", "sceLoginDialogParamInitialize", cb_param_initialize),
];

// libSceLoginService: the local users are signed in from the start, so
// the service has no state to set up or tear down.

fn cb_login_service_initialize(_ctx: &mut CpuContext) -> HleResult {
    0
}

fn cb_login_service_terminate(_ctx: &mut CpuContext) -> HleResult {
    0
}

const LOGIN_SERVICE_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("T9-NuSHAvcQ", "sceLoginServiceInitialize", cb_login_service_initialize),
    ("aR8+Hvghm0E", "sceLoginServiceTerminate", cb_login_service_terminate),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
    for &(nid, name, cb) in LOGIN_SERVICE_EXPORTS {
        reg.register_fn(nid, name, "libSceLoginService", cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Port of the guest login service test: initialize and terminate.
    #[test]
    fn login_service_lifecycle() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        assert_eq!(cb_login_service_initialize(&mut ctx), 0);
        assert_eq!(cb_login_service_terminate(&mut ctx), 0);
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS.iter().chain(LOGIN_SERVICE_EXPORTS) {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn lifecycle_selects_focus_or_primary_user() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D37_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let ctx = CpuContext::new(&vm);
        let result = BASE + 0x100;

        assert_eq!(terminate(), ERROR_NOT_INITIALIZED);
        assert_eq!(get_result(&ctx, result), ERROR_NOT_INITIALIZED);
        assert_eq!(open(&ctx, BASE), ERROR_INVALID_STATE);
        assert_eq!(initialize(), 0);
        assert_eq!(initialize(), ERROR_ALREADY_INITIALIZED);
        assert_eq!(open(&ctx, 0), ERROR_PARAM_INVALID);

        param_initialize(&ctx, BASE);
        assert_eq!(ctx.read_u32(BASE), Some(64));
        assert_eq!(ctx.read_u32(BASE + 8), Some(u32::MAX));
        assert_eq!(ctx.read_u32(BASE + 0x28), Some(u32::MAX));
        assert_eq!(ctx.read_u32(BASE + 0x2C), Some(0));
        ctx.write_u32(BASE + 4, 2);
        assert_eq!(open(&ctx, BASE), ERROR_PARAM_INVALID);
        ctx.write_u32(BASE + 4, MODE_NOT_LOGGED_IN as u32);
        assert_eq!(open(&ctx, BASE), 0);
        assert_eq!(get_status(), STATUS_RUNNING);
        assert_eq!(open(&ctx, BASE), ERROR_INVALID_STATE);
        assert_eq!(update_status(), STATUS_FINISHED);
        assert_eq!(get_result(&ctx, 0), ERROR_PARAM_INVALID);
        assert_eq!(get_result(&ctx, result), 0);
        assert_eq!(ctx.read_u32(result), Some(0));
        assert_eq!(ctx.read_u32(result + 4), Some(user_id() as u32));

        ctx.write_u32(BASE + 0x28, 0x1000_0003);
        assert_eq!(open(&ctx, BASE), 0);
        assert_eq!(close(), 0);
        assert_eq!(get_status(), STATUS_FINISHED);
        assert_eq!(get_result(&ctx, result), 0);
        assert_eq!(ctx.read_u32(result + 4), Some(0x1000_0003));
        assert_eq!(terminate(), 0);
        assert_eq!(get_status(), STATUS_NONE);
    }
}
