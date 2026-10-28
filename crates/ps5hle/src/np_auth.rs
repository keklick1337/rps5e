// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceNpAuth and libSceNpAuthAuthorizedAppDialog HLE for a user who is
//! signed out of PSN.
//!
//! libSceNpAuth tracks up to 16 requests (ids 0x10000001 + slot, the lowest
//! free slot first). Every call checks its parameter block first, then the
//! request. Each request carries one authorization: it completes with
//! SIGNED_OUT (ABORTED after sceNpAuthAbortRequest), and a second one on
//! the same request fails with INVALID_ARGUMENT. Synchronous requests return
//! the result; asynchronous ones return 0 and report it through
//! sceNpAuthPollAsync / sceNpAuthWaitAsync. The forms that name the user by
//! online id fail resolving it (SIGNED_OUT) before they touch the request.
//! Output buffers (codes, tokens, issuer ids) are written only on success.
//!
//! The authorized-app dialog follows the common-dialog status machine: Open
//! validates its parameter block and starts the dialog (RUNNING);
//! sceNpAuthAuthorizedAppDialogUpdateStatus advances it and it finishes at
//! once, because without PSN the user can only dismiss the system's
//! sign-in prompt. Its result is "user canceled", as for the sign-in
//! dialog. GetStatus only reports the status, it never advances it.
//!
//! Where the console's behaviour depends on the SDK the title was built
//! with, the behaviour of SDK 6.50 and later is modelled: a poll before the
//! authorization call reports RUNNING, and a failed authorization leaves the
//! issuer id untouched.

use std::sync::{Condvar, Mutex};

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceNpAuth";
const DIALOG_LIB: &str = "libSceNpAuthAuthorizedAppDialog";
const OK: HleResult = 0;

const ERR_INVALID_ARGUMENT: i32 = 0x8055_0301u32 as i32;
const ERR_INVALID_SIZE: i32 = 0x8055_0302u32 as i32;
const ERR_ABORTED: i32 = 0x8055_0304u32 as i32;
const ERR_REQUEST_MAX: i32 = 0x8055_0305u32 as i32;
const ERR_REQUEST_NOT_FOUND: i32 = 0x8055_0306u32 as i32;
const ERR_INVALID_ID: i32 = 0x8055_0307u32 as i32;
const ERR_NP_SIGNED_OUT: i32 = 0x8055_0006u32 as i32;
const ERR_MEMORY_FAULT: i32 = 0x8002_000Eu32 as i32;

const MAX_REQUESTS: usize = 16;
const REQUEST_ID_OFFSET: i32 = 0x1000_0000;

/// SceNpAuthCreateAsyncRequestParameter: size, cpuAffinityMask, threadPriority.
const ASYNC_PARAM_SIZE: u64 = 24;
/// SceNpAuthGetAuthorizationCodeParameter(V3): size, userId (or the online
/// id pointer of the old form) @8, clientId* @16, scope* @24.
const AUTH_CODE_PARAM_SIZE: u64 = 32;
/// SceNpAuthGetIdTokenParameter(V3): size, userId / onlineId* @8,
/// clientId* @16, clientSecret* @24, scope* @32.
const ID_TOKEN_PARAM_SIZE: u64 = 40;
/// SceNpAuthGetAuthorizedAppCodeParameter: size, userId @8, clientId* @16,
/// scope* @24, a 0/1 option @32.
const APP_CODE_PARAM_SIZE: u64 = 40;

/// sceNpAuthSetTimeout: a non-zero resolve timeout must be at least 1 s,
/// the connect, send and receive timeouts at least 10 s (microseconds).
const MIN_RESOLVE_TIMEOUT_US: u32 = 1_000_000;
const MIN_TRANSFER_TIMEOUT_US: u32 = 10_000_000;

/// SCE_NP_POLL_ASYNC_RET_RUNNING.
const POLL_ASYNC_RUNNING: HleResult = 1;

#[derive(Clone, Copy, Debug, Default)]
struct AuthRequest {
    is_async: bool,
    /// The request's one authorization call was made.
    used: bool,
    aborted: bool,
    /// An asynchronous request has a result to report.
    done: bool,
    result: i32,
    /// sceNpAuthPollAsync already reported this finished request running
    /// once (its worker has not caught up yet).
    polled: bool,
}

struct AuthState {
    requests: Vec<Option<AuthRequest>>,
}

impl AuthState {
    const fn new() -> Self {
        Self { requests: Vec::new() }
    }

    fn active(&self) -> usize {
        self.requests.iter().filter(|r| r.is_some()).count()
    }

    fn create(&mut self, is_async: bool) -> i32 {
        if self.active() >= MAX_REQUESTS {
            return ERR_REQUEST_MAX;
        }
        let request = Some(AuthRequest { is_async, ..Default::default() });
        let index = match self.requests.iter().position(Option::is_none) {
            Some(i) => {
                self.requests[i] = request;
                i
            }
            None => {
                self.requests.push(request);
                self.requests.len() - 1
            }
        };
        index as i32 + REQUEST_ID_OFFSET + 1
    }

    fn find(&mut self, id: i32) -> Result<&mut AuthRequest, i32> {
        if id <= 0 {
            return Err(ERR_INVALID_ARGUMENT);
        }
        let index = id.wrapping_sub(REQUEST_ID_OFFSET + 1);
        if index < 0 {
            return Err(ERR_REQUEST_NOT_FOUND);
        }
        self.requests.get_mut(index as usize).and_then(Option::as_mut).ok_or(ERR_REQUEST_NOT_FOUND)
    }

    /// sceNpAuthDeleteRequest: the request is aborted and removed.
    fn delete(&mut self, id: i32) -> i32 {
        match self.find(id) {
            Ok(_) => {
                self.requests[(id - REQUEST_ID_OFFSET - 1) as usize] = None;
                0
            }
            Err(e) => e,
        }
    }

    /// sceNpAuthAbortRequest: marks the request; a later authorization on
    /// it completes with ABORTED, a finished one keeps its result.
    fn abort(&mut self, id: i32) -> i32 {
        match self.find(id) {
            Ok(r) => {
                r.aborted = true;
                0
            }
            Err(e) => e,
        }
    }

    /// The authorization on request `id`: the call's return value and
    /// whether the authorization ran (a used request does not run it).
    fn authorize(&mut self, id: i32) -> (i32, bool) {
        let r = match self.find(id) {
            Ok(r) => r,
            Err(e) => return (e, false),
        };
        if r.used {
            r.result = ERR_INVALID_ARGUMENT;
            r.done |= r.is_async;
            return (ERR_INVALID_ARGUMENT, false);
        }
        r.used = true;
        r.result = if r.aborted { ERR_ABORTED } else { ERR_NP_SIGNED_OUT };
        if r.is_async {
            r.done = true;
            (0, true)
        } else {
            (r.result, true)
        }
    }

    /// An asynchronous request's result, `None` while it is running.
    fn async_result(&mut self, id: i32) -> Result<Option<i32>, i32> {
        let r = self.find(id)?;
        if !r.is_async {
            return Err(ERR_INVALID_ID);
        }
        Ok(r.done.then_some(r.result))
    }

    /// sceNpAuthPollAsync: RUNNING before the authorization call and on the
    /// first poll after it; later polls give its result.
    fn poll_result(&mut self, id: i32) -> Result<Option<i32>, i32> {
        if self.async_result(id)?.is_none() {
            return Ok(None);
        }
        let r = self.find(id)?;
        if !r.polled {
            r.polled = true;
            return Ok(None);
        }
        Ok(Some(r.result))
    }
}

static STATE: Mutex<AuthState> = Mutex::new(AuthState::new());
/// Signalled whenever a request finishes or is deleted.
static STATE_CV: Condvar = Condvar::new();

fn state() -> std::sync::MutexGuard<'static, AuthState> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Drop all requests and the dialog (emulator runtime reset).
pub fn reset_runtime_state() {
    *state() = AuthState::new();
    *dialog() = DialogState::new();
}

fn request_id(ctx: &CpuContext) -> i32 {
    ctx.rdi() as i32
}

fn u64_at(raw: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(raw[off..off + 8].try_into().unwrap())
}

fn i32_at(raw: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(raw[off..off + 4].try_into().unwrap())
}

/// A parameter block that starts with its u64 size, which must be `size`.
fn read_param(ctx: &CpuContext, addr: u64, size: u64) -> Result<Vec<u8>, i32> {
    match ctx.read_u64(addr) {
        None => Err(ERR_MEMORY_FAULT),
        Some(s) if s != size => Err(ERR_INVALID_SIZE),
        Some(_) => ctx.read_bytes(addr, size as usize).ok_or(ERR_MEMORY_FAULT),
    }
}

fn cb_create_request(_ctx: &mut CpuContext) -> HleResult {
    state().create(false) as i64
}

/// sceNpAuthCreateAsyncRequest(const SceNpAuthCreateAsyncRequestParameter*)
fn cb_create_async_request(ctx: &mut CpuContext) -> HleResult {
    let param = ctx.rdi();
    if param == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    if let Err(e) = read_param(ctx, param, ASYNC_PARAM_SIZE) {
        return e as i64;
    }
    state().create(true) as i64
}

fn cb_delete_request(ctx: &mut CpuContext) -> HleResult {
    let r = state().delete(request_id(ctx));
    STATE_CV.notify_all();
    r as i64
}

fn cb_abort_request(ctx: &mut CpuContext) -> HleResult {
    state().abort(request_id(ctx)) as i64
}

/// sceNpAuthSetTimeout(reqId, resolveRetry, resolveTimeout, connectTimeout,
/// sendTimeout, recvTimeout): at least one value is set, the retry count is
/// not negative and every timeout is 0 (default) or at least its minimum.
fn cb_set_timeout(ctx: &mut CpuContext) -> HleResult {
    let retry = ctx.rsi() as i32;
    let timeouts = [ctx.rdx() as u32, ctx.rcx() as u32, ctx.r8() as u32, ctx.r9() as u32];
    if retry == 0 && timeouts.iter().all(|&t| t == 0) {
        return ERR_INVALID_ARGUMENT as i64;
    }
    let too_short = timeouts.iter().enumerate().any(|(i, &t)| {
        let min = if i == 0 { MIN_RESOLVE_TIMEOUT_US } else { MIN_TRANSFER_TIMEOUT_US };
        t != 0 && t < min
    });
    if retry < 0 || too_short {
        return ERR_INVALID_ARGUMENT as i64;
    }
    match state().find(request_id(ctx)) {
        Ok(_) => OK,
        Err(e) => e as i64,
    }
}

/// sceNpAuthPollAsync(reqId, int* result): RUNNING, or 0 with the result.
fn cb_poll_async(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    match state().poll_result(request_id(ctx)) {
        Ok(None) => POLL_ASYNC_RUNNING,
        Ok(Some(result)) => {
            if ctx.write_u32(out, result as u32) {
                OK
            } else {
                ERR_MEMORY_FAULT as i64
            }
        }
        Err(e) => e as i64,
    }
}

/// sceNpAuthWaitAsync(reqId, int* result): blocks until the request's
/// authorization finished (another thread may still have to make the call).
fn cb_wait_async(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    let id = request_id(ctx);
    let mut st = state();
    let result = loop {
        match st.async_result(id) {
            Ok(Some(result)) => break result,
            Ok(None) => st = crate::signals::wait_cv(&STATE, &STATE_CV, st, None).0,
            Err(e) => return e as i64,
        }
    };
    drop(st);
    if ctx.write_u32(out, result as u32) { OK } else { ERR_MEMORY_FAULT as i64 }
}

/// Run the authorization of request `id` and wake async waiters.
fn authorize(id: i32) -> (i32, bool) {
    let r = state().authorize(id);
    STATE_CV.notify_all();
    r
}

/// The common checks of a V3 / A authorization: `param` and `out` are
/// set, the block has `size` bytes, the user id is valid and the pointers
/// at `pointers` are set.
fn check_user_param(ctx: &CpuContext, param: u64, out: u64, size: u64, pointers: &[usize]) -> Result<Vec<u8>, i32> {
    if param == 0 || out == 0 {
        return Err(ERR_INVALID_ARGUMENT);
    }
    let raw = read_param(ctx, param, size)?;
    if i32_at(&raw, 8) == -1 || pointers.iter().any(|&off| u64_at(&raw, off) == 0) {
        return Err(ERR_INVALID_ARGUMENT);
    }
    Ok(raw)
}

/// The forms that name the user by online id: the parameter block is
/// checked like the V3 one, then resolving the online id of a signed-out
/// user fails; the request is not touched.
fn check_online_id_param(ctx: &CpuContext, param: u64, out: u64, size: u64, pointers: &[usize]) -> i32 {
    if param == 0 || out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let raw = match read_param(ctx, param, size) {
        Ok(raw) => raw,
        Err(e) => return e,
    };
    if pointers.iter().any(|&off| u64_at(&raw, off) == 0) {
        return ERR_INVALID_ARGUMENT;
    }
    ERR_NP_SIGNED_OUT
}

/// sceNpAuthGetAuthorizationCodeV3 / A(reqId, param, SceNpAuthorizationCode*,
/// int* issuerId): the code and the issuer id are written only on success.
fn cb_get_authorization_code_v3(ctx: &mut CpuContext) -> HleResult {
    if let Err(e) = check_user_param(ctx, ctx.rsi(), ctx.rdx(), AUTH_CODE_PARAM_SIZE, &[16, 24]) {
        return e as i64;
    }
    authorize(request_id(ctx)).0 as i64
}

/// sceNpAuthGetAuthorizationCode(reqId, param with onlineId*, code*, int* issuerId)
fn cb_get_authorization_code(ctx: &mut CpuContext) -> HleResult {
    check_online_id_param(ctx, ctx.rsi(), ctx.rdx(), AUTH_CODE_PARAM_SIZE, &[8, 16, 24]) as i64
}

/// sceNpAuthGetIdTokenV3 / A(reqId, param, SceNpIdToken*)
fn cb_get_id_token_v3(ctx: &mut CpuContext) -> HleResult {
    if let Err(e) = check_user_param(ctx, ctx.rsi(), ctx.rdx(), ID_TOKEN_PARAM_SIZE, &[16, 24, 32]) {
        return e as i64;
    }
    authorize(request_id(ctx)).0 as i64
}

/// sceNpAuthGetIdToken(reqId, param with onlineId*, SceNpIdToken*)
fn cb_get_id_token(ctx: &mut CpuContext) -> HleResult {
    check_online_id_param(ctx, ctx.rsi(), ctx.rdx(), ID_TOKEN_PARAM_SIZE, &[8, 16, 24, 32]) as i64
}

/// sceNpAuthGetAuthorizedAppCode(reqId, param, SceNpAuthorizationCode*,
/// int* issuerId, bool* needsAuthorization): `*needsAuthorization` tells
/// whether the failure was "the app is not authorized"; a signed-out user
/// fails for another reason, so it is false.
fn cb_get_authorized_app_code(ctx: &mut CpuContext) -> HleResult {
    let (param, flag) = (ctx.rsi(), ctx.r8());
    if flag == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    let raw = match check_user_param(ctx, param, ctx.rdx(), APP_CODE_PARAM_SIZE, &[16, 24]) {
        Ok(raw) => raw,
        Err(e) => return e as i64,
    };
    if u32::from_le_bytes(raw[32..36].try_into().unwrap()) > 1 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    let (ret, ran) = authorize(request_id(ctx));
    if ran && !ctx.write_bytes(flag, &[0]) {
        return ERR_MEMORY_FAULT as i64;
    }
    ret as i64
}

// ---------------------------------------------------------------------------
// libSceNpAuthAuthorizedAppDialog
// ---------------------------------------------------------------------------

use crate::common_dialog::{STATUS_FINISHED, STATUS_INITIALIZED, STATUS_NONE, STATUS_RUNNING};

// SCE_COMMON_DIALOG_ERROR_*.
const CD_ERR_NOT_INITIALIZED: HleResult = 0x80B8_0003u32 as i32 as i64;
const CD_ERR_ALREADY_INITIALIZED: HleResult = 0x80B8_0004u32 as i32 as i64;
const CD_ERR_NOT_FINISHED: HleResult = 0x80B8_0005u32 as i32 as i64;
const CD_ERR_BUSY: HleResult = 0x80B8_0008u32 as i32 as i64;
const CD_ERR_PARAM_INVALID: HleResult = 0x80B8_000Au32 as i32 as i64;
const CD_ERR_NOT_RUNNING: HleResult = 0x80B8_000Bu32 as i32 as i64;
const CD_ERR_ARG_NULL: HleResult = 0x80B8_000Du32 as i32 as i64;

/// SceNpAuthAuthorizedAppDialogParam: SceCommonDialogBaseParam (48 bytes),
/// size (u64) @0x30, userId @0x38, const SceNpClientId* @0x40,
/// const char* scope @0x48, a 0/1 option @0x50, reserved[32] @0x58.
const DLG_PARAM_SIZE: u64 = 0x78;
const DLG_OFF_SIZE: usize = 0x30;
const DLG_OFF_USER_ID: usize = 0x38;
const DLG_OFF_CLIENT_ID: usize = 0x40;
const DLG_OFF_SCOPE: usize = 0x48;
const DLG_OFF_OPTION: usize = 0x50;
const DLG_OFF_RESERVED: usize = 0x58;
/// Longest scope string accepted.
const DLG_SCOPE_MAX: usize = 0x400;
/// SCE_COMMON_DIALOG_RESULT_USER_CANCELED.
const DLG_RESULT_USER_CANCELED: i32 = 1;

struct DialogState {
    status: i32,
    result: i32,
}

impl DialogState {
    const fn new() -> Self {
        Self { status: STATUS_NONE, result: 0 }
    }
}

static DIALOG: Mutex<DialogState> = Mutex::new(DialogState::new());

fn dialog() -> std::sync::MutexGuard<'static, DialogState> {
    DIALOG.lock().unwrap_or_else(|e| e.into_inner())
}

fn cb_dialog_initialize(_ctx: &mut CpuContext) -> HleResult {
    let mut d = dialog();
    if d.status != STATUS_NONE {
        return CD_ERR_ALREADY_INITIALIZED;
    }
    *d = DialogState { status: STATUS_INITIALIZED, result: 0 };
    OK
}

/// Terminate also ends a running dialog.
fn cb_dialog_terminate(_ctx: &mut CpuContext) -> HleResult {
    let mut d = dialog();
    if d.status == STATUS_NONE {
        return CD_ERR_NOT_INITIALIZED;
    }
    *d = DialogState::new();
    OK
}

/// The parameter checks of Open, in the console's order; they come before
/// the initialisation check.
fn check_dialog_param(ctx: &CpuContext, param: u64) -> HleResult {
    if param == 0 {
        return CD_ERR_ARG_NULL;
    }
    let Some(raw) = ctx.read_bytes(param, DLG_PARAM_SIZE as usize) else {
        return CD_ERR_PARAM_INVALID;
    };
    if u64_at(&raw, DLG_OFF_SIZE) != DLG_PARAM_SIZE {
        return CD_ERR_PARAM_INVALID;
    }
    let user = i32_at(&raw, DLG_OFF_USER_ID);
    if user == crate::user_service::INVALID_USER_ID || crate::user_service::user_index(user).is_none() {
        return CD_ERR_PARAM_INVALID;
    }
    // The client id and the scope are non-empty strings.
    let non_empty = |addr: u64, max: usize| ctx.read_cstr_bytes(addr, max).filter(|s| !s.is_empty());
    if non_empty(u64_at(&raw, DLG_OFF_CLIENT_ID), 1).is_none() {
        return CD_ERR_PARAM_INVALID;
    }
    match non_empty(u64_at(&raw, DLG_OFF_SCOPE), DLG_SCOPE_MAX + 1) {
        Some(scope) if scope.len() <= DLG_SCOPE_MAX => {}
        _ => return CD_ERR_PARAM_INVALID,
    }
    if raw[DLG_OFF_RESERVED..].iter().any(|&b| b != 0) || u32::from_le_bytes(raw[DLG_OFF_OPTION..DLG_OFF_OPTION + 4].try_into().unwrap()) > 1 {
        return CD_ERR_PARAM_INVALID;
    }
    OK
}

/// sceNpAuthAuthorizedAppDialogOpen(const SceNpAuthAuthorizedAppDialogParam*)
fn cb_dialog_open(ctx: &mut CpuContext) -> HleResult {
    let param = ctx.rdi();
    let checked = check_dialog_param(ctx, param);
    if checked != OK {
        return checked;
    }
    let mut d = dialog();
    match d.status {
        STATUS_NONE => return CD_ERR_NOT_INITIALIZED,
        STATUS_RUNNING => return CD_ERR_BUSY,
        _ => {}
    }
    log::info!("[NPAUTHDIALOG] open: the user is signed out of PSN, the authorization is canceled");
    *d = DialogState { status: STATUS_RUNNING, result: 0 };
    OK
}

/// sceNpAuthAuthorizedAppDialogGetStatus: the status, never advancing it.
fn cb_dialog_get_status(_ctx: &mut CpuContext) -> HleResult {
    dialog().status as HleResult
}

/// sceNpAuthAuthorizedAppDialogUpdateStatus: a running dialog finishes
/// canceled.
fn cb_dialog_update_status(_ctx: &mut CpuContext) -> HleResult {
    let mut d = dialog();
    if d.status == STATUS_RUNNING {
        *d = DialogState { status: STATUS_FINISHED, result: DLG_RESULT_USER_CANCELED };
    }
    d.status as HleResult
}

/// sceNpAuthAuthorizedAppDialogClose: only a running dialog is closed.
fn cb_dialog_close(_ctx: &mut CpuContext) -> HleResult {
    let mut d = dialog();
    match d.status {
        STATUS_NONE => CD_ERR_NOT_INITIALIZED,
        STATUS_RUNNING => {
            *d = DialogState { status: STATUS_FINISHED, result: DLG_RESULT_USER_CANCELED };
            OK
        }
        _ => CD_ERR_NOT_RUNNING,
    }
}

/// sceNpAuthAuthorizedAppDialogGetResult(SceNpAuthAuthorizedAppDialogResult*):
/// writes the result code (the first field) and returns it.
fn cb_dialog_get_result(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return CD_ERR_ARG_NULL;
    }
    let d = dialog();
    match d.status {
        STATUS_NONE => return CD_ERR_NOT_INITIALIZED,
        STATUS_FINISHED => {}
        _ => return CD_ERR_NOT_FINISHED,
    }
    if !ctx.write_u32(out, d.result as u32) {
        return ERR_MEMORY_FAULT as i64;
    }
    d.result as HleResult
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("6bwFkosYRQg", "sceNpAuthCreateRequest", cb_create_request),
    ("N+mr7GjTvr8", "sceNpAuthCreateAsyncRequest", cb_create_async_request),
    ("H8wG9Bk-nPc", "sceNpAuthDeleteRequest", cb_delete_request),
    ("cE7wIsqXdZ8", "sceNpAuthAbortRequest", cb_abort_request),
    ("PM3IZCw-7m0", "sceNpAuthSetTimeout", cb_set_timeout),
    ("SK-S7daqJSE", "sceNpAuthWaitAsync", cb_wait_async),
    ("gjSyfzSsDcE", "sceNpAuthPollAsync", cb_poll_async),
    ("KI4dHLlTNl0", "sceNpAuthGetAuthorizationCodeV3", cb_get_authorization_code_v3),
    ("RdsFVsgSpZY", "sceNpAuthGetIdTokenV3", cb_get_id_token_v3),
    ("IDX0S5EsEh4", "sceNpAuthGetAuthorizedAppCode", cb_get_authorized_app_code),
    ("KxGkOrQJTqY", "sceNpAuthGetAuthorizationCode", cb_get_authorization_code),
    ("uaB-LoJqHis", "sceNpAuthGetIdToken", cb_get_id_token),
    ("qAUXQ9GdWp8", "sceNpAuthGetAuthorizationCodeA", cb_get_authorization_code_v3),
    ("CocbHVIKPE8", "sceNpAuthGetIdTokenA", cb_get_id_token_v3),
];

const DIALOG_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("9CiV7Shy6qQ", "sceNpAuthAuthorizedAppDialogInitialize", cb_dialog_initialize),
    ("2SlriPR9TMc", "sceNpAuthAuthorizedAppDialogTerminate", cb_dialog_terminate),
    ("cTvSsJRn30w", "sceNpAuthAuthorizedAppDialogOpen", cb_dialog_open),
    ("umZaRUshsKc", "sceNpAuthAuthorizedAppDialogClose", cb_dialog_close),
    ("wQdH6qOwqhE", "sceNpAuthAuthorizedAppDialogGetStatus", cb_dialog_get_status),
    ("L5iGQ06WlTo", "sceNpAuthAuthorizedAppDialogUpdateStatus", cb_dialog_update_status),
    ("bZlJPa2hPWk", "sceNpAuthAuthorizedAppDialogGetResult", cb_dialog_get_result),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
    for &(nid, name, cb) in DIALOG_EXPORTS {
        reg.register_fn(nid, name, DIALOG_LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS.iter().chain(DIALOG_EXPORTS) {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    fn call(ctx: &mut CpuContext, f: HleCallback, args: &[u64]) -> HleResult {
        for (&r, &v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
            ctx.set(r, v);
        }
        f(ctx)
    }

    #[test]
    fn request_lifecycle() {
        let mut st = AuthState::new();
        let sync = st.create(false);
        assert_eq!(sync, REQUEST_ID_OFFSET + 1);
        assert_eq!(st.authorize(sync), (ERR_NP_SIGNED_OUT, true));
        assert_eq!(st.authorize(sync), (ERR_INVALID_ARGUMENT, false), "one authorization per request");
        assert_eq!(st.async_result(sync), Err(ERR_INVALID_ID));
        assert_eq!(st.poll_result(sync), Err(ERR_INVALID_ID));

        let asynchronous = st.create(true);
        assert_eq!(st.poll_result(asynchronous), Ok(None), "running before the call");
        assert_eq!(st.authorize(asynchronous), (0, true));
        assert_eq!(st.poll_result(asynchronous), Ok(None), "the first poll reports running");
        assert_eq!(st.poll_result(asynchronous), Ok(Some(ERR_NP_SIGNED_OUT)));
        assert_eq!(st.poll_result(asynchronous), Ok(Some(ERR_NP_SIGNED_OUT)));
        // Aborting a finished request keeps its result.
        assert_eq!(st.abort(asynchronous), 0);
        assert_eq!(st.async_result(asynchronous), Ok(Some(ERR_NP_SIGNED_OUT)));
        // A second call fails and becomes the reported result.
        assert_eq!(st.authorize(asynchronous), (ERR_INVALID_ARGUMENT, false));
        assert_eq!(st.async_result(asynchronous), Ok(Some(ERR_INVALID_ARGUMENT)));

        let aborted = st.create(true);
        assert_eq!(st.abort(aborted), 0);
        assert_eq!(st.authorize(aborted), (0, true), "an async call reports the abort later");
        assert_eq!(st.async_result(aborted), Ok(Some(ERR_ABORTED)));
        let aborted_sync = st.create(false);
        assert_eq!(st.abort(aborted_sync), 0);
        assert_eq!(st.authorize(aborted_sync), (ERR_ABORTED, true));

        assert_eq!(st.delete(sync), 0);
        assert_eq!(st.delete(sync), ERR_REQUEST_NOT_FOUND);
        assert_eq!(st.abort(sync), ERR_REQUEST_NOT_FOUND);
        assert_eq!(st.authorize(sync).0, ERR_REQUEST_NOT_FOUND);
        assert_eq!(st.create(false), sync, "the lowest free id is reused");
        for bad in [0, -5] {
            assert_eq!(st.delete(bad), ERR_INVALID_ARGUMENT);
            assert_eq!(st.abort(bad), ERR_INVALID_ARGUMENT);
        }
        assert_eq!(st.find(1).err(), Some(ERR_REQUEST_NOT_FOUND));
    }

    #[test]
    fn request_limit() {
        let mut st = AuthState::new();
        for i in 1..=MAX_REQUESTS as i32 {
            assert_eq!(st.create(false), REQUEST_ID_OFFSET + i);
        }
        assert_eq!(st.create(false), ERR_REQUEST_MAX);
        st.delete(REQUEST_ID_OFFSET + 3);
        assert_eq!(st.create(true), REQUEST_ID_OFFSET + 3);
    }

    /// Parameter checks of the authorization calls and what they write.
    #[test]
    fn authorization_calls() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_runtime_state();
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D43_0700_0000;
        vm.map(BASE, 0x2000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (param, code, issuer, flag, text) = (BASE, BASE + 0x100, BASE + 0x800, BASE + 0x900, BASE + 0xA00);
        ctx.write_bytes(text, b"client\0");
        let set_param = |ctx: &CpuContext, words: &[u64]| {
            for (i, &w) in words.iter().enumerate() {
                ctx.write_u64(param + 8 * i as u64, w);
            }
        };
        let user = crate::user_service::PRIMARY_USER_ID as u32 as u64;
        let id = cb_create_request(&mut ctx) as u64;

        // V3 authorization code: size 32, userId, clientId*, scope*.
        let v3 = cb_get_authorization_code_v3 as HleCallback;
        set_param(&ctx, &[32, user, text, text]);
        assert_eq!(call(&mut ctx, v3, &[id, 0, code, issuer]), ERR_INVALID_ARGUMENT as i64);
        assert_eq!(call(&mut ctx, v3, &[id, param, 0, issuer]), ERR_INVALID_ARGUMENT as i64);
        set_param(&ctx, &[24]);
        assert_eq!(call(&mut ctx, v3, &[id, param, code, issuer]), ERR_INVALID_SIZE as i64);
        set_param(&ctx, &[32, u32::MAX as u64, text, text]);
        assert_eq!(call(&mut ctx, v3, &[id, param, code, issuer]), ERR_INVALID_ARGUMENT as i64);
        set_param(&ctx, &[32, user, text, 0]);
        assert_eq!(call(&mut ctx, v3, &[id, param, code, issuer]), ERR_INVALID_ARGUMENT as i64);
        set_param(&ctx, &[32, user, text, text]);
        assert_eq!(call(&mut ctx, v3, &[0, param, code, issuer]), ERR_INVALID_ARGUMENT as i64);
        assert_eq!(call(&mut ctx, v3, &[0x7FFF_0000, param, code, issuer]), ERR_REQUEST_NOT_FOUND as i64);
        ctx.write_bytes(code, &[0xA5; 8]);
        ctx.write_u32(issuer, 0x5A5A_5A5A);
        assert_eq!(call(&mut ctx, v3, &[id, param, code, issuer]), ERR_NP_SIGNED_OUT as i64);
        assert_eq!(ctx.read_u32(issuer), Some(0x5A5A_5A5A), "the issuer id is written only on success");
        assert_eq!(ctx.read_bytes(code, 8).unwrap(), vec![0xA5; 8], "no code on failure");
        assert_eq!(call(&mut ctx, v3, &[id, param, code, issuer]), ERR_INVALID_ARGUMENT as i64);

        // The online-id form fails resolving the user, the request stays unused.
        let fresh = cb_create_request(&mut ctx) as u64;
        set_param(&ctx, &[32, 0, text, text]);
        assert_eq!(call(&mut ctx, cb_get_authorization_code, &[fresh, param, code, issuer]), ERR_INVALID_ARGUMENT as i64);
        set_param(&ctx, &[32, text, text, text]);
        assert_eq!(call(&mut ctx, cb_get_authorization_code, &[fresh, param, code, issuer]), ERR_NP_SIGNED_OUT as i64);
        assert_eq!(call(&mut ctx, cb_get_authorization_code, &[0x7FFF_0000, param, code, issuer]), ERR_NP_SIGNED_OUT as i64);
        assert_eq!(ctx.read_u32(issuer), Some(0x5A5A_5A5A));

        // V3 ID token: size 40, userId, clientId*, clientSecret*, scope*.
        set_param(&ctx, &[40, user, text, 0, text]);
        assert_eq!(call(&mut ctx, cb_get_id_token_v3, &[fresh, param, code]), ERR_INVALID_ARGUMENT as i64);
        set_param(&ctx, &[40, user, text, text, text]);
        assert_eq!(call(&mut ctx, cb_get_id_token_v3, &[fresh, param, code]), ERR_NP_SIGNED_OUT as i64);
        set_param(&ctx, &[40, text, text, text, text]);
        assert_eq!(call(&mut ctx, cb_get_id_token, &[fresh, param, code]), ERR_NP_SIGNED_OUT as i64);

        // Authorized app code: size 40, userId, clientId*, scope*, option 0/1.
        let app = cb_get_authorized_app_code as HleCallback;
        let req = cb_create_request(&mut ctx) as u64;
        set_param(&ctx, &[40, user, text, text, 0]);
        assert_eq!(call(&mut ctx, app, &[req, param, code, issuer, 0]), ERR_INVALID_ARGUMENT as i64);
        set_param(&ctx, &[40, user, text, text, 2]);
        assert_eq!(call(&mut ctx, app, &[req, param, code, issuer, flag]), ERR_INVALID_ARGUMENT as i64);
        set_param(&ctx, &[40, user, text, text, 1]);
        ctx.write_bytes(flag, &[1]);
        assert_eq!(call(&mut ctx, app, &[req, param, code, issuer, flag]), ERR_NP_SIGNED_OUT as i64);
        assert_eq!(ctx.read_bytes(flag, 1).unwrap(), vec![0]);
        assert_eq!(ctx.read_u32(issuer), Some(0x5A5A_5A5A));
    }

    #[test]
    fn async_requests() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_runtime_state();
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D43_0800_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (param, out, text) = (BASE, BASE + 0x100, BASE + 0x200);
        ctx.write_bytes(text, b"scope\0");
        assert_eq!(call(&mut ctx, cb_create_async_request, &[0]), ERR_INVALID_ARGUMENT as i64);
        ctx.write_u64(param, 16);
        assert_eq!(call(&mut ctx, cb_create_async_request, &[param]), ERR_INVALID_SIZE as i64);
        ctx.write_u64(param, 24);
        let id = call(&mut ctx, cb_create_async_request, &[param]) as u64;
        assert_eq!(id, REQUEST_ID_OFFSET as u64 + 1);
        let sync = cb_create_request(&mut ctx) as u64;

        assert_eq!(call(&mut ctx, cb_poll_async, &[id, 0]), ERR_INVALID_ARGUMENT as i64);
        assert_eq!(call(&mut ctx, cb_poll_async, &[sync, out]), ERR_INVALID_ID as i64);
        assert_eq!(call(&mut ctx, cb_wait_async, &[sync, out]), ERR_INVALID_ID as i64);
        assert_eq!(call(&mut ctx, cb_poll_async, &[id, out]), POLL_ASYNC_RUNNING);

        for (i, w) in [32u64, crate::user_service::PRIMARY_USER_ID as u32 as u64, text, text].iter().enumerate() {
            ctx.write_u64(param + 8 * i as u64, *w);
        }
        assert_eq!(call(&mut ctx, cb_get_authorization_code_v3, &[id, param, out + 0x10, 0]), OK);
        assert_eq!(call(&mut ctx, cb_wait_async, &[id, out]), OK);
        assert_eq!(ctx.read_u32(out), Some(ERR_NP_SIGNED_OUT as u32));
        assert_eq!(call(&mut ctx, cb_poll_async, &[id, out]), POLL_ASYNC_RUNNING);
        ctx.write_u32(out, 0);
        assert_eq!(call(&mut ctx, cb_poll_async, &[id, out]), OK);
        assert_eq!(ctx.read_u32(out), Some(ERR_NP_SIGNED_OUT as u32));

        // Wait blocks until another thread makes the call.
        let waiting = cb_create_request_async_for_test();
        let waiter = std::thread::spawn(move || {
            let vm = ps5vm::VirtualMemory::new();
            vm.map(BASE + 0x10_0000, 0x1000, ps5vm::MemProtect::RW).unwrap();
            let mut ctx = CpuContext::new(&vm);
            let r = call(&mut ctx, cb_wait_async, &[waiting as u64, BASE + 0x10_0000]);
            (r, ctx.read_u32(BASE + 0x10_0000))
        });
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(call(&mut ctx, cb_abort_request, &[waiting as u64]), OK);
        assert_eq!(call(&mut ctx, cb_get_authorization_code_v3, &[waiting as u64, param, out + 0x10, 0]), OK);
        assert_eq!(waiter.join().unwrap(), (OK, Some(ERR_ABORTED as u32)));

        // Deleting the request ends a wait.
        let deleted = cb_create_request_async_for_test();
        let waiter = std::thread::spawn(move || {
            let vm = ps5vm::VirtualMemory::new();
            vm.map(BASE + 0x20_0000, 0x1000, ps5vm::MemProtect::RW).unwrap();
            let mut ctx = CpuContext::new(&vm);
            call(&mut ctx, cb_wait_async, &[deleted as u64, BASE + 0x20_0000])
        });
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(call(&mut ctx, cb_delete_request, &[deleted as u64]), OK);
        assert_eq!(waiter.join().unwrap(), ERR_REQUEST_NOT_FOUND as i64);
        reset_runtime_state();
    }

    fn cb_create_request_async_for_test() -> i32 {
        state().create(true)
    }

    #[test]
    fn set_timeout_limits() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_runtime_state();
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        let id = cb_create_request(&mut ctx) as u64;
        let t = |ctx: &mut CpuContext, args: [u64; 6]| call(ctx, cb_set_timeout, &args);
        assert_eq!(t(&mut ctx, [id, 0, 0, 0, 0, 0]), ERR_INVALID_ARGUMENT as i64);
        assert_eq!(t(&mut ctx, [id, u32::MAX as u64, 0, 0, 0, 0]), ERR_INVALID_ARGUMENT as i64);
        assert_eq!(t(&mut ctx, [id, 0, 999_999, 0, 0, 0]), ERR_INVALID_ARGUMENT as i64);
        assert_eq!(t(&mut ctx, [id, 0, 0, 0, 9_999_999, 0]), ERR_INVALID_ARGUMENT as i64);
        assert_eq!(t(&mut ctx, [id, 0, 1_000_000, 0, 0, 0]), OK);
        assert_eq!(t(&mut ctx, [id, 2, 0, 10_000_000, 10_000_000, 30_000_000]), OK);
        assert_eq!(t(&mut ctx, [id, 2, 0, 0, 0, 0]), OK);
        assert_eq!(t(&mut ctx, [0x7FFF_0000, 2, 0, 0, 0, 0]), ERR_REQUEST_NOT_FOUND as i64);
        assert_eq!(t(&mut ctx, [0, 2, 0, 0, 0, 0]), ERR_INVALID_ARGUMENT as i64);
        reset_runtime_state();
    }

    /// Serializes the tests that use the shared request and dialog state.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn authorized_app_dialog() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_runtime_state();
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D43_0900_0000;
        vm.map(BASE, 0x2000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (param, result, client, scope) = (BASE, BASE + 0x100, BASE + 0x200, BASE + 0x400);
        ctx.write_bytes(client, b"0123456789abcdef\0");
        ctx.write_bytes(scope, b"psn:s2s\0");
        ctx.write_u64(param + 0x30, DLG_PARAM_SIZE);
        ctx.write_u32(param + 0x38, crate::user_service::PRIMARY_USER_ID as u32);
        ctx.write_u64(param + 0x40, client);
        ctx.write_u64(param + 0x48, scope);
        let st = |ctx: &mut CpuContext| call(ctx, cb_dialog_get_status, &[]);

        assert_eq!(st(&mut ctx), STATUS_NONE as i64);
        assert_eq!(call(&mut ctx, cb_dialog_update_status, &[]), STATUS_NONE as i64);
        assert_eq!(call(&mut ctx, cb_dialog_terminate, &[]), CD_ERR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, cb_dialog_close, &[]), CD_ERR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, cb_dialog_get_result, &[0]), CD_ERR_ARG_NULL, "the pointer is checked first");
        assert_eq!(call(&mut ctx, cb_dialog_get_result, &[result]), CD_ERR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, cb_dialog_open, &[0]), CD_ERR_ARG_NULL);
        assert_eq!(call(&mut ctx, cb_dialog_open, &[param]), CD_ERR_NOT_INITIALIZED, "parameters are checked first");

        assert_eq!(call(&mut ctx, cb_dialog_initialize, &[]), OK);
        assert_eq!(call(&mut ctx, cb_dialog_initialize, &[]), CD_ERR_ALREADY_INITIALIZED);
        assert_eq!(st(&mut ctx), STATUS_INITIALIZED as i64);
        assert_eq!(call(&mut ctx, cb_dialog_close, &[]), CD_ERR_NOT_RUNNING);
        assert_eq!(call(&mut ctx, cb_dialog_get_result, &[result]), CD_ERR_NOT_FINISHED);

        // Invalid parameter blocks.
        let invalid = |ctx: &mut CpuContext, off: u64, bad: u64, good: u64| {
            ctx.write_u64(param + off, bad);
            let r = call(ctx, cb_dialog_open, &[param]);
            ctx.write_u64(param + off, good);
            r
        };
        let user = crate::user_service::PRIMARY_USER_ID as u32 as u64;
        assert_eq!(invalid(&mut ctx, 0x30, 0x70, DLG_PARAM_SIZE), CD_ERR_PARAM_INVALID);
        assert_eq!(invalid(&mut ctx, 0x38, u32::MAX as u64, user), CD_ERR_PARAM_INVALID);
        assert_eq!(invalid(&mut ctx, 0x38, user + 3, user), CD_ERR_PARAM_INVALID, "a user who is not logged in");
        assert_eq!(invalid(&mut ctx, 0x40, 0, client), CD_ERR_PARAM_INVALID);
        assert_eq!(invalid(&mut ctx, 0x40, BASE + 0x1FFF, client), CD_ERR_PARAM_INVALID, "empty client id");
        assert_eq!(invalid(&mut ctx, 0x48, 0, scope), CD_ERR_PARAM_INVALID);
        assert_eq!(invalid(&mut ctx, 0x50, 2, 0), CD_ERR_PARAM_INVALID);
        assert_eq!(invalid(&mut ctx, 0x70, 1, 0), CD_ERR_PARAM_INVALID);
        ctx.write_bytes(BASE + 0x800, &[b'a'; 0x401]);
        assert_eq!(invalid(&mut ctx, 0x48, BASE + 0x800, scope), CD_ERR_PARAM_INVALID, "scope longer than 1024");
        assert_eq!(invalid(&mut ctx, 0x48, BASE + 0x801, scope), OK, "a 1024-character scope is accepted");
        assert_eq!(call(&mut ctx, cb_dialog_terminate, &[]), OK, "terminate ends a running dialog");
        assert_eq!(call(&mut ctx, cb_dialog_initialize, &[]), OK);

        // Open -> RUNNING; GetStatus does not advance; UpdateStatus finishes.
        ctx.write_u64(param + 0x50, 1);
        assert_eq!(call(&mut ctx, cb_dialog_open, &[param]), OK);
        assert_eq!(call(&mut ctx, cb_dialog_open, &[param]), CD_ERR_BUSY);
        assert_eq!(st(&mut ctx), STATUS_RUNNING as i64);
        assert_eq!(st(&mut ctx), STATUS_RUNNING as i64);
        assert_eq!(call(&mut ctx, cb_dialog_get_result, &[result]), CD_ERR_NOT_FINISHED);
        assert_eq!(call(&mut ctx, cb_dialog_update_status, &[]), STATUS_FINISHED as i64);
        assert_eq!(st(&mut ctx), STATUS_FINISHED as i64);
        assert_eq!(call(&mut ctx, cb_dialog_close, &[]), CD_ERR_NOT_RUNNING);
        ctx.write_bytes(result, &[0xEE; 8]);
        assert_eq!(call(&mut ctx, cb_dialog_get_result, &[result]), DLG_RESULT_USER_CANCELED as i64);
        assert_eq!(ctx.read_u32(result), Some(DLG_RESULT_USER_CANCELED as u32));
        assert_eq!(ctx.read_u32(result + 4), Some(0xEEEE_EEEE), "only the result code is written");

        // Reopen after it finished, then close it.
        assert_eq!(call(&mut ctx, cb_dialog_open, &[param]), OK);
        assert_eq!(call(&mut ctx, cb_dialog_close, &[]), OK);
        assert_eq!(st(&mut ctx), STATUS_FINISHED as i64);
        assert_eq!(call(&mut ctx, cb_dialog_get_result, &[result]), DLG_RESULT_USER_CANCELED as i64);
        assert_eq!(call(&mut ctx, cb_dialog_terminate, &[]), OK);
        assert_eq!(st(&mut ctx), STATUS_NONE as i64);
        assert_eq!(call(&mut ctx, cb_dialog_terminate, &[]), CD_ERR_NOT_INITIALIZED);
    }
}
