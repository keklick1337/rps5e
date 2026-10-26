// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceNpAuth HLE: authorization requests of a signed-out user.
//!
//! Up to 16 requests are tracked (ids 0x10000001 + slot). Every
//! authorization completes with SIGNED_OUT: synchronous calls return it,
//! asynchronous ones report it through sceNpAuthPollAsync / WaitAsync.

use std::sync::Mutex;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceNpAuth";
const OK: HleResult = 0;

const ERR_INVALID_ARGUMENT: i32 = 0x8055_0301u32 as i32;
const ERR_ABORTED: i32 = 0x8055_0304u32 as i32;
const ERR_REQUEST_MAX: i32 = 0x8055_0305u32 as i32;
const ERR_REQUEST_NOT_FOUND: i32 = 0x8055_0306u32 as i32;
const ERR_INVALID_ID: i32 = 0x8055_0307u32 as i32;
const ERR_NP_SIGNED_OUT: i32 = 0x8055_0006u32 as i32;
const ERR_MEMORY_FAULT: i32 = 0x8002_000Eu32 as i32;

const MAX_REQUESTS: usize = 16;
const REQUEST_ID_OFFSET: i32 = 0x1000_0000;
/// SceNpAuthorizationCode (136 bytes) and SceNpIdToken (4104 bytes).
const AUTH_CODE_SIZE: usize = 136;
const ID_TOKEN_SIZE: usize = 4104;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RequestState {
    None,
    Ready,
    Aborted,
    Complete,
}

#[derive(Clone, Copy, Debug)]
struct AuthRequest {
    state: RequestState,
    is_async: bool,
    result: i32,
    /// sceNpAuthPollAsync already reported this finished request running once.
    polled: bool,
}

struct AuthState {
    requests: Vec<AuthRequest>,
    active: usize,
}

impl AuthState {
    const fn new() -> Self {
        Self { requests: Vec::new(), active: 0 }
    }

    fn create(&mut self, is_async: bool) -> i32 {
        if self.active >= MAX_REQUESTS {
            return ERR_REQUEST_MAX;
        }
        let request = AuthRequest { state: RequestState::Ready, is_async, result: 0, polled: false };
        let index = match self.requests.iter().position(|r| r.state == RequestState::None) {
            Some(i) => {
                self.requests[i] = request;
                i
            }
            None => {
                self.requests.push(request);
                self.requests.len() - 1
            }
        };
        self.active += 1;
        index as i32 + REQUEST_ID_OFFSET + 1
    }

    fn find(&mut self, id: i32) -> Option<&mut AuthRequest> {
        let index = id.wrapping_sub(REQUEST_ID_OFFSET + 1);
        if index < 0 || self.active == 0 {
            return None;
        }
        self.requests.get_mut(index as usize).filter(|r| r.state != RequestState::None)
    }

    fn delete(&mut self, id: i32) -> i32 {
        match self.find(id) {
            Some(r) => {
                *r = AuthRequest { state: RequestState::None, is_async: false, result: 0, polled: false };
                self.active -= 1;
                0
            }
            None => ERR_REQUEST_NOT_FOUND,
        }
    }

    fn abort(&mut self, id: i32) -> i32 {
        match self.find(id) {
            Some(r) => {
                if r.state != RequestState::Complete {
                    r.state = RequestState::Aborted;
                }
                0
            }
            None => ERR_REQUEST_NOT_FOUND,
        }
    }

    /// Result of a finished asynchronous request.
    fn async_result(&mut self, id: i32) -> Result<i32, i32> {
        let r = self.find(id).ok_or(ERR_REQUEST_NOT_FOUND)?;
        if !r.is_async || r.state == RequestState::Ready {
            return Err(ERR_INVALID_ID);
        }
        Ok(r.result)
    }

    /// sceNpAuthPollAsync: the first poll of a finished request reports it
    /// still running (1); later polls give its result.
    fn poll_result(&mut self, id: i32) -> Result<Option<i32>, i32> {
        self.async_result(id)?;
        let r = self.find(id).ok_or(ERR_REQUEST_NOT_FOUND)?;
        if !r.polled {
            r.polled = true;
            return Ok(None);
        }
        Ok(Some(r.result))
    }

    /// An authorization on request `id`: the user is signed out.
    fn authorize(&mut self, id: i32) -> i32 {
        let Some(r) = self.find(id) else {
            return ERR_REQUEST_NOT_FOUND;
        };
        match r.state {
            RequestState::Aborted => {
                r.result = ERR_ABORTED;
                ERR_ABORTED
            }
            RequestState::Complete => {
                r.result = ERR_INVALID_ARGUMENT;
                ERR_INVALID_ARGUMENT
            }
            _ => {
                r.state = RequestState::Complete;
                r.result = ERR_NP_SIGNED_OUT;
                if r.is_async { 0 } else { ERR_NP_SIGNED_OUT }
            }
        }
    }
}

static STATE: Mutex<AuthState> = Mutex::new(AuthState::new());

fn state() -> std::sync::MutexGuard<'static, AuthState> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Drop all requests and restart id allocation (emulator runtime reset).
pub fn reset_runtime_state() {
    *state() = AuthState::new();
}

fn request_id(ctx: &CpuContext) -> i32 {
    ctx.rdi() as i32
}

fn cb_create_request(_ctx: &mut CpuContext) -> HleResult {
    state().create(false) as i64
}

/// sceNpAuthCreateAsyncRequest(const SceNpAuthCreateAsyncRequestParameter*)
fn cb_create_async_request(_ctx: &mut CpuContext) -> HleResult {
    state().create(true) as i64
}

fn cb_delete_request(ctx: &mut CpuContext) -> HleResult {
    state().delete(request_id(ctx)) as i64
}

fn cb_abort_request(ctx: &mut CpuContext) -> HleResult {
    state().abort(request_id(ctx)) as i64
}

/// sceNpAuthSetTimeout(reqId, resolveRetry, resolveTimeout, connectTimeout,
/// sendTimeout, recvTimeout): nothing is ever sent, so only the request is
/// checked.
fn cb_set_timeout(ctx: &mut CpuContext) -> HleResult {
    if state().find(request_id(ctx)).is_some() { OK } else { ERR_REQUEST_NOT_FOUND as i64 }
}

/// SCE_NP_POLL_ASYNC_RET_RUNNING.
const POLL_ASYNC_RUNNING: HleResult = 1;

/// sceNpAuthPollAsync(reqId, int* result): RUNNING once, then the result.
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

/// sceNpAuthWaitAsync(reqId, int* result)
fn cb_wait_or_poll_async(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    match state().async_result(request_id(ctx)) {
        Ok(result) => {
            if ctx.write_u32(out, result as u32) {
                OK
            } else {
                ERR_MEMORY_FAULT as i64
            }
        }
        Err(e) => e as i64,
    }
}

/// sceNpAuthGetAuthorizationCodeV3(reqId, param, SceNpAuthorizationCode*,
/// int* issuerId)
fn cb_get_authorization_code_v3(ctx: &mut CpuContext) -> HleResult {
    authorization_code(ctx, ctx.rcx())
}

/// sceNpAuthGetAuthorizationCode(reqId, param, SceNpAuthorizationCode*):
/// the V3 call without an issuer id.
fn cb_get_authorization_code(ctx: &mut CpuContext) -> HleResult {
    authorization_code(ctx, 0)
}

fn authorization_code(ctx: &CpuContext, issuer: u64) -> HleResult {
    let code = ctx.rdx();
    if code != 0 && !ctx.write_bytes(code, &[0u8; AUTH_CODE_SIZE]) {
        return ERR_MEMORY_FAULT as i64;
    }
    if issuer != 0 && !ctx.write_u32(issuer, 0) {
        return ERR_MEMORY_FAULT as i64;
    }
    state().authorize(request_id(ctx)) as i64
}

/// sceNpAuthGetIdTokenV3(reqId, param, SceNpIdToken*)
fn cb_get_id_token_v3(ctx: &mut CpuContext) -> HleResult {
    let token = ctx.rdx();
    if token != 0 && !ctx.write_bytes(token, &vec![0u8; ID_TOKEN_SIZE]) {
        return ERR_MEMORY_FAULT as i64;
    }
    state().authorize(request_id(ctx)) as i64
}

/// sceNpAuthGetAuthorizedAppCode(reqId, ...): same outcome as the other
/// authorizations.
fn cb_get_authorized_app_code(ctx: &mut CpuContext) -> HleResult {
    state().authorize(request_id(ctx)) as i64
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("6bwFkosYRQg", "sceNpAuthCreateRequest", cb_create_request),
    ("N+mr7GjTvr8", "sceNpAuthCreateAsyncRequest", cb_create_async_request),
    ("H8wG9Bk-nPc", "sceNpAuthDeleteRequest", cb_delete_request),
    ("cE7wIsqXdZ8", "sceNpAuthAbortRequest", cb_abort_request),
    ("PM3IZCw-7m0", "sceNpAuthSetTimeout", cb_set_timeout),
    ("SK-S7daqJSE", "sceNpAuthWaitAsync", cb_wait_or_poll_async),
    ("gjSyfzSsDcE", "sceNpAuthPollAsync", cb_poll_async),
    ("KI4dHLlTNl0", "sceNpAuthGetAuthorizationCodeV3", cb_get_authorization_code_v3),
    ("RdsFVsgSpZY", "sceNpAuthGetIdTokenV3", cb_get_id_token_v3),
    ("IDX0S5EsEh4", "sceNpAuthGetAuthorizedAppCode", cb_get_authorized_app_code),
    ("KxGkOrQJTqY", "sceNpAuthGetAuthorizationCode", cb_get_authorization_code),
    ("uaB-LoJqHis", "sceNpAuthGetIdToken", cb_get_id_token_v3),
    ("qAUXQ9GdWp8", "sceNpAuthGetAuthorizationCodeA", cb_get_authorization_code_v3),
    ("CocbHVIKPE8", "sceNpAuthGetIdTokenA", cb_get_id_token_v3),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
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
    fn request_lifecycle() {
        let mut st = AuthState::new();
        let sync = st.create(false);
        assert_eq!(sync, REQUEST_ID_OFFSET + 1);
        assert_eq!(st.authorize(sync), ERR_NP_SIGNED_OUT);
        assert_eq!(st.authorize(sync), ERR_INVALID_ARGUMENT);
        assert_eq!(st.async_result(sync), Err(ERR_INVALID_ID));

        let asynchronous = st.create(true);
        assert_eq!(st.async_result(asynchronous), Err(ERR_INVALID_ID));
        assert_eq!(st.poll_result(asynchronous), Err(ERR_INVALID_ID), "nothing to poll before the call");
        assert_eq!(st.authorize(asynchronous), 0);
        assert_eq!(st.poll_result(asynchronous), Ok(None), "the first poll reports running");
        assert_eq!(st.poll_result(asynchronous), Ok(Some(ERR_NP_SIGNED_OUT)));
        assert_eq!(st.poll_result(asynchronous), Ok(Some(ERR_NP_SIGNED_OUT)));
        assert_eq!(st.async_result(asynchronous), Ok(ERR_NP_SIGNED_OUT));
        // Completed requests are not aborted.
        assert_eq!(st.abort(asynchronous), 0);
        assert_eq!(st.async_result(asynchronous), Ok(ERR_NP_SIGNED_OUT));

        let aborted = st.create(true);
        assert_eq!(st.abort(aborted), 0);
        assert_eq!(st.authorize(aborted), ERR_ABORTED);
        assert_eq!(st.async_result(aborted), Ok(ERR_ABORTED));

        assert_eq!(st.delete(sync), 0);
        assert_eq!(st.delete(sync), ERR_REQUEST_NOT_FOUND);
        assert_eq!(st.abort(sync), ERR_REQUEST_NOT_FOUND);
        assert_eq!(st.authorize(sync), ERR_REQUEST_NOT_FOUND);
        assert_eq!(st.create(false), sync);
        assert!(st.find(1).is_none());
    }

    /// The three-argument sceNpAuthGetAuthorizationCode has no issuer id:
    /// whatever is in the fourth register is not written.
    #[test]
    fn authorization_code_forms() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D43_0700_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        use crate::context::CpuRegister::*;
        for (cb, issuer_written) in [(cb_get_authorization_code as HleCallback, false), (cb_get_authorization_code_v3, true)] {
            ctx.write_u32(BASE + 0x800, 0x5A5A_5A5A);
            for (r, v) in [(Rdi, 0x7FFF_0000u64), (Rsi, BASE), (Rdx, BASE + 0x100), (Rcx, BASE + 0x800)] {
                ctx.set(r, v);
            }
            assert_eq!(cb(&mut ctx), ERR_REQUEST_NOT_FOUND as i64);
            assert_eq!(ctx.read_u32(BASE + 0x800) == Some(0), issuer_written);
        }
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
}
