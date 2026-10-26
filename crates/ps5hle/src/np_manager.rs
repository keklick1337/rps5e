// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceNpManager HLE: the local user is signed out of PSN.
//!
//! Account queries clear their output and report SIGNED_OUT; server-side
//! checks run on tracked requests and complete with SIGNED_OUT (async
//! requests report it through sceNpPollAsync). State callbacks registered
//! with sceNpRegisterStateCallbackA receive the signed-out state once,
//! from sceNpCheckCallback.

use std::sync::Mutex;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceNpManager";
const OK: HleResult = 0;

const fn err(v: u32) -> i32 {
    v as i32
}

const ERR_INVALID_ARGUMENT: i32 = err(0x8055_0003);
const ERR_SIGNED_OUT: i32 = err(0x8055_0006);
const ERR_USER_NOT_FOUND: i32 = err(0x8055_0007);
const ERR_CALLBACK_ALREADY_REGISTERED: i32 = err(0x8055_0008);
const ERR_CALLBACK_NOT_REGISTERED: i32 = err(0x8055_0009);
const ERR_INVALID_SIZE: i32 = err(0x8055_0011);
const ERR_CALLBACK_MAX: i32 = err(0x8055_001D);
const ERR_ABORTED: i32 = err(0x8055_0012);
const ERR_REQUEST_MAX: i32 = err(0x8055_0013);
const ERR_REQUEST_NOT_FOUND: i32 = err(0x8055_0014);
const ERR_INVALID_ID: i32 = err(0x8055_0015);
const ERR_MEMORY_FAULT: i32 = err(0x8002_000E);

const REQUEST_MAX: usize = 128;
/// SCE_NP_STATE_SIGNED_OUT
const NP_STATE_SIGNED_OUT: u32 = 1;
/// SCE_NP_REACHABILITY_STATE_UNAVAILABLE
const NP_REACHABILITY_UNAVAILABLE: u32 = 0;

/// SceNpTitleId (16 bytes) / SceNpTitleSecret (128 bytes).
const NP_TITLE_ID_SIZE: usize = 16;
const NP_TITLE_SECRET_SIZE: usize = 128;
/// SceNpContentRestriction: size, defaultAgeRestriction, pad[3],
/// ageRestrictionCount, const SceNpAgeRestriction* ageRestriction.
const CONTENT_RESTRICTION_SIZE: u64 = 24;
/// SceNpOnlineId: 16-char handle, terminator, 3 padding bytes.
const ONLINE_ID_SIZE: usize = 20;
/// SceNpId: online id, opt[8], reserved[8].
const NP_ID_SIZE: usize = 36;
/// SceNpCountryCode: 2 chars, terminator, padding.
const COUNTRY_CODE_SIZE: usize = 4;
/// SceNpCreateAsyncRequestParameter: size, cpuAffinityMask, threadPriority, pad.
const ASYNC_PARAM_SIZE: u64 = 24;
/// SceNpCheckPremiumParameter: size, userId, pad, features, reserved[32].
const CHECK_PREMIUM_PARAM_SIZE: u64 = 0x38;
/// SceNpCheckPremiumResult: bool authorized, reserved[32].
const CHECK_PREMIUM_RESULT_SIZE: usize = 33;
/// SceNpPremiumFeatureNotifyParameter: size, userId, pad, features, ...
const NOTIFY_PREMIUM_PARAM_MIN_SIZE: u64 = 0x18;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RequestState {
    Free,
    Ready,
    Aborted,
    Complete,
}

#[derive(Clone, Copy, Debug)]
struct Request {
    state: RequestState,
    is_async: bool,
    result: i32,
}

#[derive(Clone, Copy, Debug)]
struct StateCallback {
    callback: u64,
    userdata: u64,
    pending: bool,
}

/// Slots of sceNpRegisterStateCallbackA; the id is the slot index + 1.
const STATE_CALLBACK_SLOTS: usize = 8;

struct NpState {
    requests: Vec<Request>,
    state_callbacks: [Option<StateCallback>; STATE_CALLBACK_SLOTS],
    /// The single callback of the non-A sceNpRegisterStateCallback.
    legacy_state_callback: Option<StateCallback>,
}

impl NpState {
    const fn new() -> Self {
        NpState { requests: Vec::new(), state_callbacks: [None; STATE_CALLBACK_SLOTS], legacy_state_callback: None }
    }

    /// sceNpRegisterStateCallbackA: the id (1..8) or the error.
    fn register_state_callback(&mut self, callback: u64, userdata: u64) -> Result<i32, i32> {
        if callback == 0 {
            return Err(ERR_INVALID_ARGUMENT);
        }
        if self.state_callbacks.iter().flatten().any(|c| c.callback == callback) {
            return Err(ERR_CALLBACK_ALREADY_REGISTERED);
        }
        let slot = self.state_callbacks.iter().position(Option::is_none).ok_or(ERR_CALLBACK_MAX)?;
        self.state_callbacks[slot] = Some(StateCallback { callback, userdata, pending: true });
        Ok(slot as i32 + 1)
    }

    /// sceNpUnregisterStateCallbackA(id): an id outside 1..8 (compared
    /// unsigned, so 0 and negative ids too) is INVALID_ID, as the module's
    /// code does.
    fn unregister_state_callback(&mut self, id: i32) -> Result<(), i32> {
        if (id as u32).wrapping_sub(1) >= STATE_CALLBACK_SLOTS as u32 {
            return Err(ERR_INVALID_ID);
        }
        self.state_callbacks[id as usize - 1].take().map(|_| ()).ok_or(ERR_CALLBACK_NOT_REGISTERED)
    }

    fn register_legacy_state_callback(&mut self, callback: u64, userdata: u64) -> Result<(), i32> {
        if callback == 0 {
            return Err(ERR_INVALID_ARGUMENT);
        }
        if self.legacy_state_callback.is_some() {
            return Err(ERR_CALLBACK_ALREADY_REGISTERED);
        }
        self.legacy_state_callback = Some(StateCallback { callback, userdata, pending: true });
        Ok(())
    }

    fn unregister_legacy_state_callback(&mut self) -> Result<(), i32> {
        self.legacy_state_callback.take().map(|_| ()).ok_or(ERR_CALLBACK_NOT_REGISTERED)
    }

    /// The callbacks not told the state yet, the legacy one first, then the
    /// A slots in order; their pending flags are cleared.
    fn take_pending(&mut self) -> (Option<StateCallback>, Vec<StateCallback>) {
        let legacy = self.legacy_state_callback.as_mut().filter(|c| c.pending).map(|c| {
            c.pending = false;
            *c
        });
        let a = self
            .state_callbacks
            .iter_mut()
            .flatten()
            .filter(|c| c.pending)
            .map(|c| {
                c.pending = false;
                *c
            })
            .collect();
        (legacy, a)
    }
}

static STATE: Mutex<NpState> = Mutex::new(NpState::new());

/// The NP reachability state callback (callback, userdata). It is never
/// called: reachability stays UNAVAILABLE.
static REACHABILITY_CALLBACK: Mutex<Option<(u64, u64)>> = Mutex::new(None);

fn reachability_callback() -> std::sync::MutexGuard<'static, Option<(u64, u64)>> {
    REACHABILITY_CALLBACK.lock().unwrap_or_else(|p| p.into_inner())
}

/// sceNpRegisterNpReachabilityStateCallback(callback, userdata): one
/// callback at a time.
fn cb_register_np_reachability_state_callback(ctx: &mut CpuContext) -> HleResult {
    let (callback, userdata) = (ctx.rdi(), ctx.rsi());
    if callback == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    let mut slot = reachability_callback();
    if slot.is_some() {
        return ERR_CALLBACK_ALREADY_REGISTERED as i64;
    }
    *slot = Some((callback, userdata));
    OK
}

/// sceNpUnregisterNpReachabilityStateCallback()
fn cb_unregister_np_reachability_state_callback(_ctx: &mut CpuContext) -> HleResult {
    match reachability_callback().take() {
        Some(_) => OK,
        None => ERR_CALLBACK_NOT_REGISTERED as i64,
    }
}

fn state() -> std::sync::MutexGuard<'static, NpState> {
    STATE.lock().unwrap_or_else(|p| p.into_inner())
}

impl NpState {
    /// Request ids are slot index + 1.
    fn create_request(&mut self, is_async: bool) -> i32 {
        let request = Request { state: RequestState::Ready, is_async, result: 0 };
        if let Some(i) = self.requests.iter().position(|r| r.state == RequestState::Free) {
            self.requests[i] = request;
            return i as i32 + 1;
        }
        if self.requests.len() >= REQUEST_MAX {
            return ERR_REQUEST_MAX;
        }
        self.requests.push(request);
        self.requests.len() as i32
    }

    fn request(&mut self, id: i32) -> Option<&mut Request> {
        if id <= 0 {
            return None;
        }
        self.requests.get_mut(id as usize - 1).filter(|r| r.state != RequestState::Free)
    }
}

/// Complete a server-side check on a signed-out console: synchronous
/// requests return SIGNED_OUT, asynchronous ones report it when polled.
fn complete_signed_out(r: &mut Request) -> i32 {
    match r.state {
        RequestState::Complete => {
            r.result = ERR_INVALID_ARGUMENT;
            ERR_INVALID_ARGUMENT
        }
        RequestState::Aborted => {
            r.result = ERR_ABORTED;
            ERR_ABORTED
        }
        _ => {
            r.state = RequestState::Complete;
            r.result = ERR_SIGNED_OUT;
            if r.is_async { 0 } else { ERR_SIGNED_OUT }
        }
    }
}

/// Run `f` on request `id`, or REQUEST_NOT_FOUND.
fn with_request(id: i32, f: impl FnOnce(&mut Request) -> i32) -> HleResult {
    match state().request(id) {
        Some(r) => f(r) as i64,
        None => ERR_REQUEST_NOT_FOUND as i64,
    }
}

fn clear(ctx: &CpuContext, addr: u64, len: usize) -> bool {
    ctx.write_bytes(addr, &vec![0u8; len])
}

fn cb_ok(_ctx: &mut CpuContext) -> HleResult {
    OK
}

/// sceNpRegisterPremiumEventCallback(callback, userdata): premium events
/// never fire offline; a NULL callback is refused.
fn cb_register_premium_event_callback(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 { ERR_INVALID_ARGUMENT as i64 } else { OK }
}

// libSceNpUtility bandwidth test: a test cannot start while signed out, so
// no test context exists to query, abort or shut down.

/// sceNpBandwidthTestInitStart{Download,Upload}(param*).
fn cb_bandwidth_test_init_start(_ctx: &mut CpuContext) -> HleResult {
    ERR_SIGNED_OUT as i64
}

/// sceNpBandwidthTestGetStatus(contextId, int* status): the status is
/// cleared, the context is unknown.
fn cb_bandwidth_test_get_status(ctx: &mut CpuContext) -> HleResult {
    let status = ctx.rsi();
    if status != 0 {
        ctx.write_u32(status, 0);
    }
    ERR_INVALID_ARGUMENT as i64
}

/// sceNpBandwidthTestAbort(contextId) / Shutdown(contextId, result*).
fn cb_bandwidth_test_no_context(_ctx: &mut CpuContext) -> HleResult {
    ERR_INVALID_ARGUMENT as i64
}

const UTILITY_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("hqzi1IHdQQQ", "sceNpBandwidthTestInitStartDownload", cb_bandwidth_test_init_start),
    ("mA0zsbqm+kA", "sceNpBandwidthTestInitStartUpload", cb_bandwidth_test_init_start),
    ("BYIZGKm6bO4", "sceNpBandwidthTestGetStatus", cb_bandwidth_test_get_status),
    ("kvdMF48mB3Y", "sceNpBandwidthTestAbort", cb_bandwidth_test_no_context),
    ("pLr1fEQS1z8", "sceNpBandwidthTestShutdown", cb_bandwidth_test_no_context),
];


/// sceNpCheckCallback(): tells newly registered state callbacks that the
/// user is signed out.
fn cb_check_callback(_ctx: &mut CpuContext) -> HleResult {
    let user = crate::user_service::PRIMARY_USER_ID as u32 as u64;
    // Called without the lock: a callback may unregister itself.
    let (legacy, a) = state().take_pending();
    if let Some(c) = legacy {
        // (userId, state, const SceNpId* npId, userdata)
        ps5cpu::exec::call_guest_nested(c.callback, [user, NP_STATE_SIGNED_OUT as u64, 0, c.userdata, 0, 0]);
    }
    for c in a {
        ps5cpu::exec::call_guest_nested(c.callback, [user, NP_STATE_SIGNED_OUT as u64, c.userdata, 0, 0, 0]);
    }
    OK
}

/// sceNpSetNpTitleId(const SceNpTitleId*, const SceNpTitleSecret*)
fn cb_set_np_title_id(ctx: &mut CpuContext) -> HleResult {
    let (title_id_addr, secret_addr) = (ctx.rdi(), ctx.rsi());
    if title_id_addr == 0 || secret_addr == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    let (Some(title_id), Some(_)) = (ctx.read_bytes(title_id_addr, NP_TITLE_ID_SIZE), ctx.read_bytes(secret_addr, NP_TITLE_SECRET_SIZE)) else {
        return ERR_MEMORY_FAULT as i64;
    };
    let len = title_id.iter().take(12).take_while(|&&b| b != 0).count();
    log::debug!("np: title id '{}'", String::from_utf8_lossy(&title_id[..len]));
    OK
}

/// sceNpSetContentRestriction(const SceNpContentRestriction*)
fn cb_set_content_restriction(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    match ctx.read_u64(addr) {
        None => ERR_MEMORY_FAULT as i64,
        Some(CONTENT_RESTRICTION_SIZE) => OK,
        Some(_) => ERR_INVALID_SIZE as i64,
    }
}

/// sceNpRegisterStateCallbackA(callback, userdata) -> callback id 1..8.
fn cb_register_state_callback_a(ctx: &mut CpuContext) -> HleResult {
    match state().register_state_callback(ctx.rdi(), ctx.rsi()) {
        Ok(id) => id as i64,
        Err(e) => e as i64,
    }
}

/// sceNpUnregisterStateCallbackA(int id)
fn cb_unregister_state_callback_a(ctx: &mut CpuContext) -> HleResult {
    state().unregister_state_callback(ctx.rdi() as i32).map_or_else(|e| e as i64, |()| OK)
}

/// sceNpRegisterStateCallback(callback, userdata): one callback.
fn cb_register_state_callback(ctx: &mut CpuContext) -> HleResult {
    state().register_legacy_state_callback(ctx.rdi(), ctx.rsi()).map_or_else(|e| e as i64, |()| OK)
}

/// sceNpUnregisterStateCallback()
fn cb_unregister_state_callback(_ctx: &mut CpuContext) -> HleResult {
    state().unregister_legacy_state_callback().map_or_else(|e| e as i64, |()| OK)
}

/// sceNpGetNpId(userId, SceNpId*)
fn cb_get_np_id(ctx: &mut CpuContext) -> HleResult {
    signed_out_output(ctx, ctx.rsi(), NP_ID_SIZE)
}

/// sceNpGetOnlineId(userId, SceNpOnlineId*)
fn cb_get_online_id(ctx: &mut CpuContext) -> HleResult {
    signed_out_output(ctx, ctx.rsi(), ONLINE_ID_SIZE)
}

/// sceNpGetAccountIdA(userId, SceNpAccountId*)
fn cb_get_account_id_a(ctx: &mut CpuContext) -> HleResult {
    signed_out_output(ctx, ctx.rsi(), 8)
}

/// sceNpGetAccountCountryA(userId, SceNpCountryCode*)
fn cb_get_account_country_a(ctx: &mut CpuContext) -> HleResult {
    signed_out_output(ctx, ctx.rsi(), COUNTRY_CODE_SIZE)
}

/// Clear an account output and report the user as signed out.
fn signed_out_output(ctx: &CpuContext, out: u64, len: usize) -> HleResult {
    if out == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    if !clear(ctx, out, len) {
        return ERR_MEMORY_FAULT as i64;
    }
    ERR_SIGNED_OUT as i64
}

/// sceNpGetAccountAge(reqId, userId, uint8_t* age)
fn cb_get_account_age(ctx: &mut CpuContext) -> HleResult {
    let (req, out) = (ctx.rdi() as i32, ctx.rdx());
    if req <= 0 || out == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    signed_out_output(ctx, out, 1)
}

/// sceNpGetAccountLanguage2(reqId, userId, SceNpLanguageCode*): the output
/// is left untouched.
fn cb_get_account_language2(ctx: &mut CpuContext) -> HleResult {
    let (req, out) = (ctx.rdi() as i32, ctx.rdx());
    if req <= 0 || out == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    with_request(req, |_| ERR_SIGNED_OUT)
}

fn cb_create_request(_ctx: &mut CpuContext) -> HleResult {
    state().create_request(false) as i64
}

/// sceNpCreateAsyncRequest(const SceNpCreateAsyncRequestParameter*)
fn cb_create_async_request(ctx: &mut CpuContext) -> HleResult {
    let param = ctx.rdi();
    if param == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    match ctx.read_u64(param) {
        None => ERR_MEMORY_FAULT as i64,
        Some(size) if size < ASYNC_PARAM_SIZE => ERR_INVALID_SIZE as i64,
        Some(_) => state().create_request(true) as i64,
    }
}

fn cb_delete_request(ctx: &mut CpuContext) -> HleResult {
    with_request(ctx.rdi() as i32, |r| {
        *r = Request { state: RequestState::Free, is_async: false, result: 0 };
        0
    })
}

fn cb_abort_request(ctx: &mut CpuContext) -> HleResult {
    with_request(ctx.rdi() as i32, |r| {
        r.state = RequestState::Aborted;
        r.result = ERR_ABORTED;
        0
    })
}

/// sceNpCheckNpAvailability(reqId, const SceNpOnlineId* onlineId): the
/// online id names a local user, and no user is signed in, so no user
/// matches it.
fn cb_check_np_availability(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    ERR_USER_NOT_FOUND as i64
}

/// Smallest non-default name resolution timeout, in microseconds.
const NP_MIN_RESOLVE_TIMEOUT_US: u32 = 1_000_000;
/// Smallest non-default connect / send / receive timeout, in microseconds.
const NP_MIN_TRANSFER_TIMEOUT_US: u32 = 10_000_000;

/// sceNpSetTimeout(reqId, int32 resolveRetry, uint32 resolveTimeout,
/// uint32 connTimeout, uint32 sendTimeout, uint32 recvTimeout): a zero
/// timeout keeps its default. The arguments are checked before the request.
/// No request ever reaches a server, so the values are not kept.
fn cb_set_timeout(ctx: &mut CpuContext) -> HleResult {
    let req = ctx.rdi() as i32;
    let retry = ctx.rsi() as i32;
    let resolve = ctx.rdx() as u32;
    let transfers = [ctx.rcx() as u32, ctx.r8() as u32, ctx.r9() as u32];
    if req <= 0 || retry < 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    if retry == 0 && resolve == 0 && transfers.iter().all(|&t| t == 0) {
        return ERR_INVALID_ARGUMENT as i64;
    }
    let below = |timeout: u32, minimum: u32| timeout != 0 && timeout < minimum;
    if below(resolve, NP_MIN_RESOLVE_TIMEOUT_US) || transfers.iter().any(|&t| below(t, NP_MIN_TRANSFER_TIMEOUT_US)) {
        return ERR_INVALID_ARGUMENT as i64;
    }
    with_request(req, |_| 0)
}

/// sceNpCheckNpReachability(reqId, userId)
fn cb_check_np_reachability(ctx: &mut CpuContext) -> HleResult {
    let req = ctx.rdi() as i32;
    if req <= 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    with_request(req, complete_signed_out)
}

/// sceNpPollAsync(reqId, int* result): only completed async requests.
fn cb_poll_async(ctx: &mut CpuContext) -> HleResult {
    let (req, out) = (ctx.rdi() as i32, ctx.rsi());
    if out == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    let result = {
        let mut st = state();
        let Some(r) = st.request(req) else {
            return ERR_REQUEST_NOT_FOUND as i64;
        };
        if !r.is_async || r.state == RequestState::Ready {
            return ERR_INVALID_ID as i64;
        }
        r.result
    };
    if ctx.write_u32(out, result as u32) { OK } else { ERR_MEMORY_FAULT as i64 }
}

/// sceNpCheckPremium(reqId, const SceNpCheckPremiumParameter*,
/// SceNpCheckPremiumResult*): not authorized, the user is signed out.
fn cb_check_premium(ctx: &mut CpuContext) -> HleResult {
    let (req, param, result) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    if param == 0 || result == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    let Some(size) = ctx.read_u64(param) else {
        return ERR_MEMORY_FAULT as i64;
    };
    if size < CHECK_PREMIUM_PARAM_SIZE {
        return ERR_INVALID_SIZE as i64;
    }
    let mut st = state();
    let Some(r) = st.request(req) else {
        return ERR_REQUEST_NOT_FOUND as i64;
    };
    match r.state {
        RequestState::Complete => {
            r.result = ERR_INVALID_ARGUMENT;
            return ERR_INVALID_ARGUMENT as i64;
        }
        RequestState::Aborted => return ERR_ABORTED as i64,
        _ => {}
    }
    if !clear(ctx, result, CHECK_PREMIUM_RESULT_SIZE) {
        return ERR_MEMORY_FAULT as i64;
    }
    complete_signed_out(r) as i64
}

/// sceNpGetState(userId, SceNpState*)
fn cb_get_state(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    if ctx.write_u32(out, NP_STATE_SIGNED_OUT) { OK } else { ERR_MEMORY_FAULT as i64 }
}

/// sceNpGetNpReachabilityState(userId, SceNpReachabilityState*)
fn cb_get_np_reachability_state(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    if ctx.write_u32(out, NP_REACHABILITY_UNAVAILABLE) { OK } else { ERR_MEMORY_FAULT as i64 }
}

/// sceNpHasSignedUp(userId, bool*)
fn cb_has_signed_up(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    if ctx.write_bytes(out, &[0]) { OK } else { ERR_MEMORY_FAULT as i64 }
}

/// sceNpNotifyPremiumFeature(const SceNpPremiumFeatureNotifyParameter*):
/// the usage report cannot reach PSN.
fn cb_notify_premium_feature(ctx: &mut CpuContext) -> HleResult {
    let param = ctx.rdi();
    if param == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    match ctx.read_u64(param) {
        None => ERR_MEMORY_FAULT as i64,
        Some(size) if size < NOTIFY_PREMIUM_PARAM_MIN_SIZE => ERR_INVALID_SIZE as i64,
        Some(_) => ERR_SIGNED_OUT as i64,
    }
}

/// sceNpGetUserIdByAccountId(SceNpAccountId, SceUserServiceUserId*): no
/// local user is signed in, so no account maps to one.
fn cb_get_user_id_by_account_id(ctx: &mut CpuContext) -> HleResult {
    let (account, out) = (ctx.rdi(), ctx.rsi());
    if out == 0 || account == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    ERR_USER_NOT_FOUND as i64
}

/// The PS4-style account queries by online id on a signed-out console:
/// a NULL argument is INVALID_ARGUMENT, otherwise `answer`; nothing is
/// written.
fn online_id_query(pointers: &[u64], answer: i32) -> HleResult {
    if pointers.contains(&0) {
        return ERR_INVALID_ARGUMENT as i64;
    }
    answer as i64
}

/// sceNpGetAccountId(onlineId*, u64* accountId)
fn cb_get_account_id(ctx: &mut CpuContext) -> HleResult {
    online_id_query(&[ctx.rdi(), ctx.rsi()], ERR_SIGNED_OUT)
}

/// sceNpGetUserIdByOnlineId(onlineId*, int* userId)
fn cb_get_user_id_by_online_id(ctx: &mut CpuContext) -> HleResult {
    online_id_query(&[ctx.rdi(), ctx.rsi()], ERR_SIGNED_OUT)
}

/// sceNpGetAccountCountry / sceNpGetAccountDateOfBirth(onlineId*, out*)
fn cb_get_account_by_online_id(ctx: &mut CpuContext) -> HleResult {
    online_id_query(&[ctx.rdi(), ctx.rsi()], ERR_USER_NOT_FOUND)
}

/// sceNpGetAccountLanguage(reqId, onlineId*, out*): the request id is not
/// checked.
fn cb_get_account_language(ctx: &mut CpuContext) -> HleResult {
    online_id_query(&[ctx.rsi(), ctx.rdx()], ERR_USER_NOT_FOUND)
}

/// sceNpGetParentalControlInfo(reqId, onlineId*, int8_t* age, info*)
fn cb_get_parental_control_info(ctx: &mut CpuContext) -> HleResult {
    online_id_query(&[ctx.rsi(), ctx.rdx(), ctx.rcx()], ERR_USER_NOT_FOUND)
}

/// sceNpCheckPlus(reqId, param*, result*)
fn cb_check_plus(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() as i32 == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    online_id_query(&[ctx.rsi(), ctx.rdx()], ERR_SIGNED_OUT)
}

/// sceNpSetGamePresenceOnline(reqId, onlineId*, presence): the request
/// must exist; the user is signed out.
fn cb_set_game_presence_online(ctx: &mut CpuContext) -> HleResult {
    let (req, online_id) = (ctx.rdi() as i32, ctx.rsi());
    if req == 0 || online_id == 0 {
        return ERR_INVALID_ARGUMENT as i64;
    }
    if state().request(req).is_none() {
        return ERR_REQUEST_NOT_FOUND as i64;
    }
    ERR_SIGNED_OUT as i64
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("3Zl8BePTh9Y", "sceNpCheckCallback", cb_check_callback),
    ("a8R9-75u4iM", "sceNpGetAccountId", cb_get_account_id),
    ("F6E4ycq9Dbg", "sceNpGetUserIdByOnlineId", cb_get_user_id_by_online_id),
    ("Ghz9iWDUtC4", "sceNpGetAccountCountry", cb_get_account_by_online_id),
    ("8VBTeRf1ZwI", "sceNpGetAccountDateOfBirth", cb_get_account_by_online_id),
    ("KZ1Mj9yEGYc", "sceNpGetAccountLanguage", cb_get_account_language),
    ("ilwLM4zOmu4", "sceNpGetParentalControlInfo", cb_get_parental_control_info),
    ("r6MyYJkryz8", "sceNpCheckPlus", cb_check_plus),
    ("KO+11cgC7N0", "sceNpSetGamePresenceOnline", cb_set_game_presence_online),
    ("xViqJdDgKl0", "sceNpUnregisterPlusEventCallback", cb_ok),
    ("Ec63y59l9tw", "sceNpSetNpTitleId", cb_set_np_title_id),
    ("A2CQ3kgSopQ", "sceNpSetContentRestriction", cb_set_content_restriction),
    ("VfRSmPmj8Q8", "sceNpRegisterStateCallback", cb_register_state_callback),
    // The non-A registration keeps no callback; there is nothing to remove.
    ("mjjTXh+NHWY", "sceNpUnregisterStateCallback", cb_unregister_state_callback),
    ("qQJfO8HAiaY", "sceNpRegisterStateCallbackA", cb_register_state_callback_a),
    ("M3wFXbYQtAA", "sceNpUnregisterStateCallbackA", cb_unregister_state_callback_a),
    ("uFJpaKNBAj4", "sceNpRegisterGamePresenceCallback", cb_ok),
    ("GImICnh+boA", "sceNpRegisterPlusEventCallback", cb_ok),
    ("+yqjab2fUJA", "sceNpRegisterPremiumEventCallback", cb_register_premium_event_callback),
    ("-Rjp3-YViXc", "sceNpUnregisterPremiumEventCallback", cb_ok),
    ("hw5KNqAAels", "sceNpRegisterNpReachabilityStateCallback", cb_register_np_reachability_state_callback),
    ("cRILAEvn+9M", "sceNpUnregisterNpReachabilityStateCallback", cb_unregister_np_reachability_state_callback),
    ("p-o74CnoNzY", "sceNpGetNpId", cb_get_np_id),
    ("XDncXQIJUSk", "sceNpGetOnlineId", cb_get_online_id),
    ("rbknaUjpqWo", "sceNpGetAccountIdA", cb_get_account_id_a),
    ("JT+t00a3TxA", "sceNpGetAccountCountryA", cb_get_account_country_a),
    ("+4DegjBqV1g", "sceNpGetAccountAge", cb_get_account_age),
    ("3Tcz5bNCfZQ", "sceNpGetAccountLanguage2", cb_get_account_language2),
    ("GpLQDNKICac", "sceNpCreateRequest", cb_create_request),
    ("eiqMCt9UshI", "sceNpCreateAsyncRequest", cb_create_async_request),
    ("S7QTn72PrDw", "sceNpDeleteRequest", cb_delete_request),
    ("OzKvTvg3ZYU", "sceNpAbortRequest", cb_abort_request),
    ("2rsFmlGWleQ", "sceNpCheckNpAvailability", cb_check_np_availability),
    ("-QglDeRr8D8", "sceNpSetTimeout", cb_set_timeout),
    ("KfGZg2y73oM", "sceNpCheckNpReachability", cb_check_np_reachability),
    ("uqcPJLWL08M", "sceNpPollAsync", cb_poll_async),
    ("O80NrhUOPGY", "sceNpCheckPremium", cb_check_premium),
    ("eQH7nWPcAgc", "sceNpGetState", cb_get_state),
    ("e-ZuhGEoeC4", "sceNpGetNpReachabilityState", cb_get_np_reachability_state),
    ("Oad3rvY-NJQ", "sceNpHasSignedUp", cb_has_signed_up),
    ("P6piso307SE", "sceNpNotifyPremiumFeature", cb_notify_premium_feature),
    ("VgYczPGB5ss", "sceNpGetUserIdByAccountId", cb_get_user_id_by_account_id),
    ("JELHf4xPufo", "sceNpCheckCallbackForLib", cb_ok),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
    reg.register_fn("0c7HbXRKUt4", "sceNpRegisterStateCallbackForToolkit", "libSceNpManagerForToolkit", cb_ok);
    for &(nid, name, cb) in UTILITY_EXPORTS {
        reg.register_fn(nid, name, "libSceNpUtility", cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    fn args(ctx: &mut CpuContext, args: &[u64]) {
        for (r, v) in [Rdi, Rsi, Rdx, Rcx].iter().zip(args) {
            ctx.set(*r, *v);
        }
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS.iter().chain(UTILITY_EXPORTS) {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// State callbacks: eight A slots with ids 1..8 reused after an
    /// unregister, one legacy callback; each is told the state once.
    #[test]
    fn state_callback_slots() {
        let mut st = NpState::new();
        assert_eq!(st.register_state_callback(0, 0), Err(ERR_INVALID_ARGUMENT));
        assert_eq!(st.register_state_callback(0x1000, 7), Ok(1));
        assert_eq!(st.register_state_callback(0x1000, 7), Err(ERR_CALLBACK_ALREADY_REGISTERED));
        let (legacy, a) = st.take_pending();
        assert!(legacy.is_none());
        assert_eq!(a.iter().map(|c| (c.callback, c.userdata)).collect::<Vec<_>>(), [(0x1000, 7)]);
        assert!(st.take_pending().1.is_empty(), "told once");
        for i in 1..8u64 {
            assert_eq!(st.register_state_callback(0x1000 + i * 0x10, 0), Ok(i as i32 + 1));
        }
        assert_eq!(st.register_state_callback(0x2000, 0), Err(ERR_CALLBACK_MAX));
        for id in [0, 9, -1, i32::MIN, i32::MAX] {
            assert_eq!(st.unregister_state_callback(id), Err(ERR_INVALID_ID), "id {id}");
        }
        assert_eq!(st.unregister_state_callback(3), Ok(()));
        assert_eq!(st.register_state_callback(0x1020, 0), Ok(3), "the freed id");
        for id in 1..=8 {
            assert_eq!(st.unregister_state_callback(id), Ok(()));
        }
        assert_eq!(st.unregister_state_callback(1), Err(ERR_CALLBACK_NOT_REGISTERED));
        // Unregistered before a check: never called.
        assert_eq!(st.register_state_callback(0x3000, 0), Ok(1));
        assert_eq!(st.unregister_state_callback(1), Ok(()));
        assert!(st.take_pending().1.is_empty());
        // The legacy callback.
        assert_eq!(st.unregister_legacy_state_callback(), Err(ERR_CALLBACK_NOT_REGISTERED));
        assert_eq!(st.register_legacy_state_callback(0x4000, 9), Ok(()));
        assert_eq!(st.register_legacy_state_callback(0x4000, 9), Err(ERR_CALLBACK_ALREADY_REGISTERED));
        assert_eq!(st.take_pending().0.map(|c| c.userdata), Some(9));
        assert_eq!(st.unregister_legacy_state_callback(), Ok(()));
        assert_eq!(st.unregister_legacy_state_callback(), Err(ERR_CALLBACK_NOT_REGISTERED));
    }

    /// Signed out: no bandwidth test starts, no context exists; a premium
    /// callback needs a function.
    #[test]
    fn bandwidth_test_and_premium_callback() {
        const BASE: u64 = 0x5D42_0300_0000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        args(&mut ctx, &[BASE, BASE]);
        assert_eq!(cb_bandwidth_test_init_start(&mut ctx), ERR_SIGNED_OUT as i64);
        ctx.write_u32(BASE, 7);
        args(&mut ctx, &[1, BASE]);
        assert_eq!(cb_bandwidth_test_get_status(&mut ctx), ERR_INVALID_ARGUMENT as i64);
        assert_eq!(ctx.read_u32(BASE), Some(0));
        args(&mut ctx, &[1, 0]);
        assert_eq!(cb_bandwidth_test_get_status(&mut ctx), ERR_INVALID_ARGUMENT as i64);
        assert_eq!(cb_bandwidth_test_no_context(&mut ctx), ERR_INVALID_ARGUMENT as i64);
        args(&mut ctx, &[0, 0]);
        assert_eq!(cb_register_premium_event_callback(&mut ctx), ERR_INVALID_ARGUMENT as i64);
        args(&mut ctx, &[0x1000, 0]);
        assert_eq!(cb_register_premium_event_callback(&mut ctx), OK);
    }

    #[test]
    fn request_state_machine() {
        let mut st = NpState::new();
        let sync = st.create_request(false);
        let asynchronous = st.create_request(true);
        assert_eq!((sync, asynchronous), (1, 2));
        assert_eq!(complete_signed_out(st.request(sync).unwrap()), ERR_SIGNED_OUT);
        // A second check on a completed request is an argument error.
        assert_eq!(complete_signed_out(st.request(sync).unwrap()), ERR_INVALID_ARGUMENT);
        assert_eq!(complete_signed_out(st.request(asynchronous).unwrap()), 0);
        assert_eq!(st.request(asynchronous).unwrap().result, ERR_SIGNED_OUT);
        st.request(sync).unwrap().state = RequestState::Free;
        assert!(st.request(sync).is_none());
        assert_eq!(st.create_request(false), 1);
        for _ in 2..REQUEST_MAX {
            st.create_request(false);
        }
        assert_eq!(st.create_request(false), ERR_REQUEST_MAX);
        assert!(st.request(0).is_none());
    }

    #[test]
    fn signed_out_queries_and_requests() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D0B_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, &[0xAA; 64]);
        args(&mut ctx, &[1, BASE]);
        assert_eq!(cb_get_np_id(&mut ctx), ERR_SIGNED_OUT as i64);
        assert_eq!(ctx.read_bytes(BASE, NP_ID_SIZE).unwrap(), vec![0; NP_ID_SIZE]);
        assert_eq!(ctx.read_bytes(BASE + NP_ID_SIZE as u64, 1).unwrap(), vec![0xAA]);
        assert_eq!(cb_get_online_id(&mut ctx), ERR_SIGNED_OUT as i64);
        assert_eq!(cb_get_account_id_a(&mut ctx), ERR_SIGNED_OUT as i64);
        ctx.write_u32(BASE + 0x40, 7);
        args(&mut ctx, &[1, BASE + 0x40]);
        assert_eq!(cb_get_state(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE + 0x40), Some(NP_STATE_SIGNED_OUT));

        // Async request: polling before completion is INVALID_ID.
        ctx.write_u64(BASE + 0x80, ASYNC_PARAM_SIZE);
        args(&mut ctx, &[BASE + 0x80]);
        let req = cb_create_async_request(&mut ctx);
        assert!(req > 0);
        args(&mut ctx, &[req as u64, BASE + 0x90]);
        assert_eq!(cb_poll_async(&mut ctx), ERR_INVALID_ID as i64);
        args(&mut ctx, &[req as u64, 1]);
        assert_eq!(cb_check_np_reachability(&mut ctx), 0);
        args(&mut ctx, &[req as u64, BASE + 0x90]);
        assert_eq!(cb_poll_async(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE + 0x90), Some(ERR_SIGNED_OUT as u32));
        // Account language needs a live request.
        args(&mut ctx, &[req as u64, 1, BASE + 0xA0]);
        assert_eq!(cb_get_account_language2(&mut ctx), ERR_SIGNED_OUT as i64);
        args(&mut ctx, &[0x7FFF, 1, BASE + 0xA0]);
        assert_eq!(cb_get_account_language2(&mut ctx), ERR_REQUEST_NOT_FOUND as i64);
        args(&mut ctx, &[0, 1, BASE + 0xA0]);
        assert_eq!(cb_get_account_age(&mut ctx), ERR_INVALID_ARGUMENT as i64);
        // The PS4-style queries by online id (reference GuestNpManager
        // cases): NULL arguments first, then signed out / user not found.
        let id = BASE + 0x300;
        for (cb, ok_args, answer) in [
            (cb_get_account_id as HleCallback, [id, BASE + 0xA0, 0, 0], ERR_SIGNED_OUT),
            (cb_get_user_id_by_online_id, [id, BASE + 0xA0, 0, 0], ERR_SIGNED_OUT),
            (cb_get_account_by_online_id, [id, BASE + 0xA0, 0, 0], ERR_USER_NOT_FOUND),
            (cb_get_account_language, [0, id, BASE + 0xA0, 0], ERR_USER_NOT_FOUND),
            (cb_get_parental_control_info, [0, id, BASE + 0xA0, BASE + 0xB0], ERR_USER_NOT_FOUND),
            (cb_check_plus, [1, id, BASE + 0xA0, 0], ERR_SIGNED_OUT),
        ] {
            args(&mut ctx, &ok_args);
            assert_eq!(cb(&mut ctx), answer as i64);
            let mut bad = ok_args;
            bad[1] = 0;
            args(&mut ctx, &bad);
            assert_eq!(cb(&mut ctx), ERR_INVALID_ARGUMENT as i64);
        }
        args(&mut ctx, &[req as u64, id, 1]);
        assert_eq!(cb_set_game_presence_online(&mut ctx), ERR_SIGNED_OUT as i64);
        args(&mut ctx, &[0x7FFF, id, 1]);
        assert_eq!(cb_set_game_presence_online(&mut ctx), ERR_REQUEST_NOT_FOUND as i64);
        args(&mut ctx, &[req as u64, 0, 1]);
        assert_eq!(cb_set_game_presence_online(&mut ctx), ERR_INVALID_ARGUMENT as i64);

        // Premium check: zeroed result, then aborted requests report ABORTED.
        ctx.write_u64(BASE + 0x100, CHECK_PREMIUM_PARAM_SIZE);
        ctx.write_bytes(BASE + 0x200, &[0xFF; 33]);
        args(&mut ctx, &[]);
        let sync = cb_create_request(&mut ctx);
        args(&mut ctx, &[sync as u64, BASE + 0x100, BASE + 0x200]);
        assert_eq!(cb_check_premium(&mut ctx), ERR_SIGNED_OUT as i64);
        assert_eq!(ctx.read_bytes(BASE + 0x200, 33).unwrap(), vec![0; 33]);
        assert_eq!(cb_check_premium(&mut ctx), ERR_INVALID_ARGUMENT as i64);
        args(&mut ctx, &[sync as u64]);
        assert_eq!(cb_abort_request(&mut ctx), OK);
        args(&mut ctx, &[sync as u64, BASE + 0x100, BASE + 0x200]);
        assert_eq!(cb_check_premium(&mut ctx), ERR_ABORTED as i64);
        ctx.write_u64(BASE + 0x100, 8);
        assert_eq!(cb_check_premium(&mut ctx), ERR_INVALID_SIZE as i64);
        for id in [sync, req] {
            args(&mut ctx, &[id as u64]);
            assert_eq!(cb_delete_request(&mut ctx), OK);
            assert_eq!(cb_delete_request(&mut ctx), ERR_REQUEST_NOT_FOUND as i64);
        }
    }

    #[test]
    fn set_timeout_checks_arguments_then_request() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D0B_0010_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let set = |ctx: &mut CpuContext, v: [u64; 6]| {
            for (r, x) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(v) {
                ctx.set(*r, x);
            }
            cb_set_timeout(ctx)
        };
        let inv = ERR_INVALID_ARGUMENT as i64;
        let nf = ERR_REQUEST_NOT_FOUND as i64;
        assert_eq!(set(&mut ctx, [0x7fff_ffff, 2, 0, 0, 0, 0]), nf, "never created request");
        assert_eq!(set(&mut ctx, [0x7fff_ffff, 0, 0, 0, 0, 0]), inv, "arguments checked before the request");
        let req = cb_create_request(&mut ctx);
        assert!(req > 0);
        let r = req as u64;
        assert_eq!(set(&mut ctx, [r, 2, 1_000_000, 10_000_000, 10_000_000, 10_000_000]), OK, "minimum timeouts");
        assert_eq!(set(&mut ctx, [r, 0, 0, 0, 0, 30_000_000]), OK, "only the receive timeout");
        assert_eq!(set(&mut ctx, [r, 3, 0, 0, 0, 0]), OK, "only the resolve retries");
        assert_eq!(set(&mut ctx, [r, 0, 0, 0, 0, 0]), inv, "every value at its default");
        assert_eq!(set(&mut ctx, [0, 2, 0, 0, 0, 0]), inv, "request id 0");
        assert_eq!(set(&mut ctx, [u32::MAX as u64, 2, 0, 0, 0, 0]), inv, "negative request id");
        assert_eq!(set(&mut ctx, [r, u32::MAX as u64, 0, 0, 0, 30_000_000]), inv, "negative retries");
        assert_eq!(set(&mut ctx, [r, 0, 999_999, 0, 0, 0]), inv, "resolve timeout below 1 s");
        assert_eq!(set(&mut ctx, [r, 0, 0, 9_999_999, 0, 0]), inv, "connect timeout below 10 s");
        assert_eq!(set(&mut ctx, [r, 0, 0, 0, 9_999_999, 0]), inv, "send timeout below 10 s");
        assert_eq!(set(&mut ctx, [r, 0, 0, 0, 0, 1]), inv, "receive timeout below 10 s");
        ctx.set(Rdi, r);
        assert_eq!(cb_delete_request(&mut ctx), OK);
        assert_eq!(set(&mut ctx, [r, 2, 0, 0, 0, 0]), nf, "deleted request");

        ctx.write_u64(BASE, ASYNC_PARAM_SIZE);
        ctx.set(Rdi, BASE);
        let a = cb_create_async_request(&mut ctx);
        assert!(a > 0);
        assert_eq!(set(&mut ctx, [a as u64, 2, 0, 0, 0, 0]), OK, "async request");
        ctx.set(Rdi, a as u64);
        assert_eq!(cb_delete_request(&mut ctx), OK);
        assert_eq!(set(&mut ctx, [a as u64, 2, 0, 0, 0, 0]), nf, "deleted async request");
    }

    #[test]
    fn reachability_callback_registration() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        let (callback, userdata) = (0xC0DE_1000u64, 0x5555u64);
        assert_eq!(cb_unregister_np_reachability_state_callback(&mut ctx), ERR_CALLBACK_NOT_REGISTERED as i64);
        args(&mut ctx, &[0, userdata]);
        assert_eq!(cb_register_np_reachability_state_callback(&mut ctx), ERR_INVALID_ARGUMENT as i64);
        args(&mut ctx, &[callback, userdata]);
        assert_eq!(cb_register_np_reachability_state_callback(&mut ctx), OK);
        args(&mut ctx, &[callback, 0]);
        assert_eq!(cb_register_np_reachability_state_callback(&mut ctx), ERR_CALLBACK_ALREADY_REGISTERED as i64);
        args(&mut ctx, &[0, 0]);
        assert_eq!(cb_register_np_reachability_state_callback(&mut ctx), ERR_INVALID_ARGUMENT as i64);
        assert_eq!(cb_unregister_np_reachability_state_callback(&mut ctx), OK);
        assert_eq!(cb_unregister_np_reachability_state_callback(&mut ctx), ERR_CALLBACK_NOT_REGISTERED as i64);
        args(&mut ctx, &[callback, 0]);
        assert_eq!(cb_register_np_reachability_state_callback(&mut ctx), OK);
        assert_eq!(cb_unregister_np_reachability_state_callback(&mut ctx), OK);
    }

    #[test]
    fn check_np_availability_finds_no_user() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D0B_0020_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, b"p\0");
        // The third register is not an argument: it must not be written.
        ctx.write_u32(BASE + 0x100, 0x5555_5555);
        args(&mut ctx, &[1, BASE, BASE + 0x100]);
        assert_eq!(cb_check_np_availability(&mut ctx), ERR_USER_NOT_FOUND as i64);
        assert_eq!(ctx.read_u32(BASE + 0x100), Some(0x5555_5555));
        args(&mut ctx, &[1, 0]);
        assert_eq!(cb_check_np_availability(&mut ctx), ERR_INVALID_ARGUMENT as i64);
    }

    #[test]
    fn state_callbacks_once() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        args(&mut ctx, &[0, 0]);
        assert_eq!(cb_register_state_callback_a(&mut ctx), ERR_INVALID_ARGUMENT as i64);
        args(&mut ctx, &[0xC0DE_0000, 0x55]);
        let id = cb_register_state_callback_a(&mut ctx);
        assert!(id > 0);
        assert_eq!(cb_register_state_callback_a(&mut ctx), ERR_CALLBACK_ALREADY_REGISTERED as i64);
        let pending = state().take_pending().1;
        assert!(pending.iter().any(|c| c.callback == 0xC0DE_0000 && c.userdata == 0x55));
        assert!(!state().take_pending().1.iter().any(|c| c.callback == 0xC0DE_0000));
        args(&mut ctx, &[id as u64]);
        assert_eq!(cb_unregister_state_callback_a(&mut ctx), OK);
        assert_eq!(cb_unregister_state_callback_a(&mut ctx), ERR_CALLBACK_NOT_REGISTERED as i64);
    }
}
