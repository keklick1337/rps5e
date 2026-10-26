// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceNpWebApi2 / libSceNpCppWebApi — offline Web API.
//!
//! Offline: no PSN, so no request ever reaches a server. Library, user,
//! request and push-event objects are still tracked so games see valid ids
//! and proper argument validation; sending a request fails with "not
//! signed in" and the response accessors report that failure. Push
//! contexts are local identifiers; no push event is ever delivered.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceNpWebApi2";
const LIB_CPP: &str = "libSceNpCppWebApi";

const fn err(v: u32) -> HleResult {
    v as i32 as i64
}

const ERR_INVALID_ARGUMENT: HleResult = err(0x8055_3402);
const ERR_INVALID_LIB_CONTEXT_ID: HleResult = err(0x8055_3403);
const ERR_LIB_CONTEXT_NOT_FOUND: HleResult = err(0x8055_3404);
const ERR_USER_CONTEXT_NOT_FOUND: HleResult = err(0x8055_3405);
const ERR_REQUEST_NOT_FOUND: HleResult = err(0x8055_3406);
const ERR_NOT_SIGNED_IN: HleResult = err(0x8055_3407);
const ERR_MEMORY_FAULT: HleResult = err(0x8002_000E);

/// SceNpWebApi2PushEventPushContextId: 36-character UUID + NUL.
const PUSH_CONTEXT_ID_SIZE: usize = 37;
/// Library pools are reserved in 16 KiB units.
const POOL_ALIGN: u64 = 0x4000;
/// SceNpWebApi2MemoryPoolStats: poolSize, maxInuseSize, currentInuseSize,
/// reserved (i32).
const MEMORY_POOL_STATS_SIZE: usize = 32;
/// The only response header a (never received) response describes.
const CONTENT_TYPE: &str = "application/json; charset=utf-8";

#[allow(dead_code)]
struct WebApiRequest {
    user_context: i32,
    api_group: String,
    path: String,
    method: i32,
    headers: Vec<(String, String)>,
    timeout_us: u32,
    aborted: bool,
    sent: bool,
}

struct WebApi2State {
    initialized: bool,
    next_library_context: i32,
    next_push_event_handle: i32,
    next_push_event_callback: i32,
    next_user_context: i32,
    next_request: i64,
    library_contexts: BTreeMap<i32, u64>,
    user_contexts: BTreeSet<i32>,
    push_event_filters: BTreeSet<i32>,
    push_event_callbacks: BTreeSet<i32>,
    push_context_callbacks: BTreeSet<i32>,
    push_contexts: BTreeSet<String>,
    requests: BTreeMap<i64, WebApiRequest>,
}

impl WebApi2State {
    const fn new() -> Self {
        Self {
            initialized: false,
            next_library_context: 0,
            next_push_event_handle: 0,
            next_push_event_callback: 0,
            next_user_context: 1000,
            next_request: 999,
            library_contexts: BTreeMap::new(),
            user_contexts: BTreeSet::new(),
            push_event_filters: BTreeSet::new(),
            push_event_callbacks: BTreeSet::new(),
            push_context_callbacks: BTreeSet::new(),
            push_contexts: BTreeSet::new(),
            requests: BTreeMap::new(),
        }
    }

    fn is_valid_library_context(&self, id: i32) -> bool {
        id > 0 && id < 0x8000 && self.library_contexts.contains_key(&id)
    }

    fn create_library_context(&mut self, pool_size: u64) -> i32 {
        self.next_library_context = self.next_library_context.wrapping_add(1);
        self.library_contexts.insert(self.next_library_context, pool_size);
        self.next_library_context
    }

    fn next_push_handle(&mut self) -> i32 {
        self.next_push_event_handle = self.next_push_event_handle.wrapping_add(1);
        self.next_push_event_handle
    }

    fn next_callback(&mut self) -> i32 {
        self.next_push_event_callback = self.next_push_event_callback.wrapping_add(1);
        self.next_push_event_callback
    }

    fn remove_library_context(&mut self, id: i32) {
        self.library_contexts.remove(&id);
        if self.library_contexts.is_empty() {
            self.initialized = false;
            self.user_contexts.clear();
            self.push_event_filters.clear();
            self.push_event_callbacks.clear();
            self.push_context_callbacks.clear();
            self.push_contexts.clear();
            self.requests.clear();
        }
    }

    fn is_registered_push_pair(&self, user_ctx: i32, filter: i32) -> bool {
        self.initialized
            && user_ctx > 0
            && filter > 0
            && self.user_contexts.contains(&user_ctx)
            && self.push_event_filters.contains(&filter)
    }
}

static STATE: Mutex<WebApi2State> = Mutex::new(WebApi2State::new());

fn state() -> std::sync::MutexGuard<'static, WebApi2State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn read_str(ctx: &CpuContext, addr: u64, max: usize) -> Option<String> {
    if addr == 0 {
        return None;
    }
    let mut bytes = Vec::new();
    for i in 0..max as u64 {
        let b = ctx.read_bytes(addr + i, 1)?;
        if b[0] == 0 {
            return Some(String::from_utf8_lossy(&bytes).into_owned());
        }
        bytes.push(b[0]);
    }
    None
}

/// sceNpWebApi2Initialize(httpCtxId, poolSize) -> library context id; the
/// pool size is rounded up to 16 KiB.
fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    let http_ctx = ctx.rdi() as i32;
    let pool_size = ctx.rsi();
    let Some(rounded) = pool_size.checked_add(POOL_ALIGN - 1).map(|v| v & !(POOL_ALIGN - 1)) else {
        return ERR_INVALID_ARGUMENT;
    };
    let mut s = state();
    let id = s.create_library_context(rounded);
    s.initialized = true;
    log::debug!("npwebapi2: init http_ctx={http_ctx} pool=0x{pool_size:x} -> lib_ctx={id}");
    id as i64
}

/// sceNpWebApi2IntInitialize(const args*) / IntInitialize2: the internal
/// initializers take { i32 httpCtxId; i32 reserved; u64 poolSize; char*
/// name; (Int2: u32 pushConfigGroup; i32 reserved;) u64 structSize }; the
/// size must be the struct's (32 / 40).
fn int_initialize(ctx: &CpuContext, size_offset: u64, size: u64) -> HleResult {
    let args = ctx.rdi();
    if args == 0 || ctx.read_u64(args + size_offset) != Some(size) {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(pool_size) = ctx.read_u64(args + 8) else { return ERR_MEMORY_FAULT };
    let mut s = state();
    let id = s.create_library_context(pool_size);
    s.initialized = true;
    id as i64
}

fn cb_int_initialize(ctx: &mut CpuContext) -> HleResult {
    int_initialize(ctx, 0x18, 32)
}

fn cb_int_initialize2(ctx: &mut CpuContext) -> HleResult {
    int_initialize(ctx, 0x20, 40)
}

/// sceNpWebApi2AddWebTraceTag(requestId, const char* value): any request
/// id is accepted; there is no trace to tag.
fn cb_add_web_trace_tag(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 { ERR_INVALID_ARGUMENT } else { 0 }
}

/// sceNpWebApi2GetMemoryPoolStats(libCtxId, SceNpWebApi2MemoryPoolStats*):
/// requests never allocate from the pool.
fn cb_get_memory_pool_stats(ctx: &mut CpuContext) -> HleResult {
    let (lib_ctx, out) = (ctx.rdi() as i32, ctx.rsi());
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if lib_ctx <= 0 {
        return ERR_INVALID_LIB_CONTEXT_ID;
    }
    let Some(pool_size) = state().library_contexts.get(&lib_ctx).copied() else {
        return ERR_LIB_CONTEXT_NOT_FOUND;
    };
    let mut b = [0u8; MEMORY_POOL_STATS_SIZE];
    b[..8].copy_from_slice(&pool_size.to_le_bytes());
    if ctx.write_bytes(out, &b) { 0 } else { ERR_MEMORY_FAULT }
}

fn cb_push_event_create_filter(ctx: &mut CpuContext) -> HleResult {
    let lib_ctx = ctx.rdi() as i32;
    let mut s = state();
    if !s.is_valid_library_context(lib_ctx) {
        return ERR_INVALID_ARGUMENT;
    }
    let filter = s.next_push_handle();
    s.push_event_filters.insert(filter);
    filter as i64
}

/// sceNpWebApi2PushEventDeleteFilter(libCtxId, filterId)
fn cb_push_event_delete_filter(ctx: &mut CpuContext) -> HleResult {
    let lib_ctx = ctx.rdi() as i32;
    let filter = ctx.rsi() as i32;
    let mut s = state();
    if !s.is_valid_library_context(lib_ctx) || !s.push_event_filters.remove(&filter) {
        return ERR_INVALID_ARGUMENT;
    }
    0
}

fn cb_push_event_create_handle(ctx: &mut CpuContext) -> HleResult {
    let lib_ctx = ctx.rdi() as i32;
    let mut s = state();
    if !s.is_valid_library_context(lib_ctx) {
        return ERR_INVALID_ARGUMENT;
    }
    let handle = s.next_push_handle();
    s.initialized = true;
    handle as i64
}

/// PushEventAbortHandle(libCtxId, handleId) / PushEventSetHandleTimeout(
/// libCtxId, handleId, timeout): no push event is ever waited for, so the
/// handle has nothing to abort or time out; the library context must exist.
fn cb_push_event_handle_control(ctx: &mut CpuContext) -> HleResult {
    let lib_ctx = ctx.rdi() as i32;
    if !state().is_valid_library_context(lib_ctx) {
        return ERR_INVALID_ARGUMENT;
    }
    0
}

fn cb_push_event_delete_handle(ctx: &mut CpuContext) -> HleResult {
    log::debug!("npwebapi2: push-event-delete-handle lib_ctx={} handle={}", ctx.rdi() as i32, ctx.rsi() as u32);
    0
}

/// sceNpWebApi2CreateUserContext(libCtxId, userId) -> user context id.
fn cb_create_user_context(ctx: &mut CpuContext) -> HleResult {
    let lib_ctx = ctx.rdi() as i32;
    let user_id = ctx.rsi() as i32;
    let mut s = state();
    if !s.initialized || !s.is_valid_library_context(lib_ctx) || user_id == -1 {
        return ERR_INVALID_ARGUMENT;
    }
    s.next_user_context = s.next_user_context.wrapping_add(1);
    let id = s.next_user_context;
    s.user_contexts.insert(id);
    id as i64
}

/// sceNpWebApi2DeleteUserContext(userCtxId): also drops its requests.
fn cb_delete_user_context(ctx: &mut CpuContext) -> HleResult {
    let user_ctx = ctx.rdi() as i32;
    let mut s = state();
    if !s.user_contexts.remove(&user_ctx) {
        return ERR_USER_CONTEXT_NOT_FOUND;
    }
    s.requests.retain(|_, r| r.user_context != user_ctx);
    0
}

/// sceNpWebApi2PushEventRegisterCallback(userCtxId, filterId, cb, arg) -> callback id.
fn cb_push_event_register_callback(ctx: &mut CpuContext) -> HleResult {
    let user_ctx = ctx.rdi() as i32;
    let filter = ctx.rsi() as i32;
    let callback = ctx.rdx();
    let mut s = state();
    if callback == 0 || !s.is_registered_push_pair(user_ctx, filter) {
        return ERR_INVALID_ARGUMENT;
    }
    let id = s.next_callback();
    s.push_event_callbacks.insert(id);
    id as i64
}

/// sceNpWebApi2PushEventUnregisterCallback(userCtxId, callbackId)
fn cb_push_event_unregister_callback(ctx: &mut CpuContext) -> HleResult {
    let user_ctx = ctx.rdi() as i32;
    let callback = ctx.rsi() as i32;
    let mut s = state();
    if !s.user_contexts.contains(&user_ctx) {
        return ERR_USER_CONTEXT_NOT_FOUND;
    }
    if !s.push_event_callbacks.remove(&callback) {
        return ERR_INVALID_ARGUMENT;
    }
    0
}

/// Random RFC 4122 version 4 UUID text.
fn new_uuid() -> String {
    let mut b = [0u8; 16];
    let mut done = 0;
    while done < b.len() {
        let r = unsafe { libc::getrandom(b[done..].as_mut_ptr() as *mut libc::c_void, b.len() - done, 0) };
        if r > 0 {
            done += r as usize;
        } else if std::io::Error::last_os_error().raw_os_error() != Some(libc::EINTR) {
            break;
        }
    }
    b[6] = (b[6] & 0x0F) | 0x40;
    b[8] = (b[8] & 0x3F) | 0x80;
    let h: Vec<String> = b.iter().map(|x| format!("{x:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        h[0..4].concat(),
        h[4..6].concat(),
        h[6..8].concat(),
        h[8..10].concat(),
        h[10..16].concat()
    )
}

fn read_push_context_id(ctx: &CpuContext, addr: u64) -> Option<String> {
    read_str(ctx, addr, PUSH_CONTEXT_ID_SIZE)
}

/// sceNpWebApi2PushEventCreatePushContext(userCtxId, SceNpWebApi2PushEventPushContextId* out):
/// a locally generated identifier.
fn cb_push_event_create_push_context(ctx: &mut CpuContext) -> HleResult {
    let user_ctx = ctx.rdi() as i32;
    let out = ctx.rsi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let mut s = state();
    if !s.user_contexts.contains(&user_ctx) {
        return ERR_USER_CONTEXT_NOT_FOUND;
    }
    let uuid = new_uuid();
    let mut bytes = uuid.clone().into_bytes();
    bytes.push(0);
    if !ctx.write_bytes(out, &bytes) {
        return ERR_MEMORY_FAULT;
    }
    s.push_contexts.insert(uuid);
    0
}

/// sceNpWebApi2PushEventDeletePushContext(userCtxId, const PushContextId*)
fn cb_push_event_delete_push_context(ctx: &mut CpuContext) -> HleResult {
    let id = read_push_context_id(ctx, ctx.rsi());
    if let Some(id) = id {
        state().push_contexts.remove(&id);
    }
    0
}

/// sceNpWebApi2PushEventRegisterPushContextCallback(userCtxId,
/// const PushContextId*, filterId, cb, arg) -> callback id.
fn cb_push_event_register_push_context_callback(ctx: &mut CpuContext) -> HleResult {
    let user_ctx = ctx.rdi() as i32;
    let filter = ctx.rdx() as i32;
    let callback = ctx.rcx();
    let push_ctx = read_push_context_id(ctx, ctx.rsi());
    let mut s = state();
    let Some(push_ctx) = push_ctx else {
        return ERR_INVALID_ARGUMENT;
    };
    if callback == 0 || !s.push_contexts.contains(&push_ctx) || !s.is_registered_push_pair(user_ctx, filter) {
        return ERR_INVALID_ARGUMENT;
    }
    let id = s.next_callback();
    s.push_context_callbacks.insert(id);
    id as i64
}

/// sceNpWebApi2PushEventStartPushContextCallback(userCtxId, const PushContextId*)
fn cb_push_event_start_push_context_callback(ctx: &mut CpuContext) -> HleResult {
    let user_ctx = ctx.rdi() as i32;
    let push_ctx = read_push_context_id(ctx, ctx.rsi());
    let s = state();
    if !s.user_contexts.contains(&user_ctx) {
        return ERR_USER_CONTEXT_NOT_FOUND;
    }
    match push_ctx {
        Some(p) if s.push_contexts.contains(&p) => 0,
        _ => ERR_INVALID_ARGUMENT,
    }
}

/// sceNpWebApi2PushEventUnregisterPushContextCallback(userCtxId, callbackId)
fn cb_push_event_unregister_push_context_callback(ctx: &mut CpuContext) -> HleResult {
    let user_ctx = ctx.rdi() as i32;
    let callback = ctx.rsi() as i32;
    let mut s = state();
    if !s.user_contexts.contains(&user_ctx) {
        return ERR_USER_CONTEXT_NOT_FOUND;
    }
    if !s.push_context_callbacks.remove(&callback) {
        return ERR_INVALID_ARGUMENT;
    }
    0
}

/// sceNpWebApi2CreateRequest(userCtxId, apiGroup, path, method,
/// const ContentParameter*, int64_t* requestId)
fn cb_create_request(ctx: &mut CpuContext) -> HleResult {
    let user_ctx = ctx.rdi() as i32;
    let out = ctx.r9();
    let api_group = read_str(ctx, ctx.rsi(), 256);
    let path = read_str(ctx, ctx.rdx(), 4096);
    let method = ctx.rcx() as i32;
    let (Some(api_group), Some(path)) = (api_group, path) else {
        return ERR_INVALID_ARGUMENT;
    };
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let mut s = state();
    if !s.user_contexts.contains(&user_ctx) {
        return ERR_USER_CONTEXT_NOT_FOUND;
    }
    s.next_request += 1;
    let id = s.next_request;
    if !ctx.write_u64(out, id as u64) {
        return ERR_MEMORY_FAULT;
    }
    log::info!("[npwebapi2] request {api_group}{path} (offline, will not be sent)");
    s.requests.insert(
        id,
        WebApiRequest {
            user_context: user_ctx,
            api_group,
            path,
            method,
            headers: Vec::new(),
            timeout_us: 0,
            aborted: false,
            sent: false,
        },
    );
    0
}

/// sceNpWebApi2DeleteRequest(requestId)
fn cb_delete_request(ctx: &mut CpuContext) -> HleResult {
    match state().requests.remove(&(ctx.rdi() as i64)) {
        Some(_) => 0,
        None => ERR_REQUEST_NOT_FOUND,
    }
}

/// sceNpWebApi2AbortRequest(requestId)
fn cb_abort_request(ctx: &mut CpuContext) -> HleResult {
    match state().requests.get_mut(&(ctx.rdi() as i64)) {
        Some(r) => {
            r.aborted = true;
            0
        }
        None => ERR_REQUEST_NOT_FOUND,
    }
}

/// sceNpWebApi2AddHttpRequestHeader(requestId, fieldName, value)
fn cb_add_http_request_header(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i64;
    let name = read_str(ctx, ctx.rsi(), 4096).filter(|n| !n.is_empty());
    let value = read_str(ctx, ctx.rdx(), 16384);
    let (Some(name), Some(value)) = (name, value) else {
        return ERR_INVALID_ARGUMENT;
    };
    match state().requests.get_mut(&id) {
        Some(r) => {
            r.headers.push((name, value));
            0
        }
        None => ERR_REQUEST_NOT_FOUND,
    }
}

/// sceNpWebApi2SetRequestTimeout(requestId, timeoutUs)
fn cb_set_request_timeout(ctx: &mut CpuContext) -> HleResult {
    let timeout = ctx.rsi() as u32;
    match state().requests.get_mut(&(ctx.rdi() as i64)) {
        Some(r) => {
            r.timeout_us = timeout;
            0
        }
        None => ERR_REQUEST_NOT_FOUND,
    }
}

/// sceNpWebApi2SendRequest(requestId, data, size, ResponseInformationOption*):
/// fails — no user is signed in to PSN. The response information reports
/// no HTTP status and no data.
fn cb_send_request(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i64;
    let info = ctx.rcx();
    {
        let mut s = state();
        let Some(r) = s.requests.get_mut(&id) else {
            return ERR_REQUEST_NOT_FOUND;
        };
        r.sent = true;
        log::info!("[npwebapi2] send {}{} -> not signed in (offline)", r.api_group, r.path);
    }
    if info != 0 {
        // httpStatus (int), errorObject (char*), errorObjectSize, responseDataSize.
        let error_object = ctx.read_u64(info + 8).unwrap_or(0);
        let error_size = ctx.read_u64(info + 16).unwrap_or(0);
        if !ctx.write_u32(info, 0) || !ctx.write_u64(info + 24, 0) {
            return ERR_MEMORY_FAULT;
        }
        if error_object != 0 && error_size > 0 {
            ctx.write_bytes(error_object, &[0]);
        }
    }
    ERR_NOT_SIGNED_IN
}

fn request_exists(id: i64) -> bool {
    state().requests.contains_key(&id)
}

/// sceNpWebApi2ReadData(requestId, data, size): the response body is
/// empty, so 0 bytes are read.
fn cb_read_data(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 || ctx.rdx() == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if request_exists(ctx.rdi() as i64) { 0 } else { ERR_REQUEST_NOT_FOUND }
}

/// Value of response header `name`: only Content-Type is known.
fn response_header(name: &[u8]) -> &'static str {
    if name == b"Content-Type" || name == b"content-type" { CONTENT_TYPE } else { "" }
}

/// sceNpWebApi2GetHttpResponseHeaderValueLength(requestId, fieldName, size_t* length):
/// the length includes the terminator; unknown headers are 0.
fn cb_get_http_response_header_value_length(ctx: &mut CpuContext) -> HleResult {
    let (id, name, out) = (ctx.rdi() as i64, ctx.rsi(), ctx.rdx());
    if name == 0 || out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if !request_exists(id) {
        return ERR_REQUEST_NOT_FOUND;
    }
    let Some(name) = ctx.read_cstr_bytes(name, 4096) else {
        return ERR_MEMORY_FAULT;
    };
    let value = response_header(&name);
    let len = if value.is_empty() { 0 } else { value.len() as u64 + 1 };
    if ctx.write_u64(out, len) { 0 } else { ERR_MEMORY_FAULT }
}

/// sceNpWebApi2GetHttpResponseHeaderValue(requestId, fieldName, char* value, size_t size):
/// the value is truncated to fit with its terminator.
fn cb_get_http_response_header_value(ctx: &mut CpuContext) -> HleResult {
    let (id, name, out, size) = (ctx.rdi() as i64, ctx.rsi(), ctx.rdx(), ctx.rcx());
    if name == 0 || out == 0 || size == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if !request_exists(id) {
        return ERR_REQUEST_NOT_FOUND;
    }
    let Some(name) = ctx.read_cstr_bytes(name, 4096) else {
        return ERR_MEMORY_FAULT;
    };
    let value = response_header(&name).as_bytes();
    let n = value.len().min(size as usize - 1);
    let mut bytes = value[..n].to_vec();
    bytes.push(0);
    if ctx.write_bytes(out, &bytes) { 0 } else { ERR_MEMORY_FAULT }
}

/// sceNpWebApi2Terminate(libCtxId)
fn cb_terminate(ctx: &mut CpuContext) -> HleResult {
    let lib_ctx = ctx.rdi() as i32;
    if lib_ctx <= 0 {
        return ERR_INVALID_LIB_CONTEXT_ID;
    }
    let mut s = state();
    if !s.library_contexts.contains_key(&lib_ctx) {
        return ERR_LIB_CONTEXT_NOT_FOUND;
    }
    s.remove_library_context(lib_ctx);
    0
}

/// Called every frame to expire timed-out requests; none are ever in flight.
fn cb_check_timeout(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// sce::Np::CppWebApi::Common::initialize(const InitParams&, LibContext&),
/// used only when no loaded module exports it.
///
/// libSceNpCppWebApi.prx ships with every game that imports this, and its
/// own initialize (read in the decrypted module) does the whole job in guest
/// code: it fails with 0x80553511 if the LibContext is already registered,
/// with 0x80553519 if `InitParams+0x00` (worker count) is above 1, maps
/// `InitParams+0x20` bytes of named flexible memory, builds an mspace on it,
/// starts the callback/web-API worker threads (stack, priority and affinity
/// at +0x08/+0x10/+0x18), stores its context object in the LibContext and
/// links it into the module's list of contexts. Every other export of the
/// module checks that list, so the module's own initialize must run: an HLE
/// success here would leave the LibContext unregistered. Without the module
/// the rest of the C++ API is missing as well; succeeding keeps a game that
/// only initializes it going.
fn cb_cpp_common_initialize(ctx: &mut CpuContext) -> HleResult {
    log::debug!("np_cppwebapi: common_initialize params=0x{:x} lib_ctx=0x{:x}", ctx.rdi(), ctx.rsi());
    0
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("dowMWFgowXY", "sceNpWebApi2InitializeForPresence", cb_initialize),
    ("zXaFo7euxsQ", "sceNpWebApi2IntInitialize", cb_int_initialize),
    ("9KSGFMRnp3k", "sceNpWebApi2IntInitialize2", cb_int_initialize2),
    ("MgsTa76wlEk", "sceNpWebApi2AddWebTraceTag", cb_add_web_trace_tag),
    ("+o9816YQhqQ", "sceNpWebApi2Initialize", cb_initialize),
    ("Xweb+naPZ8Y", "sceNpWebApi2GetMemoryPoolStats", cb_get_memory_pool_stats),
    ("MsaFhR+lPE4", "sceNpWebApi2PushEventCreateFilter", cb_push_event_create_filter),
    ("KJdPcOGmK58", "sceNpWebApi2PushEventDeleteFilter", cb_push_event_delete_filter),
    ("WV1GwM32NgY", "sceNpWebApi2PushEventCreateHandle", cb_push_event_create_handle),
    ("fIATVMo4Y1w", "sceNpWebApi2PushEventDeleteHandle", cb_push_event_delete_handle),
    ("1OLgvahaSco", "sceNpWebApi2PushEventAbortHandle", cb_push_event_handle_control),
    ("KWkc6Q3tjXc", "sceNpWebApi2PushEventSetHandleTimeout", cb_push_event_handle_control),
    ("sk54bi6FtYM", "sceNpWebApi2CreateUserContext", cb_create_user_context),
    ("9X9+cneTGUU", "sceNpWebApi2DeleteUserContext", cb_delete_user_context),
    ("fY3QqeNkF8k", "sceNpWebApi2PushEventRegisterCallback", cb_push_event_register_callback),
    ("hOnIlcGrO6g", "sceNpWebApi2PushEventUnregisterCallback", cb_push_event_unregister_callback),
    ("NNVf18SlbT8", "sceNpWebApi2PushEventCreatePushContext", cb_push_event_create_push_context),
    ("QafxeZM3WK4", "sceNpWebApi2PushEventDeletePushContext", cb_push_event_delete_push_context),
    (
        "lxtHJMwBsaU",
        "sceNpWebApi2PushEventRegisterPushContextCallback",
        cb_push_event_register_push_context_callback,
    ),
    ("AAj9X+4aGYA", "sceNpWebApi2PushEventStartPushContextCallback", cb_push_event_start_push_context_callback),
    (
        "PmyrbbJSFz0",
        "sceNpWebApi2PushEventUnregisterPushContextCallback",
        cb_push_event_unregister_push_context_callback,
    ),
    ("3EI-OSJ65Xc", "sceNpWebApi2CreateRequest", cb_create_request),
    ("vvzWO-DvG1s", "sceNpWebApi2DeleteRequest", cb_delete_request),
    ("zpiPsH7dbFQ", "sceNpWebApi2AbortRequest", cb_abort_request),
    ("egOOvrnF6mI", "sceNpWebApi2AddHttpRequestHeader", cb_add_http_request_header),
    ("TjAutbrkr60", "sceNpWebApi2SetRequestTimeout", cb_set_request_timeout),
    ("lQOCF84lvzw", "sceNpWebApi2SendRequest", cb_send_request),
    ("OOY9+ObfKec", "sceNpWebApi2ReadData", cb_read_data),
    ("HwP3aM+c85c", "sceNpWebApi2GetHttpResponseHeaderValueLength", cb_get_http_response_header_value_length),
    ("hksbskNToEA", "sceNpWebApi2GetHttpResponseHeaderValue", cb_get_http_response_header_value),
    ("bEvXpcEk200", "sceNpWebApi2Terminate", cb_terminate),
    ("3Tt9zL3tkoc", "sceNpWebApi2CheckTimeout", cb_check_timeout),
];

const CPP_EXPORTS: &[(&str, &str, HleCallback)] = &[(
    "UYPxv8MIzGo",
    "_ZN3sce2Np9CppWebApi6Common10initializeERKNS2_10InitParamsERNS2_10LibContextE",
    cb_cpp_common_initialize,
)];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
    for &(nid, name, cb) in CPP_EXPORTS {
        reg.register_fallback_fn(nid, name, LIB_CPP, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS.iter().chain(CPP_EXPORTS) {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// Port of the guest push handle checks: abort and timeout need a live
    /// library context, any handle id is accepted.
    /// The internal initializers check their struct size; AddWebTraceTag
    /// only needs a value.
    #[test]
    fn internal_initializers_and_trace_tag() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D44_0900_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, &[0; 0x40]);
        ctx.write_u64(BASE + 8, 0x8000);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_int_initialize(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.write_u64(BASE + 0x18, 32);
        let id = cb_int_initialize(&mut ctx);
        assert!(id > 0);
        assert!(state().is_valid_library_context(id as i32));
        assert_eq!(cb_int_initialize2(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.write_u64(BASE + 0x20, 40);
        assert!(cb_int_initialize2(&mut ctx) > 0);
        ctx.set(Rdi, 0);
        assert_eq!(cb_int_initialize(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.set(Rsi, 0);
        assert_eq!(cb_add_web_trace_tag(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.set(Rsi, BASE);
        assert_eq!(cb_add_web_trace_tag(&mut ctx), 0);
    }

    #[test]
    fn push_event_handle_abort_and_timeout() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        let lib_ctx = state().create_library_context(0x4000);
        ctx.set(Rdi, lib_ctx as u64);
        let handle = cb_push_event_create_handle(&mut ctx);
        assert!(handle > 0);
        for h in [handle, handle + 1000] {
            ctx.set(Rsi, h as u64);
            ctx.set(Rdx, 30_000);
            assert_eq!(cb_push_event_handle_control(&mut ctx), 0);
        }
        state().library_contexts.remove(&lib_ctx);
        assert_eq!(cb_push_event_handle_control(&mut ctx), ERR_INVALID_ARGUMENT);
    }

    #[test]
    fn cpp_initialize_defers_to_the_shipped_module() {
        let mut reg = HleRegistry::new();
        register(&mut reg);
        let e = reg.lookup("UYPxv8MIzGo").expect("Common::initialize registered");
        assert!(e.fallback, "the module's own initialize must win when it is loaded");
        assert!(!reg.lookup("+o9816YQhqQ").unwrap().fallback);
    }

    #[test]
    fn context_lifecycle() {
        let mut s = WebApi2State::new();
        let lib = s.create_library_context(POOL_ALIGN);
        s.initialized = true;
        assert_eq!(lib, 1);
        assert!(s.is_valid_library_context(1));
        assert!(!s.is_valid_library_context(2));
        let f = s.next_push_handle();
        s.push_event_filters.insert(f);
        s.user_contexts.insert(1001);
        assert!(s.is_registered_push_pair(1001, f));
        assert!(!s.is_registered_push_pair(1002, f));
        s.remove_library_context(lib);
        assert!(!s.initialized);
        assert!(!s.is_registered_push_pair(1001, f));
    }

    #[test]
    fn uuid_format() {
        let u = new_uuid();
        assert_eq!(u.len(), 36);
        assert_eq!(u.as_bytes()[14], b'4');
        assert_eq!(u.matches('-').count(), 4);
    }

    fn set(ctx: &mut CpuContext, args: &[u64]) {
        for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
            ctx.set(*r, *v);
        }
    }

    #[test]
    fn requests_and_push_contexts_offline() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C0E_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        set(&mut ctx, &[1, 0x10000]);
        let lib = cb_initialize(&mut ctx) as u64;
        set(&mut ctx, &[lib, 1]);
        let user = cb_create_user_context(&mut ctx) as u64;
        ctx.write_bytes(BASE, b"sessionManager\0/v1/sessions\0");
        set(&mut ctx, &[user, BASE, BASE + 15, 0, 0, BASE + 0x100]);
        assert_eq!(cb_create_request(&mut ctx), 0);
        let req = ctx.read_u64(BASE + 0x100).unwrap();
        ctx.write_bytes(BASE + 0x200, &[0xEE; 32]);
        ctx.write_u32(BASE + 0x200, 200);
        set(&mut ctx, &[req, 0, 0, BASE + 0x200]);
        assert_eq!(cb_send_request(&mut ctx), ERR_NOT_SIGNED_IN);
        assert_eq!(ctx.read_u32(BASE + 0x200), Some(0));
        assert_eq!(ctx.read_u64(BASE + 0x218), Some(0));
        set(&mut ctx, &[req, BASE + 0x300, 16]);
        assert_eq!(cb_read_data(&mut ctx), 0);
        set(&mut ctx, &[req, BASE, BASE + 0x300]);
        assert_eq!(cb_get_http_response_header_value_length(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE + 0x300), Some(0));
        ctx.write_bytes(BASE + 0x340, b"Content-Type\0");
        set(&mut ctx, &[req, BASE + 0x340, BASE + 0x300]);
        assert_eq!(cb_get_http_response_header_value_length(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE + 0x300), Some(CONTENT_TYPE.len() as u64 + 1));
        set(&mut ctx, &[req, BASE + 0x340, BASE + 0x360, 12]);
        assert_eq!(cb_get_http_response_header_value(&mut ctx), 0);
        assert_eq!(ctx.read_cstring(BASE + 0x360, 32).unwrap(), "application");
        set(&mut ctx, &[lib, BASE + 0x380]);
        assert_eq!(cb_get_memory_pool_stats(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE + 0x380), Some(0x10000));
        set(&mut ctx, &[0, BASE + 0x380]);
        assert_eq!(cb_get_memory_pool_stats(&mut ctx), ERR_INVALID_LIB_CONTEXT_ID);
        set(&mut ctx, &[0x7777, BASE + 0x380]);
        assert_eq!(cb_get_memory_pool_stats(&mut ctx), ERR_LIB_CONTEXT_NOT_FOUND);
        set(&mut ctx, &[req]);
        assert_eq!(cb_abort_request(&mut ctx), 0);
        assert_eq!(cb_delete_request(&mut ctx), 0);
        assert_eq!(cb_delete_request(&mut ctx), ERR_REQUEST_NOT_FOUND);

        set(&mut ctx, &[user, BASE + 0x400]);
        assert_eq!(cb_push_event_create_push_context(&mut ctx), 0);
        assert_eq!(ctx.read_cstring(BASE + 0x400, 40).unwrap().len(), 36);
        set(&mut ctx, &[lib, 0]);
        let filter = cb_push_event_create_filter(&mut ctx) as u64;
        set(&mut ctx, &[user, BASE + 0x400, filter, 0x1234, 0]);
        let cb = cb_push_event_register_push_context_callback(&mut ctx);
        assert!(cb > 0);
        set(&mut ctx, &[user, BASE + 0x400]);
        assert_eq!(cb_push_event_start_push_context_callback(&mut ctx), 0);
        set(&mut ctx, &[user, cb as u64]);
        assert_eq!(cb_push_event_unregister_push_context_callback(&mut ctx), 0);
        assert_eq!(cb_push_event_unregister_push_context_callback(&mut ctx), ERR_INVALID_ARGUMENT);
        set(&mut ctx, &[lib, filter]);
        assert_eq!(cb_push_event_delete_filter(&mut ctx), 0);
        assert_eq!(cb_push_event_delete_filter(&mut ctx), ERR_INVALID_ARGUMENT);
        set(&mut ctx, &[user]);
        assert_eq!(cb_delete_user_context(&mut ctx), 0);
        assert_eq!(cb_delete_user_context(&mut ctx), ERR_USER_CONTEXT_NOT_FOUND);
        set(&mut ctx, &[lib]);
        assert_eq!(cb_terminate(&mut ctx), 0);
        assert_eq!(cb_terminate(&mut ctx), ERR_LIB_CONTEXT_NOT_FOUND);
        set(&mut ctx, &[0]);
        assert_eq!(cb_terminate(&mut ctx), ERR_INVALID_LIB_CONTEXT_ID);
    }
}
