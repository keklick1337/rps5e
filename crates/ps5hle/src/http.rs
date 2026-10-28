// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceHttp and libSceHttp2 — contexts, templates, connections, requests
//! and URI parsing, offline.
//!
//! Object bookkeeping is complete (ids, options, headers, epoll handles) so
//! setup code succeeds, but the console is offline: sending a request times
//! out and the response getters report that result. URI parse / build /
//! escape are full implementations.
//!
//! libSceHttp ids carry the object type in their high half (1 = context,
//! 3 = template, 4 = connection, 5 = request) and a slot index in the low
//! half; a connection starts from its template's settings and a request
//! from its connection's.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex, MutexGuard};

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB_HTTP: &str = "libSceHttp";
const LIB_HTTP2: &str = "libSceHttp2";

const fn err(v: u32) -> HleResult {
    v as i32 as i64
}

const HTTP_ERROR_BEFORE_INIT: HleResult = err(0x8043_1001);
const HTTP_ERROR_OUT_OF_MEMORY: HleResult = err(0x8043_1022);
const HTTP_ERROR_ABORTED: HleResult = err(0x8043_1080);
const HTTP_ERROR_PARSE_NOT_FOUND: HleResult = err(0x8043_2025);
const HTTP_ERROR_PARSE_INVALID_RESPONSE: HleResult = err(0x8043_2060);
const HTTP_ERROR_PARSE_INVALID_VALUE: HleResult = err(0x8043_21FE);
const HTTP_ERROR_INVALID_ID: HleResult = err(0x8043_1100);
const HTTP_ERROR_INVALID_VALUE: HleResult = err(0x8043_11FE);
const HTTP_ERROR_INVALID_URL: HleResult = err(0x8043_3060);
const HTTP_ERROR_BEFORE_SEND: HleResult = err(0x8043_1065);
const HTTP_ERROR_TIMEOUT: HleResult = err(0x8043_1068);
const HTTP_ERROR_UNKNOWN_METHOD: HleResult = err(0x8043_106B);

const HTTP2_ERROR_BEFORE_SEND: HleResult = err(0x817B_1065);
const HTTP2_ERROR_TIMEOUT: HleResult = err(0x817B_1068);
const HTTP2_ERROR_ABORTED: HleResult = err(0x817B_1080);
const HTTP2_ERROR_INVALID_ID: HleResult = err(0x817B_1100);
const HTTP2_ERROR_NULL_POINTER: HleResult = err(0x817B_1225);

const ERR_MEMORY_FAULT: HleResult = err(0x8002_000E);

/// Network errno a timed-out request reports through sceHttpGetLastErrno.
const HTTP_ERRNO_TIMEDOUT: i32 = 60;

/// Longest guest string read for URLs, names and header values.
const MAX_STRING: usize = 1 << 20;

// ---------------------------------------------------------------------------
// libSceHttp objects
// ---------------------------------------------------------------------------

const ID_TYPE_HTTP: i32 = 1;
const ID_TYPE_TEMPLATE: i32 = 3;
const ID_TYPE_CONNECTION: i32 = 4;
const ID_TYPE_REQUEST: i32 = 5;
const HTTP_MAX: usize = 32;
const MAX_OBJECTS: usize = 65536;

const fn make_id(index: usize, ty: i32) -> i32 {
    (ty << 16) | index as i32
}

/// (type, slot index) of an object id.
fn split_id(id: i32) -> (i32, usize) {
    ((id as u32 >> 16) as i32, (id as u32 & 0xFFFF) as usize)
}

/// Settings shared by templates, connections and requests.
#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HttpBase {
    headers: Vec<(String, String)>,
    nonblock: bool,
    chunked_transfer: bool,
    auto_redirect: bool,
    auth_enabled: bool,
    ssl_callback: u64,
    ssl_user_arg: u64,
    /// sceHttpSetAuthInfoCallback: never called, as no server is reached.
    auth_info_callback: u64,
    auth_info_user_arg: u64,
    /// sceHttpSetRequestStatusCallback: never called, as no request changes
    /// state without a network.
    status_callback: u64,
    status_user_arg: u64,
    ssl_flags: u32,
    min_ssl_version: u32,
    http_ctx: usize,
    resolve_timeout: u32,
    resolve_retry: i32,
    connect_timeout: u32,
    send_timeout: u32,
    recv_timeout: u32,
    /// sceHttpSetRecvBlockSize: 0 keeps the library's block size.
    recv_block_size: u32,
    /// sceHttpSetInflateGZIPEnabled.
    inflate_gzip: bool,
    /// sceHttpSetResponseHeaderMaxSize: 0 keeps the library's limit.
    response_header_max_size: u64,
    /// sceHttpsSetSslVersion.
    ssl_version: i32,
    /// sceHttpSetRedirectCallback / sceHttpSetCookieRecvCallback: no
    /// response ever arrives offline, so neither is called.
    redirect_callback: u64,
    redirect_user_arg: u64,
    cookie_recv_callback: u64,
    cookie_recv_user_arg: u64,
    /// sceHttpSetAcceptEncodingGZIPEnabled; a template starts from its
    /// context's default.
    accept_encoding_gzip: bool,
}

impl HttpBase {
    fn new(http_ctx: usize) -> Self {
        Self {
            headers: Vec::new(),
            nonblock: false,
            chunked_transfer: false,
            auto_redirect: true,
            auth_enabled: true,
            ssl_callback: 0,
            ssl_user_arg: 0,
            auth_info_callback: 0,
            status_callback: 0,
            status_user_arg: 0,
            auth_info_user_arg: 0,
            ssl_flags: 0xA7,
            min_ssl_version: 0,
            http_ctx,
            resolve_timeout: 1_000_000,
            resolve_retry: 4,
            connect_timeout: 30_000_000,
            send_timeout: 120_000_000,
            recv_timeout: 120_000_000,
            recv_block_size: 0,
            inflate_gzip: false,
            response_header_max_size: 0,
            ssl_version: 0,
            redirect_callback: 0,
            redirect_user_arg: 0,
            cookie_recv_callback: 0,
            cookie_recv_user_arg: 0,
            accept_encoding_gzip: false,
        }
    }

    /// mode 1 appends; mode 0 overwrites the value of every header with
    /// that name.
    fn add_header(&mut self, name: String, value: String, append: bool) {
        if append {
            self.headers.push((name, value));
        } else {
            for h in self.headers.iter_mut().filter(|h| h.0 == name) {
                h.1 = value.clone();
            }
        }
    }
}

#[allow(dead_code)]
struct HttpContext {
    memid: i32,
    ssl_ctx: i32,
    pool_size: u64,
    /// sceHttpSetDefaultAcceptEncodingGZIPEnabled.
    accept_encoding_gzip: bool,
}

#[allow(dead_code)]
#[derive(Clone)]
struct HttpTemplate {
    base: HttpBase,
    user_agent: String,
    http_ver: i32,
    auto_proxy_conf: bool,
}

#[allow(dead_code)]
#[derive(Clone)]
struct HttpConnection {
    tmpl: HttpTemplate,
    url: String,
    keep_alive: bool,
}

#[allow(dead_code)]
struct HttpRequest {
    conn: HttpConnection,
    method: String,
    url: String,
    content_length: u64,
    send_attempted: bool,
    send_result: HleResult,
    status_code: i32,
}

impl HttpRequest {
    fn result(&self) -> HleResult {
        if self.send_attempted { self.send_result } else { HTTP_ERROR_BEFORE_SEND }
    }
}

/// A sceHttpCreateEpoll object: its owning context slot and whether
/// sceHttpAbortWaitRequest was called on it. The abort stays set, so every
/// later wait on the handle fails with ABORTED.
#[derive(Clone, Copy)]
struct HttpEpoll {
    context: usize,
    aborted: bool,
}

struct HttpState {
    contexts: Vec<Option<HttpContext>>,
    templates: Vec<Option<HttpTemplate>>,
    connections: Vec<Option<HttpConnection>>,
    requests: Vec<Option<HttpRequest>>,
    /// Live epoll handles by guest address.
    epolls: BTreeMap<u64, HttpEpoll>,
    /// Released epoll handle blocks, reused before allocating new ones.
    free_epoll_blocks: Vec<u64>,
    /// Epoll handle (guest address) and user argument sceHttpSetEpoll gave
    /// a template, connection or request, by id.
    object_epolls: BTreeMap<i32, (u64, u64)>,
}

/// Signalled when an epoll handle is aborted or destroyed, waking
/// sceHttpWaitRequest callers blocked on it.
static HTTP_EPOLL_CHANGED: std::sync::Condvar = std::sync::Condvar::new();

static HTTP: LazyLock<Mutex<HttpState>> = LazyLock::new(|| {
    Mutex::new(HttpState {
        contexts: (0..HTTP_MAX).map(|_| None).collect(),
        templates: Vec::new(),
        connections: Vec::new(),
        requests: Vec::new(),
        epolls: BTreeMap::new(),
        free_epoll_blocks: Vec::new(),
        object_epolls: BTreeMap::new(),
    })
});

fn http() -> MutexGuard<'static, HttpState> {
    HTTP.lock().unwrap_or_else(|p| p.into_inner())
}

/// Store `value` in the first free slot (or a new one) of `slots`.
fn insert_slot<T>(slots: &mut Vec<Option<T>>, value: T) -> Option<usize> {
    if let Some(i) = slots.iter().position(|s| s.is_none()) {
        slots[i] = Some(value);
        return Some(i);
    }
    if slots.len() >= MAX_OBJECTS {
        return None;
    }
    slots.push(Some(value));
    Some(slots.len() - 1)
}

fn slot<T>(slots: &mut [Option<T>], id: i32, ty: i32) -> Option<&mut T> {
    let (t, index) = split_id(id);
    if t != ty {
        return None;
    }
    slots.get_mut(index)?.as_mut()
}

impl HttpState {
    /// Whether any libSceHttp context is live (calls on epoll handles fail
    /// with BEFORE_INIT otherwise).
    fn initialized(&self) -> bool {
        self.contexts.iter().any(Option::is_some)
    }

    fn context_slot(&self, id: i32) -> Option<usize> {
        let (t, index) = split_id(id);
        (t == ID_TYPE_HTTP && index < HTTP_MAX && self.contexts[index].is_some()).then_some(index)
    }

    fn request(&mut self, id: i32) -> Option<&mut HttpRequest> {
        slot(&mut self.requests, id, ID_TYPE_REQUEST)
    }

    /// Settings of a template, connection or (with `include_request`)
    /// request id.
    fn base(&mut self, id: i32, include_request: bool) -> Option<&mut HttpBase> {
        match split_id(id).0 {
            ID_TYPE_TEMPLATE => slot(&mut self.templates, id, ID_TYPE_TEMPLATE).map(|t| &mut t.base),
            ID_TYPE_CONNECTION => slot(&mut self.connections, id, ID_TYPE_CONNECTION).map(|c| &mut c.tmpl.base),
            ID_TYPE_REQUEST if include_request => self.request(id).map(|r| &mut r.conn.tmpl.base),
            _ => None,
        }
    }
}

fn read_string(ctx: &CpuContext, addr: u64) -> Option<String> {
    ctx.read_cstr_bytes(addr, MAX_STRING).map(|b| String::from_utf8_lossy(&b).into_owned())
}

/// Apply `f` to the settings of `id`; INVALID_ID when there are none.
fn update_base(id: i32, include_request: bool, f: impl FnOnce(&mut HttpBase)) -> HleResult {
    match http().base(id, include_request) {
        Some(b) => {
            f(b);
            0
        }
        None => HTTP_ERROR_INVALID_ID,
    }
}

/// sceHttpInit(libnetMemId, libsslCtxId, poolSize) -> context id. The
/// memory pool and the SSL context must exist.
fn cb_http_init(ctx: &mut CpuContext) -> HleResult {
    let (memid, ssl_ctx, pool_size) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx());
    if pool_size == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    if !crate::ssl::ssl_valid(ssl_ctx) || !crate::net::pool_valid(memid) {
        return HTTP_ERROR_OUT_OF_MEMORY;
    }
    let mut st = http();
    match st.contexts.iter().position(|c| c.is_none()) {
        Some(i) => {
            st.contexts[i] = Some(HttpContext { memid, ssl_ctx, pool_size, accept_encoding_gzip: false });
            make_id(i, ID_TYPE_HTTP) as i64
        }
        None => HTTP_ERROR_OUT_OF_MEMORY,
    }
}

/// sceHttpTerm(libhttpCtxId)
fn cb_http_term(ctx: &mut CpuContext) -> HleResult {
    let mut st = http();
    match st.context_slot(ctx.rdi() as i32) {
        Some(i) => {
            st.contexts[i] = None;
            // The context's epoll handles die with it; waiters wake up.
            st.epolls.retain(|_, e| e.context != i);
            HTTP_EPOLL_CHANGED.notify_all();
            // With the last context the objects go too: their ids are
            // INVALID_ID after the next sceHttpInit.
            if !st.initialized() {
                st.templates.clear();
                st.connections.clear();
                st.requests.clear();
                st.object_epolls.clear();
            }
            0
        }
        None => HTTP_ERROR_INVALID_ID,
    }
}

/// sceHttpCreateTemplate(libhttpCtxId, userAgent, httpVer, isAutoProxyConf)
fn cb_http_create_template(ctx: &mut CpuContext) -> HleResult {
    let user_agent = read_string(ctx, ctx.rsi()).unwrap_or_default();
    let mut st = http();
    let Some(http_ctx) = st.context_slot(ctx.rdi() as i32) else {
        return HTTP_ERROR_OUT_OF_MEMORY;
    };
    let mut base = HttpBase::new(http_ctx);
    base.accept_encoding_gzip = st.contexts[http_ctx].as_ref().is_some_and(|c| c.accept_encoding_gzip);
    let tmpl = HttpTemplate { base, user_agent, http_ver: ctx.rdx() as i32, auto_proxy_conf: ctx.rcx() as i32 != 0 };
    match insert_slot(&mut st.templates, tmpl) {
        Some(i) => make_id(i, ID_TYPE_TEMPLATE) as i64,
        None => HTTP_ERROR_OUT_OF_MEMORY,
    }
}

fn cb_http_delete_template(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let mut st = http();
    if slot(&mut st.templates, id, ID_TYPE_TEMPLATE).is_none() {
        return HTTP_ERROR_INVALID_ID;
    }
    st.templates[split_id(id).1] = None;
    st.object_epolls.remove(&id);
    0
}

/// Create a connection from template `tmpl_id` to `url`.
fn create_connection(tmpl_id: i32, url: String, keep_alive: bool) -> HleResult {
    let mut st = http();
    let Some(tmpl) = slot(&mut st.templates, tmpl_id, ID_TYPE_TEMPLATE).cloned() else {
        return HTTP_ERROR_OUT_OF_MEMORY;
    };
    log::info!("[http] connection {url} (offline)");
    match insert_slot(&mut st.connections, HttpConnection { tmpl, url, keep_alive }) {
        Some(i) => make_id(i, ID_TYPE_CONNECTION) as i64,
        None => HTTP_ERROR_OUT_OF_MEMORY,
    }
}

/// sceHttpCreateConnection(tmplId, serverName, scheme, port, enableKeepAlive)
fn cb_http_create_connection(ctx: &mut CpuContext) -> HleResult {
    let (server, scheme) = (ctx.rsi(), ctx.rdx());
    if server == 0 || scheme == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    let (Some(server), Some(scheme)) = (read_string(ctx, server), read_string(ctx, scheme)) else {
        return ERR_MEMORY_FAULT;
    };
    let port = ctx.rcx() as u16;
    let mut url = format!("{scheme}://{server}");
    if port != 0 {
        url.push_str(&format!(":{port}"));
    }
    create_connection(ctx.rdi() as i32, url, ctx.r8() as i32 != 0)
}

/// sceHttpCreateConnectionWithURL(tmplId, url, enableKeepAlive)
fn cb_http_create_connection_with_url(ctx: &mut CpuContext) -> HleResult {
    let url = if ctx.rsi() == 0 { None } else { read_string(ctx, ctx.rsi()) };
    let Some(url) = url.filter(|u| !u.is_empty()) else {
        return HTTP_ERROR_INVALID_URL;
    };
    create_connection(ctx.rdi() as i32, url, ctx.rdx() as i32 != 0)
}

fn cb_http_delete_connection(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let mut st = http();
    if slot(&mut st.connections, id, ID_TYPE_CONNECTION).is_none() {
        return HTTP_ERROR_INVALID_ID;
    }
    st.connections[split_id(id).1] = None;
    st.object_epolls.remove(&id);
    0
}

const HTTP_METHODS: [&str; 8] = ["GET", "POST", "HEAD", "OPTIONS", "PUT", "DELETE", "TRACE", "CONNECT"];

/// Create a request on `conn_id`; an empty URL or a dead connection fail
/// with OUT_OF_MEMORY.
fn create_request(conn_id: i32, method: String, url: String, content_length: u64) -> HleResult {
    let mut st = http();
    let conn = if url.is_empty() { None } else { slot(&mut st.connections, conn_id, ID_TYPE_CONNECTION).cloned() };
    let Some(conn) = conn else {
        return HTTP_ERROR_OUT_OF_MEMORY;
    };
    log::info!("[http] request {method} {url} (offline)");
    let request = HttpRequest {
        conn,
        method,
        url,
        content_length,
        send_attempted: false,
        send_result: HTTP_ERROR_BEFORE_SEND,
        status_code: 0,
    };
    match insert_slot(&mut st.requests, request) {
        Some(i) => make_id(i, ID_TYPE_REQUEST) as i64,
        None => HTTP_ERROR_OUT_OF_MEMORY,
    }
}

/// sceHttpCreateRequest / sceHttpCreateRequestWithURL(connId, int method,
/// path or url, contentLength)
fn cb_http_create_request(ctx: &mut CpuContext) -> HleResult {
    let Some(method) = usize::try_from(ctx.rsi() as i32).ok().and_then(|m| HTTP_METHODS.get(m)) else {
        return HTTP_ERROR_UNKNOWN_METHOD;
    };
    let path = if ctx.rdx() == 0 { None } else { read_string(ctx, ctx.rdx()) };
    let Some(path) = path else {
        return HTTP_ERROR_INVALID_VALUE;
    };
    create_request(ctx.rdi() as i32, method.to_string(), path, ctx.rcx())
}

/// sceHttpCreateRequest2(connId, const char* method, path, contentLength)
fn cb_http_create_request2(ctx: &mut CpuContext) -> HleResult {
    let method = if ctx.rsi() == 0 { None } else { read_string(ctx, ctx.rsi()) };
    let path = if ctx.rdx() == 0 { None } else { read_string(ctx, ctx.rdx()) };
    let (Some(method), Some(path)) = (method, path) else {
        return HTTP_ERROR_INVALID_VALUE;
    };
    create_request(ctx.rdi() as i32, method, path, ctx.rcx())
}

/// sceHttpCreateRequestWithURL2(connId, const char* method, url, contentLength)
fn cb_http_create_request_with_url2(ctx: &mut CpuContext) -> HleResult {
    let method = if ctx.rsi() == 0 { None } else { read_string(ctx, ctx.rsi()) };
    let Some(method) = method else {
        return HTTP_ERROR_INVALID_VALUE;
    };
    let url = if ctx.rdx() == 0 { None } else { read_string(ctx, ctx.rdx()) };
    let Some(url) = url.filter(|u| !u.is_empty()) else {
        return HTTP_ERROR_INVALID_URL;
    };
    create_request(ctx.rdi() as i32, method, url, ctx.rcx())
}

fn cb_http_delete_request(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let mut st = http();
    if st.request(id).is_none() {
        return HTTP_ERROR_INVALID_ID;
    }
    st.requests[split_id(id).1] = None;
    st.object_epolls.remove(&id);
    0
}

/// sceHttpSetRequestContentLength(reqId, contentLength)
fn cb_http_set_request_content_length(ctx: &mut CpuContext) -> HleResult {
    let length = ctx.rsi();
    match http().request(ctx.rdi() as i32) {
        Some(r) => {
            r.content_length = length;
            0
        }
        None => HTTP_ERROR_INVALID_ID,
    }
}

/// sceHttpAddRequestHeader(id, name, value, mode): template, connection or
/// request ids; mode 0 overwrites, 1 adds.
fn cb_http_add_request_header(ctx: &mut CpuContext) -> HleResult {
    let mode = ctx.rcx() as u32;
    if ctx.rsi() == 0 || ctx.rdx() == 0 || mode > 1 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    let (Some(name), Some(value)) = (read_string(ctx, ctx.rsi()), read_string(ctx, ctx.rdx())) else {
        return ERR_MEMORY_FAULT;
    };
    update_base(ctx.rdi() as i32, true, |b| b.add_header(name, value, mode == 1))
}

/// sceHttpSetNonblock / sceHttpTrySetNonblock(id, int enable)
fn cb_http_set_nonblock(ctx: &mut CpuContext) -> HleResult {
    let enable = ctx.rsi() as i32 != 0;
    if !http().initialized() {
        return HTTP_ERROR_BEFORE_INIT;
    }
    update_base(ctx.rdi() as i32, true, |b| b.nonblock = enable)
}

/// The getters of a per-object switch (id, int* out): BEFORE_INIT, then a
/// NULL output INVALID_VALUE, then INVALID_ID; 0 or 1 is written.
fn get_switch(ctx: &CpuContext, f: impl FnOnce(&HttpBase) -> bool) -> HleResult {
    let (id, out) = (ctx.rdi() as i32, ctx.rsi());
    let mut st = http();
    if !st.initialized() {
        return HTTP_ERROR_BEFORE_INIT;
    }
    if out == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    let Some(base) = st.base(id, true) else { return HTTP_ERROR_INVALID_ID };
    if ctx.write_u32(out, f(base) as u32) { 0 } else { ERR_MEMORY_FAULT }
}

/// sceHttpGetNonblock / sceHttpTryGetNonblock(id, int* isEnable)
fn cb_http_get_nonblock(ctx: &mut CpuContext) -> HleResult {
    get_switch(ctx, |b| b.nonblock)
}

/// sceHttpGetAutoRedirect(id, int* isEnable)
fn cb_http_get_auto_redirect(ctx: &mut CpuContext) -> HleResult {
    get_switch(ctx, |b| b.auto_redirect)
}

/// sceHttpGetAcceptEncodingGZIPEnabled(id, int* isEnable)
fn cb_http_get_accept_encoding_gzip(ctx: &mut CpuContext) -> HleResult {
    get_switch(ctx, |b| b.accept_encoding_gzip)
}

/// sceHttpSetAcceptEncodingGZIPEnabled(id, int enable)
fn cb_http_set_accept_encoding_gzip(ctx: &mut CpuContext) -> HleResult {
    let enable = ctx.rsi() as i32 != 0;
    if !http().initialized() {
        return HTTP_ERROR_BEFORE_INIT;
    }
    update_base(ctx.rdi() as i32, true, |b| b.accept_encoding_gzip = enable)
}

/// sceHttpSetDefaultAcceptEncodingGZIPEnabled(libhttpCtxId, int enable):
/// the value the context's new templates start with.
fn cb_http_set_default_accept_encoding_gzip(ctx: &mut CpuContext) -> HleResult {
    let enable = ctx.rsi() as i32 != 0;
    let mut st = http();
    if !st.initialized() {
        return HTTP_ERROR_BEFORE_INIT;
    }
    let Some(i) = st.context_slot(ctx.rdi() as i32) else { return HTTP_ERROR_INVALID_ID };
    if let Some(c) = st.contexts[i].as_mut() {
        c.accept_encoding_gzip = enable;
    }
    0
}

/// sceHttpCacheRedirectedConnectionEnabled(id, int enable): there is no
/// connection to keep, the id must exist.
fn cb_http_cache_redirected_connection_enabled(ctx: &mut CpuContext) -> HleResult {
    let mut st = http();
    if !st.initialized() {
        return HTTP_ERROR_BEFORE_INIT;
    }
    if st.base(ctx.rdi() as i32, true).is_some() { 0 } else { HTTP_ERROR_INVALID_ID }
}

/// sceHttpsEnableOptionPrivate / sceHttpsDisableOptionPrivate(id, flags):
/// flags outside 0x2DFF are INVALID_VALUE; nothing is stored.
fn cb_https_option_private(ctx: &mut CpuContext) -> HleResult {
    const KNOWN_FLAGS: u32 = 0x2DFF;
    let flags = ctx.rsi() as u32;
    let mut st = http();
    if !st.initialized() {
        return HTTP_ERROR_BEFORE_INIT;
    }
    if flags & !KNOWN_FLAGS != 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    if st.base(ctx.rdi() as i32, true).is_some() { 0 } else { HTTP_ERROR_INVALID_ID }
}

/// sceHttpSetChunkedTransferEnabled(id, enable): 0 or 1.
fn cb_http_set_chunked_transfer_enabled(ctx: &mut CpuContext) -> HleResult {
    let enable = ctx.rsi() as u32;
    if enable > 1 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    update_base(ctx.rdi() as i32, true, |b| b.chunked_transfer = enable == 1)
}

/// sceHttpsSetSslCallback(id, cbfunc, userArg): stored; no TLS handshake
/// ever happens offline, so it is never invoked.
fn cb_https_set_ssl_callback(ctx: &mut CpuContext) -> HleResult {
    let (cb, arg) = (ctx.rsi(), ctx.rdx());
    update_base(ctx.rdi() as i32, true, |b| {
        b.ssl_callback = cb;
        b.ssl_user_arg = arg;
    })
}

fn cb_https_set_min_ssl_version(ctx: &mut CpuContext) -> HleResult {
    let v = ctx.rsi() as u32;
    update_base(ctx.rdi() as i32, true, |b| b.min_ssl_version = v)
}

fn cb_https_disable_option(ctx: &mut CpuContext) -> HleResult {
    let flags = ctx.rsi() as u32;
    update_base(ctx.rdi() as i32, true, |b| b.ssl_flags &= !flags)
}

/// Resolver settings apply to templates and connections only.
fn cb_http_set_resolve_timeout(ctx: &mut CpuContext) -> HleResult {
    let us = ctx.rsi() as u32;
    update_base(ctx.rdi() as i32, false, |b| b.resolve_timeout = us)
}

fn cb_http_set_resolve_retry(ctx: &mut CpuContext) -> HleResult {
    let retry = ctx.rsi() as i32;
    update_base(ctx.rdi() as i32, false, |b| b.resolve_retry = retry)
}

fn cb_http_set_connect_timeout(ctx: &mut CpuContext) -> HleResult {
    let us = ctx.rsi() as u32;
    update_base(ctx.rdi() as i32, true, |b| b.connect_timeout = us)
}

fn cb_http_set_send_timeout(ctx: &mut CpuContext) -> HleResult {
    let us = ctx.rsi() as u32;
    update_base(ctx.rdi() as i32, true, |b| b.send_timeout = us)
}

fn cb_http_set_recv_timeout(ctx: &mut CpuContext) -> HleResult {
    let us = ctx.rsi() as u32;
    update_base(ctx.rdi() as i32, true, |b| b.recv_timeout = us)
}

/// sceHttpSetRecvBlockSize(id, uint32_t blockSize): the size of each read
/// from the connection; no data is ever received, so it is only stored.
fn cb_http_set_recv_block_size(ctx: &mut CpuContext) -> HleResult {
    let size = ctx.rsi() as u32;
    update_base(ctx.rdi() as i32, true, |b| b.recv_block_size = size)
}

fn cb_http_set_auto_redirect(ctx: &mut CpuContext) -> HleResult {
    let enable = ctx.rsi() as i32 != 0;
    if !http().initialized() {
        return HTTP_ERROR_BEFORE_INIT;
    }
    update_base(ctx.rdi() as i32, true, |b| b.auto_redirect = enable)
}

fn cb_http_set_auth_enabled(ctx: &mut CpuContext) -> HleResult {
    let enable = ctx.rsi() as i32 != 0;
    update_base(ctx.rdi() as i32, true, |b| b.auth_enabled = enable)
}

/// sceHttpSetAuthInfoCallback(id, SceHttpAuthInfoCallback cbfunc, void* userArg):
/// no request reaches a server, so the callback is stored and never called.
fn cb_http_set_auth_info_callback(ctx: &mut CpuContext) -> HleResult {
    let (cb, arg) = (ctx.rsi(), ctx.rdx());
    update_base(ctx.rdi() as i32, true, |b| {
        b.auth_info_callback = cb;
        b.auth_info_user_arg = arg;
    })
}

/// sceHttpSetRequestStatusCallback(id, cbfunc(int reqId, int status, void*
/// arg), void* userArg): stored for a template, connection or request (a
/// NULL callback clears it); no request changes state offline, so it is
/// never called.
fn cb_http_set_request_status_callback(ctx: &mut CpuContext) -> HleResult {
    let (cb, arg) = (ctx.rsi(), ctx.rdx());
    update_base(ctx.rdi() as i32, true, |b| {
        b.status_callback = cb;
        b.status_user_arg = if cb == 0 { 0 } else { arg };
    })
}

/// sceHttpSetInflateGZIPEnabled(id, enable): 0 or 1.
fn cb_http_set_inflate_gzip_enabled(ctx: &mut CpuContext) -> HleResult {
    let enable = ctx.rsi() as u32;
    if enable > 1 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    update_base(ctx.rdi() as i32, true, |b| b.inflate_gzip = enable == 1)
}

/// sceHttpSetResponseHeaderMaxSize(id, uint64_t headerSize)
fn cb_http_set_response_header_max_size(ctx: &mut CpuContext) -> HleResult {
    let size = ctx.rsi();
    update_base(ctx.rdi() as i32, true, |b| b.response_header_max_size = size)
}

/// sceHttpSetRedirectCallback(id, cbfunc, userArg)
fn cb_http_set_redirect_callback(ctx: &mut CpuContext) -> HleResult {
    let (cb, arg) = (ctx.rsi(), ctx.rdx());
    update_base(ctx.rdi() as i32, true, |b| {
        b.redirect_callback = cb;
        b.redirect_user_arg = arg;
    })
}

/// sceHttpSetCookieRecvCallback(id, cbfunc, userArg)
fn cb_http_set_cookie_recv_callback(ctx: &mut CpuContext) -> HleResult {
    let (cb, arg) = (ctx.rsi(), ctx.rdx());
    update_base(ctx.rdi() as i32, true, |b| {
        b.cookie_recv_callback = cb;
        b.cookie_recv_user_arg = arg;
    })
}

/// sceHttpsEnableOption(id, sslFlags)
fn cb_https_enable_option(ctx: &mut CpuContext) -> HleResult {
    let flags = ctx.rsi() as u32;
    update_base(ctx.rdi() as i32, true, |b| b.ssl_flags |= flags)
}

/// sceHttpsSetSslVersion(id, int version)
fn cb_https_set_ssl_version(ctx: &mut CpuContext) -> HleResult {
    let version = ctx.rsi() as i32;
    update_base(ctx.rdi() as i32, true, |b| b.ssl_version = version)
}

/// sceHttpsGetSslError(id, int* errNum, uint32_t* detail): no TLS session
/// ever runs offline, so there is no error to report.
fn cb_https_get_ssl_error(ctx: &mut CpuContext) -> HleResult {
    let (err_num, detail) = (ctx.rsi(), ctx.rdx());
    if err_num == 0 || detail == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    if http().base(ctx.rdi() as i32, true).is_none() {
        return HTTP_ERROR_INVALID_ID;
    }
    if ctx.write_u32(err_num, 0) && ctx.write_u32(detail, 0) { 0 } else { ERR_MEMORY_FAULT }
}

/// The context slot of libhttpCtxId, or INVALID_ID.
fn http_context(id: i32) -> Result<usize, HleResult> {
    http().context_slot(id).ok_or(HTTP_ERROR_INVALID_ID)
}

/// sceHttpRedirectCacheFlush(libhttpCtxId): no redirect is ever followed
/// offline, so the cache is always empty.
fn cb_http_redirect_cache_flush(ctx: &mut CpuContext) -> HleResult {
    http_context(ctx.rdi() as i32).map_or_else(|e| e, |_| 0)
}

/// sceHttpAuthCacheFlush / sceHttpCookieFlush(libhttpCtxId): nothing is
/// stored offline; BEFORE_INIT without any libhttp context.
fn cb_http_store_flush(ctx: &mut CpuContext) -> HleResult {
    if !http().initialized() {
        return HTTP_ERROR_BEFORE_INIT;
    }
    http_context(ctx.rdi() as i32).map_or_else(|e| e, |_| 0)
}

/// sceHttpsLoadCert(libhttpCtxId, caCertNum, caList, cert, privKey): no
/// TLS session ever runs offline, so the certificates are not kept.
fn cb_https_load_cert(ctx: &mut CpuContext) -> HleResult {
    if let Err(e) = http_context(ctx.rdi() as i32) {
        return e;
    }
    log::debug!("[http] sceHttpsLoadCert: {} CA certificate(s), client certificate {}", ctx.rsi() as i32, ctx.rcx() != 0);
    0
}

/// sceHttpsUnloadCert(libhttpCtxId)
fn cb_https_unload_cert(ctx: &mut CpuContext) -> HleResult {
    http_context(ctx.rdi() as i32).map_or_else(|e| e, |_| 0)
}

/// SceHttpsCaList: certs (void**), certsNum (int), padding.
const HTTPS_CA_LIST_SIZE: usize = 16;

/// sceHttpsGetCaList / sceHttpsFreeCaList(libhttpCtxId, SceHttpsCaList*):
/// the library keeps no certificates, so the list is empty.
fn cb_https_ca_list(ctx: &mut CpuContext) -> HleResult {
    let list = ctx.rsi();
    if list == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    if let Err(e) = http_context(ctx.rdi() as i32) {
        return e;
    }
    if ctx.write_bytes(list, &[0u8; HTTPS_CA_LIST_SIZE]) { 0 } else { ERR_MEMORY_FAULT }
}

/// sceHttpSendRequest(reqId, postData, size): the console is offline, so
/// the request times out; the result is kept for the response getters.
fn cb_http_send_request(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let mut st = http();
    let Some(r) = st.request(id) else {
        return HTTP_ERROR_INVALID_ID;
    };
    r.send_attempted = true;
    r.send_result = HTTP_ERROR_TIMEOUT;
    r.status_code = 0;
    log::info!("[http] send request {id:#x} {} {} timed out (offline)", r.method, r.url);
    HTTP_ERROR_TIMEOUT
}

/// sceHttpAbortRequest(reqId): nothing is ever in flight.
fn cb_http_abort_request(ctx: &mut CpuContext) -> HleResult {
    if http().request(ctx.rdi() as i32).is_some() { 0 } else { HTTP_ERROR_INVALID_ID }
}

/// Guest address of an empty string, handed out as the response headers.
fn empty_guest_string(ctx: &CpuContext) -> u64 {
    static EMPTY: AtomicU64 = AtomicU64::new(0);
    let a = EMPTY.load(Ordering::Acquire);
    if a != 0 {
        return a;
    }
    let a = crate::pthread::alloc_guest_block(ctx, 16).unwrap_or(0);
    match EMPTY.compare_exchange(0, a, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => a,
        Err(existing) => existing,
    }
}

/// Send result and status code of a request.
fn response(id: i32) -> Result<(HleResult, i32), HleResult> {
    http().request(id).map(|r| (r.result(), r.status_code)).ok_or(HTTP_ERROR_INVALID_ID)
}

/// sceHttpGetStatusCode(reqId, int* statusCode) -> the send result.
fn cb_http_get_status_code(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    if !ctx.write_u32(out, 0) {
        return ERR_MEMORY_FAULT;
    }
    match response(ctx.rdi() as i32) {
        Ok((result, status)) => {
            ctx.write_u32(out, status as u32);
            result
        }
        Err(e) => e,
    }
}

/// sceHttpGetAllResponseHeaders(reqId, char** header, size_t* headerSize)
fn cb_http_get_all_response_headers(ctx: &mut CpuContext) -> HleResult {
    let (header, size) = (ctx.rsi(), ctx.rdx());
    if header == 0 || size == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    if !ctx.write_u64(header, 0) || !ctx.write_u64(size, 0) {
        return ERR_MEMORY_FAULT;
    }
    match response(ctx.rdi() as i32) {
        Ok((result, _)) => {
            ctx.write_u64(header, empty_guest_string(ctx));
            result
        }
        Err(e) => e,
    }
}

/// sceHttpGetResponseContentLength(reqId, int* result, uint64_t* length)
fn cb_http_get_response_content_length(ctx: &mut CpuContext) -> HleResult {
    let (result, length) = (ctx.rsi(), ctx.rdx());
    if result == 0 || length == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    if !ctx.write_u32(result, 0) || !ctx.write_u64(length, 0) {
        return ERR_MEMORY_FAULT;
    }
    match response(ctx.rdi() as i32) {
        Ok((r, _)) => {
            ctx.write_u32(result, if r == 0 { 0 } else { -1i32 as u32 });
            r
        }
        Err(e) => e,
    }
}

/// sceHttpReadData(reqId, data, size): there is never a response body.
fn cb_http_read_data(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 && ctx.rdx() != 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    match response(ctx.rdi() as i32) {
        Ok((result, _)) => result,
        Err(e) => e,
    }
}

/// sceHttpGetLastErrno(reqId, int* errNum): ETIMEDOUT after a send.
fn cb_http_get_last_errno(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    let errno = match http().request(ctx.rdi() as i32) {
        None => return HTTP_ERROR_INVALID_ID,
        Some(r) if r.send_attempted => HTTP_ERRNO_TIMEDOUT,
        Some(_) => 0,
    };
    if ctx.write_u32(out, errno as u32) { 0 } else { ERR_MEMORY_FAULT }
}

/// sceHttpCreateEpoll(libhttpCtxId, SceHttpEpollHandle* eh)
fn cb_http_create_epoll(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    let mut st = http();
    if !st.initialized() {
        return HTTP_ERROR_BEFORE_INIT;
    }
    let Some(context) = st.context_slot(ctx.rdi() as i32) else {
        return HTTP_ERROR_INVALID_ID;
    };
    if out == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    let handle = match st.free_epoll_blocks.pop() {
        Some(h) => h,
        None => match crate::pthread::alloc_guest_block(ctx, 16) {
            Some(h) => h,
            None => return HTTP_ERROR_OUT_OF_MEMORY,
        },
    };
    if !ctx.write_u64(out, handle) {
        st.free_epoll_blocks.push(handle);
        return ERR_MEMORY_FAULT;
    }
    st.epolls.insert(handle, HttpEpoll { context, aborted: false });
    0
}

/// sceHttpDestroyEpoll(libhttpCtxId, eh): detaches its requests and wakes
/// waiters, whose wait then ends without events.
fn cb_http_destroy_epoll(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rsi();
    let mut st = http();
    if !st.initialized() {
        return HTTP_ERROR_BEFORE_INIT;
    }
    let Some(context) = st.context_slot(ctx.rdi() as i32) else {
        return HTTP_ERROR_INVALID_ID;
    };
    if handle == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    if st.epolls.get(&handle).map(|e| e.context) != Some(context) {
        return HTTP_ERROR_INVALID_ID;
    }
    st.epolls.remove(&handle);
    HTTP_EPOLL_CHANGED.notify_all();
    st.free_epoll_blocks.push(handle);
    st.object_epolls.retain(|_, (h, _)| *h != handle);
    0
}

/// sceHttpSetEpoll(id, eh, userArg)
/// on a template, connection or request: BEFORE_INIT, then a handle that
/// is not a live epoll or an unknown object INVALID_ID.
fn cb_http_set_epoll(ctx: &mut CpuContext) -> HleResult {
    let (id, handle, arg) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    let mut st = http();
    if !st.initialized() {
        return HTTP_ERROR_BEFORE_INIT;
    }
    if !st.epolls.contains_key(&handle) || st.base(id, true).is_none() {
        return HTTP_ERROR_INVALID_ID;
    }
    st.object_epolls.insert(id, (handle, arg));
    0
}

/// sceHttpUnsetEpoll(id)
fn cb_http_unset_epoll(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let mut st = http();
    if !st.initialized() {
        return HTTP_ERROR_BEFORE_INIT;
    }
    if st.base(id, true).is_none() {
        return HTTP_ERROR_INVALID_ID;
    }
    st.object_epolls.remove(&id);
    0
}

/// sceHttpGetEpoll(id, SceHttpEpollHandle* eh, void** userArg): the handle
/// sceHttpSetEpoll gave the object (0 when none).
fn cb_http_get_epoll(ctx: &mut CpuContext) -> HleResult {
    let (id, out, arg_out) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    let mut st = http();
    if !st.initialized() {
        return HTTP_ERROR_BEFORE_INIT;
    }
    if out == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    if st.base(id, true).is_none() {
        return HTTP_ERROR_INVALID_ID;
    }
    let (handle, arg) = st.object_epolls.get(&id).copied().unwrap_or((0, 0));
    if !ctx.write_u64(out, handle) || (arg_out != 0 && !ctx.write_u64(arg_out, arg)) {
        return ERR_MEMORY_FAULT;
    }
    0
}

/// sceHttpWaitRequest(eh, SceHttpNBEvent* nbev, maxevents, timeout_us):
/// sends complete synchronously, so no event is ever pending. The wait
/// lasts until the timeout (a negative one never runs out), until the
/// handle is destroyed (0, no events) or until sceHttpAbortWaitRequest
/// (ABORTED, also for every later wait on the handle).
fn cb_http_wait_request(ctx: &mut CpuContext) -> HleResult {
    let (handle, out, max_events, timeout) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as i32, ctx.rcx() as i32);
    let st = http();
    if !st.initialized() {
        return HTTP_ERROR_BEFORE_INIT;
    }
    if handle == 0 || out == 0 || max_events <= 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    if !st.epolls.contains_key(&handle) {
        return HTTP_ERROR_INVALID_ID;
    }
    let deadline = (timeout >= 0).then(|| std::time::Instant::now() + std::time::Duration::from_micros(timeout as u64));
    let woken = |st: &HttpState| st.epolls.get(&handle).is_none_or(|e| e.aborted);
    let (st, _) = crate::signals::wait_while(&HTTP, &HTTP_EPOLL_CHANGED, st, deadline, woken);
    if st.epolls.get(&handle).is_some_and(|e| e.aborted) { HTTP_ERROR_ABORTED } else { 0 }
}

/// sceHttpAbortWaitRequest(eh): wakes the waits on the handle with ABORTED.
fn cb_http_abort_wait_request(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    let mut st = http();
    if !st.initialized() {
        return HTTP_ERROR_BEFORE_INIT;
    }
    if handle == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    let Some(epoll) = st.epolls.get_mut(&handle) else {
        return HTTP_ERROR_INVALID_ID;
    };
    epoll.aborted = true;
    HTTP_EPOLL_CHANGED.notify_all();
    0
}

// ---------------------------------------------------------------------------
// sceHttpUriEscape / sceHttpUriBuild / sceHttpUriParse
// ---------------------------------------------------------------------------

fn uri_unreserved(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.' | b'~')
}

/// Percent-encode everything outside the RFC 3986 unreserved set.
pub(crate) fn uri_escape(src: &[u8]) -> Vec<u8> {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = Vec::with_capacity(src.len());
    for &c in src {
        if uri_unreserved(c) {
            out.push(c);
        } else {
            out.extend_from_slice(&[b'%', HEX[(c >> 4) as usize], HEX[(c & 15) as usize]]);
        }
    }
    out
}

/// sceHttpUriEscape(char* out, size_t* require, size_t prepare, const char* in)
fn cb_http_uri_escape(ctx: &mut CpuContext) -> HleResult {
    let (out, require, prepare, input) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    if input == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    let Some(src) = ctx.read_cstr_bytes(input, MAX_STRING) else {
        return ERR_MEMORY_FAULT;
    };
    let mut escaped = uri_escape(&src);
    escaped.push(0);
    let needed = escaped.len() as u64;
    if require != 0 && !ctx.write_u64(require, needed) {
        return ERR_MEMORY_FAULT;
    }
    if out == 0 {
        return 0;
    }
    if prepare < needed {
        return HTTP_ERROR_OUT_OF_MEMORY;
    }
    if ctx.write_bytes(out, &escaped) { 0 } else { ERR_MEMORY_FAULT }
}

const URI_BUILD_WITH_SCHEME: u32 = 0x01;
const URI_BUILD_WITH_HOSTNAME: u32 = 0x02;
const URI_BUILD_WITH_PORT: u32 = 0x04;
const URI_BUILD_WITH_PATH: u32 = 0x08;
const URI_BUILD_WITH_USERNAME: u32 = 0x10;
const URI_BUILD_WITH_PASSWORD: u32 = 0x20;
const URI_BUILD_WITH_QUERY: u32 = 0x40;
const URI_BUILD_WITH_FRAGMENT: u32 = 0x80;
const URI_BUILD_WITH_ALL: u32 = 0xFF;

/// SceHttpUriElement: opaque (int), scheme, username, password, hostname,
/// path, query, fragment (char*), port (u16) @64, reserved[10]; 80 bytes.
const URI_ELEMENT_SIZE: usize = 80;
const URI_ELEMENT_PORT: usize = 64;

/// The fields of a guest SceHttpUriElement.
#[derive(Default)]
pub(crate) struct UriElement {
    pub opaque: bool,
    /// scheme, username, password, hostname, path, query, fragment.
    pub parts: [Option<Vec<u8>>; 7],
    pub port: u16,
}

/// Assemble a URI from `e` with the requested components (no component
/// bits = all of them).
pub(crate) fn uri_build(e: &UriElement, option: u32) -> Vec<u8> {
    let parts = if option & URI_BUILD_WITH_ALL == 0 { URI_BUILD_WITH_ALL } else { option };
    let [scheme, username, password, hostname, path, query, fragment] = &e.parts;
    let hierarchical = !e.opaque && hostname.is_some();
    let mut uri = Vec::new();
    if parts & URI_BUILD_WITH_SCHEME != 0 {
        if let Some(s) = scheme {
            uri.extend_from_slice(s);
            uri.push(b':');
        }
        // Scheme-only builds include the authority marker.
        if hierarchical {
            uri.extend_from_slice(b"//");
        }
    }
    let user = username.as_ref().filter(|_| parts & URI_BUILD_WITH_USERNAME != 0);
    let pass = password.as_ref().filter(|_| parts & URI_BUILD_WITH_PASSWORD != 0);
    if let Some(u) = user {
        uri.extend_from_slice(u);
    }
    if let Some(p) = pass {
        if user.is_some() {
            uri.push(b':');
        }
        uri.extend_from_slice(p);
    }
    let with_host = parts & URI_BUILD_WITH_HOSTNAME != 0 && hierarchical;
    if with_host {
        if user.is_some() || pass.is_some() {
            uri.push(b'@');
        }
        uri.extend_from_slice(hostname.as_deref().unwrap_or_default());
    }
    if parts & URI_BUILD_WITH_PORT != 0 && hierarchical && e.port != 0 {
        if with_host {
            uri.push(b':');
        }
        uri.extend_from_slice(e.port.to_string().as_bytes());
    }
    for (bit, part) in [(URI_BUILD_WITH_PATH, path), (URI_BUILD_WITH_QUERY, query), (URI_BUILD_WITH_FRAGMENT, fragment)] {
        if parts & bit != 0 {
            if let Some(p) = part {
                uri.extend_from_slice(p);
            }
        }
    }
    uri
}

/// sceHttpUriBuild(char* out, size_t* require, size_t prepare,
/// const SceHttpUriElement* srcElement, uint32_t option). A zero `prepare`
/// does not limit the output.
fn cb_http_uri_build(ctx: &mut CpuContext) -> HleResult {
    let (out, require, prepare, src, option) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8() as u32);
    if src == 0 || (out == 0 && require == 0) {
        return HTTP_ERROR_INVALID_VALUE;
    }
    let Some(raw) = ctx.read_bytes(src, URI_ELEMENT_SIZE) else {
        return ERR_MEMORY_FAULT;
    };
    let mut e = UriElement {
        opaque: i32::from_le_bytes(raw[0..4].try_into().unwrap()) != 0,
        port: u16::from_le_bytes([raw[URI_ELEMENT_PORT], raw[URI_ELEMENT_PORT + 1]]),
        ..Default::default()
    };
    for i in 0..7 {
        let p = u64::from_le_bytes(raw[8 + i * 8..16 + i * 8].try_into().unwrap());
        if p != 0 {
            match ctx.read_cstr_bytes(p, MAX_STRING) {
                Some(b) => e.parts[i] = Some(b),
                None => return ERR_MEMORY_FAULT,
            }
        }
    }
    let mut uri = uri_build(&e, option);
    uri.push(0);
    let needed = uri.len() as u64;
    if require != 0 && !ctx.write_u64(require, needed) {
        return ERR_MEMORY_FAULT;
    }
    if out != 0 {
        if prepare != 0 && prepare < needed {
            return HTTP_ERROR_OUT_OF_MEMORY;
        }
        if !ctx.write_bytes(out, &uri) {
            return ERR_MEMORY_FAULT;
        }
    }
    0
}

/// Parsed URI pieces: scheme, username, password, hostname, path, query,
/// fragment (None = absent), plus opacity and port.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct SplitUri {
    pub components: [Option<Vec<u8>>; 7],
    pub opaque: bool,
    pub port: u16,
}

impl SplitUri {
    /// Pool bytes the parsed element needs: every present component plus
    /// its terminator, and an empty query when there is none.
    pub fn required(&self) -> usize {
        let parts: usize = self.components.iter().flatten().map(|c| c.len() + 1).sum();
        parts + self.components[5].is_none() as usize
    }
}

fn is_scheme_char(c: u8, first: bool) -> bool {
    if first { c.is_ascii_alphabetic() } else { c.is_ascii_alphanumeric() || matches!(c, b'+' | b'-' | b'.') }
}

/// Split a URI the way the console library does. None = invalid URL. The
/// empty URI is opaque with empty scheme, hostname, path and query.
pub(crate) fn split_uri(src: &[u8]) -> Option<SplitUri> {
    let mut out = SplitUri::default();
    if src.is_empty() {
        out.opaque = true;
        for i in [0, 3, 4, 5] {
            out.components[i] = Some(Vec::new());
        }
        return Some(out);
    }
    if !is_scheme_char(src[0], true) {
        return None;
    }
    let mut pos = 0;
    while pos < src.len() && src[pos] != b':' {
        if !is_scheme_char(src[pos], false) {
            return None;
        }
        pos += 1;
    }
    if pos == src.len() {
        return None;
    }
    out.components[0] = Some(src[..pos].to_vec());
    pos += 1;
    if src[pos..].starts_with(b"//") {
        pos += 2;
        let authority_start = pos;
        let mut authority_end = pos;
        while authority_end < src.len() && !matches!(src[authority_end], b'/' | b'?' | b'#') {
            authority_end += 1;
        }
        let authority = &src[authority_start..authority_end];
        let mut host_begin = 0;
        for (i, &c) in authority.iter().enumerate() {
            if c == b'@' {
                let credentials = &authority[..i];
                match credentials.iter().position(|&c| c == b':') {
                    Some(colon) => {
                        out.components[1] = Some(credentials[..colon].to_vec());
                        out.components[2] = Some(credentials[colon + 1..].to_vec());
                    }
                    None => {
                        out.components[1] = Some(credentials.to_vec());
                        out.components[2] = None;
                    }
                }
                host_begin = i + 1;
            }
        }
        let mut host_end = authority.len();
        let mut port_start = None;
        if authority.get(host_begin) == Some(&b'[') {
            let close = authority[host_begin..].iter().position(|&c| c == b']')? + host_begin;
            host_end = close + 1;
            if authority.get(host_end) == Some(&b':') {
                port_start = Some(host_end + 1);
            }
        } else if let Some(colon) = authority[host_begin..].iter().position(|&c| c == b':') {
            host_end = host_begin + colon;
            port_start = Some(host_end + 1);
        }
        if host_begin < host_end {
            out.components[3] = Some(authority[host_begin..host_end].to_vec());
        }
        if let Some(ps) = port_start {
            if ps == authority.len() {
                return None;
            }
            let mut value: u32 = 0;
            for &c in &authority[ps..] {
                if !c.is_ascii_digit() {
                    return None;
                }
                value = value * 10 + (c - b'0') as u32;
                if value > u16::MAX as u32 {
                    return None;
                }
            }
            out.port = value as u16;
        }
        pos = authority_end;
    } else {
        out.opaque = true;
    }
    let path_start = pos;
    while pos < src.len() && !matches!(src[pos], b'?' | b'#') {
        pos += 1;
    }
    if pos > path_start {
        out.components[4] = Some(src[path_start..pos].to_vec());
    }
    if pos < src.len() && src[pos] == b'?' {
        let query_start = pos;
        pos += 1;
        while pos < src.len() && src[pos] != b'#' {
            pos += 1;
        }
        out.components[5] = Some(src[query_start..pos].to_vec());
    }
    if pos < src.len() {
        out.components[6] = Some(src[pos..].to_vec());
    }
    Some(out)
}

/// Pool image and element pointers of a parsed URI placed at `pool`: the
/// components in order, an absent query becoming an empty string.
fn uri_pool_image(split: &SplitUri, pool: u64) -> (Vec<u8>, [u64; 7]) {
    let mut bytes = Vec::with_capacity(split.required());
    let mut ptrs = [0u64; 7];
    for (i, c) in split.components.iter().enumerate() {
        let part: Option<&[u8]> = match c {
            Some(c) => Some(c),
            None if i == 5 => Some(&[]),
            None => None,
        };
        if let Some(part) = part {
            ptrs[i] = pool + bytes.len() as u64;
            bytes.extend_from_slice(part);
            bytes.push(0);
        }
    }
    (bytes, ptrs)
}

/// sceHttpUriParse(SceHttpUriElement* out, const char* srcUri, void* pool,
/// size_t* require, size_t prepare). A zero `prepare` does not limit the
/// pool.
fn cb_http_uri_parse(ctx: &mut CpuContext) -> HleResult {
    let (out_addr, src_addr, pool, require_addr, prepare) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8());
    if src_addr == 0 {
        return HTTP_ERROR_INVALID_URL;
    }
    if out_addr == 0 && pool == 0 && require_addr == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    let Some(src) = ctx.read_cstr_bytes(src_addr, MAX_STRING) else {
        return ERR_MEMORY_FAULT;
    };
    let Some(split) = split_uri(&src) else {
        return HTTP_ERROR_INVALID_URL;
    };
    let required = split.required() as u64;
    if require_addr != 0 && !ctx.write_u64(require_addr, required) {
        return ERR_MEMORY_FAULT;
    }
    if out_addr == 0 {
        return 0;
    }
    let mut element = [0u8; URI_ELEMENT_SIZE];
    element[0..4].copy_from_slice(&(split.opaque as i32).to_le_bytes());
    element[URI_ELEMENT_PORT..URI_ELEMENT_PORT + 2].copy_from_slice(&split.port.to_le_bytes());
    if !ctx.write_bytes(out_addr, &element) {
        return ERR_MEMORY_FAULT;
    }
    if pool == 0 {
        return 0;
    }
    if prepare != 0 && prepare < required {
        return HTTP_ERROR_OUT_OF_MEMORY;
    }
    let (bytes, ptrs) = uri_pool_image(&split, pool);
    for (i, p) in ptrs.iter().enumerate() {
        element[8 + i * 8..16 + i * 8].copy_from_slice(&p.to_le_bytes());
    }
    if !ctx.write_bytes(pool, &bytes) || !ctx.write_bytes(out_addr, &element) {
        return ERR_MEMORY_FAULT;
    }
    0
}

// ---------------------------------------------------------------------------
// sceHttpParseResponseHeader / sceHttpParseStatusLine
// ---------------------------------------------------------------------------

/// Longest field name sceHttpParseResponseHeader compares.
const HEADER_FIELD_MAX: usize = 0xFFF;

fn header_space(c: u8) -> bool {
    c == b' ' || (b'\t'..=b'\r').contains(&c)
}

/// Where the value of `field` lies in a response header block, as the
/// console finds it: Some(Some((start, length))) with the value ending at
/// start + length, Some(None) for an empty value, None when the field is
/// absent. Field names compare without case; leading white space (and a
/// line break followed by a continuation line) is skipped; continuation
/// lines (starting with a space or tab) belong to the value; a value whose
/// line break ends the block keeps that line break.
pub(crate) fn find_header_field(input: &[u8], field: &[u8]) -> Option<Option<(usize, usize)>> {
    let n = input.len();
    let fl = field.len();
    let mut start = 0;
    let mut found = false;
    while start < n {
        if !header_space(input[start])
            && fl < n - start
            && input[start + fl] == b':'
            && input[start..start + fl].eq_ignore_ascii_case(field)
        {
            start += fl + 1;
            found = true;
            break;
        }
        match input[start..].iter().position(|&c| c == b'\n') {
            Some(p) => start += p + 1,
            None => break,
        }
    }
    if !found {
        return None;
    }
    let continues = |next: usize| next < n && (input[next] == b' ' || input[next] == b'\t');
    while start < n {
        let line_feed = if input[start] == b'\r' && start + 1 < n && input[start + 1] == b'\n' { start + 1 } else { start };
        if input[line_feed] == b'\n' {
            if !continues(line_feed + 1) {
                return Some(None);
            }
            start = line_feed + 1;
        } else if header_space(input[start]) {
            start += 1;
        } else {
            break;
        }
    }
    if start == n {
        return Some(None);
    }
    let mut end = n;
    let mut scan = start;
    while scan < n {
        let Some(p) = input[scan..].iter().position(|&c| c == b'\n') else {
            break;
        };
        let newline = scan + p;
        if newline + 1 == n {
            break;
        }
        if continues(newline + 1) {
            scan = newline + 1;
            continue;
        }
        end = if newline > start && input[newline - 1] == b'\r' { newline - 1 } else { newline };
        break;
    }
    Some(Some((start, end - start)))
}

/// sceHttpParseResponseHeader(header, headerLen, fieldStr, const char**
/// fieldValue, size_t* valueLen) -> the offset where the value ends (as
/// measured on a PS5); an empty value is NULL with length 0 and returns 0.
fn cb_http_parse_response_header(ctx: &mut CpuContext) -> HleResult {
    let (header, len, field, value_out, len_out) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8());
    if header == 0 {
        return HTTP_ERROR_PARSE_INVALID_RESPONSE;
    }
    if field == 0 || value_out == 0 || len_out == 0 {
        return HTTP_ERROR_PARSE_INVALID_VALUE;
    }
    if len > i32::MAX as u64 {
        log::warn!("[http] sceHttpParseResponseHeader: header length {len:#x} does not fit the int result");
        return HTTP_ERROR_PARSE_INVALID_VALUE;
    }
    let Some(field) = ctx.read_cstr_bytes(field, HEADER_FIELD_MAX) else {
        return ERR_MEMORY_FAULT;
    };
    let input = if len == 0 {
        Vec::new()
    } else {
        match ctx.read_bytes(header, len as usize) {
            Some(b) => b,
            None => return ERR_MEMORY_FAULT,
        }
    };
    let Some(found) = find_header_field(&input, &field) else {
        return HTTP_ERROR_PARSE_NOT_FOUND;
    };
    let (start, length) = found.unwrap_or((0, 0));
    let consumed = if found.is_some() { start + length } else { 0 };
    let value = if found.is_some() { header + start as u64 } else { 0 };
    if !ctx.write_u64(value_out, value) || !ctx.write_u64(len_out, length as u64) {
        return ERR_MEMORY_FAULT;
    }
    consumed as HleResult
}

/// A parsed status line: version, code, reason phrase offset and length,
/// bytes consumed.
#[derive(Debug, PartialEq)]
pub(crate) struct StatusLine {
    pub major: i32,
    pub minor: i32,
    pub code: i32,
    pub phrase: usize,
    pub phrase_len: usize,
    pub consumed: usize,
}

/// Parse "HTTP/<major>.<minor> <3 digits><phrase>\n". On failure, the
/// version numbers parsed so far come back with the error.
pub(crate) fn parse_status_line(line: &[u8]) -> Result<StatusLine, (i32, i32)> {
    let (mut major, mut minor) = (0i32, 0i32);
    if line.len() < 8 || !line.starts_with(b"HTTP/") {
        return Err((major, minor));
    }
    let number = |i: &mut usize| -> Option<i32> {
        if *i >= line.len() || !line[*i].is_ascii_digit() {
            return None;
        }
        let mut v: u32 = 0;
        while *i < line.len() && line[*i].is_ascii_digit() {
            v = v.wrapping_mul(10).wrapping_add((line[*i] - b'0') as u32);
            *i += 1;
        }
        Some(v as i32)
    };
    let mut i = 5;
    major = number(&mut i).ok_or((major, minor))?;
    if i >= line.len() || line[i] != b'.' {
        return Err((major, minor));
    }
    i += 1;
    minor = number(&mut i).ok_or((major, minor))?;
    if i >= line.len() || line[i] != b' ' {
        return Err((major, minor));
    }
    i += 1;
    if line.len() - i < 3 || !line[i..i + 3].iter().all(u8::is_ascii_digit) {
        return Err((major, minor));
    }
    let code = line[i..i + 3].iter().fold(0, |a, &d| a * 10 + (d - b'0') as i32);
    i += 3;
    let Some(lf) = line[i..].iter().position(|&c| c == b'\n') else {
        return Err((major, minor));
    };
    let consumed = i + lf + 1;
    let mut phrase_len = lf;
    if phrase_len > 0 && line[i + phrase_len - 1] == b'\r' {
        phrase_len -= 1;
    }
    Ok(StatusLine { major, minor, code, phrase: i, phrase_len, consumed })
}

/// sceHttpParseStatusLine(statusLine, lineLen, int* major, int* minor,
/// int* code, const char** reasonPhrase, size_t* phraseLen) -> bytes
/// consumed. The version outputs are cleared first and keep what was parsed
/// when the line is invalid.
fn cb_http_parse_status_line(ctx: &mut CpuContext) -> HleResult {
    let (line, len, major, minor, code) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8());
    let phrase = ctx.r9();
    let Some(phrase_len) = ctx.read_u64(ctx.get(crate::context::CpuRegister::Rsp) + 8) else {
        return ERR_MEMORY_FAULT;
    };
    if line == 0 {
        return HTTP_ERROR_PARSE_INVALID_RESPONSE;
    }
    if major == 0 || minor == 0 || code == 0 || phrase == 0 || phrase_len == 0 {
        return HTTP_ERROR_PARSE_INVALID_VALUE;
    }
    if len > i32::MAX as u64 {
        log::warn!("[http] sceHttpParseStatusLine: line length {len:#x} does not fit the int result");
        return HTTP_ERROR_PARSE_INVALID_VALUE;
    }
    let bytes = if len == 0 { Some(Vec::new()) } else { ctx.read_bytes(line, len as usize) };
    let Some(bytes) = bytes else {
        return ERR_MEMORY_FAULT;
    };
    match parse_status_line(&bytes) {
        Ok(s) => {
            let ok = ctx.write_u32(major, s.major as u32)
                && ctx.write_u32(minor, s.minor as u32)
                && ctx.write_u32(code, s.code as u32)
                && ctx.write_u64(phrase, line + s.phrase as u64)
                && ctx.write_u64(phrase_len, s.phrase_len as u64);
            if ok { s.consumed as HleResult } else { ERR_MEMORY_FAULT }
        }
        Err((ma, mi)) => {
            if !ctx.write_u32(major, ma as u32) || !ctx.write_u32(minor, mi as u32) {
                return ERR_MEMORY_FAULT;
            }
            HTTP_ERROR_PARSE_INVALID_RESPONSE
        }
    }
}

// ---------------------------------------------------------------------------
// sceHttpUriUnescape / sceHttpUriSweepPath / sceHttpUriMerge
// ---------------------------------------------------------------------------

/// Decode %XX escapes; a '%' without two hex digits after it is
/// INVALID_URL (measured on a PS5).
pub(crate) fn uri_unescape(input: &[u8]) -> Result<Vec<u8>, HleResult> {
    let hex = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
    let mut out = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        if input[i] != b'%' {
            out.push(input[i]);
            i += 1;
            continue;
        }
        if input.len() - i < 3 {
            return Err(HTTP_ERROR_INVALID_URL);
        }
        let (Some(h), Some(l)) = (hex(input[i + 1]), hex(input[i + 2])) else {
            return Err(HTTP_ERROR_INVALID_URL);
        };
        out.push((h << 4) | l);
        i += 3;
    }
    Ok(out)
}

/// Write `value` plus its terminator to `out` (size `prepare`), storing the
/// size needed in `*require`: the libSceHttp copy-out convention.
fn copy_out(ctx: &CpuContext, value: &[u8], out: u64, require: u64, prepare: u64) -> HleResult {
    let needed = value.len() as u64 + 1;
    if require != 0 && !ctx.write_u64(require, needed) {
        return ERR_MEMORY_FAULT;
    }
    if out == 0 {
        return 0;
    }
    if prepare < needed {
        return HTTP_ERROR_OUT_OF_MEMORY;
    }
    let mut bytes = value.to_vec();
    bytes.push(0);
    if ctx.write_bytes(out, &bytes) { 0 } else { ERR_MEMORY_FAULT }
}

/// sceHttpUriUnescape(char* out, size_t* require, size_t prepare, const char* in)
fn cb_http_uri_unescape(ctx: &mut CpuContext) -> HleResult {
    let (out, require, prepare, input) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    if input == 0 || (out == 0 && require == 0) {
        return HTTP_ERROR_INVALID_VALUE;
    }
    let Some(src) = ctx.read_cstr_bytes(input, MAX_STRING) else {
        return ERR_MEMORY_FAULT;
    };
    let decoded = match uri_unescape(&src) {
        Ok(d) => d,
        Err(e) => return e,
    };
    // A short buffer gets what fits, without a terminator, and the size
    // needed is not reported.
    if out != 0 && prepare < decoded.len() as u64 + 1 {
        let n = (prepare as usize).min(decoded.len());
        return if ctx.write_bytes(out, &decoded[..n]) { HTTP_ERROR_OUT_OF_MEMORY } else { ERR_MEMORY_FAULT };
    }
    copy_out(ctx, &decoded, out, require, prepare)
}

/// Remove "./" and "dir/../" segments from the first `length` bytes of an
/// absolute path into `dst` (at least `length + 1` and 2 bytes). A path not
/// starting with '/' is copied unchanged; a trailing "." stays. As on the
/// console, a ".." test may look at the byte after the `length` bytes
/// (`src` holds it when there is one).
pub(crate) fn uri_sweep_path(dst: &mut [u8], src: &[u8], length: usize) {
    let at = |i: usize| src.get(i).copied().unwrap_or(0);
    if at(0) != b'/' {
        dst[..length].copy_from_slice(src);
        dst[length] = 0;
        return;
    }
    dst[0] = b'/';
    dst[1] = 0;
    let mut end = 0;
    let mut pos = 1;
    while pos < length {
        if src[pos] == b'.' && pos + 1 < length && src[pos + 1] == b'/' {
            pos += 2;
            continue;
        }
        if src[pos] == b'.' && pos + 2 <= length && src[pos + 1] == b'.' && at(pos + 2) == b'/' {
            if end != 0 {
                dst[end] = 0;
                let nul = dst.iter().position(|&c| c == 0).unwrap_or(dst.len());
                end = dst[..nul].iter().rposition(|&c| c == b'/').unwrap_or(0);
                dst[end + 1] = 0;
            }
            pos += 3;
            continue;
        }
        let count = src[pos..length].iter().position(|&c| c == b'/').map_or(length - pos, |p| p + 1);
        dst[end + 1..end + 1 + count].copy_from_slice(&src[pos..pos + count]);
        dst[end + 1 + count] = 0;
        end += count;
        pos += count;
    }
}

/// sceHttpUriSweepPath(char* dst, const char* src, size_t srcSize): the
/// first srcSize - 1 bytes of `src` are swept into `dst`.
fn cb_http_uri_sweep_path(ctx: &mut CpuContext) -> HleResult {
    let (dst, src, size) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if size == 0 {
        return 0;
    }
    if dst == 0 || src == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    if size > MAX_STRING as u64 {
        log::warn!("[http] sceHttpUriSweepPath: source size {size:#x} exceeds {MAX_STRING:#x}");
        return HTTP_ERROR_INVALID_VALUE;
    }
    let length = size as usize - 1;
    // The byte after the source may be looked at; it may not exist.
    let Some(path) = ctx.read_bytes(src, length + 1).or_else(|| ctx.read_bytes(src, length)) else {
        return ERR_MEMORY_FAULT;
    };
    let Some(mut out) = ctx.read_bytes(dst, (size as usize).max(2)) else {
        return ERR_MEMORY_FAULT;
    };
    uri_sweep_path(&mut out, &path, length);
    if ctx.write_bytes(dst, &out) { 0 } else { ERR_MEMORY_FAULT }
}

/// Longest URL or relative reference sceHttpUriMerge measures.
const URI_MAX_LENGTH: usize = 0x3FFF;

/// URI pieces as the merge reads them (absent pieces are empty).
#[derive(Default)]
struct MergeUri {
    opaque: bool,
    scheme: Vec<u8>,
    username: Vec<u8>,
    password: Vec<u8>,
    hostname: Vec<u8>,
    path: Vec<u8>,
    query: Vec<u8>,
    fragment: Vec<u8>,
    port: u16,
}

impl MergeUri {
    /// Pool bytes the parsed element takes: every piece plus a terminator.
    fn pool_size(&self) -> usize {
        [&self.scheme, &self.username, &self.password, &self.hostname, &self.path, &self.query, &self.fragment]
            .iter()
            .map(|p| p.len() + 1)
            .sum()
    }
}

fn default_port(scheme: &[u8]) -> u16 {
    if scheme.eq_ignore_ascii_case(b"http") {
        80
    } else if scheme.eq_ignore_ascii_case(b"https") {
        443
    } else {
        0
    }
}

/// Length of the scheme before ':' (0 when there is none).
fn scheme_length(uri: &[u8]) -> usize {
    if uri.is_empty() || !uri[0].is_ascii_alphabetic() {
        return 0;
    }
    let end = 1 + uri[1..].iter().take_while(|&&c| is_scheme_char(c, false)).count();
    if end < uri.len() && uri[end] == b':' { end } else { 0 }
}

fn uri_is_opaque(uri: &[u8]) -> bool {
    let s = scheme_length(uri);
    !uri[if s == 0 { 0 } else { s + 1 }..].starts_with(b"//")
}

/// RFC 3986 remove_dot_segments.
fn remove_dot_segments(mut input: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    while !input.is_empty() {
        if input.starts_with(b"../") {
            input = &input[3..];
        } else if input.starts_with(b"./") || input.starts_with(b"/./") {
            input = &input[2..];
        } else if input == b"/." {
            input = b"/";
        } else if input.starts_with(b"/../") || input == b"/.." {
            input = if input.len() == 3 { b"/" } else { &input[3..] };
            let slash = output.iter().rposition(|&c| c == b'/').unwrap_or(0);
            output.truncate(slash);
        } else if input == b"." || input == b".." {
            input = b"";
        } else {
            let len = input[1..].iter().position(|&c| c == b'/').map_or(input.len(), |p| p + 1);
            output.extend_from_slice(&input[..len]);
            input = &input[len..];
        }
    }
    output
}

/// Parse a URI for sceHttpUriMerge; None = invalid URL (a bad host name
/// character, an unclosed IPv6 literal, a bad or too long port).
fn parse_merge_uri(mut uri: &[u8]) -> Option<MergeUri> {
    let host_char = |c: u8| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'.' | b'_');
    let mut p = MergeUri::default();
    let s = scheme_length(uri);
    if s != 0 {
        p.scheme = uri[..s].to_vec();
        uri = &uri[s + 1..];
    }
    p.opaque = !uri.starts_with(b"//");
    if !p.opaque {
        uri = &uri[2..];
    }
    let auth_len = uri.iter().position(|c| b"/?#".contains(c)).unwrap_or(uri.len());
    let mut authority = &uri[..auth_len];
    uri = &uri[auth_len..];
    if let Some(at) = authority.iter().rposition(|&c| c == b'@') {
        let userinfo = &authority[..at];
        match userinfo.iter().position(|&c| c == b':') {
            Some(colon) => {
                p.username = userinfo[..colon].to_vec();
                p.password = userinfo[colon + 1..].to_vec();
            }
            None => p.username = userinfo.to_vec(),
        }
        authority = &authority[at + 1..];
    }
    let mut port: &[u8] = b"";
    if authority.starts_with(b"[") {
        let close = authority.iter().position(|&c| c == b']')?;
        p.hostname = authority[1..close].to_vec();
        authority = &authority[close + 1..];
        if !authority.is_empty() {
            if authority[0] != b':' {
                return None;
            }
            port = &authority[1..];
        }
        if !p.hostname.iter().all(|&c| host_char(c) || c == b':') {
            return None;
        }
    } else {
        let colon = authority.iter().position(|&c| c == b':');
        p.hostname = authority[..colon.unwrap_or(authority.len())].to_vec();
        if let Some(c) = colon {
            port = &authority[c + 1..];
        }
        if !p.hostname.iter().all(|&c| host_char(c)) {
            return None;
        }
    }
    if port.len() > 5 || !port.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let value = port.iter().fold(0u32, |a, &d| a * 10 + (d - b'0') as u32);
    if value > 0xFFFF {
        return None;
    }
    p.port = if port.is_empty() { default_port(&p.scheme) } else { value as u16 };
    let path_len = uri.iter().position(|c| b"?#".contains(c)).unwrap_or(uri.len());
    p.path = remove_dot_segments(&uri[..path_len]);
    uri = &uri[path_len..];
    if uri.starts_with(b"?") {
        let q = uri.iter().position(|&c| c == b'#').unwrap_or(uri.len());
        p.query = uri[..q].to_vec();
        uri = &uri[q..];
    }
    p.fragment = uri.to_vec();
    Some(p)
}

/// Rebuild a merged URI: scheme, authority (user info, bracketed IPv6
/// host, a port other than the scheme's default) and path.
fn build_merged_uri(p: &MergeUri) -> Vec<u8> {
    let mut uri = Vec::new();
    if !p.scheme.is_empty() {
        uri.extend_from_slice(&p.scheme);
        uri.push(b':');
    }
    if !p.opaque {
        uri.extend_from_slice(b"//");
    }
    uri.extend_from_slice(&p.username);
    if !p.password.is_empty() {
        uri.push(b':');
        uri.extend_from_slice(&p.password);
    }
    if !p.username.is_empty() || !p.password.is_empty() {
        uri.push(b'@');
    }
    if p.hostname.contains(&b':') {
        uri.push(b'[');
        uri.extend_from_slice(&p.hostname);
        uri.push(b']');
    } else {
        uri.extend_from_slice(&p.hostname);
    }
    if p.port != 0 && p.port != default_port(&p.scheme) {
        uri.extend_from_slice(format!(":{}", p.port).as_bytes());
    }
    uri.extend_from_slice(&p.path);
    uri
}

/// Outcome of a merge: the buffer size the call requires and what it writes.
#[derive(Debug, PartialEq)]
pub(crate) enum Merged {
    /// The reference is absolute: copied as is, zero-padded to `size`.
    Absolute { size: usize, text: Vec<u8> },
    /// The reference resolved against the base URL; the output budget is
    /// `prepare - budget_cut`.
    Relative { size: usize, text: Vec<u8>, budget_cut: usize },
}

/// Merge `relative` onto `url` (None = invalid URL). The size required is
/// the base's parse pool plus twice both input lengths plus 2. Dot
/// segments of the reference are kept.
pub(crate) fn uri_merge(url: &[u8], relative: &[u8]) -> Option<Merged> {
    let base = parse_merge_uri(url)?;
    let relative_opaque = uri_is_opaque(relative);
    if !relative_opaque {
        parse_merge_uri(relative)?;
    }
    let url_len = url.len().min(URI_MAX_LENGTH);
    let rel_len = relative.len().min(URI_MAX_LENGTH);
    let size = base.pool_size() + 2 + (url_len + rel_len) * 2;
    if !relative_opaque {
        return Some(Merged::Absolute { size, text: relative[..rel_len].to_vec() });
    }
    let mut path = base.path.clone();
    match path.iter().rposition(|&c| c == b'/') {
        Some(slash) => path.truncate(slash + 1),
        None => path.push(b'/'),
    }
    if relative.first() == Some(&b'/') {
        path.clear();
    }
    let room = URI_MAX_LENGTH - path.len().min(URI_MAX_LENGTH);
    path.extend_from_slice(&relative[..rel_len.min(room)]);
    let merged = MergeUri { path, query: Vec::new(), fragment: Vec::new(), ..base };
    Some(Merged::Relative { size, text: build_merged_uri(&merged), budget_cut: url_len + rel_len + 1 })
}

/// sceHttpUriMerge(char* mergedUrl, const char* url, const char* relativeUri,
/// size_t* require, size_t prepare, uint32_t option): option must be 0.
fn cb_http_uri_merge(ctx: &mut CpuContext) -> HleResult {
    let (out, url, relative, require, prepare, option) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8(), ctx.r9() as u32);
    if option != 0 || url == 0 || relative == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    let (Some(url), Some(relative)) = (ctx.read_cstr_bytes(url, MAX_STRING), ctx.read_cstr_bytes(relative, MAX_STRING)) else {
        return ERR_MEMORY_FAULT;
    };
    let Some(merged) = uri_merge(&url, &relative) else {
        return HTTP_ERROR_INVALID_URL;
    };
    let size = match &merged {
        Merged::Absolute { size, .. } | Merged::Relative { size, .. } => *size as u64,
    };
    if require != 0 && !ctx.write_u64(require, size) {
        return ERR_MEMORY_FAULT;
    }
    if out == 0 {
        return 0;
    }
    if prepare < size {
        return HTTP_ERROR_OUT_OF_MEMORY;
    }
    match merged {
        Merged::Absolute { size, text } => {
            let mut bytes = text.clone();
            bytes.resize(size, 0);
            if let Some(last) = bytes.last_mut() {
                *last = 0;
            }
            if !ctx.write_bytes(out, &bytes) || (require != 0 && !ctx.write_u64(require, text.len() as u64 + 1)) {
                return ERR_MEMORY_FAULT;
            }
            0
        }
        Merged::Relative { text, budget_cut, .. } => copy_out(ctx, &text, out, 0, prepare - budget_cut as u64),
    }
}

// ---------------------------------------------------------------------------
// libSceHttp2
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Clone, Debug, Default, PartialEq)]
struct Http2Options {
    auth_enabled: bool,
    auto_redirect: bool,
    inflate_gzip: bool,
    min_ssl_version: u32,
    ssl_options: u32,
    resolve_timeout_us: u32,
    resolve_retry: i32,
    connect_timeout_us: u32,
    connection_wait_timeout_us: u32,
    send_timeout_us: u32,
    recv_timeout_us: u32,
    timeout_us: u32,
    ssl_callback: u64,
    ssl_callback_arg: u64,
    redirect_callback: u64,
    redirect_user_arg: u64,
    cookie_box_id: i32,
}

#[allow(dead_code)]
struct Http2Context {
    libnet_mem_id: i32,
    libssl_ctx_id: i32,
    pool_size: u64,
    max_concurrent_requests: i32,
}

#[allow(dead_code)]
struct Http2Template {
    context_id: i32,
    user_agent: String,
    http_ver: i32,
    is_auto_proxy_conf: i32,
    options: Http2Options,
}

#[allow(dead_code)]
struct Http2Request {
    tmpl_id: i32,
    method: String,
    url: String,
    content_length: u64,
    headers: Vec<(String, String)>,
    send_result: HleResult,
    status_code: i32,
    response_body: Vec<u8>,
    read_offset: usize,
    async_result: HleResult,
    async_event: i32,
    options: Http2Options,
    /// Set by sceHttp2SetRequestNoContentLength: the request goes out
    /// without a Content-Length header. Requests are never sent offline, so
    /// nothing reads the flag yet.
    no_content_length: bool,
    /// Present on a request made by sceHttp2WebSocketCreateRequest.
    websocket: Option<Http2WebSocket>,
}

impl Http2Request {
    fn new(tmpl_id: i32, method: String, url: String, content_length: u64, websocket: Option<Http2WebSocket>) -> Self {
        Http2Request {
            tmpl_id,
            method,
            url,
            content_length,
            headers: Vec::new(),
            send_result: HTTP2_ERROR_BEFORE_SEND,
            status_code: 0,
            response_body: Vec::new(),
            read_offset: 0,
            async_result: HTTP2_ERROR_BEFORE_SEND,
            async_event: 0,
            options: Http2Options::default(),
            no_content_length: false,
            websocket,
        }
    }
}

/// The WebSocket side of a request. The opening handshake is the request
/// itself (sent with sceHttp2SendRequestAsync); messages flow only once it
/// succeeded, which never happens offline, so the callbacks are never
/// called and the receive buffer is never written.
#[allow(dead_code)]
#[derive(Clone, Debug, Default, PartialEq)]
struct Http2WebSocket {
    text_callback: u64,
    data_callback: u64,
    close_callback: u64,
    error_callback: u64,
    recv_buffer: u64,
    recv_buffer_size: u64,
    user_data: u64,
    /// sceHttp2WebSocketSetPingInterval / SetPingTimeout, microseconds.
    ping_interval_us: u32,
    ping_timeout_us: u32,
}

struct Http2State {
    contexts: BTreeMap<i32, Http2Context>,
    templates: BTreeMap<i32, Http2Template>,
    requests: BTreeMap<i32, Http2Request>,
    /// Cookie box id -> owning context id.
    cookie_boxes: BTreeMap<i32, i32>,
    next_context: i32,
    next_template: i32,
    next_request: i32,
}

static HTTP2: Mutex<Http2State> = Mutex::new(Http2State {
    contexts: BTreeMap::new(),
    templates: BTreeMap::new(),
    requests: BTreeMap::new(),
    cookie_boxes: BTreeMap::new(),
    next_context: 1,
    next_template: 1,
    next_request: 1,
});

fn http2() -> MutexGuard<'static, Http2State> {
    HTTP2.lock().unwrap_or_else(|p| p.into_inner())
}

impl Http2State {
    /// Options of request `id`, else of template `id`.
    fn options(&mut self, id: i32) -> Option<&mut Http2Options> {
        if let Some(r) = self.requests.get_mut(&id) {
            return Some(&mut r.options);
        }
        self.templates.get_mut(&id).map(|t| &mut t.options)
    }

    /// Store a new request and return its id.
    fn insert_request(&mut self, request: Http2Request) -> HleResult {
        let id = self.next_request;
        self.next_request += 1;
        self.requests.insert(id, request);
        id as i64
    }

    fn remove_template(&mut self, id: i32) -> bool {
        if self.templates.remove(&id).is_none() {
            return false;
        }
        self.requests.retain(|_, r| r.tmpl_id != id);
        true
    }
}

/// Apply `f` to the options of request or template `id`.
fn update_options(id: i32, f: impl FnOnce(&mut Http2Options)) -> HleResult {
    match http2().options(id) {
        Some(o) => {
            f(o);
            0
        }
        None => HTTP2_ERROR_INVALID_ID,
    }
}

/// sceHttp2Init(libnetMemId, libsslCtxId, poolSize, maxConcurrentRequest)
fn cb_http2_init(ctx: &mut CpuContext) -> HleResult {
    let mut st = http2();
    let id = st.next_context;
    st.next_context += 1;
    st.contexts.insert(
        id,
        Http2Context {
            libnet_mem_id: ctx.rdi() as i32,
            libssl_ctx_id: ctx.rsi() as i32,
            pool_size: ctx.rdx(),
            max_concurrent_requests: ctx.rcx() as i32,
        },
    );
    id as i64
}

/// sceHttp2Term(ctxId): drops its templates and their requests.
fn cb_http2_term(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let mut st = http2();
    if st.contexts.remove(&id).is_none() {
        return HTTP2_ERROR_INVALID_ID;
    }
    st.cookie_boxes.retain(|_, owner| *owner != id);
    let dead: Vec<i32> = st.templates.iter().filter(|(_, t)| t.context_id == id).map(|(k, _)| *k).collect();
    for t in dead {
        st.remove_template(t);
    }
    0
}

/// sceHttp2CreateTemplate(ctxId, userAgent, httpVer, isAutoProxyConf)
fn cb_http2_create_template(ctx: &mut CpuContext) -> HleResult {
    let context_id = ctx.rdi() as i32;
    let user_agent = read_string(ctx, ctx.rsi()).unwrap_or_default();
    let mut st = http2();
    if !st.contexts.contains_key(&context_id) {
        return HTTP2_ERROR_INVALID_ID;
    }
    let id = st.next_template;
    st.next_template += 1;
    st.templates.insert(
        id,
        Http2Template {
            context_id,
            user_agent,
            http_ver: ctx.rdx() as i32,
            is_auto_proxy_conf: ctx.rcx() as i32,
            options: Http2Options::default(),
        },
    );
    id as i64
}

/// sceHttp2DeleteTemplate(tmplId): also drops its requests.
fn cb_http2_delete_template(ctx: &mut CpuContext) -> HleResult {
    if http2().remove_template(ctx.rdi() as i32) { 0 } else { HTTP2_ERROR_INVALID_ID }
}

fn cb_http2_set_min_ssl_version(ctx: &mut CpuContext) -> HleResult {
    let v = ctx.rsi() as u32;
    update_options(ctx.rdi() as i32, |o| o.min_ssl_version = v)
}

fn cb_http2_ssl_disable_option(ctx: &mut CpuContext) -> HleResult {
    let flags = ctx.rsi() as u32;
    update_options(ctx.rdi() as i32, |o| o.ssl_options &= !flags)
}

fn cb_http2_ssl_enable_option(ctx: &mut CpuContext) -> HleResult {
    let flags = ctx.rsi() as u32;
    update_options(ctx.rdi() as i32, |o| o.ssl_options |= flags)
}

fn cb_http2_set_redirect_callback(ctx: &mut CpuContext) -> HleResult {
    let (cb, arg) = (ctx.rsi(), ctx.rdx());
    update_options(ctx.rdi() as i32, |o| {
        o.redirect_callback = cb;
        o.redirect_user_arg = arg;
    })
}

fn cb_http2_set_ssl_callback(ctx: &mut CpuContext) -> HleResult {
    let (cb, arg) = (ctx.rsi(), ctx.rdx());
    update_options(ctx.rdi() as i32, |o| {
        o.ssl_callback = cb;
        o.ssl_callback_arg = arg;
    })
}

fn cb_http2_set_auto_redirect(ctx: &mut CpuContext) -> HleResult {
    let enabled = ctx.rsi() as i32 != 0;
    update_options(ctx.rdi() as i32, |o| o.auto_redirect = enabled)
}

fn cb_http2_set_auth_enabled(ctx: &mut CpuContext) -> HleResult {
    let enabled = ctx.rsi() as i32 != 0;
    update_options(ctx.rdi() as i32, |o| o.auth_enabled = enabled)
}

fn cb_http2_set_inflate_gzip_enabled(ctx: &mut CpuContext) -> HleResult {
    let enabled = ctx.rsi() as i32 != 0;
    update_options(ctx.rdi() as i32, |o| o.inflate_gzip = enabled)
}

fn cb_http2_set_resolve_timeout(ctx: &mut CpuContext) -> HleResult {
    let us = ctx.rsi() as u32;
    update_options(ctx.rdi() as i32, |o| o.resolve_timeout_us = us)
}

/// sceHttp2SetResolveRetry(id, retry): stored; names are never resolved.
fn cb_http2_set_resolve_retry(ctx: &mut CpuContext) -> HleResult {
    let retry = ctx.rsi() as i32;
    update_options(ctx.rdi() as i32, |o| o.resolve_retry = retry)
}

/// sceHttp2GetMemoryPoolStats(ctxId, SceHttp2MemoryPoolStats* stats):
/// poolSize, maxInuseSize, currentInuseSize (size_t each), reserved. The
/// offline library allocates nothing from the pool given at sceHttp2Init.
fn cb_http2_get_memory_pool_stats(ctx: &mut CpuContext) -> HleResult {
    let (id, out) = (ctx.rdi() as i32, ctx.rsi());
    if out == 0 {
        return HTTP2_ERROR_NULL_POINTER;
    }
    let Some(pool_size) = http2().contexts.get(&id).map(|c| c.pool_size) else {
        // An unknown or terminated context zeroes the statistics.
        ctx.write_bytes(out, &[0u8; 32]);
        return HTTP2_ERROR_INVALID_ID;
    };
    let mut b = [0u8; 32];
    b[0..8].copy_from_slice(&pool_size.to_le_bytes());
    if ctx.write_bytes(out, &b) {
        0
    } else {
        ERR_MEMORY_FAULT
    }
}

fn cb_http2_set_connect_timeout(ctx: &mut CpuContext) -> HleResult {
    let us = ctx.rsi() as u32;
    update_options(ctx.rdi() as i32, |o| o.connect_timeout_us = us)
}

fn cb_http2_set_connection_wait_timeout(ctx: &mut CpuContext) -> HleResult {
    let us = ctx.rsi() as u32;
    update_options(ctx.rdi() as i32, |o| o.connection_wait_timeout_us = us)
}

fn cb_http2_set_send_timeout(ctx: &mut CpuContext) -> HleResult {
    let us = ctx.rsi() as u32;
    update_options(ctx.rdi() as i32, |o| o.send_timeout_us = us)
}

fn cb_http2_set_recv_timeout(ctx: &mut CpuContext) -> HleResult {
    let us = ctx.rsi() as u32;
    update_options(ctx.rdi() as i32, |o| o.recv_timeout_us = us)
}

fn cb_http2_set_timeout(ctx: &mut CpuContext) -> HleResult {
    let us = ctx.rsi() as u32;
    update_options(ctx.rdi() as i32, |o| o.timeout_us = us)
}

/// sceHttp2CreateCookieBox(libhttp2CtxId): the box belongs to the context
/// and dies with it. Box ids come from the context id sequence, so no box
/// id is ever also a context id.
fn cb_http2_create_cookie_box(ctx: &mut CpuContext) -> HleResult {
    let context = ctx.rdi() as i32;
    let mut st = http2();
    if !st.contexts.contains_key(&context) {
        return HTTP2_ERROR_INVALID_ID;
    }
    let id = st.next_context;
    st.next_context += 1;
    st.cookie_boxes.insert(id, context);
    id as i64
}

/// sceHttp2DeleteCookieBox(cookieBoxId)
fn cb_http2_delete_cookie_box(ctx: &mut CpuContext) -> HleResult {
    if http2().cookie_boxes.remove(&(ctx.rdi() as i32)).is_some() { 0 } else { HTTP2_ERROR_INVALID_ID }
}

/// sceHttp2SetCookieBox(id, cookieBoxId): 0 detaches.
fn cb_http2_set_cookie_box(ctx: &mut CpuContext) -> HleResult {
    let box_id = ctx.rsi() as i32;
    if box_id != 0 && !http2().cookie_boxes.contains_key(&box_id) {
        return HTTP2_ERROR_INVALID_ID;
    }
    update_options(ctx.rdi() as i32, |o| o.cookie_box_id = box_id)
}

/// sceHttp2CookieFlush(id): context or cookie box; no cookie is ever
/// received offline, so there is nothing to write back.
fn cb_http2_cookie_flush(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let st = http2();
    if st.contexts.contains_key(&id) || st.cookie_boxes.contains_key(&id) { 0 } else { HTTP2_ERROR_INVALID_ID }
}

/// sceHttp2CreateRequestWithURL(tmplId, method, url, contentLength)
fn cb_http2_create_request_with_url(ctx: &mut CpuContext) -> HleResult {
    let tmpl_id = ctx.rdi() as i32;
    if ctx.rsi() == 0 || ctx.rdx() == 0 {
        return HTTP2_ERROR_NULL_POINTER;
    }
    let (Some(method), Some(url)) = (read_string(ctx, ctx.rsi()), read_string(ctx, ctx.rdx())) else {
        return ERR_MEMORY_FAULT;
    };
    let mut st = http2();
    if !st.templates.contains_key(&tmpl_id) {
        return HTTP2_ERROR_INVALID_ID;
    }
    log::info!("[http2] request {method} {url} (offline)");
    st.insert_request(Http2Request::new(tmpl_id, method, url, ctx.rcx(), None))
}

/// sceHttp2AddRequestHeader(reqId, name, value, mode)
fn cb_http2_add_request_header(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 || ctx.rdx() == 0 {
        return HTTP2_ERROR_NULL_POINTER;
    }
    let (Some(name), Some(value)) = (read_string(ctx, ctx.rsi()), read_string(ctx, ctx.rdx())) else {
        return ERR_MEMORY_FAULT;
    };
    match http2().requests.get_mut(&(ctx.rdi() as i32)) {
        Some(r) => {
            r.headers.push((name, value));
            0
        }
        None => HTTP2_ERROR_INVALID_ID,
    }
}

fn cb_http2_set_request_content_length(ctx: &mut CpuContext) -> HleResult {
    let length = ctx.rsi();
    match http2().requests.get_mut(&(ctx.rdi() as i32)) {
        Some(r) => {
            r.content_length = length;
            0
        }
        None => HTTP2_ERROR_INVALID_ID,
    }
}

/// sceHttp2SetRequestNoContentLength(reqId): a request id only; a template
/// id is not a request.
fn cb_http2_set_request_no_content_length(ctx: &mut CpuContext) -> HleResult {
    match http2().requests.get_mut(&(ctx.rdi() as i32)) {
        Some(r) => {
            r.no_content_length = true;
            0
        }
        None => HTTP2_ERROR_INVALID_ID,
    }
}

fn cb_http2_delete_request(ctx: &mut CpuContext) -> HleResult {
    match http2().requests.remove(&(ctx.rdi() as i32)) {
        Some(_) => 0,
        None => HTTP2_ERROR_INVALID_ID,
    }
}

/// sceHttp2SendRequest(reqId, postData, size): the console is offline, so
/// the request times out.
fn cb_http2_send_request(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 && ctx.rdx() != 0 {
        return HTTP2_ERROR_NULL_POINTER;
    }
    match http2().requests.get_mut(&(ctx.rdi() as i32)) {
        Some(r) => {
            r.send_result = HTTP2_ERROR_TIMEOUT;
            r.send_result
        }
        None => HTTP2_ERROR_INVALID_ID,
    }
}

/// SceHttp2AsyncOption (the kqueue option of the asynchronous calls):
/// equeue @0, userEventId i32 @8, userData @0x10, reserved @0x18.
const HTTP2_ASYNC_OPTION_SIZE: usize = 0x20;

/// Signal the completion of an asynchronous call through its kqueue option
/// (NULL: the caller waits with sceHttp2WaitAsync): the user event
/// `userEventId` of `equeue` fires with `userData`, as
/// sceKernelTriggerUserEvent does.
fn signal_async_completion(ctx: &CpuContext, option: u64) {
    if option == 0 {
        return;
    }
    let Some(o) = ctx.read_bytes(option, HTTP2_ASYNC_OPTION_SIZE) else {
        log::warn!("[http2] unreadable kqueue option {option:#x}");
        return;
    };
    let equeue = u64::from_le_bytes(o[0..8].try_into().unwrap());
    let id = i32::from_le_bytes(o[8..12].try_into().unwrap());
    let user_data = u64::from_le_bytes(o[0x10..0x18].try_into().unwrap());
    let r = crate::sync::equeue::trigger(equeue, id as u64, crate::sync::equeue::FILTER_USER, user_data);
    if r != 0 {
        log::warn!("[http2] completion event {id} on equeue {equeue:#x} not delivered ({:#x})", r as u32);
    }
}

/// sceHttp2SendRequestAsync(reqId, postData, size, kqueueOption, option):
/// completes at once with the timeout as its asynchronous result; the
/// kqueue option's user event announces it.
fn cb_http2_send_request_async(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 && ctx.rdx() != 0 {
        return HTTP2_ERROR_NULL_POINTER;
    }
    let found = match http2().requests.get_mut(&(ctx.rdi() as i32)) {
        Some(r) => {
            r.send_result = HTTP2_ERROR_TIMEOUT;
            r.async_result = r.send_result;
            r.async_event = 0;
            true
        }
        None => false,
    };
    if !found {
        return HTTP2_ERROR_INVALID_ID;
    }
    signal_async_completion(ctx, ctx.rcx());
    0
}

/// sceHttp2AbortRequest(reqId): an unsent request reports ABORTED.
fn cb_http2_abort_request(ctx: &mut CpuContext) -> HleResult {
    match http2().requests.get_mut(&(ctx.rdi() as i32)) {
        Some(r) => {
            if r.send_result == HTTP2_ERROR_BEFORE_SEND {
                r.send_result = HTTP2_ERROR_ABORTED;
            }
            0
        }
        None => HTTP2_ERROR_INVALID_ID,
    }
}

/// SceHttp2AsyncResult: eventType, reqId, result, padding, reserved*.
const HTTP2_ASYNC_RESULT_SIZE: usize = 24;

/// sceHttp2WaitAsync(reqId, SceHttp2AsyncResult* result, timeout*, option*)
fn cb_http2_wait_async(ctx: &mut CpuContext) -> HleResult {
    let (id, out) = (ctx.rdi() as i32, ctx.rsi());
    if out == 0 {
        return HTTP2_ERROR_NULL_POINTER;
    }
    let Some((event, result)) = http2().requests.get(&id).map(|r| (r.async_event, r.async_result)) else {
        return HTTP2_ERROR_INVALID_ID;
    };
    let mut b = [0u8; HTTP2_ASYNC_RESULT_SIZE];
    b[0..4].copy_from_slice(&event.to_le_bytes());
    b[4..8].copy_from_slice(&id.to_le_bytes());
    b[8..12].copy_from_slice(&(result as i32).to_le_bytes());
    if ctx.write_bytes(out, &b) { 0 } else { ERR_MEMORY_FAULT }
}

/// Copy the next body bytes of request `id` into guest `data`: Ok(count)
/// or the send result when no response exists.
fn read_body(ctx: &CpuContext, id: i32, data: u64, size: u64) -> Result<Result<usize, HleResult>, HleResult> {
    let mut st = http2();
    let r = st.requests.get_mut(&id).ok_or(HTTP2_ERROR_INVALID_ID)?;
    if r.send_result != 0 {
        return Ok(Err(r.send_result));
    }
    let remaining = r.response_body.len().saturating_sub(r.read_offset);
    let n = (size as usize).min(remaining);
    if n != 0 {
        if !ctx.write_bytes(data, &r.response_body[r.read_offset..r.read_offset + n]) {
            return Err(ERR_MEMORY_FAULT);
        }
        r.read_offset += n;
    }
    Ok(Ok(n))
}

/// sceHttp2ReadData(reqId, data, size)
fn cb_http2_read_data(ctx: &mut CpuContext) -> HleResult {
    let (data, size) = (ctx.rsi(), ctx.rdx());
    if data == 0 && size != 0 {
        return HTTP2_ERROR_NULL_POINTER;
    }
    match read_body(ctx, ctx.rdi() as i32, data, size) {
        Ok(Ok(n)) => n as i64,
        Ok(Err(e)) | Err(e) => e,
    }
}

/// sceHttp2ReadDataAsync(reqId, data, size, kqueueOption, option): the
/// read completes at once; its result is reported by sceHttp2WaitAsync.
fn cb_http2_read_data_async(ctx: &mut CpuContext) -> HleResult {
    let (id, data, size) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    if data == 0 && size != 0 {
        return HTTP2_ERROR_NULL_POINTER;
    }
    let result = match read_body(ctx, id, data, size) {
        Ok(Ok(n)) => n as HleResult,
        Ok(Err(e)) => e,
        Err(e) => return e,
    };
    if let Some(r) = http2().requests.get_mut(&id) {
        r.async_result = result;
        r.async_event = 1;
    }
    signal_async_completion(ctx, ctx.rcx());
    0
}

/// Send result of request `id`, writing nothing on failure.
fn http2_send_result(id: i32) -> Result<HleResult, HleResult> {
    http2().requests.get(&id).map(|r| r.send_result).ok_or(HTTP2_ERROR_INVALID_ID)
}

/// sceHttp2GetStatusCode(reqId, int* statusCode)
fn cb_http2_get_status_code(ctx: &mut CpuContext) -> HleResult {
    let (id, out) = (ctx.rdi() as i32, ctx.rsi());
    if out == 0 {
        return HTTP2_ERROR_NULL_POINTER;
    }
    if !ctx.write_u32(out, 0) {
        return ERR_MEMORY_FAULT;
    }
    match http2_send_result(id) {
        Ok(0) => {
            let status = http2().requests.get(&id).map_or(0, |r| r.status_code);
            ctx.write_u32(out, status as u32);
            0
        }
        Ok(e) | Err(e) => e,
    }
}

/// sceHttp2GetResponseContentLength(reqId, int* result, uint64_t* length)
fn cb_http2_get_response_content_length(ctx: &mut CpuContext) -> HleResult {
    let (id, result, length) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    if result == 0 || length == 0 {
        return HTTP2_ERROR_NULL_POINTER;
    }
    if !ctx.write_u32(result, 0) || !ctx.write_u64(length, 0) {
        return ERR_MEMORY_FAULT;
    }
    match http2_send_result(id) {
        Ok(0) => {
            let n = http2().requests.get(&id).map_or(0, |r| r.response_body.len());
            ctx.write_u64(length, n as u64);
            0
        }
        Ok(e) => {
            ctx.write_u32(result, -1i32 as u32);
            e
        }
        Err(e) => e,
    }
}

/// sceHttp2GetAllResponseHeaders(reqId, char** header, size_t* headerSize)
fn cb_http2_get_all_response_headers(ctx: &mut CpuContext) -> HleResult {
    let (id, header, size) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    if header == 0 || size == 0 {
        return HTTP2_ERROR_NULL_POINTER;
    }
    if !ctx.write_u64(header, 0) || !ctx.write_u64(size, 0) {
        return ERR_MEMORY_FAULT;
    }
    match http2_send_result(id) {
        Ok(0) => {
            ctx.write_u64(header, empty_guest_string(ctx));
            0
        }
        Ok(e) | Err(e) => e,
    }
}

/// Argument `i` past the six register arguments.
fn stack_arg(ctx: &CpuContext, i: u64) -> u64 {
    ctx.read_u64(ctx.get(crate::context::CpuRegister::Rsp) + 8 + 8 * i).unwrap_or(0)
}

/// sceHttp2WebSocketCreateRequest(tmplId, url, textCallback, dataCallback,
/// closeCallback, errorCallback, recvBuffer, recvBufferSize, userData,
/// option): a request whose opening handshake (a GET with the upgrade
/// headers) the caller sends with sceHttp2SendRequestAsync, after adding
/// its own headers such as Sec-WebSocket-Protocol.
fn cb_http2_websocket_create_request(ctx: &mut CpuContext) -> HleResult {
    let (tmpl_id, url_ptr) = (ctx.rdi() as i32, ctx.rsi());
    if url_ptr == 0 {
        return HTTP2_ERROR_NULL_POINTER;
    }
    let Some(url) = read_string(ctx, url_ptr) else {
        return ERR_MEMORY_FAULT;
    };
    let websocket = Http2WebSocket {
        text_callback: ctx.rdx(),
        data_callback: ctx.rcx(),
        close_callback: ctx.r8(),
        error_callback: ctx.r9(),
        recv_buffer: stack_arg(ctx, 0),
        recv_buffer_size: stack_arg(ctx, 1),
        user_data: stack_arg(ctx, 2),
        ..Http2WebSocket::default()
    };
    let mut st = http2();
    if !st.templates.contains_key(&tmpl_id) {
        return HTTP2_ERROR_INVALID_ID;
    }
    log::info!("[http2] websocket {url} (offline)");
    st.insert_request(Http2Request::new(tmpl_id, "GET".into(), url, 0, Some(websocket)))
}

/// Apply `f` to the WebSocket state of request `id`; an id that is not a
/// WebSocket request is invalid.
fn update_websocket(id: i32, f: impl FnOnce(&mut Http2WebSocket)) -> HleResult {
    match http2().requests.get_mut(&id).and_then(|r| r.websocket.as_mut()) {
        Some(ws) => {
            f(ws);
            0
        }
        None => HTTP2_ERROR_INVALID_ID,
    }
}

/// sceHttp2WebSocketSetPingInterval(reqId, microseconds)
fn cb_http2_websocket_set_ping_interval(ctx: &mut CpuContext) -> HleResult {
    let us = ctx.rsi() as u32;
    update_websocket(ctx.rdi() as i32, |ws| ws.ping_interval_us = us)
}

/// sceHttp2WebSocketSetPingTimeout(reqId, microseconds)
fn cb_http2_websocket_set_ping_timeout(ctx: &mut CpuContext) -> HleResult {
    let us = ctx.rsi() as u32;
    update_websocket(ctx.rdi() as i32, |ws| ws.ping_timeout_us = us)
}

/// Common part of the WebSocket message and close calls on request `id`:
/// frames go out only on an open connection, so before the handshake was
/// sent the call fails with BEFORE_SEND and after a failed handshake with
/// its error, the way reading the body of a failed request does. Offline
/// the handshake never succeeds, so nothing is ever queued and neither
/// `kqueueOption` nor sceHttp2WaitAsync sees a completion.
fn websocket_send(id: i32) -> HleResult {
    match http2().requests.get(&id) {
        Some(r) if r.websocket.is_some() => {
            debug_assert_ne!(r.send_result, 0, "offline WebSocket handshake succeeded");
            r.send_result
        }
        _ => HTTP2_ERROR_INVALID_ID,
    }
}

/// sceHttp2WebSocketSendTextMessageAsync(reqId, text, flags, kqueueOption,
/// option)
fn cb_http2_websocket_send_text_message_async(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 {
        return HTTP2_ERROR_NULL_POINTER;
    }
    websocket_send(ctx.rdi() as i32)
}

/// sceHttp2WebSocketSendDataMessageAsync(reqId, data, size, flags,
/// kqueueOption, option)
fn cb_http2_websocket_send_data_message_async(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 && ctx.rdx() != 0 {
        return HTTP2_ERROR_NULL_POINTER;
    }
    websocket_send(ctx.rdi() as i32)
}

/// sceHttp2WebSocketCloseAsync(reqId, const uint16_t* status, reason,
/// flags, kqueueOption, option): starts the closing handshake, which is a
/// frame on the open connection like a message.
fn cb_http2_websocket_close_async(ctx: &mut CpuContext) -> HleResult {
    websocket_send(ctx.rdi() as i32)
}

const HTTP_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("A9cVMUtEp4Y", "sceHttpInit", cb_http_init),
    ("Ik-KpLTlf7Q", "sceHttpTerm", cb_http_term),
    ("0gYjPTR-6cY", "sceHttpCreateTemplate", cb_http_create_template),
    ("4I8vEpuEhZ8", "sceHttpDeleteTemplate", cb_http_delete_template),
    ("s2-NPIvz+iA", "sceHttpSetNonblock", cb_http_set_nonblock),
    ("htyBOoWeS58", "sceHttpsSetSslCallback", cb_https_set_ssl_callback),
    ("jUjp+yqMNdQ", "sceHttpsSetMinSslVersion", cb_https_set_min_ssl_version),
    ("mSQCxzWTwVI", "sceHttpsDisableOption", cb_https_disable_option),
    ("6381dWF+xsQ", "sceHttpCreateEpoll", cb_http_create_epoll),
    ("wYhXVfS2Et4", "sceHttpDestroyEpoll", cb_http_destroy_epoll),
    ("-xm7kZQNpHI", "sceHttpSetEpoll", cb_http_set_epoll),
    ("7j9VcwnrZo4", "sceHttpGetEpoll", cb_http_get_epoll),
    ("JKl06ZIAl6A", "sceHttpAbortRequestForce", cb_http_abort_request),
    ("A7n9nNg7NBg", "sceHttpCacheRedirectedConnectionEnabled", cb_http_cache_redirected_connection_enabled),
    ("I4+4hKttt1w", "sceHttpsEnableOptionPrivate", cb_https_option_private),
    ("zJYi5br6ZiQ", "sceHttpsDisableOptionPrivate", cb_https_option_private),
    ("8kzIXsRy1bY", "sceHttpSetDefaultAcceptEncodingGZIPEnabled", cb_http_set_default_accept_encoding_gzip),
    ("HRX1iyDoKR8", "sceHttpSetAcceptEncodingGZIPEnabled", cb_http_set_accept_encoding_gzip),
    ("1rpZqxdMRwQ", "sceHttpGetAcceptEncodingGZIPEnabled", cb_http_get_accept_encoding_gzip),
    ("fmOs6MzCRqk", "sceHttpTrySetNonblock", cb_http_set_nonblock),
    ("Wq4RNB3snSQ", "sceHttpGetNonblock", cb_http_get_nonblock),
    ("V-noPEjSB8c", "sceHttpTryGetNonblock", cb_http_get_nonblock),
    ("mmLexUbtnfY", "sceHttpGetAutoRedirect", cb_http_get_auto_redirect),
    ("59tL1AQBb8U", "sceHttpUnsetEpoll", cb_http_unset_epoll),
    ("Kiwv9r4IZCc", "sceHttpCreateConnection", cb_http_create_connection),
    ("qgxDBjorUxs", "sceHttpCreateConnectionWithURL", cb_http_create_connection_with_url),
    ("P6A3ytpsiYc", "sceHttpDeleteConnection", cb_http_delete_connection),
    ("tsGVru3hCe8", "sceHttpCreateRequest", cb_http_create_request),
    ("Aeu5wVKkF9w", "sceHttpCreateRequestWithURL", cb_http_create_request),
    ("rGNm+FjIXKk", "sceHttpCreateRequest2", cb_http_create_request2),
    ("Cnp77podkCU", "sceHttpCreateRequestWithURL2", cb_http_create_request_with_url2),
    ("qe7oZ+v4PWA", "sceHttpDeleteRequest", cb_http_delete_request),
    ("PTiFIUxCpJc", "sceHttpSetRequestContentLength", cb_http_set_request_content_length),
    ("EY28T2bkN7k", "sceHttpAddRequestHeader", cb_http_add_request_header),
    ("1e2BNwI-XzE", "sceHttpSendRequest", cb_http_send_request),
    ("hvG6GfBMXg8", "sceHttpAbortRequest", cb_http_abort_request),
    ("qISjDHrxONc", "sceHttpWaitRequest", cb_http_wait_request),
    ("0a2TBNfE3BU", "sceHttpGetStatusCode", cb_http_get_status_code),
    ("aCYPMSUIaP8", "sceHttpGetAllResponseHeaders", cb_http_get_all_response_headers),
    ("yuO2H2Uvnos", "sceHttpGetResponseContentLength", cb_http_get_response_content_length),
    ("Tc-hAYDKtQc", "sceHttpSetResolveTimeOut", cb_http_set_resolve_timeout),
    ("K1d1LqZRQHQ", "sceHttpSetResolveRetry", cb_http_set_resolve_retry),
    ("0S9tTH0uqTU", "sceHttpSetConnectTimeOut", cb_http_set_connect_timeout),
    ("xegFfZKBVlw", "sceHttpSetSendTimeOut", cb_http_set_send_timeout),
    ("yigr4V0-HTM", "sceHttpSetRecvTimeOut", cb_http_set_recv_timeout),
    ("mMcB2XIDoV4", "sceHttpSetRecvBlockSize", cb_http_set_recv_block_size),
    ("T-mGo9f3Pu4", "sceHttpSetAutoRedirect", cb_http_set_auto_redirect),
    ("qFg2SuyTJJY", "sceHttpSetAuthEnabled", cb_http_set_auth_enabled),
    ("jf4TB2nUO40", "sceHttpSetAuthInfoCallback", cb_http_set_auth_info_callback),
    ("vO4B-42ef-k", "sceHttpSetRequestStatusCallback", cb_http_set_request_status_callback),
    ("IWalAn-guFs", "sceHttpUriParse", cb_http_uri_parse),
    ("YuOW3dDAKYc", "sceHttpUriEscape", cb_http_uri_escape),
    ("5LZA+KPISVA", "sceHttpUriBuild", cb_http_uri_build),
    ("PDxS48xGQLs", "sceHttpSetChunkedTransferEnabled", cb_http_set_chunked_transfer_enabled),
    ("P5pdoykPYTk", "sceHttpReadData", cb_http_read_data),
    ("0onIrKx9NIE", "sceHttpGetLastErrno", cb_http_get_last_errno),
    ("sWQiqKvYTVA", "sceHttpAbortWaitRequest", cb_http_abort_wait_request),
    ("hPTXo3bICzI", "sceHttpParseResponseHeader", cb_http_parse_response_header),
    ("Qq8SfuJJJqE", "sceHttpParseStatusLine", cb_http_parse_status_line),
    ("u05NnI+P+KY", "sceHttpRedirectCacheFlush", cb_http_redirect_cache_flush),
    // No authentication or cookie is ever stored offline either.
    ("zzB0StvRab4", "sceHttpAuthCacheFlush", cb_http_store_flush),
    ("seCvUt91WHY", "sceHttpCookieFlush", cb_http_store_flush),
    ("Kh6bS2HQKbo", "sceHttpSetCookieRecvCallback", cb_http_set_cookie_recv_callback),
    ("i9mhafzkEi8", "sceHttpSetInflateGZIPEnabled", cb_http_set_inflate_gzip_enabled),
    ("h9wmFZX4i-4", "sceHttpSetRedirectCallback", cb_http_set_redirect_callback),
    ("a4VsZ4oqn68", "sceHttpSetResponseHeaderMaxSize", cb_http_set_response_header_max_size),
    ("3lgQ5Qk42ok", "sceHttpUriMerge", cb_http_uri_merge),
    ("mUU363n4yc0", "sceHttpUriSweepPath", cb_http_uri_sweep_path),
    ("thTS+57zoLM", "sceHttpUriUnescape", cb_http_uri_unescape),
    ("f42K37mm5RM", "sceHttpsEnableOption", cb_https_enable_option),
    ("7WcNoAI9Zcw", "sceHttpsFreeCaList", cb_https_ca_list),
    ("gcUjwU3fa0M", "sceHttpsGetCaList", cb_https_ca_list),
    ("JBN6N-EY+3M", "sceHttpsGetSslError", cb_https_get_ssl_error),
    ("DK+GoXCNT04", "sceHttpsLoadCert", cb_https_load_cert),
    ("U5ExQGyyx9s", "sceHttpsSetSslVersion", cb_https_set_ssl_version),
    ("zXqcE0fizz0", "sceHttpsUnloadCert", cb_https_unload_cert),
];

const HTTP2_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("3JCe3lCbQ8A", "sceHttp2Init", cb_http2_init),
    ("YiBUtz-pGkc", "sceHttp2Term", cb_http2_term),
    ("+wCt7fCijgk", "sceHttp2CreateTemplate", cb_http2_create_template),
    ("pDom5-078DA", "sceHttp2DeleteTemplate", cb_http2_delete_template),
    ("mmyOCxQMVYQ", "sceHttp2CreateRequestWithURL", cb_http2_create_request_with_url),
    ("nrPfOE8TQu0", "sceHttp2AddRequestHeader", cb_http2_add_request_header),
    ("FSAFOzi0FpM", "sceHttp2SetRequestContentLength", cb_http2_set_request_content_length),
    ("bEegosRhgM0", "sceHttp2SetRequestNoContentLength", cb_http2_set_request_no_content_length),
    ("jjFahkBPCYs", "sceHttp2SetAuthEnabled", cb_http2_set_auth_enabled),
    ("b9AvoIaOuHI", "sceHttp2SetAutoRedirect", cb_http2_set_auto_redirect),
    ("uRosf8GQbHQ", "sceHttp2SetInflateGZIPEnabled", cb_http2_set_inflate_gzip_enabled),
    ("B37SruheQ5Y", "sceHttp2SslDisableOption", cb_http2_ssl_disable_option),
    ("EWcwMpbr5F8", "sceHttp2SslEnableOption", cb_http2_ssl_enable_option),
    ("BJgi0CH7al4", "sceHttp2SetRedirectCallback", cb_http2_set_redirect_callback),
    ("izvHhqgDt44", "sceHttp2SetRecvTimeOut", cb_http2_set_recv_timeout),
    ("XPtW45xiLHk", "sceHttp2SetSendTimeOut", cb_http2_set_send_timeout),
    ("-HIO4VT87v8", "sceHttp2SetConnectTimeOut", cb_http2_set_connect_timeout),
    ("n8hMLe31OPA", "sceHttp2SetConnectionWaitTimeOut", cb_http2_set_connection_wait_timeout),
    ("ACjtE27aErY", "sceHttp2SetResolveTimeOut", cb_http2_set_resolve_timeout),
    ("Gcjh+CisAZM", "sceHttp2SetResolveRetry", cb_http2_set_resolve_retry),
    ("otUQuZa-mv0", "sceHttp2GetMemoryPoolStats", cb_http2_get_memory_pool_stats),
    ("VYMxTcBqSE0", "sceHttp2SetTimeOut", cb_http2_set_timeout),
    ("YrWX+DhPHQY", "sceHttp2SetSslCallback", cb_http2_set_ssl_callback),
    ("rbqZig38AT8", "sceHttp2SendRequest", cb_http2_send_request),
    ("A+NVAFu4eCg", "sceHttp2SendRequestAsync", cb_http2_send_request_async),
    ("o0DBQpFE13o", "sceHttp2GetResponseContentLength", cb_http2_get_response_content_length),
    ("9XYJwCf3lEA", "sceHttp2GetStatusCode", cb_http2_get_status_code),
    ("-rdXUi2XW90", "sceHttp2GetAllResponseHeaders", cb_http2_get_all_response_headers),
    ("QygCNNmbGss", "sceHttp2ReadData", cb_http2_read_data),
    ("bGN-6zbo7ms", "sceHttp2ReadDataAsync", cb_http2_read_data_async),
    ("c8D9qIjo8EY", "sceHttp2DeleteRequest", cb_http2_delete_request),
    ("MOp-AUhdfi8", "sceHttp2WaitAsync", cb_http2_wait_async),
    ("09tk+kIA1Ns", "sceHttp2SetMinSslVersion", cb_http2_set_min_ssl_version),
    ("N4UfjvWJsMw", "sceHttp2CreateCookieBox", cb_http2_create_cookie_box),
    ("jrVHsKCXA0g", "sceHttp2SetCookieBox", cb_http2_set_cookie_box),
    ("O9ync3F-JVI", "sceHttp2DeleteCookieBox", cb_http2_delete_cookie_box),
    ("IZ-qjhRqvjk", "sceHttp2AbortRequest", cb_http2_abort_request),
    ("5VlQSzXW-SQ", "sceHttp2CookieFlush", cb_http2_cookie_flush),
    ("-0wUiGX74GQ", "sceHttp2WebSocketCreateRequest", cb_http2_websocket_create_request),
    ("YfaXPnc8PJE", "sceHttp2WebSocketSetPingInterval", cb_http2_websocket_set_ping_interval),
    ("-MU9L6S3Fj4", "sceHttp2WebSocketSetPingTimeout", cb_http2_websocket_set_ping_timeout),
    ("WYszaHaIkWs", "sceHttp2WebSocketSendTextMessageAsync", cb_http2_websocket_send_text_message_async),
    ("ksn0O9Iilb0", "sceHttp2WebSocketSendDataMessageAsync", cb_http2_websocket_send_data_message_async),
    ("BlZ8niq3nHU", "sceHttp2WebSocketCloseAsync", cb_http2_websocket_close_async),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in HTTP_EXPORTS {
        reg.register_fn(nid, name, LIB_HTTP, cb);
    }
    for &(nid, name, cb) in HTTP2_EXPORTS {
        reg.register_fn(nid, name, LIB_HTTP2, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    fn args(ctx: &mut CpuContext, args: &[u64]) {
        for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
            ctx.set(*r, *v);
        }
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in HTTP_EXPORTS.iter().chain(HTTP2_EXPORTS) {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// Guest scratch for the URI tests: the URL at +0, the element at
    /// +0x100, the pool at +0x200, require at +0x80, build output at +0x400.
    struct Uri<'a> {
        ctx: CpuContext<'a>,
        base: u64,
    }

    impl Uri<'_> {
        fn parse(&mut self, url: &[u8], out: bool, pool: bool, prepare: u64) -> (HleResult, u64) {
            let mut text = url.to_vec();
            text.push(0);
            self.ctx.write_bytes(self.base, &text);
            self.ctx.write_u64(self.base + 0x80, 0);
            let b = self.base;
            args(&mut self.ctx, &[if out { b + 0x100 } else { 0 }, b, if pool { b + 0x200 } else { 0 }, b + 0x80, prepare]);
            let r = cb_http_uri_parse(&mut self.ctx);
            (r, self.ctx.read_u64(b + 0x80).unwrap())
        }

        fn field(&self, index: usize) -> u64 {
            self.ctx.read_u64(self.base + 0x108 + index as u64 * 8).unwrap()
        }

        fn string(&self, index: usize) -> Option<String> {
            let p = self.field(index);
            (p != 0).then(|| self.ctx.read_cstring(p, 256).unwrap())
        }

        fn in_pool(&self, index: usize, used: u64) -> bool {
            let p = self.field(index);
            p >= self.base + 0x200 && p < self.base + 0x200 + used
        }

        fn build(&mut self, option: u32) -> String {
            let b = self.base;
            self.ctx.write_u64(b + 0x88, 0);
            args(&mut self.ctx, &[0, b + 0x88, 0, b + 0x100, option as u64]);
            assert_eq!(cb_http_uri_build(&mut self.ctx), 0);
            let required = self.ctx.read_u64(b + 0x88).unwrap();
            assert!(required <= 256);
            self.ctx.write_u64(b + 0x90, 0);
            args(&mut self.ctx, &[b + 0x400, b + 0x90, required, b + 0x100, option as u64]);
            assert_eq!(cb_http_uri_build(&mut self.ctx), 0);
            assert_eq!(self.ctx.read_u64(b + 0x90), Some(required));
            self.ctx.read_cstring(b + 0x400, 256).unwrap()
        }
    }

    const QUERY: usize = 5;

    #[test]
    fn uri_parse_absent_query() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D04_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut u = Uri { ctx: CpuContext::new(&vm), base: BASE };
        let (r, required) = u.parse(b"http://example.com/path", false, false, 0);
        assert_eq!(r, 0);
        assert_eq!(required, 5 + 12 + 6 + 1);
        let (r, parsed) = u.parse(b"http://example.com/path", true, true, required);
        assert_eq!(r, 0);
        assert_eq!(parsed, required);
        assert!(u.in_pool(QUERY, required));
        assert_eq!(u.string(QUERY).as_deref(), Some(""));
    }

    #[test]
    fn uri_parse_empty() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D05_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut u = Uri { ctx: CpuContext::new(&vm), base: BASE };
        let (r, required) = u.parse(b"", false, false, 0);
        assert_eq!(r, 0);
        assert_eq!(required, 4);
        let (r, parsed) = u.parse(b"", true, true, required);
        assert_eq!(r, 0);
        assert_eq!(parsed, required);
        assert!(u.in_pool(QUERY, required));
        assert_eq!(u.string(QUERY).as_deref(), Some(""));
        assert_eq!(u.ctx.read_u32(BASE + 0x100), Some(1));
    }

    #[test]
    fn uri_parse_present_query() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D06_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut u = Uri { ctx: CpuContext::new(&vm), base: BASE };
        let (r, required) = u.parse(b"http://example.com/path?foo=bar", false, false, 0);
        assert_eq!(r, 0);
        let (r, parsed) = u.parse(b"http://example.com/path?foo=bar", true, true, required);
        assert_eq!(r, 0);
        assert_eq!(parsed, required);
        assert!(u.in_pool(QUERY, required));
        assert_eq!(u.string(QUERY).as_deref(), Some("?foo=bar"));
        // A pool smaller than required fails; zero means unlimited.
        let (r, _) = u.parse(b"http://example.com/path?foo=bar", true, true, required - 1);
        assert_eq!(r, HTTP_ERROR_OUT_OF_MEMORY);
        let (r, _) = u.parse(b"http://example.com/path?foo=bar", true, true, 0);
        assert_eq!(r, 0);
    }

    #[test]
    fn uri_build_honours_option() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D07_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut u = Uri { ctx: CpuContext::new(&vm), base: BASE };
        const URL: &str = "https://user:pw@gssdk1.gamesci.com.cn:8443/VersionServerImpl?x=1#frag";
        let (r, required) = u.parse(URL.as_bytes(), false, false, 0);
        assert_eq!(r, 0);
        assert_eq!(u.parse(URL.as_bytes(), true, true, required).0, 0);
        assert_eq!(u.ctx.read_bytes(BASE + 0x100 + URI_ELEMENT_PORT as u64, 2).unwrap(), 8443u16.to_le_bytes().to_vec());

        let (scheme, hostname, port, path) = (0x01, 0x02, 0x04, 0x08);
        let (username, password, query, fragment) = (0x10, 0x20, 0x40, 0x80);
        assert_eq!(u.build(scheme), "https://");
        assert_eq!(u.build(hostname), "gssdk1.gamesci.com.cn");
        assert_eq!(u.build(port), "8443");
        assert_eq!(u.build(path), "/VersionServerImpl");
        assert_eq!(u.build(username), "user");
        assert_eq!(u.build(password), "pw");
        assert_eq!(u.build(query), "?x=1");
        assert_eq!(u.build(fragment), "#frag");
        assert_eq!(u.build(scheme | hostname), "https://gssdk1.gamesci.com.cn");
        assert_eq!(u.build(hostname | path), "gssdk1.gamesci.com.cn/VersionServerImpl");
        assert_eq!(u.build(hostname | port), "gssdk1.gamesci.com.cn:8443");
        assert_eq!(u.build(0xff), URL);
        assert_eq!(u.build(0), URL);
    }

    #[test]
    fn uri_split_rules() {
        assert!(split_uri(b"1http://x").is_none());
        assert!(split_uri(b"http//x").is_none());
        assert!(split_uri(b"http://[::1").is_none());
        assert!(split_uri(b"http://host:").is_none());
        assert!(split_uri(b"http://host:65536/").is_none());
        let s = split_uri(b"http://[::1]x/p").unwrap();
        assert_eq!(s.components[3].as_deref(), Some(&b"[::1]"[..]));
        assert_eq!(s.components[4].as_deref(), Some(&b"/p"[..]));
        let s = split_uri(b"mailto:a@b").unwrap();
        assert!(s.opaque);
        assert_eq!(s.components[4].as_deref(), Some(&b"a@b"[..]));
        assert_eq!(uri_escape(b"a b/~"), b"a%20b%2F~".to_vec());
    }

    #[test]
    fn uri_escape_sizes() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D08_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, b"a b\0");
        args(&mut ctx, &[0, 0, 0, BASE]);
        assert_eq!(cb_http_uri_escape(&mut ctx), 0);
        args(&mut ctx, &[0, BASE + 0x40, 0, BASE]);
        assert_eq!(cb_http_uri_escape(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE + 0x40), Some(6));
        args(&mut ctx, &[BASE + 0x80, BASE + 0x40, 5, BASE]);
        assert_eq!(cb_http_uri_escape(&mut ctx), HTTP_ERROR_OUT_OF_MEMORY);
        args(&mut ctx, &[BASE + 0x80, BASE + 0x40, 6, BASE]);
        assert_eq!(cb_http_uri_escape(&mut ctx), 0);
        assert_eq!(ctx.read_cstring(BASE + 0x80, 16).unwrap(), "a%20b");
        args(&mut ctx, &[BASE + 0x80, BASE + 0x40, 6, 0]);
        assert_eq!(cb_http_uri_escape(&mut ctx), HTTP_ERROR_INVALID_VALUE);
    }

    /// Context -> template -> connection -> request; sending times out and
    /// the getters report it.
    #[test]
    fn http_objects_offline() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D09_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        args(&mut ctx, &[0x8000]);
        let ssl = crate::ssl::tests_init(&mut ctx);
        ctx.write_bytes(BASE, b"pool\0");
        let pool = crate::net::tests_pool(&mut ctx, BASE);

        // The pool and SSL context must exist.
        args(&mut ctx, &[pool, 0x7777, 0x10000]);
        assert_eq!(cb_http_init(&mut ctx), HTTP_ERROR_OUT_OF_MEMORY);
        args(&mut ctx, &[pool, ssl, 0x10000]);
        let http_ctx = cb_http_init(&mut ctx);
        assert_eq!(split_id(http_ctx as i32).0, ID_TYPE_HTTP);

        ctx.write_bytes(BASE + 0x20, b"agent\0");
        args(&mut ctx, &[0x7777, BASE + 0x20, 1, 0]);
        assert_eq!(cb_http_create_template(&mut ctx), HTTP_ERROR_OUT_OF_MEMORY);
        args(&mut ctx, &[http_ctx as u64, BASE + 0x20, 1, 0]);
        let tmpl = cb_http_create_template(&mut ctx) as u64;
        assert_eq!(split_id(tmpl as i32).0, ID_TYPE_TEMPLATE);

        ctx.write_bytes(BASE + 0x40, b"example.com\0http\0");
        args(&mut ctx, &[tmpl, 0, BASE + 0x4C, 80, 1]);
        assert_eq!(cb_http_create_connection(&mut ctx), HTTP_ERROR_INVALID_VALUE);
        args(&mut ctx, &[tmpl, BASE + 0x40, BASE + 0x4C, 8080, 1]);
        let conn = cb_http_create_connection(&mut ctx) as u64;
        assert_eq!(split_id(conn as i32).0, ID_TYPE_CONNECTION);
        {
            let mut st = http();
            let c = slot(&mut st.connections, conn as i32, ID_TYPE_CONNECTION).unwrap();
            assert_eq!(c.url, "http://example.com:8080");
        }
        ctx.write_bytes(BASE + 0x60, b"\0");
        args(&mut ctx, &[tmpl, BASE + 0x60, 1]);
        assert_eq!(cb_http_create_connection_with_url(&mut ctx), HTTP_ERROR_INVALID_URL);

        ctx.write_bytes(BASE + 0x80, b"/index.html\0");
        args(&mut ctx, &[conn, 9, BASE + 0x80, 0]);
        assert_eq!(cb_http_create_request(&mut ctx), HTTP_ERROR_UNKNOWN_METHOD);
        args(&mut ctx, &[conn, 0, 0, 0]);
        assert_eq!(cb_http_create_request(&mut ctx), HTTP_ERROR_INVALID_VALUE);
        args(&mut ctx, &[0x7777, 0, BASE + 0x80, 0]);
        assert_eq!(cb_http_create_request(&mut ctx), HTTP_ERROR_OUT_OF_MEMORY);
        args(&mut ctx, &[conn, 0, BASE + 0x80, 0]);
        let req = cb_http_create_request(&mut ctx) as u64;
        assert_eq!(split_id(req as i32).0, ID_TYPE_REQUEST);

        // Headers: mode 1 adds, mode 0 overwrites existing names only.
        ctx.write_bytes(BASE + 0xA0, b"X-A\0one\0two\0");
        args(&mut ctx, &[req, BASE + 0xA0, BASE + 0xA4, 0]);
        assert_eq!(cb_http_add_request_header(&mut ctx), 0);
        args(&mut ctx, &[req, BASE + 0xA0, BASE + 0xA4, 1]);
        assert_eq!(cb_http_add_request_header(&mut ctx), 0);
        args(&mut ctx, &[req, BASE + 0xA0, BASE + 0xA8, 0]);
        assert_eq!(cb_http_add_request_header(&mut ctx), 0);
        {
            let mut st = http();
            let r = st.request(req as i32).unwrap();
            assert_eq!(r.conn.tmpl.base.headers, vec![("X-A".to_string(), "two".to_string())]);
        }
        // Resolver settings do not apply to requests.
        args(&mut ctx, &[req, 5]);
        assert_eq!(cb_http_set_resolve_retry(&mut ctx), HTTP_ERROR_INVALID_ID);
        args(&mut ctx, &[conn, 5]);
        assert_eq!(cb_http_set_resolve_retry(&mut ctx), 0);

        args(&mut ctx, &[req, BASE + 0x100]);
        assert_eq!(cb_http_get_status_code(&mut ctx), HTTP_ERROR_BEFORE_SEND);
        args(&mut ctx, &[req, 0, 0]);
        assert_eq!(cb_http_send_request(&mut ctx), HTTP_ERROR_TIMEOUT);
        args(&mut ctx, &[req, BASE + 0x100]);
        assert_eq!(cb_http_get_status_code(&mut ctx), HTTP_ERROR_TIMEOUT);
        assert_eq!(ctx.read_u32(BASE + 0x100), Some(0));
        args(&mut ctx, &[req, BASE + 0x108, BASE + 0x110]);
        assert_eq!(cb_http_get_response_content_length(&mut ctx), HTTP_ERROR_TIMEOUT);
        assert_eq!(ctx.read_u32(BASE + 0x108), Some(u32::MAX));
        args(&mut ctx, &[req, BASE + 0x120, BASE + 0x128]);
        assert_eq!(cb_http_get_all_response_headers(&mut ctx), HTTP_ERROR_TIMEOUT);
        assert_eq!(ctx.read_u64(BASE + 0x128), Some(0));
        args(&mut ctx, &[http_ctx as u64, BASE + 0x1F0]);
        assert_eq!(cb_http_create_epoll(&mut ctx), 0);
        let eh = ctx.read_u64(BASE + 0x1F0).unwrap();
        args(&mut ctx, &[eh, BASE + 0x200, 2, 0]);
        assert_eq!(cb_http_wait_request(&mut ctx), 0);
        // The wait runs out its timeout (microseconds) with no events.
        let started = std::time::Instant::now();
        args(&mut ctx, &[eh, BASE + 0x200, 2, 1000]);
        assert_eq!(cb_http_wait_request(&mut ctx), 0);
        assert!(started.elapsed() >= std::time::Duration::from_micros(1000));
        for bad in [[eh, 0, 2, 0], [eh, BASE + 0x200, 0, 0], [0, BASE + 0x200, 2, 0]] {
            args(&mut ctx, &bad);
            assert_eq!(cb_http_wait_request(&mut ctx), HTTP_ERROR_INVALID_VALUE);
        }
        // Epolls on any object; a handle that is not an epoll is INVALID_ID.
        args(&mut ctx, &[conn, 0x1234, 7]);
        assert_eq!(cb_http_set_epoll(&mut ctx), HTTP_ERROR_INVALID_ID);
        args(&mut ctx, &[0x7777, eh, 7]);
        assert_eq!(cb_http_set_epoll(&mut ctx), HTTP_ERROR_INVALID_ID);
        args(&mut ctx, &[conn, eh, 7]);
        assert_eq!(cb_http_set_epoll(&mut ctx), 0);
        args(&mut ctx, &[conn, BASE + 0x400, BASE + 0x408]);
        assert_eq!(cb_http_get_epoll(&mut ctx), 0);
        assert_eq!((ctx.read_u64(BASE + 0x400), ctx.read_u64(BASE + 0x408)), (Some(eh), Some(7)));
        args(&mut ctx, &[conn, 0, 0]);
        assert_eq!(cb_http_get_epoll(&mut ctx), HTTP_ERROR_INVALID_VALUE);
        args(&mut ctx, &[conn]);
        assert_eq!(cb_http_unset_epoll(&mut ctx), 0);
        args(&mut ctx, &[conn, BASE + 0x400, 0]);
        assert_eq!(cb_http_get_epoll(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE + 0x400), Some(0));
        // The switches and their getters.
        for (get, set, default) in [
            (cb_http_get_nonblock as HleCallback, cb_http_set_nonblock as HleCallback, 0u32),
            (cb_http_get_auto_redirect, cb_http_set_auto_redirect, 1),
            (cb_http_get_accept_encoding_gzip, cb_http_set_accept_encoding_gzip, 0),
        ] {
            args(&mut ctx, &[req, BASE + 0x410]);
            assert_eq!(get(&mut ctx), 0);
            assert_eq!(ctx.read_u32(BASE + 0x410), Some(default));
            args(&mut ctx, &[req, (default ^ 1) as u64]);
            assert_eq!(set(&mut ctx), 0);
            args(&mut ctx, &[req, BASE + 0x410]);
            assert_eq!(get(&mut ctx), 0);
            assert_eq!(ctx.read_u32(BASE + 0x410), Some(default ^ 1));
            args(&mut ctx, &[req, 0]);
            assert_eq!(get(&mut ctx), HTTP_ERROR_INVALID_VALUE);
            args(&mut ctx, &[0x7777, BASE + 0x410]);
            assert_eq!(get(&mut ctx), HTTP_ERROR_INVALID_ID);
        }
        // A context's gzip default reaches its new templates.
        args(&mut ctx, &[http_ctx as u64, 1]);
        assert_eq!(cb_http_set_default_accept_encoding_gzip(&mut ctx), 0);
        args(&mut ctx, &[0x7777, 1]);
        assert_eq!(cb_http_set_default_accept_encoding_gzip(&mut ctx), HTTP_ERROR_INVALID_ID);
        put(&ctx, BASE + 0x420, b"agent");
        args(&mut ctx, &[http_ctx as u64, BASE + 0x420, 1, 1]);
        let tmpl2 = cb_http_create_template(&mut ctx);
        assert!(tmpl2 > 0);
        args(&mut ctx, &[tmpl2 as u64, BASE + 0x410]);
        assert_eq!(cb_http_get_accept_encoding_gzip(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x410), Some(1));
        args(&mut ctx, &[http_ctx as u64, 0]);
        assert_eq!(cb_http_set_default_accept_encoding_gzip(&mut ctx), 0);
        // Private SSL options: the known flags only, nothing stored.
        args(&mut ctx, &[tmpl, 0x2DFF]);
        assert_eq!(cb_https_option_private(&mut ctx), 0);
        args(&mut ctx, &[tmpl, 0x200]);
        assert_eq!(cb_https_option_private(&mut ctx), HTTP_ERROR_INVALID_VALUE);
        args(&mut ctx, &[0x7777, 1]);
        assert_eq!(cb_https_option_private(&mut ctx), HTTP_ERROR_INVALID_ID);
        args(&mut ctx, &[conn, 1]);
        assert_eq!(cb_http_cache_redirected_connection_enabled(&mut ctx), 0);
        args(&mut ctx, &[0x7777, 1]);
        assert_eq!(cb_http_cache_redirected_connection_enabled(&mut ctx), HTTP_ERROR_INVALID_ID);

        // The authentication callback is accepted for templates,
        // connections and requests; reading still fails without a network.
        for id in [tmpl, conn, req] {
            args(&mut ctx, &[id, 0xC0DE, BASE + 0x300]);
            assert_eq!(cb_http_set_auth_info_callback(&mut ctx), 0);
        }
        args(&mut ctx, &[0x7777, 0xC0DE, 0]);
        assert_eq!(cb_http_set_auth_info_callback(&mut ctx), HTTP_ERROR_INVALID_ID);

        // The request status callback: stored per object, cleared by NULL,
        // an unknown id refused (reference GuestHttp case).
        for id in [tmpl, conn, req] {
            args(&mut ctx, &[id, 0xC0DE, BASE + 0x300]);
            assert_eq!(cb_http_set_request_status_callback(&mut ctx), 0);
        }
        assert_eq!(http().request(req as i32).unwrap().conn.tmpl.base.status_callback, 0xC0DE);
        args(&mut ctx, &[req, 0, BASE + 0x300]);
        assert_eq!(cb_http_set_request_status_callback(&mut ctx), 0);
        let base = http().request(req as i32).unwrap().conn.tmpl.base.clone();
        assert_eq!((base.status_callback, base.status_user_arg), (0, 0));
        args(&mut ctx, &[0x7777, 0xC0DE, 0]);
        assert_eq!(cb_http_set_request_status_callback(&mut ctx), HTTP_ERROR_INVALID_ID);

        // The receive block size is accepted for every object kind.
        for id in [tmpl, conn, req] {
            args(&mut ctx, &[id, 0x4000]);
            assert_eq!(cb_http_set_recv_block_size(&mut ctx), 0);
        }
        assert_eq!(http().request(req as i32).unwrap().conn.tmpl.base.recv_block_size, 0x4000);
        args(&mut ctx, &[0x7777, 0x4000]);
        assert_eq!(cb_http_set_recv_block_size(&mut ctx), HTTP_ERROR_INVALID_ID);

        args(&mut ctx, &[req]);
        assert_eq!(cb_http_delete_request(&mut ctx), 0);
        assert_eq!(cb_http_delete_request(&mut ctx), HTTP_ERROR_INVALID_ID);
        args(&mut ctx, &[conn]);
        assert_eq!(cb_http_delete_connection(&mut ctx), 0);
        args(&mut ctx, &[tmpl]);
        assert_eq!(cb_http_delete_template(&mut ctx), 0);
        args(&mut ctx, &[http_ctx as u64]);
        assert_eq!(cb_http_term(&mut ctx), 0);
        assert_eq!(cb_http_term(&mut ctx), HTTP_ERROR_INVALID_ID);
    }

    /// A mapped guest page and a context over it that may cross threads.
    fn guest(base: u64) -> CpuContext<'static> {
        let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
        vm.map(base, 0x4000, ps5vm::MemProtect::RW).unwrap();
        CpuContext::new(vm)
    }

    fn put(ctx: &CpuContext, addr: u64, text: &[u8]) -> u64 {
        let mut b = text.to_vec();
        b.push(0);
        assert!(ctx.write_bytes(addr, &b));
        addr
    }

    /// A live libSceHttp context (with its memory pool and SSL context).
    fn http_context_for_test(ctx: &mut CpuContext, base: u64) -> u64 {
        args(ctx, &[0x8000]);
        let ssl = crate::ssl::tests_init(ctx);
        put(ctx, base, b"pool");
        let pool = crate::net::tests_pool(ctx, base);
        args(ctx, &[pool, ssl, 0x10000]);
        let id = cb_http_init(ctx);
        assert!(id > 0);
        id as u64
    }

    #[test]
    fn response_header_fields() {
        // (header, field, value offset, value): the value ends at offset +
        // length; None for an empty value (reference GuestHttp cases
        // measured on a PS5).
        let cases: &[(&[u8], &[u8], Option<(usize, &[u8])>)] = &[
            (b"HTTP/1.1 200 OK\r\nX: \tvalue \t\r\nNext: no\r\n", b"x", Some((21, b"value \t"))),
            (b"X: first\r\nX: second\r\n", b"X", Some((3, b"first"))),
            (b"X: abc\n def\n\tghi\nY: no\n", b"x", Some((3, b"abc\n def\n\tghi"))),
            (b"X: no-newline", b"X", Some((3, b"no-newline"))),
            (b"X: value\r", b"X", Some((3, b"value\r"))),
            (b"X:", b"X", None),
            (b"X: \t", b"X", None),
            (b"X:\r\n\r\n", b"X", None),
            (b"X:\r\nnext\r\n", b"X", None),
            (b"X:\r\n next\r\nY: 1\r\n", b"X", Some((5, b"next"))),
            (b"Prefix: wrong\n X: wrong\nXy: wrong\nX: right\n", b"X", Some((37, b"right\n"))),
            (b"X: a\0b\n", b"X", Some((3, b"a\0b\n"))),
        ];
        for &(header, field, expected) in cases {
            let r = find_header_field(header, field).expect("field present");
            assert_eq!(r.map(|(s, l)| (s, &header[s..s + l])), expected, "{field:?} in {header:?}");
        }
        for (header, field) in [(&b"X: yes"[..0], &b"X"[..]), (&b"X: yes"[..1], b"X"), (b"X: yes", b"Longer-Field"), (b"X: yes", b"X:")] {
            assert_eq!(find_header_field(header, field), None);
        }
        let long = vec![b'A'; 0xFFF];
        let mut header = long.clone();
        header.extend_from_slice(b": value\n");
        assert_eq!(find_header_field(&header, &long), Some(Some((0x1001, 6))));

        // Through the export: pointers into the guest header, NULL value
        // for an empty one, untouched outputs on failure.
        const BASE: u64 = 0x5E10_0000_0000;
        let mut ctx = guest(BASE);
        let text = b"HTTP/1.1 200 OK\r\nX: \tvalue \t\r\nNext: no\r\n";
        put(&ctx, BASE, text);
        put(&ctx, BASE + 0x100, b"x");
        args(&mut ctx, &[BASE, text.len() as u64, BASE + 0x100, BASE + 0x200, BASE + 0x208]);
        assert_eq!(cb_http_parse_response_header(&mut ctx), 28);
        assert_eq!((ctx.read_u64(BASE + 0x200), ctx.read_u64(BASE + 0x208)), (Some(BASE + 21), Some(7)));
        put(&ctx, BASE, b"X:");
        args(&mut ctx, &[BASE, 2, BASE + 0x100, BASE + 0x200, BASE + 0x208]);
        assert_eq!(cb_http_parse_response_header(&mut ctx), 0);
        assert_eq!((ctx.read_u64(BASE + 0x200), ctx.read_u64(BASE + 0x208)), (Some(0), Some(0)));
        ctx.write_u64(BASE + 0x200, 0x1111);
        put(&ctx, BASE + 0x100, b"not-present");
        assert_eq!(cb_http_parse_response_header(&mut ctx), HTTP_ERROR_PARSE_NOT_FOUND);
        assert_eq!(ctx.read_u64(BASE + 0x200), Some(0x1111));
        // A zero-length header at an unmapped address is never read.
        args(&mut ctx, &[BASE + 0x10_0000, 0, BASE + 0x100, BASE + 0x200, BASE + 0x208]);
        assert_eq!(cb_http_parse_response_header(&mut ctx), HTTP_ERROR_PARSE_NOT_FOUND);
        args(&mut ctx, &[0, 1, BASE + 0x100, BASE + 0x200, BASE + 0x208]);
        assert_eq!(cb_http_parse_response_header(&mut ctx), HTTP_ERROR_PARSE_INVALID_RESPONSE);
        for bad in [[BASE, 2, 0, BASE + 0x200, BASE + 0x208], [BASE, 2, BASE + 0x100, 0, BASE + 0x208], [BASE, 2, BASE + 0x100, BASE + 0x200, 0]] {
            args(&mut ctx, &bad);
            assert_eq!(cb_http_parse_response_header(&mut ctx), HTTP_ERROR_PARSE_INVALID_VALUE);
        }
    }

    #[test]
    fn status_lines() {
        let ok = parse_status_line(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n").unwrap();
        assert_eq!(ok, StatusLine { major: 1, minor: 1, code: 200, phrase: 12, phrase_len: 3, consumed: 17 });
        let lf = parse_status_line(b"HTTP/10.25 404 Not Found\n").unwrap();
        assert_eq!(lf, StatusLine { major: 10, minor: 25, code: 404, phrase: 14, phrase_len: 10, consumed: 25 });
        let bare = parse_status_line(b"HTTP/2.0 204\r\n").unwrap();
        assert_eq!(bare, StatusLine { major: 2, minor: 0, code: 204, phrase: 12, phrase_len: 0, consumed: 14 });
        for bad in [
            &b"HTTX/1.1 200 OK\n"[..],
            b"HTTP/1.1",
            b"HTTP/123",
            b"HTTP/x.1 200 OK\n",
            b"HTTP/1 200 OK\n",
            b"HTTP/1. 200 OK\n",
            b"HTTP/1.1/200 OK\n",
            b"HTTP/1.1 2x0 OK\n",
            b"HTTP/1.1 20",
            &b"HTTP/1.1 200 OK\r\n"[..15],
            &b"HTTP/1.1 200 OK\r\n"[..16],
        ] {
            assert!(parse_status_line(bad).is_err(), "{bad:?}");
        }

        const BASE: u64 = 0x5E10_0010_0000;
        let mut ctx = guest(BASE);
        let line = b"HTTP/1.1 200 OK\r\n";
        put(&ctx, BASE, line);
        // The seventh argument (size_t* phraseLen) is on the stack.
        ctx.set(Rsp, BASE + 0x800);
        ctx.write_u64(BASE + 0x808, BASE + 0x320);
        args(&mut ctx, &[BASE, 17, BASE + 0x300, BASE + 0x304, BASE + 0x308, BASE + 0x310]);
        assert_eq!(cb_http_parse_status_line(&mut ctx), 17);
        assert_eq!(ctx.read_u32(BASE + 0x308), Some(200));
        assert_eq!((ctx.read_u64(BASE + 0x310), ctx.read_u64(BASE + 0x320)), (Some(BASE + 12), Some(3)));
        put(&ctx, BASE, b"HTTX/1.1 200 OK\n");
        ctx.write_u32(BASE + 0x300, u32::MAX);
        assert_eq!(cb_http_parse_status_line(&mut ctx), HTTP_ERROR_PARSE_INVALID_RESPONSE);
        assert_eq!((ctx.read_u32(BASE + 0x300), ctx.read_u32(BASE + 0x304)), (Some(0), Some(0)));
        args(&mut ctx, &[0, 17, BASE + 0x300, BASE + 0x304, BASE + 0x308, BASE + 0x310]);
        assert_eq!(cb_http_parse_status_line(&mut ctx), HTTP_ERROR_PARSE_INVALID_RESPONSE);
        args(&mut ctx, &[BASE, 17, 0, BASE + 0x304, BASE + 0x308, BASE + 0x310]);
        assert_eq!(cb_http_parse_status_line(&mut ctx), HTTP_ERROR_PARSE_INVALID_VALUE);
        ctx.write_u64(BASE + 0x808, 0);
        args(&mut ctx, &[BASE, 17, BASE + 0x300, BASE + 0x304, BASE + 0x308, BASE + 0x310]);
        assert_eq!(cb_http_parse_status_line(&mut ctx), HTTP_ERROR_PARSE_INVALID_VALUE);
    }

    #[test]
    fn uri_unescape_cases() {
        let cases: &[(&[u8], &[u8])] = &[
            (b"", b""),
            (b"plain+text", b"plain+text"),
            (b"a%20b%2F~%c3%a9", b"a b/~\xC3\xA9"),
            (b"%41%4a%4F%ff", b"AJO\xFF"),
            (b"%2520", b"%20"),
            (b"a%00b", b"a\0b"),
        ];
        for &(input, expected) in cases {
            assert_eq!(uri_unescape(input), Ok(expected.to_vec()), "{input:?}");
        }
        for bad in [&b"%"[..], b"%1", b"a%1g", b"%gg", b"%+1"] {
            assert_eq!(uri_unescape(bad), Err(HTTP_ERROR_INVALID_URL), "{bad:?}");
        }
        let bytes: Vec<u8> = (1..=255).collect();
        assert_eq!(uri_unescape(&uri_escape(&bytes)), Ok(bytes));

        const BASE: u64 = 0x5E10_0020_0000;
        let mut ctx = guest(BASE);
        put(&ctx, BASE, b"%41%2f%2520");
        ctx.write_bytes(BASE + 0x100, &[b'Z'; 16]);
        args(&mut ctx, &[0, BASE + 0x80, 0, BASE]);
        assert_eq!(cb_http_uri_unescape(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE + 0x80), Some(6));
        // A short buffer gets the bytes that fit, no terminator, and the
        // size needed is not written.
        ctx.write_u64(BASE + 0x80, 99);
        args(&mut ctx, &[BASE + 0x100, BASE + 0x80, 5, BASE]);
        assert_eq!(cb_http_uri_unescape(&mut ctx), HTTP_ERROR_OUT_OF_MEMORY);
        assert_eq!(ctx.read_bytes(BASE + 0x100, 6).unwrap(), b"A/%20Z");
        assert_eq!(ctx.read_u64(BASE + 0x80), Some(99));
        args(&mut ctx, &[0, 0, 0, BASE]);
        assert_eq!(cb_http_uri_unescape(&mut ctx), HTTP_ERROR_INVALID_VALUE);
        put(&ctx, BASE + 0x200, b"%4");
        args(&mut ctx, &[BASE + 0x100, BASE + 0x80, 64, BASE + 0x200]);
        assert_eq!(cb_http_uri_unescape(&mut ctx), HTTP_ERROR_INVALID_URL);
        put(&ctx, BASE, b"%41%2f%2520");
        // In place.
        args(&mut ctx, &[BASE, BASE + 0x80, 64, BASE]);
        assert_eq!(cb_http_uri_unescape(&mut ctx), 0);
        assert_eq!(ctx.read_cstring(BASE, 64).as_deref(), Some("A/%20"));
        ctx.write_u64(BASE + 0x80, 123);
        args(&mut ctx, &[BASE + 0x100, BASE + 0x80, 64, 0]);
        assert_eq!(cb_http_uri_unescape(&mut ctx), HTTP_ERROR_INVALID_VALUE);
        assert_eq!(ctx.read_u64(BASE + 0x80), Some(123));
    }

    #[test]
    fn uri_sweep_paths() {
        let sweep_n = |src: &[u8], length: usize| {
            let mut dst = vec![b'x'; length.max(1) + 1];
            uri_sweep_path(&mut dst, src, length);
            let nul = dst.iter().position(|&c| c == 0).unwrap();
            String::from_utf8(dst[..nul].to_vec()).unwrap()
        };
        let sweep = |src: &[u8]| {
            let mut dst = vec![b'x'; src.len().max(1) + 1];
            uri_sweep_path(&mut dst, src, src.len());
            let nul = dst.iter().position(|&c| c == 0).unwrap();
            String::from_utf8(dst[..nul].to_vec()).unwrap()
        };
        for (src, expected) in [
            ("foo/../bar", "foo/../bar"),
            ("/foo/../bar", "/bar"),
            ("/foo/./bar", "/foo/bar"),
            ("/foo/.", "/foo/."),
            ("/foo/..", "/foo/.."),
            ("/foo/bar/../foo/././../../../test/index.html", "/test/index.html"),
            ("/", "/"),
            ("/a/b/c/../../d/", "/a/d/"),
            ("/../a", "/a"),
            ("/a/..b/.c", "/a/..b/.c"),
        ] {
            assert_eq!(sweep(src.as_bytes()), expected, "{src}");
        }
        // srcSize limits the source; a ".." test looks one byte past it.
        assert_eq!(sweep_n(b"/a/b/../c", 5), "/a/b/");
        assert_eq!(sweep_n(b"/ab/c", 2), "/a");
        assert_eq!(sweep_n(b"/a/./b", 4), "/a/.");
        assert_eq!(sweep_n(b"/a/b/../c", 7), "/a/");
        assert_eq!(sweep_n(b"", 0), "");

        const BASE: u64 = 0x5E10_0030_0000;
        let mut ctx = guest(BASE);
        args(&mut ctx, &[0, 0, 0]);
        assert_eq!(cb_http_uri_sweep_path(&mut ctx), 0);
        put(&ctx, BASE, b"/foo");
        args(&mut ctx, &[0, BASE, 5]);
        assert_eq!(cb_http_uri_sweep_path(&mut ctx), HTTP_ERROR_INVALID_VALUE);
        args(&mut ctx, &[BASE + 0x100, 0, 5]);
        assert_eq!(cb_http_uri_sweep_path(&mut ctx), HTTP_ERROR_INVALID_VALUE);
        // Only srcSize bytes of the output are written.
        put(&ctx, BASE, b"/");
        ctx.write_bytes(BASE + 0x100, b"xx");
        args(&mut ctx, &[BASE + 0x100, BASE, 1]);
        assert_eq!(cb_http_uri_sweep_path(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x100, 2).unwrap(), b"/\0");
        put(&ctx, BASE, b"/a/b/../c");
        args(&mut ctx, &[BASE + 0x100, BASE, 8]);
        assert_eq!(cb_http_uri_sweep_path(&mut ctx), 0);
        assert_eq!(ctx.read_cstring(BASE + 0x100, 16).as_deref(), Some("/a/"));
    }

    #[test]
    fn uri_merges() {
        let base = b"http://foo.com/foo/index.html";
        let base_size = 5 + 1 + 1 + 8 + 16 + 1 + 1 + 2;
        let relative = |url: &[u8], rel: &[u8]| match uri_merge(url, rel).unwrap() {
            Merged::Relative { text, size, .. } => (String::from_utf8(text).unwrap(), size),
            Merged::Absolute { .. } => panic!("absolute"),
        };
        assert_eq!(relative(base, b"./default.html"), ("http://foo.com/foo/./default.html".into(), base_size + 2 * (29 + 14)));
        for (url, rel, expected) in [
            (&base[..], &b"../sibling.html"[..], "http://foo.com/foo/../sibling.html"),
            (base, b"", "http://foo.com/foo/"),
            (base, b"/root.html", "http://foo.com/root.html"),
            (base, b"a?q=1#f", "http://foo.com/foo/a?q=1#f"),
            (b"https://u:p@foo.com:8443/a/b?x=1#top", b"c", "https://u:p@foo.com:8443/a/c"),
            (b"http://foo.com", b"x", "http://foo.com/x"),
            (b"http://foo.com:80/", b"x", "http://foo.com/x"),
            (b"http://foo.com/a/b", b"mailto:x", "http://foo.com/a/mailto:x"),
            (base, b"a%20b.html", "http://foo.com/foo/a%20b.html"),
            (base, b"~user/x", "http://foo.com/foo/~user/x"),
            (base, b"file(1).png", "http://foo.com/foo/file(1).png"),
            (base, b"a+b=c;d!e", "http://foo.com/foo/a+b=c;d!e"),
        ] {
            assert_eq!(relative(url, rel).0, expected);
        }
        let absolute = base_size + 2 * (29 + 20);
        assert_eq!(uri_merge(base, b"http://bar.com/other"), Some(Merged::Absolute { size: absolute, text: b"http://bar.com/other".to_vec() }));
        assert!(matches!(uri_merge(base, b"//bar.com/x"), Some(Merged::Absolute { .. })));
        assert_eq!(uri_merge(b"http://bad host/", b"./x"), None);
        assert_eq!(uri_merge(base, b"http://bad host/"), None);

        const BASE: u64 = 0x5E10_0040_0000;
        let mut ctx = guest(BASE);
        put(&ctx, BASE, base);
        put(&ctx, BASE + 0x40, b"./default.html");
        put(&ctx, BASE + 0x80, b"http://bar.com/other");
        let out = BASE + 0x1000;
        let need = (base_size + 2 * (29 + 14)) as u64;
        args(&mut ctx, &[0, BASE, BASE + 0x40, BASE + 0x200, 0, 0]);
        assert_eq!(cb_http_uri_merge(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE + 0x200), Some(need));
        ctx.write_bytes(out, &[b'Z'; 512]);
        args(&mut ctx, &[out, BASE, BASE + 0x40, BASE + 0x200, need - 1, 0]);
        assert_eq!(cb_http_uri_merge(&mut ctx), HTTP_ERROR_OUT_OF_MEMORY);
        assert_eq!(ctx.read_bytes(out, 1).unwrap(), b"Z");
        args(&mut ctx, &[out, BASE, BASE + 0x40, 0, need, 0]);
        assert_eq!(cb_http_uri_merge(&mut ctx), 0);
        assert_eq!(ctx.read_cstring(out, 512).as_deref(), Some("http://foo.com/foo/./default.html"));
        // An absolute reference is copied and zero-padded to the size.
        ctx.write_bytes(out, &[b'Z'; 512]);
        args(&mut ctx, &[out, BASE, BASE + 0x80, BASE + 0x200, 512, 0]);
        assert_eq!(cb_http_uri_merge(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE + 0x200), Some(21));
        let written = ctx.read_bytes(out, absolute + 1).unwrap();
        assert_eq!(&written[..20], b"http://bar.com/other");
        assert!(written[20..absolute].iter().all(|&b| b == 0));
        assert_eq!(written[absolute], b'Z');
        ctx.write_u64(BASE + 0x200, 123);
        for bad in [[out, 0, BASE + 0x40, BASE + 0x200, 512, 0], [out, BASE, 0, BASE + 0x200, 512, 0], [out, BASE, BASE + 0x40, BASE + 0x200, 512, 1]] {
            args(&mut ctx, &bad);
            assert_eq!(cb_http_uri_merge(&mut ctx), HTTP_ERROR_INVALID_VALUE);
        }
        put(&ctx, BASE + 0x80, b"http://bad host/");
        args(&mut ctx, &[out, BASE, BASE + 0x80, BASE + 0x200, 512, 0]);
        assert_eq!(cb_http_uri_merge(&mut ctx), HTTP_ERROR_INVALID_URL);
        assert_eq!(ctx.read_u64(BASE + 0x200), Some(123));
    }

    #[test]
    fn epoll_abort_and_destroy_wake_waiters() {
        const BASE: u64 = 0x5E10_0050_0000;
        let mut ctx = guest(BASE);
        let context = http_context_for_test(&mut ctx, BASE);
        let (eh_out, events) = (BASE + 0x100, BASE + 0x200);
        args(&mut ctx, &[0x7777, eh_out]);
        assert_eq!(cb_http_create_epoll(&mut ctx), HTTP_ERROR_INVALID_ID);
        args(&mut ctx, &[context, 0]);
        assert_eq!(cb_http_create_epoll(&mut ctx), HTTP_ERROR_INVALID_VALUE);
        args(&mut ctx, &[context, eh_out]);
        assert_eq!(cb_http_create_epoll(&mut ctx), 0);
        let eh = ctx.read_u64(eh_out).unwrap();
        args(&mut ctx, &[eh, events, 2, 0]);
        assert_eq!(cb_http_wait_request(&mut ctx), 0);
        args(&mut ctx, &[0]);
        assert_eq!(cb_http_abort_wait_request(&mut ctx), HTTP_ERROR_INVALID_VALUE);
        args(&mut ctx, &[events]);
        assert_eq!(cb_http_abort_wait_request(&mut ctx), HTTP_ERROR_INVALID_ID);
        args(&mut ctx, &[events, events, 2, -1i64 as u64]);
        assert_eq!(cb_http_wait_request(&mut ctx), HTTP_ERROR_INVALID_ID);
        args(&mut ctx, &[context, 0]);
        assert_eq!(cb_http_destroy_epoll(&mut ctx), HTTP_ERROR_INVALID_VALUE);
        args(&mut ctx, &[context, events]);
        assert_eq!(cb_http_destroy_epoll(&mut ctx), HTTP_ERROR_INVALID_ID);
        args(&mut ctx, &[0x7777, eh]);
        assert_eq!(cb_http_destroy_epoll(&mut ctx), HTTP_ERROR_INVALID_ID);

        // A waiter without a timeout blocks until woken.
        let wait_woken_by = |ctx: &mut CpuContext<'static>, eh: u64, wake: &dyn Fn(&mut CpuContext<'static>)| {
            let mut waiter_ctx = CpuContext::new(ctx.vm);
            let waiter = std::thread::spawn(move || {
                args(&mut waiter_ctx, &[eh, events, 2, -1i64 as u64]);
                cb_http_wait_request(&mut waiter_ctx)
            });
            std::thread::sleep(std::time::Duration::from_millis(100));
            assert!(!waiter.is_finished());
            wake(ctx);
            waiter.join().unwrap()
        };
        let abort = move |ctx: &mut CpuContext<'static>| {
            args(ctx, &[eh]);
            assert_eq!(cb_http_abort_wait_request(ctx), 0);
        };
        assert_eq!(wait_woken_by(&mut ctx, eh, &abort), HTTP_ERROR_ABORTED);
        // The abort sticks to the handle.
        for timeout in [0, -1i64 as u64] {
            args(&mut ctx, &[eh, events, 2, timeout]);
            assert_eq!(cb_http_wait_request(&mut ctx), HTTP_ERROR_ABORTED);
        }
        args(&mut ctx, &[context, eh]);
        assert_eq!(cb_http_destroy_epoll(&mut ctx), 0);
        args(&mut ctx, &[eh]);
        assert_eq!(cb_http_abort_wait_request(&mut ctx), HTTP_ERROR_INVALID_ID);

        // Destroying the handle ends the wait without events.
        args(&mut ctx, &[context, eh_out]);
        assert_eq!(cb_http_create_epoll(&mut ctx), 0);
        let eh = ctx.read_u64(eh_out).unwrap();
        let destroy = move |ctx: &mut CpuContext<'static>| {
            args(ctx, &[context, eh]);
            assert_eq!(cb_http_destroy_epoll(ctx), 0);
        };
        assert_eq!(wait_woken_by(&mut ctx, eh, &destroy), 0);

        // So does terminating the owning context.
        args(&mut ctx, &[context, eh_out]);
        assert_eq!(cb_http_create_epoll(&mut ctx), 0);
        let eh = ctx.read_u64(eh_out).unwrap();
        let term = move |ctx: &mut CpuContext<'static>| {
            args(ctx, &[context]);
            assert_eq!(cb_http_term(ctx), 0);
        };
        assert_eq!(wait_woken_by(&mut ctx, eh, &term), 0);
    }

    #[test]
    fn https_and_option_setters() {
        const BASE: u64 = 0x5E10_0060_0000;
        let mut ctx = guest(BASE);
        let context = http_context_for_test(&mut ctx, BASE);
        put(&ctx, BASE + 0x20, b"agent");
        args(&mut ctx, &[context, BASE + 0x20, 1, 0]);
        let tmpl = cb_http_create_template(&mut ctx) as u64;

        for enable in [0, 1] {
            args(&mut ctx, &[tmpl, enable]);
            assert_eq!(cb_http_set_inflate_gzip_enabled(&mut ctx), 0);
        }
        for enable in [2, u32::MAX as u64] {
            args(&mut ctx, &[tmpl, enable]);
            assert_eq!(cb_http_set_inflate_gzip_enabled(&mut ctx), HTTP_ERROR_INVALID_VALUE);
        }
        args(&mut ctx, &[tmpl, 8192]);
        assert_eq!(cb_http_set_response_header_max_size(&mut ctx), 0);
        args(&mut ctx, &[tmpl, 0xC0DE, 0x1234]);
        assert_eq!(cb_http_set_redirect_callback(&mut ctx), 0);
        args(&mut ctx, &[tmpl, 0xC0DF, 0x5678]);
        assert_eq!(cb_http_set_cookie_recv_callback(&mut ctx), 0);
        args(&mut ctx, &[tmpl, 0]);
        assert_eq!(cb_https_set_ssl_version(&mut ctx), 0);
        args(&mut ctx, &[tmpl, 0x07]);
        assert_eq!(cb_https_disable_option(&mut ctx), 0);
        args(&mut ctx, &[tmpl, 0x01]);
        assert_eq!(cb_https_enable_option(&mut ctx), 0);
        {
            let mut st = http();
            let b = &slot(&mut st.templates, tmpl as i32, ID_TYPE_TEMPLATE).unwrap().base;
            assert_eq!((b.response_header_max_size, b.redirect_callback, b.cookie_recv_user_arg), (8192, 0xC0DE, 0x5678));
            assert_eq!(b.ssl_flags, 0xA1);
            assert!(b.inflate_gzip);
        }
        for cb in [cb_http_set_response_header_max_size, cb_http_set_redirect_callback, cb_https_enable_option] {
            args(&mut ctx, &[0x7777, 1, 0]);
            assert_eq!(cb(&mut ctx), HTTP_ERROR_INVALID_ID);
        }

        ctx.write_u32(BASE + 0x100, u32::MAX);
        ctx.write_u32(BASE + 0x104, u32::MAX);
        args(&mut ctx, &[tmpl, BASE + 0x100, BASE + 0x104]);
        assert_eq!(cb_https_get_ssl_error(&mut ctx), 0);
        assert_eq!((ctx.read_u32(BASE + 0x100), ctx.read_u32(BASE + 0x104)), (Some(0), Some(0)));
        for bad in [[tmpl, 0, BASE + 0x104], [tmpl, BASE + 0x100, 0]] {
            args(&mut ctx, &bad);
            assert_eq!(cb_https_get_ssl_error(&mut ctx), HTTP_ERROR_INVALID_VALUE);
        }

        for cb in [cb_http_redirect_cache_flush, cb_https_unload_cert, cb_http_store_flush] {
            args(&mut ctx, &[context]);
            assert_eq!(cb(&mut ctx), 0);
            args(&mut ctx, &[tmpl]);
            assert_eq!(cb(&mut ctx), HTTP_ERROR_INVALID_ID);
        }
        args(&mut ctx, &[context, 0, 0, 0, 0]);
        assert_eq!(cb_https_load_cert(&mut ctx), 0);
        ctx.write_bytes(BASE + 0x200, &[0xAA; 16]);
        for cb in [cb_https_ca_list as HleCallback, cb_https_ca_list] {
            args(&mut ctx, &[context, BASE + 0x200]);
            assert_eq!(cb(&mut ctx), 0);
            assert_eq!(ctx.read_bytes(BASE + 0x200, 16).unwrap(), vec![0; 16]);
            args(&mut ctx, &[context, 0]);
            assert_eq!(cb(&mut ctx), HTTP_ERROR_INVALID_VALUE);
        }
        args(&mut ctx, &[context]);
        assert_eq!(cb_http_term(&mut ctx), 0);
    }

    #[test]
    fn http2_cookie_boxes() {
        const BASE: u64 = 0x5E10_0070_0000;
        let mut ctx = guest(BASE);
        args(&mut ctx, &[1, 1, 0x2000, 4]);
        let context = cb_http2_init(&mut ctx) as u64;
        args(&mut ctx, &[0x7777]);
        assert_eq!(cb_http2_create_cookie_box(&mut ctx), HTTP2_ERROR_INVALID_ID);
        args(&mut ctx, &[context]);
        let boxed = cb_http2_create_cookie_box(&mut ctx) as u64;
        let discarded = cb_http2_create_cookie_box(&mut ctx) as u64;
        assert!(boxed > 0 && discarded != boxed && boxed != context);
        args(&mut ctx, &[discarded]);
        assert_eq!(cb_http2_delete_cookie_box(&mut ctx), 0);
        assert_eq!(cb_http2_delete_cookie_box(&mut ctx), HTTP2_ERROR_INVALID_ID);
        // A context id is not a cookie box.
        args(&mut ctx, &[context]);
        assert_eq!(cb_http2_delete_cookie_box(&mut ctx), HTTP2_ERROR_INVALID_ID);
        // Terminating the context deletes its boxes.
        assert_eq!(cb_http2_term(&mut ctx), 0);
        args(&mut ctx, &[boxed]);
        assert_eq!(cb_http2_delete_cookie_box(&mut ctx), HTTP2_ERROR_INVALID_ID);
    }

    #[test]
    fn http2_offline() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D0A_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        args(&mut ctx, &[1, 2, 0x10000, 4]);
        let c = cb_http2_init(&mut ctx) as u64;
        ctx.write_bytes(BASE, b"agent\0GET\0https://example.com/\0");
        args(&mut ctx, &[c, BASE, 2, 0]);
        let t = cb_http2_create_template(&mut ctx) as u64;
        // The pool size given at init, nothing in use.
        ctx.write_bytes(BASE + 0x400, &[1; 32]);
        args(&mut ctx, &[c, BASE + 0x400]);
        assert_eq!(cb_http2_get_memory_pool_stats(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE + 0x400), Some(0x10000));
        assert!(ctx.read_bytes(BASE + 0x408, 24).unwrap().iter().all(|&b| b == 0));
        args(&mut ctx, &[c, 0]);
        assert_eq!(cb_http2_get_memory_pool_stats(&mut ctx), HTTP2_ERROR_NULL_POINTER);
        args(&mut ctx, &[0x7777, 0]);
        assert_eq!(cb_http2_get_memory_pool_stats(&mut ctx), HTTP2_ERROR_NULL_POINTER, "NULL output checked first");
        ctx.write_bytes(BASE + 0x400, &[0x5A; 0x40]);
        args(&mut ctx, &[0x7777, BASE + 0x400]);
        assert_eq!(cb_http2_get_memory_pool_stats(&mut ctx), HTTP2_ERROR_INVALID_ID);
        assert_eq!(ctx.read_bytes(BASE + 0x400, 0x40).unwrap(), [vec![0; 32], vec![0x5A; 32]].concat(), "an unknown context zeroes the output");
        args(&mut ctx, &[t, 3]);
        assert_eq!(cb_http2_set_resolve_retry(&mut ctx), 0);
        assert_eq!(http2().templates[&(t as i32)].options.resolve_retry, 3);
        args(&mut ctx, &[t, 0, BASE + 10, 0]);
        assert_eq!(cb_http2_create_request_with_url(&mut ctx), HTTP2_ERROR_NULL_POINTER);
        args(&mut ctx, &[0x7777, BASE + 6, BASE + 10, 0]);
        assert_eq!(cb_http2_create_request_with_url(&mut ctx), HTTP2_ERROR_INVALID_ID);
        args(&mut ctx, &[t, BASE + 6, BASE + 10, 0]);
        let r = cb_http2_create_request_with_url(&mut ctx) as u64;
        args(&mut ctx, &[r, 1]);
        assert_eq!(cb_http2_set_inflate_gzip_enabled(&mut ctx), 0);
        args(&mut ctx, &[r, 1000]);
        assert_eq!(cb_http2_set_timeout(&mut ctx), 0);
        assert!(http2().requests[&(r as i32)].options.inflate_gzip);

        args(&mut ctx, &[r, BASE + 0x100]);
        assert_eq!(cb_http2_get_status_code(&mut ctx), HTTP2_ERROR_BEFORE_SEND);
        args(&mut ctx, &[r, 0, 0, 0, 0]);
        assert_eq!(cb_http2_send_request_async(&mut ctx), 0);
        args(&mut ctx, &[r, BASE + 0x200, 0, 0]);
        assert_eq!(cb_http2_wait_async(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x200), Some(0));
        assert_eq!(ctx.read_u32(BASE + 0x204), Some(r as u32));
        assert_eq!(ctx.read_u32(BASE + 0x208), Some(HTTP2_ERROR_TIMEOUT as u32));
        // With a kqueue option the completion fires the option's user event.
        let eq = crate::sync::equeue::create();
        assert!(crate::sync::equeue::register_event(eq, 7, crate::sync::equeue::FILTER_USER, 0, 0, 0));
        ctx.write_u64(BASE + 0x500, eq);
        ctx.write_u32(BASE + 0x508, 7);
        ctx.write_u64(BASE + 0x510, 0xDA7A);
        ctx.write_u64(BASE + 0x518, 0);
        args(&mut ctx, &[r, 0, 0, BASE + 0x500, 0]);
        assert_eq!(cb_http2_send_request_async(&mut ctx), 0);
        let events = crate::sync::equeue::wait(eq, 4, Some(0)).expect("completion event");
        assert_eq!((events.len(), events[0].ident, events[0].udata), (1, 7, 0xDA7A));
        crate::sync::equeue::delete(eq);
        args(&mut ctx, &[r, BASE + 0x300, 16, 0, 0]);
        assert_eq!(cb_http2_read_data_async(&mut ctx), 0);
        args(&mut ctx, &[r, BASE + 0x200, 0, 0]);
        assert_eq!(cb_http2_wait_async(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x200), Some(1));
        assert_eq!(ctx.read_u32(BASE + 0x208), Some(HTTP2_ERROR_TIMEOUT as u32));
        args(&mut ctx, &[r, BASE + 0x300, 16]);
        assert_eq!(cb_http2_read_data(&mut ctx), HTTP2_ERROR_TIMEOUT);
        args(&mut ctx, &[r, BASE + 0x100, BASE + 0x108]);
        assert_eq!(cb_http2_get_response_content_length(&mut ctx), HTTP2_ERROR_TIMEOUT);
        assert_eq!(ctx.read_u32(BASE + 0x100), Some(u32::MAX));
        args(&mut ctx, &[t]);
        assert_eq!(cb_http2_delete_template(&mut ctx), 0);
        args(&mut ctx, &[r]);
        assert_eq!(cb_http2_delete_request(&mut ctx), HTTP2_ERROR_INVALID_ID);
        args(&mut ctx, &[c]);
        assert_eq!(cb_http2_term(&mut ctx), 0);
        assert_eq!(cb_http2_term(&mut ctx), HTTP2_ERROR_INVALID_ID);
    }

    /// WebSocket requests: created on a template with the callbacks and the
    /// receive buffer from the stack arguments, configured, and refused
    /// frames until a handshake succeeds, which it never does offline.
    #[test]
    fn http2_websocket_offline() {
        const BASE: u64 = 0x5D0E_0000_0000;
        let mut ctx = guest(BASE);
        args(&mut ctx, &[1, 2, 0x10000, 4]);
        let c = cb_http2_init(&mut ctx) as u64;
        ctx.write_bytes(BASE, b"agent\0wss://example.com/chat\0GET\0https://example.com/\0");
        args(&mut ctx, &[c, BASE, 2, 0]);
        let t = cb_http2_create_template(&mut ctx) as u64;
        let url = BASE + 6;
        // Stack arguments: recvBuffer, recvBufferSize, userData, option.
        let rsp = BASE + 0x800;
        for (i, v) in [BASE + 0x1000, 0x2000, 0xDA7A, 0].iter().enumerate() {
            ctx.write_u64(rsp + 8 + 8 * i as u64, *v);
        }
        ctx.set(Rsp, rsp);
        args(&mut ctx, &[t, 0, 0x100, 0x200, 0x300, 0x400]);
        assert_eq!(cb_http2_websocket_create_request(&mut ctx), HTTP2_ERROR_NULL_POINTER);
        args(&mut ctx, &[0x7777, url, 0x100, 0x200, 0x300, 0x400]);
        assert_eq!(cb_http2_websocket_create_request(&mut ctx), HTTP2_ERROR_INVALID_ID);
        args(&mut ctx, &[t, url, 0x100, 0x200, 0x300, 0x400]);
        let ws = cb_http2_websocket_create_request(&mut ctx);
        assert!(ws > 0);
        let ws = ws as u64;
        {
            let st = http2();
            let r = &st.requests[&(ws as i32)];
            assert_eq!((r.method.as_str(), r.url.as_str()), ("GET", "wss://example.com/chat"));
            let w = r.websocket.as_ref().unwrap();
            assert_eq!(
                (w.text_callback, w.data_callback, w.close_callback, w.error_callback),
                (0x100, 0x200, 0x300, 0x400)
            );
            assert_eq!((w.recv_buffer, w.recv_buffer_size, w.user_data), (BASE + 0x1000, 0x2000, 0xDA7A));
        }
        // The caller adds its subprotocol header to the handshake.
        ctx.write_bytes(BASE + 0x100, b"Sec-WebSocket-Protocol\0chat\0");
        args(&mut ctx, &[ws, BASE + 0x100, BASE + 0x117, 0]);
        assert_eq!(cb_http2_add_request_header(&mut ctx), 0);

        args(&mut ctx, &[ws, 30_000_000]);
        assert_eq!(cb_http2_websocket_set_ping_interval(&mut ctx), 0);
        args(&mut ctx, &[ws, 10_000_000]);
        assert_eq!(cb_http2_websocket_set_ping_timeout(&mut ctx), 0);
        {
            let st = http2();
            let w = st.requests[&(ws as i32)].websocket.clone().unwrap();
            assert_eq!((w.ping_interval_us, w.ping_timeout_us), (30_000_000, 10_000_000));
        }
        // A plain request is not a WebSocket.
        args(&mut ctx, &[t, url + 23, url + 27, 0]);
        let plain = cb_http2_create_request_with_url(&mut ctx) as u64;
        for id in [plain, 0x7777] {
            args(&mut ctx, &[id, 1]);
            assert_eq!(cb_http2_websocket_set_ping_interval(&mut ctx), HTTP2_ERROR_INVALID_ID);
            assert_eq!(cb_http2_websocket_set_ping_timeout(&mut ctx), HTTP2_ERROR_INVALID_ID);
            args(&mut ctx, &[id, BASE, 0, 0, 0]);
            assert_eq!(cb_http2_websocket_send_text_message_async(&mut ctx), HTTP2_ERROR_INVALID_ID);
            args(&mut ctx, &[id, BASE, 4, 0, 0, 0]);
            assert_eq!(cb_http2_websocket_send_data_message_async(&mut ctx), HTTP2_ERROR_INVALID_ID);
            args(&mut ctx, &[id, 0, 0, 0, 0, 0]);
            assert_eq!(cb_http2_websocket_close_async(&mut ctx), HTTP2_ERROR_INVALID_ID);
        }

        // NULL message pointers.
        args(&mut ctx, &[ws, 0, 0, 0, 0]);
        assert_eq!(cb_http2_websocket_send_text_message_async(&mut ctx), HTTP2_ERROR_NULL_POINTER);
        args(&mut ctx, &[ws, 0, 4, 0, 0, 0]);
        assert_eq!(cb_http2_websocket_send_data_message_async(&mut ctx), HTTP2_ERROR_NULL_POINTER);

        // Before the handshake no frame can go out.
        args(&mut ctx, &[ws, BASE, 0, 0, 0]);
        assert_eq!(cb_http2_websocket_send_text_message_async(&mut ctx), HTTP2_ERROR_BEFORE_SEND);
        args(&mut ctx, &[ws, 0, 0, 0, 0, 0]);
        assert_eq!(cb_http2_websocket_send_data_message_async(&mut ctx), HTTP2_ERROR_BEFORE_SEND);
        ctx.write_bytes(BASE + 0x180, &1000u16.to_le_bytes());
        args(&mut ctx, &[ws, BASE + 0x180, 0, 0, 0, 0]);
        assert_eq!(cb_http2_websocket_close_async(&mut ctx), HTTP2_ERROR_BEFORE_SEND);

        // The handshake times out offline; frames then fail with its error.
        args(&mut ctx, &[ws, 0, 0, 0, 0]);
        assert_eq!(cb_http2_send_request_async(&mut ctx), 0);
        args(&mut ctx, &[ws, BASE + 0x200, 0, 0]);
        assert_eq!(cb_http2_wait_async(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x208), Some(HTTP2_ERROR_TIMEOUT as u32));
        args(&mut ctx, &[ws, BASE, 0, 0, 0]);
        assert_eq!(cb_http2_websocket_send_text_message_async(&mut ctx), HTTP2_ERROR_TIMEOUT);
        args(&mut ctx, &[ws, BASE, 4, 0, 0, 0]);
        assert_eq!(cb_http2_websocket_send_data_message_async(&mut ctx), HTTP2_ERROR_TIMEOUT);
        args(&mut ctx, &[ws, BASE + 0x180, 0, 0, 0, 0]);
        assert_eq!(cb_http2_websocket_close_async(&mut ctx), HTTP2_ERROR_TIMEOUT);

        args(&mut ctx, &[ws]);
        assert_eq!(cb_http2_delete_request(&mut ctx), 0);
        args(&mut ctx, &[ws, 1]);
        assert_eq!(cb_http2_websocket_set_ping_interval(&mut ctx), HTTP2_ERROR_INVALID_ID);
        args(&mut ctx, &[c]);
        assert_eq!(cb_http2_term(&mut ctx), 0);
    }

    /// sceHttp2SetRequestNoContentLength sets the flag on a request; an
    /// unknown request id is HTTP2_ERROR_INVALID_ID.
    #[test]
    fn http2_set_request_no_content_length() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D0D_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        args(&mut ctx, &[1, 2, 0x10000, 4]);
        let c = cb_http2_init(&mut ctx) as u64;
        ctx.write_bytes(BASE, b"agent\0GET\0https://example.com/\0");
        args(&mut ctx, &[c, BASE, 2, 0]);
        let t = cb_http2_create_template(&mut ctx) as u64;
        args(&mut ctx, &[t, BASE + 6, BASE + 10, 0]);
        let r = cb_http2_create_request_with_url(&mut ctx) as u64;
        assert!(!http2().requests[&(r as i32)].no_content_length);

        args(&mut ctx, &[r]);
        assert_eq!(cb_http2_set_request_no_content_length(&mut ctx), 0);
        assert!(http2().requests[&(r as i32)].no_content_length);
        args(&mut ctx, &[0x7777]);
        assert_eq!(cb_http2_set_request_no_content_length(&mut ctx), HTTP2_ERROR_INVALID_ID);
        assert!(http2().templates.contains_key(&(t as i32)));

        args(&mut ctx, &[c]);
        assert_eq!(cb_http2_term(&mut ctx), 0);
    }
}
