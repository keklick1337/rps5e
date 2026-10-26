// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceNpUtility lookup and word filter HLE, offline.
//!
//! Both services keep title contexts (32 at most) and requests (32 at most)
//! made from them. There is no PSN, so an operation whose arguments pass
//! the checks finishes at once with SIGNED_OUT: a blocking request returns
//! it, an asynchronous one returns 0 and reports it through poll and wait.
//! An abort finishes a request that has not run with ABORTED and refuses
//! every later operation on it. A wait on an asynchronous request that has
//! not started blocks until the request finishes, is aborted or deleted.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Condvar, Mutex, MutexGuard};

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceNpUtility";
const OK: HleResult = 0;

const fn err(code: u32) -> HleResult {
    code as i32 as HleResult
}
const ERR_SIGNED_OUT: HleResult = err(0x8055_0006);
const ERR_LOOKUP_SYNCHRONOUS_REQUEST: HleResult = err(0x8055_0307);
const ERR_INVALID_ARGUMENT: HleResult = err(0x8055_0704);
const ERR_TOO_MANY_OBJECTS: HleResult = err(0x8055_0706);
const ERR_ABORTED: HleResult = err(0x8055_0707);
const ERR_INSUFFICIENT_ARGUMENT: HleResult = err(0x8055_070C);
const ERR_INVALID_ID: HleResult = err(0x8055_070E);
const ERR_INVALID_TYPE: HleResult = err(0x8055_0711);
const ERR_MEMORY_FAULT: HleResult = err(0x8002_000E);

const POLL_FINISHED: HleResult = 0;
const POLL_RUNNING: HleResult = 1;
const MAX_TITLE_CONTEXTS: usize = 32;
const MAX_REQUESTS: usize = 32;
/// The `size` an asynchronous request parameter must carry.
const ASYNC_REQUEST_PARAMETER_SIZE: u64 = 0x18;
const RESOLVE_TIMEOUT_MIN: u32 = 1_000_000;
const TRANSFER_TIMEOUT_MIN: u32 = 10_000_000;
const COMMENT_MAX_LENGTH: usize = 0x400;

#[derive(Clone, Copy)]
struct Request {
    title_ctx: i32,
    asynchronous: bool,
    used: bool,
    aborted: bool,
    finished: bool,
    result: i32,
}

struct State {
    title_contexts: BTreeSet<i32>,
    requests: BTreeMap<i32, Request>,
    next_title_ctx: i32,
    next_request: i32,
}

struct Service {
    state: Mutex<State>,
    finished: Condvar,
    /// What poll and wait answer for a blocking request.
    synchronous_error: HleResult,
}

impl Service {
    const fn new(first_title_ctx: i32, first_request: i32, synchronous_error: HleResult) -> Self {
        Service {
            state: Mutex::new(State {
                title_contexts: BTreeSet::new(),
                requests: BTreeMap::new(),
                next_title_ctx: first_title_ctx,
                next_request: first_request,
            }),
            finished: Condvar::new(),
            synchronous_error,
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn create_title_ctx(&self, self_np_id: u64) -> HleResult {
        if self_np_id == 0 {
            return ERR_INSUFFICIENT_ARGUMENT;
        }
        let mut s = self.lock();
        if s.title_contexts.len() >= MAX_TITLE_CONTEXTS {
            return ERR_TOO_MANY_OBJECTS;
        }
        let id = s.next_title_ctx;
        s.next_title_ctx += 1;
        s.title_contexts.insert(id);
        id as HleResult
    }

    fn delete_title_ctx(&self, id: i32) -> HleResult {
        let mut s = self.lock();
        if !s.title_contexts.remove(&id) {
            return ERR_INVALID_ID;
        }
        s.requests.retain(|_, r| r.title_ctx != id);
        self.finished.notify_all();
        OK
    }

    fn create_request(&self, title_ctx: i32, asynchronous: bool) -> HleResult {
        let mut s = self.lock();
        if !s.title_contexts.contains(&title_ctx) {
            return ERR_INVALID_ID;
        }
        if s.requests.len() >= MAX_REQUESTS {
            return ERR_TOO_MANY_OBJECTS;
        }
        let id = s.next_request;
        s.next_request += 1;
        s.requests.insert(id, Request { title_ctx, asynchronous, used: false, aborted: false, finished: false, result: 0 });
        id as HleResult
    }

    fn delete_request(&self, id: i32) -> HleResult {
        if self.lock().requests.remove(&id).is_none() {
            return ERR_INVALID_ID;
        }
        self.finished.notify_all();
        OK
    }

    /// Runs an operation offline: it finishes with SIGNED_OUT.
    fn start_offline(&self, id: i32) -> HleResult {
        let mut s = self.lock();
        let Some(r) = s.requests.get_mut(&id) else { return ERR_INVALID_ID };
        if r.aborted {
            return ERR_ABORTED;
        }
        if r.used {
            return ERR_INVALID_TYPE;
        }
        r.used = true;
        r.finished = true;
        r.result = ERR_SIGNED_OUT as i32;
        let asynchronous = r.asynchronous;
        self.finished.notify_all();
        if asynchronous { OK } else { ERR_SIGNED_OUT }
    }

    fn abort(&self, id: i32) -> HleResult {
        let mut s = self.lock();
        let Some(r) = s.requests.get_mut(&id) else { return ERR_INVALID_ID };
        r.aborted = true;
        if !r.finished {
            r.finished = true;
            r.result = ERR_ABORTED as i32;
            self.finished.notify_all();
        }
        OK
    }

    /// The result of a finished asynchronous request, None while it runs.
    fn poll(&self, id: i32) -> Result<Option<i32>, HleResult> {
        let s = self.lock();
        let r = s.requests.get(&id).ok_or(ERR_INVALID_ID)?;
        if !r.asynchronous {
            return Err(self.synchronous_error);
        }
        Ok(r.finished.then_some(r.result))
    }

    fn wait(&self, id: i32) -> Result<i32, HleResult> {
        let g = self.lock();
        match g.requests.get(&id) {
            None => return Err(ERR_INVALID_ID),
            Some(r) if !r.asynchronous => return Err(self.synchronous_error),
            Some(_) => {}
        }
        let (g, _) = crate::signals::wait_while(&self.state, &self.finished, g, None, |s| s.requests.get(&id).is_none_or(|r| r.finished));
        g.requests.get(&id).map(|r| r.result).ok_or(ERR_INVALID_ID)
    }

    fn set_timeout(&self, id: i32, resolve_retry: i32, timeouts: [u32; 4]) -> HleResult {
        let valid = |t: u32, min: u32| t == 0 || t >= min;
        let [resolve, conn, send, recv] = timeouts;
        if resolve_retry < 0
            || !valid(resolve, RESOLVE_TIMEOUT_MIN)
            || !valid(conn, TRANSFER_TIMEOUT_MIN)
            || !valid(send, TRANSFER_TIMEOUT_MIN)
            || !valid(recv, TRANSFER_TIMEOUT_MIN)
        {
            return ERR_INVALID_ARGUMENT;
        }
        if resolve_retry == 0 && timeouts == [0; 4] {
            return ERR_INVALID_ARGUMENT;
        }
        let s = self.lock();
        if s.title_contexts.contains(&id) || s.requests.contains_key(&id) { OK } else { ERR_INVALID_ID }
    }
}

static LOOKUP: Service = Service::new(1, 1, ERR_LOOKUP_SYNCHRONOUS_REQUEST);
static WORD_FILTER: Service = Service::new(0x0100_0001, 0x0200_0001, ERR_INVALID_ID);

// ---------------------------------------------------------------------------
// Shared entry points
// ---------------------------------------------------------------------------

fn create_async_request(service: &Service, ctx: &CpuContext) -> HleResult {
    let param = ctx.rsi();
    if param == 0 || ctx.read_u64(param) != Some(ASYNC_REQUEST_PARAMETER_SIZE) {
        return ERR_INVALID_ARGUMENT;
    }
    service.create_request(ctx.rdi() as i32, true)
}

/// (reqId, int* result): 0 with the result once finished, 1 while running.
fn poll_async(service: &Service, ctx: &CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    match service.poll(ctx.rdi() as i32) {
        Err(e) => e,
        Ok(None) => POLL_RUNNING,
        Ok(Some(result)) => {
            if ctx.write_u32(out, result as u32) { POLL_FINISHED } else { ERR_MEMORY_FAULT }
        }
    }
}

/// (reqId, int* result): blocks until the request finishes.
fn wait_async(service: &Service, ctx: &CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    match service.wait(ctx.rdi() as i32) {
        Err(e) => e,
        Ok(result) => {
            if ctx.write_u32(out, result as u32) { OK } else { ERR_MEMORY_FAULT }
        }
    }
}

/// (id, resolveRetry, resolveTimeout, connTimeout, sendTimeout, recvTimeout)
fn set_timeout(service: &Service, ctx: &CpuContext) -> HleResult {
    let timeouts = [ctx.rdx() as u32, ctx.rcx() as u32, ctx.r8() as u32, ctx.r9() as u32];
    service.set_timeout(ctx.rdi() as i32, ctx.rsi() as i32, timeouts)
}

/// A comment longer than 0x400 bytes is refused.
fn check_comment(ctx: &CpuContext, comment: u64) -> Result<(), HleResult> {
    let mut at = comment;
    let mut len = 0usize;
    while len <= COMMENT_MAX_LENGTH {
        let chunk = ctx.read_bytes(at, 64).or_else(|| ctx.read_bytes(at, 1)).ok_or(ERR_MEMORY_FAULT)?;
        if let Some(end) = chunk.iter().position(|&b| b == 0) {
            len += end;
            break;
        }
        len += chunk.len();
        at += chunk.len() as u64;
    }
    if len > COMMENT_MAX_LENGTH { Err(ERR_INVALID_ARGUMENT) } else { Ok(()) }
}

// ---------------------------------------------------------------------------
// NpLookup
// ---------------------------------------------------------------------------

fn cb_lookup_create_title_ctx(ctx: &mut CpuContext) -> HleResult {
    LOOKUP.create_title_ctx(ctx.rdi())
}
fn cb_lookup_delete_title_ctx(ctx: &mut CpuContext) -> HleResult {
    LOOKUP.delete_title_ctx(ctx.rdi() as i32)
}
fn cb_lookup_create_request(ctx: &mut CpuContext) -> HleResult {
    LOOKUP.create_request(ctx.rdi() as i32, false)
}
fn cb_lookup_create_async_request(ctx: &mut CpuContext) -> HleResult {
    create_async_request(&LOOKUP, ctx)
}
fn cb_lookup_delete_request(ctx: &mut CpuContext) -> HleResult {
    LOOKUP.delete_request(ctx.rdi() as i32)
}
fn cb_lookup_abort_request(ctx: &mut CpuContext) -> HleResult {
    LOOKUP.abort(ctx.rdi() as i32)
}
fn cb_lookup_poll_async(ctx: &mut CpuContext) -> HleResult {
    poll_async(&LOOKUP, ctx)
}
fn cb_lookup_wait_async(ctx: &mut CpuContext) -> HleResult {
    wait_async(&LOOKUP, ctx)
}
fn cb_lookup_set_timeout(ctx: &mut CpuContext) -> HleResult {
    set_timeout(&LOOKUP, ctx)
}

/// sceNpLookupNpId(reqId, const SceNpOnlineId*, SceNpId*, option)
fn cb_lookup_np_id(ctx: &mut CpuContext) -> HleResult {
    if ctx.rcx() != 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if ctx.rsi() == 0 || ctx.rdx() == 0 {
        return ERR_INSUFFICIENT_ARGUMENT;
    }
    LOOKUP.start_offline(ctx.rdi() as i32)
}

// ---------------------------------------------------------------------------
// NpWordFilter
// ---------------------------------------------------------------------------

fn cb_word_filter_create_title_ctx(ctx: &mut CpuContext) -> HleResult {
    WORD_FILTER.create_title_ctx(ctx.rdi())
}
fn cb_word_filter_delete_title_ctx(ctx: &mut CpuContext) -> HleResult {
    WORD_FILTER.delete_title_ctx(ctx.rdi() as i32)
}
fn cb_word_filter_create_request(ctx: &mut CpuContext) -> HleResult {
    WORD_FILTER.create_request(ctx.rdi() as i32, false)
}
fn cb_word_filter_create_async_request(ctx: &mut CpuContext) -> HleResult {
    create_async_request(&WORD_FILTER, ctx)
}
fn cb_word_filter_delete_request(ctx: &mut CpuContext) -> HleResult {
    WORD_FILTER.delete_request(ctx.rdi() as i32)
}
fn cb_word_filter_abort_request(ctx: &mut CpuContext) -> HleResult {
    WORD_FILTER.abort(ctx.rdi() as i32)
}
fn cb_word_filter_poll_async(ctx: &mut CpuContext) -> HleResult {
    poll_async(&WORD_FILTER, ctx)
}
fn cb_word_filter_wait_async(ctx: &mut CpuContext) -> HleResult {
    wait_async(&WORD_FILTER, ctx)
}
fn cb_word_filter_set_timeout(ctx: &mut CpuContext) -> HleResult {
    set_timeout(&WORD_FILTER, ctx)
}

/// sceNpWordFilterCensorComment(reqId, const char* comment, option)
fn cb_word_filter_censor_comment(ctx: &mut CpuContext) -> HleResult {
    let comment = ctx.rsi();
    if comment == 0 {
        return ERR_INSUFFICIENT_ARGUMENT;
    }
    if ctx.rdx() != 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if let Err(e) = check_comment(ctx, comment) {
        return e;
    }
    WORD_FILTER.start_offline(ctx.rdi() as i32)
}

/// sceNpWordFilterSanitizeComment(reqId, const char* comment,
/// char* sanitized, option)
fn cb_word_filter_sanitize_comment(ctx: &mut CpuContext) -> HleResult {
    let comment = ctx.rsi();
    if comment == 0 || ctx.rdx() == 0 {
        return ERR_INSUFFICIENT_ARGUMENT;
    }
    if ctx.rcx() != 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if let Err(e) = check_comment(ctx, comment) {
        return e;
    }
    WORD_FILTER.start_offline(ctx.rdi() as i32)
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("8533Q+LU7EQ", "sceNpLookupCreateTitleCtx", cb_lookup_create_title_ctx),
    ("mtqDK9zkoIE", "sceNpLookupDeleteTitleCtx", cb_lookup_delete_title_ctx),
    ("iQr9UxPHUFs", "sceNpLookupCreateRequest", cb_lookup_create_request),
    ("JA4+sS39GMs", "sceNpLookupCreateAsyncRequest", cb_lookup_create_async_request),
    ("wLaxchvEEnk", "sceNpLookupDeleteRequest", cb_lookup_delete_request),
    ("eYz4v5Uek9U", "sceNpLookupAbortRequest", cb_lookup_abort_request),
    ("V4EVrruHuy8", "sceNpLookupPollAsync", cb_lookup_poll_async),
    ("YX9dAus6baE", "sceNpLookupWaitAsync", cb_lookup_wait_async),
    ("0MV72WO7V34", "sceNpLookupSetTimeout", cb_lookup_set_timeout),
    ("T6tnM1Uti4g", "sceNpLookupNpId", cb_lookup_np_id),
    ("r9BgI0PfJZg", "sceNpWordFilterCreateTitleCtx", cb_word_filter_create_title_ctx),
    ("t0P5z5yuFPA", "sceNpWordFilterDeleteTitleCtx", cb_word_filter_delete_title_ctx),
    ("iCq5xW5KQW4", "sceNpWordFilterCreateRequest", cb_word_filter_create_request),
    ("IEB+vgVoQbw", "sceNpWordFilterCreateAsyncRequest", cb_word_filter_create_async_request),
    ("PYFS1H70bDs", "sceNpWordFilterDeleteRequest", cb_word_filter_delete_request),
    ("rAOOqDAxBIk", "sceNpWordFilterAbortRequest", cb_word_filter_abort_request),
    ("ur5SShyG0dk", "sceNpWordFilterPollAsync", cb_word_filter_poll_async),
    ("87ivWj5yKzg", "sceNpWordFilterWaitAsync", cb_word_filter_wait_async),
    ("Fa4dVWgmffk", "sceNpWordFilterSetTimeout", cb_word_filter_set_timeout),
    ("1dMndqL-QgE", "sceNpWordFilterCensorComment", cb_word_filter_censor_comment),
    ("Jj4mkpFO2gE", "sceNpWordFilterSanitizeComment", cb_word_filter_sanitize_comment),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    const BASE: u64 = 0x5A6C_6000_0000;

    fn call(ctx: &mut CpuContext, name: &str, args: &[u64]) -> HleResult {
        let cb = EXPORTS.iter().find(|e| e.1 == name).unwrap_or_else(|| panic!("{name}")).2;
        for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].into_iter().zip(args.iter().copied().chain(std::iter::repeat(0))) {
            ctx.set(r, v);
        }
        cb(ctx)
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// Lookup: blocking requests return SIGNED_OUT and refuse poll / wait
    /// as synchronous; asynchronous ones report it; a request runs once;
    /// an abort sticks; timeouts are checked against their minimums.
    #[test]
    fn lookup_requests_offline() {
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (param, online_id, np_id, result) = (BASE, BASE + 0x100, BASE + 0x200, BASE + 0x300);
        ctx.write_u64(param, ASYNC_REQUEST_PARAMETER_SIZE);

        assert_eq!(call(&mut ctx, "sceNpLookupCreateTitleCtx", &[0]), ERR_INSUFFICIENT_ARGUMENT);
        let title = call(&mut ctx, "sceNpLookupCreateTitleCtx", &[np_id]);
        assert!(title > 0);
        let title = title as u64;
        assert_eq!(call(&mut ctx, "sceNpLookupCreateRequest", &[0x7777]), ERR_INVALID_ID);

        let sync = call(&mut ctx, "sceNpLookupCreateRequest", &[title]) as u64;
        assert_eq!(call(&mut ctx, "sceNpLookupNpId", &[sync, online_id, np_id, 1]), ERR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, "sceNpLookupNpId", &[sync, 0, np_id, 0]), ERR_INSUFFICIENT_ARGUMENT);
        assert_eq!(call(&mut ctx, "sceNpLookupNpId", &[sync, online_id, np_id, 0]), ERR_SIGNED_OUT);
        assert_eq!(call(&mut ctx, "sceNpLookupNpId", &[sync, online_id, np_id, 0]), ERR_INVALID_TYPE);
        assert_eq!(call(&mut ctx, "sceNpLookupPollAsync", &[sync, result]), ERR_LOOKUP_SYNCHRONOUS_REQUEST);
        assert_eq!(call(&mut ctx, "sceNpLookupWaitAsync", &[sync, result]), ERR_LOOKUP_SYNCHRONOUS_REQUEST);
        assert_eq!(call(&mut ctx, "sceNpLookupPollAsync", &[sync, 0]), ERR_INVALID_ARGUMENT);

        ctx.write_u64(param, 0x10);
        assert_eq!(call(&mut ctx, "sceNpLookupCreateAsyncRequest", &[title, param]), ERR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, "sceNpLookupCreateAsyncRequest", &[title, 0]), ERR_INVALID_ARGUMENT);
        ctx.write_u64(param, ASYNC_REQUEST_PARAMETER_SIZE);
        let req = call(&mut ctx, "sceNpLookupCreateAsyncRequest", &[title, param]) as u64;
        assert_eq!(call(&mut ctx, "sceNpLookupPollAsync", &[req, result]), POLL_RUNNING);
        assert_eq!(call(&mut ctx, "sceNpLookupNpId", &[req, online_id, np_id, 0]), OK);
        ctx.write_u32(result, 0);
        assert_eq!(call(&mut ctx, "sceNpLookupPollAsync", &[req, result]), POLL_FINISHED);
        assert_eq!(ctx.read_u32(result), Some(ERR_SIGNED_OUT as u32));
        ctx.write_u32(result, 0);
        assert_eq!(call(&mut ctx, "sceNpLookupWaitAsync", &[req, result]), OK);
        assert_eq!(ctx.read_u32(result), Some(ERR_SIGNED_OUT as u32));

        // An abort finishes a request that has not run and sticks.
        let aborted = call(&mut ctx, "sceNpLookupCreateAsyncRequest", &[title, param]) as u64;
        assert_eq!(call(&mut ctx, "sceNpLookupAbortRequest", &[aborted]), OK);
        assert_eq!(call(&mut ctx, "sceNpLookupWaitAsync", &[aborted, result]), OK);
        assert_eq!(ctx.read_u32(result), Some(ERR_ABORTED as u32));
        assert_eq!(call(&mut ctx, "sceNpLookupNpId", &[aborted, online_id, np_id, 0]), ERR_ABORTED);

        // Timeouts: zero or at least the minimum; not all zero; a known id.
        for (args, expect) in [
            ([title, 1, 0, 0, 0, 0], OK),
            ([req, 0, 1_000_000, 0, 0, 0], OK),
            ([title, 0, 0, 10_000_000, 10_000_000, 10_000_000], OK),
            ([title, 0, 0, 0, 0, 0], ERR_INVALID_ARGUMENT),
            ([title, (-1i32) as u32 as u64, 0, 0, 0, 0], ERR_INVALID_ARGUMENT),
            ([title, 0, 999_999, 0, 0, 0], ERR_INVALID_ARGUMENT),
            ([title, 0, 0, 0, 9_999_999, 0], ERR_INVALID_ARGUMENT),
            ([0x7777, 1, 0, 0, 0, 0], ERR_INVALID_ID),
        ] {
            assert_eq!(call(&mut ctx, "sceNpLookupSetTimeout", &args), expect, "{args:?}");
        }

        // Deleting the title context deletes its requests.
        assert_eq!(call(&mut ctx, "sceNpLookupDeleteRequest", &[sync]), OK);
        assert_eq!(call(&mut ctx, "sceNpLookupDeleteRequest", &[sync]), ERR_INVALID_ID);
        assert_eq!(call(&mut ctx, "sceNpLookupDeleteTitleCtx", &[title]), OK);
        assert_eq!(call(&mut ctx, "sceNpLookupPollAsync", &[req, result]), ERR_INVALID_ID);
        assert_eq!(call(&mut ctx, "sceNpLookupDeleteTitleCtx", &[title]), ERR_INVALID_ID);
    }

    /// A wait on an asynchronous request that has not run returns once
    /// another thread deletes it.
    #[test]
    fn lookup_wait_ends_when_the_request_goes() {
        let vm = ps5vm::VirtualMemory::new();
        const AT: u64 = BASE + 0x10_0000;
        vm.map(AT, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_u64(AT, ASYNC_REQUEST_PARAMETER_SIZE);
        let title = call(&mut ctx, "sceNpLookupCreateTitleCtx", &[AT]) as u64;
        let req = call(&mut ctx, "sceNpLookupCreateAsyncRequest", &[title, AT]) as i32;
        let deleter = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            LOOKUP.delete_request(req)
        });
        assert_eq!(call(&mut ctx, "sceNpLookupWaitAsync", &[req as u64, AT + 8]), ERR_INVALID_ID);
        assert_eq!(deleter.join().unwrap(), OK);
        assert_eq!(call(&mut ctx, "sceNpLookupDeleteTitleCtx", &[title]), OK);
    }

    /// Word filter: comments up to 0x400 bytes; poll and wait refuse a
    /// blocking request as an unknown id.
    #[test]
    fn word_filter_requests_offline() {
        let vm = ps5vm::VirtualMemory::new();
        const AT: u64 = BASE + 0x20_0000;
        vm.map(AT, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (param, comment, long, out, result) = (AT, AT + 0x100, AT + 0x1000, AT + 0x2000, AT + 0x3000);
        ctx.write_u64(param, ASYNC_REQUEST_PARAMETER_SIZE);
        ctx.write_bytes(comment, b"hello\0");
        ctx.write_bytes(long, &[b'a'; COMMENT_MAX_LENGTH + 1]);
        ctx.write_bytes(long + COMMENT_MAX_LENGTH as u64 + 1, &[0]);

        let title = call(&mut ctx, "sceNpWordFilterCreateTitleCtx", &[AT]);
        assert!(title > 0x0100_0000);
        let title = title as u64;
        let sync = call(&mut ctx, "sceNpWordFilterCreateRequest", &[title]);
        assert!(sync > 0x0200_0000);
        let sync = sync as u64;
        assert_eq!(call(&mut ctx, "sceNpWordFilterCensorComment", &[sync, 0, 0]), ERR_INSUFFICIENT_ARGUMENT);
        assert_eq!(call(&mut ctx, "sceNpWordFilterCensorComment", &[sync, comment, 1]), ERR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, "sceNpWordFilterCensorComment", &[sync, long, 0]), ERR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, "sceNpWordFilterSanitizeComment", &[sync, comment, 0, 0]), ERR_INSUFFICIENT_ARGUMENT);
        assert_eq!(call(&mut ctx, "sceNpWordFilterSanitizeComment", &[sync, comment, out, 1]), ERR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, "sceNpWordFilterCensorComment", &[sync, comment, 0]), ERR_SIGNED_OUT);
        assert_eq!(call(&mut ctx, "sceNpWordFilterPollAsync", &[sync, result]), ERR_INVALID_ID);
        assert_eq!(call(&mut ctx, "sceNpWordFilterWaitAsync", &[sync, result]), ERR_INVALID_ID);

        let req = call(&mut ctx, "sceNpWordFilterCreateAsyncRequest", &[title, param]) as u64;
        assert_eq!(call(&mut ctx, "sceNpWordFilterSanitizeComment", &[req, comment, out, 0]), OK);
        assert_eq!(call(&mut ctx, "sceNpWordFilterWaitAsync", &[req, result]), OK);
        assert_eq!(ctx.read_u32(result), Some(ERR_SIGNED_OUT as u32));
        assert_eq!(call(&mut ctx, "sceNpWordFilterSetTimeout", &[req, 1, 0, 0, 0, 0]), OK);
        assert_eq!(call(&mut ctx, "sceNpWordFilterAbortRequest", &[req]), OK);
        assert_eq!(call(&mut ctx, "sceNpWordFilterAbortRequest", &[0x7777]), ERR_INVALID_ID);
        assert_eq!(call(&mut ctx, "sceNpWordFilterDeleteTitleCtx", &[title]), OK);
    }
}
