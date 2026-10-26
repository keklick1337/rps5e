// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceRudp — reliable UDP over the (absent) network.
//!
//! The emulator is offline: the library initialises, contexts can be
//! created, configured, bound and terminated, but no connection is ever
//! established. Initiating a connection is refused and data transfer on a
//! context reports a reset connection.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceRudp";

const fn err(code: u32) -> HleResult {
    code as i32 as i64
}

const ERROR_NOT_INITIALIZED: HleResult = err(0x8077_0001);
const ERROR_INVALID_CONTEXT_ID: HleResult = err(0x8077_0003);
const ERROR_INVALID_ARGUMENT: HleResult = err(0x8077_0004);
const ERROR_CONN_RESET: HleResult = err(0x8077_0009);
const ERROR_CONN_REFUSED: HleResult = err(0x8077_000A);

/// Largest status block cleared by sceRudpGetContextStatus.
const MAX_STATUS_BYTES: u32 = 0x1000;
const ERROR_ALREADY_INITIALIZED: HleResult = err(0x8077_0002);
const ERROR_NOT_ACCEPTABLE: HleResult = err(0x8077_0011);
/// Smallest memory pool sceRudpInit takes, and its alignment.
const POOL_MIN_SIZE: i32 = 0xC58;
const POOL_ALIGNMENT: u64 = 8;
/// SceRudpStatus: 0xF8 bytes, currentContexts (u32) at 0x64.
const STATUS_SIZE: u64 = 0xF8;
const STATUS_CURRENT_CONTEXTS: usize = 0x64;

struct Context {
    handler: u64,
    arg: u64,
    bound: bool,
    /// Option values by id (1..=19); 12..=14 have no default.
    options: BTreeMap<i32, u32>,
}

const ERROR_INVALID_OPTION: HleResult = err(0x8077_0005);
const ERROR_TOO_MANY_CONTEXTS: HleResult = err(0x8077_0020);
/// Option ids sceRudpSetOption takes.
const OPTION_IDS: std::ops::RangeInclusive<i32> = 1..=19;

/// The option values a new context starts with: MAX_PAYLOAD, SNDBUF,
/// RCVBUF, NODELAY, DELIVERY_CRITICAL, ORDER_CRITICAL, NONBLOCK, STREAM,
/// CONNECTION_TIMEOUT, CLOSE_WAIT_TIMEOUT, AGGREGATION_TIMEOUT, then
/// READ/WRITE/FLUSH_TIMEOUT and KEEP_ALIVE_INTERVAL/TIMEOUT.
const DEFAULT_OPTIONS: [(i32, u32); 16] = [
    (1, 1346),
    (2, 65536),
    (3, 65536),
    (4, 0),
    (5, 1),
    (6, 1),
    (7, 1),
    (8, 0),
    (9, 60000),
    (10, 0),
    (11, 30),
    (15, 0),
    (16, 0),
    (17, 0),
    (18, 0),
    (19, 0),
];

impl Context {
    fn new(handler: u64, arg: u64) -> Self {
        Context { handler, arg, bound: false, options: DEFAULT_OPTIONS.into_iter().collect() }
    }
}

struct State {
    initialized: bool,
    next_id: i32,
    contexts: BTreeMap<i32, Context>,
    /// Library-wide event handler (handler, arg).
    event_handler: (u64, u64),
}

static STATE: Mutex<State> =
    Mutex::new(State { initialized: false, next_id: 0, contexts: BTreeMap::new(), event_handler: (0, 0) });

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|p| p.into_inner())
}

/// Run `f` on a live context of an initialised library.
fn with_context(ctx_id: i32, f: impl FnOnce(&mut Context) -> HleResult) -> HleResult {
    let mut s = state();
    if !s.initialized {
        return ERROR_NOT_INITIALIZED;
    }
    match s.contexts.get_mut(&ctx_id) {
        Some(c) => f(c),
        None => ERROR_INVALID_CONTEXT_ID,
    }
}

const ERROR_MEMORY: HleResult = err(0x8077_0007);
const ERROR_INVALID_POLL_ID: HleResult = err(0x8077_001F);
/// Live poll objects at most.
const MAX_POLLS: usize = 4096;
/// SceRudpPollEvent: ctxId i32, reqEvents u16, rtnEvents u16.
const POLL_EVENT_SIZE: u64 = 8;
const POLL_OP_ADD: i32 = 1;
const POLL_OP_MODIFY: i32 = 2;
const POLL_OP_REMOVE: i32 = 3;

/// A poll object: up to `capacity` contexts and the events asked of each.
struct Poll {
    capacity: u32,
    entries: BTreeMap<i32, u16>,
}

struct Polls {
    next_id: i32,
    polls: BTreeMap<i32, Poll>,
    /// Bumped by sceRudpEnd: waits end with NOT_INITIALIZED.
    ended: u64,
}

static POLLS: Mutex<Polls> = Mutex::new(Polls { next_id: 0, polls: BTreeMap::new(), ended: 0 });
static POLLS_CHANGED: std::sync::Condvar = std::sync::Condvar::new();

fn polls() -> std::sync::MutexGuard<'static, Polls> {
    POLLS.lock().unwrap_or_else(|p| p.into_inner())
}

/// sceRudpPollCreate(size_t size) -> poll id: a poll object for up to
/// `size` contexts.
fn cb_poll_create(ctx: &mut CpuContext) -> HleResult {
    let size = ctx.rdi();
    if !state().initialized {
        return ERROR_NOT_INITIALIZED;
    }
    if size == 0 || size > u32::MAX as u64 {
        return ERROR_INVALID_ARGUMENT;
    }
    let mut p = polls();
    if p.polls.len() >= MAX_POLLS {
        return ERROR_MEMORY;
    }
    let id = loop {
        p.next_id = if p.next_id == i32::MAX { 1 } else { p.next_id + 1 };
        if !p.polls.contains_key(&p.next_id) {
            break p.next_id;
        }
    };
    p.polls.insert(id, Poll { capacity: size as u32, entries: BTreeMap::new() });
    id as HleResult
}

/// sceRudpPollDestroy(int pollId): waits on it end with INVALID_POLL_ID.
fn cb_poll_destroy(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    if !state().initialized {
        return ERROR_NOT_INITIALIZED;
    }
    if id < 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    if polls().polls.remove(&id).is_none() {
        return ERROR_INVALID_POLL_ID;
    }
    POLLS_CHANGED.notify_all();
    0
}

/// sceRudpPollControl(int pollId, int op, int ctxId, uint16 events): add,
/// modify or remove a context of the poll.
fn cb_poll_control(ctx: &mut CpuContext) -> HleResult {
    let (id, op, ctx_id, events) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx() as i32, ctx.rcx() as u16);
    let s = state();
    if !s.initialized {
        return ERROR_NOT_INITIALIZED;
    }
    if id < 0 || ctx_id < 0 || !(POLL_OP_ADD..=POLL_OP_REMOVE).contains(&op) {
        return ERROR_INVALID_ARGUMENT;
    }
    let mut p = polls();
    let Some(poll) = p.polls.get_mut(&id) else { return ERROR_INVALID_POLL_ID };
    if !s.contexts.contains_key(&ctx_id) {
        return ERROR_INVALID_CONTEXT_ID;
    }
    match op {
        POLL_OP_ADD => {
            if poll.entries.len() >= poll.capacity as usize {
                return ERROR_TOO_MANY_CONTEXTS;
            }
            if poll.entries.contains_key(&ctx_id) {
                return ERROR_INVALID_CONTEXT_ID;
            }
            poll.entries.insert(ctx_id, events);
        }
        POLL_OP_MODIFY => match poll.entries.get_mut(&ctx_id) {
            Some(e) => *e = events,
            None => return ERROR_INVALID_CONTEXT_ID,
        },
        _ => {
            if poll.entries.remove(&ctx_id).is_none() {
                return ERROR_INVALID_CONTEXT_ID;
            }
        }
    }
    0
}

/// sceRudpPollWait(int pollId, SceRudpPollEvent* events, size_t eventLen,
/// SceRudpUsec timeout) -> event count: no connection ever has an event, so
/// the wait runs out its timeout and reports none; it ends early with
/// INVALID_POLL_ID when the poll is destroyed and NOT_INITIALIZED when the
/// library ends.
fn cb_poll_wait(ctx: &mut CpuContext) -> HleResult {
    /// Timeouts past this many microseconds never expire.
    const UNBOUNDED_US: u64 = i64::MAX as u64 / 2000;
    let (id, events, len, timeout_us) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.rcx());
    if !state().initialized {
        return ERROR_NOT_INITIALIZED;
    }
    if id < 0 || events == 0 || len == 0 || len > u32::MAX as u64 || ctx.read_bytes(events, (len * POLL_EVENT_SIZE) as usize).is_none() {
        return ERROR_INVALID_ARGUMENT;
    }
    let p = polls();
    if !p.polls.contains_key(&id) {
        return ERROR_INVALID_POLL_ID;
    }
    if timeout_us == 0 {
        return 0;
    }
    let ended = p.ended;
    let deadline = if timeout_us > UNBOUNDED_US { None } else { std::time::Instant::now().checked_add(std::time::Duration::from_micros(timeout_us)) };
    let (p, _) = crate::signals::wait_while(&POLLS, &POLLS_CHANGED, p, deadline, |p| p.ended != ended || !p.polls.contains_key(&id));
    if p.ended != ended {
        ERROR_NOT_INITIALIZED
    } else if !p.polls.contains_key(&id) {
        ERROR_INVALID_POLL_ID
    } else {
        0
    }
}

/// sceRudpInit(void* memPool, int memPoolSize), as measured on a console:
/// ALREADY_INITIALIZED once initialised; a NULL pool or a size of 0 or less
/// INVALID_ARGUMENT; a pool under 0xC58 bytes or not 8-byte aligned MEMORY.
/// The pool is not used: nothing is allocated offline.
fn cb_init(ctx: &mut CpuContext) -> HleResult {
    let (pool, size) = (ctx.rdi(), ctx.rsi() as i32);
    let mut s = state();
    if s.initialized {
        return ERROR_ALREADY_INITIALIZED;
    }
    if pool == 0 || size <= 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    if size < POOL_MIN_SIZE || pool % POOL_ALIGNMENT != 0 {
        return ERROR_MEMORY;
    }
    log::info!("[rudp] init pool={pool:#x} size={size:#x} (offline)");
    s.initialized = true;
    0
}

/// sceRudpEnd(void): shut the library down, dropping every context.
fn cb_end(_ctx: &mut CpuContext) -> HleResult {
    let mut s = state();
    if !s.initialized {
        return ERROR_NOT_INITIALIZED;
    }
    s.initialized = false;
    s.contexts.clear();
    s.event_handler = (0, 0);
    let mut p = polls();
    p.polls.clear();
    p.ended += 1;
    POLLS_CHANGED.notify_all();
    0
}

/// sceRudpEnableInternalIOThread(stackSize, priority): there is no I/O to
/// service offline, so no thread is started.
fn cb_enable_internal_io_thread(ctx: &mut CpuContext) -> HleResult {
    if !state().initialized {
        return ERROR_NOT_INITIALIZED;
    }
    log::debug!("[rudp] internal I/O thread stack={:#x} priority={}", ctx.rdi() as u32, ctx.rsi() as u32);
    0
}

/// sceRudpNetFlush() / sceRudpProcessEvents(): nothing is queued offline.
fn cb_initialized_only(_ctx: &mut CpuContext) -> HleResult {
    if state().initialized { 0 } else { ERROR_NOT_INITIALIZED }
}

/// Largest RUDP segment payload.
const MAX_SEGMENT_SIZE: u16 = 1410;

/// sceRudpGetMaxSegmentSize(u16* size).
fn cb_get_max_segment_size(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if !state().initialized {
        return ERROR_NOT_INITIALIZED;
    }
    if out == 0 || !ctx.write_bytes(out, &MAX_SEGMENT_SIZE.to_le_bytes()) {
        return ERROR_INVALID_ARGUMENT;
    }
    0
}

/// sceRudpActivate(): there is no reliable-UDP transport to start offline,
/// so activation has nothing to do; it does not need sceRudpInit.
fn cb_activate(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// sceRudpGetStatus(void* status, size_t size), as measured on a console:
/// NOT_INITIALIZED before sceRudpInit; a NULL status or a size of 0 or past
/// 0xF8 INVALID_ARGUMENT. The first `size` bytes of the 0xF8-byte status
/// are written: the live context count at 0x64, the traffic and allocator
/// counters zero (nothing is sent, nothing allocated from the pool).
fn cb_get_status(ctx: &mut CpuContext) -> HleResult {
    let (status, size) = (ctx.rdi(), ctx.rsi());
    let contexts = {
        let s = state();
        if !s.initialized {
            return ERROR_NOT_INITIALIZED;
        }
        s.contexts.len() as u32
    };
    if status == 0 || size == 0 || size > STATUS_SIZE {
        return ERROR_INVALID_ARGUMENT;
    }
    let mut report = [0u8; STATUS_SIZE as usize];
    report[STATUS_CURRENT_CONTEXTS..STATUS_CURRENT_CONTEXTS + 4].copy_from_slice(&contexts.to_le_bytes());
    if ctx.write_bytes(status, &report[..size as usize]) { 0 } else { ERROR_INVALID_ARGUMENT }
}

/// A live context id of an initialised library: NOT_INITIALIZED,
/// INVALID_ARGUMENT for a negative id, INVALID_CONTEXT_ID for an unknown one.
fn check_context(ctx_id: i32) -> Result<(), HleResult> {
    let s = state();
    if !s.initialized {
        return Err(ERROR_NOT_INITIALIZED);
    }
    if ctx_id < 0 {
        return Err(ERROR_INVALID_ARGUMENT);
    }
    if !s.contexts.contains_key(&ctx_id) { Err(ERROR_INVALID_CONTEXT_ID) } else { Ok(()) }
}

/// sceRudpGetNumberOfPacketsToRead(int ctxId): no packet ever arrives.
fn cb_get_number_of_packets_to_read(ctx: &mut CpuContext) -> HleResult {
    check_context(ctx.rdi() as i32).map_or_else(|e| e, |()| 0)
}

/// sceRudpGetSizeWritable / sceRudpGetSizeReadable(int ctxId): a context
/// without a connection is NOT_ACCEPTABLE.
fn cb_get_size_unconnected(ctx: &mut CpuContext) -> HleResult {
    check_context(ctx.rdi() as i32).map_or_else(|e| e, |()| ERROR_NOT_ACCEPTABLE)
}

/// sceRudpSetEventHandler(handler, arg)
fn cb_set_event_handler(ctx: &mut CpuContext) -> HleResult {
    state().event_handler = (ctx.rdi(), ctx.rsi());
    0
}

/// sceRudpCreateContext(handler, arg, int* ctxId)
fn cb_create_context(ctx: &mut CpuContext) -> HleResult {
    let (handler, arg, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if out == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    let mut s = state();
    if !s.initialized {
        return ERROR_NOT_INITIALIZED;
    }
    let id = s.next_id;
    if !ctx.write_u32(out, id as u32) {
        return ERROR_INVALID_ARGUMENT;
    }
    s.next_id += 1;
    s.contexts.insert(id, Context::new(handler, arg));
    0
}

/// sceRudpTerminate(int ctxId): destroy one context.
fn cb_terminate(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let mut s = state();
    if !s.initialized {
        return ERROR_NOT_INITIALIZED;
    }
    if s.contexts.remove(&id).is_none() {
        return ERROR_INVALID_CONTEXT_ID;
    }
    for poll in polls().polls.values_mut() {
        poll.entries.remove(&id);
    }
    0
}

/// sceRudpBind(int ctxId, int soc, uint16 vport, uint8 muxMode)
fn cb_bind(ctx: &mut CpuContext) -> HleResult {
    with_context(ctx.rdi() as i32, |c| {
        c.bound = true;
        0
    })
}

/// The checks sceRudpSetOption and sceRudpGetOption share: the context,
/// then the option id (1..=19), then a 4-byte value.
fn option_call(ctx: &CpuContext, f: impl FnOnce(&mut Context, i32, u64) -> HleResult) -> HleResult {
    let (option, value, len) = (ctx.rsi() as i32, ctx.rdx(), ctx.rcx() as u32);
    with_context(ctx.rdi() as i32, |c| {
        if !OPTION_IDS.contains(&option) {
            return ERROR_INVALID_OPTION;
        }
        if value == 0 || len != 4 {
            return ERROR_INVALID_ARGUMENT;
        }
        f(c, option, value)
    })
}

/// sceRudpSetOption(int ctxId, int option, const void* value, uint32 len):
/// every option id is stored, the ones without a readable value too.
fn cb_set_option(ctx: &mut CpuContext) -> HleResult {
    option_call(ctx, |c, option, value| match ctx.read_u32(value) {
        Some(v) => {
            c.options.insert(option, v);
            0
        }
        None => ERROR_INVALID_ARGUMENT,
    })
}

/// sceRudpGetOption(int ctxId, int option, void* value, uint32 len)
fn cb_get_option(ctx: &mut CpuContext) -> HleResult {
    option_call(ctx, |c, option, value| {
        let stored = if (12..=14).contains(&option) { None } else { c.options.get(&option).copied() };
        let Some(v) = stored else {
            log::warn!("[rudp] sceRudpGetOption: option {option} has no known value");
            return ERROR_INVALID_OPTION;
        };
        if ctx.write_u32(value, v) { 0 } else { ERROR_INVALID_ARGUMENT }
    })
}

/// sceRudpInitiate(int ctxId, const SceNetSockaddr* to, socklen, int flags):
/// there is no peer to reach.
fn cb_initiate(ctx: &mut CpuContext) -> HleResult {
    let to = ctx.rsi();
    with_context(ctx.rdi() as i32, |c| {
        if to == 0 {
            return ERROR_INVALID_ARGUMENT;
        }
        log::debug!("[rudp] initiate on ctx (handler={:#x} arg={:#x} bound={}): refused, offline", c.handler, c.arg, c.bound);
        ERROR_CONN_REFUSED
    })
}

/// sceRudpGetContextStatus(int ctxId, void* status, uint32 size): an idle
/// context with no traffic (all counters zero).
fn cb_get_context_status(ctx: &mut CpuContext) -> HleResult {
    let (status, size) = (ctx.rsi(), ctx.rdx() as u32);
    with_context(ctx.rdi() as i32, |_| {
        if status != 0 && size != 0 {
            let n = size.min(MAX_STATUS_BYTES) as usize;
            if !ctx.write_bytes(status, &vec![0u8; n]) {
                return ERROR_INVALID_ARGUMENT;
            }
        }
        0
    })
}

/// sceRudpWrite(int ctxId, const void* data, uint32 len, int flags)
fn cb_write(ctx: &mut CpuContext) -> HleResult {
    with_context(ctx.rdi() as i32, |_| ERROR_CONN_RESET)
}

/// sceRudpRead(int ctxId, void* data, uint32 len, int flags, void* info)
fn cb_read(ctx: &mut CpuContext) -> HleResult {
    with_context(ctx.rdi() as i32, |_| ERROR_CONN_RESET)
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("MVbmLASjn5M", "sceRudpPollCreate", cb_poll_create),
    ("J-6d0WTjzMc", "sceRudpActivate", cb_activate),
    ("i3STzxuwPx0", "sceRudpGetStatus", cb_get_status),
    ("M6ggviwXpLs", "sceRudpPollWait", cb_poll_wait),
    ("l4SLBpKUDK4", "sceRudpBind", cb_bind),
    ("CAbbX6BuQZ0", "sceRudpCreateContext", cb_create_context),
    ("6PBNpsgyaxw", "sceRudpEnableInternalIOThread", cb_enable_internal_io_thread),
    ("3hBvwqEwqj8", "sceRudpEnd", cb_end),
    ("wIJsiqY+BMk", "sceRudpGetContextStatus", cb_get_context_status),
    ("amuBfI-AQc4", "sceRudpInit", cb_init),
    ("szEVu+edXV4", "sceRudpInitiate", cb_initiate),
    ("rZqWV3eXgOA", "sceRudpRead", cb_read),
    ("SUEVes8gvmw", "sceRudpSetEventHandler", cb_set_event_handler),
    ("0yzYdZf0IwE", "sceRudpSetOption", cb_set_option),
    ("OMYRTU0uc4w", "sceRudpTerminate", cb_terminate),
    ("KaPL3fbTLCA", "sceRudpWrite", cb_write),
    ("Px0miD2LuW0", "sceRudpGetNumberOfPacketsToRead", cb_get_number_of_packets_to_read),
    ("fRc1ahQppR4", "sceRudpGetSizeWritable", cb_get_size_unconnected),
    ("sAZqO2+5Qqo", "sceRudpGetSizeReadable", cb_get_size_unconnected),
    ("+BJ9svDmjYs", "sceRudpNetFlush", cb_initialized_only),
    ("9U9m1YH0ScQ", "sceRudpProcessEvents", cb_initialized_only),
    ("vfrL8gPlm2Y", "sceRudpGetMaxSegmentSize", cb_get_max_segment_size),
    ("mCQIhSmCP6o", "sceRudpGetOption", cb_get_option),
    ("haMpc7TFx0A", "sceRudpPollControl", cb_poll_control),
    ("LjwbHpEeW0A", "sceRudpPollDestroy", cb_poll_destroy),
    ("fJ51weR1WAI", "sceRudpEnableInternalIOThread2", cb_enable_internal_io_thread),
    ("vPzJldDSxXc", "sceRudpNetReceived", cb_initialized_only),
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

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// The library state is process-wide: its tests run one at a time.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn call(ctx: &mut CpuContext, cb: HleCallback, args: &[u64]) -> HleResult {
        for (r, v) in [Rdi, Rsi, Rdx, Rcx].into_iter().zip(args) {
            ctx.set(r, *v);
        }
        cb(ctx)
    }

    /// Poll objects, the context options (reference GuestRudp cases).
    #[test]
    fn poll_objects_and_options() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
        const BASE: u64 = 0x5DA7_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(vm);
        assert_eq!(call(&mut ctx, cb_poll_create, &[16]), ERROR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, cb_init, &[BASE + 0x800, 0x1000]), 0);
        assert_eq!(call(&mut ctx, cb_poll_create, &[0]), ERROR_INVALID_ARGUMENT);
        let id = call(&mut ctx, cb_poll_create, &[1]);
        assert!(id > 0);
        assert_eq!(call(&mut ctx, cb_create_context, &[0, 0, BASE]), 0);
        let c = ctx.read_u32(BASE).unwrap() as u64;
        assert_eq!(call(&mut ctx, cb_create_context, &[0, 0, BASE]), 0);
        let c2 = ctx.read_u32(BASE).unwrap() as u64;
        // Control: argument checks, then the poll, then the context.
        assert_eq!(call(&mut ctx, cb_poll_control, &[id as u64, 4, c, 1]), ERROR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_poll_control, &[u32::MAX as u64, 1, c, 1]), ERROR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_poll_control, &[0x7FFF_0000, 1, c, 1]), ERROR_INVALID_POLL_ID);
        assert_eq!(call(&mut ctx, cb_poll_control, &[id as u64, 1, 999, 1]), ERROR_INVALID_CONTEXT_ID);
        assert_eq!(call(&mut ctx, cb_poll_control, &[id as u64, 2, c, 1]), ERROR_INVALID_CONTEXT_ID);
        assert_eq!(call(&mut ctx, cb_poll_control, &[id as u64, 1, c, 1]), 0);
        // Full is checked before already present.
        assert_eq!(call(&mut ctx, cb_poll_control, &[id as u64, 1, c, 1]), ERROR_TOO_MANY_CONTEXTS);
        assert_eq!(call(&mut ctx, cb_poll_control, &[id as u64, 1, c2, 1]), ERROR_TOO_MANY_CONTEXTS);
        assert_eq!(call(&mut ctx, cb_poll_control, &[id as u64, 2, c, 3]), 0);
        assert_eq!(call(&mut ctx, cb_poll_control, &[id as u64, 3, c, 0]), 0);
        // Wait: the timeout runs out with no event.
        assert_eq!(call(&mut ctx, cb_poll_wait, &[id as u64, BASE, 4, 0]), 0);
        assert_eq!(call(&mut ctx, cb_poll_wait, &[id as u64, BASE, 4, 1000]), 0);
        assert_eq!(call(&mut ctx, cb_poll_wait, &[id as u64, BASE, 0, 0]), ERROR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_poll_wait, &[0x7FFF_0000, BASE, 4, 0]), ERROR_INVALID_POLL_ID);
        // A destroy ends a wait in progress.
        let waiter = std::thread::spawn(move || {
            let mut ctx = CpuContext::new(vm);
            call(&mut ctx, cb_poll_wait, &[id as u64, BASE, 4, u64::MAX])
        });
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(call(&mut ctx, cb_poll_destroy, &[u32::MAX as u64]), ERROR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_poll_destroy, &[id as u64]), 0);
        assert_eq!(waiter.join().unwrap(), ERROR_INVALID_POLL_ID);
        assert_eq!(call(&mut ctx, cb_poll_destroy, &[id as u64]), ERROR_INVALID_POLL_ID);
        // Options: context first, then the id, then a 4-byte value.
        ctx.write_u32(BASE + 0x10, 77);
        assert_eq!(call(&mut ctx, cb_set_option, &[999, 1, BASE + 0x10, 4]), ERROR_INVALID_CONTEXT_ID);
        assert_eq!(call(&mut ctx, cb_set_option, &[c, 20, BASE + 0x10, 4]), ERROR_INVALID_OPTION);
        assert_eq!(call(&mut ctx, cb_set_option, &[c, 0, BASE + 0x10, 4]), ERROR_INVALID_OPTION);
        assert_eq!(call(&mut ctx, cb_set_option, &[c, 1, 0, 4]), ERROR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_set_option, &[c, 1, BASE + 0x10, 8]), ERROR_INVALID_ARGUMENT);
        for (option, default) in [(1u64, 1346u32), (2, 65536), (7, 1), (9, 60000), (11, 30), (19, 0)] {
            assert_eq!(call(&mut ctx, cb_get_option, &[c, option, BASE + 0x20, 4]), 0);
            assert_eq!(ctx.read_u32(BASE + 0x20), Some(default), "option {option}");
        }
        assert_eq!(call(&mut ctx, cb_set_option, &[c, 9, BASE + 0x10, 4]), 0);
        assert_eq!(call(&mut ctx, cb_get_option, &[c, 9, BASE + 0x20, 4]), 0);
        assert_eq!(ctx.read_u32(BASE + 0x20), Some(77));
        assert_eq!(call(&mut ctx, cb_set_option, &[c, 13, BASE + 0x10, 4]), 0);
        assert_eq!(call(&mut ctx, cb_get_option, &[c, 13, BASE + 0x20, 4]), ERROR_INVALID_OPTION);
        // The end of the library ends a wait in progress.
        let id = call(&mut ctx, cb_poll_create, &[4]);
        let waiter = std::thread::spawn(move || {
            let mut ctx = CpuContext::new(vm);
            call(&mut ctx, cb_poll_wait, &[id as u64, BASE, 4, 10_000_000])
        });
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(cb_end(&mut ctx), 0);
        assert_eq!(waiter.join().unwrap(), ERROR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, cb_poll_destroy, &[id as u64]), ERROR_NOT_INITIALIZED);
    }

    #[test]
    fn offline_lifecycle() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C05_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);

        assert_eq!(cb_create_context(&mut ctx), ERROR_INVALID_ARGUMENT);
        ctx.set(Rdx, BASE);
        assert_eq!(cb_create_context(&mut ctx), ERROR_NOT_INITIALIZED);

        // Library status: refused before init without touching the buffer
        // (whatever the arguments). Activation works either way.
        const STATUS: u64 = BASE + 0x800;
        const POOL: u64 = BASE + 0x100;
        let status = |ctx: &mut CpuContext, at: u64, size: u64| {
            ctx.set(Rdi, at);
            ctx.set(Rsi, size);
            cb_get_status(ctx)
        };
        let init = |ctx: &mut CpuContext, pool: u64, size: i64| {
            ctx.set(Rdi, pool);
            ctx.set(Rsi, size as u64);
            cb_init(ctx)
        };
        let query = |ctx: &mut CpuContext, cb: HleCallback, id: i64| {
            ctx.set(Rdi, id as u64);
            cb(ctx)
        };
        ctx.write_bytes(STATUS, &[0x5A; 0x200]);
        assert_eq!(cb_activate(&mut ctx), 0);
        for (at, size) in [(STATUS, 0xF8), (0, 0), (STATUS, 0xF9)] {
            assert_eq!(status(&mut ctx, at, size), ERROR_NOT_INITIALIZED);
        }
        assert_eq!(ctx.read_bytes(STATUS, 0x200).unwrap(), vec![0x5A; 0x200]);
        for cb in [cb_get_number_of_packets_to_read as HleCallback, cb_get_size_unconnected] {
            assert_eq!(query(&mut ctx, cb, 0), ERROR_NOT_INITIALIZED);
        }
        // The library calls need sceRudpInit first.
        ctx.set(Rdi, BASE + 0x40);
        for cb in [cb_initialized_only as HleCallback, cb_get_max_segment_size, cb_enable_internal_io_thread] {
            assert_eq!(cb(&mut ctx), ERROR_NOT_INITIALIZED);
        }
        // Init: the pool checks measured on a console.
        assert_eq!(init(&mut ctx, 0, 0x10000), ERROR_INVALID_ARGUMENT);
        assert_eq!(init(&mut ctx, POOL, 0), ERROR_INVALID_ARGUMENT);
        assert_eq!(init(&mut ctx, POOL, -1), ERROR_INVALID_ARGUMENT);
        assert_eq!(init(&mut ctx, POOL, 0xC57), ERROR_MEMORY);
        for off in [1, 2, 4] {
            assert_eq!(init(&mut ctx, POOL + off, 0x1000), ERROR_MEMORY);
        }
        assert_eq!(init(&mut ctx, POOL + 8, 0xC58), 0);
        assert_eq!(init(&mut ctx, 0, 0), ERROR_ALREADY_INITIALIZED);
        assert_eq!(cb_initialized_only(&mut ctx), 0);
        ctx.set(Rdi, 0);
        assert_eq!(cb_get_max_segment_size(&mut ctx), ERROR_INVALID_ARGUMENT);
        ctx.set(Rdi, BASE + 0x40);
        assert_eq!(cb_get_max_segment_size(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x40, 2).unwrap(), 1410u16.to_le_bytes().to_vec());
        assert_eq!(cb_activate(&mut ctx), 0);
        // Status after init: argument checks, then the 0xF8 bytes.
        for (at, size) in [(0, 0xF8), (0, 0), (STATUS, 0), (STATUS, 0xF9)] {
            assert_eq!(status(&mut ctx, at, size), ERROR_INVALID_ARGUMENT);
        }
        assert_eq!(ctx.read_bytes(STATUS, 0x200).unwrap(), vec![0x5A; 0x200]);
        assert_eq!(status(&mut ctx, STATUS, 0xF8), 0);
        assert_eq!(ctx.read_bytes(STATUS, 0xF8).unwrap(), vec![0; 0xF8]);
        assert_eq!(ctx.read_bytes(STATUS + 0xF8, 8).unwrap(), vec![0x5A; 8]);
        // Context queries on unknown or negative ids.
        for cb in [cb_get_number_of_packets_to_read as HleCallback, cb_get_size_unconnected] {
            assert_eq!(query(&mut ctx, cb, 0), ERROR_INVALID_CONTEXT_ID);
            assert_eq!(query(&mut ctx, cb, 12345), ERROR_INVALID_CONTEXT_ID);
            assert_eq!(query(&mut ctx, cb, -1), ERROR_INVALID_ARGUMENT);
        }
        ctx.set(Rdx, BASE);
        // The handler and I/O thread calls always succeed.
        assert_eq!(cb_enable_internal_io_thread(&mut ctx), 0);
        assert_eq!(cb_create_context(&mut ctx), 0);
        let id = ctx.read_u32(BASE).unwrap() as u64;
        ctx.set(Rdx, BASE + 4);
        assert_eq!(cb_create_context(&mut ctx), 0);
        let second = ctx.read_u32(BASE + 4).unwrap() as u64;
        ctx.write_bytes(STATUS, &[0x5A; 0x200]);
        assert_eq!(status(&mut ctx, STATUS, 0xF8), 0);
        assert_eq!(ctx.read_u32(STATUS + 0x64), Some(2), "live contexts");
        ctx.write_bytes(STATUS, &[0x5A; 0x200]);
        assert_eq!(status(&mut ctx, STATUS, 1), 0);
        assert_eq!(ctx.read_bytes(STATUS, 2).unwrap(), vec![0, 0x5A], "only the bytes asked for");
        assert_eq!(query(&mut ctx, cb_get_number_of_packets_to_read, id as i64), 0);
        assert_eq!(query(&mut ctx, cb_get_size_unconnected, id as i64), ERROR_NOT_ACCEPTABLE);
        ctx.set(Rdi, second);
        assert_eq!(cb_terminate(&mut ctx), 0);
        assert_eq!(query(&mut ctx, cb_get_size_unconnected, second as i64), ERROR_INVALID_CONTEXT_ID);

        ctx.set(Rdi, id);
        assert_eq!(cb_bind(&mut ctx), 0);
        // The context is checked before the address.
        ctx.set(Rsi, 0);
        assert_eq!(cb_initiate(&mut ctx), ERROR_INVALID_ARGUMENT);
        ctx.set(Rdi, 12345);
        assert_eq!(cb_initiate(&mut ctx), ERROR_INVALID_CONTEXT_ID);
        ctx.set(Rdi, id);
        ctx.set(Rsi, BASE + 0x100);
        assert_eq!(cb_initiate(&mut ctx), ERROR_CONN_REFUSED);
        assert_eq!(cb_write(&mut ctx), ERROR_CONN_RESET);
        ctx.write_bytes(BASE + 0x200, &[0xAA; 16]);
        ctx.set(Rsi, BASE + 0x200);
        ctx.set(Rdx, 16);
        assert_eq!(cb_get_context_status(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x200, 16).unwrap(), vec![0; 16]);

        assert_eq!(cb_terminate(&mut ctx), 0);
        assert_eq!(cb_terminate(&mut ctx), ERROR_INVALID_CONTEXT_ID);
        assert_eq!(cb_end(&mut ctx), 0);
        assert_eq!(cb_end(&mut ctx), ERROR_NOT_INITIALIZED);
    }
}
