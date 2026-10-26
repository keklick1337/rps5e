// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceNetCtl — network connection state, offline.
//!
//! The console is disconnected: the state is DISCONNECTED, connection info
//! queries fail with NOT_CONNECTED, and registered callbacks receive the
//! DISCONNECTED event once, from sceNetCtlCheckCallback. IPv4 and IPv6
//! callbacks live in separate tables with their own ids.

use std::sync::Mutex;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceNetCtl";

const MAX_CALLBACKS: usize = 8;
/// SceNetCtlNatInfo: size, stunStatus, natType, mappedAddr.
const NAT_INFO_SIZE: usize = 16;
/// SceNetCtlInfo: a union whose largest member is a 256-byte string.
const INFO_SIZE: usize = 256;

const ERR_CALLBACK_MAX: HleResult = 0x8041_2103u32 as i32 as i64;
const ERR_ID_NOT_FOUND: HleResult = 0x8041_2104u32 as i32 as i64;
const ERR_INVALID_ID: HleResult = 0x8041_2105u32 as i32 as i64;
const ERR_INVALID_CODE: HleResult = 0x8041_2106u32 as i32 as i64;
const ERR_INVALID_ADDR: HleResult = 0x8041_2107u32 as i32 as i64;
const ERR_NOT_CONNECTED: HleResult = 0x8041_2108u32 as i32 as i64;
const ERR_INVALID_TYPE: HleResult = 0x8041_210Fu32 as i32 as i64;
const ERR_INVALID_ARGUMENT: HleResult = 0x8002_0016u32 as i32 as i64;
const ERR_MEMORY_FAULT: HleResult = 0x8002_000Eu32 as i32 as i64;

const STATE_DISCONNECTED: u32 = 0;
const EVENT_DISCONNECTED: i32 = 1;

#[derive(Clone, Copy, Default)]
struct CallbackSlot {
    func: u64,
    arg: u64,
    last_event: i32,
}

type CallbackTable = Mutex<[CallbackSlot; MAX_CALLBACKS]>;

const NO_CALLBACK: CallbackSlot = CallbackSlot { func: 0, arg: 0, last_event: 0 };

/// sceNetCtlRegisterCallback and sceNetCtlRegisterCallbackV6 tables.
static CALLBACKS: CallbackTable = Mutex::new([NO_CALLBACK; MAX_CALLBACKS]);
static CALLBACKS_V6: CallbackTable = Mutex::new([NO_CALLBACK; MAX_CALLBACKS]);

fn table(t: &'static CallbackTable) -> std::sync::MutexGuard<'static, [CallbackSlot; MAX_CALLBACKS]> {
    t.lock().unwrap_or_else(|p| p.into_inner())
}

fn callbacks() -> std::sync::MutexGuard<'static, [CallbackSlot; MAX_CALLBACKS]> {
    table(&CALLBACKS)
}

fn written(ok: bool) -> HleResult {
    if ok { 0 } else { ERR_MEMORY_FAULT }
}

fn cb_init(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// sceNetCtlTerm(): drops every registered callback.
fn cb_term(_ctx: &mut CpuContext) -> HleResult {
    *callbacks() = [CallbackSlot::default(); MAX_CALLBACKS];
    *table(&CALLBACKS_V6) = [CallbackSlot::default(); MAX_CALLBACKS];
    0
}

/// sceNetCtlGetNatInfo(SceNetCtlNatInfo* info): size must be 16; no NAT
/// information exists while disconnected.
fn cb_get_nat_info(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return ERR_INVALID_ADDR;
    }
    let Some(size) = ctx.read_u32(addr) else {
        return ERR_MEMORY_FAULT;
    };
    if size != NAT_INFO_SIZE as u32 {
        return ERR_INVALID_ARGUMENT;
    }
    written(ctx.write_bytes(addr + 4, &[0u8; NAT_INFO_SIZE - 4]))
}

/// Callbacks that have not seen `event` yet; marks them as having seen it.
fn take_pending(event: i32) -> Vec<(u64, u64)> {
    let mut pending = take_pending_in(&CALLBACKS, event);
    pending.extend(take_pending_in(&CALLBACKS_V6, event));
    pending
}

fn take_pending_in(t: &'static CallbackTable, event: i32) -> Vec<(u64, u64)> {
    let mut cbs = table(t);
    cbs.iter_mut()
        .filter(|c| c.func != 0 && c.last_event != event)
        .map(|c| {
            c.last_event = event;
            (c.func, c.arg)
        })
        .collect()
}

/// sceNetCtlCheckCallback(): delivers the DISCONNECTED event to every
/// callback that has not received it yet.
fn cb_check_callback(_ctx: &mut CpuContext) -> HleResult {
    for (func, arg) in take_pending(EVENT_DISCONNECTED) {
        ps5cpu::exec::call_guest_nested(func, [EVENT_DISCONNECTED as u64, arg, 0, 0, 0, 0]);
    }
    0
}

/// sceNetCtlGetState(int* state) / sceNetCtlGetStateV6: always disconnected.
fn cb_get_state(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return ERR_INVALID_ADDR;
    }
    written(ctx.write_u32(addr, STATE_DISCONNECTED))
}

/// sceNetCtlRegisterCallback(func, arg, int* cid)
fn cb_register_callback(ctx: &mut CpuContext) -> HleResult {
    register_in(ctx, &CALLBACKS)
}

/// sceNetCtlRegisterCallbackV6(func, arg, int* cid)
fn cb_register_callback_v6(ctx: &mut CpuContext) -> HleResult {
    register_in(ctx, &CALLBACKS_V6)
}

fn register_in(ctx: &CpuContext, t: &'static CallbackTable) -> HleResult {
    let (func, arg, id_addr) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if func == 0 || id_addr == 0 {
        return ERR_INVALID_ADDR;
    }
    let mut cbs = table(t);
    let Some(id) = cbs.iter().position(|c| c.func == 0) else {
        return ERR_CALLBACK_MAX;
    };
    if !ctx.write_u32(id_addr, id as u32) {
        return ERR_MEMORY_FAULT;
    }
    cbs[id] = CallbackSlot { func, arg, last_event: 0 };
    0
}

/// sceNetCtlUnregisterCallback(int cid)
fn cb_unregister_callback(ctx: &mut CpuContext) -> HleResult {
    unregister_in(ctx.rdi() as i32, &CALLBACKS)
}

/// sceNetCtlUnregisterCallbackV6(int cid): ids of the IPv6 table.
fn cb_unregister_callback_v6(ctx: &mut CpuContext) -> HleResult {
    unregister_in(ctx.rdi() as i32, &CALLBACKS_V6)
}

fn unregister_in(cid: i32, t: &'static CallbackTable) -> HleResult {
    if cid < 0 || cid as usize >= MAX_CALLBACKS {
        return ERR_INVALID_ID;
    }
    let mut cbs = table(t);
    if cbs[cid as usize].func == 0 {
        return ERR_ID_NOT_FOUND;
    }
    cbs[cid as usize] = CallbackSlot::default();
    0
}

/// sceNetCtlGetResult(int eventType, int* errorCode): events complete
/// without error.
fn cb_get_result(ctx: &mut CpuContext) -> HleResult {
    let (event, out) = (ctx.rdi() as i32, ctx.rsi());
    if out == 0 {
        return ERR_INVALID_ADDR;
    }
    if !(1..=3).contains(&event) {
        return ERR_INVALID_TYPE;
    }
    written(ctx.write_u32(out, 0))
}

/// sceNetCtlGetInfo(int code, SceNetCtlInfo* info): the info is cleared
/// and nothing is available while disconnected.
fn cb_get_info(ctx: &mut CpuContext) -> HleResult {
    let (code, addr) = (ctx.rdi() as i32, ctx.rsi());
    if addr == 0 {
        return ERR_INVALID_ADDR;
    }
    // Codes 1 (device) .. 21 (HTTP proxy port).
    if !(1..=21).contains(&code) {
        return ERR_INVALID_CODE;
    }
    if !ctx.write_bytes(addr, &[0u8; INFO_SIZE]) {
        return ERR_MEMORY_FAULT;
    }
    ERR_NOT_CONNECTED
}

/// sceNetCtlGetInfoV6(int code, SceNetCtlInfoV6* info): codes 1..5; nothing
/// is available while disconnected and the info is left alone.
fn cb_get_info_v6(ctx: &mut CpuContext) -> HleResult {
    let (code, addr) = (ctx.rdi() as i32, ctx.rsi());
    if addr == 0 {
        return ERR_INVALID_ADDR;
    }
    if !(1..=5).contains(&code) {
        return ERR_INVALID_CODE;
    }
    ERR_NOT_CONNECTED
}

/// sceNetCtlGetIfStat(SceNetCtlIfStat* stat): there are no interface
/// statistics while disconnected; the output is left untouched.
fn cb_get_if_stat(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 { ERR_INVALID_ADDR } else { ERR_NOT_CONNECTED }
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("gky0+oaNM4k", "sceNetCtlInit", cb_init),
    ("JO4yuTuMoKI", "sceNetCtlGetNatInfo", cb_get_nat_info),
    ("iQw3iQPhvUQ", "sceNetCtlCheckCallback", cb_check_callback),
    ("uBPlr0lbuiI", "sceNetCtlGetState", cb_get_state),
    ("+lxqIKeU9UY", "sceNetCtlGetStateV6", cb_get_state),
    ("UJ+Z7Q+4ck0", "sceNetCtlRegisterCallback", cb_register_callback),
    ("1NE9OWdBIww", "sceNetCtlRegisterCallbackV6", cb_register_callback_v6),
    ("hIUVeUNxAwc", "sceNetCtlUnregisterCallbackV6", cb_unregister_callback_v6),
    ("teuK4QnJTGg", "sceNetCtlGetIfStat", cb_get_if_stat),
    ("Rqm2OnZMCz0", "sceNetCtlUnregisterCallback", cb_unregister_callback),
    ("0cBgduPRR+M", "sceNetCtlGetResult", cb_get_result),
    ("H5yARg37U5g", "sceNetCtlGetResultV6", cb_get_result),
    ("Jy1EO5GdlcM", "sceNetCtlGetInfoV6", cb_get_info_v6),
    ("obuxdTiwkF8", "sceNetCtlGetInfo", cb_get_info),
    ("Z4wwCFiBELQ", "sceNetCtlTerm", cb_term),
    // The NpToolkit and internal (IpcInt) entries are the same calls on the
    // same state.
    ("wIsKy+TfeLs", "sceNetCtlRegisterCallbackForNpToolkit", cb_register_callback),
    ("2oUqKR5odGc", "sceNetCtlUnregisterCallbackForNpToolkit", cb_unregister_callback),
    ("NEtnusbZyAs", "sceNetCtlGetResultIpcInt", cb_get_result),
    ("gvnJPMkSoAY", "sceNetCtlGetStateIpcInt", cb_get_state),
    ("xstcTqAhTys", "sceNetCtlGetInfoIpcInt", cb_get_info),
    ("ipqlpcIqRsQ", "sceNetCtlRegisterCallbackV6IpcInt", cb_register_callback_v6),
    ("Hjxpy28aID8", "sceNetCtlUnregisterCallbackV6IpcInt", cb_unregister_callback_v6),
    ("vdsTa93atXY", "sceNetCtlGetResultV6IpcInt", cb_get_result),
    ("O8Fk4w5MWss", "sceNetCtlGetStateV6IpcInt", cb_get_state),
    ("arAQRFlwqaA", "sceNetCtlGetInfoV6IpcInt", cb_get_info_v6),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
    for &(nid, name, cb) in ap::EXPORTS {
        reg.register_fn(nid, name, "libSceNetCtlAp", cb);
    }
}

/// libSceNetCtlAp: the access-point (tethering) service, which never
/// starts on this console; the remote-play access point calls answer 0.
mod ap {
    use super::*;

    const ERR_NOT_STARTED: HleResult = 0x8041_298Fu32 as i32 as i64;
    const ERR_INVALID_TYPE: HleResult = 0x8041_2990u32 as i32 as i64;
    const ERR_INVALID_ADDR: HleResult = 0x8041_2991u32 as i32 as i64;
    const ERR_NOT_INITIALIZED: HleResult = 0x8041_2994u32 as i32 as i64;
    const ERR_CALLBACK_MAX: HleResult = 0x8041_2995u32 as i32 as i64;
    const ERR_INVALID_ID: HleResult = 0x8041_2996u32 as i32 as i64;
    const ERR_ID_NOT_FOUND: HleResult = 0x8041_2997u32 as i32 as i64;
    const ERR_INVALID_SIZE: HleResult = 0x8041_2998u32 as i32 as i64;
    /// SceNetCtlApInfo and SceNetCtlApConnectInfo sizes the caller declares.
    const INFO_SIZE: u64 = 152;
    const CONNECT_INFO_SIZE: u64 = 184;

    #[derive(Default)]
    struct State {
        initialized: bool,
        callbacks: [(u64, u64); MAX_CALLBACKS],
    }

    static STATE: Mutex<State> = Mutex::new(State { initialized: false, callbacks: [(0, 0); MAX_CALLBACKS] });

    fn state() -> std::sync::MutexGuard<'static, State> {
        STATE.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn initialized() -> bool {
        state().initialized
    }

    /// sceNetCtlApInit()
    pub(super) fn init(_ctx: &mut CpuContext) -> HleResult {
        let mut st = state();
        if st.initialized {
            log::warn!("[NetCtlAp] sceNetCtlApInit: already initialized");
        }
        st.initialized = true;
        0
    }

    /// sceNetCtlApTerm()
    pub(super) fn term(_ctx: &mut CpuContext) -> HleResult {
        *state() = State::default();
        0
    }

    pub(super) fn check_callback(_ctx: &mut CpuContext) -> HleResult {
        if initialized() { 0 } else { ERR_NOT_INITIALIZED }
    }

    /// sceNetCtlApClearEvent(int type)
    pub(super) fn clear_event(ctx: &mut CpuContext) -> HleResult {
        if !initialized() {
            return ERR_NOT_INITIALIZED;
        }
        if (1..=4).contains(&(ctx.rdi() as i32)) { 0 } else { ERR_INVALID_TYPE }
    }

    /// sceNetCtlApRegisterCallback(func, arg, int* cid)
    pub(super) fn register_callback(ctx: &mut CpuContext) -> HleResult {
        let (func, arg, cid) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
        let mut st = state();
        if !st.initialized {
            return ERR_NOT_INITIALIZED;
        }
        if func == 0 || cid == 0 {
            return ERR_INVALID_ADDR;
        }
        let Some(slot) = st.callbacks.iter().position(|c| c.0 == 0) else { return ERR_CALLBACK_MAX };
        if !ctx.write_u32(cid, slot as u32) {
            return ERR_MEMORY_FAULT;
        }
        st.callbacks[slot] = (func, arg);
        0
    }

    /// sceNetCtlApUnregisterCallback(int cid)
    pub(super) fn unregister_callback(ctx: &mut CpuContext) -> HleResult {
        let cid = ctx.rdi() as i32;
        let mut st = state();
        if !st.initialized {
            return ERR_NOT_INITIALIZED;
        }
        if !(0..MAX_CALLBACKS as i32).contains(&cid) {
            return ERR_INVALID_ID;
        }
        if st.callbacks[cid as usize].0 == 0 {
            return ERR_ID_NOT_FOUND;
        }
        st.callbacks[cid as usize] = (0, 0);
        0
    }

    /// sceNetCtlApGetResult(int type, int* error)
    pub(super) fn get_result(ctx: &mut CpuContext) -> HleResult {
        let (ty, out) = (ctx.rdi() as i32, ctx.rsi());
        if !initialized() {
            return ERR_NOT_INITIALIZED;
        }
        if !(1..=4).contains(&ty) {
            return ERR_INVALID_TYPE;
        }
        if out == 0 {
            return ERR_INVALID_ADDR;
        }
        written(ctx.write_u32(out, 0))
    }

    /// Checks of a sized info struct: its first u64 must be `size`.
    fn sized_info(ctx: &CpuContext, info: u64, size: u64) -> HleResult {
        if !initialized() {
            return ERR_NOT_INITIALIZED;
        }
        if info == 0 {
            return ERR_INVALID_ADDR;
        }
        match ctx.read_u64(info) {
            None => ERR_MEMORY_FAULT,
            Some(s) if s != size => ERR_INVALID_SIZE,
            Some(_) => ERR_NOT_STARTED,
        }
    }

    /// sceNetCtlApGetInfo(SceNetCtlApInfo* info)
    pub(super) fn get_info(ctx: &mut CpuContext) -> HleResult {
        sized_info(ctx, ctx.rdi(), INFO_SIZE)
    }

    /// sceNetCtlApGetConnectInfo(const char* opt, SceNetCtlApConnectInfo* info)
    pub(super) fn get_connect_info(ctx: &mut CpuContext) -> HleResult {
        sized_info(ctx, ctx.rsi(), CONNECT_INFO_SIZE)
    }

    /// sceNetCtlApGetState(int* state): STOPPED.
    pub(super) fn get_state(ctx: &mut CpuContext) -> HleResult {
        let out = ctx.rdi();
        if !initialized() {
            return ERR_NOT_INITIALIZED;
        }
        if out == 0 {
            return ERR_INVALID_ADDR;
        }
        written(ctx.write_u32(out, 0))
    }

    /// sceNetCtlApStop()
    pub(super) fn stop(_ctx: &mut CpuContext) -> HleResult {
        if initialized() { ERR_NOT_STARTED } else { ERR_NOT_INITIALIZED }
    }

    fn ok(_ctx: &mut CpuContext) -> HleResult {
        0
    }

    pub(super) const EXPORTS: &[(&str, &str, HleCallback)] = &[
        ("FdN+edNRtiw", "sceNetCtlApInit", init),
        ("cv5Y2efOTeg", "sceNetCtlApTerm", term),
        ("19Ec7WkMFfQ", "sceNetCtlApCheckCallback", check_callback),
        ("meFMaDpdsVI", "sceNetCtlApClearEvent", clear_event),
        ("pmjobSVHuY0", "sceNetCtlApRegisterCallback", register_callback),
        ("NpTcFtaQ-0E", "sceNetCtlApUnregisterCallback", unregister_callback),
        ("4jkLJc954+Q", "sceNetCtlApGetResult", get_result),
        ("LXADzTIzM9I", "sceNetCtlApGetInfo", get_info),
        ("AKZOzsb9whc", "sceNetCtlApGetState", get_state),
        ("r-pOyN6AhsM", "sceNetCtlApStop", stop),
        ("hfkLVdXmfnU", "sceNetCtlApGetConnectInfo", get_connect_info),
        // The remote-play access point: what the firmware returns.
        ("DufQZgH5ISc", "sceNetCtlApRpCheckCallback", ok),
        ("qhZbOi+2qLY", "sceNetCtlApRpClearEvent", ok),
        ("ofGsK+xoAaM", "sceNetCtlApRpRegisterCallback", ok),
        ("8eyH37Ns8tk", "sceNetCtlApRpUnregisterCallback", ok),
        ("3pxwYqHzGcw", "sceNetCtlApRpGetResult", ok),
        ("LEn8FGztKWc", "sceNetCtlApRpGetState", ok),
        ("VQl16Q+qXeY", "sceNetCtlApRpGetInfo", ok),
        ("mjFgpqNavHg", "sceNetCtlApRpStart", ok),
        ("9Dxg7XSlr2s", "sceNetCtlApRpStartWithRetry", ok),
        ("HMvaHoZWsn8", "sceNetCtlApRpStartConf", ok),
        ("6uvAl4RlEyk", "sceNetCtlApRpStop", ok),
    ];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    /// sceNetCtlGetInfoV6 / GetResultV6 and the access-point service that
    /// never starts (reference GuestSceNet / GuestNetCtlAp cases).
    #[test]
    fn info_v6_and_access_point() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A5E_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let call = |ctx: &mut CpuContext, cb: HleCallback, a: &[u64]| {
            for (r, v) in [Rdi, Rsi, Rdx].into_iter().zip(a) {
                ctx.set(r, *v);
            }
            cb(ctx)
        };
        assert_eq!(call(&mut ctx, cb_get_info_v6, &[1, 0]), ERR_INVALID_ADDR);
        assert_eq!(call(&mut ctx, cb_get_info_v6, &[6, BASE]), ERR_INVALID_CODE);
        assert_eq!(call(&mut ctx, cb_get_info_v6, &[5, BASE]), ERR_NOT_CONNECTED);
        let _g = AP_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        ap::term(&mut ctx);
        assert_eq!(call(&mut ctx, ap::get_state, &[BASE]), 0x8041_2994u32 as i32 as i64);
        assert_eq!(ap::init(&mut ctx), 0);
        ctx.write_u32(BASE, 7);
        assert_eq!(call(&mut ctx, ap::get_state, &[BASE]), 0);
        assert_eq!(ctx.read_u32(BASE), Some(0));
        assert_eq!(call(&mut ctx, ap::clear_event, &[5]), 0x8041_2990u32 as i32 as i64);
        assert_eq!(call(&mut ctx, ap::get_result, &[4, BASE]), 0);
        for slot in 0..MAX_CALLBACKS as u32 {
            assert_eq!(call(&mut ctx, ap::register_callback, &[0x1000, 0, BASE + 8]), 0);
            assert_eq!(ctx.read_u32(BASE + 8), Some(slot));
        }
        assert_eq!(call(&mut ctx, ap::register_callback, &[0x1000, 0, BASE + 8]), 0x8041_2995u32 as i32 as i64);
        assert_eq!(call(&mut ctx, ap::unregister_callback, &[8]), 0x8041_2996u32 as i32 as i64);
        assert_eq!(call(&mut ctx, ap::unregister_callback, &[3]), 0);
        assert_eq!(call(&mut ctx, ap::unregister_callback, &[3]), 0x8041_2997u32 as i32 as i64);
        ctx.write_u64(BASE + 0x100, 151);
        assert_eq!(call(&mut ctx, ap::get_info, &[BASE + 0x100]), 0x8041_2998u32 as i32 as i64);
        ctx.write_u64(BASE + 0x100, 152);
        assert_eq!(call(&mut ctx, ap::get_info, &[BASE + 0x100]), 0x8041_298Fu32 as i32 as i64);
        ctx.write_u64(BASE + 0x100, 184);
        assert_eq!(call(&mut ctx, ap::get_connect_info, &[0, BASE + 0x100]), 0x8041_298Fu32 as i32 as i64);
        assert_eq!(ap::stop(&mut ctx), 0x8041_298Fu32 as i32 as i64);
        ap::term(&mut ctx);
    }

    static AP_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// The NpToolkit and IpcInt twins are registered under their own NIDs
    /// with the public calls behind them.
    #[test]
    fn toolkit_and_ipc_twins_are_the_public_calls() {
        let mut reg = HleRegistry::new();
        register(&mut reg);
        for (twin, public) in [
            ("sceNetCtlRegisterCallbackForNpToolkit", "sceNetCtlRegisterCallback"),
            ("sceNetCtlUnregisterCallbackForNpToolkit", "sceNetCtlUnregisterCallback"),
            ("sceNetCtlGetResultIpcInt", "sceNetCtlGetResult"),
            ("sceNetCtlGetStateIpcInt", "sceNetCtlGetState"),
            ("sceNetCtlGetInfoIpcInt", "sceNetCtlGetInfo"),
            ("sceNetCtlRegisterCallbackV6IpcInt", "sceNetCtlRegisterCallbackV6"),
            ("sceNetCtlUnregisterCallbackV6IpcInt", "sceNetCtlUnregisterCallbackV6"),
            ("sceNetCtlGetResultV6IpcInt", "sceNetCtlGetResultV6"),
            ("sceNetCtlGetStateV6IpcInt", "sceNetCtlGetStateV6"),
            ("sceNetCtlGetInfoV6IpcInt", "sceNetCtlGetInfoV6"),
        ] {
            let find = |name: &str| EXPORTS.iter().find(|e| e.1 == name).map(|e| e.2 as usize);
            assert!(find(twin).is_some() && find(twin) == find(public), "{twin}");
            assert!(reg.lookup(&ps5nid::compute_nid(twin)).is_some(), "{twin}");
        }
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in ap::EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn info_state_and_nat() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A5D_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, &[0xAA; INFO_SIZE + 4]);
        ctx.set(Rdi, 14);
        ctx.set(Rsi, BASE);
        assert_eq!(cb_get_info(&mut ctx), ERR_NOT_CONNECTED);
        assert_eq!(ctx.read_bytes(BASE, INFO_SIZE).unwrap(), vec![0; INFO_SIZE]);
        assert_eq!(ctx.read_bytes(BASE + INFO_SIZE as u64, 1).unwrap(), vec![0xAA]);
        for code in [1u64, 21] {
            ctx.set(Rdi, code);
            assert_eq!(cb_get_info(&mut ctx), ERR_NOT_CONNECTED);
        }
        ctx.write_bytes(BASE, &[0xAA; 4]);
        for code in [0u64, 22] {
            ctx.set(Rdi, code);
            assert_eq!(cb_get_info(&mut ctx), ERR_INVALID_CODE);
        }
        assert_eq!(ctx.read_bytes(BASE, 4).unwrap(), vec![0xAA; 4], "untouched");

        ctx.write_bytes(BASE + 0x200, &[0xFF; 16]);
        ctx.write_u32(BASE + 0x200, 16);
        ctx.set(Rdi, BASE + 0x200);
        assert_eq!(cb_get_nat_info(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x204, 12).unwrap(), vec![0; 12]);
        ctx.write_u32(BASE + 0x200, 12);
        assert_eq!(cb_get_nat_info(&mut ctx), ERR_INVALID_ARGUMENT);

        ctx.write_u32(BASE + 0x300, 7);
        ctx.set(Rdi, BASE + 0x300);
        assert_eq!(cb_get_state(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x300), Some(STATE_DISCONNECTED));
        ctx.set(Rdi, 0);
        assert_eq!(cb_get_state(&mut ctx), ERR_INVALID_ADDR);

        for event in 1..=3 {
            ctx.write_u32(BASE + 0x310, 7);
            ctx.set(Rdi, event);
            ctx.set(Rsi, BASE + 0x310);
            assert_eq!(cb_get_result(&mut ctx), 0);
            assert_eq!(ctx.read_u32(BASE + 0x310), Some(0));
        }
        ctx.set(Rdi, 4);
        assert_eq!(cb_get_result(&mut ctx), ERR_INVALID_TYPE);
        ctx.set(Rsi, 0);
        assert_eq!(cb_get_result(&mut ctx), ERR_INVALID_ADDR);
    }

    #[test]
    fn callback_slots() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C08_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, 0x1234);
        ctx.set(Rsi, 0);
        ctx.set(Rdx, BASE);
        assert_eq!(cb_register_callback(&mut ctx), 0);
        let cid = ctx.read_u32(BASE).unwrap() as u64;
        // The DISCONNECTED event is delivered once per registration.
        let pending = take_pending_in(&CALLBACKS, EVENT_DISCONNECTED);
        assert!(pending.contains(&(0x1234, 0)));
        assert!(!take_pending_in(&CALLBACKS, EVENT_DISCONNECTED).contains(&(0x1234, 0)));
        ctx.set(Rdi, cid);
        assert_eq!(cb_unregister_callback(&mut ctx), 0);
        assert_eq!(cb_unregister_callback(&mut ctx), ERR_ID_NOT_FOUND);
        ctx.set(Rdi, MAX_CALLBACKS as u64);
        assert_eq!(cb_unregister_callback(&mut ctx), ERR_INVALID_ID);
    }

    #[test]
    fn v6_callbacks_have_their_own_ids() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C08_0010_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, 0x6666);
        ctx.set(Rsi, 7);
        ctx.set(Rdx, BASE);
        assert_eq!(cb_register_callback_v6(&mut ctx), 0);
        let cid = ctx.read_u32(BASE).unwrap() as u64;
        // The IPv6 callback receives the DISCONNECTED event too.
        assert!(take_pending_in(&CALLBACKS_V6, EVENT_DISCONNECTED).contains(&(0x6666, 7)));
        ctx.set(Rdi, cid);
        assert_eq!(cb_unregister_callback_v6(&mut ctx), 0);
        assert_eq!(cb_unregister_callback_v6(&mut ctx), ERR_ID_NOT_FOUND);
        for bad in [u64::MAX, MAX_CALLBACKS as u64] {
            ctx.set(Rdi, bad);
            assert_eq!(cb_unregister_callback_v6(&mut ctx), ERR_INVALID_ID);
        }
    }

    #[test]
    fn if_stat_needs_a_connection() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C08_0020_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, &[0xAA; 64]);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_get_if_stat(&mut ctx), ERR_NOT_CONNECTED);
        assert_eq!(ctx.read_bytes(BASE, 64).unwrap(), vec![0xAA; 64]);
        ctx.set(Rdi, 0);
        assert_eq!(cb_get_if_stat(&mut ctx), ERR_INVALID_ADDR);
    }
}
