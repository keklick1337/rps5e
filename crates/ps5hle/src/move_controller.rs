// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceMove: PlayStation Move motion controllers. None is
//! ever connected: the library initializes, handles open and close, and
//! every query of an open handle answers "no controller connected".

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceMove";

const fn err(code: u32) -> HleResult {
    code as i32 as i64
}

const OK: HleResult = 0;
/// A positive answer: the handle is open but no controller is connected.
const NO_CONTROLLER_CONNECTED: HleResult = 1;
const ERROR_NOT_INIT: HleResult = err(0x80EE_0001);
const ERROR_ALREADY_INIT: HleResult = err(0x80EE_0002);
const ERROR_INVALID_ARG: HleResult = err(0x80EE_0003);
const ERROR_INVALID_HANDLE: HleResult = err(0x80EE_0004);
const ERROR_ALREADY_OPENED: HleResult = err(0x80EE_0007);
const NOT_IMPLEMENTED: HleResult = err(0x8002_0002);
const FIRST_HANDLE: i32 = 0x030B_0100;
const HANDLE_STEP: i32 = 0x100;

struct State {
    initialized: bool,
    next_handle: i32,
    /// Open handles: (type, index).
    open: BTreeMap<i32, (i32, i32)>,
}

static STATE: Mutex<State> = Mutex::new(State { initialized: false, next_handle: FIRST_HANDLE, open: BTreeMap::new() });

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|p| p.into_inner())
}

/// NOT_INIT before sceMoveInit, INVALID_HANDLE for a handle not open.
fn check_handle(handle: i32) -> HleResult {
    let s = state();
    if !s.initialized {
        return ERROR_NOT_INIT;
    }
    if s.open.contains_key(&handle) { OK } else { ERROR_INVALID_HANDLE }
}

/// sceMoveInit()
fn cb_init(_ctx: &mut CpuContext) -> HleResult {
    let mut s = state();
    if s.initialized {
        return ERROR_ALREADY_INIT;
    }
    s.initialized = true;
    OK
}

/// sceMoveTerm(): closes every handle.
fn cb_term(_ctx: &mut CpuContext) -> HleResult {
    let mut s = state();
    if !s.initialized {
        return ERROR_NOT_INIT;
    }
    s.initialized = false;
    s.open.clear();
    OK
}

/// sceMoveOpen(userId, type, index) -> handle: one handle per (type,
/// index); only the initial user's controllers are known.
fn cb_open(ctx: &mut CpuContext) -> HleResult {
    let (user, ty, index) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx() as i32);
    let mut s = state();
    if !s.initialized {
        return ERROR_NOT_INIT;
    }
    if user != crate::user_service::PRIMARY_USER_ID {
        log::error!("[move] sceMoveOpen for user {user:#x}: only the initial user is supported");
        return NOT_IMPLEMENTED;
    }
    if s.open.values().any(|&(t, i)| t == ty && i == index) {
        return ERROR_ALREADY_OPENED;
    }
    let handle = s.next_handle;
    s.next_handle = s.next_handle.wrapping_add(HANDLE_STEP);
    s.open.insert(handle, (ty, index));
    handle as HleResult
}

/// sceMoveClose(handle)
fn cb_close(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi() as i32;
    let mut s = state();
    if !s.initialized {
        return ERROR_NOT_INIT;
    }
    if s.open.remove(&handle).is_some() { OK } else { ERROR_INVALID_HANDLE }
}

/// sceMoveGetDeviceInfo / GetExtensionPortInfo / ReadStateLatest(handle, out*)
fn cb_query_with_output(ctx: &mut CpuContext) -> HleResult {
    let r = check_handle(ctx.rdi() as i32);
    if r != OK {
        return r;
    }
    if ctx.rsi() == 0 { ERROR_INVALID_ARG } else { NO_CONTROLLER_CONNECTED }
}

/// sceMoveReadStateRecent(handle, i64 timestamp, data*, int* count)
fn cb_read_state_recent(ctx: &mut CpuContext) -> HleResult {
    let r = check_handle(ctx.rdi() as i32);
    if r != OK {
        return r;
    }
    if (ctx.rsi() as i64) < 0 || ctx.rdx() == 0 || ctx.rcx() == 0 {
        return ERROR_INVALID_ARG;
    }
    NO_CONTROLLER_CONNECTED
}

/// sceMoveSetVibration / sceMoveSetLightSphere(handle, ...)
fn cb_handle_no_controller(ctx: &mut CpuContext) -> HleResult {
    let r = check_handle(ctx.rdi() as i32);
    if r != OK { r } else { NO_CONTROLLER_CONNECTED }
}

/// sceMoveResetLightSphere(handle)
fn cb_reset_light_sphere(ctx: &mut CpuContext) -> HleResult {
    check_handle(ctx.rdi() as i32)
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("j1ITE-EoJmE", "sceMoveInit", cb_init),
    ("tsZi60H4ypY", "sceMoveTerm", cb_term),
    ("HzC60MfjJxU", "sceMoveOpen", cb_open),
    ("XX6wlxpHyeo", "sceMoveClose", cb_close),
    ("GWXTyxs4QbE", "sceMoveGetDeviceInfo", cb_query_with_output),
    ("y5h7f8H1Jnk", "sceMoveGetExtensionPortInfo", cb_query_with_output),
    ("ttU+JOhShl4", "sceMoveReadStateLatest", cb_query_with_output),
    ("f2bcpK6kJfg", "sceMoveReadStateRecent", cb_read_state_recent),
    ("IFQwtT2CeY0", "sceMoveSetVibration", cb_handle_no_controller),
    ("T8KYHPs1JE8", "sceMoveSetLightSphere", cb_handle_no_controller),
    ("zuxWAg3HAac", "sceMoveResetLightSphere", cb_reset_light_sphere),
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

    /// Reference GuestMove cases: lifecycle, one handle per controller
    /// slot, INVALID_HANDLE for a handle not open, no controller connected.
    #[test]
    fn no_controller_connected() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A72_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let call = |ctx: &mut CpuContext, cb: HleCallback, a: &[u64]| {
            for (r, v) in [Rdi, Rsi, Rdx, Rcx].into_iter().zip(a.iter().chain(std::iter::repeat(&0))) {
                ctx.set(r, *v);
            }
            cb(ctx)
        };
        let user = crate::user_service::PRIMARY_USER_ID as u64;
        assert_eq!(call(&mut ctx, cb_term, &[]), ERROR_NOT_INIT);
        assert_eq!(call(&mut ctx, cb_open, &[user, 0, 0]), ERROR_NOT_INIT);
        assert_eq!(call(&mut ctx, cb_init, &[]), OK);
        assert_eq!(call(&mut ctx, cb_init, &[]), ERROR_ALREADY_INIT);
        let h = call(&mut ctx, cb_open, &[user, 0, 0]);
        assert_eq!(h, FIRST_HANDLE as HleResult);
        assert_eq!(call(&mut ctx, cb_open, &[user, 0, 0]), ERROR_ALREADY_OPENED);
        let h2 = call(&mut ctx, cb_open, &[user, 0, 1]);
        assert_eq!(h2, (FIRST_HANDLE + HANDLE_STEP) as HleResult);
        assert_eq!(call(&mut ctx, cb_open, &[0x1234, 0, 2]), NOT_IMPLEMENTED);
        assert_eq!(call(&mut ctx, cb_query_with_output, &[h as u64, BASE]), NO_CONTROLLER_CONNECTED);
        assert_eq!(call(&mut ctx, cb_query_with_output, &[h as u64, 0]), ERROR_INVALID_ARG);
        assert_eq!(call(&mut ctx, cb_query_with_output, &[0x99, BASE]), ERROR_INVALID_HANDLE);
        assert_eq!(call(&mut ctx, cb_read_state_recent, &[h as u64, u64::MAX, BASE, BASE + 8]), ERROR_INVALID_ARG);
        assert_eq!(call(&mut ctx, cb_read_state_recent, &[h as u64, 0, BASE, BASE + 8]), NO_CONTROLLER_CONNECTED);
        assert_eq!(call(&mut ctx, cb_handle_no_controller, &[h as u64, 1]), NO_CONTROLLER_CONNECTED);
        assert_eq!(call(&mut ctx, cb_reset_light_sphere, &[h as u64]), OK);
        assert_eq!(call(&mut ctx, cb_close, &[h as u64]), OK);
        assert_eq!(call(&mut ctx, cb_close, &[h as u64]), ERROR_INVALID_HANDLE);
        assert_eq!(call(&mut ctx, cb_reset_light_sphere, &[h as u64]), ERROR_INVALID_HANDLE);
        assert_eq!(call(&mut ctx, cb_term, &[]), OK);
        assert_eq!(call(&mut ctx, cb_query_with_output, &[h2 as u64, BASE]), ERROR_NOT_INIT);
    }
}
