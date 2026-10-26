// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceNpSessionSignaling — peer signaling, offline.
//!
//! Offline: no PSN. Initialization and context management succeed, but
//! nothing can be activated: preparing a request or activating a session
//! reports the user as signed out, no connection ever exists, and local
//! network information is unavailable.

use std::collections::BTreeSet;
use std::sync::Mutex;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceNpSessionSignaling";

const fn err(v: u32) -> HleResult {
    v as i32 as i64
}

const ERR_NP_SIGNED_OUT: HleResult = err(0x8055_0006);
const ERR_UNAVAILABLE: HleResult = err(0x8055_2D06);
const ERR_INVALID_ARGUMENT: HleResult = err(0x8055_3303);
const ERR_CONN_NOT_FOUND: HleResult = err(0x8055_330C);

/// Connection status reported for connections that do not exist.

struct SignalingState {
    next_context: i32,
    contexts: BTreeSet<i32>,
}

static STATE: Mutex<SignalingState> = Mutex::new(SignalingState { next_context: 0, contexts: BTreeSet::new() });

fn state() -> std::sync::MutexGuard<'static, SignalingState> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn valid_context(id: i32) -> bool {
    state().contexts.contains(&id)
}

fn cb_initialize(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// sceNpSessionSignalingTerminate(): drops every context.
fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    state().contexts.clear();
    0
}

/// Returns a fresh positive context id.
fn cb_create_context2(_ctx: &mut CpuContext) -> HleResult {
    let mut s = state();
    s.next_context = s.next_context.wrapping_add(1).max(1);
    let id = s.next_context;
    s.contexts.insert(id);
    id as i64
}

fn cb_request_prepare(_ctx: &mut CpuContext) -> HleResult {
    ERR_NP_SIGNED_OUT
}

fn cb_destroy_context(ctx: &mut CpuContext) -> HleResult {
    state().contexts.remove(&(ctx.rdi() as i32));
    0
}

/// sceNpSessionSignalingActivateSession(ctxId, ...): needs PSN.
fn cb_activate_session(ctx: &mut CpuContext) -> HleResult {
    if !valid_context(ctx.rdi() as i32) {
        return ERR_INVALID_ARGUMENT;
    }
    ERR_NP_SIGNED_OUT
}

/// sceNpSessionSignalingDeactivate(ctxId, ...): nothing is ever active.
fn cb_deactivate(ctx: &mut CpuContext) -> HleResult {
    if !valid_context(ctx.rdi() as i32) {
        return ERR_INVALID_ARGUMENT;
    }
    0
}

/// sceNpSessionSignalingGetConnectionStatus(ctxId, connId, int* status,
/// SceNetInAddr* peerAddr, SceNetInPort_t* peerPort): no connection exists,
/// so the connection is not found and nothing is written.
fn cb_get_connection_status(ctx: &mut CpuContext) -> HleResult {
    if !valid_context(ctx.rdi() as i32) || ctx.rdx() == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    ERR_CONN_NOT_FOUND
}

/// sceNpSessionSignalingGetConnectionInfo(ctxId, connId, code, info*):
/// no connection exists.
fn cb_get_connection_info(ctx: &mut CpuContext) -> HleResult {
    if !valid_context(ctx.rdi() as i32) || ctx.rcx() == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    ERR_UNAVAILABLE
}

/// sceNpSessionSignalingGetLocalNetInfo(ctxId, info*): offline, no local
/// network information.
fn cb_get_local_net_info(ctx: &mut CpuContext) -> HleResult {
    if !valid_context(ctx.rdi() as i32) || ctx.rsi() == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    ERR_UNAVAILABLE
}

/// ActivateUser / GetConnectionFromPeerAddress2: there is no network
/// session to join or peer to look up.
fn cb_unavailable(_ctx: &mut CpuContext) -> HleResult {
    ERR_UNAVAILABLE
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("ysmw6J-P8Ak", "sceNpSessionSignalingInitialize", cb_initialize),
    ("CqJuNXo5yiM", "sceNpSessionSignalingTerminate", cb_terminate),
    ("aBuX0PX-T7I", "sceNpSessionSignalingCreateContext2", cb_create_context2),
    ("GtuZGmN-tKw", "sceNpSessionSignalingCreateContext", cb_create_context2),
    ("9r7dM3puxMk", "sceNpSessionSignalingActivateUser", cb_unavailable),
    ("DicBhsm2iR4", "sceNpSessionSignalingGetConnectionFromPeerAddress2", cb_unavailable),
    ("r8mVMwlafF8", "sceNpSessionSignalingRequestPrepare", cb_request_prepare),
    ("Z9Q9LzQDXf0", "sceNpSessionSignalingDestroyContext", cb_destroy_context),
    ("r4XacqHvkn4", "sceNpSessionSignalingActivateSession", cb_activate_session),
    ("cQkBH-pXhF0", "sceNpSessionSignalingDeactivate", cb_deactivate),
    ("n1fn2KFeLDA", "sceNpSessionSignalingGetConnectionStatus", cb_get_connection_status),
    ("yJw2m6UWDYU", "sceNpSessionSignalingGetConnectionInfo", cb_get_connection_info),
    ("OTilStjd9L8", "sceNpSessionSignalingGetLocalNetInfo", cb_get_local_net_info),
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

    #[test]
    fn offline_signaling() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C0F_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let id = cb_create_context2(&mut ctx);
        assert!(id > 0);
        ctx.set(Rdi, id as u64);
        assert_eq!(cb_activate_session(&mut ctx), ERR_NP_SIGNED_OUT);
        assert_eq!(cb_deactivate(&mut ctx), 0);
        ctx.write_u32(BASE, 7);
        ctx.set(Rsi, 1);
        ctx.set(Rdx, BASE);
        ctx.set(Rcx, 0);
        ctx.set(R8, 0);
        assert_eq!(cb_get_connection_status(&mut ctx), ERR_CONN_NOT_FOUND);
        assert_eq!(ctx.read_u32(BASE), Some(7), "nothing is written");
        ctx.set(Rdx, 0);
        assert_eq!(cb_get_connection_status(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.set(Rsi, BASE);
        assert_eq!(cb_get_local_net_info(&mut ctx), ERR_UNAVAILABLE);
        assert_eq!(cb_destroy_context(&mut ctx), 0);
        assert_eq!(cb_activate_session(&mut ctx), ERR_INVALID_ARGUMENT);
    }

    /// CreateContext makes a context like CreateContext2; activating a user
    /// or finding a peer's connection is unavailable offline.
    #[test]
    fn create_context_and_offline_lookups() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        let create = EXPORTS.iter().find(|e| e.1 == "sceNpSessionSignalingCreateContext").unwrap().2;
        let id = create(&mut ctx);
        assert!(id > 0 && valid_context(id as i32));
        assert_eq!(cb_unavailable(&mut ctx), ERR_UNAVAILABLE);
        ctx.set(Rdi, id as u64);
        assert_eq!(cb_destroy_context(&mut ctx), 0);
    }
}
