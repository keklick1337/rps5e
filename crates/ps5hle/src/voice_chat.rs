// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceVoiceChat — PSN party/session voice chat.
//!
//! Voice chat needs a signed-in PSN account and the voice chat service;
//! the emulator has neither. Library initialisation, handler registration
//! and event processing succeed, while every request (and request object
//! creation) fails with the signed-out error.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceVoiceChat";

const ERROR_SIGNED_OUT: HleResult = 0x8055_0006u32 as i32 as i64;

static INITIALIZED: AtomicBool = AtomicBool::new(false);
/// Registered handler table pointer and microphone event handler (handler, arg).
static HANDLERS: Mutex<(u64, (u64, u64))> = Mutex::new((0, (0, 0)));

fn handlers() -> std::sync::MutexGuard<'static, (u64, (u64, u64))> {
    HANDLERS.lock().unwrap_or_else(|p| p.into_inner())
}

/// sceVoiceChatInitialize(const SceVoiceChatInitParam*)
fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    log::info!("[voicechat] initialize param={:#x} (offline: requests report signed-out)", ctx.rdi());
    INITIALIZED.store(true, Ordering::Relaxed);
    0
}

/// sceVoiceChatTerminate()
fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    INITIALIZED.store(false, Ordering::Relaxed);
    *handlers() = (0, (0, 0));
    0
}

/// sceVoiceChatRegisterHandlers(const SceVoiceChatHandlers*)
fn cb_register_handlers(ctx: &mut CpuContext) -> HleResult {
    handlers().0 = ctx.rdi();
    0
}

/// sceVoiceChatRegisterMicEventHandler(handler, arg)
fn cb_register_mic_event_handler(ctx: &mut CpuContext) -> HleResult {
    handlers().1 = (ctx.rdi(), ctx.rsi());
    0
}

/// sceVoiceChatProcessEvent(): no service, so never any event to deliver.
fn cb_process_event(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// sceVoiceChatCreateRequest(param, int* requestId)
fn cb_create_request(_ctx: &mut CpuContext) -> HleResult {
    log::debug!("[voicechat] create request: signed out");
    ERROR_SIGNED_OUT
}

/// sceVoiceChatDeleteRequest(int requestId)
fn cb_delete_request(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// Every sceVoiceChatRequest*(requestId, param, response) call.
fn cb_request_offline(_ctx: &mut CpuContext) -> HleResult {
    log::debug!("[voicechat] request: signed out");
    ERROR_SIGNED_OUT
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("hR8-CKMl2JQ", "sceVoiceChatCreateRequest", cb_create_request),
    ("eEMpsX1fGHU", "sceVoiceChatDeleteRequest", cb_delete_request),
    ("spdS-hedavE", "sceVoiceChatInitialize", cb_initialize),
    ("CscDZAFA5+c", "sceVoiceChatProcessEvent", cb_process_event),
    ("Ptmkf9UnWBg", "sceVoiceChatRegisterHandlers", cb_register_handlers),
    ("ajXKK3BOVc8", "sceVoiceChatRegisterMicEventHandler", cb_register_mic_event_handler),
    ("y7MgGX889Mo", "sceVoiceChatRequestCreateGameSessionVoiceChatChannel", cb_request_offline),
    ("sW3km27c12M", "sceVoiceChatRequestCreatePlayerSessionVoiceChatChannel", cb_request_offline),
    ("7bu++dneYUU", "sceVoiceChatRequestDeleteGameSessionVoiceChatChannel", cb_request_offline),
    ("zbKF-ejbR0Q", "sceVoiceChatRequestDeletePlayerSessionVoiceChatChannel", cb_request_offline),
    ("hpG+mR4EbpE", "sceVoiceChatRequestJoinGameSessionVoiceChatChannel", cb_request_offline),
    ("X3BWlTuErbk", "sceVoiceChatRequestJoinPlayerSessionVoiceChatChannel", cb_request_offline),
    ("S+mOdmysfhw", "sceVoiceChatRequestLeaveGameSessionVoiceChatChannel", cb_request_offline),
    ("0rKMjtnMATU", "sceVoiceChatRequestLeavePlayerSessionVoiceChatChannel", cb_request_offline),
    ("po9eImAEhY4", "sceVoiceChatRequestRegisterSession", cb_request_offline),
    ("q75YGaxUT3s", "sceVoiceChatRequestUnregisterSession", cb_request_offline),
    ("8jiB8eq+BkM", "sceVoiceChatRequestCreateVoiceChatGroup", cb_request_offline),
    ("aT9KhNAu0RY", "sceVoiceChatRequestJoinVoiceChatGroup", cb_request_offline),
    ("uWNeRp70BRw", "sceVoiceChatRequestLeaveVoiceChatGroup", cb_request_offline),
    ("ZpO7G7+nd2E", "sceVoiceChatRequestDeleteVoiceChatGroup", cb_request_offline),
    ("j43P62s5SX4", "sceVoiceChatTerminate", cb_terminate),
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
    fn requests_report_signed_out() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        assert_eq!(cb_initialize(&mut ctx), 0);
        assert_eq!(cb_create_request(&mut ctx), ERROR_SIGNED_OUT);
        assert_eq!(cb_request_offline(&mut ctx), ERROR_SIGNED_OUT);
        // Every request export is an offline request.
        for &(_, name, cb) in EXPORTS.iter().filter(|e| e.1.starts_with("sceVoiceChatRequest")) {
            assert_eq!(cb(&mut ctx), ERROR_SIGNED_OUT, "{name}");
        }
        assert!(EXPORTS.iter().any(|e| e.1 == "sceVoiceChatRequestLeaveVoiceChatGroup"));
        assert!(EXPORTS.iter().any(|e| e.1 == "sceVoiceChatRequestDeleteVoiceChatGroup"));
        assert_eq!(cb_process_event(&mut ctx), 0);
        assert_eq!(cb_terminate(&mut ctx), 0);
    }
}
