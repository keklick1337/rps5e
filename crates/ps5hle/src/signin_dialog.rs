// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceSigninDialog HLE: the PSN sign-in dialog.
//!
//! A system dialog (see `common_dialog::system_dialog`): Open shows it until
//! the host UI dismisses it. There is no PSN, so the result always reports
//! that the user canceled.

use crate::common_dialog::system_dialog::{self, Kind};
use crate::common_dialog::{STATUS_FINISHED, STATUS_INITIALIZED, STATUS_NONE, STATUS_RUNNING};
use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceSigninDialog";
const OK: HleResult = 0;

const ERR_NOT_INITIALIZED: HleResult = 0x8135_0001u32 as i32 as i64;
const ERR_ALREADY_INITIALIZED: HleResult = 0x8135_0002u32 as i32 as i64;
const ERR_PARAM_INVALID: HleResult = 0x8135_0003u32 as i32 as i64;
const ERR_INVALID_STATE: HleResult = 0x8135_0005u32 as i32 as i64;
const ERR_INVALID_USER_ID: HleResult = 0x8135_0007u32 as i32 as i64;

/// SceSigninDialogParam { i32 size; i32 userId; i32 reserved[2]; }.
const PARAM_SIZE: u32 = 16;
/// SceSigninDialogResult { i32 result; i32 reserved[3]; }.
const RESULT_SIZE: usize = 16;
/// SCE_SIGNIN_DIALOG_RESULT_USER_CANCELED.
const RESULT_USER_CANCELED: i32 = 1;

fn cb_initialize(_ctx: &mut CpuContext) -> HleResult {
    let mut sh = system_dialog::lock();
    if sh.signin.status != STATUS_NONE {
        return ERR_ALREADY_INITIALIZED;
    }
    sh.signin.status = STATUS_INITIALIZED;
    OK
}

fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    let sh = system_dialog::lock();
    if sh.signin.status == STATUS_NONE {
        return ERR_NOT_INITIALIZED;
    }
    system_dialog::set_status(sh, Kind::Signin, STATUS_NONE);
    OK
}

/// sceSigninDialogOpen(const SceSigninDialogParam*)
fn cb_open(ctx: &mut CpuContext) -> HleResult {
    let sh = system_dialog::lock();
    match sh.signin.status {
        STATUS_NONE => return ERR_NOT_INITIALIZED,
        STATUS_INITIALIZED | STATUS_FINISHED => {}
        _ => return ERR_INVALID_STATE,
    }
    let param = ctx.rdi();
    if param == 0 {
        return ERR_PARAM_INVALID;
    }
    let Some(raw) = ctx.read_bytes(param, PARAM_SIZE as usize) else {
        return ERR_PARAM_INVALID;
    };
    let word = |i: usize| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
    if word(0) != PARAM_SIZE || word(2) != 0 || word(3) != 0 {
        return ERR_PARAM_INVALID;
    }
    if word(1) as i32 != crate::user_service::PRIMARY_USER_ID {
        return ERR_INVALID_USER_ID;
    }
    log::info!("[SIGNINDIALOG] open: PSN sign-in is unavailable offline");
    system_dialog::set_status(sh, Kind::Signin, STATUS_RUNNING);
    OK
}

fn cb_status(_ctx: &mut CpuContext) -> HleResult {
    system_dialog::poll(Kind::Signin) as HleResult
}

fn cb_close(_ctx: &mut CpuContext) -> HleResult {
    let sh = system_dialog::lock();
    if sh.signin.status == STATUS_NONE {
        return ERR_NOT_INITIALIZED;
    }
    system_dialog::set_status(sh, Kind::Signin, STATUS_FINISHED);
    OK
}

fn cb_get_result(ctx: &mut CpuContext) -> HleResult {
    let sh = system_dialog::lock();
    match sh.signin.status {
        STATUS_NONE => return ERR_NOT_INITIALIZED,
        STATUS_FINISHED => {}
        _ => return ERR_INVALID_STATE,
    }
    let out = ctx.rdi();
    if out == 0 {
        return ERR_PARAM_INVALID;
    }
    let mut r = [0u8; RESULT_SIZE];
    r[..4].copy_from_slice(&RESULT_USER_CANCELED.to_le_bytes());
    if ctx.write_bytes(out, &r) { OK } else { ERR_PARAM_INVALID }
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("mlYGfmqE3fQ", "sceSigninDialogInitialize", cb_initialize),
    ("LXlmS6PvJdU", "sceSigninDialogTerminate", cb_terminate),
    ("JlpJVoRWv7U", "sceSigninDialogOpen", cb_open),
    ("M3OkENHcyiU", "sceSigninDialogClose", cb_close),
    ("Bw31liTFT3A", "sceSigninDialogUpdateStatus", cb_status),
    ("2m077aeC+PA", "sceSigninDialogGetStatus", cb_status),
    ("nqG7rqnYw1U", "sceSigninDialogGetResult", cb_get_result),
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

    fn noop() {}

    #[test]
    fn open_finishes_canceled() {
        let _g = crate::common_dialog::SYSTEM_DIALOG_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A65_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let mut param = [0u8; 16];
        param[..4].copy_from_slice(&16u32.to_le_bytes());
        param[4..8].copy_from_slice(&crate::user_service::PRIMARY_USER_ID.to_le_bytes());
        ctx.write_bytes(BASE, &param);
        ctx.set(Rdi, BASE);
        system_dialog::set_visibility_callback(None);
        assert_eq!(cb_open(&mut ctx), ERR_NOT_INITIALIZED);
        assert_eq!(cb_initialize(&mut ctx), OK);
        assert_eq!(cb_initialize(&mut ctx), ERR_ALREADY_INITIALIZED);
        assert_eq!(cb_open(&mut ctx), OK);
        assert_eq!(cb_open(&mut ctx), ERR_INVALID_STATE);
        ctx.set(Rdi, BASE + 0x100);
        assert_eq!(cb_get_result(&mut ctx), ERR_INVALID_STATE);
        // Headless host: running on the first poll, dismissed on the second.
        assert_eq!(cb_status(&mut ctx), STATUS_RUNNING as i64);
        assert_eq!(cb_status(&mut ctx), STATUS_FINISHED as i64);
        assert_eq!(cb_get_result(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE + 0x100), Some(RESULT_USER_CANCELED as u32));

        // With a host UI the dialog stays up until the host closes it.
        system_dialog::set_visibility_callback(Some(noop));
        ctx.set(Rdi, BASE);
        assert_eq!(cb_open(&mut ctx), OK);
        assert_eq!(cb_status(&mut ctx), STATUS_RUNNING as i64);
        assert_eq!(cb_status(&mut ctx), STATUS_RUNNING as i64);
        let snap = system_dialog::host_snapshot().unwrap();
        assert_eq!(snap.kind, Kind::Signin);
        assert!(system_dialog::visual_state().background);
        assert!(system_dialog::host_close(snap.generation));
        assert_eq!(cb_status(&mut ctx), STATUS_FINISHED as i64);
        system_dialog::set_visibility_callback(None);

        assert_eq!(cb_terminate(&mut ctx), OK);
        assert_eq!(cb_terminate(&mut ctx), ERR_NOT_INITIALIZED);
    }
}
