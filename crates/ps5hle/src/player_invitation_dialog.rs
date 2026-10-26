// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libScePlayerInvitationDialog HLE.
//!
//! Inviting players needs PSN, which is never available: an opened dialog is
//! finished by the first status poll and reports that the user canceled.

use crate::common_dialog::{DialogState, STATUS_FINISHED, STATUS_NONE, STATUS_RUNNING};
use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libScePlayerInvitationDialog";
const OK: HleResult = 0;

// SCE_COMMON_DIALOG_ERROR_*.
const ERR_NOT_INITIALIZED: HleResult = 0x80B8_0003u32 as i32 as i64;
const ERR_NOT_FINISHED: HleResult = 0x80B8_0005u32 as i32 as i64;
const ERR_BUSY: HleResult = 0x80B8_0007u32 as i32 as i64;
const ERR_ARG_NULL: HleResult = 0x80B8_000Du32 as i32 as i64;
const ERR_MEMORY_FAULT: HleResult = 0x8002_000Eu32 as i32 as i64;

/// ScePlayerInvitationDialogResult { i32 errorCode; i32 result; u8 reserved[32]; }.
const RESULT_SIZE: usize = 40;
/// SCE_COMMON_DIALOG_RESULT_USER_CANCELED.
const RESULT_USER_CANCELED: i32 = 1;

static DIALOG: DialogState = DialogState::new();

fn cb_initialize(_ctx: &mut CpuContext) -> HleResult {
    DIALOG.initialize()
}

fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    DIALOG.terminate()
}

fn cb_open(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return ERR_ARG_NULL;
    }
    match DIALOG.get() {
        STATUS_NONE => return ERR_NOT_INITIALIZED,
        STATUS_RUNNING => return ERR_BUSY,
        _ => {}
    }
    log::info!("[PLAYERINVITATIONDIALOG] open: invitations need PSN (offline)");
    DIALOG.set(STATUS_RUNNING);
    OK
}

fn cb_update_status(_ctx: &mut CpuContext) -> HleResult {
    DIALOG.poll() as HleResult
}

/// scePlayerInvitationDialogGetStatus(): the status without polling.
fn cb_get_status(_ctx: &mut CpuContext) -> HleResult {
    DIALOG.get() as HleResult
}

fn cb_close(_ctx: &mut CpuContext) -> HleResult {
    DIALOG.close()
}

fn cb_get_result(ctx: &mut CpuContext) -> HleResult {
    if DIALOG.get() == STATUS_NONE {
        return ERR_NOT_INITIALIZED;
    }
    let out = ctx.rdi();
    if out == 0 {
        return ERR_ARG_NULL;
    }
    if DIALOG.get() != STATUS_FINISHED {
        return ERR_NOT_FINISHED;
    }
    let mut r = [0u8; RESULT_SIZE];
    r[4..8].copy_from_slice(&RESULT_USER_CANCELED.to_le_bytes());
    if ctx.write_bytes(out, &r) { OK } else { ERR_MEMORY_FAULT }
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("JDwx3Bl4bB4", "scePlayerInvitationDialogInitialize", cb_initialize),
    ("gDm5a6GSE94", "scePlayerInvitationDialogTerminate", cb_terminate),
    ("rKPTlHwGa4k", "scePlayerInvitationDialogOpen", cb_open),
    ("kFhuwHrIUqs", "scePlayerInvitationDialogUpdateStatus", cb_update_status),
    ("1m4CKJxzsW0", "scePlayerInvitationDialogGetStatus", cb_get_status),
    ("Zo52g0A1XDw", "scePlayerInvitationDialogClose", cb_close),
    ("AhqlQ8cngrk", "scePlayerInvitationDialogGetResult", cb_get_result),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
    crate::offline_dialogs::register(reg);
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
    fn lifecycle() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C0A_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_open(&mut ctx), ERR_NOT_INITIALIZED);
        assert_eq!(cb_get_status(&mut ctx), STATUS_NONE as i64);
        assert_eq!(cb_initialize(&mut ctx), OK);
        assert_eq!(cb_open(&mut ctx), OK);
        assert_eq!(cb_get_status(&mut ctx), STATUS_RUNNING as i64, "GetStatus does not poll");
        assert_eq!(cb_open(&mut ctx), ERR_BUSY);
        assert_eq!(cb_get_result(&mut ctx), ERR_NOT_FINISHED);
        assert_eq!(cb_update_status(&mut ctx), STATUS_FINISHED as i64);
        assert_eq!(cb_get_status(&mut ctx), STATUS_FINISHED as i64);
        assert_eq!(cb_get_result(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE + 4), Some(RESULT_USER_CANCELED as u32));
        assert_eq!(cb_terminate(&mut ctx), OK);
    }
}
