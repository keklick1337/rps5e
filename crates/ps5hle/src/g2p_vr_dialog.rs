// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceCdlgG2P (game-to-PlayStation dialog) and libSceVrSetupDialog HLE.
//!
//! Both are common dialogs with no UI here: Open finishes the dialog at
//! once and the result reports that the user canceled. Each library keeps
//! one dialog status; Initialize moves it out of NONE only once, Terminate
//! returns it to NONE.

use crate::common_dialog::{DialogState, STATUS_FINISHED, STATUS_INITIALIZED, STATUS_NONE, STATUS_RUNNING};
use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB_G2P: &str = "libSceCdlgG2P";
const LIB_VR: &str = "libSceVrSetupDialog";
const OK: HleResult = 0;

// SCE_COMMON_DIALOG_ERROR_*.
const ERR_NOT_INITIALIZED: HleResult = 0x80B8_0003u32 as i32 as i64;
const ERR_ALREADY_INITIALIZED: HleResult = 0x80B8_0004u32 as i32 as i64;
const ERR_NOT_FINISHED: HleResult = 0x80B8_0005u32 as i32 as i64;
const ERR_BUSY: HleResult = 0x80B8_0007u32 as i32 as i64;
const ERR_ARG_NULL: HleResult = 0x80B8_000Du32 as i32 as i64;
const ERR_MEMORY_FAULT: HleResult = 0x8002_000Eu32 as i32 as i64;

/// SCE_COMMON_DIALOG_RESULT_USER_CANCELED, the first i32 of the result.
const RESULT_USER_CANCELED: i32 = 1;

static G2P: DialogState = DialogState::new();
static VR: DialogState = DialogState::new();

fn initialize(d: &DialogState) -> HleResult {
    if d.transition(STATUS_NONE, STATUS_INITIALIZED) == STATUS_NONE { OK } else { ERR_ALREADY_INITIALIZED }
}

/// Open(const param*): the dialog finishes at once.
fn open(d: &DialogState, param: u64) -> HleResult {
    match d.get() {
        STATUS_NONE => return ERR_NOT_INITIALIZED,
        STATUS_RUNNING => return ERR_BUSY,
        _ => {}
    }
    if param == 0 {
        return ERR_ARG_NULL;
    }
    d.set(STATUS_FINISHED);
    OK
}

/// GetResult(result*): only the result code (user canceled) is written.
fn get_result(ctx: &CpuContext, d: &DialogState, out: u64) -> HleResult {
    let status = d.get();
    if status == STATUS_NONE {
        return ERR_NOT_INITIALIZED;
    }
    if out == 0 {
        return ERR_ARG_NULL;
    }
    if status != STATUS_FINISHED {
        return ERR_NOT_FINISHED;
    }
    if ctx.write_u32(out, RESULT_USER_CANCELED as u32) { OK } else { ERR_MEMORY_FAULT }
}

/// Close ends the dialog: FINISHED from any state but NONE.
fn close(d: &DialogState) -> HleResult {
    if d.get() == STATUS_NONE {
        return ERR_NOT_INITIALIZED;
    }
    d.set(STATUS_FINISHED);
    OK
}

fn terminate(d: &DialogState) -> HleResult {
    if d.set(STATUS_NONE) == STATUS_NONE { ERR_NOT_INITIALIZED } else { OK }
}

fn cb_g2p_initialize(_ctx: &mut CpuContext) -> HleResult {
    initialize(&G2P)
}

fn cb_g2p_open(ctx: &mut CpuContext) -> HleResult {
    if G2P.get() != STATUS_NONE && ctx.rdi() != 0 {
        log::info!("[G2PDIALOG] open: no dialog UI, finished as canceled");
    }
    open(&G2P, ctx.rdi())
}

fn cb_g2p_status(_ctx: &mut CpuContext) -> HleResult {
    G2P.get() as HleResult
}

fn cb_g2p_get_result(ctx: &mut CpuContext) -> HleResult {
    get_result(ctx, &G2P, ctx.rdi())
}

fn cb_g2p_close(_ctx: &mut CpuContext) -> HleResult {
    close(&G2P)
}

fn cb_g2p_terminate(_ctx: &mut CpuContext) -> HleResult {
    terminate(&G2P)
}

fn cb_vr_initialize(_ctx: &mut CpuContext) -> HleResult {
    initialize(&VR)
}

fn cb_vr_open(ctx: &mut CpuContext) -> HleResult {
    if VR.get() != STATUS_NONE && ctx.rdi() != 0 {
        log::info!("[VRSETUPDIALOG] open: no dialog UI, finished as canceled");
    }
    open(&VR, ctx.rdi())
}

fn cb_vr_update_status(_ctx: &mut CpuContext) -> HleResult {
    VR.get() as HleResult
}

fn cb_vr_get_result(ctx: &mut CpuContext) -> HleResult {
    get_result(ctx, &VR, ctx.rdi())
}

fn cb_vr_close(_ctx: &mut CpuContext) -> HleResult {
    close(&VR)
}

fn cb_vr_terminate(_ctx: &mut CpuContext) -> HleResult {
    terminate(&VR)
}

const G2P_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("SWRH9B4oil8", "sceG2PDialogInitialize", cb_g2p_initialize),
    ("biwS5HG-DBY", "sceG2PDialogOpen", cb_g2p_open),
    ("tkRwnzG4Tus", "sceG2PDialogUpdateStatus", cb_g2p_status),
    ("Wi6pRLPS8cc", "sceG2PDialogGetStatus", cb_g2p_status),
    ("vpA-xXvhfeY", "sceG2PDialogGetResult", cb_g2p_get_result),
    ("2pEUrnXGxys", "sceG2PDialogClose", cb_g2p_close),
    ("jdhjw0pRmyQ", "sceG2PDialogTerminate", cb_g2p_terminate),
];

const VR_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("sTFqbfXJGG4", "sceVrSetupDialogInitialize", cb_vr_initialize),
    ("loE6cNuCUtk", "sceVrSetupDialogOpen", cb_vr_open),
    ("KHfpjj9Pucg", "sceVrSetupDialogUpdateStatus", cb_vr_update_status),
    ("qRUACYyUNFA", "sceVrSetupDialogGetStatus", cb_vr_update_status),
    ("-umGRHNUiZU", "sceVrSetupDialogGetResult", cb_vr_get_result),
    ("ZeYpKhsLfG0", "sceVrSetupDialogClose", cb_vr_close),
    ("zdgiIPRqET4", "sceVrSetupDialogTerminate", cb_vr_terminate),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in G2P_EXPORTS {
        reg.register_fn(nid, name, LIB_G2P, cb);
    }
    for &(nid, name, cb) in VR_EXPORTS {
        reg.register_fn(nid, name, LIB_VR, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    #[test]
    fn nids_match_names() {
        // Close finishes an initialized dialog too.
        let d = DialogState::new();
        assert_eq!(close(&d), ERR_NOT_INITIALIZED);
        d.set(STATUS_INITIALIZED);
        assert_eq!(close(&d), OK);
        assert_eq!(d.get(), STATUS_FINISHED);
        for &(nid, name, _) in G2P_EXPORTS.iter().chain(VR_EXPORTS) {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// The lifecycle both dialogs share: errors before Initialize, a
    /// second Initialize refused, Open finishing at once, a canceled
    /// result that only fills the result code, and Terminate once.
    fn lifecycle(
        ctx: &mut CpuContext,
        base: u64,
        [init, open, status, get_result, close, terminate]: [HleCallback; 6],
    ) {
        let (param, result) = (base, base + 0x10);
        ctx.write_u32(result, u32::MAX);
        ctx.write_u32(result + 4, u32::MAX);
        let call = |ctx: &mut CpuContext, cb: HleCallback, arg: u64| {
            ctx.set(Rdi, arg);
            cb(ctx)
        };
        assert_eq!(call(ctx, terminate, 0), ERR_NOT_INITIALIZED);
        assert_eq!(call(ctx, open, param), ERR_NOT_INITIALIZED);
        assert_eq!(call(ctx, get_result, result), ERR_NOT_INITIALIZED);
        assert_eq!(call(ctx, init, 0), OK);
        assert_eq!(call(ctx, init, 0), ERR_ALREADY_INITIALIZED);
        assert_eq!(call(ctx, open, 0), ERR_ARG_NULL);
        assert_eq!(call(ctx, get_result, result), ERR_NOT_FINISHED);
        assert_eq!(call(ctx, open, param), OK);
        assert_eq!(call(ctx, status, 0), STATUS_FINISHED as HleResult);
        assert_eq!(call(ctx, get_result, 0), ERR_ARG_NULL);
        assert_eq!(call(ctx, get_result, result), OK);
        assert_eq!(ctx.read_u32(result), Some(RESULT_USER_CANCELED as u32));
        assert_eq!(ctx.read_u32(result + 4), Some(u32::MAX));
        assert_eq!(call(ctx, close, 0), OK);
        assert_eq!(call(ctx, terminate, 0), OK);
        assert_eq!(call(ctx, terminate, 0), ERR_NOT_INITIALIZED);
    }

    #[test]
    fn g2p_dialog_finishes_canceled() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x6A19_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        lifecycle(
            &mut ctx,
            BASE,
            [cb_g2p_initialize, cb_g2p_open, cb_g2p_status, cb_g2p_get_result, cb_g2p_close, cb_g2p_terminate],
        );
        ctx.set(Rdi, 0);
        assert_eq!(cb_g2p_status(&mut ctx), STATUS_NONE as HleResult);
    }

    #[test]
    fn vr_setup_dialog_finishes_canceled() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x6A1A_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        lifecycle(
            &mut ctx,
            BASE,
            [cb_vr_initialize, cb_vr_open, cb_vr_update_status, cb_vr_get_result, cb_vr_close, cb_vr_terminate],
        );
    }
}
