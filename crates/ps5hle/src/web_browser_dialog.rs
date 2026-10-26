// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceWebBrowserDialog: no browser exists, so an opened dialog is
//! finished by the first status poll, no page is ever loaded and the result
//! is "canceled by the user".

use crate::common_dialog::{
    DialogState, ERR_ARG_NULL, ERR_NOT_FINISHED, ERR_NOT_INITIALIZED, STATUS_FINISHED, STATUS_INITIALIZED, STATUS_NONE,
    STATUS_RUNNING,
};
use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceWebBrowserDialog";
const OK: HleResult = 0;
const ERR_ALREADY_INITIALIZED: HleResult = 0x80B8_0004u32 as i32 as i64;
const ERR_BUSY: HleResult = 0x80B8_0007u32 as i32 as i64;
/// SCE_COMMON_DIALOG_RESULT_USER_CANCELED.
const RESULT_USER_CANCELED: u32 = 1;

static WEB: DialogState = DialogState::new();

fn cb_initialize(_ctx: &mut CpuContext) -> HleResult {
    if WEB.transition(STATUS_NONE, STATUS_INITIALIZED) != STATUS_NONE {
        return ERR_ALREADY_INITIALIZED;
    }
    OK
}

/// Open / OpenForPredeterminedContent(param): a running dialog is BUSY.
fn cb_open(ctx: &mut CpuContext) -> HleResult {
    match WEB.get() {
        STATUS_NONE => return ERR_NOT_INITIALIZED,
        STATUS_RUNNING => return ERR_BUSY,
        _ => {}
    }
    if ctx.rdi() == 0 {
        return ERR_ARG_NULL;
    }
    WEB.set(STATUS_RUNNING);
    OK
}

fn cb_status(_ctx: &mut CpuContext) -> HleResult {
    WEB.poll() as HleResult
}

/// Close: a running dialog finishes; closing in any other initialized state
/// succeeds.
fn cb_close(_ctx: &mut CpuContext) -> HleResult {
    if WEB.get() == STATUS_NONE {
        return ERR_NOT_INITIALIZED;
    }
    WEB.set(STATUS_FINISHED);
    OK
}

/// GetResult(SceWebBrowserDialogResult*): the result field reads USER_CANCELED.
fn cb_get_result(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    let status = WEB.get();
    if status == STATUS_NONE {
        return ERR_NOT_INITIALIZED;
    }
    if out == 0 {
        return ERR_ARG_NULL;
    }
    if status != STATUS_FINISHED {
        return ERR_NOT_FINISHED;
    }
    if ctx.write_u32(out, RESULT_USER_CANCELED) { OK } else { ERR_ARG_NULL }
}

fn cb_ok(_ctx: &mut CpuContext) -> HleResult {
    OK
}

/// sceWebBrowserDialogResetCookie(param): no page is ever loaded, so the
/// cookie jar is always empty and there is nothing to discard.
fn cb_reset_cookie(_ctx: &mut CpuContext) -> HleResult {
    if WEB.get() == STATUS_NONE {
        return ERR_NOT_INITIALIZED;
    }
    OK
}

/// sceWebBrowserDialogSetCookie(param): the jar an empty dialog keeps —
/// the parameter is validated, the cookie itself is stored nowhere (no
/// page is ever loaded to receive it).
fn cb_set_cookie(ctx: &mut CpuContext) -> HleResult {
    if WEB.get() == STATUS_NONE {
        return ERR_NOT_INITIALIZED;
    }
    if ctx.rdi() == 0 {
        return ERR_ARG_NULL;
    }
    OK
}

/// Terminate: an initialized or finished dialog returns to NONE; a running
/// one is BUSY.
fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    match WEB.get() {
        STATUS_NONE => ERR_NOT_INITIALIZED,
        STATUS_RUNNING => ERR_BUSY,
        _ => {
            WEB.set(STATUS_NONE);
            OK
        }
    }
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("jqb7HntFQFc", "sceWebBrowserDialogInitialize", cb_initialize),
    ("FraP7debcdg", "sceWebBrowserDialogOpen", cb_open),
    ("O7dIZQrwVFY", "sceWebBrowserDialogOpenForPredeterminedContent", cb_open),
    ("CFTG6a8TjOU", "sceWebBrowserDialogGetStatus", cb_status),
    ("h1dR-t5ISgg", "sceWebBrowserDialogUpdateStatus", cb_status),
    ("PSK+Eik919Q", "sceWebBrowserDialogClose", cb_close),
    ("vCaW0fgVQmc", "sceWebBrowserDialogGetResult", cb_get_result),
    ("Wit4LjeoeX4", "sceWebBrowserDialogGetEvent", cb_ok),
    ("uYELOMVnmNQ", "sceWebBrowserDialogNavigate", cb_ok),
    ("RLhKBOoNyXY", "sceWebBrowserDialogSetZoom", cb_ok),
    ("ocHtyBwHfys", "sceWebBrowserDialogTerminate", cb_terminate),
    ("Cya+jvTtPqg", "sceWebBrowserDialogResetCookie", cb_reset_cookie),
    ("TZnDVkP91Rg", "sceWebBrowserDialogSetCookie", cb_set_cookie),
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

    /// The whole state machine (one test: the dialog state is global).
    #[test]
    fn dialog_states() {
        const BASE: u64 = 0x5D44_0300_0000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        WEB.set(STATUS_NONE);
        assert_eq!(cb_terminate(&mut ctx), ERR_NOT_INITIALIZED);
        assert_eq!(cb_close(&mut ctx), ERR_NOT_INITIALIZED);
        ctx.set(crate::context::CpuRegister::Rdi, BASE);
        assert_eq!(cb_get_result(&mut ctx), ERR_NOT_INITIALIZED);
        assert_eq!(cb_open(&mut ctx), ERR_NOT_INITIALIZED);
        assert_eq!(cb_status(&mut ctx), STATUS_NONE as i64);
        assert_eq!(cb_initialize(&mut ctx), OK);
        assert_eq!(cb_initialize(&mut ctx), ERR_ALREADY_INITIALIZED);
        assert_eq!(cb_get_result(&mut ctx), ERR_NOT_FINISHED);
        assert_eq!(cb_open(&mut ctx), OK);
        assert_eq!(cb_open(&mut ctx), ERR_BUSY);
        assert_eq!(cb_terminate(&mut ctx), ERR_BUSY);
        assert_eq!(cb_status(&mut ctx), STATUS_FINISHED as i64);
        ctx.write_u32(BASE, 0x55);
        assert_eq!(cb_get_result(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE), Some(RESULT_USER_CANCELED));
        ctx.set(crate::context::CpuRegister::Rdi, 0);
        assert_eq!(cb_get_result(&mut ctx), ERR_ARG_NULL);
        assert_eq!(cb_open(&mut ctx), ERR_ARG_NULL);
        assert_eq!(cb_close(&mut ctx), OK);
        assert_eq!(cb_terminate(&mut ctx), OK);
        assert_eq!(cb_status(&mut ctx), STATUS_NONE as i64);
    }
}
