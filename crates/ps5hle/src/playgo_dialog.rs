// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libScePlayGoDialog HLE: the install dialog of PlayGo titles.
//!
//! A title drives the dialog as a state machine: Initialize, Open, polls of
//! UpdateStatus until it reports FINISHED (3), GetResult, Close and
//! Terminate. Nothing is shown here: Open completes the dialog at once, and
//! the result is OK. This emulator reports no pending PlayGo chunks
//! (scePlayGoGetToDoList), so a title that checks for them skips the
//! dialog; a title that starts it unconditionally gets the completed
//! dialog, which is the result a fully installed copy should give (not
//! measured on a console).
//!
//! Error codes and the state transitions follow the reference model. The
//! console's behaviour in the states this model
//! never enters (the dialog running in the background) is unknown.

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};
use std::sync::{Mutex, MutexGuard};

const LIB: &str = "libScePlayGoDialog";
const OK: HleResult = 0;

const ERR_NOT_INITIALIZED: HleResult = 0x80ED_0001u32 as i32 as i64;
const ERR_ALREADY_INITIALIZED: HleResult = 0x80ED_0002u32 as i32 as i64;
const ERR_PARAM_INVALID: HleResult = 0x80ED_0003u32 as i32 as i64;
const ERR_INVALID_STATE: HleResult = 0x80ED_0005u32 as i32 as i64;

/// Status values the title polls. FINISHED is what its loop waits for.
const STATUS_NONE: i32 = 0;
const STATUS_INITIALIZED: i32 = 1;
const STATUS_FINISHED: i32 = 3;

/// PlayGoDialogResult: `int32 result; int32 reserved[9];`. The title's
/// buffer is larger; only this prefix is written.
const RESULT_SIZE: usize = 40;
const RESULT_OK: i32 = 0;

/// The dialog state. Each process has one dialog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Dialog {
    status: i32,
}

impl Dialog {
    const fn new() -> Self {
        Self { status: STATUS_NONE }
    }

    /// scePlayGoDialogInitialize: only from no state.
    fn initialize(&mut self) -> HleResult {
        if self.status != STATUS_NONE {
            return ERR_ALREADY_INITIALIZED;
        }
        self.status = STATUS_INITIALIZED;
        OK
    }

    /// scePlayGoDialogOpen: the dialog runs to completion here. The
    /// parameter is checked for null only.
    fn open(&mut self, param: u64) -> HleResult {
        if param == 0 {
            return ERR_PARAM_INVALID;
        }
        match self.status {
            STATUS_INITIALIZED | STATUS_FINISHED => {
                self.status = STATUS_FINISHED;
                OK
            }
            _ => ERR_INVALID_STATE,
        }
    }

    /// scePlayGoDialogUpdateStatus: the current status; no error code.
    fn update_status(&self) -> HleResult {
        self.status as HleResult
    }

    /// scePlayGoDialogClose: from any initialized state, the dialog ends.
    fn close(&mut self) -> HleResult {
        if self.status == STATUS_NONE {
            return ERR_NOT_INITIALIZED;
        }
        self.status = STATUS_FINISHED;
        OK
    }

    /// Checks for scePlayGoDialogGetResult. Returns the error to report, or
    /// None when the result block may be written.
    fn result_error(&self, out: u64) -> Option<HleResult> {
        if self.status == STATUS_NONE {
            return Some(ERR_NOT_INITIALIZED);
        }
        if out == 0 {
            return Some(ERR_PARAM_INVALID);
        }
        if self.status != STATUS_FINISHED {
            return Some(crate::common_dialog::ERR_NOT_FINISHED);
        }
        None
    }

    /// scePlayGoDialogTerminate: from an initialized or finished state.
    fn terminate(&mut self) -> HleResult {
        match self.status {
            STATUS_INITIALIZED | STATUS_FINISHED => {
                self.status = STATUS_NONE;
                OK
            }
            _ => ERR_NOT_INITIALIZED,
        }
    }
}

static DIALOG: Mutex<Dialog> = Mutex::new(Dialog::new());

fn dialog() -> MutexGuard<'static, Dialog> {
    DIALOG.lock().unwrap_or_else(|e| e.into_inner())
}

/// Write the result block (result = OK, the rest zero) to guest memory.
fn write_result(ctx: &mut CpuContext, out: u64) -> HleResult {
    let mut block = [0u8; RESULT_SIZE];
    block[0..4].copy_from_slice(&RESULT_OK.to_le_bytes());
    if ctx.write_bytes(out, &block) { OK } else { ERR_PARAM_INVALID }
}

/// scePlayGoDialogInitialize(void)
fn cb_initialize(_ctx: &mut CpuContext) -> HleResult {
    dialog().initialize()
}

/// scePlayGoDialogOpen(const void* param)
fn cb_open(ctx: &mut CpuContext) -> HleResult {
    dialog().open(ctx.rdi())
}

/// scePlayGoDialogUpdateStatus(void)
fn cb_update_status(_ctx: &mut CpuContext) -> HleResult {
    dialog().update_status()
}

/// scePlayGoDialogGetStatus(void): the same status UpdateStatus reports.
fn cb_get_status(_ctx: &mut CpuContext) -> HleResult {
    dialog().update_status()
}

/// scePlayGoDialogGetResult(PlayGoDialogResult* result)
fn cb_get_result(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if let Some(err) = dialog().result_error(out) {
        return err;
    }
    write_result(ctx, out)
}

/// scePlayGoDialogClose(void)
fn cb_close(_ctx: &mut CpuContext) -> HleResult {
    dialog().close()
}

/// scePlayGoDialogTerminate(void)
fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    dialog().terminate()
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("fECamTJKpsM", "scePlayGoDialogInitialize", cb_initialize),
    ("kHd72ukqbxw", "scePlayGoDialogOpen", cb_open),
    ("Yb60K7BST48", "scePlayGoDialogUpdateStatus", cb_update_status),
    ("NOAMxY2EGS0", "scePlayGoDialogGetStatus", cb_get_status),
    ("wx9TDplJKB4", "scePlayGoDialogGetResult", cb_get_result),
    ("fbigNQiZpm0", "scePlayGoDialogClose", cb_close),
    ("okgIGdr5Iz0", "scePlayGoDialogTerminate", cb_terminate),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps5vm::{MemProtect, VirtualMemory};

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn all_are_implementations() {
        let reg = HleRegistry::with_defaults();
        for &(nid, name, _) in EXPORTS {
            assert!(reg.is_implemented(nid), "{name} is still the catalog stub");
        }
    }

    /// The title's order in its install loop, with the status values it
    /// checks: nothing is accepted before Initialize, Open must come before
    /// the poll loop can see 3, and Terminate ends the run.
    #[test]
    fn title_sequence() {
        let mut d = Dialog::new();
        let param = 0x1000;
        assert_eq!(d.update_status(), 0);
        assert_eq!(d.open(param), ERR_INVALID_STATE, "Open accepted before Initialize");
        assert_eq!(d.open(0), ERR_PARAM_INVALID, "null parameter was not reported first");
        assert_eq!(d.close(), ERR_NOT_INITIALIZED);
        assert_eq!(d.terminate(), ERR_NOT_INITIALIZED);

        assert_eq!(d.initialize(), OK);
        assert_eq!(d.update_status(), 1);
        assert_eq!(d.initialize(), ERR_ALREADY_INITIALIZED);
        assert_eq!(d.open(0), ERR_PARAM_INVALID);
        assert_eq!(d.update_status(), 1, "dialog finished before Open");
        assert_eq!(d.result_error(0x1000), Some(crate::common_dialog::ERR_NOT_FINISHED), "no result before it finishes");

        assert_eq!(d.open(param), OK);
        assert_eq!(d.update_status(), 3, "the poll loop waits for 3");
        assert_eq!(d.open(param), OK, "a second Open after finishing was rejected");
        assert_eq!(d.result_error(0), Some(ERR_PARAM_INVALID));
        assert_eq!(d.result_error(0x1000), None);

        assert_eq!(d.close(), OK);
        assert_eq!(d.update_status(), 3);
        assert_eq!(d.terminate(), OK);
        assert_eq!(d.update_status(), 0);
        assert_eq!(d.terminate(), ERR_NOT_INITIALIZED);
        assert_eq!(d.result_error(0x1000), Some(ERR_NOT_INITIALIZED));
    }

    /// Close without Open still ends the dialog at 3, and a fresh
    /// Initialize after Terminate starts a new run.
    #[test]
    fn close_without_open_and_second_run() {
        let mut d = Dialog::new();
        assert_eq!(d.initialize(), OK);
        assert_eq!(d.close(), OK);
        assert_eq!(d.update_status(), 3);
        assert_eq!(d.terminate(), OK);
        assert_eq!(d.initialize(), OK);
        assert_eq!(d.update_status(), 1);
    }

    /// Port of the guest dialog test: GetStatus follows the state machine.
    #[test]
    fn get_status_follows_the_dialog() {
        let vm = VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        *dialog() = Dialog::new();
        assert_eq!(cb_get_status(&mut ctx), 0);
        assert_eq!(cb_initialize(&mut ctx), OK);
        assert_eq!(cb_get_status(&mut ctx), 1);
        dialog().open(0x1000);
        assert_eq!(cb_get_status(&mut ctx), 3);
        assert_eq!(cb_terminate(&mut ctx), OK);
        assert_eq!(cb_get_status(&mut ctx), 0);
    }

    #[test]
    fn result_block_is_written_as_forty_bytes() {
        const BASE: u64 = 0x7F50_0000_0000;
        let vm = VirtualMemory::new();
        vm.map(BASE, 0x1000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        assert!(ctx.write_bytes(BASE, &[0xAA; 0x100]));
        assert_eq!(write_result(&mut ctx, BASE), OK);
        let back = ctx.read_bytes(BASE, 0x100).expect("result block readable");
        assert_eq!(&back[0..4], &RESULT_OK.to_le_bytes(), "result field");
        assert!(back[4..RESULT_SIZE].iter().all(|&b| b == 0), "reserved fields were not zeroed");
        assert!(back[RESULT_SIZE..].iter().all(|&b| b == 0xAA), "bytes past the 40-byte block were written");
        let _ = vm.unmap(BASE, 0x1000);
    }
}
