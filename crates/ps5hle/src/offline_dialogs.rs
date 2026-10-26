// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Common dialogs that need PSN or the system network settings:
//! libSceCdlgPlayerReview (player review), libScePlayerSelectionDialog and
//! libSceNetCtlApDialog (access point settings).
//!
//! None of them can be shown offline: Open finishes the dialog at once and
//! its result is "user canceled".

use std::sync::Mutex;

use crate::common_dialog::{STATUS_FINISHED, STATUS_INITIALIZED, STATUS_NONE, STATUS_RUNNING};
use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const OK: HleResult = 0;

// SCE_COMMON_DIALOG_ERROR_*.
const ERR_NOT_INITIALIZED: HleResult = 0x80B8_0003u32 as i32 as i64;
const ERR_ALREADY_INITIALIZED: HleResult = 0x80B8_0004u32 as i32 as i64;
const ERR_NOT_FINISHED: HleResult = 0x80B8_0005u32 as i32 as i64;
const ERR_BUSY: HleResult = 0x80B8_0007u32 as i32 as i64;
const ERR_ARG_NULL: HleResult = 0x80B8_000Du32 as i32 as i64;
const ERR_MEMORY_FAULT: HleResult = 0x8002_000Eu32 as i32 as i64;

/// SCE_COMMON_DIALOG_RESULT_USER_CANCELED.
const RESULT_USER_CANCELED: i32 = 1;

/// How one dialog library differs from the others.
struct Spec {
    tag: &'static str,
    /// Close finishes a dialog that is not running (otherwise it only
    /// checks initialisation).
    close_finishes: bool,
    /// GetResult checks the output pointer before the status.
    result_null_first: bool,
    /// Bytes of the result block written (the result code, then zeros).
    result_size: usize,
}

struct OfflineDialog {
    spec: Spec,
    status: Mutex<i32>,
}

impl OfflineDialog {
    const fn new(spec: Spec) -> Self {
        Self { spec, status: Mutex::new(STATUS_NONE) }
    }

    fn status(&self) -> std::sync::MutexGuard<'_, i32> {
        self.status.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn get(&self) -> HleResult {
        *self.status() as HleResult
    }

    fn initialize(&self) -> HleResult {
        let mut s = self.status();
        if *s != STATUS_NONE {
            return ERR_ALREADY_INITIALIZED;
        }
        *s = STATUS_INITIALIZED;
        OK
    }

    fn terminate(&self) -> HleResult {
        let mut s = self.status();
        if *s == STATUS_NONE {
            return ERR_NOT_INITIALIZED;
        }
        *s = STATUS_NONE;
        OK
    }

    fn open(&self, param: u64) -> HleResult {
        let mut s = self.status();
        match *s {
            STATUS_NONE => return ERR_NOT_INITIALIZED,
            STATUS_RUNNING => return ERR_BUSY,
            _ => {}
        }
        if param == 0 {
            return ERR_ARG_NULL;
        }
        log::info!("[{}] open: not available offline, the dialog is canceled", self.spec.tag);
        *s = STATUS_FINISHED;
        OK
    }

    fn close(&self) -> HleResult {
        let mut s = self.status();
        if *s == STATUS_NONE {
            return ERR_NOT_INITIALIZED;
        }
        if self.spec.close_finishes {
            *s = STATUS_FINISHED;
        }
        OK
    }

    fn get_result(&self, ctx: &CpuContext, out: u64) -> HleResult {
        let s = *self.status();
        if s == STATUS_NONE {
            return ERR_NOT_INITIALIZED;
        }
        if out == 0 && self.spec.result_null_first {
            return ERR_ARG_NULL;
        }
        if s != STATUS_FINISHED {
            return ERR_NOT_FINISHED;
        }
        if out == 0 {
            return ERR_ARG_NULL;
        }
        let mut r = vec![0u8; self.spec.result_size];
        r[..4].copy_from_slice(&RESULT_USER_CANCELED.to_le_bytes());
        if ctx.write_bytes(out, &r) { OK } else { ERR_MEMORY_FAULT }
    }
}

/// ScePlayerReviewDialogResult starts with the result code; Close does not
/// change the status.
static PLAYER_REVIEW: OfflineDialog = OfflineDialog::new(Spec {
    tag: "PLAYERREVIEWDIALOG",
    close_finishes: false,
    result_null_first: true,
    result_size: 4,
});

static PLAYER_SELECTION: OfflineDialog = OfflineDialog::new(Spec {
    tag: "PLAYERSELECTIONDIALOG",
    close_finishes: true,
    result_null_first: true,
    result_size: 4,
});

/// SceNetCtlApDialogResult { i32 result; u8 reserved[36]; }.
static NETCTL_AP: OfflineDialog = OfflineDialog::new(Spec {
    tag: "NETCTLAPDIALOG",
    close_finishes: true,
    result_null_first: false,
    result_size: 40,
});

macro_rules! dialog_callbacks {
    ($m:ident, $d:ident) => {
        mod $m {
            use super::*;
            pub fn initialize(_ctx: &mut CpuContext) -> HleResult {
                $d.initialize()
            }
            pub fn terminate(_ctx: &mut CpuContext) -> HleResult {
                $d.terminate()
            }
            pub fn open(ctx: &mut CpuContext) -> HleResult {
                $d.open(ctx.rdi())
            }
            pub fn close(_ctx: &mut CpuContext) -> HleResult {
                $d.close()
            }
            pub fn status(_ctx: &mut CpuContext) -> HleResult {
                $d.get()
            }
            pub fn get_result(ctx: &mut CpuContext) -> HleResult {
                $d.get_result(ctx, ctx.rdi())
            }
        }
    };
}

dialog_callbacks!(review, PLAYER_REVIEW);
dialog_callbacks!(selection, PLAYER_SELECTION);
dialog_callbacks!(ap, NETCTL_AP);

const EXPORTS: &[(&str, &str, &str, HleCallback)] = &[
    ("UtXl-tmi7iw", "scePlayerReviewDialogInitialize", "libSceCdlgPlayerReview", review::initialize),
    ("RmJKkzZFNFA", "scePlayerReviewDialogTerminate", "libSceCdlgPlayerReview", review::terminate),
    ("CtygVRCL+bA", "scePlayerReviewDialogOpen", "libSceCdlgPlayerReview", review::open),
    ("HDMHJ9zsCFg", "scePlayerReviewDialogClose", "libSceCdlgPlayerReview", review::close),
    ("UYw6RlK7bcQ", "scePlayerReviewDialogGetStatus", "libSceCdlgPlayerReview", review::status),
    ("4J5F23VgTjY", "scePlayerReviewDialogUpdateStatus", "libSceCdlgPlayerReview", review::status),
    ("VuKbx6zlEG4", "scePlayerReviewDialogGetResult", "libSceCdlgPlayerReview", review::get_result),
    ("DrDfu7uq6JY", "scePlayerSelectionDialogInitialize", "libScePlayerSelectionDialog", selection::initialize),
    ("CgdJ1PkIsE4", "scePlayerSelectionDialogTerminate", "libScePlayerSelectionDialog", selection::terminate),
    ("9Oj3MIJ+Zjk", "scePlayerSelectionDialogOpen", "libScePlayerSelectionDialog", selection::open),
    ("0fyTCPOHGX4", "scePlayerSelectionDialogClose", "libScePlayerSelectionDialog", selection::close),
    ("l7oQ33rm9TI", "scePlayerSelectionDialogGetStatus", "libScePlayerSelectionDialog", selection::status),
    ("PBmrW72wbPQ", "scePlayerSelectionDialogUpdateStatus", "libScePlayerSelectionDialog", selection::status),
    ("GtF-2gJqsts", "scePlayerSelectionDialogGetResult", "libScePlayerSelectionDialog", selection::get_result),
    ("py8-vzIY-RY", "sceNetCtlApDialogInitialize", "libSceNetCtlApDialog", ap::initialize),
    ("5X+QFaVtiKg", "sceNetCtlApDialogTerminate", "libSceNetCtlApDialog", ap::terminate),
    ("SKJQwgudGFg", "sceNetCtlApDialogOpen", "libSceNetCtlApDialog", ap::open),
    ("3QtqTA0uMmU", "sceNetCtlApDialogClose", "libSceNetCtlApDialog", ap::close),
    ("eBWS-HorSWY", "sceNetCtlApDialogGetStatus", "libSceNetCtlApDialog", ap::status),
    ("ys1JAOGzNG0", "sceNetCtlApDialogUpdateStatus", "libSceNetCtlApDialog", ap::status),
    ("ZaEVF6WqBP8", "sceNetCtlApDialogGetResult", "libSceNetCtlApDialog", ap::get_result),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, lib, cb) in EXPORTS {
        reg.register_fn(nid, name, lib, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    fn setup(base: u64) -> (ps5vm::VirtualMemory, u64) {
        let vm = ps5vm::VirtualMemory::new();
        vm.map(base, 0x1000, ps5vm::MemProtect::RW).unwrap();
        (vm, base)
    }

    fn call(ctx: &mut CpuContext, f: HleCallback, arg: u64) -> HleResult {
        ctx.set(Rdi, arg);
        f(ctx)
    }

    /// Port of the guest player review dialog test.
    #[test]
    fn player_review_dialog() {
        use review::*;
        let (vm, base) = setup(0x5FA1_0000_0000);
        let mut ctx = CpuContext::new(&vm);
        let (param, result) = (base, base + 0x100);
        assert_eq!(call(&mut ctx, terminate, 0), ERR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, status, 0), STATUS_NONE as i64);
        assert_eq!(call(&mut ctx, open, param), ERR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, close, 0), ERR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, get_result, result), ERR_NOT_INITIALIZED);

        assert_eq!(call(&mut ctx, initialize, 0), OK);
        assert_eq!(call(&mut ctx, initialize, 0), ERR_ALREADY_INITIALIZED);
        assert_eq!(call(&mut ctx, status, 0), STATUS_INITIALIZED as i64);
        assert_eq!(call(&mut ctx, get_result, result), ERR_NOT_FINISHED);
        assert_eq!(call(&mut ctx, close, 0), OK);
        assert_eq!(call(&mut ctx, status, 0), STATUS_INITIALIZED as i64, "close does not finish");
        assert_eq!(call(&mut ctx, terminate, 0), OK);
        assert_eq!(call(&mut ctx, terminate, 0), ERR_NOT_INITIALIZED);

        assert_eq!(call(&mut ctx, initialize, 0), OK);
        assert_eq!(call(&mut ctx, open, 0), ERR_ARG_NULL);
        assert_eq!(call(&mut ctx, open, param), OK);
        assert_eq!(call(&mut ctx, status, 0), STATUS_FINISHED as i64);
        assert_eq!(call(&mut ctx, get_result, 0), ERR_ARG_NULL);
        ctx.write_bytes(result, &[0xEE; 8]);
        assert_eq!(call(&mut ctx, get_result, result), OK);
        assert_eq!(ctx.read_u32(result), Some(RESULT_USER_CANCELED as u32));
        assert_eq!(ctx.read_u32(result + 4), Some(0xEEEE_EEEE));
        assert_eq!(call(&mut ctx, close, 0), OK);
        assert_eq!(call(&mut ctx, terminate, 0), OK);
        assert_eq!(call(&mut ctx, status, 0), STATUS_NONE as i64);
    }

    /// Port of the guest player selection dialog test.
    #[test]
    fn player_selection_dialog() {
        use selection::*;
        let (vm, base) = setup(0x5FA2_0000_0000);
        let mut ctx = CpuContext::new(&vm);
        let (param, result) = (base, base + 0x100);
        ctx.write_bytes(result, &[0xFF; 8]);
        assert_eq!(call(&mut ctx, terminate, 0), ERR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, open, param), ERR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, close, 0), ERR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, get_result, result), ERR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, status, 0), 0);

        assert_eq!(call(&mut ctx, initialize, 0), OK);
        assert_eq!(call(&mut ctx, initialize, 0), ERR_ALREADY_INITIALIZED);
        assert_eq!(call(&mut ctx, status, 0), 1);
        assert_eq!(call(&mut ctx, open, 0), ERR_ARG_NULL);
        assert_eq!(call(&mut ctx, get_result, result), ERR_NOT_FINISHED);
        assert_eq!(call(&mut ctx, open, param), OK);
        assert_eq!(call(&mut ctx, status, 0), STATUS_FINISHED as i64);
        assert_eq!(call(&mut ctx, get_result, 0), ERR_ARG_NULL);
        assert_eq!(call(&mut ctx, get_result, result), OK);
        assert_eq!(ctx.read_u32(result), Some(1));
        assert_eq!(ctx.read_u32(result + 4), Some(u32::MAX), "only the result code is written");
        assert_eq!(call(&mut ctx, close, 0), OK);
        assert_eq!(call(&mut ctx, terminate, 0), OK);
        assert_eq!(call(&mut ctx, terminate, 0), ERR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, close, 0), ERR_NOT_INITIALIZED);
    }

    /// Port of the guest access point dialog test.
    #[test]
    fn netctl_ap_dialog() {
        use ap::*;
        let (vm, base) = setup(0x5FA3_0000_0000);
        let mut ctx = CpuContext::new(&vm);
        let (param, result) = (base, base + 0x100);
        let canceled = |ctx: &CpuContext| {
            let r = ctx.read_bytes(result, 44).unwrap();
            assert_eq!(&r[..4], &1i32.to_le_bytes());
            assert!(r[4..40].iter().all(|&b| b == 0));
            assert!(r[40..].iter().all(|&b| b == 0x5A), "40-byte result");
        };
        ctx.write_bytes(result, &[0x5A; 44]);
        assert_eq!(call(&mut ctx, terminate, 0), ERR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, open, param), ERR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, close, 0), ERR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, get_result, result), ERR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, status, 0), STATUS_NONE as i64);

        assert_eq!(call(&mut ctx, initialize, 0), OK);
        assert_eq!(call(&mut ctx, initialize, 0), ERR_ALREADY_INITIALIZED);
        assert_eq!(call(&mut ctx, open, 0), ERR_ARG_NULL);
        assert_eq!(call(&mut ctx, status, 0), STATUS_INITIALIZED as i64);
        assert_eq!(call(&mut ctx, get_result, 0), ERR_NOT_FINISHED, "status is checked before the pointer");
        assert_eq!(call(&mut ctx, get_result, result), ERR_NOT_FINISHED);
        assert_eq!(call(&mut ctx, open, param), OK);
        assert_eq!(call(&mut ctx, status, 0), STATUS_FINISHED as i64);
        assert_eq!(call(&mut ctx, get_result, 0), ERR_ARG_NULL);
        assert_eq!(call(&mut ctx, get_result, result), OK);
        canceled(&ctx);
        assert_eq!(call(&mut ctx, open, param), OK);
        assert_eq!(call(&mut ctx, close, 0), OK);
        assert_eq!(call(&mut ctx, terminate, 0), OK);
        assert_eq!(call(&mut ctx, terminate, 0), ERR_NOT_INITIALIZED);

        // Close without Open finishes the dialog as canceled.
        assert_eq!(call(&mut ctx, initialize, 0), OK);
        assert_eq!(call(&mut ctx, close, 0), OK);
        assert_eq!(call(&mut ctx, status, 0), STATUS_FINISHED as i64);
        ctx.write_bytes(result, &[0x5A; 44]);
        assert_eq!(call(&mut ctx, get_result, result), OK);
        canceled(&ctx);
        assert_eq!(call(&mut ctx, terminate, 0), OK);
    }
}
