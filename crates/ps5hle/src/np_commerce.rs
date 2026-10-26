// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceNpCommerce — PlayStation Store commerce dialog, offline.
//!
//! Offline: no store. Opening the dialog finishes it at once: a valid
//! request completes with "user canceled" so games take their no-purchase
//! path, an invalid one completes with PARAM_INVALID as its result.

use std::sync::Mutex;

use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};

const LIB: &str = "libSceNpCommerce";

const ERR_OK: HleResult = 0;
const ERR_NOT_INITIALIZED: HleResult = 0x80B8_0003u32 as i32 as i64;
const ERR_ALREADY_INITIALIZED: HleResult = 0x80B8_0004u32 as i32 as i64;
const ERR_NOT_FINISHED: HleResult = 0x80B8_0005u32 as i32 as i64;
const ERR_PARAMETER_INVALID: HleResult = 0x80B8_000Au32 as i32 as i64;
const ERR_ARGUMENT_NULL: HleResult = 0x80B8_000Du32 as i32 as i64;

const RESULT_USER_CANCELED: i32 = 1;

/// SceNpCommerceDialogParam: common dialog base (0x30), size +0x30,
/// userId +0x34, mode +0x38, serviceLabel +0x3C, targets +0x40,
/// numTargets +0x48, features +0x50, userData +0x58, reserved[32] +0x60.
const PARAM_SIZE: i32 = 0x80;
const PARAM_SIZE_OPEN2: i32 = 0x88;
const PARAM_RESERVED: u64 = 0x60;
/// SceNpCommerceDialogResult: result +0, authorized +4, userData +8, reserved[32].
const RESULT_SIZE: usize = 0x30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
enum Status {
    None = 0,
    Initialized = 1,
    Finished = 3,
}

/// Completion result kept for sceNpCommerceDialogGetResult.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DialogResult {
    result: i32,
    user_data: u64,
}

/// Fields of an open request that decide the outcome.
struct Request {
    size: i32,
    mode: i32,
    user_data: u64,
    reserved_clear: bool,
}

struct DialogState {
    status: Status,
    result: DialogResult,
}

impl DialogState {
    const fn new() -> Self {
        Self { status: Status::None, result: DialogResult { result: 0, user_data: 0 } }
    }

    fn initialize(&mut self) -> HleResult {
        if self.status != Status::None {
            return ERR_ALREADY_INITIALIZED;
        }
        self.status = Status::Initialized;
        ERR_OK
    }

    fn terminate(&mut self) -> HleResult {
        if self.status == Status::None {
            return ERR_NOT_INITIALIZED;
        }
        self.status = Status::None;
        self.result = DialogResult::default();
        ERR_OK
    }

    /// The dialog finishes at open; invalid parameters become its result.
    fn open(&mut self, req: &Request, required_size: i32) -> HleResult {
        self.result = DialogResult::default();
        if req.size == required_size {
            self.result.user_data = req.user_data;
        }
        self.status = Status::Finished;
        if req.size != required_size || !(0..=5).contains(&req.mode) || !req.reserved_clear {
            self.result.result = ERR_PARAMETER_INVALID as i32;
            return ERR_PARAMETER_INVALID;
        }
        self.result.result = RESULT_USER_CANCELED;
        ERR_OK
    }

    fn update_status(&self) -> HleResult {
        self.status as i32 as i64
    }

    /// Close ends an initialized dialog as canceled; a finished one keeps
    /// its result.
    fn close(&mut self) -> HleResult {
        match self.status {
            Status::None => ERR_NOT_INITIALIZED,
            Status::Initialized => {
                self.status = Status::Finished;
                self.result = DialogResult { result: RESULT_USER_CANCELED, user_data: 0 };
                ERR_OK
            }
            Status::Finished => ERR_OK,
        }
    }

    fn result(&self) -> Result<DialogResult, HleResult> {
        match self.status {
            Status::Finished => Ok(self.result),
            _ => Err(ERR_NOT_FINISHED),
        }
    }
}

static STATE: Mutex<DialogState> = Mutex::new(DialogState::new());

fn state() -> std::sync::MutexGuard<'static, DialogState> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn cb_dialog_initialize(_ctx: &mut CpuContext) -> HleResult {
    state().initialize()
}

fn cb_dialog_terminate(_ctx: &mut CpuContext) -> HleResult {
    state().terminate()
}

fn cb_dialog_update_status(_ctx: &mut CpuContext) -> HleResult {
    state().update_status()
}

/// sceNpCommerceDialogGetStatus(): the status without polling.
fn cb_dialog_get_status(_ctx: &mut CpuContext) -> HleResult {
    state().update_status()
}

fn cb_dialog_close(_ctx: &mut CpuContext) -> HleResult {
    state().close()
}

/// Copies the completion result out and returns its result code
/// (1 = user canceled).
fn cb_dialog_get_result(ctx: &mut CpuContext) -> HleResult {
    let st = state();
    if st.status == Status::None {
        return ERR_NOT_INITIALIZED;
    }
    let out = ctx.rdi();
    if out == 0 {
        return ERR_ARGUMENT_NULL;
    }
    let r = match st.result() {
        Ok(r) => r,
        Err(e) => return e,
    };
    let mut buf = [0u8; RESULT_SIZE];
    buf[0..4].copy_from_slice(&r.result.to_le_bytes());
    buf[8..16].copy_from_slice(&r.user_data.to_le_bytes());
    if !ctx.write_bytes(out, &buf) {
        return ERR_ARGUMENT_NULL;
    }
    r.result as i64
}

fn read_request(ctx: &CpuContext, addr: u64) -> Option<Request> {
    let reserved = ctx.read_bytes(addr + PARAM_RESERVED, 32)?;
    Some(Request {
        size: ctx.read_u32(addr + 0x30)? as i32,
        mode: ctx.read_u32(addr + 0x38)? as i32,
        user_data: ctx.read_u64(addr + 0x58)?,
        reserved_clear: reserved.iter().all(|&b| b == 0),
    })
}

fn open(ctx: &CpuContext, required_size: i32) -> HleResult {
    let mut st = state();
    if st.status == Status::None {
        return ERR_NOT_INITIALIZED;
    }
    let param = ctx.rdi();
    if param == 0 {
        return ERR_ARGUMENT_NULL;
    }
    let Some(req) = read_request(ctx, param) else { return ERR_ARGUMENT_NULL };
    log::debug!("[NpCommerce] dialog open size={:#x} mode={}", req.size, req.mode);
    st.open(&req, required_size)
}

fn cb_dialog_open(ctx: &mut CpuContext) -> HleResult {
    open(ctx, PARAM_SIZE)
}

fn cb_dialog_open2(ctx: &mut CpuContext) -> HleResult {
    open(ctx, PARAM_SIZE_OPEN2)
}

/// Position of the PS Store icon overlay (-1 = hidden). There is no system
/// overlay to draw it on; the request is only recorded.
static PS_STORE_ICON: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);
/// SCE_NP_COMMERCE_PS_STORE_ICON_CENTER / LEFT / RIGHT.
const PS_STORE_ICON_POSITIONS: std::ops::RangeInclusive<i32> = 0..=2;

/// sceNpCommerceShowPsStoreIcon(SceNpCommercePsStoreIconPos pos)
fn cb_show_ps_store_icon(ctx: &mut CpuContext) -> HleResult {
    let pos = ctx.rdi() as i32;
    if !PS_STORE_ICON_POSITIONS.contains(&pos) {
        return ERR_PARAMETER_INVALID;
    }
    PS_STORE_ICON.store(pos, std::sync::atomic::Ordering::Relaxed);
    ERR_OK
}

fn cb_hide_ps_store_icon(_ctx: &mut CpuContext) -> HleResult {
    PS_STORE_ICON.store(-1, std::sync::atomic::Ordering::Relaxed);
    ERR_OK
}

/// Layout of the PS Store icon overlay as the title last set it.
static PS_STORE_ICON_LAYOUT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// sceNpCommerceSetPsStoreIconLayout(SceNpCommercePsStoreIconLayout layout):
/// recorded; there is no system overlay to lay out.
fn cb_set_ps_store_icon_layout(ctx: &mut CpuContext) -> HleResult {
    let layout = ctx.rdi() as i32;
    log::debug!("[NpCommerce] PS Store icon layout {layout}");
    PS_STORE_ICON_LAYOUT.store(layout, std::sync::atomic::Ordering::Relaxed);
    ERR_OK
}

pub fn register(reg: &mut HleRegistry) {
    reg.register_fn("DHmwsa6S8Tc", "sceNpCommerceShowPsStoreIcon", LIB, cb_show_ps_store_icon);
    reg.register_fn("dsqCVsNM0Zg", "sceNpCommerceHidePsStoreIcon", LIB, cb_hide_ps_store_icon);
    reg.register_fn("uKTDW8hk-ts", "sceNpCommerceSetPsStoreIconLayout", LIB, cb_set_ps_store_icon_layout);
    reg.register_fn("0aR2aWmQal4", "sceNpCommerceDialogInitialize", LIB, cb_dialog_initialize);
    reg.register_fn("m-I92Ab50W8", "sceNpCommerceDialogTerminate", LIB, cb_dialog_terminate);
    reg.register_fn("DfSCDRA3EjY", "sceNpCommerceDialogOpen", LIB, cb_dialog_open);
    reg.register_fn("IXmfUaze9So", "sceNpCommerceDialogOpen2", LIB, cb_dialog_open2);
    reg.register_fn("LR5cwFMMCVE", "sceNpCommerceDialogUpdateStatus", LIB, cb_dialog_update_status);
    reg.register_fn("CCbC+lqqvF0", "sceNpCommerceDialogGetStatus", LIB, cb_dialog_get_status);
    reg.register_fn("NU3ckGHMFXo", "sceNpCommerceDialogClose", LIB, cb_dialog_close);
    reg.register_fn("r42bWcQbtZY", "sceNpCommerceDialogGetResult", LIB, cb_dialog_get_result);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ps_store_icon_nids() {
        assert_eq!(ps5nid::compute_nid("sceNpCommerceShowPsStoreIcon"), "DHmwsa6S8Tc");
        assert_eq!(ps5nid::compute_nid("sceNpCommerceHidePsStoreIcon"), "dsqCVsNM0Zg");
    }

    #[test]
    fn nids_match_names() {
        for (nid, name) in [
            ("uKTDW8hk-ts", "sceNpCommerceSetPsStoreIconLayout"),
            ("CCbC+lqqvF0", "sceNpCommerceDialogGetStatus"),
            ("NU3ckGHMFXo", "sceNpCommerceDialogClose"),
        ] {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn ps_store_icon_layout_is_recorded() {
        use crate::context::CpuRegister::Rdi;
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, 1);
        assert_eq!(cb_set_ps_store_icon_layout(&mut ctx), ERR_OK);
        assert_eq!(PS_STORE_ICON_LAYOUT.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    /// Port of the guest commerce dialog test: GetStatus tracks the dialog,
    /// Close keeps a finished dialog finished and ends an unopened one as
    /// canceled.
    #[test]
    fn dialog_status_and_close() {
        let mut d = DialogState::new();
        assert_eq!(d.update_status(), 0);
        assert_eq!(d.close(), ERR_NOT_INITIALIZED);
        assert_eq!(d.initialize(), ERR_OK);
        assert_eq!(d.update_status(), 1);
        let req = Request { size: PARAM_SIZE, mode: 0, user_data: 7, reserved_clear: true };
        assert_eq!(d.open(&req, PARAM_SIZE), ERR_OK);
        assert_eq!(d.update_status(), 3);
        assert_eq!(d.close(), ERR_OK);
        assert_eq!(d.update_status(), 3);
        assert_eq!(d.result(), Ok(DialogResult { result: RESULT_USER_CANCELED, user_data: 7 }));
        assert_eq!(d.terminate(), ERR_OK);
        assert_eq!(d.update_status(), 0);
        assert_eq!(d.initialize(), ERR_OK);
        assert_eq!(d.close(), ERR_OK);
        assert_eq!(d.update_status(), 3);
        assert_eq!(d.result().map(|r| r.result), Ok(RESULT_USER_CANCELED));
    }

    const BASE: u64 = 0x5C81_0000_0000;

    /// Port of the dialog lifecycle test: open finishes without a store,
    /// cancellation carries the caller's user data, invalid parameters are
    /// reported as the completion result.
    #[test]
    fn dialog_finishes_at_open() {
        use crate::context::CpuRegister::Rdi;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let param = BASE;
        let result = BASE + 0x200;
        ctx.write_u64(param, 0x30);
        ctx.write_u32(param + 0x30, PARAM_SIZE as u32);
        ctx.write_u32(param + 0x34, 1);
        ctx.write_u64(param + 0x58, 0xABCD_0000);
        let call = |ctx: &mut CpuContext, f: fn(&mut CpuContext) -> HleResult, arg: u64| {
            ctx.set(Rdi, arg);
            f(ctx)
        };
        let user_data = |ctx: &CpuContext| ctx.read_u64(result + 8).unwrap();
        let authorized = |ctx: &CpuContext| ctx.read_bytes(result + 4, 1).unwrap()[0] != 0;

        assert_eq!(call(&mut ctx, cb_dialog_open, param), ERR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, cb_dialog_get_result, result), ERR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, cb_dialog_initialize, 0), ERR_OK);
        assert_eq!(call(&mut ctx, cb_dialog_update_status, 0), 1);
        assert_eq!(call(&mut ctx, cb_dialog_get_result, result), ERR_NOT_FINISHED);
        assert_eq!(call(&mut ctx, cb_dialog_open, 0), ERR_ARGUMENT_NULL);
        assert_eq!(call(&mut ctx, cb_dialog_update_status, 0), 1);
        assert_eq!(call(&mut ctx, cb_dialog_get_result, 0), ERR_ARGUMENT_NULL);

        assert_eq!(call(&mut ctx, cb_dialog_open, param), ERR_OK);
        assert_eq!(call(&mut ctx, cb_dialog_update_status, 0), 3);
        ctx.write_u64(param + 0x58, 0);
        ctx.write_bytes(result + 4, &[1]);
        assert_eq!(call(&mut ctx, cb_dialog_get_result, result), 1);
        assert_eq!(ctx.read_u32(result), Some(1));
        assert!(!authorized(&ctx));
        assert_eq!(user_data(&ctx), 0xABCD_0000, "cancellation returns the original user data");

        ctx.write_u32(param + 0x38, 5);
        assert_eq!(call(&mut ctx, cb_dialog_open, param), ERR_OK);
        assert_eq!(call(&mut ctx, cb_dialog_get_result, result), 1);
        assert_eq!(user_data(&ctx), 0, "reopen gives fresh result data");

        let invalid = |ctx: &mut CpuContext| {
            assert_eq!(call(ctx, cb_dialog_open, param), ERR_PARAMETER_INVALID);
            assert_eq!(call(ctx, cb_dialog_update_status, 0), 3);
            assert_eq!(call(ctx, cb_dialog_get_result, result), ERR_PARAMETER_INVALID);
            assert_eq!(ctx.read_u32(result), Some(ERR_PARAMETER_INVALID as u32));
            assert!(!authorized(ctx));
        };
        ctx.write_u32(param + 0x38, 99);
        invalid(&mut ctx);
        ctx.write_u32(param + 0x38, -1i32 as u32);
        invalid(&mut ctx);
        ctx.write_u32(param + 0x38, 0);
        ctx.write_u32(param + 0x30, 64);
        invalid(&mut ctx);
        ctx.write_u32(param + 0x30, PARAM_SIZE as u32);
        ctx.write_bytes(param + PARAM_RESERVED, &[1]);
        invalid(&mut ctx);
        ctx.write_bytes(param + PARAM_RESERVED, &[0]);

        assert_eq!(call(&mut ctx, cb_dialog_open, param), ERR_OK);
        assert_eq!(call(&mut ctx, cb_dialog_get_result, result), 1);
        assert_eq!(call(&mut ctx, cb_dialog_terminate, 0), ERR_OK);
        assert_eq!(call(&mut ctx, cb_dialog_update_status, 0), 0);
        assert_eq!(call(&mut ctx, cb_dialog_get_result, result), ERR_NOT_INITIALIZED);
    }
}
