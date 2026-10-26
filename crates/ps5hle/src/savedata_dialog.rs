// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceSaveDataDialog: headless save-data dialog.
//!
//! An opened dialog stays RUNNING for one sceSaveDataDialogUpdateStatus
//! (some titles require seeing RUNNING before FINISHED) and finishes on the
//! next update with the OK button. A progress bar (mode 5) runs until the
//! title closes it. GetStatus only peeks. Calls in the wrong state return
//! the library's state errors.

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};
use std::sync::Mutex;

pub const STATUS_NONE: i32 = 0;
pub const STATUS_INITIALIZED: i32 = 1;
pub const STATUS_RUNNING: i32 = 2;
pub const STATUS_FINISHED: i32 = 3;
const RESULT_OK: i32 = 0;
const BUTTON_ID_OK: i32 = 1;
const MODE_PROGRESS_BAR: i32 = 5;

pub const ERROR_NOT_INITIALIZED: i32 = 0x80B8_0003u32 as i32;
pub const ERROR_ALREADY_INITIALIZED: i32 = 0x80B8_0004u32 as i32;
pub const ERROR_NOT_FINISHED: i32 = 0x80B8_0005u32 as i32;
pub const ERROR_INVALID_STATE: i32 = 0x80B8_0006u32 as i32;
pub const ERROR_NOT_RUNNING: i32 = 0x80B8_000Bu32 as i32;
pub const ERROR_ARG_NULL: i32 = 0x80B8_000Du32 as i32;

/// SceSaveDataDialogParam: baseParam[48], size @0x30, mode @0x34,
/// dispType @0x38, animParam @0x40, items @0x48, userMsgParam @0x50,
/// sysMsgParam @0x58, errorCodeParam @0x60, progBarParam @0x68,
/// userData @0x70, optionParam @0x78, wizardParam @0x80, reserved[16].
pub const PARAM_SIZE: usize = 0x98;
const PARAM_SIZE_OFFSET: usize = 0x30;
const PARAM_MODE_OFFSET: usize = 0x34;
const PARAM_DISP_TYPE_OFFSET: usize = 0x38;
const PARAM_ITEMS_OFFSET: usize = 0x48;
const PARAM_USER_DATA_OFFSET: usize = 0x70;
/// SceSaveDataDialogItems: userId @0, titleId @8, dirName @0x10, dirNameNum @0x18.
const ITEMS_DIR_NAMES_OFFSET: u64 = 0x10;
const ITEMS_DIR_NAMES_NUM_OFFSET: u64 = 0x18;
/// SceSaveDataDirName { char data[32]; }.
const DIR_NAME_SIZE: usize = 32;
/// SceSaveDataDialogResult: mode @0, result @4, buttonId @8, dirName* @0x10,
/// param* @0x18, userData @0x20, reserved[32].
const RESULT_DIR_NAME_OFFSET: u64 = 0x10;
const RESULT_USER_DATA_OFFSET: u64 = 0x20;

struct State {
    status: i32,
    running_polled: bool,
    mode: i32,
    user_data: u64,
    /// First non-empty directory name of the opened items (NUL-free).
    dir_name: Vec<u8>,
}

static STATE: Mutex<State> =
    Mutex::new(State { status: STATUS_NONE, running_polled: false, mode: 0, user_data: 0, dir_name: Vec::new() });

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|p| p.into_inner())
}

fn reset(s: &mut State, status: i32) {
    s.status = status;
    s.running_polled = false;
    s.mode = 0;
    s.user_data = 0;
    s.dir_name.clear();
}

pub fn initialize() -> i32 {
    let mut s = state();
    if s.status != STATUS_NONE {
        return ERROR_ALREADY_INITIALIZED;
    }
    reset(&mut s, STATUS_INITIALIZED);
    0
}

pub fn get_status() -> i32 {
    state().status
}

pub fn update_status() -> i32 {
    let mut s = state();
    if s.status == STATUS_RUNNING && s.mode != MODE_PROGRESS_BAR {
        if s.running_polled {
            s.status = STATUS_FINISHED;
        }
        s.running_polled = true;
    }
    s.status
}

/// First non-empty name among `num` SceSaveDataDirName at `dir_names`,
/// cut at its NUL (at most 31 chars).
fn first_dir_name(ctx: &CpuContext, dir_names: u64, num: u32) -> Vec<u8> {
    for i in 0..num as u64 {
        let Some(raw) = ctx.read_bytes(dir_names + i * DIR_NAME_SIZE as u64, DIR_NAME_SIZE) else {
            break;
        };
        if raw[0] != 0 {
            let end = raw.iter().position(|&b| b == 0).unwrap_or(DIR_NAME_SIZE).min(DIR_NAME_SIZE - 1);
            return raw[..end].to_vec();
        }
    }
    Vec::new()
}

pub fn open(ctx: &CpuContext, param: u64) -> i32 {
    let raw = if param != 0 { ctx.read_bytes(param, PARAM_SIZE) } else { None };
    let mut s = state();
    if s.status != STATUS_INITIALIZED && s.status != STATUS_FINISHED {
        return ERROR_INVALID_STATE;
    }
    if param == 0 {
        return ERROR_ARG_NULL;
    }
    if let Some(p) = raw {
        let u32_at = |o: usize| u32::from_le_bytes(p[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(p[o..o + 8].try_into().unwrap());
        s.mode = u32_at(PARAM_MODE_OFFSET) as i32;
        s.user_data = u64_at(PARAM_USER_DATA_OFFSET);
        s.dir_name.clear();
        let items = u64_at(PARAM_ITEMS_OFFSET);
        log::debug!(
            "[SAVEDATADIALOG] open size={} mode={} dispType={} items={items:#x} userData={:#x}",
            u32_at(PARAM_SIZE_OFFSET) as i32,
            s.mode,
            u32_at(PARAM_DISP_TYPE_OFFSET) as i32,
            s.user_data
        );
        if items != 0 {
            let dir_names = ctx.read_u64(items + ITEMS_DIR_NAMES_OFFSET).unwrap_or(0);
            let num = ctx.read_u32(items + ITEMS_DIR_NAMES_NUM_OFFSET).unwrap_or(0);
            if dir_names != 0 {
                s.dir_name = first_dir_name(ctx, dir_names, num);
            }
        }
    }
    s.status = STATUS_RUNNING;
    s.running_polled = false;
    0
}

/// mode, result OK, button OK, userData; the opened directory name is
/// copied into the caller's dirName buffer when it supplied one.
pub fn get_result(ctx: &CpuContext, result: u64) -> i32 {
    let s = state();
    if s.status != STATUS_FINISHED {
        return ERROR_NOT_FINISHED;
    }
    if result == 0 {
        return ERROR_ARG_NULL;
    }
    let mut head = [0u8; 12];
    head[0..4].copy_from_slice(&s.mode.to_le_bytes());
    head[4..8].copy_from_slice(&RESULT_OK.to_le_bytes());
    head[8..12].copy_from_slice(&BUTTON_ID_OK.to_le_bytes());
    ctx.write_bytes(result, &head);
    ctx.write_u64(result + RESULT_USER_DATA_OFFSET, s.user_data);
    let dir_name = ctx.read_u64(result + RESULT_DIR_NAME_OFFSET).unwrap_or(0);
    if dir_name != 0 {
        // A caller that supplied a dirName buffer always has the selected
        // name written to it: an empty selection (or one whose items had
        // no names) still terminates the buffer.
        let mut name = s.dir_name.clone();
        name.push(0);
        ctx.write_bytes(dir_name, &name);
    }
    0
}

pub fn close() -> i32 {
    let mut s = state();
    if s.status != STATUS_RUNNING {
        return ERROR_NOT_RUNNING;
    }
    s.status = STATUS_FINISHED;
    s.running_polled = false;
    0
}

pub fn terminate() -> i32 {
    let mut s = state();
    if s.status == STATUS_NONE {
        return ERROR_NOT_INITIALIZED;
    }
    reset(&mut s, STATUS_NONE);
    0
}

fn cb_initialize(_ctx: &mut CpuContext) -> HleResult {
    initialize() as HleResult
}

fn cb_open(ctx: &mut CpuContext) -> HleResult {
    open(ctx, ctx.rdi()) as HleResult
}

fn cb_get_status(_ctx: &mut CpuContext) -> HleResult {
    get_status() as HleResult
}

fn cb_update_status(_ctx: &mut CpuContext) -> HleResult {
    update_status() as HleResult
}

fn cb_is_ready_to_display(_ctx: &mut CpuContext) -> HleResult {
    1
}

fn cb_get_result(ctx: &mut CpuContext) -> HleResult {
    get_result(ctx, ctx.rdi()) as HleResult
}

fn cb_close(ctx: &mut CpuContext) -> HleResult {
    log::debug!("[SAVEDATADIALOG] close closeParam={:#x}", ctx.rdi());
    close() as HleResult
}

fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    terminate() as HleResult
}

/// ProgressBarInc / ProgressBarSetValue(target, value): no visible bar; only
/// a running dialog takes them.
fn cb_progress_bar(ctx: &mut CpuContext) -> HleResult {
    log::debug!("[SAVEDATADIALOG] progress bar target={} value={}", ctx.rdi() as i32, ctx.rsi() as u32);
    if state().status != STATUS_RUNNING {
        return ERROR_NOT_RUNNING as HleResult;
    }
    0
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("s9e3+YpRnzw", "sceSaveDataDialogInitialize", cb_initialize),
    ("4tPhsP6FpDI", "sceSaveDataDialogOpen", cb_open),
    ("ERKzksauAJA", "sceSaveDataDialogGetStatus", cb_get_status),
    ("KK3Bdg1RWK0", "sceSaveDataDialogUpdateStatus", cb_update_status),
    ("en7gNVnh878", "sceSaveDataDialogIsReadyToDisplay", cb_is_ready_to_display),
    ("yEiJ-qqr6Cg", "sceSaveDataDialogGetResult", cb_get_result),
    ("fH46Lag88XY", "sceSaveDataDialogClose", cb_close),
    ("YuH2FA7azqQ", "sceSaveDataDialogTerminate", cb_terminate),
    ("V-uEeFKARJU", "sceSaveDataDialogProgressBarInc", cb_progress_bar),
    ("hay1CfTmLyA", "sceSaveDataDialogProgressBarSetValue", cb_progress_bar),
];

pub fn register(reg: &mut HleRegistry) {
    const L: &str = "libSceSaveDataDialog";
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, L, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = 0x5D35_0000_0000;
    const PARAM: u64 = BASE;
    const ITEMS: u64 = BASE + 0x100;
    const DIR_NAMES: u64 = BASE + 0x200;
    static LOCK: Mutex<()> = Mutex::new(());

    fn check_latest_status() {
        let updated = update_status();
        assert_eq!(updated, get_status(), "UpdateStatus must return the current status");
    }

    fn running_then_finished(ctx: &CpuContext) {
        assert_eq!(open(ctx, PARAM), 0);
        for _ in 0..3 {
            assert_eq!(get_status(), STATUS_RUNNING, "peek before first update");
        }
        assert_eq!(update_status(), STATUS_RUNNING, "first update");
        for _ in 0..3 {
            assert_eq!(get_status(), STATUS_RUNNING, "peek after first update");
        }
        assert_eq!(update_status(), STATUS_FINISHED, "second update");
        assert_eq!(get_status(), STATUS_FINISHED, "finished peek");
        assert_eq!(update_status(), STATUS_FINISHED, "finished update");
    }

    fn close_and_reset(ctx: &CpuContext) {
        assert_eq!(open(ctx, PARAM), 0);
        assert_eq!(close(), 0);
        assert_eq!(update_status(), STATUS_FINISHED, "poll after Close");
        assert_eq!(get_status(), STATUS_FINISHED, "peek after Close");
        assert_eq!(close(), ERROR_NOT_RUNNING, "a finished dialog does not close");
        running_then_finished(ctx);

        assert_eq!(open(ctx, PARAM), 0);
        assert_eq!(update_status(), STATUS_RUNNING, "poll before Terminate");
        assert_eq!(terminate(), 0, "terminate while running");
        assert_eq!(terminate(), ERROR_NOT_INITIALIZED);
        assert_eq!(get_status(), STATUS_NONE);
        assert_eq!(update_status(), STATUS_NONE);
        assert_eq!(initialize(), 0);
        assert_eq!(update_status(), STATUS_INITIALIZED);
        running_then_finished(ctx);
    }

    /// Guest param with mode USER_MSG, dispType SAVE and one item.
    fn setup(ctx: &CpuContext) {
        let mut p = [0u8; PARAM_SIZE];
        p[0..8].copy_from_slice(&48u64.to_le_bytes());
        p[PARAM_SIZE_OFFSET..PARAM_SIZE_OFFSET + 4].copy_from_slice(&(PARAM_SIZE as u32).to_le_bytes());
        p[PARAM_MODE_OFFSET..PARAM_MODE_OFFSET + 4].copy_from_slice(&2u32.to_le_bytes());
        p[PARAM_DISP_TYPE_OFFSET..PARAM_DISP_TYPE_OFFSET + 4].copy_from_slice(&1u32.to_le_bytes());
        p[PARAM_ITEMS_OFFSET..PARAM_ITEMS_OFFSET + 8].copy_from_slice(&ITEMS.to_le_bytes());
        p[PARAM_USER_DATA_OFFSET..PARAM_USER_DATA_OFFSET + 8].copy_from_slice(&0xABCDu64.to_le_bytes());
        ctx.write_bytes(PARAM, &p);
        ctx.write_u32(ITEMS, 1000);
        ctx.write_u64(ITEMS + ITEMS_DIR_NAMES_OFFSET, DIR_NAMES);
        ctx.write_u32(ITEMS + ITEMS_DIR_NAMES_NUM_OFFSET, 2);
        ctx.write_bytes(DIR_NAMES, &[0u8; 64]);
        ctx.write_bytes(DIR_NAMES + 32, b"SAVE0001\0");
    }

    #[test]
    fn reference_status_polling() {
        let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let ctx = CpuContext::new(&vm);
        setup(&ctx);

        let _ = terminate();
        assert_eq!(initialize(), 0);
        check_latest_status();
        assert_eq!(open(&ctx, PARAM), 0);
        check_latest_status();
        check_latest_status();
        running_then_finished(&ctx);
        close_and_reset(&ctx);
        terminate();
        check_latest_status();
    }

    /// State errors and the progress bar, which runs until Close.
    #[test]
    fn states_and_progress_bar() {
        let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D47_0300_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let _ = terminate();
        assert_eq!(terminate(), ERROR_NOT_INITIALIZED);
        assert_eq!(close(), ERROR_NOT_RUNNING);
        assert_eq!(get_result(&ctx, BASE + 0x300), ERROR_NOT_FINISHED);
        assert_eq!(open(&ctx, BASE), ERROR_INVALID_STATE);
        assert_eq!(initialize(), 0);
        assert_eq!(initialize(), ERROR_ALREADY_INITIALIZED);
        assert_eq!(cb_progress_bar(&mut ctx), ERROR_NOT_RUNNING as HleResult);
        assert_eq!(open(&ctx, 0), ERROR_ARG_NULL);

        let mut p = [0u8; PARAM_SIZE];
        p[PARAM_MODE_OFFSET..PARAM_MODE_OFFSET + 4].copy_from_slice(&(MODE_PROGRESS_BAR as u32).to_le_bytes());
        ctx.write_bytes(BASE, &p);
        assert_eq!(open(&ctx, BASE), 0);
        assert_eq!(open(&ctx, BASE), ERROR_INVALID_STATE, "already running");
        for _ in 0..3 {
            assert_eq!(update_status(), STATUS_RUNNING, "a progress bar runs until Close");
        }
        assert_eq!(cb_progress_bar(&mut ctx), 0);
        assert_eq!(get_result(&ctx, BASE + 0x300), ERROR_NOT_FINISHED);
        assert_eq!(close(), 0);
        assert_eq!(update_status(), STATUS_FINISHED);
        assert_eq!(get_result(&ctx, 0), ERROR_ARG_NULL);
        assert_eq!(get_result(&ctx, BASE + 0x300), 0);
        assert_eq!(ctx.read_u32(BASE + 0x300), Some(MODE_PROGRESS_BAR as u32));
        assert_eq!(cb_progress_bar(&mut ctx), ERROR_NOT_RUNNING as HleResult);
        assert_eq!(terminate(), 0);
    }

    #[test]
    fn result_reports_mode_user_data_and_dir_name() {
        let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D36_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let ctx = CpuContext::new(&vm);
        let mut p = [0u8; PARAM_SIZE];
        p[PARAM_MODE_OFFSET..PARAM_MODE_OFFSET + 4].copy_from_slice(&3u32.to_le_bytes());
        p[PARAM_ITEMS_OFFSET..PARAM_ITEMS_OFFSET + 8].copy_from_slice(&(BASE + 0x100).to_le_bytes());
        p[PARAM_USER_DATA_OFFSET..PARAM_USER_DATA_OFFSET + 8].copy_from_slice(&0x1234u64.to_le_bytes());
        ctx.write_bytes(BASE, &p);
        ctx.write_u64(BASE + 0x100 + ITEMS_DIR_NAMES_OFFSET, BASE + 0x200);
        ctx.write_u32(BASE + 0x100 + ITEMS_DIR_NAMES_NUM_OFFSET, 1);
        ctx.write_bytes(BASE + 0x200, b"SLOT_A\0");

        let _ = terminate();
        assert_eq!(initialize(), 0);
        assert_eq!(open(&ctx, BASE), 0);
        assert_eq!(close(), 0);
        assert_eq!(get_result(&ctx, 0), ERROR_ARG_NULL);
        ctx.write_bytes(BASE + 0x300, &[0u8; 0x48]);
        ctx.write_u64(BASE + 0x300 + RESULT_DIR_NAME_OFFSET, BASE + 0x400);
        assert_eq!(get_result(&ctx, BASE + 0x300), 0);
        assert_eq!(ctx.read_u32(BASE + 0x300), Some(3));
        assert_eq!(ctx.read_u32(BASE + 0x304), Some(RESULT_OK as u32));
        assert_eq!(ctx.read_u32(BASE + 0x308), Some(BUTTON_ID_OK as u32));
        assert_eq!(ctx.read_u64(BASE + 0x300 + RESULT_USER_DATA_OFFSET), Some(0x1234));
        assert_eq!(ctx.read_cstring(BASE + 0x400, 32).as_deref(), Some("SLOT_A"));
        assert_eq!(terminate(), 0);
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }
}
