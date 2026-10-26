// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Common-dialog family HLE: libSceCommonDialog, libSceMsgDialog,
//! libSceErrorDialog and the system-dialog host state shared by the error
//! and sign-in dialogs.

use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Mutex;

const OK: HleResult = 0;

// SceCommonDialogStatus.
pub const STATUS_NONE: i32 = 0;
pub const STATUS_INITIALIZED: i32 = 1;
pub const STATUS_RUNNING: i32 = 2;
pub const STATUS_FINISHED: i32 = 3;

// SCE_COMMON_DIALOG_ERROR_*.
pub(crate) const ERR_NOT_INITIALIZED: HleResult = 0x80B8_0003u32 as i32 as i64;
pub(crate) const ERR_NOT_FINISHED: HleResult = 0x80B8_0005u32 as i32 as i64;
const ERR_NOT_RUNNING: HleResult = 0x80B8_000Bu32 as i32 as i64;
pub(crate) const ERR_ARG_NULL: HleResult = 0x80B8_000Du32 as i32 as i64;

/// Status machine of a headless dialog (NONE -> INITIALIZED -> RUNNING -> FINISHED).
pub struct DialogState(AtomicI32);

impl DialogState {
    pub const fn new() -> Self {
        Self(AtomicI32::new(STATUS_NONE))
    }

    pub fn get(&self) -> i32 {
        self.0.load(Ordering::SeqCst)
    }

    pub fn set(&self, s: i32) -> i32 {
        self.0.swap(s, Ordering::SeqCst)
    }

    /// CAS; returns the previous value.
    pub fn transition(&self, from: i32, to: i32) -> i32 {
        match self.0.compare_exchange(from, to, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(v) | Err(v) => v,
        }
    }

    /// Initialize is idempotent and only promotes from NONE, so a repeated
    /// call cannot clobber a running or finished dialog.
    pub fn initialize(&self) -> HleResult {
        self.transition(STATUS_NONE, STATUS_INITIALIZED);
        OK
    }

    pub fn terminate(&self) -> HleResult {
        if self.set(STATUS_NONE) == STATUS_NONE { ERR_NOT_INITIALIZED } else { OK }
    }

    /// A running dialog is dismissed by the first poll.
    pub fn poll(&self) -> i32 {
        self.transition(STATUS_RUNNING, STATUS_FINISHED);
        self.get()
    }

    pub fn close(&self) -> HleResult {
        if self.transition(STATUS_RUNNING, STATUS_FINISHED) != STATUS_RUNNING {
            ERR_NOT_RUNNING
        } else {
            OK
        }
    }
}

impl Default for DialogState {
    fn default() -> Self {
        Self::new()
    }
}


// ---------------------------------------------------------------------------
// System dialogs (error, sign-in): state shown by a host UI
// ---------------------------------------------------------------------------

/// The error and sign-in dialogs are system UI drawn by the host. While one
/// runs, a host UI shows it (`host_snapshot`) and dismisses it with
/// `host_close`. With no host UI attached (no visibility callback), the
/// emulator dismisses a running dialog on the second status poll.
pub mod system_dialog {
    use super::{STATUS_FINISHED, STATUS_RUNNING};
    use std::sync::{Mutex, MutexGuard};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Kind {
        Error,
        Signin,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct HostSnapshot {
        pub kind: Kind,
        pub generation: u64,
        pub error_code: i32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct VisualState {
        pub active: bool,
        pub background: bool,
        pub revision: u64,
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct Slot {
        pub status: i32,
        pub generation: u64,
        headless_seen: bool,
    }

    pub(crate) struct Shared {
        pub error: Slot,
        pub signin: Slot,
        generation: u64,
        revision: u64,
        pub error_code: i32,
        callback: Option<fn()>,
    }

    impl Shared {
        pub fn slot(&mut self, kind: Kind) -> &mut Slot {
            match kind {
                Kind::Error => &mut self.error,
                Kind::Signin => &mut self.signin,
            }
        }
    }

    const EMPTY: Slot = Slot { status: 0, generation: 0, headless_seen: false };
    static SHARED: Mutex<Shared> =
        Mutex::new(Shared { error: EMPTY, signin: EMPTY, generation: 0, revision: 0, error_code: 0, callback: None });

    pub(crate) fn lock() -> MutexGuard<'static, Shared> {
        SHARED.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Change a dialog's status; a running/not-running flip bumps the
    /// revision (and the generation when it starts running) and notifies
    /// the host UI after the lock is released.
    pub(crate) fn set_status(mut sh: MutexGuard<'static, Shared>, kind: Kind, status: i32) {
        let old = sh.slot(kind).status;
        let visibility_changed = (old == STATUS_RUNNING) != (status == STATUS_RUNNING);
        sh.slot(kind).status = status;
        if visibility_changed {
            sh.revision += 1;
            if status == STATUS_RUNNING {
                sh.generation += 1;
                let g = sh.generation;
                let s = sh.slot(kind);
                s.generation = g;
                s.headless_seen = false;
            }
        }
        let callback = if visibility_changed { sh.callback } else { None };
        drop(sh);
        if let Some(cb) = callback {
            cb();
        }
    }

    /// Current status of `kind`, letting the headless host dismiss it.
    pub(crate) fn poll(kind: Kind) -> i32 {
        let mut sh = lock();
        let headless = sh.callback.is_none();
        let slot = sh.slot(kind);
        if !headless || slot.status != STATUS_RUNNING {
            return slot.status;
        }
        if !slot.headless_seen {
            slot.headless_seen = true;
            return STATUS_RUNNING;
        }
        log::info!("[SYSTEMDIALOG] no host UI: dismissing the {kind:?} dialog");
        set_status(sh, kind, STATUS_FINISHED);
        STATUS_FINISHED
    }

    pub fn host_snapshot() -> Option<HostSnapshot> {
        let sh = lock();
        if sh.signin.status == STATUS_RUNNING {
            return Some(HostSnapshot { kind: Kind::Signin, generation: sh.signin.generation, error_code: 0 });
        }
        if sh.error.status == STATUS_RUNNING {
            return Some(HostSnapshot { kind: Kind::Error, generation: sh.error.generation, error_code: sh.error_code });
        }
        None
    }

    pub fn visual_state() -> VisualState {
        let sh = lock();
        let signin = sh.signin.status == STATUS_RUNNING;
        VisualState { active: sh.error.status == STATUS_RUNNING || signin, background: signin, revision: sh.revision }
    }

    /// Register the host UI; `None` returns to the headless host.
    pub fn set_visibility_callback(callback: Option<fn()>) {
        lock().callback = callback;
    }

    /// The user dismissed the dialog of `generation`.
    pub fn host_close(generation: u64) -> bool {
        let sh = lock();
        let kind = if sh.signin.status == STATUS_RUNNING { Kind::Signin } else { Kind::Error };
        let slot = match kind {
            Kind::Signin => sh.signin,
            Kind::Error => sh.error,
        };
        if slot.status != STATUS_RUNNING || generation != slot.generation {
            return false;
        }
        set_status(sh, kind, STATUS_FINISHED);
        true
    }
}

// ---------------------------------------------------------------------------
// libSceCommonDialog
// ---------------------------------------------------------------------------

fn cb_common_initialize(_ctx: &mut CpuContext) -> HleResult {
    OK
}

/// sceCommonDialogIsUsed: reported as not in use.
fn cb_common_is_used(_ctx: &mut CpuContext) -> HleResult {
    0
}

// ---------------------------------------------------------------------------
// libSceMsgDialog
// ---------------------------------------------------------------------------

/// SceMsgDialogParam: baseParam[48], size (u64) @0x30, mode @0x38,
/// userMsgParam @0x40, progBarParam @0x48, sysMsgParam @0x50, userId @0x58.
const MSG_PARAM_MODE_OFFSET: u64 = 0x38;
/// SceMsgDialogResult: mode @0, result @4, buttonId @8, reserved[32].
const MSG_RESULT_OK: i32 = 0;
const MSG_BUTTON_ID_OK: i32 = 1;

struct MsgState {
    status: i32,
    mode: i32,
}

static MSG: Mutex<MsgState> = Mutex::new(MsgState { status: STATUS_NONE, mode: 0 });

fn msg() -> std::sync::MutexGuard<'static, MsgState> {
    MSG.lock().unwrap_or_else(|p| p.into_inner())
}

/// The mode/result/buttonId words of SceMsgDialogResult.
pub fn msg_dialog_result(mode: i32) -> [u8; 12] {
    let mut r = [0u8; 12];
    r[0..4].copy_from_slice(&mode.to_le_bytes());
    r[4..8].copy_from_slice(&MSG_RESULT_OK.to_le_bytes());
    r[8..12].copy_from_slice(&MSG_BUTTON_ID_OK.to_le_bytes());
    r
}

fn cb_msg_initialize(_ctx: &mut CpuContext) -> HleResult {
    *msg() = MsgState { status: STATUS_INITIALIZED, mode: 0 };
    OK
}

fn cb_msg_terminate(_ctx: &mut CpuContext) -> HleResult {
    *msg() = MsgState { status: STATUS_NONE, mode: 0 };
    OK
}

/// The message is shown in the log only and counts as acknowledged at once.
fn cb_msg_open(ctx: &mut CpuContext) -> HleResult {
    let param = ctx.rdi();
    if param != 0 {
        log_dialog_text(ctx, param);
    }
    let mut m = msg();
    if param != 0 {
        if let Some(mode) = ctx.read_u32(param + MSG_PARAM_MODE_OFFSET) {
            m.mode = mode as i32;
        }
    }
    m.status = STATUS_FINISHED;
    OK
}

/// Printable ASCII string at `addr` (at least 2 chars), escaped for logs.
fn printable_text(ctx: &CpuContext, addr: u64) -> Option<String> {
    let bytes = ctx.read_bytes(addr, 256)?;
    let end = bytes.iter().position(|&b| b == 0)?;
    if end < 2 {
        return None;
    }
    let s = &bytes[..end];
    if s.iter().any(|&b| !(0x20..=0x7E).contains(&b) && !matches!(b, b'\n' | b'\r' | b'\t')) {
        return None;
    }
    Some(String::from_utf8_lossy(s).replace('\n', "\\n").replace('\r', "\\r"))
}

/// The dialog text is the only trace of fatal-error popups, so log every
/// qword of the param struct that points at text, chasing one nested level.
fn log_dialog_text(ctx: &CpuContext, param: u64) {
    log::info!("[MSGDIALOG] open param={param:#x}");
    fn dump(ctx: &CpuContext, base: u64, bytes: &[u8], label: &str, nested: bool) {
        for off in (0..bytes.len().saturating_sub(7)).step_by(8) {
            let v = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
            if v < 0x10000 || v == base {
                continue;
            }
            if let Some(text) = printable_text(ctx, v) {
                log::info!("[MSGDIALOG]   {label}+{off:#04x} -> {v:#x} text=\"{text}\"");
            } else if nested {
                if let Some(inner) = ctx.read_bytes(v, 0x40) {
                    dump(ctx, v, &inner, &format!("{label}+{off:#04x}"), false);
                }
            }
        }
    }
    if let Some(head) = ctx.read_bytes(param, 0xA0) {
        dump(ctx, param, &head, "param", true);
    }
}

fn cb_msg_get_status(_ctx: &mut CpuContext) -> HleResult {
    msg().status as HleResult
}

fn cb_msg_get_result(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out != 0 {
        let mode = msg().mode;
        ctx.write_bytes(out, &msg_dialog_result(mode));
    }
    OK
}

fn cb_msg_close(_ctx: &mut CpuContext) -> HleResult {
    msg().status = STATUS_FINISHED;
    OK
}

/// No visible progress bar.
fn cb_msg_progress_bar(ctx: &mut CpuContext) -> HleResult {
    log::debug!("[MSGDIALOG] progress bar target={} value={:#x}", ctx.rdi() as i32, ctx.rsi());
    OK
}

// ---------------------------------------------------------------------------
// libSceErrorDialog
// ---------------------------------------------------------------------------

use system_dialog::Kind;

const ERRDLG_NOT_INITIALIZED: HleResult = 0x80ED_0001u32 as i32 as i64;
const ERRDLG_ALREADY_INITIALIZED: HleResult = 0x80ED_0002u32 as i32 as i64;
const ERRDLG_PARAM_INVALID: HleResult = 0x80ED_0003u32 as i32 as i64;
const ERRDLG_INVALID_STATE: HleResult = 0x80ED_0005u32 as i32 as i64;
/// SceErrorDialogParam { i32 size; i32 errorCode; i32 userId; i32 reserved; }.
const ERRDLG_PARAM_SIZE: u32 = 16;

fn cb_err_initialize(_ctx: &mut CpuContext) -> HleResult {
    let mut sh = system_dialog::lock();
    if sh.error.status != STATUS_NONE {
        return ERRDLG_ALREADY_INITIALIZED;
    }
    sh.error.status = STATUS_INITIALIZED;
    OK
}

fn cb_err_open(ctx: &mut CpuContext) -> HleResult {
    let mut sh = system_dialog::lock();
    if sh.error.status != STATUS_INITIALIZED && sh.error.status != STATUS_FINISHED {
        return ERRDLG_INVALID_STATE;
    }
    let param = ctx.rdi();
    if param == 0 {
        return ERRDLG_PARAM_INVALID;
    }
    let Some(raw) = ctx.read_bytes(param, ERRDLG_PARAM_SIZE as usize) else {
        return ERRDLG_PARAM_INVALID;
    };
    let word = |i: usize| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
    if word(0) != ERRDLG_PARAM_SIZE {
        return ERRDLG_PARAM_INVALID;
    }
    log::warn!("[ERRORDIALOG] open errorCode={:#010x} userId={}", word(1), word(2) as i32);
    sh.error_code = word(1) as i32;
    system_dialog::set_status(sh, Kind::Error, STATUS_RUNNING);
    OK
}

fn cb_err_status(_ctx: &mut CpuContext) -> HleResult {
    system_dialog::poll(Kind::Error) as HleResult
}

fn cb_err_close(_ctx: &mut CpuContext) -> HleResult {
    let sh = system_dialog::lock();
    if sh.error.status != STATUS_RUNNING {
        return ERRDLG_INVALID_STATE;
    }
    system_dialog::set_status(sh, Kind::Error, STATUS_FINISHED);
    OK
}

fn cb_err_terminate(_ctx: &mut CpuContext) -> HleResult {
    let sh = system_dialog::lock();
    if sh.error.status == STATUS_NONE {
        return ERRDLG_NOT_INITIALIZED;
    }
    system_dialog::set_status(sh, Kind::Error, STATUS_NONE);
    OK
}

const EXPORTS: &[(&str, &str, &str, crate::registry::HleCallback)] = &[
    ("uoUpLGNkygk", "sceCommonDialogInitialize", "libSceCommonDialog", cb_common_initialize),
    ("BQ3tey0JmQM", "sceCommonDialogIsUsed", "libSceCommonDialog", cb_common_is_used),
    ("lDqxaY1UbEo", "sceMsgDialogInitialize", "libSceMsgDialog", cb_msg_initialize),
    ("ePw-kqZmelo", "sceMsgDialogTerminate", "libSceMsgDialog", cb_msg_terminate),
    ("b06Hh0DPEaE", "sceMsgDialogOpen", "libSceMsgDialog", cb_msg_open),
    ("CWVW78Qc3fI", "sceMsgDialogGetStatus", "libSceMsgDialog", cb_msg_get_status),
    ("6fIC3XKt2k0", "sceMsgDialogUpdateStatus", "libSceMsgDialog", cb_msg_get_status),
    ("Lr8ovHH9l6A", "sceMsgDialogGetResult", "libSceMsgDialog", cb_msg_get_result),
    ("HTrcDKlFKuM", "sceMsgDialogClose", "libSceMsgDialog", cb_msg_close),
    ("wTpfglkmv34", "sceMsgDialogProgressBarSetValue", "libSceMsgDialog", cb_msg_progress_bar),
    ("Gc5k1qcK4fs", "sceMsgDialogProgressBarInc", "libSceMsgDialog", cb_msg_progress_bar),
    ("6H-71OdrpXM", "sceMsgDialogProgressBarSetMsg", "libSceMsgDialog", cb_msg_progress_bar),
    ("I88KChlynSs", "sceErrorDialogInitialize", "libSceErrorDialog", cb_err_initialize),
    ("M2ZF-ClLhgY", "sceErrorDialogOpen", "libSceErrorDialog", cb_err_open),
    ("t2FvHRXzgqk", "sceErrorDialogGetStatus", "libSceErrorDialog", cb_err_status),
    ("WWiGuh9XfgQ", "sceErrorDialogUpdateStatus", "libSceErrorDialog", cb_err_status),
    ("ekXHb1kDBl0", "sceErrorDialogClose", "libSceErrorDialog", cb_err_close),
    ("9XAxK2PMwk8", "sceErrorDialogTerminate", "libSceErrorDialog", cb_err_terminate),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, lib, cb) in EXPORTS {
        reg.register_fn(nid, name, lib, cb);
    }
    crate::web_browser_dialog::register(reg);
}

/// Serializes unit tests that drive the shared system-dialog state.
#[cfg(test)]
pub(crate) static SYSTEM_DIALOG_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    #[test]
    fn dialog_lifecycle() {
        let d = DialogState::new();
        assert_eq!(d.terminate(), ERR_NOT_INITIALIZED);
        assert_eq!(d.initialize(), OK);
        assert_eq!(d.get(), STATUS_INITIALIZED);
        assert_eq!(d.close(), ERR_NOT_RUNNING);
        d.set(STATUS_RUNNING);
        d.initialize(); // must not clobber RUNNING
        assert_eq!(d.get(), STATUS_RUNNING);
        assert_eq!(d.poll(), STATUS_FINISHED);
        assert_eq!(d.poll(), STATUS_FINISHED);
        d.set(STATUS_RUNNING);
        assert_eq!(d.close(), OK);
        assert_eq!(d.terminate(), OK);
        assert_eq!(d.get(), STATUS_NONE);
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn msg_dialog_finishes_on_open() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D33_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        assert_eq!(cb_msg_initialize(&mut ctx), OK);
        assert_eq!(cb_msg_get_status(&mut ctx), STATUS_INITIALIZED as i64);
        ctx.write_u32(BASE + MSG_PARAM_MODE_OFFSET, 2);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_msg_open(&mut ctx), OK);
        assert_eq!(cb_msg_get_status(&mut ctx), STATUS_FINISHED as i64);
        ctx.set(Rdi, BASE + 0x100);
        assert_eq!(cb_msg_get_result(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE + 0x100), Some(2));
        assert_eq!(ctx.read_u32(BASE + 0x104), Some(0));
        assert_eq!(ctx.read_u32(BASE + 0x108), Some(1));
        assert_eq!(cb_msg_terminate(&mut ctx), OK);
        assert_eq!(cb_msg_get_status(&mut ctx), STATUS_NONE as i64);
    }

    fn noop() {}

    #[test]
    fn error_dialog_state_machine() {
        let _g = SYSTEM_DIALOG_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D34_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        system_dialog::set_visibility_callback(Some(noop));
        ctx.set(Rdi, BASE);
        assert_eq!(cb_err_terminate(&mut ctx), ERRDLG_NOT_INITIALIZED);
        assert_eq!(cb_err_open(&mut ctx), ERRDLG_INVALID_STATE);
        assert_eq!(cb_err_initialize(&mut ctx), OK);
        assert_eq!(cb_err_initialize(&mut ctx), ERRDLG_ALREADY_INITIALIZED);
        assert_eq!(cb_err_close(&mut ctx), ERRDLG_INVALID_STATE);
        ctx.write_u32(BASE, 12);
        assert_eq!(cb_err_open(&mut ctx), ERRDLG_PARAM_INVALID);
        ctx.write_u32(BASE, 16);
        ctx.write_u32(BASE + 4, 0x8002_0001);
        assert_eq!(cb_err_open(&mut ctx), OK);
        assert_eq!(cb_err_open(&mut ctx), ERRDLG_INVALID_STATE);
        // A host UI is attached: the dialog stays up until it is closed.
        assert_eq!(cb_err_status(&mut ctx), STATUS_RUNNING as i64);
        assert_eq!(cb_err_status(&mut ctx), STATUS_RUNNING as i64);
        let snap = system_dialog::host_snapshot().unwrap();
        assert_eq!((snap.kind, snap.error_code), (Kind::Error, 0x8002_0001u32 as i32));
        assert!(system_dialog::visual_state().active);
        assert!(!system_dialog::host_close(snap.generation + 1));
        assert!(system_dialog::host_close(snap.generation));
        assert_eq!(cb_err_status(&mut ctx), STATUS_FINISHED as i64);
        assert!(system_dialog::host_snapshot().is_none());

        // Headless: dismissed on the second poll.
        system_dialog::set_visibility_callback(None);
        assert_eq!(cb_err_open(&mut ctx), OK);
        assert_eq!(cb_err_status(&mut ctx), STATUS_RUNNING as i64);
        assert_eq!(cb_err_status(&mut ctx), STATUS_FINISHED as i64);
        assert_eq!(cb_err_open(&mut ctx), OK);
        assert_eq!(cb_err_close(&mut ctx), OK);
        assert_eq!(cb_err_terminate(&mut ctx), OK);
        assert_eq!(cb_err_status(&mut ctx), STATUS_NONE as i64);
    }
}
