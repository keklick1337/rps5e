// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceSystemService HLE: system parameters, status, display info and
//! the notice-screen preference.

use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

const LIB: &str = "libSceSystemService";
const OK: HleResult = 0;
pub const ERR_PARAMETER: HleResult = 0x80A1_0003u32 as i32 as i64;
pub const ERR_NO_EVENT: HleResult = 0x80A1_0004u32 as i32 as i64;

/// SceSystemServiceStatus: i32 eventNum, bool isSystemUiOverlaid,
/// bool isInBackgroundExecution, bool isVrPlayAreaOverlaid, 127 reserved
/// bytes (136 with tail padding).
const STATUS_SIZE: usize = 136;
/// float ratio + 128 reserved bytes.
const DISPLAY_SAFE_AREA_INFO_SIZE: usize = 4 + 128;
const TITLE_ID_FIELD_SIZE: usize = 0x10;
/// SceSystemServiceEvent: i32 eventType + 8192 bytes payload.
const EVENT_DATA_SIZE: usize = 8192;
/// System name reported for SCE_SYSTEM_SERVICE_PARAM_ID_SYSTEM_NAME.
pub const DEFAULT_SYSTEM_NAME: &str = "rps5e";

pub const PARAM_ID_LANG: i32 = 1;
pub const PARAM_ID_DATE_FORMAT: i32 = 2;
pub const PARAM_ID_TIME_FORMAT: i32 = 3;
pub const PARAM_ID_TIME_ZONE: i32 = 4;
pub const PARAM_ID_SUMMERTIME: i32 = 5;
pub const PARAM_ID_SYSTEM_NAME: i32 = 6;
pub const PARAM_ID_GAME_PARENTAL_LEVEL: i32 = 7;
pub const PARAM_ID_CC_ENABLE: i32 = 100;
pub const PARAM_ID_SCREEN_READER: i32 = 208;
/// System language as queried by PS5 titles.
pub const PARAM_ID_LANG_PS5: i32 = 400;
pub const PARAM_ID_ENTER_BUTTON_ASSIGN: i32 = 1000;

const PARAM_DATE_FORMAT_DDMMYYYY: i32 = 1;
const PARAM_TIME_FORMAT_24HOUR: i32 = 1;
const PARAM_GAME_PARENTAL_OFF: i32 = 0;
const PARAM_CC_DISABLED: i32 = 0;
const PARAM_ENTER_BUTTON_ASSIGN_CROSS: i32 = 1;
/// Console time zone of a host UTC offset (seconds) and DST flag: the
/// zone's standard offset in minutes (summer time excluded) and the summer
/// time flag, as the console stores them.
fn time_zone_params(offset_secs: i64, summer: bool) -> (i32, i32) {
    ((offset_secs / 60) as i32 - if summer { 60 } else { 0 }, summer as i32)
}

/// The host's time zone now (UTC when it is unknown).
fn host_time_zone() -> (i32, i32) {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64);
    let (offset, summer) = crate::rtc::host_local_offset(now).unwrap_or((0, false));
    time_zone_params(offset, summer)
}

/// SCE_SYSTEM_SERVICE_EVENT_ENTITLEMENT_UPDATE.
pub const EVENT_ENTITLEMENT_UPDATE: i32 = 0x1000_0003;

/// Pending system events, delivered in order by sceSystemServiceReceiveEvent
/// and counted by sceSystemServiceGetStatus.
static EVENTS: Mutex<VecDeque<i32>> = Mutex::new(VecDeque::new());

fn events() -> std::sync::MutexGuard<'static, VecDeque<i32>> {
    EVENTS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Queue an entitlement update (sent once AppContent is initialized).
pub fn notify_entitlement_update() {
    events().push_back(EVENT_ENTITLEMENT_UPDATE);
}

static SPLASH_HIDDEN: AtomicBool = AtomicBool::new(false);

/// True once the title asked to hide the boot splash screen.
pub fn splash_screen_hidden() -> bool {
    SPLASH_HIDDEN.load(Ordering::Relaxed)
}

fn write_or_fault(ctx: &CpuContext, addr: u64, data: &[u8]) -> HleResult {
    if ctx.write_bytes(addr, data) { OK } else { ERR_PARAMETER }
}

/// The skip-notice-screen setting of the console (`RPS5E_SKIP_NOTICE_SCREEN=1`,
/// off by default). Titles read it; their own Set call does not change it.
pub fn skip_notice_screen() -> bool {
    std::env::var("RPS5E_SKIP_NOTICE_SCREEN")
        .is_ok_and(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
}

/// Integer system parameter values (language, date/time formats, ...).
/// Unknown ids read as 0.
pub fn param_int_value(param_id: i32, language: i32) -> i32 {
    match param_id {
        PARAM_ID_LANG | PARAM_ID_LANG_PS5 => language,
        PARAM_ID_DATE_FORMAT => PARAM_DATE_FORMAT_DDMMYYYY,
        PARAM_ID_TIME_FORMAT => PARAM_TIME_FORMAT_24HOUR,
        PARAM_ID_TIME_ZONE => host_time_zone().0,
        PARAM_ID_SUMMERTIME => host_time_zone().1,
        PARAM_ID_GAME_PARENTAL_LEVEL => PARAM_GAME_PARENTAL_OFF,
        PARAM_ID_CC_ENABLE => PARAM_CC_DISABLED,
        PARAM_ID_SCREEN_READER => 0,
        PARAM_ID_ENTER_BUTTON_ASSIGN => PARAM_ENTER_BUTTON_ASSIGN_CROSS,
        _ => {
            log::warn!("[SystemService] unsupported int param_id {param_id}, using 0");
            0
        }
    }
}

/// String system parameter values: only the system name exists.
pub fn param_string_value(param_id: i32) -> Option<&'static str> {
    (param_id == PARAM_ID_SYSTEM_NAME).then_some(DEFAULT_SYSTEM_NAME)
}

fn cb_ok(_ctx: &mut CpuContext) -> HleResult {
    OK
}

fn cb_get_notice_screen_skip_flag(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_PARAMETER;
    }
    write_or_fault(ctx, out, &[skip_notice_screen() as u8])
}

fn cb_get_main_app_title_id(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_PARAMETER;
    }
    let title = crate::app_info::title_id().unwrap_or_else(|| crate::app_info::FALLBACK_TITLE_ID.into());
    // Fixed 0x10-byte field; bound the copy so a malformed id cannot overrun.
    let bytes = title.as_bytes();
    let len = bytes.len().min(TITLE_ID_FIELD_SIZE - 1);
    let mut buf = bytes[..len].to_vec();
    buf.push(0);
    write_or_fault(ctx, out, &buf)
}

fn cb_param_get_int(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let out = ctx.rsi();
    if out == 0 {
        return ERR_PARAMETER;
    }
    let v = param_int_value(id, crate::app_info::system_language());
    write_or_fault(ctx, out, &v.to_le_bytes())
}

/// Room sceSystemServiceParamGetString needs for the system name.
const MAX_SYSTEM_NAME_LENGTH: u64 = 65;

/// sceSystemServiceParamGetString(paramId, char* buf, size_t bufSize): only
/// the system name; any other id, a null buffer or one shorter than 65
/// bytes is PARAMETER and leaves the buffer alone; the name is written
/// with the rest of the buffer zeroed (measured on a PS5).
fn cb_param_get_string(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let buf = ctx.rsi();
    let size = ctx.rdx();
    let value = param_string_value(id);
    let Some(value) = value.filter(|_| buf != 0 && size >= MAX_SYSTEM_NAME_LENGTH) else {
        return ERR_PARAMETER;
    };
    let mut out = vec![0u8; size as usize];
    let n = value.len().min(MAX_SYSTEM_NAME_LENGTH as usize - 1);
    out[..n].copy_from_slice(&value.as_bytes()[..n]);
    write_or_fault(ctx, buf, &out)
}

fn cb_get_status(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_PARAMETER;
    }
    let mut status = [0u8; STATUS_SIZE];
    let (overlaid, background) = system_ui_state();
    status[4] = overlaid as u8;
    status[5] = background as u8;
    status[0..4].copy_from_slice(&(events().len() as i32).to_le_bytes());
    write_or_fault(ctx, out, &status)
}

/// (isSystemUiOverlaid, isInBackgroundExecution): a system dialog on screen
/// overlays the title.
fn system_ui_state() -> (bool, bool) {
    let dialog = crate::common_dialog::system_dialog::visual_state();
    (dialog.active, dialog.background)
}

fn cb_get_display_safe_area_info(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_PARAMETER;
    }
    let mut info = [0u8; DISPLAY_SAFE_AREA_INFO_SIZE];
    info[..4].copy_from_slice(&1.0f32.to_le_bytes());
    write_or_fault(ctx, out, &info)
}

fn cb_get_hdr_tone_map_luminance(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_PARAMETER;
    }
    // maxFullFrameToneMapLuminance, maxToneMapLuminance, minToneMapLuminance.
    let mut buf = [0u8; 12];
    buf[0..4].copy_from_slice(&80.0f32.to_le_bytes());
    buf[4..8].copy_from_slice(&1000.0f32.to_le_bytes());
    buf[8..12].copy_from_slice(&0.0f32.to_le_bytes());
    write_or_fault(ctx, out, &buf)
}

fn cb_hide_splash_screen(_ctx: &mut CpuContext) -> HleResult {
    SPLASH_HIDDEN.store(true, Ordering::Relaxed);
    OK
}

/// Pops the oldest pending event; the payload is always cleared. The only
/// event generated is the entitlement update queued by AppContent init (no
/// suspend/resume, no URL launch).
fn cb_receive_event(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_PARAMETER;
    }
    let mut ev = vec![0u8; 4 + EVENT_DATA_SIZE];
    ev[..4].copy_from_slice(&(-1i32).to_le_bytes());
    if !ctx.write_bytes(out, &ev) {
        return ERR_PARAMETER;
    }
    let mut q = events();
    let Some(&kind) = q.front() else { return ERR_NO_EVENT };
    if !ctx.write_bytes(out, &kind.to_le_bytes()) {
        return ERR_PARAMETER;
    }
    q.pop_front();
    log::debug!("[SystemService] event {:#010x}", kind as u32);
    OK
}

/// Kernel ENOTSUP: replacing the running executable is not supported.
const ERR_NOT_SUPPORTED: HleResult = 0x8002_002Du32 as i32 as i64;
/// Longest LoadExec path examined.
const MAX_EXEC_PATH: usize = 1024;

/// NUL-terminated guest string, read byte by byte (None on fault/overlong).
fn read_short_cstring(ctx: &CpuContext, addr: u64, max: usize) -> Option<String> {
    let mut bytes = Vec::new();
    for i in 0..max as u64 {
        let b = ctx.read_bytes(addr + i, 1)?;
        if b[0] == 0 {
            return Some(String::from_utf8_lossy(&bytes).into_owned());
        }
        bytes.push(b[0]);
    }
    None
}

/// SCE_SYSTEM_SERVICE_ERROR_UNAVAILABLE
const ERR_UNAVAILABLE: HleResult = 0x80A1_0002u32 as i32 as i64;
/// Bytes of argument strings (with their terminators) LoadExec accepts.
const MAX_EXEC_ARGS_BYTES: usize = 4096;

/// sceSystemServiceLoadExec(const char* path, char* const argv[]): the
/// special path "exit" terminates the application. Otherwise the target
/// must be an existing file under /app0/ and the NULL-terminated argv
/// strings must fit in 4096 bytes; replacing the running executable is not
/// available, so a valid request fails with UNAVAILABLE.
fn cb_load_exec(ctx: &mut CpuContext) -> HleResult {
    let (path_addr, argv) = (ctx.rdi(), ctx.rsi());
    if path_addr == 0 {
        return ERR_PARAMETER;
    }
    let Some(path) = read_short_cstring(ctx, path_addr, MAX_EXEC_PATH) else {
        return ERR_PARAMETER;
    };
    if path == "exit" {
        log::info!("[SYSTEMSERVICE] LoadExec(\"exit\"): application exit");
        ps5vm::shutdown::exit(0);
    }
    let exists = || crate::filesystem::resolve_guest_path(&path).is_some_and(|p| p.is_file());
    if !path.starts_with("/app0/") || !exists() {
        return ERR_PARAMETER;
    }
    let mut bytes = 0usize;
    if argv != 0 {
        for i in 0.. {
            let Some(arg) = ctx.read_u64(argv + i * 8) else {
                return ERR_PARAMETER;
            };
            if arg == 0 {
                break;
            }
            // The string and its terminator must fit in what is left.
            let left = MAX_EXEC_ARGS_BYTES - bytes;
            let len = (0..left as u64).find(|&j| ctx.read_bytes(arg + j, 1).is_none_or(|b| b[0] == 0));
            match len {
                Some(n) if ctx.read_bytes(arg + n, 1).is_some() => bytes += n as usize + 1,
                _ => return ERR_PARAMETER,
            }
        }
    }
    log::error!("[SYSTEMSERVICE] LoadExec(\"{path}\"): launching another executable is not available");
    ERR_UNAVAILABLE
}

/// sceSystemServiceLaunchWebBrowser(const char* uri, void* param): there is
/// no web browser to open, whatever the arguments; the param is not touched.
fn cb_launch_web_browser(_ctx: &mut CpuContext) -> HleResult {
    log::info!("[SYSTEMSERVICE] web browser requested (no browser)");
    ERR_NOT_SUPPORTED
}

/// sceSystemServiceInitializePlayerDialogParam(SceSystemServicePlayerDialogParam*):
/// { u64 size; i32 mode; i32 userId; u64 targetAccountId; u8 reserved[40] }
/// filled as the console does: zeros, size 0x40, user id -1.
fn cb_initialize_player_dialog_param(ctx: &mut CpuContext) -> HleResult {
    const PARAM_SIZE: usize = 0x40;
    let param = ctx.rdi();
    if param == 0 {
        return ERR_PARAMETER;
    }
    let mut b = [0u8; PARAM_SIZE];
    b[0..8].copy_from_slice(&(PARAM_SIZE as u64).to_le_bytes());
    b[0xC..0x10].copy_from_slice(&(-1i32).to_le_bytes());
    write_or_fault(ctx, param, &b)
}

/// sceSystemServiceLaunchPlayerDialog(const SceSystemServicePlayerDialogParam*):
/// the system profile overlay is fire-and-forget; there is no system UI to
/// show it on, so the request is only logged.
fn cb_launch_player_dialog(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return ERR_PARAMETER;
    }
    log::info!("[SYSTEMSERVICE] player dialog requested (no system UI)");
    OK
}

/// sceSystemServiceShowControllerSettings(userId): system overlay, logged only.
fn cb_show_controller_settings(_ctx: &mut CpuContext) -> HleResult {
    log::info!("[SYSTEMSERVICE] controller settings requested (no system UI)");
    OK
}

/// sceSystemServiceOpenChallengeActivity(): the challenge screen is a shell
/// page, and there is no shell UI to show it on. The request fails with the
/// same not-supported error as the web browser, so the title can tell that
/// the screen did not open.
fn cb_open_challenge_activity(_ctx: &mut CpuContext) -> HleResult {
    log::info!("[SYSTEMSERVICE] challenge activity requested (no system UI)");
    ERR_NOT_SUPPORTED
}

/// sceSystemServiceOpenTournamentOccurrence(): a shell page, as above.
fn cb_open_tournament_occurrence(_ctx: &mut CpuContext) -> HleResult {
    log::info!("[SYSTEMSERVICE] tournament occurrence requested (no system UI)");
    ERR_NOT_SUPPORTED
}

/// App ID of the running title. The shell assigns app IDs and their values
/// are not known; every call that takes an app ID uses this one.
const RUNNING_APP_ID: i32 = 1;

/// sceSystemServiceGetAppIdOfRunningBigApp()
fn cb_get_app_id_of_running_big_app(_ctx: &mut CpuContext) -> HleResult {
    RUNNING_APP_ID as HleResult
}

/// The one sceSystemServiceKillApp request that is supported: the running
/// title closing itself with how -1, reason 0 and no core dump.
fn kill_app_supported(app_id: i32, how: i32, reason: i32, core_dump: i32) -> Result<(), &'static str> {
    if app_id != RUNNING_APP_ID {
        return Err("application other than the running title");
    }
    if how != -1 || reason != 0 || core_dump != 0 {
        return Err("arguments other than -1, 0 and 0");
    }
    Ok(())
}

/// sceSystemServiceKillApp(appId, how, reason, coreDump): the running title
/// closes itself. On the console the shell ends the title after the call
/// returns; with no shell the emulator shuts down as for the title's own
/// exit(0), so the call does not return.
fn cb_kill_app(ctx: &mut CpuContext) -> HleResult {
    let (app_id, how, reason, core_dump) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx() as i32, ctx.rcx() as i32);
    if let Err(what) = kill_app_supported(app_id, how, reason, core_dump) {
        log::error!("[SYSTEMSERVICE] sceSystemServiceKillApp({app_id}, {how}, {reason}, {core_dump}): {what} is not supported");
        return ERR_PARAMETER;
    }
    log::info!("[SYSTEMSERVICE] sceSystemServiceKillApp: the title closes itself");
    ps5vm::shutdown::exit(0)
}

/// sceLncUtilKillApp(appId): the running title closing itself, as
/// sceSystemServiceKillApp.
fn cb_lnc_kill_app(ctx: &mut CpuContext) -> HleResult {
    lnc_kill(ctx.rdi() as u32 as i32, 0, "sceLncUtilKillApp")
}

/// sceLncUtilKillAppWithReason(appId, reason): reason 0 only.
fn cb_lnc_kill_app_with_reason(ctx: &mut CpuContext) -> HleResult {
    lnc_kill(ctx.rdi() as u32 as i32, ctx.rsi() as i32, "sceLncUtilKillAppWithReason")
}

fn lnc_kill(app_id: i32, reason: i32, function: &str) -> HleResult {
    if let Err(what) = kill_app_supported(app_id, -1, reason, 0) {
        log::error!("[SYSTEMSERVICE] {function}({app_id}, {reason}): {what} is not supported");
        return ERR_PARAMETER;
    }
    log::info!("[SYSTEMSERVICE] {function}: the title closes itself");
    ps5vm::shutdown::exit(0)
}

/// SCE_SYSTEM_SERVICE_ERROR_REJECTED
const ERR_REJECTED: HleResult = 0x80A1_0005u32 as i32 as i64;

fn cb_unavailable(_ctx: &mut CpuContext) -> HleResult {
    ERR_UNAVAILABLE
}

fn cb_rejected(_ctx: &mut CpuContext) -> HleResult {
    ERR_REJECTED
}

/// Calls the firmware answers without doing anything a title can see:
/// clock and activation requests succeed, the frame buffer and voice
/// controls are unavailable, launching apps or changing the rendering mode
/// is rejected.
const FIRMWARE_ANSWERS: &[(&str, &str, crate::registry::HleCallback)] = &[
    ("mpkohyVqCRM", "sceShellCoreUtilActivateAbort", cb_ok),
    ("xIMClZZz50k", "sceShellCoreUtilActivateRecordActivation", cb_ok),
    ("MRVnLsn-GRI", "sceShellCoreUtilActivateStart", cb_ok),
    ("CZrOHqt6oCY", "sceShellCoreUtilActivateStartAsync", cb_ok),
    ("ibXh+Mc4wbs", "sceShellCoreUtilActivateTerm", cb_ok),
    ("FI+VqGdttvI", "sceSystemServiceChangeAcpClock", cb_ok),
    ("ec72vt3WEQo", "sceSystemServiceChangeCpuClock", cb_ok),
    ("Z5RgV4Chwxg", "sceSystemServiceChangeGpuClock", cb_ok),
    ("LFo00RWzqRU", "sceSystemServiceChangeMemoryClock", cb_ok),
    ("MyBXslDE+2o", "sceSystemServiceChangeMemoryClockToBaseMode", cb_ok),
    ("qv+X8gozqF4", "sceSystemServiceChangeMemoryClockToDefault", cb_ok),
    ("fOsE5pTieqY", "sceSystemServiceChangeMemoryClockToMultiMediaMode", cb_ok),
    ("5MLppFJZyX4", "sceSystemServiceChangeNumberOfGpuCu", cb_ok),
    ("lgTlIAEJ33M", "sceSystemServiceChangeSamuClock", cb_ok),
    ("BQUi7AW+2tQ", "sceSystemServiceChangeUvdClock", cb_ok),
    ("fzguXBQzNvI", "sceSystemServiceChangeVceClock", cb_ok),
    ("sgRPNJjrWjg", "sceSystemServiceSetOutOfVrPlayZoneWarning", cb_ok),
    ("tPfQU2pD4-M", "sceSystemServiceShowDisplaySafeAreaSettings", cb_ok),
    ("0z7srulNt7U", "sceSystemServiceAcquireFb0", cb_unavailable),
    ("Oms065qIClY", "sceSystemServiceReleaseFb0", cb_unavailable),
    ("45QrFvUkrjg", "sceSystemServiceDisablePartyVoice", cb_unavailable),
    ("hU3bSlF2OKs", "sceSystemServiceReenablePartyVoice", cb_unavailable),
    ("d3OnoKtNjGg", "sceSystemServiceDisableVoiceRecognition", cb_unavailable),
    ("c-aFKhn74h0", "sceSystemServiceReenableVoiceRecognition", cb_unavailable),
    ("AmTvo3RT5ss", "sceSystemServiceLoadExecVideoServiceWebApp", cb_unavailable),
    ("YNoDjc1BPJI", "sceSystemServiceLaunchUdsApp", cb_rejected),
    ("P71fvnHyFTQ", "sceSystemServiceRequestToChangeRenderingMode", cb_rejected),
    ("SZ2uH5Abws8", "sceLncUtilKillApp", cb_lnc_kill_app),
    ("RBlEzB--JeA", "sceLncUtilKillAppWithReason", cb_lnc_kill_app_with_reason),
];

const NEW_EXPORTS: &[(&str, &str, crate::registry::HleCallback)] = &[
    ("JoBqSQt1yyA", "sceSystemServiceLoadExec", cb_load_exec),
    ("m5CYKX20wfg", "sceSystemServiceInitializePlayerDialogParam", cb_initialize_player_dialog_param),
    ("uaieF+glFPs", "sceSystemServiceLaunchPlayerDialog", cb_launch_player_dialog),
    ("-+3hY+y8bNo", "sceSystemServiceLaunchWebBrowser", cb_launch_web_browser),
    ("w9wlKcHrmm8", "sceSystemServiceShowControllerSettings", cb_show_controller_settings),
    ("sPuK5ic3GD4", "sceSystemServiceOpenChallengeActivity", cb_open_challenge_activity),
    ("gELp9ue2ccQ", "sceSystemServiceOpenTournamentOccurrence", cb_open_tournament_occurrence),
    ("NwT2R5lo8P8", "sceSystemServiceGetAppIdOfRunningBigApp", cb_get_app_id_of_running_big_app),
    ("N4RkyJh7FtA", "sceSystemServiceKillApp", cb_kill_app),
    // Media playback restrictions have no effect: there is no system media player.
    ("64nkF7LGk8w", "sceSystemServiceDisableMediaPlay", cb_ok),
    ("uInYhy7xa+U", "sceSystemServiceReenableMediaPlay", cb_ok),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in NEW_EXPORTS.iter().chain(FIRMWARE_ANSWERS) {
        reg.register_fn(nid, name, LIB, cb);
    }
    reg.register_fn("x1UB9bwDSOw", "sceSystemServiceDisableMusicPlayer", LIB, cb_ok);
    reg.register_fn("9kPCz7Or+1Y", "sceSystemServiceReenableMusicPlayer", LIB, cb_ok);
    reg.register_fn("3RQ5aQfnstU", "sceSystemServiceGetNoticeScreenSkipFlag", LIB, cb_get_notice_screen_skip_flag);
    reg.register_fn("8Lo6Zv94aho", "sceSystemServiceDisableNoticeScreenSkipFlagAutoSet", LIB, cb_ok);
    reg.register_fn("Q3utJvma4Mo", "sceSystemServiceSetNoticeScreenSkipFlag", LIB, cb_ok);
    reg.register_fn("4veE0XiIugA", "sceSystemServiceGetMainAppTitleId", LIB, cb_get_main_app_title_id);
    reg.register_fn("fZo48un7LK4", "sceSystemServiceParamGetInt", LIB, cb_param_get_int);
    reg.register_fn("SsC-m-S9JTA", "sceSystemServiceParamGetString", LIB, cb_param_get_string);
    reg.register_fn("rPo6tV8D9bM", "sceSystemServiceGetStatus", LIB, cb_get_status);
    reg.register_fn("1n37q1Bvc5Y", "sceSystemServiceGetDisplaySafeAreaInfo", LIB, cb_get_display_safe_area_info);
    reg.register_fn("mPpPxv5CZt4", "sceSystemServiceGetHdrToneMapLuminance", LIB, cb_get_hdr_tone_map_luminance);
    reg.register_fn("Vo5V8KAwCmk", "sceSystemServiceHideSplashScreen", LIB, cb_hide_splash_screen);
    reg.register_fn("3s8cHiCBKBE", "sceSystemServiceReportAbnormalTermination", LIB, cb_ok);
    reg.register_fn("656LMQSrg6U", "sceSystemServiceReceiveEvent", LIB, cb_receive_event);
    reg.register_fn("XbbJC3E+L5M", "sceSystemServicePowerTick", LIB, cb_ok);
    crate::system_gesture::register(reg);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sceSystemServiceParamGetString (reference GuestSystemService case
    /// measured on a PS5).
    #[test]
    fn param_get_string_errors_and_padding() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A6E_0000_0000;
        let base = BASE;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let call = |ctx: &mut CpuContext, id: i32, buf: u64, size: u64| {
            ctx.set(crate::context::CpuRegister::Rdi, id as u32 as u64);
            ctx.set(crate::context::CpuRegister::Rsi, buf);
            ctx.set(crate::context::CpuRegister::Rdx, size);
            cb_param_get_string(ctx)
        };
        ctx.write_bytes(base, &[b'x'; 128]);
        assert_eq!(call(&mut ctx, PARAM_ID_SYSTEM_NAME, 0, 65), ERR_PARAMETER);
        for size in 0..65 {
            assert_eq!(call(&mut ctx, PARAM_ID_SYSTEM_NAME, base, size), ERR_PARAMETER);
        }
        for id in [-1, 0, 1, 5, 7, 40, 100, 1000, i32::MAX] {
            if id != PARAM_ID_SYSTEM_NAME {
                assert_eq!(call(&mut ctx, id, base, 128), ERR_PARAMETER);
            }
        }
        assert_eq!(ctx.read_bytes(base, 128).unwrap(), vec![b'x'; 128]);
        assert_eq!(call(&mut ctx, PARAM_ID_SYSTEM_NAME, base, 68), OK);
        let name = ctx.read_cstring(base, 68).unwrap();
        assert_eq!(name, DEFAULT_SYSTEM_NAME);
        assert!(ctx.read_bytes(base + name.len() as u64, 68 - name.len()).unwrap().iter().all(|&b| b == 0));
        assert_eq!(ctx.read_bytes(base + 68, 60).unwrap(), vec![b'x'; 60]);
    }

    /// The player dialog param as the console fills it, and the calls
    /// answered with the firmware's fixed results.
    #[test]
    fn player_dialog_param_and_firmware_answers() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A6F_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, &[0xAA; 0x48]);
        ctx.set(crate::context::CpuRegister::Rdi, BASE);
        assert_eq!(cb_initialize_player_dialog_param(&mut ctx), OK);
        assert_eq!(ctx.read_u64(BASE), Some(0x40));
        assert_eq!(ctx.read_u32(BASE + 8), Some(0));
        assert_eq!(ctx.read_u32(BASE + 0xC), Some(u32::MAX));
        assert!(ctx.read_bytes(BASE + 0x10, 0x30).unwrap().iter().all(|&b| b == 0));
        assert_eq!(ctx.read_bytes(BASE + 0x40, 8).unwrap(), vec![0xAA; 8]);
        ctx.set(crate::context::CpuRegister::Rdi, 0);
        assert_eq!(cb_initialize_player_dialog_param(&mut ctx), ERR_PARAMETER);
        let answer = |name: &str| (FIRMWARE_ANSWERS.iter().find(|e| e.1 == name).unwrap().2)(&mut CpuContext::new(&vm));
        assert_eq!(answer("sceSystemServiceChangeGpuClock"), OK);
        assert_eq!(answer("sceSystemServiceAcquireFb0"), 0x80A1_0002u32 as i32 as i64);
        assert_eq!(answer("sceSystemServiceLaunchUdsApp"), 0x80A1_0005u32 as i32 as i64);
        let mut ctx = CpuContext::new(&vm);
        ctx.set(crate::context::CpuRegister::Rdi, 7);
        assert_eq!(cb_lnc_kill_app(&mut ctx), ERR_PARAMETER, "another app");
        ctx.set(crate::context::CpuRegister::Rdi, RUNNING_APP_ID as u64);
        ctx.set(crate::context::CpuRegister::Rsi, 3);
        assert_eq!(cb_lnc_kill_app_with_reason(&mut ctx), ERR_PARAMETER, "a reason");
    }

    #[test]
    fn new_nids_match_names() {
        for &(nid, name, _) in NEW_EXPORTS.iter().chain(FIRMWARE_ANSWERS) {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// The running title's app ID is stable and positive; KillApp closes
    /// only that title, and only with how -1, reason 0 and no core dump.
    #[test]
    fn kill_app_accepts_only_the_running_title() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        let app = cb_get_app_id_of_running_big_app(&mut ctx) as i32;
        assert!(app > 0);
        assert_eq!(cb_get_app_id_of_running_big_app(&mut ctx) as i32, app);
        assert_eq!(kill_app_supported(app, -1, 0, 0), Ok(()));
        for other in [app + 1, -1, 0] {
            assert_eq!(kill_app_supported(other, -1, 0, 0), Err("application other than the running title"));
        }
        for (how, reason, core_dump) in [(0, 0, 0), (-2, 0, 0), (-1, 1, 0), (-1, 0, 1)] {
            assert_eq!(kill_app_supported(app, how, reason, core_dump), Err("arguments other than -1, 0 and 0"));
        }
        use crate::context::CpuRegister::*;
        for (r, v) in [(Rdi, app as u64 + 1), (Rsi, u64::MAX), (Rdx, 0), (Rcx, 0)] {
            ctx.set(r, v);
        }
        assert_eq!(cb_kill_app(&mut ctx), ERR_PARAMETER);
    }

    #[test]
    fn media_play_restrictions_are_accepted() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        for name in ["sceSystemServiceDisableMediaPlay", "sceSystemServiceReenableMediaPlay"] {
            let cb = NEW_EXPORTS.iter().find(|e| e.1 == name).unwrap().2;
            assert_eq!(cb(&mut ctx), OK);
            assert_eq!(cb(&mut ctx), OK);
        }
    }

    /// The challenge and tournament screens are shell pages: their requests
    /// fail with the not-supported error, like the web browser.
    #[test]
    fn shell_screens_are_unavailable() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        for name in ["sceSystemServiceOpenChallengeActivity", "sceSystemServiceOpenTournamentOccurrence"] {
            let cb = NEW_EXPORTS.iter().find(|e| e.1 == name).unwrap().2;
            assert_eq!(cb(&mut ctx), ERR_NOT_SUPPORTED, "{name}");
        }
    }

    #[test]
    fn load_exec_rejects_other_executables() {
        use crate::context::CpuRegister::*;
        let root = crate::filesystem::tests::temp_root("load-exec");
        std::fs::write(root.join("other.elf"), b"ELF").unwrap();
        std::fs::create_dir_all(root.join("dir")).unwrap();
        let _app0 = crate::filesystem::APP0_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        crate::filesystem::set_app0_root(&root);

        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A6D_0000_0000;
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (path, argv, arg) = (BASE, BASE + 0x400, BASE + 0x800);
        ctx.set(Rdi, path);
        ctx.set(Rsi, 0);
        // Only existing files under /app0/ are accepted.
        for bad in [&b"\0"[..], b"/app0/missing.elf\0", b"/app0/dir\0", b"other.elf\0", b"/data/other.elf\0"] {
            ctx.write_bytes(path, bad);
            assert_eq!(cb_load_exec(&mut ctx), ERR_PARAMETER, "{:?}", String::from_utf8_lossy(bad));
        }
        ctx.write_bytes(path, b"/app0/other.elf\0");
        assert_eq!(cb_load_exec(&mut ctx), ERR_UNAVAILABLE);

        // argv strings with their terminators may take up to 4096 bytes.
        ctx.write_bytes(arg, &[b'a'; 0x1000]);
        ctx.write_bytes(arg + 0xFFF, &[0]);
        ctx.write_u64(argv, arg);
        ctx.write_u64(argv + 8, 0);
        ctx.set(Rsi, argv);
        assert_eq!(cb_load_exec(&mut ctx), ERR_UNAVAILABLE);
        ctx.write_u64(argv + 8, arg + 0xFFF);
        ctx.write_u64(argv + 16, 0);
        assert_eq!(cb_load_exec(&mut ctx), ERR_PARAMETER, "empty string past the limit");
        ctx.write_u64(argv + 8, 0);
        ctx.write_bytes(arg + 0xFFF, b"a\0");
        assert_eq!(cb_load_exec(&mut ctx), ERR_PARAMETER, "one byte too long");
    }

    #[test]
    fn web_browser_is_unavailable() {
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A6D_0010_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, b"https://example.com/\0\0");
        let param = BASE + 0x100;
        ctx.write_bytes(param, &[0x5A; 64]);
        for (uri, p) in [(BASE, 0), (BASE, param), (BASE + 20, 0), (0, 0)] {
            ctx.set(Rdi, uri);
            ctx.set(Rsi, p);
            assert_eq!(cb_launch_web_browser(&mut ctx), 0x8002_002Du32 as i32 as i64);
        }
        assert_eq!(ctx.read_bytes(param, 64).unwrap(), vec![0x5A; 64]);
    }

    #[test]
    fn entitlement_events_from_app_content_init() {
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C80_0000_0000;
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let status = BASE;
        let event = BASE + 0x100;
        let init = BASE + 0x3000;
        let boot = BASE + 0x3100;
        let event_num = |ctx: &mut CpuContext| {
            ctx.set(Rdi, status);
            assert_eq!(cb_get_status(ctx), OK);
            ctx.read_u32(status).unwrap()
        };
        assert_eq!(event_num(&mut ctx), 0, "starts without pending events");
        ctx.set(Rdi, init);
        ctx.set(Rsi, boot);
        assert_eq!(crate::app_content::cb_initialize(&mut ctx), OK);
        assert_eq!(event_num(&mut ctx), 1, "AppContent init queues an entitlement update");
        assert_eq!(event_num(&mut ctx), 1, "status queries keep pending events");
        ctx.set(Rdi, 0);
        assert_eq!(cb_receive_event(&mut ctx), ERR_PARAMETER);
        assert_eq!(event_num(&mut ctx), 1, "a bad output pointer consumes nothing");
        ctx.write_bytes(event, &[0xFF; 4 + EVENT_DATA_SIZE]);
        ctx.set(Rdi, event);
        assert_eq!(cb_receive_event(&mut ctx), OK);
        assert_eq!(ctx.read_u32(event), Some(EVENT_ENTITLEMENT_UPDATE as u32));
        let payload = ctx.read_bytes(event + 4, EVENT_DATA_SIZE).unwrap();
        assert!(payload.iter().all(|&b| b == 0), "payload is cleared");
        assert_eq!(event_num(&mut ctx), 0, "delivered exactly once");
        ctx.set(Rdi, event);
        assert_eq!(cb_receive_event(&mut ctx), ERR_NO_EVENT);
    }

    #[test]
    fn param_ints() {
        assert_eq!(param_int_value(PARAM_ID_LANG, 1), 1);
        assert_eq!(param_int_value(PARAM_ID_LANG_PS5, 5), 5);
        assert_eq!(param_int_value(PARAM_ID_ENTER_BUTTON_ASSIGN, 1), 1, "cross");
        assert_eq!(param_int_value(PARAM_ID_DATE_FORMAT, 1), 1, "DD/MM/YYYY");
        assert_eq!(param_int_value(PARAM_ID_TIME_FORMAT, 1), 1, "24 hour");
        assert_eq!((param_int_value(PARAM_ID_TIME_ZONE, 1), param_int_value(PARAM_ID_SUMMERTIME, 1)), host_time_zone());
        // IST-5:30, BRT+3, UTC, CET in summer (CEST, +2 h).
        assert_eq!(time_zone_params(19_800, false), (330, 0));
        assert_eq!(time_zone_params(-10_800, false), (-180, 0));
        assert_eq!(time_zone_params(0, false), (0, 0));
        assert_eq!(time_zone_params(7_200, true), (60, 1));
        assert_eq!(param_int_value(PARAM_ID_GAME_PARENTAL_LEVEL, 1), 0);
        assert_eq!(param_int_value(PARAM_ID_CC_ENABLE, 1), 0);
        assert_eq!(param_int_value(PARAM_ID_SCREEN_READER, 1), 0);
        assert_eq!(param_int_value(12345, 1), 0);
    }

    #[test]
    fn params_status_and_display_info() {
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D40_0100_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);

        // String parameters: only the system name, and it must fit.
        ctx.set(Rdi, PARAM_ID_SYSTEM_NAME as u64);
        ctx.set(Rsi, BASE);
        ctx.set(Rdx, DEFAULT_SYSTEM_NAME.len() as u64);
        assert_eq!(cb_param_get_string(&mut ctx), ERR_PARAMETER, "no room for the terminator");
        ctx.set(Rdx, 65);
        assert_eq!(cb_param_get_string(&mut ctx), OK);
        assert_eq!(ctx.read_cstring(BASE, 65).as_deref(), Some(DEFAULT_SYSTEM_NAME));
        ctx.set(Rdi, PARAM_ID_LANG as u64);
        assert_eq!(cb_param_get_string(&mut ctx), ERR_PARAMETER);
        ctx.set(Rdx, 0);
        assert_eq!(cb_param_get_string(&mut ctx), ERR_PARAMETER);

        // Status: 136 bytes, all flags clear without a system dialog.
        ctx.write_bytes(BASE, &[0xEE; 0x100]);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_get_status(&mut ctx), OK);
        let status = ctx.read_bytes(BASE + 4, STATUS_SIZE - 4).unwrap();
        assert!(status.iter().all(|&b| b == 0));
        assert_eq!(ctx.read_bytes(BASE + STATUS_SIZE as u64, 1).unwrap()[0], 0xEE);

        assert_eq!(cb_get_hdr_tone_map_luminance(&mut ctx), OK);
        assert_eq!(ctx.read_f32(BASE), Some(80.0));
        assert_eq!(ctx.read_f32(BASE + 4), Some(1000.0));
        assert_eq!(ctx.read_f32(BASE + 8), Some(0.0));
        assert_eq!(cb_get_display_safe_area_info(&mut ctx), OK);
        assert_eq!(ctx.read_f32(BASE), Some(1.0));

        ctx.set(Rdi, 0);
        assert_eq!(cb_get_status(&mut ctx), ERR_PARAMETER);
        assert_eq!(cb_get_hdr_tone_map_luminance(&mut ctx), ERR_PARAMETER);
        assert_eq!(cb_get_notice_screen_skip_flag(&mut ctx), ERR_PARAMETER);
    }
}
