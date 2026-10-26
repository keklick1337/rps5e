// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceUserService HLE: one to four signed-in local users (one per
//! configured local player, see `set_local_user_count`).

use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};
use std::sync::atomic::{AtomicUsize, Ordering};

const LIB: &str = "libSceUserService";
const OK: HleResult = 0;
const ERR_INVALID_ARGUMENT: HleResult = 0x8096_0005u32 as i32 as i64;
const ERR_NO_EVENT: HleResult = 0x8096_0007u32 as i32 as i64;
const ERR_NOT_LOGGED_IN: HleResult = 0x8096_0009u32 as i32 as i64;
const ERR_BUFFER_TOO_SHORT: HleResult = 0x8096_000Au32 as i32 as i64;
/// SCE_USER_SERVICE_MAX_USER_NAME_LENGTH.
const USER_NAME_MAX_LENGTH: u64 = 16;
/// Most bytes of a name buffer that are cleared (a guard against a bogus
/// size; names are 17 bytes).
const USER_NAME_FILL_MAX: u64 = 0x1000;

/// Retail user ids carry the local user slot in the 0x10000000 range; the
/// first signed-in user is slot 0. Small ids fail engines' id-to-slot math.
pub const PRIMARY_USER_ID: i32 = 0x1000_0000;
pub const INVALID_USER_ID: i32 = -1;
pub const PRIMARY_USER_NAME: &str = "rps5e";
/// Local user slots (SceUserServiceLoginUserIdList has four entries).
pub const MAX_LOCAL_USERS: usize = 4;

static LOCAL_USERS: AtomicUsize = AtomicUsize::new(1);
/// Users whose LOGIN event was delivered.
static LOGIN_EVENTS_SENT: AtomicUsize = AtomicUsize::new(0);

/// Number of signed-in local users (1..=4). Set by the host input layer
/// before the title starts; user `i` has id `PRIMARY_USER_ID + i`.
pub fn set_local_user_count(n: usize) {
    LOCAL_USERS.store(n.clamp(1, MAX_LOCAL_USERS), Ordering::Release);
}

pub fn local_user_count() -> usize {
    LOCAL_USERS.load(Ordering::Acquire)
}

/// User id of local user `index`.
pub fn user_id(index: usize) -> i32 {
    PRIMARY_USER_ID + index as i32
}

/// Local slot of a signed-in user id.
pub fn user_index(id: i32) -> Option<usize> {
    let i = id.wrapping_sub(PRIMARY_USER_ID);
    (0..local_user_count() as i32).contains(&i).then_some(i as usize)
}

fn user_name(index: usize) -> String {
    if index == 0 {
        PRIMARY_USER_NAME.to_string()
    } else {
        format!("{PRIMARY_USER_NAME}-{}", index + 1)
    }
}
/// Size of SceUserServiceGamePresets (first u64 is the struct size).
const GAME_PRESETS_SIZE: usize = 0x28;


fn write_i32(ctx: &CpuContext, addr: u64, v: i32) -> HleResult {
    if ctx.write_bytes(addr, &v.to_le_bytes()) { OK } else { ERR_INVALID_ARGUMENT }
}

/// SceUserServiceLoginUserIdList: four slots, unused ones are -1.
pub fn login_user_id_list() -> [u8; 16] {
    let mut b = [0u8; 16];
    let n = local_user_count();
    for i in 0..MAX_LOCAL_USERS {
        let id = if i < n { user_id(i) } else { INVALID_USER_ID };
        b[i * 4..i * 4 + 4].copy_from_slice(&id.to_le_bytes());
    }
    b
}

const ERR_NOT_INITIALIZED: HleResult = 0x8096_0002u32 as i32 as i64;
const ERR_ALREADY_INITIALIZED: HleResult = 0x8096_0003u32 as i32 as i64;
/// The thread priorities the service's thread may run at.
const THREAD_PRIORITY: std::ops::RangeInclusive<i32> = 0x100..=0x2FF;

static INITIALIZED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Initialize once; a second call is ALREADY_INITIALIZED (measured on a
/// PS5).
fn mark_initialized() -> HleResult {
    if INITIALIZED.swap(true, std::sync::atomic::Ordering::AcqRel) { ERR_ALREADY_INITIALIZED } else { OK }
}

/// sceUserServiceInitialize(const SceUserServiceInitializeParams*): the
/// first int is the service thread's priority (0x100..=0x2FF).
fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    let params = ctx.rdi();
    if params != 0 {
        let Some(priority) = ctx.read_u32(params) else { return ERR_INVALID_ARGUMENT };
        if !THREAD_PRIORITY.contains(&(priority as i32)) {
            return ERR_INVALID_ARGUMENT;
        }
    }
    mark_initialized()
}

/// sceUserServiceInitialize2(int threadPriority)
fn cb_initialize2(ctx: &mut CpuContext) -> HleResult {
    if !THREAD_PRIORITY.contains(&(ctx.rdi() as i32)) {
        return ERR_INVALID_ARGUMENT;
    }
    mark_initialized()
}

fn cb_get_initial_user(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    write_i32(ctx, out, PRIMARY_USER_ID)
}

fn cb_get_login_user_id_list(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if ctx.write_bytes(out, &login_user_id_list()) { OK } else { ERR_INVALID_ARGUMENT }
}

/// One LOGIN event (type 0) per signed-in user, then NO_EVENT forever.
fn cb_get_event(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let n = local_user_count();
    let next = LOGIN_EVENTS_SENT.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |s| (s < n).then_some(s + 1));
    let Ok(index) = next else {
        return ERR_NO_EVENT;
    };
    let mut ev = [0u8; 8];
    ev[4..8].copy_from_slice(&user_id(index).to_le_bytes());
    if ctx.write_bytes(out, &ev) { OK } else { ERR_INVALID_ARGUMENT }
}

fn cb_get_user_name(ctx: &mut CpuContext) -> HleResult {
    let user = ctx.rdi() as i32;
    let out = ctx.rsi();
    let cap = ctx.rdx();
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(index) = user_index(user) else {
        return ERR_NOT_LOGGED_IN;
    };
    // A buffer must hold the longest name (16 bytes) and its NUL; a shorter
    // one is cleared. The name is copied strncpy-style, zero-filled to `cap`.
    let fill = cap.min(USER_NAME_FILL_MAX) as usize;
    if cap < USER_NAME_MAX_LENGTH + 1 {
        ctx.write_bytes(out, &vec![0u8; fill]);
        return ERR_BUFFER_TOO_SHORT;
    }
    let name = user_name(index);
    let name = &name.as_bytes()[..name.len().min(USER_NAME_MAX_LENGTH as usize)];
    let mut buf = vec![0u8; fill.max(name.len() + 1)];
    buf[..name.len()].copy_from_slice(name);
    if ctx.write_bytes(out, &buf) { OK } else { ERR_INVALID_ARGUMENT }
}

/// sceUserServiceGetPlatformPrivacyWs1: accepted without effect.
fn cb_get_platform_privacy_ws1(_ctx: &mut CpuContext) -> HleResult {
    OK
}

/// sceUserServiceGetGamePresets(userId, SceUserServiceGamePresets*): the
/// caller's thisSize (0 or oversized = the full 40 bytes) is cleared and
/// thisSize set to the full size.
fn cb_get_game_presets(ctx: &mut CpuContext) -> HleResult {
    let user = ctx.rdi() as i32;
    let out = ctx.rsi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if user_index(user).is_none() {
        return ERR_NOT_LOGGED_IN;
    }
    let Some(this_size) = ctx.read_u64(out) else {
        return ERR_INVALID_ARGUMENT;
    };
    let this_size = match this_size as usize {
        0 => GAME_PRESETS_SIZE,
        n if n > GAME_PRESETS_SIZE => GAME_PRESETS_SIZE,
        n => n.max(8),
    };
    let mut presets = vec![0u8; this_size];
    presets[..8].copy_from_slice(&(GAME_PRESETS_SIZE as u64).to_le_bytes());
    if ctx.write_bytes(out, &presets) { OK } else { ERR_INVALID_ARGUMENT }
}

/// Per-user integer setting getter `(userId, int* out)`: the output pointer
/// is checked before the user.
fn write_user_setting(ctx: &CpuContext, value: i32) -> HleResult {
    let user = ctx.rdi() as i32;
    let out = ctx.rsi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if user_index(user).is_none() {
        return ERR_NOT_LOGGED_IN;
    }
    write_i32(ctx, out, value)
}

/// Age level of the signed-in users.
pub const AGE_LEVEL: i32 = 9;

fn cb_get_age_level(ctx: &mut CpuContext) -> HleResult {
    write_user_setting(ctx, AGE_LEVEL)
}
fn cb_get_setting_off(ctx: &mut CpuContext) -> HleResult {
    write_user_setting(ctx, 0)
}
fn cb_get_setting_on(ctx: &mut CpuContext) -> HleResult {
    write_user_setting(ctx, 1)
}

/// sceUserServiceGetUserNumber(userId, int32_t* number): the 1-based local
/// user number.
fn cb_get_user_number(ctx: &mut CpuContext) -> HleResult {
    let user = ctx.rdi() as i32;
    let out = ctx.rsi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(index) = user_index(user) else {
        return ERR_NOT_LOGGED_IN;
    };
    write_i32(ctx, out, index as i32 + 1)
}

/// sceUserServiceGetForegroundUser(int* userId): the first local user.
fn cb_get_foreground_user(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    write_i32(ctx, out, PRIMARY_USER_ID)
}

/// SceUserServiceRegisteredUserIdList: sixteen user ids.
const REGISTERED_USERS_MAX: usize = 16;

/// The registered users are the local users; unused slots are -1.
pub fn registered_user_id_list() -> [u8; REGISTERED_USERS_MAX * 4] {
    let mut b = [0u8; REGISTERED_USERS_MAX * 4];
    let n = local_user_count();
    for i in 0..REGISTERED_USERS_MAX {
        let id = if i < n { user_id(i) } else { INVALID_USER_ID };
        b[i * 4..i * 4 + 4].copy_from_slice(&id.to_le_bytes());
    }
    b
}

/// sceUserServiceGetRegisteredUserIdList(SceUserServiceRegisteredUserIdList*)
fn cb_get_registered_user_id_list(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if ctx.write_bytes(out, &registered_user_id_list()) { OK } else { ERR_INVALID_ARGUMENT }
}

/// sceUserServiceGetUserColor(userId, SceUserServiceUserColor*): the color
/// of the user's slot (blue, red, green, pink). An unknown user is an
/// invalid argument.
fn cb_get_user_color(ctx: &mut CpuContext) -> HleResult {
    let (user, out) = (ctx.rdi() as i32, ctx.rsi());
    let Some(index) = user_index(user).filter(|_| out != 0) else {
        return ERR_INVALID_ARGUMENT;
    };
    write_i32(ctx, out, index as i32)
}

/// sceUserServiceGetNpAccountId(userId, SceNpAccountId*): the local users
/// have no PSN account (id 0). An unknown user is an invalid argument.
fn cb_get_np_account_id(ctx: &mut CpuContext) -> HleResult {
    let (user, out) = (ctx.rdi() as i32, ctx.rsi());
    if out == 0 || user_index(user).is_none() {
        return ERR_INVALID_ARGUMENT;
    }
    if ctx.write_u64(out, 0) { OK } else { ERR_INVALID_ARGUMENT }
}

/// sceUserServiceTerminate: the local user stays logged in; the service
/// may be initialized again.
fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    if INITIALIZED.swap(false, std::sync::atomic::Ordering::AcqRel) { OK } else { ERR_NOT_INITIALIZED }
}

pub fn register(reg: &mut HleRegistry) {
    reg.register_fn("j3YMu1MVNNo", "sceUserServiceInitialize", LIB, cb_initialize);
    reg.register_fn("az-0R6eviZ0", "sceUserServiceInitialize2", LIB, cb_initialize2);
    reg.register_fn("qbwy0Ub8b3M", "sceUserServiceGetUserNumber", LIB, cb_get_user_number);
    reg.register_fn("bwFjS+bX9mA", "sceUserServiceTerminate", LIB, cb_terminate);
    reg.register_fn("CdWp0oHWGr0", "sceUserServiceGetInitialUser", LIB, cb_get_initial_user);
    reg.register_fn("fPhymKNvK-A", "sceUserServiceGetLoginUserIdList", LIB, cb_get_login_user_id_list);
    reg.register_fn("yH17Q6NWtVg", "sceUserServiceGetEvent", LIB, cb_get_event);
    reg.register_fn("1xxcMiGu2fo", "sceUserServiceGetUserName", LIB, cb_get_user_name);
    reg.register_fn("D-CzAxQL0XI", "sceUserServiceGetPlatformPrivacyWs1", LIB, cb_get_platform_privacy_ws1);
    reg.register_fn("woNpu+45RLk", "sceUserServiceGetAgeLevel", LIB, cb_get_age_level);
    reg.register_fn("-sD02mFDBh4", "sceUserServiceGetGamePresets", LIB, cb_get_game_presets);
    reg.register_fn("rnEhHqG-4xo", "sceUserServiceGetAccessibilityChatTranscription", LIB, cb_get_setting_off);
    reg.register_fn("ZKJtxdgvzwg", "sceUserServiceGetAccessibilityPressAndHoldDelay", LIB, cb_get_setting_off);
    reg.register_fn("-3Y5GO+-i78", "sceUserServiceGetAccessibilityTriggerEffect", LIB, cb_get_setting_on);
    reg.register_fn("qWYHOFwqCxY", "sceUserServiceGetAccessibilityVibration", LIB, cb_get_setting_on);
    reg.register_fn("hD-H81EN9Vg", "sceUserServiceGetAccessibilityZoomEnabled", LIB, cb_get_setting_off);
    reg.register_fn("O6IW1-Dwm-w", "sceUserServiceGetAccessibilityZoomFollowFocus", LIB, cb_get_setting_off);
    reg.register_fn("eNb53LQJmIM", "sceUserServiceGetForegroundUser", LIB, cb_get_foreground_user);
    reg.register_fn("5EiQCnL2G1Y", "sceUserServiceGetRegisteredUserIdList", LIB, cb_get_registered_user_id_list);
    reg.register_fn("lUoqwTQu4Go", "sceUserServiceGetUserColor", LIB, cb_get_user_color);
    reg.register_fn("6dfDreosXGY", "sceUserServiceGetNpAccountId", LIB, cb_get_np_account_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Initialization checks the thread priority and refuses a second
    /// initialization; Terminate needs an initialized service (reference
    /// GuestUserService cases measured on a PS5).
    #[test]
    fn initialize_priority_and_lifecycle() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A70_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        use crate::context::CpuRegister::Rdi;
        let was = INITIALIZED.swap(false, std::sync::atomic::Ordering::AcqRel);
        assert_eq!(cb_terminate(&mut ctx), ERR_NOT_INITIALIZED);
        for bad in [0u64, 0xFF, 0x300, u32::MAX as u64] {
            ctx.set(Rdi, bad);
            assert_eq!(cb_initialize2(&mut ctx), ERR_INVALID_ARGUMENT, "{bad:#x}");
            ctx.write_u32(BASE, bad as u32);
            ctx.set(Rdi, BASE);
            assert_eq!(cb_initialize(&mut ctx), ERR_INVALID_ARGUMENT, "{bad:#x}");
        }
        ctx.set(Rdi, 0x2FF);
        assert_eq!(cb_initialize2(&mut ctx), OK);
        assert_eq!(cb_initialize2(&mut ctx), ERR_ALREADY_INITIALIZED);
        ctx.set(Rdi, 0);
        assert_eq!(cb_initialize(&mut ctx), ERR_ALREADY_INITIALIZED);
        assert_eq!(cb_terminate(&mut ctx), OK);
        ctx.write_u32(BASE, 0x100);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_initialize(&mut ctx), OK);
        INITIALIZED.store(was, std::sync::atomic::Ordering::Release);
    }

    #[test]
    fn terminate_nid() {
        assert_eq!(ps5nid::compute_nid("sceUserServiceTerminate"), "bwFjS+bX9mA");
    }

    #[test]
    fn nids_match_names() {
        for (nid, name) in [
            ("az-0R6eviZ0", "sceUserServiceInitialize2"),
            ("qbwy0Ub8b3M", "sceUserServiceGetUserNumber"),
            ("D-CzAxQL0XI", "sceUserServiceGetPlatformPrivacyWs1"),
            ("eNb53LQJmIM", "sceUserServiceGetForegroundUser"),
            ("5EiQCnL2G1Y", "sceUserServiceGetRegisteredUserIdList"),
            ("lUoqwTQu4Go", "sceUserServiceGetUserColor"),
            ("6dfDreosXGY", "sceUserServiceGetNpAccountId"),
        ] {
            assert_eq!(ps5nid::compute_nid(name), nid);
        }
    }

    /// The name pointer is checked first; a buffer under 17 bytes is
    /// cleared and refused; a name fills the rest of the buffer with zeros.
    #[test]
    fn user_name_buffer() {
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D43_0300_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let call = |ctx: &mut CpuContext, user: i32, out: u64, cap: u64| {
            ctx.set(Rdi, user as u64);
            ctx.set(Rsi, out);
            ctx.set(Rdx, cap);
            cb_get_user_name(ctx)
        };
        assert_eq!(call(&mut ctx, 77, 0, 32), ERR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, 77, BASE, 32), ERR_NOT_LOGGED_IN);
        ctx.write_bytes(BASE, &[0xCC; 64]);
        assert_eq!(call(&mut ctx, PRIMARY_USER_ID, BASE, 16), ERR_BUFFER_TOO_SHORT);
        assert_eq!(ctx.read_bytes(BASE, 17).unwrap(), [vec![0; 16], vec![0xCC]].concat());
        ctx.write_bytes(BASE, &[0xCC; 64]);
        assert_eq!(call(&mut ctx, PRIMARY_USER_ID, BASE, 32), OK);
        let name = user_name(0);
        let mut expect = name.as_bytes().to_vec();
        expect.resize(32, 0);
        expect.push(0xCC);
        assert_eq!(ctx.read_bytes(BASE, 33).unwrap(), expect);
    }

    #[test]
    fn settings_presets_and_user_number() {
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D40_0200_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);

        // Output pointer is checked before the user id.
        ctx.set(Rdi, 77);
        ctx.set(Rsi, 0);
        assert_eq!(cb_get_age_level(&mut ctx), ERR_INVALID_ARGUMENT);
        assert_eq!(cb_get_user_number(&mut ctx), ERR_INVALID_ARGUMENT);
        assert_eq!(cb_get_game_presets(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.set(Rsi, BASE);
        assert_eq!(cb_get_age_level(&mut ctx), ERR_NOT_LOGGED_IN);
        assert_eq!(cb_get_user_number(&mut ctx), ERR_NOT_LOGGED_IN);

        ctx.set(Rdi, PRIMARY_USER_ID as u64);
        assert_eq!(cb_get_age_level(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE), Some(9));
        assert_eq!(cb_get_setting_on(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE), Some(1));
        assert_eq!(cb_get_user_number(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE), Some(1));

        // Presets: only thisSize bytes are cleared; thisSize becomes 40.
        ctx.write_bytes(BASE, &[0xCC; 0x40]);
        ctx.write_u64(BASE, 16);
        assert_eq!(cb_get_game_presets(&mut ctx), OK);
        assert_eq!(ctx.read_u64(BASE), Some(GAME_PRESETS_SIZE as u64));
        assert_eq!(ctx.read_bytes(BASE + 8, 8).unwrap(), vec![0; 8]);
        assert_eq!(ctx.read_bytes(BASE + 16, 1).unwrap()[0], 0xCC);
        ctx.write_u64(BASE, 0);
        assert_eq!(cb_get_game_presets(&mut ctx), OK);
        assert_eq!(ctx.read_bytes(BASE + 8, 32).unwrap(), vec![0; 32]);
        assert_eq!(ctx.read_bytes(BASE + 40, 1).unwrap()[0], 0xCC);
    }

    /// Port of the guest user service test: foreground user, registered
    /// list, color and account id of the first local user.
    #[test]
    fn foreground_registered_color_account() {
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5FA8_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);

        ctx.set(Rdi, 0);
        assert_eq!(cb_get_foreground_user(&mut ctx), ERR_INVALID_ARGUMENT);
        assert_eq!(cb_get_registered_user_id_list(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_get_foreground_user(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE), Some(PRIMARY_USER_ID as u32));
        assert_eq!(cb_get_registered_user_id_list(&mut ctx), OK);
        let list = ctx.read_bytes(BASE, 64).unwrap();
        assert_eq!(&list[..4], &PRIMARY_USER_ID.to_le_bytes());
        assert_eq!(&list[60..], &[0xFF; 4], "unused slots are -1");

        ctx.set(Rdi, PRIMARY_USER_ID as u64);
        ctx.set(Rsi, 0);
        assert_eq!(cb_get_user_color(&mut ctx), ERR_INVALID_ARGUMENT);
        assert_eq!(cb_get_np_account_id(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.set(Rsi, BASE);
        ctx.write_u64(BASE, 0xDEAD_BEEF);
        assert_eq!(cb_get_user_color(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE), Some(0), "the first user is blue");
        assert_eq!(cb_get_np_account_id(&mut ctx), OK);
        assert_eq!(ctx.read_u64(BASE), Some(0));
        ctx.set(Rdi, 123);
        assert_eq!(cb_get_user_color(&mut ctx), ERR_INVALID_ARGUMENT);
        assert_eq!(cb_get_np_account_id(&mut ctx), ERR_INVALID_ARGUMENT);
    }

    #[test]
    fn login_list_layout() {
        let b = login_user_id_list();
        assert_eq!(&b[0..4], &0x1000_0000i32.to_le_bytes());
        assert_eq!(&b[4..8], &[0xFF; 4]);
        assert_eq!(&b[12..16], &[0xFF; 4]);
        assert_eq!(user_index(0x1000_0000), Some(0));
        assert_eq!(user_index(0x1000_0001), None);
        assert_eq!(user_index(-1), None);
    }
}
