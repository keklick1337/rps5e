// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceGameUpdate HLE: update checks report that no newer patch (and no
//! add-on update) is available for the installed package.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceGameUpdate";
const OK: HleResult = 0;

const ERR_NOT_INITIALIZED: HleResult = 0x8041_2801u32 as i32 as i64;
const ERR_ALREADY_INITIALIZED: HleResult = 0x8041_2802u32 as i32 as i64;
const ERR_INVALID_ARG: HleResult = 0x8041_2803u32 as i32 as i64;
const ERR_INVALID_SIZE: HleResult = 0x8041_2804u32 as i32 as i64;
const ERR_REQUEST_NOT_FOUND: HleResult = 0x8041_2805u32 as i32 as i64;
const ERR_TOO_MANY_REQUESTS: HleResult = 0x8041_2806u32 as i32 as i64;
const ERR_ABORTED: HleResult = 0x8041_2807u32 as i32 as i64;
const ERR_MEMORY_FAULT: HleResult = 0x8002_000Eu32 as i32 as i64;

/// SceGameUpdateCheckParam { size_t size; u32 option; u32 reserved[9]; }.
const CHECK_PARAM_SIZE: u64 = 48;
/// SceGameUpdateCheckResult { size_t size; bool found; bool addcontFound;
/// pad[2]; char contentVersion[11]; pad; u32 reserved[6]; }.
const CHECK_RESULT_SIZE: u64 = 48;
/// SceGameUpdateAddcontVersionInfo { size_t size; bool found;
/// char contentVersion[11]; u32 reserved[6]; }.
const ADDCONT_VERSION_INFO_SIZE: u64 = 48;

/// The module keeps 8 requests, numbered from 0x20000000 up to 0x2FFFFFFF
/// and then from the start again, skipping ids still in use.
const MAX_REQUESTS: usize = 8;
const FIRST_REQUEST_ID: i32 = 0x2000_0000;
const LAST_REQUEST_ID: i32 = 0x2FFF_FFFF;

struct State {
    initialized: bool,
    next_request: i32,
    /// Live requests and whether each was aborted.
    requests: BTreeMap<i32, bool>,
}

static STATE: Mutex<State> = Mutex::new(State { initialized: false, next_request: FIRST_REQUEST_ID, requests: BTreeMap::new() });

fn following_request_id(id: i32) -> i32 {
    if id == LAST_REQUEST_ID { FIRST_REQUEST_ID } else { id + 1 }
}

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|p| p.into_inner())
}

fn cb_initialize(_ctx: &mut CpuContext) -> HleResult {
    let mut st = state();
    if st.initialized {
        return ERR_ALREADY_INITIALIZED;
    }
    st.initialized = true;
    OK
}

fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    let mut st = state();
    if !st.initialized {
        return ERR_NOT_INITIALIZED;
    }
    st.initialized = false;
    st.requests.clear();
    OK
}

/// The request checks of the module (measured on a PS5): an id of 0 or
/// less is REQUEST_NOT_FOUND before the library state, then
/// NOT_INITIALIZED, then an unknown id REQUEST_NOT_FOUND.
fn check_request(st: &State, id: i32) -> HleResult {
    if id <= 0 {
        return ERR_REQUEST_NOT_FOUND;
    }
    if !st.initialized {
        return ERR_NOT_INITIALIZED;
    }
    if st.requests.contains_key(&id) { OK } else { ERR_REQUEST_NOT_FOUND }
}

/// sceGameUpdateCreateRequest(): a new request id; a ninth live request is
/// TOO_MANY_REQUESTS.
fn cb_create_request(_ctx: &mut CpuContext) -> HleResult {
    let mut st = state();
    if !st.initialized {
        return ERR_NOT_INITIALIZED;
    }
    if st.requests.len() >= MAX_REQUESTS {
        return ERR_TOO_MANY_REQUESTS;
    }
    let mut id = st.next_request;
    while st.requests.contains_key(&id) {
        id = following_request_id(id);
    }
    st.next_request = following_request_id(id);
    st.requests.insert(id, false);
    id as HleResult
}

fn cb_delete_request(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let mut st = state();
    let r = check_request(&st, id);
    if r == OK {
        st.requests.remove(&id);
    }
    r
}

/// sceGameUpdateAbortRequest(requestId): checks complete synchronously, so
/// the abort only makes a later check of the request ABORTED.
fn cb_abort_request(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let mut st = state();
    let r = check_request(&st, id);
    if r == OK {
        st.requests.insert(id, true);
    }
    r
}

/// sceGameUpdateGetAddcontLatestVersion(serviceLabel, entitlementLabel*,
/// SceGameUpdateAddcontVersionInfo*): no add-on update is ever found.
fn cb_get_addcont_latest_version(ctx: &mut CpuContext) -> HleResult {
    let info = ctx.rdx();
    // The info, then the entitlement label, then the library state.
    if info == 0 || ctx.rsi() == 0 {
        return ERR_INVALID_ARG;
    }
    if !state().initialized {
        return ERR_NOT_INITIALIZED;
    }
    let Some(size) = ctx.read_u64(info) else {
        return ERR_MEMORY_FAULT;
    };
    if size < ADDCONT_VERSION_INFO_SIZE {
        return ERR_INVALID_SIZE;
    }
    let mut b = [0u8; ADDCONT_VERSION_INFO_SIZE as usize];
    b[..8].copy_from_slice(&size.to_le_bytes());
    if ctx.write_bytes(info, &b) { OK } else { ERR_MEMORY_FAULT }
}

/// sceGameUpdateCheck(requestId, const SceGameUpdateCheckParam*, SceGameUpdateCheckResult*)
fn cb_check(ctx: &mut CpuContext) -> HleResult {
    let (id, param, result) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    // Measured on a PS5: the id, the pointers, the exact sizes, the option
    // and reserved words, then the library state and the request.
    if id <= 0 {
        return ERR_REQUEST_NOT_FOUND;
    }
    if param == 0 || result == 0 {
        return ERR_INVALID_ARG;
    }
    let (Some(param_size), Some(result_size)) = (ctx.read_u64(param), ctx.read_u64(result)) else {
        return ERR_MEMORY_FAULT;
    };
    if param_size != CHECK_PARAM_SIZE || result_size != CHECK_RESULT_SIZE {
        return ERR_INVALID_SIZE;
    }
    let Some(words) = ctx.read_bytes(param + 8, 40) else { return ERR_MEMORY_FAULT };
    if words.iter().any(|&b| b != 0) {
        return ERR_INVALID_ARG;
    }
    {
        let st = state();
        let r = check_request(&st, id);
        if r != OK {
            return r;
        }
        if st.requests.get(&id) == Some(&true) {
            return ERR_ABORTED;
        }
    }
    // found = false and addcontFound = false; the rest of the result is
    // left as it is, as the module does when its patch service finds none.
    if !ctx.write_bytes(result + 8, &[0, 0]) {
        return ERR_MEMORY_FAULT;
    }
    log::debug!("[gameupdate] check request {id}: no update available");
    OK
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("YJtKLttI9fM", "sceGameUpdateInitialize", cb_initialize),
    ("NSH-C-OmoNI", "sceGameUpdateTerminate", cb_terminate),
    ("UvcvKaFvupA", "sceGameUpdateCreateRequest", cb_create_request),
    ("bcCyjHN5sn0", "sceGameUpdateDeleteRequest", cb_delete_request),
    ("LYVV9z8+owM", "sceGameUpdateCheck", cb_check),
    ("d1CNGEOaK28", "sceGameUpdateAbortRequest", cb_abort_request),
    ("0g0+Oq9xcI0", "sceGameUpdateGetAddcontLatestVersion", cb_get_addcont_latest_version),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    /// The library state is global: its tests take turns.
    static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn check_reports_no_update() {
        let _turn = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C0B_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        assert_eq!(cb_initialize(&mut ctx), OK);
        let id = cb_create_request(&mut ctx);
        assert!(id > 0);
        ctx.write_u64(BASE, CHECK_PARAM_SIZE);
        ctx.write_bytes(BASE + 0x100, &[0xFF; 48]);
        ctx.write_u64(BASE + 0x100, CHECK_RESULT_SIZE);
        ctx.set(Rdi, id as u64);
        ctx.set(Rsi, BASE);
        ctx.set(Rdx, BASE + 0x100);
        assert_eq!(cb_check(&mut ctx), OK);
        assert_eq!(ctx.read_u64(BASE + 0x100), Some(CHECK_RESULT_SIZE));
        assert_eq!(ctx.read_bytes(BASE + 0x108, 2).unwrap(), vec![0, 0]);
        assert_eq!(ctx.read_bytes(BASE + 0x10A, 38).unwrap(), vec![0xFF; 38], "the rest is left alone");
        ctx.write_u64(BASE + 0x100, 8);
        assert_eq!(cb_check(&mut ctx), ERR_INVALID_SIZE);
        assert_eq!(cb_abort_request(&mut ctx), OK);
        ctx.write_bytes(BASE + 0x200, &[0xEE; 48]);
        ctx.write_u64(BASE + 0x200, ADDCONT_VERSION_INFO_SIZE);
        ctx.set(Rdx, BASE + 0x200);
        assert_eq!(cb_get_addcont_latest_version(&mut ctx), OK);
        assert_eq!(ctx.read_u64(BASE + 0x200), Some(ADDCONT_VERSION_INFO_SIZE));
        assert_eq!(ctx.read_bytes(BASE + 0x208, 40).unwrap(), vec![0; 40]);
        ctx.write_u64(BASE + 0x200, 16);
        assert_eq!(cb_get_addcont_latest_version(&mut ctx), ERR_INVALID_SIZE);
        ctx.set(Rsi, 0);
        assert_eq!(cb_get_addcont_latest_version(&mut ctx), ERR_INVALID_ARG, "null entitlement label");
        ctx.set(Rsi, BASE);
        ctx.set(Rdx, 0);
        assert_eq!(cb_get_addcont_latest_version(&mut ctx), ERR_INVALID_ARG);
        ctx.set(Rdx, BASE + 0x100);
        // Exact sizes, then a zero option and reserved words (PS5).
        ctx.write_u64(BASE + 0x100, CHECK_RESULT_SIZE + 8);
        assert_eq!(cb_check(&mut ctx), ERR_INVALID_SIZE);
        ctx.write_u64(BASE + 0x100, CHECK_RESULT_SIZE);
        ctx.write_u32(BASE + 8, 1);
        assert_eq!(cb_check(&mut ctx), ERR_INVALID_ARG);
        ctx.write_u32(BASE + 8, 0);
        ctx.write_u32(BASE + 44, 1);
        assert_eq!(cb_check(&mut ctx), ERR_INVALID_ARG);
        ctx.write_u32(BASE + 44, 0);
        assert_eq!(cb_delete_request(&mut ctx), OK);
        assert_eq!(cb_delete_request(&mut ctx), ERR_REQUEST_NOT_FOUND);
        assert_eq!(cb_check(&mut ctx), ERR_REQUEST_NOT_FOUND);
        assert_eq!(cb_abort_request(&mut ctx), ERR_REQUEST_NOT_FOUND);
        // The library state: a second Initialize, Terminate twice; ids of 0
        // or less come before it.
        assert_eq!(cb_initialize(&mut ctx), ERR_ALREADY_INITIALIZED);
        assert_eq!(cb_terminate(&mut ctx), OK);
        assert_eq!(cb_terminate(&mut ctx), ERR_NOT_INITIALIZED);
        assert_eq!(cb_abort_request(&mut ctx), ERR_NOT_INITIALIZED);
        ctx.set(Rdi, 0);
        assert_eq!(cb_abort_request(&mut ctx), ERR_REQUEST_NOT_FOUND);
        assert_eq!(cb_check(&mut ctx), ERR_REQUEST_NOT_FOUND);
        ctx.set(Rsi, 0);
        ctx.set(Rdx, BASE + 0x200);
        assert_eq!(cb_get_addcont_latest_version(&mut ctx), ERR_INVALID_ARG, "arguments before the state");
    }

    /// Request ids start at 0x20000000; the module keeps 8 requests; a
    /// check of an aborted request is ABORTED.
    #[test]
    fn requests_have_eight_slots_and_abort() {
        let _turn = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C0B_0010_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let _ = cb_terminate(&mut ctx);
        state().next_request = FIRST_REQUEST_ID;
        assert_eq!(cb_initialize(&mut ctx), OK);
        let ids: Vec<HleResult> = (0..8).map(|_| cb_create_request(&mut ctx)).collect();
        assert_eq!(ids, (0..8).map(|i| (FIRST_REQUEST_ID + i) as HleResult).collect::<Vec<_>>());
        assert_eq!(cb_create_request(&mut ctx), ERR_TOO_MANY_REQUESTS);
        ctx.set(Rdi, ids[2] as u64);
        assert_eq!(cb_delete_request(&mut ctx), OK);
        assert_eq!(cb_create_request(&mut ctx), (FIRST_REQUEST_ID + 8) as HleResult);
        ctx.write_u64(BASE, CHECK_PARAM_SIZE);
        ctx.write_u64(BASE + 0x100, CHECK_RESULT_SIZE);
        ctx.set(Rdi, ids[0] as u64);
        ctx.set(Rsi, BASE);
        ctx.set(Rdx, BASE + 0x100);
        assert_eq!(cb_check(&mut ctx), OK);
        assert_eq!(cb_abort_request(&mut ctx), OK);
        assert_eq!(cb_check(&mut ctx), ERR_ABORTED);
        assert_eq!(cb_terminate(&mut ctx), OK);
        // Ids wrap from 0x2FFFFFFF to 0x20000000.
        state().next_request = LAST_REQUEST_ID;
        assert_eq!(cb_initialize(&mut ctx), OK);
        assert_eq!(cb_create_request(&mut ctx), LAST_REQUEST_ID as HleResult);
        assert_eq!(cb_create_request(&mut ctx), FIRST_REQUEST_ID as HleResult);
        assert_eq!(cb_terminate(&mut ctx), OK);
    }
}
