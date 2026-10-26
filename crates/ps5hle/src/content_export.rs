// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceContentExport HLE.
//!
//! The library can be initialized and export sessions started and finished
//! (at most 10 at a time), but there is no system media gallery to export
//! into: every export of a live session fails with "not supported".

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceContentExport";
const OK: HleResult = 0;

const ERR_INVALID_PARAM: HleResult = 0x809D_3016u32 as i32 as i64;
/// Kernel ENOTSUP: exporting content is not available.
const ERR_NOT_SUPPORTED: HleResult = 0x8002_002Du32 as i32 as i64;
const ERR_MEMORY_FAULT: HleResult = 0x8002_000Eu32 as i32 as i64;

/// SceContentExportInitParam2: mallocFunc, freeFunc, userData, bufferSize,
/// reserved0, reserved1 (all 8 bytes).
const INIT_PARAM2_SIZE: usize = 0x30;

static INITIALIZED: AtomicBool = AtomicBool::new(false);

fn cb_init2(ctx: &mut CpuContext) -> HleResult {
    let param = ctx.rdi();
    if param == 0 {
        return ERR_INVALID_PARAM;
    }
    let Some(raw) = ctx.read_bytes(param, INIT_PARAM2_SIZE) else {
        return ERR_MEMORY_FAULT;
    };
    let field = |i: usize| u64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().unwrap());
    if field(0) == 0 || field(1) == 0 || field(4) != 0 || field(5) != 0 {
        return ERR_INVALID_PARAM;
    }
    if INITIALIZED.swap(true, Ordering::AcqRel) {
        return ERR_INVALID_PARAM;
    }
    OK
}

/// Live export sessions at most.
const MAX_SESSIONS: usize = 10;

/// Live session ids and the last id handed out.
static SESSIONS: Mutex<(BTreeSet<i32>, i32)> = Mutex::new((BTreeSet::new(), 0));

fn sessions() -> std::sync::MutexGuard<'static, (BTreeSet<i32>, i32)> {
    SESSIONS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Term: refused while sessions are live.
fn cb_term(_ctx: &mut CpuContext) -> HleResult {
    if !sessions().0.is_empty() {
        return ERR_INVALID_PARAM;
    }
    if INITIALIZED.swap(false, Ordering::AcqRel) { OK } else { ERR_INVALID_PARAM }
}

/// sceContentExportStart() -> session id (1, 2, ...).
fn cb_start(_ctx: &mut CpuContext) -> HleResult {
    if !INITIALIZED.load(Ordering::Acquire) {
        return ERR_INVALID_PARAM;
    }
    let mut s = sessions();
    if s.0.len() >= MAX_SESSIONS || s.1 == i32::MAX {
        return ERR_NOT_SUPPORTED;
    }
    s.1 += 1;
    let id = s.1;
    s.0.insert(id);
    id as HleResult
}

fn live_session(id: i32) -> bool {
    INITIALIZED.load(Ordering::Acquire) && sessions().0.contains(&id)
}

/// sceContentExportFinish(expId): ends a live session.
fn cb_finish(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    if !INITIALIZED.load(Ordering::Acquire) || !sessions().0.remove(&id) {
        return ERR_INVALID_PARAM;
    }
    OK
}

/// FromData / FromFile / ...(expId, ...): a live session is checked; there
/// is no export destination.
fn cb_export(ctx: &mut CpuContext) -> HleResult {
    if !live_session(ctx.rdi() as i32) {
        return ERR_INVALID_PARAM;
    }
    log::warn!("[contentexport] content export is not supported");
    ERR_NOT_SUPPORTED
}

/// GetProgress / Cancel(expId): a live session never has an export in
/// progress, and what the console reports for one is unknown.
fn cb_no_running_export(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    if !live_session(id) {
        return ERR_INVALID_PARAM;
    }
    log::warn!("[contentexport] session {id} has no running export (state unknown)");
    ERR_NOT_SUPPORTED
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("0GnN4QCgIfs", "sceContentExportInit2", cb_init2),
    ("+KDWny9Y-6k", "sceContentExportTerm", cb_term),
    ("FCygF4Ec4so", "sceContentExportStart", cb_start),
    ("AOWqIYsgVHs", "sceContentExportFromData", cb_export),
    ("OWCJUmrWH1g", "sceContentExportFromFile", cb_export),
    ("mbakKLPSO4o", "sceContentExportFromFileWithThumbnail", cb_export),
    ("uZTQHI50WpY", "sceContentExportFromDataWithThumbnail", cb_export),
    ("eRJv4xU7pGU", "sceContentExportGetProgress", cb_no_running_export),
    ("efqHe7wPRbs", "sceContentExportCancel", cb_no_running_export),
    ("tb3cZTCl8Ps", "sceContentExportFinish", cb_finish),
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

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn init_and_export() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A69_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_init2(&mut ctx), ERR_INVALID_PARAM); // no allocator
        ctx.write_u64(BASE, 0x1000);
        ctx.write_u64(BASE + 8, 0x2000);
        assert_eq!(cb_init2(&mut ctx), OK);
        // Sessions 1 and 2; exports of a live session are not supported.
        assert_eq!(cb_start(&mut ctx), 1);
        assert_eq!(cb_start(&mut ctx), 2);
        ctx.set(Rdi, 1);
        assert_eq!(cb_export(&mut ctx), ERR_NOT_SUPPORTED);
        assert_eq!(cb_no_running_export(&mut ctx), ERR_NOT_SUPPORTED, "no export runs");
        ctx.set(Rdi, 7);
        assert_eq!(cb_export(&mut ctx), ERR_INVALID_PARAM, "unknown session");
        assert_eq!(cb_no_running_export(&mut ctx), ERR_INVALID_PARAM, "unknown session");
        assert_eq!(cb_finish(&mut ctx), ERR_INVALID_PARAM);
        assert_eq!(cb_term(&mut ctx), ERR_INVALID_PARAM, "sessions are live");
        for id in [1, 2] {
            ctx.set(Rdi, id);
            assert_eq!(cb_finish(&mut ctx), OK);
        }
        assert_eq!(cb_finish(&mut ctx), ERR_INVALID_PARAM);
        for _ in 0..MAX_SESSIONS {
            assert!(cb_start(&mut ctx) > 0);
        }
        assert_eq!(cb_start(&mut ctx), ERR_NOT_SUPPORTED, "ten live sessions at most");
        sessions().0.clear();
        assert_eq!(cb_term(&mut ctx), OK);
        assert_eq!(cb_term(&mut ctx), ERR_INVALID_PARAM);
        assert_eq!(cb_start(&mut ctx), ERR_INVALID_PARAM);
    }
}
