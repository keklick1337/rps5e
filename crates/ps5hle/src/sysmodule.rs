// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceSysmodule HLE: system module load bookkeeping.
//!
//! System modules are HLE libraries that are always present, so loading
//! only counts references per id: every load adds one, every unload drops
//! one, and a module with no references reports UNLOADED (games probe
//! that status before running their own initialization). The kernel module list can consult
//! [`loaded_sysmodules`] / [`known_sysmodule_names`] to expose them.
//! sceSysmoduleGetModuleInfoForUnwind lives with the module registry
//! (`modules.rs`), as it shares the kernel unwind-info implementation.

use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};
use std::collections::BTreeMap;
use std::sync::Mutex;

const LIB: &str = "libSceSysmodule";
const OK: HleResult = 0;
/// SCE_SYSMODULE_ERROR_UNLOADED: the module has no load reference.
const ERR_UNLOADED: HleResult = 0x805A_1001u32 as i32 as i64;
/// SCE_SYSMODULE_ERROR_INVALID_VALUE: bad argument (here an unwritable
/// result pointer).
const ERR_INVALID_VALUE: HleResult = 0x805A_1000u32 as i32 as i64;

/// Load reference count per module id (absent = unloaded). Public calls
/// take 16-bit ids, the internal ones 32-bit ids; small ids are shared.
static LOADED: Mutex<BTreeMap<u32, u32>> = Mutex::new(BTreeMap::new());

/// PRX file names of sysmodule ids that ship as real modules.
pub fn known_sysmodule_names(id: i32) -> &'static [&'static str] {
    match id {
        0x0006 => &["libSceFiber.sprx", "libSceFiber.prx"],
        0x0021 => &["libSceRudp.sprx", "libSceRudp.prx"],
        0x0095 => &["libSceIme.sprx", "libSceIme.prx"],
        0x0096 => &["libSceImeDialog.sprx", "libSceImeDialog.prx"],
        0x00A9 => &["libSceMouse.sprx", "libSceMouse.prx"],
        _ => &[],
    }
}

fn loaded() -> std::sync::MutexGuard<'static, BTreeMap<u32, u32>> {
    LOADED.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn is_sysmodule_loaded(id: i32) -> bool {
    loaded().contains_key(&(id as u32))
}

pub fn loaded_sysmodules() -> Vec<i32> {
    loaded().keys().map(|&id| id as i32).collect()
}

fn load(id: u32) {
    let refs = {
        let mut m = loaded();
        let r = m.entry(id).or_insert(0);
        *r += 1;
        *r
    };
    log::debug!("[SYSMODULE] load {id:#06X} (refs={refs})");
}

fn unload(id: u32) -> HleResult {
    let mut m = loaded();
    let Some(refs) = m.get_mut(&id) else { return ERR_UNLOADED };
    *refs -= 1;
    if *refs == 0 {
        m.remove(&id);
    }
    OK
}

fn is_loaded(id: u32) -> HleResult {
    if loaded().contains_key(&id) { OK } else { ERR_UNLOADED }
}

/// (u16 id)
fn cb_load_module(ctx: &mut CpuContext) -> HleResult {
    load(ctx.rdi() as u16 as u32);
    OK
}

/// (u16 id)
fn cb_unload_module(ctx: &mut CpuContext) -> HleResult {
    unload(ctx.rdi() as u16 as u32)
}

/// sceSysmoduleLoadModuleInternal(u32 id)
fn cb_load_module_internal(ctx: &mut CpuContext) -> HleResult {
    load(ctx.rdi() as u32);
    OK
}

/// sceSysmoduleIsLoadedInternal(u32 id): an id without its number (only
/// the internal bit, or 0) is INVALID_VALUE.
fn cb_is_loaded_internal(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as u32;
    if id & 0x7FFF_FFFF == 0 {
        return ERR_INVALID_VALUE;
    }
    is_loaded(id)
}

/// sceSysmoduleUnloadModuleInternal(u32 id)
fn cb_unload_module_internal(ctx: &mut CpuContext) -> HleResult {
    unload(ctx.rdi() as u32)
}

/// (u32 id, argc, argv, flags, int* result)
fn cb_load_module_internal_with_arg(ctx: &mut CpuContext) -> HleResult {
    load(ctx.rdi() as u32);
    let result = ctx.r8();
    if result != 0 && !ctx.write_u32(result, 0) {
        return ERR_INVALID_VALUE;
    }
    OK
}

/// sceSysmoduleUnloadModuleInternalWithArg(u32 id, argc, argv, flags,
/// int* result): id 0 is INVALID_VALUE; the result gets 0 once unloaded.
fn cb_unload_module_internal_with_arg(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as u32;
    if id & 0x7FFF_FFFF == 0 {
        return ERR_INVALID_VALUE;
    }
    let r = unload(id);
    if r != OK {
        return r;
    }
    let result = ctx.r8();
    if result != 0 && !ctx.write_u32(result, 0) {
        return ERR_INVALID_VALUE;
    }
    OK
}

/// (u16 id)
fn cb_is_loaded(ctx: &mut CpuContext) -> HleResult {
    is_loaded(ctx.rdi() as u16 as u32)
}

/// Kernel module handle of a loaded system module: the module loaded from
/// one of its PRX files. A system module served only by HLE has no kernel
/// module, so there is no handle to give.
fn module_handle(id: u32, find: impl Fn(&str) -> Option<i32>) -> Result<i32, HleResult> {
    if id & 0x7FFF_FFFF == 0 {
        log::warn!("[SYSMODULE] GetModuleHandleInternal: invalid id 0");
        return Err(ERR_INVALID_VALUE);
    }
    if !loaded().contains_key(&id) {
        return Err(ERR_UNLOADED);
    }
    let names = known_sysmodule_names(id as i32);
    match names.iter().find_map(|n| find(n)) {
        Some(handle) => Ok(handle),
        None => {
            log::warn!("[SYSMODULE] GetModuleHandleInternal({id:#x}): no kernel module backs this system module");
            Err(ERR_INVALID_VALUE)
        }
    }
}

/// sceSysmoduleGetModuleHandleInternal(u32 id, SceKernelModule* handle)
fn cb_get_module_handle_internal(ctx: &mut CpuContext) -> HleResult {
    let (id, out) = (ctx.rdi() as u32, ctx.rsi());
    if out == 0 {
        log::warn!("[SYSMODULE] GetModuleHandleInternal({id:#x}): null handle pointer");
        return ERR_INVALID_VALUE;
    }
    let handle = match module_handle(id, |name| crate::modules::module_by_file_name(name).map(|m| m.handle)) {
        Ok(h) => h,
        Err(e) => return e,
    };
    if ctx.write_u32(out, handle as u32) { OK } else { ERR_INVALID_VALUE }
}

pub fn register(reg: &mut HleRegistry) {
    reg.register_fn("g8cM39EUZ6o", "sceSysmoduleLoadModule", LIB, cb_load_module);
    reg.register_fn("eR2bZFAAU0Q", "sceSysmoduleUnloadModule", LIB, cb_unload_module);
    reg.register_fn("hHrGoGoNf+s", "sceSysmoduleLoadModuleInternalWithArg", LIB, cb_load_module_internal_with_arg);
    reg.register_fn("fMP5NHUOaMk", "sceSysmoduleIsLoaded", LIB, cb_is_loaded);
    reg.register_fn("39iV5E1HoCk", "sceSysmoduleLoadModuleInternal", LIB, cb_load_module_internal);
    reg.register_fn("vXZhrtJxkGc", "sceSysmoduleUnloadModuleInternal", LIB, cb_unload_module_internal);
    reg.register_fn("aKa6YfBKZs4", "sceSysmoduleUnloadModuleInternalWithArg", LIB, cb_unload_module_internal_with_arg);
    reg.register_fn("D8cuU4d72xM", "sceSysmoduleGetModuleHandleInternal", LIB, cb_get_module_handle_internal);
    reg.register_fn("ynFKQ5bfGks", "sceSysmoduleIsLoadedInternal", LIB, cb_is_loaded_internal);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuRegister;

    #[test]
    fn unload_internal_with_arg() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A7E_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(CpuRegister::Rdi, 0x8000_0000);
        assert_eq!(cb_unload_module_internal_with_arg(&mut ctx), ERR_INVALID_VALUE);
        ctx.set(CpuRegister::Rdi, 0x8000_00F3);
        assert_eq!(cb_unload_module_internal_with_arg(&mut ctx), ERR_UNLOADED);
        load(0x8000_00F3);
        ctx.write_u32(BASE, 7);
        ctx.set(CpuRegister::R8, BASE);
        assert_eq!(cb_unload_module_internal_with_arg(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE), Some(0));
    }

    #[test]
    fn load_bookkeeping() {
        load(0x0095);
        assert!(is_sysmodule_loaded(0x0095));
        assert!(loaded_sysmodules().contains(&0x0095));
        assert_eq!(known_sysmodule_names(0x0095)[0], "libSceIme.sprx");
        assert!(known_sysmodule_names(0x1234).is_empty());
    }

    fn call(f: fn(&mut CpuContext) -> HleResult, id: u32) -> HleResult {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(CpuRegister::Rdi, id as u64);
        f(&mut ctx)
    }

    #[test]
    fn nids_match_names() {
        for (nid, name) in [
            ("g8cM39EUZ6o", "sceSysmoduleLoadModule"),
            ("eR2bZFAAU0Q", "sceSysmoduleUnloadModule"),
            ("hHrGoGoNf+s", "sceSysmoduleLoadModuleInternalWithArg"),
            ("fMP5NHUOaMk", "sceSysmoduleIsLoaded"),
            ("39iV5E1HoCk", "sceSysmoduleLoadModuleInternal"),
            ("vXZhrtJxkGc", "sceSysmoduleUnloadModuleInternal"),
            ("D8cuU4d72xM", "sceSysmoduleGetModuleHandleInternal"),
            ("ynFKQ5bfGks", "sceSysmoduleIsLoadedInternal"),
            ("aKa6YfBKZs4", "sceSysmoduleUnloadModuleInternalWithArg"),
        ] {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// Internal loads share the references of the public ids; a module
    /// without references is UNLOADED for every query and unload.
    #[test]
    fn internal_loads_and_unloaded_status() {
        const FIBER: u32 = 0x0006;
        assert_eq!(call(cb_is_loaded, FIBER), ERR_UNLOADED);
        assert_eq!(call(cb_load_module_internal, FIBER), OK);
        assert_eq!(call(cb_is_loaded, FIBER), OK);
        assert_eq!(call(cb_load_module_internal, FIBER), OK);
        assert_eq!(call(cb_unload_module_internal, FIBER), OK);
        assert_eq!(call(cb_is_loaded, FIBER), OK);
        assert_eq!(call(cb_unload_module_internal, FIBER), OK);
        assert_eq!(call(cb_is_loaded, FIBER), ERR_UNLOADED);
        assert_eq!(call(cb_unload_module_internal, FIBER), ERR_UNLOADED);
        // A 32-bit internal id is not truncated onto a public one.
        assert_eq!(call(cb_load_module_internal, 0x8000_0007), OK);
        assert_eq!(call(cb_is_loaded, 0x0007), ERR_UNLOADED);
        assert_eq!(call(cb_unload_module_internal, 0x8000_0007), OK);
        // sceSysmoduleIsLoadedInternal follows the same references.
        assert_eq!(call(cb_is_loaded_internal, 0x8000_0000), ERR_INVALID_VALUE);
        assert_eq!(call(cb_is_loaded_internal, 0), ERR_INVALID_VALUE);
        assert_eq!(call(cb_is_loaded_internal, FIBER), ERR_UNLOADED);
        assert_eq!(call(cb_load_module_internal, FIBER), OK);
        assert_eq!(call(cb_is_loaded_internal, FIBER), OK);
        assert_eq!(call(cb_unload_module_internal, FIBER), OK);
        assert_eq!(call(cb_is_loaded_internal, FIBER), ERR_UNLOADED);
    }

    /// Port of the guest handle checks: an unloaded module has no handle
    /// and the output is untouched; a loaded one gives the same handle each
    /// time; a module without a kernel image or id 0 is refused.
    #[test]
    fn module_handle_internal() {
        const RUDP: u32 = 0x0021;
        let find = |name: &str| (name == "libSceRudp.prx").then_some(42);
        assert_eq!(module_handle(RUDP, find), Err(ERR_UNLOADED));
        assert_eq!(module_handle(0, find), Err(ERR_INVALID_VALUE));
        assert_eq!(module_handle(0x8000_0000, find), Err(ERR_INVALID_VALUE));
        load(RUDP);
        assert_eq!(module_handle(RUDP, find), Ok(42));
        assert_eq!(module_handle(RUDP, find), Ok(42));
        assert_eq!(module_handle(RUDP, |_| None), Err(ERR_INVALID_VALUE), "HLE-only module");
        assert_eq!(unload(RUDP), OK);
        assert_eq!(module_handle(RUDP, find), Err(ERR_UNLOADED));

        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5FA7_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_u32(BASE, u32::MAX);
        ctx.set(CpuRegister::Rdi, 0x0120);
        ctx.set(CpuRegister::Rsi, BASE);
        assert_eq!(cb_get_module_handle_internal(&mut ctx), ERR_UNLOADED);
        assert_eq!(ctx.read_u32(BASE), Some(u32::MAX));
        ctx.set(CpuRegister::Rsi, 0);
        assert_eq!(cb_get_module_handle_internal(&mut ctx), ERR_INVALID_VALUE);
    }

    #[test]
    fn shared_load_references() {
        // Loading entitlement access (0x113) does not mark AppContent (0xB4) loaded.
        assert_eq!(call(cb_load_module, 0x113), OK);
        assert_eq!(call(cb_is_loaded, 0xB4), ERR_UNLOADED);
        assert_eq!(call(cb_unload_module, 0x113), OK);
        for id in [0x113u32, 0xB4] {
            assert_eq!(call(cb_is_loaded, id), ERR_UNLOADED, "unloaded allows guest init");
            assert_eq!(call(cb_unload_module, id), ERR_UNLOADED, "absent module");
            assert_eq!(call(cb_load_module, id), OK);
            assert_eq!(call(cb_load_module, id), OK);
            assert_eq!(call(cb_is_loaded, id), OK, "repeated loads retain references");
            assert_eq!(call(cb_unload_module, id), OK);
            assert_eq!(call(cb_is_loaded, id), OK, "one unload keeps the other reference");
            assert_eq!(call(cb_unload_module, id), OK);
            assert_eq!(call(cb_is_loaded, id), ERR_UNLOADED, "last unload restores UNLOADED");
        }
        // The internal and public entry points share the same state.
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(CpuRegister::Rdi, 0xB4);
        assert_eq!(cb_load_module_internal_with_arg(&mut ctx), OK);
        assert_eq!(call(cb_is_loaded, 0xB4), OK);
        assert_eq!(call(cb_unload_module, 0xB4), OK);
        assert_eq!(call(cb_is_loaded, 0xB4), ERR_UNLOADED);
    }
}
