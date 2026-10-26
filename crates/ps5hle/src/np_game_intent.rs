// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceNpGameIntent HLE: no game intents are ever delivered offline.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};

const OK: HleResult = 0;

static INITIALIZED: AtomicBool = AtomicBool::new(false);

fn cb_initialize(_ctx: &mut CpuContext) -> HleResult {
    INITIALIZED.store(true, Ordering::Relaxed);
    OK
}

fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    INITIALIZED.store(false, Ordering::Relaxed);
    OK
}

const ERR_NOT_INITIALIZED: HleResult = 0x8055_3802u32 as i32 as i64;
const ERR_INVALID_ARGUMENT: HleResult = 0x8055_3804u32 as i32 as i64;
const ERR_INTENT_NOT_FOUND: HleResult = 0x8055_3806u32 as i32 as i64;
const ERR_VALUE_NOT_FOUND: HleResult = 0x8055_3807u32 as i32 as i64;
const ERR_MEMORY_FAULT: HleResult = 0x8002_000Eu32 as i32 as i64;

/// SceNpGameIntentInfo: size_t size @0, userId @8, intentType[33] @0xC,
/// pad[7], reserved[256], intentData (16393 + 7 bytes) @0x134; 16704 bytes.
const INTENT_INFO_SIZE: usize = 16704;
const INTENT_INFO_USER_ID_OFFSET: u64 = 8;
const INTENT_TYPE_OFFSET: u64 = 0xC;
const INTENT_TYPE_SIZE: usize = 33;
const INTENT_DATA_OFFSET: u64 = 0x134;
const INTENT_DATA_SIZE: usize = 16392;
const INVALID_USER_ID: i32 = -1;

/// sceNpGameIntentReceiveIntent(SceNpGameIntentInfo*): no intent is ever
/// delivered (no system UI / activity cards). The library must be
/// initialised (checked before the arguments) and `size` must be exactly the
/// structure size.
fn cb_receive_intent(ctx: &mut CpuContext) -> HleResult {
    if !INITIALIZED.load(Ordering::Relaxed) {
        return ERR_NOT_INITIALIZED;
    }
    let info = ctx.rdi();
    if info == 0 || ctx.read_u64(info) != Some(INTENT_INFO_SIZE as u64) {
        return ERR_INVALID_ARGUMENT;
    }
    // Only the user, type and data are cleared; size and reserved stay.
    if !ctx.write_u32(info + INTENT_INFO_USER_ID_OFFSET, INVALID_USER_ID as u32)
        || !ctx.write_bytes(info + INTENT_TYPE_OFFSET, &[0u8; INTENT_TYPE_SIZE])
        || !ctx.write_bytes(info + INTENT_DATA_OFFSET, &vec![0u8; INTENT_DATA_SIZE])
    {
        return ERR_MEMORY_FAULT;
    }
    ERR_INTENT_NOT_FOUND
}

/// sceNpGameIntentGetPropertyValueString(const SceNpGameIntentData*, key,
/// char* valueBuf, size_t bufSize): intent data never carries properties.
/// Not initialised wins over bad arguments; an empty key is invalid; the
/// buffer is written only when every argument is valid.
fn cb_get_property_value_string(ctx: &mut CpuContext) -> HleResult {
    if !INITIALIZED.load(Ordering::Relaxed) {
        return ERR_NOT_INITIALIZED;
    }
    let (data, key, buf, size) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    if data == 0 || key == 0 || buf == 0 || size == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    match ctx.read_bytes(key, 1) {
        Some(b) if b[0] != 0 => {}
        Some(_) => return ERR_INVALID_ARGUMENT,
        None => return ERR_MEMORY_FAULT,
    }
    if !ctx.write_bytes(buf, &[0]) {
        return ERR_MEMORY_FAULT;
    }
    ERR_VALUE_NOT_FOUND
}

/// sceNpAppLauncherInitialize(initParam) / sceNpAppLauncherTerminate():
/// the launcher keeps no state here (launching another title is not
/// available); the parameter is not looked at.
fn cb_app_launcher_lifecycle(_ctx: &mut CpuContext) -> HleResult {
    0
}

const EXPORTS: &[(&str, &str, crate::registry::HleCallback)] = &[
    ("jEIXUAr9XE8", "sceNpGameIntentReceiveIntent", cb_receive_intent),
    ("rPl0INNc-M8", "sceNpGameIntentGetPropertyValueString", cb_get_property_value_string),
];

pub fn register(reg: &mut HleRegistry) {
    const LIB: &str = "libSceNpGameIntent";
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
    reg.register_fn("m87BHxt-H60", "sceNpGameIntentInitialize", LIB, cb_initialize);
    reg.register_fn("0HBYxYAjmf0", "sceNpGameIntentTerminate", LIB, cb_terminate);
    reg.register_fn("od0pFsDoez0", "sceNpAppLauncherInitialize", LIB, cb_app_launcher_lifecycle);
    reg.register_fn("sy3PioM8TPE", "sceNpAppLauncherTerminate", LIB, cb_app_launcher_lifecycle);
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
    fn app_launcher() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        assert_eq!(cb_app_launcher_lifecycle(&mut ctx), 0);
        for name in ["sceNpAppLauncherInitialize", "sceNpAppLauncherTerminate"] {
            let mut reg = HleRegistry::new();
            register(&mut reg);
            assert!(reg.lookup(&ps5nid::compute_nid(name)).is_some(), "{name}");
        }
    }

    /// One test drives the shared initialised flag through its whole life,
    /// so no other test may touch it.
    #[test]
    fn intents_lifecycle_and_arguments() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A6C_0000_0000;
        vm.map(BASE, 0x5000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (key, empty, buf) = (BASE + 0x4800, BASE + 0x4810, BASE + 0x4900);
        ctx.write_bytes(key, b"k\0");
        ctx.write_bytes(empty, b"\0");
        ctx.write_u64(BASE, INTENT_INFO_SIZE as u64);
        ctx.write_bytes(BASE + 8, &[0xAA; 64]);

        // Before initialisation every call reports it, whatever the arguments.
        INITIALIZED.store(false, Ordering::Relaxed);
        for info in [0, BASE] {
            ctx.set(Rdi, info);
            assert_eq!(cb_receive_intent(&mut ctx), ERR_NOT_INITIALIZED);
            ctx.set(Rsi, key);
            ctx.set(Rdx, buf);
            ctx.set(Rcx, 16);
            assert_eq!(cb_get_property_value_string(&mut ctx), ERR_NOT_INITIALIZED);
        }
        assert_eq!(ctx.read_u32(BASE + 8), Some(0xAAAA_AAAA));

        assert_eq!(cb_initialize(&mut ctx), OK);
        ctx.set(Rdi, 0);
        assert_eq!(cb_receive_intent(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.set(Rdi, BASE);
        for size in [0u64, 0x413F, 0x4141, 0x560] {
            ctx.write_u64(BASE, size);
            assert_eq!(cb_receive_intent(&mut ctx), ERR_INVALID_ARGUMENT, "size {size:#x}");
            assert_eq!(ctx.read_u32(BASE + 8), Some(0xAAAA_AAAA));
        }
        ctx.write_u64(BASE, INTENT_INFO_SIZE as u64);
        assert_eq!(cb_receive_intent(&mut ctx), ERR_INTENT_NOT_FOUND);
        assert_eq!(ctx.read_u64(BASE), Some(INTENT_INFO_SIZE as u64));
        assert_eq!(ctx.read_u32(BASE + 8), Some(u32::MAX));
        assert_eq!(ctx.read_bytes(BASE + 12, 33).unwrap(), vec![0; 33]);
        // Bytes after the type (padding / reserved) are left alone.
        assert_eq!(ctx.read_bytes(BASE + 45, 19).unwrap(), vec![0xAA; 19]);
        assert_eq!(ctx.read_bytes(BASE + INTENT_INFO_SIZE as u64 - 8, 8).unwrap(), vec![0; 8]);

        // Bad arguments leave the buffer alone.
        let data = BASE + INTENT_DATA_OFFSET;
        for (d, k, b, n) in [(0, key, buf, 16), (data, 0, buf, 16), (data, empty, buf, 16), (data, key, 0, 16), (data, key, buf, 0)] {
            ctx.write_bytes(buf, b"X");
            ctx.set(Rdi, d);
            ctx.set(Rsi, k);
            ctx.set(Rdx, b);
            ctx.set(Rcx, n);
            assert_eq!(cb_get_property_value_string(&mut ctx), ERR_INVALID_ARGUMENT);
            assert_eq!(ctx.read_bytes(buf, 1).unwrap(), b"X");
        }
        ctx.set(Rdi, data);
        ctx.set(Rsi, key);
        ctx.set(Rdx, buf);
        for n in [16, 1] {
            ctx.write_bytes(buf, b"Y");
            ctx.set(Rcx, n);
            assert_eq!(cb_get_property_value_string(&mut ctx), ERR_VALUE_NOT_FOUND);
            assert_eq!(ctx.read_bytes(buf, 1).unwrap(), vec![0]);
        }

        assert_eq!(cb_terminate(&mut ctx), OK);
        assert_eq!(cb_get_property_value_string(&mut ctx), ERR_NOT_INITIALIZED);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_receive_intent(&mut ctx), ERR_NOT_INITIALIZED);
    }
}
