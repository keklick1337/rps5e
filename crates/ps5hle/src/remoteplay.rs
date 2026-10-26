// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceRemoteplay HLE: no Remote Play client ever connects.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceRemoteplay";
const OK: HleResult = 0;
/// The library's answer to a NULL status pointer.
const ERR_NULL_STATUS: HleResult = -1;
const ERR_MEMORY_FAULT: HleResult = 0x8002_000Eu32 as i32 as i64;

/// SCE_REMOTEPLAY_CONNECTION_STATUS_DISCONNECT.
const CONNECTION_STATUS_DISCONNECT: i32 = 0;

static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// sceRemoteplayInitialize(void* heap, size_t heapSize)
fn cb_initialize(_ctx: &mut CpuContext) -> HleResult {
    INITIALIZED.store(true, Ordering::Relaxed);
    OK
}

fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    INITIALIZED.store(false, Ordering::Relaxed);
    OK
}

/// sceRemoteplayGetConnectionStatus(userId, int* status)
fn cb_get_connection_status(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return ERR_NULL_STATUS;
    }
    if ctx.write_bytes(out, &CONNECTION_STATUS_DISCONNECT.to_le_bytes()) { OK } else { ERR_MEMORY_FAULT }
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("k1SwgkMSOM8", "sceRemoteplayInitialize", cb_initialize),
    ("BOwybKVa3Do", "sceRemoteplayTerminate", cb_terminate),
    ("g3PNjYKWqnQ", "sceRemoteplayGetConnectionStatus", cb_get_connection_status),
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
    fn never_connected() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C0C_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_u32(BASE, 0xFFFF_FFFF);
        ctx.set(Rsi, BASE);
        assert_eq!(cb_get_connection_status(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE), Some(0));
        ctx.set(Rsi, 0);
        assert_eq!(cb_get_connection_status(&mut ctx), ERR_NULL_STATUS);
    }
}
