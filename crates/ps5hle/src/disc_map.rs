// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceDiscMap: titles installed to internal storage ask, on nearly every
//! file access, whether a read has to go to the Blu-ray drive. The whole
//! application is on the HDD here, so every request is resident and no
//! disc mapping exists.

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceDiscMap";

const ERROR_INVALID_ARGUMENT: HleResult = 0x8110_0001u32 as i32 as i64;
const ERROR_MEMORY_FAULT: HleResult = 0x8002_000Eu32 as i32 as i64;

fn trace(ctx: &CpuContext, name: &str) {
    if log::log_enabled!(log::Level::Trace) {
        let path = ctx.read_cstring(ctx.rdi(), 1024).unwrap_or_default();
        log::trace!("[discmap] {name} path={path} offset={:#x} size={:#x}", ctx.rsi(), ctx.rdx());
    }
}

/// sceDiscMapIsRequestOnHDD(const char* path, u64 offset, u64 size, int* onHdd)
fn cb_is_request_on_hdd(ctx: &mut CpuContext) -> HleResult {
    let (path, out) = (ctx.rdi(), ctx.rcx());
    if path == 0 || out == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    if !ctx.write_u32(out, 1) {
        return ERROR_MEMORY_FAULT;
    }
    trace(ctx, "sceDiscMapIsRequestOnHDD");
    0
}

/// The two unnamed mapping queries (path, offset, size, u64* flags,
/// u64* out1, u64* out2): no disc mapping, all three outputs are 0.
fn cb_mapping_query(ctx: &mut CpuContext) -> HleResult {
    let (path, flags, out1, out2) = (ctx.rdi(), ctx.rcx(), ctx.r8(), ctx.r9());
    if path == 0 || flags == 0 || out1 == 0 || out2 == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    if !(ctx.write_u64(flags, 0) && ctx.write_u64(out1, 0) && ctx.write_u64(out2, 0)) {
        return ERROR_MEMORY_FAULT;
    }
    trace(ctx, "mapping query");
    0
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[("lbQKqsERhtE", "sceDiscMapIsRequestOnHDD", cb_is_request_on_hdd)];

/// Exports whose symbol names are not known (catalog names shown).
const UNNAMED_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("fJgP+wqifno", "sceDiscMapUnknownFJgP", cb_mapping_query),
    ("ioKMruft1ek", "sceDiscMapUnknownIoKM", cb_mapping_query),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS.iter().chain(UNNAMED_EXPORTS) {
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
        // The unnamed ones carry the catalog's names.
        let db = ps5nid::builtin_exports();
        for &(nid, name, _) in UNNAMED_EXPORTS {
            assert!(db.iter().any(|e| e.nid == nid && e.name == name), "{nid} not catalogued as {name}");
        }
    }

    #[test]
    fn everything_is_on_hdd() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5DA4_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, b"/app0/data.pak\0");
        for (r, v) in [(Rdi, BASE), (Rsi, 0), (Rdx, 0x1000), (Rcx, BASE + 0x100)] {
            ctx.set(r, v);
        }
        assert_eq!(cb_is_request_on_hdd(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x100), Some(1));
        ctx.set(Rcx, 0);
        assert_eq!(cb_is_request_on_hdd(&mut ctx), ERROR_INVALID_ARGUMENT);

        ctx.write_bytes(BASE + 0x200, &[0xFF; 24]);
        for (r, v) in [(Rcx, BASE + 0x200), (R8, BASE + 0x208), (R9, BASE + 0x210)] {
            ctx.set(r, v);
        }
        assert_eq!(cb_mapping_query(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x200, 24).unwrap(), vec![0; 24]);
        ctx.set(R9, 0);
        assert_eq!(cb_mapping_query(&mut ctx), ERROR_INVALID_ARGUMENT);
    }
}
