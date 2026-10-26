// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceVideodec (the first-generation decoder interface): decoders are
//! created and tracked by handle, but no picture is ever produced; decoding
//! goes through libSceVideodec2 (`videodec2.rs`) on the host decoder.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceVideodec";

const OK: HleResult = 0;
const ERROR_INVALID_ARG: HleResult = 0x8062_0801u32 as i32 as i64;

static DECODERS: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

fn with_decoders<R>(f: impl FnOnce(&mut HashSet<u64>) -> R) -> R {
    let mut g = DECODERS.lock().unwrap_or_else(|p| p.into_inner());
    f(g.get_or_insert_with(HashSet::new))
}

fn is_decoder(handle: u64) -> bool {
    with_decoders(|d| d.contains(&handle))
}

/// sceVideodecCreateDecoder(config, resource, u64* handle)
fn cb_create_decoder(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdx();
    if out == 0 {
        return ERROR_INVALID_ARG;
    }
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed) + 1;
    if !ctx.write_u64(out, handle) {
        return ERROR_INVALID_ARG;
    }
    with_decoders(|d| d.insert(handle));
    log::info!("[videodec] decoder {handle:#x} created (no pictures are produced)");
    OK
}

/// sceVideodecDecode(handle, ...): accepted, no picture comes out.
/// sceVideodecFlush(handle, ...): nothing is buffered.
fn cb_decode_or_flush(ctx: &mut CpuContext) -> HleResult {
    if is_decoder(ctx.rdi()) { OK } else { ERROR_INVALID_ARG }
}

/// sceVideodecDeleteDecoder(handle)
fn cb_delete_decoder(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    with_decoders(|d| d.remove(&handle));
    OK
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("qkgRiwHyheU", "sceVideodecCreateDecoder", cb_create_decoder),
    ("q0W5GJMovMs", "sceVideodecDecode", cb_decode_or_flush),
    ("jeigLlKdp5I", "sceVideodecFlush", cb_decode_or_flush),
    ("U0kpGF1cl90", "sceVideodecDeleteDecoder", cb_delete_decoder),
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
    fn decoder_lifecycle() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5DA5_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdx, 0);
        assert_eq!(cb_create_decoder(&mut ctx), ERROR_INVALID_ARG);
        ctx.set(Rdx, BASE);
        assert_eq!(cb_create_decoder(&mut ctx), OK);
        let handle = ctx.read_u64(BASE).unwrap();
        assert_ne!(handle, 0);
        ctx.set(Rdi, handle);
        assert_eq!(cb_decode_or_flush(&mut ctx), OK);
        assert_eq!(cb_delete_decoder(&mut ctx), OK);
        assert_eq!(cb_decode_or_flush(&mut ctx), ERROR_INVALID_ARG);
    }
}
