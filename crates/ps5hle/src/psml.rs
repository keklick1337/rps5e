// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libScePsml — PlayStation Spectral Super Resolution (MFSR upscaler).
//!
//! The library sizes and validates the upscaler's memory, shared-resource
//! and context objects. The ML upscale itself runs as GPU work that this
//! emulator does not model: dispatches validate their context and succeed
//! without emitting work, so the upscaled target keeps whatever the game
//! (or its fallback path) wrote there.
//!
//! Object layout written into guest memory: shared resources carry a magic
//! word, the direct-memory block list/count, the configuration type and the
//! virtual address start; contexts carry their magic and the shared
//! resources pointer.

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libScePsml";

const fn err(code: u32) -> HleResult {
    code as i32 as i64
}

const ERROR_NOT_INITIALIZED: HleResult = err(0x8A81_0001);
const ERROR_INVALID_OBJECT: HleResult = err(0x8A81_0005);
const ERROR_INVALID_POINTER: HleResult = err(0x8A81_0009);
const ERROR_INVALID_VALUE: HleResult = err(0x8A81_000D);
const ERROR_NULL_OBJECT: HleResult = err(0x8A81_0014);

const SHARED_RESOURCES_MAGIC: u32 = 0xA9C4;
const CONTEXT_MAGIC: u32 = 0x9231;

const MAIN_MEMORY_BLOCK_SIZE: u64 = 0x20_0000;
const MAIN_MEMORY_ALIGNMENT: u64 = 0x20_0000;
const EXTRA_VA_BYTES: u64 = 0x60_0000;
/// Dispatch packet size reported for a valid object.
const DISPATCH_PACKET_DWORDS: u32 = 0x600;

const CONTEXT_SHARED_RESOURCES: u64 = 0x360;
const CONTEXT_STATE: u64 = 0x368;

static INITIALIZED: AtomicBool = AtomicBool::new(false);
static SUPPORTED_MODES: AtomicU32 = AtomicU32::new(0x3);
static SHARED_REF_COUNT: AtomicI32 = AtomicI32::new(0);

/// Virtual address span needed by each configuration type.
fn base_va_bytes(ty: u32) -> u64 {
    match ty {
        0 => 0x0620_0000,
        1 => 0x1820_0000,
        2 => 0x1220_0000,
        _ => 0,
    }
}

fn is_supported_type(ty: u32) -> bool {
    let modes = SUPPORTED_MODES.load(Ordering::Relaxed);
    match ty {
        0 => true,
        1 => modes & 1 != 0,
        2 => modes & 2 != 0,
        _ => false,
    }
}

fn required_block_count(ty: u32) -> u64 {
    match base_va_bytes(ty) {
        0 => 0,
        base => (base + EXTRA_VA_BYTES) / MAIN_MEMORY_BLOCK_SIZE,
    }
}

fn check_initialized() -> Result<(), HleResult> {
    if INITIALIZED.load(Ordering::Relaxed) {
        Ok(())
    } else {
        Err(ERROR_NOT_INITIALIZED)
    }
}

fn has_known_magic(ctx: &CpuContext, object: u64) -> bool {
    object != 0
        && matches!(ctx.read_u32(object), Some(m) if m == SHARED_RESOURCES_MAGIC || m == CONTEXT_MAGIC)
}

/// { blockSize u64, alignment u64, blockCount u64 }
fn write_requirements(ctx: &CpuContext, out: u64, count: u64) -> bool {
    let mut b = [0u8; 24];
    b[0..8].copy_from_slice(&MAIN_MEMORY_BLOCK_SIZE.to_le_bytes());
    b[8..16].copy_from_slice(&MAIN_MEMORY_ALIGNMENT.to_le_bytes());
    b[16..24].copy_from_slice(&count.to_le_bytes());
    ctx.write_bytes(out, &b)
}

fn run(f: impl FnOnce() -> Result<HleResult, HleResult>) -> HleResult {
    f().unwrap_or_else(|e| e)
}

/// scePsmlMfsrInit()
fn cb_init(_ctx: &mut CpuContext) -> HleResult {
    INITIALIZED.store(true, Ordering::Relaxed);
    SUPPORTED_MODES.store(0x3, Ordering::Relaxed);
    SHARED_REF_COUNT.store(0, Ordering::Relaxed);
    0
}

/// scePsmlMfsrGetSharedResourcesInitRequirement(Requirements* out, const Params* params)
/// Params { type u32, reserved u32 }
fn cb_get_shared_resources_init_requirement(ctx: &mut CpuContext) -> HleResult {
    run(|| {
        check_initialized()?;
        let (out, params) = (ctx.rdi(), ctx.rsi());
        if out == 0 || params == 0 {
            return Err(ERROR_INVALID_POINTER);
        }
        let ty = ctx.read_u32(params).ok_or(ERROR_INVALID_POINTER)?;
        if !is_supported_type(ty) {
            return Err(ERROR_INVALID_VALUE);
        }
        if !write_requirements(ctx, out, required_block_count(ty)) {
            return Err(ERROR_INVALID_POINTER);
        }
        Ok(0)
    })
}

/// scePsmlMfsrCreateSharedResources(void* resources, const InitParams* params)
/// InitParams { type u32, reserved u32, blocks ptr, blockCount u64, vaStart u64 }
fn cb_create_shared_resources(ctx: &mut CpuContext) -> HleResult {
    run(|| {
        check_initialized()?;
        let (resources, params) = (ctx.rdi(), ctx.rsi());
        if resources == 0 || params == 0 {
            return Err(ERROR_INVALID_POINTER);
        }
        let p = ctx.read_bytes(params, 32).ok_or(ERROR_INVALID_POINTER)?;
        let ty = u32::from_le_bytes(p[0..4].try_into().unwrap());
        let reserved = u32::from_le_bytes(p[4..8].try_into().unwrap());
        let blocks = u64::from_le_bytes(p[8..16].try_into().unwrap());
        let block_count = u64::from_le_bytes(p[16..24].try_into().unwrap());
        let va_start = u64::from_le_bytes(p[24..32].try_into().unwrap());
        if blocks == 0 {
            return Err(ERROR_INVALID_POINTER);
        }
        if reserved != 0 || !is_supported_type(ty) || block_count < required_block_count(ty) {
            return Err(ERROR_INVALID_VALUE);
        }
        let ok = ctx.write_u32(resources, SHARED_RESOURCES_MAGIC)
            && ctx.write_u64(resources + 0x08, blocks)
            && ctx.write_u64(resources + 0x18, block_count)
            && ctx.write_u32(resources + 0x20, ty)
            && ctx.write_u64(resources + 0x28, va_start);
        if !ok {
            return Err(ERROR_INVALID_POINTER);
        }
        SHARED_REF_COUNT.fetch_add(1, Ordering::Relaxed);
        log::info!("[psml] shared resources {resources:#x}: type {ty}, {block_count} blocks (upscaler not emulated)");
        Ok(0)
    })
}

/// scePsmlMfsrReleaseSharedResources(void* resources)
fn cb_release_shared_resources(ctx: &mut CpuContext) -> HleResult {
    run(|| {
        check_initialized()?;
        if ctx.rdi() == 0 {
            return Err(ERROR_NULL_OBJECT);
        }
        SHARED_REF_COUNT.fetch_sub(1, Ordering::Relaxed);
        Ok(0)
    })
}

/// scePsmlMfsrGetContextBufferRequirement800M3_2(Requirements* out, params)
fn cb_get_context_buffer_requirement(ctx: &mut CpuContext) -> HleResult {
    run(|| {
        check_initialized()?;
        let out = ctx.rdi();
        if out == 0 || !write_requirements(ctx, out, 1) {
            return Err(ERROR_INVALID_POINTER);
        }
        Ok(0)
    })
}

/// scePsmlMfsrCreateContext800M3_2(void* context, const Params* params)
/// Params+0x08 = shared resources.
fn cb_create_context(ctx: &mut CpuContext) -> HleResult {
    run(|| {
        check_initialized()?;
        let (context, params) = (ctx.rdi(), ctx.rsi());
        if context == 0 || params == 0 {
            return Err(ERROR_INVALID_POINTER);
        }
        let shared = ctx.read_u64(params + 8).ok_or(ERROR_INVALID_POINTER)?;
        if shared == 0 {
            return Err(ERROR_INVALID_POINTER);
        }
        let ok = ctx.write_u32(context, CONTEXT_MAGIC)
            && ctx.write_u64(context + CONTEXT_SHARED_RESOURCES, shared)
            && ctx.write_bytes(context + CONTEXT_STATE, &[0]);
        if !ok {
            return Err(ERROR_INVALID_POINTER);
        }
        Ok(0)
    })
}

/// scePsmlMfsrReleaseContext(void* context)
fn cb_release_context(ctx: &mut CpuContext) -> HleResult {
    run(|| {
        check_initialized()?;
        let context = ctx.rdi();
        if !has_known_magic(ctx, context) {
            return Err(ERROR_INVALID_OBJECT);
        }
        ctx.write_u32(context, 0);
        Ok(0)
    })
}

/// scePsmlMfsrGetDispatchMfsrPacketSizeInDwords(const void* object, uint32* sizeInDwords)
fn cb_get_dispatch_packet_size(ctx: &mut CpuContext) -> HleResult {
    run(|| {
        check_initialized()?;
        let (object, out) = (ctx.rdi(), ctx.rsi());
        if !has_known_magic(ctx, object) {
            return Err(ERROR_INVALID_OBJECT);
        }
        if out == 0 || !ctx.write_u32(out, DISPATCH_PACKET_DWORDS) {
            return Err(ERROR_INVALID_POINTER);
        }
        Ok(0)
    })
}

/// scePsmlMfsrGetDispatchMfsrPacket900/1000/1100(void* context, command,
/// params): validated, no GPU work is produced.
fn cb_get_dispatch_packet_900(ctx: &mut CpuContext) -> HleResult {
    run(|| {
        check_initialized()?;
        let (context, command, params) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
        if !has_known_magic(ctx, context) {
            return Err(ERROR_INVALID_OBJECT);
        }
        if command == 0 || params == 0 {
            return Err(ERROR_INVALID_POINTER);
        }
        Ok(0)
    })
}

/// scePsmlMfsrRequestCapture(const void* object)
fn cb_request_capture(ctx: &mut CpuContext) -> HleResult {
    run(|| {
        check_initialized()?;
        if !has_known_magic(ctx, ctx.rdi()) {
            return Err(ERROR_INVALID_OBJECT);
        }
        Ok(0)
    })
}

/// scePsmlMfsrGetMipmapBias(context, float* bias, int width, int height).
///
/// Call site (Demon's Souls eboot): the context comes from the game's
/// upscaler object, `width`/`height` are the render size truncated from
/// floats, `*bias` is zeroed before the call, the return value is ignored and
/// the game uses `min(*bias + 0.5, 0)` as its texture LOD bias. The library's
/// formula is not known (the module is not in any game dump); no upscale runs
/// here, so the bias is 0 and the game's LOD bias stays at its own 0.
fn cb_get_mipmap_bias(ctx: &mut CpuContext) -> HleResult {
    run(|| {
        check_initialized()?;
        let (object, out) = (ctx.rdi(), ctx.rsi());
        if !has_known_magic(ctx, object) {
            return Err(ERROR_INVALID_OBJECT);
        }
        if out == 0 || !ctx.write_u32(out, 0.0f32.to_bits()) {
            return Err(ERROR_INVALID_POINTER);
        }
        Ok(0)
    })
}

/// scePsmlMfsrIsCaptureInProgress(context) and scePsmlMfsrSelectConfig(...):
/// the context is validated. The game reads IsCaptureInProgress as a boolean
/// (`test eax, eax; setnz al`); no capture ever starts here, so a valid
/// context answers 0. SelectConfig's other arguments are not known (no
/// direct call site in the dumps we have).
fn cb_validate_object(ctx: &mut CpuContext) -> HleResult {
    run(|| {
        check_initialized()?;
        if !has_known_magic(ctx, ctx.rdi()) {
            return Err(ERROR_INVALID_OBJECT);
        }
        Ok(0)
    })
}

/// Exports whose symbol names are not known: SHA1 of no candidate name
/// gives these NIDs; they share the behaviour of their versioned siblings.
const UNNAMED_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("IMlj247LdTo", "scePsmlMfsrGetContextBufferRequirement?", cb_get_context_buffer_requirement),
    ("vUk2pWMx3KQ", "scePsmlMfsrCreateContext?", cb_create_context),
];

/// The MFSR2 sub-library (PSSR upscaling): a console without PSSR never
/// initializes it — scePsmlMfsr2Init answers "unsupported console"
/// (firmware disassembly) and every other call answers "not initialized"
/// without touching its arguments.
const MFSR2_NOT_INITIALIZED: HleResult = 0x8A8A_0001u32 as i32 as i64;
const MFSR2_UNSUPPORTED_CONSOLE: HleResult = 0x8A8A_0012u32 as i32 as i64;

fn cb_mfsr2_not_initialized(_ctx: &mut CpuContext) -> HleResult {
    MFSR2_NOT_INITIALIZED
}

fn cb_mfsr2_init_unsupported(_ctx: &mut CpuContext) -> HleResult {
    MFSR2_UNSUPPORTED_CONSOLE
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("3WVD91e12ZQ", "scePsmlMfsrInit", cb_init),
    ("+2KpvixvL6E", "scePsmlMfsrGetSharedResourcesInitRequirement", cb_get_shared_resources_init_requirement),
    ("eWoKNeB6V-k", "scePsmlMfsrCreateSharedResources", cb_create_shared_resources),
    ("jEevBXmagOQ", "scePsmlMfsrReleaseSharedResources", cb_release_shared_resources),
    ("EO9YQXmJEN8", "scePsmlMfsrGetContextBufferRequirement", cb_get_context_buffer_requirement),
    ("sjTyYheKTrU", "scePsmlMfsrGetContextBufferRequirement800M3", cb_get_context_buffer_requirement),
    ("ArakEpzsZo0", "scePsmlMfsrGetContextBufferRequirement800M3_2", cb_get_context_buffer_requirement),
    ("VGjrQa-WqdU", "scePsmlMfsrGetContextBufferRequirement1100", cb_get_context_buffer_requirement),
    ("2ecEbQaf9VU", "scePsmlMfsrCreateContext", cb_create_context),
    ("gxv3i+MTEzU", "scePsmlMfsrCreateContext800M3_2", cb_create_context),
    ("JaLBe0P3jSU", "scePsmlMfsrReleaseContext", cb_release_context),
    ("AHalTX9wFZY", "scePsmlMfsrGetDispatchMfsrPacketSizeInDwords", cb_get_dispatch_packet_size),
    ("RUNLFro+qok", "scePsmlMfsrGetDispatchMfsrPacket900", cb_get_dispatch_packet_900),
    ("s2psNHUIdjk", "scePsmlMfsrGetDispatchMfsrPacket1000", cb_get_dispatch_packet_900),
    ("94iBp3KvIuI", "scePsmlMfsrGetDispatchMfsrPacket1100", cb_get_dispatch_packet_900),
    ("GJY0MvuTcs8", "scePsmlMfsrRequestCapture", cb_request_capture),
    ("GHna9-DvnUk", "scePsmlMfsrGetMipmapBias", cb_get_mipmap_bias),
    ("LXq+6mIxpCw", "scePsmlMfsrIsCaptureInProgress", cb_validate_object),
    ("FSGaTQze0UY", "scePsmlMfsrSelectConfig", cb_validate_object),
    ("o+NM86gEwFE", "scePsmlMfsr2Init", cb_mfsr2_init_unsupported),
    ("1ic5q-kdOsc", "scePsmlMfsr2GetSharedResourcesInitRequirement", cb_mfsr2_not_initialized),
    ("gMduXCLYrNg", "scePsmlMfsr2CreateSharedResources", cb_mfsr2_not_initialized),
    ("0GAw7SmkwII", "scePsmlMfsr2ReleaseSharedResources", cb_mfsr2_not_initialized),
    ("5z2gBlqxJ+0", "scePsmlMfsr2GetContextInitRequirement", cb_mfsr2_not_initialized),
    ("m9JLPc3wOQw", "scePsmlMfsr2CreateContext", cb_mfsr2_not_initialized),
    ("lrJwpLjKXRc", "scePsmlMfsr2GetDispatchPackets", cb_mfsr2_not_initialized),
    ("ZLL31lzzxr4", "scePsmlMfsr2GetDispatchPacketsSizeInDwords", cb_mfsr2_not_initialized),
    ("Kyy1baXgaVU", "scePsmlMfsr2ReleaseContext", cb_mfsr2_not_initialized),
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
    }

    #[test]
    fn shared_resources_and_context_lifecycle() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C07_0000_0000;
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        assert_eq!(cb_init(&mut ctx), 0);

        // Requirement for type 0: (0x6200000 + 0x600000) / 0x200000 = 52 blocks.
        ctx.write_u64(BASE, 0);
        ctx.set(Rdi, BASE + 0x100);
        ctx.set(Rsi, BASE);
        assert_eq!(cb_get_shared_resources_init_requirement(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE + 0x110), Some(52));
        ctx.write_u32(BASE, 7);
        assert_eq!(cb_get_shared_resources_init_requirement(&mut ctx), ERROR_INVALID_VALUE);

        // Too few blocks, then enough.
        let mut p = Vec::new();
        p.extend_from_slice(&0u32.to_le_bytes());
        p.extend_from_slice(&0u32.to_le_bytes());
        p.extend_from_slice(&(BASE + 0x800).to_le_bytes());
        p.extend_from_slice(&10u64.to_le_bytes());
        p.extend_from_slice(&0x1000_0000u64.to_le_bytes());
        ctx.write_bytes(BASE + 0x200, &p);
        ctx.set(Rdi, BASE + 0x1000);
        ctx.set(Rsi, BASE + 0x200);
        assert_eq!(cb_create_shared_resources(&mut ctx), ERROR_INVALID_VALUE);
        ctx.write_u64(BASE + 0x210, 52);
        assert_eq!(cb_create_shared_resources(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x1000), Some(SHARED_RESOURCES_MAGIC));

        // Context bound to the shared resources.
        ctx.write_u64(BASE + 0x308, BASE + 0x1000);
        ctx.set(Rdi, BASE + 0x2000);
        ctx.set(Rsi, BASE + 0x300);
        assert_eq!(cb_create_context(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE + 0x2000 + CONTEXT_SHARED_RESOURCES), Some(BASE + 0x1000));
        ctx.set(Rsi, BASE + 0x400);
        assert_eq!(cb_get_dispatch_packet_size(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x400), Some(DISPATCH_PACKET_DWORDS));
        ctx.set(Rdx, BASE + 0x500);
        assert_eq!(cb_get_dispatch_packet_900(&mut ctx), 0);
        ctx.set(Rsi, BASE + 0x600);
        ctx.write_u32(BASE + 0x600, 0xFFFF_FFFF);
        assert_eq!(cb_get_mipmap_bias(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x600), Some(0));
        assert_eq!(cb_validate_object(&mut ctx), 0);
        assert_eq!(cb_release_context(&mut ctx), 0);
        assert_eq!(cb_release_context(&mut ctx), ERROR_INVALID_OBJECT);
        assert_eq!(cb_validate_object(&mut ctx), ERROR_INVALID_OBJECT);
        assert_eq!(cb_get_mipmap_bias(&mut ctx), ERROR_INVALID_OBJECT);
        ctx.set(Rdi, BASE + 0x1000);
        assert_eq!(cb_release_shared_resources(&mut ctx), 0);
    }
}
