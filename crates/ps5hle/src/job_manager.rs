// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceJobManager HLE.
//!
//! The job manager is a Sony runtime that executes job graphs on a pool of
//! CPU worker threads with GPU sequence submission. The worker engine itself
//! is not implemented; what is implemented is the observable surface that a
//! title touches before any job runs: construction, memory-requirement
//! calculation, initialization validation and memory-block layout, and
//! shutdown. Behavior was recovered from a retail `libSceJobManager.prx`
//! shipped with a game (decrypted module), transfered as behavior only.
//!
//! Layout written by initialize (guest memory, little endian):
//! ```text
//! memory + 0x00: u16 0x4D4A ('JM'), u16 impl_offset,
//!           u32 0x0500_0001, u32 memory_size, u32 0
//! impl region: 712 bytes zeroed (engine state; worker fields stay zero)
//! impl + 0x20: sequence factory interface table (8 priorities x 64 bytes)
//! ```
//!
//! `calculateRequiredMemorySize` is exact for the single-affinity-mask
//! configurations (params.mode 0 and mode 1 with second_mask_count == 0).
//! The dual-mask mode-1 configuration additionally needs a per-thread GPU
//! resource size the module measures at runtime by creating worker threads
//! and querying the graphics driver; without the engine that term cannot be
//! reproduced, so the configuration is reported as unsupported (size 0),
//! matching the pre-HLE behavior.

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceJobManager";

// Module error space 0x8158_10xx.
const fn err(code: u32) -> HleResult {
    code as i32 as i64
}
const ERR_INVALID_ARG: HleResult = err(0x8158_1000);
const ERR_NULL_MEMORY: HleResult = err(0x8158_1001);
const ERR_INSUFFICIENT_MEMORY: HleResult = err(0x8158_1002);
const ERR_NOT_INITIALIZED: HleResult = err(0x8158_1010);
const ERR_ALREADY_INITIALIZED: HleResult = err(0x8158_1020);
const ERR_NOT_ALIGNED: HleResult = err(0x8158_1030);

/// Header magic and flags written at the start of the memory block.
const HDR_MAGIC: u16 = 0x4D4A;
const HDR_FLAGS: u32 = 0x0500_0001;
/// Total size of the implementation region carved after the header.
const IMPL_SIZE: u64 = 712;

/// Parameters of `calculateRequiredMemorySize` (guest layout, i32 each).
#[derive(Clone, Copy)]
pub struct MemorySizeQueryParams {
    pub mode: u32,
    pub version: u32,
    pub thread_count: u32,
    pub max_sequences: u32,
    pub max_dependencies: u32,
    pub max_priorities: u32,
    pub max_tags: u32,
    pub second_mask_thread_count: u32,
    pub affinity_mask: u64,
}

impl MemorySizeQueryParams {
    fn from_guest(ctx: &CpuContext, addr: u64) -> Option<Self> {
        Some(Self {
            mode: ctx.read_u32(addr)?,
            version: ctx.read_u32(addr + 4)?,
            thread_count: ctx.read_u32(addr + 8)?,
            max_sequences: ctx.read_u32(addr + 12)?,
            max_dependencies: ctx.read_u32(addr + 16)?,
            max_priorities: ctx.read_u32(addr + 20)?,
            max_tags: ctx.read_u32(addr + 24)?,
            second_mask_thread_count: ctx.read_u32(addr + 28)?,
            affinity_mask: ctx.read_u64(addr + 0x20)?,
        })
    }
}

fn popcnt(v: u64) -> u64 {
    v.count_ones() as u64
}

/// Exact port of `JobManager::calculateRequiredMemorySize`. Returns 0 for
/// parameter sets the original rejects.
pub fn calculate_required_memory_size(p: &MemorySizeQueryParams) -> u64 {
    if p.mode > 1 || !(1..=13).contains(&p.version) || p.thread_count > 0x20 {
        return 0;
    }
    let threads = p.thread_count as u64;
    let second = p.second_mask_thread_count as u64;
    let has_second = p.mode == 1 && second != 0;
    let total_threads = threads + second;

    if p.mode == 1 {
        if second > 0x20 || second + threads > 0x20 {
            return 0;
        }
        let pop = popcnt(p.affinity_mask);
        // Every second-mask thread needs a set affinity bit, and the mask
        // must stay inside the low 56 bits the module accepts.
        if pop < second || p.affinity_mask > 0x00FF_FFFF_FFFF_FFFF {
            return 0;
        }
    } else if threads == 0 {
        return 0;
    }

    if p.max_sequences as u64 > 0x7FFF * total_threads
        || p.max_dependencies as u64 > 1023 * total_threads
        || p.max_priorities as u64 > 255 * total_threads
        || p.max_tags as u64 > 63 * total_threads
    {
        return 0;
    }

    // Fixed per-block overhead: version-dependent, rounded through the same
    // step sequence as the original.
    let base = 72 * p.version as u64 + 791;
    let step1 = ((base & !0x3F) + 319) & 0x1_FFFF_FFFFC0;
    let step2 = (step1 + 191) & 0x3_FFFF_FFFFC0;
    let step3 = (step2 + 319) & 0x7_FFFF_FFFFC0;

    if has_second {
        // The dual-mask layout adds popcnt(mask) * per_thread_gpu_bytes,
        // where the per-thread size is measured from the live driver at
        // runtime (worker-thread and GPU queue bookkeeping). Without the
        // engine the term is unknown: report the configuration as
        // unsupported instead of inventing a size.
        return 0;
    }

    let tail = step3 + 576 * total_threads;
    ((tail + 2111) & !0x3F)
        + 72 * (p.max_tags as u64 + p.max_priorities as u64 + total_threads + p.max_dependencies as u64 + 1)
}

/// JobManager::JobManager() — zeroes the handle object and returns the
/// embedded interface pointer (this + 0x10) as the C++ return value.
fn cb_ctor(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        // The original stores its own vtable at +0x10 and links the deleter
        // at +0x30 to this+0x10; without the engine both slots hold zero.
        ctx.write_u64(this, 0);
        ctx.write_u64(this + 0x10, 0);
        ctx.write_u64(this + 0x30, this + 0x10);
        (this + 0x10) as i64
    } else {
        0
    }
}

/// JobManager::~JobManager() — same state transition as shutdown.
fn cb_dtor(ctx: &mut CpuContext) -> HleResult {
    shutdown_impl(ctx)
}

/// JobManager::initialize(memory, memory_size, version, gpu_affinity_count,
/// config_params) — validates the arguments and carves the memory block.
fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    let memory = ctx.rsi();
    let memory_size = ctx.rdx();
    let version = ctx.rcx() as u32;
    let mask_count = ctx.r8() as u32;
    let config = ctx.r9();

    if this == 0 {
        return ERR_INVALID_ARG;
    }
    if ctx.read_u64(this).unwrap_or(0) != 0 {
        return ERR_ALREADY_INITIALIZED;
    }
    if memory == 0 {
        return ERR_NULL_MEMORY;
    }
    if memory & 0xFF != 0 {
        return ERR_NOT_ALIGNED;
    }
    if memory_size == 0 || version == 0 || version > 13 || mask_count == 0 {
        return ERR_INVALID_ARG;
    }
    if mask_count > 0x20 {
        return ERR_INVALID_ARG;
    }
    if let Some(mode) = (config != 0).then(|| ctx.read_u32(config)).flatten() {
        if mode > 1 {
            return ERR_INVALID_ARG;
        }
    }

    let end = memory + memory_size;
    // Header at `memory` (256-byte aligned input), implementation region
    // after it, preferring the 16-aligned slot that fits all 712 bytes.
    let preferred = (memory + 23) & !0xF;
    let (impl_addr, min_end) = if preferred + IMPL_SIZE <= end {
        (preferred, preferred + IMPL_SIZE)
    } else {
        (memory + 16, memory + 16 + IMPL_SIZE)
    };
    if min_end > end {
        return ERR_INSUFFICIENT_MEMORY;
    }

    // Zero the implementation region (engine state stays zero without the
    // worker pool; the original zeroes it in two steps with the same effect
    // for every field a title can observe).
    let mut cur = impl_addr;
    while cur < impl_addr + IMPL_SIZE {
        if !ctx.write_u64(cur, 0) {
            return ERR_INSUFFICIENT_MEMORY;
        }
        cur += 8;
    }

    // Block header.
    if !write_u16(ctx, memory, HDR_MAGIC)
        || !write_u16(ctx, memory + 2, (impl_addr - memory) as u16)
        || !ctx.write_u32(memory + 4, HDR_FLAGS)
        || !ctx.write_u32(memory + 8, memory_size as u32)
        || !ctx.write_u32(memory + 12, 0)
    {
        return ERR_INSUFFICIENT_MEMORY;
    }

    ctx.write_u64(this, impl_addr);
    0
}

fn write_u16(ctx: &CpuContext, addr: u64, val: u16) -> bool {
    // Read-modify-write keeps the neighboring u16 intact.
    match ctx.read_u32(addr & !3) {
        Some(word) => {
            let shift = (addr & 3) * 8;
            let mask = 0xFFFFu32 << shift;
            ctx.write_u32(addr & !3, (word & !mask) | ((val as u32) << shift))
        }
        None => false,
    }
}

fn shutdown_impl(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this == 0 {
        return ERR_INVALID_ARG;
    }
    if ctx.read_u64(this).unwrap_or(0) == 0 {
        return ERR_NOT_INITIALIZED;
    }
    ctx.write_u64(this, 0);
    0
}

/// JobManager::shutdown().
fn cb_shutdown(ctx: &mut CpuContext) -> HleResult {
    shutdown_impl(ctx)
}

/// sce::Job::JobManager::calculateRequiredMemorySize(params) — exact for
/// single-mask configurations, 0 (unsupported) for the dual-mask one.
fn cb_calculate(ctx: &mut CpuContext) -> HleResult {
    let params = ctx.rdi();
    if params == 0 {
        return 0;
    }
    match MemorySizeQueryParams::from_guest(ctx, params) {
        Some(p) => calculate_required_memory_size(&p) as i64,
        None => 0,
    }
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    (
        "cwMzzdesLlY",
        "_ZN3sce3Job10JobManagerC1Ev",
        cb_ctor,
    ),
    (
        "HciaSIulWeo",
        "_ZN3sce3Job10JobManagerC2Ev",
        cb_ctor,
    ),
    (
        "4g1ee34TMH8",
        "_ZN3sce3Job10JobManagerD1Ev",
        cb_dtor,
    ),
    (
        "djt40NyfD-U",
        "_ZN3sce3Job10JobManagerD2Ev",
        cb_dtor,
    ),
    (
        "Nm2BhAwYNFk",
        "_ZN3sce3Job10JobManager27calculateRequiredMemorySizeEPKNS1_21MemorySizeQueryParamsE",
        cb_calculate,
    ),
    (
        "d205iUFMyyE",
        "_ZN3sce3Job10JobManager10initializeEPvmjjPKNS1_12ConfigParamsE",
        cb_initialize,
    ),
    (
        "GS9DSbzC+MY",
        "_ZN3sce3Job10JobManager10initializeEPvmjjjmPKNS1_12ConfigParamsE",
        cb_initialize,
    ),
    (
        "chxnvsxxfBY",
        "_ZN3sce3Job10JobManager8shutdownEv",
        cb_shutdown,
    ),
    // getSequenceFactoryInterface / getUnifiedSequenceFactoryInterface and
    // setPersistentThreadCount return live engine state (factory tables and
    // the thread budget); without the worker engine they intentionally stay
    // unregistered so the generic stubs answer "unavailable" (0).
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_u16_at(ctx: &CpuContext, addr: u64) -> Option<u16> {
        let shift = (addr & 3) * 8;
        ctx.read_u32(addr & !3).map(|w| (w >> shift) as u16)
    }

    fn params(mode: u32, version: u32, threads: u32) -> MemorySizeQueryParams {
        MemorySizeQueryParams {
            mode,
            version,
            thread_count: threads,
            max_sequences: 0,
            max_dependencies: 0,
            max_priorities: 0,
            max_tags: 0,
            second_mask_thread_count: 0,
            affinity_mask: 0,
        }
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn calc_minimal_mode0() {
        // Hand-derived from the decompiled formula:
        // base = 72*1+791 = 863; -> 832 -> 1088 -> 1216 -> 1472;
        // tail = 1472 + 576 = 2048; (2048+2111)&~63 = 4096;
        // + 72*(0+0+1+0+1) = 144 -> 4240.
        assert_eq!(calculate_required_memory_size(&params(0, 1, 1)), 4240);
    }

    #[test]
    fn calc_version13_two_threads() {
        // base = 72*13+791 = 1727; -> 1664 -> 1920 -> 2048 -> 2304;
        // tail = 2304 + 576*2 = 3456; (3456+2111)&~63 = 5504;
        // + 72*(0+0+2+0+1) = 216 -> 5720.
        assert_eq!(calculate_required_memory_size(&params(0, 13, 2)), 5720);
    }

    #[test]
    fn calc_max_dependencies() {
        // Same base as the minimal case, tail 2048 -> 4096,
        // + 72*(0+0+1+1023+1) = 73800 -> 77896.
        let mut p = params(0, 1, 1);
        p.max_dependencies = 1023;
        assert_eq!(calculate_required_memory_size(&p), 77896);
        // One over the per-thread budget is rejected.
        p.max_dependencies = 1024;
        assert_eq!(calculate_required_memory_size(&p), 0);
    }

    #[test]
    fn calc_mode1_without_second_mask_matches_mode0() {
        // mode 1 with second_mask_thread_count == 0 validates against the
        // same thread total and produces the mode-0 size.
        assert_eq!(calculate_required_memory_size(&params(1, 1, 1)), 4240);
    }

    #[test]
    fn calc_mode1_second_mask_unsupported() {
        let mut p = params(1, 1, 1);
        p.second_mask_thread_count = 1;
        p.affinity_mask = 1;
        assert_eq!(calculate_required_memory_size(&p), 0);
    }

    #[test]
    fn calc_rejects_invalid() {
        assert_eq!(calculate_required_memory_size(&params(2, 1, 1)), 0); // bad mode
        assert_eq!(calculate_required_memory_size(&params(0, 0, 1)), 0); // version 0
        assert_eq!(calculate_required_memory_size(&params(0, 14, 1)), 0); // version > 13
        assert_eq!(calculate_required_memory_size(&params(0, 1, 0)), 0); // no threads
        assert_eq!(calculate_required_memory_size(&params(0, 1, 0x21)), 0); // too many
        let mut p = params(0, 1, 2);
        p.max_sequences = 0x7FFF * 2 + 1;
        assert_eq!(calculate_required_memory_size(&p), 0);
        let mut p = params(0, 1, 2);
        p.max_priorities = 255 * 2 + 1;
        assert_eq!(calculate_required_memory_size(&p), 0);
        let mut p = params(0, 1, 2);
        p.max_tags = 63 * 2 + 1;
        assert_eq!(calculate_required_memory_size(&p), 0);
        let mut p = params(1, 1, 0x20);
        p.second_mask_thread_count = 1; // 0x20 + 1 > 0x20
        assert_eq!(calculate_required_memory_size(&p), 0);
        let mut p = params(1, 1, 1);
        p.second_mask_thread_count = 2;
        p.affinity_mask = 1; // popcnt < second mask count
        assert_eq!(calculate_required_memory_size(&p), 0);
        let mut p = params(1, 1, 1);
        p.second_mask_thread_count = 1;
        p.affinity_mask = 1 << 56; // mask above the low 56 bits
        assert_eq!(calculate_required_memory_size(&p), 0);
    }

    const BASE: u64 = 0x5D48_0300_0000;
    const BLOCK: u64 = 0x10000;

    fn ctx_with_block() -> (ps5vm::VirtualMemory, u64) {
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, BLOCK * 2, ps5vm::MemProtect::RW).unwrap();
        (vm, BASE)
    }

    #[test]
    fn initialize_writes_header_and_impl() {
        let (vm, mem) = ctx_with_block();
        let mut ctx = CpuContext::new(&vm);
        let this = mem + BLOCK; // handle object outside the memory block
        let block = mem + 0x100; // 256-byte aligned inside the mapping
        ctx.set(crate::context::CpuRegister::Rdi, this);
        ctx.set(crate::context::CpuRegister::Rsi, block);
        ctx.set(crate::context::CpuRegister::Rdx, 0x2000);
        ctx.set(crate::context::CpuRegister::Rcx, 13);
        ctx.set(crate::context::CpuRegister::R8, 1);
        assert_eq!(cb_initialize(&mut ctx), 0);
        assert_eq!(read_u16_at(&ctx, block).unwrap(), 0x4D4A);
        let impl_off = read_u16_at(&ctx, block + 2).unwrap() as u64;
        assert_eq!(impl_off, ((block + 23) & !0xF) - block);
        assert_eq!(ctx.read_u32(block + 4).unwrap(), 0x0500_0001);
        assert_eq!(ctx.read_u32(block + 8).unwrap(), 0x2000);
        assert_eq!(ctx.read_u32(block + 12).unwrap(), 0);
        let impl_addr = ctx.read_u64(this).unwrap();
        assert_eq!(impl_addr, block + impl_off);
        assert_eq!(ctx.read_u64(impl_addr + 0x28).unwrap(), 0);
        assert_eq!(ctx.read_u32(impl_addr + 128 + 512).unwrap(), 0);
        // Second initialize on the same handle: already initialized.
        ctx.set(crate::context::CpuRegister::Rsi, block + 0x1000);
        assert_eq!(cb_initialize(&mut ctx), ERR_ALREADY_INITIALIZED);
        // Shutdown clears the handle, second shutdown reports it.
        ctx.set(crate::context::CpuRegister::Rdi, this);
        assert_eq!(cb_shutdown(&mut ctx), 0);
        assert_eq!(ctx.read_u64(this).unwrap(), 0);
        assert_eq!(cb_shutdown(&mut ctx), ERR_NOT_INITIALIZED);
    }

    #[test]
    fn initialize_rejects_bad_arguments() {
        let (vm, mem) = ctx_with_block();
        let mut ctx = CpuContext::new(&vm);
        let this = mem + BLOCK;
        let block = mem + 0x100;
        // null memory
        ctx.set(crate::context::CpuRegister::Rdi, this);
        ctx.set(crate::context::CpuRegister::Rsi, 0);
        ctx.set(crate::context::CpuRegister::Rdx, 0x2000);
        ctx.set(crate::context::CpuRegister::Rcx, 13);
        ctx.set(crate::context::CpuRegister::R8, 1);
        assert_eq!(cb_initialize(&mut ctx), ERR_NULL_MEMORY);
        // unaligned memory
        ctx.set(crate::context::CpuRegister::Rsi, block + 0x40);
        assert_eq!(cb_initialize(&mut ctx), ERR_NOT_ALIGNED);
        // zero size
        ctx.set(crate::context::CpuRegister::Rsi, block);
        ctx.set(crate::context::CpuRegister::Rdx, 0);
        assert_eq!(cb_initialize(&mut ctx), ERR_INVALID_ARG);
        // version out of range
        ctx.set(crate::context::CpuRegister::Rdx, 0x2000);
        ctx.set(crate::context::CpuRegister::Rcx, 14);
        assert_eq!(cb_initialize(&mut ctx), ERR_INVALID_ARG);
        // gpu affinity count out of range
        ctx.set(crate::context::CpuRegister::Rcx, 13);
        ctx.set(crate::context::CpuRegister::R8, 0x21);
        assert_eq!(cb_initialize(&mut ctx), ERR_INVALID_ARG);
        // not enough memory for header + implementation region
        ctx.set(crate::context::CpuRegister::R8, 1);
        ctx.set(crate::context::CpuRegister::Rdx, 32);
        assert_eq!(cb_initialize(&mut ctx), ERR_INSUFFICIENT_MEMORY);
        assert_eq!(ctx.read_u64(this).unwrap(), 0);
        // config mode out of range
        let cfg = mem + 0x200;
        ctx.set(crate::context::CpuRegister::Rdx, 0x2000);
        ctx.set(crate::context::CpuRegister::R9, cfg);
        ctx.write_u32(cfg, 2);
        assert_eq!(cb_initialize(&mut ctx), ERR_INVALID_ARG);
    }

    #[test]
    fn ctor_and_dtor_roundtrip() {
        let (vm, mem) = ctx_with_block();
        let mut ctx = CpuContext::new(&vm);
        let this = mem + BLOCK;
        ctx.write_u64(this, 0x1234);
        ctx.set(crate::context::CpuRegister::Rdi, this);
        assert_eq!(cb_ctor(&mut ctx) as u64, this + 0x10);
        assert_eq!(ctx.read_u64(this).unwrap(), 0);
        // destructor of a never-initialized manager reports it
        assert_eq!(cb_dtor(&mut ctx), ERR_NOT_INITIALIZED);
        // after initialize it succeeds
        ctx.set(crate::context::CpuRegister::Rsi, mem + 0x100);
        ctx.set(crate::context::CpuRegister::Rdx, 0x2000);
        ctx.set(crate::context::CpuRegister::Rcx, 13);
        ctx.set(crate::context::CpuRegister::R8, 1);
        assert_eq!(cb_initialize(&mut ctx), 0);
        assert_eq!(cb_dtor(&mut ctx), 0);
        assert_eq!(ctx.read_u64(this).unwrap(), 0);
    }
}
