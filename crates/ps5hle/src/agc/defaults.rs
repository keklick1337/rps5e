// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! `sceAgcInit`, register-default tables (`sceAgcGetRegisterDefaults2` and
//! its internal variant), the Trinity-mode query and the AMM semaphore
//! label pool.
//!
//! A register-defaults object (allocated once per version in guest memory):
//!
//! | Offset | Field |
//! |---|---|
//! | +0x00..+0x18 | four pointers to per-table group pointer arrays (0 = empty) |
//! | +0x20..+0x2C | four u32 register counts |
//! | +0x30 | pointer to the type list |
//! | +0x38 | u32 type entry count (entries are 3 dwords) |
//!
//! Each group pointer addresses a run of `{u32 offset, u32 value}` pairs
//! inside the table's register block.
//!
//! The object `sceAgcGetRegisterDefaults` returns hands its types out as
//! 8-byte `{key, index << 2 | bank}` pairs instead (the layout titles walk to
//! find a block by key); the compact entries also pack the block's register
//! count and stride above bit 10, which the pairs drop.

use super::regdefs_data::*;
use super::{arg, mem, ptr, Export, ERR_ALREADY_EXISTS, ERR_INVALID_ARGUMENT, ERR_MEMORY_FAULT, LIB_AGC, OK};
use crate::context::CpuContext;
use crate::registry::HleResult;
use std::collections::HashMap;
use std::sync::Mutex;

/// One versioned dataset: four register tables `(offset, value)` with their
/// group start indices, and the type list.
pub(crate) struct CompactDefaults {
    pub t0: &'static [u32],
    pub p0: &'static [u16],
    pub t1: &'static [u32],
    pub p1: &'static [u16],
    pub t2: &'static [u32],
    pub p2: &'static [u16],
    pub t3: &'static [u32],
    pub p3: &'static [u16],
    pub types: &'static [u32],
}

impl CompactDefaults {
    fn tables(&self) -> [(&'static [u32], &'static [u16]); 4] {
        [(self.t0, self.p0), (self.t1, self.p1), (self.t2, self.p2), (self.t3, self.p3)]
    }
}

const HEADER_SIZE: usize = 0x40;

/// Newest defaults version; later versions use its data.
const MAX_VERSION: u32 = 13;

/// (public, internal) dataset of a defaults version.
pub(crate) fn select(version: u32) -> Option<(&'static CompactDefaults, &'static CompactDefaults)> {
    let v = version.min(MAX_VERSION) as usize;
    Some((PUBLIC_BY_VERSION[v], INTERNAL_BY_VERSION[v]))
}

fn align(v: usize, a: usize) -> usize {
    (v + a - 1) & !(a - 1)
}

/// Layout of the type list of a defaults object.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TypeLayout {
    /// The 3-dword compact entries (versioned calls).
    Compact,
    /// `{key, index << 2 | bank}` pairs (the call without a version).
    KeyBankPairs,
}

/// The `{key, index << 2 | bank}` pairs of `d`'s types, or None when an
/// index lies outside its bank's groups.
pub(crate) fn key_bank_pairs(d: &CompactDefaults) -> Option<Vec<[u32; 2]>> {
    let tables = d.tables();
    d.types
        .chunks_exact(3)
        .map(|t| {
            let bank_index = t[1] & 0x3FF;
            ((bank_index >> 2) < tables[(bank_index & 3) as usize].1.len() as u32).then_some([t[0], bank_index])
        })
        .collect()
}

/// Serialize `d` for guest address `base` (compact types).
#[cfg(test)]
pub(crate) fn build_blob(d: &CompactDefaults, base: u64) -> Option<Vec<u8>> {
    build_blob_with(d, base, TypeLayout::Compact)
}

/// Serialize `d` for guest address `base` with the given type layout.
pub(crate) fn build_blob_with(d: &CompactDefaults, base: u64, layout: TypeLayout) -> Option<Vec<u8>> {
    let tables = d.tables();
    for (regs, ptrs) in tables {
        if regs.len() % 2 != 0 || (ptrs.is_empty() != regs.is_empty()) || ptrs.iter().any(|&p| p as usize >= regs.len() / 2) {
            return None;
        }
    }
    if d.types.len() % 3 != 0 {
        return None;
    }
    let types: Vec<u32> = match layout {
        TypeLayout::Compact => d.types.to_vec(),
        TypeLayout::KeyBankPairs => key_bank_pairs(d)?.concat(),
    };
    let mut table_off = [0usize; 4];
    let mut cur = align(HEADER_SIZE, 8);
    for (i, (_, ptrs)) in tables.iter().enumerate() {
        table_off[i] = cur;
        cur += ptrs.len() * 8;
    }
    let mut regs_off = [0usize; 4];
    cur = align(cur, 8);
    for (i, (regs, _)) in tables.iter().enumerate() {
        regs_off[i] = cur;
        cur += regs.len() * 4;
    }
    let types_off = align(cur, 8);
    let len = types_off + types.len() * 4;
    let mut b = vec![0u8; len];
    let put32 = |b: &mut [u8], o: usize, v: u32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
    let put64 = |b: &mut [u8], o: usize, v: u64| b[o..o + 8].copy_from_slice(&v.to_le_bytes());
    for (i, (regs, ptrs)) in tables.iter().enumerate() {
        put64(&mut b, i * 8, if ptrs.is_empty() { 0 } else { base + table_off[i] as u64 });
        put32(&mut b, 0x20 + i * 4, (regs.len() / 2) as u32);
        for (j, &p) in ptrs.iter().enumerate() {
            put64(&mut b, table_off[i] + j * 8, base + regs_off[i] as u64 + p as u64 * 8);
        }
        for (j, &r) in regs.iter().enumerate() {
            put32(&mut b, regs_off[i] + j * 4, r);
        }
    }
    put64(&mut b, 0x30, base + types_off as u64);
    put32(&mut b, 0x38, (d.types.len() / 3) as u32);
    for (j, &t) in types.iter().enumerate() {
        put32(&mut b, types_off + j * 4, t);
    }
    Some(b)
}

fn blob_len(d: &CompactDefaults, layout: TypeLayout) -> usize {
    build_blob_with(d, 0, layout).map_or(0, |b| b.len())
}

/// version -> (public, internal) guest addresses.
static ALLOCATIONS: Mutex<Option<HashMap<u32, (u64, u64)>>> = Mutex::new(None);

fn materialize(ctx: &CpuContext, d: &CompactDefaults, layout: TypeLayout) -> Option<u64> {
    let addr = crate::pthread::alloc_guest_block(ctx, blob_len(d, layout) as u64)?;
    let blob = build_blob_with(d, addr, layout)?;
    mem::write(addr, &blob).then_some(addr)
}

/// Guest address of the object `sceAgcGetRegisterDefaults` returns.
static UNVERSIONED: Mutex<Option<u64>> = Mutex::new(None);

/// Version of the table `sceAgcGetRegisterDefaults` returns: a console
/// returns the running system software's table (137 keyed entries over 84
/// context blocks on 12.02 and 13.x), which is version 13's.
pub(crate) const UNVERSIONED_VERSION: usize = 13;

fn unversioned_defaults(ctx: &CpuContext) -> Option<u64> {
    let mut g = UNVERSIONED.lock().unwrap();
    if g.is_none() {
        *g = Some(materialize(ctx, PUBLIC_BY_VERSION[UNVERSIONED_VERSION], TypeLayout::KeyBankPairs)?);
    }
    *g
}

fn defaults_for(ctx: &CpuContext, version: u32) -> Option<(u64, u64)> {
    if version > MAX_VERSION {
        log::warn!("[agc] register defaults version {version} unsupported; using {MAX_VERSION}");
    }
    let version = version.min(MAX_VERSION);
    let mut g = ALLOCATIONS.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    if let Some(&v) = map.get(&version) {
        return Some(v);
    }
    let (public, internal) = select(version)?;
    let v = (materialize(ctx, public, TypeLayout::Compact)?, materialize(ctx, internal, TypeLayout::Compact)?);
    map.insert(version, v);
    Some(v)
}

/// The driver's direct memory: 2 MiB at a fixed address, CPU and GPU
/// readable and writable, holding the system table SDK shaders read at
/// `DRIVER_DMEM_BASE + AGC_DMEM_OFFSET`.
pub(crate) const DRIVER_DMEM_BASE: u64 = 0x0f_e000_0000;
pub(crate) const DRIVER_DMEM_SIZE: u64 = 0x20_0000;
pub(crate) const AGC_DMEM_OFFSET: u64 = 0x4_0000;
/// Direct memory type of the driver memory.
const DRIVER_DMEM_TYPE: i32 = 12;
/// CPU read/write, GPU read/write.
const DRIVER_DMEM_PROT: i32 = 0x33;

/// The ordered-count queue validator (function 2 of the system table),
/// including its return to the caller.
pub(crate) const QUEUE_VALIDATOR: [u32; 16] = [
    0xbeeb03ff, 0x00000021, 0xb96a1818, 0xbf06106a, 0xbf800000, 0x8590807e, 0xb96a0a18, 0xbf066a80, 0xbf800000, 0x85ea807e, 0x88ea106a, 0xbf870003, 0xbefc03ff, 0x8a6ca000, 0xbf920009, 0xbefd210e,
];

/// Physical offset of the driver memory once it is set up.
static DRIVER_DMEM: Mutex<Option<u64>> = Mutex::new(None);

/// The system table: a buffer descriptor (16-byte stride, 8 records) at
/// the table base addressing eight `{u64 code; u32 user_data[2]}` function
/// entries at +0x60; entry 2 runs the queue validator at +0x100 (shader
/// code is 256-byte aligned). Everything else stays zero.
pub(crate) fn system_table(base: u64) -> ([u32; 4], [u8; 8 * 16]) {
    let functions = base + 0x60;
    let code = base + 0x100;
    let descriptor = [functions as u32, (functions >> 32) as u32 | 0x0010_0000, 8, 0x5204];
    let mut entries = [0u8; 8 * 16];
    entries[2 * 16..2 * 16 + 8].copy_from_slice(&code.to_le_bytes());
    (descriptor, entries)
}

/// Reserve, map and fill the driver memory (once). Called at process start
/// (`app_info::configure`). The memory is system memory, outside the title's
/// direct memory: the title's direct memory stays whole, so its allocations
/// (which cover all of it) can still succeed. A failure is logged, and
/// `sceAgcInit` reports it.
pub(crate) fn reserve_driver_memory() {
    let mut g = DRIVER_DMEM.lock().unwrap_or_else(|e| e.into_inner());
    if g.is_some() {
        return;
    }
    let phys = match crate::memory::allocate_system_memory(DRIVER_DMEM_SIZE, DRIVER_DMEM_SIZE, DRIVER_DMEM_TYPE) {
        Ok(p) => p,
        Err(e) => {
            log::error!("[agc] no system memory for the driver ({DRIVER_DMEM_SIZE:#x} bytes): {e:#x}; sceAgcInit will fail");
            return;
        }
    };
    let addr = match crate::memory::map_named_direct(DRIVER_DMEM_BASE, DRIVER_DMEM_SIZE, DRIVER_DMEM_PROT, 0, phys, DRIVER_DMEM_SIZE, "SceAgcDriver") {
        Ok(a) => a,
        Err(e) => {
            log::error!("[agc] cannot map the driver memory at {DRIVER_DMEM_BASE:#x}: {e:#x}; sceAgcInit will fail");
            crate::memory::release_direct(phys, DRIVER_DMEM_SIZE);
            return;
        }
    };
    if addr != DRIVER_DMEM_BASE {
        log::error!("[agc] driver memory landed at {addr:#x}, not {DRIVER_DMEM_BASE:#x}: system table not installed");
        crate::memory::release_direct(phys, DRIVER_DMEM_SIZE);
        return;
    }
    let base = DRIVER_DMEM_BASE + AGC_DMEM_OFFSET;
    let (descriptor, entries) = system_table(base);
    let mut code = [0u8; 64];
    for (b, w) in code.chunks_exact_mut(4).zip(QUEUE_VALIDATOR) {
        b.copy_from_slice(&w.to_le_bytes());
    }
    if !(mem::fill(addr, DRIVER_DMEM_SIZE, 0) && mem::write(base + 0x60, &entries) && mem::write(base + 0x100, &code) && mem::write_dwords(base, &descriptor)) {
        log::error!("[agc] cannot write the driver system table; sceAgcInit will fail");
        crate::memory::release_direct(phys, DRIVER_DMEM_SIZE);
        return;
    }
    *g = Some(phys);
}

/// Unmap and release the driver memory (shutdown and tests).
#[allow(dead_code)]
pub(crate) fn release_driver_memory() {
    let mut g = DRIVER_DMEM.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(phys) = g.take() {
        crate::memory::release_direct(phys, DRIVER_DMEM_SIZE);
    }
}

/// `sceAgcInit(state, version)`: the driver memory and its system table are
/// installed at process start (`reserve_driver_memory`); this reports whether
/// they are in place. Newer versions than the register defaults know are
/// reported.
pub(crate) fn init(ctx: &mut CpuContext) -> HleResult {
    let version = arg(ctx, 1) as u32;
    if version > MAX_VERSION {
        log::warn!("[agc] AgcInit version {version} unsupported");
    }
    if DRIVER_DMEM.lock().unwrap_or_else(|e| e.into_inner()).is_some() {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

/// `sceAgcGetRegisterDefaults()`: the call has no version argument and
/// returns the newest public table with its types as key/bank pairs
/// (`RPS5E_UNVERSIONED_REG_DEFAULTS=0`: the version 0 compact table, the
/// earlier behaviour).
fn get_register_defaults(ctx: &mut CpuContext) -> HleResult {
    static OLD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OLD.get_or_init(|| std::env::var("RPS5E_UNVERSIONED_REG_DEFAULTS").is_ok_and(|v| v == "0")) {
        return ptr(defaults_for(ctx, 0).map_or(0, |(p, _)| p));
    }
    ptr(unversioned_defaults(ctx).unwrap_or(0))
}

/// `sceAgcGetRegisterDefaultsInternal()`: the version 0 internal table.
fn get_register_defaults_internal(ctx: &mut CpuContext) -> HleResult {
    ptr(defaults_for(ctx, 0).map_or(0, |(_, i)| i))
}

fn get_register_defaults2(ctx: &mut CpuContext) -> HleResult {
    let v = arg(ctx, 0) as u32;
    ptr(defaults_for(ctx, v).map_or(0, |(p, _)| p))
}

fn get_register_defaults2_internal(ctx: &mut CpuContext) -> HleResult {
    let v = arg(ctx, 0) as u32;
    ptr(defaults_for(ctx, v).map_or(0, |(_, i)| i))
}

/// Not Trinity (base console). Also stores 0 to an optional out byte.
fn get_is_trinity_mode(ctx: &mut CpuContext) -> HleResult {
    let out = arg(ctx, 0);
    if out > 0xFFFF {
        mem::w8(out, 0);
    }
    0
}

const AMM_ALIGNMENT: u64 = 0x4000;
const AMM_SLOT: u64 = 32;

static AMM_MEMORY: Mutex<Option<(u64, u64)>> = Mutex::new(None);

/// `sceAgcSetAmmSemaphoreMemory(base, size)`: zero every 32-byte slot's
/// counter; the pool can be set once.
fn set_amm_semaphore_memory(ctx: &mut CpuContext) -> HleResult {
    let (base, size) = (arg(ctx, 0), arg(ctx, 1));
    if base == 0 || size < AMM_ALIGNMENT || base & (AMM_ALIGNMENT - 1) != 0 || size & (AMM_ALIGNMENT - 1) != 0 || base.checked_add(size).is_none() {
        return ERR_INVALID_ARGUMENT;
    }
    let mut g = AMM_MEMORY.lock().unwrap();
    if g.is_some() {
        return ERR_ALREADY_EXISTS;
    }
    let mut off = 0;
    while off < size {
        if !mem::w64(base + off, 0) {
            return ERR_MEMORY_FAULT;
        }
        off += AMM_SLOT;
    }
    *g = Some((base, size));
    OK
}

/// `sceAgcGetSemaphoreLabel(slot, out)`.
fn get_semaphore_label(ctx: &mut CpuContext) -> HleResult {
    let (slot, out) = (arg(ctx, 0) as u32 as u64, arg(ctx, 1));
    let g = AMM_MEMORY.lock().unwrap();
    let Some((base, size)) = *g else { return ERR_INVALID_ARGUMENT };
    if out == 0 || slot >= size / AMM_SLOT {
        return ERR_INVALID_ARGUMENT;
    }
    if mem::w64(out, base + slot * AMM_SLOT) {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

/// Error of `sceAgcGetDefaultCxStateFlat` for a null destination.
const ERR_AGC_INVALID_ARGUMENT: HleResult = 0x8089_0001u32 as i32 as i64;

/// The default context state as a flat array of the 512 context registers:
/// version 8's public defaults (every table, in order), 0 elsewhere.
pub(crate) fn default_cx_state_flat() -> [u32; 512] {
    let mut flat = [0u32; 512];
    for (table, _) in PUBLIC_BY_VERSION[8].tables() {
        for pair in table.chunks_exact(2) {
            if let Some(slot) = flat.get_mut(pair[0] as usize) {
                *slot = pair[1];
            }
        }
    }
    flat
}

/// `sceAgcGetDefaultCxStateFlat(out, size)`: copy the first `size` bytes
/// (at most 2048) of the flat default context state.
fn get_default_cx_state_flat(ctx: &mut CpuContext) -> HleResult {
    let (out, size) = (arg(ctx, 0), arg(ctx, 1) as u32);
    if out == 0 {
        return ERR_AGC_INVALID_ARGUMENT;
    }
    let bytes: Vec<u8> = default_cx_state_flat().iter().flat_map(|w| w.to_le_bytes()).collect();
    let n = (size as usize).min(bytes.len());
    if n != 0 && !mem::write(out, &bytes[..n]) {
        return ERR_MEMORY_FAULT;
    }
    OK
}

/// Error of the static buffer calls for an index past the last buffer.
const ERR_INVALID_STATIC_BUFFER: HleResult = 0x8A6C_000Bu32 as i32 as i64;
/// The static table is the system table: six 16-byte buffers, buffer 0
/// being the system table's own descriptor; the static qwords start at
/// buffer 1 (qword `i` at +16 + 8i).
const STATIC_BUFFER_COUNT: u64 = 6;
const STATIC_BUFFER_BYTES: u64 = 16;
const STATIC_QWORD_OFFSET: u64 = 16;

fn static_table() -> u64 {
    DRIVER_DMEM_BASE + AGC_DMEM_OFFSET
}

/// `sceAgcSetStaticBuffer(index, const void* buffer)`: 16 bytes into slot
/// `index`.
pub(crate) fn set_static_buffer(ctx: &mut CpuContext) -> HleResult {
    let (index, buffer) = (arg(ctx, 0) as u32 as u64, arg(ctx, 1));
    if index >= STATIC_BUFFER_COUNT {
        return ERR_INVALID_STATIC_BUFFER;
    }
    if mem::copy(static_table() + index * STATIC_BUFFER_BYTES, buffer, STATIC_BUFFER_BYTES) {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

/// `sceAgcGetStaticBuffer(index, void* buffer)`.
pub(crate) fn get_static_buffer(ctx: &mut CpuContext) -> HleResult {
    let (index, buffer) = (arg(ctx, 0) as u32 as u64, arg(ctx, 1));
    if index >= STATIC_BUFFER_COUNT {
        return ERR_INVALID_STATIC_BUFFER;
    }
    if mem::copy(buffer, static_table() + index * STATIC_BUFFER_BYTES, STATIC_BUFFER_BYTES) {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

/// `sceAgcSetStaticQword(index, u64 value)`: the index is not checked.
pub(crate) fn set_static_qword(ctx: &mut CpuContext) -> HleResult {
    let (index, value) = (arg(ctx, 0) as u32 as u64, arg(ctx, 1));
    if mem::w64(static_table() + STATIC_QWORD_OFFSET + index * 8, value) {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

/// `sceAgcGetStaticQword(index, u64* value)`.
pub(crate) fn get_static_qword(ctx: &mut CpuContext) -> HleResult {
    let (index, out) = (arg(ctx, 0) as u32 as u64, arg(ctx, 1));
    match mem::r64(static_table() + STATIC_QWORD_OFFSET + index * 8) {
        Some(v) if mem::w64(out, v) => OK,
        _ => ERR_MEMORY_FAULT,
    }
}

pub(crate) const EXPORTS: &[Export] = &[
    ("-pjHhph0ZUc", "sceAgcSetStaticBuffer", LIB_AGC, set_static_buffer),
    ("OXDc0KCnUhs", "sceAgcGetStaticBuffer", LIB_AGC, get_static_buffer),
    ("86rJnbZUpr4", "sceAgcSetStaticQword", LIB_AGC, set_static_qword),
    ("NbiQSQ-UL+M", "sceAgcGetStaticQword", LIB_AGC, get_static_qword),
    ("23LRUSvYu1M", "sceAgcInit_0090", LIB_AGC, init),
    ("AAeX-U5-P3M", "sceAgcGetDefaultCxStateFlat", LIB_AGC, get_default_cx_state_flat),
    ("Wi82ArQtAwg", "sceAgcGetRegisterDefaults", LIB_AGC, get_register_defaults),
    ("uIwxsqDlHRc", "sceAgcGetRegisterDefaultsInternal", LIB_AGC, get_register_defaults_internal),
    ("2JtWUUiYBXs", "sceAgcGetRegisterDefaults2", LIB_AGC, get_register_defaults2),
    ("wRbq6ZjNop4", "sceAgcGetRegisterDefaults2Internal", LIB_AGC, get_register_defaults2_internal),
    ("BfBDZGbti7A", "sceAgcGetIsTrinityMode", LIB_AGC, get_is_trinity_mode),
    ("OQTgEXyihvA", "sceAgcSetAmmSemaphoreMemory", LIB_AGC, set_amm_semaphore_memory),
    ("hFQ9pUxoLQ4", "sceAgcGetSemaphoreLabel", LIB_AGC, get_semaphore_label),
];
