// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceVrTracker2 (PlayStation VR2 tracking) as the console's module
//! answers without a headset: NOT_INITIALIZED from the tracking calls, the
//! fixed memory requirement from sceVrTracker2QueryMemory, an Initialize
//! that validates its parameter, clears the work memory and stops at the
//! first tracking answer, and the answers of the module's unnamed exports
//! measured the same way.

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceVrTracker2";

const fn err(code: u32) -> HleResult {
    code as i32 as i64
}

const OK: HleResult = 0;
const ERROR_NOT_INITIALIZED: HleResult = err(0x8A74_0001);
const ERROR_INVALID_ARGUMENT: HleResult = err(0x8A74_0006);
const NOT_IMPLEMENTED: HleResult = err(0x8002_0002);
/// SceVrTracker2QueryMemoryParam / ...Requirement size.
const QUERY_SIZE: usize = 0x20;
/// SceVrTracker2InitializeParam size.
const INITIALIZE_SIZE: usize = 72;
const TRACKER_MODE: u32 = 0;
const SYSTEM_TRACKING_MODE: u32 = 1000;
const SYSTEM_CLIENT_MODE: u32 = 1001;
const MIN_ALIGNMENT: u32 = 0x10000;
const MIN_PRIORITY: i32 = 256;
const MAX_PRIORITY: i32 = 767;
const TRACKER_AFFINITY_LIMIT: u64 = 0x2000;

/// Work memory a mode needs: 32 MiB to track, 7 MiB / 3 MiB for the
/// system modes.
fn work_memory_size(mode: u32) -> u32 {
    match mode {
        SYSTEM_TRACKING_MODE => 7 << 20,
        SYSTEM_CLIENT_MODE => 3 << 20,
        _ => 32 << 20,
    }
}

fn cb_not_initialized(_ctx: &mut CpuContext) -> HleResult {
    ERROR_NOT_INITIALIZED
}

/// sceVrTracker2QueryMemory(const param*, requirement*): the param is
/// { u32 size = 0x20; mode; 24 zero bytes }, the requirement is
/// { 0x20, the mode's work memory, 64 KiB }.
fn cb_query_memory(ctx: &mut CpuContext) -> HleResult {
    let (param, out) = (ctx.rdi(), ctx.rsi());
    if param == 0 || out == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    let Some(p) = ctx.read_bytes(param, QUERY_SIZE) else { return ERROR_INVALID_ARGUMENT };
    if u32::from_le_bytes(p[0..4].try_into().unwrap()) != QUERY_SIZE as u32 || p[8..].iter().any(|&b| b != 0) {
        return ERROR_INVALID_ARGUMENT;
    }
    let mode = u32::from_le_bytes(p[4..8].try_into().unwrap());
    if mode != TRACKER_MODE && mode != SYSTEM_TRACKING_MODE && mode != SYSTEM_CLIENT_MODE {
        return ERROR_INVALID_ARGUMENT;
    }
    // The system modes ask for an SDK from 1.0 on; every title this host
    // runs is newer.
    let mut r = [0u8; QUERY_SIZE];
    for (i, v) in [QUERY_SIZE as u32, work_memory_size(mode), 64 << 10].into_iter().enumerate() {
        r[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    if ctx.write_bytes(out, &r) { OK } else { ERROR_INVALID_ARGUMENT }
}

/// sceVrTracker2Initialize(const param*): the 72-byte parameter is
/// validated field by field — size, mode (the system modes need a system
/// title's application category, which this host does not have), two
/// thread priorities 256..=767, two affinity masks below 0x2000 in
/// tracking mode, work memory pointer / size / alignment, zero
/// reserved bytes and an optionMode of -1, 0 or 1 (the option words at
/// 0x30/0x38 are only carried). Tracking mode then clears its 32 MiB work
/// memory and the module stays uninitialized: the first tracking open
/// answers -1 on a host without a headset, so the call ends there and
/// never reports ALREADY_INITIALIZED on a retry.
fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    let param = ctx.rdi();
    if param == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    let Some(p) = ctx.read_bytes(param, INITIALIZE_SIZE) else { return ERROR_INVALID_ARGUMENT };
    let u32_at = |o: usize| u32::from_le_bytes(p[o..o + 4].try_into().unwrap());
    let i32_at = |o: usize| i32::from_le_bytes(p[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(p[o..o + 8].try_into().unwrap());
    if u32_at(0) != INITIALIZE_SIZE as u32 {
        return ERROR_INVALID_ARGUMENT;
    }
    let mode = u32_at(4);
    if mode == SYSTEM_TRACKING_MODE || mode == SYSTEM_CLIENT_MODE {
        log::warn!("[vrtracker2] Initialize: system mode {mode} needs a system title's application category");
        return NOT_IMPLEMENTED;
    }
    if mode != TRACKER_MODE {
        return ERROR_INVALID_ARGUMENT;
    }
    let work_memory = u64_at(0x20);
    if work_memory == 0 || u32_at(0x28) < work_memory_size(mode) || u32_at(0x2C) < MIN_ALIGNMENT {
        return ERROR_INVALID_ARGUMENT;
    }
    if p[0x3A..0x48].iter().any(|&b| b != 0) {
        return ERROR_INVALID_ARGUMENT;
    }
    if (0..2).any(|i| !(MIN_PRIORITY..=MAX_PRIORITY).contains(&i32_at(0x08 + i * 4))) {
        return ERROR_INVALID_ARGUMENT;
    }
    if (0..2).any(|i| u64_at(0x10 + i * 8) >= TRACKER_AFFINITY_LIMIT) {
        return ERROR_INVALID_ARGUMENT;
    }
    let zero = [0u8; 0x10000];
    for at in (0..work_memory_size(mode) as u64).step_by(0x10000) {
        if !ctx.write_bytes(work_memory + at, &zero) {
            return ERROR_INVALID_ARGUMENT;
        }
    }
    // optionMode: -1 and 0 are always allowed, 1 needs an SDK from 7.0
    // on (every title this host runs).
    match p[0x39] as i8 {
        -1 | 0 | 1 => {}
        _ => return ERROR_INVALID_ARGUMENT,
    }
    -1
}

/// Zero `len` bytes of the first argument, then NOT_INITIALIZED (NULL:
/// INVALID_ARGUMENT).
fn clear_then_not_initialized(ctx: &CpuContext, len: usize) -> HleResult {
    let out = ctx.rdi();
    if out == 0 || !ctx.write_bytes(out, &vec![0u8; len]) {
        return ERROR_INVALID_ARGUMENT;
    }
    ERROR_NOT_INITIALIZED
}

fn cb_clear_wide_result(ctx: &mut CpuContext) -> HleResult {
    clear_then_not_initialized(ctx, 24)
}

fn cb_clear_narrow_result(ctx: &mut CpuContext) -> HleResult {
    clear_then_not_initialized(ctx, 16)
}

fn cb_check_second(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 { ERROR_INVALID_ARGUMENT } else { ERROR_NOT_INITIALIZED }
}

fn cb_accept(_ctx: &mut CpuContext) -> HleResult {
    OK
}

/// Fill the defaults the module writes: u32 0xC000 at +4, u32 0 at +8,
/// u16 0x40 at +12 and +14.
fn cb_fill_defaults(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        log::error!("[vrtracker2] null output");
        return NOT_IMPLEMENTED;
    }
    let ok = ctx.write_u32(out + 4, 0xC000) && ctx.write_u32(out + 8, 0) && ctx.write_bytes(out + 12, &[0x40, 0, 0x40, 0]);
    if ok { OK } else { ERROR_INVALID_ARGUMENT }
}

/// Clear one u32 output.
fn cb_clear_word(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        log::error!("[vrtracker2] null output");
        return NOT_IMPLEMENTED;
    }
    if ctx.write_u32(out, 0) { OK } else { ERROR_INVALID_ARGUMENT }
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("IQ3UD6SZbXo", "sceVrTracker2Finalize", cb_not_initialized),
    ("Y-3JCiU9bbU", "sceVrTracker2GetCoordinateSystem", cb_not_initialized),
    ("SCph4ZbkqzU", "sceVrTracker2GetPlayAreaBoundaryGeometry", cb_not_initialized),
    ("snYs7Nf-RKk", "sceVrTracker2GetPlayAreaOrientedBoundingBox", cb_not_initialized),
    ("J4Vh3VVX0iU", "sceVrTracker2GetResult", cb_not_initialized),
    ("f7G97dWnEis", "sceVrTracker2LocateCoordinateSystem", cb_not_initialized),
    ("Dog+g25QYjw", "sceVrTracker2RegisterDevice", cb_not_initialized),
    ("IdI2f+xHIeA", "sceVrTracker2ResetLocalCoordinate", cb_not_initialized),
    ("UVCMLmS-Eas", "sceVrTracker2SetCoordinateSystem", cb_not_initialized),
    ("kFt4MB3SUEk", "sceVrTracker2UnregisterDevice", cb_not_initialized),
    ("TwqZnaIjWv4", "sceVrTracker2QueryMemory", cb_query_memory),
    ("6Jy73SRfG-o", "sceVrTracker2Initialize", cb_initialize),
];

/// Exports without a known name that answer NOT_INITIALIZED.
const UNNAMED_NOT_INITIALIZED: &[&str] = &[
    "1m6DD9BBmSE",
    "SgunslIO-C0",
    "KMIToVnHrhc",
    "XfEBVemeJyE",
    "y79S3-zrJKg",
    "b777Kpfe2J8",
    "UfUYgiT7VK0",
    "MB7B6396K6U",
    "H2Mnx7gz4ac",
    "SgNHzN6XaEc",
    "sYBNLrMxjAM",
    "3gQncidodmw",
    "qnCmEbUGqWY",
    "oGNn7JIdMHw",
    "80IxOn8swIA",
    "NTbJNAZA4ro",
    "914+9oJxNts",
    "c6SZtdEjH0E",
    "74IXIMe+Lto",
    "hmahOYMiYNY",
    "pNXc6ZlLq-8",
    "omJ7b8FqAN0",
    "G2PWj1HLqn8",
    "73zcxAFawYk",
    "vegrWFNiQwc",
    "6r01IawmypA",
    "zeof7sF2fqE",
    "Z0rfHOHQV1w",
    "4l7x+uy3dWw",
    "ZXlt-Xb3da4",
    "BSwAdB3GM3o",
    "9ATuhVfn9Dw",
    "wjmVbmf58rE",
    "4+B+TcHEHks",
    "cjDz4kYJKTw",
    "BjVg0fKV9VA",
    "oDV5j-2Luic",
    "t+86o9sCZ-Y",
    "TIpYP35JzUs",
    "LrSc+KuxIkg",
    "Ur+M5wy+QBY",
    "OvKC351Udqo",
    "LQV4nz6wteY",
    "x0pGthEwf88",
    "d5Fjs0rHkBg",
    "Yed3ZONcuMk",
    "fkptW6uRdWc",
    "HDlD-xE1Xuk",
    "IGdE6nsMTAY",
    "DSTergmOvvE",
    "B7hPwvvg79E",
    "ZwnIsDe1eNE",
    "B2vw+s-0Dfo",
    "4LNTLivfRuA",
    "edajhZhz9Uo",
    "QS+7oqiszuc",
    "-dDyvzd2jf0",
    "k8JDLkCrOgs",
    "-HFx+dnDSOI",
    "wgDeDXirl0g",
    "a1kWSTYGBxg",
    "zUwxBEKr3fc",
    "TC7VTVzj0xc",
    "1DwROxtuANs",
    "wwbt4s8cbMQ",
    "eGxBJ0IolOA",
    "9BaFhspHF+E",
    "lb9q5nJ4vzk",
    "ZsD1-thjXS4",
    "QdHQnM0XICI",
    "Ek4WMOvwv6g",
    "W1YAO6G+hNc",
    "HAJB7irjl+0",
    "x2mN50nKScs",
    "cCKExge5d3A",
    "gn6e5I4g9bw",
    "OYTKt9diFOY",
    "loU5lT92Zrw",
    "o6fsgA3R2Yc",
    "AeBtEIE0idk",
    "DBa3wov7dKQ",
    "ojTGClQ52r4",
    "oOLE62H66z0",
    "MHekhnRBKAw",
    "JnSmj+5E8WY",
    "N1P+melAgb8",
    "-e7A3rAM3Yk",
    "RefT1spBleg",
    "DsOfryhbm4U",
];

/// Exports without a known name with their own answers.
const UNNAMED: &[(&str, HleCallback)] = &[
    ("XknY0IDcOwU", cb_clear_wide_result),
    ("ark13GkiskA", cb_clear_narrow_result),
    ("-YvjOUph8B0", cb_check_second),
    ("QyZvFyf-QEk", cb_accept),
    ("fTYZOZepkqE", cb_accept),
    ("TZIUIk9id3c", cb_fill_defaults),
    ("IB0qcDVu8vU", cb_clear_word),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
    for &nid in UNNAMED_NOT_INITIALIZED {
        reg.register_fn(nid, "?", LIB, cb_not_initialized);
    }
    for &(nid, cb) in UNNAMED {
        reg.register_fn(nid, "?", LIB, cb);
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
        assert_eq!(UNNAMED_NOT_INITIALIZED.len(), 89);
    }

    /// Reference GuestVrTracker2 cases (the console's module before
    /// initialization).
    #[test]
    fn answers_before_initialization() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A73_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let call = |ctx: &mut CpuContext, cb: HleCallback, a: u64, b: u64| {
            ctx.set(Rdi, a);
            ctx.set(Rsi, b);
            cb(ctx)
        };
        ctx.write_bytes(BASE, &[0; QUERY_SIZE]);
        ctx.write_u32(BASE, QUERY_SIZE as u32);
        ctx.write_bytes(BASE + 0x100, &[0xEE; QUERY_SIZE]);
        assert_eq!(call(&mut ctx, cb_query_memory, BASE, BASE + 0x100), OK);
        assert_eq!(ctx.read_u32(BASE + 0x104), Some(32 << 20));
        assert_eq!(ctx.read_u32(BASE + 0x108), Some(64 << 10));
        assert_eq!(ctx.read_bytes(BASE + 0x10C, 20).unwrap(), vec![0; 20]);
        ctx.write_bytes(BASE + 8, &[1]);
        assert_eq!(call(&mut ctx, cb_query_memory, BASE, BASE + 0x100), ERROR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_query_memory, 0, BASE + 0x100), ERROR_INVALID_ARGUMENT);
        assert_eq!(cb_not_initialized(&mut ctx), ERROR_NOT_INITIALIZED);
        ctx.write_bytes(BASE + 0x200, &[0xEE; 32]);
        assert_eq!(call(&mut ctx, cb_clear_wide_result, BASE + 0x200, 0), ERROR_NOT_INITIALIZED);
        assert_eq!(ctx.read_bytes(BASE + 0x200, 32).unwrap(), [vec![0; 24], vec![0xEE; 8]].concat());
        assert_eq!(call(&mut ctx, cb_clear_narrow_result, 0, 0), ERROR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_check_second, 0, 0), ERROR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_check_second, 0, BASE), ERROR_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, cb_fill_defaults, BASE + 0x300, 0), OK);
        assert_eq!(ctx.read_u32(BASE + 0x304), Some(0xC000));
        assert_eq!(ctx.read_bytes(BASE + 0x30C, 4).unwrap(), vec![0x40, 0, 0x40, 0]);
    }
}
