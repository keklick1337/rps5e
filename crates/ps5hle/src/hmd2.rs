// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceHmd2 — PlayStation VR2 headset.
//!
//! No headset is attached: initialisation succeeds, device information
//! reports "not detected", and reprojection can be initialised but VR
//! output mode is refused for lack of a device.
//!
//! Reprojection state lives in the caller-provided reprojection buffer:
//! { u64 output_mode; u32 timing_us } (16 bytes, 8-aligned).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

#[path = "hmd2_unnamed.rs"]
mod unnamed;

const LIB: &str = "libSceHmd2";

const fn err(code: u32) -> HleResult {
    code as i32 as i64
}

pub const ERROR_ALREADY_INITIALIZED: HleResult = err(0x8A72_0001);
pub const ERROR_NOT_INITIALIZED: HleResult = err(0x8A72_0002);
pub const ERROR_PARAMETER_NULL: HleResult = err(0x8A72_0008);
pub const ERROR_REPROJECTION_NOT_INITIALIZED: HleResult = err(0x8A72_000B);
pub const ERROR_REPROJECTION_ALREADY_INITIALIZED: HleResult = err(0x8A72_000C);
pub const ERROR_REPROJECTION_IN_VR_MODE: HleResult = err(0x8A72_001C);
const VIDEO_OUT_ERROR_INVALID_VALUE: HleResult = err(0x8029_0001);
const VIDEO_OUT_ERROR_NO_DEVICE: HleResult = err(0x8029_0018);

pub const PANEL_WIDTH: u32 = 4000;
pub const PANEL_HEIGHT: u32 = 2040;
pub const OUTPUT_MODE_89_91HZ: u64 = 0x2000C;
pub const OUTPUT_MODE_119_88HZ: u64 = 0x2000F;

/// Whether a virtual headset is presented to the game.
const VR_ENABLED: bool = false;

/// Reprojection state block: output mode (u64) then timing in µs (u32).
const REPROJECTION_STATE_SIZE: u64 = 16;
const REPROJECTION_STATE_ALIGN: u64 = 8;
const DEFAULT_TIMING_US: u32 = 3000;

/// The display target is one 10:10:10:2 render-target surface tiled in
/// 64 KiB blocks of 128x128 four-byte elements.
const RT_BLOCK_DIM: u64 = 128;
const RT_BLOCK_SIZE: u64 = 0x1_0000;
const RT_BYTES_PER_ELEMENT: u64 = 4;

static INITIALIZED: AtomicBool = AtomicBool::new(false);
static REPROJECTION_LOCK: Mutex<()> = Mutex::new(());
/// Guest address of the reprojection state (0: not initialised).
static REPROJECTION_STATE: AtomicU64 = AtomicU64::new(0);

fn display_size() -> u64 {
    let pitch = (PANEL_WIDTH as u64).next_multiple_of(RT_BLOCK_DIM);
    let height = (PANEL_HEIGHT as u64).next_multiple_of(RT_BLOCK_DIM);
    (pitch * height * RT_BYTES_PER_ELEMENT).next_multiple_of(RT_BLOCK_SIZE)
}

/// Guest address of the active reprojection state, if any.
pub fn reprojection_state() -> Option<u64> {
    match REPROJECTION_STATE.load(Ordering::Acquire) {
        0 => None,
        a => Some(a),
    }
}

/// sceHmd2Initialize(const InitializeParam* param)
fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return ERROR_PARAMETER_NULL;
    }
    if INITIALIZED.swap(true, Ordering::AcqRel) {
        return ERROR_ALREADY_INITIALIZED;
    }
    log::info!("[hmd2] initialized, virtual headset {}", if VR_ENABLED { "enabled" } else { "disabled" });
    0
}

/// sceHmd2Terminate()
fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    if INITIALIZED.swap(false, Ordering::AcqRel) { 0 } else { ERROR_NOT_INITIALIZED }
}

/// sceHmd2GetDeviceInformation(DeviceInformation* info), 28 bytes:
/// { u32 status; u32 reserve0; u32 panelWidth; u32 panelHeight;
///   u16 latency90Hz; u16 latency120Hz; u8 hmuMount; u8 lensSeparation;
///   u8 reserve1[2]; f32 virtualImageDistance }
fn cb_get_device_information(ctx: &mut CpuContext) -> HleResult {
    if !INITIALIZED.load(Ordering::Acquire) {
        return ERROR_NOT_INITIALIZED;
    }
    let info = ctx.rdi();
    if info == 0 {
        return ERROR_PARAMETER_NULL;
    }
    let mut b = [0u8; 28];
    if VR_ENABLED {
        // READY, mounted, 63 mm lens separation, 2 m virtual image distance.
        b[8..12].copy_from_slice(&PANEL_WIDTH.to_le_bytes());
        b[12..16].copy_from_slice(&PANEL_HEIGHT.to_le_bytes());
        b[20] = 1;
        b[21] = 63;
        b[24..28].copy_from_slice(&2.0f32.to_le_bytes());
    } else {
        // SCE_HMD2_DEVICE_STATUS_NOT_DETECTED
        b[0..4].copy_from_slice(&2u32.to_le_bytes());
    }
    ctx.write_bytes(info, &b);
    0
}

/// sceHmd2GetFieldOfViewWithoutHandle(FieldOfView* fov): tangents
/// { out, in, top, bottom } of a 55/42.5/51.2/51.2 degree field of view.
fn cb_get_field_of_view_without_handle(ctx: &mut CpuContext) -> HleResult {
    if !INITIALIZED.load(Ordering::Acquire) {
        return ERROR_NOT_INITIALIZED;
    }
    let fov = ctx.rdi();
    if fov == 0 {
        return ERROR_PARAMETER_NULL;
    }
    let mut b = [0u8; 16];
    for (i, v) in [1.428_148_f32, 0.916_331_2, 1.243_749_1, 1.243_749_1].iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    ctx.write_bytes(fov, &b);
    0
}

/// sceHmd2ReprojectionQueryBufferSizeAlign() -> SizeAlign { size, align };
/// the size is returned in RAX.
fn cb_reprojection_query_buffer_size_align(_ctx: &mut CpuContext) -> HleResult {
    log::debug!("[hmd2] reprojection work size={REPROJECTION_STATE_SIZE} align={REPROJECTION_STATE_ALIGN}");
    REPROJECTION_STATE_SIZE as HleResult
}

/// sceHmd2ReprojectionQueryDisplayBufferSizeAlign() -> SizeAlign; the size
/// is returned in RAX (alignment: one 64 KiB block).
fn cb_reprojection_query_display_buffer_size_align(_ctx: &mut CpuContext) -> HleResult {
    log::debug!("[hmd2] reprojection display size={:#x} align={RT_BLOCK_SIZE:#x}", display_size());
    display_size() as HleResult
}

/// sceHmd2ReprojectionInitialize(const ReprojectionInitializeParam* param, void*)
/// param: { void* reprojectionBuff; void* displayBuff; i32 threadPriority;
///          i32 cpuAffinityMask; u32 pipeId; u32 queueId; u32 type;
///          void* seeThroughBuff @40; i32 reprojectionTiming @48; u32 reserved[5] }
fn cb_reprojection_initialize(ctx: &mut CpuContext) -> HleResult {
    let _g = REPROJECTION_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    if reprojection_state().is_some() {
        return ERROR_REPROJECTION_ALREADY_INITIALIZED;
    }
    let param = ctx.rdi();
    if param == 0 {
        return ERROR_PARAMETER_NULL;
    }
    let Some(p) = ctx.read_bytes(param, 72) else { return ERROR_PARAMETER_NULL };
    let q = |o: usize| u64::from_le_bytes(p[o..o + 8].try_into().unwrap());
    let (work, display, see_through) = (q(0), q(8), q(40));
    let timing = i32::from_le_bytes(p[48..52].try_into().unwrap());
    if work == 0 || display == 0 {
        return ERROR_PARAMETER_NULL;
    }
    if see_through != 0 {
        log::warn!("[hmd2] see-through buffer {see_through:#x} is not supported");
    }
    let timing_us = if timing != 0 { timing as u32 } else { DEFAULT_TIMING_US };
    let mut state = [0u8; REPROJECTION_STATE_SIZE as usize];
    state[8..12].copy_from_slice(&timing_us.to_le_bytes());
    ctx.write_bytes(work, &state);
    REPROJECTION_STATE.store(work, Ordering::Release);
    log::info!("[hmd2] reprojection initialized work={work:#x} display={display:#x} size={:#x} timing={timing_us}", display_size());
    0
}

/// sceHmd2ReprojectionEnableVrMode(u64 outputMode)
fn cb_reprojection_enable_vr_mode(ctx: &mut CpuContext) -> HleResult {
    let Some(state) = reprojection_state() else { return ERROR_REPROJECTION_NOT_INITIALIZED };
    let mode = ctx.rdi();
    if mode != OUTPUT_MODE_89_91HZ && mode != OUTPUT_MODE_119_88HZ {
        return VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if !VR_ENABLED {
        return VIDEO_OUT_ERROR_NO_DEVICE;
    }
    let _g = REPROJECTION_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    if ctx.read_u64(state).unwrap_or(0) != 0 {
        return ERROR_REPROJECTION_IN_VR_MODE;
    }
    ctx.write_u64(state, mode);
    0
}

/// sceHmd2ReprojectionSetTiming(i32 timing): microseconds, rounded up to
/// a multiple of 100 (2000..=7000 supported).
fn cb_reprojection_set_timing(ctx: &mut CpuContext) -> HleResult {
    let Some(state) = reprojection_state() else { return ERROR_REPROJECTION_NOT_INITIALIZED };
    let timing = ctx.rdi() as i32;
    if !(2000..=7000).contains(&timing) {
        log::warn!("[hmd2] unsupported reprojection timing {timing}");
    }
    let rounded = ((timing + 99) / 100 * 100) as u32;
    ctx.write_u32(state + 8, rounded);
    0
}

/// Exports whose NIDs are not in the symbol catalog.
const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("c812oYs7Vsc", "sceHmd2Initialize", cb_initialize),
    ("QU2M1pPNbaY", "sceHmd2Terminate", cb_terminate),
    ("bIi4YUfSRys", "sceHmd2GetDeviceInformation", cb_get_device_information),
    ("gF8+lvc7GuQ", "sceHmd2GetFieldOfViewWithoutHandle", cb_get_field_of_view_without_handle),
    ("U-CnbmeyYaA", "sceHmd2ReprojectionQueryBufferSizeAlign", cb_reprojection_query_buffer_size_align),
    ("-C2nkoEYOnU", "sceHmd2ReprojectionQueryDisplayBufferSizeAlign", cb_reprojection_query_display_buffer_size_align),
    ("C0rPwER-yxg", "sceHmd2ReprojectionInitialize", cb_reprojection_initialize),
    ("VVvFh51o20s", "sceHmd2ReprojectionEnableVrMode", cb_reprojection_enable_vr_mode),
    ("FkQX7rjFomk", "sceHmd2ReprojectionSetTiming", cb_reprojection_set_timing),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
    unnamed::register(reg, LIB);
    crate::hmd::register(reg);
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
    fn hmd2_without_headset() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D53_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_get_device_information(&mut ctx), ERROR_NOT_INITIALIZED);
        ctx.set(Rdi, 0);
        assert_eq!(cb_initialize(&mut ctx), ERROR_PARAMETER_NULL);
        ctx.set(Rdi, BASE + 0x800);
        assert_eq!(cb_initialize(&mut ctx), 0);
        assert_eq!(cb_initialize(&mut ctx), ERROR_ALREADY_INITIALIZED);
        ctx.set(Rdi, BASE);
        ctx.write_bytes(BASE, &[0xAA; 28]);
        assert_eq!(cb_get_device_information(&mut ctx), 0);
        let info = ctx.read_bytes(BASE, 28).unwrap();
        assert_eq!(&info[0..4], &2u32.to_le_bytes());
        assert!(info[4..].iter().all(|&b| b == 0));
        assert_eq!(cb_get_field_of_view_without_handle(&mut ctx), 0);
        assert_eq!(cb_reprojection_query_buffer_size_align(&mut ctx), 16);
        assert_eq!(cb_reprojection_query_display_buffer_size_align(&mut ctx), 4096 * 2048 * 4);

        // Reprojection.
        assert_eq!(cb_reprojection_set_timing(&mut ctx), ERROR_REPROJECTION_NOT_INITIALIZED);
        let param = BASE + 0x100;
        ctx.write_bytes(param, &[0; 72]);
        ctx.set(Rdi, param);
        assert_eq!(cb_reprojection_initialize(&mut ctx), ERROR_PARAMETER_NULL);
        ctx.write_u64(param, BASE + 0x200);
        ctx.write_u64(param + 8, BASE + 0x300);
        assert_eq!(cb_reprojection_initialize(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x208), Some(DEFAULT_TIMING_US));
        assert_eq!(cb_reprojection_initialize(&mut ctx), ERROR_REPROJECTION_ALREADY_INITIALIZED);
        ctx.set(Rdi, 2050);
        assert_eq!(cb_reprojection_set_timing(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x208), Some(2100));
        ctx.set(Rdi, 1);
        assert_eq!(cb_reprojection_enable_vr_mode(&mut ctx), VIDEO_OUT_ERROR_INVALID_VALUE);
        ctx.set(Rdi, OUTPUT_MODE_119_88HZ);
        assert_eq!(cb_reprojection_enable_vr_mode(&mut ctx), VIDEO_OUT_ERROR_NO_DEVICE);

        // Terminate ends the session once; queries need a new Initialize.
        assert_eq!(cb_terminate(&mut ctx), 0);
        assert_eq!(cb_terminate(&mut ctx), ERROR_NOT_INITIALIZED);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_get_device_information(&mut ctx), ERROR_NOT_INITIALIZED);
        ctx.set(Rdi, BASE + 0x800);
        assert_eq!(cb_initialize(&mut ctx), 0);
    }
}
