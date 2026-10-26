// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceHmd — PlayStation VR (first generation) headset.
//!
//! No headset is ever attached: the library initialises, device queries
//! report "not detected", opening a device fails with DEVICE_DISCONNECTED
//! and, as no handle is ever handed out, every handle is invalid.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceHmd";

const fn err(code: u32) -> HleResult {
    code as i32 as i64
}

const OK: HleResult = 0;
const ERR_ALREADY_INITIALIZED: HleResult = err(0x8111_0001);
const ERR_NOT_INITIALIZED: HleResult = err(0x8111_0002);
const ERR_INVALID_HANDLE: HleResult = err(0x8111_0003);
const ERR_DEVICE_DISCONNECTED: HleResult = err(0x8111_0004);
const ERR_PARAMETER_NULL: HleResult = err(0x8111_0008);
const ERR_PARAMETER_INVALID: HleResult = err(0x8111_0009);
const ERR_UNSUPPORTED_FEATURE: HleResult = err(0x8111_0016);
const ERR_MEMORY_FAULT: HleResult = err(0x8002_000E);

/// SCE_HMD_DEVICE_STATUS_NOT_DETECTED.
const DEVICE_STATUS_NOT_DETECTED: u32 = 2;
/// SceHmdDeviceInformation: status u32, userId i32, reserved0[4],
/// panelWidth u32, panelHeight u32, latency90Hz u16, latency120Hz u16,
/// hmuMount u8, reserved1[7].
const DEVICE_INFORMATION_SIZE: usize = 32;
/// SCE_USER_SERVICE_USER_ID_SYSTEM.
const USER_ID_SYSTEM: i32 = 0xFF;

static INITIALIZED: AtomicBool = AtomicBool::new(false);

fn initialized() -> bool {
    INITIALIZED.load(Ordering::Acquire)
}

/// sceHmdInitialize / sceHmdInitialize315(const SceHmdInitializeParam*):
/// the param is { void* reserved0; u8 reserved[8] }; a non-null reserved0
/// asks for the distortion work area, which is not supported.
fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    if initialized() {
        return ERR_ALREADY_INITIALIZED;
    }
    let param = ctx.rdi();
    if param == 0 {
        return ERR_PARAMETER_NULL;
    }
    let Some(reserved0) = ctx.read_u64(param) else { return ERR_MEMORY_FAULT };
    if reserved0 != 0 {
        log::warn!("[hmd] distortion work area {reserved0:#x} is not supported");
        return ERR_UNSUPPORTED_FEATURE;
    }
    if INITIALIZED.swap(true, Ordering::AcqRel) {
        return ERR_ALREADY_INITIALIZED;
    }
    log::info!("[hmd] initialized, no headset attached");
    OK
}

fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    if INITIALIZED.swap(false, Ordering::AcqRel) { OK } else { ERR_NOT_INITIALIZED }
}

/// sceHmdOpen(userId, type, index, const SceHmdOpenParam* param): only the
/// main headset (type 0, index 0, no param) of a user can be opened, and
/// none is connected.
fn cb_open(ctx: &mut CpuContext) -> HleResult {
    if !initialized() {
        return ERR_NOT_INITIALIZED;
    }
    let (user, ty, index, param) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx() as i32, ctx.rcx());
    if ty != 0 || index != 0 || param != 0 || user == -1 || user == USER_ID_SYSTEM {
        return ERR_PARAMETER_INVALID;
    }
    ERR_DEVICE_DISCONNECTED
}

/// sceHmdClose(handle): no handle was ever opened.
fn cb_close(_ctx: &mut CpuContext) -> HleResult {
    if initialized() { ERR_INVALID_HANDLE } else { ERR_NOT_INITIALIZED }
}

/// sceHmdGetDeviceInformation(SceHmdDeviceInformation*): not detected, no
/// user. A failed query leaves the output untouched.
fn cb_get_device_information(ctx: &mut CpuContext) -> HleResult {
    let info = ctx.rdi();
    if info == 0 {
        return ERR_PARAMETER_NULL;
    }
    if !initialized() {
        return ERR_NOT_INITIALIZED;
    }
    let mut b = [0u8; DEVICE_INFORMATION_SIZE];
    b[0..4].copy_from_slice(&DEVICE_STATUS_NOT_DETECTED.to_le_bytes());
    b[4..8].copy_from_slice(&(-1i32).to_le_bytes());
    if ctx.write_bytes(info, &b) { OK } else { ERR_MEMORY_FAULT }
}

/// sceHmdGetDeviceInformationByHandle(handle, info*): no handle is valid.
fn cb_get_device_information_by_handle(_ctx: &mut CpuContext) -> HleResult {
    if initialized() { ERR_INVALID_HANDLE } else { ERR_NOT_INITIALIZED }
}

/// sceHmdInternalGetDeviceStatus(SceHmdDeviceStatus*): usable without
/// initialisation.
fn cb_internal_get_device_status(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_PARAMETER_NULL;
    }
    if ctx.write_u32(out, DEVICE_STATUS_NOT_DETECTED) { OK } else { ERR_MEMORY_FAULT }
}

/// sceHmdInternalMmapIsConnect(): 0, no headset connection.
fn cb_internal_mmap_is_connect(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// sceHmdGetAssyError(data*): the headset is not connected.
fn cb_get_assy_error(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return ERR_PARAMETER_NULL;
    }
    if initialized() { ERR_DEVICE_DISCONNECTED } else { ERR_NOT_INITIALIZED }
}

/// sceHmdGetFieldOfView(handle, fov*): no handle is valid.
fn cb_get_field_of_view(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 {
        return ERR_PARAMETER_NULL;
    }
    if initialized() { ERR_INVALID_HANDLE } else { ERR_NOT_INITIALIZED }
}

/// sceHmdGet2DEyeOffset(handle, left*, right*) and
/// sceHmdGetInertialSensorData(handle, data*, count): no handle is valid
/// (the outputs are not checked first).
fn cb_handle_query(_ctx: &mut CpuContext) -> HleResult {
    if initialized() { ERR_INVALID_HANDLE } else { ERR_NOT_INITIALIZED }
}

/// The distortion and reprojection buffer requirements the library
/// reports (fixed values, no headset needed).
fn cb_align_0x400(_ctx: &mut CpuContext) -> HleResult {
    0x400
}
fn cb_size_0x20000(_ctx: &mut CpuContext) -> HleResult {
    0x20000
}
fn cb_align_0x100(_ctx: &mut CpuContext) -> HleResult {
    0x100
}
fn cb_size_0x810(_ctx: &mut CpuContext) -> HleResult {
    0x810
}
fn cb_size_0x100000(_ctx: &mut CpuContext) -> HleResult {
    0x10_0000
}

/// sceHmdDistortionGetCompoundEyeCorrectionCommand: what the firmware
/// returns (PARAMETER_INVALID).
fn cb_compound_eye_correction(_ctx: &mut CpuContext) -> HleResult {
    ERR_PARAMETER_INVALID
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("Yx+CuF11D3Q", "sceHmdGetAssyError", cb_get_assy_error),
    ("NPQwYFqi0bs", "sceHmdGetFieldOfView", cb_get_field_of_view),
    ("BWY-qKM5hxE", "sceHmdGet2DEyeOffset", cb_handle_query),
    ("rU3HK9Q0r8o", "sceHmdGetInertialSensorData", cb_handle_query),
    ("1cS7W5J-v3k", "sceHmdDistortionGetWorkMemoryAlign", cb_align_0x400),
    ("36xDKk+Hw7o", "sceHmdDistortionGetWorkMemorySize", cb_size_0x20000),
    ("8Ick-e6cDVY", "sceHmdGetDistortionWorkMemoryAlign", cb_align_0x400),
    ("D5JfdpJKvXk", "sceHmdGetDistortionWorkMemorySize", cb_size_0x20000),
    ("IWybWbR-xvA", "sceHmdReprojectionQueryOnionBuffAlign", cb_align_0x100),
    ("TkcANcGM0s8", "sceHmdReprojectionQueryGarlicBuffAlign", cb_align_0x100),
    ("kLUAkN6a1e8", "sceHmdReprojectionQueryOnionBuffSize", cb_size_0x810),
    ("z0KtN1vqF2E", "sceHmdReprojectionQueryGarlicBuffSize", cb_size_0x100000),
    ("ER2ar8yUmbk", "sceHmdDistortionGetCompoundEyeCorrectionCommand", cb_compound_eye_correction),
    ("K4KnH0QkT2c", "sceHmdInitialize", cb_initialize),
    ("s-J66ar9g50", "sceHmdInitialize315", cb_initialize),
    ("z-RMILqP6tE", "sceHmdTerminate", cb_terminate),
    ("d2g5Ij7EUzo", "sceHmdOpen", cb_open),
    ("6biw1XHTSqQ", "sceHmdClose", cb_close),
    ("thDt9upZlp8", "sceHmdGetDeviceInformation", cb_get_device_information),
    ("1pxQfif1rkE", "sceHmdGetDeviceInformationByHandle", cb_get_device_information_by_handle),
    ("PPCqsD8B5uM", "sceHmdInternalGetDeviceStatus", cb_internal_get_device_status),
    ("1PNiQR-7L6k", "sceHmdInternalMmapIsConnect", cb_internal_mmap_is_connect),
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

    fn open(ctx: &mut CpuContext, user: i32, ty: i32, index: i32, param: u64) -> HleResult {
        ctx.set(Rdi, user as u32 as u64);
        ctx.set(Rsi, ty as u32 as u64);
        ctx.set(Rdx, index as u32 as u64);
        ctx.set(Rcx, param);
        cb_open(ctx)
    }

    fn check_disconnected(ctx: &mut CpuContext, out: u64) {
        ctx.write_u32(out, 0);
        ctx.set(Rdi, out);
        assert_eq!(cb_internal_get_device_status(ctx), OK);
        assert_eq!(ctx.read_u32(out), Some(DEVICE_STATUS_NOT_DETECTED));
        ctx.set(Rdi, 0);
        assert_eq!(cb_internal_get_device_status(ctx), ERR_PARAMETER_NULL);
        assert_eq!(cb_internal_mmap_is_connect(ctx), 0);
    }

    /// Port of the guest headset test: lifecycle, absent-device output and
    /// argument checks with no headset attached.
    #[test]
    fn no_headset_attached() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5FA0_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (param, info, status) = (BASE, BASE + 0x100, BASE + 0x200);
        ctx.write_bytes(param, &[0; 16]);
        ctx.write_bytes(info, &[0xA5; DEVICE_INFORMATION_SIZE]);

        check_disconnected(&mut ctx, status);
        ctx.set(Rdi, status);
        ctx.set(Rsi, status);
        assert_eq!(cb_get_assy_error(&mut ctx), ERR_NOT_INITIALIZED);
        assert_eq!(cb_get_field_of_view(&mut ctx), ERR_NOT_INITIALIZED);
        assert_eq!(cb_handle_query(&mut ctx), ERR_NOT_INITIALIZED);
        assert_eq!(cb_terminate(&mut ctx), ERR_NOT_INITIALIZED);
        assert_eq!(open(&mut ctx, 1, 0, 0, 0), ERR_NOT_INITIALIZED);
        assert_eq!(cb_close(&mut ctx), ERR_NOT_INITIALIZED);
        ctx.set(Rdi, 0);
        assert_eq!(cb_get_device_information(&mut ctx), ERR_PARAMETER_NULL);
        ctx.set(Rdi, info);
        assert_eq!(cb_get_device_information(&mut ctx), ERR_NOT_INITIALIZED);
        assert_eq!(ctx.read_bytes(info, DEVICE_INFORMATION_SIZE).unwrap(), vec![0xA5; DEVICE_INFORMATION_SIZE]);
        assert_eq!(cb_get_device_information_by_handle(&mut ctx), ERR_NOT_INITIALIZED);
        ctx.set(Rdi, 0);
        assert_eq!(cb_initialize(&mut ctx), ERR_PARAMETER_NULL);
        ctx.write_u64(param, param);
        ctx.set(Rdi, param);
        assert_eq!(cb_initialize(&mut ctx), ERR_UNSUPPORTED_FEATURE, "distortion unsupported");
        ctx.write_u64(param, 0);

        for _ in 0..2 {
            ctx.set(Rdi, param);
            assert_eq!(cb_initialize(&mut ctx), OK);
            assert_eq!(cb_initialize(&mut ctx), ERR_ALREADY_INITIALIZED);
            ctx.set(Rdi, 0);
            assert_eq!(cb_initialize(&mut ctx), ERR_ALREADY_INITIALIZED, "duplicate before null check");
            check_disconnected(&mut ctx, status);
            ctx.set(Rdi, info);
            assert_eq!(cb_get_device_information(&mut ctx), OK);
            let mut expected = [0u8; DEVICE_INFORMATION_SIZE];
            expected[0] = 2;
            expected[4..8].copy_from_slice(&[0xFF; 4]);
            assert_eq!(ctx.read_bytes(info, DEVICE_INFORMATION_SIZE).unwrap(), expected);

            ctx.set(Rdi, 0);
            assert_eq!(cb_get_assy_error(&mut ctx), ERR_PARAMETER_NULL);
            ctx.set(Rdi, status);
            assert_eq!(cb_get_assy_error(&mut ctx), ERR_DEVICE_DISCONNECTED);
            ctx.set(Rsi, 0);
            assert_eq!(cb_get_field_of_view(&mut ctx), ERR_PARAMETER_NULL);
            ctx.set(Rsi, status);
            assert_eq!(cb_get_field_of_view(&mut ctx), ERR_INVALID_HANDLE);
            assert_eq!(cb_handle_query(&mut ctx), ERR_INVALID_HANDLE);
            assert_eq!((cb_align_0x400(&mut ctx), cb_size_0x20000(&mut ctx)), (0x400, 0x20000));
            assert_eq!((cb_align_0x100(&mut ctx), cb_size_0x810(&mut ctx), cb_size_0x100000(&mut ctx)), (0x100, 0x810, 0x10_0000));
            assert_eq!(open(&mut ctx, -1, 0, 0, 0), ERR_PARAMETER_INVALID);
            assert_eq!(open(&mut ctx, 0xFF, 0, 0, 0), ERR_PARAMETER_INVALID);
            assert_eq!(open(&mut ctx, 1, 1, 0, 0), ERR_PARAMETER_INVALID);
            assert_eq!(open(&mut ctx, 1, 0, 1, 0), ERR_PARAMETER_INVALID);
            assert_eq!(open(&mut ctx, 1, 0, 0, param), ERR_PARAMETER_INVALID);
            assert_eq!(open(&mut ctx, 1, 0, 0, 0), ERR_DEVICE_DISCONNECTED);
            for handle in [0u64, u32::MAX as u64, 1, 0x0F00_0000] {
                ctx.set(Rdi, handle);
                ctx.set(Rsi, info);
                assert_eq!(cb_close(&mut ctx), ERR_INVALID_HANDLE);
                assert_eq!(cb_get_device_information_by_handle(&mut ctx), ERR_INVALID_HANDLE);
            }
            assert_eq!(ctx.read_bytes(info, DEVICE_INFORMATION_SIZE).unwrap(), expected);
            assert_eq!(cb_terminate(&mut ctx), OK);
            assert_eq!(cb_terminate(&mut ctx), ERR_NOT_INITIALIZED);
            ctx.set(Rdi, info);
            assert_eq!(cb_get_device_information(&mut ctx), ERR_NOT_INITIALIZED);
        }
    }
}
