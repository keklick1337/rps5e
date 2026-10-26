// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! System library calls whose firmware implementation is a single fixed
//! answer (the library does nothing a title can observe): debug versions,
//! test hooks, stubs that report "not supported" with the module's own code.

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const fn code(v: u32) -> HleResult {
    v as i32 as HleResult
}

fn fw_ok(_ctx: &mut CpuContext) -> HleResult {
    0
}
fn fw_2(_ctx: &mut CpuContext) -> HleResult {
    2
}
fn fw_0x1e(_ctx: &mut CpuContext) -> HleResult {
    0x1E
}
fn fw_0x6a00(_ctx: &mut CpuContext) -> HleResult {
    0x6A00
}
fn fw_0x80ad0008(_ctx: &mut CpuContext) -> HleResult {
    code(0x80AD_0008)
}
fn fw_0x809d000f(_ctx: &mut CpuContext) -> HleResult {
    code(0x809D_000F)
}
fn fw_0x80ed0004(_ctx: &mut CpuContext) -> HleResult {
    code(0x80ED_0004)
}
fn fw_0x80bc00ff(_ctx: &mut CpuContext) -> HleResult {
    code(0x80BC_00FF)
}
fn fw_0x809b00ff(_ctx: &mut CpuContext) -> HleResult {
    code(0x809B_00FF)
}

/// sceAjmInstanceCodecType(instanceId): the codec is the bits above the
/// 14-bit instance index.
fn fw_ajm_codec_type(ctx: &mut CpuContext) -> HleResult {
    (ctx.rdi() as u32 >> 14) as HleResult
}

const EXPORTS: &[(&str, &str, &str, HleCallback)] = &[
    ("L5jHxZ-J630", "sceBluetoothHidDebugGetVersion", "libSceBluetoothHid", fw_0x1e),
    ("cE5Msy11WhU", "sceCompanionUtilGetEvent", "libSceCompanionUtil", fw_0x80ad0008),
    ("MaVrz79mT5o", "sceCompanionUtilGetRemoteOskEvent", "libSceCompanionUtil", fw_ok),
    ("xb1xlIhf0QY", "sceCompanionUtilInitialize", "libSceCompanionUtil", fw_ok),
    ("IPN-FRSrafk", "sceCompanionUtilOptParamInitialize", "libSceCompanionUtil", fw_ok),
    ("H1fYQd5lFAI", "sceCompanionUtilTerminate", "libSceCompanionUtil", fw_ok),
    ("hvfrzWFz2aE", "sceContentBinderEnd", "libSceContentSearch", fw_0x809d000f),
    ("IElCGP1gobk", "sceContentBinderGetNumOfDifference", "libSceContentSearch", fw_0x809d000f),
    ("V5NUMrn5cEQ", "sceContentBinderInitialize", "libSceContentSearch", fw_0x809d000f),
    ("W4kZjfF08xU", "sceContentBinderIsUpdated", "libSceContentSearch", fw_0x809d000f),
    ("tzgPpLnmFbc", "sceContentBinderReset", "libSceContentSearch", fw_0x809d000f),
    ("gYr9dLq8Y10", "sceContentBinderSearchDifference", "libSceContentSearch", fw_0x809d000f),
    ("mlBEgk9o0SE", "sceContentBinderStart", "libSceContentSearch", fw_0x809d000f),
    ("XYm1wQQEDc0", "sceContentBinderTerminate", "libSceContentSearch", fw_0x809d000f),
    ("jrpnVQfJYgQ", "sceErrorDialogOpenDetail", "libSceErrorDialog", fw_0x80ed0004),
    ("wktCiyWoDTI", "sceErrorDialogOpenWithReport", "libSceErrorDialog", fw_0x80ed0004),
    ("0gRbLi4WKVg", "sceHttpCacheSystemSendStatistics", "libSceHttpCache", fw_ok),
    ("J25sxFmD+1w", "sceHttpCacheSystemTerm", "libSceHttpCache", fw_ok),
    ("UFcyYDf+e88", "sceImeDialogForTestFunction", "libSceImeDialog", fw_0x80bc00ff),
    ("wVkehxutK-U", "sceImeForTestFunction", "libSceIme", fw_0x80bc00ff),
    ("BIQaOpcytRc", "sceMbusAcquireControlList", "libSceMbus", fw_0x809b00ff),
    ("hykD8qmO6XA", "sceMbusTerm", "libSceMbus", fw_ok),
    ("16efS+mFGGc", "sceMoveTrackerGetVersion", "libSceMoveTracker", fw_2),
    ("7ZGS7F0Nl4E", "sceMoveTrackerRecordStop", "libSceMoveTracker", fw_ok),
    ("-Caws0X3+bY", "sceSystemGestureDebugGetVersion", "libSceSystemGesture", fw_0x6a00),
    ("diXjQNiMu-s", "sceAjmInstanceCodecType", "libSceAjm", fw_ajm_codec_type),
    ("Lffcxao-QMM", "sceHttpDbgEnableProfile", "libSceHttp", fw_ok),
    ("VmqSnjZ5mE4", "sceHttpDbgSetPrintf", "libSceHttp", fw_ok),
    ("+cE4Jx431wc", "scePadEnableExtensionPort", "libScePad", fw_ok),
    ("4OPOZxfVkHA", "sceSaveDataCheckIpmiIfSize", "libSceSaveData", fw_ok),
    ("gW6G4HxBBXA", "sceSaveDataDebug", "libSceSaveData", fw_ok),
    ("J23ENpIFoRc", "sceSaveDataDebugCheckBackupData", "libSceSaveData", fw_ok),
    ("bYCnxLexU7M", "sceSaveDataDebugCleanMount", "libSceSaveData", fw_ok),
    ("hVDqYB8+jkk", "sceSaveDataDebugCompiledSdkVersion", "libSceSaveData", fw_ok),
    ("K9gXXlrVLNI", "sceSaveDataDebugCreateSaveDataRoot", "libSceSaveData", fw_ok),
    ("ZIao0n0x1-w", "sceSaveDataDebugProspero", "libSceSaveData", fw_ok),
    ("UGTldPVEdB4", "sceSaveDataDebugRemoveSaveDataRoot", "libSceSaveData", fw_ok),
    ("AYBQmnRplrg", "sceSaveDataDebugTarget", "libSceSaveData", fw_ok),
    ("RXXBl6-iimo", "sceVideoRecordingInit", "libSceVideoRecording", fw_ok),
    ("gWZikoR+g9g", "sceVideoRecordingTerm", "libSceVideoRecording", fw_ok),
    ("mEy7di8MT30", "sceVideoRecordingSendCommand", "libSceVideoRecording", fw_ok),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, lib, cb) in EXPORTS {
        reg.register_fn(nid, name, lib, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn fixed_answers() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        let answer = |ctx: &mut CpuContext, name: &str| (EXPORTS.iter().find(|e| e.1 == name).unwrap().3)(ctx);
        assert_eq!(answer(&mut ctx, "sceBluetoothHidDebugGetVersion"), 0x1E);
        assert_eq!(answer(&mut ctx, "sceContentBinderStart"), code(0x809D_000F));
        assert_eq!(answer(&mut ctx, "sceMoveTrackerGetVersion"), 2);
        ctx.set(crate::context::CpuRegister::Rdi, (2 << 14) | 5);
        assert_eq!(answer(&mut ctx, "sceAjmInstanceCodecType"), 2);
    }
}
