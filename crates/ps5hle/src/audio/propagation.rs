// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceAudioPropagation (3D acoustic ray/portal/room propagation).
//!
//! Propagation is not modelled. The API is placement-style (the title
//! allocates the memory QueryMemory reports and every object lives inside
//! it), so successful entry points let audio initialisation proceed.

use super::Export;
use crate::context::CpuContext;
use crate::registry::HleResult;

pub const AUDIO_PROPAGATION_ERROR_INVALID_POINTER: HleResult = 0x8A70_0003u32 as i32 as i64;

/// SceAudioPropagationSystemMemory: desc {u32 id; size_t size} at 0x00,
/// pCpuMem 0x10, sizeCpuMem 0x18, pGpuMem 0x20, sizeGpuMem 0x28.
const SYSTEM_MEMORY_CPU_SIZE_OFFSET: u64 = 0x18;
const SYSTEM_MEMORY_GPU_SIZE_OFFSET: u64 = 0x28;
const SYSTEM_MEMORY_CPU_BYTES: u64 = 0x10_0000;

fn cb_ok(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// sceAudioPropagationSystemQueryMemory(const Option *, SystemMemory *out):
/// only the two size fields belong to this call.
fn cb_system_query_memory(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return AUDIO_PROPAGATION_ERROR_INVALID_POINTER;
    }
    ctx.write_u64(out + SYSTEM_MEMORY_CPU_SIZE_OFFSET, SYSTEM_MEMORY_CPU_BYTES);
    ctx.write_u64(out + SYSTEM_MEMORY_GPU_SIZE_OFFSET, 0);
    0
}

const LIB: &str = "libSceAudioPropagation";

pub(crate) static EXPORTS: &[Export] = &[
    ("7xyAxrusLko", "sceAudioPropagationSystemQueryMemory", LIB, cb_system_query_memory),
    ("GrA9ke1QT+E", "sceAudioPropagationSystemQueryInfo", LIB, cb_ok),
    ("aNEqtSHdUSo", "sceAudioPropagationSystemCreate", LIB, cb_ok),
    ("x5VPqg5iyAk", "sceAudioPropagationSystemDestroy", LIB, cb_ok),
    ("ile38Gl-p5M", "sceAudioPropagationSystem", LIB, cb_ok),
    ("cMl3u+7QBBM", "sceAudioPropagationSystemMemoryInit", LIB, cb_ok),
    ("3B9IabLByyM", "sceAudioPropagationSystemOptionInit", LIB, cb_ok),
    ("B2KI2AachWE", "sceAudioPropagationSystemLock", LIB, cb_ok),
    ("kIdb+iQUzCs", "sceAudioPropagationSystemSetAttributes", LIB, cb_ok),
    ("VlBT16890mA", "sceAudioPropagationSystemSetRays", LIB, cb_ok),
    ("ht-QXT3zGxo", "sceAudioPropagationSystemGetRays", LIB, cb_ok),
    ("CPLV6G-eXmk", "sceAudioPropagationSystemRegisterMaterial", LIB, cb_ok),
    ("XKCN4gpeYsM", "sceAudioPropagationSystemUnregisterMaterial", LIB, cb_ok),
    ("8bI5h8req30", "sceAudioPropagationRoomCreate", LIB, cb_ok),
    ("S0JwP2AFTTE", "sceAudioPropagationRoomDestroy", LIB, cb_ok),
    ("b-dYXrjSNZU", "sceAudioPropagationPortalCreate", LIB, cb_ok),
    ("ZQXE-xS6MTE", "sceAudioPropagationPortalDestroy", LIB, cb_ok),
    ("WXMhENV2NcA", "sceAudioPropagationPortalSetAttributes", LIB, cb_ok),
    ("i687TNRF+hw", "sceAudioPropagationPortalSettingsInit", LIB, cb_ok),
    ("d84otraxt2s", "sceAudioPropagationSourceCreate", LIB, cb_ok),
    ("wkseM3LWPuc", "sceAudioPropagationSourceDestroy", LIB, cb_ok),
    ("-wsUTr31yeg", "sceAudioPropagationSourceSetAttributes", LIB, cb_ok),
    ("PBcrVpEqUVY", "sceAudioPropagationSourceCalculateAudioPaths", LIB, cb_ok),
    ("eEeKqFeNI3o", "sceAudioPropagationSourceGetAudioPath", LIB, cb_ok),
    ("G+QLTfyLMYk", "sceAudioPropagationSourceGetAudioPathCount", LIB, cb_ok),
    ("aKJZx7wCma8", "sceAudioPropagationSourceGetRays", LIB, cb_ok),
    ("3aEY9tPXGKc", "sceAudioPropagationSourceQueryInfo", LIB, cb_ok),
    ("hhz9pITnC8k", "sceAudioPropagationSourceRender", LIB, cb_ok),
    ("SoKPzY1-3SU", "sceAudioPropagationSourceRenderInfoInit", LIB, cb_ok),
    ("tKSmk2JsMAA", "sceAudioPropagationSourceSetAudioPath", LIB, cb_ok),
    ("5vzOS2pHMFc", "sceAudioPropagationSourceSetAudioPaths", LIB, cb_ok),
    ("MNmGapXrYRs", "sceAudioPropagationSourceSetAudioPathsParamInit", LIB, cb_ok),
    ("i-0aUex3zCE", "sceAudioPropagationAudioPathInit", LIB, cb_ok),
    ("JZIkSbmt2BE", "sceAudioPropagationAudioPathPointInit", LIB, cb_ok),
    ("tL2AEPejVQE", "sceAudioPropagationPathGetNumPoints", LIB, cb_ok),
    ("2BSFmuKtRss", "sceAudioPropagationMaterialInit", LIB, cb_ok),
    ("0r2+9UTg1BA", "sceAudioPropagationRayInit", LIB, cb_ok),
    ("BbOT4vBwAjs", "sceAudioPropagationResetAttributes", LIB, cb_ok),
    ("gCmQm6dvMxw", "sceAudioPropagationReportApi", LIB, cb_ok),
];
