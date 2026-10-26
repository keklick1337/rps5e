// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceRazorCpu HLE: the Razor CPU profiler is never capturing, so job
//! manager instrumentation markers are accepted and dropped.

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceRazorCpu";

/// Marker calls: nothing records them without a profiler attached.
fn cb_marker(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// sceRazorCpuFlushOccurred(uint64_t* occurred): no capture buffer is ever
/// flushed.
fn cb_flush_occurred(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out != 0 {
        ctx.write_u64(out, 0);
    }
    0
}

/// sceRazorCpuIsCapturing() -> 0: no capture is ever in progress.
fn cb_is_capturing(_ctx: &mut CpuContext) -> HleResult {
    0
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    // Data tags and logical file accesses only annotate a capture, and no
    // capture is ever running.
    ("SfRTRZ1Sh+U", "sceRazorCpuDisableFiberUserMarkers", cb_marker),
    ("A7oRMdaOJP8", "sceRazorCpuInitDataTags", cb_marker),
    ("emklx7RK-LY", "sceRazorCpuShutdownDataTags", cb_marker),
    ("we3oTKSPSTw", "sceRazorCpuTagArray", cb_marker),
    ("vyjdThnQfQQ", "sceRazorCpuTagBuffer", cb_marker),
    ("D0yUjM33QqU", "sceRazorCpuResizeTaggedBuffer", cb_marker),
    ("0yNHPIkVTmw", "sceRazorCpuUnTagBuffer", cb_marker),
    ("JFzLJBlYIJE", "sceRazorCpuBeginLogicalFileAccess", cb_marker),
    ("gVioM9cbiDs", "sceRazorCpuEndLogicalFileAccess", cb_marker),
    ("dnEdyY4+klQ", "sceRazorCpuJobManagerDispatch", cb_marker),
    ("KP+TBWGHlgs", "sceRazorCpuJobManagerJob", cb_marker),
    ("9FowWFMEIM8", "sceRazorCpuJobManagerSequence", cb_marker),
    ("EboejOQvLL4", "sceRazorCpuIsCapturing", cb_is_capturing),
    ("zw+celG7zSI", "sceRazorCpuPushMarker", cb_marker),
    ("YpkGsMXP3ew", "sceRazorCpuPopMarker", cb_marker),
    ("uZrOwuNJX-M", "sceRazorCpuPushMarkerStatic", cb_marker),
    ("Ax7NjOzctIM", "sceRazorCpuSync", cb_marker),
    ("XCuZoBSVFG8", "sceRazorCpuNamedSync", cb_marker),
    ("njGikRrxkC0", "sceRazorCpuPlotValue", cb_marker),
    ("G3brhegfyNg", "sceRazorCpuWriteBookmark", cb_marker),
    ("sPhrQD31ClM", "sceRazorCpuFlushOccurred", cb_flush_occurred),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn flush_never_occurred() {
        const BASE: u64 = 0x5D45_0300_0000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_u64(BASE, 0x55);
        ctx.set(crate::context::CpuRegister::Rdi, BASE);
        assert_eq!(cb_flush_occurred(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE), Some(0));
        ctx.set(crate::context::CpuRegister::Rdi, 0);
        assert_eq!(cb_flush_occurred(&mut ctx), 0);
    }
}
