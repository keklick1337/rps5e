// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceVideoOut exports.

use super::*;
use crate::context::{CpuContext, CpuRegister};
use crate::registry::{HleRegistry, HleResult};
use ps5gpu::display::DisplayResolution;

const OK: HleResult = 0;
const BUFFER_ATTRIBUTE_SIZE: usize = 0x28;
const BUFFER_ATTRIBUTE2_SIZE: usize = 0x50;
const BUFFERS_ENTRY_SIZE: u64 = 0x20;
const OUTPUT_OPTIONS_SIZE: usize = 0x40;
const OUTPUT_STATUS_SIZE: usize = 0x30;
/// SceVideoOutOutputStatus followed by the VRR fields (not modelled: 0).
const OUTPUT_STATUS_WITH_VRR_SIZE: usize = 0x50;
/// Bytes of SceVideoOutColorSettings (gamma and three reserved words).
const COLOR_SETTINGS_SIZE: u64 = 0x10;
const FLIP_STATUS_SIZE: usize = 0x80;
const VBLANK_STATUS_SIZE: usize = 0x28;
const LATENCY_CONTROL_SIZE: usize = 0x20;
const LATENCY_WAIT_BY_FLIP_QUEUE_NUM: u32 = 0;
const LATENCY_WAIT_BY_FLIP_ARG: u32 = 1;
const MAX_LATENCY_EXTRA_USEC: u32 = 100_000;
const OUTPUT_MODE_119_88HZ: u64 = 0xF;
const REFRESH_RATE_59_94HZ: u64 = 3;
const REFRESH_RATE_119_88HZ: u64 = 13;
const RESOLUTION_STATUS_SIZE: usize = 0x30;
/// First word sceVideoOutInitializeOutputOptions writes (0xFF in byte 2).
const OUTPUT_OPTIONS_FIRST_WORD: u32 = 0x00FF_0000;
/// The options word the console accepts with any value.
const OUTPUT_OPTIONS_FREE_WORD: usize = 3;
/// Display options accepted by SubmitChangeBufferAttribute2: strict
/// colorimetry (bit 3) and premultiplied alpha (bit 5).
const SUPPORTED_OPTIONS: u64 = (1 << 3) | (1 << 5);

fn stack_u64(ctx: &CpuContext, index: u64) -> Option<u64> {
    ctx.read_u64(ctx.get(CpuRegister::Rsp) + 8 + index * 8)
}

fn read_i16(ctx: &CpuContext, addr: u64) -> Option<i16> {
    let b = ctx.read_bytes(addr, 2)?;
    Some(i16::from_le_bytes([b[0], b[1]]))
}

fn write_or_fault(ctx: &CpuContext, addr: u64, data: &[u8]) -> HleResult {
    if ctx.vm.write(addr, data).is_ok() {
        OK
    } else {
        ERROR_MEMORY_FAULT
    }
}

fn put_u32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

fn handle_arg(v: u64) -> i32 {
    v as i32
}

/// Read a v1 (0x28) or v2 (0x50) buffer attribute.
pub(crate) fn parse_buffer_attribute(bytes: &[u8], attribute2: bool) -> Option<BufferAttribute> {
    let need = if attribute2 { 0x28 } else { 0x1C };
    if bytes.len() < need {
        return None;
    }
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    let tiling_mode = u32_at(0x04);
    let width = u32_at(0x0C);
    let height = u32_at(0x10);
    if attribute2 {
        return Some(BufferAttribute {
            pixel_format: ps5gpu::format::normalize(u64_at(0x20)),
            tiling_mode,
            aspect_ratio: 0,
            width,
            height,
            pitch_in_pixel: width,
            option: u64_at(0x18),
        });
    }
    Some(BufferAttribute {
        pixel_format: ps5gpu::format::normalize(u32_at(0x00) as u64),
        tiling_mode,
        aspect_ratio: u32_at(0x08),
        width,
        height,
        pitch_in_pixel: u32_at(0x14),
        option: u32_at(0x18) as u64,
    })
}

fn read_buffer_attribute(ctx: &CpuContext, addr: u64, attribute2: bool) -> Option<BufferAttribute> {
    let len = if attribute2 { 0x28 } else { 0x1C };
    parse_buffer_attribute(&ctx.read_bytes(addr, len)?, attribute2)
}

/// Build the 0x30-byte SceVideoOutOutputStatus: resolution class (u32:
/// 1 under 2160p, 2 for 2160p, see [`output_resolution_class`]), dynamic
/// range (u32 at 0x04: 1 SDR, 2 HDR), refresh-rate code (u64 at 0x08),
/// flags and reserved words (zero).
pub(crate) fn output_status_bytes(resolution_class: u32, refresh_rate: u32, hdr: bool) -> [u8; OUTPUT_STATUS_SIZE] {
    let mut s = [0u8; OUTPUT_STATUS_SIZE];
    put_u32(&mut s, 0x00, resolution_class);
    put_u32(&mut s, 0x04, if hdr { 2 } else { 1 });
    // The status carries a refresh-rate code, not the pacing frequency.
    let code = if refresh_rate >= 119 { REFRESH_RATE_119_88HZ } else { REFRESH_RATE_59_94HZ };
    put_u64(&mut s, 0x08, code);
    s
}

/// param.json `attribute3` bit 2: the application detects the display's
/// resolution. Without it the system scales a 2160p output to the display
/// itself and the application is always told about a 2160p display.
pub(crate) const PARAM_ATTRIBUTE3_RESOLUTION_DETECTION: u32 = 0x4;

/// Integer member `key` of param.json (missing or not an unsigned 32-bit
/// number: 0).
fn param_u32(json: &str, key: &str) -> u32 {
    match crate::app_info::param_json_member(json, key) {
        Some(crate::app_info::JsonScalar::Num(n)) if n.fract() == 0.0 && (0.0..=u32::MAX as f64).contains(&n) => n as u32,
        _ => 0,
    }
}

/// param.json `attribute3` of an application; without a param.json at all
/// the application is treated as detecting the resolution.
pub(crate) fn param_attribute3(param_json: Option<&str>) -> u32 {
    param_json.map_or(PARAM_ATTRIBUTE3_RESOLUTION_DETECTION, |json| param_u32(json, "attribute3"))
}

/// [`param_attribute3`] of the running application (param.json read once).
fn app_attribute3() -> u32 {
    static ATTRIBUTE3: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *ATTRIBUTE3.get_or_init(|| param_attribute3(crate::app_info::read_param_json().as_deref()))
}

/// The display mode an application with param.json `attribute3` is shown
/// for the emulated `display`: the display itself when the application
/// detects the resolution, 2160p otherwise.
pub(crate) fn reported_display(display: DisplayResolution, attribute3: u32) -> DisplayResolution {
    if attribute3 & PARAM_ATTRIBUTE3_RESOLUTION_DETECTION != 0 {
        display
    } else {
        DisplayResolution::P2160
    }
}

/// Resolution class of the output status: 2 for a 2160p display, 1 for a
/// smaller one (720p, 1080p, 1440p), only told to an application that
/// detects the resolution ([`reported_display`]).
pub(crate) fn output_resolution_class(display: DisplayResolution, attribute3: u32) -> u32 {
    match reported_display(display, attribute3) {
        DisplayResolution::P2160 => 2,
        DisplayResolution::P720 | DisplayResolution::P1080 => 1,
    }
}

/// [`reported_display`] for the running application and the emulated
/// display ([`ps5gpu::display::display_resolution`]).
fn app_display() -> DisplayResolution {
    reported_display(ps5gpu::display::display_resolution(), app_attribute3())
}

/// param.json `attribute` bit 29: the application supports HDR output.
const PARAM_ATTRIBUTE_HDR: u32 = 0x2000_0000;

/// Whether the application may be told the display is HDR: its param.json
/// `attribute` has the HDR bit (an `attribute` missing from param.json
/// counts as 0); without a param.json at all, yes.
pub(crate) fn hdr_allowed(param_json: Option<&str>) -> bool {
    let Some(json) = param_json else { return true };
    param_u32(json, "attribute") & PARAM_ATTRIBUTE_HDR != 0
}

/// [`hdr_allowed`] for the running application (param.json read once).
fn app_hdr_allowed() -> bool {
    static ALLOWED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ALLOWED.get_or_init(|| hdr_allowed(crate::app_info::read_param_json().as_deref()))
}

/// How long the output status waits for the window to report its surface.
const HDR_PROBE_WAIT: std::time::Duration = std::time::Duration::from_secs(3);

/// Whether the display shows HDR10: HDR output is enabled
/// ([`ps5gpu::display::hdr_enabled`]) and the window's surface offers an
/// HDR10 format. The window is opened for the question when it is not
/// open yet (the display's dynamic range is known once its swapchain
/// exists); nothing is started with HDR off.
fn hdr_output_supported(width: u32, height: u32) -> bool {
    use ps5gpu::display::{hdr_enabled, hdr_surface, HdrSurface};
    if !hdr_enabled() {
        return false;
    }
    ps5gpu::presenter().ensure_started(width, height);
    let deadline = std::time::Instant::now() + HDR_PROBE_WAIT;
    crate::signals::sleep_until_unless(deadline, || hdr_surface() != HdrSurface::Unknown);
    hdr_surface() == HdrSurface::Hdr10
}

// ---------------------------------------------------------------- ports --

/// sceVideoOutOpen(userId, busType, index, param)
fn cb_open(ctx: &mut CpuContext) -> HleResult {
    let user_id = ctx.rdi() as i32;
    let bus_type = ctx.rsi() as i32;
    let index = ctx.rdx() as i32;
    if !matches!(bus_type, BUS_TYPE_MAIN | BUS_TYPE_OVERLAY | BUS_TYPE_SUB) || index != 0 {
        return ERROR_INVALID_VALUE;
    }
    if user_id != 0 && user_id != 255 {
        return ERROR_INVALID_VALUE;
    }
    let mut g = lock();
    if g.ports.len() >= MAX_OPEN_PORTS {
        return ERROR_RESOURCE_BUSY;
    }
    let handle = g.next_handle;
    g.next_handle += 1;
    g.ports.insert(handle, Port::new(handle, bus_type));
    log::info!("videoout: open bus {bus_type} -> handle {handle}");
    handle as i64
}

/// Answer for an output `mode` on a display refreshing at `refresh_rate`
/// Hz: 1 when supported, 0 for 119.88 Hz on an output without 120 Hz,
/// else the console's error. The other modes the console defines but the
/// display lacks are UNSUPPORTED; other values are not modes at all. 0x12
/// answered UNSUPPORTED on one console and UNKNOWN on another; the first is
/// used.
pub(crate) fn output_mode_answer(mode: u64, refresh_rate: u32) -> HleResult {
    match mode {
        OUTPUT_MODE_DEFAULT => 1,
        OUTPUT_MODE_119_88HZ => (refresh_rate >= 119) as HleResult,
        0x4 | 0x7 | 0x8 | 0xC | 0xD | 0xE | 0x10 | 0x11 | 0x12 | 0x13 => ERROR_UNSUPPORTED_OUTPUT_MODE,
        _ => ERROR_UNKNOWN_OUTPUT_MODE,
    }
}

/// Output options must keep the words sceVideoOutInitializeOutputOptions
/// wrote, except the free word.
pub(crate) fn output_options_valid(bytes: &[u8]) -> bool {
    bytes.chunks_exact(4).enumerate().all(|(i, w)| {
        let v = u32::from_le_bytes(w.try_into().unwrap());
        i == OUTPUT_OPTIONS_FREE_WORD || v == if i == 0 { OUTPUT_OPTIONS_FIRST_WORD } else { 0 }
    })
}

/// sceVideoOutIsOutputSupported(handle, mode, options*, reserved*, reserved).
/// Checks the handle, the reserved arguments, the mode, then the options.
fn cb_is_output_supported(ctx: &mut CpuContext) -> HleResult {
    let handle = handle_arg(ctx.rdi());
    let mode = ctx.rsi();
    let options = ctx.rdx();
    let Some(rate) = lock().ports.get(&handle).map(|p| p.refresh_rate) else {
        return ERROR_INVALID_HANDLE;
    };
    if ctx.rcx() != 0 || ctx.r8() != 0 {
        return ERROR_INVALID_VALUE;
    }
    let answer = output_mode_answer(mode, rate);
    if answer < 0 {
        return answer;
    }
    if options != 0 {
        let Some(b) = ctx.read_bytes(options, OUTPUT_OPTIONS_SIZE) else {
            return ERROR_MEMORY_FAULT;
        };
        if !output_options_valid(&b) {
            return ERROR_INVALID_VALUE;
        }
    }
    answer
}

/// sceVideoOutClose(handle)
fn cb_close(ctx: &mut CpuContext) -> HleResult {
    let handle = handle_arg(ctx.rdi());
    drain_gpu();
    let removed = {
        let mut g = lock();
        let ids: Vec<u64> = g.flips.iter().filter(|(_, r)| r.handle == handle).map(|(&id, _)| id).collect();
        for id in ids {
            cancel_flip_locked(&mut g, id);
        }
        let removed = g.ports.remove(&handle);
        if let Some(master) = removed.as_ref().and_then(|p| p.master).and_then(|m| g.ports.get_mut(&m)) {
            master.slave_count -= 1;
        }
        CV.notify_all();
        removed
    };
    // The port's events leave their queues, pending occurrences included.
    if let Some(p) = removed {
        for (list, ident) in [
            (&p.flip_events, EVENT_FLIP),
            (&p.vblank_events, EVENT_VBLANK),
            (&p.pre_vblank_events, EVENT_PRE_VBLANK_START),
            (&p.output_mode_events, EVENT_OUTPUT_MODE),
            (&p.vrr_status_events, EVENT_VRR_STATUS),
        ] {
            for r in list {
                equeue::delete_event(r.equeue, ident, FILTER_VIDEO_OUT);
            }
        }
    }
    OK
}

/// sceVideoOutSetFlipMaster(slaveHandle, masterHandle): the slave port's
/// flips are presented together with the master's.
fn cb_set_flip_master(ctx: &mut CpuContext) -> HleResult {
    let (slave, master) = (handle_arg(ctx.rdi()), handle_arg(ctx.rsi()));
    let mut g = lock();
    let (Some(s), Some(m)) = (g.ports.get(&slave), g.ports.get(&master)) else {
        return ERROR_INVALID_HANDLE;
    };
    if slave == master {
        return ERROR_UNSUPPORTED_OPERATION;
    }
    if s.master.is_some() || s.slave_count != 0 || m.master.is_some() {
        return ERROR_UNSUPPORTED_OPERATION;
    }
    if s.flip_pending_count != 0 || m.flip_pending_count != 0 {
        return ERROR_RESOURCE_BUSY;
    }
    if let Some(s) = g.ports.get_mut(&slave) {
        s.master = Some(master);
    }
    if let Some(m) = g.ports.get_mut(&master) {
        m.slave_count += 1;
    }
    OK
}

/// sceVideoOutAllowOutputResolutionWqhdDetection(handle)
fn cb_allow_output_resolution_wqhd_detection(ctx: &mut CpuContext) -> HleResult {
    if lock().ports.contains_key(&handle_arg(ctx.rdi())) { OK } else { ERROR_INVALID_HANDLE }
}

/// sceVideoOutSetFlipRate(handle, rate) — 0: 60 Hz, 1: 30 Hz, 2: 20 Hz.
fn cb_set_flip_rate(ctx: &mut CpuContext) -> HleResult {
    let handle = handle_arg(ctx.rdi());
    let rate = ctx.rsi() as i32;
    if !(0..=2).contains(&rate) {
        return ERROR_INVALID_VALUE;
    }
    match lock().ports.get_mut(&handle) {
        Some(p) => {
            p.flip_rate = rate;
            OK
        }
        None => ERROR_INVALID_HANDLE,
    }
}

/// sceVideoOutConfigureOutput(handle, mode, options*, reserved...)
fn cb_configure_output(ctx: &mut CpuContext) -> HleResult {
    let supported = cb_is_output_supported(ctx);
    if supported < 0 {
        return supported;
    }
    // A mode the output lacks. A title that does not declare high frame
    // rate output gets UNSUPPORTED here (no title attribute is read).
    if supported == 0 {
        return ERROR_UNSUPPORTED_OUTPUT_MODE;
    }
    let handle = handle_arg(ctx.rdi());
    let mode = ctx.rsi();
    let regs = {
        let mut g = lock();
        let Some(p) = g.ports.get_mut(&handle) else { return ERROR_INVALID_HANDLE };
        p.output_mode = mode;
        p.output_mode_events.clone()
    };
    let hint = (mode & 0x0000_FFFF_FFFF_FFFF) << 16;
    for r in regs {
        equeue::trigger_display_event(r.equeue, EVENT_OUTPUT_MODE, FILTER_VIDEO_OUT, hint, r.udata);
    }
    OK
}

/// The 64 bytes sceVideoOutInitializeOutputOptions writes.
pub(crate) fn initial_output_options() -> [u8; OUTPUT_OPTIONS_SIZE] {
    let mut o = [0u8; OUTPUT_OPTIONS_SIZE];
    put_u32(&mut o, 0, OUTPUT_OPTIONS_FIRST_WORD);
    o
}

/// sceVideoOutInitializeOutputOptions(options*)
fn cb_initialize_output_options(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    write_or_fault(ctx, addr, &initial_output_options())
}

/// sceVideoOutGetOutputStatus(handle, status*)
fn cb_get_output_status(ctx: &mut CpuContext) -> HleResult {
    let handle = handle_arg(ctx.rdi());
    let addr = ctx.rsi();
    if addr == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let Some((w, h, r)) = lock().ports.get(&handle).map(|p| (p.output_width, p.output_height, p.refresh_rate)) else {
        return ERROR_INVALID_HANDLE;
    };
    let hdr = ps5gpu::display::hdr_enabled() && app_hdr_allowed() && hdr_output_supported(w, h);
    let class = output_resolution_class(ps5gpu::display::display_resolution(), app_attribute3());
    write_or_fault(ctx, addr, &output_status_bytes(class, r, hdr))
}

/// sceVideoOutGetOutputStatusWithVrr(handle, status*): the 0x30-byte
/// output status followed by VRR fields, written as zero (the output has
/// no VRR state), 0x50 bytes in all.
fn cb_get_output_status_with_vrr(ctx: &mut CpuContext) -> HleResult {
    let handle = handle_arg(ctx.rdi());
    let addr = ctx.rsi();
    if addr == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let Some((w, h, r)) = lock().ports.get(&handle).map(|p| (p.output_width, p.output_height, p.refresh_rate)) else {
        return ERROR_INVALID_HANDLE;
    };
    let hdr = ps5gpu::display::hdr_enabled() && app_hdr_allowed() && hdr_output_supported(w, h);
    let mut status = [0u8; OUTPUT_STATUS_WITH_VRR_SIZE];
    let class = output_resolution_class(ps5gpu::display::display_resolution(), app_attribute3());
    status[..OUTPUT_STATUS_SIZE].copy_from_slice(&output_status_bytes(class, r, hdr));
    write_or_fault(ctx, addr, &status)
}

/// Build the 0x30-byte SceVideoOutResolutionStatus: full and pane size
/// (u32 x4), refresh-rate code (u64 at 0x10), screen size in inches (f32
/// at 0x18, 0 = unknown as in EDID), flags (u16 at 0x1C, 0) and reserved
/// words. The pane is the full output: no window margins are applied.
pub(crate) fn resolution_status_bytes(width: u32, height: u32, refresh_rate: u32) -> [u8; RESOLUTION_STATUS_SIZE] {
    let mut s = [0u8; RESOLUTION_STATUS_SIZE];
    put_u32(&mut s, 0x00, width);
    put_u32(&mut s, 0x04, height);
    put_u32(&mut s, 0x08, width);
    put_u32(&mut s, 0x0C, height);
    // The same refresh-rate code as the output status.
    s[0x10..0x18].copy_from_slice(&output_status_bytes(1, refresh_rate, false)[0x08..0x10]);
    s
}

/// sceVideoOutGetResolutionStatus(handle, status*): the display mode the
/// application may see ([`reported_display`]): the emulated display's size
/// for an application that detects the resolution, 2160p otherwise. This
/// is a model, not a measured answer: the size the console reports here
/// has not been checked on hardware; it follows the output status's
/// resolution class.
fn cb_get_resolution_status(ctx: &mut CpuContext) -> HleResult {
    let handle = handle_arg(ctx.rdi());
    let addr = ctx.rsi();
    if addr == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let Some(r) = lock().ports.get(&handle).map(|p| p.refresh_rate) else {
        return ERROR_INVALID_HANDLE;
    };
    let (w, h) = app_display().size();
    write_or_fault(ctx, addr, &resolution_status_bytes(w, h, r))
}

/// sceVideoOutColorSettingsSetGamma(settings*, float gamma): gamma in
/// 0.1..=2.0.
fn cb_color_settings_set_gamma(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    // The gamma argument is a float passed in xmm0.
    let gamma = ps5cpu::exec::trap_xmm_low(0).map_or(1.0f32, |bits| f32::from_bits(bits as u32));
    if !(0.1..=2.0).contains(&gamma) {
        return ERROR_INVALID_VALUE;
    }
    write_or_fault(ctx, addr, &gamma.to_le_bytes())
}

/// sceVideoOutColorSettingsSetGamma_(settings*, float gamma, settingsSize)
fn cb_color_settings_set_gamma_sized(ctx: &mut CpuContext) -> HleResult {
    if (ctx.rsi() as u32 as u64) < COLOR_SETTINGS_SIZE {
        return ERROR_INVALID_VALUE;
    }
    cb_color_settings_set_gamma(ctx)
}

/// sceVideoOutAdjustColor_(handle, settings*, settingsSize)
fn cb_adjust_color_sized(ctx: &mut CpuContext) -> HleResult {
    if (ctx.rdx() as u32 as u64) < COLOR_SETTINGS_SIZE {
        return ERROR_INVALID_VALUE;
    }
    cb_adjust_color(ctx)
}

/// sceVideoOutAdjustColor(handle, settings*): the port keeps the gamma.
fn cb_adjust_color(ctx: &mut CpuContext) -> HleResult {
    let handle = handle_arg(ctx.rdi());
    let addr = ctx.rsi();
    if addr == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    if !lock().ports.contains_key(&handle) {
        return ERROR_INVALID_HANDLE;
    }
    let Some(gamma) = ctx.read_f32(addr) else { return ERROR_MEMORY_FAULT };
    if let Some(p) = lock().ports.get_mut(&handle) {
        p.gamma = gamma;
    }
    OK
}

/// sceVideoOutColorSpaceConversionSettingsSetDefaultSdrToHdrConversion(settings*):
/// the default settings are all zero.
fn cb_csc_set_default_sdr_to_hdr(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    write_or_fault(ctx, addr, &[0u8; COLOR_SPACE_CONVERSION_SIZE])
}

/// The `_` form: `(settings*, settingsSize)`.
fn cb_csc_set_default_sdr_to_hdr_sized(ctx: &mut CpuContext) -> HleResult {
    if (ctx.rsi() as u32 as usize) < COLOR_SPACE_CONVERSION_SIZE {
        return ERROR_INVALID_VALUE;
    }
    cb_csc_set_default_sdr_to_hdr(ctx)
}

/// sceVideoOutColorSpaceConversionSettingsSetSdrToHdrConversion(settings*,
/// float lumaScale, option): the encoding of the luma scale and option in
/// the settings is not known; they are checked and left as they are.
fn cb_csc_set_sdr_to_hdr(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    OK
}

/// The `_` form: `(settings*, float lumaScale, option, settingsSize)`.
fn cb_csc_set_sdr_to_hdr_sized(ctx: &mut CpuContext) -> HleResult {
    if (ctx.rdx() as u32 as usize) < COLOR_SPACE_CONVERSION_SIZE {
        return ERROR_INVALID_VALUE;
    }
    cb_csc_set_sdr_to_hdr(ctx)
}

/// sceVideoOutAdjustColorSpaceConversion(handle, settings*): the port
/// keeps the settings (the presenter does not apply them).
fn cb_adjust_color_space_conversion(ctx: &mut CpuContext) -> HleResult {
    let (handle, addr) = (handle_arg(ctx.rdi()), ctx.rsi());
    if addr == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    if !lock().ports.contains_key(&handle) {
        return ERROR_INVALID_HANDLE;
    }
    let Some(bytes) = ctx.read_bytes(addr, COLOR_SPACE_CONVERSION_SIZE) else { return ERROR_MEMORY_FAULT };
    match lock().ports.get_mut(&handle) {
        Some(p) => {
            p.color_space_conversion.copy_from_slice(&bytes);
            OK
        }
        None => ERROR_INVALID_HANDLE,
    }
}

/// The `_` form: `(handle, settings*, settingsSize)`.
fn cb_adjust_color_space_conversion_sized(ctx: &mut CpuContext) -> HleResult {
    if (ctx.rdx() as u32 as usize) < COLOR_SPACE_CONVERSION_SIZE {
        return ERROR_INVALID_VALUE;
    }
    cb_adjust_color_space_conversion(ctx)
}

/// sceVideoOutSetGlobalBlendSpace(handle, space, option): kept by the port
/// (the presenter blends nothing).
fn cb_set_global_blend_space(ctx: &mut CpuContext) -> HleResult {
    match lock().ports.get_mut(&handle_arg(ctx.rdi())) {
        Some(p) => {
            p.blend_space = (ctx.rsi() as u32, ctx.rdx() as u32);
            OK
        }
        None => ERROR_INVALID_HANDLE,
    }
}

// -------------------------------------------------------------- latency --

fn is_latency_target_reached(p: &Port, control: u32, target: i64) -> bool {
    if control == LATENCY_WAIT_BY_FLIP_QUEUE_NUM {
        return p.flip_pending_count as i64 <= target;
    }
    let Some(&completed) = p.completed_latency.get(&target) else { return false };
    p.latency_start_points.get(&target).is_none_or(|&started| completed >= started)
}

/// sceVideoOutLatencyControlWaitBeforeInput(handle, control*, timeoutUsec*)
fn cb_latency_control_wait_before_input(ctx: &mut CpuContext) -> HleResult {
    let handle = handle_arg(ctx.rdi());
    let control_addr = ctx.rsi();
    let timeout_addr = ctx.rdx();
    if !lock().ports.contains_key(&handle) {
        return ERROR_INVALID_HANDLE;
    }
    if control_addr == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let Some(b) = ctx.read_bytes(control_addr, LATENCY_CONTROL_SIZE) else { return ERROR_MEMORY_FAULT };
    let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let control = u32_at(0);
    let pad1 = u32_at(4);
    let target = i64::from_le_bytes(b[8..16].try_into().unwrap());
    let extra_usec = u32_at(0x10);
    let pad2 = u32_at(0x14);
    let reserved = u64::from_le_bytes(b[0x18..0x20].try_into().unwrap());
    if control > LATENCY_WAIT_BY_FLIP_ARG
        || target < 0
        || extra_usec > MAX_LATENCY_EXTRA_USEC
        || pad1 != 0
        || pad2 != 0
        || reserved != 0
    {
        return ERROR_INVALID_VALUE;
    }
    let mut timeout_usec = 0u32;
    if timeout_addr != 0 {
        match ctx.read_u32(timeout_addr) {
            Some(v) => timeout_usec = v,
            None => return ERROR_MEMORY_FAULT,
        }
    }
    let deadline = (timeout_addr != 0).then(|| Instant::now() + Duration::from_micros(timeout_usec as u64));
    let remaining = || deadline.map(|d| d.saturating_duration_since(Instant::now()));
    {
        let mut g = lock();
        loop {
            let Some(p) = g.ports.get(&handle) else { return ERROR_INVALID_HANDLE };
            if is_latency_target_reached(p, control, target) {
                break;
            }
            match remaining() {
                Some(r) if r.is_zero() => {
                    drop(g);
                    ctx.write_u32(timeout_addr, 0);
                    return ERROR_TIMED_OUT;
                }
                Some(r) => g = crate::signals::wait_cv(&STATE, &CV, g, Some(Instant::now() + r.max(Duration::from_millis(1)))).0,
                None => g = crate::signals::wait_cv(&STATE, &CV, g, None).0,
            }
        }
    }
    if extra_usec != 0 {
        let extra = Duration::from_micros(extra_usec as u64);
        if let Some(r) = remaining() {
            if r < extra {
                crate::signals::sleep(r);
                ctx.write_u32(timeout_addr, 0);
                return ERROR_TIMED_OUT;
            }
        }
        crate::signals::sleep(extra);
    }
    if timeout_addr != 0 {
        let left = remaining().unwrap_or_default().as_micros().min(u32::MAX as u128) as u32;
        ctx.write_u32(timeout_addr, left);
    }
    OK
}

/// sceVideoOutLatencyMeasureSetStartPoint(handle, flipArg)
fn cb_latency_measure_set_start_point(ctx: &mut CpuContext) -> HleResult {
    let handle = handle_arg(ctx.rdi());
    let flip_arg = ctx.rsi() as i64;
    let mut g = lock();
    let Some(p) = g.ports.get_mut(&handle) else { return ERROR_INVALID_HANDLE };
    let ts = now_ts();
    p.latency_start_points.insert(flip_arg, ts);
    p.latency_start_order.push_back((flip_arg, ts));
    prune_history(&mut p.latency_start_points, &mut p.latency_start_order);
    OK
}

// --------------------------------------------------------------- vblank --

/// sceVideoOutWaitVblank(handle)
fn cb_wait_vblank(ctx: &mut CpuContext) -> HleResult {
    let handle = handle_arg(ctx.rdi());
    let target = {
        let mut g = lock();
        let Some(p) = g.ports.get_mut(&handle) else { return ERROR_INVALID_HANDLE };
        p.clock.advance(now_ts(), p.refresh_rate);
        p.clock.next_timestamp(p.refresh_rate)
    };
    sleep_until(target);
    let mut g = lock();
    let Some(p) = g.ports.get_mut(&handle) else { return ERROR_INVALID_HANDLE };
    p.clock.advance(now_ts(), p.refresh_rate);
    OK
}

/// sceVideoOutGetVblankStatus(handle, status*)
fn cb_get_vblank_status(ctx: &mut CpuContext) -> HleResult {
    let handle = handle_arg(ctx.rdi());
    let addr = ctx.rsi();
    if addr == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let mut s = [0u8; VBLANK_STATUS_SIZE];
    {
        let mut g = lock();
        let Some(p) = g.ports.get_mut(&handle) else { return ERROR_INVALID_HANDLE };
        p.clock.advance(now_ts(), p.refresh_rate);
        put_u64(&mut s, 0x00, p.clock.count);
        put_u64(&mut s, 0x08, p.clock.process_microseconds());
        put_u64(&mut s, 0x10, p.clock.timestamp_counter);
        put_u64(&mut s, 0x18, p.clock.process_counter);
    }
    write_or_fault(ctx, addr, &s)
}

fn upsert(list: &mut Vec<EventReg>, equeue: u64, udata: u64) {
    match list.iter_mut().find(|r| r.equeue == equeue) {
        Some(r) => r.udata = udata,
        None => list.push(EventReg { equeue, udata }),
    }
}

/// sceVideoOutAddFlipEvent(equeue, handle, udata)
fn cb_add_flip_event(ctx: &mut CpuContext) -> HleResult {
    let (eq, handle, udata) = (ctx.rdi(), handle_arg(ctx.rsi()), ctx.rdx());
    if !lock().ports.contains_key(&handle) {
        return ERROR_INVALID_HANDLE;
    }
    if !equeue::is_valid_equeue(eq) {
        return ERROR_INVALID_EVENT_QUEUE;
    }
    if let Some(p) = lock().ports.get_mut(&handle) {
        upsert(&mut p.flip_events, eq, udata);
    }
    OK
}

/// sceVideoOutAddVblankEvent(equeue, handle, udata)
fn cb_add_vblank_event(ctx: &mut CpuContext) -> HleResult {
    let (eq, handle, udata) = (ctx.rdi(), handle_arg(ctx.rsi()), ctx.rdx());
    if !lock().ports.contains_key(&handle) {
        return ERROR_INVALID_HANDLE;
    }
    if !equeue::is_valid_equeue(eq) {
        return ERROR_INVALID_EVENT_QUEUE;
    }
    {
        let mut g = lock();
        let Some(p) = g.ports.get_mut(&handle) else { return ERROR_INVALID_HANDLE };
        if p.vblank_events.is_empty() && p.pre_vblank_events.is_empty() {
            p.clock.advance(now_ts(), p.refresh_rate);
            p.published_vblank_count = p.clock.count;
        }
        upsert(&mut p.vblank_events, eq, udata);
        CV.notify_all();
    }
    start_vblank_thread_once();
    OK
}

/// sceVideoOutAddPreVblankStartEvent(equeue, handle, udata)
fn cb_add_pre_vblank_start_event(ctx: &mut CpuContext) -> HleResult {
    let (eq, handle, udata) = (ctx.rdi(), handle_arg(ctx.rsi()), ctx.rdx());
    if !lock().ports.contains_key(&handle) {
        return ERROR_INVALID_HANDLE;
    }
    if !equeue::is_valid_equeue(eq) {
        return ERROR_INVALID_EVENT_QUEUE;
    }
    {
        let mut g = lock();
        let Some(p) = g.ports.get_mut(&handle) else { return ERROR_INVALID_HANDLE };
        if p.vblank_events.is_empty() && p.pre_vblank_events.is_empty() {
            p.clock.advance(now_ts(), p.refresh_rate);
            p.published_vblank_count = p.clock.count;
        }
        upsert(&mut p.pre_vblank_events, eq, udata);
        CV.notify_all();
    }
    start_vblank_thread_once();
    OK
}

/// Remove `eq`'s registration from the list `pick` selects, then the event
/// (and any occurrence still pending) from the queue.
fn delete_event_registration(eq: u64, handle: i32, ident: u64, pick: fn(&mut Port) -> &mut Vec<EventReg>) -> HleResult {
    {
        let mut g = lock();
        let Some(p) = g.ports.get_mut(&handle) else { return ERROR_INVALID_HANDLE };
        pick(p).retain(|r| r.equeue != eq);
    }
    // A queue that never received the event has nothing to remove.
    equeue::delete_event(eq, ident, FILTER_VIDEO_OUT);
    OK
}

/// sceVideoOutDeletePreVblankStartEvent(equeue, handle)
fn cb_delete_pre_vblank_start_event(ctx: &mut CpuContext) -> HleResult {
    delete_event_registration(ctx.rdi(), handle_arg(ctx.rsi()), EVENT_PRE_VBLANK_START, |p| &mut p.pre_vblank_events)
}

/// sceVideoOutAddOutputModeEvent(equeue, handle, udata) — fires at once.
fn cb_add_output_mode_event(ctx: &mut CpuContext) -> HleResult {
    let (eq, handle, udata) = (ctx.rdi(), handle_arg(ctx.rsi()), ctx.rdx());
    if !lock().ports.contains_key(&handle) {
        return ERROR_INVALID_HANDLE;
    }
    if !equeue::is_valid_equeue(eq) {
        return ERROR_INVALID_EVENT_QUEUE;
    }
    let mode = {
        let mut g = lock();
        let Some(p) = g.ports.get_mut(&handle) else { return ERROR_INVALID_HANDLE };
        upsert(&mut p.output_mode_events, eq, udata);
        p.output_mode
    };
    let hint = (mode & 0x0000_FFFF_FFFF_FFFF) << 16;
    if equeue::trigger_display_event(eq, EVENT_OUTPUT_MODE, FILTER_VIDEO_OUT, hint, udata) {
        OK
    } else {
        ERROR_INVALID_EVENT_QUEUE
    }
}

/// sceVideoOutDeleteOutputModeEvent(equeue, handle): unsubscribe the queue
/// from the output mode events; a queue that never subscribed succeeds.
fn cb_delete_output_mode_event(ctx: &mut CpuContext) -> HleResult {
    let (eq, handle) = (ctx.rdi(), handle_arg(ctx.rsi()));
    if !lock().ports.contains_key(&handle) {
        return ERROR_INVALID_HANDLE;
    }
    if eq == 0 {
        return ERROR_INVALID_EVENT_QUEUE;
    }
    delete_event_registration(eq, handle, EVENT_OUTPUT_MODE, |p| &mut p.output_mode_events)
}

/// sceVideoOutDeleteVblankEvent(equeue, handle)
fn cb_delete_vblank_event(ctx: &mut CpuContext) -> HleResult {
    delete_event_registration(ctx.rdi(), handle_arg(ctx.rsi()), EVENT_VBLANK, |p| &mut p.vblank_events)
}

/// sceVideoOutDeleteFlipEvent(equeue, handle)
fn cb_delete_flip_event(ctx: &mut CpuContext) -> HleResult {
    delete_event_registration(ctx.rdi(), handle_arg(ctx.rsi()), EVENT_FLIP, |p| &mut p.flip_events)
}

/// sceVideoOutAddVrrActiveStatusEvent(equeue, handle, udata): event id 16.
/// Recorded but never triggered: the host output's VRR state never changes.
fn cb_add_vrr_active_status_event(ctx: &mut CpuContext) -> HleResult {
    let (eq, handle, udata) = (ctx.rdi(), handle_arg(ctx.rsi()), ctx.rdx());
    if !lock().ports.contains_key(&handle) {
        return ERROR_INVALID_HANDLE;
    }
    if !equeue::is_valid_equeue(eq) {
        return ERROR_INVALID_EVENT_QUEUE;
    }
    let mut g = lock();
    let Some(p) = g.ports.get_mut(&handle) else { return ERROR_INVALID_HANDLE };
    upsert(&mut p.vrr_status_events, eq, udata);
    OK
}

/// sceVideoOutDeleteVrrActiveStatusEvent(equeue, handle)
fn cb_delete_vrr_active_status_event(ctx: &mut CpuContext) -> HleResult {
    delete_event_registration(ctx.rdi(), handle_arg(ctx.rsi()), EVENT_VRR_STATUS, |p| &mut p.vrr_status_events)
}

/// Variable-refresh-rate privileges: accepted, no effect (the host output
/// has no VRR state).
fn cb_vrr_noop(_ctx: &mut CpuContext) -> HleResult {
    OK
}

/// sceVideoOutVrrUnpegFromFixedRate(handle): the output always runs at a
/// fixed rate (VRR is not modelled), so only the handle is checked.
fn cb_vrr_unpeg(ctx: &mut CpuContext) -> HleResult {
    if lock().ports.contains_key(&handle_arg(ctx.rdi())) { OK } else { ERROR_INVALID_HANDLE }
}

/// sceVideoOutVrrPegToFixedRate(handle, u64, u64): as the unpeg; the two
/// further arguments (zero in the callers seen) have no known meaning and
/// are only logged when they are not zero.
fn cb_vrr_peg(ctx: &mut CpuContext) -> HleResult {
    let (handle, a, b) = (handle_arg(ctx.rdi()), ctx.rsi(), ctx.rdx());
    if !lock().ports.contains_key(&handle) {
        return ERROR_INVALID_HANDLE;
    }
    if a != 0 || b != 0 {
        log::warn!("[videoout] sceVideoOutVrrPegToFixedRate({handle}, {a:#x}, {b:#x}): arguments of unknown meaning");
    }
    OK
}

// ---------------------------------------------------------------- flips --

/// sceVideoOutSubmitFlip(handle, bufferIndex, flipMode, flipArg)
fn cb_submit_flip(ctx: &mut CpuContext) -> HleResult {
    crate::pause::wait_if_paused();
    match cpu_submit_flip(handle_arg(ctx.rdi()), ctx.rsi() as i32, ctx.rdx() as i32, ctx.rcx() as i64) {
        Ok(()) => OK,
        Err(e) => e,
    }
}

/// sceVideoOutSubmitEopFlip(handle, bufferIndex, flipMode, flipArg, u32*
/// token): reserve a flip performed by the graphics-queue end-of-pipe
/// RELEASE_MEM (interrupt selector 2) whose context id is the token.
fn cb_submit_eop_flip(ctx: &mut CpuContext) -> HleResult {
    let token = ctx.r8();
    if token == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    match reserve_eop_flip(handle_arg(ctx.rdi()), ctx.rsi() as i32, ctx.rdx() as i32, ctx.rcx() as i64) {
        // An unwritable token leaves an unclaimable reservation, which
        // closing the port cancels.
        Ok(t) => write_or_fault(ctx, token, &t.to_le_bytes()),
        Err(e) => e,
    }
}

/// sceVideoOutGetBufferLabelAddress(handle, u64* address) -> label count:
/// the port's array of one u64 label per display buffer.
fn cb_get_buffer_label_address(ctx: &mut CpuContext) -> HleResult {
    let (handle, out) = (handle_arg(ctx.rdi()), ctx.rsi());
    if out == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let labels = match lock().ports.get(&handle) {
        Some(p) => p.labels,
        None => return ERROR_INVALID_HANDLE,
    };
    let labels = if labels != 0 {
        labels
    } else {
        let Some(a) = crate::pthread::alloc_guest_block(ctx, MAX_DISPLAY_BUFFERS as u64 * 8) else { return ERROR_MEMORY_FAULT };
        let mut g = lock();
        let Some(p) = g.ports.get_mut(&handle) else { return ERROR_INVALID_HANDLE };
        // A concurrent first query may have set it already (its block
        // stays unused).
        if p.labels == 0 {
            p.labels = a;
        }
        p.labels
    };
    match write_or_fault(ctx, out, &labels.to_le_bytes()) {
        OK => MAX_DISPLAY_BUFFERS as HleResult,
        e => e,
    }
}

/// Build the 0x80-byte SceVideoOutFlipStatus of a port.
fn flip_status_bytes(p: &Port) -> [u8; FLIP_STATUS_SIZE] {
    let mut s = [0u8; FLIP_STATUS_SIZE];
    put_u64(&mut s, 0x00, p.flip_count);
    put_u64(&mut s, 0x08, p.flip_process_time);
    put_u64(&mut s, 0x18, p.flip_arg as u64);
    put_u64(&mut s, 0x28, p.flip_process_time_counter);
    put_u32(&mut s, 0x30, p.gpu_queue_count as u32);
    put_u32(&mut s, 0x34, p.flip_pending_count as u32);
    put_u32(&mut s, 0x38, p.current_buffer as u32);
    put_u64(&mut s, 0x40, p.submit_process_time_counter);
    s
}

/// sceVideoOutGetFlipStatus(handle, status*)
fn cb_get_flip_status(ctx: &mut CpuContext) -> HleResult {
    let handle = handle_arg(ctx.rdi());
    let addr = ctx.rsi();
    if addr == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let Some(s) = lock().ports.get(&handle).map(flip_status_bytes) else {
        return ERROR_INVALID_HANDLE;
    };
    write_or_fault(ctx, addr, &s)
}

/// sceVideoOutIsFlipPending(handle) -> pending flip count
fn cb_is_flip_pending(ctx: &mut CpuContext) -> HleResult {
    match lock().ports.get(&handle_arg(ctx.rdi())) {
        Some(p) => p.flip_pending_count as i64,
        None => ERROR_INVALID_HANDLE,
    }
}

/// sceVideoOutGetEventCount(event*) — occurrences folded into the event.
fn cb_get_event_count(ctx: &mut CpuContext) -> HleResult {
    let ev = ctx.rdi();
    if ev == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let (Some(filter), Some(data)) = (read_i16(ctx, ev + 8), ctx.read_u64(ev + 0x10)) else {
        return ERROR_MEMORY_FAULT;
    };
    if filter != FILTER_VIDEO_OUT {
        return ERROR_INVALID_EVENT;
    }
    ((data >> 12) & 0xF) as HleResult
}

/// sceVideoOutAddBufferAttributeOption(attr*, option): ORs `option` into
/// the attribute's option word.
fn cb_add_buffer_attribute_option(ctx: &mut CpuContext) -> HleResult {
    let (attr, option) = (ctx.rdi(), ctx.rsi());
    if attr == 0 {
        log::error!("[videoout] sceVideoOutAddBufferAttributeOption: null attribute");
        return OK;
    }
    if let Some(old) = ctx.read_u64(attr + 0x18) {
        ctx.write_u64(attr + 0x18, old | option);
    }
    OK
}

/// sceVideoOutRemoveBufferAttributeOption(attr*, option): clears the
/// `option` bits of the attribute's option word.
fn cb_remove_buffer_attribute_option(ctx: &mut CpuContext) -> HleResult {
    let (attr, option) = (ctx.rdi(), ctx.rsi());
    if attr == 0 {
        log::error!("[videoout] sceVideoOutRemoveBufferAttributeOption: null attribute");
        return OK;
    }
    if let Some(old) = ctx.read_u64(attr + 0x18) {
        ctx.write_u64(attr + 0x18, old & !option);
    }
    OK
}

/// sceVideoOutGetEventId(event*) — 0 flip, 1 vblank, 2 pre-vblank start,
/// 8 output mode, 16 VRR status.
fn cb_get_event_id(ctx: &mut CpuContext) -> HleResult {
    let ev = ctx.rdi();
    if ev == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let (Some(ident), Some(filter)) = (ctx.read_u64(ev), read_i16(ctx, ev + 8)) else {
        return ERROR_MEMORY_FAULT;
    };
    if filter != FILTER_VIDEO_OUT {
        return ERROR_INVALID_EVENT;
    }
    match ident {
        EVENT_FLIP => 0,
        EVENT_VBLANK => 1,
        EVENT_PRE_VBLANK_START => 2,
        EVENT_OUTPUT_MODE => 8,
        EVENT_VRR_STATUS => 16,
        _ => ERROR_INVALID_EVENT,
    }
}

/// sceVideoOutGetEventData(event*, data*) — data = kevent.data >> 16.
fn cb_get_event_data(ctx: &mut CpuContext) -> HleResult {
    let (ev, out) = (ctx.rdi(), ctx.rsi());
    if ev == 0 || out == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let (Some(ident), Some(filter), Some(data)) = (ctx.read_u64(ev), read_i16(ctx, ev + 8), ctx.read_u64(ev + 0x10))
    else {
        return ERROR_MEMORY_FAULT;
    };
    if filter != FILTER_VIDEO_OUT
        || !matches!(ident, EVENT_FLIP | EVENT_VBLANK | EVENT_PRE_VBLANK_START | EVENT_OUTPUT_MODE | EVENT_VRR_STATUS)
    {
        return ERROR_INVALID_EVENT;
    }
    let decoded = ((data as i64) >> 16) as u64;
    if ctx.write_u64(out, decoded) {
        OK
    } else {
        ERROR_MEMORY_FAULT
    }
}

/// sceVideoOutSetWindowModeMargins(handle, top, bottom)
fn cb_set_window_mode_margins(ctx: &mut CpuContext) -> HleResult {
    let handle = handle_arg(ctx.rdi());
    let (top, bottom) = (ctx.rsi() as i32, ctx.rdx() as i32);
    let mut g = lock();
    let Some(p) = g.ports.get_mut(&handle) else { return ERROR_INVALID_HANDLE };
    if top < 0 || bottom < top || bottom as i64 > p.output_height as i64 {
        return ERROR_INVALID_VALUE;
    }
    p.window_top = top;
    p.window_bottom = bottom;
    OK
}

// -------------------------------------------------------------- buffers --

/// sceVideoOutUnregisterBuffers(handle, attributeIndex)
fn cb_unregister_buffers(ctx: &mut CpuContext) -> HleResult {
    let handle = handle_arg(ctx.rdi());
    let index = ctx.rsi() as i32;
    if index < 0 {
        return ERROR_INVALID_VALUE;
    }
    drain_gpu();
    let mut g = lock();
    let Some(p) = g.ports.get_mut(&handle) else { return ERROR_INVALID_HANDLE };
    if index as usize >= MAX_DISPLAY_BUFFER_GROUPS || p.groups[index as usize].is_none() {
        return ERROR_INVALID_VALUE;
    }
    p.groups[index as usize] = None;
    for s in p.slots.iter_mut().filter(|s| s.group == index) {
        *s = EMPTY_SLOT;
    }
    OK
}

/// sceVideoOutSetBufferAttribute(attr*, pixelFormat, tilingMode, aspectRatio, width, height, pitchInPixel)
fn cb_set_buffer_attribute(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    let Some(pitch) = stack_u64(ctx, 0) else { return ERROR_MEMORY_FAULT };
    if addr == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let mut a = [0u8; BUFFER_ATTRIBUTE_SIZE];
    put_u32(&mut a, 0x00, ctx.rsi() as u32);
    put_u32(&mut a, 0x04, ctx.rdx() as u32);
    put_u32(&mut a, 0x08, ctx.rcx() as u32);
    put_u32(&mut a, 0x0C, ctx.r8() as u32);
    put_u32(&mut a, 0x10, ctx.r9() as u32);
    put_u32(&mut a, 0x14, pitch as u32);
    put_u32(&mut a, 0x18, 0);
    write_or_fault(ctx, addr, &a)
}

/// Encode the 0x50-byte attribute2 block.
pub(crate) fn buffer_attribute2_bytes(
    pixel_format: u64,
    tiling: u32,
    width: u32,
    height: u32,
    option: u64,
    dcc_control: u32,
    dcc_clear_color: u64,
) -> [u8; BUFFER_ATTRIBUTE2_SIZE] {
    let mut a = [0u8; BUFFER_ATTRIBUTE2_SIZE];
    put_u32(&mut a, 0x04, tiling);
    put_u32(&mut a, 0x0C, width);
    put_u32(&mut a, 0x10, height);
    put_u64(&mut a, 0x18, option);
    put_u64(&mut a, 0x20, pixel_format);
    put_u64(&mut a, 0x28, dcc_clear_color);
    put_u32(&mut a, 0x30, dcc_control);
    a
}

/// sceVideoOutSetBufferAttribute2(attr*, pixelFormat u64, tiling, width, height, option, [dccControl], [dccClearColor])
fn cb_set_buffer_attribute2(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    let (Some(dcc_control), Some(dcc_clear)) = (stack_u64(ctx, 0), stack_u64(ctx, 1)) else {
        return ERROR_MEMORY_FAULT;
    };
    if addr == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let a = buffer_attribute2_bytes(
        ctx.rsi(),
        ctx.rdx() as u32,
        ctx.rcx() as u32,
        ctx.r8() as u32,
        ctx.r9(),
        dcc_control as u32,
        dcc_clear,
    );
    write_or_fault(ctx, addr, &a)
}

/// sceVideoOutSubmitChangeBufferAttribute2(handle, setIndex, attr*, option)
fn cb_submit_change_buffer_attribute2(ctx: &mut CpuContext) -> HleResult {
    let handle = handle_arg(ctx.rdi());
    let set_index = ctx.rsi() as i32;
    let addr = ctx.rdx();
    let option = ctx.rcx();
    let mut g = lock();
    let Some(p) = g.ports.get_mut(&handle) else { return ERROR_INVALID_HANDLE };
    if addr == 0 {
        return ERROR_INVALID_OPTION;
    }
    if set_index < 0 || set_index as usize >= MAX_DISPLAY_BUFFER_GROUPS {
        return ERROR_INVALID_INDEX;
    }
    if option != 0 {
        return ERROR_INVALID_OPTION;
    }
    if p.groups[set_index as usize].is_none() {
        return ERROR_INVALID_INDEX;
    }
    let Some(a) = read_buffer_attribute(ctx, addr, true) else { return ERROR_MEMORY_FAULT };
    if a.width == 0
        || a.width > 16384
        || a.height == 0
        || a.height > 16384
        || a.tiling_mode != 0
        || a.option & !SUPPORTED_OPTIONS != 0
        || ps5gpu::format::decode(a.pixel_format).is_none()
    {
        return ERROR_INVALID_VALUE;
    }
    // Buffer addresses and membership survive an SDR/HDR format switch.
    p.groups[set_index as usize] = Some(a);
    p.output_width = a.width;
    p.output_height = a.height;
    OK
}

fn is_valid_buffer_range(start: i32, num: i32) -> bool {
    start >= 0
        && (start as usize) < MAX_DISPLAY_BUFFERS
        && num >= 1
        && num as usize <= MAX_DISPLAY_BUFFERS
        && (start + num) as usize <= MAX_DISPLAY_BUFFERS
}

/// Register `addresses` into slots starting at `start` under a group;
/// returns the group index or an error.
fn register_buffer_range(handle: i32, start: i32, addresses: &[u64], attribute: BufferAttribute, requested_group: i32) -> HleResult {
    {
        let mut g = lock();
        let Some(p) = g.ports.get_mut(&handle) else { return ERROR_INVALID_HANDLE };
        let group = if requested_group >= 0 {
            requested_group
        } else {
            p.groups.iter().position(Option::is_none).map_or(-1, |i| i as i32)
        };
        if group < 0 || group as usize >= MAX_DISPLAY_BUFFER_GROUPS {
            return ERROR_INVALID_VALUE;
        }
        if p.groups[group as usize].is_some() {
            return ERROR_RESOURCE_BUSY;
        }
        if (0..addresses.len()).any(|i| p.slots[start as usize + i].group >= 0) {
            return ERROR_RESOURCE_BUSY;
        }
        p.groups[group as usize] = Some(attribute);
        p.output_width = attribute.width;
        p.output_height = attribute.height;
        for (i, &a) in addresses.iter().enumerate() {
            p.slots[start as usize + i] = Slot { group, address_left: a, address_right: 0 };
            p.release_label(start + i as i32);
        }
        log::info!(
            "videoout: register buffers handle={handle} group={group} start={start} count={} fmt={:#x} tile={} {}x{} pitch={} addrs={:x?}",
            addresses.len(),
            attribute.pixel_format,
            attribute.tiling_mode,
            attribute.width,
            attribute.height,
            attribute.pitch_in_pixel,
            addresses
        );
        drop(g);
        ps5gpu::presenter().ensure_started(attribute.width, attribute.height);
        group as i64
    }
}

/// sceVideoOutRegisterBuffers(handle, startIndex, addresses*, bufferNum, attr*)
fn cb_register_buffers(ctx: &mut CpuContext) -> HleResult {
    let handle = handle_arg(ctx.rdi());
    let start = ctx.rsi() as i32;
    let addrs = ctx.rdx();
    let num = ctx.rcx() as i32;
    let attr = ctx.r8();
    if !lock().ports.contains_key(&handle) {
        return ERROR_INVALID_HANDLE;
    }
    if addrs == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    if attr == 0 {
        return ERROR_INVALID_OPTION;
    }
    if !is_valid_buffer_range(start, num) {
        return ERROR_INVALID_VALUE;
    }
    let Some(a) = read_buffer_attribute(ctx, attr, false) else { return ERROR_MEMORY_FAULT };
    let mut list = Vec::with_capacity(num as usize);
    for i in 0..num as u64 {
        let Some(v) = ctx.read_u64(addrs + i * 8) else { return ERROR_MEMORY_FAULT };
        list.push(v);
    }
    register_buffer_range(handle, start, &list, a, -1)
}

/// sceVideoOutRegisterBuffers2(handle, setIndex, bufferIndexStart, buffers*, bufferNum, attr*, [category], [option])
fn cb_register_buffers2(ctx: &mut CpuContext) -> HleResult {
    let handle = handle_arg(ctx.rdi());
    let set_index = ctx.rsi() as i32;
    let start = ctx.rdx() as i32;
    let buffers = ctx.rcx();
    let num = ctx.r8() as i32;
    let attr = ctx.r9();
    let (Some(category_raw), Some(option)) = (stack_u64(ctx, 0), stack_u64(ctx, 1)) else {
        return ERROR_MEMORY_FAULT;
    };
    // The category is a 32-bit enum in a 64-bit stack slot whose upper half
    // is stale. Uncompressed (0) and compressed (1) present identically;
    // anything else is treated as the standard layout.
    let category = category_raw as u32;
    if !lock().ports.contains_key(&handle) {
        return ERROR_INVALID_HANDLE;
    }
    if buffers == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    if attr == 0 {
        return ERROR_INVALID_OPTION;
    }
    if !is_valid_buffer_range(start, num) {
        return ERROR_INVALID_VALUE;
    }
    if category > 1 || option != 0 {
        log::debug!("videoout: register_buffers2 nonstandard category={category_raw:#x} option={option:#x}");
    }
    let Some(a) = read_buffer_attribute(ctx, attr, true) else { return ERROR_MEMORY_FAULT };
    let mut list = Vec::with_capacity(num as usize);
    for i in 0..num as u64 {
        let e = buffers + i * BUFFERS_ENTRY_SIZE;
        let (Some(v), Some(_)) = (ctx.read_u64(e), ctx.read_u64(e + 8)) else { return ERROR_MEMORY_FAULT };
        list.push(v);
    }
    let r = register_buffer_range(handle, start, &list, a, set_index);
    if r < 0 {
        r
    } else {
        set_index as i64
    }
}

/// (nid, name, library, callback) for every export.
pub(crate) const EXPORTS: &[(&str, &str, &str, fn(&mut CpuContext) -> HleResult)] = &[
    ("Up36PTk687E", "sceVideoOutOpen", "libSceVideoOut", cb_open),
    ("Nv8c-Kb+DUM", "sceVideoOutIsOutputSupported", "libSceVideoOut", cb_is_output_supported),
    ("uquVH4-Du78", "sceVideoOutClose", "libSceVideoOut", cb_close),
    ("CBiu4mCE1DA", "sceVideoOutSetFlipRate", "libSceVideoOut", cb_set_flip_rate),
    ("w0hLuNarQxY", "sceVideoOutConfigureOutput", "libSceVideoOut", cb_configure_output),
    ("+I4K03i3EL0", "sceVideoOutInitializeOutputOptions", "libSceVideoOut", cb_initialize_output_options),
    ("utPrVdxio-8", "sceVideoOutGetOutputStatus", "libSceVideoOut", cb_get_output_status),
    ("DYhhWbJSeRg", "sceVideoOutColorSettingsSetGamma_", "libSceVideoOut", cb_color_settings_set_gamma_sized),
    ("pv9CI5VC+R0", "sceVideoOutAdjustColor_", "libSceVideoOut", cb_adjust_color_sized),
    ("7wEmHVceFNg", "sceVideoOutColorSettingsSetGamma", "libSceVideoOut", cb_color_settings_set_gamma),
    ("ipf9hrW6kBY", "sceVideoOutAdjustColor", "libSceVideoOut", cb_adjust_color),
    ("uXRI42H7KOw", "sceVideoOutColorSpaceConversionSettingsSetDefaultSdrToHdrConversion", "libSceVideoOut", cb_csc_set_default_sdr_to_hdr),
    ("JpAYkddCFjk", "sceVideoOutColorSpaceConversionSettingsSetDefaultSdrToHdrConversion_", "libSceVideoOut", cb_csc_set_default_sdr_to_hdr_sized),
    ("5yQ7co7ZwOc", "sceVideoOutColorSpaceConversionSettingsSetSdrToHdrConversion", "libSceVideoOut", cb_csc_set_sdr_to_hdr),
    ("Yn8KT+VTZaE", "sceVideoOutColorSpaceConversionSettingsSetSdrToHdrConversion_", "libSceVideoOut", cb_csc_set_sdr_to_hdr_sized),
    ("RJPj0N1Wc+M", "sceVideoOutAdjustColorSpaceConversion", "libSceVideoOut", cb_adjust_color_space_conversion),
    ("dYDAfB+cJEU", "sceVideoOutAdjustColorSpaceConversion_", "libSceVideoOut", cb_adjust_color_space_conversion_sized),
    ("pA8xxaySl7A", "sceVideoOutSetGlobalBlendSpace", "libSceVideoOut", cb_set_global_blend_space),
    ("gWT7X8H0bYs", "sceVideoOutGetOutputStatusWithVrr", "libSceVideoOut", cb_get_output_status_with_vrr),
    ("-q-vbO859Tw", "sceVideoOutDeleteVrrActiveStatusEvent", "libSceVideoOut", cb_delete_vrr_active_status_event),
    ("j8xl+92A0q4", "sceVideoOutSubmitEopFlip", "libSceVideoOut", cb_submit_eop_flip),
    ("OcQybQejHEY", "sceVideoOutGetBufferLabelAddress", "libSceVideoOut", cb_get_buffer_label_address),
    ("eb-gvTYQcoY", "sceVideoOutLatencyControlWaitBeforeInput", "libSceVideoOut", cb_latency_control_wait_before_input),
    ("MCJ8SkzsQxY", "sceVideoOutLatencyMeasureSetStartPoint", "libSceVideoOut", cb_latency_measure_set_start_point),
    ("j6RaAUlaLv0", "sceVideoOutWaitVblank", "libSceVideoOut", cb_wait_vblank),
    ("1FZBKy8HeNU", "sceVideoOutGetVblankStatus", "libSceVideoOut", cb_get_vblank_status),
    ("HXzjK9yI30k", "sceVideoOutAddFlipEvent", "libSceVideoOut", cb_add_flip_event),
    ("kP2L8t3j-aM", "sceVideoOutAddVrrStatusFlagsPrivilege", "libSceVideoOutVrrStatus", cb_vrr_noop),
    ("LibwuIonIBw", "sceVideoOutAddVrrActiveStatusEvent", "libSceVideoOut", cb_add_vrr_active_status_event),
    ("6kPnj51T62Y", "sceVideoOutGetResolutionStatus", "libSceVideoOut", cb_get_resolution_status),
    ("5tRaBjtdTzY", "sceVideoOutVrrPegToFixedRate", "libSceVideoOut", cb_vrr_peg),
    ("T4ucGB8CsnM", "sceVideoOutVrrUnpegFromFixedRate", "libSceVideoOut", cb_vrr_unpeg),
    ("Xru92wHJRmg", "sceVideoOutAddVblankEvent", "libSceVideoOut", cb_add_vblank_event),
    ("kmSe30JTs+E", "sceVideoOutAddOutputModeEvent", "libSceVideoOut", cb_add_output_mode_event),
    ("vwtVpHQKelI", "sceVideoOutDeleteOutputModeEvent", "libSceVideoOut", cb_delete_output_mode_event),
    ("oNOQn3knW6s", "sceVideoOutDeleteVblankEvent", "libSceVideoOut", cb_delete_vblank_event),
    ("-Ozn0F1AFRg", "sceVideoOutDeleteFlipEvent", "libSceVideoOut", cb_delete_flip_event),
    ("U46NwOiJpys", "sceVideoOutSubmitFlip", "libSceVideoOut", cb_submit_flip),
    ("SbU3dwp80lQ", "sceVideoOutGetFlipStatus", "libSceVideoOut", cb_get_flip_status),
    ("zgXifHT9ErY", "sceVideoOutIsFlipPending", "libSceVideoOut", cb_is_flip_pending),
    ("U2JJtSqNKZI", "sceVideoOutGetEventId", "libSceVideoOut", cb_get_event_id),
    ("rWUTcKdkUzQ", "sceVideoOutGetEventData", "libSceVideoOut", cb_get_event_data),
    ("MTxxrOCeSig", "sceVideoOutSetWindowModeMargins", "libSceVideoOut", cb_set_window_mode_margins),
    ("N5KDtkIjjJ4", "sceVideoOutUnregisterBuffers", "libSceVideoOut", cb_unregister_buffers),
    ("i6-sR91Wt-4", "sceVideoOutSetBufferAttribute", "libSceVideoOut", cb_set_buffer_attribute),
    ("PjS5uASwcV8", "sceVideoOutSetBufferAttribute2", "libSceVideoOut", cb_set_buffer_attribute2),
    ("HuViW4HnrOw", "sceVideoOutSubmitChangeBufferAttribute2", "libSceVideoOut", cb_submit_change_buffer_attribute2),
    ("w3BY+tAEiQY", "sceVideoOutRegisterBuffers", "libSceVideoOut", cb_register_buffers),
    ("rKBUtgRrtbk", "sceVideoOutRegisterBuffers2", "libSceVideoOut", cb_register_buffers2),
    ("WkYtyOg30do", "sceVideoOutSetFlipMaster", "libSceVideoOut", cb_set_flip_master),
    ("MBlv8YWoyOk", "sceVideoOutAddBufferAttributeOption", "libSceVideoOut", cb_add_buffer_attribute_option),
    ("k4Ak+e2D1Hg", "sceVideoOutRemoveBufferAttributeOption", "libSceVideoOut", cb_remove_buffer_attribute_option),
    ("keipklF0pMY", "sceVideoOutAddPreVblankStartEvent", "libSceVideoOut", cb_add_pre_vblank_start_event),
    ("elWQ9vERF-Q", "sceVideoOutDeletePreVblankStartEvent", "libSceVideoOut", cb_delete_pre_vblank_start_event),
    ("Mt4QHHkxkOc", "sceVideoOutGetEventCount", "libSceVideoOut", cb_get_event_count),
    ("w7Ipp9Xl7hg", "sceVideoOutAllowOutputResolutionWqhdDetection", "libSceVideoOut", cb_allow_output_resolution_wqhd_detection),
];

/// Register libSceVideoOut HLE exports.
pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, lib, cb) in EXPORTS {
        reg.register_fn(nid, name, lib, cb);
    }
}
