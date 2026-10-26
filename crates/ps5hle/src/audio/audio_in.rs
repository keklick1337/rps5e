// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceAudioIn without a capture device.
//!
//! Ports can be opened with the parameters the library accepts; every input
//! grain is silence, released at the real capture cadence (one grain of
//! `len` frames per `len / freq` seconds), and the silent state reports that
//! no input device is connected. A port is busy while an input call is in
//! progress: closing it or reading from it again meanwhile fails.

use super::Export;
use crate::context::CpuContext;
use crate::registry::HleResult;
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const AUDIO_IN_ERROR_INVALID_HANDLE: HleResult = 0x8026_0101u32 as i32 as i64;
pub const AUDIO_IN_ERROR_INVALID_SIZE: HleResult = 0x8026_0102u32 as i32 as i64;
pub const AUDIO_IN_ERROR_INVALID_FREQ: HleResult = 0x8026_0103u32 as i32 as i64;
pub const AUDIO_IN_ERROR_INVALID_TYPE: HleResult = 0x8026_0104u32 as i32 as i64;
pub const AUDIO_IN_ERROR_INVALID_PARAM: HleResult = 0x8026_0106u32 as i32 as i64;
pub const AUDIO_IN_ERROR_PORT_FULL: HleResult = 0x8026_0107u32 as i32 as i64;
pub const AUDIO_IN_ERROR_BUSY: HleResult = 0x8026_010Au32 as i32 as i64;
pub const AUDIO_IN_SILENT_STATE_DEVICE_NONE: HleResult = 0x1;

const IN_PORTS_MAX: usize = 8;

#[derive(Clone, Copy)]
struct PortIn {
    samples: u32,
    freq: u32,
    bytes_per_frame: u32,
    asynchronous: bool,
    busy: bool,
    last_input: Option<Instant>,
}

static PORTS: Mutex<[Option<PortIn>; IN_PORTS_MAX]> = Mutex::new([None; IN_PORTS_MAX]);

/// Validate open parameters; returns bytes per frame.
pub fn validate_open(port_type: i32, index: i32, len: u32, freq: u32, param: u32, asynchronous: bool) -> Result<u32, HleResult> {
    if port_type != 0 && port_type != 1 {
        return Err(AUDIO_IN_ERROR_INVALID_TYPE);
    }
    if index != 0 {
        return Err(AUDIO_IN_ERROR_INVALID_PARAM);
    }
    if len != 128 && (asynchronous || len != 256) {
        return Err(AUDIO_IN_ERROR_INVALID_SIZE);
    }
    if freq != 48000 && (asynchronous || freq != 16000) {
        return Err(AUDIO_IN_ERROR_INVALID_FREQ);
    }
    match param {
        1 => Ok(2),        // s16 mono
        2 => Ok(4),        // s16 stereo
        0x10 | 0x11 => Ok(4), // float mono
        0x12 => Ok(8),     // float stereo
        _ => Err(AUDIO_IN_ERROR_INVALID_PARAM),
    }
}

/// sceAudioInAsyncOpen's parameters: also 384-frame grains, either rate.
fn validate_async_open(port_type: i32, index: i32, len: u32, freq: u32, param: u32) -> Result<u32, HleResult> {
    if !matches!(len, 128 | 256 | 384) {
        if port_type != 0 && port_type != 1 {
            return Err(AUDIO_IN_ERROR_INVALID_TYPE);
        }
        if index != 0 {
            return Err(AUDIO_IN_ERROR_INVALID_PARAM);
        }
        return Err(AUDIO_IN_ERROR_INVALID_SIZE);
    }
    validate_open(port_type, index, 128, freq, param, false)
}

fn open(ctx: &CpuContext, asynchronous: bool) -> HleResult {
    open_with(ctx, asynchronous, |t, i, l, f, p| validate_open(t, i, l, f, p, asynchronous))
}

fn open_with(
    ctx: &CpuContext,
    asynchronous: bool,
    validate: impl FnOnce(i32, i32, u32, u32, u32) -> Result<u32, HleResult>,
) -> HleResult {
    let (port_type, index, len, freq, param) =
        (ctx.rsi() as i32, ctx.rdx() as i32, ctx.rcx() as u32, ctx.r8() as u32, ctx.r9() as u32);
    let bpf = match validate(port_type, index, len, freq, param) {
        Ok(b) => b,
        Err(e) => return e,
    };
    let mut ports = PORTS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(slot) = ports.iter().position(|p| p.is_none()) else {
        return AUDIO_IN_ERROR_PORT_FULL;
    };
    ports[slot] = Some(PortIn { samples: len, freq, bytes_per_frame: bpf, asynchronous, busy: false, last_input: None });
    log::info!("[AUDIO] AudioIn port {} opened ({len} frames, {freq} Hz, param 0x{param:X}); no capture device, input is silence", slot + 1);
    (slot + 1) as HleResult
}

fn cb_open(ctx: &mut CpuContext) -> HleResult {
    open(ctx, false)
}

fn cb_hq_open(ctx: &mut CpuContext) -> HleResult {
    open(ctx, true)
}

/// sceAudioInAsyncOpen(userId, type, index, len, freq, param): the regular
/// input pacing.
fn cb_async_open(ctx: &mut CpuContext) -> HleResult {
    open_with(ctx, false, validate_async_open)
}

fn slot_of(handle: i32) -> Option<usize> {
    (handle > 0 && (handle as usize) <= IN_PORTS_MAX).then(|| handle as usize - 1)
}

fn cb_close(ctx: &mut CpuContext) -> HleResult {
    let mut ports = PORTS.lock().unwrap_or_else(|e| e.into_inner());
    match slot_of(ctx.rdi() as i32) {
        Some(s) => match ports[s] {
            Some(p) if p.busy => AUDIO_IN_ERROR_BUSY,
            Some(_) => {
                ports[s] = None;
                0
            }
            None => AUDIO_IN_ERROR_INVALID_HANDLE,
        },
        None => AUDIO_IN_ERROR_INVALID_HANDLE,
    }
}

/// sceAudioInInput(handle, void *dest): one grain of silence; returns frames.
fn cb_input(ctx: &mut CpuContext) -> HleResult {
    let dest = ctx.rsi();
    let Some(s) = slot_of(ctx.rdi() as i32) else { return AUDIO_IN_ERROR_INVALID_HANDLE };
    let port = {
        let mut ports = PORTS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(p) = ports[s].as_mut() else { return AUDIO_IN_ERROR_INVALID_HANDLE };
        if p.busy {
            return AUDIO_IN_ERROR_BUSY;
        }
        p.busy = true;
        *p
    };
    let block = Duration::from_micros(1_000_000 * port.samples as u64 / port.freq as u64);
    let wait = if port.asynchronous {
        if dest != 0 { block } else { Duration::ZERO }
    } else {
        port.last_input
            .map(|t| (t + block).saturating_duration_since(Instant::now()))
            .unwrap_or(Duration::ZERO)
    };
    if !wait.is_zero() {
        crate::signals::sleep(wait);
    }
    if dest != 0 {
        // At most 384 frames of 8 bytes.
        static SILENCE: [u8; 3072] = [0; 3072];
        let bytes = (port.samples * port.bytes_per_frame) as usize;
        ctx.write_bytes(dest, &SILENCE[..bytes.min(SILENCE.len())]);
    }
    if let Some(p) = PORTS.lock().unwrap_or_else(|e| e.into_inner())[s].as_mut() {
        p.last_input = (dest != 0).then(Instant::now);
        p.busy = false;
    }
    if dest != 0 {
        port.samples as HleResult
    } else {
        0
    }
}

fn cb_get_silent_state(ctx: &mut CpuContext) -> HleResult {
    let ports = PORTS.lock().unwrap_or_else(|e| e.into_inner());
    match slot_of(ctx.rdi() as i32) {
        Some(s) if ports[s].is_some() => AUDIO_IN_SILENT_STATE_DEVICE_NONE,
        _ => AUDIO_IN_ERROR_INVALID_HANDLE,
    }
}

fn port_open(handle: i32) -> bool {
    let ports = PORTS.lock().unwrap_or_else(|e| e.into_inner());
    slot_of(handle).is_some_and(|s| ports[s].is_some())
}

/// sceAudioInGetGain(handle, float* gain): unity gain (written before the
/// handle is checked).
fn cb_get_gain(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        log::error!("[AUDIO] sceAudioInGetGain: null gain");
        return 0x8002_0002u32 as i32 as i64;
    }
    ctx.write_bytes(out, &1.0f32.to_le_bytes());
    if port_open(ctx.rdi() as i32) { 0 } else { AUDIO_IN_ERROR_INVALID_HANDLE }
}

/// sceAudioInGetRerouteCount(handle): input is never rerouted.
fn cb_get_reroute_count(ctx: &mut CpuContext) -> HleResult {
    if port_open(ctx.rdi() as i32) { 0 } else { AUDIO_IN_ERROR_INVALID_HANDLE }
}

/// sceAudioInInit(): the library needs no setup; ports open on demand.
fn cb_init(_ctx: &mut CpuContext) -> HleResult {
    0
}

pub(crate) static EXPORTS: &[Export] = &[
    ("SxQprgjttKE", "sceAudioInInit", "libSceAudioIn", cb_init),
    ("5NE8Sjc7VC8", "sceAudioInOpen", "libSceAudioIn", cb_open),
    ("nya-R5gDYhM", "sceAudioInHqOpen", "libSceAudioIn", cb_hq_open),
    ("3YQ1qY5ePUk", "sceAudioInAsyncOpen", "libSceAudioIn", cb_async_open),
    ("Jh6WbHhnI68", "sceAudioInClose", "libSceAudioIn", cb_close),
    ("LozEOU8+anM", "sceAudioInInput", "libSceAudioIn", cb_input),
    ("BohEAQ7DlUE", "sceAudioInGetSilentState", "libSceAudioIn", cb_get_silent_state),
    ("S-rDUfQk9sg", "sceAudioInGetGain", "libSceAudioIn", cb_get_gain),
    ("3shKmTrTw6c", "sceAudioInGetRerouteCount", "libSceAudioIn", cb_get_reroute_count),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_validation() {
        assert_eq!(validate_open(0, 0, 256, 16000, 1, false), Ok(2));
        assert_eq!(validate_open(1, 0, 128, 48000, 0x12, true), Ok(8));
        assert_eq!(validate_open(2, 0, 256, 48000, 1, false), Err(AUDIO_IN_ERROR_INVALID_TYPE));
        assert_eq!(validate_open(0, 1, 256, 48000, 1, false), Err(AUDIO_IN_ERROR_INVALID_PARAM));
        assert_eq!(validate_open(0, 0, 256, 48000, 1, true), Err(AUDIO_IN_ERROR_INVALID_SIZE));
        assert_eq!(validate_open(0, 0, 128, 44100, 1, false), Err(AUDIO_IN_ERROR_INVALID_FREQ));
        assert_eq!(validate_open(0, 0, 128, 48000, 3, false), Err(AUDIO_IN_ERROR_INVALID_PARAM));
        let vm = ps5vm::VirtualMemory::new();
        assert_eq!(cb_init(&mut CpuContext::new(&vm)), 0);
        // sceAudioInAsyncOpen: 128, 256 or 384 frames at either rate.
        assert_eq!(validate_async_open(0, 0, 384, 48000, 0x12), Ok(8));
        assert_eq!(validate_async_open(1, 0, 256, 16000, 1), Ok(2));
        assert_eq!(validate_async_open(0, 0, 512, 48000, 1), Err(AUDIO_IN_ERROR_INVALID_SIZE));
        assert_eq!(validate_async_open(0, 0, 384, 44100, 1), Err(AUDIO_IN_ERROR_INVALID_FREQ));
        assert_eq!(validate_async_open(2, 0, 384, 48000, 1), Err(AUDIO_IN_ERROR_INVALID_TYPE));
        assert_eq!(validate_async_open(0, 1, 512, 48000, 1), Err(AUDIO_IN_ERROR_INVALID_PARAM));
        assert_eq!(validate_async_open(0, 0, 384, 48000, 3), Err(AUDIO_IN_ERROR_INVALID_PARAM));
    }

    /// A 384-frame float stereo grain is all silence.
    #[test]
    fn async_open_delivers_a_384_frame_grain() {
        use crate::context::CpuRegister;
        const BASE: u64 = 0x5B01_0000_0000;
        let vm = crate::sync::test_util::vm_arena(BASE);
        let mut ctx = CpuContext::new(vm);
        ctx.set(CpuRegister::Rcx, 384);
        ctx.set(CpuRegister::R8, 48000);
        ctx.set(CpuRegister::R9, 0x12);
        let h = cb_async_open(&mut ctx);
        assert!(h > 0);
        ctx.set(CpuRegister::Rdi, h as u64);
        ctx.set(CpuRegister::Rsi, BASE);
        ctx.write_bytes(BASE, &[0x55; 3073]);
        assert_eq!(cb_input(&mut ctx), 384);
        assert_eq!(ctx.read_bytes(BASE, 3073).unwrap(), [vec![0; 3072], vec![0x55]].concat());
        assert_eq!(cb_get_gain(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE, 4).unwrap(), 1.0f32.to_le_bytes().to_vec());
        assert_eq!(cb_get_reroute_count(&mut ctx), 0);
        assert_eq!(cb_close(&mut ctx), 0);
        assert_eq!(cb_get_reroute_count(&mut ctx), AUDIO_IN_ERROR_INVALID_HANDLE);
        assert_eq!(cb_get_gain(&mut ctx), AUDIO_IN_ERROR_INVALID_HANDLE);
    }

    #[test]
    fn busy_port_and_silence() {
        use crate::context::CpuRegister;
        const BASE: u64 = 0x5CE6_0000_0000;
        let vm = crate::sync::test_util::vm_arena(BASE);
        let mut ctx = CpuContext::new(vm);
        // sceAudioInOpen(user, type 0, index 0, 128 frames, 48 kHz, s16 stereo)
        ctx.set(CpuRegister::Rcx, 128);
        ctx.set(CpuRegister::R8, 48000);
        ctx.set(CpuRegister::R9, 2);
        let h = cb_open(&mut ctx);
        assert!(h > 0);
        let s = slot_of(h as i32).unwrap();
        ctx.set(CpuRegister::Rdi, h as u64);
        ctx.set(CpuRegister::Rsi, BASE);
        PORTS.lock().unwrap()[s].as_mut().unwrap().busy = true;
        assert_eq!(cb_close(&mut ctx), AUDIO_IN_ERROR_BUSY, "busy port was closed");
        assert_eq!(cb_input(&mut ctx), AUDIO_IN_ERROR_BUSY, "busy port accepted a second input");
        PORTS.lock().unwrap()[s].as_mut().unwrap().busy = false;
        ctx.write_bytes(BASE, &[0x55; 513]);
        assert_eq!(cb_input(&mut ctx), 128);
        assert_eq!(ctx.read_bytes(BASE, 513).unwrap(), [vec![0; 512], vec![0x55]].concat());
        assert_eq!(cb_get_silent_state(&mut ctx), AUDIO_IN_SILENT_STATE_DEVICE_NONE);
        assert_eq!(cb_close(&mut ctx), 0);
        assert_eq!(cb_close(&mut ctx), AUDIO_IN_ERROR_INVALID_HANDLE);
    }
}
