// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceAcm (audio coprocessor DSP batches: FFT, panner, convolution reverb).
//!
//! No DSP job is executed. Job builders advance the batch buffer cursor by
//! the job size (a convolution reverb also clears its outputs, so no stale
//! samples are played), batch submissions complete immediately with a
//! cleared error block and a fresh batch id, and waits succeed.

use super::Export;
use crate::context::CpuContext;
use crate::registry::HleResult;
use std::sync::atomic::{AtomicU32, Ordering};

const ERR_BAD_CONTEXT: HleResult = 0x8194_0002u32 as i32 as i64;
const ERR_BAD_POINTER: HleResult = 0x8194_0006u32 as i32 as i64;
const ERR_MEMORY_FAULT: HleResult = 0x8002_000Eu32 as i32 as i64;
const ACM_BATCH_ERROR_BYTES: usize = 32;

static NEXT_CONTEXT: AtomicU32 = AtomicU32::new(1);
static NEXT_BATCH: AtomicU32 = AtomicU32::new(1);
/// Live contexts (sceAcmContextCreate until sceAcmContextDestroy).
static CONTEXTS: std::sync::Mutex<std::collections::BTreeSet<u32>> = std::sync::Mutex::new(std::collections::BTreeSet::new());

fn context_live(id: u32) -> bool {
    id != 0 && CONTEXTS.lock().unwrap_or_else(|p| p.into_inner()).contains(&id)
}

/// sceAcmContextCreate(u32* context)
fn cb_context_create(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_BAD_POINTER;
    }
    let h = NEXT_CONTEXT.fetch_add(1, Ordering::Relaxed);
    if !ctx.write_u32(out, h) {
        return ERR_MEMORY_FAULT;
    }
    CONTEXTS.lock().unwrap_or_else(|p| p.into_inner()).insert(h);
    log::info!("[AUDIO] sceAcmContextCreate -> {h}");
    0
}

/// sceAcmContextDestroy(context): an unknown context is BAD_CONTEXT.
fn cb_context_destroy(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as u32;
    if id != 0 && CONTEXTS.lock().unwrap_or_else(|p| p.into_inner()).remove(&id) { 0 } else { ERR_BAD_CONTEXT }
}

/// sceAcmBatchWait(context, batch, timeout): batches complete when started.
fn cb_batch_wait(ctx: &mut CpuContext) -> HleResult {
    if context_live(ctx.rdi() as u32) { 0 } else { ERR_BAD_CONTEXT }
}

fn complete_batch_start(ctx: &CpuContext, err_addr: u64, batch_addr: u64) -> HleResult {
    if err_addr != 0 && !ctx.write_bytes(err_addr, &[0u8; ACM_BATCH_ERROR_BYTES]) {
        return ERR_MEMORY_FAULT;
    }
    let b = NEXT_BATCH.fetch_add(1, Ordering::Relaxed);
    if !ctx.write_u32(batch_addr, b) {
        return ERR_MEMORY_FAULT;
    }
    0
}

/// sceAcmBatchStartBuffer(ctx, commands, size, error*, batch*) and
/// sceAcmBatchStartBuffers(ctx, count, infos*, error*, batch*): the batch
/// pointer, then the context.
fn batch_start(ctx: &CpuContext) -> HleResult {
    let (context, err, batch) = (ctx.rdi() as u32, ctx.rcx(), ctx.r8());
    if batch == 0 {
        return ERR_BAD_POINTER;
    }
    if !context_live(context) {
        return ERR_BAD_CONTEXT;
    }
    complete_batch_start(ctx, err, batch)
}

fn cb_batch_start_buffer(ctx: &mut CpuContext) -> HleResult {
    batch_start(ctx)
}

fn cb_batch_start_buffers(ctx: &mut CpuContext) -> HleResult {
    batch_start(ctx)
}

fn cb_ok(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// Advance the batch info {buffer, offset, size} cursor by `bytes` (capped at size).
pub(crate) fn advance_batch_info(ctx: &CpuContext, bytes: u64) -> HleResult {
    let info = ctx.rdi();
    if info == 0 {
        return 0;
    }
    let (Some(buffer), Some(offset), Some(size)) = (ctx.read_u64(info), ctx.read_u64(info + 8), ctx.read_u64(info + 16))
    else {
        return ERR_MEMORY_FAULT;
    };
    if buffer != 0 && size != 0 {
        let next = offset.saturating_add(bytes).min(size);
        if !ctx.write_u64(info + 8, next) {
            return ERR_MEMORY_FAULT;
        }
    }
    0
}

fn cb_job_notification(ctx: &mut CpuContext) -> HleResult {
    advance_batch_info(ctx, 32)
}
fn cb_conv_reverb(ctx: &mut CpuContext) -> HleResult {
    advance_batch_info(ctx, 1024)
}

/// Output descriptor of a reverb job: samples u32 @0, channels u32 @4,
/// float* buffers[channels] @0x10.
const REVERB_OUTPUT_DESCRIPTOR_SIZE: usize = 24;
const REVERB_MAX_OUTPUTS: u64 = 64;
const REVERB_MAX_CHANNELS: u32 = 64;
const REVERB_MAX_SAMPLES: u32 = 65536;

/// sceAcm_ConvReverb_SharedInput(batch, ..., outputCount (rcx), ...,
/// outputs (first stack argument)): the convolution is not run, so its
/// output channels are cleared rather than left holding stale samples.
fn cb_conv_reverb_shared_input(ctx: &mut CpuContext) -> HleResult {
    let count = ctx.rcx() as u32 as u64;
    let rsp = ctx.get(crate::context::CpuRegister::Rsp);
    let outputs = ctx.read_u64(rsp + 8).unwrap_or(0);
    if outputs != 0 {
        for i in 0..count.min(REVERB_MAX_OUTPUTS) {
            let Some(d) = ctx.read_u64(outputs + i * 8).filter(|&d| d != 0) else { continue };
            let Some(desc) = ctx.read_bytes(d, REVERB_OUTPUT_DESCRIPTOR_SIZE) else { continue };
            let samples = u32::from_le_bytes(desc[0..4].try_into().unwrap());
            let channels = u32::from_le_bytes(desc[4..8].try_into().unwrap());
            let buffers = u64::from_le_bytes(desc[16..24].try_into().unwrap());
            if samples == 0 || samples > REVERB_MAX_SAMPLES || channels == 0 || channels > REVERB_MAX_CHANNELS || buffers == 0 {
                continue;
            }
            let silence = vec![0u8; samples as usize * 4];
            for c in 0..channels as u64 {
                if let Some(b) = ctx.read_u64(buffers + c * 8).filter(|&b| b != 0) {
                    ctx.write_bytes(b, &silence);
                }
            }
        }
    }
    advance_batch_info(ctx, 1024)
}
fn cb_fft(ctx: &mut CpuContext) -> HleResult {
    advance_batch_info(ctx, 256)
}
fn cb_panner(ctx: &mut CpuContext) -> HleResult {
    advance_batch_info(ctx, 512)
}

const LIB: &str = "libSceAcm";

pub(crate) static EXPORTS: &[Export] = &[
    ("ZIXln2K3XMk", "sceAcmContextCreate", LIB, cb_context_create),
    ("jBgBjAj02R8", "sceAcmContextDestroy", LIB, cb_context_destroy),
    ("tW9W+CAG4FE", "sceAcmBatchStartBuffer", LIB, cb_batch_start_buffer),
    ("8fe55ktlNVo", "sceAcmBatchStartBuffers", LIB, cb_batch_start_buffers),
    ("WeZOIm8+8WI", "sceAcmBatchInitialize", LIB, cb_ok),
    ("Mk1xvQXIdkk", "sceAcmBatchInitializeLite", LIB, cb_ok),
    ("A5NXCXK5Gfc", "sceAcmBatchStart", LIB, cb_ok),
    ("S3BPrjCfZ90", "sceAcmBatchStartMultiple", LIB, cb_ok),
    ("uqDIauipRbo", "sceAcmBatchProcess", LIB, cb_ok),
    ("RLN3gRlXJLE", "sceAcmBatchWait", LIB, cb_batch_wait),
    ("r7z5YQFZo+U", "sceAcmBatchJobNotification", LIB, cb_job_notification),
    ("u70oWo92SYQ", "sceAcm_ConvReverb_SharedInput", LIB, cb_conv_reverb_shared_input),
    ("9nLbWmRDpa8", "sceAcm_ConvReverb_SharedIr", LIB, cb_conv_reverb),
    ("KovqaFbmtsM", "sceAcm_FFT", LIB, cb_fft),
    ("DR-ZCmvVR9Q", "sceAcm_IFFT", LIB, cb_fft),
    ("LA4RCNKnFjg", "sceAcm_Panner", LIB, cb_panner),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister;

    #[test]
    fn context_batch_and_cursor() {
        const BASE: u64 = 0x5CE0_7000_0000;
        let base = BASE;
        let vm = crate::sync::test_util::vm_arena(base);
        let mut ctx = CpuContext::new(vm);
        ctx.set(CpuRegister::Rdi, base);
        assert_eq!(cb_context_create(&mut ctx), 0);
        let c = ctx.read_u32(base).unwrap();
        ctx.write_bytes(base + 0x100, &[0xAA; 32]);
        ctx.set(CpuRegister::Rdi, c as u64);
        ctx.set(CpuRegister::Rsi, 0);
        ctx.set(CpuRegister::Rdx, 0);
        ctx.set(CpuRegister::Rcx, base + 0x100);
        ctx.set(CpuRegister::R8, base + 0x200);
        assert_eq!(cb_batch_start_buffers(&mut ctx), 0);
        assert!(ctx.read_bytes(base + 0x100, 32).unwrap().iter().all(|&b| b == 0));
        assert!(ctx.read_u32(base + 0x200).unwrap() > 0);
        assert_eq!(cb_ok(&mut ctx), 0);
        // Cursor: buffer, offset 0, size 600 -> FFT 256, then reverb capped at 600.
        ctx.write_u64(base + 0x300, base + 0x1000);
        ctx.write_u64(base + 0x308, 0);
        ctx.write_u64(base + 0x310, 600);
        ctx.set(CpuRegister::Rdi, base + 0x300);
        assert_eq!(cb_fft(&mut ctx), 0);
        assert_eq!(ctx.read_u64(base + 0x308), Some(256));
        assert_eq!(cb_conv_reverb(&mut ctx), 0);
        assert_eq!(ctx.read_u64(base + 0x308), Some(600));
        ctx.set(CpuRegister::Rdi, c as u64);
        assert_eq!(cb_batch_wait(&mut ctx), 0);
        assert_eq!(cb_context_destroy(&mut ctx), 0);
        // The context is gone; then a missing batch id pointer comes first.
        assert_eq!(cb_context_destroy(&mut ctx), ERR_BAD_CONTEXT);
        assert_eq!(cb_batch_wait(&mut ctx), ERR_BAD_CONTEXT);
        assert_eq!(cb_batch_start_buffer(&mut ctx), ERR_BAD_CONTEXT);
        ctx.set(CpuRegister::R8, 0);
        assert_eq!(cb_batch_start_buffer(&mut ctx), ERR_BAD_POINTER);
        ctx.set(CpuRegister::Rdi, 0);
        assert_eq!(cb_context_create(&mut ctx), ERR_BAD_POINTER);
        assert_eq!(cb_context_destroy(&mut ctx), ERR_BAD_CONTEXT);
    }

    /// A shared-input reverb job clears the channels of its outputs (the
    /// convolution itself is not run) and advances the batch cursor.
    #[test]
    fn conv_reverb_clears_outputs() {
        const BASE: u64 = 0x5CE0_7100_0000;
        let base = BASE;
        let vm = crate::sync::test_util::vm_arena(base);
        let mut ctx = CpuContext::new(vm);
        let (info, stack, list, desc, chans, out0, out1) =
            (base, base + 0x100, base + 0x200, base + 0x300, base + 0x400, base + 0x500, base + 0x600);
        ctx.write_u64(info, base + 0x1000);
        ctx.write_u64(info + 8, 0);
        ctx.write_u64(info + 16, 4096);
        ctx.write_u64(stack + 8, list);
        ctx.write_u64(list, desc);
        ctx.write_u32(desc, 4); // samples
        ctx.write_u32(desc + 4, 2); // channels
        ctx.write_u64(desc + 16, chans);
        ctx.write_u64(chans, out0);
        ctx.write_u64(chans + 8, out1);
        ctx.write_bytes(out0, &[0x7F; 32]);
        ctx.write_bytes(out1, &[0x7F; 32]);
        ctx.set(CpuRegister::Rdi, info);
        ctx.set(CpuRegister::Rcx, 1);
        ctx.set(CpuRegister::Rsp, stack);
        assert_eq!(cb_conv_reverb_shared_input(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(out0, 32).unwrap(), [vec![0; 16], vec![0x7F; 16]].concat(), "4 float samples");
        assert_eq!(ctx.read_bytes(out1, 16).unwrap(), vec![0; 16]);
        assert_eq!(ctx.read_u64(info + 8), Some(1024));
    }
}
