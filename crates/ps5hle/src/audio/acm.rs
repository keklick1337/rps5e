// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceAcm (audio coprocessor DSP batches: FFT, panner, convolution reverb).
//!
//! No DSP job is executed. Job builders advance the batch buffer cursor by
//! the job size and write nothing else, batch submissions complete
//! immediately with a cleared error block and a fresh batch id, and waits
//! succeed.

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

/// sceAcm_ConvReverb_SharedInput(batch, blockCount, input, count, ir[],
/// gain, out[] (first stack argument)): the job is only added to the
/// batch. Its outputs are left as they are: the layout `out[]` points at
/// is not known, and clearing through a guessed descriptor (sample and
/// channel counts and a channel table read from the first words) wrote
/// zeros wherever the audio data in those buffers happened to point —
/// Wwise heap blocks in SILENT HILL 2, whose allocator then crashed.
fn cb_conv_reverb_shared_input(ctx: &mut CpuContext) -> HleResult {
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

    /// A shared-input reverb job only advances the batch cursor: buffers
    /// its output list points at keep their bytes even when their first
    /// words look like a descriptor (sample count, channel count, channel
    /// table) that points at other memory.
    #[test]
    fn conv_reverb_writes_no_output() {
        const BASE: u64 = 0x5CE0_7100_0000;
        let base = BASE;
        let vm = crate::sync::test_util::vm_arena(base);
        let mut ctx = CpuContext::new(vm);
        let (info, stack, list, out, chans, victim) = (base, base + 0x100, base + 0x200, base + 0x300, base + 0x400, base + 0x500);
        ctx.write_u64(info, base + 0x1000);
        ctx.write_u64(info + 8, 0);
        ctx.write_u64(info + 16, 4096);
        ctx.write_u64(stack + 8, list);
        ctx.write_u64(list, out);
        ctx.write_u32(out, 4);
        ctx.write_u32(out + 4, 1);
        ctx.write_u64(out + 16, chans);
        ctx.write_u64(chans, victim);
        ctx.write_bytes(victim, &[0x7F; 32]);
        let before = ctx.read_bytes(out, 24).unwrap();
        ctx.set(CpuRegister::Rdi, info);
        ctx.set(CpuRegister::Rcx, 1);
        ctx.set(CpuRegister::Rsp, stack);
        assert_eq!(cb_conv_reverb_shared_input(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(victim, 32).unwrap(), vec![0x7F; 32], "memory the output words point at");
        assert_eq!(ctx.read_bytes(out, 24).unwrap(), before);
        assert_eq!(ctx.read_u64(info + 8), Some(1024));
    }
}
