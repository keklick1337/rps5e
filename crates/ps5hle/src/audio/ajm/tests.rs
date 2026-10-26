// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only
//
// Unit tests of the `ps5hle::audio::ajm` module.

use super::*;
use crate::audio::atrac9::tests::{build_mono_superframe, mono_config};

fn setup(base: u64) -> CpuContext<'static> {
    CpuContext::new(crate::sync::test_util::vm_arena(base))
}

fn call(ctx: &mut CpuContext, f: fn(&mut CpuContext) -> HleResult, args: &[u64]) -> HleResult {
    let regs = [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx, CpuRegister::R8, CpuRegister::R9];
    for (r, v) in regs.iter().zip(args) {
        ctx.set(*r, *v);
    }
    // Stack arguments go after the return address.
    let rsp = ctx.get(CpuRegister::Rsp);
    for (i, v) in args.iter().skip(6).enumerate() {
        ctx.write_u64(rsp + 8 + i as u64 * 8, *v);
    }
    f(ctx)
}

fn i32_at(b: &[u8], o: usize) -> i32 {
    i32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

/// Context, module registration, instance ids and the MP3 decode job.
#[test]
fn context_instance_and_batch_jobs() {
    const BASE: u64 = 0x5CE0_0000_0000;
    let mut ctx = setup(BASE);
    ctx.set(CpuRegister::Rsp, BASE + 0x800);

    assert_eq!(call(&mut ctx, cb_initialize, &[0, BASE]), 0);
    let c = ctx.read_u32(BASE).unwrap() as u64;
    assert_eq!(c, 1);
    assert_eq!(call(&mut ctx, cb_module_register, &[c, CODEC_DEC_MP3 as u64, 0]), 0);
    assert_eq!(call(&mut ctx, cb_module_register, &[2, CODEC_DEC_MP3 as u64, 0]), AJM_ERROR_INVALID_CONTEXT);
    assert_eq!(call(&mut ctx, cb_module_register, &[c, 5, 0]), AJM_ERROR_INVALID_PARAMETER);
    assert_eq!(call(&mut ctx, cb_module_register, &[c, 22, 0]), AJM_ERROR_CODEC_NOT_SUPPORTED);
    assert_eq!(call(&mut ctx, cb_module_register, &[c, CODEC_DEC_OPUS as u64, 0]), 0);

    // S16 stereo MP3 instance.
    assert_eq!(call(&mut ctx, cb_instance_create, &[c, CODEC_DEC_MP3 as u64, 2, BASE + 8]), 0);
    let inst = ctx.read_u32(BASE + 8).unwrap();
    assert_eq!(inst >> INSTANCE_SLOT_BITS, CODEC_DEC_MP3);
    assert_ne!(inst & INSTANCE_SLOT_MASK, 0);

    // Batch: 0x100 bytes at BASE+0x1000, info at BASE+0x100.
    assert_eq!(call(&mut ctx, cb_batch_initialize, &[BASE + 0x1000, 0x100, BASE + 0x100]), 0);

    // A silent MP3 frame (128 kbps 48 kHz stereo, 384 bytes).
    let mut frame = vec![0u8; 384];
    frame[..4].copy_from_slice(&[0xFF, 0xFB, 0x94, 0x00]);
    ctx.write_bytes(BASE + 0x2000, &frame);
    ctx.write_bytes(BASE + 0x3000, &[0x77u8; 1152 * 4 + 16]);
    let args = [BASE + 0x100, inst as u64, BASE + 0x2000, 384, BASE + 0x3000, 1152 * 4 + 16, BASE + 0x200];
    assert_eq!(call(&mut ctx, cb_batch_job_decode, &args), 0);
    assert_eq!(ctx.read_u64(BASE + 0x108), Some(JOB_RUN_SIZE));
    assert_eq!(ctx.read_u64(BASE + 0x118), Some(BASE + 0x1000));
    let r = ctx.read_bytes(BASE + 0x200, 32).unwrap();
    assert_eq!(i32_at(&r, 0), 0);
    assert_eq!(i32_at(&r, 8), 384);
    assert_eq!(i32_at(&r, 12), 1152 * 4);
    assert_eq!(u64::from_le_bytes(r[16..24].try_into().unwrap()), 1152);
    assert_eq!(i32_at(&r, 24), 1);
    let out = ctx.read_bytes(BASE + 0x3000, 1152 * 4 + 16).unwrap();
    assert!(out[..1152 * 4].iter().all(|&b| b == 0));
    // Bytes past the produced PCM are left alone.
    assert!(out[1152 * 4..].iter().all(|&b| b == 0x77));

    // Batch full: the job still runs but cannot be added.
    for _ in 0..3 {
        assert_eq!(call(&mut ctx, cb_batch_job_decode, &args), 0);
    }
    assert_eq!(call(&mut ctx, cb_batch_job_decode, &args), AJM_ERROR_JOB_CREATION);

    // Start / wait clear the error block and publish a batch id.
    ctx.write_bytes(BASE + 0x400, &[0xEE; BATCH_ERROR_SIZE]);
    assert_eq!(call(&mut ctx, cb_batch_start, &[c, BASE + 0x100, 0, BASE + 0x400, BASE + 0x420]), 0);
    assert!(ctx.read_u32(BASE + 0x420).unwrap() > 0);
    assert!(ctx.read_bytes(BASE + 0x400, BATCH_ERROR_SIZE).unwrap().iter().all(|&b| b == 0));
    let batch = ctx.read_u32(BASE + 0x420).unwrap() as u64;
    ctx.write_bytes(BASE + 0x400, &[0xEE; BATCH_ERROR_SIZE]);
    // Batches belong to their context: another context is INVALID_CONTEXT,
    // an unknown or already waited batch INVALID_BATCH.
    assert_eq!(call(&mut ctx, cb_batch_cancel, &[c, batch]), 0);
    assert_eq!(call(&mut ctx, cb_batch_wait, &[c + 1, batch, 0, BASE + 0x400]), AJM_ERROR_INVALID_CONTEXT);
    assert_eq!(call(&mut ctx, cb_batch_wait, &[c, batch, 0, BASE + 0x400]), 0);
    assert!(ctx.read_bytes(BASE + 0x400, BATCH_ERROR_SIZE).unwrap().iter().all(|&b| b == 0));
    assert_eq!(call(&mut ctx, cb_batch_wait, &[c, batch, 0, BASE + 0x400]), AJM_ERROR_INVALID_BATCH);
    assert_eq!(call(&mut ctx, cb_batch_cancel, &[c, batch]), AJM_ERROR_INVALID_BATCH);
    assert_eq!(call(&mut ctx, cb_batch_start, &[c + 1, BASE + 0x100, 0, BASE + 0x400, BASE + 0x420]), AJM_ERROR_INVALID_CONTEXT);

    // Unknown instances report INVALID_PARAMETER in the sideband.
    assert_eq!(call(&mut ctx, cb_batch_initialize, &[BASE + 0x1000, 0x100, BASE + 0x100]), 0);
    let bad = [BASE + 0x100, 0x3FFF, BASE + 0x2000, 384, BASE + 0x3000, 4096, BASE + 0x200];
    assert_eq!(call(&mut ctx, cb_batch_job_decode, &bad), 0);
    assert_eq!(i32_at(&ctx.read_bytes(BASE + 0x200, 8).unwrap(), 0), RESULT_INVALID_PARAMETER);

    assert_eq!(call(&mut ctx, cb_instance_destroy, &[c, inst as u64]), 0);
    assert_eq!(call(&mut ctx, cb_finalize, &[c]), 0);
}

/// Run sideband layout: Result, Stream, Format, Gapless, ResampleInfo,
/// MFrame, CodecInfo.
#[test]
fn run_sideband_layout() {
    const BASE: u64 = 0x5CE0_1000_0000;
    let ctx = setup(BASE);
    let r = DecodeResult {
        result: 0,
        internal_result: 0,
        input_consumed: 10,
        output_written: 20,
        total_decoded_samples: 30,
        frames: 2,
        format: SidebandFormat::new(2, 48000, PcmEncoding::Float),
    };
    let gapless = GaplessDecode { total_samples: 7, skip_samples: 8, skipped_samples: 9 };
    let info = [1u8, 2, 3, 4];
    let flags = FLAG_RUN_GET_CODEC_INFO
        | FLAG_SIDEBAND_RESAMPLE_INFO
        | FLAG_SIDEBAND_GAPLESS
        | FLAG_SIDEBAND_FORMAT
        | FLAG_SIDEBAND_STREAM
        | FLAG_RUN_MULTIPLE_FRAMES;
    ctx.write_bytes(BASE, &[0xAB; 256]);
    write_sideband(&ctx, flags, BASE, 200, Some((16, &info)), Some(gapless), &r);
    let b = ctx.read_bytes(BASE, 256).unwrap();
    let s = 8;
    assert_eq!((u32_at(&b, s), u32_at(&b, s + 4), u32_at(&b, s + 8)), (10, 20, 30));
    let f = s + 16;
    assert_eq!(u32_at(&b, f), 2);
    assert_eq!(u32_at(&b, f + 4), 0x3);
    assert_eq!(u32_at(&b, f + 8), 48000);
    assert_eq!(u32_at(&b, f + 12), 2);
    let g = f + 24;
    assert_eq!(GaplessDecode::from_bytes(&b[g..g + 8]), gapless);
    let ri = g + 8;
    assert_eq!(f32::from_le_bytes(b[ri..ri + 4].try_into().unwrap()), 1.0);
    let m = ri + 40;
    assert_eq!(u32_at(&b, m), 2);
    let c = m + 8;
    assert_eq!(&b[c..c + 4], &info);
    // The whole requested size is cleared, nothing past it is touched.
    assert_eq!(&b[c + 4..200], &vec![0u8; 200 - c - 4][..]);
    assert_eq!(b[200], 0xAB);

    // Stream right after Result even when Format and Gapless are requested.
    ctx.write_bytes(BASE, &[0xAB; 256]);
    write_sideband(&ctx, FLAG_SIDEBAND_STREAM | FLAG_SIDEBAND_FORMAT | FLAG_SIDEBAND_GAPLESS, BASE, 56, None, Some(gapless), &r);
    let b = ctx.read_bytes(BASE, 64).unwrap();
    assert_eq!(u32_at(&b, 8), 10);
    assert_eq!(u32_at(&b, 12), 20);
    assert_eq!(u32_at(&b, 24), 2);
    assert_eq!(GaplessDecode::from_bytes(&b[48..56]), gapless);
    assert_eq!(b[56], 0xAB);

    // Blocks that do not fit are skipped; later smaller ones still go in.
    ctx.write_bytes(BASE, &[0xAB; 256]);
    write_sideband(&ctx, FLAG_SIDEBAND_FORMAT | FLAG_RUN_MULTIPLE_FRAMES, BASE, 16, None, None, &r);
    let b = ctx.read_bytes(BASE, 24).unwrap();
    assert_eq!(&b[..8], &[0; 8]);
    assert_eq!(&b[8..16], &[2, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(b[16], 0xAB);
}

/// ATRAC9 through the batch API: initialize, run with sidebands, codec
/// info, gapless get/set, control reset.
#[test]
fn atrac9_run_jobs() {
    const BASE: u64 = 0x5CE0_2000_0000;
    let mut ctx = setup(BASE);
    ctx.set(CpuRegister::Rsp, BASE + 0x800);
    assert_eq!(call(&mut ctx, cb_instance_create, &[1, CODEC_DEC_AT9 as u64, 1, BASE]), 0);
    let inst = ctx.read_u32(BASE).unwrap() as u64;
    assert_eq!(call(&mut ctx, cb_batch_initialize, &[BASE + 0x1000, 0x1000, BASE + 0x100]), 0);
    let info = BASE + 0x100;

    // Initialize with AjmDecAt9InitializeParameters.
    let mut params = mono_config().to_vec();
    params.extend_from_slice(&[0; 4]);
    ctx.write_bytes(BASE + 0x200, &params);
    assert_eq!(call(&mut ctx, cb_batch_job_initialize, &[info, inst, BASE + 0x200, 8, BASE + 0x300]), 0);
    assert_eq!(i32_at(&ctx.read_bytes(BASE + 0x300, 8).unwrap(), 0), 0);
    assert_eq!(ctx.read_u64(info + 8), Some(JOB_CONTROL_SIZE));

    // GetInfo: format of the stream.
    assert_eq!(call(&mut ctx, cb_batch_job_get_info, &[info, inst, BASE + 0x300]), 0);
    let b = ctx.read_bytes(BASE + 0x300, 32).unwrap();
    assert_eq!((u32_at(&b, 8), u32_at(&b, 12), u32_at(&b, 16)), (1, 0x4, 48000));

    // Run: one superframe, multiple frames, stream + format + mframe.
    let mut coefficients = [0i32; 24];
    coefficients[3] = 20000;
    let sf = build_mono_superframe(&coefficients);
    ctx.write_bytes(BASE + 0x2000, &sf);
    let flags = FLAG_RUN_MULTIPLE_FRAMES | FLAG_SIDEBAND_STREAM | FLAG_SIDEBAND_FORMAT | FLAG_RUN_GET_CODEC_INFO;
    let run = [info, inst, flags, BASE + 0x2000, sf.len() as u64, BASE + 0x3000, 4096, BASE + 0x400, 128];
    assert_eq!(call(&mut ctx, cb_batch_job_run, &run), 0);
    let b = ctx.read_bytes(BASE + 0x400, 128).unwrap();
    assert_eq!(i32_at(&b, 0), 0x8, "the input ran out with room left: PARTIAL_INPUT");
    // Stream, then format, then mframe, then codec info.
    assert_eq!((u32_at(&b, 8), u32_at(&b, 12)), (512, 2048));
    assert_eq!(u64::from_le_bytes(b[16..24].try_into().unwrap()), 1024);
    assert_eq!(u32_at(&b, 24), 1);
    assert_eq!(u32_at(&b, 48), 4);
    // Codec info: superframe 512, 4 frames, next 512, 256 samples.
    assert_eq!((u32_at(&b, 56), u32_at(&b, 60), u32_at(&b, 64), u32_at(&b, 68)), (512, 4, 512, 256));
    assert!(ctx.read_bytes(BASE + 0x3000, 2048).unwrap().iter().any(|&v| v != 0));

    // Gapless set (reset) and get.
    let g = GaplessDecode { total_samples: 500, skip_samples: 10, skipped_samples: 0 };
    ctx.write_bytes(BASE + 0x500, &g.to_bytes());
    assert_eq!(call(&mut ctx, cb_batch_job_set_gapless_decode, &[info, inst, BASE + 0x500, 1, BASE + 0x300]), 0);
    assert_eq!(call(&mut ctx, cb_batch_job_get_gapless_decode, &[info, inst, BASE + 0x300]), 0);
    let b = ctx.read_bytes(BASE + 0x300, 16).unwrap();
    assert_eq!(GaplessDecode::from_bytes(&b[8..16]), g);

    // Control with reset clears the decoded sample count.
    assert_eq!(call(&mut ctx, cb_batch_job_control, &[info, inst, FLAG_CONTROL_RESET | FLAG_SIDEBAND_STREAM, 0, 0, BASE + 0x600, 32]), 0);
    let b = ctx.read_bytes(BASE + 0x600, 32).unwrap();
    assert_eq!(i32_at(&b, 0), 0);
    assert_eq!(u64::from_le_bytes(b[16..24].try_into().unwrap()), 0);

    // Codec info job.
    assert_eq!(call(&mut ctx, cb_batch_job_get_codec_info, &[info, inst, BASE + 0x700, 24]), 0);
    let b = ctx.read_bytes(BASE + 0x700, 24).unwrap();
    assert_eq!(u32_at(&b, 8), 512);

    // Resample info reports ratio 1; statistics are zeroed.
    assert_eq!(call(&mut ctx, cb_batch_job_get_resample_info, &[info, inst, BASE + 0x700]), 0);
    assert_eq!(ctx.read_f32(BASE + 0x708), Some(1.0));
    ctx.write_bytes(BASE + 0x700, &[0xCC; 64]);
    assert_eq!(call(&mut ctx, cb_batch_job_get_statistics, &[info, BASE + 0x700]), 0);
    let b = ctx.read_bytes(BASE + 0x700, 64).unwrap();
    assert!(b[..STATISTICS_RESULT_SIZE].iter().all(|&v| v == 0));
    assert_eq!(b[STATISTICS_RESULT_SIZE], 0xCC);
}

#[test]
fn job_sizes_follow_the_command_layout() {
    const BASE: u64 = 0x5CE0_3000_0000;
    let mut ctx = setup(BASE);
    ctx.set(CpuRegister::Rsp, BASE + 0x800);
    let info = BASE + 0x100;
    assert_eq!(call(&mut ctx, cb_batch_initialize, &[BASE + 0x1000, 0x1000, info]), 0);
    let mut expect = 0;
    assert_eq!(call(&mut ctx, cb_batch_job_set_resample_parameters, &[info, 0, 0, 0]), 0);
    expect += JOB_SET_RESAMPLE_PARAMETERS_SIZE;
    assert_eq!(ctx.read_u64(info + 8), Some(expect));
    assert_eq!(call(&mut ctx, cb_batch_job_get_statistics, &[info, 0]), 0);
    expect += JOB_GET_STATISTICS_SIZE;
    assert_eq!(ctx.read_u64(info + 8), Some(expect));
    // Split jobs: 32 + 16 per descriptor.
    let split = [info, 0, 0, 0, 1, 0, 1, 0, 0];
    assert_eq!(call(&mut ctx, cb_batch_job_run_split, &split), 0);
    expect += 32 + 16 * 2;
    assert_eq!(ctx.read_u64(info + 8), Some(expect));
    assert_eq!(call(&mut ctx, cb_batch_job_encode, &[info, 0, 0, 16, BASE + 0x2000, 32, BASE + 0x300]), 0);
    expect += JOB_RUN_SIZE;
    assert_eq!(ctx.read_u64(info + 8), Some(expect));
    let b = ctx.read_bytes(BASE + 0x300, 32).unwrap();
    assert_eq!((u32_at(&b, 8), u32_at(&b, 12), u32_at(&b, 24)), (16, 32, 1));
}

#[test]
fn parse_at9_config_data() {
    const BASE: u64 = 0x5CE0_4000_0000;
    let mut ctx = setup(BASE);
    ctx.write_bytes(BASE, &mono_config());
    assert_eq!(call(&mut ctx, cb_dec_at9_parse_config_data, &[BASE, BASE + 0x10]), 0);
    let b = ctx.read_bytes(BASE + 0x10, 20).unwrap();
    assert_eq!(
        (u32_at(&b, 0), u32_at(&b, 4), u32_at(&b, 8), u32_at(&b, 12), u32_at(&b, 16)),
        (1, 48000, 256, 1024, 512)
    );
    assert_eq!(call(&mut ctx, cb_dec_at9_parse_config_data, &[0, BASE + 0x10]), AJM_ERROR_INVALID_PARAMETER);

    // Third-order ambisonic stream (config header 0x30): 16 mono channels.
    ctx.write_bytes(BASE, &[0x30, 0x73, 0xC1, 0x7E]);
    assert_eq!(call(&mut ctx, cb_dec_at9_parse_config_data, &[BASE, BASE + 0x10]), 0);
    let b = ctx.read_bytes(BASE + 0x10, 20).unwrap();
    assert_eq!(
        (u32_at(&b, 0), u32_at(&b, 4), u32_at(&b, 8), u32_at(&b, 12), u32_at(&b, 16)),
        (16, 48000, 256, 1024, 6144)
    );
    // The validation bit must be clear.
    ctx.write_bytes(BASE, &[0x30, 0x73, 0xE1, 0x7E]);
    assert_eq!(call(&mut ctx, cb_dec_at9_parse_config_data, &[BASE, BASE + 0x10]), AJM_ERROR_INVALID_PARAMETER);
}

/// A single input buffer whose stated size is about 4 GiB (a remaining
/// size gone below zero) is read only as far as guest memory goes, as a
/// plain or a split job; outputs and several inputs keep the job limit
/// (reference FMOD Opus run case).
#[test]
fn oversized_single_input_is_read_in_place() {
    const BASE: u64 = 0x7C61_0000_0000;
    let ctx = setup(BASE);
    let stated = 0xFFFF_FE74u64;
    let packet = BASE + 0x8000;
    let inputs = collect_inputs(&ctx, false, packet, stated).unwrap();
    assert_eq!(inputs.len(), 1);
    assert_eq!((inputs[0].addr, inputs[0].len), (packet, 0x8000), "clamped to the mapped arena");
    assert!(collect_buffers(&ctx, false, packet, stated).is_none(), "an output keeps the limit");
    // A split job with one descriptor.
    let desc = BASE + 0x100;
    ctx.write_u64(desc, packet);
    ctx.write_u64(desc + 8, stated);
    let inputs = collect_inputs(&ctx, true, desc, 1).unwrap();
    assert_eq!((inputs[0].addr, inputs[0].len), (packet, 0x8000));
    // Several descriptors: the total is still checked.
    ctx.write_u64(desc + 16, packet);
    ctx.write_u64(desc + 24, 16);
    assert!(collect_inputs(&ctx, true, desc, 2).is_none());
    // Sizes within the limit are taken as stated.
    let inputs = collect_inputs(&ctx, false, packet, 0x20).unwrap();
    assert_eq!(inputs[0].len, 0x20);
}
