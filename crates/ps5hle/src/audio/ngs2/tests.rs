// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only
//
// Unit tests of the `ps5hle::audio::ngs2` module.

use super::*;
use crate::context::CpuRegister;

#[test]
fn ngs2_registered_nids_match_names() {
    for &(nid, name, _) in EXPORTS {
        assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
    }
    assert_eq!(EXPORTS.len(), 42);
}

const BASE: u64 = 0x5CE8_0000_0000;
const ARENA: u64 = 0x40000;

fn arena(base: u64) -> &'static ps5vm::VirtualMemory {
    let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
    vm.map(base, ARENA, ps5vm::MemProtect::RW).expect("map test arena");
    vm
}

fn call(ctx: &mut CpuContext, cb: HleCallback, args: &[u64]) -> HleResult {
    let regs = [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx, CpuRegister::R8];
    for (r, v) in regs.iter().zip(args) {
        ctx.set(*r, *v);
    }
    cb(ctx)
}

fn r(code: i32) -> HleResult {
    code as HleResult
}

fn param(id: u32, body: &[u8], next: i16) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&((8 + body.len()) as u16).to_le_bytes());
    p.extend_from_slice(&next.to_le_bytes());
    p.extend_from_slice(&id.to_le_bytes());
    p.extend_from_slice(body);
    p
}

fn le(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn ngs2_callbacks_render_graph() {
    let base = BASE;
    let vm = arena(base);
    let mut ctx = CpuContext::new(vm);
    let out = base;
    let scratch = base + 0x100;
    let sys_buf = base + 0x1000;
    let sampler_buf = base + 0x2000;
    let master_buf = base + 0x4000;
    let pcm_addr = base + 0x8000;
    let render_buf = base + 0x9000;
    let params = base + 0xA000;

    // System.
    assert_eq!(call(&mut ctx, cb_system_reset_option, &[scratch]), 0);
    assert_eq!(ctx.read_u32(scratch + 112), Some(256));
    assert_eq!(ctx.read_u32(scratch + 112 + 4), Some(48000));
    ctx.write_u32(scratch + 112, 4); // numGrainSamples
    let opt = scratch;
    assert_eq!(call(&mut ctx, cb_system_query_buffer_size, &[opt, out]), 0);
    assert_eq!(ctx.read_u64(out + 8), Some(SYSTEM_BUFFER_SIZE));
    let info = ContextBuffer { host_buffer: sys_buf, size: SYSTEM_BUFFER_SIZE, user_data: 7 };
    ctx.write_bytes(out + 0x200, &info.to_bytes());
    assert_eq!(call(&mut ctx, cb_system_create, &[opt, out + 0x200, 0]), r(ERR_INVALID_OUT_ADDRESS));
    assert_eq!(call(&mut ctx, cb_system_create, &[opt, out + 0x200, out]), 0);
    let system = ctx.read_u64(out).unwrap();
    assert_eq!(system, sys_buf);

    // Racks: a 2-voice sampler and a mastering rack (default options).
    let mut sampler_opt = RackOption::default_for(engine::RACK_SAMPLER).unwrap();
    sampler_opt.bytes[80..84].copy_from_slice(&2u32.to_le_bytes());
    ctx.write_bytes(scratch + 0x400, &sampler_opt.bytes[..216]);
    assert_eq!(call(&mut ctx, cb_rack_query_buffer_size, &[0x1000, scratch + 0x400, out]), 0);
    let sampler_size = ctx.read_u64(out + 8).unwrap();
    assert_eq!(sampler_size, RACK_HEADER_SIZE + 2 * VOICE_STRIDE);
    let info = ContextBuffer { host_buffer: sampler_buf, size: sampler_size, user_data: 0 };
    ctx.write_bytes(out + 0x200, &info.to_bytes());
    assert_eq!(call(&mut ctx, cb_rack_create, &[1, 0x1000, scratch + 0x400, out + 0x200, out]), r(ERR_INVALID_SYSTEM_HANDLE));
    assert_eq!(call(&mut ctx, cb_rack_create, &[system, 0x1000, scratch + 0x400, out + 0x200, out]), 0);
    let sampler = ctx.read_u64(out).unwrap();
    let info = ContextBuffer { host_buffer: master_buf, size: 0x1000, user_data: 0 };
    ctx.write_bytes(out + 0x200, &info.to_bytes());
    assert_eq!(call(&mut ctx, cb_rack_create, &[system, 0x3000, 0, out + 0x200, out]), 0);
    let master_rack = ctx.read_u64(out).unwrap();

    assert_eq!(call(&mut ctx, cb_rack_get_voice_handle, &[sampler, 1, out]), 0);
    let voice = ctx.read_u64(out).unwrap();
    assert_eq!(voice, sampler_buf + RACK_HEADER_SIZE + VOICE_STRIDE);
    // Rack handle, then output, then index; nothing is written on an error.
    for id in [2, 3, u32::MAX as u64] {
        ctx.write_u64(out, 0x5A5A);
        assert_eq!(call(&mut ctx, cb_rack_get_voice_handle, &[sampler, id, out]), r(engine::ERR_INVALID_VOICE_INDEX));
        assert_eq!(ctx.read_u64(out), Some(0x5A5A));
    }
    assert_eq!(call(&mut ctx, cb_rack_get_voice_handle, &[sampler, 2, 0]), r(ERR_INVALID_OUT_ADDRESS));
    assert_eq!(call(&mut ctx, cb_rack_get_voice_handle, &[0x1234, 0, 0]), r(ERR_INVALID_RACK_HANDLE));
    assert_eq!(ctx.read_u64(out), Some(0x5A5A));
    assert_eq!(call(&mut ctx, cb_rack_get_voice_handle, &[master_rack, 0, out]), 0);
    let master = ctx.read_u64(out).unwrap();

    // Mastering voice: 2 channels, output 0, play.
    let mut p = param(0x3000_0000, &le(&[2]), 12);
    p.extend(param(6, &le(&[1]), 0));
    ctx.write_bytes(params, &p);
    assert_eq!(call(&mut ctx, cb_voice_control, &[master, params]), 0);

    // Sampler voice: PCM16 mono setup, one block, patch port 0, play.
    ctx.write_bytes(pcm_addr, &[8192i16, 16384, -8192].iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>());
    let block = engine::WaveformBlock { data_size: 6, num_samples: 3, user_data: 5, ..Default::default() };
    ctx.write_bytes(params + 0x400, &block.to_bytes());
    let mut setup = le(&[0x12, 1, 48000, 0, 0, 0]);
    setup.extend_from_slice(&[0; 8]);
    let mut p = param(0x1000_0000, &setup, 40);
    let mut blocks = pcm_addr.to_le_bytes().to_vec();
    blocks.extend(le(&[1, 1])); // flags: more blocks may follow
    blocks.extend((params + 0x400).to_le_bytes());
    p.extend(param(0x1000_0001, &blocks, 32));
    let mut patch = le(&[0, 0]);
    patch.extend(master.to_le_bytes());
    p.extend(param(5, &patch, 24));
    p.extend(param(6, &le(&[1]), 0));
    ctx.write_bytes(params, &p);
    assert_eq!(call(&mut ctx, cb_voice_control, &[voice, params]), 0);
    assert_eq!(call(&mut ctx, cb_voice_control, &[1, params]), r(ERR_INVALID_VOICE_HANDLE));
    assert_eq!(call(&mut ctx, cb_voice_get_state_flags, &[voice, out]), 0);
    assert_eq!(ctx.read_u32(out), Some(1));

    // Render a stereo float grain of 4 frames into a dirty 64-byte buffer.
    ctx.write_bytes(render_buf, &[0xAB; 64]);
    let mut rb = render_buf.to_le_bytes().to_vec();
    rb.extend(64u64.to_le_bytes());
    rb.extend(le(&[0x18, 2]));
    ctx.write_bytes(scratch + 0x800, &rb);
    assert_eq!(call(&mut ctx, cb_system_render, &[system, 0, 1]), r(ERR_INVALID_ARGUMENT));
    assert_eq!(call(&mut ctx, cb_rack_lock, &[sampler]), 0);
    assert_eq!(call(&mut ctx, cb_system_render, &[system, scratch + 0x800, 1]), 0);
    assert_eq!(call(&mut ctx, cb_rack_unlock, &[sampler]), 0);
    let s: Vec<f32> = (0..8).map(|i| ctx.read_f32(render_buf + i * 4).unwrap()).collect();
    // Mono into stereo through the default channel map: both channels at -3 dB.
    let h = std::f32::consts::FRAC_1_SQRT_2;
    assert_eq!(s, [0.25 * h, 0.25 * h, 0.5 * h, 0.5 * h, -0.25 * h, -0.25 * h, 0.0, 0.0]);
    // One grain (4 stereo f32 frames, 32 bytes) is written; the rest of the
    // buffer is the title's.
    assert_eq!(ctx.read_bytes(render_buf + 32, 32).unwrap(), vec![0xABu8; 32]);
    assert_eq!(call(&mut ctx, cb_voice_get_state_flags, &[voice, out]), 0);
    // The stream played its only block and waits for more: INUSE | PLAYING | EMPTY.
    assert_eq!(ctx.read_u32(out), Some(3 | engine::VOICE_STATE_FLAG_EMPTY));

    // Sampler state image.
    ctx.write_bytes(out, &[0xFF; 64]);
    assert_eq!(call(&mut ctx, cb_voice_get_state, &[voice, out, 56]), 0);
    let st = ctx.read_bytes(out, 64).unwrap();
    assert_eq!(u64::from_le_bytes(st[24..32].try_into().unwrap()), 3);
    assert_eq!(u64::from_le_bytes(st[32..40].try_into().unwrap()), 6);
    assert_eq!(u64::from_le_bytes(st[48..56].try_into().unwrap()), pcm_addr + 6);
    assert_eq!(st[56..], [0xFF; 8]);
    // Error precedence: the voice, the size (at least 8), the address, the
    // exact size of the rack type (a sampler state is 56 bytes).
    assert_eq!(call(&mut ctx, cb_voice_get_state, &[voice, 0, 0]), r(ERR_INVALID_OUT_SIZE));
    assert_eq!(call(&mut ctx, cb_voice_get_state, &[voice, 0, 7]), r(ERR_INVALID_OUT_SIZE));
    assert_eq!(call(&mut ctx, cb_voice_get_state, &[voice, 0, 8]), r(engine::ERR_INVALID_OUT_ADDRESS));
    assert_eq!(call(&mut ctx, cb_voice_get_state, &[voice, 0, 56]), r(engine::ERR_INVALID_OUT_ADDRESS));
    assert_eq!(call(&mut ctx, cb_voice_get_state, &[voice, out, 64]), r(ERR_INVALID_OUT_SIZE));
    assert_eq!(call(&mut ctx, cb_voice_get_state, &[0x1234, 0, 0]), r(ERR_INVALID_VOICE_HANDLE));
    assert_eq!(call(&mut ctx, cb_voice_get_state_flags, &[voice, 0]), r(engine::ERR_INVALID_OUT_ADDRESS));
    // A null parameter list comes before the handle.
    assert_eq!(call(&mut ctx, cb_voice_control, &[0x1234, 0]), r(ERR_NULL_PARAM_LIST));
    assert_eq!(call(&mut ctx, cb_voice_control, &[voice, 0]), r(ERR_NULL_PARAM_LIST));
    assert_eq!(call(&mut ctx, cb_voice_run_commands, &[0x1234, 0, 1]), r(ERR_INVALID_VOICE_HANDLE));
    assert_eq!(call(&mut ctx, cb_voice_run_commands, &[voice, 0, 1]), r(engine::ERR_INVALID_ARGUMENT));

    // System info.
    assert_eq!(call(&mut ctx, cb_system_get_info, &[system, out, 10]), r(ERR_INVALID_OUT_SIZE));
    assert_eq!(call(&mut ctx, cb_system_get_info, &[system + 8, out, 184]), r(ERR_INVALID_SYSTEM_HANDLE));
    assert_eq!(call(&mut ctx, cb_system_get_info, &[system, out, 184]), 0);
    assert_eq!(ctx.read_u64(out + 64), Some(system));
    assert_eq!(ctx.read_u64(out + 128), Some(7));
    assert_eq!(ctx.read_u32(out + 152), Some(2));
    assert_eq!(ctx.read_u64(out + 168), Some(1));
    assert_eq!(ctx.read_u32(out + 180), Some(4));

    // Teardown: destroying the system takes its racks with it.
    assert_eq!(call(&mut ctx, cb_rack_destroy, &[master_rack, out]), 0);
    assert_eq!(ctx.read_u64(out), Some(master_buf));
    assert_eq!(call(&mut ctx, cb_rack_destroy, &[master_rack, 0]), r(ERR_INVALID_RACK_HANDLE));
    assert_eq!(call(&mut ctx, cb_system_destroy, &[system, out]), 0);
    assert_eq!(ctx.read_u64(out), Some(sys_buf));
    assert_eq!(call(&mut ctx, cb_system_destroy, &[system, 0]), r(ERR_INVALID_SYSTEM_HANDLE));
    assert_eq!(call(&mut ctx, cb_voice_control, &[voice, params]), r(ERR_INVALID_VOICE_HANDLE));
}

/// Only the default reverb rack size is known.
#[test]
fn ngs2_reverb_rack_size() {
    let mut option = RackOption::default_for(engine::RACK_REVERB).unwrap().bytes[..184].to_vec();
    assert_eq!(check_reverb_size(engine::RACK_REVERB, &option), Ok(()));
    for size in [0u32, 2] {
        option[180..184].copy_from_slice(&size.to_le_bytes());
        assert_eq!(check_reverb_size(engine::RACK_REVERB, &option), Err(ERR_INVALID_ARGUMENT), "reverb size {size}");
        assert_eq!(check_reverb_size(engine::RACK_SUBMIXER, &option), Ok(()));
    }
}

#[test]
fn ngs2_callbacks_parse_waveform_data() {
    let base = BASE + 0x100_0000;
    let vm = arena(base);
    let mut ctx = CpuContext::new(vm);
    let data = base + 0x1000;
    let out = base + 0x100;
    let wav = waveform::tests::riff(&waveform::tests::pcm_fmt(1, 1, 48000, 16), &[], &[0u8; 64]);
    assert!(ctx.write_bytes(data, &wav));
    assert_eq!(call(&mut ctx, cb_parse_waveform_data, &[data, wav.len() as u64, 0]), r(ERR_INVALID_OUT_ADDRESS));
    assert_eq!(call(&mut ctx, cb_parse_waveform_data, &[0, 1, out]), r(ERR_INVALID_WAVEFORM_ADDRESS));
    assert_eq!(call(&mut ctx, cb_parse_waveform_data, &[data, wav.len() as u64, out]), 0);
    assert_eq!(ctx.read_u32(out), Some(waveform::WAVEFORM_TYPE_PCM_I16L));
    assert_eq!(ctx.read_u32(out + 0x28), Some(32));
    // A caller-declared size larger than the header copy extends the data.
    let total = 0x2000u64 + 44;
    assert_eq!(call(&mut ctx, cb_parse_waveform_data, &[data, total, out]), 0);
    assert_eq!(ctx.read_u32(out + 0x1C), Some(0x2000));
    assert_eq!(ctx.read_u32(out + 0x28), Some(0x1000));
    // Unknown container vs known container with an unsupported format.
    assert!(ctx.write_bytes(data, b"OggS\0\0\0\0\0\0\0\0"));
    assert_eq!(call(&mut ctx, cb_parse_waveform_data, &[data, 12, out]), r(ERR_UNKNOWN_WAVEFORM_FORMAT));
    assert_eq!(call(&mut ctx, cb_parse_waveform_data, &[data, 8, out]), r(ERR_INVALID_WAVEFORM_DATA));
    let bad = waveform::tests::riff(&waveform::tests::pcm_fmt(2, 1, 48000, 4), &[], &[0u8; 8]);
    assert!(ctx.write_bytes(data, &bad));
    assert_eq!(call(&mut ctx, cb_parse_waveform_data, &[data, bad.len() as u64, out]), r(ERR_INVALID_WAVEFORM_FORMAT));
    assert_eq!(ctx.read_bytes(out, 8).unwrap(), vec![0u8; 8], "info is cleared on failure");
    assert_eq!(call(&mut ctx, cb_parse_waveform_user, &[0, 0, out]), r(ERR_INVALID_WAVEFORM_DATA));
}

#[test]
fn ngs2_callbacks_calc_block_pan_geom() {
    let base = BASE + 0x200_0000;
    let vm = arena(base);
    let mut ctx = CpuContext::new(vm);
    let (a, out) = (base + 0x100, base + 0x1000);

    // PCM16 stereo: 4-byte frames.
    ctx.write_bytes(a, &le(&[0x12, 2, 48000, 0, 0, 0]));
    assert_eq!(call(&mut ctx, cb_calc_waveform_block, &[a, 10, 5, out]), 0);
    let b = engine::WaveformBlock::parse(&ctx.read_bytes(out, 40).unwrap());
    assert_eq!((b.data_offset, b.data_size, b.num_skip_samples, b.num_samples), (40, 20, 0, 5));
    ctx.write_bytes(a, &le(&[0x77, 2, 48000, 0, 0, 0]));
    assert_eq!(call(&mut ctx, cb_calc_waveform_block, &[a, 0, 1, out]), r(ERR_UNKNOWN_WAVEFORM_FORMAT));
    ctx.write_bytes(a, &le(&[0x12, 0, 48000, 0, 0, 0]));
    assert_eq!(call(&mut ctx, cb_calc_waveform_block, &[a, 0, 1, out]), r(ERR_INVALID_WAVEFORM_FORMAT));

    // Pan: speakers -45 / 45 degrees, wrapped into one turn and stored in
    // the stereo input order (right first).
    TEST_XMM.with(|x| x.set([360.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    ctx.write_bytes(a, &[(-45.0f32).to_le_bytes(), 45.0f32.to_le_bytes()].concat());
    assert_eq!(call(&mut ctx, cb_pan_init, &[out, a, 2]), 0);
    assert_eq!(ctx.read_f32(out), Some(45.0));
    assert_eq!(ctx.read_f32(out + 4), Some(315.0));
    assert_eq!(ctx.read_f32(out + 8), Some(405.0));
    assert_eq!(ctx.read_f32(out + 32), Some(360.0));
    assert_eq!(ctx.read_u32(out + 36), Some(2));
    assert_eq!(call(&mut ctx, cb_pan_init, &[0, a, 2]), r(ERR_INVALID_OUT_ADDRESS));
    assert_eq!(call(&mut ctx, cb_pan_init, &[out, a, 3]), r(ERR_INVALID_PAN_SPEAKER));
    // Two sources, destination-major: (10, 890) degrees on the console.
    let params: Vec<u8> = [[10.0f32, 1.0, 1.0, 0.0], [890.0, 1.0, 1.0, 0.0]]
        .iter()
        .flatten()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    ctx.write_bytes(a + 0x40, &params);
    ctx.write_bytes(out + 0x100, &[0xEE; 32]);
    assert_eq!(call(&mut ctx, cb_pan_get_volume_matrix, &[out, a + 0x40, 2, 2, out + 0x100]), 0);
    let m: Vec<f32> = (0..4).map(|i| ctx.read_f32(out + 0x100 + i * 4).unwrap()).collect();
    let expected = [0.57357651f32, 0.664795876, 0.819152057, 0.747025073];
    assert!(m.iter().zip(expected).all(|(x, y)| (x - y).abs() < 1e-6), "{m:?}");
    assert_eq!(ctx.read_bytes(out + 0x110, 4).unwrap(), vec![0xEE; 4], "wrote past the matrix");
    assert_eq!(call(&mut ctx, cb_pan_get_volume_matrix, &[out, a + 0x40, 2, 2, 0]), r(ERR_INVALID_OUT_ADDRESS));
    assert_eq!(call(&mut ctx, cb_pan_get_volume_matrix, &[out, a + 0x40, 2, 3, out + 0x100]), r(ERR_INVALID_PAN_MATRIX_FORMAT));
    assert_eq!(call(&mut ctx, cb_pan_get_volume_matrix, &[out, 0, 0, 2, out + 0x100]), r(ERR_INVALID_PAN_PARAM));
    assert_eq!(call(&mut ctx, cb_pan_get_volume_matrix, &[0, a + 0x40, 1, 2, out + 0x100]), r(ERR_INVALID_PAN_WORK));
    assert_eq!(call(&mut ctx, cb_pan_get_volume_matrix, &[out, a + 0x40, 9, 2, out + 0x100]), r(ERR_INVALID_NUM_CHANNELS));
    assert_eq!(call(&mut ctx, cb_pan_get_volume_matrix, &[out, a + 0x40, 0, 2, out + 0x100]), 0);

    // Geometry.
    assert_eq!(call(&mut ctx, cb_geom_reset_listener_param, &[a]), 0);
    assert_eq!(ctx.read_f32(a + 20), Some(1.0));
    assert_eq!(ctx.read_f32(a + 48), Some(343.0));
    assert_eq!(call(&mut ctx, cb_geom_calc_listener, &[a, out, 1]), 0);
    assert_eq!(ctx.read_f32(out), Some(1.0));
    assert_eq!(ctx.read_f32(out + 60), Some(1.0));
    assert_eq!(ctx.read_f32(out + 76), Some(343.0));
    assert_eq!(ctx.read_u32(out + 80), Some(1));
    let src = base + 0x400;
    assert_eq!(call(&mut ctx, cb_geom_reset_source_param, &[src]), 0);
    assert_eq!(ctx.read_u32(src + 96), Some(2));
    ctx.write_bytes(src, &[1.0f32.to_le_bytes(), 2.0f32.to_le_bytes(), 3.0f32.to_le_bytes()].concat());
    ctx.write_bytes(src + 80, &0.5f32.to_le_bytes());
    let attr = base + 0x2000;
    assert_eq!(call(&mut ctx, cb_geom_apply, &[out, src, attr]), 0);
    assert_eq!(ctx.read_f32(attr), Some(1.0));
    assert_eq!(ctx.read_f32(attr + 4), Some(0.5));
    assert_eq!(ctx.read_f32(attr + 4 + 9 * 4), Some(0.5));
    assert_eq!(ctx.read_f32(attr + 8), Some(0.0));
    assert_eq!(ctx.read_f32(attr + 260 + 8), Some(3.0));
    assert_eq!(ctx.read_f32(attr + 272), Some(0.5));
}

/// sceNgs2SystemLock: other calls do not wait for it, a render on another
/// thread writes one grain of silence instead of waiting, a lock on another
/// thread waits for the last unlock, and any thread may unlock.
#[test]
fn ngs2_system_lock_across_threads() {
    let base = BASE + 0x300_0000;
    let vm = arena(base);
    let mut ctx = CpuContext::new(vm);
    let (scratch, out, sys_buf, render_buf) = (base + 0x100, base, base + 0x1000, base + 0x9000);
    assert_eq!(call(&mut ctx, cb_system_reset_option, &[scratch]), 0);
    ctx.write_u32(scratch + 112, 4); // numGrainSamples
    let info = ContextBuffer { host_buffer: sys_buf, size: SYSTEM_BUFFER_SIZE, user_data: 0 };
    ctx.write_bytes(out + 0x200, &info.to_bytes());
    assert_eq!(call(&mut ctx, cb_system_create, &[scratch, out + 0x200, out]), 0);
    let system = ctx.read_u64(out).unwrap();
    assert_eq!(call(&mut ctx, cb_system_lock, &[0x1234]), r(ERR_INVALID_SYSTEM_HANDLE));
    assert_eq!(call(&mut ctx, cb_system_unlock, &[0x1234]), r(ERR_INVALID_SYSTEM_HANDLE));
    assert_eq!(call(&mut ctx, cb_system_unlock, &[system]), 0, "an unlocked system stays unlocked");

    // A mono f32 buffer of three grains; one grain is 16 bytes.
    let mut rb = render_buf.to_le_bytes().to_vec();
    rb.extend(48u64.to_le_bytes());
    rb.extend(le(&[0x18, 1]));
    ctx.write_bytes(scratch + 0x800, &rb);
    let infos = scratch + 0x800;

    assert_eq!(call(&mut ctx, cb_system_lock, &[system]), 0);
    assert_eq!(call(&mut ctx, cb_system_lock, &[system]), 0, "recursive");
    let other = std::thread::spawn(move || {
        let mut ctx = CpuContext::new(vm);
        ctx.write_bytes(render_buf, &[0xAB; 48]);
        assert_eq!(call(&mut ctx, cb_system_get_info, &[system, base + 0x400, 184]), 0, "GetInfo does not wait");
        assert_eq!(call(&mut ctx, cb_system_render, &[system, infos, 1]), 0, "the render does not wait");
        assert_eq!(ctx.read_bytes(render_buf, 16).unwrap(), vec![0; 16]);
        assert_eq!(ctx.read_bytes(render_buf + 16, 32).unwrap(), vec![0xAB; 32]);
        assert_eq!(ctx.read_u64(base + 0x400 + 168), Some(0), "nothing rendered");
    });
    other.join().unwrap();

    let locked = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = locked.clone();
    let waiter = std::thread::spawn(move || {
        let mut ctx = CpuContext::new(vm);
        assert_eq!(call(&mut ctx, cb_system_lock, &[system]), 0);
        flag.store(true, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(call(&mut ctx, cb_system_unlock, &[system]), 0);
    });
    std::thread::sleep(std::time::Duration::from_millis(30));
    assert!(!locked.load(std::sync::atomic::Ordering::SeqCst), "the lock waits");
    assert_eq!(call(&mut ctx, cb_system_unlock, &[system]), 0);
    std::thread::sleep(std::time::Duration::from_millis(30));
    assert!(!locked.load(std::sync::atomic::Ordering::SeqCst), "still held once");
    assert_eq!(call(&mut ctx, cb_system_unlock, &[system]), 0);
    waiter.join().unwrap();
    assert!(locked.load(std::sync::atomic::Ordering::SeqCst));

    // Locked here, unlocked on another thread, rendered on a third.
    assert_eq!(call(&mut ctx, cb_system_lock, &[system]), 0);
    std::thread::spawn(move || {
        let mut ctx = CpuContext::new(vm);
        assert_eq!(call(&mut ctx, cb_system_unlock, &[system]), 0);
    })
    .join()
    .unwrap();
    std::thread::spawn(move || {
        let mut ctx = CpuContext::new(vm);
        assert_eq!(call(&mut ctx, cb_system_render, &[system, infos, 1]), 0);
        assert_eq!(call(&mut ctx, cb_system_get_info, &[system, base + 0x400, 184]), 0);
        assert_eq!(ctx.read_u64(base + 0x400 + 168), Some(1), "rendered once");
    })
    .join()
    .unwrap();
    assert_eq!(call(&mut ctx, cb_system_destroy, &[system, 0]), 0);
}

/// RackGetInfo, System user data, VoiceGetPortInfo and VoiceQueryInfo.
#[test]
fn ngs2_info_queries() {
    let base = BASE + 0x400_0000;
    let vm = arena(base);
    let mut ctx = CpuContext::new(vm);
    let (out, scratch, params) = (base, base + 0x100, base + 0xA000);
    let (sys_buf, sampler_buf, submix_buf, master_buf) = (base + 0x1000, base + 0x2000, base + 0x6000, base + 0x8000);

    assert_eq!(call(&mut ctx, cb_system_reset_option, &[scratch]), 0);
    let info = ContextBuffer { host_buffer: sys_buf, size: SYSTEM_BUFFER_SIZE, user_data: 0 };
    ctx.write_bytes(out + 0x200, &info.to_bytes());
    assert_eq!(call(&mut ctx, cb_system_create, &[scratch, out + 0x200, out]), 0);
    let system = ctx.read_u64(out).unwrap();

    // User data.
    ctx.write_u64(out, 1);
    assert_eq!(call(&mut ctx, cb_system_set_user_data, &[0x1234, 5]), r(ERR_INVALID_SYSTEM_HANDLE));
    assert_eq!(call(&mut ctx, cb_system_get_user_data, &[0x1234, out]), r(ERR_INVALID_SYSTEM_HANDLE));
    assert_eq!(ctx.read_u64(out), Some(1));
    assert_eq!(call(&mut ctx, cb_system_get_user_data, &[system, out]), 0);
    assert_eq!(ctx.read_u64(out), Some(0));
    assert_eq!(call(&mut ctx, cb_system_set_user_data, &[system, 0xfedc_ba98_7654_3210]), 0);
    assert_eq!(call(&mut ctx, cb_system_get_user_data, &[system, out]), 0);
    assert_eq!(ctx.read_u64(out), Some(0xfedc_ba98_7654_3210));
    assert_eq!(call(&mut ctx, cb_system_get_user_data, &[system, 0]), r(ERR_INVALID_ARGUMENT));

    // A default sampler rack.
    let info = ContextBuffer { host_buffer: sampler_buf, size: 0x4000, user_data: 0 };
    ctx.write_bytes(out + 0x200, &info.to_bytes());
    assert_eq!(call(&mut ctx, cb_rack_create, &[system, 0x1000, 0, out + 0x200, out]), 0);
    let sampler_rack = ctx.read_u64(out).unwrap();
    // Rack user data, and the owner of a voice.
    assert_eq!(call(&mut ctx, cb_rack_set_user_data, &[0x1234, 5]), r(ERR_INVALID_RACK_HANDLE));
    assert_eq!(call(&mut ctx, cb_rack_set_user_data, &[sampler_rack, 0x55AA]), 0);
    assert_eq!(call(&mut ctx, cb_rack_get_user_data, &[sampler_rack, out + 0x300]), 0);
    assert_eq!(ctx.read_u64(out + 0x300), Some(0x55AA));
    assert_eq!(call(&mut ctx, cb_rack_get_user_data, &[sampler_rack, 0]), r(ERR_INVALID_OUT_ADDRESS));
    let voice3 = sampler_rack + RACK_HEADER_SIZE + 3 * VOICE_STRIDE;
    assert_eq!(call(&mut ctx, cb_voice_get_owner, &[voice3, out + 0x300, out + 0x308]), 0);
    assert_eq!((ctx.read_u64(out + 0x300), ctx.read_u32(out + 0x308)), (Some(sampler_rack), Some(3)));
    assert_eq!(call(&mut ctx, cb_voice_get_owner, &[voice3 + 1, out + 0x300, out + 0x308]), r(ERR_INVALID_VOICE_HANDLE));
    assert_eq!(call(&mut ctx, cb_voice_get_owner, &[voice3, 0, out + 0x308]), r(ERR_INVALID_OUT_ADDRESS));
    // Report handlers: kept, never called.
    assert_eq!(call(&mut ctx, cb_report_register_handler, &[1, 0x4000, 7, 0]), r(ERR_INVALID_OUT_ADDRESS));
    assert_eq!(call(&mut ctx, cb_report_register_handler, &[1, 0x4000, 7, out + 0x300]), 0);
    let report = ctx.read_u64(out + 0x300).unwrap();
    assert_ne!(report, 0);
    assert_eq!(call(&mut ctx, cb_report_unregister_handler, &[report]), 0);
    assert_eq!(call(&mut ctx, cb_report_unregister_handler, &[report]), r(ERR_INVALID_ARGUMENT));
    let info_out = base + 0x400;
    assert_eq!(call(&mut ctx, cb_rack_get_info, &[0x1234, info_out, 216]), r(ERR_INVALID_RACK_HANDLE));
    assert_eq!(call(&mut ctx, cb_rack_get_info, &[sampler_rack, 0, 216]), r(ERR_INVALID_OUT_ADDRESS));
    assert_eq!(call(&mut ctx, cb_rack_get_info, &[sampler_rack, info_out, 215]), r(ERR_INVALID_OUT_SIZE));
    ctx.write_bytes(info_out, &[0xEE; 216]);
    assert_eq!(call(&mut ctx, cb_rack_get_info, &[sampler_rack, info_out, 216]), 0);
    let ri = ctx.read_bytes(info_out, 216).unwrap();
    let u32_at = |b: &[u8], o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let u64_at = |b: &[u8], o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    assert_eq!(ri[0], 0, "unnamed rack");
    assert_eq!(u64_at(&ri, 64), sampler_rack);
    assert_eq!(u64_at(&ri, 72), sampler_buf, "buffer info");
    assert_eq!(u64_at(&ri, 136), system);
    assert_eq!((u32_at(&ri, 144), u32_at(&ri, 148)), (1, 0x1000), "type and rack id");
    let sampler_uid = u32_at(&ri, 152);
    assert_ne!(sampler_uid, 0);
    assert_eq!((u32_at(&ri, 156), u32_at(&ri, 160)), (64, 512), "grain range");
    assert_eq!((u32_at(&ri, 164), u32_at(&ri, 168), u32_at(&ri, 172)), (256, 256, 0), "voices, channel works, inputs");
    assert_eq!((u32_at(&ri, 176), u32_at(&ri, 180), u32_at(&ri, 184)), (1, 8, 1), "matrices, ports, state flags");
    assert_eq!((u64_at(&ri, 200), u32_at(&ri, 208), u32_at(&ri, 212)), (0, 0, 0));

    // A named submixer rack: 63 name bytes and a terminator.
    let mut opt = RackOption::default_for(engine::RACK_SUBMIXER).unwrap();
    opt.bytes[8..72].fill(b'n');
    opt.bytes[76..80].copy_from_slice(&256u32.to_le_bytes());
    opt.bytes[80..84].copy_from_slice(&2u32.to_le_bytes());
    opt.bytes[92..96].copy_from_slice(&2u32.to_le_bytes());
    opt.bytes[176..180].copy_from_slice(&2u32.to_le_bytes());
    opt.bytes[188..192].copy_from_slice(&3u32.to_le_bytes());
    ctx.write_bytes(scratch + 0x400, &opt.bytes[..200]);
    let info = ContextBuffer { host_buffer: submix_buf, size: 0x1000, user_data: 0 };
    ctx.write_bytes(out + 0x200, &info.to_bytes());
    assert_eq!(call(&mut ctx, cb_rack_create, &[system, 0x2000, scratch + 0x400, out + 0x200, out]), 0);
    let submix_rack = ctx.read_u64(out).unwrap();
    assert_eq!(call(&mut ctx, cb_rack_get_voice_handle, &[submix_rack, 0, out]), 0);
    let submix = ctx.read_u64(out).unwrap();
    let mut p = param(0x2000_0000, &le(&[2, 0]), 16);
    p.extend(param(6, &le(&[1]), 0));
    ctx.write_bytes(params, &p);
    assert_eq!(call(&mut ctx, cb_voice_control, &[submix, params]), 0);
    assert_eq!(call(&mut ctx, cb_rack_get_info, &[submix_rack, info_out, 216]), 0);
    let ri = ctx.read_bytes(info_out, 216).unwrap();
    assert!(ri[..63].iter().all(|&c| c == b'n') && ri[63] == 0, "rack name");
    assert_eq!((u32_at(&ri, 144), u32_at(&ri, 148)), (2, 0x2000));
    assert_ne!(u32_at(&ri, 152), sampler_uid);
    assert_eq!((u32_at(&ri, 160), u32_at(&ri, 164), u32_at(&ri, 172), u32_at(&ri, 180)), (256, 2, 3, 2));
    assert_eq!(u32_at(&ri, 168), 0, "submixers have no channel works");
    assert_eq!(u32_at(&ri, 208), 1, "one active voice");

    // Voice channels.
    let ch_out = base + 0x600;
    let channels_id = VOICE_INFO_CHANNELS as u64;
    assert_eq!(call(&mut ctx, cb_voice_query_info, &[submix, channels_id, 0, 8]), r(ERR_INVALID_OUT_ADDRESS));
    assert_eq!(call(&mut ctx, cb_voice_query_info, &[submix, channels_id, ch_out, 7]), r(ERR_INVALID_OUT_SIZE));
    assert_eq!(call(&mut ctx, cb_voice_query_info, &[0x1234, channels_id, ch_out, 8]), r(ERR_INVALID_VOICE_HANDLE));
    ctx.write_bytes(ch_out, &[0xEE; 8]);
    assert_eq!(call(&mut ctx, cb_voice_query_info, &[submix, channels_id, ch_out, 8]), 0);
    assert_eq!(ctx.read_bytes(ch_out, 8).unwrap(), le(&[2, 0]));
    assert_eq!(call(&mut ctx, cb_voice_query_info, &[submix, 0x4000, ch_out, 8]), r(ERR_NOT_IMPLEMENTED));

    // Port info of a sampler voice, patched to a rack handle.
    let info = ContextBuffer { host_buffer: master_buf, size: 0x1000, user_data: 0 };
    ctx.write_bytes(out + 0x200, &info.to_bytes());
    assert_eq!(call(&mut ctx, cb_rack_create, &[system, 0x3000, 0, out + 0x200, out]), 0);
    let master_rack = ctx.read_u64(out).unwrap();
    assert_eq!(call(&mut ctx, cb_rack_get_voice_handle, &[master_rack, 0, out]), 0);
    let master = ctx.read_u64(out).unwrap();
    assert_eq!(call(&mut ctx, cb_rack_get_voice_handle, &[sampler_rack, 0, out]), 0);
    let sampler = ctx.read_u64(out).unwrap();
    let port_out = base + 0x700;
    assert_eq!(call(&mut ctx, cb_voice_get_port_info, &[sampler, 0, 0, 24]), r(ERR_INVALID_OUT_ADDRESS));
    assert_eq!(call(&mut ctx, cb_voice_get_port_info, &[sampler, 0, port_out, 23]), r(ERR_INVALID_OUT_SIZE));
    assert_eq!(call(&mut ctx, cb_voice_get_port_info, &[0x1234, 0, port_out, 24]), r(ERR_INVALID_VOICE_HANDLE));
    assert_eq!(call(&mut ctx, cb_voice_get_port_info, &[sampler, 8, port_out, 24]), r(ERR_INVALID_ARGUMENT));
    let mut patch = le(&[0, 0]);
    patch.extend(master_rack.to_le_bytes());
    let mut p = param(5, &patch, 24);
    p.extend(param(2, &le(&[0, 0.5f32.to_bits()]), 16));
    p.extend(param(3, &le(&[0, 0]), 0));
    ctx.write_bytes(params, &p);
    assert_eq!(call(&mut ctx, cb_voice_control, &[sampler, params]), 0);
    ctx.write_bytes(port_out, &[0xEE; 24]);
    assert_eq!(call(&mut ctx, cb_voice_get_port_info, &[sampler, 0, port_out, 24]), 0);
    let pi = ctx.read_bytes(port_out, 24).unwrap();
    assert_eq!(u32_at(&pi, 0), 0, "matrix id");
    assert_eq!(f32::from_bits(u32_at(&pi, 4)), 0.5);
    assert_eq!((u32_at(&pi, 8), u32_at(&pi, 12)), (0, 0));
    assert_eq!(u64_at(&pi, 16), master, "a rack handle patches into its first voice");
    assert_eq!(call(&mut ctx, cb_voice_get_port_info, &[sampler, 1, port_out, 24]), 0);
    let pi = ctx.read_bytes(port_out, 24).unwrap();
    assert_eq!((u32_at(&pi, 0) as i32, f32::from_bits(u32_at(&pi, 4)), u64_at(&pi, 16)), (-1, 1.0, 0));

    assert_eq!(call(&mut ctx, cb_system_destroy, &[system, 0]), 0);
    assert_eq!(call(&mut ctx, cb_system_get_user_data, &[system, out]), r(ERR_INVALID_SYSTEM_HANDLE));
}
