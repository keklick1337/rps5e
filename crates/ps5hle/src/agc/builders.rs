// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceAgc command-buffer packet builders (`sceAgcDcb*`, `sceAgcAcb*`,
//! `sceAgcCb*`) and their `*GetSize` companions.
//!
//! Every builder returns the guest address of the (first) packet it wrote, or
//! 0 on failure. Every `*GetSize` returns the byte size the matching builder
//! emits for the same size-relevant arguments.
//!
//! Encodings for exports without a known native layout were derived from
//! the PM4 field layout the interpreter decodes; each one is marked
//! "Inferred" below.

use super::pm4::{nop, op};
use super::{alloc_dwords, arg, emit, hi, lo, mem, pm4, prepare_dwords, ptr, size_bytes, stack_arg, Export, LIB_AGC};
use crate::context::CpuContext;
use crate::registry::HleResult;

// ---------------------------------------------------------------------------
// Encoding helpers shared with the tests
// ---------------------------------------------------------------------------

/// WAIT_REG_MEM poll interval field.
fn encode_wait_poll(cycles: u32) -> u32 {
    (cycles >> 4).min(0xFFFF)
}

fn encode_wait32_control(cmp: u32, operation: u32, cache: u32) -> u32 {
    0x10 | (cmp & 7) | ((operation & 3) << 8) | ((operation & 0xC) << 4) | ((cache & 3) << 25)
}

fn encode_wait64_control(cmp: u32, operation: u32, cache: u32) -> u32 {
    0x10 | (cmp & 7) | ((operation & 1) << 8) | ((operation & 6) << 5) | ((cache & 3) << 25)
}

/// Direct dispatch initiator: keep the caller's USE_THREAD_DIMENSIONS and
/// order bits, force COMPUTE_SHADER_EN and FORCE_START_AT_000.
fn dispatch_initiator(modifier: u32) -> u32 {
    (modifier & 0xA038) | 0x41
}

/// Scalar register base for indirect-draw modifiers (stage in bits 31:29).
fn indirect_sgpr_base(modifier: u32) -> u32 {
    if matches!(modifier >> 29, 3 | 5) {
        0x10C
    } else {
        0x8C
    }
}

/// Register locations patched by an indirect draw: base vertex (and, for
/// indexed draws, draw index) in the low half, start instance in the high
/// half.
fn indirect_patch_locations(modifier: u64, indexed: bool) -> u64 {
    let m = modifier as u32;
    let base = indirect_sgpr_base(m);
    let mut loc: u64 = if m & 1 != 0 { (base + ((m >> 9) & 0x1F)) as u64 } else { 0x280 };
    let inst: u64 = if m & 4 != 0 { (base + ((m >> 19) & 0x1F)) as u64 } else { 0x280 };
    loc |= inst << 32;
    if indexed && m & 2 != 0 {
        loc |= ((base + ((m >> 14) & 0x1F)) as u64) << 16;
    }
    loc
}

/// Draw-index location of a multi-draw (`0x280` when disabled).
fn indirect_draw_index_location(modifier: u32) -> u32 {
    if modifier & 8 != 0 {
        indirect_sgpr_base(modifier) + ((modifier >> 24) & 0x1F)
    } else {
        0x280
    }
}

fn indirect_draw_initiator(modifier: u64) -> u32 {
    if modifier & (1 << 32) != 0 {
        2
    } else {
        (((modifier as u32) >> 3) & 0x20) | 2
    }
}

fn draw_index_initiator(modifier: u64) -> u32 {
    if modifier & (1 << 32) != 0 {
        0
    } else {
        ((modifier as u32) >> 3) & 0x20
    }
}

/// Dwords emitted by the marker builders for a string of `len` bytes
/// (terminator included, at least one payload dword).
pub(crate) fn marker_dwords(len: usize) -> u32 {
    (((len as u32) + 4) / 4).max(1) + 1
}

/// Packet size of an EVENT_WRITE: events 0x38/0x39 carry an address.
pub(crate) fn event_write_dwords(event_type: u32) -> u32 {
    if event_type & !1 == 0x38 {
        4
    } else {
        2
    }
}

/// Dwords of a context-state operation (0 = invalid operation).
pub(crate) fn context_state_dwords(operation: u32) -> u32 {
    match operation {
        0 => 5,
        1 | 2 => 27,
        3 => 32,
        _ => 0,
    }
}

/// Dwords of a wait-on-address sequence: internal-data begin tag (4), the
/// wait (7 or 9) and the end tag (3).
pub(crate) fn wait_on_address_dwords(size: u32) -> u32 {
    match size {
        0 => 14,
        1 => 16,
        _ => 0,
    }
}

const REWIND_VALID: u32 = 1 << 31;
const REWIND_OFFLOAD: u32 = 1 << 24;

/// Workload stream ids and packet sizes.
pub(crate) const WORKLOAD_STREAM_MIN: u32 = 1;
pub(crate) const WORKLOAD_STREAM_MAX: u32 = 31;
const WORKLOAD_ID_MAX: u32 = 63;
const WORKLOAD_ACTIVE_DWORDS: u32 = 18;
const WORKLOAD_COMPLETE_DWORDS: u32 = 12;

// ---------------------------------------------------------------------------
// Generic NOP / dispatch
// ---------------------------------------------------------------------------

/// `sceAgcCbNop(cb, dwords)`: reserves `dwords` dwords behind a NOP header
/// and leaves the payload to the caller. One dword is the one-dword NOP
/// (0xFFFF1000).
fn cb_nop(ctx: &mut CpuContext) -> HleResult {
    let (cb, n) = (arg(ctx, 0), arg(ctx, 1) as u32);
    if cb == 0 || !(1..=super::pm4::MAX_LENGTH).contains(&n) {
        return 0;
    }
    let Some(a) = alloc_dwords(cb, n) else { return 0 };
    if !mem::w32(a, pm4(n, op::NOP, nop::ZERO)) {
        return 0;
    }
    ptr(a)
}

fn cb_nop_get_size(ctx: &mut CpuContext) -> HleResult {
    size_bytes(arg(ctx, 0) as u32)
}

fn cb_dispatch(ctx: &mut CpuContext) -> HleResult {
    let cb = arg(ctx, 0);
    let (x, y, z, m) = (arg(ctx, 1) as u32, arg(ctx, 2) as u32, arg(ctx, 3) as u32, arg(ctx, 4) as u32);
    ptr(emit(cb, &[pm4(5, op::DISPATCH_DIRECT, 0), x, y, z, dispatch_initiator(m)]))
}

fn cb_dispatch_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(5)
}

fn acb_reset_queue(ctx: &mut CpuContext) -> HleResult {
    ptr(emit(arg(ctx, 0), &[pm4(2, op::NOP, nop::DISPATCH_RESET), 0]))
}

/// `sceAgcDcbResetQueue(cb, op, state)`: CLEAR_STATE with the state bits.
fn dcb_reset_queue(ctx: &mut CpuContext) -> HleResult {
    let (opv, state) = (arg(ctx, 1) as u32, arg(ctx, 2) as u32);
    if opv & !0xFFF != 0 {
        return 0;
    }
    ptr(emit(arg(ctx, 0), &[pm4(2, op::CLEAR_STATE, 0), state & 0xF]))
}

/// `sceAgcDcbClearState(cb, command)`: CLEAR_STATE with the low four
/// command bits.
fn dcb_clear_state(ctx: &mut CpuContext) -> HleResult {
    ptr(emit(arg(ctx, 0), &[pm4(2, op::CLEAR_STATE, 0), arg(ctx, 1) as u32 & 0xF]))
}

/// PFP_SYNC_ME: the prefetch parser waits for the micro engine.
fn dcb_stall_parser(ctx: &mut CpuContext) -> HleResult {
    ptr(emit(arg(ctx, 0), &[pm4(2, op::PFP_SYNC_ME, 0), 0]))
}

fn dcb_stall_parser_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(2)
}

// ---------------------------------------------------------------------------
// Acquire / release / write / wait
// ---------------------------------------------------------------------------

/// Misaligned or out-of-range sizes/bases are only reported.
fn acquire_mem_packet(cb: u64, control0: u32, gcr: u32, base: u64, size: u64, poll: u32) -> u64 {
    let no_size = size == u64::MAX;
    if (!no_size && (size & 0xFF != 0 || size >> 40 != 0)) || base & 0xFF != 0 || base >> 40 != 0 {
        log::debug!("[agc] AcquireMem base 0x{base:x} size 0x{size:x} not 256-byte aligned or too large");
    }
    emit(
        cb,
        &[
            pm4(8, op::NOP, nop::ACQUIRE_MEM),
            control0,
            if no_size { 0 } else { (size >> 8) as u32 },
            0,
            (base >> 8) as u32,
            0,
            poll / 40,
            gcr,
        ],
    )
}

fn acb_acquire_mem(ctx: &mut CpuContext) -> HleResult {
    let (cb, gcr, base, size, poll) = (arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2), arg(ctx, 3), arg(ctx, 4) as u32);
    ptr(acquire_mem_packet(cb, 0x8000_0000, gcr, base, size, poll))
}

fn dcb_acquire_mem(ctx: &mut CpuContext) -> HleResult {
    let cb = arg(ctx, 0);
    let engine = (arg(ctx, 1) & 0xFF) as u32;
    let (cbdb, gcr, base, size) = (arg(ctx, 2) as u32, arg(ctx, 3) as u32, arg(ctx, 4), arg(ctx, 5));
    let Some(poll) = stack_arg(ctx, 1) else { return 0 };
    if engine > 1 {
        log::debug!("[agc] DcbAcquireMem engine {engine} unsupported");
    }
    ptr(acquire_mem_packet(cb, ((engine & 1) << 31) | cbdb, gcr, base, size, poll as u32))
}

fn acquire_mem_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(8)
}

/// Wait-on-address: the internal-data begin tag (address in the payload),
/// the native WAIT_REG_MEM(_64), and the end tag. Returns the begin tag.
fn wait_reg_mem_packet(cb: u64, size: u32, cmp: u32, operation: u32, cache: u32, addr: u64, reference: u64, mask: u64, poll: u32) -> u64 {
    use super::pm4::{INTERNAL_DATA_REGISTER, WAIT_USER_DATA_BEGIN32_TAG, WAIT_USER_DATA_BEGIN64_TAG, WAIT_USER_DATA_END_TAG};
    if size > 1 {
        return 0;
    }
    if cache > 3 {
        log::debug!("[agc] WaitRegMem cache policy {cache} unsupported");
    }
    let tag = if size == 0 { WAIT_USER_DATA_BEGIN32_TAG } else { WAIT_USER_DATA_BEGIN64_TAG };
    let begin = [pm4(4, op::SET_UCONFIG_REG, 1), INTERNAL_DATA_REGISTER, tag | (hi(addr) & 0xFFFF), lo(addr)];
    let end = [pm4(3, op::SET_UCONFIG_REG, 1), INTERNAL_DATA_REGISTER, WAIT_USER_DATA_END_TAG];
    let poll = encode_wait_poll(poll);
    let mut w = [0u32; 16];
    w[..4].copy_from_slice(&begin);
    let n = if size == 0 {
        w[4..11].copy_from_slice(&[
            pm4(7, op::WAIT_REG_MEM, 0),
            encode_wait32_control(cmp, operation, cache),
            (addr as u32) & !3,
            hi(addr) & 0x3FFFF,
            reference as u32,
            mask as u32,
            poll,
        ]);
        11
    } else {
        w[4..13].copy_from_slice(&[
            pm4(9, op::WAIT_REG_MEM64, 0),
            encode_wait64_control(cmp, operation, cache),
            (addr as u32) & !7,
            hi(addr) & 0x3FFFF,
            lo(reference),
            hi(reference),
            lo(mask),
            hi(mask),
            poll,
        ]);
        13
    };
    w[n..n + 3].copy_from_slice(&end);
    emit(cb, &w[..n + 3])
}

fn dcb_wait_reg_mem(ctx: &mut CpuContext) -> HleResult {
    let cb = arg(ctx, 0);
    let size = (arg(ctx, 1) & 0xFF) as u32;
    let cmp = (arg(ctx, 2) & 0xFF) as u32;
    let operation = (arg(ctx, 3) & 0xFF) as u32;
    let cache = (arg(ctx, 4) & 0xFF) as u32;
    let addr = arg(ctx, 5);
    let (Some(reference), Some(mask), Some(poll)) = (stack_arg(ctx, 1), stack_arg(ctx, 2), stack_arg(ctx, 3)) else {
        return 0;
    };
    ptr(wait_reg_mem_packet(cb, size, cmp, operation, cache, addr, reference, mask, poll as u32))
}

fn acb_wait_reg_mem(ctx: &mut CpuContext) -> HleResult {
    let cb = arg(ctx, 0);
    let size = (arg(ctx, 1) & 0xFF) as u32;
    let cmp = (arg(ctx, 2) & 0xFF) as u32;
    let cache = (arg(ctx, 3) & 0xFF) as u32;
    let (addr, reference) = (arg(ctx, 4), arg(ctx, 5));
    let (Some(mask), Some(poll)) = (stack_arg(ctx, 1), stack_arg(ctx, 2)) else { return 0 };
    ptr(wait_reg_mem_packet(cb, size, cmp, 0, cache, addr, reference, mask, poll as u32))
}

fn wait_on_address_get_size(ctx: &mut CpuContext) -> HleResult {
    size_bytes(wait_on_address_dwords(arg(ctx, 0) as u32 & 0xFF))
}

/// `sceAgcDcbWriteData(cb, dst, cache, dstAddr, data, count, increment,
/// writeConfirm)`: native WRITE_DATA; destination bit 0 selects the engine,
/// bits 1..4 DST_SEL.
fn dcb_write_data(ctx: &mut CpuContext) -> HleResult {
    let cb = arg(ctx, 0);
    let dst = (arg(ctx, 1) & 0xFF) as u32;
    let cache = (arg(ctx, 2) & 0xFF) as u32;
    let (dst_addr, data, count) = (arg(ctx, 3), arg(ctx, 4), arg(ctx, 5) as u32);
    let (Some(inc), Some(confirm)) = (stack_arg(ctx, 1), stack_arg(ctx, 2)) else { return 0 };
    if data == 0 || count > 0x3FFD {
        return 0;
    }
    let confirm = if dst == 0 { 0 } else { confirm as u32 & 1 };
    let control = ((dst & 1) << 30) | ((dst & 0x1E) << 7) | ((inc as u32 & 1) << 16) | (confirm << 20) | ((cache & 3) << 25);
    write_data_packet(cb, control, (dst_addr as u32) & !3, hi(dst_addr), data, count)
}

/// `sceAgcAcbWriteData(cb, dst, cache, dstAddr, data, count, increment,
/// writeConfirm)`: native WRITE_DATA with the raw DST_SEL.
fn acb_write_data(ctx: &mut CpuContext) -> HleResult {
    let cb = arg(ctx, 0);
    let dst = (arg(ctx, 1) & 0xFF) as u32;
    let cache = (arg(ctx, 2) & 0xFF) as u32;
    let (dst_addr, data, count) = (arg(ctx, 3), arg(ctx, 4), arg(ctx, 5) as u32);
    let (Some(inc), Some(confirm)) = (stack_arg(ctx, 1), stack_arg(ctx, 2)) else { return 0 };
    if data == 0 || count == 0 || count > 0x3FFD {
        return 0;
    }
    let confirm = if dst == 0 { 0 } else { confirm as u32 & 1 };
    let control = ((dst & 0xF) << 8) | ((inc as u32 & 1) << 16) | (confirm << 20) | ((cache & 3) << 25);
    write_data_packet(cb, control, lo(dst_addr), hi(dst_addr), data, count)
}

fn write_data_packet(cb: u64, control: u32, addr_lo: u32, addr_hi: u32, data: u64, count: u32) -> HleResult {
    let n = count + 4;
    let Some(a) = alloc_dwords(cb, n) else { return 0 };
    if !mem::write_dwords(a, &[pm4(n, op::WRITE_DATA, 0), control, addr_lo, addr_hi]) || !mem::copy(a + 16, data, count as u64 * 4) {
        return 0;
    }
    ptr(a)
}

fn dcb_write_data_get_size(ctx: &mut CpuContext) -> HleResult {
    size_bytes(arg(ctx, 0) as u32 + 4)
}

/// `sceAgcCbReleaseMem(cb, action, gcr, dst, cache, addr, dataSel, data,
/// gdsOffset, gdsSize, interrupt, contextId)`: 8-dword private packet whose
/// dwords 1..7 follow the native RELEASE_MEM layout.
fn cb_release_mem(ctx: &mut CpuContext) -> HleResult {
    let cb = arg(ctx, 0);
    let action = (arg(ctx, 1) & 0xFF) as u32;
    let gcr = (arg(ctx, 2) & 0xFFFF) as u32;
    let dst = (arg(ctx, 3) & 0xFF) as u32;
    let cache = (arg(ctx, 4) & 0xFF) as u32;
    let addr = arg(ctx, 5);
    let mut s = [0u64; 6];
    for (i, v) in s.iter_mut().enumerate() {
        match stack_arg(ctx, i as u64 + 1) {
            Some(x) => *v = x,
            None => return 0,
        }
    }
    let data_sel = (s[0] & 0xFF) as u32;
    let data = s[1];
    let gds_offset = s[2] & 0xFFFF;
    let gds_size = s[3] & 0xFFFF;
    let interrupt = (s[4] & 0xFF) as u32;
    let context_id = s[5] as u32;
    if dst > 1 || !matches!(data_sel, 0 | 1 | 2 | 3 | 5) || interrupt > 4 {
        return 0;
    }
    // GL2 invalidate implies the write-back.
    let gcr = if gcr & 0x300 == 0x100 { gcr | 0x200 } else { gcr };
    let (a, d) = if interrupt == 4 {
        (0, 0)
    } else if data_sel == 5 {
        (addr, gds_offset | gds_size << 16)
    } else {
        (addr, data)
    };
    let event_index = if action >= 0x2F { 6 } else { 5 };
    ptr(emit(
        cb,
        &[
            pm4(8, op::NOP, nop::RELEASE_MEM),
            (action & 0x3F) | (event_index << 8) | ((gcr & 0xFFF) << 12) | ((cache & 3) << 25),
            ((dst & 3) << 16) | ((interrupt & 7) << 24) | ((data_sel & 7) << 29),
            (a as u32) & !3,
            hi(a),
            lo(d),
            hi(d),
            context_id & 0x07FF_FFFF,
        ],
    ))
}

fn cb_queue_end_of_pipe_action_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(8)
}

/// End-of-shader actions use the same 8-dword release packet (inferred; the
/// builders themselves are not exported to titles here).
fn queue_end_of_shader_action_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(8)
}

/// `sceAgcCbCondWrite(cb, cmp, space, writeAddr, writeValue, readAddr, ref, mask)`.
fn cb_cond_write(ctx: &mut CpuContext) -> HleResult {
    let cb = arg(ctx, 0);
    let cmp = arg(ctx, 1) as u32;
    let space = arg(ctx, 2) as u32;
    let (waddr, wval, raddr) = (arg(ctx, 3), arg(ctx, 4) as u32, arg(ctx, 5));
    let (Some(reference), Some(mask)) = (stack_arg(ctx, 1), stack_arg(ctx, 2)) else { return 0 };
    if cmp > 6
        || !matches!(space, 1 | 2)
        || raddr == 0
        || (space == 1 && (waddr == 0 || waddr % 4 != 0))
        || raddr % 4 != 0
    {
        return 0;
    }
    let control = 0x10 | cmp | if space == 1 { 0x100 } else { 0 };
    ptr(emit(
        cb,
        &[
            pm4(9, op::COND_WRITE, 0),
            control,
            (raddr as u32) & !3,
            hi(raddr) & 0xFFFF,
            reference as u32,
            mask as u32,
            (waddr as u32) & !3,
            hi(waddr) & 0xFFFF,
            wval,
        ],
    ))
}

fn cb_cond_write_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(9)
}

// ---------------------------------------------------------------------------
// LOD statistics, markers, workloads
// ---------------------------------------------------------------------------

/// `sceAgcDcbGetLodStats(cb, cache, dst, control, counterMask, reset, enable, select)`.
fn dcb_get_lod_stats(ctx: &mut CpuContext) -> HleResult {
    let cb = arg(ctx, 0);
    let cache = arg(ctx, 1) as u32 & 3;
    let dst = arg(ctx, 2);
    let control = arg(ctx, 3) as u32;
    let mask = arg(ctx, 4) as u32 & 0xFF;
    let reset = arg(ctx, 5) as u32 & 1;
    let (Some(enable), Some(select)) = (stack_arg(ctx, 1), stack_arg(ctx, 2)) else { return 0 };
    let ctl = (cache << 28) | ((enable as u32 & 1) << 19) | (reset << 18) | (mask << 10) | ((select as u32 & 0xFF) << 2);
    ptr(emit(cb, &[pm4(5, op::GET_LOD_STATS, 0), control, (dst as u32) & !0x3F, hi(dst), ctl]))
}

/// Always the 5-dword GET_LOD_STATS packet.
fn dcb_get_lod_stats_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(5)
}

fn marker_packet(cb: u64, s: u64, custom: u32) -> u64 {
    // At most 4095 bytes plus the terminator.
    let mut text = [0u32; 1024];
    let len = if s == 0 {
        0
    } else {
        match mem::read_cstr_into(s, bytemuck_u8(&mut text)) {
            Some(n) => n,
            None => return 0,
        }
    };
    let n = marker_dwords(len);
    let Some(a) = alloc_dwords(cb, n) else { return 0 };
    if !mem::w32(a, pm4(n, op::NOP, custom)) || !mem::write_dwords(a + 4, &text[..n as usize - 1]) {
        return 0;
    }
    a
}

/// The dwords of `w` as bytes (little endian host).
fn bytemuck_u8(w: &mut [u32]) -> &mut [u8] {
    // SAFETY: u8 has no alignment or validity requirements.
    unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr() as *mut u8, w.len() * 4) }
}

fn push_marker(ctx: &mut CpuContext) -> HleResult {
    ptr(marker_packet(arg(ctx, 0), arg(ctx, 1), nop::PUSH_MARKER))
}

/// Set-marker is encoded like a push marker (an inert debug annotation).
fn set_marker(ctx: &mut CpuContext) -> HleResult {
    ptr(marker_packet(arg(ctx, 0), arg(ctx, 1), nop::PUSH_MARKER))
}

fn pop_marker(ctx: &mut CpuContext) -> HleResult {
    ptr(emit(arg(ctx, 0), &[pm4(2, op::NOP, nop::POP_MARKER), 0]))
}

/// `sceAgcDcbSetWorkloadsActive(cb, streamId, ids, count)`: an 18-dword NOP
/// carrying the stream id and the 64-bit active mask. The stream must have
/// been registered.
fn set_workloads_active(ctx: &mut CpuContext) -> HleResult {
    let (cb, stream, ids, count) = (arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2), arg(ctx, 3) as u32);
    if ids == 0 || count == 0 || count > WORKLOAD_ID_MAX || !(WORKLOAD_STREAM_MIN..=WORKLOAD_STREAM_MAX).contains(&stream) {
        return 0;
    }
    if !super::driver::workload_stream_registered(stream) {
        return 0;
    }
    let mut mask = 0u64;
    for i in 0..count as u64 {
        let Some(id) = mem::r32(ids + i * 4) else { return 0 };
        if id > WORKLOAD_ID_MAX || mask & (1 << id) != 0 {
            return 0;
        }
        mask |= 1 << id;
    }
    let mut w = [0u32; WORKLOAD_ACTIVE_DWORDS as usize];
    w[0] = pm4(WORKLOAD_ACTIVE_DWORDS, op::NOP, nop::ZERO);
    w[1] = stream;
    w[2] = lo(mask);
    w[3] = hi(mask);
    ptr(emit(cb, &w))
}

/// `sceAgcDcbSetWorkloadComplete(cb, streamId, workloadId)`: a 12-dword NOP
/// with the stream, the id and the clear mask. The stream must have been
/// registered.
fn set_workload_complete(ctx: &mut CpuContext) -> HleResult {
    let (cb, stream, id) = (arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2) as u32);
    if !(WORKLOAD_STREAM_MIN..=WORKLOAD_STREAM_MAX).contains(&stream) || id > WORKLOAD_ID_MAX {
        return 0;
    }
    if !super::driver::workload_stream_registered(stream) {
        return 0;
    }
    let clear = !(1u64 << id);
    let mut w = [0u32; WORKLOAD_COMPLETE_DWORDS as usize];
    w[0] = pm4(WORKLOAD_COMPLETE_DWORDS, op::NOP, nop::ZERO);
    w[1] = stream;
    w[2] = id;
    w[3] = lo(clear);
    w[4] = hi(clear);
    ptr(emit(cb, &w))
}

// ---------------------------------------------------------------------------
// Control flow
// ---------------------------------------------------------------------------

/// REWIND: bit 31 = valid (initial state), bit 24 = offload (async only).
fn rewind_packet(cb: u64, initial_state: u64, offload: u64) -> u64 {
    let body = if initial_state & 1 != 0 { REWIND_VALID } else { 0 } | if offload & 1 != 0 { REWIND_OFFLOAD } else { 0 };
    emit(cb, &[pm4(2, op::REWIND, 0), body])
}

/// `sceAgcDcbRewind(cb, u32 initialState)`.
fn dcb_rewind(ctx: &mut CpuContext) -> HleResult {
    ptr(rewind_packet(arg(ctx, 0), arg(ctx, 1), 0))
}

/// `sceAgcAcbRewind(cb, u8 initialState, u8 offload)`.
fn acb_rewind(ctx: &mut CpuContext) -> HleResult {
    ptr(rewind_packet(arg(ctx, 0), arg(ctx, 1) as u8 as u64, arg(ctx, 2) as u8 as u64))
}

fn rewind_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(2)
}

/// `sceAgcDcbJump(cb, mode, cache, target, dwords)`: INDIRECT_BUFFER, mode
/// bit 20 selects chain (1) or call (0).
fn dcb_jump(ctx: &mut CpuContext) -> HleResult {
    let (cb, mode, cache, target, dw) = (arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2) as u32, arg(ctx, 3), arg(ctx, 4) as u32);
    let control = 0x0F20_0000 | ((cache & 3) << 28) | ((mode & 1) << 20) | (dw & 0xFFFFF);
    ptr(emit(cb, &[pm4(4, op::INDIRECT_BUFFER, 0), (target as u32) & !3, hi(target), control]))
}

/// `sceAgcAcbJump(cb, cache, target, dwords)`: a chained INDIRECT_BUFFER.
fn acb_jump(ctx: &mut CpuContext) -> HleResult {
    let (cb, cache, target, dw) = (arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2), arg(ctx, 3) as u32);
    let control = 0x0F90_0000 | ((cache & 3) << 28) | (dw & 0xFFFFF);
    ptr(emit(cb, &[pm4(4, op::INDIRECT_BUFFER, 0), (target as u32) & !3, hi(target), control]))
}

fn jump_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(4)
}

/// `sceAgcCbBranch(cb, mode, cmp, cmpAddr, mask, ref, cache1, buf1, dw1,
/// cache2, buf2, dw2)`: the 14-dword conditional INDIRECT_BUFFER.
fn cb_branch(ctx: &mut CpuContext) -> HleResult {
    let cb = arg(ctx, 0);
    let mode = arg(ctx, 1) as u32 & 3;
    let cmp = arg(ctx, 2) as u32 & 7;
    let (cmp_addr, mask, reference) = (arg(ctx, 3), arg(ctx, 4), arg(ctx, 5));
    let mut s = [0u64; 6];
    for (i, v) in s.iter_mut().enumerate() {
        match stack_arg(ctx, i as u64 + 1) {
            Some(x) => *v = x,
            None => return 0,
        }
    }
    let (cache1, buf1, dw1, cache2, buf2, dw2) = (s[0] as u32 & 3, s[1], s[2] as u32, s[3] as u32 & 3, s[4], s[5] as u32);
    ptr(emit(
        cb,
        &[
            pm4(14, op::INDIRECT_BUFFER, 0),
            mode | (cmp << 8),
            (cmp_addr as u32) & !7,
            hi(cmp_addr),
            lo(mask),
            hi(mask),
            lo(reference),
            hi(reference),
            (buf1 as u32) & !3,
            hi(buf1),
            (dw1 & 0xFFFFF) | (cache1 << 28),
            (buf2 as u32) & !3,
            hi(buf2),
            (dw2 & 0xFFFFF) | (cache2 << 28),
        ],
    ))
}

fn cb_branch_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(14)
}

/// `sceAgcDcbCondExec(cb, predicateAddr, dwords)`: skips the next `dwords`
/// when the predicate dword is zero.
fn cond_exec(ctx: &mut CpuContext) -> HleResult {
    let (cb, addr, n) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2) as u32);
    if addr == 0 || addr & 3 != 0 || n > 0x3FFF {
        return 0;
    }
    ptr(emit(cb, &[pm4(5, op::COND_EXEC, 0), (addr as u32) & !3, hi(addr), 0, n & 0x3FFF]))
}

fn cond_exec_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(5)
}

/// `sceAgcDcbSetPredication(cb, condition, op, wait, addr)`.
fn dcb_set_predication(ctx: &mut CpuContext) -> HleResult {
    let cb = arg(ctx, 0);
    let cond = arg(ctx, 1) as u32 & 1;
    let pop = arg(ctx, 2) as u32 & 7;
    let wait = arg(ctx, 3) as u32 & 1;
    let addr = arg(ctx, 4);
    let flags = (cond << 8) | (wait << 12) | (pop << 16);
    ptr(emit(cb, &[pm4(4, op::SET_PREDICATION, 0), flags, (addr as u32) & 0xFFFF_FFF0, hi(addr)]))
}

/// Bool / ZPass predication enable and predication disable all emit one
/// 4-dword SET_PREDICATION packet.
fn predication_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(4)
}

// ---------------------------------------------------------------------------
// DMA / copy / atomics / semaphores
// ---------------------------------------------------------------------------

/// `sceAgcDcbCopyData(cb, dstSel, dstCache, dstAddr, srcSel, srcCache,
/// src, elementSize, confirm)`: graphics form (selector bit 0 is the engine).
fn dcb_copy_data(ctx: &mut CpuContext) -> HleResult {
    let cb = arg(ctx, 0);
    let dst_sel = (arg(ctx, 1) & 0xFF) as u32;
    let dst_cache = (arg(ctx, 2) & 0xFF) as u32;
    let dst = arg(ctx, 3);
    let src_sel = (arg(ctx, 4) & 0xFF) as u32;
    let src_cache = (arg(ctx, 5) & 0xFF) as u32;
    let (Some(src), Some(size), Some(confirm)) = (stack_arg(ctx, 1), stack_arg(ctx, 2), stack_arg(ctx, 3)) else {
        return 0;
    };
    let control = ((src_sel >> 1) & 0xF)
        | (((dst_sel >> 1) & 0xF) << 8)
        | ((src_cache & 3) << 13)
        | ((size as u32 & 1) << 16)
        | ((confirm as u32 & 1) << 20)
        | ((dst_cache & 3) << 25)
        | ((src_sel & 1) << 30);
    ptr(emit(cb, &[pm4(6, op::COPY_DATA, 0), control, lo(src), hi(src), lo(dst), hi(dst)]))
}

/// `sceAgcAcbCopyData(cb, dstSel, dstCache, dstAddr, srcSel, srcCache, src,
/// itemSize, confirm)`: compute form with raw 4-bit selectors.
fn acb_copy_data(ctx: &mut CpuContext) -> HleResult {
    let cb = arg(ctx, 0);
    let dst_sel = (arg(ctx, 1) & 0xFF) as u32;
    let dst_cache = (arg(ctx, 2) & 0xFF) as u32;
    let dst = arg(ctx, 3);
    let src_sel = (arg(ctx, 4) & 0xFF) as u32;
    let src_cache = (arg(ctx, 5) & 0xFF) as u32;
    let (Some(src), Some(size), Some(confirm)) = (stack_arg(ctx, 1), stack_arg(ctx, 2), stack_arg(ctx, 3)) else {
        return 0;
    };
    let control = (src_sel & 0xF)
        | ((dst_sel & 0xF) << 8)
        | ((src_cache & 3) << 13)
        | ((size as u32 & 1) << 16)
        | ((confirm as u32 & 1) << 20)
        | ((dst_cache & 3) << 25);
    ptr(emit(cb, &[pm4(6, op::COPY_DATA, 0), control, lo(src), hi(src), lo(dst), hi(dst)]))
}

fn copy_data_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(6)
}

/// `sceAgcDcbDmaData(cb, engine, dst, dstCache, dstAddr, src, srcCache,
/// srcAddr, bytes, waitPrev, confirm, blockEngine)`: native DMA_DATA.
fn dcb_dma_data(ctx: &mut CpuContext) -> HleResult {
    let cb = arg(ctx, 0);
    let engine = (arg(ctx, 1) & 0xFF) as u32;
    let dst = (arg(ctx, 2) & 0xFF) as u32;
    let dst_cache = (arg(ctx, 3) & 0xFF) as u32;
    let dst_addr = arg(ctx, 4);
    let src = (arg(ctx, 5) & 0xFF) as u32;
    let mut s = [0u64; 6];
    for (i, v) in s.iter_mut().enumerate() {
        match stack_arg(ctx, i as u64 + 1) {
            Some(x) => *v = x,
            None => return 0,
        }
    }
    // Register-space sources read the engine's own registers.
    let src_addr = match (src, engine == 1) {
        (0x14, true) => 0x30148,
        (0x14, false) => 0x30174,
        (0x24, true) => 0x30150,
        (0x24, false) => 0x3017C,
        (0x64, true) => 0x30158,
        (0x64, false) => 0x30184,
        _ => s[1],
    };
    ptr(dma_data_packet(cb, engine, dst, dst_cache, dst_addr, src, (s[0] & 0xFF) as u32, src_addr, s[2] as u32, s[3] as u32, s[4] as u32, s[5] as u32))
}

/// The 7-dword native DMA_DATA packet.
#[allow(clippy::too_many_arguments)]
fn dma_data_packet(
    cb: u64,
    engine: u32,
    dst: u32,
    dst_cache: u32,
    dst_addr: u64,
    src: u32,
    src_cache: u32,
    src_addr: u64,
    bytes: u32,
    wait: u32,
    confirm: u32,
    block: u32,
) -> u64 {
    emit(
        cb,
        &[
            pm4(7, op::DMA_DATA, 0),
            (engine & 1) | ((src_cache & 3) << 13) | ((dst & 3) << 20) | ((dst_cache & 3) << 25) | ((src & 3) << 29) | ((block & 1) << 31),
            lo(src_addr),
            hi(src_addr),
            lo(dst_addr),
            hi(dst_addr),
            (bytes & 0x03FF_FFFF)
                | ((src & 4) << 24)
                | ((dst & 4) << 25)
                | ((src & 8) << 25)
                | ((dst & 8) << 26)
                | ((wait & 1) << 30)
                | ((confirm & 1) << 31),
        ],
    )
}

/// `sceAgcAcbDmaData(cb, dst, dstCache, dstAddr, src, srcCache, srcOrImm,
/// bytes, waitPrev, confirm)`: native DMA_DATA without engine selection.
fn acb_dma_data(ctx: &mut CpuContext) -> HleResult {
    let cb = arg(ctx, 0);
    let dst = (arg(ctx, 1) & 0xFF) as u32;
    let dst_cache = (arg(ctx, 2) & 0xFF) as u32;
    let dst_addr = arg(ctx, 3);
    let src = (arg(ctx, 4) & 0xFF) as u32;
    let src_cache = (arg(ctx, 5) & 0xFF) as u32;
    let (Some(src_val), Some(bytes), Some(wait), Some(confirm)) =
        (stack_arg(ctx, 1), stack_arg(ctx, 2), stack_arg(ctx, 3), stack_arg(ctx, 4))
    else {
        return 0;
    };
    let src_addr = match src {
        0x14 => 0x30174,
        0x24 => 0x3017C,
        0x25 => 0x30184,
        _ => src_val,
    };
    ptr(dma_data_packet(cb, 0, dst, dst_cache, dst_addr, src, src_cache, src_addr, bytes as u32, wait as u32, confirm as u32, 0))
}

fn dma_data_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(7)
}

/// Inferred: `sceAgcDcbAtomicMem(cb, op, command, cache, addr, data,
/// compare, loopInterval)` -> the 9-dword ATOMIC_MEM packet
/// `[ctl = op[6:0] | cmd<<8 | cache<<25 | engine<<30, addr, data, compare,
/// loop]`.
fn atomic_mem_packet(ctx: &CpuContext, engine: u32) -> u64 {
    let cb = arg(ctx, 0);
    let aop = arg(ctx, 1) as u32 & 0x7F;
    let cmd = arg(ctx, 2) as u32 & 0xF;
    let cache = arg(ctx, 3) as u32 & 3;
    let (addr, data) = (arg(ctx, 4), arg(ctx, 5));
    let (Some(compare), Some(lp)) = (stack_arg(ctx, 1), stack_arg(ctx, 2)) else { return 0 };
    emit(
        cb,
        &[
            pm4(9, op::ATOMIC_MEM, 0),
            aop | (cmd << 8) | (cache << 25) | (engine << 30),
            lo(addr),
            hi(addr),
            lo(data),
            hi(data),
            lo(compare),
            hi(compare),
            lp as u32 & 0x1FFF,
        ],
    )
}

fn dcb_atomic_mem(ctx: &mut CpuContext) -> HleResult {
    ptr(atomic_mem_packet(ctx, 0))
}

fn acb_atomic_mem(ctx: &mut CpuContext) -> HleResult {
    ptr(atomic_mem_packet(ctx, 0))
}

fn atomic_mem_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(9)
}

/// ATOMIC_GDS is header + 10 dwords (the builder is not exported here).
fn atomic_gds_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(11)
}

/// The ATOMIC_GDS packet of `sceAgcAcbAtomicGds` (version 0x900):
/// `(control, mode, gdsOp, src, offset, index, mask, data, compare, extra)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn atomic_gds_packet(control: u32, mode: u32, gds_op: u32, src: u32, offset: u16, index: u16, mask: u32, data: u32, compare: u64, extra: u64) -> [u32; 11] {
    let operation = (control & 0xFF)
        | ((control & 0x200) << 7)
        | ((control & 0x20) << 13)
        | ((control & 0x40) << 14)
        | if control & 0x20 == 0 && mode == 1 { 0x20000 } else { 0 };
    [
        pm4(11, op::ATOMIC_GDS, 0),
        operation,
        ((gds_op & 1) << 8) | (src & 0x3F),
        offset as u32,
        index as u32,
        mask & 0x00FF_00FF,
        data,
        lo(compare),
        hi(compare),
        lo(extra),
        hi(extra),
    ]
}

/// `sceAgcAcbAtomicGds_0900(cb, control, mode, gdsOp, src, offset, index,
/// mask, data, compare, extra)`: the last five arguments are on the stack.
/// The graphics forms (`sceAgcDcbAtomicGds` and its 0x900 version) write
/// the same packet.
fn acb_atomic_gds(ctx: &mut CpuContext) -> HleResult {
    let st = |n| stack_arg(ctx, n).unwrap_or(0);
    let w = atomic_gds_packet(arg(ctx, 1) as u32, arg(ctx, 2) as u32, arg(ctx, 3) as u32, arg(ctx, 4) as u32, arg(ctx, 5) as u16, st(1) as u16, st(2) as u32, st(3) as u32, st(4), st(5));
    ptr(emit(arg(ctx, 0), &w))
}

/// `sceAgcCbMemSemaphore(cb, address, operation, modifier, value)`: a
/// 4-dword MEM_SEMAPHORE on the address rounded down to 16 bytes with the
/// control word measured on a console (0x80100000) whatever the operation,
/// modifier and value.
fn cb_mem_semaphore(ctx: &mut CpuContext) -> HleResult {
    let addr = arg(ctx, 1) & !0xF;
    ptr(emit(arg(ctx, 0), &[pm4(4, op::MEM_SEMAPHORE, 0), lo(addr), hi(addr), 0x8010_0000]))
}

/// `sceAgcDcbSetWorkloadStreamInactive(cb, streamId)` (and the ACB form):
/// nothing for a null buffer or an unregistered stream (measured); the
/// packet of a registered stream is not known.
fn set_workload_stream_inactive(ctx: &mut CpuContext) -> HleResult {
    let (cb, stream) = (arg(ctx, 0), arg(ctx, 1) as u32);
    if cb == 0 || !(WORKLOAD_STREAM_MIN..=WORKLOAD_STREAM_MAX).contains(&stream) || !super::driver::workload_stream_registered(stream) {
        return 0;
    }
    crate::agc::log_once("agc.workload.inactive", || log::warn!("[agc] SetWorkloadStreamInactive of registered stream {stream}: packet unknown, nothing written"));
    0
}

/// `sceAgcDcbMemSemaphore(cb, addr, op(6 signal / 7 wait), signalType, mailbox)`.
fn mem_semaphore(ctx: &mut CpuContext) -> HleResult {
    let (cb, addr, sop, sig, mb) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2) as u32, arg(ctx, 3) as u32, arg(ctx, 4) as u32);
    if addr == 0 || addr & 7 != 0 || !matches!(sop, 6 | 7) || sig > 1 || mb > 1 {
        return 0;
    }
    ptr(emit(cb, &[pm4(4, op::MEM_SEMAPHORE, 0), lo(addr), hi(addr), (sop << 29) | (sig << 20) | (mb << 16)]))
}

/// Inferred: `sceAgcDcbPrimeUtcl2(cb, permission, mode, addr, pages)` ->
/// PRIME_UTCL2 `[perm[2:0] | mode<<3, addrLo, addrHi, pages]`.
fn prime_utcl2(ctx: &mut CpuContext) -> HleResult {
    let (cb, perm, mode, addr, pages) = (arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2) as u32, arg(ctx, 3), arg(ctx, 4) as u32);
    ptr(emit(cb, &[pm4(5, op::PRIME_UTCL2, 0), (perm & 7) | ((mode & 7) << 3), lo(addr), hi(addr), pages]))
}

fn prime_utcl2_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(5)
}

// ---------------------------------------------------------------------------
// Events, flips
// ---------------------------------------------------------------------------

/// EVENT_WRITE: the partial flushes (7, 15, 16) carry event index 4; the
/// pixel-pipe dumps (0x38/0x39) an address with index 1.
fn event_write_packet(cb: u64, event_type: u32, addr: u64) -> u64 {
    if event_type > 0x3F {
        return 0;
    }
    if matches!(event_type, 7 | 15 | 16) {
        emit(cb, &[pm4(2, op::EVENT_WRITE, 0), 0x400 | event_type])
    } else if event_write_dwords(event_type) == 4 {
        emit(cb, &[pm4(4, op::EVENT_WRITE, 0), event_type | 0x100, (addr as u32) & !7, hi(addr)])
    } else {
        emit(cb, &[pm4(2, op::EVENT_WRITE, 0), event_type & 0x3F])
    }
}

fn event_write(ctx: &mut CpuContext) -> HleResult {
    ptr(event_write_packet(arg(ctx, 0), (arg(ctx, 1) & 0xFF) as u32, arg(ctx, 2)))
}

/// `sceAgcAcbEventWrite(cb, type, addr)`: always the 2-dword form; only the
/// compute partial flush carries event index 4.
fn acb_event_write(ctx: &mut CpuContext) -> HleResult {
    let t = (arg(ctx, 1) & 0xFF) as u32;
    ptr(emit(arg(ctx, 0), &[pm4(2, op::EVENT_WRITE, 0), (t & 0x3F) | if t == 7 { 0x400 } else { 0 }]))
}

fn acb_event_write_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(2)
}

fn event_write_get_size(ctx: &mut CpuContext) -> HleResult {
    size_bytes(event_write_dwords((arg(ctx, 0) & 0xFF) as u32))
}

/// Occlusion queries dump the pixel-pipe counters with one addressed
/// EVENT_WRITE (inferred size; builders not exported here).
fn occlusion_query_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(4)
}

/// `sceAgc{Dcb,Acb}WaitUntilSafeForRendering(cb, videoOutHandle, index)`.
fn wait_until_safe_for_rendering(ctx: &mut CpuContext) -> HleResult {
    let (cb, h, i) = (arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2) as u32);
    ptr(emit(cb, &[pm4(7, op::NOP, nop::WAIT_FLIP_DONE), h, i, 0, 0, 0, 0]))
}

/// `sceAgc{Dcb,Acb}SetFlip(cb, videoOutHandle, index, flipMode, flipArg)`.
fn set_flip(ctx: &mut CpuContext) -> HleResult {
    let (cb, h, i, m, a) = (arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2) as u32, arg(ctx, 3) as u32, arg(ctx, 4));
    ptr(emit(cb, &[pm4(6, op::NOP, nop::FLIP), h, i, m, lo(a), hi(a)]))
}

// ---------------------------------------------------------------------------
// Index / draw state and draws
// ---------------------------------------------------------------------------

fn set_index_size_packet(cb: u64, size: u32, cache: u32, instance_id: u32) -> u64 {
    emit(
        cb,
        &[
            pm4(3, op::SET_UCONFIG_REG_INDEX, 0),
            0x2000_0000 | super::pm4::VGT_INDEX_TYPE,
            0x400 | (size & 3) | ((cache & 3) << 6) | ((instance_id & 1) << 14),
        ],
    )
}

fn dcb_set_index_size(ctx: &mut CpuContext) -> HleResult {
    ptr(set_index_size_packet(arg(ctx, 0), (arg(ctx, 1) & 0xFF) as u32, (arg(ctx, 2) & 0xFF) as u32, 0))
}

/// Set-index-size with per-instance object id support (4th argument).
fn dcb_set_index_size_ex(ctx: &mut CpuContext) -> HleResult {
    ptr(set_index_size_packet(arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2) as u32, arg(ctx, 3) as u32))
}

fn dcb_set_index_size_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(3)
}

fn dcb_set_index_count(ctx: &mut CpuContext) -> HleResult {
    ptr(emit(arg(ctx, 0), &[pm4(2, op::INDEX_BUFFER_SIZE, 0), arg(ctx, 1) as u32]))
}

fn two_dword_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(2)
}

fn dcb_set_num_instances(ctx: &mut CpuContext) -> HleResult {
    ptr(emit(arg(ctx, 0), &[pm4(2, op::NUM_INSTANCES, 0), arg(ctx, 1) as u32]))
}

/// `sceAgcDcbSetIndexBuffer(cb, addr)`: INDEX_BASE.
fn dcb_set_index_buffer(ctx: &mut CpuContext) -> HleResult {
    let (cb, addr) = (arg(ctx, 0), arg(ctx, 1));
    if addr & 1 != 0 {
        return 0;
    }
    ptr(emit(cb, &[pm4(3, op::INDEX_BASE, 0), lo(addr), hi(addr)]))
}

fn dcb_set_index_buffer_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(3)
}

/// `sceAgcDcbDrawIndex(cb, count, indexAddr, modifier)`: DRAW_INDEX_2.
fn dcb_draw_index(ctx: &mut CpuContext) -> HleResult {
    let (cb, count, addr, modifier) = (arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2), arg(ctx, 3));
    if addr == 0 || addr & 1 != 0 {
        return 0;
    }
    ptr(emit(cb, &[pm4(6, op::DRAW_INDEX_2, 0), count.max(1), lo(addr), hi(addr), count, draw_index_initiator(modifier)]))
}

fn dcb_draw_index_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(6)
}

/// `sceAgcDcbDrawIndexAuto(cb, count, modifier)`: DRAW_INDEX_AUTO.
fn dcb_draw_index_auto(ctx: &mut CpuContext) -> HleResult {
    ptr(emit(arg(ctx, 0), &[pm4(3, op::DRAW_INDEX_AUTO, 0), arg(ctx, 1) as u32, draw_index_initiator(arg(ctx, 2)) | 2]))
}

fn dcb_draw_index_auto_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(3)
}

/// `sceAgcDcbDrawIndexOffset(cb, offset, count, modifier)`.
fn dcb_draw_index_offset(ctx: &mut CpuContext) -> HleResult {
    let (cb, offset, count, modifier) = (arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2) as u32, arg(ctx, 3));
    ptr(emit(cb, &[pm4(5, op::DRAW_INDEX_OFFSET_2, 0), count.max(1), offset, count, draw_index_initiator(modifier)]))
}

fn five_dword_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(5)
}

/// `sceAgcDcbDrawIndirect(cb, dataOffset, modifier)`.
fn dcb_draw_indirect(ctx: &mut CpuContext) -> HleResult {
    let (cb, offset, modifier) = (arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2));
    let loc = indirect_patch_locations(modifier, false);
    ptr(emit(cb, &[pm4(5, op::DRAW_INDIRECT, 0), offset, lo(loc), hi(loc), indirect_draw_initiator(modifier)]))
}

/// `sceAgcDcbDrawIndexIndirect(cb, dataOffset, modifier)`.
fn dcb_draw_index_indirect(ctx: &mut CpuContext) -> HleResult {
    let (cb, offset, modifier) = (arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2));
    // The single indexed packet enables the first-index destination in
    // word 3, bit 28.
    let loc = indirect_patch_locations(modifier, true) | ((modifier & 2) << 59);
    ptr(emit(cb, &[pm4(5, op::DRAW_INDEX_INDIRECT, 0), offset, lo(loc), hi(loc), draw_index_initiator(modifier)]))
}

/// `sceAgcDcbDrawIndexIndirectMulti(cb, dataOffset, countFromMemory, count,
/// countAddr, stride, modifier)`.
fn dcb_draw_index_indirect_multi(ctx: &mut CpuContext) -> HleResult {
    let cb = arg(ctx, 0);
    let (offset, from_mem, count, count_addr, stride) =
        (arg(ctx, 1) as u32, arg(ctx, 2) as u32, arg(ctx, 3) as u32, arg(ctx, 4), arg(ctx, 5) as u32);
    let Some(modifier) = stack_arg(ctx, 1) else { return 0 };
    if from_mem > 1 {
        return 0;
    }
    let loc = indirect_patch_locations(modifier, true);
    let m = modifier as u32;
    // The multi-draw moves the first-index enable to word 4, bit 28.
    let control = indirect_draw_index_location(m) | ((m & 0x10) << 23) | ((m & 2) << 27) | (from_mem << 30) | ((m & 8) << 28);
    let ca = count_addr & !3;
    ptr(emit(
        cb,
        &[
            pm4(10, op::DRAW_INDEX_INDIRECT_MULTI, 0),
            offset,
            lo(loc),
            hi(loc),
            control,
            count,
            lo(ca),
            hi(ca),
            stride,
            draw_index_initiator(modifier),
        ],
    ))
}

fn dcb_draw_index_indirect_multi_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(10)
}

/// `sceAgcDcbDrawIndirectMulti(cb, dataOffset, countFromMemory, count,
/// countAddr, stride, modifier)`: DRAW_INDIRECT_MULTI bracketed by the
/// private begin/end tags (3 + 10 + 3 dwords, inferred).
fn dcb_draw_indirect_multi(ctx: &mut CpuContext) -> HleResult {
    use super::pm4::{DRAW_INDIRECT_MULTI_BEGIN_TAG, DRAW_INDIRECT_MULTI_END_TAG, INTERNAL_DATA_REGISTER};
    let cb = arg(ctx, 0);
    let (offset, from_mem, count, count_addr, stride) =
        (arg(ctx, 1) as u32, arg(ctx, 2) as u32, arg(ctx, 3) as u32, arg(ctx, 4), arg(ctx, 5) as u32);
    let Some(modifier) = stack_arg(ctx, 1) else { return 0 };
    let m = modifier as u32;
    let loc = indirect_patch_locations(modifier, false);
    let control = indirect_draw_index_location(m) | ((m & 0x10) << 23) | ((from_mem & 1) << 30) | ((m & 8) << 28);
    let tag = |t: u32| [pm4(3, op::SET_UCONFIG_REG, 1), INTERNAL_DATA_REGISTER, t];
    let mut w = Vec::with_capacity(16);
    w.extend_from_slice(&tag(DRAW_INDIRECT_MULTI_BEGIN_TAG));
    w.extend_from_slice(&[
        pm4(10, op::DRAW_INDIRECT_MULTI, 0),
        offset,
        lo(loc),
        hi(loc),
        control,
        count,
        (count_addr as u32) & !3,
        hi(count_addr),
        stride,
        indirect_draw_initiator(modifier),
    ]);
    w.extend_from_slice(&tag(DRAW_INDIRECT_MULTI_END_TAG));
    ptr(emit(cb, &w))
}

fn dcb_draw_indirect_multi_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(16)
}

/// `sceAgcDcbDrawIndexMultiInstanced(cb, count, indexAddr, objectIds,
/// instances, modifier)`: DISPATCH_DRAW_PREAMBLE form.
fn dcb_draw_index_multi_instanced(ctx: &mut CpuContext) -> HleResult {
    let (cb, count, addr, ids, inst, modifier) =
        (arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2), arg(ctx, 3), arg(ctx, 4) as u32, arg(ctx, 5));
    if addr == 0 || ids == 0 || addr & 1 != 0 {
        return 0;
    }
    ptr(emit(
        cb,
        &[
            pm4(9, op::DISPATCH_DRAW_PREAMBLE, 0),
            count,
            lo(addr),
            hi(addr),
            inst.max(1),
            lo(ids),
            hi(ids),
            inst,
            draw_index_initiator(modifier) | 0x80,
        ],
    ))
}

fn dcb_draw_index_multi_instanced_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(9)
}

/// `sceAgcDcbSetBaseIndirectArgs(cb, shaderType, addr)`: SET_BASE index 1;
/// the header's shader-type bit selects graphics (0) or compute (1)
/// arguments.
fn set_base_packet(cb: u64, shader_type: u32, addr: u64) -> u64 {
    emit(cb, &[pm4(4, op::SET_BASE, 0) | (shader_type << 1), 1, (addr as u32) & !7, hi(addr)])
}

fn dcb_set_base_indirect_args(ctx: &mut CpuContext) -> HleResult {
    ptr(set_base_packet(arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2)))
}

/// `sceAgcDcbSetIndexIndirectArgs(cb, attributeAddress, attributeIndex)`:
/// INDEX_ATTRIBUTES_INDIRECT `[addr lo, addr hi, index]`, the console's
/// packet. The table of 16-byte attribute records must be 16-byte aligned
/// (a null base is written as 0) and the index fit 16 bits.
fn dcb_set_index_indirect_args(ctx: &mut CpuContext) -> HleResult {
    let (cb, addr, index) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2));
    if addr & 0xF != 0 || index > 0xFFFF {
        return 0;
    }
    ptr(emit(cb, &[pm4(4, op::INDEX_ATTRIBUTES_INDIRECT, 0), lo(addr), hi(addr), index as u32]))
}

fn four_dword_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(4)
}

/// `sceAgcDcbDispatchIndirect(cb, dataOffset, modifier)`.
fn dcb_dispatch_indirect(ctx: &mut CpuContext) -> HleResult {
    let (cb, offset, m) = (arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2) as u32);
    ptr(emit(cb, &[pm4(3, op::DISPATCH_INDIRECT, 0), offset, dispatch_initiator(m)]))
}

fn dcb_dispatch_indirect_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(3)
}

/// `sceAgcAcbDispatchIndirect(cb, argsAddr, modifier)`.
fn acb_dispatch_indirect(ctx: &mut CpuContext) -> HleResult {
    let (cb, a, m) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2) as u32);
    ptr(emit(cb, &[pm4(4, op::DISPATCH_INDIRECT, 0), lo(a), hi(a), dispatch_initiator(m)]))
}

// ---------------------------------------------------------------------------
// Register writes
// ---------------------------------------------------------------------------

/// `(cb, {u32 offset, u32 value})` passed by value in RSI.
fn set_register_direct(ctx: &CpuContext, opcode: u32) -> u64 {
    let packed = arg(ctx, 1);
    emit(arg(ctx, 0), &[pm4(3, opcode, 0), (packed as u32) & 0xFFFF, hi(packed)])
}

fn dcb_set_sh_register_direct(ctx: &mut CpuContext) -> HleResult {
    ptr(set_register_direct(ctx, op::SET_SH_REG))
}

fn dcb_set_uc_register_direct(ctx: &mut CpuContext) -> HleResult {
    ptr(set_register_direct(ctx, op::SET_UCONFIG_REG))
}

fn dcb_set_cx_register_direct(ctx: &mut CpuContext) -> HleResult {
    ptr(set_register_direct(ctx, op::SET_CONTEXT_REG))
}

fn three_dword_get_size(_: &mut CpuContext) -> HleResult {
    size_bytes(3)
}

/// `(cb, regsAddr, count)`: native SET_*_REG_INDIRECT table packet.
fn set_registers_indirect(ctx: &CpuContext, opcode: u32) -> u64 {
    let (cb, regs, n) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2) as u32);
    emit(cb, &[pm4(5, opcode, 0), (regs as u32) & !3, hi(regs), 0x8000_0000, n & 0x3FFF])
}

fn dcb_set_cx_registers_indirect(ctx: &mut CpuContext) -> HleResult {
    ptr(set_registers_indirect(ctx, op::SET_CONTEXT_REG_INDIRECT))
}

fn dcb_set_sh_registers_indirect(ctx: &mut CpuContext) -> HleResult {
    ptr(set_registers_indirect(ctx, op::SET_SH_REG_INDIRECT))
}

fn dcb_set_uc_registers_indirect(ctx: &mut CpuContext) -> HleResult {
    ptr(set_registers_indirect(ctx, op::SET_UCONFIG_REG_INDIRECT))
}

/// `(cb, {offset, value}[], count)`: runs of consecutive offsets become one
/// SET packet each (allocated run by run). Returns the first packet.
fn set_registers_direct(ctx: &CpuContext, opcode: u32) -> u64 {
    let (cb, regs, n) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2) as u32);
    // An empty list writes nothing and answers the cursor (the table may
    // be null).
    if n == 0 && cb != 0 {
        return mem::r64(cb + super::CB_CURSOR_UP).unwrap_or(0);
    }
    if n == 0 || cb == 0 || regs == 0 {
        return 0;
    }
    if !mem::accessible(regs, n as u64 * 8, false) {
        return 0;
    }
    // The {offset, value} table may sit at any byte address: each entry is
    // read byte for byte, never as aligned words.
    let entry = |i: u32| {
        let e = mem::r64(regs + i as u64 * 8).unwrap_or(0);
        (e as u32, (e >> 32) as u32)
    };
    let mut first = 0u64;
    let mut start = 0u32;
    while start < n {
        let (offset, _) = entry(start);
        let mut end = start + 1;
        while end < n && entry(end).0 == offset.wrapping_add(end - start) {
            end += 1;
        }
        let run = end - start;
        let Some(a) = alloc_dwords(cb, run + 2) else {
            // A run that does not fit is skipped; later runs still go out.
            start = end;
            continue;
        };
        if !mem::write_dwords(a, &[pm4(run + 2, opcode, 0), offset & 0xFFFF]) {
            return first;
        }
        for i in 0..run {
            if !mem::w32(a + 8 + i as u64 * 4, entry(start + i).1) {
                return first;
            }
        }
        if first == 0 {
            first = a;
        }
        start = end;
    }
    first
}

fn cb_set_sh_registers_direct(ctx: &mut CpuContext) -> HleResult {
    ptr(set_registers_direct(ctx, op::SET_SH_REG))
}

fn cb_set_uc_registers_direct(ctx: &mut CpuContext) -> HleResult {
    ptr(set_registers_direct(ctx, op::SET_UCONFIG_REG))
}

fn registers_direct_get_size(ctx: &mut CpuContext) -> HleResult {
    size_bytes((arg(ctx, 0) as u32).min(4096) * 3)
}

/// `sceAgcCbSetShRegisterRangeDirect(cb, offset, values, count)`: one
/// SET_SH_REG with `count` values. A null `values` reserves the payload for
/// the caller.
fn cb_set_sh_register_range_direct(ctx: &mut CpuContext) -> HleResult {
    ptr(set_register_range(ctx, op::SET_SH_REG))
}

fn cb_set_sh_register_range_direct_get_size(ctx: &mut CpuContext) -> HleResult {
    size_bytes(arg(ctx, 0) as u32 + 2)
}

/// `(cb, offset, values, count)`: one SET packet with `count` values.
fn set_register_range(ctx: &CpuContext, opcode: u32) -> u64 {
    let (cb, offset, values, n) = (arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2), arg(ctx, 3) as u32);
    if n > 0x3FFE {
        return 0;
    }
    let Some(a) = alloc_dwords(cb, n + 2) else { return 0 };
    if !mem::write_dwords(a, &[pm4(n + 2, opcode, 0), offset & 0xFFFF]) {
        return 0;
    }
    if values != 0 && !mem::copy(a + 8, values, n as u64 * 4) {
        return 0;
    }
    a
}

fn cb_set_uc_register_range_direct(ctx: &mut CpuContext) -> HleResult {
    ptr(set_register_range(ctx, op::SET_UCONFIG_REG))
}

fn cb_set_uc_register_range_direct_get_size(ctx: &mut CpuContext) -> HleResult {
    let n = arg(ctx, 0) as u32;
    if n == 0 || n > 0x3FFE {
        0
    } else {
        size_bytes(n + 2)
    }
}

/// SET_CONFIG_REG run `(cb, offset, values, count)`: 1..=0x3FFF values
/// that stay inside the 16-bit register space; a null `values` reserves
/// the payload without writing it.
fn dcb_set_cf_register_range_direct(ctx: &mut CpuContext) -> HleResult {
    let (offset, n) = (arg(ctx, 1), arg(ctx, 3) as u32);
    // The values are read byte for byte: any address is accepted.
    if n == 0 || n > 0x3FFF || offset > 0xFFFF || n as u64 > 0x1_0000 - offset {
        return 0;
    }
    ptr(set_register_range(ctx, op::SET_CONFIG_REG))
}

/// `sceAgcDcbSetCfRegisterDirect(cb, {offset, value})`: one SET_CONFIG_REG.
fn dcb_set_cf_register_direct(ctx: &mut CpuContext) -> HleResult {
    if arg(ctx, 1) as u32 > 0xFFFF {
        return 0;
    }
    ptr(set_register_direct(ctx, op::SET_CONFIG_REG))
}

// ---------------------------------------------------------------------------
// Context state
// ---------------------------------------------------------------------------

/// Context-state push/pop/clear. The first packet carries the operation in
/// a CONTEXT_STATE NOP; the rest keep the native allocation sizes as plain
/// NOPs (clear 5; push/pop 5+8+9+3+2; push-clear adds 5).
fn dcb_context_state_op(ctx: &mut CpuContext) -> HleResult {
    let (cb, operation) = (arg(ctx, 0), arg(ctx, 1) as u32);
    if cb == 0 || context_state_dwords(operation) == 0 {
        return 0;
    }
    let mut first = 0u64;
    let mut append = |n: u32| -> bool {
        let mut w = vec![0u32; n as usize];
        if first == 0 {
            w[0] = pm4(n, op::NOP, nop::CONTEXT_STATE);
            w[1] = operation;
        } else {
            w[0] = pm4(n, op::NOP, nop::ZERO);
        }
        let a = emit(cb, &w);
        if a == 0 {
            return false;
        }
        if first == 0 {
            first = a;
        }
        true
    };
    let save_restore = |append: &mut dyn FnMut(u32) -> bool| -> bool {
        prepare_dwords(cb, 22, false).is_some() && append(5) && append(8) && append(9)
    };
    let ok = match operation {
        0 => append(5),
        1 => save_restore(&mut append) && append(3) && append(2),
        2 => append(3) && save_restore(&mut append) && append(2),
        3 => save_restore(&mut append) && append(3) && append(2) && append(5),
        _ => false,
    };
    ptr(if ok { first } else { 0 })
}

fn dcb_context_state_op_get_size(ctx: &mut CpuContext) -> HleResult {
    size_bytes(context_state_dwords(arg(ctx, 0) as u32))
}

// ---------------------------------------------------------------------------
// Export table
// ---------------------------------------------------------------------------

pub(crate) const EXPORTS: &[Export] = &[
    // Generic
    ("LtTouSCZjHM", "sceAgcCbNop", LIB_AGC, cb_nop),
    ("t7PlZ9nt5Lc", "sceAgcCbNopGetSize", LIB_AGC, cb_nop_get_size),
    ("k3GhuSNmBLU", "sceAgcCbDispatch", LIB_AGC, cb_dispatch),
    ("Abendgtz+3o", "sceAgcCbDispatchGetSize", LIB_AGC, cb_dispatch_get_size),
    ("JrtiDtKeS38", "sceAgcAcbResetQueue", LIB_AGC, acb_reset_queue),
    ("TRO721eVt4g", "sceAgcDcbResetQueue", LIB_AGC, dcb_reset_queue),
    ("u2T2DiA5hRI", "sceAgcDcbStallCommandBufferParser", LIB_AGC, dcb_stall_parser),
    ("+u6dKSLWM2o", "sceAgcDcbStallCommandBufferParserGetSize", LIB_AGC, dcb_stall_parser_get_size),
    // Acquire / release / write / wait
    ("KT-hTp-Ch14", "sceAgcAcbAcquireMem", LIB_AGC, acb_acquire_mem),
    ("ewobAQeMo5k", "sceAgcAcbAcquireMemGetSize", LIB_AGC, acquire_mem_get_size),
    ("57labkp+rSQ", "sceAgcDcbAcquireMem", LIB_AGC, dcb_acquire_mem),
    ("-vnlTPPXPrw", "sceAgcDcbAcquireMemGetSize", LIB_AGC, acquire_mem_get_size),
    ("VmW0Tdpy420", "sceAgcDcbWaitRegMem", LIB_AGC, dcb_wait_reg_mem),
    ("htn36gPnBk4", "sceAgcAcbWaitRegMem", LIB_AGC, acb_wait_reg_mem),
    ("43WJ08sSugE", "sceAgcDcbWaitOnAddressGetSize", LIB_AGC, wait_on_address_get_size),
    ("idlaArvdXEs", "sceAgcAcbWaitOnAddressGetSize", LIB_AGC, wait_on_address_get_size),
    ("i1jyy49AjXU", "sceAgcDcbWriteData", LIB_AGC, dcb_write_data),
    ("eZ4+17OQz4Q", "sceAgcAcbWriteData", LIB_AGC, acb_write_data),
    ("p9tI+yTvx68", "sceAgcDcbWriteDataGetSize", LIB_AGC, dcb_write_data_get_size),
    ("wr23dPKyWc0", "sceAgcCbReleaseMem", LIB_AGC, cb_release_mem),
    ("hL7C0IRpWZI", "sceAgcCbQueueEndOfPipeActionGetSize", LIB_AGC, cb_queue_end_of_pipe_action_get_size),
    ("zg6u-N6Otxs", "sceAgcDcbQueueEndOfShaderActionGetSize", LIB_AGC, queue_end_of_shader_action_get_size),
    ("F8NLhWvFemI", "sceAgcAcbQueueEndOfShaderActionGetSize", LIB_AGC, queue_end_of_shader_action_get_size),
    ("7toV+elXqNM", "sceAgcCbCondWrite", LIB_AGC, cb_cond_write),
    ("FuVbkyKlf+s", "sceAgcCbCondWriteGetSize", LIB_AGC, cb_cond_write_get_size),
    // LOD stats, markers, workloads
    ("vuSXe69VILM", "sceAgcDcbGetLodStats", LIB_AGC, dcb_get_lod_stats),
    ("rUuVjyR+Rd4", "sceAgcDcbGetLodStatsGetSize", LIB_AGC, dcb_get_lod_stats_get_size),
    ("+kSrjIVxKFE", "sceAgcDcbPushMarker", LIB_AGC, push_marker),
    ("cpCILPya5Zk", "sceAgcAcbPushMarker", LIB_AGC, push_marker),
    ("H7uZqCoNuWk", "sceAgcDcbPopMarker", LIB_AGC, pop_marker),
    ("6mFxkVqdmbQ", "sceAgcAcbPopMarker", LIB_AGC, pop_marker),
    ("QhCbS4X9Rl8", "sceAgcDcbSetMarker", LIB_AGC, set_marker),
    ("xAeBOa0A3kk", "sceAgcAcbSetMarker", LIB_AGC, set_marker),
    ("LFSPFmGc9Hg", "sceAgcDcbSetWorkloadsActive", LIB_AGC, set_workloads_active),
    ("rVOmPz2RBlg", "sceAgcAcbSetWorkloadsActive", LIB_AGC, set_workloads_active),
    ("hEK26Wdny6s", "sceAgcDcbSetWorkloadComplete", LIB_AGC, set_workload_complete),
    ("opR1JeJZCBU", "sceAgcAcbSetWorkloadComplete", LIB_AGC, set_workload_complete),
    // Control flow
    ("zfcxg-ewMK8", "sceAgcDcbRewind", LIB_AGC, dcb_rewind),
    ("DwICrVxerkY", "sceAgcAcbRewind", LIB_AGC, acb_rewind),
    ("QIXCsbipds0", "sceAgcDcbRewindGetSize", LIB_AGC, rewind_get_size),
    ("0ZOG0jc9nRg", "sceAgcAcbRewindGetSize", LIB_AGC, rewind_get_size),
    ("xSAR0LTcRKM", "sceAgcDcbJump", LIB_AGC, dcb_jump),
    ("VEGu4dixjUg", "sceAgcDcbJumpGetSize", LIB_AGC, jump_get_size),
    ("e1DFTg+Sd8U", "sceAgcAcbJump", LIB_AGC, acb_jump),
    ("b-oySn+G2tE", "sceAgcAcbJumpGetSize", LIB_AGC, jump_get_size),
    ("w1KFAHVqpaU", "sceAgcCbBranch", LIB_AGC, cb_branch),
    ("uZW-mqsxkrM", "sceAgcCbBranchGetSize", LIB_AGC, cb_branch_get_size),
    ("BIPexNBSGog", "sceAgcDcbCondExec", LIB_AGC, cond_exec),
    ("ou16V5hh5sg", "sceAgcDcbCondExecGetSize", LIB_AGC, cond_exec_get_size),
    ("qyM2bxYFPAk", "sceAgcAcbCondExec", LIB_AGC, cond_exec),
    ("ozKzBP4aki4", "sceAgcAcbCondExecGetSize", LIB_AGC, cond_exec_get_size),
    ("bbFueFP+J4k", "sceAgcDcbSetPredication", LIB_AGC, dcb_set_predication),
    ("yheJGN-ay+A", "sceAgcDcbSetBoolPredicationEnableGetSize", LIB_AGC, predication_get_size),
    ("XN+Iuu7XsM8", "sceAgcDcbSetZPassPredicationEnableGetSize", LIB_AGC, predication_get_size),
    ("vLrBL8DQiz8", "sceAgcDcbSetPredicationDisableGetSize", LIB_AGC, predication_get_size),
    // DMA / copy / atomics / semaphores
    ("1rZSWUv1IRc", "sceAgcDcbCopyData", LIB_AGC, dcb_copy_data),
    ("b5u0Jzm8TF8", "sceAgcDcbCopyDataGetSize", LIB_AGC, copy_data_get_size),
    ("qzMN2XKGA4k", "sceAgcAcbCopyData", LIB_AGC, acb_copy_data),
    ("CbQh3DKMSno", "sceAgcAcbCopyDataGetSize", LIB_AGC, copy_data_get_size),
    ("WmAc2MEj6Io", "sceAgcDcbDmaData", LIB_AGC, dcb_dma_data),
    ("2ccJz9LQI+w", "sceAgcDcbDmaDataGetSize", LIB_AGC, dma_data_get_size),
    ("-RnpfpxIhec", "sceAgcAcbDmaData", LIB_AGC, acb_dma_data),
    ("M0ttm8h7SKA", "sceAgcAcbDmaDataGetSize", LIB_AGC, dma_data_get_size),
    ("1-gUn1PI4Sw", "sceAgcDcbAtomicMem", LIB_AGC, dcb_atomic_mem),
    ("oz6zQq1JwCE", "sceAgcDcbAtomicMemGetSize", LIB_AGC, atomic_mem_get_size),
    ("XKKuA6VkSRc", "sceAgcAcbAtomicMem", LIB_AGC, acb_atomic_mem),
    ("da1Sm8-QDoU", "sceAgcAcbAtomicMemGetSize", LIB_AGC, atomic_mem_get_size),
    ("1tB0xkLNjcw", "sceAgcDcbAtomicGdsGetSize", LIB_AGC, atomic_gds_get_size),
    ("gQkqkLttcpw", "sceAgcAcbAtomicGds_0900", LIB_AGC, acb_atomic_gds),
    ("zARR5aCmkoY", "sceAgcDcbAtomicGds_0900", LIB_AGC, acb_atomic_gds),
    ("pH3-dfRpfA0", "sceAgcDcbAtomicGds", LIB_AGC, acb_atomic_gds),
    ("vHX9guneRBY", "sceAgcCbMemSemaphore", LIB_AGC, cb_mem_semaphore),
    ("FneFypEDRgY", "sceAgcDcbSetWorkloadStreamInactive", LIB_AGC, set_workload_stream_inactive),
    ("FcgdDM3MB+k", "sceAgcAcbSetWorkloadStreamInactive", LIB_AGC, set_workload_stream_inactive),
    ("hcIxS8pmXF4", "sceAgcAcbAtomicGdsGetSize", LIB_AGC, atomic_gds_get_size),
    ("G0jrLdvEqDw", "sceAgcDcbMemSemaphore", LIB_AGC, mem_semaphore),
    ("q4VuU-QsLOE", "sceAgcAcbMemSemaphore", LIB_AGC, mem_semaphore),
    ("jt3pl7EN17o", "sceAgcDcbPrimeUtcl2", LIB_AGC, prime_utcl2),
    ("KjPeVduz6jU", "sceAgcDcbPrimeUtcl2GetSize", LIB_AGC, prime_utcl2_get_size),
    ("szG7hz2yEhA", "sceAgcAcbPrimeUtcl2", LIB_AGC, prime_utcl2),
    ("eCjKaqeeQ5s", "sceAgcAcbPrimeUtcl2GetSize", LIB_AGC, prime_utcl2_get_size),
    // Events, flips
    ("aJf+j5yntiU", "sceAgcDcbEventWrite", LIB_AGC, event_write),
    ("C4l9fB17t8w", "sceAgcDcbEventWriteGetSize", LIB_AGC, event_write_get_size),
    ("cFazmnXpJOE", "sceAgcAcbEventWrite", LIB_AGC, acb_event_write),
    ("Y-5vneiBtzk", "sceAgcAcbEventWriteGetSize", LIB_AGC, acb_event_write_get_size),
    ("ms1xVoZ-Vwc", "sceAgcDcbBeginOcclusionQueryGetSize", LIB_AGC, occlusion_query_get_size),
    ("P1CugZ99Uzc", "sceAgcDcbEndOcclusionQueryGetSize", LIB_AGC, occlusion_query_get_size),
    ("MWiElSNE8j8", "sceAgcDcbWaitUntilSafeForRendering", LIB_AGC, wait_until_safe_for_rendering),
    ("GPbUp9jXQa8", "sceAgcAcbWaitUntilSafeForRendering", LIB_AGC, wait_until_safe_for_rendering),
    ("YUeqkyT7mEQ", "sceAgcDcbSetFlip", LIB_AGC, set_flip),
    ("ebixW91gpPw", "sceAgcAcbSetFlip", LIB_AGC, set_flip),
    // Index / draw state, draws, dispatches
    ("GIIW2J37e70", "sceAgcDcbSetIndexSize", LIB_AGC, dcb_set_index_size),
    ("ca4KPvp0qLQ", "sceAgcDcbSetIndexSizeGetSize", LIB_AGC, dcb_set_index_size_get_size),
    ("-KRzWekV120", "sceAgcDcbSetIndexSize_0600", LIB_AGC, dcb_set_index_size_ex),
    ("8N2tmT3jmC8", "sceAgcDcbSetIndexCount", LIB_AGC, dcb_set_index_count),
    ("mljzuGDZRQ4", "sceAgcDcbSetIndexCountGetSize", LIB_AGC, two_dword_get_size),
    ("tSBxhAPyytQ", "sceAgcDcbSetNumInstances", LIB_AGC, dcb_set_num_instances),
    ("6DFuRKT4C9w", "sceAgcDcbSetNumInstancesGetSize", LIB_AGC, two_dword_get_size),
    ("l4fM9K-Lyks", "sceAgcDcbSetIndexBuffer", LIB_AGC, dcb_set_index_buffer),
    ("j4emHHndCPY", "sceAgcDcbSetIndexBufferGetSize", LIB_AGC, dcb_set_index_buffer_get_size),
    ("q88lQ+GP5Yk", "sceAgcDcbDrawIndex", LIB_AGC, dcb_draw_index),
    ("6ee9Hd3EWXQ", "sceAgcDcbDrawIndexGetSize", LIB_AGC, dcb_draw_index_get_size),
    ("Yw0jKSqop+E", "sceAgcDcbDrawIndexAuto", LIB_AGC, dcb_draw_index_auto),
    ("WrdP9Zxx3lQ", "sceAgcDcbDrawIndexAutoGetSize", LIB_AGC, dcb_draw_index_auto_get_size),
    ("B+aG9DUnTKA", "sceAgcDcbDrawIndexOffset", LIB_AGC, dcb_draw_index_offset),
    ("qMlfB1ZhMDc", "sceAgcDcbDrawIndexOffsetGetSize", LIB_AGC, five_dword_get_size),
    ("1q1titRBL6o", "sceAgcDcbDrawIndirect", LIB_AGC, dcb_draw_indirect),
    ("cxPZ4Wgvdj8", "sceAgcDcbDrawIndirectGetSize", LIB_AGC, five_dword_get_size),
    ("t1vNu082-jM", "sceAgcDcbDrawIndexIndirect", LIB_AGC, dcb_draw_index_indirect),
    ("mStuvI0zOtc", "sceAgcDcbDrawIndexIndirectGetSize", LIB_AGC, five_dword_get_size),
    ("ypVBz4uPKcQ", "sceAgcDcbDrawIndexIndirectMulti", LIB_AGC, dcb_draw_index_indirect_multi),
    ("r98I08t+LOg", "sceAgcDcbDrawIndexIndirectMultiGetSize", LIB_AGC, dcb_draw_index_indirect_multi_get_size),
    ("kUlvghKs-mA", "sceAgcDcbDrawIndirectMulti", LIB_AGC, dcb_draw_indirect_multi),
    ("pYoKs3lPy88", "sceAgcDcbDrawIndirectMultiGetSize", LIB_AGC, dcb_draw_indirect_multi_get_size),
    ("Rlx+bykm0r0", "sceAgcDcbDrawIndexMultiInstanced", LIB_AGC, dcb_draw_index_multi_instanced),
    ("mR9j7+SfM34", "sceAgcDcbDrawIndexMultiInstancedGetSize", LIB_AGC, dcb_draw_index_multi_instanced_get_size),
    ("RmaJwLtc8rY", "sceAgcDcbSetBaseIndirectArgs", LIB_AGC, dcb_set_base_indirect_args),
    ("MMlmJAL7N5w", "sceAgcDcbSetBaseDrawIndirectArgsGetSize", LIB_AGC, four_dword_get_size),
    ("9S4noWrUI0s", "sceAgcDcbSetBaseDispatchIndirectArgsGetSize", LIB_AGC, four_dword_get_size),
    ("0o3VDdtA6nM", "sceAgcDcbSetIndexIndirectArgs", LIB_AGC, dcb_set_index_indirect_args),
    ("AFIh8SQkYlQ", "sceAgcDcbSetIndexIndirectArgsGetSize", LIB_AGC, four_dword_get_size),
    ("CtB+A9-VxO0", "sceAgcDcbDispatchIndirect", LIB_AGC, dcb_dispatch_indirect),
    ("w8HVkEeXPv8", "sceAgcDcbDispatchIndirectGetSize", LIB_AGC, dcb_dispatch_indirect_get_size),
    ("j3EtxFkSIhQ", "sceAgcAcbDispatchIndirect", LIB_AGC, acb_dispatch_indirect),
    ("PxKWV2fVAps", "sceAgcAcbDispatchIndirectGetSize", LIB_AGC, four_dword_get_size),
    // Register writes
    ("pFLArOT53+w", "sceAgcDcbSetShRegisterDirect", LIB_AGC, dcb_set_sh_register_direct),
    ("QhPDD513V0w", "sceAgcDcbSetShRegisterDirectGetSize", LIB_AGC, three_dword_get_size),
    ("w4-d0n60hdo", "sceAgcDcbSetUcRegisterDirect", LIB_AGC, dcb_set_uc_register_direct),
    ("aP1Ki9G3++4", "sceAgcDcbSetUcRegisterDirectGetSize", LIB_AGC, three_dword_get_size),
    ("LHFXRrlTPD8", "sceAgcDcbSetCxRegisterDirect", LIB_AGC, dcb_set_cx_register_direct),
    ("1DeUNpRIDDA", "sceAgcDcbSetCxRegisterDirectGetSize", LIB_AGC, three_dword_get_size),
    ("ZvwO9euwYzc", "sceAgcDcbSetCxRegistersIndirect", LIB_AGC, dcb_set_cx_registers_indirect),
    ("GBCh3zCihoU", "sceAgcDcbSetCxRegistersIndirectGetSize", LIB_AGC, five_dword_get_size),
    ("-HOOCn0JY48", "sceAgcDcbSetShRegistersIndirect", LIB_AGC, dcb_set_sh_registers_indirect),
    ("nNlUtdDDvZ0", "sceAgcDcbSetShRegistersIndirectGetSize", LIB_AGC, five_dword_get_size),
    ("hvUfkUIQcOE", "sceAgcDcbSetUcRegistersIndirect", LIB_AGC, dcb_set_uc_registers_indirect),
    ("UQGTw4xRlcM", "sceAgcDcbSetUcRegistersIndirectGetSize", LIB_AGC, five_dword_get_size),
    ("UZbQjYAwwXM", "sceAgcCbSetShRegistersDirect", LIB_AGC, cb_set_sh_registers_direct),
    ("yUBESvCCJ4I", "sceAgcCbSetShRegistersDirectGetSize", LIB_AGC, registers_direct_get_size),
    ("03RZmELWWzw", "sceAgcCbSetUcRegistersDirect", LIB_AGC, cb_set_uc_registers_direct),
    ("TGEZzUWLbrc", "sceAgcCbSetUcRegistersDirectGetSize", LIB_AGC, registers_direct_get_size),
    ("n2fD4A+pb+g", "sceAgcCbSetShRegisterRangeDirect", LIB_AGC, cb_set_sh_register_range_direct),
    ("bxGoVxpdSPQ", "sceAgcCbSetShRegisterRangeDirectGetSize", LIB_AGC, cb_set_sh_register_range_direct_get_size),
    ("MDLD5Ly94Xk", "sceAgcCbSetUcRegisterRangeDirect", LIB_AGC, cb_set_uc_register_range_direct),
    ("JOWmDrl+j20", "sceAgcCbSetUcRegisterRangeDirectGetSize", LIB_AGC, cb_set_uc_register_range_direct_get_size),
    ("BVFg3CWU6Eo", "sceAgcDcbSetCfRegisterRangeDirect", LIB_AGC, dcb_set_cf_register_range_direct),
    ("73ZZdojLIgs", "sceAgcDcbSetCfRegisterDirect", LIB_AGC, dcb_set_cf_register_direct),
    ("PxEFhy0d5v8", "sceAgcDcbClearState", LIB_AGC, dcb_clear_state),
    // Context state
    ("qj7QZpgr9Uw", "sceAgcDcbContextStateOp_0100", LIB_AGC, dcb_context_state_op),
    ("HabmgqPwPw0", "sceAgcDcbContextStateOp", LIB_AGC, dcb_context_state_op),
    ("H6vHS5cidSA", "sceAgcDcbContextStateOpGetSize", LIB_AGC, dcb_context_state_op_get_size),
];
