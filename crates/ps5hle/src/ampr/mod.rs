// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceAmpr: AMPR command buffers (APR file reads, AMM address mapping)
//! and the libkernel APR submission calls.
//!
//! UE4 streams pak files through APR: it builds a command buffer of
//! read-file commands (plus write-address / kernel-event completion
//! commands), submits it with `sceKernelAprSubmitCommandBuffer*` and waits
//! for the completion it requested. Commands are recorded on the host
//! ([`command_buffer`]) and run asynchronously by the submission engines
//! ([`engine`]); AMM map commands are executed by [`amm`]. File ids are the
//! ones handed out by `sceKernelAprResolveFilepathsToIds*`
//! ([`crate::filesystem::apr_host_path`]).
//!
//! Return values: 0 or a value in RAX on success, `0x8002xxxx` errors
//! otherwise. Measure functions returning `uint64_t` report errors as the
//! zero-extended code; the AMM/APR map measures return it sign-extended.
//! The libkernel APR calls report errors as -1 with errno.
//!
//! Commands this implementation does not model beyond their stream space:
//! markers, nops, APR map brackets, write-counter-on-completion. The
//! write-address-from-{counter, counter pair, time counter} commands store
//! 0 at completion.

pub mod amm;
pub mod command_buffer;
pub mod engine;

use crate::context::{CpuContext, CpuRegister};
use crate::registry::{HleRegistry, HleResult};
use crate::sync::{ERR_BUSY, ERR_INVALID_ARGUMENT, ERR_MEMORY_FAULT, ERR_PERMISSION_DENIED, ERR_TRY_AGAIN, OK};
use command_buffer::{self as cbuf, AmmKind, AmmMap, CommandData};
use engine::Engine;

const LIB: &str = "libSceAmpr";

/// Error of a `uint64_t` measure function: the zero-extended code.
const MEASURE_INVALID: HleResult = ERR_INVALID_ARGUMENT as u32 as HleResult;

fn trace(op: &str, cb: u64, a: u64, b: u64) {
    log::trace!("[ampr] {op} cb={cb:#x} {a:#x} {b:#x}");
}

/// Integer argument `n` passed on the stack (0 = seventh argument).
fn stack_arg(ctx: &CpuContext, n: u64) -> u64 {
    ctx.read_u64(ctx.get(CpuRegister::Rsp) + 8 + n * 8).unwrap_or(0)
}

fn ok_or(ok: bool, err: HleResult) -> HleResult {
    if ok {
        OK
    } else {
        err
    }
}

/// Length of the NUL-terminated label at `addr` (None for NULL).
fn label_len(ctx: &CpuContext, addr: u64) -> Option<usize> {
    if addr == 0 {
        return None;
    }
    Some(ctx.read_cstr_bytes(addr, 1 << 20).map_or(0, |b| b.len()))
}

/// Append `size` (rounded up to 4) bytes of stream space.
fn append_noop(ctx: &CpuContext, cb: u64, size: u64) -> HleResult {
    if cb == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    ok_or(cbuf::append(ctx, cb, cbuf::align4(size), None), ERR_MEMORY_FAULT)
}

// ---------------------------------------------------------------------------
// Construction / buffer management
// ---------------------------------------------------------------------------

/// (cb) — zero the header and forget host state.
fn cb_constructor(ctx: &mut CpuContext) -> HleResult {
    let cb = ctx.rdi();
    if cb == 0 {
        return OK;
    }
    trace("ctor", cb, 0, 0);
    ok_or(cbuf::initialize(ctx, cb), ERR_MEMORY_FAULT)
}

/// (cb, reservedState0*, reservedState1*) — zero the two APR state words
/// (in the object unless pointers are given).
fn cb_apr_constructor(ctx: &mut CpuContext) -> HleResult {
    let (cb, s0, s1) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if cb == 0 {
        return OK;
    }
    let s0 = if s0 != 0 { s0 } else { cb + cbuf::OFF_APR_MAP };
    let s1 = if s1 != 0 { s1 } else { cb + cbuf::OFF_APR_SG };
    trace("apr_ctor", cb, s0, s1);
    ok_or(ctx.write_u64(s0, 0) && (s0 == s1 || ctx.write_u64(s1, 0)), ERR_MEMORY_FAULT)
}

/// (cb)
fn cb_noop_destructor(ctx: &mut CpuContext) -> HleResult {
    trace("dtor (no state)", ctx.rdi(), 0, 0);
    OK
}

/// (cb) — forget the host state.
fn cb_destructor(ctx: &mut CpuContext) -> HleResult {
    let cb = ctx.rdi();
    if cb != 0 {
        cbuf::erase(cb);
        trace("dtor", cb, 0, 0);
    }
    OK
}

/// (cb, buffer, uint32 size): attach 4-byte aligned command memory to an
/// object that has none.
fn cb_set_buffer(ctx: &mut CpuContext) -> HleResult {
    let (cb, buffer, size) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u32 as u64);
    if cb == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(old) = ctx.read_u64(cb + cbuf::OFF_DATA) else { return ERR_MEMORY_FAULT };
    if buffer == 0 || buffer & 3 != 0 || size == 0 || size > cbuf::BUFFER_SIZE_MAX || size & 3 != 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if old != 0 {
        return ERR_BUSY;
    }
    trace("set_buffer", cb, buffer, size);
    ok_or(cbuf::write_pointers(ctx, cb, buffer, size) && ctx.write_u32(cb + cbuf::OFF_NUM, 0), ERR_MEMORY_FAULT)
}

/// (cb) — rewind to offset 0 keeping the buffer.
fn cb_reset(ctx: &mut CpuContext) -> HleResult {
    let cb = ctx.rdi();
    if cb == 0 {
        return ERR_PERMISSION_DENIED;
    }
    let Some(st) = cbuf::try_get_state(ctx, cb) else { return ERR_MEMORY_FAULT };
    if st.buffer == 0 {
        return ERR_PERMISSION_DENIED;
    }
    trace("reset", cb, st.buffer, st.size);
    ok_or(cbuf::write_pointers(ctx, cb, st.buffer, st.size) && ctx.write_u32(cb + cbuf::OFF_NUM, 0), ERR_MEMORY_FAULT)
}

/// (cb) -> the detached buffer (NULL when there was none).
fn cb_clear_buffer(ctx: &mut CpuContext) -> HleResult {
    let cb = ctx.rdi();
    if cb == 0 {
        return 0;
    }
    let (Some(buffer), Some(size)) = (ctx.read_u64(cb + cbuf::OFF_DATA), ctx.read_u32(cb + cbuf::OFF_SIZE)) else {
        return 0;
    };
    if buffer == 0 || size == 0 || !cbuf::write_visible_pointers(ctx, cb, 0, 0) {
        return 0;
    }
    cbuf::erase(cb);
    trace("clear_buffer", cb, buffer, size as u64);
    buffer as HleResult
}

/// (cb) -> buffer size
fn cb_get_size(ctx: &mut CpuContext) -> HleResult {
    ctx.read_u32(ctx.rdi() + cbuf::OFF_SIZE).filter(|_| ctx.rdi() != 0).unwrap_or(0) as HleResult
}

/// (cb) -> write offset
fn cb_get_current_offset(ctx: &mut CpuContext) -> HleResult {
    ctx.read_u32(ctx.rdi() + cbuf::OFF_OFFSET).filter(|_| ctx.rdi() != 0).unwrap_or(0) as HleResult
}

/// (cb) -> number of commands
fn cb_get_num_commands(ctx: &mut CpuContext) -> HleResult {
    ctx.read_u32(ctx.rdi() + cbuf::OFF_NUM).filter(|_| ctx.rdi() != 0).unwrap_or(0) as HleResult
}

/// (cb) -> buffer base address
fn cb_get_buffer_base_address(ctx: &mut CpuContext) -> HleResult {
    ctx.read_u64(ctx.rdi() + cbuf::OFF_DATA).filter(|_| ctx.rdi() != 0).unwrap_or(0) as HleResult
}

/// (cb) -> type word (int)
fn cb_get_type(ctx: &mut CpuContext) -> HleResult {
    ctx.read_u32(ctx.rdi() + cbuf::OFF_TYPE).filter(|_| ctx.rdi() != 0).unwrap_or(0) as i32 as HleResult
}

// ---------------------------------------------------------------------------
// APR reads
// ---------------------------------------------------------------------------

/// (cb, -, -, fileId, dst, size, [stack] fileOffset)
fn cb_apr_read_file(ctx: &mut CpuContext) -> HleResult {
    let (cb, file_id, dst, size) = (ctx.rdi(), ctx.rcx() as u32, ctx.r8(), ctx.r9());
    let off = stack_arg(ctx, 0);
    if cb == 0 || !cbuf::valid_read_range(dst, size) || !cbuf::valid_file_offset(off) {
        return ERR_INVALID_ARGUMENT;
    }
    trace("read_file", cb, file_id as u64, off);
    let rs = cbuf::read_file_size(off);
    ok_or(cbuf::append_read(ctx, cb, cbuf::OP_READ_FILE, file_id, dst, size, off, rs), ERR_MEMORY_FAULT)
}

/// (cb, -, -, size, fileOffset): same file, destination following the
/// previous read.
fn cb_apr_read_file_gather(ctx: &mut CpuContext) -> HleResult {
    let (cb, size, off) = (ctx.rdi(), ctx.rcx(), ctx.r8());
    if cb == 0 || !cbuf::valid_read_size(size) || !cbuf::valid_file_offset(off) {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(st) = cbuf::try_get_state(ctx, cb) else { return ERR_MEMORY_FAULT };
    if !st.gs_valid || !cbuf::valid_read_range(st.gs_destination, size) {
        return ERR_INVALID_ARGUMENT;
    }
    let rs = cbuf::read_gather_size(off);
    let ok = cbuf::append_read(ctx, cb, cbuf::OP_READ_GATHER, st.gs_file_id, st.gs_destination, size, off, rs);
    ok_or(ok, ERR_MEMORY_FAULT)
}

/// (cb, -, -, dst, size): same file, file range following the previous read.
fn cb_apr_read_file_scatter(ctx: &mut CpuContext) -> HleResult {
    let (cb, dst, size) = (ctx.rdi(), ctx.rcx(), ctx.r8());
    if cb == 0 || !cbuf::valid_read_range(dst, size) {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(st) = cbuf::try_get_state(ctx, cb) else { return ERR_MEMORY_FAULT };
    if !st.gs_valid || !cbuf::valid_file_offset(st.gs_file_offset) {
        return ERR_INVALID_ARGUMENT;
    }
    let rs = cbuf::READ_SCATTER_SIZE;
    let ok = cbuf::append_read(ctx, cb, cbuf::OP_READ_SCATTER, st.gs_file_id, dst, size, st.gs_file_offset, rs);
    ok_or(ok, ERR_MEMORY_FAULT)
}

/// (cb, -, -, dst, size, fileOffset): same file as the previous read.
fn cb_apr_read_file_gather_scatter(ctx: &mut CpuContext) -> HleResult {
    let (cb, dst, size, off) = (ctx.rdi(), ctx.rcx(), ctx.r8(), ctx.r9());
    if cb == 0 || !cbuf::valid_read_range(dst, size) || !cbuf::valid_file_offset(off) {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(st) = cbuf::try_get_state(ctx, cb) else { return ERR_MEMORY_FAULT };
    if !st.gs_valid {
        return ERR_INVALID_ARGUMENT;
    }
    let rs = cbuf::read_gather_scatter_size(off);
    ok_or(cbuf::append_read(ctx, cb, cbuf::OP_READ_GATHER_SCATTER, st.gs_file_id, dst, size, off, rs), ERR_MEMORY_FAULT)
}

/// (cb, -, -): 4 bytes of stream space; clears the continuation.
fn cb_apr_reset_gather_scatter_state(ctx: &mut CpuContext) -> HleResult {
    let cb = ctx.rdi();
    if cb == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if append_noop(ctx, cb, cbuf::RESET_GATHER_SCATTER_SIZE) != OK {
        return ERR_MEMORY_FAULT;
    }
    cbuf::reset_gather_scatter(cb);
    cbuf::update_type_flags(ctx, cb, 0, cbuf::TYPE_GATHER_SCATTER_VALID);
    OK
}

/// (cb, va, size, type, prot)
fn cb_apr_map_begin(ctx: &mut CpuContext) -> HleResult {
    let (cb, va, size) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if cb == 0 || !cbuf::valid_map_args(va, size) {
        return ERR_INVALID_ARGUMENT;
    }
    if append_noop(ctx, cb, cbuf::APR_MAP_BEGIN_SIZE) != OK {
        return ERR_MEMORY_FAULT;
    }
    cbuf::update_type_flags(ctx, cb, cbuf::TYPE_MAP_ACTIVE, 0);
    OK
}

/// (cb, va, dmemOffset, size, type, prot)
fn cb_apr_map_direct_begin(ctx: &mut CpuContext) -> HleResult {
    let (cb, va, dmem, size) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    if cb == 0 || !cbuf::valid_map_args(va, size) || dmem % cbuf::AMM_PAGE_SIZE != 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if append_noop(ctx, cb, cbuf::APR_MAP_DIRECT_BEGIN_SIZE) != OK {
        return ERR_MEMORY_FAULT;
    }
    cbuf::update_type_flags(ctx, cb, cbuf::TYPE_MAP_ACTIVE, 0);
    OK
}

/// (cb): closes a map bracket.
fn cb_apr_map_end(ctx: &mut CpuContext) -> HleResult {
    let cb = ctx.rdi();
    if cb == 0 {
        return ERR_PERMISSION_DENIED;
    }
    let Some(t) = ctx.read_u32(cb + cbuf::OFF_TYPE) else { return ERR_MEMORY_FAULT };
    if t & cbuf::TYPE_MAP_ACTIVE == 0 {
        return ERR_PERMISSION_DENIED;
    }
    if append_noop(ctx, cb, cbuf::APR_MAP_END_SIZE) != OK {
        return ERR_MEMORY_FAULT;
    }
    cbuf::update_type_flags(ctx, cb, 0, cbuf::TYPE_MAP_ACTIVE);
    OK
}

// ---------------------------------------------------------------------------
// Completion, wait and counter commands
// ---------------------------------------------------------------------------

/// (cb, equeue, ident, data, -, -): trigger the queue's AMPR registration
/// `ident` with `data` when the command executes.
fn cb_write_kernel_event_queue(ctx: &mut CpuContext) -> HleResult {
    let (cb, eq, ident, data) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    if cb == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    trace("write_equeue", cb, ident, data);
    ok_or(cbuf::append_kernel_event(ctx, cb, eq, ident as i32, data), ERR_MEMORY_FAULT)
}

fn append_write_address(ctx: &CpuContext, cb: u64, address: u64, value: u64) -> HleResult {
    if cb == 0 || address == 0 || address & 7 != 0 {
        return ERR_INVALID_ARGUMENT;
    }
    trace("write_address", cb, address, value);
    ok_or(cbuf::append(ctx, cb, cbuf::FIXED32_SIZE, Some(CommandData::WriteAddress { address, value })), ERR_BUSY)
}

/// (cb, address, value[, flags]): store a u64 (release) when the command executes.
fn cb_write_address(ctx: &mut CpuContext) -> HleResult {
    append_write_address(ctx, ctx.rdi(), ctx.rsi(), ctx.rdx())
}

/// (cb, address, ...): the counter/timer value is not modelled; 0 is stored.
fn cb_write_address_zero(ctx: &mut CpuContext) -> HleResult {
    append_write_address(ctx, ctx.rdi(), ctx.rsi(), 0)
}

/// (cb, address, value, uint8 compare, uint8 flush): block the queue until
/// the u64 at `address` compares true against `value`.
fn cb_wait_on_address(ctx: &mut CpuContext) -> HleResult {
    let (cb, address, value, compare) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx() as u8);
    if cb == 0 || address == 0 || address & 7 != 0 || compare > 6 {
        return ERR_INVALID_ARGUMENT;
    }
    trace("wait_address", cb, address, value);
    let data = CommandData::Wait { source: address, value, compare, is_counter: false };
    ok_or(cbuf::append(ctx, cb, cbuf::FIXED32_SIZE, Some(data)), ERR_BUSY)
}

/// (cb, uint8 index, uint8 access, value, uint8 compare, uint8 maskOp,
/// [stack] mask, [stack] uint8 flush): wait on a shared counter.
fn cb_wait_on_counter(ctx: &mut CpuContext) -> HleResult {
    let (cb, index, access, value) = (ctx.rdi(), ctx.rsi() as u8, ctx.rdx() as u8, ctx.rcx());
    let (compare, mask_op, flush) = (ctx.r8() as u8, ctx.r9() as u8, stack_arg(ctx, 1) as u8);
    if cb == 0 || access > 7 || compare > 6 || mask_op > 1 || flush > 1 {
        return ERR_INVALID_ARGUMENT;
    }
    // A cache flush has no host equivalent; other access modes and masked
    // compares are not modelled.
    if access != 1 || mask_op != 0 || value > u32::MAX as u64 {
        log::error!("[ampr] WaitOnCounter mode not implemented: access={access} maskOp={mask_op} value={value:#x}");
        return ERR_INVALID_ARGUMENT;
    }
    trace("wait_counter", cb, index as u64, value);
    let data = CommandData::Wait { source: index as u64, value, compare, is_counter: true };
    ok_or(cbuf::append(ctx, cb, cbuf::FIXED32_SIZE, Some(data)), ERR_BUSY)
}

/// (cb, uint8 index, uint32 value, uint8 compare, uint8 flush): wait until
/// the 32-bit shared counter compares true against `value`.
fn cb_wait_on_counter_32(ctx: &mut CpuContext) -> HleResult {
    let (cb, index, value, compare, flush) = (ctx.rdi(), ctx.rsi() as u8, ctx.rdx() as u32, ctx.rcx() as u8, ctx.r8() as u8);
    if index >= engine::APP_COUNTERS || compare > 3 || flush > 1 || cb == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if ctx.read_u64(cb + cbuf::OFF_DATA).unwrap_or(0) == 0 {
        return ERR_PERMISSION_DENIED;
    }
    trace("wait_counter", cb, index as u64, value as u64);
    let data = CommandData::Wait { source: index as u64, value: value as u64, compare, is_counter: true };
    ok_or(cbuf::append(ctx, cb, cbuf::FIXED32_SIZE, Some(data)), ERR_BUSY)
}

/// (cb, uint8 index, uint8 access, value, uint8 operation, uint32 mode):
/// set an application counter.
fn cb_write_counter(ctx: &mut CpuContext) -> HleResult {
    let (cb, index, access, value) = (ctx.rdi(), ctx.rsi() as u8, ctx.rdx() as u8, ctx.rcx());
    let (operation, mode) = (ctx.r8() as u8, ctx.r9() as u32);
    if cb == 0 || index >= engine::APP_COUNTERS || access > 7 || operation > 4 || mode > 1 {
        return ERR_INVALID_ARGUMENT;
    }
    if access != 1 || operation != 0 || mode != 0 {
        log::error!("[ampr] WriteCounter mode not implemented: access={access} operation={operation} mode={mode}");
        return ERR_INVALID_ARGUMENT;
    }
    trace("write_counter", cb, index as u64, value);
    let data = CommandData::WriteCounter { index, value: value as u32 };
    ok_or(cbuf::append(ctx, cb, cbuf::FIXED32_SIZE, Some(data)), ERR_BUSY)
}

/// (cb, ...): stream space only.
fn cb_write_counter_on_completion(ctx: &mut CpuContext) -> HleResult {
    append_noop(ctx, ctx.rdi(), cbuf::FIXED32_SIZE)
}

// ---------------------------------------------------------------------------
// Markers and nops
// ---------------------------------------------------------------------------

/// (cb, uint32 count): `count` dwords, 1..=16.
fn cb_nop(ctx: &mut CpuContext) -> HleResult {
    let n = ctx.rsi() as u32 as u64;
    if n == 0 || n > 16 {
        return ERR_INVALID_ARGUMENT;
    }
    append_noop(ctx, ctx.rdi(), n * 4)
}

/// (cb, uint32 count, data): a header dword plus `count` (<= 15) dwords.
fn cb_nop_with_data(ctx: &mut CpuContext) -> HleResult {
    let n = ctx.rsi() as u32 as u64;
    if n > 15 {
        return ERR_INVALID_ARGUMENT;
    }
    append_noop(ctx, ctx.rdi(), (n + 1) * 4)
}

/// (cb, uint32, data, uint32 bytes, ...): a header dword plus `bytes`.
fn cb_construct_nop(ctx: &mut CpuContext) -> HleResult {
    append_noop(ctx, ctx.rdi(), 4 + ctx.rcx() as u32 as u64)
}

/// (cb, uint32, label, color*)
fn cb_construct_marker(ctx: &mut CpuContext) -> HleResult {
    let size = cbuf::marker_size(label_len(ctx, ctx.rdx()), ctx.rcx() != 0);
    append_noop(ctx, ctx.rdi(), size)
}

/// (cb, label)
fn cb_marker(ctx: &mut CpuContext) -> HleResult {
    append_noop(ctx, ctx.rdi(), cbuf::marker_size(label_len(ctx, ctx.rsi()), false))
}

/// (cb, label, color)
fn cb_marker_with_color(ctx: &mut CpuContext) -> HleResult {
    append_noop(ctx, ctx.rdi(), cbuf::marker_size(label_len(ctx, ctx.rsi()), true))
}

/// (cb)
fn cb_pop_marker(ctx: &mut CpuContext) -> HleResult {
    append_noop(ctx, ctx.rdi(), 4)
}

// ---------------------------------------------------------------------------
// Measure functions
// ---------------------------------------------------------------------------

/// (-, dst, size, fileOffset)
fn cb_measure_read_file(ctx: &mut CpuContext) -> HleResult {
    let (dst, size, off) = (ctx.rsi(), ctx.rdx(), ctx.rcx());
    if cbuf::valid_read_range(dst, size) && cbuf::valid_file_offset(off) {
        cbuf::read_file_size(off) as HleResult
    } else {
        MEASURE_INVALID
    }
}

/// (size, fileOffset)
fn cb_measure_read_file_gather(ctx: &mut CpuContext) -> HleResult {
    let (size, off) = (ctx.rdi(), ctx.rsi());
    if cbuf::valid_read_size(size) && cbuf::valid_file_offset(off) {
        cbuf::read_gather_size(off) as HleResult
    } else {
        MEASURE_INVALID
    }
}

/// (dst, size)
fn cb_measure_read_file_scatter(ctx: &mut CpuContext) -> HleResult {
    if cbuf::valid_read_range(ctx.rdi(), ctx.rsi()) {
        cbuf::READ_SCATTER_SIZE as HleResult
    } else {
        MEASURE_INVALID
    }
}

/// (dst, size, fileOffset)
fn cb_measure_read_file_gather_scatter(ctx: &mut CpuContext) -> HleResult {
    let (dst, size, off) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if cbuf::valid_read_range(dst, size) && cbuf::valid_file_offset(off) {
        cbuf::read_gather_scatter_size(off) as HleResult
    } else {
        MEASURE_INVALID
    }
}

fn cb_measure_kernel_event_queue(_ctx: &mut CpuContext) -> HleResult {
    cbuf::KERNEL_EVENT_SIZE as HleResult
}

/// Every wait, write-address, counter and timer command takes 32 bytes.
fn cb_measure_fixed32(_ctx: &mut CpuContext) -> HleResult {
    cbuf::FIXED32_SIZE as HleResult
}

/// (uint32 count)
fn cb_measure_nop(ctx: &mut CpuContext) -> HleResult {
    let n = ctx.rdi() as u32 as u64;
    (if n == 0 { 4 } else { cbuf::align4(n * 4) }) as HleResult
}

/// (uint32 count, data)
fn cb_measure_nop_with_data(ctx: &mut CpuContext) -> HleResult {
    cbuf::align4((ctx.rdi() as u32 as u64 + 1) * 4) as HleResult
}

/// (label)
fn cb_measure_marker(ctx: &mut CpuContext) -> HleResult {
    cbuf::marker_size(label_len(ctx, ctx.rdi()), false) as HleResult
}

/// (label, color)
fn cb_measure_marker_with_color(ctx: &mut CpuContext) -> HleResult {
    cbuf::marker_size(label_len(ctx, ctx.rdi()), true) as HleResult
}

/// Pop marker, reset gather/scatter state and map end: one dword.
fn cb_measure_dword(_ctx: &mut CpuContext) -> HleResult {
    4
}

/// (va, size, type, prot)
fn cb_measure_apr_map_begin(ctx: &mut CpuContext) -> HleResult {
    if cbuf::valid_map_args(ctx.rdi(), ctx.rsi()) {
        cbuf::APR_MAP_BEGIN_SIZE as HleResult
    } else {
        ERR_INVALID_ARGUMENT
    }
}

/// (va, dmemOffset, size, type, prot)
fn cb_measure_apr_map_direct_begin(ctx: &mut CpuContext) -> HleResult {
    if cbuf::valid_map_args(ctx.rdi(), ctx.rdx()) && ctx.rsi() % cbuf::AMM_PAGE_SIZE == 0 {
        cbuf::APR_MAP_DIRECT_BEGIN_SIZE as HleResult
    } else {
        ERR_INVALID_ARGUMENT
    }
}

fn cb_measure_amm_map(_ctx: &mut CpuContext) -> HleResult {
    cbuf::AMM_MAP_SIZE as HleResult
}

fn cb_measure_amm_map_direct(_ctx: &mut CpuContext) -> HleResult {
    cbuf::AMM_MAP_DIRECT_SIZE as HleResult
}

fn cb_measure_amm_unmap(_ctx: &mut CpuContext) -> HleResult {
    cbuf::AMM_UNMAP_SIZE as HleResult
}

// ---------------------------------------------------------------------------
// AMM
// ---------------------------------------------------------------------------

fn append_amm(ctx: &CpuContext, cb: u64, m: AmmMap) -> HleResult {
    if cb == 0 || !cbuf::valid_map_args(m.va, m.size) || (m.kind == AmmKind::MapDirect && m.dmem_offset % cbuf::AMM_PAGE_SIZE != 0)
    {
        return ERR_INVALID_ARGUMENT;
    }
    let size = match m.kind {
        AmmKind::MapAuto => cbuf::AMM_MAP_SIZE,
        AmmKind::MapDirect => cbuf::AMM_MAP_DIRECT_SIZE,
        AmmKind::Unmap | AmmKind::MapAsPrt => cbuf::AMM_UNMAP_SIZE,
        AmmKind::Remap | AmmKind::MultiMap => cbuf::AMM_REMAP_SIZE,
        AmmKind::ModifyProtect | AmmKind::ModifyMtypeProtect | AmmKind::AllocatePaForPrt => cbuf::AMM_PROTECT_SIZE,
    };
    trace("amm", cb, m.va, m.size);
    ok_or(cbuf::append(ctx, cb, size, Some(CommandData::AmmMap(m))), ERR_BUSY)
}

fn amm_map(kind: AmmKind, va: u64, dmem_offset: u64, size: u64, mtype: u64, prot: u64, gpu_mask_id: u64) -> AmmMap {
    AmmMap { kind, va, dmem_offset, size, mtype: mtype as i32, prot: prot as i32, gpu_mask_id: gpu_mask_id as u8, ..Default::default() }
}

/// AMM protection: CPU r/w/x... bits only (no bits in 0xFFFFFC0C).
fn valid_amm_protection(prot: i32) -> bool {
    prot as u32 & 0xFFFF_FC0C == 0
}

/// Remap/MultiMap arguments: both ranges valid, a valid protection.
fn valid_remap(va: u64, source: u64, size: u64, prot: i32) -> bool {
    cbuf::valid_map_args(va, size) && cbuf::valid_map_args(source, size) && valid_amm_protection(prot)
}

/// ModifyProtect arguments: a valid range, protection and mask.
fn valid_protect(va: u64, size: u64, prot: i32, mask: i32) -> bool {
    cbuf::valid_map_args(va, size) && valid_amm_protection(prot) && valid_amm_protection(mask)
}

/// The object check of the AMM records: a NULL object is invalid, one
/// without command memory EPERM.
fn amm_object_error(ctx: &CpuContext, cb: u64) -> Option<HleResult> {
    if cb == 0 {
        return Some(ERR_INVALID_ARGUMENT);
    }
    (ctx.read_u64(cb + cbuf::OFF_DATA).unwrap_or(0) == 0).then_some(ERR_PERMISSION_DENIED)
}

/// Record an AMM command of `record_size` bytes; a full buffer is EBUSY.
fn push_amm(ctx: &CpuContext, cb: u64, record_size: u64, m: AmmMap) -> HleResult {
    trace("amm", cb, m.va, m.size);
    ok_or(cbuf::append(ctx, cb, record_size, Some(CommandData::AmmMap(m))), ERR_BUSY)
}

/// (cb, va, source, size, prot[, uint8 gpuMaskId]): Remap or MultiMap.
fn amm_remap(ctx: &CpuContext, kind: AmmKind, gpu_mask_id: u64) -> HleResult {
    let (cb, va, source, size, prot) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8() as i32);
    if let Some(e) = amm_object_error(ctx, cb) {
        return e;
    }
    if !valid_remap(va, source, size, prot) {
        return ERR_INVALID_ARGUMENT;
    }
    let m = AmmMap { kind, va, source, size, prot, gpu_mask_id: gpu_mask_id as u8, ..Default::default() };
    push_amm(ctx, cb, cbuf::AMM_REMAP_SIZE, m)
}

fn cb_amm_remap(ctx: &mut CpuContext) -> HleResult {
    amm_remap(ctx, AmmKind::Remap, 0)
}
fn cb_amm_remap_with_gpu_mask_id(ctx: &mut CpuContext) -> HleResult {
    amm_remap(ctx, AmmKind::Remap, ctx.r9())
}
fn cb_amm_multi_map(ctx: &mut CpuContext) -> HleResult {
    amm_remap(ctx, AmmKind::MultiMap, 0)
}
fn cb_amm_multi_map_with_gpu_mask_id(ctx: &mut CpuContext) -> HleResult {
    amm_remap(ctx, AmmKind::MultiMap, ctx.r9())
}

/// Record ModifyProtect / ModifyMtypeProtect.
#[allow(clippy::too_many_arguments)]
fn amm_protect(ctx: &CpuContext, kind: AmmKind, va: u64, size: u64, mtype: i32, prot: i32, mask: i32, gpu_mask_id: u64) -> HleResult {
    let cb = ctx.rdi();
    if let Some(e) = amm_object_error(ctx, cb) {
        return e;
    }
    if !valid_protect(va, size, prot, mask) {
        return ERR_INVALID_ARGUMENT;
    }
    let m = AmmMap { kind, va, size, mtype, prot, mask, gpu_mask_id: gpu_mask_id as u8, ..Default::default() };
    push_amm(ctx, cb, cbuf::AMM_PROTECT_SIZE, m)
}

/// (cb, va, size, prot, mask)
fn cb_amm_modify_protect(ctx: &mut CpuContext) -> HleResult {
    amm_protect(ctx, AmmKind::ModifyProtect, ctx.rsi(), ctx.rdx(), 0, ctx.rcx() as i32, ctx.r8() as i32, 0)
}
/// (cb, va, size, prot, mask, uint8 gpuMaskId)
fn cb_amm_modify_protect_with_gpu_mask_id(ctx: &mut CpuContext) -> HleResult {
    amm_protect(ctx, AmmKind::ModifyProtect, ctx.rsi(), ctx.rdx(), 0, ctx.rcx() as i32, ctx.r8() as i32, ctx.r9())
}
/// (cb, va, size, type, prot, mask)
fn cb_amm_modify_mtype_protect(ctx: &mut CpuContext) -> HleResult {
    amm_protect(ctx, AmmKind::ModifyMtypeProtect, ctx.rsi(), ctx.rdx(), ctx.rcx() as i32, ctx.r8() as i32, ctx.r9() as i32, 0)
}
/// (cb, va, size, type, prot, mask, [stack] uint8 gpuMaskId)
fn cb_amm_modify_mtype_protect_with_gpu_mask_id(ctx: &mut CpuContext) -> HleResult {
    let gpu = stack_arg(ctx, 0);
    amm_protect(ctx, AmmKind::ModifyMtypeProtect, ctx.rsi(), ctx.rdx(), ctx.rcx() as i32, ctx.r8() as i32, ctx.r9() as i32, gpu)
}

/// Record a PRT command: a full buffer is EBUSY before a missing command
/// memory is EPERM.
fn push_prt(ctx: &CpuContext, cb: u64, record_size: u64, m: AmmMap) -> HleResult {
    let st = cbuf::try_get_state(ctx, cb);
    if st.as_ref().is_some_and(|s| s.write_offset > s.size || s.size - s.write_offset < record_size) {
        return ERR_BUSY;
    }
    if st.is_none_or(|s| s.buffer == 0) {
        return ERR_PERMISSION_DENIED;
    }
    trace("amm_prt", cb, m.va, m.size);
    ok_or(cbuf::append(ctx, cb, record_size, Some(CommandData::AmmMap(m))), ERR_BUSY)
}

/// (cb, va, size)
fn cb_amm_map_as_prt(ctx: &mut CpuContext) -> HleResult {
    let (cb, va, size) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if cb == 0 || !cbuf::valid_map_args(va, size) {
        return ERR_INVALID_ARGUMENT;
    }
    push_prt(ctx, cb, cbuf::AMM_MAP_AS_PRT_SIZE, AmmMap { kind: AmmKind::MapAsPrt, va, size, ..Default::default() })
}

/// Protection bits AllocatePaForPrt sets on pages that are already backed.
const PRT_ALLOCATION_MASK: i32 = 1019;

/// (cb, va, size, type, prot)
fn cb_amm_allocate_pa_for_prt(ctx: &mut CpuContext) -> HleResult {
    let (cb, va, size, mtype, prot) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx() as i32, ctx.r8() as i32);
    if !valid_amm_protection(prot) || cb == 0 || !cbuf::valid_map_args(va, size) {
        return ERR_INVALID_ARGUMENT;
    }
    let m = AmmMap { kind: AmmKind::AllocatePaForPrt, va, size, mtype, prot, mask: PRT_ALLOCATION_MASK, ..Default::default() };
    push_prt(ctx, cb, cbuf::AMM_PROTECT_SIZE, m)
}

/// Error of the AMM measure functions: the code sign-extended.
const AMM_MEASURE_INVALID: HleResult = ERR_INVALID_ARGUMENT;

fn measure(valid: bool, size: u64) -> HleResult {
    if valid {
        size as HleResult
    } else {
        AMM_MEASURE_INVALID
    }
}

/// (va, source, size, prot[, gpuMaskId])
fn cb_measure_amm_remap(ctx: &mut CpuContext) -> HleResult {
    measure(valid_remap(ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx() as i32), cbuf::AMM_REMAP_SIZE)
}
/// (va, size, prot, mask[, gpuMaskId])
fn cb_measure_amm_modify_protect(ctx: &mut CpuContext) -> HleResult {
    measure(valid_protect(ctx.rdi(), ctx.rsi(), ctx.rdx() as i32, ctx.rcx() as i32), cbuf::AMM_PROTECT_SIZE)
}
/// (va, size, type, prot, mask[, gpuMaskId])
fn cb_measure_amm_modify_mtype_protect(ctx: &mut CpuContext) -> HleResult {
    measure(valid_protect(ctx.rdi(), ctx.rsi(), ctx.rcx() as i32, ctx.r8() as i32), cbuf::AMM_PROTECT_SIZE)
}
/// (va, size)
fn cb_measure_amm_map_as_prt(ctx: &mut CpuContext) -> HleResult {
    measure(cbuf::valid_map_args(ctx.rdi(), ctx.rsi()), cbuf::AMM_MAP_AS_PRT_SIZE)
}
/// (va, size, type, prot)
fn cb_measure_amm_allocate_pa_for_prt(ctx: &mut CpuContext) -> HleResult {
    measure(cbuf::valid_map_args(ctx.rdi(), ctx.rsi()) && valid_amm_protection(ctx.rcx() as i32), cbuf::AMM_PROTECT_SIZE)
}

/// (cb, va, size, type, prot)
fn cb_amm_map(ctx: &mut CpuContext) -> HleResult {
    let m = amm_map(AmmKind::MapAuto, ctx.rsi(), 0, ctx.rdx(), ctx.rcx(), ctx.r8(), 0);
    append_amm(ctx, ctx.rdi(), m)
}

/// (cb, va, size, type, prot, uint8 gpuMaskId)
fn cb_amm_map_with_gpu_mask_id(ctx: &mut CpuContext) -> HleResult {
    let m = amm_map(AmmKind::MapAuto, ctx.rsi(), 0, ctx.rdx(), ctx.rcx(), ctx.r8(), ctx.r9());
    append_amm(ctx, ctx.rdi(), m)
}

/// (cb, va, dmemOffset, size, type, prot)
fn cb_amm_map_direct(ctx: &mut CpuContext) -> HleResult {
    let m = amm_map(AmmKind::MapDirect, ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8(), ctx.r9(), 0);
    append_amm(ctx, ctx.rdi(), m)
}

/// (cb, va, dmemOffset, size, type, prot, [stack] uint8 gpuMaskId)
fn cb_amm_map_direct_with_gpu_mask_id(ctx: &mut CpuContext) -> HleResult {
    let m = amm_map(AmmKind::MapDirect, ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8(), ctx.r9(), stack_arg(ctx, 0));
    append_amm(ctx, ctx.rdi(), m)
}

/// (cb, va, size)
fn cb_amm_unmap(ctx: &mut CpuContext) -> HleResult {
    let m = amm_map(AmmKind::Unmap, ctx.rsi(), 0, ctx.rdx(), 0, 0, 0);
    append_amm(ctx, ctx.rdi(), m)
}

const AMM_USAGE_DIRECT: i32 = 0;
const AMM_USAGE_AUTO: i32 = 1;
const AMM_BLOCK_SIZE: u64 = 0x20_0000;

/// (searchStart, searchEnd, size, align, int usage, int64* dmemOffset):
/// allocate direct memory for AMM; automatic usage feeds automatic maps.
fn cb_amm_give_direct_memory(ctx: &mut CpuContext) -> HleResult {
    let (start, end, size, align, usage, out) =
        (ctx.rdi() as i64, ctx.rsi() as i64, ctx.rdx(), ctx.rcx(), ctx.r8() as i32, ctx.r9());
    if out == 0
        || size == 0
        || size % AMM_BLOCK_SIZE != 0
        || align % AMM_BLOCK_SIZE != 0
        || (usage != AMM_USAGE_DIRECT && usage != AMM_USAGE_AUTO)
        || start < 0
        || end <= start
    {
        return ERR_INVALID_ARGUMENT;
    }
    let align = if align != 0 { align } else { AMM_BLOCK_SIZE };
    let end = (end as u64).min(crate::memory::direct_size());
    let Some(phys) = crate::memory::allocate_direct(start as u64, end, size, align, 0) else {
        return ERR_TRY_AGAIN;
    };
    if usage == AMM_USAGE_AUTO {
        amm::give_automatic(phys, size);
    }
    log::debug!("[ampr] AmmGiveDirectMemory size={size:#x} usage={usage} -> {phys:#x}");
    ok_or(ctx.write_u64(out, phys), ERR_MEMORY_FAULT)
}

/// (vaStart*, vaEnd*, multimapVaStart*, multimapVaEnd*)
fn cb_amm_get_virtual_address_ranges(ctx: &mut CpuContext) -> HleResult {
    let end = amm::VA_BASE + amm::VA_SIZE;
    for (p, v) in [(ctx.rdi(), amm::VA_BASE), (ctx.rsi(), end), (ctx.rdx(), end), (ctx.rcx(), end)] {
        if p != 0 {
            ctx.write_u64(p, v);
        }
    }
    0
}

const AMM_USAGE_STATS_SIZE: u64 = 0x18;

/// (stats*): `{u64 sizeInBytes, u16 pool entries, u16 snapshot, u16 high
/// watermark, u16 reserved, u32 ring idle flags}`, written up to the
/// caller's `sizeInBytes`.
fn cb_amm_get_usage_stats_data(ctx: &mut CpuContext) -> HleResult {
    let stats = ctx.rdi();
    let Some(size) = (stats != 0).then(|| ctx.read_u64(stats)).flatten() else { return ERR_INVALID_ARGUMENT };
    if size > AMM_USAGE_STATS_SIZE {
        return ERR_INVALID_ARGUMENT;
    }
    let mut out = [0u8; AMM_USAGE_STATS_SIZE as usize];
    out[0..8].copy_from_slice(&size.to_le_bytes());
    out[8..10].copy_from_slice(&512u16.to_le_bytes());
    out[16..20].copy_from_slice(&7u32.to_le_bytes());
    ok_or(ctx.write_bytes(stats, &out[..size as usize]), ERR_MEMORY_FAULT)
}

fn cb_amm_set_page_table_pool_threshold(_ctx: &mut CpuContext) -> HleResult {
    OK
}

/// (base, uint32 bytes, uint32 priority)
fn cb_amm_submit(ctx: &mut CpuContext) -> HleResult {
    engine::queue_command_buffer(ctx, Engine::Amm, ctx.rdi(), ctx.rsi() as u32, ctx.rdx() as u32, 0, 0)
}

/// (base, uint32 bytes, uint32 priority, uint32* id)
fn cb_amm_submit_and_get_id(ctx: &mut CpuContext) -> HleResult {
    if ctx.rcx() == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    engine::queue_command_buffer(ctx, Engine::Amm, ctx.rdi(), ctx.rsi() as u32, ctx.rdx() as u32, 0, ctx.rcx())
}

/// (base, uint32 bytes, uint32 priority, result*, uint32* id)
fn cb_amm_submit_and_get_result(ctx: &mut CpuContext) -> HleResult {
    engine::queue_command_buffer(ctx, Engine::Amm, ctx.rdi(), ctx.rsi() as u32, ctx.rdx() as u32, ctx.rcx(), ctx.r8())
}

/// (uint32 id)
fn cb_amm_wait(ctx: &mut CpuContext) -> HleResult {
    ok_or(engine::complete_submission(ctx.rdi() as u32), engine::ERR_NO_SUCH_SUBMISSION)
}

// ---------------------------------------------------------------------------
// libkernel APR submission (errors: -1 and errno)
// ---------------------------------------------------------------------------

fn syscall_result(ctx: &CpuContext, r: HleResult) -> HleResult {
    if r >= 0 {
        return r;
    }
    crate::kernel::set_errno(ctx, (r as u32 & 0xFFFF) as i32);
    -1
}

/// (cb, uint32 priority)
fn cb_apr_submit(ctx: &mut CpuContext) -> HleResult {
    let r = engine::queue_command_buffer(ctx, Engine::Apr, ctx.rdi(), 0, ctx.rsi() as u32, 0, 0);
    syscall_result(ctx, r)
}

/// (cb, uint32 priority, uint32* id)
fn cb_apr_submit_and_get_id(ctx: &mut CpuContext) -> HleResult {
    let r = if ctx.rdx() == 0 {
        ERR_INVALID_ARGUMENT
    } else {
        engine::queue_command_buffer(ctx, Engine::Apr, ctx.rdi(), 0, ctx.rsi() as u32, 0, ctx.rdx())
    };
    syscall_result(ctx, r)
}

/// (cb, uint32 priority, result*, uint32* id)
fn cb_apr_submit_and_get_result(ctx: &mut CpuContext) -> HleResult {
    let r = engine::queue_command_buffer(ctx, Engine::Apr, ctx.rdi(), 0, ctx.rsi() as u32, ctx.rdx(), ctx.rcx());
    syscall_result(ctx, r)
}

/// (uint32 id)
fn cb_apr_wait(ctx: &mut CpuContext) -> HleResult {
    let r = ok_or(engine::complete_submission(ctx.rdi() as u32), engine::ERR_NO_SUCH_SUBMISSION);
    syscall_result(ctx, r)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

type Export = (&'static str, &'static str, crate::registry::HleCallback);

pub(crate) const EXPORTS: &[Export] = &[
    // construction / buffer management
    ("8aI7R7WaOlc", "sceAmprCommandBufferConstructor", cb_constructor),
    ("GuchCTefuZw", "sceAmprCommandBufferDestructor", cb_destructor),
    ("a8uLzYY--tM", "sceAmprAprCommandBufferConstructor", cb_apr_constructor),
    ("Qs1xtplKo0U", "sceAmprAprCommandBufferDestructor", cb_noop_destructor),
    ("EDq5bqCqYpA", "sceAmprAmmCommandBufferConstructor", cb_noop_destructor),
    ("pvUFDOHilnE", "sceAmprAmmCommandBufferDestructor", cb_noop_destructor),
    ("N-FSPA4S3nI", "sceAmprCommandBufferSetBuffer", cb_set_buffer),
    ("baQO9ez2gL4", "sceAmprCommandBufferReset", cb_reset),
    ("ULvXMDz56po", "sceAmprCommandBufferClearBuffer", cb_clear_buffer),
    ("tZDDEo2tE5k", "sceAmprCommandBufferGetSize", cb_get_size),
    ("GnxKOHEawhk", "sceAmprCommandBufferGetCurrentOffset", cb_get_current_offset),
    ("gzndltBEzWc", "sceAmprCommandBufferGetNumCommands", cb_get_num_commands),
    ("RPCAhx-aabE", "sceAmprCommandBufferGetBufferBaseAddress", cb_get_buffer_base_address),
    ("VEDMaQmJZng", "sceAmprCommandBufferGetType", cb_get_type),
    // APR reads and map brackets
    ("mQ16-QdKv7k", "sceAmprAprCommandBufferReadFile", cb_apr_read_file),
    ("mZSbNJVJpV8", "sceAmprAprCommandBufferReadFileGather", cb_apr_read_file_gather),
    ("Jg-AgkdJHkk", "sceAmprAprCommandBufferReadFileScatter", cb_apr_read_file_scatter),
    ("BVmR1H8l+XI", "sceAmprAprCommandBufferReadFileGatherScatter", cb_apr_read_file_gather_scatter),
    ("YPxkUDhgoNI", "sceAmprAprCommandBufferResetGatherScatterState", cb_apr_reset_gather_scatter_state),
    ("Eul7AGEpjLo", "sceAmprAprCommandBufferMapBegin", cb_apr_map_begin),
    ("bFEs0Gs6D2A", "sceAmprAprCommandBufferMapDirectBegin", cb_apr_map_direct_begin),
    ("X169CE6G3Y4", "sceAmprAprCommandBufferMapEnd", cb_apr_map_end),
    // completion, wait and counter commands
    ("H896Pt-yB4I", "sceAmprCommandBufferWriteKernelEventQueue_04_00", cb_write_kernel_event_queue),
    ("o67gODLFpls", "sceAmprCommandBufferWriteKernelEventQueueOnCompletion", cb_write_kernel_event_queue),
    ("j0+3uJMxYJY", "sceAmprCommandBufferWriteAddress_04_00", cb_write_address),
    ("sJXyWHjP-F8", "sceAmprCommandBufferWriteAddressOnCompletion", cb_write_address),
    ("V7GQTEeUfhw", "sceAmprCommandBufferWaitOnAddress", cb_wait_on_address),
    ("DLfoNxTFNVk", "sceAmprCommandBufferWaitOnAddress_04_00", cb_wait_on_address),
    ("cQb8Zr8Q0Y0", "sceAmprCommandBufferWaitOnCounter_04_00", cb_wait_on_counter),
    ("jK+yuYCI7MA", "sceAmprCommandBufferWriteCounter_04_00", cb_write_counter),
    ("3wn42MWTzTs", "sceAmprCommandBufferWriteCounterOnCompletion", cb_write_counter_on_completion),
    ("bt3LHR9xjK4", "sceAmprCommandBufferWriteAddressFromTimeCounter_04_00", cb_write_address_zero),
    ("FI2JD0jAHCs", "sceAmprCommandBufferWriteAddressFromTimeCounterOnCompletion", cb_write_address_zero),
    ("t4ExS+SwLjs", "sceAmprCommandBufferWriteAddressFromCounter_04_00", cb_write_address_zero),
    ("gSF5OsXdfIg", "sceAmprCommandBufferWriteAddressFromCounterOnCompletion", cb_write_address_zero),
    ("enZm-6GjWqw", "sceAmprCommandBufferWriteAddressFromCounterPair_04_00", cb_write_address_zero),
    ("ZLWtNUP6R5E", "sceAmprCommandBufferWriteAddressFromCounterPairOnCompletion", cb_write_address_zero),
    // markers / nops
    ("tNn5WBkta60", "sceAmprCommandBufferNop", cb_nop),
    ("pFQ9UHpO52s", "sceAmprCommandBufferNopWithData", cb_nop_with_data),
    ("GmOguNIsuKk", "sceAmprCommandBufferConstructNop", cb_construct_nop),
    ("4UkZbYKVF7c", "sceAmprCommandBufferConstructMarker", cb_construct_marker),
    ("dXPaz65HNmk", "sceAmprCommandBufferPushMarker", cb_marker),
    ("f12ObAMEi9A", "sceAmprCommandBufferPushMarkerWithColor", cb_marker_with_color),
    ("4quckD2y7Pg", "sceAmprCommandBufferSetMarker", cb_marker),
    ("sWbST0oQKsc", "sceAmprCommandBufferSetMarkerWithColor", cb_marker_with_color),
    ("mv0O8Zg0woU", "sceAmprCommandBufferPopMarker", cb_pop_marker),
    // measure functions
    ("vWU-odnS+fU", "sceAmprMeasureCommandSizeReadFile", cb_measure_read_file),
    ("qesF88X4DRg", "sceAmprMeasureCommandSizeReadFileGather", cb_measure_read_file_gather),
    ("7nXGDGMXSqo", "sceAmprMeasureCommandSizeReadFileScatter", cb_measure_read_file_scatter),
    ("DXmgc5op8Yw", "sceAmprMeasureCommandSizeReadFileGatherScatter", cb_measure_read_file_gather_scatter),
    ("rddQYXM0CjM", "sceAmprMeasureCommandSizeResetGatherScatterState", cb_measure_dword),
    ("sSAUCCU1dv4", "sceAmprMeasureCommandSizeWriteKernelEventQueue_04_00", cb_measure_kernel_event_queue),
    ("Zi3dBUjgyXI", "sceAmprMeasureCommandSizeWriteKernelEventQueueOnCompletion", cb_measure_kernel_event_queue),
    ("C+IEj+BsAFM", "sceAmprMeasureCommandSizeWriteAddressOnCompletion", cb_measure_fixed32),
    ("x7SQEXfeovg", "sceAmprMeasureCommandSizeWriteAddressFromTimeCounterOnCompletion", cb_measure_fixed32),
    ("32AcaTaBPSY", "sceAmprMeasureCommandSizeWriteAddressFromCounterOnCompletion", cb_measure_fixed32),
    ("vxC58+DRk-U", "sceAmprMeasureCommandSizeWriteAddressFromCounterPairOnCompletion", cb_measure_fixed32),
    ("4muPEJ-x5N8", "sceAmprMeasureCommandSizeWriteCounterOnCompletion", cb_measure_fixed32),
    ("jIlc4p5dSD0", "sceAmprMeasureCommandSizeWaitOnAddress", cb_measure_fixed32),
    ("0BMj1hgG+kE", "sceAmprMeasureCommandSizeWaitOnAddress_04_00", cb_measure_fixed32),
    ("ClnsFLLLcss", "sceAmprMeasureCommandSizeWaitOnCounter_04_00", cb_measure_fixed32),
    ("4fgtGfXDrFc", "sceAmprMeasureCommandSizeWriteAddress_04_00", cb_measure_fixed32),
    ("gAtc79UTt5E", "sceAmprMeasureCommandSizeWriteAddressFromTimeCounter_04_00", cb_measure_fixed32),
    ("JYd9g9L+TmE", "sceAmprMeasureCommandSizeWriteAddressFromCounter_04_00", cb_measure_fixed32),
    ("2Hw8gjMdwSY", "sceAmprMeasureCommandSizeWriteAddressFromCounterPair_04_00", cb_measure_fixed32),
    ("I-Qm+MEso5c", "sceAmprMeasureCommandSizeWriteCounter_04_00", cb_measure_fixed32),
    ("NNIZ-FMyz3M", "sceAmprMeasureCommandSizeNop", cb_measure_nop),
    ("Xp85BP3+BBI", "sceAmprMeasureCommandSizeNopWithData", cb_measure_nop_with_data),
    ("0RdLmAh7WVo", "sceAmprMeasureCommandSizePushMarker", cb_measure_marker),
    ("3OfeY4pzDV0", "sceAmprMeasureCommandSizePushMarkerWithColor", cb_measure_marker_with_color),
    ("VGkEj4d6-Kg", "sceAmprMeasureCommandSizeSetMarker", cb_measure_marker),
    ("tmfr97+ED5I", "sceAmprMeasureCommandSizeSetMarkerWithColor", cb_measure_marker_with_color),
    ("pbnNnahE8vk", "sceAmprMeasureCommandSizePopMarker", cb_measure_dword),
    ("kdFImtTD0hc", "sceAmprMeasureCommandSizeMapBegin", cb_measure_apr_map_begin),
    ("qvbdJc7bG+s", "sceAmprMeasureCommandSizeMapDirectBegin", cb_measure_apr_map_direct_begin),
    ("iwTNhyaemnw", "sceAmprMeasureCommandSizeMapEnd", cb_measure_dword),
    ("6hbai6KIXkk", "sceAmprAmmMeasureAmmCommandSizeMap", cb_measure_amm_map),
    ("m+fYyX8oFqw", "sceAmprAmmMeasureAmmCommandSizeMapWithGpuMaskId", cb_measure_amm_map),
    ("ZFDZoN9IbVU", "sceAmprAmmMeasureAmmCommandSizeMapDirect", cb_measure_amm_map_direct),
    ("KUjtdPZJo5I", "sceAmprAmmMeasureAmmCommandSizeMapDirectWithGpuMaskId", cb_measure_amm_map_direct),
    ("Ayg6PIon2wA", "sceAmprAmmMeasureAmmCommandSizeUnmap", cb_measure_amm_unmap),
    // AMM
    ("JEVYGhDc97M", "sceAmprAmmCommandBufferMap", cb_amm_map),
    ("ojBkmG7+CgE", "sceAmprAmmCommandBufferMapWithGpuMaskId", cb_amm_map_with_gpu_mask_id),
    ("8TBE+9XCZbI", "sceAmprAmmCommandBufferMapDirect", cb_amm_map_direct),
    ("kOfZlhbVAkc", "sceAmprAmmCommandBufferMapDirectWithGpuMaskId", cb_amm_map_direct_with_gpu_mask_id),
    ("M-VFI2DJWQA", "sceAmprAmmCommandBufferUnmap", cb_amm_unmap),
    ("Q07J7XpvhrU", "sceAmprAmmGiveDirectMemory", cb_amm_give_direct_memory),
    ("wkQR9+xTFKY", "sceAmprAmmGetVirtualAddressRanges", cb_amm_get_virtual_address_ranges),
    ("KqiWXLgCVe0", "sceAmprAmmGetUsageStatsData", cb_amm_get_usage_stats_data),
    ("touqMEt6qXQ", "sceAmprAmmSetPageTablePoolOccupancyNotificationThreshold", cb_amm_set_page_table_pool_threshold),
    ("lwS-7y3jcBI", "sceAmprAmmSubmitCommandBuffer", cb_amm_submit),
    ("NnKhlMJtIsI", "sceAmprAmmSubmitCommandBuffer3", cb_amm_submit_and_get_id),
    ("OJf3vCckPAM", "sceAmprAmmSubmitCommandBuffer2", cb_amm_submit_and_get_result),
    ("HXymib4T8gc", "sceAmprAmmWaitCommandBufferCompletion", cb_amm_wait),
    ("ijvRMHfwwjc", "sceAmprAmmCommandBufferRemap", cb_amm_remap),
    ("Fo0Oaoqukg4", "sceAmprAmmCommandBufferRemapWithGpuMaskId", cb_amm_remap_with_gpu_mask_id),
    ("j2C9no3P4Nw", "sceAmprAmmCommandBufferMultiMap", cb_amm_multi_map),
    ("Qyz8VbxnaII", "sceAmprAmmCommandBufferMultiMapWithGpuMaskId", cb_amm_multi_map_with_gpu_mask_id),
    ("joNORJjCFhg", "sceAmprAmmCommandBufferModifyProtect", cb_amm_modify_protect),
    ("spjcxyNX-KE", "sceAmprAmmCommandBufferModifyProtectWithGpuMaskId", cb_amm_modify_protect_with_gpu_mask_id),
    ("rKq9nG+L9Xo", "sceAmprAmmCommandBufferModifyMtypeProtect", cb_amm_modify_mtype_protect),
    ("ie-UuLGbYXo", "sceAmprAmmCommandBufferModifyMtypeProtectWithGpuMaskId", cb_amm_modify_mtype_protect_with_gpu_mask_id),
    ("ZJCgt+aPHAU", "sceAmprAmmCommandBufferMapAsPrt", cb_amm_map_as_prt),
    ("d3lKWPM6odc", "sceAmprAmmCommandBufferAllocatePaForPrt", cb_amm_allocate_pa_for_prt),
    ("8sFq5s8eHnQ", "sceAmprAmmMeasureAmmCommandSizeRemap", cb_measure_amm_remap),
    ("b50TWwRmzu8", "sceAmprAmmMeasureAmmCommandSizeRemapWithGpuMaskId", cb_measure_amm_remap),
    ("U6PxwrRchl4", "sceAmprAmmMeasureAmmCommandSizeMultiMap", cb_measure_amm_remap),
    ("uGs4yyYTASM", "sceAmprAmmMeasureAmmCommandSizeMultiMapWithGpuMaskId", cb_measure_amm_remap),
    ("JPQagYw-y+4", "sceAmprAmmMeasureAmmCommandSizeModifyProtect", cb_measure_amm_modify_protect),
    ("gdSoJ6dvztU", "sceAmprAmmMeasureAmmCommandSizeModifyProtectWithGpuMaskId", cb_measure_amm_modify_protect),
    ("Ok2Nxka7qI4", "sceAmprAmmMeasureAmmCommandSizeModifyMtypeProtect", cb_measure_amm_modify_mtype_protect),
    ("c5D63Y35GjI", "sceAmprAmmMeasureAmmCommandSizeModifyMtypeProtectWithGpuMaskId", cb_measure_amm_modify_mtype_protect),
    ("4SSJ+cn+Pvg", "sceAmprAmmMeasureAmmCommandSizeMapAsPrt", cb_measure_amm_map_as_prt),
    ("2yYv9QWB3hU", "sceAmprAmmMeasureAmmCommandSizeAllocatePaForPrt", cb_measure_amm_allocate_pa_for_prt),
    ("FrCNL9TQ8ms", "sceAmprCommandBufferWaitOnCounter", cb_wait_on_counter_32),
    ("6jLL5BIZ88U", "sceAmprMeasureCommandSizeWaitOnCounter", cb_measure_fixed32),
];

/// libkernel APR submission exports.
pub(crate) const KERNEL_EXPORTS: &[Export] = &[
    ("eE4Szl8sil8", "sceKernelAprSubmitCommandBuffer", cb_apr_submit),
    ("qvMUCyyaCSI", "sceKernelAprSubmitCommandBufferAndGetId", cb_apr_submit_and_get_id),
    ("ASoW5WE-UPo", "sceKernelAprSubmitCommandBufferAndGetResult", cb_apr_submit_and_get_result),
    ("rqwFKI4PAiM", "sceKernelAprWaitCommandBuffer", cb_apr_wait),
];

/// Register the libSceAmpr and libkernel APR submission exports.
pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
    for &(nid, name, cb) in KERNEL_EXPORTS {
        reg.register_fn(nid, name, "libKernel", cb);
    }
}

#[cfg(test)]
mod tests;
