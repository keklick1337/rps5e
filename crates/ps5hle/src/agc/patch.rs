// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceAgc helpers that inspect or patch packets already written into a
//! command buffer (`*Patch*`, predication bits, packet sizes, NOP fill).

use super::pm4::{self, nop, op};
use super::{arg, mem, Export, ERR_INVALID_ARGUMENT, ERR_INVALID_PACKET, ERR_MEMORY_FAULT, LIB_AGC, OK};
use crate::context::CpuContext;
use crate::registry::HleResult;

fn identity(packet: u64) -> Option<(u32, u32, u32)> {
    if packet == 0 {
        return None;
    }
    let h = mem::r32(packet)?;
    Some((pm4::opcode(h), pm4::custom(h), h))
}

fn patch_bits(addr: u64, mask: u32, value: u32) -> bool {
    match mem::r32(addr) {
        Some(cur) => mem::w32(addr, (cur & !mask) | (value & mask)),
        None => false,
    }
}

fn status(ok: bool) -> HleResult {
    if ok {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

/// `sceAgcGetDataPacketPayloadAddress(out, packet, type)`.
fn get_data_packet_payload_address(ctx: &mut CpuContext) -> HleResult {
    let (out, packet, ty) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2) as i32);
    if out == 0 || packet == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let payload = if ty == 0 {
        let Some(h) = mem::r32(packet) else { return ERR_MEMORY_FAULT };
        if h & 0x3FFF_0000 == 0x3FFF_0000 {
            0
        } else {
            packet + 4
        }
    } else {
        packet + 8
    };
    status(mem::w64(out, payload))
}

/// `sceAgcGetDataPacketPayloadRange(range{base, size}, packet, type)`.
fn get_data_packet_payload_range(ctx: &mut CpuContext) -> HleResult {
    let (range, packet, ty) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2) as i32);
    if range == 0 || packet == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(h) = mem::r32(packet) else { return ERR_MEMORY_FAULT };
    let payload_bytes = ((h >> 14) & 0xFFFC) as u64;
    let (base, size) = if ty != 0 {
        (packet + 8, payload_bytes)
    } else if !h & 0x3FFF_0000 == 0 {
        (0, 0)
    } else {
        (packet + 4, payload_bytes + 4)
    };
    status(mem::w64(range, base) && mem::w64(range + 8, size))
}

/// `sceAgcSetPacketPredication(packet, predication)`: header bit 0.
fn set_packet_predication(ctx: &mut CpuContext) -> HleResult {
    let (packet, pred) = (arg(ctx, 0), arg(ctx, 1));
    let Some(h) = (packet != 0).then(|| mem::r32(packet)).flatten() else { return ERR_INVALID_ARGUMENT };
    status(mem::w32(packet, (h & !1) | (pred & 0xFF == 1) as u32))
}

/// `sceAgcGetPacketSize(packet)` -> dwords.
fn get_packet_size(ctx: &mut CpuContext) -> HleResult {
    mem::r32(arg(ctx, 0)).map_or(0, |h| pm4::packet_size(h) as HleResult)
}

/// `sceAgcDcbGetSystemSoftwareVersion(u32* out, packet)`: the first
/// payload dword of a type-3 packet of at least three dwords.
fn get_system_software_version(ctx: &mut CpuContext) -> HleResult {
    let (out, packet) = (arg(ctx, 0), arg(ctx, 1));
    if out == 0 || out & 3 != 0 || packet == 0 || packet & 3 != 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(h) = mem::r32(packet) else { return ERR_MEMORY_FAULT };
    if pm4::packet_type(h) != 3 || pm4::length(h) < 3 {
        return ERR_INVALID_PACKET;
    }
    let Some(version) = mem::r32(packet + 4) else { return ERR_MEMORY_FAULT };
    status(mem::w32(out, version))
}

/// `sceAgcSetRangePredication(start, end, predication)`: set or clear the
/// predicate bit on every packet in `[start, end)`.
fn set_range_predication(ctx: &mut CpuContext) -> HleResult {
    let (mut p, end, pred) = (arg(ctx, 0), arg(ctx, 1), (arg(ctx, 2) & 0xFF == 1) as u32);
    while p < end {
        let Some(h) = mem::r32(p) else { return ERR_MEMORY_FAULT };
        if !mem::w32(p, (h & !1) | pred) {
            return ERR_MEMORY_FAULT;
        }
        p += pm4::packet_size(h) as u64 * 4;
    }
    OK
}

/// Signature of a marker in the first payload dword of a NOP.
const MARKER_SIGNATURE: u32 = 0x6875_0000;

/// `sceAgcSetNop(packet)`: disable a packet already in a command buffer. It
/// becomes a NOP of the same length: the type, count, shader-type and
/// predicate bits stay, the opcode becomes NOP and the private custom code
/// is cleared, so a packet the library encodes as a custom NOP (an
/// end-of-pipe release, a flip wait) stops executing too. A first payload
/// dword carrying the marker signature loses it, so the disabled packet is
/// not read as a marker. The rest of the payload is left alone. The call
/// takes the packet only (callers do not set a second argument).
fn set_nop(ctx: &mut CpuContext) -> HleResult {
    let packet = arg(ctx, 0);
    if packet == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(h) = mem::r32(packet) else { return ERR_MEMORY_FAULT };
    let size = pm4::packet_size(h);
    if !mem::w32(packet, (h & 0xFFFF_0003) | (op::NOP << 8)) {
        return ERR_MEMORY_FAULT;
    }
    if size > 1 {
        let Some(first) = mem::r32(packet + 4) else { return ERR_MEMORY_FAULT };
        if first & 0xFFFF_0000 == MARKER_SIGNATURE && !mem::w32(packet + 4, first & 0xFFFF) {
            return ERR_MEMORY_FAULT;
        }
    }
    OK
}

/// `sceAgc[Async]RewindPatchSetRewindState(packet, u8 valid)`: only the
/// opcode is inspected; the header and the low 31 payload bits (offload
/// included) are kept.
fn rewind_patch_set_rewind_state(ctx: &mut CpuContext) -> HleResult {
    let (packet, valid) = (arg(ctx, 0), arg(ctx, 1) & 1 != 0);
    match identity(packet) {
        Some((o, _, _)) if o == op::REWIND => {}
        Some(_) => return ERR_INVALID_PACKET,
        None => return if packet == 0 { ERR_INVALID_PACKET } else { ERR_MEMORY_FAULT },
    }
    let body = packet + 4;
    let Some(b) = mem::r32(body) else { return ERR_MEMORY_FAULT };
    status(mem::w32(body, if valid { b | (1 << 31) } else { b & !(1 << 31) }))
}

/// `sceAgc[Async]CondExecPatchSetCommandAddress(packet, predicateAddr)`.
fn cond_exec_patch_set_command_address(ctx: &mut CpuContext) -> HleResult {
    let (packet, addr) = (arg(ctx, 0), arg(ctx, 1));
    match identity(packet) {
        Some((o, _, _)) if o == op::COND_EXEC && addr != 0 && addr & 3 == 0 => {}
        _ => return ERR_INVALID_PACKET,
    }
    let Some(d1) = mem::r32(packet + 4) else { return ERR_MEMORY_FAULT };
    status(mem::w32(packet + 4, (d1 & 3) | ((addr as u32) & !3)) && mem::w32(packet + 8, (addr >> 32) as u32))
}

/// `sceAgc[Async]CondExecPatchSetEnd(packet, end)`: the skip count becomes
/// the dwords between the packet's end and `end`.
fn cond_exec_patch_set_end(ctx: &mut CpuContext) -> HleResult {
    let (packet, end) = (arg(ctx, 0), arg(ctx, 1));
    match identity(packet) {
        Some((o, _, _)) if o == op::COND_EXEC && end != 0 => {}
        _ => return ERR_INVALID_PACKET,
    }
    let packet_end = packet + 20;
    if end < packet_end {
        return ERR_INVALID_PACKET;
    }
    let n = (end - packet_end) / 4;
    if n > 0x3FFF {
        return ERR_INVALID_PACKET;
    }
    status(patch_bits(packet + 16, 0x3FFF, n as u32))
}

/// Inferred: `sceAgcBranchPatchSetCompareAddress(packet, addr)` on the
/// 14-dword branch packet (compare address in dwords 2..3).
fn branch_patch_set_compare_address(ctx: &mut CpuContext) -> HleResult {
    let (packet, addr) = (arg(ctx, 0), arg(ctx, 1));
    match identity(packet) {
        Some((o, _, h)) if o == op::INDIRECT_BUFFER && pm4::length(h) == 14 => {}
        _ => return ERR_INVALID_ARGUMENT,
    }
    status(mem::w32(packet + 8, (addr as u32) & !7) && mem::w32(packet + 12, (addr >> 32) as u32))
}

/// Dword of the then-target (address low, address high, size) in a
/// 14-dword branch packet, and of the else-target.
const BRANCH_THEN: u64 = 8;
const BRANCH_ELSE: u64 = 11;

/// Retarget the then- or else-buffer (dwords `field..field + 3`) of a
/// 14-dword branch to `dwords` dwords at `target`. The low two address bits
/// and the reserved bits of the size dword are kept; `cache` (0..=3), when
/// given, replaces the cache policy in bits 28..29 of the size dword. A
/// target that is null or not dword aligned, a size of more than 20 bits
/// and a policy above 3 are refused before anything is written.
fn branch_patch_target(packet: u64, field: u64, cache: Option<u64>, target: u64, dwords: u64) -> HleResult {
    match identity(packet) {
        Some((o, _, h)) if o == op::INDIRECT_BUFFER && pm4::length(h) == 14 => {}
        Some(_) => return ERR_INVALID_PACKET,
        None => return if packet == 0 { ERR_INVALID_PACKET } else { ERR_MEMORY_FAULT },
    }
    if target == 0 || target & 3 != 0 || dwords > 0xF_FFFF || cache.is_some_and(|c| c > 3) {
        return ERR_INVALID_ARGUMENT;
    }
    let words = packet + field * 4;
    let (mask, bits) = match cache {
        Some(c) => (0x300F_FFFF, ((c as u32) << 28) | dwords as u32),
        None => (0xF_FFFF, dwords as u32),
    };
    status(patch_bits(words, !3, target as u32) && mem::w32(words + 4, (target >> 32) as u32) && patch_bits(words + 8, mask, bits))
}

/// `sceAgcBranchPatchSetThenTarget(packet, target, dwords)`.
fn branch_patch_set_then_target(ctx: &mut CpuContext) -> HleResult {
    branch_patch_target(arg(ctx, 0), BRANCH_THEN, None, arg(ctx, 1), arg(ctx, 2) & 0xFFFF_FFFF)
}

/// `sceAgcBranchPatchSetElseTarget(packet, target, dwords)`.
fn branch_patch_set_else_target(ctx: &mut CpuContext) -> HleResult {
    branch_patch_target(arg(ctx, 0), BRANCH_ELSE, None, arg(ctx, 1), arg(ctx, 2) & 0xFFFF_FFFF)
}

/// `sceAgcBranchPatchSetThenTarget_0300(packet, cachePolicy, target, dwords)`.
fn branch_patch_set_then_target_0300(ctx: &mut CpuContext) -> HleResult {
    branch_patch_target(arg(ctx, 0), BRANCH_THEN, Some(arg(ctx, 1) & 0xFF), arg(ctx, 2), arg(ctx, 3) & 0xFFFF_FFFF)
}

/// `sceAgcBranchPatchSetElseTarget_0300(packet, cachePolicy, target, dwords)`.
fn branch_patch_set_else_target_0300(ctx: &mut CpuContext) -> HleResult {
    branch_patch_target(arg(ctx, 0), BRANCH_ELSE, Some(arg(ctx, 1) & 0xFF), arg(ctx, 2), arg(ctx, 3) & 0xFFFF_FFFF)
}

/// Jump-packet retarget `(packet, cache, target, dwords)`; keeps the low
/// address bits and the call/chain mode and other control bits.
fn jump_patch_retarget(ctx: &mut CpuContext) -> HleResult {
    let (packet, cache, target, dw) = (arg(ctx, 0), arg(ctx, 1) as u32 & 3, arg(ctx, 2), arg(ctx, 3) as u32);
    match identity(packet) {
        Some((o, _, _)) if o == op::INDIRECT_BUFFER => {}
        _ => return ERR_INVALID_PACKET,
    }
    status(
        patch_bits(packet + 4, !3, target as u32)
            && mem::w32(packet + 8, (target >> 32) as u32)
            && patch_bits(packet + 12, 0x300F_FFFF, (cache << 28) | (dw & 0xFFFFF)),
    )
}

/// `sceAgcJumpPatchSetTarget(packet, target, dwords)`.
fn jump_patch_set_target(ctx: &mut CpuContext) -> HleResult {
    let (packet, target, dw) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2) as u32);
    match identity(packet) {
        Some((o, _, _)) if o == op::INDIRECT_BUFFER => {}
        _ => return ERR_INVALID_PACKET,
    }
    status(
        mem::w32(packet + 4, target as u32)
            && patch_bits(packet + 8, 0xFFFF, (target >> 32) as u32)
            && patch_bits(packet + 12, 0xFFFFF, dw),
    )
}

#[derive(Clone, Copy, PartialEq)]
enum Bank {
    Cx,
    Sh,
    Uc,
}

/// The native SET_*_REG_INDIRECT packet: address in dwords 1..2, the
/// register count in the low 14 bits of dword 4.
fn indirect_packet(packet: u64, bank: Bank) -> bool {
    let native = match bank {
        Bank::Cx => op::SET_CONTEXT_REG_INDIRECT,
        Bank::Sh => op::SET_SH_REG_INDIRECT,
        Bank::Uc => op::SET_UCONFIG_REG_INDIRECT,
    };
    matches!(identity(packet), Some((o, _, _)) if o == native)
}

fn indirect_set_address(ctx: &CpuContext, bank: Bank) -> HleResult {
    let (packet, regs) = (arg(ctx, 0), arg(ctx, 1));
    if !indirect_packet(packet, bank) {
        return ERR_INVALID_PACKET;
    }
    status(patch_bits(packet + 4, !3, regs as u32) && mem::w32(packet + 8, (regs >> 32) as u32))
}

fn indirect_set_count(ctx: &CpuContext, bank: Bank) -> HleResult {
    let (packet, n) = (arg(ctx, 0), arg(ctx, 1) as u32);
    if !indirect_packet(packet, bank) {
        return ERR_INVALID_PACKET;
    }
    status(patch_bits(packet + 16, 0x3FFF, n))
}

fn indirect_add_registers(ctx: &CpuContext, bank: Bank) -> HleResult {
    let (packet, n) = (arg(ctx, 0), arg(ctx, 1) as u32);
    if !indirect_packet(packet, bank) {
        return ERR_INVALID_PACKET;
    }
    let Some(cur) = mem::r32(packet + 16) else { return ERR_MEMORY_FAULT };
    status(mem::w32(packet + 16, (cur & !0x3FFF) | ((cur & 0x3FFF).wrapping_add(n) & 0x3FFF)))
}

fn cx_set_address(c: &mut CpuContext) -> HleResult {
    indirect_set_address(c, Bank::Cx)
}
fn sh_set_address(c: &mut CpuContext) -> HleResult {
    indirect_set_address(c, Bank::Sh)
}
fn uc_set_address(c: &mut CpuContext) -> HleResult {
    indirect_set_address(c, Bank::Uc)
}
fn cx_set_count(c: &mut CpuContext) -> HleResult {
    indirect_set_count(c, Bank::Cx)
}
fn sh_set_count(c: &mut CpuContext) -> HleResult {
    indirect_set_count(c, Bank::Sh)
}
fn uc_set_count(c: &mut CpuContext) -> HleResult {
    indirect_set_count(c, Bank::Uc)
}
fn cx_add(c: &mut CpuContext) -> HleResult {
    indirect_add_registers(c, Bank::Cx)
}
fn sh_add(c: &mut CpuContext) -> HleResult {
    indirect_add_registers(c, Bank::Sh)
}
fn uc_add(c: &mut CpuContext) -> HleResult {
    indirect_add_registers(c, Bank::Uc)
}

fn is_dma_packet(packet: u64) -> bool {
    matches!(identity(packet), Some((o, _, _)) if o == op::DMA_DATA)
}

fn dma_patch_dst(ctx: &mut CpuContext) -> HleResult {
    let (packet, v) = (arg(ctx, 0), arg(ctx, 1));
    if !is_dma_packet(packet) {
        return ERR_INVALID_PACKET;
    }
    status(mem::w32(packet + 16, v as u32) && mem::w32(packet + 20, (v >> 32) as u32))
}

fn dma_patch_src(ctx: &mut CpuContext) -> HleResult {
    let (packet, v) = (arg(ctx, 0), arg(ctx, 1));
    if !is_dma_packet(packet) {
        return ERR_INVALID_PACKET;
    }
    status(mem::w32(packet + 8, v as u32) && mem::w32(packet + 12, (v >> 32) as u32))
}

fn is_write_data(packet: u64) -> bool {
    matches!(identity(packet), Some((o, _, _)) if o == op::WRITE_DATA)
}

/// WRITE_DATA cache policy: control bits 25..26.
fn write_data_patch_cache_policy(ctx: &mut CpuContext) -> HleResult {
    let (packet, v) = (arg(ctx, 0), arg(ctx, 1) as u32);
    if !is_write_data(packet) {
        return ERR_INVALID_PACKET;
    }
    status(patch_bits(packet + 4, 3 << 25, (v & 3) << 25))
}

/// WRITE_DATA destination as the graphics builder encodes it: bit 0 the
/// engine (control bit 30), bits 1..4 DST_SEL (control bits 8..11).
fn write_data_patch_dst(ctx: &mut CpuContext) -> HleResult {
    let (packet, v) = (arg(ctx, 0), arg(ctx, 1) as u32);
    if !is_write_data(packet) {
        return ERR_INVALID_PACKET;
    }
    status(patch_bits(packet + 4, (1 << 30) | 0xF00, ((v & 1) << 30) | ((v & 0x1E) << 7)))
}

/// `sceAgcWriteDataPatchSetAddressOrOffset(packet, addr)`. The asynchronous
/// form writes the same two address dwords.
fn write_data_patch_address(ctx: &mut CpuContext) -> HleResult {
    let (packet, v) = (arg(ctx, 0), arg(ctx, 1));
    if !is_write_data(packet) {
        return ERR_INVALID_PACKET;
    }
    status(mem::w32(packet + 8, v as u32) && mem::w32(packet + 12, (v >> 32) as u32))
}

/// `sceAgcAsyncWriteDataPatchSetDst(packet, uint8 dst)`: the compute form of
/// the destination, DST_SEL in control bits 8..11. A value above 0xF is
/// rejected rather than cut down, and so is destination 0 on a packet with
/// write confirmation set (a register write cannot confirm).
fn async_write_data_patch_dst(ctx: &mut CpuContext) -> HleResult {
    let (packet, dst) = (arg(ctx, 0), arg(ctx, 1) as u8);
    if !is_write_data(packet) {
        return ERR_INVALID_PACKET;
    }
    if dst > 0xF {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(control) = mem::r32(packet + 4) else { return ERR_MEMORY_FAULT };
    if dst == 0 && (control & (1 << 20)) != 0 {
        return ERR_INVALID_ARGUMENT;
    }
    status(patch_bits(packet + 4, 0xF << 8, (dst as u32) << 8))
}

/// `sceAgcAsyncWriteDataPatchSetCachePolicy(packet, uint8 policy)`: control
/// bits 25..26; a value above 3 is rejected rather than cut down.
fn async_write_data_patch_cache_policy(ctx: &mut CpuContext) -> HleResult {
    let (packet, policy) = (arg(ctx, 0), arg(ctx, 1) as u8);
    if !is_write_data(packet) {
        return ERR_INVALID_PACKET;
    }
    if policy > 3 {
        return ERR_INVALID_ARGUMENT;
    }
    status(patch_bits(packet + 4, 3 << 25, (policy as u32) << 25))
}

/// The wait packet behind a wait-on-address begin tag, with its alignment
/// mask (3 for the 32-bit wait, 7 for the 64-bit one).
fn wait_packet(packet: u64) -> Option<(u64, u32, bool)> {
    use super::pm4::{WAIT_USER_DATA_BEGIN32_TAG, WAIT_USER_DATA_BEGIN64_TAG};
    let mut tag = [0u32; 4];
    if packet == 0 || !mem::read(packet, bytes_of(&mut tag)) || pm4::length(tag[0]) != 4 || !pm4::is_internal_data_packet(tag[0], &tag[1..]) {
        return None;
    }
    let wait = packet + 16;
    let h = mem::r32(wait)?;
    let (o, len) = (pm4::opcode(h), pm4::length(h));
    let t = tag[2] & 0xFFFF_0000;
    if pm4::custom(h) != 0 {
        return None;
    }
    if o == op::WAIT_REG_MEM && len == 7 && t == WAIT_USER_DATA_BEGIN32_TAG {
        Some((wait, 3, false))
    } else if o == op::WAIT_REG_MEM64 && len == 9 && t == WAIT_USER_DATA_BEGIN64_TAG {
        Some((wait, 7, true))
    } else {
        None
    }
}

fn bytes_of(w: &mut [u32]) -> &mut [u8] {
    // SAFETY: u8 has no alignment or validity requirements.
    unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr() as *mut u8, w.len() * 4) }
}

/// Updates the address in the begin tag and in the wait (the wait keeps
/// its low alignment bits).
fn wait_patch_address(ctx: &mut CpuContext) -> HleResult {
    let (packet, addr) = (arg(ctx, 0), arg(ctx, 1));
    let Some((wait, align, _)) = wait_packet(packet) else { return ERR_INVALID_PACKET };
    status(
        patch_bits(packet + 8, 0xFFFF, (addr >> 32) as u32)
            && mem::w32(packet + 12, addr as u32)
            && patch_bits(wait + 8, !align, addr as u32)
            && patch_bits(wait + 12, 0x3FFFF, (addr >> 32) as u32),
    )
}

fn wait_patch_compare_function(ctx: &mut CpuContext) -> HleResult {
    let (packet, f) = (arg(ctx, 0), arg(ctx, 1) as u32);
    let Some((wait, _, _)) = wait_packet(packet) else { return ERR_INVALID_PACKET };
    status(patch_bits(wait + 4, 7, f))
}

/// Only the low reference dword changes, for both wait sizes.
fn wait_patch_reference(ctx: &mut CpuContext) -> HleResult {
    let (packet, v) = (arg(ctx, 0), arg(ctx, 1));
    let Some((wait, _, _)) = wait_packet(packet) else { return ERR_INVALID_PACKET };
    status(mem::w32(wait + 16, v as u32))
}

fn wait_patch_mask(ctx: &mut CpuContext) -> HleResult {
    let (packet, v) = (arg(ctx, 0), arg(ctx, 1));
    match wait_packet(packet) {
        Some((wait, _, false)) => status(mem::w32(wait + 20, v as u32)),
        Some((wait, _, true)) => status(mem::w32(wait + 24, v as u32) && mem::w32(wait + 28, (v >> 32) as u32)),
        None => ERR_INVALID_PACKET,
    }
}

/// A release packet: the private NOP form or native RELEASE_MEM (same
/// dword layout).
fn is_release_mem(packet: u64) -> bool {
    matches!(identity(packet), Some((o, c, _)) if o == op::RELEASE_MEM || (o == op::NOP && c == nop::RELEASE_MEM))
}

fn eop_patch_address(ctx: &mut CpuContext) -> HleResult {
    let (packet, v) = (arg(ctx, 0), arg(ctx, 1));
    match identity(packet) {
        _ if is_release_mem(packet) => status(mem::w32(packet + 12, v as u32) && mem::w32(packet + 16, (v >> 32) as u32)),
        Some((o, _, _)) if o == op::EVENT_WRITE_EOP => {
            status(mem::w32(packet + 8, v as u32) && patch_bits(packet + 12, 0xFFFF, (v >> 32) as u32))
        }
        _ => ERR_INVALID_PACKET,
    }
}

/// GCR control: dword 1 bits 12..23.
fn eop_patch_gcr(ctx: &mut CpuContext) -> HleResult {
    let (packet, v) = (arg(ctx, 0), arg(ctx, 1) as u32);
    if !is_release_mem(packet) {
        return ERR_INVALID_PACKET;
    }
    status(patch_bits(packet + 4, 0xFFF << 12, (v & 0xFFF) << 12))
}

/// No effect when the packet only interrupts or reads GDS.
fn eop_patch_data(ctx: &mut CpuContext) -> HleResult {
    let (packet, v) = (arg(ctx, 0), arg(ctx, 1));
    if !is_release_mem(packet) {
        return ERR_INVALID_PACKET;
    }
    let Some(control) = mem::r32(packet + 8) else { return ERR_MEMORY_FAULT };
    if (control >> 24) & 7 == 4 || (control >> 29) & 7 == 5 {
        return OK;
    }
    status(mem::w32(packet + 20, v as u32) && mem::w32(packet + 24, (v >> 32) as u32))
}

/// Data selector: dword 2 bits 29..31.
fn eop_patch_type(ctx: &mut CpuContext) -> HleResult {
    let (packet, sel) = (arg(ctx, 0), arg(ctx, 1) as u32);
    if !is_release_mem(packet) {
        return ERR_INVALID_PACKET;
    }
    status(patch_bits(packet + 8, 7 << 29, (sel & 7) << 29))
}

pub(crate) const EXPORTS: &[Export] = &[
    ("V++UgBtQhn0", "sceAgcGetDataPacketPayloadAddress_0090", LIB_AGC, get_data_packet_payload_address),
    ("CQsSq6l6+kA", "sceAgcGetDataPacketPayloadAddress", LIB_AGC, get_data_packet_payload_address),
    ("s+VGAMDQ0AQ", "sceAgcGetDataPacketPayloadRange", LIB_AGC, get_data_packet_payload_range),
    ("w6Dj1VJt5qY", "sceAgcSetPacketPredication", LIB_AGC, set_packet_predication),
    ("n8vgpaQg6dA", "sceAgcSetRangePredication", LIB_AGC, set_range_predication),
    ("Lkf86B98qPc", "sceAgcGetPacketSize", LIB_AGC, get_packet_size),
    ("8bEoPltS6gs", "sceAgcDcbGetSystemSoftwareVersion", LIB_AGC, get_system_software_version),
    ("K2mciNVxUCE", "sceAgcSetNop", LIB_AGC, set_nop),
    ("ziVA3whp3p4", "sceAgcRewindPatchSetRewindState", LIB_AGC, rewind_patch_set_rewind_state),
    ("eWaWyFegzgQ", "sceAgcAsyncRewindPatchSetRewindState", LIB_AGC, rewind_patch_set_rewind_state),
    ("YWTKOju587o", "sceAgcCondExecPatchSetCommandAddress", LIB_AGC, cond_exec_patch_set_command_address),
    ("3ZWa3AoyWZQ", "sceAgcAsyncCondExecPatchSetCommandAddress", LIB_AGC, cond_exec_patch_set_command_address),
    ("ORWsxIbk4TE", "sceAgcCondExecPatchSetEnd", LIB_AGC, cond_exec_patch_set_end),
    ("k-JpyR2dYAM", "sceAgcAsyncCondExecPatchSetEnd", LIB_AGC, cond_exec_patch_set_end),
    ("GXBlM-ekzrI", "sceAgcBranchPatchSetCompareAddress", LIB_AGC, branch_patch_set_compare_address),
    ("xb8VgcXQhvI", "sceAgcBranchPatchSetThenTarget", LIB_AGC, branch_patch_set_then_target),
    ("QmfvaYpsOcI", "sceAgcBranchPatchSetElseTarget", LIB_AGC, branch_patch_set_else_target),
    ("7Wa3aeJgeVU", "sceAgcBranchPatchSetThenTarget_0300", LIB_AGC, branch_patch_set_then_target_0300),
    ("rP5xLdOf26k", "sceAgcBranchPatchSetElseTarget_0300", LIB_AGC, branch_patch_set_else_target_0300),
    ("Ikfdt-rIqCE", "sceAgcJumpPatchSetTarget_0300", LIB_AGC, jump_patch_retarget),
    ("2BS4EtAaF28", "sceAgcJumpPatchSetTarget", LIB_AGC, jump_patch_set_target),
    ("vcmNN+AAXnY", "sceAgcSetCxRegIndirectPatchSetAddress", LIB_AGC, cx_set_address),
    ("Qrj4c+61z4A", "sceAgcSetShRegIndirectPatchSetAddress", LIB_AGC, sh_set_address),
    ("6lNcCp+fxi4", "sceAgcSetUcRegIndirectPatchSetAddress", LIB_AGC, uc_set_address),
    ("whb1RL7K4Ss", "sceAgcSetCxRegIndirectPatchSetNumRegisters", LIB_AGC, cx_set_count),
    ("nCUgItdN2ms", "sceAgcSetShRegIndirectPatchSetNumRegisters", LIB_AGC, sh_set_count),
    ("fRG-JOH5+sI", "sceAgcSetUcRegIndirectPatchSetNumRegisters", LIB_AGC, uc_set_count),
    ("d-6uF9sZDIU", "sceAgcSetCxRegIndirectPatchAddRegisters", LIB_AGC, cx_add),
    ("z2duB-hHQSM", "sceAgcSetShRegIndirectPatchAddRegisters", LIB_AGC, sh_add),
    ("vRoArM9zaIk", "sceAgcSetUcRegIndirectPatchAddRegisters", LIB_AGC, uc_add),
    ("IxYiarKlXxM", "sceAgcDmaDataPatchSetDstAddressOrOffset", LIB_AGC, dma_patch_dst),
    ("cdDRpqcFGbU", "sceAgcDmaDataPatchSetSrcAddressOrOffsetOrImmediate", LIB_AGC, dma_patch_src),
    ("eAy8eGNsCuU", "sceAgcWriteDataPatchSetCachePolicy", LIB_AGC, write_data_patch_cache_policy),
    ("tmy-+rBpspY", "sceAgcWriteDataPatchSetDst", LIB_AGC, write_data_patch_dst),
    ("fPSCdQxgpSw", "sceAgcWriteDataPatchSetAddressOrOffset", LIB_AGC, write_data_patch_address),
    ("d4NZIlguzv0", "sceAgcAsyncWriteDataPatchSetAddressOrOffset", LIB_AGC, write_data_patch_address),
    ("EJBA4dbmvfg", "sceAgcAsyncWriteDataPatchSetDst", LIB_AGC, async_write_data_patch_dst),
    ("y5K5tPktiL8", "sceAgcAsyncWriteDataPatchSetCachePolicy", LIB_AGC, async_write_data_patch_cache_policy),
    ("3KDcnM3lrcU", "sceAgcWaitRegMemPatchAddress", LIB_AGC, wait_patch_address),
    ("n485EBnIWmk", "sceAgcWaitRegMemPatchCompareFunction", LIB_AGC, wait_patch_compare_function),
    ("7nOoijNPvEU", "sceAgcWaitRegMemPatchReference", LIB_AGC, wait_patch_reference),
    ("hXAnLgDHCoI", "sceAgcWaitRegMemPatchMask", LIB_AGC, wait_patch_mask),
    ("0fWWK5uG9rQ", "sceAgcQueueEndOfPipeActionPatchAddress", LIB_AGC, eop_patch_address),
    ("J8YCgfKAMQs", "sceAgcQueueEndOfPipeActionPatchGcrCntl", LIB_AGC, eop_patch_gcr),
    ("MlEw1feXcjg", "sceAgcQueueEndOfPipeActionPatchData", LIB_AGC, eop_patch_data),
    ("T9fjQIINoeE", "sceAgcQueueEndOfPipeActionPatchType", LIB_AGC, eop_patch_type),
];
