// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! PM4 (type-3) command packets: header encoding, opcodes, the private NOP
//! custom codes used by libSceAgc, and the command-stream interpreter that
//! executes submitted command buffers on a dedicated GPU thread.
//!
//! Header: `0xC0000000 | (dwords - 2) << 16 | opcode << 8 | custom << 2`,
//! bit 0 = predicate, bit 1 = shader type (SET_BASE / dispatch packets).
//! Any type-2 header (bits 31:30 = 2, e.g. `0x80000000`) is a one-dword
//! filler.

pub mod interp;
pub mod queue;

/// One-dword filler (type-2 packet; the other bits are ignored).
pub const FILLER: u32 = 0x8000_0000;
/// Largest packet length in dwords (count field + 2).
pub const MAX_LENGTH: u32 = 0x3FFF + 2;

#[inline]
pub const fn length(h: u32) -> u32 {
    ((h >> 16) & 0x3FFF) + 2
}

/// Size in dwords of the packet whose header is `h`, as the library and the
/// command processor walk a buffer: a type-2 header and a NOP whose count
/// field is all ones (`make(1, NOP, _)`, e.g. 0xFFFF1000) are one-dword
/// packets.
#[inline]
pub const fn packet_size(h: u32) -> u32 {
    if h >> 30 == 2 || h & 0x3FFF_FF00 == 0x3FFF_1000 {
        1
    } else {
        length(h)
    }
}

#[inline]
pub const fn opcode(h: u32) -> u32 {
    (h >> 8) & 0xFF
}

#[inline]
pub const fn custom(h: u32) -> u32 {
    (h >> 2) & 0x3F
}

#[inline]
pub const fn packet_type(h: u32) -> u32 {
    h >> 30
}

#[inline]
pub const fn predicated(h: u32) -> bool {
    h & 1 != 0
}

/// Build a type-3 header for a packet of `dwords` dwords (header included).
#[inline]
pub const fn make(dwords: u32, op: u32, custom: u32) -> u32 {
    0xC000_0000 | ((dwords.wrapping_sub(2) & 0x3FFF) << 16) | ((op & 0xFF) << 8) | ((custom & 0x3F) << 2)
}

/// Type-3 opcodes.
pub mod op {
    pub const NOP: u32 = 0x10;
    pub const SET_BASE: u32 = 0x11;
    pub const CLEAR_STATE: u32 = 0x12;
    pub const INDEX_BUFFER_SIZE: u32 = 0x13;
    pub const DISPATCH_DIRECT: u32 = 0x15;
    pub const DISPATCH_INDIRECT: u32 = 0x16;
    pub const ATOMIC_GDS: u32 = 0x1D;
    pub const ATOMIC_MEM: u32 = 0x1E;
    pub const SET_PREDICATION: u32 = 0x20;
    pub const COND_EXEC: u32 = 0x22;
    pub const DRAW_INDIRECT: u32 = 0x24;
    pub const DRAW_INDEX_INDIRECT: u32 = 0x25;
    pub const INDEX_BASE: u32 = 0x26;
    pub const INDEX_ATTRIBUTES_INDIRECT: u32 = 0x91;
    pub const DRAW_INDEX_2: u32 = 0x27;
    pub const CONTEXT_CONTROL: u32 = 0x28;
    pub const INDEX_TYPE: u32 = 0x2A;
    pub const DRAW_INDIRECT_MULTI: u32 = 0x2C;
    pub const DRAW_INDEX_AUTO: u32 = 0x2D;
    pub const NUM_INSTANCES: u32 = 0x2F;
    pub const DRAW_INDEX_MULTI_AUTO: u32 = 0x30;
    pub const DRAW_INDEX_OFFSET_2: u32 = 0x35;
    pub const WRITE_DATA: u32 = 0x37;
    pub const DRAW_INDEX_INDIRECT_MULTI: u32 = 0x38;
    pub const MEM_SEMAPHORE: u32 = 0x39;
    pub const DISPATCH_DRAW_PREAMBLE: u32 = 0x3A;
    pub const WAIT_REG_MEM: u32 = 0x3C;
    pub const INDIRECT_BUFFER: u32 = 0x3F;
    pub const COPY_DATA: u32 = 0x40;
    pub const PFP_SYNC_ME: u32 = 0x42;
    pub const COND_WRITE: u32 = 0x45;
    pub const EVENT_WRITE: u32 = 0x46;
    pub const EVENT_WRITE_EOP: u32 = 0x47;
    pub const EVENT_WRITE_EOS: u32 = 0x48;
    pub const RELEASE_MEM: u32 = 0x49;
    pub const DMA_DATA: u32 = 0x50;
    pub const ACQUIRE_MEM: u32 = 0x58;
    pub const REWIND: u32 = 0x59;
    pub const PRIME_UTCL2: u32 = 0x5A;
    pub const SET_SH_REG_INDIRECT: u32 = 0x63;
    pub const SET_UCONFIG_REG_INDIRECT: u32 = 0x64;
    pub const SET_CONFIG_REG: u32 = 0x68;
    pub const SET_CONTEXT_REG: u32 = 0x69;
    pub const SET_SH_REG: u32 = 0x76;
    pub const SET_UCONFIG_REG: u32 = 0x79;
    pub const SET_UCONFIG_REG_INDEX: u32 = 0x7A;
    pub const WRITE_CONST_RAM: u32 = 0x81;
    pub const DUMP_CONST_RAM: u32 = 0x83;
    pub const INCREMENT_CE_COUNTER: u32 = 0x84;
    pub const INCREMENT_DE_COUNTER: u32 = 0x85;
    pub const WAIT_ON_CE_COUNTER: u32 = 0x86;
    pub const WAIT_ON_DE_COUNTER_DIFF: u32 = 0x88;
    pub const GET_LOD_STATS: u32 = 0x8E;
    pub const WAIT_REG_MEM64: u32 = 0x93;
    pub const SET_CONTEXT_REG_INDIRECT: u32 = 0x9F;
}

/// Custom codes carried in a NOP header (libSceAgc private packets).
pub mod nop {
    pub const ZERO: u32 = 0x00;
    pub const DRAW_RESET: u32 = 0x05;
    pub const WAIT_FLIP_DONE: u32 = 0x06;
    pub const DISPATCH_RESET: u32 = 0x09;
    pub const PUSH_MARKER: u32 = 0x0B;
    pub const POP_MARKER: u32 = 0x0C;
    pub const ACQUIRE_MEM: u32 = 0x14;
    pub const WRITE_DATA: u32 = 0x15;
    pub const FLIP: u32 = 0x17;
    pub const RELEASE_MEM: u32 = 0x18;
    pub const DMA_DATA: u32 = 0x19;
    pub const CONTEXT_STATE: u32 = 0x1A;
}

/// Register bank sizes, nop registers and offset selector bits (the
/// register file lives in the GPU crate, shared with the slice recorder).
pub use ps5gpu::render::regfile::{
    CONTEXT_NOP, CONTEXT_REGISTER_COUNT, REGISTER_SELECTOR_MASK, SHADER_NOP, SHADER_REGISTER_COUNT, UCONFIG_NOP, UCONFIG_REGISTER_COUNT,
};
/// VGT_INDEX_TYPE (user-config bank).
pub const VGT_INDEX_TYPE: u32 = 0x243;

/// libSceAgc private-data packets: SET_UCONFIG_REG (custom 1) writes to
/// register 0x342 carrying a tag, bracketing multi-draws and user waits.
pub const INTERNAL_DATA_REGISTER: u32 = 0x342;
pub const DRAW_INDIRECT_MULTI_BEGIN_TAG: u32 = 0xC600_0008;
pub const DRAW_INDIRECT_MULTI_END_TAG: u32 = 0xC600_0000;
pub const WAIT_USER_DATA_BEGIN32_TAG: u32 = 0xC801_0000;
pub const WAIT_USER_DATA_BEGIN64_TAG: u32 = 0xC802_0000;
pub const WAIT_USER_DATA_END_TAG: u32 = 0xC800_0000;

/// True for a libSceAgc private-data packet (`payload` excludes the header).
pub fn is_internal_data_packet(h: u32, payload: &[u32]) -> bool {
    if payload.len() < 2 || opcode(h) != op::SET_UCONFIG_REG || custom(h) != 1 || payload[0] != INTERNAL_DATA_REGISTER {
        return false;
    }
    match length(h) {
        4 => matches!(payload[1] & 0xFFFF_0000, WAIT_USER_DATA_BEGIN32_TAG | WAIT_USER_DATA_BEGIN64_TAG),
        3 => matches!(payload[1], WAIT_USER_DATA_END_TAG | DRAW_INDIRECT_MULTI_BEGIN_TAG | DRAW_INDIRECT_MULTI_END_TAG),
        _ => false,
    }
}
