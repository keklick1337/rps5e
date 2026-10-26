// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! AMPR command buffer objects and the host-side command record.
//!
//! # Guest object layout (`SceAmprCommandBuffer`)
//!
//! ```text
//! +0x00 u32 type        bit 16 TYPE_GATHER_SCATTER_VALID, bit 17 TYPE_MAP_ACTIVE
//! +0x04 u32 offset      current write offset into the buffer
//! +0x08 u32 num         number of commands written
//! +0x0C u32 size        buffer size in bytes
//! +0x10 u64 buffer      command memory supplied by SetBuffer
//! +0x18 u64 map state   APR command buffers only (zeroed by the APR ctor)
//! +0x20 u64 s/g state
//! ```
//!
//! Every command reserves its measured size in the guest buffer (zero
//! filled; read records carry their opcode in byte 0 and the kernel event
//! record its type, queue, ident and payload) and advances the visible
//! offset and count. The decoded commands are kept on the host, keyed by
//! the command buffer object, together with an alias from the command
//! memory address to its object: AMM submissions name the command memory,
//! APR submissions the object.

use crate::context::CpuContext;
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

pub const OFF_TYPE: u64 = 0x00;
pub const OFF_OFFSET: u64 = 0x04;
pub const OFF_NUM: u64 = 0x08;
pub const OFF_SIZE: u64 = 0x0C;
pub const OFF_DATA: u64 = 0x10;
pub const HEADER_SIZE: usize = 0x18;
pub const OFF_APR_MAP: u64 = 0x18;
pub const OFF_APR_SG: u64 = 0x20;

/// Largest command memory SetBuffer accepts.
pub const BUFFER_SIZE_MAX: u64 = 64 * 1024 * 1024;

pub const TYPE_GATHER_SCATTER_VALID: u32 = 0x0001_0000;
pub const TYPE_MAP_ACTIVE: u32 = 0x0002_0000;

pub const READ_FILE_SIZE: u64 = 0x14;
pub const READ_FILE_SIZE_EXT: u64 = 0x18;
pub const READ_GATHER_SIZE: u64 = 0x08;
pub const READ_GATHER_SIZE_EXT: u64 = 0x0C;
pub const READ_SCATTER_SIZE: u64 = 0x0C;
pub const READ_GATHER_SCATTER_SIZE: u64 = 0x10;
pub const READ_GATHER_SCATTER_SIZE_EXT: u64 = 0x14;
pub const RESET_GATHER_SCATTER_SIZE: u64 = 0x04;
pub const KERNEL_EVENT_SIZE: u64 = 0x20;
/// Wait, write-address, counter and timer commands.
pub const FIXED32_SIZE: u64 = 0x20;
pub const APR_MAP_BEGIN_SIZE: u64 = 0x0C;
pub const APR_MAP_DIRECT_BEGIN_SIZE: u64 = 0x10;
pub const APR_MAP_END_SIZE: u64 = 0x04;
pub const AMM_MAP_SIZE: u64 = 0x20;
pub const AMM_MAP_DIRECT_SIZE: u64 = 0x30;
pub const AMM_UNMAP_SIZE: u64 = 0x20;
/// Remap/MultiMap record: header, address, source, size, protection.
pub const AMM_REMAP_SIZE: u64 = 0x28;
/// ModifyProtect/ModifyMtypeProtect/AllocatePaForPrt record: header,
/// address, size, type, protection, mask.
pub const AMM_PROTECT_SIZE: u64 = 0x28;
/// MapAsPrt record (the unmap layout).
pub const AMM_MAP_AS_PRT_SIZE: u64 = AMM_UNMAP_SIZE;
pub const AMM_PAGE_SIZE: u64 = 0x4000;

pub const OP_READ_FILE: u8 = 0x17;
pub const OP_READ_GATHER: u8 = 0x18;
pub const OP_READ_SCATTER: u8 = 0x19;
pub const OP_READ_GATHER_SCATTER: u8 = 0x1A;
pub const RECORD_KERNEL_EVENT: u32 = 2;

pub const APR_MAX_READ_LENGTH: u64 = 0x0000_0001_0000_0000;
pub const APR_MAX_FILE_OFFSET: u64 = 0x0000_0100_0000_0000;
pub const APR_MAX_APP_ADDRESS: u64 = 0x0000_F000_0000_0000;

// ---------------------------------------------------------------------------
// Sizes and argument validation
// ---------------------------------------------------------------------------

pub fn read_file_size(file_offset: u64) -> u64 {
    if file_offset >> 32 != 0 {
        READ_FILE_SIZE_EXT
    } else {
        READ_FILE_SIZE
    }
}

pub fn read_gather_size(file_offset: u64) -> u64 {
    if file_offset > 0x3FFFF {
        READ_GATHER_SIZE_EXT
    } else {
        READ_GATHER_SIZE
    }
}

pub fn read_gather_scatter_size(file_offset: u64) -> u64 {
    if file_offset >> 32 != 0 {
        READ_GATHER_SCATTER_SIZE_EXT
    } else {
        READ_GATHER_SCATTER_SIZE
    }
}

pub fn align4(v: u64) -> u64 {
    (v + 3) & !3
}

/// Push/set marker: header dword, [color dword], the label with its
/// terminator (nothing for a NULL label), padded to 4.
pub fn marker_size(label_len: Option<usize>, with_color: bool) -> u64 {
    let msg = label_len.map_or(0, |l| l as u64 + 1);
    align4(if with_color { 8 } else { 4 } + msg)
}

pub fn valid_file_offset(off: u64) -> bool {
    off < APR_MAX_FILE_OFFSET
}

pub fn valid_read_size(size: u64) -> bool {
    size != 0 && size <= APR_MAX_READ_LENGTH
}

pub fn valid_read_range(dst: u64, size: u64) -> bool {
    valid_read_size(size) && dst <= APR_MAX_APP_ADDRESS && APR_MAX_APP_ADDRESS - dst >= size
}

/// AMM/APR mapping ranges: non-empty, 16 KiB aligned, not wrapping.
pub fn valid_map_args(va: u64, size: u64) -> bool {
    va != 0 && size != 0 && va % AMM_PAGE_SIZE == 0 && size % AMM_PAGE_SIZE == 0 && va.checked_add(size).is_some()
}

// ---------------------------------------------------------------------------
// Host-side command record
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AmmKind {
    #[default]
    MapAuto,
    MapDirect,
    Unmap,
    /// Move the pages at `source` to `va` with protection `prot`.
    Remap,
    /// Map the pages at `source` at `va` as well.
    MultiMap,
    /// Protection becomes `(old & !mask) | (prot & mask)`.
    ModifyProtect,
    /// As ModifyProtect, and the memory type becomes `mtype`.
    ModifyMtypeProtect,
    /// Drop the backing; the range becomes partially resident (PRT).
    MapAsPrt,
    /// Back the unbacked pages of a PRT range from the automatic pool.
    AllocatePaForPrt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AmmMap {
    pub kind: AmmKind,
    pub va: u64,
    pub dmem_offset: u64,
    pub size: u64,
    pub mtype: i32,
    pub prot: i32,
    pub gpu_mask_id: u8,
    /// Source range of Remap/MultiMap.
    pub source: u64,
    /// Protection bits ModifyProtect changes.
    pub mask: i32,
}

/// A command that does something at execution time. Markers, nops and
/// map brackets occupy stream space but are not recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandData {
    ReadFile { file_id: u32, destination: u64, size: u64, file_offset: u64 },
    KernelEvent { equeue: u64, ident: i32, data: u64 },
    WriteAddress { address: u64, value: u64 },
    AmmMap(AmmMap),
    /// Blocks its submission queue until the address (u64) or shared
    /// counter (u32) compares true against `value`.
    Wait { source: u64, value: u64, compare: u8, is_counter: bool },
    WriteCounter { index: u8, value: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct Command {
    pub record_offset: u64,
    pub data: CommandData,
}

#[derive(Clone, Default, Debug)]
pub struct State {
    pub buffer: u64,
    pub size: u64,
    pub write_offset: u64,
    /// Ordered by record offset (append order).
    pub commands: Vec<Command>,
    pub gs_valid: bool,
    pub gs_file_id: u32,
    pub gs_destination: u64,
    pub gs_file_offset: u64,
}

#[derive(Default)]
struct Registry {
    states: HashMap<u64, State>,
    /// Command memory address -> owning command buffer object.
    aliases: HashMap<u64, u64>,
}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);

struct Guard(MutexGuard<'static, Option<Registry>>);

impl std::ops::Deref for Guard {
    type Target = Registry;
    fn deref(&self) -> &Registry {
        self.0.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for Guard {
    fn deref_mut(&mut self) -> &mut Registry {
        self.0.as_mut().unwrap()
    }
}

fn registry() -> Guard {
    let mut g = REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    g.get_or_insert_with(Registry::default);
    Guard(g)
}

impl Registry {
    fn set_alias(&mut self, cb: u64, buffer: u64) {
        self.aliases.retain(|_, owner| *owner != cb);
        if buffer != 0 && buffer != cb {
            self.aliases.insert(buffer, cb);
        }
    }

    fn erase(&mut self, cb: u64) {
        self.states.remove(&cb);
        self.aliases.retain(|k, v| *k != cb && *v != cb);
    }

    /// A populated state of `cb`; otherwise the object owning `cb` as its
    /// command memory; otherwise `cb`'s own (empty) state.
    fn resolve(&self, cb: u64) -> Option<u64> {
        if self.states.get(&cb).is_some_and(|s| !s.commands.is_empty()) {
            return Some(cb);
        }
        if let Some(&owner) = self.aliases.get(&cb) {
            if self.states.contains_key(&owner) {
                return Some(owner);
            }
        }
        self.states.contains_key(&cb).then_some(cb)
    }
}

/// The buffer/size/offset fields of a guest object.
fn load_from_guest(ctx: &CpuContext, cb: u64) -> Option<State> {
    Some(State {
        buffer: ctx.read_u64(cb + OFF_DATA)?,
        size: ctx.read_u32(cb + OFF_SIZE)? as u64,
        write_offset: ctx.read_u32(cb + OFF_OFFSET)? as u64,
        ..Default::default()
    })
}

/// Zero the object header and forget its host state.
pub fn initialize(ctx: &CpuContext, cb: u64) -> bool {
    if !ctx.write_bytes(cb, &[0u8; HEADER_SIZE]) {
        return false;
    }
    registry().erase(cb);
    true
}

/// Forget a command buffer object and its aliases.
pub fn erase(cb: u64) {
    registry().erase(cb);
}

/// Snapshot of the state a submission or continuation reads (adopting the
/// guest header when the object is unknown).
pub fn try_get_state(ctx: &CpuContext, cb: u64) -> Option<State> {
    {
        let reg = registry();
        if let Some(k) = reg.resolve(cb) {
            return reg.states.get(&k).cloned();
        }
    }
    let st = load_from_guest(ctx, cb)?;
    registry().states.insert(cb, st.clone());
    Some(st)
}

/// Write the size/data fields (the part ClearBuffer resets).
pub fn write_visible_pointers(ctx: &CpuContext, cb: u64, buffer: u64, size: u64) -> bool {
    size <= u32::MAX as u64 && ctx.write_u64(cb + OFF_DATA, buffer) && ctx.write_u32(cb + OFF_SIZE, size as u32)
}

/// Point a command buffer at `buffer`/`size`, clearing its commands and
/// gather/scatter continuation.
pub fn write_pointers(ctx: &CpuContext, cb: u64, buffer: u64, size: u64) -> bool {
    if !write_visible_pointers(ctx, cb, buffer, size) || !ctx.write_u32(cb + OFF_OFFSET, 0) {
        return false;
    }
    let mut reg = registry();
    reg.states.insert(cb, State { buffer, size, ..Default::default() });
    reg.set_alias(cb, buffer);
    true
}

pub fn update_type_flags(ctx: &CpuContext, cb: u64, set: u32, clear: u32) -> bool {
    let Some(t) = ctx.read_u32(cb + OFF_TYPE) else { return false };
    ctx.write_u32(cb + OFF_TYPE, (t | set) & !clear)
}

/// Reserve a zero-filled record of `record_size` bytes (`fill` may set its
/// contents), record `data` at its offset and advance the visible offset
/// and count. `after` runs on the state once the record is committed.
pub fn append_with(
    ctx: &CpuContext,
    cb: u64,
    record_size: u64,
    data: Option<CommandData>,
    fill: impl FnOnce(&mut [u8]),
    after: impl FnOnce(&mut State) -> bool,
) -> bool {
    let mut reg = registry();
    if !reg.states.contains_key(&cb) {
        let Some(st) = load_from_guest(ctx, cb) else { return false };
        reg.states.insert(cb, st);
    }
    let st = reg.states.get_mut(&cb).unwrap();
    if record_size == 0 || st.buffer == 0 || record_size > st.size || st.write_offset > st.size - record_size {
        return false;
    }
    let offset = st.write_offset;
    let mut record = vec![0u8; record_size as usize];
    fill(&mut record);
    if !ctx.write_bytes(st.buffer + offset, &record) {
        return false;
    }
    if let Some(data) = data {
        st.commands.push(Command { record_offset: offset, data });
    }
    st.write_offset += record_size;
    if st.write_offset > u32::MAX as u64 || !ctx.write_u32(cb + OFF_OFFSET, st.write_offset as u32) {
        return false;
    }
    let Some(num) = ctx.read_u32(cb + OFF_NUM) else { return false };
    if !ctx.write_u32(cb + OFF_NUM, num.wrapping_add(1)) {
        return false;
    }
    after(st)
}

/// Append a record that executes `data` (or nothing).
pub fn append(ctx: &CpuContext, cb: u64, record_size: u64, data: Option<CommandData>) -> bool {
    append_with(ctx, cb, record_size, data, |_| {}, |_| true)
}

/// Append a read record: the next gather continues at the end of its
/// destination, the next scatter at the end of its file range.
#[allow(clippy::too_many_arguments)]
pub fn append_read(
    ctx: &CpuContext,
    cb: u64,
    opcode: u8,
    file_id: u32,
    destination: u64,
    size: u64,
    file_offset: u64,
    record_size: u64,
) -> bool {
    let data = CommandData::ReadFile { file_id, destination, size, file_offset };
    let ok = append_with(
        ctx,
        cb,
        record_size,
        Some(data),
        |r| r[0] = opcode,
        |st| {
            st.gs_valid = true;
            st.gs_file_id = file_id;
            match (destination.checked_add(size), file_offset.checked_add(size)) {
                (Some(d), Some(o)) => {
                    st.gs_destination = d;
                    st.gs_file_offset = o;
                    true
                }
                _ => {
                    st.gs_valid = false;
                    false
                }
            }
        },
    );
    ok && update_type_flags(ctx, cb, TYPE_GATHER_SCATTER_VALID, 0)
}

/// Kernel event record `{u32 2, u32 0, u64 equeue, i32 ident, u32 0, u64 data}`.
pub fn append_kernel_event(ctx: &CpuContext, cb: u64, equeue: u64, ident: i32, data: u64) -> bool {
    append_with(
        ctx,
        cb,
        KERNEL_EVENT_SIZE,
        Some(CommandData::KernelEvent { equeue, ident, data }),
        |r| {
            r[0..4].copy_from_slice(&RECORD_KERNEL_EVENT.to_le_bytes());
            r[8..16].copy_from_slice(&equeue.to_le_bytes());
            r[16..20].copy_from_slice(&ident.to_le_bytes());
            r[24..32].copy_from_slice(&data.to_le_bytes());
        },
        |_| true,
    )
}

/// Clear the gather/scatter continuation of a known object.
pub fn reset_gather_scatter(cb: u64) {
    if let Some(st) = registry().states.get_mut(&cb) {
        st.gs_valid = false;
        st.gs_file_id = 0;
        st.gs_destination = 0;
        st.gs_file_offset = 0;
    }
}
