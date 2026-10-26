// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceAgcDriver: command-buffer submission, GPU equeue events, resource
//! and owner registration bookkeeping, workload streams and capture queries.
//! Also `sceAgcSuspendPoint` (the frame boundary).

use super::builders::{WORKLOAD_STREAM_MAX, WORKLOAD_STREAM_MIN};
use super::pm4::queue;
use super::{
    arg, mem, stack_arg, Export, ERR_DRIVER_INVALID_ARGUMENT, ERR_DRIVER_INVALID_VALUE, ERR_INVALID_ARGUMENT, ERR_MEMORY_FAULT,
    ERR_BAD_FILE, ERR_NOT_FOUND, LIB_AGC, LIB_AGC_DRIVER, OK,
};
use crate::context::CpuContext;
use crate::registry::HleResult;
use crate::sync::equeue;
use std::collections::BTreeMap;
use std::sync::Mutex;

const MAX_NAME_LENGTH: u32 = 256;
const BYTES_PER_RESOURCE: u64 = 0x118;
const BYTES_PER_OWNER: u64 = 0x1E0;
const DEFAULT_OWNER: u32 = 1;
const WORKLOAD_STREAM_RECORD: usize = 32;

#[derive(Clone)]
struct Resource {
    owner: u32,
    addr: u64,
    size: u64,
    /// Guest copy of the name (NUL-terminated), 0 when unnamed.
    name: u64,
    ty: u32,
    user_data: u64,
}

struct Registry {
    initialized: bool,
    max_owners: u32,
    default_owner: u32,
    next_owner: u32,
    next_resource: u32,
    owners: BTreeMap<u32, u64>,
    resources: BTreeMap<u32, Resource>,
    workload_streams: BTreeMap<u32, [u8; WORKLOAD_STREAM_RECORD]>,
}

static REGISTRY: Mutex<Registry> = Mutex::new(Registry {
    initialized: false,
    max_owners: 0,
    default_owner: DEFAULT_OWNER,
    next_owner: 1,
    next_resource: 1,
    owners: BTreeMap::new(),
    resources: BTreeMap::new(),
    workload_streams: BTreeMap::new(),
});

fn reg() -> std::sync::MutexGuard<'static, Registry> {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner())
}

/// Copy a guest name into a guest block the driver owns.
fn copy_name(ctx: &CpuContext, name: u64) -> Option<u64> {
    if name == 0 {
        return Some(0);
    }
    let mut bytes = mem::read_cstr(name, MAX_NAME_LENGTH as usize)?;
    bytes.push(0);
    let a = crate::pthread::alloc_guest_block(ctx, bytes.len() as u64)?;
    mem::write(a, &bytes).then_some(a)
}

// ---------------------------------------------------------------------------
// Submission
// ---------------------------------------------------------------------------

fn gpu() -> &'static queue::GpuQueue {
    let q = queue::global();
    q.ensure_worker();
    q
}

/// `sceAgcDriver[Agr]SubmitDcb(const {u64 addr; u32 dwords}*)`.
fn submit_dcb(ctx: &mut CpuContext) -> HleResult {
    crate::pause::wait_if_paused();
    let p = arg(ctx, 0);
    let (Some(addr), Some(dwords)) = ((p != 0).then(|| mem::r64(p)).flatten(), mem::r32(p.wrapping_add(8))) else {
        return ERR_INVALID_ARGUMENT;
    };
    gpu().submit_graphics(addr, dwords);
    OK
}

/// `sceAgcDriver[Agr]SubmitMultiDcbs(const u64* addrs, const u32* dwords, count)`.
fn submit_multi_dcbs(ctx: &mut CpuContext) -> HleResult {
    crate::pause::wait_if_paused();
    let (addrs, sizes, n) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2) as u32);
    if n == 0 {
        return OK;
    }
    if addrs == 0 || sizes == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let q = gpu();
    for i in 0..n as u64 {
        let (Some(a), Some(d)) = (mem::r64(addrs + i * 8), mem::r32(sizes + i * 4)) else { return ERR_MEMORY_FAULT };
        if a != 0 {
            q.submit_graphics(a, d);
        }
    }
    OK
}

/// `sceAgcDriverSubmitAcb(queue, const {u64 addr; u32 dwords}*)`.
fn submit_acb(ctx: &mut CpuContext) -> HleResult {
    let (owner, p) = (arg(ctx, 0) as u32, arg(ctx, 1));
    if p == 0 {
        return OK;
    }
    let (Some(addr), Some(dwords)) = (mem::r64(p), mem::r32(p.wrapping_add(8))) else {
        return ERR_MEMORY_FAULT;
    };
    submit_compute(owner, addr, dwords);
    OK
}

/// Queue a compute buffer (empty buffers are ignored).
fn submit_compute(owner: u32, addr: u64, dwords: u32) {
    if addr != 0 && dwords != 0 && !gpu().submit_compute(owner, addr, dwords) {
        log::warn!("[agc] compute submission to unsupported queue 0x{owner:x} dropped");
    }
}

/// `sceAgcDriverSubmitCommandBuffer(queueContext, const {u64 addr; u32
/// dwords}*)`: the queue id at +4 of the context selects a compute queue
/// (0x20..0x57) or the graphics queue.
fn submit_command_buffer(ctx: &mut CpuContext) -> HleResult {
    crate::pause::wait_if_paused();
    let (qctx, p) = (arg(ctx, 0), arg(ctx, 1));
    if qctx == 0 || p == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let (Some(queue), Some(addr), Some(dwords)) = (mem::r32(qctx + 4), mem::r64(p), mem::r32(p + 8)) else {
        return ERR_MEMORY_FAULT;
    };
    route_command_buffer(queue, addr, dwords);
    OK
}

/// `sceAgcDriverSubmitMultiCommandBuffers(queueContext, const u64* addrs,
/// const u32* dwords, count)`.
fn submit_multi_command_buffers(ctx: &mut CpuContext) -> HleResult {
    crate::pause::wait_if_paused();
    let (qctx, addrs, sizes, n) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2), arg(ctx, 3) as u32);
    if n == 0 {
        return OK;
    }
    if qctx == 0 || addrs == 0 || sizes == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(queue) = mem::r32(qctx + 4) else { return ERR_MEMORY_FAULT };
    for i in 0..n as u64 {
        let (Some(a), Some(d)) = (mem::r64(addrs + i * 8), mem::r32(sizes + i * 4)) else { return ERR_MEMORY_FAULT };
        route_command_buffer(queue, a, d);
    }
    OK
}

fn route_command_buffer(queue: u32, addr: u64, dwords: u32) {
    if addr == 0 || dwords == 0 {
        return;
    }
    if (0x20..0x58).contains(&queue) {
        submit_compute(queue, addr, dwords);
    } else {
        gpu().submit_graphics(addr, dwords);
    }
}

/// `sceAgcDriverSubmitMultiAcbs(queue, const u64* acbs, const u32* dwords, count)`.
fn submit_multi_acbs(ctx: &mut CpuContext) -> HleResult {
    let (owner, addrs, sizes, n) = (arg(ctx, 0) as u32, arg(ctx, 1), arg(ctx, 2), arg(ctx, 3) as u32);
    if n == 0 {
        return OK;
    }
    if addrs == 0 || sizes == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    for i in 0..n as u64 {
        let (Some(a), Some(d)) = (mem::r64(addrs + i * 8), mem::r32(sizes + i * 4)) else { return ERR_MEMORY_FAULT };
        submit_compute(owner, a, d);
    }
    OK
}

/// `sceAgcSuspendPoint()`: frame boundary.
fn suspend_point(_ctx: &mut CpuContext) -> HleResult {
    crate::pause::wait_if_paused();
    gpu().suspend_point();
    OK
}

/// `sceAgcSetSubmitMode(mode)`: mode 0, the only one known, is how
/// submissions already run; any other mode is not implemented.
fn set_submit_mode(ctx: &mut CpuContext) -> HleResult {
    let mode = arg(ctx, 0) as u32;
    if mode == 0 {
        return OK;
    }
    super::log_once(&format!("agc.submit_mode.{mode}"), || log::error!("[agc] sceAgcSetSubmitMode({mode:#x}): mode not implemented"));
    ERR_NOT_FOUND
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// `sceAgcDriverAddEqEvent(eq, id, udata)`: graphics filter (-14), reset
/// on delivery.
fn add_eq_event(ctx: &mut CpuContext) -> HleResult {
    let (eq, id, udata) = (arg(ctx, 0), arg(ctx, 1) as i32 as u64, arg(ctx, 2));
    if !equeue::is_valid_equeue(eq) {
        return ERR_BAD_FILE;
    }
    if equeue::register_event(eq, id, equeue::FILTER_GRAPHICS, udata, equeue::EV_CLEAR, 0) {
        OK
    } else {
        ERR_NOT_FOUND
    }
}

fn delete_eq_event(ctx: &mut CpuContext) -> HleResult {
    let (eq, id) = (arg(ctx, 0), arg(ctx, 1) as i32 as u64);
    if !equeue::is_valid_equeue(eq) {
        return ERR_BAD_FILE;
    }
    if equeue::delete_registered_event(eq, id, equeue::FILTER_GRAPHICS) {
        OK
    } else {
        ERR_NOT_FOUND
    }
}

/// Graphics events: the ident; others: the data.
fn get_eq_event_type(ctx: &mut CpuContext) -> HleResult {
    let ev = arg(ctx, 0);
    if ev == 0 {
        return 0;
    }
    let (Some(ident), Some(filter), Some(data)) = (mem::r64(ev), mem::r16(ev + 8), mem::r64(ev + 0x10)) else { return 0 };
    (if filter as i16 == equeue::FILTER_GRAPHICS { ident } else { data }) as i32 as HleResult
}

/// Graphics events: the data (the packet's context id); others: the ident.
fn get_eq_context_id(ctx: &mut CpuContext) -> HleResult {
    let ev = arg(ctx, 0);
    if ev == 0 {
        return 0;
    }
    let (Some(ident), Some(filter), Some(data)) = (mem::r64(ev), mem::r16(ev + 8), mem::r64(ev + 0x10)) else { return 0 };
    (if filter as i16 == equeue::FILTER_GRAPHICS { data } else { ident }) as u32 as HleResult
}

// ---------------------------------------------------------------------------
// Owners and resources
// ---------------------------------------------------------------------------

fn get_resource_registration_max_name_length(ctx: &mut CpuContext) -> HleResult {
    let out = arg(ctx, 0);
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if mem::w32(out, MAX_NAME_LENGTH) {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

fn get_default_owner(ctx: &mut CpuContext) -> HleResult {
    let out = arg(ctx, 0);
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let owner = reg().default_owner;
    if mem::w32(out, owner) {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

fn register_default_owner(ctx: &mut CpuContext) -> HleResult {
    reg().default_owner = arg(ctx, 0) as u32;
    OK
}

/// `(u64* outSize, resources, owners)`.
fn query_user_memory_requirements(ctx: &mut CpuContext) -> HleResult {
    let (out, resources, owners) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2));
    if out == 0 || resources == 0 || owners == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(size) = resources.checked_mul(BYTES_PER_RESOURCE).and_then(|r| owners.checked_mul(BYTES_PER_OWNER).and_then(|o| r.checked_add(o))) else {
        return ERR_INVALID_ARGUMENT;
    };
    if mem::w64(out, size) {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

/// `(memory, size, maxOwners)`: resets the registry.
fn init_resource_registration(ctx: &mut CpuContext) -> HleResult {
    let (memory, size, owners) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2));
    if memory == 0 || size == 0 || owners == 0 || owners > u32::MAX as u64 {
        return ERR_INVALID_ARGUMENT;
    }
    let mut r = reg();
    r.initialized = true;
    r.max_owners = owners as u32;
    r.owners.clear();
    r.resources.clear();
    r.default_owner = DEFAULT_OWNER;
    r.next_owner = 1;
    r.next_resource = 1;
    OK
}

/// `(u32* outOwner, const char* name)`.
fn register_owner(ctx: &mut CpuContext) -> HleResult {
    let (out, name) = (arg(ctx, 0), arg(ctx, 1));
    if out == 0 || name == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(copy) = copy_name(ctx, name) else { return ERR_INVALID_ARGUMENT };
    let mut r = reg();
    if r.initialized && r.max_owners != 0 && r.owners.len() as u32 >= r.max_owners {
        return ERR_INVALID_ARGUMENT;
    }
    let mut owner = r.next_owner;
    while owner == r.default_owner || r.owners.contains_key(&owner) {
        owner = owner.wrapping_add(1);
        if owner == 0 {
            return ERR_INVALID_ARGUMENT;
        }
    }
    r.next_owner = owner + 1;
    r.owners.insert(owner, copy);
    drop(r);
    if !mem::w32(out, owner) {
        reg().owners.remove(&owner);
        return ERR_MEMORY_FAULT;
    }
    OK
}

/// `(const char** outName, owner)`.
fn get_owner_name(ctx: &mut CpuContext) -> HleResult {
    let (out, owner) = (arg(ctx, 0), arg(ctx, 1) as u32);
    let name = reg().owners.get(&owner).copied();
    match name {
        _ if out == 0 => ERR_INVALID_ARGUMENT,
        Some(n) => {
            if mem::w64(out, n) {
                OK
            } else {
                ERR_MEMORY_FAULT
            }
        }
        None => ERR_INVALID_ARGUMENT,
    }
}

fn looks_like_string(addr: u64) -> bool {
    addr > 0xFFFF && mem::read_cstr(addr, MAX_NAME_LENGTH as usize).is_some()
}

/// `(u32* outHandle, owner, memory, size, name, type, userData)`. An older
/// layout `(outHandle, owner, name, _, type, flags)` is recognised by the
/// name position.
fn register_resource_common(ctx: &CpuContext, gds: bool) -> HleResult {
    let (out, owner) = (arg(ctx, 0), arg(ctx, 1) as u32);
    let (addr, size, name, ty, user) = if !gds && !looks_like_string(arg(ctx, 4)) && looks_like_string(arg(ctx, 2)) {
        (0, 0, arg(ctx, 2), arg(ctx, 4) as u32, 0)
    } else {
        (arg(ctx, 2), arg(ctx, 3), arg(ctx, 4), arg(ctx, 5) as u32, stack_arg(ctx, 1).unwrap_or(0))
    };
    let Some(name) = copy_name(ctx, name) else { return ERR_INVALID_ARGUMENT };
    let mut r = reg();
    let mut handle = r.next_resource.max(1);
    while r.resources.contains_key(&handle) {
        handle = handle.wrapping_add(1).max(1);
    }
    r.next_resource = handle.wrapping_add(1);
    r.resources.insert(handle, Resource { owner, addr, size, name, ty, user_data: user });
    drop(r);
    if out != 0 && !mem::w32(out, handle) {
        reg().resources.remove(&handle);
        return ERR_MEMORY_FAULT;
    }
    OK
}

fn register_resource(ctx: &mut CpuContext) -> HleResult {
    register_resource_common(ctx, false)
}

fn register_gds_resource(ctx: &mut CpuContext) -> HleResult {
    register_resource_common(ctx, true)
}

fn unregister_resource(ctx: &mut CpuContext) -> HleResult {
    if reg().resources.remove(&(arg(ctx, 0) as u32)).is_some() {
        OK
    } else {
        ERR_INVALID_ARGUMENT
    }
}

fn unregister_all_resources_for_owner(ctx: &mut CpuContext) -> HleResult {
    let owner = arg(ctx, 0) as u32;
    reg().resources.retain(|_, r| r.owner != owner);
    OK
}

fn unregister_owner_and_resources(ctx: &mut CpuContext) -> HleResult {
    let owner = arg(ctx, 0) as u32;
    let mut r = reg();
    if r.owners.remove(&owner).is_none() {
        return ERR_INVALID_ARGUMENT;
    }
    r.resources.retain(|_, x| x.owner != owner);
    OK
}

fn with_resource(handle: u32, f: impl FnOnce(&mut Resource) -> HleResult) -> HleResult {
    match reg().resources.get_mut(&handle) {
        Some(r) => f(r),
        None => ERR_INVALID_ARGUMENT,
    }
}

/// `(void** outAddr, u64* outSize, handle)`.
fn get_resource_base_and_size(ctx: &mut CpuContext) -> HleResult {
    let (oa, os, h) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2) as u32);
    with_resource(h, |r| {
        if (oa != 0 && !mem::w64(oa, r.addr)) || (os != 0 && !mem::w64(os, r.size)) {
            ERR_MEMORY_FAULT
        } else {
            OK
        }
    })
}

/// `(const char** outName, handle)`.
fn get_resource_name(ctx: &mut CpuContext) -> HleResult {
    let (out, h) = (arg(ctx, 0), arg(ctx, 1) as u32);
    with_resource(h, |r| if out != 0 && mem::w64(out, r.name) { OK } else { ERR_MEMORY_FAULT })
}

/// `(u32* outType, handle)`.
fn get_resource_type(ctx: &mut CpuContext) -> HleResult {
    let (out, h) = (arg(ctx, 0), arg(ctx, 1) as u32);
    with_resource(h, |r| if out != 0 && mem::w32(out, r.ty) { OK } else { ERR_MEMORY_FAULT })
}

/// `(u64* outUserData, handle)`.
fn get_resource_user_data(ctx: &mut CpuContext) -> HleResult {
    let (out, h) = (arg(ctx, 0), arg(ctx, 1) as u32);
    with_resource(h, |r| if out != 0 && mem::w64(out, r.user_data) { OK } else { ERR_MEMORY_FAULT })
}

/// `(handle, userData)`.
fn set_resource_user_data(ctx: &mut CpuContext) -> HleResult {
    let (h, v) = (arg(ctx, 0) as u32, arg(ctx, 1));
    with_resource(h, |r| {
        r.user_data = v;
        OK
    })
}

/// `(guid16* out, handle)`: no shader GUIDs are tracked; reports zero.
fn get_resource_shader_guid(ctx: &mut CpuContext) -> HleResult {
    let (out, h) = (arg(ctx, 0), arg(ctx, 1) as u32);
    with_resource(h, |_| if out != 0 && mem::write(out, &[0u8; 16]) { OK } else { ERR_MEMORY_FAULT })
}

/// SCE_AGC_ERROR_RESOURCE_REGISTRATION_UNAVAILABLE.
const ERR_RESOURCE_REGISTRATION_UNAVAILABLE: HleResult = 0x8A6C_9018u32 as i32 as i64;

/// `sceAgcDriverFindResourcesPublic(callback, userData)`: the address
/// lookup of the GPU crash tools. The callback's signature is not known,
/// so the lookup is reported unavailable rather than as an empty success.
fn find_resources_public(_ctx: &mut CpuContext) -> HleResult {
    ERR_RESOURCE_REGISTRATION_UNAVAILABLE
}

/// `sceAgcDriverRegisterMultipleResources(u32* outHandle, owner,
/// descriptors, count)`: the 40-byte descriptor layout is not known;
/// registration is reported unavailable (titles accept that code) and the
/// caller's memory is left alone.
fn register_multiple_resources(_ctx: &mut CpuContext) -> HleResult {
    ERR_RESOURCE_REGISTRATION_UNAVAILABLE
}

// ---------------------------------------------------------------------------
// Workload streams, capture
// ---------------------------------------------------------------------------

/// `(streamId, const record[32]*)`.
fn register_workload_stream(ctx: &mut CpuContext) -> HleResult {
    let (id, stream) = (arg(ctx, 0) as u32, arg(ctx, 1));
    if !(WORKLOAD_STREAM_MIN..=WORKLOAD_STREAM_MAX).contains(&id) {
        return ERR_DRIVER_INVALID_VALUE;
    }
    if stream == 0 {
        return ERR_DRIVER_INVALID_ARGUMENT;
    }
    let mut rec = [0u8; WORKLOAD_STREAM_RECORD];
    if !mem::read(stream, &mut rec) {
        return ERR_DRIVER_INVALID_ARGUMENT;
    }
    let mut r = reg();
    if r.workload_streams.contains_key(&id) {
        return ERR_DRIVER_INVALID_VALUE;
    }
    r.workload_streams.insert(id, rec);
    WORKLOAD_STREAM_MASK.fetch_or(1 << id, std::sync::atomic::Ordering::Relaxed);
    OK
}

/// Registered workload streams (bit per id) for the workload builders.
static WORKLOAD_STREAM_MASK: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub(crate) fn workload_stream_registered(id: u32) -> bool {
    id < 32 && WORKLOAD_STREAM_MASK.load(std::sync::atomic::Ordering::Relaxed) & (1 << id) != 0
}

fn unregister_workload_stream(ctx: &mut CpuContext) -> HleResult {
    let id = arg(ctx, 0) as u32;
    if reg().workload_streams.remove(&id).is_some() {
        WORKLOAD_STREAM_MASK.fetch_and(!(1 << id), std::sync::atomic::Ordering::Relaxed);
        OK
    } else {
        ERR_DRIVER_INVALID_VALUE
    }
}

fn returns_false(_ctx: &mut CpuContext) -> HleResult {
    0
}

fn returns_ok(_ctx: &mut CpuContext) -> HleResult {
    OK
}

/// `sceAgcDebugRaiseException(id)`.
fn debug_raise_exception(ctx: &mut CpuContext) -> HleResult {
    log::info!("[agc] DebugRaiseException 0x{:x}", arg(ctx, 0) as u32);
    OK
}

/// Tessellation factor ring and off-chip HS parameters (kept for the
/// tessellation stages). Until a title sets a ring, the console's default
/// is reported: 128 KiB at 0xFF0000000.
static TESSELLATION: Mutex<[u64; 5]> = Mutex::new([DEFAULT_TF_RING_BASE, DEFAULT_TF_RING_SIZE, 0, 0, 0]);
const DEFAULT_TF_RING_BASE: u64 = 0xF_F000_0000;
const DEFAULT_TF_RING_SIZE: u64 = 0x2_0000;

/// `sceAgcDriverSetTFRing(base, size)`: an empty ring is refused.
fn set_tf_ring(ctx: &mut CpuContext) -> HleResult {
    let (base, size) = (arg(ctx, 0), arg(ctx, 1) as u32);
    if base == 0 || size == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let mut t = TESSELLATION.lock().unwrap_or_else(|e| e.into_inner());
    t[0] = base;
    t[1] = size as u64;
    OK
}

/// `sceAgcDriverGetTFRing(u64* base, u32* size)`.
fn get_tf_ring(ctx: &mut CpuContext) -> HleResult {
    let (out_base, out_size) = (arg(ctx, 0), arg(ctx, 1));
    if out_base == 0 || out_size == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let (base, size) = {
        let t = TESSELLATION.lock().unwrap_or_else(|e| e.into_inner());
        (t[0], t[1] as u32)
    };
    if mem::w64(out_base, base) && mem::w32(out_size, size) {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

/// The rendering-wait packet: the NOP WaitFlipDone `[handle, index, mode]`
/// the command processor holds until the display buffer is no longer
/// waiting to be shown.
const RENDERING_WAIT_DWORDS: u32 = 4;

/// `sceAgcDriverGetWaitRenderingPacketSizeInDwords()`.
fn get_wait_rendering_packet_size(_ctx: &mut CpuContext) -> HleResult {
    RENDERING_WAIT_DWORDS as HleResult
}

/// The flip packet `sceAgcDcbSetFlip` writes: NOP FLIP `[handle, index,
/// mode, arg lo, arg hi]`.
const FLIP_PACKET_DWORDS: u32 = 6;

/// `sceAgcDriverGetSetFlipPacketSizeInDwords()`.
fn get_set_flip_packet_size(_ctx: &mut CpuContext) -> HleResult {
    FLIP_PACKET_DWORDS as HleResult
}

/// `sceAgcDriverWaitUntilSafeForRendering(u32** cursor, capacity, mode,
/// handle, index)`: writes the rendering-wait packet at `*cursor` and moves
/// the cursor past it. Only mode 0 exists.
fn wait_until_safe_for_rendering(ctx: &mut CpuContext) -> HleResult {
    let (cursor, capacity, mode, handle, index) = (arg(ctx, 0), arg(ctx, 1) as u32, arg(ctx, 2) as u32, arg(ctx, 3) as u32, arg(ctx, 4) as i32);
    if cursor == 0 || capacity < RENDERING_WAIT_DWORDS || mode != 0 || index < 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(at) = mem::r64(cursor).filter(|a| *a != 0 && a & 3 == 0) else { return ERR_INVALID_ARGUMENT };
    let packet = [super::pm4(RENDERING_WAIT_DWORDS, super::pm4::op::NOP, super::pm4::nop::WAIT_FLIP_DONE), handle, index as u32, mode];
    if mem::write_dwords(at, &packet) && mem::w64(cursor, at + RENDERING_WAIT_DWORDS as u64 * 4) {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

fn set_hs_offchip_param(ctx: &mut CpuContext) -> HleResult {
    let mut t = TESSELLATION.lock().unwrap_or_else(|e| e.into_inner());
    t[2] = arg(ctx, 0);
    t[3] = arg(ctx, 1);
    t[4] = arg(ctx, 2);
    OK
}

/// The retail library's answer to the submit-validation calls (validation
/// is not available): `mov eax, 0x8a6c1000; ret`, arguments untouched.
const ERR_VALIDATION_UNAVAILABLE: HleResult = 0x8A6C_1000u32 as i32 as i64;

fn validation_unavailable(_: &mut CpuContext) -> HleResult {
    ERR_VALIDATION_UNAVAILABLE
}

/// `sceAgcDriverGetShaderDebuggingStatus()`: 1, as the retail library.
fn shader_debugging_status(_: &mut CpuContext) -> HleResult {
    1
}

/// The retail library's answer to the debug and capture configuration
/// calls, as for submit validation.
fn debug_unavailable(_: &mut CpuContext) -> HleResult {
    ERR_VALIDATION_UNAVAILABLE
}

/// A driver feature (GPU printf, register writes, reference clocks, tool
/// memory) the retail library does not provide to a title.
const ERR_DRIVER_FEATURE_UNAVAILABLE: HleResult = 0x8A6D_0001u32 as i32 as i64;
/// A null output pointer.
const ERR_DRIVER_NULL_OUTPUT: HleResult = 0x8A6D_0003u32 as i32 as i64;
/// `sceAgdResetGpu`: a title cannot reset the GPU.
const ERR_GPU_RESET_UNAVAILABLE: HleResult = 0x8A6C_0045u32 as i32 as i64;
/// The `{u64 address; u32 size}` range the GPU-reset tool memory calls
/// clear.
const TOOL_MEMORY_RANGE_SIZE: u64 = 12;

fn feature_unavailable(_: &mut CpuContext) -> HleResult {
    ERR_DRIVER_FEATURE_UNAVAILABLE
}

fn gpu_reset_unavailable(_: &mut CpuContext) -> HleResult {
    ERR_GPU_RESET_UNAVAILABLE
}

/// `sceAgcDriverGetCoredumpMode(u32* mode)`: no core dump mode (0).
fn get_coredump_mode(ctx: &mut CpuContext) -> HleResult {
    let out = arg(ctx, 0);
    if out == 0 {
        return ERR_DRIVER_NULL_OUTPUT;
    }
    if mem::w32(out, 0) {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

/// `sceAgcDriverAllocateToolMemoryForGpuReset(a, b, c, range*)`: no tool
/// memory; the range is cleared.
fn allocate_tool_memory_for_gpu_reset(ctx: &mut CpuContext) -> HleResult {
    clear_tool_memory_range(arg(ctx, 3))
}

/// `sceAgcDriverGetAllocatedToolMemoryForGpuReset(a, range*)`.
fn get_allocated_tool_memory_for_gpu_reset(ctx: &mut CpuContext) -> HleResult {
    clear_tool_memory_range(arg(ctx, 1))
}

fn clear_tool_memory_range(range: u64) -> HleResult {
    if range != 0 && !mem::fill(range, TOOL_MEMORY_RANGE_SIZE, 0) {
        return ERR_MEMORY_FAULT;
    }
    ERR_DRIVER_FEATURE_UNAVAILABLE
}

/// Dwords of a `bytes`-byte payload.
fn payload_dwords(bytes: u32) -> u32 {
    ((bytes as u64 + 3) >> 2) as u32
}

/// The workload data packet that carries `bytes` of user data: 3 dwords
/// when empty, 4 for a one-dword payload, else the payload plus 7.
fn workload_data_packet_dwords(bytes: u32) -> u32 {
    match payload_dwords(bytes) {
        0 => 3,
        1 => 4,
        n => n + 7,
    }
}

/// `sceAgcDriverGetSetWorkloadsActivePacketSize()` (dwords).
fn get_set_workloads_active_packet_size(_: &mut CpuContext) -> HleResult {
    (workload_data_packet_dwords(8) + 9) as HleResult
}

/// `sceAgcDriverGetSetWorkloadCompletePacketSize()` (dwords).
fn get_set_workload_complete_packet_size(_: &mut CpuContext) -> HleResult {
    (workload_data_packet_dwords(0) + 9) as HleResult
}

/// `sceAgcDriverGetSetWorkloadStreamInactivePacketSize()` (dwords).
fn get_set_workload_stream_inactive_packet_size(_: &mut CpuContext) -> HleResult {
    workload_data_packet_dwords(8) as HleResult
}

/// `sceAgcDriverNopDataGetPacketSize(bytes)` (dwords): a NOP header and a
/// tag dword around the payload.
fn nop_data_get_packet_size(ctx: &mut CpuContext) -> HleResult {
    (payload_dwords(arg(ctx, 0) as u32) + 2) as HleResult
}

/// `sceAgcDriverUserDataGetPacketSize(bytes)` (dwords).
fn user_data_get_packet_size(ctx: &mut CpuContext) -> HleResult {
    workload_data_packet_dwords(arg(ctx, 0) as u32) as HleResult
}

/// `sceAgcDriverRegisterOwnerForSystem` / `InitResourceRegistrationInterface`:
/// signatures not known; reported unavailable as the public lookup.
fn registration_interface_unavailable(_: &mut CpuContext) -> HleResult {
    ERR_RESOURCE_REGISTRATION_UNAVAILABLE
}

pub(crate) const EXPORTS: &[Export] = &[
    ("rJUyMrDdxJg", "sceAgcDriverGetShaderDebuggingStatus", LIB_AGC_DRIVER, shader_debugging_status),
    ("8F5j7U-Bq6s", "sceAgcDriverSetValidationErrorOutputFrequency", LIB_AGC_DRIVER, validation_unavailable),
    ("+iAOE3jCnkk", "sceAgcDriverSetSubmitValidationMode", LIB_AGC_DRIVER, validation_unavailable),
    ("ICaGtkEIXTk", "sceAgcDriverGetSubmitValidationConfig", LIB_AGC_DRIVER, validation_unavailable),
    ("WHIOMbb+iIU", "sceAgcDriverGetSubmitValidationMode", LIB_AGC_DRIVER, validation_unavailable),
    ("chJWZcNSzjk", "sceAgcDriverSetSubmitValidationConfig", LIB_AGC_DRIVER, validation_unavailable),
    ("UglJIZjGssM", "sceAgcDriverSubmitDcb", LIB_AGC_DRIVER, submit_dcb),
    ("AhGvpITrf4M", "sceAgcDriverAgrSubmitDcb", LIB_AGC_DRIVER, submit_dcb),
    ("6UzEidRZwkg", "sceAgcDriverSubmitMultiDcbs", LIB_AGC_DRIVER, submit_multi_dcbs),
    ("+T8Xo6LtFJI", "sceAgcDriverAgrSubmitMultiDcbs", LIB_AGC_DRIVER, submit_multi_dcbs),
    ("gSRnr79F8tQ", "sceAgcDriverSubmitAcb", LIB_AGC_DRIVER, submit_acb),
    ("HF3YllT3mXU", "sceAgcDriverSubmitMultiAcbs", LIB_AGC_DRIVER, submit_multi_acbs),
    ("h9z6+0hEydk", "sceAgcSuspendPoint", LIB_AGC, suspend_point),
    ("-DtvmQ-tgEA", "sceAgcSetSubmitMode", LIB_AGC, set_submit_mode),
    ("w2rJhmD+dsE", "sceAgcDriverAddEqEvent", LIB_AGC_DRIVER, add_eq_event),
    ("DL2RXaXOy88", "sceAgcDriverDeleteEqEvent", LIB_AGC_DRIVER, delete_eq_event),
    ("5CdQTZIQPxM", "sceAgcDriverGetEqEventType", LIB_AGC_DRIVER, get_eq_event_type),
    ("Zw7uUVPulbw", "sceAgcDriverGetEqContextId", LIB_AGC_DRIVER, get_eq_context_id),
    ("uJziRsODk1c", "sceAgcDriverGetResourceRegistrationMaxNameLength", LIB_AGC, get_resource_registration_max_name_length),
    ("F0ZXt5q0ZTA", "sceAgcDriverGetDefaultOwner", LIB_AGC, get_default_owner),
    ("U9ueyEhSkF4", "sceAgcDriverRegisterDefaultOwner", LIB_AGC, register_default_owner),
    ("AOLcoIkQDgM", "sceAgcDriverQueryResourceRegistrationUserMemoryRequirements", LIB_AGC_DRIVER, query_user_memory_requirements),
    ("F0Y42t-3e18", "sceAgcDriverInitResourceRegistration", LIB_AGC_DRIVER, init_resource_registration),
    ("X-Nm5KLREeg", "sceAgcDriverRegisterOwner", LIB_AGC_DRIVER, register_owner),
    ("LepGrgk77sM", "sceAgcDriverGetOwnerName", LIB_AGC_DRIVER, get_owner_name),
    ("W5z4eZrjEas", "sceAgcDriverRegisterResource", LIB_AGC_DRIVER, register_resource),
    ("emP3ckeS2uo", "sceAgcDriverRegisterGdsResource", LIB_AGC_DRIVER, register_gds_resource),
    ("pWLG7WOpVcw", "sceAgcDriverUnregisterResource", LIB_AGC_DRIVER, unregister_resource),
    ("SCoAN5fYlUM", "sceAgcDriverUnregisterAllResourcesForOwner", LIB_AGC_DRIVER, unregister_all_resources_for_owner),
    ("ZLJk9r2+2Aw", "sceAgcDriverUnregisterOwnerAndResources", LIB_AGC_DRIVER, unregister_owner_and_resources),
    ("NghWEUXp1qM", "sceAgcDriverGetResourceBaseAddressAndSizeInBytes", LIB_AGC_DRIVER, get_resource_base_and_size),
    ("M9yBzRKkjPc", "sceAgcDriverGetResourceName", LIB_AGC_DRIVER, get_resource_name),
    ("mXn+K9E-wOA", "sceAgcDriverGetResourceShaderGuid", LIB_AGC_DRIVER, get_resource_shader_guid),
    ("rI9lNAXPMIw", "sceAgcDriverGetResourceType", LIB_AGC_DRIVER, get_resource_type),
    ("ls4jfY576lw", "sceAgcDriverGetResourceUserData", LIB_AGC_DRIVER, get_resource_user_data),
    ("VOMSpd9+vxU", "sceAgcDriverSetResourceUserData", LIB_AGC_DRIVER, set_resource_user_data),
    ("5l3IfCFJxBs", "sceAgcDriverFindResourcesPublic", LIB_AGC_DRIVER, find_resources_public),
    ("3AyTaWcF-H8", "sceAgcDriverRegisterWorkloadStream", LIB_AGC_DRIVER, register_workload_stream),
    ("n5ElQVYsU1A", "sceAgcDriverUnregisterWorkloadStream", LIB_AGC_DRIVER, unregister_workload_stream),
    ("Ddwk4gLT5j0", "sceAgcDriverIsCaptureInProgress", LIB_AGC_DRIVER, returns_false),
    ("+TN0oRTBxJQ", "sceAgcDriverIsTraceInProgress", LIB_AGC_DRIVER, returns_false),
    ("qspAL8bgcBY", "sceAgcDriverIsSubmitValidationEnabled", LIB_AGC_DRIVER, returns_false),
    ("SAfhzJPcjuk", "sceAgcDriverRequestCaptureStart", LIB_AGC_DRIVER, returns_ok),
    ("FOwvmNlFLjM", "sceAgcDriverRequestCaptureStop", LIB_AGC_DRIVER, returns_ok),
    ("Xq5WmbwPTnQ", "sceAgcDriverTriggerCapture", LIB_AGC_DRIVER, returns_ok),
    ("XlNp7jzGiPo", "sceAgcDriverSetTFRing", LIB_AGC_DRIVER, set_tf_ring),
    ("e-YMQ+2tj9M", "sceAgcDriverGetTFRing", LIB_AGC_DRIVER, get_tf_ring),
    ("0MtUJ3BpGhE", "sceAgcDriverGetWaitRenderingPacketSizeInDwords", LIB_AGC_DRIVER, get_wait_rendering_packet_size),
    ("2PrsbRYyZi4", "sceAgcDriverGetSetFlipPacketSizeInDwords", LIB_AGC_DRIVER, get_set_flip_packet_size),
    ("u8BkdHb1+Po", "sceAgcDriverWaitUntilSafeForRendering", LIB_AGC_DRIVER, wait_until_safe_for_rendering),
    ("JQc0956gCf0", "sceAgcDriverRegisterMultipleResources", LIB_AGC_DRIVER, register_multiple_resources),
    ("MM4IZSEYytQ", "sceAgcDriverSetHsOffchipParam", LIB_AGC_DRIVER, set_hs_offchip_param),
    ("b4fpgH5ZXxQ", "sceAgcDriverSubmitCommandBuffer", LIB_AGC_DRIVER, submit_command_buffer),
    ("Fj7r9EHzF38", "sceAgcDriverSubmitMultiCommandBuffers", LIB_AGC_DRIVER, submit_multi_command_buffers),
    ("T6xuVw0KUJo", "sceAgcDebugRaiseException", LIB_AGC, debug_raise_exception),
    ("1QTqF-hYqQE", "sceAgcDriverSetValidationDisabled", LIB_AGC_DRIVER, debug_unavailable),
    ("m9n9wJC2D+4", "sceAgcDriverGetCaptureConfig", LIB_AGC_DRIVER, debug_unavailable),
    ("jwKQ1B98Q0g", "sceAgcDriverSetCaptureConfig", LIB_AGC_DRIVER, debug_unavailable),
    ("+ADiO6FQvEA", "sceAgcDriverGetGpuPrintfWorkArea", LIB_AGC_DRIVER, feature_unavailable),
    ("J6S2TMmGWgA", "sceAgcDriverSetGpuPrintfEnabled", LIB_AGC_DRIVER, feature_unavailable),
    ("Jkypw0U-HYo", "sceAgcDriverDbgWriteRegister", LIB_AGC_DRIVER, feature_unavailable),
    ("g68eYcZS7PY", "sceAgcDriverGetGpuRefClks", LIB_AGC_DRIVER, feature_unavailable),
    ("K15VxhSJOfc", "sceAgcDriverIsSuspendPointInFlight", LIB_AGC_DRIVER, feature_unavailable),
    ("-MqhSEVvAPw", "sceAgcDriverDbgReadRegister", LIB_AGC_DRIVER, returns_ok),
    ("HMnVBVUyajk", "sceAgcDriverIsPaDebug", LIB_AGC_DRIVER, returns_false),
    ("o2gvlIFMF1o", "sceAgcDriverGetExceptionStatus", LIB_AGC_DRIVER, returns_ok),
    ("xDPdCurOujQ", "sceAgcDriverGetTraceInitiator", LIB_AGC_DRIVER, returns_ok),
    ("KtIXf8La96k", "sceAgcDriverRegisterGpuResetCallbacks", LIB_AGC_DRIVER, returns_ok),
    ("DDCQA2fPQj0", "sceAgcDriverUnregisterGpuResetCallbacks", LIB_AGC_DRIVER, returns_ok),
    ("BbI8si4o-fA", "sceAgcDriverSysSubmitFlipHandleProxy", LIB_AGC_DRIVER, returns_ok),
    ("t30LG1ibIJE", "sceAgcDriverUserDataImmediateWrite", LIB_AGC_DRIVER, returns_ok),
    ("tLTma0k0U3E", "sceAgcDriverUserDataImmediateWritePacket", LIB_AGC_DRIVER, returns_ok),
    ("Dg-D0+2-m8g", "sceAgdResetGpu", LIB_AGC_DRIVER, gpu_reset_unavailable),
    ("pto0YBGGOJo", "sceAgcDriverGetCoredumpMode", LIB_AGC_DRIVER, get_coredump_mode),
    ("T8xEHO2Y+XU", "sceAgcDriverAllocateToolMemoryForGpuReset", LIB_AGC_DRIVER, allocate_tool_memory_for_gpu_reset),
    ("4XJkjqVrJrs", "sceAgcDriverGetAllocatedToolMemoryForGpuReset", LIB_AGC_DRIVER, get_allocated_tool_memory_for_gpu_reset),
    ("gyVTZWyySpM", "sceAgcDriverGetSetWorkloadsActivePacketSize", LIB_AGC_DRIVER, get_set_workloads_active_packet_size),
    ("WNyjOWq8-Vk", "sceAgcDriverGetSetWorkloadCompletePacketSize", LIB_AGC_DRIVER, get_set_workload_complete_packet_size),
    ("k8rLr8nq-hE", "sceAgcDriverGetSetWorkloadStreamInactivePacketSize", LIB_AGC_DRIVER, get_set_workload_stream_inactive_packet_size),
    ("gtvWv7DEKD4", "sceAgcDriverNopDataGetPacketSize", LIB_AGC_DRIVER, nop_data_get_packet_size),
    ("VhLnEiTuuWo", "sceAgcDriverUserDataGetPacketSize", LIB_AGC_DRIVER, user_data_get_packet_size),
    ("-CSrlr1yY-o", "sceAgcDriverRegisterOwnerForSystem", LIB_AGC_DRIVER, registration_interface_unavailable),
    ("VerAUWCtrX0", "sceAgcDriverInitResourceRegistrationInterface", LIB_AGC_DRIVER, registration_interface_unavailable),
    ("-JFkLpCDeHM", "sceAgcDriverSetResourceRegistrationInfo", LIB_AGC_DRIVER, returns_ok),
];
