// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceNgs2 HLE: systems, racks and voices driving the [`engine`] render
//! graph (sampler -> submixer -> mastering), waveform container parsing
//! (RIFF/WAVE PCM/float/ATRAC9 and "VAGp"), panning and geometry helpers.
//!
//! Handles are guest addresses: a system or rack handle is the working
//! buffer the title supplied (or its allocator returned); voice handles are
//! fixed-stride slots inside the rack buffer. All state lives host-side in
//! one engine guarded by a recursive, guest-visible lock (the same lock
//! sceNgs2RackLock / sceNgs2SystemLock take), so voice callbacks running
//! inside a render may call back into the library on the same thread.

pub mod engine;
pub mod filter;
pub mod geom;
pub mod reverb;
pub mod vag;
pub mod waveform;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};
use engine::{
    Allocator, CallbackInfo, ContextBuffer, Engine, Env, RackOption, RenderBuffer, SystemOption, Vid,
    WaveformFormat, CONTEXT_BUFFER_INFO_SIZE, ERR_INVALID_ARGUMENT, ERR_INVALID_OUT_ADDRESS, ERR_INVALID_OUT_SIZE,
    ERR_INVALID_RACK_HANDLE, ERR_INVALID_SYSTEM_HANDLE, ERR_INVALID_VOICE_HANDLE, ERR_INVALID_WAVEFORM_DATA,
    ERR_INVALID_WAVEFORM_FORMAT, ERR_UNKNOWN_WAVEFORM_FORMAT, OK, RACK_HEADER_SIZE, RACK_OPTION_MAX_SIZE,
    RACK_INFO_SIZE, SYSTEM_BUFFER_SIZE, SYSTEM_INFO_SIZE, SYSTEM_OPTION_SIZE, VOICE_CHANNELS_INFO_SIZE,
    VOICE_INFO_CHANNELS, VOICE_PORT_INFO_SIZE, VOICE_STRIDE, WAVEFORM_BLOCK_SIZE,
};
use std::cell::{Cell, UnsafeCell};
use std::io::{Read, Seek, SeekFrom};
use std::sync::{Condvar, Mutex};
use std::thread::ThreadId;

const LIB: &str = "libSceNgs2";

const fn res(code: i32) -> HleResult {
    code as HleResult
}

/// SCE_NGS2_ERROR_INVALID_WAVEFORM_ADDRESS (null or unreadable data).
const ERR_INVALID_WAVEFORM_ADDRESS: i32 = 0x804A_8055u32 as i32;
/// SCE_NGS2_ERROR_NULL_PARAM_LIST: sceNgs2VoiceControl without parameters.
const ERR_NULL_PARAM_LIST: i32 = 0x804A_8360u32 as i32;
/// The console's answer is not known (kernel ENOENT, as an unimplemented import).
const ERR_NOT_IMPLEMENTED: i32 = 0x8002_0002u32 as i32;
/// SCE_NGS2_ERROR_INVALID_NUM_CHANNELS: more pan parameters than eight.
const ERR_INVALID_NUM_CHANNELS: i32 = 0x804A_8052u32 as i32;
/// SCE_NGS2_ERROR_INVALID_PAN_* (measured on a PS5).
const ERR_INVALID_PAN_UNIT_ANGLE: i32 = 0x804A_8450u32 as i32;
const ERR_INVALID_PAN_SPEAKER: i32 = 0x804A_8451u32 as i32;
const ERR_INVALID_PAN_MATRIX_FORMAT: i32 = 0x804A_8452u32 as i32;
const ERR_INVALID_PAN_WORK: i32 = 0x804A_8453u32 as i32;
const ERR_INVALID_PAN_PARAM: i32 = 0x804A_8454u32 as i32;

/// Container headers are small; RIFF/VAG descriptors always fit inside this.
const MAX_WAVEFORM_HEADER_BYTES: u64 = 0x1000;
const MAX_WAVEFORM_PATH_BYTES: usize = 1024;
/// sizeof(SceNgs2RenderBufferInfo): {ptr@0, size@8, waveformType@16,
/// numChannels@20}.
const RENDER_BUFFER_INFO_SIZE: u64 = 0x18;
const MAX_RENDER_BUFFERS: u32 = 64;
/// Largest voice state image written by sceNgs2VoiceGetState.
const MAX_VOICE_STATE_SIZE: u64 = 0x1000;

// ===========================================================================
// Engine access: recursive guest-visible lock
// ===========================================================================

struct EngineCell(UnsafeCell<Engine>);
// SAFETY: the engine is only touched by the thread owning GATE.
unsafe impl Sync for EngineCell {}

static ENGINE: EngineCell = EngineCell(UnsafeCell::new(Engine::new()));

struct Gate {
    owner: Option<ThreadId>,
    depth: u32,
}

static GATE: Mutex<Gate> = Mutex::new(Gate { owner: None, depth: 0 });
static GATE_CV: Condvar = Condvar::new();

thread_local! {
    /// Engine handed to a voice callback running on this thread; re-entrant
    /// calls use it instead of the global cell.
    static ACTIVE: Cell<*mut Engine> = const { Cell::new(std::ptr::null_mut()) };
}

fn gate_acquire() {
    let me = std::thread::current().id();
    let mut g = GATE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        match g.owner {
            None => {
                g.owner = Some(me);
                g.depth = 1;
                return;
            }
            Some(owner) if owner == me => {
                g.depth += 1;
                return;
            }
            Some(_) => g = crate::signals::wait_cv(&GATE, &GATE_CV, g, None).0,
        }
    }
}

/// Release one level of the lock; false when this thread does not own it.
fn gate_release() -> bool {
    let me = std::thread::current().id();
    let mut g = GATE.lock().unwrap_or_else(|e| e.into_inner());
    if g.owner != Some(me) {
        return false;
    }
    g.depth -= 1;
    if g.depth == 0 {
        g.owner = None;
        GATE_CV.notify_all();
    }
    true
}

struct GateGuard;

impl Drop for GateGuard {
    fn drop(&mut self) {
        gate_release();
    }
}

fn with_engine<R>(f: impl FnOnce(&mut Engine) -> R) -> R {
    gate_acquire();
    let _guard = GateGuard;
    let active = ACTIVE.get();
    // SAFETY: the gate is held by this thread. A non-null ACTIVE pointer is
    // the engine borrow lent to the callback currently running on this
    // thread, which does not use it while the callback runs.
    let eng = unsafe { if active.is_null() { &mut *ENGINE.0.get() } else { &mut *active } };
    f(eng)
}

// ===========================================================================
// Guest environment
// ===========================================================================

fn guest_call(func: u64, args: [u64; 6]) -> u64 {
    ps5cpu::exec::call_guest_nested(func, args)
}

struct GuestEnv<'a, 'b> {
    ctx: &'a CpuContext<'b>,
}

impl Env for GuestEnv<'_, '_> {
    fn read(&mut self, addr: u64, out: &mut [u8]) -> bool {
        out.is_empty() || self.ctx.vm.read(addr, out).is_ok()
    }

    fn write(&mut self, addr: u64, data: &[u8]) -> bool {
        data.is_empty() || self.ctx.vm.write(addr, data).is_ok()
    }

    fn call(&mut self, func: u64, args: [u64; 6]) -> u64 {
        guest_call(func, args)
    }

    fn voice_callback(&mut self, eng: &mut Engine, func: u64, info: &CallbackInfo) {
        let image = info.to_bytes();
        let previous = ACTIVE.replace(eng as *mut Engine);
        guest_call(func, [image.as_ptr() as u64, 0, 0, 0, 0, 0]);
        ACTIVE.set(previous);
    }
}

fn read_array<const N: usize>(ctx: &CpuContext, addr: u64) -> Option<[u8; N]> {
    let mut b = [0u8; N];
    (addr != 0 && ctx.vm.read(addr, &mut b).is_ok()).then_some(b)
}

thread_local! {
    static SCRATCH: Cell<u64> = const { Cell::new(0) };
}

/// Per-thread guest block used to pass a SceNgs2ContextBufferInfo to the
/// title's allocator.
fn guest_scratch(ctx: &CpuContext) -> Option<u64> {
    let cur = SCRATCH.get();
    if cur != 0 {
        return Some(cur);
    }
    let block = crate::pthread::alloc_guest_block(ctx, CONTEXT_BUFFER_INFO_SIZE as u64)?;
    SCRATCH.set(block);
    Some(block)
}

/// Run a guest alloc/free handler on a context buffer; returns the handler
/// result and the buffer as the handler left it.
fn run_buffer_handler(ctx: &CpuContext, func: u64, buffer: &ContextBuffer) -> Option<(i32, ContextBuffer)> {
    let scratch = guest_scratch(ctx)?;
    if !ctx.write_bytes(scratch, &buffer.to_bytes()) {
        return None;
    }
    let r = guest_call(func, [scratch, 0, 0, 0, 0, 0]) as i32;
    let after = read_array::<CONTEXT_BUFFER_INFO_SIZE>(ctx, scratch)?;
    Some((r, ContextBuffer::parse(&after)))
}

fn read_system_option(ctx: &CpuContext, addr: u64) -> Option<SystemOption> {
    if addr == 0 {
        return Some(SystemOption::default());
    }
    let option = SystemOption::parse(&read_array::<SYSTEM_OPTION_SIZE>(ctx, addr)?);
    if option.size != SYSTEM_OPTION_SIZE as u64 {
        log::warn!("[ngs2] system option size {}", option.size);
    }
    Some(option)
}

fn read_rack_option(ctx: &CpuContext, rack_id: u32, addr: u64) -> Result<RackOption, i32> {
    if addr == 0 {
        return RackOption::default_for(rack_id).ok_or(ERR_INVALID_ARGUMENT);
    }
    let size = ctx.read_u64(addr).ok_or(ERR_INVALID_ARGUMENT)?;
    if size < 176 {
        log::warn!("[ngs2] rack option size {size}");
        return Err(ERR_INVALID_ARGUMENT);
    }
    let bytes = ctx.read_bytes(addr, (size as usize).min(RACK_OPTION_MAX_SIZE)).ok_or(ERR_INVALID_ARGUMENT)?;
    check_reverb_size(rack_id, &bytes)?;
    Ok(RackOption::from_bytes(&bytes))
}

/// SceNgs2ReverbRackOption::reverbSize: only the default size 1 is known.
const REVERB_SIZE_OFFSET: usize = 180;

fn check_reverb_size(rack_id: u32, option: &[u8]) -> Result<(), i32> {
    if rack_id != engine::RACK_REVERB {
        return Ok(());
    }
    match option.get(REVERB_SIZE_OFFSET..REVERB_SIZE_OFFSET + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())) {
        Some(1) | None => Ok(()),
        Some(size) => {
            log::warn!("[ngs2] reverb rack size {size} is not supported");
            Err(ERR_INVALID_ARGUMENT)
        }
    }
}

fn rack_buffer_size(option: &RackOption) -> u64 {
    RACK_HEADER_SIZE + VOICE_STRIDE * option.max_voices() as u64
}

// ===========================================================================
// Systems
// ===========================================================================

/// sceNgs2SystemResetOption(SceNgs2SystemOption *).
fn cb_system_reset_option(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 || !ctx.write_bytes(out, &SystemOption::default().to_bytes()) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

/// sceNgs2SystemQueryBufferSize(option, SceNgs2ContextBufferInfo *).
fn cb_system_query_buffer_size(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    let _ = read_system_option(ctx, ctx.rdi());
    let info = ContextBuffer { host_buffer: 0, size: SYSTEM_BUFFER_SIZE, user_data: 0 };
    if !ctx.write_bytes(out, &info.to_bytes()) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

fn create_system(ctx: &CpuContext, option: SystemOption, buffer: ContextBuffer, allocator: Allocator, out: u64) -> HleResult {
    let handle = buffer.host_buffer;
    let uid = with_engine(|e| e.system_create(handle, option.clone(), buffer, allocator));
    log::debug!(
        "[ngs2] system create handle={handle:#x} uid={uid} grain={} rate={}",
        option.num_grain_samples,
        option.sample_rate
    );
    if !ctx.write_u64(out, handle) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

/// sceNgs2SystemCreate(option, const SceNgs2ContextBufferInfo *, *outHandle).
fn cb_system_create(ctx: &mut CpuContext) -> HleResult {
    let (option, info, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if out == 0 {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    let Some(raw) = read_array::<CONTEXT_BUFFER_INFO_SIZE>(ctx, info) else {
        return res(ERR_INVALID_ARGUMENT);
    };
    let buffer = ContextBuffer::parse(&raw);
    if buffer.host_buffer == 0 || buffer.size < SYSTEM_BUFFER_SIZE {
        return res(ERR_INVALID_ARGUMENT);
    }
    let Some(option) = read_system_option(ctx, option) else {
        return res(ERR_INVALID_ARGUMENT);
    };
    create_system(ctx, option, buffer, Allocator::default(), out)
}

/// sceNgs2SystemCreateWithAllocator(option, const SceNgs2BufferAllocator *, *outHandle).
fn cb_system_create_with_allocator(ctx: &mut CpuContext) -> HleResult {
    let (option, allocator, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if out == 0 {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    let Some(allocator) = read_array::<24>(ctx, allocator).map(|b| Allocator::parse(&b)) else {
        return res(ERR_INVALID_ARGUMENT);
    };
    if allocator.alloc == 0 || allocator.free == 0 {
        return res(ERR_INVALID_ARGUMENT);
    }
    let Some(option) = read_system_option(ctx, option) else {
        return res(ERR_INVALID_ARGUMENT);
    };
    let request = ContextBuffer { host_buffer: 0, size: SYSTEM_BUFFER_SIZE, user_data: allocator.user_data };
    let Some((r, buffer)) = run_buffer_handler(ctx, allocator.alloc, &request) else {
        return res(ERR_INVALID_ARGUMENT);
    };
    if r != OK {
        return r as HleResult;
    }
    if buffer.host_buffer == 0 {
        return res(ERR_INVALID_ARGUMENT);
    }
    create_system(ctx, option, buffer, allocator, out)
}

/// Return a destroyed object's buffer: through the title's free handler
/// when it was allocated with one, otherwise into `out` (if given).
fn release_buffer(ctx: &CpuContext, buffer: &ContextBuffer, allocator: &Allocator, out: u64) -> i32 {
    if allocator.free != 0 {
        return run_buffer_handler(ctx, allocator.free, buffer).map_or(ERR_INVALID_ARGUMENT, |(r, _)| r);
    }
    if out != 0 && !ctx.write_bytes(out, &buffer.to_bytes()) {
        return ERR_INVALID_OUT_ADDRESS;
    }
    OK
}

/// sceNgs2SystemDestroy(handle, SceNgs2ContextBufferInfo *). Racks of the
/// system are destroyed with it.
fn cb_system_destroy(ctx: &mut CpuContext) -> HleResult {
    let (handle, out) = (ctx.rdi(), ctx.rsi());
    if with_engine(|e| rendering(e, handle)) {
        log::error!("[ngs2] sceNgs2SystemDestroy({handle:#x}) from inside the system's own render is not supported");
        return NOT_IMPLEMENTED;
    }
    if out != 0 {
        ctx.write_bytes(out, &[0u8; CONTEXT_BUFFER_INFO_SIZE]);
    }
    let removed = with_engine(|e| e.system_destroy(&mut GuestEnv { ctx }, handle));
    let Some((system, racks)) = removed else {
        return res(ERR_INVALID_SYSTEM_HANDLE);
    };
    system_lock_forget(handle);
    for rack in &racks {
        release_buffer(ctx, &rack.buffer, &rack.allocator, 0);
    }
    res(release_buffer(ctx, &system.buffer, &system.allocator, out))
}

/// sceNgs2SystemGetInfo(handle, SceNgs2SystemInfo *, infoSize).
fn cb_system_get_info(ctx: &mut CpuContext) -> HleResult {
    let (handle, out, size) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if out == 0 {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    if size != SYSTEM_INFO_SIZE as u64 {
        return res(ERR_INVALID_OUT_SIZE);
    }
    let info = with_engine(|e| {
        let sys = e.systems.get(&handle)?;
        let mut b = [0u8; SYSTEM_INFO_SIZE];
        b[0..64].copy_from_slice(&sys.option.name);
        b[64..72].copy_from_slice(&handle.to_le_bytes());
        b[72..72 + CONTEXT_BUFFER_INFO_SIZE].copy_from_slice(&sys.buffer.to_bytes());
        b[136..140].copy_from_slice(&sys.uid.to_le_bytes());
        b[140..144].copy_from_slice(&64u32.to_le_bytes());
        b[144..148].copy_from_slice(&sys.option.max_grain_samples.to_le_bytes());
        b[148..152].copy_from_slice(&1u32.to_le_bytes());
        b[152..156].copy_from_slice(&e.rack_count(handle).to_le_bytes());
        b[168..176].copy_from_slice(&sys.render_count.to_le_bytes());
        b[176..180].copy_from_slice(&sys.option.sample_rate.to_le_bytes());
        b[180..184].copy_from_slice(&sys.option.num_grain_samples.to_le_bytes());
        Some(b)
    });
    let Some(info) = info else {
        return res(ERR_INVALID_SYSTEM_HANDLE);
    };
    if !ctx.write_bytes(out, &info) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

/// Unimplemented-import result for the cases the console's answer to is
/// not known.
const NOT_IMPLEMENTED: HleResult = 0x8002_0002u32 as i32 as i64;

/// sceNgs2RackSetUserData(rack, uintptr_t userData).
fn cb_rack_set_user_data(ctx: &mut CpuContext) -> HleResult {
    let (handle, value) = (ctx.rdi(), ctx.rsi());
    with_engine(|e| match e.rack_slot(handle).and_then(|s| e.racks[s].as_mut()) {
        Some(rack) => {
            rack.user_data = value;
            OK as HleResult
        }
        None => res(ERR_INVALID_RACK_HANDLE),
    })
}

/// sceNgs2RackGetUserData(rack, uintptr_t* userData).
fn cb_rack_get_user_data(ctx: &mut CpuContext) -> HleResult {
    let (handle, out) = (ctx.rdi(), ctx.rsi());
    let Some(value) = with_engine(|e| e.rack_slot(handle).and_then(|s| e.rack(s)).map(|r| r.user_data)) else {
        return res(ERR_INVALID_RACK_HANDLE);
    };
    if out == 0 || !ctx.write_u64(out, value) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

/// sceNgs2VoiceGetOwner(voice, SceNgs2Handle* rack, u32* index): the rack
/// that holds the voice and the voice's index in it.
fn cb_voice_get_owner(ctx: &mut CpuContext) -> HleResult {
    let (voice, rack_out, index_out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let Some((rack, index)) = with_engine(|e| e.find_voice(voice).and_then(|v| e.rack(v.slot).map(|r| (r.handle, v.idx as u32)))) else {
        return res(ERR_INVALID_VOICE_HANDLE);
    };
    if rack_out == 0 || index_out == 0 || !ctx.write_u64(rack_out, rack) || !ctx.write_u32(index_out, index) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

/// Report handlers registered with sceNgs2ReportRegisterHandler: handle ->
/// (report type, handler, user data). Reports are never produced, so the
/// handlers are never called.
static REPORT_HANDLERS: std::sync::Mutex<std::collections::BTreeMap<u64, (u32, u64, u64)>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());
static NEXT_REPORT_HANDLE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// sceNgs2ReportRegisterHandler(u32 reportType, handler, uintptr_t userData,
/// SceNgs2Handle* outHandle)
fn cb_report_register_handler(ctx: &mut CpuContext) -> HleResult {
    let (kind, handler, user, out) = (ctx.rdi() as u32, ctx.rsi(), ctx.rdx(), ctx.rcx());
    if out == 0 {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    if handler == 0 {
        log::error!("[ngs2] sceNgs2ReportRegisterHandler: null handler");
        return res(ERR_INVALID_ARGUMENT);
    }
    let handle = NEXT_REPORT_HANDLE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if !ctx.write_u64(out, handle) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    REPORT_HANDLERS.lock().unwrap_or_else(|p| p.into_inner()).insert(handle, (kind, handler, user));
    OK as HleResult
}

/// sceNgs2ReportUnregisterHandler(SceNgs2Handle handle)
fn cb_report_unregister_handler(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    if REPORT_HANDLERS.lock().unwrap_or_else(|p| p.into_inner()).remove(&handle).is_none() {
        log::error!("[ngs2] sceNgs2ReportUnregisterHandler: unknown handle {handle:#x}");
        return res(ERR_INVALID_ARGUMENT);
    }
    OK as HleResult
}

/// sceNgs2SystemSetUserData(handle, uintptr_t userData).
fn cb_system_set_user_data(ctx: &mut CpuContext) -> HleResult {
    let (handle, value) = (ctx.rdi(), ctx.rsi());
    with_engine(|e| match e.systems.get_mut(&handle) {
        Some(sys) => {
            sys.user_data = value;
            OK as HleResult
        }
        None => res(ERR_INVALID_SYSTEM_HANDLE),
    })
}

/// sceNgs2SystemGetUserData(handle, uintptr_t *userData).
fn cb_system_get_user_data(ctx: &mut CpuContext) -> HleResult {
    let (handle, out) = (ctx.rdi(), ctx.rsi());
    let Some(value) = with_engine(|e| e.systems.get(&handle).map(|s| s.user_data)) else {
        return res(ERR_INVALID_SYSTEM_HANDLE);
    };
    if out == 0 || !ctx.write_u64(out, value) {
        log::error!("[ngs2] sceNgs2SystemGetUserData: invalid output address {out:#x}");
        return res(ERR_INVALID_ARGUMENT);
    }
    OK as HleResult
}

fn cb_system_set_grain_samples(ctx: &mut CpuContext) -> HleResult {
    let (handle, grain) = (ctx.rdi(), ctx.rsi() as u32);
    with_engine(|e| match e.systems.get_mut(&handle) {
        Some(sys) => {
            sys.option.num_grain_samples = grain;
            OK as HleResult
        }
        None => res(ERR_INVALID_SYSTEM_HANDLE),
    })
}

fn cb_system_set_sample_rate(ctx: &mut CpuContext) -> HleResult {
    let (handle, rate) = (ctx.rdi(), ctx.rsi() as u32);
    with_engine(|e| match e.systems.get_mut(&handle) {
        Some(_) if !(8000..=192000).contains(&rate) => res(ERR_INVALID_ARGUMENT),
        Some(sys) => {
            sys.option.sample_rate = rate;
            OK as HleResult
        }
        None => res(ERR_INVALID_SYSTEM_HANDLE),
    })
}

// ---------------------------------------------------------------------------
// System locks: sceNgs2SystemLock keeps a system's render from running
// while the title edits voices. It does not block the other calls, a render
// on another thread skips its grain (silence) instead of waiting, and the
// unlock may come from any thread.
// ---------------------------------------------------------------------------

/// Owner and depth of each locked system.
static SYSTEM_LOCKS: Mutex<Vec<(u64, ThreadId, u32)>> = Mutex::new(Vec::new());
static SYSTEM_LOCKS_CV: Condvar = Condvar::new();

fn system_locks() -> std::sync::MutexGuard<'static, Vec<(u64, ThreadId, u32)>> {
    SYSTEM_LOCKS.lock().unwrap_or_else(|e| e.into_inner())
}

fn system_exists(handle: u64) -> bool {
    with_engine(|e| e.systems.contains_key(&handle))
}

/// Take `handle`'s system lock for this thread (recursively), waiting while
/// another thread holds it. False when the system went away.
fn system_lock_acquire(handle: u64) -> bool {
    let me = std::thread::current().id();
    let mut locks = system_locks();
    loop {
        match locks.iter_mut().find(|l| l.0 == handle) {
            None => {
                if !system_exists(handle) {
                    return false;
                }
                locks.push((handle, me, 1));
                return true;
            }
            Some(l) if l.1 == me => {
                l.2 += 1;
                return true;
            }
            Some(_) => locks = crate::signals::wait_cv(&SYSTEM_LOCKS, &SYSTEM_LOCKS_CV, locks, None).0,
        }
    }
}

/// Take the lock for a render: false (and nothing taken) when another
/// thread holds it.
fn system_lock_try(handle: u64) -> bool {
    let me = std::thread::current().id();
    let mut locks = system_locks();
    match locks.iter_mut().find(|l| l.0 == handle) {
        None => {
            locks.push((handle, me, 1));
            true
        }
        Some(l) if l.1 == me => {
            l.2 += 1;
            true
        }
        Some(_) => false,
    }
}

/// Release one level of `handle`'s lock, whoever holds it.
fn system_lock_release(handle: u64) {
    let mut locks = system_locks();
    if let Some(i) = locks.iter().position(|l| l.0 == handle) {
        locks[i].2 -= 1;
        if locks[i].2 == 0 {
            locks.swap_remove(i);
            SYSTEM_LOCKS_CV.notify_all();
        }
    }
}

/// Forget the lock of a destroyed system and wake its waiters.
fn system_lock_forget(handle: u64) {
    let mut locks = system_locks();
    locks.retain(|l| l.0 != handle);
    SYSTEM_LOCKS_CV.notify_all();
}

/// sceNgs2SystemLock(system).
fn cb_system_lock(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    if !system_exists(handle) || !system_lock_acquire(handle) {
        return res(ERR_INVALID_SYSTEM_HANDLE);
    }
    OK as HleResult
}

/// sceNgs2SystemUnlock(system): from any thread; an unlocked system stays
/// unlocked.
fn cb_system_unlock(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    if !system_exists(handle) {
        return res(ERR_INVALID_SYSTEM_HANDLE);
    }
    system_lock_release(handle);
    OK as HleResult
}

/// sceNgs2SystemRender(system, const SceNgs2RenderBufferInfo *, count).
fn cb_system_render(ctx: &mut CpuContext) -> HleResult {
    let (system, infos, count) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u32);
    if count != 0 && infos == 0 {
        return res(ERR_INVALID_ARGUMENT);
    }
    let mut outputs = [RenderBuffer::default(); MAX_RENDER_BUFFERS as usize];
    let count = count.min(MAX_RENDER_BUFFERS) as usize;
    for (i, o) in outputs.iter_mut().enumerate().take(count) {
        let Some(b) = read_array::<{ RENDER_BUFFER_INFO_SIZE as usize }>(ctx, infos + i as u64 * RENDER_BUFFER_INFO_SIZE)
        else {
            return res(ERR_INVALID_OUT_ADDRESS);
        };
        *o = RenderBuffer {
            buffer: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            size: u64::from_le_bytes(b[8..16].try_into().unwrap()),
            waveform_type: u32::from_le_bytes(b[16..20].try_into().unwrap()),
            num_channels: u32::from_le_bytes(b[20..24].try_into().unwrap()),
        };
    }
    if !system_lock_try(system) {
        // Another thread edits the system: this grain is silence.
        return res(with_engine(|e| e.system_silence(&mut GuestEnv { ctx }, system, &outputs[..count])));
    }
    let depth = |e: &mut Engine, delta: i32| {
        if let Some(s) = e.systems.get_mut(&system) {
            s.render_depth = s.render_depth.saturating_add_signed(delta);
        }
    };
    let r = with_engine(|e| {
        depth(e, 1);
        let r = e.system_render(&mut GuestEnv { ctx }, system, &outputs[..count]);
        depth(e, -1);
        r
    });
    system_lock_release(system);
    res(r)
}

/// A system that is rendering on this thread cannot be destroyed (with
/// its racks) from a voice callback.
fn rendering(e: &Engine, system: u64) -> bool {
    e.systems.get(&system).is_some_and(|s| s.render_depth > 0)
}

// ===========================================================================
// Racks
// ===========================================================================

/// sceNgs2RackQueryBufferSize(rackId, option, SceNgs2ContextBufferInfo *).
fn cb_rack_query_buffer_size(ctx: &mut CpuContext) -> HleResult {
    let (rack_id, option, out) = (ctx.rdi() as u32, ctx.rsi(), ctx.rdx());
    if out == 0 {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    let option = match read_rack_option(ctx, rack_id, option) {
        Ok(o) => o,
        Err(e) => return res(e),
    };
    if !ctx.write_u64(out + 8, rack_buffer_size(&option)) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

fn create_rack(
    ctx: &CpuContext,
    system: u64,
    rack_id: u32,
    option: RackOption,
    buffer: ContextBuffer,
    allocator: Allocator,
    out: u64,
) -> HleResult {
    let handle = buffer.host_buffer;
    let created = with_engine(|e| {
        let slot = e.rack_create(&mut GuestEnv { ctx }, system, rack_id, option, buffer, handle)?;
        if let Some(rack) = e.rack_mut(slot) {
            rack.allocator = allocator;
        }
        Ok::<_, i32>(())
    });
    if let Err(e) = created {
        return res(e);
    }
    log::debug!("[ngs2] rack create id={rack_id:#x} handle={handle:#x} system={system:#x}");
    if !ctx.write_u64(out, handle) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

/// sceNgs2RackCreate(system, rackId, option, const SceNgs2ContextBufferInfo *, *outHandle).
fn cb_rack_create(ctx: &mut CpuContext) -> HleResult {
    let (system, rack_id, option, info, out) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx(), ctx.r8());
    if out == 0 {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    if !with_engine(|e| e.systems.contains_key(&system)) {
        return res(ERR_INVALID_SYSTEM_HANDLE);
    }
    let Some(raw) = read_array::<CONTEXT_BUFFER_INFO_SIZE>(ctx, info) else {
        return res(ERR_INVALID_ARGUMENT);
    };
    let buffer = ContextBuffer::parse(&raw);
    if buffer.host_buffer == 0 || buffer.size == 0 {
        return res(ERR_INVALID_ARGUMENT);
    }
    let option = match read_rack_option(ctx, rack_id, option) {
        Ok(o) => o,
        Err(e) => return res(e),
    };
    create_rack(ctx, system, rack_id, option, buffer, Allocator::default(), out)
}

/// sceNgs2RackCreateWithAllocator(system, rackId, option, const SceNgs2BufferAllocator *, *outHandle).
fn cb_rack_create_with_allocator(ctx: &mut CpuContext) -> HleResult {
    let (system, rack_id, option, allocator, out) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx(), ctx.r8());
    if out == 0 {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    if !with_engine(|e| e.systems.contains_key(&system)) {
        return res(ERR_INVALID_SYSTEM_HANDLE);
    }
    let Some(allocator) = read_array::<24>(ctx, allocator).map(|b| Allocator::parse(&b)) else {
        return res(ERR_INVALID_ARGUMENT);
    };
    if allocator.alloc == 0 || allocator.free == 0 {
        return res(ERR_INVALID_ARGUMENT);
    }
    let option = match read_rack_option(ctx, rack_id, option) {
        Ok(o) => o,
        Err(e) => return res(e),
    };
    let request = ContextBuffer { host_buffer: 0, size: rack_buffer_size(&option), user_data: allocator.user_data };
    let Some((r, buffer)) = run_buffer_handler(ctx, allocator.alloc, &request) else {
        return res(ERR_INVALID_ARGUMENT);
    };
    if r != OK {
        return r as HleResult;
    }
    if buffer.host_buffer == 0 {
        return res(ERR_INVALID_ARGUMENT);
    }
    create_rack(ctx, system, rack_id, option, buffer, allocator, out)
}

/// sceNgs2RackDestroy(rack, SceNgs2ContextBufferInfo *).
fn cb_rack_destroy(ctx: &mut CpuContext) -> HleResult {
    let (handle, out) = (ctx.rdi(), ctx.rsi());
    if out != 0 {
        ctx.write_bytes(out, &[0u8; CONTEXT_BUFFER_INFO_SIZE]);
    }
    if handle == 0 {
        return res(ERR_INVALID_RACK_HANDLE);
    }
    let in_render = with_engine(|e| e.rack_slot(handle).and_then(|s| e.rack(s)).is_some_and(|r| rendering(e, r.system)));
    if in_render {
        log::error!("[ngs2] sceNgs2RackDestroy({handle:#x}) from inside its system's render is not supported");
        return NOT_IMPLEMENTED;
    }
    let Some(rack) = with_engine(|e| e.rack_destroy(&mut GuestEnv { ctx }, handle)) else {
        return res(ERR_INVALID_RACK_HANDLE);
    };
    res(release_buffer(ctx, &rack.buffer, &rack.allocator, out))
}

/// sceNgs2RackLock: take the library lock (held across the return).
fn cb_rack_lock(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    if with_engine(|e| e.rack_slot(handle)).is_none() {
        return res(ERR_INVALID_RACK_HANDLE);
    }
    gate_acquire();
    OK as HleResult
}

fn cb_rack_unlock(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    if with_engine(|e| e.rack_slot(handle)).is_none() {
        return res(ERR_INVALID_RACK_HANDLE);
    }
    gate_release();
    OK as HleResult
}

/// sceNgs2RackGetVoiceHandle(rack, voiceId, *outHandle).
fn cb_rack_get_voice_handle(ctx: &mut CpuContext) -> HleResult {
    let (rack, id, out) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx());
    // Checked in this order: the rack handle, the output, the index.
    if !with_engine(|e| e.rack_slot(rack).is_some()) {
        return res(ERR_INVALID_RACK_HANDLE);
    }
    if out == 0 {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    match with_engine(|e| e.rack_voice_handle(rack, id)) {
        Ok(handle) if ctx.write_u64(out, handle) => OK as HleResult,
        Ok(_) => res(ERR_INVALID_OUT_ADDRESS),
        Err(e) => res(e),
    }
}

/// sceNgs2RackGetInfo(rack, SceNgs2RackInfo *, infoSize).
fn cb_rack_get_info(ctx: &mut CpuContext) -> HleResult {
    let (handle, out, size) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if out == 0 {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    if size != RACK_INFO_SIZE as u64 {
        return res(ERR_INVALID_OUT_SIZE);
    }
    let Some(info) = with_engine(|e| e.rack_info(handle)) else {
        return res(ERR_INVALID_RACK_HANDLE);
    };
    if !ctx.write_bytes(out, &info) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

// ===========================================================================
// Voices
// ===========================================================================

/// sceNgs2VoiceGetPortInfo(voice, port, SceNgs2VoicePortInfo *, infoSize).
fn cb_voice_get_port_info(ctx: &mut CpuContext) -> HleResult {
    let (voice, port, out, size) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx());
    if out == 0 {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    if size != VOICE_PORT_INFO_SIZE as u64 {
        return res(ERR_INVALID_OUT_SIZE);
    }
    let info = with_engine(|e| e.find_voice(voice).map(|vid| e.voice_port_info(vid, port)));
    let info = match info {
        None => return res(ERR_INVALID_VOICE_HANDLE),
        Some(None) => {
            log::error!("[ngs2] sceNgs2VoiceGetPortInfo: port {port} is out of range");
            return res(ERR_INVALID_ARGUMENT);
        }
        Some(Some(info)) => info,
    };
    if !ctx.write_bytes(out, &info) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

/// sceNgs2VoiceQueryInfo(voice, infoId, void *info, infoSize). Only the
/// channel count (0x4001) is implemented.
fn cb_voice_query_info(ctx: &mut CpuContext) -> HleResult {
    let (voice, id, out, size) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx());
    if out == 0 {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    let Some(channels) = with_engine(|e| e.find_voice(voice).and_then(|vid| e.voice(vid)).map(|v| v.channels)) else {
        return res(ERR_INVALID_VOICE_HANDLE);
    };
    if id != VOICE_INFO_CHANNELS {
        log::error!("[ngs2] sceNgs2VoiceQueryInfo: info {id:#x} is not implemented");
        return res(ERR_NOT_IMPLEMENTED);
    }
    if size != VOICE_CHANNELS_INFO_SIZE as u64 {
        return res(ERR_INVALID_OUT_SIZE);
    }
    let mut b = [0u8; VOICE_CHANNELS_INFO_SIZE];
    b[0..4].copy_from_slice(&channels.to_le_bytes());
    if !ctx.write_bytes(out, &b) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

fn with_voice(handle: u64, f: impl FnOnce(&mut Engine, Vid) -> i32) -> HleResult {
    res(with_engine(|e| match e.find_voice(handle) {
        Some(vid) => f(e, vid),
        None => ERR_INVALID_VOICE_HANDLE,
    }))
}

/// sceNgs2VoiceControl(voice, const SceNgs2VoiceParamHead *).
fn cb_voice_control(ctx: &mut CpuContext) -> HleResult {
    let (voice, params) = (ctx.rdi(), ctx.rsi());
    // A null list is refused before the handle is looked at.
    if params == 0 {
        return res(ERR_NULL_PARAM_LIST);
    }
    with_voice(voice, |e, vid| e.voice_control(&mut GuestEnv { ctx }, vid, params))
}

/// sceNgs2VoiceRunCommands(voice, commands, numCommands): 16-byte commands
/// {id, flags u8, type u8, count u16, value}.
fn cb_voice_run_commands(ctx: &mut CpuContext) -> HleResult {
    let (voice, commands, count) = (ctx.rdi(), ctx.rsi(), ctx.rdx().min(4096) as usize);
    // The voice is checked first, then the command list.
    let known = with_voice(voice, |_, _| OK);
    if known != OK as HleResult {
        return known;
    }
    let raw = if count == 0 { Some(Vec::new()) } else if commands == 0 { None } else { ctx.read_bytes(commands, count * 16) };
    let Some(raw) = raw else {
        return res(ERR_INVALID_ARGUMENT);
    };
    with_voice(voice, |e, vid| {
        for c in raw.chunks_exact(16) {
            let id = u32::from_le_bytes(c[0..4].try_into().unwrap());
            let ty = c[5];
            let n = u16::from_le_bytes([c[6], c[7]]) as usize;
            let value = u64::from_le_bytes(c[8..16].try_into().unwrap());
            let index = id >> 24;
            match (id & 0x00ff_ffff, ty) {
                (2, 4) => e.voice_mut(vid).map_or((), |v| v.set_event(value as u32)),
                (5, 0x11) => {
                    let levels: Vec<f32> = ctx
                        .read_bytes(value, n.min(64 * 64) * 4)
                        .unwrap_or_default()
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    if let Some(v) = e.voice_mut(vid) {
                        v.set_matrix(index, &levels);
                    }
                }
                (6, 1) => e.voice_mut(vid).map_or((), |v| v.set_volume(index, f32::from_bits(value as u32))),
                (7, 3) => e.voice_mut(vid).map_or((), |v| v.set_port_matrix(index, value as u32 as i32)),
                _ => log::warn!("[ngs2] unknown voice command id={id:#x} type={ty:#x}"),
            }
        }
        OK
    })
}

/// sceNgs2VoiceGetState(voice, *state, stateSize).
fn cb_voice_get_state(ctx: &mut CpuContext) -> HleResult {
    let (voice, out, size) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    // The voice first, then the size (at least the flags), the address and
    // the exact size of the voice's rack type.
    let mut want = None;
    let r = with_voice(voice, |e, vid| {
        want = e.voice_state_size(vid);
        OK
    });
    if r != OK as HleResult {
        return r;
    }
    if size < 8 {
        return res(ERR_INVALID_OUT_SIZE);
    }
    if out == 0 {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    if want.is_some_and(|w| w as u64 != size) {
        return res(ERR_INVALID_OUT_SIZE);
    }
    let mut image = vec![0u8; size.min(MAX_VOICE_STATE_SIZE) as usize];
    let r = with_voice(voice, |e, vid| {
        e.voice_state(vid, &mut image);
        OK
    });
    if r != OK as HleResult {
        return r;
    }
    if !ctx.write_bytes(out, &image) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

/// sceNgs2VoiceGetStateFlags(voice, uint32_t *flags).
fn cb_voice_get_state_flags(ctx: &mut CpuContext) -> HleResult {
    let (voice, out) = (ctx.rdi(), ctx.rsi());
    if out == 0 {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    let mut flags = 0;
    let r = with_voice(voice, |e, vid| {
        flags = e.voice(vid).map_or(0, |v| v.state_flags);
        OK
    });
    if r != OK as HleResult {
        return r;
    }
    if !ctx.write_u32(out, flags) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

// ===========================================================================
// Waveform parsing
// ===========================================================================

/// sceNgs2ParseWaveformData(const void *data, size_t size, SceNgs2WaveformInfo *out).
fn cb_parse_waveform_data(ctx: &mut CpuContext) -> HleResult {
    let (data, size, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if out == 0 {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    ctx.write_bytes(out, &[0u8; waveform::WAVEFORM_INFO_SIZE]);
    if data == 0 || size == 0 {
        return res(ERR_INVALID_WAVEFORM_ADDRESS);
    }
    // Only the container header is needed.
    let mut header = vec![0u8; size.min(MAX_WAVEFORM_HEADER_BYTES) as usize];
    if ctx.vm.read(data, &mut header).is_err() {
        return res(ERR_INVALID_WAVEFORM_ADDRESS);
    }
    write_parsed_waveform(ctx, &header, size, out)
}

/// sceNgs2ParseWaveformFile(const char *path, uint64_t offset, SceNgs2WaveformInfo *out).
fn cb_parse_waveform_file(ctx: &mut CpuContext) -> HleResult {
    let (path_addr, offset, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if out == 0 {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    let Some(guest_path) = ctx.read_cstring(path_addr, MAX_WAVEFORM_PATH_BYTES).filter(|p| !p.is_empty()) else {
        return res(ERR_INVALID_WAVEFORM_ADDRESS);
    };
    let read = || -> std::io::Result<Option<(Vec<u8>, u64)>> {
        let host = crate::filesystem::resolve_guest_path(&guest_path)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "unresolved guest path"))?;
        let mut f = std::fs::File::open(host)?;
        let len = f.metadata()?.len();
        if offset >= len {
            return Ok(None);
        }
        f.seek(SeekFrom::Start(offset))?;
        let mut header = vec![0u8; (len - offset).min(MAX_WAVEFORM_HEADER_BYTES) as usize];
        f.read_exact(&mut header)?;
        Ok(Some((header, len)))
    };
    match read() {
        Ok(Some((header, len))) => write_parsed_waveform(ctx, &header, len - offset, out),
        Ok(None) => res(ERR_INVALID_WAVEFORM_DATA),
        Err(e) => {
            log::debug!("[ngs2] parse-waveform-file open failed path={guest_path} error={e}");
            res(ERR_INVALID_WAVEFORM_DATA)
        }
    }
}

/// sceNgs2ParseWaveformUser(handler, userData, out): the read handler is a
/// guest callback the HLE does not drive, so the data is reported as
/// unparsable.
fn cb_parse_waveform_user(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdx() == 0 { res(ERR_INVALID_OUT_ADDRESS) } else { res(ERR_INVALID_WAVEFORM_DATA) }
}

fn write_parsed_waveform(ctx: &CpuContext, header: &[u8], total: u64, out: u64) -> HleResult {
    if header.len() < 12 {
        return res(ERR_INVALID_WAVEFORM_DATA);
    }
    let Some(mut parsed) = waveform::try_parse(header) else {
        let known = waveform::is_supported_container(header);
        log::debug!("[ngs2] parse-waveform {}", if known { "unsupported-format" } else { "unknown-container" });
        return res(if known { ERR_INVALID_WAVEFORM_FORMAT } else { ERR_UNKNOWN_WAVEFORM_FORMAT });
    };
    // Only the header was copied in; when the image is longer the data chunk
    // continues past it, so size it from the caller's total length.
    if (header.len() as u64) < total && total > parsed.data_offset as u64 {
        let available = total - parsed.data_offset as u64;
        parsed.data_size = available.min(u32::MAX as u64) as u32;
        if parsed.audio_unit_size != 0 && parsed.waveform_type != waveform::WAVEFORM_TYPE_ATRAC9 {
            parsed.num_samples =
                (parsed.data_size / parsed.audio_unit_size).wrapping_mul(parsed.num_audio_unit_samples);
            if parsed.loop_end_position == 0 && parsed.num_samples != 0 {
                parsed.loop_end_position = parsed.num_samples - 1;
            }
        }
    }
    if !ctx.write_bytes(out, &waveform::write(&parsed)) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    log::debug!(
        "[ngs2] parse-waveform type={:#x} channels={} rate={} dataOffset={:#x} dataSize={:#x} samples={}",
        parsed.waveform_type,
        parsed.num_channels,
        parsed.sample_rate,
        parsed.data_offset,
        parsed.data_size,
        parsed.num_samples
    );
    OK as HleResult
}

/// sceNgs2CalcWaveformBlock(const SceNgs2WaveformFormat *, samplePos, numSamples, SceNgs2WaveformBlock *).
fn cb_calc_waveform_block(ctx: &mut CpuContext) -> HleResult {
    let (format, pos, num, out) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx() as u32, ctx.rcx());
    if out == 0 {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    let format = read_array::<24>(ctx, format).map(|b| WaveformFormat::parse(&b));
    let (r, block) = geom::calc_waveform_block(format.as_ref(), pos, num);
    if !ctx.write_bytes(out, &block.to_bytes()[..WAVEFORM_BLOCK_SIZE]) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    res(r)
}

// ===========================================================================
// Pan / geometry
// ===========================================================================

/// Single-precision floating-point argument `i` (xmm`i`).
fn float_arg(i: usize) -> f32 {
    #[cfg(test)]
    {
        TEST_XMM.with(|x| x.get()[i])
    }
    #[cfg(not(test))]
    {
        ps5cpu::exec::trap_xmm_low(i).map_or(0.0, |v| f32::from_bits(v as u32))
    }
}

#[cfg(test)]
thread_local! {
    pub(crate) static TEST_XMM: Cell<[f32; 8]> = const { Cell::new([0.0; 8]) };
}

fn pan_error(e: geom::PanError) -> HleResult {
    use geom::PanError as E;
    match e {
        E::InvalidOutAddress => res(ERR_INVALID_OUT_ADDRESS),
        E::UnitAngle => res(ERR_INVALID_PAN_UNIT_ANGLE),
        E::Speaker => res(ERR_INVALID_PAN_SPEAKER),
        E::MatrixFormat => res(ERR_INVALID_PAN_MATRIX_FORMAT),
        E::Work => res(ERR_INVALID_PAN_WORK),
        E::Param => res(ERR_INVALID_PAN_PARAM),
        E::NumChannels => res(ERR_INVALID_NUM_CHANNELS),
        E::NoReturn => {
            log::error!("[ngs2] pan angle the console's library does not return for (infinite or too large to wrap)");
            res(ERR_INVALID_ARGUMENT)
        }
    }
}

/// sceNgs2PanInit(SceNgs2PanWork *, const float *speakerAngles, float unitAngle, numSpeakers).
fn cb_pan_init(ctx: &mut CpuContext) -> HleResult {
    let (work, angles, num) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u32);
    if work == 0 {
        return pan_error(geom::PanError::InvalidOutAddress);
    }
    let n = num.min(8) as usize;
    let speaker_angles: Option<Vec<f32>> = if angles != 0 {
        let Some(raw) = ctx.read_bytes(angles, n * 4) else { return res(ERR_INVALID_ARGUMENT) };
        Some(raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    } else {
        None
    };
    let Some(current) = ctx.read_bytes(work, geom::PAN_WORK_SIZE) else { return res(ERR_INVALID_OUT_ADDRESS) };
    let mut image: [u8; geom::PAN_WORK_SIZE] = current.try_into().unwrap();
    if let Err(e) = geom::pan_init(&mut image, speaker_angles.as_deref(), float_arg(0), num) {
        return pan_error(e);
    }
    if !ctx.write_bytes(work, &image) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

/// sceNgs2PanGetVolumeMatrix(work, const SceNgs2PanParam *, numParams, matrixFormat, float *out):
/// a matrixFormat x numParams matrix; nothing is written on an error.
fn cb_pan_get_volume_matrix(ctx: &mut CpuContext) -> HleResult {
    let (work, params, num, format, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u32, ctx.rcx() as u32, ctx.r8());
    if work == 0 {
        return pan_error(geom::PanError::Work);
    }
    if params == 0 {
        return pan_error(geom::PanError::Param);
    }
    if out == 0 {
        return pan_error(geom::PanError::InvalidOutAddress);
    }
    let Some(work) = ctx.read_bytes(work, geom::PAN_WORK_SIZE) else { return res(ERR_INVALID_ARGUMENT) };
    if let Err(e) = geom::pan_check(&work, num as usize, format) {
        return pan_error(e);
    }
    let num = num as usize;
    let Some(raw) = ctx.read_bytes(params, num * geom::PAN_PARAM_SIZE) else { return res(ERR_INVALID_ARGUMENT) };
    let params: Vec<geom::PanParam> = raw.chunks_exact(geom::PAN_PARAM_SIZE).map(geom::PanParam::parse).collect();
    let matrix = match geom::pan_volume_matrix(&work, &params, format) {
        Ok(m) => m,
        Err(e) => return pan_error(e),
    };
    let bytes: Vec<u8> = matrix.iter().flat_map(|f| f.to_le_bytes()).collect();
    if !ctx.write_bytes(out, &bytes) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

fn cb_geom_reset_listener_param(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 || !ctx.write_bytes(out, &geom::geom_reset_listener_param()) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

fn cb_geom_reset_source_param(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 || !ctx.write_bytes(out, &geom::geom_reset_source_param()) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

/// sceNgs2GeomCalcListener(const SceNgs2GeomListenerParam *, SceNgs2GeomListenerWork *, flags).
fn cb_geom_calc_listener(ctx: &mut CpuContext) -> HleResult {
    let (param, out, flags) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u32);
    let Some(param) = read_array::<{ geom::GEOM_LISTENER_PARAM_SIZE }>(ctx, param) else {
        return res(ERR_INVALID_ARGUMENT);
    };
    if out == 0 || !ctx.write_bytes(out, &geom::geom_calc_listener(&param, flags)) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

/// sceNgs2GeomApply(const SceNgs2GeomListenerWork *, const SceNgs2GeomSourceParam *, SceNgs2GeomAttribute *, flags).
fn cb_geom_apply(ctx: &mut CpuContext) -> HleResult {
    let (listener, source, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if read_array::<{ geom::GEOM_LISTENER_WORK_SIZE }>(ctx, listener).is_none() {
        return res(ERR_INVALID_ARGUMENT);
    }
    let Some(source) = read_array::<{ geom::GEOM_SOURCE_PARAM_SIZE }>(ctx, source) else {
        return res(ERR_INVALID_ARGUMENT);
    };
    if out == 0 || !ctx.write_bytes(out, &geom::geom_apply(&source)) {
        return res(ERR_INVALID_OUT_ADDRESS);
    }
    OK as HleResult
}

// ===========================================================================
// Registration
// ===========================================================================

type Export = (&'static str, &'static str, HleCallback);

static EXPORTS: &[Export] = &[
    ("AQkj7C0f3PY", "sceNgs2SystemResetOption", cb_system_reset_option),
    ("pgFAiLR5qT4", "sceNgs2SystemQueryBufferSize", cb_system_query_buffer_size),
    ("koBbCMvOKWw", "sceNgs2SystemCreate", cb_system_create),
    ("mPYgU4oYpuY", "sceNgs2SystemCreateWithAllocator", cb_system_create_with_allocator),
    ("u-WrYDaJA3k", "sceNgs2SystemDestroy", cb_system_destroy),
    ("vU7TQ62pItw", "sceNgs2SystemGetInfo", cb_system_get_info),
    ("U546k6orxQo", "sceNgs2RackCreateWithAllocator", cb_rack_create_with_allocator),
    ("cLV4aiT9JpA", "sceNgs2RackCreate", cb_rack_create),
    ("0eFLVCfWVds", "sceNgs2RackQueryBufferSize", cb_rack_query_buffer_size),
    ("lCqD7oycmIM", "sceNgs2RackDestroy", cb_rack_destroy),
    ("MwmHz8pAdAo", "sceNgs2RackGetVoiceHandle", cb_rack_get_voice_handle),
    ("MzTa7VLjogY", "sceNgs2RackLock", cb_rack_lock),
    ("++YZ7P9e87U", "sceNgs2RackUnlock", cb_rack_unlock),
    ("uu94irFOGpA", "sceNgs2VoiceControl", cb_voice_control),
    ("AbYvTOZ8Pts", "sceNgs2VoiceRunCommands", cb_voice_run_commands),
    ("-TOuuAQ-buE", "sceNgs2VoiceGetState", cb_voice_get_state),
    ("rEh728kXk3w", "sceNgs2VoiceGetStateFlags", cb_voice_get_state_flags),
    ("i0VnXM-C9fc", "sceNgs2SystemRender", cb_system_render),
    ("l4Q2dWEH6UM", "sceNgs2SystemSetGrainSamples", cb_system_set_grain_samples),
    ("-tbc2SxQD60", "sceNgs2SystemSetSampleRate", cb_system_set_sample_rate),
    ("gThZqM5PYlQ", "sceNgs2SystemLock", cb_system_lock),
    ("JXRC5n0RQls", "sceNgs2SystemUnlock", cb_system_unlock),
    ("hyVLT2VlOYk", "sceNgs2ParseWaveformData", cb_parse_waveform_data),
    ("iprCTXPVWMI", "sceNgs2ParseWaveformFile", cb_parse_waveform_file),
    ("t9T0QM17Kvo", "sceNgs2ParseWaveformUser", cb_parse_waveform_user),
    ("3pCNbVM11UA", "sceNgs2CalcWaveformBlock", cb_calc_waveform_block),
    ("xa8oL9dmXkM", "sceNgs2PanInit", cb_pan_init),
    ("gbMKV+8Enuo", "sceNgs2PanGetVolumeMatrix", cb_pan_get_volume_matrix),
    ("7Lcfo8SmpsU", "sceNgs2GeomResetListenerParam", cb_geom_reset_listener_param),
    ("0lbbayqDNoE", "sceNgs2GeomResetSourceParam", cb_geom_reset_source_param),
    ("1WsleK-MTkE", "sceNgs2GeomCalcListener", cb_geom_calc_listener),
    ("eF8yRCC6W64", "sceNgs2GeomApply", cb_geom_apply),
    ("M4LYATRhRUE", "sceNgs2RackGetInfo", cb_rack_get_info),
    ("4lFaRxd-aLs", "sceNgs2SystemGetUserData", cb_system_get_user_data),
    ("GZB2v0XnG0k", "sceNgs2SystemSetUserData", cb_system_set_user_data),
    ("JNTMIaBIbV4", "sceNgs2RackSetUserData", cb_rack_set_user_data),
    ("Mn4XNDg03XY", "sceNgs2RackGetUserData", cb_rack_get_user_data),
    ("W-Z8wWMBnhk", "sceNgs2VoiceGetOwner", cb_voice_get_owner),
    ("uBIN24Tv2MI", "sceNgs2ReportRegisterHandler", cb_report_register_handler),
    ("nPzb7Ly-VjE", "sceNgs2ReportUnregisterHandler", cb_report_unregister_handler),
    ("WCayTgob7-o", "sceNgs2VoiceGetPortInfo", cb_voice_get_port_info),
    ("9eic4AmjGVI", "sceNgs2VoiceQueryInfo", cb_voice_query_info),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod sampler_tests;
#[cfg(test)]
mod tests;
