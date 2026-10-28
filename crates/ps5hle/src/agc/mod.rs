// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceAgc / libSceAgcDriver HLE and the PM4 command interpreter.
//!
//! The AGC library is implemented at the HLE level: every `sceAgcDcb*`,
//! `sceAgcAcb*` and `sceAgcCb*` builder writes real PM4 dwords into the
//! guest's own command buffer, and `sceAgcDriverSubmit*` hands the buffer to
//! the PM4 interpreter ([`pm4`]), which runs on a dedicated GPU command
//! thread (one graphics queue plus 56 compute queues).
//!
//! Guest command buffer object (offsets 0x00/0x08 are not used here):
//!
//! | Offset | Field |
//! |---|---|
//! | +0x10 | u64 cursorUp (write pointer) |
//! | +0x18 | u64 cursorDown (end of free space) |
//! | +0x20 | u64 callback `cb(cbuf, dwordsNeeded, userData)` |
//! | +0x28 | u64 userData |
//! | +0x30 | u32 reservedDw |
//!
//! When a builder needs more dwords than remain, the guest callback is
//! invoked (nested guest call on the current guest thread) and the cursors
//! are re-read.
//!
//! `*GetSize` exports return the **byte** size of the packet(s) the matching
//! builder emits; the unit tests check every pair.
//!
//! VideoOut integration goes through [`set_flip_hooks`]: the video-out module
//! registers its GPU flip entry points at start-up, so this module has no
//! compile-time dependency on it.

pub mod builders;
pub(crate) mod defaults;
mod driver;
pub mod frame_times;
pub mod mem;
mod patch;
pub mod pm4;
mod regdefs_data;
mod shader;
mod shader_dump;

#[cfg(test)]
mod slice_tests;
#[cfg(test)]
mod tests;

use crate::context::{CpuContext, CpuRegister};
use crate::registry::{HleRegistry, HleResult};
use std::sync::RwLock;

pub(crate) const LIB_AGC: &str = "libSceAgc";
pub(crate) const LIB_AGC_DRIVER: &str = "libSceAgcDriver";

pub(crate) const OK: HleResult = 0;
pub(crate) const ERR_NOT_FOUND: HleResult = 0x8002_0002u32 as i32 as i64; // ENOENT
pub(crate) const ERR_BAD_FILE: HleResult = 0x8002_0009u32 as i32 as i64; // EBADF
/// Kernel-style errors: 0x80020000 | errno.
pub(crate) const ERR_INVALID_ARGUMENT: HleResult = 0x8002_0016u32 as i32 as i64; // EINVAL
pub(crate) const ERR_ALREADY_EXISTS: HleResult = 0x8002_0011u32 as i32 as i64; // EEXIST
pub(crate) const ERR_MEMORY_FAULT: HleResult = 0x8002_000Eu32 as i32 as i64; // EFAULT
/// Shader register table lacks the program address registers.
pub(crate) const ERR_INCOMPLETE_SHADER_REGISTERS: HleResult = 0x8A6C_0005u32 as i32 as i64;
/// Graphics driver "invalid value" / "invalid argument".
/// SCE_AGC_ERROR_INVALID_PACKET: a patch was handed the wrong packet type.
pub(crate) const ERR_INVALID_PACKET: HleResult = 0x8A6C_000Cu32 as i32 as i64;
pub(crate) const ERR_DRIVER_INVALID_VALUE: HleResult = 0x8A6C_0033u32 as i32 as i64;
pub(crate) const ERR_DRIVER_INVALID_ARGUMENT: HleResult = 0x8A6C_0035u32 as i32 as i64;

// ---------------------------------------------------------------------------
// VideoOut flip hooks
// ---------------------------------------------------------------------------

/// Submit a GPU flip: `(handle, buffer_index, flip_mode, flip_arg)`.
pub type FlipFn = fn(i32, i32, u32, i64) -> Result<(), i64>;
/// Whether the presenter is done with `(handle, buffer_index)`.
pub type FlipDoneFn = fn(i32, i32) -> bool;

/// Explicitly installed hooks take precedence over the VideoOut defaults
/// that `register()` installs (re-registering never replaces them).
static FLIP_HOOKS: RwLock<Option<(FlipFn, FlipDoneFn)>> = RwLock::new(None);
static DEFAULT_FLIP_HOOKS: RwLock<Option<(FlipFn, FlipDoneFn)>> = RwLock::new(None);

/// Install GPU-flip entry points, overriding the VideoOut defaults.
pub fn set_flip_hooks(flip: FlipFn, done: FlipDoneFn) {
    *FLIP_HOOKS.write().unwrap() = Some((flip, done));
}

/// Remove explicitly installed hooks (the defaults remain).
pub fn clear_flip_hooks() {
    *FLIP_HOOKS.write().unwrap() = None;
}

fn flip_hooks() -> Option<(FlipFn, FlipDoneFn)> {
    (*FLIP_HOOKS.read().unwrap()).or(*DEFAULT_FLIP_HOOKS.read().unwrap())
}

/// Route a flip packet to VideoOut. Without hooks the flip is dropped.
pub(crate) fn hook_gpu_flip(handle: i32, index: i32, mode: u32, arg: i64) -> Result<(), i64> {
    let hooks = flip_hooks();
    match hooks {
        Some((flip, _)) => flip(handle, index, mode, arg),
        None => {
            log_once("agc.flip_no_hook", || {
                log::warn!("[agc] GPU flip (handle={handle} index={index}) with no VideoOut hook registered");
            });
            Ok(())
        }
    }
}

/// Ask VideoOut whether a display buffer may be rendered to again. Without
/// hooks every buffer is free.
pub(crate) fn hook_is_flip_done(handle: i32, index: i32) -> bool {
    let hooks = flip_hooks();
    hooks.map_or(true, |(_, done)| done(handle, index))
}

/// Claim the flip `sceVideoOutSubmitEopFlip` reserved under a token (an
/// end-of-pipe RELEASE_MEM's context id); single use.
pub type EopClaimFn = fn(u32) -> Option<u64>;
/// Perform a claimed EOP flip.
pub type EopFlipFn = fn(u64) -> Result<(), i64>;

/// Explicitly installed EOP flip hooks (VideoOut's otherwise).
static EOP_FLIP_HOOKS: RwLock<Option<(EopClaimFn, EopFlipFn)>> = RwLock::new(None);

/// Install EOP flip entry points, overriding VideoOut's (None restores
/// them).
pub fn set_eop_flip_hooks(hooks: Option<(EopClaimFn, EopFlipFn)>) {
    *EOP_FLIP_HOOKS.write().unwrap() = hooks;
}

fn eop_flip_hooks() -> (EopClaimFn, EopFlipFn) {
    (*EOP_FLIP_HOOKS.read().unwrap()).unwrap_or((crate::videoout::claim_eop_flip, crate::videoout::eop_flip))
}

pub(crate) fn hook_claim_eop_flip(token: u32) -> Option<u64> {
    (eop_flip_hooks().0)(token)
}

pub(crate) fn hook_eop_flip(id: u64) -> Result<(), i64> {
    (eop_flip_hooks().1)(id)
}

// ---------------------------------------------------------------------------
// GPU back-end
// ---------------------------------------------------------------------------

static RENDER_EXECUTOR: RwLock<Option<std::sync::Arc<dyn ps5gpu::render::RenderExecutor>>> = RwLock::new(None);

/// Install the render executor new PM4 interpreters hand draws to.
pub fn set_render_executor(ex: Option<std::sync::Arc<dyn ps5gpu::render::RenderExecutor>>) {
    *RENDER_EXECUTOR.write().unwrap() = ex;
}

/// The executor new interpreters use (None: draws are only counted).
pub fn render_executor() -> Option<std::sync::Arc<dyn ps5gpu::render::RenderExecutor>> {
    RENDER_EXECUTOR.read().unwrap().clone()
}

/// Block until every command stream submitted so far has executed (VideoOut
/// Close/UnregisterBuffers). Bounded so a stream parked on a wait that only
/// the caller could satisfy cannot hang it forever.
pub fn drain_gpu() {
    pm4::queue::global().wait_idle(std::time::Duration::from_secs(2));
    if let Some(ex) = render_executor() {
        ex.drain();
    }
}

/// Run `f` the first time `key` is seen (warning de-duplication).
pub(crate) fn log_once(key: &str, f: impl FnOnce()) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap();
    if g.get_or_insert_with(HashSet::new).insert(key.to_string()) {
        drop(g);
        f();
    }
}

// ---------------------------------------------------------------------------
// Argument helpers
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn arg(ctx: &CpuContext, i: usize) -> u64 {
    const REGS: [CpuRegister; 6] =
        [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx, CpuRegister::R8, CpuRegister::R9];
    ctx.get(REGS[i])
}

/// The `n`-th stack argument (1-based: `[rsp + 8*n]`, past the return
/// address).
#[inline]
pub(crate) fn stack_arg(ctx: &CpuContext, n: u64) -> Option<u64> {
    mem::r64(ctx.get(CpuRegister::Rsp).wrapping_add(8 * n))
}

#[inline]
pub(crate) fn ptr(addr: u64) -> HleResult {
    addr as i64
}

#[inline]
pub(crate) fn size_bytes(dwords: u32) -> HleResult {
    (dwords as i64) * 4
}

// ---------------------------------------------------------------------------
// Command-buffer allocation
// ---------------------------------------------------------------------------

pub(crate) const CB_CURSOR_UP: u64 = 0x10;
pub(crate) const CB_CURSOR_DOWN: u64 = 0x18;
pub(crate) const CB_CALLBACK: u64 = 0x20;
pub(crate) const CB_USER_DATA: u64 = 0x28;
pub(crate) const CB_RESERVED_DW: u64 = 0x30;

/// Calls a guest function from the current HLE call. Replaceable in tests.
pub(crate) type GuestCallFn = fn(u64, [u64; 6]) -> u64;

fn default_guest_call(func: u64, args: [u64; 6]) -> u64 {
    ps5cpu::exec::call_guest_nested(func, args)
}

static GUEST_CALL: RwLock<GuestCallFn> = RwLock::new(default_guest_call);

#[cfg(test)]
pub(crate) fn set_guest_call(f: GuestCallFn) {
    *GUEST_CALL.write().unwrap() = f;
}

/// Free dwords between the cursors minus the reserved tail. The library
/// computes `(cursorDown - cursorUp) >> 2` in unsigned 64-bit arithmetic, so
/// a buffer whose cursorDown is 0 (titles set it so for a buffer they sized
/// beforehand) never reports itself full.
fn remaining_dwords(up: u64, down: u64, reserved: u32) -> u64 {
    (down.wrapping_sub(up) >> 2).saturating_sub(reserved as u64)
}

/// Reserve `n` dwords in the command buffer at `cb`; with `advance` the
/// write cursor moves past them. Calls the guest's buffer-full callback when
/// the free space is too small. Returns the address of the first dword.
pub(crate) fn prepare_dwords(cb: u64, n: u32, advance: bool) -> Option<u64> {
    if cb == 0 || n == 0 {
        return None;
    }
    // The whole header in one checked read.
    let mut header = [0u8; (CB_RESERVED_DW - CB_CURSOR_UP + 4) as usize];
    if !mem::read(cb + CB_CURSOR_UP, &mut header) {
        return None;
    }
    let field = |off: u64| u64::from_le_bytes(header[(off - CB_CURSOR_UP) as usize..][..8].try_into().unwrap());
    let mut up = field(CB_CURSOR_UP);
    let mut down = field(CB_CURSOR_DOWN);
    let callback = field(CB_CALLBACK);
    let user = field(CB_USER_DATA);
    let mut reserved = u32::from_le_bytes(header[(CB_RESERVED_DW - CB_CURSOR_UP) as usize..][..4].try_into().unwrap());
    if n as u64 > remaining_dwords(up, down, reserved) {
        if callback == 0 {
            log::warn!("[agc] command buffer 0x{cb:x} full (need {n} dwords) and no callback");
            return None;
        }
        log::debug!("[agc] cb 0x{cb:x} full: need {n}, up 0x{up:x} down 0x{down:x} reserved {reserved}; callback 0x{callback:x} user 0x{user:x}");
        let call = *GUEST_CALL.read().unwrap();
        let r = call(callback, [cb, n as u64 + reserved as u64, user, 0, 0, 0]);
        log::debug!("[agc] cb 0x{cb:x} callback returned 0x{r:x}");
        up = mem::r64(cb + CB_CURSOR_UP)?;
        down = mem::r64(cb + CB_CURSOR_DOWN)?;
        reserved = mem::r32(cb + CB_RESERVED_DW)?;
        if n as u64 > remaining_dwords(up, down, reserved) {
            log::warn!("[agc] command buffer 0x{cb:x} still full after callback (need {n}, result 0x{r:x})");
            return None;
        }
    }
    if advance && !mem::w64(cb + CB_CURSOR_UP, up + n as u64 * 4) {
        return None;
    }
    Some(up)
}

/// Allocate `n` dwords and advance the cursor.
#[inline]
pub(crate) fn alloc_dwords(cb: u64, n: u32) -> Option<u64> {
    prepare_dwords(cb, n, true)
}

/// Allocate `words.len()` dwords and write them. Returns the packet address
/// or 0.
pub(crate) fn emit(cb: u64, words: &[u32]) -> u64 {
    match alloc_dwords(cb, words.len() as u32) {
        Some(a) if mem::write_dwords(a, words) => a,
        _ => 0,
    }
}

/// Type-3 header (see [`pm4::make`]).
#[inline]
pub(crate) const fn pm4(dwords: u32, op: u32, custom: u32) -> u32 {
    pm4::make(dwords, op, custom)
}

#[inline]
pub(crate) const fn lo(v: u64) -> u32 {
    v as u32
}

#[inline]
pub(crate) const fn hi(v: u64) -> u32 {
    (v >> 32) as u32
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// One export: (NID, name, library, callback).
pub(crate) type Export = (&'static str, &'static str, &'static str, crate::registry::HleCallback);

/// NIDs whose catalogue name is a working label rather than the real symbol
/// name (the NID was taken from shipped titles), so `compute_nid(name)` does
/// not reproduce them.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const LABELED_NIDS: &[&str] = &[
    "V++UgBtQhn0", // GetDataPacketPayloadAddress
    "23LRUSvYu1M", // Init
    "HV4j+E0MBHE", // CreateInterpolantMapping
    "dbOlWdppb4o", // CreateInterpolantMapping2
    "dolOmWH+huQ", // GetFusedShaderSize
    "fd5Bp5tGTgo", // FuseShaderHalves (newer form)
    "qj7QZpgr9Uw", // context-state operation builder
    "Ikfdt-rIqCE", // jump-packet patch
    "-KRzWekV120", // set-index-size with instance id support
    "gQkqkLttcpw", // AcbAtomicGds_0900
];

/// Every export of both libraries.
pub(crate) fn all_exports() -> Vec<Export> {
    let mut v = Vec::new();
    v.extend_from_slice(builders::EXPORTS);
    v.extend_from_slice(patch::EXPORTS);
    v.extend_from_slice(shader::EXPORTS);
    v.extend_from_slice(defaults::EXPORTS);
    v.extend_from_slice(driver::EXPORTS);
    v
}

pub fn register(reg: &mut HleRegistry) {
    *DEFAULT_FLIP_HOOKS.write().unwrap() = Some((crate::videoout::gpu_flip, crate::videoout::is_flip_done));
    // Draws, dispatches and end-of-pipe interrupts go to the GPU back-end
    // (initialised on first use; RPS5E_GPU_BACKEND=0 keeps them counted only).
    set_render_executor(ps5gpu::backend::executor());
    crate::videoout::set_gpu_drain_hook(drain_gpu);
    for (nid, name, lib, cb) in all_exports() {
        reg.register_fn(nid, name, lib, cb);
    }
}
