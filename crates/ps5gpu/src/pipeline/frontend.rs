// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Shader front-end glue: guest program → decoded [`Program`] → resource
//! plan (cached per program), and per draw: materialised descriptor words,
//! the specialised resource tables, the binding layout and the push
//! ("shader") data.
//!
//! Programs are cached by entry address and re-validated against their
//! code bytes on every use, so code rewritten in place is planned again.
//! The specialised tables and the binding layout of a stage are cached per
//! specialization (the materialisation cache hands out the same
//! specialization while only addresses change), so a draw does not hash or
//! lay out its resources again.

use crate::memory::deps::PageDeps;
use crate::memory::GuestGpuMemory;
use crate::memory::tracker::GuestPageTracker;
use crate::scheduler::Scheduler;
use ps5shader::resources::{
    apply_specialization, BindingLayout, MaterializationError, ResourceMaterializationCache, ResourceRuntimeInputs, ResourceSnapshot,
    ResourceSpecialization, ShaderCompileRequest, ShaderResourcePlan, ShaderStage, SpecializedResourceInfo,
};
use ps5shader::{decode_program, Program};
use std::cell::RefCell;
use crate::fxhash::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// FNV-1a over bytes.
pub fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Hash of any hashable value (std SipHash, deterministic per process).
pub fn hash_of<T: Hash + ?Sized>(v: &T) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

/// A decoded and planned guest program.
pub struct StageProgram {
    pub address: u64,
    pub stage: ShaderStage,
    /// Hash of the code bytes.
    pub code_hash: u64,
    pub code_len: u64,
    /// The code bytes the plan was built from.
    pub code: Vec<u8>,
    pub user_data_base: u32,
    pub user_data_count: u32,
    pub wave_size: u32,
    pub plan: ShaderResourcePlan,
    pub user_data_registers: Vec<u32>,
    pub uses_global_data_share: bool,
    /// The program addresses LDS (see `BindingLayout::uses_local_data_share`).
    pub uses_local_data_share: bool,
    pub reads_shader_base: bool,
    /// The program stores to no memory through buffers or addresses (its
    /// buffers may be read in guest memory).
    pub reads_memory_only: bool,
    /// The translated program does not depend on where the code lives: no
    /// PC reads or absolute jumps (`s_getpc` / `s_setpc` / `s_swappc`) and
    /// no split halves. Module keys then leave the address out, so the
    /// same code loaded at another address (in this run or a later one)
    /// reuses its modules.
    pub address_independent: bool,
    /// The program stores or performs atomics through device addresses (a
    /// V# held only in registers, flat/global accesses): writes the
    /// scheduler cannot see as ranges.
    pub writes_device_memory: bool,
    /// Parameters the program exports (bit i: EXP target PARAM i).
    pub param_exports: u64,
    /// User-data registers holding the code addresses of the callees or
    /// the continuation linked into this program (empty: nothing to link;
    /// see [`ShaderFrontend::program_for`]).
    pub link_registers: Vec<u32>,
    /// The back half the program was decoded with (0: none).
    pub continuation: u64,
    /// Every code word the decode read (the linked code of a linked
    /// program): with the plan inputs above, what translates the program
    /// again ([`super::sources`]).
    pub image: Arc<super::sources::CodeImage>,
    /// Planned with the guest-memory plan options (a stage reading its
    /// buffers in guest memory).
    pub guest_plan: bool,
}

/// Decode the program at `address` from `code`: a front half that ends in
/// s_setpc_b64 to its back half at `continuation` decodes as one stream.
pub fn decode_code<C: ps5shader::CodeSource + ?Sized>(code: &C, address: u64, continuation: u64) -> Result<Program, ps5shader::DecodeError> {
    let fused = (continuation != 0)
        .then(|| {
            // The sizes only bound the walk.
            let size = ps5shader::MAX_DECLARED_SHADER_SIZE;
            let layout = ps5shader::FusedLayout { entry_address: address, entry_size: size, continuation_address: continuation, continuation_size: size };
            ps5shader::decode_fused_program(code, layout).ok()
        })
        .flatten();
    match fused {
        Some(p) => Ok(p),
        None => decode_program(code, address),
    }
}

/// The parameters a program exports: bit i for every EXP to PARAM i
/// (targets 32..63).
pub fn param_exports(program: &ps5shader::Program) -> u64 {
    program.exports().filter(|e| (32..64).contains(&e.target)).fold(0, |m, e| m | 1 << (e.target - 32))
}


/// Identity of a program lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProgramKey {
    pub address: u64,
    pub stage_tag: u8,
    pub user_data_base: u32,
    pub user_data_count: u32,
    pub wave_size: u32,
    /// Back half of a split geometry program (0: none).
    pub continuation: u64,
}

fn stage_tag(s: ShaderStage) -> u8 {
    match s {
        ShaderStage::Unknown => 0,
        ShaderStage::Vertex => 1,
        ShaderStage::Pixel => 2,
        ShaderStage::Compute => 3,
    }
}

/// Why a stage could not be prepared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrontendError {
    /// The code could not be read or decoded.
    Decode(String),
    /// The resource planner rejected the program.
    Plan(String),
    /// The draw's descriptors could not be materialised.
    Materialize(String),
    /// The binding layout could not be allocated.
    Layout(String),
}

impl std::fmt::Display for FrontendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrontendError::Decode(m) => write!(f, "decode: {m}"),
            FrontendError::Plan(m) => write!(f, "plan: {m}"),
            FrontendError::Materialize(m) => write!(f, "materialize: {m}"),
            FrontendError::Layout(m) => write!(f, "layout: {m}"),
        }
    }
}

/// Push ("shader") data of a stage, inline for the common sizes.
pub type ShaderData = smallvec::SmallVec<[u32; 48]>;

/// One stage prepared for one draw.
#[derive(Clone)]
pub struct PreparedStage {
    pub program: Arc<StageProgram>,
    pub snapshot: Arc<ResourceSnapshot>,
    pub specialization: Arc<ResourceSpecialization>,
    pub resources: Arc<SpecializedResourceInfo>,
    pub layout: Arc<BindingLayout>,
    /// Push data (user data, shader base, memory offsets, limits) laid out
    /// per `layout`; the memory offsets are filled when buffers are bound.
    pub shader_data: ShaderData,
    pub shader_base: u64,
    /// Identity of `specialization` and `layout` for module keys
    /// ([`PreparedStage::refresh_resource_key`] after changing `layout`).
    pub resource_key: u64,
}

fn resource_key(specialization: &ResourceSpecialization, layout: &BindingLayout) -> u64 {
    // Every field the layout's equality compares (what `describe` shows).
    let l = layout;
    hash_of(&(
        hash_of(specialization),
        (l.push_data_start_dword, l.shader_base_dword, l.memory_offset_dword, l.memory_offset_count, l.memory_offset_dwords(), l.wide_offsets, l.uses_dispatch_thread_limits),
        &l.user_data_registers,
        &l.descriptors,
    ))
}

impl PreparedStage {
    /// Recompute [`PreparedStage::resource_key`] after editing the layout.
    pub fn refresh_resource_key(&mut self) {
        self.resource_key = resource_key(&self.specialization, &self.layout);
    }

    /// A compile request for this stage (the caller fills stage options).
    pub fn request(&self) -> ShaderCompileRequest<'_> {
        let mut r = ShaderCompileRequest::new(&self.program.plan, (*self.resources).clone(), (*self.layout).clone());
        r.wave_size = self.program.wave_size;
        r
    }
}

/// Front-end counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrontendStats {
    pub programs: usize,
    pub plan_failures: u64,
    pub decode_failures: u64,
    pub materialize_failures: u64,
    pub revalidations_changed: u64,
    /// Stages prepared again under an unchanged content stamp (no replay).
    pub prepared_reuses: u64,
    /// Compute stages prepared as their device-reads variant, and programs
    /// whose variant waited too (`RPS5E_DEVICE_READS`).
    pub device_reads: u64,
    pub device_reads_given_up: u64,
}

/// Identity of a stage preparation: the program (its `Arc`, kept alive by
/// the memoised stage), the push cursor and the user data.
/// Recent preparations compared per draw.
const RECENT_PREPARED: usize = 8;

#[derive(Clone, PartialEq, Eq, Hash)]
struct PreparedKey {
    program: usize,
    push_cursor: u32,
    thread_limits: Option<[u32; 3]>,
    bda: bool,
    user_data: smallvec::SmallVec<[u32; 40]>,
}

/// `RPS5E_PREPARE_PAGES=0`: a memoised stage is reused only while
/// nothing in guest memory may have changed (the buffer cache's content
/// stamp), instead of while the pages its materialisation read are
/// unchanged ([`PageDeps`]).
pub(crate) fn prepare_pages_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_PREPARE_PAGES"))
}

/// `RPS5E_PREFETCH_MEMO=0`: the interpreter thread's prefetch
/// (`RPS5E_GPU_RECORD_THREAD=1`) materialises every stage again instead
/// of reusing a recent one whose pages are unchanged (A/B scope release
/// `prefetch-memo`). Without it the interpreter paid a full
/// materialisation for every stage the recorder's memo would have reused,
/// and became the slower end of the pipeline in light scenes.
fn prefetch_memo_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_PREFETCH_MEMO")) && crate::backend::ab::release_fast("prefetch-memo")
}

/// `RPS5E_CONTENT_CHECKS=0`: a memoised or prefetched stage whose pages
/// changed is always materialised again. By default the words its
/// materialisation read are compared first ([`PageDeps::revalidate`]): the
/// same words give the same result, so the stage stands (A/B scope
/// release `content-checks`). The prefetch lags the recorder by up to a
/// queue of draws; pages it read are often rewritten with the same words
/// or pass a synchronisation point meanwhile.
fn content_checks_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_CONTENT_CHECKS")) && crate::backend::ab::release_fast("content-checks")
}

/// Stages whose changed pages still held the words read (reused).
pub(crate) static CONTENT_REVALIDATED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Stages the interpreter thread's prefetch took from its memo.
pub(crate) static PREFETCH_MEMO_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `RPS5E_PREPARE_MEMO=0`: every draw materialises its stages again.
fn prepare_memo_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::fast() && *ON.get_or_init(|| !std::env::var("RPS5E_PREPARE_MEMO").is_ok_and(|v| v == "0"))
}

/// A specialization's tables and its layout at one push cursor.
struct LaidOut {
    specialization: Arc<ResourceSpecialization>,
    resources: Arc<SpecializedResourceInfo>,
    push_cursor: u32,
    thread_limits: bool,
    bda: bool,
    layout: Arc<BindingLayout>,
    resource_key: u64,
}

/// Layouts kept per plan, most recently used first (with content
/// interning; 8 without).
const LAYOUTS_PER_PLAN: usize = 32;

/// `RPS5E_LAYOUT_INTERN=0`: a specialization handed out again (a new
/// `Arc` of equal contents, after the materialisation cache dropped its
/// template) gets a new binding layout (draw recipes and pipeline layouts

/// The room a deferred NUM_RECORDS binding covers at most (bytes).
const MAX_RECORDS_ROOM: u64 = 1 << 30;
/// ... and at least, when the mirrors have it.
const RECORDS_ROOM_MIN: u64 = 64 << 10;

/// `RPS5E_DEFER_RECORDS`: a buffer whose V# NUM_RECORDS is a word GPU work
/// is still writing, and which the program reads itself, is bound over
/// the mirror holding its base instead of waiting for the GPU to read the
/// word on the CPU (`1`: on; default off while measured; `check`: on,
/// and the word is also read on the CPU to count records past the
/// binding, `[gpu]` line).
pub(crate) fn defer_records() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let v = std::env::var("RPS5E_DEFER_RECORDS").unwrap_or_default();
        if v == "check" {
            ps5shader::resources::RECORDS_CHECK.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        !crate::backend::ab::safe() && (v == "1" || v == "check")
    })
}

/// keyed by it miss), and eight layouts are kept per plan.
/// `RPS5E_DESC_WAIT_WHY=1`: the first waits of 2 ms or more for a
/// GPU-written word a materialisation reads are logged with the program,
/// the word's address and value and (with `RPS5E_MEM_HISTORY=1`) the
/// GPU history of its page.
fn note_slow_word_wait(program: u64, addr: u64, waited: std::time::Duration) {
    static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if waited.as_micros() < 2000 || !crate::backend::profile::desc_wait_why() || LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 400 {
        return;
    }
    log::info!("ps5gpu: desc-wait p{program:#x} word {addr:#x} waited {} us {}", waited.as_micros(), crate::memory::history::describe(addr, 8).unwrap_or_default());
}

/// `RPS5E_WORD_WAIT_LOG=<seconds>` (diagnostics): the first GPU-written words a
/// materialisation waits for are logged with the exact write record and
/// the page's history (`RPS5E_MEM_HISTORY=1`).
/// `=<seconds>`: from that many seconds after the first wait on.
fn word_wait_log() -> bool {
    static ON: std::sync::OnceLock<Option<(std::time::Instant, u64)>> = std::sync::OnceLock::new();
    ON.get_or_init(|| std::env::var("RPS5E_WORD_WAIT_LOG").ok().and_then(|v| v.parse().ok()).map(|s| (std::time::Instant::now(), s)))
        .is_some_and(|(t0, s)| t0.elapsed().as_secs() >= s)
}

/// `RPS5E_DEVICE_READS=1`: a compute program whose materialisation waited
/// for a GPU-written word reads its buffers through the V#s in its
/// registers from then on (see [`ShaderFrontend::device_reads_variant`]).
/// Off by default: in GTA V it removes ~95 waits per frame but each such
/// dispatch then reads unranged memory (mirror synchronisation, a barrier
/// against every unordered shader write), which costs as much (A/B in one
/// run with `RPS5E_AB_SCOPE=release RPS5E_AB_ONLY=device-reads`).
pub(crate) fn device_reads_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::backend::ab::safe() && std::env::var("RPS5E_DEVICE_READS").is_ok_and(|v| v == "1")) && crate::backend::ab::release_fast("device-reads")
}

pub(crate) fn layout_intern_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_LAYOUT_INTERN"))
}

/// The guest memory of the live provider and its storage-buffer binding
/// limit: the interpreter thread's own front end instance (below) and its
/// clamp caches are built from them. Registered at the provider's
/// creation; off-line tools with several providers never prefetch (the
/// queued recording runs with the one live back-end only).
static PREFETCH_MEMORY: std::sync::OnceLock<&'static GuestGpuMemory> = std::sync::OnceLock::new();
static PREFETCH_LIMIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1 << 30);

/// Remember the provider's guest memory and binding limit for the prefetch path.
pub(crate) fn set_prefetch_memory(memory: &'static GuestGpuMemory, binding_limit: u64) {
    let _ = PREFETCH_MEMORY.set(memory);
    PREFETCH_LIMIT.store(binding_limit, std::sync::atomic::Ordering::Relaxed);
}

thread_local! {
    /// The interpreter thread's clamp caches (the provider holds its own;
    /// see `ClampCaches`).
    static CLAMP: std::cell::RefCell<crate::pipeline::provider::ClampCaches> = std::cell::RefCell::new(crate::pipeline::provider::ClampCaches::default());
}

/// The BDA decision of the recorder's front end (0 unknown, 1 plain, 2
/// BDA), mirrored for the prefetch instance whose own `bda_checked` no
/// recorder prepare ever sets.
static PREFETCH_BDA: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Publish the recorder's BDA decision.
pub(crate) fn note_bda(bda: bool) {
    PREFETCH_BDA.store(if bda { 2 } else { 1 }, std::sync::atomic::Ordering::Release);
}

thread_local! {
    /// The interpreter thread's own front end: its caches
    /// are private, so the interpreter's materialisation never holds the
    /// provider's front end lock the recording thread waits for.
    static PREFETCH_FRONTEND: std::cell::RefCell<Option<Box<ShaderFrontend>>> = const { std::cell::RefCell::new(None) };
}

/// Run `f` on this thread's prefetch front end (None before the provider
/// registered its memory).
fn with_prefetch<R>(f: impl FnOnce(&mut ShaderFrontend) -> Option<R>) -> Option<R> {
    let memory = *PREFETCH_MEMORY.get()?;
    PREFETCH_FRONTEND.with(|slot| {
        let mut slot = slot.borrow_mut();
        let front = slot.get_or_insert_with(|| Box::new(ShaderFrontend::new(memory)));
        f(front)
    })
}

/// The page tracker of the live back-end (set at its creation): the
/// interpreter-side prefetch checks page state without the back-end lock.
static TRACKER: std::sync::OnceLock<std::sync::Arc<GuestPageTracker>> = std::sync::OnceLock::new();

/// Remember the back-end's tracker for the prefetch path.
pub(crate) fn set_tracker(tracker: std::sync::Arc<GuestPageTracker>) {
    let _ = TRACKER.set(tracker);
}

/// `RPS5E_GPU_PREFETCH=0`: the interpreter thread does not
/// pre-materialise the stages of the draws it hands to the recording
/// thread; the recorder prepares them itself.
fn prefetch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| !std::env::var("RPS5E_GPU_PREFETCH").is_ok_and(|v| v == "0"))
}

/// `RPS5E_WIDE_OFFSETS=1`: every buffer's byte offset takes a full
/// shader-data dword instead of four 8-bit slots. The
/// substrate for binding whole ring buffers with the rotating base
/// patched into the offset; today's bindings still bind the exact range,
/// so the flag only widens the slot (and re-keys the layout caches).
fn wide_offsets_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| std::env::var("RPS5E_WIDE_OFFSETS").is_ok_and(|v| v == "1"))
}

/// `RPS5E_PREFETCH_RANGES=1`: the interpreter thread also clamps the
/// draw's stage ranges. Measured slower live: the range
/// work on the interpreter lengthens the queue's fill while the recorder
/// waits at synchronous calls (flips drain the whole queue), so the heavy
/// SH phase lost ~half its fps at every queue depth tried (64, 2048,
/// 8192). Off by default; kept as the substrate for true slice threads,
/// where a slice owns its ranges end to end.
fn prefetch_ranges_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| std::env::var("RPS5E_PREFETCH_RANGES").is_ok_and(|v| v == "1"))
}

/// One stage of a draw pre-materialised on the interpreter thread: the
/// recording thread validates it against the program identity (by value:
/// the two front ends are separate instances), the user data and the
/// pages the materialisation read (a mismatch or a dirty page re-prepares
/// on the recorder).
pub(crate) struct PrefetchedStage {
    /// The program identity the stage was prepared from.
    address: u64,
    code_hash: u64,
    user_data_base: u32,
    user_data_count: u32,
    wave_size: u32,
    user: smallvec::SmallVec<[u32; 40]>,
    cursor: u32,
    /// The guest pages the materialisation read (validated on take).
    deps: PageDeps,
    stage: PreparedStage,
}

/// The stages of one draw pre-materialised on the interpreter thread.
pub(crate) struct PrefetchedDraw {
    vs: Option<PrefetchedStage>,
    ps: Option<PrefetchedStage>,
}

impl PrefetchedStage {
    fn matches(&self, program: &Arc<StageProgram>, user: &[u32], cursor: u32) -> bool {
        self.address == program.address
            && self.code_hash == program.code_hash
            && self.user_data_base == program.user_data_base
            && self.user_data_count == program.user_data_count
            && self.wave_size == program.wave_size
            && self.cursor == cursor
            && user.len() >= self.user.len()
            && self.user[..] == user[..self.user.len()]
    }
}

impl Default for PrefetchedDraw {
    fn default() -> Self {
        PrefetchedDraw { vs: None, ps: None }
    }
}

impl PrefetchedDraw {
    /// The prepared stage for (program, user data, push cursor), if this
    /// is the stage the recorder is about to prepare and the pages it was
    /// derived from still hold the bytes read.
    pub(crate) fn take_stage(
        &mut self,
        stage: ShaderStage,
        program: &Arc<StageProgram>,
        user: &[u32],
        cursor: u32,
        tracker: &GuestPageTracker,
        memory: &GuestGpuMemory,
    ) -> Option<PreparedStage> {
        let slot = match stage {
            ShaderStage::Vertex => &mut self.vs,
            ShaderStage::Pixel => &mut self.ps,
            _ => return None,
        };
        let Some(s) = slot.as_ref() else {
            prefetch_miss(PrefetchMiss::Absent);
            return None;
        };
        if !s.matches(program, user, cursor) {
            prefetch_miss(PrefetchMiss::Identity);
            return None;
        }
        let s = slot.as_mut()?;
        if !s.deps.current(tracker, memory) {
            if !(content_checks_enabled() && s.deps.revalidate(tracker, memory)) {
                prefetch_miss(PrefetchMiss::Stale);
                return None;
            }
            CONTENT_REVALIDATED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Some(s.stage.clone())
    }
}

/// Pre-materialise the stages of `draw` on the interpreter thread for the
/// recording thread: the plain vertex/pixel path only —
/// geometry subgroups (mesh), indirect probes and traced or captured
/// frames prepare on the recorder. A GPU-written guest word aborts (the
/// recorder's full prepare downloads it).
pub(crate) fn prefetch_draw(d: &crate::render::ResolvedDraw) -> Option<Box<PrefetchedDraw>> {
    if !prefetch_enabled() || crate::capture::active() || crate::pipeline::trace::active() {
        return None;
    }
    let tracker = TRACKER.get()?.clone();
    with_prefetch(|f| {
        let vs_stage = d.shaders.vertex_program();
        if !vs_stage.bound() || d.shaders.stages & crate::pipeline::provider::STAGES_GS_EN != 0 {
            // Geometry subgroups: the mesh path prepares (and patches) itself.
            return None;
        }
        let merged = d.shaders.es.bound() || d.shaders.gs.bound();
        let vs_count = vs_stage.user_sgpr_count().min(32);
        let vs_wave = crate::pipeline::provider::plain_vertex_wave(d.shaders.stages);
        let program = f.program_for(vs_stage.address, ShaderStage::Vertex, if merged { 8 } else { 0 }, vs_count, vs_wave, &vs_stage.user_data).ok()?;
        let (vs, vs_deps) = f.prepare_prefetch(&program, &vs_stage.user_data, 0, None, &tracker)?;
        let vs_identity = (program.address, program.code_hash, program.user_data_base, program.user_data_count, program.wave_size);
        let vs_user: smallvec::SmallVec<[u32; 40]> = vs_stage.user_data[..(vs_count as usize).min(32)].into();
        let ps_stage = &d.shaders.ps;
        let mut ps = None;
        let ps_active = ps_stage.bound() && (!d.color_targets.is_empty() || d.pixel_side_effects());
        if ps_active {
            let count = ps_stage.user_sgpr_count().min(32);
            let program = f.program_for(ps_stage.address, ShaderStage::Pixel, 0, count, pixel_wave_size(d.ps_in_control), &ps_stage.user_data).ok()?;
            let mut cursor = 0;
            vs.layout.advance_push_data(&mut cursor);
            if let Some((p, deps)) = f.prepare_prefetch(&program, &ps_stage.user_data, cursor, None, &tracker) {
                ps = Some(PrefetchedStage {
                    address: program.address,
                    code_hash: program.code_hash,
                    user_data_base: program.user_data_base,
                    user_data_count: program.user_data_count,
                    wave_size: program.wave_size,
                    user: ps_stage.user_data[..(count as usize).min(32)].into(),
                    cursor,
                    deps,
                    stage: p,
                });
            }
        }
        // `RPS5E_PREFETCH_RANGES=1`: the interpreter thread
        // also clamps the draw's stage ranges on its own caches — measured
        // slower live (the queue's end-to-end latency), so off by default;
        // the shared functions stay as the slice-thread substrate. The
        // result warms the caches for the recorder's own pass regardless.
        if prefetch_ranges_enabled() {
            let memory = *PREFETCH_MEMORY.get()?;
            let limit = PREFETCH_LIMIT.load(std::sync::atomic::Ordering::Relaxed);
            CLAMP.with(|c| {
                let mut caches = c.borrow_mut();
                let mut out: crate::pipeline::provider::RangeList = Default::default();
                let mut maps: [smallvec::SmallVec<[u32; 16]>; 2] = Default::default();
                crate::pipeline::provider::stage_ranges_shared(memory, &mut caches, None, limit, &vs, &mut out, Some(&mut maps[0]));
                if let Some(p) = &ps {
                    crate::pipeline::provider::stage_ranges_shared(memory, &mut caches, None, limit, &p.stage, &mut out, Some(&mut maps[1]));
                }
            });
        }
        let (address, code_hash, user_data_base, user_data_count, wave_size) = vs_identity;
        Some(Box::new(PrefetchedDraw {
            vs: Some(PrefetchedStage { address, code_hash, user_data_base, user_data_count, wave_size, user: vs_user, cursor: 0, deps: vs_deps, stage: vs }),
            ps,
        }))
    })
}

/// A stage of a compute dispatch pre-materialised the same way (the
/// interpreter thread, before the hand-over). An empty one (no stage)
/// stands for "no prefetch".
pub(crate) struct PrefetchedDispatch {
    stage: Option<PreparedStage>,
    address: u64,
    code_hash: u64,
    user_data_base: u32,
    user_data_count: u32,
    wave_size: u32,
    user: smallvec::SmallVec<[u32; 40]>,
    cursor: u32,
    thread_limits: Option<[u32; 3]>,
    deps: PageDeps,
}

impl Default for PrefetchedDispatch {
    fn default() -> Self {
        PrefetchedDispatch { stage: None, address: 0, code_hash: 0, user_data_base: 0, user_data_count: 0, wave_size: 0, user: Default::default(), cursor: 0, thread_limits: None, deps: PageDeps::default() }
    }
}

impl PrefetchedDispatch {
    pub(crate) fn take(&mut self, program: &Arc<StageProgram>, user: &[u32], cursor: u32, thread_limits: Option<[u32; 3]>, tracker: &GuestPageTracker, memory: &GuestGpuMemory) -> Option<PreparedStage> {
        let same = self.address == program.address
            && self.code_hash == program.code_hash
            && self.user_data_base == program.user_data_base
            && self.user_data_count == program.user_data_count
            && self.wave_size == program.wave_size
            && self.cursor == cursor
            && self.thread_limits == thread_limits
            && user.len() >= self.user.len()
            && self.user[..] == user[..self.user.len()];
        if self.stage.is_none() {
            prefetch_miss(PrefetchMiss::AbsentDispatch);
            return None;
        }
        if !same {
            prefetch_miss(PrefetchMiss::Identity);
            return None;
        }
        if !self.deps.current(tracker, memory) {
            if !(content_checks_enabled() && self.deps.revalidate(tracker, memory)) {
                prefetch_miss(PrefetchMiss::Stale);
                return None;
            }
            CONTENT_REVALIDATED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.stage.take()
    }
}

/// [`prefetch_draw`] for a compute dispatch.
pub(crate) fn prefetch_dispatch(d: &crate::render::ResolvedDispatch) -> Option<Box<PrefetchedDispatch>> {
    if !prefetch_enabled() || crate::capture::active() || crate::pipeline::trace::active() {
        return None;
    }
    let tracker = TRACKER.get()?.clone();
    with_prefetch(|f| {
        let wave = if d.packet.dispatch_initiator & (1 << 15) != 0 { 32 } else { 64 };
        let count = d.shader.user_sgpr_count().min(16);
        let (cursor, thread_limits) = crate::pipeline::provider::dispatch_layout(d);
        let program = f.program_for(d.shader.address, ShaderStage::Compute, 0, count, wave, &d.shader.user_data).ok()?;
        let (stage, deps) = f.prepare_prefetch(&program, &d.shader.user_data, cursor, thread_limits, &tracker)?;
        Some(Box::new(PrefetchedDispatch {
            stage: Some(stage),
            address: program.address,
            code_hash: program.code_hash,
            user_data_base: program.user_data_base,
            user_data_count: program.user_data_count,
            wave_size: program.wave_size,
            user: d.shader.user_data[..(count as usize).min(d.shader.user_data.len())].into(),
            cursor,
            thread_limits,
            deps,
        }))
    })
}

/// Why a stage the recorder prepares was not taken from the prefetch
/// (diagnostics, the `prefetch-miss` field of the pipelines line): none
/// was made (the interpreter's prepare aborted on a GPU-written word or
/// the path does not prefetch), it was made for other inputs, or a page
/// it read changed since.
#[derive(Clone, Copy)]
pub(crate) enum PrefetchMiss {
    Absent = 0,
    Identity = 1,
    Stale = 2,
    AbsentDispatch = 3,
}

pub(crate) static PREFETCH_MISSES: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

fn prefetch_miss(why: PrefetchMiss) {
    PREFETCH_MISSES[why as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// The pre-materialised stage a `prepare_stage` call is about to compute.
pub(crate) enum Prefetched<'a> {
    Draw(&'a mut PrefetchedDraw),
    Dispatch(&'a mut PrefetchedDispatch),
    None,
}

impl Prefetched<'_> {
    pub(crate) fn take(
        &mut self,
        stage: ShaderStage,
        program: &Arc<StageProgram>,
        user: &[u32],
        cursor: u32,
        thread_limits: Option<[u32; 3]>,
        tracker: &GuestPageTracker,
        memory: &GuestGpuMemory,
    ) -> Option<PreparedStage> {
        match self {
            Prefetched::Draw(d) if stage == ShaderStage::Vertex || stage == ShaderStage::Pixel => {
                d.take_stage(stage, program, user, cursor, tracker, memory)
            }
            Prefetched::Dispatch(d) if stage == ShaderStage::Compute => d.take(program, user, cursor, thread_limits, tracker, memory),
            _ => None,
        }
    }
}

pub struct ShaderFrontend {
    memory: &'static GuestGpuMemory,
    /// Programs with the CPU synchronisation point their code was last
    /// compared at.
    programs: HashMap<ProgramKey, (Result<Arc<StageProgram>, FrontendError>, u64)>,
    specialized: HashMap<(u64, bool), HashMap<ResourceSpecialization, Arc<SpecializedResourceInfo>>>,
    /// Read-only stages read their buffers in guest memory (`RPS5E_BDA=1`
    /// with guest memory imported; decided at the first preparation).
    bda: bool,
    bda_checked: bool,
    /// Programs planned for reading guest memory (no data reads flattened),
    /// by the address of their ordinary program (kept alive here).
    bda_programs: HashMap<usize, (Arc<StageProgram>, Result<Arc<StageProgram>, FrontendError>)>,
    /// Compute programs whose materialisation waited for a GPU-written
    /// word, by the address of their ordinary program (kept alive here),
    /// with the variant that reads its buffers through the V#s in its
    /// registers (`RPS5E_DEVICE_READS`): None once the variant waited too.
    device_reads: HashMap<usize, (Arc<StageProgram>, Option<Result<Arc<StageProgram>, FrontendError>>)>,
    /// The last materialisation waited for a GPU-written word.
    waited: bool,
    /// Programs with their callees or continuation linked, by the program
    /// they were linked from and the user-data words naming the code, with
    /// the CPU synchronisation point they were last linked at.
    linked: HashMap<(usize, Vec<u32>), LinkedEntry>,
    /// Recent layouts per plan id, found by specialization identity.
    laid_out: HashMap<u64, Vec<LaidOut>>,
    materializer: ResourceMaterializationCache,
    code_scratch: Vec<u8>,
    /// Recent stages with the buffer cache's content stamp they were
    /// prepared under and the pages their materialisation read (most
    /// recent first): the same program with the same user data under the
    /// same stamp, or while those pages are unchanged, reads the same guest
    /// words, so materialisation would give the same result.
    prepared: std::collections::VecDeque<(PreparedKey, [u64; 4], PageDeps, PreparedStage)>,
    pub stats: FrontendStats,
}

const MAX_CODE_BYTES: u64 = ps5shader::MAX_DECLARED_SHADER_SIZE as u64;

/// The wave size of a draw's pixel program: SPI_PS_IN_CONTROL.PS_W32_EN
/// (bit 15) selects wave32, otherwise the hardware runs wave64
/// (`RPS5E_PS_WAVE64=0`: every pixel program wave32, as before).
pub(crate) fn pixel_wave_size(ps_in_control: u32) -> u32 {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let wave64 = *ON.get_or_init(|| std::env::var("RPS5E_PS_WAVE64").map_or(true, |v| v != "0"));
    if !wave64 || ps_in_control & (1 << 15) != 0 {
        32
    } else {
        64
    }
}

/// A linked program, the synchronisation point it was linked at and the
/// program it was linked from (kept alive: its address is the key).
type LinkedEntry = (Result<Arc<StageProgram>, FrontendError>, u64, Arc<StageProgram>);

/// Code reached through user data is linked into the programs that call or
/// continue in it (`RPS5E_SHADER_LINK=0`: off, such programs fail
/// translation).
fn link_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_SHADER_LINK").map_or(true, |v| v != "0"))
}

/// No access of the plan stores to memory through a buffer, a global or
/// flat address or a scalar address (LDS, GDS, scratch and images aside).
fn reads_memory_only(plan: &ShaderResourcePlan) -> bool {
    use ps5shader::resources::{MemoryAccess, MemoryResourceKind as K};
    plan.memory().entries().iter().all(|m| {
        !matches!(m.access, MemoryAccess::Write | MemoryAccess::Atomic)
            || !matches!(m.kind, K::Buffer | K::Flat | K::Global | K::ScalarAddress | K::ScalarBuffer)
    })
}

impl ShaderFrontend {
    /// Materialisation cache counters.
    pub fn materialization_statistics(&self) -> ps5shader::resources::CacheStatistics {
        self.materializer.statistics()
    }

    pub fn new(memory: &'static GuestGpuMemory) -> ShaderFrontend {
        ShaderFrontend {
            memory,
            programs: HashMap::default(),
            specialized: HashMap::default(),
            laid_out: HashMap::default(),
            materializer: ResourceMaterializationCache::default(),
            code_scratch: Vec::new(),
            prepared: Default::default(),
            stats: FrontendStats::default(),
            bda: false,
            bda_checked: false,
            bda_programs: HashMap::default(),
            device_reads: HashMap::default(),
            waited: false,
            linked: HashMap::default(),
        }
    }

    /// `program` planned for a stage reading guest memory: its scalar data
    /// reads stay in the program (read through the import table) instead of
    /// being read by the host per draw.
    fn bda_program(&mut self, program: &Arc<StageProgram>) -> Result<Arc<StageProgram>, FrontendError> {
        let me = Arc::as_ptr(program) as usize;
        if let Some((_, p)) = self.bda_programs.get(&me) {
            return p.clone();
        }
        let planned = Self::plan_variant(program, crate::buffers::guest::guest_plan_options(), 0xB0A5_0000_0000_0001, true);
        if self.bda_programs.len() > 4096 {
            self.bda_programs.clear();
        }
        self.bda_programs.insert(me, (program.clone(), planned.clone()));
        planned
    }

    /// The variant of a compute program whose buffer reads take the V# from
    /// the program's registers and the data through device addresses (the
    /// mirrors and guest memory of unranged reads), so that a V# GPU work
    /// is still writing needs no CPU wait: its words only feed table slots,
    /// which the GPU fills (`RPS5E_DEVICE_READS`). Writes, atomics and
    /// formatted loads keep their bound descriptors. Made once the ordinary
    /// program's materialisation waited for a GPU-written word, and given
    /// up when the variant's waits too. None: the ordinary program.
    fn device_reads_variant(&mut self, program: &Arc<StageProgram>) -> Option<Arc<StageProgram>> {
        if program.stage != ShaderStage::Compute || program.guest_plan || !device_reads_enabled() {
            return None;
        }
        let me = Arc::as_ptr(program) as usize;
        let entry = self.device_reads.get_mut(&me)?;
        let planned = entry.1.get_or_insert_with(|| {
            let options = ps5shader::resources::PlanOptions { flatten_data_reads: true, buffers_on_device: true };
            Self::plan_variant(program, options, 0xDE71_CE00_0000_0001, false)
        });
        planned.as_ref().ok().cloned()
    }

    /// Note that the last materialisation of `program` (the ordinary
    /// program or its device-reads variant `used`) waited for a GPU-written
    /// word.
    fn note_waited(&mut self, program: &Arc<StageProgram>, used: &Arc<StageProgram>) {
        if program.stage != ShaderStage::Compute || program.guest_plan {
            return;
        }
        let me = Arc::as_ptr(program) as usize;
        if !Arc::ptr_eq(program, used) {
            // The variant waited as well: the ordinary program from now on.
            if let Some(e) = self.device_reads.get_mut(&me) {
                if !matches!(e.1, Some(Err(_))) {
                    log::info!("ps5gpu: cs {:#x}: its device-reads variant waits for GPU-written words too; bound reads again", program.address);
                    self.stats.device_reads_given_up += 1;
                }
                e.1 = Some(Err(FrontendError::Plan("device reads wait too".into())));
            }
            return;
        }
        if self.device_reads.len() > 4096 {
            self.device_reads.clear();
        }
        if !self.device_reads.contains_key(&me) && device_reads_enabled() {
            log::info!("ps5gpu: cs {:#x} waited for a GPU-written word: its buffer reads take their V#s on the device from now on", program.address);
        }
        self.device_reads.entry(me).or_insert_with(|| (program.clone(), None));
    }

    /// `program` planned again with `options` (plan hash changed by
    /// `hash_xor`).
    fn plan_variant(program: &Arc<StageProgram>, options: ps5shader::resources::PlanOptions, hash_xor: u64, guest_plan: bool) -> Result<Arc<StageProgram>, FrontendError> {
        let p = &program.plan;
        let hash = p.hash ^ hash_xor;
        ShaderResourcePlan::extract_with(p.program().clone(), program.stage, hash, program.user_data_base, program.user_data_count, None, program.wave_size, options)
            .map_err(|e| FrontendError::Plan(e.to_string()))
            .map(|plan| {
                Arc::new(StageProgram {
                    address: program.address,
                    stage: program.stage,
                    code_hash: program.code_hash,
                    code_len: program.code_len,
                    code: program.code.clone(),
                    user_data_base: program.user_data_base,
                    user_data_count: program.user_data_count,
                    wave_size: program.wave_size,
                    plan,
                    user_data_registers: program.user_data_registers.clone(),
                    uses_global_data_share: program.uses_global_data_share,
                    uses_local_data_share: program.uses_local_data_share,
                    reads_shader_base: program.reads_shader_base,
                    reads_memory_only: program.reads_memory_only,
                    address_independent: program.address_independent,
                    writes_device_memory: program.writes_device_memory,
                    param_exports: program.param_exports,
                    link_registers: program.link_registers.clone(),
                    continuation: program.continuation,
                    image: program.image.clone(),
                    guest_plan,
                })
            })
    }

    /// A stage of `program` reads its buffers in guest memory now.
    fn bda_for(&self, program: &StageProgram) -> bool {
        self.bda && program.reads_memory_only && crate::backend::ab::bda_fast()
    }

    fn read_word(&self, a: u64) -> Option<u32> {
        let mut b = [0u8; 4];
        self.memory.read_backing(a, &mut b).then(|| u32::from_le_bytes(b))
    }

    /// The code at `address` still equals `code`.
    fn code_unchanged(&mut self, address: u64, code: &[u8]) -> bool {
        self.code_scratch.resize(code.len(), 0);
        self.memory.read_backing(address, &mut self.code_scratch) && self.code_scratch[..] == *code
    }

    fn decode(&self, address: u64, continuation: u64) -> Result<(Program, u64, Vec<u8>, super::sources::CodeImage), FrontendError> {
        // Every word read is kept: the program can be decoded again from it.
        let read = std::cell::RefCell::new(Vec::new());
        let reader = |a: u64| {
            let w = self.read_word(a);
            if let Some(v) = w {
                read.borrow_mut().push((a, v));
            }
            w
        };
        let program = decode_code(&reader, address, continuation).map_err(|e| FrontendError::Decode(e.to_string()))?;
        let end = program.instructions.iter().map(|i| i.pc as u64 + i.words.len() as u64 * 4).max().unwrap_or(4);
        let len = end.clamp(4, MAX_CODE_BYTES);
        let mut code = vec![0u8; len as usize];
        if !self.memory.read_backing(address, &mut code) {
            return Err(FrontendError::Decode(format!("code {address:#x}+{len:#x} unreadable")));
        }
        Ok((program, hash_bytes(&code), code, super::sources::CodeImage::from_words(read.into_inner())))
    }

    /// The planned program at `address` (decoded and planned on first use,
    /// re-planned when its code changed).
    pub fn program(
        &mut self,
        address: u64,
        stage: ShaderStage,
        user_data_base: u32,
        user_data_count: u32,
        wave_size: u32,
    ) -> Result<Arc<StageProgram>, FrontendError> {
        self.program_with(address, 0, stage, user_data_base, user_data_count, wave_size)
    }

    /// [`Self::program`] of a program that may continue at `continuation`
    /// (the back half of split geometry halves; 0: none).
    pub fn program_with(
        &mut self,
        address: u64,
        continuation: u64,
        stage: ShaderStage,
        user_data_base: u32,
        user_data_count: u32,
        wave_size: u32,
    ) -> Result<Arc<StageProgram>, FrontendError> {
        let key = ProgramKey { address, stage_tag: stage_tag(stage), user_data_base, user_data_count, wave_size, continuation };
        let sync = crate::memory::sync_generation();
        // The CPU writes code before the synchronisation point of the
        // commands that run it: compared once per point (below).
        if let Some((Ok(p), checked)) = self.programs.get(&key) {
            if *checked == sync {
                return Ok(p.clone());
            }
        }
        if let Some((entry, checked)) = self.programs.get(&key).map(|(e, c)| (e.clone(), *c)) {
            match entry {
                Ok(p) => {
                    // The CPU writes code before the synchronisation point
                    // of the commands that run it: compare once per point.
                    if checked == sync {
                        return Ok(p.clone());
                    }
                    // While a frame capture records, the program is decoded
                    // again: the capture then holds every word a decode
                    // reads (the back half of a fused program too).
                    let unchanged = if crate::capture::active() {
                        self.decode(address, continuation).is_ok_and(|(_, _, code, _)| code == p.code)
                    } else {
                        self.code_unchanged(address, &p.code)
                    };
                    if unchanged {
                        let p = p.clone();
                        if let Some(e) = self.programs.get_mut(&key) {
                            e.1 = sync;
                        }
                        return Ok(p);
                    }
                    self.stats.revalidations_changed += 1;
                }
                Err(e) => {
                    // Failed programs are retried only when their first
                    // bytes change (cheap check).
                    if matches!(e, FrontendError::Plan(_)) {
                        return Err(e.clone());
                    }
                }
            }
        }
        let result = self.build(address, continuation, stage, user_data_base, user_data_count, wave_size);
        match &result {
            Err(FrontendError::Decode(_)) => self.stats.decode_failures += 1,
            Err(_) => self.stats.plan_failures += 1,
            Ok(_) => {}
        }
        self.programs.insert(key, (result.clone(), sync));
        self.stats.programs = self.programs.len();
        result
    }

    fn build(&self, address: u64, continuation: u64, stage: ShaderStage, base: u32, count: u32, wave_size: u32) -> Result<Arc<StageProgram>, FrontendError> {
        let (program, code_hash, code, image) = self.decode(address, continuation)?;
        let code_len = code.len() as u64;
        let link_registers = if continuation == 0 { self.link_registers(&program, address, code_len, stage, base, count, wave_size) } else { Vec::new() };
        self.plan_program(address, continuation, program, code_hash, code, code_len, image, stage, base, count, wave_size, link_registers)
    }

    /// User-data registers naming code this program calls or continues in:
    /// calls through user data or call tables (a compute program's
    /// subroutines, a vertex program's fetch subroutine), and a vertex or
    /// pixel program ending in a jump through user data.
    #[allow(clippy::too_many_arguments)]
    fn link_registers(&self, program: &Program, address: u64, code_len: u64, stage: ShaderStage, base: u32, count: u32, wave_size: u32) -> Vec<u32> {
        use ps5shader::cfg::link;
        if !link_enabled() {
            return Vec::new();
        }
        let mut registers = Vec::new();
        if program.instructions.iter().any(|i| i.opcode == ps5shader::opcode::Opcode::SSwappcB64) {
            let site = link::CallSite { user_data_base: base, user_data_count: count, wave_size };
            for c in link::find_user_data_calls(program, site) {
                registers.extend([c.user_data_register, c.user_data_register + 1]);
            }
            for c in link::find_table_calls(program, site) {
                registers.extend([c.user_data_register, c.user_data_register + 1]);
            }
        }
        if registers.is_empty() && matches!(stage, ShaderStage::Vertex | ShaderStage::Pixel) {
            let after = self.read_word(address + code_len).unwrap_or(0);
            if let Some(t) = link::find_user_data_tail(program, base, count, after) {
                registers.push(t.user_data_register);
                if t.constant_high.is_none() {
                    registers.push(t.user_data_register + 1);
                }
            }
        }
        registers.sort_unstable();
        registers.dedup();
        registers
    }

    /// The code `raw` runs with `user_data`, its callees or continuation
    /// read from guest memory now and linked in.
    fn link_code(&self, raw: &StageProgram, user_data: &[u32]) -> Result<Vec<u32>, FrontendError> {
        use ps5shader::cfg::link;
        let words: Vec<u32> = raw.code.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let read = |a: u64, buf: &mut [u8]| self.memory.read_backing(a, buf);
        let program = raw.plan.program();
        let fail = |e: String| FrontendError::Decode(format!("linking: {e}"));
        let site = link::CallSite { user_data_base: raw.user_data_base, user_data_count: raw.user_data_count, wave_size: raw.wave_size };
        if let Some(l) = link::link_program(program, &words, site, user_data, &read).map_err(fail)? {
            return Ok(l.linked.code);
        }
        let after = self.read_word(raw.address + raw.code_len).unwrap_or(0);
        let tail = link::find_user_data_tail(program, raw.user_data_base, raw.user_data_count, after)
            .ok_or_else(|| FrontendError::Decode("linking: no user-data tail".into()))?;
        let slot = (tail.user_data_register - raw.user_data_base) as usize;
        let low = user_data.get(slot).copied().unwrap_or(0) as u64;
        let high = tail.constant_high.unwrap_or_else(|| user_data.get(slot + 1).copied().unwrap_or(0)) as u64;
        let target = low | high << 32;
        if target == 0 || target & 3 != 0 {
            return Err(fail(format!("user-data tail continuation address {target:#x} is null or not dword aligned")));
        }
        let reader = |a: u64| self.read_word(a);
        let continuation = decode_program(&reader, target).map_err(|e| fail(format!("continuation at {target:#x}: {e}")))?;
        let end = continuation.instructions.last().map_or(0, |l| l.next_pc() as usize);
        let mut bytes = vec![0u8; end];
        if !self.memory.read_backing(target, &mut bytes) {
            return Err(fail(format!("continuation at {target:#x} unreadable")));
        }
        let continuation_words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        link::link_user_data_tail(program, &words, tail, &continuation, &continuation_words).map_err(fail)
    }

    /// [`Self::program`] for one dispatch or draw: a program that calls
    /// code through user data or continues in code named by user data
    /// gets that code linked in (read again at each CPU synchronisation
    /// point; `RPS5E_SHADER_LINK=0` off).
    pub fn program_for(
        &mut self,
        address: u64,
        stage: ShaderStage,
        user_data_base: u32,
        user_data_count: u32,
        wave_size: u32,
        user_data: &[u32],
    ) -> Result<Arc<StageProgram>, FrontendError> {
        let raw = self.program(address, stage, user_data_base, user_data_count, wave_size)?;
        if raw.link_registers.is_empty() {
            return Ok(raw);
        }
        let values: Vec<u32> = raw.link_registers.iter().map(|&r| user_data.get((r - user_data_base) as usize).copied().unwrap_or(0)).collect();
        let key = (Arc::as_ptr(&raw) as usize, values);
        let sync = crate::memory::sync_generation();
        if let Some((p, checked, _)) = self.linked.get(&key) {
            if *checked == sync {
                return p.clone();
            }
        }
        let result = self.link_code(&raw, user_data).and_then(|words| {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let code_hash = hash_bytes(&bytes);
            if let Some((Ok(p), _, _)) = self.linked.get(&key) {
                if p.code_hash == code_hash {
                    return Ok(p.clone());
                }
            }
            let program = ps5shader::decode_blob(&bytes, address).map_err(|e| FrontendError::Decode(format!("linked program: {e}")))?;
            let image = super::sources::CodeImage::from_bytes(address, &bytes);
            self.plan_program(address, 0, program, code_hash, raw.code.clone(), bytes.len() as u64, image, stage, user_data_base, user_data_count, wave_size, Vec::new())
        });
        if let Err(e) = &result {
            crate::log_once(&format!("ps5gpu: {stage:?} program {address:#x}: {e}"));
        }
        if self.linked.len() > 4096 {
            self.linked.clear();
        }
        self.linked.insert(key, (result.clone(), sync, raw));
        result
    }

    /// Plan a decoded program.
    #[allow(clippy::too_many_arguments)]
    fn plan_program(
        &self,
        address: u64,
        continuation: u64,
        program: Program,
        code_hash: u64,
        code: Vec<u8>,
        code_len: u64,
        image: super::sources::CodeImage,
        stage: ShaderStage,
        base: u32,
        count: u32,
        wave_size: u32,
        link_registers: Vec<u32>,
    ) -> Result<Arc<StageProgram>, FrontendError> {
        let user_data_registers = BindingLayout::collect_user_data_registers(&program, base, count);
        let uses_global_data_share = BindingLayout::uses_global_data_share(&program);
        let uses_local_data_share = BindingLayout::uses_local_data_share(&program);
        let reads_shader_base = BindingLayout::reads_shader_base(&program);
        let address_independent = continuation == 0
            && !program.instructions.iter().any(|i| matches!(i.opcode, ps5shader::opcode::Opcode::SGetpcB64 | ps5shader::opcode::Opcode::SSetpcB64 | ps5shader::opcode::Opcode::SSwappcB64));
        let hash = code_hash ^ (stage_tag(stage) as u64).rotate_left(56);
        let plan = ShaderResourcePlan::extract(program, stage, hash, base, count, None, wave_size)
            .map_err(|e| FrontendError::Plan(e.to_string()))?;
        let writes_device_memory = plan.resource_uses().iter().any(|u| {
            u.binding == ps5shader::resources::ResourceBinding::DeviceAddress
                && matches!(plan.graph.memory().get(u.memory_index).access, ps5shader::resources::MemoryAccess::Write | ps5shader::resources::MemoryAccess::Atomic)
        });
        let reads_memory_only = reads_memory_only(&plan);
        let param_exports = param_exports(plan.program());
        Ok(Arc::new(StageProgram {
            address,
            stage,
            code_hash,
            code_len,
            code,
            user_data_base: base,
            user_data_count: count,
            wave_size,
            plan,
            user_data_registers,
            uses_global_data_share,
            uses_local_data_share,
            reads_shader_base,
            reads_memory_only,
            address_independent,
            writes_device_memory,
            param_exports,
            link_registers,
            continuation,
            image: Arc::new(image),
            guest_plan: false,
        }))
    }

    /// Materialise `program` for one draw and allocate its binding layout
    /// at push cursor `push_cursor`. `user_data`: the stage's user-data
    /// SGPR values from the plan's base on. `thread_limits`: the thread
    /// counts of a compute dispatch whose last groups are partial (shader
    /// data the program compares its global invocation id with).
    pub fn prepare(
        &mut self,
        program: &Arc<StageProgram>,
        user_data: &[u32],
        push_cursor: u32,
        thread_limits: Option<[u32; 3]>,
        sched: &mut Scheduler,
        buffers: &mut crate::slices::spec::Buffers,
    ) -> Result<PreparedStage, FrontendError> {
        if !self.bda_checked {
            if let Some(spec) = buffers.spec() {
                // A slice thread takes the decision the committing thread's
                // first prepare made.
                if !buffers.cache().bda_tried() {
                    spec.bail(crate::slices::spec::Hole::Frontend);
                    return Err(FrontendError::Materialize("guest-memory path not decided yet".into()));
                }
                self.bda_checked = true;
                self.bda = buffers.cache().bda_active();
            } else {
                self.bda_checked = true;
                self.bda = buffers.bda_imports(sched).is_some();
                // Publish for the interpreter thread's prefetch instance.
                note_bda(self.bda);
            }
        }
        // A stage reading guest memory runs the program planned for it.
        let variant;
        let ordinary = program;
        let program = if self.bda_for(program) && crate::buffers::guest::guest_plan_enabled() {
            variant = self.bda_program(program)?;
            &variant
        } else if let Some(v) = self.device_reads_variant(program).filter(|_| buffers.spec().is_none()) {
            variant = v;
            self.stats.device_reads += 1;
            &variant
        } else {
            program
        };
        self.waited = false;
        let r = self.prepare_program(program, user_data, push_cursor, thread_limits, sched, buffers);
        if self.waited {
            self.note_waited(ordinary, program);
            if !Arc::ptr_eq(ordinary, program) {
                // The variant waited as well: the ordinary program, whose
                // words just landed, and no module built for the variant.
                self.waited = false;
                return self.prepare_program(ordinary, user_data, push_cursor, thread_limits, sched, buffers);
            }
        }
        r
    }

    fn prepare_program(
        &mut self,
        program: &Arc<StageProgram>,
        user_data: &[u32],
        push_cursor: u32,
        thread_limits: Option<[u32; 3]>,
        sched: &mut Scheduler,
        buffers: &mut crate::slices::spec::Buffers,
    ) -> Result<PreparedStage, FrontendError> {
        let count = (program.user_data_count as usize).min(user_data.len());
        let user = &user_data[..count];
        let speculative = buffers.spec().is_some();
        if !prepare_memo_enabled() || crate::pipeline::trace::active() {
            let tracker = buffers.tracker().clone();
            if speculative {
                let deps = RefCell::new(PageDeps::begin(self.memory));
                let r = self.prepare_uncached(program, user, push_cursor, thread_limits, None, &tracker, Some(&deps));
                return Self::speculative_result(buffers, &tracker, &deps.into_inner(), r);
            }
            let cell = RefCell::new((sched, buffers));
            return self.prepare_uncached(program, user, push_cursor, thread_limits, Some(&cell), &tracker, None);
        }
        let stamp = buffers.content_stamp();
        let me = Arc::as_ptr(program) as usize;
        let bda = self.bda_for(program);
        // While a frame capture records, every read of a materialisation
        // must happen again (the capture holds what the reads saw).
        let pages = prepare_pages_enabled() && !crate::capture::active();
        let memory = self.memory;
        let content = pages && content_checks_enabled() && !speculative;
        let found = crate::backend::profile::dtime(35, || {
            self.prepared.iter_mut().position(|(k, st, deps, _)| {
                k.program == me
                    && k.push_cursor == push_cursor
                    && k.thread_limits == thread_limits
                    && k.bda == bda
                    && k.user_data[..] == *user
                    && if pages {
                        deps.current(buffers.tracker(), memory) || (content && deps.revalidate(buffers.tracker(), memory) && {
                            CONTENT_REVALIDATED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            true
                        })
                    } else {
                        *st == stamp
                    }
            })
        });
        if let Some(i) = found {
            self.stats.prepared_reuses += 1;
            if let Some(spec) = buffers.spec() {
                spec.note_deps(buffers.tracker(), &self.prepared[i].2);
                if !pages {
                    // Valid while the content stamp holds: on a slice
                    // thread that is the frozen state; the bytes read are
                    // not known, so the draw cannot be checked at commit.
                    spec.bail(crate::slices::spec::Hole::Frontend);
                }
            }
            return Ok(self.prepared[i].3.clone());
        }
        let deps = RefCell::new(if speculative { PageDeps::begin_with_limit(memory, 4096) } else { PageDeps::begin(memory) });
        if content_checks_enabled() && !speculative {
            deps.borrow_mut().record_contents();
        }
        let tracker = buffers.tracker().clone();
        if speculative {
            let r = self.prepare_uncached(program, user, push_cursor, thread_limits, None, &tracker, Some(&deps));
            let deps = deps.into_inner();
            let r = Self::speculative_result(buffers, &tracker, &deps, r);
            if let Ok(p) = &r {
                if !deps.overflowed() {
                    let key = PreparedKey { program: me, push_cursor, thread_limits, bda, user_data: user.into() };
                    if self.prepared.len() >= RECENT_PREPARED {
                        self.prepared.pop_back();
                    }
                    self.prepared.push_front((key, stamp, deps, p.clone()));
                }
            }
            return r;
        }
        let cell = RefCell::new((sched, buffers));
        let p = self.prepare_uncached(program, user, push_cursor, thread_limits, Some(&cell), &tracker, Some(&deps))?;
        // Reads that synchronised GPU data may have moved the stamp: the
        // result stands for the new one.
        let stamp = cell.into_inner().1.content_stamp();
        let key = PreparedKey { program: me, push_cursor, thread_limits, bda, user_data: user.into() };
        if self.prepared.len() >= RECENT_PREPARED {
            self.prepared.pop_back();
        }
        self.prepared.push_front((key, stamp, deps.into_inner(), p.clone()));
        Ok(p)
    }

    /// A slice thread's materialisation: its reads go into the draw's read
    /// set; one that needed the recording state (a GPU-written word) or
    /// read too many pages to check fails the draw (a hole).
    fn speculative_result(
        buffers: &crate::slices::spec::Buffers,
        tracker: &GuestPageTracker,
        deps: &PageDeps,
        r: Result<PreparedStage, FrontendError>,
    ) -> Result<PreparedStage, FrontendError> {
        let Some(spec) = buffers.spec() else { return r };
        if r.is_err() || deps.overflowed() {
            spec.bail(crate::slices::spec::Hole::Frontend);
        } else {
            spec.note_deps(tracker, deps);
        }
        r
    }

    /// [`Self::prepare`] for the interpreter thread (no recording state,
    /// GPU_PLAN §50): a GPU-written guest word - which the recorder would
    /// download - aborts (None), and the result is handed to the recorder
    /// directly instead of the memo.
    pub(crate) fn prepare_prefetch(
        &mut self,
        program: &Arc<StageProgram>,
        user_data: &[u32],
        push_cursor: u32,
        thread_limits: Option<[u32; 3]>,
        tracker: &GuestPageTracker,
    ) -> Option<(PreparedStage, PageDeps)> {
        match PREFETCH_BDA.load(std::sync::atomic::Ordering::Acquire) {
            // The recorder's first prepare decides the guest-memory path.
            0 => return None,
            v => self.bda = v == 2,
        }
        let variant;
        let program = if self.bda_for(program) && crate::buffers::guest::guest_plan_enabled() {
            variant = self.bda_program(program).ok()?;
            &variant
        } else {
            program
        };
        let count = (program.user_data_count as usize).min(user_data.len());
        let user = &user_data[..count];
        // The interpreter's own memo of recent stages, valid while the
        // pages their materialisation read are (as the recorder's memo with
        // page checks): a draw repeating its stage costs a lookup here too.
        let memo = prefetch_memo_enabled() && prepare_memo_enabled() && prepare_pages_enabled();
        let me = Arc::as_ptr(program) as usize;
        let bda = self.bda_for(program);
        if memo {
            let memory = self.memory;
            let content = content_checks_enabled();
            let found = self.prepared.iter_mut().position(|(k, _, deps, _)| {
                k.program == me
                    && k.push_cursor == push_cursor
                    && k.thread_limits == thread_limits
                    && k.bda == bda
                    && k.user_data[..] == *user
                    && (deps.current(tracker, memory) || (content && deps.revalidate(tracker, memory)))
            });
            if let Some(i) = found {
                self.stats.prepared_reuses += 1;
                PREFETCH_MEMO_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let (_, _, deps, stage) = &self.prepared[i];
                return Some((stage.clone(), deps.clone()));
            }
        }
        let deps = RefCell::new(PageDeps::begin(self.memory));
        if content_checks_enabled() {
            deps.borrow_mut().record_contents();
        }
        let stage = self.prepare_uncached(program, user, push_cursor, thread_limits, None, tracker, Some(&deps)).ok()?;
        let deps = deps.into_inner();
        if memo && !deps.overflowed() {
            if self.prepared.len() >= RECENT_PREPARED {
                self.prepared.pop_back();
            }
            let key = PreparedKey { program: me, push_cursor, thread_limits, bda, user_data: user.into() };
            self.prepared.push_front((key, [0; 4], deps.clone(), stage.clone()));
        }
        Some((stage, deps))
    }

    fn prepare_uncached(
        &mut self,
        program: &Arc<StageProgram>,
        user: &[u32],
        push_cursor: u32,
        thread_limits: Option<[u32; 3]>,
        record: Option<&RefCell<(&mut Scheduler, &mut crate::slices::spec::Buffers)>>,
        tracker: &GuestPageTracker,
        deps: Option<&RefCell<PageDeps>>,
    ) -> Result<PreparedStage, FrontendError> {
        let memory = self.memory;
        let shader_base = program.address;
        // Every guest read of the materialisation is noted (before it
        // happens) in `deps`.
        let note = |a: u64, n: u64| {
            if let Some(d) = deps {
                d.borrow_mut().note(tracker, a, n);
            }
        };
        let page = crate::memory::PAGE_BYTES;
        let waited = std::cell::Cell::new(false);
        let read = |a: u64| -> Option<u32> {
            if tracker.has_gpu_dirty_pages(a, 4) {
                // The recording state downloads the current bytes; the
                // prefetch path has none - the read fails and the draw is
                // prepared on the recorder.
                if let Some(cell) = record {
                    let mut g = cell.borrow_mut();
                    let (s, b) = &mut *g;
                    log::trace!("ps5gpu: program {shader_base:#x} reads GPU-written word {a:#x} {}", crate::memory::history::describe(a, 6).unwrap_or_default());
                    if word_wait_log() {
                        static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3000 {
                            log::info!(
                                "ps5gpu: word-wait p{shader_base:#x} {a:#x} {} | {}",
                                b.cache().describe_word(a, s.current_tick()),
                                crate::memory::history::describe(a, 8).unwrap_or_default()
                            );
                        }
                    }
                    let t0 = std::time::Instant::now();
                    crate::backend::profile::preparing(shader_base, || b.download_window(s, a));
                    note_slow_word_wait(shader_base, a, t0.elapsed());
                    waited.set(true);
                } else {
                    return None;
                }
            }
            let mut w = [0u8; 4];
            memory.read_backing(a, &mut w).then(|| u32::from_le_bytes(w))
        };
        let read_clean = |a: u64| -> Option<u32> {
            if tracker.has_gpu_dirty_pages(a, 4) {
                return None;
            }
            let mut w = [0u8; 4];
            memory.read_backing(a, &mut w).then(|| u32::from_le_bytes(w))
        };
        // Pages found resident and clean during this call, with their
        // backing: nothing but GPU work recorded under the back-end lock
        // (which the caller holds) makes a page GPU-dirty again.
        let clean_pages: RefCell<[(u64, usize); 4]> = RefCell::new([(u64::MAX, 0); 4]);
        let resident = |a: u64, out: &mut [u8], _clean: bool| -> bool {
            let n = out.len() as u64;
            let base = a & !(page - 1);
            if (a - base) + n <= page {
                if let Some(&(_, ptr)) = clean_pages.borrow().iter().find(|(p, _)| *p == base) {
                    // SAFETY: the backing alias of a resident guest page stays mapped.
                    unsafe { std::ptr::copy_nonoverlapping((ptr as *const u8).add((a - base) as usize), out.as_mut_ptr(), out.len()) };
                    return true;
                }
                if tracker.has_gpu_dirty_pages(base, page) {
                    return false;
                }
                if let Some(ptr) = memory.backing_ptr(base, page) {
                    crate::capture::note_read(memory, base, page);
                    let mut pages = clean_pages.borrow_mut();
                    pages.rotate_right(1);
                    pages[0] = (base, ptr as usize);
                    // SAFETY: as above.
                    unsafe { std::ptr::copy_nonoverlapping(ptr.add((a - base) as usize), out.as_mut_ptr(), out.len()) };
                    return true;
                }
            }
            if tracker.has_gpu_dirty_pages(a, n) {
                return false;
            }
            memory.read_backing(a, out)
        };
        // Words GPU work recorded before is still producing: a table slot
        // that only copies one is filled on the GPU instead of waiting.
        let gpu_written = |a: u64| tracker.has_gpu_dirty_pages(a, 4);
        // What each read gave goes into `deps` too (see
        // `PageDeps::revalidate`).
        let got_bytes = |a: u64, b: &[u8]| {
            if let Some(d) = deps {
                d.borrow_mut().read(a, b);
            }
        };
        let got_gpu_state = |a: u64, n: u64, dirty: bool| {
            if let Some(d) = deps {
                d.borrow_mut().gpu_state(a, n, dirty);
            }
        };
        let gpu_written_noted = |a: u64| {
            note(a, 4);
            let dirty = gpu_written(a);
            got_gpu_state(a, 4, dirty);
            dirty
        };
        let read_noted = |a: u64| {
            note(a, 4);
            let v = read(a);
            match v {
                Some(w) => got_bytes(a, &w.to_le_bytes()),
                None => got_gpu_state(a, 4, true),
            }
            v
        };
        let read_clean_noted = |a: u64| {
            note(a, 4);
            let v = read_clean(a);
            match v {
                Some(w) => got_bytes(a, &w.to_le_bytes()),
                None => got_gpu_state(a, 4, true),
            }
            v
        };
        let resident_noted = |a: u64, out: &mut [u8], clean: bool| {
            note(a, out.len() as u64);
            let ok = resident(a, out, clean);
            if ok {
                got_bytes(a, out);
            } else {
                got_gpu_state(a, out.len() as u64, true);
            }
            ok
        };
        // The room of the mirrors from a buffer's base on: a NUM_RECORDS
        // word GPU work still writes is left to the program, and the
        // binding covers the mirrored memory from the base on.
        let records_room = |base: u64, stride: u32, word: u64| -> Option<u64> {
            let cell = record?;
            let room = {
                let g = cell.borrow();
                g.1.cache().mirrored_end(base, MAX_RECORDS_ROOM)? - base
            };
            // The count the word held when it last reached guest memory
            // (the previous frame's, as a rule): a buffer whose records
            // then needed more than half the room keeps waiting for the
            // word (its records may reach past the mirrors).
            let mut w = [0u8; 4];
            let stale = if memory.read_backing(word, &mut w) { u32::from_le_bytes(w) as u64 } else { return None };
            let need = if stride == 0 { stale } else { stale * stride as u64 };
            // The binding covers twice that (a larger written range costs
            // write-throughs and CPU faults on every use).
            let twice = need.saturating_mul(2);
            (twice <= room).then(|| twice.max(RECORDS_ROOM_MIN).min(room))
        };
        let mut inputs = ResourceRuntimeInputs::new(user, shader_base);
        // RPS5E_SAFE: such words are read back (synchronous download).
        if !crate::backend::ab::safe() {
            inputs.gpu_written = Some(&gpu_written_noted);
            if defer_records() && crate::backend::ab::release_fast("RPS5E_DEFER_RECORDS") {
                inputs.records_room = Some(&records_room);
            }
        }
        inputs.read_memory = Some(&read_noted);
        inputs.read_clean_memory = Some(&read_clean_noted);
        inputs.read_resident_memory = Some(&resident_noted);
        let plan = &program.plan;
        let materialized = crate::backend::profile::time(22, || self.materializer.materialize(plan, &inputs, &resident_noted));
        self.waited |= waited.get();
        let (snapshot, specialization) = match materialized {
            Ok(v) => v,
            Err(e) => {
                self.stats.materialize_failures += 1;
                let m = match e {
                    MaterializationError::Other(m) => m,
                    other => other.to_string(),
                };
                return Err(FrontendError::Materialize(m));
            }
        };
        // The sub-phase laps of the detailed profile (RPS5E_GPU_PROFILE=2).
        let mut lap = crate::backend::profile::Lap::start();
        let mut dlap = |k: usize| {
            if crate::backend::profile::detailed() {
                lap.lap(k);
            }
        };
        let bda = self.bda_for(program);
        let intern = layout_intern_enabled();
        let laid_out = self.laid_out.entry(plan.id).or_default();
        let same_place = |l: &LaidOut| l.push_cursor == push_cursor && l.thread_limits == thread_limits.is_some() && l.bda == bda;
        let found = laid_out.iter().position(|l| Arc::ptr_eq(&l.specialization, &specialization) && same_place(l)).or_else(|| {
            // The same specialization in another `Arc`: its layout is
            // kept, so everything keyed by the layout's identity hits.
            let i = laid_out.iter().position(|l| intern && *l.specialization == *specialization && same_place(l))?;
            laid_out[i].specialization = specialization.clone();
            Some(i)
        });
        dlap(37);
        let (resources, layout, resource_key) = match found {
            Some(index) => {
                let l = &laid_out[index];
                let found = (l.resources.clone(), l.layout.clone(), l.resource_key);
                if index != 0 {
                    let l = laid_out.remove(index);
                    laid_out.insert(0, l);
                }
                found
            }
            None => {
                let by_spec = self.specialized.entry((plan.id, bda)).or_default();
                let resources = match by_spec.get(&*specialization) {
                    Some(r) => r.clone(),
                    None => {
                        let mut r = apply_specialization(plan, &specialization).map_err(|e| FrontendError::Materialize(e.to_string()))?;
                        if bda && r.info.buffers.iter().all(|b| !b.written && !b.atomic) {
                            r.info.buffers_in_guest_memory = true;
                        }
                        let r = Arc::new(r);
                        if by_spec.len() > 256 {
                            by_spec.clear();
                        }
                        by_spec.insert((*specialization).clone(), r.clone());
                        r
                    }
                };
                let uses_flattened = ShaderCompileRequest::requires_flattened_table(plan, &resources);
                let layout = BindingLayout::allocate(
                    &resources.info,
                    program.user_data_registers.clone(),
                    program.uses_global_data_share,
                    uses_flattened,
                    program.reads_shader_base,
                    push_cursor,
                    thread_limits.is_some(),
                    wide_offsets_enabled(),
                )
                .map_err(|e| FrontendError::Layout(e.to_string()))?;
                let key = resource_key(&specialization, &layout);
                let layout = Arc::new(layout);
                laid_out.insert(0, LaidOut {
                    specialization: specialization.clone(),
                    resources: resources.clone(),
                    push_cursor,
                    thread_limits: thread_limits.is_some(),
                    bda,
                    layout: layout.clone(),
                    resource_key: key,
                });
                laid_out.truncate(if intern { LAYOUTS_PER_PLAN } else { 8 });
                (resources, layout, key)
            }
        };
        dlap(36);
        let mut shader_data: ShaderData = smallvec::smallvec![0u32; layout.shader_data_dword_count() as usize];
        for (i, &reg) in layout.user_data_registers.iter().enumerate() {
            let index = reg.wrapping_sub(program.user_data_base) as usize;
            shader_data[i] = snapshot.user_data.get(index).copied().or_else(|| user.get(index).copied()).unwrap_or(0);
        }
        if layout.uses_shader_base() {
            let d = layout.shader_base_dword as usize;
            shader_data[d] = shader_base as u32;
            shader_data[d + 1] = (shader_base >> 32) as u32;
        }
        if let Some(limits) = thread_limits.filter(|_| layout.uses_dispatch_thread_limits) {
            let d = layout.dispatch_thread_limits_dword() as usize;
            shader_data[d..d + 3].copy_from_slice(&limits);
        }
        dlap(38);
        Ok(PreparedStage { program: program.clone(), snapshot, specialization, resources, layout, shader_data, shader_base, resource_key })
    }
}
