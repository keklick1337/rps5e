// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The GPU back-end: scheduler, buffer cache, image cache and the pipeline
//! provider behind one lock, driven through the [`RenderExecutor`]
//! interface by the PM4 interpreter.
//!
//! Per draw the back-end
//! 1. handles colour metadata operations (fast-clear eliminate, FMASK/DCC
//!    decompress) and MSAA resolves without a shader;
//! 2. binds the render targets: finds or creates the cached images, applies
//!    pending clears (DCC clear codes found in guest metadata or recorded
//!    metadata fills, DB_RENDER_CONTROL depth/stencil clears, HTILE fills)
//!    and uploads guest contents otherwise;
//! 3. asks the [`PipelineProvider`] to execute the draw. The built-in
//!    [`StubPipelines`] provider records "needs shader" until the shader
//!    back-end is plugged in with [`set_pipeline_provider`].
//!
//! End-of-pipe callbacks run on the GPU priority thread once everything
//! recorded before them completed. Display flips take the image the GPU
//! rendered when there is one ([`capture_display`]).
//!
//! The fault handler reaches the caches through small stores attached to
//! [`crate::memory::global`]; they take the back-end lock only to download
//! GPU-written buffer pages and refuse when the faulting thread already
//! holds it.

use crate::buffers::{BufferCache, DownloadStep};
use crate::device::{GpuDevice, Image};
use crate::images::{color_key, depth_key, ImageCache};
use crate::memory::lock::thread_id;
use crate::memory::tracker::GuestPageTracker;
use crate::memory::{BufferStore, GuestGpuMemory, ImageStore, UnmapListener};
use crate::capture::format::Event;
use crate::render::{color_mode, ColorTarget, DepthTarget, GdsTransfer, Rect, RenderExecutor, ResolvedDispatch, ResolvedDraw};
use crate::scheduler::Scheduler;
use crate::slice_shadow as shadow;

mod slice_rec;
pub use slice_rec::{CpuBytes, PriorWrite, SegDraw, SegItem, SegmentOut, SliceItem, SliceRecorder, SliceUnit};
use ash::vk;
use crate::fxhash::HashMap;
use std::sync::atomic::{AtomicI32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// Guest bytes of cached images above which images idle for a few frames
/// are evicted (GPU-modified ones written back first).
const IMAGE_BUDGET: u64 = 3 << 30;

/// What the pipeline provider did with a draw or dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawOutcome {
    /// No host pipeline exists for the shaders yet.
    NeedsShader,
    /// The work was recorded; the bound targets were written.
    Recorded,
    /// Nothing to do (e.g. culled).
    Skipped,
    /// A slice thread's draw needs the exact state: the committing thread
    /// records it (R7b).
    Hole,
}

/// The images bound for a draw.
pub struct DrawTargets {
    /// `(target, cached image id)` per bound color slot.
    pub colors: smallvec::SmallVec<[(ColorTarget, u64); 8]>,
    pub depth: Option<(DepthTarget, u64)>,
}

impl DrawTargets {
    /// A host pass takes one sample count. When the bound attachments mix
    /// counts (the pass count is the depth's, else the first colour's),
    /// keep only those whose count equals the raster count
    /// (PA_SC_AA_CONFIG.MSAA_NUM_SAMPLES, `raster`). True when any was
    /// dropped. `RPS5E_MIXED_SAMPLES_DROP=0` keeps them all.
    pub fn drop_mixed_samples(&mut self, raster: u32) -> bool {
        let Some(pass) = self.depth.map(|(z, _)| z.samples).or(self.colors.first().map(|(t, _)| t.samples)) else {
            return false;
        };
        if self.colors.iter().all(|(t, _)| t.samples == pass) {
            return false;
        }
        let before = self.colors.len() + self.depth.is_some() as usize;
        self.colors.retain(|(t, _)| t.samples == raster);
        if self.depth.is_some_and(|(z, _)| z.samples != raster) {
            self.depth = None;
        }
        self.colors.len() + self.depth.is_some() as usize != before
    }
}

/// `RPS5E_MIXED_SAMPLES_DROP=0`: attachments of different sample counts
/// stay bound together (an invalid host pass).
fn mixed_samples_drop_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_MIXED_SAMPLES_DROP").is_ok_and(|v| v == "0"))
}

/// Apply [`DrawTargets::drop_mixed_samples`] for draw `d` (logged once).
pub(crate) fn drop_mixed_sample_targets(targets: &mut DrawTargets, d: &ResolvedDraw) {
    if mixed_samples_drop_enabled() && targets.drop_mixed_samples(1 << (d.aa_config & 7).min(3)) {
        crate::log_once("ps5gpu: dropping attachments whose sample count differs from the raster");
    }
}

/// Everything a pipeline provider may use to record a draw.
pub struct DrawContext<'a> {
    pub dev: &'a Arc<GpuDevice>,
    /// The recording scheduler (a slice thread's own, R7b).
    pub sched: &'a mut Scheduler,
    /// The caches, exact or (on a slice thread) speculative.
    pub buffers: crate::slices::spec::Buffers<'a>,
    pub images: crate::slices::spec::Images<'a>,
    pub targets: &'a DrawTargets,
}

impl DrawContext<'_> {
    /// The speculative state of a slice thread's draw (None: exact).
    pub fn spec(&self) -> Option<&crate::slices::spec::SpecCtx> {
        self.buffers.spec()
    }

    /// Fail a speculative draw (it becomes a hole); true when the draw is
    /// speculative.
    pub fn bail(&self, why: crate::slices::spec::Hole) -> bool {
        match self.buffers.spec() {
            Some(s) => {
                s.bail(why);
                true
            }
            None => false,
        }
    }

    /// The speculative draw failed already.
    pub fn bailed(&self) -> bool {
        self.buffers.spec().is_some_and(|s| s.bailed().is_some())
    }
}

/// Turns resolved draws/dispatches into recorded GPU work (the shader
/// back-end implements this).
pub trait PipelineProvider: Send {
    fn draw(&mut self, ctx: &mut DrawContext, draw: &ResolvedDraw) -> DrawOutcome;
    fn dispatch(&mut self, ctx: &mut DrawContext, dispatch: &ResolvedDispatch) -> DrawOutcome;
    /// Whether [`Self::dispatch`] runs this indirect dispatch with thread
    /// dimensions from its argument record on the GPU. When false the
    /// back-end reads the record (waiting for its writers) and passes a
    /// direct dispatch instead.
    fn indirect_thread_dimensions(&self, _dispatch: &ResolvedDispatch) -> bool {
        false
    }
    /// Guest ranges the last recorded dispatch overwrote through buffers it
    /// does not read (candidates for a metadata clear).
    fn take_overwrites(&mut self) -> Vec<(u64, u64)> {
        Vec::new()
    }
    /// DCC keys of the surfaces the work recorded since the last call
    /// stored into through image descriptors with DCC compression on.
    fn take_compressed_stores(&mut self) -> Vec<crate::render::StoredKeys> {
        Vec::new()
    }
    /// The device buffer holding the global data share (64 KiB) shader
    /// programs use; None when the provider has none.
    fn gds(&mut self, _sched: &mut Scheduler) -> Option<vk::Buffer> {
        None
    }
    /// A diagnostics line for the periodic back-end summary.
    fn summary(&self) -> Option<String> {
        None
    }
    /// Wait for background compile work (optimised rebuilds of pipelines
    /// in use) and take its results in.
    fn settle_compiles(&mut self) {}
    /// Module and pipeline build times, and waits for them.
    fn compile_report(&self) -> Option<String> {
        None
    }
    /// Bytes of the provider's staging ring still reserved for uncompleted
    /// ticks and its size ((0, 0): none). A nearly full ring forces early
    /// submits (an allocation would otherwise wait for the GPU).
    fn staging_usage(&self) -> (u64, u64) {
        (0, 0)
    }
    /// A display buffer was presented (frame boundary).
    fn frame_presented(&mut self) {}
    /// A provider for a slice thread (R7b): records speculatively over the
    /// same pipelines (None: slices are not supported).
    fn slice_provider(&self) -> Option<Box<dyn PipelineProvider>> {
        None
    }
    /// The committing provider has set up what slice providers share.
    fn slice_ready(&self) -> bool {
        false
    }
    /// Staging handed out at provisional tick `from` is read in tick `to`.
    fn retag_tick(&mut self, _from: u64, _to: u64) {}
    /// `RPS5E_SLICE_VERIFY`: the decision digest of the last draw.
    fn take_digest(&mut self) -> u64 {
        0
    }
    /// The last draw or dispatch was given up unrecorded for a compile it
    /// must wait for (only while [`crate::pipeline::unlocked_wait`] allows
    /// it): the caller waits with the back-end unlocked and tries it again.
    fn take_compile_pending(&mut self) -> bool {
        false
    }
    /// Moves whenever a pipeline a draw may have bound is replaced (an
    /// optimised build swapped in): slice draws recorded before are stale.
    fn swap_generation(&self) -> u64 {
        0
    }
}

/// The provider used until a shader back-end is installed: every draw and
/// dispatch "needs a shader"; the distinct programs are counted.
#[derive(Default)]
pub struct StubPipelines {
    /// (vertex-stage address, pixel address) -> draws.
    pub graphics: HashMap<(u64, u64), u64>,
    pub compute: HashMap<u64, u64>,
}

impl PipelineProvider for StubPipelines {
    fn draw(&mut self, _ctx: &mut DrawContext, d: &ResolvedDraw) -> DrawOutcome {
        let vs = d.shaders.vertex_program_address();
        *self.graphics.entry((vs, d.shaders.ps.address)).or_default() += 1;
        DrawOutcome::NeedsShader
    }

    fn dispatch(&mut self, _ctx: &mut DrawContext, d: &ResolvedDispatch) -> DrawOutcome {
        *self.compute.entry(d.shader.address).or_default() += 1;
        DrawOutcome::NeedsShader
    }
}

/// Back-end counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct BackendStats {
    pub draws: u64,
    pub dispatches: u64,
    pub needs_shader_draws: u64,
    pub needs_shader_dispatches: u64,
    pub recorded_draws: u64,
    pub targetless_draws: u64,
    pub metadata_ops: u64,
    pub resolves: u64,
    pub color_clears: u64,
    pub depth_clears: u64,
    pub end_of_pipe: u64,
    /// Ticks submitted early because the memory retired into them reached
    /// [`retired_bytes_budget`].
    pub retired_budget_submits: u64,
    pub captures: u64,
    pub fills: u64,
    /// Pattern-fill kernel dispatches run as buffer fills (or skipped as
    /// storing nothing).
    pub pattern_fills: u64,
    /// Image-copy kernel dispatches run as image copies.
    pub copy_kernels: u64,
    pub errors: u64,
    /// Compute-written fast-clear metadata decided on the GPU / read back
    /// on the CPU (a GPU wait each).
    pub dcc_gpu_decisions: u64,
    pub dcc_cpu_reads: u64,
    /// Fast-clear codes in DCC keys spent by stores through descriptors
    /// with DCC compression on ([`Backend::compressed_stores_done`]).
    pub dcc_store_spent: u64,
    /// Ticks submitted before the end of their command-stream slice
    /// ([`EarlySubmit`]), and write-throughs recorded right after the
    /// dispatch writing their ranges.
    pub early_submits: u64,
    /// Early submits forced by a CPU access waiting for the open buffer
    ///.
    pub submit_requests: u64,
    pub eager_throughs: u64,
    /// Ticks submitted early only for the write-throughs of ranges a CPU
    /// access waited for before (`RPS5E_EAGER_HOT_THROUGH`).
    pub hot_eager_throughs: u64,
}

/// Work recorded into the open tick since its first draw or dispatch.
#[derive(Clone, Copy, Debug, Default)]
struct EarlySubmit {
    tick: u64,
    draws: u32,
    dispatches: u32,
    since: Option<std::time::Instant>,
}

/// When the open tick is submitted before the end of its slice
/// ([`Backend::early_submit`]): after this many dispatches or draws, or
/// this much recording time — the counts only outside a rendering scope
/// (a submit ends the open pass, dynamic rendering, and the next item
/// opens it again, so inside one only `scope_micros` applies) —, or
/// sooner when write-throughs of GPU results wait for it (they are
/// recorded at the submit: the copies follow their writers by a fraction
/// of a millisecond, and a CPU access waits for that tick only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EarlyPolicy {
    pub dispatches: u32,
    pub draws: u32,
    pub micros: u64,
    pub scope_micros: u64,
    pub through_micros: u64,
}

impl EarlyPolicy {
    /// The tick is due given the work recorded into it, the time since its
    /// first item, an open rendering scope, write-throughs waiting and a
    /// CPU access waiting for the open buffer (GPU_PLAN §57: it ends the
    /// open pass — the access is blocked now, the pass churn is cheaper).
    /// With a rendering scope open only its time threshold applies: the
    /// count thresholds are reached long before `scope_micros` (hundreds
    /// of draws at ~13 µs each), while every submit ends the pass and the
    /// next item reopens it — the pass churn once doubled the recording
    /// cost per draw.
    pub(crate) fn due(&self, draws: u32, dispatches: u32, micros: u64, scope_open: bool, through: bool, request: bool) -> bool {
        if request {
            return true;
        }
        if scope_open {
            return micros >= self.scope_micros;
        }
        if dispatches >= self.dispatches || draws >= self.draws {
            return true;
        }
        micros >= self.micros || (through && micros >= self.through_micros)
    }
}

/// A CPU access waiting for GPU results whose copy sits in the open
/// command buffer names its tick here; the recording thread submits at
/// its next item (GPU_PLAN §57: a fault's wait ends with the tick, not
/// the slice — the GPU is idle 66 % of the time while the fault waits).
static SUBMIT_REQUESTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn request_submit(tick: u64) {
    if submit_request_enabled() {
        SUBMIT_REQUESTED.fetch_max(tick, std::sync::atomic::Ordering::AcqRel);
    }
}

/// `RPS5E_SUBMIT_REQUEST=0`: a CPU access waiting for a copy in the open
/// command buffer no longer asks the recording thread to submit at its
/// next item (the wait then ends with the slice, as before).
fn submit_request_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    ab::new_fast() && *ON.get_or_init(|| ab::switch_on("RPS5E_SUBMIT_REQUEST"))
}

/// `RPS5E_EARLY_SUBMIT=0`: the open tick is submitted at the end of its
/// command-stream slice only (and when its retired memory or a wait needs
/// it): the GPU gets nothing of a long slice until it was recorded whole.
/// `RPS5E_EARLY_SUBMIT=<dispatches>,<draws>,<us>`: the thresholds
/// (default 64 dispatches, 512 draws, 500 us; the counts apply outside a
/// rendering scope only — inside one the submit waits 8 times the us,
/// because it ends the open pass).
fn early_policy() -> Option<EarlyPolicy> {
    static P: OnceLock<Option<EarlyPolicy>> = OnceLock::new();
    let p = *P.get_or_init(|| parse_early_policy(ab::safe(), std::env::var("RPS5E_EARLY_SUBMIT").ok().as_deref()));
    p.filter(|_| ab::new_fast())
}

pub(crate) fn parse_early_policy(safe: bool, v: Option<&str>) -> Option<EarlyPolicy> {
    let mut p = EarlyPolicy { dispatches: 64, draws: 512, micros: 500, scope_micros: 4000, through_micros: 250 };
    match v.map(str::trim) {
        _ if safe => return None,
        Some("0") | Some("off") => return None,
        Some(v) if v.contains(',') => {
            let n: Vec<u64> = v.split(',').map(|x| x.trim().parse().unwrap_or(0)).collect();
            if n.len() != 3 || n.iter().any(|&x| x == 0) {
                log::warn!("ps5gpu: RPS5E_EARLY_SUBMIT={v}: expected <dispatches>,<draws>,<us>; defaults used");
            } else {
                p.dispatches = n[0].min(u32::MAX as u64) as u32;
                p.draws = n[1].min(u32::MAX as u64) as u32;
                p.micros = n[2];
                p.scope_micros = n[2] * 8;
                p.through_micros = (n[2] / 2).max(1);
            }
        }
        _ => {}
    }
    Some(p)
}

/// `RPS5E_EAGER_THROUGH=0`: write-throughs of a dispatch's ranges are
/// recorded right after the dispatch instead of at the next submit (end of
/// the slice, an early submit, end of pipe, a CPU access). On by default
/// (GPU_PLAN §52 live A/B): +8-12 % in SILENT HILL's heavy apartment (the
/// CPU's fault on a written page waits for less GPU backlog, wfault -22 %)
/// and neutral in ASTRO BOT's eop-heavy profile (its in-run A/B counter).
/// A/B in one run with `RPS5E_AB_SCOPE=release RPS5E_AB_ONLY=eager-through`.
fn eager_through_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    ab::new_fast() && *ON.get_or_init(|| !ab::safe() && !std::env::var("RPS5E_EAGER_THROUGH").is_ok_and(|v| v == "0")) && ab::release_fast("eager-through")
}

/// `RPS5E_WHOLE_METADATA_WRITES=0`: a compute write of any size at a colour
/// target's fast-clear metadata address is decided as a clear right away
/// (default: only writes covering a whole metadata slice).
fn whole_metadata_writes_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| ab::switch_value(false, std::env::var("RPS5E_WHOLE_METADATA_WRITES").ok().as_deref()))
}

/// `RPS5E_EAGER_HOT_THROUGH=0`: write-throughs of ranges a CPU access
/// waited for before no longer follow their writing dispatch at once
/// (they wait for the next submit, like every other write-through).
fn eager_hot_through_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    ab::new_fast() && *ON.get_or_init(|| ab::switch_on("RPS5E_EAGER_HOT_THROUGH"))
}

/// A presentation snapshot: a copy of a rendered display image.
pub struct Snapshot {
    pub image: Image,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
}

/// A display frame the GPU produced: wait for `tick` on `timeline`, then
/// read `snapshot` (GENERAL layout).
#[derive(Clone)]
pub struct GpuFrame {
    pub snapshot: Arc<Snapshot>,
    pub timeline: vk::Semaphore,
    pub tick: u64,
}

/// A metadata slice found without a clear code: the decision's words
/// (slice, format and clear word, CMASK), the guest pages its bytes were
/// read from (their generations) and the range the GPU-dirty check
/// covers. Valid while the words hold, no GPU write covers the range (a
/// write-back would change the bytes without a fault) and every page
/// keeps its generation — GPU_PLAN §56: per-page checks replace the
/// global epochs, which any guest write anywhere moved (every draw).
struct DccNone {
    words: [u64; 2],
    deps: crate::memory::deps::PageDeps,
    checked: u64,
}

pub struct Backend {
    pub dev: Arc<GpuDevice>,
    pub sched: Scheduler,
    pub memory: &'static GuestGpuMemory,
    pub tracker: Arc<GuestPageTracker>,
    pub buffers: BufferCache,
    pub images: ImageCache,
    pub pipelines: Box<dyn PipelineProvider>,
    pub stats: BackendStats,
    /// The last draw or dispatch was given up for a pending compile
    /// ([`Backend::take_compile_pending`]).
    compile_pending: bool,
    snapshots: Vec<Arc<Snapshot>>,
    trace_left: u32,
    housekeeping: u64,
    /// Recorded dispatches per compute shader address (journal context).
    dispatch_counts: HashMap<u64, u64>,
    /// HTILE addresses of the depth targets seen so far.
    htile: crate::fxhash::HashSet<u64>,
    /// CMASK bases of bound fast-clear color targets: fills of them are
    /// recorded as metadata fills.
    cmask: crate::fxhash::HashSet<u64>,
    /// DCC addresses of the colour targets seen so far: true when a
    /// compute program overwrote the metadata since the last bind.
    dcc: HashMap<u64, bool>,
    /// Metadata slices found without a clear code, with what they depend
    /// on (see `pending_dcc_clear_inner`).
    dcc_none: HashMap<u64, DccNone>,
    /// The colour target last bound with each DCC address.
    dcc_targets: HashMap<u64, ColorTarget>,
    /// Recent target registers with their image keys (most recent first):
    /// draws bind the same targets over and over.
    color_keys: Vec<(ColorTarget, crate::images::ImageKey)>,
    depth_keys: Vec<(crate::render::DepthTarget, crate::images::ImageKey)>,
    /// GPU-side DCC clear decisions (None: decided on the CPU).
    dcc_gpu: Option<crate::dcc::DccResolver>,
    /// HTILE fills not yet consumed, per surface base: (depth slices,
    /// stencil slices) the fill cleared, one bit per slice still to clear
    /// at its next bind.
    htile_cleared: HashMap<u64, (SliceMask, SliceMask)>,
    /// HTILE bases a compute program overwrote (bytes written) since the
    /// last bind of their surface: the words decide the clear at the bind.
    htile_kernel_written: HashMap<u64, u64>,
    /// Memory-pressure marks of the caches, and the tick of the last
    /// pressure collection.
    gc: crate::gc::DeviceMemory,
    gc_tick: u64,
    gc_ticks: crate::gc::CollectionTicks,
    /// Work of the open tick (see [`Backend::early_submit`]).
    early: EarlySubmit,
    /// The targets of the last `bind_targets`, with everything their
    /// binding depended on (see [`TargetMemo`]).
    target_memo: TargetMemo,
}

/// The targets of the last [`Backend::bind_targets`], with everything
/// their binding depended on: the same targets under the same image-cache
/// state (CPU writes over images, image creation, deletion and every GPU
/// event that changes a lookup, the metadata epochs the DCC memo trusts)
/// bind the same images without the key lookups and the prepares - the
/// draws of one pass hit it for every target. The depth half keeps its
/// per-draw clear handling; only its image lookup is memoised.
#[derive(Default)]
struct TargetMemo {
    colors: smallvec::SmallVec<[(crate::render::ColorTarget, u64); 8]>,
    /// The lookup generation of each colour image when bound (the hit
    /// path's validation).
    gens: smallvec::SmallVec<[u64; 8]>,
    depth: Option<(crate::render::DepthTarget, u64)>,
    /// The lookup generation of the depth image (`depth`'s id).
    depth_gen: Option<u64>,
    /// (tick, lookup epoch, generations, cpu-dirty epoch, sync
    /// generation, gpu-write epoch, dcc, dcc-none, cmask, htile,
    /// htile-cleared lens). With `RPS5E_TARGET_GENERATIONS=1` the poison
    /// fields hold 0 and the hit path validates per target instead
    ///.
    stamp: (u64, u64, u64, u64, u64, u64, usize, usize, usize, usize, usize),
}

/// Per-slice metadata clears: (layer, clear value).
type SliceClears = smallvec::SmallVec<[(u32, vk::ClearColorValue); 4]>;

/// How a layer under single-colour DCC keys is cleared.
enum SingleClear {
    /// Cleared to this value (the colour read from guest memory).
    Value(vk::ClearColorValue),
    /// Filled on the GPU from the colour GPU work stored.
    Filled,
    /// Not a clear the back-end can reproduce.
    Undecided,
}

/// Two clear values with the same bits.
fn same_clear(a: &vk::ClearColorValue, b: &vk::ClearColorValue) -> bool {
    // SAFETY: every member of the clear value union is plain data of the same size.
    unsafe { a.uint32 == b.uint32 }
}

/// Runs of consecutive layers with the same clear value: (first, count,
/// index of the value in `clears`).
fn clear_runs(clears: &[(u32, vk::ClearColorValue)]) -> smallvec::SmallVec<[(u32, u32, usize); 4]> {
    let mut runs = smallvec::SmallVec::new();
    let mut i = 0;
    while i < clears.len() {
        let (first, value) = clears[i];
        let mut n = 1;
        while i + n < clears.len() && clears[i + n].0 == first + n as u32 && same_clear(&clears[i + n].1, &value) {
            n += 1;
        }
        runs.push((first, n as u32, i));
        i += n;
    }
    runs
}

/// The planes a uniform HTILE fill of `pattern` clears: ZMask (bits 0..3)
/// 0 the depth, SMem (bits 8..9) 0 the stencil when the stencil keeps its
/// state in HTILE. An expanded fill (ZMask 0xF, SMem 3) clears nothing.
pub fn htile_fill_clears(pattern: u32, stencil_in_htile: bool) -> (bool, bool) {
    (pattern & 0xF == 0, stencil_in_htile && (pattern >> 8) & 3 == 0)
}

/// Bytes of a depth surface's pipe-aligned HTILE (`SW_64KB_Z_X`): 32 KiB
/// per 1024x512 pixels. A fill smaller than this misses words of the
/// surface.
pub fn htile_metadata_bytes(width: u32, height: u32) -> u64 {
    width.div_ceil(1024) as u64 * height.div_ceil(512) as u64 * 32768
}

/// `RPS5E_HTILE_FILL_DECODE=0`: any fill at a depth surface's HTILE
/// clears its depth (the earlier behaviour), instead of the planes the
/// fill's ZMask / SMem clear over the whole metadata.
fn htile_fill_decode_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_HTILE_FILL_DECODE").is_ok_and(|v| v == "0"))
}

/// `RPS5E_HTILE_PER_SLICE=0`: a fill at a depth surface's HTILE clears
/// every slice of the array, and fills at the later slices' own HTILE are
/// not seen (the earlier behaviour), instead of each slice taking the
/// clear of the HTILE it owns (slice s at base + s * the slice's bytes).
fn htile_per_slice_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_HTILE_PER_SLICE").is_ok_and(|v| v == "0"))
}

/// Slices of a depth array whose kernel-stored HTILE is read back at most.
const MAX_KERNEL_HTILE_SLICES: u64 = 256;

/// `RPS5E_HTILE_KERNEL_DECODE=0`: a compute program overwriting a depth
/// surface's HTILE counts as a fill of ZMask 0 / SMem 0 (a depth and
/// stencil clear, whatever it stored: the earlier behaviour), instead of
/// the words it stored deciding the clear at the next bind.
fn htile_kernel_decode_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_HTILE_KERNEL_DECODE").is_ok_and(|v| v == "0"))
}

/// `RPS5E_HTILE_STORE_BACK=0`: a consumed HTILE clear leaves the metadata
/// as cleared in guest memory (the earlier behaviour) instead of storing it
/// back expanded, as the depth block does once it has written the depth.
fn htile_store_back_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_HTILE_STORE_BACK").is_ok_and(|v| v == "0"))
}

/// The planes HTILE words clear: every word's ZMask 0 clears the depth,
/// every word's SMem 0 the stencil (when the stencil keeps its state in
/// HTILE). A slice past the first is only decided by uniform words.
pub fn htile_words_clear(words: &[u32], stencil_in_htile: bool, first_slice: bool) -> (bool, bool) {
    if words.is_empty() || (!first_slice && words.iter().any(|&w| w != words[0])) {
        return (false, false);
    }
    words.iter().fold((true, stencil_in_htile), |(d, s), &w| {
        let (wd, ws) = htile_fill_clears(w, stencil_in_htile);
        (d && wd, s && ws)
    })
}

/// An HTILE word after the depth block applied its clear: the cleared
/// planes are expanded (depth-only: ZMask 0xF with the full depth range;
/// with stencil: ZMask 0xF and the depth bits set, SMem and SR bits set
/// for a cleared stencil).
pub fn expanded_htile_word(word: u32, stencil_in_htile: bool, depth: bool, stencil: bool) -> u32 {
    if !stencil_in_htile {
        return if depth { 0xFFFC_000F } else { word };
    }
    let mut w = word;
    if depth {
        w = (w & 0x3F0) | 0xFFFF_F00F;
    }
    if stencil {
        w |= 0x3F0;
    }
    w
}

/// Array slices a depth view can address: DB_DEPTH_VIEW holds 13-bit
/// first and last slice indices.
pub const DEPTH_VIEW_SLICES: u32 = 1 << 13;

/// One bit per array slice of a depth surface (every slice a depth view
/// can address).
#[derive(Clone)]
pub struct SliceMask(Box<[u64; DEPTH_VIEW_SLICES as usize / 64]>);

impl SliceMask {
    /// Every slice set (`all`) or none.
    pub fn new(all: bool) -> SliceMask {
        SliceMask(Box::new([if all { u64::MAX } else { 0 }; DEPTH_VIEW_SLICES as usize / 64]))
    }

    pub fn test(&self, slice: u32) -> bool {
        slice < DEPTH_VIEW_SLICES && self.0[slice as usize / 64] & (1 << (slice % 64)) != 0
    }

    /// Set the bits of slices `first..first + n` to `on`.
    pub fn set_range(&mut self, first: u32, n: u32, on: bool) {
        let end = first.saturating_add(n).min(DEPTH_VIEW_SLICES);
        for slice in first.min(end)..end {
            let (w, b) = (slice as usize / 64, 1u64 << (slice % 64));
            if on {
                self.0[w] |= b;
            } else {
                self.0[w] &= !b;
            }
        }
    }

    /// Take the bit of `slice`: true when it was set (it is clear after).
    pub fn take(&mut self, slice: u32) -> bool {
        if !self.test(slice) {
            return false;
        }
        self.0[slice as usize / 64] &= !(1 << (slice % 64));
        true
    }
}

#[cfg(test)]
mod slice_tests {
    use super::*;

    fn v(x: u32) -> vk::ClearColorValue {
        vk::ClearColorValue { uint32: [x, 0, 0, 0] }
    }

    #[test]
    fn clear_runs_merge_consecutive_equal_layers() {
        let clears = [(0, v(1)), (1, v(1)), (2, v(2)), (4, v(2)), (5, v(2))];
        assert_eq!(clear_runs(&clears).as_slice(), &[(0, 2, 0), (2, 1, 2), (4, 2, 3)]);
        assert!(clear_runs(&[]).is_empty());
    }

    #[test]
    fn retired_bytes_budget_parses_mib() {
        assert_eq!(super::parse_retired_budget(None), 512 << 20);
        assert_eq!(super::parse_retired_budget(Some(" 64 ")), 64 << 20);
        assert_eq!(super::parse_retired_budget(Some("0")), 0);
        assert_eq!(super::parse_retired_budget(Some("lots")), 512 << 20);
    }

    #[test]
    fn slice_bits_are_consumed_once() {
        let mut mask = SliceMask::new(true);
        assert!(mask.take(3));
        assert!(!mask.take(3));
        assert!(mask.take(0));
        assert!(mask.test(1) && !mask.test(0) && !mask.test(3));
        // Slices past 32 of a depth array are cleared too (a fill clears
        // every slice the 13-bit view indices reach).
        assert!(mask.take(40));
        assert!(mask.take(DEPTH_VIEW_SLICES - 1));
        assert!(!mask.take(DEPTH_VIEW_SLICES));
        let mut none = SliceMask::new(false);
        assert!(!none.take(0) && !none.take(100));
    }

    #[test]
    fn early_submit_policy_and_its_switch() {
        assert_eq!(parse_early_policy(false, Some("0")), None);
        assert_eq!(parse_early_policy(true, None), None, "safe mode");
        let p = parse_early_policy(false, None).unwrap();
        assert_eq!((p.dispatches, p.draws, p.micros, p.scope_micros), (64, 512, 500, 4000));
        let q = parse_early_policy(false, Some("8,32,1000")).unwrap();
        assert_eq!((q.dispatches, q.draws, q.micros, q.scope_micros, q.through_micros), (8, 32, 1000, 8000, 500));
        assert_eq!(parse_early_policy(false, Some("8,0,1")), Some(p), "malformed: defaults");
        // Counts outside a rendering scope.
        assert!(!p.due(0, 63, 10, false, false, false));
        assert!(p.due(0, 64, 10, false, false, false));
        assert!(p.due(512, 0, 10, false, false, false));
        // Time: sooner with a write-through waiting in the tick.
        assert!(!p.due(1, 1, 499, false, false, false));
        assert!(p.due(1, 1, 500, false, false, false));
        assert!(!p.due(1, 1, 249, false, true, false));
        assert!(p.due(1, 1, 250, false, true, false));
        // An open rendering scope: only its (long) time threshold
        // applies, whatever the counts or a waiting write-through say.
        assert!(!p.due(600, 0, 400, true, true, false));
        assert!(!p.due(600, 0, 500, true, false, false), "counts do not submit inside a scope");
        assert!(!p.due(600, 0, 3999, true, true, false));
        assert!(!p.due(10, 0, 3999, true, true, false));
        assert!(p.due(600, 0, 4000, true, false, false));
        assert!(p.due(10, 0, 4000, true, true, false));
        // A CPU access waiting for the open buffer submits at once, even
        // inside a rendering scope.
        assert!(p.due(0, 0, 0, true, false, true));
        assert!(p.due(0, 0, 0, false, false, true));
    }

    #[test]
    fn fault_windows_grow_over_consecutive_faults() {
        const M: u64 = 1 << 20;
        let base = 0x10_0000_0000u64;
        // A first fault: the 1 MiB window around it.
        let (s, e, w) = fault_window(base + 0x1234, 4, 0, 0, true);
        assert_eq!((s, e, w), (base, base + M, M));
        // The next one right after it doubles, and again.
        let (s, e, w) = fault_window(base + M + 8, 4, e, w, true);
        assert_eq!((s, e, w), (base + M, base + 3 * M, 2 * M));
        let (s, e, w) = fault_window(base + 3 * M, 64, e, w, true);
        assert_eq!((s, e, w), (base + 3 * M, base + 7 * M, 4 * M));
        // A fault inside the previous window doubles it too (a polled
        // page the GPU rewrites).
        let (s, e, w) = fault_window(base + 3 * M + 0x2000, 4, e, w, true);
        assert_eq!((s, e, w), (base + 3 * M, base + 11 * M, 8 * M));
        // A fault in no recent window starts over.
        let (s, e, w) = fault_window(base + 64 * M, 4, e, w, true);
        assert_eq!((s, e, w), (base + 64 * M, base + 65 * M, M));
        // Up to the limit.
        let (mut e, mut w) = (e, w);
        for _ in 0..8 {
            let r = fault_window(e, 4, e, w, true);
            (e, w) = (r.1, r.2);
        }
        assert_eq!(w, FAULT_WINDOW_MAX);
        // A fault elsewhere starts over; without the switch nothing grows.
        assert_eq!(fault_window(base + 100 * M, 4, e, w, true).2, M);
        assert_eq!(fault_window(e, 4, e, w, false), (e, e + M, M));
        // An access crossing the window's end is covered whole.
        assert_eq!(fault_window(base + M - 4, 8, 0, 0, true).1, base + 2 * M);
    }

    #[test]
    fn dispatch_journal_entries_read_as_before() {
        let packet = crate::render::DispatchPacket { groups: [4, 1, 1], dispatch_initiator: 1, ..Default::default() };
        let note = crate::scheduler::Note::Dispatch { n: 7, cs: 0x1234, packet, threads: [64, 1, 1], user: [0; 16] };
        assert_eq!(note.to_string(), format!("dispatch #7 of cs={:#x} packet={:?} threads={:?}", 0x1234, packet, [64u32, 1, 1]));
    }
}

/// `RPS5E_GPU_SUBMIT_EACH=1`: submit every recorded draw and dispatch in a
/// tick of its own, so a tick that never completes names exactly one item.
/// `=2` also waits for each such tick and logs items the GPU needed more
/// than 20 ms for.
fn submit_each_mode() -> u8 {
    static V: OnceLock<u8> = OnceLock::new();
    *V.get_or_init(|| std::env::var("RPS5E_GPU_SUBMIT_EACH").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

fn submit_each() -> bool {
    submit_each_mode() != 0
}

/// Bytes of memory retired into the open tick at which it is submitted
/// (`RPS5E_RETIRED_BYTES_MIB`, default 512; 0 never submits for it).
fn retired_bytes_budget() -> u64 {
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| parse_retired_budget(std::env::var("RPS5E_RETIRED_BYTES_MIB").ok().as_deref()))
}

fn parse_retired_budget(v: Option<&str>) -> u64 {
    v.and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(512) << 20
}

/// `RPS5E_GPU_SKIP=<addr>[,<addr>...]` (hex): draws and dispatches using a
/// shader at one of these code addresses are not executed (bisecting a
/// shader that misbehaves on the GPU).
/// `RPS5E_HTILE_WHOLE_CLEAR=0`: a depth clear from cleared HTILE stays
/// within the binding draw's scissor, and a clear draw with an empty
/// scissor clears the whole surface (the earlier behaviour).
fn htile_whole_clear_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| ab::switch_on("RPS5E_HTILE_WHOLE_CLEAR"))
}

/// `RPS5E_METADATA_CONSUME_GPU=0`: a consumed fast clear is written to
/// guest memory only, even when the GPU holds the metadata bytes.
fn consume_on_gpu_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| ab::switch_on("RPS5E_METADATA_CONSUME_GPU"))
}

/// `RPS5E_TARGET_GENERATIONS=1`: the target memo validates per target —
/// each bound image's lookup generation, the lookup epoch and the DCC
/// state lens — instead of the open tick and the global CPU-dirty, sync
/// and GPU-write epochs, which any guest write anywhere moved (every
/// draw; GPU_PLAN §56: 96 % of draws would hit).
fn target_generations_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| ab::switch_on("RPS5E_TARGET_GENERATIONS"))
}

/// `RPS5E_DCC_GPU_ANY=0`: compute-written fast-clear metadata is decided
/// on the GPU for plain 2D single-level targets only; volumes and other
/// targets read the metadata back (a GPU wait per clear).
fn dcc_gpu_any_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| ab::switch_on("RPS5E_DCC_GPU_ANY"))
}

/// `RPS5E_DCC_EVERY_LAYER=0`: of a layered target whose metadata a
/// compute program overwrote, only the first bound layer is read back
/// for a clear code (the others were taken as compressed data).
fn dcc_every_layer_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| ab::switch_on("RPS5E_DCC_EVERY_LAYER"))
}

/// `RPS5E_DB_DEPTH_COPY=0`: a DB depth copy into colour target 0 runs as
/// the draw it is programmed as (the earlier behaviour).
pub(crate) fn db_depth_copy_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_DB_DEPTH_COPY").is_ok_and(|v| v == "0"))
}

fn skipped_shader(addresses: &[u64]) -> bool {
    static V: OnceLock<Vec<u64>> = OnceLock::new();
    let skip = V.get_or_init(|| {
        std::env::var("RPS5E_GPU_SKIP")
            .map(|v| v.split(',').filter_map(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()).collect())
            .unwrap_or_default()
    });
    !skip.is_empty() && addresses.iter().any(|a| skip.contains(a))
}

/// `RPS5E_GPU_TRACE_CS=<addr>` (hex): log the user data and the first 256
/// bytes its resource-table pointer (SGPRs 12:13) addresses for every
/// dispatch of that compute shader.
fn traced_cs() -> Option<u64> {
    static V: OnceLock<Option<u64>> = OnceLock::new();
    *V.get_or_init(|| std::env::var("RPS5E_GPU_TRACE_CS").ok().and_then(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()))
}

fn trace_budget() -> u32 {
    std::env::var("RPS5E_GPU_TRACE").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

impl Backend {
    pub fn new(dev: &Arc<GpuDevice>, memory: &'static GuestGpuMemory) -> Result<Backend, vk::Result> {
        let tracker = Arc::new(GuestPageTracker::new(memory.pages.clone()));
        // The interpreter-side prefetch checks page state without the
        // back-end lock.
        crate::pipeline::frontend::set_tracker(tracker.clone());
        // Read the switches here: the fault handler must not be the first
        // to ask (environment lookups allocate).
        crate::memory::landing_enabled();
        Ok(Backend {
            dev: dev.clone(),
            sched: Scheduler::new(dev)?,
            memory,
            buffers: BufferCache::new(dev, memory, tracker.clone())?,
            images: ImageCache::new(dev, memory)?,
            tracker,
            pipelines: crate::pipeline::default_provider(dev, memory),
            stats: BackendStats::default(),
            compile_pending: false,
            snapshots: Vec::new(),
            trace_left: trace_budget(),
            housekeeping: 0,
            dispatch_counts: HashMap::default(),
            htile: crate::fxhash::HashSet::default(),
            cmask: crate::fxhash::HashSet::default(),
            dcc: HashMap::default(),
            dcc_none: HashMap::default(),
            dcc_targets: HashMap::default(),
            color_keys: Vec::new(),
            depth_keys: Vec::new(),
            dcc_gpu: if std::env::var("RPS5E_DCC_GPU").is_ok_and(|v| v == "0") { None } else { crate::dcc::DccResolver::new(dev) },
            htile_cleared: HashMap::default(),
            htile_kernel_written: HashMap::default(),
            gc: crate::gc::DeviceMemory::new(dev),
            gc_tick: 0,
            gc_ticks: crate::gc::CollectionTicks::default(),
            early: EarlySubmit::default(),
            target_memo: Default::default(),
        })
    }

    fn trace_draw(&mut self, d: &ResolvedDraw) {
        if self.trace_left == 0 {
            return;
        }
        self.trace_left -= 1;
        let colors: Vec<String> = d
            .color_targets
            .iter()
            .map(|c| {
                format!(
                    "s{} {:#x} {}x{} {:?} sw{} dcc={}@{:#x} fc={} cmask={:#x} clr={:#x}",
                    c.slot, c.base, c.width, c.height, c.format.host, c.swizzle, c.dcc_enabled, c.dcc_address, c.fast_clear, c.cmask_address, c.clear_word0
                )
            })
            .collect();
        let depth = d.depth.map(|z| {
            format!(
                "{:#x} {}x{} {:?} htile={}@{:#x} clear={}/{} ({})",
                z.z_read_base, z.width, z.height, z.host_format, z.htile_enabled, z.htile_base, d.depth_state.depth_clear_enable,
                d.depth_state.stencil_clear_enable, d.depth_state.depth_clear_value
            )
        });
        log::info!(
            "[gpu-trace] draw {:?} n={} inst={} prim={} mode={} colors=[{}] depth={:?} scissor={:?} vs={:#x} ps={:#x}",
            d.packet.kind,
            d.packet.count,
            d.packet.instance_count,
            d.primitive_type,
            d.color_mode,
            colors.join("; "),
            depth,
            d.scissor,
            d.shaders.vertex_program_address(),
            d.shaders.ps.address
        );
    }

    fn rect_of(scissor: Rect, w: u32, h: u32) -> Option<vk::Rect2D> {
        if scissor.covers(w, h) || scissor.is_empty() {
            return None;
        }
        let x0 = scissor.x0.max(0);
        let y0 = scissor.y0.max(0);
        let x1 = scissor.x1.min(w as i32);
        let y1 = scissor.y1.min(h as i32);
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        Some(vk::Rect2D { offset: vk::Offset2D { x: x0, y: y0 }, extent: vk::Extent2D { width: (x1 - x0) as u32, height: (y1 - y0) as u32 } })
    }

    /// Fast clears pending for the bound slices of `t`, slice by slice: a
    /// slice whose DCC (or CMASK) bytes all hold one clear code (in guest
    /// memory, or from a recorded fill of the metadata) is cleared to the
    /// value the code stands for, and its keys then mark uncompressed data.
    fn pending_dcc_clears(&mut self, t: &ColorTarget) -> SliceClears {
        profile::time(16, || {
            let mut clears = SliceClears::new();
            let Some(metadata) = t.clear_metadata() else { return clears };
            let base = metadata.0;
            let fill = self.images.take_fill(base);
            // Whether a compute program overwrote the metadata since the
            // last bind, for every bound layer (the first layer's check
            // resets the flag).
            let overwritten = dcc_every_layer_enabled().then(|| self.dcc.get(&base) == Some(&true));
            for layer in t.base_layer..t.base_layer + t.layers.max(1) {
                if let Some(value) = self.pending_dcc_clear_inner(t, metadata, layer, fill, overwritten) {
                    clears.push((layer, value));
                }
            }
            if !clears.is_empty() {
                shadow::hole(shadow::Hole::MetadataClear);
            }
            clears
        })
    }

    /// `metadata`: `t.clear_metadata()`.
    fn pending_dcc_clear_inner(&mut self, t: &ColorTarget, metadata: (u64, u64, bool), layer: u32, fill: Option<crate::images::MetadataFill>, overwritten: Option<bool>) -> Option<vk::ClearColorValue> {
        let (base, slice, cmask) = metadata;
        let addr = base + layer as u64 * slice;
        // Metadata found without a clear code stays so while nothing can
        // have written it and the pages the decision read hold their
        // generations: a CPU write or a write-back landing moves the page
        // (or the GPU-dirty check fires), a fill of this base is consumed
        // before it reaches here, and the compute-overwrite flag is
        // compared directly. GPU_PLAN §56: per-page checks replace the
        // global CPU/GPU epochs, which any guest write anywhere moved
        // (every draw). The format and the clear word are not inputs of
        // the no-clear decision (only of the value a code stands for), so
        // they stay out of the identity — a target reused at the same
        // address with another format still hits.
        let words = [slice, cmask as u64];
        let memo = fill.is_none()
            && overwritten != Some(true)
            && self.dcc.get(&base) == Some(&false)
            && self.dcc_none.get(&addr).is_some_and(|m| {
                // Two kinds of no-clear decision: one read the bytes (its
                // pages hold their generations — a GPU write or a landing
                // bumps them, `change_state`), the other answered from the
                // GPU-dirty state without a read and holds exactly while
                // the range stays dirty (a landing clears it and bumps).
                m.words == words
                    && match m.deps.pages() {
                        0 => self.buffers.has_gpu_dirty(addr, m.checked),
                        _ => m.deps.current(&self.tracker, &self.memory),
                    }
            });
        if memo {
            return None;
        }
        let mut deps = crate::memory::deps::PageDeps::begin(&self.memory);
        let mut checked = 0u64;
        let r = self.pending_dcc_clear_uncached(t, base, addr, slice, cmask, fill, overwritten, &mut deps, &mut checked);
        if r.is_none() && self.dcc.get(&base) == Some(&false) && !deps.overflowed() {
            if self.dcc_none.len() >= 4096 {
                self.dcc_none.clear();
            }
            self.dcc_none.insert(addr, DccNone { words, deps, checked });
        } else {
            self.dcc_none.remove(&addr);
        }
        r
    }

    /// `overwritten`: a compute program overwrote the metadata since the
    /// last bind (None: the flag the first layer resets).
    #[allow(clippy::too_many_arguments)]
    fn pending_dcc_clear_uncached(&mut self, t: &ColorTarget, base: u64, addr: u64, slice: u64, cmask: bool, fill: Option<crate::images::MetadataFill>, overwritten: Option<bool>, deps: &mut crate::memory::deps::PageDeps, checked: &mut u64) -> Option<vk::ClearColorValue> {
        let code = if let Some(fill) = fill {
            // The fill covers this slice whole with one repeated byte.
            let b = fill.value as u8;
            (fill.value == u32::from_ne_bytes([b; 4]) && fill.size >= addr - base + slice).then_some(b)?
        } else {
            // Most binds find compressed metadata: decide on the first bytes
            // before reading the whole slice. Metadata a compute program
            // overwrote without reading it (a clear kernel) is fetched from
            // the GPU once; other GPU-written metadata is compressed data.
            let head_bytes = slice.min(64);
            *checked = head_bytes;
            let flagged = self.dcc.insert(base, false).unwrap_or(false);
            let cleared = overwritten.unwrap_or(flagged);
            if self.buffers.has_gpu_dirty(addr, head_bytes) {
                if !cleared {
                    return None;
                }
                self.stats.dcc_cpu_reads += 1;
                crate::log_once(&format!("ps5gpu: metadata download (overwritten head) {base:#x} slice {slice:#x} cmask={cmask}"));
                if !self.buffers.download(&mut self.sched, addr, head_bytes) {
                    return None;
                }
            }
            deps.note(&self.tracker, addr, head_bytes);
            let mut head = [0u8; 64];
            let head = &mut head[..head_bytes as usize];
            if !self.memory.read_backing(addr, head) || !ColorTarget::is_clear_code(head[0], cmask) || head.iter().any(|&x| x != head[0]) {
                return None;
            }
            if self.buffers.has_gpu_dirty(addr, slice) {
                crate::log_once(&format!("ps5gpu: metadata download (dirty slice) {base:#x} slice {slice:#x} cmask={cmask} head {:#x}", head[0]));
                if !self.buffers.download(&mut self.sched, addr, slice) {
                    return None;
                }
            }
            *checked = slice;
            deps.note(&self.tracker, addr, slice);
            let mut bytes = vec![0u8; slice as usize];
            if !self.memory.read_backing(addr, &mut bytes) {
                return None;
            }
            let b = bytes[0];
            if !ColorTarget::is_clear_code(b, cmask) || bytes.iter().any(|&x| x != b) {
                return None;
            }
            b
        };
        let value = if !cmask && code == crate::render::formats::DCC_CLEAR_SINGLE {
            match self.dcc_single_clear(t, (addr - base) / slice) {
                SingleClear::Value(v) => v,
                SingleClear::Filled => {
                    // The image holds the colour already: nothing to clear.
                    self.consume_metadata_clear(addr, slice);
                    return None;
                }
                SingleClear::Undecided => return None,
            }
        } else {
            t.metadata_clear(code, cmask)?
        };
        self.consume_metadata_clear(addr, slice);
        Some(value)
    }

    /// Layer `layer` of `t` whose DCC keys all hold the single-colour code:
    /// every 256-byte block is the element stored at its start (a clear
    /// writes one colour to every block start). Block starts the GPU wrote
    /// stay there: the layer is filled from the first one on the GPU
    /// (reading it back would wait for the GPU each frame). Block starts in
    /// guest memory give the clear value; the first, middle and last must
    /// agree (blocks of different colours are not a clear).
    fn dcc_single_clear(&mut self, t: &ColorTarget, layer: u64) -> SingleClear {
        const BLOCK: u64 = 256;
        let bpe = t.format.bytes_per_element as u64;
        let blocks = t.slice_size / BLOCK;
        if blocks == 0 || bpe == 0 || bpe > 8 {
            crate::log_once(&format!("ps5gpu: single-colour DCC clear of {:#x} ({:?}, {bpe} bytes per element) not decoded", t.base, t.format.host));
            return SingleClear::Undecided;
        }
        let data = t.base + layer * t.slice_size;
        if self.buffers.has_gpu_dirty(data, bpe) {
            return if self.dcc_single_fill_on_gpu(t, layer, data, bpe) { SingleClear::Filled } else { SingleClear::Undecided };
        }
        let mut element = None;
        for block in [0, blocks / 2, blocks - 1] {
            let at = data + block * BLOCK;
            if self.buffers.has_gpu_dirty(at, bpe) && !self.buffers.download(&mut self.sched, at, bpe) {
                return SingleClear::Undecided;
            }
            let mut bytes = [0u8; 8];
            if !self.memory.read_backing(at, &mut bytes[..bpe as usize]) {
                return SingleClear::Undecided;
            }
            match element {
                None => element = Some(bytes),
                Some(e) if e == bytes => {}
                Some(_) => {
                    crate::log_once(&format!("ps5gpu: single-colour DCC keys of {:#x} over blocks of different colours: not a clear", t.base));
                    return SingleClear::Undecided;
                }
            }
        }
        let Some(e) = element else { return SingleClear::Undecided };
        match t.stored_clear([u32::from_le_bytes(e[0..4].try_into().unwrap()), u32::from_le_bytes(e[4..8].try_into().unwrap())]) {
            Some(v) => SingleClear::Value(v),
            None => SingleClear::Undecided,
        }
    }

    /// [`Self::dcc_single_clear`] of a layer whose first block start the GPU
    /// wrote: the layer of the target's image is filled with that element
    /// on the GPU. Only for single-layer images whose host texels are the
    /// stored elements (no transform); false otherwise.
    fn dcc_single_fill_on_gpu(&mut self, t: &ColorTarget, layer: u64, element: u64, bpe: u64) -> bool {
        if t.format.transform() != crate::tiler::ColorTransform::None {
            return false;
        }
        let Ok(id) = self.images.find_or_create(&mut self.sched, &mut self.buffers, &color_key(t)) else { return false };
        if self.images.get(id).is_none_or(|i| i.key.slices() != 1) || layer != 0 {
            return false;
        }
        let Ok(src) = self.buffers.obtain(&mut self.sched, element, bpe, false) else { return false };
        // The fill covers the whole image: its memory is the GPU's.
        self.images.prepare(&mut self.sched, &mut self.buffers, id, true);
        if !self.images.fill_layer_with_element(&mut self.sched, id, 0, (src.buffer, src.offset), bpe) {
            return false;
        }
        self.stats.color_clears += 1;
        if crate::pipeline::trace::active() {
            crate::pipeline::trace::log(&format!("    (DCC single-colour clear of color target #{id} {:#x} from its first block on the GPU)", t.base));
        }
        true
    }

    /// Consume a metadata clear of `[addr, addr+slice)`: the metadata now
    /// describes uncompressed data (0xFF). When the GPU wrote those bytes
    /// (the clear came from a fill or kernel on the GPU) its copy is the
    /// current one: the 0xFF goes there, after that work, or the GPU copy
    /// later lands over guest memory with the clear code again and the
    /// next bind clears the target a second time.
    fn consume_metadata_clear(&mut self, addr: u64, slice: u64) {
        shadow::hole(shadow::Hole::ConsumeClear);
        shadow::changed(addr, slice);
        if consume_on_gpu_enabled() && self.buffers.has_gpu_dirty(addr, slice) {
            match self.buffers.fill(&mut self.sched, addr, slice, u32::MAX) {
                Ok(true) => return,
                Ok(false) => {}
                Err(e) => crate::log_once(&format!("ps5gpu: metadata consume fill {addr:#x}+{slice:#x}: {e:?}")),
            }
        }
        let ones = vec![0xFFu8; slice as usize];
        if self.memory.write_backing(addr, &ones) {
            self.tracker.mark_cpu_dirty_pages(addr, slice);
        }
    }

    /// Stores through image descriptors with DCC compression on (T#
    /// COMPRESSION_EN) rewrote the keys of the blocks they wrote: on the
    /// hardware no fast-clear code is left over those surfaces. A layer
    /// whose keys still hold one clear code is spent as a consumed clear
    /// (its keys become uncompressed, 0xFF), so the next bind of the
    /// surface as a colour target does not clear what the stores wrote; a
    /// pending fill of the keys and the compute-overwrite flag are spent
    /// too. Keys the GPU wrote (a fill on the GPU) are made uncompressed
    /// there, after the stores.
    fn compressed_stores_done(&mut self) {
        for k in self.pipelines.take_compressed_stores() {
            self.images.take_fill(k.address);
            if let Some(flag) = self.dcc.get_mut(&k.address) {
                *flag = false;
            }
            let total = k.slice * k.layers as u64;
            if self.buffers.has_gpu_dirty(k.address, total) {
                self.consume_metadata_clear(k.address, total);
                continue;
            }
            let mut bytes = vec![0u8; k.slice as usize];
            for layer in 0..k.layers as u64 {
                let addr = k.address + layer * k.slice;
                let mut head = [0u8; 1];
                if !self.memory.read_backing(addr, &mut head) || !crate::render::formats::is_dcc_clear_code(head[0]) {
                    continue;
                }
                if self.memory.read_backing(addr, &mut bytes) && bytes.iter().all(|&b| b == head[0]) {
                    self.stats.dcc_store_spent += 1;
                    self.consume_metadata_clear(addr, k.slice);
                }
            }
        }
    }

    /// Bind (find, clear or upload) the draw's render targets.
    fn bind_targets(&mut self, d: &ResolvedDraw) -> DrawTargets {
        let mut targets = DrawTargets { colors: Default::default(), depth: None };
        // Everything the binding below depends on: with the same targets
        // under the same state the same images bind (the draws of one
        // pass, whose own GPU writes move none of it: writing the image
        // that already holds the newest contents changes no lookup).
        // GPU_PLAN §56: the open tick and the global CPU-dirty, sync and
        // GPU-write epochs move on every draw (any guest write anywhere),
        // so the memo never hit; with per-target generations the hit path
        // validates each image instead and the poison fields hold 0.
        let tick = self.sched.current_tick();
        let per_target = target_generations_enabled();
        let stamp = (
            if per_target { 0 } else { tick },
            self.images.lookup_epoch(),
            if per_target { 0 } else { self.images.generations_value() },
            if per_target { 0 } else { self.tracker.cpu_dirty_epoch() },
            if per_target { 0 } else { crate::memory::sync_generation() },
            if per_target { 0 } else { self.buffers.gpu_write_epoch() },
            self.dcc.len(),
            self.dcc_none.len(),
            self.cmask.len(),
            self.htile.len(),
            self.htile_cleared.len(),
        );
        let colors_bound = d.color_mode != color_mode::DISABLE;
        let memo_hit = colors_bound
            && self.target_memo.stamp == stamp
            && self.target_memo.colors.len() == d.color_targets.len()
            && self.target_memo.colors.iter().zip(&d.color_targets).all(|(a, b)| a.0 == *b)
            && self
                .target_memo
                .colors
                .iter()
                .zip(&self.target_memo.gens)
                .all(|(&(_, id), &g)| self.images.image_generation(id) == Some(g));
        if memo_hit {
            targets.colors = self.target_memo.colors.clone();
            for &(t, id) in &targets.colors {
                self.images.touch(id, tick);
                // The fill and metadata-write decoders read these by
                // address; another target of the same address may have
                // rebound in between, so the entries are re-taken (as on
                // the miss path).
                if t.dcc_address != 0 {
                    self.dcc_targets.insert(t.dcc_address, t);
                }
                if let Some((base, _, true)) = t.clear_metadata() {
                    self.cmask.insert(base);
                    self.dcc_targets.insert(base, t);
                }
                // The DCC decision is still taken per draw: metadata a
                // write-back landed in can newly describe a clear, and its
                // memo is per page. `prepare` is not
                // repeated: the store condition (`upload_covered`) plus
                // the lookup epoch is exactly its upload condition.
                let clears = self.pending_dcc_clears(&t);
                if !clears.is_empty() {
                    self.apply_slice_clears(id, &clears);
                }
            }
        } else if colors_bound {
            let mut store = true;
            for t in &d.color_targets {
                let key = profile::time(20, || recent_key(&mut self.color_keys, t, color_key));
                let id = match profile::time(20, || self.images.find_or_create(&mut self.sched, &mut self.buffers, &key)) {
                    Ok(id) => id,
                    Err(e) => {
                        crate::log_once(&format!("ps5gpu: color target image {:?} {}x{}: {e:?}", key.format, key.width, key.height));
                        self.stats.errors += 1;
                        continue;
                    }
                };
                if t.dcc_address != 0 {
                    self.dcc_targets.insert(t.dcc_address, *t);
                }
                if let Some((base, _, true)) = t.clear_metadata() {
                    self.cmask.insert(base);
                    // A compute program overwriting the CMASK decides its
                    // clear (on the GPU) like a DCC clear.
                    self.dcc_targets.insert(base, *t);
                }
                let clears = self.pending_dcc_clears(t);
                if clears.is_empty() {
                    if crate::pipeline::trace::active() && self.images.needs_upload(id) {
                        crate::pipeline::trace::log(&format!("    (reload color target #{id} {:#x} from guest memory)", t.base));
                    }
                    self.images.prepare(&mut self.sched, &mut self.buffers, id, false);
                } else {
                    if crate::pipeline::trace::active() {
                        // SAFETY: every member of the clear value union is plain data.
                        let values: Vec<(u32, [u32; 4])> = clears.iter().map(|(l, v)| (*l, unsafe { v.uint32 })).collect();
                        crate::pipeline::trace::log(&format!("    (DCC fast clear of color target #{id} {:#x}: {values:x?})", t.base));
                    }
                    self.apply_slice_clears(id, &clears);
                }
                targets.colors.push((*t, id));
                // An unwatched image without GPU data must keep its
                // per-draw upload check.
                store &= self.images.upload_covered(id);
            }
            if store {
                self.target_memo.colors = targets.colors.clone();
                self.target_memo.gens = targets.colors.iter().map(|&(_, id)| self.images.image_generation(id).unwrap_or(u64::MAX)).collect();
            } else {
                self.target_memo.colors.clear();
                self.target_memo.gens.clear();
            }
        } else {
            self.target_memo.colors.clear();
            self.target_memo.gens.clear();
        }
        if let Some(z) = d.depth {
            // The image of the same depth target under the same state
            // (the clear handling below still runs per draw).
            let memo_id = (self.target_memo.stamp == stamp)
                .then(|| self.target_memo.depth.filter(|(m, _)| *m == z).map(|(_, id)| id))
                .flatten()
                .filter(|&id| !per_target || self.images.image_generation(id) == self.target_memo.depth_gen);
            let key = recent_key(&mut self.depth_keys, &z, depth_key);
            let found = match memo_id {
                Some(id) => Ok(id),
                None => self.images.find_or_create(&mut self.sched, &mut self.buffers, &key),
            };
            match found {
                Ok(id) => {
                    if z.htile_enabled {
                        self.htile.insert(z.htile_base);
                    }
                    let (htile_fill, htile_stencil) = if z.htile_enabled { self.htile_slice_cleared(&z, z.base_layer) } else { (false, false) };
                    if htile_fill || htile_stencil {
                        shadow::hole(shadow::Hole::MetadataClear);
                    }
                    let ds = &d.depth_state;
                    let whole = htile_whole_clear_enabled();
                    // A clear draw clears within its scissor (nothing when
                    // it is empty); cleared HTILE clears the whole surface
                    // (a clearing load of the attachment).
                    let s = d.scissor;
                    let scissor_empty = whole && (s.is_empty() || s.x1.min(z.width as i32) <= s.x0.max(0) || s.y1.min(z.height as i32) <= s.y0.max(0));
                    let depth_clear = (ds.depth_clear_enable && !scissor_empty) || htile_fill;
                    let stencil_clear = z.has_stencil && ((ds.stencil_clear_enable && !ds.stencil_write_disable && !scissor_empty) || htile_stencil);
                    if depth_clear || stencil_clear {
                        // The bound slices are cleared.
                        let draw_scissor = Self::rect_of(d.scissor, z.width, z.height);
                        let depth_rect = if htile_fill && whole { None } else { draw_scissor };
                        let scissor = if htile_stencil && whole { None } else { draw_scissor };
                        let same = |a: Option<vk::Rect2D>, b: Option<vk::Rect2D>| match (a, b) {
                            (None, None) => true,
                            (Some(a), Some(b)) => (a.offset.x, a.offset.y, a.extent.width, a.extent.height) == (b.offset.x, b.offset.y, b.extent.width, b.extent.height),
                            _ => false,
                        };
                        let layers = (z.base_layer, z.layers.max(1));
                        let all_layers = z.base_layer == 0 && self.images.get(id).is_some_and(|i| i.key.layers <= z.layers.max(1));
                        let discard = all_layers && depth_clear && depth_rect.is_none() && (!z.has_stencil || (stencil_clear && scissor.is_none()));
                        self.images.prepare(&mut self.sched, &mut self.buffers, id, discard);
                        let (depth_value, stencil_value) = (depth_clear.then_some(ds.depth_clear_value), stencil_clear.then_some(ds.stencil_clear_value));
                        if !depth_clear || !stencil_clear || same(depth_rect, scissor) {
                            let rect = if depth_clear { depth_rect } else { scissor };
                            self.images.clear_depth_stencil(&mut self.sched, id, depth_value, stencil_value, rect, Some(layers));
                        } else {
                            self.images.clear_depth_stencil(&mut self.sched, id, depth_value, None, depth_rect, Some(layers));
                            self.images.clear_depth_stencil(&mut self.sched, id, None, stencil_value, scissor, Some(layers));
                        }
                        self.stats.depth_clears += 1;
                    } else {
                        if crate::pipeline::trace::active() && self.images.needs_upload(id) {
                            crate::pipeline::trace::log(&format!("    (reload depth target #{id} {:#x} from guest memory)", z.z_read_base));
                        }
                        self.images.prepare(&mut self.sched, &mut self.buffers, id, false);
                    }
                    targets.depth = Some((z, id));
                }
                Err(e) => {
                    crate::log_once(&format!("ps5gpu: depth target image {:?} {}x{}: {e:?}", key.format, key.width, key.height));
                    self.stats.errors += 1;
                }
            }
        }
        // Remember (or keep) the memo for the next draw.
        self.target_memo.depth = targets.depth;
        self.target_memo.depth_gen = targets.depth.map(|(_, id)| self.images.image_generation(id).unwrap_or(u64::MAX));
        self.target_memo.stamp = stamp;
        drop_mixed_sample_targets(&mut targets, d);
        targets
    }

    /// The (depth, stencil) planes of `z`'s slice `slice` an HTILE fill
    /// cleared since its last bind (a fill clears every slice; each bind
    /// consumes its own).
    fn htile_slice_cleared(&mut self, z: &crate::render::DepthTarget, slice: u32) -> (bool, bool) {
        let stencil_in_htile = z.has_stencil && (z.stencil_info >> 29) & 1 == 0;
        // Each slice of a depth array has its own HTILE: slice s at
        // base + s * bytes.
        let bytes = htile_metadata_bytes(z.width, z.height);
        let per_slice = htile_per_slice_enabled();
        let base = z.htile_base;
        let mut fills = smallvec::SmallVec::<[(u32, crate::images::MetadataFill); 2]>::new();
        if let Some(fill) = self.images.take_fill(base) {
            fills.push((0, fill));
        }
        if per_slice && slice > 0 && slice < DEPTH_VIEW_SLICES {
            if let Some(fill) = self.images.take_fill(base + slice as u64 * bytes) {
                fills.push((slice, fill));
            }
        }
        for (first, fill) in fills {
            let (depth, stencil) = if htile_fill_decode_enabled() {
                if fill.size >= bytes {
                    htile_fill_clears(fill.value, stencil_in_htile)
                } else {
                    (false, false)
                }
            } else {
                (true, false)
            };
            // The slices the fill spans (every slice: the earlier rule).
            let n = if per_slice { (fill.size / bytes).min((DEPTH_VIEW_SLICES - first) as u64) as u32 } else { DEPTH_VIEW_SLICES };
            if (depth || stencil) && htile_store_back_enabled() && fill.size >= bytes {
                let word = expanded_htile_word(fill.value, stencil_in_htile, depth, stencil);
                let spanned = if per_slice { n as u64 * bytes } else { bytes };
                self.store_back_htile(base + first as u64 * bytes, spanned, &[word]);
            }
            let masks = self.htile_cleared.entry(base).or_insert_with(|| (SliceMask::new(false), SliceMask::new(false)));
            if per_slice {
                masks.0.set_range(first, n, depth);
                masks.1.set_range(first, n, stencil);
            } else {
                *masks = (SliceMask::new(depth), SliceMask::new(stencil));
            }
        }
        if let Some(written) = self.htile_kernel_written.remove(&base) {
            // A compute program stored the words: they decide, slice by
            // slice (the GPU copy is read back once per such store).
            let n = if per_slice { (written / bytes).min(MAX_KERNEL_HTILE_SLICES) as u32 } else { (written >= bytes) as u32 };
            let mut cleared_slices: Vec<(bool, bool)> = Vec::new();
            if n != 0 {
                let total = n as u64 * bytes;
                self.buffers.download(&mut self.sched, base, total);
                let mut raw = vec![0u8; total as usize];
                if self.memory.read_backing(base, &mut raw) {
                    let mut words: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    let per = (bytes / 4) as usize;
                    let mut changed = false;
                    for (k, chunk) in words.chunks_mut(per).enumerate() {
                        let cleared = htile_words_clear(chunk, stencil_in_htile, k == 0);
                        if (cleared.0 || cleared.1) && htile_store_back_enabled() {
                            for w in chunk.iter_mut() {
                                *w = expanded_htile_word(*w, stencil_in_htile, cleared.0, cleared.1);
                            }
                            changed = true;
                        }
                        cleared_slices.push(cleared);
                    }
                    if changed {
                        self.store_back_htile(base, total, &words);
                    }
                }
            }
            if crate::pipeline::trace::active() {
                crate::pipeline::trace::log(&format!("    (HTILE {base:#x} written by a compute program: slices cleared (depth, stencil) {cleared_slices:?})"));
            }
            let masks = self.htile_cleared.entry(base).or_insert_with(|| (SliceMask::new(false), SliceMask::new(false)));
            if per_slice {
                for (k, &(d, s)) in cleared_slices.iter().enumerate() {
                    masks.0.set_range(k as u32, 1, d);
                    masks.1.set_range(k as u32, 1, s);
                }
            } else {
                let (d, s) = cleared_slices.first().copied().unwrap_or((false, false));
                *masks = (SliceMask::new(d), SliceMask::new(s));
            }
        }
        let Some((depth, stencil)) = self.htile_cleared.get_mut(&base) else { return (false, false) };
        (depth.take(slice), stencil.take(slice))
    }

    /// Write `words` (one word: every word holds it) over the `bytes` of
    /// HTILE at `base`: on the GPU when it holds the newest bytes, else in
    /// guest memory.
    fn store_back_htile(&mut self, base: u64, bytes: u64, words: &[u32]) {
        shadow::hole(shadow::Hole::ConsumeClear);
        shadow::changed(base, bytes);
        if let [word] = words {
            if self.buffers.has_gpu_dirty(base, bytes) {
                match self.buffers.fill(&mut self.sched, base, bytes, *word) {
                    Ok(true) => return,
                    Ok(false) => {}
                    Err(e) => crate::log_once(&format!("ps5gpu: HTILE store-back fill {base:#x}+{bytes:#x}: {e:?}")),
                }
            }
            let data: Vec<u8> = std::iter::repeat_n(word.to_le_bytes(), (bytes / 4) as usize).flatten().collect();
            if self.memory.write_backing(base, &data) {
                self.tracker.mark_cpu_dirty_pages(base, bytes);
            }
            return;
        }
        // The words were read back: guest memory is current.
        let data: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        if self.memory.write_backing(base, &data) {
            self.tracker.mark_cpu_dirty_pages(base, bytes);
        }
    }

    /// Apply per-slice fast clears to colour image `id`: the image is
    /// loaded first unless the clears cover all of it; runs of layers with
    /// the same value are cleared together.
    fn apply_slice_clears(&mut self, id: u64, clears: &[(u32, vk::ClearColorValue)]) {
        let slices = self.images.get(id).map_or(0, |i| i.key.slices());
        let whole = clears.len() as u32 == slices && clears.iter().enumerate().all(|(i, (l, _))| *l == i as u32);
        self.images.prepare(&mut self.sched, &mut self.buffers, id, whole);
        for (first, n, i) in clear_runs(clears) {
            self.images.clear_color_layers(&mut self.sched, id, clears[i].1, first, n);
        }
        self.stats.color_clears += 1;

    }

    /// Color metadata operations: a pending fast clear (DCC clear codes or
    /// a recorded CMASK fill) is materialised into the image.
    fn metadata_operation(&mut self, d: &ResolvedDraw) {
        self.stats.metadata_ops += 1;
        for t in &d.color_targets {
            let key = color_key(t);
            let Ok(id) = self.images.find_or_create(&mut self.sched, &mut self.buffers, &key) else { continue };
            // CMASK holds the fast-clear state only in the configuration
            // `cmask_slice_size` describes (no DCC, one mip; multisampled
            // targets too).
            let cmask_slice = t.cmask_slice_size();
            let cmask_fill = if t.fast_clear && t.cmask_address != 0 && (cmask_slice != 0 || !consume_on_gpu_enabled()) { self.images.take_fill(t.cmask_address) } else { None };
            let cmask_clear = cmask_fill.is_some_and(|f| f.value == 0);
            let clears = self.pending_dcc_clears(t);
            if !clears.is_empty() {
                self.apply_slice_clears(id, &clears);
            } else if let Some(v) = cmask_clear.then(|| t.register_clear()).flatten() {
                self.images.prepare(&mut self.sched, &mut self.buffers, id, true);
                self.images.clear_color(&mut self.sched, id, v, None);
                self.stats.color_clears += 1;
                // The elimination leaves the CMASK expanded: a later bind
                // must not find the zero fill and clear the target again.
                if consume_on_gpu_enabled() {
                    let layers = self.images.get(id).map_or(1, |i| i.key.slices()) as u64;
                    let bytes = cmask_fill.map_or(0, |f| f.size).min(cmask_slice * layers) & !3;
                    if bytes != 0 {
                        self.consume_metadata_clear(t.cmask_address, bytes);
                    }
                }
            } else {
                self.images.prepare(&mut self.sched, &mut self.buffers, id, false);
            }
        }
    }

    fn resolve(&mut self, d: &ResolvedDraw) {
        let Some((src, dst)) = d.resolve_targets else { return };
        // Resolving a surface level into itself has nothing to do.
        if src.base == dst.base && src.mip_level == dst.mip_level && src.base_layer == dst.base_layer {
            return;
        }
        self.stats.resolves += 1;
        let (Ok(s), Ok(t)) = (
            self.images.find_or_create(&mut self.sched, &mut self.buffers, &color_key(&src)),
            self.images.find_or_create(&mut self.sched, &mut self.buffers, &color_key(&dst)),
        ) else {
            return;
        };
        self.images.prepare(&mut self.sched, &mut self.buffers, s, false);
        // The destination is overwritten whole only when the resolve covers it.
        let whole = dst.mip_levels == 1 && dst.base_layer == 0 && src.layers >= dst.layers;
        self.images.prepare(&mut self.sched, &mut self.buffers, t, whole);
        self.images.resolve(&mut self.sched, s, (src.mip_level, src.base_layer), t, (dst.mip_level, dst.base_layer), src.layers.min(dst.layers));
    }

    /// A draw with color disabled whose DB_RENDER_OVERRIDE forces the depth
    /// or stencil plane valid and dirty, with distinct read and write
    /// bases, copies the plane from the read surface to the write surface
    /// instead of drawing. True when it did.
    fn depth_stencil_copy(&mut self, d: &ResolvedDraw) -> bool {
        let Some(z) = d.depth else { return false };
        if d.color_mode != color_mode::DISABLE {
            return false;
        }
        let (depth_flag, stencil_flag) = d.depth_state.depth_copy();
        let depth = depth_flag && z.z_format != crate::render::formats::depth::INVALID && z.z_read_base != 0 && z.z_write_base != 0 && z.z_read_base != z.z_write_base;
        let stencil = stencil_flag && z.has_stencil && z.stencil_read_base != 0 && z.stencil_write_base != 0 && z.stencil_read_base != z.stencil_write_base;
        if !depth && !stencil {
            return false;
        }
        let write = crate::render::DepthTarget { z_read_base: z.z_write_base, stencil_read_base: z.stencil_write_base, ..z };
        let (Ok(src), Ok(dst)) = (
            self.images.find_or_create(&mut self.sched, &mut self.buffers, &depth_key(&z)),
            self.images.find_or_create(&mut self.sched, &mut self.buffers, &depth_key(&write)),
        ) else {
            return false;
        };
        self.images.prepare(&mut self.sched, &mut self.buffers, src, false);
        // Both planes copied: the destination is overwritten whole.
        self.images.prepare(&mut self.sched, &mut self.buffers, dst, depth && (stencil || !z.has_stencil));
        let mut aspect = vk::ImageAspectFlags::empty();
        if depth {
            aspect |= vk::ImageAspectFlags::DEPTH;
        }
        if stencil {
            aspect |= vk::ImageAspectFlags::STENCIL;
        }
        self.images.copy_depth_stencil(&mut self.sched, src, dst, aspect)
    }

    /// A DB depth copy ([`ResolvedDraw::db_depth_copy`]) copies the depth
    /// plane into colour target 0 instead of drawing. True when it did. A
    /// single-sample plane is copied; a sample of a multisampled plane is
    /// not (it needs a sample fetch pass) and the draw runs as drawn.
    fn db_depth_copy(&mut self, d: &ResolvedDraw) -> bool {
        if !db_depth_copy_enabled() {
            return false;
        }
        let Some(copy) = d.db_depth_copy() else { return false };
        let why = match copy {
            Ok(_) if d.depth.is_some_and(|z| z.samples != 1) => "of a multisampled depth plane",
            Ok(_) => {
                let (Some(z), Some(t)) = (d.depth, d.color_targets.iter().find(|t| t.slot == 0)) else { return false };
                let (src, dst) = match (
                    self.images.find_or_create(&mut self.sched, &mut self.buffers, &depth_key(&z)),
                    self.images.find_or_create(&mut self.sched, &mut self.buffers, &crate::images::color_key(t)),
                ) {
                    (Ok(s), Ok(c)) => (s, c),
                    _ => return false,
                };
                self.images.prepare(&mut self.sched, &mut self.buffers, src, false);
                self.images.prepare(&mut self.sched, &mut self.buffers, dst, false);
                if self.images.copy_depth_to_color(&mut self.sched, src, z.base_layer, dst, t.base_layer) {
                    return true;
                }
                "between images the copy cannot join"
            }
            Err(why) => why,
        };
        crate::log_once(&format!("ps5gpu: DB depth copy {why} drawn as a draw"));
        false
    }

    /// Whether the last [`Backend::draw`] or [`Backend::dispatch`] was given
    /// up unrecorded for a compile it must wait for (only while
    /// [`crate::pipeline::unlocked_wait`] allows it): the caller releases
    /// the back-end, waits and calls it again.
    pub fn take_compile_pending(&mut self) -> bool {
        std::mem::take(&mut self.compile_pending)
    }

    pub fn draw(&mut self, d: &ResolvedDraw) {
        let mem = self.memory;
        crate::capture::around(mem, || self.draw_inner(d), |_| Event::Draw(Box::new(d.clone())))
    }

    fn draw_inner(&mut self, d: &ResolvedDraw) {
        prefetch_value(d);
        self.stats.draws += 1;
        crate::perf::DRAWS.fetch_add(1, Ordering::Relaxed);
        self.trace_draw(d);
        if self.depth_stencil_copy(d) || self.db_depth_copy(d) {
            shadow::hole(shadow::Hole::MetaOp);
            return;
        }
        if shadow::enabled() && self.sched.occlusion_active() {
            shadow::hole(shadow::Hole::Occlusion);
        }
        match d.color_mode {
            color_mode::ELIMINATE_FAST_CLEAR | color_mode::FMASK_DECOMPRESS | color_mode::DCC_DECOMPRESS => {
                shadow::hole(shadow::Hole::MetaOp);
                let t = self.sched.time_item_begin(gpu_label::METADATA);
                self.metadata_operation(d);
                self.sched.time_item_end(t);
                return;
            }
            color_mode::RESOLVE => {
                shadow::hole(shadow::Hole::MetaOp);
                let t = self.sched.time_item_begin(gpu_label::RESOLVE);
                self.resolve(d);
                self.sched.time_item_end(t);
                return;
            }
            _ => {}
        }
        // `RPS5E_TRACE_DRAW=<addr>`: every draw of the program at `addr`
        // logged with its packet (diagnostics; no cost when unset).
        static TRACE_DRAW: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
        let traced = *TRACE_DRAW.get_or_init(|| std::env::var("RPS5E_TRACE_DRAW").ok().and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok()));
        if traced.is_some_and(|a| a == d.shaders.vertex_program_address() || a == d.shaders.ps.address) {
            log::info!(
                "[trace-draw] vs={:#x} ps={:#x} kind={:?} count={} instances={} index={:?} indirect={:#x} topology={:?} colors={}",
                d.shaders.vertex_program_address(),
                d.shaders.ps.address,
                d.packet.kind,
                d.packet.count,
                d.packet.instance_count,
                d.index.map(|i| (i.kind, i.address, i.bytes())),
                d.packet.indirect_address,
                d.primitive_type,
                d.color_targets.len()
            );
        }
        let t = self.sched.time_item_begin(gpu_label::TARGETS);
        let targets = profile::time(8, || self.bind_targets(d));
        self.sched.time_item_end(t);
        if shadow::enabled() {
            for &(_, id) in targets.colors.iter() {
                shadow::read_image(id, self.images.guest_range(id));
            }
            if let Some((_, id)) = targets.depth {
                shadow::read_image(id, self.images.guest_range(id));
            }
        }
        if crate::pipeline::trace::active() && std::env::var_os("RPS5E_FRAME_TRACE_BEFORE").is_some() {
            // Debug: the colour targets as the pass finds them.
            let seq = crate::pipeline::trace::peek_seq();
            for &(t, id) in &targets.colors {
                let what = format!("pre{}", t.slot);
                if crate::pipeline::trace::dump_wanted(seq, &what) {
                    crate::pipeline::trace::dump_image(&self.dev, &mut self.sched, &mut self.images, id, seq, &what);
                }
            }
        }
        if targets.colors.is_empty() && targets.depth.is_none() && !d.shaders.ps.bound() {
            self.stats.targetless_draws += 1;
        }
        if skipped_shader(&[d.shaders.vertex_program_address(), d.shaders.ps.address]) {
            return;
        }
        self.isolate_preparation("draw");
        // GPU time under the pixel shader's address (its cost dominates
        // most draws), the vertex program's without one.
        let label = if d.shaders.ps.bound() { d.shaders.ps.address } else { d.shaders.vertex_program_address() };
        let timed = self.sched.time_item_begin(label);
        let mut ctx = DrawContext { dev: &self.dev, sched: &mut self.sched, buffers: (&mut self.buffers).into(), images: (&mut self.images).into(), targets: &targets };
        let outcome = self.pipelines.draw(&mut ctx, d);
        self.sched.time_item_end(timed);
        if self.pipelines.take_compile_pending() {
            // Nothing recorded: the caller tries the draw again once the
            // compile it waits for is further along.
            self.stats.draws -= 1;
            crate::perf::DRAWS.fetch_sub(1, Ordering::Relaxed);
            self.compile_pending = true;
            return;
        }
        match outcome {
            DrawOutcome::NeedsShader => self.stats.needs_shader_draws += 1,
            DrawOutcome::Recorded => {
                self.stats.recorded_draws += 1;
                let sh = &d.shaders;
                self.sched.note(crate::scheduler::Note::Draw {
                    count: d.packet.count,
                    instances: d.packet.instance_count,
                    vs: sh.vertex_program_address(),
                    ps: sh.ps.address,
                    colors: targets.colors.len(),
                    depth: targets.depth.is_some(),
                });
                if submit_each() {
                    self.submit_isolated();
                }
                let tick = self.sched.current_tick();
                for &(_, id) in &targets.colors {
                    self.images.mark_gpu_written(id, tick);
                }
                if let Some((z, id)) = targets.depth {
                    let has_depth = z.z_format != crate::render::formats::depth::INVALID;
                    let (depth, stencil) = crate::pipeline::state::depth_attachment_writes(d, has_depth, z.has_stencil);
                    if depth || stencil {
                        self.images.mark_gpu_written(id, tick);
                    }
                }
            }
            DrawOutcome::Skipped | DrawOutcome::Hole => {}
        }
        self.compressed_stores_done();
        // The commit evaluates these (R7b), not the slice thread.
        shadow::commit_side(|| {
            let t = self.sched.time_item_begin(gpu_label::HOUSEKEEPING);
            self.housekeeping();
            self.sched.time_item_end(t);
            if outcome == DrawOutcome::Recorded {
                self.early_submit(false);
            }
        });
    }

    /// After a recorded draw or dispatch: submit the open tick when enough
    /// work went into it ([`EarlyPolicy`]), so the GPU starts on a long
    /// slice before it was recorded whole, and CPU accesses waiting for its
    /// write-throughs wait for the work before them only. Between items:
    /// nothing of the next one is recorded yet.
    fn early_submit(&mut self, dispatch: bool) {
        let Some(policy) = early_policy() else { return };
        if !self.sched.is_recording() {
            self.early = EarlySubmit::default();
            return;
        }
        let tick = self.sched.current_tick();
        let now = std::time::Instant::now();
        if self.early.tick != tick || self.early.since.is_none() {
            self.early = EarlySubmit { tick, draws: 0, dispatches: 0, since: Some(now) };
        }
        if dispatch {
            self.early.dispatches += 1;
        } else {
            self.early.draws += 1;
        }
        let micros = self.early.since.map_or(0, |t| now.duration_since(t).as_micros() as u64);
        let scope_open = self.sched.rendering_scope().is_some();
        // Write-throughs waiting for the submit: CPU accesses wait for them.
        let through = self.buffers.has_pending_through();
        // A CPU access waiting for the open buffer. Any
        // outstanding request names a recorded landing tick, at most the
        // open one — submitting it satisfies or overshoots the request.
        let request = submit_request_enabled()
            .then(|| SUBMIT_REQUESTED.load(Ordering::Acquire))
            .is_some_and(|r| r != 0);
        // A staging ring nearly full would stall its next allocation on the
        // GPU: the submit recycles it (inside a rendering scope after the
        // short through threshold — the pass churn is cheaper than the
        // wait, and it only happens under real pressure).
        let (used, size) = self.pipelines.staging_usage();
        let pressure = size > 0 && used * 4 >= size * 3;
        if policy.due(self.early.draws, self.early.dispatches, micros, scope_open, through || pressure, request) {
            self.stats.early_submits += 1;
            self.stats.submit_requests += request as u64;
            self.flush_inner();
            self.early = EarlySubmit::default();
            // The flush submitted the open tick: every request up to it is
            // satisfied (a request set meanwhile, for a later tick, stays).
            loop {
                let r = SUBMIT_REQUESTED.load(Ordering::Acquire);
                if r == 0 || r > tick {
                    break;
                }
                if SUBMIT_REQUESTED.compare_exchange(r, 0, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                    break;
                }
            }
        }
    }

    /// The SDK fill kernel (64 threads, each storing the 16-byte pattern
    /// of user data 4..7 through the V# in user data 0..3): `(base, bytes,
    /// pattern)` when this dispatch runs it.
    fn fill_kernel(&self, d: &ResolvedDispatch) -> Option<(u64, u64, [u32; 4])> {
        const CODE: [u32; 9] = [0xd746_0004, 0x0401_0c08, 0x7e00_0204, 0x7e02_0205, 0x7e04_0206, 0x7e06_0207, 0xe01c_2000, 0x8000_0004, 0xbf81_0000];
        static OFF: OnceLock<bool> = OnceLock::new();
        if *OFF.get_or_init(|| std::env::var("RPS5E_FILL_HLE").is_ok_and(|v| v == "0")) {
            return None;
        }
        if d.threads != [64, 1, 1] || d.packet.indirect_address != 0 || d.start != [0, 0, 0] {
            return None;
        }
        let u = &d.shader.user_data;
        let stride = (u[1] >> 16) & 0x3FFF;
        let swizzled = (u[1] >> 31) & 1 != 0;
        let dst_sel = u[3] & 0xFFF;
        let add_tid = (u[3] >> 23) & 1 != 0;
        let format = (u[3] >> 12) & 0x7F;
        if u[3] >> 30 != 0 || stride != 16 || swizzled || add_tid || dst_sel != 0xFAC || format != 0x4B {
            return None;
        }
        let mut code = [0u8; 36];
        if !self.memory.read_backing(d.shader.address, &mut code) {
            return None;
        }
        if code.chunks(4).zip(CODE).any(|(c, w)| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) != w) {
            return None;
        }
        let mut groups = d.packet.groups;
        if d.packet.dispatch_initiator & 0x20 != 0 {
            for (g, t) in groups.iter_mut().zip(d.threads) {
                *g = g.div_ceil(t.max(1));
            }
        }
        if groups[1] != 1 || groups[2] != 1 {
            return None;
        }
        let records = (groups[0] as u64 * 64).min(u[2] as u64);
        let base = u[0] as u64 | ((u[1] & 0xFFFF) as u64) << 32;
        Some((base, records * 16, [u[4], u[5], u[6], u[7]]))
    }

    /// The SDK pattern-fill kernel ([`crate::fill_kernels`]) with a zero
    /// mask: `Some(None)` when it stores nothing, `Some(Some((base, bytes,
    /// value)))` for a fill with one repeated dword. The count, mask and
    /// pattern are read from guest memory now; when the GPU wrote them and
    /// they are not back yet the dispatch runs instead (no wait here).
    /// `RPS5E_PATTERN_FILL_HLE=0` (or `RPS5E_FILL_HLE=0`) runs it as a
    /// dispatch.
    fn pattern_fill_kernel(&self, d: &ResolvedDispatch) -> Option<Option<(u64, u64, u32)>> {
        use crate::fill_kernels::{decode_pattern_fill, matches_pattern_fill, ComputeShape, PATTERN_FILL_KERNEL};
        static OFF: OnceLock<bool> = OnceLock::new();
        let off = *OFF.get_or_init(|| {
            let zero = |name: &str| std::env::var(name).is_ok_and(|v| v == "0");
            zero("RPS5E_PATTERN_FILL_HLE") || zero("RPS5E_FILL_HLE")
        });
        if off || d.packet.indirect_address != 0 || d.start != [0, 0, 0] || d.shader.user_sgpr_count() != 12 {
            return None;
        }
        let user = &d.shader.user_data[..12];
        let shape = ComputeShape::from_rsrc2(d.threads, d.shader.rsrc2);
        let mut bytes = [0u8; PATTERN_FILL_KERNEL.len() * 4];
        // Cheap checks before the code read.
        if shape.threads != [64, 1, 1] || !shape.group_id_x || !self.memory.read_backing(d.shader.address, &mut bytes) {
            return None;
        }
        let code: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        if !matches_pattern_fill(&code, user, &shape) {
            return None;
        }
        let range = decode_pattern_fill(user, d.packet.groups, d.uses_thread_dimensions())?;
        if range.invocations == 0 {
            return Some(None);
        }
        let tracker = &self.buffers.tracker;
        if tracker.has_gpu_dirty_pages(range.control, 8) || tracker.has_gpu_dirty_pages(range.source, 4) {
            return None;
        }
        let mut control = [0u8; 8];
        let mut source = [0u8; 4];
        if !self.memory.read_backing(range.control, &mut control) || !self.memory.read_backing(range.source, &mut source) {
            return None;
        }
        let count = u32::from_le_bytes(control[..4].try_into().unwrap());
        let mask = u32::from_le_bytes(control[4..].try_into().unwrap());
        let bytes = range.uniform_bytes(count, mask)?;
        if bytes == 0 {
            return Some(None);
        }
        Some(Some((range.destination, bytes, u32::from_le_bytes(source))))
    }

    /// The SDK indexed pattern-fill kernel ([`crate::fill_kernels`]): the
    /// pattern words and the element count come from user data, the store
    /// goes through the V# in user data. `Some(None)` when it stores
    /// nothing, `Some(Some((base, bytes, pattern)))` for the fill (the
    /// pattern must be one repeated dword, checked by [`Self::run_fill_kernel`]).
    /// `RPS5E_INDEXED_FILL_HLE=0` (or `RPS5E_FILL_HLE=0`) runs it as a
    /// dispatch.
    fn indexed_fill_kernel(&self, d: &ResolvedDispatch) -> Option<Option<(u64, u64, [u32; 4])>> {
        use crate::fill_kernels::{detect_indexed_pattern_fill, ComputeShape};
        static OFF: OnceLock<bool> = OnceLock::new();
        let off = *OFF.get_or_init(|| {
            let zero = |name: &str| std::env::var(name).is_ok_and(|v| v == "0");
            zero("RPS5E_INDEXED_FILL_HLE") || zero("RPS5E_FILL_HLE")
        });
        if off || d.packet.indirect_address != 0 || d.start != [0, 0, 0] || d.packet.groups[1] != 1 || d.packet.groups[2] != 1 {
            return None;
        }
        let shape = ComputeShape::from_rsrc2(d.threads, d.shader.rsrc2);
        if shape.threads != [64, 1, 1] || !shape.group_id_x || shape.thread_id_components != 1 {
            return None;
        }
        let f = match crate::fill_kernels::memo::indexed_at(self.memory, d.shader.address) {
            Some(f) => f?,
            None => {
                let mut bytes = [0u8; 256];
                if !self.memory.read_backing(d.shader.address, &mut bytes) {
                    return None;
                }
                let program = ps5shader::program::decode_blob(&bytes, d.shader.address).ok()?;
                detect_indexed_pattern_fill(&program)?
            }
        };
        // The work-group id must sit in the first system SGPR (after the
        // user data) and every register the fill reads must be user data.
        if f.group != d.shader.user_sgpr_count() {
            return None;
        }
        let user = &d.shader.user_data;
        let word = |r: u32| user.get(r as usize).copied();
        let (w0, w1, w2, w3) = (word(f.resource)?, word(f.resource + 1)?, word(f.resource + 2)?, word(f.resource + 3)?);
        let pattern = [word(f.pattern[0])?, word(f.pattern[1])?, word(f.pattern[2])?, word(f.pattern[3])?];
        // A plain dword V#: stride 4, no swizzle, no thread-id add, no
        // out-of-bounds handling, a buffer, and a 32-bit format that
        // stores the dword unchanged.
        let base = w0 as u64 | ((w1 & 0xFFFF) as u64) << 32;
        if (w1 >> 16) & 0x3FFF != 4 || w1 >> 31 != 0 || w3 >> 23 & 1 != 0 || w3 >> 24 & 3 != 0 || w3 >> 30 != 0 || base % 4 != 0 {
            return None;
        }
        if !matches!((w3 >> 12) & 0x7F, 20 | 21 | 22) {
            return None;
        }
        let threads = if d.uses_thread_dimensions() { d.packet.groups[0] as u64 } else { d.packet.groups[0] as u64 * 64 };
        let count = word(f.count)? as u64;
        let end = count.min(threads).min(w2 as u64);
        let bytes = end * 4;
        if bytes == 0 {
            return Some(None);
        }
        Some(Some((base, bytes, pattern)))
    }

    /// The SDK constant-fill kernel ([`crate::fill_kernels::detect_constant_fill`]):
    /// one constant dword per thread through the V# in user data, as the
    /// SDK clears metadata. `Some(None)` when it stores nothing,
    /// `Some(Some((base, bytes, value)))` for the fill. Run as a buffer fill,
    /// a clear code in DCC keys then clears the surface on the CPU's
    /// decision (no GPU-decided clear, no refill of the surface before it).
    /// `RPS5E_CONSTANT_FILL_HLE=0` (or `RPS5E_FILL_HLE=0`) runs it as a
    /// dispatch.
    fn constant_fill_kernel(&self, d: &ResolvedDispatch) -> Option<Option<(u64, u64, u32)>> {
        use crate::fill_kernels::{detect_constant_fill, ComputeShape};
        static OFF: OnceLock<bool> = OnceLock::new();
        let off = *OFF.get_or_init(|| {
            let zero = |name: &str| std::env::var(name).is_ok_and(|v| v == "0");
            zero("RPS5E_CONSTANT_FILL_HLE") || zero("RPS5E_FILL_HLE")
        });
        let why = |r: &str| {
            if crate::backend::profile::on() {
                crate::log_once(&format!("ps5gpu: cs={:#x} is no constant fill: {r}", d.shader.address));
            }
        };
        if off || d.packet.indirect_address != 0 || d.start != [0, 0, 0] || d.packet.groups[1] != 1 || d.packet.groups[2] != 1 {
            why(&format!("dispatch shape start {:?} groups {:?}", d.start, d.packet.groups));
            return None;
        }
        let shape = ComputeShape::from_rsrc2(d.threads, d.shader.rsrc2);
        if shape.threads != [64, 1, 1] || !shape.group_id_x || shape.thread_id_components != 1 {
            why(&format!("threads {:?} rsrc2 {:#x}", shape.threads, d.shader.rsrc2));
            return None;
        }
        let f = match crate::fill_kernels::memo::constant_at(self.memory, d.shader.address, 64) {
            Some(f) => f?,
            None => {
                let mut bytes = [0u8; 64];
                if !self.memory.read_backing(d.shader.address, &mut bytes) {
                    return None;
                }
                let program = ps5shader::program::decode_blob(&bytes, d.shader.address).ok()?;
                detect_constant_fill(&program)?
            }
        };
        why("(matched code)");
        // The work-group id must sit in the first system SGPR (after the
        // user data) and the V# must be user data.
        if f.group != d.shader.user_sgpr_count() {
            why(&format!("group sgpr {} vs user sgprs {}", f.group, d.shader.user_sgpr_count()));
            return None;
        }
        let user = &d.shader.user_data;
        let word = |r: u32| user.get(r as usize).copied();
        let (w0, w1, w2, w3) = (word(f.resource)?, word(f.resource + 1)?, word(f.resource + 2)?, word(f.resource + 3)?);
        // A plain dword V#: stride 4, no swizzle, no thread-id add, no
        // out-of-bounds handling, a buffer, and a 32-bit format that
        // stores the dword unchanged.
        let base = w0 as u64 | ((w1 & 0xFFFF) as u64) << 32;
        if (w1 >> 16) & 0x3FFF != 4 || w1 >> 31 != 0 || w3 >> 23 & 1 != 0 || w3 >> 24 & 3 != 0 || w3 >> 30 != 0 || base % 4 != 0 || !matches!((w3 >> 12) & 0x7F, 20 | 21 | 22) {
            why(&format!("V# {w0:08x} {w1:08x} {w2:08x} {w3:08x}"));
            return None;
        }
        let threads = if d.uses_thread_dimensions() { d.packet.groups[0] as u64 } else { d.packet.groups[0] as u64 * 64 };
        let bytes = threads.min(w2 as u64) * 4;
        if bytes == 0 {
            return Some(None);
        }
        let value = match f.loaded {
            None => f.value,
            Some((r, offset)) => {
                if !loaded_fill_enabled() {
                    why("loaded value (RPS5E_LOADED_FILL_HLE=0)");
                    return None;
                }
                // The scalar load of a plain V#: the dword at base + offset
                // when inside the buffer (num_records units of the stride,
                // bytes without one), else 0. Read on the CPU only when the
                // GPU has not written it.
                let (v0, v1, v2) = (word(r)?, word(r + 1)?, word(r + 2)?);
                let src = (v0 as u64 | ((v1 & 0xFFFF) as u64) << 32) + offset as u64;
                if (offset as u64) + 4 > crate::fill_kernels::scalar_buffer_bytes(v1, v2) {
                    0
                } else {
                    if self.buffers.has_gpu_dirty(src & !3, 4) {
                        why("loaded value written by the GPU");
                        return None;
                    }
                    let mut b = [0u8; 4];
                    if src % 4 != 0 || !self.memory.read_backing(src, &mut b) {
                        why("loaded value unreadable");
                        return None;
                    }
                    u32::from_le_bytes(b)
                }
            }
        };
        Some(Some((base, bytes, value)))
    }

    /// The SDK image-copy kernel ([`crate::copy_kernels`]) run as a copy
    /// between the cached images of its T#s when that reproduces it
    /// exactly; false when the dispatch must run.
    /// `RPS5E_COPY_KERNEL_HLE=0` always runs it as a dispatch.
    fn copy_kernel(&mut self, d: &ResolvedDispatch) -> bool {
        static OFF: OnceLock<bool> = OnceLock::new();
        if *OFF.get_or_init(|| std::env::var("RPS5E_COPY_KERNEL_HLE").is_ok_and(|v| v == "0")) {
            return false;
        }
        let memory = self.memory;
        if !crate::copy_kernels::dispatch_matches(d, |a, b| memory.read_backing(a, b)) {
            return false;
        }
        match crate::copy_kernels::run_image_copy(&mut self.sched, &mut self.buffers, &mut self.images, d) {
            Ok(c) => {
                self.stats.copy_kernels += 1;
                if crate::pipeline::trace::active() {
                    crate::pipeline::trace::log(&format!("    (copy kernel cs={:#x}: {}x{} texels, level {} -> {})", d.shader.address, c.width, c.height, c.src_level, c.dst_level));
                }
                true
            }
            Err(why) => {
                crate::log_once(&format!("ps5gpu: image-copy kernel cs={:#x} runs as a dispatch: {why}", d.shader.address));
                false
            }
        }
    }

    /// Run the SDK fill kernel as a buffer fill (as the command processor
    /// would): no shader, and images under the range are cleared (zero
    /// fills covering them) or reloaded, DCC keys holding a clear code clear
    /// their surface, HTILE fills clear depth at the next bind. False when
    /// the pattern is not one repeated dword.
    pub(crate) fn run_fill_kernel(&mut self, base: u64, bytes: u64, pattern: [u32; 4]) -> bool {
        let p = pattern[0];
        if pattern.iter().any(|&w| w != p) || bytes == 0 || base % 4 != 0 {
            return false;
        }
        match self.buffers.fill(&mut self.sched, base, bytes, p) {
            Ok(true) => {}
            _ => return false,
        }
        self.stats.fills += 1;
        if crate::pipeline::trace::active() {
            crate::pipeline::trace::log(&format!("    (fill kernel: {base:#x}+{bytes:#x} = {p:08x})"));
        }
        let dcc = self.dcc_targets.get(&base).copied();
        // Single-colour DCC keys: the colour is stored in the surface's
        // blocks, which the clear writes after the keys; the bind decides.
        let single = dcc.is_some() && p == u32::from_ne_bytes([crate::render::formats::DCC_CLEAR_SINGLE; 4]);
        if self.htile.contains(&base) || self.cmask.contains(&base) || single {
            self.images.note_fill(base, bytes, p);
        }
        self.images.filled(&mut self.sched, base, base + bytes, p);
        if let Some(t) = dcc {
            let code = p as u8;
            let slice = t.dcc_slice_size();
            if p == u32::from_ne_bytes([code; 4]) && bytes >= slice && slice != 0 {
                self.dcc.insert(base, false);
                if let Some(value) = t.dcc_clear(code) {
                    if let Ok(id) = self.images.find_or_create(&mut self.sched, &mut self.buffers, &color_key(&t)) {
                        if crate::pipeline::trace::active() {
                            crate::pipeline::trace::log(&format!("    (DCC fast clear of color target #{id} {:#x} by the fill kernel: code {code:#x})", t.base));
                        }
                        // Every slice the fill covers whole is cleared.
                        let slices = self.images.get(id).map_or(0, |i| i.key.slices());
                        let covered = ((bytes / slice).min(slices as u64)) as u32;
                        let clears: SliceClears = (0..covered).map(|l| (l, value)).collect();
                        self.apply_slice_clears(id, &clears);
                        // Consumed: the keys of the cleared slices mark
                        // uncompressed data, or a later bind finds the code
                        // (once the fill reached guest memory) and clears
                        // everything drawn since.
                        if consume_on_gpu_enabled() && covered != 0 {
                            self.consume_metadata_clear(base, covered as u64 * slice);
                        }
                    }
                }
            }
        }
        true
    }

    /// An indirect dispatch with thread dimensions as a direct one: its
    /// argument record (thread counts) read once the GPU work writing it is
    /// done. None when the record is unreadable. Only when the provider
    /// cannot resolve the record on the GPU
    /// ([`PipelineProvider::indirect_thread_dimensions`];
    /// `RPS5E_GPU_INDIRECT_THREADS=0`).
    fn read_thread_dimensions(&mut self, d: &ResolvedDispatch) -> Option<ResolvedDispatch> {
        let a = d.packet.indirect_address;
        self.buffers.download(&mut self.sched, a, 12);
        let mut b = [0u8; 12];
        if !self.memory.read_backing(a, &mut b) {
            crate::log_once(&format!("ps5gpu: dispatch of cs={:#x}: thread dimensions at {a:#x} unreadable; skipped", d.shader.address));
            return None;
        }
        let mut r = *d;
        r.packet.groups = std::array::from_fn(|i| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()));
        r.packet.indirect_address = 0;
        Some(r)
    }

    pub fn dispatch(&mut self, d: &ResolvedDispatch) {
        let mem = self.memory;
        crate::capture::around(mem, || self.dispatch_inner(d), |_| Event::Dispatch(Box::new(*d)))
    }

    fn dispatch_inner(&mut self, d: &ResolvedDispatch) {
        prefetch_value(d);
        self.stats.dispatches += 1;
        crate::perf::DISPATCHES.fetch_add(1, Ordering::Relaxed);
        let direct;
        // A record that is not dword-aligned cannot be an indirect host
        // dispatch's argument: it is read on the CPU as well.
        let thread_record = d.packet.dispatch_initiator & crate::render::DISPATCH_USE_THREAD_DIMENSIONS != 0 && !self.pipelines.indirect_thread_dimensions(d);
        let d = if d.packet.indirect_address != 0 && (thread_record || d.packet.indirect_address % 4 != 0) {
            match self.read_thread_dimensions(d) {
                Some(r) => {
                    direct = r;
                    &direct
                }
                None => return,
            }
        } else {
            d
        };
        if let Some((base, bytes, pattern)) = self.fill_kernel(d) {
            if self.run_fill_kernel(base, bytes, pattern) {
                return;
            }
        }
        match self.pattern_fill_kernel(d) {
            Some(None) => {
                self.stats.pattern_fills += 1;
                return;
            }
            Some(Some((base, bytes, value))) => {
                if self.run_fill_kernel(base, bytes, [value; 4]) {
                    self.stats.pattern_fills += 1;
                    return;
                }
            }
            None => {}
        }
        match self.indexed_fill_kernel(d) {
            Some(None) => {
                self.stats.pattern_fills += 1;
                return;
            }
            Some(Some((base, bytes, pattern))) => {
                if self.run_fill_kernel(base, bytes, pattern) {
                    self.stats.pattern_fills += 1;
                    return;
                }
            }
            None => {}
        }
        match self.constant_fill_kernel(d) {
            Some(None) => {
                self.stats.pattern_fills += 1;
                return;
            }
            Some(Some((base, bytes, value))) => {
                if self.run_fill_kernel(base, bytes, [value; 4]) {
                    self.stats.pattern_fills += 1;
                    return;
                }
            }
            None => {}
        }
        if self.copy_kernel(d) {
            return;
        }
        let targets = DrawTargets { colors: Default::default(), depth: None };
        if skipped_shader(&[d.shader.address]) {
            return;
        }
        self.isolate_preparation("dispatch");
        let timed = self.sched.time_item_begin(d.shader.address);
        let mut ctx = DrawContext { dev: &self.dev, sched: &mut self.sched, buffers: (&mut self.buffers).into(), images: (&mut self.images).into(), targets: &targets };
        let outcome = self.pipelines.dispatch(&mut ctx, d);
        self.sched.time_item_end(timed);
        if self.pipelines.take_compile_pending() {
            self.stats.dispatches -= 1;
            crate::perf::DISPATCHES.fetch_sub(1, Ordering::Relaxed);
            self.compile_pending = true;
            return;
        }
        match outcome {
            DrawOutcome::NeedsShader => self.stats.needs_shader_dispatches += 1,
            DrawOutcome::Recorded => {
                let n = self.dispatch_counts.entry(d.shader.address).or_default();
                *n += 1;
                let n = *n;
                if traced_cs() == Some(d.shader.address) {
                    let u = &d.shader.user_data;
                    let table = u[12] as u64 | (u[13] as u64) << 32;
                    let mut bytes = [0u8; 256];
                    let ok = self.memory.read_backing(table & 0xFFFF_FFFF_FFFF, &mut bytes);
                    let words: Vec<String> = bytes.chunks(4).map(|c| format!("{:08x}", u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();
                    log::info!("[gpu-trace] cs {:#x} #{n} user={:08x?} table {table:#x} ({ok}): {}", d.shader.address, &u[..16], words.join(" "));
                }
                let mut user = [0u32; 16];
                user.copy_from_slice(&d.shader.user_data[..16]);
                self.sched.note(crate::scheduler::Note::Dispatch { n, cs: d.shader.address, packet: d.packet, threads: d.threads, user });
                if submit_each() {
                    self.submit_isolated();
                } else if eager_through_enabled() && self.buffers.has_pending_through() {
                    // The copies of the ranges it wrote follow it now.
                    self.stats.eager_throughs += 1;
                    self.buffers.flush_write_through(&mut self.sched);
                } else if eager_hot_through_enabled() && self.buffers.has_hot_pending_through() {
                    // Only the ranges a CPU access waited for before: their
                    // copies follow the writer at once, and the submit lets
                    // the waiting CPU see them (it waits for this tick only,
                    // not for the rest of the slice).
                    if self.buffers.flush_write_through_hot(&mut self.sched) {
                        self.stats.hot_eager_throughs += 1;
                        self.stats.early_submits += 1;
                        self.flush_inner();
                    }
                }
            }
            DrawOutcome::Skipped | DrawOutcome::Hole => {}
        }
        // A compute program that overwrites a depth target's HTILE without
        // reading it clears the metadata: the next bind of the target loads
        // DB_DEPTH_CLEAR.
        let mut dcc_cleared = Vec::new();
        for (addr, size) in self.pipelines.take_overwrites() {
            if self.htile.contains(&addr) {
                self.stats.fills += 1;
                if htile_kernel_decode_enabled() {
                    // The stored words replace an earlier fill.
                    self.images.take_fill(addr);
                    self.htile_kernel_written.insert(addr, size);
                } else {
                    self.images.note_fill(addr, size, 0);
                }
            }
            if let Some(written) = self.dcc.get_mut(&addr) {
                *written = true;
                // A clear kernel writes whole slices; a smaller write is other data over reused memory.
                let whole = |t: &ColorTarget| t.clear_metadata().is_some_and(|(_, slice, _)| size >= slice) || !whole_metadata_writes_enabled();
                if let Some(t) = self.dcc_targets.get(&addr).filter(|t| whole(t)) {
                    dcc_cleared.push(*t);
                    if crate::backend::profile::on() {
                        let mut code = [0u8; 64];
                        let _ = self.memory.read_backing(d.shader.address, &mut code);
                        let words: Vec<String> = code.chunks(4).map(|c| format!("{:08x}", u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();
                        crate::log_once(&format!("ps5gpu: DCC keys {addr:#x} written by cs={:#x} (decided on the GPU): code {}", d.shader.address, words.join(" ")));
                    }
                }
            }
        }
        // A clear code written into the metadata clears the surface now:
        // work between this dispatch and the next bind of the target (e.g.
        // compute programs storing into it) must see the cleared image and
        // must not be wiped by a clear applied later.
        for t in dcc_cleared {
            self.apply_dcc_clear(&t);
        }
        self.compressed_stores_done();
        self.bound_retired_bytes();
        if outcome == DrawOutcome::Recorded {
            self.early_submit(true);
        }
    }

    /// Materialise a pending DCC fast clear of colour target `t` into its
    /// cached image.
    fn apply_dcc_clear(&mut self, t: &ColorTarget) {
        let Ok(id) = self.images.find_or_create(&mut self.sched, &mut self.buffers, &color_key(t)) else { return };
        if self.gpu_dcc_clear(t, id) {
            return;
        }
        let clears = self.pending_dcc_clears(t);
        if clears.is_empty() {
            return;
        }
        if crate::pipeline::trace::active() {
            // SAFETY: every member of the clear value union is plain data.
            let values: Vec<(u32, [u32; 4])> = clears.iter().map(|(l, v)| (*l, unsafe { v.uint32 })).collect();
            crate::pipeline::trace::log(&format!("    (DCC fast clear of color target #{id} {:#x} by a compute program: {values:x?})", t.base));
        }
        self.apply_slice_clears(id, &clears);
        let tick = self.sched.current_tick();
        self.images.mark_gpu_written(id, tick);
    }


    /// Decide the DCC fast clears of `t` (image `id`) on the GPU, slice by
    /// slice: scan each bound slice's keys, clear that layer under
    /// conditional rendering, mark the keys uncompressed. False when the
    /// GPU path is unavailable.
    fn gpu_dcc_clear(&mut self, t: &ColorTarget, id: u64) -> bool {
        /// Slices decided in one go (result slots of the ring).
        const MAX_SLICES: u32 = 256;
        // DCC keys, or CMASK (fast clear without DCC: byte 0 = every tile
        // of the block holds the register clear).
        let Some((meta, slice, cmask)) = t.clear_metadata() else { return false };
        if slice == 0 || slice % 4 != 0 || meta % 4 != 0 || t.layers.max(1) > MAX_SLICES {
            crate::log_once(&format!("ps5gpu: metadata clears decided on the CPU: {meta:#x} slice {slice:#x}, {} layers", t.layers));
            return false;
        }
        let Some(cond) = self.dev.conditional.clone() else {
            crate::log_once("ps5gpu: metadata clears decided on the CPU: no conditional rendering");
            return false;
        };
        let any = dcc_gpu_any_enabled();
        // Plain 2D single-sample single-level surfaces only (the old rule,
        // `RPS5E_DCC_GPU_ANY=0`); otherwise any single-sample colour
        // attachment whose layers (depth slices of a volume) have 2D views
        // of level 0: the attachment clears cover level 0 of a layer.
        let plain = |i: &crate::images::CachedImage| {
            i.image.usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) && i.key.dim == crate::images::ImageDim::D2 && i.key.samples == 1 && i.image.mip_levels == 1
        };
        let decidable = |i: &crate::images::CachedImage| i.image.usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) && i.key.samples == 1 && i.key.dim != crate::images::ImageDim::D1;
        let Some(image) = self.images.get(id) else { return false };
        if self.dcc_gpu.is_none() || !(if any { decidable(image) } else { plain(image) }) {
            crate::log_once(&format!(
                "ps5gpu: metadata clears of {:?} {}x{}x{} {:?} ({} layers, {} levels, {} samples, usage {:?}) decided on the CPU",
                image.key.dim, image.key.width, image.key.height, image.key.depth, image.key.format, image.key.layers, image.image.mip_levels, image.key.samples, image.image.usage
            ));
            return false;
        }
        let image_layers = if any { image.key.slices() } else { image.key.layers };
        let layers: Vec<u32> = (t.base_layer..t.base_layer + t.layers.max(1)).filter(|&l| l < image_layers).collect();
        if layers.is_empty() {
            return false;
        }
        // The attachment view of the bound layers, made before anything is
        // recorded (without it the clears could not land).
        let format = image.key.format;
        if any && self.images.view(id, format, vk::ImageAspectFlags::COLOR, layers[0], layers.len() as u32).is_none() {
            crate::log_once(&format!("ps5gpu: metadata clears of {format:?} without an attachment view decided on the CPU"));
            return false;
        }
        let values: Vec<Option<vk::ClearColorValue>> = crate::dcc::CODES.iter().map(|&c| t.metadata_clear(c, cmask)).collect();
        let mut codes = [u32::MAX; 5];
        for (k, v) in values.iter().enumerate() {
            if v.is_some() {
                codes[k] = crate::dcc::CODES[k] as u32;
            }
        }
        // The keys of the bound slices, each with a result slot.
        let first = meta + layers[0] as u64 * slice;
        let all = match self.buffers.obtain(&mut self.sched, first, slice * layers.len() as u64, true) {
            Ok(s) => s,
            Err(e) => {
                crate::log_once(&format!("ps5gpu: metadata clears decided on the CPU: keys {first:#x}+{:#x}: {e:?}", slice * layers.len() as u64));
                return false;
            }
        };
        // The image keeps its contents unless a clear lands.
        self.images.prepare(&mut self.sched, &mut self.buffers, id, false);
        let r = self.dcc_gpu.as_mut().unwrap();
        let slices: Vec<(u32, crate::dcc::Range, u64)> = layers
            .iter()
            .enumerate()
            .map(|(i, &l)| (l, crate::dcc::Range { buffer: all.buffer, offset: all.offset + i as u64 * slice, size: slice }, r.slot()))
            .collect();
        let results = r.results.buffer;
        self.sched.full_barrier();
        let cb = self.sched.cmd();
        for &(_, _, slot) in &slices {
            // SAFETY: recording buffer; the slot lies inside the result ring.
            unsafe { self.dev.device.cmd_fill_buffer(cb, results, slot, 256, 0) };
        }
        for mode in [0, 1] {
            self.sched.full_barrier();
            let cb = self.sched.cmd();
            let r = self.dcc_gpu.as_ref().unwrap();
            for &(_, keys, slot) in &slices {
                r.record(cb, mode, keys, slot, &codes);
            }
        }
        // The predicates are read by the conditional-rendering stage.
        let mb = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::CONDITIONAL_RENDERING_EXT)
            .dst_access_mask(vk::AccessFlags2::CONDITIONAL_RENDERING_READ_EXT);
        // SAFETY: recording buffer.
        unsafe { self.dev.device.cmd_pipeline_barrier2(self.sched.cmd(), &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mb))) };
        self.sched.full_barrier();
        // The attachment layout before the conditional clears.
        self.images.transit(&mut self.sched, id, crate::images::COLOR_ATTACHMENT, Some(crate::images::layout::SubRange { base_level: 0, levels: 1, base_layer: 0, layers: image_layers }));
        if any {
            // One rendering scope over the bound layers, one conditional
            // clear per layer and decodable code.
            let clears: Vec<(u32, vk::ClearColorValue, vk::Buffer, u64)> = slices
                .iter()
                .flat_map(|&(layer, _, slot)| values.iter().enumerate().filter_map(move |(k, v)| v.as_ref().map(|v| (layer, *v, results, slot + crate::dcc::predicate_offset(k)))))
                .collect();
            if !self.images.conditional_layer_clears(&mut self.sched, &cond, id, layers[0], layers.len() as u32, &clears) {
                // The view was checked above. Without the clears the keys
                // are not consumed: the CPU decides (the passes recorded so
                // far only wrote their result slots).
                crate::log_once("ps5gpu: conditional metadata clears could not be recorded; decided on the CPU");
                return false;
            }
        } else {
            for &(layer, _, slot) in &slices {
                for (k, v) in values.iter().enumerate() {
                    let Some(v) = v else { continue };
                    let begin = vk::ConditionalRenderingBeginInfoEXT::default().buffer(results).offset(slot + crate::dcc::predicate_offset(k));
                    // SAFETY: recording buffer; the predicate word is 4-byte aligned.
                    unsafe { (cond.fp().cmd_begin_conditional_rendering_ext)(self.sched.cmd(), &begin) };
                    self.images.clear_color_attachment(&mut self.sched, id, *v, layer, 1);
                    // SAFETY: matches the begin above.
                    unsafe { (cond.fp().cmd_end_conditional_rendering_ext)(self.sched.cmd()) };
                }
            }
        }
        self.sched.full_barrier();
        let cb = self.sched.cmd();
        let r = self.dcc_gpu.as_ref().unwrap();
        for &(_, keys, slot) in &slices {
            r.record(cb, 2, keys, slot, &codes);
        }
        self.sched.full_barrier();
        let tick = self.sched.current_tick();
        self.images.mark_gpu_written(id, tick);
        // Decided: the keys are GPU-written data from now on.
        self.dcc.insert(meta, false);
        self.stats.dcc_gpu_decisions += 1;
        self.stats.color_clears += 1;
        if crate::pipeline::trace::active() {
            crate::pipeline::trace::log(&format!("    (DCC fast clear of color target #{id} {:#x} decided on the GPU, {} slices)", t.base, slices.len()));
        }
        true
    }

    /// With `RPS5E_GPU_SUBMIT_EACH=1`, submit the work recorded before a
    /// draw or dispatch (uploads, clears, detiling) in a tick of its own.
    fn isolate_preparation(&mut self, what: &str) {
        if submit_each() && self.sched.is_recording() {
            self.sched.note(crate::scheduler::Note::Text(format!("work recorded before a {what}")));
            self.submit_isolated();
        }
    }

    /// Submit the journaled item in a tick of its own (and time it in mode 2).
    fn submit_isolated(&mut self) {
        let start = std::time::Instant::now();
        let tick = self.sched.flush();
        if submit_each_mode() >= 2 {
            self.sched.timeline.wait(tick);
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            if ms > 20.0 {
                log::warn!("ps5gpu: tick {tick} took {ms:.1} ms: {}", self.sched.last_note().unwrap_or_default());
            }
        }
    }

    pub fn end_of_pipe(&mut self, done: Box<dyn FnOnce() + Send>) {
        let mem = self.memory;
        crate::capture::around(
            mem,
            || {
                self.stats.end_of_pipe += 1;
                self.buffers.flush_write_through(&mut self.sched);
                self.sched.queue_priority(done);
                self.sched.flush();
            },
            |_| Event::EndOfPipe,
        )
    }

    /// Run `done` once the work recorded so far completed, without a
    /// submit of its own: an open command buffer is submitted by the
    /// slice's [`Backend::flush`]; with none, everything recorded before is
    /// already submitted.
    pub fn after_pipe(&mut self, done: Box<dyn FnOnce() + Send>) {
        let mem = self.memory;
        crate::capture::around(mem, || self.after_pipe_inner(done), |_| Event::AfterPipe)
    }

    fn after_pipe_inner(&mut self, done: Box<dyn FnOnce() + Send>) {
        self.buffers.flush_write_through(&mut self.sched);
        if self.sched.is_recording() {
            self.sched.queue_priority(done);
        } else {
            self.sched.queue_priority_at(self.sched.current_tick().saturating_sub(1), done);
        }
    }

    /// The command processor publishes a write the guest can observe
    /// (see [`BufferCache::publish_point`]).
    pub fn publish_point(&mut self) {
        // The guest may reuse what queued upload copies read once it sees
        // the write.
        crate::buffers::async_copy::drain();
        let done = self.sched.timeline.completed_tick();
        self.buffers.publish_point(done);
    }

    /// Write-throughs recorded and not yet completed by the GPU (see
    /// [`RenderExecutor::writes_through_pending`]).
    pub fn writes_through_pending(&mut self) -> bool {
        if !self.buffers.writes_through_after(self.sched.timeline.completed_tick()) {
            return false;
        }
        // The known completed tick may lag: read the semaphore once.
        self.buffers.writes_through_after(self.sched.timeline.refresh())
    }

    /// Submit recorded work (end of a command-stream slice).
    pub fn flush(&mut self) {
        let mem = self.memory;
        crate::capture::around(mem, || self.flush_inner(), |_| Event::Flush)
    }

    fn flush_inner(&mut self) {
        self.buffers.flush_write_through(&mut self.sched);
        if self.sched.is_recording() {
            self.sched.flush();
        }
        self.sched.run_completed();
        // Right after the submit, on the recording thread (no queue lock
        // held): the reference-clock offset is re-measured once a second
        // with one tiny dispatch; a busy GPU only skips the sample.
        crate::clock::recalibrate_if_due(&self.dev);
    }

    /// Once per completed submission (and frame boundary), as the
    /// reference collects: copy back the linear images written since the
    /// last time (when enabled), then the memory-pressure collections of
    /// the image and buffer caches ([`crate::gc`]). Not per slice: a
    /// submission resumed after each wait would age the caches' entries by
    /// many collection ticks per frame and evict the frame's own working
    /// set (SILENT HILL reloaded its textures every few frames).
    pub fn pressure_gc(&mut self) {
        let mem = self.memory;
        crate::capture::around(mem, || self.pressure_gc_once(), |_| Event::SubmissionComplete)
    }

    fn pressure_gc_once(&mut self) {
        let tick = self.sched.current_tick();
        if tick == self.gc_tick {
            return;
        }
        self.gc_tick = tick;
        self.gc_ticks.push(tick);
        let t = self.sched.time_item_begin(gpu_label::PRESSURE_GC);
        self.pressure_gc_inner();
        self.sched.time_item_end(t);
    }

    fn pressure_gc_inner(&mut self) {
        self.images.process_downloads(&mut self.sched, &mut self.buffers);
        let usage = self.gc.usage(&self.dev);
        GC_USAGE_MIB.store(usage.unwrap_or(0) >> 20, Ordering::Relaxed);
        let thresholds = self.gc.images;
        self.images.collect_pressure(&mut self.sched, &mut self.buffers, usage, &thresholds, &self.gc_ticks);
        let thresholds = self.gc.buffers;
        let images = &self.images;
        let tick = self.sched.current_tick();
        self.buffers.collect_pressure(&mut self.sched, usage, &thresholds, &self.gc_ticks, |frames| images.cutoff_after(tick, frames));
    }

    pub fn memory_filled(&mut self, addr: u64, size: u64, value: u32) {
        let mem = self.memory;
        crate::capture::around(
            mem,
            || {
                self.stats.fills += 1;
                self.images.note_fill(addr, size, value);
                // A fill replaces words a compute program stored.
                if !self.htile_kernel_written.is_empty() {
                    self.htile_kernel_written.retain(|&a, _| a < addr || a >= addr.saturating_add(size));
                }
            },
            |_| Event::MemoryFilled { addr, size, value },
        )
    }

    /// A command-processor copy: on the GPU when either range holds
    /// GPU-written pages (a CPU copy would wait for the GPU page by page).
    pub fn copy_memory(&mut self, dst: u64, src: u64, size: u64) -> bool {
        let mem = self.memory;
        crate::capture::around(
            mem,
            || {
                let t = self.sched.time_item_begin(gpu_label::CP_COPY);
                let r = self.copy_memory_inner(dst, src, size);
                self.sched.time_item_end(t);
                r
            },
            |&gpu| Event::CopyMemory { dst, src, size, gpu },
        )
    }

    fn copy_memory_inner(&mut self, dst: u64, src: u64, size: u64) -> bool {
        // The copy replaces what a pending metadata fill there wrote.
        self.images.drop_fills(dst, size);
        // Images the GPU modified hold the newest bytes of their memory
        // (guest memory is stale until a write-back): a copy reading them
        // takes them through their write-back, on the GPU.
        let written = self.images.writeback_range(&mut self.sched, &mut self.buffers, src, size);
        if written != 0 {
            static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if LOGGED.fetch_add(1, Ordering::Relaxed) < 16 {
                log::info!("ps5gpu: copy {src:#x} -> {dst:#x} ({size:#x} bytes) reads {written} GPU-modified image(s): written back first");
            }
        }
        if !self.buffers.has_gpu_dirty(src, size) && !self.buffers.has_gpu_dirty(dst, size) {
            return false;
        }
        match self.buffers.copy(&mut self.sched, dst, src, size) {
            Ok(true) => {
                self.images.invalidate_range(dst, size);
                true
            }
            Ok(false) => false,
            Err(e) => {
                crate::log_once(&format!("ps5gpu: GPU copy {src:#x} -> {dst:#x} ({size:#x}): {e:?}"));
                false
            }
        }
    }

    /// A command-processor fill: on the GPU when the range holds
    /// GPU-written pages.
    pub fn fill_memory(&mut self, dst: u64, size: u64, value: u32) -> bool {
        let mem = self.memory;
        crate::capture::around(mem, || self.fill_memory_inner(dst, size, value), |&gpu| Event::FillMemory { dst, size, value, gpu })
    }

    fn fill_memory_inner(&mut self, dst: u64, size: u64, value: u32) -> bool {
        if !self.buffers.has_gpu_dirty(dst, size) {
            return false;
        }
        match self.buffers.fill(&mut self.sched, dst, size, value) {
            Ok(true) => {
                self.images.invalidate_range(dst, size);
                true
            }
            Ok(false) => false,
            Err(e) => {
                crate::log_once(&format!("ps5gpu: GPU fill {dst:#x} ({size:#x}): {e:?}"));
                false
            }
        }
    }

    /// A write of the command processor the guest can observe (a label,
    /// WRITE_DATA), made where the queue is now: true when it was recorded
    /// on the GPU because its page holds GPU-written bytes
    /// (`RPS5E_CP_WRITES_ON_GPU`), false when the caller writes guest
    /// memory itself.
    ///
    /// Such a page is protected until downloaded: writing it on the CPU
    /// would fault, wait for all work recorded so far and download the
    /// fault window, and leave the page CPU-dirty for its next GPU use.
    /// On the GPU the bytes land in the page's mirror in order with that
    /// work and reach guest memory with its write-through; a CPU read of
    /// them before that waits as for any GPU result.
    pub fn cp_write(&mut self, dst: u64, data: &[u8]) -> bool {
        cp_writes_on_gpu_enabled() && self.write_memory(dst, data)
    }

    /// A command-processor write: on the GPU when the range holds
    /// GPU-written pages.
    pub fn write_memory(&mut self, dst: u64, data: &[u8]) -> bool {
        let mem = self.memory;
        crate::capture::around(mem, || self.write_memory_inner(dst, data), |&gpu| Event::WriteMemory { dst, data: data.to_vec(), gpu })
    }

    fn write_memory_inner(&mut self, dst: u64, data: &[u8]) -> bool {
        let size = data.len() as u64;
        if !self.buffers.has_gpu_dirty(dst, size) {
            return false;
        }
        match self.buffers.write(&mut self.sched, dst, data) {
            Ok(true) => {
                self.images.invalidate_range(dst, size);
                true
            }
            Ok(false) => false,
            Err(e) => {
                crate::log_once(&format!("ps5gpu: GPU write {dst:#x} ({size:#x}): {e:?}"));
                false
            }
        }
    }

    /// A command-processor transfer touching GDS, executed on the GPU in
    /// order with the recorded work (shader programs use the same GDS
    /// buffer). False when the provider has no GDS buffer.
    pub fn gds_transfer(&mut self, op: GdsTransfer) -> bool {
        if !crate::capture::active() {
            return self.gds_transfer_inner(op);
        }
        let mem = self.memory;
        let recorded = op.clone();
        crate::capture::around(mem, || self.gds_transfer_inner(op), |_| Event::Gds(recorded))
    }

    fn gds_transfer_inner(&mut self, op: GdsTransfer) -> bool {
        const GDS_BYTES: u64 = 64 << 10;
        let Some(gds) = self.pipelines.gds(&mut self.sched) else { return false };
        let (offset, size) = match &op {
            GdsTransfer::Write { offset, data } => (*offset, data.len() as u64),
            GdsTransfer::Fill { offset, size, .. } | GdsTransfer::FromMemory { offset, size, .. } | GdsTransfer::ToMemory { offset, size, .. } => (*offset, *size),
        };
        let size = size.min(GDS_BYTES.saturating_sub(offset));
        if size == 0 || offset % 4 != 0 || size % 4 != 0 {
            crate::log_once(&format!("ps5gpu: GDS transfer at {offset:#x} size {size:#x} not dword aligned"));
            return true;
        }
        let result = (|| -> Result<(), vk::Result> {
            match op {
                GdsTransfer::Write { data, .. } => {
                    self.sched.full_barrier();
                    let cb = self.sched.cmd();
                    // SAFETY: recording buffer; aligned, at most 64 KiB.
                    unsafe { self.dev.device.cmd_update_buffer(cb, gds, offset, &data[..size as usize]) };
                }
                GdsTransfer::Fill { value, .. } => {
                    self.sched.full_barrier();
                    let cb = self.sched.cmd();
                    // SAFETY: recording buffer; aligned range inside GDS.
                    unsafe { self.dev.device.cmd_fill_buffer(cb, gds, offset, size, value) };
                }
                GdsTransfer::FromMemory { src, .. } => {
                    let s = self.buffers.obtain(&mut self.sched, src, size, false)?;
                    self.sched.full_barrier();
                    let cb = self.sched.cmd();
                    let region = vk::BufferCopy { src_offset: s.offset, dst_offset: offset, size };
                    // SAFETY: recording buffer; both buffers alive until the tick completes.
                    unsafe { self.dev.device.cmd_copy_buffer(cb, s.buffer, gds, &[region]) };
                }
                GdsTransfer::ToMemory { dst, .. } => {
                    let d = self.buffers.obtain(&mut self.sched, dst, size, true)?;
                    self.sched.full_barrier();
                    let cb = self.sched.cmd();
                    let region = vk::BufferCopy { src_offset: offset, dst_offset: d.offset, size };
                    // SAFETY: recording buffer; both buffers alive until the tick completes.
                    unsafe { self.dev.device.cmd_copy_buffer(cb, gds, d.buffer, &[region]) };
                    self.sched.full_barrier();
                    self.buffers.write_through(&mut self.sched, dst, size);
                    self.images.invalidate_range(dst, size);
                }
            }
            self.sched.full_barrier();
            Ok(())
        })();
        if let Err(e) = result {
            crate::log_once(&format!("ps5gpu: GDS transfer: {e:?}"));
        }
        true
    }

    /// Submit the open tick once the memory retired into it (temporary
    /// upload buffers, mirrors replaced by merges) reaches the budget: it
    /// is freed only after the tick completed, so a tick that keeps
    /// recording (a load uploading many textures) would hold all of it.
    /// Called between draws and dispatches.
    fn bound_retired_bytes(&mut self) {
        let budget = retired_bytes_budget();
        if budget != 0 && self.sched.is_recording() && self.sched.open_retired_bytes() >= budget {
            self.stats.retired_budget_submits += 1;
            self.flush_inner();
        }
    }

    fn housekeeping(&mut self) {
        self.bound_retired_bytes();
        self.housekeeping += 1;
        if self.housekeeping % 256 == 0 {
            self.memory.pages.sync_mode();
            self.sched.run_completed();
            self.images.collect(&mut self.sched, &mut self.buffers, IMAGE_BUDGET);
            let tick = self.sched.current_tick();
            let done = self.sched.timeline.refresh();
            let idle = self.images.idle_cutoff(tick, done);
            self.buffers.collect(&mut self.sched, idle);
        }
    }

    /// A snapshot image of `(w, h, format)` no presentation still holds.
    fn snapshot(&mut self, w: u32, h: u32, format: vk::Format) -> Option<Arc<Snapshot>> {
        if let Some(s) = self.snapshots.iter().find(|s| Arc::strong_count(s) == 1 && s.width == w && s.height == h && s.format == format) {
            return Some(s.clone());
        }
        self.snapshots.retain(|s| Arc::strong_count(s) > 1 || (s.width == w && s.height == h && s.format == format));
        let image = Image::new(
            &self.dev,
            format,
            vk::Extent3D { width: w, height: h, depth: 1 },
            1,
            1,
            vk::SampleCountFlags::TYPE_1,
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageCreateFlags::empty(),
        )
        .ok()?;
        let cb = self.sched.cmd();
        let b = vk::ImageMemoryBarrier2::default()
            .image(image.image)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .subresource_range(image.full_range());
        // SAFETY: recording buffer; fresh image.
        unsafe { self.dev.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&b))) };
        let s = Arc::new(Snapshot { image, width: w, height: h, format });
        self.snapshots.push(s.clone());
        Some(s)
    }

    /// Before a frame capture: GPU results are
    /// written back into guest memory and the image and buffer caches are
    /// emptied, so the captured frames read every resource they use from
    /// guest memory (and the capture records it). Images that cannot be
    /// written back (multisampled) stay. Returns (images dropped, images
    /// kept, buffers dropped).
    pub fn capture_reset(&mut self) -> (usize, usize, usize) {
        self.flush_inner();
        let (dropped, kept) = self.images.capture_reset(&mut self.sched, &mut self.buffers);
        let buffers = self.buffers.capture_reset(&mut self.sched);
        self.sched.flush_and_wait();
        (dropped, kept, buffers)
    }

    /// Write the GPU-rendered image of a display buffer back to guest
    /// memory. False when the GPU did not render it.
    pub fn writeback_display(&mut self, desc: &crate::SurfaceDesc, addr: u64) -> bool {
        let Some(id) = self.images.find_display(addr, desc.width, desc.height) else { return false };
        let Some(key) = self.images.get(id).map(|i| i.key) else { return false };
        if !self.images.writeback(&mut self.sched, &mut self.buffers, id) {
            return false;
        }
        // Colour display images have one plane.
        self.buffers.download(&mut self.sched, key.addr(), key.size())
    }

    /// Copy the GPU-rendered image of a display buffer into a snapshot.
    /// None when the GPU did not render it (the CPU path shows guest
    /// memory) or its format cannot be copied.
    pub fn capture_display(&mut self, desc: &crate::SurfaceDesc, addr: u64) -> Option<GpuFrame> {
        let t = self.sched.time_item_begin(gpu_label::CAPTURE);
        let r = self.capture_display_inner(desc, addr);
        self.sched.time_item_end(t);
        r
    }

    fn capture_display_inner(&mut self, desc: &crate::SurfaceDesc, addr: u64) -> Option<GpuFrame> {
        use crate::format::Layout;
        crate::pipeline::trace::frame_presented();
        self.pipelines.frame_presented();
        self.images.end_frame(self.sched.current_tick());
        let layout = crate::format::decode(desc.pixel_format)?;
        let format = match layout {
            Layout::Rgba8 => vk::Format::R8G8B8A8_UNORM,
            Layout::Bgra8 => vk::Format::B8G8R8A8_UNORM,
            Layout::Packed10RedLow => vk::Format::A2B10G10R10_UNORM_PACK32,
            Layout::Packed10BlueLow => vk::Format::A2R10G10B10_UNORM_PACK32,
            // B, G, R, A in memory: an RGBA16 image rendered in that order
            // reaches the snapshot through a colour transform.
            Layout::Rgba16F | Layout::Bgra16F => vk::Format::R16G16B16A16_SFLOAT,
        };
        let id = self.images.find_display(addr, desc.width, desc.height)?;
        let img = self.images.get(id)?;
        if img.key.main.bytes_per_element as usize != layout.bytes_per_pixel() || img.key.samples != 1 {
            return None;
        }
        let src = img.image.image;
        let transform = if layout == Layout::Bgra16F && img.key.transform == crate::tiler::ColorTransform::None {
            crate::tiler::ColorTransform::SwapBgra16
        } else {
            crate::tiler::ColorTransform::None
        };
        let snap = self.snapshot(desc.width, desc.height, format)?;
        self.sched.full_barrier();
        if transform != crate::tiler::ColorTransform::None {
            if !self.images.copy_transformed(&mut self.sched, id, snap.image.image, desc.width, desc.height, transform) {
                return None;
            }
            self.sched.full_barrier();
            let tick = self.sched.flush();
            self.stats.captures += 1;
            return Some(GpuFrame { snapshot: snap, timeline: self.sched.timeline.semaphore, tick });
        }
        self.images.transit(&mut self.sched, id, crate::images::TRANSFER_SRC, None);
        let sub = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
        let region = vk::ImageCopy::default().src_subresource(sub).dst_subresource(sub).extent(vk::Extent3D {
            width: desc.width,
            height: desc.height,
            depth: 1,
        });
        let cb = self.sched.cmd();
        // SAFETY: recording buffer; the image in TRANSFER_SRC, the snapshot in GENERAL layout, size-compatible formats.
        unsafe { self.dev.device.cmd_copy_image(cb, src, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, snap.image.image, vk::ImageLayout::GENERAL, &[region]) };
        self.sched.full_barrier();
        let tick = self.sched.flush();
        self.stats.captures += 1;
        Some(GpuFrame { snapshot: snap, timeline: self.sched.timeline.semaphore, tick })
    }

    fn unmapped(&mut self, addr: u64, size: u64) {
        self.buffers.unmap(&mut self.sched, addr, size);
        self.images.unmap(&mut self.sched, addr, size);
    }
}

// ---------------------------------------------------------------------------
/// GPU-time labels of back-end work other than draws and dispatches
/// (`RPS5E_GPU_PROFILE=1`; draws and dispatches use shader addresses).
use crate::scheduler::gpu_timer::label as gpu_label;

// The process-wide back-end
// ---------------------------------------------------------------------------

const UNINIT: u8 = 0;
const READY: u8 = 1;
const FAILED: u8 = 2;

struct Cell {
    inner: Mutex<Option<Backend>>,
    owner: AtomicI32,
    state: AtomicU8,
}

static CELL: Cell = Cell { inner: Mutex::new(None), owner: AtomicI32::new(0), state: AtomicU8::new(UNINIT) };

struct Guard {
    g: MutexGuard<'static, Option<Backend>>,
    /// The call site and the moment it took the lock (`RPS5E_LOCK_SITES`).
    held: Option<(&'static std::panic::Location<'static>, std::time::Instant, u64)>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        CELL.owner.store(0, Ordering::Release);
        if let Some((at, t0, waited)) = self.held.take() {
            profile::note_lock(at, waited, t0.elapsed().as_nanos() as u64);
        }
    }
}

fn enabled() -> bool {
    !std::env::var("RPS5E_GPU_BACKEND").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off"))
}

#[track_caller]
fn lock() -> Option<Guard> {
    if CELL.state.load(Ordering::Acquire) == FAILED {
        return None;
    }
    let me = thread_id();
    if CELL.owner.load(Ordering::Acquire) == me {
        return None;
    }
    let sites = profile::lock_sites();
    let at = std::panic::Location::caller();
    let t0 = sites.then(std::time::Instant::now);
    let tw = (ab::enabled() && crate::perf::sys::is_recorder()).then(std::time::Instant::now);
    let mut g = profile::time(7, || CELL.inner.lock().unwrap_or_else(|e| e.into_inner()));
    CELL.owner.store(me, Ordering::Release);
    if let Some(tw) = tw {
        note_pipeline_ab(2, tw.elapsed().as_micros() as u64);
    }
    let held = t0.map(|t0| {
        let now = std::time::Instant::now();
        (at, now, (now - t0).as_nanos() as u64)
    });
    if g.is_none() {
        if CELL.state.load(Ordering::Acquire) != UNINIT || !enabled() {
            CELL.state.store(FAILED, Ordering::Release);
            CELL.owner.store(0, Ordering::Release);
            return None;
        }
        let Some(dev) = GpuDevice::global() else {
            CELL.state.store(FAILED, Ordering::Release);
            CELL.owner.store(0, Ordering::Release);
            return None;
        };
        let memory = crate::memory::global();
        match Backend::new(dev, memory) {
            Ok(b) => {
                attach_faults(&b, memory);
                log::info!("ps5gpu: GPU back-end ready on '{}'", dev.name);
                *g = Some(b);
                CELL.state.store(READY, Ordering::Release);
                // Teardown: the recording thread stops by its own hook
                // (same order, registered earlier); dropping the back-end
                // joins the pipeline compile pool and the cache writer
                // (bounded, in their shutdown) and runs the final
                // pipeline-cache save before the driver can be unloaded.
                ps5vm::shutdown::on_shutdown(ps5vm::shutdown::ORDER_GPU, "GPU back-end", || {
                    let mut g = CELL.inner.lock().unwrap_or_else(|e| e.into_inner());
                    *g = None;
                });
            }
            Err(e) => {
                log::warn!("ps5gpu: GPU back-end initialisation failed: {e:?}");
                CELL.state.store(FAILED, Ordering::Release);
                CELL.owner.store(0, Ordering::Release);
                return None;
            }
        }
    }
    Some(Guard { g, held })
}

/// Route the CPU faults on `memory`'s watched pages to the caches of `b`
/// (once per memory manager).
pub(crate) fn attach_faults(b: &Backend, memory: &GuestGpuMemory) {
    memory.attach(
        Arc::new(FaultBuffers { tracker: b.tracker.clone(), timeline: b.sched.timeline.clone(), window_end: AtomicU64::new(0), window_size: AtomicU64::new(0) }),
        Arc::new(FaultImages { table: b.images.table.clone(), tracker: b.tracker.clone() }),
        Arc::new(Unmap),
    );
}

/// Run `f` with the back-end (initialised on first use). None when the
/// back-end is unavailable or the calling thread already holds it.
#[track_caller]
pub fn with_backend<R>(f: impl FnOnce(&mut Backend) -> R) -> Option<R> {
    let mut g = lock()?;
    let b = g.g.as_mut()?;
    Some(f(b))
}

/// A write of the command processor the guest can observe is about to be
/// made (a label): GPU results it orders reach guest memory before a CPU
/// read of them ([`Backend::publish_point`]).
#[track_caller]
pub fn publish_point() {
    with_ready_backend(|b| b.publish_point());
}

/// An ordered write of the command processor (see [`Backend::cp_write`]):
/// the publish point, then the write on the GPU, or `write` on the CPU.
#[track_caller]
pub fn ordered_cp_write(addr: u64, data: &[u8], write: Box<dyn FnOnce() + Send>) {
    publish_point();
    if !with_ready_backend(|b| b.cp_write(addr, data)).unwrap_or(false) {
        write();
    }
}

/// Run `f` only when the back-end already exists (fault / flip paths).
#[track_caller]
pub fn with_ready_backend<R>(f: impl FnOnce(&mut Backend) -> R) -> Option<R> {
    if CELL.state.load(Ordering::Acquire) != READY {
        return None;
    }
    with_backend(f)
}

/// Install the shader back-end's pipeline provider.
pub fn set_pipeline_provider(p: Box<dyn PipelineProvider>) -> bool {
    with_backend(|b| b.pipelines = p).is_some()
}

/// Back-end counters (None before initialisation).
pub fn stats() -> Option<BackendStats> {
    with_ready_backend(|b| b.stats)
}

/// Write the GPU-rendered image of a display buffer back to guest memory
/// (debug dumps). False when the GPU did not render it.
pub fn writeback_display(desc: &crate::SurfaceDesc, addr: u64) -> bool {
    with_ready_backend(|b| b.writeback_display(desc, addr)).unwrap_or(false)
}

/// Bring the GPU-written pages of `[addr, addr+size)` back to guest memory
/// (waits for the GPU only when there are any).
#[track_caller]
pub fn download_range(addr: u64, size: u64) {
    with_ready_backend(|b| {
        if b.buffers.has_gpu_dirty(addr, size) {
            b.buffers.download(&mut b.sched, addr, size);
        }
    });
}

/// The rendered image of a display buffer, when the GPU produced it.
pub fn capture_display(desc: &crate::SurfaceDesc, addr: u64) -> Option<GpuFrame> {
    profile::time(4, || with_ready_backend(|b| b.capture_display(desc, addr))).flatten()
}

/// Aligned guest window whose GPU-written pages a CPU fault downloads.
const FAULT_DOWNLOAD_WINDOW: u64 = 1 << 20;
/// Largest window a run of consecutive faults grows to.
const FAULT_WINDOW_MAX: u64 = 32 << 20;

/// `RPS5E_FAULT_WINDOW_GROW=0`: every fault downloads the 1 MiB window
/// around it. Otherwise a fault right after the window of the previous one
/// (a CPU copying a large GPU-written range out) doubles the window, up to
/// [`FAULT_WINDOW_MAX`]: one GPU wait per doubling instead of per MiB.
fn fault_window_grow_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    ab::new_fast() && *ON.get_or_init(|| ab::switch_on("RPS5E_FAULT_WINDOW_GROW"))
}

/// The window a fault at `addr` downloads, given the end of the previous
/// fault's window and its size: `(start, end, size)`; the next fault
/// passes back `end` and `size`. A fault inside the previous window also
/// grows it (the GPU keeps rewriting pages the CPU polls), not only one
/// right after it.
pub(crate) fn fault_window(addr: u64, size: u64, last_end: u64, last_size: u64, grow: bool) -> (u64, u64, u64) {
    let min = FAULT_DOWNLOAD_WINDOW;
    let start = addr & !(min - 1);
    let repeat = last_size >= min && start >= last_end - last_size && start < last_end;
    let w = if grow && (start == last_end || repeat) { (last_size * 2).min(FAULT_WINDOW_MAX) } else { min };
    let end = (start + w).max((addr + size).div_ceil(min) * min);
    (start, end, w)
}
/// Unlocked download rounds before a fault downloads holding the back-end.
const FAULT_DOWNLOAD_ROUNDS: usize = 4;

struct FaultBuffers {
    tracker: Arc<GuestPageTracker>,
    timeline: Arc<crate::scheduler::Timeline>,
    /// End and size of the last fault's download window (a hint shared by
    /// the faulting threads).
    window_end: AtomicU64,
    window_size: AtomicU64,
}

/// CPU accesses that waited (without the back-end) for a copy into guest
/// memory already submitted.
pub static LANDING_WAITS: AtomicU64 = AtomicU64::new(0);

impl FaultBuffers {
    /// Without the back-end and allocation-free: every page of
    /// `[first, last]` (page addresses) is clean, or its GPU contents were
    /// copied into guest memory by a completed tick (the page is made
    /// clean). A copy already submitted is waited for; one still being
    /// recorded, or none at all, needs the back-end (false).
    fn take_landed(&self, first: u64, last: u64) -> bool {
        use crate::memory::tracker::Landing;
        let mut page = first;
        let mut done = self.timeline.completed_tick();
        // GPU work recorded meanwhile may rewrite (and copy out) a page
        // again: a few rounds, then the back-end decides.
        let mut waits = 0;
        loop {
            match self.tracker.take_landed_in_fault(page, done) {
                Some(Landing::Clean | Landing::Landed) => {
                    if page >= last {
                        return true;
                    }
                    page += crate::memory::PAGE_BYTES;
                }
                Some(Landing::InFlight(tick)) => {
                    let fresh = self.timeline.refresh();
                    if fresh < tick {
                        // The tick of the open command buffer is submitted
                        // only by the thread recording it: ask it to submit
                        // at its next item, so the wait the
                        // caller takes below ends with the tick, not the
                        // slice.
                        if tick >= self.timeline.current_tick() || waits == FAULT_DOWNLOAD_ROUNDS {
                            request_submit(tick);
                            return false;
                        }
                        waits += 1;
                        LANDING_WAITS.fetch_add(1, Ordering::Relaxed);
                        self.timeline.wait(tick);
                    }
                    done = self.timeline.completed_tick().max(tick);
                }
                Some(Landing::Missing) | None => return false,
            }
        }
    }

    /// A CPU access to GPU-dirty pages with landing copies
    /// ([`crate::memory::landing_enabled`]): pages whose copy into guest
    /// memory completed or is submitted cost no back-end lock and no new
    /// GPU work. Otherwise the window's pages are copied out (one GPU wait
    /// for the neighbours too, recorded behind the queued work), and the
    /// access is done as soon as its own pages are: GPU work rewriting
    /// other pages of the window meanwhile does not hold it. The faulting
    /// pages are remembered, so later GPU writes over them are written
    /// through after the work and the next access finds them landed.
    fn download_landed(&self, addr: u64, size: u64, start: u64, end: u64) -> bool {
        let page_bytes = crate::memory::PAGE_BYTES;
        let first = addr & !(page_bytes - 1);
        let last = (addr + size.max(1) - 1) & !(page_bytes - 1);
        let own = last + page_bytes - first;
        if self.take_landed(first, last) {
            return true;
        }
        let r = (|| {
            for round in 0..FAULT_DOWNLOAD_ROUNDS {
                let (s, n) = if round == 0 { (start, end - start) } else { (first, own) };
                let step = with_ready_backend(|b| {
                    if round == 0 {
                        b.buffers.note_cpu_access(first, own);
                    }
                    b.buffers.download_async(&mut b.sched, s, n)
                })?;
                match step {
                    DownloadStep::Done(ok) => {
                        if round > 0 || self.tracker.has_gpu_dirty_pages_in_fault(first, own) == Some(false) {
                            return Some(ok);
                        }
                    }
                    DownloadStep::Wait(tick) => {
                        request_submit(tick);
                        self.timeline.wait(tick);
                        if self.take_landed(first, last) {
                            return Some(true);
                        }
                    }
                }
            }
            // GPU work keeps rewriting the pages themselves: download them
            // while holding the back-end.
            with_ready_backend(|b| b.buffers.download(&mut b.sched, first, own))
        })();
        r.unwrap_or(false)
    }
}

impl BufferStore for FaultBuffers {
    fn mark_cpu_write(&self, addr: u64, size: u64) -> bool {
        if profile::on() {
            profile::note_write_fault(addr);
        }
        match self.tracker.invalidate_region(addr, size, true) {
            Some((tracked, false)) => tracked,
            Some((_, true)) => {
                self.download_to_cpu(addr, size) && matches!(self.tracker.invalidate_region(addr, size, true), Some((_, false)))
            }
            None => false,
        }
    }

    fn download_to_cpu(&self, addr: u64, size: u64) -> bool {
        if self.tracker.has_gpu_dirty_pages_in_fault(addr, size) != Some(true) {
            return false;
        }
        crate::memory::history::note(crate::memory::history::Event::CpuRead, addr, size, 0);
        // Every download waits for the GPU, while the copy itself is cheap:
        // take the GPU-written pages around the fault along (CPU accesses
        // are mostly sequential).
        let (start, end, w) =
            fault_window(addr, size, self.window_end.load(Ordering::Relaxed), self.window_size.load(Ordering::Relaxed), fault_window_grow_enabled());
        self.window_end.store(end, Ordering::Relaxed);
        self.window_size.store(w, Ordering::Relaxed);
        let t = profile::on().then(std::time::Instant::now);
        if crate::memory::landing_enabled() {
            let r = self.download_landed(addr, size, start, end);
            if let Some(t) = t {
                let ns = t.elapsed().as_nanos() as u64;
                profile::NANOS[5].fetch_add(ns, Ordering::Relaxed);
                profile::note_fault(start, addr, ns);
            }
            return r;
        }
        let r = (|| {
            // The back-end is not held while the GPU catches up with the
            // copies: the render thread keeps recording meanwhile.
            for _ in 0..FAULT_DOWNLOAD_ROUNDS {
                let (step, timeline) = with_ready_backend(|b| (b.buffers.download_async(&mut b.sched, start, end - start), b.sched.timeline.clone()))?;
                match step {
                    DownloadStep::Done(ok) => return Some(ok),
                    DownloadStep::Wait(tick) => timeline.wait(tick),
                }
            }
            // GPU work keeps rewriting the pages: download while holding it.
            with_ready_backend(|b| b.buffers.download(&mut b.sched, start, end - start))
        })()
        .unwrap_or(false);
        if let Some(t) = t {
            let ns = t.elapsed().as_nanos() as u64;
            profile::NANOS[5].fetch_add(ns, Ordering::Relaxed);
            profile::note_fault(start, addr, ns);
        }
        r
    }

    fn settle(&self, addr: u64, write: bool) -> Option<bool> {
        let page_bytes = crate::memory::PAGE_BYTES;
        let page = addr & !(page_bytes - 1);
        let kind = if write { crate::memory::FaultKind::Write } else { crate::memory::FaultKind::Read };
        // Holding the back-end: no work is recorded meanwhile, so nothing
        // dirties or watches the page again before the access retries.
        with_ready_backend(|b| {
            // GPU-written bytes of the page first (waits for the GPU).
            if self.tracker.has_gpu_dirty_pages_in_fault(page, page_bytes) == Some(true) {
                b.buffers.download(&mut b.sched, page, page_bytes);
            }
            // Images holding GPU results there are read back.
            let table = b.images.table.clone();
            if let Some(hits) = table.take_read_backs(page) {
                if !hits.is_empty() {
                    let tracker = b.tracker.clone();
                    b.images.resolve_read_backs(&mut b.sched, &mut b.buffers, &tracker, &hits, true);
                }
            }
            let pages = table.pages();
            // A read watch nobody holds any more (no image read-back left
            // and no GPU-dirty page under it).
            if pages.watchers(page).0 && self.tracker.has_gpu_dirty_pages_in_fault(page, page_bytes) == Some(false) {
                pages.remove_watch_in_fault(page, page_bytes, true);
            }
            let (mut invalidated, mut marked) = (None, false);
            if write {
                invalidated = self.tracker.invalidate_region(addr, 8, true);
                marked = table.mark_cpu_write(addr, 8, true);
            }
            let ok = pages.allows(addr, kind);
            if !ok {
                static LOGGED: AtomicU64 = AtomicU64::new(0);
                if LOGGED.fetch_add(1, Ordering::Relaxed) < 4 {
                    log::warn!(
                        "ps5gpu: settle of {addr:#x} ({kind:?}) left it watched: watchers {:?}, buffer page (cpu-dirty, gpu-dirty) {:?}, invalidate {invalidated:?}, image hit {marked}, images [{}]",
                        pages.watchers(page),
                        self.tracker.page_state(page),
                        table.describe_page(page)
                    );
                }
            }
            ok
        })
    }

    fn mapped(&self, addr: u64, size: u64) {
        // GPU results over memory mapped anew belonged to the old pages (an
        // unmap downloaded them first; on the console GPU writes to an
        // uncommitted page are dropped): the guest bytes are current.
        if self.tracker.has_gpu_dirty_pages(addr, size) {
            self.tracker.clear_gpu_dirty_pages(addr, size);
        }
        let _ = self.tracker.invalidate_region(addr, size, false);
    }
}

struct FaultImages {
    table: Arc<crate::images::WatchTable>,
    tracker: Arc<GuestPageTracker>,
}

impl ImageStore for FaultImages {
    fn mark_cpu_write(&self, addr: u64, size: u64) -> bool {
        let hit = self.table.mark_cpu_write(addr, size, true);
        if hit && profile::on() {
            profile::note_image_write(addr);
        }
        hit
    }

    fn mapped(&self, addr: u64, size: u64) {
        self.table.mark_mapped(addr, size);
    }

    fn cpu_access(&self, addr: u64, _size: u64, _write: bool) -> bool {
        let Some(hits) = self.table.take_read_backs(addr) else { return false };
        if hits.is_empty() {
            // A read watch nobody holds (an image's, left behind when its
            // removal met a busy lock): no buffer has GPU data there.
            let page = addr & !(crate::memory::PAGE_BYTES - 1);
            let pages = self.table.pages();
            if pages.watchers(page).0 && self.tracker.has_gpu_dirty_pages_in_fault(page, crate::memory::PAGE_BYTES) == Some(false) {
                return pages.remove_watch_in_fault(page, crate::memory::PAGE_BYTES, true);
            }
            return false;
        }
        if crate::images::unlocked_target_read_enabled() {
            // The GPU catches up with the read-back's copies while the
            // back-end records on; the last round finishes without a wait.
            for _ in 0..FAULT_DOWNLOAD_ROUNDS {
                let step = with_ready_backend(|b| {
                    match b.images.start_read_backs(&mut b.sched, &mut b.buffers, &hits) {
                        Some(tick) => Some((tick, b.sched.timeline.clone())),
                        None => {
                            let tracker = b.tracker.clone();
                            b.images.resolve_read_backs(&mut b.sched, &mut b.buffers, &tracker, &hits, true);
                            None
                        }
                    }
                });
                match step {
                    Some(Some((tick, timeline))) => timeline.wait(tick),
                    _ => return true,
                }
            }
        }
        with_ready_backend(|b| {
            let tracker = b.tracker.clone();
            // A write too: the GPU work storing to the image was submitted
            // first, so its results land before the CPU's bytes.
            b.images.resolve_read_backs(&mut b.sched, &mut b.buffers, &tracker, &hits, true);
        });
        true
    }
}

struct Unmap;

impl UnmapListener for Unmap {
    fn unmapped(&self, addr: u64, size: u64) {
        with_ready_backend(|b| b.unmapped(addr, size));
    }
}

/// `RPS5E_GPU_PROFILE=1`: wall time per back-end entry point, in the
/// periodic summary (ms since start).
/// A/B measurement of the per-draw fast paths in one run
/// (`RPS5E_AB_PERIOD=<seconds>`): they are on during even periods and off
/// during odd ones (memoised bindings, stream copies and preparations,
/// precise dispatch barriers, read-only depth scopes, draw state across
/// scopes, the recording thread). Draws and recording-thread CPU time are
/// counted per mode (`ab=` in the summary), so both modes see the same
/// scenes and the same machine load.
/// `RPS5E_LOADED_FILL_HLE=0`: a constant-fill kernel whose value is loaded
/// through a V# runs as a dispatch (its DCC key fills are then decided on
/// the GPU after refreshing the target). On by default; A/B in one run with
/// `RPS5E_AB_SCOPE=release RPS5E_AB_ONLY=loaded-fill`.
fn loaded_fill_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| ab::switch_on("RPS5E_LOADED_FILL_HLE")) && ab::release_fast("loaded-fill")
}

/// `RPS5E_CP_WRITES_ON_GPU=0`: labels and WRITE_DATA into pages holding
/// GPU-written bytes are written on the CPU (the store faults and downloads
/// the page first). On by default ([`Backend::cp_write`]); A/B in one run
/// with `RPS5E_AB_SCOPE=release RPS5E_AB_ONLY=cp-writes-on-gpu`.
fn cp_writes_on_gpu_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| ab::switch_on("RPS5E_CP_WRITES_ON_GPU")) && ab::release_fast("cp-writes-on-gpu")
}

pub mod ab {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    fn period() -> Option<u64> {
        static P: OnceLock<Option<u64>> = OnceLock::new();
        *P.get_or_init(|| std::env::var("RPS5E_AB_PERIOD").ok().and_then(|v| v.parse::<u64>().ok()).filter(|&p| p > 0))
    }

    fn start() -> std::time::Instant {
        static S: OnceLock<std::time::Instant> = OnceLock::new();
        *S.get_or_init(std::time::Instant::now)
    }

    #[inline]
    fn toggle() -> bool {
        live_fast()
            && match config().period {
                None => true,
                Some(p) => in_fast_period(p),
            }
    }

    #[inline(never)]
    fn in_fast_period(p: u64) -> bool {
        (start().elapsed().as_secs() / p) % 2 == 0
    }

    /// The per-draw fast paths can also be turned off while running (the
    /// in-game menu; `RPS5E_FAST_PATHS=0` at start): the same switch as
    /// the A/B slow periods. 0: not read yet, 1: off, 2: on.
    static LIVE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

    pub fn set_live_fast(on: bool) {
        LIVE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    }

    #[inline]
    pub fn live_fast() -> bool {
        match LIVE.load(Ordering::Relaxed) {
            0 => live_fast_first(),
            v => v == 2,
        }
    }

    #[cold]
    fn live_fast_first() -> bool {
        let on = super::ab::switch_value(false, std::env::var("RPS5E_FAST_PATHS").ok().as_deref());
        let _ = LIVE.compare_exchange(0, if on { 2 } else { 1 }, Ordering::Relaxed, Ordering::Relaxed);
        LIVE.load(Ordering::Relaxed) == 2
    }

    /// `RPS5E_AB_SCOPE`: `all` (default), `thread` (only the recording
    /// thread toggles), `paths` (only the per-draw paths toggle), `new`
    /// (only the paths of GPU_PLAN §23 toggle: draw recipes, upload
    /// prologues, settle skips, range and stream trimming) or `bda` (only
    /// guest-memory reads of GPU_PLAN §27) or `release` (only the paths of
    /// the release check: the watch-table index).
    /// Parsed once: the per-draw switches ask on every call.
    fn scope() -> Scope {
        static S: OnceLock<Scope> = OnceLock::new();
        *S.get_or_init(|| match std::env::var("RPS5E_AB_SCOPE").as_deref() {
            Ok("thread") => Scope::Thread,
            Ok("paths") => Scope::Paths,
            Ok("new") => Scope::New,
            Ok("bda") => Scope::Bda,
            Ok("release") => Scope::Release,
            _ => Scope::All,
        })
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Scope {
        All,
        Thread,
        Paths,
        New,
        Bda,
        Release,
    }

    /// `RPS5E_SAFE=1`: every optimisation that may hand the GPU older
    /// bytes or skip work is off at once (diagnostic switch):
    /// the per-draw memos and recipes of §21/§23 (and through them the
    /// label sync points), the upload prologue, read memos, view memos,
    /// shared scopes, asynchronous pipelines (no skipped draws), EXEC runs,
    /// GPU-filled table words (read back instead), indirect draws on the
    /// GPU, on-access linear-image read-back (eager instead), hot pages,
    /// and the BDA path.
    pub fn safe() -> bool {
        static S: OnceLock<bool> = OnceLock::new();
        *S.get_or_init(|| {
            let on = std::env::var("RPS5E_SAFE").is_ok_and(|v| v == "1");
            if on {
                log::warn!("ps5gpu: RPS5E_SAFE=1: fast paths off (memos, recipes, prologue, async pipelines, exec runs, deferred table words, GPU indirect)");
            }
            on
        })
    }

    /// `!safe()` and the switch `name` is not `0` (memoised by the caller).
    pub fn switch_on(name: &str) -> bool {
        switch_value(safe(), std::env::var(name).ok().as_deref())
    }

    /// A fast-path switch: off in safe mode or when set to `0`/`off`.
    pub(crate) fn switch_value(safe: bool, value: Option<&str>) -> bool {
        !safe && !value.is_some_and(|v| v == "0" || v.eq_ignore_ascii_case("off"))
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn safe_mode_turns_every_switch_off() {
            use super::switch_value;
            assert!(switch_value(false, None));
            assert!(switch_value(false, Some("1")));
            assert!(!switch_value(false, Some("0")));
            assert!(!switch_value(false, Some("OFF")));
            assert!(!switch_value(true, None));
            assert!(!switch_value(true, Some("1")));
        }
    }

    /// The start-up switches the per-draw questions below combine, read
    /// once (they are asked many times per draw).
    struct Config {
        safe: bool,
        scope: Scope,
        period: Option<u64>,
        /// `RPS5E_AB_ONLY`: the release-check switches scope `release`
        /// toggles, comma-separated (None: all of them).
        only: Option<String>,
    }

    #[inline]
    fn config() -> &'static Config {
        static C: OnceLock<Config> = OnceLock::new();
        C.get_or_init(|| Config {
            safe: safe(),
            scope: scope(),
            period: period(),
            only: std::env::var("RPS5E_AB_ONLY").ok().filter(|v| !v.is_empty()),
        })
    }

    /// The per-draw fast paths are on now.
    #[inline]
    pub fn fast() -> bool {
        let c = config();
        !c.safe && (matches!(c.scope, Scope::Thread | Scope::New | Scope::Bda | Scope::Release) || toggle())
    }

    /// The fast paths of GPU_PLAN §23 are on now.
    #[inline]
    pub fn new_fast() -> bool {
        let c = config();
        !c.safe && (matches!(c.scope, Scope::Thread | Scope::Bda | Scope::Release) || toggle())
    }

    /// Draws go through the recording thread now.
    #[inline]
    pub fn thread_fast() -> bool {
        matches!(config().scope, Scope::Paths | Scope::New | Scope::Bda | Scope::Release) || toggle()
    }

    /// The release-check path behind switch `name` is on now: on unless
    /// scope `release` is in a slow period (with `RPS5E_AB_ONLY=<switch>,…`
    /// only those paths toggle). Its own switch is separate.
    #[inline]
    pub fn release_fast(name: &str) -> bool {
        let c = config();
        c.scope != Scope::Release || c.only.as_deref().is_some_and(|o| !o.split(',').any(|s| s.trim() == name)) || toggle()
    }

    /// Read-only stages read guest memory through its imports now
    /// (`RPS5E_BDA=1`; scope `bda` toggles only this).
    #[inline]
    pub fn bda_fast() -> bool {
        let c = config();
        !c.safe && (c.scope != Scope::Bda || toggle())
    }

    /// The mode the draw counters are charged to.
    pub fn mode_fast() -> bool {
        toggle()
    }

    pub fn enabled() -> bool {
        period().is_some()
    }

    /// Draws recorded and recording CPU nanoseconds, fast mode then slow.
    pub static DRAWS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    pub static CPU_NS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    pub fn note_draw(fast: bool, cpu_ns: u64) {
        let m = !fast as usize;
        DRAWS[m].fetch_add(1, Ordering::Relaxed);
        CPU_NS[m].fetch_add(cpu_ns, Ordering::Relaxed);
    }

    /// Flips, fast mode then slow: frames per mode, the game's own speed.
    pub static FLIPS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    /// Flips longer than 100 ms after the one before (freezes), per mode.
    pub static FREEZES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

    /// GPU busy nanoseconds of the command buffers submitted in each mode
    /// (with the GPU profile's timer).
    pub static GPU_NS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    pub fn note_gpu(fast: bool, ns: u64) {
        if enabled() {
            GPU_NS[!fast as usize].fetch_add(ns, Ordering::Relaxed);
        }
    }

    /// A flip (counted in the mode of the moment).
    pub fn note_flip() {
        if !enabled() {
            return;
        }
        static LAST: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
        let m = !mode_fast() as usize;
        FLIPS[m].fetch_add(1, Ordering::Relaxed);
        let now = std::time::Instant::now();
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if last.is_some_and(|l| now.duration_since(l).as_millis() > 100) {
            FREEZES[m].fetch_add(1, Ordering::Relaxed);
        }
        *last = Some(now);
    }

    pub fn line() -> String {
        if !enabled() {
            return String::new();
        }
        let g = |a: &[AtomicU64; 2]| [a[0].load(Ordering::Relaxed), a[1].load(Ordering::Relaxed)];
        format!(" ab_draws={:?} ab_cpu_ms={:?} ab_flips={:?} ab_freezes={:?} ab_gpu_ms={:?}", g(&DRAWS), g(&CPU_NS).map(|n| n / 1_000_000), g(&FLIPS), g(&FREEZES), g(&GPU_NS).map(|n| n / 1_000_000))
    }
}

pub(crate) mod profile {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    /// Spans of the base profile (`RPS5E_GPU_PROFILE=1`); the ones after
    /// them are the sub-phases of draw recording, printed (and measured)
    /// only with `RPS5E_GPU_PROFILE=2`.
    pub const BASE_SPANS: usize = 35;
    pub const NAMES: [&str; 51] = [
        "draw", "dispatch", "eop", "flush", "capture", "fault_dl", "fill", "lock", "targets", "prep", "bind", "module", "obtain", "imgview", "img_prep", "devrange", "dcc", "tkey", "tfind",
        "tview", "tgt_find", "wfault", "mat", "d_stages", "d_modules", "d_targets", "d_vinput", "d_ranges", "d_descs", "d_index", "d_pipe", "d_record", "d_post", "draw_cpu", "cmd_cpu",
        // Sub-phases (RPS5E_GPU_PROFILE=2): the front-end preparation,
        // the descriptor loop of `bind_stage` and the record phase.
        "prep_memo", "spec_apply", "layout_find", "shader_data", "stage_ranges", "desc_words", "desc_bind", "commit", "scope_state", "draw_cmd", "hazards", "sdata_build", "stream_copy", "rendering",
        // GPU_PLAN §56: the flattened-table binding (compare and upload).
        "table_cmp", "table_up",
    ];
    pub static NANOS: [AtomicU64; 51] = [const { AtomicU64::new(0) }; 51];
    /// CPU time of the calling thread (ns).
    #[cfg(windows)]
    pub fn thread_cpu() -> u64 {
        ps5vm::os::thread_cpu_ns()
    }
    #[cfg(unix)]
    pub fn thread_cpu() -> u64 {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: valid out pointer.
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
        ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
    }
    /// `draw_cpu`: the thread CPU time of `f`; `cmd_cpu`: the total CPU
    /// time of the command thread (the thread that records draws).
    pub fn time_cpu<R>(f: impl FnOnce() -> R) -> R {
        if !on() {
            return f();
        }
        let t = thread_cpu();
        let r = f();
        let now = thread_cpu();
        add(33, now - t);
        NANOS[34].store(now, Ordering::Relaxed);
        r
    }
    /// `RPS5E_PROFILE_LOCAL=0`: every span is added to the shared counters
    /// at once (two threads recording contend on their cache lines).
    fn local_enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| !std::env::var("RPS5E_PROFILE_LOCAL").is_ok_and(|v| v == "0"))
    }
    /// Spans added between looks at the clock.
    const LOCAL_CHECK: u32 = 256;
    /// Spans are moved to the shared counters at least this often.
    const LOCAL_FLUSH: std::time::Duration = std::time::Duration::from_millis(50);
    /// The calling thread's spans not yet in [`NANOS`].
    struct Local {
        nanos: [std::cell::Cell<u64>; NAMES.len()],
        adds: std::cell::Cell<u32>,
        flushed: std::cell::Cell<Option<std::time::Instant>>,
    }
    impl Local {
        fn flush(&self) {
            for (n, c) in NANOS.iter().zip(&self.nanos) {
                let v = c.replace(0);
                if v != 0 {
                    n.fetch_add(v, Ordering::Relaxed);
                }
            }
        }
    }
    impl Drop for Local {
        fn drop(&mut self) {
            self.flush();
        }
    }
    thread_local! {
        static LOCAL: Local = const { Local { nanos: [const { std::cell::Cell::new(0) }; NAMES.len()], adds: std::cell::Cell::new(0), flushed: std::cell::Cell::new(None) } };
    }
    /// Charge `ns` to span `k` (thread-locally, moved to [`NANOS`] every
    /// [`LOCAL_FLUSH`]).
    fn add(k: usize, ns: u64) {
        if crate::perf::items::enabled() {
            crate::perf::items::note_span(k, ns);
        }
        if !local_enabled() {
            NANOS[k].fetch_add(ns, Ordering::Relaxed);
            return;
        }
        let kept = LOCAL.try_with(|l| {
            l.nanos[k].set(l.nanos[k].get() + ns);
            let n = l.adds.get().wrapping_add(1);
            l.adds.set(n);
            if n % LOCAL_CHECK == 0 {
                let now = std::time::Instant::now();
                if l.flushed.get().is_none_or(|t| now - t >= LOCAL_FLUSH) {
                    l.flushed.set(Some(now));
                    l.flush();
                }
            }
        });
        if kept.is_err() {
            NANOS[k].fetch_add(ns, Ordering::Relaxed);
        }
    }
    /// Move the calling thread's spans to [`NANOS`] now.
    pub fn flush_local() {
        let _ = LOCAL.try_with(|l| l.flush());
    }
    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn thread_local_spans_reach_the_shared_counters() {
            let before = NANOS[30].load(Ordering::Relaxed);
            add(30, 5);
            flush_local();
            assert!(NANOS[30].load(Ordering::Relaxed) >= before + 5, "flushed on request");
            // A thread's spans are flushed when it ends.
            let before = NANOS[31].load(Ordering::Relaxed);
            std::thread::spawn(|| add(31, 7)).join().unwrap();
            assert!(NANOS[31].load(Ordering::Relaxed) >= before + 7, "flushed at thread exit");
            // And now and then while it adds.
            let before = NANOS[32].load(Ordering::Relaxed);
            std::thread::spawn(move || {
                for _ in 0..LOCAL_CHECK {
                    add(32, 1);
                }
                assert!(NANOS[32].load(Ordering::Relaxed) >= before + LOCAL_CHECK as u64, "flushed by the first clock check");
            })
            .join()
            .unwrap();
        }
    }
    /// Charge `ns` to span `k` from this thread (sub-phase spans: called
    /// where a closure cannot borrow).
    pub fn add_local(k: usize, ns: u64) {
        if detailed() {
            add(k, ns);
        }
    }

    /// Consecutive spans of one function: `lap(k)` charges the time since
    /// the previous lap (or creation) to span `k`.
    pub struct Lap(Option<std::time::Instant>);
    impl Lap {
        pub fn start() -> Lap {
            Lap(on().then(std::time::Instant::now))
        }
        pub fn lap(&mut self, k: usize) {
            if let Some(t) = &mut self.0 {
                let now = std::time::Instant::now();
                add(k, (now - *t).as_nanos() as u64);
                *t = now;
            }
        }
    }
    pub fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("RPS5E_GPU_PROFILE").is_ok_and(|v| v == "1" || v == "2"))
    }
    /// `RPS5E_GPU_PROFILE=2`: the sub-phase spans after [`BASE_SPANS`] are
    /// measured and printed (the sub-phases of the draw path; off by
    /// default — every extra span costs a clock read on the recording
    /// thread).
    pub fn detailed() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("RPS5E_GPU_PROFILE").is_ok_and(|v| v == "2"))
    }
    pub fn time<R>(k: usize, f: impl FnOnce() -> R) -> R {
        if !on() {
            return f();
        }
        let t = std::time::Instant::now();
        let r = f();
        add(k, t.elapsed().as_nanos() as u64);
        r
    }
    /// [`Self::time`] for the sub-phase spans (>= [`BASE_SPANS`]):
    /// measured only in the detailed profile.
    pub fn dtime<R>(k: usize, f: impl FnOnce() -> R) -> R {
        if !detailed() {
            return f();
        }
        let t = std::time::Instant::now();
        let r = f();
        add(k, t.elapsed().as_nanos() as u64);
        r
    }
    /// Fault downloads per 1 MiB window (open addressing, no allocation:
    /// the fault path runs in a signal handler).
    const SLOTS: usize = 256;
    static FAULT_WINDOW: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    static FAULT_COUNT: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    static FAULT_NANOS: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    static FAULT_LAST: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    /// Instruction pointer of the last access that faulted in the window
    /// (the guest code that touched it, or host code for HLE copies).
    static FAULT_RIP: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    pub fn note_fault(window: u64, addr: u64, ns: u64) {
        let key = window | 1;
        let mut i = ((window >> 20).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as usize % SLOTS;
        for _ in 0..SLOTS {
            match FAULT_WINDOW[i].compare_exchange(0, key, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {}
                Err(k) if k == key => {}
                Err(_) => {
                    i = (i + 1) % SLOTS;
                    continue;
                }
            }
            FAULT_COUNT[i].fetch_add(1, Ordering::Relaxed);
            FAULT_NANOS[i].fetch_add(ns, Ordering::Relaxed);
            FAULT_LAST[i].store(addr, Ordering::Relaxed);
            FAULT_RIP[i].store(ps5vm::fault_site::get(), Ordering::Relaxed);
            return;
        }
    }
    /// The work being recorded, for the last GPU writer of a fault window:
    /// kind in bits 56.. ([`WORK_DRAW`] ...), shader address below.
    static WORK: AtomicU64 = AtomicU64::new(0);
    pub const WORK_DRAW: u64 = 1;
    pub const WORK_DISPATCH: u64 = 2;
    pub const WORK_COPY: u64 = 3;
    pub const WORK_FILL: u64 = 4;
    pub const WORK_WRITE: u64 = 5;
    pub const WORK_GDS: u64 = 6;
    pub fn set_work(kind: u64, address: u64) {
        if on() {
            WORK.store(kind << 56 | (address & ((1 << 56) - 1)), Ordering::Relaxed);
        }
    }
    static GPU_WRITE_WINDOW: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    static GPU_WRITE_LAST: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    fn window_slot(window: u64, table: &[AtomicU64; SLOTS], insert: bool) -> Option<usize> {
        let key = window | 1;
        let mut i = ((window >> 20).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as usize % SLOTS;
        for _ in 0..SLOTS {
            let cur = table[i].load(Ordering::Acquire);
            if cur == key {
                return Some(i);
            }
            if cur == 0 {
                if !insert {
                    return None;
                }
                match table[i].compare_exchange(0, key, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => return Some(i),
                    Err(k) if k == key => return Some(i),
                    Err(_) => {}
                }
            }
            i = (i + 1) % SLOTS;
        }
        None
    }
    /// GPU work recorded now writes `[addr, addr+size)`: remember it as the
    /// last writer of the range's 1 MiB windows.
    pub fn note_gpu_write(addr: u64, size: u64) {
        if !on() {
            return;
        }
        let work = WORK.load(Ordering::Relaxed);
        let first = addr >> 20;
        let last = (addr + size.max(1) - 1) >> 20;
        for w in first..=last.min(first + 15) {
            let i = window_slot(w << 20, &GPU_WRITE_WINDOW, true).unwrap_or_else(|| {
                // The table is full: the window takes its home slot (the
                // windows written last stay named).
                let home = ((w).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as usize % SLOTS;
                GPU_WRITE_WINDOW[home].store(w << 20 | 1, Ordering::Release);
                home
            });
            GPU_WRITE_LAST[i].store(work, Ordering::Relaxed);
        }
        if last > first + 15 {
            // Windows past the first 16 are not entered one by one: the
            // write is remembered whole.
            let i = LARGE_NEXT.fetch_add(1, Ordering::Relaxed) as usize % LARGE_SLOTS;
            LARGE_WRITES[i][0].store(addr, Ordering::Relaxed);
            LARGE_WRITES[i][1].store(size, Ordering::Relaxed);
            LARGE_WRITES[i][2].store(work, Ordering::Relaxed);
        }
    }
    /// The latest GPU writes longer than 16 MiB: (address, size, work).
    const LARGE_SLOTS: usize = 32;
    static LARGE_WRITES: [[AtomicU64; 3]; LARGE_SLOTS] = [const { [const { AtomicU64::new(0) }; 3] }; LARGE_SLOTS];
    static LARGE_NEXT: AtomicU64 = AtomicU64::new(0);
    fn writer_of(window: u64) -> String {
        let w = match window_slot(window, &GPU_WRITE_WINDOW, false) {
            Some(i) => GPU_WRITE_LAST[i].load(Ordering::Relaxed),
            None => {
                let large = (0..LARGE_SLOTS).map(|i| [0, 1, 2].map(|k| LARGE_WRITES[i][k].load(Ordering::Relaxed))).find(|&[a, n, _]| n != 0 && a <= window && window < a + n);
                let Some([a, n, work]) = large else { return String::new() };
                return format!("{}[{a:#x}+{}MiB]", work_name(work), n >> 20);
            }
        };
        work_name(w)
    }
    /// `(symbol+offset)` of a host instruction address (an HLE function
    /// that touched guest memory), empty for guest code.
    #[cfg(windows)]
    fn host_symbol(_rip: u64) -> String {
        String::new()
    }
    #[cfg(unix)]
    fn host_symbol(rip: u64) -> String {
        if rip == 0 || ps5vm::ranges::covered(rip, 1) {
            return String::new();
        }
        // SAFETY: Dl_info is plain data; all zeros is a valid value.
        let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
        // SAFETY: dladdr only reads the loader's tables; the out pointer is valid.
        if unsafe { libc::dladdr(rip as *const libc::c_void, &mut info) } == 0 {
            return String::new();
        }
        if !info.dli_sname.is_null() {
            // SAFETY: dladdr returned a NUL-terminated symbol name.
            let name = unsafe { std::ffi::CStr::from_ptr(info.dli_sname) }.to_string_lossy();
            return format!("({name}+{:#x})", rip - info.dli_saddr as u64);
        }
        if info.dli_fname.is_null() {
            return String::new();
        }
        // An internal function (no exported name): the object and offset.
        // SAFETY: dladdr returned a NUL-terminated file name.
        let file = unsafe { std::ffi::CStr::from_ptr(info.dli_fname) }.to_string_lossy();
        format!("({}+{:#x})", file.rsplit('/').next().unwrap_or(&file), rip - info.dli_fbase as u64)
    }
    fn work_name(w: u64) -> String {
        let a = w & ((1 << 56) - 1);
        match w >> 56 {
            WORK_DRAW => format!("<-d{a:#x}"),
            WORK_DISPATCH => format!("<-c{a:#x}"),
            WORK_COPY => "<-copy".into(),
            WORK_FILL => "<-fill".into(),
            WORK_WRITE => "<-write".into(),
            WORK_GDS => "<-gds".into(),
            _ => "<-?".into(),
        }
    }
    /// Synchronous downloads (the caller waits for the GPU) by call site
    /// and 64 KiB window: (count, ns).
    type SiteKey = (&'static str, u32, u64, u64);
    std::thread_local! {
        /// Guest address of the program whose resource preparation runs on
        /// this thread (0: none): synchronous downloads name it.
        static PREPARING: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }
    /// Run `f` with `program` noted as the program being prepared.
    pub fn preparing<T>(program: u64, f: impl FnOnce() -> T) -> T {
        let old = PREPARING.with(|c| c.replace(program));
        let r = f();
        PREPARING.with(|c| c.set(old));
        r
    }
    type SiteMap = std::collections::HashMap<SiteKey, (u64, u64)>;
    static DOWNLOADS: std::sync::Mutex<Option<SiteMap>> = std::sync::Mutex::new(None);
    pub struct DownloadSite(Option<(SiteKey, std::time::Instant)>);
    impl DownloadSite {
        pub fn new(at: &'static std::panic::Location<'static>, addr: u64) -> Self {
            let file = at.file().rsplit('/').next().unwrap_or("");
            DownloadSite(on().then(|| ((file, at.line(), addr & !0xFFFF, PREPARING.with(|c| c.get())), std::time::Instant::now())))
        }
    }
    impl Drop for DownloadSite {
        fn drop(&mut self) {
            if let Some((k, t)) = self.0.take() {
                let ns = t.elapsed().as_nanos() as u64;
                let mut g = DOWNLOADS.lock().unwrap_or_else(|e| e.into_inner());
                let e = g.get_or_insert_with(Default::default).entry(k).or_default();
                e.0 += 1;
                e.1 += ns;
            }
        }
    }
    /// Full barriers by call site: count.
    static BARRIERS: std::sync::Mutex<Option<std::collections::HashMap<(&'static str, u32), u64>>> = std::sync::Mutex::new(None);
    /// `RPS5E_BARRIER_SITES=1` (with the profile): full barriers counted by
    /// call site in the summary.
    fn barrier_sites() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("RPS5E_BARRIER_SITES").is_ok_and(|v| v == "1"))
    }
    pub fn note_full_barrier(at: &'static std::panic::Location<'static>) {
        if !barrier_sites() || !on() {
            return;
        }
        let file = at.file().rsplit('/').next().unwrap_or("");
        let mut g = BARRIERS.lock().unwrap_or_else(|e| e.into_inner());
        *g.get_or_insert_with(Default::default).entry((file, at.line())).or_default() += 1;
    }
    /// Blocking GPU waits by call site: count and wall nanoseconds.
    static WAITS: std::sync::Mutex<Option<std::collections::HashMap<(&'static str, u32, u64), (u64, u64)>>> = std::sync::Mutex::new(None);
    /// `RPS5E_BARRIER_SITES=1` (with the profile) also counts the GPU waits
    /// that blocked, by call site.
    pub fn wait_sites() -> bool {
        barrier_sites() && on()
    }
    pub fn note_wait(at: &'static std::panic::Location<'static>, ns: u64) {
        let file = at.file().rsplit('/').next().unwrap_or("");
        let mut g = WAITS.lock().unwrap_or_else(|e| e.into_inner());
        let program = PREPARING.with(|c| c.get());
        let e = g.get_or_insert_with(Default::default).entry((file, at.line(), program)).or_default();
        e.0 += 1;
        e.1 += ns;
    }
    fn wait_line() -> String {
        let g = WAITS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(m) = g.as_ref() else { return String::new() };
        let mut v: Vec<(&(&str, u32, u64), &(u64, u64))> = m.iter().collect();
        v.sort_unstable_by(|a, b| b.1 .1.cmp(&a.1 .1));
        v.iter().take(12).map(|((f, l, p), (n, ns))| if *p != 0 { format!("{f}:{l}<-p{p:#x}:{n}/{}ms", ns / 1_000_000) } else { format!("{f}:{l}:{n}/{}ms", ns / 1_000_000) }).collect::<Vec<_>>().join(" ")
    }
    /// `RPS5E_LOCK_SITES=1` (with the profile): the back-end lock's holds
    /// by call site and thread (count, held and waited nanoseconds).
    pub fn lock_sites() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("RPS5E_LOCK_SITES").is_ok_and(|v| v == "1")) && on()
    }
    type LockKey = (&'static str, u32, String);
    static LOCKS: std::sync::Mutex<Option<std::collections::HashMap<LockKey, (u64, u64, u64)>>> = std::sync::Mutex::new(None);
    pub fn note_lock(at: &'static std::panic::Location<'static>, waited: u64, held: u64) {
        let file = at.file().rsplit('/').next().unwrap_or("");
        let thread = std::thread::current()
            .name()
            .map(|n| n.split(|c: char| c.is_ascii_digit()).next().unwrap_or(n).chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-').collect::<String>())
            .unwrap_or_else(|| "guest".into());
        let mut g = LOCKS.lock().unwrap_or_else(|e| e.into_inner());
        let e = g.get_or_insert_with(Default::default).entry((file, at.line(), thread)).or_default();
        e.0 += 1;
        e.1 += held;
        e.2 += waited;
    }
    fn lock_line() -> String {
        let g = LOCKS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(m) = g.as_ref() else { return String::new() };
        let mut v: Vec<(&LockKey, &(u64, u64, u64))> = m.iter().collect();
        v.sort_unstable_by(|a, b| (b.1 .1 + b.1 .2).cmp(&(a.1 .1 + a.1 .2)));
        v.iter().take(14).map(|((f, l, t), (n, h, w))| format!("{f}:{l}[{t}]:{n}/{}ms/w{}ms", h / 1_000_000, w / 1_000_000)).collect::<Vec<_>>().join(" ")
    }
    fn barrier_line() -> String {
        let g = BARRIERS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(m) = g.as_ref() else { return String::new() };
        let mut v: Vec<(&(&str, u32), &u64)> = m.iter().collect();
        v.sort_unstable_by(|a, b| b.1.cmp(a.1));
        v.iter().take(8).map(|((f, l), n)| format!("{f}:{l}:{n}")).collect::<Vec<_>>().join(" ")
    }
    fn download_line() -> String {
        let g = DOWNLOADS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(m) = g.as_ref() else { return String::new() };
        let mut v: Vec<(&SiteKey, &(u64, u64))> = m.iter().collect();
        v.sort_unstable_by(|a, b| b.1 .1.cmp(&a.1 .1));
        v.iter().take(8).map(|((f, l, a, p), (n, ns))| if *p != 0 { format!("{f}:{l}@{a:#x}<-p{p:#x}:{n}/{}ms", ns / 1_000_000) } else { format!("{f}:{l}@{a:#x}:{n}/{}ms", ns / 1_000_000) }).collect::<Vec<_>>().join(" ")
    }
    static WRITE_FAULT_WINDOW: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    static WRITE_FAULT_COUNT: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    static WRITE_FAULT_LAST: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    /// A CPU write fault on buffer-watched pages, per 1 MiB window
    /// (allocation-free): count and the guest instruction of the last one.
    pub fn note_write_fault(addr: u64) {
        let window = (addr & !0xF_FFFF) | 1;
        let mut i = ((addr >> 20).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as usize % SLOTS;
        for _ in 0..SLOTS {
            match WRITE_FAULT_WINDOW[i].compare_exchange(0, window, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {}
                Err(k) if k == window => {}
                Err(_) => {
                    i = (i + 1) % SLOTS;
                    continue;
                }
            }
            WRITE_FAULT_COUNT[i].fetch_add(1, Ordering::Relaxed);
            WRITE_FAULT_LAST[i].store(ps5vm::fault_site::get(), Ordering::Relaxed);
            return;
        }
    }
    fn write_fault_line() -> String {
        let mut v: Vec<(u64, u64, u64)> = (0..SLOTS)
            .filter_map(|i| {
                let w = WRITE_FAULT_WINDOW[i].load(Ordering::Relaxed);
                (w != 0).then(|| (WRITE_FAULT_COUNT[i].load(Ordering::Relaxed), w & !1, WRITE_FAULT_LAST[i].load(Ordering::Relaxed)))
            })
            .collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        v.iter().take(8).map(|(n, w, rip)| format!("{w:#x}:{n}@rip{rip:#x}")).collect::<Vec<_>>().join(" ")
    }
    static IMAGE_WRITE_WINDOW: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    static IMAGE_WRITE_COUNT: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    /// A CPU write fault on image pages (allocation-free).
    pub fn note_image_write(addr: u64) {
        let window = (addr & !0xFFFF) | 1;
        let mut i = ((addr >> 16).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as usize % SLOTS;
        for _ in 0..SLOTS {
            match IMAGE_WRITE_WINDOW[i].compare_exchange(0, window, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {}
                Err(k) if k == window => {}
                Err(_) => {
                    i = (i + 1) % SLOTS;
                    continue;
                }
            }
            IMAGE_WRITE_COUNT[i].fetch_add(1, Ordering::Relaxed);
            return;
        }
    }
    pub fn line() -> String {
        if !on() {
            return String::new();
        }
        flush_local();
        let mut iw: Vec<(u64, u64)> = (0..SLOTS)
            .filter_map(|i| {
                let w = IMAGE_WRITE_WINDOW[i].load(Ordering::Relaxed);
                (w != 0).then(|| (IMAGE_WRITE_COUNT[i].load(Ordering::Relaxed), w & !1))
            })
            .collect();
        iw.sort_unstable_by(|a, b| b.cmp(a));
        let iw: Vec<String> = iw.iter().take(6).map(|(n, w)| format!("{w:#x}:{n}")).collect();
        let detailed = detailed();
        let parts: Vec<String> = NAMES
            .iter()
            .zip(NANOS.iter())
            .enumerate()
            .filter(|&(i, _)| i < BASE_SPANS || detailed)
            .map(|(_, (n, v))| format!("{n}={}", v.load(Ordering::Relaxed) / 1_000_000))
            .collect();
        let mut faults: Vec<(u64, u64, u64, u64, u64)> = (0..SLOTS)
            .filter_map(|i| {
                let w = FAULT_WINDOW[i].load(Ordering::Relaxed);
                (w != 0).then(|| (FAULT_NANOS[i].load(Ordering::Relaxed), FAULT_COUNT[i].load(Ordering::Relaxed), FAULT_LAST[i].load(Ordering::Relaxed), w & !1, FAULT_RIP[i].load(Ordering::Relaxed)))
            })
            .collect();
        faults.sort_unstable_by(|a, b| b.cmp(a));
        let top: Vec<String> = faults.iter().take(6).map(|(ns, n, a, w, rip)| format!("{a:#x}:{n}/{}ms{}@rip{rip:#x}{}", ns / 1_000_000, writer_of(*w), host_symbol(*rip))).collect();
        format!(
            " | ms: {} | fault windows: {} | CPU image writes (64 KiB): {} | write faults (1 MiB): {} | sync downloads: {} | full barriers: {} | gpu waits: {} | lock holds: {}{}",
            parts.join(" "),
            top.join(" "),
            iw.join(" "),
            write_fault_line(),
            download_line(),
            barrier_line(),
            wait_line(),
            lock_line(),
            blocked_line()
        )
    }
    /// `RPS5E_DESC_WAIT_WHY=1` (with the profile): the GPU-written words
    /// materialisation waited for, per program and read, with what uses
    /// them (`ps5shader::resources::BLOCKED_READS`).
    pub fn desc_wait_why() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            let on = std::env::var("RPS5E_DESC_WAIT_WHY").is_ok_and(|v| v == "1");
            ps5shader::resources::WHY_BLOCKED.store(on, Ordering::Relaxed);
            on
        })
    }
    fn blocked_line() -> String {
        let st = &ps5shader::resources::RECORDS_STATS;
        let records = if crate::pipeline::frontend::defer_records() {
            let last = *st.last_over.lock().unwrap_or_else(|e| e.into_inner());
            let largest = *st.largest.lock().unwrap_or_else(|e| e.into_inner());
            let mut programs = st.programs.lock().unwrap_or_else(|e| e.into_inner()).clone();
            programs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            let programs: Vec<String> = programs.iter().take(12).map(|(p, n, o)| format!("{p:#x}:{n}/{o}")).collect();
            format!(
                " | deferred records: {} checked {} over {} last-over {:x?} largest {:x?} programs {}",
                st.deferred.load(Ordering::Relaxed),
                st.checked.load(Ordering::Relaxed),
                st.over.load(Ordering::Relaxed),
                last,
                largest,
                programs.join(" ")
            )
        } else {
            String::new()
        };
        if !desc_wait_why() {
            return records;
        }
        let g = ps5shader::resources::BLOCKED_READS.lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<_> = g.iter().collect();
        v.sort_unstable_by(|a, b| b.2.cmp(&a.2));
        let parts: Vec<String> = v.iter().take(64).map(|((p, op), uses, n, a)| format!("p{p:#x} op{op} n{n} @{a:#x} {uses}")).collect();
        format!("{records} | blocked reads: {}", parts.join("; "))
    }
}

/// Ask the processor to bring the bytes of `v` into its caches. A draw's
/// resolved state arrives from another thread (live) or from a recording
/// far larger than the caches (replay) and is then read field by field
/// all over the recording of the draw, every first touch of a line a
/// serial cache miss; prefetched together, the misses overlap.
#[inline]
fn prefetch_value<T>(v: &T) {
    #[cfg(target_arch = "x86_64")]
    {
        let p = v as *const T as *const i8;
        let mut off = 0;
        while off < std::mem::size_of::<T>() {
            // SAFETY: a prefetch is only a hint; it never faults and has no
            // visible effect (the address lies inside `v` anyway).
            unsafe { std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p.wrapping_add(off)) };
            off += 64;
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = v;
}

/// The key `f` computes for `t`, from the recent ones in `memo` (most
/// recent first, at most eight kept).
fn recent_key<T: PartialEq + Copy>(memo: &mut Vec<(T, crate::images::ImageKey)>, t: &T, f: fn(&T) -> crate::images::ImageKey) -> crate::images::ImageKey {
    if let Some(i) = memo.iter().position(|(m, _)| m == t) {
        if i != 0 {
            let e = memo.remove(i);
            memo.insert(0, e);
        }
        return memo[0].1;
    }
    let key = f(t);
    memo.truncate(7);
    memo.insert(0, (*t, key));
    key
}

/// The [`RenderExecutor`] the PM4 interpreter uses.
pub struct GpuExecutor;

/// Device memory in use at the last memory-pressure collection (MiB).
static GC_USAGE_MIB: AtomicU64 = AtomicU64::new(0);

static DRAWS_WITHOUT_BACKEND: AtomicU64 = AtomicU64::new(0);

/// Whether a draw or dispatch of the executor may be given up for a
/// pending compile and tried again: not while a capture or the slice
/// shadow logs every try.
fn unlocked_waits_allowed() -> bool {
    !crate::capture::active() && !shadow::enabled()
}

/// Give a completion callback to the back-end (`queue`); it must run exactly
/// once: without a back-end it runs now.
fn hand_over(done: Box<dyn FnOnce() + Send>, queue: fn(&mut Backend, Box<dyn FnOnce() + Send>)) {
    let slot = Arc::new(Mutex::new(Some(done)));
    let s = slot.clone();
    let queued = profile::time(2, || {
        with_backend(move |b| {
            if let Some(f) = s.lock().unwrap_or_else(|e| e.into_inner()).take() {
                queue(b, f);
            }
        })
    });
    if queued.is_none() {
        if let Some(f) = slot.lock().unwrap_or_else(|e| e.into_inner()).take() {
            f();
        }
    }
}

impl RenderExecutor for GpuExecutor {
    fn draw(&self, draw: &ResolvedDraw) {
        let rt = if shadow::enabled() { shadow::target_key(draw) } else { 0 };
        shadow::item(shadow::Kind::Draw, rt, || {
            let measured = ab::enabled().then(|| (ab::mode_fast(), profile::thread_cpu()));
            profile::set_work(profile::WORK_DRAW, if draw.shaders.ps.address != 0 { draw.shaders.ps.address } else { draw.shaders.vertex_program_address() });
            // A draw waiting for a compile does so with the back-end
            // unlocked (CPU faults on GPU-written pages need it meanwhile).
            let program = if draw.shaders.ps.address != 0 { draw.shaders.ps.address } else { draw.shaders.vertex_program_address() };
            let ran = crate::pipeline::unlocked_wait::record(unlocked_waits_allowed(), || {
                let r = crate::perf::items::measure(crate::perf::items::DRAW, program, || profile::time_cpu(|| profile::time(0, || with_backend(|b| (b.draw(draw), b.take_compile_pending()).1))));
                (r.is_some(), r == Some(true))
            });
            if !ran {
                DRAWS_WITHOUT_BACKEND.fetch_add(1, Ordering::Relaxed);
            }
            if let Some((fast, t)) = measured {
                ab::note_draw(fast, profile::thread_cpu() - t);
            }
        })
    }

    fn dispatch(&self, dispatch: &ResolvedDispatch) {
        shadow::item(shadow::Kind::Dispatch, 0, || {
            profile::set_work(profile::WORK_DISPATCH, dispatch.shader.address);
            crate::pipeline::unlocked_wait::record(unlocked_waits_allowed(), || {
                let r = crate::perf::items::measure(crate::perf::items::DISPATCH, dispatch.shader.address, || profile::time(1, || with_backend(|b| (b.dispatch(dispatch), b.take_compile_pending()).1)));
                ((), r == Some(true))
            });
        })
    }

    fn after_pipe(&self, done: Box<dyn FnOnce() + Send>) {
        shadow::item(shadow::Kind::AfterPipe, 0, || hand_over(done, Backend::after_pipe))
    }

    fn end_of_pipe(&self, done: Box<dyn FnOnce() + Send>) {
        shadow::item(shadow::Kind::EndOfPipe, 0, || hand_over(done, Backend::end_of_pipe))
    }

    fn writes_through_pending(&self) -> bool {
        with_ready_backend(|b| b.writes_through_pending()).unwrap_or(false)
    }

    fn write_ordered(&self, addr: u64, data: &[u8], write: Box<dyn FnOnce() + Send>) {
        ordered_cp_write(addr, data, write);
    }

    fn memory_filled(&self, addr: u64, size: u64, value: u32) {
        shadow::item(shadow::Kind::MemoryFilled, 0, || profile::time(6, || with_backend(|b| b.memory_filled(addr, size, value))));
    }

    fn copy_memory(&self, dst: u64, src: u64, size: u64) -> bool {
        shadow::item(shadow::Kind::CpCopy, 0, || {
            profile::set_work(profile::WORK_COPY, 0);
            with_ready_backend(|b| b.copy_memory(dst, src, size)).unwrap_or(false)
        })
    }

    fn fill_memory(&self, dst: u64, size: u64, value: u32) -> bool {
        shadow::item(shadow::Kind::CpFill, 0, || {
            profile::set_work(profile::WORK_FILL, 0);
            with_ready_backend(|b| b.fill_memory(dst, size, value)).unwrap_or(false)
        })
    }

    fn write_memory(&self, dst: u64, data: &[u8]) -> bool {
        shadow::item(shadow::Kind::CpWrite, 0, || {
            profile::set_work(profile::WORK_WRITE, 0);
            with_ready_backend(|b| b.write_memory(dst, data)).unwrap_or(false)
        })
    }

    fn gds_transfer(&self, op: GdsTransfer) -> bool {
        shadow::item(shadow::Kind::Gds, 0, || {
            profile::set_work(profile::WORK_GDS, 0);
            with_backend(|b| b.gds_transfer(op)).unwrap_or(false)
        })
    }

    fn occlusion_counter_dump(&self, addr: u64, write: &crate::render::OcclusionWrite) -> bool {
        shadow::item(shadow::Kind::Occlusion, 0, || crate::scheduler::occlusion::enabled() && with_backend(|b| b.sched.occlusion_counter_dump(addr, write)).unwrap_or(false))
    }

    fn fill_memory_or(&self, dst: u64, size: u64, value: u32, cpu: Box<dyn FnOnce() -> bool + Send>) -> bool {
        let ok = self.fill_memory(dst, size, value) || cpu();
        if ok {
            self.memory_filled(dst, size, value);
        }
        ok
    }

    fn flush(&self) {
        shadow::item(shadow::Kind::Flush, 0, || profile::time(3, || with_ready_backend(|b| b.flush())));
    }

    fn submission_complete(&self) {
        shadow::item(shadow::Kind::Completion, 0, || profile::time(3, || with_ready_backend(|b| b.pressure_gc())));
    }
}

/// The process-wide executor (None when the back-end is disabled with
/// `RPS5E_GPU_BACKEND=0` or `RPS5E_GPU=0`).
pub fn executor() -> Option<Arc<dyn RenderExecutor>> {
    static EXEC: OnceLock<Option<Arc<dyn RenderExecutor>>> = OnceLock::new();
    EXEC.get_or_init(|| {
        let disabled = std::env::var("RPS5E_GPU").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off"));
        (enabled() && !disabled).then(|| {
            if crate::slices::configured() > 0 {
                crate::slices::SliceExecutor::start_global() as Arc<dyn RenderExecutor>
            } else if queued_recording() {
                Arc::new(QueuedExecutor::start()) as Arc<dyn RenderExecutor>
            } else {
                Arc::new(GpuExecutor) as Arc<dyn RenderExecutor>
            }
        })
    })
    .clone()
}

/// The interpreter thread's pre-materialised stages of the draw or
/// dispatch the recording thread is about to prepare:
/// set by the queued executor's consumer right before it calls the
/// provider, taken by the prepare path. Every queued draw sets the slot,
/// so a value never outlives its draw.
pub(crate) mod draw_prefetch {
    use std::cell::RefCell;

    thread_local! {
        static DRAW: RefCell<Option<Box<crate::pipeline::frontend::PrefetchedDraw>>> = const { RefCell::new(None) };
        static DISPATCH: RefCell<Option<Box<crate::pipeline::frontend::PrefetchedDispatch>>> = const { RefCell::new(None) };
    }

    pub(crate) fn set(p: Option<Box<crate::pipeline::frontend::PrefetchedDraw>>) {
        DRAW.with(|s| *s.borrow_mut() = p);
    }

    pub(crate) fn take() -> Box<crate::pipeline::frontend::PrefetchedDraw> {
        // A default (no stages) instead of Option: the prepare path checks
        // the slots anyway.
        DRAW.with(|s| s.borrow_mut().take().unwrap_or_default())
    }

    pub(crate) fn set_dispatch(p: Option<Box<crate::pipeline::frontend::PrefetchedDispatch>>) {
        DISPATCH.with(|s| *s.borrow_mut() = p);
    }

    pub(crate) fn take_dispatch() -> Box<crate::pipeline::frontend::PrefetchedDispatch> {
        DISPATCH.with(|s| s.borrow_mut().take().unwrap_or_default())
    }
}

/// `RPS5E_GPU_RECORD_THREAD`: the PM4 interpreter hands its draws to
/// the recording thread (`gpu-record`, on the recorder's reserved core)
/// instead of recording them itself, and pre-materialises their stages
/// meanwhile (a pipeline of two threads). `auto` (default): while the
/// recorder stays busy ([`pipeline_policy`]); `1`: always; `0`: never.
/// It pays where recording is CPU-bound (SILENT HILL's heavy room
/// +10-20 %); where the frame is bound by synchronisations with the GPU,
/// every round trip crosses two threads and one thread does better
/// (SILENT HILL's light rooms -13 % with `1`).
fn queued_recording() -> bool {
    record_thread_mode() != RecordThread::Off
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RecordThread {
    Off,
    On,
    /// On while it pays (see [`pipeline_policy`]).
    Auto,
}

fn record_thread_mode() -> RecordThread {
    static MODE: OnceLock<RecordThread> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("RPS5E_GPU_RECORD_THREAD").as_deref() {
        Ok("1") | Ok("on") => RecordThread::On,
        Ok("0") | Ok("off") => RecordThread::Off,
        _ if ab::safe() => RecordThread::Off,
        _ => RecordThread::Auto,
    })
}

/// Draws and dispatches go to the recording thread now
/// (`RPS5E_GPU_RECORD_THREAD=1`, not in an A/B slow period, and with
/// `auto` while the policy keeps it on): the PM4 interpreter then leaves
/// the recorder's core to it.
pub fn records_on_own_thread() -> bool {
    queued_recording() && crate::slices::configured() == 0 && pipeline_route()
}

/// Commands go through the recording thread now (see
/// [`records_on_own_thread`]; the executor exists).
fn pipeline_route() -> bool {
    ab::thread_fast()
        && (record_thread_mode() != RecordThread::Auto || {
            pipeline_policy::tick();
            pipeline_policy::on()
        })
}

/// `RPS5E_GPU_RECORD_THREAD=auto`: the recording thread is used while the
/// recorder is the busy end of the pipeline. Signal: the share of time the
/// recorder idles on an empty queue (`PIPELINE_US[0]`). A pipeline whose
/// recorder idles is bound by the latency of its synchronisations (the
/// interpreter waits for labels the recorded work writes, and every such
/// round trip now crosses two threads), and records slower than one
/// thread doing both (SILENT HILL's light rooms: recorder idle 31-37 %,
/// -13 % fps); a CPU-bound one keeps its recorder busy (SILENT HILL's
/// heavy room 0-3 % idle, +10-20 % fps; GTA V's city).
///
/// On: a 1 s window with the recorder idle more than [`IDLE_LIMIT`]
/// twice in a row turns it off for a back-off (15 s, doubling to 120 s
/// while probes fail). Off: at the end of the back-off a 2 s probe runs
/// with it on; a probe under the limit turns it on.
mod pipeline_policy {
    use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};
    use std::sync::Mutex;

    const IDLE_LIMIT: f64 = 0.20;
    const WINDOW_MS: u64 = 1000;
    const PROBE_MS: u64 = 2000;
    const BACKOFF_MS: u64 = 15_000;
    const BACKOFF_MAX_MS: u64 = 120_000;

    const ON: u8 = 0;
    const OFF: u8 = 1;
    const PROBE: u8 = 2;
    /// The first moments of a probe: the recorder's wait that spans the
    /// off period ends now and is not counted.
    const WARMUP: u8 = 3;
    const WARMUP_MS: u64 = 250;

    static STATE: AtomicU8 = AtomicU8::new(ON);
    /// Milliseconds since the start at which the policy looks again.
    static NEXT_MS: AtomicU64 = AtomicU64::new(WINDOW_MS);
    /// Transitions (diagnostics: on, off).
    pub static SWITCHES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

    struct Window {
        start_ms: u64,
        idle_us: u64,
        strikes: u8,
        backoff_ms: u64,
    }
    static WINDOW: Mutex<Window> = Mutex::new(Window { start_ms: 0, idle_us: 0, strikes: 0, backoff_ms: BACKOFF_MS });

    fn now_ms() -> u64 {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
    }

    pub fn on() -> bool {
        STATE.load(Relaxed) != OFF
    }

    /// The next state after a window in `state` whose recorder idled
    /// `share` of the time, with the strike count and back-off carried.
    pub(super) fn decide(state: u8, share: f64, strikes: u8, backoff_ms: u64) -> (u8, u8, u64) {
        match state {
            ON if share > IDLE_LIMIT && strikes + 1 >= 2 => (OFF, 0, backoff_ms),
            ON if share > IDLE_LIMIT => (ON, strikes + 1, backoff_ms),
            ON => (ON, 0, BACKOFF_MS),
            PROBE if share > IDLE_LIMIT => (OFF, 0, (backoff_ms * 2).min(BACKOFF_MAX_MS)),
            PROBE => (ON, 0, BACKOFF_MS),
            WARMUP => (PROBE, strikes, backoff_ms),
            _ => (WARMUP, strikes, backoff_ms),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// An idle recorder turns the thread off after two windows, a busy
        /// one keeps it; probes come after a warm-up, a failed probe
        /// doubles the back-off up to its cap, a passed one resets it.
        #[test]
        fn the_policy_follows_the_recorders_idle_share() {
            assert_eq!(decide(ON, 0.05, 0, BACKOFF_MS), (ON, 0, BACKOFF_MS));
            assert_eq!(decide(ON, 0.35, 0, BACKOFF_MS), (ON, 1, BACKOFF_MS));
            assert_eq!(decide(ON, 0.05, 1, BACKOFF_MS), (ON, 0, BACKOFF_MS), "one idle window alone changes nothing");
            assert_eq!(decide(ON, 0.35, 1, BACKOFF_MS), (OFF, 0, BACKOFF_MS));
            assert_eq!(decide(OFF, 1.0, 0, BACKOFF_MS), (WARMUP, 0, BACKOFF_MS));
            assert_eq!(decide(WARMUP, 5.0, 0, BACKOFF_MS), (PROBE, 0, BACKOFF_MS), "the warm-up is not judged");
            assert_eq!(decide(PROBE, 0.35, 0, BACKOFF_MS), (OFF, 0, 2 * BACKOFF_MS));
            assert_eq!(decide(PROBE, 0.35, 0, BACKOFF_MAX_MS), (OFF, 0, BACKOFF_MAX_MS));
            assert_eq!(decide(PROBE, 0.10, 0, 4 * BACKOFF_MS), (ON, 0, BACKOFF_MS));
        }
    }

    /// Called by the interpreter for every command it hands over or keeps:
    /// cheap unless a window ended.
    pub fn tick() {
        let now = now_ms();
        if now < NEXT_MS.load(Relaxed) {
            return;
        }
        let Ok(mut w) = WINDOW.try_lock() else { return };
        let idle = super::PIPELINE_US[0].load(Relaxed);
        let span_ms = now.saturating_sub(w.start_ms).max(1);
        let share = idle.saturating_sub(w.idle_us) as f64 / (span_ms * 1000) as f64;
        let state = STATE.load(Relaxed);
        let (next, strikes, backoff) = decide(state, share, w.strikes, w.backoff_ms);
        w.strikes = strikes;
        w.backoff_ms = backoff;
        if next != state && next != PROBE {
            log::info!(
                "ps5gpu: recording thread {} (recorder idle {:.0} % of the last {} ms){}",
                match next {
                    ON => "on",
                    OFF => "off",
                    _ => "probed",
                },
                share * 100.0,
                span_ms,
                if next == OFF { format!(", next probe in {} s", w.backoff_ms / 1000) } else { String::new() }
            );
            SWITCHES[(next == OFF) as usize].fetch_add(1, Relaxed);
        }
        STATE.store(next, Relaxed);
        w.start_ms = now;
        w.idle_us = super::PIPELINE_US[0].load(Relaxed);
        NEXT_MS.store(
            now + match next {
                OFF => w.backoff_ms,
                PROBE => PROBE_MS,
                WARMUP => WARMUP_MS,
                _ => WINDOW_MS,
            },
            Relaxed,
        );
    }
}

/// One executor call handed to the recording thread.
enum Recorded {
    Draw(Box<ResolvedDraw>, Option<Box<crate::pipeline::frontend::PrefetchedDraw>>),
    Dispatch(Box<ResolvedDispatch>, Option<Box<crate::pipeline::frontend::PrefetchedDispatch>>),
    EndOfPipe(Box<dyn FnOnce() + Send>),
    AfterPipe(Box<dyn FnOnce() + Send>),
    MemoryFilled(u64, u64, u32),
    Flush,
    SubmissionComplete,
    SyncPoint,
    /// Run on the recording thread (calls whose result the interpreter
    /// waits for, drains).
    Call(Box<dyn FnOnce() + Send>),
}

/// Calls queued at most before the interpreter waits for the recorder.
/// Measured both ways: a small queue keeps the interpreter's
/// stage prefetch fresh (its page validation accepts ~100 % of the stages
/// instead of ~54 % at 8192), but with the old sleep-polling `send` the
/// interpreter burned up to 50 µs per full queue and the heavy SH phase fell
/// to 3.2 fps. The blocking `send` (§53) removed that cost, so the queue is
/// small again: the recorder lags at most ~2 ms (`RPS5E_RECORD_QUEUE`).
fn record_queue() -> usize {
    static ON: OnceLock<usize> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_RECORD_QUEUE").ok().and_then(|v| v.parse().ok()).filter(|&n| n >= 8).unwrap_or(128))
}

/// The executor of the PM4 interpreters when recording runs on its own
/// thread (`gpu-record`): the interpreter decodes packets and resolves
/// register state into draws, the recording thread binds and records them
/// in the same order. Every executor call goes through one FIFO, so
/// completion callbacks, flushes and CPU synchronisation points keep their
/// place between the draws; calls with a result, and drains (flips, which
/// capture what the queued draws render), wait for the recorder to reach
/// them.
pub struct QueuedExecutor {
    tx: std::sync::mpsc::SyncSender<Recorded>,
    transfers: Arc<crate::slices::pending::PendingEvents>,
    /// Calls queued and not yet executed.
    pending: Arc<AtomicU64>,
    /// Signalled by the recorder after a command it takes while a sender
    /// waits: that sender wakes the moment a slot frees.
    space: Arc<QueueSpace>,
}

/// The wake-up of an interpreter waiting for room in the recording queue.
/// The recorder signals only while a sender has announced itself in
/// `waiting` (a condition variable's notify is a system call even
/// without waiters, which was one per recorded command). The sender
/// announces itself before it tries the queue again and the recorder
/// looks after it took a command, each behind a sequentially consistent
/// fence: either the sender's retry sees the free slot or the recorder
/// sees the sender (`RPS5E_QUEUE_SIGNAL_ALWAYS=1`: signal after every
/// command, as before).
/// Bytes of an occlusion query's result block from its dump address on.
const OCCLUSION_BLOCK_BYTES: u64 = 256;

/// `RPS5E_ASYNC_OCCLUSION=0`: with the recording thread, the interpreter
/// waits at every occlusion counter dump until the recorder has recorded
/// everything before it (to learn whether the GPU counts it). By default
/// the dump is queued like a draw and its conservative CPU result, when
/// the GPU does not count it, is written in order by the recorder (A/B
/// scope release `async-occlusion`). SILENT HILL dumps ~5 000 times a
/// second: every dump drained the queue and left the recorder idle a
/// third of the time in light rooms.
fn async_occlusion_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| ab::switch_on("RPS5E_ASYNC_OCCLUSION")) && ab::release_fast("async-occlusion")
}

/// The recording pipeline's stalls (`RPS5E_GPU_RECORD_THREAD=1`), in
/// microseconds: the recorder idle on an empty queue, the interpreter
/// waiting for room, the interpreter waiting for a call's result (the
/// queue drained first); then the number of such calls.
pub static PIPELINE_US: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// With `RPS5E_AB_PERIOD`: the recorder idle on an empty queue, the
/// interpreter waiting for room, the recorder waiting for the back-end
/// lock and for the GPU, in microseconds per A/B mode (fast, then slow).
pub static PIPELINE_AB_US: [[AtomicU64; 4]; 2] = [const { [const { AtomicU64::new(0) }; 4] }; 2];

pub(crate) fn note_pipeline_ab(kind: usize, us: u64) {
    if ab::enabled() {
        PIPELINE_AB_US[!ab::mode_fast() as usize][kind].fetch_add(us, Ordering::Relaxed);
    }
}

/// Calls that waited for the recorder, by kind: occlusion dump, copy,
/// fill, write, GDS, drain (flips, reads), memory hazard, ordered read.
pub static PIPELINE_CALLS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

fn count_call(kind: usize) {
    PIPELINE_CALLS[kind].fetch_add(1, Ordering::Relaxed);
}

/// `pipeline rec-idle .. full .. calls ../..` for the `[gpu-sys]` line.
pub fn pipeline_stalls() -> String {
    if !queued_recording() {
        return String::new();
    }
    let g = |i: usize| PIPELINE_US[i].load(Ordering::Relaxed);
    let k: Vec<u64> = PIPELINE_CALLS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let sw = |i: usize| pipeline_policy::SWITCHES[i].load(Ordering::Relaxed);
    let ab = |m: usize, i: usize| PIPELINE_AB_US[m][i].load(Ordering::Relaxed) / 1000;
    format!(
        " pipeline rec-idle {} ms full {} ms calls {} ms/{} by kind {:?} switched on/off {}/{} ab_rec_idle_ms=[{}, {}] ab_full_ms=[{}, {}] ab_rec_lock_ms=[{}, {}] ab_rec_gpu_ms=[{}, {}]",
        g(0) / 1000,
        g(1) / 1000,
        g(2) / 1000,
        g(3),
        k,
        sw(0),
        sw(1),
        ab(0, 0),
        ab(1, 0),
        ab(0, 1),
        ab(1, 1),
        ab(0, 2),
        ab(1, 2),
        ab(0, 3),
        ab(1, 3)
    )
}

/// Waits for queue room that ended by the timeout instead of a signal
/// (a lost wake-up would show here; tests).
static QUEUE_SPACE_TIMEOUTS: AtomicU64 = AtomicU64::new(0);

struct QueueSpace {
    lock: std::sync::Mutex<()>,
    woke: std::sync::Condvar,
    waiting: std::sync::atomic::AtomicUsize,
}

impl QueueSpace {
    fn new() -> QueueSpace {
        QueueSpace { lock: std::sync::Mutex::new(()), woke: std::sync::Condvar::new(), waiting: std::sync::atomic::AtomicUsize::new(0) }
    }

    /// Queue `item`, waiting for room when the queue is full.
    fn send<T>(&self, tx: &std::sync::mpsc::SyncSender<T>, item: T) {
        let mut item = match tx.try_send(item) {
            Ok(()) => return,
            Err(std::sync::mpsc::TrySendError::Full(c)) => c,
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => return,
        };
        let t = std::time::Instant::now();
        let mut guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        // Announced before the retry.
        self.waiting.fetch_add(1, Ordering::SeqCst);
        std::sync::atomic::fence(Ordering::SeqCst);
        loop {
            match tx.try_send(item) {
                Ok(()) => break,
                Err(std::sync::mpsc::TrySendError::Full(c)) => {
                    if ps5vm::shutdown::requested() {
                        break;
                    }
                    item = c;
                    let (g, timeout) = self.woke.wait_timeout(guard, std::time::Duration::from_millis(100)).unwrap_or_else(|e| e.into_inner());
                    if timeout.timed_out() {
                        QUEUE_SPACE_TIMEOUTS.fetch_add(1, Ordering::Relaxed);
                    }
                    guard = g;
                }
                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => break,
            }
        }
        self.waiting.fetch_sub(1, Ordering::SeqCst);
        let us = t.elapsed().as_micros() as u64;
        PIPELINE_US[1].fetch_add(us, Ordering::Relaxed);
        note_pipeline_ab(1, us);
    }

    /// The recorder took a command (its slot is free).
    fn freed(&self) {
        static ALWAYS: OnceLock<bool> = OnceLock::new();
        let always = *ALWAYS.get_or_init(|| std::env::var("RPS5E_QUEUE_SIGNAL_ALWAYS").is_ok_and(|v| v == "1"));
        std::sync::atomic::fence(Ordering::SeqCst);
        if always || self.waiting.load(Ordering::SeqCst) != 0 {
            // Under the lock: a sender between its failed retry and its
            // wait holds it, so the signal cannot fall in between.
            let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            self.woke.notify_all();
        }
    }
}

impl QueuedExecutor {
    fn start() -> QueuedExecutor {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Recorded>(record_queue());
        let pending = Arc::new(AtomicU64::new(0));
        let p = pending.clone();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let s = stop.clone();
        let wake = tx.clone();
        let space = Arc::new(QueueSpace::new());
        let freed = space.clone();
        std::thread::Builder::new()
            .name("gpu-record".into())
            .spawn(move || {
                // The recorder is the busy end of the pipeline: it takes the
                // core kept for the recorder (the interpreter moves off it
                // while commands go through this thread, see
                // `records_on_own_thread`).
                ps5vm::topology::pin_current_thread_to_recorder_core("gpu-record");
                crate::perf::sys::mark_recorder_thread();
                let ex = GpuExecutor;
                let mut taken = 0u32;
                loop {
                    // The recorder idles on an empty queue (counted: the
                    // pipeline's balance, see `pipeline_stalls`).
                    let cmd = match rx.try_recv() {
                        Ok(c) => c,
                        Err(std::sync::mpsc::TryRecvError::Empty) => {
                            let t = std::time::Instant::now();
                            let r = rx.recv();
                            let us = t.elapsed().as_micros() as u64;
                            PIPELINE_US[0].fetch_add(us, Ordering::Relaxed);
                            note_pipeline_ab(0, us);
                            match r {
                                Ok(c) => c,
                                Err(_) => break,
                            }
                        }
                        Err(_) => break,
                    };
                    taken = taken.wrapping_add(1);
                    if taken % 32 == 0 {
                        crate::perf::sys::account_ab_mode_recorder();
                    }
                    if s.load(Ordering::Acquire) {
                        break;
                    }
                    // A sender waiting for queue space wakes now: its slot
                    // is free the moment the command is taken.
                    freed.freed();
                    match cmd {
                        Recorded::Draw(d, pre) => {
                            draw_prefetch::set(pre);
                            ex.draw(&d);
                        }
                        Recorded::Dispatch(d, pre) => {
                            draw_prefetch::set_dispatch(pre);
                            ex.dispatch(&d);
                        }
                        Recorded::EndOfPipe(f) => ex.end_of_pipe(f),
                        Recorded::AfterPipe(f) => ex.after_pipe(f),
                        Recorded::MemoryFilled(a, n, v) => ex.memory_filled(a, n, v),
                        Recorded::Flush => ex.flush(),
                        Recorded::SubmissionComplete => ex.submission_complete(),
                        Recorded::SyncPoint => crate::memory::note_sync_point(),
                        Recorded::Call(f) => f(),
                    }
                    p.fetch_sub(1, Ordering::AcqRel);
                }
            })
            .expect("spawn the GPU recording thread");
        ps5vm::shutdown::on_shutdown(ps5vm::shutdown::ORDER_GPU, "GPU recording thread", move || {
            stop.store(true, Ordering::Release);
            let _ = wake.try_send(Recorded::Flush);
        });
        QueuedExecutor { tx, pending, transfers: Arc::default(), space }
    }

    /// The back-end records (before it exists nothing is queued: calls
    /// have no GPU side and run at once, in order).
    fn recording(&self) -> bool {
        if !pipeline_route() {
            // A/B slow mode: calls run on the interpreter, after the queue.
            self.drain();
            return false;
        }
        self.pending.load(Ordering::Acquire) != 0 || CELL.state.load(Ordering::Acquire) == READY
    }

    fn send(&self, cmd: Recorded) {
        self.pending.fetch_add(1, Ordering::AcqRel);
        // Waiting for space sleeps on the recorder's per-command signal:
        // the interpreter wakes the instant a slot frees. (A `try_send` +
        // 50 microsecond sleep poll once cost the pipeline up to that per
        // full queue - at a small queue that alone halved the frame rate,
        // GPU_PLAN §53.) The 100 ms timeout only checks for a shutdown;
        // the recorder drains the queue before it stops.
        self.space.send(&self.tx, cmd);
    }

    /// Run `f` on the recording thread after every call queued before it
    /// and wait for its result.
    fn call<R: Send + 'static>(&self, f: impl FnOnce() -> R + Send + 'static) -> Option<R> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        self.send(Recorded::Call(Box::new(move || {
            let _ = tx.send(f());
        })));
        let t = std::time::Instant::now();
        PIPELINE_US[3].fetch_add(1, Ordering::Relaxed);
        loop {
            match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(r) => {
                    PIPELINE_US[2].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                    return Some(r);
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) if !ps5vm::shutdown::requested() => {}
                Err(_) => return None,
            }
        }
    }
}

impl RenderExecutor for QueuedExecutor {
    fn draw(&self, draw: &ResolvedDraw) {
        if !pipeline_route() && !self.recording() {
            return GpuExecutor.draw(draw);
        }
        // The interpreter thread pre-materialises the draw's stages while
        // the recorder binds and records the ones before it.
        let pre = crate::pipeline::frontend::prefetch_draw(draw);
        self.send(Recorded::Draw(Box::new(draw.clone()), pre));
    }

    fn dispatch(&self, dispatch: &ResolvedDispatch) {
        if !pipeline_route() && !self.recording() {
            return GpuExecutor.dispatch(dispatch);
        }
        let pre = crate::pipeline::frontend::prefetch_dispatch(dispatch);
        self.send(Recorded::Dispatch(Box::new(dispatch.clone()), pre));
    }

    fn after_pipe(&self, done: Box<dyn FnOnce() + Send>) {
        if !pipeline_route() && !self.recording() {
            return GpuExecutor.after_pipe(done);
        }
        self.send(Recorded::AfterPipe(done));
    }

    fn end_of_pipe(&self, done: Box<dyn FnOnce() + Send>) {
        if !pipeline_route() && !self.recording() {
            return GpuExecutor.end_of_pipe(done);
        }
        self.send(Recorded::EndOfPipe(done));
    }

    fn memory_filled(&self, addr: u64, size: u64, value: u32) {
        if !pipeline_route() && !self.recording() {
            return GpuExecutor.memory_filled(addr, size, value);
        }
        self.send(Recorded::MemoryFilled(addr, size, value));
    }

    fn occlusion_counter_dump(&self, addr: u64, write: &crate::render::OcclusionWrite) -> bool {
        let write = write.clone();
        count_call(0);
        self.call(move || GpuExecutor.occlusion_counter_dump(addr, &write)).unwrap_or(false)
    }

    fn occlusion_counter_dump_or(&self, addr: u64, write: &crate::render::OcclusionWrite, cpu: Box<dyn FnOnce() + Send>) -> bool {
        if !async_occlusion_enabled() || (!pipeline_route() && !self.recording()) {
            return false;
        }
        // Queued: the interpreter goes on. Its reads of the result block
        // wait for the dump (and a conservative result written by `cpu`).
        let write = write.clone();
        let transfers = self.transfers.clone();
        let serial = transfers.add(addr, OCCLUSION_BLOCK_BYTES, None);
        self.send(Recorded::Call(Box::new(move || {
            if !GpuExecutor.occlusion_counter_dump(addr, &write) {
                cpu();
            }
            transfers.remove(serial);
        })));
        true
    }

    fn copy_memory(&self, dst: u64, src: u64, size: u64) -> bool {
        count_call(1);
        self.call(move || GpuExecutor.copy_memory(dst, src, size)).unwrap_or(false)
    }

    fn fill_memory(&self, dst: u64, size: u64, value: u32) -> bool {
        count_call(2);
        self.call(move || GpuExecutor.fill_memory(dst, size, value)).unwrap_or(false)
    }

    fn write_memory(&self, dst: u64, data: &[u8]) -> bool {
        let data = data.to_vec();
        count_call(3);
        self.call(move || GpuExecutor.write_memory(dst, &data)).unwrap_or(false)
    }

    fn gds_transfer(&self, op: GdsTransfer) -> bool {
        count_call(4);
        self.call(move || GpuExecutor.gds_transfer(op)).unwrap_or(false)
    }

    fn flush(&self) {
        if !pipeline_route() && !self.recording() {
            return GpuExecutor.flush();
        }
        self.send(Recorded::Flush);
    }

    fn submission_complete(&self) {
        if !pipeline_route() && !self.recording() {
            return GpuExecutor.submission_complete();
        }
        self.send(Recorded::SubmissionComplete);
    }

    fn sync_point(&self) {
        if !pipeline_route() && !self.recording() {
            return crate::memory::note_sync_point();
        }
        self.send(Recorded::SyncPoint);
    }

    fn drain(&self) {
        if self.pending.load(Ordering::Acquire) != 0 {
            count_call(5);
            self.call(|| ());
        }
    }

    fn run_ordered(&self, f: Box<dyn FnOnce() + Send>) {
        if !self.recording() {
            return f();
        }
        self.send(Recorded::Call(f));
    }

    fn fill_memory_or(&self, dst: u64, size: u64, value: u32, cpu: Box<dyn FnOnce() -> bool + Send>) -> bool {
        if !self.recording() {
            return GpuExecutor.fill_memory_or(dst, size, value, cpu);
        }
        let transfers = self.transfers.clone();
        let serial = transfers.add(dst, size, Some(crate::slices::pending::EventData::Pattern(value)));
        self.send(Recorded::Call(Box::new(move || {
            if GpuExecutor.fill_memory(dst, size, value) || cpu() {
                GpuExecutor.memory_filled(dst, size, value);
            } else {
                crate::log_once(&format!("ps5gpu: command-processor fill of {dst:#x} ({size:#x} bytes) outside guest memory"));
            }
            transfers.remove(serial);
        })));
        true
    }

    fn copy_memory_or(&self, dst: u64, src: u64, size: u64, cpu: Box<dyn FnOnce() -> bool + Send>) -> bool {
        if !self.recording() {
            return GpuExecutor.copy_memory_or(dst, src, size, cpu);
        }
        let transfers = self.transfers.clone();
        let serial = transfers.add(dst, size, None);
        self.send(Recorded::Call(Box::new(move || {
            if !GpuExecutor.copy_memory(dst, src, size) && !cpu() {
                crate::log_once(&format!("ps5gpu: command-processor copy {src:#x} -> {dst:#x} ({size:#x} bytes) outside guest memory"));
            }
            transfers.remove(serial);
        })));
        true
    }

    fn memory_hazard(&self, addr: u64, len: u64) {
        if self.transfers.overlaps(addr, len) {
            count_call(6);
            self.drain();
        }
    }


    fn write_ordered(&self, addr: u64, data: &[u8], write: Box<dyn FnOnce() + Send>) {
        // Always behind the queued work: a label the CPU sees must come
        // after the draws and transfers before it were recorded (their
        // pages GPU-dirty, so a CPU read waits for their results).
        if !self.recording() {
            ordered_cp_write(addr, data, write);
            return;
        }
        let len = data.len() as u64;
        let transfers = self.transfers.clone();
        let serial = transfers.add(addr, len, Some(crate::slices::pending::EventData::Bytes(data.to_vec())));
        let bytes = data.to_vec();
        self.send(Recorded::Call(Box::new(move || {
            ordered_cp_write(addr, &bytes, write);
            transfers.remove(serial);
        })));
    }

    fn read_ordered(&self, addr: u64, out: &mut [u8], read: &dyn Fn(&mut [u8]) -> bool) -> bool {
        if !pipeline_route() {
            self.drain();
            return read(out);
        }
        // Queued transfers over the range, in queue order, taken before
        // memory is read: one that runs meanwhile writes the same bytes.
        match self.transfers.compose(addr, out, read) {
            Some(ok) => ok,
            None => {
                // A queued copy over the range: its bytes come from memory.
                count_call(7);
                self.drain();
                read(out)
            }
        }
    }
}

/// Periodic summary line (diagnostics).
pub fn summary() -> Option<String> {
    // The slice shadow's frames are analysed outside the back-end's lock.
    let mut line = summary_locked()?;
    {
        use crate::memory::layout_memo::{HITS, MISSES};
        line.push_str(&format!(" | layout-memo={}/{}miss", HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed)));
    }
    if let Some(sl) = crate::slices::summary() {
        line = format!("{line} | {sl}");
    }
    Some(match shadow::summary() {
        Some(sh) => format!("{line} | {sh}"),
        None => line,
    })
}

fn summary_locked() -> Option<String> {
    with_ready_backend(|b| {
        let s = b.stats;
        let i = b.images.stats();
        let bf = b.buffers.stats();
        let m = b.memory.stats();
        let mut provider = b.pipelines.summary().map(|p| format!(" | {p}")).unwrap_or_default();
        if profile::on() {
            provider.push_str(&format!(" | top uploads: {} | top invalidations: {}", b.images.upload_top(6), b.images.invalidate_top(6)));
            provider.push_str(&format!(" | stale: {}", b.images.stale_top(8)));
            provider.push_str(&format!(" | retiles: {}", b.images.retile_top(40)));
            provider.push_str(&format!(" | refills by caller: {}", b.images.refill_callers_top(24)));
        }
        format!(
            "draws={} needs_shader={} dispatches={} clears={}c/{}d meta={} dcc-decide={}gpu/{}cpu/{}spent resolves={} eop={} captures={} fills={}+{}p copies={} images={} (up {} [new {} cpu {} gpu {} unw {}] wb {} conv {} retile {} overlay {} known-keys {} imp {} direct {} reuse {} discard {} drop {:?} msdepth {} gc {}/{} at {}/{} MiB superseded {} refills [kept {} own-wb {}] readback {} {}/{} MiB) buffers={} ({} MiB, up {}/{} MiB, untracked {} MiB, {} ms, dn {}/{} MiB {} ms, {} direct, {} through, {} words/{} ms, cp {} fill {} wr {}, settled {}/{} MiB (pressure {}), freed {}/{} MiB, chunk-views {}) vram buf/img {}/{} MiB faults={}/{} [w {} img {} r {} landed {} waited {}] hot {} pages, {}/{} changed, fault-runs {}/{}p of {} heated, premarked {}/{}p, sync points {} ticks={} eager={}/{}hot req={} scopes={} merged={} ends={:?} barriers={}/{}img clean-skips={} memo-reads={}/{} (dirty {} sync {}) written-uses={} streamed={} uploads={:?}/{}tick/{}w prologue={}runs/{}cp/{}touched/{}new arena={}blk/{}MiB/{}live/{}MiB{}",
            s.draws,
            s.needs_shader_draws,
            s.dispatches,
            s.color_clears,
            s.depth_clears,
            s.metadata_ops,
            s.dcc_gpu_decisions,
            s.dcc_cpu_reads,
            s.dcc_store_spent,
            s.resolves,
            s.end_of_pipe,
            s.captures,
            s.fills,
            s.pattern_fills,
            s.copy_kernels,
            i.images,
            i.uploads,
            i.upload_reasons[1],
            i.upload_reasons[2],
            i.upload_reasons[3],
            i.upload_reasons[4],
            i.writebacks,
            i.conversions,
            i.retiles,
            i.alias_overlays,
            i.known_fill_keys,
            i.import_uploads,
            i.direct_copies,
            i.reused,
            i.discards,
            i.drops,
            i.reinterpretations,
            i.pressure_runs,
            i.pressure_drops,
            GC_USAGE_MIB.load(Ordering::Relaxed),
            b.gc.images.trigger >> 20,
            i.superseded_writebacks,
            i.kept_merges,
            i.own_writebacks,
            i.readbacks,
            i.modified_bytes >> 20,
            i.bytes >> 20,
            bf.buffers,
            bf.bytes >> 20,
            bf.uploads,
            bf.uploaded_bytes >> 20,
            bf.untracked_bytes >> 20,
            bf.upload_us / 1000,
            bf.downloads,
            bf.downloaded_bytes >> 20,
            bf.download_us / 1000,
            bf.direct_downloads,
            bf.written_through_waits,
            bf.word_landings,
            bf.word_landing_us / 1000,
            bf.gpu_copies,
            bf.gpu_fills,
            bf.gpu_writes,
            bf.settled,
            bf.settled_bytes >> 20,
            bf.pressure_settles,
            bf.collected,
            bf.collected_bytes >> 20,
            bf.chunk_host_views,
            crate::device::DEVICE_LOCAL_BYTES[0].load(Ordering::Relaxed) >> 20,
            crate::device::DEVICE_LOCAL_BYTES[1].load(Ordering::Relaxed) >> 20,
            m.faults_resolved,
            m.faults_declined,
            m.write_faults,
            m.image_faults,
            m.read_faults,
            crate::memory::region::LANDED_PAGES.load(Ordering::Relaxed),
            LANDING_WAITS.load(Ordering::Relaxed),
            crate::memory::region::HOT_PAGES.load(Ordering::Relaxed),
            crate::memory::region::HOT_CHANGED.load(Ordering::Relaxed),
            crate::memory::region::HOT_COMPARES.load(Ordering::Relaxed),
            crate::memory::region::FAULT_RUNS.load(Ordering::Relaxed),
            crate::memory::region::FAULT_RUN_OPENED.load(Ordering::Relaxed),
            crate::memory::region::HEATED_WRITE_FAULTS.load(Ordering::Relaxed),
            crate::memory::PREMARKED_WRITES.load(Ordering::Relaxed),
            crate::memory::PREMARKED_PAGES.load(Ordering::Relaxed),
            crate::memory::sync_generation(),
            b.sched.timeline.completed_tick(),
            // eager write-throughs (after every writing dispatch) / hot
            // (only the ranges a CPU access waited for before):
            s.eager_throughs,
            s.hot_eager_throughs,
            s.submit_requests,
            b.sched.stats().rendering_scopes,
            b.sched.stats().merged_draws,
            b.sched.stats().scope_ends,
            b.sched.stats().full_barriers,
            b.sched.stats().image_barriers,
            bf.clean_skips,
            bf.memo_reads,
            bf.read_uses,
            bf.memo_dirty_misses,
            bf.memo_sync_misses,
            bf.written_uses,
            bf.streamed,
            bf.upload_sizes,
            bf.upload_used_in_tick,
            bf.upload_written,
            bf.prologue_runs,
            b.sched.stats().prologue_copies,
            bf.prologue_touched,
            bf.prologue_new,
            b.dev.arena.stats().blocks,
            b.dev.arena.stats().block_bytes >> 20,
            b.dev.arena.stats().live,
            b.dev.arena.stats().used_bytes >> 20,
            provider
        ) + &format!(" unranged-syncs={}/{}buf", bf.unranged_syncs, bf.unranged_synced_buffers) + &format!(" watch-index-misses={}", crate::images::WATCH_INDEX_MISSES.load(Ordering::Relaxed))
            + &gpu_time_line(b.sched.stats().gpu, &b.sched.gpu_top(gpu_top_count()))
            + &profile::line()
            + &ab::line()
            + &b.buffers.import_stats().map_or(String::new(), |g| {
                format!(
                    " imports=[{} chunks {} MiB, {} failed, private {}/{} MiB ({} failed), {} ms; {} map events, {} entries; through {}/{} MiB; resident +{} -{} pages]",
                    g.chunks,
                    g.chunk_bytes >> 20,
                    g.failed_chunks,
                    g.private_imports,
                    g.private_bytes >> 20,
                    g.failed_private,
                    g.import_us / 1000,
                    g.map_events,
                    g.entries,
                    g.write_throughs,
                    g.write_through_bytes >> 20,
                    g.resident_pages,
                    g.evicted_pages
                )
            })
    })
}

/// `RPS5E_GPU_TOP=<n>`: programs listed in the profile's top shaders
/// (default 10).
fn gpu_top_count() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::env::var("RPS5E_GPU_TOP").ok().and_then(|v| v.parse().ok()).unwrap_or(10))
}

/// GPU execution time of the submitted command buffers (profile only).
fn gpu_time_line(t: crate::scheduler::gpu_timer::GpuTime, top: &str) -> String {
    if !profile::on() || t.buffers == 0 {
        return String::new();
    }
    format!(
        " | gpu: busy {} ms in {} buffers, longest {:.1} ms, {} of 16 ms or more; top shaders {top}",
        t.busy_ns / 1_000_000,
        t.buffers,
        t.max_ns as f64 / 1e6,
        t.long
    )
}

#[cfg(test)]
mod queue_space_tests {
    use super::*;

    /// A producer filling a two-slot queue faster than its consumer takes
    /// from it is woken by the consumer's signals, never by the timeout,
    /// and every item arrives in order.
    #[test]
    fn full_queue_senders_wake_without_timeouts() {
        let (tx, rx) = std::sync::mpsc::sync_channel::<u32>(2);
        let space = Arc::new(QueueSpace::new());
        let freed = space.clone();
        const N: u32 = 20_000;
        let consumer = std::thread::spawn(move || {
            let mut next = 0;
            while let Ok(v) = rx.recv() {
                freed.freed();
                assert_eq!(v, next);
                next += 1;
                if next % 64 == 0 {
                    // Let the queue fill now and then.
                    std::thread::yield_now();
                }
            }
            next
        });
        let before = QUEUE_SPACE_TIMEOUTS.load(Ordering::Relaxed);
        let t0 = std::time::Instant::now();
        for i in 0..N {
            space.send(&tx, i);
        }
        drop(tx);
        assert_eq!(consumer.join().unwrap(), N);
        assert_eq!(QUEUE_SPACE_TIMEOUTS.load(Ordering::Relaxed), before, "a sender waited for the timeout");
        assert!(t0.elapsed() < std::time::Duration::from_secs(10));
    }
}
