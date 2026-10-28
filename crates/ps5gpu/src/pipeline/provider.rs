// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! [`ShaderPipelines`]: the [`PipelineProvider`] that turns resolved draws
//! and dispatches into recorded Vulkan work.
//!
//! Per draw:
//! 1. the front-end plans the vertex and pixel programs and materialises
//!    their descriptors (user data, guest memory) into a binding layout and
//!    push data;
//! 2. each stage permutation is compiled once by the installed
//!    [`ShaderCompiler`] (cached by permutation key);
//! 3. every guest buffer range the draw touches is reserved in the buffer
//!    cache first (so merges cannot invalidate a binding), then bound:
//!    storage buffers from V#s (aligned down; the adjustment goes into the
//!    push data), images from T#s through the image cache (views with
//!    format, swizzle, aspect and view type), samplers from S#s, the
//!    flattened table / shader data / GDS / device-address page table;
//! 4. index data is bound from the buffer cache, or converted on the CPU
//!    (8-bit indices, quad lists, non-native restart index); rectangle lists
//!    go through generated tessellation stages (or VK_NV_fill_rectangle, or
//!    a strip/list fallback);
//! 5. the graphics pipeline is looked up by (modules, layout, fixed-function
//!    key) or created through the on-disk pipeline cache;
//! 6. the draw is recorded inside a dynamic-rendering scope over the bound
//!    targets, with dynamic state, push descriptors (or a pool set above the
//!    push-descriptor limit) and push constants.
//!
//! Ordering is conservative: a full memory barrier before and after every
//! draw/dispatch (all images stay in GENERAL layout). Finer barriers and
//! render-pass merging across draws are future work.
//!
//! Without a compiler every draw and dispatch reports
//! [`DrawOutcome::NeedsShader`]; with `RPS5E_GPU_PLAN=1` the front-end
//! still runs (decode, plan, materialise) for diagnostics.

use super::indirect;
use super::indirect_mesh::{self, IndirectMeshKernel, MeshIndirectArgs, MeshRules};
use super::indirect_threads::{self, IndirectThreadsKernel, ThreadRules};
use super::trace;
use super::cache::PipelineDiskCache;
use super::compiler::{CompileError, ComponentKind, MeshConfig, ShaderCompileInput, ShaderCompiler, ShaderReflection};
use super::descriptors::{set_bindings, DescriptorHeap, DummyResources, LayoutCache, LayoutEntry, SamplerCache, PUSH_CONSTANT_BYTES};
use super::frontend::{hash_of, FrontendError, Prefetched, PreparedStage};
use super::hazards;
use super::primitives::{self, HostIndices};
use super::rectlist;
use super::sharps::{self, BufferSharp, SamplerSharp};
use super::state::{self, FixedFunctionKey, RectMode, VertexAttributeKey, VertexBindingKey, VertexInputKey, MAX_COLOR_ATTACHMENTS};
use crate::backend::{DrawContext, DrawOutcome, PipelineProvider};
use crate::device::{Buffer, GpuDevice, MemoryUse};
use crate::memory::GuestGpuMemory;
use crate::images::layout::SubRange;
use crate::render::{primitive, ColorTarget, IndexType, ResolvedDispatch, ResolvedDraw};
use crate::scheduler::{RenderingKey, StagingRing};
use ash::vk;
use ps5shader::resources::{
    image_binding, native_binding_index, DescriptorBindingKind, ImageDimension, ImageNumericClass, ImageResourceClass, ShaderCompileRequest,
    ShaderStage,
};
use ps5shader::{ColorComponentMapping, ComputeSystemRegisters, PixelOutputBinding, PixelOutputKind};
use crate::fxhash::HashMap;
use crate::slices::frozen::Shared;
use std::sync::Arc;

/// A fixed-function vertex buffer binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexBufferSource {
    pub address: u64,
    pub size: u64,
    pub stride: u32,
    pub per_instance: bool,
}

/// A fixed-function vertex attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexAttribute {
    pub location: u32,
    pub binding: u32,
    pub format: vk::Format,
    pub offset: u32,
}

/// The fixed-function vertex input of a draw.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VertexInputs {
    pub buffers: Vec<VertexBufferSource>,
    pub attributes: Vec<VertexAttribute>,
}

/// Supplies fixed-function vertex inputs (vertex fetch the host performs
/// instead of the program). Without one, programs fetch vertices through
/// their buffer descriptors.
pub trait VertexInputResolver: Send {
    fn resolve(&mut self, draw: &ResolvedDraw, vertex: &PreparedStage) -> Option<VertexInputs>;
}

/// Provider counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProviderStats {
    pub draws: u64,
    pub dispatches: u64,
    pub recorded_draws: u64,
    pub recorded_dispatches: u64,
    pub needs_shader: u64,
    pub skipped: u64,
    pub errors: u64,
    pub frontend_failures: u64,
    /// Stages taken from the interpreter-thread prefetch.
    pub prefetched_stages: u64,
    pub compile_failures: u64,
    pub modules: usize,
    pub module_hits: u64,
    pub graphics_pipelines: usize,
    pub graphics_hits: u64,
    pub compute_pipelines: usize,
    pub compute_hits: u64,
    pub cpu_index_conversions: u64,
    /// Strip mesh draws given a restart table.
    pub mesh_restart_tables: u64,
    pub rect_draws: u64,
    pub indirect_draws: u64,
    /// Indirect draws recorded as host indirect draws.
    pub gpu_indirect_draws: u64,
    /// Of those, mesh draws launched from records resolved on the GPU.
    pub gpu_indirect_mesh_draws: u64,
    /// Written buffers of dispatches bound to the bytes their stores reach.
    pub bounded_writes: u64,
    /// Indirect dispatches with thread dimensions launched from records
    /// resolved on the GPU.
    pub gpu_thread_dispatches: u64,
    /// Indirect group dispatches launched through a clamped record copy
    /// (`RPS5E_INDIRECT_GROUP_CLAMP=1`).
    pub gpu_group_clamps: u64,
    /// Indirect dispatches skipped for a zero count in a record no GPU work
    /// was writing.
    pub empty_indirect_dispatches: u64,
    pub heap_sets: u64,
    pub dummy_images: u64,
    pub view_memo_hits: u64,
    pub module_key_misses: u64,
    /// Draws skipped while their pipeline compiled; waits for a compile.
    pub compiling_skips: u64,
    /// Draws and dispatches given up for a pending compile and tried again
    /// after a wait with the back-end unlocked.
    pub pending_tries: u64,
    pub compile_waits: u64,
    /// Upcoming dispatches whose module and pipeline the compile lookahead
    /// queued while a draw or dispatch waited.
    pub lookahead_queued: u64,
    /// Compile jobs the command thread ran itself: it blocked on them and
    /// no worker had started them.
    pub inline_compiles: u64,
    /// Compile jobs moved to the front of the queue for an item given up
    /// to wait for them with the back-end unlocked.
    pub hastened_compiles: u64,
    /// Compiles finished within the per-frame wait budget (draws not skipped).
    pub compile_budget_waits: u64,
    /// Streamed ranges reused from an earlier copy of the tick.
    pub stream_memo_hits: u64,
    /// Draws and dispatches during which a submit moved the open tick
    /// (their staging regions were kept for the later tick).
    pub staging_retags: u64,
    /// Texture lookups the view memo did not answer.
    pub view_lookups: u64,
    /// Draws replayed from a recipe / looked up and recorded a new one.
    pub recipe_hits: u64,
    pub recipe_misses: u64,
    /// Stream copies made and their bytes.
    pub stream_copies: u64,
    pub stream_bytes: u64,
    /// Streamed bindings copied only up to the bytes their loads reach.
    pub stream_trimmed: u64,
    /// Bindings reused from the previous work of the stage slot.
    pub binding_reuses: u64,
    /// Descriptor bindings evaluated (the memo misses of `binding_reuses`).
    pub descriptor_bindings: u64,
    /// Whole-buffer pins made, and buffer resources served from them
    ///.
    pub ring_binds: u64,
    pub ring_reuses: u64,
    /// Buffer resources and device-address ranges of changed bindings
    /// reused from the previous work, and those bound or obtained again.
    pub resource_reuses: u64,
    pub resource_binds: u64,
    /// The memo misses by binding kind (GPU_PLAN §54: the address of the
    /// fingerprint work).
    pub miss_buffers: u64,
    pub miss_images: u64,
    pub miss_samplers: u64,
    /// TEMPORARY diagnostic (RPS5E_RING_DEBUG): last refusal reason
    /// (0 eligible, 1 not cacheable, 2 gate, 3 narrow layout, 4 resolved
    /// ranges, 5 written) and eligible bindings seen.
    pub ring_why: u32,
    pub ring_candidates: u64,
    /// Flattened-table words copied on the GPU from words earlier GPU
    /// work was still producing (no CPU wait).
    pub deferred_table_words: u64,
    /// Flattened-table binding evaluations: word-compares that reused the
    /// staged copy, re-uploads, and dwords uploaded.
    pub table_compares: u64,
    pub table_reuses: u64,
    pub table_dwords: u64,
    /// Draws in a scope of their own because of: GDS, storage image
    /// writes, buffer writes, attachments bound as resources.
    pub isolated: [u64; 4],
    /// Stages bound reading guest memory through its imports, and mirrors
    /// they still obtained (ranges without an import), GPU_PLAN §27.
    pub bda_stages: u64,
    pub bda_mirrored: u64,
    /// V#s such stages still materialise (formatted loads of several
    /// components need the format on the host).
    pub bda_materialized: u64,
}

/// The plain identity of a stage permutation lookup (with its raw options).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ModuleInputs {
    stage: u8,
    /// The float controls of the program's RSRC1.
    float_mode: ps5shader::translate::FloatMode,
    plan: u64,
    code: u64,
    address: u64,
    resource: u64,
    mesh: Option<MeshConfig>,
}

struct Module {
    module: vk::ShaderModule,
    reflection: ShaderReflection,
    /// The guest program (0: unknown, a prewarm entry) and the SPIR-V
    /// size in words (build-time logs).
    address: u64,
    words: usize,
    /// A relocatable module (the program's code address comes from the
    /// shader data): the code words its translation read, which the
    /// program must hold wherever it is used (see `Module::serves`).
    relocation: Option<Arc<CodeCheck>>,
}

/// The code words a relocatable module's translation read, as offsets
/// from the program's entry.
#[derive(Debug, Default)]
pub(crate) struct CodeCheck {
    reads: Vec<(u64, u32)>,
}

impl CodeCheck {
    /// The reads of a translation of the program at `base`.
    fn of(base: u64, reads: impl IntoIterator<Item = (u64, u32)>) -> CodeCheck {
        CodeCheck { reads: reads.into_iter().map(|(a, w)| (a.wrapping_sub(base), w)).collect() }
    }

    /// Whether the program at `address` holds the same words.
    fn holds(&self, address: u64, read: &dyn Fn(u64) -> Option<u32>) -> bool {
        self.reads.iter().all(|&(off, w)| read(address.wrapping_add(off)) == Some(w))
    }
}

impl Module {
    /// Whether the module serves the program at `address`: a relocatable
    /// module wherever the code words its translation read are the same,
    /// any other only where it was translated.
    fn serves(&self, address: u64, read: &dyn Fn(u64) -> Option<u32>) -> bool {
        match &self.relocation {
            Some(c) => c.holds(address, read),
            None => self.address == address,
        }
    }
}

/// How the relocatable translation of a module went.
#[derive(Clone, Copy, Debug)]
struct Relocated {
    /// The translation at another address gave the same module.
    proven: bool,
    /// The key of the module at its own address (where an unproven one
    /// is kept).
    plain_key: u64,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct GraphicsKey {
    vs: u64,
    ps: u64,
    layout: vk::PipelineLayout,
    ff: FixedFunctionKey,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ComputeKey {
    cs: u64,
    layout: vk::PipelineLayout,
    subgroup32: bool,
}

/// Stage options of a compile request (applied on a compile worker).
type Options = Arc<dyn Fn(&mut ShaderCompileRequest<'_>) + Send + Sync>;

/// A finished compile job.
enum Compiled {
    Module {
        key: u64,
        stage: ShaderStage,
        address: u64,
        compiler: String,
        result: Result<(Arc<Module>, Vec<u32>), CompileError>,
        translate_ns: u64,
        module_ns: u64,
        /// The guest inputs of the translation (kept in the prewarm list).
        source: Option<Arc<super::sources::ModuleSource>>,
        /// A relocatable translation and its outcome.
        relocated: Option<Relocated>,
    },
    Graphics { key: GraphicsKey, result: Result<vk::Pipeline, vk::Result>, record: Option<super::prewarm::GraphicsRecord>, built: Built, rebuild: Option<Rebuild> },
    Compute { key: ComputeKey, result: Result<vk::Pipeline, vk::Result>, record: Option<super::prewarm::ComputeRecord>, built: Built, rebuild: Option<Rebuild> },
}

/// How a pipeline was built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tier {
    /// With the driver's optimisation (cached or compiled).
    Full,
    /// Found in the pipeline cache by the probe (no compile).
    Cached,
    /// Without driver optimisation: used at once, replaced by its
    /// [`Tier::Optimised`] build when that is ready.
    Fast,
    /// The optimised build replacing a [`Tier::Fast`] pipeline.
    Optimised,
}

/// The tier of a built pipeline and how long the driver took.
#[derive(Clone, Copy, Debug)]
struct Built {
    tier: Tier,
    ns: u64,
}

/// `RPS5E_FAST_PIPELINES=1`: a pipeline the pipeline cache does not hold
/// is first built without driver optimisation
/// (`VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT`; draws and dispatches use
/// it at once) while its optimised build runs on a compile worker behind
/// every other job and replaces it when ready. `only` (diagnostic): no
/// optimised rebuilds. Off by default: the NVIDIA driver (615) takes as
/// long without optimisation as with it, so the first tier
/// gains nothing there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FastPipelines {
    Off,
    On,
    Only,
}

fn fast_pipelines_mode() -> FastPipelines {
    static MODE: std::sync::OnceLock<FastPipelines> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("RPS5E_FAST_PIPELINES").as_deref() {
        _ if crate::backend::ab::safe() => FastPipelines::Off,
        Ok("only") => FastPipelines::Only,
        Ok("1") | Ok("on") => FastPipelines::On,
        _ => FastPipelines::Off,
    })
}

/// Build a pipeline with `create` (given the pipeline cache and extra
/// flags). `fast`: probe the cache first, and when it does not hold the
/// pipeline, build it without driver optimisation (not kept in the cache:
/// the optimised build replaces it). `what` names the pipeline in the log
/// of slow builds.
fn build_pipeline(
    cache: vk::PipelineCache,
    fast: bool,
    optimised: bool,
    what: &dyn Fn() -> String,
    create: &dyn Fn(vk::PipelineCache, vk::PipelineCreateFlags) -> Result<vk::Pipeline, vk::Result>,
) -> (Result<vk::Pipeline, vk::Result>, Built) {
    let _watch = super::watch::start(format!("{} pipeline", what()));
    let t = std::time::Instant::now();
    let (result, tier) = if fast {
        match create(cache, vk::PipelineCreateFlags::FAIL_ON_PIPELINE_COMPILE_REQUIRED) {
            Ok(p) if p != vk::Pipeline::null() => (Ok(p), Tier::Cached),
            Err(e) if e != vk::Result::PIPELINE_COMPILE_REQUIRED => (Err(e), Tier::Cached),
            _ => (create(vk::PipelineCache::null(), vk::PipelineCreateFlags::DISABLE_OPTIMIZATION), Tier::Fast),
        }
    } else {
        (create(cache, vk::PipelineCreateFlags::empty()), if optimised { Tier::Optimised } else { Tier::Full })
    };
    let ns = t.elapsed().as_nanos() as u64;
    if ns >= super::timing::SLOW_MS * 1_000_000 {
        log::info!("ps5gpu: {} pipeline built in {} ms ({tier:?})", what(), ns / 1_000_000);
    }
    (result, Built { tier, ns })
}

/// `RPS5E_AUTO_INDEX_OFFSET=0`: auto draws start at vertex 0 whatever
/// GE_INDX_OFFSET holds.
fn auto_index_offset_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_AUTO_INDEX_OFFSET").is_ok_and(|v| v == "0"))
}

/// `RPS5E_COMPILE_WAIT_MS`: per presented frame, how long draws may wait
/// for their modules and pipelines to finish compiling before they are
/// skipped (0 skips at once); `exact` (or unset) makes draws always wait.
/// A skipped pass leaves its targets without the frame's results, and the
/// targets games accumulate over frames (exposure, temporal history,
/// lighting) then fade back in from black, so tests, replays and smoke
/// runs keep the exact mode; interactive play (`rps5e-cli run`, the
/// launcher) sets a small budget by default.
fn compile_wait_budget() -> std::time::Duration {
    static BUDGET: std::sync::OnceLock<std::time::Duration> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| parse_compile_wait(std::env::var("RPS5E_COMPILE_WAIT_MS").ok().as_deref()))
}

/// The budget a `RPS5E_COMPILE_WAIT_MS` value names (None: unset).
pub fn parse_compile_wait(v: Option<&str>) -> std::time::Duration {
    match v.map(str::trim) {
        None | Some("") => std::time::Duration::MAX,
        Some(v) if ["exact", "wait", "inf"].iter().any(|w| v.eq_ignore_ascii_case(w)) => std::time::Duration::MAX,
        Some(v) => v.parse().map_or(std::time::Duration::MAX, std::time::Duration::from_millis),
    }
}

/// `RPS5E_RELOCATABLE_CODE=0`: a program that reads its own address
/// (S_GETPC_B64) gets a module per address it is loaded at, as before
/// relocatable modules.
fn relocation_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_RELOCATABLE_CODE").is_ok_and(|v| v == "0"))
}

/// The address the key of a relocatable module names instead of the
/// program's.
const RELOCATABLE_ADDRESS: u64 = u64::MAX;

/// The key of the relocatable module of a permutation (None: the program
/// does not read its address, or cannot be relocated: a split program, or
/// one without the shader base in its shader data).
fn relocation_key(p: &super::frontend::StageProgram, prepared: &PreparedStage, options: u64, mesh: Option<MeshConfig>, float_mode: ps5shader::translate::FloatMode) -> Option<u64> {
    (relocation_enabled() && !p.address_independent && p.continuation == 0 && prepared.layout.uses_shader_base())
        .then(|| hash_of(&(p.plan.hash, p.code_hash, RELOCATABLE_ADDRESS, prepared.resource_key, options, mesh, float_mode)))
}

/// The address and code check of a module of the prewarm list with
/// inputs `source`: a proven relocatable module checks the code words its
/// translation read; any other is the module of its key only.
fn proven_relocation(source: Option<&super::sources::ModuleSource>) -> (u64, Option<Arc<CodeCheck>>) {
    match source {
        Some(s) if super::sources::proven_relocatable(&s.request) => {
            let base = s.program.address;
            (base, Some(Arc::new(CodeCheck::of(base, s.reads.words_iter()))))
        }
        _ => (0, None),
    }
}

/// `RPS5E_ASYNC_PIPELINES=0`: modules and pipelines are compiled on the
/// command thread when first needed.
fn async_pipelines_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_ASYNC_PIPELINES"))
}

/// The compile workers of a running game (none with
/// `RPS5E_ASYNC_PIPELINES=0`): every core but two, for the modules and
/// pipelines draws and dispatches ask for and for the prewarm list alike
/// (`RPS5E_COMPILE_THREADS=N` overrides); background rebuilds of usable
/// pipelines leave two of them free. Workers sleep when
/// nothing is queued.
fn default_pool() -> Option<super::pool::CompilePool<Compiled>> {
    async_pipelines_enabled().then(|| {
        let (threads, share) = pool_size(std::thread::available_parallelism().map_or(4, |n| n.get()), compile_threads_env());
        super::pool::CompilePool::with_share(threads, share)
    })
}

/// `RPS5E_COMPILE_THREADS`: compile workers in all.
fn compile_threads_env() -> Option<usize> {
    std::env::var("RPS5E_COMPILE_THREADS").ok().and_then(|v| v.trim().parse().ok()).filter(|&n| n > 0)
}

/// (workers, prewarm share) of the game's pool on `cores` logical CPUs.
fn pool_size(cores: usize, threads: Option<usize>) -> (usize, usize) {
    let threads = threads.unwrap_or_else(|| cores.saturating_sub(2).max(2)).max(1);
    (threads, threads)
}

/// The compile lookahead state of a provider (see `compile_ahead`).
#[derive(Default)]
struct Ahead {
    tracker: Option<Arc<crate::memory::tracker::GuestPageTracker>>,
    mark: Option<(u64, u64)>,
    position: u64,
    handled: crate::fxhash::HashSet<u64>,
    queue_only: bool,
}

/// The compute stage's module permutation of a dispatch: the stage
/// options as plain words, their hash (part of the module key) and what
/// the request takes from them.
struct ComputePermutation {
    threads: [u32; 3],
    system: ComputeSystemRegisters,
    lds_dwords: u32,
    shared_int64: bool,
    raw: [u32; 8],
    key: u64,
}

impl ComputePermutation {
    fn of(d: &ResolvedDispatch, shared_int64: bool) -> ComputePermutation {
        let threads = [d.threads[0].max(1), d.threads[1].max(1), d.threads[2].max(1)];
        let rsrc2 = d.shader.rsrc2;
        // System SGPRs follow the user data in this order when enabled.
        let mut next = d.shader.user_sgpr_count();
        let mut system_sgpr = |bit: u32| -> Option<u32> {
            ((rsrc2 >> bit) & 1 != 0).then(|| {
                next += 1;
                next - 1
            })
        };
        let system = ComputeSystemRegisters {
            work_group_x: system_sgpr(7),
            work_group_y: system_sgpr(8),
            work_group_z: system_sgpr(9),
            thread_group_size: system_sgpr(10),
        };
        let lds_dwords = ((rsrc2 >> 15) & 0x1FF) * 128;
        let key = hash_of(&(threads, system.work_group_x, system.work_group_y, system.work_group_z, system.thread_group_size, lds_dwords));
        let sys = |r: Option<u32>| r.map_or(0, |v| v + 1);
        let raw = [threads[0], threads[1], threads[2], sys(system.work_group_x), sys(system.work_group_y), sys(system.work_group_z), sys(system.thread_group_size), lds_dwords];
        ComputePermutation { threads, system, lds_dwords, shared_int64, raw, key }
    }

    fn options(&self) -> Options {
        let (threads, system, lds_dwords, shared_int64) = (self.threads, self.system, self.lds_dwords, self.shared_int64);
        Arc::new(move |r: &mut ShaderCompileRequest<'_>| {
            r.local_size = threads;
            r.compute_system_registers = Some(system);
            r.local_data_share_dwords = lds_dwords;
            r.supports_shared_int64_atomics = shared_int64;
        })
    }
}

/// A pipeline of the prewarm list.
enum PrewarmPipeline {
    Graphics(super::prewarm::GraphicsRecord),
    Compute(super::prewarm::ComputeRecord),
}

#[derive(Clone, Copy, Debug, Default)]
struct PrewarmCounts {
    /// Modules of the list translated again / that failed to.
    translated: usize,
    failed: usize,
}

/// Progress of [`ShaderPipelines::prewarm_now`].
#[derive(Clone, Copy, Debug)]
pub struct PrewarmProgress {
    /// Jobs (module translations and pipelines) done and in all.
    pub done: usize,
    pub total: usize,
    pub elapsed: std::time::Duration,
}

impl PrewarmProgress {
    /// The time left at the rate so far.
    pub fn eta(&self) -> Option<std::time::Duration> {
        (self.done > 0 && self.done < self.total).then(|| self.elapsed.mul_f64((self.total - self.done) as f64 / self.done as f64))
    }
}

/// What [`ShaderPipelines::prewarm_now`] did.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrewarmSummary {
    /// Modules of the list, those translated again and those that failed to.
    pub modules: usize,
    pub translated: usize,
    pub translate_failures: usize,
    /// Pipelines of the list, built and failed (the others lacked a module).
    pub pipelines: usize,
    pub built: usize,
    pub pipeline_failures: usize,
    pub seconds: f64,
    /// Stopped early (process shutdown).
    pub cancelled: bool,
}

impl std::fmt::Display for PrewarmSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} modules ({} translated again, {} failed), {} of {} pipelines built ({} failed) in {:.1} s{}",
            self.modules,
            self.translated,
            self.translate_failures,
            self.built,
            self.pipelines,
            self.pipeline_failures,
            self.seconds,
            if self.cancelled { " (cancelled)" } else { "" }
        )
    }
}

/// A module of `words`.
fn create_shader_module(dev: &GpuDevice, words: &[u32]) -> Result<vk::ShaderModule, vk::Result> {
    // SAFETY: valid SPIR-V words (the compiler's contract).
    unsafe { dev.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None) }
}

/// Descriptor writes of one draw (all stages), with owned infos. Kept by
/// the provider and cleared per draw (no allocation once grown).
#[derive(Default)]
struct SetWrites {
    /// Guest ranges the work writes through buffers (images over them go
    /// stale).
    written_ranges: Vec<(u64, u64)>,
    /// Written ranges whose buffers the work does not read.
    overwritten: Vec<(u64, u64)>,
    /// Cached images bound as textures or storage (feedback checks).
    bound_images: Vec<u64>,
    /// Cached images the work writes (storage).
    written_images: Vec<u64>,
    /// DCC keys of the storage images written through a descriptor with
    /// DCC compression on ([`compressed_store_keys`]).
    compressed_stores: Vec<crate::render::StoredKeys>,
    /// Image descriptors of cached images; their layouts are set once
    /// every binding is known.
    image_slots: Vec<ImageSlot>,
    /// A stage binds the global data share (ordered against other work).
    gds: bool,
    buffers: Vec<vk::DescriptorBufferInfo>,
    images: Vec<vk::DescriptorImageInfo>,
    entries: Vec<(u32, vk::DescriptorType, usize, usize, bool)>,
}

/// An image descriptor of a cached image.
#[derive(Clone, Copy)]
struct ImageSlot {
    /// Index in `SetWrites::images` (in a [`BindingMemo`]: from the
    /// binding's first image).
    index: usize,
    id: u64,
    storage: bool,
    /// The subresources of the view, and whether the image is a depth
    /// image (looked up once, when the view is).
    range: Option<SubRange>,
    depth: bool,
}

/// Write descriptors of one set held on the stack.
type WriteList<'a> = smallvec::SmallVec<[vk::WriteDescriptorSet<'a>; 24]>;

impl SetWrites {
    fn clear(&mut self) {
        self.written_ranges.clear();
        self.overwritten.clear();
        self.bound_images.clear();
        self.written_images.clear();
        self.compressed_stores.clear();
        self.image_slots.clear();
        self.gds = false;
        self.buffers.clear();
        self.images.clear();
        self.entries.clear();
    }

    /// Start a binding; its infos follow with [`Self::buffer`] /
    /// [`Self::image`].
    fn open(&mut self, binding: u32, ty: vk::DescriptorType, buffer: bool) {
        let start = if buffer { self.buffers.len() } else { self.images.len() };
        self.entries.push((binding, ty, start, 0, buffer));
    }

    fn buffer(&mut self, info: vk::DescriptorBufferInfo) {
        self.buffers.push(info);
        if let Some(e) = self.entries.last_mut() {
            e.3 += 1;
        }
    }

    fn image(&mut self, info: vk::DescriptorImageInfo) {
        self.images.push(info);
        if let Some(e) = self.entries.last_mut() {
            e.3 += 1;
        }
    }

    /// A binding of one buffer.
    fn single_buffer(&mut self, binding: u32, info: vk::DescriptorBufferInfo) {
        self.open(binding, vk::DescriptorType::STORAGE_BUFFER, true);
        self.buffer(info);
    }

    fn writes(&self, set: vk::DescriptorSet) -> WriteList<'_> {
        self.entries
            .iter()
            .map(|&(binding, ty, start, n, buffer)| {
                let w = vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding).descriptor_type(ty);
                if buffer {
                    w.buffer_info(&self.buffers[start..start + n])
                } else {
                    w.image_info(&self.images[start..start + n])
                }
            })
            .collect()
    }
}

/// Slot of a stage's binding memo.
/// `RPS5E_BDA_PROBE=<addr>[,<addr>...]`: log what stages at these program
/// addresses read through guest memory (V#s, flattened words, guest bytes).
fn bda_probe(address: u64) -> bool {
    static LIST: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        std::env::var("RPS5E_BDA_PROBE")
            .map(|v| v.split(',').filter_map(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()).collect())
            .unwrap_or_default()
    })
    .contains(&address)
}

/// Bytes from a V#'s base that a stage reading guest memory gets written
/// through before it runs (unbounded V#s reach the whole address space).
const BDA_CURRENT_LIMIT: u64 = 64 << 20;

fn stage_slot(stage: ShaderStage) -> usize {
    match stage {
        ShaderStage::Pixel => 1,
        ShaderStage::Compute => 2,
        _ => 0,
    }
}

/// `RPS5E_SPLIT_STAMPS=0`: a buffer content or layout change drops every
/// memoised binding of a stage, images and samplers included.
fn split_stamps_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| !std::env::var("RPS5E_SPLIT_STAMPS").is_ok_and(|v| v == "0"))
}

/// Stage memos kept per stage slot (UE4 changes material, and with it the
/// stage's binding layout, on most draws).
const STAGE_MEMOS: usize = 16;

/// `RPS5E_STAGE_MEMO_LRU=0`: one stage memo per stage slot (a draw with
/// another binding layout drops the previous one's bindings).
fn stage_memo_lru_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_STAGE_MEMO_LRU"))
}

/// `RPS5E_RESOURCE_MEMO=0`: a buffer binding whose words changed binds
/// every buffer of it again, and device-address ranges are obtained again
/// unless all of them are unchanged. On, the buffers and ranges of a
/// changed binding that repeat the previous work's (same V# words, same
/// range, under the same buffer stamp) reuse its results: the memo's
/// validity argument, per resource instead of per binding.
pub(crate) fn resource_memo_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_RESOURCE_MEMO"))
}

/// `RPS5E_SLOT_LOOKUPS=0`: the layout of every sampled image is worked
/// out from its view and image again on every draw instead of from what
/// its binding found.
fn slot_lookups_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_SLOT_LOOKUPS"))
}

/// ANISO_OVERRIDE applies to sampler `r` when every image it filters has
/// one level (BASE_LEVEL == LAST_LEVEL).
fn sampler_single_level(info: &ps5shader::resources::ShaderResourceInfo, snapshot: &ps5shader::resources::ResourceSnapshot, r: u32) -> bool {
    let mut paired = info.sampled_pairs.iter().filter(|pair| pair.sampler == r).peekable();
    paired.peek().is_some()
        && paired.all(|pair| {
            snapshot.images.get(pair.image as usize).is_some_and(|w| {
                let t = sharps::TextureSharp::from_words(w);
                t.base_level() == t.last_level()
            })
        })
}

/// Image `r` is filtered by a sampler with unnormalized coordinates
/// (FORCE_UNNORMALIZED): such a read takes the descriptor's base level,
/// and the host view must be one level (`RPS5E_UNNORMALIZED_BASE_LEVEL=0`
/// keeps the whole chain).
pub(crate) fn image_unnormalized(info: &ps5shader::resources::ShaderResourceInfo, snapshot: &ps5shader::resources::ResourceSnapshot, r: u32) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_UNNORMALIZED_BASE_LEVEL").is_ok_and(|v| v == "0"))
        && info
            .sampled_pairs
            .iter()
            .any(|pair| pair.image == r && snapshot.samplers.get(pair.sampler as usize).and_then(|w| w.first()).is_some_and(|w0| (w0 >> 15) & 1 != 0))
}

/// Image `r` is filtered by a sampler with FORCE_DEGAMMA (S# word 0 bit
/// 20): the sampler decodes the texels as sRGB whatever the image format
/// (`RPS5E_FORCE_DEGAMMA=0` ignores the bit).
pub(crate) fn image_force_degamma(info: &ps5shader::resources::ShaderResourceInfo, snapshot: &ps5shader::resources::ResourceSnapshot, r: u32) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_FORCE_DEGAMMA").is_ok_and(|v| v == "0"))
        && info
            .sampled_pairs
            .iter()
            .any(|pair| pair.image == r && snapshot.samplers.get(pair.sampler as usize).and_then(|w| w.first()).is_some_and(|w0| (w0 >> 20) & 1 != 0))
}

/// `RPS5E_MESH_LINES=0`: geometry programs emitting line strips are not
/// drawn (as before line output was modelled).
/// `RPS5E_MESH_POINTS=0`: geometry programs emitting point lists are
/// refused (as before they were drawn as mesh points).
fn mesh_points_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_MESH_POINTS").is_ok_and(|v| v == "0"))
}

fn mesh_lines_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_MESH_LINES").is_ok_and(|v| v == "0"))
}

/// `RPS5E_STENCIL_EXPORT=0`: the stencil test value export of pixel
/// programs is ignored (the reference register is used).
fn stencil_export_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_STENCIL_EXPORT").is_ok_and(|v| v == "0"))
}

/// `RPS5E_DISPATCH_SPLIT=0`: a direct dispatch over the device's group
/// count limit is skipped instead of split into chunks.
fn dispatch_split_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_DISPATCH_SPLIT").is_ok_and(|v| v == "0"))
}

/// The (base group, group count) chunks of a dispatch of `count` groups
/// from group `start`, each count within `limit` per axis.
pub(crate) fn dispatch_chunks(start: [u32; 3], count: [u32; 3], limit: [u32; 3]) -> Vec<([u32; 3], [u32; 3])> {
    let axis = |i: usize| -> Vec<(u32, u32)> {
        let step = limit[i].max(1);
        let mut v = Vec::new();
        let mut done = 0u32;
        while done < count[i] {
            let n = (count[i] - done).min(step);
            v.push((start[i].wrapping_add(done), n));
            done += n;
        }
        v
    };
    let (xs, ys, zs) = (axis(0), axis(1), axis(2));
    let mut out = Vec::with_capacity(xs.len() * ys.len() * zs.len());
    for &(bz, nz) in &zs {
        for &(by, ny) in &ys {
            for &(bx, nx) in &xs {
                out.push(([bx, by, bz], [nx, ny, nz]));
            }
        }
    }
    out
}

#[cfg(test)]
mod dispatch_chunk_tests {
    /// A dispatch over the device limit splits into chunks within it that
    /// cover every group once, at their base groups.
    #[test]
    fn chunks_cover_the_dispatch() {
        let chunks = super::dispatch_chunks([0, 0, 0], [150_000, 2, 1], [65_535, 65_535, 65_535]);
        assert_eq!(chunks, vec![([0, 0, 0], [65_535, 2, 1]), ([65_535, 0, 0], [65_535, 2, 1]), ([131_070, 0, 0], [18_930, 2, 1])]);
        let chunks = super::dispatch_chunks([3, 0, 0], [10, 7, 1], [4, 5, 1]);
        let mut seen = std::collections::HashSet::new();
        for (b, n) in &chunks {
            assert!(n[0] <= 4 && n[1] <= 5 && n[2] <= 1);
            for x in b[0]..b[0] + n[0] {
                for y in b[1]..b[1] + n[1] {
                    assert!(seen.insert((x, y)));
                }
            }
        }
        assert_eq!(seen.len(), 70);
        assert!(seen.contains(&(3, 0)) && seen.contains(&(12, 6)));
        assert_eq!(super::dispatch_chunks([0, 0, 0], [5, 1, 1], [8, 8, 8]), vec![([0, 0, 0], [5, 1, 1])]);
    }
}

/// `RPS5E_PS_ITER_SAMPLE=0`: pixel programs run once per pixel whatever
/// PA_SC_MODE_CNTL_1.PS_ITER_SAMPLE says.
fn ps_iter_sample_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_PS_ITER_SAMPLE").is_ok_and(|v| v == "0"))
}

/// The pipeline sample mask of PA_SC_AA_MASK: the samples the four pixels
/// of a quad keep when they agree (all ones when every sample is kept or
/// the pixels differ: a per-pixel mask is not modelled yet).
/// `RPS5E_AA_MASK=0` keeps every sample.
pub(crate) fn static_sample_mask(d: &ResolvedDraw, samples: u32) -> u32 {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if d.aa_mask == [u32::MAX; 2] || !*ON.get_or_init(|| !std::env::var("RPS5E_AA_MASK").is_ok_and(|v| v == "0")) {
        return u32::MAX;
    }
    let masks = crate::render::quad_sample_masks(d.aa_mask, samples);
    let full = if samples >= 16 { 0xFFFF } else { (1u32 << samples.max(1)) - 1 };
    if masks.iter().all(|&m| m == full) {
        return u32::MAX;
    }
    if masks.iter().any(|&m| m != masks[0]) {
        // The pixel program masks per pixel (`quad_sample_masks`).
        return u32::MAX;
    }
    masks[0]
}

/// `RPS5E_QUAD_MASK_DATA=0`: per-pixel sample masks are constants of the
/// pixel module (one module per mask) instead of shader data.
fn quad_mask_data_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_QUAD_MASK_DATA").is_ok_and(|v| v == "0"))
}

/// The `pixel_quad_sample_masks` of a module that reads the masks from its
/// shader data (their values are not part of the module).
pub(crate) const QUAD_MASKS_AT_RUN_TIME: [u32; 4] = [u32::MAX; 4];

/// The shader-data words of per-pixel sample masks `m` (16 bits each).
pub(crate) fn pack_quad_masks(m: [u32; 4]) -> [u32; 2] {
    [m[0] & 0xFFFF | (m[1] & 0xFFFF) << 16, m[2] & 0xFFFF | (m[3] & 0xFFFF) << 16]
}

/// PA_SC_AA_MASK when the four pixels of a quad keep different samples:
/// the pixel program's sample mask starts from its pixel's mask.
pub(crate) fn quad_sample_masks(d: &ResolvedDraw, samples: u32) -> Option<[u32; 4]> {
    if d.aa_mask == [u32::MAX; 2] || static_sample_mask(d, samples) != u32::MAX {
        return None;
    }
    let masks = crate::render::quad_sample_masks(d.aa_mask, samples);
    masks.iter().any(|&m| m != masks[0]).then_some(masks)
}

/// The class of the primitives a draw rasterizes, as the pixel ancillary
/// VGPR reports it: 0 point, 1 line, 2 triangle, 3 rectangle. Geometry
/// subgroups rasterize their output primitives, tessellation triangles.
fn raster_primitive_class(d: &ResolvedDraw, mesh: Option<&MeshConfig>) -> u32 {
    if d.primitive_type == primitive::RECT_LIST {
        return 3;
    }
    if let Some(m) = mesh {
        return if m.lines { 1 } else { 2 };
    }
    if d.shaders.stages & STAGES_HS_EN != 0 {
        return 2;
    }
    match d.primitive_type {
        primitive::POINT_LIST => 0,
        primitive::LINE_LIST | primitive::LINE_STRIP => 1,
        _ => 2,
    }
}

/// `RPS5E_BIND_MEMO=0`: every draw binds every resource again.
fn bind_memo_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::fast() && *ON.get_or_init(|| !std::env::var("RPS5E_BIND_MEMO").is_ok_and(|v| v == "0"))
}

/// What the binding results of a stage depend on beyond their input
/// words: the open tick (staging copies, image last-use ticks), the buffer
/// cache's content stamp (guest bytes, CPU/GPU dirtiness), the image
/// cache's lookup epoch (which image and view a T# names) and the buffer
/// layout epoch (which buffer holds a range).
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct StageStamp {
    tick: u64,
    content: [u64; 4],
    images: u64,
    buffers: u64,
}

/// One read-only buffer resource pinned to its whole containing buffer
///: the binding is the buffer itself and the rotating V#
/// base rides in the wide offset dword, so a constant ring handing out a
/// new address every draw keeps its binding identity.
#[derive(Clone)]
struct RingBind {
    /// This draw's clamped range (the exact bytes the mirror must hold).
    a: u64,
    n: u64,
    /// Identity: the containing buffer's cache key and serial.
    key: u64,
    serial: u64,
    /// The byte offset of `a` in the buffer (the wide slot's value).
    offset: u32,
    /// The whole-buffer binding.
    info: vk::DescriptorBufferInfo,
    /// The wide slot in the shader data.
    dword: u32,
}

/// The results of one binding of the previous work of a stage slot.
#[derive(Default)]
struct BindingMemo {
    valid: bool,
    /// Input words (the descriptor words of its resources, separated).
    words: smallvec::SmallVec<[u32; 32]>,
    ty: vk::DescriptorType,
    buffers: smallvec::SmallVec<[vk::DescriptorBufferInfo; 4]>,
    images: smallvec::SmallVec<[vk::DescriptorImageInfo; 8]>,
    /// The cached images (index in `images`).
    slots: smallvec::SmallVec<[ImageSlot; 8]>,
    /// The lookup generation of each slot's image when it was bound.
    generations: smallvec::SmallVec<[u64; 8]>,
    /// The tick the slots' images were last marked used in.
    tick: u64,
    /// Shader-data offset adjustments: (dword, bits or-ed in).
    adjust: smallvec::SmallVec<[(u16, u32); 4]>,
    /// A slice thread's binding: what making it read (replayed into the
    /// read set of every draw that reuses it, R7b).
    spec_reads: Option<Box<crate::slices::spec::BindingReads>>,
}

impl BindingMemo {
    /// The memoised images are still what their lookups give: each image's
    /// lookup generation is unchanged and, once per tick, the image needs
    /// no upload (and is marked used by the tick).
    fn images_current(&mut self, images: &mut crate::slices::spec::Images, tick: u64) -> bool {
        if self.slots.iter().zip(&self.generations).any(|(s, &g)| images.image_generation(s.id) != Some(g)) {
            return false;
        }
        if let crate::slices::spec::Images::Spec(cache, spec) = images {
            for s in &self.slots {
                spec.note_image(cache, s.id);
            }
        }
        if self.tick != tick {
            if self.slots.iter().any(|s| images.needs_upload(s.id)) {
                return false;
            }
            for s in &self.slots {
                images.touch(s.id, tick);
            }
            self.tick = tick;
        }
        true
    }
}

/// Binding results of the previous work of one stage slot (vertex, pixel,
/// compute). Work binding a stage of the same layout and specialised
/// resources under the same [`StageStamp`] reuses each binding whose input
/// words are unchanged: constant-buffer pointers change from draw to draw
/// while textures and samplers stay. Bindings with side effects (written
/// buffers and images, the global data share) are always bound again.
/// Binding memos kept per binding, most recent first: a material's ring
/// revisits a few word sets (UE4's constant-address slots), so misses of
/// the newest still hit an older way.
const BINDING_WAYS: usize = 4;

/// Boxed: ways rotate and are recycled by pointer (a memo is ~1 KiB).
type BindingWays = smallvec::SmallVec<[Box<BindingMemo>; BINDING_WAYS]>;

/// The binding of one read-only buffer resource of a stage (see
/// [`resource_memo_enabled`]): valid while its generation is the memo's.
#[derive(Clone, Copy, Default)]
struct ResourceMemo {
    generation: u64,
    words: [u32; 4],
    /// The guest range bound.
    range: (u64, u64),
    info: vk::DescriptorBufferInfo,
    /// The offset adjustment patched into the shader data.
    adjustment: u32,
}

#[derive(Default)]
struct StageMemo {
    stamp: StageStamp,
    layout: Option<Arc<ps5shader::resources::BindingLayout>>,
    resources: Option<Arc<ps5shader::resources::SpecializedResourceInfo>>,
    bindings: Vec<BindingWays>,
    /// Device-address ranges obtained (none written).
    ranges: Option<smallvec::SmallVec<[ps5shader::resources::DeviceAddressRange; 4]>>,
    /// The buffer content stamp, layout epoch and tick the buffer bindings
    /// and ranges were made under (with split stamps): mirrors are marked
    /// used, and stream copies made, per tick.
    buffer_stamp: ([u64; 4], u64, u64),
    /// The tick of `shader_data` and `table`.
    staging_tick: u64,
    /// Staging copies of the shader data and of the flattened table (with
    /// its reserved size) in the stamp's tick.
    shader_data: Option<(smallvec::SmallVec<[u32; 48]>, vk::DescriptorBufferInfo)>,
    table: Option<(smallvec::SmallVec<[u32; 64]>, usize, vk::DescriptorBufferInfo)>,
    /// Read-only buffer bindings per resource index, and the generation
    /// they are valid in (moved where buffer bindings are dropped).
    buffer_memos: Vec<ResourceMemo>,
    buffer_generation: u64,
}

impl StageMemo {
    fn matches(&self, stamp: &StageStamp, layout: &Arc<ps5shader::resources::BindingLayout>, resources: &Arc<ps5shader::resources::SpecializedResourceInfo>) -> bool {
        self.stamp == *stamp && self.layout.as_ref().is_some_and(|l| Arc::ptr_eq(l, layout)) && self.resources.as_ref().is_some_and(|r| Arc::ptr_eq(r, resources))
    }

    fn reset(&mut self, stamp: StageStamp, layout: &Arc<ps5shader::resources::BindingLayout>, resources: &Arc<ps5shader::resources::SpecializedResourceInfo>) {
        self.stamp = stamp;
        self.layout = Some(layout.clone());
        self.resources = Some(resources.clone());
        for b in &mut self.bindings {
            for m in b.iter_mut() {
                m.valid = false;
            }
        }
        self.ranges = None;
        self.buffer_generation += 1;
    }
}

/// What a draw's modules, attachments, layout and pipeline follow from:
/// the prepared stages (program, resource key and binding layout, by
/// identity: the recipe holds them), their stage options as plain words,
/// the primitive type, the bound targets (registers and cached images) and
/// the rasterizer registers the fixed-function state reads. The vertex
/// input layout is compared separately (without building its key).
#[derive(Clone, PartialEq)]
struct RecipeKey {
    vs: usize,
    vs_resource: u64,
    vs_layout: usize,
    vs_rsrc1: u32,
    vs_raw: [u32; VS_RAW_WORDS],
    ps: usize,
    ps_resource: u64,
    ps_layout: usize,
    ps_rsrc1: u32,
    ps_raw: smallvec::SmallVec<[u32; 4 * MAX_COLOR_ATTACHMENTS + 36]>,
    primitive: u32,
    single_rect: bool,
    restart: bool,
    colors: smallvec::SmallVec<[RecipeColor; MAX_COLOR_ATTACHMENTS]>,
    depth: Option<RecipeDepth>,
    aa_config: u32,
    raster: [u32; 6],
    /// The whole bound targets (`RPS5E_RECIPE_KEY=0`; boxed: the key is
    /// built for every draw).
    #[allow(clippy::type_complexity)]
    whole: Option<Box<(smallvec::SmallVec<[(ColorTarget, u64); MAX_COLOR_ATTACHMENTS]>, Option<(crate::render::DepthTarget, u64)>)>>,
}

/// What a recipe takes from a bound colour target: its image, the view
/// (format, level, layers), the extent and the blend and write state.
/// Metadata addresses, clear words and the surface layout (the image's
/// key) are left out.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct RecipeColor {
    image: u64,
    /// slot, host format, export mapping, layout, number type, order,
    /// width, height, samples, level, base layer, layers, write mask,
    /// CB_BLEND_CONTROL, blend enable, CB_COLOR_INFO.
    words: [u32; 16],
}

impl RecipeColor {
    fn of(t: &ColorTarget, image: u64) -> RecipeColor {
        RecipeColor {
            image,
            words: [
                t.slot,
                t.format.host.as_raw() as u32,
                t.format.export_mapping.0 as u32,
                t.layout,
                t.number_type,
                t.order,
                t.width,
                t.height,
                t.samples,
                t.mip_level,
                t.base_layer,
                t.layers,
                t.write_mask,
                t.blend.raw,
                t.blend.enable as u32 | (t.attrib >> 17 & 1) << 1,
                t.info,
            ],
        }
    }
}

/// What a recipe takes from the bound depth target: its image, layers and
/// extent.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct RecipeDepth {
    image: u64,
    words: [u32; 5],
}

impl RecipeDepth {
    fn of(z: &crate::render::DepthTarget, image: u64) -> RecipeDepth {
        RecipeDepth { image, words: [z.base_layer, z.layers, z.width, z.height, z.samples] }
    }
}

impl RecipeKey {
    fn hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = rustc_hash::FxHasher::default();
        (self.vs, self.vs_resource, self.ps, self.ps_resource, self.primitive, self.vs_raw).hash(&mut h);
        self.ps_raw.hash(&mut h);
        if self.whole.is_some() {
            for c in &self.colors {
                (c.image, c.words[0]).hash(&mut h);
            }
            self.depth.map(|z| z.image).hash(&mut h);
        } else {
            (&self.colors, &self.depth, self.vs_layout, self.ps_layout, self.vs_rsrc1, self.ps_rsrc1, self.aa_config, self.raster, self.single_rect, self.restart).hash(&mut h);
        }
        h.finish()
    }
}

/// `RPS5E_RECIPE_KEY=0`: recipe keys hold the whole bound targets and
/// hash only their images, one recipe per hash.
pub(crate) fn compact_recipe_key_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_RECIPE_KEY"))
}

/// Recipes kept per key hash (other vertex layouts of one key, colliding
/// keys).
const RECIPE_BUCKET: usize = 4;

/// A recorded draw's per-draw derivations (the reference driver's draw
/// recipe): stage modules, rectangle stages, attachment formats, blend
/// state, views and extent, pipeline layout and graphics pipeline. A later
/// draw with an equal [`RecipeKey`] and vertex input layout takes them as
/// they are instead of looking each up again. Everything referenced lives
/// as long as the provider (modules and pipelines are never destroyed,
/// views live with their image, whose id is part of the key and never
/// reused), and the stages' programs and layouts are held so that their
/// addresses in the key stay theirs.
struct DrawRecipe {
    key: RecipeKey,
    vertex: VertexInputKey,
    vs_key: u64,
    vs_module: Arc<Module>,
    ps: Option<(u64, Arc<Module>)>,
    tess: Option<(vk::ShaderModule, vk::ShaderModule)>,
    color_formats: [i32; MAX_COLOR_ATTACHMENTS],
    blend: [state::BlendKey; MAX_COLOR_ATTACHMENTS],
    color_views: [vk::ImageView; MAX_COLOR_ATTACHMENTS],
    color_count: u32,
    attachments: smallvec::SmallVec<[(u64, bool, SubRange); MAX_COLOR_ATTACHMENTS + 1]>,
    attachment_slots: smallvec::SmallVec<[usize; MAX_COLOR_ATTACHMENTS + 1]>,
    extent: (u32, u32, u32, u32),
    depth: (vk::ImageView, vk::Format, vk::Format),
    layout: LayoutEntry,
    pipeline: vk::Pipeline,
    #[allow(clippy::type_complexity)]
    _held: (Arc<super::frontend::StageProgram>, Arc<ps5shader::resources::BindingLayout>, Option<(Arc<super::frontend::StageProgram>, Arc<ps5shader::resources::BindingLayout>)>),
}

/// Recipes by key hash, most recent first ([`RECIPE_BUCKET`] per hash;
/// one with `RPS5E_RECIPE_KEY=0`).
#[derive(Default)]
struct RecipeCache {
    map: crate::fxhash::HashMap<u64, smallvec::SmallVec<[Arc<DrawRecipe>; 2]>>,
    len: usize,
}

/// Recipes kept at most (the cache starts over beyond).
const MAX_RECIPES: usize = 1 << 14;

impl RecipeCache {
    fn find(&self, key: &RecipeKey, vertex: &VertexInputs) -> Option<Arc<DrawRecipe>> {
        let same_vertex = |r: &DrawRecipe| {
            r.vertex.bindings.len() == vertex.buffers.len()
                && r.vertex.attributes.len() == vertex.attributes.len()
                && r.vertex.bindings.iter().zip(&vertex.buffers).all(|(k, b)| k.stride == b.stride && k.per_instance == b.per_instance)
                && r.vertex
                    .attributes
                    .iter()
                    .zip(&vertex.attributes)
                    .all(|(k, a)| k.location == a.location && k.binding == a.binding && k.format == a.format.as_raw() && k.offset == a.offset)
        };
        self.map.get(&key.hash())?.iter().find(|r| same_vertex(r) && r.key == *key).cloned()
    }

    fn insert(&mut self, recipe: DrawRecipe) {
        if self.len >= MAX_RECIPES {
            self.map.clear();
            self.len = 0;
        }
        let ways = if recipe.key.whole.is_some() { 1 } else { RECIPE_BUCKET };
        let bucket = self.map.entry(recipe.key.hash()).or_default();
        bucket.insert(0, Arc::new(recipe));
        if bucket.len() > ways {
            bucket.truncate(ways);
        } else {
            self.len += 1;
        }
    }
}

/// `RPS5E_DRAW_RECIPES=0`: every draw looks up its modules, views,
/// layout and pipeline.
fn recipes_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| !std::env::var("RPS5E_DRAW_RECIPES").is_ok_and(|v| v == "0"))
}

/// Guest ranges a draw touches: (address, size, written, bound through a
/// descriptor / vertex / index binding).
pub(crate) type RangeList = smallvec::SmallVec<[(u64, u64, bool, bool); 32]>;

/// A stage ready to bind.
struct BoundStage {
    stage: ShaderStage,
    prepared: PreparedStage,
}

enum Failure {
    NeedsShader,
    Skip,
    /// A module or pipeline the draw needs is being compiled on a worker;
    /// the draw (which writes only its attachments) is skipped.
    Compiling,
    /// The indirect draw cannot run as a host indirect draw: read its
    /// records on the CPU instead.
    NoGpuIndirect,
    /// The staging ring has no room left in the open tick: submit and
    /// record the draw again.
    StagingFull,
    Error(String),
    /// A slice thread's draw needs the exact state (R7b): it becomes a hole
    /// (the reason is in the draw context's speculative state).
    Bail,
    /// A module or pipeline the item must wait for is still compiling and
    /// the caller waits for it with the back-end unlocked
    /// ([`super::unlocked_wait`]): the item is given up and recorded again
    /// from the start, as after [`Failure::StagingFull`].
    Pending,
}

impl From<vk::Result> for Failure {
    fn from(e: vk::Result) -> Self {
        Failure::Error(format!("{e:?}"))
    }
}

const STAGING_BYTES: u64 = 128 << 20;
/// The staging ring of a slice thread's provider (R7b).
const SLICE_STAGING_BYTES: u64 = 32 << 20;

/// An indirect draw the GPU executes from the guest's argument records:
/// a host indirect draw (the record layouts are Vulkan's indirect command
/// layouts) or, for a mesh draw, a mesh launch the indirect-mesh kernel
/// makes from the record (see `indirect_mesh`).
#[derive(Clone, Copy, Debug)]
struct GpuIndirect {
    buffer: vk::Buffer,
    offset: u64,
    draw_count: u32,
    stride: u32,
    /// Draw count in memory (clamped to `draw_count` by the host).
    count: Option<(vk::Buffer, u64)>,
    /// User SGPRs (of the vertex program) receiving the base vertex, the
    /// first instance and the first index (0xFF: none).
    sgprs: [u8; 3],
    /// Guest ranges of the records and of the count.
    guest: [(u64, u64); 2],
    /// The draw may run as a host indirect draw (vertex path) / as a mesh
    /// launch from the record (mesh path).
    host: bool,
    mesh: bool,
}

/// `RPS5E_GPU_INDIRECT=0`: read every indirect draw's records on the CPU.
fn gpu_indirect_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_GPU_INDIRECT"))
}

/// `RPS5E_GPU_INDIRECT_MESH=0`: indirect mesh draws (geometry and
/// primitive subgroups) read their record on the CPU.
fn gpu_indirect_mesh_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_GPU_INDIRECT_MESH"))
}

/// `RPS5E_GPU_INDIRECT_RECT=0`: indirect rectangle lists read their
/// records on the CPU.
fn gpu_indirect_rect_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_GPU_INDIRECT_RECT"))
}

/// Whether a rectangle list expanded as `rect` draws the vertex count of
/// its argument record unchanged, so a host indirect draw can read the
/// record: tessellation patches and fill rectangles expand three vertices
/// per rectangle on the GPU; the compatibility modes choose by the count.
fn rect_mode_reads_record(rect: RectMode) -> bool {
    matches!(rect, RectMode::Tessellation | RectMode::FillRectangle) && gpu_indirect_rect_enabled()
}
const MAX_BUFFER_BINDING: u64 = 1 << 30;
/// VGT_SHADER_STAGES_EN.GS_EN: a geometry stage follows the ES.
pub(crate) const STAGES_GS_EN: u32 = 1 << 5;
/// VGT_SHADER_STAGES_EN.GS_W32_EN: the geometry stage runs wave32.
const STAGES_GS_W32_EN: u32 = 1 << 22;
/// VGT_SHADER_STAGES_EN.HS_EN, PRIMGEN_EN and PRIMGEN_PASSTHRU_EN.
const STAGES_HS_EN: u32 = 1 << 2;
const STAGES_PRIMGEN_EN: u32 = 1 << 13;
const STAGES_PRIMGEN_PASSTHRU_EN: u32 = 1 << 25;

/// Whether VGT_SHADER_STAGES_EN describes a primitive subgroup a vertex
/// program can run as a mesh workgroup: primitive generation without a
/// geometry stage, pass-through primitives or tessellation. A pass-through
/// subgroup reads its primitive from v0 as the hardware packs it (vertex
/// indices), not the ES vertex offsets the mesh entry computes.
/// `RPS5E_NGG_MESH_STAGES=0`: any merged program qualifies (the earlier
/// rule).
pub fn primitive_subgroup_stages(stages: u32) -> bool {
    static CHECK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let check = *CHECK.get_or_init(|| !std::env::var("RPS5E_NGG_MESH_STAGES").is_ok_and(|v| v == "0"));
    !check || stages & (STAGES_HS_EN | STAGES_GS_EN | STAGES_PRIMGEN_EN | STAGES_PRIMGEN_PASSTHRU_EN) == STAGES_PRIMGEN_EN
}
/// Push cursor for the mesh stage's shader data: past the block, so it
/// lives in the shader-data buffer, followed by the draw parameters.
const MESH_SHADER_DATA_CURSOR: u32 = 32;
/// Push cursor of a compute program run by an indirect dispatch with
/// thread dimensions: past the block, so its shader data (with the thread
/// limits the GPU writes) lives in the shader-data buffer.
const INDIRECT_THREADS_SHADER_DATA_CURSOR: u32 = ps5shader::resources::push_data::DWORD_COUNT;

/// `RPS5E_GPU_INDIRECT_THREADS=0`: indirect dispatches with thread
/// dimensions read their record on the CPU (waiting for the GPU work that
/// wrote it) and run as direct dispatches.
pub fn gpu_indirect_threads_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_GPU_INDIRECT_THREADS"))
}

/// An indirect dispatch whose record counts threads.
fn indirect_thread_dimensions(d: &ResolvedDispatch) -> bool {
    d.packet.indirect_address != 0 && d.packet.dispatch_initiator & crate::render::DISPATCH_USE_THREAD_DIMENSIONS != 0
}

/// How a program run by an indirect dispatch is laid out: the push cursor
/// and thread limits it is prepared with. An indirect dispatch with thread
/// dimensions always reads limits (the GPU writes them; the record may
/// make any group partial) from the shader-data buffer; the placeholder
/// limits are replaced on the GPU before the dispatch runs. Every other
/// dispatch keeps push data, with limits only when its last groups are
/// partial.
pub(crate) fn dispatch_layout(d: &ResolvedDispatch) -> (u32, Option<[u32; 3]>) {
    if indirect_thread_dimensions(d) {
        (INDIRECT_THREADS_SHADER_DATA_CURSOR, Some([0; 3]))
    } else {
        (0, d.thread_limits())
    }
}

pub struct ShaderPipelines {
    /// Pipelines, modules, layouts, samplers, the compile pool and shared
    /// buffers: shared by the recording threads (R7b stage 2; written by
    /// the committing thread between waves only).
    pub(crate) t: Arc<PipelineTables>,
    /// `RPS5E_SLICE_VERIFY`: the digest of the last recorded draw's
    /// decisions (see [`crate::backend::PipelineProvider::take_digest`]).
    digest: u64,
    /// Why the last speculative draw failed with [`Failure::Bail`] (when
    /// its draw context has no reason).
    bail_why: Option<crate::slices::spec::Hole>,
    /// The tables' pipeline swaps when the recipes were last valid.
    swaps_seen: u64,
    /// A slice thread's instance ([`Self::slice_instance`]): it records
    /// against frozen committed state and never writes the tables; a
    /// missing module, pipeline, layout or sampler fails the draw with
    /// [`Failure::Bail`] (the committing thread records it).
    speculative: bool,
    /// The compile lookahead (`lookahead`): the page tracker the stages of
    /// upcoming dispatches are prepared against, the command position and
    /// finished-job count of the last scan, the dispatches handled at that
    /// position, and whether lookups only queue compiles (never wait).
    ahead: Ahead,
    dev: Arc<GpuDevice>,
    memory: &'static GuestGpuMemory,
    /// The shader front end of this provider (program decode and stage
    /// materialisation), shared with the interpreter thread's prefetch
    ///.
    pub(crate) frontend: Arc<std::sync::Mutex<super::frontend::ShaderFrontend>>,
    compiler: Option<Arc<dyn ShaderCompiler>>,
    plan_only: bool,
    heap: DescriptorHeap,
    /// The bindless set of this provider (made when a stage first selects
    /// images through it; None inside when the device has none). A slice
    /// instance has none: its draws using one are holes.
    bindless: Option<Option<super::bindless::BindlessSet>>,
    staging: StagingRing,
    vertex_inputs: Option<Box<dyn VertexInputResolver>>,
    subgroup32_compute: bool,
    /// Compute subgroups may run narrower than 32 lanes unless required.
    narrow_compute: bool,
    max_subgroup_invocations: u32,
    pub stats: ProviderStats,
    /// Frame trace: notes on the current pass and the images it writes.
    trace_notes: Vec<String>,
    /// Primitive-subgroup programs already reported as mesh workgroups.
    subgroup_programs: crate::fxhash::HashSet<u64>,
    /// The indirect draw being recorded as a host indirect draw (set
    /// around `try_draw`).
    gpu_indirect: Option<GpuIndirect>,
    /// Vertex programs (address, plan, offset SGPRs, mesh entry) whose
    /// resources do not depend on the user SGPRs an indirect draw writes
    /// (offset registers): true when they may run from the GPU records.
    indirect_independent: HashMap<(u64, u64, [u8; 3], bool), bool>,
    /// The stage whose shader data is patched on the GPU (the vertex stage
    /// of a mesh indirect draw, the compute stage of an indirect dispatch
    /// with thread dimensions): it gets a fresh upload, reported in
    /// `patched_shader_data`.
    patch_shader_data: Option<ShaderStage>,
    patched_shader_data: Option<vk::DescriptorBufferInfo>,
    /// Readable mappings `clamp_range` found, most recent first, at a
    /// (mapping, range registry) generation.
    clamp_memo: std::cell::RefCell<((u64, u64, u64), ReadableRuns)>,
    /// Clamps that walked the mapping registry (memo misses).
    clamp_misses: std::cell::Cell<u64>,
    /// The last clamped requests, exact (address, size) → result, under
    /// the generation of `clamp_memo`: the ranges of a draw are clamped
    /// once for the reservation and again when bound.
    clamp_exact: std::cell::RefCell<[(u64, u64, Option<(u64, u64)>); 32]>,
    /// Stage layouts extended for the mesh entry point, by the plain
    /// layout's resource key: (layout, resource key).
    mesh_layouts: HashMap<u64, (Arc<ps5shader::resources::BindingLayout>, u64)>,
    /// Pixel layouts with the per-pixel sample masks added, by the plain
    /// layout's resource key (see `with_quad_masks`).
    quad_layouts: HashMap<u64, (Arc<ps5shader::resources::BindingLayout>, u64)>,
    trace_written: Vec<u64>,
    /// Frame trace: tiny sampled images (constant textures) whose texels are logged.
    trace_sampled: Vec<u64>,
    /// `RPS5E_FRAME_TRACE_LAST=1`: dumps held back until a pass no longer
    /// uses the image (image, pass, kind), so a run of passes on the same
    /// targets dumps them once, after the last of them.
    trace_pending: Vec<(u64, u64, String)>,
    /// Ranges the last dispatch wrote through buffers it does not read.
    overwrites: Vec<(u64, u64)>,
    /// DCC keys the work recorded since the back-end last asked stored
    /// into ([`compressed_store_keys`]).
    compressed_stores: Vec<crate::render::StoredKeys>,
    /// Texture lookups at `view_memo_epoch` of the image cache: (view,
    /// image, tick of the last use, the image's lookup generation).
    view_memo: HashMap<ViewMemoKey, (vk::ImageView, u64, u64, u64)>,
    view_memo_epoch: u64,
    /// What the previous draw of the open rendering scope left bound.
    scope_state: super::recording::ScopeState,
    /// Draws of the current frame that were not recorded
    /// (`RPS5E_DRAW_SKIP_LOG=1`), and the frames presented so far.
    skip_log: super::skiplog::SkipLog,
    /// The check that refused the last draw with `Failure::Skip` (for the
    /// skip log).
    skip_why: &'static str,
    /// What is left of this frame's budget for waiting on compiles before
    /// a draw is skipped ([`compile_wait_budget`]).
    compile_wait_left: std::time::Duration,
    /// The per-frame budget ([`compile_wait_budget`]).
    compile_wait: std::time::Duration,
    /// The last item was given up for a pending compile
    /// ([`Failure::Pending`]; taken by the back-end).
    compile_pending: bool,
    presented_frames: u64,
    /// Ranges of the current draw bound from copies of their guest bytes.
    /// The ranges the open draw's bindings stream (GPU_PLAN §52: the
    /// flags parallel to the draw's ranges; the tuple list serves the
    /// binding loop's rare un-resolved lookups).
    streams: Vec<(u64, u64)>,
    stream_flags: smallvec::SmallVec<[bool; 32]>,
    /// The buffers of the draw's vertex and pixel stages as its reservation
    /// resolved them, per resource: the clamped range and whether it is
    /// bound from a copy (None inside: a null descriptor). Set while the
    /// draw binds them.
    #[allow(clippy::type_complexity)]
    resolved: [Option<smallvec::SmallVec<[Option<(u64, u64, bool)>; 16]>>; 2],
    /// The current draw or dispatch writes buffer ranges.
    draw_writes_ranges: bool,
    /// Binding results of the recent vertex, pixel and compute stages,
    /// most recent first, one per binding layout ([`STAGE_MEMOS`]).
    stage_memos: [Vec<Box<StageMemo>>; 3],
    /// Draw recipes (see [`DrawRecipe`]).
    recipes: RecipeCache,
    /// Copies of guest ranges in the staging ring made in the open tick,
    /// reused while the bytes a new copy would read are the same (the
    /// pages read are unchanged): constant buffers many draws bind
    /// (per-view, per-pass constants) are copied once.
    stream_memo: HashMap<(u64, u64), (vk::DescriptorBufferInfo, crate::memory::deps::PageDeps, u64)>,
    /// (tick, content stamp or zeros) of `stream_memo`.
    stream_stamp: (u64, [u64; 4]),
    /// The tick whose copies `stream_memo` holds when copies do not carry
    /// into later ticks.
    stream_memo_tick: u64,
    /// Descriptor writes of the current draw or dispatch (reused).
    writes: SetWrites,
    /// Guest ranges of the dispatch being recorded (reused buffer).
    touched_ranges: Vec<(u64, u64)>,
    /// The bytes the dispatch being recorded stores to through written
    /// buffers whose stores are bounded: (buffer, guest address, bytes).
    write_limits: smallvec::SmallVec<[(u32, u64, u64); 4]>,
    /// Store bounds per compute program (see `pipeline::write_extent`).
    store_bounds: crate::fxhash::HashMap<(usize, super::write_extent::SystemSgprs, bool), Arc<ProgramStores>>,
    /// Image barriers of the current draw or dispatch (reused).
    image_barriers: Vec<vk::ImageMemoryBarrier2<'static>>,
    /// Draw pipeline layouts by the stages' binding layouts (held).
    #[allow(clippy::type_complexity)]
    layout_memo: HashMap<(usize, usize, bool, u32), (Arc<ps5shader::resources::BindingLayout>, Option<Arc<ps5shader::resources::BindingLayout>>, LayoutEntry)>,
}

/// The part of [`ShaderPipelines`] every recording thread shares (R7b
/// stage 2): read by slice threads while a wave records, written by the
/// committing thread only (see [`crate::slices::frozen`]).
pub(crate) struct PipelineTables {
    /// Graphics pipelines swapped for their optimised builds so far: every
    /// instance drops its recipes (which hold pipelines) when it moved.
    swaps: Shared<u64>,
    modules: Shared<HashMap<u64, Result<Arc<Module>, CompileError>>>,
    /// Module keys by the plain inputs (see `module`).
    module_keys: Shared<HashMap<u64, Vec<(ModuleInputs, Box<[u32]>, u64)>>>,
    rect_modules: Shared<HashMap<u64, Option<(vk::ShaderModule, vk::ShaderModule)>>>,
    layouts: Shared<LayoutCache>,
    graphics: Shared<HashMap<GraphicsKey, vk::Pipeline>>,
    compute: Shared<HashMap<ComputeKey, vk::Pipeline>>,
    samplers: Shared<SamplerCache>,
    dummies: Shared<DummyResources>,
    pub(crate) disk: Shared<PipelineDiskCache>,
    gds: Shared<Buffer>,
    fault: Shared<Buffer>,
    null_buffer: Shared<Buffer>,
    /// GDS and the null buffer were zeroed on the device.
    zeroed: Shared<bool>,
    /// The kernel turning a mesh draw's record into its launch (None: no
    /// mesh shaders, or it failed to build).
    indirect_mesh: Shared<Option<IndirectMeshKernel>>,
    /// The kernel turning the thread counts of an indirect dispatch with
    /// thread dimensions into its groups and thread limits (None: it failed
    /// to build).
    indirect_threads: Shared<Option<IndirectThreadsKernel>>,
    /// Compile workers (None: compile on this thread).
    pool: Shared<Option<super::pool::CompilePool<Compiled>>>,
    pending_modules: Shared<crate::fxhash::HashSet<u64>>,
    /// Relocatable module keys queued, with the address of the program
    /// being translated; and those whose translation proved address
    /// dependent (this run).
    relocating: Shared<HashMap<u64, u64>>,
    not_relocatable: Shared<crate::fxhash::HashSet<u64>>,
    pending_graphics: Shared<crate::fxhash::HashSet<GraphicsKey>>,
    pending_compute: Shared<crate::fxhash::HashSet<ComputeKey>>,
    /// Build and wait times (`create=` in the summary).
    compile_times: Shared<super::timing::CompileTimes>,
    /// Fast pipelines replaced by their optimised build, not yet handed
    /// to the scheduler ([`Self::retire_swapped`]).
    swapped_out: Shared<Vec<vk::Pipeline>>,
    /// Pipelines missing from the cache are built fast first
    /// ([`fast_pipelines_mode`], with compile workers and cache probes).
    fast_pipelines: Shared<bool>,
    /// Pipelines whose creation failed (not tried again).
    failed_graphics: Shared<crate::fxhash::HashSet<GraphicsKey>>,
    failed_compute: Shared<crate::fxhash::HashSet<ComputeKey>>,
    /// The prewarm list (loaded at the first draw or dispatch).
    prewarm: Shared<super::prewarm::PrewarmList>,
    prewarm_loaded: Shared<bool>,
    /// Modules of the list being translated again, and the pipelines of
    /// the list queued once they are done.
    prewarm_translating: Shared<crate::fxhash::HashSet<u64>>,
    prewarm_deferred: Shared<Vec<PrewarmPipeline>>,
    prewarm_counts: Shared<PrewarmCounts>,
    cache_path: Shared<Option<std::path::PathBuf>>,
    /// Set-layout bindings and push stages of each pipeline layout made.
    layout_descs: Shared<HashMap<vk::PipelineLayout, (Vec<super::descriptors::BindingDesc>, u32)>>,
}

/// `RPS5E_STATE_FILTER=0`: every draw records its whole state.
fn redundant_state_filter() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::fast() && *ON.get_or_init(|| !std::env::var("RPS5E_STATE_FILTER").is_ok_and(|v| v == "0"))
}

/// A T# lookup (the words and how the binding views them).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ViewMemoKey {
    words: [u32; 8],
    numeric: ImageNumericClass,
    dimension: ImageDimension,
    storage: bool,
    cube: bool,
    host_swizzle: bool,
    mip_offset: u32,
    single_level: bool,
    atomic64: bool,
    srgb: bool,
}

/// `RPS5E_DEPTH_READ_ONLY=0`: a draw sampling its depth attachment runs
/// in a scope of its own with the attachment in GENERAL.
fn depth_read_only_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::fast() && *ON.get_or_init(|| !std::env::var("RPS5E_DEPTH_READ_ONLY").is_ok_and(|v| v == "0"))
}

/// `RPS5E_DEPTH_STENCIL_SPLIT=0`: a draw writing only the stencil of a
/// depth/stencil attachment it samples the depth of runs in a scope of its
/// own with the attachment in GENERAL (full barriers on either side).
fn depth_split_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_DEPTH_STENCIL_SPLIT"))
}

/// `RPS5E_PRECISE_BARRIERS=0`: a full barrier before and after every
/// dispatch.
fn precise_barriers() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::fast() && *ON.get_or_init(|| !std::env::var("RPS5E_PRECISE_BARRIERS").is_ok_and(|v| v == "0"))
}

/// `RPS5E_WRITE_EXTENT=0`: a buffer a dispatch writes is GPU-dirty over
/// its whole descriptor range. Otherwise, when the program's stores are
/// bounded by the dispatch's thread and group ids (`pipeline::write_extent`), only
/// the bytes they can reach.
fn write_extent_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_WRITE_EXTENT"))
}

/// A compute program's store bounds and the pcs of the stores (and
/// atomics) of each written buffer.
pub(crate) struct ProgramStores {
    /// Keeps the program (the cache key is its address) alive.
    _program: Arc<super::frontend::StageProgram>,
    bounds: Option<super::write_extent::StoreBounds>,
    buffers: Vec<(u32, Vec<u32>)>,
}

/// `RPS5E_UNCHECKED_EXTENT=0`: a stride-0 V# without range checking
/// (OOB_SELECT 2) is bound for its NUM_RECORDS bytes only, as before, even
/// where its accesses read past them.
fn unchecked_extent_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_UNCHECKED_EXTENT").is_ok_and(|v| v == "0"))
}

/// The DCC keys a store through the image descriptor `words` updates: the
/// keys of every layer of the stored image `key`, when the descriptor has
/// DCC compression on (T# word 6 COMPRESSION_EN, the keys at
/// META_DATA_ADDRESS) and the surface's keys are modelled
/// ([`crate::render::dcc_slice_bytes`]). The hardware rewrites the keys of
/// the blocks a store writes, so no fast-clear code survives there.
pub(crate) fn compressed_store_keys(words: &[u32], key: Option<&crate::images::ImageKey>) -> Option<crate::render::StoredKeys> {
    let (w6, w7) = (*words.get(6)?, *words.get(7)?);
    if (w6 >> 21) & 1 == 0 {
        return None;
    }
    let address = (((w6 >> 24) as u64) | ((w7 as u64) << 8)) << 8;
    let key = key?;
    let slice = crate::render::dcc_slice_bytes(key.width, key.height, key.main.bytes_per_element, key.main.swizzle, key.samples, key.levels);
    (address != 0 && slice != 0).then(|| crate::render::StoredKeys { address, slice, layers: key.slices().max(1) })
}

/// The guest bytes a bound buffer covers (see [`BufferSharp::extent`]).
fn buffer_extent(v: &BufferSharp, resource: Option<&ps5shader::resources::BufferResource>) -> u64 {
    if unchecked_extent_enabled() {
        v.extent(resource.map_or(0, |b| b.max_byte_extent))
    } else {
        v.footprint()
    }
}

/// Readable runs of tracked mappings the clamps found, `(start, end)`
/// sorted by start (each run starts at a mapping and goes on through up to
/// 64 contiguous readable ones), valid at one registry generation. The
/// constant rings and resources of a frame live in a hundred or more
/// mappings: a short most-recent list missed on every few clamps.
#[derive(Default)]
pub(crate) struct ReadableRuns {
    runs: Vec<(u64, u64)>,
    /// The run found last (asked first).
    last: usize,
}

/// Runs kept at most (a full memo starts over).
const MAX_READABLE_RUNS: usize = 4096;

impl ReadableRuns {
    pub(crate) fn clear(&mut self) {
        self.runs.clear();
        self.last = 0;
    }

    /// The end of a run holding `address`. A run starting earlier that
    /// holds it passes every mapping the run starting last before it
    /// does: that one holds it as well (and reaches at least as far).
    pub(crate) fn end_of(&mut self, address: u64) -> Option<u64> {
        if let Some(&(s, e)) = self.runs.get(self.last) {
            if s <= address && address < e {
                return Some(e);
            }
        }
        let i = self.runs.partition_point(|r| r.0 <= address).checked_sub(1)?;
        let (_, e) = self.runs[i];
        (address < e).then(|| {
            self.last = i;
            e
        })
    }

    /// Remember the run `[start, end)`.
    pub(crate) fn insert(&mut self, start: u64, end: u64) {
        if self.runs.len() >= MAX_READABLE_RUNS {
            self.clear();
        }
        let i = self.runs.partition_point(|r| r.0 < start);
        match self.runs.get_mut(i) {
            Some(r) if r.0 == start => r.1 = r.1.max(end),
            _ => self.runs.insert(i, (start, end)),
        }
        self.last = i;
    }
}

/// The per-thread caches of `clamp_range_shared` (the provider's
/// `clamp_memo`/`clamp_exact` pair; the interpreter thread's prefetch
/// holds its own).
#[derive(Default)]
pub(crate) struct ClampCaches {
    generation: (u64, u64, u64),
    runs: ReadableRuns,
    exact: [(u64, u64, Option<(u64, u64)>); 32],
}

/// `size` limited to the largest storage-buffer binding the device takes.
pub(crate) fn binding_limit_of(size: u64, max_storage_buffer_range: u32) -> u64 {
    size.min((max_storage_buffer_range as u64).saturating_sub(256).clamp(4096, MAX_BUFFER_BINDING))
}

/// DISPATCH_INITIATOR.ORDERED_APPEND_ENABLE.
const DISPATCH_ORDERED_APPEND_ENABLE: u32 = 1 << 3;

/// Groups launched one by one at most (a larger ordered dispatch runs as
/// one dispatch, unordered).
const MAX_ORDERED_GROUPS: u64 = 1 << 16;

/// An ordered-append dispatch (ORDERED_APPEND_ENABLE, GDS used) whose
/// groups are one guest wave each and use no LDS runs as one host dispatch
/// per group in group order, each after the previous one's memory writes,
/// as the GPU orders the waves' DS_ORDERED_COUNT appends.
/// `RPS5E_ORDERED_APPEND=0` dispatches all groups at once.
pub(crate) fn ordered_append_serial(initiator: u32, gds: bool, invocations: u32, wave: u32, lds_dwords: u32, groups: [u32; 3]) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let total = groups.iter().map(|&n| n as u64).product::<u64>();
    initiator & DISPATCH_ORDERED_APPEND_ENABLE != 0
        && gds
        && invocations.max(1) <= wave
        && lds_dwords == 0
        && (1..=MAX_ORDERED_GROUPS).contains(&total)
        && *ON.get_or_init(|| !std::env::var("RPS5E_ORDERED_APPEND").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
}

#[cfg(test)]
mod ordered_append_tests {
    use super::ordered_append_serial;

    /// Only one-wave, LDS-free GDS programs dispatched with
    /// ORDERED_APPEND_ENABLE are launched group by group.
    #[test]
    fn ordered_append_runs_group_by_group() {
        let on = 1 << 3;
        assert!(ordered_append_serial(on, true, 64, 64, 0, [4, 2, 1]));
        assert!(ordered_append_serial(on | 1, true, 32, 32, 0, [1, 1, 1]));
        assert!(!ordered_append_serial(0, true, 64, 64, 0, [4, 1, 1]), "not ordered");
        assert!(!ordered_append_serial(on, false, 64, 64, 0, [4, 1, 1]), "no GDS");
        assert!(!ordered_append_serial(on, true, 128, 64, 0, [4, 1, 1]), "two waves per group");
        assert!(!ordered_append_serial(on, true, 64, 64, 128, [4, 1, 1]), "LDS");
        assert!(!ordered_append_serial(on, true, 64, 64, 0, [0, 1, 1]), "no groups");
        assert!(!ordered_append_serial(on, true, 64, 64, 0, [1 << 16, 2, 1]), "too many groups");
    }
}

/// `RPS5E_PRT_HOLE_RANGES=0`: a finite buffer range is cut at the first
/// unmapped page, as before sparse ranges kept their committed tail.
fn prt_hole_ranges_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_PRT_HOLE_RANGES").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
}

/// A finite buffer (under 1 GiB) of a partially resident range keeps the
/// committed pages past an uncommitted hole: the clamp `r` (cut at the
/// first unreadable page) grows to the end of the last committed piece
/// within the bound size (`limited`). The holes read as zeros
/// ([`GuestGpuMemory::read_committed`]); oversized and unbounded
/// descriptors keep the plain clamp.
pub(crate) fn extend_over_holes(memory: &GuestGpuMemory, address: u64, size: u64, limited: u64, r: Option<(u64, u64)>) -> Option<(u64, u64)> {
    let Some((a, n)) = r else { return r };
    let full = size.min(limited);
    if size >= 1 << 30 || n >= full || !prt_hole_ranges_enabled() {
        return r;
    }
    match memory.committed_extent(address, full) {
        (runs, end) if runs > 1 && end.saturating_sub(address) > n => Some((a, end - address)),
        _ => r,
    }
}

/// The clamp of one range over shared caches: the body of
/// `ShaderPipelines::clamp_range` without the provider's statistics.
pub(crate) fn clamp_range_shared(memory: &GuestGpuMemory, caches: &mut ClampCaches, misses: Option<&std::cell::Cell<u64>>, address: u64, size: u64, limit: u64) -> Option<(u64, u64)> {
    if address == 0 || size == 0 {
        return None;
    }
    let generation = (memory.unmap_generation(), ps5vm::ranges::shrink_generation(), ps5vm::ranges::generation());
    if caches.generation != generation {
        caches.generation = generation;
        caches.runs.clear();
        caches.exact = [(0, 0, None); 32];
    }
    let exact = crate::backend::ab::new_fast();
    let slot = ((address >> 4) ^ size).wrapping_mul(0x9E37_79B9_7F4A_7C15) as usize >> 59;
    if exact && caches.exact[slot].0 == address && caches.exact[slot].1 == size {
        return caches.exact[slot].2;
    }
    let r = (|| {
        if let Some(e) = caches.runs.end_of(address) {
            let mut n = limit;
            while address.saturating_add(n) > e {
                if n <= 256 {
                    return (address + 4 <= e).then_some((address, 4));
                }
                n /= 2;
            }
            return Some((address, n));
        }
        if let Some(m) = misses {
            m.set(m.get() + 1);
        }
        let mut n = limit;
        let readable = |a: u64, s: u64| memory.backing_ptr(a, s).is_some() || ps5vm::ranges::accessible(a, s, false);
        while !readable(address, n) {
            if n <= 256 {
                return readable(address, 4).then_some((address, 4));
            }
            n /= 2;
        }
        let r = Some((address, n));
        if let Some((s, mut e, prot)) = ps5vm::ranges::containing(address).filter(|r| r.2.read) {
            for _ in 0..64 {
                match ps5vm::ranges::containing(e) {
                    Some((_, next, p)) if p.read => e = next,
                    _ => break,
                }
            }
            let _ = prot;
            caches.runs.insert(s, e);
        }
        r
    })();
    let r = extend_over_holes(memory, address, size, size.min(limit), r);
    if exact {
        caches.exact[slot] = (address, size, r);
    }
    r
}

/// The stage-range walk over shared caches: the body of
/// `ShaderPipelines::stage_ranges_mapped` (which keeps the provider's
/// statistics around it).
pub(crate) fn stage_ranges_shared(
    memory: &GuestGpuMemory,
    caches: &mut ClampCaches,
    misses: Option<&std::cell::Cell<u64>>,
    limit: u64,
    s: &PreparedStage,
    out: &mut RangeList,
    mut map: Option<&mut smallvec::SmallVec<[u32; 16]>>,
) {
    let info = &s.resources.info;
    if info.buffers_in_guest_memory {
        return;
    }
    for (i, words) in s.snapshot.buffers.iter().enumerate() {
        let v = BufferSharp::from_words(words);
        let written = info.buffers.get(i).is_some_and(|b| b.written || b.atomic);
        let index = match clamp_range_shared(memory, caches, misses, v.address(), buffer_extent(&v, info.buffers.get(i)), limit) {
            Some((a, n)) => {
                out.push((a, n, written, true));
                out.len() as u32 - 1
            }
            None => u32::MAX,
        };
        if let Some(m) = map.as_deref_mut() {
            m.push(index);
        }
    }
    for r in &s.snapshot.device_address_ranges {
        if r.planned && r.size != 0 {
            if let Some((a, n)) = clamp_range_shared(memory, caches, misses, r.base, r.size, limit) {
                out.push((a, n, r.written, false));
            }
        }
    }
}

/// `RPS5E_STREAM_TRIM=0`: stream copies cover the whole bound range, not
/// only the bytes a buffer read at constant offsets reaches.
fn stream_trim_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| !std::env::var("RPS5E_STREAM_TRIM").is_ok_and(|v| v == "0"))
}

/// `RPS5E_RESOLVED_RANGES=0`: a draw's buffer bindings clamp their V#
/// ranges and look up their stream decisions again instead of taking what
/// the reservation resolved.
fn resolved_ranges_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_RESOLVED_RANGES"))
}

/// `RPS5E_RING_BINDING=1` (needs `RPS5E_WIDE_OFFSETS=1`): a read-only
/// buffer binding pins its whole containing buffer and carries the rotating
/// V# base in the wide offset dword — a constant ring that
/// hands out a new address every draw keeps hitting the binding memo
/// (a data-word split). Null, streamed and oversized ranges keep the
/// exact binding and identity of today.
fn ring_binding_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| std::env::var("RPS5E_RING_BINDING").is_ok_and(|v| v == "1"))
}

/// `RPS5E_RING_DEBUG`: count why buffer bindings are not ring bindings
/// (diagnostic).
fn ring_debug_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("RPS5E_RING_DEBUG").is_some())
}

/// `RPS5E_TABLE_CARRY=1`: a flattened-resource-table staging copy carries
/// across ticks while its ring bytes stay reserved: the table
/// is re-uploaded only when its words change or the ring recycles it, not on
/// every tick (eager write-throughs multiply ticks).
fn table_carry_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| std::env::var("RPS5E_TABLE_CARRY").is_ok_and(|v| v == "1"))
}

/// `RPS5E_STREAM_PAGES=0`: stream copies are reused only while the
/// buffer cache's content stamp is unchanged, not while their own pages
/// are.
pub(crate) fn stream_pages_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::new_fast() && *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_STREAM_PAGES"))
}

/// `RPS5E_STREAM_MEMO=0`: every streamed binding copies its bytes again.
fn stream_memo_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::backend::ab::fast() && *ON.get_or_init(|| !std::env::var("RPS5E_STREAM_MEMO").is_ok_and(|v| v == "0"))
}

/// `RPS5E_STREAM_CARRY=1`: staging copies of ranges whose guest pages
/// stayed unchanged are bound again by later ticks (see
/// [`Self::stream_copy`]).
fn stream_carry_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::backend::ab::safe() && std::env::var("RPS5E_STREAM_CARRY").is_ok_and(|v| v == "1"))
}

/// `RPS5E_VIEW_MEMO=0`: every texture binding looks its image up again.
fn view_memo_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_VIEW_MEMO"))
}


/// Vertex programs outside geometry subgroups translate with their guest
/// wave size (VGT_SHADER_STAGES_EN.GS_W32_EN; a wave64 program on a
/// 32-wide host keeps 64-bit lane masks with an empty upper half) unless
/// `RPS5E_VERTEX_GUEST_WAVE=0` translates every one as wave32 (a wave64
/// program's VOPC then leaves the high mask word unwritten).
fn vertex_guest_wave_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_VERTEX_GUEST_WAVE").is_ok_and(|v| v == "0"))
}

/// Primitive subgroups without a geometry stage that use LDS run as mesh
/// workgroups unless `RPS5E_NGG_MESH=0`.
fn primitive_subgroups_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_NGG_MESH").is_ok_and(|v| v == "0"))
}

/// The vertex stage option words compared in full when a draw looks up
/// its module: LDS, PA_CL_VS_OUT_CNTL, the clip-space transform (7), the
/// indirect SGPRs, UCP_ENA and the 24 user clip plane words.
const VS_RAW_WORDS: usize = 35;

/// The LDS dwords a pixel program may address: EXTRA_LDS_SIZE
/// (SPI_SHADER_PGM_RSRC2_PS bits 8..15) in 128-dword units, at least one
/// (hardware reads zero and drops writes past it).
pub(crate) fn pixel_lds_dwords(rsrc2: u32) -> u32 {
    (((rsrc2 >> 8) & 0xFF) * 128).max(1)
}

/// `RPS5E_PIXEL_LDS_SIZE=1` (opt-in): a pixel program's per-invocation
/// LDS array is sized by EXTRA_LDS_SIZE, not by its accesses alone (up to
/// 64 KiB). Off by default: the pixel LDS device tests (`pixel_lds`) draw
/// with EXTRA_LDS_SIZE 0 and address 3 KiB.
fn pixel_lds_size_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_PIXEL_LDS_SIZE").is_ok_and(|v| v == "1"))
}

/// `RPS5E_USER_CLIP_PLANES=0`: PA_CL_CLIP_CNTL.UCP_ENA planes are not
/// applied (no clip distances are written for them).
fn user_clip_planes_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_USER_CLIP_PLANES").is_ok_and(|v| v == "0"))
}

/// Alpha-to-coverage draws turn the MRT0 alpha into the sample mask
/// unless `RPS5E_ALPHA_TO_COVERAGE=0`.
fn alpha_to_coverage_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_ALPHA_TO_COVERAGE").is_ok_and(|v| v == "0"))
}

/// Indexed strip mesh draws with primitive restart get a restart table
/// unless `RPS5E_MESH_STRIP_RESTART=0`.
fn mesh_strip_restart_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_MESH_STRIP_RESTART").is_ok_and(|v| v == "0"))
}

/// `RPS5E_SHARED_SCOPES=0`: every draw in a rendering scope of its own.
fn shared_rendering_scopes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_SHARED_SCOPES"))
}

fn plan_only_env() -> bool {
    std::env::var("RPS5E_GPU_PLAN").is_ok_and(|v| v == "1")
}

impl ShaderPipelines {
    /// GDS starts out zero (as after a GPU reset), and so do the reads of the
    /// null buffer bound for missing resources.
    fn zero_buffers(&mut self, sched: &mut crate::scheduler::Scheduler) {
        if *self.t.zeroed {
            return;
        }
        *self.t.zeroed.w() = true;
        let cb = sched.cmd();
        // SAFETY: recording buffer; both buffers are owned by the provider.
        unsafe {
            self.dev.device.cmd_fill_buffer(cb, self.t.gds.buffer, 0, vk::WHOLE_SIZE, 0);
            self.dev.device.cmd_fill_buffer(cb, self.t.null_buffer.buffer, 0, vk::WHOLE_SIZE, 0);
        }
        sched.full_barrier();
    }

    /// A provider on `dev` reading guest memory through `memory`, with the
    /// on-disk pipeline cache at `cache_path` (None: in memory only).
    pub fn new(
        dev: &Arc<GpuDevice>,
        memory: &'static GuestGpuMemory,
        compiler: Option<Arc<dyn ShaderCompiler>>,
        cache_path: Option<std::path::PathBuf>,
    ) -> Result<ShaderPipelines, vk::Result> {
        Self::with_workers(dev, memory, compiler, cache_path, None)
    }

    /// [`Self::new`] with `workers` compile workers (None: the default
    /// share of the cores, and none with `RPS5E_ASYNC_PIPELINES=0`).
    pub fn with_workers(
        dev: &Arc<GpuDevice>,
        memory: &'static GuestGpuMemory,
        compiler: Option<Arc<dyn ShaderCompiler>>,
        cache_path: Option<std::path::PathBuf>,
        workers: Option<usize>,
    ) -> Result<ShaderPipelines, vk::Result> {
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        // GDS is also the source and destination of command-processor
        // transfers (DMA_DATA, RELEASE_MEM / EOS read-backs).
        let gds = Buffer::new(dev, 64 << 10, storage | vk::BufferUsageFlags::TRANSFER_SRC, MemoryUse::Device)?;
        let fault = Buffer::new(dev, 4096, storage, MemoryUse::Download)?;
        fault.bytes_mut().fill(0);
        fault.flush(0, 4096);
        let null_buffer = Buffer::new(dev, 64 << 10, storage, MemoryUse::Device)?;
        let t = Arc::new(PipelineTables {
            swaps: Shared::new(0),
            modules: Shared::new(HashMap::default()),
            module_keys: Shared::new(HashMap::default()),
            rect_modules: Shared::new(HashMap::default()),
            layouts: Shared::new(LayoutCache::new(dev)),
            graphics: Shared::new(HashMap::default()),
            compute: Shared::new(HashMap::default()),
            samplers: Shared::new(SamplerCache::new(dev)),
            dummies: Shared::new(DummyResources::new(dev)),
            disk: Shared::new(PipelineDiskCache::new(dev, cache_path.clone())?),
            gds: Shared::new(gds),
            fault: Shared::new(fault),
            null_buffer: Shared::new(null_buffer),
            zeroed: Shared::new(false),
            indirect_mesh: Shared::new(if dev.caps.mesh_shader && dev.mesh.is_some() { IndirectMeshKernel::new(dev) } else { None }),
            indirect_threads: Shared::new(IndirectThreadsKernel::new(dev)),
            pool: Shared::new(match workers {
                Some(n) => Some(super::pool::CompilePool::new(n)),
                None => default_pool(),
            }),
            pending_modules: Shared::new(Default::default()),
            relocating: Shared::new(Default::default()),
            not_relocatable: Shared::new(Default::default()),
            pending_graphics: Shared::new(Default::default()),
            pending_compute: Shared::new(Default::default()),
            compile_times: Shared::new(Default::default()),
            swapped_out: Shared::new(Vec::new()),
            fast_pipelines: Shared::new(fast_pipelines_mode() != FastPipelines::Off && dev.caps.pipeline_creation_cache_control),
            failed_graphics: Shared::new(Default::default()),
            failed_compute: Shared::new(Default::default()),
            prewarm: Shared::new(Default::default()),
            prewarm_loaded: Shared::new(false),
            prewarm_translating: Shared::new(Default::default()),
            prewarm_deferred: Shared::new(Vec::new()),
            prewarm_counts: Shared::new(Default::default()),
            cache_path: Shared::new(cache_path),
            layout_descs: Shared::new(HashMap::default()),
        });
        let p = Self::with_tables(dev, memory, t, compiler, STAGING_BYTES, false)?;
        super::frontend::set_prefetch_memory(memory, binding_limit_of(u64::MAX, dev.caps.max_storage_buffer_range));
        Ok(p)
    }

    /// The per-thread part of a provider over shared tables `t`.
    fn with_tables(
        dev: &Arc<GpuDevice>,
        memory: &'static GuestGpuMemory,
        t: Arc<PipelineTables>,
        compiler: Option<Arc<dyn ShaderCompiler>>,
        staging_bytes: u64,
        speculative: bool,
    ) -> Result<ShaderPipelines, vk::Result> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDIRECT_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let mut sub = vk::PhysicalDeviceVulkan13Properties::default();
        let mut sub11 = vk::PhysicalDeviceVulkan11Properties::default();
        {
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sub).push_next(&mut sub11);
            // SAFETY: valid chain on the stack.
            unsafe { dev.instance.get_physical_device_properties2(dev.phys, &mut p2) };
        }
        let can_require32 = dev.caps.subgroup_size_control
            && sub.min_subgroup_size <= 32
            && sub.max_subgroup_size >= 32
            && sub.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE);
        // A device whose compute subgroups may also be narrower than its
        // default 32 (variable SIMD width, Intel) gets every compute pipeline
        // that does not require 64 pinned to 32 lanes: one guest wave32 (or
        // half of a wave64) per host subgroup.
        let narrow_compute = can_require32 && sub.min_subgroup_size < 32;
        let subgroup32_compute = can_require32 && (sub11.subgroup_size != 32 || narrow_compute);
        let frontend = Arc::new(std::sync::Mutex::new(super::frontend::ShaderFrontend::new(memory)));
        let swaps_seen = *t.swaps;
        Ok(ShaderPipelines {
            t,
            digest: 0,
            bail_why: None,
            swaps_seen,
            speculative,
            ahead: Ahead::default(),
            dev: dev.clone(),
            memory,
            frontend,
            compiler,
            plan_only: plan_only_env(),
            heap: DescriptorHeap::new(dev),
            staging: StagingRing::new(dev, staging_bytes, usage, MemoryUse::Upload)?,
            bindless: None,
            vertex_inputs: None,
            subgroup32_compute,
            narrow_compute,
            max_subgroup_invocations: sub.max_compute_workgroup_subgroups.max(1) * 32,
            stats: ProviderStats::default(),
            trace_notes: Vec::new(),
            subgroup_programs: Default::default(),
            gpu_indirect: None,
            indirect_independent: HashMap::default(),
            patch_shader_data: None,
            patched_shader_data: None,
            mesh_layouts: HashMap::default(),
            quad_layouts: HashMap::default(),
            clamp_memo: Default::default(),
            clamp_misses: Default::default(),
            clamp_exact: std::cell::RefCell::new([(0, 0, None); 32]),
            trace_written: Vec::new(),
            trace_sampled: Vec::new(),
            trace_pending: Vec::new(),
            overwrites: Vec::new(),
            compressed_stores: Vec::new(),
            view_memo: HashMap::default(),
            view_memo_epoch: u64::MAX,
            scope_state: Default::default(),
            skip_log: Default::default(),
            skip_why: "",
            compile_wait_left: compile_wait_budget(),
            compile_wait: compile_wait_budget(),
            compile_pending: false,
            presented_frames: 0,
            layout_memo: HashMap::default(),
            streams: Vec::new(),
            stream_flags: Default::default(),
            resolved: Default::default(),
            draw_writes_ranges: false,
            stage_memos: Default::default(),
            recipes: Default::default(),
            stream_memo: HashMap::default(),
            stream_stamp: (0, [0; 4]),
            stream_memo_tick: 0,
            writes: SetWrites::default(),
            touched_ranges: Vec::new(),
            write_limits: smallvec::SmallVec::new(),
            store_bounds: Default::default(),
            image_barriers: Vec::new(),

        })
    }

    /// A provider for a slice thread (R7b stage 2): its own front end,
    /// memos, recipes, staging ring and descriptor heap over this
    /// provider's tables. It records speculatively: a missing module,
    /// pipeline, layout, sampler or dummy fails the draw (a hole).
    pub fn slice_instance(&self) -> Result<ShaderPipelines, vk::Result> {
        Self::with_tables(&self.dev, self.memory, self.t.clone(), self.compiler.clone(), SLICE_STAGING_BYTES, true)
    }

    /// A slice instance's draw needs the exact state (`why`, reported by
    /// the draw when its context has no reason yet).
    /// GPU work about to write `[a, a+n)` through a buffer: the images the
    /// GPU changed there are written back first
    /// ([`crate::images::ImageCache::writeback_before_buffer_write`]). A
    /// slice thread's draw becomes a hole when one would be.
    fn writeback_before_buffer_write(&mut self, ctx: &mut DrawContext, a: u64, n: u64) -> Result<(), Failure> {
        match (ctx.images.exact(), ctx.buffers.exact()) {
            (Some(images), Some(buffers)) => {
                images.writeback_before_buffer_write(ctx.sched, buffers, a, n);
            }
            _ => {
                if ctx.images.cache().has_pending_writeback(a, n) {
                    return Err(self.bail(crate::slices::spec::Hole::Written));
                }
            }
        }
        Ok(())
    }

    fn bail(&mut self, why: crate::slices::spec::Hole) -> Failure {
        self.bail_why.get_or_insert(why);
        Failure::Bail
    }

    /// The tables a slice instance needs are set up (the committing
    /// thread's first draw zeroes the shared buffers and loads the
    /// prewarm list).
    pub fn slice_ready(&self) -> bool {
        *self.t.zeroed && *self.t.prewarm_loaded
    }

    /// Staging and descriptor sets handed out at the provisional tick
    /// `from` are read by commands of tick `to` (a committed segment).
    pub fn retag_tick(&mut self, from: u64, to: u64) {
        if from != to {
            self.staging.retag(from, to);
            self.heap.retag(from, to);
        }
    }

    pub fn set_compiler(&mut self, compiler: Option<Arc<dyn ShaderCompiler>>) {
        self.compiler = compiler;
    }

    /// How long the draws of a frame may wait for compiles before they are
    /// skipped (`Duration::MAX`: always wait).
    pub fn set_compile_wait(&mut self, budget: std::time::Duration) {
        self.compile_wait = budget;
        self.compile_wait_left = budget;
    }

    /// Compile modules and pipelines on workers (`true`, the default unless
    /// `RPS5E_ASYNC_PIPELINES=0`) or on the calling thread.
    pub fn set_async_compile(&mut self, on: bool) {
        if !on {
            if let Some(mut pool) = self.t.pool.w().take() {
                for c in pool.shutdown() {
                    self.finish(c);
                }
            }
        } else if self.t.pool.is_none() {
            *self.t.pool.w() = Some(super::pool::CompilePool::new(2));
        }
    }

    /// Pipelines the cache does not hold are built fast first (see
    /// [`fast_pipelines_mode`]); needs compile workers.
    fn fast_builds(&self) -> bool {
        *self.t.fast_pipelines && self.t.pool.is_some()
    }

    /// Build missing pipelines fast first (`true`) or optimised at once
    /// (tests; the default is `RPS5E_FAST_PIPELINES`).
    pub fn set_fast_pipelines(&mut self, on: bool) {
        *self.t.fast_pipelines.w() = on && self.dev.caps.pipeline_creation_cache_control;
    }

    pub fn set_vertex_input_resolver(&mut self, r: Option<Box<dyn VertexInputResolver>>) {
        self.vertex_inputs = r;
    }

    /// Module and pipeline build times so far.
    pub fn compile_times(&self) -> super::timing::CompileTimes {
        *self.t.compile_times
    }

    pub fn stats(&self) -> ProviderStats {
        ProviderStats {
            modules: self.t.modules.len(),
            graphics_pipelines: self.t.graphics.len(),
            compute_pipelines: self.t.compute.len(),
            ..self.stats
        }
    }

    // ------------------------------------------------------------------
    // Modules
    // ------------------------------------------------------------------

    fn create_module(&self, words: &[u32]) -> Result<vk::ShaderModule, vk::Result> {
        // SAFETY: valid SPIR-V words (the compiler's contract).
        unsafe { self.dev.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None) }
    }

    /// The module of one permutation (compiled on first use, on a compile
    /// worker when there are any: `wait` blocks until it is ready, else a
    /// missing module is [`Failure::Compiling`]). `raw`: the stage options
    /// as plain words; `options_key` hashes them into the permutation
    /// identity (only computed when the raw lookup misses).
    #[allow(clippy::too_many_arguments)]
    fn module(
        &mut self,
        compiler: &Arc<dyn ShaderCompiler>,
        stage: ShaderStage,
        prepared: &PreparedStage,
        options: Options,
        options_key: &dyn Fn() -> u64,
        raw: &[u32],
        mesh: Option<MeshConfig>,
        rsrc1: u32,
        wait: bool,
    ) -> Result<(u64, Arc<Module>), Failure> {
        crate::backend::profile::time(11, || self.module_inner(compiler, stage, prepared, options, options_key, raw, mesh, rsrc1, wait))
    }

    #[allow(clippy::too_many_arguments)]
    fn module_inner(
        &mut self,
        compiler: &Arc<dyn ShaderCompiler>,
        stage: ShaderStage,
        prepared: &PreparedStage,
        options: Options,
        options_key: &dyn Fn() -> u64,
        raw: &[u32],
        mesh: Option<MeshConfig>,
        rsrc1: u32,
        wait: bool,
    ) -> Result<(u64, Arc<Module>), Failure> {
        let p = &prepared.program;
        let translate_stage = match stage {
            ShaderStage::Compute => ps5shader::ssa::Stage::Compute,
            ShaderStage::Pixel => ps5shader::ssa::Stage::Pixel,
            _ => ps5shader::ssa::Stage::Vertex,
        };
        let float_mode = ps5shader::translate::FloatMode::from_rsrc1(rsrc1, ps5shader::translate::FloatMode::fp16_overflow_bit(translate_stage));
        // Draws repeat permutations: find the key by the plain inputs
        // (compared in full) before hashing them into the identity.
        let head = ModuleInputs { stage: stage as u8, float_mode, plan: p.plan.hash, code: p.code_hash, address: p.address, resource: prepared.resource_key, mesh };
        let memo_hash = {
            use std::hash::{Hash, Hasher};
            let mut h = rustc_hash::FxHasher::default();
            head.hash(&mut h);
            raw.hash(&mut h);
            h.finish()
        };
        let key = match self.t.module_keys.get(&memo_hash).and_then(|v| v.iter().find(|(h, r, _)| *h == head && r[..] == *raw)) {
            Some(&(_, _, key)) => key,
            None => {
                // Address-independent programs share modules wherever their
                // code lives (and across runs, through the on-disk cache).
                let address = if p.address_independent { 0 } else { p.address };
                let options = options_key();
                let key = hash_of(&(p.plan.hash, p.code_hash, address, prepared.resource_key, options, mesh, float_mode));
                // A program that reads its own address shares one module
                // wherever it is loaded once a relocatable translation of it
                // proved independent of the address.
                let key = match relocation_key(p, prepared, options, mesh, float_mode) {
                    Some(rkey) if self.relocation_serves(rkey, p.address) => rkey,
                    _ => key,
                };
                self.stats.module_key_misses += 1;
                if !self.speculative {
                    if self.t.module_keys.len() > 1 << 16 {
                        self.t.module_keys.w().clear();
                    }
                    self.t.module_keys.w().entry(memo_hash).or_default().push((head, raw.into(), key));
                }
                key
            }
        };
        if let Some(entry) = self.t.modules.get(&key) {
            self.stats.module_hits += 1;
            return match entry {
                Ok(m) => Ok((key, m.clone())),
                Err(_) => Err(Failure::NeedsShader),
            };
        }
        if self.speculative {
            return Err(self.bail(crate::slices::spec::Hole::Pipeline));
        }
        // The relocatable translation of this key (with the key of the
        // program at its own address).
        let relocating = {
            let options = options_key();
            relocation_key(p, prepared, options, mesh, float_mode).filter(|&r| r == key).map(|_| {
                let address = if p.address_independent { 0 } else { p.address };
                hash_of(&(p.plan.hash, p.code_hash, address, prepared.resource_key, options, mesh, float_mode))
            })
        };
        if relocating.is_some() {
            self.t.relocating.w().insert(key, p.address);
        }
        let job = {
            let (dev, compiler, prepared, address) = (self.dev.clone(), compiler.clone(), prepared.clone(), p.address);
            move || {
                let mut request = prepared.request();
                request.float_mode = float_mode;
                options(&mut request);
                request.relocatable_code = relocating.is_some();
                let input = ShaderCompileInput { stage, mesh, request: &request, specialization: &prepared.specialization, guest_address: address, key, code: None };
                let compiling = crate::perf::CompileGuard::start();
                let _watch = super::watch::start(format!("{stage:?} program {address:#x} (module {key:016x}, {} code bytes)", prepared.program.code_len));
                let t = std::time::Instant::now();
                // A relocatable translation is kept by the prewarm list only
                // once proven (the module cache would serve it unproven).
                let compiled = if relocating.is_some() { compiler.compile_uncached(&input) } else { compiler.compile(&input) };
                let mut relocated = None;
                let mut source = None;
                let result = compiled.map(|c| {
                    let mut src = super::sources::ModuleSource {
                        program: Arc::new(super::sources::ProgramSource::of(&prepared.program)),
                        request: super::sources::encode_request(&request, &prepared.specialization, mesh),
                        reads: Arc::new(super::sources::CodeImage::from_words(c.code_reads.clone())),
                    };
                    let mut check = None;
                    if let Some(plain_key) = relocating {
                        let proven = super::sources::proves_relocatable(&*compiler, key, &src, &c);
                        if proven {
                            src.request = super::sources::mark_proven(&src.request);
                            check = Some(Arc::new(CodeCheck::of(address, c.code_reads.iter().copied())));
                        }
                        relocated = Some(Relocated { proven, plain_key });
                    }
                    source = Some(Arc::new(src));
                    (c, check)
                });
                let translate_ns = t.elapsed().as_nanos() as u64;
                let t = std::time::Instant::now();
                let result = match result {
                    Ok((c, relocation)) => match create_shader_module(&dev, &c.spirv) {
                        Ok(module) => Ok((Arc::new(Module { module, reflection: c.reflection, address, words: c.spirv.len(), relocation }), c.spirv)),
                        Err(e) => Err(CompileError::Failed(format!("vkCreateShaderModule: {e:?}"))),
                    },
                    Err(e) => Err(e),
                };
                let module_ns = t.elapsed().as_nanos() as u64;
                drop(compiling);
                if translate_ns + module_ns >= super::timing::SLOW_MS * 1_000_000 {
                    let words = result.as_ref().map_or(0, |(m, _)| m.words);
                    let (code, wave) = (prepared.program.code_len, prepared.program.wave_size);
                    log::info!(
                        "ps5gpu: {stage:?} program {address:#x} ({code} code bytes, wave{wave}, {words} SPIR-V words): translated in {} ms, module {} ms",
                        translate_ns / 1_000_000,
                        module_ns / 1_000_000
                    );
                }
                Compiled::Module { key, stage, address, compiler: compiler.name().to_string(), result, translate_ns, module_ns, source, relocated }
            }
        };
        let Some(pool) = &*self.t.pool else {
            self.finish(job());
            return self.module_ready(key);
        };
        if self.t.pending_modules.w().insert(key) {
            pool.submit_urgent(job, key);
        } else {
            // Queued by the prewarm list: needed now.
            pool.hurry(key);
        }
        if self.ahead.queue_only {
            return Err(Failure::Compiling);
        }
        if !wait {
            // A module from the disk cache is ready in well under a
            // millisecond: wait within the frame's budget before skipping.
            if self.wait_within_budget(|s| s.t.modules.contains_key(&key))? {
                return self.module_ready(key);
            }
            return Err(Failure::Compiling);
        }
        self.stats.compile_waits += 1;
        let t = std::time::Instant::now();
        self.run_queued(key);
        while !self.t.modules.contains_key(&key) {
            self.wait_compile_step(t)?;
        }
        self.note_blocking_wait(t, || format!("{stage:?} program {:#x} (module {key:016x})", p.address));
        self.module_ready(key)
    }

    /// Whether the relocatable module `rkey` serves (or, queued or not yet
    /// made, will serve) the program at `address`.
    fn relocation_serves(&self, rkey: u64, address: u64) -> bool {
        if self.t.not_relocatable.contains(&rkey) {
            return false;
        }
        match self.t.modules.get(&rkey) {
            Some(Ok(m)) => {
                let memory = self.memory;
                m.serves(address, &|a| {
                    let mut b = [0u8; 4];
                    memory.read_backing(a, &mut b).then(|| u32::from_le_bytes(b))
                })
            }
            Some(Err(_)) => false,
            None => self.t.relocating.get(&rkey).is_none_or(|&a| a == address),
        }
    }

    fn module_ready(&self, key: u64) -> Result<(u64, Arc<Module>), Failure> {
        match self.t.modules.get(&key) {
            Some(Ok(m)) => Ok((key, m.clone())),
            _ => Err(Failure::NeedsShader),
        }
    }

    /// Wait for the next finished compile job and collect it.
    /// Wait for compile jobs until `done` or the frame's compile-wait
    /// budget (`RPS5E_COMPILE_WAIT_MS` per presented frame, 0 skips at
    /// once; unset, no limit) runs out. True when `done`.
    /// Without a limit (the exact mode) the wait may give the item up as
    /// pending, as [`Self::wait_compile_step`] does.
    fn wait_within_budget(&mut self, done: impl Fn(&Self) -> bool) -> Result<bool, Failure> {
        if done(self) {
            return Ok(true);
        }
        let start = std::time::Instant::now();
        let mut ok = false;
        while !self.compile_wait_left.is_zero() {
            let t0 = std::time::Instant::now();
            if self.compile_wait == std::time::Duration::MAX && self.may_give_up_pending() && (self.compile_pending || start.elapsed() >= super::unlocked_wait::GRACE) {
                self.t.compile_times.w().waits.add(start.elapsed().as_nanos() as u64);
                return Err(self.give_up_pending());
            }
            if self.wait_compile().is_err() {
                break;
            }
            self.compile_wait_left = self.compile_wait_left.saturating_sub(t0.elapsed());
            if done(self) {
                self.stats.compile_budget_waits += 1;
                ok = true;
                break;
            }
        }
        if !self.compile_wait.is_zero() {
            self.t.compile_times.w().waits.add(start.elapsed().as_nanos() as u64);
        }
        Ok(ok)
    }

    /// Whether the item being recorded may be given up for a pending
    /// compile ([`super::unlocked_wait`]): the caller allows it, and this
    /// is the exact recording with the compile pool (not a slice thread,
    /// not the compile lookahead, not a frame trace that logs each try).
    fn may_give_up_pending(&self) -> bool {
        super::unlocked_wait::allowed() && !self.speculative && !self.ahead.queue_only && self.t.pool.is_some() && !trace::active()
    }

    /// Give the item up for a pending compile: queue the lookahead's
    /// compiles first (as a wait would), then [`Failure::Pending`].
    fn give_up_pending(&mut self) -> Failure {
        self.compile_ahead();
        self.compile_pending = true;
        Failure::Pending
    }

    /// One step of a wait, begun at `started`, for a compile the item
    /// cannot do without: past [`super::unlocked_wait::GRACE`] the item is
    /// given up when [`Self::may_give_up_pending`], otherwise
    /// [`Self::wait_compile`].
    fn wait_compile_step(&mut self, started: std::time::Instant) -> Result<(), Failure> {
        if ps5vm::shutdown::requested() {
            return Err(Failure::Skip);
        }
        // Another stage of the item was given up already: its other modules
        // are queued, nothing is left to wait for under the lock.
        if self.may_give_up_pending() && (self.compile_pending || started.elapsed() >= super::unlocked_wait::GRACE) {
            return Err(self.give_up_pending());
        }
        self.wait_compile()
    }

    /// Fails once the process shuts down: the command thread leaves the
    /// draw, stops, and the back-end's teardown saves the prewarm list
    /// with what this run compiled (a thread left waiting for a long
    /// compile kept the teardown from running).
    fn wait_compile(&mut self) -> Result<(), Failure> {
        if ps5vm::shutdown::requested() {
            return Err(Failure::Skip);
        }
        self.compile_ahead();
        if let Some(pool) = &*self.t.pool {
            pool.wait(std::time::Duration::from_millis(2));
        }
        self.harvest();
        Ok(())
    }

    /// The item being recorded cannot go on without the queued compile job
    /// `tag`: queued behind the compiles of skipped draws and of the
    /// lookahead, it would wait for all of them (GTA V waited up to a
    /// second for jobs of a few ms each). When the item will be given up
    /// and waited for with the back-end unlocked, the job moves to the
    /// front of the queue; otherwise this thread, which blocks on it
    /// anyway, runs it when no worker has started it.
    fn run_queued(&mut self, tag: u64) {
        if self.may_give_up_pending() {
            if self.t.pool.as_ref().is_some_and(|p| p.hasten(tag)) {
                self.stats.hastened_compiles += 1;
            }
            return;
        }
        let job = self.t.pool.as_ref().and_then(|p| p.take_queued(tag));
        if let Some(job) = job {
            self.stats.inline_compiles += 1;
            self.finish(job());
        }
    }

    /// Count a wait the command thread blocked in, started at `t`; one of
    /// [`super::timing::SLOW_MS`] or more is logged with what it waited for.
    fn note_blocking_wait(&mut self, t: std::time::Instant, what: impl FnOnce() -> String) {
        let ns = t.elapsed().as_nanos() as u64;
        self.t.compile_times.w().waits.add(ns);
        if ns >= super::timing::SLOW_MS * 1_000_000 {
            let queued = self.t.pool.as_ref().map_or(0, |p| p.in_flight());
            log::info!("ps5gpu: the command thread waited {} ms for {} ({queued} compile jobs in flight)", ns / 1_000_000, what());
        }
    }

    /// Collect the finished compile jobs.
    fn harvest(&mut self) {
        let Some(pool) = &*self.t.pool else { return };
        for c in pool.take() {
            self.finish(c);
        }
    }

    /// Enter a finished compile job into the caches.
    fn finish(&mut self, c: Compiled) {
        match c {
            Compiled::Module { key, stage, address, compiler, result, translate_ns, module_ns, source, relocated } => {
                self.t.pending_modules.w().remove(&key);
                self.t.relocating.w().remove(&key);
                self.t.compile_times.w().translate.add(translate_ns);
                if result.is_ok() {
                    self.t.compile_times.w().module.add(module_ns);
                }
                let from_list = self.t.prewarm_translating.w().remove(&key);
                let entry = match result {
                    Ok((m, spirv)) if relocated.is_some_and(|r| !r.proven) => {
                        // Address dependent after all: the module serves the
                        // program where it was translated, under that
                        // address's key.
                        let plain = relocated.map_or(key, |r| r.plain_key);
                        self.t.not_relocatable.w().insert(key);
                        self.t.prewarm.w().add_module(plain, stage as u8, spirv, m.reflection.clone(), source);
                        self.t.modules.w().insert(plain, Ok(m.clone()));
                        Ok(m)
                    }
                    Ok((m, spirv)) => {
                        self.t.prewarm.w().add_module(key, stage as u8, spirv, m.reflection.clone(), source);
                        if from_list {
                            self.t.prewarm_counts.w().translated += 1;
                        }
                        Ok(m)
                    }
                    Err(e) if from_list => {
                        // Not entered as failed: a draw that needs the module
                        // translates it from guest memory.
                        log::debug!("ps5gpu: prewarm: {stage:?} program {address:#x} not translated again: {e}");
                        self.t.prewarm_counts.w().failed += 1;
                        self.t.prewarm.w().remove_module(key);
                        self.prewarm_module_done();
                        return;
                    }
                    Err(e) => {
                        self.stats.compile_failures += 1;
                        crate::log_once(&format!("ps5gpu: {compiler} could not compile {stage:?} program {address:#x}: {e}"));
                        Err(e)
                    }
                };
                self.t.modules.w().insert(key, entry);
                if from_list {
                    self.prewarm_module_done();
                }
            }
            Compiled::Graphics { key, result, record, built, rebuild } => {
                self.note_built(built, true);
                if built.tier == Tier::Optimised {
                    match result {
                        Ok(p) => self.swap_in(Some(&key), None, p),
                        Err(e) => crate::log_once(&format!("ps5gpu: optimised graphics pipeline creation failed: {e:?} (the unoptimised one stays)")),
                    }
                    return;
                }
                self.t.pending_graphics.w().remove(&key);
                match result {
                    Ok(p) => {
                        crate::perf::PIPELINE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        self.t.disk.w().note_created();
                        if let Some(r) = record {
                            self.t.prewarm.w().add_graphics(r);
                        }
                        self.queue_rebuild(rebuild);
                        if let Some(old) = self.t.graphics.w().insert(key, p) {
                            // SAFETY: created twice (prewarm and a draw); the older one was never bound.
                            unsafe { self.dev.device.destroy_pipeline(old, None) };
                        }
                    }
                    Err(e) => {
                        crate::log_once(&format!("ps5gpu: graphics pipeline creation failed: {e:?}"));
                        self.t.failed_graphics.w().insert(key);
                    }
                }
            }
            Compiled::Compute { key, result, record, built, rebuild } => {
                self.note_built(built, false);
                if built.tier == Tier::Optimised {
                    match result {
                        Ok(p) => self.swap_in(None, Some(key), p),
                        Err(e) => crate::log_once(&format!("ps5gpu: optimised compute pipeline creation failed: {e:?} (the unoptimised one stays)")),
                    }
                    return;
                }
                self.t.pending_compute.w().remove(&key);
                match result {
                    Ok(p) => {
                        crate::perf::PIPELINE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        self.t.disk.w().note_created();
                        if let Some(r) = record {
                            self.t.prewarm.w().add_compute(r);
                        }
                        self.queue_rebuild(rebuild);
                        if let Some(old) = self.t.compute.w().insert(key, p) {
                            // SAFETY: as above.
                            unsafe { self.dev.device.destroy_pipeline(old, None) };
                        }
                    }
                    Err(e) => {
                        crate::log_once(&format!("ps5gpu: compute pipeline creation failed: {e:?}"));
                        self.t.failed_compute.w().insert(key);
                    }
                }
            }
        }
    }

    fn note_built(&mut self, built: Built, graphics: bool) {
        let t = &mut self.t.compile_times.w();
        match built.tier {
            Tier::Full | Tier::Cached if graphics => t.graphics.add(built.ns),
            Tier::Full | Tier::Cached => t.compute.add(built.ns),
            Tier::Fast => t.fast.add(built.ns),
            Tier::Optimised => t.optimised.add(built.ns),
        }
        if built.tier == Tier::Cached {
            t.probe_hits += 1;
        }
    }

    /// Queue the optimised build of a fast pipeline behind every other
    /// compile job.
    fn queue_rebuild(&mut self, rebuild: Option<Rebuild>) {
        let (Some(job), Some(pool)) = (rebuild, &*self.t.pool) else { return };
        pool.submit_background(job);
    }

    /// Replace the fast pipeline of `graphics` / `compute` by its optimised
    /// build `p`. The fast one may be recorded in the open tick and named
    /// by draw recipes: the recipes start over, and the pipeline is
    /// destroyed once the GPU is done with the work recorded so far
    /// ([`Self::retire_swapped`]).
    fn swap_in(&mut self, graphics: Option<&GraphicsKey>, compute: Option<ComputeKey>, p: vk::Pipeline) {
        self.t.disk.w().note_created();
        let old = match (graphics, compute) {
            (Some(k), _) => self.t.graphics.w().insert(k.clone(), p),
            (_, Some(k)) => self.t.compute.w().insert(k, p),
            _ => None,
        };
        self.t.compile_times.w().swaps += 1;
        if let Some(old) = old {
            self.t.swapped_out.w().push(old);
        }
        if graphics.is_some() {
            self.recipes = Default::default();
            *self.t.swaps.w() += 1;
            self.swaps_seen = *self.t.swaps;
        }
    }

    /// Hand the pipelines replaced by their optimised builds to the
    /// scheduler: destroyed after the open tick completes (every use was
    /// recorded in it or before).
    fn retire_swapped(&mut self, sched: &mut crate::scheduler::Scheduler) {
        if self.t.swapped_out.is_empty() {
            return;
        }
        struct Retired(Arc<GpuDevice>, Vec<vk::Pipeline>);
        impl Drop for Retired {
            fn drop(&mut self) {
                for &p in &self.1 {
                    // SAFETY: the tick that last could use it completed.
                    unsafe { self.0.device.destroy_pipeline(p, None) };
                }
            }
        }
        sched.retire(Retired(self.dev.clone(), std::mem::take(&mut self.t.swapped_out.w())));
    }

    /// Wait until no optimised rebuild is queued or running, and swap the
    /// results in (a benchmark settles its pipelines before timing).
    pub fn settle_rebuilds(&mut self) {
        while self.t.pool.as_ref().is_some_and(|p| p.background_pending() != 0) {
            if let Some(pool) = &*self.t.pool {
                pool.wait(std::time::Duration::from_millis(5));
            }
            self.harvest();
        }
        self.harvest();
    }

    /// Load the prewarm list once: its modules and layouts now, its
    /// pipelines on the compile workers behind the work draws wait for.
    /// Modules of another translator are translated again on the workers
    /// first (the pipelines that use them follow when all are done).
    fn load_prewarm(&mut self) {
        if *self.t.prewarm_loaded {
            return;
        }
        *self.t.prewarm_loaded.w() = true;
        let stamp = self.compiler.as_ref().and_then(|c| c.translator_stamp()).unwrap_or(0);
        *self.t.prewarm.w() = super::prewarm::PrewarmList::load(self.t.cache_path.as_deref(), stamp);
        if self.t.pool.is_none() || !self.t.prewarm.is_active() {
            return;
        }
        let t = std::time::Instant::now();
        for m in self.t.prewarm.modules.iter().filter(|m| m.fresh) {
            if let Ok(module) = create_shader_module(&self.dev, &m.spirv) {
                let (address, relocation) = proven_relocation(m.source.as_deref());
                self.t.modules.w().insert(m.key, Ok(Arc::new(Module { module, reflection: m.reflection.clone(), address, words: m.spirv.len(), relocation })));
            }
        }
        let stale: Vec<(u64, Arc<super::sources::ModuleSource>)> = self.t.prewarm.stale().filter_map(|m| Some((m.key, m.source.clone()?))).collect();
        if let (Some(compiler), Some(pool)) = (self.compiler.clone(), &*self.t.pool) {
            for (key, source) in stale {
                if self.t.modules.contains_key(&key) || !self.t.pending_modules.w().insert(key) {
                    continue;
                }
                self.t.prewarm_translating.w().insert(key);
                pool.submit_prewarm(translate_job(self.dev.clone(), compiler.clone(), key, source), key);
            }
        }
        let records: Vec<PrewarmPipeline> =
            self.t.prewarm.graphics.iter().cloned().map(PrewarmPipeline::Graphics).chain(self.t.prewarm.compute.iter().cloned().map(PrewarmPipeline::Compute)).collect();
        let queued = self.queue_prewarm_pipelines(records);
        if !self.t.prewarm.modules.is_empty() {
            log::info!(
                "ps5gpu: shader prewarm: {} modules in {:?}, {} to translate again, {queued} pipelines queued, {} waiting for their modules",
                self.t.prewarm.modules.len(),
                t.elapsed(),
                self.t.prewarm_translating.len(),
                self.t.prewarm_deferred.len()
            );
        }
    }

    /// Queue the prewarm builds of `records` whose modules are ready (those
    /// waiting for a translation are kept for later). Returns the count.
    fn queue_prewarm_pipelines(&mut self, records: Vec<PrewarmPipeline>) -> usize {
        let mut queued = 0;
        for r in records {
            let waits = |k: u64| k != 0 && self.t.prewarm_translating.contains(&k);
            let (job, tag): (Box<dyn FnOnce() -> Compiled + Send>, u64) = match r {
                PrewarmPipeline::Graphics(g) => {
                    if waits(g.vs) || waits(g.ps) {
                        self.t.prewarm_deferred.w().push(PrewarmPipeline::Graphics(g));
                        continue;
                    }
                    let Some(Ok(vs)) = self.t.modules.get(&g.vs).cloned() else { continue };
                    let ps = match g.ps {
                        0 => None,
                        k => match self.t.modules.get(&k).cloned() {
                            Some(Ok(m)) => Some(m),
                            _ => continue,
                        },
                    };
                    let Ok(layout) = self.t.layouts.w().get(&g.layout, vk::ShaderStageFlags::from_raw(g.push_stages)) else { continue };
                    self.t.layout_descs.w().entry(layout.pipeline_layout).or_insert_with(|| (g.layout.clone(), g.push_stages));
                    let tess = if g.ff.rect == RectMode::Tessellation {
                        match self.rect_modules(&vs.reflection) {
                            Ok(Some(t)) => Some(t),
                            _ => continue,
                        }
                    } else {
                        None
                    };
                    let key = GraphicsKey { vs: g.vs, ps: g.ps, layout: layout.pipeline_layout, ff: g.ff.clone() };
                    if self.t.graphics.contains_key(&key) || !self.t.pending_graphics.w().insert(key.clone()) {
                        continue;
                    }
                    let tag = hash_of(&key);
                    (Box::new(graphics_job(self.dev.clone(), self.t.disk.cache, key, vs, ps, tess, g.mesh, None, self.fast_builds())), tag)
                }
                PrewarmPipeline::Compute(c) => {
                    if waits(c.cs) {
                        self.t.prewarm_deferred.w().push(PrewarmPipeline::Compute(c));
                        continue;
                    }
                    let Some(Ok(cs)) = self.t.modules.get(&c.cs).cloned() else { continue };
                    let Ok(layout) = self.t.layouts.w().get(&c.layout, vk::ShaderStageFlags::COMPUTE) else { continue };
                    self.t.layout_descs.w().entry(layout.pipeline_layout).or_insert_with(|| (c.layout.clone(), vk::ShaderStageFlags::COMPUTE.as_raw()));
                    let key = ComputeKey { cs: c.cs, layout: layout.pipeline_layout, subgroup32: c.subgroup32 };
                    if self.t.compute.contains_key(&key) || !self.t.pending_compute.w().insert(key) {
                        continue;
                    }
                    (Box::new(compute_job(self.dev.clone(), self.t.disk.cache, key, cs, None, self.fast_builds())), hash_of(&key))
                }
            };
            if let Some(pool) = &*self.t.pool {
                pool.submit_prewarm(job, tag);
                queued += 1;
            }
        }
        queued
    }

    /// A module of the list was translated again (or failed to): when it
    /// was the last one, queue the pipelines that waited for them.
    fn prewarm_module_done(&mut self) {
        if self.t.prewarm_translating.is_empty() && !self.t.prewarm_deferred.is_empty() {
            let deferred = std::mem::take(&mut *self.t.prewarm_deferred.w());
            self.queue_prewarm_pipelines(deferred);
        }
    }

    /// Load the prewarm list (when not loaded yet) and build all of it now:
    /// every module translated again and every pipeline built, on all
    /// compile workers, before anything else runs. `progress` is called
    /// about four times a second; returning false stops early (the queued jobs
    /// finish in the background). The list and the pipeline cache are
    /// saved at the end.
    pub fn prewarm_now(&mut self, progress: &mut dyn FnMut(&PrewarmProgress) -> bool) -> PrewarmSummary {
        let start = std::time::Instant::now();
        self.load_prewarm();
        let pipelines = self.t.prewarm.graphics.len() + self.t.prewarm.compute.len();
        let mut summary = PrewarmSummary { modules: self.t.prewarm.modules.len(), pipelines, ..Default::default() };
        let left = |s: &Self| s.t.pool.as_ref().map_or(0, |p| p.in_flight()) + s.t.prewarm_deferred.len();
        // Every queued pipeline and module translation, and every deferred
        // pipeline.
        let total = left(self);
        let mut reported = std::time::Instant::now();
        loop {
            self.harvest();
            let n = left(self);
            if n == 0 {
                break;
            }
            if reported.elapsed() >= std::time::Duration::from_millis(250) {
                reported = std::time::Instant::now();
                if !progress(&PrewarmProgress { done: total.saturating_sub(n), total, elapsed: start.elapsed() }) {
                    summary.cancelled = true;
                    break;
                }
            }
            if let Some(pool) = &*self.t.pool {
                pool.wait(std::time::Duration::from_millis(50));
            }
        }
        if !summary.cancelled {
            progress(&PrewarmProgress { done: total, total, elapsed: start.elapsed() });
        }
        let counts = *self.t.prewarm_counts;
        summary.translated = counts.translated;
        summary.translate_failures = counts.failed;
        summary.modules = self.t.prewarm.modules.len();
        summary.built = self.t.graphics.len() + self.t.compute.len();
        summary.pipeline_failures = self.t.failed_graphics.len() + self.t.failed_compute.len();
        summary.seconds = start.elapsed().as_secs_f64();
        self.t.prewarm.w().save();
        self.t.disk.w().save_changes();
        summary
    }

    /// Replace the compile workers: `workers` threads (None: the default
    /// share of the cores). The jobs in flight are finished first.
    pub fn set_compile_workers(&mut self, workers: Option<usize>) {
        if let Some(mut pool) = self.t.pool.w().take() {
            for c in pool.shutdown() {
                self.finish(c);
            }
        }
        *self.t.pool.w() = match workers {
            Some(n) => Some(super::pool::CompilePool::new(n)),
            None => default_pool(),
        };
    }

    /// Tessellation control/evaluation modules for a rect-list draw of a
    /// vertex shader with `reflection`.
    fn rect_modules(&mut self, reflection: &ShaderReflection) -> Result<Option<(vk::ShaderModule, vk::ShaderModule)>, Failure> {
        let key = hash_of(&(&reflection.outputs, reflection.clip_distances));
        if let Some(e) = self.t.rect_modules.get(&key) {
            return Ok(*e);
        }
        if self.speculative {
            return Err(self.bail(crate::slices::spec::Hole::Pipeline));
        }
        let built = (|| {
            let tcs = rectlist::control_shader(&reflection.outputs, reflection.clip_distances)?;
            let tes = rectlist::evaluation_shader(&reflection.outputs, reflection.clip_distances)?;
            let a = self.create_module(&tcs).ok()?;
            let b = match self.create_module(&tes) {
                Ok(b) => b,
                Err(_) => {
                    // SAFETY: unused module.
                    unsafe { self.dev.device.destroy_shader_module(a, None) };
                    return None;
                }
            };
            Some((a, b))
        })();
        self.t.rect_modules.w().insert(key, built);
        Ok(built)
    }

    // ------------------------------------------------------------------
    // Buffers
    // ------------------------------------------------------------------

    /// The range of a buffer binding clamped to readable guest memory and
    /// the device limit; None for a null or empty descriptor.
    fn clamp_range(&self, address: u64, size: u64) -> Option<(u64, u64)> {
        if address == 0 || size == 0 {
            return None;
        }
        // The readable extents found so far (maximal runs of readable
        // tracked mappings) hold while no mapping comes, goes or changes
        // access: constant rings hand out a new range per draw inside a few
        // mappings, and unbounded descriptors are cut at the same end.
        let generation = (self.memory.unmap_generation(), ps5vm::ranges::shrink_generation(), ps5vm::ranges::generation());
        let limited = self.binding_limit(size);
        let mut memo = self.clamp_memo.borrow_mut();
        if memo.0 != generation {
            memo.0 = generation;
            memo.1.clear();
            *self.clamp_exact.borrow_mut() = [(0, 0, None); 32];
        }
        let exact = crate::backend::ab::new_fast();
        let slot = ((address >> 4) ^ size).wrapping_mul(0x9E37_79B9_7F4A_7C15) as usize >> 59;
        if exact {
            let e = self.clamp_exact.borrow()[slot];
            if e.0 == address && e.1 == size {
                return e.2;
            }
        }
        let r = self.clamp_range_memo(&mut memo, address, size, limited);
        let r = extend_over_holes(self.memory, address, size, limited, r);
        if exact {
            self.clamp_exact.borrow_mut()[slot] = (address, size, r);
        }
        r
    }

    fn clamp_range_memo(&self, memo: &mut ((u64, u64, u64), ReadableRuns), address: u64, size: u64, limited: u64) -> Option<(u64, u64)> {
        if let Some(e) = memo.1.end_of(address) {
            // As `clamp_range_uncached`: halved until readable, at least 4 bytes.
            let mut n = limited;
            while address.saturating_add(n) > e {
                if n <= 256 {
                    return (address + 4 <= e).then_some((address, 4));
                }
                n /= 2;
            }
            return Some((address, n));
        }
        self.clamp_misses.set(self.clamp_misses.get() + 1);
        let r = self.clamp_range_uncached(address, size);
        if r.is_some() {
            // The readable run of tracked mappings from the one holding the
            // address on.
            if let Some((s, mut e, prot)) = ps5vm::ranges::containing(address).filter(|r| r.2.read) {
                for _ in 0..64 {
                    match ps5vm::ranges::containing(e) {
                        Some((_, next, p)) if p.read => e = next,
                        _ => break,
                    }
                }
                let _ = prot;
                memo.1.insert(s, e);
            }
        }
        r
    }

    /// `size` limited to the largest storage-buffer binding.
    fn binding_limit(&self, size: u64) -> u64 {
        size.min((self.dev.caps.max_storage_buffer_range as u64).saturating_sub(256).clamp(4096, MAX_BUFFER_BINDING))
    }

    fn clamp_range_uncached(&self, address: u64, size: u64) -> Option<(u64, u64)> {
        let mut size = self.binding_limit(size);
        let readable = |a: u64, n: u64| self.memory.backing_ptr(a, n).is_some() || ps5vm::ranges::accessible(a, n, false);
        while !readable(address, size) {
            if size <= 256 {
                return readable(address, 4).then_some((address, 4));
            }
            size /= 2;
        }
        Some((address, size))
    }

    /// Every guest range the stages' buffers and device-address ranges
    /// touch, `(address, size, written)`.
    fn stage_ranges(&self, s: &PreparedStage, out: &mut RangeList) {
        self.stage_ranges_mapped(s, out, None);
    }

    /// [`Self::stage_ranges`]; `map` receives the index in `out` of each
    /// buffer's range (`u32::MAX`: a null or unreadable descriptor).
    fn stage_ranges_mapped(&self, s: &PreparedStage, out: &mut RangeList, mut map: Option<&mut smallvec::SmallVec<[u32; 16]>>) {
        let info = &s.resources.info;
        // A stage reading guest memory reserves no mirror.
        if info.buffers_in_guest_memory {
            return;
        }
        for (i, words) in s.snapshot.buffers.iter().enumerate() {
            let v = BufferSharp::from_words(words);
            let written = info.buffers.get(i).is_some_and(|b| b.written || b.atomic);
            let index = match self.clamp_range(v.address(), buffer_extent(&v, info.buffers.get(i))) {
                Some((a, n)) => {
                    out.push((a, n, written, true));
                    out.len() as u32 - 1
                }
                None => u32::MAX,
            };
            if let Some(m) = map.as_deref_mut() {
                m.push(index);
            }
        }
        for r in &s.snapshot.device_address_ranges {
            if r.planned && r.size != 0 {
                if let Some((a, n)) = self.clamp_range(r.base, r.size) {
                    out.push((a, n, r.written, false));
                }
            }
        }
    }

    /// Upload `words` followed by zeros up to `count` words.
    fn upload_words_padded(&mut self, ctx: &mut DrawContext, words: &[u32], count: usize) -> Result<vk::DescriptorBufferInfo, Failure> {
        let align = self.dev.caps.min_storage_buffer_offset_alignment.max(16);
        let bytes = (count as u64 * 4).max(16);
        let offset = self.staging.alloc(ctx.sched, bytes, align).ok_or(Failure::StagingFull)?;
        let dst = self.staging.bytes(offset, bytes);
        // SAFETY: plain u32s viewed as their (little-endian host) bytes.
        let src = unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words)) };
        dst[..src.len()].copy_from_slice(src);
        dst[src.len()..].fill(0);
        self.staging.buffer.flush(offset, bytes);
        Ok(vk::DescriptorBufferInfo { buffer: self.staging.buffer.buffer, offset, range: bytes })
    }

    /// Copy the table slots whose words earlier GPU work is still producing
    /// from the buffer mirror of their guest dwords into the uploaded table
    /// `table`, ordered after that work and before the work that binds the
    /// table: the shader sees the word as the GPU wrote it, and the CPU
    /// never waits for it.
    fn fill_deferred_table_words(&mut self, ctx: &mut DrawContext, table: vk::DescriptorBufferInfo, deferred: &[(u32, u64)]) -> Result<(), Failure> {
        let mut regions: smallvec::SmallVec<[(vk::Buffer, vk::BufferCopy); 8]> = smallvec::SmallVec::new();
        for &(slot, address) in deferred {
            let src = ctx.buffers.obtain(ctx.sched, address, 4, false)?;
            regions.push((src.buffer, vk::BufferCopy { src_offset: src.offset, dst_offset: table.offset + slot as u64 * 4, size: 4 }));
        }
        let cb = ctx.sched.cmd();
        for (buffer, region) in &regions {
            // SAFETY: recording buffer outside any rendering scope; the
            // mirror and the staging ring stay alive for the tick.
            unsafe { self.dev.device.cmd_copy_buffer(cb, *buffer, table.buffer, std::slice::from_ref(region)) };
        }
        ctx.sched.full_barrier();
        self.stats.deferred_table_words += deferred.len() as u64;
        Ok(())
    }

    /// Upload `words` to the staging ring as a storage-buffer binding.
    fn upload_words(&mut self, ctx: &mut DrawContext, words: &[u32]) -> Result<vk::DescriptorBufferInfo, Failure> {
        // SAFETY: plain u32s viewed as their (little-endian host) bytes.
        let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words)) };
        self.upload(ctx, bytes, self.dev.caps.min_storage_buffer_offset_alignment.max(16))
    }

    /// The read-only bound ranges of a draw that are bound from copies of
    /// their guest bytes (decided once, before any buffer is reserved):
    /// not written by the draw, not overlapping a range it writes, and
    /// newer in guest memory than in their mirror.
    fn choose_streams(&mut self, ctx: &mut DrawContext, ranges: &[(u64, u64, bool, bool)]) {
        self.streams.clear();
        self.refresh_stream_memo(ctx);
        // The streamed flags sit parallel to `ranges` (the tuple list stays
        // for the binding loop's rare un-resolved lookups).
        self.stream_flags.clear();
        self.stream_flags.resize(ranges.len(), false);
        let written: smallvec::SmallVec<[&(u64, u64, bool, bool); 8]> = ranges.iter().filter(|r| r.2).collect();
        // With page checks the memo's stamp only moves with the tick: the
        // refresh above holds for the whole draw.
        let refresh = !stream_pages_enabled();
        for (i, &(a, n, written_flag, bindable)) in ranges.iter().enumerate() {
            if written_flag || !bindable {
                continue;
            }
            if !written.is_empty() && written.iter().any(|&&(b, m, _, _)| b < a + n && a < b + m) {
                continue;
            }
            // A range copied earlier while its bytes are unchanged still
            // prefers its copy (it is neither GPU-dirty nor CPU-clean now).
            if self.stream_copy(ctx, a, n, refresh).is_some() || ctx.buffers.prefers_stream(a, n) {
                self.streams.push((a, n));
                self.stream_flags[i] = true;
            }
        }
    }

    /// A copy of the guest bytes of `[addr, addr+size)` in the staging
    /// ring, bound in place of the mirror (None: unreadable or no room).
    fn stream_guest(&mut self, ctx: &mut DrawContext, addr: u64, size: u64) -> Option<vk::DescriptorBufferInfo> {
        if let Some(info) = self.stream_copy(ctx, addr, size, true) {
            self.stats.stream_memo_hits += 1;
            return Some(info);
        }
        let align = self.dev.caps.min_storage_buffer_offset_alignment.max(16);
        let n = size.max(16);
        let offset = self.staging.alloc(ctx.sched, n, align)?;
        // The pages' generations before the bytes are read.
        let mut deps = crate::memory::deps::PageDeps::begin(self.memory);
        deps.note(&ctx.buffers.tracker(), addr, size);
        if let Some(spec) = ctx.spec() {
            spec.note_pages(ctx.buffers.tracker(), addr, size);
        }
        // The binding spans the whole slot: bytes past the copy read 0, not
        // what an earlier copy left in the ring (the mirror binds exactly
        // `size` bytes, past which reads are 0 as well).
        let copied = crate::backend::profile::dtime(47, || {
            let (copy, tail) = self.staging.bytes(offset, n).split_at_mut(size as usize);
            let ok = self.memory.read_backing(addr, copy);
            if ok {
                tail.fill(0);
            }
            ok
        });
        if !copied {
            return None;
        }
        self.stats.stream_copies += 1;
        self.stats.stream_bytes += size;
        self.staging.buffer.flush(offset, n);
        let info = vk::DescriptorBufferInfo { buffer: self.staging.buffer.buffer, offset, range: n };
        if stream_memo_enabled() && !trace::active() {
            if self.stream_memo.len() >= 1 << 14 {
                self.stream_memo.clear();
            }
            let tick = ctx.sched.current_tick();
            self.stream_memo.insert((addr, size), (info, deps, tick));
        }
        Some(info)
    }

    /// The copy of `[addr, addr+size)` made in the open tick, while its
    /// guest bytes are unchanged (the pages it read, see `PageDeps`; with
    /// `RPS5E_STREAM_PAGES=0`: while the buffer cache's content stamp is).
    /// With `RPS5E_STREAM_CARRY=1` a copy outlives its tick: a later tick
    /// that finds its pages unchanged binds the same ring bytes again
    /// (the region is held for that tick through `StagingRing::holds`/
    /// `retag`). Off by default: on the recorded frame only ~7 % of the
    /// copies are re-read with unchanged pages, while every lookup of a
    /// stale entry pays the page-generation walk.
    fn stream_copy(&mut self, ctx: &DrawContext, addr: u64, size: u64, refresh: bool) -> Option<vk::DescriptorBufferInfo> {
        if refresh {
            self.refresh_stream_memo(ctx);
        }
        let pages = stream_pages_enabled();
        let (info, deps, tick) = self.stream_memo.get_mut(&(addr, size))?;
        if pages && !deps.current(&ctx.buffers.tracker(), self.memory) {
            self.stream_memo.remove(&(addr, size));
            return None;
        }
        if let Some(spec) = ctx.spec() {
            if !pages || deps.overflowed() {
                // The copy's bytes cannot be checked at commit.
                self.stream_memo.remove(&(addr, size));
                return None;
            }
            spec.note_deps(ctx.buffers.tracker(), deps);
        }
        let now = ctx.sched.current_tick();
        if *tick != now {
            if !stream_carry_enabled() {
                self.stream_memo.remove(&(addr, size));
                return None;
            }
            let (used, size) = self.staging.used();
            // A copy held for this tick keeps its ring bytes reserved: only
            // a ring with room for the new tick's copies keeps it, a full
            // one re-copies instead (the allocations must not stall).
            if size > 0 && used * 2 >= size || !self.staging.holds(ctx.sched.timeline.completed_tick(), info.offset, info.range, *tick) {
                self.stream_memo.remove(&(addr, size));
                return None;
            }
            self.staging.retag(*tick, now);
            *tick = now;
        }
        Some(*info)
    }

    /// Forget the stream copies when the content stamp moved (without page
    /// checks the guest bytes may differ); with page checks a copy stays
    /// until its pages change (`stream_copy` validates it).
    fn refresh_stream_memo(&mut self, ctx: &DrawContext) {
        if !stream_pages_enabled() {
            let stamp = (ctx.sched.current_tick(), ctx.buffers.content_stamp());
            if self.stream_stamp != stamp {
                self.stream_memo.clear();
                self.stream_stamp = stamp;
            }
            return;
        }
        // With page checks the stamp stays zero (compared as a constant:
        // a stamp built on the stack and compared as a whole stalled on
        // store forwarding, on every lookup).
        if self.stream_stamp.0 != 0 || self.stream_stamp.1 != [0; 4] {
            self.stream_memo.clear();
            self.stream_stamp = (0, [0; 4]);
        }
        // Without carrying, a copy of an earlier tick is never bound again
        // (`stream_copy` drops it when looked up): dropping them all when
        // the tick moves keeps the map small (the constant rings hand out a
        // new range per draw, nearly none of them is looked up again).
        if !stream_carry_enabled() {
            let tick = ctx.sched.current_tick();
            if self.stream_memo_tick != tick {
                self.stream_memo.clear();
                self.stream_memo_tick = tick;
            }
        }
    }

    fn upload(&mut self, ctx: &mut DrawContext, bytes: &[u8], align: u64) -> Result<vk::DescriptorBufferInfo, Failure> {
        let size = (bytes.len() as u64).max(16);
        let offset = self.staging.alloc(ctx.sched, size, align).ok_or(Failure::StagingFull)?;
        self.staging.bytes(offset, bytes.len() as u64).copy_from_slice(bytes);
        self.staging.buffer.flush(offset, size);
        Ok(vk::DescriptorBufferInfo { buffer: self.staging.buffer.buffer, offset, range: size })
    }

    // ------------------------------------------------------------------
    // Descriptors
    // ------------------------------------------------------------------

    /// Bind one stage's resources: storage buffers (with offset
    /// adjustments into the push data), images, samplers and shared
    /// buffers; the image ids the stage writes go to `writes`.
    fn bind_stage(&mut self, ctx: &mut DrawContext, b: &mut BoundStage, writes: &mut SetWrites) -> Result<(), Failure> {
        crate::backend::profile::time(10, || self.bind_stage_inner(ctx, b, writes))
    }

    fn bind_stage_inner(&mut self, ctx: &mut DrawContext, b: &mut BoundStage, writes: &mut SetWrites) -> Result<(), Failure> {
        let slot = stage_slot(b.stage);
        let ways = if stage_memo_lru_enabled() { STAGE_MEMOS } else { 1 };
        let (layout, resources) = (&b.prepared.layout, &b.prepared.resources);
        let list = &mut self.stage_memos[slot];
        let found = list.iter().position(|m| m.layout.as_ref().is_some_and(|l| Arc::ptr_eq(l, layout)) && m.resources.as_ref().is_some_and(|r| Arc::ptr_eq(r, resources)));
        // The memo used moves to the front (most recent first); memos are
        // large and boxed, so only pointers move.
        let mut memo = match found {
            Some(i) => list.remove(i),
            // The least recently used memo is reset for this layout.
            None if list.len() >= ways => list.pop().unwrap_or_default(),
            None => Box::default(),
        };
        let r = self.bind_stage_memo(ctx, b, writes, &mut memo);
        let list = &mut self.stage_memos[slot];
        list.insert(0, memo);
        list.truncate(ways);
        r
    }

    fn bind_stage_memo(&mut self, ctx: &mut DrawContext, b: &mut BoundStage, writes: &mut SetWrites, memo: &mut StageMemo) -> Result<(), Failure> {
        let stage = b.stage;
        let p = &mut b.prepared;
        // Shared, immutable parts are borrowed (only the shader data
        // changes).
        let resources = &p.resources;
        let info = &resources.info;
        let snapshot = &p.snapshot;
        let align = self.dev.caps.min_storage_buffer_offset_alignment.max(4);
        // Reads at addresses no range bounds may reach any page: mirrors
        // current, guest memory where no mirror is (before the page table
        // is bound).
        if !(info.buffers_in_guest_memory && ctx.buffers.bda_active())
            && crate::buffers::unranged_sync_enabled()
            && hazards::reads_unranged_memory(info, &snapshot.device_address_ranges)
        {
            ctx.buffers.prepare_unranged_reads(ctx.sched);
        }
        let layout = &p.layout;
        // The previous results of this stage slot are reusable per binding
        // while the stamp, the layout and the binding's input words are
        // unchanged (see `StageMemo`).
        let content = ctx.buffers.content_stamp();
        let buffers_epoch = ctx.buffers.layout_epoch();
        let tick = ctx.sched.current_tick();
        // Image and sampler bindings do not depend on buffer contents or
        // layout: with split stamps only buffer bindings and ranges are
        // dropped when those move (guest CPU writes move the content stamp
        // every few draws).
        let split = split_stamps_enabled();
        // With per-image lookup generations image bindings are checked
        // image by image and outlive the tick; buffer bindings stay in it.
        let tick_free = split && crate::images::image_generations_enabled();
        let stamp = StageStamp {
            tick: if tick_free { 0 } else { tick },
            content: if split { [0; 4] } else { content },
            images: ctx.images.lookup_epoch(),
            buffers: if split { 0 } else { buffers_epoch },
        };
        if memo.staging_tick != tick {
            // Staging copies belong to their tick — except a flattened-table
            // copy whose ring bytes are still reserved: it carries into the
            // new tick (retagged), so the table is re-uploaded only when the
            // ring recycled it or its words changed (GPU_PLAN §55; with
            // eager write-throughs the tick count grew and the per-tick
            // re-upload became the dominant always-miss binding).
            memo.shader_data = None;
            if let Some((w, n, info)) = memo.table.take() {
                let carry = table_carry_enabled() && {
                    let (used, size) = self.staging.used();
                    size == 0 || used * 2 < size && self.staging.holds(ctx.sched.timeline.completed_tick(), info.offset, info.range, memo.staging_tick)
                };
                if carry {
                    self.staging.retag(memo.staging_tick, tick);
                    memo.table = Some((w, n, info));
                }
            }
            memo.staging_tick = tick;
        }
        let buffer_stamp = (content, buffers_epoch, tick);
        let enabled = bind_memo_enabled() && !trace::active();
        if !(enabled && memo.matches(&stamp, layout, resources)) {
            memo.reset(stamp, layout, resources);
            memo.buffer_stamp = buffer_stamp;
        } else if memo.buffer_stamp != buffer_stamp {
            memo.buffer_stamp = buffer_stamp;
            for (bi, binding) in layout.descriptors.iter().enumerate() {
                if matches!(binding.kind, DescriptorBindingKind::Buffers) {
                    if let Some(ways) = memo.bindings.get_mut(bi) {
                        for m in ways.iter_mut() {
                            m.valid = false;
                        }
                    }
                }
            }
            memo.ranges = None;
            memo.buffer_generation += 1;
        }
        // Read-only buffer bindings are reused only by work that writes no
        // buffer range (a stream copy is wrong next to a write of its range).
        let buffers_cacheable = enabled && !self.draw_writes_ranges;
        // Images selected through the bindless set: their maps' words go
        // into the flattened table.
        let bindless_table = self.bindless_tables(ctx, info, snapshot)?;
        let mut words: smallvec::SmallVec<[u32; 64]> = smallvec::SmallVec::new();
        for (bi, binding) in layout.descriptors.iter().enumerate() {
            let number = native_binding_index(stage, binding.kind);
            // The inputs of a binding without side effects.
            words.clear();
            let cacheable = enabled
                && match binding.kind {
                    DescriptorBindingKind::Buffers => {
                        buffers_cacheable && binding.resources.iter().all(|&r| info.buffers.get(r as usize).is_none_or(|b| !(b.written || b.atomic)))
                    }
                    DescriptorBindingKind::Image(_) => binding.resources.iter().all(|&r| info.images.get(r as usize).is_none_or(|i| !(i.written || i.atomic))),
                    DescriptorBindingKind::Samplers => true,
                    _ => false,
                };
            // GPU_PLAN §54: with wide offset dwords a read-only buffer
            // binding can pin its whole containing buffer and carry the
            // rotating V# base in the offset slot — the binding's
            // identity becomes the buffer (address + serial) and the
            // size, so a constant ring handing out a new address every
            // draw keeps hitting the memo. Null, streamed and
            // oversized ranges keep today's exact binding (and
            // raw-word identity).
            let mut ring: smallvec::SmallVec<[Option<RingBind>; 8]> = smallvec::SmallVec::new();
            let ring_mode = cacheable
                && ring_binding_enabled()
                && matches!(binding.kind, DescriptorBindingKind::Buffers)
                && layout.wide_offsets
                // The reservation's resolved ranges carry the stream
                // decisions; the ring plan re-derives both (its `obtain`
                // keeps the mirror current), so it takes precedence.
                && binding.resources.iter().all(|&r| info.buffers.get(r as usize).is_none_or(|b| !(b.written || b.atomic)));
            if ring_debug_enabled() && matches!(binding.kind, DescriptorBindingKind::Buffers) {
                // TEMPORARY diagnostic: why ring bindings do not happen.
                self.stats.ring_why = match () {
                    _ if !cacheable => 1,
                    _ if !ring_binding_enabled() => 2,
                    _ if !layout.wide_offsets => 3,
                    _ if binding.resources.iter().any(|&r| info.buffers.get(r as usize).is_some_and(|b| b.written || b.atomic)) => 5,
                    _ => 0,
                };
                if ring_mode {
                    self.stats.ring_candidates += 1;
                }
            }
            if cacheable {
                let source = match binding.kind {
                    DescriptorBindingKind::Buffers => &snapshot.buffers,
                    DescriptorBindingKind::Samplers => &snapshot.samplers,
                    _ => &snapshot.images,
                };
                // Phase 1 — the raw-word lookup: a hit costs nothing extra.
                // A miss falls to the ring plan (phase 2): the V# words are
                // replaced by the containing buffer's identity (address +
                // serial), so a ring handing out a new address inside the
                // same buffer keeps hitting — the footprint and format
                // dwords stay compared (a different resource still misses).
                // The plan's clamp + buffer search are paid only on misses
                // (paying them on every draw measured slower).
                let hit: Option<&BindingMemo> = crate::backend::profile::dtime(40, || {
                    for &r in &binding.resources {
                        words.extend_from_slice(source.get(r as usize).map_or(&[][..], |w| &w[..]));
                        // A sampler also follows the levels of the images it
                        // filters (ANISO_OVERRIDE).
                        if matches!(binding.kind, DescriptorBindingKind::Samplers) {
                            words.push(sampler_single_level(info, &snapshot, r) as u32);
                        }
                        // An image follows the coordinates of its samplers.
                        if matches!(binding.kind, DescriptorBindingKind::Image(_)) {
                            words.push(image_unnormalized(info, &snapshot, r) as u32 | (image_force_degamma(info, &snapshot, r) as u32) << 1);
                        }
                        words.push(u32::MAX);
                    }
                    let ways = memo.bindings.get_mut(bi)?;
                    let found = ways.iter_mut().position(|m| m.valid && m.words[..] == words[..] && m.images_current(&mut ctx.images, tick))?;
                    // Most recent first (rotated: the way found is
                    // usually the first, and ways are large).
                    ways[..=found].rotate_right(1);
                    Some(unsafe { &*(&**ways.first_mut().unwrap() as *const BindingMemo) })
                });
                match hit {
                    Some(m) => {
                    if let (Some(spec), Some(r)) = (ctx.spec(), &m.spec_reads) {
                        spec.replay(r, ctx.buffers.tracker());
                    }
                    writes.open(number, m.ty, !m.buffers.is_empty() || matches!(binding.kind, DescriptorBindingKind::Buffers));
                    for &i in &m.buffers {
                        writes.buffer(i);
                    }
                    let base = writes.images.len();
                    for &i in &m.images {
                        writes.image(i);
                    }
                    for s in &m.slots {
                        writes.bound_images.push(s.id);
                        writes.image_slots.push(ImageSlot { index: base + s.index, ..*s });
                    }
                    for &(dword, bits) in &m.adjust {
                        if let Some(d) = p.shader_data.get_mut(dword as usize) {
                            *d |= bits;
                        }
                    }
                    if ring_mode {
                        // The pinned buffers' rotating bases are data: this
                        // draw's offsets go into their wide slots (the memo
                        // identity covers the buffer, not the address; the
                        // content stamp guards the mirror, as for every
                        // memo hit).
                        for e in ring.iter().flatten() {
                            if let Some(d) = p.shader_data.get_mut(e.dword as usize) {
                                *d = e.offset;
                            }
                        }
                        self.stats.ring_reuses += ring.iter().flatten().count() as u64;
                    }
                    self.stats.binding_reuses += 1;
                    continue;
                }
                // Phase 2: the raw words missed — try the
                // ring identity. Only now is the plan paid for (the clamp
                // and the containing-buffer search per resource); its hit
                // reuses the binding with this draw's base patched into
                // the wide offset dword.
                None if ring_mode => {
                    for &r in &binding.resources {
                        let words_r = source.get(r as usize).map_or(&[][..], |w| &w[..]);
                        let v = BufferSharp::from_words(words_r);
                        let entry = (|| {
                            let (a, n) = self.clamp_range(v.address(), buffer_extent(&v, info.buffers.get(r as usize)))?;
                            if self.streams.contains(&(a, n)) {
                                return None;
                            }
                            let key = ctx.buffers.find_containing(a, n)?;
                            let gb = ctx.buffers.get(key)?;
                            let offset = a.checked_sub(gb.addr)?;
                            let whole = self.binding_limit(gb.size);
                            (whole >= offset.saturating_add(n) && offset <= u32::MAX as u64).then(|| RingBind {
                                a,
                                n,
                                key,
                                serial: gb.serial,
                                offset: offset as u32,
                                info: vk::DescriptorBufferInfo { buffer: gb.buf.buffer, offset: 0, range: whole },
                                dword: layout.memory_offset_dword + r,
                            })
                        })();
                        ring.push(entry);
                    }
                    words.clear();
                    let hit: Option<&BindingMemo> = crate::backend::profile::dtime(40, || {
                        // Identity is the containing buffer: the rotating
                        // V# base and its footprint dword are data.
                        for (&r, e) in binding.resources.iter().zip(ring.iter()) {
                            match e {
                                Some(rb) => words.extend_from_slice(&[rb.key as u32, (rb.key >> 32) as u32, rb.serial as u32]),
                                None => words.extend_from_slice(source.get(r as usize).map_or(&[][..], |w| &w[..])),
                            }
                            words.push(u32::MAX);
                        }
                        let ways = memo.bindings.get_mut(bi)?;
                        let found = ways.iter_mut().position(|m| m.valid && m.words[..] == words[..] && m.images_current(&mut ctx.images, tick))?;
                        ways[..=found].rotate_right(1);
                        Some(unsafe { &*(&**ways.first_mut().unwrap() as *const BindingMemo) })
                    });
                    if let Some(m) = hit {
                        if let (Some(spec), Some(r)) = (ctx.spec(), &m.spec_reads) {
                            spec.replay(r, ctx.buffers.tracker());
                        }
                        writes.open(number, m.ty, !m.buffers.is_empty() || matches!(binding.kind, DescriptorBindingKind::Buffers));
                        for &i in &m.buffers {
                            writes.buffer(i);
                        }
                        let base = writes.images.len();
                        for &i in &m.images {
                            writes.image(i);
                        }
                        for s in &m.slots {
                            writes.bound_images.push(s.id);
                            writes.image_slots.push(ImageSlot { index: base + s.index, ..*s });
                        }
                        for &(dword, bits) in &m.adjust {
                            if let Some(d) = p.shader_data.get_mut(dword as usize) {
                                *d |= bits;
                            }
                        }
                        for e in ring.iter().flatten() {
                            if let Some(d) = p.shader_data.get_mut(e.dword as usize) {
                                *d = e.offset;
                            }
                        }
                        self.stats.ring_reuses += ring.iter().flatten().count() as u64;
                        self.stats.binding_reuses += 1;
                        continue;
                    }
                }
                None => {}
            }
        }
            // Where this binding's outputs start (for the memo).
            let (buffers_at, images_at, slots_at) = (writes.buffers.len(), writes.images.len(), writes.image_slots.len());
            let spec_mark = ctx.spec().map(|s| s.mark());
            let mut adjust: smallvec::SmallVec<[(u16, u32); 4]> = smallvec::SmallVec::new();
            let mut ty_used = vk::DescriptorType::STORAGE_BUFFER;
            let mut storable = cacheable;
            let lookups = ctx.images.lookup_stamp();
            self.stats.descriptor_bindings += 1;
            // GPU_PLAN §54: the miss composition — which binding kinds pay
            // the miss loop (the address of the fingerprint work).
            match binding.kind {
                DescriptorBindingKind::Buffers => self.stats.miss_buffers += 1,
                DescriptorBindingKind::Image(_) => self.stats.miss_images += 1,
                DescriptorBindingKind::Samplers => self.stats.miss_samplers += 1,
                _ => {}
            }
            match binding.kind {
                DescriptorBindingKind::Buffers => {
                    writes.open(number, vk::DescriptorType::STORAGE_BUFFER, true);
                    // The ranges the reservation resolved (a draw's stages).
                    let owned = self.resolved.get_mut(stage_slot(stage)).and_then(|r| r.take());
                    let resolved = owned.as_ref();
                    let t_bind = crate::backend::profile::detailed().then(std::time::Instant::now);
                    // Read-only resources repeating the previous work's
                    // words reuse its binding (see `resource_memo_enabled`).
                    let per_resource = cacheable && resource_memo_enabled() && ctx.spec().is_none();
                    for (ri, &r) in binding.resources.iter().enumerate() {
                        if ring_mode {
                            if let Some(rb) = ring.get(ri).cloned().flatten() {
                                // Pin the whole buffer; the rotating base is
                                // the wide offset dword. `obtain` keeps the
                                // mirror current (a page-memo hit is no
                                // work, as on every memo-validated reuse).
                                let slice = crate::backend::profile::time(12, || ctx.buffers.obtain(ctx.sched, rb.a, rb.n, false))?;
                                if slice.buffer == rb.info.buffer {
                                    writes.buffer(rb.info);
                                    if let Some(d) = p.shader_data.get_mut(rb.dword as usize) {
                                        *d = rb.offset;
                                    }
                                    self.stats.ring_binds += 1;
                                    continue;
                                }
                                // A merge the plan could not see (the layout
                                // epoch bump invalidates the memo next
                                // draw): the exact binding below.
                            }
                        }
                        let words = snapshot.buffers.get(r as usize).map_or(&[][..], |w| &w[..]);
                        let v = BufferSharp::from_words(words);
                        let written = info.buffers.get(r as usize).is_some_and(|b| b.written || b.atomic);
                        let overwrite = info.buffers.get(r as usize).is_some_and(|b| b.written && !b.read && !b.atomic);
                        let range = match resolved.and_then(|m| m.get(r as usize)) {
                            Some(&resolved) => resolved,
                            None => self.clamp_range(v.address(), buffer_extent(&v, info.buffers.get(r as usize))).map(|(a, n)| (a, n, self.streams.contains(&(a, n)))),
                        };
                        let Some((addr, size, streamed)) = range else {
                            writes.buffer(vk::DescriptorBufferInfo { buffer: self.t.null_buffer.buffer, offset: 0, range: vk::WHOLE_SIZE });
                            continue;
                        };
                        if !written && streamed {
                            // A buffer read only by scalar loads at constant
                            // offsets is copied up to the last byte they reach.
                            let used = info.buffers.get(r as usize).filter(|b| b.constant_offsets && b.max_byte_extent > 0 && stream_trim_enabled());
                            let copied = used.map_or(size, |b| size.min((b.max_byte_extent as u64).next_multiple_of(16)));
                            self.stats.stream_trimmed += (copied < size) as u64;
                            if let Some(info) = self.stream_guest(ctx, addr, copied) {
                                if trace::active() {
                                    self.trace_notes.push(format!("    buffer {addr:#x}+{size:#x} streamed V# {words:08x?}{}", trace::guest_words(addr, size.min(0x60))));
                                }
                                writes.buffer(info);
                                continue;
                            }
                        }
                        let key: Option<[u32; 4]> = if per_resource && !written { words.try_into().ok() } else { None };
                        let reused = key.and_then(|k| {
                            let m = memo.buffer_memos.get(r as usize)?;
                            (m.generation != 0 && m.generation == memo.buffer_generation && m.words == k && m.range == (addr, size)).then_some(*m)
                        });
                        let (slice_buffer, aligned, adjustment) = match reused {
                            Some(m) => {
                                self.stats.resource_reuses += 1;
                                (m.info.buffer, m.info.offset, m.adjustment as u64)
                            }
                            None => {
                                // A dispatch whose stores are bounded writes (and makes
                                // GPU-dirty) only those bytes of the range it binds.
                                let limit = if written { self.write_limits.iter().find(|w| w.0 == r).map(|w| (w.1, w.2)) } else { None };
                                if written {
                                    let (a, n) = limit.unwrap_or((addr, size));
                                    self.writeback_before_buffer_write(ctx, a, n)?;
                                }
                                let slice = crate::backend::profile::time(12, || ctx.buffers.obtain(ctx.sched, addr, size, written && limit.is_none()))?;
                                if let Some((a, n)) = limit {
                                    crate::backend::profile::time(12, || ctx.buffers.obtain(ctx.sched, a, n, true))?;
                                    writes.written_ranges.push((a, n));
                                } else if written {
                                    writes.written_ranges.push((addr, size));
                                }
                                if overwrite {
                                    writes.overwritten.push((addr, size));
                                }
                                if trace::active() {
                                    self.trace_notes.push(format!("    buffer {addr:#x}+{size:#x} V# {words:08x?}{}{}", if written { " written" } else { "" }, trace::guest_words(addr, size.min(0x60))));
                                }
                                let aligned = slice.offset - slice.offset % align;
                                let adjustment = slice.offset - aligned;
                                if adjustment >= 256 {
                                    return Err(Failure::Error(format!("buffer {r} offset adjustment {adjustment} too large")));
                                }
                                if let Some(k) = key {
                                    self.stats.resource_binds += 1;
                                    if memo.buffer_memos.len() <= r as usize {
                                        memo.buffer_memos.resize(r as usize + 1, ResourceMemo::default());
                                    }
                                    memo.buffer_memos[r as usize] = ResourceMemo {
                                        generation: memo.buffer_generation,
                                        words: k,
                                        range: (addr, size),
                                        info: vk::DescriptorBufferInfo { buffer: slice.buffer, offset: aligned, range: size + adjustment },
                                        adjustment: adjustment as u32,
                                    };
                                }
                                (slice.buffer, aligned, adjustment)
                            }
                        };
                        if layout.memory_offset_count > r {
                            let (dword_ofs, shift, bits) = layout.memory_offset_slot(r as u32);
                            let dword = dword_ofs as usize;
                            if let Some(d) = p.shader_data.get_mut(dword) {
                                let bits = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
                                let v = (adjustment as u32) << shift;
                                *d = (*d & !(bits << shift)) | (v & (bits << shift));
                                adjust.push((dword as u16, v));
                            }
                        }
                        writes.buffer(vk::DescriptorBufferInfo { buffer: slice_buffer, offset: aligned, range: size + adjustment });
                    }
                    if let Some(slot) = self.resolved.get_mut(stage_slot(stage)) {
                        *slot = owned;
                    }
                    if let Some(t) = t_bind {
                        crate::backend::profile::add_local(41, t.elapsed().as_nanos() as u64);
                    }
                }
                DescriptorBindingKind::Image(_) => {
                    let (class, numeric, dimension, _) = image_binding::describe(binding.kind);
                    let storage = class == ImageResourceClass::Storage;
                    let ty = if storage { vk::DescriptorType::STORAGE_IMAGE } else { vk::DescriptorType::SAMPLED_IMAGE };
                    ty_used = ty;
                    writes.open(number, ty, false);
                    // Sampled cube bindings take cube views; repeated
                    // elements of one image are successive storage mips.
                    let cube = !storage && image_binding::is_cube(binding.kind) && sharps::CUBE_VIEWS;
                    let mut previous = None;
                    let mut mip_offset = 0;
                    for &r in &binding.resources {
                        mip_offset = if previous == Some(r) { mip_offset + 1 } else { 0 };
                        previous = Some(r);
                        let resource = info.images.get(r as usize);
                        let words = snapshot.images.get(r as usize).map_or(&[][..], |w| &w[..]);
                        let spec_swizzle = p.specialization.images.get(r as usize).map(|s| (s.shader_swizzle, s.conversion_format));
                        let host_swizzle = !storage && spec_swizzle.is_none_or(|(sw, conv)| conv == 0 && sw == ps5shader::resources::IDENTITY_IMAGE_SWIZZLE);
                        let written = resource.is_some_and(|i| i.written || i.atomic);
                        let single_level = resource.is_some_and(|i| i.mip_mode == ps5shader::resources::ImageMipMode::Dynamic) || (!storage && image_unnormalized(info, &snapshot, r));
                        let atomic64 = image_binding::is_atomic64(binding.kind);
                        let srgb = !storage && image_force_degamma(info, &snapshot, r);
                        let (view, id) =
                            crate::backend::profile::time(13, || self.image_view(ctx, words, numeric, dimension, storage, host_swizzle, cube, mip_offset, single_level, atomic64, srgb));
                        if let Some(id) = id {
                            writes.bound_images.push(id);
                            if written {
                                writes.written_images.push(id);
                                if storage {
                                    if let Some(keys) = compressed_store_keys(words, ctx.images.get(id).map(|i| &i.key)) {
                                        writes.compressed_stores.push(keys);
                                    }
                                }
                            }
                            let range = ctx.images.view_subresources(view);
                            let depth = ctx.images.get(id).is_some_and(|i| i.key.is_depth());
                            writes.image_slots.push(ImageSlot { index: writes.images.len(), id, storage, range, depth });
                            // An image to upload at its next use is looked
                            // up again (as the view memo does).
                            storable &= !ctx.images.needs_upload(id);
                        }
                        // Cached images get the layout of their use (see
                        // `image_layouts`); dummies stay in GENERAL.
                        writes.image(vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL });
                    }
                }
                DescriptorBindingKind::Samplers => {
                    ty_used = vk::DescriptorType::SAMPLER;
                    writes.open(number, vk::DescriptorType::SAMPLER, false);
                    for &r in &binding.resources {
                        let words = snapshot.samplers.get(r as usize).map_or(&[][..], |w| &w[..]);
                        let res = info.samplers.get(r as usize);
                        let integer = info.sampled_pairs.iter().any(|pair| {
                            pair.sampler == r
                                && info.images.get(pair.image as usize).is_some_and(|i| matches!(i.numeric_class, ImageNumericClass::Uint | ImageNumericClass::Sint))
                        });
                        let single_level = sampler_single_level(info, &snapshot, r);
                        let sampler_words = SamplerSharp::from_words(words);
                        let sampler_words = if single_level { sampler_words.for_single_level() } else { sampler_words };
                        let key = sampler_words.key(
                            res.is_some_and(|s| s.depth_compare),
                            integer,
                            res.is_some_and(|s| s.force_point_filtering),
                        );
                        let sampler = if self.speculative {
                            match self.t.samplers.find(&key) {
                                Some(s) => s,
                                None => return Err(self.bail(crate::slices::spec::Hole::Pipeline)),
                            }
                        } else {
                            self.t.samplers.w().get(&key)?
                        };
                        if trace::active() {
                            self.trace_notes.push(format!("    sampler S# {:08x?}{}", sampler_words.0, if integer { " (integer: point)" } else { "" }));
                        }
                        writes.image(vk::DescriptorImageInfo { sampler, image_view: vk::ImageView::null(), image_layout: vk::ImageLayout::UNDEFINED });
                    }
                }
                DescriptorBindingKind::GlobalDataShare => {
                    if ctx.bail(crate::slices::spec::Hole::Special) {
                        return Err(Failure::Bail);
                    }
                    writes.gds = true;
                    writes.single_buffer(number, vk::DescriptorBufferInfo { buffer: self.t.gds.buffer, offset: 0, range: vk::WHOLE_SIZE });
                }
                DescriptorBindingKind::DeviceAddressPageTable => {
                    // A stage reading guest memory translates through the
                    // import table, one reading at unbounded addresses
                    // through mirrors and guest memory, others through
                    // mirrors.
                    let unranged = crate::buffers::unranged_sync_enabled() && hazards::reads_unranged_memory(info, &snapshot.device_address_ranges);
                    let table = match info.buffers_in_guest_memory.then(|| ctx.buffers.bda_imports(ctx.sched)).flatten() {
                        Some(t) => t,
                        None => match unranged.then(|| ctx.buffers.cache().unranged_table()).flatten() {
                            Some(t) => t,
                            None => ctx.buffers.bda_page_table(ctx.sched)?.0,
                        },
                    };
                    writes.single_buffer(number, vk::DescriptorBufferInfo { buffer: table, offset: 0, range: vk::WHOLE_SIZE });
                }
                DescriptorBindingKind::FaultBuffer => {
                    writes.single_buffer(number, vk::DescriptorBufferInfo { buffer: self.t.fault.buffer, offset: 0, range: vk::WHOLE_SIZE });
                }
                DescriptorBindingKind::FlattenedResourceTable => {
                    // The table, zero-extended to its reserved size, straight
                    // into the staging ring (once per tick while unchanged).
                    let table: &[u32] = bindless_table.as_deref().unwrap_or(&snapshot.flattened_resource_table);
                    let reserved = p.program.plan.flattened_table_reserved_count().max(1).max(table.len());
                    let deferred = &snapshot.deferred_table_reads;
                    let reused = crate::backend::profile::dtime(49, || {
                        self.stats.table_compares += 1;
                        memo.table.as_ref().filter(|(w, n, _)| enabled && deferred.is_empty() && *n == reserved && w[..] == table[..]).map(|t| t.2)
                    });
                    if reused.is_some() {
                        self.stats.table_reuses += 1;
                    }
                    let info = match reused {
                        Some(info) => info,
                        None => {
                            self.stats.table_dwords += reserved as u64;
                            let info = crate::backend::profile::dtime(50, || self.upload_words_padded(ctx, table, reserved))?;
                            if deferred.is_empty() {
                                memo.table = Some((smallvec::SmallVec::from_slice(table), reserved, info));
                            } else {
                                memo.table = None;
                                if ctx.bail(crate::slices::spec::Hole::Frontend) {
                                    return Err(Failure::Bail);
                                }
                                self.fill_deferred_table_words(ctx, info, deferred)?;
                            }
                            info
                        }
                    };
                    writes.single_buffer(number, info);
                }
                DescriptorBindingKind::ShaderData => {
                    // Written after every buffer adjustment is known.
                }
            }
            if cacheable {
                if memo.bindings.len() <= bi {
                    memo.bindings.resize_with(bi + 1, Default::default);
                }
                let ways = &mut memo.bindings[bi];
                // A stream copy made outside this tick is not kept (the
                // memo of copies was cleared meanwhile).
                let streamed_ok = writes.buffers[buffers_at..].iter().all(|i| i.buffer != self.staging.buffer.buffer || self.stream_stamp.0 == tick);
                // Nor are images whose lookups changed the image cache.
                let valid = storable && streamed_ok && ctx.images.lookup_stamp() == lookups;
                if valid {
                    // The oldest way's storage is reused when all are taken.
                    let mut m = if ways.len() >= BINDING_WAYS { ways.pop().unwrap_or_default() } else { Box::default() };
                    m.valid = true;
                    m.words.clear();
                    m.words.extend_from_slice(&words);
                    m.ty = ty_used;
                    m.buffers.clear();
                    m.buffers.extend_from_slice(&writes.buffers[buffers_at..]);
                    m.images.clear();
                    m.images.extend_from_slice(&writes.images[images_at..]);
                    m.slots.clear();
                    m.slots.extend(writes.image_slots[slots_at..].iter().map(|s| ImageSlot { index: s.index - images_at, ..*s }));
                    m.generations.clear();
                    m.spec_reads = ctx.spec().zip(spec_mark).map(|(s, mark)| Box::new(s.since(mark)));
                    m.generations.extend(m.slots.iter().map(|s| ctx.images.image_generation(s.id).unwrap_or(u64::MAX)));
                    m.tick = tick;
                    m.adjust = adjust;
                    ways.insert(0, m);
                } else if let Some(last) = ways.get_mut(BINDING_WAYS - 1) {
                    // Nothing learned: the oldest way is dropped when all
                    // are taken (kept as storage for the next store; an
                    // invalid way is never found).
                    last.valid = false;
                }
            }
        }
        // A stage reading guest memory needs mirrors only where guest
        // memory has no import (the import table points at them).
        let guest = info.buffers_in_guest_memory && ctx.buffers.bda_active();
        if guest && bda_probe(p.program.address) && ctx.spec().is_none() {
            self.probe_stage(ctx, p.program.address, &snapshot, info);
        }
        if guest {
            self.stats.bda_stages += 1;
            self.stats.bda_materialized += info.buffers.len() as u64;
            // Mirrors serve ranges without an import, and large ranges in
            // device memory while the CPU leaves them alone (a mirror is
            // current once obtained; a CPU write points the pages back at
            // guest memory until the next obtain).
            let large = crate::buffers::guest::mirror_min_bytes();
            for (r, b) in info.buffers.iter().enumerate() {
                let v = BufferSharp::from_words(snapshot.buffers.get(r).map_or(&[][..], |w| &w[..]));
                let base = v.address();
                if base == 0 {
                    continue;
                }
                let extent = buffer_extent(&v, Some(b));
                ctx.buffers.bda_make_current(ctx.sched, base, extent.min(BDA_CURRENT_LIMIT));
                if ctx.buffers.bda_resident(base) {
                    continue;
                }
                let backed = ctx.buffers.bda_backed(base, 1);
                if backed && (extent < large || !ctx.buffers.bda_residency()) {
                    continue;
                }
                if let Some((addr, size)) = self.clamp_range(base, extent) {
                    if backed && size < large {
                        continue;
                    }
                    self.stats.bda_mirrored += 1;
                    crate::backend::profile::time(12, || ctx.buffers.obtain(ctx.sched, addr, size, b.written || b.atomic))?;
                }
            }
        }
        // Device-address ranges: their buffers must exist (page table).
        let ranges = &snapshot.device_address_ranges;
        if guest {
            let large = crate::buffers::guest::mirror_min_bytes();
            for r in ranges {
                if r.planned && r.size != 0 {
                    ctx.buffers.bda_make_current(ctx.sched, r.base, r.size.min(BDA_CURRENT_LIMIT));
                }
                let promote = r.size >= large && ctx.buffers.bda_residency();
                if r.planned && r.size != 0 && !ctx.buffers.bda_resident(r.base) && (promote || !ctx.buffers.bda_backed(r.base, 1)) {
                    if let Some((a, n)) = self.clamp_range(r.base, r.size) {
                        self.stats.bda_mirrored += 1;
                        crate::backend::profile::time(15, || ctx.buffers.obtain(ctx.sched, a, n, false))?;
                    }
                }
            }
        } else if buffers_cacheable && ctx.spec().is_none() && memo.ranges.as_ref().is_some_and(|r| r[..] == ranges[..]) {
            self.stats.binding_reuses += 1;
        } else {
            let mut any_written = false;
            // The previous work's ranges (all read-only, obtained under the
            // same stamps) need no second obtain.
            let previous = if buffers_cacheable && resource_memo_enabled() && ctx.spec().is_none() { memo.ranges.take() } else { None };
            for r in ranges {
                if r.planned && r.size != 0 {
                    if !r.written && previous.as_ref().is_some_and(|p| p.iter().any(|q| q.planned && q.base == r.base && q.size == r.size)) {
                        self.stats.resource_reuses += 1;
                        continue;
                    }
                    if let Some((a, n)) = self.clamp_range(r.base, r.size) {
                        if r.written {
                            self.writeback_before_buffer_write(ctx, a, n)?;
                        }
                        crate::backend::profile::time(15, || ctx.buffers.obtain(ctx.sched, a, n, r.written))?;
                        if r.written {
                            any_written = true;
                            writes.written_ranges.push((a, n));
                        }
                    }
                }
            }
            memo.ranges = (buffers_cacheable && !any_written).then(|| ranges.as_slice().into());
        }
        if let Some(binding) = layout.find(DescriptorBindingKind::ShaderData) {
            let data = &p.shader_data;
            // Shader data the GPU patches is a copy of its own, reused by
            // no other work.
            let patched = self.patch_shader_data == Some(stage);
            let reused = memo.shader_data.as_ref().filter(|(w, _)| enabled && !patched && w[..] == data[..]).map(|t| t.1);
            let info = match reused {
                Some(info) => info,
                None => {
                    let info = self.upload_words(ctx, data)?;
                    if patched {
                        self.patched_shader_data = Some(info);
                    } else {
                        memo.shader_data = Some((smallvec::SmallVec::from_slice(data), info));
                    }
                    info
                }
            };
            writes.single_buffer(native_binding_index(stage, binding.kind), info);
        }
        Ok(())
    }

    /// Device-address reads without a page-table entry since the last call
    /// (count and up to 8 distinct 64 KiB windows), then the buffer is reset.
    fn take_faults(&self) -> String {
        self.t.fault.invalidate(0, 4096);
        let b = self.t.fault.bytes_mut();
        let w = |i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        let n = w(0);
        if n == 0 {
            return String::new();
        }
        let mut windows: Vec<u64> = Vec::new();
        for k in 0..(n.min(crate::shader::memory::FAULT_SLOTS)) as usize {
            let a = w(2 + 2 * k) as u64 | (w(3 + 2 * k) as u64) << 32;
            let win = a & !0xFFFF;
            if !windows.contains(&win) && windows.len() < 8 {
                windows.push(win);
            }
        }
        b[..(2 + 2 * crate::shader::memory::FAULT_SLOTS as usize) * 4].fill(0);
        self.t.fault.flush(0, 4096);
        let mem = crate::memory::global();
        let described: Vec<String> = windows
            .iter()
            .map(|&w| format!("{w:#x}{}{}", if mem.backing_ptr(w, 4).is_some() { ":memfd" } else { "" }, if ps5vm::ranges::accessible(w, 4, false) { ":acc" } else { "" }))
            .collect();
        format!(" unmapped-reads={n} {described:?}")
    }

    fn probe_stage(&mut self, ctx: &mut DrawContext, address: u64, snapshot: &ps5shader::resources::ResourceSnapshot, info: &ps5shader::resources::ShaderResourceInfo) {
        static COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mem = crate::memory::global();
        // The first uses, then every 200th, and every use whose first V#
        // reads a zero word.
        let zero = info.buffers.first().is_some_and(|_| {
            let v = BufferSharp::from_words(snapshot.buffers.first().map_or(&[][..], |w| &w[..]));
            let mut b = [0u8; 4];
            mem.read_backing(v.address(), &mut b) && b == [0; 4]
        });
        // What the GPU reads there when it runs this point of the stream.
        if let Some(words) = snapshot.buffers.first() {
            let base = BufferSharp::from_words(words).address();
            let mut host = [0u8; 16];
            mem.read_backing(base, &mut host);
            ctx.buffers.bda_capture(ctx.sched, base, 16, move |gpu| {
                if gpu[..] != host[..] {
                    log::warn!("bda probe {address:#x} use {n}: {base:#x} recorded {host:x?} but the GPU read {gpu:x?}");
                }
            });
        }
        if !(n < 20 || n % 200 == 0 || zero) {
            return;
        }
        let mut line = format!("bda probe {address:#x} tick {}: table {:x?}", ctx.sched.current_tick(), &snapshot.flattened_resource_table[..snapshot.flattened_resource_table.len().min(16)]);
        for (r, _) in info.buffers.iter().enumerate() {
            let words = snapshot.buffers.get(r).map_or(&[][..], |w| &w[..]);
            let v = BufferSharp::from_words(words);
            let mut bytes = [0u8; 16];
            let ok = mem.read_backing(v.address(), &mut bytes);
            let w: Vec<u32> = bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            line += &format!(
                " | v#{r} {words:x?} base {:#x} fp {:#x} backed {} resident {} gpu-dirty {} read {ok} {w:x?}",
                v.address(),
                v.footprint(),
                ctx.buffers.bda_backed(v.address(), 4),
                ctx.buffers.bda_resident(v.address()),
                ctx.buffers.has_gpu_dirty(v.address(), 16)
            );
        }
        for r in &snapshot.device_address_ranges {
            line += &format!(" | range {:#x}+{:#x} planned {} backed {}", r.base, r.size, r.planned, ctx.buffers.bda_backed(r.base, 4));
        }
        log::info!("{line}");
    }

    /// Layouts of the cached images a draw or dispatch binds, as the
    /// reference renderer chooses them: an image also used as an
    /// attachment of the draw, or bound both sampled and as storage, goes to
    /// GENERAL whole (attachment, shader read and write accesses); other
    /// attachments to the colour or depth/stencil attachment layout, storage
    /// views to GENERAL and sampled views to the read-only layout of their
    /// aspect, each for the subresources it covers. Descriptor layouts in
    /// `writes` are set accordingly and the barriers appended to `out`.
    /// Returns a mask of the attachments (by position in `attachments`)
    /// left in GENERAL.
    /// `depth_read_only`: the draw writes neither depth nor stencil. Its
    /// depth attachment then also bound as a sampled texture stays in the
    /// read-only depth layout (both are reads: no feedback loop), and a
    /// draw that only tests depth keeps an attachment already in that
    /// layout there, so sampling and testing draws share one scope. The
    /// `stencil_split`: the draw writes stencil but not depth and samples
    /// its depth attachment only through depth-aspect views: the
    /// attachment, sampled, goes to the layout with read-only depth and a
    /// writable stencil aspect (both uses in it), and a draw that only
    /// tests depth and writes stencil keeps an attachment already there.
    /// The second value is the layout of the depth attachment.
    fn image_layouts(
        images: &mut crate::slices::spec::Images,
        writes: &mut SetWrites,
        attachments: &[(u64, bool, SubRange)],
        depth_read_only: bool,
        stencil_split: bool,
        out: &mut Vec<vk::ImageMemoryBarrier2<'static>>,
    ) -> (u32, vk::ImageLayout) {
        use crate::images::{COLOR_ATTACHMENT, DEPTH_ATTACHMENT, DEPTH_READ_ONLY, DEPTH_SAMPLED_STENCIL_WRITE, SAMPLED, SAMPLED_DEPTH, STORAGE};
        const SAMPLED_USE: u8 = 1;
        const STORAGE_USE: u8 = 2;
        const COLOR_TARGET: u8 = 4;
        const DEPTH_TARGET: u8 = 8;
        let mut uses: smallvec::SmallVec<[(u64, u8); 16]> = smallvec::SmallVec::new();
        let mut note = |id: u64, bit: u8| match uses.iter_mut().find(|u| u.0 == id) {
            Some(u) => u.1 |= bit,
            None => uses.push((id, bit)),
        };
        for s in &writes.image_slots {
            note(s.id, if s.storage { STORAGE_USE } else { SAMPLED_USE });
        }
        for &(id, depth, _) in attachments {
            note(id, if depth { DEPTH_TARGET } else { COLOR_TARGET });
        }
        // A sampled depth attachment of a draw writing no depth or stencil.
        let read_only_depth = |f: u8| depth_read_only && f & (SAMPLED_USE | STORAGE_USE | COLOR_TARGET | DEPTH_TARGET) == SAMPLED_USE | DEPTH_TARGET;
        // A sampled depth attachment of a draw writing only its stencil.
        let split_depth = |f: u8| stencil_split && f & (SAMPLED_USE | STORAGE_USE | COLOR_TARGET | DEPTH_TARGET) == SAMPLED_USE | DEPTH_TARGET;
        let general = |f: u8| {
            !read_only_depth(f) && !split_depth(f) && ((f & (COLOR_TARGET | DEPTH_TARGET) != 0 && f & (SAMPLED_USE | STORAGE_USE) != 0) || f & (SAMPLED_USE | STORAGE_USE) == SAMPLED_USE | STORAGE_USE)
        };
        for &(id, f) in &uses {
            if !general(f) {
                continue;
            }
            let mut access = vk::AccessFlags2::SHADER_READ;
            if f & STORAGE_USE != 0 {
                access |= vk::AccessFlags2::SHADER_WRITE;
            }
            if f & COLOR_TARGET != 0 {
                access |= COLOR_ATTACHMENT.1;
            }
            if f & DEPTH_TARGET != 0 {
                access |= DEPTH_ATTACHMENT.1;
            }
            images.transition(id, (vk::ImageLayout::GENERAL, access), None, out);
        }
        let flags = |id: u64| uses.iter().find(|u| u.0 == id).map_or(0, |u| u.1);
        let mut general_mask = 0u32;
        let mut depth_layout = DEPTH_ATTACHMENT.0;
        let mut split_id = None;
        for (i, &(id, depth, range)) in attachments.iter().enumerate() {
            if general(flags(id)) {
                general_mask |= 1 << i;
                continue;
            }
            let usage = if !depth {
                COLOR_ATTACHMENT
            } else if read_only_depth(flags(id)) || (depth_read_only && images.uniform_layout(id, range) == Some(DEPTH_READ_ONLY.0)) {
                depth_layout = DEPTH_READ_ONLY.0;
                DEPTH_READ_ONLY
            } else if split_depth(flags(id)) || (stencil_split && images.uniform_layout(id, range) == Some(DEPTH_SAMPLED_STENCIL_WRITE.0)) {
                depth_layout = DEPTH_SAMPLED_STENCIL_WRITE.0;
                split_id = Some(id);
                DEPTH_SAMPLED_STENCIL_WRITE
            } else {
                DEPTH_ATTACHMENT
            };
            images.transition(id, usage, Some(range), out);
        }
        for &ImageSlot { index, id, storage, range, depth } in &writes.image_slots {
            let info = &mut writes.images[index];
            if general(flags(id)) {
                info.image_layout = vk::ImageLayout::GENERAL;
                continue;
            }
            let (range, depth) = if slot_lookups_enabled() {
                (range, depth)
            } else {
                (images.view_subresources(info.image_view), images.get(id).is_some_and(|i| i.key.is_depth()))
            };
            let usage = if storage {
                STORAGE
            } else if split_id == Some(id) {
                DEPTH_SAMPLED_STENCIL_WRITE
            } else if depth {
                SAMPLED_DEPTH
            } else {
                SAMPLED
            };
            info.image_layout = usage.0;
            images.transition(id, usage, range, out);
        }
        (general_mask, depth_layout)
    }

    /// The view bound for one T# (a dummy when the image cannot be
    /// represented), and the cached image id when there is one.

    #[allow(clippy::too_many_arguments)]
    fn image_view(
        &mut self,
        ctx: &mut DrawContext,
        words: &[u32],
        numeric: ImageNumericClass,
        dimension: ImageDimension,
        storage: bool,
        host_swizzle: bool,
        cube: bool,
        mip_offset: u32,
        single_level: bool,
        atomic64: bool,
        srgb: bool,
    ) -> (vk::ImageView, Option<u64>) {
        let req = sharps::TextureRequest { numeric, dimension, storage, cube, host_swizzle, mip_offset, single_level, atomic64, srgb };
        // A T# looked up again while the image cache is unchanged gives the
        // same image and view.
        let memo_key = (words.len() == 8 && !trace::active() && view_memo_enabled()).then(|| {
            let mut w = [0u32; 8];
            w.copy_from_slice(words);
            ViewMemoKey { words: w, numeric, dimension, storage, cube, host_swizzle, mip_offset, single_level, atomic64, srgb }
        });
        let epoch = ctx.images.lookup_epoch();
        if self.view_memo_epoch != epoch {
            self.view_memo.clear();
            self.view_memo_epoch = epoch;
        }
        if let Some(k) = &memo_key {
            if let Some(&(view, id, tick, generation)) = self.view_memo.get(k) {
                let now = ctx.sched.current_tick();
                // The image, or one overlapping it, changed since; and once
                // per tick the image is looked at for CPU writes its watches
                // could not see.
                let current = ctx.images.image_generation(id) == Some(generation)
                    && (tick == now || !crate::images::image_generations_enabled() || !ctx.images.needs_upload(id));
                if current {
                    if tick != now {
                        ctx.images.touch(id, now);
                        self.view_memo.insert(*k, (view, id, now, generation));
                    }
                    if let crate::slices::spec::Images::Spec(cache, spec) = &ctx.images {
                        spec.note_image(cache, id);
                    }
                    self.stats.view_memo_hits += 1;
                    return (view, Some(id));
                }
                self.view_memo.remove(k);
            }
        }
        let lookups = ctx.images.lookup_stamp();
        self.stats.view_lookups += 1;
        let t = sharps::TextureSharp::from_words(words);
        let dummies = if self.speculative { None } else { Some(self.t.dummies.w()) };
        match sharps::texture_view(ctx, dummies, &self.dev, words, &req) {
            Ok(v) => {
                if let (Some(k), Some(id)) = (memo_key, v.1) {
                    // Only while the lookup changed nothing and the image
                    // needs no upload at its next use.
                    if ctx.images.lookup_stamp() == lookups && !ctx.images.needs_upload(id) {
                        if self.view_memo.len() >= 4096 {
                            self.view_memo.clear();
                        }
                        let generation = ctx.images.image_generation(id).unwrap_or(u64::MAX);
                        self.view_memo.insert(k, (v.0, id, ctx.sched.current_tick(), generation));
                    }
                }
                if trace::active() {
                    let note = format!(
                        "    image {:#x} fmt {} {}x{} {dimension:?}{} -> {:?} (depth {} sw {} levels {}) T# {:08x?}",
                        t.address(),
                        t.format(),
                        t.width(),
                        t.height(),
                        if storage { " storage" } else { "" },
                        v.1,
                        t.depth(),
                        t.swizzle(),
                        t.levels(),
                        t.0
                    );
                    self.trace_notes.push(note);
                    if let Some(id) = v.1 {
                        if !storage && (t.width() * t.height() <= 16 || (dimension == ImageDimension::Dim3D && t.width() * t.height() <= 64 * 64) || trace::dump_address(t.address()) || (trace::dump_arrays() && dimension == ImageDimension::Dim2DArray)) {
                            self.trace_sampled.push(id);
                        }
                    }
                }
                v
            }
            Err(_) if ctx.bailed() => (vk::ImageView::null(), None),
            Err(why) => {
                self.stats.dummy_images += 1;
                crate::log_once(&format!("ps5gpu: binding a dummy image: {why}"));
                if trace::active() {
                    self.trace_notes.push(format!("    image {:#x} -> DUMMY ({why})", t.address()));
                }
                let msaa = matches!(dimension, ImageDimension::Dim2DMsaa | ImageDimension::Dim2DMsaaArray);
                let ty = sharps::view_type(dimension, cube);
                let v = if self.speculative {
                    match self.t.dummies.find_image_view(ty, numeric, msaa, storage) {
                        Some(v) => v,
                        None => {
                            ctx.bail(crate::slices::spec::Hole::Pipeline);
                            vk::ImageView::null()
                        }
                    }
                } else {
                    self.t.dummies.w().image_view(ctx.sched, ty, numeric, msaa, storage).unwrap_or_default()
                };
                (v, None)
            }
        }
    }

    /// Push the descriptor writes and push data of `stages`.
    #[allow(clippy::too_many_arguments)]
    fn commit(
        &mut self,
        ctx: &mut DrawContext,
        cb: vk::CommandBuffer,
        bind_point: vk::PipelineBindPoint,
        layout: &LayoutEntry,
        push_stages: vk::ShaderStageFlags,
        writes: &SetWrites,
        stages: &[BoundStage],
        in_scope: bool,
    ) -> Result<(), Failure> {
        let mut push = [0u32; 32];
        let mut has_push = false;
        for s in stages {
            let l = &s.prepared.layout;
            if l.uses_push_data() {
                for (i, &w) in s.prepared.shader_data.iter().enumerate() {
                    if let Some(slot) = push.get_mut(l.push_data_start_dword as usize + i) {
                        *slot = w;
                    }
                }
                has_push = true;
            }
        }
        if has_push && !(in_scope && self.scope_state.same_push(layout.pipeline_layout, &push)) {
            // SAFETY: plain u32s viewed as their (little-endian host) bytes.
            let bytes = unsafe { std::slice::from_raw_parts(push.as_ptr() as *const u8, std::mem::size_of_val(&push)) };
            // SAFETY: recording buffer; the layout declares the range.
            unsafe { self.dev.device.cmd_push_constants(cb, layout.pipeline_layout, push_stages, 0, &bytes[..PUSH_CONSTANT_BYTES as usize]) };
        }
        if writes.entries.is_empty() {
            return Ok(());
        }
        if layout.push_descriptors {
            if in_scope && self.scope_state.same_descriptors(layout.pipeline_layout, &writes.entries, &writes.buffers, &writes.images) {
                return Ok(());
            }
            let w = writes.writes(vk::DescriptorSet::null());
            // SAFETY: recording buffer; writes match the push-descriptor set layout.
            unsafe { self.dev.push_descriptor.cmd_push_descriptor_set(cb, bind_point, layout.pipeline_layout, 0, &w) };
        } else {
            let set = self.heap.allocate(ctx.sched, layout.set_layout)?;
            self.stats.heap_sets += 1;
            if in_scope {
                self.scope_state.bound_set(layout.pipeline_layout);
            }
            let w = writes.writes(set);
            // SAFETY: fresh set of the layout; infos live through the call.
            unsafe {
                self.dev.device.update_descriptor_sets(&w, &[]);
                self.dev.device.cmd_bind_descriptor_sets(cb, bind_point, layout.pipeline_layout, 0, &[set], &[]);
            }
        }
        if layout.bindless {
            // After set 0: a set 0 of another layout disturbs it.
            let Some(Some(b)) = &self.bindless else { return Err(Failure::Error("bindless layout without the bindless set".into())) };
            // SAFETY: recording buffer; the layout's set 1 is the bindless set's layout.
            unsafe { self.dev.device.cmd_bind_descriptor_sets(cb, bind_point, layout.pipeline_layout, ps5shader::resources::bindless::SET, &[b.set], &[]) };
        }
        Ok(())
    }

    /// The pipeline layout of a draw's stages (memoised by the stages'
    /// shared binding layouts, which the memo keeps alive).
    /// Counters of the bindless set (None before it is made).
    pub(crate) fn bindless_stats(&self) -> Option<super::bindless::BindlessStats> {
        self.bindless.as_ref().and_then(|s| s.as_ref()).map(|s| s.stats)
    }

    /// The images the bindless set's slots hold views of.
    #[cfg(test)]
    pub(crate) fn bindless_held_images(&self) -> Vec<u64> {
        self.bindless.as_ref().and_then(|s| s.as_ref()).map(|s| s.held_images()).unwrap_or_default()
    }

    /// The bindless set, made on first use.
    fn bindless_set(&mut self) -> Result<&mut super::bindless::BindlessSet, Failure> {
        if self.speculative {
            // The set pins images in the cache: the committing thread's.
            return Err(self.bail(crate::slices::spec::Hole::Special));
        }
        if self.bindless.is_none() {
            let set = match super::bindless::BindlessSet::new(&self.dev) {
                Ok(s) => Some(s),
                Err(e) => {
                    log::warn!("ps5gpu: no bindless set: {e:?}");
                    None
                }
            };
            if let Some(s) = &set {
                self.t.layouts.w().bindless = Some(s.layout);
            }
            self.bindless = Some(set);
        }
        self.bindless.as_mut().and_then(|s| s.as_mut()).ok_or_else(|| Failure::Error("the device has no bindless set (descriptor indexing)".into()))
    }

    /// The flattened table of a stage with the words of its bindless
    /// tables (None: it has none). Their records and images are brought up
    /// to date first.
    fn bindless_tables(
        &mut self,
        ctx: &mut DrawContext,
        info: &ps5shader::resources::ShaderResourceInfo,
        snapshot: &ps5shader::resources::ResourceSnapshot,
    ) -> Result<Option<smallvec::SmallVec<[u32; 64]>>, Failure> {
        if snapshot.bindless_tables.is_empty() {
            return Ok(None);
        }
        self.bindless_set()?;
        let memory = self.memory;
        let Some(Some(set)) = self.bindless.as_mut() else { unreachable!() };
        let mut table = smallvec::SmallVec::from_slice(&snapshot.flattened_resource_table);
        for t in &snapshot.bindless_tables {
            let kind = info.images.get(t.image as usize).and_then(image_binding::for_image).ok_or_else(|| Failure::Error(format!("bindless image {} has no binding", t.image)))?;
            let words = crate::backend::profile::time(13, || set.bind(ctx, self.t.dummies.w(), memory, t, kind)).map_err(Failure::Error)?;
            let at = t.mapping_offset as usize;
            table.get_mut(at..at + words.len()).ok_or_else(|| Failure::Error("bindless table words past the flattened table".into()))?.copy_from_slice(&words);
        }
        Ok(Some(table))
    }

    fn draw_layout(&mut self, bound: &[BoundStage], mesh: bool, push_stages: vk::ShaderStageFlags) -> Result<LayoutEntry, Failure> {
        let vs = &bound[0].prepared.layout;
        let ps = bound.get(1).map(|b| &b.prepared.layout);
        let key = (Arc::as_ptr(vs) as usize, ps.map_or(0, |p| Arc::as_ptr(p) as usize), mesh, push_stages.as_raw());
        if let Some((_, _, e)) = self.layout_memo.get(&key) {
            return Ok(*e);
        }
        if bound.iter().any(|b| b.prepared.layout.bindless_tables) {
            self.bindless_set()?;
        }
        let layout_bindings: Vec<_> = bound.iter().map(|b| (b.stage, &*b.prepared.layout)).collect();
        let mut binding_descs = set_bindings(&layout_bindings);
        if mesh {
            for b in binding_descs.iter_mut() {
                if b.stages == vk::ShaderStageFlags::VERTEX.as_raw() {
                    b.stages = vk::ShaderStageFlags::MESH_EXT.as_raw();
                }
            }
        }
        let entry = if self.speculative {
            match self.t.layouts.find(&binding_descs, push_stages) {
                Some(e) => e,
                None => return Err(self.bail(crate::slices::spec::Hole::Pipeline)),
            }
        } else {
            let entry = self.t.layouts.w().get(&binding_descs, push_stages)?;
            self.t.layout_descs.w().entry(entry.pipeline_layout).or_insert_with(|| (binding_descs.clone(), push_stages.as_raw()));
            entry
        };
        if self.layout_memo.len() >= 1 << 14 {
            self.layout_memo.clear();
        }
        self.layout_memo.insert(key, (vs.clone(), ps.cloned(), entry));
        Ok(entry)
    }

    // ------------------------------------------------------------------
    // Pipelines
    // ------------------------------------------------------------------

    /// The graphics pipeline of `key` (created on a compile worker when
    /// there are any: `wait` blocks until it is ready, else a missing
    /// pipeline is [`Failure::Compiling`]).
    #[allow(clippy::too_many_arguments)]
    fn graphics_pipeline(
        &mut self,
        key: GraphicsKey,
        vs: &Arc<Module>,
        ps: Option<&Arc<Module>>,
        tess: Option<(vk::ShaderModule, vk::ShaderModule)>,
        mesh: bool,
        wait: bool,
    ) -> Result<vk::Pipeline, Failure> {
        if let Some(&p) = self.t.graphics.get(&key) {
            self.stats.graphics_hits += 1;
            crate::perf::PIPELINE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(p);
        }
        if self.t.failed_graphics.contains(&key) {
            return Err(Failure::Error("graphics pipeline creation failed".into()));
        }
        if self.speculative {
            return Err(self.bail(crate::slices::spec::Hole::Pipeline));
        }
        let record = self.t.layout_descs.get(&key.layout).map(|(layout, push)| super::prewarm::GraphicsRecord {
            vs: key.vs,
            ps: key.ps,
            layout: layout.clone(),
            push_stages: *push,
            ff: key.ff.clone(),
            mesh,
        });
        let job = graphics_job(self.dev.clone(), self.t.disk.cache, key.clone(), vs.clone(), ps.cloned(), tess, mesh, record, self.fast_builds());
        let Some(pool) = &*self.t.pool else {
            self.finish(job());
            return self.t.graphics.get(&key).copied().ok_or_else(|| Failure::Error("graphics pipeline creation failed".into()));
        };
        if self.t.pending_graphics.w().insert(key.clone()) {
            pool.submit_urgent(job, hash_of(&key));
        } else {
            pool.hurry(hash_of(&key));
        }
        if !wait {
            // A pipeline the pipeline cache already holds builds in a
            // fraction of a millisecond: wait within the frame's budget
            // before skipping the draw (a skipped pass leaves a dark area
            // that temporal filters fade out over several frames).
            if self.wait_within_budget(|s| s.t.graphics.contains_key(&key) || s.t.failed_graphics.contains(&key))? {
                if let Some(&p) = self.t.graphics.get(&key) {
                    return Ok(p);
                }
                return Err(Failure::Error("graphics pipeline creation failed".into()));
            }
            return Err(Failure::Compiling);
        }
        self.stats.compile_waits += 1;
        let t = std::time::Instant::now();
        self.run_queued(hash_of(&key));
        loop {
            if let Some(&p) = self.t.graphics.get(&key) {
                self.note_blocking_wait(t, || format!("graphics pipeline vs {:016x} ps {:016x}", key.vs, key.ps));
                return Ok(p);
            }
            if self.t.failed_graphics.contains(&key) {
                return Err(Failure::Error("graphics pipeline creation failed".into()));
            }
            self.wait_compile_step(t)?;
        }
    }

    /// The compute pipeline of `key` (dispatches wait for it: later work
    /// may depend on them).
    fn compute_pipeline(&mut self, key: ComputeKey, cs: &Arc<Module>) -> Result<vk::Pipeline, Failure> {
        if let Some(&p) = self.t.compute.get(&key) {
            self.stats.compute_hits += 1;
            crate::perf::PIPELINE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(p);
        }
        if self.t.failed_compute.contains(&key) {
            return Err(Failure::Error("compute pipeline creation failed".into()));
        }
        let record = self.t.layout_descs.get(&key.layout).map(|(layout, _)| super::prewarm::ComputeRecord { cs: key.cs, layout: layout.clone(), subgroup32: key.subgroup32 });
        let job = compute_job(self.dev.clone(), self.t.disk.cache, key, cs.clone(), record, self.fast_builds());
        let t = std::time::Instant::now();
        match &*self.t.pool {
            None => self.finish(job()),
            Some(pool) => {
                if self.t.pending_compute.w().insert(key) {
                    pool.submit_urgent(job, hash_of(&key));
                } else {
                    pool.hurry(hash_of(&key));
                }
                if self.ahead.queue_only {
                    return Err(Failure::Compiling);
                }
                self.stats.compile_waits += 1;
                self.run_queued(hash_of(&key));
            }
        }
        loop {
            if let Some(&p) = self.t.compute.get(&key) {
                self.note_blocking_wait(t, || format!("compute pipeline cs {:016x}", key.cs));
                return Ok(p);
            }
            if self.t.failed_compute.contains(&key) {
                return Err(Failure::Error("compute pipeline creation failed".into()));
            }
            self.wait_compile_step(t)?;
        }
    }

    // ------------------------------------------------------------------
    // Draws
    // ------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn prepare_stage(
        &mut self,
        ctx: &mut DrawContext,
        stage: ShaderStage,
        address: u64,
        base: u32,
        user_data: &[u32],
        count: u32,
        wave: u32,
        cursor: u32,
        pre: Prefetched<'_>,
    ) -> Result<PreparedStage, Failure> {
        self.prepare_stage_limited(ctx, stage, address, base, user_data, count, wave, cursor, None, pre)
    }

    /// [`Self::prepare_stage`] with the thread limits of a compute dispatch
    /// whose last groups are partial.
    #[allow(clippy::too_many_arguments)]
    fn prepare_stage_limited(
        &mut self,
        ctx: &mut DrawContext,
        stage: ShaderStage,
        address: u64,
        base: u32,
        user_data: &[u32],
        count: u32,
        wave: u32,
        cursor: u32,
        thread_limits: Option<[u32; 3]>,
        mut pre: Prefetched<'_>,
    ) -> Result<PreparedStage, Failure> {
        crate::backend::profile::time(9, || {
            // The program is decoded and prepared under one lock; the
            // failure paths (which need `&mut self`) run after the guard
            // is gone.
            let mut f = self.frontend.lock().unwrap_or_else(|e| e.into_inner());
            let program = match f.program_for(address, stage, base, count, wave, user_data) {
                Ok(p) => p,
                Err(e) => {
                    drop(f);
                    return Err(self.frontend_failure(stage, address, e));
                }
            };
            if let Some(p) = pre.take(stage, &program, user_data, cursor, thread_limits, &ctx.buffers.tracker(), self.memory) {
                self.stats.prefetched_stages += 1;
                return Ok(p);
            }
            let r = f.prepare(&program, user_data, cursor, thread_limits, ctx.sched, &mut ctx.buffers);
            drop(f);
            r.map_err(|e| self.frontend_failure(stage, address, e))
        })
    }

    /// Give the pixel stage `p` the per-pixel sample masks `masks` as
    /// shader data: its layout gains the mask words (one extended layout
    /// per plain layout, so layout caches keyed by identity keep hitting)
    /// and its shader data holds this draw's masks.
    fn with_quad_masks(&mut self, p: &mut PreparedStage, masks: [u32; 4]) {
        if !p.layout.uses_quad_masks() {
            let plain = p.resource_key;
            if let Some((layout, key)) = self.quad_layouts.get(&plain) {
                p.layout = layout.clone();
                p.resource_key = *key;
            } else {
                Arc::make_mut(&mut p.layout).add_quad_masks();
                p.refresh_resource_key();
                if self.quad_layouts.len() > 4096 {
                    self.quad_layouts.clear();
                }
                self.quad_layouts.insert(plain, (p.layout.clone(), p.resource_key));
            }
        }
        let at = p.layout.quad_masks_dword as usize;
        let n = p.layout.shader_data_dword_count() as usize;
        if p.shader_data.len() < n {
            p.shader_data.resize(n, 0);
        }
        p.shader_data[at..at + 2].copy_from_slice(&pack_quad_masks(masks));
    }

    fn frontend_failure(&mut self, stage: ShaderStage, address: u64, e: FrontendError) -> Failure {
        if self.speculative {
            return self.bail(crate::slices::spec::Hole::Frontend);
        }
        self.stats.frontend_failures += 1;
        crate::log_once(&format!("ps5gpu: {stage:?} program {address:#x}: {e}"));
        Failure::NeedsShader
    }

    /// The fixed-function part of a draw's pipeline key.
    #[allow(clippy::too_many_arguments)]
    fn fixed_function(
        &self,
        d: &ResolvedDraw,
        vertex_inputs: &VertexInputs,
        rect: RectMode,
        restart: bool,
        mesh: bool,
        viewport_count: u32,
        colors: ([i32; MAX_COLOR_ATTACHMENTS], u32, [state::BlendKey; MAX_COLOR_ATTACHMENTS]),
        depth: (vk::Format, vk::Format),
        samples: u32,
        alpha_to_coverage: bool,
    ) -> Result<FixedFunctionKey, Failure> {
        let topology = if mesh { vk::PrimitiveTopology::TRIANGLE_LIST } else { state::host_topology(d.primitive_type, rect).ok_or(Failure::Skip)? };
        let (depth_clip, depth_clamp) = state::depth_clip(d.raster.clip_cntl);
        let has_depth = depth.0 != vk::Format::UNDEFINED || depth.1 != vk::Format::UNDEFINED;
        let centered_samples = match state::sample_placement(samples, d.msaa_enable, topology, has_depth, alpha_to_coverage) {
            state::SamplePlacement::Standard => false,
            state::SamplePlacement::Centered if self.dev.caps.sample_location_counts.contains(samples_flag(samples)) => true,
            state::SamplePlacement::Centered => {
                crate::log_once(&format!("ps5gpu: {samples}x target without MSAA_ENABLE drawn at the standard sample locations (no VK_EXT_sample_locations for {samples} samples)"));
                false
            }
            state::SamplePlacement::CenteredUnsupported(why) => {
                crate::log_once(&format!("ps5gpu: {samples}x target without MSAA_ENABLE drawn at the standard sample locations ({why})"));
                false
            }
        };
        Ok(FixedFunctionKey {
            topology: topology.as_raw(),
            rect,
            primitive_restart: restart && matches!(topology, vk::PrimitiveTopology::LINE_STRIP | vk::PrimitiveTopology::TRIANGLE_STRIP | vk::PrimitiveTopology::TRIANGLE_FAN),
            vertex_input: VertexInputKey {
                bindings: vertex_inputs.buffers.iter().map(|b| VertexBindingKey { stride: b.stride, per_instance: b.per_instance }).collect(),
                attributes: vertex_inputs
                    .attributes
                    .iter()
                    .map(|a| VertexAttributeKey { location: a.location, binding: a.binding, format: a.format.as_raw(), offset: a.offset })
                    .collect(),
            },
            polygon_mode: if rect == RectMode::FillRectangle {
                vk::PolygonMode::FILL_RECTANGLE_NV.as_raw()
            } else {
                state::polygon_mode(d.raster.mode_cntl, self.dev.caps.fill_mode_non_solid, self.dev.caps.maintenance5).as_raw()
            },
            depth_clip,
            depth_clamp,
            negative_one_to_one: !d.raster.dx_clip_space,
            color_formats: colors.0,
            color_count: colors.1,
            blend: colors.2,
            depth_format: depth.0.as_raw(),
            stencil_format: depth.1.as_raw(),
            samples,
            centered_samples,
            sample_shading: samples > 1 && d.ps_iter_sample && ps_iter_sample_enabled() && self.dev.caps.sample_rate_shading,
            sample_mask: static_sample_mask(d, samples),
            logic_op: None,
            viewport_count,
            provoking_last: self.provoking_last(d),
        })
    }

    /// PROVOKING_VTX_LAST of a draw as the device can draw it (the first
    /// vertex provokes without VK_EXT_provoking_vertex).
    fn provoking_last(&self, d: &ResolvedDraw) -> bool {
        if !d.raster.provoking_vertex_last {
            return false;
        }
        if !self.dev.caps.provoking_vertex_last {
            crate::log_once("ps5gpu: PROVOKING_VTX_LAST without VK_EXT_provoking_vertex provokingVertexLast: flat attributes come from the first vertex");
            return false;
        }
        true
    }

    fn try_draw(&mut self, ctx: &mut DrawContext, d: &ResolvedDraw, compiler: Option<&Arc<dyn ShaderCompiler>>) -> Result<DrawOutcome, Failure> {
        if d.topology.is_none() || d.packet.count == 0 {
            self.skip_why = if d.topology.is_none() { "no topology" } else { "zero count" };
            return Err(Failure::Skip);
        }
        let mut lap = crate::backend::profile::Lap::start();
        // The interpreter thread pre-materialised this draw's stages
        //; an unmatched or absent prefetch prepares here.
        let mut pre = crate::backend::draw_prefetch::take();
        let vs_stage = d.shaders.vertex_program();
        if !vs_stage.bound() {
            self.skip_why = "vertex program not bound";
            return Err(Failure::Skip);
        }
        // A real geometry stage (ES and GS halves sharing LDS) runs each
        // subgroup as a mesh workgroup.
        let geometry = d.shaders.stages & STAGES_GS_EN != 0;
        let mesh_capable = self.dev.caps.mesh_shader && self.dev.mesh.is_some();
        if geometry && !mesh_capable {
            return Err(Failure::NeedsShader);
        }
        let merged = d.shaders.es.bound() || d.shaders.gs.bound();
        let vs_count = vs_stage.user_sgpr_count().min(32);
        // The subgroup program: the hardware runs it with s[0:1] = the
        // user-data address and the user SGPRs from s8.
        let wave = if d.shaders.stages & STAGES_GS_W32_EN != 0 { 32 } else { 64 };
        let subgroup_program = |this: &mut Self| {
            let continuation = if d.shaders.gs.bound() && d.shaders.gs.address != vs_stage.address { d.shaders.gs.address } else { 0 };
            let r = {
                let mut f = this.frontend.lock().unwrap_or_else(|e| e.into_inner());
                f.program_with(vs_stage.address, continuation, ShaderStage::Vertex, 0, 8 + vs_count, wave)
            };
            r.map_err(|e| this.frontend_failure(ShaderStage::Vertex, vs_stage.address, e))
        };
        // Primitive subgroups without a geometry stage (NGG vertex programs)
        // that use LDS exchange vertex data between lanes through it (culling
        // and compaction): they run as mesh workgroups too, so their LDS is
        // shared per subgroup instead of a private copy per vertex.
        let mut shape = None;
        let mut program = None;
        if geometry {
            program = Some(subgroup_program(self)?);
        } else if merged && mesh_capable && primitive_subgroups_enabled() {
            let p = subgroup_program(self)?;
            if p.uses_local_data_share && !primitive_subgroup_stages(d.shaders.stages) {
                crate::log_once(&format!("ps5gpu: primitive subgroup {:#x} kept per vertex: stages {:#x}", vs_stage.address, d.shaders.stages));
            }
            if p.uses_local_data_share && primitive_subgroup_stages(d.shaders.stages) {
                match self.mesh_config(d, wave, 0, false) {
                    Ok(m) => {
                        if self.subgroup_programs.insert(vs_stage.address) {
                            log::info!(
                                "ps5gpu: primitive subgroup {:#x} as mesh workgroups: {m:?}, LDS {} dwords",
                                vs_stage.address,
                                ((d.shaders.gs.rsrc2 >> 19) & 0xFF) * 128
                            );
                        }
                        shape = Some(m);
                        program = Some(p);
                    }
                    Err(m) => crate::log_once(&format!("ps5gpu: primitive subgroup kept per vertex: {m}")),
                }
            }
        }
        let mesh_draw = program.is_some();
        let gpu_indirect = self.gpu_indirect;
        if let Some(g) = gpu_indirect {
            if !(if mesh_draw { g.mesh } else { g.host }) {
                self.skip_why = if mesh_draw { "mesh draw" } else { "vertex path" };
                return Err(Failure::NoGpuIndirect);
            }
        }
        // A host indirect draw (vertex path) / a mesh launch from the record.
        let host_indirect = gpu_indirect.filter(|_| !mesh_draw);
        let mesh_indirect = gpu_indirect.filter(|_| mesh_draw);
        let vs_prepared = if let Some(program) = program {
            // s[0:1] hold the user-data address, the user SGPRs follow s8.
            let mut user = [0u32; 8 + 32];
            let user = &mut user[..8 + vs_count as usize];
            let a = d.shaders.geometry.user_data_address;
            user[0] = a as u32;
            user[1] = (a >> 32) as u32;
            user[8..].copy_from_slice(&vs_stage.user_data[..vs_count as usize]);
            // The guard is scoped: the probe closure below re-locks the
            // front end, and the same thread locking one mutex twice
            // deadlocks it.
            let mut p = {
                let mut f = self.frontend.lock().unwrap_or_else(|e| e.into_inner());
                let r = f.prepare(&program, user, MESH_SHADER_DATA_CURSOR, None, ctx.sched, &mut ctx.buffers);
                drop(f);
                r.map_err(|e| self.frontend_failure(ShaderStage::Vertex, vs_stage.address, e))?
            };
            if let Some(g) = mesh_indirect {
                // The record's offsets are patched into the shader data:
                // nothing else of the stage may depend on them.
                let independent = self.indirect_resources_independent(ctx, &p, vs_stage.address, user, 8, g.sgprs, true, |this, ctx, probe| {
                    let mut f = this.frontend.lock().unwrap_or_else(|e| e.into_inner());
                    f.prepare(&program, probe, MESH_SHADER_DATA_CURSOR, None, ctx.sched, &mut ctx.buffers).map_err(|_| Failure::Skip)
                });
                if !independent || p.layout.uses_push_data() {
                    self.skip_why = if independent { "mesh shader data in push constants" } else { "resources read the offsets" };
                    return Err(Failure::NoGpuIndirect);
                }
            }
            // The entry point reads the index buffer through device
            // addresses (the extended layout is kept per stage layout).
            let plain = p.resource_key;
            if let Some((layout, key)) = self.mesh_layouts.get(&plain) {
                p.layout = layout.clone();
                p.resource_key = *key;
            } else {
                let layout = Arc::make_mut(&mut p.layout);
                if layout.find(DescriptorBindingKind::DeviceAddressPageTable).is_none() {
                    for kind in [DescriptorBindingKind::DeviceAddressPageTable, DescriptorBindingKind::FaultBuffer] {
                        layout.descriptors.push(ps5shader::resources::DescriptorBinding { kind, resources: vec![] });
                    }
                }
                if layout.find(DescriptorBindingKind::ShaderData).is_none() {
                    layout.descriptors.push(ps5shader::resources::DescriptorBinding { kind: DescriptorBindingKind::ShaderData, resources: vec![] });
                }
                p.refresh_resource_key();
                if self.mesh_layouts.len() > 4096 {
                    self.mesh_layouts.clear();
                }
                self.mesh_layouts.insert(plain, (p.layout.clone(), p.resource_key));
            }
            p
        } else {
            let vs_base = if merged { 8 } else { 0 };
            let vs_wave = if vertex_guest_wave_enabled() { wave } else { 32 };
            let p = self.prepare_stage(ctx, ShaderStage::Vertex, vs_stage.address, vs_base, &vs_stage.user_data, vs_count, vs_wave, 0, Prefetched::Draw(&mut pre))?;
            if let Some(g) = host_indirect {
                let user = &vs_stage.user_data[..vs_stage.user_data.len().min(32)];
                let independent = self.indirect_resources_independent(ctx, &p, vs_stage.address, user, 0, g.sgprs, false, |this, ctx, probe| {
                    this.prepare_stage(ctx, ShaderStage::Vertex, vs_stage.address, vs_base, probe, vs_count, vs_wave, 0, Prefetched::None)
                });
                if !independent {
                    self.skip_why = "resources read the offsets";
                    return Err(Failure::NoGpuIndirect);
                }
            }
            p
        };
        let mut cursor = 0;
        vs_prepared.layout.advance_push_data(&mut cursor);
        let ps_stage = d.shaders.ps;
        // The pixel shader runs when colour targets are bound or it has
        // effects of its own (depth, coverage or mask exports, kills, work
        // without targets); a depth-only draw rasterises without it.
        let ps_active = ps_stage.bound() && (!ctx.targets.colors.is_empty() || d.pixel_side_effects());
        let mut ps_prepared = if ps_active {
            let count = ps_stage.user_sgpr_count().min(32);
            let wave = super::frontend::pixel_wave_size(d.ps_in_control);
            Some(self.prepare_stage(ctx, ShaderStage::Pixel, ps_stage.address, 0, &ps_stage.user_data, count, wave, cursor, Prefetched::Draw(&mut pre))?)
        } else {
            None
        };
        let Some(compiler) = compiler else { return Err(Failure::NeedsShader) };
        let targets = ctx.targets;
        // Without targets only a pixel shader with effects of its own (UAV
        // writes behind EXEC_ON_NOOP, kills, depth or mask exports) does
        // anything: it runs in a rendering scope without attachments.
        let attachmentless = targets.colors.is_empty() && targets.depth.is_none();
        if attachmentless && !(ps_prepared.is_some() && d.pixel_side_effects()) {
            self.skip_why = "no targets, no pixel side effects";
            return Err(Failure::Skip);
        }
        let mesh = if mesh_draw {
            let draw_dword = vs_prepared.shader_data.len() as u32;
            let m = match shape {
                Some(m) => Ok(m),
                None => self.mesh_config(d, vs_prepared.program.wave_size, draw_dword, true),
            };
            match m {
                Ok(m) => Some(MeshConfig { draw_dword, ..m }),
                Err(m) => {
                    crate::log_once(&format!("ps5gpu: geometry subgroup not supported: {m}"));
                    return Err(Failure::NeedsShader);
                }
            }
        } else {
            None
        };

        // ---- vertex input ----
        let vertex_inputs = match self.vertex_inputs.as_mut() {
            Some(r) if mesh.is_none() => r.resolve(d, &vs_prepared).unwrap_or_default(),
            _ => VertexInputs::default(),
        };
        if gpu_indirect.is_some() && !vertex_inputs.buffers.is_empty() {
            self.skip_why = "vertex buffers";
            return Err(Failure::NoGpuIndirect);
        }
        let index = d.index.filter(|_| d.packet.kind.indexed());
        let restart = d.primitive_restart.then_some(d.restart_index);

        lap.lap(23);
        // ---- modules ----
        let clip_space = state::clip_space_transform(d, &self.dev.props.limits);
        let mut user_clip = state::user_clip_planes(d, user_clip_planes_enabled());
        if user_clip.mask != 0 && mesh.is_some() {
            crate::log_once("ps5gpu: user clip planes of a geometry subgroup are not applied");
            user_clip = Default::default();
        }
        let vs_out_cntl = d.raster.vs_out_cntl;
        let indirect_sgprs = host_indirect.map(|g| [g.sgprs[0], g.sgprs[1]]);
        let gs_lds = ((d.shaders.gs.rsrc2 >> 19) & 0xFF) * 128;
        let geometry_waves = d.shaders.gs.bound() || d.shaders.es.bound();
        let vs_options: Options = Arc::new(move |r: &mut ShaderCompileRequest<'_>| {
            r.position_export_control = vs_out_cntl;
            r.clip_space = clip_space;
            r.user_clip_planes = user_clip;
            r.indirect_draw = indirect_sgprs;
            if let Some(m) = &mesh {
                r.local_size = [m.threads, 1, 1];
                r.local_data_share_dwords = gs_lds;
            } else if geometry_waves {
                // The geometry wave's LDS allocation bounds the addresses
                // its (per-invocation) LDS copy must cover.
                r.local_data_share_dwords = gs_lds;
            }
        });
        // A draw that writes memory (or runs without attachments) waits for
        // its pipeline; others are skipped while it compiles.
        let writes_memory = |p: &PreparedStage| {
            let info = &p.resources.info;
            info.buffers.iter().any(|b| b.written || b.atomic)
                || info.images.iter().any(|i| i.written || i.atomic)
                || p.snapshot.device_address_ranges.iter().any(|r| r.written)
                || p.layout.find(DescriptorBindingKind::GlobalDataShare).is_some()
        };
        let wait = attachmentless || writes_memory(&vs_prepared) || ps_prepared.as_ref().is_some_and(writes_memory);
        let lds_alloc = (d.shaders.gs.rsrc2 >> 19) & 0xFF;
        let clip_words = [
            clip_space.enabled as u32,
            clip_space.scale_x.to_bits(),
            clip_space.scale_y.to_bits(),
            clip_space.offset_x.to_bits(),
            clip_space.offset_y.to_bits(),
            clip_space.half_extent_x.to_bits(),
            clip_space.half_extent_y.to_bits(),
        ];
        let vs_options_key = || {
            // Without planes the key is the one from before they existed.
            if user_clip.mask == 0 {
                hash_of(&(mesh, lds_alloc, vs_out_cntl, clip_words, indirect_sgprs))
            } else {
                hash_of(&(mesh, lds_alloc, vs_out_cntl, clip_words, indirect_sgprs, user_clip))
            }
        };
        let indirect_word = indirect_sgprs.map_or(0, |r| 1 << 16 | (r[1] as u32) << 8 | r[0] as u32);
        let mut vs_raw = [0u32; VS_RAW_WORDS];
        vs_raw[..10].copy_from_slice(&[lds_alloc, vs_out_cntl, clip_words[0], clip_words[1], clip_words[2], clip_words[3], clip_words[4], clip_words[5], clip_words[6], indirect_word]);
        vs_raw[10] = user_clip.mask;
        vs_raw[11..].copy_from_slice(&user_clip.planes);
        let mut outputs: smallvec::SmallVec<[PixelOutputBinding; MAX_COLOR_ATTACHMENTS]> = smallvec::SmallVec::new();
        for (t, _) in &targets.colors {
            let kind = match t.number_type {
                4 => PixelOutputKind::Uint,
                5 => PixelOutputKind::Sint,
                _ => PixelOutputKind::Float,
            };
            let packed = t.format.guest == crate::render::formats::GuestFormat::R11G11B10Unorm;
            let packed_4444 = t.format.guest == crate::render::formats::GuestFormat::Rgba4Unorm;
            let kind = if packed || packed_4444 { PixelOutputKind::Uint } else { kind };
            let mut o = PixelOutputBinding::new(t.slot, t.slot, kind);
            o.component_mapping = ColorComponentMapping::new(t.format.export_mapping.0);
            if packed {
                // CB_COLOR_INFO.ROUND_MODE (bit 18): truncate.
                o.packed_unorm_11_11_10 = Some(t.info & (1 << 18) != 0);
                if t.write_mask & 7 != 7 && t.write_mask & 7 != 0 {
                    crate::log_once("ps5gpu: a packed 11_11_10 UNORM target with a partial write mask writes every field");
                }
            }
            if packed_4444 {
                // Rounded half up; a truncating ROUND_MODE is not modelled.
                o.packed_unorm_4_4_4_4 = true;
                if t.info & (1 << 18) != 0 {
                    crate::log_once("ps5gpu: a truncating (ROUND_MODE) 4_4_4_4 UNORM target is written rounded half up");
                }
                if t.write_mask & 0xF != 0xF && t.write_mask != 0 {
                    crate::log_once("ps5gpu: a packed 4_4_4_4 UNORM target with a partial write mask writes every field");
                }
            }
            o.export_target = state::export_for_slot(d.shader_col_format, d.shader_mask, t.slot);
            outputs.push(o);
        }
        // Inputs naming a parameter the vertex stage does not export read
        // their default (0) instead of an unwritten interpolant.
        // (Only for a vertex program whose exports are all in its own code:
        // no jumps through computed addresses.)
        let defaulted_inputs = if unexported_inputs_enabled() && vs_prepared.program.address_independent { default_unexported_inputs(&d.ps_input_cntl, d.ps_in_control & 0x3F, vs_prepared.program.param_exports) } else { None };
        let ps_input_cntl = defaulted_inputs.as_ref().unwrap_or(&d.ps_input_cntl);
        let color_list: smallvec::SmallVec<[&ColorTarget; MAX_COLOR_ATTACHMENTS]> = targets.colors.iter().map(|(t, _)| t).collect();
        let (alpha_source, dual_source) = state::pixel_blend_sources(&color_list, d.shader_col_format, self.dev.caps.dual_src_blend);
        // The depth export is used only when the DB takes it.
        let depth_export = d.db_shader_control & 1 != 0 && d.shader_z_format != 0;
        // The stencil test value export (32_GR and 32_ABGR keep it in y).
        let stencil_export = d.db_shader_control & 2 != 0 && matches!(d.shader_z_format, 2 | 9) && stencil_export_enabled() && targets.depth.is_some_and(|(z, _)| z.has_stencil);
        if stencil_export && !self.dev.caps.stencil_export {
            crate::log_once("ps5gpu: stencil test value export without VK_EXT_shader_stencil_export: the reference register is used");
        }
        let stencil_export = stencil_export && self.dev.caps.stencil_export;
        // The stencil operation value export (REPLACE_OP with the program's
        // value as the host reference).
        let stencil_op_export = state::stencil_op_export(d) && stencil_export_enabled() && targets.depth.is_some_and(|(z, _)| z.has_stencil);
        if stencil_op_export && !self.dev.caps.stencil_export {
            crate::log_once("ps5gpu: stencil operation value export without VK_EXT_shader_stencil_export: the register value is used");
        }
        let stencil_op_export = stencil_op_export && self.dev.caps.stencil_export;
        // Alpha-to-coverage (DB_ALPHA_TO_MASK.ALPHA_TO_MASK_ENABLE): the MRT0
        // alpha decides the sample mask. The undithered setting (0xAA01:
        // offsets 2, no rounding) is the hardware's threshold at one half;
        // dithered or rounded settings take the same threshold.
        let alpha_to_coverage = d.alpha_to_coverage() && alpha_to_coverage_enabled();
        let early_tests = early_fragment_tests_enabled() && state::early_fragment_tests(d.db_shader_control, alpha_to_coverage);
        // PA_SC_AA_MASK that differs between the pixels of a quad: the pixel
        // program masks its samples.
        let raster_samples = targets.colors.first().map(|(t, _)| t.samples).or(targets.depth.map(|(z, _)| z.samples)).unwrap_or(1 << (d.aa_config & 7).min(3));
        let quad_masks = quad_sample_masks(d, raster_samples);
        // The masks are shader data (one module for every mask a game
        // steps through, e.g. a dithered fade); the module only knows that
        // it masks.
        let quad_masks = match (quad_masks, ps_prepared.as_mut()) {
            (Some(m), Some(p)) if quad_mask_data_enabled() => {
                self.with_quad_masks(p, m);
                Some(QUAD_MASKS_AT_RUN_TIME)
            }
            (m, _) => m,
        };
        // PS_ITER_SAMPLE on a multisampled target: the pixel program runs
        // per sample, and only then does the ancillary sample number vary.
        let sample_rate = raster_samples > 1 && d.ps_iter_sample && ps_iter_sample_enabled() && self.dev.caps.sample_rate_shading;
        if quad_masks.is_some() && ps_prepared.is_none() {
            crate::log_once("ps5gpu: a per-pixel PA_SC_AA_MASK without a pixel program keeps every sample");
        }
        if alpha_to_coverage && d.alpha_to_mask != 0xAA01 {
            crate::log_once(&format!("ps5gpu: alpha-to-coverage {:#x} drawn undithered", d.alpha_to_mask));
        }
        // The pixel stage options as plain words (the module permutation).
        // A pixel program addresses only the LDS SPI_SHADER_PGM_RSRC2_PS.
        // EXTRA_LDS_SIZE reserves (128-dword units); its per-invocation
        // array is sized by it.
        let ps_lds = ps_prepared.as_ref().filter(|p| p.program.uses_local_data_share && pixel_lds_size_enabled()).map(|_| pixel_lds_dwords(d.shaders.ps.rsrc2));
        let mut ps_raw: smallvec::SmallVec<[u32; 4 * MAX_COLOR_ATTACHMENTS + 36]> = smallvec::SmallVec::new();
        if ps_prepared.is_some() {
            for o in outputs.iter().take(MAX_COLOR_ATTACHMENTS) {
                ps_raw.extend_from_slice(&[o.guest_slot, o.kind as u32 | o.packed_unorm_11_11_10.map_or(0, |t| 2 | t as u32) << 8, o.component_mapping.packed as u32, o.export_target.map_or(0, |t| t + 1)]);
            }
            ps_raw.extend_from_slice(&[
                outputs.len() as u32 | (alpha_source as u32) << 8 | (dual_source as u32) << 12 | (depth_export as u32) << 16 | (alpha_to_coverage as u32) << 20 | (early_tests as u32) << 21 | (stencil_export as u32) << 22 | (sample_rate as u32) << 23 | (stencil_op_export as u32) << 24,
                d.ps_input_ena,
                d.ps_input_addr,
                d.shader_col_format,
            ]);
            ps_raw.extend_from_slice(ps_input_cntl);
            if let Some(m) = quad_masks {
                ps_raw.extend_from_slice(&m);
            }
            if let Some(l) = ps_lds {
                ps_raw.extend_from_slice(&[u32::MAX, l]);
            }
        }
        // A draw whose stages, stage options, targets and fixed-function
        // inputs equal an earlier draw's replays that draw's modules, views,
        // layout and pipeline (see `DrawRecipe`).
        let recipe_key = (recipes_enabled() && mesh.is_none() && gpu_indirect.is_none() && !trace::active()).then(|| RecipeKey {
            vs: Arc::as_ptr(&vs_prepared.program) as usize,
            vs_resource: vs_prepared.resource_key,
            vs_layout: Arc::as_ptr(&vs_prepared.layout) as usize,
            vs_rsrc1: vs_stage.rsrc1,
            vs_raw,
            ps: ps_prepared.as_ref().map_or(0, |p| Arc::as_ptr(&p.program) as usize),
            ps_resource: ps_prepared.as_ref().map_or(0, |p| p.resource_key),
            ps_layout: ps_prepared.as_ref().map_or(0, |p| Arc::as_ptr(&p.layout) as usize),
            ps_rsrc1: d.shaders.ps.rsrc1,
            ps_raw: smallvec::SmallVec::from_slice(&ps_raw),
            primitive: d.primitive_type,
            single_rect: d.primitive_type == primitive::RECT_LIST && primitives::single_rectangle(d.packet.count),
            restart: restart.is_some() && (index.is_some() || d.primitive_type == primitive::QUAD_LIST_LEGACY),
            colors: targets.colors.iter().map(|(t, id)| RecipeColor::of(t, *id)).collect(),
            depth: targets.depth.map(|(z, id)| RecipeDepth::of(&z, id)),
            aa_config: if attachmentless { d.aa_config } else { 0 },
            raster: [d.raster.mode_cntl, d.raster.clip_cntl, d.raster.dx_clip_space as u32, d.msaa_enable as u32 | (d.ps_iter_sample as u32) << 1, d.aa_mask[0], d.aa_mask[1]],
            whole: (!compact_recipe_key_enabled()).then(|| Box::new((targets.colors.iter().copied().collect(), targets.depth))),
        });
        let hit = recipe_key.as_ref().and_then(|k| self.recipes.find(k, &vertex_inputs));
        if hit.is_some() {
            self.stats.recipe_hits += 1;
        } else if recipe_key.is_some() {
            self.stats.recipe_misses += 1;
        }
        let vs = match &hit {
            Some(r) => Ok((r.vs_key, r.vs_module.clone())),
            None => self.module(compiler, ShaderStage::Vertex, &vs_prepared, vs_options, &vs_options_key, &vs_raw, mesh, vs_stage.rsrc1, wait),
        };
        // The primitive class only matters to a program that loads the
        // ancillary VGPR (SPI_PS_INPUT_ENA.ANCILLARY_ENA).
        let raster_primitive = if d.ps_input_ena & (1 << 13) != 0 { raster_primitive_class(d, mesh.as_ref()) } else { 2 };
        // DB_SHADER_CONTROL.PRIMITIVE_ORDERED_PIXEL_SHADER.
        let ordered = d.db_shader_control >> 16 & 1 != 0 && self.dev.caps.fragment_shader_pixel_interlock;
        if d.db_shader_control >> 16 & 1 != 0 && !ordered {
            crate::log_once("ps5gpu: a primitive-ordered pixel program runs unordered: the device has no fragmentShaderPixelInterlock");
        }
        let ps = match (&ps_prepared, &hit) {
            (Some(_), Some(r)) => r.ps.clone().map(Ok),
            (Some(p), None) => {
                let options_key = || {
                    let key = hash_of(&(
                        outputs.iter().map(|o| (o.guest_slot, o.kind as u32, o.component_mapping.packed, o.export_target, o.packed_unorm_11_11_10)).collect::<Vec<_>>(),
                        d.ps_input_ena,
                        d.ps_input_addr,
                        &ps_input_cntl,
                        d.shader_col_format,
                        (alpha_source, dual_source, depth_export, alpha_to_coverage, early_tests, sample_rate),
                        d.front_face_all_bits,
                        quad_masks,
                        (stencil_export, stencil_op_export),
                        (raster_primitive, ordered),
                    ));
                    // Without LDS the key is the one from before its size counted.
                    match ps_lds {
                        None => key,
                        Some(l) => hash_of(&(key, l)),
                    }
                };
                let front_face_all_bits = d.front_face_all_bits;
                let (pixel_outputs, col_format, ena, addr, cntl) = (outputs.to_vec(), d.shader_col_format, d.ps_input_ena, d.ps_input_addr, ps_input_cntl.to_vec());
                let opts: Options = Arc::new(move |r: &mut ShaderCompileRequest<'_>| {
                    r.pixel_outputs = pixel_outputs.clone();
                    r.color_export_format = col_format;
                    r.pixel_input_enable = ena;
                    r.pixel_input_address = addr;
                    r.pixel_input_cntl = Some(cntl.clone());
                    r.supports_clip_distance = true;
                    r.pixel_alpha_blend_source = alpha_source;
                    r.pixel_dual_source_blend = dual_source;
                    r.pixel_depth_export = depth_export;
                    r.pixel_alpha_to_coverage = alpha_to_coverage;
                    r.pixel_early_fragment_tests = early_tests;
                    r.front_face_all_bits = front_face_all_bits;
                    r.pixel_raster_primitive = raster_primitive;
                    r.pixel_ordered = ordered;
                    r.pixel_sample_rate = sample_rate;
                    r.pixel_quad_sample_masks = quad_masks;
                    r.pixel_stencil_export = stencil_export;
                    r.pixel_stencil_op_export = stencil_op_export;
                    if let Some(l) = ps_lds {
                        r.local_data_share_dwords = l;
                    }
                });
                Some(self.module(compiler, ShaderStage::Pixel, p, opts, &options_key, &ps_raw, None, d.shaders.ps.rsrc1, wait))
            }
            (None, _) => None,
        };
        // Both stages are queued before a missing one skips the draw.
        let (vs_key, vs_module) = vs?;
        let ps = ps.transpose()?;

        // ---- rectangle lists ----
        let mut rect = RectMode::None;
        if d.primitive_type == primitive::RECT_LIST {
            self.stats.rect_draws += 1;
            rect = if self.dev.caps.tessellation_shader {
                RectMode::Tessellation
            } else if self.dev.caps.fill_rectangle {
                RectMode::FillRectangle
            } else if primitives::single_rectangle(d.packet.count) {
                RectMode::Strip
            } else {
                RectMode::List
            };
        }
        let tess = match &hit {
            Some(r) => r.tess,
            None if rect == RectMode::Tessellation => Some(self.rect_modules(&vs_module.reflection)?.ok_or_else(|| Failure::Error("rect-list stages".into()))?),
            None => None,
        };

        lap.lap(24);
        // ---- targets ----
        let mut color_formats = [vk::Format::UNDEFINED.as_raw(); MAX_COLOR_ATTACHMENTS];
        let mut blend = [state::BlendKey::default(); MAX_COLOR_ATTACHMENTS];
        let mut color_views = [vk::ImageView::null(); MAX_COLOR_ATTACHMENTS];
        let mut color_count = 0u32;
        // Attachments for the layout transitions: (image, depth, range),
        // with the colour slot of each (depth last).
        let mut attachments: smallvec::SmallVec<[(u64, bool, SubRange); MAX_COLOR_ATTACHMENTS + 1]> = smallvec::SmallVec::new();
        let mut attachment_slots: smallvec::SmallVec<[usize; MAX_COLOR_ATTACHMENTS + 1]> = smallvec::SmallVec::new();
        let (mut width, mut height, mut layers, mut samples) = (u32::MAX, u32::MAX, u32::MAX, 1u32);
        let mut depth_view = vk::ImageView::null();
        let (mut depth_format, mut stencil_format) = (vk::Format::UNDEFINED, vk::Format::UNDEFINED);
        if let Some(r) = &hit {
            color_formats = r.color_formats;
            blend = r.blend;
            color_views = r.color_views;
            color_count = r.color_count;
            attachments = smallvec::SmallVec::from_slice(&r.attachments);
            attachment_slots = smallvec::SmallVec::from_slice(&r.attachment_slots);
            (width, height, layers, samples) = r.extent;
            (depth_view, depth_format, stencil_format) = r.depth;
        }
        for (t, id) in targets.colors.iter().filter(|_| hit.is_none()) {
            let slot = t.slot as usize;
            if slot >= MAX_COLOR_ATTACHMENTS {
                continue;
            }
            // The bound level (CB_COLOR_VIEW.MIP_LEVEL) sets the extent.
            let Some(view) = ctx.images.view_level(*id, t.format.host, vk::ImageAspectFlags::COLOR, t.mip_level, t.base_layer, t.layers) else { continue };
            color_views[slot] = view;
            attachments.push((*id, false, SubRange { base_level: t.mip_level, levels: 1, base_layer: t.base_layer, layers: t.layers.max(1) }));
            attachment_slots.push(slot);
            color_formats[slot] = t.format.host.as_raw();
            let source = if slot == 0 && ps_prepared.is_some() { alpha_source } else { Default::default() };
            blend[slot] = state::blend_key(t, self.dev.caps.dual_src_blend, source);
            color_count = color_count.max(slot as u32 + 1);
            width = width.min((t.width >> t.mip_level).max(1));
            height = height.min((t.height >> t.mip_level).max(1));
            layers = layers.min(t.layers);
            samples = t.samples;
        }
        if let Some((z, id)) = targets.depth.as_ref().filter(|_| hit.is_none()) {
            if let Some(img) = ctx.images.get(*id) {
                let aspect = img.image.aspect;
                let format = img.key.format;
                if let Some(v) = ctx.images.view(*id, format, aspect, z.base_layer, z.layers) {
                    depth_view = v;
                    attachments.push((*id, true, SubRange { base_level: 0, levels: 1, base_layer: z.base_layer, layers: z.layers.max(1) }));
                    attachment_slots.push(MAX_COLOR_ATTACHMENTS);
                    if aspect.contains(vk::ImageAspectFlags::DEPTH) {
                        depth_format = format;
                    }
                    if aspect.contains(vk::ImageAspectFlags::STENCIL) {
                        stencil_format = format;
                    }
                    width = width.min(z.width);
                    height = height.min(z.height);
                    layers = layers.min(z.layers);
                    samples = z.samples;
                }
            }
        }
        if attachmentless && hit.is_none() {
            width = self.dev.caps.max_framebuffer_width.max(1);
            height = self.dev.caps.max_framebuffer_height.max(1);
            layers = 1;
            // PA_SC_AA_CONFIG.MSAA_NUM_SAMPLES (log2).
            samples = 1 << (d.aa_config & 7).min(3);
        }
        if width == u32::MAX || !attachmentless && color_count == 0 && depth_view == vk::ImageView::null() {
            self.skip_why = if width == u32::MAX { "no render area" } else { "targets not bindable" };
            return Err(Failure::Skip);
        }
        let layers = layers.max(1);

        lap.lap(25);
        // Mesh draws: workgroups, and the entry point's draw parameters
        // after the stage's shader data.
        let mut vs_prepared = vs_prepared;
        // An indirect mesh draw's workgroups and parameters come from its
        // record (`record_mesh_launch`): the template's are placeholders.
        let mesh_launch = match &mesh {
            Some(m) => {
                let mut params = Self::mesh_params(d, index);
                // Strip restarts are found on the CPU: a record-launched
                // draw's indices are not known here.
                let restart_strip = index.is_some() && d.primitive_restart && d.primitive_type == primitive::TRI_STRIP && mesh_strip_restart_enabled();
                if restart_strip && mesh_indirect.is_some() {
                    self.skip_why = "mesh strip restart";
                    return Err(Failure::NoGpuIndirect);
                }
                if let Some(table) = self.mesh_restart_table(ctx, d, index)? {
                    params[crate::shader::mesh::DRAW_RESTART_TABLE as usize] = (vs_prepared.shader_data.len() + params.len()) as u32;
                    vs_prepared.shader_data.extend_from_slice(&params);
                    vs_prepared.shader_data.extend_from_slice(&table);
                    self.stats.mesh_restart_tables += 1;
                } else {
                    vs_prepared.shader_data.extend_from_slice(&params);
                }
                let groups = self.mesh_groups(d, m);
                if trace::active() {
                    self.trace_notes.push(format!("    mesh {m:?} launch {groups:?} {params:?} regs {:?}", d.shaders.geometry));
                }
                if groups[0] == 0 && mesh_indirect.is_none() {
                    self.skip_why = "mesh launch";
                    return Err(Failure::Skip);
                }
                Some((groups[0], groups[1]))
            }
            None => None,
        };

        lap.lap(26);
        // ---- buffer reservation (merges happen before any binding) ----
        let mut ranges: RangeList = RangeList::new();
        let mut maps: [smallvec::SmallVec<[u32; 16]>; 2] = Default::default();
        crate::backend::profile::dtime(39, || {
            self.stage_ranges_mapped(&vs_prepared, &mut ranges, Some(&mut maps[0]));
            if let Some(p) = &ps_prepared {
                self.stage_ranges_mapped(p, &mut ranges, Some(&mut maps[1]));
            }
        });
        for b in &vertex_inputs.buffers {
            if let Some((a, n)) = self.clamp_range(b.address, b.size) {
                ranges.push((a, n, false, true));
            }
        }
        let convert = mesh.is_none() && index.is_some_and(|i| primitives::needs_conversion(i.kind, d.primitive_type, restart));
        if gpu_indirect.is_some() && (convert || (rect != RectMode::None && !rect_mode_reads_record(rect))) {
            self.skip_why = "conversion or rectangles";
            return Err(Failure::NoGpuIndirect);
        }
        if let Some(i) = index {
            if !convert {
                // Mesh draws read indices through device addresses.
                ranges.push((i.address, i.bytes().max(4), false, mesh.is_none()));
            }
        }
        self.choose_streams(ctx, &ranges);
        self.draw_writes_ranges = ranges.iter().any(|r| r.2);
        let streamed: smallvec::SmallVec<[bool; 32]> = smallvec::SmallVec::from_slice(&self.stream_flags);
        for (&(a, n, _, _), &stream) in ranges.iter().zip(&streamed) {
            if !stream {
                ctx.buffers.find_or_create(ctx.sched, a, n)?;
            }
        }
        // The stages' buffers bind what the reservation resolved.
        if resolved_ranges_enabled() {
            let stages = [Some(&vs_prepared), ps_prepared.as_ref()];
            for (k, (map, p)) in maps.iter().zip(stages).enumerate() {
                self.resolved[k] = p.filter(|p| map.len() == p.snapshot.buffers.len()).map(|_| {
                    map.iter().map(|&i| ranges.get(i as usize).map(|r| (r.0, r.1, streamed[i as usize]))).collect()
                });
            }
        }

        lap.lap(27);
        // ---- descriptors ----
        let mut bound: smallvec::SmallVec<[BoundStage; 2]> = smallvec::smallvec![BoundStage { stage: ShaderStage::Vertex, prepared: vs_prepared }];
        if let (Some(p), Some(_)) = (ps_prepared, &ps) {
            bound.push(BoundStage { stage: ShaderStage::Pixel, prepared: p });
        }
        let mut writes = std::mem::take(&mut self.writes);
        writes.clear();
        self.patch_shader_data = mesh_indirect.is_some().then_some(ShaderStage::Vertex);
        self.patched_shader_data = None;
        let mut bind = Ok(());
        for b in bound.iter_mut() {
            bind = self.bind_stage(ctx, b, &mut writes);
            if bind.is_err() {
                break;
            }
        }
        self.patch_shader_data = None;
        self.resolved = Default::default();
        if let Err(e) = bind {
            self.writes = writes;
            return Err(e);
        }
        if let Some(spec) = ctx.spec() {
            // Nothing of the draw is recorded yet: a slice thread stops here
            // when it needs the exact state, or writes memory.
            if writes.gds || !writes.written_images.is_empty() || !writes.written_ranges.is_empty() {
                spec.bail(crate::slices::spec::Hole::Written);
            }
            if spec.bailed().is_some() {
                self.writes = writes;
                return Err(Failure::Bail);
            }
        }
        if crate::slice_shadow::enabled() {
            // The draw's read set: its buffer ranges and sampled images.
            for (&(a, n, _, _), &stream) in ranges.iter().zip(streamed.iter()) {
                if stream {
                    crate::slice_shadow::read_streamed(a, n);
                } else {
                    crate::slice_shadow::read(a, n);
                }
            }
            for &id in writes.bound_images.iter() {
                crate::slice_shadow::read_image(id, ctx.images.guest_range(id));
            }
            if writes.gds {
                crate::slice_shadow::hole(crate::slice_shadow::Hole::Gds);
            }
        }
        let first_stage = if mesh.is_some() { vk::ShaderStageFlags::MESH_EXT } else { vk::ShaderStageFlags::VERTEX };
        let push_stages = first_stage | vk::ShaderStageFlags::FRAGMENT;
        let layout = match &hit {
            Some(r) => r.layout,
            None => self.draw_layout(&bound, mesh.is_some(), push_stages)?,
        };

        lap.lap(28);
        // ---- indices ----
        let mut host_indices: Option<HostIndices> = None;
        let mut index_binding: Option<(vk::Buffer, u64, vk::IndexType)> = None;
        if mesh.is_some() {
            // The mesh entry point fetches indices itself (through the
            // device-address page table): the buffer only has to be current.
            if let Some(i) = index {
                ctx.buffers.obtain(ctx.sched, i.address, i.bytes().max(4), false)?;
            }
        } else if let Some(i) = index {
            if convert {
                let bytes_len = d.packet.count as u64 * i.kind.bytes();
                if ctx.buffers.has_gpu_dirty(i.address, bytes_len) {
                    ctx.buffers.download(ctx.sched, i.address, bytes_len);
                }
                let mut bytes = vec![0u8; bytes_len as usize];
                if let Some(spec) = ctx.spec() {
                    spec.note_pages(ctx.buffers.tracker(), i.address, bytes_len);
                }
                if !self.memory.read_backing(i.address, &mut bytes) {
                    return Err(Failure::Error(format!("index buffer {:#x}+{bytes_len:#x} unreadable", i.address)));
                }
                let h = primitives::convert(i.kind, &bytes, d.primitive_type, restart, self.provoking_last(d));
                self.stats.cpu_index_conversions += 1;
                let info = self.upload(ctx, &h.bytes, 16)?;
                index_binding = Some((info.buffer, info.offset, h.kind));
                host_indices = Some(h);
            } else {
                let n = i.bytes().max(4);
                let kind = if i.kind == IndexType::U32 { vk::IndexType::UINT32 } else { vk::IndexType::UINT16 };
                let streamed = if self.streams.contains(&(i.address, n)) { self.stream_guest(ctx, i.address, n) } else { None };
                index_binding = Some(match streamed {
                    Some(info) => (info.buffer, info.offset, kind),
                    None => {
                        let slice = ctx.buffers.obtain(ctx.sched, i.address, n, false)?;
                        (slice.buffer, slice.offset, kind)
                    }
                });
            }
        } else if let Some(h) = primitives::convert_auto(d.primitive_type, d.packet.count, self.provoking_last(d)) {
            self.stats.cpu_index_conversions += 1;
            let info = self.upload(ctx, &h.bytes, 16)?;
            index_binding = Some((info.buffer, info.offset, h.kind));
            host_indices = Some(h);
        }
        let mut vertex_buffers = Vec::new();
        for b in &vertex_inputs.buffers {
            match self.clamp_range(b.address, b.size) {
                Some((a, n)) => {
                    let streamed = if self.streams.contains(&(a, n)) { self.stream_guest(ctx, a, n) } else { None };
                    match streamed {
                        Some(info) => vertex_buffers.push((info.buffer, info.offset)),
                        None => {
                            let s = ctx.buffers.obtain(ctx.sched, a, n, false)?;
                            vertex_buffers.push((s.buffer, s.offset));
                        }
                    }
                }
                None => vertex_buffers.push((self.t.null_buffer.buffer, 0)),
            }
        }

        lap.lap(29);
        // ---- pipeline ----
        // A vertex stage that selects the viewport gets all 16.
        let viewport_count = if vs_module.reflection.writes_viewport_index && self.dev.caps.multi_viewport { state::MAX_VIEWPORTS as u32 } else { 1 };
        let pipeline = match &hit {
            Some(r) => {
                self.stats.graphics_hits += 1;
                r.pipeline
            }
            None => {
                let ff = self.fixed_function(d, &vertex_inputs, rect, restart.is_some() && index_binding.is_some(), mesh.is_some(), viewport_count, (color_formats, color_count, blend), (depth_format, stencil_format), samples, alpha_to_coverage).inspect_err(|e| {
                    if matches!(e, Failure::Skip) {
                        self.skip_why = "unsupported primitive type";
                    }
                })?;
                let gkey = GraphicsKey { vs: vs_key, ps: ps.as_ref().map_or(0, |(k, _)| *k), layout: layout.pipeline_layout, ff };
                let vertex = gkey.ff.vertex_input.clone();
                let pipeline = self.graphics_pipeline(gkey, &vs_module, ps.as_ref().map(|(_, m)| m), tess, mesh.is_some(), wait)?;
                if let Some(key) = recipe_key {
                    self.recipes.insert(DrawRecipe {
                        key,
                        vertex,
                        vs_key,
                        vs_module: vs_module.clone(),
                        ps: ps.clone(),
                        tess,
                        color_formats,
                        blend,
                        color_views,
                        color_count,
                        attachments: attachments.clone(),
                        attachment_slots: attachment_slots.clone(),
                        extent: (width, height, layers, samples),
                        depth: (depth_view, depth_format, stencil_format),
                        layout,
                        pipeline,
                        _held: (bound[0].prepared.program.clone(), bound[0].prepared.layout.clone(), bound.get(1).map(|b| (b.prepared.program.clone(), b.prepared.layout.clone()))),
                    });
                }
                pipeline
            }
        };

        lap.lap(30);
        // ---- record ----
        let dynamic = state::dynamic_state(
            d,
            width,
            height,
            if depth_format != vk::Format::UNDEFINED { depth_format } else { stencil_format },
            depth_format != vk::Format::UNDEFINED,
            stencil_format != vk::Format::UNDEFINED,
            rect != RectMode::None,
            viewport_count,
            state::clip_disabled(d).then(|| state::clip_disabled_extent(&self.dev.props.limits)),
        );
        // Consecutive draws to the same attachments share a rendering scope
        // (in order, as the hardware rasterises them); between scopes only
        // the images whose layout or access changes get barriers. A draw
        // that writes memory through shaders, uses the global data share or
        // reads one of its own attachments as a texture gets a scope of its
        // own: a full barrier on either side.
        // The draw writes neither depth nor stencil: a sampled depth
        // attachment is read in the read-only layout (no feedback loop).
        let depth_writes = targets.depth.map(|(z, _)| state::depth_attachment_writes(d, depth_format != vk::Format::UNDEFINED, z.has_stencil));
        let depth_read_only = targets.depth.is_some_and(|(z, _)| {
            let (dw, sw) = depth_writes.unwrap_or_default();
            !dw && !sw && (!z.has_stencil || !d.depth_state.stencil_test) && depth_read_only_enabled()
        });
        // The draw writes stencil but not depth and reads its depth/stencil
        // attachment only through views of the depth aspect: depth is
        // read-only and the stencil aspect an attachment (aspects are
        // separate subresources: no feedback loop).
        let stencil_split = !depth_read_only
            && depth_format != vk::Format::UNDEFINED
            && depth_split_enabled()
            && targets.depth.is_some_and(|(z, zid)| {
                z.has_stencil
                    && depth_writes == Some((false, true))
                    && writes.image_slots.iter().filter(|s| s.id == zid).all(|s| !s.storage && !ctx.images.is_stencil_view(writes.images[s.index].image_view))
            });
        let feedback = writes.bound_images.iter().any(|id| targets.colors.iter().any(|(_, c)| c == id) || (!depth_read_only && !stencil_split && targets.depth.is_some_and(|(_, z)| z == *id)));
        // Stores through device addresses without an evaluated range may
        // write anything: the draw is ordered against all other work.
        let unknown_hazards = hazards::unknown_hazards_enabled();
        let unranged_writes = unknown_hazards && bound.iter().any(|b| hazards::writes_unranged_memory(b.prepared.program.writes_device_memory, &b.prepared.snapshot.device_address_ranges));
        let isolated = !shared_rendering_scopes() || writes.gds || !writes.written_images.is_empty() || !writes.written_ranges.is_empty() || feedback || unranged_writes;
        if isolated {
            let why = &mut self.stats.isolated;
            why[0] += writes.gds as u64;
            why[1] += !writes.written_images.is_empty() as u64;
            why[2] += !writes.written_ranges.is_empty() as u64;
            why[3] += feedback as u64;
        }
        // Buffers a dispatch wrote without a barrier after it (index, vertex,
        // descriptor and indirect-argument ranges); a stage reading memory it
        // has no ranges for waits for every such write.
        let reads_shader_writes = ctx.sched.shader_write_overlaps(ranges.iter().map(|r| (r.0, r.1)).chain(gpu_indirect.iter().flat_map(|g| g.guest)))
            || (unknown_hazards
                && ctx.sched.has_unordered_shader_writes()
                && bound.iter().any(|b| hazards::reads_unranged_memory(&b.prepared.resources.info, &b.prepared.snapshot.device_address_ranges)));
        if isolated || reads_shader_writes {
            ctx.sched.full_barrier();
        }
        // What the draw reads: later dispatches writing it wait for it.
        ctx.sched.note_draw_reads(
            ranges.iter().map(|r| (r.0, r.1)).chain(gpu_indirect.iter().flat_map(|g| g.guest)),
            unknown_hazards && bound.iter().any(|b| hazards::reads_unranged_memory(&b.prepared.resources.info, &b.prepared.snapshot.device_address_ranges)),
        );
        let mesh_command = match (mesh_indirect, &mesh) {
            (Some(g), Some(m)) => match self.record_mesh_launch(ctx, d, &g, m, &bound[0].prepared, vs_count) {
                Ok(c) => Some(c),
                Err(e) => {
                    self.writes = writes;
                    return Err(e);
                }
            },
            _ => None,
        };
        // Image layouts of the draw: only changes are recorded, before the
        // scope begins (a change ends an open scope).
        let mut barriers = std::mem::take(&mut self.image_barriers);
        barriers.clear();
        let (general, mut depth_layout) = Self::image_layouts(&mut ctx.images, &mut writes, &attachments, depth_read_only, stencil_split, &mut barriers);
        let mut color_layouts = [crate::images::COLOR_ATTACHMENT.0; MAX_COLOR_ATTACHMENTS];
        for (i, &slot) in attachment_slots.iter().enumerate() {
            if general & (1 << i) != 0 {
                match color_layouts.get_mut(slot) {
                    Some(l) => *l = vk::ImageLayout::GENERAL,
                    None => depth_layout = vk::ImageLayout::GENERAL,
                }
            }
        }
        let scope = RenderingKey {
            colors: color_views,
            color_count,
            depth: depth_view,
            depth_attachment: depth_format != vk::Format::UNDEFINED,
            stencil_attachment: stencil_format != vk::Format::UNDEFINED,
            width,
            height,
            layers,
            general: general != 0,
            depth_layout,
            provoking_last: !self.dev.caps.provoking_vertex_per_pipeline && self.provoking_last(d),
        };
        if crate::backend::profile::detailed() {
            lap.lap(45);
        }
        if crate::slices::verify() {
            self.digest = draw_digest(pipeline, &layout, &scope, &ranges, &writes, &bound, index_binding.is_some(), d);
        }
        // Most draws continue the open scope: its attachments are built
        // only for a scope that begins.
        let cb = match ctx.sched.continue_rendering_scope(&scope, !barriers.is_empty()) {
            Some(cb) => cb,
            None => {
                let color_attachments: [vk::RenderingAttachmentInfo; MAX_COLOR_ATTACHMENTS] = std::array::from_fn(|i| {
                    vk::RenderingAttachmentInfo::default()
                        .image_view(color_views[i])
                        .image_layout(color_layouts[i])
                        .load_op(vk::AttachmentLoadOp::LOAD)
                        .store_op(vk::AttachmentStoreOp::STORE)
                });
                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_view)
                    .image_layout(depth_layout)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE);
                let mut rendering = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width, height } })
                    .layer_count(layers)
                    .color_attachments(&color_attachments[..color_count as usize]);
                if depth_format != vk::Format::UNDEFINED {
                    rendering = rendering.depth_attachment(&depth_attachment);
                }
                if stencil_format != vk::Format::UNDEFINED {
                    rendering = rendering.stencil_attachment(&depth_attachment);
                }
                ctx.sched.begin_rendering_scope(scope, &rendering, &barriers)
            }
        };
        if crate::backend::profile::detailed() {
            lap.lap(46);
        }
        if redundant_state_filter() {
            // Bound state survives rendering scopes and barriers within one
            // command buffer until other work binds its own.
            self.scope_state.enter(if crate::backend::ab::fast() { ctx.sched.state_serial() } else { ctx.sched.scope_serial() });
        } else {
            self.scope_state = Default::default();
        }
        self.scope_state.bind(&self.dev, cb, pipeline, &dynamic);
        if crate::backend::profile::detailed() {
            lap.lap(43);
        }
        let committed = self.commit(ctx, cb, vk::PipelineBindPoint::GRAPHICS, &layout, push_stages, &writes, &bound, true);
        self.image_barriers = barriers;
        if let Err(e) = committed {
            self.writes = writes;
            return Err(e);
        }
        if crate::backend::profile::detailed() {
            lap.lap(42);
        }
        let cb = ctx.sched.rendering_cmd();
        let device = &self.dev.device;
        let instances = d.packet.instance_count.max(1);
        // Faces with different depth biases: front faces, then back faces.
        let back_faces = dynamic.back_faces();
        if let Some((groups, mesh_instances)) = mesh_launch {
            let mesh_device = self.dev.mesh.as_ref().expect("mesh device");
            let record = |cb: vk::CommandBuffer| {
                // SAFETY: recording inside the rendering scope with a mesh
                // pipeline; the command lives in the staging ring for the tick.
                unsafe {
                    match mesh_command {
                        Some((buffer, offset)) => mesh_device.cmd_draw_mesh_tasks_indirect(cb, buffer, offset, 1, indirect_mesh::COMMAND_BYTES as u32),
                        None => mesh_device.cmd_draw_mesh_tasks(cb, groups, mesh_instances, 1),
                    }
                }
            };
            record(cb);
            if let Some(back) = &back_faces {
                self.scope_state.bind(&self.dev, cb, pipeline, back);
                record(cb);
            }
            if mesh_command.is_some() {
                self.stats.gpu_indirect_mesh_draws += 1;
            }
        } else {
            // SAFETY: recording inside the rendering scope; buffers are cache/staging buffers alive for the tick.
            unsafe {
                if !vertex_buffers.is_empty() {
                    let bufs: smallvec::SmallVec<[vk::Buffer; 8]> = vertex_buffers.iter().map(|v| v.0).collect();
                    let offs: smallvec::SmallVec<[u64; 8]> = vertex_buffers.iter().map(|v| v.1).collect();
                    device.cmd_bind_vertex_buffers(cb, 0, &bufs, &offs);
                }
                if let Some((buffer, offset, kind)) = index_binding {
                    if !self.scope_state.same_index((buffer, offset, kind)) {
                        device.cmd_bind_index_buffer(cb, buffer, offset, kind);
                    }
                }
            }
            // GE_INDX_OFFSET is added to every guest index, and the vertex
            // IDs of an auto-indexed draw start at it (also when its indices
            // are generated here, as for quads).
            let vertex_offset = if index.is_some() || auto_index_offset_enabled() { d.index_offset as i32 } else { 0 };
            let record = |cb: vk::CommandBuffer| {
                // SAFETY: recording inside the rendering scope; buffers are cache/staging buffers alive for the tick.
                unsafe {
                    if let Some(g) = gpu_indirect {
                        let indexed = index_binding.is_some();
                        let stride = g.stride;
                        match (g.count, indexed) {
                            (Some((cbuf, coff)), true) => device.cmd_draw_indexed_indirect_count(cb, g.buffer, g.offset, cbuf, coff, g.draw_count, stride),
                            (Some((cbuf, coff)), false) => device.cmd_draw_indirect_count(cb, g.buffer, g.offset, cbuf, coff, g.draw_count, stride),
                            (None, true) => device.cmd_draw_indexed_indirect(cb, g.buffer, g.offset, g.draw_count, stride),
                            (None, false) => device.cmd_draw_indirect(cb, g.buffer, g.offset, g.draw_count, stride),
                        }
                    } else if index_binding.is_some() {
                        let count = host_indices.as_ref().map_or(d.packet.count, |h| h.count);
                        device.cmd_draw_indexed(cb, count, instances, 0, vertex_offset, 0);
                    } else {
                        let count = if rect == RectMode::Strip { 4 } else { d.packet.count };
                        device.cmd_draw(cb, count, instances, vertex_offset as u32, 0);
                    }
                }
            };
            record(cb);
            if let Some(back) = &back_faces {
                self.scope_state.bind(&self.dev, cb, pipeline, back);
                record(cb);
            }
        }
        if crate::backend::profile::detailed() {
            lap.lap(44);
        }
        lap.lap(31);
        if isolated {
            ctx.sched.full_barrier();
        }
        let tick = ctx.sched.current_tick();
        if trace::active() {
            self.trace_written.extend(writes.written_images.iter().copied());
        }
        for &id in &writes.written_images {
            ctx.images.mark_gpu_written(id, tick);
            ctx.images.note_storage_write(id);
        }
        for &(a, n) in &writes.written_ranges {
            ctx.images.invalidate_shader_write(a, n);
        }
        self.compressed_stores.extend_from_slice(&writes.compressed_stores);
        self.writes = writes;
        lap.lap(32);
        Ok(DrawOutcome::Recorded)
    }

    /// Record an indirect draw from the guest's argument records on the GPU
    /// (they are usually written by compute work of the same frame:
    /// reading them on the CPU waits for the GPU). On the vertex path it is
    /// a host indirect draw, the record offsets the packet writes to user
    /// SGPRs reaching the vertex program as the host draw parameters
    /// (`indirect_draw` of the compile request). A mesh draw of one record
    /// is launched by `vkCmdDrawMeshTasksIndirectEXT` from a command the
    /// indirect-mesh kernel makes from the record, which also writes the
    /// offsets to the stage's shader data (`indirect_mesh`). None when the
    /// draw needs the CPU path: see the reasons logged.
    fn gpu_indirect_draw(&mut self, ctx: &mut DrawContext, d: &ResolvedDraw) -> Option<DrawOutcome> {
        use crate::render::{DrawKind, IndexBuffer, SH_NOP};
        if !gpu_indirect_enabled() {
            crate::log_once("ps5gpu: indirect draws on the CPU (RPS5E_GPU_INDIRECT=0)");
            return None;
        }
        let caps = &self.dev.caps;
        let p = &d.packet;
        let indexed = p.kind.indexed();
        let multi = matches!(p.kind, DrawKind::IndirectMulti | DrawKind::IndexIndirectMulti);
        let draw_count = if multi { p.indirect_count } else { 1 };
        if draw_count == 0 {
            return None;
        }
        let regs = p.indirect_registers;
        let geometry = d.shaders.stages & STAGES_GS_EN != 0;
        let rect = d.primitive_type == primitive::RECT_LIST;
        // The rectangle expansion `try_draw` picks for this device.
        let rect_mode = if caps.tessellation_shader { RectMode::Tessellation } else if caps.fill_rectangle { RectMode::FillRectangle } else { RectMode::List };
        let kind = IndexType::from_raw(p.index_type);
        let restart = d.primitive_restart.then_some(d.restart_index);
        // Why the draw cannot run as a host indirect draw / as a mesh
        // launch from its record.
        let mut host_why = if !caps.shader_draw_parameters {
            Some("no shaderDrawParameters")
        } else if (draw_count > 1 && !caps.multi_draw_indirect) || (multi && p.count_address != 0 && !caps.draw_indirect_count) {
            Some("draw count / device limits")
        } else if indexed && regs.first_index != SH_NOP {
            Some("first index written to a register")
        } else if geometry {
            Some("geometry stage")
        } else if rect && !rect_mode_reads_record(rect_mode) {
            Some("rectangles")
        } else if indexed && primitives::needs_conversion(kind, d.primitive_type, restart) {
            Some("index conversion")
        } else if !indexed && primitives::convert_auto(d.primitive_type, 3, false).is_some() {
            Some("primitive conversion")
        } else {
            None
        };
        let mut mesh_why = if !gpu_indirect_mesh_enabled() {
            Some("RPS5E_GPU_INDIRECT_MESH=0")
        } else if self.t.indirect_mesh.is_none() {
            Some("no mesh shaders or kernel")
        } else if draw_count != 1 {
            Some("several records")
        } else if rect {
            Some("rectangles")
        } else {
            None
        };
        if indexed && p.index_buffer_size == 0 {
            (host_why, mesh_why) = (Some("index buffer size 0"), Some("index buffer size 0"));
        }
        // Each record of a multi-draw writes its own index to the
        // draw-index register; one record writes 0, as the template does.
        if draw_count > 1 && regs.draw_index_target() != SH_NOP {
            (host_why, mesh_why) = (Some("draw index written to a register"), Some("draw index written to a register"));
        }
        if let (Some(h), Some(m)) = (host_why, mesh_why) {
            crate::log_once(&format!("ps5gpu: indirect draw on the CPU: {h} (mesh: {m})"));
            return None;
        }
        // The offsets must land in the vertex program's user data.
        let base = if d.shaders.es.bound() || d.shaders.gs.bound() { crate::render::regs::sh::SPI_SHADER_USER_DATA_GS_0 } else { crate::render::regs::sh::SPI_SHADER_USER_DATA_VS_0 };
        let sgpr = |reg: u32| -> Option<u8> {
            if reg == SH_NOP {
                return Some(0xFF);
            }
            reg.checked_sub(base).filter(|&i| i < 32).map(|i| i as u8)
        };
        let first_index = if indexed { regs.first_index } else { SH_NOP };
        let (Some(v), Some(i), Some(f)) = (sgpr(regs.vertex_offset), sgpr(regs.instance_offset), sgpr(first_index)) else {
            crate::log_once(&format!(
                "ps5gpu: indirect draw on the CPU: offsets to registers {:#x}/{:#x}/{:#x} outside the vertex program's user data",
                regs.vertex_offset, regs.instance_offset, first_index
            ));
            return None;
        };
        let sgprs = [v, i, f];
        let size = if indexed { indirect::INDEXED_RECORD } else { indirect::AUTO_RECORD };
        let stride = if multi { p.stride as u64 } else { size };
        if stride < size || stride % 4 != 0 || p.indirect_address % 4 != 0 {
            crate::log_once("ps5gpu: indirect draw on the CPU: record stride/alignment");
            return None;
        }
        // A direct draw standing for every record: zero offsets, the whole
        // index buffer bound (the records' first index moves within it).
        let template = if indexed { [p.index_buffer_size, 1, 0, 0, 0] } else { [3, 1, 0, 0, 0] };
        let mut sub = indirect::direct_draw(d, &template, 0)?;
        if indexed {
            sub.index = Some(IndexBuffer { address: p.index_base, kind, count: p.index_buffer_size });
            sub.packet.index_address = p.index_base;
        }
        let records = ctx.buffers.obtain(ctx.sched, p.indirect_address, stride * (draw_count as u64 - 1) + size, false).ok()?;
        let count = if multi && p.count_address != 0 {
            let c = ctx.buffers.obtain(ctx.sched, p.count_address, 4, false).ok()?;
            Some((c.buffer, c.offset))
        } else {
            None
        };
        if records.offset % 4 != 0 || count.is_some_and(|c| c.1 % 4 != 0) {
            return None;
        }
        let guest = [(p.indirect_address, stride * (draw_count as u64 - 1) + size), (p.count_address, if count.is_some() { 4 } else { 0 })];
        self.gpu_indirect = Some(GpuIndirect {
            buffer: records.buffer,
            offset: records.offset,
            draw_count,
            stride: stride as u32,
            count,
            sgprs,
            guest,
            host: host_why.is_none(),
            mesh: mesh_why.is_none(),
        });
        let compiler = self.compiler.clone();
        let mut r = self.try_draw(ctx, &sub, compiler.as_ref());
        if matches!(r, Err(Failure::StagingFull)) {
            ctx.sched.flush();
            r = self.try_draw(ctx, &sub, compiler.as_ref());
        }
        self.gpu_indirect = None;
        if matches!(r, Err(Failure::NoGpuIndirect)) {
            crate::log_once(&format!(
                "ps5gpu: indirect draw on the CPU: {} (host: {}, mesh: {})",
                std::mem::take(&mut self.skip_why),
                host_why.unwrap_or("-"),
                mesh_why.unwrap_or("-")
            ));
            return None;
        }
        let o = self.draw_outcome(d, r);
        if o == DrawOutcome::Recorded {
            self.stats.recorded_draws += 1;
            self.stats.gpu_indirect_draws += 1;
        }
        Some(o)
    }

    /// True when the resources of `vs` (prepared from `user` with zero
    /// offsets) are the same with other values in the offset SGPRs, which
    /// are `user[first + sgpr]` (memoised per program). `prepare` prepares
    /// the stage from other user SGPRs.
    #[allow(clippy::too_many_arguments)]
    fn indirect_resources_independent(
        &mut self,
        ctx: &mut DrawContext,
        vs: &PreparedStage,
        address: u64,
        user: &[u32],
        first: usize,
        sgprs: [u8; 3],
        mesh: bool,
        prepare: impl FnOnce(&mut Self, &mut DrawContext, &[u32]) -> Result<PreparedStage, Failure>,
    ) -> bool {
        let key = (address, vs.program.plan.id, sgprs, mesh);
        if let Some(&ok) = self.indirect_independent.get(&key) {
            return ok;
        }
        let mut probe = [0u32; 8 + 32];
        let n = user.len().min(probe.len());
        probe[..n].copy_from_slice(&user[..n]);
        for (k, v) in sgprs.iter().zip([0x0012_3457u32, 0x0000_0A5B, 0x0000_3C1D]) {
            if let Some(slot) = (*k != 0xFF).then(|| first + *k as usize).filter(|&s| s < n) {
                probe[slot] = v;
            }
        }
        let ok = match prepare(self, ctx, &probe[..n]) {
            Ok(q) => {
                let (a, b) = (&*vs.snapshot, &*q.snapshot);
                q.resource_key == vs.resource_key
                    && a.buffers == b.buffers
                    && a.images == b.images
                    && a.samplers == b.samplers
                    && a.flattened_resource_table == b.flattened_resource_table
                    && a.device_address_ranges == b.device_address_ranges
            }
            Err(_) => false,
        };
        if !ok {
            log::info!("ps5gpu: vertex program {address:#x} reads indirect offsets in its resources: its indirect draws stay on the CPU");
        }
        self.indirect_independent.insert(key, ok);
        ok
    }

    /// Record the indirect-mesh kernel for the mesh draw being recorded
    /// (outside any rendering scope, ordered after the record's writers and
    /// before the draw): it writes the launch command to a fresh staging
    /// slot, returned, and the draw parameters and record offsets to the
    /// vertex stage's patched shader data.
    fn record_mesh_launch(&mut self, ctx: &mut DrawContext, d: &ResolvedDraw, g: &GpuIndirect, m: &MeshConfig, vs: &PreparedStage, vs_count: u32) -> Result<(vk::Buffer, u64), Failure> {
        let data = self.patched_shader_data.take().ok_or_else(|| Failure::Error("mesh indirect draw without shader data".into()))?;
        let align = self.dev.caps.min_storage_buffer_offset_alignment.max(16);
        let command = self.upload(ctx, &[0u8; 16], align)?;
        // The shader-data dword of user SGPR `k` (s8 + k of the subgroup
        // program); none when the program does not read it.
        let program = &vs.program;
        let dword = |k: u8| -> u32 {
            if k == 0xFF || k as u32 >= vs_count {
                return indirect_mesh::NO_DWORD;
            }
            let reg = 8 + k as u32;
            vs.layout.user_data_registers.iter().position(|&r| r.wrapping_sub(program.user_data_base) == reg).map_or(indirect_mesh::NO_DWORD, |i| i as u32)
        };
        let p = &d.packet;
        let indexed = p.kind.indexed();
        let index = d.index.filter(|_| indexed);
        let caps = &self.dev.caps;
        let args = MeshIndirectArgs {
            indexed,
            index_limit: if indexed { p.index_buffer_size } else { 0 },
            index_bytes: index.map_or(0, |i| i.kind.bytes() as u32),
            index_base: index.map_or(0, |i| i.address),
            rules: MeshRules::new(m, caps.max_mesh_work_group_count, caps.max_mesh_work_group_total_count),
            draw_dword: m.draw_dword,
            vertex_dword: dword(g.sgprs[0]),
            instance_dword: dword(g.sgprs[1]),
            first_index_dword: dword(g.sgprs[2]),
        };
        let record = crate::tiler::TilerBuffer { buffer: g.buffer, offset: g.offset, size: args.record_bytes() };
        let count = g.count.map(|(buffer, offset)| crate::tiler::TilerBuffer { buffer, offset, size: 4 });
        let command_slot = crate::tiler::TilerBuffer { buffer: command.buffer, offset: command.offset, size: indirect_mesh::COMMAND_BYTES };
        let data_slot = crate::tiler::TilerBuffer { buffer: data.buffer, offset: data.offset, size: data.range };
        let Some(kernel) = self.t.indirect_mesh.as_ref() else {
            self.skip_why = "no indirect-mesh kernel";
            return Err(Failure::NoGpuIndirect);
        };
        // The record's writers (transfers and dispatches recorded earlier;
        // `cmd` orders unfenced shader work and ends an open rendering
        // scope with a barrier) come first; the draw reads the command
        // (indirect) and the shader data (mesh stage) after the kernel.
        if ctx.sched.has_unordered_work() {
            ctx.sched.full_barrier();
        }
        let cb = ctx.sched.cmd();
        kernel.record(cb, record, count, command_slot, data_slot, &args);
        ctx.sched.full_barrier();
        if trace::active() {
            self.trace_notes.push(format!("    gpu-indirect mesh {args:?}"));
        }
        Ok((command.buffer, command.offset))
    }

    /// The mesh workgroup layout of a primitive subgroup (registers as the
    /// hardware reads them). `geometry`: a geometry stage follows the ES
    /// (GS_EN) and sets the output primitives; without one each input
    /// primitive is one output primitive of its own vertices.
    fn mesh_config(&self, d: &ResolvedDraw, wave: u32, draw_dword: u32, geometry: bool) -> Result<MeshConfig, String> {
        let g = &d.shaders.geometry;
        let primitive = d.primitive_type;
        if !matches!(primitive, 1 | 2 | 4 | 5 | 6) || !geometry && !matches!(primitive, 4 | 5 | 6) {
            return Err(format!("input primitive {primitive}"));
        }
        let vertices_per_primitive = if geometry { g.max_vert_out } else { 3 };
        // Output point lists (0), line strips (1) and triangle strips (2).
        let points = geometry && g.out_prim_type == 0 && mesh_points_enabled();
        let lines = geometry && g.out_prim_type == 1 && mesh_lines_enabled();
        let strip_vertices = if points {
            1
        } else if lines {
            2
        } else {
            3
        };
        if geometry && (!(g.out_prim_type == 2 || lines || points) || vertices_per_primitive < strip_vertices) {
            return Err(format!("output primitive {} with {} vertices", g.out_prim_type, vertices_per_primitive));
        }
        let input_size = match primitive {
            1 => 1,
            2 => 2,
            _ => 3,
        };
        let input_step = if matches!(primitive, 5 | 6) { 1 } else { input_size };
        let vertices = (g.ge_cntl >> 9) & 0x1FF;
        let primitives = g.ge_cntl & 0x1FF;
        let max_vertices = if geometry { g.max_output_per_subgroup } else { vertices };
        if vertices < input_size || max_vertices == 0 || max_vertices > 256 || vertices_per_primitive > 256 {
            return Err(format!("subgroup of {vertices} vertices, {max_vertices} outputs"));
        }
        let group_primitives = primitives.min((vertices - input_size) / input_step + 1).min(max_vertices / vertices_per_primitive);
        if group_primitives == 0 {
            return Err("empty subgroup".into());
        }
        if g.esgs_ring_item_size == 0 || g.esgs_ring_item_size * vertices > 0xFFFF {
            return Err(format!("ESGS item size {}", g.esgs_ring_item_size));
        }
        let vertices_per_group = (group_primitives - 1) * input_step + input_size;
        let (max_vertices, max_primitives) = if geometry { (max_vertices, primitives * (vertices_per_primitive - (strip_vertices - 1))) } else { (vertices_per_group, group_primitives) };
        let threads = vertices_per_group.max(primitives).max(max_vertices).max(max_primitives).next_multiple_of(wave.max(64));
        let caps = &self.dev.caps;
        let lanes = if wave == 64 && caps.subgroup_size == 32 { 2 } else { 1 };
        let lds_bytes = ((d.shaders.gs.rsrc2 >> 19) & 0xFF) * 128 * 4;
        if max_vertices > caps.max_mesh_output_vertices
            || max_primitives > caps.max_mesh_output_primitives
            || threads / lanes > caps.max_mesh_work_group_invocations
            || lds_bytes > caps.max_mesh_shared_memory
        {
            return Err(format!("limits: {max_vertices} vertices, {max_primitives} primitives, {threads} lanes, {lds_bytes} LDS bytes"));
        }
        Ok(MeshConfig {
            input_primitive: primitive,
            primitives_per_group: group_primitives,
            vertices_per_group,
            max_vertices,
            max_primitives,
            threads,
            esgs_item_size: g.esgs_ring_item_size,
            draw_dword,
            lines,
            points,
        })
    }

    /// Workgroups (x, instances, z) of a mesh draw: all zero when it draws
    /// nothing (indirect draws arrive here as direct draws, see
    /// `indirect`; or launch from their record, see `indirect_mesh`).
    fn mesh_groups(&self, d: &ResolvedDraw, m: &MeshConfig) -> [u32; 3] {
        let caps = &self.dev.caps;
        let rules = MeshRules::new(m, caps.max_mesh_work_group_count, caps.max_mesh_work_group_total_count);
        indirect_mesh::launch(d.packet.count, d.packet.instance_count.max(1), &rules)
    }

    /// The draw parameters of a mesh draw (see `shader::mesh`).
    fn mesh_params(d: &ResolvedDraw, index: Option<crate::render::IndexBuffer>) -> [u32; crate::shader::mesh::DRAW_DWORDS as usize] {
        let (index_bytes, address) = index.map_or((0, 0), |i| (i.kind.bytes() as u32, i.address));
        // GE_INDX_OFFSET is added to every index (the generated ones of
        // auto draws too).
        let base = if index.is_some() || auto_index_offset_enabled() { d.index_offset } else { 0 };
        [d.packet.count, base, 0, index_bytes, address as u32, (address >> 32) as u32, 0]
    }

    /// The restart table of an indexed strip mesh draw with primitive
    /// restart whose indices restart (`shader::mesh::restart_table`).
    /// `RPS5E_MESH_STRIP_RESTART=0`: none (restart indices fetched as
    /// vertices).
    fn mesh_restart_table(&mut self, ctx: &mut DrawContext, d: &ResolvedDraw, index: Option<crate::render::IndexBuffer>) -> Result<Option<Vec<u32>>, Failure> {
        let Some(i) = index else { return Ok(None) };
        if !d.primitive_restart || d.primitive_type != primitive::TRI_STRIP || !mesh_strip_restart_enabled() {
            return Ok(None);
        }
        let bytes_len = d.packet.count as u64 * i.kind.bytes();
        if bytes_len == 0 {
            return Ok(None);
        }
        if ctx.buffers.has_gpu_dirty(i.address, bytes_len) {
            ctx.buffers.download(ctx.sched, i.address, bytes_len);
        }
        let mut bytes = vec![0u8; bytes_len as usize];
        if !self.memory.read_backing(i.address, &mut bytes) {
            return Err(Failure::Error(format!("index buffer {:#x}+{bytes_len:#x} unreadable", i.address)));
        }
        Ok(crate::shader::mesh::restart_table(&bytes, i.kind.bytes() as u32, d.restart_index))
    }

    /// Whether an indirect dispatch with thread dimensions runs from its
    /// record on the GPU (see `indirect_threads`). Not when the switch is
    /// off, the kernel is missing, the record is not dword-aligned, or a
    /// start group is set (an indirect host dispatch has no base group).
    fn gpu_thread_dimensions(&self, d: &ResolvedDispatch) -> bool {
        gpu_indirect_threads_enabled() && self.t.indirect_threads.is_some() && d.packet.indirect_address % 4 == 0 && d.start == [0, 0, 0]
    }

    /// Whether a compute pipeline of `module` with `threads` per group of
    /// guest `wave` width is pinned to 32-lane subgroups.
    fn subgroup32(&self, threads: [u32; 3], wave: u32, module: &Module) -> bool {
        let invocations = threads[0] * threads[1] * threads[2];
        let pin32 = wave == 32 || (self.narrow_compute && module.reflection.required_subgroup_size == 0);
        self.subgroup32_compute && pin32 && invocations <= self.max_subgroup_invocations
    }

    /// While a draw or dispatch waits for a compile: queue the modules and
    /// pipelines of the dispatches the command stream holds after it (see
    /// `lookahead`), once per command position and finished job.
    fn compile_ahead(&mut self) {
        if self.ahead.queue_only || self.speculative || self.t.pool.is_none() || !super::lookahead::enabled() {
            return;
        }
        let Some(position) = super::lookahead::position() else { return };
        let mark = (position, super::pool::foreground_done());
        if self.ahead.mark == Some(mark) {
            return;
        }
        self.ahead.mark = Some(mark);
        if self.ahead.position != position {
            self.ahead.position = position;
            self.ahead.handled.clear();
        }
        let upcoming = super::lookahead::upcoming(super::lookahead::limit());
        self.ahead.queue_only = true;
        for d in &upcoming {
            let id = hash_of(&(d.shader.address, &d.shader.user_data, d.threads, d.shader.rsrc1, d.shader.rsrc2, d.packet.dispatch_initiator & (1 << 15)));
            if self.ahead.handled.contains(&id) {
                continue;
            }
            if self.compile_dispatch_ahead(d) {
                self.ahead.handled.insert(id);
                self.stats.lookahead_queued += 1;
            }
        }
        self.ahead.queue_only = false;
    }

    /// Queue the module of upcoming dispatch `d`, and its pipeline once the
    /// module is there. True when nothing is left to queue for it.
    fn compile_dispatch_ahead(&mut self, d: &ResolvedDispatch) -> bool {
        if !d.shader.bound() || indirect_thread_dimensions(d) {
            return true;
        }
        let (Some(compiler), Some(tracker)) = (self.compiler.clone(), self.ahead.tracker.clone()) else { return true };
        let wave = if d.packet.dispatch_initiator & (1 << 15) != 0 { 32 } else { 64 };
        let count = d.shader.user_sgpr_count().min(16);
        let (cursor, thread_limits) = dispatch_layout(d);
        let prepared = {
            let mut f = self.frontend.lock().unwrap_or_else(|e| e.into_inner());
            let Ok(program) = f.program_for(d.shader.address, ShaderStage::Compute, 0, count, wave, &d.shader.user_data) else { return true };
            // Prepared from guest memory as it is now (GPU-written words
            // fail: the dispatch is prepared for real when it runs).
            match f.prepare_prefetch(&program, &d.shader.user_data, cursor, thread_limits, &tracker) {
                Some((p, _)) => p,
                None => return false,
            }
        };
        let perm = ComputePermutation::of(d, self.dev.caps.shader_shared_int64_atomics);
        let module = match self.module(&compiler, ShaderStage::Compute, &prepared, perm.options(), &|| perm.key, &perm.raw, None, d.shader.rsrc1, false) {
            Ok((key, m)) => (key, m),
            Err(Failure::Compiling) => return false,
            Err(_) => return true,
        };
        let bound = [BoundStage { stage: ShaderStage::Compute, prepared }];
        let Ok(layout) = self.draw_layout(&bound, false, vk::ShaderStageFlags::COMPUTE) else { return true };
        let subgroup32 = self.subgroup32(perm.threads, wave, &module.1);
        // Queued (or there, or failed): nothing is left to do for it.
        let _ = self.compute_pipeline(ComputeKey { cs: module.0, layout: layout.pipeline_layout, subgroup32 }, &module.1);
        true
    }

    fn try_dispatch(&mut self, ctx: &mut DrawContext, d: &ResolvedDispatch, compiler: Option<Arc<dyn ShaderCompiler>>) -> Result<DrawOutcome, Failure> {
        let direct = d.packet.indirect_address == 0;
        let groups = d.group_counts();
        if !d.shader.bound() || (direct && groups.contains(&0)) {
            return Err(Failure::Skip);
        }
        // The record counts threads, not groups: a host indirect dispatch
        // cannot read it as it is. The back-end reads it on the CPU when
        // the GPU path is not available.
        if indirect_thread_dimensions(d) && !self.gpu_thread_dimensions(d) {
            crate::log_once(&format!("ps5gpu: indirect dispatch of cs={:#x} with thread dimensions reached the provider without the GPU path; skipped", d.shader.address));
            return Err(Failure::Skip);
        }
        // An indirect dispatch whose record no GPU work is writing and holds
        // a zero count launches nothing: skipped before its program is
        // prepared.
        if !direct && !indirect_thread_dimensions(d) && empty_indirect_skip_enabled() && !ctx.sched.is_slice() && d.packet.indirect_address % 4 == 0 && !ctx.buffers.has_gpu_dirty(d.packet.indirect_address, 12) {
            let mut b = [0u8; 12];
            if self.memory.read_backing(d.packet.indirect_address, &mut b) && b.chunks_exact(4).any(|c| c == [0, 0, 0, 0]) {
                self.stats.empty_indirect_dispatches += 1;
                return Err(Failure::Skip);
            }
        }
        // A group count above the device limit is invalid in Vulkan (the
        // driver may run it as a near-endless dispatch): such a dispatch is
        // split into chunks of at most the limit, each launched at its base
        // group (`RPS5E_DISPATCH_SPLIT=0`: skipped and reported once per
        // shader).
        let limit = self.dev.props.limits.max_compute_work_group_count;
        if direct && !dispatch_split_enabled() && (0..3).any(|i| groups[i] > limit[i]) {
            static REPORTED: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());
            let mut seen = REPORTED.lock().unwrap_or_else(|e| e.into_inner());
            if !seen.contains(&d.shader.address) {
                seen.push(d.shader.address);
                log::warn!("ps5gpu: dispatch of cs={:#x} with {:?} groups exceeds the device limit {limit:?}; skipped", d.shader.address, groups);
            }
            return Err(Failure::Skip);
        }
        let wave = if d.packet.dispatch_initiator & (1 << 15) != 0 { 32 } else { 64 };
        let count = d.shader.user_sgpr_count().min(16);
        let (cursor, thread_limits) = dispatch_layout(d);
        self.ahead.tracker = Some(ctx.buffers.tracker().clone());
        let mut dpre = crate::backend::draw_prefetch::take_dispatch();
        let prepared = self.prepare_stage_limited(ctx, ShaderStage::Compute, d.shader.address, 0, &d.shader.user_data, count, wave, cursor, thread_limits, Prefetched::Dispatch(&mut dpre))?;
        let Some(compiler) = compiler else { return Err(Failure::NeedsShader) };
        let perm = ComputePermutation::of(d, self.dev.caps.shader_shared_int64_atomics);
        let (threads, system) = (perm.threads, perm.system);
        // Later work may depend on a dispatch: it waits for its module.
        let (cs_key, module) = self.module(&compiler, ShaderStage::Compute, &prepared, perm.options(), &|| perm.key, &perm.raw, None, d.shader.rsrc1, true)?;

        self.write_limits.clear();
        if direct && write_extent_enabled() {
            let system = super::write_extent::SystemSgprs { work_group: [system.work_group_x, system.work_group_y, system.work_group_z], thread_group_size: system.thread_group_size };
            self.bound_writes(&prepared, d, system, wave == 64, groups, threads);
        }
        let mut ranges = RangeList::new();
        self.stage_ranges(&prepared, &mut ranges);
        if !direct {
            ranges.push((d.packet.indirect_address, 12, false, false));
        }
        self.streams.clear();
        self.draw_writes_ranges = ranges.iter().any(|r| r.2);
        for &(a, n, _, _) in &ranges {
            ctx.buffers.find_or_create(ctx.sched, a, n)?;
        }
        let mut bound = [BoundStage { stage: ShaderStage::Compute, prepared }];
        let mut writes = std::mem::take(&mut self.writes);
        writes.clear();
        let r = self.dispatch_bound(ctx, d, direct, threads, wave, cs_key, &module, &mut bound, &mut writes, &ranges);
        self.writes = writes;
        self.write_limits.clear();
        r
    }

    /// The bytes the dispatch can store to through each written buffer
    /// whose stores the program bounds (`write_limits`).
    fn bound_writes(&mut self, prepared: &PreparedStage, d: &ResolvedDispatch, system: super::write_extent::SystemSgprs, wave64: bool, groups: [u32; 3], threads: [u32; 3]) {
        let program = &prepared.program;
        let key = (Arc::as_ptr(program) as usize, system, wave64);
        let stores = match self.store_bounds.get(&key) {
            Some(s) => s.clone(),
            None => {
                let plan = &program.plan;
                let mut buffers: Vec<(u32, Vec<u32>)> = Vec::new();
                for u in plan.resource_uses() {
                    let ps5shader::resources::ResourceBinding::Buffer(i) = u.binding else { continue };
                    if !matches!(plan.graph.memory().get(u.memory_index).access, ps5shader::resources::MemoryAccess::Write | ps5shader::resources::MemoryAccess::Atomic) {
                        continue;
                    }
                    match buffers.iter_mut().find(|b| b.0 == i) {
                        Some(b) => b.1.push(u.pc),
                        None => buffers.push((i, vec![u.pc])),
                    }
                }
                let pcs: Vec<u32> = buffers.iter().flat_map(|b| b.1.iter().copied()).collect();
                let bounds = if pcs.is_empty() { None } else { super::write_extent::StoreBounds::analyze(plan.program(), program.user_data_base, program.user_data_count, system, wave64, &pcs) };
                let s = Arc::new(ProgramStores { _program: program.clone(), bounds, buffers });
                if self.store_bounds.len() >= 4096 {
                    self.store_bounds.clear();
                }
                self.store_bounds.insert(key, s.clone());
                s
            }
        };
        let Some(bounds) = &stores.bounds else { return };
        let info = &prepared.resources.info;
        let inputs = super::write_extent::DispatchInputs { user_data: &d.shader.user_data, user_base: 0, groups, local: threads };
        for (i, pcs) in &stores.buffers {
            let Some(words) = prepared.snapshot.buffers.get(*i as usize) else { continue };
            let v = BufferSharp::from_words(words);
            if v.swizzle_enabled() || v.add_tid() {
                continue;
            }
            let Some((addr, size)) = self.clamp_range(v.address(), buffer_extent(&v, info.buffers.get(*i as usize))) else { continue };
            let Some((lo, end)) = bounds.extent(pcs, v.stride(), &inputs) else { continue };
            let (a, e) = ((v.address() + lo).max(addr), (v.address() + end).min(addr + size));
            // Pages are tracked whole: a bound within the range's pages
            // changes nothing.
            let page = crate::memory::PAGE_BYTES;
            let pages = |a: u64, e: u64| (a / page, e.div_ceil(page));
            if a < e && pages(a, e) != pages(addr, addr + size) {
                self.write_limits.push((*i, a, e - a));
                self.stats.bounded_writes += 1;
            }
        }
    }

    /// Bind and record a dispatch whose ranges are reserved.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_bound(
        &mut self,
        ctx: &mut DrawContext,
        d: &ResolvedDispatch,
        direct: bool,
        threads: [u32; 3],
        wave: u32,
        cs_key: u64,
        module: &Arc<Module>,
        bound: &mut [BoundStage; 1],
        writes: &mut SetWrites,
        ranges: &[(u64, u64, bool, bool)],
    ) -> Result<DrawOutcome, Failure> {
        // An indirect dispatch with thread dimensions gets its thread limits
        // on the GPU, in a shader-data copy of its own.
        let thread_dims = indirect_thread_dimensions(d);
        self.patch_shader_data = thread_dims.then_some(ShaderStage::Compute);
        self.patched_shader_data = None;
        let bind = self.bind_stage(ctx, &mut bound[0], writes);
        self.patch_shader_data = None;
        bind?;
        let layout = self.draw_layout(&bound[..], false, vk::ShaderStageFlags::COMPUTE)?;
        let invocations = threads[0] * threads[1] * threads[2];
        let subgroup32 = self.subgroup32(threads, wave, module);
        let pipeline = self.compute_pipeline(ComputeKey { cs: cs_key, layout: layout.pipeline_layout, subgroup32 }, module)?;
        let args = if direct { None } else { Some(ctx.buffers.obtain(ctx.sched, d.packet.indirect_address, 12, false)?) };

        // Only the hazards that exist get a full barrier: work recorded
        // through `cmd` (it may have written anything), shader writes of
        // earlier work to a range this dispatch touches, and earlier shader
        // work before a dispatch that writes memory (write after read).
        // Images order themselves through their layouts.
        // Stores through device addresses have no ranges: such a dispatch
        // is ordered against everything (a counter reset by one dispatch
        // must land before the next one's atomics).
        let device_writes = bound[0].prepared.program.writes_device_memory;
        let precise = precise_barriers() && !writes.gds && !device_writes;
        let writes_memory = !writes.written_ranges.is_empty() || !writes.written_images.is_empty();
        // A dispatch reading memory it has no ranges for waits for every
        // unordered shader write.
        let reads_unranged = hazards::unknown_hazards_enabled() && hazards::reads_unranged_memory(&bound[0].prepared.resources.info, &bound[0].prepared.snapshot.device_address_ranges);
        let unranged_reads = reads_unranged && ctx.sched.has_unordered_shader_writes();
        let mut touched = std::mem::take(&mut self.touched_ranges);
        touched.clear();
        touched.extend(ranges.iter().map(|r| (r.0, r.1)));
        if precise && crate::scheduler::hazard::range_barriers_enabled() {
            // Ordered only against the earlier work whose ranges overlap:
            // read or write after a dispatch's write (memory dependency),
            // write after a read (the readers' stages finish first).
            if ctx.sched.has_unordered_work() {
                ctx.sched.full_barrier();
            } else {
                let b = ctx.sched.dispatch_barrier(&touched, &writes.written_ranges, reads_unranged);
                ctx.sched.dispatch_barrier_record(b);
            }
        } else if !precise
            || ctx.sched.has_unordered_work()
            || (writes_memory && ctx.sched.has_shader_work())
            || ctx.sched.shader_write_overlaps(touched.iter().copied())
            || unranged_reads
        {
            ctx.sched.full_barrier();
        }
        let mut barriers = std::mem::take(&mut self.image_barriers);
        barriers.clear();
        Self::image_layouts(&mut ctx.images, writes, &[], false, false, &mut barriers);
        ctx.sched.image_barriers(&barriers);
        self.image_barriers = barriers;
        // The groups the record's thread counts launch, resolved on the GPU
        // (the barrier above ordered the record's writers before it).
        let args = match args {
            Some(record) if thread_dims => Some(self.record_thread_launch(ctx, d, threads, &bound[0].prepared, record)?),
            Some(record) if indirect_group_clamp_enabled() && self.t.indirect_threads.is_some() => Some(self.record_group_clamp(ctx, record)?),
            Some(record) => Some((record.buffer, record.offset)),
            None => None,
        };
        let cb = ctx.sched.ordered_cmd();
        // SAFETY: recording buffer; compatible pipeline.
        crate::scheduler::disturb_state();
        unsafe { self.dev.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline) };
        self.commit(ctx, cb, vk::PipelineBindPoint::COMPUTE, &layout, vk::ShaderStageFlags::COMPUTE, writes, &bound[..], false)?;
        let cb = ctx.sched.ordered_cmd();
        self.dev.checkpoint(cb, d.shader.address);
        // SAFETY: recording buffer; descriptors and push data bound above.
        unsafe {
            match args {
                Some((buffer, offset)) => self.dev.device.cmd_dispatch_indirect(cb, buffer, offset),
                None => {
                    let g = d.group_counts();
                    let s = d.start;
                    let n = [g[0].saturating_sub(s[0]).max(1), g[1].saturating_sub(s[1]).max(1), g[2].saturating_sub(s[2]).max(1)];
                    let limit = self.dev.props.limits.max_compute_work_group_count;
                    let lds_dwords = ((d.shader.rsrc2 >> 15) & 0x1FF) * 128;
                    if ordered_append_serial(d.packet.dispatch_initiator, writes.gds, invocations, wave, lds_dwords, g) {
                        // Ordered append: one wave per group, launched in
                        // group order, each after the previous one's writes
                        // (the guest workgroup ids are kept).
                        let barrier = vk::MemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
                            .dst_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ);
                        let mut first = true;
                        for z in s[2]..g[2].max(s[2] + 1) {
                            for y in s[1]..g[1].max(s[1] + 1) {
                                for x in s[0]..g[0].max(s[0] + 1) {
                                    if !first {
                                        self.dev.device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[barrier], &[], &[]);
                                    }
                                    first = false;
                                    self.dev.device.cmd_dispatch_base(cb, x, y, z, 1, 1, 1);
                                }
                            }
                        }
                    } else if s == [0, 0, 0] && (0..3).all(|i| n[i] <= limit[i]) {
                        self.dev.device.cmd_dispatch(cb, g[0], g[1], g[2]);
                    } else {
                        for (base, count) in dispatch_chunks(s, n, limit) {
                            self.dev.device.cmd_dispatch_base(cb, base[0], base[1], base[2], count[0], count[1], count[2]);
                        }
                    }
                }
            }
        }
        if precise {
            // Later work that reads what this dispatch wrote, or writes
            // what it read, waits for it.
            ctx.sched.note_dispatch(&touched, &writes.written_ranges, reads_unranged);
        } else {
            ctx.sched.full_barrier();
        }
        self.touched_ranges = touched;
        let tick = ctx.sched.current_tick();
        if trace::active() {
            self.trace_written.extend(writes.written_images.iter().copied());
        }
        for &id in &writes.written_images {
            ctx.images.mark_gpu_written(id, tick);
            ctx.images.note_storage_write(id);
        }
        for &(a, n) in &writes.written_ranges {
            ctx.images.invalidate_shader_write(a, n);
        }
        self.overwrites.clear();
        self.overwrites.extend_from_slice(&writes.overwritten);
        self.compressed_stores.extend_from_slice(&writes.compressed_stores);
        Ok(DrawOutcome::Recorded)
    }

    /// Record the thread-dimension kernel for the indirect dispatch being
    /// recorded (outside any rendering scope, after the record's writers):
    /// it writes the dispatch's groups to a fresh staging slot, returned as
    /// the indirect command, and the thread counts to the limits of the
    /// compute stage's patched shader data. A full barrier then orders both
    /// before the dispatch (indirect command read, shader read).
    /// The launch of an indirect dispatch whose record counts groups,
    /// through the record kernel with one thread per group: a record over
    /// the device's group limits launches nothing instead of a near-endless
    /// (invalid) dispatch.
    fn record_group_clamp(&mut self, ctx: &mut DrawContext, record: crate::buffers::BufferSlice) -> Result<(vk::Buffer, u64), Failure> {
        let align = self.dev.caps.min_storage_buffer_offset_alignment.max(16);
        let command = self.upload(ctx, &[0u8; 16], align)?;
        let scratch = self.upload(ctx, &[0u8; 16], align)?;
        let rules = ThreadRules::new([1, 1, 1], self.dev.props.limits.max_compute_work_group_count);
        let Some(kernel) = self.t.indirect_threads.as_ref() else {
            return Err(Failure::Error("no record kernel".into()));
        };
        let record = crate::tiler::TilerBuffer { buffer: record.buffer, offset: record.offset, size: indirect_threads::RECORD_BYTES };
        let command_slot = crate::tiler::TilerBuffer { buffer: command.buffer, offset: command.offset, size: indirect_threads::COMMAND_BYTES };
        let scratch_slot = crate::tiler::TilerBuffer { buffer: scratch.buffer, offset: scratch.offset, size: 16 };
        let cb = ctx.sched.cmd();
        kernel.record(cb, record, command_slot, scratch_slot, 0, &rules);
        ctx.sched.full_barrier();
        self.stats.gpu_group_clamps += 1;
        Ok((command.buffer, command.offset))
    }

    fn record_thread_launch(&mut self, ctx: &mut DrawContext, d: &ResolvedDispatch, threads: [u32; 3], cs: &PreparedStage, record: crate::buffers::BufferSlice) -> Result<(vk::Buffer, u64), Failure> {
        let data = self.patched_shader_data.take().ok_or_else(|| Failure::Error("thread-dimension dispatch without shader data".into()))?;
        if !cs.layout.uses_dispatch_thread_limits || cs.layout.uses_push_data() {
            return Err(Failure::Error(format!("thread-dimension dispatch of cs={:#x} without thread limits in its shader-data buffer", d.shader.address)));
        }
        let align = self.dev.caps.min_storage_buffer_offset_alignment.max(16);
        let command = self.upload(ctx, &[0u8; 16], align)?;
        let rules = ThreadRules::new(threads, self.dev.props.limits.max_compute_work_group_count);
        let Some(kernel) = self.t.indirect_threads.as_ref() else {
            return Err(Failure::Error("no thread-dimension kernel".into()));
        };
        let record = crate::tiler::TilerBuffer { buffer: record.buffer, offset: record.offset, size: indirect_threads::RECORD_BYTES };
        let command_slot = crate::tiler::TilerBuffer { buffer: command.buffer, offset: command.offset, size: indirect_threads::COMMAND_BYTES };
        let data_slot = crate::tiler::TilerBuffer { buffer: data.buffer, offset: data.offset, size: data.range };
        let cb = ctx.sched.cmd();
        kernel.record(cb, record, command_slot, data_slot, cs.layout.dispatch_thread_limits_dword(), &rules);
        ctx.sched.full_barrier();
        self.stats.gpu_thread_dispatches += 1;
        if trace::active() {
            self.trace_notes.push(format!("    gpu-indirect threads {rules:?} limits at dword {}", cs.layout.dispatch_thread_limits_dword()));
        }
        Ok((command.buffer, command.offset))
    }

    /// Frame trace of one draw (direct, or a GPU-indirect draw standing for
    /// all of its records): targets, notes, dumps of what it wrote.
    fn trace_draw(&mut self, ctx: &mut DrawContext, draw: &ResolvedDraw, o: DrawOutcome) {
        let targets: Vec<String> = ctx.targets.colors.iter().map(|(t, id)| format!("s{} {:#x} {}x{} {:?} #{id}", t.slot, t.base, t.width, t.height, t.format.host)).collect();
        let depth = ctx.targets.depth.map(|(z, id)| format!("{:#x} {}x{} #{id}", z.z_read_base, z.width, z.height));
        let written: Vec<u64> = std::mem::take(&mut self.trace_written);
        let mut dumps: Vec<(u64, String)> = Vec::new();
        if o == DrawOutcome::Recorded {
            dumps.extend(ctx.targets.colors.iter().map(|(t, id)| (*id, format!("rt{}", t.slot))));
            dumps.extend(ctx.targets.depth.iter().map(|(_, id)| (*id, "z".to_string())));
            dumps.extend(written.iter().map(|&id| (id, "img".to_string())));
        }
        dumps.extend(std::mem::take(&mut self.trace_sampled).into_iter().map(|id| (id, "tex".to_string())));
        self.trace_pass(
            ctx,
            &format!(
                "draw {:?} n={} prim={} es={:#x} gs={:#x} vs={:#x} ps={:#x} -> {o:?} colors=[{}] depth={depth:?}",
                draw.packet.kind,
                draw.packet.count,
                draw.primitive_type,
                draw.shaders.es.address,
                draw.shaders.gs.address,
                draw.shaders.vertex_program_address(),
                draw.shaders.ps.address,
                targets.join("; ")
            ),
            &dumps,
        );
        trace::log(&format!(
            "    index {:?} base {:#x} max {} offset_reg {} inst {} | depth {:?} vp0 {:?} scissor {:?} raster {:?}",
            draw.index, draw.packet.index_base, draw.packet.index_buffer_size, draw.index_offset, draw.packet.instance_count, draw.depth_state, draw.viewports[0], draw.scissor, draw.raster
        ));
        for t in &draw.color_targets {
            trace::log(&format!(
                "    blend s{} {:?} mask {:#x} col_format {:#x} shader_mask {:#x} | dcc={}@{:#x} fc={} cmask={:#x} clr={:#x} layers {}+{} attrib3 {:#x} view {:#x}",
                t.slot,
                t.blend,
                draw.target_mask,
                draw.shader_col_format,
                draw.shader_mask,
                t.dcc_enabled,
                t.dcc_address,
                t.fast_clear,
                t.cmask_address,
                t.clear_word0,
                t.base_layer,
                t.layers,
                t.attrib3,
                t.view
            ));
        }
    }

    /// Frame trace: log one pass with its notes and dump the images it wrote.
    fn trace_pass(&mut self, ctx: &mut DrawContext, line: &str, dumps: &[(u64, String)]) {
        let seq = trace::next_seq();
        trace::log(&format!("#{seq} {line}"));
        for n in std::mem::take(&mut self.trace_notes) {
            trace::log(&n);
        }
        let dumps = if dumps.iter().filter(|(_, w)| w.starts_with("rt")).count() < trace::dump_min_targets() { &[][..] } else { dumps };
        if trace::dump_last_only() {
            // Images the previous passes used and this one does not are in
            // their final state for that run of passes.
            for (id, s, what) in std::mem::take(&mut self.trace_pending) {
                if dumps.iter().any(|(d, _)| *d == id) {
                    continue;
                }
                if let Some(images) = ctx.images.exact() {
                    trace::dump_image(&self.dev, ctx.sched, images, id, s, &what);
                }
            }
            for (id, what) in dumps {
                if trace::dump_wanted(seq, what) {
                    match self.trace_pending.iter_mut().find(|p| p.0 == *id) {
                        Some(p) => p.1 = seq,
                        None => self.trace_pending.push((*id, seq, what.clone())),
                    }
                }
            }
            return;
        }
        let mut seen = Vec::new();
        for (id, what) in dumps {
            if !seen.contains(id) && trace::dump_wanted(seq, what) {
                seen.push(*id);
                if let Some(images) = ctx.images.exact() {
                    trace::dump_image(&self.dev, ctx.sched, images, *id, seq, what);
                }
            }
        }
    }

    /// [`Self::outcome`] of a draw, noted in the per-frame skip log
    /// (`RPS5E_DRAW_SKIP_LOG=1`) when it was not recorded.
    fn draw_outcome(&mut self, d: &ResolvedDraw, r: Result<DrawOutcome, Failure>) -> DrawOutcome {
        if super::skiplog::enabled() {
            use super::skiplog::SkipReason;
            let reason = match &r {
                Ok(_) | Err(Failure::NoGpuIndirect | Failure::Bail | Failure::Pending) => None,
                Err(Failure::Compiling) => Some(SkipReason::Compiling),
                Err(Failure::Skip) => Some(SkipReason::Skip),
                Err(Failure::NeedsShader) => Some(SkipReason::Shader),
                Err(Failure::StagingFull | Failure::Error(_)) => Some(SkipReason::Error),
            };
            if let Some(reason) = reason {
                let why = if reason == SkipReason::Skip { std::mem::take(&mut self.skip_why) } else { "" };
                self.skip_log.note_why(reason, why, d.shaders.vertex_program_address(), d.shaders.ps.address);
            }
        }
        self.outcome(r)
    }

    fn outcome(&mut self, r: Result<DrawOutcome, Failure>) -> DrawOutcome {
        match r {
            Ok(o) => o,
            Err(Failure::NeedsShader) => {
                self.stats.needs_shader += 1;
                DrawOutcome::NeedsShader
            }
            Err(Failure::Skip) => {
                self.stats.skipped += 1;
                DrawOutcome::Skipped
            }
            Err(Failure::NoGpuIndirect) => {
                self.stats.skipped += 1;
                DrawOutcome::Skipped
            }
            Err(Failure::Compiling) => {
                self.stats.skipped += 1;
                self.stats.compiling_skips += 1;
                DrawOutcome::Skipped
            }
            Err(Failure::StagingFull) => {
                self.stats.errors += 1;
                crate::log_once("ps5gpu: draw failed: staging ring exhausted");
                DrawOutcome::Skipped
            }
            Err(Failure::Error(m)) => {
                self.stats.errors += 1;
                crate::log_once(&format!("ps5gpu: draw failed: {m}"));
                DrawOutcome::Skipped
            }
            Err(Failure::Bail) => DrawOutcome::Hole,
            // Nothing is recorded: the caller tries the item again
            // (`PipelineProvider::take_compile_pending`).
            Err(Failure::Pending) => {
                self.stats.pending_tries += 1;
                DrawOutcome::Skipped
            }
        }
    }
}

/// The optimised build of a fast pipeline (a background compile job).
type Rebuild = Box<dyn FnOnce() -> Compiled + Send>;

/// The compile job of the graphics pipeline of `key` (`fast`: see
/// [`build_pipeline`]; its result then carries the optimised rebuild).
#[allow(clippy::too_many_arguments)]
fn graphics_job(
    dev: Arc<GpuDevice>,
    cache: vk::PipelineCache,
    key: GraphicsKey,
    vs: Arc<Module>,
    ps: Option<Arc<Module>>,
    tess: Option<(vk::ShaderModule, vk::ShaderModule)>,
    mesh: bool,
    record: Option<super::prewarm::GraphicsRecord>,
    fast: bool,
) -> impl FnOnce() -> Compiled + Send + 'static {
    fn build(dev: &GpuDevice, cache: vk::PipelineCache, key: &GraphicsKey, vs: &Module, ps: Option<&Module>, tess: Option<(vk::ShaderModule, vk::ShaderModule)>, mesh: bool, fast: bool, optimised: bool) -> (Result<vk::Pipeline, vk::Result>, Built) {
        let what = || match ps {
            Some(p) => format!("graphics vs {:#x} ({} words) ps {:#x} ({} words)", vs.address, vs.words, p.address, p.words),
            None => format!("graphics vs {:#x} ({} words)", vs.address, vs.words),
        };
        build_pipeline(cache, fast, optimised, &what, &|cache, flags| create_graphics_pipeline(dev, cache, key, vs, ps, tess, mesh, flags))
    }
    move || {
        let (result, built) = build(&dev, cache, &key, &vs, ps.as_deref(), tess, mesh, fast, false);
        let rebuild = (built.tier == Tier::Fast && result.is_ok() && fast_pipelines_mode() != FastPipelines::Only).then(|| {
            let key = key.clone();
            Box::new(move || {
                let (result, built) = build(&dev, cache, &key, &vs, ps.as_deref(), tess, mesh, false, true);
                Compiled::Graphics { key, result, record: None, built, rebuild: None }
            }) as Rebuild
        });
        Compiled::Graphics { key, result, record, built, rebuild }
    }
}

/// The job translating module `key` of the prewarm list again from its
/// guest inputs.
fn translate_job(dev: Arc<GpuDevice>, compiler: Arc<dyn ShaderCompiler>, key: u64, source: Arc<super::sources::ModuleSource>) -> impl FnOnce() -> Compiled + Send + 'static {
    move || {
        let compiling = crate::perf::CompileGuard::start();
        let _watch = super::watch::start(format!("program {:#x} of the prewarm list (module {key:016x})", source.program.address));
        let t = std::time::Instant::now();
        // A list may come from another player (an imported shader cache):
        // a translation that panics fails like any other, so the
        // pipelines waiting for the list's modules are still queued.
        let compiled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| super::sources::retranslate(&*compiler, key, &source)))
            .unwrap_or_else(|_| Err(CompileError::Failed("the translation panicked".into())));
        let translate_ns = t.elapsed().as_nanos() as u64;
        let t = std::time::Instant::now();
        let address = source.program.address;
        let stage = super::sources::stage_of(source.program.stage).unwrap_or(ShaderStage::Unknown);
        let mut kept = None;
        let result = compiled.and_then(|c| {
            let src = super::sources::ModuleSource {
                program: source.program.clone(),
                request: source.request.clone(),
                reads: Arc::new(super::sources::CodeImage::from_words(c.code_reads.clone())),
            };
            // A relocatable module of the list serves other addresses only
            // while this translator still proves it address independent.
            let relocation = if super::sources::proven_relocatable(&src.request) {
                if !super::sources::proves_relocatable(&*compiler, key, &src, &c) {
                    return Err(CompileError::Failed("no longer relocatable".into()));
                }
                Some(Arc::new(CodeCheck::of(address, c.code_reads.iter().copied())))
            } else {
                None
            };
            kept = Some(Arc::new(src));
            match create_shader_module(&dev, &c.spirv) {
                Ok(module) => Ok((Arc::new(Module { module, reflection: c.reflection, address, words: c.spirv.len(), relocation }), c.spirv)),
                Err(e) => Err(CompileError::Failed(format!("vkCreateShaderModule: {e:?}"))),
            }
        });
        let module_ns = t.elapsed().as_nanos() as u64;
        drop(compiling);
        Compiled::Module { key, stage, address, compiler: compiler.name().to_string(), result, translate_ns, module_ns, source: kept, relocated: None }
    }
}

/// The compile job of the compute pipeline of `key` (as [`graphics_job`]).
fn compute_job(dev: Arc<GpuDevice>, cache: vk::PipelineCache, key: ComputeKey, cs: Arc<Module>, record: Option<super::prewarm::ComputeRecord>, fast: bool) -> impl FnOnce() -> Compiled + Send + 'static {
    fn build(dev: &GpuDevice, cache: vk::PipelineCache, key: &ComputeKey, cs: &Module, fast: bool, optimised: bool) -> (Result<vk::Pipeline, vk::Result>, Built) {
        let what = || format!("compute cs {:#x} ({} words)", cs.address, cs.words);
        build_pipeline(cache, fast, optimised, &what, &|cache, flags| create_compute_pipeline(dev, cache, key, cs, flags))
    }
    move || {
        let (result, built) = build(&dev, cache, &key, &cs, fast, false);
        let rebuild = (built.tier == Tier::Fast && result.is_ok() && fast_pipelines_mode() != FastPipelines::Only).then(|| {
            Box::new(move || {
                let (result, built) = build(&dev, cache, &key, &cs, false, true);
                Compiled::Compute { key, result, record: None, built, rebuild: None }
            }) as Rebuild
        });
        Compiled::Compute { key, result, record, built, rebuild }
    }
}

/// Create the graphics pipeline of `key` (any thread: the pipeline cache
/// synchronises itself).
fn create_graphics_pipeline(
    dev: &GpuDevice,
    pipeline_cache: vk::PipelineCache,
    key: &GraphicsKey,
    vs: &Module,
    ps: Option<&Module>,
    tess: Option<(vk::ShaderModule, vk::ShaderModule)>,
    mesh: bool,
    flags: vk::PipelineCreateFlags,
) -> Result<vk::Pipeline, vk::Result> {
    let ff = &key.ff;
    let caps = &dev.caps;
    let names: Vec<std::ffi::CString> = [Some(vs.reflection.entry()), ps.map(|p| p.reflection.entry())]
        .into_iter()
        .flatten()
        .map(|n| std::ffi::CString::new(n).unwrap_or_default())
        .collect();
    let main = c"main";
    let first = if mesh { vk::ShaderStageFlags::MESH_EXT } else { vk::ShaderStageFlags::VERTEX };
    // Mesh workgroups hold 32-lane subgroups where the device could run
    // them narrower.
    let mut mesh32 = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(32);
    let mut first_stage = vk::PipelineShaderStageCreateInfo::default().stage(first).module(vs.module).name(&names[0]);
    if mesh && caps.mesh_subgroup32 {
        first_stage = first_stage.push_next(&mut mesh32);
    }
    let mut stages = vec![first_stage];
    if let Some((tcs, tes)) = tess {
        stages.push(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::TESSELLATION_CONTROL).module(tcs).name(main));
        stages.push(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::TESSELLATION_EVALUATION).module(tes).name(main));
    }
    if let Some(p) = ps {
        stages.push(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(p.module).name(&names[1]));
    }
    let bindings: Vec<vk::VertexInputBindingDescription> = ff
        .vertex_input
        .bindings
        .iter()
        .enumerate()
        .map(|(i, b)| vk::VertexInputBindingDescription {
            binding: i as u32,
            stride: b.stride,
            input_rate: if b.per_instance { vk::VertexInputRate::INSTANCE } else { vk::VertexInputRate::VERTEX },
        })
        .collect();
    let attributes: Vec<vk::VertexInputAttributeDescription> = ff
        .vertex_input
        .attributes
        .iter()
        .map(|a| vk::VertexInputAttributeDescription { location: a.location, binding: a.binding, format: vk::Format::from_raw(a.format), offset: a.offset })
        .collect();
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default().vertex_binding_descriptions(&bindings).vertex_attribute_descriptions(&attributes);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::from_raw(ff.topology))
        .primitive_restart_enable(ff.primitive_restart);
    let tessellation = vk::PipelineTessellationStateCreateInfo::default().patch_control_points(3);
    let mut clip_control = vk::PipelineViewportDepthClipControlCreateInfoEXT::default().negative_one_to_one(true);
    let mut viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(ff.viewport_count.max(1)).scissor_count(ff.viewport_count.max(1));
    if ff.negative_one_to_one && caps.depth_clip_control {
        viewport = viewport.push_next(&mut clip_control);
    }
    let mut depth_clip = vk::PipelineRasterizationDepthClipStateCreateInfoEXT::default().depth_clip_enable(ff.depth_clip);
    let mut provoking = vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT::default().provoking_vertex_mode(vk::ProvokingVertexModeEXT::LAST_VERTEX);
    let mut raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::from_raw(ff.polygon_mode))
        .depth_clamp_enable(ff.depth_clamp && caps.depth_clamp)
        .line_width(1.0);
    if caps.depth_clip_enable {
        raster = raster.push_next(&mut depth_clip);
    }
    if ff.provoking_last && caps.provoking_vertex_last {
        raster = raster.push_next(&mut provoking);
    }
    let (grid, locations) = state::centered_sample_locations(ff.samples);
    let mut sample_locations = vk::PipelineSampleLocationsStateCreateInfoEXT::default().sample_locations_enable(true).sample_locations_info(
        vk::SampleLocationsInfoEXT::default()
            .sample_locations_per_pixel(samples_flag(ff.samples))
            .sample_location_grid_size(grid)
            .sample_locations(&locations),
    );
    let sample_mask = [ff.sample_mask];
    let mut multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(samples_flag(ff.samples));
    if ff.sample_mask != u32::MAX {
        multisample = multisample.sample_mask(&sample_mask);
    }
    if ff.sample_shading {
        multisample = multisample.sample_shading_enable(true).min_sample_shading(1.0);
    }
    if ff.centered_samples {
        multisample = multisample.push_next(&mut sample_locations);
    }
    let blends: Vec<vk::PipelineColorBlendAttachmentState> = ff.blend[..ff.color_count as usize].iter().map(|b| b.attachment()).collect();
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default().max_depth_bounds(1.0);
    let mut dynamic = vec![
        vk::DynamicState::VIEWPORT,
        vk::DynamicState::SCISSOR,
        vk::DynamicState::LINE_WIDTH,
        vk::DynamicState::DEPTH_BIAS,
        vk::DynamicState::BLEND_CONSTANTS,
        vk::DynamicState::STENCIL_COMPARE_MASK,
        vk::DynamicState::STENCIL_WRITE_MASK,
        vk::DynamicState::STENCIL_REFERENCE,
        vk::DynamicState::CULL_MODE,
        vk::DynamicState::FRONT_FACE,
        vk::DynamicState::DEPTH_TEST_ENABLE,
        vk::DynamicState::DEPTH_WRITE_ENABLE,
        vk::DynamicState::DEPTH_COMPARE_OP,
        vk::DynamicState::STENCIL_TEST_ENABLE,
        vk::DynamicState::STENCIL_OP,
        vk::DynamicState::DEPTH_BIAS_ENABLE,
    ];
    if caps.depth_bounds {
        dynamic.push(vk::DynamicState::DEPTH_BOUNDS);
        dynamic.push(vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE);
    }
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
    let formats: Vec<vk::Format> = ff.color_formats[..ff.color_count as usize].iter().map(|&f| vk::Format::from_raw(f)).collect();
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&formats)
        .depth_attachment_format(vk::Format::from_raw(ff.depth_format))
        .stencil_attachment_format(vk::Format::from_raw(ff.stencil_format));
    let has_depth = ff.depth_format != vk::Format::UNDEFINED.as_raw() || ff.stencil_format != vk::Format::UNDEFINED.as_raw();
    let mut info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic_state)
        .layout(key.layout)
        .flags(flags)
        .push_next(&mut rendering);
    if !mesh {
        info = info.vertex_input_state(&vertex_input).input_assembly_state(&input_assembly);
    }
    if tess.is_some() {
        info = info.tessellation_state(&tessellation);
    }
    if has_depth {
        info = info.depth_stencil_state(&depth_stencil);
    }
    // SAFETY: every referenced structure lives through the call.
    let pipelines = unsafe { dev.device.create_graphics_pipelines(pipeline_cache, &[info], None) }.map_err(|(_, e)| e)?;
    Ok(pipelines[0])
}

/// Create the compute pipeline of `key` (any thread).
fn create_compute_pipeline(dev: &GpuDevice, pipeline_cache: vk::PipelineCache, key: &ComputeKey, cs: &Module, flags: vk::PipelineCreateFlags) -> Result<vk::Pipeline, vk::Result> {
    let name = std::ffi::CString::new(cs.reflection.entry()).unwrap_or_default();
    // Wave32 programs on a device whose default subgroup is wider, wave64
    // programs on one whose default is narrower (the module says so).
    let size = if key.subgroup32 { 32 } else { cs.reflection.required_subgroup_size };
    let mut required = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(size);
    let mut stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(cs.module).name(&name);
    if size != 0 {
        stage = stage.push_next(&mut required);
    }
    // A module that keeps every buffer index in range runs without the
    // driver's storage-buffer robustness (images keep theirs).
    let mut robustness = vk::PipelineRobustnessCreateInfoEXT::default()
        .storage_buffers(vk::PipelineRobustnessBufferBehaviorEXT::DISABLED)
        .uniform_buffers(vk::PipelineRobustnessBufferBehaviorEXT::DEVICE_DEFAULT)
        .vertex_inputs(vk::PipelineRobustnessBufferBehaviorEXT::DEVICE_DEFAULT)
        .images(vk::PipelineRobustnessImageBehaviorEXT::DEVICE_DEFAULT);
    if dev.caps.pipeline_robustness && cs.reflection.own_buffer_checks {
        stage = stage.push_next(&mut robustness);
    }
    let mut flags = vk::PipelineCreateFlags::DISPATCH_BASE | flags;
    if dev.caps.pipeline_executable_info {
        flags |= vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR;
    }
    let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(key.layout).flags(flags);
    let t = std::time::Instant::now();
    log::trace!("ps5gpu: creating compute pipeline of module {:#018x}", key.cs);
    // SAFETY: valid create info.
    let pipelines = unsafe { dev.device.create_compute_pipelines(pipeline_cache, &[info], None) }.map_err(|(_, e)| e)?;
    if dev.caps.pipeline_executable_info {
        log::info!("[pipeline-stats] cs key {:#018x} created in {:.1} ms", key.cs, t.elapsed().as_secs_f64() * 1e3);
        log_pipeline_statistics(dev, pipelines[0], key.cs);
    }
    Ok(pipelines[0])
}

/// `RPS5E_PIPELINE_STATS=1`: the driver's statistics of every executable
/// of a pipeline (registers, instructions, ...), as one log line.
fn log_pipeline_statistics(dev: &GpuDevice, pipeline: vk::Pipeline, key: u64) {
    let ext = ash::khr::pipeline_executable_properties::Device::new(&dev.instance, &dev.device);
    let info = vk::PipelineInfoKHR::default().pipeline(pipeline);
    // SAFETY: a pipeline created with CAPTURE_STATISTICS on this device.
    let Ok(executables) = (unsafe { ext.get_pipeline_executable_properties(&info) }) else { return };
    for (i, e) in executables.iter().enumerate() {
        let ei = vk::PipelineExecutableInfoKHR::default().pipeline(pipeline).executable_index(i as u32);
        // SAFETY: as above, `i` indexes the executables just listed.
        let Ok(stats) = (unsafe { ext.get_pipeline_executable_statistics(&ei) }) else { continue };
        let mut line = format!("[pipeline-stats] cs key {key:#018x} {}:", e.name_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
        for st in &stats {
            let name = st.name_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            // SAFETY: the union member read is the one `format` names.
            let v = unsafe {
                match st.format {
                    vk::PipelineExecutableStatisticFormatKHR::BOOL32 => st.value.b32.to_string(),
                    vk::PipelineExecutableStatisticFormatKHR::INT64 => st.value.i64.to_string(),
                    vk::PipelineExecutableStatisticFormatKHR::UINT64 => st.value.u64.to_string(),
                    vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => st.value.f64.to_string(),
                    _ => "?".into(),
                }
            };
            line.push_str(&format!(" {name}={v}"));
        }
        log::info!("{line}");
    }
}

fn samples_flag(n: u32) -> vk::SampleCountFlags {
    match n {
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        _ => vk::SampleCountFlags::TYPE_1,
    }
}

/// `RPS5E_STAGING_RETAG=0`: staging regions stay reserved only until the
/// tick they were handed out in completes, even when a submit in the middle
/// of the draw recorded the commands reading them into a later tick.
fn staging_retag_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_STAGING_RETAG").is_ok_and(|v| v == "0"))
}

impl ShaderPipelines {
    /// After a draw or dispatch that began in tick `started`: when a submit
    /// inside it (a full upload ring, a download for a texture) moved the
    /// open tick, its uploaded words, stream copies and tables, handed out
    /// in the earlier tick, are read by commands of the current one. They
    /// stay reserved until that tick completes; otherwise the ring could
    /// hand their bytes to later work while the GPU has not read them yet.
    fn keep_staging_for(&mut self, ctx: &DrawContext, started: u64) {
        let now = ctx.sched.current_tick();
        if now != started && staging_retag_enabled() {
            self.staging.retag(started, now);
            self.stats.staging_retags += 1;
        }
    }

    fn draw_in_tick(&mut self, ctx: &mut DrawContext, draw: &ResolvedDraw) -> DrawOutcome {
        self.stats.draws += 1;
        if self.compiler.is_none() && !self.plan_only {
            self.stats.needs_shader += 1;
            return DrawOutcome::NeedsShader;
        }
        self.digest = 0;
        if self.speculative {
            // Indirect records are read on the committing thread; the
            // shared buffers and the prewarm list are set up there first.
            if draw.packet.kind.indirect() || !self.slice_ready() {
                ctx.bail(crate::slices::spec::Hole::Special);
                return DrawOutcome::Hole;
            }
            if self.swaps_seen != *self.t.swaps {
                // The committing thread swapped pipelines the recipes hold.
                self.recipes = Default::default();
                self.swaps_seen = *self.t.swaps;
            }
            let compiler = self.compiler.take();
            self.bail_why = None;
            let r = self.try_draw(ctx, draw, compiler.as_ref());
            self.compiler = compiler;
            if let Some(why) = self.bail_why.take() {
                ctx.bail(why);
            }
            if matches!(r, Err(Failure::StagingFull)) {
                ctx.bail(crate::slices::spec::Hole::Staging);
            }
            let o = match r {
                // Errors and compile skips are decided by the exact path.
                Err(Failure::Skip | Failure::NoGpuIndirect | Failure::NeedsShader) | Ok(_) if !ctx.bailed() => self.draw_outcome(draw, r),
                _ => {
                    if !ctx.bailed() {
                        ctx.bail(crate::slices::spec::Hole::Other);
                    }
                    DrawOutcome::Hole
                }
            };
            if o == DrawOutcome::Recorded {
                self.stats.recorded_draws += 1;
            }
            return o;
        }
        self.zero_buffers(ctx.sched);
        self.load_prewarm();
        self.harvest();
        self.retire_swapped(ctx.sched);
        if draw.packet.kind.indirect() {
            self.stats.indirect_draws += 1;
            if let Some(o) = self.gpu_indirect_draw(ctx, draw) {
                if trace::active() {
                    self.trace_draw(ctx, draw, o);
                    let p = &draw.packet;
                    trace::log(&format!("    gpu-indirect records {:#x} max {} stride {} count at {:#x}", p.indirect_address, p.indirect_count, p.stride, p.count_address));
                }
                return o;
            }
            let Some(records) = indirect::records(ctx, self.memory, draw) else {
                self.skip_why = "indirect records unreadable";
                return self.draw_outcome(draw, Err(Failure::Skip));
            };
            // Recorded in parts: a later part cannot be given up for a
            // pending compile once earlier ones are recorded.
            return super::unlocked_wait::forbidden(|| {
                records.iter().enumerate().filter_map(|(i, r)| indirect::direct_draw(draw, r, i as u32)).fold(DrawOutcome::Skipped, |o, sub| indirect::merge(o, self.draw(ctx, &sub)))
            });
        }
        // One reference held for the draw (the compiler is shared with the
        // compile workers: each clone writes its count).
        let compiler = self.compiler.clone();
        let mut r = self.try_draw(ctx, draw, compiler.as_ref());
        if matches!(r, Err(Failure::StagingFull)) {
            // The open tick filled the ring: submit it, then the draw
            // allocates from space its completion frees.
            ctx.sched.flush();
            r = self.try_draw(ctx, draw, compiler.as_ref());
        }
        let o = self.draw_outcome(draw, r);
        if o == DrawOutcome::Recorded {
            self.stats.recorded_draws += 1;
        }
        if trace::active() {
            self.trace_draw(ctx, draw, o);
        }
        o
    }

    fn dispatch_in_tick(&mut self, ctx: &mut DrawContext, dispatch: &ResolvedDispatch) -> DrawOutcome {
        self.stats.dispatches += 1;
        self.overwrites.clear();
        if self.compiler.is_none() && !self.plan_only {
            self.stats.needs_shader += 1;
            return DrawOutcome::NeedsShader;
        }
        self.zero_buffers(ctx.sched);
        self.load_prewarm();
        self.harvest();
        self.retire_swapped(ctx.sched);
        let compiler = self.compiler.clone();
        let mut r = self.try_dispatch(ctx, dispatch, compiler.clone());
        if matches!(r, Err(Failure::StagingFull)) {
            ctx.sched.flush();
            self.overwrites.clear();
            r = self.try_dispatch(ctx, dispatch, compiler);
        }
        let o = self.outcome(r);
        if o == DrawOutcome::Recorded {
            self.stats.recorded_dispatches += 1;
        }
        if trace::active() {
            let written: Vec<u64> = std::mem::take(&mut self.trace_written);
            let mut dumps: Vec<(u64, String)> = if o == DrawOutcome::Recorded { written.iter().map(|&id| (id, "img".to_string())).collect() } else { Vec::new() };
            dumps.extend(std::mem::take(&mut self.trace_sampled).into_iter().map(|id| (id, "tex".to_string())));
            let a = dispatch.packet.indirect_address;
            if a != 0 && trace::readbacks() {
                // The group counts the GPU read (waits for the GPU).
                ctx.buffers.download(ctx.sched, a, 12);
                let mut b = [0u8; 12];
                if self.memory.read_backing(a, &mut b) {
                    let w: Vec<u32> = b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                    self.trace_notes.push(format!("    indirect groups {w:?}"));
                }
            }
            self.trace_pass(
                ctx,
                &format!(
                    "dispatch cs={:#x} groups={:?} threads={:?} indirect={:#x} -> {o:?}",
                    dispatch.shader.address, dispatch.packet.groups, dispatch.threads, dispatch.packet.indirect_address
                ),
                &dumps,
            );
        }
        o
    }
}

/// `RPS5E_SERIALIZE` (diagnostic, off by default): `barrier` records a
/// full barrier before every draw and dispatch (each in a rendering scope of
/// its own), `wait` also submits after each one and waits for the GPU. A
/// glitch that goes away with `barrier` is a dependency between passes the
/// scheduler does not see; one that needs `wait` sits between the CPU and
/// the GPU (as when a frame trace serialises the frame).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Serialize {
    Off,
    Barrier,
    Wait,
}

fn serialize_mode() -> Serialize {
    static MODE: std::sync::OnceLock<Serialize> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("RPS5E_SERIALIZE").as_deref() {
        Ok("barrier") => {
            log::warn!("ps5gpu: RPS5E_SERIALIZE=barrier: a full barrier before every draw and dispatch (diagnostic, slow)");
            Serialize::Barrier
        }
        Ok("wait") => {
            log::warn!("ps5gpu: RPS5E_SERIALIZE=wait: every draw and dispatch waits for the GPU (diagnostic, very slow)");
            Serialize::Wait
        }
        _ => Serialize::Off,
    })
}

impl ShaderPipelines {
    fn serialize_before(ctx: &mut DrawContext) {
        if serialize_mode() != Serialize::Off {
            ctx.sched.full_barrier();
        }
    }

    fn serialize_after(ctx: &mut DrawContext) {
        if serialize_mode() == Serialize::Wait {
            ctx.sched.full_barrier();
            ctx.sched.flush_and_wait();
        }
    }
}

impl PipelineProvider for ShaderPipelines {
    fn draw(&mut self, ctx: &mut DrawContext, draw: &ResolvedDraw) -> DrawOutcome {
        self.compile_pending = false;
        Self::serialize_before(ctx);
        let started = ctx.sched.current_tick();
        let o = self.draw_in_tick(ctx, draw);
        self.keep_staging_for(ctx, started);
        Self::serialize_after(ctx);
        if self.compile_pending {
            // Counted when it is tried again.
            self.stats.draws -= 1;
        }
        o
    }

    fn dispatch(&mut self, ctx: &mut DrawContext, dispatch: &ResolvedDispatch) -> DrawOutcome {
        self.compile_pending = false;
        Self::serialize_before(ctx);
        let started = ctx.sched.current_tick();
        let o = self.dispatch_in_tick(ctx, dispatch);
        self.keep_staging_for(ctx, started);
        Self::serialize_after(ctx);
        if self.compile_pending {
            self.stats.dispatches -= 1;
        }
        o
    }

    fn slice_provider(&self) -> Option<Box<dyn PipelineProvider>> {
        if self.speculative || self.vertex_inputs.is_some() {
            return None;
        }
        match self.slice_instance() {
            Ok(p) => Some(Box::new(p)),
            Err(e) => {
                log::warn!("ps5gpu: slice provider: {e:?}");
                None
            }
        }
    }

    fn slice_ready(&self) -> bool {
        ShaderPipelines::slice_ready(self)
    }

    fn retag_tick(&mut self, from: u64, to: u64) {
        ShaderPipelines::retag_tick(self, from, to)
    }

    fn take_digest(&mut self) -> u64 {
        std::mem::take(&mut self.digest)
    }

    fn take_compile_pending(&mut self) -> bool {
        std::mem::take(&mut self.compile_pending)
    }

    fn swap_generation(&self) -> u64 {
        *self.t.swaps
    }

    fn indirect_thread_dimensions(&self, dispatch: &ResolvedDispatch) -> bool {
        self.gpu_thread_dimensions(dispatch)
    }

    fn frame_presented(&mut self) {
        self.presented_frames += 1;
        self.compile_wait_left = self.compile_wait;
        // Finished compiles are entered once a frame even when no draw asks.
        self.harvest();
        self.t.prewarm.w().tick();
        super::watch::check();
        if super::skiplog::enabled() {
            if let Some(line) = self.skip_log.frame_line(self.presented_frames) {
                log::info!("{line}");
            }
        }
    }

    fn take_overwrites(&mut self) -> Vec<(u64, u64)> {
        std::mem::take(&mut self.overwrites)
    }

    fn take_compressed_stores(&mut self) -> Vec<crate::render::StoredKeys> {
        std::mem::take(&mut self.compressed_stores)
    }

    fn gds(&mut self, sched: &mut crate::scheduler::Scheduler) -> Option<vk::Buffer> {
        self.zero_buffers(sched);
        Some(self.t.gds.buffer)
    }

    fn staging_usage(&self) -> (u64, u64) {
        self.staging.used()
    }

    fn settle_compiles(&mut self) {
        self.settle_rebuilds();
    }

    fn compile_report(&self) -> Option<String> {
        Some(self.t.compile_times.format())
    }

    fn summary(&self) -> Option<String> {
        let s = self.stats();
        let (f, m) = {
            let g = self.frontend.lock().unwrap_or_else(|e| e.into_inner());
            (g.stats, g.materialization_statistics())
        };
        Some(format!(
            "pipelines: recorded={}d/{}c needs_shader={} skipped={} errors={} programs={} (plan_fail={} decode_fail={} materialize_fail={}) modules={} (fail={}) gfx={} (hits {}) cs={} (hits {}) cpu_idx={} rect={} indirect={}/{}gpu/{}mesh cs-thread-dims={}gpu cs-group-clamps={} cs-empty-indirect={} heap_sets={} dummies={} view-memo={}/{}miss clamp-misses={} compiling={}/{}w/{}bw/{}q/{}ahead/{}inline/{}first recipes={}/{} stream-memo={} stream-copies={}/{}MiB/{}trim staging-retags={} isolated={:?} bind-reuse={} bindings={} res-reuse={}/{} ring={}/{}r miss={}/{}/{}i/s(why={} cand={}) prep-reuse={} prefetched={} deferred-table={} table={}/{}r/{}dwords module-key-misses={} mat=[same {} patched {} recycled {} new {} fallback {} rec hit {} miss {}] bda=[stages {} mirrored {} materialized {}]{}",
            s.recorded_draws,
            s.recorded_dispatches,
            s.needs_shader,
            s.skipped,
            s.errors,
            f.programs,
            f.plan_failures,
            f.decode_failures,
            f.materialize_failures,
            s.modules,
            s.compile_failures,
            s.graphics_pipelines,
            s.graphics_hits,
            s.compute_pipelines,
            s.compute_hits,
            s.cpu_index_conversions,
            s.rect_draws,
            s.indirect_draws,
            s.gpu_indirect_draws,
            s.gpu_indirect_mesh_draws,
            s.gpu_thread_dispatches,
            s.gpu_group_clamps,
            s.empty_indirect_dispatches,
            s.heap_sets,
            s.dummy_images,
            s.view_memo_hits,
            s.view_lookups,
            self.clamp_misses.get(),
            s.compiling_skips,
            s.compile_waits,
            s.compile_budget_waits,
            self.t.pool.as_ref().map_or(0, |p| p.in_flight()),
            s.lookahead_queued,
            s.inline_compiles,
            s.hastened_compiles,
            s.recipe_hits,
            s.recipe_misses,
            s.stream_memo_hits,
            s.stream_copies,
            s.stream_bytes >> 20,
            s.stream_trimmed,
            s.staging_retags,
            s.isolated,
            s.binding_reuses,
            s.descriptor_bindings,
            s.resource_reuses,
            s.resource_binds,
            s.ring_binds,
            s.ring_reuses,
            s.miss_buffers,
            s.miss_images,
            s.miss_samplers,
            s.ring_why,
            s.ring_candidates,
            f.prepared_reuses,
            s.prefetched_stages,
            s.deferred_table_words,
            s.table_compares,
            s.table_reuses,
            s.table_dwords,
            s.module_key_misses,
            m.replay_same,
            m.replay_patched,
            m.replay_recycled,
            m.replay_templates,
            m.replay_fallbacks,
            m.hits - m.replay_same - m.replay_patched,
            m.misses - m.replay_templates,
            s.bda_stages,
            s.bda_mirrored,
            s.bda_materialized,
            self.take_faults()
        ) + " " + &self.t.compile_times.format() + &crate::shader::stats::line()
            + &self.bindless_stats().map_or(String::new(), |b| {
                format!(
                    " bindless=[binds {} reads {} records {} looks {} maps {} slots {} tables {}]",
                    b.binds, b.reads, b.resolved, b.revalidated, b.maps, b.slots, b.tables
                )
            }))
    }
}

impl Drop for ShaderPipelines {
    fn drop(&mut self) {
        if self.speculative {
            // The tables belong to the committing thread's provider.
            return;
        }
        // The list first: the pool's shutdown waits for the jobs in flight
        // (seconds for a large pipeline) and the process exit may not wait
        // that long; what those jobs add is saved after.
        self.t.prewarm.w().save();
        if let Some(mut pool) = self.t.pool.w().take() {
            for c in pool.shutdown() {
                self.finish(c);
            }
        }
        self.t.prewarm.w().save();
        self.dev.wait_idle();
        let d = &self.dev.device;
        // SAFETY: device idle; every handle below belongs to this provider.
        unsafe {
            for &p in self.t.graphics.values().chain(self.t.compute.values()).chain(self.t.swapped_out.iter()) {
                d.destroy_pipeline(p, None);
            }
            for m in self.t.modules.values().flatten() {
                d.destroy_shader_module(m.module, None);
            }
            for &(a, b) in self.t.rect_modules.values().flatten() {
                d.destroy_shader_module(a, None);
                d.destroy_shader_module(b, None);
            }
        }
    }
}

/// The decisions of a draw that the speculative and the exact recording
/// must agree on (`RPS5E_SLICE_VERIFY`): pipeline and layout, rendering
/// area and formats, the guest ranges and images bound, the draw call. Host
/// buffers (mirror or stream copy), view handles and layouts may differ
/// between equivalent recordings and are left out.
#[allow(clippy::too_many_arguments)]
fn draw_digest(pipeline: vk::Pipeline, layout: &LayoutEntry, scope: &RenderingKey, ranges: &RangeList, writes: &SetWrites, bound: &[BoundStage], indexed: bool, d: &ResolvedDraw) -> u64 {
    use std::hash::{Hash, Hasher};
    // One byte per part (which part differs shows in the log), the rest a
    // hash of everything.
    let part = |f: &dyn Fn(&mut rustc_hash::FxHasher)| {
        let mut h = rustc_hash::FxHasher::default();
        f(&mut h);
        h.finish()
    };
    let parts = [
        part(&|h| (vk::Handle::as_raw(pipeline), vk::Handle::as_raw(layout.pipeline_layout), indexed, d.packet.count, d.packet.instance_count, d.index_offset).hash(h)),
        part(&|h| (scope.color_count, scope.width, scope.height, scope.layers, scope.provoking_last).hash(h)),
        part(&|h| ranges.iter().for_each(|r| (r.0, r.1, r.2).hash(h))),
        part(&|h| writes.bound_images.hash(h)),
        part(&|h| writes.buffers.iter().for_each(|e| e.range.hash(h))),
        part(&|h| bound.iter().for_each(|b| b.prepared.shader_data.hash(h))),
    ];
    let mut all = rustc_hash::FxHasher::default();
    parts.hash(&mut all);
    parts.iter().enumerate().fold(all.finish() & 0xFFFF, |acc, (i, p)| acc | ((p & 0xFF) << (16 + 8 * i)))
}

/// Component kind of a pixel output.
pub fn output_kind(kind: PixelOutputKind) -> ComponentKind {
    match kind {
        PixelOutputKind::Float => ComponentKind::Float,
        PixelOutputKind::Uint => ComponentKind::Uint,
        PixelOutputKind::Sint => ComponentKind::Sint,
    }
}

/// `RPS5E_EMPTY_INDIRECT_SKIP=0`: an indirect dispatch with a zero count
/// in a CPU-written record is recorded like any other.
fn empty_indirect_skip_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_EMPTY_INDIRECT_SKIP"))
}

/// `RPS5E_EARLY_FRAGMENT_TESTS=0`: pixel programs never declare early
/// fragment tests (the earlier behaviour: the driver decides, and programs
/// with stores always test late).
fn early_fragment_tests_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_EARLY_FRAGMENT_TESTS").is_ok_and(|v| v == "0"))
}

/// `RPS5E_INDIRECT_GROUP_CLAMP=1`: an indirect dispatch whose record counts
/// groups runs from a copy of the record the record kernel makes, all zero
/// when a count exceeds the device's group limits (a garbage or stale
/// record otherwise launches a near-endless dispatch the driver reports
/// as a lost device). Costs a kernel and a barrier per indirect dispatch.
fn indirect_group_clamp_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_INDIRECT_GROUP_CLAMP").is_ok_and(|v| v.trim() == "1"))
}

/// `RPS5E_UNEXPORTED_INPUTS=0`: pixel inputs naming a parameter the
/// vertex stage does not export keep their interpolation (an undefined
/// value), as before.
fn unexported_inputs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_UNEXPORTED_INPUTS").is_ok_and(|v| v == "0"))
}

/// The SPI_PS_INPUT_CNTL words with every interpolated input (of the first
/// `num_interp`) whose parameter is not in `exported` turned into a
/// default of 0 (DEFAULT_VAL 0, not flat; an FP16 input's high half too);
/// None when every input is exported.
pub fn default_unexported_inputs(cntl: &[u32; 32], num_interp: u32, exported: u64) -> Option<[u32; 32]> {
    let missing = |c: u32| c & 0x20 == 0 && exported & (1u64 << (c & 0x1F)) == 0;
    let n = (num_interp as usize).min(32);
    if !cntl[..n].iter().any(|&c| missing(c)) {
        return None;
    }
    let mut out = *cntl;
    for c in &mut out[..n] {
        if missing(*c) {
            *c = (*c & !0x700) | 0x20;
            if *c & (1 << 19) != 0 {
                *c = (*c & !0x60_0000) | 0x10_0000;
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod readable_runs_tests {
    use super::ReadableRuns;

    #[test]
    fn readable_runs_answer_as_a_scan_of_the_runs() {
        // Chains of contiguous mappings: a run from an earlier mapping of a
        // chain ends no later than one from a later mapping.
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut runs = ReadableRuns::default();
        let mut list: Vec<(u64, u64)> = Vec::new();
        for chain in 0..40u64 {
            let base = chain << 32;
            let maps: Vec<u64> = (0..=8).map(|k| base + k * 0x10_0000).collect();
            for _ in 0..3 {
                let first = (next() % 8) as usize;
                let (s, e) = (maps[first], maps[8]);
                if list.iter().any(|&(a, b)| a <= s && s < b) {
                    // Found by the run holding its start already.
                    continue;
                }
                runs.insert(s, e);
                list.push((s, e));
            }
        }
        for _ in 0..20000 {
            let address = (next() % 41) << 32 | next() % 0x90_0000;
            let scan = list.iter().filter(|&&(s, e)| s <= address && address < e).map(|r| r.1).max();
            assert_eq!(runs.end_of(address), scan, "{address:#x}");
        }
        runs.clear();
        assert_eq!(runs.end_of(list[0].0), None);
    }
}



#[cfg(test)]
mod unnormalized_tests {
    use ps5shader::resources::{ResourceSnapshot, SampledImagePair, ShaderResourceInfo};

    /// An image filtered by a FORCE_UNNORMALIZED sampler takes a one-level
    /// view at its base level (the host sampler cannot pick levels).
    #[test]
    fn images_of_unnormalized_samplers_are_single_level() {
        let mut info = ShaderResourceInfo::default();
        info.sampled_pairs.push(SampledImagePair { image: 0, sampler: 1, ..Default::default() });
        info.sampled_pairs.push(SampledImagePair { image: 1, sampler: 0, ..Default::default() });
        let mut snapshot = ResourceSnapshot::default();
        snapshot.samplers.push([0u32, 0, 0, 0].into_iter().collect());
        snapshot.samplers.push([1u32 << 15, 0, 0, 0].into_iter().collect());
        assert!(super::image_unnormalized(&info, &snapshot, 0));
        assert!(!super::image_unnormalized(&info, &snapshot, 1));
        assert!(!super::image_unnormalized(&info, &snapshot, 2), "not sampled");
    }

    /// FORCE_DEGAMMA (S# word 0 bit 20) makes its images read through an
    /// sRGB view.
    #[test]
    fn force_degamma_samplers_read_srgb_views() {
        let mut info = ShaderResourceInfo::default();
        info.sampled_pairs.push(SampledImagePair { image: 0, sampler: 1, ..Default::default() });
        info.sampled_pairs.push(SampledImagePair { image: 1, sampler: 0, ..Default::default() });
        let mut snapshot = ResourceSnapshot::default();
        snapshot.samplers.push([0u32, 0, 0, 0].into_iter().collect());
        snapshot.samplers.push([1u32 << 20, 0, 0, 0].into_iter().collect());
        assert!(super::image_force_degamma(&info, &snapshot, 0));
        assert!(!super::image_force_degamma(&info, &snapshot, 1));
        use ash::vk::Format as F;
        assert_eq!(crate::pipeline::sharps::srgb_twin(F::R8G8B8A8_UNORM), Some(F::R8G8B8A8_SRGB));
        assert_eq!(crate::pipeline::sharps::srgb_twin(F::BC7_UNORM_BLOCK), Some(F::BC7_SRGB_BLOCK));
        assert_eq!(crate::pipeline::sharps::srgb_twin(F::R16G16B16A16_SFLOAT), None);
    }
}

#[cfg(test)]
mod compile_mode_tests {
    use super::*;

    /// A relocatable module serves the program wherever the code words its
    /// translation read are the same (a table the game patched per load is
    /// not); the per-pixel masks pack 16 bits per pixel.
    #[test]
    fn relocation_checks_and_quad_mask_words() {
        let check = CodeCheck::of(0x1000, [(0x1100, 7), (0x1104, 9)]);
        let memory = |m: Vec<(u64, u32)>| move |a: u64| m.iter().find(|e| e.0 == a).map(|e| e.1);
        assert!(check.holds(0x1000, &memory(vec![(0x1100, 7), (0x1104, 9)])));
        assert!(check.holds(0x8_0000, &memory(vec![(0x8_0100, 7), (0x8_0104, 9)])));
        assert!(!check.holds(0x8_0000, &memory(vec![(0x8_0100, 7), (0x8_0104, 8)])));
        assert!(!check.holds(0x8_0000, &memory(vec![(0x8_0100, 7)])), "unreadable words fail");
        assert_eq!(pack_quad_masks([1, 0xF, 0x3, 0xFFFF]), [0x000F_0001, 0xFFFF_0003]);
    }

    #[test]
    fn compile_wait_values() {
        use std::time::Duration;
        assert_eq!(parse_compile_wait(None), Duration::MAX, "unset: exact");
        assert_eq!(parse_compile_wait(Some("exact")), Duration::MAX);
        assert_eq!(parse_compile_wait(Some("EXACT")), Duration::MAX);
        assert_eq!(parse_compile_wait(Some("3")), Duration::from_millis(3));
        assert_eq!(parse_compile_wait(Some("0")), Duration::ZERO);
        assert_eq!(parse_compile_wait(Some("nonsense")), Duration::MAX, "an unreadable value keeps the exact mode");
    }

    #[test]
    fn game_pool_sizes() {
        // Every core but two, for urgent work and the list alike.
        assert_eq!(pool_size(32, None), (30, 30));
        assert_eq!(pool_size(16, None), (14, 14));
        assert_eq!(pool_size(4, None), (2, 2));
        assert_eq!(pool_size(2, None), (2, 2));
        assert_eq!(pool_size(32, Some(8)), (8, 8));
        assert_eq!(pool_size(32, Some(1)), (1, 1));
    }
}
