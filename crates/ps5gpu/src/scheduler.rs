// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Timeline-semaphore submission scheduler.
//!
//! Every submission signals the next value ("tick") of one timeline
//! semaphore. The tick a command buffer will signal is known while it is
//! still recording ([`Scheduler::current_tick`]), so work can be attached to
//! "everything recorded so far":
//!
//! * [`Scheduler::queue_priority`] runs a callback on the dedicated
//!   `rps5e-gpu-priority` thread once the tick has completed on the GPU
//!   (end-of-pipe interrupts, flip completion);
//! * [`Scheduler::queue_completion`] runs a callback on the owning thread
//!   the next time it calls [`Scheduler::run_completed`] after the tick
//!   completed;
//! * [`Scheduler::retire`] keeps a resource alive until the tick completed.
//!
//! The scheduler is owned by one thread at a time (the GPU back-end lock);
//! the priority worker shares only the timeline and its own queue.

use crate::device::GpuDevice;
use ash::vk;
use std::any::Any;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

pub mod occlusion;
pub mod gpu_timer;
pub mod hazard;

/// A journal entry of the work recorded in a tick (formatted only when a
/// tick hangs: draws are journaled thousands of times per frame).
pub enum Note {
    Text(String),
    Draw { count: u32, instances: u32, vs: u64, ps: u64, colors: usize, depth: bool },
    /// The `n`th dispatch of compute program `cs`.
    Dispatch { n: u64, cs: u64, packet: crate::render::DispatchPacket, threads: [u32; 3] },
}

impl std::fmt::Display for Note {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Note::Text(t) => f.write_str(t),
            Note::Draw { count, instances, vs, ps, colors, depth } => {
                write!(f, "draw n={count} inst={instances} vs={vs:#x} ps={ps:#x} colors={colors} depth={depth}")
            }
            Note::Dispatch { n, cs, packet, threads } => write!(f, "dispatch #{n} of cs={cs:#x} packet={packet:?} threads={threads:?}"),
        }
    }
}

/// The timeline semaphore and its tick counters.
pub struct Timeline {
    dev: Arc<GpuDevice>,
    pub semaphore: vk::Semaphore,
    /// Next tick to reserve (the tick of the recording command buffer).
    next: AtomicU64,
    /// Highest tick known to have completed.
    completed: AtomicU64,
    /// Recent work per tick, reported when a tick does not complete.
    journal: Mutex<VecDeque<(u64, Note)>>,
}

/// Journal entries kept (the newest win).
const JOURNAL_LEN: usize = 4096;

/// Move `notes` (oldest first) to the end of `journal`, which keeps its
/// newest `len` entries.
fn append_journal(journal: &mut VecDeque<(u64, Note)>, notes: &mut Vec<(u64, Note)>, len: usize) {
    let keep = notes.len().min(len);
    let excess = (journal.len() + keep).saturating_sub(len);
    journal.drain(..excess.min(journal.len()));
    let skip = notes.len() - keep;
    journal.extend(notes.drain(..).skip(skip));
}

/// Hazard spans kept before everything is ordered at the next look.
const MAX_HAZARD_SPANS: usize = 1 << 16;

// SAFETY: the semaphore handle is only used with thread-safe Vulkan calls
// (wait / get counter value), and signalled through the locked queue.
unsafe impl Send for Timeline {}
unsafe impl Sync for Timeline {}

impl Timeline {
    pub fn new(dev: &Arc<GpuDevice>) -> Result<Arc<Timeline>, vk::Result> {
        let mut ty = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut ty);
        // SAFETY: valid create info.
        let semaphore = unsafe { dev.device.create_semaphore(&info, None) }?;
        Ok(Arc::new(Timeline { dev: dev.clone(), semaphore, next: AtomicU64::new(1), completed: AtomicU64::new(0), journal: Mutex::new(VecDeque::new()) }))
    }

    /// Tick the next submission will signal.
    pub fn current_tick(&self) -> u64 {
        self.next.load(Ordering::Acquire)
    }

    /// Highest tick known complete (call [`Timeline::refresh`] to update).
    pub fn completed_tick(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }

    fn reserve(&self) -> u64 {
        self.next.fetch_add(1, Ordering::AcqRel)
    }

    pub fn refresh(&self) -> u64 {
        // SAFETY: valid timeline semaphore.
        let v = unsafe { self.dev.device.get_semaphore_counter_value(self.semaphore) }.unwrap_or(0);
        self.completed.fetch_max(v, Ordering::AcqRel);
        self.completed_tick()
    }

    pub fn is_complete(&self, tick: u64) -> bool {
        self.completed_tick() >= tick || self.refresh() >= tick
    }

    /// Block until `tick` completed. A tick that was never submitted would
    /// block forever; callers only wait on submitted ticks.
    pub fn wait(&self, tick: u64) {
        // A lost device completes nothing: the waiting thread stops (the
        // process is shutting down) instead of failing every later wait.
        if crate::device::is_lost() {
            crate::device::stop_if_lost_or_shutting_down();
        }
        if self.is_complete(tick) {
            return;
        }
        crate::slice_shadow::hole(crate::slice_shadow::Hole::FlushWait);
        let sems = [self.semaphore];
        let values = [tick];
        let info = vk::SemaphoreWaitInfo::default().semaphores(&sems).values(&values);
        let mut reported = false;
        // The completed tick and when it last advanced.
        let mut progress = (self.completed_tick(), std::time::Instant::now());
        loop {
            // SAFETY: valid wait info.
            match unsafe { self.dev.device.wait_semaphores(&info, 1_000_000_000) } {
                Ok(()) => break,
                Err(vk::Result::TIMEOUT) => {
                    // A GPU that stopped making progress is not waited for
                    // past a shutdown request.
                    crate::device::stop_if_lost_or_shutting_down();
                    let done = self.refresh();
                    log::warn!("ps5gpu: still waiting for GPU tick {tick} (completed {done})");
                    if !reported {
                        reported = true;
                        self.report_stuck(done + 1);
                    }
                    if done != progress.0 {
                        progress = (done, std::time::Instant::now());
                    } else if progress.1.elapsed() >= hang_timeout() {
                        // The driver may keep a hung queue without ever
                        // reporting the loss: the GPU counts as lost.
                        log::error!("ps5gpu: GPU tick {} made no progress for {:?}: the GPU is hung", done + 1, hang_timeout());
                        self.dev.report_hang();
                        crate::device::stop_if_lost_or_shutting_down();
                    }
                }
                Err(vk::Result::ERROR_DEVICE_LOST) => {
                    if !crate::device::is_lost() {
                        log::error!("ps5gpu: vkWaitSemaphores(tick {tick}) failed: ERROR_DEVICE_LOST");
                        self.report_stuck(self.refresh() + 1);
                    }
                    // Logs the driver's fault report once and starts shutdown.
                    self.dev.report_device_fault();
                    crate::device::stop_if_lost_or_shutting_down();
                }
                Err(e) => {
                    log::error!("ps5gpu: vkWaitSemaphores(tick {tick}) failed: {e:?}");
                    break;
                }
            }
        }
        self.completed.fetch_max(tick, Ordering::AcqRel);
    }
}

/// How long the GPU may complete no work at all while a thread waits for
/// it before it counts as hung (`RPS5E_GPU_HANG_TIMEOUT` seconds, 0: never;
/// default 30). Other processes' GPU hangs stall ours for a few seconds only.
fn hang_timeout() -> Duration {
    static T: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *T.get_or_init(|| match std::env::var("RPS5E_GPU_HANG_TIMEOUT").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(30) {
        0 => Duration::MAX,
        s => Duration::from_secs(s),
    })
}

impl Timeline {
    /// Record journal entries of earlier ticks' work (drained), in order.
    pub fn note_all(&self, notes: &mut Vec<(u64, Note)>) {
        if notes.is_empty() {
            return;
        }
        append_journal(&mut self.journal.lock().unwrap_or_else(|e| e.into_inner()), notes, JOURNAL_LEN);
    }

    /// The most recent journal entry.
    pub fn last_note(&self) -> Option<String> {
        self.journal.lock().unwrap_or_else(|e| e.into_inner()).back().map(|(_, w)| w.to_string())
    }

    /// Log the journaled work of the first tick that did not complete.
    fn report_stuck(&self, tick: u64) {
        let j = self.journal.lock().unwrap_or_else(|e| e.into_inner());
        let work: Vec<&Note> = j.iter().filter(|(t, _)| *t == tick).map(|(_, w)| w).collect();
        log::error!("ps5gpu: GPU tick {tick} does not complete; it contains {} journaled item(s):", work.len());
        for w in work {
            log::error!("ps5gpu:   tick {tick}: {w}");
        }
    }
}

impl Drop for Timeline {
    fn drop(&mut self) {
        // SAFETY: the owner waited for the device before dropping.
        unsafe { self.dev.device.destroy_semaphore(self.semaphore, None) };
    }
}

type Callback = Box<dyn FnOnce() + Send>;

struct PriorityQueue {
    items: VecDeque<(u64, Callback)>,
    /// Tick of the callback being run (0 when idle).
    active: u64,
    stop: bool,
}

/// The "GPU priority" worker: runs callbacks after their tick completes.
struct PriorityWorker {
    timeline: Arc<Timeline>,
    queue: Mutex<PriorityQueue>,
    cv: Condvar,
}

thread_local! {
    static IN_CALLBACK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// True while the calling thread runs a scheduler completion callback.
pub fn in_completion_callback() -> bool {
    IN_CALLBACK.with(|c| c.get())
}

fn run_callback(f: Callback) {
    let prev = IN_CALLBACK.with(|c| c.replace(true));
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err() {
        log::error!("ps5gpu: GPU completion callback panicked");
    }
    IN_CALLBACK.with(|c| c.set(prev));
}

impl PriorityWorker {
    fn run(self: Arc<Self>) {
        loop {
            let (tick, f) = {
                let mut q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    if q.stop && q.items.is_empty() {
                        return;
                    }
                    if let Some(item) = q.items.pop_front() {
                        q.active = item.0;
                        break item;
                    }
                    q = self.cv.wait(q).unwrap_or_else(|e| e.into_inner());
                }
            };
            self.timeline.wait(tick);
            run_callback(f);
            let mut q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
            q.active = 0;
            self.cv.notify_all();
        }
    }

    /// Wait until every priority callback for ticks <= `tick` has run.
    fn drain_until(&self, tick: u64) {
        let mut q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        while (q.active != 0 && q.active <= tick) || q.items.front().is_some_and(|i| i.0 <= tick) {
            q = self.cv.wait_timeout(q, Duration::from_millis(50)).unwrap_or_else(|e| e.into_inner()).0;
        }
    }
}

/// Counters for diagnostics.
#[derive(Clone, Copy, Debug, Default)]
pub struct SchedulerStats {
    pub submits: u64,
    pub command_buffers: usize,
    pub retired_pending: usize,
    /// Rendering scopes ended, and draws that continued an open one.
    pub rendering_scopes: u64,
    pub merged_draws: u64,
    /// Full (all-commands) barriers and image barriers recorded.
    pub full_barriers: u64,
    /// Barriers from the compute stage recorded before dispatches whose
    /// ranges overlap earlier work ([`hazard::DispatchBarrier`]), and
    /// barriers ordering transfer reads after compute writes.
    pub compute_barriers: u64,
    pub transfer_read_barriers: u64,
    pub image_barriers: u64,
    /// Why rendering scopes ended: other attachments, image barriers,
    /// `cmd` work, ordered work, a full barrier, a submit.
    pub scope_ends: [u64; 6],
    /// Copies recorded into the upload prologue, and prologues submitted.
    pub prologue_copies: u64,
    pub prologues: u64,
    /// GPU execution time of the command buffers (`RPS5E_GPU_PROFILE=1`).
    pub gpu: gpu_timer::GpuTime,
}

/// Bumped by work outside the provider's draws that binds pipelines,
/// descriptors, push constants or dynamic state ([`disturb_state`]).
static STATE_DISTURBED: AtomicU64 = AtomicU64::new(0);

/// Work other than the provider's draws (copy kernels, the DCC resolver,
/// blit passes, dispatches) binds pipelines, descriptors, push constants
/// or dynamic state: draws record their whole state again.
pub fn disturb_state() {
    STATE_DISTURBED.fetch_add(1, Ordering::AcqRel);
}

/// The attachments and area of a dynamic-rendering scope: consecutive
/// draws with equal keys share one scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderingKey {
    pub colors: [vk::ImageView; 8],
    pub color_count: u32,
    pub depth: vk::ImageView,
    pub depth_attachment: bool,
    pub stencil_attachment: bool,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    /// The attachments are in GENERAL layout (also bound as resources).
    pub general: bool,
    /// The layout of the depth/stencil attachment (read-only depth, depth
    /// read-only with stencil written, or writable).
    pub depth_layout: vk::ImageLayout,
    /// The last vertex provokes, on a device whose pipelines of one
    /// rendering scope must share the provoking vertex mode.
    pub provoking_last: bool,
}

/// Command recording + submission with completion callbacks.
pub struct Scheduler {
    pub dev: Arc<GpuDevice>,
    pub timeline: Arc<Timeline>,
    pool: vk::CommandPool,
    /// Command buffers with the tick they were last submitted with.
    ring: Vec<(vk::CommandBuffer, u64)>,
    /// Ring slot to search from next.
    hint: usize,
    current: Option<vk::CommandBuffer>,
    /// The upload prologue of the open tick (see [`Self::prologue_cmd`]).
    prologue: Option<vk::CommandBuffer>,
    retired: VecDeque<(u64, Box<dyn Any + Send>)>,
    /// Bytes retired with [`Self::retire_sized`] into the open tick.
    open_retired_bytes: u64,
    pending: VecDeque<(u64, Callback)>,
    worker: Arc<PriorityWorker>,
    thread: Option<std::thread::JoinHandle<()>>,
    stats: SchedulerStats,
    /// Debug label of the recording buffer (reported when a submit fails).
    pub debug_label: &'static str,
    /// Journal entries of the open tick, moved into the timeline's
    /// journal when the tick is submitted (one lock per submit instead of
    /// one per draw; only submitted ticks can hang).
    notes: Vec<(u64, Note)>,
    device_lost: Arc<AtomicBool>,
    /// The open rendering scope of the recording buffer: draws to the same
    /// attachments continue it; any other command ends it first.
    rendering: Option<RenderingKey>,
    /// Serial of the open (or last) rendering scope.
    scope_serial: u64,
    /// Changes when the recording buffer changes or work outside the draws
    /// binds pipelines, descriptors or push constants (see
    /// [`Self::disturb_state`]): state draws bound stays bound until then,
    /// across rendering scopes.
    state_serial: u64,
    /// Work recorded through [`Scheduler::cmd`] since the last full
    /// barrier: the next rendering scope begins after a full barrier.
    dirty: bool,
    /// Rendering scopes ended without a barrier (only image barriers
    /// followed them) since the last full barrier: work recorded through
    /// [`Scheduler::cmd`] waits for them behind a full barrier.
    unfenced: bool,
    /// Guest buffer ranges written by dispatches recorded without a full
    /// barrier after them, and ranges read by work since the last full
    /// barrier: later work touching them is ordered after it (see
    /// [`hazard`]).
    hazards: hazard::Hazards,
    /// Shader work (draws, dispatches) was recorded since the last full
    /// barrier: work writing memory waits for it (write after read).
    shader_work: bool,
    /// Transfers into host memory were recorded into the open buffer: it
    /// ends with a barrier making them visible to host reads.
    host_reads: bool,
    /// Guest occlusion queries counted on the GPU (created on first use).
    occlusion: Option<Box<occlusion::Occlusion>>,
    gpu_timer: Option<Box<gpu_timer::GpuTimer>>,
    /// Ended command buffers of the open tick, submitted after the
    /// prologue and before the recording buffer, in order: slice segments
    /// recorded on other threads and the seams between them (R7b,
    /// [`Self::splice`]); `true` for buffers of this scheduler's ring.
    chain: Vec<(vk::CommandBuffer, bool)>,
    /// The next recording buffer begins with a full barrier (a spliced
    /// segment's work precedes it and none of its accesses are tracked).
    seam: bool,
    /// Slice mode (R7b): a recorder of one slice thread (see
    /// [`Self::new_slice`]).
    slice: Option<SliceMode>,
}

/// The state of a slice-mode scheduler (R7b, [`Scheduler::new_slice`]).
#[derive(Default)]
struct SliceMode {
    /// Something asked for work only the committing thread may do (a
    /// submit, a wait, a completion callback, a host read-back): the
    /// recording since the last [`Scheduler::take_slice_bail`] is unusable.
    bailed: bool,
    /// Calls that recorded into the command buffer (see
    /// [`Scheduler::recorded_count`]).
    recorded: u64,
}

// SAFETY: the command pool and buffers are only used by the thread that
// currently owns the scheduler (it lives behind the back-end lock).
unsafe impl Send for Scheduler {}

impl Scheduler {
    pub fn new(dev: &Arc<GpuDevice>) -> Result<Scheduler, vk::Result> {
        let timeline = Timeline::new(dev)?;
        // SAFETY: valid create info.
        let pool = unsafe {
            dev.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(dev.queue_family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }?;
        let worker = Arc::new(PriorityWorker {
            timeline: timeline.clone(),
            queue: Mutex::new(PriorityQueue { items: VecDeque::new(), active: 0, stop: false }),
            cv: Condvar::new(),
        });
        let w = worker.clone();
        let thread = std::thread::Builder::new()
            .name("rps5e-gpu-priority".into())
            .spawn(move || w.run())
            .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
        Ok(Scheduler {
            dev: dev.clone(),
            timeline,
            pool,
            ring: Vec::new(),
            hint: 0,
            current: None,
            prologue: None,
            retired: VecDeque::new(),
            open_retired_bytes: 0,
            pending: VecDeque::new(),
            worker,
            thread: Some(thread),
            stats: SchedulerStats::default(),
            debug_label: "",
            notes: Vec::new(),
            device_lost: Arc::new(AtomicBool::new(false)),
            rendering: None,
            scope_serial: 0,
            state_serial: 0,
            dirty: false,
            unfenced: false,
            hazards: hazard::Hazards::default(),
            shader_work: false,
            host_reads: false,
            occlusion: None,
            // Spliced segments split a tick into several buffers the timer
            // does not follow: no GPU timing with parallel slices.
            gpu_timer: if crate::backend::profile::on() && crate::slices::configured() <= 1 { gpu_timer::GpuTimer::new(dev).map(Box::new) } else { None },
            chain: Vec::new(),
            seam: false,
            slice: None,
        })
    }

    /// A recorder for one slice thread (R7b stage 2): its own command pool
    /// and buffers, rendering scope, barrier and hazard state, sharing
    /// `main`'s timeline (its ticks are provisional: the open tick of the
    /// committed state when the slice was recorded). It never submits,
    /// waits or queues callbacks: such requests mark the recording
    /// unusable ([`Self::take_slice_bail`]) and do nothing. Its segments
    /// ([`Self::end_segment`]) are spliced into `main`'s submission by the
    /// committing thread ([`Self::splice`]).
    pub fn new_slice(main: &Scheduler) -> Result<Scheduler, vk::Result> {
        let dev = &main.dev;
        // SAFETY: valid create info.
        let pool = unsafe {
            dev.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(dev.queue_family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }?;
        Ok(Scheduler {
            dev: dev.clone(),
            timeline: main.timeline.clone(),
            pool,
            ring: Vec::new(),
            hint: 0,
            current: None,
            prologue: None,
            retired: VecDeque::new(),
            open_retired_bytes: 0,
            pending: VecDeque::new(),
            worker: main.worker.clone(),
            thread: None,
            stats: SchedulerStats::default(),
            debug_label: "",
            device_lost: main.device_lost.clone(),
            rendering: None,
            scope_serial: 0,
            state_serial: 0,
            dirty: false,
            unfenced: false,
            hazards: hazard::Hazards::default(),
            shader_work: false,
            host_reads: false,
            occlusion: None,
            gpu_timer: None,
            notes: Vec::new(),
            chain: Vec::new(),
            seam: false,
            slice: Some(SliceMode::default()),
        })
    }

    /// A slice-mode scheduler ([`Self::new_slice`]).
    pub fn is_slice(&self) -> bool {
        self.slice.is_some()
    }

    /// Slice mode: whether work only the committing thread may do was
    /// asked for since the last call (the flag is reset).
    pub fn take_slice_bail(&mut self) -> bool {
        self.slice.as_mut().is_some_and(|s| std::mem::take(&mut s.bailed))
    }

    /// Slice mode: mark the recording unusable.
    fn slice_bail(&mut self) -> bool {
        match self.slice.as_mut() {
            Some(s) => {
                s.bailed = true;
                true
            }
            None => false,
        }
    }

    /// Slice mode: how many times a command buffer was handed out for
    /// recording (a draw that bails after the count moved left commands
    /// behind).
    pub fn recorded_count(&self) -> u64 {
        self.slice.as_ref().map_or(0, |s| s.recorded)
    }

    /// Slice mode: end the open segment — its rendering scope ends without
    /// a barrier and its recording buffer and upload prologue are ended
    /// and returned (prologue, buffer); the next command begins a new
    /// segment, in the state after a full barrier (the committing thread
    /// records one, the seam, before each segment it splices).
    pub fn end_segment(&mut self) -> (Option<vk::CommandBuffer>, Option<vk::CommandBuffer>) {
        debug_assert!(self.slice.is_some());
        self.end_rendering_plain(5);
        if std::mem::take(&mut self.host_reads) {
            if let Some(c) = self.current {
                Self::record_host_read_barrier(&self.dev, c);
            }
        }
        let prologue = self.prologue.take();
        if let Some(p) = prologue {
            Self::record_full_barrier(&self.dev, p);
        }
        let cb = self.current.take();
        for c in prologue.iter().chain(cb.iter()) {
            // SAFETY: recording buffer of this scheduler's pool.
            if let Err(e) = unsafe { self.dev.device.end_command_buffer(*c) } {
                log::error!("ps5gpu: vkEndCommandBuffer (slice) failed: {e:?}");
            }
        }
        self.dirty = false;
        self.unfenced = false;
        self.shader_work = false;
        self.hazards.full_barrier();
        (prologue, cb)
    }

    /// Slice mode: buffers of this scheduler handed to the committing
    /// thread are in use until `tick` completes (0: never submitted, free
    /// again).
    pub fn release(&mut self, cbs: &[vk::CommandBuffer], tick: u64) {
        for c in cbs {
            if let Some(slot) = self.ring.iter_mut().find(|(r, _)| r == c) {
                slot.1 = tick;
            }
        }
    }

    /// Slice mode: the objects retired while recording (kept alive by the
    /// committing thread from the tick its segments were submitted in).
    pub fn take_retired(&mut self) -> Vec<Box<dyn Any + Send>> {
        self.open_retired_bytes = 0;
        self.retired.drain(..).map(|(_, o)| o).collect()
    }

    /// Splice ended command buffers recorded elsewhere (a slice segment:
    /// `cbs`, its prologue first) into the open tick after the work
    /// recorded so far: the recording buffer ends behind a full barrier
    /// and `barriers` (the segment's image layout requirements, the seam)
    /// and the next recording buffer begins with a full barrier.
    pub fn splice(&mut self, barriers: &[vk::ImageMemoryBarrier2], cbs: &[vk::CommandBuffer]) {
        debug_assert!(self.slice.is_none());
        self.end_rendering_plain(5);
        let cb = self.raw_cmd();
        self.barrier_raw(cb);
        if !barriers.is_empty() {
            // SAFETY: recording buffer outside any rendering scope.
            unsafe { self.dev.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(barriers)) };
            self.stats.image_barriers += barriers.len() as u64;
        }
        if std::mem::take(&mut self.host_reads) {
            Self::record_host_read_barrier(&self.dev, cb);
        }
        if let Some(t) = &mut self.gpu_timer {
            t.end(cb);
        }
        // SAFETY: recording buffer owned by this scheduler.
        if let Err(e) = unsafe { self.dev.device.end_command_buffer(cb) } {
            log::error!("ps5gpu: vkEndCommandBuffer failed: {e:?} ({})", self.debug_label);
        }
        self.current = None;
        self.chain.push((cb, true));
        self.chain.extend(cbs.iter().map(|&c| (c, false)));
        self.seam = true;
    }

    /// Time the GPU work recorded until [`Self::time_item_end`] under
    /// `label` (`RPS5E_GPU_PROFILE=1`; a shader address).
    /// Returns the label it replaces, for [`Self::time_item_end`].
    #[inline]
    pub fn time_item_begin(&mut self, label: u64) -> Option<u64> {
        let cb = self.current;
        Some(self.gpu_timer.as_mut()?.enter(cb, label))
    }

    /// [`Self::time_item_end`] after commands were recorded straight into
    /// the recording buffer (not through [`Self::cmd`] and its kin).
    #[inline]
    fn time_recorded(&mut self, prev: Option<u64>) {
        if let Some(t) = &mut self.gpu_timer {
            t.touch();
        }
        self.time_item_end(prev);
    }

    /// The work begun by [`Self::time_item_begin`] (which returned `prev`)
    /// was recorded: `prev` is the active label again.
    #[inline]
    pub fn time_item_end(&mut self, prev: Option<u64>) {
        let (Some(prev), Some(t)) = (prev, &mut self.gpu_timer) else { return };
        t.enter(self.current, prev);
    }

    /// The shaders with the most GPU time (profile only).
    /// GPU time per draw / dispatch label so far (`RPS5E_GPU_PROFILE=1`).
    pub fn gpu_labels(&self) -> Vec<(u64, (u64, u64, u64))> {
        self.gpu_timer.as_ref().map(|t| t.labels()).unwrap_or_default()
    }

    pub fn gpu_top(&self, n: usize) -> String {
        self.gpu_timer.as_ref().map(|t| t.top(n)).unwrap_or_default()
    }

    /// The tick the recording command buffer will signal.
    pub fn current_tick(&self) -> u64 {
        self.timeline.current_tick()
    }

    pub fn is_complete(&self, tick: u64) -> bool {
        self.timeline.is_complete(tick)
    }

    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            command_buffers: self.ring.len(),
            retired_pending: self.retired.len(),
            gpu: self.gpu_timer.as_ref().map(|t| t.time).unwrap_or_default(),
            ..self.stats
        }
    }

    pub fn device_lost(&self) -> bool {
        self.device_lost.load(Ordering::Relaxed)
    }

    /// True while commands are being recorded for the current tick.
    pub fn is_recording(&self) -> bool {
        self.current.is_some() || self.prologue.is_some() || !self.chain.is_empty()
    }

    /// A free command buffer: searched round-robin from the slot after the
    /// last one taken, first against the known completed tick, then after
    /// reading the timeline, then from the start; four more are allocated
    /// when none is free.
    fn acquire_buffer(&mut self) -> Result<vk::CommandBuffer, vk::Result> {
        let n = self.ring.len();
        let search = |ring: &[(vk::CommandBuffer, u64)], from: usize, to: usize, done: u64| (from..to).find(|&i| ring[i].1 <= done);
        let mut done = self.timeline.completed_tick();
        let mut found = search(&self.ring, self.hint.min(n), n, done);
        if found.is_none() {
            done = self.timeline.refresh();
            found = search(&self.ring, self.hint.min(n), n, done);
        }
        if found.is_none() {
            found = search(&self.ring, 0, self.hint.min(n), done);
        }
        if let Some(i) = found {
            self.ring[i].1 = u64::MAX;
            self.hint = (i + 1) % n;
            return Ok(self.ring[i].0);
        }
        // SAFETY: valid pool.
        let fresh = unsafe {
            self.dev.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(4),
            )
        }?;
        let first = fresh[0];
        self.hint = (n + 1) % (n + fresh.len());
        for (i, cb) in fresh.into_iter().enumerate() {
            self.ring.push((cb, if i == 0 { u64::MAX } else { 0 }));
        }
        Ok(first)
    }

    /// The recording command buffer (begun on first use), outside any
    /// rendering scope: an open one ends here (with a full barrier, the
    /// ordering every draw had with the work after it), and so do scopes
    /// ended earlier without one. The next rendering scope begins after a
    /// full barrier.
    pub fn cmd(&mut self) -> vk::CommandBuffer {
        if !self.end_rendering_scope_for(2) && self.unfenced {
            let cb = self.raw_cmd();
            self.barrier_raw(cb);
        }
        self.dirty = true;
        self.raw_cmd()
    }

    /// The recording command buffer for work that orders itself with
    /// precise barriers (image layout transitions, buffer barriers whose
    /// source is every earlier command): an open rendering scope ends
    /// without a barrier, and later draws need none for this work.
    pub fn ordered_cmd(&mut self) -> vk::CommandBuffer {
        self.end_rendering_plain(3);
        self.raw_cmd()
    }

    /// Work recorded through [`Self::cmd`] since the last full barrier
    /// (it may have written anything).
    pub fn has_unordered_work(&self) -> bool {
        self.dirty
    }

    /// Shader work was recorded since the last full barrier.
    pub fn has_shader_work(&self) -> bool {
        self.shader_work
    }

    /// One of `ranges` overlaps a range shader work wrote since the last
    /// full barrier.
    pub fn shader_write_overlaps(&self, ranges: impl IntoIterator<Item = (u64, u64)>) -> bool {
        self.hazards.writes.overlaps_any(ranges)
    }

    /// Shader work wrote guest ranges since the last full barrier (work
    /// reading memory it has no ranges for must wait for it).
    pub fn has_unordered_shader_writes(&self) -> bool {
        !self.hazards.writes.is_empty()
    }

    /// Shader work writing `written` (guest buffer ranges) was recorded
    /// without a barrier after it: later work that touches those ranges,
    /// work recorded through [`Self::cmd`] and work writing memory get a
    /// full barrier first; images order themselves by their layouts.
    pub fn note_shader_writes(&mut self, written: &[(u64, u64)]) {
        self.note_dispatch(&[], written, false);
    }

    /// A dispatch reading `reads` and writing `written` (guest buffer
    /// ranges; `unknown_reads`: and memory it has no ranges for) was
    /// recorded without a barrier after it: later work touching those
    /// ranges is ordered after it ([`Self::dispatch_barrier`], and a full
    /// barrier before draws, work recorded through [`Self::cmd`] and host
    /// reads); images order themselves by their layouts.
    pub fn note_dispatch(&mut self, reads: &[(u64, u64)], written: &[(u64, u64)], unknown_reads: bool) {
        self.shader_work = true;
        self.unfenced = true;
        self.hazards.note_dispatch(reads, written, unknown_reads);
        if self.hazards.spans_exceed(MAX_HAZARD_SPANS) {
            // Too many to keep: order everything at the next look.
            self.dirty = true;
        }
    }

    /// The dependency a dispatch reading `reads` and writing `written`
    /// needs on earlier work recorded since the last full barrier (see
    /// [`hazard`]); record it with [`Self::dispatch_barrier_record`].
    pub fn dispatch_barrier(&mut self, reads: &[(u64, u64)], written: &[(u64, u64)], unknown_reads: bool) -> hazard::DispatchBarrier {
        self.hazards.dispatch_barrier(reads, written, unknown_reads)
    }

    /// Record `b` (outside any rendering scope; an open one ends without a
    /// barrier): earlier compute writes become visible to compute shaders
    /// and indirect reads, and later compute work waits for the readers.
    pub fn dispatch_barrier_record(&mut self, b: hazard::DispatchBarrier) {
        let hazard::DispatchBarrier::Compute { graphics, transfer } = b else { return };
        let mut src = vk::PipelineStageFlags2::COMPUTE_SHADER;
        if graphics {
            src |= vk::PipelineStageFlags2::ALL_GRAPHICS;
        }
        if transfer {
            src |= vk::PipelineStageFlags2::ALL_TRANSFER;
        }
        let cb = self.ordered_cmd();
        let mb = vk::MemoryBarrier2::default()
            .src_stage_mask(src)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::DRAW_INDIRECT)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::UNIFORM_READ | vk::AccessFlags2::INDIRECT_COMMAND_READ);
        // SAFETY: recording buffer outside any rendering scope.
        unsafe { self.dev.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mb))) };
        self.hazards.compute_barrier_recorded(graphics, transfer);
        self.stats.compute_barriers += 1;
    }

    /// The next draw recorded into a rendering scope reads `reads` (guest
    /// buffer ranges; `unknown`: and memory it has no ranges for). A draw
    /// that does not report them counts as reading everything.
    pub fn note_draw_reads(&mut self, reads: impl IntoIterator<Item = (u64, u64)>, unknown: bool) {
        self.hazards.note_draw_reads(reads, unknown);
    }

    /// The recording command buffer for transfers that only read `ranges`
    /// of guest buffer mirrors (copies out to host memory nothing on the
    /// GPU reads): an open rendering scope ends without a barrier (draws
    /// there wrote no buffer), and they wait for earlier compute writes to
    /// those ranges only. Later dispatches writing the ranges wait for
    /// them. Work recorded through [`Self::cmd`] before (it may have
    /// written anything) gets a full barrier first.
    pub fn transfer_read_cmd(&mut self, ranges: &[(u64, u64)]) -> vk::CommandBuffer {
        self.end_rendering_plain(2);
        let cb = self.raw_cmd();
        if self.dirty {
            self.barrier_raw(cb);
        } else if self.hazards.transfer_read_barrier(ranges) {
            let mb = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
            // SAFETY: recording buffer outside any rendering scope.
            unsafe { self.dev.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mb))) };
            self.hazards.transfer_barrier_recorded();
            self.stats.transfer_read_barriers += 1;
        }
        self.hazards.note_transfer_reads(ranges);
        // Transfers recorded through `cmd` (uploads writing these ranges)
        // wait for the copies.
        self.unfenced = true;
        cb
    }

    /// Transfer writes recorded so far become visible to host reads once
    /// the submission completed: one barrier at the end of the recording
    /// buffer (nothing on the GPU waits for it).
    pub fn host_read_barrier(&mut self) {
        // A host read waits for the submission (committing thread only).
        self.slice_bail();
        self.host_reads = true;
    }

    fn record_host_read_barrier(dev: &GpuDevice, cb: vk::CommandBuffer) {
        let mb = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ);
        // SAFETY: recording buffer outside any rendering scope.
        unsafe { dev.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mb))) };
    }

    /// End the open rendering scope, if any, without a barrier.
    fn end_rendering_plain(&mut self, reason: usize) -> bool {
        if self.rendering.take().is_none() {
            return false;
        }
        self.stats.scope_ends[reason] += 1;
        let cb = self.raw_cmd();
        self.end_occlusion_segment(cb);
        // SAFETY: recording buffer inside the scope begun by begin_rendering_scope.
        unsafe { self.dev.device.cmd_end_rendering(cb) };
        self.publish_occlusion_results(cb);
        self.unfenced = true;
        self.stats.rendering_scopes += 1;
        true
    }

    /// End the open rendering scope, if any, followed by a full barrier.
    /// True when one was open.
    pub fn end_rendering_scope(&mut self) -> bool {
        self.end_rendering_scope_for(4)
    }

    fn end_rendering_scope_for(&mut self, reason: usize) -> bool {
        if !self.end_rendering_plain(reason) {
            return false;
        }
        let cb = self.raw_cmd();
        self.barrier_raw(cb);
        true
    }

    /// Continue the open rendering scope when its key equals `key` and no
    /// image needs a barrier, else end it (without a barrier: draws in a
    /// scope only read memory, images are ordered by `barriers`) and begin
    /// `info` after the image barriers, behind a full barrier when other
    /// work was recorded since the last one. Returns the command buffer to
    /// record the draw into (use [`Self::rendering_cmd`] for later commands
    /// of the draw).
    /// The open rendering scope when it is `key`'s and no barrier has to
    /// come first: the draw continues it (as [`Self::begin_rendering_scope`]
    /// would) and its rendering info need not be built.
    pub fn continue_rendering_scope(&mut self, key: &RenderingKey, barriers: bool) -> Option<vk::CommandBuffer> {
        if barriers || self.rendering.as_ref() != Some(key) {
            return None;
        }
        self.hazards.draw_recorded();
        self.stats.merged_draws += 1;
        Some(self.raw_cmd())
    }

    pub fn begin_rendering_scope(&mut self, key: RenderingKey, info: &vk::RenderingInfo, barriers: &[vk::ImageMemoryBarrier2]) -> vk::CommandBuffer {
        self.hazards.draw_recorded();
        if self.rendering == Some(key) && barriers.is_empty() {
            self.stats.merged_draws += 1;
            return self.raw_cmd();
        }
        let reason = if self.rendering == Some(key) { 1 } else { 0 };
        self.end_rendering_plain(reason);
        let cb = self.raw_cmd();
        if self.dirty {
            self.barrier_raw(cb);
        }
        if !barriers.is_empty() {
            let t = self.time_item_begin(gpu_timer::label::IMAGE_BARRIER);
            // SAFETY: recording buffer outside any rendering scope.
            unsafe { self.dev.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(barriers)) };
            self.time_recorded(t);
            self.stats.image_barriers += barriers.len() as u64;
        }
        // SAFETY: recording buffer outside any rendering scope.
        unsafe { self.dev.device.cmd_begin_rendering(cb, info) };
        self.shader_work = true;
        self.rendering = Some(key);
        self.scope_serial += 1;
        self.begin_occlusion_segment(cb);
        cb
    }

    /// Record image barriers (outside any rendering scope; an open one
    /// ends without a barrier).
    pub fn image_barriers(&mut self, barriers: &[vk::ImageMemoryBarrier2]) {
        if barriers.is_empty() {
            return;
        }
        let cb = self.ordered_cmd();
        let t = self.time_item_begin(gpu_timer::label::IMAGE_BARRIER);
        // SAFETY: recording buffer outside any rendering scope.
        unsafe { self.dev.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(barriers)) };
        self.time_recorded(t);
        self.stats.image_barriers += barriers.len() as u64;
    }

    /// Record a memory barrier from `src` to `dst` (outside any rendering
    /// scope; an open one ends without a barrier).
    pub fn memory_barrier(&mut self, src: (vk::PipelineStageFlags2, vk::AccessFlags2), dst: (vk::PipelineStageFlags2, vk::AccessFlags2)) {
        let cb = self.ordered_cmd();
        let t = self.time_item_begin(gpu_timer::label::BARRIER);
        let mb = vk::MemoryBarrier2::default().src_stage_mask(src.0).src_access_mask(src.1).dst_stage_mask(dst.0).dst_access_mask(dst.1);
        // SAFETY: recording buffer outside any rendering scope.
        unsafe { self.dev.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mb))) };
        self.time_recorded(t);
    }

    /// Serial of the open rendering scope: draws recorded under the same
    /// serial follow each other with nothing in between.
    pub fn scope_serial(&self) -> u64 {
        self.scope_serial
    }

    /// Serial of the bound draw state (see `state_serial`).
    pub fn state_serial(&self) -> u64 {
        self.state_serial.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ STATE_DISTURBED.load(Ordering::Acquire)
    }

    /// The recording buffer inside the open rendering scope (state, push
    /// data and descriptors of the draw being recorded).
    pub fn rendering_cmd(&mut self) -> vk::CommandBuffer {
        debug_assert!(self.rendering.is_some());
        self.raw_cmd()
    }

    /// The key of the open rendering scope.
    pub fn rendering_scope(&self) -> Option<&RenderingKey> {
        self.rendering.as_ref()
    }

    /// The upload prologue of the open tick: a command buffer submitted
    /// right before the recording buffer, behind a full barrier against
    /// every earlier submission and followed by one, for transfers into
    /// memory that no work of the open tick has touched yet (buffer
    /// uploads a draw needs, recorded without ending its rendering scope).
    pub fn prologue_cmd(&mut self) -> vk::CommandBuffer {
        if let Some(c) = self.prologue {
            return c;
        }
        let cb = match self.acquire_buffer() {
            Ok(cb) => cb,
            Err(e) => panic!("ps5gpu: cannot allocate a command buffer: {e:?}"),
        };
        // SAFETY: the buffer is not in use (its tick completed).
        unsafe {
            self.dev.device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()).expect("reset command buffer");
            self.dev
                .device
                .begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .expect("begin command buffer");
        }
        Self::record_full_barrier(&self.dev, cb);
        if let Some(t) = &mut self.gpu_timer {
            t.prologue_begin(cb);
        }
        if let Some(s) = self.slice.as_mut() {
            s.recorded += 1;
        }
        self.prologue = Some(cb);
        self.stats.prologues += 1;
        cb
    }

    /// Count a copy recorded into the prologue.
    pub fn note_prologue_copy(&mut self) {
        self.stats.prologue_copies += 1;
    }

    fn record_full_barrier(dev: &GpuDevice, cb: vk::CommandBuffer) {
        let mb = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
        // SAFETY: recording buffer.
        unsafe { dev.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mb))) };
    }

    fn raw_cmd(&mut self) -> vk::CommandBuffer {
        if let Some(s) = self.slice.as_mut() {
            s.recorded += 1;
        }
        if let Some(c) = self.current {
            if let Some(t) = &mut self.gpu_timer {
                t.touch();
            }
            return c;
        }
        let cb = match self.acquire_buffer() {
            Ok(cb) => cb,
            Err(e) => panic!("ps5gpu: cannot allocate a command buffer: {e:?}"),
        };
        // SAFETY: the buffer is not in use (its tick completed).
        unsafe {
            self.dev.device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()).expect("reset command buffer");
            self.dev
                .device
                .begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .expect("begin command buffer");
        }
        if let Some(t) = &mut self.gpu_timer {
            t.begin(cb);
            t.touch();
        }
        self.current = Some(cb);
        self.state_serial += 1;
        if std::mem::take(&mut self.seam) {
            // A spliced segment precedes this buffer.
            self.barrier_raw(cb);
        }
        cb
    }

    /// Submit the recording command buffer (or an empty batch) signalling
    /// the current tick, with extra semaphore waits/signals. Returns the tick.
    pub fn submit_with(
        &mut self,
        waits: &[vk::SemaphoreSubmitInfo],
        signals: &[vk::SemaphoreSubmitInfo],
    ) -> u64 {
        crate::slice_shadow::hole(crate::slice_shadow::Hole::FlushWait);
        if self.slice_bail() {
            return self.current_tick();
        }
        // Nothing is submitted to a lost device.
        if crate::device::is_lost() {
            crate::device::stop_if_lost_or_shutting_down();
        }
        self.end_rendering_scope_for(5);
        self.open_retired_bytes = 0;
        if std::mem::take(&mut self.host_reads) {
            if let Some(c) = self.current {
                Self::record_host_read_barrier(&self.dev, c);
            }
        }
        let cb = self.current.take();
        let prologue = self.prologue.take();
        if let (Some(t), Some(c)) = (&mut self.gpu_timer, cb) {
            t.end(c);
        }
        for c in prologue.iter().chain(cb.iter()) {
            if Some(*c) == prologue {
                if let Some(t) = &mut self.gpu_timer {
                    t.prologue_end(*c);
                }
                Self::record_full_barrier(&self.dev, *c);
            }
            // SAFETY: recording buffer owned by this scheduler.
            if let Err(e) = unsafe { self.dev.device.end_command_buffer(*c) } {
                log::error!("ps5gpu: vkEndCommandBuffer failed: {e:?} ({})", self.debug_label);
            }
        }
        self.timeline.note_all(&mut self.notes);
        let queue = self.dev.queue();
        let tick = self.timeline.reserve();
        if let (Some(t), Some(_)) = (&mut self.gpu_timer, cb) {
            t.submitted(tick);
        }
        let mut sigs: Vec<vk::SemaphoreSubmitInfo> = signals.to_vec();
        sigs.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.timeline.semaphore)
                .value(tick)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        );
        let chain = std::mem::take(&mut self.chain);
        let cbis: Vec<vk::CommandBufferSubmitInfo> = prologue
            .iter()
            .chain(chain.iter().map(|(c, _)| c))
            .chain(cb.iter())
            .map(|&c| vk::CommandBufferSubmitInfo::default().command_buffer(c))
            .collect();
        let submit = vk::SubmitInfo2::default().wait_semaphore_infos(waits).command_buffer_infos(&cbis).signal_semaphore_infos(&sigs);
        // SAFETY: queue access is serialised by `queue`; all handles valid.
        let r = unsafe { self.dev.device.queue_submit2(*queue, &[submit], vk::Fence::null()) };
        drop(queue);
        if let Err(e) = r {
            log::error!("ps5gpu: vkQueueSubmit2 (tick {tick}, {}) failed: {e:?}", self.debug_label);
            if e == vk::Result::ERROR_DEVICE_LOST {
                self.device_lost.store(true, Ordering::Relaxed);
                self.dev.report_device_fault();
                crate::device::stop_if_lost_or_shutting_down();
            }
            // Keep tick accounting consistent: signal the tick from the host.
            let si = vk::SemaphoreSignalInfo::default().semaphore(self.timeline.semaphore).value(tick);
            // SAFETY: valid semaphore; the value is larger than any prior one.
            unsafe {
                let _ = self.dev.device.signal_semaphore(&si);
            }
        }
        for c in prologue.iter().chain(chain.iter().filter(|(_, own)| *own).map(|(c, _)| c)).chain(cb.iter()) {
            if let Some(slot) = self.ring.iter_mut().find(|(r, _)| r == c) {
                slot.1 = tick;
            }
        }
        self.stats.submits += 1;
        self.debug_label = "";
        tick
    }

    /// Journal `what` as work of the open tick (reported when its tick
    /// does not complete).
    pub fn note(&mut self, what: Note) {
        if self.slice.is_some() {
            // Slice segments are noted by the committing thread.
            return;
        }
        if self.notes.len() >= JOURNAL_LEN {
            self.timeline.note_all(&mut self.notes);
        }
        let tick = self.current_tick();
        self.notes.push((tick, what));
    }

    /// The most recent journal entry, of the open tick or a submitted one.
    pub fn last_note(&self) -> Option<String> {
        match self.notes.last() {
            Some((_, w)) => Some(w.to_string()),
            None => self.timeline.last_note(),
        }
    }

    /// Submit everything recorded so far. Returns the signalled tick.
    pub fn flush(&mut self) -> u64 {
        self.submit_with(&[], &[])
    }

    /// Submit and wait for the GPU to finish it.
    pub fn flush_and_wait(&mut self) -> u64 {
        let t = self.flush();
        if self.slice.is_some() {
            return t;
        }
        self.timeline.wait(t);
        self.run_completed();
        t
    }

    /// Wait for `tick`, submitting the recording buffer first when `tick`
    /// is the current one.
    pub fn wait(&mut self, tick: u64) {
        if self.slice_bail() {
            return;
        }
        if tick >= self.current_tick() {
            let t = self.flush();
            self.timeline.wait(t.min(tick));
        } else {
            self.timeline.wait(tick);
        }
    }

    /// Submit, wait for every submitted tick and run completion work,
    /// including priority callbacks up to the last tick.
    pub fn finish(&mut self) {
        if self.slice_bail() {
            return;
        }
        let t = self.flush();
        self.timeline.wait(t);
        self.run_completed();
        self.worker.drain_until(t);
    }

    /// Keep `obj` alive until the current tick completes.
    pub fn retire<T: Send + 'static>(&mut self, obj: T) {
        let t = self.current_tick();
        self.retired.push_back((t, Box::new(obj)));
    }

    /// [`Self::retire`] of an object holding `bytes` of memory (a temporary
    /// upload buffer, a mirror replaced by a merge): counted until the open
    /// tick is submitted ([`Self::open_retired_bytes`]).
    pub fn retire_sized<T: Send + 'static>(&mut self, obj: T, bytes: u64) {
        self.open_retired_bytes += bytes;
        self.retire(obj);
    }

    /// Bytes of memory retired into the open tick (freed once it completed).
    pub fn open_retired_bytes(&self) -> u64 {
        self.open_retired_bytes
    }

    /// Run `f` on the owning thread after the current tick completes (see
    /// [`Scheduler::run_completed`]).
    pub fn queue_completion(&mut self, f: impl FnOnce() + Send + 'static) {
        if self.slice_bail() {
            return;
        }
        let t = self.current_tick();
        self.pending.push_back((t, Box::new(f)));
    }

    /// Run `f` on the GPU priority thread after the current tick completes.
    /// The caller must eventually submit the current tick ([`Scheduler::flush`]).
    pub fn queue_priority(&mut self, f: impl FnOnce() + Send + 'static) {
        let t = self.current_tick();
        self.queue_priority_at(t, f);
    }

    /// Run `f` on the GPU priority thread after `tick` completes (submitted
    /// ticks, or the current one; callbacks run in queue order, so ticks
    /// must not decrease).
    pub fn queue_priority_at(&mut self, t: u64, f: impl FnOnce() + Send + 'static) {
        if self.slice_bail() {
            return;
        }
        let mut q = self.worker.queue.lock().unwrap_or_else(|e| e.into_inner());
        q.items.push_back((t, Box::new(f)));
        self.worker.cv.notify_all();
    }

    /// True while priority callbacks are queued or running.
    pub fn has_pending_priority(&self) -> bool {
        let q = self.worker.queue.lock().unwrap_or_else(|e| e.into_inner());
        !q.items.is_empty() || q.active != 0
    }

    /// Wait until every priority callback of a tick up to `tick` has run.
    pub fn wait_priority(&self, tick: u64) {
        self.worker.drain_until(tick);
    }

    /// Release retired resources and run owner-thread callbacks whose tick
    /// completed.
    pub fn run_completed(&mut self) {
        if self.slice.is_some() {
            return;
        }
        let done = self.timeline.refresh();
        if let Some(t) = &mut self.gpu_timer {
            t.collect(done);
        }
        while self.retired.front().is_some_and(|(t, _)| *t <= done) {
            self.retired.pop_front();
        }
        while self.pending.front().is_some_and(|(t, _)| *t <= done) {
            let (t, f) = self.pending.pop_front().unwrap();
            self.worker.drain_until(t);
            run_callback(f);
        }
    }

    /// Record a full pipeline barrier (simple, conservative ordering between
    /// transfer/compute/graphics work recorded before and after).
    pub fn full_barrier(&mut self) {
        // Ending an open rendering scope records the barrier.
        if self.end_rendering_scope() {
            return;
        }
        let cb = self.raw_cmd();
        self.barrier_raw(cb);
    }

    fn barrier_raw(&mut self, cb: vk::CommandBuffer) {
        self.dirty = false;
        self.unfenced = false;
        self.shader_work = false;
        self.hazards.full_barrier();
        self.stats.full_barriers += 1;
        let mb = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
        let dep = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mb));
        let t = self.time_item_begin(gpu_timer::label::FULL_BARRIER);
        // SAFETY: recording buffer.
        unsafe { self.dev.device.cmd_pipeline_barrier2(cb, &dep) };
        self.time_recorded(t);
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        if self.slice.is_some() {
            // The worker and the timeline belong to the main scheduler.
            self.dev.wait_idle();
            self.retired.clear();
            // SAFETY: device idle; buffers belong to the pool.
            unsafe { self.dev.device.destroy_command_pool(self.pool, None) };
            return;
        }
        self.finish();
        {
            let mut q = self.worker.queue.lock().unwrap_or_else(|e| e.into_inner());
            q.stop = true;
            self.worker.cv.notify_all();
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.dev.wait_idle();
        self.retired.clear();
        // SAFETY: device idle; buffers belong to the pool.
        unsafe { self.dev.device.destroy_command_pool(self.pool, None) };
    }
}

/// A host-visible upload ring with tick-based reclamation. Allocations that
/// do not fit fall back to a temporary buffer retired with the tick.
pub struct StagingRing {
    pub buffer: crate::device::Buffer,
    head: u64,
    /// (start, end, tick) of in-flight regions, oldest first.
    inflight: VecDeque<(u64, u64, u64)>,
}

impl StagingRing {
    pub fn new(dev: &Arc<GpuDevice>, size: u64, usage: vk::BufferUsageFlags, mem: crate::device::MemoryUse) -> Result<StagingRing, vk::Result> {
        Ok(StagingRing { buffer: crate::device::Buffer::new(dev, size, usage, mem)?, head: 0, inflight: VecDeque::new() })
    }

    fn reclaim(&mut self, completed: u64) {
        while self.inflight.front().is_some_and(|r| r.2 <= completed) {
            self.inflight.pop_front();
        }
    }

    /// Reserve `size` bytes aligned to `align` for use by the current tick
    /// of `sched`. Returns the buffer offset, or None when it does not fit.
    pub fn alloc(&mut self, sched: &Scheduler, size: u64, align: u64) -> Option<u64> {
        let cap = self.buffer.size;
        if size == 0 || size > cap {
            return None;
        }
        // The known completed tick first: the semaphore is read only when
        // the ring has no room without it.
        self.reclaim(sched.timeline.completed_tick());
        let tick = sched.current_tick();
        let align = align.max(1);
        let mut refreshed = false;
        loop {
            let start = self.head.div_ceil(align) * align;
            let (lo, hi) = (start, start + size);
            let wraps = hi > cap;
            let (lo, hi) = if wraps { (0, size) } else { (lo, hi) };
            let busy = self.inflight.iter().any(|&(s, e, _)| s < hi && lo < e);
            if !busy {
                self.head = hi;
                // Regions of one tick are consecutive: keep them as one
                // entry, so the overlap scan stays short with thousands
                // of small uploads per tick.
                match self.inflight.back_mut() {
                    Some(b) if b.2 == tick && b.1 <= lo && (lo - b.1) < align => b.1 = hi,
                    _ => self.inflight.push_back((lo, hi, tick)),
                }
                return Some(lo);
            }
            if !refreshed {
                refreshed = true;
                self.reclaim(sched.timeline.refresh());
                continue;
            }
            // Wait for the oldest region (a large request may overlap
            // several) while it belongs to submitted work; regions of the
            // open tick need a submit by the caller first.
            crate::slice_shadow::hole(crate::slice_shadow::Hole::StagingFull);
            match self.inflight.front() {
                Some(&(_, _, t)) if t < tick => {
                    sched.timeline.wait(t);
                    self.reclaim(sched.timeline.completed_tick().max(t));
                }
                _ => return None,
            }
        }
    }

    pub fn bytes(&self, offset: u64, size: u64) -> &mut [u8] {
        &mut self.buffer.bytes_mut()[offset as usize..(offset + size) as usize]
    }

    /// The regions handed out since tick `from` are read by commands of
    /// tick `to` (a submit in the middle of a draw moved its commands into
    /// a later tick than its allocations): they stay reserved until `to`
    /// completes, not only until `from` does.
    pub fn retag(&mut self, from: u64, to: u64) {
        retag_regions(&mut self.inflight, from, to);
    }

    /// Whether `[offset, offset+size)` lies in a region still reserved for
    /// a tick at least `through` (after `completed` is reclaimed): a copy
    /// of an earlier tick can only be bound again while its bytes were not
    /// handed out anew.
    pub fn holds(&mut self, completed: u64, offset: u64, size: u64, through: u64) -> bool {
        self.reclaim(completed);
        let (lo, hi) = (offset, offset + size);
        self.inflight.iter().any(|&(s, e, t)| s <= lo && hi <= e && t >= through)
    }

    /// Bytes still reserved for uncompleted ticks, and the ring's size.
    pub fn used(&self) -> (u64, u64) {
        (self.inflight.iter().map(|&(s, e, _)| e - s).sum(), self.buffer.size)
    }
}

/// Move the regions of ticks `from..to` to tick `to` (oldest-first order
/// and tick order are kept: older regions have smaller ticks).
fn retag_regions(inflight: &mut VecDeque<(u64, u64, u64)>, from: u64, to: u64) {
    if to <= from {
        return;
    }
    for r in inflight.iter_mut().rev() {
        if r.2 < from {
            break;
        }
        if r.2 < to {
            r.2 = to;
        }
    }
}

#[cfg(test)]
mod staging_tests {
    use super::{retag_regions, StagingRing};
    use std::collections::VecDeque;

    #[test]
    fn regions_of_a_draw_moved_by_a_submit_wait_for_the_later_tick() {
        let mut inflight: VecDeque<(u64, u64, u64)> = [(0, 100, 4), (100, 200, 5), (200, 300, 5), (300, 400, 6)].into_iter().collect();
        retag_regions(&mut inflight, 5, 6);
        assert_eq!(inflight.iter().map(|r| r.2).collect::<Vec<_>>(), [4, 6, 6, 6]);
        // A completed tick 5 no longer frees them.
        let completed = 5;
        while inflight.front().is_some_and(|r| r.2 <= completed) {
            inflight.pop_front();
        }
        assert_eq!(inflight.len(), 3);
        // Nothing moves without a tick change.
        retag_regions(&mut inflight, 6, 6);
        assert_eq!(inflight.iter().map(|r| r.2).collect::<Vec<_>>(), [6, 6, 6]);
    }

    /// `StagingRing::holds`: a copy of an earlier tick is bindable again
    /// only while its region was not handed out anew.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn holds_reports_whether_a_regions_bytes_are_still_reserved() {
        let Ok(dev) = crate::device::GpuDevice::create(false) else { return };
        let mut ring = StagingRing::new(&dev, 1 << 20, ash::vk::BufferUsageFlags::TRANSFER_SRC, crate::device::MemoryUse::Upload).unwrap();
        // Regions of "tick" 4 and 6; tick 4 completed and was reclaimed.
        ring.inflight.push_back((0, 100, 4));
        ring.inflight.push_back((100, 200, 6));
        ring.reclaim(4);
        // The tick-4 region is gone: not held through 4 any more.
        assert!(!ring.holds(4, 16, 16, 4));
        // The tick-6 region is held through 6 (and any lower tick).
        assert!(ring.holds(4, 100, 64, 6));
        // A range past the region's end is not held.
        assert!(!ring.holds(4, 150, 100, 6));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn journal_keeps_the_newest_entries_in_order() {
        let text = |j: &VecDeque<(u64, Note)>| j.iter().map(|(t, w)| format!("{t}:{w}")).collect::<Vec<_>>();
        let mut j = VecDeque::new();
        let mut notes: Vec<(u64, Note)> = (0..3).map(|i| (1, Note::Text(format!("a{i}")))).collect();
        append_journal(&mut j, &mut notes, 4);
        assert!(notes.is_empty());
        assert_eq!(text(&j), ["1:a0", "1:a1", "1:a2"]);
        // Older entries make room for newer ones.
        let mut notes: Vec<(u64, Note)> = (0..2).map(|i| (2, Note::Text(format!("b{i}")))).collect();
        append_journal(&mut j, &mut notes, 4);
        assert_eq!(text(&j), ["1:a1", "1:a2", "2:b0", "2:b1"]);
        // More new entries than the journal holds: only the newest stay.
        let mut notes: Vec<(u64, Note)> = (0..6).map(|i| (3, Note::Text(format!("c{i}")))).collect();
        append_journal(&mut j, &mut notes, 4);
        assert!(notes.is_empty());
        assert_eq!(text(&j), ["3:c2", "3:c3", "3:c4", "3:c5"]);
        // Nothing to add changes nothing.
        append_journal(&mut j, &mut Vec::new(), 4);
        assert_eq!(j.len(), 4);
    }

    /// Needs a Vulkan 1.3 GPU: `cargo test -p ps5gpu --release -- --ignored`.
    #[test]
    #[ignore]
    fn ticks_complete_and_callbacks_run_in_order() {
        let dev = GpuDevice::create(false).expect("device");
        let mut s = Scheduler::new(&dev).unwrap();
        let first = s.current_tick();
        let order = Arc::new(Mutex::new(Vec::new()));
        let o = order.clone();
        s.queue_priority(move || o.lock().unwrap().push(1));
        let o = order.clone();
        let ran_on = Arc::new(Mutex::new(String::new()));
        let r = ran_on.clone();
        s.queue_priority(move || {
            o.lock().unwrap().push(2);
            *r.lock().unwrap() = std::thread::current().name().unwrap_or("").to_string();
        });
        // Record a real command so the tick carries GPU work.
        let buf = crate::device::Buffer::new(&dev, 1 << 20, vk::BufferUsageFlags::TRANSFER_DST, crate::device::MemoryUse::Device).unwrap();
        let cb = s.cmd();
        // SAFETY: recording buffer, valid destination.
        unsafe { dev.device.cmd_fill_buffer(cb, buf.buffer, 0, vk::WHOLE_SIZE, 0x1234_5678) };
        s.retire(buf);
        let t = s.flush();
        assert_eq!(t, first);
        assert_eq!(s.current_tick(), first + 1);
        let owner_ran = Arc::new(AtomicUsize::new(0));
        let or = owner_ran.clone();
        s.queue_completion(move || {
            or.fetch_add(1, Ordering::SeqCst);
        });
        s.finish();
        assert!(s.is_complete(t));
        assert_eq!(*order.lock().unwrap(), vec![1, 2]);
        assert_eq!(ran_on.lock().unwrap().as_str(), "rps5e-gpu-priority");
        assert_eq!(owner_ran.load(Ordering::SeqCst), 1);
        assert_eq!(s.stats().retired_pending, 0);
        // Many empty submissions keep ticks monotonic.
        let mut last = 0;
        for _ in 0..100 {
            let t = s.flush();
            assert!(t > last);
            last = t;
        }
        s.wait(last);
        assert!(s.timeline.completed_tick() >= last);
    }

    #[test]
    #[ignore]
    fn staging_ring_reuses_completed_space() {
        let dev = GpuDevice::create(false).expect("device");
        let mut s = Scheduler::new(&dev).unwrap();
        let mut ring = StagingRing::new(&dev, 1 << 16, vk::BufferUsageFlags::TRANSFER_SRC, crate::device::MemoryUse::Upload).unwrap();
        let a = ring.alloc(&s, 40_000, 16).unwrap();
        assert_eq!(a, 0);
        // Does not fit next to the in-flight region within the same tick.
        assert!(ring.alloc(&s, 40_000, 16).is_none());
        s.flush_and_wait();
        let b = ring.alloc(&s, 40_000, 16).unwrap();
        assert_eq!(b, 0);
        s.flush_and_wait();
        // Three submitted regions; a wrapped request overlapping two of
        // them waits for both (it used to give up after the oldest).
        for k in 0..3u64 {
            let o = ring.alloc(&s, 20_000, 16).unwrap();
            assert_eq!(o, if k == 0 { 40_000 } else { (k - 1) * 20_000 });
            s.flush();
        }
        assert_eq!(ring.alloc(&s, 40_000, 16), Some(0));
    }
}
