// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceVideoOut: display ports, buffer registration, flips and vblanks.
//!
//! Each opened port keeps up to 4 buffer groups (one attribute each) and 16
//! buffer slots, flip/vblank/output-mode event registrations and an emulated
//! display clock (59.94 Hz by default, `RPS5E_GUEST_REFRESH_RATE=<Hz>`
//! overrides it). Display buffers are shown through [`ps5gpu::presenter`].
//!
//! # Flip interface (used by the GPU command processor)
//!
//! * [`gpu_flip`] — a PM4 Flip packet executed on the GPU command thread.
//!   Reserves a flip slot, hands registered buffer `buffer_index` (or a blank
//!   flip for `-1`) to the presenter, scheduled at the port's next flip
//!   boundary, and completes the flip right away: FlipCount/FlipArg/
//!   CurrentBuffer/process-time counters are published and flip events
//!   (equeue ident 6, filter -13, hint `flip_arg << 16`) are triggered.
//!   The buffer is looked up when the GPU reaches the flip; an index with no
//!   registered buffer at that point shows a black frame and completes.
//!   Flips on an overlay-bus port complete without being shown.
//!   Returns the VideoOut error code on an invalid handle or index.
//! * [`is_flip_done`] — WaitFlipDone: true once no flip of `buffer_index` on
//!   `handle` is still waiting for the presenter (presented or discarded).
//! * [`reserve_eop_flip`] / [`claim_eop_flip`] / [`eop_flip`] —
//!   `sceVideoOutSubmitEopFlip` reserves a flip (pending from then on) and
//!   hands out a single-use token; the graphics-queue RELEASE_MEM carrying
//!   that token in its context id claims it and performs the flip as a GPU
//!   flip once the work before it is done. Closing the port cancels the
//!   reservation and its token.
//! * Buffer labels (`sceVideoOutGetBufferLabelAddress`): one u64 per
//!   display buffer, zeroed when the buffer is registered and when the
//!   buffer shown after it has been presented.
//! * [`cpu_submit_flip`] — `sceVideoOutSubmitFlip`: paces the calling guest
//!   thread to the display clock, presents and completes the flip.
//! * [`set_gpu_drain_hook`] — optional: a function that waits until queued
//!   GPU command streams have executed; Close/UnregisterBuffers run it first.

mod clock;
mod exports;

pub use clock::{FLIP_MODE_ASAP, FLIP_MODE_VSYNC, FLIP_MODE_VSYNC_MULTIPLE, FLIP_MODE_WINDOW};
pub use exports::register;

use crate::sync::equeue;
use clock::DisplayClock;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Condvar, LazyLock, Mutex, MutexGuard, OnceLock, RwLock};
use std::time::{Duration, Instant};

const fn err(code: u32) -> i64 {
    code as i32 as i64
}

pub const ERROR_INVALID_VALUE: i64 = err(0x8029_0001);
pub const ERROR_INVALID_ADDRESS: i64 = err(0x8029_0002);
pub const ERROR_INVALID_FLIP_MODE: i64 = err(0x8029_0006);
pub const ERROR_RESOURCE_BUSY: i64 = err(0x8029_0009);
pub const ERROR_INVALID_INDEX: i64 = err(0x8029_000A);
pub const ERROR_INVALID_HANDLE: i64 = err(0x8029_000B);
pub const ERROR_INVALID_EVENT_QUEUE: i64 = err(0x8029_000C);
pub const ERROR_INVALID_EVENT: i64 = err(0x8029_000D);
pub const ERROR_FLIP_QUEUE_FULL: i64 = err(0x8029_0012);
pub const ERROR_UNSUPPORTED_OUTPUT_MODE: i64 = err(0x8029_0016);
pub const ERROR_UNAVAILABLE_OUTPUT_MODE: i64 = err(0x8029_0019);
pub const ERROR_INVALID_OPTION: i64 = err(0x8029_001A);
pub const ERROR_UNSUPPORTED_OPERATION: i64 = err(0x8029_001C);
/// A value that is not an output mode (the console's answer, name ours).
pub const ERROR_UNKNOWN_OUTPUT_MODE: i64 = err(0x8029_001E);
/// Kernel EFAULT / ETIMEDOUT.
pub const ERROR_MEMORY_FAULT: i64 = err(0x8002_000E);
pub const ERROR_TIMED_OUT: i64 = err(0x8002_003C);

const MAX_OPEN_PORTS: usize = 4;
pub const MAX_DISPLAY_BUFFERS: usize = 16;
pub const MAX_DISPLAY_BUFFER_GROUPS: usize = 4;
const MAX_LATENCY_HISTORY: usize = 256;
/// Flips that may wait for the presenter before an EOP flip reservation
/// is refused (FLIP_QUEUE_FULL).
const FLIP_QUEUE_CAPACITY: usize = 16;
/// EOP flip tokens: never repeated within a process, and inside the 27-bit
/// context id of RELEASE_MEM.
const EOP_TOKEN_BASE: u32 = 0x0400_0000;
const EOP_TOKEN_LIMIT: u32 = 0x07FF_FFFF;
const OUTPUT_MODE_DEFAULT: u64 = 1;

const EVENT_FLIP: u64 = equeue::VIDEO_OUT_EVENT_FLIP;
const EVENT_VBLANK: u64 = equeue::VIDEO_OUT_EVENT_VBLANK;
const EVENT_OUTPUT_MODE: u64 = equeue::VIDEO_OUT_EVENT_OUTPUT_MODE;
const EVENT_PRE_VBLANK_START: u64 = equeue::VIDEO_OUT_EVENT_PRE_VBLANK_START;
const EVENT_VRR_STATUS: u64 = equeue::VIDEO_OUT_EVENT_VRR_STATUS;
/// Output buses of sceVideoOutOpen.
pub const BUS_TYPE_MAIN: i32 = 0;
pub const BUS_TYPE_OVERLAY: i32 = 1;
pub const BUS_TYPE_SUB: i32 = 2;
/// Flip modes of ports joined by sceVideoOutSetFlipMaster.
const FLIP_MODE_SLAVE: i32 = 8;
const FLIP_MODE_MASTER: i32 = 9;
const FILTER_VIDEO_OUT: i16 = equeue::FILTER_VIDEO_OUT;

fn guest_refresh_rate() -> u32 {
    static RATE: OnceLock<u32> = OnceLock::new();
    *RATE.get_or_init(|| {
        std::env::var("RPS5E_GUEST_REFRESH_RATE")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|r| (30..=1000).contains(r))
            .unwrap_or(60)
    })
}

fn flip_pacing_disabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("RPS5E_NO_FLIP_PACING").is_ok_and(|v| v == "1"))
}

fn epoch() -> Instant {
    static E: OnceLock<Instant> = OnceLock::new();
    *E.get_or_init(Instant::now)
}

/// Now on the video-out timeline (ns).
pub(crate) fn now_ts() -> i64 {
    epoch().elapsed().as_nanos() as i64
}

fn ts_to_instant(ts: i64) -> Instant {
    epoch() + Duration::from_nanos(ts.max(0) as u64)
}

fn sleep_until(ts: i64) {
    let now = now_ts();
    if ts > now {
        crate::signals::sleep(Duration::from_nanos((ts - now) as u64));
    }
}

/// A buffer attribute as registered (v1 or v2 layout).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferAttribute {
    pub pixel_format: u64,
    pub tiling_mode: u32,
    pub aspect_ratio: u32,
    pub width: u32,
    pub height: u32,
    pub pitch_in_pixel: u32,
    pub option: u64,
}

/// A registered display buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisplayBufferInfo {
    pub address: u64,
    pub attribute: BufferAttribute,
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    group: i32,
    address_left: u64,
    /// Right-eye buffer (stereo); never registered by the current exports.
    #[allow(dead_code)]
    address_right: u64,
}

const EMPTY_SLOT: Slot = Slot { group: -1, address_left: 0, address_right: 0 };

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EventReg {
    equeue: u64,
    udata: u64,
}

struct Port {
    #[allow(dead_code)]
    handle: i32,
    /// Output bus the port was opened on. Overlay flips complete and signal
    /// their events but are not shown: the overlay is not composited over
    /// the main output, and showing it as a whole frame would replace it.
    bus_type: i32,
    flip_rate: i32,
    clock: DisplayClock,
    published_vblank_count: u64,
    flip_count: u64,
    flip_process_time: u64,
    flip_process_time_counter: u64,
    submit_process_time_counter: u64,
    flip_arg: i64,
    gpu_queue_count: i32,
    flip_pending_count: i32,
    current_buffer: i32,
    output_width: u32,
    output_height: u32,
    refresh_rate: u32,
    output_mode: u64,
    gamma: f32,
    latency_start_points: HashMap<i64, i64>,
    latency_start_order: VecDeque<(i64, i64)>,
    completed_latency: HashMap<i64, i64>,
    completed_latency_order: VecDeque<(i64, i64)>,
    last_latency_first_section_usec: i64,
    groups: [Option<BufferAttribute>; MAX_DISPLAY_BUFFER_GROUPS],
    slots: [Slot; MAX_DISPLAY_BUFFERS],
    flip_events: Vec<EventReg>,
    vblank_events: Vec<EventReg>,
    pre_vblank_events: Vec<EventReg>,
    output_mode_events: Vec<EventReg>,
    /// VRR status registrations: never triggered, the host output's VRR
    /// state does not change.
    vrr_status_events: Vec<EventReg>,
    /// Port whose flips this port follows (sceVideoOutSetFlipMaster).
    master: Option<i32>,
    /// Number of ports following this one.
    slave_count: u32,
    open_ts: i64,
    last_cpu_flip_ts: i64,
    window_top: i32,
    window_bottom: i32,
    last_presentation_ts: i64,
    /// When the latest GPU flip was scheduled to be shown.
    last_scheduled_ts: i64,
    /// Guest array of MAX_DISPLAY_BUFFERS u64 buffer labels (0 until
    /// sceVideoOutGetBufferLabelAddress first asks for it).
    labels: u64,
    /// Buffer of the latest flip presented (or discarded), -1 for none:
    /// its label is released once the next flip has been presented.
    shown_buffer: i32,
    /// sceVideoOutAdjustColorSpaceConversion settings and the global
    /// blend space (accepted; the presenter does not apply them).
    color_space_conversion: [u8; COLOR_SPACE_CONVERSION_SIZE],
    blend_space: (u32, u32),
}

/// Bytes of SceVideoOutColorSpaceConversionSettings.
pub(crate) const COLOR_SPACE_CONVERSION_SIZE: usize = 32;

impl Port {
    fn new(handle: i32, bus_type: i32) -> Self {
        let open_ts = now_ts();
        Self {
            handle,
            bus_type,
            flip_rate: 0,
            clock: DisplayClock::new(open_ts, crate::time::process_time_counter(), crate::time::tsc_now(), crate::time::tsc_hz()),
            published_vblank_count: 0,
            flip_count: 0,
            flip_process_time: 0,
            flip_process_time_counter: 0,
            submit_process_time_counter: 0,
            flip_arg: -1,
            gpu_queue_count: 0,
            flip_pending_count: 0,
            current_buffer: -1,
            output_width: 1920,
            output_height: 1080,
            refresh_rate: guest_refresh_rate(),
            output_mode: OUTPUT_MODE_DEFAULT,
            gamma: 1.0,
            latency_start_points: HashMap::new(),
            latency_start_order: VecDeque::new(),
            completed_latency: HashMap::new(),
            completed_latency_order: VecDeque::new(),
            last_latency_first_section_usec: 0,
            groups: [None; MAX_DISPLAY_BUFFER_GROUPS],
            slots: [EMPTY_SLOT; MAX_DISPLAY_BUFFERS],
            flip_events: Vec::new(),
            vblank_events: Vec::new(),
            pre_vblank_events: Vec::new(),
            output_mode_events: Vec::new(),
            vrr_status_events: Vec::new(),
            master: None,
            slave_count: 0,
            open_ts,
            last_cpu_flip_ts: -1,
            window_top: 0,
            window_bottom: i32::MAX,
            last_presentation_ts: -1,
            last_scheduled_ts: -1,
            labels: 0,
            shown_buffer: -1,
            color_space_conversion: [0; COLOR_SPACE_CONVERSION_SIZE],
            blend_space: (0, 0),
        }
    }

    /// Release (zero) the label of display buffer `index`.
    fn release_label(&self, index: i32) {
        if self.labels != 0 && (0..MAX_DISPLAY_BUFFERS as i32).contains(&index) && !crate::agc::mem::w64(self.labels + index as u64 * 8, 0) {
            log::warn!("videoout: buffer label 0x{:x} of buffer {index} not writable", self.labels);
        }
    }

    fn display_buffer(&self, index: i32) -> Option<DisplayBufferInfo> {
        let slot = self.slots.get(usize::try_from(index).ok()?)?;
        if slot.address_left == 0 || slot.group < 0 {
            return None;
        }
        let attribute = (*self.groups.get(slot.group as usize)?)?;
        Some(DisplayBufferInfo { address: slot.address_left, attribute })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlipState {
    Reserved,
    Completed,
}

/// What became of a flip on the presenter side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlipOutcome {
    Pending,
    Presented,
    Discarded,
    Cancelled,
}

struct FlipRequest {
    handle: i32,
    buffer_index: i32,
    flip_mode: i32,
    flip_arg: i64,
    gpu_queued: bool,
    ready_ts: Option<i64>,
    event_hint: u64,
    flip_events: Vec<EventReg>,
    state: FlipState,
    outcome: FlipOutcome,
    ticket: Option<ps5gpu::PresentTicket>,
}

struct Global {
    ports: HashMap<i32, Port>,
    next_handle: i32,
    flips: BTreeMap<u64, FlipRequest>,
    next_flip_id: u64,
    discarded_flips: u64,
    hdr_requested: bool,
    vblank_stop: bool,
    /// EOP flip token -> reserved flip request.
    eop_tokens: HashMap<u32, u64>,
    next_eop_token: u32,
}

static STATE: LazyLock<Mutex<Global>> = LazyLock::new(|| {
    Mutex::new(Global {
        ports: HashMap::new(),
        next_handle: 1,
        flips: BTreeMap::new(),
        next_flip_id: 0,
        discarded_flips: 0,
        hdr_requested: false,
        vblank_stop: false,
        eop_tokens: HashMap::new(),
        next_eop_token: EOP_TOKEN_BASE,
    })
});
static CV: Condvar = Condvar::new();

fn lock() -> MutexGuard<'static, Global> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

static GPU_DRAIN_HOOK: RwLock<Option<fn()>> = RwLock::new(None);

/// Install a function that blocks until all GPU command streams submitted
/// so far have executed. Close and UnregisterBuffers run it before
/// tearing state down, so in-flight flips of those buffers finish first.
pub fn set_gpu_drain_hook(hook: fn()) {
    *GPU_DRAIN_HOOK.write().unwrap_or_else(|e| e.into_inner()) = Some(hook);
}

fn drain_gpu() {
    let hook = *GPU_DRAIN_HOOK.read().unwrap_or_else(|e| e.into_inner());
    if let Some(h) = hook {
        h();
    }
}

fn prune_history(map: &mut HashMap<i64, i64>, order: &mut VecDeque<(i64, i64)>) {
    while order.len() > MAX_LATENCY_HISTORY {
        let (arg, ts) = order.pop_front().unwrap();
        if map.get(&arg) == Some(&ts) {
            map.remove(&arg);
        }
    }
}

/// Registered display buffer `buffer_index` of `handle`.
pub fn display_buffer_info(handle: i32, buffer_index: i32) -> Option<DisplayBufferInfo> {
    lock().ports.get(&handle)?.display_buffer(buffer_index)
}

/// True when the latest flipped buffer uses an HDR (PQ) format.
pub fn is_hdr_output_requested() -> bool {
    lock().hdr_requested
}

fn surface_desc(info: &DisplayBufferInfo) -> ps5gpu::SurfaceDesc {
    let a = &info.attribute;
    ps5gpu::SurfaceDesc {
        width: a.width,
        height: a.height,
        pitch_pixels: if a.pitch_in_pixel == 0 { a.width } else { a.pitch_in_pixel },
        pixel_format: a.pixel_format,
        tiling: a.tiling_mode,
    }
}

// ---------------------------------------------------------------- flips --

/// Validate a flip and take its counters; returns the request id.
fn reserve_flip(handle: i32, buffer_index: i32, flip_mode: i32, flip_arg: i64, gpu_queued: bool) -> Result<u64, i64> {
    if buffer_index < -1 || buffer_index >= MAX_DISPLAY_BUFFERS as i32 {
        return Err(ERROR_INVALID_INDEX);
    }
    let mut g = lock();
    refresh_outcomes_locked(&mut g);
    let g = &mut *g;
    let Some(port) = g.ports.get_mut(&handle) else {
        return Err(ERROR_INVALID_HANDLE);
    };
    if port.master.is_some() != (flip_mode == FLIP_MODE_SLAVE) || (port.slave_count != 0) != (flip_mode == FLIP_MODE_MASTER) {
        return Err(ERROR_INVALID_FLIP_MODE);
    }
    // A command-buffer flip names its buffer when the GPU reaches it; a
    // buffer that is still unregistered then is shown black (see
    // `gpu_flip`). Only the CPU path rejects it up front.
    if !gpu_queued && buffer_index != -1 && port.slots[buffer_index as usize].group < 0 {
        return Err(ERROR_INVALID_INDEX);
    }
    let submitted_at = now_ts();
    if let Some(started) = port.latency_start_points.remove(&flip_arg) {
        port.last_latency_first_section_usec = ((submitted_at - started) / 1000).max(0);
    }
    port.flip_pending_count += 1;
    if gpu_queued {
        port.gpu_queue_count += 1;
    }
    port.submit_process_time_counter = crate::time::process_time_counter();
    g.next_flip_id += 1;
    let id = g.next_flip_id;
    // Overlay buffers are not shown, so they do not select the output format.
    let hdr = (buffer_index >= 0 && port.bus_type != BUS_TYPE_OVERLAY)
        .then(|| port.display_buffer(buffer_index))
        .flatten()
        .map(|b| ps5gpu::format::is_hdr(b.attribute.pixel_format));
    let request = FlipRequest {
        handle,
        buffer_index,
        flip_mode,
        flip_arg,
        gpu_queued,
        ready_ts: None,
        event_hint: EVENT_FLIP | ((flip_arg as u64 & 0x0000_FFFF_FFFF_FFFF) << 16),
        flip_events: port.flip_events.clone(),
        state: FlipState::Reserved,
        outcome: FlipOutcome::Pending,
        ticket: None,
    };
    g.flips.insert(id, request);
    if let Some(h) = hdr {
        g.hdr_requested = h;
    }
    Ok(id)
}

fn remove_if_finished(g: &mut Global, id: u64) {
    if let Some(r) = g.flips.get(&id) {
        if r.state == FlipState::Completed && r.outcome != FlipOutcome::Pending {
            g.flips.remove(&id);
        }
    }
}

/// Publish completion counters and events unless the request was cancelled.
fn complete_flip(id: u64) {
    let (events, hint) = {
        let mut g = lock();
        let g = &mut *g;
        let Some(r) = g.flips.get_mut(&id) else { return };
        if r.state != FlipState::Reserved {
            return;
        }
        r.state = FlipState::Completed;
        let completed_at = now_ts();
        if let Some(port) = g.ports.get_mut(&r.handle) {
            port.flip_count += 1;
            port.flip_process_time = crate::time::process_time_usec();
            port.flip_process_time_counter = crate::time::process_time_counter();
            port.flip_arg = r.flip_arg;
            port.current_buffer = r.buffer_index;
            port.flip_pending_count = (port.flip_pending_count - 1).max(0);
            if r.gpu_queued {
                port.gpu_queue_count = (port.gpu_queue_count - 1).max(0);
            }
            r.ready_ts = Some(completed_at);
            port.completed_latency.insert(r.flip_arg, completed_at);
            port.completed_latency_order.push_back((r.flip_arg, completed_at));
            prune_history(&mut port.completed_latency, &mut port.completed_latency_order);
        }
        let events = std::mem::take(&mut r.flip_events);
        let hint = r.event_hint;
        remove_if_finished(g, id);
        CV.notify_all();
        (events, hint)
    };
    for e in events {
        equeue::trigger_display_event(e.equeue, EVENT_FLIP, FILTER_VIDEO_OUT, hint, e.udata);
    }
}

fn set_flip_outcome_locked(g: &mut Global, id: u64, outcome: FlipOutcome) -> bool {
    let Some(r) = g.flips.get_mut(&id) else { return false };
    if r.outcome != FlipOutcome::Pending {
        return false;
    }
    r.outcome = outcome;
    let (handle, buffer_index) = (r.handle, r.buffer_index);
    if matches!(outcome, FlipOutcome::Presented | FlipOutcome::Discarded) {
        if let Some(port) = g.ports.get_mut(&handle) {
            let previous = std::mem::replace(&mut port.shown_buffer, buffer_index);
            port.release_label(previous);
        }
    }
    if outcome == FlipOutcome::Presented && buffer_index >= 0 {
        // Overlay-bus flips complete without reaching the screen.
        if let Some(port) = g.ports.get_mut(&handle).filter(|p| p.bus_type != BUS_TYPE_OVERLAY) {
            port.last_presentation_ts = now_ts();
        }
    }
    if outcome == FlipOutcome::Discarded {
        g.discarded_flips += 1;
    }
    remove_if_finished(g, id);
    CV.notify_all();
    true
}

/// The presenter showed (or will never show) the frame.
fn mark_flip_presented(id: u64) {
    set_flip_outcome_locked(&mut lock(), id, FlipOutcome::Presented);
}

/// Pull presenter results into pending requests.
fn refresh_outcomes_locked(g: &mut Global) {
    let p = ps5gpu::presenter();
    let done: Vec<(u64, FlipOutcome)> = g
        .flips
        .iter()
        .filter(|(_, r)| r.outcome == FlipOutcome::Pending)
        .filter_map(|(&id, r)| {
            let t = r.ticket?;
            match p.outcome(t) {
                ps5gpu::PresentOutcome::Pending => None,
                ps5gpu::PresentOutcome::Presented => Some((id, FlipOutcome::Presented)),
                ps5gpu::PresentOutcome::Discarded => Some((id, FlipOutcome::Discarded)),
            }
        })
        .collect();
    for (id, o) in done {
        set_flip_outcome_locked(g, id, o);
    }
}

fn cancel_flip_locked(g: &mut Global, id: u64) {
    g.eop_tokens.retain(|_, v| *v != id);
    let Some(r) = g.flips.remove(&id) else { return };
    if r.state == FlipState::Reserved {
        if let Some(port) = g.ports.get_mut(&r.handle) {
            port.flip_pending_count = (port.flip_pending_count - 1).max(0);
            if r.gpu_queued {
                port.gpu_queue_count = (port.gpu_queue_count - 1).max(0);
            }
        }
    }
}

fn cancel_flip(id: u64) {
    let mut g = lock();
    cancel_flip_locked(&mut g, id);
    CV.notify_all();
}

/// Outcome of a flip request (None once it has been retired).
pub fn flip_outcome(id: u64) -> Option<FlipOutcome> {
    let mut g = lock();
    refresh_outcomes_locked(&mut g);
    g.flips.get(&id).map(|r| r.outcome)
}

/// Number of flip requests still tracked (diagnostics).
pub fn pending_flip_request_count() -> usize {
    let mut g = lock();
    refresh_outcomes_locked(&mut g);
    g.flips.len()
}

/// WaitFlipDone: true when no flip of `buffer_index` on `handle` still
/// waits for the presenter (a presented, discarded or cancelled flip no
/// longer does).
pub fn is_flip_done(handle: i32, buffer_index: i32) -> bool {
    let mut g = lock();
    refresh_outcomes_locked(&mut g);
    !g.flips
        .values()
        .any(|r| r.handle == handle && r.buffer_index == buffer_index && r.outcome == FlipOutcome::Pending)
}

/// What a GPU flip shows.
enum GpuFlipImage {
    Buffer(DisplayBufferInfo),
    /// The flipped index had no registered buffer when the GPU reached the
    /// flip: a black frame of the port's size.
    Black(u32, u32),
}

fn black_frame(width: u32, height: u32) -> Vec<u8> {
    [0u8, 0, 0, 0xFF].repeat(width as usize * height as usize)
}

fn report_unregistered_flip(handle: i32, buffer_index: i32) {
    static REPORTED: Mutex<u16> = Mutex::new(0);
    let bit = 1u16 << (buffer_index as u32 & 15);
    let mut r = REPORTED.lock().unwrap_or_else(|e| e.into_inner());
    if *r & bit == 0 {
        *r |= bit;
        log::warn!("videoout: GPU flip of unregistered buffer {buffer_index} on handle {handle} presents black");
    }
}

/// GPU-path flip (PM4 Flip packet). See the module docs.
pub fn gpu_flip(handle: i32, buffer_index: i32, flip_mode: u32, flip_arg: i64) -> Result<(), i64> {
    let id = reserve_flip(handle, buffer_index, flip_mode as i32, flip_arg, true)?;
    present_reserved_flip(id)
}

/// `sceVideoOutSubmitEopFlip`: validate and reserve a flip (pending from
/// now on) and return the single-use token an end-of-pipe RELEASE_MEM
/// names to perform it.
pub fn reserve_eop_flip(handle: i32, buffer_index: i32, flip_mode: i32, flip_arg: i64) -> Result<u32, i64> {
    {
        let mut g = lock();
        refresh_outcomes_locked(&mut g);
        let Some(port) = g.ports.get(&handle) else { return Err(ERROR_INVALID_HANDLE) };
        if !(FLIP_MODE_VSYNC..=6).contains(&flip_mode) {
            return Err(ERROR_INVALID_VALUE);
        }
        if !(-1..MAX_DISPLAY_BUFFERS as i32).contains(&buffer_index) || (buffer_index >= 0 && port.slots[buffer_index as usize].group < 0) {
            return Err(ERROR_INVALID_INDEX);
        }
        if g.flips.values().filter(|r| r.outcome == FlipOutcome::Pending).count() >= FLIP_QUEUE_CAPACITY {
            return Err(ERROR_FLIP_QUEUE_FULL);
        }
    }
    let id = reserve_flip(handle, buffer_index, flip_mode, flip_arg, true)?;
    let mut g = lock();
    let token = g.next_eop_token;
    if token > EOP_TOKEN_LIMIT {
        cancel_flip_locked(&mut g, id);
        log::error!("videoout: EOP flip token space exhausted");
        return Err(ERROR_FLIP_QUEUE_FULL);
    }
    g.next_eop_token += 1;
    g.eop_tokens.insert(token, id);
    Ok(token)
}

/// The reserved flip an EOP release names by `token` (single use); None
/// for a token never handed out, already used or cancelled.
pub fn claim_eop_flip(token: u32) -> Option<u64> {
    let mut g = lock();
    let id = g.eop_tokens.remove(&token)?;
    g.flips.contains_key(&id).then_some(id)
}

/// Perform a claimed EOP flip once the GPU work before its release is
/// done: as a GPU flip of the reserved buffer.
pub fn eop_flip(id: u64) -> Result<(), i64> {
    present_reserved_flip(id)
}

/// Schedule, present and complete the reserved GPU flip `id`.
fn present_reserved_flip(id: u64) -> Result<(), i64> {
    let Some((handle, buffer_index, flip_mode, flip_arg)) = lock().flips.get(&id).map(|r| (r.handle, r.buffer_index, r.flip_mode, r.flip_arg)) else {
        // Cancelled (the port was closed) before the GPU reached it.
        return Err(ERROR_INVALID_HANDLE);
    };

    // Schedule the presentation at the port's next flip boundary.
    let (target, show) = {
        let mut g = lock();
        let Some(port) = g.ports.get_mut(&handle) else {
            drop(g);
            cancel_flip(id);
            return Err(ERROR_INVALID_HANDLE);
        };
        if port.bus_type == BUS_TYPE_OVERLAY {
            drop(g);
            complete_flip(id);
            mark_flip_presented(id);
            log::trace!("videoout: overlay gpu flip handle={handle} index={buffer_index} arg={flip_arg} (not shown)");
            return Ok(());
        }
        let now = now_ts();
        let target = if flip_pacing_disabled() {
            now
        } else {
            clock::next_flip_timestamp(
                port.open_ts,
                port.last_scheduled_ts,
                now,
                port.refresh_rate,
                port.flip_rate,
                flip_mode,
                port.output_height,
                port.window_top,
                port.window_bottom,
                now,
            )
        };
        if buffer_index >= 0 {
            port.last_scheduled_ts = target.max(now);
        }
        let show = match port.display_buffer(buffer_index) {
            Some(b) => Some(GpuFlipImage::Buffer(b)),
            None if buffer_index >= 0 => Some(GpuFlipImage::Black(port.output_width, port.output_height)),
            None => None,
        };
        (target, show)
    };

    let ticket = show.map(|s| {
        let nb = (target > now_ts()).then(|| ts_to_instant(target));
        match s {
            GpuFlipImage::Buffer(b) => ps5gpu::presenter().present_guest_surface_at(surface_desc(&b), b.address, nb),
            GpuFlipImage::Black(w, h) => {
                report_unregistered_flip(handle, buffer_index);
                ps5gpu::presenter().present_bgra8(w, h, black_frame(w, h), nb)
            }
        }
    });
    if let Some(t) = ticket {
        if let Some(r) = lock().flips.get_mut(&id) {
            r.ticket = Some(t);
        }
    }
    complete_flip(id);
    if ticket.is_none() {
        // Blank flip (index -1) or unregistered buffer: nothing to show.
        mark_flip_presented(id);
    }
    log::trace!("videoout: gpu flip handle={handle} index={buffer_index} mode={flip_mode} arg={flip_arg}");
    Ok(())
}

/// CPU-path flip (`sceVideoOutSubmitFlip`): paced on the calling thread,
/// completed at submit.
pub fn cpu_submit_flip(handle: i32, buffer_index: i32, flip_mode: i32, flip_arg: i64) -> Result<(), i64> {
    let id = reserve_flip(handle, buffer_index, flip_mode, flip_arg, false)?;
    if !pace_flip(handle, flip_mode) {
        cancel_flip(id);
        return Err(ERROR_INVALID_HANDLE);
    }
    let overlay = lock().ports.get(&handle).is_some_and(|p| p.bus_type == BUS_TYPE_OVERLAY);
    let ticket = if buffer_index >= 0 && !overlay {
        display_buffer_info(handle, buffer_index)
            .map(|b| ps5gpu::presenter().present_guest_surface(surface_desc(&b), b.address))
    } else {
        None
    };
    if let Some(t) = ticket {
        if let Some(r) = lock().flips.get_mut(&id) {
            r.ticket = Some(t);
        }
    }
    complete_flip(id);
    if ticket.is_none() {
        mark_flip_presented(id);
    }
    log::trace!("videoout: submit flip handle={handle} index={buffer_index} mode={flip_mode} arg={flip_arg}");
    Ok(())
}

/// Sleep the guest thread until the display clock allows a CPU flip.
fn pace_flip(handle: i32, flip_mode: i32) -> bool {
    let mut ready_at: Option<i64> = None;
    loop {
        let target = {
            let mut g = lock();
            let Some(port) = g.ports.get_mut(&handle) else { return false };
            let now = now_ts();
            let ready = *ready_at.get_or_insert(now);
            let target = clock::next_flip_timestamp(
                port.open_ts,
                port.last_cpu_flip_ts,
                now,
                port.refresh_rate,
                port.flip_rate,
                flip_mode,
                port.output_height,
                port.window_top,
                port.window_bottom,
                ready,
            );
            if flip_pacing_disabled() || target <= now {
                port.last_cpu_flip_ts = now;
                return true;
            }
            target
        };
        sleep_until(target);
    }
}

// --------------------------------------------------------------- vblank --

fn start_vblank_thread_once() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        let _ = std::thread::Builder::new().name("rps5e-vblank".into()).spawn(vblank_loop);
    });
}

fn vblank_loop() {
    // (equeue, ident, hint, udata); pre-vblank-start events precede the
    // vblank events of the same count.
    let mut pending: Vec<(u64, u64, u64, u64)> = Vec::new();
    loop {
        pending.clear();
        let next = {
            let mut g = lock();
            if g.vblank_stop {
                return;
            }
            let now = now_ts();
            let mut next = i64::MAX;
            for port in g.ports.values_mut() {
                if port.vblank_events.is_empty() && port.pre_vblank_events.is_empty() {
                    continue;
                }
                port.clock.advance(now, port.refresh_rate);
                next = next.min(port.clock.next_timestamp(port.refresh_rate));
                if port.clock.count <= port.published_vblank_count {
                    continue;
                }
                port.published_vblank_count = port.clock.count;
                let hint = (port.clock.count & 0x0000_FFFF_FFFF_FFFF) << 16;
                for e in &port.pre_vblank_events {
                    pending.push((e.equeue, EVENT_PRE_VBLANK_START, hint, e.udata));
                }
                for e in &port.vblank_events {
                    pending.push((e.equeue, EVENT_VBLANK, hint, e.udata));
                }
            }
            if next == i64::MAX {
                let _g = CV.wait(g).unwrap_or_else(|e| e.into_inner());
                continue;
            }
            next
        };
        for &(eq, ident, hint, udata) in &pending {
            equeue::trigger_display_event(eq, ident, FILTER_VIDEO_OUT, hint, udata);
        }
        sleep_until(next);
    }
}

#[cfg(test)]
mod tests;
