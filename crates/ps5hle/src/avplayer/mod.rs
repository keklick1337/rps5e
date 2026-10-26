// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceAvPlayer: movie playback (MP4 / WebM through the host decoder).
//!
//! A player is created by `sceAvPlayerInit(Ex)` with the title's memory
//! callbacks and optional file and event callbacks. `AddSource` opens the
//! movie (from the host file system, or through the file callbacks when the
//! title supplies them). `Start` opens the decoders, allocates the output
//! buffers through the title's allocator and starts the pipeline
//! (`source.rs`): a demuxer and one decoder per enabled stream, decoding
//! straight into those buffers.
//!
//! `GetVideoData(Ex)` hands out a picture once the presentation clock
//! reaches it (immediately with A/V sync off); `GetAudioData` hands out the
//! next PCM chunk and re-anchors the clock to it. Pictures stay untouched
//! until `num_output_video_framebuffers - 1` newer pictures were taken.
//!
//! Pictures are NV12: luma pitch = width (aligned to 16 unless the title
//! asked for the Software2 decoder) aligned to 256, chroma right after the
//! luma rows. Audio is interleaved s16.
//!
//! Guest code (allocators, file callbacks, the event callback) is only
//! called on the guest thread inside an export and never while a player
//! lock is held, because callbacks re-enter the library.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
pub mod host;
pub mod source;
pub mod stream;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};
use host::{Media, MediaType, StreamInfo};
use source::{FrameKind, GuestBuf, Pipeline, ReadyFrame, Settings, VideoSurface};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use stream::{FileStreamer, GuestFile};

const LIB: &str = "libSceAvPlayer";

const OK: HleResult = 0;
const ERROR_INVALID_PARAMS: u32 = 0x806A_0001;
const ERROR_OPERATION_FAILED: u32 = 0x806A_0002;
const ERROR_NO_MEMORY: u32 = 0x806A_0003;
const ERROR_NOT_SUPPORTED: u32 = 0x806A_0004;
const WARNING_LOOPING_BACK: u32 = 0x806A_00A1;
const WARNING_JUMP_COMPLETE: u32 = 0x806A_00A3;

const EVENT_STATE_STOP: i32 = 0x01;
const EVENT_STATE_READY: i32 = 0x02;
const EVENT_STATE_PLAY: i32 = 0x03;
const EVENT_STATE_PAUSE: i32 = 0x04;
const EVENT_WARNING_ID: i32 = 0x20;

const STREAM_TYPE_UNKNOWN: u32 = 0;
const STREAM_TYPE_VIDEO: u32 = 1;
const STREAM_TYPE_AUDIO: u32 = 2;
const STREAM_TYPE_TIMED_TEXT: u32 = 3;

const SOURCE_TYPE_UNKNOWN: u32 = 0;
const SOURCE_TYPE_FILE_MP4: u32 = 1;
const SOURCE_TYPE_FILE_WEBM: u32 = 2;
const SOURCE_TYPE_HLS: u32 = 8;

const VIDEO_DECODER_SOFTWARE2: u32 = 1;
const SYNC_MODE_NONE: u32 = 1;
const TRICK_SPEED_NORMAL: i32 = 100;
const TRICK_SPEED_MIN: i32 = 400;
const TRICK_SPEED_MAX: i32 = 3200;

const BUFFER_ALIGN: u64 = 0x100;
const AUDIO_BUFFERS: usize = 8;
/// Extra video pictures allocated at start for decoding ahead.
const VIDEO_DECODE_AHEAD_FRAMES: usize = 4;
const STREAM_INFO_SIZE: usize = 32;
const STREAM_INFO_EX_SIZE: usize = 104;
const FRAME_INFO_SIZE: usize = 40;
const FRAME_INFO_EX_SIZE: usize = 104;
/// Event payload (the warning code) lives at this offset in the handle block.
const EVENT_DATA_OFFSET: u64 = 0x80;

fn err(code: u32) -> HleResult {
    code as i32 as HleResult
}

fn align_up(v: u32, a: u32) -> u32 {
    v.div_ceil(a) * a
}

// ---------------------------------------------------------------------------
// Guest calls (replaceable in tests)
// ---------------------------------------------------------------------------

pub(crate) type GuestCallFn = fn(u64, [u64; 6]) -> u64;

fn default_guest_call(func: u64, args: [u64; 6]) -> u64 {
    ps5cpu::exec::call_guest_nested(func, args)
}

static GUEST_CALL: RwLock<GuestCallFn> = RwLock::new(default_guest_call);

#[cfg(test)]
pub(crate) fn set_guest_call(f: GuestCallFn) {
    *GUEST_CALL.write().unwrap() = f;
}

fn guest_call(func: u64, args: [u64; 6]) -> u64 {
    let f = *GUEST_CALL.read().unwrap_or_else(|e| e.into_inner());
    f(func, args)
}

// ---------------------------------------------------------------------------
// HLE-owned guest memory (file callback scratch)
// ---------------------------------------------------------------------------

const ARENA_BASE: u64 = 0x6C00_0000_0000;
const ARENA_SIZE: u64 = 0x1_0000_0000;
const ARENA_GRANULE: u64 = 0x1_0000;

struct Arena {
    mapped: bool,
    next: u64,
    free: Vec<(u64, u64)>,
    live: std::collections::BTreeMap<u64, u64>,
}

static ARENA: Mutex<Arena> = Mutex::new(Arena { mapped: false, next: ARENA_BASE, free: Vec::new(), live: std::collections::BTreeMap::new() });

/// Allocate guest-visible memory owned by the HLE layer.
pub(crate) fn arena_alloc(size: u64) -> Option<u64> {
    let len = size.max(1).div_ceil(ARENA_GRANULE) * ARENA_GRANULE;
    let mut a = ARENA.lock().unwrap_or_else(|e| e.into_inner());
    if !a.mapped {
        let r = unsafe {
            libc::mmap(
                ARENA_BASE as *mut libc::c_void,
                ARENA_SIZE as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | libc::MAP_FIXED_NOREPLACE,
                -1,
                0,
            )
        };
        if r == libc::MAP_FAILED || r as u64 != ARENA_BASE {
            log::error!("[avplayer] cannot map the HLE buffer arena");
            return None;
        }
        ps5vm::ranges::track(ARENA_BASE, ARENA_SIZE, ps5vm::MemProtect::RW);
        a.mapped = true;
    }
    let addr = if let Some(i) = a.free.iter().position(|&(_, l)| l == len) {
        a.free.swap_remove(i).0
    } else if a.next + len <= ARENA_BASE + ARENA_SIZE {
        let p = a.next;
        a.next += len;
        p
    } else {
        log::error!("[avplayer] HLE buffer arena exhausted ({len:#x} bytes)");
        return None;
    };
    a.live.insert(addr, len);
    Some(addr)
}

pub(crate) fn arena_free(addr: u64) {
    let mut a = ARENA.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(len) = a.live.remove(&addr) {
        unsafe {
            libc::madvise(addr as *mut libc::c_void, len as usize, libc::MADV_DONTNEED);
        }
        a.free.push((addr, len));
    }
}

// ---------------------------------------------------------------------------
// Guest callback tables
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default, Debug)]
struct MemReplacement {
    object: u64,
    allocate: u64,
    deallocate: u64,
    allocate_texture: u64,
    deallocate_texture: u64,
}

impl MemReplacement {
    fn valid(&self) -> bool {
        self.allocate != 0 && self.deallocate != 0 && self.allocate_texture != 0 && self.deallocate_texture != 0
    }

    fn alloc(&self, size: u32, texture: bool) -> Option<GuestBuf> {
        let f = if texture { self.allocate_texture } else { self.allocate };
        let p = guest_call(f, [self.object, BUFFER_ALIGN, size as u64, 0, 0, 0]);
        if p == 0 {
            log::warn!("[avplayer] guest allocator returned null for {size:#x} bytes");
            return None;
        }
        Some(GuestBuf { addr: p, size })
    }

    fn free(&self, b: GuestBuf, texture: bool) {
        let f = if texture { self.deallocate_texture } else { self.deallocate };
        guest_call(f, [self.object, b.addr, 0, 0, 0, 0]);
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct FileReplacement {
    object: u64,
    open: u64,
    close: u64,
    read_offset: u64,
    size: u64,
}

#[derive(Clone, Copy, Default, Debug)]
struct EventReplacement {
    object: u64,
    callback: u64,
}

/// Events collected under the player lock and delivered after releasing it.
#[derive(Default)]
struct Events {
    list: Vec<(i32, Option<u32>)>,
}

impl Events {
    fn state(&mut self, id: i32) {
        self.list.push((id, None));
    }
    fn warning(&mut self, code: u32) {
        self.list.push((EVENT_WARNING_ID, Some(code)));
    }
}

fn deliver(ctx: &CpuContext, handle: u64, cb: EventReplacement, events: Events) {
    if cb.callback == 0 {
        return;
    }
    for (id, warning) in events.list {
        let data = match warning {
            Some(code) => {
                let addr = handle + EVENT_DATA_OFFSET;
                ctx.write_u32(addr, code);
                addr
            }
            None => 0,
        };
        log::debug!("[avplayer] event {id:#x} handle={handle:#x}");
        guest_call(cb.callback, [cb.object, id as u32 as u64, 0, data, 0, 0]);
    }
}

/// Output buffers to hand back to the title's allocator once unlocked.
#[derive(Default)]
struct FreeList {
    video: Vec<GuestBuf>,
    audio: Vec<GuestBuf>,
}

impl FreeList {
    fn release(self, mem: MemReplacement) {
        for b in self.video {
            mem.free(b, true);
        }
        for b in self.audio {
            mem.free(b, false);
        }
    }
}

// ---------------------------------------------------------------------------
// Player state
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Ready,
    Playing,
    Stopped,
}

struct Source {
    source_type: u32,
    /// The movie while no pipeline owns it.
    media: Option<Media>,
    pipeline: Option<Pipeline>,
    guest_file: Option<Arc<GuestFile>>,
    /// Every container stream, indexed by stream id.
    streams: Vec<StreamInfo>,
    container_ms: u64,
    vdec2: bool,
    max_video_buffers: usize,
    demux_bytes: u32,
    video_id: Option<usize>,
    audio_id: Option<usize>,
    state: State,
    paused: bool,
    seek_video_frame_pending: bool,
    looping: bool,
    trick_speed: i32,
    sync_mode: u32,
    /// The title pulls audio (since the last start): pictures the clock has
    /// passed are skipped even with A/V sync off.
    audio_driving: bool,
    start_time_ms: u64,
    clock_start: Instant,
    pause_time: Instant,
    paused_extra: Duration,
    last_output_loop_offset: u64,
    /// Loop offset of the pass pictures are currently paced in.
    pass_offset: u64,
    /// Shift (ms) applied to that pass so it continues the previous one.
    pass_shift: i64,
    /// Presentation time of the last picture handed out.
    last_due: Option<u64>,
    pending_loop_warnings: u32,
    /// The end of a non-looping movie was reported (STOP event).
    end_reported: bool,
    /// Output buffers not lent to the pipeline or the title.
    idle_video: Vec<GuestBuf>,
    idle_audio: Vec<GuestBuf>,
    /// The picture last handed to the title; its buffer goes back to the
    /// pool when the next picture is handed out.
    current_video: Option<ReadyFrame>,
    current_audio: Option<ReadyFrame>,
}

/// Reported picture (width, height) and NV12 pitch of a video stream: the
/// default decoder aligns the picture to 16, Software2 does not.
fn video_dims(vdec2: bool, s: &StreamInfo) -> (u32, u32, u32) {
    let (w, h) = if vdec2 { (s.width, s.height) } else { (align_up(s.width, 16), align_up(s.height, 16)) };
    (w, h, align_up(w, 256))
}

impl Source {

    fn stream_supported(&self, id: usize) -> bool {
        let Some(s) = self.streams.get(id) else { return false };
        if !matches!(s.media_type, MediaType::Video | MediaType::Audio) {
            return true;
        }
        if self.source_type == SOURCE_TYPE_FILE_WEBM && !s.webm_ok {
            log::info!("[avplayer] unsupported codec for a WebM source: stream {id}");
            return false;
        }
        if !s.decodable {
            log::info!("[avplayer] no host decoder for stream {id}");
            return false;
        }
        true
    }

    fn auto_enable(&mut self) {
        for i in 0..self.streams.len() {
            if !self.stream_supported(i) {
                continue;
            }
            match self.streams[i].media_type {
                MediaType::Video if self.video_id.is_none() => self.video_id = Some(i),
                MediaType::Audio if self.audio_id.is_none() => self.audio_id = Some(i),
                _ => {}
            }
        }
    }

    fn current_time(&self) -> u64 {
        if self.state != State::Playing {
            return 0;
        }
        let now = if self.paused { self.pause_time } else { Instant::now() };
        let elapsed = now.saturating_duration_since(self.clock_start).saturating_sub(self.paused_extra);
        self.start_time_ms + elapsed.as_millis() as u64
    }

    /// Stop the pipeline and take back the media and its buffers.
    fn halt(&mut self) {
        let Some(pipeline) = self.pipeline.take() else { return };
        if let Some(g) = &self.guest_file {
            g.cancel();
        }
        let (media, video, audio) = pipeline.stop();
        if let Some(g) = &self.guest_file {
            g.set_remote(false);
        }
        if media.is_some() {
            self.media = media;
        }
        self.idle_video.extend(video);
        self.idle_audio.extend(audio);
    }

    /// Drop queued output (and, unless retained, the buffers the title may
    /// still be showing).
    fn reset(&mut self, retain_output: bool) -> FreeList {
        let mut f = FreeList { video: std::mem::take(&mut self.idle_video), audio: std::mem::take(&mut self.idle_audio) };
        if !retain_output {
            f.video.extend(self.current_video.take().map(|c| c.buf));
            f.audio.extend(self.current_audio.take().map(|c| c.buf));
        }
        self.seek_video_frame_pending = false;
        self.last_output_loop_offset = 0;
        self.pass_offset = 0;
        self.pass_shift = 0;
        self.last_due = None;
        self.pending_loop_warnings = 0;
        f
    }

    fn record_loop_boundary(&mut self, offset: u64) {
        if offset > self.last_output_loop_offset {
            self.last_output_loop_offset = offset;
            self.pending_loop_warnings += 1;
        }
    }

    fn take_warnings(&mut self, events: &mut Events) {
        while self.pending_loop_warnings > 0 {
            self.pending_loop_warnings -= 1;
            events.warning(WARNING_LOOPING_BACK);
        }
    }

    /// Once a playing movie has handed out its last frame (the pipeline is
    /// drained, so it is not looping), report STOP; the title may still
    /// call sceAvPlayerStop.
    fn take_end(&mut self, events: &mut Events) {
        if self.state != State::Playing || self.end_reported {
            return;
        }
        if self.pipeline.as_ref().is_some_and(|p| {
            let p = p.lock();
            !p.failed && p.drained()
        }) {
            self.end_reported = true;
            events.state(EVENT_STATE_STOP);
            if self.video_id.is_some() {
                log::info!("[avplayer] video reached its end");
            }
        }
    }

    fn active(&self) -> bool {
        match self.state {
            State::Ready => true,
            State::Playing => self.pipeline.as_ref().is_some_and(|p| {
                let p = p.lock();
                !p.failed && !p.drained()
            }),
            State::Stopped => false,
        }
    }

    fn frame_ms(&self) -> u64 {
        match self.video_id.map(|i| self.streams[i].fps) {
            Some(fps) if fps > 0.0 && fps.is_finite() => (1000.0 / fps) as u64,
            _ => 33,
        }
    }

    /// Presentation time of a picture. The first picture of a looped pass is
    /// due one frame after the last picture of the previous pass, and the
    /// rest of that pass keeps the same shift.
    fn due_time(&self, f: &ReadyFrame) -> u64 {
        let raw = f.ts + f.offset;
        if f.offset > self.pass_offset {
            return self.last_due.map_or(raw, |d| d + self.frame_ms());
        }
        (raw as i64 + self.pass_shift).max(0) as u64
    }

    /// The next picture if it is due.
    ///
    /// Pictures are released on the presentation clock in both A/V sync
    /// modes; the mode only decides whether audio delivery drives the clock.
    /// (Handing pictures out unpaced with sync off makes a title that polls
    /// until a picture reaches its tick time drain looped passes forever.)
    fn video(&mut self) -> Option<ReadyFrame> {
        if self.state != State::Playing || self.video_id.is_none() {
            return None;
        }
        let deliver_seek_frame = self.paused && self.seek_video_frame_pending;
        // While paused, only a picture before the first one shown is
        // delivered (a title that pauses right after audio starts still
        // gets its first picture).
        if self.paused && !deliver_seek_frame && self.current_video.is_some() {
            return None;
        }
        let now = self.current_time();
        let pipeline = self.pipeline.as_ref()?;
        let (frame, due) = {
            let mut p = pipeline.lock();
            let f = p.video_frames.front()?;
            let due = self.due_time(f);
            // A picture within three quarters of a frame of the clock is
            // presented now.
            let tolerance = self.frame_ms() * 3 / 4;
            if !(deliver_seek_frame || now == 0 || due <= now + tolerance) {
                return None;
            }
            let mut f = p.video_frames.pop_front()?;
            // Pictures of the same pass that the clock has also passed are
            // skipped (back to the pool, never shown), so a title that asks
            // for pictures slower than the film's rate keeps up with the
            // audio. With A/V sync off this applies while audio drives.
            let skip_late = self.sync_mode == 0 || self.audio_driving;
            if skip_late && !deliver_seek_frame && now != 0 && f.offset <= self.pass_offset {
                while let Some(next) = p.video_frames.front() {
                    if next.offset != f.offset || self.due_time(next) > now {
                        break;
                    }
                    let late = std::mem::replace(&mut f, p.video_frames.pop_front()?);
                    p.video_pool.push(late.buf);
                }
            }
            let due = if f.offset <= self.pass_offset { self.due_time(&f) } else { due };
            // The previous picture's buffer is free again: the decoder may
            // run ahead into every buffer but the one now shown.
            if let Some(current) = self.current_video.take() {
                p.video_pool.push(current.buf);
            }
            (f, due)
        };
        pipeline.notify();
        if frame.offset > self.pass_offset {
            self.pass_offset = frame.offset;
            self.pass_shift = due as i64 - (frame.ts + frame.offset) as i64;
        }
        self.last_due = Some(due);
        self.current_video = Some(frame);
        self.record_loop_boundary(frame.offset);
        if deliver_seek_frame {
            self.seek_video_frame_pending = false;
        }
        Some(frame)
    }

    /// The next audio chunk; re-anchors the clock to it.
    fn audio(&mut self) -> Option<ReadyFrame> {
        if self.paused || self.state != State::Playing || self.audio_id.is_none() || self.trick_speed != TRICK_SPEED_NORMAL {
            return None;
        }
        let pipeline = self.pipeline.as_ref()?;
        let frame = {
            let mut p = pipeline.lock();
            let f = p.audio_frames.pop_front()?;
            if let Some(current) = self.current_audio.take() {
                p.audio_pool.push(current.buf);
            }
            f
        };
        pipeline.notify();
        self.current_audio = Some(frame);
        self.audio_driving = true;
        // Audio drives the presentation clock unless A/V sync is off.
        if self.sync_mode == 0 {
            let shift = if frame.offset == self.pass_offset { self.pass_shift } else { 0 };
            self.start_time_ms = ((frame.ts + frame.offset) as i64 + shift).max(0) as u64;
            self.clock_start = Instant::now();
            self.paused_extra = Duration::ZERO;
        }
        self.record_loop_boundary(frame.offset);
        Some(frame)
    }
}

struct Player {
    handle: u64,
    mem: MemReplacement,
    file: FileReplacement,
    event: EventReplacement,
    auto_start: bool,
    video_buffers: usize,
    demux_video_buffer_size: u32,
    video_decoder_type: u32,
    sync_mode: u32,
    bandwidth: (u32, u32, u32),
    source: Option<Source>,
}

type PlayerRef = Arc<Mutex<Player>>;

static PLAYERS: Mutex<Option<HashMap<u64, PlayerRef>>> = Mutex::new(None);

fn player(handle: u64) -> Option<PlayerRef> {
    PLAYERS.lock().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&handle).cloned()
}

fn lock(p: &PlayerRef) -> std::sync::MutexGuard<'_, Player> {
    p.lock().unwrap_or_else(|e| e.into_inner())
}

/// Perform a file read the demuxer is waiting for.
fn service(p: &PlayerRef) {
    let g = lock(p).source.as_ref().and_then(|s| s.guest_file.clone());
    if let Some(g) = g {
        g.service();
    }
}

// ---------------------------------------------------------------------------
// Init / Close
// ---------------------------------------------------------------------------

fn read_mem(ctx: &CpuContext, base: u64) -> MemReplacement {
    let r = |o: u64| ctx.read_u64(base + o).unwrap_or(0);
    MemReplacement { object: r(0), allocate: r(8), deallocate: r(16), allocate_texture: r(24), deallocate_texture: r(32) }
}

fn read_file(ctx: &CpuContext, base: u64) -> FileReplacement {
    let r = |o: u64| ctx.read_u64(base + o).unwrap_or(0);
    FileReplacement { object: r(0), open: r(8), close: r(16), read_offset: r(24), size: r(32) }
}

fn read_event(ctx: &CpuContext, base: u64) -> EventReplacement {
    let r = |o: u64| ctx.read_u64(base + o).unwrap_or(0);
    EventReplacement { object: r(0), callback: r(8) }
}

fn create_player(ctx: &CpuContext, mem: MemReplacement, file: FileReplacement, event: EventReplacement, auto_start: bool, buffers: i32) -> Option<u64> {
    if !mem.valid() {
        log::warn!("[avplayer] init without a complete memory replacement");
        return None;
    }
    let handle = crate::pthread::alloc_guest_block(ctx, 0x100)?;
    let video_buffers = (if buffers <= 0 { 2 } else { buffers }).clamp(2, 16) as usize;
    log::info!(
        "[avplayer] init handle={handle:#x} file_open={:#x} event={:#x} auto_start={auto_start} buffers={video_buffers}",
        file.open,
        event.callback
    );
    let player = Player {
        handle,
        mem,
        file,
        event,
        auto_start: auto_start || event.callback == 0,
        video_buffers,
        demux_video_buffer_size: source::DEMUX_BUFFER_DEFAULT,
        video_decoder_type: 0,
        sync_mode: 0,
        bandwidth: (0, 0, 0),
        source: None,
    };
    PLAYERS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(handle, Arc::new(Mutex::new(player)));
    Some(handle)
}

/// sceAvPlayerInit(SceAvPlayerInitData*) -> SceAvPlayerHandle
fn cb_init(ctx: &mut CpuContext) -> HleResult {
    let init = ctx.rdi();
    if init == 0 {
        return 0;
    }
    let buffers = ctx.read_u32(init + 104).unwrap_or(0) as i32;
    let auto_start = ctx.read_bytes(init + 108, 1).is_some_and(|b| b[0] != 0);
    create_player(ctx, read_mem(ctx, init), read_file(ctx, init + 40), read_event(ctx, init + 80), auto_start, buffers).unwrap_or(0)
        as HleResult
}

/// Size of the PS5 SceAvPlayerInitDataEx: auto start is the byte after
/// debug_level (0x74), nine 0x30-byte thread parameter blocks fill
/// 0x78..0x228 and the output video frame buffer count is at 0x228.
const INIT_DATA_EX_PS5_SIZE: u64 = 0x230;

/// (auto start, video frame buffer count) offsets of an
/// SceAvPlayerInitDataEx of `this_size` bytes: smaller structures use the
/// older layout (twelve u32 thread parameters, the count at 0xA4, auto
/// start at 0xA8).
fn init_ex_offsets(this_size: u64) -> (u64, u64) {
    if this_size >= INIT_DATA_EX_PS5_SIZE {
        (0x74, 0x228)
    } else {
        (0xA8, 0xA4)
    }
}

/// sceAvPlayerInitEx(const SceAvPlayerInitDataEx*, SceAvPlayerHandle*)
fn cb_init_ex(ctx: &mut CpuContext) -> HleResult {
    let (init, out) = (ctx.rdi(), ctx.rsi());
    if init == 0 || out == 0 {
        return err(ERROR_INVALID_PARAMS);
    }
    let (auto_start_at, buffers_at) = init_ex_offsets(ctx.read_u64(init).unwrap_or(0));
    let auto_start = ctx.read_bytes(init + auto_start_at, 1).is_some_and(|b| b[0] != 0);
    let buffers = ctx.read_u32(init + buffers_at).unwrap_or(0) as i32;
    let h = create_player(ctx, read_mem(ctx, init + 8), read_file(ctx, init + 48), read_event(ctx, init + 88), auto_start, buffers);
    ctx.write_u64(out, h.unwrap_or(0));
    if h.is_some() {
        OK
    } else {
        err(ERROR_INVALID_PARAMS)
    }
}

/// sceAvPlayerPostInit(handle, const SceAvPlayerPostInitData*)
fn cb_post_init(ctx: &mut CpuContext) -> HleResult {
    let Some(p) = player(ctx.rdi()) else { return err(ERROR_INVALID_PARAMS) };
    let data = ctx.rsi();
    if data != 0 {
        let mut g = lock(&p);
        g.demux_video_buffer_size = ctx.read_u32(data).unwrap_or(0);
        g.video_decoder_type = ctx.read_u32(data + 8).unwrap_or(0);
        log::info!("[avplayer] post init demux={:#x} video decoder={}", g.demux_video_buffer_size, g.video_decoder_type);
    }
    OK
}

/// Stop playback (STOP event when it was playing) and release everything.
fn stop_source(src: &mut Source, events: &mut Events) -> FreeList {
    if src.state == State::Playing {
        events.state(EVENT_STATE_STOP);
        if src.video_id.is_some() {
            log::info!("[avplayer] video stopped");
        }
    }
    src.state = State::Stopped;
    src.halt();
    src.reset(false)
}

/// sceAvPlayerClose(handle)
fn cb_close(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    let removed = PLAYERS.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|m| m.remove(&handle));
    let Some(p) = removed else { return err(ERROR_INVALID_PARAMS) };
    let mut events = Events::default();
    let (mem, event, free, src) = {
        let mut g = lock(&p);
        let free = g.source.as_mut().map(|s| stop_source(s, &mut events)).unwrap_or_default();
        (g.mem, g.event, free, g.source.take())
    };
    deliver(ctx, handle, event, events);
    free.release(mem);
    // Closes the title's file through its callbacks.
    drop(src);
    log::info!("[avplayer] close handle={handle:#x}");
    OK
}

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

fn source_type_from_path(path: &str) -> u32 {
    let name = path.split(['?', '#']).next().unwrap_or(path);
    let Some(dot) = name.rfind('.') else { return SOURCE_TYPE_UNKNOWN };
    let ext = name[dot..].split('/').next().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        ".mp4" | ".m4v" | ".m4a" | ".mov" | ".m3d" => SOURCE_TYPE_FILE_MP4,
        ".webm" => SOURCE_TYPE_FILE_WEBM,
        ".m3u8" => SOURCE_TYPE_HLS,
        _ => SOURCE_TYPE_UNKNOWN,
    }
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let hex = |c: u8| (c as char).to_digit(16);
            if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Guest paths to try for a media URI: file: URIs and engine-relative
/// paths ("../../../Game/Content/Movies/x.mp4") map into /app0.
fn path_candidates(uri: &str) -> Vec<String> {
    let mut p = uri.replace('\\', "/");
    let lower = p.to_ascii_lowercase();
    if lower.starts_with("file://") {
        p = percent_decode(&p[7..]);
    } else if lower.starts_with("file:") {
        p = percent_decode(&p[5..]);
    }
    let mut out = vec![p.clone()];
    let mut rel = p.as_str();
    while let Some(r) = rel.strip_prefix("../").or_else(|| rel.strip_prefix("./")) {
        rel = r;
    }
    if rel != p {
        out.push(rel.to_string());
        out.push(format!("/app0/{rel}"));
    } else if !p.starts_with('/') {
        out.push(format!("/app0/{p}"));
    }
    out
}

fn resolve_host_file(uri: &str) -> Option<std::path::PathBuf> {
    path_candidates(uri)
        .iter()
        .filter_map(|c| crate::filesystem::resolve_guest_path(c))
        .find(|p| p.is_file())
}

/// Open the movie: through the title's file callbacks when it supplied an
/// open callback, otherwise from the host file system.
fn open_streamer(file: FileReplacement, path: &str) -> Option<FileStreamer> {
    if file.open != 0 {
        return FileStreamer::open_guest(file, path);
    }
    let host = resolve_host_file(path)?;
    log::info!("[avplayer] source {path} -> {}", host.display());
    FileStreamer::open_native(&host)
}

fn add_source(ctx: &mut CpuContext, handle: u64, path: &str, requested: u32) -> HleResult {
    let Some(p) = player(handle) else { return err(ERROR_INVALID_PARAMS) };
    let (file, event, auto_start, buffers, demux_bytes, vdec2, sync_mode) = {
        let g = lock(&p);
        if g.source.is_some() {
            return err(ERROR_OPERATION_FAILED);
        }
        (
            g.file,
            g.event,
            g.auto_start,
            g.video_buffers,
            g.demux_video_buffer_size,
            g.video_decoder_type == VIDEO_DECODER_SOFTWARE2,
            g.sync_mode,
        )
    };
    let source_type = if requested == SOURCE_TYPE_UNKNOWN { source_type_from_path(path) } else { requested };
    if source_type == SOURCE_TYPE_HLS {
        log::warn!("[avplayer] HLS source {path} is not supported");
        return err(ERROR_NOT_SUPPORTED);
    }
    if !matches!(source_type, SOURCE_TYPE_UNKNOWN | SOURCE_TYPE_FILE_MP4 | SOURCE_TYPE_FILE_WEBM) {
        return err(ERROR_INVALID_PARAMS);
    }
    let Some(streamer) = open_streamer(file, path) else {
        log::warn!("[avplayer] cannot open source {path}");
        return err(ERROR_OPERATION_FAILED);
    };
    let guest_file = streamer.guest().cloned();
    let probe_name = (file.open == 0).then_some(path);
    let media = match Media::open(streamer, probe_name) {
        Ok(m) => m,
        Err(e) => {
            log::warn!("[avplayer] cannot open source {path}: {e}");
            return err(ERROR_OPERATION_FAILED);
        }
    };
    let streams = media.streams();
    for (i, s) in streams.iter().enumerate() {
        log::info!(
            "[avplayer]   stream {i} {:?} {}x{} {:.3}fps {}ch {}Hz {}ms",
            s.media_type,
            s.width,
            s.height,
            s.fps,
            s.channels,
            s.sample_rate,
            s.duration_ms
        );
    }
    let now = Instant::now();
    let src = Source {
        source_type,
        container_ms: media.duration_ms(),
        media: Some(media),
        pipeline: None,
        guest_file,
        streams,
        vdec2,
        max_video_buffers: buffers,
        demux_bytes: if demux_bytes == 0 { source::DEMUX_BUFFER_DEFAULT } else { demux_bytes },
        video_id: None,
        audio_id: None,
        state: State::Ready,
        paused: false,
        seek_video_frame_pending: false,
        looping: false,
        trick_speed: TRICK_SPEED_NORMAL,
        sync_mode,
        audio_driving: false,
        start_time_ms: 0,
        clock_start: now,
        pause_time: now,
        paused_extra: Duration::ZERO,
        last_output_loop_offset: 0,
        pass_offset: 0,
        pass_shift: 0,
        last_due: None,
        pending_loop_warnings: 0,
        end_reported: false,
        idle_video: Vec::new(),
        idle_audio: Vec::new(),
        current_video: None,
        current_audio: None,
    };
    {
        let mut g = lock(&p);
        if g.source.is_some() {
            return err(ERROR_OPERATION_FAILED);
        }
        g.source = Some(src);
    }
    let mut events = Events::default();
    events.state(EVENT_STATE_READY);
    deliver(ctx, handle, event, events);
    if auto_start {
        // The READY callback may already have started playback: the auto
        // start then leaves it alone.
        let Some(r) = start_pipeline(&p, 0, false, false, true) else { return OK };
        if r == OK {
            let mut events = Events::default();
            events.state(EVENT_STATE_PLAY);
            deliver(ctx, handle, event, events);
        }
        return r;
    }
    OK
}

/// sceAvPlayerAddSource(handle, const char* filename)
fn cb_add_source(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    if player(handle).is_none() || ctx.rsi() == 0 {
        return err(ERROR_INVALID_PARAMS);
    }
    let Some(path) = ctx.read_cstring(ctx.rsi(), 4096) else { return err(ERROR_INVALID_PARAMS) };
    add_source(ctx, handle, &path, SOURCE_TYPE_UNKNOWN)
}

/// sceAvPlayerAddSourceEx(handle, SceAvPlayerUriType, SceAvPlayerSourceDetails*)
fn cb_add_source_ex(ctx: &mut CpuContext) -> HleResult {
    let (handle, uri_type, details) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx());
    if player(handle).is_none() || uri_type != 0 || details == 0 {
        return err(ERROR_INVALID_PARAMS);
    }
    let name = ctx.read_u64(details).unwrap_or(0);
    let len = ctx.read_u32(details + 8).unwrap_or(0) as usize;
    let source_type = ctx.read_u32(details + 80).unwrap_or(0);
    if name == 0 || len == 0 || len > 4096 {
        return err(ERROR_INVALID_PARAMS);
    }
    if !matches!(source_type, SOURCE_TYPE_UNKNOWN | SOURCE_TYPE_FILE_MP4 | SOURCE_TYPE_FILE_WEBM | SOURCE_TYPE_HLS) {
        return err(ERROR_INVALID_PARAMS);
    }
    let Some(bytes) = ctx.read_bytes(name, len) else { return err(ERROR_INVALID_PARAMS) };
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let path = String::from_utf8_lossy(&bytes[..end]).into_owned();
    add_source(ctx, handle, &path, source_type)
}

/// sceAvPlayerStreamCount(handle)
fn cb_stream_count(ctx: &mut CpuContext) -> HleResult {
    let Some(p) = player(ctx.rdi()) else { return err(ERROR_INVALID_PARAMS) };
    let g = lock(&p);
    g.source.as_ref().map_or(0, |s| s.streams.len() as HleResult)
}

fn stream_type(s: &StreamInfo) -> u32 {
    match s.media_type {
        MediaType::Video => STREAM_TYPE_VIDEO,
        MediaType::Audio => STREAM_TYPE_AUDIO,
        MediaType::Subtitle => STREAM_TYPE_TIMED_TEXT,
        MediaType::Other => STREAM_TYPE_UNKNOWN,
    }
}

fn put_u16(b: &mut [u8], o: usize, v: u16) {
    b[o..o + 2].copy_from_slice(&v.to_le_bytes());
}
fn put_u32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut [u8], o: usize, v: u64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
}
fn put_f32(b: &mut [u8], o: usize, v: f32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_f64(b: &mut [u8], o: usize, v: f64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
}

/// SceAvPlayerVideo / SceAvPlayerAudio (16 bytes) at `o`.
fn encode_details(vdec2: bool, s: &StreamInfo, b: &mut [u8], o: usize, sar: f32, audio_size: u32) {
    match s.media_type {
        MediaType::Video => {
            let (w, h, _) = video_dims(vdec2, s);
            put_u32(b, o, w);
            put_u32(b, o + 4, h);
            put_f32(b, o + 8, sar);
            b[o + 12..o + 16].copy_from_slice(&s.language);
        }
        MediaType::Audio => {
            put_u16(b, o, s.channels as u16);
            put_u32(b, o + 4, s.sample_rate);
            put_u32(b, o + 8, audio_size);
            b[o + 12..o + 16].copy_from_slice(&s.language);
        }
        _ => {}
    }
}

/// SceAvPlayerVideoEx / SceAvPlayerAudioEx (80 bytes) at `o`.
fn encode_details_ex(vdec2: bool, s: &StreamInfo, b: &mut [u8], o: usize, sar: f32, crop: [u32; 4], audio_size: u32) {
    match s.media_type {
        MediaType::Video => {
            let (w, h, pitch) = video_dims(vdec2, s);
            put_u32(b, o, w);
            put_u32(b, o + 4, h);
            put_f32(b, o + 8, sar);
            b[o + 12..o + 16].copy_from_slice(&s.language);
            for (i, c) in crop.iter().enumerate() {
                put_u32(b, o + 20 + i * 4, *c);
            }
            put_u32(b, o + 36, pitch);
            b[o + 40] = 8;
            b[o + 41] = 8;
            b[o + 42] = s.full_range as u8;
            put_f64(b, o + 48, s.fps);
            put_u32(b, o + 56, s.color_primaries);
            put_u32(b, o + 60, s.color_trc);
        }
        MediaType::Audio => {
            put_u16(b, o, s.channels as u16);
            put_u32(b, o + 4, s.sample_rate);
            put_u32(b, o + 8, audio_size);
            b[o + 12..o + 16].copy_from_slice(&s.language);
        }
        _ => {}
    }
}

/// SceAvPlayerStreamInfo: type, details, duration.
fn encode_stream_info(vdec2: bool, s: &StreamInfo) -> [u8; STREAM_INFO_SIZE] {
    let mut b = [0u8; STREAM_INFO_SIZE];
    put_u32(&mut b, 0, stream_type(s));
    encode_details(vdec2, s, &mut b, 8, s.sar, 0);
    put_u64(&mut b, 24, s.duration_ms);
    b
}

/// SceAvPlayerStreamInfoEx: this_size, type, 80-byte details, duration.
fn encode_stream_info_ex(vdec2: bool, s: &StreamInfo) -> [u8; STREAM_INFO_EX_SIZE] {
    let mut b = [0u8; STREAM_INFO_EX_SIZE];
    put_u64(&mut b, 0, STREAM_INFO_EX_SIZE as u64);
    put_u32(&mut b, 8, stream_type(s));
    encode_details_ex(vdec2, s, &mut b, 16, s.sar, [0; 4], 0);
    put_u64(&mut b, 96, s.duration_ms);
    b
}

fn get_stream_info(ctx: &mut CpuContext, ex: bool) -> HleResult {
    let (id, out) = (ctx.rsi() as u32 as usize, ctx.rdx());
    let Some(p) = player(ctx.rdi()) else { return err(ERROR_INVALID_PARAMS) };
    if out == 0 {
        return err(ERROR_INVALID_PARAMS);
    }
    let g = lock(&p);
    let Some(src) = g.source.as_ref() else { return err(ERROR_OPERATION_FAILED) };
    let s = src.streams.get(id);
    let ok = if ex {
        let b = s.map_or_else(
            || {
                let mut b = [0u8; STREAM_INFO_EX_SIZE];
                put_u64(&mut b, 0, STREAM_INFO_EX_SIZE as u64);
                b
            },
            |s| encode_stream_info_ex(src.vdec2, s),
        );
        ctx.write_bytes(out, &b)
    } else {
        ctx.write_bytes(out, &s.map_or([0; STREAM_INFO_SIZE], |s| encode_stream_info(src.vdec2, s)))
    };
    if s.is_none() {
        err(ERROR_OPERATION_FAILED)
    } else if ok {
        OK
    } else {
        err(ERROR_INVALID_PARAMS)
    }
}

/// sceAvPlayerGetStreamInfo(handle, stream_id, SceAvPlayerStreamInfo*)
fn cb_get_stream_info(ctx: &mut CpuContext) -> HleResult {
    get_stream_info(ctx, false)
}

/// sceAvPlayerGetStreamInfoEx(handle, stream_id, SceAvPlayerStreamInfoEx*)
fn cb_get_stream_info_ex(ctx: &mut CpuContext) -> HleResult {
    get_stream_info(ctx, true)
}

fn with_source(ctx: &CpuContext, f: impl FnOnce(&mut Source) -> HleResult) -> HleResult {
    let Some(p) = player(ctx.rdi()) else { return err(ERROR_INVALID_PARAMS) };
    let mut g = lock(&p);
    match g.source.as_mut() {
        Some(s) => f(s),
        None => err(ERROR_INVALID_PARAMS),
    }
}

/// sceAvPlayerEnableStream(handle, stream_id)
fn cb_enable_stream(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rsi() as u32 as usize;
    with_source(ctx, |src| {
        if src.state == State::Playing || id >= src.streams.len() {
            return err(ERROR_OPERATION_FAILED);
        }
        if !src.stream_supported(id) {
            return err(ERROR_NOT_SUPPORTED);
        }
        match src.streams[id].media_type {
            MediaType::Video => src.video_id = Some(id),
            MediaType::Audio => src.audio_id = Some(id),
            _ => return err(ERROR_NOT_SUPPORTED),
        }
        OK
    })
}

/// sceAvPlayerDisableStream(handle, stream_id)
fn cb_disable_stream(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rsi() as u32 as usize;
    with_source(ctx, |src| {
        if src.state == State::Playing {
            return err(ERROR_OPERATION_FAILED);
        }
        if src.video_id == Some(id) {
            src.video_id = None;
        } else if src.audio_id == Some(id) {
            src.audio_id = None;
        } else {
            return err(ERROR_OPERATION_FAILED);
        }
        OK
    })
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

/// What `start_impl` prepares outside the player lock.
struct StartPlan {
    media: Media,
    video: Option<(usize, VideoSurface)>,
    audio: Option<usize>,
    video_count: usize,
    audio_count: usize,
    loop_period: u64,
}

/// (Re)start the pipeline at `ms`. Allocation and seeking run guest code
/// (allocator, file callbacks), so they happen without the player lock.
fn start_impl(p: &PlayerRef, ms: u64, paused_after_start: bool, show_seek_frame: bool) -> HleResult {
    start_pipeline(p, ms, paused_after_start, show_seek_frame, false).unwrap_or(OK)
}

/// [`start_impl`]; with `only_if_ready` (the auto start) nothing happens
/// (None) unless the source is still ready and nobody is starting it, as
/// checked under the player lock.
fn start_pipeline(p: &PlayerRef, ms: u64, paused_after_start: bool, show_seek_frame: bool, only_if_ready: bool) -> Option<HleResult> {
    let (mem, plan, free) = {
        let mut g = lock(p);
        let mem = g.mem;
        let Some(src) = g.source.as_mut() else { return Some(err(ERROR_INVALID_PARAMS)) };
        if only_if_ready && !(src.state == State::Ready && src.pipeline.is_none() && src.media.is_some()) {
            return None;
        }
        src.state = State::Ready;
        src.audio_driving = false;
        src.halt();
        let free = src.reset(true);
        if src.video_id.is_none() && src.audio_id.is_none() {
            src.auto_enable();
        }
        let media = match src.media.take() {
            Some(m) if src.video_id.is_some() || src.audio_id.is_some() => m,
            m => {
                src.media = m;
                src.state = State::Stopped;
                drop(g);
                free.release(mem);
                return Some(err(ERROR_OPERATION_FAILED));
            }
        };
        let video = src.video_id.map(|i| {
            let s = &src.streams[i];
            let (_, rows, pitch) = video_dims(src.vdec2, s);
            (i, VideoSurface { pitch, rows, full_range: s.full_range })
        });
        let retained = src.current_video.is_some() as usize;
        let mut loop_period = src.container_ms;
        if loop_period == 0 {
            loop_period = [src.video_id, src.audio_id].iter().flatten().map(|&i| src.streams[i].duration_ms).max().unwrap_or(0);
        }
        let plan = StartPlan {
            media,
            video,
            audio: src.audio_id,
            video_count: src.max_video_buffers.saturating_sub(retained),
            audio_count: AUDIO_BUFFERS - src.current_audio.is_some() as usize,
            loop_period,
        };
        (mem, plan, free)
    };
    free.release(mem);
    let StartPlan { mut media, video, audio, video_count, audio_count, loop_period } = plan;

    // Restore the media on failure.
    let fail = |media: Media, bufs: FreeList, code: u32| -> HleResult {
        let mut g = lock(p);
        if let Some(src) = g.source.as_mut() {
            src.media = Some(media);
            src.state = State::Stopped;
        }
        drop(g);
        bufs.release(mem);
        err(code)
    };

    let video_dec = match video {
        Some((i, surface)) => match media.decoder(i) {
            Some(d) => Some((i, d, surface)),
            None => {
                log::warn!("[avplayer] cannot open the video decoder (stream {i})");
                return Some(fail(media, FreeList::default(), ERROR_OPERATION_FAILED));
            }
        },
        None => None,
    };
    let audio_dec = match audio {
        Some(i) => match media.decoder(i) {
            Some(d) => Some((i, d)),
            None => {
                log::warn!("[avplayer] cannot open the audio decoder (stream {i})");
                return Some(fail(media, FreeList::default(), ERROR_OPERATION_FAILED));
            }
        },
        None => None,
    };

    let mut bufs = FreeList::default();
    if let Some((_, _, surface)) = &video_dec {
        let size = surface.bytes() as u32;
        for _ in 0..video_count {
            match mem.alloc(size, true) {
                Some(b) => bufs.video.push(b),
                None => return Some(fail(media, bufs, ERROR_NO_MEMORY)),
            }
        }
        // Decode-ahead pictures beyond the title's count, so the decoder can
        // run ahead of a title that holds its pictures and late pictures can
        // be skipped. They are optional: all or none.
        let mut extra = Vec::with_capacity(VIDEO_DECODE_AHEAD_FRAMES);
        for _ in 0..VIDEO_DECODE_AHEAD_FRAMES {
            match mem.alloc(size, true) {
                Some(b) => extra.push(b),
                None => break,
            }
        }
        if extra.len() == VIDEO_DECODE_AHEAD_FRAMES {
            bufs.video.extend(extra);
        } else {
            FreeList { video: extra, audio: Vec::new() }.release(mem);
        }
    }
    if let Some((_, d)) = &audio_dec {
        let (channels, frame_size) = d.params();
        let size = channels.max(1) * frame_size.max(1024) * 2;
        for _ in 0..audio_count {
            match mem.alloc(size, false) {
                Some(b) => bufs.audio.push(b),
                None => return Some(fail(media, bufs, ERROR_NO_MEMORY)),
            }
        }
    }
    media.seek(video.map(|v| v.0).or(audio), ms);

    let mut g = lock(p);
    let Some(src) = g.source.as_mut() else {
        drop(g);
        bufs.release(mem);
        return Some(err(ERROR_OPERATION_FAILED));
    };
    let settings = Settings {
        video: video_dec,
        audio: audio_dec,
        loop_period,
        demux_bytes: src.demux_bytes,
        looping: src.looping,
        video_pool: bufs.video,
        audio_pool: bufs.audio,
    };
    if let Some(gf) = &src.guest_file {
        gf.set_remote(true);
    }
    Some(match Pipeline::start(media, settings) {
        Ok(pipeline) => {
            src.pipeline = Some(pipeline);
            let now = Instant::now();
            src.start_time_ms = ms;
            src.clock_start = now;
            src.pause_time = now;
            src.paused_extra = Duration::ZERO;
            src.paused = paused_after_start;
            src.seek_video_frame_pending = paused_after_start && show_seek_frame && src.video_id.is_some();
            src.state = State::Playing;
            src.end_reported = false;
            if src.video_id.is_some() {
                log::info!("[avplayer] video started playing at {ms}ms");
            }
            OK
        }
        Err((media, video, audio)) => {
            if let Some(gf) = &src.guest_file {
                gf.set_remote(false);
            }
            src.media = Some(media);
            src.state = State::Stopped;
            drop(g);
            FreeList { video, audio }.release(mem);
            err(ERROR_OPERATION_FAILED)
        }
    })
}

fn emit(ctx: &CpuContext, p: &PlayerRef, id: i32, warning: Option<u32>) {
    let (handle, event) = {
        let g = lock(p);
        (g.handle, g.event)
    };
    let mut events = Events::default();
    match warning {
        Some(w) => events.warning(w),
        None => events.state(id),
    }
    deliver(ctx, handle, event, events);
}

fn has_source(p: &PlayerRef) -> bool {
    lock(p).source.is_some()
}

fn start_export(ctx: &mut CpuContext, ms: u64) -> HleResult {
    let Some(p) = player(ctx.rdi()) else { return err(ERROR_INVALID_PARAMS) };
    if !has_source(&p) {
        return err(ERROR_INVALID_PARAMS);
    }
    let r = start_impl(&p, ms, false, false);
    if r == OK {
        emit(ctx, &p, EVENT_STATE_PLAY, None);
    }
    r
}

/// sceAvPlayerStart(handle)
fn cb_start(ctx: &mut CpuContext) -> HleResult {
    start_export(ctx, 0)
}

/// sceAvPlayerStartEx(handle, const SceAvPlayerStartInfoEx*)
fn cb_start_ex(ctx: &mut CpuContext) -> HleResult {
    let info = ctx.rsi();
    let ms = if info != 0 { ctx.read_u64(info + 8).unwrap_or(0) } else { 0 };
    start_export(ctx, ms)
}

/// sceAvPlayerStop(handle)
fn cb_stop(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    let Some(p) = player(handle) else { return err(ERROR_INVALID_PARAMS) };
    let mut events = Events::default();
    let (mem, event, free) = {
        let mut g = lock(&p);
        let (mem, event) = (g.mem, g.event);
        let Some(src) = g.source.as_mut() else { return err(ERROR_INVALID_PARAMS) };
        (mem, event, stop_source(src, &mut events))
    };
    free.release(mem);
    deliver(ctx, handle, event, events);
    OK
}

/// sceAvPlayerPause(handle)
fn cb_pause(ctx: &mut CpuContext) -> HleResult {
    let Some(p) = player(ctx.rdi()) else { return err(ERROR_INVALID_PARAMS) };
    let r = {
        let mut g = lock(&p);
        let Some(src) = g.source.as_mut() else { return err(ERROR_INVALID_PARAMS) };
        if src.state != State::Playing {
            err(ERROR_OPERATION_FAILED)
        } else {
            src.paused = true;
            src.pause_time = Instant::now();
            OK
        }
    };
    if r == OK {
        emit(ctx, &p, EVENT_STATE_PAUSE, None);
    }
    r
}

/// sceAvPlayerResume(handle)
fn cb_resume(ctx: &mut CpuContext) -> HleResult {
    let Some(p) = player(ctx.rdi()) else { return err(ERROR_INVALID_PARAMS) };
    {
        let mut g = lock(&p);
        let Some(src) = g.source.as_mut() else { return err(ERROR_INVALID_PARAMS) };
        if src.paused {
            src.paused_extra += Instant::now().saturating_duration_since(src.pause_time);
        }
        src.paused = false;
        src.seek_video_frame_pending = false;
    }
    emit(ctx, &p, EVENT_STATE_PLAY, None);
    OK
}

/// sceAvPlayerJumpToTime(handle, uint64_t ms)
fn cb_jump_to_time(ctx: &mut CpuContext) -> HleResult {
    let ms = ctx.rsi();
    let Some(p) = player(ctx.rdi()) else { return err(ERROR_INVALID_PARAMS) };
    let was_paused = {
        let g = lock(&p);
        let Some(src) = g.source.as_ref() else { return err(ERROR_INVALID_PARAMS) };
        if src.state != State::Playing {
            return err(ERROR_OPERATION_FAILED);
        }
        src.paused
    };
    let r = start_impl(&p, ms, was_paused, true);
    if r == OK {
        emit(ctx, &p, EVENT_WARNING_ID, Some(WARNING_JUMP_COMPLETE));
    }
    r
}

/// sceAvPlayerSetLooping(handle, bool)
fn cb_set_looping(ctx: &mut CpuContext) -> HleResult {
    let looping = ctx.rsi() as u8 != 0;
    with_source(ctx, |src| {
        src.looping = looping;
        if let Some(pl) = &src.pipeline {
            pl.lock().looping = looping;
        }
        OK
    })
}

/// sceAvPlayerSetTrickSpeed(handle, int32_t percent)
/// A player without a source is OPERATION_FAILED, before the speed is
/// looked at (as the module's code checks it).
fn cb_set_trick_speed(ctx: &mut CpuContext) -> HleResult {
    let speed = (ctx.rsi() as u32 as i32).clamp(-TRICK_SPEED_MAX, TRICK_SPEED_MAX);
    let Some(p) = player(ctx.rdi()) else { return err(ERROR_INVALID_PARAMS) };
    if lock(&p).source.is_none() {
        return err(ERROR_OPERATION_FAILED);
    }
    drop(p);
    with_source(ctx, |src| {
        if speed != TRICK_SPEED_NORMAL && speed > -TRICK_SPEED_MIN && speed < TRICK_SPEED_MIN {
            return err(ERROR_NOT_SUPPORTED);
        }
        src.trick_speed = speed;
        if speed != TRICK_SPEED_NORMAL {
            src.audio_driving = false;
        }
        if speed == TRICK_SPEED_NORMAL {
            OK
        } else {
            err(ERROR_NOT_SUPPORTED)
        }
    })
}

/// sceAvPlayerSetAvSyncMode(handle, mode)
fn cb_set_av_sync_mode(ctx: &mut CpuContext) -> HleResult {
    let mode = ctx.rsi() as u32;
    let Some(p) = player(ctx.rdi()) else { return err(ERROR_INVALID_PARAMS) };
    if mode > SYNC_MODE_NONE {
        return err(ERROR_NOT_SUPPORTED);
    }
    let mut g = lock(&p);
    g.sync_mode = mode;
    if let Some(src) = g.source.as_mut() {
        src.sync_mode = mode;
    }
    OK
}

/// sceAvPlayerChangeStream(handle, old_id, new_id): switch the audio track.
fn cb_change_stream(ctx: &mut CpuContext) -> HleResult {
    let (old_id, new_id) = (ctx.rsi() as u32 as usize, ctx.rdx() as u32 as usize);
    let Some(p) = player(ctx.rdi()) else { return err(ERROR_INVALID_PARAMS) };
    let (time, restart, was_paused) = {
        let mut g = lock(&p);
        let Some(src) = g.source.as_mut() else { return err(ERROR_INVALID_PARAMS) };
        if new_id >= src.streams.len() {
            return err(ERROR_OPERATION_FAILED);
        }
        if !src.stream_supported(new_id) {
            return err(ERROR_NOT_SUPPORTED);
        }
        if src.audio_id != Some(old_id) || src.streams[new_id].media_type != MediaType::Audio {
            return err(ERROR_OPERATION_FAILED);
        }
        let time = src.current_time();
        let restart = src.state == State::Playing;
        let was_paused = src.paused;
        if restart {
            src.state = State::Ready;
            src.halt();
        }
        src.audio_id = Some(new_id);
        (time, restart, was_paused)
    };
    if !restart {
        return OK;
    }
    start_impl(&p, time, was_paused, false)
}

/// sceAvPlayerSetAvailableBW(handle, bandwidth): the short form, only the
/// start bandwidth; streaming sources are never fetched over a network.
fn cb_set_available_bw(ctx: &mut CpuContext) -> HleResult {
    let Some(p) = player(ctx.rdi()) else { return err(ERROR_INVALID_PARAMS) };
    lock(&p).bandwidth.0 = ctx.rsi() as u32;
    OK
}

/// sceAvPlayerSetAvailableBandwidth(handle, start, min, max)
fn cb_set_available_bandwidth(ctx: &mut CpuContext) -> HleResult {
    let (start, min, max) = (ctx.rsi() as u32, ctx.rdx() as u32, ctx.rcx() as u32);
    let Some(p) = player(ctx.rdi()) else { return err(ERROR_INVALID_PARAMS) };
    if min != 0 && max != 0 && min > max {
        return err(ERROR_INVALID_PARAMS);
    }
    let mut g = lock(&p);
    g.bandwidth = (start, min, max);
    if g.source.is_none() {
        OK
    } else {
        err(ERROR_OPERATION_FAILED)
    }
}

fn cb_set_log_callback(_ctx: &mut CpuContext) -> HleResult {
    OK
}

fn cb_printf(_ctx: &mut CpuContext) -> HleResult {
    OK
}

// ---------------------------------------------------------------------------
// Data delivery
// ---------------------------------------------------------------------------

/// Run `f` on the source, then report pending loop warnings.
fn deliver_data(ctx: &mut CpuContext, f: impl FnOnce(&mut CpuContext, &mut Source) -> bool) -> HleResult {
    let handle = ctx.rdi();
    let Some(p) = player(handle) else { return 0 };
    if ctx.rsi() == 0 {
        return 0;
    }
    service(&p);
    let mut events = Events::default();
    let (ok, event) = {
        let mut g = lock(&p);
        let event = g.event;
        let Some(src) = g.source.as_mut() else { return 0 };
        let ok = f(ctx, src);
        src.take_warnings(&mut events);
        src.take_end(&mut events);
        (ok, event)
    };
    deliver(ctx, handle, event, events);
    ok as HleResult
}

fn get_video_data(ctx: &mut CpuContext, ex: bool) -> HleResult {
    let out = ctx.rsi();
    deliver_data(ctx, |ctx, src| {
        let Some(frame) = src.video() else { return false };
        let FrameKind::Video { sar, crop } = frame.kind else { return false };
        let Some(s) = src.video_id.map(|i| &src.streams[i]) else { return false };
        let mut b = [0u8; FRAME_INFO_EX_SIZE];
        put_u64(&mut b, 0, frame.buf.addr);
        put_u64(&mut b, 16, frame.ts);
        if ex {
            encode_details_ex(src.vdec2, s, &mut b, 24, sar, crop, 0);
            ctx.write_bytes(out, &b)
        } else {
            encode_details(src.vdec2, s, &mut b, 24, sar, 0);
            ctx.write_bytes(out, &b[..FRAME_INFO_SIZE])
        }
    })
}

/// sceAvPlayerGetVideoData(handle, SceAvPlayerFrameInfo*) -> bool
fn cb_get_video_data(ctx: &mut CpuContext) -> HleResult {
    get_video_data(ctx, false)
}

/// sceAvPlayerGetVideoDataEx(handle, SceAvPlayerFrameInfoEx*) -> bool
fn cb_get_video_data_ex(ctx: &mut CpuContext) -> HleResult {
    get_video_data(ctx, true)
}

/// sceAvPlayerGetAudioData(handle, SceAvPlayerFrameInfo*) -> bool
fn cb_get_audio_data(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    deliver_data(ctx, |ctx, src| {
        let Some(frame) = src.audio() else { return false };
        let FrameKind::Audio { channels, sample_rate, size } = frame.kind else { return false };
        let lang = src.audio_id.map_or(*b"und\0", |i| src.streams[i].language);
        let mut b = [0u8; FRAME_INFO_SIZE];
        put_u64(&mut b, 0, frame.buf.addr);
        put_u64(&mut b, 16, frame.ts);
        put_u16(&mut b, 24, channels);
        put_u32(&mut b, 28, sample_rate);
        put_u32(&mut b, 32, size);
        b[36..40].copy_from_slice(&lang);
        ctx.write_bytes(out, &b)
    })
}

/// sceAvPlayerIsActive(handle) -> bool
fn cb_is_active(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    let Some(p) = player(handle) else { return 0 };
    service(&p);
    let mut events = Events::default();
    let (active, event) = {
        let mut g = lock(&p);
        let event = g.event;
        let active = g.source.as_mut().is_none_or(|s| {
            s.take_end(&mut events);
            s.active()
        });
        (active, event)
    };
    deliver(ctx, handle, event, events);
    active as HleResult
}

/// sceAvPlayerCurrentTime(handle) -> uint64_t ms
fn cb_current_time(ctx: &mut CpuContext) -> HleResult {
    let Some(p) = player(ctx.rdi()) else { return 0 };
    service(&p);
    let g = lock(&p);
    g.source.as_ref().map_or(0, |s| s.current_time()) as HleResult
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub(crate) type Export = (&'static str, &'static str, &'static str, HleCallback);

pub(crate) const EXPORTS: &[Export] = &[
    ("KMcEa+rHsIo", "sceAvPlayerAddSource", LIB, cb_add_source),
    ("x8uvuFOPZhU", "sceAvPlayerAddSourceEx", LIB, cb_add_source_ex),
    ("buMCiJftcfw", "sceAvPlayerChangeStream", LIB, cb_change_stream),
    ("NkJwDzKmIlw", "sceAvPlayerClose", LIB, cb_close),
    ("wwM99gjFf1Y", "sceAvPlayerCurrentTime", LIB, cb_current_time),
    ("BOVKAzRmuTQ", "sceAvPlayerDisableStream", LIB, cb_disable_stream),
    ("ODJK2sn9w4A", "sceAvPlayerEnableStream", LIB, cb_enable_stream),
    ("Wnp1OVcrZgk", "sceAvPlayerGetAudioData", LIB, cb_get_audio_data),
    ("d8FcbzfAdQw", "sceAvPlayerGetStreamInfo", LIB, cb_get_stream_info),
    ("ctTAcF5DiKQ", "sceAvPlayerGetStreamInfoEx", LIB, cb_get_stream_info_ex),
    ("o3+RWnHViSg", "sceAvPlayerGetVideoData", LIB, cb_get_video_data),
    ("JdksQu8pNdQ", "sceAvPlayerGetVideoDataEx", LIB, cb_get_video_data_ex),
    ("aS66RI0gGgo", "sceAvPlayerInit", LIB, cb_init),
    ("o9eWRkSL+M4", "sceAvPlayerInitEx", LIB, cb_init_ex),
    ("UbQoYawOsfY", "sceAvPlayerIsActive", LIB, cb_is_active),
    ("XC9wM+xULz8", "sceAvPlayerJumpToTime", LIB, cb_jump_to_time),
    ("9y5v+fGN4Wk", "sceAvPlayerPause", LIB, cb_pause),
    ("HD1YKVU26-M", "sceAvPlayerPostInit", LIB, cb_post_init),
    ("agig-iDRrTE", "sceAvPlayerPrintf", LIB, cb_printf),
    ("w5moABNwnRY", "sceAvPlayerResume", LIB, cb_resume),
    ("k-q+xOxdc3E", "sceAvPlayerSetAvSyncMode", LIB, cb_set_av_sync_mode),
    ("N6Oy-EjduiY", "sceAvPlayerSetAvailableBandwidth", LIB, cb_set_available_bandwidth),
    ("+7xJ+lFxmkQ", "sceAvPlayerSetAvailableBW", LIB, cb_set_available_bw),
    ("eBTreZ84JFY", "sceAvPlayerSetLogCallback", LIB, cb_set_log_callback),
    ("OVths0xGfho", "sceAvPlayerSetLooping", LIB, cb_set_looping),
    ("av8Z++94rs0", "sceAvPlayerSetTrickSpeed", LIB, cb_set_trick_speed),
    ("ET4Gr-Uu07s", "sceAvPlayerStart", LIB, cb_start),
    ("NxSdL9t-KXk", "sceAvPlayerStartEx", LIB, cb_start_ex),
    ("ZC17w3vB5Lo", "sceAvPlayerStop", LIB, cb_stop),
    ("hdTyRzCXQeQ", "sceAvPlayerStreamCount", LIB, cb_stream_count),
    ("yN7Jhuv8g24", "sceAvPlayerVprintf", LIB, cb_printf),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, lib, cb) in EXPORTS {
        reg.register_fn(nid, name, lib, cb);
    }
    if !host::available() {
        log::warn!("[avplayer] built without FFmpeg: movies will fail to open");
    }
}

#[cfg(test)]
mod tests;
