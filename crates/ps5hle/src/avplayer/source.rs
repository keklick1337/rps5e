// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Playback pipeline of one AvPlayer source: a demuxer thread feeding
//! byte-bounded packet queues, and one decoder thread per enabled stream
//! that decodes straight into the title's output buffers.
//!
//! Output buffers circulate between a free pool (filled by the guest side
//! as pictures are retired) and the ready-frame queues (drained by the
//! GetVideoData / GetAudioData exports). Workers never call guest code.
//!
//! Looping: at the end of the container the demuxer bumps a per-stream
//! timestamp offset by the movie duration, queues a marker that makes the
//! decoder drain and flush, and seeks back to the start. Frames keep their
//! container timestamps and carry the offset for pacing.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use super::host::{FrameMeta, Media, Packet, StreamDecoder};
use super::stream::AVERROR_EOF;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

const AVERROR_EAGAIN: i32 = -libc::EAGAIN;

pub const DEMUX_BUFFER_DEFAULT: u32 = 4 * 1024 * 1024;
pub const DEMUX_BUFFER_MIN: u32 = 128 * 1024;
pub const DEMUX_BUFFER_MAX: u32 = 32 * 1024 * 1024;

/// An output buffer allocated through the title's allocator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuestBuf {
    pub addr: u64,
    pub size: u32,
}

#[derive(Clone, Copy, Debug)]
pub enum FrameKind {
    Video { sar: f32, crop: [u32; 4] },
    Audio { channels: u16, sample_rate: u32, size: u32 },
}

/// A decoded frame waiting in (or taken from) a ready queue.
#[derive(Clone, Copy, Debug)]
pub struct ReadyFrame {
    pub buf: GuestBuf,
    pub ts: u64,
    pub offset: u64,
    pub kind: FrameKind,
}

#[derive(Default)]
struct PacketQueue {
    q: VecDeque<(Option<Packet>, u64)>,
    bytes: usize,
}

pub struct Pipe {
    pub stop: bool,
    pub failed: bool,
    pub demux_eof: bool,
    pub video_done: bool,
    pub audio_done: bool,
    pub looping: bool,
    capacity: usize,
    video_packets: PacketQueue,
    audio_packets: PacketQueue,
    pub video_pool: Vec<GuestBuf>,
    pub audio_pool: Vec<GuestBuf>,
    pub video_frames: VecDeque<ReadyFrame>,
    pub audio_frames: VecDeque<ReadyFrame>,
}

impl Pipe {
    /// Everything was demuxed, decoded and handed out.
    pub fn drained(&self) -> bool {
        self.demux_eof && self.video_done && self.audio_done && self.video_frames.is_empty() && self.audio_frames.is_empty()
    }
}

struct Shared {
    m: Mutex<Pipe>,
    cv: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Pipe> {
        self.m.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn wait<'a>(&self, g: MutexGuard<'a, Pipe>) -> MutexGuard<'a, Pipe> {
        self.cv.wait(g).unwrap_or_else(|e| e.into_inner())
    }

    fn fail(&self) {
        let mut p = self.lock();
        p.failed = true;
        p.stop = true;
        drop(p);
        self.cv.notify_all();
    }
}

/// NV12 surface the video decoder writes.
#[derive(Clone, Copy, Debug)]
pub struct VideoSurface {
    pub pitch: u32,
    pub rows: u32,
    /// Stream colour range: decides the black of the padding.
    pub full_range: bool,
}

impl VideoSurface {
    pub fn bytes(&self) -> u64 {
        self.pitch as u64 * (self.rows as u64 + self.rows.div_ceil(2) as u64)
    }
}

pub struct Settings {
    pub video: Option<(usize, StreamDecoder, VideoSurface)>,
    pub audio: Option<(usize, StreamDecoder)>,
    pub loop_period: u64,
    pub demux_bytes: u32,
    pub looping: bool,
    pub video_pool: Vec<GuestBuf>,
    pub audio_pool: Vec<GuestBuf>,
}

pub struct Pipeline {
    shared: Arc<Shared>,
    demux: Option<JoinHandle<Option<Media>>>,
    decoders: Vec<JoinHandle<()>>,
}

fn spawn<T: Send + 'static>(name: &str, f: impl FnOnce() -> T + Send + 'static) -> std::io::Result<JoinHandle<T>> {
    std::thread::Builder::new().name(name.into()).spawn(move || {
        block_async_signals();
        f()
    })
}

impl Pipeline {
    /// Start the workers; on failure the media and the buffers come back.
    #[allow(clippy::type_complexity)]
    pub fn start(mut media: Media, s: Settings) -> Result<Self, (Media, Vec<GuestBuf>, Vec<GuestBuf>)> {
        let capacity = s.demux_bytes.clamp(DEMUX_BUFFER_MIN, DEMUX_BUFFER_MAX) as usize;
        let video_id = s.video.as_ref().map(|v| v.0);
        let audio_id = s.audio.as_ref().map(|a| a.0);
        media.set_discard(video_id, audio_id);
        let shared = Arc::new(Shared {
            m: Mutex::new(Pipe {
                stop: false,
                failed: false,
                demux_eof: false,
                video_done: video_id.is_none(),
                audio_done: audio_id.is_none(),
                looping: s.looping,
                capacity,
                video_packets: PacketQueue::default(),
                audio_packets: PacketQueue::default(),
                video_pool: s.video_pool,
                audio_pool: s.audio_pool,
                video_frames: VecDeque::new(),
                audio_frames: VecDeque::new(),
            }),
            cv: Condvar::new(),
        });
        let mut decoders = Vec::new();
        let mut ok = true;
        if let Some((_, dec, surface)) = s.video {
            let sh = shared.clone();
            match spawn("AvPlayerVideoDecoder", move || decoder(&sh, dec, Output::Video(surface))) {
                Ok(h) => decoders.push(h),
                Err(_) => ok = false,
            }
        }
        if let Some((_, dec)) = s.audio {
            let sh = shared.clone();
            match spawn("AvPlayerAudioDecoder", move || decoder(&sh, dec, Output::Audio)) {
                Ok(h) => decoders.push(h),
                Err(_) => ok = false,
            }
        }
        let mut pipeline = Self { shared: shared.clone(), demux: None, decoders };
        if !ok {
            pipeline.stop_threads();
            let (v, a) = pipeline.take_buffers();
            return Err((media, v, a));
        }
        let period = s.loop_period;
        // The demuxer gives the media back when it ends; a panic loses it
        // (the source then fails to restart) but never drops it elsewhere.
        let media = Arc::new(Mutex::new(Some(media)));
        let m2 = media.clone();
        match spawn("AvPlayerDemuxer", move || {
            let mut slot = m2.lock().unwrap_or_else(|e| e.into_inner());
            let mut m = slot.take()?;
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                demux(&shared, &mut m, video_id, audio_id, period)
            }));
            if r.is_err() {
                log::error!("[avplayer] demuxer panicked");
                shared.fail();
            }
            Some(m)
        }) {
            Ok(h) => {
                pipeline.demux = Some(h);
                Ok(pipeline)
            }
            Err(_) => {
                pipeline.stop_threads();
                let m = media.lock().unwrap_or_else(|e| e.into_inner()).take();
                let (v, a) = pipeline.take_buffers();
                Err((m.expect("media not handed to a demuxer"), v, a))
            }
        }
    }

    pub fn lock(&self) -> MutexGuard<'_, Pipe> {
        self.shared.lock()
    }

    /// Wake workers after returning buffers to a pool.
    pub fn notify(&self) {
        self.shared.cv.notify_all();
    }

    fn stop_threads(&mut self) -> Option<Media> {
        self.shared.lock().stop = true;
        self.notify();
        let media = self.demux.take().and_then(|h| h.join().ok().flatten());
        for h in self.decoders.drain(..) {
            let _ = h.join();
        }
        media
    }

    /// Stop all workers; returns the media and every output buffer the
    /// pipeline held (pools and undelivered frames).
    pub fn stop(mut self) -> (Option<Media>, Vec<GuestBuf>, Vec<GuestBuf>) {
        let media = self.stop_threads();
        let (video, audio) = self.take_buffers();
        (media, video, audio)
    }

    fn take_buffers(&self) -> (Vec<GuestBuf>, Vec<GuestBuf>) {
        let mut p = self.shared.lock();
        let mut video: Vec<GuestBuf> = std::mem::take(&mut p.video_pool);
        video.extend(p.video_frames.drain(..).map(|f| f.buf));
        let mut audio: Vec<GuestBuf> = std::mem::take(&mut p.audio_pool);
        audio.extend(p.audio_frames.drain(..).map(|f| f.buf));
        p.video_packets = PacketQueue::default();
        p.audio_packets = PacketQueue::default();
        (video, audio)
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        if self.demux.is_some() || !self.decoders.is_empty() {
            self.stop_threads();
        }
    }
}

/// Keep process-directed timer/IO signals on the emulator's own threads.
fn block_async_signals() {
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        for s in [libc::SIGALRM, libc::SIGVTALRM, libc::SIGPROF, libc::SIGUSR1, libc::SIGUSR2, libc::SIGCHLD, libc::SIGPIPE] {
            libc::sigaddset(&mut set, s);
        }
        libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
    }
}

// ---------------------------------------------------------------------------
// Demuxer
// ---------------------------------------------------------------------------

fn queue(p: &mut Pipe, video: bool) -> &mut PacketQueue {
    if video {
        &mut p.video_packets
    } else {
        &mut p.audio_packets
    }
}

/// Queue a packet (None = loop marker); false when stopping.
fn push(sh: &Shared, video: bool, packet: Option<Packet>, offset: u64) -> bool {
    let size = packet.as_ref().map_or(0, |p| p.size);
    let mut p = sh.lock();
    loop {
        if p.stop {
            return false;
        }
        let cap = p.capacity;
        let q = queue(&mut p, video);
        if q.bytes + size <= cap || (q.q.is_empty() && q.bytes == 0) {
            q.bytes += size;
            q.q.push_back((packet, offset));
            drop(p);
            sh.cv.notify_all();
            return true;
        }
        p = sh.wait(p);
    }
}

fn demux(sh: &Shared, media: &mut Media, video: Option<usize>, audio: Option<usize>, period: u64) {
    let (mut video_offset, mut audio_offset) = (0u64, 0u64);
    loop {
        let looping = {
            let p = sh.lock();
            if p.stop {
                break;
            }
            p.looping
        };
        let packet = match media.read() {
            Ok(p) => p,
            Err(r) => {
                if r == AVERROR_EOF && looping {
                    video_offset += period;
                    audio_offset += period;
                    if (video.is_some() && !push(sh, true, None, video_offset))
                        || (audio.is_some() && !push(sh, false, None, audio_offset))
                    {
                        break;
                    }
                    if media.seek(video.or(audio), 0) < 0 {
                        break;
                    }
                    continue;
                }
                if r != AVERROR_EOF && !sh.lock().stop {
                    log::warn!("[avplayer] av_read_frame failed: {r}");
                    sh.fail();
                }
                break;
            }
        };
        let s = packet.stream;
        if video == Some(s) {
            if !push(sh, true, Some(packet), video_offset) {
                break;
            }
        } else if audio == Some(s) && !push(sh, false, Some(packet), audio_offset) {
            break;
        }
    }
    sh.lock().demux_eof = true;
    sh.cv.notify_all();
}

// ---------------------------------------------------------------------------
// Decoders
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Output {
    Video(VideoSurface),
    Audio,
}

impl Output {
    fn video(&self) -> bool {
        matches!(self, Output::Video(_))
    }
}

enum Item {
    Packet(Option<Packet>, u64),
    /// Nothing more will come (true = the demuxer finished, false = stop).
    End(bool),
}

fn pop_packet(sh: &Shared, video: bool) -> Item {
    let mut p = sh.lock();
    loop {
        if p.stop {
            return Item::End(false);
        }
        let eof = p.demux_eof;
        let q = queue(&mut p, video);
        if let Some((packet, offset)) = q.q.pop_front() {
            q.bytes -= packet.as_ref().map_or(0, |p| p.size);
            drop(p);
            sh.cv.notify_all();
            return Item::Packet(packet, offset);
        }
        if eof {
            return Item::End(true);
        }
        p = sh.wait(p);
    }
}

fn stopping(sh: &Shared) -> bool {
    sh.lock().stop
}

fn decoder(sh: &Shared, mut dec: StreamDecoder, out: Output) {
    let mut offset = 0u64;
    loop {
        match pop_packet(sh, out.video()) {
            Item::End(eof) => {
                if eof && !decode_packet(sh, &mut dec, None, offset, out) && !stopping(sh) {
                    sh.fail();
                }
                break;
            }
            Item::Packet(None, new_offset) => {
                if !decode_packet(sh, &mut dec, None, offset, out) {
                    if !stopping(sh) {
                        sh.fail();
                    }
                    break;
                }
                dec.flush();
                offset = new_offset;
            }
            Item::Packet(Some(packet), new_offset) => {
                offset = new_offset;
                if !decode_packet(sh, &mut dec, Some(&packet), offset, out) {
                    if !stopping(sh) {
                        sh.fail();
                    }
                    break;
                }
            }
        }
    }
    let mut p = sh.lock();
    if out.video() {
        p.video_done = true;
    } else {
        p.audio_done = true;
    }
    drop(p);
    sh.cv.notify_all();
}

fn decode_packet(sh: &Shared, dec: &mut StreamDecoder, packet: Option<&Packet>, offset: u64, out: Output) -> bool {
    let mut r = dec.send(packet);
    if r == AVERROR_EAGAIN {
        if !receive_frames(sh, dec, offset, out) {
            return false;
        }
        r = dec.send(packet);
    }
    if r == AVERROR_EOF {
        return true;
    }
    if r < 0 {
        log::warn!("[avplayer] avcodec_send_packet ({}) failed: {r}", if out.video() { "video" } else { "audio" });
        return false;
    }
    receive_frames(sh, dec, offset, out)
}

/// Take a free output buffer, waiting for the guest to retire one.
fn pop_buffer(sh: &Shared, video: bool) -> Option<GuestBuf> {
    let mut p = sh.lock();
    loop {
        if p.stop {
            return None;
        }
        let pool = if video { &mut p.video_pool } else { &mut p.audio_pool };
        if let Some(b) = pool.pop() {
            return Some(b);
        }
        p = sh.wait(p);
    }
}

fn give_back(sh: &Shared, video: bool, buf: GuestBuf) {
    let mut p = sh.lock();
    if video {
        p.video_pool.push(buf);
    } else {
        p.audio_pool.push(buf);
    }
}

fn push_frame(sh: &Shared, frame: ReadyFrame) {
    let mut p = sh.lock();
    match frame.kind {
        FrameKind::Video { .. } => p.video_frames.push_back(frame),
        FrameKind::Audio { .. } => p.audio_frames.push_back(frame),
    }
    drop(p);
    sh.cv.notify_all();
}

fn receive_frames(sh: &Shared, dec: &mut StreamDecoder, offset: u64, out: Output) -> bool {
    let mut meta = FrameMeta::default();
    while !stopping(sh) {
        let r = dec.receive(&mut meta);
        if r == AVERROR_EAGAIN || r == AVERROR_EOF {
            return true;
        }
        if r < 0 {
            log::warn!("[avplayer] avcodec_receive_frame failed: {r}");
            return false;
        }
        let ok = match out {
            Output::Video(surface) => prepare_video(sh, dec, &meta, surface, offset),
            Output::Audio => prepare_audio(sh, dec, &meta, offset),
        };
        if !ok {
            return false;
        }
    }
    false
}

fn prepare_video(sh: &Shared, dec: &mut StreamDecoder, m: &FrameMeta, s: VideoSurface, offset: u64) -> bool {
    let Some(buf) = pop_buffer(sh, true) else { return false };
    let fits = m.width > 0
        && m.height > 0
        && m.width <= s.pitch
        && m.height <= s.rows
        && s.bytes() <= buf.size as u64
        && ps5vm::ranges::accessible(buf.addr, s.bytes(), true);
    // SAFETY: the buffer is mapped and at least `s.bytes()` long; the guest
    // does not own it while it sits in the free pool.
    if !fits || !unsafe { dec.write_nv12(buf.addr as *mut u8, s.pitch, s.rows, s.full_range) } {
        give_back(sh, true, buf);
        return false;
    }
    let crop = [m.crop[0], m.crop[1] + (s.pitch - m.width), m.crop[2], m.crop[3] + (s.rows - m.height)];
    push_frame(sh, ReadyFrame { buf, ts: m.ts_ms, offset, kind: FrameKind::Video { sar: m.sar, crop } });
    true
}

/// Convert the frame to s16 and hand it out; a frame larger than one output
/// buffer spans several consecutive buffers.
fn prepare_audio(sh: &Shared, dec: &mut StreamDecoder, m: &FrameMeta, offset: u64) -> bool {
    let channels = m.channels.max(1);
    let Some(pcm) = dec.s16(channels) else { return false };
    let samples = pcm.len() / channels as usize;
    let mut done = 0usize;
    while done < samples {
        let Some(buf) = pop_buffer(sh, false) else { return false };
        let per_buf = buf.size as usize / (channels as usize * 2);
        let take = per_buf.min(samples - done);
        let bytes = take * channels as usize * 2;
        if take == 0 || !ps5vm::ranges::accessible(buf.addr, bytes as u64, true) {
            give_back(sh, false, buf);
            return false;
        }
        // SAFETY: mapped and at least `bytes` long (checked above).
        unsafe {
            std::ptr::copy_nonoverlapping(
                pcm[done * channels as usize..].as_ptr() as *const u8,
                buf.addr as *mut u8,
                bytes,
            )
        };
        let rate = m.sample_rate.max(1) as u64;
        push_frame(
            sh,
            ReadyFrame {
                buf,
                ts: m.ts_ms + done as u64 * 1000 / rate,
                offset,
                kind: FrameKind::Audio { channels: channels as u16, sample_rate: m.sample_rate, size: bytes as u32 },
            },
        );
        done += take;
    }
    true
}
