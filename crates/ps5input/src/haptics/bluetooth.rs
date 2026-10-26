// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! DualSense audio over Bluetooth: raw HID output report 0x39 carrying the
//! haptic actuator waveform (3 kHz, 8-bit) and the speaker as Opus packets,
//! sent by one paced sender thread.
//!
//! Each report plays two 32-frame stereo haptic blocks at 3 kHz
//! (21.333 ms); two 480-frame Opus packets occupy the same slot, so the
//! speaker is resampled to 45 kHz before encoding at 48 kHz to compensate.

use super::opus::{Encoder, PACKET_BYTES};
use super::resample::Resampler;
use super::{Backend, HidDevice, HidInfo};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

pub const PERIOD_NS: u64 = 64_000_000_000 / 3000;
const HAPTIC_RATE: u32 = 3000;
const HAPTIC_REPORT_FLOATS: usize = 2 * 32 * 2;
const SPEAKER_RATE: u32 = 45000;
const OPUS_FRAMES: usize = 480;
pub const REPORT_BYTES: usize = 547;
const MAX_QUEUE_MS: u32 = 80;
const COALESCE_NS: u64 = 1_000_000;
const SPEAKER_FLOATS: usize = 2 * OPUS_FRAMES * 2;

/// Bluetooth address from a serial ("aa-bb-cc-dd-ee-ff", any case, ':' or
/// '-' separated): 12 lower-case hex digits, empty when malformed.
pub fn address(serial: &str) -> String {
    let mut out = String::new();
    for c in serial.chars() {
        match c {
            '0'..='9' | 'a'..='f' => out.push(c),
            'A'..='F' => out.push(c.to_ascii_lowercase()),
            ':' | '-' => {}
            _ => return String::new(),
        }
    }
    if out.len() == 12 { out } else { String::new() }
}

/// CRC-32 (IEEE, reflected), as zlib.
pub fn crc32(mut crc: u32, data: &[u8]) -> u32 {
    crc = !crc;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

/// Append the report CRC (over the 0xA2 output header and the report).
pub fn finish_report(report: &mut [u8; REPORT_BYTES]) {
    let crc = crc32(crc32(0, &[0xa2]), &report[..REPORT_BYTES - 4]);
    report[REPORT_BYTES - 4..].copy_from_slice(&crc.to_le_bytes());
}

pub struct Session {
    hid: Box<dyn HidDevice>,
    encoder: Mutex<Option<Box<dyn Encoder>>>,
    io: Mutex<()>,
    active: AtomicBool,
    failed: AtomicBool,
    generation: AtomicU64,
    sched: Mutex<Sched>,
}

#[derive(Default)]
struct Sched {
    streams: Vec<Arc<Stream>>,
    next_send: u64,
    pending_deadline: u64,
    sequence: u8,
    counter: u8,
}

impl Session {
    fn sched(&self) -> MutexGuard<'_, Sched> {
        self.sched.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn create_encoder(&self, backend: &dyn Backend) -> bool {
        let mut e = self.encoder.lock().unwrap_or_else(|e| e.into_inner());
        if e.is_none() {
            *e = backend.create_encoder();
        }
        e.is_some()
    }
}

/// One port's stream on a controller session.
pub struct Stream {
    session: Arc<Session>,
    audio: Mutex<Resampler>,
    freq: u32,
    speaker: bool,
}

impl Stream {
    fn audio(&self) -> MutexGuard<'_, Resampler> {
        self.audio.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// A report prepared for sending.
pub struct PendingReport {
    pub session: Arc<Session>,
    pub generation: u64,
    pub data: [u8; REPORT_BYTES],
    pub audio: Vec<f32>,
    pub speaker: bool,
}

pub type Tick = Arc<dyn Fn() + Send + Sync>;

#[derive(Default)]
struct Registry {
    sessions: HashMap<i32, Arc<Session>>,
    thread: Option<Option<JoinHandle<()>>>,
    stopping: bool,
    tick: Option<Tick>,
}

pub struct Bluetooth<B: Backend> {
    backend: Arc<B>,
    inner: Mutex<Registry>,
    wake: Condvar,
}

fn mix(audio: &mut Resampler, mixed: &mut [f32]) {
    let mut pcm = vec![0.0f32; mixed.len()];
    let n = audio.get(&mut pcm);
    for (m, p) in mixed.iter_mut().zip(&pcm[..n]) {
        *m += p;
    }
}

impl<B: Backend> Bluetooth<B> {
    pub fn new(backend: Arc<B>) -> Arc<Self> {
        Arc::new(Self { backend, inner: Mutex::new(Registry::default()), wake: Condvar::new() })
    }

    fn inner(&self) -> MutexGuard<'_, Registry> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// HID device of a wireless DualSense, matched by Bluetooth address;
    /// ambiguous matches are not opened.
    fn open_device(&self, controller: i32) -> Option<Box<dyn HidDevice>> {
        let (vendor, product, serial) = self.backend.wireless_identity(controller)?;
        let addr = address(&serial);
        if vendor != 0x054c || (product != 0x0ce6 && product != 0x0df2) || addr.is_empty() {
            return None;
        }
        let mut path: Option<String> = None;
        for d in self.backend.hid_enumerate(vendor, product) {
            let HidInfo { path: p, serial, bluetooth, usage_page, usage } = d;
            if bluetooth && !p.is_empty() && address(&serial) == addr && (usage_page == 0 || (usage_page == 1 && usage == 5)) {
                if path.is_some() {
                    return None;
                }
                path = Some(p);
            }
        }
        self.backend.hid_open(&path?)
    }

    /// Open a stream of `freq` Hz stereo for the controller.
    pub fn open(self: &Arc<Self>, freq: u32, speaker: bool, controller: i32, tick: Tick) -> Option<Arc<Stream>> {
        if freq == 0 || freq > 192_000 {
            return None;
        }
        let mut inner = self.inner();
        if inner.stopping {
            return None;
        }
        let session = match inner.sessions.get(&controller) {
            Some(s) if !s.failed.load(Ordering::Acquire) => s.clone(),
            _ => Arc::new(Session {
                hid: self.open_device(controller)?,
                encoder: Mutex::new(None),
                io: Mutex::new(()),
                active: AtomicBool::new(true),
                failed: AtomicBool::new(false),
                generation: AtomicU64::new(0),
                sched: Mutex::new(Sched::default()),
            }),
        };
        if speaker && !session.create_encoder(&*self.backend) {
            return None;
        }
        let rate = if speaker { SPEAKER_RATE } else { HAPTIC_RATE };
        if inner.thread.is_none() {
            let me = Arc::downgrade(self);
            let run = Box::new(move || {
                if let Some(me) = me.upgrade() {
                    me.sender();
                }
            });
            inner.thread = Some(self.backend.spawn_sender(run)?);
        }
        inner.tick = Some(tick);
        let stream = Arc::new(Stream { session: session.clone(), audio: Mutex::new(Resampler::new(freq, rate)), freq, speaker });
        {
            let mut sched = session.sched();
            sched.streams.push(stream.clone());
            sched.pending_deadline = 0;
        }
        inner.sessions.insert(controller, session);
        self.wake.notify_one();
        Some(stream)
    }

    /// Queue stereo frames; returns how long the queued audio plays (us).
    pub fn queue(&self, stream: &Stream, stereo: &[f32], frames: u32) -> u64 {
        if frames == 0 || stereo.len() < frames as usize * 2 {
            return 0;
        }
        let _inner = self.inner();
        if !stream.session.active.load(Ordering::Acquire) || stream.session.failed.load(Ordering::Acquire) {
            return 0;
        }
        let max_frames = stream.freq * MAX_QUEUE_MS / 1000;
        let (mut stereo, mut frames) = (stereo, frames);
        if frames > max_frames {
            stereo = &stereo[(frames - max_frames) as usize * 2..];
            frames = max_frames;
        }
        let mut audio = stream.audio();
        let bytes = frames as usize * 8;
        let max_bytes = max_frames as usize * 8;
        if audio.queued_bytes() + bytes > max_bytes {
            // Discard complete old report slots, keeping the newest audio.
            let discard = if stream.speaker { SPEAKER_FLOATS } else { HAPTIC_REPORT_FLOATS };
            let target = max_bytes.saturating_sub(bytes);
            let mut scratch = vec![0.0f32; discard];
            while audio.queued_bytes() > target {
                if audio.get(&mut scratch) == 0 {
                    audio.clear();
                    break;
                }
            }
        }
        let queued = audio.queued_bytes();
        audio.put(&stereo[..frames as usize * 2]);
        drop(audio);
        self.wake.notify_one();
        (queued + bytes) as u64 * 1_000_000 / (stream.freq as u64 * 8)
    }

    pub fn close(&self, stream: &Arc<Stream>) {
        let session = stream.session.clone();
        {
            let mut inner = self.inner();
            let mut sched = session.sched();
            sched.streams.retain(|s| !Arc::ptr_eq(s, stream));
            session.generation.fetch_add(1, Ordering::AcqRel);
            sched.pending_deadline = 0;
            if sched.streams.is_empty() {
                session.active.store(false, Ordering::Release);
                inner.sessions.retain(|_, s| !Arc::ptr_eq(s, &session));
            }
        }
        // Drain a write in flight before the caller changes speaker routing.
        drop(session.io.lock().unwrap_or_else(|e| e.into_inner()));
    }

    /// True once a write to the session failed.
    pub fn failed(stream: &Stream) -> bool {
        stream.session.failed.load(Ordering::Acquire)
    }

    fn prepare_report(sched: &mut Sched, pending: &mut PendingReport, now: u64) -> bool {
        let (mut has_data, mut ready, mut speaker_ready) = (false, false, false);
        for s in &sched.streams {
            let audio = s.audio();
            let available = audio.available_frames() > 0;
            // Both payloads cover 64 / 3000 s of input.
            let frames = (s.freq as u64 * (HAPTIC_REPORT_FLOATS as u64 / 2)).div_ceil(HAPTIC_RATE as u64);
            let full = available && audio.queued_bytes() as u64 >= frames * 8;
            has_data |= available;
            ready |= full;
            speaker_ready |= s.speaker && full;
            pending.speaker |= s.speaker;
        }
        if !has_data {
            sched.pending_deadline = 0;
            return false;
        }
        if !ready || (!speaker_ready && (pending.speaker || sched.next_send == 0)) {
            // Accumulate short blocks for at most one period; a full stream
            // only briefly waits for speaker audio of the same batch.
            let deadline = now + if ready { COALESCE_NS } else { PERIOD_NS };
            sched.pending_deadline =
                if sched.pending_deadline == 0 { deadline } else { sched.pending_deadline.min(deadline) };
            if now < sched.pending_deadline {
                return false;
            }
        }
        sched.pending_deadline = 0;
        let mut haptics = [0.0f32; HAPTIC_REPORT_FLOATS];
        for s in &sched.streams {
            let mut audio = s.audio();
            if s.speaker {
                mix(&mut audio, &mut pending.audio);
            } else {
                mix(&mut audio, &mut haptics);
            }
        }
        let r = &mut pending.data;
        r[0] = 0x39;
        r[1] = sched.sequence << 4;
        sched.sequence = (sched.sequence + 1) & 0x0f;
        r[2] = 0x91;
        r[3] = 6;
        r[4] = 0x7e;
        r[5..9].fill(48);
        sched.counter = sched.counter.wrapping_add(2);
        r[9] = sched.counter;
        r[10] = 0xd2;
        r[11] = 64;
        for (i, &v) in haptics.iter().enumerate() {
            let v = if v.is_finite() { v } else { 0.0 };
            r[12 + i] = (v.clamp(-1.0, 1.0) * 127.0).round() as i32 as u8;
        }
        if pending.speaker {
            r[140] = 0xd3;
            r[141] = PACKET_BYTES as u8;
        }
        true
    }

    fn encode_report(pending: &mut PendingReport) -> bool {
        let s = &pending.session;
        if !s.active.load(Ordering::Acquire) || s.failed.load(Ordering::Acquire) {
            return false;
        }
        if pending.speaker {
            for v in pending.audio.iter_mut() {
                *v = if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 };
            }
            let mut enc = s.encoder.lock().unwrap_or_else(|e| e.into_inner());
            let Some(enc) = enc.as_mut() else {
                s.failed.store(true, Ordering::Release);
                return false;
            };
            for packet in 0..2 {
                let pcm = &pending.audio[packet * OPUS_FRAMES * 2..];
                let out = &mut pending.data[142 + packet * PACKET_BYTES..142 + (packet + 1) * PACKET_BYTES];
                if enc.encode(pcm, OPUS_FRAMES, out) != PACKET_BYTES as i32 {
                    s.failed.store(true, Ordering::Release);
                    return false;
                }
            }
        }
        finish_report(&mut pending.data);
        true
    }

    /// Reports due at `now` (ns), encoded outside the registry lock.
    pub fn prepare(&self, now: u64) -> Vec<PendingReport> {
        let mut pending = Vec::new();
        {
            let inner = self.inner();
            for session in inner.sessions.values() {
                let mut sched = session.sched();
                if session.failed.load(Ordering::Acquire) || now < sched.next_send {
                    continue;
                }
                let mut report = PendingReport {
                    session: session.clone(),
                    generation: session.generation.load(Ordering::Acquire),
                    data: [0; REPORT_BYTES],
                    audio: vec![0.0; SPEAKER_FLOATS],
                    speaker: false,
                };
                if Self::prepare_report(&mut sched, &mut report, now) {
                    pending.push(report);
                    // Regain timing after a late write without bursting;
                    // the cadence starts at the first report.
                    sched.next_send = if sched.next_send == 0 {
                        now + PERIOD_NS
                    } else {
                        (sched.next_send + PERIOD_NS).max(now + PERIOD_NS / 2)
                    };
                }
            }
        }
        pending.retain_mut(Self::encode_report);
        pending
    }

    pub fn send(&self, pending: &[PendingReport]) {
        for report in pending {
            let s = &report.session;
            let _io = s.io.lock().unwrap_or_else(|e| e.into_inner());
            if !s.active.load(Ordering::Acquire)
                || s.failed.load(Ordering::Acquire)
                || report.generation != s.generation.load(Ordering::Acquire)
            {
                continue;
            }
            if s.hid.write(&report.data) != REPORT_BYTES as i32 {
                s.failed.store(true, Ordering::Release);
                log::warn!("haptics: DualSense Bluetooth audio HID write failed");
            }
        }
    }

    fn sender(&self) {
        let mut inner = self.inner();
        while !inner.stopping {
            let tick = inner.tick.clone();
            drop(inner);
            let pending = self.prepare(self.backend.ticks_ns());
            self.send(&pending);
            if let Some(tick) = tick {
                tick();
            }
            inner = self.inner();
            if inner.sessions.is_empty() {
                while !inner.stopping && inner.sessions.is_empty() {
                    inner = self.wake.wait(inner).unwrap_or_else(|e| e.into_inner());
                }
            } else {
                let now = self.backend.ticks_ns();
                // Check rumble expiry at most 10 ms late.
                let mut wait = 10_000_000u64;
                for s in inner.sessions.values() {
                    let sched = s.sched();
                    if sched.next_send > now {
                        wait = wait.min(sched.next_send - now);
                    }
                    if sched.pending_deadline > now {
                        wait = wait.min(sched.pending_deadline - now);
                    }
                }
                inner = self.wake.wait_timeout(inner, Duration::from_nanos(wait)).unwrap_or_else(|e| e.into_inner()).0;
            }
        }
    }

    pub fn shutdown(&self) {
        let thread = {
            let mut inner = self.inner();
            inner.stopping = true;
            for s in inner.sessions.values() {
                s.active.store(false, Ordering::Release);
            }
            self.wake.notify_one();
            inner.thread.take()
        };
        if let Some(Some(t)) = thread {
            let _ = t.join();
        }
        let mut inner = self.inner();
        inner.sessions.clear();
        inner.thread = None;
        inner.tick = None;
        inner.stopping = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_and_crc() {
        assert_eq!(address("AA:BB:CC:DD:EE:01"), "aabbccddee01");
        assert_eq!(address("aa-bb-cc-dd-ee-03"), "aabbccddee03");
        assert_eq!(address("aa-bb-cc-dd-ee"), "");
        assert_eq!(address("zz:bb:cc:dd:ee:01"), "");
        assert_eq!(crc32(0, b"123456789"), 0xCBF4_3926);
        // Incremental CRC equals the one-shot CRC.
        assert_eq!(crc32(crc32(0, b"1234"), b"56789"), 0xCBF4_3926);
        let mut r = [0u8; REPORT_BYTES];
        r[0] = 0x39;
        finish_report(&mut r);
        let mut whole = vec![0xa2];
        whole.extend_from_slice(&r[..REPORT_BYTES - 4]);
        assert_eq!(u32::from_le_bytes(r[REPORT_BYTES - 4..].try_into().unwrap()), crc32(0, &whole));
    }
}
