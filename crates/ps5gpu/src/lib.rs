// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Host GPU side of the emulator: the presentation window and Vulkan device.
//!
//! A single render thread owns the winit event loop, the window ("rps5e")
//! and a Vulkan 1.3 device/swapchain. Guest display buffers are handed over
//! with [`Presenter::present_guest_surface`]. A buffer the GPU back-end
//! rendered is copied on the GPU into a presentation frame (a snapshot of
//! the image cache entry, [`backend::capture_display`]) and blitted into the
//! swapchain image: no CPU copy. Only a buffer the CPU wrote is read,
//! detiled ([`tiling`]) and converted ([`format`]) on the calling thread and
//! uploaded through a staging buffer.
//!
//! # Pacing
//!
//! Presentation is decoupled from the guest, like a display controller
//! scanning out at vblank: a present request is complete (the guest may
//! reuse its buffer and sees the flip done) once its `not_before` time — the
//! emulated vblank the flip was scheduled for — has passed. The frame itself
//! is already owned by the presenter (GPU snapshot or converted pixels).
//! The render thread shows the newest due frame on each wake-up and skips
//! older ones it fell behind on, so a slow or blocking host swapchain
//! (compositor throttling, FIFO waits) never stalls the guest. The default
//! present mode is mailbox (`RPS5E_PRESENT_MODE=fifo|mailbox|immediate`).
//!
//! Headless mode (`RPS5E_HEADLESS=1`, or automatically when no display or
//! Vulkan device is available) completes requests the same way without
//! showing anything.
//!
//! All entry points are thread-safe.

pub use ash;
pub use egui;

pub mod backend;
pub mod buffers;
pub mod capture;
pub mod clock;
pub mod dcc;
pub mod copy_kernels;
pub mod fill_kernels;
pub mod device;
pub mod display;
pub mod images;
pub mod format;
pub mod gc;
pub mod fxhash;
pub mod hdr;
pub mod memory;
pub mod menu;
pub mod page_table;
pub mod pipeline;
pub mod render;
pub mod scheduler;
pub mod slice_shadow;
pub mod slices;
pub mod shader;
pub mod suballoc;
pub mod tiler;
pub mod input;
pub mod osd;
pub mod perf;
pub mod tiling;
pub mod tiling_layout;
mod tiling_equations;
mod vulkan;

/// The application's name in window titles.
pub const APP_TITLE: &str = "RustPS5Emu";
/// Wayland app_id / X11 class of every window; matches `rps5e.desktop`.
pub const APP_ID: &str = "rps5e";
/// The window icon: the logo as 64x64 RGBA8 pixels.
pub const WINDOW_ICON_RGBA: &[u8] = include_bytes!("../../../assets/icons/rps5e-64.rgba");
/// Width and height of [`WINDOW_ICON_RGBA`].
pub const WINDOW_ICON_SIZE: u32 = 64;
#[cfg(test)]
mod testutil;

pub use input::{set_input_sink, InputEvent};
pub use vulkan::request_screenshot;

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// VideoOut tiling mode 0: 64 KiB `R_X` render-target tiling.
pub const TILING_RENDER_TARGET: u32 = 0;
/// VideoOut tiling mode 1: linear rows of `pitch_pixels` pixels.
pub const TILING_LINEAR: u32 = 1;

/// Maximum frames waiting for the render thread; older ones are discarded.
pub const MAX_PENDING_FRAMES: usize = 4;

/// A guest display surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceDesc {
    pub width: u32,
    pub height: u32,
    pub pitch_pixels: u32,
    /// VideoOut pixel format code (legacy 32-bit or 64-bit format2).
    pub pixel_format: u64,
    /// VideoOut tiling mode ([`TILING_RENDER_TARGET`] or [`TILING_LINEAR`]).
    pub tiling: u32,
}

/// Identifies one present request. Ticket 0 is never issued.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PresentTicket(pub u64);

/// What happened to a present request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentOutcome {
    Pending,
    Presented,
    Discarded,
}

/// Counters for diagnostics and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PresenterStats {
    pub submitted: u64,
    pub presented: u64,
    pub discarded: u64,
    pub headless: bool,
    pub started: bool,
    /// Presents that showed a GPU-rendered image.
    pub gpu_frames: u64,
    /// Frames the window showed.
    pub shown: u64,
    /// Due frames the window skipped for a newer one.
    pub skipped: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    NotStarted,
    /// Render thread is starting up or running a window.
    Window,
    Headless,
}

/// A converted frame waiting for presentation.
pub(crate) struct Frame {
    pub ticket: u64,
    pub width: u32,
    pub height: u32,
    /// BGRA8, tightly packed (empty for GPU frames).
    pub pixels: Arc<Vec<u8>>,
    /// A frame the GPU rendered (shown instead of `pixels`).
    pub gpu: Option<backend::GpuFrame>,
    pub not_before: Option<Instant>,
    /// The colour space the frame is in: `Some(true)` HDR10 (a BT.2100 PQ
    /// display buffer), `Some(false)` SDR, `None` any (a black frame keeps
    /// the window's current one).
    pub hdr: Option<bool>,
}

pub(crate) struct State {
    mode: Mode,
    force_headless: bool,
    requested: (u32, u32),
    next_ticket: u64,
    /// Frames for the render thread, in submission order.
    pub(crate) queue: VecDeque<Frame>,
    /// Tickets still waiting for their `not_before` (the emulated vblank).
    due: HashMap<u64, Option<Instant>>,
    pending: HashSet<u64>,
    discarded: HashSet<u64>,
    discarded_order: VecDeque<u64>,
    stats: PresenterStats,
    pub(crate) waker: Option<Box<dyn Fn() + Send>>,
}

/// The process-wide presenter. Obtain it with [`presenter`].
pub struct Presenter {
    pub(crate) state: Mutex<State>,
    pub(crate) cv: Condvar,
}

static PRESENTER: OnceLock<Presenter> = OnceLock::new();

/// The process-wide presenter singleton.
pub fn presenter() -> &'static Presenter {
    PRESENTER.get_or_init(Presenter::new)
}

impl Presenter {
    fn new() -> Self {
        Presenter {
            state: Mutex::new(State {
                mode: Mode::NotStarted,
                force_headless: false,
                requested: (1920, 1080),
                next_ticket: 1,
                queue: VecDeque::new(),
                due: HashMap::new(),
                pending: HashSet::new(),
                discarded: HashSet::new(),
                discarded_order: VecDeque::new(),
                stats: PresenterStats::default(),
                waker: None,
            }),
            cv: Condvar::new(),
        }
    }
}

fn env_headless() -> bool {
    std::env::var("RPS5E_HEADLESS").is_ok_and(|v| !v.is_empty() && v != "0")
}

fn display_available() -> bool {
    // A Windows session always has a desktop.
    cfg!(windows) || std::env::var_os("WAYLAND_DISPLAY").is_some() || std::env::var_os("DISPLAY").is_some()
}

impl Presenter {
    pub(crate) fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Run without a window regardless of the environment. Only effective
    /// before the presenter has started.
    pub fn force_headless(&self) {
        let mut st = self.lock();
        st.force_headless = true;
        if st.mode == Mode::NotStarted {
            st.mode = Mode::Headless;
            st.stats.headless = true;
            st.stats.started = true;
            display::note_hdr_surface(false);
        }
    }

    /// Start the presenter (window + Vulkan on the render thread, or
    /// headless) sized for a `width` x `height` display. Idempotent.
    pub fn ensure_started(&'static self, width: u32, height: u32) {
        let mut st = self.lock();
        if st.mode != Mode::NotStarted {
            return;
        }
        st.requested = (width.max(1), height.max(1));
        st.stats.started = true;
        if st.force_headless || env_headless() || !display_available() {
            log::info!("ps5gpu: headless presenter ({}x{})", width, height);
            st.mode = Mode::Headless;
            st.stats.headless = true;
            display::note_hdr_surface(false);
            return;
        }
        st.mode = Mode::Window;
        drop(st);
        let spawned = std::thread::Builder::new()
            .name("rps5e-render".into())
            .spawn(move || vulkan::render_thread(self, width.max(1), height.max(1)));
        if let Err(e) = spawned {
            log::warn!("ps5gpu: cannot spawn render thread: {e}; running headless");
            self.fall_back_to_headless();
        }
    }

    /// Switch to headless after the window or Vulkan failed.
    pub(crate) fn fall_back_to_headless(&self) {
        let mut st = self.lock();
        st.mode = Mode::Headless;
        st.stats.headless = true;
        st.waker = None;
        st.queue.clear();
        display::note_hdr_surface(false);
        self.cv.notify_all();
    }

    /// True when presents are not shown on a window.
    pub fn is_headless(&self) -> bool {
        self.lock().mode == Mode::Headless
    }

    pub fn stats(&self) -> PresenterStats {
        let mut st = self.lock();
        Self::poll_due(&mut st);
        st.stats
    }

    /// Present a guest display surface at `guest_ptr` as soon as possible.
    pub fn present_guest_surface(&'static self, desc: SurfaceDesc, guest_ptr: u64) -> PresentTicket {
        self.present_guest_surface_at(desc, guest_ptr, None)
    }

    /// Present a guest display surface no earlier than `not_before`.
    /// The surface is read, detiled and converted on the calling thread.
    pub fn present_guest_surface_at(
        &'static self,
        desc: SurfaceDesc,
        guest_ptr: u64,
        not_before: Option<Instant>,
    ) -> PresentTicket {
        // Frame capture (§46): the flip is recorded, and a capture starts or
        // ends once it was presented.
        capture::flip_begin(&desc, guest_ptr, !self.is_headless());
        let ticket = slice_shadow::item(slice_shadow::Kind::Flip, 0, || self.present_guest_surface_inner(desc, guest_ptr, not_before));
        slice_shadow::frame_end();
        capture::flip_end();
        ticket
    }

    fn present_guest_surface_inner(&'static self, desc: SurfaceDesc, guest_ptr: u64, not_before: Option<Instant>) -> PresentTicket {
        // Debug dumps of GPU-rendered frames: write the image back to guest
        // memory, then dump it like a CPU surface.
        if let Some(n) = dump_due() {
            let gpu = backend::writeback_display(&desc, guest_ptr);
            if let Ok(pixels) = read_guest_surface(&desc, guest_ptr) {
                write_dump(n, desc.width, desc.height, &pixels, if gpu { "gpu" } else { "cpu" });
            }
        }
        let hdr = Some(format::is_hdr(desc.pixel_format));
        // A display buffer the GPU rendered is taken from the image cache,
        // headless too: reading it from guest memory would download it and
        // wait for the GPU on every flip, which a window never does (the
        // frame pacing of a headless run would differ from a real one).
        if let Some(frame) = backend::capture_display(&desc, guest_ptr) {
            return self.present_gpu_frame(desc.width, desc.height, frame, not_before, hdr);
        }
        match read_guest_surface(&desc, guest_ptr) {
            Ok(pixels) => self.present_converted(desc.width, desc.height, pixels, not_before, hdr),
            Err(reason) => {
                log_once(&format!("ps5gpu: cannot present surface at {guest_ptr:#x} {desc:?}: {reason}"));
                self.discarded_ticket()
            }
        }
    }

    /// Present an already converted BGRA8 frame (`width * height * 4` bytes).
    /// It is shown in the window's current colour space (black frames).
    pub fn present_bgra8(
        &'static self,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        not_before: Option<Instant>,
    ) -> PresentTicket {
        self.present_converted(width, height, pixels, not_before, None)
    }

    fn present_converted(
        &'static self,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        not_before: Option<Instant>,
        hdr: Option<bool>,
    ) -> PresentTicket {
        if width == 0 || height == 0 || pixels.len() < width as usize * height as usize * 4 {
            return self.discarded_ticket();
        }
        if self.lock().mode == Mode::NotStarted {
            self.ensure_started(width, height);
        }
        self.submit(Frame { ticket: 0, width, height, pixels: Arc::new(pixels), gpu: None, not_before, hdr })
    }

    /// Present a frame the GPU rendered (see [`backend::capture_display`]);
    /// `hdr`: `Some(true)` for a BT.2100 PQ (HDR10) image, `Some(false)`
    /// for SDR, `None` to keep the window's colour space.
    pub fn present_gpu_frame(
        &'static self,
        width: u32,
        height: u32,
        frame: backend::GpuFrame,
        not_before: Option<Instant>,
        hdr: Option<bool>,
    ) -> PresentTicket {
        self.submit(Frame { ticket: 0, width, height, pixels: Arc::new(Vec::new()), gpu: Some(frame), not_before, hdr })
    }

    /// Issue a ticket for `frame`: it completes at its `not_before`; the
    /// window (if any) gets the frame to show.
    fn submit(&self, mut frame: Frame) -> PresentTicket {
        let mut st = self.lock();
        let ticket = st.next_ticket;
        st.next_ticket += 1;
        st.stats.submitted += 1;
        perf::GUEST_FLIPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if frame.gpu.is_some() {
            st.stats.gpu_frames += 1;
        }
        st.pending.insert(ticket);
        st.due.insert(ticket, frame.not_before);
        if st.mode == Mode::Window {
            frame.ticket = ticket;
            st.queue.push_back(frame);
            // The window is far behind: drop the oldest frames.
            while st.queue.len() > MAX_PENDING_FRAMES {
                st.queue.pop_front();
                st.stats.skipped += 1;
            }
            if let Some(w) = &st.waker {
                w();
            }
        }
        Self::poll_due(&mut st);
        drop(st);
        self.cv.notify_all();
        PresentTicket(ticket)
    }

    /// The newest frame due for display; older due frames are skipped.
    pub(crate) fn take_newest_due(&self) -> Option<Frame> {
        let mut st = self.lock();
        let now = Instant::now();
        let mut newest = None;
        while st.queue.front().is_some_and(|f| f.not_before.is_none_or(|t| t <= now)) {
            if newest.replace(st.queue.pop_front().unwrap()).is_some() {
                st.stats.skipped += 1;
            }
        }
        newest
    }

    /// The window showed a frame.
    pub(crate) fn note_shown(&self) {
        self.lock().stats.shown += 1;
    }

    fn discarded_ticket(&self) -> PresentTicket {
        let mut st = self.lock();
        let ticket = st.next_ticket;
        st.next_ticket += 1;
        st.stats.submitted += 1;
        st.pending.insert(ticket);
        Self::finish_locked(&mut st, ticket, PresentOutcome::Discarded);
        PresentTicket(ticket)
    }

    /// Complete every request whose display time has come.
    fn poll_due(st: &mut State) {
        if st.due.is_empty() {
            return;
        }
        let now = Instant::now();
        let ready: Vec<u64> = st.due.iter().filter(|(_, nb)| nb.is_none_or(|t| t <= now)).map(|(&t, _)| t).collect();
        for t in ready {
            st.due.remove(&t);
            Self::finish_locked(st, t, PresentOutcome::Presented);
        }
    }

    pub(crate) fn finish_locked(st: &mut State, ticket: u64, outcome: PresentOutcome) {
        if !st.pending.remove(&ticket) {
            return;
        }
        match outcome {
            PresentOutcome::Presented => st.stats.presented += 1,
            PresentOutcome::Discarded => {
                st.stats.discarded += 1;
                st.discarded.insert(ticket);
                st.discarded_order.push_back(ticket);
                while st.discarded_order.len() > 4096 {
                    let old = st.discarded_order.pop_front().unwrap();
                    st.discarded.remove(&old);
                }
            }
            PresentOutcome::Pending => {}
        }
    }

    /// State of a present request.
    pub fn outcome(&self, ticket: PresentTicket) -> PresentOutcome {
        let mut st = self.lock();
        Self::poll_due(&mut st);
        if ticket.0 == 0 || ticket.0 >= st.next_ticket {
            return PresentOutcome::Presented;
        }
        if st.pending.contains(&ticket.0) {
            PresentOutcome::Pending
        } else if st.discarded.contains(&ticket.0) {
            PresentOutcome::Discarded
        } else {
            PresentOutcome::Presented
        }
    }

    /// True once the request has been presented or discarded (the buffer
    /// is no longer needed by the presenter).
    pub fn is_presented(&self, ticket: PresentTicket) -> bool {
        self.outcome(ticket) != PresentOutcome::Pending
    }

    /// Block until `ticket` is done or `timeout` passes; true when done.
    pub fn wait_presented(&self, ticket: PresentTicket, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if self.is_presented(ticket) {
                return true;
            }
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let st = self.lock();
            let slice = (deadline - now).min(Duration::from_millis(2));
            let _ = self.cv.wait_timeout(st, slice);
        }
    }
}

fn log_once(msg: &str) {
    // Asked per draw by some paths (dummy bindings): a message this thread
    // saw costs a hash and a thread-local lookup (no lock, no allocation).
    std::thread_local! {
        static LOCAL: std::cell::RefCell<crate::fxhash::HashSet<u64>> = std::cell::RefCell::new(Default::default());
    }
    let hash = {
        use std::hash::{BuildHasher, Hasher};
        let mut h = rustc_hash::FxBuildHasher.build_hasher();
        h.write(msg.as_bytes());
        h.finish()
    };
    if LOCAL.with(|l| !l.borrow_mut().insert(hash)) {
        return;
    }
    static SEEN: Mutex<Option<crate::fxhash::HashSet<String>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let seen = g.get_or_insert_with(Default::default);
    if !seen.contains(msg) {
        seen.insert(msg.to_string());
        log::warn!("{msg}");
    }
}

/// Read, detile and convert a guest surface into BGRA8.
pub fn read_guest_surface(desc: &SurfaceDesc, guest_ptr: u64) -> Result<Vec<u8>, String> {
    if desc.width == 0 || desc.height == 0 || desc.width > 16384 || desc.height > 16384 {
        return Err("bad extent".into());
    }
    let layout = format::decode(desc.pixel_format).ok_or("unsupported pixel format")?;
    let bpp = layout.bytes_per_pixel();
    let (mode, pitch) = match desc.tiling {
        TILING_RENDER_TARGET => (tiling::SWIZZLE_64KB_R_X, desc.pitch_pixels.max(desc.width)),
        TILING_LINEAR => (tiling::SWIZZLE_LINEAR, desc.pitch_pixels.max(desc.width)),
        t => return Err(format!("unsupported tiling mode {t}")),
    };
    let params = tiling::DetileParams::new(mode, bpp, pitch).ok_or("unsupported element size")?;
    let size = params.surface_size(desc.height);
    if guest_ptr == 0 || !ps5vm::ranges::accessible(guest_ptr, size as u64, false) {
        return Err(format!("guest range {guest_ptr:#x}+{size:#x} not readable"));
    }
    // Pages GPU work wrote come back in one download instead of one GPU
    // wait per faulting page.
    backend::download_range(guest_ptr, size as u64);
    // SAFETY: guest memory is identity-mapped into this process and the
    // range was validated as mapped and readable above.
    let tiled = unsafe { std::slice::from_raw_parts(guest_ptr as *const u8, size) };
    capture::note_read(memory::global(), guest_ptr, size as u64);
    convert_surface(desc, &params, layout, tiled)
}

/// Detile + convert a surface already held in host memory.
pub fn convert_surface(
    desc: &SurfaceDesc,
    params: &tiling::DetileParams,
    layout: format::Layout,
    tiled: &[u8],
) -> Result<Vec<u8>, String> {
    let bpp = layout.bytes_per_pixel();
    let mut linear = vec![0u8; desc.width as usize * desc.height as usize * bpp];
    if !tiling::detile(params, tiled, &mut linear, desc.width, desc.height) {
        return Err("detile failed".into());
    }
    if layout == format::Layout::Bgra8 {
        for px in linear.chunks_exact_mut(4) {
            px[3] = 0xFF;
        }
        return Ok(linear);
    }
    let mut out = vec![0u8; desc.width as usize * desc.height as usize * 4];
    format::convert_to_bgra8(layout, &linear, &mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screenshots_encode_as_png() {
        let rgb: Vec<u8> = (0..2 * 3 * 3).map(|i| i as u8 * 10).collect();
        let png_bytes = encode_png_rgb(2, 3, &rgb).unwrap();
        let mut reader = png::Decoder::new(std::io::Cursor::new(png_bytes)).read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!((info.width, info.height, info.color_type), (2, 3, png::ColorType::Rgb));
        assert_eq!(&buf[..info.buffer_size()], &rgb[..]);
    }

    #[test]
    fn frame_dumps_scale_by_box_average_and_parse_windows() {
        // 4x2 BGRA: left 2x2 box red 0/200 alternating, right box blue 100.
        let mut px = Vec::new();
        for y in 0..2 {
            for x in 0..4 {
                if x < 2 {
                    px.extend_from_slice(&[0, 0, if (x + y) % 2 == 0 { 0 } else { 200 }, 255]);
                } else {
                    px.extend_from_slice(&[100, 0, 0, 255]);
                }
            }
        }
        let out = ppm_scaled(4, 2, &px, 2);
        let header = b"P6\n2 1\n255\n";
        assert_eq!(&out[..header.len()], header);
        assert_eq!(&out[header.len()..], &[100, 0, 0, 0, 0, 100]);
        // Scale 1 is the plain conversion.
        assert_eq!(ppm_scaled(4, 2, &px, 1).len(), b"P6\n4 2\n255\n".len() + 4 * 2 * 3);
        assert_eq!(parse_window("30-45.5"), Some((30.0, 45.5)));
        assert_eq!(parse_window("45-30"), None);
        assert!(process_seconds() >= 0.0);
    }

    #[test]
    fn headless_present_counts_and_paces() {
        let p = presenter();
        p.force_headless();
        p.ensure_started(64, 32);
        assert!(p.is_headless());
        let before = p.stats();
        let t1 = p.present_bgra8(64, 32, vec![0u8; 64 * 32 * 4], None);
        assert!(p.is_presented(t1));
        assert_eq!(p.outcome(t1), PresentOutcome::Presented);
        let later = Instant::now() + Duration::from_millis(30);
        let t2 = p.present_bgra8(64, 32, vec![0u8; 64 * 32 * 4], Some(later));
        assert!(!p.is_presented(t2));
        assert!(p.wait_presented(t2, Duration::from_secs(1)));
        assert!(Instant::now() >= later);
        let bad = p.present_bgra8(64, 32, vec![0u8; 4], None);
        assert_eq!(p.outcome(bad), PresentOutcome::Discarded);
        let after = p.stats();
        assert!(after.presented >= before.presented + 2);
        assert!(after.discarded > before.discarded);
    }

    #[test]
    fn headless_present_guest_surface_detiles() {
        let p = presenter();
        p.force_headless();
        let (w, h) = (200u32, 130u32);
        let params = tiling::DetileParams::new(tiling::SWIZZLE_64KB_R_X, 4, w).unwrap();
        let size = params.surface_size(h);
        let mut lin = vec![0u8; (w * h * 4) as usize];
        for (i, px) in lin.chunks_exact_mut(4).enumerate() {
            px.copy_from_slice(&[(i % 251) as u8, (i % 7) as u8, (i % 13) as u8, 0]);
        }
        // 64 KiB-aligned host buffer standing in for guest memory.
        let mut backing = vec![0u8; size + 65536];
        let base = (backing.as_mut_ptr() as usize + 65535) & !65535;
        // SAFETY: base..base+size lies inside `backing`.
        let surf = unsafe { std::slice::from_raw_parts_mut(base as *mut u8, size) };
        tiling::tile(&params, &lin, surf, w, h);
        ps5vm::ranges::track(base as u64, size as u64, ps5vm::MemProtect { read: true, write: true, exec: false });
        let desc = SurfaceDesc {
            width: w,
            height: h,
            pitch_pixels: w,
            pixel_format: format::FMT2_R8G8B8A8_SRGB,
            tiling: TILING_RENDER_TARGET,
        };
        let out = read_guest_surface(&desc, base as u64).unwrap();
        for i in 0..(w * h) as usize {
            assert_eq!(out[i * 4], lin[i * 4 + 2]);
            assert_eq!(out[i * 4 + 1], lin[i * 4 + 1]);
            assert_eq!(out[i * 4 + 2], lin[i * 4]);
            assert_eq!(out[i * 4 + 3], 0xFF);
        }
        let t = p.present_guest_surface(desc, base as u64);
        assert_eq!(p.outcome(t), PresentOutcome::Presented);
        // Unmapped memory is rejected and discarded.
        let t = p.present_guest_surface(desc, 0x10);
        assert_eq!(p.outcome(t), PresentOutcome::Discarded);
        ps5vm::ranges::untrack(base as u64, size as u64);
    }

    /// A window that is not showing frames never holds up the guest: requests
    /// complete at their display time and the window gets only the newest.
    #[test]
    fn window_requests_complete_without_the_window() {
        let p = Presenter::new();
        p.lock().mode = Mode::Window;
        let t1 = p.queue_frame(4, 4);
        let t2 = p.queue_frame(4, 4);
        assert!(p.is_presented(t1) && p.is_presented(t2));
        let later = Instant::now() + Duration::from_millis(20);
        let t3 = p.submit(Frame { ticket: 0, width: 4, height: 4, pixels: Arc::new(vec![0; 64]), gpu: None, not_before: Some(later), hdr: None });
        assert!(!p.is_presented(t3));
        // Only the newest due frame is taken; the other one is skipped.
        assert_eq!(p.take_newest_due().map(|f| f.ticket), Some(t2.0));
        assert!(p.take_newest_due().is_none());
        assert!(p.wait_presented(t3, Duration::from_secs(1)));
        assert_eq!(p.take_newest_due().map(|f| f.ticket), Some(t3.0));
        let s = p.stats();
        assert_eq!((s.submitted, s.presented, s.skipped), (3, 3, 1));
        // The window falls far behind: the oldest frames are dropped.
        for _ in 0..MAX_PENDING_FRAMES + 2 {
            p.queue_frame(4, 4);
        }
        assert_eq!(p.lock().queue.len(), MAX_PENDING_FRAMES);
    }

    impl Presenter {
        fn queue_frame(&self, w: u32, h: u32) -> PresentTicket {
            self.submit(Frame { ticket: 0, width: w, height: h, pixels: Arc::new(vec![0; (w * h * 4) as usize]), gpu: None, not_before: None, hdr: None })
        }
    }

    /// Manual smoke test: opens the real window and shows a moving gradient
    /// for ~2 s. Run with `cargo test -p ps5gpu --release -- --ignored`.
    #[test]
    #[ignore]
    fn window_smoke_gradient() {
        let _ = env_logger::builder().is_test(true).try_init();
        let p = presenter();
        let (w, h) = (1280u32, 720u32);
        p.ensure_started(w, h);
        let secs = std::env::var("RPS5E_SMOKE_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        let start = Instant::now();
        let mut frames = 0u32;
        let mut last = PresentTicket(0);
        while start.elapsed() < Duration::from_secs(secs) {
            let phase = frames * 4;
            let mut px = vec![0u8; (w * h * 4) as usize];
            for yy in 0..h {
                for xx in 0..w {
                    let i = ((yy * w + xx) * 4) as usize;
                    px[i] = ((xx + phase) * 255 / w) as u8;
                    px[i + 1] = (yy * 255 / h) as u8;
                    px[i + 2] = 255 - ((xx * 255) / w) as u8;
                    px[i + 3] = 255;
                }
            }
            last = p.present_bgra8(w, h, px, None);
            p.wait_presented(last, Duration::from_millis(100));
            frames += 1;
        }
        p.wait_presented(last, Duration::from_secs(1));
        let s = p.stats();
        eprintln!("smoke: headless={} submitted={} presented={} discarded={}", s.headless, s.submitted, s.presented, s.discarded);
        assert!(s.presented > 0);
    }
}

/// Debug aid: with `RPS5E_DUMP_FRAMES=<dir>`, every `RPS5E_DUMP_EVERY`-th
/// (default 60) presented guest surface is written as
/// `<dir>/frame_NNNNN_{gpu,cpu}.ppm` (`gpu`: the GPU-rendered image written
/// back to guest memory; `cpu`: guest memory as the CPU sees it).
/// `RPS5E_DUMP_WINDOW=<from>-<to>` dumps only presents between those
/// seconds of process time (consecutive frames for the glitch scan,
/// `scripts/glitch_scan.py`); `RPS5E_DUMP_SCALE=<n>` writes every n-th
/// pixel's n x n box average (a 4K frame at 4 is 960x540).
struct DumpConfig {
    dir: std::path::PathBuf,
    every: u64,
    window: Option<(f64, f64)>,
    scale: u32,
}

fn dump_config() -> Option<&'static DumpConfig> {
    static CONFIG: OnceLock<Option<DumpConfig>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let dir = std::env::var_os("RPS5E_DUMP_FRAMES")?;
            let every = std::env::var("RPS5E_DUMP_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(60u64).max(1);
            let window = std::env::var("RPS5E_DUMP_WINDOW").ok().and_then(|v| parse_window(&v));
            let scale = std::env::var("RPS5E_DUMP_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(1u32).clamp(1, 16);
            std::fs::create_dir_all(&dir).ok()?;
            Some(DumpConfig { dir: dir.into(), every, window, scale })
        })
        .as_ref()
}

/// `"<from>-<to>"` seconds.
fn parse_window(v: &str) -> Option<(f64, f64)> {
    let (a, b) = v.split_once('-')?;
    let (a, b) = (a.trim().parse::<f64>().ok()?, b.trim().parse::<f64>().ok()?);
    (a <= b).then_some((a, b))
}

/// Seconds since the process started (`/proc/self/stat` start time).
pub(crate) fn process_seconds() -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    let start = *START.get_or_init(|| {
        let now = Instant::now();
        #[cfg(unix)]
        let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1) as f64;
        #[cfg(windows)]
        let ticks = 100.0;
        let since_boot = std::fs::read_to_string("/proc/uptime").ok().and_then(|u| u.split_whitespace().next()?.parse::<f64>().ok());
        // Field 22 (start time), counted after the parenthesised name.
        let started = std::fs::read_to_string("/proc/self/stat")
            .ok()
            .and_then(|s| s.rsplit_once(')').and_then(|(_, rest)| rest.split_whitespace().nth(19)?.parse::<f64>().ok()));
        match (since_boot, started) {
            (Some(up), Some(st)) if up >= st / ticks => now.checked_sub(Duration::from_secs_f64(up - st / ticks)).unwrap_or(now),
            _ => now,
        }
    });
    start.elapsed().as_secs_f64()
}

/// The present counter when this present is to be dumped.
fn dump_due() -> Option<u64> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let cfg = dump_config()?;
    if let Some((from, to)) = cfg.window {
        let t = process_seconds();
        if t < from || t > to {
            return None;
        }
    }
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    (n % cfg.every == 0).then_some(n)
}

fn write_dump(n: u64, width: u32, height: u32, bgra: &[u8], tag: &str) {
    let Some(cfg) = dump_config() else { return };
    let out = ppm_scaled(width, height, bgra, cfg.scale);
    let _ = std::fs::write(cfg.dir.join(format!("frame_{n:05}_{tag}.ppm")), out);
}

/// An 8-bit RGB image as PNG file bytes.
pub(crate) fn encode_png_rgb(width: u32, height: u32, rgb: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, width, height);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().map_err(|e| e.to_string())?;
        w.write_image_data(rgb).map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// A binary PPM of a BGRA8 image, box-averaged by `scale`.
pub(crate) fn ppm_scaled(width: u32, height: u32, bgra: &[u8], scale: u32) -> Vec<u8> {
    let s = scale.max(1) as usize;
    let (w, h) = (width as usize, height as usize);
    let (ow, oh) = (w / s, h / s);
    let (ow, oh, s) = if ow == 0 || oh == 0 { (w, h, 1) } else { (ow, oh, s) };
    let mut out = format!("P6\n{ow} {oh}\n255\n").into_bytes();
    out.reserve(ow * oh * 3);
    if bgra.len() < w * h * 4 {
        out.resize(out.len() + ow * oh * 3, 0);
        return out;
    }
    let area = (s * s) as u32;
    for oy in 0..oh {
        for ox in 0..ow {
            let mut sum = [0u32; 3];
            for y in oy * s..oy * s + s {
                let row = &bgra[(y * w + ox * s) * 4..(y * w + ox * s + s) * 4];
                for px in row.chunks_exact(4) {
                    sum[0] += px[2] as u32;
                    sum[1] += px[1] as u32;
                    sum[2] += px[0] as u32;
                }
            }
            out.extend_from_slice(&[(sum[0] / area) as u8, (sum[1] / area) as u8, (sum[2] / area) as u8]);
        }
    }
    out
}
