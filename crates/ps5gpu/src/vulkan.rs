// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Render thread: winit window + Vulkan 1.3 swapchain presentation (mailbox
//! by default; the newest due frame is shown, older ones are skipped).
//!
//! The winit event loop lives on this thread (created with `any_thread`),
//! and is pumped between presents. The swapchain lives on the shared
//! [`GpuDevice`]. Frames arrive through the presenter's queue: CPU frames
//! are copied into a per-frame host-visible staging buffer and uploaded to a
//! per-frame device image; GPU frames are snapshots the back-end copied out
//! of its image cache (the submit waits for their timeline tick). Either is
//! then blitted (letterboxed, linear filter) into the acquired swapchain
//! image and presented.
//!
//! With HDR output on ([`crate::hdr`]) a PQ frame switches the swapchain to
//! an HDR10 format and the frame's code values are blitted unchanged; the
//! overlays drawn over it are converted from sRGB. Black frames keep the
//! current colour space; an SDR frame switches back.

use crate::backend::GpuFrame;
use crate::device::GpuDevice;
use crate::Presenter;
use ash::{khr, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::window::{Window, WindowId};

const MAX_FRAMES_IN_FLIGHT: usize = 2;
const GUEST_IMAGE_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

pub(crate) fn render_thread(p: &'static Presenter, width: u32, height: u32) {
    let mut builder = EventLoop::builder();
    #[cfg(target_os = "linux")]
    {
        winit::platform::x11::EventLoopBuilderExtX11::with_any_thread(&mut builder, true);
        winit::platform::wayland::EventLoopBuilderExtWayland::with_any_thread(&mut builder, true);
    }
    #[cfg(windows)]
    winit::platform::windows::EventLoopBuilderExtWindows::with_any_thread(&mut builder, true);
    let mut event_loop = match builder.build() {
        Ok(e) => e,
        Err(e) => {
            log::warn!("ps5gpu: cannot create the window event loop ({e}); running headless");
            p.fall_back_to_headless();
            return;
        }
    };
    let proxy = event_loop.create_proxy();
    let stop_proxy = proxy.clone();
    p.lock().waker = Some(Box::new(move || {
        let _ = proxy.send_event(());
    }));
    ps5vm::shutdown::on_shutdown(ps5vm::shutdown::ORDER_WINDOW, "window", move || {
        let _ = stop_proxy.send_event(());
        wait_render_stopped(RENDER_STOP_WAIT);
    });

    let mut app = App {
        p,
        size: (width, height),
        window: None,
        vk: None,
        failed: false,
        last: None,
        redraw: false,
        hint: None,
        status: None,
        pace: PaceStats::new(),
        overlay_on: false,
        last_present: Instant::now(),
        menu: None,
        ui_on: false,
        compile: None,
        compile_on: false,
        controls_on: false,
    };
    loop {
        let timeout = app.timeout();
        let status = event_loop.pump_app_events(Some(timeout), &mut app);
        if ps5vm::shutdown::requested() {
            // Process shutdown: close the window and stop presenting; the
            // event loop is not pumped again.
            app.vk = None;
            app.window = None;
            p.fall_back_to_headless();
            set_render_stopped();
            loop {
                std::thread::park();
            }
        }
        if app.failed {
            app.vk = None;
            app.window = None;
            p.fall_back_to_headless();
            set_render_stopped();
            return;
        }
        if let PumpStatus::Exit(code) = status {
            log::info!("ps5gpu: window event loop exited ({code})");
            app.vk = None;
            p.fall_back_to_headless();
            set_render_stopped();
            return;
        }
        crate::input::apply_requests(app.window.as_deref());
        app.render();
    }
}

/// How long shutdown waits for the render thread to close the window.
const RENDER_STOP_WAIT: Duration = Duration::from_millis(500);

static RENDER_STOPPED: (std::sync::Mutex<bool>, std::sync::Condvar) =
    (std::sync::Mutex::new(false), std::sync::Condvar::new());

fn set_render_stopped() {
    *RENDER_STOPPED.0.lock().unwrap_or_else(|e| e.into_inner()) = true;
    RENDER_STOPPED.1.notify_all();
}

fn wait_render_stopped(timeout: Duration) {
    let g = RENDER_STOPPED.0.lock().unwrap_or_else(|e| e.into_inner());
    let (_g, r) = RENDER_STOPPED.1.wait_timeout_while(g, timeout, |stopped| !*stopped).unwrap_or_else(|e| e.into_inner());
    if r.timed_out() {
        log::warn!("ps5gpu: render thread did not stop in time");
    }
}

struct App {
    p: &'static Presenter,
    size: (u32, u32),
    window: Option<Arc<Window>>,
    vk: Option<VkState>,
    failed: bool,
    last: Option<(u32, u32, Arc<Vec<u8>>, Option<GpuFrame>, Option<bool>)>,
    redraw: bool,
    /// On-screen hint drawn on the last present.
    hint: Option<String>,
    /// Status line shown in the window title.
    status: Option<String>,
    pace: PaceStats,
    /// Performance overlay shown on the last present.
    overlay_on: bool,
    /// Time of the last present (the overlay refreshes a still image).
    last_present: Instant,
    /// egui of the in-game menu and the FPS counter (created on first use).
    menu: Option<crate::menu::Host>,
    /// The menu or the FPS counter was drawn on the last present.
    ui_on: bool,
    /// The shader compile overlay of the last UI frame.
    compile: Option<crate::osd::CompileOverlay>,
    compile_on: bool,
    /// The controls overlay was drawn on the last UI frame.
    controls_on: bool,
}

/// How often the shader compile overlay is redrawn when nothing else is.
const COMPILE_OVERLAY_PERIOD: Duration = Duration::from_millis(100);

/// Presentation timing, logged every 10 s: shown frames per second, the
/// gaps between them and the render thread's time spent presenting.
struct PaceStats {
    window_start: Instant,
    last_present: Option<Instant>,
    presents: u32,
    max_gap: Duration,
    busy: Duration,
    max_busy: Duration,
}

impl PaceStats {
    const PERIOD: Duration = Duration::from_secs(10);

    fn new() -> Self {
        Self { window_start: Instant::now(), last_present: None, presents: 0, max_gap: Duration::ZERO, busy: Duration::ZERO, max_busy: Duration::ZERO }
    }

    fn record(&mut self, started: Instant) {
        let now = Instant::now();
        let busy = now - started;
        self.busy += busy;
        self.max_busy = self.max_busy.max(busy);
        if let Some(last) = self.last_present {
            self.max_gap = self.max_gap.max(now - last);
        }
        self.last_present = Some(now);
        self.presents += 1;
    }

    fn report(&mut self, p: &Presenter) {
        let elapsed = self.window_start.elapsed();
        if elapsed < Self::PERIOD {
            return;
        }
        let s = p.stats();
        let secs = elapsed.as_secs_f64();
        log::info!(
            "ps5gpu presenter: {:.1} fps, max gap {:.1} ms, present {:.2} ms avg / {:.1} ms max, render thread busy {:.0}%, skipped {} total",
            self.presents as f64 / secs,
            self.max_gap.as_secs_f64() * 1e3,
            if self.presents > 0 { self.busy.as_secs_f64() * 1e3 / self.presents as f64 } else { 0.0 },
            self.max_busy.as_secs_f64() * 1e3,
            self.busy.as_secs_f64() * 100.0 / secs,
            s.skipped,
        );
        *self = Self { last_present: self.last_present, ..Self::new() };
    }
}

impl App {
    fn timeout(&self) -> Duration {
        if self.vk.is_none() && self.window.is_none() {
            return Duration::from_millis(1);
        }
        let st = self.p.lock();
        let frame = match st.queue.front() {
            None => Duration::from_millis(16),
            Some(f) => match f.not_before {
                None => Duration::ZERO,
                Some(t) => t.saturating_duration_since(Instant::now()).min(Duration::from_millis(16)),
            },
        };
        // An open menu and the controls overlay redraw at ~60 Hz, the compile overlay at ~10 Hz
        // (it also shows before the game's first frame).
        let frame = if crate::menu::is_open() || crate::menu::overlay_shown() { frame.min(crate::menu::UI_PERIOD) } else { frame };
        let frame = if self.compile_on || crate::osd::compile_progress().is_some() { frame.min(COMPILE_OVERLAY_PERIOD) } else { frame };
        // Wake up when the on-screen hint expires.
        match crate::osd::hint_deadline() {
            Some(t) => frame.min(t.saturating_duration_since(Instant::now())),
            None => frame,
        }
    }

    /// Run a UI frame for the in-game menu / FPS counter when one is due:
    /// with every new game frame, and while the menu is open at least every
    /// [`crate::menu::UI_PERIOD`] and on input.
    fn update_ui(&mut self, new_frame: bool) {
        let open = crate::menu::is_open();
        let controls = crate::menu::overlay_shown();
        let fps_on = crate::display::fps_counter();
        let compile = crate::osd::compile_overlay();
        let (Some(vk), Some(window)) = (self.vk.as_mut(), self.window.as_ref()) else { return };
        let want = (open || controls || fps_on || compile.is_some()) && !vk.ui_failed;
        // The controls overlay appearing or going redraws a still image.
        let controls_changed = controls != self.controls_on;
        self.controls_on = controls;
        // The overlay appearing, changing or going redraws a still image.
        let compile_changed = compile != self.compile;
        if compile_changed {
            self.compile = compile.clone();
            self.compile_on = compile.is_some();
        }
        if want != self.ui_on {
            self.ui_on = want;
            self.redraw = true;
            if !want {
                vk.ui_frame = None;
            }
        }
        if !want {
            return;
        }
        let max_side = vk.dev.caps.max_framebuffer_width.clamp(2048, 8192) as usize;
        let host = self.menu.get_or_insert_with(|| crate::menu::Host::new(window, max_side));
        let stale = host.last_run.is_none_or(|t| t.elapsed() >= crate::menu::UI_PERIOD);
        let spinner_due = compile.is_some() && host.last_run.is_none_or(|t| t.elapsed() >= COMPILE_OVERLAY_PERIOD);
        if !(new_frame || vk.ui_frame.is_none() || compile_changed || controls_changed || spinner_due || (open && host.input_pending) || ((open || controls) && stale)) {
            return;
        }
        let fps = fps_on.then(|| vk.hud.frames.summary(Instant::now(), 1.0).fps);
        let (frame, textures) = host.run(window, open, controls, fps, compile.as_ref());
        if !vk.ensure_ui() {
            crate::menu::set_open(false);
            return;
        }
        if let Some(ui) = vk.ui.as_mut() {
            ui.queue_textures(textures);
        }
        vk.ui_frame = Some(frame);
        if !new_frame {
            self.redraw = true;
        }
    }

    fn render(&mut self) {
        if self.vk.is_none() {
            return;
        }
        // The menu / FPS counter UI runs before the present it is drawn on.
        let newest = self.p.take_newest_due();
        self.update_ui(newest.is_some());
        let Some(vk) = self.vk.as_mut() else { return };
        // A new or expired hint redraws the last frame.
        // (Its display time starts with the first frame.)
        let has_frame = self.last.is_some() || newest.is_some() || !self.p.lock().queue.is_empty();
        let status = crate::osd::status();
        if status != self.status {
            if let Some(w) = &self.window {
                w.set_title(&status.as_ref().map_or_else(|| crate::APP_TITLE.to_string(), |s| format!("{} - {s}", crate::APP_TITLE)));
            }
            self.status = status.clone();
        }
        // A status line (work before the game starts) shows over black,
        // unless the compile overlay already shows that work.
        let status_line = status.filter(|_| vk.ui_failed || crate::osd::compile_progress().is_none());
        let hint = status_line.or_else(|| if has_frame { crate::osd::current_hint() } else { None });
        if hint != self.hint {
            self.hint = hint.clone();
            self.redraw = true;
        }
        vk.hint = hint;
        // Toggling the performance overlay redraws; while it is shown, a
        // still image is redrawn with fresh statistics.
        let overlay_on = crate::perf::overlay_enabled();
        if overlay_on != self.overlay_on {
            self.overlay_on = overlay_on;
            self.redraw = true;
        }
        if overlay_on && self.last_present.elapsed() >= crate::perf::SAMPLE_PERIOD {
            self.redraw = true;
        }
        // Only the newest due frame is shown: the guest never waits for
        // the window, so the window never queues up latency either.
        let mut presented_any = false;
        if let Some(frame) = newest {
            let started = Instant::now();
            vk.hud.frames.record(started);
            self.last_present = started;
            match vk.present(frame.width, frame.height, &frame.pixels, frame.gpu.as_ref(), frame.hdr) {
                Ok(true) => self.p.note_shown(),
                Ok(false) => {}
                Err(e) => {
                    log::error!("ps5gpu: present failed: {e:?}");
                    if e == vk::Result::ERROR_DEVICE_LOST {
                        crate::device::device_lost("present");
                        self.failed = true;
                    }
                }
            }
            self.pace.record(started);
            self.last = Some((frame.width, frame.height, frame.pixels, frame.gpu, frame.hdr));
            presented_any = true;
            if self.failed {
                return;
            }
        }
        if !presented_any && self.redraw {
            if let Some((w, h, px, gpu, hdr)) = self.last.clone() {
                let _ = vk.present(w, h, &px, gpu.as_ref(), hdr);
                self.last_present = Instant::now();
            } else if self.ui_on || self.hint.is_some() {
                // The menu or a status line before the game's first frame:
                // over black.
                let _ = vk.present(1, 1, &[0, 0, 0, 255], None, None);
                self.last_present = Instant::now();
            }
        }
        self.redraw = false;
        self.pace.report(self.p);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() || self.failed {
            return;
        }
        let (w, h) = self.size;
        let scale = (1280.0 / w as f64).min(1.0);
        let mut attrs = Window::default_attributes()
            .with_title(crate::APP_TITLE)
            .with_window_icon(winit::window::Icon::from_rgba(crate::WINDOW_ICON_RGBA.to_vec(), crate::WINDOW_ICON_SIZE, crate::WINDOW_ICON_SIZE).ok())
            .with_inner_size(winit::dpi::LogicalSize::new(w as f64 * scale, h as f64 * scale));
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            use winit::platform::{wayland::WindowAttributesExtWayland, x11::WindowAttributesExtX11};
            attrs = WindowAttributesExtWayland::with_name(attrs, crate::APP_ID, crate::APP_ID);
            attrs = WindowAttributesExtX11::with_name(attrs, crate::APP_ID, crate::APP_ID);
        }
        if crate::input::start_fullscreen() {
            attrs = attrs.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
        }
        crate::input::note_fullscreen(crate::input::start_fullscreen());
        let window = match el.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                log::warn!("ps5gpu: cannot create window ({e}); running headless");
                self.failed = true;
                return;
            }
        };
        match VkState::new(&window) {
            Ok(vk) => {
                log::info!(
                    "ps5gpu: Vulkan presenter on '{}' ({:?}, {}x{}, {:?})",
                    vk.gpu_name,
                    vk.swap_format,
                    vk.swap_extent.width,
                    vk.swap_extent.height,
                    vk.present_mode
                );
                if crate::display::hdr_enabled() {
                    if vk.hdr_supported {
                        log::info!("ps5gpu: HDR10 output available (used while the game shows HDR frames)");
                    } else {
                        log::warn!("ps5gpu: HDR10 surface unavailable; using SDR output");
                    }
                }
                self.vk = Some(vk);
                self.window = Some(window);
            }
            Err(e) => {
                log::warn!("ps5gpu: Vulkan initialisation failed ({e}); running headless");
                self.failed = true;
            }
        }
    }

    fn user_event(&mut self, _el: &ActiveEventLoop, _event: ()) {}

    fn device_event(&mut self, _el: &ActiveEventLoop, _id: winit::event::DeviceId, event: winit::event::DeviceEvent) {
        // Raw mouse motion is mouse-look: not while the menu is open.
        if !crate::menu::is_open() {
            crate::input::on_device_event(&event);
        }
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // An open menu takes the keyboard and mouse; the game only sees
        // focus changes (it releases what it holds).
        if crate::menu::is_open() {
            if let (Some(host), Some(window)) = (self.menu.as_mut(), self.window.as_ref()) {
                host.on_window_event(window, &event);
            }
            if matches!(event, WindowEvent::Focused(_)) {
                crate::input::on_window_event(&event);
            }
        } else {
            crate::input::on_window_event(&event);
        }
        match event {
            WindowEvent::CloseRequested => {
                log::info!("ps5gpu: window closed, exiting");
                ps5vm::shutdown::request(0);
            }
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(vk) = self.vk.as_mut() {
                    vk.resize_pending = true;
                }
                self.redraw = true;
            }
            WindowEvent::RedrawRequested => self.redraw = true,
            _ => {}
        }
    }
}

struct Staging {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    ptr: *mut u8,
}

struct GuestImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    width: u32,
    height: u32,
}

struct FrameRes {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    image_available: vk::Semaphore,
    staging: Option<Staging>,
    image: Option<GuestImage>,
    /// GPU snapshot read by the in-flight submission of this frame slot.
    held: Option<GpuFrame>,
    /// On-screen hint texels for this frame slot.
    osd: Option<Staging>,
    /// Performance overlay pixels (storage buffer) for this frame slot.
    perf: Option<Staging>,
}

/// Rasterised hint in the swapchain format.
struct OsdImage {
    text: String,
    format: vk::Format,
    hdr: bool,
    height_px: u32,
    width: u32,
    height: u32,
    texels: Vec<u8>,
}

struct VkState {
    window: Arc<Window>,
    dev: Arc<GpuDevice>,
    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    phys: vk::PhysicalDevice,
    gpu_name: String,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,
    swapchain_loader: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swap_images: Vec<vk::Image>,
    swap_format: vk::Format,
    swap_extent: vk::Extent2D,
    present_mode: vk::PresentModeKHR,
    /// The game image format can be a blit source.
    guest_blit_src: bool,
    /// HDR output is on and the surface offers an HDR10 format.
    hdr_supported: bool,
    /// The last frame with a colour space was HDR (PQ).
    hdr_requested: bool,
    /// The swapchain is HDR10.
    hdr_active: bool,
    /// A PQ frame was shown on an SDR swapchain (warned once).
    pq_on_sdr_warned: bool,
    render_finished: Vec<vk::Semaphore>,
    pool: vk::CommandPool,
    frames: Vec<FrameRes>,
    frame_index: usize,
    blit: bool,
    resize_pending: bool,
    /// Hint to draw over the next presents.
    hint: Option<String>,
    osd_image: Option<OsdImage>,
    /// Views of `swap_images` (the overlay pass renders into them).
    swap_views: Vec<vk::ImageView>,
    /// Performance overlay state and its pipeline (created on first use).
    pub(crate) hud: Hud,
    overlay: Option<OverlayPipeline>,
    overlay_failed: bool,
    /// In-game menu / FPS counter pass (created on first use) and the UI
    /// frame drawn over every present while set.
    ui: Option<crate::menu::renderer::UiRenderer>,
    ui_failed: bool,
    ui_frame: Option<crate::menu::renderer::UiFrame>,
    /// [`crate::display::present_generation`] the swapchain was made for.
    present_gen: u32,
    /// Presents so far (`RPS5E_DUMP_WINDOW_FRAMES`) and the read-back
    /// buffer of the window dump.
    presents: u64,
    window_dump_buf: Option<Staging>,
}

/// `RPS5E_DUMP_WINDOW_FRAMES=<dir>`: every `RPS5E_DUMP_WINDOW_EVERY`-th
/// present (default 30; every third while the shader compile panel shows)
/// is written as `window_<n>.ppm` (half size): the
/// window as shown, with the menu, the compile overlay and the hints over
/// the game frame (the guest frames of `RPS5E_DUMP_FRAMES` have none).
fn window_dump() -> Option<&'static (std::path::PathBuf, u64)> {
    static CFG: std::sync::OnceLock<Option<(std::path::PathBuf, u64)>> = std::sync::OnceLock::new();
    CFG.get_or_init(|| {
        let dir = std::path::PathBuf::from(std::env::var_os("RPS5E_DUMP_WINDOW_FRAMES").filter(|v| !v.is_empty())?);
        std::fs::create_dir_all(&dir).ok()?;
        let every = std::env::var("RPS5E_DUMP_WINDOW_EVERY").ok().and_then(|v| v.parse().ok()).filter(|&n| n > 0).unwrap_or(30);
        Some((dir, every))
    })
    .as_ref()
}

/// A screenshot asked for by [`request_screenshot`]: the next present is
/// read back and saved there.
static SCREENSHOT: std::sync::Mutex<Option<std::path::PathBuf>> = std::sync::Mutex::new(None);

/// Save the next presented window frame (the game with what is drawn over
/// it) as a PNG file at `path`.
pub fn request_screenshot(path: std::path::PathBuf) {
    *SCREENSHOT.lock().unwrap_or_else(|e| e.into_inner()) = Some(path);
    crate::input::wake();
}

fn take_screenshot() -> Option<std::path::PathBuf> {
    SCREENSHOT.lock().unwrap_or_else(|e| e.into_inner()).take()
}

/// Encode and write a screenshot off the render thread.
fn save_screenshot(path: std::path::PathBuf, w: u32, h: u32, bgra: Vec<u8>) {
    let _ = std::thread::Builder::new().name("rps5e-screenshot".into()).spawn(move || {
        let rgb: Vec<u8> = bgra.chunks_exact(4).flat_map(|p| [p[2], p[1], p[0]]).collect();
        match crate::encode_png_rgb(w, h, &rgb).and_then(|png| {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            }
            std::fs::write(&path, png).map_err(|e| e.to_string())
        }) {
            Ok(()) => log::info!("screenshot saved to {}", path.display()),
            Err(e) => log::warn!("screenshot {}: {e}", path.display()),
        }
    });
}

/// The performance overlay of the window: frame-time history and the
/// panel (see [`crate::perf`]).
pub(crate) struct Hud {
    pub frames: crate::perf::frames::FrameTimes,
    panel: Option<crate::perf::panel::Panel>,
    next_text: Instant,
}

impl Hud {
    fn new() -> Self {
        Hud { frames: crate::perf::frames::FrameTimes::new(), panel: None, next_text: Instant::now() }
    }

    /// The panel pixels for a `sw` x `sh` swapchain image, refreshing the
    /// text at 2 Hz and the graph every call; `None` when it does not fit.
    fn update(&mut self, sw: u32, sh: u32) -> Option<(u32, u32, &[u32])> {
        use crate::perf::panel::{Panel, GRAPH_SECONDS};
        crate::perf::ensure_sampler();
        let px = ((sh as f32 / 1080.0) * 15.0).round().clamp(11.0, 30.0) as u32;
        let now = Instant::now();
        if self.panel.as_ref().is_none_or(|p| p.text_px() != px) {
            self.panel = Panel::new(px);
            self.next_text = now;
        }
        let panel = self.panel.as_mut()?;
        if now >= self.next_text {
            self.next_text = now + crate::perf::SAMPLE_PERIOD;
            let mut summary = self.frames.summary(now, 1.0);
            summary.low1_fps = self.frames.summary(now, GRAPH_SECONDS).low1_fps;
            panel.set_stats(&summary, &crate::perf::latest_sample());
        }
        if panel.width + 16 > sw || panel.height + 16 > sh {
            return None;
        }
        self.frames.graph(now, GRAPH_SECONDS, &mut panel.columns);
        let (w, h) = (panel.width, panel.height);
        Some((w, h, panel.compose()))
    }

    /// The first `len` pixels last composed by [`Hud::update`].
    fn pixels(&self, len: usize) -> &[u32] {
        self.panel.as_ref().map_or(&[], |p| &p.pixels[..len])
    }
}

/// The overlay pass: blends the panel (a storage buffer of RGBA8 words)
/// over the swapchain image.
struct OverlayPipeline {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    format: vk::Format,
    hdr: bool,
}

/// Where the overlay is drawn: (buffer, x, y, width, height).
type OverlayDraw = (vk::Buffer, i32, i32, u32, u32);

type VkResult<T> = Result<T, vk::Result>;

/// Where the hint is drawn: (buffer, x, y, width, height).
type OsdCopy = (vk::Buffer, i32, i32, u32, u32);

impl VkState {
    fn new(window: &Arc<Window>) -> Result<Self, String> {
        let dev = GpuDevice::global().ok_or("no Vulkan device")?.clone();
        if !dev.surface_capable || !dev.caps.swapchain {
            return Err("the Vulkan device was created without surface/swapchain support".into());
        }
        let display = window.display_handle().map_err(|e| e.to_string())?.as_raw();
        let whandle = window.window_handle().map_err(|e| e.to_string())?.as_raw();
        let (entry, instance, phys, qf) = (&dev.entry, &dev.instance, dev.phys, dev.queue_family);
        // SAFETY: valid create info; all objects are destroyed in Drop.
        unsafe {
            let surface = ash_window::create_surface(entry, instance, display, whandle, None).map_err(|e| format!("surface: {e:?}"))?;
            let surface_loader = khr::surface::Instance::new(entry, instance);
            if !surface_loader.get_physical_device_surface_support(phys, qf, surface).unwrap_or(false) {
                surface_loader.destroy_surface(surface, None);
                return Err("the GPU queue family cannot present to the window".into());
            }
            let gpu_name = dev.name.clone();
            let mem_props = dev.mem_props;
            let device = dev.device.clone();
            let swapchain_loader = khr::swapchain::Device::new(instance, &device);
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(qf)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|e| format!("command pool: {e:?}"))?;
            let cmds = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32),
                )
                .map_err(|e| format!("command buffers: {e:?}"))?;
            let mut frames = Vec::new();
            for cmd in cmds {
                frames.push(FrameRes {
                    cmd,
                    fence: device
                        .create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)
                        .map_err(|e| format!("{e:?}"))?,
                    image_available: device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .map_err(|e| format!("{e:?}"))?,
                    staging: None,
                    image: None,
                    held: None,
                    osd: None,
                    perf: None,
                });
            }
            let src_feats = instance.get_physical_device_format_properties(phys, GUEST_IMAGE_FORMAT).optimal_tiling_features;
            let mut st = VkState {
                window: window.clone(),
                dev: dev.clone(),
                surface_loader,
                surface,
                phys,
                gpu_name,
                mem_props,
                device,
                swapchain_loader,
                swapchain: vk::SwapchainKHR::null(),
                swap_images: Vec::new(),
                swap_format: vk::Format::UNDEFINED,
                swap_extent: vk::Extent2D::default(),
                present_mode: vk::PresentModeKHR::FIFO,
                guest_blit_src: src_feats.contains(vk::FormatFeatureFlags::BLIT_SRC),
                hdr_supported: false,
                hdr_requested: false,
                hdr_active: false,
                pq_on_sdr_warned: false,
                render_finished: Vec::new(),
                pool,
                frames,
                frame_index: 0,
                blit: src_feats.contains(vk::FormatFeatureFlags::BLIT_SRC),
                resize_pending: false,
                hint: None,
                osd_image: None,
                swap_views: Vec::new(),
                hud: Hud::new(),
                overlay: None,
                overlay_failed: false,
                ui: None,
                ui_failed: false,
                ui_frame: None,
                present_gen: crate::display::present_generation(),
                presents: 0,
                window_dump_buf: None,
            };
            st.create_swapchain().map_err(|e| format!("swapchain: {e:?}"))?;
            Ok(st)
        }
    }

    fn window_extent(&self) -> vk::Extent2D {
        let s = self.window.inner_size();
        vk::Extent2D { width: s.width, height: s.height }
    }

    unsafe fn create_swapchain(&mut self) -> VkResult<()> {
        let caps = self.surface_loader.get_physical_device_surface_capabilities(self.phys, self.surface)?;
        let formats = self.surface_loader.get_physical_device_surface_formats(self.phys, self.surface)?;
        let modes = self.surface_loader.get_physical_device_surface_present_modes(self.phys, self.surface)?;
        // HDR10 needs the game image blitted into it (the copy fallback
        // would reinterpret the texels).
        let blit_dst = |f: vk::Format| {
            let feats = self.dev.instance.get_physical_device_format_properties(self.phys, f).optimal_tiling_features;
            feats.contains(vk::FormatFeatureFlags::BLIT_DST)
        };
        let hdr_usable = |f: vk::Format| self.guest_blit_src && blit_dst(f);
        let choice = crate::hdr::choose_surface_format(&formats, crate::display::hdr_enabled(), self.hdr_requested, hdr_usable)
            .ok_or(vk::Result::ERROR_FORMAT_NOT_SUPPORTED)?;
        let fmt = choice.format;
        if choice.hdr_active != self.hdr_active {
            log::info!("ps5gpu: presentation {}", if choice.hdr_active { "HDR10 (PQ)" } else { "SDR" });
        }
        self.hdr_supported = choice.hdr_supported;
        self.hdr_active = choice.hdr_active;
        crate::display::note_hdr_surface(choice.hdr_supported);
        self.blit = self.guest_blit_src && blit_dst(fmt.format);
        // Mailbox by default: a present never waits for the host vblank
        // (the guest's display clock does the pacing).
        let wanted_mode = match crate::display::present_mode() {
            crate::display::PresentMode::Fifo => vk::PresentModeKHR::FIFO,
            crate::display::PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
            crate::display::PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        };
        let mode = if modes.contains(&wanted_mode) {
            wanted_mode
        } else {
            if self.swapchain == vk::SwapchainKHR::null() {
                log::warn!("ps5gpu: present mode {wanted_mode:?} unavailable; falling back to FIFO");
            }
            vk::PresentModeKHR::FIFO
        };
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            let w = self.window_extent();
            vk::Extent2D {
                width: w.width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: w.height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        self.swap_extent = extent;
        self.swap_format = fmt.format;
        self.present_mode = mode;
        if extent.width == 0 || extent.height == 0 {
            return Ok(());
        }
        let mut count = caps.min_image_count + 1;
        if caps.max_image_count > 0 {
            count = count.min(caps.max_image_count);
        }
        let alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::INHERIT,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        ]
        .into_iter()
        .find(|a| caps.supported_composite_alpha.contains(*a))
        .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE);
        self.destroy_swap_views();
        let old = self.swapchain;
        let ci = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(count)
            .image_format(fmt.format)
            .image_color_space(fmt.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(
                vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | if window_dump().is_some() { caps.supported_usage_flags & vk::ImageUsageFlags::TRANSFER_SRC } else { vk::ImageUsageFlags::empty() },
            )
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(alpha)
            .present_mode(mode)
            .clipped(true)
            .old_swapchain(old);
        self.swapchain = self.swapchain_loader.create_swapchain(&ci, None)?;
        if old != vk::SwapchainKHR::null() {
            self.swapchain_loader.destroy_swapchain(old, None);
        }
        self.swap_images = self.swapchain_loader.get_swapchain_images(self.swapchain)?;
        for &image in &self.swap_images {
            let view = self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(fmt.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )?;
            self.swap_views.push(view);
        }
        for s in self.render_finished.drain(..) {
            self.device.destroy_semaphore(s, None);
        }
        for _ in 0..self.swap_images.len() {
            self.render_finished.push(self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?);
        }
        Ok(())
    }

    unsafe fn recreate_swapchain(&mut self) -> VkResult<()> {
        self.wait_frames()?;
        self.resize_pending = false;
        self.create_swapchain()?;
        // The overlay pipeline renders to the swapchain format.
        if self.overlay.as_ref().is_some_and(|o| o.format != self.swap_format || o.hdr != self.hdr_active) {
            let o = self.overlay.take().unwrap();
            self.destroy_overlay(o);
        }
        if let Some(ui) = self.ui.as_mut() {
            if let Err(e) = ui.set_format(self.swap_format, self.hdr_active) {
                log::warn!("ps5gpu: in-game menu unavailable (pipeline: {e:?})");
                self.ui = None;
                self.ui_failed = true;
            }
        }
        Ok(())
    }

    /// Create the menu pass on first use; false when it cannot be.
    fn ensure_ui(&mut self) -> bool {
        if self.ui.is_some() {
            return true;
        }
        if self.ui_failed {
            return false;
        }
        match crate::menu::renderer::UiRenderer::new(&self.device, self.mem_props, self.swap_format, self.hdr_active, MAX_FRAMES_IN_FLIGHT) {
            Ok(r) => {
                self.ui = Some(r);
                true
            }
            Err(e) => {
                log::warn!("ps5gpu: in-game menu unavailable (pipeline: {e:?})");
                self.ui_failed = true;
                crate::osd::show_hint("In-game menu unavailable (see the log)", Duration::from_secs(4));
                false
            }
        }
    }

    /// Prepare the UI frame (if any) for frame slot `fi`.
    fn prepare_ui(&mut self, fi: usize) -> VkResult<Option<crate::menu::renderer::UiDraw>> {
        let (Some(ui), Some(frame)) = (self.ui.as_mut(), self.ui_frame.as_ref()) else { return Ok(None) };
        if self.swap_views.len() != self.swap_images.len() {
            return Ok(None);
        }
        ui.prepare(fi, frame, self.swap_extent.width, self.swap_extent.height)
    }

    fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem_props.memory_type_count).find(|&i| {
            bits & (1 << i) != 0 && self.mem_props.memory_types[i as usize].property_flags.contains(flags)
        })
    }

    /// Host-visible upload buffer of at least `size` bytes, reusing `old`
    /// when it is large enough.
    unsafe fn ensure_staging(&self, old: &mut Option<Staging>, size: u64) -> VkResult<()> {
        self.ensure_host_buffer(old, size, vk::BufferUsageFlags::TRANSFER_SRC)
    }

    /// Host-visible buffer of `usage` and at least `size` bytes, reusing
    /// `old` when it is large enough.
    unsafe fn ensure_host_buffer(&self, old: &mut Option<Staging>, size: u64, usage: vk::BufferUsageFlags) -> VkResult<()> {
        if old.as_ref().is_some_and(|s| s.size >= size) {
            return Ok(());
        }
        if let Some(s) = old.take() {
            self.device.unmap_memory(s.memory);
            self.device.destroy_buffer(s.buffer, None);
            self.device.free_memory(s.memory, None);
        }
        let buffer = self.device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )?;
        let req = self.device.get_buffer_memory_requirements(buffer);
        let ty = self
            .memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
            .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
        let memory = self
            .device
            .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None)?;
        self.device.bind_buffer_memory(buffer, memory, 0)?;
        let ptr = self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())? as *mut u8;
        *old = Some(Staging { buffer, memory, size, ptr });
        Ok(())
    }

    /// Upload the current hint for frame slot `fi`; where to copy it.
    unsafe fn prepare_osd(&mut self, fi: usize) -> VkResult<Option<OsdCopy>> {
        let Some(text) = self.hint.clone() else { return Ok(None) };
        let (sw, sh) = (self.swap_extent.width, self.swap_extent.height);
        let height_px = ((sh as f32 / 720.0) * 20.0).clamp(14.0, 40.0).round() as u32;
        let stale = self
            .osd_image
            .as_ref()
            .is_none_or(|o| o.text != text || o.format != self.swap_format || o.hdr != self.hdr_active || o.height_px != height_px);
        if stale {
            self.osd_image = None;
            if let Some((w, h, rgba)) = crate::osd::rasterize(&text, height_px as f32) {
                let texels = if self.hdr_active {
                    crate::osd::convert_hdr10(&rgba, self.swap_format)
                } else {
                    crate::osd::convert(&rgba, self.swap_format)
                };
                if let Some(texels) = texels {
                    let (format, hdr) = (self.swap_format, self.hdr_active);
                    self.osd_image = Some(OsdImage { text, format, hdr, height_px, width: w, height: h, texels });
                }
            }
        }
        let Some(img) = self.osd_image.as_ref() else { return Ok(None) };
        let margin = (height_px as i32).max(8);
        if img.width as i32 + 2 * margin > sw as i32 || img.height as i32 + 2 * margin > sh as i32 {
            return Ok(None);
        }
        let mut slot = self.frames[fi].osd.take();
        let r = self.ensure_staging(&mut slot, img.texels.len() as u64);
        self.frames[fi].osd = slot;
        r?;
        let st = self.frames[fi].osd.as_ref().unwrap();
        std::ptr::copy_nonoverlapping(img.texels.as_ptr(), st.ptr, img.texels.len());
        Ok(Some((st.buffer, margin, margin, img.width, img.height)))
    }

    /// Copy the hint into the swapchain image (TRANSFER_DST layout, after
    /// the frame blit).
    unsafe fn record_osd(&self, cmd: vk::CommandBuffer, swap: vk::Image, osd: Option<OsdCopy>) {
        let Some((buffer, x, y, w, h)) = osd else { return };
        let d = &self.device;
        let mb = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        let copy = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x, y, z: 0 })
            .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
        d.cmd_copy_buffer_to_image(cmd, buffer, swap, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy]);
    }

    unsafe fn destroy_swap_views(&mut self) {
        for v in self.swap_views.drain(..) {
            self.device.destroy_image_view(v, None);
        }
    }

    unsafe fn destroy_overlay(&self, o: OverlayPipeline) {
        self.device.destroy_pipeline(o.pipeline, None);
        self.device.destroy_pipeline_layout(o.layout, None);
        self.device.destroy_descriptor_set_layout(o.set_layout, None);
    }

    /// The overlay pipeline for the swapchain format.
    unsafe fn create_overlay_pipeline(&self) -> VkResult<OverlayPipeline> {
        let d = &self.device;
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let set_layout = d.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default()
                .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                .bindings(&binding),
            None,
        )?;
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .size(crate::perf::shaders::PUSH_CONSTANT_BYTES)];
        let set_layouts = [set_layout];
        let layout = match d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts).push_constant_ranges(&ranges), None) {
            Ok(l) => l,
            Err(e) => {
                d.destroy_descriptor_set_layout(set_layout, None);
                return Err(e);
            }
        };
        let partial = OverlayPipeline { set_layout, layout, pipeline: vk::Pipeline::null(), format: self.swap_format, hdr: self.hdr_active };
        let fs_words = if self.hdr_active { crate::perf::shaders::fragment_hdr() } else { crate::perf::shaders::fragment() };
        let vs_words = crate::perf::shaders::vertex();
        let module = |words: &[u32]| d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None);
        let (vs, fs) = match (module(&vs_words), module(&fs_words)) {
            (Ok(vs), Ok(fs)) => (vs, fs),
            (a, b) => {
                for m in [a, b].into_iter().flatten() {
                    d.destroy_shader_module(m, None);
                }
                self.destroy_overlay(partial);
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
        };
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vs).name(c"main"),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(fs).name(c"main"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let formats = [self.swap_format];
        let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering);
        let r = d.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None);
        d.destroy_shader_module(vs, None);
        d.destroy_shader_module(fs, None);
        match r {
            Ok(p) => Ok(OverlayPipeline { pipeline: p[0], ..partial }),
            Err((_, e)) => {
                self.destroy_overlay(partial);
                Err(e)
            }
        }
    }

    /// Compose the performance overlay and upload it for frame slot `fi`;
    /// where to draw it.
    unsafe fn prepare_overlay(&mut self, fi: usize) -> VkResult<Option<OverlayDraw>> {
        if !crate::perf::overlay_enabled() || self.overlay_failed {
            return Ok(None);
        }
        if fi >= self.frames.len() || self.swap_views.len() != self.swap_images.len() {
            return Ok(None);
        }
        if self.overlay.is_none() {
            match self.create_overlay_pipeline() {
                Ok(o) => self.overlay = Some(o),
                Err(e) => {
                    log::warn!("ps5gpu: performance overlay unavailable (pipeline: {e:?})");
                    self.overlay_failed = true;
                    return Ok(None);
                }
            }
        }
        let (sw, sh) = (self.swap_extent.width, self.swap_extent.height);
        let Some((w, h, len)) = self.hud.update(sw, sh).map(|(w, h, px)| (w, h, px.len())) else { return Ok(None) };
        let bytes = (len * 4) as u64;
        let mut slot = self.frames[fi].perf.take();
        let r = self.ensure_host_buffer(&mut slot, bytes, vk::BufferUsageFlags::STORAGE_BUFFER);
        self.frames[fi].perf = slot;
        r?;
        let buf = self.frames[fi].perf.as_ref().unwrap();
        let pixels = self.hud.pixels(len);
        std::ptr::copy_nonoverlapping(pixels.as_ptr() as *const u8, buf.ptr, bytes as usize);
        let margin = ((sh as f32 / 1080.0) * 12.0).round().max(6.0) as u32;
        let x = sw.saturating_sub(w + margin);
        if margin + h > sh {
            return Ok(None);
        }
        Ok(Some((buf.buffer, x as i32, margin as i32, w, h)))
    }

    /// Draw the overlay and the menu (if any) and move the swapchain image
    /// from TRANSFER_DST to PRESENT_SRC.
    unsafe fn finish_swap(
        &self,
        cmd: vk::CommandBuffer,
        swap: vk::Image,
        idx: u32,
        overlay: Option<OverlayDraw>,
        ui: Option<&crate::menu::renderer::UiDraw>,
    ) {
        use vk::AccessFlags as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags as S;
        let d = &self.device;
        let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let barrier = |old, new, src, dst| {
            vk::ImageMemoryBarrier::default()
                .image(swap)
                .old_layout(old)
                .new_layout(new)
                .src_access_mask(src)
                .dst_access_mask(dst)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(range)
        };
        let ui = ui.zip(self.ui.as_ref());
        let overlay = overlay.zip(self.overlay.as_ref());
        let view = self.swap_views.get(idx as usize).copied();
        let (Some(view), true) = (view, overlay.is_some() || ui.is_some()) else {
            let b = barrier(L::TRANSFER_DST_OPTIMAL, L::PRESENT_SRC_KHR, A::TRANSFER_WRITE, A::empty());
            d.cmd_pipeline_barrier(cmd, S::TRANSFER, S::BOTTOM_OF_PIPE, vk::DependencyFlags::empty(), &[], &[], &[b]);
            return;
        };
        if let Some((draw, r)) = ui {
            r.record_uploads(cmd, draw);
        }
        let b = barrier(L::TRANSFER_DST_OPTIMAL, L::COLOR_ATTACHMENT_OPTIMAL, A::TRANSFER_WRITE, A::COLOR_ATTACHMENT_READ | A::COLOR_ATTACHMENT_WRITE);
        d.cmd_pipeline_barrier(cmd, S::TRANSFER, S::COLOR_ATTACHMENT_OUTPUT, vk::DependencyFlags::empty(), &[], &[], &[b]);
        if let Some(((buffer, x, y, w, h), o)) = overlay {
            self.record_overlay(cmd, view, o, buffer, x, y, w, h);
        }
        if let Some((draw, r)) = ui {
            r.record_draws(cmd, &self.dev.push_descriptor, view, self.swap_extent.width, self.swap_extent.height, draw);
        }
        let b = barrier(L::COLOR_ATTACHMENT_OPTIMAL, L::PRESENT_SRC_KHR, A::COLOR_ATTACHMENT_WRITE, A::empty());
        d.cmd_pipeline_barrier(cmd, S::COLOR_ATTACHMENT_OUTPUT, S::BOTTOM_OF_PIPE, vk::DependencyFlags::empty(), &[], &[], &[b]);
    }

    /// The performance overlay pass (swapchain image in COLOR_ATTACHMENT_OPTIMAL).
    #[allow(clippy::too_many_arguments)]
    unsafe fn record_overlay(&self, cmd: vk::CommandBuffer, view: vk::ImageView, o: &OverlayPipeline, buffer: vk::Buffer, x: i32, y: i32, w: u32, h: u32) {
        use vk::ImageLayout as L;
        let d = &self.device;
        let area = vk::Rect2D { offset: vk::Offset2D { x, y }, extent: vk::Extent2D { width: w, height: h } };
        let color = [vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(L::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)];
        d.cmd_begin_rendering(cmd, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&color));
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, o.pipeline);
        d.cmd_set_viewport(cmd, 0, &[vk::Viewport { x: x as f32, y: y as f32, width: w as f32, height: h as f32, min_depth: 0.0, max_depth: 1.0 }]);
        d.cmd_set_scissor(cmd, 0, &[area]);
        let info = [vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE }];
        let write = vk::WriteDescriptorSet::default().dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info);
        self.dev.push_descriptor.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::GRAPHICS, o.layout, 0, &[write]);
        let mut pc = [0u8; 16];
        pc[0..4].copy_from_slice(&x.to_le_bytes());
        pc[4..8].copy_from_slice(&y.to_le_bytes());
        pc[8..12].copy_from_slice(&w.to_le_bytes());
        d.cmd_push_constants(cmd, o.layout, vk::ShaderStageFlags::FRAGMENT, 0, &pc);
        d.cmd_draw(cmd, 3, 1, 0, 0);
        d.cmd_end_rendering(cmd);
    }

    unsafe fn ensure_frame_resources(&mut self, fi: usize, w: u32, h: u32) -> VkResult<()> {
        let need = w as u64 * h as u64 * 4;
        let mut slot = self.frames[fi].staging.take();
        let r = self.ensure_staging(&mut slot, need);
        self.frames[fi].staging = slot;
        r?;
        if self.frames[fi].image.as_ref().is_none_or(|g| g.width != w || g.height != h) {
            if let Some(g) = self.frames[fi].image.take() {
                self.device.destroy_image(g.image, None);
                self.device.free_memory(g.memory, None);
            }
            let image = self.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(GUEST_IMAGE_FORMAT)
                    .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?;
            let req = self.device.get_image_memory_requirements(image);
            // Device-local first; a full device heap (the GPU is shared)
            // places the frame in host memory instead of losing it.
            let memory = match self.dev.allocate(req, vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::empty(), false) {
                Ok((memory, _)) => memory,
                Err(e) => {
                    self.device.destroy_image(image, None);
                    return Err(e);
                }
            };
            if let Err(e) = self.device.bind_image_memory(image, memory, 0) {
                self.device.destroy_image(image, None);
                self.device.free_memory(memory, None);
                return Err(e);
            }
            self.frames[fi].image = Some(GuestImage { image, memory, width: w, height: h });
        }
        Ok(())
    }

    /// Wait for every frame slot's last submission (instead of a device
    /// wait idle, which would also wait for the GPU back-end's work).
    unsafe fn wait_frames(&mut self) -> VkResult<()> {
        let fences: Vec<vk::Fence> = self.frames.iter().map(|f| f.fence).collect();
        self.device.wait_for_fences(&fences, true, u64::MAX)
    }

    /// The colour space of the frame to show (`None`: keep the current
    /// one); true when the swapchain must be recreated for it.
    fn set_hdr(&mut self, hdr: Option<bool>) -> bool {
        let Some(hdr) = hdr else { return false };
        if hdr && !self.hdr_supported && !self.pq_on_sdr_warned {
            self.pq_on_sdr_warned = true;
            log::warn!("ps5gpu: HDR output is unavailable; HDR-to-SDR conversion is not implemented, displaying PQ video-out pixels unchanged");
        }
        let changed = self.hdr_active != (hdr && self.hdr_supported);
        self.hdr_requested = hdr;
        changed
    }

    /// Show one BGRA8 frame, or the GPU snapshot `gpu`, in the colour space
    /// `hdr` (see [`VkState::set_hdr`]). Ok(true) when it reached the
    /// presentation engine, Ok(false) when it was dropped (e.g. minimised
    /// window).
    fn present(&mut self, w: u32, h: u32, pixels: &[u8], gpu: Option<&GpuFrame>, hdr: Option<bool>) -> VkResult<bool> {
        // SAFETY: all handles are owned by self and used on this thread only.
        unsafe {
            let gen = crate::display::present_generation();
            if gen != self.present_gen {
                self.present_gen = gen;
                self.resize_pending = true;
            }
            if self.set_hdr(hdr) {
                self.resize_pending = true;
            }
            if self.resize_pending || self.swap_images.is_empty() {
                self.recreate_swapchain()?;
            }
            for attempt in 0..2 {
                if self.swap_extent.width == 0 || self.swap_extent.height == 0 {
                    return Ok(false);
                }
                let fi = self.frame_index;
                let fence = self.frames[fi].fence;
                self.device.wait_for_fences(&[fence], true, u64::MAX)?;
                let acquire = self.swapchain_loader.acquire_next_image(
                    self.swapchain,
                    acquire_timeout_ns(),
                    self.frames[fi].image_available,
                    vk::Fence::null(),
                );
                let (idx, _suboptimal) = match acquire {
                    Ok(v) => v,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) if attempt == 0 => {
                        self.recreate_swapchain()?;
                        continue;
                    }
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(false),
                    Err(e) if acquire_drops(e) => {
                        // No image within the bound (a hidden window under
                        // FIFO never returns one): the frame is dropped,
                        // the guest's flip completes.
                        crate::log_once(&format!("ps5gpu: swapchain image not available ({e:?}); frame dropped"));
                        return Ok(false);
                    }
                    Err(e) => return Err(e),
                };
                self.device.reset_fences(&[fence])?;
                // The previous submission of this slot finished: release its snapshot.
                self.frames[fi].held = None;
                let swap = self.swap_images[idx as usize];
                let fr_cmd = self.frames[fi].cmd;
                let osd = self.prepare_osd(fi)?;
                let overlay = self.prepare_overlay(fi)?;
                let ui = match self.prepare_ui(fi) {
                    Ok(u) => u,
                    Err(e) => {
                        log::warn!("ps5gpu: in-game menu frame dropped: {e:?}");
                        None
                    }
                };
                let mut waits = vec![vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.frames[fi].image_available)
                    .stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)];
                if let Some(g) = gpu {
                    self.record_gpu(fr_cmd, g, swap, osd, idx, overlay, ui.as_ref())?;
                    waits.push(
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(g.timeline)
                            .value(g.tick)
                            .stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER),
                    );
                    self.frames[fi].held = Some(g.clone());
                } else {
                    self.ensure_frame_resources(fi, w, h)?;
                    let len = (w as usize * h as usize * 4).min(pixels.len());
                    let fr = &self.frames[fi];
                    let staging = fr.staging.as_ref().unwrap();
                    std::ptr::copy_nonoverlapping(pixels.as_ptr(), staging.ptr, len);
                    let gimg = fr.image.as_ref().unwrap().image;
                    self.record(fr.cmd, staging.buffer, gimg, w, h, swap, osd, idx, overlay, ui.as_ref())?;
                }

                let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(fr_cmd)];
                let signal = [self.render_finished[idx as usize]];
                let signal_info = [vk::SemaphoreSubmitInfo::default().semaphore(signal[0]).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
                let submit = vk::SubmitInfo2::default().wait_semaphore_infos(&waits).command_buffer_infos(&cmds).signal_semaphore_infos(&signal_info);
                let queue = self.dev.queue();
                self.device.queue_submit2(*queue, &[submit], fence)?;
                self.frame_index = (fi + 1) % MAX_FRAMES_IN_FLIGHT;
                self.presents += 1;
                if let Some(path) = take_screenshot() {
                    drop(queue);
                    match self.read_window(swap, fence) {
                        Ok((w, h, px)) => save_screenshot(path, w, h, px),
                        Err(e) => log::warn!("ps5gpu: screenshot failed: {e:?}"),
                    }
                    return self.queue_present_now(idx);
                }
                if let Some((dir, every)) = window_dump() {
                    // While the shader compile panel shows (a few presents
                    // a second over black), every third present.
                    let panel = crate::osd::compile_progress().is_some() && self.presents % 3 == 0;
                    if self.presents % every == 1 || *every == 1 || panel {
                        drop(queue);
                        if let Err(e) = self.dump_window(swap, fence, dir) {
                            crate::log_once(&format!("ps5gpu: window dump failed: {e:?}"));
                        }
                        return self.queue_present_now(idx);
                    }
                }

                drop(queue);
                return self.queue_present_now(idx);
            }
            Ok(false)
        }
    }

    /// Present swapchain image `idx` (its submission signals
    /// `render_finished[idx]`).
    unsafe fn queue_present_now(&mut self, idx: u32) -> VkResult<bool> {
        self.window.pre_present_notify();
        let signal = [self.render_finished[idx as usize]];
        let swapchains = [self.swapchain];
        let indices = [idx];
        let pi = vk::PresentInfoKHR::default().wait_semaphores(&signal).swapchains(&swapchains).image_indices(&indices);
        let queue = self.dev.queue();
        let r = self.swapchain_loader.queue_present(*queue, &pi);
        drop(queue);
        match r {
            Ok(suboptimal) => {
                if suboptimal {
                    self.resize_pending = true;
                }
                Ok(true)
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.resize_pending = true;
                Ok(true)
            }
            Err(e) => Err(e),
        }
    }

    /// Read the finished swapchain image `swap` back (after `fence`) and
    /// write it into `dir` (see [`window_dump`]). Debug only: it waits.
    unsafe fn dump_window(&mut self, swap: vk::Image, fence: vk::Fence, dir: &std::path::Path) -> VkResult<()> {
        let (w, h, px) = self.read_window(swap, fence)?;
        let _ = std::fs::write(dir.join(format!("window_{:05}.ppm", self.presents)), crate::ppm_scaled(w, h, &px, 2));
        Ok(())
    }

    /// The finished swapchain image `swap` (after `fence`) as BGRA pixels.
    /// It waits for the GPU.
    unsafe fn read_window(&mut self, swap: vk::Image, fence: vk::Fence) -> VkResult<(u32, u32, Vec<u8>)> {
        let (w, h) = (self.swap_extent.width, self.swap_extent.height);
        let swizzle = match self.swap_format {
            vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => false,
            vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => true,
            _ => return Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED),
        };
        self.device.wait_for_fences(&[fence], true, u64::MAX)?;
        let size = w as u64 * h as u64 * 4;
        let mut buf = self.window_dump_buf.take();
        let r = self.ensure_host_buffer(&mut buf, size, vk::BufferUsageFlags::TRANSFER_DST);
        self.window_dump_buf = buf;
        r?;
        let buffer = self.window_dump_buf.as_ref().unwrap().buffer;
        let cmd = self.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
        let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let barrier = |from, to, src, dst| {
            vk::ImageMemoryBarrier::default().old_layout(from).new_layout(to).src_access_mask(src).dst_access_mask(dst).image(swap).subresource_range(range)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        };
        let d = &self.device;
        d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[barrier(
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::MEMORY_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        )]);
        let copy = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
            .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
        d.cmd_copy_image_to_buffer(cmd, swap, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buffer, &[copy]);
        d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::BOTTOM_OF_PIPE, vk::DependencyFlags::empty(), &[], &[], &[barrier(
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::empty(),
        )]);
        d.end_command_buffer(cmd)?;
        let done = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cmds = [cmd];
        let queue = self.dev.queue();
        let r = d.queue_submit(*queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], done);
        drop(queue);
        let r = r.and_then(|_| d.wait_for_fences(&[done], true, u64::MAX));
        d.destroy_fence(done, None);
        d.free_command_buffers(self.pool, &cmds);
        r?;
        let st = self.window_dump_buf.as_ref().unwrap();
        let mut px = std::slice::from_raw_parts(st.ptr, size as usize).to_vec();
        if swizzle {
            px.chunks_exact_mut(4).for_each(|p| p.swap(0, 2));
        }
        Ok((w, h, px))
    }

    /// Blit a GPU snapshot (GENERAL layout) into the swapchain image.
    #[allow(clippy::too_many_arguments)]
    unsafe fn record_gpu(
        &self,
        cmd: vk::CommandBuffer,
        g: &GpuFrame,
        swap: vk::Image,
        osd: Option<OsdCopy>,
        idx: u32,
        overlay: Option<OverlayDraw>,
        ui: Option<&crate::menu::renderer::UiDraw>,
    ) -> VkResult<()> {
        let d = &self.device;
        d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
        let to_dst = vk::ImageMemoryBarrier::default()
            .image(swap)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(range);
        let snap_read = vk::ImageMemoryBarrier::default()
            .image(g.snapshot.image.image)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(range);
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_dst, snap_read],
        );
        let black = vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] };
        d.cmd_clear_color_image(cmd, swap, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &black, &[range]);
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        let (w, h) = (g.snapshot.width, g.snapshot.height);
        let (sw, sh) = (self.swap_extent.width, self.swap_extent.height);
        let (ox, oy, dw, dh) = crate::display::dest_rect(w, h, sw, sh, crate::display::scale_mode());
        let region = vk::ImageBlit::default()
            .src_subresource(layers)
            .src_offsets([vk::Offset3D::default(), vk::Offset3D { x: w as i32, y: h as i32, z: 1 }])
            .dst_subresource(layers)
            .dst_offsets([vk::Offset3D { x: ox, y: oy, z: 0 }, vk::Offset3D { x: ox + dw, y: oy + dh, z: 1 }]);
        d.cmd_blit_image(cmd, g.snapshot.image.image, vk::ImageLayout::GENERAL, swap, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region], blit_filter());
        self.record_osd(cmd, swap, osd);
        self.finish_swap(cmd, swap, idx, overlay, ui);
        d.end_command_buffer(cmd)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn record(
        &self,
        cmd: vk::CommandBuffer,
        staging: vk::Buffer,
        gimg: vk::Image,
        w: u32,
        h: u32,
        swap: vk::Image,
        osd: Option<OsdCopy>,
        idx: u32,
        overlay: Option<OverlayDraw>,
        ui: Option<&crate::menu::renderer::UiDraw>,
    ) -> VkResult<()> {
        let d = &self.device;
        d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let barrier = |image, old, new, src_access, dst_access| {
            vk::ImageMemoryBarrier::default()
                .image(image)
                .old_layout(old)
                .new_layout(new)
                .src_access_mask(src_access)
                .dst_access_mask(dst_access)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(range)
        };
        let pipeline_barrier = |src: vk::PipelineStageFlags, dst: vk::PipelineStageFlags, b: &[vk::ImageMemoryBarrier]| {
            d.cmd_pipeline_barrier(cmd, src, dst, vk::DependencyFlags::empty(), &[], &[], b);
        };
        use vk::AccessFlags as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags as S;
        pipeline_barrier(
            S::TOP_OF_PIPE,
            S::TRANSFER,
            &[
                barrier(gimg, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, A::empty(), A::TRANSFER_WRITE),
                barrier(swap, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, A::empty(), A::TRANSFER_WRITE),
            ],
        );
        let copy = vk::BufferImageCopy::default()
            .image_subresource(layers)
            .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
        d.cmd_copy_buffer_to_image(cmd, staging, gimg, L::TRANSFER_DST_OPTIMAL, &[copy]);
        let black = vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] };
        d.cmd_clear_color_image(cmd, swap, L::TRANSFER_DST_OPTIMAL, &black, &[range]);
        pipeline_barrier(
            S::TRANSFER,
            S::TRANSFER,
            &[
                barrier(gimg, L::TRANSFER_DST_OPTIMAL, L::TRANSFER_SRC_OPTIMAL, A::TRANSFER_WRITE, A::TRANSFER_READ),
                barrier(swap, L::TRANSFER_DST_OPTIMAL, L::TRANSFER_DST_OPTIMAL, A::TRANSFER_WRITE, A::TRANSFER_WRITE),
            ],
        );
        let (sw, sh) = (self.swap_extent.width, self.swap_extent.height);
        let (ox, oy, dw, dh) = crate::display::dest_rect(w, h, sw, sh, crate::display::scale_mode());
        if self.blit {
            let region = vk::ImageBlit::default()
                .src_subresource(layers)
                .src_offsets([vk::Offset3D::default(), vk::Offset3D { x: w as i32, y: h as i32, z: 1 }])
                .dst_subresource(layers)
                .dst_offsets([vk::Offset3D { x: ox, y: oy, z: 0 }, vk::Offset3D { x: ox + dw, y: oy + dh, z: 1 }]);
            d.cmd_blit_image(cmd, gimg, L::TRANSFER_SRC_OPTIMAL, swap, L::TRANSFER_DST_OPTIMAL, &[region], blit_filter());
        } else {
            let region = vk::ImageCopy::default()
                .src_subresource(layers)
                .dst_subresource(layers)
                .extent(vk::Extent3D { width: w.min(sw), height: h.min(sh), depth: 1 });
            d.cmd_copy_image(cmd, gimg, L::TRANSFER_SRC_OPTIMAL, swap, L::TRANSFER_DST_OPTIMAL, &[region]);
        }
        self.record_osd(cmd, swap, osd);
        self.finish_swap(cmd, swap, idx, overlay, ui);
        d.end_command_buffer(cmd)
    }
}

/// The filter of the game image blit ([`crate::display::scale_filter`]).
fn blit_filter() -> vk::Filter {
    match crate::display::scale_filter() {
        crate::display::ScaleFilter::Linear => vk::Filter::LINEAR,
        crate::display::ScaleFilter::Nearest => vk::Filter::NEAREST,
    }
}

impl Drop for VkState {
    fn drop(&mut self) {
        // SAFETY: tearing down objects this state created, after the device is idle.
        unsafe {
            let _ = self.wait_frames();
            self.ui = None;
            for f in self.frames.drain(..) {
                if let Some(s) = f.staging {
                    self.device.unmap_memory(s.memory);
                    self.device.destroy_buffer(s.buffer, None);
                    self.device.free_memory(s.memory, None);
                }
                if let Some(g) = f.image {
                    self.device.destroy_image(g.image, None);
                    self.device.free_memory(g.memory, None);
                }
                for s in [f.osd, f.perf].into_iter().flatten() {
                    self.device.unmap_memory(s.memory);
                    self.device.destroy_buffer(s.buffer, None);
                    self.device.free_memory(s.memory, None);
                }
                self.device.destroy_fence(f.fence, None);
                self.device.destroy_semaphore(f.image_available, None);
            }
            for s in self.render_finished.drain(..) {
                self.device.destroy_semaphore(s, None);
            }
            if let Some(s) = self.window_dump_buf.take() {
                self.device.unmap_memory(s.memory);
                self.device.destroy_buffer(s.buffer, None);
                self.device.free_memory(s.memory, None);
            }
            self.destroy_swap_views();
            if let Some(o) = self.overlay.take() {
                self.destroy_overlay(o);
            }
            self.device.destroy_command_pool(self.pool, None);
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            }
            // The device and instance belong to the shared GpuDevice.
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

/// `RPS5E_ACQUIRE_TIMEOUT_MS` (default 100): how long the presenter waits
/// for a swapchain image before it drops the frame. An unbounded wait (a
/// hidden window under FIFO never hands one out) held every later flip and
/// the guest waiting for them.
fn acquire_timeout_ns() -> u64 {
    static NS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *NS.get_or_init(|| parse_acquire_timeout(std::env::var("RPS5E_ACQUIRE_TIMEOUT_MS").ok().as_deref()))
}

fn parse_acquire_timeout(v: Option<&str>) -> u64 {
    v.and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(100).saturating_mul(1_000_000)
}

/// Acquire results that drop the frame instead of failing the presenter.
fn acquire_drops(e: vk::Result) -> bool {
    matches!(e, vk::Result::TIMEOUT | vk::Result::NOT_READY)
}

#[cfg(test)]
mod acquire_tests {
    use super::*;

    #[test]
    fn acquire_waits_are_bounded_and_drop_the_frame() {
        assert_eq!(parse_acquire_timeout(None), 100_000_000);
        assert_eq!(parse_acquire_timeout(Some("0")), 0);
        assert_eq!(parse_acquire_timeout(Some(" 16 ")), 16_000_000);
        assert_eq!(parse_acquire_timeout(Some("x")), 100_000_000);
        assert!(acquire_drops(vk::Result::TIMEOUT) && acquire_drops(vk::Result::NOT_READY));
        assert!(!acquire_drops(vk::Result::ERROR_DEVICE_LOST) && !acquire_drops(vk::Result::ERROR_SURFACE_LOST_KHR));
    }
}
