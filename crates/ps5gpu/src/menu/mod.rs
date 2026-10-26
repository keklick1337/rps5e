// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! In-game menu host: an egui UI drawn over the game in the game window.
//!
//! The menu itself (its sections and settings) lives in the frontend and
//! is installed with [`install`]; this module owns its open state and the
//! render thread drives it:
//!
//! - while the menu is open, window keyboard/mouse events go to egui
//!   (through `egui-winit`), not to the game's input sink (it only sees
//!   focus changes), and the UI runs and is drawn over every present (the
//!   last game frame is redrawn at ~60 Hz when the game shows nothing new);
//! - [`MenuApp::raw_input`] lets the frontend add events of its own
//!   (controller navigation) to each UI frame;
//! - [`on_open_change`] listeners run when the menu opens or closes (the
//!   input layer neutralises the pads, the frontend pauses the game);
//! - while the controls overlay key is held ([`set_overlay`]) and the menu
//!   is closed, [`MenuApp::overlay`] draws over the running game; it takes
//!   no input (the game keeps it).
//!
//! The egui output is rendered by a small Vulkan pass ([`renderer`],
//! SPIR-V built with rspirv in [`shaders`]): one pipeline, one combined
//! image sampler per mesh (push descriptors), per-frame host-visible vertex,
//! index and texture upload buffers. When the menu is closed nothing of it
//! runs (one atomic load per pump), unless the FPS counter
//! ([`crate::display::fps_counter`]) is on: that draws one label the same way.

pub(crate) mod renderer;
pub mod shaders;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use winit::window::Window;

/// The frontend's menu.
pub trait MenuApp: Send {
    /// Add events of this UI frame (e.g. controller navigation as keys).
    fn raw_input(&mut self, _raw: &mut egui::RawInput) {}
    /// Build the UI of one frame. The menu closes itself with
    /// [`set_open`]`(false)`.
    fn ui(&mut self, ctx: &egui::Context);
    /// Draw the read-only controls overlay (the menu is closed).
    fn overlay(&mut self, _ctx: &egui::Context) {}
}

type Listener = Arc<dyn Fn(bool) + Send + Sync>;

static APP: Mutex<Option<Box<dyn MenuApp>>> = Mutex::new(None);
static INSTALLED: AtomicBool = AtomicBool::new(false);
static OPEN: AtomicBool = AtomicBool::new(false);
static OVERLAY: AtomicBool = AtomicBool::new(false);
static LISTENERS: Mutex<Vec<Listener>> = Mutex::new(Vec::new());

/// Install the menu (replacing an earlier one).
pub fn install(app: Box<dyn MenuApp>) {
    *APP.lock().unwrap_or_else(|e| e.into_inner()) = Some(app);
    INSTALLED.store(true, Ordering::Release);
}

/// Whether a menu is installed (it can be opened).
pub fn installed() -> bool {
    INSTALLED.load(Ordering::Acquire)
}

pub fn is_open() -> bool {
    OPEN.load(Ordering::Acquire)
}

/// Open or close the menu; listeners run on a change. Opening without an
/// installed menu does nothing. Must not be called with a lock held that a
/// listener takes (the input layer's state).
pub fn set_open(open: bool) {
    if open && !installed() {
        return;
    }
    if OPEN.swap(open, Ordering::AcqRel) == open {
        return;
    }
    // The menu replaces the overlay; its key's release goes to the menu.
    if open {
        OVERLAY.store(false, Ordering::Release);
    }
    log::info!("in-game menu {}", if open { "opened" } else { "closed" });
    let listeners: Vec<Listener> = LISTENERS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for l in listeners {
        l(open);
    }
    crate::input::wake();
}

/// Show the controls overlay (its key held) or hide it.
pub fn set_overlay(on: bool) {
    if OVERLAY.swap(on, Ordering::AcqRel) != on {
        crate::input::wake();
    }
}

/// Whether the controls overlay is drawn: requested, a menu installed to
/// draw it, and the menu closed.
pub fn overlay_shown() -> bool {
    OVERLAY.load(Ordering::Acquire) && installed() && !is_open()
}

/// Flip the menu; returns the new state.
pub fn toggle() -> bool {
    let open = !is_open();
    set_open(open);
    is_open()
}

/// Run `f(open)` on every open/close (on the thread that changed it).
pub fn on_open_change(f: impl Fn(bool) + Send + Sync + 'static) {
    LISTENERS.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::new(f));
}

/// Run the installed menu (or, closed, its controls overlay when
/// `overlay`) for one UI frame (render thread); `extra` adds to the same
/// frame (the FPS counter).
pub(crate) fn run_app(raw: &mut egui::RawInput, ctx: &egui::Context, open: bool, overlay: bool, extra: impl Fn(&egui::Context)) -> egui::FullOutput {
    let mut g = APP.lock().unwrap_or_else(|e| e.into_inner());
    match g.as_mut().filter(|_| open || overlay) {
        Some(app) if open => {
            app.raw_input(raw);
            ctx.run(raw.take(), |ctx| {
                app.ui(ctx);
                extra(ctx);
            })
        }
        Some(app) => ctx.run(raw.take(), |ctx| {
            app.overlay(ctx);
            extra(ctx);
        }),
        None => {
            drop(g);
            ctx.run(raw.take(), |ctx| extra(ctx))
        }
    }
}

/// How often an open menu redraws when nothing else happens.
pub(crate) const UI_PERIOD: Duration = Duration::from_millis(16);

/// UI scale for a window `height_px` tall at `native_ppp`: the menu keeps
/// about the same share of the window (it is read from a couch, too).
pub(crate) fn zoom_for(height_px: u32, native_ppp: f32) -> f32 {
    let points = height_px as f32 / native_ppp.max(0.25);
    (points / 760.0).clamp(1.0, 3.0)
}

/// The FPS counter label.
pub(crate) fn fps_label(ctx: &egui::Context, fps: f32) {
    egui::Area::new(egui::Id::new("rps5e-fps"))
        .fixed_pos(egui::pos2(8.0, 8.0))
        .order(egui::Order::Foreground)
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(egui::Color32::from_black_alpha(160))
                .corner_radius(4.0)
                .inner_margin(egui::Margin::symmetric(6, 2))
                .show(ui, |ui| {
                    let color = if fps >= 55.0 {
                        egui::Color32::from_rgb(120, 230, 120)
                    } else if fps >= 28.0 {
                        egui::Color32::from_rgb(240, 210, 90)
                    } else {
                        egui::Color32::from_rgb(240, 100, 90)
                    };
                    ui.label(egui::RichText::new(format!("{fps:.0} FPS")).monospace().strong().color(color));
                });
        });
}

/// The shader compile overlay ([`crate::osd::compile_overlay`]): the full
/// panel (title, counts, percent, time left, bar) over the window while
/// the game waits for its shaders, else a small corner indicator.
pub(crate) fn compile_overlay(ctx: &egui::Context, overlay: &crate::osd::CompileOverlay) {
    use crate::osd::CompileOverlay;
    match overlay {
        CompileOverlay::Full(p) => {
            let screen = ctx.screen_rect();
            egui::Area::new(egui::Id::new("rps5e-compile-backdrop")).order(egui::Order::Background).fixed_pos(screen.min).interactable(false).show(ctx, |ui| {
                ui.painter().rect_filled(screen, 0.0, egui::Color32::from_black_alpha(200));
            });
            let width = (screen.width() * 0.5).clamp(280.0, 560.0);
            egui::Area::new(egui::Id::new("rps5e-compile-panel"))
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .order(egui::Order::Foreground)
                .interactable(false)
                .show(ctx, |ui| {
                    egui::Frame::new()
                        .fill(egui::Color32::from_rgba_unmultiplied(24, 26, 30, 240))
                        .corner_radius(8.0)
                        .inner_margin(egui::Margin::same(20))
                        .show(ui, |ui| {
                            ui.set_width(width);
                            ui.label(egui::RichText::new("Compiling shaders").size(24.0).strong().color(egui::Color32::WHITE));
                            ui.add_space(8.0);
                            ui.label(egui::RichText::new(p.counts()).size(16.0).color(egui::Color32::from_gray(220)));
                            ui.add_space(6.0);
                            ui.add(egui::ProgressBar::new(p.fraction()).desired_width(width).desired_height(18.0).text(format!("{}%", p.percent())));
                            ui.add_space(6.0);
                            ui.label(egui::RichText::new(p.eta_text().unwrap_or_else(|| "estimating the time left".into())).size(15.0).color(egui::Color32::from_gray(200)));
                            if !p.note.is_empty() {
                                ui.add_space(4.0);
                                ui.label(egui::RichText::new(&p.note).size(13.0).color(egui::Color32::from_gray(150)));
                            }
                        });
                });
        }
        CompileOverlay::Background { done, total } => {
            let pct = if *total == 0 { 100 } else { done * 100 / total };
            egui::Area::new(egui::Id::new("rps5e-compile-indicator"))
                .anchor(egui::Align2::RIGHT_BOTTOM, [-10.0, -10.0])
                .order(egui::Order::Foreground)
                .interactable(false)
                .show(ctx, |ui| {
                    egui::Frame::new()
                        .fill(egui::Color32::from_black_alpha(170))
                        .corner_radius(4.0)
                        .inner_margin(egui::Margin::symmetric(8, 4))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.add(egui::Spinner::new().size(14.0).color(egui::Color32::from_rgb(240, 210, 90)));
                                ui.label(egui::RichText::new(format!("Compiling shaders: {done} / {total} ({pct}%)")).color(egui::Color32::from_gray(230)));
                            });
                            let frac = if *total == 0 { 1.0 } else { *done as f32 / *total as f32 };
                            ui.add(egui::ProgressBar::new(frac).desired_width(220.0).desired_height(6.0));
                        });
                });
        }
    }
}

/// The egui side of the game window (render thread): context, winit input
/// translation and when the UI last ran.
pub(crate) struct Host {
    pub ctx: egui::Context,
    state: egui_winit::State,
    pub last_run: Option<Instant>,
    /// Window input reached egui since the last UI frame.
    pub input_pending: bool,
}

impl Host {
    pub(crate) fn new(window: &Window, max_texture_side: usize) -> Self {
        let ctx = egui::Context::default();
        ctx.set_theme(egui::Theme::Dark);
        ctx.options_mut(|o| o.zoom_with_keyboard = false);
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            Some(max_texture_side),
        );
        Host { ctx, state, last_run: None, input_pending: false }
    }

    pub(crate) fn on_window_event(&mut self, window: &Window, event: &winit::event::WindowEvent) {
        let r = self.state.on_window_event(window, event);
        self.input_pending |= r.repaint;
    }

    /// Run one UI frame: the menu when it is open (else the controls
    /// overlay with `overlay`), the FPS counter when `fps` is given, the
    /// compile overlay when `compile` is.
    pub(crate) fn run(&mut self, window: &Window, open: bool, overlay: bool, fps: Option<f32>, compile: Option<&crate::osd::CompileOverlay>) -> (renderer::UiFrame, egui::TexturesDelta) {
        let zoom = zoom_for(window.inner_size().height, window.scale_factor() as f32);
        if (self.ctx.zoom_factor() - zoom).abs() > 0.01 {
            self.ctx.set_zoom_factor(zoom);
        }
        let mut raw = self.state.take_egui_input(window);
        if !open {
            // Closed: the game has the input.
            raw.events.clear();
        }
        let out = run_app(&mut raw, &self.ctx, open, overlay, |ctx| {
            if let Some(f) = fps {
                fps_label(ctx, f);
            }
            if let Some(c) = compile {
                compile_overlay(ctx, c);
            }
        });
        if open {
            self.state.handle_platform_output(window, out.platform_output);
        }
        self.last_run = Some(Instant::now());
        self.input_pending = false;
        let primitives = self.ctx.tessellate(out.shapes, out.pixels_per_point);
        (renderer::UiFrame { primitives, pixels_per_point: out.pixels_per_point }, out.textures_delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    struct Nop;
    impl MenuApp for Nop {
        fn ui(&mut self, ctx: &egui::Context) {
            egui::Area::new(egui::Id::new("t")).show(ctx, |ui| ui.label("menu"));
        }
        fn overlay(&mut self, ctx: &egui::Context) {
            egui::Area::new(egui::Id::new("o")).show(ctx, |ui| ui.label("controls"));
        }
    }

    #[test]
    fn open_state_and_listeners() {
        static OPENS: AtomicU32 = AtomicU32::new(0);
        static CLOSES: AtomicU32 = AtomicU32::new(0);
        set_open(true);
        assert!(!is_open(), "nothing to open before a menu is installed");
        on_open_change(|o| {
            if o {
                OPENS.fetch_add(1, Ordering::Relaxed);
            } else {
                CLOSES.fetch_add(1, Ordering::Relaxed);
            }
        });
        install(Box::new(Nop));
        assert!(toggle());
        set_open(true);
        assert_eq!(OPENS.load(Ordering::Relaxed), 1, "no event without a change");
        let ctx = egui::Context::default();
        // (egui lays new areas out invisibly in their first frame.)
        let _ = run_app(&mut egui::RawInput::default(), &ctx, true, false, |_| {});
        let out = run_app(&mut egui::RawInput::default(), &ctx, true, false, |_| {});
        assert!(!out.shapes.is_empty());
        // Closed: only the extra content (the FPS counter).
        let out = run_app(&mut egui::RawInput::default(), &ctx, false, false, |_| {});
        assert!(out.shapes.is_empty());
        let _ = run_app(&mut egui::RawInput::default(), &ctx, false, false, |c| fps_label(c, 60.0));
        let out = run_app(&mut egui::RawInput::default(), &ctx, false, false, |c| fps_label(c, 60.0));
        assert!(!out.shapes.is_empty());
        assert!(!toggle());
        assert_eq!(CLOSES.load(Ordering::Relaxed), 1);
        // The controls overlay draws while the menu is closed; opening the
        // menu hides it.
        set_overlay(true);
        assert!(overlay_shown());
        let _ = run_app(&mut egui::RawInput::default(), &ctx, false, true, |_| {});
        let out = run_app(&mut egui::RawInput::default(), &ctx, false, true, |_| {});
        assert!(!out.shapes.is_empty(), "the overlay draws");
        assert!(toggle());
        assert!(!overlay_shown());
        assert!(!toggle());
        assert!(!overlay_shown(), "the key was released while the menu was open");
    }

    #[test]
    fn compile_overlay_draws_panel_and_indicator() {
        use crate::osd::{CompileOverlay, CompileProgress};
        let ctx = egui::Context::default();
        let full = CompileOverlay::Full(CompileProgress { done: 10, total: 40, eta: Some(Duration::from_secs(30)), note: "The game starts when this finishes.".into() });
        let shapes = |o: &CompileOverlay| {
            let _ = run_app(&mut egui::RawInput::default(), &ctx, false, false, |c| compile_overlay(c, o));
            let out = run_app(&mut egui::RawInput::default(), &ctx, false, false, |c| compile_overlay(c, o));
            out.shapes.len()
        };
        let n_full = shapes(&full);
        let n_small = shapes(&CompileOverlay::Background { done: 3, total: 7 });
        assert!(n_full > 0 && n_small > 0);
        assert!(n_full > n_small, "the panel ({n_full} shapes) is more than the indicator ({n_small})");
    }

    #[test]
    fn zoom_keeps_the_menu_readable() {
        assert_eq!(zoom_for(720, 1.0), 1.0);
        assert!((zoom_for(1080, 1.0) - 1080.0 / 760.0).abs() < 1e-4);
        // HiDPI: the native scale already enlarges.
        assert!((zoom_for(2160, 2.0) - 1080.0 / 760.0).abs() < 1e-4);
        assert_eq!(zoom_for(8640, 1.0), 3.0);
    }
}
