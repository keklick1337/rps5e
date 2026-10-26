// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Input settings widgets shared by the launcher's settings window and the
//! in-game menu: device mode and controller choice, controller features,
//! analog and mouse settings, and the key / button binding table with its
//! "press a key" capture.

use eframe::egui::{self, Color32, RichText, Sense};
use ps5input::capture::{Capture, CaptureInput, CaptureState, CaptureTarget, Captured};
use ps5input::mapping::{
    default_bindings, default_keyboard_bindings, default_pad_bindings, pad_button_actions, Config, Control, DeviceMode, PadInput, Stick,
    StickMode, ACTIONS, MAX_PLAYERS,
};
use crate::dualsense;
use ps5gpu::input::{KeyCode, MouseButton};
use ps5input::mapping::Source;
use ps5input::sdl::DeviceInfo;
use std::time::{Duration, Instant};

/// Which binding set the rebind table shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingSet {
    KeyboardMouse,
    Keyboard,
    Controller,
}

pub struct PendingCapture {
    pub capture: Capture,
    /// Add an alternate instead of replacing.
    pub add: bool,
}

/// How the bindings are shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingView {
    /// The DualSense drawing with a callout per control.
    Picture,
    /// The table of every action.
    Table,
}

/// The binding table and its capture state.
pub struct BindingEditor {
    pub set: BindingSet,
    pub view: BindingView,
    pub capture: Option<PendingCapture>,
    /// Result of the last capture (inputs taken off other actions).
    pub note: Option<String>,
    /// Selected hotspot of the picture (index into `dualsense::SPOTS`).
    pub selected: Option<usize>,
    prev_modifiers: egui::Modifiers,
    prev_pad: Vec<(u32, Vec<PadInput>)>,
    /// Keys and mouse buttons held now (the picture lights their controls).
    held_keys: Vec<KeyCode>,
    held_mouse: Vec<MouseButton>,
}

impl Default for BindingEditor {
    fn default() -> Self {
        Self {
            set: BindingSet::KeyboardMouse,
            view: BindingView::Picture,
            capture: None,
            note: None,
            selected: None,
            prev_modifiers: egui::Modifiers::NONE,
            prev_pad: Vec::new(),
            held_keys: Vec::new(),
            held_mouse: Vec::new(),
        }
    }
}

impl BindingEditor {
    pub fn capturing(&self) -> bool {
        self.capture.is_some()
    }

    /// Start capturing the next input for `target`.
    pub fn start(&mut self, target: CaptureTarget, add: bool, now: Instant) {
        self.note = None;
        self.capture = Some(PendingCapture { capture: Capture::new(target, now), add });
    }

    /// Feed this frame's input to a pending capture: egui's key, pointer
    /// and wheel events plus the raw inputs held on each controller
    /// (`pad_now`). A completed capture is stored in `cfg`; returns whether
    /// `cfg` changed.
    pub fn feed(&mut self, ctx: &egui::Context, pad_now: Vec<(u32, Vec<PadInput>)>, cfg: &mut Config) -> bool {
        let prev_pad = std::mem::replace(&mut self.prev_pad, pad_now.clone());
        let (events, mods, keys_held, pointer_held) =
            ctx.input(|i| (i.events.clone(), i.modifiers, !i.keys_down.is_empty() || !i.modifiers.is_none(), i.pointer.any_down()));
        let scroll = ctx.input(|i| i.raw_scroll_delta.y);
        let prev_mods = std::mem::replace(&mut self.prev_modifiers, mods);
        let (any_key, mouse) = ctx.input(|i| {
            let mouse: Vec<MouseButton> = [
                egui::PointerButton::Primary,
                egui::PointerButton::Secondary,
                egui::PointerButton::Middle,
                egui::PointerButton::Extra1,
                egui::PointerButton::Extra2,
            ]
            .into_iter()
            .filter(|b| i.pointer.button_down(*b))
            .map(crate::keys::mouse_button)
            .collect();
            (!i.keys_down.is_empty(), mouse)
        });
        update_held_keys(&mut self.held_keys, &events, any_key);
        self.held_keys.retain(|k| !matches!(k, KeyCode::ShiftLeft | KeyCode::ControlLeft | KeyCode::AltLeft));
        self.held_keys.extend(crate::keys::pressed_modifiers(egui::Modifiers::NONE, mods));
        self.held_mouse = mouse;
        if self.capture.is_none() {
            return false;
        }
        let now = Instant::now();
        let pad_held = pad_now.iter().any(|(_, v)| !v.is_empty());
        // Armed once the click that started it and the inputs of the
        // target's kind are released: a controller button held while a key
        // is bound (or a key while a controller button is) does not hold
        // the capture back.
        let held = pointer_held
            || match self.capture.as_ref().map(|c| c.capture.target) {
                Some(CaptureTarget::Gamepad(_)) => pad_held,
                _ => keys_held,
            };
        let mut presses = key_presses(&events);
        presses.extend(crate::keys::pressed_modifiers(prev_mods, mods).into_iter().map(CaptureInput::Key));
        if scroll != 0.0 {
            presses.push(CaptureInput::Wheel(if scroll > 0.0 { 1 } else { -1 }));
        }
        presses.extend(pad_presses(&prev_pad, &pad_now));
        let changed = self.apply(presses, !held, now, cfg);
        ctx.request_repaint_after(Duration::from_millis(16));
        changed
    }

    /// Feed presses (and whether nothing is held) to the capture.
    pub fn apply(&mut self, presses: Vec<CaptureInput>, nothing_held: bool, now: Instant, cfg: &mut Config) -> bool {
        let Some(pc) = self.capture.as_mut() else { return false };
        // Escape cancels (Default puts it back on the actions it drives).
        if presses.contains(&CaptureInput::Key(KeyCode::Escape)) {
            self.capture = None;
            return false;
        }
        for p in presses {
            pc.capture.press(p, now);
            if pc.capture.is_finished() {
                break;
            }
        }
        pc.capture.set_idle(nothing_held);
        pc.capture.tick(now);
        match pc.capture.state.clone() {
            CaptureState::Done(c) => {
                let (target, add) = (pc.capture.target, pc.add);
                self.capture = None;
                let moved = apply_capture(cfg, target, c, add);
                self.note = (!moved.is_empty()).then(|| format!("Also removed from: {}", moved.join(", ")));
                true
            }
            CaptureState::TimedOut => {
                self.capture = None;
                false
            }
            _ => false,
        }
    }

    /// Keys, mouse buttons (with `mouse`) and controller inputs held now.
    pub fn held(&self, mouse: bool) -> (Vec<Source>, Vec<PadInput>) {
        let mut src: Vec<Source> = self.held_keys.iter().map(|k| Source::Key(*k)).collect();
        if mouse {
            src.extend(self.held_mouse.iter().map(|b| Source::Mouse(*b)));
        }
        let mut pad: Vec<PadInput> = self.prev_pad.iter().flat_map(|(_, v)| v.iter().copied()).collect();
        pad.sort();
        pad.dedup();
        (src, pad)
    }

    /// The set tabs.
    pub fn set_tabs(&mut self, ui: &mut egui::Ui) {
        let before = self.set;
        ui.selectable_value(&mut self.set, BindingSet::KeyboardMouse, "Keyboard + mouse");
        ui.selectable_value(&mut self.set, BindingSet::Keyboard, "Keyboard only");
        ui.selectable_value(&mut self.set, BindingSet::Controller, "Controller buttons");
        if self.set != before {
            self.capture = None;
            self.selected = None;
        }
    }

    /// The bindings of the selected set: the DualSense picture or the table.
    pub fn show(&mut self, ui: &mut egui::Ui, cfg: &mut Config) {
        ui.horizontal(|ui| {
            self.set_tabs(ui);
            ui.separator();
            ui.selectable_value(&mut self.view, BindingView::Picture, "Picture");
            ui.selectable_value(&mut self.view, BindingView::Table, "Table");
        });
        if self.view == BindingView::Picture {
            self.show_picture(ui, cfg);
        } else {
            self.show_table(ui, cfg);
        }
    }

    /// The DualSense picture: click a control to rebind it.
    pub fn show_picture(&mut self, ui: &mut egui::Ui, cfg: &mut Config) {
        let capturing = self.capture.is_some();
        let (held, pad) = self.held(false);
        let pressed = dualsense::pressed_controls(cfg, self.set, &held, &pad);
        let params = dualsense::PictureParams {
            cfg,
            set: self.set,
            selected: self.selected,
            capturing,
            pressed: &pressed,
            interactive: !capturing && ui.is_enabled(),
            max_width: 980.0,
        };
        let r = dualsense::picture(ui, &params);
        if let Some(i) = r.clicked {
            self.click_spot(i, Instant::now());
        }
        if self.set == BindingSet::KeyboardMouse {
            sticks_row(ui, cfg);
        }
        self.spot_detail(ui, cfg);
        self.extra_actions(ui, cfg);
    }

    /// The selected hotspot: its inputs and the buttons that change them.
    fn spot_detail(&mut self, ui: &mut egui::Ui, cfg: &mut Config) {
        let Some(sel) = self.selected else {
            if self.capture.is_none() {
                ui.label(
                    RichText::new("Click a button, stick or trigger to rebind it (a stick: what drives it). Held keys and buttons light up.")
                        .small()
                        .weak(),
                );
            }
            return;
        };
        let spot = &dualsense::SPOTS[sel];
        let target = dualsense::spot_target(spot, self.set);
        // A whole stick (or a stick direction): the stick's settings.
        if let Some(stick) = spot.stick.or(spot.control.and_then(Control::stick)) {
            if self.set != BindingSet::Controller && self.capture.is_none() {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.label(RichText::new(stick.label()).strong());
                    stick_settings(ui, cfg, stick, self.set == BindingSet::KeyboardMouse);
                });
                if spot.stick.is_some() {
                    return;
                }
            }
        }
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(spot.name).strong());
            match target {
                Some(t) => {
                    ui.label(RichText::new(sources_text(cfg, t)).monospace());
                }
                None => {
                    ui.label(RichText::new(dualsense::spot_help(spot, self.set).unwrap_or("not bindable in this set")).italics());
                }
            }
        });
        let mut start = None;
        if let Some(pc) = &self.capture {
            let mut cancel = false;
            ui.horizontal(|ui| {
                let what = capture_what(pc.capture.target);
                let secs = pc.capture.remaining(Instant::now()).as_secs_f32().ceil();
                ui.colored_label(Color32::YELLOW, format!("Press {what}... Esc cancels ({secs:.0} s)"));
                if let CaptureState::Rejected(why) = &pc.capture.state {
                    ui.colored_label(Color32::from_rgb(255, 110, 110), why);
                }
                cancel = ui.button("Cancel").clicked();
            });
            if cancel {
                self.capture = None;
            }
        } else if let Some(t) = target {
            ui.horizontal(|ui| {
                if ui.button("Set").on_hover_text("Replace with the next input").clicked() {
                    start = Some((t, false));
                }
                if ui.button("Add").on_hover_text("Add an alternate input").clicked() {
                    start = Some((t, true));
                }
                if ui.button("Clear").on_hover_text("Remove every input").clicked() {
                    clear_binding(cfg, t);
                }
                if ui.button("Default").on_hover_text("Back to the built-in inputs").clicked() {
                    default_binding(cfg, t);
                }
                if let Some(n) = &self.note {
                    ui.label(RichText::new(n).color(Color32::YELLOW));
                }
            });
        }
        if let Some((t, add)) = start {
            self.start(t, add, Instant::now());
        }
    }

    /// The motion and touch pad actions of the keyboard sets (not places on
    /// the drawing), with their settings.
    pub fn extra_actions(&mut self, ui: &mut egui::Ui, cfg: &mut Config) {
        if self.set == BindingSet::Controller {
            ui.label(RichText::new("A controller's own motion sensors and touch pad are passed to the game.").small().weak());
            return;
        }
        // A capture started from this list (no hotspot selected).
        if self.selected.is_none() {
            if let Some(pc) = &self.capture {
                let mut cancel = false;
                ui.horizontal(|ui| {
                    let secs = pc.capture.remaining(Instant::now()).as_secs_f32().ceil();
                    ui.colored_label(Color32::YELLOW, format!("Press {}... Esc cancels ({secs:.0} s)", capture_what(pc.capture.target)));
                    if let CaptureState::Rejected(why) = &pc.capture.state {
                        ui.colored_label(Color32::from_rgb(255, 110, 110), why);
                    }
                    cancel = ui.button("Cancel").clicked();
                });
                if cancel {
                    self.capture = None;
                }
            } else if let Some(n) = &self.note {
                ui.label(RichText::new(n).color(Color32::YELLOW));
            }
        }
        let kb = self.set == BindingSet::Keyboard;
        let mut start = None;
        let capturing = self.capture.is_some();
        egui::CollapsingHeader::new(RichText::new("Motion sensors").strong()).id_salt(("motion-actions", kb)).default_open(true).show(ui, |ui| {
            ui.label(
                RichText::new("Held tilt keys tip the controller and it comes back level on release; the game sees gravity and the turn rate like from a DualSense. While mouse motion is held, the mouse turns the controller (gyro aiming).")
                    .small()
                    .weak(),
            );
            binding_grid(ui, cfg, ("motion-grid", kb), &extra_rows(dualsense::MOTION_ACTIONS, kb), capturing, &mut start);
            motion_settings(ui, cfg);
        });
        egui::CollapsingHeader::new(RichText::new("Touch pad").strong()).id_salt(("touch-actions", kb)).default_open(true).show(ui, |ui| {
            ui.label(
                RichText::new("Touch pad mode turns numpad 1-9 into touch zones (7 8 9 is the top row; two keys are two fingers) and lets the mouse move a finger: the left button touches, the right one clicks. The touch pad click key clicks under the fingers that are down.")
                    .small()
                    .weak(),
            );
            binding_grid(ui, cfg, ("touch-grid", kb), &extra_rows(dualsense::TOUCH_ACTIONS, kb), capturing, &mut start);
            touch_settings(ui, cfg);
        });
        egui::CollapsingHeader::new(RichText::new("Emulator keys").strong()).id_salt(("emulator-actions", kb)).default_open(false).show(ui, |ui| {
            ui.label(RichText::new("Window keys of the game: the menu, the controls overlay (shown while held), mouse capture, overlays, fullscreen, screenshot, mute.").small().weak());
            binding_grid(ui, cfg, ("emulator-grid", kb), &extra_rows(EMULATOR_ACTIONS, kb), capturing, &mut start);
        });
        if let Some((t, add)) = start {
            self.selected = None;
            self.start(t, add, Instant::now());
        }
    }

    /// A hotspot was clicked: select it and wait for its new input (a
    /// pending capture is kept).
    pub fn click_spot(&mut self, i: usize, now: Instant) {
        if self.capture.is_some() || i >= dualsense::SPOTS.len() {
            return;
        }
        self.selected = Some(i);
        if let Some(t) = dualsense::spot_target(&dualsense::SPOTS[i], self.set) {
            self.start(t, false, now);
        }
    }

    /// The binding table of the selected set.
    pub fn show_table(&mut self, ui: &mut egui::Ui, cfg: &mut Config) {
        ui.label(
            RichText::new("Set replaces an action's inputs (the input is taken off other actions), Add keeps them. Double-click a row to set it.")
                .small(),
        );
        if let Some(pc) = &self.capture {
            let what = capture_what(pc.capture.target);
            ui.horizontal(|ui| {
                ui.colored_label(
                    Color32::YELLOW,
                    format!("Press {what}... Esc cancels ({:.0} s)", pc.capture.remaining(Instant::now()).as_secs_f32().ceil()),
                );
                if let CaptureState::Rejected(why) = &pc.capture.state {
                    ui.colored_label(Color32::from_rgb(255, 110, 110), why);
                }
            });
            if ui.button("Cancel").clicked() {
                self.capture = None;
            }
        } else if let Some(n) = &self.note {
            ui.label(RichText::new(n).color(Color32::YELLOW));
        }
        if self.set == BindingSet::KeyboardMouse {
            sticks_row(ui, cfg);
        }
        let rows = binding_rows(self.set, cfg);
        let capturing = self.capture.is_some();
        let mut start: Option<(CaptureTarget, bool)> = None;
        egui::Grid::new(("binds", self.set as u8)).num_columns(3).striped(true).spacing([12.0, 4.0]).show(ui, |ui| {
            // Sticks not on their keys: one row each in place of four.
            for stick in Stick::BOTH {
                if let Some(mode) = dualsense::stick_mode(cfg, stick, self.set).filter(|m| *m != StickMode::Keys) {
                    ui.label(stick.label());
                    ui.label(RichText::new(dualsense::stick_text(cfg, mode)).monospace());
                    if mode == StickMode::Mouse {
                        ui.horizontal(|ui| {
                            ui.add(egui::Slider::new(&mut cfg.mouse_sensitivity, 0.1..=10.0).logarithmic(true).text("sensitivity"));
                            ui.checkbox(&mut cfg.mouse_invert_x, "invert X");
                            ui.checkbox(&mut cfg.mouse_invert_y, "invert Y");
                        });
                    } else {
                        ui.label(RichText::new("only a controller moves it").small().weak());
                    }
                    ui.end_row();
                }
            }
            binding_rows_ui(ui, cfg, &rows, capturing, &mut start);
        });
        if self.set != BindingSet::Controller {
            ui.add_space(6.0);
            ui.label(RichText::new("Motion sensors and touch pad").strong());
            motion_settings(ui, cfg);
            touch_settings(ui, cfg);
        }
        if let Some((target, add)) = start {
            self.start(target, add, Instant::now());
        }
    }
}

/// The emulator's window keys (not pad input).
pub const EMULATOR_ACTIONS: &[Control] = &[
    Control::ToggleMenu,
    Control::ShowControls,
    Control::ToggleMouse,
    Control::ReleaseMouse,
    Control::TogglePerfOverlay,
    Control::ToggleFullscreen,
    Control::Create,
    Control::Mute,
    Control::SpeakerVolume,
    Control::VibrationIntensity,
    Control::TriggerEffectIntensity,
];

/// Rows of motion or touch actions in a keyboard set (`kb`: keyboard only,
/// without the mouse actions).
pub fn extra_rows(list: &[Control], kb: bool) -> Vec<(String, CaptureTarget)> {
    list.iter()
        .filter(|c| !(kb && c.mouse_only()))
        .map(|&c| (c.name().to_string(), if kb { CaptureTarget::Keyboard(c) } else { CaptureTarget::KeyboardMouse(c) }))
        .collect()
}

/// A grid of binding rows.
fn binding_grid(
    ui: &mut egui::Ui,
    cfg: &mut Config,
    id: impl std::hash::Hash,
    rows: &[(String, CaptureTarget)],
    capturing: bool,
    start: &mut Option<(CaptureTarget, bool)>,
) {
    egui::Grid::new(id).num_columns(3).striped(true).spacing([12.0, 4.0]).show(ui, |ui| binding_rows_ui(ui, cfg, rows, capturing, start));
}

/// One row per action: its name (double-click to set), inputs and the
/// Set / Add / Clear / Default buttons.
fn binding_rows_ui(ui: &mut egui::Ui, cfg: &mut Config, rows: &[(String, CaptureTarget)], capturing: bool, start: &mut Option<(CaptureTarget, bool)>) {
    for (name, target) in rows {
        let target = *target;
        let r = ui.add(egui::Label::new(action_label(name)).sense(Sense::click()));
        if r.double_clicked() && !capturing {
            *start = Some((target, false));
        }
        ui.label(RichText::new(sources_text(cfg, target)).monospace());
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!capturing, |ui| {
                if ui.small_button("Set").on_hover_text("Replace with the next input").clicked() {
                    *start = Some((target, false));
                }
                if ui.small_button("Add").on_hover_text("Add an alternate input").clicked() {
                    *start = Some((target, true));
                }
                if ui.small_button("Clear").clicked() {
                    clear_binding(cfg, target);
                }
                if ui.small_button("Default").clicked() {
                    default_binding(cfg, target);
                }
            });
        });
        ui.end_row();
    }
}

/// Tilt amount and speed, and the mouse motion sensitivity.
pub fn motion_settings(ui: &mut egui::Ui, cfg: &mut Config) {
    let m = &mut cfg.motion;
    egui::Grid::new("motion-settings").num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
        ui.label("Tilt");
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut m.tilt_angle, 1.0..=90.0).suffix("\u{b0}").fixed_decimals(0).text("angle"));
            ui.add(egui::Slider::new(&mut m.tilt_time, 0.02..=2.0).logarithmic(true).suffix(" s").fixed_decimals(2).text("time"));
        });
        ui.end_row();
        ui.label("Mouse motion");
        ui.add(egui::Slider::new(&mut m.mouse_sensitivity, 0.05..=20.0).logarithmic(true).fixed_decimals(2).text("sensitivity"));
        ui.end_row();
    });
    if ui.small_button("Default motion settings").clicked() {
        cfg.motion = Default::default();
    }
}

/// Touch pad mode settings.
pub fn touch_settings(ui: &mut egui::Ui, cfg: &mut Config) {
    ui.horizontal(|ui| {
        ui.label("Mouse finger");
        ui.add(egui::Slider::new(&mut cfg.touch_mouse_sensitivity, 0.05..=20.0).logarithmic(true).fixed_decimals(2).text("sensitivity"));
    });
}

fn capture_what(t: CaptureTarget) -> &'static str {
    match t {
        CaptureTarget::Gamepad(_) => "a controller button",
        CaptureTarget::Keyboard(_) => "a key",
        CaptureTarget::KeyboardMouse(_) => "a key, mouse button or wheel",
    }
}

/// Track held keys (physical positions) from key events; `any_key`: egui
/// still reports a key down (none: everything was released, focus lost).
pub fn update_held_keys(held: &mut Vec<KeyCode>, events: &[egui::Event], any_key: bool) {
    for ev in events {
        if let egui::Event::Key { physical_key: Some(k), pressed, .. } = ev {
            if let Some(code) = crate::keys::key_code(*k) {
                held.retain(|c| *c != code);
                if *pressed {
                    held.push(code);
                }
            }
        }
    }
    if !any_key {
        held.clear();
    }
}

/// Release every key egui still holds (`keys_down`) when the window's
/// focus changes in `raw` or when `force` is set (the in-game menu opens).
///
/// egui holds a key until its release event reaches it and never drops it
/// on its own. A key released while another window had the focus, or
/// after the in-game menu closed (the game gets the release), stays held
/// for good: its next presses come as repeats, and a binding capture,
/// which starts once every key is up, waits for nothing and times out.
/// The releases go before this frame's events.
pub fn release_stale_keys(keys_down: &std::collections::HashSet<egui::Key>, raw: &mut egui::RawInput, force: bool) {
    let focus_changed = raw.events.iter().any(|e| matches!(e, egui::Event::WindowFocused(_)));
    if keys_down.is_empty() || !(force || focus_changed) {
        return;
    }
    let mut keys: Vec<egui::Key> = keys_down.iter().copied().collect();
    keys.sort();
    let releases = keys
        .into_iter()
        .map(|key| egui::Event::Key { key, physical_key: Some(key), pressed: false, repeat: false, modifiers: egui::Modifiers::NONE });
    raw.events.splice(0..0, releases);
}

/// Key and pointer presses of egui events. Bindings are key positions: the
/// logical key depends on the active layout (C types 'с' on a Russian one),
/// so a key without a physical code is not captured.
pub fn key_presses(events: &[egui::Event]) -> Vec<CaptureInput> {
    let mut presses = Vec::new();
    for ev in events {
        match ev {
            egui::Event::Key { physical_key: Some(physical_key), pressed: true, repeat: false, .. } => {
                if let Some(code) = crate::keys::key_code(*physical_key) {
                    presses.push(CaptureInput::Key(code));
                }
            }
            egui::Event::PointerButton { button, pressed: true, .. } => {
                presses.push(CaptureInput::Mouse(crate::keys::mouse_button(*button)));
            }
            _ => {}
        }
    }
    presses
}

/// Controller inputs newly held in `now` compared with `before`.
pub fn pad_presses(before: &[(u32, Vec<PadInput>)], now: &[(u32, Vec<PadInput>)]) -> Vec<CaptureInput> {
    let mut v = Vec::new();
    for (id, inputs) in now {
        let prev = before.iter().find(|(i, _)| i == id).map(|(_, v)| v.as_slice()).unwrap_or(&[]);
        v.extend(inputs.iter().filter(|p| !prev.contains(p)).map(|p| CaptureInput::Pad(*p)));
    }
    v
}

/// Rows of a binding set: action name and capture target. The directions
/// of a stick in mouse or controller-only mode are left out.
pub fn binding_rows(set: BindingSet, cfg: &Config) -> Vec<(String, CaptureTarget)> {
    let rows: Vec<(String, CaptureTarget)> = match set {
        BindingSet::KeyboardMouse => ACTIONS.iter().map(|&(n, c)| (n.to_string(), CaptureTarget::KeyboardMouse(c))).collect(),
        BindingSet::Keyboard => ACTIONS
            .iter()
            .filter(|(_, c)| !c.mouse_only())
            .map(|&(n, c)| (n.to_string(), CaptureTarget::Keyboard(c)))
            .collect(),
        BindingSet::Controller => pad_button_actions().map(|(n, bit)| (n.to_string(), CaptureTarget::Gamepad(bit))).collect(),
    };
    rows.into_iter()
        .filter(|(_, t)| match t {
            CaptureTarget::KeyboardMouse(c) | CaptureTarget::Keyboard(c) => {
                c.stick().is_none_or(|s| dualsense::stick_mode(cfg, s, set) == Some(StickMode::Keys))
            }
            CaptureTarget::Gamepad(_) => true,
        })
        .collect()
}

/// "Left stick [mode]  Right stick [mode]" (keyboard + mouse set).
pub fn sticks_row(ui: &mut egui::Ui, cfg: &mut Config) {
    ui.horizontal_wrapped(|ui| {
        for stick in Stick::BOTH {
            ui.label(stick.label());
            stick_mode_combo(ui, cfg, stick);
            ui.add_space(12.0);
        }
        ui.label(RichText::new("The mouse drives one stick.").small().weak());
    });
    ui.horizontal_wrapped(|ui| {
        if Stick::BOTH.iter().any(|&s| cfg.stick_mode(s) == StickMode::Mouse) {
            ui.label("Mouse-look");
            ui.add(egui::Slider::new(&mut cfg.mouse_sensitivity, 0.1..=10.0).logarithmic(true).text("sensitivity"));
            ui.checkbox(&mut cfg.mouse_invert_y, "invert Y");
            ui.add_space(12.0);
        }
        ui.checkbox(&mut cfg.swap_sticks, "Swap sticks").on_hover_text(
            "The keys and the mouse drive the other stick: WASD moves the right stick, the mouse the left one. A controller's sticks are not swapped.",
        );
        if cfg.swap_sticks {
            ui.label(RichText::new("the left stick's keys and mouse move the right stick").small().color(Color32::from_rgb(220, 180, 90)));
        }
    });
}

pub fn stick_mode_combo(ui: &mut egui::Ui, cfg: &mut Config, stick: Stick) {
    let mut mode = cfg.stick_mode(stick);
    egui::ComboBox::from_id_salt(("stick-mode", stick as u8)).selected_text(mode.label()).width(150.0).show_ui(ui, |ui| {
        for m in StickMode::ALL {
            ui.selectable_value(&mut mode, m, m.label());
        }
    });
    if mode != cfg.stick_mode(stick) {
        cfg.set_stick_mode(stick, mode);
    }
}

/// Everything about one stick: what drives it (`kbm`: the keyboard + mouse
/// set, where the mouse is a choice), the mouse-look settings when the
/// mouse does, and the controller stick's dead zone and curve.
pub fn stick_settings(ui: &mut egui::Ui, cfg: &mut Config, stick: Stick, kbm: bool) {
    egui::Grid::new(("stick", stick as u8)).num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
        if kbm {
            ui.label("Driven by");
            ui.horizontal(|ui| {
                let mut mode = cfg.stick_mode(stick);
                for m in StickMode::ALL {
                    ui.selectable_value(&mut mode, m, m.label());
                }
                if mode != cfg.stick_mode(stick) {
                    cfg.set_stick_mode(stick, mode);
                }
            });
            ui.end_row();
        }
        if kbm && cfg.stick_mode(stick) == StickMode::Mouse {
            mouse_look_settings(ui, cfg);
        }
        let shape = if stick == Stick::Left { &mut cfg.left_stick } else { &mut cfg.right_stick };
        ui.label("Controller stick");
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut shape.deadzone, 0.0..=0.9).text("dead zone").fixed_decimals(2));
            ui.add(egui::Slider::new(&mut shape.curve, 0.2..=5.0).logarithmic(true).text("curve").fixed_decimals(2));
        });
        ui.end_row();
    });
    if ui.button("Default").on_hover_text("This stick's built-in settings").clicked() {
        cfg.reset_stick(stick);
    }
}

/// Mouse-look rows: sensitivity, inversion and the capture keys.
fn mouse_look_settings(ui: &mut egui::Ui, cfg: &mut Config) {
    ui.label("Mouse");
    ui.horizontal(|ui| {
        ui.add(egui::Slider::new(&mut cfg.mouse_sensitivity, 0.1..=10.0).logarithmic(true).text("sensitivity"));
        ui.checkbox(&mut cfg.mouse_invert_x, "invert X");
        ui.checkbox(&mut cfg.mouse_invert_y, "invert Y");
    });
    ui.end_row();
    let keys: Vec<String> = cfg.sources_of(Control::ToggleMouse, false).iter().map(|s| dualsense::short_source(&s.to_string())).collect();
    if !keys.is_empty() {
        ui.label("");
        ui.label(RichText::new(format!("The mouse moves the stick while captured: {} captures it.", keys.join(" or "))).small().weak());
        ui.end_row();
    }
}

/// Device mode, local players and their controllers.
pub fn device_settings(ui: &mut egui::Ui, cfg: &mut Config, devices: &[DeviceInfo], sdl_error: Option<&str>, players_editable: bool) {
    egui::Grid::new("dev").num_columns(2).spacing([16.0, 8.0]).show(ui, |ui| {
        ui.label("Device mode");
        egui::ComboBox::from_id_salt("mode").selected_text(cfg.mode.label()).width(300.0).show_ui(ui, |ui| {
            for m in DeviceMode::ALL {
                ui.selectable_value(&mut cfg.mode, m, m.label());
            }
        });
        ui.end_row();
        ui.label("Players");
        let mut n = cfg.players as u32;
        ui.add_enabled_ui(players_editable, |ui| {
            ui.add(egui::Slider::new(&mut n, 1..=MAX_PLAYERS as u32).text("local players"))
                .on_disabled_hover_text("Takes effect when the game starts");
        });
        cfg.players = n as usize;
        ui.end_row();
        for p in 0..cfg.players {
            if p == 0 && !cfg.mode.uses_controller() {
                continue;
            }
            ui.label(format!("Player {} controller", p + 1));
            controller_picker(ui, p, &mut cfg.controllers[p], devices);
            ui.end_row();
        }
    });
    if let Some(e) = sdl_error {
        ui.label(RichText::new(format!("Controllers unavailable: {e}")).small());
    }
}

pub fn controller_picker(ui: &mut egui::Ui, player: usize, selector: &mut String, devices: &[DeviceInfo]) {
    let current = if selector.is_empty() {
        "Any free controller".to_string()
    } else {
        match devices.iter().find(|d| d.selector() == *selector) {
            Some(d) => format!("{} ({})", d.name, d.kind.label()),
            None => format!("{selector} (not connected)"),
        }
    };
    egui::ComboBox::from_id_salt(("player", player)).selected_text(current).width(300.0).show_ui(ui, |ui| {
        ui.selectable_value(selector, String::new(), "Any free controller");
        for d in devices {
            let label = if d.serial.is_empty() { d.name.clone() } else { format!("{} [{}]", d.name, d.serial) };
            ui.selectable_value(selector, d.selector(), label);
        }
    });
}

pub fn controller_settings(ui: &mut egui::Ui, cfg: &mut Config) {
    let f = &mut cfg.features;
    egui::Grid::new("controller").num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
        ui.label("Light bar colour");
        ui.horizontal(|ui| {
            let mut fixed = f.lightbar_color.is_some();
            if ui.checkbox(&mut fixed, "fixed (overrides the game's colour)").changed() {
                f.lightbar_color = fixed.then_some(ps5input::sdl::DEFAULT_LIGHTBAR);
            }
            if let Some(c) = f.lightbar_color.as_mut() {
                egui::color_picker::color_edit_button_srgb(ui, c);
                ui.label(RichText::new(ps5input::mapping::color_string(*c)).monospace());
            }
        });
        ui.end_row();
        ui.label("Speaker volume");
        ui.add(egui::Slider::new(&mut f.speaker_volume, 0..=100).text("(50 = unity)"));
        ui.end_row();
        ui.label("Vibration intensity");
        ui.add(egui::Slider::new(&mut f.vibration_intensity, 0..=100));
        ui.end_row();
    });
}

pub fn analog_settings(ui: &mut egui::Ui, cfg: &mut Config) {
    egui::Grid::new("analog").num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
        for stick in Stick::BOTH {
            ui.label(stick.label());
            ui.horizontal(|ui| {
                stick_mode_combo(ui, cfg, stick);
                let shape = if stick == Stick::Left { &mut cfg.left_stick } else { &mut cfg.right_stick };
                ui.add(egui::Slider::new(&mut shape.deadzone, 0.0..=0.9).text("dead zone").fixed_decimals(2));
                ui.add(egui::Slider::new(&mut shape.curve, 0.2..=5.0).logarithmic(true).text("curve").fixed_decimals(2));
                if ui.small_button("Default").on_hover_text("This stick's built-in settings").clicked() {
                    cfg.reset_stick(stick);
                }
            });
            ui.end_row();
        }
        ui.label("Triggers");
        ui.horizontal(|ui| {
            let shape = &mut cfg.triggers;
            ui.add(egui::Slider::new(&mut shape.deadzone, 0.0..=0.9).text("dead zone").fixed_decimals(2));
            ui.add(egui::Slider::new(&mut shape.curve, 0.2..=5.0).logarithmic(true).text("curve").fixed_decimals(2));
        });
        ui.end_row();
        mouse_look_settings(ui, cfg);
        ui.label("Swap sticks");
        ui.checkbox(&mut cfg.swap_sticks, "keys and mouse drive the other stick (WASD on the right, the mouse on the left)");
        ui.end_row();
        ui.label("Guest mouse");
        ui.checkbox(&mut cfg.guest_mouse, "Expose the host mouse to games (libSceMouse)");
        ui.end_row();
        let f = &mut cfg.features;
        ui.label("Controllers");
        ui.checkbox(&mut cfg.gamepad, "Use game controllers (SDL3)");
        ui.end_row();
        ui.label("Rumble");
        ui.horizontal(|ui| {
            ui.checkbox(&mut f.rumble, "enabled");
            ui.add(egui::Slider::new(&mut f.rumble_strength, 0.0..=2.0).text("strength"));
        });
        ui.end_row();
        ui.label("DualSense / DS4");
        ui.horizontal(|ui| {
            ui.checkbox(&mut f.lightbar, "light bar");
            ui.checkbox(&mut f.adaptive_triggers, "adaptive triggers");
            ui.checkbox(&mut f.motion, "motion sensors");
            ui.checkbox(&mut f.touchpad, "touch pad");
        });
        ui.end_row();
    });
}

/// Friendlier names of the action keys.
pub fn action_label(name: &str) -> String {
    match name {
        "SpeakerVolume" => "Speaker volume (cycle)".into(),
        "VibrationIntensity" => "Vibration intensity (cycle)".into(),
        "TriggerEffectIntensity" => "Trigger effect intensity (cycle)".into(),
        "TouchPad" => "Touch pad click".into(),
        "TouchLeft" => "Touch pad click, left half".into(),
        "TouchRight" => "Touch pad click, right half".into(),
        "TiltForward" => "Tilt forward (far edge down)".into(),
        "TiltBack" => "Tilt back (far edge up)".into(),
        "TiltLeft" => "Tilt left (left grip down)".into(),
        "TiltRight" => "Tilt right (right grip down)".into(),
        "TurnLeft" => "Turn left (yaw)".into(),
        "TurnRight" => "Turn right (yaw)".into(),
        "MouseMotion" => "Mouse motion (hold: the mouse turns the controller)".into(),
        "TouchpadMode" => "Touch pad mode (numpad zones, mouse finger)".into(),
        "TapLeft" => "Touch left (no click)".into(),
        "TapCenter" => "Touch middle (no click)".into(),
        "TapRight" => "Touch right (no click)".into(),
        "SwipeUp" => "Swipe up".into(),
        "SwipeDown" => "Swipe down".into(),
        "SwipeLeft" => "Swipe left".into(),
        "SwipeRight" => "Swipe right".into(),
        "TwoFingerTouch" => "Two fingers (hold)".into(),
        "Create" => "Create (screenshot)".into(),
        "Mute" => "Mute (game sound)".into(),
        "ToggleMenu" => "In-game menu".into(),
        "ShowControls" => "Show controls (hold)".into(),
        "ToggleMouse" => "Capture mouse (mouse-look)".into(),
        "ReleaseMouse" => "Release mouse".into(),
        "ToggleFullscreen" => "Fullscreen".into(),
        "TogglePerfOverlay" => "Performance overlay".into(),
        _ if name.starts_with("LeftStick") => format!("Left stick {}", name["LeftStick".len()..].to_ascii_lowercase()),
        _ if name.starts_with("RightStick") => format!("Right stick {}", name["RightStick".len()..].to_ascii_lowercase()),
        _ => name.to_string(),
    }
}

pub fn sources_text(cfg: &Config, target: CaptureTarget) -> String {
    let parts: Vec<String> = match target {
        CaptureTarget::KeyboardMouse(c) => cfg.sources_of(c, false).iter().map(|s| s.to_string()).collect(),
        CaptureTarget::Keyboard(c) => cfg.sources_of(c, true).iter().map(|s| s.to_string()).collect(),
        CaptureTarget::Gamepad(bit) => cfg.pad_inputs_of(bit).iter().map(|p| p.to_config_string()).collect(),
    };
    if parts.is_empty() {
        "-".into()
    } else {
        parts.join(", ")
    }
}

/// Store a captured input: replace the action's inputs, or add one. A
/// replacing input is taken off the other actions of the same set (one
/// input, one action), except Escape-style "release mouse" sharing.
/// Returns the actions it was taken from.
pub fn apply_capture(cfg: &mut Config, target: CaptureTarget, captured: Captured, add: bool) -> Vec<String> {
    let mut moved = Vec::new();
    match (target, captured) {
        (CaptureTarget::KeyboardMouse(c) | CaptureTarget::Keyboard(c), Captured::Source(src)) => {
            let kb = matches!(target, CaptureTarget::Keyboard(_));
            let mut list = if add { cfg.sources_of(c, kb) } else { Vec::new() };
            if !list.contains(&src) {
                list.push(src);
            }
            if cfg.set_sources(c, &list, kb).is_ok() && !add && c != Control::ReleaseMouse {
                for &(name, other) in ACTIONS {
                    if other == c || other == Control::ReleaseMouse {
                        continue;
                    }
                    let theirs = cfg.sources_of(other, kb);
                    if theirs.contains(&src) {
                        let rest: Vec<_> = theirs.into_iter().filter(|s| *s != src).collect();
                        if cfg.set_sources(other, &rest, kb).is_ok() {
                            moved.push(name.to_string());
                        }
                    }
                }
            }
        }
        (CaptureTarget::Gamepad(bit), Captured::Pad(p)) => {
            let mut list = if add { cfg.pad_inputs_of(bit) } else { Vec::new() };
            if !list.contains(&p) {
                list.push(p);
            }
            cfg.set_pad_inputs(bit, &list);
            if !add {
                for (name, other) in pad_button_actions() {
                    let theirs = cfg.pad_inputs_of(other);
                    if other != bit && theirs.contains(&p) {
                        let rest: Vec<_> = theirs.into_iter().filter(|x| *x != p).collect();
                        cfg.set_pad_inputs(other, &rest);
                        moved.push(name.to_string());
                    }
                }
            }
        }
        _ => {}
    }
    moved
}

pub fn clear_binding(cfg: &mut Config, target: CaptureTarget) {
    match target {
        CaptureTarget::KeyboardMouse(c) => {
            let _ = cfg.set_sources(c, &[], false);
        }
        CaptureTarget::Keyboard(c) => {
            let _ = cfg.set_sources(c, &[], true);
        }
        CaptureTarget::Gamepad(bit) => cfg.set_pad_inputs(bit, &[]),
    }
}

pub fn default_binding(cfg: &mut Config, target: CaptureTarget) {
    match target {
        CaptureTarget::KeyboardMouse(c) => {
            let d: Vec<_> = default_bindings().into_iter().filter(|b| b.control == c).map(|b| b.source).collect();
            let _ = cfg.set_sources(c, &d, false);
        }
        CaptureTarget::Keyboard(c) => {
            let d: Vec<_> = default_keyboard_bindings().into_iter().filter(|b| b.control == c).map(|b| b.source).collect();
            let _ = cfg.set_sources(c, &d, true);
        }
        CaptureTarget::Gamepad(bit) => {
            let d: Vec<_> = default_pad_bindings().into_iter().filter(|b| b.bit == bit).map(|b| b.input).collect();
            cfg.set_pad_inputs(bit, &d);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps5input::button;

    #[test]
    fn capture_results_replace_or_add() {
        let mut cfg = Config::default();
        let cross = Control::Button(button::CROSS);
        apply_capture(&mut cfg, CaptureTarget::KeyboardMouse(cross), Captured::Source(Source::Key(KeyCode::KeyX)), false);
        assert_eq!(cfg.sources_of(cross, false), [Source::Key(KeyCode::KeyX)]);
        apply_capture(&mut cfg, CaptureTarget::KeyboardMouse(cross), Captured::Source(Source::Key(KeyCode::KeyZ)), true);
        assert_eq!(cfg.sources_of(cross, false), [Source::Key(KeyCode::KeyX), Source::Key(KeyCode::KeyZ)]);
        apply_capture(&mut cfg, CaptureTarget::Gamepad(button::CROSS), Captured::Pad(PadInput(1)), false);
        assert_eq!(cfg.pad_inputs_of(button::CROSS), [PadInput(1)]);
        clear_binding(&mut cfg, CaptureTarget::Keyboard(cross));
        assert!(cfg.sources_of(cross, true).is_empty());
        default_binding(&mut cfg, CaptureTarget::Keyboard(cross));
        assert_eq!(cfg.sources_of(cross, true).len(), 2);
        default_binding(&mut cfg, CaptureTarget::Gamepad(button::CROSS));
        assert_eq!(cfg.pad_inputs_of(button::CROSS), [PadInput(0)]);
    }

    #[test]
    fn replacing_takes_the_input_off_other_actions() {
        let mut cfg = Config::default();
        let circle = Control::Button(button::CIRCLE);
        let cross = Control::Button(button::CROSS);
        let src = cfg.sources_of(cross, false)[0];
        let moved = apply_capture(&mut cfg, CaptureTarget::KeyboardMouse(circle), Captured::Source(src), false);
        assert_eq!(moved, ["Cross"]);
        assert!(!cfg.sources_of(cross, false).contains(&src));
        assert_eq!(cfg.sources_of(circle, false), [src]);
        // Adding keeps sharing.
        let moved = apply_capture(&mut cfg, CaptureTarget::KeyboardMouse(cross), Captured::Source(src), true);
        assert!(moved.is_empty());
        assert!(cfg.sources_of(circle, false).contains(&src));
        // Escape keeps releasing the mouse when bound to Options.
        let esc = Source::Key(KeyCode::Escape);
        let moved = apply_capture(&mut cfg, CaptureTarget::KeyboardMouse(Control::Button(button::OPTIONS)), Captured::Source(esc), false);
        assert!(!moved.contains(&"ReleaseMouse".to_string()));
        assert!(cfg.sources_of(Control::ReleaseMouse, false).contains(&esc));
        // Controller buttons.
        let p = cfg.pad_inputs_of(button::CROSS)[0];
        let moved = apply_capture(&mut cfg, CaptureTarget::Gamepad(button::CIRCLE), Captured::Pad(p), false);
        assert_eq!(moved, ["Cross"]);
        assert!(cfg.pad_inputs_of(button::CROSS).is_empty());
    }

    #[test]
    fn mapping_names_are_shown() {
        let cfg = Config::default();
        assert_eq!(sources_text(&cfg, CaptureTarget::KeyboardMouse(Control::ToggleFullscreen)), "KEY:F11");
        assert_eq!(sources_text(&cfg, CaptureTarget::KeyboardMouse(Control::ToggleMenu)), "KEY:F1");
        assert_eq!(sources_text(&cfg, CaptureTarget::Gamepad(button::CROSS)), "BUTTON:South");
        assert_eq!(sources_text(&cfg, CaptureTarget::KeyboardMouse(Control::SpeakerVolume)), "-");
        assert_eq!(action_label("SpeakerVolume"), "Speaker volume (cycle)");
        assert_eq!(action_label("RightStickUp"), "Right stick up");
    }

    #[test]
    fn table_rows_follow_stick_modes_and_resets() {
        let has = |rows: &[(String, CaptureTarget)], c: Control| {
            rows.iter().any(|(_, t)| matches!(t, CaptureTarget::KeyboardMouse(x) | CaptureTarget::Keyboard(x) if *x == c))
        };
        let mut cfg = Config::default();
        let rows = binding_rows(BindingSet::KeyboardMouse, &cfg);
        assert!(has(&rows, Control::LeftStickUp) && !has(&rows, Control::RightStickUp));
        assert!(has(&binding_rows(BindingSet::Keyboard, &cfg), Control::RightStickUp), "keyboard only: keys");
        cfg.set_stick_mode(Stick::Left, StickMode::Mouse);
        cfg.mouse_sensitivity = 3.0;
        let rows = binding_rows(BindingSet::KeyboardMouse, &cfg);
        assert!(!has(&rows, Control::LeftStickUp) && has(&rows, Control::RightStickUp));
        // Per-binding default, per-stick default, then everything.
        let up = CaptureTarget::KeyboardMouse(Control::RightStickUp);
        clear_binding(&mut cfg, up);
        default_binding(&mut cfg, up);
        assert_eq!(cfg.sources_of(Control::RightStickUp, false), [Source::Key(KeyCode::KeyI)]);
        cfg.reset_stick(Stick::Left);
        assert_eq!((cfg.left_stick_mode, cfg.right_stick_mode), (StickMode::Keys, StickMode::Keys));
        cfg.reset_stick(Stick::Right);
        let d = Config::default();
        assert_eq!((cfg.left_stick_mode, cfg.right_stick_mode, cfg.mouse_sensitivity), (d.left_stick_mode, d.right_stick_mode, 1.0));
        // "Reset to defaults" is the built-in configuration.
        cfg = Config::default();
        assert!(!has(&binding_rows(BindingSet::KeyboardMouse, &cfg), Control::RightStickUp));
    }

    #[test]
    fn editor_capture_flow_rebinds() {
        // Rebinding through the editor: the click that started it is not
        // taken; the first press after everything is released is.
        let t = Instant::now();
        let mut cfg = Config::default();
        let mut e = BindingEditor::default();
        let square = Control::Button(button::SQUARE);
        e.start(CaptureTarget::KeyboardMouse(square), false, t);
        assert!(!e.apply(vec![CaptureInput::Mouse(ps5gpu::input::MouseButton::Left)], false, t, &mut cfg));
        assert!(e.capturing());
        assert!(!e.apply(Vec::new(), true, t, &mut cfg));
        assert!(e.apply(vec![CaptureInput::Key(KeyCode::KeyZ)], false, t, &mut cfg));
        assert!(!e.capturing());
        assert_eq!(cfg.sources_of(square, false), [Source::Key(KeyCode::KeyZ)]);
        // Controller target: pad presses from the raw input snapshots.
        e.start(CaptureTarget::Gamepad(button::TRIANGLE), false, t);
        e.apply(Vec::new(), true, t, &mut cfg);
        let presses = pad_presses(&[(3, vec![PadInput(0)])], &[(3, vec![PadInput(0), PadInput(16)])]);
        assert_eq!(presses, [CaptureInput::Pad(PadInput(16))]);
        assert!(e.apply(presses, false, t, &mut cfg));
        assert_eq!(cfg.pad_inputs_of(button::TRIANGLE), [PadInput(16)]);
        // A capture that times out changes nothing.
        e.start(CaptureTarget::Gamepad(button::CROSS), false, t);
        assert!(!e.apply(Vec::new(), true, t + ps5input::capture::CAPTURE_TIMEOUT, &mut cfg));
        assert!(!e.capturing());
    }

    #[test]
    fn picture_click_rebinds_and_escape_cancels() {
        let t = Instant::now();
        let mut cfg = Config::default();
        let mut e = BindingEditor::default();
        let spot = |n: &str| dualsense::SPOTS.iter().position(|s| s.name == n).unwrap();
        let cross = spot("Cross");
        e.click_spot(cross, t);
        assert_eq!(e.selected, Some(cross));
        assert!(e.capturing());
        // Another click while waiting changes nothing.
        e.click_spot(spot("Circle"), t);
        assert_eq!(e.selected, Some(cross));
        // Escape cancels and keeps the binding.
        let before = cfg.clone();
        assert!(!e.apply(vec![CaptureInput::Key(KeyCode::Escape)], true, t, &mut cfg));
        assert!(!e.capturing());
        assert_eq!(cfg, before);
        // Click, release, press: the hotspot's action gets the key.
        e.click_spot(cross, t);
        e.apply(Vec::new(), true, t, &mut cfg);
        assert!(e.apply(vec![CaptureInput::Key(KeyCode::KeyK)], false, t, &mut cfg));
        assert_eq!(cfg.sources_of(Control::Button(button::CROSS), false), [Source::Key(KeyCode::KeyK)]);
        // Stick directions rebind keyboard stick actions.
        e.click_spot(spot("LS up"), t);
        e.apply(Vec::new(), true, t, &mut cfg);
        assert!(e.apply(vec![CaptureInput::Key(KeyCode::KeyU)], false, t, &mut cfg));
        assert_eq!(cfg.sources_of(Control::LeftStickUp, false), [Source::Key(KeyCode::KeyU)]);
        // Controller set: a controller button goes to the pad layout; the
        // Create button is shown but not bindable.
        e.set = BindingSet::Controller;
        e.click_spot(spot("Triangle"), t);
        e.apply(Vec::new(), true, t, &mut cfg);
        assert!(e.apply(vec![CaptureInput::Pad(PadInput(2))], false, t, &mut cfg));
        assert_eq!(cfg.pad_inputs_of(button::TRIANGLE), [PadInput(2)]);
        assert!(cfg.pad_inputs_of(button::SQUARE).is_empty(), "taken off Square");
        e.click_spot(spot("Create"), t);
        assert!(!e.capturing());
        assert_eq!(e.selected, Some(spot("Create")));
        // Clear and Default from the detail row.
        clear_binding(&mut cfg, CaptureTarget::Gamepad(button::TRIANGLE));
        assert!(cfg.pad_inputs_of(button::TRIANGLE).is_empty());
        default_binding(&mut cfg, CaptureTarget::Gamepad(button::TRIANGLE));
        assert_eq!(cfg.pad_inputs_of(button::TRIANGLE), [PadInput(3)]);
    }

    #[test]
    fn held_keys_follow_key_events() {
        let key = |k: egui::Key, pressed: bool| egui::Event::Key {
            key: k,
            physical_key: Some(k),
            pressed,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        };
        let mut held = Vec::new();
        update_held_keys(&mut held, &[key(egui::Key::W, true), key(egui::Key::A, true)], true);
        assert_eq!(held, [KeyCode::KeyW, KeyCode::KeyA]);
        update_held_keys(&mut held, &[key(egui::Key::W, false)], true);
        assert_eq!(held, [KeyCode::KeyA]);
        // egui reports nothing held (focus lost): all released.
        update_held_keys(&mut held, &[], false);
        assert!(held.is_empty());
    }

    #[test]
    fn layout_independent_key_capture() {
        let ev = egui::Event::Key {
            key: egui::Key::C,
            physical_key: Some(egui::Key::C),
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        };
        let no_physical = egui::Event::Key { key: egui::Key::C, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::NONE };
        assert_eq!(key_presses(&[ev, no_physical]), [CaptureInput::Key(KeyCode::KeyC)]);
    }

    /// Frames of one egui context through `BindingEditor::feed`, with the
    /// launcher's stale key release before each.
    struct Driver {
        ctx: egui::Context,
        down: std::collections::HashSet<egui::Key>,
        mods: egui::Modifiers,
    }

    impl Driver {
        fn new() -> Self {
            Self { ctx: egui::Context::default(), down: Default::default(), mods: egui::Modifiers::NONE }
        }

        fn frame(&mut self, e: &mut BindingEditor, cfg: &mut Config, events: Vec<egui::Event>) -> bool {
            let mut raw = egui::RawInput { events, modifiers: self.mods, ..Default::default() };
            release_stale_keys(&self.down, &mut raw, false);
            let mut changed = false;
            let mut down = Default::default();
            let _ = self.ctx.run(raw, |c| {
                changed = e.feed(c, Vec::new(), cfg);
                down = c.input(|i| i.keys_down.clone());
            });
            self.down = down;
            changed
        }
    }

    /// A key as egui-winit sends it: the key with its physical position,
    /// then the typed text.
    fn typed(logical: egui::Key, physical: egui::Key, text: &str) -> Vec<egui::Event> {
        let key = egui::Event::Key { key: logical, physical_key: Some(physical), pressed: true, repeat: false, modifiers: egui::Modifiers::NONE };
        vec![key, egui::Event::Text(text.into())]
    }

    fn released(k: egui::Key) -> egui::Event {
        egui::Event::Key { key: k, physical_key: Some(k), pressed: false, repeat: false, modifiers: egui::Modifiers::NONE }
    }

    /// Capture for Cross: an empty frame (arms), `events` (captures),
    /// `after`; Cross's inputs afterwards.
    fn capture(d: &mut Driver, e: &mut BindingEditor, cfg: &mut Config, events: Vec<egui::Event>, after: Vec<egui::Event>) -> Vec<Source> {
        let cross = Control::Button(button::CROSS);
        e.start(CaptureTarget::KeyboardMouse(cross), false, Instant::now());
        d.frame(e, cfg, Vec::new());
        assert!(d.frame(e, cfg, events), "captured");
        assert!(!e.capturing());
        d.frame(e, cfg, after);
        cfg.sources_of(cross, false)
    }

    #[test]
    fn every_kind_of_input_is_captured_through_egui() {
        let mut d = Driver::new();
        let mut e = BindingEditor::default();
        let mut cfg = Config::default();
        let cross = Control::Button(button::CROSS);
        let pos = egui::pos2(10.0, 10.0);
        for (logical, physical, text, code) in [
            (egui::Key::X, egui::Key::X, "x", KeyCode::KeyX),
            (egui::Key::C, egui::Key::C, "c", KeyCode::KeyC),
            (egui::Key::Z, egui::Key::Z, "z", KeyCode::KeyZ),
            (egui::Key::Num7, egui::Key::Num7, "7", KeyCode::Digit7),
            (egui::Key::Semicolon, egui::Key::Semicolon, ";", KeyCode::Semicolon),
            (egui::Key::OpenBracket, egui::Key::OpenBracket, "[", KeyCode::BracketLeft),
            // Russian layout: the X key types a Cyrillic letter, egui-winit
            // falls back to the physical key.
            (egui::Key::X, egui::Key::X, "\u{447}", KeyCode::KeyX),
            // AZERTY: the logical A is on the physical Q key.
            (egui::Key::A, egui::Key::Q, "a", KeyCode::KeyQ),
        ] {
            let got = capture(&mut d, &mut e, &mut cfg, typed(logical, physical, text), vec![released(logical)]);
            assert_eq!(got, [Source::Key(code)], "{text}");
        }
        let f5 = egui::Event::Key { key: egui::Key::F5, physical_key: Some(egui::Key::F5), pressed: true, repeat: false, modifiers: egui::Modifiers::NONE };
        assert_eq!(capture(&mut d, &mut e, &mut cfg, vec![f5], vec![released(egui::Key::F5)]), [Source::Key(KeyCode::F5)]);
        // Shift alone: egui reports a modifier change, no key event.
        e.start(CaptureTarget::KeyboardMouse(cross), false, Instant::now());
        d.frame(&mut e, &mut cfg, Vec::new());
        d.mods = egui::Modifiers::SHIFT;
        assert!(d.frame(&mut e, &mut cfg, Vec::new()));
        d.mods = egui::Modifiers::NONE;
        d.frame(&mut e, &mut cfg, Vec::new());
        assert_eq!(cfg.sources_of(cross, false), [Source::Key(KeyCode::ShiftLeft)]);
        // Mouse buttons and the wheel.
        let click = |b: egui::PointerButton, pressed: bool| egui::Event::PointerButton { pos, button: b, pressed, modifiers: egui::Modifiers::NONE };
        let got = capture(
            &mut d,
            &mut e,
            &mut cfg,
            vec![egui::Event::PointerMoved(pos), click(egui::PointerButton::Secondary, true)],
            vec![click(egui::PointerButton::Secondary, false)],
        );
        assert_eq!(got, [Source::Mouse(MouseButton::Right)]);
        let got = capture(&mut d, &mut e, &mut cfg, vec![click(egui::PointerButton::Extra1, true)], vec![click(egui::PointerButton::Extra1, false)]);
        assert_eq!(got, [Source::Mouse(MouseButton::X1)]);
        let wheel = egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Line, delta: egui::vec2(0.0, 1.0), modifiers: egui::Modifiers::NONE };
        assert_eq!(capture(&mut d, &mut e, &mut cfg, vec![wheel], Vec::new()), [Source::Wheel(1)]);
    }

    #[test]
    fn key_released_in_another_window_does_not_block_the_capture() {
        let mut d = Driver::new();
        let mut e = BindingEditor::default();
        let mut cfg = Config::default();
        let cross = Control::Button(button::CROSS);
        // Enter starts a game; the game window takes the focus before the
        // key is released.
        d.frame(&mut e, &mut cfg, typed(egui::Key::Enter, egui::Key::Enter, ""));
        d.frame(&mut e, &mut cfg, vec![egui::Event::WindowFocused(false)]);
        d.frame(&mut e, &mut cfg, vec![egui::Event::WindowFocused(true)]);
        assert!(d.ctx.input(|i| i.keys_down.is_empty()), "released on the focus change");
        assert_eq!(capture(&mut d, &mut e, &mut cfg, typed(egui::Key::X, egui::Key::X, "x"), vec![released(egui::Key::X)]), [Source::Key(KeyCode::KeyX)]);
        // A key still held when the capture starts is not taken: the
        // capture waits for its release.
        e.start(CaptureTarget::KeyboardMouse(cross), false, Instant::now());
        d.frame(&mut e, &mut cfg, typed(egui::Key::Space, egui::Key::Space, " "));
        assert!(!d.frame(&mut e, &mut cfg, typed(egui::Key::C, egui::Key::C, "c")), "not armed while Space is held");
        d.frame(&mut e, &mut cfg, vec![released(egui::Key::Space), released(egui::Key::C)]);
        assert!(d.frame(&mut e, &mut cfg, typed(egui::Key::Z, egui::Key::Z, "z")));
        assert_eq!(cfg.sources_of(cross, false), [Source::Key(KeyCode::KeyZ)]);
    }

    #[test]
    fn stale_keys_are_released_on_focus_changes() {
        let down: std::collections::HashSet<egui::Key> = [egui::Key::Escape, egui::Key::A].into_iter().collect();
        let mut raw = egui::RawInput { events: vec![released(egui::Key::B)], ..Default::default() };
        release_stale_keys(&down, &mut raw, false);
        assert_eq!(raw.events.len(), 1, "no focus change: kept");
        release_stale_keys(&down, &mut raw, true);
        assert_eq!(raw.events, [released(egui::Key::Escape), released(egui::Key::A), released(egui::Key::B)]);
        let mut raw = egui::RawInput { events: vec![egui::Event::WindowFocused(true)], ..Default::default() };
        release_stale_keys(&down, &mut raw, false);
        assert_eq!(raw.events.len(), 3);
    }

    #[test]
    fn a_held_controller_button_does_not_block_a_key_capture() {
        let ctx = egui::Context::default();
        let mut e = BindingEditor::default();
        let mut cfg = Config::default();
        let cross = Control::Button(button::CROSS);
        let held = vec![(1u32, vec![PadInput(0)])];
        e.start(CaptureTarget::KeyboardMouse(cross), false, Instant::now());
        let _ = ctx.run(egui::RawInput::default(), |c| {
            e.feed(c, held.clone(), &mut cfg);
        });
        let mut changed = false;
        let raw = egui::RawInput { events: typed(egui::Key::X, egui::Key::X, "x"), ..Default::default() };
        let _ = ctx.run(raw, |c| changed = e.feed(c, held.clone(), &mut cfg));
        assert!(changed);
        assert_eq!(cfg.sources_of(cross, false), [Source::Key(KeyCode::KeyX)]);
    }
}
