// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The controls overlay: while its key (action `ShowControls`, F2 by
//! default) is held, the game window shows the bindings in effect over the
//! running game: the DualSense picture of the binding set of the device
//! mode ([`crate::dualsense::picture`]) and a compact list of the sticks,
//! buttons, motion sensors, touch pad and emulator keys. It is drawn by
//! the game window's egui host (`ps5gpu::menu`) without taking input, so
//! the game keeps running and receiving everything but the overlay key.
//!
//! The in-game menu's "Controls hint" is the small version of the same
//! picture (with the held inputs lit, inside the menu).

use crate::dualsense::{self, PictureParams};
use crate::input_ui::{self, BindingSet};
use eframe::egui::{self, Color32, FontId, Pos2, RichText, Sense, Stroke, Vec2};
use ps5input::button;
use ps5input::mapping::{pad_button_actions, Config, Control, DeviceMode, Stick, StickMode};

/// The binding set a device mode uses for player 1.
pub fn set_for(mode: DeviceMode) -> BindingSet {
    match mode {
        DeviceMode::Controller => BindingSet::Controller,
        DeviceMode::Keyboard => BindingSet::Keyboard,
        DeviceMode::Auto | DeviceMode::KeyboardMouse => BindingSet::KeyboardMouse,
    }
}

/// `cfg` as the game receives it, for showing: with swapped sticks the
/// stick keys and the mouse are moved to the stick they drive.
pub fn shown_config(cfg: &Config) -> Config {
    let mut c = cfg.clone();
    if c.swap_sticks {
        for b in c.bindings.iter_mut().chain(c.keyboard_bindings.iter_mut()) {
            b.control = ps5input::state::swapped(b.control);
        }
        std::mem::swap(&mut c.left_stick_mode, &mut c.right_stick_mode);
        c.swap_sticks = false;
    }
    c
}

/// One line of the list.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub label: String,
    pub value: String,
}

/// A titled part of the list.
#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub title: &'static str,
    pub rows: Vec<Row>,
}

fn row(label: impl Into<String>, value: impl Into<String>) -> Row {
    Row { label: label.into(), value: value.into() }
}

/// A source as short as the list needs: `Keypad 8` -> `Num8`.
fn short(s: &impl ToString) -> String {
    let t = dualsense::short_source(&s.to_string());
    match t.strip_prefix("Keypad ") {
        Some(k) => format!("Num{k}"),
        None => t,
    }
}

/// The keyboard / mouse inputs of `c` in `set`, `-` for none.
fn keys(cfg: &Config, set: BindingSet, c: Control) -> String {
    let v: Vec<String> = cfg.sources_of(c, set == BindingSet::Keyboard).iter().map(short).collect();
    if v.is_empty() {
        "-".into()
    } else {
        v.join(", ")
    }
}

/// The first input of each control, `a / b / ...`; `None` when none of
/// them has one.
fn keys_of(cfg: &Config, set: BindingSet, cs: &[Control]) -> Option<String> {
    let parts: Vec<String> = cs.iter().map(|&c| keys(cfg, set, c)).collect();
    parts.iter().any(|p| p != "-").then(|| parts.join(" / "))
}

/// The emulator keys in effect for the device mode, with what they do.
const EMULATOR_KEYS: &[(Control, &str)] = &[
    (Control::ToggleMenu, "Menu"),
    (Control::ShowControls, "Controls (hold)"),
    (Control::ToggleMouse, "Capture mouse"),
    (Control::ReleaseMouse, "Release mouse"),
    (Control::TogglePerfOverlay, "Perf. overlay"),
    (Control::ToggleFullscreen, "Fullscreen"),
    (Control::Create, "Screenshot"),
    (Control::Mute, "Sound off / on"),
    (Control::SpeakerVolume, "Speaker volume"),
    (Control::VibrationIntensity, "Vibration"),
    (Control::TriggerEffectIntensity, "Trigger effects"),
];

/// The keys bound to an emulator action in the device mode.
pub fn emulator_key(cfg: &Config, c: Control) -> Option<String> {
    let v: Vec<String> =
        cfg.active_bindings().iter().filter(|b| b.control == c).map(|b| dualsense::short_source(&b.source.to_string())).collect();
    (!v.is_empty()).then(|| v.join(", "))
}

/// The list for the settings in effect `cfg`.
pub fn groups(cfg_in: &Config) -> Vec<Group> {
    let set = set_for(cfg_in.mode);
    let cfg = shown_config(cfg_in);
    let pad = set == BindingSet::Controller;
    let mut out = Vec::new();

    let mut sticks = Vec::new();
    for stick in Stick::BOTH {
        let value = match dualsense::stick_mode(&cfg, stick, set) {
            None => "controller".to_string(),
            Some(StickMode::Keys) => {
                let dirs = match stick {
                    Stick::Left => [Control::LeftStickUp, Control::LeftStickLeft, Control::LeftStickDown, Control::LeftStickRight],
                    Stick::Right => [Control::RightStickUp, Control::RightStickLeft, Control::RightStickDown, Control::RightStickRight],
                };
                dirs.iter()
                    .map(|&c| cfg.sources_of(c, set == BindingSet::Keyboard).first().map(short).unwrap_or("-".into()))
                    .collect::<Vec<_>>()
                    .join(" ")
            }
            Some(StickMode::Mouse) => match emulator_key(cfg_in, Control::ToggleMouse) {
                Some(k) => format!("{} ({k})", dualsense::stick_text(&cfg, StickMode::Mouse)),
                None => dualsense::stick_text(&cfg, StickMode::Mouse),
            },
            Some(StickMode::Off) => "controller only".to_string(),
        };
        sticks.push(row(stick.label(), value));
    }
    if cfg_in.swap_sticks && !pad {
        sticks.push(row("Sticks swapped", "yes"));
    }
    out.push(Group { title: "Sticks", rows: sticks });

    let buttons = pad_button_actions()
        .map(|(name, bit)| {
            let value = if pad {
                let v: Vec<String> = cfg.pad_inputs_of(bit).iter().map(|p| dualsense::short_source(&p.to_config_string())).collect();
                match (v.is_empty(), bit) {
                    (true, button::L2) => "analog L2".into(),
                    (true, button::R2) => "analog R2".into(),
                    (true, _) => "-".into(),
                    _ => v.join(", "),
                }
            } else {
                keys(&cfg, set, Control::Button(bit))
            };
            row(input_ui::action_label(name), value)
        })
        .collect();
    out.push(Group { title: "Buttons", rows: buttons });

    if pad {
        let own = |on: bool| if on { "the controller's own" } else { "off" };
        out.push(Group { title: "Motion", rows: vec![row("Motion sensors", own(cfg.features.motion))] });
        out.push(Group { title: "Touch pad", rows: vec![row("Touch pad", own(cfg.features.touchpad))] });
    } else {
        let mut motion = Vec::new();
        for (label, cs) in [
            ("Tilt forward / back", &[Control::TiltForward, Control::TiltBack][..]),
            ("Tilt left / right", &[Control::TiltLeft, Control::TiltRight]),
            ("Turn left / right", &[Control::TurnLeft, Control::TurnRight]),
            ("Shake", &[Control::Shake]),
            ("Gyro aim (hold)", &[Control::MouseMotion]),
        ] {
            if set == BindingSet::Keyboard && cs.iter().any(|c| c.mouse_only()) {
                continue;
            }
            if let Some(v) = keys_of(&cfg, set, cs) {
                motion.push(row(label, v));
            }
        }
        out.push(Group { title: "Motion", rows: motion });
        let mut touch = vec![row("Touch pad mode", keys(&cfg, set, Control::TouchpadMode))];
        if keys_of(&cfg, set, &[Control::TouchpadMode]).is_some() {
            let finger = if set == BindingSet::Keyboard { "" } else { " + mouse" };
            touch.push(row("Zones", format!("Num1-9{finger}")));
        }
        for (label, cs) in [
            ("Swipe U / D / L / R", &[Control::SwipeUp, Control::SwipeDown, Control::SwipeLeft, Control::SwipeRight][..]),
            ("Touch L / M / R", &[Control::TapLeft, Control::TapCenter, Control::TapRight]),
            ("Two fingers (hold)", &[Control::TwoFingerTouch]),
            ("Click L / R half", &[Control::TouchLeft, Control::TouchRight]),
        ] {
            if let Some(v) = keys_of(&cfg, set, cs) {
                touch.push(row(label, v));
            }
        }
        out.push(Group { title: "Touch pad", rows: touch });
    }

    let mut emu = Vec::new();
    for &(c, label) in EMULATOR_KEYS {
        let mut v = emulator_key(cfg_in, c);
        if c == Control::ToggleMenu && cfg_in.mode.uses_controller() {
            v = Some(v.map_or("PS button".into(), |k| format!("{k}, PS button")));
        }
        if let Some(v) = v {
            emu.push(row(label, v));
        }
    }
    out.push(Group { title: "Emulator keys", rows: emu });
    out
}

/// What the overlay shows besides the bindings.
pub struct OverlayView<'a> {
    /// The input settings in effect.
    pub cfg: &'a Config,
    pub title: &'a str,
    /// Where the bindings come from (per-game profile or global settings).
    pub source: &'a str,
}

/// The line under the title: source, device mode, swapped sticks.
pub fn source_line(cfg: &Config, source: &str) -> String {
    let mut s = format!("{source} \u{00b7} {}", cfg.mode.label());
    if cfg.swap_sticks {
        s.push_str(" \u{00b7} sticks swapped");
    }
    s
}

const ROW_H: f32 = 17.0;
const GROUP_GAP: f32 = 8.0;

fn accent() -> Color32 {
    Color32::from_rgb(110, 175, 255)
}

/// Groups split into `n` columns of about the same height (in order, a
/// group never split).
pub fn split_columns(groups: &[Group], n: usize) -> Vec<Vec<&Group>> {
    let n = n.max(1);
    let weight = |g: &Group| g.rows.len() as f32 + 1.5;
    let total: f32 = groups.iter().map(weight).sum();
    let mut cols: Vec<Vec<&Group>> = vec![Vec::new()];
    let mut acc = 0.0;
    for g in groups.iter().filter(|g| !g.rows.is_empty()) {
        let target = total * cols.len() as f32 / n as f32;
        if cols.len() < n && !cols.last().is_some_and(Vec::is_empty) && acc + weight(g) / 2.0 > target {
            cols.push(Vec::new());
        }
        acc += weight(g);
        if let Some(c) = cols.last_mut() {
            c.push(g);
        }
    }
    cols
}

fn group_ui(ui: &mut egui::Ui, g: &Group, width: f32) {
    ui.label(RichText::new(g.title).strong().size(13.5).color(accent()));
    let label_w = (width * 0.42).min(170.0);
    let weak = Color32::from_gray(185);
    let strong = Color32::from_gray(245);
    for r in &g.rows {
        let (rect, _) = ui.allocate_exact_size(Vec2::new(width, ROW_H), Sense::hover());
        let painter = ui.painter();
        let text = |s: &str, font: FontId, color: Color32, w: f32| {
            let mut job = egui::text::LayoutJob::simple_singleline(s.to_string(), font, color);
            job.wrap = egui::text::TextWrapping::truncate_at_width(w.max(10.0));
            painter.layout_job(job)
        };
        let l = text(&r.label, FontId::proportional(12.5), weak, label_w - 8.0);
        painter.galley(Pos2::new(rect.left(), rect.center().y - l.size().y / 2.0), l, weak);
        let v = text(&r.value, FontId::monospace(12.5), strong, width - label_w);
        painter.galley(Pos2::new(rect.left() + label_w, rect.center().y - v.size().y / 2.0), v, strong);
    }
    ui.add_space(GROUP_GAP);
}

fn columns_ui(ui: &mut egui::Ui, groups: &[Group], n: usize, width: f32) {
    let gap = 18.0;
    let col_w = (width - gap * (n as f32 - 1.0)) / n as f32;
    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = gap;
        for col in split_columns(groups, n) {
            ui.vertical(|ui| {
                ui.set_width(col_w);
                ui.spacing_mut().item_spacing.y = 1.0;
                for g in col {
                    group_ui(ui, g, col_w);
                }
            });
        }
    });
}

/// Draw the overlay over the whole window (no input taken).
pub fn draw(ctx: &egui::Context, v: &OverlayView) {
    let screen = ctx.screen_rect();
    egui::Area::new(egui::Id::new("rps5e-controls-backdrop")).order(egui::Order::Background).fixed_pos(screen.min).interactable(false).show(ctx, |ui| {
        ui.painter().rect_filled(screen, 0.0, Color32::from_black_alpha(90));
    });
    let set = set_for(v.cfg.mode);
    let shown = shown_config(v.cfg);
    let groups = groups(v.cfg);
    let panel_w = (screen.width() - 32.0).clamp(380.0, 1560.0);
    egui::Area::new(egui::Id::new("rps5e-controls-overlay"))
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .order(egui::Order::Foreground)
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(Color32::from_rgba_unmultiplied(16, 18, 22, 212))
                .stroke(Stroke::new(1.0f32, Color32::from_white_alpha(28)))
                .corner_radius(10.0)
                .inner_margin(egui::Margin::same(14))
                .show(ui, |ui| {
                    ui.set_width(panel_w);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Controls").strong().size(18.0).color(Color32::WHITE));
                        ui.label(RichText::new(v.title).size(14.0).color(Color32::from_gray(190)));
                    });
                    ui.label(RichText::new(source_line(v.cfg, v.source)).size(12.5).color(Color32::from_gray(170)));
                    ui.add_space(6.0);
                    let wide = panel_w >= 1000.0;
                    let pic_w = if wide { (panel_w * 0.53).min(1000.0) } else { panel_w };
                    let picture = |ui: &mut egui::Ui| {
                        ui.set_width(pic_w);
                        let p = PictureParams { cfg: &shown, set, selected: None, capturing: false, pressed: &[], interactive: false, max_width: pic_w };
                        dualsense::picture(ui, &p);
                    };
                    if wide {
                        ui.horizontal_top(|ui| {
                            ui.spacing_mut().item_spacing.x = 18.0;
                            ui.vertical(picture);
                            let rest = ui.available_width();
                            ui.vertical(|ui| columns_ui(ui, &groups, 2, rest));
                        });
                    } else {
                        ui.vertical(picture);
                        ui.add_space(6.0);
                        let n = if panel_w >= 640.0 { 3 } else { 2 };
                        columns_ui(ui, &groups, n, panel_w);
                    }
                    let mut foot = match emulator_key(v.cfg, Control::ShowControls) {
                        Some(k) => format!("Release {k} to hide."),
                        None => String::new(),
                    };
                    if let Some(k) = emulator_key(v.cfg, Control::ToggleMenu) {
                        foot.push_str(&format!(" {k} opens the menu, where Controls changes the bindings."));
                    }
                    ui.label(RichText::new(foot.trim()).size(12.0).color(Color32::from_gray(150)));
                });
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::Rect;

    fn find<'a>(gs: &'a [Group], title: &str, label: &str) -> Option<&'a str> {
        gs.iter().find(|g| g.title == title)?.rows.iter().find(|r| r.label == label).map(|r| r.value.as_str())
    }

    #[test]
    fn list_follows_the_device_mode() {
        let cfg = Config::default();
        let g = groups(&cfg);
        assert_eq!(find(&g, "Sticks", "Left stick"), Some("W A S D"));
        assert!(find(&g, "Sticks", "Right stick").unwrap().starts_with("Mouse"));
        assert!(find(&g, "Sticks", "Right stick").unwrap().contains("F9"));
        assert_eq!(find(&g, "Buttons", "Cross"), Some("Space, Return"));
        assert_eq!(find(&g, "Motion", "Shake"), Some("Num5"));
        assert_eq!(find(&g, "Touch pad", "Swipe U / D / L / R"), Some("Home / End / Delete / PageDown"));
        assert_eq!(find(&g, "Emulator keys", "Controls (hold)"), Some("F2"));
        assert_eq!(find(&g, "Emulator keys", "Menu"), Some("F1, PS button"));
        assert_eq!(find(&g, "Emulator keys", "Capture mouse"), Some("F9"));
        assert_eq!(find(&g, "Emulator keys", "Perf. overlay"), Some("F10"));
        assert_eq!(find(&g, "Emulator keys", "Fullscreen"), Some("F11"));
        assert_eq!(find(&g, "Emulator keys", "Screenshot"), Some("F12"));
        assert_eq!(find(&g, "Emulator keys", "Sound off / on"), Some("F8"));
        // Controller: the controller layout, the keyboard keeps the window keys.
        let pad = Config { mode: DeviceMode::Controller, ..Config::default() };
        let g = groups(&pad);
        assert_eq!(find(&g, "Buttons", "Cross"), Some("Cross"));
        assert_eq!(find(&g, "Buttons", "L2"), Some("analog L2"));
        assert_eq!(find(&g, "Sticks", "Left stick"), Some("controller"));
        assert_eq!(find(&g, "Emulator keys", "Controls (hold)"), Some("F2"));
        assert_eq!(find(&g, "Emulator keys", "Capture mouse"), None);
        // Keyboard only: no mouse.
        let kb = Config { mode: DeviceMode::Keyboard, ..Config::default() };
        let g = groups(&kb);
        assert_eq!(find(&g, "Sticks", "Right stick"), Some("I J K L"));
        assert_eq!(find(&g, "Buttons", "L2"), Some("1"));
        assert_eq!(find(&g, "Motion", "Gyro aim (hold)"), None);
    }

    #[test]
    fn swapped_sticks_are_shown_where_they_act() {
        let cfg = Config { mode: DeviceMode::KeyboardMouse, swap_sticks: true, ..Config::default() };
        let g = groups(&cfg);
        assert_eq!(find(&g, "Sticks", "Right stick"), Some("W A S D"));
        assert!(find(&g, "Sticks", "Left stick").unwrap().starts_with("Mouse"));
        assert!(find(&g, "Sticks", "Sticks swapped").is_some());
        let shown = shown_config(&cfg);
        assert_eq!(shown.right_stick_mode, StickMode::Keys);
        assert!(source_line(&cfg, "Global settings").contains("sticks swapped"));
    }

    #[test]
    fn columns_keep_groups_whole_and_balanced() {
        let g = groups(&Config::default());
        let cols = split_columns(&g, 2);
        assert_eq!(cols.len(), 2);
        assert_eq!(cols.iter().map(Vec::len).sum::<usize>(), g.len());
        let h = |c: &Vec<&Group>| c.iter().map(|g| g.rows.len() + 1).sum::<usize>() as i32;
        assert!((h(&cols[0]) - h(&cols[1])).abs() <= 12, "{} vs {}", h(&cols[0]), h(&cols[1]));
    }

    /// A stand-in for a game frame: colour bands.
    fn sample_frame(ctx: &egui::Context) {
        let screen = ctx.screen_rect();
        egui::Area::new(egui::Id::new("game")).order(egui::Order::Background).fixed_pos(screen.min).interactable(false).show(ctx, |ui| {
            let p = ui.painter();
            let n = 24;
            for i in 0..n {
                let t = i as f32 / n as f32;
                let x0 = screen.left() + screen.width() * t;
                let r = Rect::from_min_max(Pos2::new(x0, screen.top()), Pos2::new(x0 + screen.width() / n as f32 + 1.0, screen.bottom()));
                let c = Color32::from_rgb((40.0 + 150.0 * t) as u8, (90.0 + 60.0 * (1.0 - t)) as u8, (140.0 - 80.0 * t) as u8);
                p.rect_filled(r, 0.0, c);
            }
            p.circle_filled(screen.center() + Vec2::new(-200.0, 60.0), 140.0, Color32::from_rgb(230, 200, 120));
        });
    }

    fn render(size: [f32; 2], ppp: f32, cfg: &Config) -> (usize, usize, Vec<u8>) {
        let mut r = crate::softrender::SoftRenderer::new(size, ppp);
        r.ctx.set_theme(egui::Theme::Dark);
        let mut img = (0, 0, Vec::new());
        for _ in 0..3 {
            img = r.frame(Vec::new(), |ctx| {
                sample_frame(ctx);
                draw(ctx, &OverlayView { cfg, title: "SAMPLE GAME", source: "Per-game profile (PPSA00000)" });
            });
        }
        img
    }

    #[test]
    fn overlay_draws_over_the_frame() {
        let img = render([1352.0, 760.0], 1.0, &Config::default());
        // The middle is the dark panel, the corner the dimmed frame.
        let px = |x: usize, y: usize| {
            let i = (y * img.0 + x) * 4;
            [img.2[i], img.2[i + 1], img.2[i + 2]]
        };
        let corner = px(4, 4);
        assert!(corner.iter().any(|&c| c > 25), "the game shows through: {corner:?}");
        let lit = img.2.chunks_exact(4).filter(|p| p[0] > 200 && p[1] > 200 && p[2] > 200).count();
        assert!(lit > 2000, "text and the drawing are drawn ({lit} bright pixels)");
    }

    /// `RPS5E_GUI_RENDER_DIR=<dir> cargo test -p rps5e-gui render_controls_overlay -- --ignored`
    /// writes the overlay over a sample frame at 1080p and 4K window sizes
    /// (egui points as the game window scales them) and in controller mode.
    #[test]
    #[ignore]
    fn render_controls_overlay() {
        let Some(dir) = std::env::var_os("RPS5E_GUI_RENDER_DIR") else { return };
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        // 1080p: zoom 1080 / 760; 4K: zoom 2160 / 760 (`ps5gpu::menu`).
        for (name, cfg, ppp) in [
            ("controls-overlay-1080p.png", Config::default(), 1080.0 / 760.0),
            ("controls-overlay-4k.png", Config::default(), 2160.0 / 760.0),
            ("controls-overlay-controller.png", Config { mode: DeviceMode::Controller, ..Config::default() }, 1080.0 / 760.0),
            ("controls-overlay-swapped.png", Config { swap_sticks: true, ..Config::default() }, 1080.0 / 760.0),
        ] {
            // A 16:9 window is 1351 x 760 points at either size.
            let size = [1920.0 * 760.0 / 1080.0, 760.0];
            let img = render(size, ppp, &cfg);
            crate::softrender::write_png(&dir.join(name), img.0, img.1, &img.2);
        }
    }
}
