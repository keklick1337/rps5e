// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! DualSense picture of the bindings: every button, stick direction and
//! trigger has a hotspot on the controller drawing with a callout naming
//! what drives it in the selected binding set. In the editor a click on a
//! hotspot (or its callout) selects it and starts rebinding it; the
//! read-only hint (in-game "Controls hint") lights up the buttons whose
//! inputs are held.
//!
//! The drawing is `assets/dualsense/body.png` (1000x562, from the
//! DualSense layout artwork with the callout marks removed); [`SPOTS`]
//! holds the hotspot of every control in that image's pixels.

use crate::input_ui::BindingSet;
use eframe::egui::{self, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use ps5input::button;
use ps5input::capture::CaptureTarget;
use ps5input::mapping::{Config, Control, PadInput, Source, Stick, StickMode, PAD_INPUTS};
use ps5input::sdl::analog_trigger_reported;

/// The controller drawing (RGBA PNG).
pub const BODY_PNG: &[u8] = include_bytes!("../assets/dualsense/body.png");
/// Size of [`BODY_PNG`]; hotspot coordinates are in these pixels.
pub const IMAGE_SIZE: [f32; 2] = [1000.0, 562.0];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// x0, y0, x1, y1.
    Rect([f32; 4]),
    /// Centre and radius.
    Circle([f32; 2], f32),
}

impl Shape {
    pub fn contains(self, p: [f32; 2]) -> bool {
        match self {
            Shape::Rect([x0, y0, x1, y1]) => p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1,
            Shape::Circle([cx, cy], r) => (p[0] - cx).powi(2) + (p[1] - cy).powi(2) <= r * r,
        }
    }

    pub fn center(self) -> [f32; 2] {
        match self {
            Shape::Rect([x0, y0, x1, y1]) => [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
            Shape::Circle(c, _) => c,
        }
    }

    /// Where a callout's leader line ends.
    fn anchor(spot: &Spot) -> [f32; 2] {
        let [x, y] = spot.shape.center();
        // A whole-stick callout points at the stick's rim, not at L3/R3.
        if spot.stick.is_some() {
            [x, y - DIR_OFF]
        } else {
            [x, y]
        }
    }

    fn area(self) -> f32 {
        match self {
            Shape::Rect([x0, y0, x1, y1]) => (x1 - x0) * (y1 - y0),
            Shape::Circle(_, r) => std::f32::consts::PI * r * r,
        }
    }
}

/// Where a hotspot's callout goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
    Top,
    Bottom,
}

/// One control on the drawing.
#[derive(Clone, Copy, Debug)]
pub struct Spot {
    /// Short name on the callout.
    pub name: &'static str,
    /// The action it rebinds (`None`: shown, not bindable).
    pub control: Option<Control>,
    pub shape: Shape,
    pub side: Side,
    /// The whole stick of a stick not on its keys (mouse or controller
    /// only): one callout in place of the four directions.
    pub stick: Option<Stick>,
}

const fn spot(name: &'static str, control: Control, shape: Shape, side: Side) -> Spot {
    Spot { name, control: Some(control), shape, side, stick: None }
}

const fn whole_stick(name: &'static str, stick: Stick, c: [f32; 2], side: Side) -> Spot {
    Spot { name, control: None, shape: circle(c[0], c[1], DIR_OFF + DIR_HALF), side, stick: Some(stick) }
}

const fn b(bit: u32) -> Control {
    Control::Button(bit)
}

const fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Shape {
    Shape::Rect([x0, y0, x1, y1])
}

const fn circle(x: f32, y: f32, r: f32) -> Shape {
    Shape::Circle([x, y], r)
}

/// Left stick centre (the right one mirrors it around `MIRROR_X`).
const LS: [f32; 2] = [336.0, 440.0];
const MIRROR_X: f32 = 1001.0;
const RS: [f32; 2] = [MIRROR_X - LS[0], LS[1]];
/// Stick direction zones: offset of the zone centre from the stick centre
/// and the zone half size.
const DIR_OFF: f32 = 58.0;
const DIR_HALF: f32 = 15.0;

const fn dir(name: &'static str, control: Control, c: [f32; 2], dx: f32, dy: f32, side: Side) -> Spot {
    let (x, y) = (c[0] + dx * DIR_OFF, c[1] + dy * DIR_OFF);
    spot(name, control, rect(x - DIR_HALF, y - DIR_HALF, x + DIR_HALF, y + DIR_HALF), side)
}

/// Every hotspot, in the pixels of [`BODY_PNG`].
pub const SPOTS: &[Spot] = &[
    spot("L2", b(button::L2), rect(122.0, 4.0, 236.0, 50.0), Side::Left),
    spot("L1", b(button::L1), rect(104.0, 52.0, 250.0, 104.0), Side::Left),
    spot("Create", Control::Create, rect(240.0, 170.0, 273.0, 222.0), Side::Left),
    spot("Up", b(button::UP), rect(148.0, 222.0, 207.0, 280.0), Side::Left),
    spot("Left", b(button::LEFT), rect(90.0, 268.0, 160.0, 326.0), Side::Left),
    spot("Right", b(button::RIGHT), rect(193.0, 268.0, 263.0, 326.0), Side::Left),
    spot("Down", b(button::DOWN), rect(148.0, 312.0, 207.0, 370.0), Side::Left),
    dir("LS up", Control::LeftStickUp, LS, 0.0, -1.0, Side::Left),
    dir("LS left", Control::LeftStickLeft, LS, -1.0, 0.0, Side::Left),
    dir("LS right", Control::LeftStickRight, LS, 1.0, 0.0, Side::Left),
    dir("LS down", Control::LeftStickDown, LS, 0.0, 1.0, Side::Left),
    whole_stick("Left stick", Stick::Left, LS, Side::Left),
    spot("L3", b(button::L3), circle(LS[0], LS[1], 36.0), Side::Left),
    spot("Touch left", Control::TouchLeft, rect(305.0, 112.0, 420.0, 318.0), Side::Left),
    spot("Touch click", b(button::TOUCH_PAD), rect(305.0, 112.0, 697.0, 318.0), Side::Top),
    spot("Touch right", Control::TouchRight, rect(582.0, 112.0, 697.0, 318.0), Side::Right),
    spot("R2", b(button::R2), rect(765.0, 4.0, 879.0, 50.0), Side::Right),
    spot("R1", b(button::R1), rect(751.0, 52.0, 897.0, 104.0), Side::Right),
    spot("Options", b(button::OPTIONS), rect(728.0, 170.0, 762.0, 222.0), Side::Right),
    spot("Triangle", b(button::TRIANGLE), circle(825.0, 233.0, 33.0), Side::Right),
    spot("Circle", b(button::CIRCLE), circle(897.0, 297.0, 33.0), Side::Right),
    spot("Square", b(button::SQUARE), circle(746.0, 295.0, 33.0), Side::Right),
    spot("Cross", b(button::CROSS), circle(822.0, 357.0, 33.0), Side::Right),
    dir("RS up", Control::RightStickUp, RS, 0.0, -1.0, Side::Right),
    dir("RS left", Control::RightStickLeft, RS, -1.0, 0.0, Side::Right),
    dir("RS right", Control::RightStickRight, RS, 1.0, 0.0, Side::Right),
    dir("RS down", Control::RightStickDown, RS, 0.0, 1.0, Side::Right),
    whole_stick("Right stick", Stick::Right, RS, Side::Right),
    spot("R3", b(button::R3), circle(RS[0], RS[1], 36.0), Side::Right),
    spot("PS", Control::ToggleMenu, circle(502.0, 392.0, 26.0), Side::Bottom),
    spot("Mute", Control::Mute, rect(474.0, 426.0, 530.0, 452.0), Side::Bottom),
];

/// Keyboard/mouse actions that are not a place on the drawing: the motion
/// sensors and the touch pad gestures (shown in a list under it).
pub const MOTION_ACTIONS: &[Control] = &[
    Control::TiltForward,
    Control::TiltBack,
    Control::TiltLeft,
    Control::TiltRight,
    Control::TurnLeft,
    Control::TurnRight,
    Control::Shake,
    Control::MouseMotion,
];
pub const TOUCH_ACTIONS: &[Control] = &[
    Control::TouchpadMode,
    Control::TapLeft,
    Control::TapCenter,
    Control::TapRight,
    Control::SwipeUp,
    Control::SwipeDown,
    Control::SwipeLeft,
    Control::SwipeRight,
    Control::TwoFingerTouch,
];

/// What a hotspot rebinds in `set` (`None`: not bindable there).
pub fn spot_target(spot: &Spot, set: BindingSet) -> Option<CaptureTarget> {
    let c = spot.control?;
    match set {
        BindingSet::KeyboardMouse => Some(CaptureTarget::KeyboardMouse(c)),
        BindingSet::Keyboard => (!c.mouse_only()).then_some(CaptureTarget::Keyboard(c)),
        BindingSet::Controller => match c {
            Control::Button(bit) => Some(CaptureTarget::Gamepad(bit)),
            _ => None,
        },
    }
}

/// The mode of a stick as the keyboard sets use it (`None`: the
/// controller set, where sticks are not bound).
pub fn stick_mode(cfg: &Config, stick: Stick, set: BindingSet) -> Option<StickMode> {
    match set {
        BindingSet::Controller => None,
        BindingSet::KeyboardMouse => Some(cfg.stick_mode_in(stick, false)),
        BindingSet::Keyboard => Some(cfg.stick_mode_in(stick, true)),
    }
}

/// Text of a whole-stick callout.
pub fn stick_text(cfg: &Config, mode: StickMode) -> String {
    match mode {
        StickMode::Mouse => {
            let mut t = format!("Mouse \u{00d7}{:.1}", cfg.mouse_sensitivity);
            if cfg.mouse_invert_x || cfg.mouse_invert_y {
                t.push_str(" inv");
            }
            t
        }
        StickMode::Off => "controller only".into(),
        StickMode::Keys => "keys".into(),
    }
}

/// The callout text of a hotspot that is shown but cannot be rebound in
/// `set`; `None` hides the hotspot.
fn fixed_text(spot: &Spot, set: BindingSet) -> Option<&'static str> {
    match (spot.control, set) {
        (None, _) if spot.stick.is_none() => Some("system only"),
        (Some(Control::ToggleMenu), BindingSet::Controller) => Some("in-game menu"),
        (Some(Control::Create | Control::Mute), BindingSet::Controller) => Some("system only"),
        _ => None,
    }
}

/// Whether a hotspot is drawn for `set`: a stick on its keys shows its four
/// directions, one in mouse or controller-only mode a single callout.
pub fn spot_visible(cfg: &Config, spot: &Spot, set: BindingSet) -> bool {
    if let Some(stick) = spot.stick {
        return stick_mode(cfg, stick, set).is_some_and(|m| m != StickMode::Keys);
    }
    if let Some(stick) = spot.control.and_then(Control::stick) {
        if stick_mode(cfg, stick, set) != Some(StickMode::Keys) {
            return false;
        }
    }
    spot_target(spot, set).is_some() || fixed_text(spot, set).is_some()
}

/// Longer explanation of a hotspot (the callout's hover text).
pub fn spot_help(spot: &Spot, set: BindingSet) -> Option<&'static str> {
    if spot.stick.is_some() {
        return Some("Click to choose what drives this stick (keys, mouse or the controller only) and the mouse settings.");
    }
    match (spot.control, set) {
        (Some(Control::Create), BindingSet::Controller) => Some("Create is a system button (screenshots, sharing): games never see it."),
        (Some(Control::Create), _) => Some("Create: the key saves a screenshot of the game window (games never see this button)."),
        (Some(Control::Mute), BindingSet::Controller) => Some("Mute is a system button: games never see it."),
        (Some(Control::Mute), _) => Some("Mute: the key turns the game's sound off and on."),
        (Some(Control::ToggleMenu), BindingSet::Controller) => Some("PS opens the in-game menu."),
        (Some(Control::ToggleMenu), _) => Some("PS opens the in-game menu: the key that opens it."),
        (Some(Control::Button(button::TOUCH_PAD)), _) => Some("Touch pad click in the middle."),
        (Some(Control::TouchLeft), _) => Some("Touch pad click on the left half."),
        (Some(Control::TouchRight), _) => Some("Touch pad click on the right half."),
        _ => None,
    }
}

/// The hotspot under `p` (image pixels): the smallest visible one.
pub fn hit(cfg: &Config, set: BindingSet, p: [f32; 2]) -> Option<usize> {
    SPOTS
        .iter()
        .enumerate()
        .filter(|(_, s)| spot_visible(cfg, s, set) && s.shape.contains(p))
        .min_by(|a, b| a.1.shape.area().total_cmp(&b.1.shape.area()))
        .map(|(i, _)| i)
}

/// A binding source in a few characters: `KEY:Space` -> `Space`,
/// `MOUSE:Right` -> `Mouse Right`, `BUTTON:South` -> `Cross`.
pub fn short_source(s: &str) -> String {
    let Some((kind, value)) = s.split_once(':') else { return s.to_string() };
    match kind {
        // Left Shift -> LShift (the arrow keys keep their names).
        "KEY" => match value.split_once(' ') {
            Some(("Left", k)) => format!("L{k}"),
            Some(("Right", k)) => format!("R{k}"),
            _ => value.to_string(),
        },
        "MOUSE" => format!("Mouse {value}"),
        "WHEEL" => format!("Wheel {value}"),
        "BUTTON" => match PAD_INPUTS.iter().find(|(names, _)| names[0] == value).and_then(|(names, _)| names.get(1)) {
            Some(alias) if alias.chars().any(|c| c.is_ascii_digit()) => alias.to_ascii_uppercase(),
            Some(alias) => {
                let mut c = alias.chars();
                c.next().map(|f| f.to_ascii_uppercase().to_string() + c.as_str()).unwrap_or_default()
            }
            None => value.to_string(),
        },
        _ => s.to_string(),
    }
}

/// The callout text of a hotspot: its inputs in `set`, `-` for none.
pub fn spot_text(cfg: &Config, spot: &Spot, set: BindingSet) -> String {
    if let Some(mode) = spot.stick.and_then(|s| stick_mode(cfg, s, set)) {
        return stick_text(cfg, mode);
    }
    if let Some(t) = fixed_text(spot, set) {
        return t.to_string();
    }
    let Some(target) = spot_target(spot, set) else { return String::new() };
    let parts: Vec<String> = match target {
        CaptureTarget::KeyboardMouse(c) => cfg.sources_of(c, false).iter().map(|s| short_source(&s.to_string())).collect(),
        CaptureTarget::Keyboard(c) => cfg.sources_of(c, true).iter().map(|s| short_source(&s.to_string())).collect(),
        CaptureTarget::Gamepad(bit) => cfg.pad_inputs_of(bit).iter().map(|p| short_source(&p.to_config_string())).collect(),
    };
    if parts.is_empty() {
        // The triggers press L2 / R2 from their analog value.
        match target {
            CaptureTarget::Gamepad(button::L2) => "analog L2".into(),
            CaptureTarget::Gamepad(button::R2) => "analog R2".into(),
            _ => "-".into(),
        }
    } else {
        parts.join(" / ")
    }
}

/// Controls driven by the held inputs: keys and mouse buttons through the
/// bindings of `set`, controller inputs through the controller layout.
pub fn pressed_controls(cfg: &Config, set: BindingSet, held: &[Source], pad: &[PadInput]) -> Vec<Control> {
    let mut v: Vec<Control> = Vec::new();
    let bindings = if set == BindingSet::Keyboard { &cfg.keyboard_bindings } else { &cfg.bindings };
    if set != BindingSet::Controller {
        for b in bindings {
            if held.contains(&b.source) && !v.contains(&b.control) {
                v.push(b.control);
            }
        }
    }
    let mut push = |c: Control| {
        if !v.contains(&c) {
            v.push(c);
        }
    };
    for b in &cfg.pad_bindings {
        if pad.contains(&b.input) {
            push(Control::Button(b.bit));
        }
    }
    for (input, bit) in [(PadInput::LEFT_TRIGGER, button::L2), (PadInput::RIGHT_TRIGGER, button::R2)] {
        if pad.contains(&input) && analog_trigger_reported(&cfg.pad_bindings, input, bit) {
            push(Control::Button(bit));
        }
    }
    v
}

/// Vertical positions (tops) of callouts `h` tall wanted centred at
/// `desired` (ascending) between `lo` and `hi`: in order, not overlapping,
/// each as close to its wish as the others allow.
pub fn spread(desired: &[f32], lo: f32, hi: f32, h: f32) -> Vec<f32> {
    let n = desired.len();
    let mut y: Vec<f32> = desired.iter().map(|d| d - h / 2.0).collect();
    for i in 0..n {
        let min = if i == 0 { lo } else { y[i - 1] + h };
        y[i] = y[i].max(min);
    }
    for i in (0..n).rev() {
        let max = if i + 1 == n { hi - h } else { y[i + 1] - h };
        y[i] = y[i].min(max);
    }
    // Too many to fit: keep the order from the top.
    for i in 0..n {
        let min = if i == 0 { lo } else { y[i - 1] + h };
        y[i] = y[i].max(min);
    }
    y
}

/// What the picture shows.
pub struct PictureParams<'a> {
    pub cfg: &'a Config,
    pub set: BindingSet,
    /// Selected hotspot (index into [`SPOTS`]).
    pub selected: Option<usize>,
    /// The selected hotspot waits for an input.
    pub capturing: bool,
    /// Controls whose inputs are held now.
    pub pressed: &'a [Control],
    /// Hotspots and callouts react to the pointer.
    pub interactive: bool,
    /// Widest the picture with its callouts may be.
    pub max_width: f32,
}

#[derive(Default)]
pub struct PictureResponse {
    pub clicked: Option<usize>,
    pub hovered: Option<usize>,
}

/// The controller drawing as a texture (loaded once per context).
pub fn body_texture(ctx: &egui::Context) -> egui::TextureHandle {
    let id = egui::Id::new("rps5e-dualsense-body");
    if let Some(t) = ctx.data(|d| d.get_temp::<egui::TextureHandle>(id)) {
        return t;
    }
    let image = decode_png(BODY_PNG).unwrap_or_else(|e| {
        log::warn!("controller drawing: {e}");
        egui::ColorImage::new([1, 1], Color32::TRANSPARENT)
    });
    let t = ctx.load_texture("dualsense-body", image, egui::TextureOptions::LINEAR);
    ctx.data_mut(|d| d.insert_temp(id, t.clone()));
    t
}

/// Decode an 8-bit PNG to an egui image.
pub fn decode_png(bytes: &[u8]) -> Result<egui::ColorImage, String> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().map_err(|e| e.to_string())?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let (w, h) = (info.width as usize, info.height as usize);
    let px = &buf[..info.buffer_size()];
    let rgba: Vec<u8> = match info.color_type {
        png::ColorType::Rgba => px.to_vec(),
        png::ColorType::Rgb => px.chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect(),
        png::ColorType::GrayscaleAlpha => px.chunks_exact(2).flat_map(|c| [c[0], c[0], c[0], c[1]]).collect(),
        png::ColorType::Grayscale => px.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::Indexed => return Err("indexed PNG not expanded".into()),
    };
    Ok(egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba))
}

const ROW_H: f32 = 22.0;
const GAP: f32 = 14.0;

fn accent() -> Color32 {
    Color32::from_rgb(70, 150, 255)
}

fn pressed_color() -> Color32 {
    Color32::from_rgb(90, 220, 120)
}

/// Draw the controller with its callouts.
pub fn picture(ui: &mut egui::Ui, p: &PictureParams) -> PictureResponse {
    let visible: Vec<usize> = (0..SPOTS.len()).filter(|&i| spot_visible(p.cfg, &SPOTS[i], p.set)).collect();
    let count = |side: Side| visible.iter().filter(|&&i| SPOTS[i].side == side).count();
    let rows = count(Side::Left).max(count(Side::Right)) as f32;

    let width = ui.available_width().min(p.max_width).max(420.0);
    let col_w = (width * 0.22).clamp(118.0, 200.0);
    let pic_w = width - 2.0 * (col_w + GAP);
    let pic_h = pic_w * IMAGE_SIZE[1] / IMAGE_SIZE[0];
    let band = pic_h.max(rows * ROW_H);
    let height = ROW_H + 8.0 + band + 8.0 + ROW_H;
    let sense = if p.interactive { Sense::click() } else { Sense::hover() };
    let (outer, resp) = ui.allocate_exact_size(Vec2::new(width, height), sense);
    let painter = ui.painter_at(outer.expand(2.0));

    let band_top = outer.top() + ROW_H + 8.0;
    let pic = Rect::from_min_size(Pos2::new(outer.left() + col_w + GAP, band_top + (band - pic_h) / 2.0), Vec2::new(pic_w, pic_h));
    let scale = pic_w / IMAGE_SIZE[0];
    let to_screen = |q: [f32; 2]| Pos2::new(pic.left() + q[0] * scale, pic.top() + q[1] * scale);
    let to_image = |q: Pos2| [(q.x - pic.left()) / scale, (q.y - pic.top()) / scale];

    // A light card behind the drawing: its dark parts stay visible on a
    // dark theme and over the game.
    let dark = ui.visuals().dark_mode;
    let card = if dark { Color32::from_rgb(78, 84, 96) } else { Color32::from_rgb(236, 238, 242) };
    painter.rect_filled(pic.expand2(Vec2::new(10.0, 6.0)), 14.0, card);
    painter.image(body_texture(ui.ctx()).id(), pic, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);

    // Callout boxes.
    let mut boxes: Vec<(usize, Rect)> = Vec::new();
    for side in [Side::Left, Side::Right] {
        let mut ids: Vec<usize> = visible.iter().copied().filter(|&i| SPOTS[i].side == side).collect();
        ids.sort_by(|&a, &b| Shape::anchor(&SPOTS[a])[1].total_cmp(&Shape::anchor(&SPOTS[b])[1]));
        let want: Vec<f32> = ids.iter().map(|&i| to_screen(Shape::anchor(&SPOTS[i])).y).collect();
        let ys = spread(&want, band_top, band_top + band, ROW_H);
        let x = if side == Side::Left { outer.left() } else { outer.right() - col_w };
        for (&i, y) in ids.iter().zip(ys) {
            boxes.push((i, Rect::from_min_size(Pos2::new(x, y + 1.0), Vec2::new(col_w, ROW_H - 3.0))));
        }
    }
    for &i in &visible {
        let s = &SPOTS[i];
        let w = (pic_w * 0.42).max(col_w);
        let cx = to_screen(s.shape.center()).x;
        match s.side {
            Side::Top => boxes.push((i, Rect::from_center_size(Pos2::new(cx, outer.top() + ROW_H / 2.0), Vec2::new(w, ROW_H - 3.0)))),
            _ => {}
        }
    }
    // The bottom callouts (PS, Mute) side by side under the drawing.
    let bottom: Vec<usize> = visible.iter().copied().filter(|&i| SPOTS[i].side == Side::Bottom).collect();
    if !bottom.is_empty() {
        let n = bottom.len() as f32;
        let w = ((pic_w + 2.0 * GAP) / n - GAP).min((pic_w * 0.42).max(col_w));
        let left = pic.center().x - (n * w + (n - 1.0) * GAP) / 2.0;
        for (k, &i) in bottom.iter().enumerate() {
            let x = left + k as f32 * (w + GAP);
            boxes.push((i, Rect::from_min_size(Pos2::new(x, outer.bottom() - ROW_H + 1.5), Vec2::new(w, ROW_H - 3.0))));
        }
    }

    // Pointer: a hotspot on the drawing or a callout.
    let mut out = PictureResponse::default();
    if p.interactive {
        if let Some(pos) = resp.hover_pos() {
            out.hovered = boxes.iter().find(|(_, r)| r.contains(pos)).map(|(i, _)| *i);
            if out.hovered.is_none() && pic.contains(pos) {
                out.hovered = hit(p.cfg, p.set, to_image(pos));
            }
        }
        if resp.clicked() {
            out.clicked = out.hovered;
        }
        if out.hovered.is_some() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
    }

    let text_color = ui.visuals().strong_text_color();
    let weak = ui.visuals().weak_text_color();
    let box_fill = if dark { Color32::from_rgba_unmultiplied(20, 22, 27, 230) } else { Color32::from_rgba_unmultiplied(250, 250, 252, 240) };
    let line_color = if dark { Color32::from_white_alpha(70) } else { Color32::from_black_alpha(70) };
    let name_font = FontId::proportional(11.5);
    let value_font = FontId::monospace(12.0);

    // Hotspot highlights, under the lines.
    for &i in &visible {
        let s = &SPOTS[i];
        let pressed = s.control.is_some_and(|c| p.pressed.contains(&c));
        let active = p.selected == Some(i) || out.hovered == Some(i);
        let color = if pressed {
            pressed_color()
        } else if active {
            accent()
        } else {
            continue;
        };
        let fill = color.gamma_multiply(0.35);
        let stroke = Stroke::new(2.0f32, color);
        match s.shape {
            Shape::Rect(r) => {
                let r = Rect::from_min_max(to_screen([r[0], r[1]]), to_screen([r[2], r[3]]));
                painter.rect(r, 6.0, fill, stroke, egui::StrokeKind::Middle);
            }
            Shape::Circle(c, r) => {
                painter.circle(to_screen(c), r * scale, fill, stroke);
            }
        }
    }

    for &(i, r) in &boxes {
        let s = &SPOTS[i];
        let pressed = s.control.is_some_and(|c| p.pressed.contains(&c));
        let selected = p.selected == Some(i);
        let hovered = out.hovered == Some(i);
        let highlight = if pressed {
            Some(pressed_color())
        } else if selected || hovered {
            Some(accent())
        } else {
            None
        };
        // Leader line from the callout to the hotspot.
        let anchor = to_screen(Shape::anchor(s));
        let from = match s.side {
            Side::Left => r.right_center(),
            Side::Right => r.left_center(),
            Side::Top => r.center_bottom(),
            Side::Bottom => r.center_top(),
        };
        let lc = highlight.unwrap_or(line_color);
        if matches!(s.side, Side::Left | Side::Right) {
            let elbow = Pos2::new(if s.side == Side::Left { pic.left() - 4.0 } else { pic.right() + 4.0 }, from.y);
            painter.line_segment([from, elbow], Stroke::new(1.0f32, lc));
            painter.line_segment([elbow, anchor], Stroke::new(1.0f32, lc));
        } else {
            painter.line_segment([from, anchor], Stroke::new(1.0f32, lc));
        }
        painter.circle_filled(anchor, if highlight.is_some() { 4.0 } else { 3.0 }, highlight.unwrap_or(Color32::from_rgb(255, 214, 0)));

        painter.rect(r, 5.0, box_fill, Stroke::new(if highlight.is_some() { 1.5f32 } else { 1.0 }, highlight.unwrap_or(line_color)), egui::StrokeKind::Inside);
        let inner = r.shrink2(Vec2::new(6.0, 0.0));
        let name = painter.layout_no_wrap(s.name.to_string(), name_font.clone(), weak);
        let name_w = name.size().x;
        painter.galley(Pos2::new(inner.left(), r.center().y - name.size().y / 2.0), name, weak);
        let (value, color) = if selected && p.capturing {
            ("press...".to_string(), if dark { Color32::from_rgb(255, 214, 0) } else { Color32::from_rgb(190, 110, 0) })
        } else {
            let off = s.stick.and_then(|st| stick_mode(p.cfg, st, p.set)) == Some(StickMode::Off);
            let fixed = fixed_text(s, p.set).is_some() || off;
            (spot_text(p.cfg, s, p.set), if fixed { weak } else { text_color })
        };
        let avail = (inner.width() - name_w - 8.0).max(10.0);
        let mut job = egui::text::LayoutJob::simple_singleline(value, value_font.clone(), color);
        job.wrap = egui::text::TextWrapping::truncate_at_width(avail);
        job.halign = egui::Align::RIGHT;
        let g = painter.layout_job(job);
        painter.galley(Pos2::new(inner.right(), r.center().y - g.size().y / 2.0), g, color);
    }
    // Hover: the full callout text (long ones are cut) and what it means.
    if let Some(i) = out.hovered {
        let s = &SPOTS[i];
        let text = spot_text(p.cfg, s, p.set);
        resp.on_hover_ui_at_pointer(|ui| {
            ui.label(RichText::new(s.name).strong());
            if !text.is_empty() {
                ui.label(RichText::new(text).monospace());
            }
            if let Some(h) = spot_help(s, p.set) {
                ui.label(h);
            }
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps5gpu::input::{KeyCode, MouseButton};

    #[test]
    fn every_bindable_control_has_a_hotspot() {
        // All pad buttons of the binding table and the stick and touch
        // actions are on the drawing.
        for (name, bit) in ps5input::mapping::pad_button_actions() {
            assert!(SPOTS.iter().any(|s| s.control == Some(Control::Button(bit))), "{name}");
        }
        for c in [
            Control::LeftStickUp,
            Control::RightStickDown,
            Control::TouchLeft,
            Control::TouchRight,
            Control::ToggleMenu,
            Control::Create,
            Control::Mute,
        ] {
            assert!(SPOTS.iter().any(|s| s.control == Some(c)), "{c:?}");
        }
        // Motion and touch actions are on the drawing or in the list under it.
        for &(name, c) in ps5input::mapping::ACTIONS {
            if c.is_motion() || c.is_touch() {
                let listed = MOTION_ACTIONS.contains(&c) || TOUCH_ACTIONS.contains(&c) || SPOTS.iter().any(|s| s.control == Some(c));
                assert!(listed, "{name}");
            }
        }
        for s in SPOTS {
            let [x, y] = s.shape.center();
            assert!((0.0..=IMAGE_SIZE[0]).contains(&x) && (0.0..=IMAGE_SIZE[1]).contains(&y), "{}", s.name);
        }
    }

    #[test]
    fn hotspot_targets_per_set() {
        let find = |n: &str| SPOTS.iter().find(|s| s.name == n).unwrap();
        let cross = find("Cross");
        assert_eq!(spot_target(cross, BindingSet::Controller), Some(CaptureTarget::Gamepad(button::CROSS)));
        assert_eq!(spot_target(cross, BindingSet::Keyboard), Some(CaptureTarget::Keyboard(Control::Button(button::CROSS))));
        // Stick directions and touch halves are keyboard actions only.
        let up = find("LS up");
        assert!(spot_target(up, BindingSet::Controller).is_none() && !spot_visible(&Config::default(), up, BindingSet::Controller));
        assert_eq!(spot_target(up, BindingSet::KeyboardMouse), Some(CaptureTarget::KeyboardMouse(Control::LeftStickUp)));
        // PS opens the menu: a hotkey on the keyboard, fixed on a controller.
        let ps = find("PS");
        assert!(spot_target(ps, BindingSet::Controller).is_none() && spot_visible(&Config::default(), ps, BindingSet::Controller));
        assert_eq!(spot_text(&Config::default(), ps, BindingSet::Controller), "in-game menu");
        // Create and Mute: keys on the keyboard sets, system buttons on a
        // controller.
        assert_eq!(spot_target(find("Create"), BindingSet::KeyboardMouse), Some(CaptureTarget::KeyboardMouse(Control::Create)));
        assert_eq!(spot_target(find("Mute"), BindingSet::Keyboard), Some(CaptureTarget::Keyboard(Control::Mute)));
        assert!(spot_target(find("Create"), BindingSet::Controller).is_none());
        assert_eq!(spot_text(&Config::default(), find("Mute"), BindingSet::Controller), "system only");
        assert_eq!(spot_text(&Config::default(), find("Create"), BindingSet::KeyboardMouse), "F12");
    }

    #[test]
    fn hit_prefers_the_smaller_zone() {
        // The touch pad halves sit on the touch pad; the stick press inside
        // the stick ring.
        let cfg = Config::default();
        let tl = SPOTS.iter().position(|s| s.name == "Touch left").unwrap();
        let tp = SPOTS.iter().position(|s| s.name == "Touch click").unwrap();
        assert_eq!(hit(&cfg, BindingSet::KeyboardMouse, [330.0, 200.0]), Some(tl));
        assert_eq!(hit(&cfg, BindingSet::Controller, [330.0, 200.0]), Some(tp));
        assert_eq!(hit(&cfg, BindingSet::KeyboardMouse, [500.0, 200.0]), Some(tp));
        let l3 = SPOTS.iter().position(|s| s.name == "L3").unwrap();
        assert_eq!(hit(&cfg, BindingSet::Controller, LS), Some(l3));
        assert_eq!(hit(&cfg, BindingSet::KeyboardMouse, LS), Some(l3));
        assert_eq!(hit(&cfg, BindingSet::Controller, [5.0, 550.0]), None);
        // The right stick on the mouse: its ring is one zone.
        let rs = SPOTS.iter().position(|s| s.name == "Right stick").unwrap();
        assert_eq!(hit(&cfg, BindingSet::KeyboardMouse, [RS[0], RS[1] - DIR_OFF]), Some(rs));
    }

    #[test]
    fn callout_texts_are_short() {
        let cfg = Config::default();
        let find = |n: &str| SPOTS.iter().find(|s| s.name == n).unwrap();
        assert_eq!(short_source("KEY:Space"), "Space");
        assert_eq!(short_source("KEY:Left Shift"), "LShift");
        assert_eq!(short_source("KEY:Right Ctrl"), "RCtrl");
        assert_eq!(short_source("KEY:Left"), "Left");
        assert_eq!(spot_text(&cfg, find("L3"), BindingSet::KeyboardMouse), "LShift / RShift");
        assert_eq!(short_source("MOUSE:Right"), "Mouse Right");
        assert_eq!(short_source("WHEEL:Up"), "Wheel Up");
        assert_eq!(short_source("BUTTON:South"), "Cross");
        assert_eq!(short_source("BUTTON:LeftShoulder"), "L1");
        assert_eq!(short_source("BUTTON:DpadUp"), "Up");
        assert_eq!(spot_text(&cfg, find("L2"), BindingSet::Controller), "analog L2");
        assert_eq!(spot_text(&cfg, find("Cross"), BindingSet::Controller), "Cross");
        assert!(spot_text(&cfg, find("Cross"), BindingSet::KeyboardMouse).contains("Space"));
        let mut cfg = cfg;
        cfg.set_pad_inputs(button::SQUARE, &[]);
        assert_eq!(spot_text(&cfg, find("Square"), BindingSet::Controller), "-");
    }

    #[test]
    fn held_inputs_light_their_controls() {
        let mut cfg = Config::default();
        let space = Source::Key(KeyCode::Space);
        let p = pressed_controls(&cfg, BindingSet::KeyboardMouse, &[space], &[]);
        assert_eq!(p, [Control::Button(button::CROSS)]);
        // Controller set: keys do not count, controller inputs do.
        assert!(pressed_controls(&cfg, BindingSet::Controller, &[space], &[]).is_empty());
        cfg.set_pad_inputs(button::L1, &[PadInput(9)]);
        assert_eq!(pressed_controls(&cfg, BindingSet::Controller, &[], &[PadInput(9)]), [Control::Button(button::L1)]);
        // A held trigger lights L2 unless it was rebound to another button.
        assert_eq!(pressed_controls(&cfg, BindingSet::Controller, &[], &[PadInput::LEFT_TRIGGER]), [Control::Button(button::L2)]);
        cfg.set_pad_inputs(button::CROSS, &[PadInput::LEFT_TRIGGER]);
        assert_eq!(pressed_controls(&cfg, BindingSet::Controller, &[], &[PadInput::LEFT_TRIGGER]), [Control::Button(button::CROSS)]);
        let m = Source::Mouse(MouseButton::Right);
        let _ = pressed_controls(&cfg, BindingSet::Keyboard, &[m], &[]);
        assert!(pressed_controls(&cfg, BindingSet::Keyboard, &[m], &[]).iter().all(|c| cfg.sources_of(*c, true).contains(&m)));
    }

    #[test]
    fn mouse_stick_has_one_callout() {
        let names = |cfg: &Config, set| -> Vec<&str> {
            SPOTS.iter().filter(|s| spot_visible(cfg, s, set)).map(|s| s.name).collect()
        };
        let stick_spots = |v: &[&str], p: &str| v.iter().filter(|n| n.starts_with(p)).count();
        // Default keyboard + mouse: left on keys (4 directions), right on
        // the mouse (one callout).
        let mut cfg = Config::default();
        let v = names(&cfg, BindingSet::KeyboardMouse);
        assert_eq!(stick_spots(&v, "LS "), 4);
        assert_eq!(stick_spots(&v, "RS "), 0);
        assert!(v.contains(&"Right stick") && !v.contains(&"Left stick"));
        let rs = SPOTS.iter().find(|s| s.name == "Right stick").unwrap();
        assert_eq!(spot_text(&cfg, rs, BindingSet::KeyboardMouse), "Mouse \u{00d7}1.0");
        cfg.mouse_sensitivity = 2.5;
        cfg.mouse_invert_y = true;
        assert_eq!(spot_text(&cfg, rs, BindingSet::KeyboardMouse), "Mouse \u{00d7}2.5 inv");
        // Mouse on the left stick: the right one shows its keys.
        cfg.set_stick_mode(Stick::Left, StickMode::Mouse);
        let v = names(&cfg, BindingSet::KeyboardMouse);
        assert_eq!((stick_spots(&v, "LS "), stick_spots(&v, "RS ")), (0, 4));
        assert!(v.contains(&"Left stick") && !v.contains(&"Right stick"));
        // Keyboard only: no mouse, the stick is on its keys.
        let v = names(&cfg, BindingSet::Keyboard);
        assert_eq!((stick_spots(&v, "LS "), stick_spots(&v, "RS ")), (4, 4));
        // Controller only: one "controller only" callout.
        cfg.set_stick_mode(Stick::Right, StickMode::Off);
        let v = names(&cfg, BindingSet::Keyboard);
        assert!(v.contains(&"Right stick") && stick_spots(&v, "RS ") == 0);
        assert_eq!(spot_text(&cfg, rs, BindingSet::Keyboard), "controller only");
        // Controller set: no stick callouts.
        let v = names(&cfg, BindingSet::Controller);
        assert!(!v.iter().any(|n| n.contains("stick") || n.starts_with("LS") || n.starts_with("RS")));
        // Create explains itself.
        let create = SPOTS.iter().find(|s| s.name == "Create").unwrap();
        assert!(spot_help(create, BindingSet::Controller).unwrap().contains("system"));
        assert!(spot_help(create, BindingSet::KeyboardMouse).unwrap().contains("screenshot"));
    }

    #[test]
    fn callouts_spread_without_overlap() {
        let y = spread(&[10.0, 12.0, 14.0, 200.0], 0.0, 300.0, 20.0);
        assert_eq!(y[0], 0.0);
        for w in y.windows(2) {
            assert!(w[1] - w[0] >= 20.0 - 1e-3);
        }
        assert_eq!(y[3], 190.0, "a callout with room stays at its hotspot");
        // Crowded at the bottom: pushed up, still inside.
        let y = spread(&[290.0, 295.0, 299.0], 0.0, 300.0, 20.0);
        assert_eq!(y, [240.0, 260.0, 280.0]);
        // More than fit: ordered from the top.
        let y = spread(&[5.0; 4], 0.0, 50.0, 20.0);
        assert_eq!(y, [0.0, 20.0, 40.0, 60.0]);
    }

    #[test]
    fn drawing_decodes() {
        let img = decode_png(BODY_PNG).unwrap();
        assert_eq!(img.size, [IMAGE_SIZE[0] as usize, IMAGE_SIZE[1] as usize]);
    }

    /// Render the picture (and, for the editor, its detail row) offscreen.
    fn render(dark: bool, width: f32, f: impl Fn(&mut egui::Ui)) -> (usize, usize, Vec<u8>) {
        render_sized(dark, [width, 520.0], f)
    }

    fn render_sized(dark: bool, size: [f32; 2], f: impl Fn(&mut egui::Ui)) -> (usize, usize, Vec<u8>) {
        let mut r = crate::softrender::SoftRenderer::new(size, 1.5);
        r.ctx.set_theme(if dark { egui::Theme::Dark } else { egui::Theme::Light });
        let mut last = (0, 0, Vec::new());
        for _ in 0..2 {
            last = r.frame(Vec::new(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| f(ui));
            });
        }
        last
    }

    fn pixel(img: &(usize, usize, Vec<u8>), x: usize, y: usize) -> [u8; 3] {
        let i = (y * img.0 + x) * 4;
        [img.2[i], img.2[i + 1], img.2[i + 2]]
    }

    #[test]
    fn held_button_lights_up_on_the_picture() {
        let cfg = Config::default();
        let draw = |pressed: Vec<Control>| {
            let cfg = &cfg;
            render(true, 900.0, move |ui| {
                let p = PictureParams {
                    cfg,
                    set: BindingSet::KeyboardMouse,
                    selected: None,
                    capturing: false,
                    pressed: &pressed,
                    interactive: false,
                    max_width: 900.0,
                };
                picture(ui, &p);
            })
        };
        let off = draw(Vec::new());
        let on = draw(vec![Control::Button(button::CROSS)]);
        let diff: Vec<(usize, usize)> = (0..off.1)
            .flat_map(|y| (0..off.0).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                let (a, b) = (pixel(&off, x, y), pixel(&on, x, y));
                b[1] as i32 - a[1] as i32 > 20 && b[1] > b[0]
            })
            .collect();
        assert!(diff.len() > 500, "the Cross hotspot turns green ({} pixels)", diff.len());
        // The change is on the right (Cross and its callout), not elsewhere.
        let cx = diff.iter().map(|d| d.0).sum::<usize>() / diff.len();
        assert!(cx > off.0 / 2, "lit area centre x {cx} of {}", off.0);
    }

    /// `RPS5E_GUI_RENDER_DIR=<dir> cargo test -p rps5e-gui render_pictures -- --ignored`
    /// writes the picture in its states as PNG files to look at.
    #[test]
    #[ignore]
    fn render_pictures() {
        let Some(dir) = std::env::var_os("RPS5E_GUI_RENDER_DIR") else { return };
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let save = |name: &str, img: (usize, usize, Vec<u8>)| crate::softrender::write_png(&dir.join(name), img.0, img.1, &img.2);
        let cfg = Config::default();
        let cross = SPOTS.iter().position(|s| s.name == "Cross");
        // Stick modes: keyboard + mouse default (right stick on the mouse),
        // the mouse on the left stick, keyboard only; the editor with the
        // mouse stick selected, and the table.
        let mut left = cfg.clone();
        left.set_stick_mode(Stick::Left, StickMode::Mouse);
        let rs = SPOTS.iter().position(|s| s.name == "Right stick");
        let ls = SPOTS.iter().position(|s| s.name == "Left stick");
        for (name, c, set, sel, view) in [
            ("sticks-kbm-default.png", &cfg, BindingSet::KeyboardMouse, rs, crate::input_ui::BindingView::Picture),
            ("sticks-mouse-left.png", &left, BindingSet::KeyboardMouse, ls, crate::input_ui::BindingView::Picture),
            ("sticks-keyboard-only.png", &left, BindingSet::Keyboard, None, crate::input_ui::BindingView::Picture),
            ("sticks-table-kbm.png", &cfg, BindingSet::KeyboardMouse, None, crate::input_ui::BindingView::Table),
        ] {
            save(
                name,
                render_sized(true, [980.0, 860.0], |ui| {
                    let mut e = crate::input_ui::BindingEditor::default();
                    (e.set, e.view, e.selected) = (set, view, sel);
                    let mut c = c.clone();
                    egui::ScrollArea::vertical().show(ui, |ui| e.show(ui, &mut c));
                }),
            );
        }
        for (name, set, dark) in [
            ("editor-kbm-dark.png", BindingSet::KeyboardMouse, true),
            ("editor-pad-light.png", BindingSet::Controller, false),
            ("editor-keyboard-dark.png", BindingSet::Keyboard, true),
        ] {
            save(
                name,
                render(dark, 980.0, |ui| {
                    let pressed = [Control::Button(button::L1)];
                    let p = PictureParams { cfg: &cfg, set, selected: cross, capturing: true, pressed: &pressed, interactive: true, max_width: 980.0 };
                    picture(ui, &p);
                }),
            );
        }
        // Every hotspot lit: checks the hotspot table against the drawing.
        for (name, set) in [("spots-kbm.png", BindingSet::KeyboardMouse), ("spots-pad.png", BindingSet::Controller)] {
            save(
                name,
                render(true, 980.0, |ui| {
                    let all: Vec<Control> = SPOTS.iter().filter_map(|s| s.control).collect();
                    let p = PictureParams { cfg: &cfg, set, selected: None, capturing: false, pressed: &all, interactive: false, max_width: 980.0 };
                    picture(ui, &p);
                }),
            );
        }
        // The motion and touch pad list under the picture.
        for (name, set) in [("motion-touch-kbm.png", BindingSet::KeyboardMouse), ("motion-touch-keyboard.png", BindingSet::Keyboard)] {
            save(
                name,
                render_sized(true, [980.0, 760.0], |ui| {
                    let mut e = crate::input_ui::BindingEditor::default();
                    e.set = set;
                    let mut c = cfg.clone();
                    e.extra_actions(ui, &mut c);
                }),
            );
        }
        // Swapped sticks.
        let mut swapped = cfg.clone();
        swapped.swap_sticks = true;
        save(
            "sticks-swapped.png",
            render_sized(true, [980.0, 860.0], |ui| {
                let mut e = crate::input_ui::BindingEditor::default();
                let mut c = swapped.clone();
                e.show(ui, &mut c);
            }),
        );
        // The in-game menu's width, with the editor around it.
        save(
            "menu-width.png",
            render(true, 700.0, |ui| {
                let mut e = crate::input_ui::BindingEditor::default();
                let mut c = cfg.clone();
                e.show(ui, &mut c);
            }),
        );
    }
}
