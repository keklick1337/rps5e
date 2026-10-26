// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The launcher's look: dark and light palettes built around the logo's
//! blue, text sizes and spacing, and the small widgets the pages share
//! (cards, pills, primary buttons, section titles, the logo texture).

use eframe::egui::{self, Color32, CornerRadius, FontId, RichText, Stroke, TextStyle, Vec2};

/// Colours of one theme.
#[derive(Clone, Copy, Debug)]
pub struct Palette {
    /// Window background (pages).
    pub bg: Color32,
    /// Sidebar and bars.
    pub chrome: Color32,
    /// Cards and panels.
    pub card: Color32,
    /// Hovered card / row.
    pub hover: Color32,
    pub border: Color32,
    pub text: Color32,
    pub weak: Color32,
    pub accent: Color32,
    pub accent_text: Color32,
    pub danger: Color32,
    pub ok: Color32,
    pub warn: Color32,
}

pub const DARK: Palette = Palette {
    bg: Color32::from_rgb(0x15, 0x17, 0x1c),
    chrome: Color32::from_rgb(0x0f, 0x11, 0x15),
    card: Color32::from_rgb(0x1e, 0x21, 0x28),
    hover: Color32::from_rgb(0x28, 0x2c, 0x35),
    border: Color32::from_rgb(0x2e, 0x33, 0x3d),
    text: Color32::from_rgb(0xe8, 0xea, 0xee),
    weak: Color32::from_rgb(0x95, 0x9c, 0xa8),
    accent: Color32::from_rgb(0x3d, 0x7b, 0xf0),
    accent_text: Color32::WHITE,
    danger: Color32::from_rgb(0xe5, 0x53, 0x4b),
    ok: Color32::from_rgb(0x4c, 0xaf, 0x50),
    warn: Color32::from_rgb(0xf0, 0xb4, 0x29),
};

pub const LIGHT: Palette = Palette {
    bg: Color32::from_rgb(0xf3, 0xf4, 0xf7),
    chrome: Color32::from_rgb(0xe6, 0xe8, 0xed),
    card: Color32::WHITE,
    hover: Color32::from_rgb(0xe9, 0xee, 0xf8),
    border: Color32::from_rgb(0xd5, 0xd9, 0xe1),
    text: Color32::from_rgb(0x1b, 0x1e, 0x24),
    weak: Color32::from_rgb(0x5f, 0x66, 0x73),
    accent: Color32::from_rgb(0x2a, 0x63, 0xd6),
    accent_text: Color32::WHITE,
    danger: Color32::from_rgb(0xc6, 0x32, 0x2b),
    ok: Color32::from_rgb(0x2e, 0x8b, 0x37),
    warn: Color32::from_rgb(0xb7, 0x7c, 0x00),
};

/// The palette of the theme `ui` shows.
pub fn pal(ui: &egui::Ui) -> Palette {
    if ui.visuals().dark_mode {
        DARK
    } else {
        LIGHT
    }
}

/// The palette of the theme `ctx` shows.
pub fn pal_ctx(ctx: &egui::Context) -> Palette {
    if ctx.style().visuals.dark_mode {
        DARK
    } else {
        LIGHT
    }
}

fn visuals(p: &Palette, dark: bool) -> egui::Visuals {
    let mut v = if dark { egui::Visuals::dark() } else { egui::Visuals::light() };
    // Small enough that check boxes stay square.
    let r = CornerRadius::same(4);
    v.panel_fill = p.bg;
    v.window_fill = p.card;
    v.extreme_bg_color = if dark { Color32::from_rgb(0x10, 0x12, 0x16) } else { Color32::WHITE };
    v.faint_bg_color = if dark { Color32::from_rgb(0x1a, 0x1d, 0x23) } else { Color32::from_rgb(0xf6, 0xf7, 0xf9) };
    v.code_bg_color = v.extreme_bg_color;
    v.window_stroke = Stroke::new(1.0f32, p.border);
    v.window_corner_radius = CornerRadius::same(10);
    v.menu_corner_radius = CornerRadius::same(8);
    v.hyperlink_color = p.accent;
    v.selection.bg_fill = p.accent.gamma_multiply(if dark { 0.55 } else { 0.35 });
    v.selection.stroke = Stroke::new(1.0f32, p.accent);
    v.widgets.noninteractive.bg_fill = p.card;
    v.widgets.noninteractive.weak_bg_fill = p.card;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0f32, p.border);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0f32, p.text);
    v.widgets.noninteractive.corner_radius = r;
    let btn = if dark { Color32::from_rgb(0x2a, 0x2f, 0x39) } else { Color32::from_rgb(0xe4, 0xe7, 0xee) };
    let btn_hover = if dark { Color32::from_rgb(0x34, 0x3a, 0x46) } else { Color32::from_rgb(0xd8, 0xde, 0xea) };
    v.widgets.inactive.bg_fill = btn;
    v.widgets.inactive.weak_bg_fill = btn;
    v.widgets.inactive.bg_stroke = Stroke::NONE;
    v.widgets.inactive.fg_stroke = Stroke::new(1.0f32, p.text);
    v.widgets.inactive.corner_radius = r;
    v.widgets.hovered.bg_fill = btn_hover;
    v.widgets.hovered.weak_bg_fill = btn_hover;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0f32, p.accent.gamma_multiply(0.6));
    v.widgets.hovered.fg_stroke = Stroke::new(1.5f32, p.text);
    v.widgets.hovered.corner_radius = r;
    v.widgets.active.bg_fill = p.accent.gamma_multiply(0.8);
    v.widgets.active.weak_bg_fill = p.accent.gamma_multiply(0.8);
    v.widgets.active.fg_stroke = Stroke::new(1.5f32, p.text);
    v.widgets.active.corner_radius = r;
    v.widgets.open.bg_fill = btn_hover;
    v.widgets.open.weak_bg_fill = btn_hover;
    v.widgets.open.corner_radius = r;
    v
}

/// Install both palettes, the text sizes and the spacing.
pub fn install(ctx: &egui::Context) {
    ctx.set_visuals_of(egui::Theme::Dark, visuals(&DARK, true));
    ctx.set_visuals_of(egui::Theme::Light, visuals(&LIGHT, false));
    ctx.all_styles_mut(|s| {
        use egui::FontFamily::{Monospace, Proportional};
        s.text_styles = [
            (TextStyle::Heading, FontId::new(17.0, Proportional)),
            (TextStyle::Body, FontId::new(14.0, Proportional)),
            (TextStyle::Button, FontId::new(14.0, Proportional)),
            (TextStyle::Small, FontId::new(11.5, Proportional)),
            (TextStyle::Monospace, FontId::new(12.5, Monospace)),
        ]
        .into();
        s.spacing.item_spacing = Vec2::new(8.0, 7.0);
        s.spacing.button_padding = Vec2::new(12.0, 5.0);
        s.spacing.interact_size.y = 26.0;
        s.spacing.window_margin = egui::Margin::same(14);
        s.spacing.menu_margin = egui::Margin::same(8);
        s.spacing.combo_width = 160.0;
        s.spacing.scroll.bar_width = 8.0;
        s.spacing.scroll.floating = true;
    });
}

/// A card: a rounded panel with a border.
pub fn card(ui: &egui::Ui) -> egui::Frame {
    let p = pal(ui);
    egui::Frame::new().fill(p.card).stroke(Stroke::new(1.0f32, p.border)).corner_radius(10).inner_margin(egui::Margin::same(16))
}

/// A card's title with an optional line under it.
pub fn card_title(ui: &mut egui::Ui, title: &str, note: Option<&str>) {
    ui.label(RichText::new(title).size(16.5).strong().color(pal(ui).text));
    if let Some(n) = note {
        ui.label(RichText::new(n).color(pal(ui).weak));
    }
    ui.add_space(4.0);
}

/// A page title with a subtitle.
pub fn page_title(ui: &mut egui::Ui, title: &str, subtitle: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(title).size(24.0).strong().color(pal(ui).text));
        if !subtitle.is_empty() {
            ui.add_space(6.0);
            ui.label(RichText::new(subtitle).size(14.0).color(pal(ui).weak));
        }
    });
}

/// The main action of a view: filled with the accent colour.
pub fn primary_button(ui: &mut egui::Ui, enabled: bool, text: &str, size: Vec2) -> egui::Response {
    let p = pal(ui);
    let fill = if enabled { p.accent } else { p.accent.gamma_multiply(0.35) };
    ui.add_enabled(
        enabled,
        egui::Button::new(RichText::new(text).size(16.0).strong().color(p.accent_text)).fill(fill).min_size(size).corner_radius(8),
    )
}

/// A destructive action (Stop): red.
pub fn danger_button(ui: &mut egui::Ui, text: &str, size: Vec2) -> egui::Response {
    let p = pal(ui);
    ui.add(egui::Button::new(RichText::new(text).size(15.0).strong().color(Color32::WHITE)).fill(p.danger).min_size(size).corner_radius(8))
}

/// A secondary action with a fixed size.
pub fn button(ui: &mut egui::Ui, enabled: bool, text: &str, size: Vec2) -> egui::Response {
    ui.add_enabled(enabled, egui::Button::new(text).min_size(size).corner_radius(7))
}

/// A small rounded label: a title id, a count, a state.
pub fn pill(ui: &mut egui::Ui, text: &str, color: Color32) -> egui::Response {
    let font = FontId::proportional(11.5);
    let galley = ui.painter().layout_no_wrap(text.to_string(), font, color);
    let size = galley.size() + Vec2::new(14.0, 6.0);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(9), color.gamma_multiply(0.16));
    ui.painter().galley(rect.center() - galley.size() / 2.0, galley, color);
    resp
}

/// An on/off switch; the response is changed when it was flipped.
pub fn toggle(ui: &mut egui::Ui, on: &mut bool) -> egui::Response {
    let p = pal(ui);
    let size = Vec2::new(36.0, 20.0);
    let (rect, mut resp) = ui.allocate_exact_size(size, egui::Sense::click());
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    let t = ui.ctx().animate_bool_responsive(resp.id, *on);
    let off = if ui.visuals().dark_mode { Color32::from_rgb(0x3a, 0x40, 0x4c) } else { Color32::from_rgb(0xc4, 0xc9, 0xd3) };
    let fill = lerp_color(off, p.accent, t);
    ui.painter().rect_filled(rect, CornerRadius::same(10), fill);
    let x = egui::lerp((rect.left() + 10.0)..=(rect.right() - 10.0), t);
    ui.painter().circle_filled(egui::pos2(x, rect.center().y), 7.0, Color32::WHITE);
    resp
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let m = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgb(m(a.r(), b.r()), m(a.g(), b.g()), m(a.b(), b.b()))
}

/// A filled status dot.
pub fn dot(ui: &mut egui::Ui, color: Color32, size: f32) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size + 4.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), size / 2.0, color);
    resp
}

/// A segmented choice (tabs, view switch): the selected segment filled.
pub fn segmented<T: PartialEq + Copy>(ui: &mut egui::Ui, value: &mut T, options: &[(T, &str)]) -> bool {
    let p = pal(ui);
    let mut changed = false;
    egui::Frame::new().fill(p.chrome).corner_radius(8).inner_margin(egui::Margin::same(3)).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for (v, label) in options {
                let on = *value == *v;
                let text = RichText::new(*label).color(if on { p.accent_text } else { p.weak });
                let b =
                    egui::Button::new(text).fill(if on { p.accent } else { Color32::TRANSPARENT }).corner_radius(6).min_size(Vec2::new(0.0, 24.0));
                if ui.add(b).clicked() && !on {
                    *value = *v;
                    changed = true;
                }
            }
        });
    });
    changed
}

/// The logo (128 px PNG of the icon set).
pub fn logo(ctx: &egui::Context) -> Option<egui::TextureHandle> {
    let id = egui::Id::new("rps5e-logo");
    if let Some(t) = ctx.data(|d| d.get_temp::<egui::TextureHandle>(id)) {
        return Some(t);
    }
    let bytes: &[u8] = include_bytes!("../../../assets/icons/rps5e-128.png");
    let (w, h, rgba) = crate::library::decode_png_rgba(bytes)?;
    let img = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
    let tex = ctx.load_texture("rps5e-logo", img, egui::TextureOptions::LINEAR);
    ctx.data_mut(|d| d.insert_temp(id, tex.clone()));
    Some(tex)
}

/// "3 min", "1 h 05 min", "12 s".
pub fn duration_text(secs: u64) -> String {
    match secs {
        0..=59 => format!("{secs} s"),
        60..=3599 => format!("{} min", secs / 60),
        _ => format!("{} h {:02} min", secs / 3600, secs % 3600 / 60),
    }
}

/// How long ago `then` (seconds since the epoch) was, from `now`.
pub fn ago_text(then: u64, now: u64) -> String {
    let d = now.saturating_sub(then);
    match d {
        0..=59 => "just now".into(),
        60..=3599 => format!("{} min ago", d / 60),
        3600..=86_399 => format!("{} h ago", d / 3600),
        86_400..=172_799 => "yesterday".into(),
        _ => format!("{} days ago", d / 86_400),
    }
}

/// Seconds since the Unix epoch.
pub fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_texts() {
        assert_eq!(duration_text(12), "12 s");
        assert_eq!(duration_text(185), "3 min");
        assert_eq!(duration_text(3600 + 5 * 60), "1 h 05 min");
        assert_eq!(ago_text(100, 110), "just now");
        assert_eq!(ago_text(0, 7200), "2 h ago");
        assert_eq!(ago_text(0, 100_000), "yesterday");
        assert_eq!(ago_text(0, 86_400 * 9), "9 days ago");
        assert_eq!(ago_text(500, 100), "just now", "clock went back");
    }

    #[test]
    fn logo_decodes() {
        let ctx = egui::Context::default();
        let t = logo(&ctx).expect("the logo PNG decodes");
        assert_eq!(t.size(), [128, 128]);
        assert_eq!(logo(&ctx).unwrap().id(), t.id(), "loaded once");
    }
}
