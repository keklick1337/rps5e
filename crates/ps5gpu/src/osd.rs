// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! On-screen hints of the presentation window (mouse capture, hotkeys):
//! a line of text rasterised on the CPU into an opaque box, copied into
//! the corner of the swapchain image after the frame.

use ab_glyph::{point, Font, FontRef, PxScale, ScaleFont};
use ash::vk;
use std::sync::Mutex;
use std::time::{Duration, Instant};

struct Hint {
    text: String,
    duration: Duration,
    /// Set when the hint is first drawn (the window may open later).
    until: Option<Instant>,
}

static HINT: Mutex<Option<Hint>> = Mutex::new(None);

/// Show `text` in the game window for `duration` (replaces any hint).
pub fn show_hint(text: impl Into<String>, duration: Duration) {
    let text = text.into();
    log::info!("hint: {text}");
    *HINT.lock().unwrap_or_else(|e| e.into_inner()) = Some(Hint { text, duration, until: None });
    crate::input::wake();
}

/// The hint to draw now, if any; its time starts on the first call.
pub(crate) fn current_hint() -> Option<String> {
    let mut h = HINT.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    let hint = h.as_mut()?;
    let until = *hint.until.get_or_insert(now + hint.duration);
    if now < until {
        Some(hint.text.clone())
    } else {
        *h = None;
        None
    }
}

static STATUS: Mutex<Option<String>> = Mutex::new(None);

/// Show `text` in the window (title and corner, over black before the
/// game's first frame) until replaced or cleared with None: a long task
/// before the game starts (the shader prewarm).
pub fn set_status(text: Option<String>) {
    *STATUS.lock().unwrap_or_else(|e| e.into_inner()) = text;
    crate::input::wake();
}

/// The status line shown now, if any.
pub(crate) fn status() -> Option<String> {
    STATUS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Shader compilation shown over the game window: the full
/// progress panel while the game waits for its shaders, else a small
/// "compiling N shaders" indicator while draws are skipped for compiles.
#[derive(Clone, Debug, PartialEq)]
pub struct CompileProgress {
    /// Jobs (module translations and pipelines) done and in all.
    pub done: usize,
    pub total: usize,
    /// Time left at the rate so far.
    pub eta: Option<Duration>,
    /// A line under the bar (what happens when it finishes).
    pub note: String,
}

impl CompileProgress {
    /// Percent done (0 while the list loads: no jobs known yet).
    pub fn percent(&self) -> u32 {
        if self.total == 0 {
            0
        } else {
            (self.done.min(self.total) as u64 * 100 / self.total as u64) as u32
        }
    }

    /// `1234 / 5678 shaders and pipelines (21%)`.
    pub fn counts(&self) -> String {
        if self.total == 0 {
            return "Loading the shader list".into();
        }
        format!("{} / {} shaders and pipelines ({}%)", self.done.min(self.total), self.total, self.percent())
    }

    /// The bar's fill, 0..1.
    pub fn fraction(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.done.min(self.total) as f32 / self.total as f32
        }
    }

    /// `about 1m 05s left` (None until a rate is known).
    pub fn eta_text(&self) -> Option<String> {
        self.eta.map(|t| if t < Duration::from_secs(1) { "less than a second left".into() } else { format!("about {} left", format_duration(t)) })
    }
}

/// `1m 05s`, `42s`.
pub fn format_duration(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 60 {
        format!("{}m {:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

/// What the compile overlay shows now.
#[derive(Clone, Debug, PartialEq)]
pub enum CompileOverlay {
    /// The game waits for its shaders: the full panel.
    Full(CompileProgress),
    /// Compiles in the background while the game runs: the jobs done and
    /// in all since they started piling up (the bar under the count).
    Background { done: u64, total: u64 },
}

static PROGRESS: Mutex<Option<CompileProgress>> = Mutex::new(None);

/// Show (Some) or hide (None) the full compile progress panel.
pub fn set_compile_progress(p: Option<CompileProgress>) {
    *PROGRESS.lock().unwrap_or_else(|e| e.into_inner()) = p;
    crate::input::wake();
}

/// The full compile progress shown now, if any.
pub fn compile_progress() -> Option<CompileProgress> {
    PROGRESS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// `RPS5E_COMPILE_INDICATOR=0`: no "compiling N shaders" indicator.
fn indicator_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_COMPILE_INDICATOR").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
}

/// Pending compiles show only after they have been pending this long: a
/// pipeline the driver cache holds is built in a millisecond and should
/// not flash the indicator.
pub const INDICATOR_DELAY: Duration = Duration::from_millis(300);

/// The background indicator's state: since when jobs have been pending,
/// and the finished-jobs count then (the progress counts from it).
#[derive(Default)]
pub struct Indicator {
    since: Option<(Instant, u64)>,
}

impl Indicator {
    /// (done, total) to show for `pending` jobs at `now`, `finished` jobs
    /// having finished in all so far (None: hidden). The total grows as
    /// the game asks for more while the earlier ones compile.
    pub fn update(&mut self, pending: usize, finished: u64, now: Instant) -> Option<(u64, u64)> {
        if pending == 0 {
            self.since = None;
            return None;
        }
        let (since, base) = *self.since.get_or_insert((now, finished));
        let done = finished.saturating_sub(base);
        (now.saturating_duration_since(since) >= INDICATOR_DELAY).then_some((done, done + pending as u64))
    }
}

static INDICATOR: Mutex<Indicator> = Mutex::new(Indicator { since: None });

/// The compile overlay to draw now, if any.
pub fn compile_overlay() -> Option<CompileOverlay> {
    if let Some(p) = compile_progress() {
        return Some(CompileOverlay::Full(p));
    }
    if !indicator_enabled() {
        return None;
    }
    let pending = crate::pipeline::pool::foreground_pending();
    let finished = crate::pipeline::pool::foreground_done();
    INDICATOR.lock().unwrap_or_else(|e| e.into_inner()).update(pending, finished, Instant::now()).map(|(done, total)| CompileOverlay::Background { done, total })
}

/// Time until the current hint expires (the window redraws then).
pub(crate) fn hint_deadline() -> Option<Instant> {
    HINT.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|h| h.until)
}

const BACKGROUND: [u8; 3] = [24, 24, 28];
const FOREGROUND: [u8; 3] = [240, 240, 240];
const PADDING: u32 = 10;

/// Rasterise `text` at `px` pixels into an RGBA8 box; (width, height, pixels).
pub fn rasterize(text: &str, px: f32) -> Option<(u32, u32, Vec<u8>)> {
    let font = FontRef::try_from_slice(epaint_default_fonts::HACK_REGULAR).ok()?;
    let scaled = font.as_scaled(PxScale::from(px));
    let mut caret = 0.0f32;
    let mut glyphs = Vec::new();
    let mut prev = None;
    for c in text.chars() {
        let id = scaled.glyph_id(c);
        if let Some(p) = prev {
            caret += scaled.kern(p, id);
        }
        glyphs.push(id.with_scale_and_position(PxScale::from(px), point(caret, scaled.ascent())));
        caret += scaled.h_advance(id);
        prev = Some(id);
    }
    let text_w = caret.ceil() as u32;
    let text_h = (scaled.ascent() - scaled.descent()).ceil() as u32;
    let (w, h) = (text_w + 2 * PADDING, text_h + 2 * PADDING);
    let mut cover = vec![0f32; (w * h) as usize];
    for g in glyphs {
        if let Some(outline) = font.outline_glyph(g) {
            let b = outline.px_bounds();
            outline.draw(|x, y, c| {
                let px_x = b.min.x as i32 + x as i32 + PADDING as i32;
                let px_y = b.min.y as i32 + y as i32 + PADDING as i32;
                if px_x >= 0 && px_y >= 0 && (px_x as u32) < w && (px_y as u32) < h {
                    let i = (px_y as u32 * w + px_x as u32) as usize;
                    cover[i] = (cover[i] + c).min(1.0);
                }
            });
        }
    }
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for c in cover {
        for ch in 0..3 {
            let v = BACKGROUND[ch] as f32 + (FOREGROUND[ch] as f32 - BACKGROUND[ch] as f32) * c;
            rgba.push(v.round() as u8);
        }
        rgba.push(255);
    }
    Some((w, h, rgba))
}

/// Convert RGBA8 pixels to the swapchain format's texel bytes; `None` for
/// formats the hint cannot be written in.
pub fn convert(rgba: &[u8], format: vk::Format) -> Option<Vec<u8>> {
    use vk::Format as F;
    match format {
        F::R8G8B8A8_UNORM | F::R8G8B8A8_SRGB => Some(rgba.to_vec()),
        F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB => {
            Some(rgba.chunks_exact(4).flat_map(|p| [p[2], p[1], p[0], p[3]]).collect())
        }
        F::A2B10G10R10_UNORM_PACK32 => Some(
            rgba.chunks_exact(4)
                .flat_map(|p| {
                    let c = |v: u8| (v as u32 * 1023 + 127) / 255;
                    (c(p[0]) | (c(p[1]) << 10) | (c(p[2]) << 20) | (3 << 30)).to_le_bytes()
                })
                .collect(),
        ),
        _ => None,
    }
}

/// Convert sRGB RGBA8 pixels to texels of an HDR10 swapchain format
/// (BT.2020 PQ, [`crate::hdr::sdr_to_hdr10`]); `None` for other formats.
pub fn convert_hdr10(rgba: &[u8], format: vk::Format) -> Option<Vec<u8>> {
    if !crate::hdr::is_hdr10_format(format) {
        return None;
    }
    let mut out = Vec::with_capacity(rgba.len());
    for p in rgba.chunks_exact(4) {
        let rgb = crate::hdr::sdr_to_hdr10([p[0], p[1], p[2]].map(|v| v as f32 / 255.0), false);
        out.extend_from_slice(&crate::hdr::pack_unorm10(format, rgb, p[3])?.to_le_bytes());
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_is_rasterised_into_a_box() {
        let (w, h, px) = rasterize("F9: capture mouse", 20.0).unwrap();
        assert!(w > 100 && h > 20, "{w}x{h}");
        assert_eq!(px.len(), (w * h * 4) as usize);
        // Corners are background, some pixels are text.
        assert_eq!(&px[0..3], &BACKGROUND);
        assert!(px.chunks_exact(4).any(|p| p[0] > 200));
    }

    #[test]
    fn compile_progress_text() {
        let p = CompileProgress { done: 250, total: 1000, eta: Some(Duration::from_secs(65)), note: String::new() };
        assert_eq!(p.percent(), 25);
        assert_eq!(p.counts(), "250 / 1000 shaders and pipelines (25%)");
        assert_eq!(p.eta_text().as_deref(), Some("about 1m 05s left"));
        assert!((p.fraction() - 0.25).abs() < 1e-6);
        let loading = CompileProgress { done: 0, total: 0, eta: None, note: String::new() };
        assert_eq!((loading.percent(), loading.fraction()), (0, 0.0));
        assert_eq!(loading.counts(), "Loading the shader list");
        let empty = loading.clone();
        assert_eq!(CompileProgress { eta: Some(Duration::from_millis(300)), ..p.clone() }.eta_text().as_deref(), Some("less than a second left"));
        assert_eq!(empty.eta_text(), None);
        assert_eq!(format_duration(Duration::from_secs(42)), "42s");
    }

    #[test]
    fn indicator_waits_before_showing_and_hides_at_once() {
        let mut ind = Indicator::default();
        let t0 = Instant::now();
        assert_eq!(ind.update(0, 100, t0), None);
        assert_eq!(ind.update(5, 100, t0), None, "not shown at once");
        assert_eq!(ind.update(3, 102, t0 + INDICATOR_DELAY / 2), None);
        // Two of five done; then the game asks for four more.
        assert_eq!(ind.update(3, 102, t0 + INDICATOR_DELAY), Some((2, 5)));
        assert_eq!(ind.update(6, 103, t0 + INDICATOR_DELAY), Some((3, 9)));
        assert_eq!(ind.update(0, 109, t0 + INDICATOR_DELAY * 2), None, "hidden when nothing is pending");
        assert_eq!(ind.update(1, 109, t0 + INDICATOR_DELAY * 2), None, "the delay starts again");
        assert_eq!(ind.update(1, 109, t0 + INDICATOR_DELAY * 3), Some((0, 1)), "counted from the new start");
    }

    #[test]
    fn format_conversion() {
        let rgba = [10u8, 20, 30, 255];
        assert_eq!(convert(&rgba, vk::Format::B8G8R8A8_UNORM).unwrap(), vec![30, 20, 10, 255]);
        assert_eq!(convert(&rgba, vk::Format::R8G8B8A8_UNORM).unwrap(), rgba.to_vec());
        let packed = u32::from_le_bytes(convert(&[255, 0, 0, 255], vk::Format::A2B10G10R10_UNORM_PACK32).unwrap().try_into().unwrap());
        assert_eq!(packed & 0x3FF, 1023);
        assert!(convert(&rgba, vk::Format::R16G16B16A16_SFLOAT).is_none());
    }

    #[test]
    fn hdr10_conversion() {
        // White at the SDR reference level (PQ code 616), opaque.
        let a2b = u32::from_le_bytes(convert_hdr10(&[255, 255, 255, 255], vk::Format::A2B10G10R10_UNORM_PACK32).unwrap().try_into().unwrap());
        assert_eq!((a2b & 0x3FF, (a2b >> 10) & 0x3FF, (a2b >> 20) & 0x3FF, a2b >> 30), (616, 616, 616, 3));
        // Red: R and B swap places between the two layouts.
        let red = |f| u32::from_le_bytes(convert_hdr10(&[255, 0, 0, 255], f).unwrap().try_into().unwrap());
        let (b, r) = (red(vk::Format::A2B10G10R10_UNORM_PACK32), red(vk::Format::A2R10G10B10_UNORM_PACK32));
        assert_eq!((b & 0x3FF, (b >> 20) & 0x3FF), (567, 239));
        assert_eq!(((r >> 20) & 0x3FF, r & 0x3FF), (567, 239));
        assert!(convert_hdr10(&[0, 0, 0, 255], vk::Format::B8G8R8A8_UNORM).is_none());
    }
}
