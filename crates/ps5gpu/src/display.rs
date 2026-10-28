// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Window display settings that can change while a game runs: how the
//! guest image is scaled into the window, the present mode (vsync) and the
//! small FPS counter. Each starts from its environment variable and is
//! changed with its setter (the in-game menu); the render thread reads
//! them on every present.
//!
//! - `RPS5E_SCALE_MODE=fit|stretch|integer` (default `fit`: letterboxed,
//!   aspect kept).
//! - `RPS5E_SCALE_FILTER=linear|nearest` (default `linear`).
//! - `RPS5E_PRESENT_MODE=fifo|mailbox|immediate` (default `mailbox`); a
//!   change recreates the swapchain.
//! - `RPS5E_FPS_COUNTER=1`: presented frames per second in the top-left
//!   corner (the full panel is the performance overlay, F10).
//! - `RPS5E_HDR=1`: HDR10 output for games that flip PQ display buffers,
//!   when the display offers an HDR10 surface format (default off; read at
//!   start, see [`crate::hdr`]).
//! - `RPS5E_DISPLAY_RESOLUTION=720p|1080p|2160p`: the resolution of the
//!   emulated display the game is told about (default `2160p`; read once
//!   at start, see [`display_resolution`]).

use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};

/// How the guest image fills the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleMode {
    /// As large as fits, aspect ratio kept (black bars).
    Fit,
    /// The whole window, aspect ratio ignored.
    Stretch,
    /// The largest whole multiple that fits (Fit when even 1x does not).
    Integer,
}

impl ScaleMode {
    pub const ALL: [ScaleMode; 3] = [ScaleMode::Fit, ScaleMode::Stretch, ScaleMode::Integer];

    pub fn as_str(self) -> &'static str {
        match self {
            ScaleMode::Fit => "fit",
            ScaleMode::Stretch => "stretch",
            ScaleMode::Integer => "integer",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.as_str().eq_ignore_ascii_case(s.trim()))
    }
}

/// Filter of the scaling blit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleFilter {
    Linear,
    Nearest,
}

impl ScaleFilter {
    pub const ALL: [ScaleFilter; 2] = [ScaleFilter::Linear, ScaleFilter::Nearest];

    pub fn as_str(self) -> &'static str {
        match self {
            ScaleFilter::Linear => "linear",
            ScaleFilter::Nearest => "nearest",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.as_str().eq_ignore_ascii_case(s.trim()))
    }
}

/// Window present mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentMode {
    /// Vsync, every frame shown.
    Fifo,
    /// Vsync, the newest frame replaces a waiting one (low latency).
    Mailbox,
    /// No vsync (tearing).
    Immediate,
}

impl PresentMode {
    pub const ALL: [PresentMode; 3] = [PresentMode::Fifo, PresentMode::Mailbox, PresentMode::Immediate];

    pub fn as_str(self) -> &'static str {
        match self {
            PresentMode::Fifo => "fifo",
            PresentMode::Mailbox => "mailbox",
            PresentMode::Immediate => "immediate",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.as_str().eq_ignore_ascii_case(s.trim()))
    }
}

/// A setting stored as `index + 1` (0: not read from the environment yet).
struct Choice(AtomicU8);

impl Choice {
    const fn new() -> Self {
        Choice(AtomicU8::new(0))
    }

    fn get(&self, env: &str, parse: impl Fn(&str) -> Option<u8>, default: u8) -> u8 {
        match self.0.load(Ordering::Relaxed) {
            0 => {
                let v = std::env::var(env).ok().and_then(|s| parse(&s)).unwrap_or(default);
                let _ = self.0.compare_exchange(0, v + 1, Ordering::Relaxed, Ordering::Relaxed);
                self.0.load(Ordering::Relaxed) - 1
            }
            v => v - 1,
        }
    }

    fn set(&self, v: u8) {
        self.0.store(v + 1, Ordering::Relaxed);
    }
}

static SCALE_MODE: Choice = Choice::new();
static SCALE_FILTER: Choice = Choice::new();
static PRESENT_MODE: Choice = Choice::new();
static FPS_COUNTER: Choice = Choice::new();
/// Bumped by every present mode change (the swapchain is recreated).
static PRESENT_GENERATION: AtomicU32 = AtomicU32::new(0);

fn index_of<T: PartialEq + Copy>(all: &[T], v: T) -> u8 {
    all.iter().position(|x| *x == v).unwrap_or(0) as u8
}

pub fn scale_mode() -> ScaleMode {
    let i = SCALE_MODE.get("RPS5E_SCALE_MODE", |s| ScaleMode::parse(s).map(|m| index_of(&ScaleMode::ALL, m)), 0);
    ScaleMode::ALL[i as usize % ScaleMode::ALL.len()]
}

pub fn set_scale_mode(m: ScaleMode) {
    SCALE_MODE.set(index_of(&ScaleMode::ALL, m));
    crate::input::wake();
}

pub fn scale_filter() -> ScaleFilter {
    let i = SCALE_FILTER.get("RPS5E_SCALE_FILTER", |s| ScaleFilter::parse(s).map(|m| index_of(&ScaleFilter::ALL, m)), 0);
    ScaleFilter::ALL[i as usize % ScaleFilter::ALL.len()]
}

pub fn set_scale_filter(f: ScaleFilter) {
    SCALE_FILTER.set(index_of(&ScaleFilter::ALL, f));
    crate::input::wake();
}

/// The present mode the swapchain is (re)created with.
pub fn present_mode() -> PresentMode {
    let mailbox = index_of(&PresentMode::ALL, PresentMode::Mailbox);
    let i = PRESENT_MODE.get("RPS5E_PRESENT_MODE", |s| PresentMode::parse(s).map(|m| index_of(&PresentMode::ALL, m)), mailbox);
    PresentMode::ALL[i as usize % PresentMode::ALL.len()]
}

/// Change the present mode; the window recreates its swapchain.
pub fn set_present_mode(m: PresentMode) {
    if present_mode() != m {
        PRESENT_MODE.set(index_of(&PresentMode::ALL, m));
        PRESENT_GENERATION.fetch_add(1, Ordering::Relaxed);
        crate::input::wake();
    }
}

pub(crate) fn present_generation() -> u32 {
    PRESENT_GENERATION.load(Ordering::Relaxed)
}

pub fn fps_counter() -> bool {
    FPS_COUNTER.get("RPS5E_FPS_COUNTER", |s| Some(u8::from(!s.is_empty() && s != "0")), 0) == 1
}

pub fn set_fps_counter(on: bool) {
    FPS_COUNTER.set(u8::from(on));
    crate::input::wake();
}

/// Whether a `RPS5E_HDR` value turns HDR output on.
pub fn parse_hdr(value: &str) -> bool {
    let v = value.trim();
    !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("off") && !v.eq_ignore_ascii_case("false")
}

static HDR: Choice = Choice::new();

/// HDR output enabled (`RPS5E_HDR=1`). Read once: the guest learns the
/// display's dynamic range when it starts.
pub fn hdr_enabled() -> bool {
    HDR.get("RPS5E_HDR", |s| Some(u8::from(parse_hdr(s))), 0) == 1
}

/// Resolution of the emulated display (the TV the console is connected
/// to). Games that ask for it pick the size of their display buffers from
/// it; the window scales whatever they show.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DisplayResolution {
    /// 1280x720.
    P720,
    /// 1920x1080.
    P1080,
    /// 3840x2160.
    #[default]
    P2160,
}

impl DisplayResolution {
    pub const ALL: [DisplayResolution; 3] = [DisplayResolution::P720, DisplayResolution::P1080, DisplayResolution::P2160];

    pub fn as_str(self) -> &'static str {
        match self {
            DisplayResolution::P720 => "720p",
            DisplayResolution::P1080 => "1080p",
            DisplayResolution::P2160 => "2160p",
        }
    }

    /// `720p`, `1080p`, `2160p` (also `4k`), any case.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("4k") {
            return Some(DisplayResolution::P2160);
        }
        Self::ALL.into_iter().find(|m| m.as_str().eq_ignore_ascii_case(s))
    }

    /// Width and height in pixels.
    pub fn size(self) -> (u32, u32) {
        match self {
            DisplayResolution::P720 => (1280, 720),
            DisplayResolution::P1080 => (1920, 1080),
            DisplayResolution::P2160 => (3840, 2160),
        }
    }
}

/// The emulated display's resolution (`RPS5E_DISPLAY_RESOLUTION`, default
/// 2160p). Read once: the guest learns it when it starts. An invalid value
/// is reported and the default used.
pub fn display_resolution() -> DisplayResolution {
    static RES: std::sync::OnceLock<DisplayResolution> = std::sync::OnceLock::new();
    *RES.get_or_init(|| match std::env::var("RPS5E_DISPLAY_RESOLUTION") {
        Ok(v) if !v.trim().is_empty() => DisplayResolution::parse(&v).unwrap_or_else(|| {
            log::warn!("RPS5E_DISPLAY_RESOLUTION={v:?} is not 720p, 1080p or 2160p: using 2160p");
            DisplayResolution::default()
        }),
        _ => DisplayResolution::default(),
    })
}

/// What the window's surface offers for HDR output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HdrSurface {
    /// The window has not created its swapchain yet.
    Unknown,
    /// SDR only: HDR is off, the display has no HDR10 format, or there is
    /// no window.
    Sdr,
    /// An HDR10 swapchain can be used.
    Hdr10,
}

static HDR_SURFACE: AtomicU8 = AtomicU8::new(0);

/// HDR support of the current window surface (updated with every
/// swapchain the window creates).
pub fn hdr_surface() -> HdrSurface {
    match HDR_SURFACE.load(Ordering::Acquire) {
        0 => HdrSurface::Unknown,
        2 => HdrSurface::Hdr10,
        _ => HdrSurface::Sdr,
    }
}

/// Record the surface's HDR support (false also when there is no window).
pub(crate) fn note_hdr_surface(supported: bool) {
    HDR_SURFACE.store(if supported { 2 } else { 1 }, Ordering::Release);
}

/// Where a `w` x `h` image goes in a `sw` x `sh` window: (x, y, width,
/// height), centred.
pub fn dest_rect(w: u32, h: u32, sw: u32, sh: u32, mode: ScaleMode) -> (i32, i32, i32, i32) {
    if w == 0 || h == 0 || sw == 0 || sh == 0 {
        return (0, 0, sw.max(1) as i32, sh.max(1) as i32);
    }
    let fit = (sw as f64 / w as f64).min(sh as f64 / h as f64);
    let (dw, dh) = match mode {
        ScaleMode::Stretch => (sw as i32, sh as i32),
        ScaleMode::Fit => ((w as f64 * fit).round() as i32, (h as f64 * fit).round() as i32),
        ScaleMode::Integer => {
            let n = fit.floor();
            if n >= 1.0 {
                ((w as f64 * n) as i32, (h as f64 * n) as i32)
            } else {
                ((w as f64 * fit).round() as i32, (h as f64 * fit).round() as i32)
            }
        }
    };
    let dw = dw.clamp(1, sw as i32);
    let dh = dh.clamp(1, sh as i32);
    ((sw as i32 - dw) / 2, (sh as i32 - dh) / 2, dw, dh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_letterboxes_like_before() {
        // 16:9 into 16:10: bars at the top and bottom.
        assert_eq!(dest_rect(1920, 1080, 1920, 1200, ScaleMode::Fit), (0, 60, 1920, 1080));
        // 4:3 window.
        assert_eq!(dest_rect(3840, 2160, 1024, 768, ScaleMode::Fit), (0, 96, 1024, 576));
        // The same as the former blit formula.
        let (w, h, sw, sh) = (1280u32, 720u32, 1366u32, 768u32);
        let scale = (sw as f64 / w as f64).min(sh as f64 / h as f64);
        let dw = ((w as f64 * scale).round() as i32).clamp(1, sw as i32);
        let dh = ((h as f64 * scale).round() as i32).clamp(1, sh as i32);
        assert_eq!(dest_rect(w, h, sw, sh, ScaleMode::Fit), ((sw as i32 - dw) / 2, (sh as i32 - dh) / 2, dw, dh));
    }

    #[test]
    fn stretch_and_integer() {
        assert_eq!(dest_rect(1920, 1080, 1920, 1200, ScaleMode::Stretch), (0, 0, 1920, 1200));
        // 2x fits into 2560x1600 (not 2.22x).
        assert_eq!(dest_rect(1152, 648, 2560, 1600, ScaleMode::Integer), (128, 152, 2304, 1296));
        // Below 1x: scaled down like Fit.
        assert_eq!(dest_rect(3840, 2160, 1920, 1080, ScaleMode::Integer), (0, 0, 1920, 1080));
        assert_eq!(dest_rect(0, 0, 800, 600, ScaleMode::Fit), (0, 0, 800, 600));
    }

    #[test]
    fn names_round_trip() {
        for m in ScaleMode::ALL {
            assert_eq!(ScaleMode::parse(m.as_str()), Some(m));
        }
        for f in ScaleFilter::ALL {
            assert_eq!(ScaleFilter::parse(f.as_str()), Some(f));
        }
        for p in PresentMode::ALL {
            assert_eq!(PresentMode::parse(p.as_str()), Some(p));
        }
        assert_eq!(PresentMode::parse("FIFO"), Some(PresentMode::Fifo));
        assert_eq!(ScaleMode::parse("zoom"), None);
    }

    #[test]
    fn display_resolution_names_and_sizes() {
        for r in DisplayResolution::ALL {
            assert_eq!(DisplayResolution::parse(r.as_str()), Some(r));
        }
        assert_eq!(DisplayResolution::parse(" 4K "), Some(DisplayResolution::P2160));
        assert_eq!(DisplayResolution::parse("1080P"), Some(DisplayResolution::P1080));
        assert_eq!(DisplayResolution::parse("1440p"), None);
        assert_eq!(DisplayResolution::default(), DisplayResolution::P2160);
        assert_eq!(DisplayResolution::P720.size(), (1280, 720));
        assert_eq!(DisplayResolution::P1080.size(), (1920, 1080));
        assert_eq!(DisplayResolution::P2160.size(), (3840, 2160));
    }

    #[test]
    fn hdr_switch_values() {
        for on in ["1", "yes", "on", " 1 "] {
            assert!(parse_hdr(on), "{on:?}");
        }
        for off in ["", "0", "off", "OFF", "false", " 0 "] {
            assert!(!parse_hdr(off), "{off:?}");
        }
    }

    #[test]
    fn hdr_surface_state() {
        note_hdr_surface(true);
        assert_eq!(hdr_surface(), HdrSurface::Hdr10);
        note_hdr_surface(false);
        assert_eq!(hdr_surface(), HdrSurface::Sdr);
    }

    #[test]
    fn setters_take_effect() {
        set_scale_filter(ScaleFilter::Nearest);
        assert_eq!(scale_filter(), ScaleFilter::Nearest);
        set_scale_filter(ScaleFilter::Linear);
        assert_eq!(scale_filter(), ScaleFilter::Linear);
        let g = present_generation();
        let other = if present_mode() == PresentMode::Fifo { PresentMode::Immediate } else { PresentMode::Fifo };
        set_present_mode(other);
        assert_eq!(present_mode(), other);
        assert_eq!(present_generation(), g + 1);
        set_present_mode(other);
        assert_eq!(present_generation(), g + 1, "no change, no swapchain recreation");
    }
}
