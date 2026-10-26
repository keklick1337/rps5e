// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! CPU rasteriser of the overlay panel.
//!
//! Text uses a glyph atlas of printable ASCII, rasterised once per size
//! from the bundled monospace font (Hack). The panel is composed into
//! buffers allocated when the size changes; the text layer is rebuilt when
//! the statistics change (2 Hz), the frame-time graph on every frame.
//! Pixels are packed RGBA8 words (`R` in the low byte), not premultiplied.

use super::frames::FrameSummary;
use super::SystemSample;
use ab_glyph::{point, Font, FontRef, PxScale, ScaleFont};
use std::fmt::Write;

const FIRST: u8 = b' ';
const LAST: u8 = b'~';

/// Fixed-cell coverage atlas of ASCII 0x20..=0x7E.
pub struct GlyphAtlas {
    pub px: u32,
    pub cell_w: u32,
    pub cell_h: u32,
    coverage: Vec<u8>,
}

impl GlyphAtlas {
    pub fn new(px: u32) -> Option<GlyphAtlas> {
        let font = FontRef::try_from_slice(epaint_default_fonts::HACK_REGULAR).ok()?;
        let scale = PxScale::from(px as f32);
        let sf = font.as_scaled(scale);
        let cell_w = sf.h_advance(sf.glyph_id('M')).ceil().max(1.0) as u32;
        let cell_h = (sf.ascent() - sf.descent()).ceil().max(1.0) as u32;
        let n = (LAST - FIRST + 1) as usize;
        let mut coverage = vec![0u8; n * (cell_w * cell_h) as usize];
        for c in FIRST..=LAST {
            let cell = &mut coverage[(c - FIRST) as usize * (cell_w * cell_h) as usize..][..(cell_w * cell_h) as usize];
            let g = sf.glyph_id(c as char).with_scale_and_position(scale, point(0.0, sf.ascent()));
            if let Some(o) = font.outline_glyph(g) {
                let b = o.px_bounds();
                o.draw(|x, y, v| {
                    let px = b.min.x as i32 + x as i32;
                    let py = b.min.y as i32 + y as i32;
                    if px >= 0 && py >= 0 && (px as u32) < cell_w && (py as u32) < cell_h {
                        let i = (py as u32 * cell_w + px as u32) as usize;
                        cell[i] = cell[i].max((v.clamp(0.0, 1.0) * 255.0).round() as u8);
                    }
                });
            }
        }
        Some(GlyphAtlas { px, cell_w, cell_h, coverage })
    }

    fn glyph(&self, c: u8) -> &[u8] {
        let c = if (FIRST..=LAST).contains(&c) { c } else { b'?' };
        let n = (self.cell_w * self.cell_h) as usize;
        &self.coverage[(c - FIRST) as usize * n..][..n]
    }
}

/// A line of text in a fixed buffer (formatting allocates nothing).
#[derive(Clone, Copy)]
pub struct TextLine {
    buf: [u8; COLUMNS],
    len: usize,
    /// Leading characters drawn in the label colour.
    label: usize,
}

/// Characters per line.
pub const COLUMNS: usize = 36;
/// Text lines at most.
pub const MAX_LINES: usize = 14;

impl Default for TextLine {
    fn default() -> Self {
        TextLine { buf: [b' '; COLUMNS], len: 0, label: 0 }
    }
}

impl Write for TextLine {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        for &b in s.as_bytes() {
            if self.len < COLUMNS {
                self.buf[self.len] = if b.is_ascii() { b } else { b'?' };
                self.len += 1;
            }
        }
        Ok(())
    }
}

impl TextLine {
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
    }
}

const BG: u32 = rgba(12, 12, 16, 168);
const GRAPH_BG: u32 = rgba(0, 0, 0, 120);
const FG: [u8; 3] = [236, 236, 236];
const LABEL: [u8; 3] = [120, 200, 255];
const GOOD: u32 = rgba(110, 220, 110, 230);
const WARN: u32 = rgba(240, 200, 80, 230);
const BAD: u32 = rgba(240, 90, 80, 230);
const GUIDE: u32 = rgba(255, 255, 255, 60);

const fn rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
    r as u32 | (g as u32) << 8 | (b as u32) << 16 | (a as u32) << 24
}

/// Seconds of history the graph shows.
pub const GRAPH_SECONDS: f64 = 4.0;

/// The panel: layout, text and pixels.
pub struct Panel {
    atlas: GlyphAtlas,
    pad: u32,
    pub width: u32,
    pub height: u32,
    graph_y: u32,
    graph_h: u32,
    lines: [TextLine; MAX_LINES],
    line_count: usize,
    /// Background and text, rebuilt on new statistics.
    base: Vec<u32>,
    /// `base` with the graph, rebuilt every frame.
    pub pixels: Vec<u32>,
    /// Per-column frame-time maxima for the graph.
    pub columns: Vec<f32>,
}

fn fmt_bytes(l: &mut TextLine, b: u64) {
    let g = b as f64 / (1u64 << 30) as f64;
    if g >= 1.0 {
        let _ = write!(l, "{g:.1}G");
    } else {
        let _ = write!(l, "{:.0}M", b as f64 / (1u64 << 20) as f64);
    }
}

fn fmt_rate(l: &mut TextLine, v: f32) {
    if v >= 1e6 {
        let _ = write!(l, "{:.1}M", v / 1e6);
    } else if v >= 1e4 {
        let _ = write!(l, "{:.0}k", v / 1e3);
    } else {
        let _ = write!(l, "{v:.0}");
    }
}

impl Panel {
    /// A panel with text `px` pixels high.
    pub fn new(px: u32) -> Option<Panel> {
        let atlas = GlyphAtlas::new(px)?;
        let pad = (px / 2).max(4);
        let width = atlas.cell_w * COLUMNS as u32 + 2 * pad;
        let graph_h = atlas.cell_h * 3;
        let max_h = atlas.cell_h * MAX_LINES as u32 + graph_h + 3 * pad;
        let max = (width * max_h) as usize;
        let columns = vec![0.0; (width - 2 * pad) as usize];
        Some(Panel {
            atlas,
            pad,
            width,
            height: 0,
            graph_y: 0,
            graph_h,
            lines: [TextLine::default(); MAX_LINES],
            line_count: 0,
            base: vec![0; max],
            pixels: vec![0; max],
            columns,
        })
    }

    pub fn text_px(&self) -> u32 {
        self.atlas.px
    }

    fn line(&mut self, label: &str) -> Option<&mut TextLine> {
        if self.line_count == MAX_LINES {
            return None;
        }
        let l = &mut self.lines[self.line_count];
        *l = TextLine::default();
        let _ = write!(l, "{label:<6} ");
        l.label = label.len();
        self.line_count += 1;
        Some(l)
    }

    /// Lay out the text for `f` (frame times) and `s` (system sample) and
    /// rebuild the text layer.
    pub fn set_stats(&mut self, f: &FrameSummary, s: &SystemSample) {
        self.line_count = 0;
        if let Some(l) = self.line("FPS") {
            let _ = write!(l, "{:5.1}  {:5.1}ms  1%low {:.0}", f.fps, f.avg_ms, f.low1_fps);
        }
        // The guest's own rate when frames are skipped or repeated.
        let flips = s.flips_per_sec;
        if s.seq > 0 && (flips - f.fps).abs() > 1.0f32.max(f.fps * 0.05) {
            if let Some(l) = self.line("Guest") {
                let _ = write!(l, "{flips:5.1} flips/s");
            }
        }
        if s.cpu_system.is_some() || s.cpu_process.is_some() {
            if let Some(l) = self.line("CPU") {
                if let Some(c) = s.cpu_system {
                    let _ = write!(l, "{c:3.0}%  ");
                }
                if let Some(c) = s.cpu_process {
                    let _ = write!(l, "emu {c:.0}%");
                    if s.cpus > 0 {
                        let _ = write!(l, " of {}", s.cpus * 100);
                    }
                }
            }
        }
        for t in s.threads.iter().filter(|t| t.percent > 0.0) {
            if let Some(l) = self.line("") {
                let _ = write!(l, "{:<16}{:5.1}%", t.name(), t.percent);
            }
        }
        if s.rss.is_some() || s.mem.is_some() {
            if let Some(l) = self.line("RAM") {
                if let Some(r) = s.rss {
                    l.write_str("emu ").ok();
                    fmt_bytes(l, r);
                    l.write_str("  ").ok();
                }
                if let Some((total, used)) = s.mem {
                    l.write_str("sys ").ok();
                    fmt_bytes(l, used);
                    l.write_str("/").ok();
                    fmt_bytes(l, total);
                }
            }
        }
        let g = s.gpu;
        if g.busy.is_some() || g.temp_c.is_some() || g.clock_mhz.is_some() {
            if let Some(l) = self.line("GPU") {
                if let Some(b) = g.busy {
                    let _ = write!(l, "{b:3}%  ");
                }
                if let Some(t) = g.temp_c {
                    let _ = write!(l, "{t}C  ");
                }
                if let Some(c) = g.clock_mhz {
                    let _ = write!(l, "{c}MHz");
                }
            }
        }
        if g.vram_total.is_some() {
            if let Some(l) = self.line("VRAM") {
                if let Some(u) = g.vram_used {
                    fmt_bytes(l, u);
                    l.write_str("/").ok();
                }
                fmt_bytes(l, g.vram_total.unwrap_or(0));
                if let Some(p) = g.vram_process {
                    l.write_str("  emu ").ok();
                    fmt_bytes(l, p);
                }
            }
        }
        if let Some(l) = self.line("Draws") {
            let _ = write!(l, "{:.0}/f  disp {:.0}/f", s.draws_per_flip, s.dispatches_per_flip);
        }
        if let Some(l) = self.line("Shader") {
            let _ = write!(l, "{:.1}/s", s.compiles_per_sec);
            if s.compiling > 0 {
                let _ = write!(l, " ({} busy)", s.compiling);
            }
            match s.pipeline_hit_percent {
                Some(h) => {
                    let _ = write!(l, "  pso {h:.1}%");
                }
                None => {
                    let _ = write!(l, "  pso -");
                }
            }
        }
        if let Some(hle) = s.hle_per_sec {
            if let Some(l) = self.line("HLE") {
                fmt_rate(l, hle);
                l.write_str(" calls/s").ok();
            }
        }
        self.rebuild_base();
    }

    fn rebuild_base(&mut self) {
        let (cw, ch, pad) = (self.atlas.cell_w, self.atlas.cell_h, self.pad);
        // FPS line, graph, the other lines.
        self.graph_y = pad + ch + pad / 2;
        let rest = self.line_count.saturating_sub(1) as u32;
        self.height = self.graph_y + self.graph_h + pad / 2 + rest * ch + pad;
        let w = self.width;
        let px = &mut self.base[..(w * self.height) as usize];
        px.fill(BG);
        for y in self.graph_y..self.graph_y + self.graph_h {
            px[(y * w + pad) as usize..(y * w + w - pad) as usize].fill(GRAPH_BG);
        }
        for (i, line) in self.lines[..self.line_count].iter().enumerate() {
            let y = if i == 0 { pad } else { self.graph_y + self.graph_h + pad / 2 + (i as u32 - 1) * ch };
            for (k, &c) in line.buf[..line.len].iter().enumerate() {
                if c == b' ' {
                    continue;
                }
                let color = if k < line.label { LABEL } else { FG };
                let x = pad + k as u32 * cw;
                let glyph = self.atlas.glyph(c);
                for gy in 0..ch {
                    let row = ((y + gy) * w + x) as usize;
                    for gx in 0..cw {
                        let a = glyph[(gy * cw + gx) as usize];
                        if a != 0 {
                            let p = &mut px[row + gx as usize];
                            *p = blend(*p, color, a);
                        }
                    }
                }
            }
        }
    }

    /// Compose the frame: the text layer plus the graph of `columns`
    /// (fill them first, e.g. with [`super::frames::FrameTimes::graph`]).
    pub fn compose(&mut self) -> &[u32] {
        let n = (self.width * self.height) as usize;
        self.pixels[..n].copy_from_slice(&self.base[..n]);
        let (w, pad, gy, gh) = (self.width, self.pad, self.graph_y, self.graph_h);
        // Scale: at least 33.3 ms, else the largest frame with headroom.
        let peak = self.columns.iter().copied().fold(0.0f32, f32::max);
        let scale = (peak * 1.15).max(1000.0 / 30.0);
        for target in [1000.0 / 60.0, 1000.0 / 30.0] {
            let y = gy + gh - 1 - ((target / scale * (gh - 1) as f32) as u32).min(gh - 1);
            for x in pad..w - pad {
                let p = &mut self.pixels[(y * w + x) as usize];
                *p = over(*p, GUIDE);
            }
        }
        for (i, &ms) in self.columns.iter().enumerate() {
            if ms <= 0.0 {
                continue;
            }
            let color = if ms <= 1000.0 / 55.0 {
                GOOD
            } else if ms <= 1000.0 / 28.0 {
                WARN
            } else {
                BAD
            };
            let bar = ((ms / scale * gh as f32).ceil() as u32).clamp(1, gh);
            let x = pad + i as u32;
            for y in gy + gh - bar..gy + gh {
                self.pixels[(y * w + x) as usize] = color;
            }
        }
        &self.pixels[..n]
    }

    /// Text of the current lines (tests and logs).
    pub fn lines(&self) -> impl Iterator<Item = &str> {
        self.lines[..self.line_count].iter().map(TextLine::as_str)
    }
}

/// `color` over `p` with coverage `a` (alpha grows towards opaque).
fn blend(p: u32, color: [u8; 3], a: u8) -> u32 {
    let a = a as u32;
    let mix = |dst: u32, src: u8| (dst * (255 - a) + src as u32 * a + 127) / 255;
    let r = mix(p & 0xFF, color[0]);
    let g = mix((p >> 8) & 0xFF, color[1]);
    let b = mix((p >> 16) & 0xFF, color[2]);
    let pa = p >> 24;
    let na = pa + ((255 - pa) * a + 127) / 255;
    r | g << 8 | b << 16 | na << 24
}

/// Non-premultiplied `src` over `dst`.
fn over(dst: u32, src: u32) -> u32 {
    let a = (src >> 24) as u8;
    blend(dst, [(src & 0xFF) as u8, (src >> 8 & 0xFF) as u8, (src >> 16 & 0xFF) as u8], a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_has_ink_for_letters_only() {
        let a = GlyphAtlas::new(16).unwrap();
        assert!(a.cell_w >= 7 && a.cell_h >= 14, "{}x{}", a.cell_w, a.cell_h);
        assert!(a.glyph(b' ').iter().all(|&c| c == 0));
        assert!(a.glyph(b'M').iter().any(|&c| c > 200));
        // Out of range characters fall back to '?'.
        assert_eq!(a.glyph(0x01), a.glyph(b'?'));
    }

    #[test]
    fn text_line_truncates() {
        let mut l = TextLine::default();
        for _ in 0..10 {
            l.write_str("abcdef").unwrap();
        }
        assert_eq!(l.as_str().len(), COLUMNS);
    }

    fn sample() -> SystemSample {
        let mut s = SystemSample {
            seq: 3,
            cpu_system: Some(23.0),
            cpu_process: Some(145.0),
            cpus: 16,
            rss: Some(3 << 30),
            mem: Some((32 << 30, 12 << 30)),
            flips_per_sec: 30.0,
            draws_per_flip: 930.0,
            dispatches_per_flip: 12.0,
            compiles_per_sec: 2.0,
            compiling: 1,
            pipeline_hit_percent: Some(99.5),
            hle_per_sec: Some(152_000.0),
            ..Default::default()
        };
        s.threads[0] = super::super::ThreadLoad::new("rps5e-render", 45.0);
        s.gpu.busy = Some(40);
        s.gpu.vram_total = Some(8 << 30);
        s.gpu.vram_used = Some(3 << 30);
        s
    }

    #[test]
    fn layout_and_compose() {
        let mut p = Panel::new(14).unwrap();
        let f = FrameSummary { frames: 60, fps: 60.0, avg_ms: 16.7, low1_fps: 50.0, max_ms: 25.0 };
        p.set_stats(&f, &sample());
        let lines: Vec<&str> = p.lines().collect();
        assert!(lines[0].starts_with("FPS"), "{lines:?}");
        assert!(lines[0].contains("60.0") && lines[0].contains("1%low 50"));
        // 30 flips/s against 60 presented: the guest rate is shown.
        assert!(lines.iter().any(|l| l.contains("30.0 flips/s")));
        assert!(lines.iter().any(|l| l.contains("rps5e-render") && l.contains("45.0%")));
        assert!(lines.iter().any(|l| l.starts_with("RAM") && l.contains("emu 3.0G") && l.contains("12.0G/32.0G")));
        assert!(lines.iter().any(|l| l.starts_with("VRAM") && l.contains("3.0G/8.0G")));
        assert!(lines.iter().any(|l| l.contains("930/f") && l.contains("disp 12/f")));
        assert!(lines.iter().any(|l| l.contains("1 busy") && l.contains("pso 99.5%")));
        assert!(lines.iter().any(|l| l.contains("152k calls/s")));
        p.columns.fill(0.0);
        p.columns[0] = 16.0;
        p.columns[1] = 100.0;
        let (w, h) = (p.width, p.height);
        let px = p.compose().to_vec();
        assert_eq!(px.len(), (w * h) as usize);
        // Corner: translucent background.
        assert_eq!(px[0], BG);
        // Some text pixels are bright, the bars are drawn.
        assert!(px.iter().any(|&c| c & 0xFF > 200 && c >> 24 > 200));
        assert!(px.contains(&BAD) && px.contains(&GOOD));
        // Matching rates hide the guest line.
        let mut s = sample();
        s.flips_per_sec = 60.2;
        p.set_stats(&f, &s);
        assert!(!p.lines().any(|l| l.contains("flips/s")));
        assert!(p.height < h);
    }

    #[test]
    fn blending() {
        assert_eq!(blend(rgba(0, 0, 0, 100), [255, 255, 255], 255), rgba(255, 255, 255, 255));
        assert_eq!(blend(rgba(10, 20, 30, 100), [255, 255, 255], 0), rgba(10, 20, 30, 100));
        let half = blend(rgba(0, 0, 0, 0), [200, 100, 0], 128);
        assert_eq!(half & 0xFF, 100);
        assert_eq!(half >> 24, 128);
    }
}
