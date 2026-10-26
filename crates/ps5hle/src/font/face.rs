// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Font faces: OpenType/TrueType parsing, metrics, outlines, rasterisation
//! and the system font set lookup.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use ab_glyph_rasterizer::{point, Point, Rasterizer};
use ttf_parser::gpos::{PairAdjustment, PositioningSubtable};
use ttf_parser::{Face, GlyphId, OutlineBuilder};

/// Round to the 26.6 fixed-point grid the font engine reports in.
pub fn q64(v: f32) -> f32 {
    (v * 64.0).round() / 64.0
}

/// An opened font file (or one face of a collection).
pub struct FaceData {
    bytes: Vec<u8>,
    index: u32,
    pub units_per_em: u16,
    /// head.{xMin, yMin, xMax, yMax}
    pub bbox: [i16; 4],
}

/// Unscaled glyph metrics in font units.
#[derive(Clone, Copy, Debug, Default)]
pub struct GlyphUnits {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
    pub advance: f32,
}

/// Outline segment in font units (y up).
#[derive(Clone, Copy, Debug)]
pub enum Seg {
    Line(Point, Point),
    Quad(Point, Point, Point),
    Cubic(Point, Point, Point, Point),
}

struct Collector {
    segs: Vec<Seg>,
    start: Point,
    cur: Point,
}

impl OutlineBuilder for Collector {
    fn move_to(&mut self, x: f32, y: f32) {
        self.start = point(x, y);
        self.cur = self.start;
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p = point(x, y);
        self.segs.push(Seg::Line(self.cur, p));
        self.cur = p;
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let p = point(x, y);
        self.segs.push(Seg::Quad(self.cur, point(x1, y1), p));
        self.cur = p;
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let p = point(x, y);
        self.segs.push(Seg::Cubic(self.cur, point(x1, y1), point(x2, y2), p));
        self.cur = p;
    }
    fn close(&mut self) {
        if self.cur != self.start {
            self.segs.push(Seg::Line(self.cur, self.start));
        }
        self.cur = self.start;
    }
}

impl FaceData {
    pub fn parse(bytes: Vec<u8>, index: u32) -> Option<Arc<FaceData>> {
        let (units_per_em, bbox) = {
            let face = Face::parse(&bytes, index).ok()?;
            let b = face.global_bounding_box();
            (face.units_per_em(), [b.x_min, b.y_min, b.x_max, b.y_max])
        };
        if units_per_em == 0 {
            return None;
        }
        Some(Arc::new(FaceData { bytes, index, units_per_em, bbox }))
    }

    fn face(&self) -> Face<'_> {
        // Validated in parse(); the bytes never change afterwards.
        Face::parse(&self.bytes, self.index).expect("validated face")
    }

    /// Glyph for a Unicode code point (every Unicode cmap subtable is tried).
    pub fn glyph_index(&self, code: u32) -> Option<u16> {
        let c = char::from_u32(code)?;
        self.face().glyph_index(c).map(|g| g.0).filter(|&g| g != 0)
    }

    pub fn glyph_units(&self, gid: u16) -> GlyphUnits {
        let face = self.face();
        let g = GlyphId(gid);
        let advance = face.glyph_hor_advance(g).unwrap_or(0) as f32;
        match face.glyph_bounding_box(g) {
            Some(r) => GlyphUnits {
                x_min: r.x_min as f32,
                y_min: r.y_min as f32,
                x_max: r.x_max as f32,
                y_max: r.y_max as f32,
                advance,
            },
            None => GlyphUnits { advance, ..Default::default() },
        }
    }

    /// Vertical metrics of a face with vertical tables (vhea + vmtx): the
    /// glyph's (top side bearing, advance height) in font units.
    pub fn vertical_units(&self, gid: u16) -> Option<(f32, f32)> {
        let face = self.face();
        let tables = face.tables();
        if tables.vhea.is_none() || tables.vmtx.is_none() {
            return None;
        }
        let g = GlyphId(gid);
        Some((face.glyph_ver_side_bearing(g).unwrap_or(0) as f32, face.glyph_ver_advance(g).unwrap_or(0) as f32))
    }

    /// Advance height made up for a face without vertical tables: the
    /// typographic ascender minus descender of the OS/2 table, or of the
    /// horizontal header when there is no OS/2 table.
    pub fn synthesized_vertical_advance(&self) -> f32 {
        let face = self.face();
        match (face.typographic_ascender(), face.typographic_descender()) {
            (Some(a), Some(d)) => (a as i32 - d as i32) as f32,
            _ => {
                let hhea = face.tables().hhea;
                (hhea.ascender as i32 - hhea.descender as i32) as f32
            }
        }
    }

    pub fn glyph_count(&self) -> u16 {
        self.face().number_of_glyphs()
    }

    /// Horizontal kerning between two glyphs in font units: GPOS pair
    /// adjustments when the font has a GPOS table, else the `kern` table.
    pub fn kerning(&self, left: u16, right: u16) -> i32 {
        let face = self.face();
        let (l, r) = (GlyphId(left), GlyphId(right));
        if let Some(gpos) = face.tables().gpos {
            for lookup in gpos.lookups {
                for i in 0..lookup.subtables.len() {
                    let Some(PositioningSubtable::Pair(pair)) = lookup.subtables.get::<PositioningSubtable>(i) else {
                        continue;
                    };
                    let value = match pair {
                        PairAdjustment::Format1 { coverage, sets } => {
                            coverage.get(l).and_then(|ci| sets.get(ci)).and_then(|set| set.get(r))
                        }
                        PairAdjustment::Format2 { coverage, classes, matrix } => {
                            if coverage.get(l).is_none() {
                                continue;
                            }
                            matrix.get((classes.0.get(l), classes.1.get(r)))
                        }
                    };
                    if let Some((first, _)) = value {
                        return first.x_advance as i32;
                    }
                }
            }
            return 0;
        }
        face.tables()
            .kern
            .and_then(|kern| {
                kern.subtables
                    .into_iter()
                    .filter(|t| t.horizontal && !t.variable && !t.has_cross_stream)
                    .find_map(|t| t.glyphs_kerning(l, r))
            })
            .map_or(0, |v| v as i32)
    }

    pub fn outline(&self, gid: u16) -> Vec<Seg> {
        let mut c = Collector { segs: Vec::new(), start: point(0.0, 0.0), cur: point(0.0, 0.0) };
        self.face().outline_glyph(GlyphId(gid), &mut c);
        c.segs
    }

    /// Outline contours in font units (y up): each point with its on-curve
    /// flag (control points of quadratic and cubic curves are off-curve).
    pub fn contours(&self, gid: u16) -> Vec<Vec<OutlinePoint>> {
        let mut c = ContourCollector(Vec::new());
        self.face().outline_glyph(GlyphId(gid), &mut c);
        c.0.retain(|contour| !contour.is_empty());
        c.0
    }
}

/// A contour point: (x, y, on curve).
pub type OutlinePoint = (f32, f32, bool);

struct ContourCollector(Vec<Vec<OutlinePoint>>);

impl ContourCollector {
    fn push(&mut self, x: f32, y: f32, on: bool) {
        if let Some(c) = self.0.last_mut() {
            c.push((x, y, on));
        }
    }
}

impl OutlineBuilder for ContourCollector {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.push(vec![(x, y, true)]);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.push(x, y, true);
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.push(x1, y1, false);
        self.push(x, y, true);
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.push(x1, y1, false);
        self.push(x2, y2, false);
        self.push(x, y, true);
    }
    fn close(&mut self) {
        // A closing point that repeats the start is implied by the contour.
        if let Some(c) = self.0.last_mut() {
            if c.len() > 1 && c.first() == c.last() {
                c.pop();
            }
        }
    }
}

/// Largest glyph bitmap edge in pixels: glyphs are rasterised at the
/// requested size; the limit only rejects absurd scales.
pub const MAX_GLYPH_DIM: usize = 4096;

/// A rendered coverage bitmap positioned relative to the pen origin.
pub struct Bitmap {
    /// Device pixel column of the bitmap's left edge (relative to the
    /// integer origin).
    pub left: i32,
    /// Device pixel row of the bitmap's top edge (y down, relative to the
    /// integer origin).
    pub top: i32,
    pub width: usize,
    pub height: usize,
    pub coverage: Vec<u8>,
}

/// Rasterise `segs` (font units, y up) at scale (sx, sy) with a horizontal
/// shear `slant` (x += slant * y) and a sub-pixel pen offset (fx, fy) in
/// device space (y down).
pub fn rasterize(segs: &[Seg], sx: f32, sy: f32, slant: f32, fx: f32, fy: f32) -> Bitmap {
    let map = |p: &Point| point(fx + (p.x + slant * p.y) * sx, fy - p.y * sy);
    let mut min = point(f32::MAX, f32::MAX);
    let mut max = point(f32::MIN, f32::MIN);
    let mut extend = |p: Point| {
        min.x = min.x.min(p.x);
        min.y = min.y.min(p.y);
        max.x = max.x.max(p.x);
        max.y = max.y.max(p.y);
    };
    for s in segs {
        match s {
            Seg::Line(a, b) => [a, b].into_iter().for_each(|p| extend(map(p))),
            Seg::Quad(a, b, c) => [a, b, c].into_iter().for_each(|p| extend(map(p))),
            Seg::Cubic(a, b, c, d) => [a, b, c, d].into_iter().for_each(|p| extend(map(p))),
        }
    }
    if segs.is_empty() || !(min.x.is_finite() && max.x.is_finite()) {
        return Bitmap { left: 0, top: 0, width: 0, height: 0, coverage: Vec::new() };
    }
    let left = min.x.floor() as i32;
    let top = min.y.floor() as i32;
    let width = (max.x.ceil() as i32 - left).max(0) as usize;
    // Rows past the limit are cropped (the rasteriser clips vertically).
    let height = ((max.y.ceil() as i32 - top).max(0) as usize).min(MAX_GLYPH_DIM);
    if width == 0 || height == 0 || width > MAX_GLYPH_DIM {
        return Bitmap { left, top, width: 0, height: 0, coverage: Vec::new() };
    }
    let off = point(-(left as f32), -(top as f32));
    let dev = |p: &Point| {
        let m = map(p);
        point(m.x + off.x, m.y + off.y)
    };
    let mut r = Rasterizer::new(width, height);
    for s in segs {
        match s {
            Seg::Line(a, b) => r.draw_line(dev(a), dev(b)),
            Seg::Quad(a, b, c) => r.draw_quad(dev(a), dev(b), dev(c)),
            Seg::Cubic(a, b, c, d) => r.draw_cubic(dev(a), dev(b), dev(c), dev(d)),
        }
    }
    let mut coverage = vec![0u8; width * height];
    r.for_each_pixel(|i, a| coverage[i] = (a.clamp(0.0, 1.0) * 255.0).round() as u8);
    Bitmap { left, top, width, height, coverage }
}

// ---------------------------------------------------------------------------
// System font sets
// ---------------------------------------------------------------------------

const SYSTEM_FONT_SET_TYPES: [u32; 120] = [
    0x18070043, 0x18070044, 0x18070045, 0x18070047, 0x18070053, 0x18070054, 0x18070055, 0x18070057, 0x180700C3,
    0x180700C4, 0x180700C5, 0x180700C7, 0x18070444, 0x18070447, 0x18070454, 0x18070457, 0x180704C4, 0x180704C7,
    0x18071053, 0x18071054, 0x18071055, 0x18071057, 0x18071454, 0x18071457, 0x18072444, 0x18072447, 0x180724C4,
    0x180724C7, 0x18073454, 0x18073457, 0x180734D4, 0x180734D7, 0x18078044, 0x180780C4, 0x18079054, 0x1807A044,
    0x1807A0C4, 0x1807A444, 0x1807A4C4, 0x1807AC44, 0x1807ACC4, 0x1807B054, 0x1807B0D4, 0x1807B454, 0x1807B4D4,
    0x1807BC54, 0x1807BCD4, 0x18080444, 0x18080447, 0x18080454, 0x18080457, 0x180804C4, 0x180804C7, 0x18081454,
    0x18081457, 0x18082444, 0x18082447, 0x180824C4, 0x180824C7, 0x18083454, 0x18083457, 0x180834D4, 0x180834D7,
    0x1808A444, 0x1808A4C4, 0x1808B454, 0x1808B4D4, 0x180C8044, 0x180C80C4, 0x180C9054, 0x180CA044, 0x180CA0C4,
    0x180CAC44, 0x180CACC4, 0x180CB054, 0x180CB0D4, 0x180CBC54, 0x180CBCD4, 0x18170043, 0x18170044, 0x18170045,
    0x18170047, 0x18170444, 0x18170447, 0x18370044, 0x18370047, 0x18370444, 0x18370447, 0x1A070543, 0x1A070547,
    0x1A070553, 0x1A070557, 0x1A0705C3, 0x1A0705C7, 0x1A071553, 0x1A071557, 0x1A072543, 0x1A072547, 0x1A0725C3,
    0x1A0725C7, 0x1A073553, 0x1A073557, 0x1A0735D3, 0x1A0735D7, 0x1A080543, 0x1A080547, 0x1A080553, 0x1A080557,
    0x1A0805C3, 0x1A0805C7, 0x1A081553, 0x1A081557, 0x1A082543, 0x1A082547, 0x1A0825C3, 0x1A0825C7, 0x1A083553,
    0x1A083557, 0x1A0835D3, 0x1A0835D7,
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Family {
    Latin,
    LatinItalic,
    Typewriter,
    Vietnamese,
    Thai,
    Japanese,
    ChineseGb,
}

/// (console font files by weight, substitute files by weight, substitute
/// collection index, generic host fallbacks by weight)
struct FamilyFiles {
    system: [&'static str; 4],
    substitute: [&'static str; 4],
    substitute_index: u32,
    host: [&'static [&'static str]; 4],
}

const LATIN_HOST: [&[&str]; 4] = [
    &["DejaVuSans-ExtraLight.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "DroidSans.ttf"],
    &["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "DroidSans.ttf", "OpenSans-Regular.ttf"],
    &["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "DroidSans.ttf", "OpenSans-Regular.ttf"],
    &["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "DroidSans-Bold.ttf", "OpenSans-Bold.ttf"],
];
const ITALIC_HOST: [&[&str]; 4] = [
    &["DejaVuSans-Oblique.ttf", "LiberationSans-Italic.ttf"],
    &["DejaVuSans-Oblique.ttf", "LiberationSans-Italic.ttf"],
    &["DejaVuSans-Oblique.ttf", "LiberationSans-Italic.ttf"],
    &["DejaVuSans-BoldOblique.ttf", "LiberationSans-BoldItalic.ttf"],
];
const MONO_HOST: [&[&str]; 4] = [
    &["DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "DroidSansMono.ttf"],
    &["DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "DroidSansMono.ttf"],
    &["DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "DroidSansMono.ttf"],
    &["DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf", "DroidSansMono.ttf"],
];
const THAI_HOST: [&[&str]; 4] = [
    &["NotoSansThai-Regular.ttf", "DroidSansThai.ttf"],
    &["NotoSansThai-Regular.ttf", "DroidSansThai.ttf"],
    &["NotoSansThai-Regular.ttf", "DroidSansThai.ttf"],
    &["NotoSansThai-Bold.ttf", "DroidSansThai.ttf"],
];
const CJK_HOST: [&[&str]; 4] = [
    &["NotoSansCJK-Regular.ttc", "NotoSansJP-Regular.otf", "SourceHanSans-Regular.ttc", "DroidSansJapanese.ttf", "DroidSansFallback.ttf"],
    &["NotoSansCJK-Regular.ttc", "NotoSansJP-Regular.otf", "SourceHanSans-Regular.ttc", "DroidSansJapanese.ttf", "DroidSansFallback.ttf"],
    &["NotoSansCJK-Medium.ttc", "NotoSansJP-Medium.otf", "SourceHanSans-Medium.ttc", "DroidSansJapanese.ttf", "DroidSansFallback.ttf"],
    &["NotoSansCJK-Bold.ttc", "NotoSansJP-Bold.otf", "SourceHanSans-Bold.ttc", "DroidSansJapanese.ttf", "DroidSansFallback.ttf"],
];

const FAMILY_FILES: [FamilyFiles; 7] = [
    FamilyFiles {
        system: ["SST-Light.otf", "SST-Roman.otf", "SST-Medium.otf", "SST-Bold.otf"],
        substitute: ["NotoSans-Light.ttf", "NotoSans-Regular.ttf", "NotoSans-Medium.ttf", "NotoSans-Bold.ttf"],
        substitute_index: 0,
        host: LATIN_HOST,
    },
    FamilyFiles {
        system: ["SST-LightItalic.otf", "SST-Italic.otf", "SST-MediumItalic.otf", "SST-BoldItalic.otf"],
        substitute: ["NotoSans-LightItalic.ttf", "NotoSans-Italic.ttf", "NotoSans-MediumItalic.ttf", "NotoSans-BoldItalic.ttf"],
        substitute_index: 0,
        host: ITALIC_HOST,
    },
    FamilyFiles {
        system: ["SSTTypewriter-Roman.otf", "SSTTypewriter-Roman.otf", "SSTTypewriter-Roman.otf", "SSTTypewriter-Bd.otf"],
        substitute: ["NotoSansMono-Light.ttf", "NotoSansMono-Regular.ttf", "NotoSansMono-Medium.ttf", "NotoSansMono-Bold.ttf"],
        substitute_index: 0,
        host: MONO_HOST,
    },
    FamilyFiles {
        system: ["SSTVietnamese-Light.otf", "SSTVietnamese-Roman.otf", "SSTVietnamese-Medium.otf", "SSTVietnamese-Bold.otf"],
        substitute: ["NotoSans-Light.ttf", "NotoSans-Regular.ttf", "NotoSans-Medium.ttf", "NotoSans-Bold.ttf"],
        substitute_index: 0,
        host: LATIN_HOST,
    },
    FamilyFiles {
        system: ["SSTThai-Light.otf", "SSTThai-Roman.otf", "SSTThai-Medium.otf", "SSTThai-Bold.otf"],
        substitute: ["NotoSansThai-Light.ttf", "NotoSansThai-Regular.ttf", "NotoSansThai-Medium.ttf", "NotoSansThai-Bold.ttf"],
        substitute_index: 0,
        host: THAI_HOST,
    },
    FamilyFiles {
        system: ["SSTJpPro-Regular.otf", "SSTJpPro-Regular.otf", "SSTJpPro-Regular.otf", "SSTJpPro-Bold.otf"],
        substitute: ["NotoSansCJK-Light.ttc", "NotoSansCJK-Regular.ttc", "NotoSansCJK-Medium.ttc", "NotoSansCJK-Bold.ttc"],
        substitute_index: 0,
        host: CJK_HOST,
    },
    FamilyFiles {
        system: ["DFHEI5-SONY.ttf", "DFHEI5-SONY.ttf", "DFHEI5-SONY.ttf", "DFHEI5-SONY.ttf"],
        substitute: ["NotoSansCJK-Light.ttc", "NotoSansCJK-Regular.ttc", "NotoSansCJK-Medium.ttc", "NotoSansCJK-Bold.ttc"],
        substitute_index: 2,
        host: CJK_HOST,
    },
];

fn family_of(font_set_type: u32) -> Family {
    let region = (font_set_type >> 16) & 0xFF;
    let scripts = (font_set_type >> 8) & 0xFE;
    let vietnamese = font_set_type & 0x10 != 0;
    match region {
        0x17 => Family::LatinItalic,
        0x37 => Family::Typewriter,
        0x08 => {
            if scripts == 0x04 && vietnamese {
                Family::Vietnamese
            } else {
                Family::Japanese
            }
        }
        0x0C => Family::ChineseGb,
        _ if scripts == 0x10 => Family::Thai,
        _ if scripts == 0 && vietnamese => Family::Vietnamese,
        _ => Family::Latin,
    }
}

fn weight_of(font_set_type: u32) -> usize {
    match font_set_type & 0x0F {
        0x3 => 0,
        0x5 => 2,
        0x7 => 3,
        _ => 1,
    }
}

pub fn is_system_font_set(font_set_type: u32) -> bool {
    SYSTEM_FONT_SET_TYPES.contains(&font_set_type)
}

/// Directories searched for font files: the configured console font
/// directory first, then the host's font directories.
pub(crate) fn font_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = std::env::var_os("RPS5E_SYSTEM_FONTS") {
        dirs.push(PathBuf::from(d));
    }
    for d in ["/usr/share/fonts", "/usr/local/share/fonts"] {
        dirs.push(PathBuf::from(d));
    }
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(Path::new(&home).join(".local/share/fonts"));
        dirs.push(Path::new(&home).join(".fonts"));
    }
    dirs
}

/// File name -> first path found (directories in priority order, depth-first).
fn font_index() -> &'static HashMap<String, PathBuf> {
    static INDEX: OnceLock<HashMap<String, PathBuf>> = OnceLock::new();
    INDEX.get_or_init(|| {
        let mut map = HashMap::new();
        fn walk(dir: &Path, depth: u32, map: &mut HashMap<String, PathBuf>) {
            let Ok(rd) = std::fs::read_dir(dir) else { return };
            let mut entries: Vec<_> = rd.flatten().map(|e| e.path()).collect();
            entries.sort();
            for p in entries {
                if p.is_dir() {
                    if depth < 6 {
                        walk(&p, depth + 1, map);
                    }
                } else if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    map.entry(name.to_string()).or_insert_with(|| p.clone());
                }
            }
        }
        for d in font_dirs() {
            walk(&d, 0, &mut map);
        }
        map
    })
}

/// Resolve the file backing a system font set: the console's own font when
/// a font directory provides it, else a metric-compatible substitute, else
/// a generic host face of the same style.
pub fn find_system_font_file(font_set_type: u32) -> Option<(PathBuf, u32)> {
    if !is_system_font_set(font_set_type) {
        return None;
    }
    let files = &FAMILY_FILES[family_of(font_set_type) as usize];
    let w = weight_of(font_set_type);
    let index = font_index();
    if let Some(p) = index.get(files.system[w]) {
        return Some((p.clone(), 0));
    }
    if let Some(p) = index.get(files.substitute[w]) {
        return Some((p.clone(), files.substitute_index));
    }
    files.host[w].iter().find_map(|n| index.get(*n)).map(|p| (p.clone(), 0))
}

/// Load a font file, sharing the bytes between handles opened on the same
/// path.
pub fn load_font_file(path: &Path, sub_font: u32) -> Option<Arc<FaceData>> {
    static LOADED: Mutex<Vec<(PathBuf, u32, Weak<FaceData>)>> = Mutex::new(Vec::new());
    let mut loaded = LOADED.lock().unwrap_or_else(|p| p.into_inner());
    loaded.retain(|(_, _, w)| w.strong_count() > 0);
    if let Some(f) = loaded.iter().find(|(p, i, _)| p == path && *i == sub_font).and_then(|(_, _, w)| w.upgrade()) {
        return Some(f);
    }
    let bytes = std::fs::read(path).ok()?;
    let face = FaceData::parse(bytes, sub_font)?;
    loaded.push((path.to_path_buf(), sub_font, Arc::downgrade(&face)));
    Some(face)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_and_weight_selection() {
        assert_eq!(family_of(0x18070044), Family::Latin);
        assert_eq!(weight_of(0x18070044), 1);
        assert_eq!(weight_of(0x18070047), 3);
        assert_eq!(family_of(0x18170044), Family::LatinItalic);
        assert_eq!(family_of(0x18370044), Family::Typewriter);
        assert_eq!(family_of(0x18080444), Family::Japanese);
        assert_eq!(family_of(0x180C8044), Family::ChineseGb);
        assert!(is_system_font_set(0x18070044));
        assert!(!is_system_font_set(0x12345678));
    }

    #[test]
    fn rasterize_square() {
        // A 10x10 unit square at scale 1: fully covered 10x10 bitmap whose
        // top edge sits 10 pixels above the baseline.
        let p = |x, y| point(x, y);
        let segs = vec![
            Seg::Line(p(0.0, 0.0), p(10.0, 0.0)),
            Seg::Line(p(10.0, 0.0), p(10.0, 10.0)),
            Seg::Line(p(10.0, 10.0), p(0.0, 10.0)),
            Seg::Line(p(0.0, 10.0), p(0.0, 0.0)),
        ];
        let b = rasterize(&segs, 1.0, 1.0, 0.0, 0.0, 0.0);
        assert_eq!((b.left, b.top, b.width, b.height), (0, -10, 10, 10));
        assert!(b.coverage.iter().all(|&c| c == 255));
    }

    /// Host-dependent: resolves the common system font sets to loadable
    /// faces on this machine (`cargo test -- --ignored`).
    #[test]
    #[ignore]
    fn host_system_font_lookup() {
        for set in [0x18070044u32, 0x18070047, 0x18170044, 0x18370044, 0x18080444] {
            let found = find_system_font_file(set);
            eprintln!("{set:#010x} -> {found:?}");
            let (path, sub) = found.expect("no font for system set");
            let face = load_font_file(&path, sub).expect("unloadable font");
            assert!(face.glyph_index('A' as u32).is_some());
        }
    }
}
