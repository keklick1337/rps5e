// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Software rendering of egui output to RGBA pixels, for tests that look
//! at a widget without a window or GPU: the tessellated meshes are
//! rasterised on the CPU (premultiplied alpha, bilinear texture sampling)
//! and written as PNG.

use eframe::egui::{self, epaint, Color32, ImageData, TextureId};
use std::collections::HashMap;
use std::path::Path;

struct Texture {
    size: [usize; 2],
    pixels: Vec<Color32>,
}

/// Renders frames of one egui context.
pub struct SoftRenderer {
    pub ctx: egui::Context,
    textures: HashMap<TextureId, Texture>,
    size: [f32; 2],
}

impl SoftRenderer {
    /// A context with a `size` (points) screen at `ppp` pixels per point.
    pub fn new(size: [f32; 2], ppp: f32) -> Self {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(ppp);
        Self { ctx, textures: HashMap::new(), size }
    }

    /// Run one frame with `events` and return its pixels (width, height, RGBA).
    pub fn frame(&mut self, events: Vec<egui::Event>, mut ui: impl FnMut(&egui::Context)) -> (usize, usize, Vec<u8>) {
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(self.size[0], self.size[1]))),
            events,
            ..Default::default()
        };
        let out = self.ctx.run(raw, |ctx| ui(ctx));
        for (id, delta) in out.textures_delta.set {
            self.apply(id, delta);
        }
        let ppp = out.pixels_per_point;
        let prims = self.ctx.tessellate(out.shapes, ppp);
        let w = (self.size[0] * ppp).round() as usize;
        let h = (self.size[1] * ppp).round() as usize;
        let mut px = vec![[0f32, 0.0, 0.0, 1.0]; w * h];
        for p in prims {
            let epaint::Primitive::Mesh(mesh) = p.primitive else { continue };
            let Some(tex) = self.textures.get(&mesh.texture_id) else { continue };
            let clip = [
                (p.clip_rect.min.x * ppp).max(0.0),
                (p.clip_rect.min.y * ppp).max(0.0),
                (p.clip_rect.max.x * ppp).min(w as f32),
                (p.clip_rect.max.y * ppp).min(h as f32),
            ];
            for tri in mesh.indices.chunks_exact(3) {
                let v = [mesh.vertices[tri[0] as usize], mesh.vertices[tri[1] as usize], mesh.vertices[tri[2] as usize]];
                raster(&mut px, w, &v, ppp, clip, tex);
            }
        }
        let rgba = px.iter().flat_map(|c| c.map(|x| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)).collect();
        (w, h, rgba)
    }

    fn apply(&mut self, id: TextureId, delta: epaint::ImageDelta) {
        let (size, pixels): ([usize; 2], Vec<Color32>) = match &delta.image {
            ImageData::Color(c) => (c.size, c.pixels.clone()),
            ImageData::Font(f) => (f.size, f.srgba_pixels(None).collect()),
        };
        match delta.pos {
            None => {
                self.textures.insert(id, Texture { size, pixels });
            }
            Some([x0, y0]) => {
                let Some(t) = self.textures.get_mut(&id) else { return };
                for y in 0..size[1] {
                    for x in 0..size[0] {
                        t.pixels[(y0 + y) * t.size[0] + x0 + x] = pixels[y * size[0] + x];
                    }
                }
            }
        }
    }
}

fn sample(t: &Texture, u: f32, v: f32) -> [f32; 4] {
    let x = (u * t.size[0] as f32 - 0.5).clamp(0.0, t.size[0] as f32 - 1.0);
    let y = (v * t.size[1] as f32 - 0.5).clamp(0.0, t.size[1] as f32 - 1.0);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(t.size[0] - 1), (y0 + 1).min(t.size[1] - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let at = |x: usize, y: usize| t.pixels[y * t.size[0] + x].to_array().map(|c| c as f32 / 255.0);
    let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
    let mut o = [0.0; 4];
    for i in 0..4 {
        o[i] = (a[i] * (1.0 - fx) + b[i] * fx) * (1.0 - fy) + (c[i] * (1.0 - fx) + d[i] * fx) * fy;
    }
    o
}

fn raster(px: &mut [[f32; 4]], w: usize, v: &[epaint::Vertex; 3], ppp: f32, clip: [f32; 4], tex: &Texture) {
    let p: Vec<[f32; 2]> = v.iter().map(|v| [v.pos.x * ppp, v.pos.y * ppp]).collect();
    let area = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
    if area.abs() < 1e-8 {
        return;
    }
    let minx = p.iter().map(|q| q[0]).fold(f32::MAX, f32::min).max(clip[0]).floor() as i64;
    let maxx = p.iter().map(|q| q[0]).fold(f32::MIN, f32::max).min(clip[2]).ceil() as i64;
    let miny = p.iter().map(|q| q[1]).fold(f32::MAX, f32::min).max(clip[1]).floor() as i64;
    let maxy = p.iter().map(|q| q[1]).fold(f32::MIN, f32::max).min(clip[3]).ceil() as i64;
    let col: Vec<[f32; 4]> = v.iter().map(|v| v.color.to_array().map(|c| c as f32 / 255.0)).collect();
    for y in miny.max(0)..maxy {
        for x in minx.max(0)..maxx {
            let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
            if cx < clip[0] || cx >= clip[2] || cy < clip[1] || cy >= clip[3] {
                continue;
            }
            let w0 = ((p[1][0] - cx) * (p[2][1] - cy) - (p[2][0] - cx) * (p[1][1] - cy)) / area;
            let w1 = ((p[2][0] - cx) * (p[0][1] - cy) - (p[0][0] - cx) * (p[2][1] - cy)) / area;
            let w2 = 1.0 - w0 - w1;
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }
            let u = w0 * v[0].uv.x + w1 * v[1].uv.x + w2 * v[2].uv.x;
            let vv = w0 * v[0].uv.y + w1 * v[1].uv.y + w2 * v[2].uv.y;
            let t = sample(tex, u, vv);
            let mut s = [0.0; 4];
            for i in 0..4 {
                s[i] = (w0 * col[0][i] + w1 * col[1][i] + w2 * col[2][i]) * t[i];
            }
            let d = &mut px[y as usize * w + x as usize];
            for i in 0..4 {
                d[i] = s[i] + d[i] * (1.0 - s[3]);
            }
        }
    }
}

/// Write RGBA pixels as a PNG.
pub fn write_png(path: &Path, w: usize, h: usize, rgba: &[u8]) {
    let f = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(rgba).unwrap();
}
