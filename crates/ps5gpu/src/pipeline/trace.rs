// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Frame trace (debug aid): with `RPS5E_FRAME_TRACE=<list>` (frame
//! numbers or `a-b` ranges, comma-separated), every draw and dispatch of
//! those presented frames (the work before their present) is logged with its outcome, shaders, targets and bound images;
//! `@<seconds>` picks the first frame presented that long after start. With
//! `RPS5E_FRAME_TRACE_DIR=<dir>` the images each recorded pass writes are
//! read back after the pass and written as `<dir>/NNNN_<what>_<addr>.ppm`
//! (layer 0, downscaled to at most 960 pixels wide; float formats clamped to
//! 0..1), with per-channel min/max/mean in the log.

use crate::device::{Buffer, GpuDevice, MemoryUse};
use crate::images::ImageCache;
use crate::scheduler::Scheduler;
use ash::vk;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

struct Config {
    frames: Vec<(u64, u64)>,
    /// `@<seconds>`: the first frame presented that long after start
    /// (frame numbers drift between runs; wall time is steadier).
    at_seconds: Vec<u64>,
    start: std::time::Instant,
    dir: Option<std::path::PathBuf>,
    /// `RPS5E_FRAME_TRACE_COUNT=<n>`: consecutive frames traced from each
    /// `@<seconds>` entry (default 1).
    count: u64,
}

/// Frame numbers chosen by `@<seconds>` entries (set once reached).
static TIMED: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

impl Config {
    fn traced(&self, f: u64) -> bool {
        self.frames.iter().any(|&(a, b)| f >= a && f <= b)
            || TIMED.iter().any(|t| {
                let s = t.load(Ordering::Relaxed);
                s != 0 && f >= s && f < s + self.count
            })
    }
}

fn config() -> Option<&'static Config> {
    static C: OnceLock<Option<Config>> = OnceLock::new();
    C.get_or_init(|| {
        let v = std::env::var("RPS5E_FRAME_TRACE").ok()?;
        let at_seconds: Vec<u64> = v.split(',').filter_map(|p| p.trim().strip_prefix('@')?.parse().ok()).take(TIMED.len()).collect();
        let frames: Vec<(u64, u64)> = v
            .split(',')
            .filter_map(|p| {
                let p = p.trim();
                match p.split_once('-') {
                    Some((a, b)) => Some((a.trim().parse().ok()?, b.trim().parse().ok()?)),
                    None => p.parse().ok().map(|a| (a, a)),
                }
            })
            .collect();
        let dir = std::env::var_os("RPS5E_FRAME_TRACE_DIR").map(std::path::PathBuf::from);
        if let Some(d) = &dir {
            let _ = std::fs::create_dir_all(d);
        }
        let count = std::env::var("RPS5E_FRAME_TRACE_COUNT").ok().and_then(|v| v.parse().ok()).unwrap_or(1u64).max(1);
        Some(Config { frames, at_seconds, start: std::time::Instant::now(), dir, count })
    })
    .as_ref()
}

static FRAME: AtomicU64 = AtomicU64::new(0);
static SEQ: AtomicU64 = AtomicU64::new(0);

/// A frame was presented.
pub fn frame_presented() {
    if let Some(c) = config() {
        let f = FRAME.fetch_add(1, Ordering::Relaxed) + 1;
        let elapsed = c.start.elapsed().as_secs();
        for (i, &s) in c.at_seconds.iter().enumerate() {
            if elapsed >= s {
                let _ = TIMED[i].compare_exchange(0, f, Ordering::Relaxed, Ordering::Relaxed);
            }
        }
        if c.traced(f) {
            log::info!("[frame-trace] ===== frame {f} =====");
        }
    }
}

/// Is the current frame traced?
pub fn active() -> bool {
    config().is_some_and(|c| c.traced(FRAME.load(Ordering::Relaxed)))
}

/// Sequence number the next traced pass will get.
/// True when passes read their results back (an image directory is set).
pub fn readbacks() -> bool {
    config().is_some_and(|c| c.dir.is_some())
}

pub fn peek_seq() -> u64 {
    SEQ.load(Ordering::Relaxed)
}

/// Sequence number of the next traced pass.
pub fn next_seq() -> u64 {
    SEQ.fetch_add(1, Ordering::Relaxed)
}

/// Log one line of the trace.
pub fn log(line: &str) {
    log::info!("[frame-trace] {line}");
}

fn f16(h: u16) -> f32 {
    let s = (h >> 15) as u32;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    let v = if e == 0 {
        m * 2f32.powi(-24)
    } else if e == 31 {
        if m == 0.0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    };
    if s != 0 {
        -v
    } else {
        v
    }
}

fn uf(bits: u32, mant: u32) -> f32 {
    let e = (bits >> mant) as i32;
    let m = (bits & ((1 << mant) - 1)) as f32;
    if e == 0 {
        m / (1 << mant) as f32 * 2f32.powi(-14)
    } else if e == 31 {
        f32::INFINITY
    } else {
        (1.0 + m / (1 << mant) as f32) * 2f32.powi(e - 15)
    }
}

/// Bytes per texel and a decoder to RGBA floats.
fn decoder(format: vk::Format) -> Option<(u32, fn(&[u8]) -> [f32; 4])> {
    use vk::Format as F;
    fn u8n(b: u8) -> f32 {
        b as f32 / 255.0
    }
    fn word(b: &[u8]) -> u32 {
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
    fn half(b: &[u8], i: usize) -> f32 {
        f16(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]))
    }
    fn float(b: &[u8], i: usize) -> f32 {
        f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
    }
    Some(match format {
        F::R8G8B8A8_UNORM | F::R8G8B8A8_SRGB | F::R8G8B8A8_UINT | F::R8G8B8A8_SNORM => (4, |b| [u8n(b[0]), u8n(b[1]), u8n(b[2]), u8n(b[3])]),
        F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB => (4, |b| [u8n(b[2]), u8n(b[1]), u8n(b[0]), u8n(b[3])]),
        F::R8_UNORM | F::R8_UINT => (1, |b| [u8n(b[0]), 0.0, 0.0, 1.0]),
        F::R8G8_UNORM | F::R8G8_UINT => (2, |b| [u8n(b[0]), u8n(b[1]), 0.0, 1.0]),
        F::R16G16B16A16_SFLOAT => (8, |b| [half(b, 0), half(b, 1), half(b, 2), half(b, 3)]),
        F::R16G16_SFLOAT => (4, |b| [half(b, 0), half(b, 1), 0.0, 1.0]),
        F::R16_SFLOAT => (2, |b| [half(b, 0), 0.0, 0.0, 1.0]),
        F::R16_UNORM | F::R16_UINT | F::D16_UNORM => (2, |b| [u16::from_le_bytes([b[0], b[1]]) as f32 / 65535.0, 0.0, 0.0, 1.0]),
        F::R16G16_UNORM => (4, |b| [u16::from_le_bytes([b[0], b[1]]) as f32 / 65535.0, u16::from_le_bytes([b[2], b[3]]) as f32 / 65535.0, 0.0, 1.0]),
        F::R16G16B16A16_UNORM => (8, |b| {
            let c = |i: usize| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as f32 / 65535.0;
            [c(0), c(1), c(2), c(3)]
        }),
        F::R32_SFLOAT | F::D32_SFLOAT => (4, |b| [float(b, 0), 0.0, 0.0, 1.0]),
        F::R32_UINT => (4, |b| [word(b) as f32, 0.0, 0.0, 1.0]),
        F::R32G32_SFLOAT => (8, |b| [float(b, 0), float(b, 1), 0.0, 1.0]),
        F::R32G32B32A32_SFLOAT => (16, |b| [float(b, 0), float(b, 1), float(b, 2), float(b, 3)]),
        F::B10G11R11_UFLOAT_PACK32 => (4, |b| {
            let w = word(b);
            [uf(w & 0x7ff, 6), uf((w >> 11) & 0x7ff, 6), uf(w >> 22, 5), 1.0]
        }),
        F::A2B10G10R10_UNORM_PACK32 => (4, |b| {
            let w = word(b);
            [(w & 0x3ff) as f32 / 1023.0, ((w >> 10) & 0x3ff) as f32 / 1023.0, ((w >> 20) & 0x3ff) as f32 / 1023.0, (w >> 30) as f32 / 3.0]
        }),
        F::A2R10G10B10_UNORM_PACK32 => (4, |b| {
            let w = word(b);
            [((w >> 20) & 0x3ff) as f32 / 1023.0, ((w >> 10) & 0x3ff) as f32 / 1023.0, (w & 0x3ff) as f32 / 1023.0, (w >> 30) as f32 / 3.0]
        }),
        _ => return None,
    })
}

/// The leading dwords of a guest range as ` [hex...]` (empty when the
/// range is not readable). Debug aid: shows constants and vertex records.
pub fn guest_words(addr: u64, size: u64) -> String {
    let n = (size / 4) as usize;
    let mut words = vec![0u8; n * 4];
    if n == 0 || !crate::memory::global().read_backing(addr, &mut words) {
        return String::new();
    }
    let hex: Vec<String> = words.chunks_exact(4).map(|c| format!("{:08x}", u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();
    format!(" [{}]", hex.join(" "))
}

/// Read back layer 0 of image `id` and write it (plus stats) when an image
/// directory is configured. Waits for the GPU.
/// `RPS5E_FRAME_TRACE_RAW=1`: also write unclamped values of non-8-bit
/// images as `.pfm` (precision checks of HDR targets).
/// `RPS5E_FRAME_TRACE_SKIP=<kinds>` (comma-separated, e.g. `z,tex`):
/// image dumps of those kinds are left out; `RPS5E_FRAME_TRACE_FROM=<n>`:
/// no image dumps before traced pass `n` (a whole frame of dumps takes
/// minutes).
pub fn dump_wanted(seq: u64, what: &str) -> bool {
    static CFG: OnceLock<(Vec<String>, u64)> = OnceLock::new();
    let (skip, from) = CFG.get_or_init(|| {
        let skip = std::env::var("RPS5E_FRAME_TRACE_SKIP").map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()).unwrap_or_default();
        let from = std::env::var("RPS5E_FRAME_TRACE_FROM").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        (skip, from)
    });
    seq >= *from && !skip.iter().any(|k| k == what)
}

/// `RPS5E_FRAME_TRACE_ADDR=<hex,...>`: sampled images at these addresses
/// are dumped too (as `tex`), whatever their size.
pub fn dump_address(addr: u64) -> bool {
    static LIST: OnceLock<Vec<u64>> = OnceLock::new();
    LIST.get_or_init(|| {
        std::env::var("RPS5E_FRAME_TRACE_ADDR")
            .map(|v| v.split(',').filter_map(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()).collect())
            .unwrap_or_default()
    })
    .contains(&addr)
}

/// `RPS5E_FRAME_TRACE_ARRAYS=1`: sampled array and cube images are
/// dumped too (as `tex`).
pub fn dump_arrays() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_FRAME_TRACE_ARRAYS").is_ok_and(|v| v == "1"))
}

/// `RPS5E_FRAME_TRACE_MRT=<n>`: only passes writing at least `n` colour
/// targets dump images (e.g. 4: the G-buffer passes).
pub fn dump_min_targets() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::env::var("RPS5E_FRAME_TRACE_MRT").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// `RPS5E_FRAME_TRACE_LAST=1`: an image used by consecutive passes is
/// dumped once, after the last of them (named by that pass).
pub fn dump_last_only() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_FRAME_TRACE_LAST").is_ok_and(|v| v == "1"))
}

fn raw_dumps() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_FRAME_TRACE_RAW").is_ok_and(|v| v == "1"))
}

fn is_8bit(f: vk::Format) -> bool {
    matches!(
        f,
        vk::Format::R8_UNORM
            | vk::Format::R8_SRGB
            | vk::Format::R8G8_UNORM
            | vk::Format::R8G8_SRGB
            | vk::Format::R8G8B8A8_UNORM
            | vk::Format::R8G8B8A8_SRGB
            | vk::Format::B8G8R8A8_UNORM
            | vk::Format::B8G8R8A8_SRGB
    )
}

pub fn dump_image(dev: &Arc<GpuDevice>, sched: &mut Scheduler, images: &mut ImageCache, id: u64, seq: u64, what: &str) {
    let Some(c) = config() else { return };
    // Read-backs wait for the GPU after every pass: only with an image
    // directory, so wide frame ranges can be logged cheaply without one.
    if c.dir.is_none() {
        return;
    }
    let Some(img) = images.get(id) else { return };
    let key = img.key;
    let fmt = key.format;
    let Some((bpp, decode)) = decoder(if fmt == vk::Format::D32_SFLOAT_S8_UINT { vk::Format::D32_SFLOAT } else { fmt }) else {
        log(&format!("  {what} {:#x} {:?} {}x{}: no readback decoder", key.addr(), fmt, key.width, key.height));
        return;
    };
    if key.samples > 1 {
        log(&format!("  {what} {:#x}: multisampled, not read back", key.addr()));
        return;
    }
    let (w, h) = (key.width, key.height);
    let size = w as u64 * h as u64 * bpp as u64;
    // Small volumes and arrays: every slice is read back and its mean logged.
    let slices = if size * (key.depth.max(key.layers) as u64) <= 4 << 20 { key.depth.max(key.layers).max(1) } else { 1 };
    let full = size * slices as u64;
    let Ok(buf) = Buffer::new(dev, full, vk::BufferUsageFlags::TRANSFER_DST, MemoryUse::Download) else { return };
    let aspect = if key.is_depth() { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR };
    if aspect == vk::ImageAspectFlags::DEPTH && !matches!(fmt, vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D16_UNORM) {
        log(&format!("  {what} {:#x}: depth format {fmt:?} not read back", key.addr()));
        return;
    }
    let image = img.image.image;
    sched.full_barrier();
    images.transit(sched, id, crate::images::TRANSFER_SRC, None);
    let region = vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: 0, layer_count: if key.depth > 1 { 1 } else { slices } })
        .image_extent(vk::Extent3D { width: w, height: h, depth: if key.depth > 1 { slices } else { 1 } });
    let cb = sched.cmd();
    // SAFETY: recording buffer; image in TRANSFER_SRC layout.
    unsafe { dev.device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buf.buffer, &[region]) };
    sched.full_barrier();
    sched.flush_and_wait();
    buf.invalidate(0, full);
    let all = &buf.bytes_mut()[..full as usize];
    if slices > 1 {
        let means: Vec<String> = all
            .chunks_exact(size as usize)
            .map(|sl| {
                let mut m = [0f64; 4];
                for px in sl.chunks_exact(bpp as usize) {
                    let v = decode(px);
                    for i in 0..4 {
                        m[i] += v[i] as f64;
                    }
                }
                let n = (w * h) as f64;
                format!("({:.3} {:.3} {:.3} {:.3})", m[0] / n, m[1] / n, m[2] / n, m[3] / n)
            })
            .collect();
        log(&format!("  {what} {:#x} slice means: {}", key.addr(), means.join(" ")));
    }
    let bytes = &all[..size as usize];
    let step = w.div_ceil(960).max(1);
    let (ow, oh) = (w.div_ceil(step), h.div_ceil(step));
    let mut lo = [f32::INFINITY; 4];
    let mut hi = [f32::NEG_INFINITY; 4];
    let mut sum = [0f64; 4];
    let mut n = 0f64;
    let mut out = format!("P6\n{ow} {oh}\n255\n").into_bytes();
    let mut alpha = format!("P5\n{ow} {oh}\n255\n").into_bytes();
    // Unclamped values (PFM, bottom row first) for float and >8-bit formats.
    let raw_wanted = raw_dumps() && !is_8bit(fmt);
    let mut raw: Vec<[f32; 3]> = Vec::new();
    for y in (0..h).step_by(step as usize) {
        for x in (0..w).step_by(step as usize) {
            let o = ((y * w + x) * bpp) as usize;
            let px = decode(&bytes[o..o + bpp as usize]);
            for i in 0..4 {
                if px[i].is_finite() {
                    lo[i] = lo[i].min(px[i]);
                    hi[i] = hi[i].max(px[i]);
                    sum[i] += px[i] as f64;
                }
            }
            n += 1.0;
            for &v in &px[..3] {
                out.push((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
            alpha.push((px[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            if raw_wanted {
                raw.push([px[0], px[1], px[2]]);
            }
        }
    }
    let mean: Vec<String> = sum.iter().map(|s| format!("{:.4}", s / n)).collect();
    log(&format!("  {what} {:#x} {fmt:?} {w}x{h}x{}x{} min={lo:?} max={hi:?} mean={mean:?}", key.addr(), key.depth, key.layers));
    if let Some(dir) = &c.dir {
        let _ = std::fs::write(dir.join(format!("{seq:04}_{what}_{:x}.ppm", key.addr())), out);
        if lo[3] != hi[3] {
            let _ = std::fs::write(dir.join(format!("{seq:04}_{what}_{:x}_a.pgm", key.addr())), alpha);
        }
        if raw_wanted {
            let mut pfm = format!("PF\n{ow} {oh}\n-1.0\n").into_bytes();
            for row in raw.chunks_exact(ow as usize).rev() {
                for px in row {
                    for v in px {
                        pfm.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            let _ = std::fs::write(dir.join(format!("{seq:04}_{what}_{:x}.pfm", key.addr())), pfm);
        }
    }
}
