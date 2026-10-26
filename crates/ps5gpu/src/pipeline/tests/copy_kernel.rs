// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The SDK image-copy kernel (`crate::copy_kernels`): the same dispatch
//! through the shader path and as an image copy through the back-end
//! leaves the same bytes in the destination's guest memory; dispatches the
//! copy would not reproduce run the shader.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored copy_kernel`.

use super::*;
use crate::copy_kernels::IMAGE_COPY_KERNEL;
use crate::pipeline::sharps::{image_type, texel_format, texture_key, TextureSharp};
use crate::tiling::{SWIZZLE_64KB_R_X, SWIZZLE_64KB_S_X, SWIZZLE_LINEAR};

/// RSRC2 of the kernel: 16 user SGPRs, TGID_X / TGID_Y, thread ids x and y.
const RSRC2: u32 = 16 << 1 | 1 << 7 | 1 << 8 | 1 << 11;

#[derive(Clone, Copy, Debug)]
struct Surface {
    format: u32,
    w: u32,
    h: u32,
    /// BASE_LEVEL; LAST_LEVEL and MAX_MIP are `last`.
    base: u32,
    last: u32,
    swizzle: u32,
    sel: u32,
}

impl Surface {
    fn new(format: u32, w: u32, h: u32, swizzle: u32) -> Surface {
        Surface { format, w, h, base: 0, last: 0, swizzle, sel: 0xFAC }
    }

    fn words(&self, addr: u64) -> [u32; 8] {
        let (w, h) = (self.w - 1, self.h - 1);
        [
            (addr >> 8) as u32,
            ((addr >> 40) as u32 & 0xFF) | self.format << 20 | (w & 3) << 30,
            w >> 2 | h << 14,
            self.sel | self.base << 12 | self.last << 16 | self.swizzle << 20 | image_type::TYPE_2D << 28,
            0,
            self.last << 4,
            0,
            0,
        ]
    }

    /// Guest bytes of the surface (every level).
    fn bytes(&self) -> u64 {
        let words = self.words(0x100_0000);
        let t = TextureSharp(words);
        texture_key(&t, &texel_format(self.format).unwrap()).unwrap().size()
    }
}

/// Deterministic bytes (xorshift).
fn noise(len: u64, seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

fn backend(g: &Guest) -> Backend {
    let dev = GpuDevice::create(false).expect("device");
    let mut b = Backend::new(&dev, g.mem).unwrap();
    let compiler: Arc<dyn ShaderCompiler> = Arc::new(crate::shader::SpirvCompiler::for_device(&dev, g.mem));
    let mut p = ShaderPipelines::new(&dev, g.mem, Some(compiler), None).unwrap();
    p.set_async_compile(false);
    b.pipelines = Box::new(p);
    b
}

fn copy_dispatch(cs: u64, src: [u32; 8], dst: [u32; 8], groups: [u32; 3]) -> ResolvedDispatch {
    let user: Vec<u32> = src.iter().chain(&dst).copied().collect();
    let mut shader = stage(cs, &user);
    shader.rsrc1 = (0xC0 << 12) | (1 << 21);
    shader.rsrc2 = RSRC2;
    ResolvedDispatch { packet: DispatchPacket { groups, dispatch_initiator: 1, ..DispatchPacket::default() }, shader, rsrc3: 0, threads: [8, 8, 1], start: [0; 3] }
}

/// The dispatch through the shader pipelines only (no back-end hook).
fn shader_path(b: &mut Backend, d: &ResolvedDispatch) {
    let targets = DrawTargets { colors: Default::default(), depth: None };
    let mut ctx = DrawContext { dev: &b.dev, sched: &mut b.sched, buffers: (&mut b.buffers).into(), images: (&mut b.images).into(), targets: &targets };
    assert_eq!(b.pipelines.dispatch(&mut ctx, d), DrawOutcome::Recorded, "the shader path did not record the dispatch");
}

/// Guest bytes of `[addr, addr + size)` once the GPU's images reached them.
fn guest_bytes(b: &mut Backend, g: &Guest, addr: u64, size: u64) -> Vec<u8> {
    b.images.writeback_range(&mut b.sched, &mut b.buffers, addr, size);
    b.buffers.download(&mut b.sched, addr, size);
    let mut out = vec![0u8; size as usize];
    assert!(g.mem.read_backing(addr, &mut out));
    out
}

struct Outcome {
    shader: Vec<u8>,
    back_end: Vec<u8>,
    poison: Vec<u8>,
    copies: u64,
}

/// Run the kernel from `src` into two destinations of shape `dst` holding
/// the same bytes: through the shader path into one, through the
/// back-end's dispatch (and its copy-kernel hook) into the other.
fn run(src: Surface, dst: Surface, groups: [u32; 3], seed: u64) -> Outcome {
    let mut g = Guest::new(64 << 20);
    let cs = g.program(&IMAGE_COPY_KERNEL);
    let source = g.alloc(&noise(src.bytes(), seed));
    let poison = noise(dst.bytes(), seed ^ 0x5555);
    let a = g.alloc(&poison);
    let c = g.alloc(&poison);
    let mut b = backend(&g);
    shader_path(&mut b, &copy_dispatch(cs, src.words(source), dst.words(a), groups));
    let before = b.stats.copy_kernels;
    b.dispatch(&copy_dispatch(cs, src.words(source), dst.words(c), groups));
    let copies = b.stats.copy_kernels - before;
    let size = dst.bytes();
    Outcome { shader: guest_bytes(&mut b, &g, a, size), back_end: guest_bytes(&mut b, &g, c, size), poison, copies }
}

fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

#[test]
#[ignore]
fn copy_kernel_matches_the_shader_path() {
    let mut r16f = Surface::new(71, 64, 48, SWIZZLE_64KB_R_X);
    r16f.sel = 0xFAC;
    let mut r32f = Surface::new(22, 30, 30, SWIZZLE_LINEAR);
    r32f.sel = 0x204;
    let mut r32f_small = r32f;
    (r32f_small.w, r32f_small.h) = (13, 9);
    let mut mips = Surface::new(20, 64, 40, SWIZZLE_64KB_S_X);
    (mips.base, mips.last, mips.sel) = (1, 3, 0x204);
    let mut mips2 = mips;
    mips2.base = 2;
    let mut bgra = Surface::new(56, 40, 24, SWIZZLE_64KB_R_X);
    bgra.sel = 0xF2E;
    let cases: [(&str, Surface, Surface, [u32; 3]); 8] = [
        ("RGBA8 linear, grid past the image", Surface::new(56, 37, 21, SWIZZLE_LINEAR), Surface::new(56, 37, 21, SWIZZLE_LINEAR), [5, 3, 1]),
        ("RGBA8 tiled, grid inside the image", Surface::new(56, 100, 60, SWIZZLE_64KB_R_X), Surface::new(56, 100, 60, SWIZZLE_64KB_R_X), [6, 4, 2]),
        ("RGBA16F tiled", r16f, r16f, [6, 4, 1]),
        ("R32F, destination smaller", r32f, r32f_small, [8, 8, 1]),
        ("R32 uint, level 1 into level 1", mips, mips, [8, 8, 1]),
        ("R32 uint, level 2 into level 1 (grid inside level 2)", mips2, mips, [2, 1, 1]),
        ("RGBA8 BGRA swizzle on both", bgra, bgra, [5, 3, 1]),
        ("R8 linear tiled", Surface::new(1, 50, 17, SWIZZLE_LINEAR), Surface::new(1, 50, 17, SWIZZLE_64KB_S_X), [7, 3, 1]),
    ];
    for (k, (name, src, dst, groups)) in cases.into_iter().enumerate() {
        let o = run(src, dst, groups, 0x1234_5678 + k as u64);
        assert_eq!(o.copies, 1, "{name}: the back-end did not run the copy");
        assert!(o.shader != o.poison, "{name}: the shader path wrote nothing");
        if let Some(i) = first_difference(&o.shader, &o.back_end) {
            panic!("{name}: byte {i:#x} differs: shader {:02x?} copy {:02x?}", &o.shader[i..(i + 16).min(o.shader.len())], &o.back_end[i..(i + 16).min(o.back_end.len())]);
        }
    }
}

/// RGBA16F and R32F with every bit pattern the noise holds (NaNs,
/// infinities and denormals included) and RG16 / RGBA32F.
#[test]
#[ignore]
fn copy_kernel_matches_the_shader_path_on_float_bits() {
    for (format, w, h) in [(13u32, 33u32, 17u32), (29, 16, 16), (64, 24, 8), (77, 16, 16), (23, 19, 11), (65, 16, 16), (50, 16, 16)] {
        let s = Surface::new(format, w, h, SWIZZLE_LINEAR);
        let o = run(s, s, [w.div_ceil(8), h.div_ceil(8), 1], format as u64 * 77);
        assert_eq!(o.copies, 1, "format {format}");
        assert!(o.shader != o.poison, "format {format}: the shader path wrote nothing");
        assert_eq!(first_difference(&o.shader, &o.back_end), None, "format {format}");
    }
}

/// The copied texels of a linear surface are the source's, the rest of
/// the destination keeps its bytes. Signed integer formats are checked
/// here only: the shader path binds no view for a stored R32 sint image
/// (and refuses the other sint storage formats).
#[test]
#[ignore]
fn copy_kernel_copies_the_grid_inside_the_images() {
    for (format, sel, bpe) in [(56u32, 0xFACu32, 4usize), (21, 0x204, 4), (61, 0xFAC, 4), (12, 0x204, 2), (76, 0xFAC, 16)] {
        let mut s = Surface::new(format, 37, 21, SWIZZLE_LINEAR);
        s.sel = sel;
        let mut g = Guest::new(64 << 20);
        let cs = g.program(&IMAGE_COPY_KERNEL);
        let src_bytes = noise(s.bytes(), 7 + format as u64);
        let source = g.alloc(&src_bytes);
        let poison = noise(s.bytes(), 8);
        let dst = g.alloc(&poison);
        let mut b = backend(&g);
        // 3x2 groups: 24x16 texels.
        b.dispatch(&copy_dispatch(cs, s.words(source), s.words(dst), [3, 2, 1]));
        assert_eq!(b.stats.copy_kernels, 1, "format {format}");
        let out = guest_bytes(&mut b, &g, dst, s.bytes());
        let pitch = texture_key(&TextureSharp(s.words(source)), &texel_format(format).unwrap()).unwrap().main.pitch as usize;
        for y in 0..21usize {
            for x in 0..37usize {
                let o = (y * pitch + x) * bpe;
                let want = if x < 24 && y < 16 { &src_bytes[o..o + bpe] } else { &poison[o..o + bpe] };
                assert_eq!(&out[o..o + bpe], want, "format {format} texel ({x}, {y})");
            }
        }
    }
}

/// Dispatches the copy would not reproduce run the shader: different
/// formats, a source smaller than the copied area (zeros past it), sRGB,
/// snorm, swizzles that do not cancel.
#[test]
#[ignore]
fn copy_kernel_falls_back_to_the_shader() {
    let mut bgra = Surface::new(56, 32, 32, SWIZZLE_LINEAR);
    bgra.sel = 0xF2E;
    let cases: [(&str, Surface, Surface, [u32; 3]); 5] = [
        ("unorm into uint", Surface::new(56, 32, 32, SWIZZLE_LINEAR), Surface::new(60, 32, 32, SWIZZLE_LINEAR), [4, 4, 1]),
        ("source smaller than the copied area", Surface::new(56, 13, 9, SWIZZLE_LINEAR), Surface::new(56, 32, 32, SWIZZLE_LINEAR), [4, 4, 1]),
        ("sRGB", Surface::new(130, 32, 32, SWIZZLE_LINEAR), Surface::new(130, 32, 32, SWIZZLE_LINEAR), [4, 4, 1]),
        ("snorm", Surface::new(57, 32, 32, SWIZZLE_LINEAR), Surface::new(57, 32, 32, SWIZZLE_LINEAR), [4, 4, 1]),
        ("BGRA source into RGBA", bgra, Surface::new(56, 32, 32, SWIZZLE_LINEAR), [4, 4, 1]),
    ];
    for (k, (name, src, dst, groups)) in cases.into_iter().enumerate() {
        let o = run(src, dst, groups, 99 + k as u64);
        assert_eq!(o.copies, 0, "{name}: the copy ran");
        assert!(o.shader != o.poison, "{name}: the shader path wrote nothing");
        assert_eq!(first_difference(&o.shader, &o.back_end), None, "{name}: the fallback differs from the shader path");
    }
}
