// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Image instructions whose MIMG DIM differs from the T#'s type (1D, cube
//! and volume T#s read or written through other dimensions), through the
//! SPIR-V compiler on the device, against hardware-measured results.
//!
//! The programs, inputs and expected values are the reference's driver
//! execution tests of the same names (ImagePlaneAddressOnLine,
//! ImageAddressDimension, ImageVolumeAddressOnLine, ...).
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored image_dims`.

use super::*;
use crate::tiling_layout::SurfaceLayout;
use ps5shader::testing::buffer_descriptor;

pub(super) const THREADS: u32 = 32;
pub(super) const WORDS: usize = 32;
pub(super) const FORMAT_8888_UNORM: u32 = 56;
pub(super) const POISON: u32 = 0xDEAD_BEEF;

/// A raw buffer descriptor as the reference programs were run with.
pub(super) fn raw(base: u64, bytes: u32) -> [u32; 4] {
    buffer_descriptor(base, 0, bytes, 0x3101_6FAC)
}

pub(super) fn bits(v: f32) -> u32 {
    v.to_bits()
}

/// Geometry of a test texture.
#[derive(Clone, Copy)]
pub(super) struct Shape {
    pub(super) format: u32,
    pub(super) bpe: u32,
    pub(super) ty: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    /// Depth of a volume or layers of an array.
    pub(super) slices: u32,
    pub(super) levels: u32,
    pub(super) tile: u32,
}

impl Shape {
    pub(super) fn new(ty: u32, width: u32, height: u32, slices: u32, levels: u32) -> Shape {
        Shape { format: FORMAT_8888_UNORM, bpe: 4, ty, width, height, slices, levels, tile: 0 }
    }

    pub(super) fn volume(&self) -> bool {
        self.ty == sharps::image_type::TYPE_3D
    }

    pub(super) fn layout(&self) -> SurfaceLayout {
        crate::tiling_layout::describe(self.tile, self.bpe, 1, self.width, self.height, self.slices, self.levels, self.volume()).expect("test surface layout")
    }

    pub(super) fn descriptor(&self, base: u64) -> [u32; 8] {
        [
            (base >> 8) as u32,
            ((base >> 40) as u32 & 0xFF) | (self.format << 20) | (((self.width - 1) & 3) << 30),
            ((self.width - 1) >> 2) | ((self.height - 1) << 14),
            0xFAC | ((self.levels - 1) << 16) | (self.tile << 20) | (self.ty << 28),
            self.slices - 1,
            (self.levels - 1) << 4,
            0,
            0,
        ]
    }
}

/// A texture in guest memory: `fill` gives the bytes of texel (x, y) of
/// `slice` of `level` (unfilled bytes are 0xEE).
pub(super) fn texture(g: &mut Guest, s: Shape, fill: impl Fn(u32, u32, u32, u32) -> Option<Vec<u8>>) -> [u32; 8] {
    let layout = s.layout();
    let mut bytes = vec![0xEEu8; layout.guest_bytes as usize];
    for level in 0..s.levels {
        let m = layout.mips[level as usize];
        for slice in 0..layout.level_slices(level) {
            for y in 0..m.height {
                for x in 0..m.width {
                    if let Some(t) = fill(level, slice, x, y) {
                        let o = layout.element_offset(level, slice, x, y).unwrap() as usize;
                        bytes[o..o + t.len()].copy_from_slice(&t);
                    }
                }
            }
        }
    }
    let base = g.alloc(&bytes);
    s.descriptor(base)
}

pub(super) fn rig(g: &Guest) -> Rig {
    // A refused program logs why.
    let _ = env_logger::builder().is_test(true).try_init();
    Rig::with_compiler(g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None)
}

/// One wave32 group of 32 lanes of `code` with `user` SGPRs.
pub(super) fn dispatch(g: &mut Guest, rig: &mut Rig, code: &[u32], user: &[u32]) {
    dispatch_waves(g, rig, code, user, THREADS, 32);
}

/// One group of `threads` lanes of `code` in waves of `wave`.
pub(super) fn dispatch_waves(g: &mut Guest, rig: &mut Rig, code: &[u32], user: &[u32], threads: u32, wave: u32) {
    let cs = g.program(code);
    let mut shader = stage(cs, user);
    shader.rsrc1 = (0xC0 << 12) | (1 << 21);
    let d = ResolvedDispatch {
        packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1 | if wave == 32 { 1 << 15 } else { 0 }, ..DispatchPacket::default() },
        shader,
        rsrc3: 0,
        threads: [threads, 1, 1],
        start: [0; 3],
    };
    assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
}

pub(super) fn read_words(g: &Guest, rig: &mut Rig, addr: u64, words: usize) -> Vec<u32> {
    assert!(rig.buffers.download(&mut rig.sched, addr, words as u64 * 4));
    let mut out = vec![0u8; words * 4];
    assert!(g.mem.read_backing(addr, &mut out));
    out.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// The reference's layout: s[0:3] one buffer (input and results, 32 dwords
/// a lane), s[4:11] the T#, s[12:15] the S#. Returns the buffer after the
/// dispatch.
pub(super) fn run(g: &mut Guest, rig: &mut Rig, code: &[u32], input: &[u32], texture: &[u32; 8], sampler: [u32; 4]) -> Vec<u32> {
    let buffer = g.alloc(&asm::bytes(input));
    let mut user = raw(buffer, (input.len() * 4) as u32).to_vec();
    user.extend_from_slice(texture);
    user.extend_from_slice(&sampler);
    dispatch(g, rig, code, &user);
    read_words(g, rig, buffer, input.len())
}

/// The lanes' input words (32 a lane, the rest poisoned).
pub(super) fn lanes(words: impl Fn(u32) -> Vec<u32>) -> Vec<u32> {
    let mut b = vec![POISON; THREADS as usize * WORDS];
    for tid in 0..THREADS {
        let w = words(tid);
        b[tid as usize * WORDS..tid as usize * WORDS + w.len()].copy_from_slice(&w);
    }
    b
}

/// Point sampler, point mips, clamp to edge off (wrap), MAX_LOD 0xFFF.
pub(super) fn point_sampler() -> [u32; 4] {
    [0, 0xFFF << 12, 1 << 26, 0]
}

/// Component `c` of the result at `result` of lane `tid` scaled to 0..255.
pub(super) fn unorm(b: &[u32], tid: u32, result: usize, c: usize) -> f32 {
    f32::from_bits(b[tid as usize * WORDS + result + c]) * 255.0
}

pub(super) fn expect(b: &[u32], tid: u32, result: usize, expected: [u32; 4], what: &str) {
    for c in 0..4 {
        let v = unorm(b, tid, result, c);
        assert_eq!(v.round() as i64, expected[c] as i64, "{what}: thread {tid} component {c} is {v}, expected {}", expected[c]);
    }
}

// ---- ImagePlaneAddressOnLine ----

mod plane_on_line {
    use super::*;

    pub const SIZE: u32 = 16;
    pub const MIPS: u32 = 5;
    pub const SAMPLE_RESULT: usize = 8;
    pub const LOAD_RESULT: usize = 12;

    pub const CODE: [u32; 36] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c, 0x80000501, 0xe0301010, 0x80000601, 0xe0301014,
        0x80000701, 0xbf8c3f70, 0xf09c0f08, 0x00610c02, 0xf0001f08, 0x00011005, 0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028,
        0x80000e01, 0xe070102c, 0x80000f01, 0xe0701030, 0x80001001, 0xe0701034, 0x80001101, 0xe0701038, 0x80001201, 0xe070103c, 0x80001301, 0xbf810000,
    ];
    pub const EXPLICIT_LOD_CODE: [u32; 22] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c, 0x80000501, 0xbf8c3f70, 0xf0900f08, 0x00610c02,
        0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028, 0x80000e01, 0xe070102c, 0x80000f01, 0xbf810000,
    ];
    pub const ARRAY_EXPLICIT_LOD_CODE: [u32; 22] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c, 0x80000501, 0xbf8c3f70, 0xf0900f28, 0x00610c02,
        0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028, 0x80000e01, 0xe070102c, 0x80000f01, 0xbf810000,
    ];
    pub const ARRAY_LEVEL_ZERO_CODE: [u32; 22] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c, 0x80000501, 0xbf8c3f70, 0xf09c0f28, 0x00610c02,
        0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028, 0x80000e01, 0xe070102c, 0x80000f01, 0xbf810000,
    ];
    pub const LINE_LOD_CODE: [u32; 18] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301008, 0x80000301, 0xbf8c3f70, 0xf0900f00, 0x00610c02, 0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024,
        0x80000d01, 0xe0701028, 0x80000e01, 0xe070102c, 0x80000f01, 0xbf810000,
    ];

    pub const CLAMP_WRAP: u32 = 0;
    pub const CLAMP_EDGE: u32 = 2;
    pub const CLAMP_BORDER: u32 = 6;
    pub const FILTER_POINT: u32 = 0;
    pub const FILTER_LINEAR: u32 = 1;
    pub const MIP_POINT: u32 = 1;
    pub const MIP_LINEAR: u32 = 2;

    /// (name, clamp, filter, mip filter).
    pub const SAMPLERS: [(&str, u32, u32, u32); 5] = [
        ("point filtering, point mips, clamp to edge", CLAMP_EDGE, FILTER_POINT, MIP_POINT),
        ("point filtering, point mips, wrap", CLAMP_WRAP, FILTER_POINT, MIP_POINT),
        ("linear filtering, linear mips, clamp to edge", CLAMP_EDGE, FILTER_LINEAR, MIP_LINEAR),
        ("linear filtering, linear mips, wrap", CLAMP_WRAP, FILTER_LINEAR, MIP_LINEAR),
        ("linear filtering, linear mips, clamp to border", CLAMP_BORDER, FILTER_LINEAR, MIP_LINEAR),
    ];

    pub const LOD_U: [f32; 8] = [0.03125, 0.33125, 0.96875, 0.0, 1.0, -0.2, 1.3, 0.6];
    pub const LODS: [f32; 8] = [0.0, 1.0, 2.0, 1.25, 2.75, 3.6, 6.0, -1.0];
    pub const SLICES: [f32; 5] = [0.0, 1.0, 2.7, -1.0, 5.0];
    pub const LINE_V: [f32; 8] = [0.0, 0.5, 1.0, 1.5, -0.25, 3.0, 0.25, 0.75];
    pub const LINE_Y: [u32; 8] = [0, 1, 2, 3, 0xffff_ffff, 7, 0, 5];

    pub fn sampler((_, clamp, filter, mip): (&str, u32, u32, u32)) -> [u32; 4] {
        [clamp | (clamp << 3) | (clamp << 6), 0xFFF << 12, (filter << 20) | (filter << 22) | (mip << 26), 0]
    }

    pub fn texel_of(tid: u32) -> u32 {
        (tid * 7 + 3) % SIZE
    }

    pub fn input(line: bool) -> Vec<u32> {
        lanes(|tid| {
            let x = texel_of(tid);
            let u = bits((x as f32 + 0.5) / SIZE as f32);
            let v = if line { bits(LINE_V[tid as usize % 8]) } else { bits((((tid * 5 + 1) % SIZE) as f32 + 0.5) / SIZE as f32) };
            let y = if line { LINE_Y[tid as usize % 8] } else { (tid * 5 + 1) % SIZE };
            vec![u, v, 0, x, y, 0]
        })
    }

    pub fn mip_texel(level: u32, x: u32) -> [u32; 4] {
        [10 + 40 * level, x * 12 + 3, 200 - 30 * level, 255 - 7 * x - 11 * level]
    }

    pub fn mipped_line(g: &mut Guest) -> [u32; 8] {
        texture(g, Shape::new(sharps::image_type::TYPE_1D, SIZE, 1, 1, MIPS), |level, _, x, _| Some(mip_texel(level, x).iter().map(|&c| c as u8).collect()))
    }

    pub fn lane_u(tid: u32) -> f32 {
        LOD_U[((tid + tid / 8) % 8) as usize]
    }

    pub fn lane_lod(tid: u32) -> f32 {
        LODS[(tid / 4) as usize]
    }

    pub fn point_level(lod: f32) -> u32 {
        (lod.round() as i64).clamp(0, MIPS as i64 - 1) as u32
    }

    pub fn point_texel(u: f32, level: u32, clamp: u32) -> u32 {
        let width = (SIZE >> level) as i64;
        let t = (u * width as f32).floor() as i64;
        if clamp == CLAMP_WRAP {
            (t.rem_euclid(width)) as u32
        } else {
            t.clamp(0, width - 1) as u32
        }
    }

    pub fn sample_results(b: &[u32]) -> Vec<u32> {
        (0..THREADS).flat_map(|tid| b[tid as usize * WORDS + SAMPLE_RESULT..tid as usize * WORDS + SAMPLE_RESULT + 4].to_vec()).collect()
    }
}

#[test]
#[ignore]
fn image_dims_plane_address_on_line() {
    use plane_on_line::*;
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let line = texture(&mut g, Shape::new(sharps::image_type::TYPE_1D, SIZE, 1, 1, 1), |_, _, x, y| Some(vec![77, x as u8, y as u8, 255]));
    let b = run(&mut g, &mut rig, &CODE, &input(true), &line, point_sampler());
    for tid in 0..THREADS {
        let t = [77, texel_of(tid), 0, 255];
        expect(&b, tid, SAMPLE_RESULT, t, &format!("image_sample_lz 2d on a 1D texture with v {}", LINE_V[tid as usize % 8]));
        expect(&b, tid, LOAD_RESULT, t, &format!("image_load 2d on a 1D texture with y {}", LINE_Y[tid as usize % 8] as i32));
    }
    let plane = texture(&mut g, Shape::new(sharps::image_type::TYPE_2D, SIZE, SIZE, 1, 1), |_, _, x, y| Some(vec![77, x as u8, y as u8, 255]));
    let b = run(&mut g, &mut rig, &CODE, &input(false), &plane, point_sampler());
    for tid in 0..THREADS {
        let t = [77, texel_of(tid), (tid * 5 + 1) % SIZE, 255];
        expect(&b, tid, SAMPLE_RESULT, t, "image_sample_lz 2d on a 2D texture");
        expect(&b, tid, LOAD_RESULT, t, "image_load 2d on a 2D texture");
    }
}

/// Explicit-LOD 2D samples of a mipped 1D T# read what the 1D sample
/// returns (bit for bit), and the point-filtered texel of the rounded LOD.
#[test]
#[ignore]
fn image_dims_plane_address_on_line_explicit_lod() {
    use plane_on_line::*;
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let line = mipped_line(&mut g);
    let lod_input = || lanes(|tid| vec![bits(lane_u(tid)), bits(LINE_V[((tid / 2) % 8) as usize]), bits(lane_lod(tid)), 0]);
    for s in SAMPLERS {
        let words = sampler(s);
        let reference = sample_results(&run(&mut g, &mut rig, &LINE_LOD_CODE, &lod_input(), &line, words));
        let b = run(&mut g, &mut rig, &EXPLICIT_LOD_CODE, &lod_input(), &line, words);
        for tid in 0..THREADS {
            let (u, lod) = (lane_u(tid), lane_lod(tid));
            let what = format!("image_sample_l 2d on a 1D texture with {}, u {u}, v {}, LOD {lod}", s.0, LINE_V[((tid / 2) % 8) as usize]);
            for c in 0..4 {
                let value = b[tid as usize * WORDS + SAMPLE_RESULT + c];
                let expected = reference[tid as usize * 4 + c];
                assert_eq!(value, expected, "{what}: thread {tid} component {c} has bits {value:#x}, image_sample_l 1d returned bits {expected:#x}");
            }
            if s.2 == FILTER_POINT {
                let level = point_level(lod);
                expect(&b, tid, SAMPLE_RESULT, mip_texel(level, point_texel(u, level, s.1)), &what);
            } else if s.1 != CLAMP_BORDER {
                let mark = unorm(&b, tid, SAMPLE_RESULT, 0);
                let expected = 10.0 + 40.0 * lod.clamp(0.0, (MIPS - 1) as f32);
                assert!((mark - expected).abs() <= 1.0, "{what}: thread {tid} blends the levels to {mark}, expected {expected}");
            }
        }
    }
}

/// 2D-array samples (explicit LOD and level zero) of a mipped 1D T# read
/// what the 1D sample returns: v and the slice are ignored.
#[test]
#[ignore]
fn image_dims_plane_address_on_line_array() {
    use plane_on_line::*;
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let line = mipped_line(&mut g);
    for level_zero in [false, true] {
        let lod = |tid: u32| if level_zero { 0.0 } else { lane_lod(tid) };
        for s in SAMPLERS {
            let words = sampler(s);
            let line_input = lanes(|tid| vec![bits(lane_u(tid)), 0, bits(lod(tid)), 0]);
            let reference = sample_results(&run(&mut g, &mut rig, &LINE_LOD_CODE, &line_input, &line, words));
            let array_input = lanes(|tid| vec![bits(lane_u(tid)), bits(LINE_V[((tid / 2) % 8) as usize]), bits(SLICES[(tid % 5) as usize]), bits(lod(tid))]);
            let code: &[u32] = if level_zero { &ARRAY_LEVEL_ZERO_CODE } else { &ARRAY_EXPLICIT_LOD_CODE };
            let b = run(&mut g, &mut rig, code, &array_input, &line, words);
            for tid in 0..THREADS {
                let what = format!(
                    "{} 2d array on a 1D texture with {}, u {}, v {}, slice {}",
                    if level_zero { "image_sample_lz" } else { "image_sample_l" },
                    s.0,
                    lane_u(tid),
                    LINE_V[((tid / 2) % 8) as usize],
                    SLICES[(tid % 5) as usize]
                );
                for c in 0..4 {
                    let value = b[tid as usize * WORDS + SAMPLE_RESULT + c];
                    let expected = reference[tid as usize * 4 + c];
                    assert_eq!(value, expected, "{what}: thread {tid} component {c} has bits {value:#x}, image_sample_l 1d returned bits {expected:#x}");
                }
            }
        }
    }
}

// ---- ImageAddressDimension ----

mod address_dimension {
    use super::*;

    pub const SIZE: u32 = 16;
    pub const LEVELS: u32 = 5;
    pub const DEPTH: u32 = 2;
    pub const ARRAY_RESULT: usize = 8;
    pub const PLAIN_INPUT: usize = 16;
    pub const PLAIN_RESULT: usize = 24;

    pub const CODE: [u32; 71] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c, 0x80000501, 0xe0301010, 0x80000601, 0xe0301014,
        0x80000701, 0xe0301018, 0x80000801, 0xe030101c, 0x80000901, 0xe0301040, 0x80001001, 0xe0301044, 0x80001101, 0xe0301048, 0x80001201, 0xe030104c,
        0x80001301, 0xe0301050, 0x80001401, 0xe0301054, 0x80001501, 0xbf8c3f70, 0xf0900f28, 0x00610c02, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01,
        0xe0701028, 0x80000e01, 0xe070102c, 0x80000f01, 0xf0041f28, 0x00010c06, 0xe0701030, 0x80000c01, 0xe0701034, 0x80000d01, 0xe0701038, 0x80000e01,
        0xe070103c, 0x80000f01, 0xf0900f08, 0x00610c10, 0xe0701060, 0x80000c01, 0xe0701064, 0x80000d01, 0xe0701068, 0x80000e01, 0xe070106c, 0x80000f01,
        0xf0041f08, 0x00010c13, 0xe0701070, 0x80000c01, 0xe0701074, 0x80000d01, 0xe0701078, 0x80000e01, 0xe070107c, 0x80000f01, 0xbf810000,
    ];
    pub const VOLUME_CODE: [u32; 17] = [
        0x34020087, 0xe030104c, 0x80001301, 0xe0301050, 0x80001401, 0xbf8c3f70, 0xf0001f08, 0x00010c13, 0xe0701070, 0x80000c01, 0xe0701074, 0x80000d01,
        0xe0701078, 0x80000e01, 0xe070107c, 0x80000f01, 0xbf810000,
    ];
    pub const VOLUME_ARRAY_CODE: [u32; 19] = [
        0x34020087, 0xe030104c, 0x80001301, 0xe0301050, 0x80001401, 0xe0301054, 0x80001501, 0xbf8c3f70, 0xf0001f28, 0x00010c13, 0xe0701070, 0x80000c01,
        0xe0701074, 0x80000d01, 0xe0701078, 0x80000e01, 0xe070107c, 0x80000f01, 0xbf810000,
    ];
    pub const NARROW_CODE: [u32; 15] = [
        0x34020087, 0xe0301040, 0x80001001, 0xe0301048, 0x80001101, 0xbf8c3f70, 0xf0900f00, 0x00610c10, 0xe0701070, 0x80000c01, 0xe0701074, 0x80000d01,
        0xe0701078, 0x80000e01, 0xbf810000,
    ];

    /// (level, x, y) of a lane.
    pub fn sample_of(tid: u32) -> (u32, u32, u32) {
        (tid % LEVELS, (tid * 7 + 3) % SIZE, (tid * 5 + 1) % SIZE)
    }

    pub fn input() -> Vec<u32> {
        lanes(|tid| {
            let (level, x, y) = sample_of(tid);
            let u = bits((x as f32 + 0.5) / SIZE as f32);
            let v = bits((y as f32 + 0.5) / SIZE as f32);
            let lod = bits(level as f32);
            let mut w = vec![u, v, bits(0.0), lod, x >> level, y >> level, 0, level];
            w.extend_from_slice(&[POISON; 8]);
            w.extend_from_slice(&[u, v, lod, x >> level, y >> level, level]);
            w
        })
    }

    pub fn slice_marker(z: u32) -> u32 {
        z * 100 + 7
    }

    pub fn mipped_plane(g: &mut Guest) -> [u32; 8] {
        texture(g, Shape::new(sharps::image_type::TYPE_2D, SIZE, SIZE, 1, LEVELS), |level, _, x, y| Some(vec![level as u8, x as u8, y as u8, 255]))
    }

    pub fn volume(g: &mut Guest) -> [u32; 8] {
        texture(g, Shape::new(sharps::image_type::TYPE_3D, SIZE, SIZE, DEPTH, 1), |_, z, x, y| Some(vec![slice_marker(z) as u8, x as u8, y as u8, 255]))
    }
}

/// 2D-array samples and loads of a mipped 2D T# (slice 0), and the 2D ones.
#[test]
#[ignore]
fn image_dims_address_dimension() {
    use address_dimension::*;
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let plane = mipped_plane(&mut g);
    let b = run(&mut g, &mut rig, &CODE, &input(), &plane, point_sampler());
    let names = ["image_sample_l 2d_array", "image_load_mip 2d_array", "image_sample_l 2d", "image_load_mip 2d"];
    let results = [ARRAY_RESULT, ARRAY_RESULT + 4, PLAIN_RESULT, PLAIN_RESULT + 4];
    for tid in 0..THREADS {
        let (level, x, y) = sample_of(tid);
        for (name, result) in names.iter().zip(results) {
            expect(&b, tid, result, [level, x >> level, y >> level, 255], &format!("{name} on a 2D texture"));
        }
    }
}

/// 2D loads of a 3D T# read slice 0; 2D-array loads read the slice the
/// array index names.
#[test]
#[ignore]
fn image_dims_address_dimension_volume() {
    use address_dimension::*;
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let vol = volume(&mut g);
    let b = run(&mut g, &mut rig, &VOLUME_CODE, &input(), &vol, point_sampler());
    for tid in 0..THREADS {
        let (level, x, y) = sample_of(tid);
        expect(&b, tid, PLAIN_RESULT + 4, [slice_marker(0), x >> level, y >> level, 255], "image_load 2d on a 3D texture");
    }
    let mut array_input = input();
    for tid in 0..THREADS as usize {
        array_input[tid * WORDS + PLAIN_INPUT + 5] = tid as u32 % DEPTH;
    }
    let b = run(&mut g, &mut rig, &VOLUME_ARRAY_CODE, &array_input, &vol, point_sampler());
    for tid in 0..THREADS {
        let (level, x, y) = sample_of(tid);
        expect(&b, tid, PLAIN_RESULT + 4, [slice_marker(tid % DEPTH), x >> level, y >> level, 255], "image_load 2d_array on a 3D texture");
    }
}

/// 1D samples of a 2D T# read row 0 (point, and bilinear clamped).
#[test]
#[ignore]
fn image_dims_address_dimension_narrow() {
    use address_dimension::*;
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let plane = mipped_plane(&mut g);
    let b = run(&mut g, &mut rig, &NARROW_CODE, &input(), &plane, point_sampler());
    for tid in 0..THREADS {
        let (level, x, _) = sample_of(tid);
        let expected = [level, x >> level, 0];
        for c in 0..3 {
            let v = unorm(&b, tid, PLAIN_RESULT + 4, c);
            assert_eq!(v.round() as i64, expected[c] as i64, "image_sample_l 1d on a 2D texture: thread {tid} component {c} is {v}, expected {}", expected[c]);
        }
    }
    let mut bilinear = input();
    for tid in 0..THREADS {
        let (_, x, _) = sample_of(tid);
        bilinear[tid as usize * WORDS + PLAIN_INPUT] = bits((x as f32 + 0.5) / SIZE as f32);
        bilinear[tid as usize * WORDS + PLAIN_INPUT + 2] = bits(0.0);
    }
    let clamp_sampler = [2 << 3, 0xFFF << 12, (1 << 20) | (1 << 22) | (1 << 26), 0];
    let b = run(&mut g, &mut rig, &NARROW_CODE, &bilinear, &plane, clamp_sampler);
    for tid in 0..THREADS {
        let (_, x, _) = sample_of(tid);
        let expected = [0.0, x as f32, 0.0];
        for c in 0..3 {
            let v = unorm(&b, tid, PLAIN_RESULT + 4, c);
            assert!((v - expected[c]).abs() < 0.25, "image_sample_l 1d on a 2D texture, bilinear clamp: thread {tid} component {c} is {v}, expected {}", expected[c]);
        }
    }
}

// ---- ImageVolumeAddressOnLine / ImageVolumeAddressOnPlane ----

mod volume_on_lower {
    use super::*;

    pub const SIZE: u32 = 16;
    pub const DEPTH: u32 = 2;
    pub const SAMPLE_RESULT: usize = 8;
    pub const LOAD_RESULT: usize = 12;

    /// image_sample_lz 3D and image_load 3D.
    pub const CODE: [u32; 36] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c, 0x80000501, 0xe0301010, 0x80000601, 0xe0301014,
        0x80000701, 0xbf8c3f70, 0xf09c0f10, 0x00610c02, 0xf0001f10, 0x00011005, 0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028,
        0x80000e01, 0xe070102c, 0x80000f01, 0xe0701030, 0x80001001, 0xe0701034, 0x80001101, 0xe0701038, 0x80001201, 0xe070103c, 0x80001301, 0xbf810000,
    ];
    /// image_sample 3D (implicit LOD).
    pub const IMPLICIT_LOD_CODE: [u32; 26] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c, 0x80000501, 0xe0301010, 0x80000601, 0xe0301014,
        0x80000701, 0xbf8c3f70, 0xf0800f10, 0x00610c02, 0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028, 0x80000e01, 0xe070102c,
        0x80000f01, 0xbf810000,
    ];

    pub fn lane_of(tid: u32) -> (u32, u32) {
        ((tid * 7 + 3) % SIZE, (tid * 5 + 1) % SIZE)
    }

    pub fn slice_marker(z: u32) -> u32 {
        z * 100 + 7
    }

    /// The lanes' (u, v, r, x, y, z) through a lower-dimension T#.
    pub fn lower_input(v: impl Fn(u32) -> f32, r: impl Fn(u32) -> f32, y: impl Fn(u32) -> u32, z: impl Fn(u32) -> u32) -> Vec<u32> {
        lanes(|tid| {
            let (lx, _) = lane_of(tid);
            vec![bits((lx as f32 + 0.5) / SIZE as f32), bits(v(tid)), bits(r(tid)), lx, y(tid), z(tid)]
        })
    }

    /// A volume lane addresses (x, y, tid % 2).
    pub fn volume_input() -> Vec<u32> {
        lanes(|tid| {
            let (x, y) = lane_of(tid);
            let u = bits((x as f32 + 0.5) / SIZE as f32);
            let v = bits((y as f32 + 0.5) / SIZE as f32);
            let r = bits(((tid % DEPTH) as f32 + 0.5) / DEPTH as f32);
            vec![u, v, r, x, y, tid % DEPTH]
        })
    }

    pub fn filled(g: &mut Guest, ty: u32, height: u32, slices: u32) -> [u32; 8] {
        texture(g, Shape::new(ty, SIZE, height, slices, 1), |_, z, x, y| Some(vec![slice_marker(z) as u8, x as u8, y as u8, 255]))
    }

    pub fn check_volume(g: &mut Guest, rig: &mut Rig) {
        let vol = filled(g, sharps::image_type::TYPE_3D, SIZE, DEPTH);
        let b = run(g, rig, &CODE, &volume_input(), &vol, point_sampler());
        for tid in 0..THREADS {
            let (x, y) = lane_of(tid);
            let t = [slice_marker(tid % DEPTH), x, y, 255];
            expect(&b, tid, SAMPLE_RESULT, t, "image_sample_lz 3d on a 3D texture");
            expect(&b, tid, LOAD_RESULT, t, "image_load 3d on a 3D texture");
        }
    }
}

/// 3D samples of a 1D T# ignore v and r; 3D loads ignore y and read 0
/// unless z is 0.
#[test]
#[ignore]
fn image_dims_volume_address_on_line() {
    use volume_on_lower::*;
    const LINE_V: [f32; 8] = [0.0, 0.5, 1.0, 1.5, -0.25, 3.0, 0.25, 0.75];
    const LINE_R: [f32; 8] = LINE_V;
    const LINE_Y: [u32; 5] = [0, 1, 2, 3, 0xffff_ffff];
    const LINE_Z: [u32; 7] = [0, 1, 0, 2, 3, 0, 0xffff_ffff];
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let line = filled(&mut g, sharps::image_type::TYPE_1D, 1, 1);
    let line_input = || lower_input(|t| LINE_V[t as usize % 8], |t| LINE_R[(t as usize + 3) % 8], |t| LINE_Y[t as usize % 5], |t| LINE_Z[t as usize % 7]);
    let check_line = |b: &[u32], op: &str| {
        for tid in 0..THREADS {
            let t = [slice_marker(0), lane_of(tid).0, 0, 255];
            expect(b, tid, SAMPLE_RESULT, t, &format!("{op} 3d on a 1D texture with v {} and r {}", LINE_V[tid as usize % 8], LINE_R[(tid as usize + 3) % 8]));
        }
    };
    let b = run(&mut g, &mut rig, &CODE, &line_input(), &line, point_sampler());
    check_line(&b, "image_sample_lz");
    for tid in 0..THREADS {
        let z = LINE_Z[tid as usize % 7];
        let t = if z == 0 { [slice_marker(0), lane_of(tid).0, 0, 255] } else { [0; 4] };
        expect(&b, tid, LOAD_RESULT, t, &format!("image_load 3d on a 1D texture with y {} and z {}", LINE_Y[tid as usize % 5] as i32, z as i32));
    }
    let b = run(&mut g, &mut rig, &IMPLICIT_LOD_CODE, &line_input(), &line, point_sampler());
    check_line(&b, "image_sample");
    check_volume(&mut g, &mut rig);
}

/// 3D samples of a 2D T# ignore r; 3D loads read 0 unless z is 0.
#[test]
#[ignore]
fn image_dims_volume_address_on_plane() {
    use volume_on_lower::*;
    const PLANE_R: [f32; 8] = [0.0, 0.5, 1.0, 1.5, -0.25, 3.0, 0.25, 0.75];
    const PLANE_Z: [u32; 8] = [0, 1, 0, 2, 0xffff_ffff, 4, 0, 3];
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let plane = filled(&mut g, sharps::image_type::TYPE_2D, SIZE, 1);
    let plane_input = lower_input(|t| (lane_of(t).1 as f32 + 0.5) / SIZE as f32, |t| PLANE_R[t as usize % 8], |t| lane_of(t).1, |t| PLANE_Z[t as usize % 8]);
    let b = run(&mut g, &mut rig, &CODE, &plane_input, &plane, point_sampler());
    for tid in 0..THREADS {
        let (x, y) = lane_of(tid);
        let t = [slice_marker(0), x, y, 255];
        let z = PLANE_Z[tid as usize % 8];
        expect(&b, tid, SAMPLE_RESULT, t, &format!("image_sample_lz 3d on a 2D texture with r {}", PLANE_R[tid as usize % 8]));
        expect(&b, tid, LOAD_RESULT, if z == 0 { t } else { [0; 4] }, &format!("image_load 3d on a 2D texture with z {}", z as i32));
    }
    check_volume(&mut g, &mut rig);
}

// ---- ImageVolumeAddressPlanarTexture ----

/// 3D accesses of 2D and 2D-array T#s: samples read the layer r rounds to
/// (layer 0 of a 2D T#), loads and mip loads read 0 for a z past the
/// layers, resinfo reports depth 1 (the layers of an array).
#[test]
#[ignore]
fn image_dims_volume_address_planar_texture() {
    const SIZE: u32 = 16;
    const LAYERS: u32 = 2;
    const SAMPLE_RESULT: usize = 8;
    const LOAD_RESULT: usize = 12;
    const LOAD_MIP_RESULT: usize = 16;
    const RESINFO_RESULT: usize = 20;
    const CODE: [u32; 60] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c, 0x80000501, 0xe0301010, 0x80000601, 0xe0301014,
        0x80000701, 0xe0301018, 0x80000801, 0xe030101c, 0x80000901, 0xbf8c3f70, 0xf09c0f10, 0x00610c02, 0xf0001f10, 0x00011005, 0xf0041f10, 0x00011405,
        0xf0381f10, 0x00011809, 0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028, 0x80000e01, 0xe070102c, 0x80000f01, 0xe0701030,
        0x80001001, 0xe0701034, 0x80001101, 0xe0701038, 0x80001201, 0xe070103c, 0x80001301, 0xe0701040, 0x80001401, 0xe0701044, 0x80001501, 0xe0701048,
        0x80001601, 0xe070104c, 0x80001701, 0xe0701050, 0x80001801, 0xe0701054, 0x80001901, 0xe0701058, 0x80001a01, 0xe070105c, 0x80001b01, 0xbf810000,
    ];
    const PLANAR_R: [f32; 6] = [0.0, 0.5, 1.0, 1.5, -0.25, 3.0];
    const PLANAR_Z: [u32; 8] = [0, 1, 0, 4, 0, 0xffff_ffff, 0, 2];
    const ARRAY_R: [f32; 6] = [0.0, 1.0, 3.0, -0.25, 0.75, 1.2];
    const ARRAY_LAYER: [u32; 6] = [0, 1, 1, 0, 1, 1];
    // (x, y, r, z, sampled layer, load inside)
    let lane_of = |tid: u32, array: bool| {
        let (x, y) = ((tid * 7 + 3) % SIZE, (tid * 5 + 1) % SIZE);
        if array {
            (x, y, ARRAY_R[tid as usize % 6], tid % LAYERS, ARRAY_LAYER[tid as usize % 6], true)
        } else {
            let z = PLANAR_Z[tid as usize % 8];
            (x, y, PLANAR_R[tid as usize % 6], z, 0, z == 0)
        }
    };
    let marker = |layer: u32| layer * 100 + 7;
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    for array in [false, true] {
        let input = lanes(|tid| {
            let (x, y, r, z, _, _) = lane_of(tid, array);
            vec![bits((x as f32 + 0.5) / SIZE as f32), bits((y as f32 + 0.5) / SIZE as f32), bits(r), x, y, z, 0, 0]
        });
        let shape = if array { Shape::new(sharps::image_type::TYPE_2D_ARRAY, SIZE, SIZE, LAYERS, 1) } else { Shape::new(sharps::image_type::TYPE_2D, SIZE, SIZE, 1, 1) };
        let t = texture(&mut g, shape, |_, layer, x, y| Some(vec![marker(layer) as u8, x as u8, y as u8, 255]));
        let b = run(&mut g, &mut rig, &CODE, &input, &t, point_sampler());
        let name = if array { "a 2D-array texture" } else { "a 2D texture" };
        let resinfo = [SIZE, SIZE, if array { LAYERS } else { 1 }, 1];
        for tid in 0..THREADS {
            let (x, y, r, z, sampled_layer, inside) = lane_of(tid, array);
            let loaded = if inside { [marker(z), x, y, 255] } else { [0; 4] };
            expect(&b, tid, SAMPLE_RESULT, [marker(sampled_layer), x, y, 255], &format!("image_sample_lz 3d on {name} with r {r}"));
            expect(&b, tid, LOAD_RESULT, loaded, &format!("image_load 3d on {name} with z {}", z as i32));
            expect(&b, tid, LOAD_MIP_RESULT, loaded, &format!("image_load_mip 3d on {name} with z {}", z as i32));
            for c in 0..4 {
                let v = b[tid as usize * WORDS + RESINFO_RESULT + c];
                assert_eq!(v, resinfo[c], "image_get_resinfo 3d on {name}: thread {tid} component {c} is {v}, expected {}", resinfo[c]);
            }
        }
    }
}

// ---- ImageArraySampleOnVolume ----

/// 2D-array samples of a 3D T# sample the volume: r is the normalized z
/// coordinate (wrapped by the sampler), level 0 or an explicit LOD.
#[test]
#[ignore]
fn image_dims_array_sample_on_volume() {
    const SIZE: u32 = 16;
    const DEPTH: u32 = 4;
    const SAMPLE_RESULT: usize = 8;
    const LOAD_RESULT: usize = 12;
    const ARRAY_SAMPLE_CODE: [u32; 36] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c, 0x80000501, 0xe0301010, 0x80000601, 0xe0301014,
        0x80000701, 0xbf8c3f70, 0xf09c0f28, 0x00610c02, 0xf0001f10, 0x00011005, 0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028,
        0x80000e01, 0xe070102c, 0x80000f01, 0xe0701030, 0x80001001, 0xe0701034, 0x80001101, 0xe0701038, 0x80001201, 0xe070103c, 0x80001301, 0xbf810000,
    ];
    const VOLUME_SAMPLE_CODE: [u32; 36] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c, 0x80000501, 0xe0301010, 0x80000601, 0xe0301014,
        0x80000701, 0xbf8c3f70, 0xf09c0f10, 0x00610c02, 0xf0001f10, 0x00011005, 0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028,
        0x80000e01, 0xe070102c, 0x80000f01, 0xe0701030, 0x80001001, 0xe0701034, 0x80001101, 0xe0701038, 0x80001201, 0xe070103c, 0x80001301, 0xbf810000,
    ];
    const ARRAY_EXPLICIT_LOD_CODE: [u32; 22] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c, 0x80000501, 0xbf8c3f70, 0xf0900f28, 0x00610c02,
        0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028, 0x80000e01, 0xe070102c, 0x80000f01, 0xbf810000,
    ];
    const VOLUME_EXPLICIT_LOD_CODE: [u32; 22] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c, 0x80000501, 0xbf8c3f70, 0xf0900f10, 0x00610c02,
        0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028, 0x80000e01, 0xe070102c, 0x80000f01, 0xbf810000,
    ];
    // (x, y, z, r): r is z's centre, shifted by -1..2 whole wraps.
    let lane_of = |tid: u32| {
        let z = tid % DEPTH;
        let wraps = ((tid / DEPTH) % 4) as f32 - 1.0;
        ((tid * 7 + 3) % SIZE, (tid * 5 + 1) % SIZE, z, (z as f32 + 0.5) / DEPTH as f32 + wraps)
    };
    let input = |lod: Option<f32>| {
        lanes(|tid| {
            let (x, y, z, r) = lane_of(tid);
            let u = bits((x as f32 + 0.5) / SIZE as f32);
            let v = bits((y as f32 + 0.5) / SIZE as f32);
            vec![u, v, bits(r), lod.map_or(x, bits), y, z]
        })
    };
    let marker = |level: u32, z: u32| level * 25 + z * 50 + 7;
    let volume = |g: &mut Guest, levels: u32| {
        let shape = Shape::new(sharps::image_type::TYPE_3D, SIZE, SIZE, DEPTH, levels);
        texture(g, shape, |level, z, x, y| Some(vec![marker(level, z) as u8, x as u8, y as u8, 255]))
    };
    let check = |b: &[u32], level: u32, load: bool, what: &str| {
        for tid in 0..THREADS {
            let (x, y, z, r) = lane_of(tid);
            expect(b, tid, SAMPLE_RESULT, [marker(level, z >> level), x >> level, y >> level, 255], &format!("{what} (r {r})"));
            if load {
                expect(b, tid, LOAD_RESULT, [marker(0, z), x, y, 255], "image_load 3d on a 3D texture");
            }
        }
    };
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let t = volume(&mut g, 1);
    let b = run(&mut g, &mut rig, &VOLUME_SAMPLE_CODE, &input(None), &t, point_sampler());
    check(&b, 0, true, "image_sample_lz 3d on a 3D texture");
    let b = run(&mut g, &mut rig, &ARRAY_SAMPLE_CODE, &input(None), &t, point_sampler());
    check(&b, 0, true, "image_sample_lz 2d_array on a 3D texture");
    let b = run(&mut g, &mut rig, &ARRAY_EXPLICIT_LOD_CODE, &input(Some(0.0)), &t, point_sampler());
    check(&b, 0, false, "image_sample_l 2d_array on a 3D texture");
    let t = volume(&mut g, 2);
    let b = run(&mut g, &mut rig, &VOLUME_EXPLICIT_LOD_CODE, &input(Some(1.0)), &t, point_sampler());
    check(&b, 1, false, "image_sample_l 3d with LOD 1 on a two-level 3D texture");
    let b = run(&mut g, &mut rig, &ARRAY_EXPLICIT_LOD_CODE, &input(Some(1.0)), &t, point_sampler());
    check(&b, 1, false, "image_sample_l 2d_array with LOD 1 on a two-level 3D texture");
}

// ---- ImageOneDimensionalCompare ----

/// Depth-compare samples (LESS) of 16_UNORM 1D, 1D-array and 2D T#s:
/// level zero, implicit and explicit LOD, constant offsets; 1D samples of
/// a 1D array read its first layer.
#[test]
#[ignore]
fn image_dims_one_dimensional_compare() {
    const SIZE: u32 = 16;
    const LAYERS: u32 = 3;
    const REFERENCE_WORD: usize = 6;
    const RESULT: usize = 8;
    const STEP_1D: u32 = 4000;
    const STEP_ARRAY: u32 = 1200;
    const STEP_2D: u32 = 200;
    const ROW_2D: u32 = 2;
    const C1D_LZ: [u32; 13] = [0x34020087, 0xe0301000, 0x80000201, 0xe0301018, 0x80000801, 0xbf8c3f70, 0x7e280308, 0x7e2a0302, 0xf0bc0100, 0x00610c14, 0xbf8c3f70, 0xe0701020, 0x80000c01];
    const C1D_IMPLICIT: [u32; 13] = [0x34020087, 0xe0301000, 0x80000201, 0xe0301018, 0x80000801, 0xbf8c3f70, 0x7e280308, 0x7e2a0302, 0xf0a00100, 0x00610c14, 0xbf8c3f70, 0xe0701020, 0x80000c01];
    const C1D_EXPLICIT: [u32; 14] =
        [0x34020087, 0xe0301000, 0x80000201, 0xe0301018, 0x80000801, 0xbf8c3f70, 0x7e280308, 0x7e2a0302, 0x7e2c0280, 0xf0b00100, 0x00610c14, 0xbf8c3f70, 0xe0701020, 0x80000c01];
    const C1D_LZ_OFFSET: [u32; 15] =
        [0x34020087, 0xe0301000, 0x80000201, 0xe0301018, 0x80000801, 0xbf8c3f70, 0x7e2802ff, 0x00000f00, 0x7e2a0308, 0x7e2c0302, 0xf0fc0100, 0x00610c14, 0xbf8c3f70, 0xe0701020, 0x80000c01];
    const C1DA_LZ_OFFSET: [u32; 18] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301018, 0x80000801, 0xbf8c3f70, 0x7e2802ff, 0x00000f00, 0x7e2a0308, 0x7e2c0302, 0x7e2e0303,
        0xf0fc0120, 0x00610c14, 0xbf8c3f70, 0xe0701020, 0x80000c01,
    ];
    const C1DA_LZ: [u32; 16] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301018, 0x80000801, 0xbf8c3f70, 0x7e280308, 0x7e2a0302, 0x7e2c0303, 0xf0bc0120, 0x00610c14,
        0xbf8c3f70, 0xe0701020, 0x80000c01,
    ];
    const C1DA_IMPLICIT: [u32; 16] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301018, 0x80000801, 0xbf8c3f70, 0x7e280308, 0x7e2a0302, 0x7e2c0303, 0xf0a00120, 0x00610c14,
        0xbf8c3f70, 0xe0701020, 0x80000c01,
    ];
    const C2D_LZ: [u32; 16] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301018, 0x80000801, 0xbf8c3f70, 0x7e280308, 0x7e2a0302, 0x7e2c0303, 0xf0bc0108, 0x00610c14,
        0xbf8c3f70, 0xe0701020, 0x80000c01,
    ];
    const C2D_IMPLICIT: [u32; 16] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301018, 0x80000801, 0xbf8c3f70, 0x7e280308, 0x7e2a0302, 0x7e2c0303, 0xf0a00108, 0x00610c14,
        0xbf8c3f70, 0xe0701020, 0x80000c01,
    ];
    let end = |code: &[u32]| [code, &[0xbf810000u32][..]].concat();
    let texel_of = |tid: u32| (tid * 7 + 3) % SIZE;
    let below = |tid: u32| tid & 1 == 0;
    let u = |x: u32| bits((x as f32 + 0.5) / SIZE as f32);
    // (coordinate words, the texel's 16-bit value, the step).
    let input = |coords: &dyn Fn(u32) -> (Vec<u32>, u32), step: u32| {
        lanes(|tid| {
            let (words, value) = coords(tid);
            let texel = value as f32 / 65535.0;
            let half = (step / 2) as f32 / 65535.0;
            let mut w = vec![POISON; RESULT];
            w[..words.len()].copy_from_slice(&words);
            w[REFERENCE_WORD] = bits(if below(tid) { texel - half } else { texel + half });
            w
        })
    };
    let line_input = input(&|tid| (vec![u(texel_of(tid))], (texel_of(tid) + 1) * STEP_1D), STEP_1D);
    let array_input = input(
        &|tid| {
            let layer = (tid / 7) % LAYERS;
            (vec![u(texel_of(tid)), bits(layer as f32)], (layer * SIZE + texel_of(tid) + 1) * STEP_ARRAY)
        },
        STEP_ARRAY,
    );
    let first_layer_input = input(&|tid| (vec![u(texel_of(tid))], (texel_of(tid) + 1) * STEP_ARRAY), STEP_ARRAY);
    let plane_input = input(&|tid| (vec![u(texel_of(tid)), bits((ROW_2D as f32 + 0.5) / SIZE as f32)], (ROW_2D * SIZE + texel_of(tid) + 1) * STEP_2D), STEP_2D);
    let sixteen = |v: u32| (v as u16).to_le_bytes().to_vec();
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let shape = |ty, height, slices| Shape { format: 7, bpe: 2, ..Shape::new(ty, SIZE, height, slices, 1) };
    let line = texture(&mut g, shape(sharps::image_type::TYPE_1D, 1, 1), |_, _, x, _| Some(sixteen((x + 1) * STEP_1D)));
    let array = texture(&mut g, shape(sharps::image_type::TYPE_1D_ARRAY, 1, LAYERS), |_, layer, x, _| Some(sixteen((layer * SIZE + x + 1) * STEP_ARRAY)));
    let plane = texture(&mut g, shape(sharps::image_type::TYPE_2D, SIZE, 1), |_, _, x, y| Some(sixteen((y * SIZE + x + 1) * STEP_2D)));
    // LESS, point filtering, wrap.
    let sampler = [1 << 12, 0xFFF << 12, 1 << 26, 0];
    let cases: [(&str, &[u32], &Vec<u32>, &[u32; 8]); 11] = [
        ("1D image_sample_c_lz", &C1D_LZ, &line_input, &line),
        ("1D image_sample_c", &C1D_IMPLICIT, &line_input, &line),
        ("1D image_sample_c_l", &C1D_EXPLICIT, &line_input, &line),
        ("1D image_sample_c_lz_o", &C1D_LZ_OFFSET, &line_input, &line),
        ("1D array image_sample_c_lz", &C1DA_LZ, &array_input, &array),
        ("1D array image_sample_c_lz_o", &C1DA_LZ_OFFSET, &array_input, &array),
        ("1D array image_sample_c", &C1DA_IMPLICIT, &array_input, &array),
        ("1D image_sample_c_lz on a 1D array's first layer", &C1D_LZ, &first_layer_input, &array),
        ("1D image_sample_c on a 1D array's first layer", &C1D_IMPLICIT, &first_layer_input, &array),
        ("2D image_sample_c_lz", &C2D_LZ, &plane_input, &plane),
        ("2D image_sample_c", &C2D_IMPLICIT, &plane_input, &plane),
    ];
    for (name, code, input, t) in cases {
        let b = run(&mut g, &mut rig, &end(code), input, t, sampler);
        for tid in 0..THREADS {
            let v = f32::from_bits(b[tid as usize * WORDS + RESULT]);
            let e = if below(tid) { 1.0 } else { 0.0 };
            assert_eq!(v, e, "{name}: thread {tid} is {v}, expected {e}");
        }
    }
}

// ---- ImagePlaneAddressOnCube ----

/// 2D samples (level zero, explicit and implicit LOD) and loads of a cube
/// T# read face 0.
#[test]
#[ignore]
fn image_dims_plane_address_on_cube() {
    const SIZE: u32 = 8;
    const FACES: u32 = 6;
    const RESULT: usize = 8;
    const CODE: [u32; 30] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c, 0x80000501, 0xe0301010, 0x80000601, 0xbf8c3f70,
        0xf09c0108, 0x00610c02, 0xf0900108, 0x00610d02, 0xf0800108, 0x00610e02, 0xf0001108, 0x00010f05, 0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024,
        0x80000d01, 0xe0701028, 0x80000e01, 0xe070102c, 0x80000f01, 0xbf810000,
    ];
    // (x, y, load x, load y)
    let lane_of = |tid: u32| (tid % SIZE, (tid * 3) % SIZE, (tid * 5) % SIZE, (tid * 7 + 1) % SIZE);
    let value = |face: u32, x: u32, y: u32| (face * 1000 + y * 16 + x) as f32;
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let shape = Shape { format: 22, ..Shape::new(sharps::image_type::CUBE, SIZE, SIZE, FACES, 1) };
    let cube = texture(&mut g, shape, |_, face, x, y| Some(value(face, x, y).to_le_bytes().to_vec()));
    let input = lanes(|tid| {
        let (x, y, lx, ly) = lane_of(tid);
        let u = 1.0 + (x as f32 + 0.5) / SIZE as f32;
        let v = 1.0 + (y as f32 + 0.5) / SIZE as f32;
        vec![bits(u), bits(v), 0, lx, ly]
    });
    let b = run(&mut g, &mut rig, &CODE, &input, &cube, [0, 0xFFF << 12, 0, 0]);
    let names = ["image_sample_lz", "image_sample_l", "image_sample", "image_load"];
    for tid in 0..THREADS {
        let (x, y, lx, ly) = lane_of(tid);
        let expected = [value(0, x, y), value(0, x, y), value(0, x, y), value(0, lx, ly)];
        for (op, name) in names.iter().enumerate() {
            let v = f32::from_bits(b[tid as usize * WORDS + RESULT + op]);
            assert_eq!(v, expected[op], "{name} 2d on a cube texture, thread {tid} is {v}, expected {}", expected[op]);
        }
    }
}

// ---- ImageOneDimensionalArray, ImageGatherOneDimensional(Array) ----

/// The reference's second layout: s[0:3] the input (4 dwords a lane),
/// s[4:7] the output (`out_words` a lane), s[8:15] the T#, s[16:19] the S#.
/// A compute dispatch has 16 user SGPRs: the program starts by moving the
/// S# into s[16:19] (`s_mov_b32` of literals).
pub(super) fn run_split(g: &mut Guest, rig: &mut Rig, code: &[u32], input: &[u32], out_words: usize, texture: &[u32; 8], sampler: [u32; 4]) -> Vec<u32> {
    run_split_sgprs(g, rig, code, input, out_words, texture, &sampler)
}

/// [`run_split`] with `high` the SGPRs from s16 on (moved there by the
/// program's first instructions).
pub(super) fn run_split_sgprs(g: &mut Guest, rig: &mut Rig, code: &[u32], input: &[u32], out_words: usize, texture: &[u32; 8], high: &[u32]) -> Vec<u32> {
    let ib = g.alloc(&asm::bytes(input));
    let words = THREADS as usize * out_words;
    let ob = g.alloc(&asm::bytes(&vec![(-1.0f32).to_bits(); words]));
    let mut user = raw(ib, (input.len() * 4) as u32).to_vec();
    user.extend(raw(ob, (words * 4) as u32));
    user.extend_from_slice(texture);
    let mut program: Vec<u32> = high.iter().enumerate().flat_map(|(k, &w)| [0xBE80_03FF | ((16 + k as u32) << 16), w]).collect();
    program.extend_from_slice(code);
    dispatch(g, rig, &program, &user);
    read_words(g, rig, ob, words)
}

/// An exact n/255 value of the output.
pub(super) fn expect_exact(out: &[u32], index: usize, expected: u32, what: &str) {
    let scaled = f32::from_bits(out[index]) * 255.0;
    assert!((scaled - scaled.round()).abs() < 1e-3 && scaled.round() as i64 == expected as i64, "{what} is {scaled}/255, expected {expected}/255");
}

pub(super) const CLAMP_WRAP: u32 = 0;
pub(super) const CLAMP_EDGE: u32 = 2;

/// Sample (level zero) and load of a 1D-array T#: the layer rounds to
/// nearest even and clamps.
#[test]
#[ignore]
fn image_dims_one_dimensional_array() {
    const WIDTH: u32 = 8;
    const LAYERS: u32 = 3;
    const CODE: [u32; 15] = [
        0x34020084, 0xe0381000, 0x80000201, 0xbf8c3f70, 0xf09c0f20, 0x00820602, 0xf0001f20, 0x00020a04, 0x34020085, 0xbf8c3f70, 0xe0781000, 0x80010601,
        0xe0781010, 0x80010a01, 0xbf810000,
    ];
    const LAYER_INPUT: [f32; 8] = [0.0, 1.0, 2.0, 0.4, 1.6, 2.4, -1.0, 5.0];
    let texel = |x: u32, layer: u32| [((layer * WIDTH + x) * 9 + 3) as u8, (x * 30) as u8, (layer * 100) as u8, 0xFF];
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let t = texture(&mut g, Shape::new(sharps::image_type::TYPE_1D_ARRAY, WIDTH, 1, LAYERS, 1), |_, layer, x, _| Some(texel(x, layer).to_vec()));
    let input: Vec<u32> =
        (0..THREADS).flat_map(|tid| [bits(-0.2 + 0.045 * tid as f32), bits(LAYER_INPUT[tid as usize % 8]), tid % WIDTH, (tid / WIDTH) % LAYERS]).collect();
    let out = run_split(&mut g, &mut rig, &CODE, &input, 8, &t, [CLAMP_EDGE | (CLAMP_EDGE << 3) | (CLAMP_EDGE << 6), 0, 0, 0]);
    for tid in 0..THREADS as usize {
        let u = f32::from_bits(input[tid * 4]);
        let layer = f32::from_bits(input[tid * 4 + 1]).round_ties_even().clamp(0.0, (LAYERS - 1) as f32) as u32;
        let x = ((u * WIDTH as f32).floor() as i32).clamp(0, WIDTH as i32 - 1) as u32;
        let sampled = texel(x, layer);
        let loaded = texel(input[tid * 4 + 2], input[tid * 4 + 3]);
        for c in 0..4 {
            expect_exact(&out, tid * 8 + c, sampled[c] as u32, &format!("image_sample_lz on a 1D array: thread {tid} component {c}"));
            expect_exact(&out, tid * 8 + 4 + c, loaded[c] as u32, &format!("image_load on a 1D array: thread {tid} component {c}"));
        }
    }
}

/// image_gather4 of 1D and 1D-array T#s (with and without offsets, clamp
/// to edge and wrap): the two footprint texels of the row as
/// (left, right, right, left).
fn gather_one_dimensional(array: bool) {
    const WIDTH: u32 = 8;
    const LAYERS: u32 = 3;
    const GATHER_1D: [u32; 10] = [0x34020084, 0xe0381000, 0x80000201, 0xbf8c3f70, 0xf11c0100, 0x00820603, 0xbf8c3f70, 0xe0781000, 0x80010601, 0xbf810000];
    const GATHER_1D_OFFSET: [u32; 10] = [0x34020084, 0xe0381000, 0x80000201, 0xbf8c3f70, 0xf15c0100, 0x00820602, 0xbf8c3f70, 0xe0781000, 0x80010601, 0xbf810000];
    const GATHER_1D_ARRAY: [u32; 10] = [0x34020084, 0xe0381000, 0x80000201, 0xbf8c3f70, 0xf11c0120, 0x00820603, 0xbf8c3f70, 0xe0781000, 0x80010601, 0xbf810000];
    const GATHER_1D_ARRAY_OFFSET: [u32; 10] =
        [0x34020084, 0xe0381000, 0x80000201, 0xbf8c3f70, 0xf15c0120, 0x00820602, 0xbf8c3f70, 0xe0781000, 0x80010601, 0xbf810000];
    const LAYER_INPUT: [f32; 8] = [0.0, 1.0, 2.0, 0.4, 1.6, 2.4, -1.0, 5.0];
    let red = |layer: u32, x: u32| if array { ((layer * WIDTH + x) * 10 + 5) as u8 } else { (x * 31 + 5) as u8 };
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let shape = if array { Shape::new(sharps::image_type::TYPE_1D_ARRAY, WIDTH, 1, LAYERS, 1) } else { Shape::new(sharps::image_type::TYPE_1D, WIDTH, 1, 1, 1) };
    let t = texture(&mut g, shape, |_, layer, x, _| Some(vec![red(layer, x), 0x11, 0x22, 0xFF]));
    let input: Vec<u32> = (0..THREADS)
        .flat_map(|tid| {
            let offset = (tid % 11) as i32 - 5;
            let first = (offset as u32 & 0x3F) | if tid % 3 == 0 { 0x3F00 } else { 0 };
            let layer = if array { bits(LAYER_INPUT[tid as usize % 8]) } else { 0 };
            [first, bits(-0.3 + 0.05 * tid as f32), layer, 0]
        })
        .collect();
    let variants: [(&[u32], bool, u32, &str); 4] = if array {
        [
            (&GATHER_1D_ARRAY, false, CLAMP_EDGE, "1D array gather, clamp to edge"),
            (&GATHER_1D_ARRAY, false, CLAMP_WRAP, "1D array gather, wrap"),
            (&GATHER_1D_ARRAY_OFFSET, true, CLAMP_EDGE, "1D array gather with offsets, clamp to edge"),
            (&GATHER_1D_ARRAY_OFFSET, true, CLAMP_WRAP, "1D array gather with offsets, wrap"),
        ]
    } else {
        [
            (&GATHER_1D, false, CLAMP_EDGE, "1D gather, clamp to edge"),
            (&GATHER_1D, false, CLAMP_WRAP, "1D gather, wrap"),
            (&GATHER_1D_OFFSET, true, CLAMP_EDGE, "1D gather with offsets, clamp to edge"),
            (&GATHER_1D_OFFSET, true, CLAMP_WRAP, "1D gather with offsets, wrap"),
        ]
    };
    for (code, offset, clamp, name) in variants {
        let out = run_split(&mut g, &mut rig, code, &input, 4, &t, [clamp | (CLAMP_EDGE << 3) | (CLAMP_EDGE << 6), 0, 0, 0]);
        for tid in 0..THREADS as usize {
            let u = f32::from_bits(input[tid * 4 + 1]);
            let mut left = (u * WIDTH as f32 - 0.5).floor() as i32;
            if offset {
                let field = input[tid * 4] & 0x3F;
                left += (field ^ 0x20) as i32 - 0x20;
            }
            let layer = if array { f32::from_bits(input[tid * 4 + 2]).round_ties_even().clamp(0.0, (LAYERS - 1) as f32) as u32 } else { 0 };
            let texel = |x: i32| {
                let a = if clamp == CLAMP_EDGE { x.clamp(0, WIDTH as i32 - 1) } else { x.rem_euclid(WIDTH as i32) };
                red(layer, a as u32) as u32
            };
            let (first, second) = (texel(left), texel(left + 1));
            for (c, e) in [first, second, second, first].into_iter().enumerate() {
                expect_exact(&out, tid * 4 + c, e, &format!("{name}: thread {tid} component {c}"));
            }
        }
    }
}

#[test]
#[ignore]
fn image_dims_gather_one_dimensional() {
    gather_one_dimensional(false);
}

#[test]
#[ignore]
fn image_dims_gather_one_dimensional_array() {
    gather_one_dimensional(true);
}

// ---- ImagePlaneCompareOnLine ----

/// 2D depth-compare samples (level zero and implicit LOD) of a 16-bit
/// UNORM 1D T# compare against the texel u addresses, whatever v.
#[test]
#[ignore]
fn image_dims_plane_compare_on_line() {
    const SIZE: u32 = 16;
    const REFERENCE_WORD: usize = 6;
    const RESULT: usize = 8;
    const STEP: u32 = 4000;
    const COMPARE_LESS: u32 = 1;
    const LEVEL_ZERO_CODE: [u32; 17] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301018, 0x80000801, 0xbf8c3f70, 0x7e280308, 0x7e2a0302, 0x7e2c0303, 0xf0bc0108,
        0x00610c14, 0xbf8c3f70, 0xe0701020, 0x80000c01, 0xbf810000,
    ];
    const IMPLICIT_LOD_CODE: [u32; 17] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301018, 0x80000801, 0xbf8c3f70, 0x7e280308, 0x7e2a0302, 0x7e2c0303, 0xf0a00108,
        0x00610c14, 0xbf8c3f70, 0xe0701020, 0x80000c01, 0xbf810000,
    ];
    const LINE_V: [f32; 8] = [0.0, 0.5, 1.0, 1.5, -0.25, 3.0, 0.25, 0.75];
    let texel_of = |tid: u32| (tid * 7 + 3) % SIZE;
    let below = |tid: u32| tid & 1 == 0;
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let shape = Shape { format: 7, bpe: 2, ..Shape::new(sharps::image_type::TYPE_1D, SIZE, 1, 1, 1) };
    let line = texture(&mut g, shape, |_, _, x, _| Some((((x + 1) * STEP) as u16).to_le_bytes().to_vec()));
    let input = || {
        lanes(|tid| {
            let x = texel_of(tid);
            let texel = ((x + 1) * STEP) as f32 / 65535.0;
            let half = (STEP / 2) as f32 / 65535.0;
            let mut w = vec![bits((x as f32 + 0.5) / SIZE as f32), bits(LINE_V[tid as usize % 8]), POISON, POISON, POISON, POISON];
            w.push(bits(if below(tid) { texel - half } else { texel + half }));
            assert_eq!(w.len(), REFERENCE_WORD + 1);
            w
        })
    };
    for (code, op) in [(&LEVEL_ZERO_CODE, "image_sample_c_lz"), (&IMPLICIT_LOD_CODE, "image_sample_c")] {
        let b = run(&mut g, &mut rig, code, &input(), &line, [COMPARE_LESS << 12, 0xFFF << 12, 1 << 26, 0]);
        for tid in 0..THREADS {
            let v = f32::from_bits(b[tid as usize * WORDS + RESULT]);
            let expected = if below(tid) { 1.0 } else { 0.0 };
            assert_eq!(v, expected, "{op} 2d on a 1D texture with v {}: thread {tid} is {v}, expected {expected}", LINE_V[tid as usize % 8]);
        }
    }
}

// ---- ImageArrayCompareOnColour ----

/// image_sample_c_lz (DIM 2D_ARRAY): v20 reference, v21 u, v22 v, v23
/// slice, read from the lane's words 6, 0, 1 and 2; the result to word 8.
const ARRAY_COMPARE_LZ_CODE: [u32; 20] = [
    0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe0301018, 0x80000801, 0xbf8c3f70, 0x7e280308, 0x7e2a0302,
    0x7e2c0303, 0x7e2e0304, 0xf0bc0128, 0x00610c14, 0xbf8c3f70, 0xe0701020, 0x80000c01, 0xbf810000,
];

/// Depth-compare samples of colour array textures (compared in the
/// shader): the layer is the slice coordinate rounded half to even, as the
/// host sampler selects it (0.49999997 and 0.5 are layer 0, 2.5 is layer
/// 2), and NOT_EQUAL passes against a NaN reference.
#[test]
#[ignore]
fn image_dims_array_compare_on_colour() {
    const SIZE: u32 = 8;
    const LAYERS: u32 = 4;
    const RESULT: usize = 8;
    const COMPARE_LESS: u32 = 1;
    const COMPARE_NOT_EQUAL: u32 = 5;
    // 16_FLOAT has no depth view: its compares are evaluated in the shader.
    const FORMAT_16_FLOAT: u32 = 13;
    const HALF: [u16; 4] = [0x3C00, 0x4000, 0x4200, 0x4400];
    let slices: [(f32, u32); 8] = [(0.49999997, 0), (0.5, 0), (1.5, 2), (2.5, 2), (0.0, 0), (1.4, 1), (2.6, 3), (3.5, 3)];
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let shape = Shape { format: FORMAT_16_FLOAT, bpe: 2, ..Shape::new(sharps::image_type::TYPE_2D_ARRAY, SIZE, SIZE, LAYERS, 1) };
    // Layer n holds n + 1.
    let layered = texture(&mut g, shape, |_, layer, _, _| Some(HALF[layer as usize].to_le_bytes().to_vec()));
    // LESS against the selected layer's value plus 0.5: 0 on that layer,
    // 1 on the next one.
    let input = lanes(|tid| {
        let (slice, layer) = slices[tid as usize % 8];
        vec![bits(0.5), bits(0.5), bits(slice), POISON, POISON, POISON, bits(layer as f32 + 1.5)]
    });
    let b = run(&mut g, &mut rig, &ARRAY_COMPARE_LZ_CODE, &input, &layered, [COMPARE_LESS << 12, 0xFFF << 12, 1 << 26, 0]);
    for tid in 0..THREADS {
        let (slice, layer) = slices[tid as usize % 8];
        let v = f32::from_bits(b[tid as usize * WORDS + RESULT]);
        assert_eq!(v, 0.0, "slice {slice} selects layer {layer}: thread {tid} is {v}");
    }
    let input = lanes(|tid| vec![bits(0.5), bits(0.5), bits(0.0), POISON, POISON, POISON, if tid % 2 == 0 { bits(f32::NAN) } else { bits(1.0) }]);
    let b = run(&mut g, &mut rig, &ARRAY_COMPARE_LZ_CODE, &input, &layered, [COMPARE_NOT_EQUAL << 12, 0xFFF << 12, 1 << 26, 0]);
    for tid in 0..THREADS {
        let v = f32::from_bits(b[tid as usize * WORDS + RESULT]);
        let expected = if tid % 2 == 0 { 1.0 } else { 0.0 };
        assert_eq!(v, expected, "NOT_EQUAL with reference {}: thread {tid} is {v}", if tid % 2 == 0 { "NaN" } else { "equal" });
    }
}

// ---- ImageStoreArrayThroughVolume, ImageStoreLineThroughPlane ----

/// The bytes of a storage texture after the dispatch wrote it.
pub(super) fn written_back(g: &Guest, rig: &mut Rig, base: u64, size: u64) -> Vec<u8> {
    rig.images.writeback_range(&mut rig.sched, &mut rig.buffers, base, size);
    rig.buffers.download(&mut rig.sched, base, size);
    let mut out = vec![0u8; size as usize];
    assert!(g.mem.read_backing(base, &mut out));
    out
}

/// A storage dispatch: s[0:3] the input (4 dwords a lane), s[4:11] the T#.
pub(super) fn store_dispatch(g: &mut Guest, rig: &mut Rig, code: &[u32], input: &[u32], texture: &[u32; 8]) {
    let ib = g.alloc(&asm::bytes(input));
    let mut user = raw(ib, (input.len() * 4) as u32).to_vec();
    user.extend_from_slice(texture);
    dispatch(g, rig, code, &user);
}

pub(super) fn initial_bytes(len: usize) -> Vec<u8> {
    (0..len).map(|i| (0x80 + 37 * i + 11 * (i >> 4)) as u8).collect()
}

/// 2D-array stores through a 3D T# write the slice the array index names;
/// a slice past the depth writes nothing.
#[test]
#[ignore]
fn image_dims_store_array_through_volume() {
    const SIZE: u32 = 4;
    const DEPTH: u32 = 2;
    const TEXEL_BYTES: usize = 16384;
    const STORE_CODE: [u32; 10] = [0x34020084, 0xe0381000, 0x80001401, 0xbf8c3f70, 0xf0201128, 0x00011714, 0x7e2c0282, 0xf0201128, 0x00011714, 0xbf810000];
    let value = |tid: u32| (tid + 1).wrapping_mul(0x9e37_79b1);
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let shape = Shape { format: 20, ..Shape::new(sharps::image_type::TYPE_3D, SIZE, SIZE, DEPTH, 1) };
    let layout = shape.layout();
    assert!(layout.guest_bytes as usize <= TEXEL_BYTES);
    let base = g.alloc(&initial_bytes(TEXEL_BYTES));
    let input: Vec<u32> = (0..THREADS).flat_map(|tid| [tid % SIZE, (tid / SIZE) % SIZE, tid / (SIZE * SIZE), value(tid)]).collect();
    store_dispatch(&mut g, &mut rig, &STORE_CODE, &input, &shape.descriptor(base));
    let bytes = written_back(&g, &mut rig, base, TEXEL_BYTES as u64);
    for tid in 0..THREADS {
        let (x, y, z) = (tid % SIZE, (tid / SIZE) % SIZE, tid / (SIZE * SIZE));
        let o = layout.element_offset(0, z, x, y).unwrap() as usize;
        let actual = u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        assert_eq!(actual, value(tid), "image_store 2d_array through a 3D T#: texel ({x}, {y}, {z}) is {actual:#010x}, expected {:#010x}", value(tid));
    }
    let end = layout.guest_bytes as usize;
    assert!(bytes[end..] == initial_bytes(TEXEL_BYTES)[end..], "image_store 2d_array through a 3D T#: a slice past the volume changed bytes after it");
}

/// 2D stores through a 1D T# ignore y: the second store (y = 1, DMASK
/// 0x5) overwrites every texel, its unselected components 0.
#[test]
#[ignore]
fn image_dims_store_line_through_plane() {
    const WIDTH: u32 = 32;
    const TEXEL_BYTES: usize = 4096;
    const STORE_CODE: [u32; 12] =
        [0x34020084, 0xe0381000, 0x80001401, 0xbf8c3f70, 0x7e3c0300, 0x7e3e0280, 0xf0201f08, 0x0001141e, 0x7e3e0281, 0xf0201508, 0x0001141e, 0xbf810000];
    let value = |tid: u32, index: u32| (tid * 4 + index + 1).wrapping_mul(0x9e37_79b1);
    let second = |x: u32, c: u32| -> u32 {
        const DMASK: u32 = 0x5;
        if (DMASK >> c) & 1 == 0 {
            return 0;
        }
        value(x, (0..c).map(|l| (DMASK >> l) & 1).sum())
    };
    for (name, format, components) in [("32_SINT", 21u32, 1u32), ("32_32_32_32_SINT", 76, 4)] {
        let mut g = Guest::new(16 << 20);
        let mut rig = rig(&g);
        let shape = Shape { format, bpe: components * 4, ..Shape::new(sharps::image_type::TYPE_1D, WIDTH, 1, 1, 1) };
        let layout = shape.layout();
        let base = g.alloc(&initial_bytes(TEXEL_BYTES));
        let input: Vec<u32> = (0..THREADS).flat_map(|tid| (0..4).map(move |i| value(tid, i))).collect();
        store_dispatch(&mut g, &mut rig, &STORE_CODE, &input, &shape.descriptor(base));
        let bytes = written_back(&g, &mut rig, base, TEXEL_BYTES as u64);
        for x in 0..WIDTH {
            let o = layout.element_offset(0, 0, x, 0).unwrap() as usize;
            for c in 0..components {
                let at = o + c as usize * 4;
                let actual = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
                let wanted = second(x, c);
                assert_eq!(actual, wanted, "image_store 2d through a 1D {name} T#: texel {x} component {c} is {actual:#010x}, expected the y = 1 store with dmask 0x5 {wanted:#010x}");
            }
        }
        let end = layout.element_offset(0, 0, WIDTH - 1, 0).unwrap() as usize + (components * 4) as usize;
        assert!(bytes[end..] == initial_bytes(TEXEL_BYTES)[end..], "image_store 2d through a 1D {name} T#: bytes past the texture changed");
    }
}

/// 2D-array stores through a 1D T# write texel x (y ignored) only at
/// slice 0: the first store (y = 1, slice 0) lands, the stores to slices
/// 0xffffffff and 1 are dropped.
#[test]
#[ignore]
fn image_dims_store_line_through_plane_array() {
    const WIDTH: u32 = 32;
    const TEXEL_BYTES: usize = 4096;
    const STORE_CODE: [u32; 17] = [
        0x34020084, 0xe0381000, 0x80001401, 0xbf8c3f70, 0x7e3c0300, 0x7e3e0281, 0x7e400280, 0xf0201f28, 0x0001141e, 0x7e3e0280, 0x7e4002c1, 0xf0201528, 0x0001141e,
        0x7e400281, 0xf0201528, 0x0001141e, 0xbf810000,
    ];
    let value = |tid: u32, index: u32| (tid * 4 + index + 1).wrapping_mul(0x9e37_79b1);
    for (name, format, components) in [("32_SINT", 21u32, 1u32), ("32_32_32_32_SINT", 76, 4)] {
        let mut g = Guest::new(16 << 20);
        let mut rig = rig(&g);
        let shape = Shape { format, bpe: components * 4, ..Shape::new(sharps::image_type::TYPE_1D, WIDTH, 1, 1, 1) };
        let layout = shape.layout();
        let base = g.alloc(&initial_bytes(TEXEL_BYTES));
        let input: Vec<u32> = (0..THREADS).flat_map(|tid| (0..4).map(move |i| value(tid, i))).collect();
        store_dispatch(&mut g, &mut rig, &STORE_CODE, &input, &shape.descriptor(base));
        let bytes = written_back(&g, &mut rig, base, TEXEL_BYTES as u64);
        for x in 0..WIDTH {
            let o = layout.element_offset(0, 0, x, 0).unwrap() as usize;
            for c in 0..components {
                let at = o + c as usize * 4;
                let actual = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
                let wanted = value(x, c);
                assert_eq!(actual, wanted, "image_store 2d_array through a 1D {name} T#: texel {x} component {c} is {actual:#010x}, expected the slice 0 store {wanted:#010x} (slices 0xffffffff and 1 dropped)");
            }
        }
        let end = layout.element_offset(0, 0, WIDTH - 1, 0).unwrap() as usize + (components * 4) as usize;
        assert!(bytes[end..] == initial_bytes(TEXEL_BYTES)[end..], "image_store 2d_array through a 1D {name} T#: bytes past the texture changed");
    }
}

// ---- ImageXorSwizzleVolume ----

/// 3D loads of thick (SW_64KB_S_X, SW_64KB_D_X) and thin (SW_64KB_Z_X)
/// swizzled volumes read the texel at (x, y, z), corners and block edges
/// included. The reference also runs a thick SW_4KB_S_X volume: our tiler
/// has no thick SW_4KB_S_X equation yet.
#[test]
#[ignore]
fn image_dims_xor_swizzle_volume() {
    const WIDTH: u32 = 40;
    const HEIGHT: u32 = 24;
    const DEPTH: u32 = 20;
    const CODE: [u32; 10] = [0x34020084, 0xe0381000, 0x80000201, 0xbf8c3f70, 0xf0001f10, 0x00020602, 0xbf8c3f70, 0xe0781000, 0x80010601, 0xbf810000];
    let texel = |x: u32, y: u32, z: u32| [(x * 6 + 1) as u8, (y * 10 + 2) as u8, (z * 12 + 3) as u8, 0xFF];
    let mut input: Vec<u32> = (0..THREADS).flat_map(|tid| [(tid * 13 + 7) % WIDTH, (tid * 5 + 3) % HEIGHT, (tid * 7 + 1) % DEPTH, 0]).collect();
    input[0..3].copy_from_slice(&[0, 0, 0]);
    input[4..7].copy_from_slice(&[WIDTH - 1, HEIGHT - 1, DEPTH - 1]);
    input[8..11].copy_from_slice(&[33, 17, 16]);
    for (tile, name) in [(0x19, "thick SW_64KB_S_X volume"), (0x1a, "thick SW_64KB_D_X volume"), (0x18, "thin SW_64KB_Z_X volume")] {
        let mut g = Guest::new(16 << 20);
        let mut rig = rig(&g);
        let shape = Shape { tile, ..Shape::new(sharps::image_type::TYPE_3D, WIDTH, HEIGHT, DEPTH, 1) };
        let t = texture(&mut g, shape, |_, z, x, y| Some(texel(x, y, z).to_vec()));
        let out = run_split(&mut g, &mut rig, &CODE, &input, 4, &t, [0; 4]);
        for tid in 0..THREADS as usize {
            let (x, y, z) = (input[tid * 4], input[tid * 4 + 1], input[tid * 4 + 2]);
            for (c, e) in texel(x, y, z).into_iter().enumerate() {
                expect_exact(&out, tid * 4 + c, e as u32, &format!("{name}: texel ({x}, {y}, {z}) component {c}"));
            }
        }
    }
}

// ---- ImagePlaneAddressOnLineImplicitLod, ImagePlaneBiasOnSingleLevelVolume ----

pub(super) const PIXEL_WIDTH: u32 = 64;
pub(super) const PIXEL_HEIGHT: u32 = 16;

/// Full-target draws of pixel programs through the SPIR-V compiler (POS_X
/// and POS_Y in v0 and v1, wave64): `setup` places the textures in guest
/// memory and returns each draw's (program, user data). Returns the RGBA8
/// pixels of each draw.
pub(super) fn draw_pixels(setup: impl FnOnce(&mut Guest) -> Vec<(Vec<u32>, Vec<u32>)>) -> Vec<Vec<u8>> {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut g = Guest::new(16 << 20);
    let programs = setup(&mut g);
    let vs = g.program(&[asm::S_ENDPGM]);
    let pss: Vec<u64> = programs.iter().map(|(code, _)| g.program(code)).collect();
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.5, 1.0, 3.0, -1.0, 0.5, 1.0, -1.0, 3.0, 0.5, 1.0]));
    let rest = TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 });
    let inputs = vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)]);
    let spirv_programs = pss.clone();
    let mut rig = Rig::with_compiler(
        &g,
        |dev| Arc::new(SpirvVertices { spirv: crate::shader::SpirvCompiler::for_device(dev, g.mem), vertices: spirv_programs, rest }),
        Some(inputs),
        None,
    );
    programs
        .iter()
        .zip(pss)
        .map(|((_, user), ps)| {
            let target = rgba8_target(g.zeroed((PIXEL_WIDTH * PIXEL_HEIGHT * 4) as usize), PIXEL_WIDTH, PIXEL_HEIGHT);
            let mut d = draw_into(target, stage(vs, &[]), stage(ps, user), primitive::TRI_LIST, auto(3, 1), None);
            d.ps_input_ena = (1 << 8) | (1 << 9);
            d.ps_input_addr = d.ps_input_ena;
            assert_eq!(rig.draw(&d), DrawOutcome::Recorded);
            rig.pixels(&target)
        })
        .collect()
}

/// Pixel user data: s[4:11] the T#, s[12:15] the S#, then `rest`.
pub(super) fn pixel_user(texture: &[u32; 8], sampler: [u32; 4], rest: &[u32]) -> Vec<u32> {
    let mut user = vec![0u32; 4];
    user.extend_from_slice(texture);
    user.extend_from_slice(&sampler);
    user.extend_from_slice(rest);
    user
}

/// Implicit-LOD 2D samples of a mipped 1D T# take the LOD from u alone (v
/// changes by three texels a pixel in both directions).
#[test]
#[ignore]
fn image_dims_plane_address_on_line_implicit_lod() {
    const SIZE: u32 = 16;
    const MIPS: u32 = 5;
    // s_wqm_b64 exec, exec; v2 = s16 * x; v3 = s17 * (x + y);
    // image_sample v[4:7], v[2:3], s[4:11], s[12:15] dmask 0xF 2D;
    // exp mrt0 v4..v7 done vm.
    const PIXEL_CODE: [u32; 10] = [0xbefe0a7e, 0x10040010, 0x06060300, 0x10060611, 0xf0800f08, 0x00610402, 0xbf8c3f70, 0xf800180f, 0x07060504, 0xbf810000];
    let mark = |level: u32| 10 + 40 * level;
    let pixels = draw_pixels(|g| {
        let line = texture(g, Shape::new(sharps::image_type::TYPE_1D, SIZE, 1, 1, MIPS), |level, _, x, _| Some(vec![mark(level) as u8, (x * 8) as u8, 99, 255]));
        (0..3).map(|level| (PIXEL_CODE.to_vec(), pixel_user(&line, point_sampler(), &[bits((1u32 << level) as f32 / SIZE as f32), bits(3.0)]))).collect()
    });
    for (level, p) in pixels.iter().enumerate() {
        let level = level as u32;
        let what = format!("image_sample 2d on a 1D texture with the first coordinate advancing {} texels per pixel", 1 << level);
        check_pixels(p, &what, |x, _| [mark(level), (x % (SIZE >> level)) * 8, 99, 255]);
    }
}

pub(super) fn check_pixels(p: &[u8], what: &str, expected: impl Fn(u32, u32) -> [u32; 4]) {
    for y in 0..PIXEL_HEIGHT {
        for x in 0..PIXEL_WIDTH {
            let got = px(p, PIXEL_WIDTH, x, y);
            let e = expected(x, y);
            for c in 0..4 {
                assert_eq!(got[c] as u32, e[c], "{what}: pixel {x},{y} component {c} is {}, expected {}", got[c], e[c]);
            }
        }
    }
}

/// Biased 2D samples of a single-level, single-slice volume read its
/// slice, whatever the bias and the sampler's reduction mode.
#[test]
#[ignore]
fn image_dims_plane_bias_on_single_level_volume() {
    const SIZE: u32 = 16;
    // v2 = s16 * x; v3 = s17 * (x + y); v1 = s18 (bias);
    // image_sample_b v[4:7], v[1:3], s[4:11], s[12:15] dmask 0xF 2D;
    // exp mrt0 v4..v7 done vm.
    const PIXEL_CODE: [u32; 10] = [0x10040010, 0x06060300, 0x10060611, 0x7e020212, 0xf0940f08, 0x00610401, 0xbf8c3f70, 0xf800180f, 0x07060504, 0xbf810000];
    let mut cases: Vec<(f32, u32)> = [-16.0, -0.5, 0.0, 0.5, 16.0].iter().map(|&b| (b, 0)).collect();
    cases.extend([(8.0, 1), (8.0, 2)]);
    let pixels = draw_pixels(|g| {
        let volume = texture(g, Shape::new(sharps::image_type::TYPE_3D, SIZE, SIZE, 1, 1), |_, _, x, y| Some(vec![(x * 8) as u8, (y * 8) as u8, 99, 255]));
        cases
            .iter()
            .map(|&(bias, reduction)| {
                let mut sampler = point_sampler();
                sampler[0] |= reduction << 29;
                (PIXEL_CODE.to_vec(), pixel_user(&volume, sampler, &[bits(1.0 / SIZE as f32), bits(1.0 / SIZE as f32), bits(bias)]))
            })
            .collect()
    });
    for (p, (bias, reduction)) in pixels.iter().zip(&cases) {
        check_pixels(p, &format!("single-level volume, bias {bias}, reduction {reduction}"), |x, y| [(x % SIZE) * 8, ((x + y + 1) % SIZE) * 8, 99, 255]);
    }
}
