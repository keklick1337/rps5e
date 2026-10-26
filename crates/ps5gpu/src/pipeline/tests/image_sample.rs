// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Image samples through the SPIR-V compiler on the device, against the
//! reference's hardware-measured driver execution tests of the same names
//! (ImageSampleBiasA16, ImageSampleLodClamp, ImageSampleCubeDerivatives,
//! ImageSampleBorderColor, ...).
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored image_sample_ref`.

use super::image_dims::*;
use super::*;

const SIZE: u32 = 16;
const LEVELS: u32 = 5;
const RESULT_WORD: usize = 8;

/// 16x16 RGBA8 with five levels: texel (x, y) of level l is (l, x, y, 255).
fn mipped_plane(g: &mut Guest) -> [u32; 8] {
    texture(g, Shape::new(sharps::image_type::TYPE_2D, SIZE, SIZE, 1, LEVELS), |level, _, x, y| Some(vec![level as u8, x as u8, y as u8, 255]))
}

fn wrap(coordinate: u32, offset: i32, size: u32) -> u32 {
    (coordinate as i32 + offset).rem_euclid(size as i32) as u32
}

/// f32 to f16 bits as the reference test converts (normal values only).
fn half(v: f32) -> u32 {
    if v == 0.0 {
        return 0;
    }
    let b = v.to_bits();
    ((((b >> 23) & 0xFF) - 112) << 10) | ((b >> 13) & 0x3FF)
}

/// Results at RESULT_WORD + 4 * k: texel (level, x >> level wrapped by the
/// offset, y >> level wrapped, 255).
fn check_levels(b: &[u32], names: &[&str], offsets: &[[i32; 2]], level_of: impl Fn(u32, usize) -> u32, xy: impl Fn(u32) -> (u32, u32)) {
    for tid in 0..THREADS {
        let (x, y) = xy(tid);
        for (k, name) in names.iter().enumerate() {
            let level = level_of(tid, k);
            let size = SIZE >> level;
            let e = [level, wrap(x >> level, offsets[k][0], size), wrap(y >> level, offsets[k][1], size), 255];
            expect(b, tid, RESULT_WORD + 4 * k, e, name);
        }
    }
}

/// A16 biased samples in a compute program read level 0 (the offsets
/// wrap); the LOD clamp raises the level to the clamp.
#[test]
#[ignore]
fn image_sample_ref_bias_a16() {
    const CLAMPS: u32 = 4;
    const BIAS_CODE: [u32; 41] = [
        0x34020087, 0xe0301000, 0x80000501, 0xe0301004, 0x80000601, 0xbf8c3f70, 0xf0940f08, 0x40610c05, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01,
        0xe0701028, 0x80000e01, 0xe070102c, 0x80000f01, 0x7e0802ff, 0x00000201, 0xf0d40f08, 0x40610c04, 0xe0701030, 0x80000c01, 0xe0701034, 0x80000d01,
        0xe0701038, 0x80000e01, 0xe070103c, 0x80000f01, 0x7e0802ff, 0x00003e3f, 0xf0d40f08, 0x40610c04, 0xe0701040, 0x80000c01, 0xe0701044, 0x80000d01,
        0xe0701048, 0x80000e01, 0xe070104c, 0x80000f01, 0xbf810000,
    ];
    const BIAS_CLAMP_CODE: [u32; 19] = [
        0x34020087, 0xe0301000, 0x80000501, 0xe0301004, 0x80000601, 0xe0301008, 0x80000701, 0xbf8c3f70, 0xf0980f08, 0x40610c05, 0xe0701050, 0x80000c01,
        0xe0701054, 0x80000d01, 0xe0701058, 0x80000e01, 0xe070105c, 0x80000f01, 0xbf810000,
    ];
    let sample = |tid: u32| ((tid * 7 + 3) % SIZE, (tid * 5 + 1) % SIZE, (tid / 3) % CLAMPS);
    let coordinate = |texel: u32| half((texel as f32 + 0.5) / SIZE as f32);
    let input = lanes(|tid| {
        let (x, y, clamp) = sample(tid);
        vec![half(0.0) | (coordinate((x + 5) % SIZE) << 16), coordinate(x) | (coordinate(y) << 16), half(clamp as f32) | (half(4.0) << 16)]
    });
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let t = mipped_plane(&mut g);
    let b = run(&mut g, &mut rig, &BIAS_CODE, &input, &t, point_sampler());
    check_levels(&b, &["image_sample_b a16", "image_sample_b_o a16 (1, 2)", "image_sample_b_o a16 (-1, -2)"], &[[0, 0], [1, 2], [-1, -2]], |_, _| 0, |t| (sample(t).0, sample(t).1));
    // The clamp sample stores at word 8 + 12: its own run, checked alone.
    let b = run(&mut g, &mut rig, &BIAS_CLAMP_CODE, &input, &t, point_sampler());
    for tid in 0..THREADS {
        let (x, y, clamp) = sample(tid);
        let size = SIZE >> clamp;
        expect(&b, tid, RESULT_WORD + 12, [clamp, wrap(x >> clamp, 0, size), wrap(y >> clamp, 0, size), 255], "image_sample_b_cl a16");
    }
}

/// Gradient samples with an LOD clamp: the level is the larger of the
/// gradient's and the clamp; the clamp alone (no gradients in a compute
/// program) reads the clamp level; offsets wrap.
#[test]
#[ignore]
fn image_sample_ref_lod_clamp() {
    const GRADIENTS: u32 = 4;
    const CLAMPS: u32 = 4;
    const CODE: [u32; 61] = [
        0x34020087, 0xe0301000, 0x80000501, 0xe0301004, 0x80000601, 0xe0301008, 0x80000701, 0xe030100c, 0x80000801, 0xe0301010, 0x80000901, 0xe0301014,
        0x80000a01, 0xe0301018, 0x80000b01, 0xbf8c3f70, 0xf08c0f08, 0x00610c05, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028, 0x80000e01,
        0xe070102c, 0x80000f01, 0xf0840f08, 0x00610c09, 0xe0701030, 0x80000c01, 0xe0701034, 0x80000d01, 0xe0701038, 0x80000e01, 0xe070103c, 0x80000f01,
        0x7e0802ff, 0x00000201, 0xf0cc0f08, 0x00610c04, 0xe0701040, 0x80000c01, 0xe0701044, 0x80000d01, 0xe0701048, 0x80000e01, 0xe070104c, 0x80000f01,
        0x7e0802ff, 0x00003e3f, 0xf0cc0f08, 0x00610c04, 0xe0701050, 0x80000c01, 0xe0701054, 0x80000d01, 0xe0701058, 0x80000e01, 0xe070105c, 0x80000f01,
        0xbf810000,
    ];
    // (level, gradient, clamp, x, y)
    let sample = |tid: u32| (tid % LEVELS, (tid / LEVELS) % GRADIENTS, (tid / 3) % CLAMPS, (tid * 7 + 3) % SIZE, (tid * 5 + 1) % SIZE);
    let input = lanes(|tid| {
        let (level, gradient, clamp, x, y) = sample(tid);
        let mut w: Vec<u32> = (0..GRADIENTS).map(|k| bits(if k == gradient { (level as f32 - 0.25).exp2() / SIZE as f32 } else { 0.0 })).collect();
        w.extend([bits((x as f32 + 0.5) / SIZE as f32), bits((y as f32 + 0.5) / SIZE as f32), bits(clamp as f32)]);
        w
    });
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let t = mipped_plane(&mut g);
    let b = run(&mut g, &mut rig, &CODE, &input, &t, point_sampler());
    let names = ["image_sample_d_cl", "image_sample_cl", "image_sample_d_cl_o (1, 2)", "image_sample_d_cl_o (-1, -2)"];
    let level_of = |tid: u32, k: usize| {
        let (level, _, clamp, _, _) = sample(tid);
        if k == 1 {
            clamp
        } else {
            level.max(clamp)
        }
    };
    check_levels(&b, &names, &[[0, 0], [0, 0], [1, 2], [-1, -2]], level_of, |t| (sample(t).3, sample(t).4));
}

/// image_sample_d on a cube: face-relative gradients select the level.
#[test]
#[ignore]
fn image_sample_ref_cube_derivatives() {
    const FACES: u32 = 6;
    const GRADIENTS: u32 = 4;
    const CODE: [u32; 28] = [
        0x34020087, 0xe0301000, 0x80000501, 0xe0301004, 0x80000601, 0xe0301008, 0x80000701, 0xe030100c, 0x80000801, 0xe0301010, 0x80000901, 0xe0301014,
        0x80000a01, 0xe0301018, 0x80000b01, 0xbf8c3f70, 0xf0880f18, 0x00610c05, 0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028,
        0x80000e01, 0xe070102c, 0x80000f01, 0xbf810000,
    ];
    // (level, face, gradient, x, y)
    let sample = |tid: u32| (tid % LEVELS, (tid / LEVELS) % FACES, (tid / 3) % GRADIENTS, (tid * 7 + 3) % SIZE, (tid * 5 + 1) % SIZE);
    let input = lanes(|tid| {
        let (level, face, gradient, x, y) = sample(tid);
        let mut w: Vec<u32> = (0..GRADIENTS).map(|k| bits(if k == gradient { (level as f32 - 0.25).exp2() / SIZE as f32 } else { 0.0 })).collect();
        w.extend([bits(1.0 + (x as f32 + 0.5) / SIZE as f32), bits(1.0 + (y as f32 + 0.5) / SIZE as f32), bits(face as f32)]);
        w
    });
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let t = texture(&mut g, Shape::new(sharps::image_type::CUBE, SIZE, SIZE, FACES, LEVELS), |level, face, _, _| Some(vec![level as u8, face as u8, 0, 255]));
    let b = run(&mut g, &mut rig, &CODE, &input, &t, point_sampler());
    for tid in 0..THREADS {
        let (level, face, ..) = sample(tid);
        expect(&b, tid, RESULT_WORD, [level, face, 0, 255], &format!("image_sample_d cube (face {face}, level {level})"));
    }
}

/// Samples outside a clamp-to-border texture return the sampler's border
/// colour: 0/1 per channel as float for UNORM views, as integers for UINT
/// ones, also under ANISO_OVERRIDE filters.
#[test]
#[ignore]
fn image_sample_ref_border_color() {
    const WIDTH: u32 = 4;
    const ONE: u32 = 0x3f80_0000;
    const CLAMP_BORDER: u32 = 0x1b6;
    const ANISO_OVERRIDE: u32 = (1 << 29) | (2 << 22) | (2 << 20);
    // v3 = tid * 16; v20 = v21 = 2.0; image_sample_lz v[10:13], v[20:21],
    // s[4:11], s[12:15]; buffer_store_dwordx4 v[10:13], v3, s[0:3].
    const SAMPLE_OUTSIDE: [u32; 9] = [0x34060084, 0x7e2802f4, 0x7e2a02f4, 0xf09c0f08, 0x00610a14, 0xbf8c3f70, 0xe0781000, 0x80000a03, 0xbf810000];
    let cases: [(u32, u32, [u32; 4], &str, u32); 8] = [
        (56, 0, [0, 0, 0, 0], "8_8_8_8_UNORM transparent black border", 0),
        (56, 1, [0, 0, 0, ONE], "8_8_8_8_UNORM opaque black border", 0),
        (56, 2, [ONE; 4], "8_8_8_8_UNORM opaque white border", 0),
        (60, 0, [0, 0, 0, 0], "8_8_8_8_UINT transparent black border", 0),
        (60, 1, [0, 0, 0, 1], "8_8_8_8_UINT opaque black border", 0),
        (60, 2, [1; 4], "8_8_8_8_UINT opaque white border", 0),
        (56, 2, [ONE; 4], "8_8_8_8_UNORM opaque white border under ANISO_OVERRIDE", ANISO_OVERRIDE),
        (60, 2, [1; 4], "8_8_8_8_UINT opaque white border under ANISO_OVERRIDE", ANISO_OVERRIDE),
    ];
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    for (format, border, expected, what, filters) in cases {
        let t = texture(&mut g, Shape { format, ..Shape::new(sharps::image_type::TYPE_2D, WIDTH, 1, 1, 1) }, |_, _, _, _| Some(vec![0; 4]));
        let words = THREADS as usize * 4;
        let out = g.alloc(&asm::bytes(&vec![POISON; words]));
        let mut user = raw(out, (words * 4) as u32).to_vec();
        user.extend_from_slice(&t);
        user.extend_from_slice(&[CLAMP_BORDER, 0x00ff_f000, filters, border << 30]);
        dispatch(&mut g, &mut rig, &SAMPLE_OUTSIDE, &user);
        let o = read_words(&g, &mut rig, out, words);
        for tid in 0..THREADS as usize {
            for c in 0..4 {
                assert_eq!(o[tid * 4 + c], expected[c], "{what}: thread {tid} component {c} is {:#010x}, expected {:#010x}", o[tid * 4 + c], expected[c]);
            }
        }
    }
}

// ---- ImageSampleDerivatives ----

mod derivatives {
    use super::*;
    include!("image_sample_derivatives_data.rs");

    const INPUTS: usize = 12;
    const RESULTS: usize = 16;
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 4;
    const DEPTH: u32 = 2;

    /// The input row lane `tid` takes (wave64 lanes repeat rows from the
    /// middle).
    fn source_row(tid: u32) -> u32 {
        if tid < THREADS {
            tid
        } else {
            (tid - THREADS + THREADS / 2) % THREADS
        }
    }

    /// The program with its result stores moved past the inputs of
    /// `threads` lanes.
    fn moved_stores(code: &[u32], threads: u32) -> Vec<u32> {
        code.iter()
            .map(|&w| if w & 0xFFFF_F000 == 0xE070_1000 && (w & 0xFFF) as usize >= THREADS as usize * INPUTS * 4 { w + (threads - THREADS) * INPUTS as u32 * 4 } else { w })
            .collect()
    }

    /// Runs `code` in a wave of `threads` on a 32-bit float texture of
    /// type `ty` (texel x + 100 y + 1000 z) and checks every case against
    /// the measured values (within 1/64; EXEC-masked cases on wave32 only).
    fn run_case(code: &[u32], threads: u32, ty: u32, height: u32, depth: u32, names: &[&str], masked: &[bool], expected: &[u32]) {
        let cases = names.len();
        let mut g = Guest::new(16 << 20);
        let mut rig = rig(&g);
        let shape = Shape { format: 22, ..Shape::new(ty, WIDTH, height, depth, 1) };
        let mut t = texture(&mut g, shape, |_, z, x, y| Some(((x + 100 * y + 1000 * z) as f32).to_le_bytes().to_vec()));
        t[2] |= 1 << 31;
        let mut buffer = vec![POISON; threads as usize * (INPUTS + RESULTS)];
        for tid in 0..threads as usize {
            let row = source_row(tid as u32) as usize;
            buffer[tid * INPUTS..tid * INPUTS + INPUTS].copy_from_slice(&INPUT[row * INPUTS..row * INPUTS + INPUTS]);
        }
        let b = g.alloc(&asm::bytes(&buffer));
        let mut user = raw(b, (buffer.len() * 4) as u32).to_vec();
        user.extend_from_slice(&t);
        user.extend_from_slice(&[0x92, (4 * 256) << 12, (1 << 22) | (1 << 26), 0]);
        let cs = g.program(&moved_stores(code, threads));
        let mut shader = stage(cs, &user);
        shader.rsrc1 = (0xC0 << 12) | (1 << 21);
        let d = ResolvedDispatch {
            packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1 | if threads == 32 { 1 << 15 } else { 0 }, ..DispatchPacket::default() },
            shader,
            rsrc3: 0,
            threads: [threads, 1, 1],
            start: [0; 3],
        };
        assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
        let out = read_words(&g, &mut rig, b, buffer.len());
        for tid in 0..threads {
            for k in 0..cases {
                if masked[k] && tid >= THREADS {
                    continue;
                }
                let actual = out[threads as usize * INPUTS + tid as usize * RESULTS + k];
                let wanted = expected[source_row(tid) as usize * cases + k];
                let close = (f32::from_bits(actual) - f32::from_bits(wanted)).abs() <= 1.0 / 64.0;
                assert!(close, "{}: wave{threads} lane {tid} is {actual:#010x} ({}), expected {wanted:#010x} ({})", names[k], f32::from_bits(actual), f32::from_bits(wanted));
            }
        }
    }

    const NAMES_2D: [&str; 12] = [
        "image_sample_d",
        "image_sample_cd",
        "image_sample_cd (exec)",
        "image_sample_d_g16",
        "image_sample_cd_g16",
        "image_sample_d a16",
        "image_sample_d_g16 a16",
        "image_sample_cd_o",
        "image_sample_d_o_g16",
        "image_sample_cd_o_g16",
        "image_sample_d_o_g16 a16",
        "image_sample_cd_o_g16 (exec)",
    ];
    const MASKED_2D: [bool; 12] = [false, false, true, false, false, false, false, false, false, false, false, true];
    const NAMES_3D: [&str; 6] = ["image_sample_d 3D", "image_sample_cd 3D", "image_sample_d_g16 3D", "image_sample_cd_g16 3D", "image_sample_d a16 3D", "image_sample_d_g16 a16 3D"];
    const NAMES_1D: [&str; 5] = ["image_sample_d 1D", "image_sample_d_g16 1D", "image_sample_cd_g16 1D", "image_sample_d a16 1D", "image_sample_cd_g16 a16 1D"];

    /// image_sample_d / _cd (coarse), G16 and A16 gradients, offsets and
    /// EXEC-masked lanes on 2D, 3D and 1D textures, wave32 and wave64.
    #[test]
    #[ignore]
    fn image_sample_ref_derivatives_2d() {
        for threads in [32, 64] {
            run_case(&CODE_2D, threads, sharps::image_type::TYPE_2D, HEIGHT, 1, &NAMES_2D, &MASKED_2D, &EXPECTED_2D);
        }
    }

    #[test]
    #[ignore]
    fn image_sample_ref_derivatives_3d() {
        for threads in [32, 64] {
            run_case(&CODE_3D, threads, sharps::image_type::TYPE_3D, HEIGHT, DEPTH, &NAMES_3D, &[false; 6], &EXPECTED_3D);
        }
    }

    #[test]
    #[ignore]
    fn image_sample_ref_derivatives_1d() {
        for threads in [32, 64] {
            run_case(&CODE_1D, threads, sharps::image_type::TYPE_1D, 1, 1, &NAMES_1D, &[false; 5], &EXPECTED_1D);
        }
    }
}

// ---- ImageSampleReduction ----

mod reduction {
    use super::*;

    const SIZE: u32 = 4;
    const LEVELS: u32 = 2;
    const MIN: u32 = 1;
    const MAX: u32 = 2;
    const POINT: u32 = 0;
    const BILINEAR: u32 = 1;
    const ANISO_BILINEAR: u32 = 3;
    const MIP_NONE: u32 = 0;
    const MIP_POINT: u32 = 1;
    const LEVEL0: [f32; 16] = [17.0, 11.0, 23.0, 14.0, 20.0, 25.0, 12.0, 19.0, 13.0, 22.0, 16.0, 24.0, 21.0, 10.0, 18.0, 15.0];
    const LEVEL1: [f32; 4] = [5.0, 30.0, 16.5, 9.0];
    /// v10 = tid * 12; load (u, v, lod); image_sample_l v4 (DMASK 0x1),
    /// s[8:15], s[16:19]; store v4 at tid * 4 of s[4:7].
    const CODE: [u32; 13] = [0x1614008c, 0xe03c1000, 0x8000010a, 0xbf8c3f70, 0xf0900108, 0x00820401, 0xbf8c3f70, 0x34160082, 0xe0701000, 0x8001040b, 0xbf810000, 0xbf810000, 0xbf810000];
    const GATHER_CODE: [u32; 13] = [0x1614008c, 0xe03c1000, 0x8000010a, 0xbf8c3f70, 0xf11c0108, 0x00820401, 0xbf8c3f70, 0x34160084, 0xe0781000, 0x8001040b, 0xbf810000, 0xbf810000, 0xbf810000];
    /// The sample above plus image_load of the 8_8_8_8 T# in s[20:27].
    const LOADED_CODE: [u32; 19] = [
        0x1614008c, 0xe03c1000, 0x8000010a, 0x36180083, 0x7e1a0280, 0xbf8c3f70, 0xf0900108, 0x00820401, 0xf0000108, 0x0005050c, 0xbf8c3f70, 0x34160082,
        0xe0701000, 0x8001040b, 0xe0701080, 0x8001050b, 0xbf810000, 0xbf810000, 0xbf810000,
    ];
    /// The sample above plus a sample of the 8_8_8_8 T# through the same S#.
    const SAMPLED_CODE: [u32; 17] = [
        0x1614008c, 0xe03c1000, 0x8000010a, 0xbf8c3f70, 0xf0900108, 0x00820401, 0xf0900108, 0x00850501, 0xbf8c3f70, 0x34160082, 0xe0701000, 0x8001040b,
        0xe0701080, 0x8001050b, 0xbf810000, 0xbf810000, 0xbf810000,
    ];

    #[derive(Clone, Copy)]
    struct Sampler {
        reduction: u32,
        filter: u32,
        mip: u32,
        aniso: u32,
    }

    fn sampler_words(s: Sampler) -> [u32; 4] {
        [0x92 | (s.aniso << 9) | (s.reduction << 29), 0x00ff_f000, (s.filter << 20) | (s.filter << 22) | (1 << 24) | (s.mip << 26), 0]
    }

    fn input() -> Vec<f32> {
        (0..THREADS)
            .flat_map(|tid| {
                let u = (((tid * 5 + 1) % 12) as f32 - 1.7) / 8.0;
                let v = (((tid * 7 + 3) % 12) as f32 - 1.7) / 8.0;
                [u, v, if tid % 2 == 0 { 0.25 } else { 0.75 }]
            })
            .collect()
    }

    fn texel(level: u32, x: i32, y: i32) -> f32 {
        let size = (SIZE >> level) as i32;
        let (cx, cy) = (x.clamp(0, size - 1) as usize, y.clamp(0, size - 1) as usize);
        if level == 0 {
            LEVEL0[cy * SIZE as usize + cx]
        } else {
            LEVEL1[cy * (SIZE / 2) as usize + cx]
        }
    }

    fn red(x: i32, y: i32) -> u32 {
        let (cx, cy) = (x.clamp(0, SIZE as i32 - 1) as u32, y.clamp(0, SIZE as i32 - 1) as u32);
        17 * ((cy * 7 + cx * 5) % 16)
    }

    fn expected(i: &[f32], tid: usize, s: Sampler) -> f32 {
        let (u, v, lod) = (i[tid * 3], i[tid * 3 + 1], i[tid * 3 + 2]);
        let level = if s.mip == MIP_POINT && lod > 0.5 { 1 } else { 0 };
        let size = (SIZE >> level) as f32;
        let footprint: Vec<f32> = if s.filter == POINT {
            vec![texel(level, (u * size).floor() as i32, (v * size).floor() as i32)]
        } else {
            let (x, y) = ((u * size - 0.5).floor() as i32, (v * size - 0.5).floor() as i32);
            vec![texel(level, x, y), texel(level, x + 1, y), texel(level, x, y + 1), texel(level, x + 1, y + 1)]
        };
        let it = footprint.into_iter();
        if s.reduction == MIN {
            it.fold(f32::INFINITY, f32::min)
        } else {
            it.fold(f32::NEG_INFINITY, f32::max)
        }
    }

    struct Setup {
        g: Guest,
        rig: Rig,
        texels: [u32; 8],
        colors: [u32; 8],
        input: Vec<f32>,
    }

    fn setup() -> Setup {
        let mut g = Guest::new(16 << 20);
        let rig = rig(&g);
        let shape = Shape { format: 22, ..Shape::new(sharps::image_type::TYPE_2D, SIZE, SIZE, 1, LEVELS) };
        let texels = texture(&mut g, shape, |level, _, x, y| Some(texel(level, x as i32, y as i32).to_le_bytes().to_vec()));
        let colors = texture(&mut g, Shape::new(sharps::image_type::TYPE_2D, SIZE, SIZE, 1, 1), |_, _, x, y| Some(vec![red(x as i32, y as i32) as u8, 0, 0, 255]));
        Setup { g, rig, texels, colors, input: input() }
    }

    fn run(s: &mut Setup, code: &[u32], sampler: Sampler, out_words: usize) -> Vec<u32> {
        let input: Vec<u32> = s.input.iter().map(|f| f.to_bits()).collect();
        let mut high = sampler_words(sampler).to_vec();
        high.extend_from_slice(&s.colors);
        run_split_sgprs(&mut s.g, &mut s.rig, code, &input, out_words, &s.texels, &high)
    }

    /// Min / max reduction samplers return the minimum / maximum of the
    /// footprint (point, bilinear, 2x anisotropic bilinear; point mips or
    /// none).
    #[test]
    #[ignore]
    fn image_sample_ref_reduction() {
        let mut s = setup();
        let cases = [
            (Sampler { reduction: MIN, filter: BILINEAR, mip: MIP_POINT, aniso: 0 }, "min, bilinear, point mip"),
            (Sampler { reduction: MAX, filter: BILINEAR, mip: MIP_POINT, aniso: 0 }, "max, bilinear, point mip"),
            (Sampler { reduction: MIN, filter: POINT, mip: MIP_POINT, aniso: 0 }, "min, point, point mip"),
            (Sampler { reduction: MAX, filter: POINT, mip: MIP_POINT, aniso: 0 }, "max, point, point mip"),
            (Sampler { reduction: MIN, filter: BILINEAR, mip: MIP_NONE, aniso: 0 }, "min, bilinear, no mip"),
            (Sampler { reduction: MAX, filter: POINT, mip: MIP_NONE, aniso: 0 }, "max, point, no mip"),
            (Sampler { reduction: MIN, filter: ANISO_BILINEAR, mip: MIP_NONE, aniso: 1 }, "min, 2x anisotropic bilinear, no mip"),
            (Sampler { reduction: MAX, filter: ANISO_BILINEAR, mip: MIP_NONE, aniso: 1 }, "max, 2x anisotropic bilinear, no mip"),
        ];
        for (sampler, name) in cases {
            let out = run(&mut s, &CODE, sampler, 4);
            for tid in 0..THREADS as usize {
                let e = expected(&s.input, tid, sampler);
                let got = f32::from_bits(out[tid]);
                assert_eq!(got, e, "{name}: thread {tid} sampled {got}, expected {e}");
            }
        }
    }

    /// Gathers ignore the reduction mode.
    #[test]
    #[ignore]
    fn image_sample_ref_reduction_gather() {
        let mut s = setup();
        for reduction in [MIN, MAX] {
            let out = run(&mut s, &GATHER_CODE, Sampler { reduction, filter: BILINEAR, mip: MIP_POINT, aniso: 0 }, 4);
            for tid in 0..THREADS as usize {
                let (x, y) = ((s.input[tid * 3] * SIZE as f32 - 0.5).floor() as i32, (s.input[tid * 3 + 1] * SIZE as f32 - 0.5).floor() as i32);
                let e = [texel(0, x, y + 1), texel(0, x + 1, y + 1), texel(0, x + 1, y), texel(0, x, y)];
                for c in 0..4 {
                    let got = f32::from_bits(out[tid * 4 + c]);
                    assert_eq!(got, e[c], "gather4_lz, reduction {reduction}: thread {tid} component {c} gathered {got}, expected {}", e[c]);
                }
            }
        }
    }

    /// A min sampler next to an image_load of an 8_8_8_8 T# (no pairing),
    /// and next to a sample of it through the same S# (min-filtered).
    #[test]
    #[ignore]
    fn image_sample_ref_reduction_with_colors() {
        let sampler = Sampler { reduction: MIN, filter: BILINEAR, mip: MIP_POINT, aniso: 0 };
        for (code, sampled) in [(&LOADED_CODE[..], false), (&SAMPLED_CODE[..], true)] {
            let mut s = setup();
            let out = run(&mut s, code, sampler, 8);
            for tid in 0..THREADS as usize {
                let e = expected(&s.input, tid, sampler);
                let got = f32::from_bits(out[tid]);
                assert_eq!(got, e, "sampled {sampled}: thread {tid} sampled {got} from 32_FLOAT, expected {e}");
                let color = if sampled {
                    let (x, y) = ((s.input[tid * 3] * SIZE as f32 - 0.5).floor() as i32, (s.input[tid * 3 + 1] * SIZE as f32 - 0.5).floor() as i32);
                    [red(x, y), red(x + 1, y), red(x, y + 1), red(x + 1, y + 1)].into_iter().min().unwrap() as f32 / 255.0
                } else {
                    red((tid & 3) as i32, 0) as f32 / 255.0
                };
                let got = f32::from_bits(out[THREADS as usize + tid]);
                assert!((got - color).abs() <= 1.0 / 1024.0, "sampled {sampled}: thread {tid} read {got} from 8_8_8_8, expected {color}");
            }
        }
    }
}

// ---- ImageRepackedFloat ----

mod repacked_float {
    use super::*;

    const MIP_SIZE: u32 = 16;
    const MIP_LEVELS: u32 = 5;
    const CUBE_SIZE: u32 = 8;
    const FACES: u32 = 6;
    const REPACKED: u32 = 43;
    const NATIVE: u32 = 36;
    const RESULT: usize = 8;
    const POINT_MIP: [u32; 4] = [0, 0xFFF << 12, 1 << 26, 0];
    const BILINEAR: [u32; 4] = [0, 0xFFF << 12, (1 << 20) | (1 << 22), 0];
    /// image_sample_lz, _l, implicit image_sample and image_load (DMASK
    /// 0x1 each) of s[4:11] / s[12:15].
    const CODE: [u32; 30] = [
        0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c, 0x80000501, 0xe0301010, 0x80000601, 0xbf8c3f70,
        0xf09c0108, 0x00610c02, 0xf0900108, 0x00610d02, 0xf0800108, 0x00610e02, 0xf0001108, 0x00010f05, 0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024,
        0x80000d01, 0xe0701028, 0x80000e01, 0xe070102c, 0x80000f01, 0xbf810000,
    ];

    fn guest_word(mut seed: u32) -> u32 {
        seed ^= seed >> 16;
        seed = seed.wrapping_mul(0x7feb_352d);
        seed ^= seed >> 15;
        seed = seed.wrapping_mul(0x846c_a68b);
        seed ^ (seed >> 16)
    }

    fn small_float(bits: u32, mantissa_bits: u32) -> f32 {
        let exponent = bits >> mantissa_bits;
        let mantissa = bits & ((1 << mantissa_bits) - 1);
        let scale = (1u32 << mantissa_bits) as f32;
        if exponent == 31 {
            return if mantissa == 0 { f32::INFINITY } else { f32::NAN };
        }
        if exponent == 0 {
            return (mantissa as f32 / scale) * 2f32.powi(-14);
        }
        (1.0 + mantissa as f32 / scale) * 2f32.powi(exponent as i32 - 15)
    }

    /// Component `c` of a 10_11_11 FLOAT word (R 10 bits low, then G 11,
    /// B 11).
    fn component(word: u32, c: u32) -> f32 {
        match c {
            0 => small_float(word & 0x3FF, 5),
            1 => small_float((word >> 10) & 0x7FF, 6),
            _ => small_float(word >> 21, 6),
        }
    }

    fn same(a: f32, e: f32) -> bool {
        (a.is_nan() && e.is_nan()) || a.to_bits() == e.to_bits()
    }

    /// DST_SEL with X taking component `c`.
    fn swizzle(c: u32) -> u32 {
        (4 + c) | (5 << 3) | (6 << 6) | (7 << 9)
    }

    fn shape(format: u32, size: u32, ty: u32, slices: u32, levels: u32, c: u32) -> (Shape, u32) {
        (Shape { format, ..Shape::new(ty, size, size, slices, levels) }, swizzle(c))
    }

    fn make(g: &mut Guest, (s, sw): (Shape, u32), word: impl Fn(u32, u32, u32, u32) -> u32) -> [u32; 8] {
        let mut t = texture(g, s, |level, layer, x, y| Some(word(layer, level, x, y).to_le_bytes().to_vec()));
        t[3] = (t[3] & !0xFFF) | sw;
        t
    }

    fn input(size: u32, base: f32, lod: u32, nudge: f32) -> Vec<u32> {
        lanes(|tid| {
            let (x, y) = (tid % size, (tid * 3) % size);
            let u = base + (x as f32 + 0.5) / size as f32 + nudge;
            let v = base + (y as f32 + 0.5) / size as f32;
            vec![bits(u), bits(v), bits(lod as f32), x, y]
        })
    }

    fn mip_word(_: u32, level: u32, x: u32, y: u32) -> u32 {
        guest_word(level * 4096 + y * 64 + x)
    }

    /// Point samples of each level and image_load of level 0 of a
    /// 10_11_11 FLOAT texture read each component exactly (denormals,
    /// infinities and NaNs too).
    #[test]
    #[ignore]
    fn image_sample_ref_repacked_float_levels() {
        let mut g = Guest::new(16 << 20);
        let mut rig = rig(&g);
        for lod in 1..MIP_LEVELS {
            let size = MIP_SIZE >> lod;
            for c in 0..3 {
                let t = make(&mut g, shape(REPACKED, MIP_SIZE, sharps::image_type::TYPE_2D, 1, MIP_LEVELS, c), mip_word);
                let b = run(&mut g, &mut rig, &CODE, &input(size, 0.0, lod, 0.0), &t, POINT_MIP);
                for tid in 0..THREADS {
                    let (x, y) = (tid % size, (tid * 3) % size);
                    let sampled = f32::from_bits(b[tid as usize * WORDS + RESULT + 1]);
                    let expected = component(mip_word(0, lod, x, y), c);
                    assert!(same(sampled, expected), "image_sample_l at LOD {lod}, thread {tid} component {c} is {sampled}, expected {expected}");
                    let loaded = f32::from_bits(b[tid as usize * WORDS + RESULT + 3]);
                    let expected = component(mip_word(0, 0, x, y), c);
                    assert!(same(loaded, expected), "image_load of mip 0, thread {tid} component {c} is {loaded}, expected {expected}");
                }
            }
        }
    }

    fn cube_word(face: u32, _: u32, x: u32, y: u32) -> u32 {
        let exponent = 12 + (face + x + y) % 6;
        ((exponent << 5) | ((x * 7 + y) & 31)) | ((((exponent + 1) << 6) | ((y * 5 + face) & 63)) << 10) | ((((exponent + 2) << 6) | ((x * 3 + face * 11) & 63)) << 21)
    }

    /// The same word as an 11_11_10 FLOAT texel (X 11, Y 11, Z 10 bits):
    /// the fields in reverse order.
    fn repack(w: u32) -> u32 {
        (w >> 21) | (((w >> 10) & 0x7FF) << 11) | ((w & 0x3FF) << 22)
    }

    /// Bilinear 2D samples of a 10_11_11 FLOAT cube's face 0 match those of
    /// the same texels as a native 11_11_10 FLOAT cube; image_load reads
    /// face 0 exactly.
    #[test]
    #[ignore]
    fn image_sample_ref_repacked_float_cube_face() {
        const NATIVE_CHANNEL: [u32; 3] = [2, 1, 0];
        for c in 0..3 {
            let mut g = Guest::new(16 << 20);
            let mut rig = rig(&g);
            let input = input(CUBE_SIZE, 1.0, 0, 0.3 / CUBE_SIZE as f32);
            let t = make(&mut g, shape(REPACKED, CUBE_SIZE, sharps::image_type::CUBE, FACES, 1, c), cube_word);
            let repacked = run(&mut g, &mut rig, &CODE, &input, &t, BILINEAR);
            let n = make(&mut g, shape(NATIVE, CUBE_SIZE, sharps::image_type::CUBE, FACES, 1, NATIVE_CHANNEL[c as usize]), |f, l, x, y| repack(cube_word(f, l, x, y)));
            let native = run(&mut g, &mut rig, &CODE, &input, &n, BILINEAR);
            for tid in 0..THREADS as usize {
                for op in 0..4 {
                    let (a, e) = (f32::from_bits(repacked[tid * WORDS + RESULT + op]), f32::from_bits(native[tid * WORDS + RESULT + op]));
                    assert!(same(a, e), "cube face sample {op}, thread {tid} component {c} is {a}, the native 11_11_10 cube gives {e}");
                }
                let face = f32::from_bits(repacked[tid * WORDS + RESULT + 3]);
                let e = component(cube_word(0, 0, tid as u32 % CUBE_SIZE, (tid as u32 * 3) % CUBE_SIZE), c);
                assert!(same(face, e), "image_load of cube face 0, thread {tid} component {c} is {face}, expected {e}");
            }
        }
    }
}

// ---- ImageSampleSharedSampler ----

/// One bilinear S# shared by a 32_SINT T# (sampled with point filtering,
/// as integers are) and an 8_8_8_8_UNORM one (filtered).
#[test]
#[ignore]
fn image_sample_ref_shared_sampler() {
    const SIZE: u32 = 8;
    const SAMPLE_X: u32 = 3;
    const SAMPLE_Y: u32 = 5;
    const LINEAR_SAMPLER: [u32; 4] = [0x92, 0x00ff_f000, 0x0050_0000, 0];
    const CODE: [u32; 16] = [
        0x34060085, 0x7e2802ff, 0x3ef00000, 0x7e2a02ff, 0x3f300000, 0xf09c0f08, 0x00610a14, 0xf09c0f08, 0x00640e14, 0xbf8c3f70, 0xe0781000, 0x80000a03,
        0xe0781010, 0x80000e03, 0xbf810000, 0xbf810000,
    ];
    let integer = |x: u32, y: u32| -((100 * x + 7 * y + 1) as i32);
    let mut g = Guest::new(16 << 20);
    let mut rig = rig(&g);
    let mut ints = texture(&mut g, Shape { format: 21, ..Shape::new(sharps::image_type::TYPE_2D, SIZE, SIZE, 1, 1) }, |_, _, x, y| Some(integer(x, y).to_le_bytes().to_vec()));
    ints[3] = (ints[3] & !0xFFF) | 0x24C;
    let colors = texture(&mut g, Shape::new(sharps::image_type::TYPE_2D, SIZE, SIZE, 1, 1), |_, _, x, _| Some(vec![if x > SAMPLE_X { 255 } else { 0 }, 0, 0, 255]));
    let words = THREADS as usize * 8;
    let out = g.alloc(&asm::bytes(&vec![POISON; words]));
    let mut user = raw(out, (words * 4) as u32).to_vec();
    user.extend_from_slice(&ints);
    user.extend_from_slice(&LINEAR_SAMPLER);
    let mut program: Vec<u32> = colors.iter().enumerate().flat_map(|(k, &w)| [0xBE80_03FF | ((16 + k as u32) << 16), w]).collect();
    program.extend_from_slice(&CODE);
    dispatch(&mut g, &mut rig, &program, &user);
    let o = read_words(&g, &mut rig, out, words);
    let int_expected = [integer(SAMPLE_X, SAMPLE_Y) as u32, 1, 1, 1];
    let color_expected = [0.25f32, 0.0, 0.0, 1.0];
    for lane in 0..THREADS as usize {
        for c in 0..4 {
            assert_eq!(o[lane * 8 + c], int_expected[c], "32_SINT sample through a bilinear sampler shared with 8_8_8_8_UNORM: lane {lane} component {c}");
            let got = f32::from_bits(o[lane * 8 + 4 + c]);
            assert!((got - color_expected[c]).abs() <= 1.0 / 64.0, "8_8_8_8_UNORM sample through the shared sampler: lane {lane} component {c} is {got}, expected {}", color_expected[c]);
        }
    }
}
