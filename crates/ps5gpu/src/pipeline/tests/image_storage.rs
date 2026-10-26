// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Storage image accesses (atomics, integer stores and loads) through the
//! SPIR-V compiler on the device, against the reference's hardware-measured
//! driver execution tests of the same names (ImageAtomicsBitwise,
//! ImageStoreSint, ...).
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored image_storage_ref`.

use super::image_dims::*;
use super::*;

/// s[0:3] the input, s[4:7] the output (`out_words` a lane, poisoned),
/// s[8:15] the T#; `threads` lanes in waves of `wave`.
fn run_io(g: &mut Guest, rig: &mut Rig, code: &[u32], input: &[u32], out_words: usize, texture: &[u32; 8], threads: u32, wave: u32) -> Vec<u32> {
    let ib = g.alloc(&asm::bytes(input));
    let words = threads as usize * out_words;
    let ob = g.alloc(&asm::bytes(&vec![POISON; words]));
    let mut user = raw(ib, (input.len() * 4) as u32).to_vec();
    user.extend(raw(ob, (words * 4) as u32));
    user.extend_from_slice(texture);
    dispatch_waves(g, rig, code, user.as_slice(), threads, wave);
    read_words(g, rig, ob, words)
}

/// image_atomic_and / or / xor / umin / umax return the old texel and
/// store the result (wave32 and wave64).
#[test]
#[ignore]
fn image_storage_ref_atomics_bitwise() {
    const THREADS64: u32 = 64;
    const INPUTS: usize = 4;
    const RESULTS: usize = 16;
    const HEIGHT: u32 = 8;
    const CODE: [u32; 92] = [
        0x34020084, 0x34060086, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x7e100300, 0x7e120280, 0x7e140304, 0xf03c0108, 0x00020a08, 0x7e160305, 0xf0602108,
        0x00020b08, 0x7e180280, 0xf0442108, 0x00020c08, 0xbf8c3f70, 0x7e28030b, 0x7e2a030c, 0x7e120281, 0x7e140304, 0xf03c0108, 0x00020a08, 0x7e160305,
        0xf0642108, 0x00020b08, 0x7e180280, 0xf0442108, 0x00020c08, 0xbf8c3f70, 0x7e2c030b, 0x7e2e030c, 0x7e120282, 0x7e140304, 0xf03c0108, 0x00020a08,
        0x7e160305, 0xf0682108, 0x00020b08, 0x7e180280, 0xf0442108, 0x00020c08, 0xbf8c3f70, 0x7e30030b, 0x7e32030c, 0x7e120283, 0x7e140304, 0xf03c0108,
        0x00020a08, 0x7e160305, 0xf0542108, 0x00020b08, 0x7e180280, 0xf0442108, 0x00020c08, 0xbf8c3f70, 0x7e34030b, 0x7e36030c, 0x7e120284, 0x7e140304,
        0xf03c0108, 0x00020a08, 0x7e160305, 0xf05c2108, 0x00020b08, 0x7e180280, 0xf0442108, 0x00020c08, 0xbf8c3f70, 0x7e38030b, 0x7e3a030c, 0xe0701000,
        0x80011403, 0xe0701004, 0x80011503, 0xe0701008, 0x80011603, 0xe070100c, 0x80011703, 0xe0701010, 0x80011803, 0xe0701014, 0x80011903, 0xe0701018,
        0x80011a03, 0xe070101c, 0x80011b03, 0xe0701020, 0x80011c03, 0xe0701024, 0x80011d03, 0xbf810000,
    ];
    const EDGES: [[u32; 2]; 14] = [
        [0x0, 0x0],
        [0x0, 0xffffffff],
        [0xffffffff, 0x0],
        [0xffffffff, 0xffffffff],
        [0x7fffffff, 0x80000000],
        [0x80000000, 0x7fffffff],
        [0xffffffff, 0x1],
        [0x1, 0xffffffff],
        [0xaaaaaaaa, 0x55555555],
        [0xf0f0f0f0, 0xff00ff00],
        [0x5, 0x5],
        [0x4, 0x5],
        [0x6, 0x5],
        [0x80000000, 0x80000000],
    ];
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut input = vec![0u32; THREADS64 as usize * INPUTS];
    for tid in 0..THREADS64 as usize {
        for j in 0..2 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            input[tid * INPUTS + j] = (state >> 32) as u32;
        }
        if tid < EDGES.len() {
            input[tid * INPUTS] = EDGES[tid][0];
            input[tid * INPUTS + 1] = EDGES[tid][1];
        }
    }
    let names = ["image_atomic_and", "image_atomic_or", "image_atomic_xor", "image_atomic_umin", "image_atomic_umax"];
    for wave in [32, 64] {
        let mut g = Guest::new(16 << 20);
        let mut rig = rig(&g);
        let shape = Shape { format: 20, ..Shape::new(sharps::image_type::TYPE_2D, THREADS64, HEIGHT, 1, 1) };
        let t = texture(&mut g, shape, |_, _, _, _| Some(POISON.to_le_bytes().to_vec()));
        let out = run_io(&mut g, &mut rig, &CODE, &input, RESULTS, &t, THREADS64, wave);
        for tid in 0..THREADS64 as usize {
            let (a, b) = (input[tid * INPUTS], input[tid * INPUTS + 1]);
            let expected = [a & b, a | b, a ^ b, a.min(b), a.max(b)];
            for (j, name) in names.iter().enumerate() {
                let returned = out[tid * RESULTS + 2 * j];
                let stored = out[tid * RESULTS + 2 * j + 1];
                assert_eq!(returned, a, "{name} wave{wave}: thread {tid} returned {returned:#x}, expected {a:#x}");
                assert_eq!(stored, expected[j], "{name} wave{wave}: thread {tid} stored {stored:#x}, expected {:#x}", expected[j]);
            }
        }
    }
}

// ---- ImageTexelStatus ----

mod texel_status {
    use super::*;
    include!("image_texel_status_data.rs");
    include!("image_cache_bits_data.rs");

    /// DLC / GLC / SLC cache bits change nothing a program sees (stores,
    /// atomics, samples, gathers, resinfo, loads).
    #[test]
    #[ignore]
    fn image_storage_ref_cache_bits() {
        let out = run_status(
            &CACHE_STORAGE_CODE,
            &CACHE_STORAGE_INPUTS,
            |g| texture(g, Shape { format: 20, ..Shape::new(sharps::image_type::TYPE_2D, 32, 8, 1, 1) }, |_, _, _, _| Some(POISON.to_le_bytes().to_vec())),
            [0; 4],
        );
        check(&out, &CACHE_STORAGE_EXPECTED, &["image_store dlc", "image_store glc slc dlc"]);
        let out = run_status(
            &CACHE_SAMPLE_CODE,
            &CACHE_SAMPLE_INPUTS,
            |g| texture(g, Shape { format: 22, ..Shape::new(sharps::image_type::TYPE_2D, 64, 2, 1, 1) }, |_, _, x, y| Some(((x + 100 * y) as f32).to_le_bytes().to_vec())),
            [0, 0xFFF << 12, 1 << 26, 0],
        );
        let names = [
            "image_sample dlc",
            "image_sample glc slc dlc",
            "image_gather4 dlc x",
            "image_gather4 dlc y",
            "image_gather4 dlc z",
            "image_gather4 dlc w",
            "image_get_resinfo dlc width",
            "image_get_resinfo dlc height",
            "image_sample_l dlc",
            "image_load dlc",
        ];
        check(&out, &CACHE_SAMPLE_EXPECTED, &names);
    }

    const INPUTS: usize = 4;
    const RESULTS: usize = 16;

    /// s[0:3] the buffer (inputs, then results), s[4:11] the T#, s[12:15]
    /// the S#; returns the results.
    fn run_status(code: &[u32], inputs: &[u32], texture: impl FnOnce(&mut Guest) -> [u32; 8], sampler: [u32; 4]) -> Vec<u32> {
        let mut g = Guest::new(16 << 20);
        let mut rig = rig(&g);
        let t = texture(&mut g);
        let mut buffer = vec![POISON; THREADS as usize * (INPUTS + RESULTS)];
        for tid in 0..THREADS as usize {
            buffer[tid * INPUTS..tid * INPUTS + 3].copy_from_slice(&inputs[tid * 3..tid * 3 + 3]);
            buffer[tid * INPUTS + 3] = 0;
        }
        let b = g.alloc(&asm::bytes(&buffer));
        let mut user = raw(b, (buffer.len() * 4) as u32).to_vec();
        user.extend_from_slice(&t);
        user.extend_from_slice(&sampler);
        dispatch(&mut g, &mut rig, code, &user);
        read_words(&g, &mut rig, b, buffer.len())[THREADS as usize * INPUTS..].to_vec()
    }

    fn check(out: &[u32], expected: &[u32], names: &[&str]) {
        let n = names.len();
        for tid in 0..THREADS as usize {
            for j in 0..n {
                let (a, e) = (out[tid * RESULTS + j], expected[tid * n + j]);
                assert_eq!(a, e, "{}: thread {tid} is {a:#010x}, expected {e:#010x}", names[j]);
            }
        }
    }

    /// TFE / LWE stores, atomics and loads of a resident 32_UINT image: the
    /// data as without them, the status dword left as it was.
    #[test]
    #[ignore]
    fn image_storage_ref_texel_status_storage() {
        let out = run_status(
            &STORAGE_CODE,
            &STORAGE_INPUTS,
            |g| texture(g, Shape { format: 20, ..Shape::new(sharps::image_type::TYPE_2D, 32, 8, 1, 1) }, |_, _, _, _| Some(POISON.to_le_bytes().to_vec())),
            [0; 4],
        );
        check(&out, &STORAGE_EXPECTED, &["image_store tfe", "image_store tfe lwe", "image_store tfe data status", "image_store tfe lwe data status"]);
    }

    /// TFE / LWE samples, gathers, resinfo and loads (D16 too) of a
    /// resident 32_FLOAT texture: the data as without them, the status
    /// dword left as it was.
    #[test]
    #[ignore]
    fn image_storage_ref_texel_status_sample() {
        let out = run_status(
            &SAMPLE_CODE,
            &SAMPLE_INPUTS,
            |g| texture(g, Shape { format: 22, ..Shape::new(sharps::image_type::TYPE_2D, 64, 2, 1, 1) }, |_, _, x, y| Some(((x + 100 * y) as f32).to_le_bytes().to_vec())),
            [0, 0xFFF << 12, 1 << 26, 0],
        );
        let names = [
            "image_sample tfe lwe",
            "image_sample tfe lwe status",
            "image_gather4 tfe lwe x",
            "image_gather4 tfe lwe y",
            "image_gather4 tfe lwe z",
            "image_gather4 tfe lwe w",
            "image_gather4 tfe lwe status",
            "image_get_resinfo tfe lwe width",
            "image_get_resinfo tfe lwe height",
            "image_get_resinfo tfe lwe status",
            "image_sample_l tfe",
            "image_sample_l tfe status",
            "image_load tfe lwe",
            "image_load tfe lwe status",
            "image_load tfe d16",
            "image_load tfe d16 status",
        ];
        check(&out, &SAMPLE_EXPECTED, &names);
    }
}

/// Runs `code` in `groups` groups of 32 lanes (group id in s16, after the
/// 16 user SGPRs: s[0:3] the output, s[4:11] the T#, s[12:15] the S#);
/// returns `words` output dwords.
fn run_groups(g: &mut Guest, rig: &mut Rig, code: &[u32], texture: &[u32; 8], sampler: [u32; 4], groups: u32, words: usize) -> Vec<u32> {
    let out = g.alloc(&asm::bytes(&vec![POISON; words]));
    let mut user = raw(out, (words * 4) as u32).to_vec();
    user.extend_from_slice(texture);
    user.extend_from_slice(&sampler);
    let cs = g.program(code);
    let mut shader = stage(cs, &user);
    shader.rsrc1 = (0xC0 << 12) | (1 << 21);
    shader.rsrc2 |= 1 << 7;
    let d = ResolvedDispatch {
        packet: DispatchPacket { groups: [groups, 1, 1], dispatch_initiator: 1 | (1 << 15), ..DispatchPacket::default() },
        shader,
        rsrc3: 0,
        threads: [THREADS, 1, 1],
        start: [0; 3],
    };
    assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
    read_words(g, rig, out, words)
}

fn check_slot(o: &[u32], slot: usize, e: [u32; 4], what: &str) {
    for c in 0..4 {
        assert_eq!(o[slot * 4 + c], e[c], "{what}: result {slot} component {c} is {:#010x}, expected {:#010x}", o[slot * 4 + c], e[c]);
    }
}

/// 8_SINT textures read sign-extended: image_load (DMASK 0x1 and 0xF,
/// DST_SEL X,1,1,1), image_sample_lz (point) and image_gather4_lz
/// (through a bilinear sampler).
#[test]
#[ignore]
fn image_storage_ref_sint8() {
    const WIDTH: u32 = 128;
    const HEIGHT: u32 = 128;
    const TEXELS: u32 = WIDTH * HEIGHT;
    const POINT: [u32; 4] = [0x92, 0x00ff_f000, 0, 0];
    const LINEAR: [u32; 4] = [0x92, 0x00ff_f000, 0x0050_0000, 0];
    // v4 = s16 (the group id); texel group * 32 + tid at (x, y) in v[30:31];
    // image op v[10:13], v[30:31], s[4:11]; store at texel * 16 of s[0:3].
    let texel_code = |word0: u32| {
        [
            0x7e080210, 0x343c0885, 0x4a3c3d00, 0x34063c84, 0x2c3e3c87, 0x363c3cff, 0x0000007f, 0x7e140280, 0x7e160280, 0x7e180280, 0x7e1a0280, word0, 0x00010a1e,
            0xbf8c3f70, 0xe0781000, 0x80000a03, 0xbf810000,
        ]
    };
    let sampler_code = |word0: u32, u: u32, v: u32| [0x34060084, 0x7e2802ff, u, 0x7e2a02ff, v, word0, 0x00610a14, 0xbf8c3f70, 0xe0781000, 0x80000a03, 0xbf810000];
    let byte = |x: u32, y: u32| (0x80 + 37 * x + 11 * y) as u8;
    let signed = |x: u32, y: u32| byte(x, y) as i8 as i32 as u32;
    let mut g = Guest::new(32 << 20);
    let mut rig = rig(&g);
    let shape = Shape { format: 6, bpe: 1, ..Shape::new(sharps::image_type::TYPE_2D, WIDTH, HEIGHT, 1, 1) };
    let mut t = texture(&mut g, shape, |_, _, x, y| Some(vec![byte(x, y)]));
    t[3] = (t[3] & !0xFFF) | 0x24C;
    let o = run_groups(&mut g, &mut rig, &texel_code(0xf000_1108), &t, POINT, TEXELS / THREADS, TEXELS as usize * 4);
    for i in 0..TEXELS {
        check_slot(&o, i as usize, [signed(i % WIDTH, i / WIDTH), 0, 0, 0], "8_SINT image_load dmask:0x1");
    }
    let o = run_groups(&mut g, &mut rig, &texel_code(0xf000_1f08), &t, POINT, TEXELS / THREADS, TEXELS as usize * 4);
    for i in 0..TEXELS {
        check_slot(&o, i as usize, [signed(i % WIDTH, i / WIDTH), 1, 1, 1], "8_SINT image_load dmask:0xf X 1 1 1");
    }
    let o = run_groups(&mut g, &mut rig, &sampler_code(0xf09c_0f08, 0x3ce0_0000, 0x3d30_0000), &t, POINT, 1, THREADS as usize * 4);
    for lane in 0..THREADS as usize {
        check_slot(&o, lane, [signed(3, 5), 1, 1, 1], "8_SINT image_sample_lz with point filtering");
    }
    let o = run_groups(&mut g, &mut rig, &sampler_code(0xf11c_0108, 0x3c00_0000, 0x3c00_0000), &t, LINEAR, 1, THREADS as usize * 4);
    for lane in 0..THREADS as usize {
        check_slot(&o, lane, [signed(0, 1), signed(1, 1), signed(1, 0), signed(0, 0)], "8_SINT image_gather4_lz through a bilinear sampler");
    }
}

// ---- ImageConvertedUnorm ----

mod converted {
    use super::*;

    const LOAD_WIDTH: u32 = 2048;
    const LOAD_GROUPS: u32 = LOAD_WIDTH / THREADS;
    const UNORM: u32 = 30;
    const UINT: u32 = 34;
    const FLOAT: u32 = 43;
    const XYZ1: u32 = 0x3AC;
    const XZY1: u32 = 0x374;
    const ZYX1: u32 = 0x32E;
    const X011: u32 = 0x244;
    const YXZ1: u32 = 0x3A5;
    const DIVISION_ULPS: u32 = 3;
    const POINT: [u32; 4] = [0x92, 0x00ff_f000, 0, 0];
    const STORE_VALUES: [u32; 11] = [0x00000000, 0x3f800000, 0x3f000000, 0x3eaaaaab, 0x80000000, 0xbf800000, 0x40000000, 0x7f800000, 0x7fc00000, 0x000116c2, 0x3f7fbe77];

    /// The group id is in s16 (after 16 user SGPRs): the programs read it
    /// from s24, where the reference's 24 user SGPRs put it; `high` are
    /// SGPRs from s16 on, moved there after the copy.
    fn prologue(code: &[u32], high: &[u32]) -> Vec<u32> {
        let mut p = vec![0xBE98_0310];
        p.extend(high.iter().enumerate().flat_map(|(k, &w)| [0xBE80_03FF | ((16 + k as u32) << 16), w]));
        p.extend_from_slice(code);
        p
    }

    fn load_code(dmask: u32) -> Vec<u32> {
        vec![0x7e080218, 0x343c0885, 0x4a3c3d00, 0x34063c84, 0x7e3e0280, 0x7e140280, 0x7e160280, 0x7e180280, 0x7e1a0280, 0xf000_1008 | (dmask << 8), 0x00010a1e, 0xbf8c3f70, 0xe0781000, 0x80000a03, 0xbf810000]
    }

    fn store_code(dmask: u32) -> Vec<u32> {
        vec![0x34060084, 0xe0381000, 0x80030a03, 0x7e3c0300, 0x7e3e0280, 0xbf8c3f70, 0xf020_1008 | (dmask << 8), 0x00010a1e, 0xbf810000]
    }

    fn descriptor(base: u64, format: u32, swizzle: u32, width: u32) -> [u32; 8] {
        let mut d = Shape { format, ..Shape::new(sharps::image_type::TYPE_2D, width, 1, 1, 1) }.descriptor(base);
        d[3] = (d[3] & !0xFFF) | swizzle;
        d
    }

    fn load_texel(index: u32) -> u32 {
        index | ((2047 - index) << 11) | ((index & 1023) << 22)
    }

    fn field(texel: u32, c: u32) -> u32 {
        if c == 2 {
            texel >> 22
        } else {
            (texel >> (c * 11)) & 2047
        }
    }

    /// (bits, ULPs) of component `c` of a texel read through `swizzle`.
    fn selected(texel: u32, swizzle: u32, c: u32, unorm: bool) -> (u32, u32) {
        let s = (swizzle >> (c * 3)) & 7;
        if s == 0 {
            return (0, 0);
        }
        if s == 1 {
            return (if unorm { 0x3f80_0000 } else { 1 }, 0);
        }
        let f = field(texel, s - 4);
        let max = if s == 6 { 1023 } else { 2047 };
        if !unorm {
            return (f, 0);
        }
        (((f as f64 / max as f64) as f32).to_bits(), if f == 0 || f == max { 0 } else { DIVISION_ULPS })
    }

    /// Runs `code` (group id in s24, `high` from s16) over `groups` groups:
    /// s[0:3] the output (4 dwords a lane), s[4:11] `texture`, s[12:15]
    /// `extra`.
    fn run(g: &mut Guest, rig: &mut Rig, code: &[u32], texture: &[u32; 8], extra: [u32; 4], groups: u32, high: &[u32]) -> Vec<u32> {
        let words = (groups * THREADS) as usize * 4;
        let out = g.alloc(&asm::bytes(&vec![POISON; words]));
        let mut user = raw(out, (words * 4) as u32).to_vec();
        user.extend_from_slice(texture);
        user.extend_from_slice(&extra);
        let cs = g.program(&prologue(code, high));
        let mut shader = stage(cs, &user);
        shader.rsrc1 = (0xC0 << 12) | (1 << 21);
        shader.rsrc2 |= 1 << 7;
        let d = ResolvedDispatch {
            packet: DispatchPacket { groups: [groups, 1, 1], dispatch_initiator: 1 | (1 << 15), ..DispatchPacket::default() },
            shader,
            rsrc3: 0,
            threads: [THREADS, 1, 1],
            start: [0; 3],
        };
        assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
        read_words(g, rig, out, words)
    }

    fn load_texels(g: &mut Guest) -> u64 {
        g.alloc(&asm::bytes(&(0..LOAD_WIDTH).map(load_texel).collect::<Vec<u32>>()))
    }

    /// image_load of 11_11_10 UNORM (UNORM components, ±3 ULP of the
    /// division) and UINT through DST_SEL permutations.
    #[test]
    #[ignore]
    fn image_storage_ref_converted_loads() {
        let mut g = Guest::new(16 << 20);
        let mut rig = rig(&g);
        let base = load_texels(&mut g);
        for (dmask, format, swizzle, what) in [
            (0xF, UNORM, XYZ1, "image_load dmask:0xf X Y Z 1"),
            (0x3, UNORM, XZY1, "image_load dmask:0x3 X Z Y 1"),
            (0xF, UNORM, ZYX1, "image_load dmask:0xf Z Y X 1"),
            (0xF, UNORM, X011, "image_load dmask:0xf X 0 1 1"),
            (0x7, UINT, YXZ1, "image_load of 11_11_10_UINT dmask:0x7 Y X Z 1"),
        ] {
            let out = run(&mut g, &mut rig, &load_code(dmask), &descriptor(base, format, swizzle, LOAD_WIDTH), [0; 4], LOAD_GROUPS, &[]);
            for index in 0..LOAD_WIDTH {
                let mut slot = 0;
                for c in 0..4 {
                    if (dmask >> c) & 1 == 0 {
                        continue;
                    }
                    let (bits, ulps) = selected(load_texel(index), swizzle, c, format == UNORM);
                    let actual = out[index as usize * 4 + slot];
                    slot += 1;
                    assert!(actual.abs_diff(bits) <= ulps, "{what}: texel {index} component {c} is {actual:#010x}, expected {bits:#010x} within {ulps} ULP");
                }
            }
        }
    }

    /// image_load then image_store of 11_11_10 UNORM keeps every texel.
    #[test]
    #[ignore]
    fn image_storage_ref_converted_round_trip() {
        const ROUND_TRIP: [u32; 10] = [0x7e080218, 0x343c0885, 0x4a3c3d00, 0x7e3e0280, 0xf0001708, 0x00010a1e, 0xbf8c3f70, 0xf0201708, 0x00040a1e, 0xbf810000];
        let mut g = Guest::new(16 << 20);
        let mut rig = rig(&g);
        let base = load_texels(&mut g);
        let target = g.alloc(&vec![0xA5u8; LOAD_WIDTH as usize * 4]);
        let target_t = descriptor(target, UNORM, XYZ1, LOAD_WIDTH);
        run(&mut g, &mut rig, &ROUND_TRIP, &descriptor(base, UNORM, XYZ1, LOAD_WIDTH), [0; 4], LOAD_GROUPS, &target_t);
        let bytes = written_back(&g, &mut rig, target, LOAD_WIDTH as u64 * 4);
        for index in 0..LOAD_WIDTH as usize {
            let actual = u32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap());
            assert_eq!(actual, load_texel(index as u32), "image_load then image_store: texel {index} is {actual:#010x}");
        }
    }

    fn store_field(bits: u32, max: u32) -> u32 {
        let v = f32::from_bits(bits);
        if !(v > 0.0) {
            return 0;
        }
        if v >= 1.0 {
            return max;
        }
        (v as f64 * max as f64).round_ties_even() as u32
    }

    /// image_store of 11_11_10 UNORM: each component clamped to [0, 1] and
    /// rounded to nearest even (NaN and -0 store 0); unselected ones 0.
    #[test]
    #[ignore]
    fn image_storage_ref_converted_stores() {
        for (dmask, what) in [(0x7, "image_store dmask:0x7"), (0x5, "image_store dmask:0x5")] {
            let mut g = Guest::new(16 << 20);
            let mut rig = rig(&g);
            let texels = g.alloc(&vec![0xA5u8; THREADS as usize * 4]);
            let mut input = vec![0u32; THREADS as usize * 4];
            let mut expected = vec![0u32; THREADS as usize];
            for tid in 0..THREADS as usize {
                let values = [STORE_VALUES[tid % 11], STORE_VALUES[(tid + 4) % 11], STORE_VALUES[(tid + 8) % 11]];
                let mut slot = 0;
                for c in 0..3 {
                    if (dmask >> c) & 1 == 0 {
                        continue;
                    }
                    input[tid * 4 + slot] = values[c];
                    slot += 1;
                    expected[tid] |= store_field(values[c], if c == 2 { 1023 } else { 2047 }) << (c * 11);
                }
            }
            let ib = g.alloc(&asm::bytes(&input));
            run(&mut g, &mut rig, &store_code(dmask), &descriptor(texels, UNORM, XYZ1, THREADS), raw(ib, (input.len() * 4) as u32), 1, &[]);
            let bytes = written_back(&g, &mut rig, texels, THREADS as u64 * 4);
            for tid in 0..THREADS as usize {
                let actual = u32::from_le_bytes(bytes[tid * 4..tid * 4 + 4].try_into().unwrap());
                assert_eq!(actual, expected[tid], "{what}: texel {tid} is {actual:#010x}, expected {:#010x}", expected[tid]);
            }
        }
    }

    /// Point samples and gathers of 11_11_10 UINT unpack the words they
    /// read: DST_SEL Z,Y,X,W reads (Z, Y, X, X) (a missing W repeats X),
    /// the gather of Y reads the Y field of the footprint's texels.
    #[test]
    #[ignore]
    fn image_storage_ref_converted_uint_sample_and_gather() {
        const ZYXW: u32 = 0xF2E;
        // v20 = u, v21 = 0.5; image op v[10:13], v[20:21], s[4:11],
        // s[12:15]; store at tid * 16 of s[0:3].
        let code = |word0: u32, u: f32| vec![0x34060084, 0x7e2802ff, u.to_bits(), 0x7e2a02ff, 0x3f00_0000, word0, 0x00610a14, 0xbf8c3f70, 0xe0781000, 0x80000a03, 0xbf810000];
        let mut g = Guest::new(16 << 20);
        let mut rig = rig(&g);
        let base = g.alloc(&asm::bytes(&(0..THREADS).map(load_texel).collect::<Vec<u32>>()));
        let t = descriptor(base, UINT, ZYXW, THREADS);
        // Texel 5 (its centre).
        let o = run(&mut g, &mut rig, &code(0xf09c_0f08, 5.5 / THREADS as f32), &t, POINT, 1, &[]);
        let texel = load_texel(5);
        let e = [field(texel, 2), field(texel, 1), field(texel, 0), field(texel, 0)];
        for tid in 0..THREADS as usize {
            assert_eq!(o[tid * 4..tid * 4 + 4], e, "image_sample_lz of 11_11_10 UINT through Z,Y,X,W: thread {tid}");
        }
        // Between texels 1 and 2: footprint (x0, x1) = (1, 2), one row.
        let o = run(&mut g, &mut rig, &code(0xf11c_0208, 2.0 / THREADS as f32), &t, POINT, 1, &[]);
        let (y1, y2) = (field(load_texel(1), 1), field(load_texel(2), 1));
        for tid in 0..THREADS as usize {
            assert_eq!(o[tid * 4..tid * 4 + 4], [y1, y2, y2, y1], "image_gather4_lz (Y) of 11_11_10 UINT: thread {tid}");
        }
    }

    /// 5_6_5, 1_5_5_5 and 4_4_4_4 UNORM: each component's field in its
    /// place (X lowest), loaded and point-sampled through X,Y,Z,W and
    /// Z,Y,X,W; 5_6_5 reads its X for W.
    #[test]
    #[ignore]
    fn image_storage_ref_packed_16_load() {
        const WIDTH: u32 = 256;
        const ONE: u32 = 0x3f80_0000;
        const FORMATS: [(u32, [u16; 4], &str); 3] =
            [(133, [0x001f, 0x07e0, 0xf800, 0x0000], "5_6_5_UNORM"), (134, [0x001f, 0x03e0, 0x7c00, 0x8000], "1_5_5_5_UNORM"), (136, [0x000f, 0x00f0, 0x0f00, 0xf000], "4_4_4_4_UNORM")];
        let sample_lz = [0x34060084, 0x7e280280, 0x7e2a0280, 0xf09c0f08, 0x00610a14, 0xbf8c3f70, 0xe0781000, 0x80000a03, 0xbf810000];
        let pattern = |x: u32| (x + 1) & 15;
        for (format, masks, name) in FORMATS {
            for (swizzle, sw_name) in [(0xFAC, "X Y Z W"), (0xF2E, "Z Y X W")] {
                let mut g = Guest::new(16 << 20);
                let mut rig = rig(&g);
                let texels: Vec<u16> = (0..WIDTH).map(|x| (0..4).filter(|c| (pattern(x) >> c) & 1 != 0).fold(0u16, |t, c| t | masks[c as usize])).collect();
                let base = g.alloc(&texels.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>());
                let mut t = Shape { format, bpe: 2, ..Shape::new(sharps::image_type::TYPE_2D, WIDTH, 1, 1, 1) }.descriptor(base);
                t[3] = (t[3] & !0xFFF) | swizzle;
                let verify = |o: &[u32], count: u32, sampled: bool, what: &str| {
                    for i in 0..count {
                        let x = if sampled { 0 } else { i };
                        let mut ch: Vec<u32> = (0..4).map(|c| if masks[c] != 0 && (pattern(x) >> c) & 1 != 0 { ONE } else { 0 }).collect();
                        if masks[3] == 0 {
                            ch[3] = ch[0];
                        }
                        for c in 0..4 {
                            let s = (swizzle >> (c * 3)) & 7;
                            let e = match s {
                                0 => 0,
                                1 => ONE,
                                _ => ch[(s - 4) as usize],
                            };
                            assert_eq!(o[(i * 4 + c) as usize], e, "{name} {what} {sw_name}: {} {i} component {c}", if sampled { "thread" } else { "texel" });
                        }
                    }
                };
                let o = run(&mut g, &mut rig, &load_code(0xF), &t, POINT, WIDTH / THREADS, &[]);
                verify(&o, WIDTH, false, "image_load");
                let o = run(&mut g, &mut rig, &sample_lz, &t, POINT, 1, &[]);
                verify(&o, THREADS, true, "image_sample_lz");
            }
        }
    }

    /// Bilinear image_sample_lz of an 11_11_10 UNORM texture (no host
    /// format: filtered in the shader from its gathered words) at texel
    /// centres, between texels and past the edges (clamped): each field
    /// weighted bilinearly over its maximum; W reads 1.
    #[test]
    #[ignore]
    fn image_storage_ref_converted_unorm_sample_lz() {
        const W: u32 = 8;
        const H: u32 = 4;
        const BILINEAR_EDGE: [u32; 4] = [0x92, 0xFFF << 12, (1 << 20) | (1 << 22), 0];
        let word = |x: u32, y: u32| {
            let r = (x * 263 + y * 97 + 5) & 0x7FF;
            let g = (2047 - x * 151 - y * 31) & 0x7FF;
            let b = (x * 71 + y * 211 + 3) & 0x3FF;
            r | (g << 11) | (b << 22)
        };
        // v20 = u, v21 = v from the lane's input; image_sample_lz
        // v[10:13], v[20:21], s[4:11], s[12:15]; store at tid * 16.
        let code = [0x34060084, 0xe0381000, 0x80031403, 0xbf8c3f70, 0xf09c0f08, 0x00610a14, 0xbf8c3f70, 0xe0781000, 0x80000a03, 0xbf810000];
        let points: Vec<(f32, f32)> = (0..THREADS)
            .map(|t| {
                let (x, y) = (t % W, (t / W) % H);
                let (du, dv) = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.75), (-0.75, 0.25)][(t % 4) as usize];
                ((x as f32 + 0.5 + du) / W as f32, (y as f32 + 0.5 + dv) / H as f32)
            })
            .collect();
        let expected = |u: f32, v: f32, c: u32| -> f64 {
            let (fx, fy) = (u as f64 * W as f64 - 0.5, v as f64 * H as f64 - 0.5);
            let (x0, y0) = (fx.floor(), fy.floor());
            let (wx, wy) = (fx - x0, fy - y0);
            let field = |x: f64, y: f64| {
                let (cx, cy) = ((x as i32).clamp(0, W as i32 - 1) as u32, (y as i32).clamp(0, H as i32 - 1) as u32);
                let w = word(cx, cy);
                let (shift, bits) = [(0, 11), (11, 11), (22, 10)][c as usize];
                ((w >> shift) & ((1 << bits) - 1)) as f64 / ((1u32 << bits) - 1) as f64
            };
            field(x0, y0) * (1.0 - wx) * (1.0 - wy) + field(x0 + 1.0, y0) * wx * (1.0 - wy) + field(x0, y0 + 1.0) * (1.0 - wx) * wy + field(x0 + 1.0, y0 + 1.0) * wx * wy
        };
        // image_sample_lz, and image_sample (filtered from level 0 too).
        for (op_word, op_name) in [(0xf09c0f08u32, "image_sample_lz"), (0xf0800f08, "image_sample")] {
        for (swizzle, sw_name) in [(XYZ1, "X Y Z 1"), (ZYX1, "Z Y X 1")] {
            let mut g = Guest::new(16 << 20);
            let mut rig = rig(&g);
            let mut t = texture(&mut g, Shape { format: UNORM, ..Shape::new(sharps::image_type::TYPE_2D, W, H, 1, 1) }, |_, _, x, y| Some(word(x, y).to_le_bytes().to_vec()));
            t[3] = (t[3] & !0xFFF) | swizzle;
            let input: Vec<u32> = points.iter().flat_map(|&(u, v)| [u.to_bits(), v.to_bits(), 0, 0]).collect();
            let ib = g.alloc(&asm::bytes(&input));
            // s[12:15] the input buffer: the S# moves to s[16:19].
            let sample_code: Vec<u32> = code.iter().map(|&w| if w == 0x00610a14 { 0x00810a14 } else if w == 0xf09c0f08 { op_word } else { w }).collect();
            let o = run(&mut g, &mut rig, &sample_code, &t, raw(ib, (input.len() * 4) as u32), 1, &BILINEAR_EDGE);
            for (tid, &(u, v)) in points.iter().enumerate() {
                for c in 0..4u32 {
                    let s = (swizzle >> (c * 3)) & 7;
                    let e = if s == 1 { 1.0 } else { expected(u, v, s - 4) };
                    let a = f32::from_bits(o[tid * 4 + c as usize]) as f64;
                    assert!((a - e).abs() <= 2e-5, "11_11_10 UNORM {op_name} {sw_name}: thread {tid} (u {u}, v {v}) component {c} is {a}, expected {e}");
                }
            }
        }
        }
    }

    const FLOAT_VALUES: [f32; 11] = [0.0, 1.0, 0.5, 2.0, 0.25, 3.0, 1.5, 10.0, 0.75, 6.5, 0.125];
    const FLOAT_STORE_EDGES: [[u32; 4]; 7] = [
        [0xbf800000, 0x80000000, 0x7fc00000, 0xfc000000],
        [0x7f800000, 0x7149f2ca, 0x477fff00, 0xf7feffe0],
        [0x3f830000, 0x3f818000, 0x35800000, 0x002f01e0],
        [0x00000001, 0xff800000, 0xffc00000, 0xfc000000],
        [0x477c0000, 0x4788b800, 0x33000000, 0x001effdf],
        [0x36400000, 0x477e0000, 0x3dcccccd, 0x5cdefc01],
        [0x7f800001, 0x7f810000, 0x7fa00000, 0xfa1f07e1],
    ];
    const FLOAT_LOAD_EDGES: [[u32; 4]; 3] = [
        [0x3e0 | (0x7c0 << 10) | (0x03f << 21), 0x7f800000, 0x7f800000, 0x387c0000],
        [0x001 | (0x041 << 10) | (0x7bf << 21), 0x36000000, 0x38820000, 0x477e0000],
        [0x3df | (0x001 << 10), 0x477c0000, 0x35800000, 0x00000000],
    ];

    fn small_float(v: f32, mantissa_bits: u32) -> u32 {
        if v == 0.0 {
            return 0;
        }
        let b = v.to_bits();
        let exponent = ((b >> 23) & 0xFF) - 127 + 15;
        (exponent << mantissa_bits) | ((b & 0x7F_FFFF) >> (23 - mantissa_bits))
    }

    fn float_texel(i: usize) -> u32 {
        small_float(FLOAT_VALUES[i % 11], 5) | (small_float(FLOAT_VALUES[(i + 4) % 11], 6) << 10) | (small_float(FLOAT_VALUES[(i + 8) % 11], 6) << 21)
    }

    /// 10_11_11 FLOAT: image_store rounds and flushes as measured (edges),
    /// image_sample_lz of the stored texels before they reach memory, then
    /// image_load (denormals, infinities, NaN edges) and a sample of the
    /// loaded memory.
    #[test]
    #[ignore]
    fn image_storage_ref_converted_float() {
        let mut g = Guest::new(16 << 20);
        let mut rig = rig(&g);
        let texels = g.alloc(&vec![0xA5u8; THREADS as usize * 4]);
        let mut input = vec![0u32; THREADS as usize * 4];
        for tid in 0..THREADS as usize {
            for c in 0..3 {
                input[tid * 4 + c] = if tid < FLOAT_STORE_EDGES.len() { FLOAT_STORE_EDGES[tid][c] } else { FLOAT_VALUES[(tid + 4 * c) % 11].to_bits() };
            }
        }
        let ib = g.alloc(&asm::bytes(&input));
        let t = descriptor(texels, FLOAT, XYZ1, THREADS);
        run(&mut g, &mut rig, &store_code(0x7), &t, raw(ib, (input.len() * 4) as u32), 1, &[]);
        let bytes = written_back(&g, &mut rig, texels, THREADS as u64 * 4);
        for tid in 0..THREADS as usize {
            let actual = u32::from_le_bytes(bytes[tid * 4..tid * 4 + 4].try_into().unwrap());
            let e = if tid < FLOAT_STORE_EDGES.len() { FLOAT_STORE_EDGES[tid][3] } else { float_texel(tid) };
            assert_eq!(actual, e, "image_store of 10_11_11 FLOAT: texel {tid} is {actual:#010x}, expected {e:#010x}");
        }
        // A sample of the stored texels (texel (0, 0): the first edge row).
        let sample_lz = [0x34060084, 0x7e280280, 0x7e2a0280, 0xf09c0f08, 0x00610a14, 0xbf8c3f70, 0xe0781000, 0x80000a03, 0xbf810000];
        let check_sample = |o: &[u32], e: [u32; 4], what: &str| {
            for tid in 0..THREADS as usize {
                for c in 0..4 {
                    let a = o[tid * 4 + c];
                    let nan = e[c] & 0x7f80_0000 == 0x7f80_0000 && e[c] & 0x7f_ffff != 0;
                    assert!(if nan { f32::from_bits(a).is_nan() } else { a == e[c] }, "{what}: thread {tid} component {c} is {a:#010x}, expected {:#010x}", e[c]);
                }
            }
        };
        let o = run(&mut g, &mut rig, &sample_lz, &t, POINT, 1, &[]);
        check_sample(&o, [0, 0, 0x7fc0_0000, 0x3f80_0000], "image_sample_lz of stored 10_11_11 FLOAT texels");
        let words: Vec<u32> = (0..LOAD_WIDTH as usize).map(|i| if i < FLOAT_LOAD_EDGES.len() { FLOAT_LOAD_EDGES[i][0] } else { float_texel(i) }).collect();
        let loads = g.alloc(&asm::bytes(&words));
        let o = run(&mut g, &mut rig, &load_code(0xF), &descriptor(loads, FLOAT, XYZ1, LOAD_WIDTH), [0; 4], LOAD_GROUPS, &[]);
        for i in 0..LOAD_WIDTH as usize {
            let e = if i < FLOAT_LOAD_EDGES.len() {
                [FLOAT_LOAD_EDGES[i][1], FLOAT_LOAD_EDGES[i][2], FLOAT_LOAD_EDGES[i][3], 0x3f80_0000]
            } else {
                [FLOAT_VALUES[i % 11].to_bits(), FLOAT_VALUES[(i + 4) % 11].to_bits(), FLOAT_VALUES[(i + 8) % 11].to_bits(), 0x3f80_0000]
            };
            for c in 0..4 {
                assert_eq!(o[i * 4 + c], e[c], "image_load of 10_11_11 FLOAT: texel {i} component {c} is {:#010x}, expected {:#010x}", o[i * 4 + c], e[c]);
            }
        }
        let o = run(&mut g, &mut rig, &sample_lz, &descriptor(loads, FLOAT, XYZ1, THREADS), POINT, 1, &[]);
        check_sample(&o, [0x7f80_0000, 0x7f80_0000, 0x387c_0000, 0x3f80_0000], "image_sample_lz of 10_11_11 FLOAT");
    }
}

// ---- ImageStoreSint ----

mod store_sint {
    use super::*;

    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 8;
    const TEXEL_BYTES: usize = 8192;
    const IDENTITY: u32 = 0xFAC;
    const TWO_CHANNEL: u32 = 0x22C;
    const TITLE: u32 = 0x24C;
    /// v[20:23] = the lane's values; (x, y) = (tid, 0) stored DMASK 0xF,
    /// then (tid, 1) DMASK 0x5.
    const STORE_CODE: [u32; 12] = [0x34020084, 0xe0381000, 0x80001401, 0xbf8c3f70, 0x7e3c0300, 0x7e3e0280, 0xf0201f08, 0x0001141e, 0x7e3e0281, 0xf0201508, 0x0001141e, 0xbf810000];
    const TITLE_CODE: [u32; 9] = [0x34020084, 0xe0381000, 0x80001401, 0xbf8c3f70, 0x7e3c0300, 0x7e3e0280, 0xf0202108, 0x0001141e, 0xbf810000];
    const LOAD_CODE: [u32; 9] = [0x34020084, 0x7e3c0300, 0x7e3e0280, 0xf0001f08, 0x0001141e, 0xbf8c3f70, 0xe0781000, 0x80001401, 0xbf810000];
    const D16_CODE: [u32; 12] = [0x34020084, 0xe0381000, 0x80001401, 0xbf8c3f70, 0x7e3c0300, 0x7e3e0280, 0xf0201f08, 0x8001141e, 0x7e3e0281, 0xf0201508, 0x8001141e, 0xbf810000];
    const ATOMIC_D16_CODE: [u32; 11] = [0x34020084, 0xe0381000, 0x80001401, 0xbf8c3f70, 0x7e3c0300, 0x7e3e0280, 0xf0441108, 0x0001141e, 0xf0201108, 0x8001141e, 0xbf810000];
    const EDGES: [u32; 24] = [
        0x00000000, 0x00000001, 0xffffffff, 0x0000007f, 0x00000080, 0xffffff80, 0xffffff7f, 0x000000ff, 0x00000100, 0x00007fff, 0x00008000, 0xffff8000,
        0xffff7fff, 0x0000ffff, 0x00010000, 0xffff0000, 0x7fffffff, 0x80000000, 0x12345678, 0xedcba988, 0x00007ffe, 0xffff8001, 0x0000007e, 0xffffff81,
    ];
    /// (name, format, components, bits)
    const FORMATS: [(&str, u32, u32, u32); 8] = [
        ("16_SINT", 12, 1, 16),
        ("8_8_SINT", 19, 2, 8),
        ("32_SINT", 21, 1, 32),
        ("16_16_SINT", 28, 2, 16),
        ("8_8_8_8_SINT", 61, 4, 8),
        ("32_32_SINT", 63, 2, 32),
        ("16_16_16_16_SINT", 70, 4, 16),
        ("32_32_32_32_SINT", 76, 4, 32),
    ];

    fn value(tid: u32, index: u32) -> u32 {
        if (tid as usize) < EDGES.len() {
            return EDGES[(tid as usize + 7 * index as usize) % EDGES.len()];
        }
        let mixed = (tid * 4 + index + 1).wrapping_mul(0x9e37_79b1);
        ((mixed as i32) >> ((tid * 4 + index) % 25)) as u32
    }

    fn saturate(v: u32, bits: u32) -> u32 {
        if bits == 32 {
            return v;
        }
        let high = (1i64 << (bits - 1)) - 1;
        ((v as i32 as i64).clamp(-high - 1, high) as u32) & ((1u32 << bits) - 1)
    }

    fn sign_extend(v: u32, bits: u32) -> u32 {
        if bits == 32 {
            v
        } else {
            (((v << (32 - bits)) as i32) >> (32 - bits)) as u32
        }
    }

    fn stored(bits: u32, tid: u32, dmask: u32, c: u32, d16: bool) -> u32 {
        if (dmask >> c) & 1 == 0 {
            return 0;
        }
        let index: u32 = (0..c).map(|l| (dmask >> l) & 1).sum();
        if d16 {
            let half = (value(tid, index / 2) >> (16 * (index & 1))) & 0xFFFF;
            saturate(sign_extend(half, 16), bits)
        } else {
            saturate(value(tid, index), bits)
        }
    }

    fn shape(format: u32, components: u32, bits: u32) -> Shape {
        Shape { format, bpe: components * bits / 8, ..Shape::new(sharps::image_type::TYPE_2D, WIDTH, HEIGHT, 1, 1) }
    }

    fn t_sharp(s: Shape, base: u64, swizzle: u32) -> [u32; 8] {
        let mut d = s.descriptor(base);
        d[3] = (d[3] & !0xFFF) | swizzle;
        d
    }

    fn initial() -> Vec<u8> {
        initial_bytes(TEXEL_BYTES)
    }

    /// Stores `code` into a fresh texture and returns its bytes.
    fn store(code: &[u32], s: Shape, swizzle: u32) -> Vec<u8> {
        let mut g = Guest::new(16 << 20);
        let mut rig = rig(&g);
        let base = g.alloc(&initial());
        let input: Vec<u32> = (0..THREADS).flat_map(|tid| (0..4).map(move |i| value(tid, i))).collect();
        store_dispatch(&mut g, &mut rig, code, &input, &t_sharp(s, base, swizzle));
        written_back(&g, &mut rig, base, TEXEL_BYTES as u64)
    }

    /// Every texel of the texture against `writes` ((row, DMASK)) over
    /// the initial bytes.
    fn check(bytes: &[u8], name: &str, format: (u32, u32, u32), writes: &[(u32, u32)], d16: bool, what: &str) {
        let (f, components, bits) = format;
        let layout = shape(f, components, bits).layout();
        let mut expected = initial();
        let size = (bits / 8) as usize;
        for tid in 0..THREADS {
            for &(row, dmask) in writes {
                let o = layout.element_offset(0, 0, tid, row).unwrap() as usize;
                for c in 0..components {
                    let v = stored(bits, tid, dmask, c, d16);
                    expected[o + c as usize * size..o + (c as usize + 1) * size].copy_from_slice(&v.to_le_bytes()[..size]);
                }
            }
        }
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let o = layout.element_offset(0, 0, x, y).unwrap() as usize;
                for c in 0..components as usize {
                    let at = o + c * size;
                    assert_eq!(bytes[at..at + size], expected[at..at + size], "{what} {name}: texel ({x}, {y}) component {c}");
                }
            }
        }
    }

    /// Stores of signed formats saturate each field to its width (DMASK
    /// 0xF, then 0x5 with the unselected fields 0); a load after them reads
    /// the fields sign-extended.
    #[test]
    #[ignore]
    fn image_storage_ref_store_sint() {
        for (name, format, components, bits) in FORMATS {
            let s = shape(format, components, bits);
            let swizzle = if components == 2 { TWO_CHANNEL } else { IDENTITY };
            let bytes = store(&STORE_CODE, s, swizzle);
            check(&bytes, name, (format, components, bits), &[(0, 0xF), (1, 0x5)], false, "image_store");
            // The load after the store.
            let mut g = Guest::new(16 << 20);
            let mut rig = rig(&g);
            let base = g.alloc(&bytes);
            let out = g.alloc(&asm::bytes(&vec![POISON; THREADS as usize * 4]));
            let mut user = raw(out, THREADS * 16).to_vec();
            user.extend_from_slice(&t_sharp(s, base, swizzle));
            dispatch(&mut g, &mut rig, &LOAD_CODE, &user);
            let o = read_words(&g, &mut rig, out, THREADS as usize * 4);
            for tid in 0..THREADS {
                for c in 0..components {
                    let actual = o[(tid * 4 + c) as usize];
                    let wanted = sign_extend(stored(bits, tid, 0xF, c, false), bits);
                    assert_eq!(actual, wanted, "image_load after image_store of {name}: thread {tid} component {c} is {actual:#010x}, expected {wanted:#010x}");
                }
            }
        }
    }

    /// DMASK 0x1 GLC stores through DST_SEL X,1,1,1; D16 stores (the
    /// halves sign-extended, then saturated); a D16 store after an
    /// atomic add.
    #[test]
    #[ignore]
    fn image_storage_ref_store_sint_title_and_d16() {
        let (name, format, components, bits) = FORMATS[0];
        let bytes = store(&TITLE_CODE, shape(format, components, bits), TITLE);
        check(&bytes, name, (format, components, bits), &[(0, 0x1)], false, "image_store dmask:0x1 glc with DST_SEL X,1,1,1");
        for (name, format, components, bits) in FORMATS {
            let swizzle = if components == 2 { TWO_CHANNEL } else { IDENTITY };
            let bytes = store(&D16_CODE, shape(format, components, bits), swizzle);
            check(&bytes, name, (format, components, bits), &[(0, 0xF), (1, 0x5)], true, "image_store d16");
        }
        let (name, format, components, bits) = FORMATS[2];
        let bytes = store(&ATOMIC_D16_CODE, shape(format, components, bits), IDENTITY);
        check(&bytes, name, (format, components, bits), &[(0, 0x1)], true, "image_store d16 after image_atomic_add");
    }
}
