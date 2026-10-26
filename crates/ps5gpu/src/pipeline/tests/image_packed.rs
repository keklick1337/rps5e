// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! MIMG raw-texel (PCK), multi-texel (BY2 / BY4, PCK2 / PCK4) and packed
//! horizontal gather instructions through the SPIR-V compiler on the
//! device, against the reference's hardware-measured driver execution
//! tests of the same names (ImageLoadPacked, ImageStorePacked, ImagePckN,
//! ImageBy, ImagePackedHorizontalGather).
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored image_packed_ref`.

use super::image_dims::*;
use super::*;

/// A 2D texture of `width` x `height` with `levels` and `bpe`-byte texels
/// whose bytes `byte(level, x, y, i)` gives (the rest 0xEE).
fn bytes_texture(g: &mut Guest, format: u32, bpe: u32, width: u32, height: u32, levels: u32, swizzle: u32, byte: impl Fn(u32, u32, u32, u32) -> u8) -> ([u32; 8], u64, Shape) {
    let s = Shape { format, bpe, ..Shape::new(sharps::image_type::TYPE_2D, width, height, 1, levels) };
    let layout = s.layout();
    let mut bytes = vec![0xEEu8; layout.guest_bytes as usize];
    for level in 0..levels {
        let m = layout.mips[level as usize];
        for y in 0..m.height {
            for x in 0..m.width {
                let o = layout.element_offset(level, 0, x, y).unwrap() as usize;
                for i in 0..bpe {
                    bytes[o + i as usize] = byte(level, x, y, i);
                }
            }
        }
    }
    let base = g.alloc(&bytes);
    let mut d = s.descriptor(base);
    d[3] = (d[3] & !0xFFF) | swizzle;
    (d, base, s)
}

// ---- ImageLoadPacked ----

mod load_packed {
    use super::*;

    const WIDTH: u32 = 32;
    const LEVELS: u32 = 2;
    const RESULTS: usize = 17;
    /// pck dmask 0xF, pck_sgn 0xF, pck 0xA, pck_sgn 0x4 at (tid, 0); mip_pck
    /// 0xF and mip_pck_sgn 0x3 at (tid & 7, 0, (tid >> 3) & 1); 17 dwords
    /// stored at tid * 128.
    const CODE: [u32; 55] = [
        0x34060087, 0x7e3c0300, 0x7e3e0280, 0x36420087, 0x7e440280, 0xd5480023, 0x02050700, 0xf0081f08, 0x00010a1e, 0xf00c1f08, 0x00010e1e, 0xf0081a08,
        0x0001121e, 0xf00c1408, 0x0001141e, 0xf0101f08, 0x00011521, 0xf0141308, 0x00011921, 0xbf8c3f70, 0xe0701000, 0x80000a03, 0xe0701004, 0x80000b03,
        0xe0701008, 0x80000c03, 0xe070100c, 0x80000d03, 0xe0701010, 0x80000e03, 0xe0701014, 0x80000f03, 0xe0701018, 0x80001003, 0xe070101c, 0x80001103,
        0xe0701020, 0x80001203, 0xe0701024, 0x80001303, 0xe0701028, 0x80001403, 0xe070102c, 0x80001503, 0xe0701030, 0x80001603, 0xe0701034, 0x80001703,
        0xe0701038, 0x80001803, 0xe070103c, 0x80001903, 0xe0701040, 0x80001a03, 0xbf810000,
    ];
    /// (name, format, bytes, float)
    const FORMATS: [(&str, u32, u32, bool); 11] = [
        ("8_UNORM", 1, 1, false),
        ("16_UNORM", 7, 2, false),
        ("16_SINT", 12, 2, false),
        ("8_8_UNORM", 14, 2, false),
        ("32_FLOAT", 22, 4, true),
        ("16_16_SINT", 28, 4, false),
        ("8_8_8_8_UNORM", 56, 4, false),
        ("8_8_8_8_SINT", 61, 4, false),
        ("32_32_UINT", 62, 8, false),
        ("16_16_16_16_UNORM", 65, 8, false),
        ("32_32_32_32_FLOAT", 77, 16, true),
    ];

    /// The texel's bytes (floats with their exponent kept in range).
    fn texel(bytes: u32, float: bool, level: u32, x: u32) -> Vec<u8> {
        let mut t: Vec<u8> = (0..bytes).map(|b| (0x80 + 37 * (x * bytes + b) + 7 * x + 101 * level) as u8).collect();
        if float {
            for w in t.chunks_exact_mut(4) {
                let mut v = u32::from_le_bytes(w.try_into().unwrap());
                v = (v & 0x807f_ffff) | ((0x60 + ((v >> 23) & 0x3f)) << 23);
                w.copy_from_slice(&v.to_le_bytes());
            }
        }
        t
    }

    fn raw(bytes: u32, float: bool, level: u32, x: u32, sign: bool) -> [u32; 4] {
        let mut t = texel(bytes, float, level, x);
        t.resize(16, 0);
        let mut w: [u32; 4] = std::array::from_fn(|i| u32::from_le_bytes(t[i * 4..i * 4 + 4].try_into().unwrap()));
        if sign && bytes < 4 {
            let shift = 32 - bytes * 8;
            w[0] = (((w[0] << shift) as i32) >> shift) as u32;
        }
        w
    }

    /// image_load(_mip)_pck(_sgn) return the texel's raw bytes as dwords
    /// (DMASK-selected; _sgn sign-extends texels narrower than a dword).
    #[test]
    #[ignore]
    fn image_packed_ref_load() {
        for (name, format, bytes, float) in FORMATS {
            let mut g = Guest::new(16 << 20);
            let mut rig = rig(&g);
            let (t, ..) = bytes_texture(&mut g, format, bytes, WIDTH, 1, LEVELS, 0xFAC, |level, x, _, i| texel(bytes, float, level, x)[i as usize]);
            let b = run(&mut g, &mut rig, &CODE, &vec![POISON; THREADS as usize * WORDS], &t, [0; 4]);
            for tid in 0..THREADS {
                let plain = raw(bytes, float, 0, tid, false);
                let sign = raw(bytes, float, 0, tid, true);
                let level = (tid >> 3) & 1;
                let mip_plain = raw(bytes, float, level, tid & 7, false);
                let mip_sign = raw(bytes, float, level, tid & 7, true);
                let expected: [u32; RESULTS] = [
                    plain[0], plain[1], plain[2], plain[3], sign[0], sign[1], sign[2], sign[3], plain[1], plain[3], sign[2], mip_plain[0], mip_plain[1], mip_plain[2],
                    mip_plain[3], mip_sign[0], mip_sign[1],
                ];
                for (k, e) in expected.iter().enumerate() {
                    let a = b[tid as usize * WORDS + k];
                    assert_eq!(a, *e, "{name} result {k}: thread {tid} is {a:#010x}, expected {e:#010x}");
                }
            }
        }
    }
}

// ---- ImageStorePacked ----

mod store_packed {
    use super::*;

    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 8;
    const TEXEL_BYTES: usize = 8192;
    /// Data (tid + 1) * 0x9e3779b1 + i * 0x7f4a7c15 in v[20:23]; stored
    /// (DMASK 0xF, 0x5, 0xA, 0x8) at (tid, row 0..3).
    const STORE_CODE: [u32; 24] = [
        0x4a500081, 0xd5690014, 0x0201ff28, 0x9e3779b1, 0x4a2a28ff, 0x7f4a7c15, 0x4a2c28ff, 0xfe94f82a, 0x4a2e28ff, 0x7ddf743f, 0x7e3c0300, 0x7e3e0280,
        0xf0281f08, 0x0001141e, 0x7e3e0281, 0xf0281508, 0x0001141e, 0x7e3e0282, 0xf0281a08, 0x0001141e, 0x7e3e0283, 0xf0281808, 0x0001141e, 0xbf810000,
    ];
    /// image_store_mip_pck DMASK 0xB at (tid >> 1, 4 - 2 * level), level
    /// tid & 1.
    const MIP_CODE: [u32; 17] = [
        0x4a500081, 0xd5690014, 0x0201ff28, 0x9e3779b1, 0x4a2a28ff, 0x7f4a7c15, 0x4a2c28ff, 0xfe94f82a, 0x4a2e28ff, 0x7ddf743f, 0x2c3c0081, 0x36400081,
        0x34424081, 0x4c3e4284, 0xf02c1b08, 0x0001141e, 0xbf810000,
    ];
    const FORMATS: [(&str, u32, u32); 10] = [
        ("8_UINT", 5, 1),
        ("16_UINT", 11, 2),
        ("8_8_UINT", 18, 2),
        ("32_SINT", 21, 4),
        ("32_FLOAT", 22, 4),
        ("16_16_UINT", 27, 4),
        ("8_8_8_8_UINT", 60, 4),
        ("32_32_UINT", 62, 8),
        ("16_16_16_16_UINT", 69, 8),
        ("32_32_32_32_FLOAT", 77, 16),
    ];

    fn data(tid: u32, index: u32) -> u32 {
        (tid + 1).wrapping_mul(0x9e37_79b1).wrapping_add(index.wrapping_mul(0x7f4a_7c15))
    }

    /// The texel bytes a store of DMASK `dmask` by lane `tid` writes.
    fn stored(bytes: u32, dmask: u32, tid: u32) -> Vec<u8> {
        let mut next = 0;
        let mut words = [0u32; 4];
        for (w, word) in words.iter_mut().enumerate() {
            if (dmask >> w) & 1 != 0 {
                *word = data(tid, next);
                next += 1;
            }
        }
        words.iter().flat_map(|w| w.to_le_bytes()).take(bytes as usize).collect()
    }

    fn run_store(code: &[u32], format: u32, bytes: u32, swizzle: u32, levels: u32) -> (Vec<u8>, crate::tiling_layout::SurfaceLayout) {
        let mut g = Guest::new(16 << 20);
        let mut rig = rig(&g);
        let s = Shape { format, bpe: bytes, ..Shape::new(sharps::image_type::TYPE_2D, WIDTH, HEIGHT, 1, levels) };
        let base = g.alloc(&initial_bytes(TEXEL_BYTES));
        let mut d = s.descriptor(base);
        d[3] = (d[3] & !0xFFF) | swizzle;
        let mut user = vec![0u32; 4];
        user.extend_from_slice(&d);
        dispatch(&mut g, &mut rig, code, &user);
        (written_back(&g, &mut rig, base, TEXEL_BYTES as u64), s.layout())
    }

    fn check(actual: &[u8], expected: &[u8], layout: &crate::tiling_layout::SurfaceLayout, bytes: u32, what: &str) {
        for level in 0..layout.mips.len() as u32 {
            let m = layout.mips[level as usize];
            for y in 0..m.height {
                for x in 0..m.width {
                    let o = layout.element_offset(level, 0, x, y).unwrap() as usize;
                    assert_eq!(actual[o..o + bytes as usize], expected[o..o + bytes as usize], "{what}: level {level} texel ({x}, {y})");
                }
            }
        }
    }

    /// image_store(_mip)_pck write the DMASK-selected dwords as the texel's
    /// raw bytes (unselected dwords 0), whatever the T# swizzle.
    #[test]
    #[ignore]
    fn image_packed_ref_store() {
        for (name, format, bytes) in FORMATS {
            let (actual, layout) = run_store(&STORE_CODE, format, bytes, 0xFAC, 1);
            let mut expected = initial_bytes(TEXEL_BYTES);
            for tid in 0..THREADS {
                for (row, dmask) in [0xF, 0x5, 0xA, 0x8].into_iter().enumerate() {
                    let o = layout.element_offset(0, 0, tid, row as u32).unwrap() as usize;
                    expected[o..o + bytes as usize].copy_from_slice(&stored(bytes, dmask, tid));
                }
            }
            check(&actual, &expected, &layout, bytes, &format!("image_store_pck {name}"));
            if format == 60 {
                let (actual, layout) = run_store(&STORE_CODE, format, bytes, 0x977, 1);
                check(&actual, &expected, &layout, bytes, &format!("image_store_pck {name} with a swizzled T#"));
            }
        }
        for (name, format, bytes) in [FORMATS[6], FORMATS[9]] {
            let (actual, layout) = run_store(&MIP_CODE, format, bytes, 0xFAC, 2);
            let mut expected = initial_bytes(TEXEL_BYTES);
            for tid in 0..THREADS {
                let level = tid & 1;
                let o = layout.element_offset(level, 0, tid >> 1, 4 - 2 * level).unwrap() as usize;
                expected[o..o + bytes as usize].copy_from_slice(&stored(bytes, 0xB, tid));
            }
            check(&actual, &expected, &layout, bytes, &format!("image_store_mip_pck {name}"));
        }
    }
}

// ---- ImagePckN, ImageBy ----

mod multi_texel {
    use super::*;

    const WIDTH: u32 = 61;
    const HEIGHT: u32 = 3;
    const LEVELS: u32 = 2;
    const SENTINEL: u32 = 0xcafe_0000;
    /// (format, bytes per component, components)
    const R8: (u32, u32, u32) = (5, 1, 1);
    const R16: (u32, u32, u32) = (11, 2, 1);
    const RG8: (u32, u32, u32) = (18, 1, 2);

    /// v[30:32] = the lane's (x, y, mip); v[10:13] sentinels; the op of
    /// `opcode` with `dmask` into v[10:13]; v[10:13] stored at tid * 16.
    fn code(opcode: u32, dmask: u32) -> Vec<u32> {
        let mut c = vec![0x34060084, 0xe0381000, 0x80031e03, 0xbf8c3f70];
        for i in 0..4u32 {
            c.push(0x7e00_02ff | ((10 + i) << 17));
            c.push(SENTINEL + i);
        }
        c.extend([0xf000_0008 | (opcode << 18) | (dmask << 8), 0x00010a1e, 0xbf8c3f70, 0xe0781000, 0x80000a03, 0xbf810000]);
        c
    }

    fn value(level: u32, x: u32, y: u32, c: u32, bytes: u32) -> u32 {
        let v = level * 0x9e37 + y * 0x51 + x * 0x13 + c * 0x2b + 1;
        if bytes == 1 {
            v & 0xFF
        } else {
            v & 0xFFFF
        }
    }

    fn input() -> Vec<u32> {
        let xs = [0, 1, 2, 3, 5, 6, 7, WIDTH - 4, WIDTH - 3, WIDTH - 2, WIDTH - 1, WIDTH, 29, 30, 0xffff_ffff, 0x8000_0000];
        let ys = [0, 1, 2, HEIGHT, 0xffff_ffff];
        let mips = [0, 1, LEVELS];
        (0..THREADS).flat_map(|t| [xs[t as usize % 16], ys[(t as usize * 3) % 5], mips[(t as usize / 2) % 3], 0]).collect()
    }

    /// Runs `opcode` on a texture of `format`; returns v[10:13] of every
    /// lane and the input.
    fn run_op(opcode: u32, (format, bytes, components): (u32, u32, u32), dmask: u32) -> (Vec<u32>, Vec<u32>) {
        let mut g = Guest::new(16 << 20);
        let mut rig = rig(&g);
        let (t, ..) = bytes_texture(&mut g, format, bytes * components, WIDTH, HEIGHT, LEVELS, 0xFAC, |level, x, y, i| {
            (value(level, x, y, i / bytes, bytes) >> (8 * (i % bytes))) as u8
        });
        let input = input();
        let ib = g.alloc(&asm::bytes(&input));
        let out = g.alloc(&asm::bytes(&vec![POISON; THREADS as usize * 4]));
        let mut user = raw(out, THREADS * 16).to_vec();
        user.extend_from_slice(&t);
        user.extend(raw(ib, (input.len() * 4) as u32));
        dispatch(&mut g, &mut rig, &code(opcode, dmask), &user);
        (read_words(&g, &mut rig, out, THREADS as usize * 4), input)
    }

    fn lane(input: &[u32], tid: usize, mip: bool, elements: u32) -> (bool, u32, u32, u32) {
        let (x, y) = (input[tid * 4], input[tid * 4 + 1]);
        let level = if mip { input[tid * 4 + 2] } else { 0 };
        let width = if level < LEVELS { (WIDTH >> level).max(1) } else { 0 };
        let height = if level < LEVELS { (HEIGHT >> level).max(1) } else { 0 };
        let inside = level < LEVELS && x < width && width - x >= elements && y < height;
        (inside, x & !(elements - 1), y, level)
    }

    /// image_load(_mip)_pck2 / _pck4: the raw bits of 2 or 4 texels from
    /// x & !(n - 1) packed into one dword (0 outside), v11..v13 untouched.
    #[test]
    #[ignore]
    fn image_packed_ref_pck_n() {
        let cases = [(0x70, R8), (0x70, R16), (0x70, RG8), (0x73, R8), (0x73, R16), (0x73, RG8), (0x71, R8), (0x74, R8)];
        for (opcode, format) in cases {
            let (out, input) = run_op(opcode, format, 0x1);
            let elements = if matches!(opcode, 0x71 | 0x74) { 4 } else { 2 };
            let mip = matches!(opcode, 0x73 | 0x74);
            let (_, bytes, components) = format;
            let texel_bits = bytes * components * 8;
            for tid in 0..THREADS as usize {
                let (inside, first, y, level) = lane(&input, tid, mip, elements);
                let mut packed = 0u32;
                for element in 0..elements {
                    if !inside {
                        break;
                    }
                    let texel = (0..components).fold(0u32, |t, c| t | (value(level, first + element, y, c, bytes) << (c * bytes * 8)));
                    packed |= texel << (element * texel_bits);
                }
                for i in 0..4 {
                    let e = if i == 0 { packed } else { SENTINEL + i as u32 };
                    assert_eq!(out[tid * 4 + i], e, "PCK opcode {opcode:#x} format {}: thread {tid} v{}", format.0, 10 + i);
                }
            }
        }
    }

    /// image_load(_mip)_by2 / _by4: each of 2 or 4 texels' components in
    /// turn (0 outside), the dwords past the DMASK untouched.
    #[test]
    #[ignore]
    fn image_packed_ref_by() {
        let cases = [(0x42, R8, 0x3), (0x42, R16, 0x3), (0x42, RG8, 0xF), (0x4A, R8, 0x3), (0x4A, R16, 0x3), (0x4A, RG8, 0xF), (0x43, R8, 0xF), (0x4B, R8, 0xF)];
        for (opcode, format, dmask) in cases {
            let (out, input) = run_op(opcode, format, dmask);
            let elements = if opcode & 1 != 0 { 4 } else { 2 };
            let mip = opcode & 8 != 0;
            let (_, bytes, components) = format;
            let dwords = (dmask as u32).count_ones();
            for tid in 0..THREADS as usize {
                let (inside, first, y, level) = lane(&input, tid, mip, elements);
                for i in 0..4u32 {
                    let e = if i < dwords {
                        if inside {
                            value(level, first + i / components, y, i % components, bytes)
                        } else {
                            0
                        }
                    } else {
                        SENTINEL + i
                    };
                    assert_eq!(out[tid * 4 + i as usize], e, "BY opcode {opcode:#x} format {} dmask {dmask:#x}: thread {tid} v{}", format.0, 10 + i);
                }
            }
        }
    }
}

// ---- ImagePackedHorizontalGather ----

/// image_gather4h_pck: the raw bits of the four texels of the row around
/// the footprint (clamped to the row), DMASK-selected dwords; a row past
/// the image reads 0, one above it row 0.
#[test]
#[ignore]
fn image_packed_ref_horizontal_gather() {
    const WIDTH: u32 = 16;
    const HEIGHT: u32 = 4;
    const EDGES: [[f32; 2]; 16] = [
        [0.03125, 0.0],
        [0.09375, 0.25],
        [0.53125, 0.5],
        [0.96875, 0.75],
        [-0.25, -0.3],
        [1.5, 0.99],
        [0.4, 1.0],
        [0.2, 2.0],
        [0.3, 0.6],
        [0.7, 0.1],
        [0.0, -1.0],
        [1.0, 0.5],
        [0.0625, 0.0],
        [0.15, 0.375],
        [0.85, 0.875],
        [-3.0, 0.2],
    ];
    let texel_bits = |x: i32, y: i32| -> u32 {
        let cx = x.clamp(0, WIDTH as i32 - 1) as u32;
        let cy = y.max(0) as u32;
        if cy >= HEIGHT {
            return 0;
        }
        (0..4).fold(0, |b, c| b | ((((cy * WIDTH + cx) * 4 + c + 1) & 0xFF) << (8 * c)))
    };
    let input: Vec<u32> = (0..THREADS)
        .flat_map(|tid| {
            let (mut u, mut v) = ((tid % WIDTH) as f32 / WIDTH as f32, (tid % HEIGHT) as f32 / HEIGHT as f32);
            if tid >= WIDTH {
                [u, v] = EDGES[(tid - WIDTH) as usize];
                if tid % 2 == 1 && tid < WIDTH + 4 {
                    u = f32::from_bits(if u > 0.0 { u.to_bits() - 1 } else { u.to_bits() + 1 });
                }
            }
            [u.to_bits(), v.to_bits(), 0, 0]
        })
        .collect();
    for dmask in [0xF, 0x1, 0x5, 0xA, 0x8] {
        let mut g = Guest::new(16 << 20);
        let mut rig = rig(&g);
        let (t, ..) = bytes_texture(&mut g, 56, 4, WIDTH, HEIGHT, 1, 0xFAC, |_, x, y, c| (((y * WIDTH + x) * 4 + c + 1) & 0xFF) as u8);
        let code = [0x34020084, 0xe0381000, 0x80000401, 0xbf8c3f70, 0xf188_0008 | (dmask << 8), 0x00820804, 0xbf8c3f70, 0xe0781000, 0x80010801, 0xbf810000];
        let out = run_split(&mut g, &mut rig, &code, &input, 4, &t, [(2 << 3) | (2 << 6), 0, 0, 0]);
        for tid in 0..THREADS as usize {
            let (u, v) = (f32::from_bits(input[tid * 4]), f32::from_bits(input[tid * 4 + 1]));
            let anchor = (u * WIDTH as f32 - 0.5).floor() as i32;
            let row = (v * HEIGHT as f32).floor() as i32;
            let stream: Vec<u32> = (0..4).map(|e| texel_bits(anchor + e - 1, row)).collect();
            let selected: Vec<u32> = (0..4).filter(|b| (dmask >> b) & 1 != 0).map(|b| stream[b as usize]).collect();
            for (k, e) in selected.iter().enumerate() {
                assert_eq!(out[tid * 4 + k], *e, "packed horizontal gather dmask {dmask:#x}: thread {tid} register {k} (u {u}, v {v})");
            }
        }
    }
}
