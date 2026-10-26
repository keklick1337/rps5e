// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! 64-bit image atomics on 32_32 storage images (the reference driver's
//! ImageAtomics64 execution test): every operation returns the old texel
//! and stores the combined value; sub, signed min/max and compare-swap
//! included.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored image_atomics`.

use super::*;
use ps5shader::testing::raw_buffer_descriptor;

include!("image_atomics64_code.rs");

mod integer32 {
    use super::*;
    include!("image_atomics32_code.rs");

    const EDGES: [[u32; 2]; 12] = [
        [0, 0],
        [0, 5],
        [5, 0],
        [5, 5],
        [4, 5],
        [6, 5],
        [0x7fff_ffff, 0x8000_0000],
        [0x8000_0000, 0x7fff_ffff],
        [0xffff_ffff, 1],
        [1, 0xffff_ffff],
        [0xffff_ffff, 0xffff_ffff],
        [0xffff_fffe, 0xffff_ffff],
    ];

    /// image_atomic_sub / smin / smax / inc / dec / cmpswap on 32-bit
    /// UINT, SINT and FLOAT texels (the reference's ImageAtomics test).
    #[test]
    #[ignore]
    fn image_atomics_32_bit_operations() {
        const THREADS: usize = 32;
        let mut g = Guest::new(16 << 20);
        let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
        let cs = g.program(&CODE32);
        let mut input = vec![0u32; THREADS * 4];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for tid in 0..THREADS {
            for j in 0..3 {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                input[tid * 4 + j] = (state >> 32) as u32;
            }
            if tid < EDGES.len() {
                input[tid * 4] = EDGES[tid][0];
                input[tid * 4 + 1] = EDGES[tid][1];
            }
            if tid & 1 != 0 {
                input[tid * 4 + 2] = input[tid * 4];
            }
        }
        let names = ["image_atomic_sub", "image_atomic_smin", "image_atomic_smax", "image_atomic_inc", "image_atomic_dec", "image_atomic_cmpswap"];
        let mut failures = Vec::new();
        for format in [20u32, 21, 22] {
            let inp = g.alloc(&asm::bytes(&input));
            let out = g.alloc(&asm::bytes(&vec![0xdead_beefu32; THREADS * 16]));
            let tex = g.alloc(&asm::bytes(&vec![0xdead_beefu32; 4096]));
            let (w, h) = (THREADS as u32, 8u32);
            let t_sharp = [(tex >> 8) as u32, ((tex >> 40) as u32 & 0xFF) | (format << 20) | (((w - 1) & 3) << 30), ((w - 1) >> 2) | ((h - 1) << 14), 0xFAC | (sharps::image_type::TYPE_2D << 28), 0, 0, 0, 0];
            let mut user = raw_buffer_descriptor(inp, (input.len() * 4) as u32).to_vec();
            user.extend_from_slice(&raw_buffer_descriptor(out, (THREADS * 16 * 4) as u32));
            user.extend_from_slice(&t_sharp);
            let mut shader = stage(cs, &user);
            shader.rsrc1 = (0xC0 << 12) | (1 << 21);
            let d = ResolvedDispatch { packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1 | (1 << 15), ..DispatchPacket::default() }, shader, rsrc3: 0, threads: [THREADS as u32, 1, 1], start: [0; 3] };
            assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded, "format {format}");
            assert!(rig.buffers.download(&mut rig.sched, out, (THREADS * 16 * 4) as u64));
            let mut bytes = vec![0u8; THREADS * 16 * 4];
            assert!(g.mem.read_backing(out, &mut bytes));
            let o: Vec<u32> = bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            for tid in 0..THREADS {
                let (a, b, c) = (input[tid * 4], input[tid * 4 + 1], input[tid * 4 + 2]);
                let (sa, sb) = (a as i32, b as i32);
                let expected = [a.wrapping_sub(b), sa.min(sb) as u32, sa.max(sb) as u32, if a >= b { 0 } else { a + 1 }, if a == 0 || a > b { b } else { a - 1 }, if a == c { b } else { a }];
                for j in 0..6 {
                    let (returned, stored) = (o[tid * 16 + 2 * j], o[tid * 16 + 2 * j + 1]);
                    if returned != a || stored != expected[j] {
                        failures.push(format!("{}, format {format}: thread {tid} returned {returned:#x} (expected {a:#x}), stored {stored:#x} (expected {:#x})", names[j], expected[j]));
                    }
                }
            }
        }
        failures.truncate(30);
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}

mod float {
    use super::*;
    include!("image_atomics_float_vectors.rs");

    const MAX_LANES: usize = 64;
    const INPUTS: usize = 4;
    const RESULTS: usize = 16;
    const CHECKED: usize = 8;
    const ROWS: usize = 800;

    /// (initial texel, source, comparator) of a row.
    fn input_row(row: usize) -> [u32; 3] {
        let n = VALUES.len();
        let initial = VALUES[row % (n * n) / n];
        let source = VALUES[row % n];
        [initial, source, if row < n * n { source } else { initial }]
    }

    fn expected_row(row: usize) -> [u32; CHECKED] {
        let [initial, source, _] = input_row(row);
        let m = &EXPECTED[row * 3..];
        [initial, m[0], initial, m[1], initial, m[2], source, m[0]]
    }

    fn t_sharp(tex: u64, format: u32) -> [u32; 8] {
        let (w, h) = (64u32, 4u32);
        [(tex >> 8) as u32, ((tex >> 40) as u32 & 0xFF) | (format << 20) | (((w - 1) & 3) << 30), ((w - 1) >> 2) | ((h - 1) << 14), 0xFAC | (sharps::image_type::TYPE_2D << 28), 0, 0, 0, 0]
    }

    /// image_atomic_fmin / fmax / fcmpswap (with GLC) and fmin without GLC
    /// on 32-bit UINT, SINT and FLOAT texels against the measured rows
    /// (signed zeros, NaNs, denormals); a 16_16 float texel is refused.
    #[test]
    #[ignore]
    fn image_atomics_float_operations() {
        let mut g = Guest::new(32 << 20);
        let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
        let cs = g.program(&CODE);
        let mut failures = Vec::new();
        for format in [20u32, 21, 22] {
            for wave in [32usize, 64] {
                let tex = g.alloc(&asm::bytes(&vec![0xdead_beefu32; 64 * 4]));
                let mut first = 0;
                while first < ROWS {
                    let mut buffer = vec![0xdead_beefu32; MAX_LANES * INPUTS + MAX_LANES * RESULTS];
                    for lane in 0..wave {
                        let r = input_row((first + lane) % ROWS);
                        buffer[lane * INPUTS..][..3].copy_from_slice(&r);
                        buffer[lane * INPUTS + 3] = 0;
                    }
                    let buf = g.alloc(&asm::bytes(&buffer));
                    let mut user = raw_buffer_descriptor(buf, (buffer.len() * 4) as u32).to_vec();
                    user.extend_from_slice(&t_sharp(tex, format));
                    let mut shader = stage(cs, &user);
                    shader.rsrc1 = (0xC0 << 12) | (1 << 21);
                    let initiator = 1 | if wave == 32 { 1 << 15 } else { 0 };
                    let d = ResolvedDispatch {
                        packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: initiator, ..DispatchPacket::default() },
                        shader,
                        rsrc3: 0,
                        threads: [wave as u32, 1, 1],
                        start: [0; 3],
                    };
                    assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded, "format {format} wave{wave}");
                    let bytes_len = buffer.len() * 4;
                    assert!(rig.buffers.download(&mut rig.sched, buf, bytes_len as u64));
                    let mut bytes = vec![0u8; bytes_len];
                    assert!(g.mem.read_backing(buf, &mut bytes));
                    let o: Vec<u32> = bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                    let names = ["fmin returned", "fmin texel", "fmax returned", "fmax texel", "fcmpswap returned", "fcmpswap texel", "fmin without glc vdata", "fmin without glc texel"];
                    for lane in 0..wave {
                        let row = (first + lane) % ROWS;
                        let want = expected_row(row);
                        for j in 0..CHECKED {
                            let got = o[MAX_LANES * INPUTS + lane * RESULTS + j];
                            if got != want[j] {
                                failures.push(format!("wave{wave}, format {format}: row {row} {} is {got:#010x}, expected {:#010x}", names[j], want[j]));
                            }
                        }
                    }
                    first += wave;
                }
            }
        }
        // 16_16 FLOAT texels: the atomic image descriptor is refused.
        let tex = g.zeroed(64 * 4 * 4);
        let buf = g.zeroed((MAX_LANES * (INPUTS + RESULTS) * 4) as usize);
        let mut user = raw_buffer_descriptor(buf, (MAX_LANES * (INPUTS + RESULTS) * 4) as u32).to_vec();
        user.extend_from_slice(&t_sharp(tex, 29));
        let mut shader = stage(cs, &user);
        shader.rsrc1 = (0xC0 << 12) | (1 << 21);
        let d = ResolvedDispatch { packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1 | (1 << 15), ..DispatchPacket::default() }, shader, rsrc3: 0, threads: [32, 1, 1], start: [0; 3] };
        assert_ne!(rig.dispatch(&d), DrawOutcome::Recorded, "16_16 float atomics were accepted");
        let total = failures.len();
        failures.truncate(200);
        assert!(failures.is_empty(), "{total} failures\n{}", failures.join("\n"));
    }
}

const THREADS: usize = 32;
const INPUTS: usize = 8;
const RESULTS: usize = 64;
const OPERATIONS: usize = 11;

const EDGES: [[u64; 2]; 12] = [
    [0, 0],
    [0, 5],
    [5, 0],
    [0x0000_0000_ffff_ffff, 1],
    [0x0000_0001_0000_0000, 1],
    [0x0000_0001_0000_0000, 0x0000_0000_ffff_ffff],
    [0x8000_0000_0000_0000, 0x7fff_ffff_ffff_ffff],
    [0x7fff_ffff_ffff_ffff, 0x8000_0000_0000_0000],
    [0xffff_ffff_ffff_ffff, 1],
    [1, 0xffff_ffff_ffff_ffff],
    [0xffff_ffff_0000_0000, 0x0000_0000_ffff_ffff],
    [0x7fff_ffff_0000_0000, 0x8000_0000_ffff_ffff],
];

fn word(w: &[u32]) -> u64 {
    w[0] as u64 | (w[1] as u64) << 32
}

fn set_word(w: &mut [u32], v: u64) {
    w[0] = v as u32;
    w[1] = (v >> 32) as u32;
}

fn input() -> Vec<u32> {
    let mut input = vec![0u32; THREADS * INPUTS];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for tid in 0..THREADS {
        let words = &mut input[tid * INPUTS..][..INPUTS];
        for j in 0..3 {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            set_word(&mut words[2 * j..], state ^ (state << 29));
        }
        if tid < EDGES.len() {
            set_word(words, EDGES[tid][0]);
            set_word(&mut words[2..], EDGES[tid][1]);
        }
        if tid & 1 != 0 {
            let a = word(words);
            set_word(&mut words[4..], a);
        }
    }
    input
}

#[test]
#[ignore]
fn image_atomics_64_bit_operations() {
    let mut g = Guest::new(16 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    if !rig.dev.caps.shader_image_int64_atomics {
        eprintln!("skipped: no shaderImageInt64Atomics");
        return;
    }
    let cs = g.program(&CODE);
    let input = input();
    let names = [
        "image_atomic_swap", "image_atomic_add", "image_atomic_sub", "image_atomic_smin", "image_atomic_umin", "image_atomic_smax", "image_atomic_umax",
        "image_atomic_and", "image_atomic_or", "image_atomic_xor", "image_atomic_cmpswap",
    ];
    let mut failures = Vec::new();
    // 32_32 UINT, SINT and FLOAT.
    for format in [62u32, 63, 64] {
        let inp = g.alloc(&asm::bytes(&input));
        let out = g.alloc(&asm::bytes(&vec![0xdead_beefu32; THREADS * RESULTS]));
        let tex = g.alloc(&asm::bytes(&vec![0xdead_beefu32; 4096]));
        let (w, h) = (THREADS as u32, OPERATIONS as u32);
        let t_sharp = [
            (tex >> 8) as u32,
            ((tex >> 40) as u32 & 0xFF) | (format << 20) | (((w - 1) & 3) << 30),
            ((w - 1) >> 2) | ((h - 1) << 14),
            0xFAC | (sharps::image_type::TYPE_2D << 28),
            0,
            0,
            0,
            0,
        ];
        let mut user = raw_buffer_descriptor(inp, (input.len() * 4) as u32).to_vec();
        user.extend_from_slice(&raw_buffer_descriptor(out, (THREADS * RESULTS * 4) as u32));
        user.extend_from_slice(&t_sharp);
        let mut shader = stage(cs, &user);
        shader.rsrc1 = (0xC0 << 12) | (1 << 21);
        let d = ResolvedDispatch {
            packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1 | (1 << 15), ..DispatchPacket::default() },
            shader,
            rsrc3: 0,
            threads: [THREADS as u32, 1, 1],
            start: [0; 3],
        };
        assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded, "format {format}");
        assert!(rig.buffers.download(&mut rig.sched, out, (THREADS * RESULTS * 4) as u64));
        let mut bytes = vec![0u8; THREADS * RESULTS * 4];
        assert!(g.mem.read_backing(out, &mut bytes));
        let o: Vec<u32> = bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        for tid in 0..THREADS {
            let i = &input[tid * INPUTS..];
            let (a, b, c) = (word(i), word(&i[2..]), word(&i[4..]));
            let (sa, sb) = (a as i64, b as i64);
            let expected = [b, a.wrapping_add(b), a.wrapping_sub(b), sa.min(sb) as u64, a.min(b), sa.max(sb) as u64, a.max(b), a & b, a | b, a ^ b, if a == c { b } else { a }];
            for j in 0..OPERATIONS {
                let returned = word(&o[tid * RESULTS + 4 * j..]);
                let stored = word(&o[tid * RESULTS + 4 * j + 2..]);
                if returned != a {
                    failures.push(format!("{}, format {format}: thread {tid} returned {returned:#018x}, expected {a:#018x}", names[j]));
                }
                if stored != expected[j] {
                    failures.push(format!("{}, format {format}: thread {tid} stored {stored:#018x}, expected {:#018x}", names[j], expected[j]));
                }
            }
        }
    }
    let total = failures.len();
    failures.truncate(40);
    assert!(failures.is_empty(), "{total} failures\n{}", failures.join("\n"));
}
