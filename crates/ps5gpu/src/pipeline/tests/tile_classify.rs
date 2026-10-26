// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! A Demon's Souls (PPSA01342) deferred-lighting tile classifier on the
//! device. Each 8x8 work-group (one wave64) reads one word per tile (two
//! 16-bit light-class ids), and for every class 0..=64 counts the
//! matching lanes with a whole-wave prefix sum (`s_orn2_saveexec` to all
//! lanes, DPP `row_shr` adds, `v_permlanex16`, `v_readlane` 31 and 63);
//! lane 0 adds the count to the x group count of that class's indirect
//! dispatch. When the counts stay zero every lighting dispatch runs no
//! groups and the scene is unlit (fog and emissive only).
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored tile_classify`.

use super::*;
use ps5shader::testing::buffer_descriptor;

/// The classifier, as the game created it (`cs` 0x2088aa800).
const CLASSIFY: [u32; 77] = [
    0xBFA00003, 0xD7460004, 0x04050603, 0xF4040200, 0xFA000008, 0xBF8CC07F, 0xF4041A84, 0xFA000004, 0xD7460003, 0x04010602, 0xD7460000, 0x04010701, 0x7E0202C1,
    0x7E0402C1, 0xBF8CC07F, 0x7DA8086B, 0xF4080004, 0xFA000070, 0x7DA8066A, 0xBF88000F, 0xF4001A84, 0xFA000000, 0xBF8CC07F, 0xD5690001, 0x0002086A, 0xF4080104,
    0xFA000080, 0x4A020303, 0xBF8CC07F, 0xE0002000, 0x80010201, 0xBF8C3F70, 0x2C020490, 0x360404FF, 0x0000FFFF, 0xBE880380, 0xBEFE04C1, 0x7D8404F9, 0x06868408,
    0x7D840208, 0x7D8A00F9, 0x06868680, 0x88EA6A04, 0xBE840806, 0x020702F9, 0x86860680, 0xBEEA287E, 0x02080680, 0x4A0808FA, 0xFF011104, 0x4A0808FA, 0xFF011204,
    0x4A0808FA, 0xFF011404, 0x4A0808FA, 0xFF011804, 0xD7781003, 0x03058304, 0x4A080704, 0xBEFE0404, 0xD7600006, 0x00013F04, 0xD7600007, 0x00017F04, 0x81090706,
    0xBF880006, 0x7E060209, 0x936A8308, 0x7E08026A, 0xBF8CC07F, 0xE0C82000, 0x80000304, 0x81088108, 0xBEFE04C1, 0xBF0BC008, 0xBF85FFD9, 0xBF810000,
];

/// The game's V# word 3 for both buffers: BUF_FMT_32_UINT, X from x.
const DWORD3: u32 = 0x0001_4004;

#[test]
#[ignore]
fn tile_classify_counts_every_class() {
    let mut g = Guest::new(32 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    // A partial last row and column of groups, as 2560x1440 gives (320x180
    // tiles in 40x23 groups).
    let (w, h) = (21u32, 13u32);
    let tiles: Vec<u32> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let lo = (x * 3 + y) % 7;
            let hi = if (x + y) % 3 == 0 { 0xFFFF } else { 9 + (x % 2) };
            (hi << 16) | lo
        })
        .collect();
    let mut expected = vec![0u32; 65 * 3];
    for c in 0..65u32 {
        expected[c as usize * 3] = tiles.iter().filter(|&&t| t & 0xFFFF == c || t >> 16 == c).count() as u32;
        expected[c as usize * 3 + 1] = 1;
        expected[c as usize * 3 + 2] = 1;
    }
    let tile_buf = g.alloc(&asm::bytes(&tiles));
    let mut initial = vec![0u32; 65 * 3];
    for c in 0..65 {
        initial[c * 3 + 1] = 1;
        initial[c * 3 + 2] = 1;
    }
    let args = g.alloc(&asm::bytes(&initial));
    // Constants: +0 pitch, +4 width, +8 height, +112 args V#, +128 tiles V#.
    let mut consts = vec![0u32; 36];
    consts[0] = w;
    consts[1] = w;
    consts[2] = h;
    consts[28..32].copy_from_slice(&buffer_descriptor(args, 4, initial.len() as u32, DWORD3));
    consts[32..36].copy_from_slice(&buffer_descriptor(tile_buf, 4, tiles.len() as u32, DWORD3));
    let consts = g.alloc(&asm::bytes(&consts));
    // The shader resource table: +8 the constants' address.
    let srt = g.alloc(&asm::bytes(&[0, 0, consts as u32, (consts >> 32) as u32]));
    let cs = g.program(&CLASSIFY);
    let mut shader = stage(cs, &[srt as u32, (srt >> 32) as u32]);
    // TGID_X_EN, TGID_Y_EN, thread ids x and y (the game's 0x984).
    shader.rsrc2 = 0x984;
    let d = ResolvedDispatch {
        packet: DispatchPacket { groups: [w.div_ceil(8), h.div_ceil(8), 1], dispatch_initiator: 1, ..DispatchPacket::default() },
        shader,
        rsrc3: 0,
        threads: [8, 8, 1],
        start: [0; 3],
    };
    assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
    let bytes = expected.len() as u64 * 4;
    assert!(rig.buffers.download(&mut rig.sched, args, bytes));
    let mut out = vec![0u8; bytes as usize];
    assert!(g.mem.read_backing(args, &mut out));
    let got: Vec<u32> = out.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let counts = |v: &[u32]| v.chunks(3).map(|c| c[0]).collect::<Vec<_>>();
    assert_eq!(counts(&got), counts(&expected), "x group counts per class");
}
