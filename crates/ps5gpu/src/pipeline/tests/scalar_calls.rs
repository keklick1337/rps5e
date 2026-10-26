// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Scalar subroutine calls on the device: static calls inside the
//! program, and callees reached through code addresses in user data,
//! linked in by the frontend (re-linked when the user data names other
//! code).
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored scalar_calls`.

use super::*;
use ps5shader::testing::raw_buffer_descriptor;

const STORING_CALLER: [u32; 19] = [
    0x7e020280, 0xbe940383, 0xbe8e0300, 0xbe8f0301, 0xbe8e210e, 0xbe8e0302, 0xbe8f0303, 0xbe8e210e, 0x80948114, 0xbf078014, 0xbf85fff7, 0x34040082,
    0xe0701000, 0x80010102, 0xbf810000, 0xbf9f0000, 0x30306c73, 0x00000040, 0x00000008,
];
const ADD_FIVE: [u32; 2] = [0x4a020285, 0xbe80200e];
const ADD_SIXTY_FOUR: [u32; 2] = [0x4a0202c0, 0xbe80200e];
const ADD_ONE: [u32; 2] = [0x4a020281, 0xbe80200e];

/// The static-call variant of the driver's execution test: lanes load
/// (selector, value, x, y) through s[0:3] and store four results through
/// s[4:7], computed in subroutines (`s_call_b64`, then nested calls mixing
/// `s_swappc_b64` and `s_call_b64`).
const CALL_FORWARD: [u32; 17] = [
    0x34020084, 0xe0381000, 0x80000401, 0xbf8c3f70, 0xbb080001, 0xbf820008, 0x4a0808ff, 0x13579bdf, 0x3a0a0aff, 0xa5a5a5a5, 0x4a0c0f06, 0x3a0e0eff,
    0x5a5a5a5a, 0xbe802008, 0xe0781000, 0x80010401, 0xbf810000,
];
const MIXED_NESTED: [u32; 24] = [
    0x34020084, 0xe0381000, 0x80000401, 0xbf8c3f70, 0xbe8a1f00, 0x800aff0a, 0x00000014, 0x820b800b, 0xbe88210a, 0xbf82000b, 0x4a0808ff, 0x13579bdf,
    0xbb0c0001, 0xbf820006, 0x3a0a0aff, 0xa5a5a5a5, 0x4a0c0f06, 0x3a0e0eff, 0x5a5a5a5a, 0xbe80200c, 0xbe802008, 0xe0781000, 0x80010401, 0xbf810000,
];

fn one_wave(cs: u64, user: &[u32], wave32: bool) -> ResolvedDispatch {
    let mut shader = stage(cs, user);
    shader.rsrc2 = (user.len() as u32) << 1;
    let initiator = 1 | if wave32 { 1 << 15 } else { 0 };
    ResolvedDispatch { packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: initiator, ..DispatchPacket::default() }, shader, rsrc3: 0, threads: [64, 1, 1], start: [0; 3] }
}

fn read_words(rig: &mut Rig, g: &Guest, addr: u64, n: usize) -> Vec<u32> {
    assert!(rig.buffers.download(&mut rig.sched, addr, n as u64 * 4));
    let mut out = vec![0u8; n * 4];
    assert!(g.mem.read_backing(addr, &mut out));
    out.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn write_words(addr: u64, words: &[u32]) {
    let bytes = asm::bytes(words);
    // SAFETY: a chunk of the test's guest view the GPU only reads through
    // the uploads made after this write.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), addr as *mut u8, bytes.len()) };
}

#[test]
#[ignore]
fn scalar_calls_static_calls_on_device() {
    let mut g = Guest::new(16 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    let selectors = [0u32, 1, 2, 3, 4, 7, 0x8000_0000, 0xffff_ffff];
    let mut input = vec![0u32; 64 * 4];
    for tid in 0..64usize {
        input[tid * 4] = selectors[tid % selectors.len()];
        input[tid * 4 + 1] = 0x9e37_79b9u32.wrapping_mul(tid as u32 + 1);
        input[tid * 4 + 2] = 0xffff_ffff - tid as u32;
        input[tid * 4 + 3] = 17 * tid as u32 + 1;
    }
    let inp = g.alloc(&asm::bytes(&input));
    for (name, code) in [("CALL forward", &CALL_FORWARD[..]), ("mixed nested calls", &MIXED_NESTED[..])] {
        let cs = g.program(code);
        for wave32 in [false, true] {
            let out = g.zeroed(64 * 16);
            let mut user = raw_buffer_descriptor(inp, 64 * 16).to_vec();
            user.extend_from_slice(&raw_buffer_descriptor(out, 64 * 16));
            assert_eq!(rig.dispatch(&one_wave(cs, &user, wave32)), DrawOutcome::Recorded, "{name}");
            let got = read_words(&mut rig, &g, out, 64 * 4);
            for tid in 0..64usize {
                let i = &input[tid * 4..tid * 4 + 4];
                let want = [i[0].wrapping_add(0x1357_9bdf), i[1] ^ 0xa5a5_a5a5, i[2].wrapping_add(i[3]), i[3] ^ 0x5a5a_5a5a];
                assert_eq!(&got[tid * 4..tid * 4 + 4], &want, "{name} wave32={wave32} lane {tid}");
            }
        }
    }
}

#[test]
#[ignore]
fn scalar_calls_user_data_callees_on_device() {
    let mut g = Guest::new(16 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    let cs = g.program(&STORING_CALLER);
    let five = g.program(&ADD_FIVE);
    let sixty_four = g.program(&ADD_SIXTY_FOUR);
    let one = g.program(&ADD_ONE);
    for wave32 in [false, true] {
        for (first, second, want) in [(five, sixty_four, 3 * (5 + 64)), (one, sixty_four, 3 * 65), (sixty_four, sixty_four, 3 * 128)] {
            let out = g.zeroed(64 * 4);
            let mut user = vec![first as u32, (first >> 32) as u32, second as u32, (second >> 32) as u32];
            user.extend_from_slice(&raw_buffer_descriptor(out, 64 * 4));
            assert_eq!(rig.dispatch(&one_wave(cs, &user, wave32)), DrawOutcome::Recorded);
            let got = read_words(&mut rig, &g, out, 64);
            assert!(got.iter().all(|&v| v == want), "wave32={wave32}: {got:?}, expected {want}");
        }
    }
    // A callee changed in place is linked again at the next
    // synchronisation point.
    write_words(one, &ADD_FIVE);
    crate::memory::note_sync_point();
    let out = g.zeroed(64 * 4);
    let mut user = vec![one as u32, (one >> 32) as u32, sixty_four as u32, (sixty_four >> 32) as u32];
    user.extend_from_slice(&raw_buffer_descriptor(out, 64 * 4));
    assert_eq!(rig.dispatch(&one_wave(cs, &user, false)), DrawOutcome::Recorded);
    let got = read_words(&mut rig, &g, out, 64);
    assert!(got.iter().all(|&v| v == 3 * 69), "{got:?}");
}
