// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! `s_memrealtime` on the device: the captured frame-pacing program spins
//! until the real-time clock passes a deadline taken from the host's
//! reference clock (as the game takes it from a CP timestamp). With the
//! device clock calibrated against that clock the wait ends on time; a
//! clock that does not track it (a per-invocation counter starting at 0)
//! spins up to `max_wait` iterations and reports a time long before the
//! deadline.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored clock`.

use super::*;
use ps5shader::testing::buffer_descriptor;

#[path = "../../../../ps5shader/tests/vectors/frame_pacing_wait.rs"]
mod program;
use program::FRAME_PACING_WAIT;

const OUT_WORDS: usize = 72;

/// Run the program once; returns (start, last clock read) and checks the
/// words it always writes.
fn wait(g: &mut Guest, rig: &mut Rig, cs: u64, deadline: u64, max_wait: u32) -> (u64, u64) {
    let cb = g.alloc(&asm::bytes(&[deadline as u32, (deadline >> 32) as u32, max_wait, 0]));
    let ob = g.zeroed(OUT_WORDS * 4);
    let mut user = buffer_descriptor(ob, 4, OUT_WORDS as u32, 0x0101_6FAC).to_vec();
    user.extend(buffer_descriptor(cb, 0, 16, 0x0101_6FAC));
    let d = ResolvedDispatch {
        // Wave64, one invocation, as the game dispatches it.
        packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 0x41, ..DispatchPacket::default() },
        shader: stage(cs, &user),
        rsrc3: 0,
        threads: [1, 1, 1],
        start: [0; 3],
    };
    assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
    assert!(rig.buffers.download(&mut rig.sched, ob, OUT_WORDS as u64 * 4));
    let mut out = vec![0u8; OUT_WORDS * 4];
    assert!(g.mem.read_backing(ob, &mut out));
    let w: Vec<u32> = out.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(&w[..64], &(0..64).collect::<Vec<u32>>()[..]);
    let pair = |i: usize| w[i] as u64 | (w[i + 1] as u64) << 32;
    assert_eq!(pair(66), deadline);
    (pair(64), pair(68))
}

#[test]
#[ignore]
fn clock_frame_pacing_wait_ends_at_the_deadline() {
    let mut g = Guest::new(16 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    if rig.dev.clock.get().is_none() {
        eprintln!("no calibrated device clock: skipped");
        return;
    }
    let cs = g.program(&FRAME_PACING_WAIT);
    // The game's bound: 33 ms after the first read.
    let max_wait = 3_336_720;
    for _ in 0..3 {
        let deadline = crate::clock::reference_clock() + 300_000; // +3 ms
        let (start, last) = wait(&mut g, &mut rig, cs, deadline, max_wait);
        let now = crate::clock::reference_clock();
        assert!(last >= deadline, "last read {last} before the deadline {deadline} (start {start})");
        // The calibration error is tens of microseconds (more under load): 5 ms of slack.
        assert!(last <= now + 500_000, "last read {last} after the host clock {now}");
        assert!(start + 1_000_000 > deadline, "start {start} far before the deadline {deadline}");
    }
    // A deadline already passed: one read, no wait.
    let deadline = crate::clock::reference_clock() - 1;
    let (start, last) = wait(&mut g, &mut rig, cs, deadline, max_wait);
    assert_eq!(start, last);
}
