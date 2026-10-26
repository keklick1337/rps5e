// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! s_memrealtime spin-wait: the captured frame-pacing program
//! (`vectors/frame_pacing_wait.rs`) on the interpreter, whose real-time
//! clock is `realtime_base` plus the steps run. The device side (the clock
//! scaled from the device clock and calibrated against the host's CP
//! timestamps) is covered by `ps5gpu` `pipeline::tests::clock`.

use ps5shader::testing::{buffer_descriptor, ComputeRun};

#[path = "vectors/frame_pacing_wait.rs"]
mod program;
use program::FRAME_PACING_WAIT;

const OUT: u64 = 0x2_0000_0000;
const CB: u64 = 0x1_0000_0000;
const OUT_WORDS: usize = 72;

struct Outcome {
    start: u64,
    last: u64,
    words: Vec<u32>,
}

fn run(base: u64, deadline: u64, max_wait: u32, interval: u32, previous: u64) -> Outcome {
    let mut user = buffer_descriptor(OUT, 4, OUT_WORDS as u32, 0x0101_6FAC).to_vec();
    user.extend(buffer_descriptor(CB, 0, 16, 0x0101_6FAC));
    let run = ComputeRun::new(FRAME_PACING_WAIT.to_vec(), 64, 1, user);
    let cb = [deadline as u32, (deadline >> 32) as u32, max_wait, interval];
    let setup = move |m: &mut ps5shader::interp::Machine| {
        m.realtime_base = base;
        m.memory.write_words(CB, &cb);
        let mut out = vec![0xDEAD_BEEF; OUT_WORDS];
        out[64] = previous as u32;
        out[65] = (previous >> 32) as u32;
        m.memory.write_words(OUT, &out);
    };
    let m = run.run_checked(&setup, &[(OUT, OUT_WORDS)]).expect("run");
    let words = m.memory.read_words(OUT, OUT_WORDS);
    let pair = |i: usize| words[i] as u64 | (words[i + 1] as u64) << 32;
    assert_eq!(&words[..64], &(0..64).collect::<Vec<u32>>()[..]);
    assert_eq!(pair(66), deadline);
    assert_eq!(words[70], max_wait);
    Outcome { start: pair(64), last: pair(68), words }
}

/// Waits until the absolute deadline (the clock starts before it).
#[test]
fn waits_for_the_deadline() {
    let base = 0x6a36_56cf_0000;
    let r = run(base, base + 2000, 3_336_720, 0, 0);
    assert!(r.start >= base && r.start < base + 100, "start {:#x}", r.start);
    assert!(r.last >= base + 2000 && r.last < base + 2200, "last {:#x}", r.last);
}

/// A deadline already passed returns after one clock read.
#[test]
fn passed_deadline_does_not_wait() {
    let base = 0x6a36_56cf_0000;
    let r = run(base, base - 5, 3_336_720, 0, 0);
    assert_eq!(r.last, r.start);
}

/// A deadline far away is bounded by `max_wait` after the first read.
#[test]
fn max_wait_bounds_the_spin() {
    let base = 0x1000;
    let r = run(base, u64::MAX >> 1, 1500, 0, 0);
    assert!(r.last >= r.start + 1500 && r.last < r.start + 1700, "start {:#x} last {:#x}", r.start, r.last);
}

/// A positive interval waits for `previous start + interval` when that is
/// later than the deadline.
#[test]
fn interval_after_previous_start() {
    let base = 0x10_0000;
    let r = run(base, base, 1_000_000, 1200, base + 300);
    assert!(r.last >= base + 1500 && r.last < base + 1700, "last {:#x}", r.last);
    assert_ne!(r.words[64], 0xDEAD_BEEF);
}
