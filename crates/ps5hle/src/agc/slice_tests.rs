// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Slice mode against the direct interpreter: random packet streams run
//! once with draws resolved on the command thread and once through the
//! sequencer, its register log and the slice thread. The executor must see
//! the same draws (byte-equal), dispatches and command-processor work in
//! the same order, and guest memory must end the same.

use super::mem;
use super::pm4::queue::{GpuQueue, SliceResult};
use super::pm4::{self as p4, nop, op};
use ps5gpu::render::{RenderExecutor, ResolvedDispatch, ResolvedDraw};
use ps5gpu::slices::{Runner, SliceExecutor};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

const ARENA: u64 = 0x40000;
const RECORDS: u64 = 0x1000;
const COUNT: u64 = 0x1400;
const FLAGS: u64 = 0x1800;
const TABLES: u64 = 0x2000;
const LABELS: u64 = 0x6000;
const CMD: u64 = 0x10000;
/// Bytes compared after a run (records, flags, labels).
const COMPARED: std::ops::Range<u64> = RECORDS..LABELS + 0x1000;

fn arena() -> u64 {
    let buf: &'static mut [u8] = Box::leak(vec![0u8; ARENA as usize + 0x1000].into_boxed_slice());
    let base = (buf.as_ptr() as u64 + 0xFFF) & !0xFFF;
    ps5vm::ranges::track(base, ARENA, ps5vm::MemProtect::RW);
    base
}

/// What the executor saw, in order (draws as debug text: random words make
/// NaN floats, unequal to themselves).
#[derive(Default)]
struct Log(Mutex<Vec<String>>);

impl Log {
    fn push(&self, s: String) {
        self.0.lock().unwrap().push(s);
    }
}

impl RenderExecutor for Log {
    fn draw(&self, d: &ResolvedDraw) {
        self.push(format!("draw {d:?}"));
    }
    fn dispatch(&self, d: &ResolvedDispatch) {
        self.push(format!("dispatch {d:?}"));
    }
    fn end_of_pipe(&self, done: Box<dyn FnOnce() + Send>) {
        self.push("eop".into());
        done();
    }
    fn after_pipe(&self, done: Box<dyn FnOnce() + Send>) {
        self.push("after".into());
        done();
    }
    fn memory_filled(&self, addr: u64, size: u64, value: u32) {
        self.push(format!("filled {addr:x} {size} {value:x}"));
    }
    fn flush(&self) {
        self.push("flush".into());
    }
    fn sync_point(&self) {
        self.push("sync".into());
    }
    fn submission_complete(&self) {
        self.push("complete".into());
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn u32(&mut self) -> u32 {
        self.next() as u32
    }
    fn pick<T: Copy>(&mut self, v: &[T]) -> T {
        v[self.below(v.len() as u64) as usize]
    }
}

fn lo(a: u64) -> u32 {
    a as u32
}

fn hi(a: u64) -> u32 {
    (a >> 32) as u32
}

/// A random command stream over the arena at `base`: submissions (each a
/// dword vector) and the register tables they reference (address, words).
struct Stream {
    submissions: Vec<Vec<u32>>,
    tables: Vec<(u64, Vec<u32>)>,
    memory: Vec<u8>,
}

/// Shader registers indirect draws and the stream write: user data of the
/// graphics stages, the nop register, outside the bank.
const SH_TARGETS: [u32; 9] = [0x8C, 0x8D, 0x8E, 0x8F, 0x0C, 0x0D, 0x10, 0x280, 0x300];

fn stream(base: u64, seed: u64) -> Stream {
    let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let a = |off: u64| base + off;
    let mut memory = vec![0u8; (COMPARED.end - COMPARED.start) as usize];
    for (i, b) in memory.iter_mut().enumerate() {
        let off = COMPARED.start + i as u64;
        // Small record words (counts, offsets), 0/1 flags.
        *b = if off >= FLAGS && off < TABLES { (r.below(2) as u8) * (i % 4 == 0) as u8 } else if i % 4 == 0 { r.below(6) as u8 } else { 0 };
    }
    let mut tables = Vec::new();
    let mut submissions = Vec::new();
    for _ in 0..1 + r.below(3) {
        let mut w: Vec<u32> = Vec::new();
        for _ in 0..120 + r.below(120) {
            let predicate = (r.below(6) == 0) as u32;
            match r.below(22) {
                0..=3 => {
                    let (any, target, sh, uc) = (r.below(0x404) as u32, 0x318 + 15 * r.below(8) as u32, r.pick(&SH_TARGETS), 0x240 + r.below(0x10) as u32);
                    let (opc, off) = match r.below(3) {
                        0 => (op::SET_CONTEXT_REG, r.pick(&[any, target, 0x12, 0x8E, 0xDB])),
                        1 => (op::SET_SH_REG, r.pick(&[any % 0x304, sh])),
                        _ => (op::SET_UCONFIG_REG, r.pick(&[uc, p4::VGT_INDEX_TYPE, 0xA2, 0x3FFE])),
                    };
                    let n = 1 + r.below(5) as u32;
                    w.push(p4::make(n + 2, opc, 0) | predicate);
                    w.push(off);
                    for _ in 0..n {
                        w.push(if r.below(3) == 0 { r.u32() } else { r.u32() % 16 });
                    }
                }
                4 => {
                    let (opc, offs): (u32, Vec<u32>) = match r.below(3) {
                        0 => (op::SET_CONTEXT_REG_INDIRECT, vec![0x8E, 0x318, 0x202, 0x7000_0100, u32::MAX]),
                        1 => (op::SET_SH_REG_INDIRECT, SH_TARGETS.to_vec()),
                        _ => (op::SET_UCONFIG_REG_INDIRECT, vec![p4::VGT_INDEX_TYPE, 0x242, 0x7000_0243, u32::MAX]),
                    };
                    let count = 1 + r.below(5) as u32;
                    let at = a(TABLES) + tables.len() as u64 * 0x40;
                    let words: Vec<u32> = (0..count).flat_map(|_| [r.pick(&offs), r.u32() % 64]).collect();
                    tables.push((at, words));
                    w.extend_from_slice(&[p4::make(5, opc, 0), lo(at) & !3, hi(at), 0, count]);
                }
                5 => w.extend_from_slice(&[p4::make(3, op::NOP, nop::CONTEXT_STATE), r.below(4) as u32, 0]),
                6 if r.below(3) == 0 => w.extend_from_slice(&[p4::make(2, op::CLEAR_STATE, 0), 0]),
                7 if r.below(4) == 0 => w.extend_from_slice(&[p4::make(2, op::NOP, nop::DISPATCH_RESET), 0]),
                8 => w.extend_from_slice(&[p4::make(2, op::NUM_INSTANCES, 0) | predicate, r.below(4) as u32]),
                9 => match r.below(3) {
                    0 => w.extend_from_slice(&[p4::make(2, op::INDEX_TYPE, 0), r.below(3) as u32]),
                    1 => w.extend_from_slice(&[p4::make(3, op::INDEX_BASE, 0), 0x1000 * r.below(4) as u32, 0]),
                    _ => {
                        let base_addr = if r.below(5) == 0 { 0 } else { a(RECORDS) };
                        w.extend_from_slice(&[p4::make(4, op::SET_BASE, 0), 1, lo(base_addr) & !7, hi(base_addr) & 0xFFFF]);
                    }
                },
                10..=13 => {
                    let header = |n: u32, o: u32| p4::make(n, o, 0) | predicate;
                    let target = |r: &mut Rng| r.pick(&SH_TARGETS);
                    match r.below(6) {
                        0 => w.extend_from_slice(&[header(3, op::DRAW_INDEX_AUTO), r.below(100) as u32, 2]),
                        1 => w.extend_from_slice(&[header(6, op::DRAW_INDEX_2), 64, 0x5000, 0, r.below(100) as u32, 0]),
                        2 => w.extend_from_slice(&[header(5, op::DRAW_INDEX_OFFSET_2), 64, r.below(8) as u32, r.below(100) as u32, 0]),
                        3 => {
                            let o = if r.below(2) == 0 { op::DRAW_INDIRECT } else { op::DRAW_INDEX_INDIRECT };
                            let locations = target(&mut r) | target(&mut r) << 16;
                            let enable = target(&mut r) | (r.below(2) as u32) << 28;
                            w.extend_from_slice(&[header(5, o), 0x14 * r.below(4) as u32, locations, enable, 0]);
                        }
                        4 => {
                            let o = if r.below(2) == 0 { op::DRAW_INDIRECT_MULTI } else { op::DRAW_INDEX_INDIRECT_MULTI };
                            let locations = target(&mut r) | target(&mut r) << 16;
                            let control = target(&mut r) | (r.below(2) as u32) << 28 | (r.below(2) as u32) << 30 | (r.below(2) as u32) << 31;
                            let count = a(COUNT);
                            w.extend_from_slice(&[
                                header(10, o),
                                0x14 * r.below(3) as u32,
                                locations,
                                target(&mut r),
                                control,
                                r.below(5) as u32,
                                lo(count),
                                hi(count),
                                r.pick(&[16, 20, 24, 8]),
                                0,
                            ]);
                        }
                        _ => w.extend_from_slice(&[header(9, op::DISPATCH_DRAW_PREAMBLE), r.below(100) as u32, 0x6000, 0, 4, 0, 0, r.below(3) as u32, 0]),
                    }
                }
                14 => w.extend_from_slice(&[p4::make(5, op::DISPATCH_DIRECT, 0) | 2 | predicate, 1 + r.below(4) as u32, 1, 1, 1]),
                15 => {
                    // Predication on a flag (cleared again now and then).
                    let flag = a(FLAGS) + 0x10 * r.below(4);
                    let operation = if r.below(4) == 0 { 0 } else { 4 };
                    w.extend_from_slice(&[p4::make(4, op::SET_PREDICATION, 0), (r.below(2) as u32) << 8 | operation << 16, lo(flag) & !0xF, hi(flag)]);
                }
                16 => {
                    // Conditional execution of the next packet.
                    let flag = a(FLAGS) + 0x10 * r.below(4);
                    w.extend_from_slice(&[p4::make(5, op::COND_EXEC, 0), lo(flag) & !3, hi(flag), 0, 3]);
                    w.extend_from_slice(&[p4::make(3, op::SET_SH_REG, 0), r.pick(&SH_TARGETS), r.u32() % 16]);
                }
                17 | 18 => {
                    // CP writes: argument records, flags, labels.
                    let at = match r.below(3) {
                        0 => a(RECORDS) + 4 * r.below(0x40),
                        1 => a(FLAGS) + 0x10 * r.below(4),
                        _ => a(LABELS) + 4 * r.below(0x100),
                    };
                    let v = if r.below(2) == 0 { r.below(2) as u32 } else { r.below(8) as u32 };
                    w.extend_from_slice(&[p4::make(5, op::WRITE_DATA, 0), 5 << 8, lo(at), hi(at), v]);
                }
                19 => {
                    // End of pipe: a label, with or without an interrupt.
                    let at = a(LABELS) + 8 * r.below(0x80);
                    let int_sel = r.pick(&[0u32, 2]);
                    w.extend_from_slice(&[p4::make(6, op::EVENT_WRITE_EOP, 0), 0x2F, lo(at), (hi(at) & 0xFFFF) | 1 << 29 | int_sel << 24, r.u32(), 0]);
                }
                20 => {
                    // DMA fill or copy over records and labels.
                    let dst = if r.below(2) == 0 { a(RECORDS) + 4 * r.below(0x30) } else { a(LABELS) + 4 * r.below(0x100) };
                    let bytes = 4 * (1 + r.below(8) as u32);
                    if r.below(2) == 0 {
                        w.extend_from_slice(&[p4::make(7, op::DMA_DATA, 0), 2 << 29, r.below(6) as u32, 0, lo(dst), hi(dst), bytes]);
                    } else {
                        let src = a(RECORDS) + 4 * r.below(0x30);
                        w.extend_from_slice(&[p4::make(7, op::DMA_DATA, 0), 0, lo(src), hi(src), lo(dst), hi(dst), bytes]);
                    }
                }
                _ => w.extend_from_slice(&[p4::make(2, op::NOP, 0), 0]),
            }
        }
        submissions.push(w);
    }
    Stream { submissions, tables, memory }
}

/// Run `s` on a fresh graphics queue whose executor is `ex`; returns guest
/// memory as it ends.
fn run(base: u64, s: &Stream, ex: Arc<dyn RenderExecutor>) -> Vec<u8> {
    assert!(mem::write(base + COMPARED.start, &s.memory));
    for (at, words) in &s.tables {
        assert!(mem::write_dwords(*at, words));
    }
    let q = GpuQueue::new(1);
    q.with_interpreter(0, |i| i.executor = Some(ex.clone()));
    let mut at = base + CMD;
    for (n, w) in s.submissions.iter().enumerate() {
        assert!(mem::write_dwords(at, w));
        if n > 0 {
            // A frame boundary between submissions: a processor reset.
            q.suspend_point();
        }
        q.submit_graphics(at, w.len() as u32);
        at += (w.len() as u64 * 4 + 0xFF) & !0xFF;
    }
    for _ in 0..10_000 {
        match q.process_one() {
            SliceResult::NoWork => break,
            SliceResult::AllBlocked => panic!("the stream blocked"),
            _ => {}
        }
    }
    ex.drain();
    let mut out = vec![0u8; s.memory.len()];
    assert!(mem::read(base + COMPARED.start, &mut out));
    out
}

#[test]
fn slice_mode_matches_the_direct_interpreter_on_random_streams() {
    let base = arena();
    let mut draws = 0usize;
    let mut slices = 0u64;
    for seed in 1..=60u64 {
        let s = stream(base, seed);
        let direct = Arc::new(Log::default());
        let mem_direct = run(base, &s, direct.clone());
        let sliced = Arc::new(Log::default());
        let runner = Runner::new(true);
        let mismatches = runner.mismatches();
        let budget = 1 + (seed % 7) as u32;
        let ex = SliceExecutor::start(sliced.clone(), budget, 1 + (seed % 3) as usize, runner);
        let mem_sliced = run(base, &s, ex.clone());
        let (a, b) = (direct.0.lock().unwrap(), sliced.0.lock().unwrap());
        assert_eq!(a.len(), b.len(), "seed {seed}: executor items");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x, y, "seed {seed} (budget {budget}): item {i}");
        }
        assert!(mem_direct == mem_sliced, "seed {seed}: guest memory after the stream");
        assert_eq!(mismatches.load(Ordering::Relaxed), 0, "seed {seed}: snapshots match the log's state");
        draws += a.iter().filter(|x| x.starts_with("draw")).count();
        slices += ex.progress().0;
        ex.stop();
    }
    assert!(draws > 1000 && slices > 200, "the streams draw ({draws}) and cut ({slices} slices)");
}

/// What the compile lookahead shows each dispatch: its own program and
/// the programs of the dispatches after it.
#[derive(Default)]
struct Ahead(Mutex<Vec<(u64, Vec<u64>)>>);

impl RenderExecutor for Ahead {
    fn draw(&self, _: &ResolvedDraw) {}
    fn dispatch(&self, d: &ResolvedDispatch) {
        let ahead = ps5gpu::pipeline::lookahead::upcoming(16).iter().map(|u| u.shader.address).collect();
        self.0.lock().unwrap().push((d.shader.address, ahead));
    }
    fn end_of_pipe(&self, done: Box<dyn FnOnce() + Send>) {
        done();
    }
    fn after_pipe(&self, done: Box<dyn FnOnce() + Send>) {
        done();
    }
    fn memory_filled(&self, _: u64, _: u64, _: u32) {}
    fn flush(&self) {}
    fn sync_point(&self) {}
    fn submission_complete(&self) {}
}

/// While a dispatch runs, the command stream after it is resolved without
/// executing it: register writes (direct and through a called buffer)
/// apply to the dispatches after them, and the scan stops at a wait (what
/// follows depends on its outcome).
#[test]
fn lookahead_resolves_the_dispatches_after_the_current_one() {
    let base = arena();
    let pgm = |w: &mut Vec<u32>, a: u64| w.extend_from_slice(&[p4::make(4, op::SET_SH_REG, 0), 0x20C, lo(a >> 8), hi(a >> 8)]);
    let dispatch = |w: &mut Vec<u32>| w.extend_from_slice(&[p4::make(5, op::DISPATCH_DIRECT, 0), 1, 1, 1, 1]);
    let (a1, a2, a3, a4) = (0x1_0000_0100u64, 0x1_0000_0200u64, 0x1_0000_0300u64, 0x1_0000_0400u64);
    // A called buffer: another program and its dispatch.
    let sub_at = base + CMD + 0x800;
    let mut sub = Vec::new();
    pgm(&mut sub, a3);
    dispatch(&mut sub);
    assert!(mem::write_dwords(sub_at, &sub));
    let label = base + LABELS;
    assert!(mem::write_dwords(label, &[1]));
    let mut w = Vec::new();
    pgm(&mut w, a1);
    dispatch(&mut w);
    pgm(&mut w, a2);
    dispatch(&mut w);
    w.extend_from_slice(&[p4::make(4, op::INDIRECT_BUFFER, 0), lo(sub_at), hi(sub_at), sub.len() as u32]);
    // WAIT_REG_MEM: memory, equal, the label holds 1 (passes when run).
    w.extend_from_slice(&[p4::make(7, op::WAIT_REG_MEM, 0), 0x13, lo(label), hi(label), 1, 0xFFFF_FFFF, 4]);
    pgm(&mut w, a4);
    dispatch(&mut w);
    let at = base + CMD;
    assert!(mem::write_dwords(at, &w));
    let ex = Arc::new(Ahead::default());
    let q = GpuQueue::new(1);
    q.with_interpreter(0, |i| i.executor = Some(ex.clone()));
    q.submit_graphics(at, w.len() as u32);
    for _ in 0..100 {
        if matches!(q.process_one(), SliceResult::NoWork) {
            break;
        }
    }
    let seen = ex.0.lock().unwrap().clone();
    assert_eq!(seen, vec![(a1, vec![a2, a3]), (a2, vec![a3]), (a3, vec![]), (a4, vec![])]);
}
