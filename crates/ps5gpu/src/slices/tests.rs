// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Slice recording: the log replayed on the slice thread against the
//! registers written directly, positioned events, shutdown.

use super::exec::SliceExecutor;
use super::reglog::{apply_fixup, guest_read, InstanceSource};
use super::run::Runner;
use crate::render::regfile::{Bank, ContextState, RegisterFile};
use crate::render::{resolve_dispatch, resolve_draw, DispatchPacket, DrawKind, DrawPacket, IndirectRegisters, RenderExecutor, ResolvedDispatch, ResolvedDraw, SH_NOP};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Guest memory for a test: a tracked, page-aligned range.
fn arena(bytes: usize) -> u64 {
    let buf: &'static mut [u8] = Box::leak(vec![0u8; bytes + 0x1000].into_boxed_slice());
    let base = (buf.as_ptr() as u64 + 0xFFF) & !0xFFF;
    ps5vm::ranges::track(base, bytes as u64 & !0xFFF, ps5vm::MemProtect::RW);
    base
}

fn write(addr: u64, data: &[u8]) {
    // SAFETY: inside a test arena.
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), addr as *mut u8, data.len()) };
}

fn words(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// What the executor saw, in order (draws as their debug text: random
/// register words make NaN floats, which compare unequal to themselves).
#[derive(Default)]
struct Log(Mutex<Vec<String>>);

impl Log {
    fn push(&self, s: String) {
        self.0.lock().unwrap().push(s);
    }
    fn take(&self) -> Vec<String> {
        std::mem::take(&mut *self.0.lock().unwrap())
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
}

/// The direct path the sequencer replaces: registers written in place,
/// draws resolved from them (fixups applied at the draw).
struct Direct {
    regs: RegisterFile,
    log: Arc<Log>,
}

impl Direct {
    fn draw(&mut self, packet: &DrawPacket, instances: InstanceSource, fixup: bool) {
        let mut p = *packet;
        match instances {
            InstanceSource::Packet => {}
            InstanceSource::State => p.instance_count = self.regs.num_instances.max(1),
            InstanceSource::StateRaw => p.instance_count = self.regs.num_instances,
        }
        if fixup {
            apply_fixup(&p, &mut self.regs, &guest_read);
        }
        self.log.draw(&resolve_draw(&self.regs.banks(), &p));
    }
}

/// Random register work, draws (some indirect), dispatches and events
/// through both paths: the slice thread resolves byte-equal draws in the
/// same order, with snapshots verified against the log's state.
#[test]
fn log_replay_resolves_the_draws_of_the_direct_path() {
    let mem = arena(0x10000);
    // Argument records the indirect draws read (rewritten in between).
    let records = mem;
    let count_word = mem + 0x800;
    for seed in 1..=40u64 {
        let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let fill = |r: &mut Rng| {
            let w: Vec<u32> = (0..64).map(|_| r.u32() % 7).collect();
            write(records, &words(&w));
            write(count_word, &(r.u32() % 5).to_le_bytes());
        };
        fill(&mut r);
        let start = r.0;
        let direct_log = Arc::new(Log::default());
        let mut direct = Direct { regs: RegisterFile::new(), log: direct_log.clone() };
        let sliced_log = Arc::new(Log::default());
        let budget = 1 + r.below(6) as u32;
        let runner = Runner::new(true);
        let mismatches = runner.mismatches();
        let ex = SliceExecutor::start(sliced_log.clone(), budget, 1 + r.below(3) as usize, runner);
        // The sequencer's registers (no fixups applied), as the
        // interpreter keeps them in slice mode.
        let mut seq = RegisterFile::new();
        // Same random choices for both paths: replay the generator.
        for pass in 0..2 {
            let mut r = Rng(start);
            let sliced = pass == 1;
            fill(&mut r);
            for _ in 0..300 {
                match r.below(16) {
                    0..=4 => {
                        let bank = [Bank::Context, Bank::Shader, Bank::UserConfig][r.below(3) as usize];
                        let (_, count) = bank.limits();
                        let offset = if bank == Bank::Shader && r.below(2) == 0 { 0x8C + r.below(8) as u32 } else { r.below(count as u64 + 4) as u32 };
                        let values: Vec<u32> = (0..1 + r.below(6)).map(|_| r.u32()).collect();
                        if sliced {
                            ex.log_run(0, &seq, bank, offset, &values);
                            seq.set_run(bank, offset, &values);
                        } else {
                            direct.regs.set_run(bank, offset, &values);
                        }
                    }
                    5 => {
                        let bank = [Bank::Context, Bank::Shader, Bank::UserConfig][r.below(3) as usize];
                        let pairs: Vec<u32> = (0..2 * (1 + r.below(4)))
                            .map(|i| if i % 2 == 0 { [0x8C + r.u32() % 8, r.u32() % 0x400, u32::MAX, 0x7000_0010][r.below(4) as usize] } else { r.u32() })
                            .collect();
                        if sliced {
                            ex.log_table(0, &seq, bank, &pairs);
                            seq.table(bank, &pairs);
                        } else {
                            direct.regs.table(bank, &pairs);
                        }
                    }
                    6 => {
                        let op = [ContextState::Clear, ContextState::Push, ContextState::Pop, ContextState::PushClear][r.below(4) as usize];
                        if sliced {
                            ex.log_context_state(0, &seq, op);
                            let _ = seq.context_state(op);
                        } else {
                            let _ = direct.regs.context_state(op);
                        }
                    }
                    7 if r.below(8) == 0 => {
                        if sliced {
                            ex.log_reset(0, &seq);
                            seq.reset();
                        } else {
                            direct.regs.reset();
                        }
                    }
                    8 => {
                        let n = r.below(4) as u32;
                        if sliced {
                            ex.log_num_instances(0, &seq, n);
                            seq.num_instances = n;
                        } else {
                            direct.regs.num_instances = n;
                        }
                    }
                    9..=12 => {
                        let kind = [DrawKind::Auto, DrawKind::Index, DrawKind::Indirect, DrawKind::IndexIndirectMulti][r.below(4) as usize];
                        let target = |r: &mut Rng| [0x8C + r.u32() % 6, SH_NOP, 0x400][r.below(3) as usize];
                        let packet = DrawPacket {
                            kind,
                            count: r.u32() % 100,
                            instance_count: 1 + r.u32() % 3,
                            indirect_address: records + 0x20 * r.below(3),
                            indirect_count: r.u32() % 4,
                            count_address: if r.below(3) == 0 { count_word } else { 0 },
                            stride: [16, 20, 24, 8][r.below(4) as usize],
                            indirect_registers: IndirectRegisters {
                                vertex_offset: target(&mut r),
                                instance_offset: target(&mut r),
                                first_index: target(&mut r),
                                draw_index: target(&mut r),
                            },
                            ..Default::default()
                        };
                        let instances = [InstanceSource::Packet, InstanceSource::State, InstanceSource::StateRaw][r.below(3) as usize];
                        let fixup = kind.indirect() && r.below(4) != 0;
                        if sliced {
                            ex.log_draw(0, &seq, &packet, instances, fixup);
                        } else {
                            direct.draw(&packet, instances, fixup);
                        }
                    }
                    13 => {
                        let packet = DispatchPacket { groups: [1 + r.u32() % 4, 1, 1], ..Default::default() };
                        if sliced {
                            ex.log_dispatch(0, &seq, &packet);
                        } else {
                            direct.log.dispatch(&resolve_dispatch(&direct.regs.banks(), &packet));
                        }
                    }
                    14 => {
                        // The records change between draws, in order.
                        let w: Vec<u32> = (0..8).map(|_| r.u32() % 9).collect();
                        let at = records + 4 * r.below(48);
                        let data = words(&w);
                        if sliced {
                            let d = data.clone();
                            ex.write_ordered(at, &data, Box::new(move || write(at, &d)));
                        } else {
                            write(at, &data);
                        }
                    }
                    _ => {
                        let which = r.below(3);
                        let target: &dyn RenderExecutor = if sliced { &*ex } else { &*direct.log };
                        match which {
                            0 => target.end_of_pipe(Box::new(|| {})),
                            1 => target.flush(),
                            _ => target.sync_point(),
                        }
                    }
                }
            }
            if sliced {
                ex.drain();
            }
        }
        let (a, b) = (direct_log.take(), sliced_log.take());
        assert_eq!(a.len(), b.len(), "seed {seed}: item count");
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(x, y, "seed {seed}, item {i} (budget {budget})");
        }
        let (handed, done) = ex.progress();
        assert_eq!(handed, done);
        assert!(handed > 1, "seed {seed}: the work was cut into slices");
        assert_eq!(mismatches.load(Ordering::Relaxed), 0, "seed {seed}: snapshots match the state the log left");
        ex.stop();
    }
}

/// CP writes queued behind earlier work: the sequencer's reads compose
/// them, the slice thread's indirect-argument read sees them at the
/// draw's position, and a queued copy makes a read wait.
#[test]
fn positioned_events_order_reads_and_fixups() {
    let mem = arena(0x2000);
    let record = mem + 0x100;
    write(record, &words(&[3, 7, 11, 13, 17]));
    let log = Arc::new(Log::default());
    let runner = Runner::new(true);
    let mismatches = runner.mismatches();
    let ex = SliceExecutor::start(log.clone(), 4, 2, runner);
    let mut regs = RegisterFile::new();
    let packet = DrawPacket {
        kind: DrawKind::IndexIndirect,
        indirect_address: record,
        indirect_count: 1,
        indirect_registers: IndirectRegisters { vertex_offset: 0x8C, instance_offset: 0x8D, first_index: 0x8E, draw_index: SH_NOP },
        ..Default::default()
    };
    // A draw of other state keeps the slice thread busy at first.
    ex.log_run(0, &regs, Bank::Shader, 0x10, &[1]);
    regs.set_run(Bank::Shader, 0x10, &[1]);
    ex.log_draw(0, &regs, &DrawPacket::default(), InstanceSource::State, false);
    // The record is rewritten behind that draw, then the indirect draw.
    let data = words(&[4, 9, 21, 23, 29]);
    let d = data.clone();
    ex.write_ordered(record, &data, Box::new(move || write(record, &d)));
    let mut seen = [0u8; 8];
    assert!(ex.read_ordered(record + 4, &mut seen, &|o: &mut [u8]| guest_read(record + 4, o)));
    assert_eq!(seen, words(&[9, 21])[..], "the sequencer's read composes the queued write");
    ex.log_draw(0, &regs, &packet, InstanceSource::State, true);
    // A later slice starts from a snapshot that carries the fixup.
    ex.log_dispatch(0, &regs, &DispatchPacket::default());
    ex.log_draw(0, &regs, &DrawPacket::default(), InstanceSource::State, false);
    // A queued copy over a range: a read of it waits for it.
    let copied = Arc::new(AtomicBool::new(false));
    let c = copied.clone();
    ex.copy_memory_or(mem + 0x400, record, 20, Box::new(move || {
        std::thread::sleep(Duration::from_millis(20));
        // SAFETY: inside the test arena.
        unsafe { std::ptr::copy_nonoverlapping(record as *const u8, (mem + 0x400) as *mut u8, 20) };
        c.store(true, Ordering::SeqCst);
        true
    }));
    let mut out = [0u8; 4];
    assert!(ex.read_ordered(mem + 0x400, &mut out, &|o: &mut [u8]| guest_read(mem + 0x400, o)));
    assert!(copied.load(Ordering::SeqCst), "the read waited for the copy");
    assert_eq!(out, 4u32.to_le_bytes());
    ex.drain();
    let items = log.take();
    let draws: Vec<&String> = items.iter().filter(|s| s.starts_with("draw")).collect();
    // What the direct path resolves: the record as rewritten.
    let mut direct = RegisterFile::new();
    direct.set_run(Bank::Shader, 0x10, &[1]);
    let first = resolve_draw(&direct.banks(), &DrawPacket { instance_count: 1, ..Default::default() });
    let own = DrawPacket { instance_count: 1, ..packet };
    apply_fixup(&own, &mut direct, &guest_read);
    assert_eq!((direct.shader[0x8C], direct.shader[0x8D], direct.shader[0x8E], direct.num_instances), (23, 29, 21, 9));
    let indirect = resolve_draw(&direct.banks(), &own);
    let after = resolve_draw(&direct.banks(), &DrawPacket { instance_count: 9, ..Default::default() });
    assert_eq!(draws.len(), 3);
    assert_eq!(*draws[0], format!("draw {first:?}"));
    assert_eq!(*draws[1], format!("draw {indirect:?}"), "the indirect draw sees the rewritten record, its own count from before");
    assert_eq!(*draws[2], format!("draw {after:?}"), "a later slice's snapshot carries the record's registers and count");
    assert_eq!(mismatches.load(Ordering::Relaxed), 0);
    ex.stop();
}

/// End-of-pipe work, flips and transfers run where they were queued
/// between the draws.
#[test]
fn ordered_work_keeps_its_place_between_draws() {
    let log = Arc::new(Log::default());
    let ex = SliceExecutor::start(log.clone(), 2, 1, Runner::new(false));
    let regs = RegisterFile::new();
    let order = Arc::new(Mutex::new(Vec::new()));
    for i in 0..6u32 {
        ex.log_draw(0, &regs, &DrawPacket { count: i, ..Default::default() }, InstanceSource::Packet, false);
        let o = order.clone();
        let l = log.clone();
        match i % 3 {
            0 => ex.end_of_pipe(Box::new(move || o.lock().unwrap().push(i))),
            1 => ex.run_ordered(Box::new(move || {
                o.lock().unwrap().push(i);
                l.push(format!("flip {i}"));
            })),
            _ => ex.after_pipe(Box::new(move || o.lock().unwrap().push(i))),
        }
    }
    ex.drain();
    let items = log.take();
    let kinds: Vec<String> = items.iter().map(|s| if s.starts_with("draw") { "draw".to_string() } else { s.clone() }).collect();
    assert_eq!(kinds, ["draw", "eop", "draw", "flip 1", "draw", "after", "draw", "eop", "draw", "flip 4", "draw", "after"]);
    assert_eq!(*order.lock().unwrap(), vec![0, 1, 2, 3, 4, 5]);
    ex.stop();
}

/// Stopping while the sequencer waits for room and the slice thread is
/// inside a slice: everyone returns.
#[test]
fn shutdown_releases_a_blocked_sequencer_and_drains() {
    #[derive(Default)]
    struct Slow {
        release: AtomicBool,
    }
    impl RenderExecutor for Slow {
        fn draw(&self, _: &ResolvedDraw) {
            while !self.release.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        fn dispatch(&self, _: &ResolvedDispatch) {}
    }
    let slow = Arc::new(Slow::default());
    let ex = SliceExecutor::start(slow.clone(), 1, 1, Runner::new(false));
    let regs = RegisterFile::new();
    let e = ex.clone();
    let sequencer = std::thread::spawn(move || {
        // Budget 1, one slice in flight: this blocks for room.
        for i in 0..8 {
            e.log_draw(0, &regs, &DrawPacket { count: i, ..Default::default() }, InstanceSource::Packet, false);
        }
        e.drain();
    });
    std::thread::sleep(Duration::from_millis(30));
    let e = ex.clone();
    let stopper = std::thread::spawn(move || e.stop());
    std::thread::sleep(Duration::from_millis(30));
    slow.release.store(true, Ordering::SeqCst);
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !(sequencer.is_finished() && stopper.is_finished()) {
        assert!(std::time::Instant::now() < deadline, "shutdown deadlocked");
        std::thread::sleep(Duration::from_millis(5));
    }
    sequencer.join().unwrap();
    stopper.join().unwrap();
    // After the stop nothing waits.
    ex.drain();
    ex.log_draw(0, &RegisterFile::new(), &DrawPacket::default(), InstanceSource::Packet, false);
    ex.drain();
}
