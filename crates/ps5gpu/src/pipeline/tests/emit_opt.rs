// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The GPU-cost switches of the SPIR-V emitter (`shader::EmitOptions`)
//! on the device: every program runs through the compiler with the code
//! emitted before the switches and with all of them on, and both are
//! compared word for word with the IR interpreter (and, where the program
//! is a hardware-measured one, with the hardware rows). The programs reach
//! the edges of the rewritten paths: multi-dword buffer accesses partly
//! past the bound range, unaligned offsets, lanes off in EXEC, partial
//! waves, scalar loads of one instruction past the end of their buffer and
//! across a device page, device-address loads and stores across a page,
//! DPP / permute / read-lane forms.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored emit_opt`.

use super::*;
use crate::shader::{EmitOptions, Features, SpirvCompiler};
use ps5shader::testing::float_rows::{midpoint_rows, special_rows, CHAIN_W32, MIX_W32};
use ps5shader::testing::{buffer_descriptor, ComputeRun};

const POISON: u32 = 0xDEAD_BEEF;
const RAW_DWORD3: u32 = 0x3101_6FAC;

/// The default switches on this device.
fn defaults(dev: &GpuDevice) -> EmitOptions {
    EmitOptions { broadcast_dynamic: dev.caps.subgroup_broadcast_dynamic_id, shuffle_relative: dev.caps.subgroup_shuffle_relative, ..EmitOptions::DEFAULT }
}

/// Every exact switch (the opt-in ones too) as this device runs them.
fn all_on(dev: &GpuDevice) -> EmitOptions {
    EmitOptions { guarded_loads: dev.caps.pipeline_robustness, ..EmitOptions { bound_once: true, flush_elision: true, float_class_compare: true, const_fold: true, lazy_select: true, half_fma_fast: true, ..defaults(dev) } }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Emit {
    Before,
    /// The default switches.
    Default,
    /// Every exact switch, the opt-in ones too.
    New,
    /// The new code with the opt-in fast transcendentals.
    Fast,
    /// The new code with one switch off (benchmark bisection).
    Without(&'static str),
    /// The old code with one switch on (benchmark bisection).
    Only(&'static str),
    /// The default switches with one more on.
    Plus(&'static str),
}

/// `o` with the switch `name` on.
fn with(mut o: EmitOptions, name: &str, dev: &GpuDevice) -> EmitOptions {
    let all = all_on(dev);
    match name {
        "log_series_branch" => o.log_series_branch = true,
        "readonly" => o.readonly = true,
        "device_once" => o.device_once = true,
        "bound_once" => o.bound_once = true,
        "subgroup_ops" => {
            o.subgroup_ops = true;
            o.broadcast_dynamic = all.broadcast_dynamic;
            o.shuffle_relative = all.shuffle_relative;
        }
        "guarded_loads" => o.guarded_loads = all.guarded_loads,
        "lean_flush" => o.lean_flush = true,
        "global_cse" => o.global_cse = true,
        "fault_log" => o.fault_log = false,
        "flush_elision" => o.flush_elision = true,
        "float_class_compare" => o.float_class_compare = true,
        "const_fold" => o.const_fold = true,
        "lazy_select" => o.lazy_select = true,
        "half_fma_fast" => o.half_fma_fast = true,
        "nan_once" => o.nan_once = true,
        "zero_max" => o.zero_max = true,
        _ => panic!("no switch {name}"),
    }
    o
}

/// `o` with the switch `name` off.
fn without(mut o: EmitOptions, name: &str) -> EmitOptions {
    match name {
        "log_series_branch" => o.log_series_branch = false,
        "readonly" => o.readonly = false,
        "device_once" => o.device_once = false,
        "bound_once" => o.bound_once = false,
        "subgroup_ops" => o.subgroup_ops = false,
        "guarded_loads" => o.guarded_loads = false,
        "lean_flush" => o.lean_flush = false,
        "global_cse" => o.global_cse = false,
        "flush_elision" => o.flush_elision = false,
        "float_class_compare" => o.float_class_compare = false,
        "const_fold" => o.const_fold = false,
        "lazy_select" => o.lazy_select = false,
        "half_fma_fast" => o.half_fma_fast = false,
        "nan_once" => o.nan_once = false,
        "zero_max" => o.zero_max = false,
        _ => panic!("no switch {name}"),
    }
    o
}

fn rig(g: &Guest, emit: Emit) -> Rig {
    Rig::with_compiler(
        g,
        |dev| {
            let options = match emit {
                Emit::Before => EmitOptions::BEFORE,
                Emit::Default => defaults(dev),
                Emit::New => all_on(dev),
                Emit::Fast => EmitOptions { fast_transcendental: true, ..defaults(dev) },
                Emit::Without(name) => without(all_on(dev), name),
                Emit::Only(name) => with(EmitOptions::BEFORE, name, dev),
                Emit::Plus(name) => with(defaults(dev), name, dev),
            };
            let features = Features { emit: options, ..Features::of(dev) };
            Arc::new(SpirvCompiler::new(features, Some(g.mem)))
        },
        None,
        None,
    )
}

thread_local! {
    /// One guest memory and rig per variant for the test (each guest takes
    /// a window of the test address space below 1 TiB, where device
    /// addresses translate).
    static DEVICES: std::cell::RefCell<Vec<(Emit, Guest, Rig)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Drop this test's guest memories and rigs (a test of many runs starts
/// over before the guests' address windows run out).
fn reset_devices() {
    DEVICES.with(|d| d.borrow_mut().clear());
}

/// Drops the test's rigs when the test ends: left to the thread-local's
/// destructor they were destroyed after the test thread reported its
/// result, while the process was already exiting and the driver tearing
/// down (the test binary segfaulted after every test had passed).
struct DevicesScope;

impl Drop for DevicesScope {
    fn drop(&mut self) {
        reset_devices();
    }
}

/// Run `f` on this test's guest memory and rig of `emit`.
fn with_device<T>(emit: Emit, f: impl FnOnce(&mut Guest, &mut Rig) -> T) -> T {
    DEVICES.with(|d| {
        let mut d = d.borrow_mut();
        let i = match d.iter().position(|x| x.0 == emit) {
            Some(i) => i,
            None => {
                let g = Guest::new(1 << 30);
                let rig = rig(&g, emit);
                d.push((emit, g, rig));
                d.len() - 1
            }
        };
        let (_, g, rig) = &mut d[i];
        // Each run's memory starts 16 MiB past the last one: away from the
        // pages the earlier runs' mirrors and watches cover.
        const GAP: u64 = 16 << 20;
        g.next = (g.next + GAP).next_multiple_of(GAP);
        f(g, rig)
    })
}

/// One program run: the input buffer at s[0:3], the output buffer at
/// s[4:7] (raw V#s), more user data after them, and a guest block.
#[derive(Clone)]
struct Run<'a> {
    code: &'a [u32],
    wave: u32,
    lanes: u32,
    input: Vec<u32>,
    /// Bytes of the input V# (records); the whole input when 0.
    input_records: u32,
    out_words: usize,
    /// Bytes of the output V#; the whole output when 0.
    out_records: u32,
    /// User data from s8 (s[0:7] are the V#s).
    user: Vec<u32>,
    /// A 64 KiB guest block: its contents and the user SGPR pairs that
    /// get its address plus an offset. Its contents after the run are
    /// compared too.
    block: Option<(Vec<u8>, Vec<(usize, u64)>)>,
    /// COMPUTE_PGM_RSRC1 float controls: FLOAT_MODE (bits 7:0) and
    /// IEEE_MODE (bit 11, RSRC1 bit 23).
    float_mode: u32,
}

impl<'a> Run<'a> {
    fn new(code: &'a [u32], wave: u32, lanes: u32, input: Vec<u32>, out_words: usize) -> Run<'a> {
        Run { code, wave, lanes, input, input_records: 0, out_words, out_records: 0, user: Vec::new(), block: None, float_mode: 0xC0 }
    }

    fn descriptors(&self, input: u64, output: u64, block: u64) -> Vec<u32> {
        let in_records = if self.input_records == 0 { self.input.len() as u32 * 4 } else { self.input_records };
        let out_records = if self.out_records == 0 { self.out_words as u32 * 4 } else { self.out_records };
        let mut user = buffer_descriptor(input, 0, in_records, RAW_DWORD3).to_vec();
        user.extend(buffer_descriptor(output, 0, out_records, RAW_DWORD3));
        user.extend(&self.user);
        if let Some((_, pairs)) = &self.block {
            for &(p, offset) in pairs {
                if user.len() < p + 2 {
                    user.resize(p + 2, 0);
                }
                let a = block + offset;
                user[p] = a as u32;
                user[p + 1] = (a >> 32) as u32;
            }
        }
        user
    }

    /// COMPUTE_PGM_RSRC1 with the float controls and DX10_CLAMP.
    fn rsrc1(&self) -> u32 {
        ((self.float_mode & 0xFF) << 12) | (1 << 21) | if self.float_mode & 0x800 != 0 { 1 << 23 } else { 0 }
    }

    fn block_words(&self) -> usize {
        self.block.as_ref().map_or(0, |(b, _)| b.len() / 4)
    }

    /// The interpreter's output words, then the block's (guest addresses
    /// as on the device).
    fn interpret(&self, input: u64, output: u64, block: u64) -> Result<Vec<u32>, String> {
        let user = self.descriptors(input, output, block);
        let mut run = ComputeRun::new(self.code.to_vec(), self.wave, self.lanes.min(self.wave), user);
        run.float_mode = ps5shader::translate::FloatMode::from_rsrc1(self.rsrc1(), 26);
        let f = run.compile(true)?;
        let mut memory = ps5shader::interp::Memory::default();
        memory.write_words(input, &self.input);
        memory.write_words(output, &vec![POISON; self.out_words]);
        if let Some((bytes, _)) = &self.block {
            let words: Vec<u32> = bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            memory.write_words(block, &words);
        }
        memory.write_words(run.code_address, &run.code);
        let mut first = 0;
        while first < self.lanes {
            let n = self.wave.min(self.lanes - first);
            let mut m = ps5shader::interp::Machine::new(self.wave, n);
            m.user_data = run.user_data.clone();
            m.memory = std::mem::take(&mut memory);
            m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(0), (first..first + n).collect());
            m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(1), vec![0; n as usize]);
            m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(2), vec![0; n as usize]);
            m.run(&f).map_err(|e| format!("run: {e}"))?;
            memory = std::mem::take(&mut m.memory);
            first += n;
        }
        let mut out = memory.read_words(output, self.out_words);
        out.extend(memory.read_words(block, self.block_words()));
        Ok(out)
    }

    /// The device's output words (then the block's) and the interpreter's
    /// for the same addresses.
    fn device(&self, emit: Emit) -> (Vec<u32>, Vec<u32>) {
        with_device(emit, |g, rig| self.device_on(emit, g, rig))
    }

    fn device_on(&self, emit: Emit, g: &mut Guest, rig: &mut Rig) -> (Vec<u32>, Vec<u32>) {
        let cs = g.program(self.code);
        let ib = g.alloc(&asm::bytes(&self.input));
        let ob = g.alloc(&asm::bytes(&vec![POISON; self.out_words]));
        let block = self.block.as_ref().map_or(0, |(bytes, _)| g.alloc(bytes));
        let user = self.descriptors(ib, ob, block);
        let mut shader = stage(cs, &user);
        shader.rsrc1 = self.rsrc1();
        shader.rsrc2 = (user.len() as u32) << 1;
        let d = ResolvedDispatch {
            packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1 | if self.wave == 32 { 1 << 15 } else { 0 }, ..DispatchPacket::default() },
            shader,
            rsrc3: 0,
            threads: [self.lanes, 1, 1],
            start: [0; 3],
        };
        assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded, "{emit:?}: dispatch");
        let read = |rig: &mut Rig, g: &Guest, a: u64, words: usize| -> Vec<u32> {
            assert!(rig.buffers.download(&mut rig.sched, a, words as u64 * 4));
            let mut bytes = vec![0u8; words * 4];
            assert!(g.mem.read_backing(a, &mut bytes));
            bytes.chunks(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect()
        };
        let mut words = read(rig, g, ob, self.out_words);
        if self.block.is_some() {
            words.extend(read(rig, g, block, self.block_words()));
        }
        let want = self.interpret(ib, ob, block).unwrap_or_else(|e| panic!("interpreter: {e}"));
        (words, want)
    }
}

/// Words where `got` differs from `want` (NaNs compare equal).
fn differences(name: &str, emit: Emit, got: &[u32], want: &[u32]) -> Vec<String> {
    let nan = |b: u32| b & 0x7FFF_FFFF > 0x7F80_0000;
    got.iter()
        .zip(want)
        .enumerate()
        .filter(|(_, (a, e))| a != e && !(nan(**a) && nan(**e)))
        .map(|(i, (a, e))| format!("{name} {emit:?} word {i}: device {a:#010x}, interpreter {e:#010x}"))
        .collect()
}

/// Run `r` with the old and the new code: both must match the
/// interpreter; returns the new code's output.
fn check_both(name: &str, r: &Run, fails: &mut Vec<String>) -> Vec<u32> {
    check_both_except(name, r, &|_| false, fails)
}

/// [`check_both`] where the words `apart(i)` need only agree between the
/// old and the new code (an existing divergence from the interpreter).
fn check_both_except(name: &str, r: &Run, apart: &dyn Fn(usize) -> bool, fails: &mut Vec<String>) -> Vec<u32> {
    let (before, want) = r.device(Emit::Before);
    let (default, _) = r.device(Emit::Default);
    let (new, _) = r.device(Emit::New);
    for (emit, got) in [(Emit::Before, &before), (Emit::Default, &default), (Emit::New, &new)] {
        let mut d = differences(name, emit, got, &want);
        d.retain(|m| !m.split(" word ").nth(1).and_then(|t| t.split(':').next()).and_then(|t| t.parse::<usize>().ok()).is_some_and(apart));
        fails.extend(d);
    }
    for (i, ((a, d), b)) in new.iter().zip(&default).zip(&before).enumerate() {
        if apart(i) && (a != b || d != b) {
            fails.push(format!("{name} word {i}: new {a:#010x}, default {d:#010x}, before {b:#010x}"));
        }
    }
    new
}

fn report(fails: Vec<String>) {
    let mut runs: Vec<(String, usize)> = Vec::new();
    for f in &fails {
        let key = f.split(" word ").next().unwrap_or(f).split(" lane ").next().unwrap_or(f).to_string();
        match runs.iter_mut().find(|r| r.0 == key) {
            Some(r) => r.1 += 1,
            None => runs.push((key, 1)),
        }
    }
    let summary: Vec<String> = runs.iter().map(|(k, n)| format!("{k}: {n}")).collect();
    assert!(fails.is_empty(), "{} mismatches ({}):\n{}", fails.len(), summary.join("; "), fails.iter().take(60).cloned().collect::<Vec<_>>().join("\n"));
}

/// Input words with a distinct pattern per word.
fn pattern(words: usize, seed: u32) -> Vec<u32> {
    (0..words as u32).map(|i| (i.wrapping_mul(0x9E37_79B9) ^ seed).rotate_left(i % 29) | 1).collect()
}

// Programs assembled with llvm-mc (gfx1013); the source is in each comment.

/// v1 = tid*4, v2 = v1 + 2; buffer_load_dwordx4 v[10:13], v1 offen;
/// buffer_load_dwordx3 v[14:16], v2 offen offset:8; buffer_load_dwordx2
/// v[17:18], v1 offen offset:4; buffer_load_dword v19, v2 offen offset:1;
/// stores of them at tid*64 (+0/16/28/36); lanes below 20 (wave64: 40)
/// store v[10:13] at +40; every lane stores v[14:17] at +56.
const BUFFERS_W32: [u32; 28] = [
    0x34020082, 0x4a040282, 0xe0381000, 0x80000a01, 0xe03c1008, 0x80000e02, 0xe0341004, 0x80001101, 0xe0301001, 0x80001302,
    0xbf8c3f70, 0x34060086, 0xe0781000, 0x80010a03, 0xe07c1010, 0x80010e03, 0xe074101c, 0x80011103, 0xe0701024, 0x80011303,
    0x7d880094, 0xbe943c6a, 0xe0781028, 0x80010a03, 0xbefe0314, 0xe0781038, 0x80010e03, 0xbf810000,
];
const BUFFERS_W64: [u32; 28] = [
    0x34020082, 0x4a040282, 0xe0381000, 0x80000a01, 0xe03c1008, 0x80000e02, 0xe0341004, 0x80001101, 0xe0301001, 0x80001302,
    0xbf8c3f70, 0x34060086, 0xe0781000, 0x80010a03, 0xe07c1010, 0x80010e03, 0xe074101c, 0x80011103, 0xe0701024, 0x80011303,
    0x7d8800a8, 0xbe94246a, 0xe0781028, 0x80010a03, 0xbefe0414, 0xe0781038, 0x80010e03, 0xbf810000,
];

/// s_buffer_load_dwordx8 s[16:23], s[0:3], 0x4c; s_buffer_load_dwordx16
/// s[24:39], s[0:3], 0x40; s_buffer_load_dwordx4 s[40:43], s[0:3], s12;
/// s_buffer_load_dwordx2 s[44:45], s[0:3], 0x5e; s_load_dwordx8 s[48:55],
/// s[14:15], 0; s_load_dwordx4 s[56:59], s[14:15], 0x24; s_load_dword
/// s60, s[14:15], 4; every lane stores s16..s45, s48..s60 (and a 0) at
/// tid*256.
const SCALAR_W32: [u32; 83] = [
    0xf42c0400, 0xfa00004c, 0xf4300600, 0xfa000040, 0xf4280a00, 0x18000000, 0xf4240b00, 0xfa00005e, 0xf40c0c07, 0xfa000000,
    0xf4080e07, 0xfa000024, 0xf4000f07, 0xfa000004, 0xbf8cc07f, 0x34060088, 0x7e140210, 0x7e160211, 0x7e180212, 0x7e1a0213,
    0x7e1c0214, 0x7e1e0215, 0x7e200216, 0x7e220217, 0x7e240218, 0x7e260219, 0x7e28021a, 0x7e2a021b, 0x7e2c021c, 0x7e2e021d,
    0x7e30021e, 0x7e32021f, 0x7e340220, 0x7e360221, 0x7e380222, 0x7e3a0223, 0x7e3c0224, 0x7e3e0225, 0x7e400226, 0x7e420227,
    0x7e440228, 0x7e460229, 0x7e48022a, 0x7e4a022b, 0x7e4c022c, 0x7e4e022d, 0x7e500230, 0x7e520231, 0x7e540232, 0x7e560233,
    0x7e580234, 0x7e5a0235, 0x7e5c0236, 0x7e5e0237, 0x7e600238, 0x7e620239, 0x7e64023a, 0x7e66023b, 0x7e68023c, 0x7e6a0280,
    0xe0781000, 0x80010a03, 0xe0781010, 0x80010e03, 0xe0781020, 0x80011203, 0xe0781030, 0x80011603, 0xe0781040, 0x80011a03,
    0xe0781050, 0x80011e03, 0xe0781060, 0x80012203, 0xe0781070, 0x80012603, 0xe0781080, 0x80012a03, 0xe0781090, 0x80012e03,
    0xe07810a0, 0x80013203, 0xbf810000,
];

/// v4 = input[tid * 4]; s16 = v_readlane v4, 5 (wave64: 40); s17 =
/// v_readlane v4, s12; with EXEC = lanes > 3 (wave64: > 35) s18 =
/// v_readfirstlane v4; every lane stores s16..s18 at tid*64.
const LANES_W32: [u32; 19] = [
    0x34020084, 0xe0301000, 0x80000401, 0xbf8c3f70, 0xd7600010, 0x00010b04, 0xd7600011, 0x00001904, 0x7d820083, 0xbe943c6a,
    0x7e240504, 0xbefe0314, 0x7e140210, 0x7e160211, 0x7e180212, 0x34060086, 0xe07c1000, 0x80010a03, 0xbf810000,
];
const LANES_W64: [u32; 19] = [
    0x34020084, 0xe0301000, 0x80000401, 0xbf8c3f70, 0xd7600010, 0x00015104, 0xd7600011, 0x00001904, 0x7d8200a3, 0xbe94246a,
    0x7e240504, 0xbefe0414, 0x7e140210, 0x7e160211, 0x7e180212, 0x34060086, 0xe07c1000, 0x80010a03, 0xbf810000,
];

/// v1 = tid*4 + 0x3ff0 (across the 16 KiB page at 0x4000);
/// global_load_dwordx4 v[10:13], v1, s[14:15]; global_load_dwordx2
/// v[14:15], v1 + 2 (unaligned), s[14:15]; global_load_dword v16, v1,
/// s[14:15] offset:-4; global_load_dwordx3 v[17:19], v1, s[14:15]
/// offset:8; stores of them at tid*64; lanes below 9 store v[10:13] with
/// global_store_dwordx4 v3 (tid*64), s[16:17].
const GLOBAL_W32: [u32; 26] = [
    0x34020082, 0x4a0202ff, 0x00003ff0, 0xdc388000, 0x0a0e0001, 0x4a040282, 0xdc348000, 0x0e0e0002, 0xdc308ffc, 0x100e0001,
    0xdc3c8008, 0x110e0001, 0xbf8c3f70, 0x34060086, 0xe0781000, 0x80010a03, 0xe0781010, 0x80010e03, 0xe0741020, 0x80011203,
    0x7d880089, 0xbe943c6a, 0xdc788000, 0x00100a03, 0xbefe0314, 0xbf810000,
];

/// Multi-dword raw loads and stores of bound buffers: parts past the
/// input's 100 bytes and the output's end, unaligned offsets, lanes off in
/// EXEC, partial waves (wave64 runs as two guest lanes per invocation on a
/// 32-wide subgroup).
#[test]
#[ignore]
fn emit_opt_bound_buffer_accesses_match_the_interpreter() {
    let _devices = DevicesScope;
    let mut fails = Vec::new();
    for (code, wave) in [(&BUFFERS_W32[..], 32u32), (&BUFFERS_W64[..], 64)] {
        for lanes in [wave, wave - 5] {
            for input_records in [100u32, 64 * 4 + 6] {
                let mut r = Run::new(code, wave, lanes, pattern(80, 0x5151), lanes as usize * 16);
                r.input_records = input_records;
                r.out_records = lanes * 64 - 20;
                check_both(&format!("buffers w{wave} lanes {lanes} records {input_records}"), &r, &mut fails);
            }
        }
    }
    report(fails);
}

/// The dwords of one scalar load: `s_buffer_load` groups ending past the
/// buffer's 100 bytes, an unaligned SOFFSET and immediate; `s_load` groups
/// from a device address within a page and across the 16 KiB page.
#[test]
#[ignore]
fn emit_opt_scalar_load_groups_match_the_interpreter() {
    let _devices = DevicesScope;
    let block: Vec<u8> = pattern(16384, 0x7777).iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut fails = Vec::new();
    for offset in [0x100u64, 0x3FF4, 0x3FE0] {
        for lanes in [1u32, 32] {
            let mut r = Run::new(&SCALAR_W32, 32, lanes, pattern(32, 0x1234), lanes as usize * 64);
            r.input_records = 100;
            // s12 = 6 (unaligned SOFFSET); s[14:15] = block + offset.
            r.user = vec![0, 0, 0, 0, 6, 0];
            r.block = Some((block.clone(), vec![(14, offset)]));
            check_both(&format!("scalar loads at block+{offset:#x} lanes {lanes}"), &r, &mut fails);
        }
    }
    report(fails);
}

/// v_readlane with a constant and an SGPR lane, v_readfirstlane under a
/// partial EXEC; full and partial waves.
#[test]
#[ignore]
fn emit_opt_read_lanes_match_the_interpreter() {
    let _devices = DevicesScope;
    let mut fails = Vec::new();
    for (code, wave, lane) in [(&LANES_W32[..], 32u32, 7u32), (&LANES_W64[..], 64, 37)] {
        for lanes in [wave, wave - 4] {
            let mut r = Run::new(code, wave, lanes, pattern(lanes as usize * 4, 0x4242), lanes as usize * 16);
            r.user = vec![0, 0, 0, 0, lane];
            check_both(&format!("read lanes w{wave} lanes {lanes}"), &r, &mut fails);
        }
    }
    report(fails);
}

/// Device-address loads of 2..4 dwords across a page, at unaligned
/// addresses, and stores of 4 dwords across it.
#[test]
#[ignore]
fn emit_opt_device_accesses_match_the_interpreter() {
    let _devices = DevicesScope;
    let block: Vec<u8> = pattern(16384, 0x9999).iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut fails = Vec::new();
    for lanes in [32u32, 13] {
        let mut r = Run::new(&GLOBAL_W32, 32, lanes, pattern(4, 1), lanes as usize * 16);
        // Lane 0's store crosses the page at 0x8000 (away from the loads).
        r.block = Some((block.clone(), vec![(14, 0), (16, 0x7FF8)]));
        // Compared between the old and the new code only:
        // - the unaligned dwordx2 (words 4, 5 of each lane): the
        //   interpreter reads unaligned device dwords byte for byte, the
        //   device code aligns them down;
        // - the block after the stores: this rig does not write device
        //   stores back to guest memory for the read-back.
        let out = lanes as usize * 16;
        let apart = |i: usize| i >= out || matches!(i % 16, 4 | 5);
        check_both_except(&format!("device accesses lanes {lanes}"), &r, &apart, &mut fails);
    }
    report(fails);
}

#[allow(dead_code, clippy::all)]
mod dpp16_inactive {
    include!("../../../../ps5shader/tests/vectors/fmod_dpp16_inactive_source.rs");
    pub fn waves() -> [(&'static [u32], u32, &'static [[u32; 4]], Vec<Vec<u32>>); 2] {
        [(&WAVE32_CODE, 32, &ROWS, EXPECTED32.iter().map(|r| r.to_vec()).collect()), (&WAVE64_CODE, 64, &ROWS, EXPECTED64.iter().map(|r| r.to_vec()).collect())]
    }
}

#[allow(dead_code, clippy::all)]
mod dpp16_share {
    include!("../../../../ps5shader/tests/vectors/fmod_dpp16_row_share_xmask.rs");
    pub fn waves() -> [(&'static [u32], u32, &'static [[u32; 4]], Vec<Vec<u32>>); 2] {
        [(&WAVE32_CODE, 32, &ROWS, EXPECTED32.iter().map(|r| r.to_vec()).collect()), (&WAVE64_CODE, 64, &ROWS, EXPECTED64.iter().map(|r| r.to_vec()).collect())]
    }
}

#[allow(dead_code, clippy::all)]
mod dpp8 {
    include!("../../../../ps5shader/tests/vectors/fmod_dpp8.rs");
    pub fn waves() -> [(&'static [u32], u32, &'static [[u32; 4]], Vec<Vec<u32>>); 2] {
        [(&WAVE32_CODE, 32, &ROWS, EXPECTED32.iter().map(|r| r.to_vec()).collect()), (&WAVE64_CODE, 64, &ROWS, EXPECTED64.iter().map(|r| r.to_vec()).collect())]
    }
}

#[allow(dead_code, clippy::all)]
mod permlane16 {
    include!("../../../../ps5shader/tests/vectors/fmod_permlane16.rs");
    pub fn waves() -> [(&'static [u32], u32, &'static [[u32; 4]], Vec<Vec<u32>>); 2] {
        [(&WAVE32_CODE, 32, &ROWS, EXPECTED32.iter().map(|r| r.to_vec()).collect()), (&WAVE64_CODE, 64, &ROWS, EXPECTED64.iter().map(|r| r.to_vec()).collect())]
    }
}

#[allow(dead_code, clippy::all)]
mod dpp16_operands {
    include!("../../../../ps5shader/tests/vectors/fmod_dpp16_operands.rs");
    pub fn wave() -> (&'static [u32], u32, &'static [[u32; 4]], Vec<Vec<u32>>) {
        (&CODE, 32, &ROWS, EXPECTED.iter().map(|r| r.to_vec()).collect())
    }
}

#[allow(dead_code, clippy::all)]
mod ds_permute {
    include!("../../../../ps5shader/tests/vectors/mem_ds_permute.rs");
    pub fn wave() -> (&'static [u32], Vec<[u32; 4]>, Vec<Vec<u32>>) {
        (&CODE, ROWS.iter().map(|r| [r[0], r[1], 0, 0]).collect(), EXPECTED.iter().map(|r| r.to_vec()).collect())
    }
}

#[allow(dead_code, clippy::all)]
mod load_store {
    include!("../../../../ps5shader/tests/vectors/mem_load_store.rs");
    pub fn waves() -> [(&'static [u32], u32, Vec<Vec<u32>>); 2] {
        [(&WAVE32_CODE, 32, EXPECTED32.iter().map(|r| r.to_vec()).collect()), (&WAVE64_CODE, 64, EXPECTED64.iter().map(|r| r.to_vec()).collect())]
    }
    pub fn input(lanes: usize) -> Vec<u32> {
        let mut v: Vec<u32> = ROWS[..lanes].iter().flatten().copied().collect();
        v.extend_from_slice(&SHARED_INIT);
        v
    }
}

/// Lanes x columns of hardware rows: every lane's first `columns` words
/// of `stride` per lane must equal them.
fn hardware(name: &str, out: &[u32], expected: &[Vec<u32>], stride: usize, fails: &mut Vec<String>) {
    for (lane, row) in expected.iter().enumerate() {
        for (k, &e) in row.iter().enumerate() {
            let a = out[lane * stride + k];
            if a != e {
                fails.push(format!("{name} lane {lane} column {k}: device {a:#010x}, hardware {e:#010x}"));
            }
        }
    }
}

/// The hardware-measured DPP16 (inactive sources, row share / xmask,
/// operand forms), DPP8, permlane16 and ds_permute programs.
#[test]
#[ignore]
fn emit_opt_cross_lane_rows_match_hardware() {
    let _devices = DevicesScope;
    let mut fails = Vec::new();
    let mut cases: Vec<(&str, &[u32], u32, Vec<u32>, Vec<Vec<u32>>)> = Vec::new();
    for (name, waves) in [("dpp16 inactive", dpp16_inactive::waves()), ("dpp16 share/xmask", dpp16_share::waves()), ("dpp8", dpp8::waves()), ("permlane16", permlane16::waves())] {
        for (code, wave, rows, expected) in waves {
            cases.push((name, code, wave, rows.iter().flatten().copied().collect(), expected));
        }
    }
    let (code, wave, rows, expected) = dpp16_operands::wave();
    cases.push(("dpp16 operands", code, wave, rows.iter().flatten().copied().collect(), expected));
    let (code, rows, expected) = ds_permute::wave();
    cases.push(("ds_permute", code, 32, rows.iter().flatten().copied().collect(), expected));
    for (name, code, wave, input, expected) in cases {
        let lanes = expected.len() as u32;
        let r = Run::new(code, wave, lanes, input, lanes as usize * 16);
        let name = format!("{name} w{wave}");
        let out = check_both(&name, &r, &mut fails);
        hardware(&name, &out, &expected, 16, &mut fails);
    }
    report(fails);
}

/// The hardware-measured buffer load / store program (sub-dword and
/// multi-dword loads, soffset / offset forms, a per-lane scratch region).
#[test]
#[ignore]
fn emit_opt_buffer_rows_match_hardware() {
    let _devices = DevicesScope;
    let mut fails = Vec::new();
    for (code, wave, expected) in load_store::waves() {
        let lanes = wave as usize;
        let r = Run::new(code, wave, wave, load_store::input(lanes), lanes * 128);
        let name = format!("load/store w{wave}");
        let out = check_both(&name, &r, &mut fails);
        hardware(&name, &out, &expected, 128, &mut fails);
    }
    report(fails);
}

#[allow(dead_code, clippy::all)]
mod float_unary {
    include!("../../../../ps5shader/tests/vectors/valu_vop1_float_unary.rs");
    pub fn case() -> (&'static [u32], Vec<u32>, usize, usize) {
        (&CODE, ROWS.iter().flatten().copied().collect(), RESULTS as usize, ROWS.len())
    }
}

/// `RPS5E_FAST_TRANSCENDENTAL=1` (opt-in, inexact): the f32 unary
/// program (exp/log/rcp/rsq/sqrt among sin/cos/fract/rounding) stays
/// within 2^-20 relative (or 2^-126 absolute: host denormal results)
/// of the exact translation; NaNs stay NaNs; the operations it leaves
/// alone are exact.
#[test]
#[ignore]
fn emit_opt_fast_transcendentals_stay_close() {
    let _devices = DevicesScope;
    let (code, input, results, lanes) = float_unary::case();
    let r = Run::new(code, 32, lanes as u32, input, lanes * results);
    let (fast, want) = r.device(Emit::Fast);
    let (exact, _) = r.device(Emit::Default);
    let nan = |b: u32| b & 0x7FFF_FFFF > 0x7F80_0000;
    let mut fails = Vec::new();
    let mut inexact = 0;
    for (i, ((&a, &e), &x)) in fast.iter().zip(&want).zip(&exact).enumerate() {
        if a == x || (nan(a) && nan(x)) {
            continue;
        }
        inexact += 1;
        let (fa, fe) = (f32::from_bits(a), f32::from_bits(e));
        let close = !nan(a) && !nan(e) && ((fa - fe).abs() <= 2f32.powi(-20) * fe.abs() || (fa - fe).abs() <= f32::MIN_POSITIVE || (fa.is_infinite() && fa == fe));
        if !close {
            fails.push(format!("word {i} (lane {}, column {}): fast {a:#010x}, exact {e:#010x}", i / results, i % results));
        }
    }
    eprintln!("fast transcendentals: {inexact} of {} words differ from the exact translation", fast.len());
    report(fails);
}

/// v4..v7 = input[tid * 4..]; s[16:23] = input[0..8]; 128 times: v_mul,
/// v_exp, v_log, v_mul, v_rcp, v_mad, v_sqrt, v_mul, v_add, v_mul, v_mul,
/// v_fma, v_mul, v_add, v_mul (f32); v[4:7] stored at tid * 16.
const ALU_W64: [u32; 36] = [
    0x34020084, 0xe0381000, 0x80000401, 0xf42c0400, 0xfa000000, 0xbf8c0070, 0xbe9e03ff, 0x00000080, 0xd5080005, 0x00002104,
    0x7e0c4b05, 0x7e0e4f06, 0xd5080008, 0x00002307, 0x7e125506, 0xd5410004, 0x04242508, 0x7e146704, 0x1016150a, 0x06081704,
    0xd5080004, 0x00002704, 0x10180904, 0xd54b000d, 0x0410290c, 0xd508000e, 0x00002b0d, 0x06081d04, 0xd5080004, 0x00002504,
    0x809e811e, 0xbf07801e, 0xbf85ffe7, 0xe0781000, 0x80010401, 0xbf810000,
];

/// 64 times: buffer_load_dwordx4 at tid * 16 + 4096 * i, buffer_load_dwordx2
/// 2048 bytes further, s_buffer_load_dwordx4 at 16 * i, integer sums of
/// them; v[4:5] stored at tid * 8.
const MEM_W64: [u32; 32] = [
    0x34020084, 0x7e080280, 0x7e0a0280, 0xbe9e03c0, 0xbe9f0380, 0xe0381000, 0x80001401, 0xe0341800, 0x80001801, 0xf4280600,
    0x3e000000, 0xbf8c0070, 0x4a082904, 0x3a0a2b05, 0x4a082d04, 0x3a0a2f05, 0x4a083104, 0x3a0a3305, 0x4a080818, 0x3a0a0a19,
    0x4a08081a, 0x3a0a0a1b, 0x4a0202ff, 0x00001000, 0x801f901f, 0x809e811e, 0xbf07801e, 0xbf85ffe9, 0x34040083, 0xe0741000,
    0x80010402, 0xbf810000,
];

/// MEM_W64 through device addresses: 64 times global_load_dwordx4 at
/// s[14:15] + tid * 16 + 1024 * i, global_load_dwordx2 1024 bytes further,
/// s_load_dwordx8 at s[14:15] + 32 * i, integer sums of them.
const DEV_W64: [u32; 32] = [
    0x34020084, 0x7e080280, 0x7e0a0280, 0xbe9e03c0, 0xbe9f0380, 0xdc388000, 0x140e0001, 0xdc348400, 0x180e0001, 0xf40c0807,
    0x3e000000, 0xbf8c0070, 0x4a082904, 0x3a0a2b05, 0x4a082d04, 0x3a0a2f05, 0x4a083104, 0x3a0a3305, 0x4a080820, 0x3a0a0a21,
    0x4a080826, 0x3a0a0a27, 0x4a0202ff, 0x00000400, 0x801fa01f, 0x809e811e, 0xbf07801e, 0xbf85ffe9, 0x34040083, 0xe0741000,
    0x80010402, 0xbf810000,
];

/// GPU time of `groups` work-groups of 64 (wave64) running `code`: the
/// best of five submissions after a warm-up (wall time of submit and wait).
fn time_dispatch(emit: Emit, code: &[u32], input: &[u32], out_words: usize, groups: u32) -> f64 {
    with_device(emit, |g, rig| time_dispatch_on(g, rig, code, input, out_words, groups))
}

fn time_dispatch_on(g: &mut Guest, rig: &mut Rig, code: &[u32], input: &[u32], out_words: usize, groups: u32) -> f64 {
    let cs = g.program(code);
    let ib = g.alloc(&asm::bytes(input));
    let ob = g.alloc(&asm::bytes(&vec![0u32; out_words]));
    let mut user = buffer_descriptor(ib, 0, input.len() as u32 * 4, RAW_DWORD3).to_vec();
    user.extend(buffer_descriptor(ob, 0, out_words as u32 * 4, RAW_DWORD3));
    // s[14:15]: the input's device address.
    user.resize(16, 0);
    user[14] = ib as u32;
    user[15] = (ib >> 32) as u32;
    let mut shader = stage(cs, &user);
    shader.rsrc1 = (0xC0 << 12) | (1 << 21);
    shader.rsrc2 = (user.len() as u32) << 1;
    let d = ResolvedDispatch {
        packet: DispatchPacket { groups: [groups, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() },
        shader,
        rsrc3: 0,
        threads: [64, 1, 1],
        start: [0; 3],
    };
    assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
    rig.sched.flush_and_wait();
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
        rig.sched.flush_and_wait();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

/// Synthetic GPU-time comparison (printed, not asserted): an f32 kernel
/// heavy in v_mul / v_exp / v_log / v_rcp / v_sqrt and a load-heavy one
/// (buffer_load_dwordx4 / x2, s_buffer_load_dwordx4 per iteration).
#[test]
#[ignore]
fn emit_opt_gpu_time_benchmark() {
    let _devices = DevicesScope;
    let alu_input: Vec<u32> = (0..64 * 4)
        .map(|k| match k {
            0 => 0.9f32,
            1 => 1.1,
            2 | 3 => 0.5,
            4 => 0.25,
            5 => 0.75,
            _ => 1.0 + (k % 7) as f32 * 0.05,
        })
        .map(f32::to_bits)
        .collect();
    let mem_input = pattern(72 << 10, 0x3131);
    for (name, code, input, out_words, groups) in [("alu", &ALU_W64[..], &alu_input, 64 * 4, 65536u32), ("memory", &MEM_W64[..], &mem_input, 64 * 2, 65536), ("device", &DEV_W64[..], &mem_input, 64 * 2, 65536)] {
        let mut line = format!("gpu time {name} ({groups} groups of 64):");
        let mut variants = vec![Emit::Before, Emit::Default, Emit::New, Emit::Fast];
        // RPS5E_TEST_BISECT=1: also the new code without each switch.
        if std::env::var_os("RPS5E_TEST_BISECT").is_some() {
            for name in ["log_series_branch", "readonly", "device_once", "bound_once", "subgroup_ops", "guarded_loads", "lean_flush", "global_cse"] {
                variants.push(Emit::Without(name));
            }
            for name in ["log_series_branch", "readonly", "device_once", "bound_once", "subgroup_ops", "guarded_loads", "lean_flush", "global_cse", "fault_log"] {
                variants.push(Emit::Only(name));
            }
        }
        for emit in variants {
            let ms = time_dispatch(emit, code, input, out_words, groups);
            line += &format!(" {emit:?} {ms:.3} ms");
        }
        eprintln!("{line}");
    }
}

macro_rules! float_programs {
    ($($m:ident => $file:literal, $($code:ident),+;)*) => {
        $(
            #[allow(dead_code, unused_imports, clippy::all)]
            mod $m {
                include!(concat!("../../../../ps5shader/tests/vectors/", $file));
                pub fn programs() -> Vec<&'static [u32]> {
                    vec![$(&$code[..]),+]
                }
            }
        )*
        fn float_programs() -> Vec<(&'static str, &'static [u32])> {
            let mut v = Vec::new();
            $(for code in $m::programs() { v.push((stringify!($m), code)); })*
            v
        }
    };
}

float_programs! {
    fp_denormal_flush => "valu_f32_denormal_flush.rs", CODE, TINY_CODE, MIXED_TINY_CODE;
    fp_arithmetic_nan => "valu_f32_arithmetic_nan.rs", CODE;
    fp_min_max_nan => "valu_f32_min_max_nan.rs", CODE;
    fp_output_modifier => "valu_f32_output_modifier.rs", CODE;
    fp_result_clamp => "valu_f32_result_clamp.rs", CODE;
    fp_fma_legacy => "valu_fma_legacy.rs", FMA_LEGACY_CODE, MAC_LEGACY_CODE;
    fp_fma_rounding => "valu_fma_rounding.rs", FMA_CODE;
    fp_modifier_forms => "valu_vop2_modifier_forms.rs", CODE;
    fp_legacy_mods => "valu_vop3_legacy_source_modifiers.rs", CODE;
    fp_mad_clamp => "valu_vop3_mad_clamp.rs", CODE;
    fp_aliases => "valu_vop3_aliases.rs", ALIAS_CODE;
    fp_compare_modifiers => "valu_vop3_compare_modifiers.rs", MODIFIER_CODE;
    fp_nan_quieting => "valu_nan_quieting.rs", CODE;
    fp_division_results => "fmod_division_result_modifiers.rs", CODE;
}

/// v1 = -4, v2 = -8, v3 = tid*32; buffer_load_dwordx2 v[4:5], v1 offen;
/// buffer_load_dwordx4 v[6:9], v2 offen; buffer_load_dwordx3 v[10:12],
/// v2 offen offset:4; stores of them at v3 (+8, +24).
const WRAP_W32: [u32; 17] = [
    0x7e0202c4, 0x7e0402c8, 0x34060085, 0xe0341000, 0x80000401, 0xe0381000, 0x80000602, 0xe03c1004, 0x80000a02, 0xbf8c3f70, 0xe0741000, 0x80010403,
    0xe0781008, 0x80010603, 0xe07c1018, 0x80010a03, 0xbf810000,
];

/// Multi-dword raw loads whose byte offsets wrap at 2^32: the dwords at
/// offsets 0xFFFFFFF8..0xFFFFFFFC are out of the range, the ones that wrap
/// to 0 and 4 are in it and read the buffer's first words (the guest's
/// 32-bit offset arithmetic); with each switch alone and all of them.
#[test]
#[ignore]
fn emit_opt_wrapping_dword_offsets_match_the_interpreter() {
    let _devices = DevicesScope;
    let mut fails = Vec::new();
    let r = Run::new(&WRAP_W32, 32, 32, pattern(64, 0x2468), 32 * 8);
    let out = check_both("wrapping offsets", &r, &mut fails);
    for emit in [Emit::Plus("const_fold"), Emit::Plus("flush_elision"), Emit::Plus("float_class_compare")] {
        let (got, want) = r.device(emit);
        fails.extend(differences("wrapping offsets", emit, &got, &want));
    }
    // Lane 0: (0, input[0]); (0, 0, input[0], input[1]); (0, input[0], input[1]).
    let input = pattern(64, 0x2468);
    assert_eq!(&out[..9], &[0, input[0], 0, 0, input[0], input[1], 0, input[0], input[1]], "lane 0 words");
    report(fails);
}

/// v1 = tid*16, v3 = tid*64; buffer_load_dwordx4 v[4:7], v1 offen;
/// v_max_f32 v10, 0, v4; v_max_f32 v11, 0, v5; v_max_f32 v12, v6, 0
/// (VOP3); v_max_f32 v13, 0, v7; buffer_store_dwordx4 v[10:13], v3 offen
/// (s[4:7]): max with a constant +0 (`RPS5E_ZERO_MAX`) on both sides.
const ZERO_MAX_W32: [u32; 13] = [
    0x34020084, 0x34060086, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x20140880, 0x20160a80, 0xd510000c, 0x00010106, 0x201a0e80, 0xe0781000, 0x80010a03,
    0xbf810000,
];

/// The f32 switches (`RPS5E_FLUSH_ELISION`, `RPS5E_FLOAT_CLASS_COMPARE`,
/// `RPS5E_EMIT_FOLD`; `RPS5E_NAN_ONCE`, `RPS5E_ZERO_MAX` off), each alone and all together, give the device
/// words bit for bit equal to the code without them (NaN payloads and
/// signed zeros included), on special-value rows in every f32 denormal
/// mode, round modes and IEEE_MODE.
#[test]
#[ignore]
fn emit_opt_float_switches_are_exact() {
    let _devices = DevicesScope;
    let mut programs: Vec<(&str, &[u32])> = vec![("chain", &CHAIN_W32[..]), ("zero max", &ZERO_MAX_W32[..])];
    programs.extend(float_programs());
    let mut fails = Vec::new();
    for (name, code) in programs {
        reset_devices();
        for mode in [0xC0u32, 0xF0, 0xD0, 0xE0, 0xC1, 0xC3, 0x8C0, 0x8F0] {
            for seed in 1..=2u64 {
                let mut r = Run::new(code, 32, 32, special_rows(32 * 16, seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)), 32 * 64);
                r.float_mode = mode;
                let tag = format!("{name} mode {mode:#x} seed {seed}");
                // Compared with the code without the switches only: the
                // device's transcendentals differ from the interpreter's by
                // a few ULP, and with f32 denormals kept it flushes them
                // (float_ref checks the interpreter agreement).
                let (base, _) = r.device(Emit::Default);
                for emit in [Emit::Plus("flush_elision"), Emit::Plus("float_class_compare"), Emit::Plus("const_fold"), Emit::New, Emit::Without("nan_once"), Emit::Without("zero_max")] {
                    let (got, _) = r.device(emit);
                    for (i, (a, b)) in got.iter().zip(&base).enumerate() {
                        if a != b {
                            fails.push(format!("{tag} {emit:?} word {i}: {a:#010x}, without the switches {b:#010x} (inputs {:08x?})", &r.input[(i / 32) * 4..(i / 32) * 4 + 4]));
                        }
                    }
                }
            }
        }
    }
    report(fails);
}

/// The f16 fma-mix rounding through the f32 fma with the f64 round-to-odd
/// sum as a lazy select arm (`RPS5E_HALF_FMA_FAST`, `RPS5E_LAZY_SELECT`)
/// gives the device words of the f64 sum alone, bit for bit, at and
/// around f16 midpoints, in the f32 / f16 denormal modes, with FP16_OVFL
/// and IEEE_MODE; with f32 denormals flushed those agree with the
/// interpreter.
#[test]
#[ignore]
fn emit_opt_half_fma_rounding_is_exact() {
    let _devices = DevicesScope;
    let mut fails = Vec::new();
    for mode in [0xC0u32, 0xF0, 0x30, 0x8C0, 0xC4] {
        for seed in 1..=4u64 {
            let mut r = Run::new(&MIX_W32, 32, 32, midpoint_rows(32, seed.wrapping_mul(0xD1B5_4A32_D192_ED03)), 32 * 8);
            r.float_mode = mode;
            let tag = format!("fma mix mode {mode:#x} seed {seed}");
            let (base, want) = r.device(Emit::Without("half_fma_fast"));
            if mode & 0x30 == 0 {
                // With f32 denormals flushed the device agrees with the
                // interpreter (f16 NaNs aside: the device's f64 -> f16
                // conversion returns its canonical NaN); with them kept it
                // flushes some, with or without the fast path.
                let nan16 = |h: u32| h & 0x7FFF > 0x7C00;
                for (i, (&a, &e)) in base.iter().zip(&want).enumerate() {
                    let same_half = |s: u32| (a >> s) & 0xFFFF == (e >> s) & 0xFFFF || (nan16((a >> s) & 0xFFFF) && nan16((e >> s) & 0xFFFF));
                    let nan32 = a & 0x7FFF_FFFF > 0x7F80_0000 && e & 0x7FFF_FFFF > 0x7F80_0000;
                    if a != e && !(same_half(0) && same_half(16)) && !nan32 {
                        fails.push(format!("{tag} f64 sum word {i}: device {a:#010x}, interpreter {e:#010x}"));
                    }
                }
            }
            for emit in [Emit::New, Emit::Without("lazy_select"), Emit::Default] {
                let (got, _) = r.device(emit);
                for (i, (a, b)) in got.iter().zip(&base).enumerate() {
                    if a != b {
                        fails.push(format!("{tag} {emit:?} word {i}: {a:#010x}, f64 sum {b:#010x} (inputs {:08x?})", &r.input[(i / 8) * 4..(i / 8) * 4 + 4]));
                    }
                }
            }
        }
    }
    report(fails);
}
