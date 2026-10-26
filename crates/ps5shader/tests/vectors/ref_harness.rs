// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Shared runner for the reference compute cases (`tests/ref_*.rs`).
//!
//! Each case is a compute program plus one storage buffer: `initial` words
//! are loaded into it before the dispatch and the first `expected.len()`
//! words are compared afterwards. The reference harness binds every buffer
//! descriptor the program uses to that one buffer, so it is mapped here at
//! guest address 0 (the default descriptors all have base 0):
//!
//! * no explicit user data: s[0:3] is a raw V# (base 0, NUM_RECORDS =
//!   `initial` bytes, OOB_SELECT 3) and s[48:51] is a raw V# (base 0,
//!   NUM_RECORDS 1 MiB, OOB_SELECT 3); the helper stores of the cases
//!   (`buffer_store_dword ... s[48:51]`) write through the latter;
//! * explicit user data: the 64 words are used as given; if s[48:51] is all
//!   zero (the case did not describe the store buffer) it receives the
//!   default store V# so the helper stores still reach the result buffer,
//!   and a stride-0 s[48:51] with a zero dword3 (only its size given)
//!   becomes raw (OOB_SELECT 3);
//! * all 64 user-data words are preloaded into s0..s63; work-group ids and
//!   the TG_SIZE word overwrite the SGPRs from `workgroup_register` on.
//!
//! Work-groups are run one after another and their waves one after another
//! (the interpreter runs one wave at a time), sharing memory; LDS is
//! per work-group and sized by `lds_dwords`. Every wave has all of its
//! lanes; lanes past the work-group's invocations (and, with
//! `dispatch_dims`, outside the dispatch extents) start with EXEC clear.
//! Flat/global mappings (`bda`) mirror the buffer at their guest address.
//! FLOAT_MODE (`float_mode`) is the case's RSRC1 FLOAT_MODE; the reference
//! cases run with DX10_CLAMP set and IEEE_MODE clear (what compiled
//! shaders use).

#![allow(dead_code)]

use ps5shader::interp::{GroupMemory, Machine, WaveGroup};
use std::sync::Arc;
use ps5shader::program::decode_blob;
use ps5shader::ssa::{Function, Stage, SystemValue};
use ps5shader::translate::{compile, FloatMode, InitialExec, InputValue, TranslateOptions};

/// Where the code object is mapped (away from the result buffer at 0).
pub const CODE_ADDRESS: u64 = 0x10_0000_0000;

/// Where the result buffer's bytes live; the guest addresses the program
/// uses (V# windows, flat mappings) are aliases of it.
pub const HOME: u64 = 0x7F00_0000_0000;

/// Compute-dispatch parameters of a case.
#[derive(Clone, Copy, Debug)]
pub struct Compute {
    /// Work-group size (0 means 1).
    pub threads: [u32; 3],
    pub lds_dwords: u32,
    pub wave_size: u32,
    pub float_mode: u32,
    /// Thread extents of the dispatch (with `dispatch_dims`).
    pub dispatch_threads: [u32; 3],
    pub workgroup_register: u32,
    /// Number of thread-id VGPRs (v0..).
    pub thread_ids: u32,
    pub group_id: [bool; 3],
    pub dispatch_dims: bool,
    pub tg_size_en: bool,
}

/// Extra memory of a run (see `RefCase::execute_env`).
#[derive(Default)]
pub struct Env<'e> {
    /// Guest addresses at which the result buffer is mapped (default: 0).
    pub windows: Option<&'e [u64]>,
    /// GDS contents, in and out.
    pub gds: Vec<u8>,
    /// Base address given to the default result-buffer V#s (instead of 0).
    pub vsharp_base: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub struct RefCase {
    pub name: &'static str,
    pub code: &'static [u32],
    pub initial: &'static [u32],
    pub expected: &'static [u32],
    /// 64 user-data words, when the case supplies them.
    pub user_data: Option<&'static [u32]>,
    pub compute: Compute,
    /// Work-group counts.
    pub dispatch: [u32; 3],
    pub compile_only: bool,
    pub tolerance: f32,
    /// Flat/global address mappings: guest address -> byte offset in the
    /// buffer. The buffer contents from that offset are mirrored at the
    /// guest address before the dispatch and bytes changed there are copied
    /// back afterwards.
    pub bda: &'static [(u64, u32)],
}

impl RefCase {
    pub fn user_data(&self) -> Vec<u32> {
        let mut ud = vec![0u32; 64];
        match self.user_data {
            Some(words) => {
                ud.copy_from_slice(words);
                if ud[48..52].iter().all(|&w| w == 0) {
                    ud[50] = 1 << 20;
                    ud[51] = 3 << 28;
                } else if ud[49] >> 16 & 0x3FFF == 0 && ud[51] == 0 {
                    // A case that only sizes the helper-store V# (the
                    // reference harness's result binding) gets a raw one.
                    ud[51] = 3 << 28;
                }
            }
            None => {
                ud[2] = (self.initial.len() * 4) as u32;
                ud[3] = 3 << 28;
                ud[50] = 1 << 20;
                ud[51] = 3 << 28;
            }
        }
        ud
    }

    fn threads(&self) -> [u32; 3] {
        let t = self.compute.threads;
        [t[0].max(1), t[1].max(1), t[2].max(1)]
    }

    fn total_threads(&self) -> u32 {
        let t = self.threads();
        t[0] * t[1] * t[2]
    }

    pub fn options(&self, optimize: bool) -> TranslateOptions {
        let c = &self.compute;
        let mut o = TranslateOptions::with_user_data(c.wave_size, Stage::Compute, 64);
        let mut set = |reg: u32, v: InputValue| {
            if let Some(e) = o.sgprs.iter_mut().find(|e| e.0 as u32 == reg) {
                e.1 = v;
            } else {
                o.sgprs.push((reg as u8, v));
            }
        };
        let mut reg = c.workgroup_register;
        for axis in 0..3u8 {
            if c.group_id[axis as usize] {
                set(reg, InputValue::System(SystemValue::WorkgroupId(axis)));
                reg += 1;
            }
        }
        if c.tg_size_en {
            set(reg, InputValue::System(SystemValue::Custom(0)));
        }
        o.vgprs = (0..c.thread_ids.min(3) as u16).map(|a| (a, InputValue::System(SystemValue::LocalInvocationId(a as u8)))).collect();
        // Lanes past the work-group's invocations and outside the dispatch
        // extents start inactive.
        o.initial_exec = InitialExec::NotSystem(SystemValue::HelperInvocation);
        o.optimize = optimize;
        o.float_mode = FloatMode { mode: c.float_mode as u8, dx10_clamp: true, ieee: false, fp16_overflow: false };
        o.strict_mode_writes = true;
        o
    }

    pub fn compile(&self, optimize: bool) -> Result<Function, String> {
        let bytes: Vec<u8> = self.code.iter().flat_map(|w| w.to_le_bytes()).collect();
        let program = decode_blob(&bytes, CODE_ADDRESS).map_err(|e| format!("decode: {e}"))?;
        let code = self.code;
        let reader = move |a: u64| -> Option<u32> {
            let off = a.checked_sub(CODE_ADDRESS)?;
            if off % 4 != 0 {
                return None;
            }
            code.get((off / 4) as usize).copied()
        };
        let c = compile(&program, &self.options(optimize), Some(&reader)).map_err(|e| format!("compile: {e}"))?;
        Ok(c.function)
    }

    /// Run every work-group; returns the first `words` buffer words.
    pub fn execute(&self, optimize: bool, words: usize) -> Result<Vec<u32>, String> {
        self.execute_env(optimize, words, &mut Env::default())
    }

    /// `execute` with extra memory: guest windows of the result buffer
    /// (replacing the mapping at address 0) and GDS.
    pub fn execute_env(&self, optimize: bool, words: usize, env: &mut Env) -> Result<Vec<u32>, String> {
        let f = self.compile(optimize)?;
        self.execute_function(&f, words, env)
    }

    /// The optimized function with every run of two or more EXEC-masked
    /// writes grouped under one branch (the back-end's last pass).
    pub fn compile_grouped(&self) -> Result<Function, String> {
        let mut f = self.compile(true)?;
        ps5shader::opt::exec_runs::group_exec_runs(&mut f, 2);
        ps5shader::ssa::verify::verify(&f, true).map_err(|e| format!("grouped: {e}\n{f}"))?;
        Ok(f)
    }

    /// Run every work-group of `f`; returns the first `words` buffer words.
    pub fn execute_function(&self, f: &Function, words: usize, env: &mut Env) -> Result<Vec<u32>, String> {
        let c = &self.compute;
        let wave = c.wave_size;
        let t = self.threads();
        let total = self.total_threads();
        let mut ud = self.user_data();
        if let Some(base) = env.vsharp_base {
            // Move the result buffer's V#s (base 0) to `base`.
            for q in [0, 48] {
                if ud[q] == 0 && ud[q + 1] & 0xFFFF == 0 {
                    ud[q] = base as u32;
                    ud[q + 1] |= (base >> 32) as u32 & 0xFFFF;
                }
            }
        }
        let mut memory = ps5shader::interp::Memory::default();
        memory.write_words(CODE_ADDRESS, self.code);
        let mut bytes: Vec<u8> = self.initial.iter().flat_map(|w| w.to_le_bytes()).collect();
        bytes.resize(bytes.len().max(self.expected.len() * 4), 0);
        // The buffer lives at HOME; every guest view of it is an alias.
        let mut maps: Vec<(u64, u32)> = match env.windows {
            Some(w) => w.iter().map(|&g| (g, 0)).collect(),
            None => vec![(env.vsharp_base.unwrap_or(0), 0)],
        };
        maps.extend_from_slice(self.bda);
        for (k, b) in bytes.iter().enumerate() {
            memory.write_u8(HOME + k as u64, *b);
        }
        for &(guest, backing) in &maps {
            memory.alias(guest, bytes.len() as u64 - (backing as u64).min(bytes.len() as u64), HOME + backing as u64);
        }
        let waves = total.div_ceil(wave);
        for gz in 0..self.dispatch[2].max(1) {
            for gy in 0..self.dispatch[1].max(1) {
                for gx in 0..self.dispatch[0].max(1) {
                    let group = [gx, gy, gz];
                    let lds = vec![0u8; (c.lds_dwords * 4) as usize];
                    let shared = GroupMemory { memory: std::mem::take(&mut memory), lds, gds: std::mem::take(&mut env.gds) };
                    // The waves run on their own threads and hand the
                    // shared memory over at barriers.
                    let wg = Arc::new(WaveGroup::new(waves, shared));
                    let results: Vec<Result<(), String>> = std::thread::scope(|scope| {
                        let handles: Vec<_> = (0..waves)
                            .map(|w| {
                                let wg = wg.clone();
                                let ud = &ud;
                                scope.spawn(move || {
                                    let first = w * wave;
                                    // Every wave has all its lanes; the ones past the
                                    // work-group's invocations start inactive and
                                    // never access memory.
                                    let lanes = wave;
                                    let mut m = Machine::new(wave, lanes);
                                    m.present = Some(wave.min(total - first));
                                    m.user_data = ud.clone();
                                    m.group = Some(wg);
                                    let idx: Vec<u32> = (first..first + lanes).collect();
                                    let local = |axis: usize| -> Vec<u32> {
                                        idx.iter()
                                            .map(|&i| match axis {
                                                0 => i % t[0],
                                                1 => (i / t[0]) % t[1],
                                                _ => i / (t[0] * t[1]),
                                            })
                                            .collect()
                                    };
                                    let mut outside: Vec<u32> = idx.iter().map(|&i| (i >= total) as u32).collect();
                                    for axis in 0..3 {
                                        let l = local(axis);
                                        let global: Vec<u32> = l.iter().map(|&x| group[axis] * t[axis] + x).collect();
                                        if c.dispatch_dims {
                                            for (k, g) in global.iter().enumerate() {
                                                if *g >= c.dispatch_threads[axis] {
                                                    outside[k] = 1;
                                                }
                                            }
                                        }
                                        m.system.insert(SystemValue::LocalInvocationId(axis as u8), l);
                                        m.system.insert(SystemValue::GlobalInvocationId(axis as u8), global);
                                        m.system.insert(SystemValue::WorkgroupId(axis as u8), vec![group[axis]; lanes as usize]);
                                    }
                                    m.system.insert(SystemValue::HelperInvocation, outside);
                                    m.system.insert(SystemValue::LocalInvocationIndex, idx.clone());
                                    m.system.insert(SystemValue::SubgroupId, vec![w; lanes as usize]);
                                    let tg = waves.min(0x3F) | (w << 20) | if w == 0 { 0x8000_0000 } else { 0 };
                                    m.system.insert(SystemValue::Custom(0), vec![tg; lanes as usize]);
                                    m.run(f).map_err(|e| format!("run (group {group:?} wave {w}): {e}\n{f}"))
                                })
                            })
                            .collect();
                        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("wave panicked".into()))).collect()
                    });
                    for r in results {
                        r?;
                    }
                    let shared = Arc::try_unwrap(wg).ok().expect("waves ended").into_memory();
                    memory = shared.memory;
                    env.gds = shared.gds;
                }
            }
        }
        let mut out: Vec<u8> = (0..bytes.len() as u64).map(|k| memory.read_u8(HOME + k)).collect();
        out.resize(out.len().max(words * 4), 0);
        Ok(out.chunks(4).take(words).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect())
    }

    fn matches(&self, actual: &[u32]) -> bool {
        if actual == self.expected {
            return true;
        }
        self.tolerance > 0.0
            && actual.len() == self.expected.len()
            && actual.iter().zip(self.expected).all(|(a, e)| (f32::from_bits(*a) - f32::from_bits(*e)).abs() <= self.tolerance)
    }

    /// Compile (and unless compile-only, run) unoptimized and optimized;
    /// both results must match `expected`.
    pub fn check(&self) -> Result<(), String> {
        let mut errors = Vec::new();
        for (mode, optimize, grouped) in [("optimize=false", false, false), ("optimize=true", true, false), ("exec runs", true, true)] {
            if self.compile_only {
                let r = if grouped { self.compile_grouped().map(|_| ()) } else { self.compile(optimize).map(|_| ()) };
                if let Err(e) = r {
                    errors.push(format!("[{mode}] {e}"));
                }
                continue;
            }
            let result = if grouped {
                self.compile_grouped().and_then(|f| self.execute_function(&f, self.expected.len(), &mut Env::default()))
            } else {
                self.execute(optimize, self.expected.len())
            };
            match result {
                Err(e) => errors.push(format!("[{mode}] {e}")),
                Ok(actual) if !self.matches(&actual) => {
                    let diff: Vec<String> = actual
                        .iter()
                        .zip(self.expected)
                        .enumerate()
                        .filter(|(_, (a, e))| a != e)
                        .map(|(i, (a, e))| format!("  word {i}: expected {e:#010x} actual {a:#010x}"))
                        .collect();
                    errors.push(format!("[{mode}] {} of {} words differ:\n{}", diff.len(), actual.len(), diff.join("\n")));
                }
                Ok(_) => {}
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(format!("{}: {}", self.name, errors.join("\n")))
        }
    }
}

pub fn run(case: &RefCase) {
    if let Err(e) = case.check() {
        panic!("{e}");
    }
}
