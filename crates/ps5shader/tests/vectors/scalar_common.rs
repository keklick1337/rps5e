// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

// Shared harness for the hardware-measured scalar / wave test ports: a
// compute dispatch of one workgroup (split into waves that run one after the
// other on shared memory), buffer descriptors and row checking.

#[allow(unused_imports)]
use ps5shader::interp::{Machine, Memory};
#[allow(unused_imports)]
use ps5shader::ssa::SystemValue;
#[allow(unused_imports)]
use ps5shader::testing::ComputeRun;

#[allow(dead_code)]
pub const IN_BASE: u64 = 0x1_0000_0000;
#[allow(dead_code)]
pub const OUT_BASE: u64 = 0x2_0000_0000;
#[allow(dead_code)]
pub const UNTOUCHED: u32 = 0xdead_beef;

/// Raw (stride 0) buffer descriptor of `bytes` bytes.
#[allow(dead_code)]
pub fn raw_desc(base: u64, bytes: u32) -> [u32; 4] {
    [base as u32, (base >> 32) as u32 & 0xFFFF, bytes, 0x3101_6FAC]
}

/// Structured buffer descriptor (stride 4) of `count` records.
#[allow(dead_code)]
pub fn strided_desc(base: u64, count: u32) -> [u32; 4] {
    [base as u32, ((base >> 32) as u32 & 0xFFFF) | (4 << 16), count, 0x1101_6FAC]
}

/// User data: `[s0..s3] = a`, `[s4..s7] = b`, padded to `len` words.
#[allow(dead_code)]
pub fn user_data(a: [u32; 4], b: [u32; 4], len: usize) -> Vec<u32> {
    let mut u = vec![0u32; len.max(8)];
    u[..4].copy_from_slice(&a);
    u[4..8].copy_from_slice(&b);
    u
}

/// One compute workgroup.
#[allow(dead_code)]
#[derive(Clone)]
pub struct Dispatch {
    pub code: Vec<u32>,
    pub wave: u32,
    /// Workgroup dimensions (threads = product).
    pub local: [u32; 3],
    pub user_data: Vec<u32>,
    /// Initial memory contents.
    pub buffers: Vec<(u64, Vec<u32>)>,
    /// Ranges whose contents are returned (and compared between the
    /// optimized and unoptimized pipelines).
    pub outputs: Vec<(u64, usize)>,
    pub thread_id_vgprs: u32,
}

#[allow(dead_code)]
impl Dispatch {
    pub fn new(code: &[u32], wave: u32, threads: u32, user_data: Vec<u32>) -> Self {
        Dispatch { code: code.to_vec(), wave, local: [threads, 1, 1], user_data, buffers: Vec::new(), outputs: Vec::new(), thread_id_vgprs: 1 }
    }

    pub fn buffer(mut self, base: u64, words: Vec<u32>) -> Self {
        self.buffers.push((base, words));
        self
    }

    /// An output buffer prefilled with 0xdeadbeef.
    pub fn output(mut self, base: u64, words: usize) -> Self {
        self.buffers.push((base, vec![UNTOUCHED; words]));
        self.outputs.push((base, words));
        self
    }

    pub fn compile(&self) -> Result<(), String> {
        ComputeRun::new(self.code.clone(), self.wave, self.wave, self.user_data.clone()).compile(true).map(|_| ())
    }

    /// Run every wave of the workgroup in order; returns the output ranges.
    pub fn run(&self) -> Result<Vec<Vec<u32>>, String> {
        let threads = self.local[0] * self.local[1] * self.local[2];
        let waves = threads.div_ceil(self.wave);
        let mut memory: Option<Memory> = None;
        for k in 0..waves {
            let lanes = (threads - k * self.wave).min(self.wave);
            let mut run = ComputeRun::new(self.code.clone(), self.wave, lanes, self.user_data.clone());
            run.thread_id_vgprs = self.thread_id_vgprs;
            let carried = memory.clone();
            let buffers = &self.buffers;
            let local = self.local;
            let wave = self.wave;
            let setup = move |m: &mut Machine| {
                match &carried {
                    Some(mem) => m.memory = mem.clone(),
                    None => {
                        for (base, words) in buffers {
                            m.memory.write_words(*base, words);
                        }
                    }
                }
                let ids: Vec<u32> = (0..lanes).map(|l| k * wave + l).collect();
                m.system.insert(SystemValue::LocalInvocationId(0), ids.iter().map(|i| i % local[0]).collect());
                m.system.insert(SystemValue::LocalInvocationId(1), ids.iter().map(|i| (i / local[0]) % local[1]).collect());
                m.system.insert(SystemValue::LocalInvocationId(2), ids.iter().map(|i| i / (local[0] * local[1])).collect());
                m.system.insert(SystemValue::LocalInvocationIndex, ids.clone());
                m.system.insert(SystemValue::SubgroupId, vec![k; lanes as usize]);
            };
            let m = run.run_checked(&setup, &self.outputs).map_err(|e| format!("wave {k}: {e}"))?;
            memory = Some(m.memory.clone());
        }
        let mem = memory.unwrap();
        Ok(self.outputs.iter().map(|&(a, n)| mem.read_words(a, n)).collect())
    }
}

/// Compare `out[lane * stride + i]` against `expected[lane][i]` for the
/// named columns (`None` = unchecked); `keep` filters columns.
#[allow(dead_code)]
pub fn check_rows<const K: usize>(label: &str, out: &[u32], stride: usize, expected: &[[u32; K]], names: &[Option<&str>], keep: &dyn Fn(usize) -> bool) -> Vec<String> {
    let mut failures = Vec::new();
    for (lane, row) in expected.iter().enumerate() {
        for (i, &want) in row.iter().enumerate() {
            let Some(name) = names.get(i).copied().flatten() else { continue };
            if !keep(i) {
                continue;
            }
            let got = out[lane * stride + i];
            if got != want {
                failures.push(format!("{label}: lane {lane} {name} is {got:#010x}, expected {want:#010x}"));
            }
        }
    }
    failures
}

/// Replace every occurrence of the single-dword instructions `words` by
/// s_nop, so checks that do not depend on them stay runnable while the
/// compiler lacks them.
#[allow(dead_code)]
pub fn without(code: &[u32], words: &[u32]) -> Vec<u32> {
    let mut found = 0;
    let out = code
        .iter()
        .map(|&w| {
            if words.contains(&w) {
                found += 1;
                0xBF80_0000
            } else {
                w
            }
        })
        .collect();
    assert!(found >= words.len(), "patched instruction not found");
    out
}

/// Replace every two-dword SMEM instruction whose opcode is in `ops` by two
/// s_nop.
#[allow(dead_code)]
pub fn without_smem(code: &[u32], ops: &[u32]) -> Vec<u32> {
    let mut out = code.to_vec();
    let mut i = 0;
    let mut found = 0;
    while i + 1 < out.len() {
        let w = out[i];
        if w >> 26 == 0x3D && ops.contains(&((w >> 18) & 0xFF)) {
            out[i] = 0xBF80_0000;
            out[i + 1] = 0xBF80_0000;
            found += 1;
            i += 2;
        } else {
            i += 1;
        }
    }
    assert!(found > 0, "patched SMEM instruction not found");
    out
}

#[allow(dead_code)]
pub fn assert_clean(failures: Vec<String>) {
    if !failures.is_empty() {
        let shown: Vec<&String> = failures.iter().take(400).collect();
        panic!("{} failures:\n{}", failures.len(), shown.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n"));
    }
}
