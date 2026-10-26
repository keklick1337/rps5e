// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

// Shared helpers for the hardware-measured memory tests (hw_mem_*.rs).
// Pulled in with `include!("vectors/mem_common.rs");` at the crate root of a
// test file.

#[allow(unused_imports)]
use ps5shader::interp::Machine;
#[allow(unused_imports)]
use ps5shader::testing::ComputeRun;

/// Buffer descriptor used by the reference tests: raw (stride 0), `bytes`
/// records, dword3 0x31016fac (OOB_SELECT 3, raw).
#[allow(dead_code)]
pub fn vdesc(base: u64, bytes: u32) -> [u32; 4] {
    [base as u32, ((base >> 32) as u32) & 0xffff, bytes, 0x3101_6fac]
}

/// Buffer descriptor with an explicit dword1 stride and dword3.
#[allow(dead_code)]
pub fn vdesc_full(base: u64, stride: u32, records: u32, dword3: u32) -> [u32; 4] {
    [base as u32, (((base >> 32) as u32) & 0xffff) | ((stride & 0x3fff) << 16), records, dword3]
}

/// Copy `desc` into `user[at..at + 4]` (growing `user` as needed).
#[allow(dead_code)]
pub fn put_desc(user: &mut Vec<u32>, at: usize, desc: [u32; 4]) {
    if user.len() < at + 4 {
        user.resize(at + 4, 0);
    }
    user[at..at + 4].copy_from_slice(&desc);
}

/// Bytes -> little-endian words (zero padded).
#[allow(dead_code)]
pub fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks(4).map(|c| {
        let mut b = [0u8; 4];
        b[..c.len()].copy_from_slice(c);
        u32::from_le_bytes(b)
    }).collect()
}

/// Write raw bytes into guest memory.
#[allow(dead_code)]
pub fn write_bytes(m: &mut Machine, addr: u64, bytes: &[u8]) {
    for (k, b) in bytes.iter().enumerate() {
        m.memory.write_u8(addr + k as u64, *b);
    }
}

/// Compile + run one compute wave (optimized and unoptimized runs must
/// agree on `outputs`). Returns the words of each `(address, dwords)`
/// output region.
#[allow(dead_code)]
pub fn dispatch(code: &[u32], wave: u32, threads: u32, user: Vec<u32>, setup: &dyn Fn(&mut Machine), outputs: &[(u64, usize)]) -> Result<Vec<Vec<u32>>, String> {
    let run = ComputeRun::new(code.to_vec(), wave, threads, user);
    let m = run.run_checked(setup, outputs)?;
    Ok(outputs.iter().map(|&(a, n)| m.memory.read_words(a, n)).collect())
}

/// Like `dispatch` but also returns the final machine (LDS etc.).
#[allow(dead_code)]
pub fn dispatch_machine(code: &[u32], wave: u32, threads: u32, user: Vec<u32>, setup: &dyn Fn(&mut Machine), outputs: &[(u64, usize)]) -> Result<Machine<'static>, String> {
    let run = ComputeRun::new(code.to_vec(), wave, threads, user);
    run.run_checked(setup, outputs)
}

/// Collects mismatches and asserts at the end.
#[allow(dead_code)]
#[derive(Default)]
pub struct Checker {
    pub failures: Vec<String>,
    pub checks: usize,
}

#[allow(dead_code)]
impl Checker {
    pub fn new() -> Self {
        Checker::default()
    }

    pub fn eq(&mut self, what: impl FnOnce() -> String, actual: u32, expected: u32) {
        self.checks += 1;
        if actual != expected {
            self.failures.push(format!("{} is {:#010x}, expected {:#010x}", what(), actual, expected));
        }
    }

    pub fn finish(self) {
        let n = self.failures.len();
        let shown: Vec<_> = self.failures.iter().take(64).cloned().collect();
        assert!(n == 0, "{n} of {} checks failed:\n{}", self.checks, shown.join("\n"));
    }
}
