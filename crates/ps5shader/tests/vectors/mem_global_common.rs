// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

// Shared helpers for the global/flat atomic tests (hw_mem_global_*.rs).
// Pulled in with `include!("vectors/mem_global_common.rs");` after
// `include!("vectors/mem_common.rs");`.

/// Guest address of the 64 KiB block the reference tests point s[8:9] at
/// (64 KiB aligned, like the reference allocation).
#[allow(dead_code)]
pub const GUEST_BASE: u64 = 0x3_0000_0000;
#[allow(dead_code)]
pub const GLOBAL_IN_BASE: u64 = 0x1_0000_0000;
#[allow(dead_code)]
pub const GLOBAL_OUT_BASE: u64 = 0x2_0000_0000;

/// User data of the reference layout: s[0:3] input V#, s[4:7] output V#,
/// s[8:9] the 64-bit guest block address.
#[allow(dead_code)]
pub fn global_user_data(in_bytes: u32, out_bytes: u32) -> Vec<u32> {
    let mut user = vec![0u32; 10];
    put_desc(&mut user, 0, vdesc(GLOBAL_IN_BASE, in_bytes));
    put_desc(&mut user, 4, vdesc(GLOBAL_OUT_BASE, out_bytes));
    user[8] = GUEST_BASE as u32;
    user[9] = (GUEST_BASE >> 32) as u32;
    user
}

/// Run a `threads`-invocation work group as consecutive waves of
/// `wave` lanes (each wave sees the memory left by the previous one and
/// local ids `w * wave ..`). `regions` holds the initial contents of
/// every guest region; returns their final contents.
#[allow(dead_code)]
pub fn run_waves(code: &[u32], wave: u32, threads: u32, user: Vec<u32>, regions: &[(u64, Vec<u32>)]) -> Result<Vec<Vec<u32>>, String> {
    let mut state: Vec<(u64, Vec<u32>)> = regions.to_vec();
    let outputs: Vec<(u64, usize)> = state.iter().map(|(a, w)| (*a, w.len())).collect();
    let mut first = 0;
    while first < threads {
        let lanes = wave.min(threads - first);
        let st = state.clone();
        let setup = move |m: &mut Machine| {
            for (a, w) in &st {
                m.memory.write_words(*a, w);
            }
            m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(0), (first..first + lanes).collect());
        };
        let m = dispatch_machine(code, wave, lanes, user.clone(), &setup, &outputs)?;
        for (a, w) in state.iter_mut() {
            *w = m.memory.read_words(*a, w.len());
        }
        first += lanes;
    }
    Ok(state.into_iter().map(|(_, w)| w).collect())
}

/// Little-endian byte image -> value of `bytes` bytes at `offset`.
#[allow(dead_code)]
pub fn get_le(words: &[u32], offset: usize, bytes: usize) -> u64 {
    let mut v = 0u64;
    for k in 0..bytes {
        let b = (words[(offset + k) / 4] >> (8 * ((offset + k) % 4))) as u8;
        v |= (b as u64) << (8 * k);
    }
    v
}

/// Size of the guest block and its fill byte.
#[allow(dead_code)]
pub const BLOCK_BYTES: usize = 65536;
#[allow(dead_code)]
pub const BLOCK_FILL: u8 = 0xcd;
/// Output prefill of the table-driven tests.
#[allow(dead_code)]
pub const TABLE_SENTINEL: u32 = 0x05e4_71e1;
/// Bytes between the per-operation regions of the guest block.
#[allow(dead_code)]
pub const OP_STRIDE: usize = 0x200;

/// One table-driven dispatch (the reference "one op per 0x200 bytes"
/// layout): lane `tid` of op `op` has its slot at `op * 0x200 + tid * 8`
/// holding `memory` (4 bytes for ops below `narrow_ops`, else 8), and
/// input dwords `op * threads + tid` x4 holding data/comparator.
/// `row(op, tid)` returns `[memory, data, comparator]`.
#[allow(dead_code)]
pub struct TableRun {
    pub initial: Vec<u8>,
    pub output: Vec<u32>,
    pub guest: Vec<u32>,
}

#[allow(dead_code)]
pub fn run_table(code: &[u32], wave: u32, threads: u32, op_count: u32, narrow_ops: u32, out_words: usize, row: &dyn Fn(u32, u32) -> [u64; 3]) -> Result<TableRun, String> {
    let mut initial = vec![BLOCK_FILL; BLOCK_BYTES];
    let mut input = vec![0u32; (op_count * threads * 4) as usize];
    for op in 0..op_count {
        let bytes = if op < narrow_ops { 4 } else { 8 };
        for tid in 0..threads {
            let [memory, data, comparator] = row(op, tid);
            put_le(&mut initial, op as usize * OP_STRIDE + tid as usize * 8, memory, bytes);
            let w = &mut input[((op * threads + tid) * 4) as usize..][..4];
            if bytes == 4 {
                w.copy_from_slice(&[data as u32, comparator as u32, 0, 0]);
            } else {
                w.copy_from_slice(&[data as u32, (data >> 32) as u32, comparator as u32, (comparator >> 32) as u32]);
            }
        }
    }
    let user = global_user_data((input.len() * 4) as u32, (out_words * 4) as u32);
    let regions = vec![(GUEST_BASE, bytes_to_words(&initial)), (GLOBAL_IN_BASE, input), (GLOBAL_OUT_BASE, vec![TABLE_SENTINEL; out_words])];
    let mut r = run_waves(code, wave, threads, user, &regions)?;
    let output = r.remove(2);
    let guest = r.remove(0);
    Ok(TableRun { initial, output, guest })
}

/// Store `bytes` bytes of `value` at byte `offset` of a byte image.
#[allow(dead_code)]
pub fn put_le(image: &mut [u8], offset: usize, value: u64, bytes: usize) {
    for k in 0..bytes {
        image[offset + k] = (value >> (8 * k)) as u8;
    }
}
