// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Scalar compares and waits with immediates, EXEC lane bits of wave32
//! programs, waterfall loops over EXEC, work-groups that wait for each
//! other through global memory, and the relative vector register window
//! (s_movrel / v_movrel through M0): the reference driver's execution tests
//! of the same names, with their expected values computed the same way.

include!("vectors/scalar_common.rs");

include!("vectors/exec_programs.rs");

use exec_programs as ep;

/// Strided (4-byte records) output V# of `count` records.
fn records_desc(base: u64, count: u32) -> [u32; 4] {
    [base as u32, ((base >> 32) as u32 & 0xFFFF) | (4 << 16), count, 0x1101_6FAC]
}

/// One wave32 of 32 lanes: s[4:7] = output V# (64 records per lane),
/// s8.. = `values`; returns the 64 words of each lane.
fn run_sopk(code: &[u32], values: &[u32]) -> Vec<u32> {
    const RESULTS: usize = 64;
    let words = 32 * RESULTS;
    let mut user = vec![0u32; 8 + values.len()];
    user[4..8].copy_from_slice(&records_desc(OUT_BASE, words as u32));
    user[8..].copy_from_slice(values);
    Dispatch::new(code, 32, 32, user).output(OUT_BASE, words).run().unwrap().remove(0)
}

mod sopk_compare {
    use super::*;

    type Eval = fn(u32, u16) -> bool;
    const COMPARES: [(u32, &str, Eval); 12] = [
        (0x03, "s_cmpk_eq_i32", |v, i| v as i32 == i as i16 as i32),
        (0x04, "s_cmpk_lg_i32", |v, i| v as i32 != i as i16 as i32),
        (0x05, "s_cmpk_gt_i32", |v, i| v as i32 > i as i16 as i32),
        (0x06, "s_cmpk_ge_i32", |v, i| v as i32 >= i as i16 as i32),
        (0x07, "s_cmpk_lt_i32", |v, i| (v as i32) < i as i16 as i32),
        (0x08, "s_cmpk_le_i32", |v, i| v as i32 <= i as i16 as i32),
        (0x09, "s_cmpk_eq_u32", |v, i| v == i as u32),
        (0x0A, "s_cmpk_lg_u32", |v, i| v != i as u32),
        (0x0B, "s_cmpk_gt_u32", |v, i| v > i as u32),
        (0x0C, "s_cmpk_ge_u32", |v, i| v >= i as u32),
        (0x0D, "s_cmpk_lt_u32", |v, i| v < i as u32),
        (0x0E, "s_cmpk_le_u32", |v, i| v <= i as u32),
    ];
    #[test]
    fn sopk_compare() {
        let out = run_sopk(&ep::sopk_compare(), &ep::SOPK_VALUES);
        let mut failures = Vec::new();
        for tid in 0..32usize {
            for (v, &value) in ep::SOPK_VALUES.iter().enumerate() {
                for (i, &imm) in ep::SOPK_IMMEDIATES.iter().enumerate() {
                    let mask = out[tid * 64 + v * ep::SOPK_IMMEDIATES.len() + i];
                    let want = COMPARES.iter().enumerate().fold(0u32, |m, (k, c)| m | ((c.2(value, imm) as u32) << k));
                    if mask != want {
                        failures.push(format!("lane {tid} {value:#x}, {imm:#x}: flags {mask:#05x}, expected {want:#05x}"));
                    }
                }
            }
        }
        assert_clean(failures);
    }
}

mod sopk_waitcnt {
    use super::*;

    const WAITS: [(u32, &str); 4] = [(0x17, "s_waitcnt_vscnt"), (0x18, "s_waitcnt_vmcnt"), (0x19, "s_waitcnt_expcnt"), (0x1A, "s_waitcnt_lgkmcnt")];
    #[test]
    fn sopk_waitcnt() {
        let out = run_sopk(&ep::sopk_waitcnt(), &[ep::SOPK_WAIT_VALUE]);
        let mut failures = Vec::new();
        for tid in 0..32usize {
            let mut result = 0;
            for w in WAITS {
                for reg in ep::SOPK_WAIT_REGISTERS {
                    for level in ep::SOPK_WAIT_LEVELS {
                        let (mask, value) = (out[tid * 64 + result], out[tid * 64 + result + 1]);
                        result += 2;
                        if mask != 1 || value != ep::SOPK_WAIT_VALUE {
                            failures.push(format!("lane {tid} {} register {reg:#x} level {level:#x}: flags {mask:#x} (expected 1), s8 {value:#x}", w.1));
                        }
                    }
                }
            }
        }
        assert_clean(failures);
    }
}

/// Wave32 compares, s_and_saveexec and EXEC restores: the bits of the
/// 32-lane masks are the lanes.
#[test]
fn wave32_exec_lane_bits() {
    let user = user_data(raw_desc(OUT_BASE, 128), [0; 4], 8);
    let out = Dispatch::new(&ep::WAVE32_EXEC_LANE_BITS, 32, 32, user).output(OUT_BASE, 32).run().unwrap().remove(0);
    let expected = |lane: u32| {
        let mut v = if lane < 13 { lane + 64 } else { 0 };
        if lane > 20 {
            v += 32;
        }
        if lane > 10 {
            v += 1;
        }
        v
    };
    let failures: Vec<String> = (0..32).filter(|&l| out[l as usize] != expected(l)).map(|l| format!("lane {l}: {}, expected {}", out[l as usize], expected(l))).collect();
    assert_clean(failures);
}

/// A waterfall loop over EXEC copied with s_mov_b64 in a 192-invocation
/// work-group of wave64s: each lane loads its key, the loop scalarizes the
/// keys (s_mulk, s_buffer_load through the key) and counts its passes.
#[test]
fn exec_host_lanes_compute() {
    const THREADS: u32 = 192;
    const KEYS: u32 = ep::EXEC_HOST_KEYS;
    const KEY_STRIDE: u32 = ep::EXEC_HOST_KEY_STRIDE;
    const KEYS_BASE: u64 = 0x3_0000_0000;
    let entry = |key: u32| 0x5a00_0000 + key * 0x0001_0101;
    let keys: Vec<u32> = (0..THREADS).map(|l| (l * 5 + l / 7) % KEYS).collect();
    let mut table = vec![0u32; (KEYS * KEY_STRIDE / 4) as usize];
    for k in 0..KEYS {
        table[(k * KEY_STRIDE / 4) as usize] = entry(k);
    }
    let desc = |base: u64, stride: u32, count: u32| [base as u32, ((base >> 32) as u32 & 0xFFFF) | (stride << 16), count, 0x0101_6FAC];
    // User data starts at s8 in the reference (s0..s7 zero).
    let mut user = vec![0u32; 20];
    user[8..12].copy_from_slice(&desc(KEYS_BASE, 4, THREADS));
    user[12..16].copy_from_slice(&desc(IN_BASE, 0, table.len() as u32 * 4));
    user[16..20].copy_from_slice(&desc(OUT_BASE, 16, THREADS));
    let out = Dispatch::new(&ep::EXEC_HOST_LANES, 64, THREADS, user).buffer(KEYS_BASE, keys.clone()).buffer(IN_BASE, table).output(OUT_BASE, THREADS as usize * 4).run().unwrap().remove(0);
    let mut failures = Vec::new();
    for lane in 0..THREADS as usize {
        let key = keys[lane];
        let inside = lane & 1 == 0;
        let r = &out[lane * 4..lane * 4 + 4];
        if r[2] != key {
            failures.push(format!("lane {lane} read key {}, expected {key}", r[2]));
        }
        if !(1..=KEYS).contains(&r[1]) {
            failures.push(format!("lane {lane} ran {} passes, expected 1 to {KEYS}", r[1]));
        }
        if r[3] != (!inside) as u32 {
            failures.push(format!("lane {lane} left with v5 {}, expected {}", r[3], (!inside) as u32));
        }
        let want = if inside { 0xffff_ffff } else { entry(key) };
        if r[0] != want {
            failures.push(format!("lane {lane} loaded {:#x}, expected {want:#x}", r[0]));
        }
    }
    assert_clean(failures);
}

mod lds_mask_waterfall {
    use super::*;

    const SLOTS: u32 = 16;

    fn run(code: &[u32], exec: u64) -> Vec<u32> {
        let mut user = vec![0u32; 8];
        user[0] = exec as u32;
        user[1] = (exec >> 32) as u32;
        user[4..8].copy_from_slice(&[OUT_BASE as u32, ((OUT_BASE >> 32) as u32 & 0xFFFF) | (16 << 16), 64, 0x1101_6FAC]);
        Dispatch::new(code, 64, 64, user).output(OUT_BASE, 64 * 4).run().unwrap().remove(0)
    }

    #[test]
    fn waterfall_over_lds_masks() {
        let mut failures = Vec::new();
        for exec in ep::LDS_WATERFALL_EXECS {
            let out = run(&ep::LDS_WATERFALL, exec);
            let mut slots = [0u64; 64];
            for lane in 0..64 {
                slots[((lane * 5) & (SLOTS - 1)) as usize] |= 1u64 << lane;
            }
            let (mut iterations, mut index_sum) = (0u32, 0u32);
            for slot in &slots[..SLOTS as usize] {
                let mut mask = *slot;
                while mask != 0 {
                    iterations += 1;
                    index_sum += mask.trailing_zeros();
                    mask &= mask - 1;
                }
            }
            for lane in 0..64usize {
                let active = (exec >> lane) & 1 != 0;
                let want = [if active { iterations } else { 0 }, if active { index_sum } else { 0 }, iterations, slots[lane] as u32];
                for (f, &w) in want.iter().enumerate() {
                    if out[lane * 4 + f] != w {
                        failures.push(format!("exec {exec:#018x} lane {lane} field {f}: {}, expected {w}", out[lane * 4 + f]));
                    }
                }
            }
        }
        assert_clean(failures);
    }

    #[test]
    fn masked_ds_or_b64() {
        let mut failures = Vec::new();
        for exec in ep::LDS_MASKED_OR_EXECS {
            let out = run(&ep::LDS_MASKED_OR, exec);
            let mut slots = [0u64; 64];
            for lane in 0..64 {
                if (exec >> lane) & 1 != 0 {
                    slots[(lane & (SLOTS - 1)) as usize] |= 1u64 << lane;
                }
            }
            for lane in 0..64usize {
                for (w, want) in [slots[lane] as u32, (slots[lane] >> 32) as u32].into_iter().enumerate() {
                    if out[lane * 4 + w] != want {
                        failures.push(format!("exec {exec:#018x} lane {lane} word {w}: {:#x}, expected {want:#x}", out[lane * 4 + w]));
                    }
                }
            }
        }
        assert_clean(failures);
    }
}

/// Two work-groups: the first stores a flag, the second spins on a GLC
/// load until it sees it (2) or gives up after 1e6 passes (3). The
/// interpreter runs the groups in order, so the second always sees the
/// flag; a flag stored before the dispatch is seen either way.
#[test]
fn cross_workgroup_flag() {
    const BLOCK: u64 = 0x3_0000_0000;
    for wave in [64u32, 32] {
        for (preset, groups) in [(1u32, &[1u32][..]), (1, &[0, 1][..]), (0, &[0, 1][..])] {
            let mut memory: Option<Memory> = None;
            for &group in groups {
                for k in 0..64 / wave {
                    let mut run = ComputeRun::new(ep::CROSS_WORKGROUP_FLAG.to_vec(), wave, wave, vec![BLOCK as u32, (BLOCK >> 32) as u32]);
                    run.workgroup_id_sgprs = 1;
                    let carried = memory.clone();
                    let setup = move |m: &mut Machine| {
                        match &carried {
                            Some(mem) => m.memory = mem.clone(),
                            None => m.memory.write_words(BLOCK, &[preset, 0]),
                        }
                        m.system.insert(SystemValue::WorkgroupId(0), vec![group; wave as usize]);
                        m.system.insert(SystemValue::LocalInvocationId(0), (k * wave..(k + 1) * wave).collect());
                    };
                    let m = run.run_checked(&setup, &[(BLOCK, 2)]).unwrap();
                    memory = Some(m.memory.clone());
                }
            }
            let words = memory.unwrap().read_words(BLOCK, 2);
            assert_eq!(words[1], 2, "wave{wave} preset {preset} groups {groups:?}: flag {}, result {}", words[0], words[1]);
        }
    }
}

mod vector_window {
    use super::*;

    const NAMES: [&str; 14] = [
        "valu_direct", "rewritten_direct", "writelane_direct", "loaded_direct", "readlane_sum_direct", "movrels_valu", "movrels_valu_neighbour",
        "movrels_writelane", "movrels_loaded", "movrels_readlane_sum", "movrels_rewritten", "movrels_of_movrels_result", "readlane_written_lane",
        "readlane_other_lane",
    ];

    /// Direct and v_movrels reads of registers written by VALU ops,
    /// v_writelane, loads and v_readlane sums.
    #[test]
    fn direct_access() {
        let first = |t: u32| 0x1000 * (t + 1);
        let second = |t: u32| 0x0077_0000 + t * 0x101;
        const LANE_VALUE: u32 = 0x5a5a_0000;
        let mut failures = Vec::new();
        for wave in [32u32, 64] {
            let mut input = vec![0u32; 64 * 4];
            for t in 0..wave {
                input[t as usize * 4] = first(t);
                input[t as usize * 4 + 1] = second(t);
            }
            let user = user_data(raw_desc(IN_BASE, 64 * 16), raw_desc(OUT_BASE, 64 * 64), 8);
            let out = Dispatch::new(&ep::VECTOR_WINDOW_DIRECT, wave, wave, user).buffer(IN_BASE, input).output(OUT_BASE, 64 * 16).run().unwrap().remove(0);
            for t in 0..wave {
                let (x, y) = (first(t), second(t));
                let lane = if t == 3 { LANE_VALUE } else { x };
                let want = [x + 1, x << 4, lane, y, x + LANE_VALUE, x + 1, x + 2, lane, y, x + LANE_VALUE, x << 4, x + 1, LANE_VALUE, first(5)];
                for (i, &w) in want.iter().enumerate() {
                    let got = out[t as usize * 16 + i];
                    if got != w {
                        failures.push(format!("wave{wave} lane {t} {}: {got:#010x}, expected {w:#010x}", NAMES[i]));
                    }
                }
            }
        }
        assert_clean(failures);
    }


    /// A 32-register window v[10:41] read and written through M0 = s8 & 3
    /// and M0 = s9 (a swap of two slots).
    #[test]
    fn narrowing() {
        const BOUNDED: u32 = 7;
        let mut failures = Vec::new();
        for data in [20u32, 2, 31] {
            for wave in [32u32, 64] {
                let mut user = user_data([0; 4], raw_desc(OUT_BASE, 64 * 64), 10);
                user[8] = BOUNDED;
                user[9] = data;
                let out = Dispatch::new(&ep::VECTOR_WINDOW_NARROWING, wave, wave, user).output(OUT_BASE, 64 * 16).run().unwrap().remove(0);
                for t in 0..wave {
                    let mut window: Vec<u32> = (0..32).map(|k| (t << 8) + k).collect();
                    let v4 = window[(BOUNDED & 3) as usize];
                    let v5 = window[data as usize];
                    window[data as usize] = v4;
                    window[(BOUNDED & 3) as usize] = v5;
                    let want = [v4, v5, window[2], window[3], window[20], window[31]];
                    for (i, (&w, name)) in want.iter().zip(["v4", "v5", "v12", "v13", "v30", "v41"]).enumerate() {
                        let got = out[t as usize * 16 + i];
                        if got != w {
                            failures.push(format!("wave{wave} data {data} lane {t} {name}: {got:#010x}, expected {w:#010x}"));
                        }
                    }
                }
            }
        }
        assert_clean(failures);
    }
}
