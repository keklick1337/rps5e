// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Scalar ALU (SOP1 / SOP2 / SOPK / bit counts / s_getreg) against
//! hardware-measured reference rows.

include!("vectors/scalar_common.rs");

const THREADS: u32 = 32;

/// The common layout: s[0:3] = raw input V#, s[4:7] = raw output V#, one
/// row of `inputs` words in and `results` words out per lane.
fn run_rows(code: &[u32], rows: &[[u32; 4]], results: usize) -> Vec<u32> {
    let input: Vec<u32> = rows.iter().flatten().copied().collect();
    let in_bytes = (THREADS as usize * 4 * 4) as u32;
    let mut input_words = input;
    input_words.resize(THREADS as usize * 4, 0);
    let out_words = THREADS as usize * results;
    let ud = user_data(raw_desc(IN_BASE, in_bytes), raw_desc(OUT_BASE, (out_words * 4) as u32), 8);
    let d = Dispatch::new(code, 32, THREADS, ud).buffer(IN_BASE, input_words).output(OUT_BASE, out_words);
    d.run().unwrap().remove(0)
}

mod sop1 {
    use super::*;
    include!("vectors/scalar_sop1.rs");

    fn columns(keep: &dyn Fn(usize) -> bool) {
        let out = run_rows(&CODE, &ROWS, 176);
        assert_clean(check_rows("scalar sop1", &out, 176, &EXPECTED, &NAMES, keep));
    }

    fn name(i: usize) -> &'static str {
        NAMES[i].unwrap_or("")
    }

    /// Everything except the columns split out below.
    #[test]
    fn sop1_rows() {
        columns(&|i| !is_split(name(i)));
    }

    fn is_split(n: &str) -> bool {
        n.starts_with("movrel") || n.contains("wrexec") || n.contains("andn1_") || n.contains("orn1_") || n.starts_with("scc")
    }

    #[test]
    fn sop1_movrel() {
        columns(&|i| name(i).starts_with("movrel"));
    }

    #[test]
    fn sop1_n1_saveexec_wrexec() {
        columns(&|i| !name(i).starts_with("scc") && (name(i).contains("wrexec") || name(i).contains("andn1_") || name(i).contains("orn1_")));
    }

    #[test]
    fn sop1_scc() {
        columns(&|i| name(i).starts_with("scc"));
    }
}

mod sop2 {
    use super::*;
    include!("vectors/scalar_sop2.rs");

    #[test]
    fn sop2_all_columns() {
        let out = run_rows(&CODE, &ROWS, 64);
        assert_clean(check_rows("scalar sop2", &out, 64, &EXPECTED, &NAMES, &|_| true));
    }
}

mod bit_count {
    use super::*;
    include!("vectors/scalar_bit_count.rs");

    const INPUTS: usize = 4;
    const RESULTS: usize = 8;

    fn input() -> Vec<u32> {
        let mut input = vec![0u32; THREADS as usize * INPUTS];
        let mut state: u64 = 0x9e3779b97f4a7c15;
        for tid in 0..THREADS as usize {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let value = if tid < EDGES.len() { EDGES[tid] } else { state >> (tid as u32 % 13) };
            input[tid * INPUTS] = value as u32;
            input[tid * INPUTS + 1] = (value >> 32) as u32;
        }
        input
    }

    fn signed_leading_bits(value: u64, sign: u64, width: u32) -> u32 {
        let folded = if value & sign != 0 { !value & (sign | (sign - 1)) } else { value };
        if folded == 0 {
            0xffff_ffff
        } else {
            folded.leading_zeros() - (64 - width)
        }
    }

    /// s_bcnt0_i32_b64, s_ff0_i32_b64, s_flbit_i32, s_flbit_i32_i64,
    /// s_quadmask_b32 and their SCC (strided input / output buffers).
    #[test]
    fn bit_count() {
        let input = input();
        let ud = user_data(strided_desc(IN_BASE, input.len() as u32), strided_desc(OUT_BASE, THREADS * RESULTS as u32), 8);
        let d = Dispatch::new(&BIT_COUNT_CODE, 32, THREADS, ud).buffer(IN_BASE, input.clone()).output(OUT_BASE, THREADS as usize * RESULTS);
        let out = d.run().unwrap().remove(0);
        let mut failures = Vec::new();
        for tid in 0..THREADS as usize {
            let value = input[tid * INPUTS] as u64 | ((input[tid * INPUTS + 1] as u64) << 32);
            let low = value as u32;
            let mut quads = 0u32;
            for quad in 0..8 {
                if (low >> (quad * 4)) & 0xF != 0 {
                    quads |= 1 << quad;
                }
            }
            let zeros = 64 - value.count_ones();
            let expected = [
                zeros,
                if !value == 0 { 0xffff_ffff } else { value.trailing_ones() },
                signed_leading_bits(low as u64, 0x8000_0000, 32),
                signed_leading_bits(value, 0x8000_0000_0000_0000, 64),
                quads,
                (zeros != 0) as u32,
                (quads != 0) as u32,
            ];
            for (j, &want) in expected.iter().enumerate() {
                let got = out[tid * RESULTS + j];
                if got != want {
                    failures.push(format!("scalar bit count: lane {tid} ({value:#018x}) result {j} is {got:#x}, expected {want:#x}"));
                }
            }
        }
        assert_clean(failures);
    }
}

mod sopk_compare {
    use super::*;

    const RESULTS: usize = 64;
    const OUTPUT_REGISTER: u32 = 4;
    const VALUE_REGISTER: u32 = 8;
    const MASK_REGISTER: u32 = 30;
    const BIT_REGISTER: u32 = 31;
    const INDEX_VECTOR: u32 = 3;
    const DATA_VECTOR: u32 = 10;
    const ZERO: u32 = 0x80;
    const SIX: u32 = 0x86;
    const LITERAL: u32 = 0xff;

    const VALUES: [u32; 10] = [0x00000000, 0x00000001, 0x00001234, 0x00007fff, 0x00008000, 0x0000ffff, 0x7fffffff, 0x80000000, 0xffff8000, 0xffffffff];
    const IMMEDIATES: [u16; 6] = [0x0000, 0x0001, 0x1234, 0x7fff, 0x8000, 0xffff];

    type Eval = fn(u32, u16) -> bool;
    const COMPARES: [(u32, &str, Eval); 12] = [
        (0x03, "s_cmpk_eq_i32", |v, i| v as i32 == i as i16 as i32),
        (0x04, "s_cmpk_lg_i32", |v, i| v as i32 != i as i16 as i32),
        (0x05, "s_cmpk_gt_i32", |v, i| v as i32 > i as i16 as i32),
        (0x06, "s_cmpk_ge_i32", |v, i| v as i32 >= i as i16 as i32),
        (0x07, "s_cmpk_lt_i32", |v, i| (v as i32) < i as i16 as i32),
        (0x08, "s_cmpk_le_i32", |v, i| v as i32 <= i as i16 as i32),
        (0x09, "s_cmpk_eq_u32", |v, i| v == i as u32),
        (0x0a, "s_cmpk_lg_u32", |v, i| v != i as u32),
        (0x0b, "s_cmpk_gt_u32", |v, i| v > i as u32),
        (0x0c, "s_cmpk_ge_u32", |v, i| v >= i as u32),
        (0x0d, "s_cmpk_lt_u32", |v, i| v < i as u32),
        (0x0e, "s_cmpk_le_u32", |v, i| v <= i as u32),
    ];

    fn sop1(op: u32, d: u32, s: u32) -> u32 {
        0xbe80_0000 | (d << 16) | (op << 8) | s
    }
    fn sop2(op: u32, d: u32, s0: u32, s1: u32) -> u32 {
        0x8000_0000 | (op << 23) | (d << 16) | (s1 << 8) | s0
    }
    fn sopk(op: u32, r: u32, imm: u16) -> u32 {
        0xb000_0000 | (op << 23) | (r << 16) | imm as u32
    }
    fn vop1(op: u32, d: u32, s: u32) -> u32 {
        0x7e00_0000 | (d << 17) | (op << 9) | s
    }
    fn vop2(op: u32, d: u32, s0: u32, s1: u32) -> u32 {
        (op << 25) | (d << 17) | (s1 << 9) | s0
    }

    fn code() -> Vec<u32> {
        let store_ops = (ZERO << 24) | ((OUTPUT_REGISTER / 4) << 16) | (DATA_VECTOR << 8) | INDEX_VECTOR;
        let mut code = vec![vop2(0x1a, INDEX_VECTOR, SIX, 0)];
        for value in 0..VALUES.len() as u32 {
            for (imm_index, &imm) in IMMEDIATES.iter().enumerate() {
                code.push(sop1(0x03, MASK_REGISTER, ZERO));
                for (c, &(op, _, _)) in COMPARES.iter().enumerate() {
                    code.push(sopk(op, VALUE_REGISTER + value, imm));
                    code.push(sop2(0x0a, BIT_REGISTER, LITERAL, ZERO));
                    code.push(1 << c);
                    code.push(sop2(0x10, MASK_REGISTER, MASK_REGISTER, BIT_REGISTER));
                }
                let result = value * IMMEDIATES.len() as u32 + imm_index as u32;
                code.push(vop1(0x01, DATA_VECTOR, MASK_REGISTER));
                code.push(0xe070_2000 | (result * 4));
                code.push(store_ops);
            }
        }
        code.push(0xbf81_0000);
        code
    }

    /// s_cmpk_* signed / unsigned compares of every value against every
    /// 16-bit immediate.
    #[test]
    fn sopk_compare() {
        let mut ud = vec![0u32; VALUE_REGISTER as usize + VALUES.len()];
        ud[4..8].copy_from_slice(&strided_desc(OUT_BASE, THREADS * RESULTS as u32));
        ud[8..].copy_from_slice(&VALUES);
        let d = Dispatch::new(&code(), 32, THREADS, ud).output(OUT_BASE, THREADS as usize * RESULTS);
        let out = d.run().unwrap().remove(0);
        let mut failures = Vec::new();
        for tid in 0..THREADS as usize {
            for (vi, &v) in VALUES.iter().enumerate() {
                for (ii, &imm) in IMMEDIATES.iter().enumerate() {
                    let mask = out[tid * RESULTS + vi * IMMEDIATES.len() + ii];
                    if mask >> COMPARES.len() != 0 {
                        failures.push(format!("thread {tid} result for {v:#x}, {imm:#x} is {mask:#x}"));
                        continue;
                    }
                    for (c, &(_, name, eval)) in COMPARES.iter().enumerate() {
                        let got = (mask >> c) & 1 != 0;
                        let want = eval(v, imm);
                        if got != want {
                            failures.push(format!("thread {tid} {name} {v:#x}, {imm:#x} is {got}, expected {want}"));
                        }
                    }
                }
            }
        }
        assert_clean(failures);
    }

    /// s_waitcnt_{vscnt,vmcnt,expcnt,lgkmcnt} (SOPK) keep SCC and their
    /// register operand.
    #[test]
    fn sopk_waitcnt() {
        const WAITS: [(u32, &str); 4] = [(0x17, "s_waitcnt_vscnt"), (0x18, "s_waitcnt_vmcnt"), (0x19, "s_waitcnt_expcnt"), (0x1a, "s_waitcnt_lgkmcnt")];
        const NULL: u32 = 0x7d;
        const VALUE: u32 = 0x12345678;
        const REGISTERS: [u32; 2] = [NULL, VALUE_REGISTER];
        const LEVELS: [u16; 3] = [0x0000, 0x003f, 0xffff];
        let sopc = |op: u32, s0: u32, s1: u32| 0xbf00_0000 | (op << 16) | (s1 << 8) | s0;
        let store_ops = (ZERO << 24) | ((OUTPUT_REGISTER / 4) << 16) | (DATA_VECTOR << 8) | INDEX_VECTOR;
        let mut code = vec![vop2(0x1a, INDEX_VECTOR, SIX, 0)];
        let mut result = 0u32;
        for &(op, _) in &WAITS {
            for &reg in &REGISTERS {
                for &level in &LEVELS {
                    code.push(sop1(0x03, MASK_REGISTER, ZERO));
                    code.push(sopc(0x00, VALUE_REGISTER, VALUE_REGISTER));
                    code.push(sopk(op, reg, level));
                    code.push(sop2(0x0a, BIT_REGISTER, 0x81, ZERO));
                    code.push(sop2(0x10, MASK_REGISTER, MASK_REGISTER, BIT_REGISTER));
                    code.push(sopc(0x01, VALUE_REGISTER, VALUE_REGISTER));
                    code.push(sopk(op, reg, level));
                    code.push(sop2(0x0a, BIT_REGISTER, 0x82, ZERO));
                    code.push(sop2(0x10, MASK_REGISTER, MASK_REGISTER, BIT_REGISTER));
                    code.push(vop1(0x01, DATA_VECTOR, MASK_REGISTER));
                    code.push(0xe070_2000 | (result * 4));
                    result += 1;
                    code.push(store_ops);
                    code.push(vop1(0x01, DATA_VECTOR, VALUE_REGISTER));
                    code.push(0xe070_2000 | (result * 4));
                    result += 1;
                    code.push(store_ops);
                }
            }
        }
        code.push(0xbf81_0000);
        let mut ud = vec![0u32; VALUE_REGISTER as usize + 1];
        ud[4..8].copy_from_slice(&strided_desc(OUT_BASE, THREADS * RESULTS as u32));
        ud[8] = VALUE;
        let d = Dispatch::new(&code, 32, THREADS, ud).output(OUT_BASE, THREADS as usize * RESULTS);
        let out = d.run().unwrap().remove(0);
        let mut failures = Vec::new();
        for tid in 0..THREADS as usize {
            let mut r = 0;
            for &(_, name) in &WAITS {
                for &reg in &REGISTERS {
                    for &level in &LEVELS {
                        let mask = out[tid * RESULTS + r];
                        let value = out[tid * RESULTS + r + 1];
                        r += 2;
                        if mask != 1 {
                            failures.push(format!("thread {tid} {name} register {reg:#x} level {level:#x}: flags {mask:#x}, expected 1 (SCC kept)"));
                        }
                        if value != VALUE {
                            failures.push(format!("thread {tid} {name} register {reg:#x} level {level:#x}: s8 is {value:#x}, expected {VALUE:#x}"));
                        }
                    }
                }
            }
        }
        assert_clean(failures);
    }
}

mod sopk_misc {
    use super::*;
    include!("vectors/scalar_sopk_misc.rs");

    /// s_cmovk_i32 taken / not taken, s_setreg_imm32_b32 (non-rounding
    /// field) and a positive s_cmovk.
    #[test]
    fn sopk_misc() {
        let rows = vec![[0u32; 4]; THREADS as usize];
        let out = run_rows(&CODE, &rows, 16);
        let mut failures = Vec::new();
        for tid in 0..THREADS as usize {
            for i in 0..4 {
                let got = out[tid * 16 + i];
                if got != EXPECTED[i] {
                    failures.push(format!("scalar sopk misc: lane {tid} {} is {got:#010x}, expected {:#010x}", NAMES[i].unwrap(), EXPECTED[i]));
                }
            }
        }
        assert_clean(failures);
    }
}

mod getreg {
    use super::*;
    include!("vectors/scalar_getreg.rs");

    /// s_getreg_b32 hwreg(MODE, 0, 4) and hwreg(MODE, 2, 2) read 0 with the
    /// default round mode.
    #[test]
    fn getreg_mode_round() {
        let rows = vec![[0u32; 4]; THREADS as usize];
        let out = run_rows(&CODE, &rows, 16);
        let mut failures = Vec::new();
        for tid in 0..THREADS as usize {
            for (i, name) in ["hwreg(MODE, 0, 4)", "hwreg(MODE, 2, 2)"].iter().enumerate() {
                let got = out[tid * 16 + i];
                if got != 0 {
                    failures.push(format!("s_getreg_b32: lane {tid} {name} is {got:#010x}, expected 0"));
                }
            }
        }
        assert_clean(failures);
    }
}
