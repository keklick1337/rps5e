// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Scalar program flow (s_endpgm variants, branches, s_setpc / s_swappc,
//! loops, s_sleep, uniform loops, shader clocks) against hardware-measured
//! reference rows.

include!("vectors/scalar_common.rs");

/// s[0:3] = raw input V# over `input`, s[4:7] = raw output V# of
/// `out_words` words.
fn raw_io(code: &[u32], wave: u32, threads: u32, input: Vec<u32>, out_words: usize) -> Result<Vec<u32>, String> {
    let ud = user_data(raw_desc(IN_BASE, (input.len() * 4) as u32), raw_desc(OUT_BASE, (out_words * 4) as u32), 8);
    Dispatch::new(code, wave, threads, ud).buffer(IN_BASE, input).output(OUT_BASE, out_words).run().map(|mut v| v.remove(0))
}

/// The strided layout with a selector in s8: s[0:3] input V#, s[4:7]
/// output V# (stride 4, one record per dword), `threads` lanes, words
/// `tid * 4` hold `tid * 0x01010101 + 7`.
fn selector_io(code: &[u32], wave: u32, threads: u32, selector: u32) -> (Vec<u32>, Vec<u32>) {
    let words = threads as usize * 4;
    let mut input = vec![0u32; words];
    for tid in 0..threads as usize {
        input[tid * 4] = (tid as u32).wrapping_mul(0x0101_0101).wrapping_add(7);
    }
    let mut ud = user_data(strided_desc(IN_BASE, words as u32), strided_desc(OUT_BASE, words as u32), 9);
    ud[8] = selector;
    let out = Dispatch::new(code, wave, threads, ud).buffer(IN_BASE, input.clone()).output(OUT_BASE, words).run().unwrap().remove(0);
    (input, out)
}

mod program_end {
    use super::*;

    fn check(code: &[u32]) {
        let mut input = vec![0u32; 32 * 4];
        for tid in 0..32u32 {
            input[tid as usize * 4] = tid * 0x0101_0101 + 7;
        }
        let out = raw_io(code, 32, 32, input.clone(), 32 * 16).unwrap();
        let mut failures = Vec::new();
        for tid in 0..32usize {
            if out[tid * 16] != input[tid * 4] + 1 {
                failures.push(format!("lane {tid} store before the end is {:#x}, expected {:#x}", out[tid * 16], input[tid * 4] + 1));
            }
            if out[tid * 16 + 1] != UNTOUCHED {
                failures.push(format!("lane {tid} stored past the end: {:#x}", out[tid * 16 + 1]));
            }
        }
        assert_clean(failures);
    }

    mod plain {
        include!("vectors/scalar_program_end.rs");
    }
    mod saved {
        include!("vectors/scalar_program_end_saved.rs");
    }

    /// s_atc_probe / s_atc_probe_buffer / s_dcache_discard(_x2) (SMEM
    /// 0x26..0x29), s_round_mode 0, then s_endpgm_ordered_ps_done and
    /// s_endpgm_saved ending the program.
    #[test]
    fn endpgm_variants() {
        check(&plain::CODE);
    }

    /// s_setkill 0 / s_setkill 1 around the stores.
    #[test]
    fn setkill() {
        check(&plain::SETKILL_CODE);
    }

    /// s_endpgm_saved first, then s_endpgm_ordered_ps_done.
    #[test]
    fn endpgm_saved() {
        check(&saved::CODE);
    }

}

mod program_flow {
    use super::*;
    include!("vectors/scalar_program_flow.rs");

    fn run(code: &[u32], wave: u32) -> Vec<u32> {
        let input: Vec<u32> = ROWS.iter().flatten().copied().collect();
        raw_io(code, wave, wave, input, 64 * 16).unwrap()
    }

    /// s_cbranch_exec{z,nz} / vcc{z,nz} on partial masks, s_cbranch_cdbg*,
    /// branch hints.
    #[test]
    fn program_flow_wave32() {
        let out = run(&WAVE32_CODE, 32);
        assert_clean(check_rows("program flow wave32", &out, 16, &EXPECTED32, &NAMES, &|_| true));
    }

    #[test]
    fn program_flow_wave64() {
        let out = run(&WAVE64_CODE, 64);
        assert_clean(check_rows("program flow wave64", &out, 16, &EXPECTED64, &NAMES, &|_| true));
    }
}

mod setpc {
    use super::*;
    include!("vectors/scalar_setpc.rs");

    fn run(code: &[u32], wave: u32) -> Vec<u32> {
        let input: Vec<u32> = ROWS.iter().flatten().copied().collect();
        raw_io(code, wave, wave, input, 64 * 16).unwrap()
    }

    /// Jump tables through s_getpc_b64 + s_load + s_setpc_b64, forward
    /// skips and loops built from s_setpc_b64.
    #[test]
    fn setpc_wave32() {
        let out = run(&WAVE32_CODE, 32);
        assert_clean(check_rows("setpc wave32", &out, 16, &EXPECTED32, &NAMES, &|_| true));
    }

    #[test]
    fn setpc_wave64() {
        let out = run(&WAVE64_CODE, 64);
        assert_clean(check_rows("setpc wave64", &out, 16, &EXPECTED64, &NAMES, &|_| true));
    }

    /// The program with a v_mov scheduled between the jump table's bound
    /// (s_min_u32) and shift (s_lshl_b32) and another between the shift and
    /// the table's s_getpc_b64; branches relocated.
    fn scheduled(original: &[u32]) -> Vec<u32> {
        let mut code = original.to_vec();
        code.insert(25, 0x7e3c_0280);
        code.insert(24, 0x7e3e_0280);
        let relocated = |word: i32| word + (word >= 24) as i32 + (word >= 25) as i32;
        for (word, &instruction) in original.iter().enumerate() {
            let opcode = instruction & 0xffff_0000;
            if opcode != 0xbf82_0000 && opcode != 0xbf85_0000 {
                continue;
            }
            let word = word as i32;
            let target = word + 1 + (instruction & 0xffff) as u16 as i16 as i32;
            let offset = relocated(target) - relocated(word) - 1;
            code[relocated(word) as usize] = opcode | (offset as u32 & 0xffff);
        }
        code
    }

    /// The bound and shift of a jump table index found when other
    /// instructions are scheduled between them (reference eaa6d89ff).
    #[test]
    fn setpc_scheduled_table_index() {
        let out = run(&scheduled(&WAVE32_CODE), 32);
        assert_clean(check_rows("setpc wave32 scheduled", &out, 16, &EXPECTED32, &NAMES, &|_| true));
        let out = run(&scheduled(&WAVE64_CODE), 64);
        assert_clean(check_rows("setpc wave64 scheduled", &out, 16, &EXPECTED64, &NAMES, &|_| true));
    }
}

mod swappc {
    use super::*;
    include!("vectors/scalar_swappc.rs");

    fn check(code: &[u32], wave: u32) {
        let mut input = vec![0u32; 64 * 4];
        for tid in 0..64usize {
            input[tid * 4] = SELECTORS[tid % SELECTORS.len()];
            input[tid * 4 + 1] = 0x9e37_79b9u32.wrapping_mul(tid as u32 + 1);
        }
        let out = raw_io(code, wave, wave, input.clone(), 64 * 4).unwrap();
        let mut failures = Vec::new();
        for tid in 0..wave as usize {
            let selector = input[tid * 4];
            let value = input[tid * 4 + 1];
            let expected = [
                value.wrapping_add(GETPC_FIRST_CASES[selector.min(3) as usize]),
                value.wrapping_add(INDEX_FIRST_CASES[selector.wrapping_sub(1).min(3) as usize]),
                value.wrapping_add(0x600d_0000),
                value.wrapping_mul(3),
            ];
            for (i, &want) in expected.iter().enumerate() {
                let got = out[tid * 4 + i];
                if got != want {
                    failures.push(format!("swappc wave{wave}: lane {tid} {} is {got:#010x}, expected {want:#010x}", NAMES[i].unwrap()));
                }
            }
        }
        assert_clean(failures);
    }

    /// s_swappc_b64 dword jump tables (s_getpc first / index first) and long
    /// branches forward through VCC and backward.
    #[test]
    fn swappc_wave32() {
        check(&WAVE32_CODE, 32);
    }

    #[test]
    fn swappc_wave64() {
        check(&WAVE64_CODE, 64);
    }
}

mod branch_past_endpgm {
    use super::*;
    include!("vectors/scalar_branch_past_endpgm.rs");

    /// Blocks placed after an s_endpgm and entered by a branch from below
    /// (64 invocations = two wave32 waves).
    #[test]
    fn branch_past_endpgm() {
        let cases: [(&str, &[u32], u32, u32); 6] = [
            ("return before the tail blocks", &TAIL_BLOCK_CODE, 0, 1),
            ("tail block entered from the tail block after it", &TAIL_BLOCK_CODE, 1, 2),
            ("last tail block", &TAIL_BLOCK_CODE, 2, 3),
            ("return before the loop", &LOOP_BODY_CODE, 0, 64),
            ("loop body entered once from the test after it", &LOOP_BODY_CODE, 1, 1),
            ("loop body entered three times from the test after it", &LOOP_BODY_CODE, 3, 3),
        ];
        let mut failures = Vec::new();
        for (name, code, selector, added) in cases {
            let (input, out) = selector_io(code, 32, 64, selector);
            for tid in 0..64usize {
                let want = input[tid * 4].wrapping_add(added);
                if out[tid * 4] != want {
                    failures.push(format!("{name}: thread {tid} is {:#x}, expected {want:#x}", out[tid * 4]));
                }
            }
        }
        assert_clean(failures);
    }
}

mod loop_ending_exits {
    use super::*;
    include!("vectors/scalar_loop_ending_exits.rs");

    /// Early exits (s_endpgm) inside a loop whose last block ends the loop.
    #[test]
    fn loop_ending_exits() {
        let cases: [(&str, u32, u32); 5] = [
            ("early exit before the loop", 0, 32),
            ("early exit from the first iteration", 1, 33),
            ("early exit from the third iteration", 3, 35),
            ("early exit from the last iteration", 4, 36),
            ("loop exit to the end of the program", 5, 68),
        ];
        let mut failures = Vec::new();
        for wave in [32, 64] {
            for (name, selector, added) in cases {
                let (input, out) = selector_io(&CODE, wave, 64, selector);
                for tid in 0..64usize {
                    let want = input[tid * 4].wrapping_add(added);
                    if out[tid * 4] != want {
                        failures.push(format!("wave{wave} {name}: thread {tid} is {:#x}, expected {want:#x}", out[tid * 4]));
                    }
                }
            }
        }
        assert_clean(failures);
    }
}

mod wave_sleep {
    use super::*;
    include!("vectors/scalar_wave_sleep.rs");

    /// s_sleep / s_nop / s_setprio-free code around a load + add + store
    /// (64 invocations = two wave32 waves).
    #[test]
    fn wave_sleep() {
        check(&WAVE_SLEEP_CODE);
    }

    fn check(code: &[u32]) {
        let (input, out) = selector_io(code, 32, 64, 0);
        let mut failures = Vec::new();
        for tid in 0..64usize {
            let want = input[tid * 4].wrapping_add(1);
            if out[tid * 4] != want {
                failures.push(format!("wave sleep: thread {tid} is {:#x}, expected {want:#x}", out[tid * 4]));
            }
        }
        assert_clean(failures);
    }
}

mod wave_uniform {
    use super::*;
    include!("vectors/scalar_wave_uniform.rs");

    /// A uniform scalar loop over s_buffer_load_dwordx2 values feeding a
    /// vector result (wave64).
    #[test]
    fn wave_uniform() {
        let input = vec![3u32, 5, 0, 0];
        let mut sum = 0u32;
        for count in 0..input[1] {
            sum = sum.wrapping_add(input[0]);
            sum = sum.wrapping_add(sum.wrapping_mul(count));
        }
        let uniform = sum + 7;
        let out = raw_io(&WAVE_CODE, 64, 64, input, 64).unwrap();
        let mut failures = Vec::new();
        for tid in 0..64u32 {
            let want = tid * 20 + uniform;
            if out[tid as usize] != want {
                failures.push(format!("wave uniform: thread {tid} stored {}, expected {want}", out[tid as usize]));
            }
        }
        assert_clean(failures);
    }
}

mod shader_clock {
    use super::*;
    include!("vectors/scalar_shader_clock.rs");

    fn run(code: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let mut input = vec![0u32; 32 * 4];
        for tid in 0..32u32 {
            input[tid as usize * 4] = tid + 1;
        }
        let out = raw_io(code, 32, 32, input.clone(), 32 * 16).unwrap();
        (input, out)
    }

    fn loop_result(x: u32) -> u32 {
        let mut sum = 0u32;
        for _ in 0..2000 {
            sum = (sum & 0xff_ffff).wrapping_mul(3).wrapping_add(x);
        }
        sum
    }

    fn pair(w: &[u32]) -> u64 {
        w[0] as u64 | ((w[1] as u64) << 32)
    }

    /// s_memtime / s_memrealtime: wave-uniform, monotonic, non-zero; plus a
    /// 2000-iteration scalar loop between the reads.
    #[test]
    fn shader_clock() {
        let (input, out) = run(&CODE);
        let mut failures = Vec::new();
        for tid in 0..32usize {
            let o = &out[tid * 16..tid * 16 + 16];
            let first = &out[(tid & !7) * 16..(tid & !7) * 16 + 8];
            if o[..8] != first[..] {
                failures.push(format!("lane {tid}: clock read not uniform over the wave"));
            }
            if pair(&o[4..]) < pair(&o[0..]) || pair(&o[6..]) < pair(&o[2..]) {
                failures.push(format!("lane {tid}: clock went back"));
            }
            if pair(&o[4..]) == 0 || pair(&o[6..]) == 0 {
                failures.push(format!("lane {tid}: clocks read zero"));
            }
            let want = loop_result(input[tid * 4]);
            if o[8] != want {
                failures.push(format!("lane {tid}: loop result is {:#x}, expected {want:#x}", o[8]));
            }
        }
        assert_clean(failures);
    }

}
