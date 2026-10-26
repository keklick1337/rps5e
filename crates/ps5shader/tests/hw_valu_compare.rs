// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! VOPC compare semantics against hardware-measured reference rows:
//! f16/f32/i64/u64 compares with NaN/edge operands, f64 compares and
//! v_cmp_class_f64, and the full e32/e64 VOPC/VOPCX table in wave32 and
//! wave64.

use ps5shader::interp::Machine;
use ps5shader::ssa::SystemValue;
use ps5shader::testing::{buffer_descriptor, ComputeRun};

const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;
const POISON: u32 = 0xDEAD_BEEF;

/// One compute dispatch of `threads` invocations split into waves of
/// `wave`: s[0:3] input V#, s[4:7] output V#, v0 = local invocation id.
/// `raw` selects stride-0 descriptors sized in bytes, otherwise stride-4
/// descriptors sized in records. The output starts out as 0xdeadbeef.
fn dispatch(code: &[u32], wave: u32, threads: u32, input: &[u32], out_words: usize, raw: bool) -> Result<Vec<u32>, String> {
    dispatch_mode(code, wave, threads, input, out_words, raw, ps5shader::translate::FloatMode::default())
}

/// [`dispatch`] compiled for the float controls `mode`.
fn dispatch_mode(code: &[u32], wave: u32, threads: u32, input: &[u32], out_words: usize, raw: bool, mode: ps5shader::translate::FloatMode) -> Result<Vec<u32>, String> {
    let (i, o) = if raw {
        (buffer_descriptor(IN_BASE, 0, input.len() as u32 * 4, 0x3101_6FAC), buffer_descriptor(OUT_BASE, 0, out_words as u32 * 4, 0x3101_6FAC))
    } else {
        (buffer_descriptor(IN_BASE, 4, input.len() as u32, 0x1101_6FAC), buffer_descriptor(OUT_BASE, 4, out_words as u32, 0x1101_6FAC))
    };
    let user: Vec<u32> = i.iter().chain(o.iter()).copied().collect();
    let mut out = vec![POISON; out_words];
    let mut first = 0;
    while first < threads {
        let lanes = (threads - first).min(wave);
        let mut run = ComputeRun::new(code.to_vec(), wave, lanes, user.clone());
        run.float_mode = mode;
        let setup = |m: &mut Machine| {
            m.memory.write_words(IN_BASE, input);
            m.memory.write_words(OUT_BASE, &vec![POISON; out_words]);
            m.system.insert(SystemValue::LocalInvocationId(0), (first..first + lanes).collect());
        };
        let m = run.run_checked(&setup, &[(OUT_BASE, out_words)])?;
        for (k, w) in m.memory.read_words(OUT_BASE, out_words).into_iter().enumerate() {
            if w != POISON {
                out[k] = w;
            }
        }
        first += lanes;
    }
    Ok(out)
}

fn assert_clean(what: &str, fails: &[String]) {
    assert!(fails.is_empty(), "{what}: {} failures:\n{}", fails.len(), fails.join("\n"));
}

fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

mod vopc_compare {
    use super::*;
    include!("vectors/valu_vopc_compare.rs");

    fn half_value(bits: u32) -> f32 {
        let exponent = (bits >> 10) & 0x1F;
        let fraction = bits & 0x3FF;
        let magnitude = if exponent == 0x1F {
            if fraction != 0 {
                f32::NAN
            } else {
                f32::INFINITY
            }
        } else if exponent == 0 {
            fraction as f32 * 2f32.powi(-24)
        } else {
            (fraction | 0x400) as f32 * 2f32.powi(exponent as i32 - 25)
        };
        if bits & 0x8000 != 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    fn fill_input() -> Vec<u32> {
        let mut state = 0x853c49e6748fea9bu64;
        let mut next = || (lcg(&mut state) >> 33) as u32;
        let mut input = vec![0u32; (THREADS * INPUTS) as usize];
        let (nf, ni, nh) = (FLOAT_EDGES.len() as u32, INTEGER_EDGES.len() as u32, HALF_EDGES.len() as u32);
        for tid in 0..THREADS {
            let w = &mut input[(tid * INPUTS) as usize..];
            w[0] = FLOAT_EDGES[(tid % nf) as usize];
            w[1] = FLOAT_EDGES[((tid / nf + tid) % nf) as usize];
            let a = if tid < 16 { INTEGER_EDGES[(tid % ni) as usize] } else { ((next() as u64) << 32) | next() as u64 };
            let b = if tid < 16 {
                INTEGER_EDGES[((tid / ni + tid * 3) % ni) as usize]
            } else if tid % 5 == 0 {
                a
            } else {
                ((next() as u64) << 32) | next() as u64
            };
            w[2] = a as u32;
            w[3] = (a >> 32) as u32;
            w[4] = b as u32;
            w[5] = (b >> 32) as u32;
            w[6] = 0xABCD_0000 | HALF_EDGES[(tid % nh) as usize] as u32;
            w[7] = 0x1234_0000 | HALF_EDGES[((tid / nh + tid * 5) % nh) as usize] as u32;
        }
        input
    }

    fn check(input: &[u32], round: &str, fails: &mut Vec<String>) {
        let out = match dispatch(&COMPARE_CODE, 32, THREADS, input, (THREADS * RESULTS) as usize, false) {
            Ok(o) => o,
            Err(e) => return fails.push(e),
        };
        for tid in 0..THREADS as usize {
            let i = &input[tid * INPUTS as usize..];
            let (fa, fb) = (f32::from_bits(i[0]), f32::from_bits(i[1]));
            let ua = i[2] as u64 | ((i[3] as u64) << 32);
            let ub = i[4] as u64 | ((i[5] as u64) << 32);
            let (sa, sb) = (ua as i64, ub as i64);
            let (ha, hb) = (half_value(i[6] & 0xFFFF), half_value(i[7] & 0xFFFF));
            let expected: [bool; 54] = [
                false,
                !fa.is_nan() && !fb.is_nan(),
                fa.is_nan() || fb.is_nan(),
                true,
                false,
                true,
                false,
                true,
                false,
                sa < sb,
                sa <= sb,
                sa > sb,
                sa != sb,
                sa >= sb,
                true,
                false,
                sa < sb,
                sa == sb,
                sa <= sb,
                sa > sb,
                sa >= sb,
                true,
                false,
                ua <= ub,
                ua >= ub,
                true,
                false,
                ua < ub,
                ua == ub,
                ua <= ub,
                ua > ub,
                ua >= ub,
                true,
                false,
                !ha.is_nan() && !hb.is_nan(),
                ha.is_nan() || hb.is_nan(),
                !(ha >= hb),
                !(ha < hb || ha > hb),
                !(ha > hb),
                !(ha <= hb),
                !(ha < hb),
                true,
                false,
                ha < hb || ha > hb,
                !ha.is_nan() && !hb.is_nan(),
                ha.is_nan() || hb.is_nan(),
                !(ha >= hb),
                !(ha < hb || ha > hb),
                !(ha <= hb),
                true,
                !(ha >= hb.abs()),
                sb <= sa,
                ub >= ua,
                !fa.is_nan() && !fb.is_nan(),
            ];
            for (k, &e) in expected.iter().enumerate() {
                let actual = (out[tid * RESULTS as usize + k / 32] >> (k % 32)) & 1 != 0;
                if actual != e {
                    fails.push(format!("{round}: thread {tid} ({:08x?}) compare {k} is {actual}, expected {e}", &i[..8]));
                }
            }
        }
    }

    #[test]
    fn results() {
        let mut input = fill_input();
        let mut fails = Vec::new();
        check(&input, "edges", &mut fails);
        let nh = HALF_EDGES.len() as u32;
        let mut pair_base = 0;
        while pair_base < nh * nh {
            for tid in 0..THREADS {
                let pair = (pair_base + tid) % (nh * nh);
                input[(tid * INPUTS + 6) as usize] = 0xABCD_0000 | HALF_EDGES[(pair / nh) as usize] as u32;
                input[(tid * INPUTS + 7) as usize] = 0x1234_0000 | HALF_EDGES[(pair % nh) as usize] as u32;
            }
            check(&input, &format!("half pairs from {pair_base}"), &mut fails);
            pair_base += THREADS;
        }
        assert_clean("vopc compares", &fails);
    }
}

mod vopc_compare_f64 {
    use super::*;
    include!("vectors/valu_vopc_compare_f64.rs");

    const EDGE_ROUNDS: u32 = (EDGES.len() as u32 * EDGES.len() as u32).div_ceil(THREADS);

    fn fill_input(round: u32) -> Vec<u32> {
        let mut state = 0x853c49e6748fea9bu64.wrapping_add(round as u64);
        let mut next = || {
            let high = (lcg(&mut state) >> 32) as u32;
            let low = (lcg(&mut state) >> 32) as u32;
            ((high as u64) << 32) | low as u64
        };
        let mut input = vec![0u32; (THREADS * INPUTS) as usize];
        let ne = EDGES.len() as u32;
        for tid in 0..THREADS {
            let (a, b, mask);
            if round < EDGE_ROUNDS {
                let pair = round * THREADS + tid;
                let first = pair % ne;
                let second = (pair / ne) % ne;
                a = EDGES[first as usize];
                b = EDGES[second as usize];
                mask = MASKS[((first + second) % MASKS.len() as u32) as usize];
            } else {
                a = next();
                b = if tid % 5 == 0 {
                    a
                } else if tid % 7 == 0 {
                    a ^ 0x8000_0000_0000_0000
                } else {
                    next()
                };
                mask = next() as u32;
            }
            let w = &mut input[(tid * INPUTS) as usize..];
            w[0] = a as u32;
            w[1] = (a >> 32) as u32;
            w[2] = b as u32;
            w[3] = (b >> 32) as u32;
            w[4] = mask;
        }
        input
    }

    fn class_bit(bits: u64) -> u32 {
        use std::num::FpCategory::*;
        let negative = bits >> 63 != 0;
        match f64::from_bits(bits).classify() {
            Nan => (bits & 0x0008_0000_0000_0000 != 0) as u32,
            Infinite => {
                if negative {
                    2
                } else {
                    9
                }
            }
            Normal => {
                if negative {
                    3
                } else {
                    8
                }
            }
            Subnormal => {
                if negative {
                    4
                } else {
                    7
                }
            }
            Zero => {
                if negative {
                    5
                } else {
                    6
                }
            }
        }
    }

    fn class_match(bits: u64, mask: u32) -> bool {
        (mask >> class_bit(bits)) & 1 != 0
    }

    /// FLOAT_MODE / IEEE_MODE pairs the compares run in and whether f64
    /// input denormals flush (FLOAT_MODE bit 6 clear).
    fn modes() -> Vec<(&'static str, ps5shader::translate::FloatMode, bool)> {
        use ps5shader::translate::FloatMode;
        let mut v = vec![("default mode", FloatMode::default(), false)];
        for ieee in [false, true] {
            for (name, mode, flushed) in [("mode 0", 0x30u8, true), ("mode 1", 0x60, false), ("mode 2", 0x90, true), ("mode 3", 0xc0, false)] {
                v.push((name, FloatMode { mode, dx10_clamp: true, ieee, fp16_overflow: false }, flushed));
            }
        }
        v
    }

    #[test]
    fn results() {
        let mut fails = Vec::new();
        for (mode_name, mode, flushed) in modes() {
            let value = |bits: u64| {
                let v = f64::from_bits(bits);
                if flushed && v.is_subnormal() {
                    0.0f64.copysign(v)
                } else {
                    v
                }
            };
            check_mode(mode_name, mode, &value, flushed, &mut fails);
        }
        assert_clean("vopc f64 compares", &fails);
    }

    fn check_mode(mode_name: &str, mode: ps5shader::translate::FloatMode, value: &dyn Fn(u64) -> f64, flushed: bool, fails: &mut Vec<String>) {
        for round in 0..EDGE_ROUNDS + RANDOM_ROUNDS {
            let input = fill_input(round);
            let out = match dispatch_mode(&COMPARE_CODE, 32, THREADS, &input, (THREADS * RESULTS) as usize, false, mode) {
                Ok(o) => o,
                Err(e) => {
                    fails.push(e);
                    break;
                }
            };
            for tid in 0..THREADS as usize {
                let i = &input[tid * INPUTS as usize..];
                let abits = i[0] as u64 | ((i[1] as u64) << 32);
                let bbits = i[2] as u64 | ((i[3] as u64) << 32);
                let mask = i[4];
                let (a, b) = (value(abits), value(bbits));
                let unordered = a.is_nan() || b.is_nan();
                let base = [
                    false,
                    a < b,
                    a == b,
                    a <= b,
                    a > b,
                    a < b || a > b,
                    a >= b,
                    !unordered,
                    unordered,
                    !(a >= b),
                    !(a < b || a > b),
                    !(a > b),
                    !(a <= b),
                    !(a == b),
                    !(a < b),
                    true,
                    class_match(abits, mask),
                ];
                let mut expected = [false; COMPARES as usize];
                for (k, &v) in base.iter().enumerate() {
                    expected[k] = v;
                    expected[base.len() + k] = v;
                }
                expected[34] = 3.0 < a;
                expected[35] = 1.0 >= a;
                expected[36] = f64::from_bits(0x3fc45f306dc9c882) == a;
                expected[37] = (if flushed { 0.0 } else { f64::from_bits(1) }) < a;
                expected[38] = a.abs() < b;
                expected[39] = class_match(abits & 0x7FFF_FFFF_FFFF_FFFF, mask);
                expected[40] = !(a.abs() > b.abs());
                for (k, &e) in expected.iter().enumerate() {
                    let actual = (out[tid * RESULTS as usize + k / 32] >> (k % 32)) & 1 != 0;
                    if actual != e {
                        fails.push(format!("{mode_name} ieee {} round {round} thread {tid} ({abits:#018x}, {bbits:#018x}, mask {mask:#x}) compare {k} is {actual}, expected {e}", mode.ieee));
                    }
                }
            }
        }
    }
}

mod vopc_compare_table {
    use super::*;
    include!("vectors/valu_vopc_compare_table.rs");

    fn check(code: &[u32], wave: u32) {
        let input: Vec<u32> = ROWS.iter().flatten().copied().collect();
        let out = dispatch(code, wave, wave, &input, (MAX_THREADS * RESULTS) as usize, true).unwrap();
        let mut fails = Vec::new();
        for tid in 0..wave as usize {
            for index in 0..COMPARES as usize {
                let actual = (out[tid * RESULTS as usize + index / 32] >> (index % 32)) & 1;
                let expected = (EXPECTED[tid][index / 32] >> (index % 32)) & 1;
                if actual != expected {
                    fails.push(format!("wave{wave}: lane {tid} ({:08x?}) {} is {actual}, expected {expected}", ROWS[tid], NAMES[index]));
                }
            }
        }
        assert_clean("vopc compare table", &fails);
    }

    #[test]
    fn wave32() {
        check(&WAVE32_CODE, 32);
    }

    #[test]
    fn wave64() {
        check(&WAVE64_CODE, 64);
    }
}
