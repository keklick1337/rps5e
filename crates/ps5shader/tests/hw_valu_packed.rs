// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Packed/dot/mixed-precision semantics against hardware-measured
//! reference rows: v_dot* integer dot products, VOP3P 16-bit integer ops
//! with clamp and op_sel, and v_fma_mix*.

use ps5shader::interp::Machine;
use ps5shader::ssa::SystemValue;
use ps5shader::testing::{buffer_descriptor, ComputeRun};
use ps5shader::translate::FloatMode;

const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;
const POISON: u32 = 0xDEAD_BEEF;

/// One compute dispatch of `threads` invocations split into waves of
/// `wave`: s[0:3] input V#, s[4:7] output V#, v0 = local invocation id.
/// `raw` selects stride-0 descriptors sized in bytes, otherwise stride-4
/// descriptors sized in records. The output starts out as 0xdeadbeef.
fn dispatch(code: &[u32], wave: u32, threads: u32, input: &[u32], out_words: usize, raw: bool) -> Result<Vec<u32>, String> {
    dispatch_mode(code, wave, threads, input, out_words, raw, FloatMode::default())
}

/// `dispatch` compiled for the float controls `mode`.
fn dispatch_mode(code: &[u32], wave: u32, threads: u32, input: &[u32], out_words: usize, raw: bool, mode: FloatMode) -> Result<Vec<u32>, String> {
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

mod integer_dot {
    use super::*;
    include!("vectors/valu_integer_dot.rs");

    fn element(value: u32, offset: u32, bits: u32, sign: bool) -> u32 {
        let field = (value >> offset) & ((1u32 << bits) - 1);
        let sign_bit = 1u32 << (bits - 1);
        if sign {
            (field ^ sign_bit).wrapping_sub(sign_bit)
        } else {
            field
        }
    }

    fn dot(l: u32, r: u32, bits: u32, sign: bool) -> u32 {
        (0..32).step_by(bits as usize).fold(0u32, |sum, offset| sum.wrapping_add(element(l, offset, bits, sign).wrapping_mul(element(r, offset, bits, sign))))
    }

    const NAMES: [&str; 10] = [
        "v_dot2_i32_i16",
        "v_dot2_u32_u16",
        "v_dot4_i32_i8",
        "v_dot4_u32_u8",
        "v_dot8_i32_i4",
        "v_dot8_u32_u4",
        "v_dot4c_i32_i8",
        "v_dot2_i32_i16 with swapped halves",
        "v_dot4_u32_u8 with a literal",
        "v_dot8_i32_i4 with an inline constant",
    ];

    /// v_dot2_i32_i16 op_sel:[1,0,0] op_sel_hi:[0,1,1] (swapped halves of src0).
    const SWAPPED_HALVES: usize = 7;

    fn failures(check: &dyn Fn(usize) -> bool) -> Vec<String> {
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut input = vec![0u32; (THREADS * INPUTS) as usize];
        for tid in 0..THREADS as usize {
            for j in 0..3 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                input[tid * INPUTS as usize + j] = if tid < EDGES.len() { EDGES[tid][j] } else { (state >> 32) as u32 };
            }
        }
        let out = match dispatch(&DOT_CODE, 32, THREADS, &input, (THREADS * RESULTS) as usize, false) {
            Ok(o) => o,
            Err(e) => return vec![e],
        };
        let mut fails = Vec::new();
        for tid in 0..THREADS as usize {
            let w = &input[tid * INPUTS as usize..];
            let (a, b, c) = (w[0], w[1], w[2]);
            let expected = [
                dot(a, b, 16, true).wrapping_add(c),
                dot(a, b, 16, false).wrapping_add(c),
                dot(a, b, 8, true).wrapping_add(c),
                dot(a, b, 8, false).wrapping_add(c),
                dot(a, b, 4, true).wrapping_add(c),
                dot(a, b, 4, false).wrapping_add(c),
                dot(a, b, 8, true).wrapping_add(c),
                dot(a.rotate_left(16), b, 16, true).wrapping_add(c),
                dot(a, LITERAL, 8, false).wrapping_add(c),
                dot(0xFFFF_FFFF, b, 4, true).wrapping_add(c),
            ];
            for (j, &e) in expected.iter().enumerate() {
                let actual = out[tid * RESULTS as usize + j];
                if check(j) && actual != e {
                    fails.push(format!("{}: thread {tid} ({a:#x}, {b:#x}, {c:#x}) is {actual:#x}, expected {e:#x}", NAMES[j]));
                }
            }
        }
        fails
    }

    #[test]
    fn results() {
        assert_clean("integer dot", &failures(&|j| j != SWAPPED_HALVES));
    }

    #[test]
    fn dot2_op_sel_swapped_halves() {
        assert_clean("integer dot", &failures(&|j| j == SWAPPED_HALVES));
    }
}

mod packed_integer_clamp {
    use super::*;
    include!("vectors/valu_packed_integer_clamp.rs");

    #[test]
    fn rows() {
        let input: Vec<u32> = ROWS.iter().flatten().copied().collect();
        let results = RESULTS as usize;
        let out = dispatch(&CODE, 32, THREADS, &input, ROWS.len() * results, true).unwrap();
        let mut fails = Vec::new();
        for (lane, row) in EXPECTED.iter().enumerate() {
            for (col, &e) in row.iter().enumerate() {
                let a = out[lane * results + col];
                if a != e {
                    fails.push(format!("lane {lane} {} ({:08x?}) is {a:#010x}, expected {e:#010x}", NAMES[col], ROWS[lane]));
                }
            }
        }
        assert_clean("packed integer clamp", &fails);
    }
}

mod mix_rounding {
    use super::*;
    include!("vectors/valu_mix_rounding.rs");

    /// v_fma_mixlo/mixhi_f16 round the exact product-sum once to f16; the
    /// other half of the destination keeps its 0xabcd.
    #[test]
    fn mix_rounding() {
        let mut input = vec![0u32; 64 * 4];
        for (tid, row) in MIX_ROUNDING_ROWS.iter().enumerate() {
            input[tid * 4..tid * 4 + 3].copy_from_slice(&row[..3]);
        }
        let out = dispatch(&MIX_ROUNDING_CODE, 32, 64, &input, 64 * 16, false).unwrap();
        let mut fails = Vec::new();
        for (tid, row) in MIX_ROUNDING_ROWS.iter().enumerate() {
            let o = &out[tid * 16..];
            let e = 0xABCD_0000 | row[4];
            if o[row[3] as usize] != e {
                fails.push(format!("lane {tid} ({:#x}, {:#x}, {:#x}) result {} is {:#010x}, expected {e:#010x}", row[0], row[1], row[2], row[3], o[row[3] as usize]));
            }
            if row[3] == 0 && o[1] != (row[4] << 16 | 0xABCD) {
                fails.push(format!("lane {tid} ({:#x}, {:#x}, {:#x}) v_fma_mixhi_f16 is {:#010x}", row[0], row[1], row[2], o[1]));
            }
        }
        assert_clean("mix rounding", &fails);
    }
}

mod mix_precision {
    use super::*;
    include!("vectors/valu_mix_precision.rs");

    fn half_value(bits: u32) -> f32 {
        let exponent = (bits >> 10) & 0x1F;
        let fraction = bits & 0x3FF;
        let magnitude = if exponent == 0 { fraction as f32 * 2f32.powi(-24) } else { (fraction | 0x400) as f32 * 2f32.powi(exponent as i32 - 25) };
        if bits & 0x8000 != 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    /// f16 bits of a value that is exactly a normal f16 (or +-0).
    fn half_bits(value: f32) -> u32 {
        let b = value.to_bits();
        let sign = (b >> 16) & 0x8000;
        if value == 0.0 {
            return sign;
        }
        let exponent = ((b >> 23) & 0xFF) as i32 - 127 + 15;
        assert!((1..31).contains(&exponent) && b & 0x1FFF == 0, "mix precision: a model value is not a normal f16");
        sign | ((exponent as u32) << 10) | ((b >> 13) & 0x3FF)
    }

    fn quarter(index: u32) -> f32 {
        ((index % 17) as i32 - 8) as f32 / 4.0
    }

    fn pack(low: f32, high: f32) -> u32 {
        half_bits(low) | (half_bits(high) << 16)
    }

    fn low(w: u32) -> f32 {
        half_value(w & 0xFFFF)
    }

    fn high(w: u32) -> f32 {
        half_value(w >> 16)
    }

    #[test]
    fn results() {
        let mut input = vec![0u32; (THREADS * INPUTS) as usize];
        for tid in 0..THREADS {
            let w = &mut input[(tid * INPUTS) as usize..];
            w[0] = pack(quarter(tid), quarter(tid * 3 + 5));
            w[1] = pack(quarter(tid * 5 + 2), quarter(tid * 7 + 11));
            w[2] = pack(quarter(tid * 11 + 9), quarter(tid * 13 + 4));
            w[3] = quarter(tid * 2 + 7).to_bits();
        }
        let out = dispatch(&MIX_CODE, 32, THREADS, &input, (THREADS * RESULTS) as usize, false).unwrap();
        let mut fails = Vec::new();
        for tid in 0..THREADS as usize {
            let i = &input[tid * INPUTS as usize..];
            let o = &out[tid * RESULTS as usize..];
            let d = f32::from_bits(i[3]);
            let expected = [
                low(i[0]).mul_add(low(i[1]), low(i[2])),
                high(i[0]).mul_add(high(i[1]), high(i[2])),
                low(i[1]) - low(i[2]),
                high(i[1]) - high(i[2]),
                d.mul_add(high(i[0]), d),
                high(i[0]).mul_add(low(i[1]).abs(), low(i[2])),
            ];
            for (j, &e) in expected.iter().enumerate() {
                let a = f32::from_bits(o[j]);
                if a != e {
                    fails.push(format!("thread {tid} result {j} is {a}, expected {e}"));
                }
            }
            if (o[6] >> 16) != 0xABCD || low(o[6]) != expected[0] {
                fails.push(format!("thread {tid} result 6 is {:#x}: v_fma_mixlo_f16 must write the low half only ({})", o[6], expected[0]));
            }
            if (o[7] & 0xFFFF) != 0xABCD || high(o[7]) != expected[1] {
                fails.push(format!("thread {tid} result 7 is {:#x}: v_fma_mixhi_f16 must write the high half only ({})", o[7], expected[1]));
            }
            let square = f32::from_bits(o[8]);
            if square != d.mul_add(d, d) {
                fails.push(format!("thread {tid} result 8 is {square}, expected {}", d.mul_add(d, d)));
            }
        }
        assert_clean("mix precision", &fails);
    }
}

mod packed_alu {
    use super::*;
    include!("vectors/valu_packed_alu.rs");

    /// Every checked word is exact, NaN bits included: packed f16 NaN
    /// results are picked on the bits (min/max return the other operand,
    /// a signaling NaN is quieted only with IEEE_MODE, and with clamp and
    /// DX10_CLAMP that quieted NaN is +0).
    fn check(mode: FloatMode, expected: &[[u32; 48]; 32], what: &str) {
        let input: Vec<u32> = ROWS.iter().flatten().copied().collect();
        let out = dispatch_mode(&CODE, 32, 32, &input, 32 * 64, true, mode).unwrap();
        let mut fails = Vec::new();
        for (lane, row) in expected.iter().enumerate() {
            for (col, &e) in row.iter().enumerate() {
                let a = out[lane * 64 + col];
                if a != e {
                    fails.push(format!("lane {lane} {:08x?} word {col}: {a:#010x}, expected {e:#010x}", ROWS[lane]));
                }
            }
        }
        assert_clean(what, &fails);
    }

    #[test]
    fn ieee_off() {
        check(FloatMode::default(), &EXPECTED_IEEE_OFF, "packed alu");
    }

    #[test]
    fn ieee_on() {
        check(FloatMode { mode: 0xF0, dx10_clamp: true, ieee: true, fp16_overflow: false }, &EXPECTED, "packed alu, IEEE");
    }
}

/// VOP3P neg_lo / neg_hi on the 16-bit integer ops flip bit 15 of the
/// selected half before the operation (UE5's TSR history update clamps its
/// -1 neighbour offsets with `v_pk_max_i16 v, neg(lit 0x7fff), v`; without
/// the modifier the 3x3 filter's x-1 / y-1 taps read out of bounds). Every
/// packed integer op with neg_lo / neg_hi, op_sel interplay, clamp and the
/// TSR literal case, against the model of the execution test.
mod packed_integer_negate {
    use super::*;

    const RESULTS: usize = 16;
    const CODE: [u32; 47] = [
        0x34020084, 0x34060086, 0xe0381000, 0x80000401, 0xbf8c3f70, 0xcc07410a, 0x300208ff, 0x00007fff,
        0xcc0a410b, 0x58020b04, 0xcc03420c, 0x38020b04, 0xcc08430d, 0x78020b04, 0xcc0c420e, 0x58020b04,
        0xcc0d410f, 0x38020b04, 0xcc014210, 0x38020b04, 0xcc004511, 0x9c1a0b04, 0xcc094612, 0x7c1a0b04,
        0xcc044213, 0x58020b04, 0xcc064314, 0x58020b04, 0xcc054215, 0x78020b04, 0xcc024a16, 0x30020b04,
        0xcc0b5317, 0x68020b04, 0xcc02c218, 0x58020b04, 0xcc074919, 0x30020906, 0xe0781000, 0x80010a03,
        0xe0781010, 0x80010e03, 0xe0781020, 0x80011203, 0xe0781030, 0x80011603, 0xbf810000,
    ];
    const ROWS: [[u32; 4]; 32] = [
        [0x1234021b, 0xffff7fff, 0x8001edcc, 0], [0xc000021b, 0x0002021b, 0x000fedcc, 0], [0xedcc0001, 0x00ff0218, 0x0002fffe, 0],
        [0x000f00ff, 0x000f0218, 0x80100218, 0], [0x000f8000, 0x000f021b, 0x00ff021b, 0], [0x7fff0218, 0x4000000f, 0x80018000, 0],
        [0xffff4000, 0x00ff8000, 0x00021234, 0], [0x7ffeffff, 0x021bffff, 0x8010c000, 0], [0x80107fff, 0x80010010, 0xedcc8001, 0],
        [0xc0000000, 0x80008000, 0x0218021b, 0], [0x00020001, 0x00027fff, 0x40000218, 0], [0x00ffc000, 0xffff0002, 0xfffefffe, 0],
        [0x00100000, 0x00ff000f, 0xfffe4000, 0], [0x021bfffe, 0x00ff8001, 0x00fffffe, 0], [0xedccedcc, 0x021b021b, 0x7ffe8001, 0],
        [0x00007ffe, 0x12340218, 0xfffe0001, 0], [0xc0000218, 0x0010ffff, 0xffff8001, 0], [0x400000ff, 0x7ffec000, 0x80008000, 0],
        [0x000f4000, 0xedcc0001, 0x80008000, 0], [0x7fff0218, 0x8010c000, 0x80108010, 0], [0x80008000, 0x80000218, 0x00007ffe, 0],
        [0x021bffff, 0x000fc000, 0x7fffedcc, 0], [0x02180002, 0x000f8000, 0x0001edcc, 0], [0x021b4000, 0x021b0001, 0x02188001, 0],
        [0x7fff8000, 0x40008010, 0xffff1234, 0], [0x80100000, 0x00010001, 0x80000218, 0], [0x4000c000, 0xffff021b, 0x40008010, 0],
        [0x7fff4000, 0x7ffe0002, 0x000f00ff, 0], [0x000fedcc, 0x8000fffe, 0x021b4000, 0], [0xedcc0001, 0x8000021b, 0xedcc8000, 0],
        [0x02180000, 0x00ffedcc, 0x12344000, 0], [0xc000ffff, 0x80010218, 0x7fffc000, 0],
    ];

    #[derive(Clone, Copy)]
    enum Kind {
        MaxI,
        MinI,
        MaxU,
        MinU,
        AddU,
        AddI,
        SubU,
        SubI,
        MulLo,
        MadI,
        MadU,
        Lshl,
        Ashr,
        Lshr,
    }
    use Kind::*;

    /// Source register (or -1: the literal 0x7fff), op_sel, op_sel_hi,
    /// neg_lo, neg_hi per source, clamp.
    struct Case(&'static str, Kind, [i32; 3], [bool; 3], [bool; 3], [bool; 3], [bool; 3], bool);
    const F: bool = false;
    const T: bool = true;
    const CASES: [Case; RESULTS] = [
        Case("v_pk_max_i16 lit neg", MaxI, [-1, 0, 0], [F; 3], [F, T, T], [T, F, F], [T, F, F], F),
        Case("v_pk_add_u16", AddU, [0, 1, 0], [F; 3], [T; 3], [F, T, F], [T, F, F], F),
        Case("v_pk_sub_i16", SubI, [0, 1, 0], [F; 3], [T; 3], [T, F, F], [F, T, F], F),
        Case("v_pk_min_i16", MinI, [0, 1, 0], [F; 3], [T; 3], [T, T, F], [T, T, F], F),
        Case("v_pk_max_u16", MaxU, [0, 1, 0], [F; 3], [T; 3], [F, T, F], [F, T, F], F),
        Case("v_pk_min_u16", MinU, [0, 1, 0], [F; 3], [T; 3], [T, F, F], [T, F, F], F),
        Case("v_pk_mul_lo_u16", MulLo, [0, 1, 0], [F; 3], [T; 3], [T, F, F], [F, T, F], F),
        Case("v_pk_mad_i16", MadI, [0, 1, 2], [F; 3], [T; 3], [F, F, T], [T, F, T], F),
        Case("v_pk_mad_u16", MadU, [0, 1, 2], [F; 3], [T; 3], [T, T, F], [F, T, T], F),
        Case("v_pk_lshlrev_b16", Lshl, [0, 1, 0], [F; 3], [T; 3], [F, T, F], [F, T, F], F),
        Case("v_pk_ashrrev_i16", Ashr, [0, 1, 0], [F; 3], [T; 3], [F, T, F], [T, T, F], F),
        Case("v_pk_lshrrev_b16", Lshr, [0, 1, 0], [F; 3], [T; 3], [T, T, F], [F, T, F], F),
        Case("v_pk_add_i16 op_sel", AddI, [0, 1, 0], [T, F, F], [F, T, T], [T, F, F], [F, T, F], F),
        Case("v_pk_sub_u16 op_sel", SubU, [0, 1, 0], [F, T, F], [T, F, T], [T, T, F], [T, T, F], F),
        Case("v_pk_add_i16 clamp", AddI, [0, 1, 0], [F; 3], [T; 3], [F, T, F], [F, T, F], T),
        Case("v_pk_max_i16 op_sel", MaxI, [2, 0, 0], [T, F, F], [F, T, T], [T, F, F], [T, F, F], F),
    ];

    fn signed(h: u32) -> i64 {
        h as u16 as i16 as i64
    }

    fn source(c: &Case, row: &[u32; 4], i: usize, high_lane: bool) -> u32 {
        let word = if c.2[i] < 0 { 0x7fff } else { row[c.2[i] as usize] };
        let high = if high_lane { c.4[i] } else { c.3[i] };
        let negate = if high_lane { c.6[i] } else { c.5[i] };
        ((if high { word >> 16 } else { word }) & 0xffff) ^ if negate { 0x8000 } else { 0 }
    }

    fn lane(c: &Case, row: &[u32; 4], high: bool) -> u32 {
        let (a, b, s) = (source(c, row, 0, high), source(c, row, 1, high), source(c, row, 2, high));
        let mut v: i64 = match c.1 {
            MaxI => signed(a).max(signed(b)),
            MinI => signed(a).min(signed(b)),
            MaxU => a.max(b) as i64,
            MinU => a.min(b) as i64,
            AddU => (a + b) as i64,
            AddI => signed(a) + signed(b),
            SubU => a as i64 - b as i64,
            SubI => signed(a) - signed(b),
            MulLo => a as i64 * b as i64,
            MadI => signed(a) * signed(b) + signed(s),
            MadU => a as i64 * b as i64 + s as i64,
            Lshl => (b as i64) << (a & 15),
            Ashr => signed(b) >> (a & 15),
            Lshr => (b >> (a & 15)) as i64,
        };
        if c.7 {
            v = v.clamp(-32768, 32767);
        }
        v as u32 & 0xffff
    }

    #[test]
    fn packed_integer_negate() {
        let input: Vec<u32> = ROWS.iter().flatten().copied().collect();
        let out = dispatch(&CODE, 32, 32, &input, 32 * RESULTS, true).unwrap();
        let mut failures = Vec::new();
        for (tid, row) in ROWS.iter().enumerate() {
            for (i, c) in CASES.iter().enumerate() {
                let want = lane(c, row, false) | lane(c, row, true) << 16;
                let got = out[tid * RESULTS + i];
                if got != want {
                    failures.push(format!("lane {tid} {}: {got:#010x}, expected {want:#010x}", c.0));
                }
            }
        }
        assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
    }
}
