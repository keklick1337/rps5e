// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! VOP3 integer semantics against hardware-measured reference rows:
//! op_sel forms, carry-out, saturating/clamped forms, SAD/QSAD, lerp,
//! 64-bit mad, pack/ldexp/pknorm.

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
        let run = ComputeRun::new(code.to_vec(), wave, lanes, user.clone());
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

/// The table layout: one row of 4 input dwords per lane, `results` output
/// dwords per lane, raw descriptors, one wave32. Returns the per-column
/// failures, leaving `skip` columns to their own (ignored) tests and
/// checking only `only` when given.
fn table<const N: usize>(code: &[u32], rows: &[[u32; 4]], results: usize, expected: &[[u32; N]], names: &[&str], skip: &[usize], only: Option<&[usize]>) -> Vec<String> {
    let input: Vec<u32> = rows.iter().flatten().copied().collect();
    let out = match dispatch(code, 32, rows.len() as u32, &input, rows.len() * results, true) {
        Ok(o) => o,
        Err(e) => return vec![e],
    };
    let mut fails = Vec::new();
    for (lane, row) in expected.iter().enumerate() {
        for (col, &e) in row.iter().enumerate() {
            if skip.contains(&col) || only.is_some_and(|o| !o.contains(&col)) {
                continue;
            }
            let a = out[lane * results + col];
            if a != e {
                let r = rows[lane];
                fails.push(format!("lane {lane} {} ({:#010x}, {:#010x}, {:#010x}, {:#010x}) is {a:#010x}, expected {e:#010x}", names[col], r[0], r[1], r[2], r[3]));
            }
        }
    }
    fails
}

fn assert_clean(what: &str, fails: &[String]) {
    assert!(fails.is_empty(), "{what}: {} failures:\n{}", fails.len(), fails.join("\n"));
}

/// The reference tests' input generator (64-bit LCG, high word).
fn lcg_inputs(seed: u64, threads: usize, inputs: usize, per_lane: usize, edges: &[&[u32]]) -> Vec<u32> {
    let mut state = seed;
    let mut words = vec![0u32; threads * inputs];
    for tid in 0..threads {
        for j in 0..per_lane {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            words[tid * inputs + j] = if tid < edges.len() { edges[tid][j] } else { (state >> 32) as u32 };
        }
    }
    words
}

macro_rules! table_tests {
    ($what:literal, $file:literal, [$($ignored:ident = [$($col:literal),*]: $reason:literal),*]) => {
        include!($file);
        const SKIP: &[usize] = &[$($($col,)*)*];

        #[test]
        fn rows() {
            assert_clean($what, &table(&CODE, &ROWS, RESULTS as usize, &EXPECTED, &NAMES, SKIP, None));
        }

        $(
            #[test]
            fn $ignored() {
                assert_clean($what, &table(&CODE, &ROWS, RESULTS as usize, &EXPECTED, &NAMES, &[], Some(&[$($col),*])));
            }
        )*
    };
}

mod vop3_align_op_sel {
    use super::*;
    table_tests!("vop3 align op sel", "vectors/valu_vop3_align_op_sel.rs", [
        op_sel_shift_byte_and_high_destination = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]: "op_sel on v_alignbit/v_alignbyte is ignored (src2 byte select, dst high half)"
    ]);
}

mod vop3_integer16_ternary_op_sel {
    use super::*;
    table_tests!("vop3 integer16 ternary op sel", "vectors/valu_vop3_integer16_ternary_op_sel.rs", []);
}

mod vop3_integer_alu {
    use super::*;
    table_tests!("vop3 integer alu", "vectors/valu_vop3_integer_alu.rs", []);
}

mod vop3_integer_clamp {
    use super::*;
    table_tests!("vop3 integer clamp", "vectors/valu_vop3_integer_clamp.rs", []);
}

mod vop3_integer_clamp_wide {
    use super::*;
    table_tests!("vop3 integer clamp wide", "vectors/valu_vop3_integer_clamp_wide.rs", []);
}

mod vop3_mad_clamp {
    use super::*;
    table_tests!("vop3 mad clamp", "vectors/valu_vop3_mad_clamp.rs", []);
}

mod vop3_mad_i64 {
    use super::*;
    table_tests!("vop3 mad i64", "vectors/valu_vop3_mad_i64.rs", []);
}

mod vop3_quad_sad {
    use super::*;
    table_tests!("vop3 quad sad", "vectors/valu_vop3_quad_sad.rs", []);
}

mod vop3_pack_ldexp {
    use super::*;
    table_tests!("vop3 pack ldexp", "vectors/valu_vop3_pack_ldexp.rs", []);
}

mod vop3_carry_out {
    use super::*;
    include!("vectors/valu_vop3_carry_out.rs");

    fn add(l: u32, r: u32) -> [u32; 2] {
        let s = l as u64 + r as u64;
        [s as u32, (s >> 32) as u32]
    }

    fn sub(l: u32, r: u32) -> [u32; 2] {
        [l.wrapping_sub(r), (r > l) as u32]
    }

    const NAMES: [&str; 8] = [
        "v_add_co_u32 into an sgpr",
        "v_sub_co_u32 into an sgpr",
        "v_subrev_co_u32 into an sgpr",
        "v_add_co_u32 into vcc",
        "v_sub_co_u32 into vcc",
        "v_subrev_co_u32 into vcc",
        "v_sub_co_u32 from a literal",
        "v_subrev_co_u32 of an inline constant",
    ];

    fn check(wave: u32) {
        let edges: Vec<&[u32]> = EDGES.iter().map(|e| &e[..]).collect();
        let input = lcg_inputs(0x9e3779b97f4a7c15, THREADS as usize, INPUTS as usize, 2, &edges);
        let out = dispatch(&CARRY_OUT_CODE, wave, THREADS, &input, (THREADS * RESULTS) as usize, false).unwrap();
        let mut fails = Vec::new();
        for tid in 0..THREADS as usize {
            let a = input[tid * INPUTS as usize];
            let b = input[tid * INPUTS as usize + 1];
            let edge = tid < EDGES.len();
            let expected = [
                if edge { [EDGES[tid][2], EDGES[tid][3]] } else { add(a, b) },
                if edge { [EDGES[tid][4], EDGES[tid][5]] } else { sub(a, b) },
                if edge { [EDGES[tid][6], EDGES[tid][7]] } else { sub(b, a) },
                add(a, b),
                sub(a, b),
                sub(b, a),
                sub(LITERAL, a),
                sub(b, 1),
            ];
            for (j, e) in expected.iter().enumerate() {
                let value = out[tid * RESULTS as usize + j * 2];
                let carry = out[tid * RESULTS as usize + j * 2 + 1];
                if value != e[0] || carry != e[1] {
                    fails.push(format!("wave{wave} {}: thread {tid} ({a:#x}, {b:#x}) is {value:#x} carry {carry}, expected {:#x} carry {}", NAMES[j], e[0], e[1]));
                }
            }
        }
        assert_clean("vop3 carry out", &fails);
    }

    #[test]
    fn wave32() {
        check(32);
    }

    #[test]
    fn wave64() {
        check(64);
    }
}

mod vop3_integer {
    use super::*;
    include!("vectors/valu_vop3_integer.rs");

    fn saturate(v: i64) -> u32 {
        v.clamp(i32::MIN as i64, i32::MAX as i64) as i32 as u32
    }

    fn s16(v: u32) -> u32 {
        v as u16 as i16 as i32 as u32
    }

    fn failures() -> Vec<String> {
        let edges: Vec<&[u32]> = EDGES.iter().map(|e| &e[..]).collect();
        let input = lcg_inputs(0x2545f4914f6cdd1d, THREADS as usize, INPUTS as usize, 3, &edges);
        let out = match dispatch(&INTEGER_CODE, 32, THREADS, &input, (THREADS * RESULTS) as usize, false) {
            Ok(o) => o,
            Err(e) => return vec![e],
        };
        let mut fails = Vec::new();
        for tid in 0..THREADS as usize {
            let w = &input[tid * INPUTS as usize..];
            let (a, b, c) = (w[0], w[1], w[2]);
            let (al, ah, bl, bh) = (a & 0xFFFF, a >> 16, b & 0xFFFF, b >> 16);
            let (sa, sb) = (a as i32 as i64, b as i32 as i64);
            let shifted = ((a as u64 | ((b as u64) << 32)) as i64 >> (c & 63)) as u64;
            let expected: [u32; 14] = [
                a.wrapping_add(b),
                saturate(sa + sb),
                saturate(sa - sb),
                al.wrapping_mul(bl) & 0xFFFF,
                (al.wrapping_mul(bl).wrapping_add(c & 0xFFFF)) & 0xFFFF,
                (s16(a).wrapping_mul(s16(b)).wrapping_add(s16(c))) & 0xFFFF,
                al.wrapping_mul(bl).wrapping_add(c),
                s16(a).wrapping_mul(s16(b)).wrapping_add(c),
                shifted as u32,
                (shifted >> 32) as u32,
                ah.wrapping_mul(bh).wrapping_add(c),
                (ah.wrapping_mul(bl).wrapping_add(c >> 16)) & 0xFFFF,
                al.wrapping_mul(bh) & 0xFFFF,
                a.wrapping_sub(b),
            ];
            let half_results: u32 = (1 << 3) | (1 << 4) | (1 << 5) | (1 << 11) | (1 << 12);
            for (j, &e) in expected.iter().enumerate() {
                let mask = if (half_results >> j) & 1 != 0 { 0xFFFF } else { 0xFFFF_FFFF };
                let actual = out[tid * RESULTS as usize + j] & mask;
                if actual != e {
                    fails.push(format!("thread {tid} ({a:#x}, {b:#x}, {c:#x}) result {j} is {actual:#x}, expected {e:#x}"));
                }
            }
        }
        fails
    }

    #[test]
    fn results() {
        assert_clean("vop3 integer", &failures());
    }
}

mod vop3_lerp {
    use super::*;
    include!("vectors/valu_vop3_lerp.rs");

    fn lerp(l: u32, r: u32, rounding: u32) -> u32 {
        let mut result = 0;
        for offset in (0..32).step_by(8) {
            let sum = ((l >> offset) & 0xFF) + ((r >> offset) & 0xFF) + ((rounding >> offset) & 1);
            result |= (sum >> 1) << offset;
        }
        result
    }

    const NAMES: [&str; 8] = [
        "v_lerp_u8",
        "v_lerp_u8 with swapped sources",
        "v_lerp_u8 truncating",
        "v_lerp_u8 rounding every byte",
        "v_lerp_u8 with a literal round mode",
        "v_lerp_u8 with an inline constant",
        "v_lerp_u8 of a value with itself",
        "v_lerp_u8 into its first source",
    ];

    #[test]
    fn results() {
        let edges: Vec<&[u32]> = EDGES.iter().map(|e| &e[..]).collect();
        let input = lcg_inputs(0xd1b54a32d192ed03, THREADS as usize, INPUTS as usize, 3, &edges);
        let out = dispatch(&LERP_CODE, 32, THREADS, &input, (THREADS * RESULTS) as usize, false).unwrap();
        let mut fails = Vec::new();
        for tid in 0..THREADS as usize {
            let w = &input[tid * INPUTS as usize..];
            let (a, b, c) = (w[0], w[1], w[2]);
            let expected = [
                if tid < EDGES.len() { EDGES[tid][3] } else { lerp(a, b, c) },
                lerp(a, b, c),
                lerp(a, b, 0),
                lerp(a, b, 0xFFFF_FFFF),
                lerp(a, b, LITERAL),
                lerp(0xFFFF_FFFF, b, c),
                a,
                lerp(a, b, c),
            ];
            for (j, &e) in expected.iter().enumerate() {
                let actual = out[tid * RESULTS as usize + j];
                if actual != e {
                    fails.push(format!("{}: thread {tid} ({a:#x}, {b:#x}, {c:#x}) is {actual:#x}, expected {e:#x}", NAMES[j]));
                }
            }
        }
        assert_clean("vop3 lerp", &fails);
    }
}

mod vop3_sad {
    use super::*;
    include!("vectors/valu_vop3_sad.rs");

    fn sad(l: u32, r: u32, field_bits: u32, masked: bool) -> u32 {
        let field_mask = ((1u64 << field_bits) - 1) as u32;
        let mut sum = 0u32;
        let mut offset = 0;
        while offset < 32 {
            let left = (l >> offset) & field_mask;
            let right = (r >> offset) & field_mask;
            offset += field_bits;
            if masked && right == 0 {
                continue;
            }
            sum = sum.wrapping_add(left.abs_diff(right));
        }
        sum
    }

    const NAMES: [&str; 10] = [
        "v_sad_u8",
        "v_sad_hi_u8",
        "v_sad_u16",
        "v_msad_u8",
        "v_sad_u32",
        "v_sad_u8 with a zero reference",
        "v_sad_u8 with an inline constant",
        "v_sad_hi_u8 accumulating into a v_sad_u8 result",
        "v_sad_u16 with a literal",
        "v_msad_u8 with swapped sources",
    ];

    #[test]
    fn results() {
        let edges: Vec<&[u32]> = EDGES.iter().map(|e| &e[..]).collect();
        let mut input = lcg_inputs(0x9e3779b97f4a7c15, THREADS as usize, INPUTS as usize, 3, &edges);
        for tid in EDGES.len()..THREADS as usize {
            if tid & 1 != 0 {
                input[tid * INPUTS as usize + 1] &= if tid & 2 != 0 { 0x00FF_FF00 } else { 0xFF00_00FF };
            }
        }
        let out = dispatch(&SAD_CODE, 32, THREADS, &input, (THREADS * RESULTS) as usize, false).unwrap();
        let mut fails = Vec::new();
        for tid in 0..THREADS as usize {
            let w = &input[tid * INPUTS as usize..];
            let (a, b, c) = (w[0], w[1], w[2]);
            let bytes = sad(a, b, 8, false).wrapping_add(c);
            let expected = [
                bytes,
                (sad(a, b, 8, false) << 16).wrapping_add(c),
                sad(a, b, 16, false).wrapping_add(c),
                sad(a, b, 8, true).wrapping_add(c),
                a.abs_diff(b).wrapping_add(c),
                sad(a, 0, 8, false).wrapping_add(c),
                sad(0xFFFF_FFFF, b, 8, false).wrapping_add(c),
                (sad(b, a, 8, false) << 16).wrapping_add(bytes),
                sad(a, LITERAL, 16, false).wrapping_add(c),
                sad(b, a, 8, true).wrapping_add(c),
            ];
            for (j, &e) in expected.iter().enumerate() {
                let actual = out[tid * RESULTS as usize + j];
                if actual != e {
                    fails.push(format!("{}: thread {tid} ({a:#x}, {b:#x}, {c:#x}) is {actual:#x}, expected {e:#x}", NAMES[j]));
                }
            }
        }
        assert_clean("vop3 sad", &fails);
    }
}
