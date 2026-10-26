// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Conversion, compare and dot-product rows measured on the hardware or
//! modelled bit-exactly by the reference execution tests (F16PackTowardZero,
//! F32CompareConstant, F32PackRounding, IntegerToF32Rounding, HalfExpansion,
//! LdexpF32Denormal, F32Dot2Rounding). Every program is the reference's own
//! code: s[0:3] input V#, s[4:7] output V#, v0 the thread id and, for the
//! multi-group dispatches, s8 the work-group id.

include!("vectors/fmod_common.rs");

use ps5shader::translate::FloatMode;

const fn mode(mode: u8, dx10_clamp: bool, ieee: bool) -> FloatMode {
    FloatMode { mode, dx10_clamp, ieee, fp16_overflow: false }
}

/// Dispatch `groups` work-groups of `threads` wave32 invocations each, the
/// work-group id in s8, raw input/output buffers. Optimized pipeline; the
/// first group is also checked against the unoptimized one.
fn run_groups(code: &[u32], threads: u32, groups: u32, input: &[u32], out_words: usize, float_mode: FloatMode) -> Vec<u32> {
    run_groups_rules(code, threads, groups, input, out_words, float_mode, ps5shader::translate::float_rules::ALL)
}

fn run_groups_rules(code: &[u32], threads: u32, groups: u32, input: &[u32], out_words: usize, float_mode: FloatMode, rules: u32) -> Vec<u32> {
    let mut user = vec![0u32; 8];
    user[..4].copy_from_slice(&fmod_descriptor(FMOD_INPUT_BASE, input.len(), 0));
    user[4..].copy_from_slice(&fmod_descriptor(FMOD_OUTPUT_BASE, out_words, 0));
    let mut run = ComputeRun::new(code.to_vec(), 32, threads.min(32), user);
    run.float_mode = float_mode;
    run.workgroup_id_sgprs = 1;
    run.float_rules = rules;
    let functions = [run.compile(true).unwrap_or_else(|e| panic!("{e}")), run.compile(false).unwrap_or_else(|e| panic!("{e}"))];
    let execute = |f: &ps5shader::ssa::Function, groups: std::ops::Range<u32>| {
        let mut memory = ps5shader::interp::Memory::default();
        memory.write_words(FMOD_INPUT_BASE, input);
        memory.write_words(FMOD_OUTPUT_BASE, &vec![0xDEAD_BEEF; out_words]);
        memory.write_words(run.code_address, &run.code);
        for group in groups {
            let mut first = 0;
            while first < threads {
                let lanes = 32.min(threads - first);
                let mut m = ps5shader::interp::Machine::new(32, lanes);
                m.user_data = run.user_data.clone();
                m.memory = std::mem::take(&mut memory);
                m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(0), (first..first + lanes).collect());
                m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(1), vec![0; lanes as usize]);
                m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(2), vec![0; lanes as usize]);
                m.system.insert(ps5shader::ssa::SystemValue::WorkgroupId(0), vec![group; lanes as usize]);
                m.run(f).unwrap_or_else(|e| panic!("run: {e}"));
                memory = std::mem::take(&mut m.memory);
                first += lanes;
            }
        }
        memory.read_words(FMOD_OUTPUT_BASE, out_words)
    };
    let out = execute(&functions[0], 0..groups);
    let plain = execute(&functions[1], 0..1);
    let per_group = out_words / groups as usize;
    if let Some(i) = plain[..per_group].iter().zip(&out[..per_group]).position(|(a, b)| a != b) {
        panic!("optimized run differs at word {i}: {:#010x} vs {:#010x}", plain[i], out[i]);
    }
    out
}

fn half_to_f32_bits(half: u32) -> u32 {
    let exponent = (half >> 10) & 0x1f;
    let mantissa = half & 0x3ff;
    if exponent != 0 {
        return ((exponent + 112) << 23) | (mantissa << 13);
    }
    if mantissa == 0 {
        return 0;
    }
    let mut shift = 0;
    while (mantissa << shift) & 0x400 == 0 {
        shift += 1;
    }
    ((113 - shift) << 23) | (((mantissa << shift) & 0x3ff) << 13)
}

// ---- F16PackTowardZero ----

fn toward_zero(mut bits: u32) -> u32 {
    if bits & 0x7f80_0000 == 0 {
        bits &= 0x8000_0000;
    }
    let sign = (bits >> 16) & 0x8000;
    let exponent = (bits >> 23) & 0xff;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return if mantissa == 0 { sign | 0x7c00 } else { sign | 0x7e00 | (mantissa >> 13) };
    }
    if exponent >= 143 {
        return sign | 0x7bff;
    }
    if exponent >= 113 {
        return sign | ((exponent - 112) << 10) | (mantissa >> 13);
    }
    if exponent >= 103 {
        return sign | ((mantissa | 0x80_0000) >> (126 - exponent));
    }
    sign
}

/// v_cvt_pkrtz_f16_f32 v5, v4, -v4 in FLOAT_MODE 0xC0: f32 denormal inputs
/// flush, values past the largest finite half clamp to it (round toward
/// zero never reaches infinity), NaNs stay quiet with their top payload.
#[test]
fn f16_pack_toward_zero() {
    let mut inputs: Vec<u32> = vec![
        0x00000000, 0x00000001, 0x007fffff, 0x00800000, 0x33000000, 0x33000001, 0x337fffff, 0x33800000, 0x387fffff, 0x38800000, 0x477fe000,
        0x477fefff, 0x477ff000, 0x477fffff, 0x47800000, 0x7f7fffff, 0x7f800000, 0x7fc00000, 0x7f800001, 0x7fffe000, 0x7fbfffff, 0x7f802000,
    ];
    for half in 0..0x7c00u32 {
        let exact = half_to_f32_bits(half);
        let next = half_to_f32_bits(half + 1);
        let middle = exact + (next - exact) / 2;
        inputs.extend([exact, exact + 1, middle - 1, middle, middle + 1]);
    }
    while inputs.len() % 32 != 0 {
        inputs.push(0);
    }
    let code = [0x34020082, 0x8f098708, 0x4a020209, 0xe0301000, 0x80000401, 0xbf8c3f70, 0xd52f0005, 0x40020904, 0xe0701000, 0x80010501, 0xbf810000];
    let out = run_groups(&code, 32, (inputs.len() / 32) as u32, &inputs, inputs.len(), mode(0xC0, true, false));
    let mut failures = Vec::new();
    for (i, &source) in inputs.iter().enumerate() {
        let expected = toward_zero(source) | (toward_zero(source ^ 0x8000_0000) << 16);
        if out[i] != expected {
            failures.push(format!("v_cvt_pkrtz_f16_f32 of {source:#010x}, -{source:#010x} is {:#010x}, expected {expected:#010x}", out[i]));
        }
    }
    fmod_report(&failures);
}

// ---- F32CompareConstant ----

const COMPARE_INPUT: [u32; 32] = [
    0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x00800001, 0x80800001, 0x3f000000,
    0xbf000000, 0x3effffff, 0x3f000001, 0xc0800000, 0xc07fffff, 0xc0800001, 0x3f800000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00001,
    0x7f800001, 0x7f7fffff, 0xff7fffff, 0x00400000, 0x80400000, 0x0000ffff, 0x3e800000, 0xbe800000, 0x80000000, 0x01000000,
];
/// (bits, inline source operand; 0xff = literal)
const COMPARE_CONSTANTS: [(u32, u32); 9] = [
    (0x00000000, 0x80),
    (0x80000000, 0xff),
    (0x00000001, 0xff),
    (0x807fffff, 0xff),
    (0x3f000000, 0xf0),
    (0xc0800000, 0xf7),
    (0x00800000, 0xff),
    (0x7f800000, 0xff),
    (0x7fc00000, 0xff),
];
const COMPARE_PREDICATES: [u32; 12] = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e];
const COMPARE_WORDS: u32 = 8;

fn compare_program(register_operand: bool) -> Vec<u32> {
    let mut code = vec![0x34020082, 0x34060085, 0xe0301000, 0x80000401, 0xbf8c3f70, 0x7e2c0281];
    for word in 0..COMPARE_WORDS {
        code.push(0x7e000280 | ((30 + word) << 17));
    }
    let mut bit = 0u32;
    for &(bits, inline) in &COMPARE_CONSTANTS {
        if register_operand {
            code.extend([0x7e2a02ff, bits]);
        }
        let source = if register_operand { 256 + 21 } else { inline };
        let literal = !register_operand && source == 0xff;
        for &predicate in &COMPARE_PREDICATES {
            for constant_first in [true, false] {
                if constant_first {
                    code.push(0x7c000000 | (predicate << 17) | (4 << 9) | source);
                } else {
                    code.push(0xd4000000 | (predicate << 16) | 106);
                    code.push((source << 9) | (256 + 4));
                }
                if literal {
                    code.push(bits);
                }
                let accumulator = 30 + bit / 32;
                code.push((0x01 << 25) | (20 << 17) | (22 << 9) | 0x80);
                code.push(0xd4000000 | (0x36f << 16) | accumulator);
                code.push(((256 + accumulator) << 18) | ((0x80 + bit % 32) << 9) | (256 + 20));
                bit += 1;
            }
        }
    }
    for word in 0..COMPARE_WORDS {
        code.extend([0xe0701000 | (word * 4), 0x80010003 | ((30 + word) << 8)]);
    }
    code.push(0xbf810000);
    code
}

fn compare_f32(predicate: u32, l: u32, r: u32) -> bool {
    let (l, r) = (f32::from_bits(l), f32::from_bits(r));
    match predicate {
        0x01 => l < r,
        0x02 => l == r,
        0x03 => l <= r,
        0x04 => l > r,
        0x05 => l < r || l > r,
        0x06 => l >= r,
        0x09 => !(l >= r),
        0x0a => !(l < r || l > r),
        0x0b => !(l > r),
        0x0c => !(l <= r),
        0x0d => l != r,
        0x0e => !(l < r),
        _ => unreachable!(),
    }
}

/// v_cmp_*_f32 against inline, literal and register constants: when f32
/// denormals are flushed the constant operand is flushed like the register
/// one (a literal 0x00000001 equals zero).
#[test]
fn f32_compare_constant_flushes_denormal_constants() {
    let flush = |b: u32| if b & 0x7f80_0000 == 0 { b & 0x8000_0000 } else { b };
    use ps5shader::translate::float_rules::{ALL, COMPARE_CONSTANT_FOLD};
    let mut failures = Vec::new();
    for (name, float_mode, rules) in [
        ("IEEE=0 f32 denormals flushed", mode(0xC0, true, false), ALL),
        ("IEEE=1 all denormals flushed", mode(0x00, false, true), ALL),
        ("IEEE=0 f32 denormals flushed, flush not folded", mode(0xC0, true, false), ALL & !COMPARE_CONSTANT_FOLD),
    ] {
        for register_operand in [true, false] {
            let code = compare_program(register_operand);
            let out = run_groups_rules(&code, 32, 1, &COMPARE_INPUT, 32 * COMPARE_WORDS as usize, float_mode, rules);
            for (tid, &input) in COMPARE_INPUT.iter().enumerate() {
                let value = flush(input);
                let mut bit = 0u32;
                for &(bits, _) in &COMPARE_CONSTANTS {
                    let constant = flush(bits);
                    for &predicate in &COMPARE_PREDICATES {
                        for constant_first in [true, false] {
                            let expected = if constant_first { compare_f32(predicate, constant, value) } else { compare_f32(predicate, value, constant) };
                            let actual = (out[tid * COMPARE_WORDS as usize + (bit / 32) as usize] >> (bit % 32)) & 1 != 0;
                            if actual != expected {
                                let operand = if register_operand { "register" } else { "constant" };
                                failures.push(format!("{name}, {operand} operand, lane {tid}: predicate {predicate:#x} constant {bits:#010x} first={constant_first} value {input:#010x} is {actual}, expected {expected}"));
                            }
                            bit += 1;
                        }
                    }
                }
            }
        }
    }
    fmod_report(&failures);
}

/// With f32 denormals flushed, a compare against a constant is one compare
/// of the unflushed register (no flush selects): fewer instructions than
/// with the fold off, for normal and zero constants on either side.
#[test]
fn f32_compare_constant_folds_the_flush() {
    use ps5shader::translate::float_rules::{ALL, COMPARE_CONSTANT_FOLD};
    // v_cmp_gt_f32 vcc, 0.5, v4 / v_cmp_lt_f32 vcc, 0, v4 (VOPC) and
    // v_cmp_nge_f32 vcc_lo, v4, 0 (VOP3), each stored as 0/1.
    for compare in [vec![0x7c0808f0u32], vec![0x7c020880], vec![0xd409006a, 0x00010104]] {
        let mut code = vec![0x34020082, 0xe0301000, 0x80000401, 0xbf8c3f70];
        code.extend(&compare);
        // v_mov_b32 v22, 1; v_cndmask_b32 v20, 0, v22; store v20.
        code.extend([0x7e2c0281, 0x02282c80, 0xe0701000, 0x80011401, 0xbf810000]);
        let count = |rules: u32| {
            let mut user = vec![0u32; 8];
            user[..4].copy_from_slice(&fmod_descriptor(FMOD_INPUT_BASE, 32, 0));
            user[4..].copy_from_slice(&fmod_descriptor(FMOD_OUTPUT_BASE, 32, 0));
            let mut run = ComputeRun::new(code.clone(), 32, 32, user);
            run.float_rules = rules;
            let f = run.compile(true).unwrap_or_else(|e| panic!("{e}"));
            let out_on = fmod_run_waves(&run, true, 32, &COMPARE_INPUT, 32).unwrap();
            (f.instruction_count(), out_on)
        };
        let (folded, a) = count(ALL);
        let (plain, b) = count(ALL & !COMPARE_CONSTANT_FOLD);
        assert_eq!(a, b, "{compare:08x?}: the folded compare disagrees");
        assert!(folded < plain, "{compare:08x?}: folded {folded} instructions, not fewer than {plain}");
    }
}

/// `!(+inf < x)` and `!(-inf > x)` (v_cmp_nlt / v_cmp_ngt with the
/// infinity in src0) hold for every x, NaN included: the compare folds to
/// true and leaves no float compare in the program.
#[test]
fn unordered_infinity_compares_fold_to_true() {
    for compare in [[0x7c1c_08ffu32, 0x7f80_0000], [0x7c16_08ff, 0xff80_0000]] {
        let mut code = vec![0x34020082, 0xe0301000, 0x80000401, 0xbf8c3f70];
        code.extend(compare);
        code.extend([0x7e2c0281, 0x02282c80, 0xe0701000, 0x80011401, 0xbf810000]);
        let mut user = vec![0u32; 8];
        user[..4].copy_from_slice(&fmod_descriptor(FMOD_INPUT_BASE, 32, 0));
        user[4..].copy_from_slice(&fmod_descriptor(FMOD_OUTPUT_BASE, 32, 0));
        let run = ComputeRun::new(code.clone(), 32, 32, user);
        let f = run.compile(true).unwrap_or_else(|e| panic!("{e}"));
        let compares = f.order.iter().flat_map(|b| f.blocks[b.index()].insts.iter()).filter(|&&v| matches!(f.value(v).op, ps5shader::ssa::Op::FCmp(_))).count();
        assert_eq!(compares, 0, "{compare:08x?}: the compare was not folded");
        let out = fmod_run_waves(&run, true, 32, &COMPARE_INPUT, 32).unwrap();
        assert!(out.iter().all(|&w| w == 1), "{compare:08x?}: {out:08x?}");
    }
}

// ---- F32PackRounding ----

fn is_nan(bits: u32) -> bool {
    bits & 0x7fff_ffff > 0x7f80_0000
}

fn flushed(bits: u32) -> u32 {
    if bits & 0x7f80_0000 == 0 {
        bits & 0x8000_0000
    } else {
        bits
    }
}

fn pack_norm(source: u32, signed: bool) -> u32 {
    let bits = flushed(source);
    if is_nan(bits) {
        return 0;
    }
    let scale = if signed { 32767.0 } else { 65535.0 };
    let negative = bits & 0x8000_0000 != 0;
    let magnitude = (f32::from_bits(bits) as f64).abs();
    let value = if magnitude >= 1.0 { scale as u32 } else { (magnitude * scale).round_ties_even() as u32 };
    if !negative {
        return value;
    }
    if signed {
        0u32.wrapping_sub(value) & 0xffff
    } else {
        0
    }
}

fn pack_u8(source: u32) -> u32 {
    let value = f32::from_bits(flushed(source));
    if is_nan(source) || !(value > 0.0) {
        return 0;
    }
    if value >= 255.0 {
        return 255;
    }
    (value as f64).round_ties_even() as u32
}

/// v_cvt_pknorm_u16_f32 / v_cvt_pknorm_i16_f32 / v_cvt_pk_u8_f32 in FLOAT_MODE
/// 0xC0: the scaled value rounds once to nearest even (no f32 rounding of
/// the product), negative signed values are the negated magnitude, f32
/// denormals flush, NaN gives 0.
#[test]
fn f32_pack_rounding() {
    let mut inputs: Vec<u32> = vec![
        0x00000000, 0x80000000, 0x00000001, 0x807fffff, 0x00800000, 0x3f000000, 0xbf000000, 0x3f7fffff, 0xbf7fffff, 0x3f800000, 0xbf800000,
        0x3f800001, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00001, 0x7f800001, 0x437f0000, 0x437f8000, 0x43800000, 0x7f7fffff, 0x37800000,
        0x377fffff, 0x38000000,
    ];
    let mut around = |exact: f64| {
        let nearest = (exact as f32).to_bits();
        for delta in [-2i32, -1, 0, 1, 2] {
            inputs.push(nearest.wrapping_add(delta as u32));
        }
    };
    for scale in [65535.0f64, 32767.0] {
        for k in 0..(scale as u32 - 1) {
            around((k as f64 + 0.5) / scale);
        }
    }
    for k in 0..256u32 {
        around(k as f64 + 0.5);
    }
    let positive = inputs.len();
    for index in (0..positive).step_by(7) {
        inputs.push(inputs[index] ^ 0x8000_0000);
    }
    while inputs.len() % 32 != 0 {
        inputs.push(0);
    }
    let code = [
        0x34020082, 0x8f098708, 0x4a020209, 0xe0301000, 0x80000401, 0x34060282, 0xbf8c3f70, 0xd7690005, 0x00020904, 0xd7680006, 0x00020904,
        0xd55e0007, 0x02010104, 0xe0781000, 0x80010503, 0xbf810000,
    ];
    use ps5shader::translate::float_rules::{ALL, PKNORM_F32};
    let mut failures = Vec::new();
    for (path, rules) in [("f32", ALL), ("integer", ALL & !PKNORM_F32)] {
        let out = run_groups_rules(&code, 32, (inputs.len() / 32) as u32, &inputs, inputs.len() * 4, mode(0xC0, true, false), rules);
        for (i, &source) in inputs.iter().enumerate() {
            let o = &out[i * 4..];
            let (u, s) = (pack_norm(source, false), pack_norm(source, true));
            for (name, actual, expected) in [("v_cvt_pknorm_u16_f32", o[0], u | (u << 16)), ("v_cvt_pknorm_i16_f32", o[1], s | (s << 16)), ("v_cvt_pk_u8_f32", o[2], pack_u8(source))] {
                if actual != expected {
                    failures.push(format!("{path} path: {name} of {source:#010x} is {actual:#010x}, expected {expected:#010x}"));
                }
            }
        }
    }
    fmod_report(&failures);
}

// ---- IntegerToF32Rounding ----

const INT_VALUES: [u32; 32] = [
    0x00000000, 0x00000001, 0x00000002, 0x00000003, 0x00000005, 0x00000008, 0x0000000f, 0x00000010, 0x00000018, 0x0000001f, 0x00000020,
    0x00000021, 0x0000003f, 0x00007fff, 0x00008000, 0x0000ffff, 0x00010000, 0x007fffff, 0x00800000, 0x00ffffff, 0x01000000, 0x12345678,
    0x7fffffff, 0x80000000, 0x80000001, 0xfffffffe, 0xffffffff, 0xdeadbeef, 0x3f800000, 0x00ff00ff, 0xffff8000, 0x0000807f,
];

/// v_cvt_f32_i32 / v_cvt_f32_u32 follow the f32 round mode (hardware rows,
/// round toward +inf, -inf and zero; columns i32 and u32).
#[test]
fn integer_to_f32_rounding() {
    #[rustfmt::skip]
    let expected: [[[u32; 2]; 32]; 3] = [
        [[0x00000000, 0x00000000], [0x3f800000, 0x3f800000], [0x40000000, 0x40000000], [0x40400000, 0x40400000],
         [0x40a00000, 0x40a00000], [0x41000000, 0x41000000], [0x41700000, 0x41700000], [0x41800000, 0x41800000],
         [0x41c00000, 0x41c00000], [0x41f80000, 0x41f80000], [0x42000000, 0x42000000], [0x42040000, 0x42040000],
         [0x427c0000, 0x427c0000], [0x46fffe00, 0x46fffe00], [0x47000000, 0x47000000], [0x477fff00, 0x477fff00],
         [0x47800000, 0x47800000], [0x4afffffe, 0x4afffffe], [0x4b000000, 0x4b000000], [0x4b7fffff, 0x4b7fffff],
         [0x4b800000, 0x4b800000], [0x4d91a2b4, 0x4d91a2b4], [0x4f000000, 0x4f000000], [0xcf000000, 0x4f000000],
         [0xceffffff, 0x4f000001], [0xc0000000, 0x4f800000], [0xbf800000, 0x4f800000], [0xce054904, 0x4f5eadbf],
         [0x4e7e0000, 0x4e7e0000], [0x4b7f00ff, 0x4b7f00ff], [0xc7000000, 0x4f7fff80], [0x47007f00, 0x47007f00]],
        [[0x00000000, 0x00000000], [0x3f800000, 0x3f800000], [0x40000000, 0x40000000], [0x40400000, 0x40400000],
         [0x40a00000, 0x40a00000], [0x41000000, 0x41000000], [0x41700000, 0x41700000], [0x41800000, 0x41800000],
         [0x41c00000, 0x41c00000], [0x41f80000, 0x41f80000], [0x42000000, 0x42000000], [0x42040000, 0x42040000],
         [0x427c0000, 0x427c0000], [0x46fffe00, 0x46fffe00], [0x47000000, 0x47000000], [0x477fff00, 0x477fff00],
         [0x47800000, 0x47800000], [0x4afffffe, 0x4afffffe], [0x4b000000, 0x4b000000], [0x4b7fffff, 0x4b7fffff],
         [0x4b800000, 0x4b800000], [0x4d91a2b3, 0x4d91a2b3], [0x4effffff, 0x4effffff], [0xcf000000, 0x4f000000],
         [0xcf000000, 0x4f000000], [0xc0000000, 0x4f7fffff], [0xbf800000, 0x4f7fffff], [0xce054905, 0x4f5eadbe],
         [0x4e7e0000, 0x4e7e0000], [0x4b7f00ff, 0x4b7f00ff], [0xc7000000, 0x4f7fff80], [0x47007f00, 0x47007f00]],
        [[0x00000000, 0x00000000], [0x3f800000, 0x3f800000], [0x40000000, 0x40000000], [0x40400000, 0x40400000],
         [0x40a00000, 0x40a00000], [0x41000000, 0x41000000], [0x41700000, 0x41700000], [0x41800000, 0x41800000],
         [0x41c00000, 0x41c00000], [0x41f80000, 0x41f80000], [0x42000000, 0x42000000], [0x42040000, 0x42040000],
         [0x427c0000, 0x427c0000], [0x46fffe00, 0x46fffe00], [0x47000000, 0x47000000], [0x477fff00, 0x477fff00],
         [0x47800000, 0x47800000], [0x4afffffe, 0x4afffffe], [0x4b000000, 0x4b000000], [0x4b7fffff, 0x4b7fffff],
         [0x4b800000, 0x4b800000], [0x4d91a2b3, 0x4d91a2b3], [0x4effffff, 0x4effffff], [0xcf000000, 0x4f000000],
         [0xceffffff, 0x4f000000], [0xc0000000, 0x4f7fffff], [0xbf800000, 0x4f7fffff], [0xce054904, 0x4f5eadbe],
         [0x4e7e0000, 0x4e7e0000], [0x4b7f00ff, 0x4b7f00ff], [0xc7000000, 0x4f7fff80], [0x47007f00, 0x47007f00]],
    ];
    let code = [0x34020084, 0x34040086, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x7e1402ff, 0x5a5a5a5a, 0x7e1602ff, 0x5a5a5a5a, 0x7e140b04, 0x7e160d05, 0xe0781000, 0x80010a02, 0xbf810000];
    let rows: Vec<[u32; 4]> = INT_VALUES.iter().map(|&v| [v, v, 0, 0]).collect();
    for rounding in 1..4u8 {
        fmod_set_float_mode(mode(0xF0 | (rounding << 2) | rounding, true, true));
        for column in 0..2 {
            let name = format!("round mode {rounding} {}", if column == 0 { "v_cvt_f32_i32" } else { "v_cvt_f32_u32" });
            fmod_check_rows(&code, &rows, &expected[rounding as usize - 1], None, column, &name, &fmod_exact);
        }
    }
    fmod_set_float_mode(FloatMode::default());
}

// ---- HalfExpansion ----

/// v_cvt_f32_f16 of both halves (f16 denormals expand exactly, NaNs keep
/// their payload and are quieted) and SDWA v_ldexp_f16 BYTE_0 / BYTE_1
/// sources (hardware rows, wave32 and wave64).
#[test]
fn half_expansion() {
    const VECTORS: [(u32, [u32; 4]); 20] = [
        (0x3c003c00, [0x3f800000, 0x3f800000, 0x00000000, 0x0000003c]),
        (0xc000bc00, [0xbf800000, 0xc0000000, 0x00000000, 0x000000bc]),
        (0x80000000, [0x00000000, 0x80000000, 0x00000000, 0x00000000]),
        (0x00008000, [0x80000000, 0x00000000, 0x00000000, 0x00000080]),
        (0x03ff0001, [0x33800000, 0x387fc000, 0x00000001, 0x00000000]),
        (0x040003ff, [0x387fc000, 0x38800000, 0x000000ff, 0x00000003]),
        (0x7bff0400, [0x38800000, 0x477fe000, 0x00000000, 0x00000004]),
        (0xfbff7bff, [0x477fe000, 0xc77fe000, 0x000000ff, 0x0000007b]),
        (0xfc007c00, [0x7f800000, 0xff800000, 0x00000000, 0x0000007c]),
        (0x7e01fe01, [0xffc02000, 0x7fc02000, 0x00000001, 0x000000fe]),
        (0xfe017e01, [0x7fc02000, 0xffc02000, 0x00000001, 0x0000007e]),
        (0x7fff7e55, [0x7fcaa000, 0x7fffe000, 0x00000055, 0x0000007e]),
        (0xfffffe55, [0xffcaa000, 0xffffe000, 0x00000055, 0x000000fe]),
        (0xfff0fff0, [0xfffe0000, 0xfffe0000, 0x000000f0, 0x000000ff]),
        (0xa5a500f0, [0x37700000, 0xbcb4a000, 0x000000f0, 0x00000000]),
        (0xdead00ff, [0x377f0000, 0xc3d5a000, 0x000000ff, 0x00000000]),
        (0x7c003555, [0x3eaaa000, 0x7f800000, 0x00000055, 0x00000035]),
        (0x3555fc00, [0xff800000, 0x3eaaa000, 0x00000000, 0x000000fc]),
        (0x3c000200, [0x38000000, 0x3f800000, 0x00000000, 0x00000002]),
        (0x80018001, [0xb3800000, 0xb3800000, 0x00000001, 0x00000080]),
    ];
    let code = [0xe0302000, 0x80000400, 0xbf8c3f70, 0x7e141704, 0x2c0a0890, 0x7e161705, 0x7e0c0280, 0x76180cf9, 0x06001604, 0x761a0cf9, 0x06011604, 0xe0782000, 0x80010a00, 0xbf810000];
    let input: Vec<u32> = (0..64).map(|t| VECTORS[t % VECTORS.len()].0).collect();
    let names = ["low half", "high half", "BYTE_0 ldexp", "BYTE_1 ldexp"];
    let mut failures = Vec::new();
    for wave in [32u32, 64] {
        let mut user = vec![0u32; 8];
        user[..4].copy_from_slice(&buffer_descriptor(FMOD_INPUT_BASE, 4, 64, 0x0101_6FAC));
        user[4..].copy_from_slice(&buffer_descriptor(FMOD_OUTPUT_BASE, 16, 64, 0x0101_6FAC));
        let run = ComputeRun::new(code.to_vec(), wave, wave, user);
        let out = fmod_run_waves(&run, true, wave, &input, 64 * 4).unwrap_or_else(|e| panic!("{e}"));
        let plain = fmod_run_waves(&run, false, wave, &input, 64 * 4).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(out, plain, "optimized and unoptimized runs differ");
        for tid in 0..wave as usize {
            let (value, expected) = VECTORS[tid % VECTORS.len()];
            for i in 0..4 {
                if out[tid * 4 + i] != expected[i] {
                    failures.push(format!("wave{wave} lane {tid} input {value:#010x} {} is {:#010x}, expected {:#010x}", names[i], out[tid * 4 + i], expected[i]));
                }
            }
        }
    }
    fmod_report(&failures);
}

// ---- LdexpF32Denormal ----

/// v_ldexp_f32 with denormal inputs and results in every f32 denormal and
/// round mode (hardware rows): results round once in the selected mode,
/// overflow saturates to the largest finite value in the modes that round
/// away from infinity, flushed modes flush denormal inputs and results,
/// IEEE mode quiets signalling NaNs.
#[test]
fn ldexp_f32_denormal() {
    // value, exponent, then: kept, kept IEEE, flushed, kept +inf, kept -inf,
    // kept zero, flushed +inf, flushed -inf, flushed zero.
    #[rustfmt::skip]
    const VECTORS: [[u32; 11]; 36] = [
        [0x00000001, 0x00000003, 0x00000008, 0x00000008, 0x00000000, 0x00000008, 0x00000008, 0x00000008, 0x00000000, 0x00000000, 0x00000000],
        [0x00000001, 0x00000016, 0x00400000, 0x00400000, 0x00000000, 0x00400000, 0x00400000, 0x00400000, 0x00000000, 0x00000000, 0x00000000],
        [0x007fffff, 0xffffffe9, 0x00000001, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
        [0x007fffff, 0xfffffffd, 0x00100000, 0x00100000, 0x00000000, 0x00100000, 0x000fffff, 0x000fffff, 0x00000000, 0x00000000, 0x00000000],
        [0x00000001, 0x00000019, 0x01800000, 0x01800000, 0x00000000, 0x01800000, 0x01800000, 0x01800000, 0x00000000, 0x00000000, 0x00000000],
        [0x00000001, 0x0000007d, 0x33800000, 0x33800000, 0x00000000, 0x33800000, 0x33800000, 0x33800000, 0x00000000, 0x00000000, 0x00000000],
        [0x00000001, 0x000003e8, 0x7f800000, 0x7f800000, 0x00000000, 0x7f800000, 0x7f7fffff, 0x7f7fffff, 0x00000000, 0x00000000, 0x00000000],
        [0x80000001, 0x000003e8, 0xff800000, 0xff800000, 0x80000000, 0xff7fffff, 0xff800000, 0xff7fffff, 0x80000000, 0x80000000, 0x80000000],
        [0x80000001, 0x00000003, 0x80000008, 0x80000008, 0x80000000, 0x80000008, 0x80000008, 0x80000008, 0x80000000, 0x80000000, 0x80000000],
        [0x3f800000, 0xffffff6c, 0x00000002, 0x00000002, 0x00000000, 0x00000002, 0x00000002, 0x00000002, 0x00000000, 0x00000000, 0x00000000],
        [0x3f000000, 0xffffff6c, 0x00000001, 0x00000001, 0x00000000, 0x00000001, 0x00000001, 0x00000001, 0x00000000, 0x00000000, 0x00000000],
        [0x3f000000, 0xffffff82, 0x00400000, 0x00400000, 0x00000000, 0x00400000, 0x00400000, 0x00400000, 0x00000000, 0x00000000, 0x00000000],
        [0xbf000000, 0xffffff6c, 0x80000001, 0x80000001, 0x80000000, 0x80000001, 0x80000001, 0x80000001, 0x80000000, 0x80000000, 0x80000000],
        [0x00800000, 0xffffffe8, 0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
        [0x3eaaaaab, 0xffffff83, 0x00555556, 0x00555556, 0x00000000, 0x00555556, 0x00555555, 0x00555555, 0x00000000, 0x00000000, 0x00000000],
        [0x80800000, 0xffffffe8, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000001, 0x80000000, 0x80000000, 0x80000000, 0x80000000],
        [0x3f7fffff, 0xffffff82, 0x00800000, 0x00800000, 0x00000000, 0x00800000, 0x007fffff, 0x007fffff, 0x00000000, 0x00000000, 0x00000000],
        [0xbfc00000, 0xffffff6a, 0x80000001, 0x80000001, 0x80000000, 0x80000000, 0x80000001, 0x80000000, 0x80000000, 0x80000000, 0x80000000],
        [0x3f7fffff, 0xffffff6c, 0x00000002, 0x00000002, 0x00000000, 0x00000002, 0x00000001, 0x00000001, 0x00000000, 0x00000000, 0x00000000],
        [0x3fc00000, 0xffffff6c, 0x00000003, 0x00000003, 0x00000000, 0x00000003, 0x00000003, 0x00000003, 0x00000000, 0x00000000, 0x00000000],
        [0x00000000, 0x00000003, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
        [0x80000000, 0x00000003, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000],
        [0x7f800000, 0x00000003, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000],
        [0xff800000, 0xfffffffd, 0xff800000, 0xff800000, 0xff800000, 0xff800000, 0xff800000, 0xff800000, 0xff800000, 0xff800000, 0xff800000],
        [0x7fc00000, 0x00000003, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000],
        [0x7fa00000, 0x00000003, 0x7fa00000, 0x7fe00000, 0x7fa00000, 0x7fa00000, 0x7fa00000, 0x7fa00000, 0x7fa00000, 0x7fa00000, 0x7fa00000],
        [0xffbfffff, 0xfffffffd, 0xffbfffff, 0xffffffff, 0xffbfffff, 0xffbfffff, 0xffbfffff, 0xffbfffff, 0xffbfffff, 0xffbfffff, 0xffbfffff],
        [0x7f800001, 0x00000019, 0x7f800001, 0x7fc00001, 0x7f800001, 0x7f800001, 0x7f800001, 0x7f800001, 0x7f800001, 0x7f800001, 0x7f800001],
        [0x3f800000, 0x000003e8, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0x7f7fffff, 0x7f7fffff],
        [0x3f800000, 0xfffffc17, 0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
        [0x3f800000, 0x00000081, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0x7f7fffff, 0x7f7fffff],
        [0xbf800000, 0x00000081, 0xff800000, 0xff800000, 0xff800000, 0xff7fffff, 0xff800000, 0xff7fffff, 0xff7fffff, 0xff800000, 0xff7fffff],
        [0x40200000, 0xffffff6a, 0x00000001, 0x00000001, 0x00000000, 0x00000002, 0x00000001, 0x00000001, 0x00000000, 0x00000000, 0x00000000],
        [0xc0e80000, 0xffffff6c, 0x8000000e, 0x8000000e, 0x80000000, 0x8000000e, 0x8000000f, 0x8000000e, 0x80000000, 0x80000000, 0x80000000],
        [0x3f800000, 0xfffe7960, 0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
        [0xbf800000, 0xfffe7960, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000001, 0x80000000, 0x80000000, 0x80000000, 0x80000000],
    ];
    let code = [0x34020084, 0x34060086, 0xe0381000, 0x80000401, 0xbf8c3f70, 0xd762000a, 0x02020b04, 0xe0701000, 0x80010a03, 0xbf810000];
    let mut input = vec![0u32; 64 * 4];
    for (lane, v) in VECTORS.iter().enumerate() {
        input[lane * 4] = v[0];
        input[lane * 4 + 1] = v[1];
    }
    let modes = [
        ("denormals kept, IEEE=0", mode(0xF0, true, false), 2),
        ("denormals kept, IEEE=1", mode(0xF0, true, true), 3),
        ("denormals kept, round toward +inf", mode(0xF1, true, false), 5),
        ("denormals kept, round toward -inf", mode(0xF2, true, false), 6),
        ("denormals kept, round toward zero", mode(0xF3, true, false), 7),
        ("denormals flushed", mode(0xC0, true, false), 4),
        ("denormals flushed, round toward +inf", mode(0xC1, true, false), 8),
        ("denormals flushed, round toward -inf", mode(0xC2, true, false), 9),
        ("denormals flushed, round toward zero", mode(0xC3, true, false), 10),
    ];
    let mut failures = Vec::new();
    for (name, float_mode, column) in modes {
        let out = fmod_dispatch_mode(&code, 32, 64, &input, 64 * 16, 0, float_mode).unwrap_or_else(|e| panic!("{e}"));
        for (lane, v) in VECTORS.iter().enumerate() {
            let actual = out[lane * 16];
            if actual != v[column] {
                failures.push(format!("{name}: v_ldexp_f32({:#010x}, {:#010x}) is {actual:#010x}, expected {:#010x}", v[0], v[1], v[column]));
            }
        }
    }
    fmod_report(&failures);
}

// ---- F32Dot2Rounding ----

/// The C++ standard library's mt19937 (the reference's case generator).
struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1_812_433_253u32.wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { state, index: 624 }
    }

    fn next(&mut self) -> u32 {
        if self.index >= 624 {
            for k in 0..624 {
                let y = (self.state[k] & 0x8000_0000) | (self.state[(k + 1) % 624] & 0x7fff_ffff);
                let mut v = self.state[(k + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.state[k] = v;
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
}

#[derive(Clone, Copy)]
struct Half {
    bits: u32,
    nan: bool,
    infinite: bool,
    zero: bool,
    negative: bool,
    significand: u64,
    scale: u32,
}

fn decode_half(packed: u32, shift: u32) -> Half {
    let bits = (packed >> shift) & 0xffff;
    let magnitude = bits & 0x7fff;
    let exponent = magnitude >> 10;
    let fraction = magnitude & 0x3ff;
    Half {
        bits,
        nan: magnitude > 0x7c00,
        infinite: magnitude == 0x7c00,
        zero: magnitude == 0,
        negative: bits & 0x8000 != 0,
        significand: if exponent == 0 { fraction as u64 } else { (fraction | 0x400) as u64 },
        scale: if exponent == 0 { 0 } else { exponent - 1 },
    }
}

/// Exact fixed-point accumulator (two's complement, 320 bits).
#[derive(Default)]
struct Exact {
    limbs: [u32; 10],
}

impl Exact {
    fn add(&mut self, magnitude: u64, shift: u32, negative: bool) {
        let mut term = [0u32; 10];
        for bit in 0..64 {
            if (magnitude >> bit) & 1 != 0 {
                let p = bit + shift;
                term[(p / 32) as usize] |= 1 << (p % 32);
            }
        }
        let mut carry = 0u64;
        if negative {
            for limb in &mut term {
                *limb = !*limb;
            }
            carry = 1;
        }
        for i in 0..10 {
            let sum = self.limbs[i] as u64 + term[i] as u64 + carry;
            self.limbs[i] = sum as u32;
            carry = sum >> 32;
        }
    }

    fn zero(&self) -> bool {
        self.limbs.iter().all(|&l| l == 0)
    }

    fn rounded(&self) -> Option<u32> {
        let mut magnitude = self.limbs;
        let negative = magnitude[9] >> 31 != 0;
        if negative {
            let mut carry = 1u64;
            for limb in &mut magnitude {
                let sum = (!*limb) as u64 + carry;
                *limb = sum as u32;
                carry = sum >> 32;
            }
        }
        let bit = |b: i32| (magnitude[(b / 32) as usize] >> (b % 32)) & 1 != 0;
        let top = (0..320).rev().find(|&b| bit(b))?;
        if top < 23 {
            return None;
        }
        let mut mantissa = 0u64;
        for b in ((top - 23)..=top).rev() {
            mantissa = (mantissa << 1) | bit(b) as u64;
        }
        let guard_bit = top - 24;
        let guard = guard_bit >= 0 && bit(guard_bit);
        let sticky = (0..guard_bit.max(0)).any(bit);
        let mut exponent = top - 22;
        if guard && (sticky || mantissa & 1 != 0) {
            mantissa += 1;
            if mantissa == 1 << 24 {
                mantissa >>= 1;
                exponent += 1;
            }
        }
        if !(1..=254).contains(&exponent) {
            return None;
        }
        Some(if negative { 0x8000_0000 } else { 0 } | (exponent as u32) << 23 | (mantissa as u32 & 0x7f_ffff))
    }
}

fn dot2(a: u32, b: u32, c: u32) -> Option<u32> {
    let (al, bl, ah, bh) = (decode_half(a, 0), decode_half(b, 0), decode_half(a, 16), decode_half(b, 16));
    for h in [al, bl, ah, bh] {
        if h.nan {
            return Some(if h.negative { 0x8000_0000 } else { 0 } | 0x7fc0_0000 | ((h.bits & 0x3ff) << 13));
        }
    }
    let mc = c & 0x7fff_ffff;
    if mc > 0x7f80_0000 {
        return Some(c | 0x0040_0000);
    }
    let neg_c = c & 0x8000_0000 != 0;
    let inf_c = mc == 0x7f80_0000;
    let sign_low = al.negative != bl.negative;
    let sign_high = ah.negative != bh.negative;
    let invalid_product = |l: &Half, r: &Half| (l.infinite && r.zero) || (r.infinite && l.zero);
    let infinite_product = |l: &Half, r: &Half| (l.infinite && !r.zero) || (r.infinite && !l.zero);
    let inf_low = infinite_product(&al, &bl);
    let inf_high = infinite_product(&ah, &bh);
    let invalid = invalid_product(&al, &bl)
        || invalid_product(&ah, &bh)
        || (inf_low && inf_high && sign_low != sign_high)
        || (inf_low && inf_c && sign_low != neg_c)
        || (inf_high && inf_c && sign_high != neg_c);
    if invalid {
        return Some(0xffc0_0000);
    }
    if inf_low || inf_high || inf_c {
        let negative = if inf_low { sign_low } else if inf_high { sign_high } else { neg_c };
        return Some(if negative { 0x8000_0000 } else { 0 } | 0x7f80_0000);
    }
    let mut sum = Exact::default();
    sum.add(al.significand * bl.significand, al.scale + bl.scale + 101, sign_low);
    sum.add(ah.significand * bh.significand, ah.scale + bh.scale + 101, sign_high);
    if mc >= 0x0080_0000 {
        sum.add(((c & 0x7f_ffff) | 0x80_0000) as u64, (mc >> 23) - 1, neg_c);
    }
    if sum.zero() {
        return Some(if sign_low && sign_high && neg_c { 0x8000_0000 } else { 0 });
    }
    let r = sum.rounded()?;
    let e = (r >> 23) & 0xff;
    if !(27..254).contains(&e) {
        return None;
    }
    Some(r)
}

fn half_to_f32(half: u32) -> u32 {
    let d = decode_half(half, 0);
    if d.zero {
        return if d.negative { 0x8000_0000 } else { 0 };
    }
    let mut significand = d.significand;
    let mut exponent = d.scale + 103;
    while significand & 0x400 == 0 {
        significand <<= 1;
        exponent -= 1;
    }
    let sign = if d.negative { 0x8000_0000 } else { 0 };
    sign | (exponent << 23) | ((significand & 0x3ff) << 13) as u32
}

/// v_dot2_f32_f16 in FLOAT_MODE 0xC0: both products and the f32 addend sum
/// exactly and round once to nearest even; NaN inputs propagate quieted
/// (half payloads widened), invalid combinations give the default NaN.
#[test]
fn f32_dot2_rounds_once() {
    let mut random = Mt19937::new(2026);
    let mut cases: Vec<[u32; 4]> = Vec::new();
    let mut add = |a: u32, b: u32, c: u32| {
        if let Some(e) = dot2(a, b, c) {
            cases.push([a, b, c, e]);
        }
    };
    let random_half = |r: &mut Mt19937| {
        let exponent = r.next() % 31;
        let sign = r.next() & 1;
        let fraction = r.next() & 0x3ff;
        (sign << 15) | (exponent << 10) | fraction
    };
    let random_f32 = |r: &mut Mt19937, low: u32, high: u32| {
        let exponent = low + r.next() % (high - low + 1);
        let sign = r.next() & 1;
        let fraction = r.next() & 0x7f_ffff;
        (sign << 31) | (exponent << 23) | fraction
    };
    let random_pair = |r: &mut Mt19937| {
        let low = random_half(r);
        low | (random_half(r) << 16)
    };
    for _ in 0..60000 {
        let a = random_pair(&mut random);
        let b = random_pair(&mut random);
        let c = random_f32(&mut random, 60, 170);
        add(a, b, c);
    }
    for _ in 0..30000 {
        let a = random_pair(&mut random);
        let b = random_pair(&mut random);
        add(a, b, 0);
    }
    for _ in 0..60000 {
        let a = random_pair(&mut random);
        let b = random_pair(&mut random);
        let low = f32::from_bits(half_to_f32(a & 0xffff)) * f32::from_bits(half_to_f32(b & 0xffff));
        let high = f32::from_bits(half_to_f32(a >> 16)) * f32::from_bits(half_to_f32(b >> 16));
        let pick = random.next() % 4;
        let base = match pick {
            0 => -low,
            1 => -high,
            _ => -(low + high),
        };
        let nudge = ((random.next() % 9) as i32 - 4) as u32;
        add(a, b, base.to_bits().wrapping_add(nudge));
    }
    let powers = [0x3c00u32, 0x4000, 0x3800, 0x0400, 0x0001, 0x7bff, 0x3c01, 0x3bff];
    for &la in &powers {
        for &ha in &powers {
            for &lb in &powers {
                for &hb in &powers {
                    for c in [0x00000000u32, 0x3f800000, 0xbf800000, 0x33800000, 0x4b800000, 0x3f800001, 0xcb7fffff, 0x0d800000, 0x00800000, 0x8c7fffff] {
                        add(la | (ha << 16), lb | (hb << 16), c);
                        add(la | (ha << 16), (lb ^ 0x8000) | (hb << 16), c);
                    }
                }
            }
        }
    }
    let specials = [0x7c00u32, 0xfc00, 0x7e00, 0x7c01, 0xfe55, 0x0000, 0x8000, 0x3c00, 0xbc00, 0x0001];
    for &la in &specials {
        for &ha in &specials {
            for &lb in &specials {
                for c in [0x00000000u32, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0x7f800123, 0x3f800000, 0x00000001] {
                    add(la | (ha << 16), lb | (0x3c00 << 16), c);
                }
            }
        }
    }
    while cases.len() % 32 != 0 {
        cases.push([0, 0, 0x3f80_0000, 0x3f80_0000]);
    }
    let input: Vec<u32> = cases.iter().flat_map(|c| [c[0], c[1], c[2], 0]).collect();
    let code = [0x34020084, 0x8f098908, 0x4a020209, 0xe0381000, 0x80000401, 0xbf8c3f70, 0xcc134008, 0x1c1a0b04, 0x2c060282, 0xe0701000, 0x80010803, 0xbf810000];
    let out = run_groups(&code, 32, (cases.len() / 32) as u32, &input, cases.len(), mode(0xC0, true, false));
    let mut failures = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        if out[i] != c[3] {
            failures.push(format!("v_dot2_f32_f16 {:#010x}, {:#010x}, {:#010x} is {:#010x}, expected {:#010x}", c[0], c[1], c[2], out[i], c[3]));
        }
    }
    fmod_report(&failures);
}
