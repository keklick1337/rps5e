// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The hardware-measured ALU reference programs through the SPIR-V compiler
//! on the device, compared word for word with the IR interpreter (which
//! the `ps5shader` reference tests hold to the hardware rows). A mismatch
//! is an emitter or host-driver divergence: NaN / signed-zero handling,
//! f16 conversion rounding, denormals.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored float_ref`.

use super::*;
use ps5shader::testing::{buffer_descriptor, ComputeRun};
use ps5shader::translate::FloatMode;

const INTERP_IN: u64 = 0x1_0000_0000;
const INTERP_OUT: u64 = 0x2_0000_0000;
const POISON: u32 = 0xDEAD_BEEF;
/// Raw buffers (OOB_SELECT 3) and one record per word (OOB_SELECT 1, the
/// index check), as the reference programs were run.
const RAW_DWORD3: u32 = 0x3101_6FAC;
const STRIDED_DWORD3: u32 = 0x1101_6FAC;

fn descriptor(base: u64, words: usize, stride: u32) -> [u32; 4] {
    let records = if stride == 0 { words as u32 * 4 } else { words as u32 };
    buffer_descriptor(base, stride, records, if stride == 0 { RAW_DWORD3 } else { STRIDED_DWORD3 })
}

/// One workgroup of `lanes` invocations in waves of `wave` on the
/// interpreter.
fn interpret(code: &[u32], wave: u32, lanes: u32, input: &[u32], out_words: usize, stride: u32, mode: FloatMode) -> Result<Vec<u32>, String> {
    let mut user = descriptor(INTERP_IN, input.len(), stride).to_vec();
    user.extend(descriptor(INTERP_OUT, out_words, stride));
    let mut run = ComputeRun::new(code.to_vec(), wave, lanes.min(wave), user);
    run.float_mode = mode;
    let f = run.compile(true)?;
    let mut memory = ps5shader::interp::Memory::default();
    memory.write_words(INTERP_IN, input);
    memory.write_words(INTERP_OUT, &vec![POISON; out_words]);
    memory.write_words(run.code_address, &run.code);
    let mut first = 0;
    while first < lanes {
        let n = wave.min(lanes - first);
        let mut m = ps5shader::interp::Machine::new(wave, n);
        m.user_data = run.user_data.clone();
        m.memory = std::mem::take(&mut memory);
        m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(0), (first..first + n).collect());
        m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(1), vec![0; n as usize]);
        m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(2), vec![0; n as usize]);
        m.run(&f).map_err(|e| format!("run: {e}"))?;
        memory = std::mem::take(&mut m.memory);
        first += n;
    }
    Ok(memory.read_words(INTERP_OUT, out_words))
}

/// The same workgroup on the device through the SPIR-V compiler.
#[allow(clippy::too_many_arguments)]
fn execute(g: &mut Guest, rig: &mut Rig, code: &[u32], wave: u32, lanes: u32, input: &[u32], out_words: usize, stride: u32, float_mode: u32) -> Vec<u32> {
    execute_lds(g, rig, code, wave, lanes, input, out_words, stride, float_mode, 0)
}

/// COMPUTE_PGM_RSRC1 float controls of `mode`.
fn rsrc1_of(mode: FloatMode) -> u32 {
    (mode.mode as u32) << 12 | (mode.dx10_clamp as u32) << 21 | (mode.ieee as u32) << 23 | (mode.fp16_overflow as u32) << 26
}

/// [`execute`] with `lds_blocks` 128-dword blocks of LDS (RSRC2 LDS_SIZE).
#[allow(clippy::too_many_arguments)]
fn execute_lds(g: &mut Guest, rig: &mut Rig, code: &[u32], wave: u32, lanes: u32, input: &[u32], out_words: usize, stride: u32, float_mode: u32, lds_blocks: u32) -> Vec<u32> {
    // FLOAT_MODE and DX10_CLAMP as the reference rows were measured with.
    execute_rsrc1(g, rig, code, wave, lanes, input, out_words, stride, (float_mode << 12) | (1 << 21), lds_blocks)
}

/// [`execute_lds`] with the float controls of RSRC1 `rsrc1`.
#[allow(clippy::too_many_arguments)]
fn execute_rsrc1(g: &mut Guest, rig: &mut Rig, code: &[u32], wave: u32, lanes: u32, input: &[u32], out_words: usize, stride: u32, rsrc1: u32, lds_blocks: u32) -> Vec<u32> {
    let cs = g.program(code);
    let ib = g.alloc(&asm::bytes(input));
    let ob = g.alloc(&asm::bytes(&vec![POISON; out_words]));
    let mut user = descriptor(ib, input.len(), stride).to_vec();
    user.extend(descriptor(ob, out_words, stride));
    let mut shader = stage(cs, &user);
    shader.rsrc1 = rsrc1;
    shader.rsrc2 = (user.len() as u32) << 1 | lds_blocks << 15;
    let d = ResolvedDispatch {
        packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1 | if wave == 32 { 1 << 15 } else { 0 }, ..DispatchPacket::default() },
        shader,
        rsrc3: 0,
        threads: [lanes, 1, 1],
        start: [0; 3],
    };
    assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
    assert!(rig.buffers.download(&mut rig.sched, ob, out_words as u64 * 4));
    let mut out = vec![0u8; out_words * 4];
    assert!(g.mem.read_backing(ob, &mut out));
    out.chunks(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect()
}

fn is_nan(bits: u32) -> bool {
    bits & 0x7FFF_FFFF > 0x7F80_0000
}

fn is_nan16(bits: u32) -> bool {
    bits & 0x7FFF > 0x7C00
}

fn half(bits: u32) -> f32 {
    let e = (bits >> 10) & 0x1F;
    let m = (bits & 0x3FF) as f32;
    let v = if e == 0 { m * 2f32.powi(-24) } else { (m + 1024.0) * 2f32.powi(e as i32 - 25) };
    if bits & 0x8000 != 0 {
        -v
    } else {
        v
    }
}

/// Whether the device word `a` matches the interpreter word `e`: equal;
/// both NaN (as f32, or half by half as f16: the device returns its own
/// canonical NaN); or, for cases with a precision allowance, within `ulp`
/// f32 units or `abs` (as f32 or half by half).
fn same(a: u32, e: u32, ulp: u32, abs: f32) -> bool {
    if a == e || (is_nan(a) && is_nan(e)) {
        return true;
    }
    let halves = |f: &dyn Fn(u32, u32) -> bool| f(a & 0xFFFF, e & 0xFFFF) && f(a >> 16, e >> 16);
    if halves(&|x, y| x == y || (is_nan16(x) && is_nan16(y))) {
        return true;
    }
    if ulp > 0 && ulps(a, e) <= ulp {
        return true;
    }
    abs > 0.0
        && ((f32::from_bits(a) - f32::from_bits(e)).abs() <= abs
            || halves(&|x, y| x == y || (!is_nan16(x) && !is_nan16(y) && (half(x) - half(y)).abs() <= abs)))
}

/// Distance in f32 units in the last place (same sign only).
fn ulps(a: u32, b: u32) -> u32 {
    if (a ^ b) & 0x8000_0000 != 0 {
        return u32::MAX;
    }
    a.abs_diff(b)
}

struct Case {
    name: &'static str,
    code: &'static [u32],
    rows: Vec<Vec<u32>>,
    inputs: usize,
    results: usize,
    /// Buffer stride: 0 = raw buffers, 4 = one record per word.
    stride: u32,
    /// Allowed f32 ULP distance (transcendental results use the host's
    /// precision on the device and Rust's in the interpreter).
    ulp: u32,
    /// Allowed absolute difference (0 = none).
    abs: f32,
    /// Per-row bit masks of the result words the hardware rows check
    /// (unchecked cells are host-defined in the reference as well).
    checked: Option<Vec<u64>>,
}

/// Pseudo-random inputs: a mix of special values and random bit patterns.
fn random_rows(n: usize, inputs: usize) -> Vec<Vec<u32>> {
    const SPECIAL: [u32; 16] = [
        0, 0x8000_0000, 0x3F80_0000, 0xBF80_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0000, 0x7F80_0001, 0x0000_0001, 0x8000_0001, 0x7F7F_FFFF, 0x3C00_3C00,
        0x7C00_FC00, 0x0001_8001, 0x3555_4000, 0x4049_0FDB,
    ];
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|_| {
            (0..inputs)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    if x % 4 == 0 {
                        SPECIAL[(x >> 8) as usize % SPECIAL.len()]
                    } else {
                        (x >> 16) as u32
                    }
                })
                .collect()
        })
        .collect()
}

fn run_cases(cases: &[Case]) {
    run_cases_mode(cases, FloatMode::default());
}

/// [`run_cases`] with the float controls `mode` on both sides.
fn run_cases_mode(cases: &[Case], mode: FloatMode) {
    let mut g = Guest::new(256 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    let mut fails = Vec::new();
    for c in cases {
        for (chunk, rows) in c.rows.chunks(32).enumerate() {
            let lanes = rows.len() as u32;
            let mut input = vec![0u32; rows.len() * c.inputs];
            for (lane, r) in rows.iter().enumerate() {
                let n = r.len().min(c.inputs);
                input[lane * c.inputs..lane * c.inputs + n].copy_from_slice(&r[..n]);
            }
            let out_words = rows.len() * c.results;
            let want = match interpret(c.code, 32, lanes, &input, out_words, c.stride, mode) {
                Ok(w) => w,
                Err(e) => {
                    fails.push(format!("{}: interpreter: {e}", c.name));
                    continue;
                }
            };
            let got = execute_rsrc1(&mut g, &mut rig, c.code, 32, lanes, &input, out_words, c.stride, rsrc1_of(mode), 0);
            for (i, (&a, &e)) in got.iter().zip(&want).enumerate() {
                let row = chunk * 32 + i / c.results;
                if c.checked.as_ref().is_some_and(|m| (m[row] >> (i % c.results)) & 1 == 0) {
                    continue;
                }
                // NaN payloads are not compared: the device returns its
                // canonical NaN.
                let ok = same(a, e, c.ulp, c.abs);
                if !ok {
                    let lane = i / c.results;
                    fails.push(format!(
                        "{} mode {:#04x} ieee {} ovfl {} row {} word {}: device {a:#010x}, interpreter {e:#010x} (inputs {:08x?})",
                        c.name,
                        mode.mode,
                        mode.ieee,
                        mode.fp16_overflow,
                        chunk * 32 + lane,
                        i % c.results,
                        &input[lane * c.inputs..(lane + 1) * c.inputs]
                    ));
                }
            }
        }
    }
    assert!(fails.is_empty(), "{} mismatches:\n{}", fails.len(), fails.iter().take(80).cloned().collect::<Vec<_>>().join("\n"));
}

/// Reference tables with THREADS / INPUTS / RESULTS / ROWS and a code array.
macro_rules! table_vectors {
    ($($m:ident => $file:literal, $code:ident $(, $checked:ident)?;)*) => {
        $(
            #[allow(dead_code, unused_imports, clippy::all)]
            mod $m {
                include!(concat!("../../../../ps5shader/tests/vectors/", $file));
                pub fn case(name: &'static str, stride: u32, ulp: u32) -> super::Case {
                    let rows = ROWS.iter().map(|r| r.to_vec()).collect();
                    #[allow(unused_mut, unused_assignments)]
                    let mut checked: Option<Vec<u64>> = None;
                    $(checked = Some($checked.iter().map(|&m| m as u64).collect());)?
                    super::Case { name, code: &$code, rows, inputs: INPUTS as usize, results: RESULTS as usize, stride, ulp, abs: if ulp > 0 { 2f32.powi(-11) } else { 0.0 }, checked }
                }
            }
        )*
    };
}

/// Reference tables without input rows (the tests generate them).
macro_rules! generated_vectors {
    ($($m:ident => $file:literal, $code:ident;)*) => {
        $(
            #[allow(dead_code, unused_imports, clippy::all)]
            mod $m {
                include!(concat!("../../../../ps5shader/tests/vectors/", $file));
                pub fn case(name: &'static str, stride: u32) -> super::Case {
                    let rows = super::random_rows(THREADS as usize, INPUTS as usize);
                    super::Case { name, code: &$code, rows, inputs: INPUTS as usize, results: RESULTS as usize, stride, ulp: 0, abs: 0.0, checked: None }
                }
            }
        )*
    };
}

/// The "rows in, 16 words per lane out" layout (raw buffers).
macro_rules! row_vectors {
    ($($m:ident => $file:literal, $code:ident $(, $checked:ident)?;)*) => {
        $(
            #[allow(dead_code, unused_imports, clippy::all)]
            mod $m {
                include!(concat!("../../../../ps5shader/tests/vectors/", $file));
                pub fn case(name: &'static str, abs: f32) -> super::Case {
                    let rows = ROWS.iter().map(|r| r.to_vec()).collect();
                    #[allow(unused_mut, unused_assignments)]
                    let mut checked: Option<Vec<u64>> = None;
                    $(checked = Some($checked.iter().map(|&m| m as u64).collect());)?
                    super::Case { name, code: &$code, rows, inputs: 4, results: 16, stride: 0, ulp: 0, abs, checked }
                }
            }
        )*
    };
}

table_vectors! {
    min_max_nan => "valu_f32_min_max_nan.rs", CODE;
    output_modifier => "valu_f32_output_modifier.rs", CODE, CHECKED;
    result_clamp => "valu_f32_result_clamp.rs", CODE, CHECKED;
    fma_legacy => "valu_fma_legacy.rs", FMA_LEGACY_CODE;
    fma_rounding => "valu_fma_rounding.rs", FMA_CODE;
    float_unary => "valu_vop1_float_unary.rs", CODE, CHECKED;
    modifier_forms => "valu_vop2_modifier_forms.rs", CODE, CHECKED;
    cube_clamp => "valu_vop3_cube_clamp.rs", CODE;
    legacy_mods => "valu_vop3_legacy_source_modifiers.rs", CODE, CHECKED;
    mad_clamp => "valu_vop3_mad_clamp.rs", CODE;
    pack_ldexp => "valu_vop3_pack_ldexp.rs", CODE;
    f32_denormal_flush => "valu_f32_denormal_flush.rs", CODE;
}

generated_vectors! {
    mix_precision => "valu_mix_precision.rs", MIX_CODE;
    aliases => "valu_vop3_aliases.rs", ALIAS_CODE;
}

row_vectors! {
    div_fixup_f16 => "fmod_div_fixup_f16.rs", CODE;
    div_result_mods => "fmod_division_result_modifiers.rs", CODE, CHECKED;
    div_output_mods => "fmod_division_output_modifiers.rs", CODE, CHECKED;
    f16_ternary_clamp => "fmod_f16_ternary_clamp.rs", CODE, CHECKED;
    f16_dot => "fmod_f16_dot.rs", CODE;
    f16_result_mods => "fmod_f16_result_modifiers.rs", CODE;
    f16_modifiers => "fmod_f16_modifiers.rs", CODE;
    sdwa_float => "fmod_sdwa_float_selectors.rs", CODE, CHECKED;
    sdwa_ldexp => "fmod_sdwa_ldexp_f16.rs", CODE, CHECKED;
    f64_arith => "fmod_f64_arithmetic.rs", CODE;
    f64_conv => "fmod_f64_conversions.rs", CODE;
    f64_nan_conv => "fmod_f64_nan_conversions.rs", CODE;
    f64_rounding => "fmod_f64_rounding.rs", CODE;
    nan_quieting => "valu_nan_quieting.rs", CODE;
    f32_arith_nan => "valu_f32_arithmetic_nan.rs", CODE;
}

#[test]
#[ignore]
fn float_ref_valu_f32_on_device() {
    run_cases(&[
        min_max_nan::case("min/max/med3 NaN", 0, 0),
        output_modifier::case("output modifier", 0, 0),
        result_clamp::case("result clamp", 0, 0),
        fma_legacy::case("fma legacy", 4, 0),
        fma_rounding::case("fma rounding", 4, 0),
        float_unary::case("vop1 float unary", 0, 4),
        modifier_forms::case("vop2 modifier forms", 0, 0),
        cube_clamp::case("vop3 cube clamp", 0, 0),
        legacy_mods::case("vop3 legacy source modifiers", 0, 0),
        mad_clamp::case("vop3 mad clamp", 0, 0),
        pack_ldexp::case("vop3 pack ldexp", 0, 0),
        mix_precision::case("mix precision", 4),
        aliases::case("vop3 aliases", 4),
        f32_arith_nan::case("f32 arithmetic NaN", 0.0),
    ]);
}

#[test]
#[ignore]
fn float_ref_f16_f64_on_device() {
    run_cases(&[
        div_fixup_f16::case("div_fixup_f16", 0.0),
        div_result_mods::case("division result modifiers", 0.0),
        div_output_mods::case("division output modifiers", 0.0),
        f16_ternary_clamp::case("f16 ternary clamp", 0.0),
        f16_dot::case("f16 dot", 0.0),
        f16_result_mods::case("f16 result modifiers", 2f32.powi(-11)),
        f16_modifiers::case("f16 modifiers", 0.0),
        sdwa_float::case("sdwa float selectors", 0.0),
        sdwa_ldexp::case("sdwa ldexp f16", 0.0),
        f64_arith::case("f64 arithmetic", 0.0),
        f64_conv::case("f64 conversions", 0.0),
        f64_nan_conv::case("f64 NaN conversions", 0.0),
        f64_rounding::case("f64 rounding", 0.0),
        f32_denormal_flush::case("f32 denormal flush", 0, 0),
        nan_quieting::case("NaN quieting", 2f32.powi(-11)),
    ]);
}

/// f32 values around every f16 rounding boundary: for each half exponent,
/// a few mantissas and both signs, the exact value, its f32 neighbours
/// and the midpoint to the next half with its neighbours (ties, subnormal
/// results, 65504 / 65520 overflow), plus infinities and NaNs.
fn half_boundary_rows() -> Vec<Vec<u32>> {
    let mut rows = Vec::new();
    for sign in [0u32, 0x8000] {
        for exponent in 0..31u32 {
            for mantissa in [0u32, 1, 2, 0x1FF, 0x200, 0x3FE, 0x3FF] {
                let h = sign | exponent << 10 | mantissa;
                let here = half(h).to_bits();
                let next = half(h + 1).to_bits();
                let mid = (half(h) as f64 + half(h + 1) as f64) / 2.0;
                let mid = (mid as f32).to_bits();
                for v in [here, here.wrapping_add(1), here.wrapping_sub(1), mid, mid.wrapping_add(1), mid.wrapping_sub(1), next] {
                    rows.push(vec![v]);
                }
            }
        }
    }
    for v in [0x7F80_0000u32, 0xFF80_0000, 0x7FC0_0000, 0x7F80_0001, 0xFFFF_FFFF, 0x477F_F000, 0x477F_EFFF, 0x4780_0000, 0x7F7F_FFFF, 0xFF7F_FFFF, 0x5000_0000, 0x4780_0001] {
        rows.push(vec![v]);
    }
    rows
}

/// v_cvt_f16_f32 (round to nearest even) and v_cvt_pkrtz_f16_f32 (toward
/// zero) of values around every half rounding boundary on the device,
/// against the interpreter: the native conversion (where the device
/// guarantees 16-bit round-to-nearest-even) and the integer one agree with
/// the hardware model bit for bit.
#[test]
#[ignore]
fn float_ref_f16_conversions_on_device() {
    // v1 = tid * 4; v4 = input[tid]; v20 = convert(v4); output[tid] = v20.
    const LOAD: [u32; 4] = [0x3402_0082, 0xe030_1000, 0x8000_0401, 0xbf8c_3f70];
    const STORE: [u32; 3] = [0xe070_1000, 0x8001_1401, 0xbf81_0000];
    let program = |op: &[u32]| -> &'static [u32] { Box::leak([&LOAD[..], op, &STORE[..]].concat().into_boxed_slice()) };
    // VOP1 v_cvt_f16_f32 v20, v4; VOP2 v_cvt_pkrtz_f16_f32 v20, v4, v4.
    let rte = program(&[(0x3F << 25) | (20 << 17) | (0x0A << 9) | 260]);
    let rtz = program(&[(0x2F << 25) | (20 << 17) | (4 << 9) | 260]);
    let native = GpuDevice::create(false).map(|d| d.caps.float16_rte).unwrap_or(false);
    eprintln!("float_ref_f16_conversions: native round-to-nearest-even f16 conversions: {native}");
    let rows = half_boundary_rows();
    let case = |name, code| Case { name, code, rows: rows.clone(), inputs: 1, results: 1, stride: 0, ulp: 0, abs: 0.0, checked: None };
    run_cases(&[case("v_cvt_f16_f32", rte), case("v_cvt_pkrtz_f16_f32", rtz)]);
}

#[allow(dead_code, clippy::all)]
mod log_near_one {
    include!("../../../../ps5shader/tests/vectors/valu_log_near_one.rs");
    pub const LANES: u32 = THREADS;
    pub const WORDS: usize = RESULTS as usize;
    pub fn code() -> &'static [u32] {
        &CODE
    }
    pub fn rows() -> &'static [[u32; 4]] {
        &ROWS
    }
    pub fn expected() -> &'static [[u32; 2]] {
        &EXPECTED
    }
    pub fn tolerance() -> &'static [u32] {
        &TOLERANCE
    }
}

/// v_log_f32 / v_log_f16 near 1.0 on the device against the hardware rows
/// (within their ULP tolerance; zeros, infinities and NaNs exact): the
/// host's log2 alone strays there, the emitted series keeps it in range
/// (`RPS5E_LOG_SERIES`).
#[test]
#[ignore]
fn float_ref_log_near_one_on_device() {
    use log_near_one::{code, expected, rows, tolerance, LANES, WORDS};
    let (rows, expected, tolerance) = (rows(), expected(), tolerance());
    let ordered = |bits: u32, width: u32| {
        let sign = 1u32 << (width - 1);
        if bits & sign != 0 {
            sign - (bits & !sign)
        } else {
            bits | sign
        }
    };
    let mut g = Guest::new(256 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    let input: Vec<u32> = rows.iter().flatten().copied().collect();
    let out = execute(&mut g, &mut rig, code(), 32, LANES, &input, LANES as usize * WORDS, 0, 0xC0);
    let mut fails = Vec::new();
    for tid in 0..LANES as usize {
        for column in 0..2 {
            let width = if column == 0 { 32 } else { 16 };
            let mask: u32 = if width == 16 { 0xffff } else { 0xffff_ffff };
            let a = out[tid * WORDS + column] & mask;
            let e = expected[tid][column] & mask;
            let magnitude = e & (mask >> 1);
            let pinned = magnitude == 0 || magnitude >= if width == 16 { 0x7c00 } else { 0x7f80_0000 };
            let ok = if pinned { a == e } else { ordered(a, width).abs_diff(ordered(e, width)) <= tolerance[tid] };
            if !ok {
                fails.push(format!("lane {tid} column {column} of {:#010x}: device {a:#010x}, hardware {e:#010x} (tolerance {})", rows[tid][column], tolerance[tid]));
            }
        }
    }
    assert!(fails.is_empty(), "{} mismatches:\n{}", fails.len(), fails.join("\n"));
}

#[allow(dead_code, clippy::all)]
mod lds_mixed_width {
    include!("../../../../ps5shader/tests/vectors/mem_lds_mixed_width.rs");
    pub const LANES: u32 = THREADS;
    pub fn programs() -> Vec<(&'static str, Vec<u32>, u64, u32)> {
        vec![
            ("uniform", uniform_code(), THREADS as u64 * ITERATIONS as u64, 1),
            ("divergent", divergent_code(), THREADS as u64 / 2 * ITERATIONS as u64, 1),
            ("split lanes", parity_code(), THREADS as u64 / 2, 32),
            ("split lanes with per-lane operands", loaded_parity_code(), THREADS as u64 / 2, 32),
        ]
    }
}

/// ds_add_u64 and ds_add_u32 on one LDS slot from eight wave32 waves on the
/// device: the 64-bit adds are native workgroup atomics on a u64 view of
/// the LDS and the 32-bit ones native atomics on the high dword, and every
/// add of both widths lands (the split-lane programs repeat 32 times, as
/// the reference runs them).
#[test]
#[ignore]
fn lds_mixed_width_atomics_on_device() {
    use lds_mixed_width::{programs, LANES};
    let mut g = Guest::new(256 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    let input: Vec<u32> = (0..LANES).flat_map(|_| [0xffff_ffff, 0, 1, 0]).collect();
    let mut fails = Vec::new();
    for (name, code, operations, repeats) in programs() {
        let value = operations * 0xffff_ffff + (operations << 32);
        for run in 0..repeats {
            let out = execute_lds(&mut g, &mut rig, &code, 32, LANES, &input, LANES as usize * 2, 0, 0xC0, 1);
            for lane in 0..LANES as usize {
                let got = out[lane * 2] as u64 | (out[lane * 2 + 1] as u64) << 32;
                if got != value {
                    fails.push(format!("{name} run {run} lane {lane}: {got:#018x}, expected {value:#018x}"));
                }
            }
        }
    }
    assert!(fails.is_empty(), "{} mismatches:\n{}", fails.len(), fails.iter().take(40).cloned().collect::<Vec<_>>().join("\n"));
}

/// A GLC buffer_load_dwordx2 after a buffer_atomic_add_x2 on the same
/// pair: read with one 64-bit atomic load on the device, the value the
/// interpreter computes (the carry into the high dword included).
#[test]
#[ignore]
fn coherent_dwordx2_load_after_add_x2_on_device() {
    // v1 = tid * 8; v[2:3] = 1; buffer_atomic_add_x2 v[2:3], v1, s[0:3]
    // offen; buffer_load_dwordx2 v[4:5], v1, s[0:3] offen glc; s_waitcnt;
    // buffer_store_dwordx2 v[4:5], v1, s[4:7] offen.
    static CODE: [u32; 11] = [0x3402_0083, 0x7E04_0281, 0x7E06_0280, 0xE148_1000, 0x8000_0201, 0xE034_5000, 0x8000_0401, 0xBF8C_0000, 0xE074_1000, 0x8001_0401, 0xBF81_0000];
    let rows: Vec<Vec<u32>> = (0..32u32).map(|k| vec![if k % 2 == 0 { 0xFFFF_FFFF } else { k }, k * 3]).collect();
    run_cases(&[Case { name: "buffer_load_dwordx2 glc after buffer_atomic_add_x2", code: &CODE, rows, inputs: 2, results: 2, stride: 0, ulp: 0, abs: 0.0, checked: None }]);
}

#[allow(dead_code, clippy::all)]
mod lds_add_f32 {
    include!("../../../../ps5shader/tests/vectors/lds_add_f32.rs");
}

/// ds_add_f32 / ds_add_rtn_f32 on the device (a compare-exchange loop with
/// the measured flush and NaN rules) in FLOAT_MODE 0xC0, where both flushes
/// apply and no row depends on the host's denormals: the stored values of
/// the hardware rows.
#[test]
#[ignore]
fn lds_add_f32_on_device() {
    use lds_add_f32::{CODE, MEASURED, OPERANDS};
    let mut g = Guest::new(256 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    let mut fails = Vec::new();
    for first in [0usize, 32] {
        let input: Vec<u32> = OPERANDS[first..first + 32].iter().flatten().copied().collect();
        let out = execute_lds(&mut g, &mut rig, &CODE, 32, 32, &input, 32 * 4, 0, 0xC0, 4);
        for lane in 0..32 {
            let row = first + lane;
            let want = [OPERANDS[row][0], MEASURED[0][row], MEASURED[0][row]];
            let got = [out[lane * 4], out[lane * 4 + 1], out[lane * 4 + 2]];
            if got != want {
                fails.push(format!("row {row} {:08x?}: {got:08x?}, expected {want:08x?}", OPERANDS[row]));
            }
        }
    }
    assert!(fails.is_empty(), "{} mismatches:\n{}", fails.len(), fails.join("\n"));
}

#[allow(dead_code, clippy::all)]
mod denormal_modes {
    include!("../../../../ps5shader/tests/vectors/valu_denormal_modes.rs");
}

fn leak(code: Vec<u32>) -> &'static [u32] {
    Box::leak(code.into_boxed_slice())
}

fn float_mode(mode: u8, ieee: bool, fp16_overflow: bool) -> FloatMode {
    FloatMode { mode, dx10_clamp: true, ieee, fp16_overflow }
}

/// The f16 unary operations and conversions of the reference's
/// F16DenormalFlush programs (v4 = half; v10 = 0x5a5a5a5a; v10 = op(v4)).
fn f16_unary_cases() -> Vec<Case> {
    use denormal_modes::f16_unary::*;
    let tables: [(&'static str, u32, &[[u32; 5]]); 14] = [
        ("v_ceil_f16", 0x7e14b904, &CEIL_ROWS),
        ("v_cvt_f32_f16", 0x7e141704, &CVT_F32_ROWS),
        ("v_cvt_norm_i16_f16", 0x7e14c704, &CVT_NORM_I16_ROWS),
        ("v_cvt_norm_u16_f16", 0x7e14c904, &CVT_NORM_U16_ROWS),
        ("v_exp_f16", 0x7e14b104, &EXP_ROWS),
        ("v_floor_f16", 0x7e14b704, &FLOOR_ROWS),
        ("v_fract_f16", 0x7e14bf04, &FRACT_ROWS),
        ("v_frexp_exp_i16_f16", 0x7e14b504, &FREXP_EXP_ROWS),
        ("v_frexp_mant_f16", 0x7e14b304, &FREXP_MANT_ROWS),
        ("v_log_f16", 0x7e14af04, &LOG_ROWS),
        ("v_rcp_f16", 0x7e14a904, &RCP_ROWS),
        ("v_rsq_f16", 0x7e14ad04, &RSQ_ROWS),
        ("v_sin_f16", 0x7e14c104, &SIN_ROWS),
        ("v_sqrt_f16", 0x7e14ab04, &SQRT_ROWS),
    ];
    tables
        .into_iter()
        .map(|(name, word, rows)| {
            let code = leak(vec![0x34020084, 0x34040082, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x7e1402ff, 0x5a5a5a5a, word, 0xe0701000, 0x80010a02, 0xbf810000]);
            Case { name, code, rows: rows.iter().map(|r| vec![r[0]]).collect(), inputs: 4, results: 1, stride: 0, ulp: 0, abs: 0.0, checked: None }
        })
        .collect()
}

/// The reference's f16/f64 denormal-mode programs with their input rows.
fn denormal_mode_cases() -> Vec<Case> {
    use denormal_modes::*;
    let mut cases = f16_unary_cases();
    let rows3 = |rows: &[[u32; 3]]| rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>();
    let compare_rows = |values: &[u32; 16]| (0..256).map(|p| vec![values[p / 16], values[p % 16]]).collect::<Vec<_>>();
    cases.push(Case { name: "v_cmp_*_f16", code: &vcmp_f16::COMPARE_CODE, rows: compare_rows(&vcmp_f16::VALUES), inputs: 4, results: 16, stride: 0, ulp: 0, abs: 0.0, checked: None });
    cases.push(Case { name: "v_cmpx_*_f16", code: &vcmp_f16::COMPARE_EXEC_CODE, rows: compare_rows(&vcmp_f16::VALUES), inputs: 4, results: 16, stride: 0, ulp: 0, abs: 0.0, checked: None });
    cases.push(Case { name: "v_pk_add/mul/fma_f16", code: &packed_f16::CODE, rows: rows3(&packed_f16::ROWS), inputs: 4, results: 4, stride: 0, ulp: 0, abs: 0.0, checked: None });
    let pack_rows = (0..256).map(|p| vec![f16_pack::SINGLES[p / 16], f16_pack::SINGLES[p % 16], f16_pack::HALVES[p / 16], f16_pack::HALVES[p % 16]]).collect();
    cases.push(Case { name: "v_cvt_pkrtz_f16_f32 / v_pack_b32_f16", code: &f16_pack::CODE, rows: pack_rows, inputs: 4, results: 2, stride: 0, ulp: 0, abs: 0.0, checked: None });
    cases.push(Case { name: "v_fma_f16 / v_fmac_f16", code: &f16_fma::CODE, rows: rows3(&f16_fma::ROWS), inputs: 4, results: 2, stride: 0, ulp: 0, abs: 0.0, checked: None });
    cases.push(Case { name: "v_fma_mix_f32 f16 sources", code: &fma_mix_f16::CODE, rows: rows3(&fma_mix_f16::ROWS), inputs: 4, results: 2, stride: 0, ulp: 0, abs: 0.0, checked: None });
    let v = &div_fixup_f16::VALUES;
    let fixup_rows = (0..1000).map(|r| vec![v[r / 100], v[r / 10 % 10], v[r % 10]]).collect();
    cases.push(Case { name: "v_div_fixup_f16", code: &div_fixup_f16::CODE, rows: fixup_rows, inputs: 4, results: 1, stride: 0, ulp: 0, abs: 0.0, checked: None });
    let d = &div_fixup_f64::VALUES;
    let fixup64_rows = (0..1000)
        .map(|r| [d[r / 100], d[r / 10 % 10], d[r % 10]].iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect())
        .collect();
    cases.push(Case { name: "v_div_fixup_f64", code: &div_fixup_f64::CODE, rows: fixup64_rows, inputs: 8, results: 2, stride: 0, ulp: 0, abs: 0.0, checked: None });
    for (name, word, rows) in [("v_cvt_pknorm_i16_f16", 0xd712000au32, &pknorm_f16::PKNORM_I16_ROWS), ("v_cvt_pknorm_u16_f16", 0xd713000a, &pknorm_f16::PKNORM_U16_ROWS)] {
        let code = leak(vec![0x34020084, 0x34040082, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x7e140280, word, 0x00020b04, 0xe0701000, 0x80010a02, 0xbf810000]);
        cases.push(Case { name, code, rows: rows.iter().map(|r| vec![r[0], r[1]]).collect(), inputs: 4, results: 1, stride: 0, ulp: 0, abs: 0.0, checked: None });
    }
    let ldexp_rows = f16_rounding::LDEXP_DENORMAL_VECTORS.iter().map(|r| vec![r[0], r[1]]).collect();
    cases.push(Case { name: "v_ldexp_f16 denormals", code: &f16_rounding::LDEXP_DENORMAL_CODE, rows: ldexp_rows, inputs: 4, results: 16, stride: 4, ulp: 0, abs: 0.0, checked: None });
    cases
}

/// The reference's f16 conversion programs (F16ConversionFloatMode):
/// v_cvt_f16_f32 (plain and clamp), v_cvt_f16_u16 / i16, v_ldexp_f16.
fn f16_conversion_cases() -> Vec<Case> {
    use denormal_modes::f16_conversion::*;
    let shader = |word: u32, second: Option<u32>| {
        let mut code = vec![0x34020084, 0x34040082, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x7e140280, word];
        code.extend(second);
        code.extend([0xe0701000, 0x80010a02, 0xbf810000]);
        leak(code)
    };
    let tables: [(&'static str, &'static [u32], &[[u32; 18]]); 5] = [
        ("v_cvt_f16_f32", shader(0x7e141504, None), &V_CVT_F16_F32_ROWS),
        ("v_cvt_f16_f32 clamp", shader(0xd58a800a, Some(0x00000104)), &V_CVT_F16_F32_CLAMP_ROWS),
        ("v_cvt_f16_u16", shader(0x7e14a104, None), &V_CVT_F16_U16_ROWS),
        ("v_cvt_f16_i16", shader(0x7e14a304, None), &V_CVT_F16_I16_ROWS),
        ("v_ldexp_f16", shader(0x76140b04, None), &V_LDEXP_F16_ROWS),
    ];
    let mut cases: Vec<Case> = tables
        .into_iter()
        .map(|(name, code, rows)| Case { name, code, rows: rows.iter().map(|r| vec![r[0], r[1]]).collect(), inputs: 4, results: 1, stride: 0, ulp: 0, abs: 0.0, checked: None })
        .collect();
    let boundary: Vec<Vec<u32>> = half_boundary_rows();
    cases.push(Case { name: "v_cvt_f16_f32 boundaries", code: shader(0x7e141504, None), rows: boundary, inputs: 4, results: 1, stride: 0, ulp: 0, abs: 0.0, checked: None });
    cases
}

/// The f16/f64 denormal modes (FLOAT_MODE[7:6] 0..3, f32 denormals kept,
/// IEEE_MODE on and off) of the reference programs on the device against
/// the interpreter (which the `hw_valu_denormal_modes` tests hold to the
/// hardware rows); FLOAT_MODE 0xC0 and 0x00 for v_fma_mix_f32.
#[test]
#[ignore]
fn float_ref_f16_denormal_modes_on_device() {
    let cases = denormal_mode_cases();
    for field in 0..4u8 {
        for ieee in [false, true] {
            run_cases_mode(&cases, float_mode(0x30 | field << 6, ieee, false));
        }
    }
    for mode in [0x00u8, 0x40, 0x80, 0xC0] {
        run_cases_mode(&cases, float_mode(mode, false, false));
    }
    run_cases_mode(&f16_conversion_cases(), float_mode(0x30, false, false));
}

/// The directed f16 round modes (FLOAT_MODE[3:2] 1..3) of the f16
/// conversions, v_ldexp_f16 (with FP16_OVFL, f32 round modes nearest and
/// toward zero) and v_fma_mixlo/mixhi_f16 on the device against the
/// interpreter.
#[test]
#[ignore]
fn float_ref_f16_round_modes_on_device() {
    use denormal_modes::{f16_rounding, mix_directed};
    let conversions = f16_conversion_cases();
    for round16 in 1..4u8 {
        for denorm16 in 0..4u8 {
            run_cases_mode(&conversions, float_mode(round16 << 2 | denorm16 << 6, false, false));
        }
    }
    let directed = [Case {
        name: "directed conversions and ldexp",
        code: &f16_rounding::DIRECTED_CODE,
        rows: f16_rounding::DIRECTED_VECTORS.iter().map(|v| v[..4].to_vec()).collect(),
        inputs: 4,
        results: 16,
        stride: 0,
        ulp: 0,
        abs: 0.0,
        checked: None,
    }];
    for round16 in 1..4u8 {
        for overflow in [false, true] {
            for round32 in [0u8, 3] {
                run_cases_mode(&directed, float_mode(0xF0 | round16 << 2 | round32, true, overflow));
            }
        }
    }
    let mix = [Case {
        name: "v_fma_mixlo/mixhi_f16 rounding",
        code: &mix_directed::ROUNDING_CODE,
        rows: mix_directed::DIRECTED_ROWS.iter().map(|r| r[..3].to_vec()).collect(),
        inputs: 4,
        results: 16,
        stride: 4,
        ulp: 0,
        abs: 0.0,
        checked: None,
    }];
    for round16 in 0..4u8 {
        for overflow in [false, true] {
            run_cases_mode(&mix, float_mode(0xC0 | round16 << 2, false, overflow));
        }
    }
}

/// MODE round-field changes at run time (s_round_mode, s_setreg_imm32_b32)
/// before the f16 conversions and v_cvt_f32_i32 / u32 on the device against
/// the interpreter: v4 = input, s_round_mode (or the setreg), v10 = op(v4).
#[test]
#[ignore]
fn float_ref_mode_round_changes_on_device() {
    // v1 = tid * 16; v[4:7] = input; ...; output[tid] = v10.
    let program = |write: &[u32], op: u32| leak([&[0x34020084u32, 0x34040082, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x7e140280][..], write, &[op, 0xe0701000, 0x80010a02, 0xbf810000]].concat());
    let rows = half_boundary_rows();
    let ints: Vec<Vec<u32>> = (0..256u32).map(|i| vec![i.wrapping_mul(0x9E37_79B9) | 1, 0]).collect();
    let mut cases = Vec::new();
    for round in 1..4u32 {
        let f16 = program(&[0xBFA4_0000 | round << 2], 0x7e141504);
        cases.push(Case { name: "s_round_mode; v_cvt_f16_f32", code: f16, rows: rows.clone(), inputs: 4, results: 1, stride: 0, ulp: 0, abs: 0.0, checked: None });
        let setreg = program(&[0xBA80_0881, round], 0x7e141504);
        cases.push(Case { name: "s_setreg_imm32_b32 MODE[3:2]; v_cvt_f16_f32", code: setreg, rows: rows.clone(), inputs: 4, results: 1, stride: 0, ulp: 0, abs: 0.0, checked: None });
        for (name, op) in [("s_round_mode; v_cvt_f32_i32", 0x7e140b04u32), ("s_round_mode; v_cvt_f32_u32", 0x7e140d04)] {
            cases.push(Case { name, code: program(&[0xBFA4_0000 | round], op), rows: ints.clone(), inputs: 4, results: 1, stride: 0, ulp: 0, abs: 0.0, checked: None });
        }
    }
    run_cases_mode(&cases, float_mode(0xC0, false, false));
}

#[allow(dead_code, clippy::all)]
mod sin_cos_near_zero {
    include!("../../../../ps5shader/tests/vectors/valu_sin_cos_near_zero.rs");
    pub fn code() -> &'static [u32] {
        &CODE
    }
    pub fn rows() -> &'static [[u32; 4]] {
        &ROWS
    }
    pub fn expected() -> &'static [[u32; 4]] {
        &EXPECTED
    }
    pub fn tolerance() -> &'static [u32] {
        &TOLERANCE
    }
}

/// v_sin / v_cos (f32 and f16) on the device against the hardware rows
/// near the zeros of the functions (within their ULP tolerance; zeros,
/// infinities and NaNs exact), and over a sweep of two turns against the
/// interpreter (Rust's sine) within 8 ulp: a host sine / cosine that loses
/// precision takes the series in turns (float rule SIN_COS_SERIES), which
/// it then keeps within 4 ulp of, and the series is within 4 ulp of the
/// sine. Without the series a driver's MUFU-style sine is hundreds of ulp
/// off near the zeros.
#[test]
#[ignore]
fn float_ref_sin_cos_on_device() {
    let (rows, expected, tolerance) = (sin_cos_near_zero::rows(), sin_cos_near_zero::expected(), sin_cos_near_zero::tolerance());
    let ordered = |bits: u32, width: u32| {
        let sign = 1u32 << (width - 1);
        if bits & sign != 0 {
            sign - (bits & !sign)
        } else {
            bits | sign
        }
    };
    let mut g = Guest::new(256 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    let input: Vec<u32> = rows.iter().flatten().copied().collect();
    let out = execute(&mut g, &mut rig, sin_cos_near_zero::code(), 32, 32, &input, 32 * 4, 0, 0xC0);
    let mut fails = Vec::new();
    for tid in 0..32 {
        for column in 0..4 {
            let width = if column < 2 { 32 } else { 16 };
            let mask: u32 = if width == 16 { 0xffff } else { 0xffff_ffff };
            let (a, e) = (out[tid * 4 + column] & mask, expected[tid][column] & mask);
            let magnitude = e & (mask >> 1);
            let pinned = magnitude == 0 || magnitude >= if width == 16 { 0x7c00 } else { 0x7f80_0000 };
            let ok = if pinned { a == e } else { ordered(a, width).abs_diff(ordered(e, width)) <= tolerance[tid] };
            if !ok {
                fails.push(format!("near zero lane {tid} column {column} of {:#010x}: device {a:#010x}, hardware {e:#010x} (tolerance {})", rows[tid][column / 2], tolerance[tid]));
            }
        }
    }
    assert!(fails.is_empty(), "{} mismatches:\n{}", fails.len(), fails.join("\n"));
    // v1 = tid * 4; v4 = input[tid]; v10 = sin(v4); v11 = cos(v4).
    let code = leak(vec![0x3402_0082, 0xe030_1000, 0x8000_0401, 0xbf8c_3f70, 0x7e14_6b04, 0x7e16_6d04, 0x3406_0083, 0xe070_1000, 0x8001_0a03, 0xe070_1004, 0x8001_0b03, 0xbf81_0000]);
    let sweep: Vec<Vec<u32>> = (0..4096u32).map(|i| vec![(-1.0f32 + i as f32 / 2048.0 + 1.0 / 8192.0).to_bits()]).collect();
    run_cases(&[Case { name: "v_sin_f32 / v_cos_f32 sweep", code, rows: sweep, inputs: 1, results: 2, stride: 0, ulp: 8, abs: 0.0, checked: None }]);
}
