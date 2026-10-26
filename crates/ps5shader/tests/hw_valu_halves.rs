// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! 16-bit VALU semantics from the reference driver's execution tests
//! (MulLoU16HighHalf, Float16MinMaxBits, VCvtPknormF16SourceModifiers,
//! VCmpF32Denormals, VCmpEqF16Clamp, F16ToF32Unpack, Dpp8PkFmac,
//! Float16Native, PackedFloat16Native), with their expected values computed
//! the same way (or their measured tables).

use ps5shader::interp::{Machine, Memory};
use ps5shader::ssa::{Function, SystemValue};
use ps5shader::testing::{buffer_descriptor, ComputeRun};
use ps5shader::translate::FloatMode;

include!("vectors/exec_programs.rs");

use exec_programs::{self as ep, SplitMix};

const IN: u64 = 0x1_0000_0000;
const OUT: u64 = 0x2_0000_0000;
const POISON: u32 = 0xdead_beef;

const RAW: u32 = 0x3101_6fac;
const STRIDED: u32 = 0x1101_6fac;

fn raw(base: u64, words: usize) -> [u32; 4] {
    buffer_descriptor(base, 0, words as u32 * 4, RAW)
}

fn strided(base: u64, words: usize) -> [u32; 4] {
    buffer_descriptor(base, 4, words as u32, STRIDED)
}

const IEEE: FloatMode = FloatMode { mode: 0xC0, dx10_clamp: true, ieee: true, fp16_overflow: false };

fn mode(mode: u8, dx10_clamp: bool, ieee: bool, fp16_overflow: bool) -> FloatMode {
    FloatMode { mode, dx10_clamp, ieee, fp16_overflow }
}

/// A compute program compiled once (optimized and not) and run on
/// work-groups of `threads` invocations in waves of `wave`.
struct Program {
    run: ComputeRun,
    functions: [Function; 2],
}

impl Program {
    fn new(code: &[u32], wave: u32, user: Vec<u32>, float_mode: FloatMode, workgroup_ids: u32) -> Self {
        let mut run = ComputeRun::new(code.to_vec(), wave, wave, user);
        run.float_mode = float_mode;
        run.workgroup_id_sgprs = workgroup_ids;
        let functions = [run.compile(false).unwrap(), run.compile(true).unwrap()];
        Program { run, functions }
    }

    /// Run one work-group (`group` = work-group id x) over `input` at IN;
    /// returns `out_words` words at OUT. The optimized and unoptimized
    /// programs must agree.
    fn dispatch(&self, threads: u32, group: u32, input: &[u32], out_words: usize) -> Vec<u32> {
        self.dispatch_groups(threads, group..group + 1, input, out_words)
    }

    /// [`Self::dispatch`] of consecutive work-groups over one memory.
    fn dispatch_groups(&self, threads: u32, groups: std::ops::Range<u32>, input: &[u32], out_words: usize) -> Vec<u32> {
        let mut results = Vec::new();
        for f in &self.functions {
            let mut memory = Memory::default();
            memory.write_words(IN, input);
            memory.write_words(OUT, &vec![POISON; out_words]);
            memory.write_words(self.run.code_address, &self.run.code);
            let wave = self.run.wave_size;
            for group in groups.clone() {
                let mut first = 0;
                while first < threads {
                    let lanes = wave.min(threads - first);
                    let mut m = Machine::new(wave, lanes);
                    m.user_data = self.run.user_data.clone();
                    m.memory = std::mem::take(&mut memory);
                    m.system.insert(SystemValue::LocalInvocationId(0), (first..first + lanes).collect());
                    m.system.insert(SystemValue::LocalInvocationId(1), vec![0; lanes as usize]);
                    m.system.insert(SystemValue::LocalInvocationId(2), vec![0; lanes as usize]);
                    m.system.insert(SystemValue::WorkgroupId(0), vec![group; lanes as usize]);
                    m.run(f).unwrap();
                    memory = std::mem::take(&mut m.memory);
                    first += lanes;
                }
            }
            results.push(memory.read_words(OUT, out_words));
        }
        let [plain, opt]: [Vec<u32>; 2] = results.try_into().unwrap();
        if let Some(i) = plain.iter().zip(&opt).position(|(a, b)| a != b) {
            panic!("optimized run differs at word {i}: {:#010x} vs {:#010x}", plain[i], opt[i]);
        }
        opt
    }
}

fn assert_clean(failures: Vec<String>) {
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.iter().take(80).cloned().collect::<Vec<_>>().join("\n"));
}

mod mul_lo_u16 {
    use super::*;

    fn low16(a: u32, b: u32) -> u32 {
        ((a & 0xffff) * (b & 0xffff)) & 0xffff
    }

    /// Products into registers whose high half is zero, holds another value,
    /// or comes from a cleared register, and packed products of halves.
    #[test]
    fn mul_lo_u16_high_half() {
        let input = ep::mul_lo_u16_input();
        type Want = fn(u32, u32, u32) -> u32;
        let wants: [(&str, Want); 6] = [
            ("zero high", |a, b, _| low16(a, b)),
            ("add then mul", |a, b, c| low16(c.wrapping_add(a), b)),
            ("kept high", |a, b, c| (c & 0xffff_0000) | low16(a, b)),
            ("masked factors", |a, b, _| low16(a, b)),
            ("packed low", |a, b, _| low16(a, b)),
            ("packed high", |a, b, _| low16(a >> 16, b >> 16) << 16),
        ];
        let mut failures = Vec::new();
        for ((name, want), code) in wants.into_iter().zip(ep::mul_lo_u16()) {
            let mut user = strided(IN, 256).to_vec();
            user.extend(strided(OUT, 64));
            let out = Program::new(&code, 32, user, FloatMode::default(), 0).dispatch(64, 0, &input, 64);
            for tid in 0..64 {
                let i = &input[tid * 4..tid * 4 + 3];
                let w = want(i[0], i[1], i[2]);
                if out[tid] != w {
                    failures.push(format!("{name}: thread {tid} ({:08x?}) wrote {:#010x}, expected {w:#010x}", i, out[tid]));
                }
            }
        }
        assert_clean(failures);
    }
}

mod f16_min_max {
    use super::*;


    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        Min,
        Max,
        Min3,
        Max3,
        Med3,
    }

    #[derive(Clone, Copy, Default)]
    struct Src {
        reg: usize,
        op_sel: bool,
        op_sel_hi: bool,
        abs: bool,
        neg: bool,
        neg_hi: bool,
    }

    const fn s(reg: usize) -> Src {
        Src { reg, op_sel: false, op_sel_hi: true, abs: false, neg: false, neg_hi: false }
    }

    struct Check {
        name: &'static str,
        kind: Kind,
        src: [Src; 3],
        clamp: bool,
    }

    fn scalar_checks() -> Vec<Check> {
        let mut neg_abs4 = s(0);
        neg_abs4.abs = true;
        neg_abs4.neg = true;
        let mut neg6 = s(2);
        neg6.neg = true;
        let mut hi4 = s(0);
        hi4.op_sel = true;
        let mut neg5 = s(1);
        neg5.neg = true;
        let mut hi6 = s(2);
        hi6.op_sel = true;
        let mut abs4 = s(0);
        abs4.abs = true;
        let mut hi5 = s(1);
        hi5.op_sel = true;
        let mut neg7 = s(3);
        neg7.neg = true;
        vec![
            Check { name: "v_min_f16 v10, v4, v5", kind: Kind::Min, src: [s(0), s(1), s(0)], clamp: false },
            Check { name: "v_max_f16 v11, v4, v5", kind: Kind::Max, src: [s(0), s(1), s(0)], clamp: false },
            Check { name: "v_max_f16 v12, -|v4|, v6 clamp", kind: Kind::Max, src: [neg_abs4, s(2), s(0)], clamp: true },
            Check { name: "v_min_f16 v13, v5, -v6 clamp", kind: Kind::Min, src: [s(1), neg6, s(0)], clamp: true },
            Check { name: "v_min3_f16 v14, v4, v5, v6", kind: Kind::Min3, src: [s(0), s(1), s(2)], clamp: false },
            Check { name: "v_max3_f16 v15, v4, -v5, v6 op_sel:[1,0,1,0]", kind: Kind::Max3, src: [hi4, neg5, hi6], clamp: false },
            Check { name: "v_med3_f16 v16, v4, v5, v6", kind: Kind::Med3, src: [s(0), s(1), s(2)], clamp: false },
            Check { name: "v_med3_f16 v17, |v4|, v5, -v7 op_sel:[0,1,0,0] clamp", kind: Kind::Med3, src: [abs4, hi5, neg7], clamp: true },
        ]
    }

    fn packed_checks() -> Vec<Check> {
        let a = Src { reg: 0, op_sel: true, op_sel_hi: false, abs: false, neg: true, neg_hi: false };
        let b = Src { reg: 2, op_sel: false, op_sel_hi: true, abs: false, neg: false, neg_hi: true };
        let mut neg7 = s(3);
        neg7.neg = true;
        vec![
            Check { name: "v_pk_min_f16 v10, v4, v5", kind: Kind::Min, src: [s(0), s(1), s(0)], clamp: false },
            Check { name: "v_pk_max_f16 v11, v4, v5", kind: Kind::Max, src: [s(0), s(1), s(0)], clamp: false },
            Check { name: "v_pk_max_f16 v12, v4, v6 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[1,0] neg_hi:[0,1] clamp", kind: Kind::Max, src: [a, b, s(0)], clamp: true },
            Check { name: "v_pk_min_f16 v13, v5, v7 neg_lo:[0,1] clamp", kind: Kind::Min, src: [s(1), neg7, s(0)], clamp: true },
        ]
    }

    fn is_nan(b: u32) -> bool {
        b & 0x7fff > 0x7c00
    }

    fn is_snan(b: u32) -> bool {
        is_nan(b) && b & 0x200 == 0
    }

    fn value(b: u32) -> f64 {
        let sign = if b & 0x8000 != 0 { -1.0 } else { 1.0 };
        let e = (b >> 10) & 0x1f;
        let m = (b & 0x3ff) as f64;
        match e {
            0 => sign * m * 2f64.powi(-24),
            31 if m != 0.0 => f64::NAN,
            31 => sign * f64::INFINITY,
            _ => sign * (m + 1024.0) * 2f64.powi(e as i32 - 25),
        }
    }

    #[derive(Clone, Copy)]
    struct H {
        v: f64,
        b: u32,
    }

    struct Model {
        ieee: bool,
        dx10: bool,
        flush: bool,
    }

    impl Model {
        fn numeric(&self, max: bool, l: H, r: H) -> H {
            let ordered = if max { l.v >= r.v } else { l.v < r.v };
            let both_zero = (l.b | r.b) & 0x7fff == 0;
            let zero = if max { l.b & r.b } else { l.b | r.b };
            H { v: if ordered { l.v } else { r.v }, b: if both_zero { zero } else if ordered { l.b } else { r.b } }
        }

        fn pair(&self, max: bool, l: H, r: H) -> H {
            let n = self.numeric(max, l, r);
            let mut b = if is_nan(l.b) { r.b } else { n.b };
            b = if is_nan(r.b) { l.b } else { b };
            if self.ieee {
                let quiet = |x: u32| x | 0x200;
                b = if is_snan(r.b) { quiet(r.b) } else { b };
                b = if is_snan(l.b) { quiet(l.b) } else { b };
            }
            H { v: if is_nan(r.b) { l.v } else { n.v }, b }
        }

        fn clamp(&self, clamp: bool, b: u32) -> u32 {
            if !clamp {
                return b;
            }
            let mag = b & 0x7fff;
            let zero = b > 0x7fff || mag > 0x7c00;
            let clamped = if zero { 0 } else if mag > 0x3c00 { 0x3c00 } else { b };
            if self.dx10 || mag <= 0x7c00 {
                clamped
            } else {
                b
            }
        }

        fn read(&self, mut b: u32) -> H {
            if self.flush && b & 0x7fff < 0x400 {
                b &= 0x8000;
            }
            H { v: value(b), b }
        }

        fn scalar(&self, c: &Check, regs: &[u32]) -> u32 {
            let args: Vec<H> = c
                .src
                .iter()
                .map(|s| {
                    let mut b = (regs[s.reg] >> if s.op_sel { 16 } else { 0 }) & 0xffff;
                    if s.abs {
                        b &= 0x7fff;
                    }
                    if s.neg {
                        b ^= 0x8000;
                    }
                    self.read(b)
                })
                .collect();
            let r = match c.kind {
                Kind::Min => self.pair(false, args[0], args[1]).b,
                Kind::Max => self.pair(true, args[0], args[1]).b,
                Kind::Min3 => self.pair(false, self.pair(false, args[0], args[1]), args[2]).b,
                Kind::Max3 => self.pair(true, self.pair(true, args[0], args[1]), args[2]).b,
                Kind::Med3 => {
                    let any_nan = args.iter().any(|a| is_nan(a.b));
                    let minimum = self.pair(false, self.pair(false, args[0], args[1]), args[2]);
                    let median = self.numeric(true, self.numeric(false, args[0], args[1]), self.numeric(false, self.numeric(true, args[0], args[1]), args[2]));
                    if any_nan {
                        minimum.b
                    } else {
                        median.b
                    }
                }
            };
            self.clamp(c.clamp, r)
        }

        fn packed(&self, c: &Check, regs: &[u32]) -> u32 {
            let mut result = 0;
            for high in 0..2 {
                let mut bits = [0u32; 2];
                for (i, s) in c.src[..2].iter().enumerate() {
                    let sel = if high != 0 { s.op_sel_hi } else { s.op_sel };
                    bits[i] = (regs[s.reg] >> if sel { 16 } else { 0 }) & 0xffff;
                    if if high != 0 { s.neg_hi } else { s.neg } {
                        bits[i] ^= 0x8000;
                    }
                }
                result |= self.clamp(c.clamp, self.pair(c.kind == Kind::Max, self.read(bits[0]), self.read(bits[1])).b) << (16 * high);
            }
            result
        }
    }


    /// (name, FLOAT_MODE or None for the default, flushes f16 inputs)
    fn run(modes: &[(&str, Option<FloatMode>)]) {
        let mut rng = SplitMix(0x2545_f491_4f6c_dd1d);
        let mut user = raw(IN, 256).to_vec();
        user.extend(raw(OUT, 64 * 8));
        let mut failures = Vec::new();
        // The sequence of inputs is drawn across every mode in order, as
        // in the reference.
        let all = [
            ("no FLOAT_MODE", None),
            ("dx10 ieee", Some(mode(0xc0, true, true, false))),
            ("dx10", Some(mode(0xc0, true, false, false))),
            ("ieee", Some(mode(0xc0, false, true, false))),
            ("none", Some(mode(0xc0, false, false, false))),
            ("f16 denormals flushed, dx10 ieee", Some(mode(0x30, true, true, false))),
            ("f16 input denormals flushed", Some(mode(0xb0, false, false, false))),
        ];
        for (name, fm) in all {
            let selected = modes.iter().any(|m| m.0 == name);
            let model = Model { ieee: fm.is_some_and(|m| m.ieee), dx10: fm.is_none_or(|m| m.dx10_clamp), flush: fm.is_some_and(|m| m.mode & 0x40 == 0) };
            let float_mode = fm.unwrap_or_default();
            let programs = selected.then(|| (Program::new(&ep::F16_MIN_MAX_SCALAR, 32, user.clone(), float_mode, 0), Program::new(&ep::F16_MIN_MAX_PACKED, 32, user.clone(), float_mode, 0)));
            for _ in 0..32 {
                let mut input = vec![0u32; 256];
                for tid in 0..64 {
                    let row = &mut input[tid * 4..tid * 4 + 4];
                    for w in row.iter_mut() {
                        *w = ep::f16_min_max_half(&mut rng) | (ep::f16_min_max_half(&mut rng) << 16);
                    }
                    match tid % 8 {
                        0 => row[1] = row[0],
                        1 => row[1] = row[0] ^ 0x8000_8000,
                        2 => row[2] = row[1],
                        3 => row[2] = row[0] ^ 0x8000_8000,
                        _ => {}
                    }
                }
                let Some((scalar, packed)) = &programs else { continue };
                for (program, checks, is_packed) in [(scalar, scalar_checks(), false), (packed, packed_checks(), true)] {
                    let out = program.dispatch(64, 0, &input, 64 * 8);
                    for tid in 0..64 {
                        let regs = &input[tid * 4..tid * 4 + 4];
                        for (i, c) in checks.iter().enumerate() {
                            let want = if is_packed { model.packed(c, regs) } else { model.scalar(c, regs) };
                            let got = out[tid * 8 + i];
                            // Scalar results: the low half.
                            let got = if is_packed { got } else { got & 0xffff };
                            if got != want {
                                failures.push(format!("{name}, {} with {:08x?}: {got:#010x}, expected {want:#010x}", c.name, regs));
                            }
                        }
                    }
                }
            }
        }
        assert_clean(failures);
    }

    #[test]
    fn f16_min_max_bits() {
        run(&[("no FLOAT_MODE", None), ("dx10 ieee", None), ("dx10", None), ("ieee", None), ("none", None)]);
    }

    /// FLOAT_MODE with f16 input denormals flushed (0x30, 0xb0).
    #[test]
    fn f16_min_max_bits_flushed_inputs() {
        run(&[("f16 denormals flushed, dx10 ieee", None), ("f16 input denormals flushed", None)]);
    }
}

mod cvt_pknorm {
    use super::*;

    const ROWS: [[u32; 2]; 8] = [
        [0x00000400, 0x00000400],
        [0x000083ff, 0x000083ff],
        [0x00003555, 0x0000d640],
        [0x0000bc00, 0x00003c00],
        [0x00007bff, 0x00000400],
        [0xbeef0000, 0xbeef3c00],
        [0xbeef8000, 0xbeef0000],
        [0xbeef0000, 0xbeeffc00],
    ];
    // (operands, abs, neg, op_sel)
    const VARIANTS: [(&str, u32, u32, u32); 8] =
        [("|v4|, v5", 1, 0, 0), ("v4, |v5|", 2, 0, 0), ("-v4, v5", 0, 1, 0), ("v4, -v5", 0, 2, 0), ("-|v4|, v5", 1, 1, 0), ("v4, -|v5|", 2, 2, 0), ("v4, v5 op_sel:[1,0,0]", 0, 0, 1), ("v4, v5 op_sel:[0,1,0]", 0, 0, 2)];
    const EXPECTED: [[[u32; 8]; 2]; 8] = [
        [[0x00020002, 0xfffe0002, 0x80012aa8, 0x7fff7fff, 0x00027fff, 0x7fff0000, 0x00000000, 0x80010000], [0x00040004, 0x00000004, 0x00005550, 0xffffffff, 0x0004ffff, 0xffff0000, 0x00000000, 0x00000000]],
        [[0x00020002, 0x0002fffe, 0x7fff2aa8, 0x7fff8001, 0x00027fff, 0x7fff0000, 0x00000000, 0x7fff0000], [0x00040004, 0x00040000, 0xffff5550, 0xffff0000, 0x0004ffff, 0xffff0000, 0x00000000, 0xffff0000]],
        [[0x0002fffe, 0xfffe0002, 0x8001d558, 0x7fff7fff, 0x00028001, 0x7fff0000, 0x00000000, 0x80010000], [0x00040000, 0x00000004, 0x00000000, 0xffffffff, 0x00040000, 0xffff0000, 0x00000000, 0x00000000]],
        [[0xfffe0002, 0x0002fffe, 0x7fff2aa8, 0x80018001, 0xfffe7fff, 0x80010000, 0x00000000, 0x7fff0000], [0x00000004, 0x00040000, 0xffff5550, 0x00000000, 0x0000ffff, 0x00000000, 0x00000000, 0xffff0000]],
        [[0x0002fffe, 0xfffefffe, 0x8001d558, 0x7fff8001, 0x00028001, 0x7fff0000, 0x00000000, 0x80010000], [0x00040000, 0x00000000, 0x00000000, 0xffff0000, 0x00040000, 0xffff0000, 0x00000000, 0x00000000]],
        [[0xfffe0002, 0xfffefffe, 0x80012aa8, 0x80018001, 0xfffe7fff, 0x80010000, 0x00000000, 0x80010000], [0x00000004, 0x00000000, 0x00005550, 0x00000000, 0x0000ffff, 0x00000000, 0x00000000, 0x00000000]],
        [[0x00020000, 0xfffe0000, 0x80010000, 0x7fff0000, 0x00020000, 0x7fff8001, 0x00008001, 0x80018001], [0x00040000, 0x00000000, 0x00000000, 0xffff0000, 0x00040000, 0xffff0000, 0x00000000, 0x00000000]],
        [[0x00000002, 0x0000fffe, 0x00002aa8, 0x00008001, 0x00007fff, 0x80010000, 0x80010000, 0x80010000], [0x00000004, 0x00000000, 0x00005550, 0x00000000, 0x0000ffff, 0x00000000, 0x00000000, 0x00000000]],
    ];

    /// v_cvt_pknorm_i16_f16 / v_cvt_pknorm_u16_f16 (VOP3) with abs, neg and
    /// op_sel source modifiers, f16 denormals kept or flushed in FLOAT_MODE.
    #[test]
    fn cvt_pknorm_f16_source_modifiers() {
        let mut input = vec![0u32; 32 * 4];
        for (i, r) in ROWS.iter().enumerate() {
            input[i * 4..i * 4 + 2].copy_from_slice(r);
        }
        let mut user = raw(IN, 128).to_vec();
        user.extend(raw(OUT, 64));
        let mut failures = Vec::new();
        for (v, &(operands, abs, neg, op_sel)) in VARIANTS.iter().enumerate() {
            let code = ep::cvt_pknorm(abs, neg, op_sel);
            for (mode_name, fm) in [("FLOAT_MODE 0xf0", mode(0xf0, true, true, false)), ("FLOAT_MODE 0xc0", mode(0xc0, true, false, false))] {
                let out = Program::new(&code, 32, user.clone(), fm, 0).dispatch(32, 0, &input, 64);
                for slot in 0..2 {
                    for row in 0..8 {
                        let (got, want) = (out[row * 2 + slot], EXPECTED[v][slot][row]);
                        if got != want {
                            let op = ["v_cvt_pknorm_i16_f16", "v_cvt_pknorm_u16_f16"][slot];
                            failures.push(format!("{mode_name} row {row} {op} {operands} of {:08x?}: {got:#010x}, expected {want:#010x}", ROWS[row]));
                        }
                    }
                }
            }
        }
        assert_clean(failures);
    }
}

mod cmp_f32_denormals {
    use super::*;

    const VALUES: [u32; 16] = ep::CMP_F32_VALUES;
    // (condition, cmpx, cmp, kept, flushed)
    type Row = (&'static str, u32, u32, [u32; 8], [u32; 8]);
    const ROWS: [Row; 12] = [
        ("eq", 0x7c240b04, 0x7c040b04, [0x00030003, 0x00080004, 0x00200010, 0x00800040, 0x02000100, 0x08000400, 0x00000000, 0x80000000], [0x003f003f, 0x003f003f, 0x003f003f, 0x00800040, 0x02000100, 0x08000400, 0x00000000, 0x80000000]),
        ("ge", 0x7c2c0b04, 0x7c0c0b04, [0x0aab0aab, 0x0aa80aaf, 0x0aa00abf, 0x0a800aff, 0x0a000bff, 0x08008fff, 0x00000000, 0x8bff0000], [0x0abf0abf, 0x0abf0abf, 0x0abf0abf, 0x0a800aff, 0x0a000bff, 0x08008fff, 0x00000000, 0x8bff0000]),
        ("gt", 0x7c280b04, 0x7c080b04, [0x0aa80aa8, 0x0aa00aab, 0x0a800aaf, 0x0a000abf, 0x08000aff, 0x00008bff, 0x00000000, 0x0bff0000], [0x0a800a80, 0x0a800a80, 0x0a800a80, 0x0a000abf, 0x08000aff, 0x00008bff, 0x00000000, 0x0bff0000]),
        ("le", 0x7c260b04, 0x7c060b04, [0x85578557, 0x855f8554, 0x857f8550, 0x85ff8540, 0x87ff8500, 0x8fff0400, 0x00000000, 0x84000000], [0x857f857f, 0x857f857f, 0x857f857f, 0x85ff8540, 0x87ff8500, 0x8fff0400, 0x00000000, 0x84000000]),
        ("lg", 0x7c2a0b04, 0x7c0a0b04, [0x8ffc8ffc, 0x8ff78ffb, 0x8fdf8fef, 0x8f7f8fbf, 0x8dff8eff, 0x87ff8bff, 0x00000000, 0x0fff0000], [0x8fc08fc0, 0x8fc08fc0, 0x8fc08fc0, 0x8f7f8fbf, 0x8dff8eff, 0x87ff8bff, 0x00000000, 0x0fff0000]),
        ("lt", 0x7c220b04, 0x7c020b04, [0x85548554, 0x85578550, 0x855f8540, 0x857f8500, 0x85ff8400, 0x87ff0000, 0x00000000, 0x04000000], [0x85408540, 0x85408540, 0x85408540, 0x857f8500, 0x85ff8400, 0x87ff0000, 0x00000000, 0x04000000]),
        ("neq", 0x7c3a0b04, 0x7c1a0b04, [0xfffcfffc, 0xfff7fffb, 0xffdfffef, 0xff7fffbf, 0xfdfffeff, 0xf7fffbff, 0xffffffff, 0x7fffffff], [0xffc0ffc0, 0xffc0ffc0, 0xffc0ffc0, 0xff7fffbf, 0xfdfffeff, 0xf7fffbff, 0xffffffff, 0x7fffffff]),
        ("nge", 0x7c320b04, 0x7c120b04, [0xf554f554, 0xf557f550, 0xf55ff540, 0xf57ff500, 0xf5fff400, 0xf7ff7000, 0xffffffff, 0x7400ffff], [0xf540f540, 0xf540f540, 0xf540f540, 0xf57ff500, 0xf5fff400, 0xf7ff7000, 0xffffffff, 0x7400ffff]),
        ("ngt", 0x7c360b04, 0x7c160b04, [0xf557f557, 0xf55ff554, 0xf57ff550, 0xf5fff540, 0xf7fff500, 0xffff7400, 0xffffffff, 0xf400ffff], [0xf57ff57f, 0xf57ff57f, 0xf57ff57f, 0xf5fff540, 0xf7fff500, 0xffff7400, 0xffffffff, 0xf400ffff]),
        ("nle", 0x7c380b04, 0x7c180b04, [0x7aa87aa8, 0x7aa07aab, 0x7a807aaf, 0x7a007abf, 0x78007aff, 0x7000fbff, 0xffffffff, 0x7bffffff], [0x7a807a80, 0x7a807a80, 0x7a807a80, 0x7a007abf, 0x78007aff, 0x7000fbff, 0xffffffff, 0x7bffffff]),
        ("nlg", 0x7c340b04, 0x7c140b04, [0x70037003, 0x70087004, 0x70207010, 0x70807040, 0x72007100, 0x78007400, 0xffffffff, 0xf000ffff], [0x703f703f, 0x703f703f, 0x703f703f, 0x70807040, 0x72007100, 0x78007400, 0xffffffff, 0xf000ffff]),
        ("nlt", 0x7c3c0b04, 0x7c1c0b04, [0x7aab7aab, 0x7aa87aaf, 0x7aa07abf, 0x7a807aff, 0x7a007bff, 0x7800ffff, 0xffffffff, 0xfbffffff], [0x7abf7abf, 0x7abf7abf, 0x7abf7abf, 0x7a807aff, 0x7a007bff, 0x7800ffff, 0xffffffff, 0xfbffffff]),
    ];

    /// v_cmp_*_f32 (VCC captured with s_mov_b32 s21, vcc_lo) and
    /// v_cmpx_*_f32 (EXEC captured) of every pair of 16 values, f32
    /// denormals kept (FLOAT_MODE 0xf0) or flushed (0xc0), IEEE on or off.
    #[test]
    fn cmp_f32_denormals() {
        let mut user = raw(IN, 128).to_vec();
        user.extend(raw(OUT, 32));
        let mut failures = Vec::new();
        for (cond, cmpx, cmp, kept, flushed) in ROWS {
            for exec in [true, false] {
                let code = ep::cmp_f32(if exec { cmpx } else { cmp }, exec);
                let name = format!("v_cmp{}_{cond}_f32", if exec { "x" } else { "" });
                for (mode_name, fm, expected) in [
                    ("denormals kept, IEEE on", mode(0xf0, true, true, false), kept),
                    ("denormals kept, IEEE off", mode(0xf0, false, false, false), kept),
                    ("denormals flushed, IEEE on", mode(0xc0, true, true, false), flushed),
                    ("denormals flushed, IEEE off", mode(0xc0, false, false, false), flushed),
                ] {
                    let program = Program::new(&code, 32, user.clone(), fm, 0);
                    for (group, &want) in expected.iter().enumerate() {
                        let mut input = vec![0u32; 128];
                        for lane in 0..32 {
                            let pair = group * 32 + lane;
                            input[lane * 4] = VALUES[pair / 16];
                            input[lane * 4 + 1] = VALUES[pair % 16];
                        }
                        let out = program.dispatch(32, 0, &input, 32);
                        let mut mask = 0u32;
                        for (lane, &o) in out.iter().enumerate() {
                            if o > 1 {
                                failures.push(format!("{name} {mode_name}: lane {lane} stored {o:#x}"));
                            }
                            mask |= (o & 1) << lane;
                        }
                        let diff = mask ^ want;
                        for lane in (0..32).filter(|l| diff >> l & 1 != 0) {
                            let pair = group * 32 + lane;
                            failures.push(format!("{name} {mode_name} of {:#010x} and {:#010x} is {}, expected {}", VALUES[pair / 16], VALUES[pair % 16], mask >> lane & 1, want >> lane & 1));
                        }
                    }
                }
            }
        }
        assert_clean(failures);
    }
}

mod cmp_clamp {
    use super::*;

    fn run(name: &str, code: [u32; 18], fm: FloatMode, rows: &[([u32; 4], u32)]) -> Vec<String> {
        let mut input = vec![0u32; 128];
        for (i, r) in rows.iter().enumerate() {
            input[i * 4..i * 4 + 4].copy_from_slice(&r.0);
        }
        let mut user = strided(IN, 128).to_vec();
        user.extend(strided(OUT, 128));
        let out = Program::new(&code, 32, user, fm, 0).dispatch(32, 0, &input, 128);
        rows.iter().enumerate().filter(|(i, r)| out[i * 4] != r.1).map(|(i, r)| format!("{name} clamp row {i}: {:#010x}, expected {:#010x}", out[i * 4], r.1)).collect()
    }

    /// VOP3 compares with the clamp bit set write their lane bits as
    /// usual.
    #[test]
    fn cmp_clamp_bit() {
        let kept = mode(0xf0, true, true, false);
        let flushed = mode(0xc0, true, false, false);
        let mut failures = run(
            "v_cmp_eq_f16_e64 vcc_lo, v4, v5",
            ep::cmp_clamp(0xca, 4, 5),
            kept,
            &[
                ([0, 0, 0, 0], 1),
                ([0xbc00, 0, 0, 0], 0),
                ([0x3555, 0, 0, 0], 0),
                ([0x0001, 0x8000, 0, 0], 0),
                ([0xfc00, 0x8000, 0, 0], 0),
                ([0x7c01, 0x3c00, 0, 0], 0),
                ([0x6400, 0x3c00, 0, 0], 0),
                ([0xbeef8000, 0xbeefbc00, 0xbeef0000, 0], 0),
            ],
        );
        failures.extend(run(
            "v_cmp_le_f64_e64 vcc_lo, v[4:5], v[6:7]",
            ep::cmp_clamp(0x23, 4, 6),
            flushed,
            &[
                ([0, 0, 0, 0], 1),
                ([0, 0xbff00000, 0, 0xfff00000], 0),
                ([0, 0xc0040000, 0, 0xbff00000], 1),
                ([0, 0x00100000, 1, 0x7ff00000], 0),
                ([0, 0xfff00000, 0, 0x40040000], 1),
                ([1, 0x7ff00000, 0, 0x41e00000], 0),
            ],
        ));
        failures.extend(run(
            "v_cmp_ngt_f64_e64 vcc_lo, v[4:5], v[6:7]",
            ep::cmp_clamp(0x2b, 4, 6),
            kept,
            &[([0, 0, 0, 0], 1), ([0, 0xbff00000, 0, 0xfff00000], 0), ([0, 0xc0040000, 0, 0xbff00000], 1), ([0, 0x00100000, 1, 0x7ff00000], 1)],
        ));
        assert_clean(failures);
    }
}

mod f16_to_f32 {
    use super::*;

    const HALVES: usize = 1 << 16;

    fn half_to_float_bits(h: u32) -> u32 {
        let sign = (h & 0x8000) << 16;
        let e = (h >> 10) & 0x1f;
        let m = h & 0x3ff;
        match e {
            0x1f => sign | 0x7f80_0000 | (m << 13),
            0 => sign | (m as f32 * 2f32.powi(-24)).to_bits(),
            _ => sign | ((e + 112) << 23) | (m << 13),
        }
    }

    fn run(code: &[u32], out_words: usize) -> (Vec<u32>, Vec<u32>) {
        let input: Vec<u32> = (0..HALVES as u32).map(|l| l | ((l ^ 0xa5a5) << 16)).collect();
        let mut user = raw(IN, HALVES).to_vec();
        user.extend(raw(OUT, out_words));
        let program = Program::new(code, 32, user, mode(0xc0, true, false, false), 1);
        let out = program.dispatch_groups(32, 0..(HALVES / 32) as u32, &input, out_words);
        (input, out)
    }

    /// v_cvt_f32_f16 of every half: exact, NaN payloads and signs kept,
    /// subnormals widened (f16 denormals kept, f32 flushed: FLOAT_MODE
    /// 0xc0).
    #[test]
    fn cvt_f32_f16_every_half() {
        let (input, out) = run(&ep::F16_TO_F32_CONVERT, HALVES);
        let failures: Vec<String> =
            (0..HALVES).filter(|&l| out[l] != half_to_float_bits(l as u32)).map(|l| format!("v_cvt_f32_f16 of {:#010x}: {:#010x}, expected {:#010x}", input[l], out[l], half_to_float_bits(l as u32))).collect();
        assert_clean(failures);
    }

    /// v_add_f16 with SDWA src0_sel:BYTE_1 of x and -0: the selected byte as
    /// a half (a subnormal or zero), returned exactly.
    #[test]
    fn add_f16_sdwa_byte_every_half() {
        let (input, out) = run(&ep::F16_TO_F32_BYTE, HALVES);
        let failures: Vec<String> = (0..HALVES)
            .filter(|&l| out[l] & 0xffff != (input[l] >> 8) & 0xff)
            .map(|l| format!("v_add_f16 src0_sel:BYTE_1 of {:#010x} and -0: {:#010x}, expected {:#06x}", input[l], out[l], (input[l] >> 8) & 0xff))
            .collect();
        assert_clean(failures);
    }

    /// v_cmp_u_f16 of every half with itself and v_cmp_lt_f16 against 1.0.
    #[test]
    fn cmp_f16_every_half() {
        let (_, out) = run(&ep::F16_TO_F32_COMPARE, HALVES * 2);
        let mut failures = Vec::new();
        for l in 0..HALVES {
            let v = f32::from_bits(half_to_float_bits(l as u32));
            let (u, lt) = (v.is_nan() as u32, (v < 1.0) as u32);
            if out[l * 2] != u {
                failures.push(format!("v_cmp_u_f16 of {l:#06x} with itself: {}, expected {u}", out[l * 2]));
            }
            if out[l * 2 + 1] != lt {
                failures.push(format!("v_cmp_lt_f16 of {l:#06x} and 1.0: {}, expected {lt}", out[l * 2 + 1]));
            }
        }
        assert_clean(failures);
    }
}

mod dpp8_pk_fmac {
    use super::*;

    /// v_pk_fmac_f16 with a DPP8 source equals v_mov_b32 DPP8 then
    /// v_pk_fmac_f16, and differs from the unpermuted product somewhere.
    #[test]
    fn dpp8_pk_fmac() {
        let input = ep::dpp8_input();
        let mut user = raw(IN, 256).to_vec();
        user.extend(raw(OUT, 256));
        let mut failures = Vec::new();
        for wave in [32u32, 64] {
            for (what, lanes) in [("reversed groups of eight", [7u32, 6, 5, 4, 3, 2, 1, 0]), ("swapped neighbours", [1, 0, 3, 2, 5, 4, 7, 6])] {
                let out = Program::new(&ep::dpp8_pk_fmac(lanes), wave, user.clone(), FloatMode::default(), 0).dispatch(wave, 0, &input, 256);
                let mut permuted = false;
                for lane in 0..wave as usize {
                    let (fused, separate, plain) = (out[lane * 4], out[lane * 4 + 1], out[lane * 4 + 2]);
                    if fused != separate {
                        failures.push(format!("{what} (wave{wave}) lane {lane}: dpp8 v_pk_fmac_f16 {fused:#010x}, v_mov dpp8 then v_pk_fmac_f16 {separate:#010x}"));
                    }
                    permuted |= fused != plain;
                }
                if !permuted {
                    failures.push(format!("{what} (wave{wave}): the dpp8 result equals the unpermuted one in every lane"));
                }
            }
        }
        assert_clean(failures);
    }
}

mod f16_native {
    use super::*;


    pub fn modes() -> Vec<(&'static str, FloatMode)> {
        vec![
            ("dx10 ieee ovfl", mode(0xc0, true, true, true)),
            ("dx10 ieee", mode(0xc0, true, true, false)),
            ("dx10 ovfl", mode(0xc0, true, false, true)),
            ("dx10", mode(0xc0, true, false, false)),
            ("ieee ovfl", mode(0xc0, false, true, true)),
            ("ieee", IEEE),
            ("ovfl", mode(0xc0, false, false, true)),
            ("none", mode(0xc0, false, false, false)),
        ]
    }


    /// The f16 add/sub/mul/fma/fmac/fmamk/fmaak forms with modifiers, and
    /// the packed ones, over every float mode: the optimized and
    /// unoptimized programs agree (the device comparison is
    /// `ps5gpu` float_ref_f16_native_on_device).
    #[test]
    fn f16_native_programs_run() {
        let mut rng = SplitMix(0x9e37_79b9_7f4a_7c15);
        for (code, results) in [(&ep::F16_NATIVE_SCALAR[..], 16usize), (&ep::F16_NATIVE_PACKED[..], 8)] {
            for (_, fm) in modes() {
                let mut user = raw(IN, 256).to_vec();
                user.extend(raw(OUT, 64 * results));
                let program = Program::new(code, 32, user, fm, 0);
                for _ in 0..4 {
                    let input: Vec<u32> = (0..256).map(|_| ep::f16_native_half(&mut rng) | (ep::f16_native_half(&mut rng) << 16)).collect();
                    let out = program.dispatch(64, 0, &input, 64 * results);
                    assert!(out.iter().all(|&w| w != POISON), "a result was not written");
                }
            }
        }
    }
}
