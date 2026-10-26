// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Math library HLE: cos, sin, tan, exp, log, pow, sqrt, sincos, etc.
//!
//! Guest code runs natively, so these are only used when no loaded module
//! exports the function (`register_fallback_fn`). They follow the SysV ABI
//! of the C prototypes: floating-point arguments in xmm0/xmm1, pointers in
//! rdi/rsi, the result in xmm0 (`float` in its low 32 bits).

use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};

fn arg_f64(i: usize) -> f64 {
    f64::from_bits(ps5cpu::exec::trap_xmm_low(i).unwrap_or(0))
}

fn arg_f32(i: usize) -> f32 {
    f32::from_bits(ps5cpu::exec::trap_xmm_low(i).unwrap_or(0) as u32)
}

macro_rules! unary {
    ($name:ident, $name_f:ident, $op:expr) => {
        fn $name(ctx: &mut CpuContext) -> HleResult {
            let f: fn(f64) -> f64 = $op;
            ctx.set_return_f64(f(arg_f64(0)));
            0
        }
        fn $name_f(ctx: &mut CpuContext) -> HleResult {
            let f: fn(f32) -> f32 = $op;
            ctx.set_return_f32(f(arg_f32(0)));
            0
        }
    };
}

macro_rules! binary {
    ($name:ident, $name_f:ident, $op:expr) => {
        fn $name(ctx: &mut CpuContext) -> HleResult {
            let f: fn(f64, f64) -> f64 = $op;
            ctx.set_return_f64(f(arg_f64(0), arg_f64(1)));
            0
        }
        fn $name_f(ctx: &mut CpuContext) -> HleResult {
            let f: fn(f32, f32) -> f32 = $op;
            ctx.set_return_f32(f(arg_f32(0), arg_f32(1)));
            0
        }
    };
}

unary!(cb_cos, cb_cosf, |x| x.cos());
unary!(cb_sin, cb_sinf, |x| x.sin());
unary!(cb_tan, cb_tanf, |x| x.tan());
unary!(cb_sqrt, cb_sqrtf, |x| x.sqrt());
unary!(cb_exp, cb_expf, |x| x.exp());

/// logf as FreeBSD msun's `e_logf.c`, which the games' libc.prx carries
/// (same bits for every argument); errno is not set here.
fn logf(x: f32) -> f32 {
    const LN2_HI: f32 = 6.931_381_225_6e-01;
    const LN2_LO: f32 = 9.058_000_614_5e-06;
    const TWO25: f32 = 3.355_443_200e+07;
    const LG1: f32 = f32::from_bits(0x3F2A_AAAA);
    const LG2: f32 = f32::from_bits(0x3ECC_CE13);
    const LG3: f32 = f32::from_bits(0x3E91_E9EE);
    const LG4: f32 = f32::from_bits(0x3E78_9E26);
    let mut x = x;
    let mut k: i32 = 0;
    let mut ix = x.to_bits() as i32;
    if ix < 0x0080_0000 {
        if ix & 0x7FFF_FFFF == 0 {
            return f32::NEG_INFINITY;
        }
        if ix < 0 {
            return (x - x) / 0.0;
        }
        k -= 25;
        x *= TWO25;
        ix = x.to_bits() as i32;
    }
    if ix >= 0x7F80_0000 {
        return x + x;
    }
    k += (ix >> 23) - 127;
    ix &= 0x007F_FFFF;
    let mut i = (ix + (0x95F64 << 3)) & 0x0080_0000;
    x = f32::from_bits((ix | (i ^ 0x3F80_0000)) as u32);
    k += i >> 23;
    let f = x - 1.0;
    let dk = k as f32;
    if (0x007F_FFFF & (0x8000 + ix)) < 0xC000 {
        if f == 0.0 {
            return if k == 0 { 0.0 } else { dk * LN2_HI + dk * LN2_LO };
        }
        let r = f * f * (0.5 - 0.333_333_333_333_333_33_f32 * f);
        return if k == 0 { f - r } else { dk * LN2_HI - ((r - dk * LN2_LO) - f) };
    }
    let s = f / (2.0 + f);
    let z = s * s;
    i = ix - (0x6147A << 3);
    let w = z * z;
    let j = (0x6B851 << 3) - ix;
    let t1 = w * (LG2 + w * LG4);
    let t2 = z * (LG1 + w * LG3);
    i |= j;
    let r = t2 + t1;
    if i > 0 {
        let hfsq = 0.5 * f * f;
        return if k == 0 { f - (hfsq - s * (hfsq + r)) } else { dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f) };
    }
    if k == 0 {
        f - s * (f - r)
    } else {
        dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
    }
}

/// acosf as FreeBSD msun's `e_acosf.c`, which the games' libc.prx carries
/// (same constants, `pio2_lo` read at run time and the 0xfffff000 mask on
/// `sqrtf(z)`). |x| > 1 gives the default NaN and a NaN comes back quiet;
/// errno is not set here.
fn acosf_msun(x: f32) -> f32 {
    const PI: f32 = 3.141_592_502_6e+00;
    const PIO2_HI: f32 = 1.570_796_251_3e+00;
    let pio2_lo: f32 = std::hint::black_box(7.549_789_415_9e-08);
    let rational = |z: f32| {
        const P0: f32 = 1.666_658_669_7e-01;
        const P1: f32 = -4.274_342_209_1e-02;
        const P2: f32 = -8.656_363_003_0e-03;
        const Q1: f32 = -7.066_296_339_0e-01;
        z * (P0 + z * (P1 + z * P2)) / (1.0 + z * Q1)
    };
    let magnitude = x.to_bits() & 0x7FFF_FFFF;
    if magnitude == 0x3F80_0000 {
        return if x.is_sign_negative() { PI + 2.0 * pio2_lo } else { 0.0 };
    }
    if magnitude > 0x7F80_0000 {
        return f32::from_bits(x.to_bits() | 0x0040_0000);
    }
    if magnitude > 0x3F80_0000 {
        return f32::from_bits(0xFFC0_0000);
    }
    if magnitude < 0x3F00_0000 {
        if magnitude <= 0x3280_0000 {
            return PIO2_HI + pio2_lo;
        }
        return PIO2_HI - (x - (pio2_lo - x * rational(x * x)));
    }
    if x.is_sign_negative() {
        let z = (1.0 + x) * 0.5;
        let s = z.sqrt();
        return PI - 2.0 * (s + (rational(z) * s - pio2_lo));
    }
    let z = (1.0 - x) * 0.5;
    let s = z.sqrt();
    let high = f32::from_bits(s.to_bits() & 0xFFFF_F000);
    let correction = (z - high * high) / (s + high);
    2.0 * (high + (rational(z) * s + correction))
}

/// acosf(x)
fn cb_acosf(ctx: &mut CpuContext) -> HleResult {
    ctx.set_return_f32(acosf_msun(arg_f32(0)));
    0
}
// A NaN comes back as x + x, so a signalling NaN is quietened with its
// payload and sign, as the games' libc.prx does.
unary!(cb_floor, cb_floorf, |x| if x.is_nan() { x + x } else { x.floor() });
unary!(cb_ceil, cb_ceilf, |x| if x.is_nan() { x + x } else { x.ceil() });
unary!(cb_fabs, cb_fabsf, |x| x.abs());
binary!(cb_pow, cb_powf, |x, y| x.powf(y));
binary!(cb_atan2, cb_atan2f, |y, x| y.atan2(x));
/// One-precision unary function: `f64` arguments and results, or `f32`.
macro_rules! unary_one {
    ($name:ident, f64, $op:expr) => {
        fn $name(ctx: &mut CpuContext) -> HleResult {
            let f: fn(f64) -> f64 = $op;
            ctx.set_return_f64(f(arg_f64(0)));
            0
        }
    };
    ($name:ident, f32, $op:expr) => {
        fn $name(ctx: &mut CpuContext) -> HleResult {
            let f: fn(f32) -> f32 = $op;
            ctx.set_return_f32(f(arg_f32(0)));
            0
        }
    };
}

unary!(cb_log1p, cb_log1pf, |x| x.ln_1p());
unary_one!(cb_expm1, f64, |x| x.exp_m1());
unary_one!(cb_trunc, f64, |x| if x.is_nan() { x + x } else { x.trunc() });
unary_one!(cb_sinh, f64, |x| x.sinh());
unary!(cb_cosh, cb_coshf, |x| x.cosh());
unary_one!(cb_tanhf, f32, |x| x.tanh());
unary_one!(cb_tanh, f64, tanh_msun);

/// tanh as FreeBSD msun's `s_tanh.c`, which the games' libc.prx carries:
/// x itself below 2^-28, 1 - 2/(expm1(2|x|) + 2) from 1, -t/(t + 2) with
/// t = expm1(-2|x|) below, and 1 - tiny (`tiny` read at run time) from 22;
/// errno is not set.
fn tanh_msun(x: f64) -> f64 {
    let tiny: f64 = std::hint::black_box(1.0e-300);
    const HUGE: f64 = 1.0e300;
    let jx = (x.to_bits() >> 32) as i32;
    let ix = jx & 0x7FFF_FFFF;
    if ix >= 0x7FF0_0000 {
        return if jx >= 0 { 1.0 / x + 1.0 } else { 1.0 / x - 1.0 };
    }
    let z = if ix < 0x4036_0000 {
        if ix < 0x3E30_0000 && HUGE + x > 1.0 {
            return x;
        }
        if ix >= 0x3FF0_0000 {
            let t = (2.0 * x.abs()).exp_m1();
            1.0 - 2.0 / (t + 2.0)
        } else {
            let t = (-2.0 * x.abs()).exp_m1();
            -t / (t + 2.0)
        }
    } else {
        1.0 - tiny
    };
    if jx >= 0 { z } else { -z }
}
unary!(cb_asinh, cb_asinhf, |x| x.asinh());
unary_one!(cb_log, f64, |x| x.ln());
unary_one!(cb_logf, f32, logf);
unary_one!(cb_acosh, f64, |x| x.acosh());

/// acoshf(x): below 1, negative inputs and negative NaNs included, the
/// result is NaN and errno EDOM, as the games' libc.prx does.
fn cb_acoshf(ctx: &mut CpuContext) -> HleResult {
    let x = arg_f32(0);
    if acoshf_out_of_domain(x) {
        crate::kernel::set_errno(ctx, EDOM);
    }
    ctx.set_return_f32(acoshf(x));
    0
}

const EDOM: i32 = 33;

fn acoshf_out_of_domain(x: f32) -> bool {
    let bits = x.to_bits();
    bits & 0x8000_0000 != 0 || bits & 0x7FFF_FFFF < 0x3F80_0000
}

/// acoshf with FreeBSD msun's `e_acoshf.c` range handling, as the games'
/// libc.prx computes it: log(x) + ln2 from 2^28 up (no overflow of x*x),
/// log(2x - 1/(x + sqrt(x*x - 1))) above 2 (msun logf), log1p(t + sqrt(2t + t*t))
/// with t = x - 1 below; x < 1 and negative inputs are NaN.
fn acoshf(x: f32) -> f32 {
    const LN2: f32 = f32::from_bits(0x3F31_7218);
    let magnitude = x.to_bits() & 0x7FFF_FFFF;
    if acoshf_out_of_domain(x) {
        return (x - x) / (x - x);
    }
    if magnitude >= 0x7F80_0000 {
        return x + x;
    }
    if magnitude == 0x3F80_0000 {
        return 0.0;
    }
    if magnitude >= 0x4D80_0000 {
        return logf(x) + LN2;
    }
    if magnitude > 0x4000_0000 {
        let square = x * x;
        return logf(2.0 * x + (-1.0 / (x + (square - 1.0).sqrt())));
    }
    let t = x - 1.0;
    (t + (2.0 * t + t * t).sqrt()).ln_1p()
}
unary_one!(cb_atanh, f64, |x| x.atanh());

/// nan(tag) / nanf(tag): the default quiet NaN, whatever the tag.
fn cb_nan(ctx: &mut CpuContext) -> HleResult {
    ctx.set_return_f64(f64::from_bits(0x7FF8_0000_0000_0000));
    0
}

fn cb_nanf(ctx: &mut CpuContext) -> HleResult {
    ctx.set_return_f32(f32::from_bits(0x7FC0_0000));
    0
}

/// abs(int): INT_MIN stays INT_MIN (two's complement wrap).
fn cb_abs(ctx: &mut CpuContext) -> HleResult {
    (ctx.rdi() as i32).wrapping_abs() as HleResult
}

/// hypot(x, y): +inf when either operand is infinite, even with a NaN
/// for the other (C Annex F, FreeBSD msun e_hypot.c).
fn hypot(x: f64, y: f64) -> f64 {
    if x.is_infinite() || y.is_infinite() {
        return f64::INFINITY;
    }
    x.hypot(y)
}

fn cb_hypot(ctx: &mut CpuContext) -> HleResult {
    ctx.set_return_f64(hypot(arg_f64(0), arg_f64(1)));
    0
}

/// fma(x, y, z): x * y + z with one rounding.
fn cb_fma(ctx: &mut CpuContext) -> HleResult {
    ctx.set_return_f64(arg_f64(0).mul_add(arg_f64(1), arg_f64(2)));
    0
}
// C fmod: the remainder of x / y with the sign of x (Rust `%`).
binary!(cb_fmod, cb_fmodf, |x, y| x % y);

/// sincos(x, *sin, *cos)
fn cb_sincos(ctx: &mut CpuContext) -> HleResult {
    let x = arg_f64(0);
    let (sin_ptr, cos_ptr) = (ctx.rdi(), ctx.rsi());
    if sin_ptr != 0 {
        ctx.write_u64(sin_ptr, x.sin().to_bits());
    }
    if cos_ptr != 0 {
        ctx.write_u64(cos_ptr, x.cos().to_bits());
    }
    0
}

/// sincosf(x, *sin, *cos)
fn cb_sincosf(ctx: &mut CpuContext) -> HleResult {
    let x = arg_f32(0);
    let (sin_ptr, cos_ptr) = (ctx.rdi(), ctx.rsi());
    if sin_ptr != 0 {
        ctx.write_u32(sin_ptr, x.sin().to_bits());
    }
    if cos_ptr != 0 {
        ctx.write_u32(cos_ptr, x.cos().to_bits());
    }
    0
}

const ERANGE: i32 = 34;

/// nextafter over the bit patterns of one IEEE format, as FreeBSD msun's
/// `s_nextafter.c` / `s_nextafterf.c`, which the games' libc.prx carries
/// (measured bits and errno): a NaN operand (x first) comes back quiet
/// with its payload; a step away from zero gives the smallest subnormal
/// with the sign of `y` and no errno; a step that overflows to infinity
/// or lands on a subnormal or zero is ERANGE. Returns (bits, errno).
macro_rules! next_after {
    ($name:ident, $float:ty, $bits:ty, $sign:expr, $exponent:expr, $quiet:expr) => {
        fn $name(x: $float, y: $float) -> ($float, Option<i32>) {
            let (ux, uy) = (x.to_bits(), y.to_bits());
            let magnitude: $bits = !$sign;
            if ux & magnitude > $exponent || uy & magnitude > $exponent {
                let nan = if ux & magnitude > $exponent { ux } else { uy };
                return (<$float>::from_bits(nan | $quiet), None);
            }
            if x == y {
                return (y, None);
            }
            if ux & magnitude == 0 {
                return (<$float>::from_bits((uy & $sign) | 1), None);
            }
            let toward_zero = (x > 0.0) == (x > y);
            let next = if toward_zero { ux - 1 } else { ux + 1 };
            match next & $exponent {
                e if e == $exponent => (x + x, Some(ERANGE)),
                0 => (<$float>::from_bits(next), Some(ERANGE)),
                _ => (<$float>::from_bits(next), None),
            }
        }
    };
}

next_after!(next_after_f64, f64, u64, 0x8000_0000_0000_0000, 0x7FF0_0000_0000_0000, 0x0008_0000_0000_0000);
next_after!(next_after_f32, f32, u32, 0x8000_0000, 0x7F80_0000, 0x0040_0000);

/// nextafter(x, y): the adjacent double of x in the direction of y.
fn cb_nextafter(ctx: &mut CpuContext) -> HleResult {
    let (value, errno) = next_after_f64(arg_f64(0), arg_f64(1));
    if let Some(e) = errno {
        crate::kernel::set_errno(ctx, e);
    }
    ctx.set_return_f64(value);
    0
}

/// nextafterf(x, y): the adjacent float of x in the direction of y.
fn cb_nextafterf(ctx: &mut CpuContext) -> HleResult {
    let (value, errno) = next_after_f32(arg_f32(0), arg_f32(1));
    if let Some(e) = errno {
        crate::kernel::set_errno(ctx, e);
    }
    ctx.set_return_f32(value);
    0
}

/// lround/llround family (the guest long is 64 bits): halfway cases away
/// from zero. As FreeBSD amd64 converts the rounded value with a plain
/// cast, NaN, the infinities and values outside the 64-bit range give
/// 0x8000000000000000; errno is not touched.
fn round_to_i64(rounded: f64) -> i64 {
    const LIMIT: f64 = 9_223_372_036_854_775_808.0; // 2^63
    if rounded >= -LIMIT && rounded < LIMIT {
        rounded as i64
    } else {
        i64::MIN
    }
}

fn cb_lround(_ctx: &mut CpuContext) -> HleResult {
    round_to_i64(arg_f64(0).round())
}

fn cb_lroundf(_ctx: &mut CpuContext) -> HleResult {
    round_to_i64(arg_f32(0).round() as f64)
}

fn cb_llround(ctx: &mut CpuContext) -> HleResult {
    cb_lround(ctx)
}

fn cb_llroundf(ctx: &mut CpuContext) -> HleResult {
    cb_lroundf(ctx)
}

/// __isinf(double): 1 for either infinity.
fn cb_isinf(_ctx: &mut CpuContext) -> HleResult {
    arg_f64(0).is_infinite() as HleResult
}

/// FreeBSD's fpclassify classes.
const FP_INFINITE: HleResult = 0x01;
const FP_NAN: HleResult = 0x02;
const FP_NORMAL: HleResult = 0x04;
const FP_SUBNORMAL: HleResult = 0x08;
const FP_ZERO: HleResult = 0x10;

/// __fpclassifyf(float) -> FP_* class.
fn cb_fpclassifyf(_ctx: &mut CpuContext) -> HleResult {
    let bits = arg_f32(0).to_bits();
    let (exponent, fraction) = (bits & 0x7F80_0000, bits & 0x007F_FFFF);
    match (exponent, fraction != 0) {
        (0, true) => FP_SUBNORMAL,
        (0, false) => FP_ZERO,
        (0x7F80_0000, true) => FP_NAN,
        (0x7F80_0000, false) => FP_INFINITE,
        _ => FP_NORMAL,
    }
}

/// ldiv(long, long) -> ldiv_t { quot, rem } in rax:rdx (the guest long is
/// 64 bits).
fn cb_ldiv(ctx: &mut CpuContext) -> HleResult {
    let (n, d) = (ctx.rdi() as i64, ctx.rsi() as i64);
    if d == 0 {
        // Undefined in C (the console's division faults); the host must
        // not trap inside the HLE call.
        log::error!("[libc] ldiv({n}, 0): division by zero");
        ctx.set_return_rdx(0);
        return 0;
    }
    ctx.set_return_rdx(n.wrapping_rem(d) as u64);
    n.wrapping_div(d)
}

// ---------------------------------------------------------------------------
// long double: the x87 80-bit extended format, passed in memory on the
// stack and returned in ST0.
// ---------------------------------------------------------------------------

/// The first `long double` argument (the first stack argument slot).
fn arg_x87(ctx: &CpuContext) -> Option<[u8; 10]> {
    let rsp = ctx.get(crate::context::CpuRegister::Rsp);
    ctx.read_bytes(rsp + 8, 10)?.try_into().ok()
}

/// acos of an x87 extended value, computed on the x87 unit (under the
/// caller's x87 control word) as atan2(sqrt((1 + x)(1 - x)), x): NaN
/// outside [-1, 1].
fn acos_x87(x: &[u8; 10]) -> [u8; 10] {
    let mut out = [0u8; 10];
    unsafe {
        std::arch::asm!(
            "fld tbyte ptr [{x}]",
            "fld1",
            "fadd st(0), st(1)",
            "fld1",
            "fsub st(0), st(2)",
            "fmulp st(1), st(0)",
            "fsqrt",
            "fxch st(1)",
            "fpatan",
            "fstp tbyte ptr [{out}]",
            x = in(reg) x.as_ptr(),
            out = in(reg) out.as_mut_ptr(),
            options(nostack),
        );
    }
    out
}

/// frexp of an x87 extended value: the fraction in [0.5, 1) with the sign
/// of `x` and the binary exponent, which is None (left alone) for an
/// infinity or NaN; zero keeps its sign and has exponent 0.
fn frexp_x87(x: &[u8; 10]) -> ([u8; 10], Option<i32>) {
    let mant = u64::from_le_bytes(x[..8].try_into().unwrap());
    let se = u16::from_le_bytes([x[8], x[9]]);
    let (sign, exp) = (se & 0x8000, (se & 0x7FFF) as i32);
    if exp == 0x7FFF {
        return (*x, None);
    }
    if mant == 0 {
        return (*x, Some(0));
    }
    let (mant, e) = if exp == 0 {
        // Denormal: normalise the significand.
        let shift = mant.leading_zeros() as i32;
        (mant << shift, 2 - 0x3FFF - shift)
    } else {
        (mant, exp - 0x3FFE)
    };
    let mut out = [0u8; 10];
    out[..8].copy_from_slice(&mant.to_le_bytes());
    out[8..].copy_from_slice(&(sign | 0x3FFE).to_le_bytes());
    (out, Some(e))
}

/// long double acosl(long double x)
fn cb_acosl(ctx: &mut CpuContext) -> HleResult {
    let x = arg_x87(ctx).unwrap_or_default();
    ctx.set_return_st0(acos_x87(&x));
    0
}

/// long double frexpl(long double x, int* exponent)
fn cb_frexpl(ctx: &mut CpuContext) -> HleResult {
    let x = arg_x87(ctx).unwrap_or_default();
    let (fraction, exponent) = frexp_x87(&x);
    if let Some(e) = exponent {
        ctx.write_u32(ctx.rdi(), e as u32);
    }
    ctx.set_return_st0(fraction);
    0
}

// ---------------------------------------------------------------------------
// Floating-point environment. The guest fenv_t is 32 bytes: the 28-byte
// x87 environment fnstenv stores, then MXCSR at +28. The x87 state and
// MXCSR are the host's own (guest code runs natively), so the console's
// instructions apply directly.
// ---------------------------------------------------------------------------

/// Offset of MXCSR inside the guest fenv_t.
const FENV_MXCSR: u64 = 28;

/// Read MXCSR (stmxcsr).
fn get_mxcsr() -> u32 {
    let mut csr = 0u32;
    unsafe { std::arch::asm!("stmxcsr [{p}]", p = in(reg) &mut csr, options(nostack)) };
    csr
}

/// Write MXCSR (ldmxcsr).
fn set_mxcsr(csr: u32) {
    unsafe { std::arch::asm!("ldmxcsr [{p}]", p = in(reg) &csr, options(nostack)) };
}

/// fegetenv(fenv_t*): store the x87 environment (reloading the control
/// word, as the console does) and MXCSR.
fn cb_fegetenv(ctx: &mut CpuContext) -> HleResult {
    let env = ctx.rdi();
    if env == 0 {
        return 0;
    }
    let csr = get_mxcsr();
    unsafe {
        std::arch::asm!(
            "fnstenv [{env}]",
            "fldcw [{env}]",
            env = in(reg) env,
            options(nostack),
        );
    }
    ctx.write_u32(env + FENV_MXCSR, csr);
    0
}

/// feholdexcept(fenv_t*): store the environment, clear the x87 exception
/// flags (fnclex) and, in MXCSR, the flags and every exception mask (the
/// console's behaviour measured on hardware).
fn cb_feholdexcept(ctx: &mut CpuContext) -> HleResult {
    let env = ctx.rdi();
    if env == 0 {
        return 0;
    }
    let csr = get_mxcsr();
    unsafe {
        std::arch::asm!(
            "fnstenv [{env}]",
            "fnclex",
            env = in(reg) env,
            options(nostack),
        );
    }
    ctx.write_u32(env + FENV_MXCSR, csr);
    set_mxcsr((csr & !0x3F) | 0x1F80);
    0
}

/// The 512-byte state fxsave writes, for MXCSR's supported-bit mask.
#[repr(align(16))]
struct FxState([u32; 128]);

/// feupdateenv(const fenv_t*): install the stored environment, then raise
/// the exceptions that were pending when the call began.
fn cb_feupdateenv(ctx: &mut CpuContext) -> HleResult {
    let env = ctx.rdi();
    if env == 0 {
        return 0;
    }
    let Some(saved_mxcsr) = ctx.read_u32(env + FENV_MXCSR) else { return -1 };
    let mut state = FxState([0; 128]);
    unsafe {
        std::arch::asm!("fxsave [{p}]", p = in(reg) state.0.as_mut_ptr(), options(nostack));
    }
    let supported = if state.0[7] != 0 { state.0[7] } else { 0xFFBF };
    if saved_mxcsr & !supported != 0 {
        log::warn!("feupdateenv: MXCSR bits {saved_mxcsr:#x} unsupported by the host FPU");
        return -1;
    }
    let csr_before = get_mxcsr();
    let mut status_word = [0u8; 2];
    let mut raised = [0u8; 32];
    unsafe {
        std::arch::asm!("fnstsw [{p}]", p = in(reg) status_word.as_mut_ptr(), options(nostack));
    }
    let pending = (u16::from_le_bytes(status_word) as u32 | csr_before) & 0x3F;
    unsafe {
        // Install the saved environment...
        std::arch::asm!("fldenv [{p}]", p = in(reg) env, options(nostack));
        set_mxcsr(saved_mxcsr);
        // ...then raise the exceptions that were pending before it.
        std::arch::asm!("fnstenv [{p}]", p = in(reg) raised.as_mut_ptr(), options(nostack));
    }
    raised[4] |= pending as u8;
    let mxcsr = get_mxcsr() | pending;
    unsafe {
        std::arch::asm!("fldenv [{p}]", p = in(reg) raised.as_ptr(), options(nostack));
        set_mxcsr(mxcsr);
        std::arch::asm!("fwait");
    }
    0
}

type Export = (&'static str, &'static str, fn(&mut CpuContext) -> HleResult);

/// _Divide(x, y) / _FDivide: the CRT's division contract — IEEE-754
/// signed zero, infinity and NaN behaviour. _Recip(x) / _FRecip: 1 / x
/// the same way.
fn cb_dinkum_divide(ctx: &mut CpuContext) -> HleResult {
    ctx.set_return_f64(arg_f64(0) / arg_f64(1));
    0
}

fn cb_dinkum_fdivide(ctx: &mut CpuContext) -> HleResult {
    ctx.set_return_f32(arg_f32(0) / arg_f32(1));
    0
}

fn cb_dinkum_recip(ctx: &mut CpuContext) -> HleResult {
    ctx.set_return_f64(1.0 / arg_f64(0));
    0
}

fn cb_dinkum_frecip(ctx: &mut CpuContext) -> HleResult {
    ctx.set_return_f32(1.0 / arg_f32(0));
    0
}

const EXPORTS: &[Export] = &[
    ("2WE3BTYVwKM", "cos", cb_cos),
    ("-P6FNMzk2Kc", "cosf", cb_cosf),
    ("H8ya2H00jbI", "sin", cb_sin),
    ("Q4rRL34CEeE", "sinf", cb_sinf),
    ("jMB7EFyu30Y", "sincos", cb_sincos),
    ("pztV4AF18iI", "sincosf", cb_sincosf),
    ("T7uyNqP7vQA", "tan", cb_tan),
    ("ZE6RNL+eLbk", "tanf", cb_tanf),
    ("MXRNWnosNlM", "sqrt", cb_sqrt),
    ("Q+xU11-h0xQ", "sqrtf", cb_sqrtf),
    ("NVadfnzQhHQ", "exp", cb_exp),
    ("8zsu04XNsZ4", "expf", cb_expf),
    ("rtV7-jWC6Yg", "log", cb_log),
    ("RQXLbdT2lc4", "logf", cb_logf),
    ("9LCjpWyQ5Zc", "pow", cb_pow),
    ("1D0H2KNjshE", "powf", cb_powf),
    ("mpcTgMzhUY8", "floor", cb_floor),
    ("gacfOmO8hNs", "ceil", cb_ceil),
    ("388LcMWHRCA", "fabs", cb_fabs),
    ("HUbZmOnT-Dg", "atan2", cb_atan2),
    ("pKwslsMUmSk", "fmod", cb_fmod),
    ("3m2ro+Di+Ck", "nextafterf", cb_nextafterf),
    ("h+J60RRlfnk", "nextafter", cb_nextafter),
    ("QI-x0SL8jhw", "acosf", cb_acosf),
    ("eQhBFnTOp40", "llroundf", cb_llroundf),
    ("V02oFv+-JzA", "__isinf", cb_isinf),
    ("z7aCCd9hMsI", "__fpclassifyf", cb_fpclassifyf),
    ("gfP0im5Z3g0", "ldiv", cb_ldiv),
    ("VfsML+n9cDM", "log1p", cb_log1p),
    ("MFe91s8apQk", "log1pf", cb_log1pf),
    ("gqKfOiJaCOo", "expm1", cb_expm1),
    ("a4gLGspPEDM", "trunc", cb_trunc),
    ("YKbL5KR6RDI", "fma", cb_fma),
    ("zck+6bVj5pA", "nan", cb_nan),
    ("DZU+K1wozGI", "nanf", cb_nanf),
    ("Ye20uNnlglA", "abs", cb_abs),
    ("ZjtRqSMJwdw", "sinh", cb_sinh),
    ("m7iLTaO9RMs", "cosh", cb_cosh),
    ("RCQAffkEh9A", "coshf", cb_coshf),
    ("JM4EBvWT9rc", "tanh", cb_tanh),
    ("SAd0Z3wKwLA", "tanhf", cb_tanhf),
    ("2eQpqTjJ5Y4", "asinh", cb_asinh),
    ("yPPtp1RMihw", "asinhf", cb_asinhf),
    ("Fk7-KFKZi-8", "acosh", cb_acosh),
    ("XJp2C-b0tRU", "acoshf", cb_acoshf),
    ("YjbpxXpi6Zk", "atanh", cb_atanh),
    ("YFoOw5GkkK0", "hypot", cb_hypot),
    ("J3XuGS-cC0Q", "lround", cb_lround),
    ("C6gWCWJKM+U", "lroundf", cb_lroundf),
    ("w-BvXF4O6xo", "llround", cb_llround),
    ("iH4YAIRcecA", "acosl", cb_acosl),
    ("YZk9sHO0yNg", "frexpl", cb_frexpl),
    ("psx0YiAKm7k", "fegetenv", cb_fegetenv),
    ("1kduKXMqx7k", "feholdexcept", cb_feholdexcept),
    ("YXQ4gXamCrY", "feupdateenv", cb_feupdateenv),
];

/// Single-precision variants without a registered NID yet are kept for
/// the table below (their NIDs come from the names).
const EXPORTS_BY_NAME: &[(&str, fn(&mut CpuContext) -> HleResult)] = &[
    ("floorf", cb_floorf),
    ("ceilf", cb_ceilf),
    ("fabsf", cb_fabsf),
    ("atan2f", cb_atan2f),
    ("fmodf", cb_fmodf),
];

const DINKUM_EXPORTS: &[Export] = &[
    ("VGhcd0QwhhY", "_Divide", cb_dinkum_divide),
    ("fuzzBVdpRG0", "_FDivide", cb_dinkum_fdivide),
    ("YLEc5sPn1Rg", "_Recip", cb_dinkum_recip),
    ("lP9zfrhtpBc", "_FRecip", cb_dinkum_frecip),
];

/// Register math HLE exports.
pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS.iter().chain(DINKUM_EXPORTS) {
        reg.register_fallback_fn(nid, name, "LibcInternal", cb);
    }
    for &(name, cb) in EXPORTS_BY_NAME {
        let nid: &'static str = Box::leak(ps5nid::compute_nid(name).into_boxed_str());
        reg.register_fallback_fn(nid, name, "LibcInternal", cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps5vm::{MemProtect, VirtualMemory};

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS.iter().chain(DINKUM_EXPORTS) {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    fn call(cb: fn(&mut CpuContext) -> HleResult, xmm: &[u64]) -> u128 {
        let vm = VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        let (_, ret) = ps5cpu::exec::with_import_xmm(xmm, || cb(&mut ctx));
        ret.expect("result in xmm0")
    }

    #[test]
    fn dinkumware_divide_and_recip() {
        let d = |args: &[f64]| f64::from_bits(call(cb_dinkum_divide, &[args[0].to_bits(), args[1].to_bits()]) as u64);
        assert_eq!(d(&[3.0, 2.0]), 1.5);
        let plus = d(&[1.0, 0.0]);
        assert!(plus.is_infinite() && plus.is_sign_positive());
        let minus = d(&[-1.0, 0.0]);
        assert!(minus.is_infinite() && minus.is_sign_negative());
        assert!(d(&[0.0, 0.0]).is_nan());
        let recip = |x: f64| f64::from_bits(call(cb_dinkum_recip, &[x.to_bits()]) as u64);
        assert_eq!(recip(4.0), 0.25);
        let neg = recip(-0.0);
        assert!(neg.is_infinite() && neg.is_sign_negative());
        let fd = |args: &[f32]| f32::from_bits(call(cb_dinkum_fdivide, &[args[0].to_bits() as u64, args[1].to_bits() as u64]) as u32);
        assert_eq!(fd(&[3.0, 2.0]), 1.5);
        let fr = |x: f32| f32::from_bits(call(cb_dinkum_frecip, &[x.to_bits() as u64]) as u32);
        assert_eq!(fr(4.0), 0.25);
        let fneg = fr(-0.0);
        assert!(fneg.is_infinite() && fneg.is_sign_negative());
    }

    fn d(cb: fn(&mut CpuContext) -> HleResult, args: &[f64]) -> f64 {
        let bits: Vec<u64> = args.iter().map(|a| a.to_bits()).collect();
        f64::from_bits(call(cb, &bits) as u64)
    }

    fn f(cb: fn(&mut CpuContext) -> HleResult, args: &[f32]) -> f32 {
        let bits: Vec<u64> = args.iter().map(|a| a.to_bits() as u64).collect();
        f32::from_bits(call(cb, &bits) as u32)
    }

    /// Arguments come from xmm0/xmm1 and results go to xmm0, as the C
    /// prototypes say; results match the host's libm.
    #[test]
    fn float_arguments_and_results_use_xmm_registers() {
        let x = 0.7_f64;
        assert_eq!(d(cb_cos, &[x]), x.cos());
        assert_eq!(d(cb_sin, &[x]), x.sin());
        assert_eq!(d(cb_tan, &[x]), x.tan());
        assert_eq!(d(cb_sqrt, &[2.0]), 2f64.sqrt());
        assert_eq!(d(cb_exp, &[x]), x.exp());
        assert_eq!(d(cb_log, &[x]), x.ln());
        assert_eq!(d(cb_pow, &[1.5, 2.5]), 1.5f64.powf(2.5));
        assert_eq!(d(cb_atan2, &[1.0, -2.0]), 1f64.atan2(-2.0));
        assert_eq!(d(cb_fmod, &[-7.5, 2.0]), -1.5);
        assert_eq!(d(cb_floor, &[-1.5]), -2.0);
        assert_eq!(d(cb_ceil, &[-1.5]), -1.0);
        assert_eq!(d(cb_fabs, &[-3.25]), 3.25);
        let y = 0.7_f32;
        assert_eq!(f(cb_cosf, &[y]), y.cos());
        assert_eq!(f(cb_sinf, &[y]), y.sin());
        assert_eq!(f(cb_tanf, &[y]), y.tan());
        assert_eq!(f(cb_sqrtf, &[2.0]), 2f32.sqrt());
        assert_eq!(f(cb_expf, &[y]), y.exp());
        assert_eq!(f(cb_logf, &[y]), y.ln());
        assert_eq!(f(cb_powf, &[1.5, 2.5]), 1.5f32.powf(2.5));
        assert_eq!(f(cb_fmodf, &[7.5, -2.0]), 1.5);
        // float results leave the upper lanes alone (low 32 bits).
        assert_eq!(call(cb_sinf, &[y.to_bits() as u64]) >> 32, 0);
    }

    fn int(cb: fn(&mut CpuContext) -> HleResult, xmm: &[u64]) -> HleResult {
        let vm = VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        ps5cpu::exec::with_import_xmm(xmm, || cb(&mut ctx)).0
    }

    /// log1p, log1pf and expm1 keep the sign of zero, are exact for tiny
    /// arguments and follow the C special values (reference checks).
    #[test]
    fn log1p_and_expm1() {
        let inf = f64::INFINITY;
        assert_eq!(d(cb_log1p, &[0.0]), 0.0);
        assert!(d(cb_log1p, &[-0.0]).is_sign_negative());
        assert_eq!(d(cb_log1p, &[-1.0]), -inf);
        assert!(d(cb_log1p, &[-2.0]).is_nan());
        assert_eq!(d(cb_log1p, &[inf]), inf);
        assert_eq!(d(cb_log1p, &[1e-300]), 1e-300);
        assert_eq!(f(cb_log1pf, &[1e-30]), 1e-30);
        assert!((d(cb_log1p, &[1.0]) - 0.6931471805599453).abs() < 1e-16);
        assert!((d(cb_log1p, &[1e-10]) - 9.9999999995e-11).abs() < 1e-25);
        assert!((f(cb_log1pf, &[1.0]) - 0.6931472).abs() < 1e-7 && f(cb_log1pf, &[-2.0]).is_nan());
        assert_eq!(d(cb_expm1, &[0.0]), 0.0);
        assert!(d(cb_expm1, &[-0.0]).is_sign_negative());
        assert_eq!(d(cb_expm1, &[1e-300]), 1e-300);
        assert_eq!(d(cb_expm1, &[-1000.0]), -1.0);
        assert_eq!(d(cb_expm1, &[-inf]), -1.0);
        assert!((d(cb_expm1, &[1.0]) - 1.718281828459045).abs() < 1e-15);
        assert!((d(cb_expm1, &[1e-10]) - 1.00000000005e-10).abs() < 1e-25);
    }

    /// fabs, trunc and fma are exact (reference checks).
    #[test]
    fn exact_operations() {
        let inf = f64::INFINITY;
        let nan = f64::from_bits(0xFFF8_0000_0000_0000);
        assert_eq!(d(cb_fabs, &[-3.5]), 3.5);
        assert_eq!(d(cb_fabs, &[-inf]), inf);
        assert_eq!(d(cb_fabs, &[-0.0]).to_bits(), 0);
        assert!(d(cb_fabs, &[nan]).is_nan() && !d(cb_fabs, &[nan]).is_sign_negative());
        assert_eq!(d(cb_trunc, &[2.7]), 2.0);
        assert_eq!(d(cb_trunc, &[-2.7]), -2.0);
        assert_eq!(d(cb_trunc, &[-0.5]).to_bits(), 0x8000_0000_0000_0000);
        assert_eq!(d(cb_trunc, &[4503599627370497.0]), 4503599627370497.0);
        assert_eq!(d(cb_trunc, &[-1e300]), -1e300);
        assert_eq!(d(cb_trunc, &[-inf]), -inf);
        assert!(d(cb_trunc, &[f64::NAN]).is_nan());
        assert_eq!(d(cb_fma, &[2.0, 3.0, 4.0]), 10.0);
        assert_eq!(d(cb_fma, &[0.1, 10.0, -1.0]), 2f64.powi(-54));
        assert_eq!(d(cb_fma, &[2f64.powi(1023), 2.0, -(2f64.powi(1023))]), 2f64.powi(1023));
        assert!(d(cb_fma, &[inf, 0.0, 1.0]).is_nan());
    }

    /// nan and nanf give the positive default quiet NaN for any tag.
    #[test]
    fn nan_ignores_the_tag() {
        const BASE: u64 = 0x7B34_0000_0000;
        let vm = VirtualMemory::new();
        vm.map(BASE, 0x1000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, b"123\0");
        ctx.write_bytes(BASE + 8, b"\0");
        for tag in [0, BASE, BASE + 8] {
            ctx.set(crate::context::CpuRegister::Rdi, tag);
            let (_, r) = ps5cpu::exec::with_import_xmm(&[], || cb_nanf(&mut ctx));
            assert_eq!(r.unwrap() as u32, 0x7FC0_0000);
            let (_, r) = ps5cpu::exec::with_import_xmm(&[], || cb_nan(&mut ctx));
            assert_eq!(r.unwrap() as u64, 0x7FF8_0000_0000_0000);
        }
        vm.unmap(BASE, 0x1000).unwrap();
    }

    /// abs of an int, INT_MIN included (reference checks); only the low
    /// 32 bits of the argument count.
    #[test]
    fn abs_of_int() {
        let vm = VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        let cases: [(u64, i32); 6] = [
            (0, 0),
            (42, 42),
            ((-42i64) as u64, 42),
            (i32::MAX as u64, i32::MAX),
            (i32::MIN as u32 as u64, i32::MIN),
            (0xFFFF_FFFF_0000_0005, 5),
        ];
        for (x, r) in cases {
            ctx.set(crate::context::CpuRegister::Rdi, x);
            assert_eq!(cb_abs(&mut ctx) as i32, r, "abs({x:#x})");
        }
    }

    /// The hyperbolic functions: signs of zero, domain edges, values and
    /// the large-argument ends (reference checks).
    #[test]
    fn hyperbolic() {
        let near = |v: f64, e: f64, tol: f64| (v - e).abs() <= tol * e.abs();
        let inf = f64::INFINITY;
        assert_eq!(d(cb_sinh, &[0.0]), 0.0);
        assert!(d(cb_sinh, &[-0.0]).is_sign_negative());
        assert_eq!(d(cb_cosh, &[0.0]), 1.0);
        assert_eq!(f(cb_coshf, &[0.0]), 1.0);
        assert_eq!(d(cb_asinh, &[0.0]), 0.0);
        assert!(d(cb_asinh, &[-0.0]).is_sign_negative() && f(cb_asinhf, &[-0.0]).is_sign_negative());
        assert_eq!(d(cb_acosh, &[1.0]), 0.0);
        assert_eq!(f(cb_acoshf, &[1.0]), 0.0);
        assert_eq!(d(cb_atanh, &[0.0]), 0.0);
        assert_eq!(d(cb_atanh, &[1.0]), inf);
        assert_eq!(d(cb_atanh, &[-1.0]), -inf);
        assert!(d(cb_acosh, &[0.5]).is_nan() && f(cb_acoshf, &[0.5]).is_nan() && d(cb_atanh, &[2.0]).is_nan());
        assert!(near(d(cb_sinh, &[1.0]), 1.1752011936438014, 1e-15));
        assert!(near(d(cb_cosh, &[1.0]), 1.5430806348152437, 1e-15));
        assert!(near(f(cb_coshf, &[1.0]) as f64, 1.5430806348152437, 1e-6));
        assert!(near(d(cb_asinh, &[1.0]), 0.881373587019543, 1e-15));
        assert!(near(f(cb_asinhf, &[1.0]) as f64, 0.881373587019543, 1e-6));
        assert!(near(d(cb_acosh, &[2.0]), 1.3169578969248166, 1e-15));
        assert!(near(f(cb_acoshf, &[2.0]) as f64, 1.3169578969248166, 1e-6));
        assert!(near(d(cb_atanh, &[0.5]), 0.5493061443340549, 1e-15));
        assert!(near(d(cb_tanh, &[0.5]), 0.46211715726000974, 1e-15));
        assert!(near(f(cb_tanhf, &[0.5]) as f64, 0.46211715726000974, 1e-6));
        assert_eq!(d(cb_tanh, &[inf]), 1.0);
        assert_eq!(d(cb_sinh, &[1e-300]), 1e-300);
        assert_eq!(d(cb_asinh, &[1e-300]), 1e-300);
        assert_eq!(d(cb_atanh, &[1e-300]), 1e-300);
        assert!(d(cb_cosh, &[710.4]).is_finite());
        let s = d(cb_sinh, &[-710.4]);
        assert!(s.is_finite() && s < 0.0);
    }

    /// acoshf: round-to-nearest bits of the games' libc.prx at the msun
    /// range boundaries; FLT_MAX stays finite (x*x is not formed above
    /// 2^28).
    #[test]
    fn acoshf_msun_boundaries() {
        for (input, expected) in [
            (0x3f80_0000u32, 0x0000_0000u32),
            (0x3f80_0001, 0x39ff_ffff),
            (0x3fff_ffff, 0x3fa8_9213),
            (0x4000_0000, 0x3fa8_9214),
            (0x4000_0001, 0x3fa8_9215),
            (0x4000_0002, 0x3fa8_9216),
            (0x4000_0003, 0x3fa8_9217),
            (0x4d7f_ffff, 0x41a0_cf66),
            (0x4d80_0000, 0x41a0_cf66),
            (0x4d80_0001, 0x41a0_cf66),
            (0x4d80_50f2, 0x41a0_d473),
            (0x7f7f_ffff, 0x42b2_d4fc),
        ] {
            let got = f(cb_acoshf, &[f32::from_bits(input)]).to_bits();
            assert_eq!(got, expected, "acoshf({input:#010x}) = {got:#010x}");
        }
        assert_eq!(f(cb_acoshf, &[f32::INFINITY]), f32::INFINITY);
        assert!(f(cb_acoshf, &[-f32::INFINITY]).is_nan() && f(cb_acoshf, &[f32::NAN]).is_nan());
    }

    /// acoshf: the console's bits for out-of-domain inputs (negative NaNs
    /// too), which are the ones that set EDOM; 1, infinity and positive
    /// NaNs are in the domain.
    #[test]
    fn acoshf_domain_bits() {
        for (input, expected, edom) in [
            (0x0000_0000u32, 0xffc0_0000u32, true),
            (0x8000_0000, 0xffc0_0000, true),
            (0x3f00_0000, 0xffc0_0000, true),
            (0x3f7f_ffff, 0xffc0_0000, true),
            (0xbf80_0000, 0xffc0_0000, true),
            (0xff80_0000, 0xffc0_0000, true),
            (0xffc0_0001, 0xffc0_0001, true),
            (0xff80_0001, 0xffc0_0001, true),
            (0x3f80_0000, 0x0000_0000, false),
            (0x7f80_0000, 0x7f80_0000, false),
            (0x7fc0_0001, 0x7fc0_0001, false),
            (0x7f80_0001, 0x7fc0_0001, false),
        ] {
            let x = f32::from_bits(input);
            assert_eq!(acoshf(x).to_bits(), expected, "acoshf({input:#010x})");
            assert_eq!(acoshf_out_of_domain(x), edom, "EDOM for {input:#010x}");
        }
    }

    /// tanh: the bits of the games' libc.prx (msun) at the range edges and
    /// on sampled arguments.
    #[test]
    fn tanh_console_bits() {
        for (input, expected) in [
            (0x0000000000000000u64, 0x0000000000000000u64),
            (0x8000000000000000, 0x8000000000000000),
            (0x3e2fffffffffffff, 0x3e2fffffffffffff),
            (0x3e30000000000000, 0x3e30000000000000),
            (0xbe30000000000000, 0xbe30000000000000),
            (0x3fefffffffffffff, 0x3fe85efab514f394),
            (0x3ff0000000000000, 0x3fe85efab514f394),
            (0xbff0000000000000, 0xbfe85efab514f394),
            (0x3fe0000000000000, 0x3fdd9353d7568af3),
            (0x4035ffffffffffff, 0x3ff0000000000000),
            (0x4036000000000000, 0x3ff0000000000000),
            (0xc036000000000000, 0xbff0000000000000),
            (0x4049000000000000, 0x3ff0000000000000),
            (0x7fefffffffffffff, 0x3ff0000000000000),
            (0x7ff0000000000000, 0x3ff0000000000000),
            (0xfff0000000000000, 0xbff0000000000000),
            (0x7ff8000000000001, 0x7ff8000000000001),
            (0xfff8000000000002, 0xfff8000000000002),
            (0x3f927ed0db8370b1, 0x3f927e4d1092a580),
            (0x3f95d74e4cee78bf, 0x3f95d67547d523a0),
            (0x3f18f1b062cb523c, 0x3f18f1b06187faae),
            (0x3f937129f25ae835, 0x3f937090dd8c9cb3),
            (0x3facb0d89f8aa91c, 0x3faca92afe31a32b),
            (0x3f07f462f020acde, 0x3f07f462efd91530),
        ] {
            let got = d(cb_tanh, &[f64::from_bits(input)]).to_bits();
            assert_eq!(got, expected, "tanh({input:#018x}) = {got:#018x}");
        }
    }

    /// floor, ceil and trunc return a signalling NaN quietened (payload and
    /// sign kept), a quiet NaN and the infinities unchanged.
    #[test]
    fn rounding_quiets_signalling_nans() {
        let doubles = [
            (0x7ff0000000000001u64, 0x7ff8000000000001u64),
            (0xfff4000000000123, 0xfffc000000000123),
            (0x7ff8000000000456, 0x7ff8000000000456),
            (0x7ff0000000000000, 0x7ff0000000000000),
            (0xfff0000000000000, 0xfff0000000000000),
        ];
        for cb in [cb_floor, cb_ceil, cb_trunc] {
            for (input, expected) in doubles {
                assert_eq!(d(cb, &[f64::from_bits(input)]).to_bits(), expected, "{input:#018x}");
            }
        }
        let floats = [
            (0x7f800001u32, 0x7fc00001u32),
            (0xffa00123, 0xffe00123),
            (0x7fc00456, 0x7fc00456),
            (0x7f800000, 0x7f800000),
            (0xff800000, 0xff800000),
        ];
        for cb in [cb_floorf, cb_ceilf] {
            for (input, expected) in floats {
                assert_eq!(f(cb, &[f32::from_bits(input)]).to_bits(), expected, "{input:#010x}");
            }
        }
    }

    /// hypot: an infinite operand wins over a NaN (quiet or signalling, of
    /// either sign); other inputs follow the host's hypot.
    #[test]
    fn hypot_infinity_beats_nan() {
        let inf = f64::INFINITY;
        let snan = f64::from_bits(0x7FF0_0000_0000_0001);
        for (x, y) in [(inf, f64::NAN), (f64::NAN, -inf), (-inf, snan), (snan, inf), (inf, 3.0)] {
            assert_eq!(d(cb_hypot, &[x, y]), inf, "hypot({x}, {y})");
        }
        assert!(d(cb_hypot, &[f64::NAN, 3.0]).is_nan());
        assert_eq!(d(cb_hypot, &[3.0, 4.0]), 5.0);
        assert_eq!(d(cb_hypot, &[-0.0, 0.0]).to_bits(), 0);
        assert_eq!(d(cb_hypot, &[1e300, 1e300]), 1e300f64.hypot(1e300));
    }

    /// lround, lroundf, llround and llroundf: halfway away from zero; NaN,
    /// infinities and out-of-range values give LONG_MIN (reference edge
    /// cases).
    #[test]
    fn lround_family_out_of_range_is_long_min() {
        let dl = |v: f64| v.to_bits();
        let fl = |v: f32| v.to_bits() as u64;
        for cb in [cb_lround as fn(&mut CpuContext) -> HleResult, cb_llround] {
            assert_eq!(int(cb, &[dl(2.5)]), 3);
            assert_eq!(int(cb, &[dl(-2.5)]), -3);
            assert_eq!(int(cb, &[dl(-0.4)]), 0);
            assert_eq!(int(cb, &[dl(9007199254740991.0)]), 9007199254740991);
            assert_eq!(int(cb, &[dl(-9223372036854775808.0)]), i64::MIN);
            assert_eq!(int(cb, &[dl(9223372036854774784.0)]), 9223372036854774784);
            for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 9223372036854775808.0, -9223372036854777856.0, 1e300] {
                assert_eq!(int(cb, &[dl(bad)]), i64::MIN, "{bad}");
            }
        }
        for cb in [cb_lroundf as fn(&mut CpuContext) -> HleResult, cb_llroundf] {
            assert_eq!(int(cb, &[fl(2.5)]), 3);
            assert_eq!(int(cb, &[fl(-2.5)]), -3);
            assert_eq!(int(cb, &[fl(-(2f32.powi(63)))]), i64::MIN);
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2f32.powi(63), 1e30] {
                assert_eq!(int(cb, &[fl(bad)]), i64::MIN, "{bad}");
            }
        }
    }

    #[test]
    fn nextafterf_steps_to_the_adjacent_float() {
        let denorm = f32::from_bits(1);
        assert_eq!(f(cb_nextafterf, &[1.0, 2.0]), f32::from_bits(0x3F80_0001));
        assert_eq!(f(cb_nextafterf, &[1.0, 0.0]), f32::from_bits(0x3F7F_FFFF));
        assert_eq!(f(cb_nextafterf, &[0.0, 1.0]), denorm);
        assert_eq!(f(cb_nextafterf, &[0.0, -1.0]), -denorm);
        assert_eq!(f(cb_nextafterf, &[denorm, -1.0]), 0.0);
        assert_eq!(f(cb_nextafterf, &[-1.0, -2.0]), f32::from_bits(0xBF80_0001));
        assert_eq!(f(cb_nextafterf, &[1.0, 1.0]), 1.0);
        assert!(!f(cb_nextafterf, &[-0.0, 0.0]).is_sign_negative());
        assert_eq!(f(cb_nextafterf, &[f32::MAX, f32::INFINITY]), f32::INFINITY);
        assert!(f(cb_nextafterf, &[f32::NAN, 1.0]).is_nan() && f(cb_nextafterf, &[1.0, f32::NAN]).is_nan());
    }

    /// acosf gives the bits of the games' libc.prx.
    #[test]
    fn acosf_console_bits() {
        for (x, want) in [
            (0x3d58d031u32, 0x3fc2488au32),
            (0x3f00b99a, 0x3f859f4f),
            (0xbf00b99a, 0x40064033),
            (0x3f800000, 0),
            (0xbf800000, 0x40490fdb),
            (0x32800000, 0x3fc90fdb),
            (0x3f800001, 0xffc00000),
            (0x7f800001, 0x7fc00001),
        ] {
            assert_eq!(acosf_msun(f32::from_bits(x)).to_bits(), want, "{x:#x}");
            assert_eq!(f(cb_acosf, &[f32::from_bits(x)]).to_bits(), want, "{x:#x}");
        }
    }

    /// nextafter and nextafterf give the console's bits and errno: NaN
    /// payloads come back quiet (x's first), ERANGE on overflow and on a
    /// subnormal or zero result, none on a step away from zero.
    #[test]
    fn nextafter_console_bits_and_errno() {
        let r = Some(ERANGE);
        let doubles: [(u64, u64, u64, Option<i32>); 16] = [
            (0x0000000000000000, 0x3ff0000000000000, 0x0000000000000001, None),
            (0x8000000000000000, 0xbff0000000000000, 0x8000000000000001, None),
            (0x0000000000000001, 0x0000000000000000, 0x0000000000000000, r),
            (0x8000000000000001, 0x3ff0000000000000, 0x8000000000000000, r),
            (0x0010000000000000, 0x0000000000000000, 0x000fffffffffffff, r),
            (0x000fffffffffffff, 0x3ff0000000000000, 0x0010000000000000, None),
            (0x7fefffffffffffff, 0x7ff0000000000000, 0x7ff0000000000000, r),
            (0xffefffffffffffff, 0xfff0000000000000, 0xfff0000000000000, r),
            (0x7ff0000000000000, 0x0000000000000000, 0x7fefffffffffffff, None),
            (0x3ff0000000000000, 0x4000000000000000, 0x3ff0000000000001, None),
            (0x3ff0000000000000, 0x3ff0000000000000, 0x3ff0000000000000, None),
            (0x8000000000000000, 0x0000000000000000, 0x0000000000000000, None),
            (0x7ff0000000000001, 0x3ff0000000000000, 0x7ff8000000000001, None),
            (0x3ff0000000000000, 0xfff8000000000123, 0xfff8000000000123, None),
            (0x7ff8000000000123, 0xfff8000000000456, 0x7ff8000000000123, None),
            (0xfff8000000000456, 0x7ff0000000000001, 0xfff8000000000456, None),
        ];
        for (x, y, want, errno) in doubles {
            let (got, e) = next_after_f64(f64::from_bits(x), f64::from_bits(y));
            assert_eq!((got.to_bits(), e), (want, errno), "{x:#x} {y:#x}");
            assert_eq!(d(cb_nextafter, &[f64::from_bits(x), f64::from_bits(y)]).to_bits(), want);
        }
        let floats: [(u32, u32, u32, Option<i32>); 16] = [
            (0x00000000, 0x3f800000, 0x00000001, None),
            (0x80000000, 0xbf800000, 0x80000001, None),
            (0x00000001, 0x00000000, 0x00000000, r),
            (0x80000001, 0x3f800000, 0x80000000, r),
            (0x00800000, 0x00000000, 0x007fffff, r),
            (0x007fffff, 0x3f800000, 0x00800000, None),
            (0x7f7fffff, 0x7f800000, 0x7f800000, r),
            (0xff7fffff, 0xff800000, 0xff800000, r),
            (0x7f800000, 0x00000000, 0x7f7fffff, None),
            (0x3f800000, 0x40000000, 0x3f800001, None),
            (0x3f800000, 0x3f800000, 0x3f800000, None),
            (0x80000000, 0x00000000, 0x00000000, None),
            (0x7f800001, 0x3f800000, 0x7fc00001, None),
            (0x3f800000, 0xffc00123, 0xffc00123, None),
            (0x7fc00123, 0xffc00456, 0x7fc00123, None),
            (0xffc00456, 0x7f800001, 0xffc00456, None),
        ];
        for (x, y, want, errno) in floats {
            let (got, e) = next_after_f32(f32::from_bits(x), f32::from_bits(y));
            assert_eq!((got.to_bits(), e), (want, errno), "{x:#x} {y:#x}");
            assert_eq!(f(cb_nextafterf, &[f32::from_bits(x), f32::from_bits(y)]).to_bits(), want);
        }
    }

    #[test]
    fn llroundf_isinf_and_fpclassifyf() {
        let fl = |v: f32| v.to_bits() as u64;
        assert_eq!(int(cb_llroundf, &[fl(2.5)]), 3);
        assert_eq!(int(cb_llroundf, &[fl(-2.5)]), -3);
        assert_eq!(int(cb_llroundf, &[fl(-0.4)]), 0);
        assert_eq!(int(cb_llroundf, &[fl(8589934592.0)]), 8589934592);
        assert_eq!(int(cb_llroundf, &[fl(-(2f32.powi(62)))]), -(1i64 << 62));
        assert_eq!(int(cb_llroundf, &[fl(0.49999997)]), 0);
        assert_eq!(int(cb_llroundf, &[fl(16777215.0)]), 16777215);
        let dl = |v: f64| v.to_bits();
        assert_eq!(int(cb_isinf, &[dl(f64::INFINITY)]), 1);
        assert_eq!(int(cb_isinf, &[dl(f64::NEG_INFINITY)]), 1);
        assert_eq!(int(cb_isinf, &[dl(f64::NAN)]), 0);
        assert_eq!(int(cb_isinf, &[dl(f64::MAX)]), 0);
        assert_eq!(int(cb_isinf, &[dl(0.0)]), 0);
        for (bits, class) in [
            (0x0000_0000u32, FP_ZERO),
            (0x8000_0000, FP_ZERO),
            (0x0000_0001, FP_SUBNORMAL),
            (0x807F_FFFF, FP_SUBNORMAL),
            (0x0080_0000, FP_NORMAL),
            (0x3F80_0000, FP_NORMAL),
            (0x7F80_0000, FP_INFINITE),
            (0xFF80_0000, FP_INFINITE),
            (0x7FC0_0000, FP_NAN),
            (0xFF80_0001, FP_NAN),
        ] {
            assert_eq!(int(cb_fpclassifyf, &[bits as u64]), class, "{bits:#x}");
        }
    }

    #[test]
    fn ldiv_returns_quotient_and_remainder_in_rax_rdx() {
        let vm = VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        for n in [-7i64, -1, 0, 1, 7, i64::MAX] {
            for d in [-3i64, -1, 1, 3] {
                ctx.set(crate::context::CpuRegister::Rdi, n as u64);
                ctx.set(crate::context::CpuRegister::Rsi, d as u64);
                assert_eq!(cb_ldiv(&mut ctx), n / d);
                assert_eq!(ctx.get(crate::context::CpuRegister::Rdx) as i64, n % d);
            }
        }
        ctx.set(crate::context::CpuRegister::Rdi, i64::MIN as u64);
        ctx.set(crate::context::CpuRegister::Rsi, 2);
        assert_eq!(cb_ldiv(&mut ctx), i64::MIN / 2);
        assert_eq!(ctx.get(crate::context::CpuRegister::Rdx), 0);
    }

    #[test]
    fn sincos_writes_through_rdi_and_rsi() {
        const BASE: u64 = 0x7B33_0000_0000;
        let vm = VirtualMemory::new();
        vm.map(BASE, 0x1000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(crate::context::CpuRegister::Rdi, BASE);
        ctx.set(crate::context::CpuRegister::Rsi, BASE + 8);
        ps5cpu::exec::with_import_xmm(&[0.5f64.to_bits()], || cb_sincos(&mut ctx));
        assert_eq!(f64::from_bits(ctx.read_u64(BASE).unwrap()), 0.5f64.sin());
        assert_eq!(f64::from_bits(ctx.read_u64(BASE + 8).unwrap()), 0.5f64.cos());
        ctx.set(crate::context::CpuRegister::Rdi, BASE + 16);
        ctx.set(crate::context::CpuRegister::Rsi, BASE + 20);
        ps5cpu::exec::with_import_xmm(&[0.5f32.to_bits() as u64], || cb_sincosf(&mut ctx));
        assert_eq!(f32::from_bits(ctx.read_u32(BASE + 16).unwrap()), 0.5f32.sin());
        assert_eq!(f32::from_bits(ctx.read_u32(BASE + 20).unwrap()), 0.5f32.cos());
        vm.unmap(BASE, 0x1000).unwrap();
    }

    /// sqrt and sqrtf: correctly rounded, the sign of zero kept, NaN below
    /// zero (reference bit patterns).
    #[test]
    fn square_root_bits() {
        assert_eq!(d(cb_sqrt, &[4.0]), 2.0);
        assert_eq!(f(cb_sqrtf, &[6.25]), 2.5);
        assert_eq!(d(cb_sqrt, &[2.0]).to_bits(), 0x3ff6a09e667f3bcd);
        assert_eq!(f(cb_sqrtf, &[2.0]).to_bits(), 0x3fb504f3);
        assert_eq!(d(cb_sqrt, &[f64::from_bits(1)]).to_bits(), 0x1e60000000000000);
        assert_eq!(d(cb_sqrt, &[-0.0]).to_bits(), 0x8000000000000000);
        assert_eq!(f(cb_sqrtf, &[-0.0]).to_bits(), 0x80000000);
        assert_eq!(d(cb_sqrt, &[f64::INFINITY]), f64::INFINITY);
        assert_eq!(f(cb_sqrtf, &[f32::INFINITY]), f32::INFINITY);
        assert!(d(cb_sqrt, &[-1.0]).is_nan() && f(cb_sqrtf, &[-1.0]).is_nan());
    }

    fn x87(mantissa: u64, sign_exponent: u16) -> [u8; 10] {
        let mut b = [0u8; 10];
        b[..8].copy_from_slice(&mantissa.to_le_bytes());
        b[8..].copy_from_slice(&sign_exponent.to_le_bytes());
        b
    }

    fn is_x87_nan(v: &[u8; 10]) -> bool {
        let mant = u64::from_le_bytes(v[..8].try_into().unwrap());
        u16::from_le_bytes([v[8], v[9]]) & 0x7FFF == 0x7FFF && mant << 1 != 0
    }

    /// acosl and frexpl take the x87 extended argument from the stack and
    /// return in ST0; frexpl leaves the exponent alone for infinities and
    /// NaN (reference cases, extended-precision bits).
    #[test]
    fn long_double_acosl_and_frexpl() {
        const BASE: u64 = 0x7C64_0000_0000;
        let vm = VirtualMemory::new();
        vm.map(BASE, 0x1000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(crate::context::CpuRegister::Rsp, BASE + 0x100);
        ctx.set(crate::context::CpuRegister::Rdi, BASE + 0x200);
        let mut call = |cb: fn(&mut CpuContext) -> HleResult, x: [u8; 10]| {
            ctx.write_bytes(BASE + 0x108, &x);
            let (_, st0) = ps5cpu::exec::with_import_st0(|| cb(&mut ctx));
            (st0.expect("result in ST0"), ctx.read_u32(BASE + 0x200).unwrap() as i32)
        };
        let ld = crate::scanf::f64_to_x87;
        const PI_MANT: u64 = 0xc90fdaa22168c235;
        assert_eq!(call(cb_acosl, ld(-1.0)).0, x87(PI_MANT, 0x4000));
        assert_eq!(call(cb_acosl, ld(0.0)).0, x87(PI_MANT, 0x3fff));
        assert_eq!(call(cb_acosl, ld(1.0)).0, x87(0, 0));
        assert!(is_x87_nan(&call(cb_acosl, ld(1.5)).0));
        assert!(is_x87_nan(&call(cb_acosl, ld(-2.0)).0));

        vm.write(BASE + 0x200, &77u32.to_le_bytes()).unwrap();
        assert_eq!(call(cb_frexpl, ld(8.0)), (ld(0.5), 4));
        assert_eq!(call(cb_frexpl, ld(-3.0)), (ld(-0.75), 2));
        assert_eq!(call(cb_frexpl, ld(0.375)), (ld(0.75), -1));
        assert_eq!(call(cb_frexpl, x87(1, 0)), (x87(0x8000000000000000, 0x3ffe), -16444));
        vm.write(BASE + 0x200, &77u32.to_le_bytes()).unwrap();
        assert_eq!(call(cb_frexpl, x87(0, 0x8000)), (x87(0, 0x8000), 0));
        for special in [x87(1 << 63, 0x7FFF), x87(1 << 63, 0xFFFF), x87(0xC000_0000_0000_0000, 0x7FFF)] {
            vm.write(BASE + 0x200, &77u32.to_le_bytes()).unwrap();
            assert_eq!(call(cb_frexpl, special), (special, 77));
        }
        vm.unmap(BASE, 0x1000).unwrap();
    }
}
