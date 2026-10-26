// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! printf-family formatting engine with SysV AMD64 va_list support.
//!
//! Implements the variadic argument reading and %-directive formatting that
//! PS5 games rely on for logging (snprintf, vsnprintf, swprintf, vswprintf).
//! The va_list layout follows the SysV AMD64 ABI:
//!   struct va_list { u32 gp_offset; u32 fp_offset; u64 overflow_arg_area;
//!                    u64 reg_save_area; }
//! GP registers are saved at reg_save_area[0..48] (8 bytes each: rdi, rsi,
//! rdx, rcx, r8, r9). FP/SSE registers at reg_save_area[48..176] (16 bytes
//! each, xmm0..xmm7). Overflow args are read from the guest stack.

use crate::context::CpuContext;

/// SysV AMD64 va_list cursor. Reads successive arguments from the register
/// save area first, then from the overflow (stack) area.
pub struct VaListCursor<'a> {
    ctx: &'a CpuContext<'a>,
    gp_offset: u32,
    fp_offset: u32,
    overflow_arg_area: u64,
    reg_save_area: u64,
}

const GP_SAVE_LIMIT: u32 = 48; // 6 GP regs * 8 bytes
const FP_SAVE_LIMIT: u32 = 176; // 48 + 8 xmm regs * 16 bytes

impl<'a> VaListCursor<'a> {
    /// Create a cursor from a va_list guest address.
    pub fn new(ctx: &'a CpuContext<'a>, va_list_addr: u64) -> Option<Self> {
        if va_list_addr == 0 {
            return None;
        }
        let gp_offset = ctx.read_u32(va_list_addr)?;
        let fp_offset = ctx.read_u32(va_list_addr + 4)?;
        let overflow_arg_area = ctx.read_u64(va_list_addr + 8)?;
        let reg_save_area = ctx.read_u64(va_list_addr + 16)?;
        Some(Self {
            ctx,
            gp_offset,
            fp_offset,
            overflow_arg_area,
            reg_save_area,
        })
    }

    /// Read the next integer/pointer argument.
    pub fn next_gp(&mut self) -> u64 {
        let addr = if self.reg_save_area != 0 && self.gp_offset <= GP_SAVE_LIMIT - 8 {
            let a = self.reg_save_area + self.gp_offset as u64;
            self.gp_offset += 8;
            a
        } else {
            let a = self.overflow_arg_area;
            self.overflow_arg_area += 8;
            a
        };
        self.ctx.read_u64(addr).unwrap_or(0)
    }

    /// Read the next floating-point argument (double).
    pub fn next_fp(&mut self) -> f64 {
        let addr = if self.reg_save_area != 0 && self.fp_offset <= FP_SAVE_LIMIT - 16 {
            let a = self.reg_save_area + self.fp_offset as u64;
            self.fp_offset += 16;
            a
        } else {
            let a = self.overflow_arg_area;
            self.overflow_arg_area += 8;
            a
        };
        self.ctx
            .read_u64(addr)
            .map(f64::from_bits)
            .unwrap_or(0.0)
    }
}

/// A fixed-register argument source used by the non-variadic printf
/// wrappers (snprintf-style where args come from CPU registers + stack).
pub struct RegArgsCursor<'a> {
    ctx: &'a CpuContext<'a>,
    /// Next integer argument register (0 = rdi .. 5 = r9), then the stack.
    gp_index: usize,
    /// Next vector argument register (xmm0..xmm7), then the stack.
    fp_index: usize,
    /// Next stack argument.
    stack: u64,
}

impl<'a> RegArgsCursor<'a> {
    /// Variadic arguments of a SysV call made through an import stub, after
    /// `fixed_gp` integer parameters. Floating-point arguments come from the
    /// trapped xmm registers.
    pub fn new(ctx: &'a CpuContext<'a>, fixed_gp: usize) -> Self {
        let rsp = ctx.get(crate::context::CpuRegister::Rsp);
        Self { ctx, gp_index: fixed_gp, fp_index: 0, stack: rsp + 8 }
    }

    fn next_stack(&mut self) -> u64 {
        let v = self.ctx.read_u64(self.stack).unwrap_or(0);
        self.stack += 8;
        v
    }

    pub fn next_gp(&mut self) -> u64 {
        let regs = [self.ctx.rdi(), self.ctx.rsi(), self.ctx.rdx(), self.ctx.rcx(), self.ctx.r8(), self.ctx.r9()];
        if self.gp_index < regs.len() {
            self.gp_index += 1;
            regs[self.gp_index - 1]
        } else {
            self.next_stack()
        }
    }

    pub fn next_fp(&mut self) -> f64 {
        if self.fp_index < 8 {
            if let Some(bits) = ps5cpu::exec::trap_xmm_low(self.fp_index) {
                self.fp_index += 1;
                return f64::from_bits(bits);
            }
        }
        f64::from_bits(self.next_stack())
    }
}

/// Trait abstracting the argument sources.
pub trait ArgSource {
    fn next_gp(&mut self) -> u64;
    fn next_fp(&mut self) -> f64;
    /// Continue at the 0-based argument `index` (only positional sources
    /// can move; a sequential one ignores this).
    fn seek(&mut self, _index: usize) {}
}

/// The class of a variadic argument: an integer/pointer register slot or a
/// floating-point one.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ArgClass {
    Gp,
    Fp,
}

/// Arguments of a format with numbered (`%n$`) conversions: every argument
/// is read once, in order, by its class, then served by position.
struct PositionalArgs {
    values: Vec<u64>,
    next: usize,
}

impl ArgSource for PositionalArgs {
    fn next_gp(&mut self) -> u64 {
        let v = self.values.get(self.next).copied().unwrap_or(0);
        self.next += 1;
        v
    }
    fn next_fp(&mut self) -> f64 {
        f64::from_bits(self.next_gp())
    }
    fn seek(&mut self, index: usize) {
        self.next = index;
    }
}

/// Most numbered arguments a format may use (the console's NL_ARGMAX,
/// measured on hardware).
const MAX_POSITIONAL: usize = 65535;

/// `digits$` at `i` (an argument number, 1-based), advancing past it.
fn parse_position(fmt: &[u8], i: &mut usize) -> Option<usize> {
    let start = *i;
    let mut j = start;
    while fmt.get(j).is_some_and(u8::is_ascii_digit) {
        j += 1;
    }
    if j == start || fmt.get(j) != Some(&b'$') {
        return None;
    }
    let n: usize = std::str::from_utf8(&fmt[start..j]).ok()?.parse().ok()?;
    if n == 0 || n > MAX_POSITIONAL {
        return None;
    }
    *i = j + 1;
    Some(n)
}

/// The argument classes of a format that numbers its arguments (`%2$d`,
/// `%*1$d`), indexed by position; None for a sequential format.
fn positional_classes(fmt: &[u8]) -> Option<Vec<Option<ArgClass>>> {
    let at = |i: usize| fmt.get(i).copied().unwrap_or(0);
    let mut classes: Vec<Option<ArgClass>> = Vec::new();
    let mut numbered = false;
    fn set(classes: &mut Vec<Option<ArgClass>>, n: usize, c: ArgClass) {
        if classes.len() < n {
            classes.resize(n, None);
        }
        classes[n - 1] = Some(c);
    }
    let mut i = 0;
    while at(i) != 0 {
        if at(i) != b'%' {
            i += 1;
            continue;
        }
        i += 1;
        if at(i) == b'%' {
            i += 1;
            continue;
        }
        let position = parse_position(fmt, &mut i);
        numbered |= position.is_some();
        while matches!(at(i), b'0' | b'-' | b'+' | b' ' | b'#') {
            i += 1;
        }
        for precision in [false, true] {
            if precision {
                if at(i) != b'.' {
                    break;
                }
                i += 1;
            }
            if at(i) == b'*' {
                i += 1;
                if let Some(m) = parse_position(fmt, &mut i) {
                    numbered = true;
                    set(&mut classes, m, ArgClass::Gp);
                }
            } else {
                while at(i).is_ascii_digit() {
                    i += 1;
                }
            }
        }
        while matches!(at(i), b'l' | b'h' | b't' | b'j' | b'z') {
            i += 1;
        }
        let class = match at(i) {
            b'd' | b'i' | b'u' | b'x' | b'X' | b'o' | b'b' | b'c' | b'C' | b's' | b'S' | b'p' | b'n' => Some(ArgClass::Gp),
            b'f' | b'F' | b'e' | b'E' | b'g' | b'G' | b'a' | b'A' => Some(ArgClass::Fp),
            _ => None,
        };
        if let (Some(n), Some(c)) = (position, class) {
            set(&mut classes, n, c);
        }
        if at(i) != 0 {
            i += 1;
        }
    }
    numbered.then_some(classes)
}
impl<'a> ArgSource for VaListCursor<'a> {
    fn next_gp(&mut self) -> u64 { self.next_gp() }
    fn next_fp(&mut self) -> f64 { self.next_fp() }
}
impl<'a> ArgSource for RegArgsCursor<'a> {
    fn next_gp(&mut self) -> u64 { self.next_gp() }
    fn next_fp(&mut self) -> f64 { self.next_fp() }
}

// ---------------------------------------------------------------------------
// Formatter
// ---------------------------------------------------------------------------
//
// Directive grammar: %[n$][flags][width][.precision][length]specifier with
//   n$         the 1-based number of the argument to convert
//   flags      - + space 0 #
//   width      digits or '*' / '*m$' (a negative '*' width left-justifies)
//   precision  '.' then digits or '*' / '*m$' (a negative '*' precision is
//              omitted)
//   length     hh h l ll j z t
//   specifier  d i u x X o b f F e E g G a A c s p n %
// Any other specifier character is emitted literally (no argument is
// consumed), so %L directives print their letters.
//
// Integers are built into a 32-byte reversed digit buffer. Floating-point
// conversions are correctly rounded (round half to even on the exact binary
// value), as a standard C library prints them.

const FLAGS_ZEROPAD: u32 = 1 << 0;
const FLAGS_LEFT: u32 = 1 << 1;
const FLAGS_PLUS: u32 = 1 << 2;
const FLAGS_SPACE: u32 = 1 << 3;
const FLAGS_HASH: u32 = 1 << 4;
const FLAGS_UPPERCASE: u32 = 1 << 5;
const FLAGS_CHAR: u32 = 1 << 6;
const FLAGS_SHORT: u32 = 1 << 7;
const FLAGS_LONG: u32 = 1 << 8;
const FLAGS_LONG_LONG: u32 = 1 << 9;
const FLAGS_PRECISION: u32 = 1 << 10;

const NTOA_BUFFER_SIZE: usize = 32;

/// Longest guest string a %s / %ls directive reads.
const MAX_STRING_ARG: usize = 1 << 20;

/// Emit `buf` reversed, padded with spaces to `width`.
fn out_rev(out: &mut Vec<u8>, buf: &[u8], width: u32, flags: u32) {
    let start = out.len();
    if flags & FLAGS_LEFT == 0 && flags & FLAGS_ZEROPAD == 0 {
        for _ in buf.len()..width as usize {
            out.push(b' ');
        }
    }
    out.extend(buf.iter().rev());
    if flags & FLAGS_LEFT != 0 {
        while out.len() - start < width as usize {
            out.push(b' ');
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn ntoa_format(
    out: &mut Vec<u8>,
    buf: &mut [u8; NTOA_BUFFER_SIZE],
    mut len: usize,
    negative: bool,
    base: u64,
    prec: u32,
    mut width: u32,
    flags: u32,
) {
    // Leading zeros.
    if flags & FLAGS_LEFT == 0 {
        if width != 0 && flags & FLAGS_ZEROPAD != 0 && (negative || flags & (FLAGS_PLUS | FLAGS_SPACE) != 0) {
            width -= 1;
        }
        while len < prec as usize && len < NTOA_BUFFER_SIZE {
            buf[len] = b'0';
            len += 1;
        }
        while flags & FLAGS_ZEROPAD != 0 && len < width as usize && len < NTOA_BUFFER_SIZE {
            buf[len] = b'0';
            len += 1;
        }
    }

    // Alternate octal form: the first digit is a zero (one more only when
    // the padding did not already put one there), as in C.
    if flags & FLAGS_HASH != 0 && base == 8 {
        if (len == 0 || buf[len - 1] != b'0') && len < NTOA_BUFFER_SIZE {
            buf[len] = b'0';
            len += 1;
        }
    } else if flags & FLAGS_HASH != 0 {
        if flags & FLAGS_PRECISION == 0 && len != 0 && (len == prec as usize || len == width as usize) {
            len -= 1;
            if len != 0 && base == 16 {
                len -= 1;
            }
        }
        if base == 16 && flags & FLAGS_UPPERCASE == 0 && len < NTOA_BUFFER_SIZE {
            buf[len] = b'x';
            len += 1;
        } else if base == 16 && flags & FLAGS_UPPERCASE != 0 && len < NTOA_BUFFER_SIZE {
            buf[len] = b'X';
            len += 1;
        } else if base == 2 && len < NTOA_BUFFER_SIZE {
            buf[len] = b'b';
            len += 1;
        }
        if len < NTOA_BUFFER_SIZE {
            buf[len] = b'0';
            len += 1;
        }
    }

    if len < NTOA_BUFFER_SIZE {
        if negative {
            buf[len] = b'-';
            len += 1;
        } else if flags & FLAGS_PLUS != 0 {
            buf[len] = b'+';
            len += 1;
        } else if flags & FLAGS_SPACE != 0 {
            buf[len] = b' ';
            len += 1;
        }
    }

    out_rev(out, &buf[..len], width, flags);
}

/// Integer conversion of a magnitude `value` (sign carried separately).
fn ntoa(out: &mut Vec<u8>, mut value: u64, negative: bool, base: u64, prec: u32, width: u32, mut flags: u32) {
    let mut buf = [0u8; NTOA_BUFFER_SIZE];
    let mut len = 0usize;
    let alternate_octal = flags & FLAGS_HASH != 0 && base == 8;

    // No prefix for a zero value.
    if value == 0 {
        flags &= !FLAGS_HASH;
    }

    // An explicit precision of zero prints nothing for zero, except the
    // zero the alternate octal form keeps ("%#.0o").
    if flags & FLAGS_PRECISION != 0 && value == 0 && alternate_octal {
        buf[0] = b'0';
        len = 1;
    } else if flags & FLAGS_PRECISION == 0 || value != 0 {
        loop {
            let digit = (value % base) as u8;
            buf[len] = if digit < 10 {
                b'0' + digit
            } else {
                (if flags & FLAGS_UPPERCASE != 0 { b'A' } else { b'a' }) + digit - 10
            };
            len += 1;
            value /= base;
            if value == 0 || len >= NTOA_BUFFER_SIZE {
                break;
            }
        }
    }

    ntoa_format(out, &mut buf, len, negative, base, prec, width, flags);
}

/// Exponent form `d.ddd` + `e±XX` of a finite non-negative value with
/// `prec` fraction digits, correctly rounded.
fn exp_form(value: f64, prec: usize, alt: bool, upper: bool) -> String {
    let text = format!("{:.*e}", prec, value);
    let (mantissa, exp) = text.split_once('e').unwrap_or((&text, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let mut s = String::with_capacity(mantissa.len() + 6);
    s.push_str(mantissa);
    if alt && prec == 0 {
        s.push('.');
    }
    s.push(if upper { 'E' } else { 'e' });
    s.push(if exp < 0 { '-' } else { '+' });
    s.push_str(&format!("{:02}", exp.unsigned_abs()));
    s
}

/// Decimal exponent of `value` once rounded to `digits` significant digits.
fn rounded_exponent(value: f64, digits: usize) -> i32 {
    if value == 0.0 {
        return 0;
    }
    let text = format!("{:.*e}", digits - 1, value);
    text.split_once('e').and_then(|(_, e)| e.parse().ok()).unwrap_or(0)
}

/// Remove trailing fractional zeros (and a bare '.') before any exponent.
fn strip_fraction_zeros(s: &mut String) {
    let exp_at = s.find(['e', 'E']).unwrap_or(s.len());
    let (body, exp) = s.split_at(exp_at);
    if !body.contains('.') {
        return;
    }
    let trimmed = body.trim_end_matches('0').trim_end_matches('.');
    *s = format!("{trimmed}{exp}");
}

/// Hexadecimal form `0x1.hhhp±d` of a finite non-negative value; subnormals
/// are normalized. Without a precision the digits are exact with trailing
/// zeros removed; with one they round to nearest even.
fn hex_form(value: f64, prec: Option<usize>, alt: bool, upper: bool) -> String {
    let (mut lead, mut frac, mut exp) = if value == 0.0 {
        (0u64, 0u64, 0i32)
    } else {
        let bits = value.to_bits();
        let raw_exp = ((bits >> 52) & 0x7FF) as i32;
        let mut mant = bits & ((1u64 << 52) - 1);
        let mut exp = raw_exp - 1023;
        if raw_exp == 0 {
            // Subnormal: shift the leading one into the implicit position.
            exp = -1022;
            while mant & (1u64 << 52) == 0 {
                mant <<= 1;
                exp -= 1;
            }
            mant &= (1u64 << 52) - 1;
        }
        (1u64, mant, exp)
    };
    let mut digits = match prec {
        Some(p) if p < 13 => {
            let shift = 52 - 4 * p as u32;
            let rem = frac & ((1u64 << shift) - 1);
            frac >>= shift;
            let half = 1u64 << (shift - 1);
            let odd = if p == 0 { lead & 1 != 0 } else { frac & 1 != 0 };
            if rem > half || (rem == half && odd) {
                frac += 1;
                if frac >> (4 * p) != 0 {
                    // Carry into the leading digit: renormalize.
                    frac = 0;
                    lead = 1;
                    exp += 1;
                }
            }
            if p == 0 { String::new() } else { format!("{:0width$x}", frac, width = p) }
        }
        Some(p) => {
            let mut d = format!("{:013x}", frac);
            d.extend(std::iter::repeat_n('0', p - 13));
            d
        }
        None => format!("{:013x}", frac).trim_end_matches('0').to_string(),
    };
    if upper {
        digits.make_ascii_uppercase();
    }
    let mut s = String::from(if upper { "0X" } else { "0x" });
    s.push(char::from(b'0' + lead as u8));
    if !digits.is_empty() || alt {
        s.push('.');
    }
    s.push_str(&digits);
    s.push(if upper { 'P' } else { 'p' });
    s.push(if exp < 0 { '-' } else { '+' });
    s.push_str(&exp.unsigned_abs().to_string());
    s
}

/// Floating-point conversions %f %F %e %E %g %G %a %A, formatted as a
/// standard C library does: correctly rounded digits, "inf"/"nan" without
/// zero padding, the sign bit (also of a NaN) giving '-', the '+' and ' '
/// flags their sign, a sign on negative zero.
fn format_float(out: &mut Vec<u8>, value: f64, spec: u8, prec: Option<u32>, width: u32, flags: u32) {
    let upper = spec.is_ascii_uppercase();
    let alt = flags & FLAGS_HASH != 0;
    let mut zero_pad = flags & FLAGS_ZEROPAD != 0 && flags & FLAGS_LEFT == 0;
    let sign: &str = if value.is_sign_negative() {
        "-"
    } else if flags & FLAGS_PLUS != 0 {
        "+"
    } else if flags & FLAGS_SPACE != 0 {
        " "
    } else {
        ""
    };
    let v = value.abs();
    let prec = prec.map(|p| p as usize);
    let body = if !v.is_finite() {
        zero_pad = false;
        let t = if v.is_nan() { "nan" } else { "inf" };
        if upper { t.to_ascii_uppercase() } else { t.to_string() }
    } else {
        match spec.to_ascii_lowercase() {
            b'f' => {
                let mut s = format!("{:.*}", prec.unwrap_or(6), v);
                if alt && prec == Some(0) {
                    s.push('.');
                }
                s
            }
            b'e' => exp_form(v, prec.unwrap_or(6), alt, upper),
            b'a' => hex_form(v, prec, alt, upper),
            _ => {
                let p = prec.unwrap_or(6).max(1);
                let x = rounded_exponent(v, p);
                let mut s = if (p as i32) > x && x >= -4 {
                    format!("{:.*}", (p as i32 - 1 - x) as usize, v)
                } else {
                    exp_form(v, p - 1, false, upper)
                };
                if alt {
                    if !s.contains('.') {
                        let at = s.find(['e', 'E']).unwrap_or(s.len());
                        s.insert(at, '.');
                    }
                } else {
                    strip_fraction_zeros(&mut s);
                }
                s
            }
        }
    };
    // Zero padding goes after the sign (and after "0x" for %a).
    let prefix_len = if zero_pad && (body.starts_with("0x") || body.starts_with("0X")) { 2 } else { 0 };
    let pad = (width as usize).saturating_sub(sign.len() + body.len());
    if flags & FLAGS_LEFT == 0 && !zero_pad {
        out.extend(std::iter::repeat_n(b' ', pad));
    }
    out.extend_from_slice(sign.as_bytes());
    out.extend_from_slice(&body.as_bytes()[..prefix_len]);
    if zero_pad {
        out.extend(std::iter::repeat_n(b'0', pad));
    }
    out.extend_from_slice(&body.as_bytes()[prefix_len..]);
    if flags & FLAGS_LEFT != 0 {
        out.extend(std::iter::repeat_n(b' ', pad));
    }
}

/// Parse a decimal field, wrapping like an unsigned int.
fn parse_uint(fmt: &[u8], i: &mut usize) -> u32 {
    let mut v = 0u32;
    while let Some(&c) = fmt.get(*i).filter(|c| c.is_ascii_digit()) {
        v = v.wrapping_mul(10).wrapping_add((c - b'0') as u32);
        *i += 1;
    }
    v
}

/// %s / %ls argument bytes, at most `limit` bytes of output.
fn string_arg(ctx: &CpuContext, addr: u64, limit: usize, wide: bool) -> Vec<u8> {
    if addr == 0 {
        return b"(null)"[..6.min(limit)].to_vec();
    }
    let cap = limit.min(MAX_STRING_ARG);
    if !wide {
        return ctx.read_cstr_bytes(addr, cap).unwrap_or_default();
    }
    // Wide strings convert each 16-bit unit on its own (the console's
    // behaviour: a surrogate is its own three bytes, pairs are never
    // combined), and the precision limit never cuts inside a unit.
    let units = ctx.read_wide_string(addr, cap).unwrap_or_default();
    let mut converted = Vec::new();
    for &unit in &units {
        if converted.len() >= limit {
            break;
        }
        let bytes: &[u8] = if unit < 0x80 {
            &[unit as u8]
        } else if unit < 0x800 {
            &[0xC0 | (unit >> 6) as u8, 0x80 | (unit & 0x3F) as u8]
        } else {
            &[0xE0 | (unit >> 12) as u8, 0x80 | ((unit >> 6) & 0x3F) as u8, 0x80 | (unit & 0x3F) as u8]
        };
        if converted.len() + bytes.len() <= limit {
            converted.extend_from_slice(bytes);
        }
    }
    converted
}

/// Core printf formatter over a byte format string. Returns the produced
/// bytes; their count is the value printf-family functions return. A format
/// that numbers its arguments (`%2$s %1$d`) reads them all first, in order,
/// by the classes its conversions give them.
pub fn format_bytes(ctx: &CpuContext, format: &[u8], args: &mut dyn ArgSource) -> Vec<u8> {
    format_bytes_mode(ctx, format, args, false).unwrap_or_default()
}

/// [`format_bytes`] under the C11 Annex K rules of the `_s` variants: any
/// `%n` and a null `%s` / `%ls` / `%S` argument are constraint violations
/// (None, nothing is produced).
pub fn format_bytes_secure(ctx: &CpuContext, format: &[u8], args: &mut dyn ArgSource) -> Option<Vec<u8>> {
    format_bytes_mode(ctx, format, args, true)
}

fn format_bytes_mode(ctx: &CpuContext, format: &[u8], args: &mut dyn ArgSource, secure: bool) -> Option<Vec<u8>> {
    let Some(classes) = positional_classes(format) else {
        return format_core_mode(ctx, format, args, secure);
    };
    let values = classes
        .iter()
        .map(|c| match c {
            Some(ArgClass::Fp) => args.next_fp().to_bits(),
            _ => args.next_gp(),
        })
        .collect();
    format_core_mode(ctx, format, &mut PositionalArgs { values, next: 0 }, secure)
}

/// Store the %n count at `ptr` in the width its length modifier gives.
fn store_count(ctx: &CpuContext, ptr: u64, count: usize, flags: u32) {
    if ptr == 0 {
        return;
    }
    let bytes = (count as u64).to_le_bytes();
    let width = if flags & FLAGS_CHAR != 0 {
        1
    } else if flags & FLAGS_SHORT != 0 {
        2
    } else if flags & (FLAGS_LONG | FLAGS_LONG_LONG) != 0 {
        8
    } else {
        4
    };
    ctx.write_bytes(ptr, &bytes[..width]);
}

fn format_core(ctx: &CpuContext, format: &[u8], args: &mut dyn ArgSource) -> Vec<u8> {
    format_core_mode(ctx, format, args, false).unwrap_or_default()
}

fn format_core_mode(ctx: &CpuContext, format: &[u8], args: &mut dyn ArgSource, secure: bool) -> Option<Vec<u8>> {
    let at = |i: usize| format.get(i).copied().unwrap_or(0);
    let mut out = Vec::with_capacity(format.len() + 32);
    let mut i = 0usize;

    while at(i) != 0 {
        if at(i) != b'%' {
            out.push(at(i));
            i += 1;
            continue;
        }
        i += 1;
        // A numbered conversion (`%2$d`) takes that argument.
        let position = parse_position(format, &mut i);

        // Flags.
        let mut flags = 0u32;
        loop {
            let f = match at(i) {
                b'0' => FLAGS_ZEROPAD,
                b'-' => FLAGS_LEFT,
                b'+' => FLAGS_PLUS,
                b' ' => FLAGS_SPACE,
                b'#' => FLAGS_HASH,
                _ => break,
            };
            flags |= f;
            i += 1;
        }

        // Width.
        let mut width = 0u32;
        if at(i).is_ascii_digit() {
            width = parse_uint(format, &mut i);
        } else if at(i) == b'*' {
            i += 1;
            if let Some(m) = parse_position(format, &mut i) {
                args.seek(m - 1);
            }
            let w = args.next_gp() as i32;
            if w < 0 {
                flags |= FLAGS_LEFT;
                width = w.wrapping_neg() as u32;
            } else {
                width = w as u32;
            }
        }

        // Precision.
        let mut precision = 0u32;
        if at(i) == b'.' {
            flags |= FLAGS_PRECISION;
            i += 1;
            if at(i).is_ascii_digit() {
                precision = parse_uint(format, &mut i);
            } else if at(i) == b'*' {
                i += 1;
                if let Some(m) = parse_position(format, &mut i) {
                    args.seek(m - 1);
                }
                let p = args.next_gp() as i32;
                if p < 0 {
                    flags &= !FLAGS_PRECISION;
                } else {
                    precision = p as u32;
                }
            }
        }

        // Length.
        match at(i) {
            b'l' => {
                flags |= FLAGS_LONG;
                i += 1;
                if at(i) == b'l' {
                    flags |= FLAGS_LONG_LONG;
                    i += 1;
                }
            }
            b'h' => {
                flags |= FLAGS_SHORT;
                i += 1;
                if at(i) == b'h' {
                    flags |= FLAGS_CHAR;
                    i += 1;
                }
            }
            b't' | b'j' | b'z' => {
                flags |= FLAGS_LONG_LONG;
                i += 1;
            }
            _ => {}
        }

        let spec = at(i);
        if let Some(n) = position {
            args.seek(n - 1);
        }
        match spec {
            b'n' if secure => return None,
            b'n' => {
                // The count of bytes produced so far.
                store_count(ctx, args.next_gp(), out.len(), flags);
                i += 1;
            }
            b'd' | b'i' | b'u' | b'x' | b'X' | b'o' | b'b' => {
                let base: u64 = match spec {
                    b'x' | b'X' => 16,
                    b'o' => 8,
                    b'b' => 2,
                    _ => {
                        flags &= !FLAGS_HASH;
                        10
                    }
                };
                if spec == b'X' {
                    flags |= FLAGS_UPPERCASE;
                }
                if spec != b'i' && spec != b'd' {
                    flags &= !(FLAGS_PLUS | FLAGS_SPACE);
                }
                if flags & FLAGS_PRECISION != 0 {
                    flags &= !FLAGS_ZEROPAD;
                }

                let wide = flags & (FLAGS_LONG | FLAGS_LONG_LONG) != 0;
                if spec == b'i' || spec == b'd' {
                    let value = if wide {
                        args.next_gp() as i64
                    } else {
                        let v = args.next_gp() as i32;
                        if flags & FLAGS_CHAR != 0 {
                            v as i8 as i64
                        } else if flags & FLAGS_SHORT != 0 {
                            v as i16 as i64
                        } else {
                            v as i64
                        }
                    };
                    ntoa(&mut out, value.unsigned_abs(), value < 0, base, precision, width, flags);
                } else {
                    let value = if wide {
                        args.next_gp()
                    } else {
                        let v = args.next_gp() as u32;
                        if flags & FLAGS_CHAR != 0 {
                            v as u8 as u64
                        } else if flags & FLAGS_SHORT != 0 {
                            v as u16 as u64
                        } else {
                            v as u64
                        }
                    };
                    ntoa(&mut out, value, false, base, precision, width, flags);
                }
                i += 1;
            }
            b'f' | b'F' | b'e' | b'E' | b'g' | b'G' | b'a' | b'A' => {
                let prec = (flags & FLAGS_PRECISION != 0).then_some(precision);
                format_float(&mut out, args.next_fp(), spec, prec, width, flags);
                i += 1;
            }
            b'c' | b'C' => {
                // %lc and %C (the same conversion) take a wide character,
                // written as UTF-8.
                let value = args.next_gp() as u32;
                let bytes: Vec<u8> = if spec == b'C' || flags & FLAGS_LONG != 0 {
                    let unit = value & 0xFFFF;
                    char::from_u32(unit).map(|c| c.to_string().into_bytes()).unwrap_or_else(|| vec![b'?'])
                } else {
                    vec![value as u8]
                };
                let pad = (width as usize).saturating_sub(1);
                if flags & FLAGS_LEFT == 0 {
                    out.extend(std::iter::repeat_n(b' ', pad));
                }
                out.extend_from_slice(&bytes);
                if flags & FLAGS_LEFT != 0 {
                    out.extend(std::iter::repeat_n(b' ', pad));
                }
                i += 1;
            }
            b's' | b'S' => {
                // %S is %ls: a wide (16-bit) string.
                let limit = if flags & FLAGS_PRECISION != 0 { precision as usize } else { usize::MAX };
                let addr = args.next_gp();
                if addr == 0 && secure {
                    return None;
                }
                let text = string_arg(ctx, addr, limit, spec == b'S' || flags & FLAGS_LONG != 0);
                let pad = (width as usize).saturating_sub(text.len());
                if flags & FLAGS_LEFT == 0 {
                    out.extend(std::iter::repeat_n(b' ', pad));
                }
                out.extend_from_slice(&text);
                if flags & FLAGS_LEFT != 0 {
                    out.extend(std::iter::repeat_n(b' ', pad));
                }
                i += 1;
            }
            b'p' => {
                // As the console's libc prints it: bare lowercase hex with no
                // prefix ("0" for NULL), '#', '+' and ' ' ignored, the width
                // and precision applied to the digits, zero padding only
                // without a precision.
                flags &= !(FLAGS_PLUS | FLAGS_SPACE | FLAGS_HASH | FLAGS_UPPERCASE);
                if flags & FLAGS_PRECISION != 0 {
                    flags &= !FLAGS_ZEROPAD;
                }
                ntoa(&mut out, args.next_gp(), false, 16, precision, width, flags);
                i += 1;
            }
            b'%' => {
                out.push(b'%');
                i += 1;
            }
            0 => {
                // A '%' ending the format emits the terminator character.
                out.push(0);
                break;
            }
            other => {
                // A conversion the console does not have is copied through
                // as `%` and the letter, padded as a string conversion.
                let mut text = vec![b'%', other];
                if flags & FLAGS_PRECISION != 0 {
                    text.truncate(precision as usize);
                }
                let pad = (width as usize).saturating_sub(text.len());
                if flags & FLAGS_LEFT == 0 {
                    out.extend(std::iter::repeat_n(b' ', pad));
                }
                out.extend_from_slice(&text);
                if flags & FLAGS_LEFT != 0 {
                    out.extend(std::iter::repeat_n(b' ', pad));
                }
                i += 1;
            }
        }
    }
    Some(out)
}

/// [`format_bytes`] over a UTF-8 format, producing UTF-8 text.
pub fn format_printf(ctx: &CpuContext, format: &str, args: &mut dyn ArgSource) -> String {
    String::from_utf8_lossy(&format_bytes(ctx, format.as_bytes(), args)).into_owned()
}

/// Append the UTF-16 units of a narrow (UTF-8) string, at most `limit`
/// units; a byte that does not start a valid sequence stands for itself,
/// and a surrogate pair that does not fit is left out.
fn append_utf16(out: &mut Vec<u16>, text: &[u8], limit: usize) {
    let mut index = 0;
    let start = out.len();
    while out.len() - start < limit && index < text.len() {
        let lead = text[index];
        let mut extra = if lead >= 0xF0 { 3 } else if lead >= 0xE0 { 2 } else if lead >= 0xC0 { 1 } else { 0 };
        let mut code: u32 = match extra {
            3 => (lead & 0x07) as u32,
            2 => (lead & 0x0F) as u32,
            1 => (lead & 0x1F) as u32,
            _ => lead as u32,
        };
        for k in 1..=extra {
            match text.get(index + k) {
                Some(&next) if next & 0xC0 == 0x80 => code = (code << 6) | (next & 0x3F) as u32,
                _ => {
                    extra = 0;
                    code = lead as u32;
                    break;
                }
            }
        }
        index += extra + 1;
        if code >= 0x10000 {
            if limit - (out.len() - start) < 2 {
                break;
            }
            let c = code - 0x10000;
            out.push(0xD800 + (c >> 10) as u16);
            out.push(0xDC00 + (c & 0x3FF) as u16);
        } else {
            out.push(code as u16);
        }
    }
}

/// Pad `text` with spaces to `width` units.
fn append_padded(out: &mut Vec<u16>, text: &[u16], left: bool, width: usize) {
    let pad = width.saturating_sub(text.len());
    if !left {
        out.extend(std::iter::repeat_n(b' ' as u16, pad));
    }
    out.extend_from_slice(text);
    if left {
        out.extend(std::iter::repeat_n(b' ' as u16, pad));
    }
}

/// The wide printf engine (swprintf family): a format of guest wide (16-bit)
/// units producing wide units, widths and precisions counted in units.
/// `%s` converts a narrow (UTF-8) argument, `%ls` / `%S` copy a wide one,
/// `%c` takes a byte and `%lc` / `%C` a wide unit. Numbers use the narrow
/// engine. None for a format the console rejects: an unknown conversion or
/// length, a conversion cut by the end of the format, a null `%n` pointer,
/// and in `secure` mode (the `_s` variants) any `%n` or null string.
pub fn format_wide(ctx: &CpuContext, format: &[u16], args: &mut dyn ArgSource, secure: bool) -> Option<Vec<u16>> {
    let at = |i: usize| format.get(i).copied().unwrap_or(0);
    let ascii = |u: u16, set: &[u8]| u != 0 && u < 0x80 && set.contains(&(u as u8));
    let mut out: Vec<u16> = Vec::with_capacity(format.len() + 16);
    let mut i = 0usize;
    while at(i) != 0 {
        if at(i) != b'%' as u16 {
            out.push(at(i));
            i += 1;
            continue;
        }
        i += 1;
        if at(i) == b'%' as u16 {
            out.push(at(i));
            i += 1;
            continue;
        }
        let mut spec = String::from("%");
        let mut left = false;
        let mut width = 0usize;
        let mut precision: Option<usize> = None;
        while ascii(at(i), b"-+ #0") {
            left |= at(i) == b'-' as u16;
            spec.push(at(i) as u8 as char);
            i += 1;
        }
        if at(i) == b'*' as u16 {
            i += 1;
            let w = args.next_gp() as i32;
            if w < 0 {
                left = true;
                spec.push('-');
            }
            width = w.unsigned_abs() as usize;
            spec.push_str(&width.to_string());
        } else {
            while ascii(at(i), b"0123456789") {
                width = width.saturating_mul(10).saturating_add((at(i) - b'0' as u16) as usize);
                spec.push(at(i) as u8 as char);
                i += 1;
            }
        }
        if at(i) == b'.' as u16 {
            i += 1;
            if at(i) == b'*' as u16 {
                i += 1;
                let p = args.next_gp() as i32;
                if p >= 0 {
                    precision = Some(p as usize);
                    spec.push_str(&format!(".{p}"));
                }
            } else {
                let mut p = 0usize;
                spec.push('.');
                while ascii(at(i), b"0123456789") {
                    p = p.saturating_mul(10).saturating_add((at(i) - b'0' as u16) as usize);
                    spec.push(at(i) as u8 as char);
                    i += 1;
                }
                precision = Some(p);
            }
        }
        let mut length = String::new();
        if ascii(at(i), b"hljztL") {
            length.push(at(i) as u8 as char);
            i += 1;
            if (length == "h" && at(i) == b'h' as u16) || (length == "l" && at(i) == b'l' as u16) {
                length.push(at(i) as u8 as char);
                i += 1;
            }
        }
        let conversion = at(i);
        if conversion == 0 {
            return None;
        }
        i += 1;
        let integer_length = matches!(length.as_str(), "" | "h" | "hh" | "l" | "ll" | "j" | "z" | "t");
        let c = if conversion < 0x80 { conversion as u8 } else { 0 };
        match c {
            b'd' | b'i' | b'o' | b'u' | b'x' | b'X' if integer_length => {
                let narrow = format_core(ctx, format!("{spec}{length}{}", c as char).as_bytes(), args);
                out.extend(narrow.iter().map(|&b| b as u16));
            }
            b'a' | b'A' | b'e' | b'E' | b'f' | b'F' | b'g' | b'G' if length.is_empty() || length == "l" => {
                let narrow = format_core(ctx, format!("{spec}{}", c as char).as_bytes(), args);
                out.extend(narrow.iter().map(|&b| b as u16));
            }
            b'p' if length.is_empty() => {
                let narrow = format_core(ctx, format!("{spec}p").as_bytes(), args);
                out.extend(narrow.iter().map(|&b| b as u16));
            }
            b'c' | b'C' => {
                let value = args.next_gp() as u32;
                let unit = if c == b'C' || length == "l" { value as u16 } else { value as u8 as u16 };
                append_padded(&mut out, &[unit], left, width);
            }
            b's' | b'S' => {
                let limit = precision.unwrap_or(usize::MAX);
                let addr = args.next_gp();
                if addr == 0 && secure {
                    return None;
                }
                let mut text: Vec<u16> = Vec::new();
                if c == b'S' || length == "l" {
                    if addr == 0 {
                        text.extend("(null)".encode_utf16().take(limit));
                    } else {
                        text = ctx.read_wide_string(addr, limit.min(MAX_STRING_ARG)).unwrap_or_default();
                    }
                } else {
                    let bytes = if addr == 0 {
                        b"(null)".to_vec()
                    } else {
                        ctx.read_cstr_bytes(addr, MAX_STRING_ARG).unwrap_or_default()
                    };
                    append_utf16(&mut text, &bytes, limit);
                }
                append_padded(&mut out, &text, left, width);
            }
            b'n' if integer_length && spec == "%" && !secure => {
                let ptr = args.next_gp();
                if ptr == 0 {
                    return None;
                }
                let flags = match length.as_str() {
                    "hh" => FLAGS_CHAR,
                    "h" => FLAGS_SHORT,
                    "" => 0,
                    _ => FLAGS_LONG,
                };
                store_count(ctx, ptr, out.len(), flags);
            }
            _ => return None,
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Arguments taken from fixed integer and floating-point lists, the way
    /// a va_list hands them out.
    struct Args {
        gp: Vec<u64>,
        fp: Vec<f64>,
    }

    impl ArgSource for Args {
        fn next_gp(&mut self) -> u64 {
            if self.gp.is_empty() { 0 } else { self.gp.remove(0) }
        }
        fn next_fp(&mut self) -> f64 {
            if self.fp.is_empty() { 0.0 } else { self.fp.remove(0) }
        }
    }

    const BASE: u64 = 0x5D50_0000_0000;

    /// Each test maps a page of its own: a fixed shared address would be
    /// re-mmap'd (and zeroed) by a parallel test, since the tests run in one
    /// process and the backend mmaps with MAP_FIXED.
    fn memory() -> (ps5vm::VirtualMemory, u64) {
        static SLOT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let base = BASE + SLOT.fetch_add(1, std::sync::atomic::Ordering::SeqCst) * 0x1000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(base, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let ctx = CpuContext::new(&vm);
        ctx.write_bytes(base, b"hello\0");
        let wide: Vec<u8> = "W\u{e9}\u{65e5}\u{1F600}z\0".encode_utf16().flat_map(|u| u.to_le_bytes()).collect();
        ctx.write_bytes(base + 0x100, &wide);
        (vm, base)
    }

    fn check(vm: &ps5vm::VirtualMemory, fmt: &str, gp: &[u64], fp: &[f64], expected: &[u8], ret: usize) {
        let ctx = CpuContext::new(vm);
        let mut args = Args { gp: gp.to_vec(), fp: fp.to_vec() };
        let out = format_bytes(&ctx, fmt.as_bytes(), &mut args);
        assert_eq!(out.len(), ret, "count for {fmt:?}: {:?}", String::from_utf8_lossy(&out));
        let visible = out.split(|&b| b == 0).next().unwrap();
        assert_eq!(visible, expected, "{fmt:?} gave {:?}", String::from_utf8_lossy(&out));
    }

    #[test]
    fn printf_integers_flags_and_lengths() {
        let (vm, _) = memory();
        let m = |v: i64| v as u64;
        check(&vm, "%d|%i|%5d|%-5d|%05d|%+d|% d|%+05d", &[42, m(-7), 42, 42, m(-42), 42, 42, 42], &[],
            b"42|-7|   42|42   |-0042|+42| 42|+0042", 37);
        check(&vm, "%.3d|%8.3d|%-8.3d|%08.3d|%.0d|%.0x|%#.0x", &[5, 5, 5, 5, 0, 0, 0], &[],
            b"005|     005|5       |     005|||", 33);
        check(&vm, "%x|%X|%#x|%#X|%#o|%o|%b|%#b|%#010x|%#5x", &[255, 255, 255, 255, 8, 8, 5, 5, 255, 0], &[],
            b"ff|FF|0xff|0XFF|010|10|101|0b101|0x000000ff|    0", 49);
        check(&vm, "%hhd|%hd|%hhu|%hu|%ld|%lld|%lu|%jd|%zu|%td",
            &[0x1ff, 0x18000, 0x1ff, 0x18000, m(-1), 0x8000_0000_0000_0000, m(-1), m(-2), 7, m(-3)], &[],
            b"-1|-32768|255|32768|-1|-9223372036854775808|18446744073709551615|-2|7|-3", 72);
        check(&vm, "%u|%d|%lx|%llX", &[u64::MAX, 0x1_0000_0001, 0xdead_beef_cafe, 0xab_cdef_0123], &[],
            b"4294967295|1|deadbeefcafe|ABCDEF0123", 36);
        check(&vm, "%-+5d|%- 5d|%0-5d|%+ d|%#5o|%#.3o|%.5x|%-#8x|", &[3, 3, 3, 3, 8, 8, 0xab, 0xab], &[],
            b"+3   | 3   |3    |+3|  010|010|000ab|0xab    |", 46);
        // The alternate octal form keeps one leading zero (FreeBSD): a zero
        // with precision 0, no second zero after precision or zero padding.
        check(&vm, "%#.0o|%#o|%#.5o|%#05o|%#5o|%#.0o|%.0o", &[0, 0, 8, 8, 8, 8, 0], &[], b"0|0|00010|00010|  010|010|", 26);
        check(&vm, "%d %d %d %d %d %d %d %d", &[1, 2, 3, 4, 5, 6, 7, 8], &[], b"1 2 3 4 5 6 7 8", 15);
    }

    #[test]
    fn printf_star_width_precision_and_strings() {
        let (vm, base) = memory();
        let (narrow, wide) = (base, base + 0x100);
        let m = |v: i64| v as u64;
        check(&vm, "%*d|%-*d|%*d|%.*d|%.*d|%*.*s", &[6, 1, 6, 2, m(-6), 3, 3, 4, m(-1), 5, 8, 2, narrow], &[],
            b"     1|2     |3     |004|5|      he", 35);
        check(&vm, "%s|%.2s|%7s|%-7s|%c|%3c|%-3c|%%|%lc", &[narrow, narrow, narrow, narrow, 65, 66, 67, 0x41], &[],
            b"hello|he|  hello|hello  |A|  B|C  |%|A", 38);
        // %ls converts each 16-bit unit on its own: a surrogate pair
        // becomes two three-byte sequences, never a combined character.
        let mut cesu = b"W\xC3\xA9\xE6\x97\xA5".to_vec();
        cesu.extend_from_slice(&[0xED, 0xA0, 0xBD, 0xED, 0xB8, 0x80, b'z']);
        let mut expected = cesu.clone();
        expected.extend_from_slice(b"|W|W\xC3\xA9|W\xC3\xA9\xE6\x97\xA5|");
        expected.extend_from_slice(&cesu);
        check(&vm, "%ls|%.2ls|%.3ls|%.6ls|%8ls", &[wide, wide, wide, wide, wide], &[], &expected, 40);
        // %S and %C are %ls and %lc (Demon's Souls builds resource paths
        // with "%S_b": printed as the letter, every part file was missing).
        let mut expected = cesu.clone();
        expected.extend_from_slice(b"_b|W|\xC3\xA9|  A");
        check(&vm, "%S_b|%.2S|%C|%3C", &[wide, wide, 0xe9, 0x41], &[], &expected, 24);
        check(&vm, "%p|%p|%p", &[0, 0x1234, 0xde_adbe_ef12], &[], b"0|1234|deadbeef12", 17);
        check(&vm, "[%p][%5p][%-5p][%05p][%.8p][%10.6p][%.0p]", &[0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0], &[],
            b"[1234][ 1234][1234 ][01234][00001234][    001234][]", 51);
        check(&vm, "[%#p][%+p][% p][%lp][%20p][%-20p]", &[0x1234; 6], &[],
            b"[1234][1234][1234][1234][                1234][1234                ]", 68);
        check(&vm, "[%08p][%.6p][%8p]", &[0x1a, 0x1a, 0xf], &[], b"[0000001a][00001a][       f]", 28);
        check(&vm, "[%*p][%-*p]", &[6, 0x1a, (-6i64) as u64, 0x1a], &[], b"[    1a][1a    ]", 16);
        check(&vm, "%p", &[u64::MAX], &[], b"ffffffffffffffff", 16);
        // Unsupported directives are copied through with their `%` (the
        // console's behaviour) and take no argument.
        check(&vm, "%Lf|%q|%k", &[], &[1.5], b"%Lf|%q|%k", 9);
        // A trailing '%' emits the terminator, which is counted.
        check(&vm, "abc%", &[], &[], b"abc", 4);
    }

    /// %n stores the bytes produced so far in the width of its length
    /// modifier; a null pointer stores nothing.
    #[test]
    fn printf_n_stores_the_count() {
        let (vm, base) = memory();
        let ctx = CpuContext::new(&vm);
        let count = base + 0x800;
        ctx.write_bytes(count, &[0xAA; 16]);
        check(&vm, "ab%ncd", &[count], &[], b"abcd", 4);
        assert_eq!(ctx.read_bytes(count, 5).unwrap(), [2, 0, 0, 0, 0xAA]);
        check(&vm, "%3d%hhn", &[5, count], &[], b"  5", 3);
        assert_eq!(ctx.read_bytes(count, 2).unwrap(), [3, 0]);
        ctx.write_bytes(count, &[0xAA; 16]);
        check(&vm, "%5d%hn", &[5, count], &[], b"    5", 5);
        assert_eq!(ctx.read_bytes(count, 3).unwrap(), [5, 0, 0xAA]);
        ctx.write_bytes(count, &[0xAA; 16]);
        check(&vm, "%4d%lln|%2d%ln", &[5, count, 6, count + 8], &[], b"   5| 6", 7);
        assert_eq!(ctx.read_u64(count), Some(4));
        assert_eq!(ctx.read_u64(count + 8), Some(7));
        check(&vm, "%ls%n", &[base + 0x100, count], &[], &[0x57, 0xC3, 0xA9, 0xE6, 0x97, 0xA5, 0xED, 0xA0, 0xBD, 0xED, 0xB8, 0x80, 0x7A], 13);
        assert_eq!(ctx.read_u32(count), Some(13));
        check(&vm, "a%nb", &[0], &[], b"ab", 2);
        // A wide character 0 is still one (NUL) byte of output.
        check(&vm, "a%lcb%C", &[0, 0], &[], b"a", 4);
    }

    /// Numbered conversions take their argument by position, '*m$' widths
    /// too; integer and floating-point arguments keep their own classes.
    #[test]
    fn printf_positional_arguments() {
        let (vm, base) = memory();
        check(&vm, "%%ls:%2$d:%1$d", &[3, 7], &[], b"%ls:7:3", 7);
        check(&vm, "%2$s %1$s", &[base + 3, base], &[], b"hello lo", 8);
        check(&vm, "%3$s|%1$d|%2$.2f", &[42, base], &[2.5], b"hello|42|2.50", 13);
        check(&vm, "%1$*2$d|%1$-*2$d|", &[7, 4], &[], b"   7|7   |", 10);
        check(&vm, "%2$.*1$f", &[3], &[1.0], b"1.000", 5);
    }

    #[test]
    fn printf_fixed_point() {
        let (vm, _) = memory();
        check(&vm, "%f|%.2f|%.0f|%.0f|%.0f|%10.3f|%-10.3f|%010.3f|%+f|% f", &[], &[3.14159, 2.675, 0.5, 1.5, 2.5, -3.14159, 3.14159, -3.14159, 1.0, 1.0],
            b"3.141590|2.67|0|2|2|    -3.142|3.142     |-00003.142|+1.000000| 1.000000", 72);
        check(&vm, "%F|%.12f|%.3f|%f|%f", &[], &[1.25, 0.1, 1e-07, 1000000000.0, 10000000000.0],
            b"1.250000|0.100000000000|0.000|1000000000.000000|10000000000.000000", 66);
        check(&vm, "%3.1f|%.1f|%.1f|%.1f|%.1f|%.5f|%.9f|%.10f", &[], &[9.96, 0.25, 0.35, 0.05, 0.15, 1.000005, 0.123456789, 0.5],
            b"10.0|0.2|0.3|0.1|0.1|1.00001|0.123456789|0.5000000000", 53);
        check(&vm, "%08.2f|% 08.2f|%-+8.2f|%8.0f|%#.0f|%.0e|%.0g|%#g", &[], &[-1.5, 1.5, 1.5, 7.5, 3.0, 12345.0, 15.0, 1.0],
            b"-0001.50| 0001.50|+1.50   |       8|3.|1e+04|2e+01|1.00000", 58);
        check(&vm, "%.2f|%.1f|%.3f|%.2f|%.0f|%.1f|%f|%.15f", &[], &[0.995, 9.95, 1.0005, 99.995, 999999.5, 0.95, 1e+20, 0.1],
            b"0.99|9.9|1.000|100.00|1000000|0.9|100000000000000000000.000000|0.100000000000000", 80);
        check(&vm, "%.2f|%.3f|%.1f|%.4f|%.0f|%.20f|%#.0f", &[], &[0.125, 0.0625, 99.95, 9.99995, 0.5000001, 0.1, 0.5],
            b"0.12|0.062|100.0|10.0000|1|0.10000000000000000555|0.", 52);
        check(&vm, "%f|%e|%g|%+.1f|% .0e", &[], &[-0.0, -0.0, -0.0, -0.0, 0.0],
            b"-0.000000|-0.000000e+00|-0|-0.0| 0e+00", 38);
    }

    /// A conversion the console does not have (%m among them) is copied
    /// through as `%` and the letter and takes no argument; a NaN carries
    /// its sign bit, with '+' and ' ' giving their signs, and the sign
    /// flags do not apply to unsigned conversions.
    #[test]
    fn printf_unknown_nan_and_unsigned_flags() {
        std::thread::spawn(|| {
            // A page of its own: the shared one is remapped by other tests.
            const OWN: u64 = BASE + 0x10_0000;
            let vm = ps5vm::VirtualMemory::new();
            vm.map(OWN, 0x1000, ps5vm::MemProtect::RW).unwrap();
            let ctx = CpuContext::new(&vm);
            ctx.write_bytes(OWN, b"hello\0");
            let errno = crate::kernel::bind_test_errno(&ctx);
            ctx.write_u32(errno, 35);
            check(&vm, "[%m][%.8m][%-10.5m][%d]", &[7], &[], b"[%m][%m][%m        ][7]", 23);
            assert_eq!(ctx.read_u32(errno), Some(35));
            check(&vm, "%s: %m", &[OWN], &[], b"hello: %m", 9);
            check(&vm, "[%f][%+f][% F][%5.1e]", &[], &[-f64::NAN, -f64::NAN, f64::NAN, -f64::NAN], b"[-nan][-nan][ NAN][ -nan]", 25);
            check(&vm, "[% u][% 4u][%+ lu][%- 3hhu][% x][%+o]", &[5, 5, 5, 5, 255, 8], &[], b"[5][   5][5][5  ][ff][10]", 25);
            let _ = vm.unmap(OWN, 0x1000);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn printf_exponent_and_special_values() {
        let (vm, _) = memory();
        check(&vm, "%e|%E|%.2e|%e|%e|%12.3e|%-12.3e|%e", &[], &[12345.678, 0.000123, 1.0, 0.0, 1e+100, 3.5, 3.5, -2.5e-300],
            b"1.234568e+04|1.230000E-04|1.00e+00|0.000000e+00|1.000000e+100|   3.500e+00|3.500e+00   |-2.500000e-300", 102);
        check(&vm, "%g|%g|%g|%g|%g|%G|%.3g|%.10g|%g", &[], &[100000.0, 1000000.0, 0.0001, 1e-05, 0.0, 1e-10, 3.14159, 3.14159, 123.456],
            b"100000|1e+06|0.0001|1e-05|0|1E-10|3.14|3.14159|123.456", 54);
        check(&vm, "%10.4g|%-10.2e|%010.2e|%+.3e|%e|%E", &[], &[1.2345e-05, 98765.4321, 98765.4321, 9.9999, 9.9999995, 1e-05],
            b" 1.234e-05|9.88e+04  |009.88e+04|+1.000e+01|9.999999e+00|1.000000E-05", 69);
        check(&vm, "%g|%g|%.0e|%#.0e|%#.3g|%g|%G|%g|%.1g", &[], &[999999.5, 0.0, 0.0, 2.0, 1.0, 1e-05, 1e+300, 123456789.0, 0.05],
            b"1e+06|0|0e+00|2.e+00|1.00|1e-05|1E+300|1.23457e+08|0.05", 55);
        check(&vm, "%5.1e|%-9.2g|%09.3g|%+g|%#g|%.0g|%#.0g|%e", &[], &[123.0, 0.0001234, -1.5, 1000000.0, 100000.0, 0.0, 0.0, 5e-324],
            b"1.2e+02|0.00012  |-000001.5|+1e+06|100000.|0|0.|4.940656e-324", 61);
        check(&vm, "%g|%g|%.3g|%.4e|%.17g|%g", &[], &[9.999995e-05, 0.0001, 9995.0, 1.00005, 0.1, 1.7976931348623157e+308],
            b"0.0001|0.0001|1e+04|1.0001e+00|0.10000000000000001|1.79769e+308", 63);
        check(&vm, "%f|%f|%f|%e|%g|%5f|%-6f|%+f", &[],
            &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, -f64::NAN, f64::INFINITY, f64::NAN, f64::INFINITY],
            b"nan|inf|-inf|inf|-nan|  inf|nan   |+inf", 39);
        // No zero padding for non-finite values; upper-case conversions.
        check(&vm, "%010f|%F|%E|%G|%-+6e", &[], &[f64::INFINITY, f64::NAN, f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY],
            b"       inf|NAN|-INF|INF|+inf  ", 30);
    }

    #[test]
    fn printf_hex_float() {
        let (vm, _) = memory();
        check(&vm, "%a|%A|%a|%a|%a|%010a|%-12a|%a", &[], &[1.5, 255.0, 0.0, 0.1, 5e-324, 1.0, -2.0, f64::MAX],
            b"0x1.8p+0|0X1.FEP+7|0x0p+0|0x1.999999999999ap-4|0x1p-1074|0x00001p+0|-0x1p+1     |0x1.fffffffffffffp+1023", 104);
        // Rounding to nearest even; a carry into the leading digit renormalizes.
        check(&vm, "%.2a|%.0a|%.1a|%.1a|%.1a|%#.0a|%.15a", &[], &[1.999, 1.5, 1.96875, 1.03125, 1.09375, 1.0, 1.0],
            b"0x1.00p+1|0x1p+1|0x1.0p+1|0x1.0p+0|0x1.2p+0|0x1.p+0|0x1.000000000000000p+0", 74);
    }
}
