// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! scanf-family conversion engine.
//!
//! Reads characters from a source with one character of lookahead and
//! stores converted values through guest pointers taken from an argument
//! source. Supports `%[*][width][hh|h|l|ll|j|z|t|L|q]` followed by
//! `d i o u x X p f F e E g G a A c s [ n %`. Wide (`l`) strings and
//! characters are stored as 16-bit units.
//!
//! The engine works on 16-bit units: byte sources and formats are widened,
//! which leaves their behaviour unchanged, and the wide functions
//! (swscanf_s) pass guest `wchar_t` units. Numbers only ever consist of
//! ASCII units. The secure (`_s`) variants take a capacity after the
//! pointer of every `c`, `s` and `[` conversion and treat a null target as
//! a runtime-constraint violation.

use crate::context::CpuContext;
use crate::printf::ArgSource;

/// Result value for an input failure before the first conversion.
pub const EOF: i64 = -1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Len {
    None,
    Char,
    Short,
    Long,
    LongDouble,
}

/// How a scan reads its input and arguments.
#[derive(Clone, Copy, Default, Debug)]
pub struct Options {
    /// The `_s` rules: capacities after string targets, null targets fail.
    pub secure: bool,
    /// The input is wide: a narrow string target receives each unit as one
    /// C-locale byte (units of 256 and up cannot be stored narrow).
    pub wide_input: bool,
}

struct Reader<'a> {
    src: &'a mut dyn FnMut() -> Option<u16>,
    peeked: Option<Option<u16>>,
    count: u64,
}

impl Reader<'_> {
    fn peek(&mut self) -> Option<u16> {
        if self.peeked.is_none() {
            self.peeked = Some((self.src)());
        }
        self.peeked.unwrap()
    }

    /// The lookahead as an ASCII byte (None for wider units and the end).
    fn peek_ascii(&mut self) -> Option<u8> {
        self.peek().filter(|&u| u < 0x80).map(|u| u as u8)
    }

    fn bump(&mut self) -> Option<u16> {
        let c = self.peek();
        if c.is_some() {
            self.count += 1;
            self.peeked = None;
        }
        c
    }

    fn skip_ws(&mut self) {
        while self.peek().is_some_and(is_space) {
            self.bump();
        }
    }
}

/// C-locale white space (the same for bytes and wide units).
fn is_space(c: u16) -> bool {
    matches!(c, 0x20 | 0x09 | 0x0A | 0x0D | 0x0B | 0x0C)
}

/// Outcome of one directive.
enum Step {
    Ok,
    /// Input ended (input failure).
    Eof,
    /// Input did not match (matching failure).
    Mismatch,
}

/// Run the conversion over a byte format and byte source. Returns the
/// number of assigned items (or `EOF`) and a byte that was read from the
/// source but not consumed, which the caller must push back.
pub fn scan(
    ctx: &CpuContext,
    fmt: &[u8],
    src: &mut dyn FnMut() -> Option<u8>,
    args: &mut dyn ArgSource,
) -> (i64, Option<u8>) {
    let fmt: Vec<u16> = fmt.iter().map(|&b| b as u16).collect();
    let mut units = || src().map(u16::from);
    let (n, left) = scan_units(ctx, &fmt, &mut units, args, Options::default());
    (n, left.map(|u| u as u8))
}

/// Run the conversion over 16-bit units with `opts`; see [`scan`].
pub fn scan_units(
    ctx: &CpuContext,
    fmt: &[u16],
    src: &mut dyn FnMut() -> Option<u16>,
    args: &mut dyn ArgSource,
    opts: Options,
) -> (i64, Option<u16>) {
    let mut r = Reader { src, peeked: None, count: 0 };
    let mut assigned = 0i64;
    let mut converted_any = false;
    let mut i = 0usize;
    let result = loop {
        if i >= fmt.len() {
            break assigned;
        }
        let f = fmt[i];
        if is_space(f) {
            while i < fmt.len() && is_space(fmt[i]) {
                i += 1;
            }
            r.skip_ws();
            continue;
        }
        if f != b'%' as u16 {
            i += 1;
            match r.peek() {
                Some(c) if c == f => {
                    r.bump();
                    continue;
                }
                Some(_) => break assigned,
                None => break if converted_any { assigned } else { EOF },
            }
        }
        // Directive.
        i += 1;
        if fmt.get(i) == Some(&(b'%' as u16)) {
            i += 1;
            r.skip_ws();
            match r.peek() {
                Some(0x25) => {
                    r.bump();
                    continue;
                }
                Some(_) => break assigned,
                None => break if converted_any { assigned } else { EOF },
            }
        }
        let suppress = fmt.get(i) == Some(&(b'*' as u16));
        if suppress {
            i += 1;
        }
        let mut width = 0usize;
        while let Some(d) = fmt.get(i).filter(|&&c| (b'0' as u16..=b'9' as u16).contains(&c)) {
            width = width.saturating_mul(10).saturating_add((d - b'0' as u16) as usize);
            i += 1;
        }
        let mut len = Len::None;
        while let Some(&c) = fmt.get(i) {
            match u8::try_from(c).unwrap_or(0) {
                b'h' => len = if len == Len::Short { Len::Char } else { Len::Short },
                b'l' | b'j' | b'z' | b't' | b'q' => len = Len::Long,
                b'L' => len = Len::LongDouble,
                _ => break,
            }
            i += 1;
        }
        let Some(&conv_unit) = fmt.get(i) else { break assigned };
        let conv = u8::try_from(conv_unit).unwrap_or(0);
        i += 1;
        if conv == b'n' {
            if !suppress {
                let p = args.next_gp();
                if p == 0 && opts.secure {
                    break EOF;
                }
                store_int(ctx, p, r.count as i64, len);
            }
            continue;
        }
        let set = if conv == b'[' {
            match parse_scanset(fmt, &mut i) {
                Some(s) => Some(s),
                None => break assigned,
            }
        } else {
            None
        };
        let text = matches!(conv, b'c' | b's' | b'[');
        let (dest, capacity) = if suppress {
            (0, None)
        } else {
            let dest = args.next_gp();
            let capacity = (opts.secure && text).then(|| args.next_gp() as u32 as u64);
            (dest, capacity)
        };
        if !suppress && dest == 0 && opts.secure {
            break EOF;
        }
        if !matches!(conv, b'c' | b'[') {
            r.skip_ws();
        }
        if r.peek().is_none() {
            break if converted_any { assigned } else { EOF };
        }
        let target = Target { dest, capacity, len, wide_input: opts.wide_input };
        let step = match conv {
            b'd' => scan_int(ctx, &mut r, width, 10, dest, len),
            b'i' => scan_int(ctx, &mut r, width, 0, dest, len),
            b'u' => scan_int(ctx, &mut r, width, 10, dest, len),
            b'o' => scan_int(ctx, &mut r, width, 8, dest, len),
            b'x' | b'X' => scan_int(ctx, &mut r, width, 16, dest, len),
            b'p' => scan_int(ctx, &mut r, width, 16, dest, Len::Long),
            b'f' | b'F' | b'e' | b'E' | b'g' | b'G' | b'a' | b'A' => scan_float(ctx, &mut r, width, dest, len),
            b'c' => scan_chars(ctx, &mut r, if width == 0 { 1 } else { width }, &target),
            b's' => scan_string(ctx, &mut r, width, &target, |c| !is_space(c)),
            b'[' => {
                let set = set.unwrap();
                scan_string(ctx, &mut r, width, &target, |c| set.contains(c))
            }
            _ => Step::Mismatch,
        };
        match step {
            Step::Ok => {
                converted_any = true;
                if !suppress {
                    assigned += 1;
                }
            }
            Step::Eof => break if converted_any { assigned } else { EOF },
            Step::Mismatch => break assigned,
        }
    };
    let leftover = r.peeked.flatten();
    (result, leftover)
}

/// A `[...]` scanset: unit ranges, possibly negated.
struct Scanset {
    ranges: Vec<(u16, u16)>,
    negate: bool,
}

impl Scanset {
    fn contains(&self, c: u16) -> bool {
        self.ranges.iter().any(|&(a, b)| (a..=b).contains(&c)) != self.negate
    }
}

/// Parse a `[...]` scanset starting after the `[`; leaves `i` after `]`.
/// A `]` first in the set is a member; `a-b` is a range.
fn parse_scanset(fmt: &[u16], i: &mut usize) -> Option<Scanset> {
    let close = b']' as u16;
    let dash = b'-' as u16;
    let negate = fmt.get(*i) == Some(&(b'^' as u16));
    if negate {
        *i += 1;
    }
    let mut ranges = Vec::new();
    let mut first = true;
    loop {
        let &c = fmt.get(*i)?;
        if c == close && !first {
            *i += 1;
            break;
        }
        first = false;
        if fmt.get(*i + 1) == Some(&dash) && fmt.get(*i + 2).is_some_and(|&e| e != close) {
            let e = fmt[*i + 2];
            if c <= e {
                ranges.push((c, e));
            }
            *i += 3;
        } else {
            ranges.push((c, c));
            *i += 1;
        }
    }
    Some(Scanset { ranges, negate })
}

fn store_int(ctx: &CpuContext, p: u64, v: i64, len: Len) {
    if p == 0 {
        return;
    }
    let _ = match len {
        Len::Char => ctx.write_bytes(p, &[v as u8]),
        Len::Short => ctx.write_bytes(p, &(v as u16).to_le_bytes()),
        Len::None => ctx.write_u32(p, v as u32),
        Len::Long | Len::LongDouble => ctx.write_u64(p, v as u64),
    };
}

fn digit_value(c: u8) -> Option<u32> {
    (c as char).to_digit(16)
}

fn scan_int(ctx: &CpuContext, r: &mut Reader, width: usize, mut base: u32, dest: u64, len: Len) -> Step {
    let max = if width == 0 { usize::MAX } else { width };
    let mut used = 0usize;
    let mut neg = false;
    if let Some(c @ (b'+' | b'-')) = r.peek_ascii() {
        if used < max {
            neg = c == b'-';
            r.bump();
            used += 1;
        }
    }
    let mut digits = 0usize;
    let mut value: u64 = 0;
    if (base == 0 || base == 16) && used < max && r.peek_ascii() == Some(b'0') {
        r.bump();
        used += 1;
        digits = 1;
        if used < max && matches!(r.peek_ascii(), Some(b'x' | b'X')) {
            r.bump();
            used += 1;
            base = 16;
            digits = 0;
        } else if base == 0 {
            base = 8;
        }
    }
    if base == 0 {
        base = 10;
    }
    while used < max {
        let Some(c) = r.peek_ascii() else { break };
        let Some(d) = digit_value(c).filter(|&d| d < base) else { break };
        r.bump();
        used += 1;
        digits += 1;
        value = value.wrapping_mul(base as u64).wrapping_add(d as u64);
    }
    if digits == 0 {
        return if r.peek().is_none() && used == 0 { Step::Eof } else { Step::Mismatch };
    }
    let v = if neg { (value as i64).wrapping_neg() } else { value as i64 };
    store_int(ctx, dest, v, len);
    Step::Ok
}

/// Collect the longest prefix that looks like a floating-point number.
fn collect_float(r: &mut Reader, max: usize) -> String {
    let mut s = String::new();
    let take = |r: &mut Reader, s: &mut String| {
        if let Some(c) = r.bump() {
            s.push(c as u8 as char);
        }
    };
    if s.len() < max && matches!(r.peek_ascii(), Some(b'+' | b'-')) {
        take(r, &mut s);
    }
    // inf / infinity / nan
    if s.len() < max && matches!(r.peek_ascii(), Some(b'i' | b'I' | b'n' | b'N')) {
        let words: &[&str] = &["infinity", "inf", "nan"];
        let mut word = String::new();
        while s.len() + word.len() < max {
            let Some(c) = r.peek_ascii() else { break };
            let mut cand = word.clone();
            cand.push(c.to_ascii_lowercase() as char);
            if !words.iter().any(|w| w.starts_with(&cand)) {
                break;
            }
            r.bump();
            word = cand;
        }
        s.push_str(&word);
        return s;
    }
    let mut hex = false;
    if s.len() < max && r.peek_ascii() == Some(b'0') {
        take(r, &mut s);
        if s.len() < max && matches!(r.peek_ascii(), Some(b'x' | b'X')) {
            take(r, &mut s);
            hex = true;
        }
    }
    let is_digit = |c: u8| if hex { c.is_ascii_hexdigit() } else { c.is_ascii_digit() };
    while s.len() < max && r.peek_ascii().is_some_and(is_digit) {
        take(r, &mut s);
    }
    if s.len() < max && r.peek_ascii() == Some(b'.') {
        take(r, &mut s);
        while s.len() < max && r.peek_ascii().is_some_and(is_digit) {
            take(r, &mut s);
        }
    }
    let exp = if hex { [b'p', b'P'] } else { [b'e', b'E'] };
    let has_mantissa = s.bytes().any(|c| if hex { c.is_ascii_hexdigit() && c != b'x' && c != b'X' } else { c.is_ascii_digit() })
        || (hex && s.len() >= 2);
    if has_mantissa && s.len() < max && r.peek_ascii().is_some_and(|c| exp.contains(&c)) {
        take(r, &mut s);
        if s.len() < max && matches!(r.peek_ascii(), Some(b'+' | b'-')) {
            take(r, &mut s);
        }
        while s.len() < max && r.peek_ascii().is_some_and(|c| c.is_ascii_digit()) {
            take(r, &mut s);
        }
    }
    s
}

/// Parse the collected text; None when it is not a complete number.
pub fn parse_float(text: &str) -> Option<f64> {
    let (neg, body) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let lower = body.to_ascii_lowercase();
    let v = if lower == "inf" || lower == "infinity" {
        f64::INFINITY
    } else if lower == "nan" {
        f64::NAN
    } else if let Some(hex) = lower.strip_prefix("0x") {
        parse_hex_float(hex)?
    } else {
        if lower.ends_with(['e', '+', '-']) {
            return None;
        }
        lower.parse::<f64>().ok()?
    };
    Some(if neg { -v } else { v })
}

fn parse_hex_float(s: &str) -> Option<f64> {
    let (mant, exp) = match s.find('p') {
        Some(p) => (&s[..p], s[p + 1..].parse::<i32>().ok()?),
        None => (s, 0),
    };
    let (int, frac) = match mant.find('.') {
        Some(d) => (&mant[..d], &mant[d + 1..]),
        None => (mant, ""),
    };
    if int.is_empty() && frac.is_empty() {
        // "0x" alone reads as zero.
        return Some(0.0);
    }
    let mut v = 0f64;
    for c in int.chars() {
        v = v * 16.0 + c.to_digit(16)? as f64;
    }
    let mut scale = 1.0 / 16.0;
    for c in frac.chars() {
        v += c.to_digit(16)? as f64 * scale;
        scale /= 16.0;
    }
    Some(v * 2f64.powi(exp))
}

/// x87 80-bit extended encoding of `v`.
pub fn f64_to_x87(v: f64) -> [u8; 10] {
    let bits = v.to_bits();
    let sign = ((bits >> 63) as u16) << 15;
    let exp = ((bits >> 52) & 0x7FF) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (e, m): (u16, u64) = if exp == 0 {
        if frac == 0 {
            (0, 0)
        } else {
            // Subnormal double: normalise.
            let shift = frac.leading_zeros() - 11;
            let m = frac << (shift + 11);
            ((1 - 1023 + 16383 - shift as i32) as u16, m)
        }
    } else if exp == 0x7FF {
        (0x7FFF, (1u64 << 63) | (frac << 11))
    } else {
        ((exp - 1023 + 16383) as u16, (1u64 << 63) | (frac << 11))
    };
    let mut out = [0u8; 10];
    out[..8].copy_from_slice(&m.to_le_bytes());
    out[8..].copy_from_slice(&(sign | e).to_le_bytes());
    out
}

fn scan_float(ctx: &CpuContext, r: &mut Reader, width: usize, dest: u64, len: Len) -> Step {
    let max = if width == 0 { usize::MAX } else { width };
    let text = collect_float(r, max);
    let Some(v) = parse_float(&text) else {
        return Step::Mismatch;
    };
    if dest != 0 {
        let _ = match len {
            Len::Long => ctx.write_u64(dest, v.to_bits()),
            Len::LongDouble => ctx.write_bytes(dest, &f64_to_x87(v)),
            _ => ctx.write_u32(dest, (v as f32).to_bits()),
        };
    }
    Step::Ok
}

/// Where a `c`, `s` or `[` conversion stores its text.
struct Target {
    dest: u64,
    /// The `_s` capacity in elements of the target.
    capacity: Option<u64>,
    len: Len,
    wide_input: bool,
}

fn scan_chars(ctx: &CpuContext, r: &mut Reader, width: usize, t: &Target) -> Step {
    let mut out = Vec::with_capacity(width);
    while out.len() < width {
        match r.bump() {
            Some(c) => out.push(c),
            None => break,
        }
    }
    if out.len() < width {
        // The wide scanner treats a field cut short by the end of the input
        // as a matching failure (0 conversions, not EOF); the input was not
        // empty when the field began.
        return if t.wide_input { Step::Mismatch } else { Step::Eof };
    }
    store_text(ctx, t, &out, false)
}

fn scan_string(ctx: &CpuContext, r: &mut Reader, width: usize, t: &Target, accept: impl Fn(u16) -> bool) -> Step {
    let max = if width == 0 { usize::MAX } else { width };
    let mut out = Vec::new();
    while out.len() < max {
        match r.peek() {
            Some(c) if accept(c) => {
                r.bump();
                out.push(c);
            }
            _ => break,
        }
    }
    if out.is_empty() {
        return if r.peek().is_none() { Step::Eof } else { Step::Mismatch };
    }
    store_text(ctx, t, &out, true)
}

/// Store the scanned units: as 16-bit units for an `l` target, else as
/// bytes. A `_s` target too small for the text (and its terminator) gets
/// an empty string and the conversion is a matching failure.
fn store_text(ctx: &CpuContext, t: &Target, units: &[u16], terminate: bool) -> Step {
    if t.dest == 0 {
        return Step::Ok;
    }
    let wide = t.len == Len::Long;
    let elements = units.len() as u64 + terminate as u64;
    if t.capacity.is_some_and(|cap| elements > cap) {
        if t.capacity != Some(0) {
            let _ = ctx.write_bytes(t.dest, if wide { &[0, 0][..] } else { &[0][..] });
        }
        return Step::Mismatch;
    }
    let mut buf = Vec::with_capacity(units.len() * 2 + 2);
    if wide {
        for &u in units {
            buf.extend_from_slice(&u.to_le_bytes());
        }
        if terminate {
            buf.extend_from_slice(&[0, 0]);
        }
    } else {
        for &u in units {
            // A wide unit is stored as its C-locale byte; narrow input
            // units are bytes already.
            if t.wide_input && u >= 0x100 {
                log::warn!("[libc] scanf: wide character {u:#x} has no single-byte form");
                return Step::Mismatch;
            }
            buf.push(u as u8);
        }
        if terminate {
            buf.push(0);
        }
    }
    let _ = ctx.write_bytes(t.dest, &buf);
    Step::Ok
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Ptrs(Vec<u64>, usize);
    impl ArgSource for Ptrs {
        fn next_gp(&mut self) -> u64 {
            self.1 += 1;
            self.0[self.1 - 1]
        }
        fn next_fp(&mut self) -> f64 {
            0.0
        }
    }

    fn run(ctx: &CpuContext, fmt: &str, input: &str, ptrs: Vec<u64>) -> (i64, Option<u8>, usize) {
        let bytes = input.as_bytes().to_vec();
        let mut pos = 0usize;
        let mut src = || {
            let c = bytes.get(pos).copied();
            if c.is_some() {
                pos += 1;
            }
            c
        };
        let mut args = Ptrs(ptrs, 0);
        let (n, left) = scan(ctx, fmt.as_bytes(), &mut src, &mut args);
        (n, left, pos)
    }

    /// A wide or narrow `_s` scan over a whole string.
    fn run_units(ctx: &CpuContext, fmt: &str, input: &str, args: Vec<u64>, wide_input: bool) -> i64 {
        let units: Vec<u16> = if wide_input { input.encode_utf16().collect() } else { input.bytes().map(u16::from).collect() };
        let fmt: Vec<u16> = if wide_input { fmt.encode_utf16().collect() } else { fmt.bytes().map(u16::from).collect() };
        let mut pos = 0usize;
        let mut src = || {
            let c = units.get(pos).copied();
            if c.is_some() {
                pos += 1;
            }
            c
        };
        let mut args = Ptrs(args, 0);
        scan_units(ctx, &fmt, &mut src, &mut args, Options { secure: true, wide_input }).0
    }

    #[test]
    fn integers_strings_and_sets() {
        let base: u64 = 0x7A30_0000_0000 + 0 * 0x10_0000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(base, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let ctx = CpuContext::new(&vm);
        let (n, left, _) = run(&ctx, "%d %x %s %[a-c]%n", "  -42 0x1F hello abcz", vec![base, base + 8, base + 16, base + 48, base + 64]);
        assert_eq!(n, 4);
        assert_eq!(ctx.read_u32(base), Some(-42i32 as u32));
        assert_eq!(ctx.read_u32(base + 8), Some(0x1F));
        assert_eq!(ctx.read_cstring(base + 16, 16).as_deref(), Some("hello"));
        assert_eq!(ctx.read_cstring(base + 48, 16).as_deref(), Some("abc"));
        assert_eq!(ctx.read_u32(base + 64), Some(20));
        assert_eq!(left, Some(b'z'));
    }

    #[test]
    fn widths_lengths_and_suppression() {
        let base: u64 = 0x7A30_0000_0000 + 1 * 0x10_0000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(base, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let ctx = CpuContext::new(&vm);
        let (n, _, _) = run(&ctx, "%3d%*d %lld %hhu %i %c", "12345 -7000000000 300 010 Q", vec![base, base + 8, base + 16, base + 24, base + 32]);
        assert_eq!(n, 5);
        assert_eq!(ctx.read_u32(base), Some(123));
        assert_eq!(ctx.read_u64(base + 8), Some(-7_000_000_000i64 as u64));
        assert_eq!(ctx.read_bytes(base + 16, 1).unwrap()[0], 300u32 as u8);
        assert_eq!(ctx.read_u32(base + 24), Some(8));
        assert_eq!(ctx.read_bytes(base + 32, 1).unwrap()[0], b'Q');
    }

    #[test]
    fn floats() {
        let base: u64 = 0x7A30_0000_0000 + 2 * 0x10_0000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(base, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let ctx = CpuContext::new(&vm);
        let (n, _, _) = run(&ctx, "%f %lf %Lf %le", "1.5 -2.25e2 3 0x1p4", vec![base, base + 8, base + 16, base + 32]);
        assert_eq!(n, 4);
        assert_eq!(ctx.read_f32(base), Some(1.5));
        assert_eq!(ctx.read_u64(base + 8).map(f64::from_bits), Some(-225.0));
        assert_eq!(ctx.read_bytes(base + 16, 10).unwrap(), f64_to_x87(3.0).to_vec());
        assert_eq!(ctx.read_u64(base + 32).map(f64::from_bits), Some(16.0));
        assert_eq!(f64_to_x87(1.0), [0, 0, 0, 0, 0, 0, 0, 0x80, 0xFF, 0x3F]);
        assert_eq!(parse_float("inf"), Some(f64::INFINITY));
        assert_eq!(parse_float("1e"), None);
    }

    #[test]
    fn failures() {
        let base: u64 = 0x7A30_0000_0000 + 3 * 0x10_0000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(base, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let ctx = CpuContext::new(&vm);
        assert_eq!(run(&ctx, "%d", "", vec![base]).0, EOF);
        assert_eq!(run(&ctx, "%d", "   ", vec![base]).0, EOF);
        let (n, left, _) = run(&ctx, "%d", "abc", vec![base]);
        assert_eq!((n, left), (0, Some(b'a')));
        assert_eq!(run(&ctx, "%d,%d", "1;2", vec![base, base + 8]).0, 1);
        assert_eq!(run(&ctx, "x%%%d", "x% 5", vec![base]).0, 1);
        assert_eq!(ctx.read_u32(base), Some(5));
    }

    /// vsscanf_s: capacities after string targets; a target too small gets
    /// an empty string.
    #[test]
    fn secure_narrow_scan_checks_capacities() {
        let base: u64 = 0x7A30_0000_0000 + 4 * 0x10_0000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(base, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let ctx = CpuContext::new(&vm);
        let (num, word, small) = (base, base + 0x10, base + 0x20);
        assert_eq!(run_units(&ctx, "%d %s", "123 test", vec![num, word, 8], false), 2);
        assert_eq!(ctx.read_u32(num), Some(123));
        assert_eq!(ctx.read_cstring(word, 8).as_deref(), Some("test"));
        ctx.write_bytes(small, b"xxxx");
        assert_eq!(run_units(&ctx, "%s", "abcdef", vec![small, 4], false), 0);
        assert_eq!(ctx.read_bytes(small, 1).unwrap(), b"\0");
        assert_eq!(run_units(&ctx, "%d", "5", vec![0], false), EOF, "a null target");
    }

    /// swscanf_s over guest wide units.
    #[test]
    fn secure_wide_scan() {
        let base: u64 = 0x7A30_0000_0000 + 5 * 0x10_0000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(base, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let ctx = CpuContext::new(&vm);
        let (a, b, text, narrow) = (base, base + 0x10, base + 0x100, base + 0x200);
        let scan = |fmt: &str, input: &str, args: Vec<u64>| run_units(&ctx, fmt, input, args, true);
        let units = |addr: u64, n: usize| -> Vec<u16> {
            ctx.read_bytes(addr, n * 2).unwrap().chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
        };
        let wide_str = |addr: u64| String::from_utf16_lossy(&ctx.read_wide_string(addr, 64).unwrap());
        assert_eq!(scan("%d%d", "  42 -7", vec![a, b]), 2);
        assert_eq!((ctx.read_u32(a), ctx.read_u32(b)), (Some(42), Some(-7i32 as u32)));
        assert_eq!(scan("x=%i", "x=0x1f", vec![a]), 1);
        assert_eq!(ctx.read_u32(a), Some(31));
        assert_eq!(scan("%*d%d", "1 2", vec![a]), 1);
        assert_eq!(ctx.read_u32(a), Some(2));
        assert_eq!(scan("%3d%d", "12345", vec![a, b]), 2);
        assert_eq!((ctx.read_u32(a), ctx.read_u32(b)), (Some(123), Some(45)));
        assert_eq!(scan("%ld", "5000000000", vec![a]), 1);
        assert_eq!(ctx.read_u64(a), Some(5_000_000_000));
        assert_eq!(scan("%hd", "-2", vec![a]), 1);
        assert_eq!(ctx.read_bytes(a, 2).unwrap(), (-2i16).to_le_bytes());
        assert_eq!(scan("%x", "0xg", vec![a]), 0, "a prefix without digits");

        assert_eq!(scan("%ls", "hello world", vec![text, 6]), 1);
        assert_eq!(wide_str(text), "hello");
        ctx.write_bytes(text, &[b'?'; 10]);
        assert_eq!(scan("%ls", "hello", vec![text, 5]), 0, "too small");
        assert_eq!(units(text, 1), [0]);
        assert_eq!(scan("%d%ls", "7 hello", vec![a, text, 5]), 1);
        assert_eq!(scan("%s", "abc def", vec![narrow, 4]), 1);
        assert_eq!(ctx.read_cstring(narrow, 8).as_deref(), Some("abc"));
        ctx.write_bytes(text, &[b'?', 0, b'?', 0, b'?', 0]);
        assert_eq!(scan("%2lc", "xyz", vec![text, 2]), 1);
        assert_eq!(units(text, 3), [b'x' as u16, b'y' as u16, b'?' as u16]);
        assert_eq!(scan("%2lc", "xyz", vec![text, 1]), 0, "%2lc too small");
        assert_eq!(scan("%2lc", "x", vec![text, 2]), 0, "%2lc short input");
        assert_eq!(scan("%l[abc]%d", "cab12", vec![text, 6, a]), 2);
        assert_eq!((wide_str(text).as_str(), ctx.read_u32(a)), ("cab", Some(12)));
        assert_eq!(scan("%l[^!]", "\u{e9}t\u{e9}!", vec![text, 6]), 1);
        assert_eq!(wide_str(text), "\u{e9}t\u{e9}");
        assert_eq!(scan("%d %ln", " 12 ab", vec![a, b]), 1);
        assert_eq!(ctx.read_u64(b), Some(4));

        assert_eq!(scan("%lf", "1.5e3", vec![a]), 1);
        assert_eq!(ctx.read_u64(a).map(f64::from_bits), Some(1500.0));
        assert_eq!(scan("%lf", "1e+x", vec![a]), 0, "an incomplete exponent");
        assert_eq!(scan("%f", "-inf", vec![a]), 1);
        assert_eq!(ctx.read_f32(a), Some(f32::NEG_INFINITY));

        assert_eq!(scan("b%d", "a", vec![a]), 0, "a literal mismatch");
        assert_eq!(scan("%d", "", vec![a]), EOF);
        assert_eq!(scan("%ls", "   ", vec![text, 6]), EOF);
        assert_eq!(scan("%d%d", "5", vec![a, b]), 1);
        assert_eq!(scan("%d", "5", vec![0]), EOF, "a null %d target");
        assert_eq!(scan("%d%ls", "5 abc", vec![a, 0, 4]), EOF, "a null %ls target");
        assert_eq!(scan("%s", "\u{100}", vec![narrow, 4]), 0, "no single-byte form");
    }
}
