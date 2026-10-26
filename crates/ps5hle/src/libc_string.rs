// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libc string helpers served when no loaded libc module exports them:
//! byte search, case-insensitive compare, bounded concatenation, `perror`,
//! `strerror` / `strerror_r` (FreeBSD errlist texts) and the wide (UTF-16, `wchar_t` is 2 bytes) string family.
//!
//! The `_s` variants return errno values (0, EINVAL, ERANGE, STRUNCATE) and
//! leave an empty string in the destination on failure.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceLibcInternal";

const EINVAL: HleResult = 22;
const ERANGE: HleResult = 34;
const STRUNCATE: HleResult = 80;
/// Longest string any of these helpers walks.
const MAX_UNITS: usize = 1 << 20;
const MAX_PERROR_TEXT: usize = 2048;

// ---------------------------------------------------------------------------
// Narrow strings
// ---------------------------------------------------------------------------

/// memchr(s, c, n) -> first byte equal to (unsigned char)c, or NULL.
fn cb_memchr(ctx: &mut CpuContext) -> HleResult {
    let (addr, needle, count) = (ctx.rdi(), ctx.rsi() as u8, ctx.rdx());
    let mut done = 0u64;
    let mut chunk = [0u8; 256];
    while done < count {
        let cur = addr.wrapping_add(done);
        // Chunks stay inside one page so a search ending at a mapping's
        // edge never reads past it.
        let page_left = 0x1000 - (cur & 0xFFF);
        let n = page_left.min(chunk.len() as u64).min(count - done) as usize;
        let Some(bytes) = ctx.read_bytes(cur, n) else { return 0 };
        chunk[..n].copy_from_slice(&bytes);
        if let Some(i) = chunk[..n].iter().position(|&b| b == needle) {
            return (cur + i as u64) as HleResult;
        }
        done += n as u64;
    }
    0
}

/// memrchr(s, c, n) -> last byte of the n equal to (unsigned char)c, or
/// NULL.
fn cb_memrchr(ctx: &mut CpuContext) -> HleResult {
    let (addr, needle, count) = (ctx.rdi(), ctx.rsi() as u8, ctx.rdx());
    let mut left = count;
    while left > 0 {
        let end = addr.wrapping_add(left);
        // Chunks stay inside one page, from the end backwards.
        let page_used = match end & 0xFFF {
            0 => 0x1000,
            n => n,
        };
        let n = page_used.min(256).min(left);
        let start = end - n;
        let Some(bytes) = ctx.read_bytes(start, n as usize) else { return 0 };
        if let Some(i) = bytes.iter().rposition(|&b| b == needle) {
            return (start + i as u64) as HleResult;
        }
        left -= n;
    }
    0
}

/// strsep(char** text, delimiters) -> the token at *text, cut at the first
/// delimiter (empty tokens included); *text moves past the delimiter, or
/// becomes NULL after the last token. NULL when *text is NULL.
fn cb_strsep(ctx: &mut CpuContext) -> HleResult {
    let (cursor, delimiters) = (ctx.rdi(), ctx.rsi());
    let Some(token) = ctx.read_u64(cursor) else { return 0 };
    if token == 0 {
        return 0;
    }
    let Some(text) = ctx.read_cstr_bytes(token, MAX_UNITS) else { return 0 };
    let delimiters = ctx.read_cstr_bytes(delimiters, MAX_UNITS).unwrap_or_default();
    match text.iter().position(|b| delimiters.contains(b)) {
        Some(i) => {
            let end = token + i as u64;
            ctx.write_bytes(end, &[0]);
            ctx.write_u64(cursor, end + 1);
        }
        None => {
            ctx.write_u64(cursor, 0);
        }
    }
    token as HleResult
}

/// memset_s(dest, destsz, c, n) -> 0 or EINVAL, as FreeBSD's: EINVAL for
/// a NULL destination or a size above RSIZE_MAX (nothing written), and
/// for n > destsz (destsz bytes written first).
fn cb_memset_s(ctx: &mut CpuContext) -> HleResult {
    const RSIZE_MAX: u64 = u64::MAX >> 1;
    let (dest, destsz, value, count) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u8, ctx.rcx());
    if dest == 0 || destsz > RSIZE_MAX || count > RSIZE_MAX {
        return EINVAL;
    }
    let length = count.min(destsz);
    crate::memops::fill("memset_s", dest, value, length, crate::memops::caller(ctx));
    if count > destsz { EINVAL } else { 0 }
}

/// Guest address of the `__progname` data symbol (a `const char*`), set by
/// the loader when it places the HLE data symbols.
static PROGNAME_SYMBOL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn set_progname_symbol(addr: u64) {
    PROGNAME_SYMBOL.store(addr, std::sync::atomic::Ordering::Relaxed);
}

/// getprogname() -> the current value of `__progname` (a title that
/// assigns `__progname` changes what it returns).
fn cb_getprogname(ctx: &mut CpuContext) -> HleResult {
    match PROGNAME_SYMBOL.load(std::sync::atomic::Ordering::Relaxed) {
        0 => 0,
        symbol => ctx.read_u64(symbol).unwrap_or(0) as HleResult,
    }
}

/// Size of the per-thread dirname buffer.
const DIRNAME_BUFFER: usize = 1024;
const ENAMETOOLONG: i32 = 63;

thread_local! {
    /// Guest buffer dirname returns (one per thread).
    static DIRNAME_RESULT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The directory part of a path, as FreeBSD's dirname computes it:
/// trailing slashes go, then the last component, then the slashes before
/// it; "." when nothing is left of a relative path, "/" for the root.
fn dirname_of(path: &[u8]) -> Vec<u8> {
    let mut len = path.len();
    while len > 0 && path[len - 1] == b'/' {
        len -= 1;
    }
    if len == 0 {
        return if path.is_empty() { b".".to_vec() } else { b"/".to_vec() };
    }
    let mut sep = len;
    while sep > 0 && path[sep - 1] != b'/' {
        sep -= 1;
    }
    if sep == 0 {
        return b".".to_vec();
    }
    while sep > 1 && path[sep - 1] == b'/' {
        sep -= 1;
    }
    path[..sep].to_vec()
}

/// dirname(path) -> the per-thread buffer; the input is not modified.
/// NULL with ENAMETOOLONG for a path of 1024 bytes or more.
fn cb_dirname(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    let path = if addr == 0 { Vec::new() } else { ctx.read_cstr_bytes(addr, MAX_UNITS).unwrap_or_default() };
    if path.len() >= DIRNAME_BUFFER {
        crate::kernel::set_errno(ctx, ENAMETOOLONG);
        return 0;
    }
    let mut out = DIRNAME_RESULT.with(|c| c.get());
    if out == 0 {
        let Some(p) = crate::pthread::alloc_guest_block(ctx, DIRNAME_BUFFER as u64) else { return 0 };
        DIRNAME_RESULT.with(|c| c.set(p));
        out = p;
    }
    let mut dir = dirname_of(&path);
    dir.push(0);
    if ctx.write_bytes(out, &dir) { out as HleResult } else { 0 }
}

/// strcasecmp(a, b): ASCII case-insensitive compare of the lowered bytes.
fn cb_strcasecmp(ctx: &mut CpuContext) -> HleResult {
    let a = ctx.read_cstr_bytes(ctx.rdi(), MAX_UNITS).unwrap_or_default();
    let b = ctx.read_cstr_bytes(ctx.rsi(), MAX_UNITS).unwrap_or_default();
    let (a, b) = (a.iter().map(u8::to_ascii_lowercase), b.iter().map(u8::to_ascii_lowercase));
    let mut a = a.chain(std::iter::once(0));
    let mut b = b.chain(std::iter::once(0));
    loop {
        let (x, y) = (a.next().unwrap_or(0), b.next().unwrap_or(0));
        if x != y || x == 0 {
            return x as HleResult - y as HleResult;
        }
    }
}

/// strcoll(a, b): the "C" locale collates by byte value; the result is the
/// difference of the first differing unsigned bytes, as the games'
/// libc.prx returns it.
fn cb_strcoll(ctx: &mut CpuContext) -> HleResult {
    let a = ctx.read_cstr_bytes(ctx.rdi(), MAX_UNITS).unwrap_or_default();
    let b = ctx.read_cstr_bytes(ctx.rsi(), MAX_UNITS).unwrap_or_default();
    let mut a = a.into_iter().chain(std::iter::once(0));
    let mut b = b.into_iter().chain(std::iter::once(0));
    loop {
        let (x, y) = (a.next().unwrap_or(0), b.next().unwrap_or(0));
        if x != y || x == 0 {
            return x as HleResult - y as HleResult;
        }
    }
}

/// strncat(dst, src, n): appends at most n bytes of src and a terminator.
fn cb_strncat(ctx: &mut CpuContext) -> HleResult {
    let (dst, src, n) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let limit = usize::try_from(n).unwrap_or(usize::MAX).min(MAX_UNITS);
    let mut tail = ctx.read_cstr_bytes(src, limit).unwrap_or_default();
    let head = ctx.read_cstr_bytes(dst, MAX_UNITS).unwrap_or_default();
    tail.push(0);
    crate::memops::store("strncat", dst + head.len() as u64, &tail, crate::memops::caller(ctx));
    dst as HleResult
}

/// FreeBSD `sys_errlist`: the guest's errno numbering and texts (0..=96).
const ERRLIST: [&str; 97] = [
    "No error: 0", "Operation not permitted", "No such file or directory",
    "No such process", "Interrupted system call", "Input/output error",
    "Device not configured", "Argument list too long", "Exec format error",
    "Bad file descriptor", "No child processes", "Resource deadlock avoided",
    "Cannot allocate memory", "Permission denied", "Bad address",
    "Block device required", "Device busy", "File exists",
    "Cross-device link", "Operation not supported by device", "Not a directory",
    "Is a directory", "Invalid argument", "Too many open files in system",
    "Too many open files", "Inappropriate ioctl for device", "Text file busy",
    "File too large", "No space left on device", "Illegal seek",
    "Read-only file system", "Too many links", "Broken pipe",
    "Numerical argument out of domain", "Result too large", "Resource temporarily unavailable",
    "Operation now in progress", "Operation already in progress", "Socket operation on non-socket",
    "Destination address required", "Message too long", "Protocol wrong type for socket",
    "Protocol not available", "Protocol not supported", "Socket type not supported",
    "Operation not supported", "Protocol family not supported", "Address family not supported by protocol family",
    "Address already in use", "Can't assign requested address", "Network is down",
    "Network is unreachable", "Network dropped connection on reset", "Software caused connection abort",
    "Connection reset by peer", "No buffer space available", "Socket is already connected",
    "Socket is not connected", "Can't send after socket shutdown", "Too many references: can't splice",
    "Operation timed out", "Connection refused", "Too many levels of symbolic links",
    "File name too long", "Host is down", "No route to host",
    "Directory not empty", "Too many processes", "Too many users",
    "Disc quota exceeded", "Stale NFS file handle", "Too many levels of remote in path",
    "RPC struct is bad", "RPC version wrong", "RPC prog. not avail",
    "Program version wrong", "Bad procedure for program", "No locks available",
    "Function not implemented", "Inappropriate file type or format", "Authentication error",
    "Need authenticator", "Identifier removed", "No message of desired type",
    "Value too large to be stored in data type", "Operation canceled", "Illegal byte sequence",
    "Attribute not found", "Programming error", "Bad message",
    "Multihop attempted", "Link has been severed", "Protocol error",
    "Capabilities insufficient", "Not permitted in capability mode", "State not recoverable",
    "Previous owner died",
];

/// The errlist entry of `errno`, None outside 0..=96.
fn known_errno_text(errno: i32) -> Option<&'static str> {
    usize::try_from(errno).ok().and_then(|i| ERRLIST.get(i).copied())
}

/// Text of an errno value: its errlist entry, else "Unknown error: <n>".
pub(crate) fn errno_text(errno: i32) -> String {
    match known_errno_text(errno) {
        Some(t) => t.to_string(),
        None => format!("Unknown error: {errno}"),
    }
}

/// Size of the per-thread strerror buffer.
const STRERROR_BUFFER: usize = 128;

thread_local! {
    /// Guest buffer strerror returns (one per thread).
    static STRERROR_RESULT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// strerror(errnum) -> the per-thread text buffer. errno becomes EINVAL for
/// an unknown number and is kept otherwise.
fn cb_strerror(ctx: &mut CpuContext) -> HleResult {
    let errnum = ctx.rdi() as i32;
    let mut text = errno_text(errnum).into_bytes();
    text.truncate(STRERROR_BUFFER - 1);
    text.push(0);
    let mut out = STRERROR_RESULT.with(|c| c.get());
    if out == 0 {
        let Some(p) = crate::pthread::alloc_guest_block(ctx, STRERROR_BUFFER as u64) else { return 0 };
        STRERROR_RESULT.with(|c| c.set(p));
        out = p;
    }
    if known_errno_text(errnum).is_none() {
        crate::kernel::set_errno(ctx, EINVAL as i32);
    }
    if ctx.write_bytes(out, &text) { out as HleResult } else { 0 }
}

/// strerror_r(errnum, buf, len): 0, EINVAL for an unknown number (also
/// when its text does not fit), ERANGE when the text of a known number and
/// its terminator do not fit (the truncated text is still stored). errno
/// is not changed.
fn cb_strerror_r(ctx: &mut CpuContext) -> HleResult {
    let (errnum, buf, len) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    let text = errno_text(errnum);
    let required = text.len() as u64 + 1;
    let known = known_errno_text(errnum).is_some();
    let mut result = if known { 0 } else { EINVAL };
    if known && (buf == 0 || len < required) {
        result = ERANGE;
    }
    if buf != 0 && len != 0 {
        let count = if required <= len { required - 1 } else { len - 1 } as usize;
        let mut bytes = text.as_bytes()[..count].to_vec();
        bytes.push(0);
        ctx.write_bytes(buf, &bytes);
    }
    result
}

/// perror(s): "s: <errno text>" (or just the text for NULL / "") on stderr.
fn cb_perror(ctx: &mut CpuContext) -> HleResult {
    let errno_addr = crate::kernel::errno_address();
    let errno = if errno_addr == 0 { 0 } else { ctx.read_u32(errno_addr).unwrap_or(0) as i32 };
    let prefix = ctx.read_cstr_bytes(ctx.rdi(), MAX_PERROR_TEXT).unwrap_or_default();
    let text = errno_text(errno);
    if prefix.is_empty() {
        log::info!("[stderr] {text}");
    } else {
        log::info!("[stderr] {}: {text}", String::from_utf8_lossy(&prefix));
    }
    0
}

/// stpcpy(dst, src) -> the terminator of the copy in dst.
fn cb_stpcpy(ctx: &mut CpuContext) -> HleResult {
    let (dst, src) = (ctx.rdi(), ctx.rsi());
    let Some(mut bytes) = ctx.read_cstr_bytes(src, MAX_UNITS) else {
        crate::memops::note_failure("stpcpy", src, 1, false, crate::memops::caller(ctx));
        return dst as HleResult;
    };
    let length = bytes.len() as u64;
    bytes.push(0);
    crate::memops::store("stpcpy", dst, &bytes, crate::memops::caller(ctx));
    (dst + length) as HleResult
}

// ---------------------------------------------------------------------------
// IPv4 text conversion (FreeBSD lib/libc/inet/inet_addr.c)
// ---------------------------------------------------------------------------

/// FreeBSD `inet_aton`: one to four parts in decimal, octal (leading 0) or
/// hex (0x), separated by dots, the last part filling the remaining bytes;
/// the text ends at a NUL or white space. Parts accumulate in 64 bits and
/// a one-part address keeps the low 32 of them. Returns the address in host
/// order (192.0.2.1 is 0xC0000201).
pub(crate) fn inet_aton_freebsd(text: &[u8]) -> Option<u32> {
    let at = |i: usize| text.get(i).copied().unwrap_or(0);
    let mut parts = [0u8; 3];
    let mut count = 0usize;
    let mut value: u64;
    let mut digit: bool;
    let mut i = 0usize;
    let mut c = at(i);
    loop {
        if !c.is_ascii_digit() {
            return None;
        }
        value = 0;
        let mut base = 10u64;
        digit = false;
        if c == b'0' {
            i += 1;
            c = at(i);
            if c == b'x' || c == b'X' {
                base = 16;
                i += 1;
                c = at(i);
            } else {
                base = 8;
                digit = true;
            }
        }
        loop {
            let lower = c | 0x20;
            if c.is_ascii_digit() {
                if base == 8 && c >= b'8' {
                    return None;
                }
                value = value.wrapping_mul(base).wrapping_add((c - b'0') as u64);
            } else if base == 16 && (b'a'..=b'f').contains(&lower) {
                value = (value << 4) | (lower - b'a' + 10) as u64;
            } else {
                break;
            }
            digit = true;
            i += 1;
            c = at(i);
        }
        if c != b'.' {
            break;
        }
        if count == 3 || value > 0xFF {
            return None;
        }
        parts[count] = value as u8;
        count += 1;
        i += 1;
        c = at(i);
    }
    if c != 0 && c != b' ' && !(b'\t'..=b'\r').contains(&c) {
        return None;
    }
    if !digit {
        return None;
    }
    const LIMITS: [u64; 4] = [u64::MAX, 0xFF_FFFF, 0xFFFF, 0xFF];
    if value > LIMITS[count] {
        return None;
    }
    for (index, &part) in parts[..count].iter().enumerate() {
        value |= (part as u64) << (24 - 8 * index);
    }
    Some(value as u32)
}

/// __inet_aton(text, struct in_addr*) -> 1 valid / 0; a null address only
/// validates.
fn cb_inet_aton(ctx: &mut CpuContext) -> HleResult {
    let (text, out) = (ctx.rdi(), ctx.rsi());
    let Some(address) = ctx.read_cstr_bytes(text, MAX_INET_TEXT).and_then(|t| inet_aton_freebsd(&t)) else {
        return 0;
    };
    if out != 0 {
        ctx.write_bytes(out, &address.to_be_bytes());
    }
    1
}

/// __inet_addr(text) -> in_addr_t in network byte order, INADDR_NONE
/// (0xFFFFFFFF) when the text is not an address.
fn cb_inet_addr(ctx: &mut CpuContext) -> HleResult {
    match ctx.read_cstr_bytes(ctx.rdi(), MAX_INET_TEXT).and_then(|t| inet_aton_freebsd(&t)) {
        Some(address) => u32::from_le_bytes(address.to_be_bytes()) as HleResult,
        None => 0xFFFF_FFFF,
    }
}

/// Longest address text read from the guest.
const MAX_INET_TEXT: usize = 4096;

thread_local! {
    /// The per-thread buffer __inet_ntoa returns (16 bytes).
    static INET_NTOA_BUFFER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Dotted decimal of the network-order bytes of `address`.
fn dotted(address: u32) -> String {
    let b = address.to_le_bytes();
    format!("{}.{}.{}.{}", b[0], b[1], b[2], b[3])
}

/// __inet_ntoa(struct in_addr) -> the dotted text in a per-thread buffer,
/// overwritten by the thread's next call.
fn cb_inet_ntoa(ctx: &mut CpuContext) -> HleResult {
    let mut buffer = INET_NTOA_BUFFER.with(|c| c.get());
    if buffer == 0 {
        let Some(p) = crate::pthread::alloc_guest_block(ctx, 16) else { return 0 };
        INET_NTOA_BUFFER.with(|c| c.set(p));
        buffer = p;
    }
    let mut text = dotted(ctx.rdi() as u32).into_bytes();
    text.push(0);
    ctx.write_bytes(buffer, &text);
    buffer as HleResult
}

// ---------------------------------------------------------------------------
// syslog (FreeBSD vsyslog: %m, unknown bits, LOG_USER default)
// ---------------------------------------------------------------------------

const LOG_PRIMASK: i32 = 0x07;
const LOG_FACMASK: i32 = 0x3F8;
const LOG_USER: i32 = 1 << 3;
/// The priority vsyslog reports its own complaint with:
/// LOG_ERR | LOG_CONS | LOG_PERROR | LOG_PID.
const INTERNAL_LOG_PRIORITY: i32 = 3 | 0x02 | 0x20 | 0x01;

/// The format with each %m replaced by the text of `errno` (its own '%'
/// doubled) and %% kept for the formatter.
fn expand_error_text(format: &[u8], errno: i32) -> Vec<u8> {
    let mut out = Vec::with_capacity(format.len());
    let mut i = 0;
    while i < format.len() {
        match (format[i], format.get(i + 1)) {
            (b'%', Some(b'm')) => {
                for &b in errno_text(errno).as_bytes() {
                    out.push(b);
                    if b == b'%' {
                        out.push(b'%');
                    }
                }
                i += 2;
            }
            (b'%', Some(b'%')) => {
                out.extend_from_slice(b"%%");
                i += 2;
            }
            (b, _) => {
                out.push(b);
                i += 1;
            }
        }
    }
    out
}

/// The (priority, message) lines a vsyslog call emits: a complaint about
/// bits outside the priority and facility masks first, then the message,
/// LOG_USER when no facility is given. `errno` is the caller's, for %m.
fn syslog_lines(
    ctx: &CpuContext,
    mut priority: i32,
    format: Option<&[u8]>,
    args: &mut dyn crate::printf::ArgSource,
    errno: i32,
) -> Vec<(i32, Vec<u8>)> {
    let mut lines = Vec::with_capacity(2);
    if priority & !(LOG_PRIMASK | LOG_FACMASK) != 0 {
        lines.push((INTERNAL_LOG_PRIORITY, format!("syslog: unknown facility/priority: {priority:x}").into_bytes()));
        priority &= LOG_PRIMASK | LOG_FACMASK;
    }
    if priority & LOG_FACMASK == 0 {
        priority |= LOG_USER;
    }
    let message = match format {
        Some(f) => crate::printf::format_bytes(ctx, &expand_error_text(f, errno), args),
        None => Vec::new(),
    };
    lines.push((priority, message));
    lines
}

fn emit_syslog(ctx: &CpuContext, priority: i32, format_addr: u64, args: &mut dyn crate::printf::ArgSource) {
    let errno_addr = crate::kernel::errno_address();
    let errno = if errno_addr == 0 { 0 } else { ctx.read_u32(errno_addr).unwrap_or(0) as i32 };
    let format = (format_addr != 0).then(|| ctx.read_cstr_bytes(format_addr, MAX_PERROR_TEXT * 8)).flatten();
    for (p, message) in syslog_lines(ctx, priority, format.as_deref(), args, errno) {
        let visible = message.split(|&b| b == 0).next().unwrap_or_default();
        log::info!("[syslog:{p}] {}", String::from_utf8_lossy(visible).trim_end_matches('\n'));
    }
}

/// syslog(priority, format, ...)
fn cb_syslog(ctx: &mut CpuContext) -> HleResult {
    let (priority, format) = (ctx.rdi() as i32, ctx.rsi());
    let mut args = crate::printf::RegArgsCursor::new(ctx, 2);
    emit_syslog(ctx, priority, format, &mut args);
    0
}

/// vsyslog(priority, format, va_list)
fn cb_vsyslog(ctx: &mut CpuContext) -> HleResult {
    let (priority, format, va_list) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    match crate::printf::VaListCursor::new(ctx, va_list) {
        Some(mut args) => emit_syslog(ctx, priority, format, &mut args),
        None => log::warn!("[libc] vsyslog with an unreadable va_list {va_list:#x}"),
    }
    0
}

// ---------------------------------------------------------------------------
// C locale character classes of wide characters (FreeBSD rune locale)
// ---------------------------------------------------------------------------

const CTYPE_A: u64 = 0x100; // alpha
const CTYPE_C: u64 = 0x200; // control
const CTYPE_D: u64 = 0x400; // digit
const CTYPE_G: u64 = 0x800; // graph
const CTYPE_L: u64 = 0x1000; // lower
const CTYPE_P: u64 = 0x2000; // punct
const CTYPE_S: u64 = 0x4000; // space
const CTYPE_U: u64 = 0x8000; // upper
const CTYPE_X: u64 = 0x10000; // hex digit
const CTYPE_B: u64 = 0x20000; // blank
const CTYPE_R: u64 = 0x40000; // print
const CTYPE_I: u64 = 0x80000; // ideogram
const CTYPE_T: u64 = 0x10_0000; // special
const CTYPE_Q: u64 = 0x20_0000; // phonogram
const CTYPE_N: u64 = 0x40_0000; // number
const WEOF: u32 = 0xFFFF_FFFF;

/// The class bits of `c` in the C locale: ASCII only; every value from 128
/// up (and WEOF) has none.
fn c_locale_class(c: u32) -> u64 {
    let Ok(b) = u8::try_from(c) else { return 0 };
    if !b.is_ascii() {
        return 0;
    }
    let mut t = 0;
    if b.is_ascii_control() {
        t |= CTYPE_C;
    }
    if matches!(b, b'\t'..=b'\r' | b' ') {
        t |= CTYPE_S;
    }
    if b == b'\t' || b == b' ' {
        t |= CTYPE_B;
    }
    if (0x20..0x7F).contains(&b) {
        t |= CTYPE_R;
    }
    if b.is_ascii_graphic() {
        t |= CTYPE_G;
        if !b.is_ascii_alphanumeric() {
            t |= CTYPE_P;
        }
    }
    if b.is_ascii_digit() {
        t |= CTYPE_D | CTYPE_N;
    }
    if b.is_ascii_hexdigit() {
        t |= CTYPE_X;
    }
    if b.is_ascii_uppercase() {
        t |= CTYPE_A | CTYPE_U;
    }
    if b.is_ascii_lowercase() {
        t |= CTYPE_A | CTYPE_L;
    }
    t
}

fn wide_class(ctx: &CpuContext, mask: u64) -> HleResult {
    (c_locale_class(ctx.rdi() as u32) & mask != 0) as HleResult
}

fn cb_iswalnum(ctx: &mut CpuContext) -> HleResult {
    wide_class(ctx, CTYPE_A | CTYPE_N)
}
fn cb_iswalpha(ctx: &mut CpuContext) -> HleResult {
    wide_class(ctx, CTYPE_A)
}
fn cb_iswblank(ctx: &mut CpuContext) -> HleResult {
    wide_class(ctx, CTYPE_B)
}
fn cb_iswcntrl(ctx: &mut CpuContext) -> HleResult {
    wide_class(ctx, CTYPE_C)
}
fn cb_iswdigit(ctx: &mut CpuContext) -> HleResult {
    wide_class(ctx, CTYPE_D)
}
fn cb_iswgraph(ctx: &mut CpuContext) -> HleResult {
    wide_class(ctx, CTYPE_G)
}
fn cb_iswlower(ctx: &mut CpuContext) -> HleResult {
    wide_class(ctx, CTYPE_L)
}
fn cb_iswprint(ctx: &mut CpuContext) -> HleResult {
    wide_class(ctx, CTYPE_R)
}
fn cb_iswpunct(ctx: &mut CpuContext) -> HleResult {
    wide_class(ctx, CTYPE_P)
}
/// iswascii(wint_t c): `c` is a 7-bit value, whatever the locale.
fn cb_iswascii(ctx: &mut CpuContext) -> HleResult {
    (ctx.rdi() as u32 & !0x7F == 0) as HleResult
}
fn cb_iswspace(ctx: &mut CpuContext) -> HleResult {
    wide_class(ctx, CTYPE_S)
}
fn cb_iswupper(ctx: &mut CpuContext) -> HleResult {
    wide_class(ctx, CTYPE_U)
}
fn cb_iswxdigit(ctx: &mut CpuContext) -> HleResult {
    wide_class(ctx, CTYPE_X)
}

/// iswctype(c, wctype_t): any of the mask's class bits.
fn cb_iswctype(ctx: &mut CpuContext) -> HleResult {
    wide_class(ctx, ctx.rsi())
}

/// The classes of the C++ runtime's `_Iswctype(c, class)`, numbered from 1
/// in the order of its isw* macros: alnum, alpha, cntrl, digit, graph,
/// lower, print, punct, space, upper, xdigit, blank.
const ISWCTYPE_CLASSES: [u64; 12] =
    [CTYPE_A | CTYPE_N, CTYPE_A, CTYPE_C, CTYPE_D, CTYPE_G, CTYPE_L, CTYPE_R, CTYPE_P, CTYPE_S, CTYPE_U, CTYPE_X, CTYPE_B];

/// _Iswctype(c, class): the class is an index (1..=12), not a mask.
pub(crate) fn cb_iswctype_class(ctx: &mut CpuContext) -> HleResult {
    let class = ctx.rsi();
    match class.checked_sub(1).and_then(|i| ISWCTYPE_CLASSES.get(i as usize)) {
        Some(&mask) => wide_class(ctx, mask),
        None => {
            log::warn!("[libc] _Iswctype: unknown class {class}");
            0
        }
    }
}

/// The wctype_t of a class name; 0 for an unknown one (names are exact).
fn wctype_mask(name: &[u8]) -> u64 {
    match name {
        b"alnum" => CTYPE_A | CTYPE_N,
        b"alpha" => CTYPE_A,
        b"blank" => CTYPE_B,
        b"cntrl" => CTYPE_C,
        b"digit" => CTYPE_D,
        b"graph" => CTYPE_G,
        b"lower" => CTYPE_L,
        b"print" => CTYPE_R,
        b"punct" => CTYPE_P,
        b"space" => CTYPE_S,
        b"upper" => CTYPE_U,
        b"xdigit" => CTYPE_X,
        b"ideogram" => CTYPE_I,
        b"special" => CTYPE_T,
        b"phonogram" => CTYPE_Q,
        b"number" => CTYPE_N,
        b"rune" => 0xFFFF_FF00,
        _ => 0,
    }
}

/// wctype(name) -> wctype_t
fn cb_wctype(ctx: &mut CpuContext) -> HleResult {
    ctx.read_cstr_bytes(ctx.rdi(), 64).map_or(0, |n| wctype_mask(&n)) as HleResult
}

/// towlower(c): ASCII in the C locale; anything else unchanged.
fn cb_towlower(ctx: &mut CpuContext) -> HleResult {
    let c = ctx.rdi() as u32;
    (if c < 0x80 { (c as u8).to_ascii_lowercase() as u32 } else { c }) as HleResult
}

/// towupper(c)
fn cb_towupper(ctx: &mut CpuContext) -> HleResult {
    let c = ctx.rdi() as u32;
    (if c < 0x80 { (c as u8).to_ascii_uppercase() as u32 } else { c }) as HleResult
}

/// btowc(int c): EOF is WEOF, any other value its byte.
fn cb_btowc(ctx: &mut CpuContext) -> HleResult {
    let c = ctx.rdi() as i32;
    (if c == -1 { WEOF } else { c as u8 as u32 }) as HleResult
}

/// wctob(c): a single-byte value, else EOF.
fn cb_wctob(ctx: &mut CpuContext) -> HleResult {
    let c = ctx.rdi() as u32;
    if c < 256 { c as HleResult } else { -1 }
}

// ---------------------------------------------------------------------------
// Wide strings
// ---------------------------------------------------------------------------

const EILSEQ: i32 = 86;

/// wcsrtombs(dst, const wchar_t** src, len, mbstate_t*) in the C locale:
/// every guest wide unit below 256 is one byte, anything else is EILSEQ.
/// With a destination, at most `len` bytes are stored, *src advances past
/// the converted units (NULL once the terminator was stored); without
/// one, the whole length is measured and *src is left alone.
fn cb_wcsrtombs(ctx: &mut CpuContext) -> HleResult {
    let (dst, src_ptr, len) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let Some(src) = ctx.read_u64(src_ptr) else { return -1 };
    let mut converted = 0u64;
    let mut out = Vec::new();
    let result: HleResult = loop {
        if dst != 0 && converted >= len {
            break converted as HleResult;
        }
        let Some(unit) = ctx.read_bytes(src + converted * 2, 2).map(|b| u16::from_le_bytes([b[0], b[1]])) else {
            break -1;
        };
        if unit >= 256 {
            crate::kernel::set_errno(ctx, EILSEQ);
            break -1;
        }
        if dst != 0 {
            out.push(unit as u8);
        }
        if unit == 0 {
            if dst != 0 {
                crate::memops::store("wcsrtombs", dst, &out, crate::memops::caller(ctx));
                ctx.write_u64(src_ptr, 0);
            }
            return converted as HleResult;
        }
        converted += 1;
    };
    if dst != 0 {
        crate::memops::store("wcsrtombs", dst, &out, crate::memops::caller(ctx));
        ctx.write_u64(src_ptr, src + converted * 2);
    }
    result
}

const EILSEQ_S: HleResult = 86;

/// wcsrtombs_s core: C-locale units below 256 are one byte each. `cell` is
/// the guest `const wchar_t**` updated as wcsrtombs_s does (0 for
/// wcstombs_s, whose source pointer is a copy). Returns the errno.
#[allow(clippy::too_many_arguments)]
fn wcsrtombs_s_core(ctx: &CpuContext, result: u64, dst: u64, capacity: u64, cell: u64, src: u64, limit: u64, state: u64) -> HleResult {
    const FAILED: u64 = u64::MAX;
    let set_result = |v: u64| {
        if result != 0 {
            ctx.write_u64(result, v);
        }
    };
    let set_source = |v: u64| {
        if cell != 0 {
            ctx.write_u64(cell, v);
        }
    };
    if result == 0 || state == 0 || src == 0 || (dst != 0 && ((capacity as i64) <= 0 || (limit as i64) < 0)) {
        if dst != 0 && (capacity as i64) > 0 {
            ctx.write_bytes(dst, &[0]);
        }
        set_result(FAILED);
        return EINVAL;
    }
    if dst == 0 && capacity != 0 {
        set_result(FAILED);
        return EINVAL;
    }
    if dst == 0 {
        for i in 0..MAX_UNITS as u64 {
            let Some(unit) = read_u16(ctx, src + i * 2) else { break };
            if unit >= 256 {
                set_result(FAILED);
                return EILSEQ_S;
            }
            if unit == 0 {
                set_result(i);
                return 0;
            }
        }
        set_result(FAILED);
        return EINVAL;
    }
    let (mut written, mut capacity, mut limit, mut wide) = (0u64, capacity, limit, src);
    while limit != 0 {
        let Some(unit) = read_u16(ctx, wide) else {
            set_result(FAILED);
            return EINVAL;
        };
        if unit >= 256 {
            ctx.write_bytes(dst + written, &[0]);
            set_result(FAILED);
            return EILSEQ_S;
        }
        ctx.write_bytes(dst + written, &[unit as u8]);
        written += 1;
        if unit == 0 {
            set_source(0);
            set_result(written - 1);
            return 0;
        }
        wide += 2;
        capacity -= 1;
        if capacity == 0 {
            set_source(wide);
            ctx.write_bytes(dst, &[0]);
            set_result(FAILED);
            return ERANGE;
        }
        limit -= 1;
    }
    set_source(wide);
    ctx.write_bytes(dst + written, &[0]);
    set_result(written);
    0
}

/// wcsrtombs_s(size_t* result, char* dst, size_t capacity,
/// const wchar_t** src, size_t limit, mbstate_t* state)
fn cb_wcsrtombs_s(ctx: &mut CpuContext) -> HleResult {
    let cell = ctx.rcx();
    // A null cell reads as a null *src: EINVAL.
    let src = if cell == 0 { 0 } else { ctx.read_u64(cell).unwrap_or(0) };
    wcsrtombs_s_core(ctx, ctx.rdi(), ctx.rsi(), ctx.rdx(), cell, src, ctx.r8(), ctx.r9())
}

/// wcstombs_s(size_t* result, char* dst, size_t capacity,
/// const wchar_t* src, size_t limit): wcsrtombs_s with a fresh state.
fn cb_wcstombs_s(ctx: &mut CpuContext) -> HleResult {
    wcsrtombs_s_core(ctx, ctx.rdi(), ctx.rsi(), ctx.rdx(), 0, ctx.rcx(), ctx.r8(), 1)
}

/// _WStoul(const wchar_t* s, wchar_t** end, int base): strtoul over the
/// ASCII prefix of the wide string (a unit of 128 or more ends the number;
/// only ASCII white space is skipped). The result is 64 bits.
fn cb_wstoul(ctx: &mut CpuContext) -> HleResult {
    let (s, end, base) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as i32);
    if s == 0 {
        return 0;
    }
    if base < 0 || base == 1 || base > 36 {
        log::warn!("[libc] _WStoul: unsupported base {base}");
        if end != 0 {
            ctx.write_u64(end, s);
        }
        crate::kernel::set_errno(ctx, EINVAL as i32);
        return 0;
    }
    let mut text = Vec::new();
    for i in 0..MAX_UNITS as u64 {
        match read_u16(ctx, s + i * 2) {
            Some(u) if u != 0 && u < 0x80 => text.push(u as u8),
            _ => break,
        }
    }
    let Ok(c) = std::ffi::CString::new(text) else { return 0 };
    let mut stop: *mut libc::c_char = std::ptr::null_mut();
    // SAFETY: the host thread's errno, and a NUL-terminated host copy.
    let (value, host_errno) = unsafe {
        *libc::__errno_location() = 0;
        let v = libc::strtoull(c.as_ptr(), &mut stop, base);
        (v, *libc::__errno_location())
    };
    if host_errno == libc::ERANGE {
        crate::kernel::set_errno(ctx, ERANGE as i32);
    }
    if end != 0 {
        let consumed = (stop as usize).wrapping_sub(c.as_ptr() as usize) as u64;
        ctx.write_u64(end, s + consumed * 2);
    }
    value as HleResult
}

/// The units of a wide string, at most `max` of them, and whether its
/// terminator was found within that bound.
fn read_wide(ctx: &CpuContext, addr: u64, max: usize) -> Option<(Vec<u16>, bool)> {
    let units = ctx.read_wide_string(addr, max)?;
    let terminated = units.len() < max;
    Some((units, terminated))
}

/// Write `units` followed by a wide terminator.
fn write_wide_terminated(ctx: &CpuContext, dst: u64, units: &[u16]) -> bool {
    let mut bytes: Vec<u8> = units.iter().flat_map(|u| u.to_le_bytes()).collect();
    bytes.extend_from_slice(&[0, 0]);
    crate::memops::store("wcscpy", dst, &bytes, crate::memops::caller(ctx)) == bytes.len() as u64
}

/// Compare at most `limit` units of two wide strings, units unsigned:
/// -1, 0 or 1.
fn compare_wide(ctx: &CpuContext, a: u64, b: u64, limit: u64) -> HleResult {
    if a == 0 || b == 0 {
        return 0;
    }
    let max = limit.min(MAX_UNITS as u64);
    for i in 0..max {
        let (Some(x), Some(y)) = (read_u16(ctx, a + i * 2), read_u16(ctx, b + i * 2)) else { return 0 };
        if x != y {
            return if x < y { -1 } else { 1 };
        }
        if x == 0 {
            return 0;
        }
    }
    0
}

fn read_u16(ctx: &CpuContext, addr: u64) -> Option<u16> {
    ctx.read_bytes(addr, 2).map(|b| u16::from_le_bytes([b[0], b[1]]))
}

/// wcslen(s) under its alternate export.
fn cb_wcsmisc(ctx: &mut CpuContext) -> HleResult {
    ctx.read_wide_string(ctx.rdi(), MAX_UNITS).map_or(0, |u| u.len() as HleResult)
}

/// wcscmp(a, b)
fn cb_wcscmp(ctx: &mut CpuContext) -> HleResult {
    compare_wide(ctx, ctx.rdi(), ctx.rsi(), u64::MAX)
}

/// wcscoll(a, b): the "C" locale collates by code unit.
fn cb_wcscoll(ctx: &mut CpuContext) -> HleResult {
    compare_wide(ctx, ctx.rdi(), ctx.rsi(), u64::MAX)
}

/// wcsncmp(a, b, n)
fn cb_wcsncmp(ctx: &mut CpuContext) -> HleResult {
    compare_wide(ctx, ctx.rdi(), ctx.rsi(), ctx.rdx())
}

/// wcscpy(dst, src) -> dst. A NULL source leaves an empty string.
fn cb_wcscpy(ctx: &mut CpuContext) -> HleResult {
    let (dst, src) = (ctx.rdi(), ctx.rsi());
    let units = ctx.read_wide_string(src, MAX_UNITS).unwrap_or_default();
    write_wide_terminated(ctx, dst, &units);
    dst as HleResult
}

/// wcsncpy(dst, src, n) -> dst: copies up to n units and pads the rest of
/// the n with terminators; no terminator when src has n or more units.
fn cb_wcsncpy(ctx: &mut CpuContext) -> HleResult {
    let (dst, src, n) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if n == 0 {
        return dst as HleResult;
    }
    let count = n.min(MAX_UNITS as u64) as usize;
    let units = ctx.read_wide_string(src, count).unwrap_or_default();
    let mut bytes = vec![0u8; count * 2];
    for (i, u) in units.iter().enumerate() {
        bytes[i * 2..i * 2 + 2].copy_from_slice(&u.to_le_bytes());
    }
    crate::memops::store("wcsncpy", dst, &bytes, crate::memops::caller(ctx));
    dst as HleResult
}

/// Empty the destination of a failed `_s` copy.
fn clear_wide(ctx: &CpuContext, dst: u64, size: u64) {
    if dst != 0 && size != 0 {
        ctx.write_bytes(dst, &[0, 0]);
    }
}

/// wcscpy_s(dst, dstsz, src) -> errno_t
fn cb_wcscpy_s(ctx: &mut CpuContext) -> HleResult {
    let (dst, size, src) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if dst == 0 || size == 0 {
        return EINVAL;
    }
    if src == 0 {
        clear_wide(ctx, dst, size);
        return EINVAL;
    }
    let Some((units, terminated)) = read_wide(ctx, src, size.min(MAX_UNITS as u64) as usize) else {
        clear_wide(ctx, dst, size);
        return EINVAL;
    };
    if !terminated || units.len() as u64 + 1 > size {
        clear_wide(ctx, dst, size);
        return ERANGE;
    }
    write_wide_terminated(ctx, dst, &units);
    0
}

/// wcsncpy_s(dst, dstsz, src, n) -> errno_t. n == SIZE_MAX (_TRUNCATE)
/// copies what fits and returns STRUNCATE when the source was cut.
fn cb_wcsncpy_s(ctx: &mut CpuContext) -> HleResult {
    let (dst, size, src, n) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    if dst == 0 || size == 0 {
        return EINVAL;
    }
    if src == 0 {
        clear_wide(ctx, dst, size);
        return EINVAL;
    }
    if n == 0 {
        write_wide_terminated(ctx, dst, &[]);
        return 0;
    }
    if n == u64::MAX {
        let limit = (size - 1).min(MAX_UNITS as u64) as usize;
        let Some((units, terminated)) = read_wide(ctx, src, limit) else {
            clear_wide(ctx, dst, size);
            return EINVAL;
        };
        write_wide_terminated(ctx, dst, &units);
        // A source exactly `limit` long still fits.
        let fits = terminated || read_u16(ctx, src + units.len() as u64 * 2) == Some(0);
        return if fits { 0 } else { STRUNCATE };
    }
    let count = n.min(MAX_UNITS as u64);
    let Some((units, terminated)) = read_wide(ctx, src, count as usize) else {
        clear_wide(ctx, dst, size);
        return EINVAL;
    };
    let required = if terminated { units.len() as u64 + 1 } else { count + 1 };
    if required > size {
        clear_wide(ctx, dst, size);
        return ERANGE;
    }
    write_wide_terminated(ctx, dst, &units);
    0
}

/// wcschr(s, c) -> first unit equal to c (the terminator included), or NULL.
fn cb_wcschr(ctx: &mut CpuContext) -> HleResult {
    let (addr, needle) = (ctx.rdi(), ctx.rsi() as u16);
    for i in 0..MAX_UNITS as u64 {
        let Some(u) = read_u16(ctx, addr + i * 2) else { return 0 };
        if u == needle {
            return (addr + i * 2) as HleResult;
        }
        if u == 0 {
            return 0;
        }
    }
    0
}

const FALLBACK_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("8u8lPzUEq+U", "memchr", cb_memchr),
    ("5G2ONUzRgjY", "memrchr", cb_memrchr),
    ("cJWGxiQPmDQ", "strsep", cb_strsep),
    ("E4wZaG1zSFc", "dirname", cb_dirname),
    ("h8GwqPFbu6I", "memset_s", cb_memset_s),
    ("7Psx1DlAyE4", "getprogname", cb_getprogname),
    ("AV6ipCNa4Rw", "strcasecmp", cb_strcasecmp),
    ("gjbmYpP-XJQ", "strcoll", cb_strcoll),
    ("kHg45qPC6f0", "strncat", cb_strncat),
    ("EMutwaQ34Jo", "perror", cb_perror),
    ("RIa6GnWp+iU", "strerror", cb_strerror),
    ("RBcs3uut1TA", "strerror_r", cb_strerror_r),
    ("LHMrG7e8G78", "wcsmisc", cb_wcsmisc),
    ("pNtJdE3x49E", "wcscmp", cb_wcscmp),
    ("fV2xHER+bKE", "wcscoll", cb_wcscoll),
    ("E8wCoUEbfzk", "wcsncmp", cb_wcsncmp),
    ("FM5NPnLqBc8", "wcscpy", cb_wcscpy),
    ("0nV21JjYCH8", "wcsncpy", cb_wcsncpy),
    ("6f5f-qx4ucA", "wcscpy_s", cb_wcscpy_s),
    ("Slmz4HMpNGs", "wcsncpy_s", cb_wcsncpy_s),
    ("Ezzq78ZgHPs", "wcschr", cb_wcschr),
    ("sOOMlZoy1pg", "wcsrtombs", cb_wcsrtombs),
    ("79s2tnYQI6I", "wcsrtombs_s", cb_wcsrtombs_s),
    ("sZLrjx-yEx4", "wcstombs_s", cb_wcstombs_s),
    ("JhVR7D4Ax6Y", "_WStoul", cb_wstoul),
    ("CyXs2l-1kNA", "_Iswctype", cb_iswctype_class),
    ("ZSnX-xZBGCg", "stpcpy", cb_stpcpy),
    ("a7ToDPsIQrc", "__inet_aton", cb_inet_aton),
    ("ISTLytNGT0c", "inet_aton", cb_inet_aton),
    ("7p7kTAJcuGg", "__inet_addr", cb_inet_addr),
    ("sBBuXmJ5Kjk", "inet_addr", cb_inet_addr),
    ("6i5aLrxRhG0", "__inet_ntoa", cb_inet_ntoa),
    ("7iTp7O6VOXQ", "inet_ntoa", cb_inet_ntoa),
    ("UNS2V4S097M", "syslog", cb_syslog),
    ("zxecOOffO68", "vsyslog", cb_vsyslog),
    ("wDmL2EH0CBs", "iswalnum", cb_iswalnum),
    ("D-qDARDb1aM", "iswalpha", cb_iswalpha),
    ("p6DbM0OAHNo", "iswblank", cb_iswblank),
    ("6A+1YZ79qFk", "iswcntrl", cb_iswcntrl),
    ("n0kT+8Eeizs", "iswdigit", cb_iswdigit),
    ("wjG0GyCyaP0", "iswgraph", cb_iswgraph),
    ("Ok8KPy3nFls", "iswlower", cb_iswlower),
    ("U7IhU4VEB-0", "iswprint", cb_iswprint),
    ("AEPvEZkaLsU", "iswpunct", cb_iswpunct),
    ("SiZ8B5ja1uM", "iswascii", cb_iswascii),
    ("vqtytrxgLMs", "iswspace", cb_iswspace),
    ("1QcrrL9UDRQ", "iswupper", cb_iswupper),
    ("cjmSjRlnMAs", "iswxdigit", cb_iswxdigit),
    ("45E7omS0vvc", "iswctype", cb_iswctype),
    ("+3PtYiUxl-U", "wctype", cb_wctype),
    ("J3J1T9fjUik", "towlower", cb_towlower),
    ("1uf1SQsj5go", "towupper", cb_towupper),
    ("LEvm25Sxi7I", "btowc", cb_btowc),
    ("CL7VJxznu6g", "wctob", cb_wctob),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in FALLBACK_EXPORTS {
        reg.register_fallback_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::{Rcx, Rdi, Rdx, Rsi};

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in FALLBACK_EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    fn set(ctx: &mut CpuContext, v: &[u64]) {
        for (r, x) in [Rdi, Rsi, Rdx, Rcx].iter().zip(v) {
            ctx.set(*r, *x);
        }
    }

    /// strerror / strerror_r give the FreeBSD errlist texts; unknown
    /// numbers are "Unknown error: <n>" with EINVAL; a short buffer gets
    /// the truncated text and ERANGE; strerror_r never touches errno.
    #[test]
    fn strerror_uses_the_freebsd_errlist() {
        std::thread::spawn(|| {
            let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
            const BASE: u64 = 0x6A10_0000_0000;
            vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
            let mut ctx = CpuContext::new(vm);
            let errno = crate::kernel::bind_test_errno(&ctx);
            let text = |ctx: &CpuContext, addr: u64| String::from_utf8(ctx.read_cstr_bytes(addr, 256).unwrap()).unwrap();
            ctx.write_u32(errno, 13);
            set(&mut ctx, &[35]);
            let s = cb_strerror(&mut ctx) as u64;
            assert_eq!(text(&ctx, s), "Resource temporarily unavailable");
            assert_eq!(ctx.read_u32(errno), Some(13));
            let buf = BASE;
            for (n, expected) in ERRLIST.iter().enumerate() {
                set(&mut ctx, &[n as u64, buf, 128]);
                assert_eq!(cb_strerror_r(&mut ctx), 0);
                assert_eq!(text(&ctx, buf), *expected);
                set(&mut ctx, &[n as u64]);
                let s = cb_strerror(&mut ctx) as u64;
                assert_eq!(text(&ctx, s), *expected);
            }
            assert_eq!(ERRLIST[30], "Read-only file system");
            assert_eq!(ERRLIST[62], "Too many levels of symbolic links");
            set(&mut ctx, &[(-1i64) as u64, buf, 128]);
            assert_eq!(cb_strerror_r(&mut ctx), EINVAL);
            assert_eq!(text(&ctx, buf), "Unknown error: -1");
            assert_eq!(ctx.read_u32(errno), Some(13));
            for (n, expected) in [(97i32, "Unknown error: 97"), (i32::MAX, "Unknown error: 2147483647"), (i32::MIN, "Unknown error: -2147483648")] {
                set(&mut ctx, &[n as i64 as u64, buf, 128]);
                assert_eq!(cb_strerror_r(&mut ctx), EINVAL);
                assert_eq!(text(&ctx, buf), expected);
                assert_eq!(ctx.read_u32(errno), Some(13));
                set(&mut ctx, &[n as i64 as u64]);
                let s = cb_strerror(&mut ctx) as u64;
                assert_eq!(text(&ctx, s), expected);
                assert_eq!(ctx.read_u32(errno), Some(22));
                ctx.write_u32(errno, 13);
            }
            ctx.write_bytes(buf, b"xyz");
            set(&mut ctx, &[22, buf, 1]);
            assert_eq!(cb_strerror_r(&mut ctx), ERANGE);
            assert_eq!(ctx.read_bytes(buf, 2).unwrap(), b"\0y");
            set(&mut ctx, &[22, buf, 5]);
            assert_eq!(cb_strerror_r(&mut ctx), ERANGE);
            assert_eq!(text(&ctx, buf), "Inva");
            set(&mut ctx, &[22, 0, 0]);
            assert_eq!(cb_strerror_r(&mut ctx), ERANGE);
            assert_eq!(ctx.read_u32(errno), Some(13));
            // An unknown number keeps EINVAL with a short buffer or none.
            ctx.write_bytes(buf, b"xyzw");
            set(&mut ctx, &[1000, buf, 4]);
            assert_eq!(cb_strerror_r(&mut ctx), EINVAL);
            assert_eq!(text(&ctx, buf), "Unk");
            set(&mut ctx, &[1000, 0, 0]);
            assert_eq!(cb_strerror_r(&mut ctx), EINVAL);
        })
        .join()
        .unwrap();
    }

    /// memrchr finds the last match among the first n bytes; only the low
    /// byte of c counts (reference checks), and a search that ends at the
    /// start of a mapping never reads before it.
    #[test]
    fn memrchr_finds_the_last_match() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x7C40_0000_0000;
        vm.map(BASE, 0x3000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let bytes = BASE + 0x100;
        ctx.write_bytes(bytes, &[0xFF, 0x01, 0xFF, 0x00]);
        set(&mut ctx, &[bytes, 0xFF, 4]);
        assert_eq!(cb_memrchr(&mut ctx) as u64, bytes + 2);
        set(&mut ctx, &[bytes, 0x1FF, 4]);
        assert_eq!(cb_memrchr(&mut ctx) as u64, bytes + 2);
        set(&mut ctx, &[bytes, 0xFF, 2]);
        assert_eq!(cb_memrchr(&mut ctx) as u64, bytes);
        set(&mut ctx, &[bytes, 0x02, 4]);
        assert_eq!(cb_memrchr(&mut ctx), 0);
        set(&mut ctx, &[bytes, 0xFF, 0]);
        assert_eq!(cb_memrchr(&mut ctx), 0);
        // Across pages: the match is in the first page, the search starts
        // at the very beginning of the mapping.
        ctx.write_bytes(BASE, &[0x5A]);
        set(&mut ctx, &[BASE, 0x5A, 0x2800]);
        assert_eq!(cb_memrchr(&mut ctx) as u64, BASE);
        ctx.write_bytes(BASE + 0x27FF, &[0x5A]);
        assert_eq!(cb_memrchr(&mut ctx) as u64, BASE + 0x27FF);
        vm.unmap(BASE, 0x3000).unwrap();
    }

    /// strsep returns empty fields and leaves the cursor NULL after the
    /// last token (reference checks).
    #[test]
    fn strsep_splits_at_each_delimiter() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x7C41_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (cursor, delims, fields) = (BASE, BASE + 0x10, BASE + 0x100);
        let text = |ctx: &CpuContext, a: u64| ctx.read_cstr_bytes(a, 64).unwrap();
        ctx.write_bytes(delims, b",;\0");
        ctx.write_bytes(fields, b",a,,b;c\0");
        ctx.write_u64(cursor, fields);
        set(&mut ctx, &[cursor, delims]);
        assert_eq!(cb_strsep(&mut ctx) as u64, fields);
        assert_eq!(text(&ctx, fields), b"");
        assert_eq!(ctx.read_u64(cursor), Some(fields + 1));
        let mut tokens = Vec::new();
        for _ in 0..3 {
            let t = cb_strsep(&mut ctx) as u64;
            tokens.push(text(&ctx, t));
        }
        assert_eq!(tokens, [b"a".to_vec(), b"".to_vec(), b"b".to_vec()]);
        let last = ctx.read_u64(cursor).unwrap();
        assert_eq!(cb_strsep(&mut ctx) as u64, last);
        assert_eq!(text(&ctx, last), b"c");
        assert_eq!(ctx.read_u64(cursor), Some(0));
        assert_eq!(cb_strsep(&mut ctx), 0);
        assert_eq!(ctx.read_u64(cursor), Some(0));
        // No delimiters: the whole string, untouched.
        let whole = BASE + 0x200;
        ctx.write_bytes(whole, b"a,b\0");
        ctx.write_bytes(delims, b"\0");
        ctx.write_u64(cursor, whole);
        assert_eq!(cb_strsep(&mut ctx) as u64, whole);
        assert_eq!(text(&ctx, whole), b"a,b");
        assert_eq!(ctx.read_u64(cursor), Some(0));
        // An empty string is one empty token.
        let empty = BASE + 0x300;
        ctx.write_bytes(empty, b"\0");
        ctx.write_bytes(delims, b",\0");
        ctx.write_u64(cursor, empty);
        assert_eq!(cb_strsep(&mut ctx) as u64, empty);
        assert_eq!(ctx.read_u64(cursor), Some(0));
        vm.unmap(BASE, 0x1000).unwrap();
    }

    /// dirname: separators and boundaries (reference checks); the input
    /// path is left as it was; 1024 bytes or more is ENAMETOOLONG.
    #[test]
    fn dirname_separator_and_boundary_cases() {
        std::thread::spawn(|| {
            let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
            const BASE: u64 = 0x7C42_0000_0000;
            vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
            let mut ctx = CpuContext::new(vm);
            let errno = crate::kernel::bind_test_errno(&ctx);
            let input = BASE + 0x100;
            let dir = |ctx: &mut CpuContext, path: Option<&[u8]>| -> Vec<u8> {
                let arg = match path {
                    None => 0,
                    Some(p) => {
                        ctx.write_bytes(input, &[p, b"\0"].concat());
                        input
                    }
                };
                set(ctx, &[arg]);
                let r = cb_dirname(ctx) as u64;
                assert_ne!(r, 0);
                ctx.read_cstr_bytes(r, 2048).unwrap()
            };
            let cases: [(Option<&[u8]>, &[u8]); 12] = [
                (None, b"."),
                (Some(b""), b"."),
                (Some(b"file"), b"."),
                (Some(b"one/two///"), b"one"),
                (Some(b"one//two"), b"one"),
                (Some(b"/one//two"), b"/one"),
                (Some(b"/"), b"/"),
                (Some(b"foo/"), b"."),
                (Some(b"/one/two"), b"/one"),
                (Some(b"/file"), b"/"),
                (Some(b"////"), b"/"),
                (Some(b"//a"), b"/"),
            ];
            for (path, expected) in cases {
                assert_eq!(dir(&mut ctx, path), expected, "dirname({path:?})");
            }
            ctx.write_bytes(input, b"one/two///\0");
            set(&mut ctx, &[input]);
            cb_dirname(&mut ctx);
            assert_eq!(ctx.read_cstr_bytes(input, 64).unwrap(), b"one/two///", "the input changed");
            ctx.write_bytes(input, &[vec![b'x'; 1024], vec![0]].concat());
            ctx.write_u32(errno, 0);
            assert_eq!(cb_dirname(&mut ctx), 0);
            assert_eq!(ctx.read_u32(errno), Some(63));
            ctx.write_bytes(input, &[vec![b'x'; 1023], vec![0]].concat());
            assert_ne!(cb_dirname(&mut ctx), 0, "1023 bytes fit");
        })
        .join()
        .unwrap();
    }

    /// memset_s follows FreeBSD: EINVAL for every constraint violation,
    /// nothing written above RSIZE_MAX, destsz bytes written when n is
    /// larger (reference checks).
    #[test]
    fn memset_s_errors_follow_freebsd() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x7C43_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (bytes, sentinel) = (BASE + 0x100, BASE + 0x200);
        let rsize_max = u64::MAX >> 1;
        ctx.write_bytes(bytes, &[0; 8]);
        set(&mut ctx, &[bytes, 4, 9, 8]);
        assert_eq!(cb_memset_s(&mut ctx), EINVAL);
        assert_eq!(ctx.read_bytes(bytes, 5).unwrap(), [9, 9, 9, 9, 0]);
        ctx.write_bytes(sentinel, &[1, 2, 3, 4]);
        set(&mut ctx, &[sentinel, 4, 7, 2]);
        assert_eq!(cb_memset_s(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(sentinel, 4).unwrap(), [7, 7, 3, 4]);
        set(&mut ctx, &[sentinel, rsize_max + 1, 0, 1]);
        assert_eq!(cb_memset_s(&mut ctx), EINVAL);
        set(&mut ctx, &[sentinel, 4, 0, rsize_max + 1]);
        assert_eq!(cb_memset_s(&mut ctx), EINVAL);
        assert_eq!(ctx.read_bytes(sentinel, 4).unwrap(), [7, 7, 3, 4], "nothing written above RSIZE_MAX");
        set(&mut ctx, &[sentinel, rsize_max, 5, 1]);
        assert_eq!(cb_memset_s(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(sentinel, 2).unwrap(), [5, 7]);
        set(&mut ctx, &[0, 4, 0, 1]);
        assert_eq!(cb_memset_s(&mut ctx), EINVAL);
        vm.unmap(BASE, 0x1000).unwrap();
    }

    /// getprogname follows the `__progname` pointer, also after the title
    /// reassigns it (reference checks).
    #[test]
    fn getprogname_reads_the_progname_symbol() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x7C44_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (symbol, original, renamed, empty) = (BASE, BASE + 0x10, BASE + 0x20, BASE + 0x30);
        ctx.write_bytes(original, b"eboot.bin\0");
        ctx.write_bytes(renamed, b"Quake2\0");
        ctx.write_bytes(empty, b"\0");
        ctx.write_u64(symbol, original);
        set_progname_symbol(symbol);
        assert_eq!(cb_getprogname(&mut ctx) as u64, original);
        assert_eq!(ctx.read_cstr_bytes(original, 16).unwrap(), b"eboot.bin");
        ctx.write_u64(symbol, renamed);
        assert_eq!(cb_getprogname(&mut ctx) as u64, renamed);
        ctx.write_u64(symbol, empty);
        assert_eq!(cb_getprogname(&mut ctx) as u64, empty);
        ctx.write_u64(symbol, original);
        assert_eq!(cb_getprogname(&mut ctx) as u64, original);
        vm.unmap(BASE, 0x1000).unwrap();
    }

    fn wide(s: &str) -> Vec<u8> {
        s.encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect()
    }

    fn read_w(ctx: &CpuContext, addr: u64) -> String {
        String::from_utf16_lossy(&ctx.read_wide_string(addr, 256).unwrap())
    }

    #[test]
    fn narrow_helpers() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5DA0_0000_0000;
        vm.map(BASE, 0x2000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        // memchr stops at the mapping edge without faulting.
        let tail = BASE + 0x2000 - 4;
        ctx.write_bytes(tail, b"abcd");
        set(&mut ctx, &[tail, b'c' as u64, 4]);
        assert_eq!(cb_memchr(&mut ctx), (tail + 2) as HleResult);
        set(&mut ctx, &[tail, b'z' as u64, 4]);
        assert_eq!(cb_memchr(&mut ctx), 0);
        set(&mut ctx, &[tail, 0x163, 4]); // (unsigned char)0x163 == 'c'
        assert_eq!(cb_memchr(&mut ctx), (tail + 2) as HleResult);

        ctx.write_bytes(BASE, b"Hello\0");
        ctx.write_bytes(BASE + 0x40, b"hELLO\0");
        ctx.write_bytes(BASE + 0x80, b"help\0");
        set(&mut ctx, &[BASE, BASE + 0x40]);
        assert_eq!(cb_strcasecmp(&mut ctx), 0);
        set(&mut ctx, &[BASE, BASE + 0x80]);
        assert!(cb_strcasecmp(&mut ctx) < 0);
        set(&mut ctx, &[BASE + 0x80, BASE]);
        assert!(cb_strcasecmp(&mut ctx) > 0);

        // strcoll returns the byte difference.
        for (a, b, want) in [
            (&b"a\0"[..], &b"z\0"[..], -25),
            (b"z\0", b"a\0", 25),
            (b"ab\0", b"abc\0", -99),
            (b"A\0", b"a\0", -32),
            (b"same\0", b"same\0", 0),
            (b"\xff\0", b"\x01\0", 254),
        ] {
            ctx.write_bytes(BASE + 0x200, a);
            ctx.write_bytes(BASE + 0x240, b);
            set(&mut ctx, &[BASE + 0x200, BASE + 0x240]);
            assert_eq!(cb_strcoll(&mut ctx), want, "{a:?} {b:?}");
        }

        ctx.write_bytes(BASE + 0x100, b"ab\0\xAA\xAA\xAA\xAA");
        set(&mut ctx, &[BASE + 0x100, BASE, 3]);
        assert_eq!(cb_strncat(&mut ctx), (BASE + 0x100) as HleResult);
        assert_eq!(ctx.read_bytes(BASE + 0x100, 6).unwrap(), b"abHel\0");
    }

    #[test]
    fn stpcpy_returns_the_end_of_the_copy() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E63_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let chained = BASE + 0x100;
        ctx.write_bytes(chained, b"zzzzzzz\0");
        ctx.write_bytes(BASE, b"ab\0");
        ctx.write_bytes(BASE + 0x10, b"cd\0");
        ctx.write_bytes(BASE + 0x20, b"\0");
        set(&mut ctx, &[chained, BASE]);
        let end = cb_stpcpy(&mut ctx) as u64;
        assert_eq!(end, chained + 2);
        assert_eq!(ctx.read_bytes(chained, 4).unwrap(), b"ab\0z");
        set(&mut ctx, &[end, BASE + 0x10]);
        let end = cb_stpcpy(&mut ctx) as u64;
        assert_eq!(end, chained + 4);
        assert_eq!(ctx.read_cstr_bytes(chained, 16).unwrap(), b"abcd");
        set(&mut ctx, &[end, BASE + 0x20]);
        assert_eq!(cb_stpcpy(&mut ctx) as u64, end);
        assert_eq!(ctx.read_bytes(chained + 5, 1).unwrap(), b"z");
        vm.unmap(BASE, 0x1000).unwrap();
    }

    /// FreeBSD inet_aton / inet_addr / inet_ntoa.
    #[test]
    fn inet_text_conversions() {
        let dotted_of = |text: &str| inet_aton_freebsd(text.as_bytes()).map(|a| {
            let b = a.to_be_bytes();
            format!("{}.{}.{}.{}", b[0], b[1], b[2], b[3])
        });
        for (text, expected) in [
            ("192.0.2.42", "192.0.2.42"),
            ("10.1.2", "10.1.0.2"),
            ("127.1", "127.0.0.1"),
            ("3232235777", "192.168.1.1"),
            ("0x7f.0.0.0x1", "127.0.0.1"),
            ("0377.0.0.010", "255.0.0.8"),
            ("1.2.3.4 trailing", "1.2.3.4"),
            ("1.2.3.4\n", "1.2.3.4"),
            ("0.0.0.0", "0.0.0.0"),
            ("255.255.255.255", "255.255.255.255"),
            // A one-part address accumulates in 64 bits and keeps the low 32.
            ("4294967296", "0.0.0.0"),
            ("4294967297", "0.0.0.1"),
            ("18446744073709551617", "0.0.0.1"),
            ("0x.1", "0.0.0.1"),
            ("1.16777215", "1.255.255.255"),
            ("1.2.65535", "1.2.255.255"),
            ("0X7F.0x1", "127.0.0.1"),
            ("0177.01", "127.0.0.1"),
        ] {
            assert_eq!(dotted_of(text).as_deref(), Some(expected), "{text:?}");
        }
        for invalid in ["", " 1.2.3.4", "1.2.3.4.5", "256.1.1.1", "1.2.3.256", "1.2.65536", "08", "1..2", "a.b.c.d", "1.2.3.4x", "0x", "1.2.3.", "-1"] {
            assert_eq!(inet_aton_freebsd(invalid.as_bytes()), None, "{invalid:?}");
        }

        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E64_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (text, out) = (BASE, BASE + 0x100);
        ctx.write_bytes(text, b"192.0.2.42\0");
        ctx.write_bytes(out, &[0xA5; 4]);
        set(&mut ctx, &[text, out]);
        assert_eq!(cb_inet_aton(&mut ctx), 1);
        assert_eq!(ctx.read_bytes(out, 4).unwrap(), [192, 0, 2, 42]);
        set(&mut ctx, &[text, 0]);
        assert_eq!(cb_inet_aton(&mut ctx), 1, "a null address only validates");
        assert_eq!(cb_inet_addr(&mut ctx) as u32, u32::from_le_bytes([192, 0, 2, 42]));
        ctx.write_bytes(text, b"1.2.3.256\0");
        ctx.write_bytes(out, &[0xA5; 4]);
        set(&mut ctx, &[text, out]);
        assert_eq!(cb_inet_aton(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(out, 4).unwrap(), [0xA5; 4]);
        assert_eq!(cb_inet_addr(&mut ctx), 0xFFFF_FFFF);

        for (bytes, expected) in [
            ([127, 0, 0, 1], "127.0.0.1"),
            ([255, 255, 255, 255], "255.255.255.255"),
            ([0, 0, 0, 0], "0.0.0.0"),
            ([10, 200, 3, 45], "10.200.3.45"),
            ([100, 9, 99, 0], "100.9.99.0"),
        ] {
            set(&mut ctx, &[u32::from_le_bytes(bytes) as u64]);
            let p = cb_inet_ntoa(&mut ctx) as u64;
            assert_ne!(p, 0);
            assert_eq!(ctx.read_cstr_bytes(p, 16).unwrap(), expected.as_bytes());
            // The same per-thread buffer every time.
            assert_eq!(cb_inet_ntoa(&mut ctx) as u64, p);
        }
        vm.unmap(BASE, 0x1000).unwrap();
    }

    /// FreeBSD vsyslog: %m is the caller's errno text, unknown bits are
    /// reported first (with the internal priority 0x23) and masked, and a
    /// message without a facility goes to LOG_USER.
    #[test]
    fn syslog_lines_follow_freebsd_vsyslog() {
        struct Args(Vec<u64>);
        impl crate::printf::ArgSource for Args {
            fn next_gp(&mut self) -> u64 {
                if self.0.is_empty() { 0 } else { self.0.remove(0) }
            }
            fn next_fp(&mut self) -> f64 {
                0.0
            }
        }
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E65_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, b"radio\0");
        let line = |p: i32, s: &str| (p, s.as_bytes().to_vec());
        assert_eq!(
            syslog_lines(&ctx, 6, Some(b"station %s at %d kbps"), &mut Args(vec![BASE, 128]), 2),
            [line(14, "station radio at 128 kbps")]
        );
        assert_eq!(
            syslog_lines(&ctx, 16 | 3, Some(b"open: %m, %%m kept, %d%%\n"), &mut Args(vec![50]), 2),
            [line(19, "open: No such file or directory, %m kept, 50%\n")]
        );
        assert_eq!(
            syslog_lines(&ctx, 0x10000 | 4, Some(b"masked"), &mut Args(vec![]), 2),
            [line(35, "syslog: unknown facility/priority: 10004"), line(12, "masked")]
        );
        assert_eq!(syslog_lines(&ctx, 7, Some(b""), &mut Args(vec![]), 2), [line(15, "")]);
        assert_eq!(syslog_lines(&ctx, 7, None, &mut Args(vec![]), 2), [line(15, "")]);
        vm.unmap(BASE, 0x1000).unwrap();
    }

    /// The C locale's wide character classes are the ASCII ones; values
    /// from 128 up have none.
    #[test]
    fn c_locale_wide_character_classes() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        let classes: [(&str, u64, HleCallback, fn(&u8) -> bool); 12] = [
            ("alnum", 0x40_0100, cb_iswalnum, u8::is_ascii_alphanumeric),
            ("alpha", 0x100, cb_iswalpha, u8::is_ascii_alphabetic),
            ("blank", 0x2_0000, cb_iswblank, |c| *c == b' ' || *c == b'\t'),
            ("cntrl", 0x200, cb_iswcntrl, u8::is_ascii_control),
            ("digit", 0x400, cb_iswdigit, u8::is_ascii_digit),
            ("graph", 0x800, cb_iswgraph, u8::is_ascii_graphic),
            ("lower", 0x1000, cb_iswlower, u8::is_ascii_lowercase),
            ("print", 0x4_0000, cb_iswprint, |c| (0x20..0x7F).contains(c)),
            ("punct", 0x2000, cb_iswpunct, u8::is_ascii_punctuation),
            ("space", 0x4000, cb_iswspace, |c| matches!(*c, b'\t'..=b'\r' | b' ')),
            ("upper", 0x8000, cb_iswupper, u8::is_ascii_uppercase),
            ("xdigit", 0x1_0000, cb_iswxdigit, u8::is_ascii_hexdigit),
        ];
        for (name, mask, cb, narrow) in classes {
            assert_eq!(wctype_mask(name.as_bytes()), mask, "{name}");
            for c in 0..300u32 {
                let expected = c < 128 && narrow(&(c as u8));
                set(&mut ctx, &[c as u64, mask]);
                assert_eq!(cb(&mut ctx) != 0, expected, "{name}({c:#x})");
                assert_eq!(cb_iswctype(&mut ctx) != 0, expected, "iswctype({c:#x}, {name})");
            }
            set(&mut ctx, &[0xFFFF_FFFF, mask]);
            assert_eq!(cb(&mut ctx), 0);
            assert_eq!(cb_iswctype(&mut ctx), 0);
        }
        for c in (0..300u64).chain([0xFFFF_FFFF, 0x1_0000_0041]) {
            set(&mut ctx, &[c]);
            assert_eq!(cb_iswascii(&mut ctx) != 0, (c as u32) < 0x80, "iswascii({c:#x})");
        }
        for (name, mask) in [("ideogram", 0x8_0000), ("special", 0x10_0000), ("phonogram", 0x20_0000), ("number", 0x40_0000), ("rune", 0xFFFF_FF00), ("", 0), ("Alpha", 0)] {
            assert_eq!(wctype_mask(name.as_bytes()), mask, "{name:?}");
        }
        set(&mut ctx, &[b'A' as u64, 0]);
        assert_eq!(cb_iswctype(&mut ctx), 0);
        set(&mut ctx, &[b'7' as u64, 0x40_0000]);
        assert_eq!(cb_iswctype(&mut ctx), 1);
        for c in [0u32, b'A' as u32, b'z' as u32, b'5' as u32, 0xC9, 0x100, 0xFFFF_FFFF] {
            set(&mut ctx, &[c as u64]);
            let lower = if c < 128 { (c as u8).to_ascii_lowercase() as u32 } else { c };
            let upper = if c < 128 { (c as u8).to_ascii_uppercase() as u32 } else { c };
            assert_eq!(cb_towlower(&mut ctx) as u32, lower);
            assert_eq!(cb_towupper(&mut ctx) as u32, upper);
        }
        for (c, expected) in [(-1i64, 0xFFFF_FFFFu32), (0, 0), (b'A' as i64, b'A' as u32), (0xE9, 0xE9), (-23, 0xE9)] {
            set(&mut ctx, &[c as u64]);
            assert_eq!(cb_btowc(&mut ctx) as u32, expected);
        }
        for (c, expected) in [(b'A' as u64, b'A' as i64), (0xE9, 0xE9), (0, 0), (0x100, -1), (0x20AC, -1), (0xFFFF_FFFF, -1)] {
            set(&mut ctx, &[c]);
            assert_eq!(cb_wctob(&mut ctx), expected);
        }
    }

    /// wcsrtombs converts guest (16-bit) units in the C locale.
    #[test]
    fn wcsrtombs_converts_guest_units() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E66_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let units = |u: &[u16]| -> Vec<u8> { u.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let (text, empty, invalid, bytes) = (BASE, BASE + 0x20, BASE + 0x40, BASE + 0x60);
        ctx.write_bytes(text, &units(&[b'a' as u16, b'b' as u16, b'c' as u16, 0]));
        ctx.write_bytes(empty, &units(&[0]));
        ctx.write_bytes(invalid, &units(&[b'a' as u16, 0x100, b'b' as u16, 0]));
        ctx.write_bytes(bytes, &units(&[0x7F, 0x80, 0xE9, 0xFF, 0]));
        let (src, out) = (BASE + 0x100, BASE + 0x200);
        let run = |ctx: &mut CpuContext, dst: u64, from: u64, len: u64| {
            ctx.write_u64(src, from);
            set(ctx, &[dst, src, len, 0]);
            cb_wcsrtombs(ctx)
        };
        let resume = |ctx: &mut CpuContext, dst: u64, len: u64| {
            set(ctx, &[dst, src, len, 0]);
            cb_wcsrtombs(ctx)
        };
        let fresh = |ctx: &CpuContext| ctx.write_bytes(out, b"xxxxxxxx");
        fresh(&ctx);
        assert_eq!(run(&mut ctx, out, text, 8), 3);
        assert_eq!(ctx.read_u64(src), Some(0));
        assert_eq!(ctx.read_bytes(out, 8).unwrap(), b"abc\0xxxx");

        fresh(&ctx);
        assert_eq!(run(&mut ctx, out, text, 0), 0);
        assert_eq!(ctx.read_u64(src), Some(text));
        assert_eq!(ctx.read_bytes(out, 1).unwrap(), b"x");
        assert_eq!(resume(&mut ctx, out, 2), 2);
        assert_eq!(ctx.read_u64(src), Some(text + 4));
        assert_eq!(ctx.read_bytes(out, 8).unwrap(), b"abxxxxxx");
        assert_eq!(resume(&mut ctx, out + 2, 1), 1);
        assert_eq!(ctx.read_u64(src), Some(text + 6));
        assert_eq!(ctx.read_bytes(out, 8).unwrap(), b"abcxxxxx");
        assert_eq!(resume(&mut ctx, out + 3, 1), 0);
        assert_eq!(ctx.read_u64(src), Some(0));
        assert_eq!(ctx.read_bytes(out, 8).unwrap(), b"abc\0xxxx");

        fresh(&ctx);
        assert_eq!(run(&mut ctx, out, empty, 1), 0);
        assert_eq!(ctx.read_u64(src), Some(0));
        assert_eq!(ctx.read_bytes(out, 2).unwrap(), b"\0x");
        assert_eq!(run(&mut ctx, out, bytes, 8), 4);
        assert_eq!(ctx.read_bytes(out, 5).unwrap(), b"\x7f\x80\xe9\xff\0");

        assert_eq!(run(&mut ctx, 0, text, 0), 3);
        assert_eq!(ctx.read_u64(src), Some(text), "a length query leaves the source");
        assert_eq!(run(&mut ctx, 0, empty, 0), 0);

        fresh(&ctx);
        assert_eq!(run(&mut ctx, out, invalid, 8), -1);
        assert_eq!(ctx.read_u64(src), Some(invalid + 2), "points at the invalid unit");
        assert_eq!(ctx.read_bytes(out, 8).unwrap(), b"axxxxxxx");
        assert_eq!(resume(&mut ctx, out, 0), 0);
        assert_eq!(ctx.read_u64(src), Some(invalid + 2));
        assert_eq!(run(&mut ctx, 0, invalid, 0), -1);
        assert_eq!(ctx.read_u64(src), Some(invalid));
        assert_eq!(run(&mut ctx, out, invalid, 1), 1);
        assert_eq!(ctx.read_u64(src), Some(invalid + 2));
        vm.unmap(BASE, 0x1000).unwrap();
    }

    /// _Iswctype takes a class index 1..=12 in the order of the isw*
    /// functions.
    #[test]
    fn iswctype_classes_are_indices() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        let classes: [HleCallback; 12] = [
            cb_iswalnum, cb_iswalpha, cb_iswcntrl, cb_iswdigit, cb_iswgraph, cb_iswlower, cb_iswprint, cb_iswpunct,
            cb_iswspace, cb_iswupper, cb_iswxdigit, cb_iswblank,
        ];
        for (i, cb) in classes.iter().enumerate() {
            let class = i as u64 + 1;
            for c in 0..0x300u64 {
                set(&mut ctx, &[c, class]);
                let indexed = cb_iswctype_class(&mut ctx) != 0;
                assert_eq!(indexed, cb(&mut ctx) != 0, "_Iswctype({c:#x}, {class})");
            }
            set(&mut ctx, &[0xFFFF_FFFF, class]);
            assert_eq!(cb_iswctype_class(&mut ctx), 0, "WEOF");
        }
        let check = |ctx: &mut CpuContext, c: u8, class: u64| {
            set(ctx, &[c as u64, class]);
            cb_iswctype_class(ctx) != 0
        };
        assert!(check(&mut ctx, b'A', 2) && !check(&mut ctx, b'1', 2), "alpha");
        assert!(check(&mut ctx, b'7', 4) && check(&mut ctx, b'f', 11), "digit and xdigit");
        assert!(check(&mut ctx, b'\t', 12) && !check(&mut ctx, b'\n', 12), "blank");
        assert!(check(&mut ctx, b'\n', 9) && !check(&mut ctx, b' ', 5), "space and graph");
        assert!(check(&mut ctx, b'a', 6) && !check(&mut ctx, b'a', 10), "lower and upper");
        assert!(check(&mut ctx, 0x7f, 3) && check(&mut ctx, b'!', 8), "cntrl and punct");
        assert!(!check(&mut ctx, 0xe9, 2), "the C locale has no non-ASCII letters");
        for class in [0, 13, 0x100] {
            assert!(!check(&mut ctx, b'a', class), "unknown class {class:#x}");
        }
    }

    /// wcstombs_s / wcsrtombs_s.
    #[test]
    fn wcstombs_s_bounds_and_errors() {
        use crate::context::CpuRegister::R8;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5F80_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let units = |u: &[u16]| -> Vec<u8> { u.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let (text, wide, result, buf) = (BASE, BASE + 0x20, BASE + 0x40, BASE + 0x60);
        ctx.write_bytes(text, &units(&[b'a' as u16, b'b' as u16, b'c' as u16, 0]));
        ctx.write_bytes(wide, &units(&[b'a' as u16, 0x100, 0]));
        let failed = u64::MAX;
        let call = |ctx: &mut CpuContext, res: u64, dst: u64, cap: u64, src: u64, limit: u64| {
            ctx.write_bytes(buf, b"xxxxxxxx");
            ctx.write_u64(result, 0);
            set(ctx, &[res, dst, cap, src]);
            ctx.set(R8, limit);
            cb_wcstombs_s(ctx)
        };
        let res = |ctx: &CpuContext| ctx.read_u64(result).unwrap();
        let out = |ctx: &CpuContext| ctx.read_bytes(buf, 8).unwrap();
        assert_eq!(call(&mut ctx, result, buf, 8, text, 8), 0);
        assert_eq!((res(&ctx), &out(&ctx)[..4]), (3, &b"abc\0"[..]));
        assert_eq!(call(&mut ctx, result, buf, 8, text, 2), 0);
        assert_eq!((res(&ctx), &out(&ctx)[..3]), (2, &b"ab\0"[..]));
        assert_eq!(call(&mut ctx, result, 0, 0, text, 0), 0);
        assert_eq!(res(&ctx), 3);
        assert_eq!(call(&mut ctx, result, buf, 3, text, 3), ERANGE);
        assert_eq!((res(&ctx), out(&ctx)[0]), (failed, 0));
        assert_eq!(call(&mut ctx, result, 0, 4, text, 4), EINVAL);
        assert_eq!(res(&ctx), failed);
        assert_eq!(call(&mut ctx, result, buf, 8, 0, 8), EINVAL);
        assert_eq!((res(&ctx), out(&ctx)[0]), (failed, 0));
        assert_eq!(call(&mut ctx, 0, buf, 8, text, 8), EINVAL);
        assert_eq!(out(&ctx)[0], 0);
        assert_eq!(call(&mut ctx, result, buf, 0, text, 4), EINVAL);
        assert_eq!((res(&ctx), out(&ctx)[0]), (failed, b'x'));
        assert_eq!(call(&mut ctx, result, buf, 8, wide, 8), EILSEQ_S);
        assert_eq!((res(&ctx), &out(&ctx)[..2]), (failed, &b"a\0"[..]));
        assert_eq!(call(&mut ctx, result, 0, 0, wide, 0), EILSEQ_S);
        assert_eq!(res(&ctx), failed);
        // wcsrtombs_s moves the caller's source pointer.
        let (cell, state) = (BASE + 0x100, BASE + 0x200);
        ctx.write_u64(cell, text);
        set(&mut ctx, &[result, buf, 8, cell]);
        ctx.set(R8, 2);
        ctx.set(crate::context::CpuRegister::R9, state);
        assert_eq!(cb_wcsrtombs_s(&mut ctx), 0);
        assert_eq!((res(&ctx), ctx.read_u64(cell)), (2, Some(text + 4)));
        ctx.set(R8, 8);
        assert_eq!(cb_wcsrtombs_s(&mut ctx), 0);
        assert_eq!((res(&ctx), ctx.read_u64(cell)), (1, Some(0)));
        ctx.set(crate::context::CpuRegister::R9, 0);
        assert_eq!(cb_wcsrtombs_s(&mut ctx), EINVAL, "a null state");
        vm.unmap(BASE, 0x1000).unwrap();
    }

    /// _WStoul over the ASCII prefix.
    #[test]
    fn wstoul_reads_the_ascii_prefix() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5F81_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (text, end) = (BASE, BASE + 0x800);
        let parse = |ctx: &mut CpuContext, s: &str, base: u64| {
            let units: Vec<u8> = s.encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect();
            ctx.write_bytes(text, &units);
            set(ctx, &[text, end, base]);
            let v = cb_wstoul(ctx) as u64;
            (v, (ctx.read_u64(end).unwrap() - text) / 2)
        };
        assert_eq!(parse(&mut ctx, "  4294967296", 10), (4294967296, 12));
        assert_eq!(parse(&mut ctx, "18446744073709551615", 10), (u64::MAX, 20));
        assert_eq!(parse(&mut ctx, "18446744073709551616", 10), (u64::MAX, 20), "overflow");
        assert_eq!(parse(&mut ctx, "-1", 10), (u64::MAX, 2));
        assert_eq!(parse(&mut ctx, "\t+7xyz", 10), (7, 3));
        assert_eq!(parse(&mut ctx, "0x1F", 16), (31, 4));
        assert_eq!(parse(&mut ctx, "0x1f", 0), (31, 4));
        assert_eq!(parse(&mut ctx, "017", 0), (15, 3));
        assert_eq!(parse(&mut ctx, "z", 36), (35, 1));
        assert_eq!(parse(&mut ctx, "0b1", 0), (0, 1), "no binary prefix");
        assert_eq!(parse(&mut ctx, "0xg", 16), (0, 1), "prefix without digits");
        assert_eq!(parse(&mut ctx, "12\u{e9}3", 10), (12, 2), "stops at a non-ASCII unit");
        assert_eq!(parse(&mut ctx, "\u{3000}5", 10), (0, 0), "non-ASCII space is not skipped");
        assert_eq!(parse(&mut ctx, "", 10), (0, 0));
        assert_eq!(parse(&mut ctx, "1", 1), (0, 0), "base 1");
        assert_eq!(parse(&mut ctx, "1", 37), (0, 0), "base 37");
        set(&mut ctx, &[text, 0, 10]);
        ctx.write_bytes(text, &[b'9', 0, 0, 0]);
        assert_eq!(cb_wstoul(&mut ctx), 9, "a null end pointer");
        vm.unmap(BASE, 0x1000).unwrap();
    }

    #[test]
    fn wide_helpers() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5DA1_0000_0000;
        vm.map(BASE, 0x2000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (a, b, d) = (BASE, BASE + 0x100, BASE + 0x200);
        ctx.write_bytes(a, &wide("abc"));
        ctx.write_bytes(b, &wide("abd"));
        set(&mut ctx, &[a, b]);
        assert_eq!(cb_wcscmp(&mut ctx), -1);
        assert_eq!(cb_wcscoll(&mut ctx), -1);
        set(&mut ctx, &[b, a]);
        assert_eq!(cb_wcscmp(&mut ctx), 1);
        set(&mut ctx, &[a, b, 2]);
        assert_eq!(cb_wcsncmp(&mut ctx), 0);
        set(&mut ctx, &[a]);
        assert_eq!(cb_wcsmisc(&mut ctx), 3);
        set(&mut ctx, &[a, b'c' as u64]);
        assert_eq!(cb_wcschr(&mut ctx), (a + 4) as HleResult);
        set(&mut ctx, &[a, 0]);
        assert_eq!(cb_wcschr(&mut ctx), (a + 6) as HleResult);
        set(&mut ctx, &[a, b'z' as u64]);
        assert_eq!(cb_wcschr(&mut ctx), 0);

        set(&mut ctx, &[d, a]);
        assert_eq!(cb_wcscpy(&mut ctx), d as HleResult);
        assert_eq!(read_w(&ctx, d), "abc");

        // wcsncpy pads with terminators and leaves the rest untouched.
        ctx.write_bytes(d, &[0xAA; 16]);
        set(&mut ctx, &[d, a, 5]);
        assert_eq!(cb_wcsncpy(&mut ctx), d as HleResult);
        assert_eq!(ctx.read_bytes(d, 12).unwrap(), b"a\0b\0c\0\0\0\0\0\xAA\xAA");
        ctx.write_bytes(d, &[0xAA; 8]);
        set(&mut ctx, &[d, a, 2]);
        cb_wcsncpy(&mut ctx);
        assert_eq!(ctx.read_bytes(d, 6).unwrap(), b"a\0b\0\xAA\xAA");

        // wcscpy_s
        set(&mut ctx, &[d, 3, a]);
        assert_eq!(cb_wcscpy_s(&mut ctx), ERANGE);
        assert_eq!(read_w(&ctx, d), "");
        set(&mut ctx, &[d, 4, a]);
        assert_eq!(cb_wcscpy_s(&mut ctx), 0);
        assert_eq!(read_w(&ctx, d), "abc");
        set(&mut ctx, &[d, 4, 0]);
        assert_eq!(cb_wcscpy_s(&mut ctx), EINVAL);
        set(&mut ctx, &[0, 4, a]);
        assert_eq!(cb_wcscpy_s(&mut ctx), EINVAL);

        // wcsncpy_s
        set(&mut ctx, &[d, 3, a, 2]);
        assert_eq!(cb_wcsncpy_s(&mut ctx), 0);
        assert_eq!(read_w(&ctx, d), "ab");
        set(&mut ctx, &[d, 3, a, 3]);
        assert_eq!(cb_wcsncpy_s(&mut ctx), ERANGE);
        assert_eq!(read_w(&ctx, d), "");
        set(&mut ctx, &[d, 3, a, u64::MAX]);
        assert_eq!(cb_wcsncpy_s(&mut ctx), STRUNCATE);
        assert_eq!(read_w(&ctx, d), "ab");
        set(&mut ctx, &[d, 4, a, u64::MAX]);
        assert_eq!(cb_wcsncpy_s(&mut ctx), 0);
        assert_eq!(read_w(&ctx, d), "abc");
        set(&mut ctx, &[d, 4, a, 0]);
        assert_eq!(cb_wcsncpy_s(&mut ctx), 0);
        assert_eq!(read_w(&ctx, d), "");
    }
}
