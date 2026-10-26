// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libScePosixForWebKit: fnmatch, strcasestr, isatty, mkstemp, the
//! getaddrinfo family and getservbyname.
//!
//! The console is offline: getaddrinfo and getnameinfo only convert numeric
//! addresses (a host name fails with EAI_AGAIN, as gethostbyname fails with
//! TRY_AGAIN); service names come from the host's services database.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use std::cell::Cell;
use std::ffi::{CStr, CString};
use std::sync::OnceLock;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libScePosixForWebKit";

/// Longest guest string read for patterns, names and services.
const MAX_STRING: usize = 1 << 16;

// ---------------------------------------------------------------------------
// fnmatch
// ---------------------------------------------------------------------------

const FNM_NOMATCH: i32 = 1;
const FNM_NOESCAPE: i32 = 0x01;
const FNM_PATHNAME: i32 = 0x02;
const FNM_PERIOD: i32 = 0x04;
const FNM_LEADING_DIR: i32 = 0x08;
const FNM_CASEFOLD: i32 = 0x10;

enum Range {
    Match(usize),
    NoMatch,
    Error,
}

fn fold(c: u8, flags: i32) -> u8 {
    if flags & FNM_CASEFOLD != 0 { c.to_ascii_lowercase() } else { c }
}

/// Byte `i` of a NUL-terminated string held without its terminator.
fn at(s: &[u8], i: usize) -> u8 {
    s.get(i).copied().unwrap_or(0)
}

/// Match `test` against the bracket expression starting at `p` (just
/// past '['); Match carries the index after the closing ']'.
fn match_range(pattern: &[u8], mut p: usize, test: u8, flags: i32) -> Range {
    let negate = matches!(at(pattern, p), b'!' | b'^');
    if negate {
        p += 1;
    }
    let test = fold(test, flags);
    let mut ok = false;
    let origin = p;
    loop {
        let c = at(pattern, p);
        if c == b']' && p > origin {
            p += 1;
            break;
        }
        if c == 0 {
            return Range::Error;
        }
        if c == b'/' && flags & FNM_PATHNAME != 0 {
            return Range::NoMatch;
        }
        if c == b'\\' && flags & FNM_NOESCAPE == 0 {
            p += 1;
        }
        let low = fold(at(pattern, p), flags);
        if at(pattern, p) != 0 {
            p += 1;
        }
        if at(pattern, p) == b'-' && at(pattern, p + 1) != 0 && at(pattern, p + 1) != b']' {
            p += 1;
            if at(pattern, p) == b'\\' && flags & FNM_NOESCAPE == 0 {
                p += 1;
            }
            let high = at(pattern, p);
            if high == 0 {
                return Range::Error;
            }
            p += 1;
            if low <= test && test <= fold(high, flags) {
                ok = true;
            }
        } else if low == test {
            ok = true;
        }
    }
    if ok == negate { Range::NoMatch } else { Range::Match(p) }
}

/// BSD fnmatch: 0 on a match, FNM_NOMATCH otherwise. A malformed bracket
/// expression matches its '[' literally; character classes are not
/// supported.
pub(crate) fn fnmatch(pattern: &[u8], string: &[u8], flags: i32) -> i32 {
    let (mut p, mut s) = (0usize, 0usize);
    let mut backtrack: Option<(usize, usize)> = None;
    let leading_period =
        |at_s: usize| at(string, at_s) == b'.' && flags & FNM_PERIOD != 0 && (at_s == 0 || (flags & FNM_PATHNAME != 0 && string[at_s - 1] == b'/'));
    loop {
        let pc = at(pattern, p);
        if pc != 0 {
            p += 1;
        }
        let sc = at(string, s);
        let mut mismatch = false;
        let mut literal = false;
        let mut expected = pc;
        match pc {
            0 => {
                if (flags & FNM_LEADING_DIR != 0 && sc == b'/') || sc == 0 {
                    return 0;
                }
                mismatch = true;
            }
            b'?' => {
                if sc == 0 {
                    return FNM_NOMATCH;
                }
                if (sc == b'/' && flags & FNM_PATHNAME != 0) || leading_period(s) {
                    mismatch = true;
                } else {
                    s += 1;
                }
            }
            b'*' => {
                while at(pattern, p) == b'*' {
                    p += 1;
                }
                let c = at(pattern, p);
                if leading_period(s) {
                    mismatch = true;
                } else if c == 0 {
                    if flags & FNM_PATHNAME != 0 {
                        let slash = string[s.min(string.len())..].contains(&b'/');
                        return if flags & FNM_LEADING_DIR != 0 || !slash { 0 } else { FNM_NOMATCH };
                    }
                    return 0;
                } else if c == b'/' && flags & FNM_PATHNAME != 0 {
                    match string[s.min(string.len())..].iter().position(|&b| b == b'/') {
                        Some(off) => s += off,
                        None => return FNM_NOMATCH,
                    }
                } else {
                    backtrack = Some((p, s));
                }
            }
            b'[' => {
                if sc == 0 {
                    return FNM_NOMATCH;
                }
                if (sc == b'/' && flags & FNM_PATHNAME != 0) || leading_period(s) {
                    mismatch = true;
                } else {
                    match match_range(pattern, p, sc, flags) {
                        Range::Error => literal = true,
                        Range::Match(next) => {
                            p = next;
                            s += 1;
                        }
                        Range::NoMatch => mismatch = true,
                    }
                }
            }
            b'\\' => {
                if flags & FNM_NOESCAPE == 0 {
                    expected = at(pattern, p);
                    if expected == 0 {
                        return FNM_NOMATCH;
                    }
                    p += 1;
                }
                literal = true;
            }
            _ => literal = true,
        }
        if literal {
            if sc != 0 {
                s += 1;
            }
            if fold(expected, flags) != fold(sc, flags) {
                mismatch = true;
            }
        }
        if mismatch {
            let Some((bp, bs)) = backtrack else {
                return FNM_NOMATCH;
            };
            let bc = at(string, bs);
            if bc == 0 || (bc == b'/' && flags & FNM_PATHNAME != 0) {
                return FNM_NOMATCH;
            }
            backtrack = Some((bp, bs + 1));
            p = bp;
            s = bs + 1;
        }
    }
}

/// fnmatch(pattern, string, flags)
fn cb_fnmatch(ctx: &mut CpuContext) -> HleResult {
    let (pattern, string) = (ctx.rdi(), ctx.rsi());
    let (Some(pattern), Some(string)) = (ctx.read_cstr_bytes(pattern, MAX_STRING), ctx.read_cstr_bytes(string, MAX_STRING)) else {
        log::warn!("[libc] fnmatch: NULL or unreadable pattern {:#x} / string {:#x}", ctx.rdi(), ctx.rsi());
        return FNM_NOMATCH as HleResult;
    };
    fnmatch(&pattern, &string, ctx.rdx() as i32) as HleResult
}

/// strcasestr(text, needle) -> the first case-insensitive occurrence of
/// `needle` in `text` (ASCII folding), or NULL.
fn cb_strcasestr(ctx: &mut CpuContext) -> HleResult {
    let (text_addr, needle_addr) = (ctx.rdi(), ctx.rsi());
    let (Some(text), Some(needle)) = (ctx.read_cstr_bytes(text_addr, MAX_STRING), ctx.read_cstr_bytes(needle_addr, MAX_STRING)) else {
        log::warn!("[libc] strcasestr: NULL or unreadable string {text_addr:#x} / {needle_addr:#x}");
        return 0;
    };
    if needle.is_empty() {
        return text_addr as HleResult;
    }
    text.windows(needle.len())
        .position(|w| w.eq_ignore_ascii_case(&needle))
        .map_or(0, |i| (text_addr + i as u64) as HleResult)
}

// ---------------------------------------------------------------------------
// isatty / mkstemp
// ---------------------------------------------------------------------------

const EBADF: i32 = 9;
const EFAULT: i32 = 14;
const EEXIST: i32 = 17;
const EINVAL: i32 = 22;
const ENOTTY: i32 = 25;
const ENAMETOOLONG: i32 = 63;

fn fail(ctx: &CpuContext, errno: i32) -> HleResult {
    crate::kernel::set_errno(ctx, errno);
    -1
}

/// isatty(fd): the standard descriptors are the host's; sockets and files
/// are not terminals (ENOTTY); anything else is EBADF.
fn cb_isatty(ctx: &mut CpuContext) -> HleResult {
    let fd = ctx.rdi() as i32;
    if (0..=2).contains(&fd) {
        if unsafe { libc::isatty(fd) } == 1 {
            return 1;
        }
        crate::kernel::set_errno(ctx, ENOTTY);
        return 0;
    }
    let open = fd >= 0 && (crate::net::is_kernel_socket(fd) || crate::filesystem::fd_is_open(fd));
    crate::kernel::set_errno(ctx, if open { ENOTTY } else { EBADF });
    0
}

const TEMPLATE_CHARACTERS: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
/// mkstemp templates of this length or more are ENAMETOOLONG.
const MKSTEMP_MAX: usize = 1024;

/// mkstemp(template): the trailing X's become random letters and digits
/// and the file is created with O_RDWR | O_CREAT | O_EXCL. When the name
/// exists, the next candidates are tried in order until every one was.
fn cb_mkstemp(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return fail(ctx, EFAULT);
    }
    let Some(mut pattern) = ctx.read_cstr_bytes(addr, MKSTEMP_MAX) else {
        return fail(ctx, EFAULT);
    };
    if pattern.is_empty() {
        return fail(ctx, EINVAL);
    }
    if pattern.len() >= MKSTEMP_MAX {
        return fail(ctx, ENAMETOOLONG);
    }
    let length = pattern.len();
    let start = length - pattern.iter().rev().take_while(|&&c| c == b'X').count();
    let mut random = vec![0u8; length - start];
    crate::misc_services::fill_random(&mut random);
    for (c, r) in pattern[start..].iter_mut().zip(&random) {
        *c = TEMPLATE_CHARACTERS[*r as usize % TEMPLATE_CHARACTERS.len()];
    }
    let initial = pattern[start..].to_vec();
    loop {
        if !ctx.write_bytes(addr + start as u64, &pattern[start..]) {
            return fail(ctx, EFAULT);
        }
        match crate::filesystem::create_exclusive(ctx, addr) {
            Ok(fd) => return fd,
            Err(EEXIST) => {}
            Err(e) => return fail(ctx, e),
        }
        let mut position = start;
        while position < length {
            let index = TEMPLATE_CHARACTERS.iter().position(|&c| c == pattern[position]).unwrap_or(0);
            pattern[position] = TEMPLATE_CHARACTERS[(index + 1) % TEMPLATE_CHARACTERS.len()];
            if pattern[position] != initial[position - start] {
                break;
            }
            position += 1;
        }
        if position == length {
            return fail(ctx, EEXIST);
        }
    }
}

// ---------------------------------------------------------------------------
// getaddrinfo / freeaddrinfo / getnameinfo / gai_strerror
// ---------------------------------------------------------------------------

const EAI_AGAIN: i32 = 2;
const EAI_BADFLAGS: i32 = 3;
const EAI_FAIL: i32 = 4;
const EAI_FAMILY: i32 = 5;
const EAI_MEMORY: i32 = 6;
const EAI_NONAME: i32 = 8;
const EAI_SERVICE: i32 = 9;
const EAI_SOCKTYPE: i32 = 10;
const EAI_BADHINTS: i32 = 12;
const EAI_OVERFLOW: i32 = 14;

const AI_PASSIVE: i32 = 0x1;
const AI_CANONNAME: i32 = 0x2;
const AI_NUMERICHOST: i32 = 0x4;
const AI_NUMERICSERV: i32 = 0x8;
const AI_ALL: i32 = 0x100;
const AI_ADDRCONFIG: i32 = 0x400;
const AI_V4MAPPED: i32 = 0x800;
const AI_KNOWN: i32 = 0xD0F;

const NI_NOFQDN: i32 = 0x1;
const NI_NUMERICHOST: i32 = 0x2;
const NI_NAMEREQD: i32 = 0x4;
const NI_NUMERICSERV: i32 = 0x8;
const NI_DGRAM: i32 = 0x10;

const AF_INET: i32 = 2;
const AF_INET6: i32 = 28;

/// struct addrinfo: flags, family, socktype, protocol (int), addrlen
/// (u32), canonname @24, addr @32, next @40; 48 bytes.
const ADDRINFO_SIZE: usize = 48;

/// Host EAI_* -> the console's value.
fn guest_eai(error: i32) -> i32 {
    match error {
        0 => 0,
        libc::EAI_AGAIN => EAI_AGAIN,
        libc::EAI_BADFLAGS => EAI_BADFLAGS,
        libc::EAI_FAMILY => EAI_FAMILY,
        libc::EAI_MEMORY => EAI_MEMORY,
        libc::EAI_NONAME => EAI_NONAME,
        libc::EAI_SERVICE => EAI_SERVICE,
        libc::EAI_SOCKTYPE => EAI_SOCKTYPE,
        libc::EAI_OVERFLOW => EAI_OVERFLOW,
        _ => EAI_FAIL,
    }
}

/// A guest sockaddr for a host one: [len, family, port, ...] with the
/// IPv6 flow info, address and scope after the port.
fn guest_sockaddr(sa: *const libc::sockaddr) -> Option<Vec<u8>> {
    let family = unsafe { (*sa).sa_family } as i32;
    if family == libc::AF_INET {
        let v4 = unsafe { &*(sa as *const libc::sockaddr_in) };
        let mut b = vec![16u8, AF_INET as u8];
        b.extend_from_slice(&v4.sin_port.to_ne_bytes());
        b.extend_from_slice(&v4.sin_addr.s_addr.to_ne_bytes());
        b.resize(16, 0);
        Some(b)
    } else if family == libc::AF_INET6 {
        let v6 = unsafe { &*(sa as *const libc::sockaddr_in6) };
        let mut b = vec![28u8, AF_INET6 as u8];
        b.extend_from_slice(&v6.sin6_port.to_ne_bytes());
        b.extend_from_slice(&v6.sin6_flowinfo.to_ne_bytes());
        b.extend_from_slice(&v6.sin6_addr.s6_addr);
        b.extend_from_slice(&v6.sin6_scope_id.to_ne_bytes());
        Some(b)
    } else {
        None
    }
}

/// One converted addrinfo: family, socktype, protocol, guest sockaddr and
/// canonical name.
#[derive(Debug, PartialEq)]
pub(crate) struct AddrInfo {
    pub family: i32,
    pub socktype: i32,
    pub protocol: i32,
    pub address: Vec<u8>,
    pub canonical: Option<Vec<u8>>,
}

/// Guest hints: flags, family, socktype, protocol.
#[derive(Clone, Copy, Default)]
pub(crate) struct Hints {
    pub flags: i32,
    pub family: i32,
    pub socktype: i32,
    pub protocol: i32,
}

/// Resolve offline: hosts must be numeric. Err is a guest EAI_* value.
pub(crate) fn lookup(node: Option<&[u8]>, service: Option<&[u8]>, hints: Option<Hints>) -> Result<Vec<AddrInfo>, i32> {
    let mut native: libc::addrinfo = unsafe { std::mem::zeroed() };
    let h = hints.unwrap_or_default();
    if hints.is_some() {
        if h.flags & !AI_KNOWN != 0 {
            return Err(EAI_BADFLAGS);
        }
        native.ai_family = match h.family {
            0 => libc::AF_UNSPEC,
            AF_INET => libc::AF_INET,
            AF_INET6 => libc::AF_INET6,
            _ => return Err(EAI_FAMILY),
        };
        native.ai_socktype = match h.socktype {
            0 => 0,
            1 => libc::SOCK_STREAM,
            2 => libc::SOCK_DGRAM,
            _ => return Err(EAI_SOCKTYPE),
        };
        native.ai_protocol = h.protocol;
        for (guest, host) in [
            (AI_PASSIVE, libc::AI_PASSIVE),
            (AI_CANONNAME, libc::AI_CANONNAME),
            (AI_NUMERICHOST, libc::AI_NUMERICHOST),
            (AI_NUMERICSERV, libc::AI_NUMERICSERV),
            (AI_ALL, libc::AI_ALL),
            (AI_ADDRCONFIG, libc::AI_ADDRCONFIG),
            (AI_V4MAPPED, libc::AI_V4MAPPED),
        ] {
            if h.flags & guest != 0 {
                native.ai_flags |= host;
            }
        }
    }
    // No DNS offline: only numeric hosts convert.
    native.ai_flags |= libc::AI_NUMERICHOST;
    let node_c = node.map(|n| CString::new(n).map_err(|_| EAI_NONAME)).transpose()?;
    let service_c = service.map(|s| CString::new(s).map_err(|_| EAI_SERVICE)).transpose()?;
    let mut result: *mut libc::addrinfo = std::ptr::null_mut();
    let error = unsafe {
        libc::getaddrinfo(
            node_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            service_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            &native,
            &mut result,
        )
    };
    if error != 0 {
        if error == libc::EAI_NONAME && node.is_some() && h.flags & AI_NUMERICHOST == 0 {
            log::info!("[net] getaddrinfo('{}') -> no DNS (offline)", String::from_utf8_lossy(node.unwrap_or_default()));
            return Err(EAI_AGAIN);
        }
        return Err(guest_eai(error));
    }
    let mut out = Vec::new();
    let mut failure = 0;
    let mut item = result;
    while !item.is_null() {
        let entry = unsafe { &*item };
        item = entry.ai_next;
        let Some(address) = guest_sockaddr(entry.ai_addr) else {
            failure = EAI_FAMILY;
            break;
        };
        out.push(AddrInfo {
            family: address[1] as i32,
            socktype: match entry.ai_socktype {
                libc::SOCK_STREAM => 1,
                libc::SOCK_DGRAM => 2,
                _ => 0,
            },
            protocol: entry.ai_protocol,
            address,
            canonical: (!entry.ai_canonname.is_null()).then(|| unsafe { CStr::from_ptr(entry.ai_canonname) }.to_bytes().to_vec()),
        });
    }
    unsafe { libc::freeaddrinfo(result) };
    if failure != 0 { Err(failure) } else { Ok(out) }
}

/// getaddrinfo(node, service, const struct addrinfo* hints, struct
/// addrinfo** res) -> 0 or EAI_*. Each result entry is one guest block
/// (entry, sockaddr, canonical name) released by freeaddrinfo.
fn cb_getaddrinfo(ctx: &mut CpuContext) -> HleResult {
    let (node, service, hints_addr, res) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    if res == 0 {
        return EAI_FAIL as HleResult;
    }
    if !ctx.write_u64(res, 0) {
        return EAI_FAIL as HleResult;
    }
    let hints = if hints_addr == 0 {
        None
    } else {
        let Some(raw) = ctx.read_bytes(hints_addr, ADDRINFO_SIZE) else {
            return EAI_FAIL as HleResult;
        };
        let i32_at = |o: usize| i32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
        if raw[16..20] != [0; 4] || raw[24..48].iter().any(|&b| b != 0) {
            return EAI_BADHINTS as HleResult;
        }
        Some(Hints { flags: i32_at(0), family: i32_at(4), socktype: i32_at(8), protocol: i32_at(12) })
    };
    let node = if node == 0 { None } else { ctx.read_cstr_bytes(node, MAX_STRING) };
    let service = if service == 0 { None } else { ctx.read_cstr_bytes(service, MAX_STRING) };
    let entries = match lookup(node.as_deref(), service.as_deref(), hints) {
        Ok(e) => e,
        Err(e) => return e as HleResult,
    };
    let flags = hints.map_or(0, |h| h.flags);
    let mut blocks = Vec::with_capacity(entries.len());
    for entry in &entries {
        let canonical = entry.canonical.as_ref().map_or(0, |c| c.len() + 1);
        match crate::font::galloc(ctx, (ADDRINFO_SIZE + entry.address.len() + canonical) as u64) {
            Some(b) => blocks.push(b),
            None => {
                blocks.into_iter().for_each(crate::font::gfree);
                return EAI_MEMORY as HleResult;
            }
        }
    }
    for (i, (entry, &block)) in entries.iter().zip(&blocks).enumerate() {
        let address = block + ADDRINFO_SIZE as u64;
        let mut bytes = Vec::with_capacity(ADDRINFO_SIZE + entry.address.len());
        for v in [flags, entry.family, entry.socktype, entry.protocol, entry.address.len() as i32, 0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let canonical = match &entry.canonical {
            Some(_) => address + entry.address.len() as u64,
            None => 0,
        };
        bytes.extend_from_slice(&canonical.to_le_bytes());
        bytes.extend_from_slice(&address.to_le_bytes());
        bytes.extend_from_slice(&blocks.get(i + 1).copied().unwrap_or(0).to_le_bytes());
        bytes.extend_from_slice(&entry.address);
        if let Some(c) = &entry.canonical {
            bytes.extend_from_slice(c);
            bytes.push(0);
        }
        if !ctx.write_bytes(block, &bytes) {
            blocks.into_iter().for_each(crate::font::gfree);
            return EAI_FAIL as HleResult;
        }
    }
    let first = blocks.first().copied().unwrap_or(0);
    ctx.write_u64(res, first);
    0
}

/// freeaddrinfo(struct addrinfo* ai): releases the list from `ai` on.
fn cb_freeaddrinfo(ctx: &mut CpuContext) -> HleResult {
    let mut entry = ctx.rdi();
    while entry != 0 {
        let next = ctx.read_u64(entry + 40).unwrap_or(0);
        crate::font::gfree(entry);
        entry = next;
    }
    0
}

/// getnameinfo(const struct sockaddr* sa, socklen_t salen, char* host,
/// socklen_t hostlen, char* serv, socklen_t servlen, int flags) -> 0 or
/// EAI_*. Hosts come back numeric (offline: NI_NAMEREQD cannot be met);
/// neither output is touched when one does not fit.
pub(crate) fn name_info(sa: &[u8], want_host: bool, want_service: bool, flags: i32) -> Result<(String, String), i32> {
    if sa.len() < 2 {
        return Err(EAI_FAMILY);
    }
    if flags & !0x1F != 0 {
        return Err(EAI_BADFLAGS);
    }
    let mut ss: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let len = match sa[1] as i32 {
        AF_INET if sa.len() >= 16 && sa[0] == 16 => {
            let v4 = unsafe { &mut *(&mut ss as *mut _ as *mut libc::sockaddr_in) };
            v4.sin_family = libc::AF_INET as libc::sa_family_t;
            v4.sin_port = u16::from_ne_bytes([sa[2], sa[3]]);
            v4.sin_addr.s_addr = u32::from_ne_bytes(sa[4..8].try_into().unwrap());
            std::mem::size_of::<libc::sockaddr_in>()
        }
        AF_INET6 if sa.len() >= 28 && sa[0] == 28 => {
            let v6 = unsafe { &mut *(&mut ss as *mut _ as *mut libc::sockaddr_in6) };
            v6.sin6_family = libc::AF_INET6 as libc::sa_family_t;
            v6.sin6_port = u16::from_ne_bytes([sa[2], sa[3]]);
            v6.sin6_flowinfo = u32::from_ne_bytes(sa[4..8].try_into().unwrap());
            v6.sin6_addr.s6_addr.copy_from_slice(&sa[8..24]);
            v6.sin6_scope_id = u32::from_ne_bytes(sa[24..28].try_into().unwrap());
            std::mem::size_of::<libc::sockaddr_in6>()
        }
        _ => return Err(EAI_FAMILY),
    };
    if want_host && flags & NI_NAMEREQD != 0 && flags & NI_NUMERICHOST == 0 {
        log::info!("[net] getnameinfo: a host name is required -> no DNS (offline)");
        return Err(EAI_AGAIN);
    }
    let mut native = libc::NI_NUMERICHOST;
    for (guest, host) in [(NI_NOFQDN, libc::NI_NOFQDN), (NI_NUMERICSERV, libc::NI_NUMERICSERV), (NI_DGRAM, libc::NI_DGRAM)] {
        if flags & guest != 0 {
            native |= host;
        }
    }
    let mut host = [0 as libc::c_char; libc::NI_MAXHOST as usize];
    let mut service = [0 as libc::c_char; 32];
    let error = unsafe {
        libc::getnameinfo(
            &ss as *const _ as *const libc::sockaddr,
            len as libc::socklen_t,
            if want_host { host.as_mut_ptr() } else { std::ptr::null_mut() },
            if want_host { host.len() as libc::socklen_t } else { 0 },
            if want_service { service.as_mut_ptr() } else { std::ptr::null_mut() },
            if want_service { service.len() as libc::socklen_t } else { 0 },
            native,
        )
    };
    if error != 0 {
        return Err(guest_eai(error));
    }
    let text = |b: &[libc::c_char]| unsafe { CStr::from_ptr(b.as_ptr()) }.to_string_lossy().into_owned();
    Ok((text(&host), text(&service)))
}

fn cb_getnameinfo(ctx: &mut CpuContext) -> HleResult {
    let (sa, salen, host, hostlen, serv, servlen) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx() as u32, ctx.r8(), ctx.r9() as u32);
    let flags = ctx.read_u64(ctx.get(crate::context::CpuRegister::Rsp) + 8).unwrap_or(0) as i32;
    if sa == 0 || salen < 2 {
        return EAI_FAMILY as HleResult;
    }
    if (host == 0 && hostlen != 0) || (serv == 0 && servlen != 0) {
        return EAI_FAIL as HleResult;
    }
    let Some(bytes) = ctx.read_bytes(sa, salen.min(28) as usize) else {
        return EAI_FAIL as HleResult;
    };
    let (want_host, want_service) = (host != 0 && hostlen != 0, serv != 0 && servlen != 0);
    let (h, s) = match name_info(&bytes, want_host, want_service, flags) {
        Ok(r) => r,
        Err(e) => return e as HleResult,
    };
    if (want_host && h.len() >= hostlen as usize) || (want_service && s.len() >= servlen as usize) {
        return EAI_OVERFLOW as HleResult;
    }
    let write = |addr: u64, text: &str| {
        let mut b = text.as_bytes().to_vec();
        b.push(0);
        ctx.write_bytes(addr, &b)
    };
    if (want_host && !write(host, &h)) || (want_service && !write(serv, &s)) {
        return EAI_FAIL as HleResult;
    }
    0
}

const GAI_MESSAGES: [&str; 15] = [
    "Success",
    "Address family for hostname not supported",
    "Temporary failure in name resolution",
    "Invalid value for ai_flags",
    "Non-recoverable failure in name resolution",
    "ai_family not supported",
    "Memory allocation failure",
    "No address associated with hostname",
    "hostname nor servname provided, or not known",
    "servname not supported for ai_socktype",
    "ai_socktype not supported",
    "System error returned in errno",
    "Invalid value for hints",
    "Resolved protocol is unknown",
    "Argument buffer overflow",
];
const GAI_UNKNOWN: &str = "Unknown error";

/// The message of an EAI_* value.
pub(crate) fn gai_message(error: i32) -> &'static str {
    usize::try_from(error).ok().and_then(|i| GAI_MESSAGES.get(i)).copied().unwrap_or(GAI_UNKNOWN)
}

/// Guest copies of the messages: the address of each, then of "Unknown error".
static GAI_STRINGS: OnceLock<Vec<u64>> = OnceLock::new();

/// gai_strerror(int ecode) -> const char*
fn cb_gai_strerror(ctx: &mut CpuContext) -> HleResult {
    let strings = GAI_STRINGS.get_or_init(|| {
        let all: Vec<&str> = GAI_MESSAGES.iter().copied().chain([GAI_UNKNOWN]).collect();
        let size: usize = all.iter().map(|m| m.len() + 1).sum();
        let Some(block) = crate::pthread::alloc_guest_block(ctx, size as u64) else {
            return Vec::new();
        };
        let mut addrs = Vec::new();
        let mut offset = 0u64;
        for m in all {
            let mut b = m.as_bytes().to_vec();
            b.push(0);
            ctx.write_bytes(block + offset, &b);
            addrs.push(block + offset);
            offset += b.len() as u64;
        }
        addrs
    });
    let error = ctx.rdi() as i32;
    let message = gai_message(error);
    let index = GAI_MESSAGES.iter().position(|&m| m == message).unwrap_or(GAI_MESSAGES.len());
    strings.get(index).copied().unwrap_or(0) as HleResult
}

// ---------------------------------------------------------------------------
// getservbyname
// ---------------------------------------------------------------------------

/// (name, port, protocol, aliases) of the services the library knows.
const SERVICES: &[(&str, u16, &str, &[&str])] = &[
    ("ftp-data", 20, "tcp", &[]),
    ("ftp-data", 20, "udp", &[]),
    ("ftp-data", 20, "sctp", &[]),
    ("ftp", 21, "tcp", &[]),
    ("ftp", 21, "udp", &[]),
    ("ftp", 21, "sctp", &[]),
    ("ssh", 22, "tcp", &[]),
    ("ssh", 22, "udp", &[]),
    ("ssh", 22, "sctp", &[]),
    ("telnet", 23, "tcp", &[]),
    ("telnet", 23, "udp", &[]),
    ("smtp", 25, "tcp", &["mail"]),
    ("smtp", 25, "udp", &["mail"]),
    ("domain", 53, "tcp", &[]),
    ("domain", 53, "udp", &[]),
    ("tftp", 69, "tcp", &[]),
    ("tftp", 69, "udp", &[]),
    ("http", 80, "tcp", &["www", "www-http"]),
    ("http", 80, "udp", &["www", "www-http"]),
    ("http", 80, "sctp", &["www", "www-http"]),
    ("pop3", 110, "tcp", &[]),
    ("pop3", 110, "udp", &[]),
    ("ntp", 123, "tcp", &[]),
    ("ntp", 123, "udp", &[]),
    ("imap", 143, "tcp", &["imap2", "imap4"]),
    ("imap", 143, "udp", &["imap2", "imap4"]),
    ("https", 443, "tcp", &[]),
    ("https", 443, "udp", &[]),
    ("https", 443, "sctp", &[]),
    ("rtsp", 554, "tcp", &[]),
    ("rtsp", 554, "udp", &[]),
    ("submission", 587, "tcp", &[]),
    ("submission", 587, "udp", &[]),
    ("imaps", 993, "tcp", &[]),
    ("imaps", 993, "udp", &[]),
    ("pop3s", 995, "tcp", &["spop3"]),
    ("pop3s", 995, "udp", &["spop3"]),
];

/// The first service named `name` (or one of its aliases), for
/// `protocol` when one is given.
pub(crate) fn find_service(name: &[u8], protocol: Option<&[u8]>) -> Option<&'static (&'static str, u16, &'static str, &'static [&'static str])> {
    SERVICES.iter().find(|(n, _, p, aliases)| {
        (n.as_bytes() == name || aliases.iter().any(|a| a.as_bytes() == name)) && protocol.is_none_or(|q| q == p.as_bytes())
    })
}

thread_local! {
    static SERVENT_BLOCK: Cell<u64> = const { Cell::new(0) };
}

/// struct servent (name @0, aliases @8, port @16 in network order,
/// protocol @24) and its strings, in per-thread storage.
const SERVENT_BLOCK_SIZE: u64 = 256;

/// getservbyname(name, proto) -> struct servent* (overwritten by the
/// thread's next call) or NULL.
fn cb_getservbyname(ctx: &mut CpuContext) -> HleResult {
    let (name, proto) = (ctx.rdi(), ctx.rsi());
    let Some(name) = ctx.read_cstr_bytes(name, MAX_STRING) else {
        log::warn!("[net] getservbyname: NULL or unreadable name {name:#x}");
        return 0;
    };
    let protocol = if proto == 0 { None } else { ctx.read_cstr_bytes(proto, MAX_STRING) };
    let Some(&(sname, port, sproto, aliases)) = find_service(&name, protocol.as_deref()) else {
        return 0;
    };
    let mut block = SERVENT_BLOCK.with(Cell::get);
    if block == 0 {
        let Some(b) = crate::pthread::alloc_guest_block(ctx, SERVENT_BLOCK_SIZE) else {
            return 0;
        };
        SERVENT_BLOCK.with(|c| c.set(b));
        block = b;
    }
    // Layout: entry (32), alias pointers (3 * 8), then the strings.
    let mut strings = Vec::new();
    let mut place = |text: &str| {
        let addr = block + 56 + strings.len() as u64;
        strings.extend_from_slice(text.as_bytes());
        strings.push(0);
        addr
    };
    let name_addr = place(sname);
    let proto_addr = place(sproto);
    let alias_addrs: Vec<u64> = aliases.iter().map(|a| place(a)).collect();
    let mut bytes = Vec::with_capacity(56 + strings.len());
    bytes.extend_from_slice(&name_addr.to_le_bytes());
    bytes.extend_from_slice(&(block + 32).to_le_bytes());
    bytes.extend_from_slice(&(port.to_be() as i32).to_le_bytes());
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(&proto_addr.to_le_bytes());
    for i in 0..3 {
        bytes.extend_from_slice(&alias_addrs.get(i).copied().unwrap_or(0).to_le_bytes());
    }
    bytes.extend_from_slice(&strings);
    if ctx.write_bytes(block, &bytes) { block as HleResult } else { 0 }
}

/// pthread_setname_np(thread, name): EINVAL (22) for a null thread or
/// name, ENAMETOOLONG (63) from 32 bytes on, ESRCH (3) for an unknown
/// thread and 0 otherwise; the stored name reaches pthread_getname_np.
fn cb_pthread_setname_np(ctx: &mut CpuContext) -> HleResult {
    let (handle, name) = (ctx.rdi(), ctx.rsi());
    if handle == 0 || name == 0 {
        return 22;
    }
    let Some(bytes) = ctx.read_cstr_bytes(name, 32) else {
        return 63;
    };
    if crate::pthread::set_thread_name(handle, &bytes) { 0 } else { 3 }
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("60q1rziOzzQ", "fnmatch", cb_fnmatch),
    ("eDmbt0P120g", "strcasestr", cb_strcasestr),
    ("InoC17C0v7k", "isatty", cb_isatty),
    ("uEtgPWgPxOk", "mkstemp", cb_mkstemp),
    ("+GYqID5YVVg", "getaddrinfo", cb_getaddrinfo),
    ("KMlPItNRcZ4", "freeaddrinfo", cb_freeaddrinfo),
    ("YGm7Vn3fkbk", "getnameinfo", cb_getnameinfo),
    ("5gTI2SV8lTM", "gai_strerror", cb_gai_strerror),
    ("H1bArJPagV0", "getservbyname", cb_getservbyname),
    ("cB4rMoKU4UI", "pthread_setname_np", cb_pthread_setname_np),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    fn args(ctx: &mut CpuContext, args: &[u64]) {
        for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
            ctx.set(*r, *v);
        }
    }

    fn put(ctx: &CpuContext, addr: u64, text: &[u8]) -> u64 {
        let mut b = text.to_vec();
        b.push(0);
        assert!(ctx.write_bytes(addr, &b));
        addr
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// Reference GuestFnmatch table.
    #[test]
    fn fnmatch_cases() {
        let cases: &[(&str, &str, i32, i32)] = &[
            ("*.txt", "notes.txt", 0x00, 0),
            ("*.txt", "notes.TXT", 0x00, 1),
            ("*.txt", "notes.TXT", 0x10, 0),
            ("a?c", "abc", 0x00, 0),
            ("a?c", "ac", 0x00, 1),
            ("a[bc]d", "acd", 0x00, 0),
            ("a[!bc]d", "acd", 0x00, 1),
            ("a[^bc]d", "aed", 0x00, 0),
            ("a[a-c]d", "abd", 0x00, 0),
            ("a[a-c]d", "add", 0x00, 1),
            ("a[]]d", "a]d", 0x00, 0),
            ("a[a-]d", "a-d", 0x00, 0),
            ("[A-C]x", "bx", 0x10, 0),
            ("*", "dir/file", 0x00, 0),
            ("*", "dir/file", 0x02, 1),
            ("dir/*", "dir/file", 0x02, 0),
            ("*/file", "dir/file", 0x02, 0),
            ("dir?file", "dir/file", 0x02, 1),
            ("dir[/]file", "dir/file", 0x02, 1),
            ("*", ".hidden", 0x04, 1),
            (".*", ".hidden", 0x04, 0),
            ("?hidden", ".hidden", 0x04, 1),
            ("dir/*", "dir/.hidden", 0x06, 1),
            ("dir/*", "dir/.hidden", 0x04, 0),
            ("dir/.*", "dir/.hidden", 0x06, 0),
            ("dir", "dir/sub/file", 0x08, 0),
            ("dir/*", "dir/sub/file", 0x0a, 0),
            ("dir", "dirx", 0x08, 1),
            ("\\*", "*", 0x00, 0),
            ("\\*", "x", 0x00, 1),
            ("\\*", "\\*", 0x01, 0),
            ("a\\", "a", 0x00, 1),
            ("a\\", "a\\", 0x01, 0),
            ("a*b*c", "axxbyyc", 0x00, 0),
            ("a*b*c", "axxbyy", 0x00, 1),
            ("*a*", "bbb", 0x00, 1),
            ("**a", "ba", 0x00, 0),
            ("", "", 0x00, 0),
            ("", "x", 0x00, 1),
            ("x", "", 0x00, 1),
            ("[", "[", 0x00, 0),
            ("[a", "[a", 0x00, 0),
            ("a[", "a[", 0x00, 0),
            ("[!]x", "x", 0x00, 1),
            ("*/", "dir/", 0x02, 0),
            ("*/", "dir", 0x02, 1),
            ("[[:digit:]]", "7", 0x00, 1),
            ("[[:digit:]]", "d]", 0x00, 0),
            ("[[:digit:]]", ":]", 0x00, 0),
            ("a[[:alpha:]]", "ab", 0x00, 1),
        ];
        for &(pattern, string, flags, result) in cases {
            assert_eq!(fnmatch(pattern.as_bytes(), string.as_bytes(), flags), result, "fnmatch({pattern:?}, {string:?}, {flags:#x})");
        }
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        args(&mut ctx, &[0, 0, 0]);
        assert_eq!(cb_fnmatch(&mut ctx), FNM_NOMATCH as HleResult);
    }

    /// Reference GuestStrings strcasestr cases.
    #[test]
    fn strcasestr_cases() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E12_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let text = put(&ctx, BASE, b"aABAbC");
        let find = |ctx: &mut CpuContext, hay: u64, needle: &[u8]| {
            let n = put(ctx, BASE + 0x100, needle);
            args(ctx, &[hay, n]);
            cb_strcasestr(ctx) as u64
        };
        assert_eq!(find(&mut ctx, text, b"ababc"), text + 1);
        assert_eq!(find(&mut ctx, text, b""), text);
        assert_eq!(find(&mut ctx, text, b"abcdef"), 0);
        let empty = BASE + 0x200;
        ctx.write_bytes(empty, &[0]);
        assert_eq!(find(&mut ctx, empty, b"a"), 0);
        let high = BASE + 0x300;
        ctx.write_bytes(high, &[0xFF, b'A', 0]);
        assert_eq!(find(&mut ctx, high, b"a"), high + 1);
    }

    /// Reference GuestPosixFilesystem: mkstemp names, collisions and errors;
    /// isatty on files, sockets and closed descriptors.
    #[test]
    fn mkstemp_and_isatty() {
        let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
        const BASE: u64 = 0x5E12_0010_0000;
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(vm);
        let errno = crate::kernel::bind_test_errno(&ctx);
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../local/tmp/posix-webkit-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        assert!(crate::filesystem::register_guest_path_mount("/k5wk0", &root));
        let mkstemp = |ctx: &mut CpuContext, path: u64| {
            args(ctx, &[path]);
            cb_mkstemp(ctx)
        };
        let prefix = "/k5wk0/file-";
        let pattern = put(&ctx, BASE, format!("{prefix}XXXXXXXXX").as_bytes());
        let fd = mkstemp(&mut ctx, pattern);
        assert!(fd >= 0);
        let name = ctx.read_cstring(pattern, 256).unwrap();
        assert!(name.starts_with(prefix) && name.len() == prefix.len() + 9);
        assert!(name[prefix.len()..].bytes().all(|c| TEMPLATE_CHARACTERS.contains(&c)));
        assert_eq!(std::fs::metadata(root.join(&name["/k5wk0/".len()..])).unwrap().len(), 0);
        ctx.write_u32(errno, 123);
        args(&mut ctx, &[fd as u64]);
        assert_eq!(cb_isatty(&mut ctx), 0);
        assert_eq!(ctx.read_u32(errno), Some(ENOTTY as u32));
        assert_eq!(crate::filesystem::close_fd(fd as i32), Ok(0));
        assert_eq!(cb_isatty(&mut ctx), 0);
        assert_eq!(ctx.read_u32(errno), Some(EBADF as u32));

        // Every candidate taken: EEXIST, and the files are not touched.
        for c in TEMPLATE_CHARACTERS {
            std::fs::write(root.join(format!("occupied-{}", *c as char)), "preserved").unwrap();
        }
        let occupied = put(&ctx, BASE + 0x100, b"/k5wk0/occupied-X");
        assert_eq!(mkstemp(&mut ctx, occupied), -1);
        assert_eq!(ctx.read_u32(errno), Some(EEXIST as u32));
        assert_eq!(std::fs::read_to_string(root.join("occupied-a")).unwrap(), "preserved");
        // The free name is found by stepping through the candidates.
        std::fs::remove_file(root.join("occupied-7")).unwrap();
        put(&ctx, occupied, b"/k5wk0/occupied-X");
        let fd = mkstemp(&mut ctx, occupied);
        assert!(fd >= 0);
        assert_eq!(ctx.read_cstring(occupied, 64).as_deref(), Some("/k5wk0/occupied-7"));
        assert_eq!(crate::filesystem::close_fd(fd as i32), Ok(0));
        // No X: the name is used as is, once.
        let fixed = put(&ctx, BASE + 0x200, b"/k5wk0/fixed");
        let fd = mkstemp(&mut ctx, fixed);
        assert!(fd >= 0);
        assert_eq!(crate::filesystem::close_fd(fd as i32), Ok(0));
        assert_eq!(mkstemp(&mut ctx, fixed), -1);
        assert_eq!(ctx.read_u32(errno), Some(EEXIST as u32));

        let empty = put(&ctx, BASE + 0x300, b"");
        assert_eq!(mkstemp(&mut ctx, empty), -1);
        assert_eq!(ctx.read_u32(errno), Some(EINVAL as u32));
        assert_eq!(mkstemp(&mut ctx, 0), -1);
        assert_eq!(ctx.read_u32(errno), Some(EFAULT as u32));
        let long = put(&ctx, BASE + 0x1000, &[b'X'; 1024]);
        assert_eq!(mkstemp(&mut ctx, long), -1);
        assert_eq!(ctx.read_u32(errno), Some(ENAMETOOLONG as u32));
        assert_eq!(ctx.read_bytes(long, 1024).unwrap(), vec![b'X'; 1024]);

        for fd in [-1i64 as u64, 100_000] {
            args(&mut ctx, &[fd]);
            assert_eq!(cb_isatty(&mut ctx), 0);
            assert_eq!(ctx.read_u32(errno), Some(EBADF as u32));
        }
        crate::filesystem::unregister_guest_path_mount("/k5wk0");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Reference GuestResolver: getnameinfo, gai_strerror, numeric
    /// getaddrinfo round trips and hint validation.
    #[test]
    fn resolver_numeric() {
        let v4 = [16u8, 2, 0x6d, 0x06, 127, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(name_info(&v4, true, true, 10), Ok(("127.0.0.1".into(), "27910".into())));
        let mut v6 = [0u8; 28];
        v6[0] = 28;
        v6[1] = 28;
        v6[23] = 1;
        assert_eq!(name_info(&v6, true, false, 10).map(|r| r.0), Ok("::1".into()));
        assert_eq!(name_info(&v4[..2], true, false, 10), Err(EAI_FAMILY));
        assert_eq!(name_info(&v4, true, false, 32), Err(EAI_BADFLAGS));
        assert_eq!(name_info(&v4, true, false, NI_NAMEREQD), Err(EAI_AGAIN));

        for (e, m) in [
            (0, "Success"),
            (3, "Invalid value for ai_flags"),
            (8, "hostname nor servname provided, or not known"),
            (10, "ai_socktype not supported"),
            (14, "Argument buffer overflow"),
            (-1, "Unknown error"),
            (15, "Unknown error"),
        ] {
            assert_eq!(gai_message(e), m);
        }

        let hints = Hints { flags: AI_NUMERICHOST | AI_NUMERICSERV, socktype: 1, ..Default::default() };
        for numeric in ["127.0.0.1", "::1"] {
            let entries = lookup(Some(numeric.as_bytes()), Some(b"27910"), Some(hints)).unwrap();
            assert!(!entries.is_empty());
            for e in entries {
                assert!(e.family == AF_INET || e.family == AF_INET6);
                assert_eq!(e.socktype, 1);
                assert_eq!(name_info(&e.address, true, true, 10), Ok((numeric.into(), "27910".into())));
            }
        }
        assert_eq!(lookup(Some(b"127.0.0.1"), Some(b"1"), Some(Hints { family: 1234, ..hints })), Err(EAI_FAMILY));
        assert_eq!(lookup(Some(b"127.0.0.1"), Some(b"1"), Some(Hints { flags: 0x200, ..hints })), Err(EAI_BADFLAGS));
        assert_eq!(lookup(Some(b"127.0.0.1"), Some(b"1"), Some(Hints { socktype: 3, ..hints })), Err(EAI_SOCKTYPE));
        assert_eq!(lookup(Some(b"invalid-numeric-address"), Some(b"1"), Some(hints)), Err(EAI_NONAME));
        // Offline: a host name never resolves.
        assert_eq!(lookup(Some(b"example.com"), Some(b"80"), None), Err(EAI_AGAIN));
    }

    #[test]
    fn resolver_guest_calls() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E12_0020_0000;
        vm.map(BASE, 0x2000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let node = put(&ctx, BASE, b"127.0.0.1");
        let service = put(&ctx, BASE + 0x20, b"27910");
        let (hints, res) = (BASE + 0x40, BASE + 0x80);
        let mut h = [0u8; ADDRINFO_SIZE];
        h[0..4].copy_from_slice(&(AI_NUMERICHOST | AI_NUMERICSERV).to_le_bytes());
        h[4..8].copy_from_slice(&AF_INET.to_le_bytes());
        h[8..12].copy_from_slice(&1i32.to_le_bytes());
        ctx.write_bytes(hints, &h);
        args(&mut ctx, &[node, service, hints, res]);
        assert_eq!(cb_getaddrinfo(&mut ctx), 0);
        let first = ctx.read_u64(res).unwrap();
        assert_ne!(first, 0);
        assert_eq!(ctx.read_u32(first + 4), Some(AF_INET as u32));
        assert_eq!(ctx.read_u32(first + 16), Some(16));
        let sa = ctx.read_u64(first + 32).unwrap();
        assert_eq!(ctx.read_bytes(sa, 8).unwrap(), [16, 2, 0x6d, 0x06, 127, 0, 0, 1]);

        // getnameinfo back to text; the flags are the seventh argument.
        ctx.set(Rsp, BASE + 0x800);
        ctx.write_u64(BASE + 0x808, 10);
        let (host, serv) = (BASE + 0x100, BASE + 0x180);
        args(&mut ctx, &[sa, 16, host, 64, serv, 32]);
        assert_eq!(cb_getnameinfo(&mut ctx), 0);
        assert_eq!(ctx.read_cstring(host, 64).as_deref(), Some("127.0.0.1"));
        assert_eq!(ctx.read_cstring(serv, 32).as_deref(), Some("27910"));
        args(&mut ctx, &[sa, 16, host, 1, 0, 0]);
        assert_eq!(cb_getnameinfo(&mut ctx), EAI_OVERFLOW as HleResult);
        args(&mut ctx, &[sa, 16, 0, 64, 0, 0]);
        assert_eq!(cb_getnameinfo(&mut ctx), EAI_FAIL as HleResult);
        args(&mut ctx, &[first]);
        assert_eq!(cb_freeaddrinfo(&mut ctx), 0);
        args(&mut ctx, &[0]);
        assert_eq!(cb_freeaddrinfo(&mut ctx), 0);

        // Bad hints leave *res NULL.
        ctx.write_u64(res, 1);
        h[16..20].copy_from_slice(&16u32.to_le_bytes());
        ctx.write_bytes(hints, &h);
        args(&mut ctx, &[node, service, hints, res]);
        assert_eq!(cb_getaddrinfo(&mut ctx), EAI_BADHINTS as HleResult);
        assert_eq!(ctx.read_u64(res), Some(0));
        args(&mut ctx, &[node, service, 0, 0]);
        assert_eq!(cb_getaddrinfo(&mut ctx), EAI_FAIL as HleResult);

        args(&mut ctx, &[8]);
        let msg = cb_gai_strerror(&mut ctx) as u64;
        assert_eq!(ctx.read_cstring(msg, 64).as_deref(), Some("hostname nor servname provided, or not known"));
        args(&mut ctx, &[99]);
        let msg = cb_gai_strerror(&mut ctx) as u64;
        assert_eq!(ctx.read_cstring(msg, 64).as_deref(), Some("Unknown error"));
    }

    #[test]
    fn services() {
        assert_eq!(find_service(b"www", Some(b"udp")).map(|s| (s.0, s.1, s.2)), Some(("http", 80, "udp")));
        assert_eq!(find_service(b"https", None).map(|s| (s.1, s.2)), Some((443, "tcp")));
        assert!(find_service(b"https", Some(b"xyz")).is_none());
        assert!(find_service(b"nope", None).is_none());

        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E12_0030_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let name = put(&ctx, BASE, b"imap4");
        let proto = put(&ctx, BASE + 0x20, b"tcp");
        args(&mut ctx, &[name, proto]);
        let entry = cb_getservbyname(&mut ctx) as u64;
        assert_ne!(entry, 0);
        assert_eq!(ctx.read_cstring(ctx.read_u64(entry).unwrap(), 32).as_deref(), Some("imap"));
        assert_eq!(ctx.read_u32(entry + 16), Some((143u16.to_be()) as u32));
        assert_eq!(ctx.read_cstring(ctx.read_u64(entry + 24).unwrap(), 32).as_deref(), Some("tcp"));
        let aliases = ctx.read_u64(entry + 8).unwrap();
        let alias = |i: u64| ctx.read_u64(aliases + 8 * i).unwrap();
        assert_eq!(ctx.read_cstring(alias(0), 32).as_deref(), Some("imap2"));
        assert_eq!(ctx.read_cstring(alias(1), 32).as_deref(), Some("imap4"));
        assert_eq!(alias(2), 0);
        let missing = put(&ctx, BASE + 0x40, b"missing");
        args(&mut ctx, &[missing, 0]);
        assert_eq!(cb_getservbyname(&mut ctx), 0);
    }
}
