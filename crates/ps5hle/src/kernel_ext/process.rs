// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Process credentials, priorities, limits and configuration calls:
//! getgroups/setgroups, get/setpriority, getsid, setuid/setreuid/setregid,
//! setrlimit, wait, getlogin_r, sysconf, sysctlnametomib, the terminal
//! calls (no descriptor is a terminal), clock_settime, mlockall and
//! munlockall.
//!
//! The guest is root (uid 0, gid 0) in a process without children or a
//! controlling terminal. Requests whose console answer is unknown (another
//! user id, changing a limit other than lowering the open-file one) are
//! logged and refused with EINVAL.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use super::*;
use std::sync::atomic::AtomicI32;

const ENOTTY: i32 = 25;
const EOPNOTSUPP: i32 = 45;
const EADDRNOTAVAIL: i32 = 49;
const ENAMETOOLONG: i32 = 63;
const EBADF: i32 = 9;
const EIO: i32 = 5;

/// Guest page size (sysconf(_SC_PAGESIZE), hw.pagesize).
pub(super) const GUEST_PAGE_SIZE: u64 = 0x4000;

// ---------------------------------------------------------------------------
// Groups, priority, session, user ids
// ---------------------------------------------------------------------------

/// FreeBSD NGROUPS_MAX.
const GROUPS_MAX: i32 = 1023;

/// Supplementary groups; the first is the effective gid (0).
static GROUPS: Mutex<Vec<u32>> = Mutex::new(Vec::new());

fn groups() -> std::sync::MutexGuard<'static, Vec<u32>> {
    let mut g = GROUPS.lock().unwrap_or_else(|e| e.into_inner());
    if g.is_empty() {
        g.push(0);
    }
    g
}

/// getgroups(int size, gid_t list[]): the count with size 0, EINVAL when
/// the list is too small.
fn cb_getgroups(ctx: &mut CpuContext) -> HleResult {
    let (cap, out) = (ctx.rdi() as i32, ctx.rsi());
    let g = groups().clone();
    let count = g.len() as i32;
    if cap == 0 {
        return count as HleResult;
    }
    if cap < count {
        return posix_failure(ctx, EINVAL);
    }
    if out == 0 {
        return posix_failure(ctx, EFAULT);
    }
    for (i, gid) in g.iter().enumerate() {
        if !ctx.write_u32(out + i as u64 * 4, *gid) {
            return posix_failure(ctx, EFAULT);
        }
    }
    count as HleResult
}

/// setgroups(int count, const gid_t list[]): the first entry is the
/// effective gid and must stay 0; count 0 keeps only it.
fn cb_setgroups(ctx: &mut CpuContext) -> HleResult {
    let (count, list) = (ctx.rdi() as i32, ctx.rsi());
    if !(0..=GROUPS_MAX).contains(&count) {
        return posix_failure(ctx, EINVAL);
    }
    if count > 0 && list == 0 {
        return posix_failure(ctx, EFAULT);
    }
    let Some(new): Option<Vec<u32>> = (0..count as u64).map(|i| ctx.read_u32(list + i * 4)).collect() else {
        return posix_failure(ctx, EFAULT);
    };
    if new.first().is_some_and(|&g| g != 0) {
        log::error!("[kernel] setgroups: effective group id {} is not supported", new[0]);
        return posix_failure(ctx, EINVAL);
    }
    let mut g = groups();
    if new.is_empty() {
        g.truncate(1);
    } else {
        *g = new;
    }
    0
}

const PRIO_PROCESS: i32 = 0;
const PRIO_PGRP: i32 = 1;
const PRIO_USER: i32 = 2;
const NICE_MIN: i32 = -20;
const NICE_MAX: i32 = 20;
static NICE: AtomicI32 = AtomicI32::new(0);

/// Check `(which, who)` of get/setpriority; the errno on failure.
fn priority_target(which: i32, who: i32) -> Result<(), i32> {
    if ![PRIO_PROCESS, PRIO_PGRP, PRIO_USER].contains(&which) {
        return Err(EINVAL);
    }
    let me = std::process::id() as i32;
    let found = if which == PRIO_USER { who == 0 || who == REAL_ID as i32 } else { who == 0 || who == me };
    if found { Ok(()) } else { Err(ESRCH) }
}

/// getpriority(which, who): the process nice value.
fn cb_getpriority(ctx: &mut CpuContext) -> HleResult {
    match priority_target(ctx.rdi() as i32, ctx.rsi() as i32) {
        Ok(()) => NICE.load(Ordering::Relaxed) as HleResult,
        Err(e) => posix_failure(ctx, e),
    }
}

/// setpriority(which, who, nice): clamped to -20..=20.
fn cb_setpriority(ctx: &mut CpuContext) -> HleResult {
    if let Err(e) = priority_target(ctx.rdi() as i32, ctx.rsi() as i32) {
        return posix_failure(ctx, e);
    }
    NICE.store((ctx.rdx() as i32).clamp(NICE_MIN, NICE_MAX), Ordering::Relaxed);
    0
}

/// getsid(pid): the process leads its own session.
fn cb_getsid(ctx: &mut CpuContext) -> HleResult {
    let (pid, me) = (ctx.rdi() as i32, std::process::id() as i32);
    if pid == 0 || pid == me { me as HleResult } else { posix_failure(ctx, ESRCH) }
}

const ID_UNCHANGED: u32 = u32::MAX;

fn only_root(ctx: &CpuContext, what: &str, ids: &[u32]) -> HleResult {
    if let Some(id) = ids.iter().find(|&&id| id != 0 && id != ID_UNCHANGED) {
        log::error!("[kernel] {what}: id {id} is not supported (the process stays root)");
        return posix_failure(ctx, EINVAL);
    }
    0
}

/// setgid(gid): only the root group (0).
fn cb_setgid(ctx: &mut CpuContext) -> HleResult {
    let gid = ctx.rdi() as u32;
    if gid == ID_UNCHANGED {
        return only_root(ctx, "setgid", &[1]);
    }
    only_root(ctx, "setgid", &[gid])
}

/// cfsetspeed(struct termios* t, speed_t speed): both speeds (c_ispeed at
/// +36, c_ospeed at +40 of the 44-byte FreeBSD termios).
fn cb_cfsetspeed(ctx: &mut CpuContext) -> HleResult {
    let (t, speed) = (ctx.rdi(), ctx.rsi() as u32);
    if t == 0 || !(ctx.write_u32(t + 36, speed) && ctx.write_u32(t + 40, speed)) {
        return posix_failure(ctx, EFAULT);
    }
    0
}

/// setuid(uid): only root (0).
fn cb_setuid(ctx: &mut CpuContext) -> HleResult {
    let uid = ctx.rdi() as u32;
    if uid == ID_UNCHANGED {
        return only_root(ctx, "setuid", &[1]);
    }
    only_root(ctx, "setuid", &[uid])
}

/// setreuid(ruid, euid): 0 or -1 (unchanged) only.
fn cb_setreuid(ctx: &mut CpuContext) -> HleResult {
    only_root(ctx, "setreuid", &[ctx.rdi() as u32, ctx.rsi() as u32])
}

/// setregid(rgid, egid): 0 or -1 (unchanged) only.
fn cb_setregid(ctx: &mut CpuContext) -> HleResult {
    only_root(ctx, "setregid", &[ctx.rdi() as u32, ctx.rsi() as u32])
}

// ---------------------------------------------------------------------------
// Limits, children, login name
// ---------------------------------------------------------------------------

const RLIMIT_NOFILE: i32 = 8;
const RLIMIT_NPROC: i32 = 7;

/// The open-file limit the guest lowered (cur, max), if it did.
static OPEN_FILE_LIMIT: Mutex<Option<(i64, i64)>> = Mutex::new(None);

/// The guest's own RLIMIT_NOFILE, when it set one.
pub(super) fn open_file_limit() -> Option<(i64, i64)> {
    *OPEN_FILE_LIMIT.lock().unwrap_or_else(|e| e.into_inner())
}

/// setrlimit(resource, const struct rlimit*): negative values are
/// RLIM_INFINITY, cur is capped at max. RLIMIT_NOFILE may be lowered;
/// setting a limit to its current value succeeds; other changes are
/// refused.
fn cb_setrlimit(ctx: &mut CpuContext) -> HleResult {
    let (resource, limit) = (ctx.rdi() as i32, ctx.rsi());
    if limit == 0 {
        return posix_failure(ctx, EFAULT);
    }
    if !(0..RLIM_NLIMITS).contains(&resource) {
        return posix_failure(ctx, EINVAL);
    }
    let (Some(cur), Some(max)) = (ctx.read_u64(limit), ctx.read_u64(limit + 8)) else { return posix_failure(ctx, EFAULT) };
    let unlimited = |v: i64| if v < 0 { RLIM_INFINITY } else { v };
    let max = unlimited(max as i64);
    let cur = unlimited(cur as i64).min(max);
    let Some(current) = current_limit(resource) else { return posix_failure(ctx, EINVAL) };
    if resource == RLIMIT_NOFILE && max <= current.1 {
        *OPEN_FILE_LIMIT.lock().unwrap_or_else(|e| e.into_inner()) = Some((cur, max));
        return 0;
    }
    if (cur, max) != current || resource == RLIMIT_NPROC {
        log::error!("[kernel] setrlimit: changing resource {resource} to ({cur}, {max}) is not supported");
        return posix_failure(ctx, EINVAL);
    }
    0
}

/// (cur, max) as getrlimit reports them.
pub(super) fn current_limit(resource: i32) -> Option<(i64, i64)> {
    // Measured on a PS5: RLIMIT_STACK, RSS, MEMLOCK and NPROC.
    match resource {
        3 => return Some((0x20_0000, 0x20_0000)),
        5 | 6 => return Some((super::RLIM_INFINITY, super::RLIM_INFINITY)),
        RLIMIT_NPROC => return Some((1604, 1604)),
        _ => {}
    }
    if resource == RLIMIT_NOFILE {
        if let Some(l) = open_file_limit() {
            return Some(l);
        }
    }
    match host_resource(resource) {
        Some(r) => {
            let mut l = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
            // SAFETY: plain query into a local.
            if unsafe { libc::getrlimit(r, &mut l) } != 0 {
                return None;
            }
            Some((guest_limit(l.rlim_cur), guest_limit(l.rlim_max)))
        }
        None => Some((RLIM_INFINITY, RLIM_INFINITY)),
    }
}

/// wait(status*): there are never children.
fn cb_wait(ctx: &mut CpuContext) -> HleResult {
    posix_failure(ctx, ECHILD)
}

/// getlogin_r(char* name, int size) -> 0 or an error number: the host
/// user's account name.
fn cb_getlogin_r(ctx: &mut CpuContext) -> HleResult {
    let (out, size) = (ctx.rdi(), ctx.rsi() as i32);
    if size <= 0 {
        return ERANGE as HleResult;
    }
    if out == 0 {
        return EFAULT as HleResult;
    }
    let Some(name) = host_login_name() else {
        log::error!("[kernel] getlogin_r: the host user has no account name");
        return EIO as HleResult;
    };
    let mut bytes = name.into_bytes();
    bytes.push(0);
    if bytes.len() > size as usize {
        return ERANGE as HleResult;
    }
    if ctx.write_bytes(out, &bytes) { 0 } else { EFAULT as HleResult }
}

/// getlogin() -> char*: the login name in a static 33-byte buffer (MAXLOGNAME),
/// NULL with errno when getlogin_r fails (ERANGE for a longer name) or the
/// name is empty.
fn cb_getlogin(ctx: &mut CpuContext) -> HleResult {
    static BUFFER: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    const MAXLOGNAME: usize = 33;
    let Some(name) = host_login_name() else {
        log::error!("[kernel] getlogin: the host user has no account name");
        posix_failure(ctx, EIO);
        return 0;
    };
    getlogin_into(ctx, &BUFFER, MAXLOGNAME, &name)
}

fn getlogin_into(ctx: &mut CpuContext, buffer: &std::sync::OnceLock<u64>, size: usize, name: &str) -> HleResult {
    if name.len() + 1 > size {
        posix_failure(ctx, ERANGE);
        return 0;
    }
    if name.is_empty() {
        return 0;
    }
    let addr = *buffer.get_or_init(|| crate::pthread::alloc_guest_block(ctx, size as u64).unwrap_or(0));
    let mut bytes = name.as_bytes().to_vec();
    bytes.push(0);
    if addr == 0 || !ctx.write_bytes(addr, &bytes) {
        posix_failure(ctx, ENOMEM);
        return 0;
    }
    addr as HleResult
}

#[cfg(windows)]
fn host_login_name() -> Option<String> {
    std::env::var("USERNAME").ok().filter(|n| !n.is_empty())
}

#[cfg(unix)]
fn host_login_name() -> Option<String> {
    let mut storage = vec![0u8; 16384];
    // SAFETY: getpwuid_r fills `entry` with pointers into `storage`.
    unsafe {
        let mut entry: libc::passwd = std::mem::zeroed();
        let mut found: *mut libc::passwd = std::ptr::null_mut();
        let r = libc::getpwuid_r(libc::geteuid(), &mut entry, storage.as_mut_ptr() as *mut libc::c_char, storage.len(), &mut found);
        if r != 0 || found.is_null() || entry.pw_name.is_null() {
            return None;
        }
        let name = std::ffi::CStr::from_ptr(entry.pw_name).to_string_lossy().into_owned();
        (!name.is_empty()).then_some(name)
    }
}

// ---------------------------------------------------------------------------
// sysconf, sysctlnametomib
// ---------------------------------------------------------------------------

/// Host physical memory in guest pages.
pub(super) fn physical_pages() -> u64 {
    // SAFETY: plain sysconf queries.
    let (pages, size) = unsafe { (libc::sysconf(libc::_SC_PHYS_PAGES), libc::sysconf(libc::_SC_PAGESIZE)) };
    if pages > 0 && size > 0 { pages as u64 * size as u64 / GUEST_PAGE_SIZE } else { 0 }
}

/// sysconf(name): _SC_ARG_MAX 256 KiB, _SC_CLK_TCK 128, _SC_OPEN_MAX the
/// RLIMIT_NOFILE soft limit, _SC_PAGESIZE 16 KiB, _SC_IOV_MAX 1024, the
/// processors a game may use, the host's physical memory in guest pages;
/// any other name is EINVAL. errno is untouched on success.
fn cb_sysconf(ctx: &mut CpuContext) -> HleResult {
    let value: i64 = match ctx.rdi() as i32 {
        1 => 262_144,
        3 => 128,
        5 => current_limit(RLIMIT_NOFILE).map_or(0, |l| l.0),
        56 => 1024,
        47 => GUEST_PAGE_SIZE as i64,
        57 | 58 => GUEST_CPU_COUNT as i64,
        121 => physical_pages() as i64,
        _ => return posix_failure(ctx, EINVAL),
    };
    if value <= 0 {
        return posix_failure(ctx, EIO);
    }
    value
}

/// getdtablesize(): the RLIMIT_NOFILE soft limit, capped at the host's
/// descriptor table size; never fails.
fn cb_getdtablesize(_ctx: &mut CpuContext) -> HleResult {
    // SAFETY: plain query.
    let host = unsafe { libc::getdtablesize() } as i64;
    let soft = current_limit(RLIMIT_NOFILE).map_or(host, |l| l.0);
    soft.min(host).clamp(0, i32::MAX as i64)
}

/// Names sysctl knows and their MIBs.
pub(super) const KNOWN_SYSCTL_NAMES: &[(&str, &[u32])] = &[
    ("hw.machine", &[6, 1]),
    ("kern.hostname", &[1, 10]),
    ("kern.osrelease", &[1, 2]),
    ("kern.ostype", &[1, 1]),
    ("kern.version", &[1, 4]),
    ("hw.model", &[6, 2]),
    ("hw.ncpu", &[6, 3]),
    ("hw.pagesize", &[6, 7]),
    ("hw.physmem", &[6, 5]),
    ("hw.realmem", &[6, 12]),
    ("kern.arandom", &[1, 37]),
    ("kern.proc.pathname", &[1, 14, 12]),
    ("kern.osreldate", &[1, 24]),
    ("kern.rng_pseudo", &[1, 50]),
    // Names whose MIB the console gives although sysctl answers ENOENT.
    ("hw.byteorder", &[6, 4]),
    ("hw.machine_arch", &[6, 11]),
    ("kern.clockrate", &[1, 12]),
    ("kern.maxfiles", &[1, 7]),
    ("kern.maxfilesperproc", &[1, 27]),
    ("kern.maxproc", &[1, 6]),
    ("kern.ps_strings", &[1, 32]),
];

/// Longest sysctl name the name-based calls take: 1024 bytes or more is
/// ENAMETOOLONG (measured on a PS5, firmware 13.60).
pub(crate) const MAX_SYSCTL_NAME: usize = 1024;

/// FreeBSD's top-level sysctl nodes, numbered from CTL_SYSCTL (0) to
/// CTL_P1003_1B (9).
const TOP_LEVEL_SYSCTL_NODES: [&str; 10] = ["sysctl", "kern", "vm", "vfs", "net", "debug", "hw", "machdep", "user", "p1003_1b"];

/// The MIB number of a top-level node name (`kern` is 1).
pub(crate) fn top_level_sysctl_node(name: &str) -> Option<u32> {
    TOP_LEVEL_SYSCTL_NODES.iter().position(|n| *n == name).map(|i| i as u32)
}

/// A sysctl name with one trailing dot dropped: `hw.ncpu.` names `hw.ncpu`
/// on the console (firmware 13.60), `kern..` stays unknown.
pub(crate) fn sysctl_name_without_trailing_dot(name: &str) -> &str {
    name.strip_suffix('.').unwrap_or(name)
}

/// sysctlnametomib(name, int* mib, size_t* size): a top-level node name
/// gives its one-level MIB.
fn cb_sysctlnametomib(ctx: &mut CpuContext) -> HleResult {
    let (name, mib, size) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if name == 0 || size == 0 {
        log::error!("[kernel] sysctlnametomib: null name or size");
        return posix_failure(ctx, EFAULT);
    }
    let Some(text) = ctx.read_cstring(name, MAX_SYSCTL_NAME + 1) else { return posix_failure(ctx, EFAULT) };
    if text.len() >= MAX_SYSCTL_NAME {
        return posix_failure(ctx, ENAMETOOLONG);
    }
    let text = sysctl_name_without_trailing_dot(&text);
    if text.is_empty() || matches!(text, "machdep.cacheline_size" | "hw.cacheline_size" | "hw.usermem") {
        return posix_failure(ctx, ENOENT);
    }
    let top: [u32; 1];
    let entry: &[u32] = if let Some(node) = top_level_sysctl_node(text) {
        top = [node];
        &top
    } else if let Some(&(_, entry)) = KNOWN_SYSCTL_NAMES.iter().find(|(n, _)| *n == text) {
        entry
    } else {
        log::warn!("[kernel] sysctlnametomib: unknown name '{text}'");
        return posix_failure(ctx, ENOENT);
    };
    let Some(room) = ctx.read_u64(size) else { return posix_failure(ctx, EFAULT) };
    let bytes: Vec<u8> = entry.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut len_bytes = room.saturating_mul(4);
    let e = sysctl_copy_out_len(ctx, &bytes, mib, &mut len_bytes);
    if !ctx.write_u64(size, len_bytes / 4) {
        return posix_failure(ctx, EFAULT);
    }
    if e != 0 { posix_failure(ctx, e) } else { 0 }
}

/// Copy `value` into `old` with room `*len` bytes like sysctl: `*len`
/// becomes the bytes copied (or the full size without a buffer); ENOMEM
/// when cut short.
fn sysctl_copy_out_len(ctx: &CpuContext, value: &[u8], old: u64, len: &mut u64) -> i32 {
    if old == 0 {
        *len = value.len() as u64;
        return 0;
    }
    let n = (*len as usize).min(value.len());
    if !ctx.write_bytes(old, &value[..n]) {
        return EFAULT;
    }
    *len = n as u64;
    if n < value.len() { ENOMEM } else { 0 }
}

// ---------------------------------------------------------------------------
// Terminals
// ---------------------------------------------------------------------------

/// No descriptor is a terminal: sockets report what FreeBSD's socket ioctl
/// does (EADDRNOTAVAIL for inet, EOPNOTSUPP otherwise), closed descriptors
/// EBADF, everything else ENOTTY.
fn terminal_error(fd: i32) -> i32 {
    if let Some(family) = crate::net::kernel_socket_family(fd) {
        return if family == 2 { EADDRNOTAVAIL } else { EOPNOTSUPP };
    }
    if fd < 0 || !crate::filesystem::descriptor_open(fd) {
        return EBADF;
    }
    ENOTTY
}

/// The process-group calls on a pipe report its SIGIO owner, which is not
/// modelled.
fn process_group_error(fd: i32, what: &str) -> i32 {
    let e = terminal_error(fd);
    if e == ENOTTY && fd >= 3 && crate::filesystem::pipe::is_pipe(fd) {
        log::error!("[kernel] {what}: the SIGIO owner of a pipe is not supported");
        return EINVAL;
    }
    e
}

/// tcgetattr(fd, termios*)
fn cb_tcgetattr(ctx: &mut CpuContext) -> HleResult {
    posix_failure(ctx, terminal_error(ctx.rdi() as i32))
}

/// tcsetattr(fd, action, const termios*)
fn cb_tcsetattr(ctx: &mut CpuContext) -> HleResult {
    const TCSASOFT: i32 = 0x10;
    let (fd, action, attrs) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx());
    if action & TCSASOFT != 0 && attrs == 0 {
        log::error!("[kernel] tcsetattr: null attributes with TCSASOFT");
        return posix_failure(ctx, EFAULT);
    }
    if !(0..=2).contains(&(action & !TCSASOFT)) {
        return posix_failure(ctx, EINVAL);
    }
    if attrs == 0 {
        return posix_failure(ctx, EFAULT);
    }
    posix_failure(ctx, terminal_error(fd))
}

/// tcsendbreak(fd, duration), tcdrain(fd)
fn cb_tc_terminal_only(ctx: &mut CpuContext) -> HleResult {
    posix_failure(ctx, terminal_error(ctx.rdi() as i32))
}

/// tcflush(fd, queue): TCIFLUSH 1, TCOFLUSH 2, TCIOFLUSH 3.
fn cb_tcflush(ctx: &mut CpuContext) -> HleResult {
    if !(1..=3).contains(&(ctx.rsi() as i32)) {
        return posix_failure(ctx, EINVAL);
    }
    posix_failure(ctx, terminal_error(ctx.rdi() as i32))
}

/// tcflow(fd, action): TCOOFF 1 .. TCION 4.
fn cb_tcflow(ctx: &mut CpuContext) -> HleResult {
    if !(1..=4).contains(&(ctx.rsi() as i32)) {
        return posix_failure(ctx, EINVAL);
    }
    posix_failure(ctx, terminal_error(ctx.rdi() as i32))
}

/// tcgetpgrp(fd)
fn cb_tcgetpgrp(ctx: &mut CpuContext) -> HleResult {
    posix_failure(ctx, process_group_error(ctx.rdi() as i32, "tcgetpgrp"))
}

/// tcsetpgrp(fd, pgrp)
fn cb_tcsetpgrp(ctx: &mut CpuContext) -> HleResult {
    posix_failure(ctx, process_group_error(ctx.rdi() as i32, "tcsetpgrp"))
}

// ---------------------------------------------------------------------------
// Clock, memory locking
// ---------------------------------------------------------------------------

/// clock_settime(clock, const timespec*): only CLOCK_REALTIME with a valid
/// time is accepted for checking; a sandboxed game may not set the clock
/// (EPERM), and the host clock is never changed.
fn cb_clock_settime(ctx: &mut CpuContext) -> HleResult {
    let (clock, tp) = (ctx.rdi() as i32, ctx.rsi());
    if tp == 0 {
        return posix_failure(ctx, EFAULT);
    }
    let (Some(sec), Some(nsec)) = (ctx.read_u64(tp), ctx.read_u64(tp + 8)) else { return posix_failure(ctx, EFAULT) };
    if clock != 0 || (sec as i64) < 0 || !(0..1_000_000_000).contains(&(nsec as i64)) {
        return posix_failure(ctx, EINVAL);
    }
    log::warn!("[kernel] clock_settime(CLOCK_REALTIME, {}.{:09}): refused, the host clock is not changed", sec as i64, nsec);
    posix_failure(ctx, EPERM)
}

const MCL_CURRENT: u64 = 1;
const MCL_FUTURE: u64 = 2;

/// mlockall(flags): guest memory is never paged out, so every mapping is
/// already resident; only the flags are checked.
fn cb_mlockall(ctx: &mut CpuContext) -> HleResult {
    let flags = ctx.rdi() as u32 as u64;
    if flags == 0 || flags & !(MCL_CURRENT | MCL_FUTURE) != 0 {
        return posix_failure(ctx, EINVAL);
    }
    0
}

/// munlockall() / sceKernelMunlockall(): nothing is ever pinned.
fn cb_munlockall(_ctx: &mut CpuContext) -> HleResult {
    OK
}

pub(super) const PROCESS_EXPORTS: &[(&str, &str, &str, HleCallback)] = &[
    ("3yjZbSvan2g", "getgroups", LIB_KERNEL, cb_getgroups),
    ("M8VZ3iIlmyg", "setgroups", LIB_KERNEL, cb_setgroups),
    ("miQ1wVwSexE", "getpriority", LIB_KERNEL, cb_getpriority),
    ("TUC9xC1YQjs", "setpriority", LIB_KERNEL, cb_setpriority),
    ("gokWod7GAH4", "getsid", LIB_KERNEL, cb_getsid),
    ("JVmUZwK-HJU", "setuid", LIB_KERNEL, cb_setuid),
    ("fqFN+ydTbT0", "setgid", LIB_KERNEL, cb_setgid),
    ("C1MQ6OaXRjo", "cfsetspeed", LIB_KERNEL, cb_cfsetspeed),
    ("6w8tPp+Yk6E", "setreuid", LIB_KERNEL, cb_setreuid),
    ("ROILLZdYZPc", "setregid", LIB_KERNEL, cb_setregid),
    ("4X0QwvuCfjc", "setrlimit", LIB_KERNEL, cb_setrlimit),
    ("ldVb3lc75PE", "wait", LIB_KERNEL, cb_wait),
    ("78FiHequa9U", "getlogin_r", LIB_KERNEL, cb_getlogin_r),
    ("AfMxh2e9V3Q", "getlogin", LIB_KERNEL, cb_getlogin),
    ("mkawd0NA9ts", "sysconf", LIB_KERNEL, cb_sysconf),
    ("sZuwaDPATKs", "getdtablesize", LIB_KERNEL, cb_getdtablesize),
    ("UkEV4hvMo1E", "sysctlnametomib", LIB_KERNEL, cb_sysctlnametomib),
    ("5kBfZ40bAuk", "tcgetattr", LIB_KERNEL, cb_tcgetattr),
    ("La7xW-jcZwQ", "tcsetattr", LIB_KERNEL, cb_tcsetattr),
    ("vXE-dIUIZlE", "tcsendbreak", LIB_KERNEL, cb_tc_terminal_only),
    ("brGGJWsZauw", "tcdrain", LIB_KERNEL, cb_tc_terminal_only),
    ("FxHW5NuEXsY", "tcflush", LIB_KERNEL, cb_tcflush),
    ("OG8xRaGIpqk", "tcflow", LIB_KERNEL, cb_tcflow),
    ("5OEl1mRhUm0", "tcgetpgrp", LIB_KERNEL, cb_tcgetpgrp),
    ("YMuesZTyTl4", "tcsetpgrp", LIB_KERNEL, cb_tcsetpgrp),
    ("d7nUj1LOdDU", "clock_settime", LIB_KERNEL, cb_clock_settime),
    ("x7g7Ebeo8-U", "mlockall", LIB_KERNEL, cb_mlockall),
    ("NpLBpgVV7PU", "munlockall", LIB_KERNEL, cb_munlockall),
    ("lvD71jP2Mnk", "sceKernelMunlockall", LIB_KERNEL, cb_munlockall),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister;
    use ps5vm::{MemProtect, VirtualMemory};

    const BASE: u64 = 0x7E60_0000_0000;

    fn call(ctx: &mut CpuContext, cb: HleCallback, args: &[u64]) -> HleResult {
        let regs = [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx];
        for (r, v) in regs.iter().zip(args.iter().chain(std::iter::repeat(&0))) {
            ctx.set(*r, *v);
        }
        cb(ctx)
    }

    /// One thread with errno; `f(ctx, errno address, base)`.
    fn with_ctx(offset: u64, f: impl FnOnce(&mut CpuContext, u64, u64) + Send + 'static) {
        std::thread::spawn(move || {
            let base = BASE + offset;
            let vm: &'static VirtualMemory = Box::leak(Box::new(VirtualMemory::new()));
            vm.map(base, 0x10000, MemProtect::RW).unwrap();
            let mut ctx = CpuContext::new(vm);
            let errno = crate::kernel::bind_test_errno(&ctx);
            f(&mut ctx, errno, base);
        })
        .join()
        .unwrap();
    }

    fn errno(ctx: &CpuContext, at: u64) -> i32 {
        ctx.read_u32(at).unwrap() as i32
    }

    /// getrlimit values measured on a PS5; setgid like setuid; cfsetspeed.
    #[test]
    fn limits_setgid_and_speed() {
        with_ctx(0x40000, |ctx, e, base| {
            for (resource, value) in [(3u64, 0x20_0000u64), (5, i64::MAX as u64), (6, i64::MAX as u64), (7, 1604)] {
                assert_eq!(call(ctx, super::super::cb_getrlimit, &[resource, base]), 0);
                assert_eq!((ctx.read_u64(base), ctx.read_u64(base + 8)), (Some(value), Some(value)), "resource {resource}");
            }
            // The same stack limit may be set again; NPROC never.
            ctx.write_u64(base, 0x20_0000);
            ctx.write_u64(base + 8, 0x20_0000);
            assert_eq!(call(ctx, cb_setrlimit, &[3, base]), 0);
            ctx.write_u64(base, 1604);
            ctx.write_u64(base + 8, 1604);
            assert_eq!(call(ctx, cb_setrlimit, &[7, base]), -1);
            assert_eq!(call(ctx, cb_setgid, &[0]), 0);
            assert_eq!(call(ctx, cb_setgid, &[5]), -1);
            assert_eq!(errno(ctx, e), EINVAL);
            assert_eq!(call(ctx, cb_setgid, &[u32::MAX as u64]), -1);
            ctx.write_bytes(base + 0x100, &[0xAA; 48]);
            assert_eq!(call(ctx, cb_cfsetspeed, &[base + 0x100, 9600]), 0);
            assert_eq!((ctx.read_u32(base + 0x100 + 36), ctx.read_u32(base + 0x100 + 40)), (Some(9600), Some(9600)));
            assert_eq!(ctx.read_u32(base + 0x100 + 44), Some(0xAAAA_AAAA));
            assert_eq!(call(ctx, cb_cfsetspeed, &[0, 9600]), -1);
            assert_eq!(errno(ctx, e), EFAULT);
        });
    }

    #[test]
    fn groups_priority_session_and_ids() {
        with_ctx(0, |ctx, e, base| {
            assert_eq!(call(ctx, cb_getgroups, &[0, 0]), 1);
            assert_eq!(call(ctx, cb_getgroups, &[4, 0]), -1);
            assert_eq!(errno(ctx, e), EFAULT);
            assert_eq!(call(ctx, cb_getgroups, &[4, base]), 1);
            assert_eq!(ctx.read_u32(base), Some(0));
            assert_eq!(call(ctx, cb_setgroups, &[1024, base]), -1);
            assert_eq!(errno(ctx, e), EINVAL);
            assert_eq!(call(ctx, cb_setgroups, &[2, 0]), -1);
            assert_eq!(errno(ctx, e), EFAULT);
            ctx.write_u32(base + 0x100, 0);
            ctx.write_u32(base + 0x104, 5);
            assert_eq!(call(ctx, cb_setgroups, &[2, base + 0x100]), 0);
            assert_eq!(call(ctx, cb_getgroups, &[1, base]), -1);
            assert_eq!(errno(ctx, e), EINVAL);
            assert_eq!(call(ctx, cb_getgroups, &[2, base]), 2);
            assert_eq!(ctx.read_u32(base + 4), Some(5));
            ctx.write_u32(base + 0x100, 7);
            assert_eq!(call(ctx, cb_setgroups, &[1, base + 0x100]), -1);
            assert_eq!(call(ctx, cb_setgroups, &[0, 0]), 0);
            assert_eq!(call(ctx, cb_getgroups, &[0, 0]), 1);

            assert_eq!(call(ctx, cb_getpriority, &[3, 0]), -1);
            assert_eq!(errno(ctx, e), EINVAL);
            assert_eq!(call(ctx, cb_getpriority, &[PRIO_PROCESS as u64, 1]), -1);
            assert_eq!(errno(ctx, e), ESRCH);
            assert_eq!(call(ctx, cb_setpriority, &[PRIO_PROCESS as u64, 0, 50]), 0);
            assert_eq!(call(ctx, cb_getpriority, &[PRIO_USER as u64, 0]), NICE_MAX as HleResult);
            assert_eq!(call(ctx, cb_setpriority, &[PRIO_PGRP as u64, 0, (-3i64) as u64]), 0);
            assert_eq!(call(ctx, cb_getpriority, &[PRIO_PROCESS as u64, std::process::id() as u64]), -3);
            assert_eq!(call(ctx, cb_setpriority, &[PRIO_PROCESS as u64, 0, 0]), 0);

            let me = std::process::id() as HleResult;
            assert_eq!(call(ctx, cb_getsid, &[0]), me);
            assert_eq!(call(ctx, cb_getsid, &[me as u64]), me);
            assert_eq!(call(ctx, cb_getsid, &[1]), -1);
            assert_eq!(errno(ctx, e), ESRCH);

            assert_eq!(call(ctx, cb_setuid, &[0]), 0);
            assert_eq!(call(ctx, cb_setuid, &[5]), -1);
            assert_eq!(call(ctx, cb_setreuid, &[u32::MAX as u64, 0]), 0);
            assert_eq!(call(ctx, cb_setregid, &[0, 3]), -1);
            assert_eq!(call(ctx, cb_wait, &[0]), -1);
            assert_eq!(errno(ctx, e), ECHILD);
        });
    }

    #[test]
    fn limits_login_and_configuration() {
        with_ctx(0x10_0000, |ctx, e, base| {
            assert_eq!(call(ctx, cb_setrlimit, &[RLIMIT_NOFILE as u64, 0]), -1);
            assert_eq!(errno(ctx, e), EFAULT);
            assert_eq!(call(ctx, cb_setrlimit, &[15, base]), -1);
            assert_eq!(errno(ctx, e), EINVAL);
            let (cur, max) = current_limit(RLIMIT_NOFILE).unwrap();
            ctx.write_u64(base, 64);
            ctx.write_u64(base + 8, max as u64);
            assert_eq!(call(ctx, cb_setrlimit, &[RLIMIT_NOFILE as u64, base]), 0);
            assert_eq!(current_limit(RLIMIT_NOFILE), Some((64, max)));
            assert_eq!(call(ctx, cb_sysconf, &[5]), 64, "_SC_OPEN_MAX is the soft limit");
            assert_eq!(call(ctx, cb_getdtablesize, &[]), 64);
            ctx.write_u64(base, cur as u64);
            assert_eq!(call(ctx, cb_setrlimit, &[RLIMIT_NOFILE as u64, base]), 0);
            // The current value of another resource is accepted.
            let (cur, max) = current_limit(0).unwrap();
            ctx.write_u64(base, cur as u64);
            ctx.write_u64(base + 8, max as u64);
            assert_eq!(call(ctx, cb_setrlimit, &[0, base]), 0);

            assert_eq!(call(ctx, cb_getlogin_r, &[base, 0]), ERANGE as HleResult);
            assert_eq!(call(ctx, cb_getlogin_r, &[0, 64]), EFAULT as HleResult);
            assert_eq!(call(ctx, cb_getlogin_r, &[base, 1]), ERANGE as HleResult);
            assert_eq!(call(ctx, cb_getlogin_r, &[base, 256]), 0);
            assert!(!ctx.read_cstring(base, 256).unwrap().is_empty());
            // getlogin: one static buffer, NULL with ERANGE for a name
            // longer than MAXLOGNAME - 1, NULL for an empty one.
            let buffer = std::sync::OnceLock::new();
            let name = getlogin_into(ctx, &buffer, 33, "player") as u64;
            assert_ne!(name, 0);
            assert_eq!(ctx.read_cstring(name, 64).as_deref(), Some("player"));
            assert_eq!(getlogin_into(ctx, &buffer, 33, &"n".repeat(40)), 0);
            assert_eq!(errno(ctx, e), ERANGE);
            assert_eq!(getlogin_into(ctx, &buffer, 33, ""), 0);
            assert_eq!(getlogin_into(ctx, &buffer, 33, "again") as u64, name);
            let host = call(ctx, cb_getlogin, &[]) as u64;
            assert_ne!(host, 0);

            assert_eq!(call(ctx, cb_sysconf, &[3]), 128);
            assert_eq!(call(ctx, cb_sysconf, &[1]), 262_144);
            assert_eq!(call(ctx, cb_sysconf, &[56]), 1024);
            // SAFETY: plain query.
            assert!(call(ctx, cb_getdtablesize, &[]) <= unsafe { libc::getdtablesize() } as HleResult);
            assert_eq!(call(ctx, cb_sysconf, &[47]), 0x4000);
            assert_eq!(call(ctx, cb_sysconf, &[58]), GUEST_CPU_COUNT as HleResult);
            assert!(call(ctx, cb_sysconf, &[121]) > 0);
            assert_eq!(call(ctx, cb_sysconf, &[9999]), -1);
            assert_eq!(errno(ctx, e), EINVAL);

            let (name, mib, size) = (base + 0x100, base + 0x200, base + 0x300);
            ctx.write_bytes(name, b"kern.proc.pathname\0");
            ctx.write_u64(size, 4);
            assert_eq!(call(ctx, cb_sysctlnametomib, &[name, mib, size]), 0);
            assert_eq!(ctx.read_u64(size), Some(3));
            assert_eq!((ctx.read_u32(mib), ctx.read_u32(mib + 4), ctx.read_u32(mib + 8)), (Some(1), Some(14), Some(12)));
            ctx.write_u64(size, 1);
            assert_eq!(call(ctx, cb_sysctlnametomib, &[name, mib, size]), -1);
            assert_eq!(errno(ctx, e), ENOMEM);
            assert_eq!(ctx.read_u64(size), Some(1));
            ctx.write_bytes(name, b"machdep.cacheline_size\0");
            assert_eq!(call(ctx, cb_sysctlnametomib, &[name, mib, size]), -1);
            assert_eq!(errno(ctx, e), ENOENT);
            ctx.write_bytes(name, b"\0");
            assert_eq!(call(ctx, cb_sysctlnametomib, &[name, mib, size]), -1);
            assert_eq!(errno(ctx, e), ENOENT);
            // Top-level names, with or without a trailing dot, are one
            // level; one trailing dot is dropped from any name.
            for (i, node) in ["sysctl", "kern", "vm", "vfs", "net", "debug", "hw", "machdep", "user", "p1003_1b"].iter().enumerate() {
                for dotted in [false, true] {
                    let text = format!("{node}{}\0", if dotted { "." } else { "" });
                    ctx.write_bytes(name, text.as_bytes());
                    ctx.write_u32(mib + 4, 0x5a5a_5a5a);
                    ctx.write_u64(size, 3);
                    assert_eq!(call(ctx, cb_sysctlnametomib, &[name, mib, size]), 0, "{text}");
                    assert_eq!((ctx.read_u64(size), ctx.read_u32(mib), ctx.read_u32(mib + 4)), (Some(1), Some(i as u32), Some(0x5a5a_5a5a)));
                }
            }
            ctx.write_bytes(name, b"hw.ncpu.\0");
            ctx.write_u64(size, 3);
            assert_eq!(call(ctx, cb_sysctlnametomib, &[name, mib, size]), 0);
            assert_eq!((ctx.read_u64(size), ctx.read_u32(mib), ctx.read_u32(mib + 4)), (Some(2), Some(6), Some(3)));
            for bad in [&b"kern..\0"[..], b".\0", b"kernel\0"] {
                ctx.write_bytes(name, bad);
                ctx.write_u64(size, 3);
                assert_eq!(call(ctx, cb_sysctlnametomib, &[name, mib, size]), -1);
                assert_eq!(errno(ctx, e), ENOENT);
            }
        });
    }

    #[test]
    fn terminals_clock_and_locking() {
        with_ctx(0x20_0000, |ctx, e, base| {
            // stdout is open but not a terminal.
            assert_eq!(call(ctx, cb_tcgetattr, &[1, base]), -1);
            assert_eq!(errno(ctx, e), ENOTTY);
            assert_eq!(call(ctx, cb_tcgetattr, &[0x0FFF_FFF0, base]), -1);
            assert_eq!(errno(ctx, e), EBADF);
            assert_eq!(call(ctx, cb_tcsetattr, &[1, 7, base]), -1);
            assert_eq!(errno(ctx, e), EINVAL);
            assert_eq!(call(ctx, cb_tcsetattr, &[1, 0, 0]), -1);
            assert_eq!(errno(ctx, e), EFAULT);
            assert_eq!(call(ctx, cb_tcflush, &[1, 0]), -1);
            assert_eq!(errno(ctx, e), EINVAL);
            assert_eq!(call(ctx, cb_tcflow, &[1, 5]), -1);
            assert_eq!(errno(ctx, e), EINVAL);
            assert_eq!(call(ctx, cb_tcflush, &[1, 3]), -1);
            assert_eq!(errno(ctx, e), ENOTTY);
            assert_eq!(call(ctx, cb_tc_terminal_only, &[2]), -1);
            assert_eq!(errno(ctx, e), ENOTTY);
            assert_eq!(call(ctx, cb_tcgetpgrp, &[1]), -1);
            assert_eq!(errno(ctx, e), ENOTTY);
            let (a, b) = crate::net::test_socketpair(ctx, 1, base + 0x80);
            assert_eq!(call(ctx, cb_tcgetattr, &[a as u64, base]), -1);
            assert_eq!(errno(ctx, e), EOPNOTSUPP);
            assert!(crate::net::kernel_socket_close(a) && crate::net::kernel_socket_close(b));

            assert_eq!(call(ctx, cb_clock_settime, &[0, 0]), -1);
            assert_eq!(errno(ctx, e), EFAULT);
            ctx.write_u64(base, 10);
            ctx.write_u64(base + 8, 1_000_000_000);
            assert_eq!(call(ctx, cb_clock_settime, &[0, base]), -1);
            assert_eq!(errno(ctx, e), EINVAL);
            ctx.write_u64(base + 8, 0);
            assert_eq!(call(ctx, cb_clock_settime, &[4, base]), -1);
            assert_eq!(errno(ctx, e), EINVAL);
            assert_eq!(call(ctx, cb_clock_settime, &[0, base]), -1);
            assert_eq!(errno(ctx, e), EPERM);

            assert_eq!(call(ctx, cb_mlockall, &[0]), -1);
            assert_eq!(errno(ctx, e), EINVAL);
            assert_eq!(call(ctx, cb_mlockall, &[4]), -1);
            assert_eq!(call(ctx, cb_mlockall, &[MCL_CURRENT | MCL_FUTURE]), 0);
            assert_eq!(call(ctx, cb_munlockall, &[]), 0);
        });
    }

    /// sysctlbyname / sysctl names beyond hw.ncpu.
    #[test]
    fn sysctl_names() {
        with_ctx(0x30_0000, |ctx, e, base| {
            let (name, old, len, mib) = (base, base + 0x100, base + 0x400, base + 0x500);
            let by_name = |ctx: &mut CpuContext, n: &[u8], old: u64, room: u64, new: u64| {
                ctx.write_bytes(name, n);
                ctx.write_u64(len, room);
                call(ctx, super::super::cb_sysctlbyname, &[name, old, len, new])
            };
            assert_eq!(by_name(ctx, b"hw.pagesize\0", old, 4, 0), 0);
            assert_eq!(ctx.read_u32(old), Some(0x4000));
            assert_eq!(by_name(ctx, b"hw.physmem\0", old, 8, 0), 0);
            assert!(ctx.read_u64(old).unwrap() >= 0x4000);
            assert_eq!(by_name(ctx, b"hw.model\0", 0, 0, 0), 0);
            assert!(ctx.read_u64(len).unwrap() > 1);
            assert_eq!(by_name(ctx, b"kern.proc.pathname\0", old, 64, 0), -1);
            assert_eq!(errno(ctx, e), EINVAL);
            assert_eq!(by_name(ctx, b"kern.arandom\0", old, 1000, 0), 0);
            assert_eq!(ctx.read_u64(len), Some(256));
            assert_eq!(by_name(ctx, b"kern.sdk_version\0", old, 4, 0), 0);
            assert_eq!(ctx.read_u32(old), Some(0x0100_0000));
            assert_eq!(by_name(ctx, b"machdep.tsc_freq\0", old, 8, 0), 0);
            assert!(ctx.read_u64(old).unwrap() > 0);
            assert_eq!(by_name(ctx, b"vm.overcommit\0", old, 4, base + 0x800), -1);
            assert_eq!(errno(ctx, e), EPERM);
            assert_eq!(by_name(ctx, b"machdep.cacheline_size\0", old, 4, 0), -1);
            assert_eq!(errno(ctx, e), ENOENT);
            // kern.proc.pathname with the pid (-1: this process).
            for (i, v) in [1u32, 14, 12, u32::MAX].iter().enumerate() {
                ctx.write_u32(mib + i as u64 * 4, *v);
            }
            ctx.write_u64(len, 64);
            assert_eq!(call(ctx, super::super::cb_sysctl, &[mib, 4, old, len, 0]), 0);
            assert_eq!(ctx.read_cstring(old, 64).as_deref(), Some("/app0/eboot.bin"));
            ctx.write_u32(mib + 12, 1);
            assert_eq!(call(ctx, super::super::cb_sysctl, &[mib, 4, old, len, 0]), -1);
            assert_eq!(errno(ctx, e), ESRCH);
            // Under a leaf: ENOTDIR.
            for (i, v) in [6u32, 7, 1].iter().enumerate() {
                ctx.write_u32(mib + i as u64 * 4, *v);
            }
            assert_eq!(call(ctx, super::super::cb_sysctl, &[mib, 3, old, len, 0]), -1);
            assert_eq!(errno(ctx, e), 20);
        });
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _, _) in PROCESS_EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }
}
