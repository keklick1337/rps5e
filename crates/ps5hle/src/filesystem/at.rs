// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The working directory and the directory-relative calls: chdir, fchdir,
//! chroot, access, chflags, pathconf, lutimes and openat / mkdirat /
//! unlinkat / renameat / fchmodat.
//!
//! A relative path argument is looked up from its start directory: the
//! working directory (initially "/"), or for an `*at` call with a
//! directory descriptor, the guest path that descriptor was opened by.
//! The `*at` calls run the plain call with that start directory set for
//! their path arguments.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use super::*;
use std::cell::RefCell;

/// FreeBSD AT_FDCWD, AT_SYMLINK_NOFOLLOW, AT_REMOVEDIR.
const AT_FDCWD: i32 = -100;
const AT_SYMLINK_NOFOLLOW: u64 = 0x200;
const AT_REMOVEDIR: u64 = 0x800;
const PATH_MAX: usize = 1024;
const NAME_MAX: usize = 255;

/// The working directory set by chdir/fchdir; empty until then, when a
/// relative path keeps the process's initial lookup (the `/app0` image and
/// the relative mount spellings, see `app0_rel`).
static CWD: Mutex<String> = Mutex::new(String::new());

fn cwd() -> Option<String> {
    let c = CWD.lock().unwrap_or_else(|p| p.into_inner());
    (!c.is_empty()).then(|| c.clone())
}

thread_local! {
    /// (path argument address, start directory) of the running `*at` call.
    static START_DIRS: RefCell<Vec<(u64, String)>> = const { RefCell::new(Vec::new()) };
}

/// `path` with its start directory: a relative path is joined to the start
/// directory of the argument at `addr` (an `*at` descriptor, else the
/// working directory once one was set).
pub(super) fn absolute(addr: u64, path: String) -> String {
    if path.is_empty() || path.starts_with('/') {
        return path;
    }
    let start = START_DIRS.with(|s| s.borrow().iter().find(|(a, _)| *a == addr).map(|(_, d)| d.clone()));
    let Some(base) = start.or_else(cwd) else { return path };
    if base.ends_with('/') { format!("{base}{path}") } else { format!("{base}/{path}") }
}

/// The absolute, normalized directory a guest path names: relative
/// spellings resolve as the initial lookup does (below `/app0`, or the
/// relative mount names).
fn canonical_dir(guest: &str) -> String {
    if guest.starts_with('/') {
        return normalized(guest);
    }
    for mount in ["devlog/app", "download0", "hostapp"] {
        if strip_mount(guest, mount).is_some() {
            return normalized(&format!("/{guest}"));
        }
    }
    normalized(&format!("/app0/{}", app0_rel(guest).unwrap_or(guest)))
}

/// Run `f` with the given start directories for these path arguments.
fn with_start_dirs<R>(dirs: Vec<(u64, String)>, f: impl FnOnce() -> R) -> R {
    let n = dirs.len();
    START_DIRS.with(|s| s.borrow_mut().extend(dirs));
    let r = f();
    START_DIRS.with(|s| {
        let mut v = s.borrow_mut();
        let keep = v.len() - n;
        v.truncate(keep);
    });
    r
}

/// The start directory a `dirfd` gives a relative path (None: the working
/// directory).
fn dirfd_start(fd: i32) -> FsResult<Option<String>> {
    if fd == AT_FDCWD {
        return Ok(None);
    }
    match get_handle(fd) {
        Some(Handle::Dir(d)) => Ok(Some(canonical_dir(&d.lock().unwrap_or_else(|p| p.into_inner()).guest))),
        Some(_) => Err(ENOTDIR),
        None if is_stdio(fd) || crate::net::is_kernel_socket(fd) || crate::sync::kqueue::is_kqueue(fd) => Err(ENOTDIR),
        None => Err(EBADF),
    }
}

/// Raw path checks shared by the calls: NULL is EFAULT, "" ENOENT, too
/// long ENAMETOOLONG. Returns the raw string.
fn raw_path(ctx: &CpuContext, addr: u64) -> FsResult<String> {
    if addr == 0 {
        return Err(EFAULT);
    }
    let raw = read_guest_cstr(ctx, addr, MAX_GUEST_PATH)?;
    if raw.is_empty() {
        return Err(ENOENT);
    }
    if raw.len() >= PATH_MAX || raw.split(|&b| b == b'/').any(|c| c.len() > NAME_MAX) {
        return Err(ENAMETOOLONG);
    }
    Ok(String::from_utf8_lossy(&raw).into_owned())
}

/// Start directories for path arguments `(dirfd, addr)`: only relative
/// paths with a directory descriptor get one.
fn starts(ctx: &CpuContext, args: &[(i32, u64)]) -> FsResult<Vec<(u64, String)>> {
    let mut out = Vec::new();
    for &(fd, addr) in args {
        let raw = raw_path(ctx, addr)?;
        if raw.starts_with('/') {
            continue;
        }
        if let Some(dir) = dirfd_start(fd)? {
            out.push((addr, dir));
        }
    }
    Ok(out)
}

/// An existing host directory for `guest`, or its lookup error.
fn existing_dir(guest: &str) -> FsResult<PathBuf> {
    if let Some((img, rel)) = image_target(guest) {
        return match img.lookup(&rel) {
            Some(id) if img.node(id).is_dir => Ok(img.cache.join(rel)),
            Some(_) => Err(ENOTDIR),
            None => Err(ENOENT),
        };
    }
    let host = resolve_no_copy(guest).ok_or(ENOENT)?;
    match fs::metadata(&host) {
        Ok(m) if m.is_dir() => Ok(host),
        Ok(_) => Err(ENOTDIR),
        Err(e) => Err(host_errno(&e)),
    }
}

/// The normalized absolute form of an absolute guest path.
fn normalized(guest: &str) -> String {
    let parts = normalize_relative(guest);
    format!("/{}", parts.join("/"))
}

fn chdir_core(ctx: &CpuContext, addr: u64) -> FsResult<i64> {
    raw_path(ctx, addr)?;
    let guest = read_guest_path(ctx, addr)?;
    existing_dir(&guest)?;
    *CWD.lock().unwrap_or_else(|p| p.into_inner()) = canonical_dir(&guest);
    Ok(0)
}

fn fchdir_core(fd: i32) -> FsResult<i64> {
    let Some(dir) = dirfd_start(fd)? else { return Err(EBADF) };
    *CWD.lock().unwrap_or_else(|p| p.into_inner()) = dir;
    Ok(0)
}

/// chroot(path): the guest's root is fixed; "/" itself is accepted, any
/// other existing directory is refused (EPERM, as in a jail).
fn chroot_core(ctx: &CpuContext, addr: u64) -> FsResult<i64> {
    raw_path(ctx, addr)?;
    let guest = read_guest_path(ctx, addr)?;
    // The root always exists (no mount serves it).
    if canonical_dir(&guest) == "/" {
        return Ok(0);
    }
    existing_dir(&guest)?;
    log::error!("[fs] chroot('{guest}'): changing the root directory is not supported");
    Err(EPERM)
}

/// access(path, mode): F_OK 0, X_OK 1, W_OK 2, R_OK 4.
pub(super) fn access_core(ctx: &CpuContext, addr: u64, mode: i32) -> FsResult<i64> {
    // The mode is checked before the path (measured on a PS5).
    if mode & !7 != 0 {
        return Err(EINVAL);
    }
    if addr == 0 {
        return Err(EFAULT);
    }
    let raw = read_guest_cstr(ctx, addr, MAX_GUEST_PATH)?;
    if raw.is_empty() {
        return Err(ENOENT);
    }
    let guest = read_guest_path(ctx, addr)?;
    if is_random_device(&guest) {
        return Ok(0);
    }
    if let Some((img, rel)) = image_target(&guest) {
        img.lookup(&rel).ok_or(ENOENT)?;
        return if mode & 2 != 0 { Err(EROFS) } else { Ok(0) };
    }
    let host = resolve_no_copy(&guest).ok_or(ENOENT)?;
    fs::symlink_metadata(&host).map_err(|e| host_errno(&e))?;
    if mode & 2 != 0 && is_read_only(&guest, Some(&host)) {
        return Err(EROFS);
    }
    let c = std::ffi::CString::new(host.as_os_str().as_encoded_bytes()).map_err(|_| EINVAL)?;
    // SAFETY: a NUL-terminated host path.
    if unsafe { libc::access(c.as_ptr(), mode) } != 0 {
        return Err(host_errno(&io::Error::last_os_error()));
    }
    Ok(0)
}

/// The BSD file flags chflags knows (UF_* and SF_*).
const KNOWN_FILE_FLAGS: u64 = 0x0000_9F9F | 0x0037_0000;
const SF_SNAPSHOT: u64 = 0x0020_0000;

/// chflags(path, flags): clearing every flag succeeds; host files keep no
/// BSD flags, so setting one is EOPNOTSUPP (as on a FreeBSD file system
/// without them); SF_SNAPSHOT is EPERM, unknown bits EOPNOTSUPP.
fn chflags_core(ctx: &CpuContext, addr: u64, flags: u64) -> FsResult<i64> {
    raw_path(ctx, addr)?;
    let guest = read_guest_path(ctx, addr)?;
    if let Some((img, rel)) = image_target(&guest) {
        let id = img.lookup(&rel).ok_or(ENOENT)?;
        if has_trailing_slash(&guest) && !img.node(id).is_dir {
            return Err(ENOTDIR);
        }
    } else {
        let host = resolve_no_copy(&guest).ok_or(ENOENT)?;
        // chflags follows a symbolic link (lchflags does not): a dangling
        // link is ENOENT, a trailing slash on a file ENOTDIR.
        let meta = fs::metadata(&host).map_err(|e| host_errno(&e))?;
        if has_trailing_slash(&guest) && !meta.is_dir() {
            return Err(ENOTDIR);
        }
    }
    set_file_flags("chflags", &guest, flags)
}

/// The flag rules chflags and fchflags share.
fn set_file_flags(function: &str, target: &str, flags: u64) -> FsResult<i64> {
    if flags & !KNOWN_FILE_FLAGS != 0 {
        return Err(EOPNOTSUPP);
    }
    if flags & SF_SNAPSHOT != 0 {
        return Err(EPERM);
    }
    if flags != 0 {
        log::warn!("[fs] {function}({target}, {flags:#x}): host files keep no BSD file flags");
        return Err(EOPNOTSUPP);
    }
    Ok(0)
}

/// fchflags(fd, flags): as chflags on an open file or directory; a socket
/// or a pipe is EINVAL, a closed descriptor EBADF.
pub(super) fn fchflags_core(fd: i32, flags: u64) -> FsResult<i64> {
    if crate::net::is_kernel_socket(fd) {
        return Err(EINVAL);
    }
    match get_handle(fd) {
        Some(Handle::Pipe(_)) => Err(EINVAL),
        Some(_) => set_file_flags("fchflags", &format!("descriptor {fd}"), flags),
        None if is_stdio(fd) => set_file_flags("fchflags", &format!("descriptor {fd}"), flags),
        None => Err(EBADF),
    }
}

/// fstatat(fd, path, stat*, flag): stat (lstat with AT_SYMLINK_NOFOLLOW)
/// of a path relative to a directory descriptor.
pub(super) fn fstatat_core(ctx: &CpuContext, fd: i32, addr: u64, stat: u64, flag: u64) -> FsResult<i64> {
    if flag & !AT_SYMLINK_NOFOLLOW != 0 {
        return Err(EINVAL);
    }
    let dirs = starts(ctx, &[(fd, addr)])?;
    with_start_dirs(dirs, || if flag & AT_SYMLINK_NOFOLLOW != 0 { lstat_core(ctx, addr, stat) } else { stat_core(ctx, addr, stat) })
}

/// pathconf(path, name): the limits of the console's file systems.
fn pathconf_core(ctx: &CpuContext, addr: u64, name: i32) -> FsResult<i64> {
    raw_path(ctx, addr)?;
    let guest = read_guest_path(ctx, addr)?;
    if let Some((img, rel)) = image_target(&guest) {
        img.lookup(&rel).ok_or(ENOENT)?;
    } else if !is_random_device(&guest) {
        let host = resolve_no_copy(&guest).ok_or(ENOENT)?;
        fs::metadata(&host).map_err(|e| host_errno(&e))?;
    }
    Ok(match name {
        1 => 32767,               // _PC_LINK_MAX
        4 => NAME_MAX as i64,     // _PC_NAME_MAX
        5 | 18 => PATH_MAX as i64, // _PC_PATH_MAX, _PC_SYMLINK_MAX
        6 => 512,                 // _PC_PIPE_BUF
        7 | 8 => 1,               // _PC_CHOWN_RESTRICTED, _PC_NO_TRUNC
        12 => 64,                 // _PC_FILESIZEBITS
        15 => -1,                 // _PC_REC_MAX_XFER_SIZE
        54 | 55 => 0,             // _PC_PRIO_IO, _PC_SYNC_IO
        2 | 3 | 9 | 10 | 14 | 16 | 17 | 21 | 53 | 59 | 60 | 63 | 64 => {
            log::error!("[fs] pathconf('{guest}', {name}): depends on the console file system");
            return Err(EINVAL);
        }
        _ => return Err(EINVAL),
    })
}

/// lutimes(path, const timeval times[2]): utimes on the link itself.
pub(super) fn lutimes_core(ctx: &CpuContext, addr: u64, times: u64) -> FsResult<i64> {
    let tv = read_timevals(ctx, times)?;
    raw_path(ctx, addr)?;
    let guest = read_guest_path(ctx, addr)?;
    let host = resolve_no_copy(&guest);
    if is_read_only(&guest, host.as_deref()) {
        return Err(EROFS);
    }
    let host = host.ok_or(ENOENT)?;
    fs::symlink_metadata(&host).map_err(|e| host_errno(&e))?;
    let c = std::ffi::CString::new(host.as_os_str().as_encoded_bytes()).map_err(|_| EINVAL)?;
    let p = tv.as_ref().map_or(std::ptr::null(), |t| t.as_ptr());
    // SAFETY: a NUL-terminated path and NULL or two timevals.
    if unsafe { libc::lutimes(c.as_ptr(), p) } != 0 {
        return Err(host_errno(&io::Error::last_os_error()));
    }
    Ok(0)
}

pub(super) fn openat_core(ctx: &CpuContext, fd: i32, addr: u64, flags: u64) -> FsResult<i64> {
    if invalid_open_flags(flags) {
        return Err(EINVAL);
    }
    let dirs = starts(ctx, &[(fd, addr)])?;
    with_start_dirs(dirs, || open_core(ctx, addr, flags))
}

fn mkdirat_core(ctx: &CpuContext, fd: i32, addr: u64) -> FsResult<i64> {
    let dirs = starts(ctx, &[(fd, addr)])?;
    with_start_dirs(dirs, || mkdir_core(ctx, addr))
}

fn unlinkat_core(ctx: &CpuContext, fd: i32, addr: u64, flag: u64) -> FsResult<i64> {
    if flag & !AT_REMOVEDIR != 0 {
        return Err(EINVAL);
    }
    let dirs = starts(ctx, &[(fd, addr)])?;
    with_start_dirs(dirs, || if flag & AT_REMOVEDIR != 0 { rmdir_core(ctx, addr) } else { unlink_core(ctx, addr) })
}

fn renameat_core(ctx: &CpuContext, oldfd: i32, from: u64, newfd: i32, to: u64) -> FsResult<i64> {
    if from == 0 || to == 0 {
        return Err(EFAULT);
    }
    let dirs = starts(ctx, &[(oldfd, from), (newfd, to)])?;
    with_start_dirs(dirs, || rename_core(ctx, from, to))
}

fn fchmodat_core(ctx: &CpuContext, fd: i32, addr: u64, mode: u32, flag: u64) -> FsResult<i64> {
    if flag != 0 && flag != AT_SYMLINK_NOFOLLOW {
        return Err(EINVAL);
    }
    let dirs = starts(ctx, &[(fd, addr)])?;
    with_start_dirs(dirs, || chmod_core(ctx, addr, mode))
}

fn cb_chdir(ctx: &mut CpuContext) -> HleResult {
    let r = chdir_core(ctx, ctx.rdi());
    posix(ctx, r)
}
fn cb_fchdir(ctx: &mut CpuContext) -> HleResult {
    let r = fchdir_core(ctx.rdi() as i32);
    posix(ctx, r)
}
fn cb_chroot(ctx: &mut CpuContext) -> HleResult {
    let r = chroot_core(ctx, ctx.rdi());
    posix(ctx, r)
}
fn cb_access(ctx: &mut CpuContext) -> HleResult {
    let r = access_core(ctx, ctx.rdi(), ctx.rsi() as i32);
    posix(ctx, r)
}
fn cb_chflags(ctx: &mut CpuContext) -> HleResult {
    let r = chflags_core(ctx, ctx.rdi(), ctx.rsi());
    posix(ctx, r)
}
fn cb_fchflags(ctx: &mut CpuContext) -> HleResult {
    let r = fchflags_core(ctx.rdi() as i32, ctx.rsi());
    posix(ctx, r)
}
fn cb_fstatat(ctx: &mut CpuContext) -> HleResult {
    let r = fstatat_core(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.rcx() as u32 as u64);
    posix(ctx, r)
}
fn cb_pathconf(ctx: &mut CpuContext) -> HleResult {
    let r = pathconf_core(ctx, ctx.rdi(), ctx.rsi() as i32);
    posix(ctx, r)
}
fn cb_lutimes(ctx: &mut CpuContext) -> HleResult {
    let r = lutimes_core(ctx, ctx.rdi(), ctx.rsi());
    posix(ctx, r)
}
fn cb_openat(ctx: &mut CpuContext) -> HleResult {
    let r = openat_core(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    posix(ctx, r)
}
fn cb_mkdirat(ctx: &mut CpuContext) -> HleResult {
    let r = mkdirat_core(ctx, ctx.rdi() as i32, ctx.rsi());
    posix(ctx, r)
}
fn cb_unlinkat(ctx: &mut CpuContext) -> HleResult {
    let r = unlinkat_core(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as u32 as u64);
    posix(ctx, r)
}
fn cb_renameat(ctx: &mut CpuContext) -> HleResult {
    let r = renameat_core(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as i32, ctx.rcx());
    posix(ctx, r)
}
fn cb_fchmodat(ctx: &mut CpuContext) -> HleResult {
    let r = fchmodat_core(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as u32, ctx.rcx() as u32 as u64);
    posix(ctx, r)
}

pub(super) const AT_EXPORTS: &[(&str, &str, &str, HleCallback)] = &[
    ("6mMQ1MSPW-Q", "chdir", "libKernel", cb_chdir),
    ("gnyVA6Tj-ak", "fchdir", "libKernel", cb_fchdir),
    ("AKknh8Cx-V8", "chroot", "libKernel", cb_chroot),
    ("8vE6Z6VEYyk", "access", "libKernel", cb_access),
    ("ixrw0h2tWuI", "chflags", "libKernel", cb_chflags),
    ("UJrQCyYpyic", "fchflags", "libKernel", cb_fchflags),
    ("t6haf4s-eE0", "fstatat", "libkernel_sys", cb_fstatat),
    ("oc98+zOImus", "pathconf", "libKernel", cb_pathconf),
    ("PmZfv0lup1c", "lutimes", "libKernel", cb_lutimes),
    ("A8Yk5AhNDck", "openat", "libKernel", cb_openat),
    ("uHkueep1tc8", "mkdirat", "libKernel", cb_mkdirat),
    ("tCFbw1wGrqk", "unlinkat", "libKernel", cb_unlinkat),
    ("3P7i3vxeZ18", "renameat", "libKernel", cb_renameat),
    ("nsBDTu22CBU", "fchmodat", "libKernel", cb_fchmodat),
];

#[cfg(test)]
fn reset_cwd() {
    CWD.lock().unwrap_or_else(|p| p.into_inner()).clear();
}

#[cfg(test)]
mod tests;
