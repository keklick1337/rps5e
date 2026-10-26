// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Kernel file system HLE.
//!
//! Guest paths (`/app0/...`, `/download0/...`, `/temp0/...`, registered
//! mounts such as `/savedata0`) are translated to host paths, contained
//! inside their mount roots, and served through a guest descriptor table.
//! Covers the POSIX and `sceKernel*` file calls, directory enumeration,
//! stat, AIO (completed synchronously) and the APR file-id services.
//!
//! Error convention: `sceKernel*` exports return `0x8002_0000 | errno`;
//! POSIX-named exports return -1 and store errno in the calling thread's
//! errno slot (the address `__error()` hands out).

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, Metadata};
use std::io::{self, Read, Seek, SeekFrom, Write};
use crate::host::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// FreeBSD errno values (the PS5 kernel ABI).
pub mod errno {
    pub const EPERM: i32 = 1;
    pub const ENOENT: i32 = 2;
    pub const ESRCH: i32 = 3;
    pub const EIO: i32 = 5;
    pub const EBADF: i32 = 9;
    pub const ENOMEM: i32 = 12;
    pub const EACCES: i32 = 13;
    pub const EFAULT: i32 = 14;
    pub const EBUSY: i32 = 16;
    pub const EEXIST: i32 = 17;
    pub const EXDEV: i32 = 18;
    pub const ENOTDIR: i32 = 20;
    pub const EISDIR: i32 = 21;
    pub const EINVAL: i32 = 22;
    pub const ENFILE: i32 = 23;
    pub const EMFILE: i32 = 24;
    pub const EFBIG: i32 = 27;
    pub const ENOSPC: i32 = 28;
    pub const ESPIPE: i32 = 29;
    pub const EROFS: i32 = 30;
    pub const EPIPE: i32 = 32;
    pub const EAGAIN: i32 = 35;
    pub const EWOULDBLOCK: i32 = EAGAIN;
    pub const EOPNOTSUPP: i32 = 45;
    pub const ETIMEDOUT: i32 = 60;
    pub const ELOOP: i32 = 62;
    pub const ENAMETOOLONG: i32 = 63;
    pub const ENOTEMPTY: i32 = 66;
    pub const ENOSYS: i32 = 78;
    pub const EOVERFLOW: i32 = 84;
}
use errno::*;

type FsResult<T> = Result<T, i32>;

/// `SCE_KERNEL_ERROR_*` code for an errno value.
pub const fn sce_error(errno: i32) -> HleResult {
    (0x8002_0000u32 | errno as u32) as i32 as i64
}

/// [`host_errno`] for the other HLE modules.
pub(crate) fn io_errno(e: &io::Error) -> i32 {
    host_errno(e)
}

/// Map a host I/O error to the guest (FreeBSD) errno.
fn host_errno(e: &io::Error) -> i32 {
    #[cfg(windows)]
    let translated = crate::host::errno_of(e).map(io::Error::from_raw_os_error);
    #[cfg(windows)]
    let e = translated.as_ref().unwrap_or(e);
    if let Some(code) = e.raw_os_error() {
        return match code {
            libc::EPERM => EPERM,
            libc::ENOENT => ENOENT,
            libc::EBADF => EBADF,
            libc::ENOMEM => ENOMEM,
            libc::EACCES => EACCES,
            libc::EFAULT => EFAULT,
            libc::EBUSY => EBUSY,
            libc::EEXIST => EEXIST,
            libc::EXDEV => EXDEV,
            libc::ENOTDIR => ENOTDIR,
            libc::EISDIR => EISDIR,
            libc::EINVAL => EINVAL,
            libc::EMFILE | libc::ENFILE => EMFILE,
            libc::ENOSPC => ENOSPC,
            libc::ESPIPE => ESPIPE,
            libc::EPIPE => EPIPE,
            libc::EROFS => EROFS,
            libc::EAGAIN => EAGAIN,
            libc::ELOOP => ELOOP,
            libc::ENAMETOOLONG => ENAMETOOLONG,
            libc::ENOTEMPTY => ENOTEMPTY,
            _ => EIO,
        };
    }
    match e.kind() {
        io::ErrorKind::NotFound => ENOENT,
        io::ErrorKind::PermissionDenied => EACCES,
        io::ErrorKind::AlreadyExists => EEXIST,
        io::ErrorKind::InvalidInput => EINVAL,
        _ => EIO,
    }
}

fn sce(r: FsResult<i64>) -> HleResult {
    match r {
        Ok(v) => v,
        Err(e) => sce_error(e),
    }
}

fn posix(ctx: &CpuContext, r: FsResult<i64>) -> HleResult {
    match r {
        Ok(v) => v,
        Err(e) => {
            set_errno(ctx, e);
            -1
        }
    }
}

// ---------------------------------------------------------------------------
// errno
// ---------------------------------------------------------------------------

/// Store `value` into the calling thread's errno.
pub fn set_errno(ctx: &CpuContext, value: i32) -> bool {
    crate::kernel::set_errno(ctx, value);
    crate::kernel::errno_address() != 0
}

// ---------------------------------------------------------------------------
// Guest memory helpers
// ---------------------------------------------------------------------------

const MAX_GUEST_PATH: usize = 4096;
const IO_CHUNK: usize = 1 << 20;

/// Read a NUL-terminated byte string without crossing into unmapped pages
/// past the terminator.
fn read_guest_cstr(ctx: &CpuContext, addr: u64, max: usize) -> FsResult<Vec<u8>> {
    if addr == 0 {
        return Err(EFAULT);
    }
    let mut out = Vec::new();
    let mut cur = addr;
    while out.len() < max {
        let page_left = 0x1000 - (cur & 0xFFF);
        let n = page_left.min((max - out.len()) as u64) as usize;
        let mut buf = vec![0u8; n];
        ctx.vm.read(cur, &mut buf).map_err(|_| EFAULT)?;
        if let Some(p) = buf.iter().position(|&b| b == 0) {
            out.extend_from_slice(&buf[..p]);
            return Ok(out);
        }
        out.extend_from_slice(&buf);
        cur += n as u64;
    }
    Err(ENAMETOOLONG)
}

/// A guest path argument; a relative one is taken from the lookup start
/// directory of that argument (the working directory, or the directory
/// descriptor of an `*at` call).
fn read_guest_path(ctx: &CpuContext, addr: u64) -> FsResult<String> {
    let raw = read_guest_cstr(ctx, addr, MAX_GUEST_PATH)?;
    check_path_components(addr, &raw)?;
    Ok(at::absolute(addr, String::from_utf8_lossy(&raw).into_owned()))
}

/// FreeBSD's PATH_MAX and NAME_MAX.
const GUEST_PATH_MAX: usize = 1024;
const GUEST_NAME_MAX: usize = 255;

/// Whether the guest path `prefix` (relative to the start of the path
/// argument at `addr`; empty: the start itself) names a directory, or the
/// error looking it up gives.
fn prefix_is_dir(addr: u64, prefix: &[u8]) -> FsResult<bool> {
    let text = if prefix.is_empty() { ".".to_string() } else { String::from_utf8_lossy(prefix).into_owned() };
    let guest = at::absolute(addr, text);
    if let Some((img, rel)) = image_target(&guest) {
        return img.lookup(&rel).map(|id| img.node(id).is_dir).ok_or(ENOENT);
    }
    let host = resolve_no_copy(&guest).ok_or(ENOENT)?;
    fs::metadata(&host).map(|m| m.is_dir()).map_err(|e| host_errno(&e))
}

/// What namei finds wrong with a path before the host sees it: a path of
/// PATH_MAX bytes or more, or a component longer than NAME_MAX below an
/// existing directory, is ENAMETOOLONG; a "." or ".." component below a
/// file that is not a directory is ENOTDIR (the host resolver would drop
/// them).
fn check_path_components(addr: u64, raw: &[u8]) -> FsResult<()> {
    if raw.len() >= GUEST_PATH_MAX {
        return Err(ENAMETOOLONG);
    }
    let mut start = 0;
    for comp in raw.split(|&c| c == b'/') {
        let prefix = &raw[..start];
        start += comp.len() + 1;
        if comp.len() > GUEST_NAME_MAX {
            if prefix.is_empty() || prefix_is_dir(addr, prefix) == Ok(true) {
                return Err(ENAMETOOLONG);
            }
            return Ok(());
        }
        if (comp == b"." || comp == b"..") && !prefix.is_empty() && prefix_is_dir(addr, prefix) == Ok(false) {
            return Err(ENOTDIR);
        }
    }
    Ok(())
}

/// The rename check of a path whose last component (after trailing
/// slashes) is "." or "..": EINVAL when the rest names a directory,
/// ENOTDIR when it names something else, the lookup error otherwise.
fn rename_dot_error(addr: u64, raw: &[u8]) -> Option<i32> {
    let mut end = raw.len();
    while end > 1 && raw[end - 1] == b'/' {
        end -= 1;
    }
    let path = &raw[..end];
    let cut = path.iter().rposition(|&c| c == b'/').map_or(0, |p| p + 1);
    let last = &path[cut..];
    if last != b"." && last != b".." {
        return None;
    }
    Some(match prefix_is_dir(addr, &path[..cut]) {
        Ok(true) => EINVAL,
        Ok(false) => ENOTDIR,
        Err(e) => e,
    })
}

fn read_guest(ctx: &CpuContext, addr: u64, len: usize) -> FsResult<Vec<u8>> {
    let mut buf = vec![0u8; len];
    if len > 0 {
        ctx.vm.read(addr, &mut buf).map_err(|_| EFAULT)?;
    }
    Ok(buf)
}

fn write_guest(ctx: &CpuContext, addr: u64, data: &[u8]) -> FsResult<()> {
    if data.is_empty() {
        return Ok(());
    }
    ctx.vm.write(addr, data).map_err(|_| EFAULT)
}

fn write_u32(ctx: &CpuContext, addr: u64, v: u32) -> FsResult<()> {
    write_guest(ctx, addr, &v.to_le_bytes())
}

fn write_u64(ctx: &CpuContext, addr: u64, v: u64) -> FsResult<()> {
    write_guest(ctx, addr, &v.to_le_bytes())
}

/// Stream up to `total` bytes from `src` into guest memory at `dst`.
/// `src(buf, done)` reads into `buf`; `done` is the count transferred so far.
fn transfer_to_guest(
    ctx: &CpuContext,
    dst: u64,
    total: usize,
    mut src: impl FnMut(&mut [u8], u64) -> io::Result<usize>,
) -> FsResult<usize> {
    let mut chunk = vec![0u8; total.min(IO_CHUNK)];
    let mut done = 0usize;
    while done < total {
        let want = (total - done).min(chunk.len());
        let got = match src(&mut chunk[..want], done as u64) {
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) if done == 0 => return Err(host_errno(&e)),
            Err(_) => break,
        };
        if got == 0 {
            break;
        }
        write_guest(ctx, dst + done as u64, &chunk[..got])?;
        done += got;
        if got < want {
            break;
        }
    }
    Ok(done)
}

/// Stream `total` bytes of guest memory at `src` into `sink(buf, done)`.
fn transfer_from_guest(
    ctx: &CpuContext,
    src: u64,
    total: usize,
    mut sink: impl FnMut(&[u8], u64) -> io::Result<()>,
) -> FsResult<usize> {
    let mut done = 0usize;
    while done < total {
        let n = (total - done).min(IO_CHUNK);
        let data = read_guest(ctx, src + done as u64, n)?;
        if let Err(e) = sink(&data, done as u64) {
            if done == 0 {
                return Err(host_errno(&e));
            }
            break;
        }
        done += n;
    }
    Ok(done)
}

// ---------------------------------------------------------------------------
// Mount configuration and path translation
// ---------------------------------------------------------------------------

/// Host roots backing the guest mount points.
#[derive(Clone, Debug, Default)]
pub struct MountConfig {
    /// Read-only game image (`/app0`, `$/`, `/contents`, relative paths).
    pub app0: Option<PathBuf>,
    pub temp0: Option<PathBuf>,
    pub download0: Option<PathBuf>,
    pub hostapp: Option<PathBuf>,
    pub devlog_app: Option<PathBuf>,
    pub data: Option<PathBuf>,
    /// Dynamically registered mounts (`/savedata0`, ...), keyed by the
    /// normalized guest mount point.
    pub registered: Vec<(String, PathBuf)>,
    /// Allow writes into `/app0` (dev dumps that write their Saved/ tree).
    pub writable_app0: bool,
}

struct FsState {
    config: MountConfig,
    title_id: String,
    temp0_reset: bool,
    resolved: HashMap<String, PathBuf>,
    verified_components: HashSet<PathBuf>,
    negative_stat: HashSet<String>,
    /// Read-only image serving `/app0` instead of `config.app0`.
    image: Option<Arc<App0Image>>,
}

static STATE: Mutex<Option<FsState>> = Mutex::new(None);
const RESOLVED_CACHE_LIMIT: usize = 1 << 20;

fn with_state<R>(f: impl FnOnce(&mut FsState) -> R) -> R {
    let mut guard = STATE.lock().unwrap_or_else(|p| p.into_inner());
    let st = guard.get_or_insert_with(|| FsState {
        config: MountConfig {
            app0: env_dir("RPS5E_APP0_DIR"),
            writable_app0: std::env::var("RPS5E_WRITABLE_APP0").as_deref() == Ok("1"),
            ..Default::default()
        },
        title_id: "UNKNOWN".to_string(),
        temp0_reset: false,
        resolved: HashMap::new(),
        verified_components: HashSet::new(),
        negative_stat: HashSet::new(),
        image: None,
    });
    f(st)
}

fn env_dir(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(|v| absolute(Path::new(&v)))
}

fn absolute(p: &Path) -> PathBuf {
    std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf())
}

/// Uppercase alnum / '-' / '_' title id; anything else becomes '_'.
fn sanitize_title_id(raw: &str) -> String {
    let t = raw.trim();
    if t.is_empty() {
        return "UNKNOWN".to_string();
    }
    t.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

/// Extract `"titleId": "XXXX00000"` from a param.json document.
fn parse_title_id(json: &str) -> Option<String> {
    let key = json.find("\"titleId\"")?;
    let rest = &json[key + "\"titleId\"".len()..];
    let rest = rest.trim_start().strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Bind `/app0` to the game directory (the directory holding eboot.bin).
/// A `decrypted` sidecar directory whose parent holds the content is
/// redirected to that parent. Also derives the title id from
/// `sce_sys/param.json` for the per-title writable mounts.
pub fn set_app0_root(path: &Path) {
    let mut root = absolute(path);
    // An image mounted at /app0 stays mounted when the loader binds /app0
    // to the image's own cache directory (where eboot.bin was copied).
    let keep_image = with_state(|st| st.image.as_ref().is_some_and(|img| root.starts_with(&img.cache)));
    if keep_image {
        log::info!("[fs] /app0 stays on the mounted image ({})", root.display());
        return;
    }
    if root.file_name().is_some_and(|n| n.eq_ignore_ascii_case("decrypted")) {
        if let Some(parent) = root.parent() {
            if parent.join("Media").is_dir() || parent.join("sce_sys").is_dir() {
                root = parent.to_path_buf();
            }
        }
    }
    let title = fs::read_to_string(root.join("sce_sys").join("param.json"))
        .ok()
        .and_then(|j| parse_title_id(&j));
    with_state(|st| {
        if env_dir("RPS5E_APP0_DIR").is_none() {
            st.config.app0 = Some(root.clone());
        }
        if let Some(t) = &title {
            st.title_id = sanitize_title_id(t);
        }
        reset_writable_roots(st);
        invalidate_all_locked(st);
    });
    log::info!(
        "[fs] /app0 -> {} (title {})",
        root.display(),
        title.as_deref().unwrap_or("UNKNOWN")
    );
}

/// Override the title id used to name the per-title writable directory.
pub fn set_title_id(title_id: &str) {
    with_state(|st| {
        st.title_id = sanitize_title_id(title_id);
        reset_writable_roots(st);
        invalidate_all_locked(st);
    });
}

/// Base directory for per-title writable data:
/// `$RPS5E_DATA_DIR`, else `$XDG_DATA_HOME/rps5e`, else `~/.local/share/rps5e`;
/// on Windows `%LOCALAPPDATA%\rps5e`.
pub fn user_data_dir() -> PathBuf {
    if let Some(d) = env_dir("RPS5E_DATA_DIR") {
        return d;
    }
    #[cfg(windows)]
    if let Some(d) = env_dir("LOCALAPPDATA") {
        return ps5vm::appenv::app_dir(&d);
    }
    if let Some(d) = env_dir("XDG_DATA_HOME") {
        return ps5vm::appenv::app_dir(&d);
    }
    if let Some(h) = env_dir("HOME") {
        return ps5vm::appenv::app_dir(&h.join(".local").join("share"));
    }
    std::env::temp_dir().join(ps5vm::appenv::APP_DIR)
}

fn title_dir(st: &FsState) -> PathBuf {
    user_data_dir().join(&st.title_id)
}

/// Host root for save data of every title (`<root>/<titleId>/<dirName>`).
pub fn savedata_root() -> PathBuf {
    crate::savedata::storage_root(None)
}

/// Mount `host_root` at `guest_mount` (e.g. `/savedata0`). The host
/// directory is created on demand.
pub fn register_guest_path_mount(guest_mount: &str, host_root: &Path) -> bool {
    let Some(mount) = normalize_guest_path(guest_mount) else {
        return false;
    };
    if mount == "/" {
        return false;
    }
    let root = absolute(host_root);
    if fs::create_dir_all(&root).is_err() {
        return false;
    }
    with_state(|st| {
        st.config.registered.retain(|(m, _)| !m.eq_ignore_ascii_case(&mount));
        st.config.registered.push((mount.clone(), root));
        invalidate_all_locked(st);
    });
    true
}

/// Remove a mount added by [`register_guest_path_mount`].
pub fn unregister_guest_path_mount(guest_mount: &str) -> bool {
    let Some(mount) = normalize_guest_path(guest_mount) else {
        return false;
    };
    with_state(|st| {
        let before = st.config.registered.len();
        st.config.registered.retain(|(m, _)| !m.eq_ignore_ascii_case(&mount));
        let removed = st.config.registered.len() != before;
        if removed {
            invalidate_all_locked(st);
        }
        removed
    })
}

/// Forget the per-title writable roots so they are recomputed.
fn reset_writable_roots(st: &mut FsState) {
    let c = &mut st.config;
    c.temp0 = None;
    c.download0 = None;
    c.hostapp = None;
    c.devlog_app = None;
    c.data = None;
}

fn invalidate_all_locked(st: &mut FsState) {
    st.resolved.clear();
    st.verified_components.clear();
    st.negative_stat.clear();
}

/// Fill in the writable roots (creating them) for the current title.
fn ensure_writable_roots(st: &mut FsState) {
    let c = &st.config;
    if c.temp0.is_some() && c.download0.is_some() && c.hostapp.is_some() && c.devlog_app.is_some() && c.data.is_some()
    {
        return;
    }
    let base = title_dir(st);
    let pick = |env: &str, sub: &str| env_dir(env).unwrap_or_else(|| base.join(sub));
    if st.config.download0.is_none() {
        st.config.download0 = Some(pick("RPS5E_DOWNLOAD0_DIR", "download0"));
    }
    if st.config.hostapp.is_none() {
        st.config.hostapp = Some(pick("RPS5E_HOSTAPP_DIR", "hostapp"));
    }
    if st.config.devlog_app.is_none() {
        st.config.devlog_app = Some(pick("RPS5E_DEVLOG_APP_DIR", "devlog/app"));
    }
    if st.config.data.is_none() {
        st.config.data = Some(pick("RPS5E_DATA0_DIR", "data"));
    }
    if st.config.temp0.is_none() {
        let configured = env_dir("RPS5E_TEMP0_DIR");
        let root = configured.clone().unwrap_or_else(|| base.join("temp0"));
        // /temp0 starts empty for every boot of the title.
        if configured.is_none() && !st.temp0_reset {
            let _ = fs::remove_dir_all(&root);
            st.temp0_reset = true;
        }
        st.config.temp0 = Some(root);
    }
    for root in [
        &st.config.download0,
        &st.config.hostapp,
        &st.config.devlog_app,
        &st.config.data,
        &st.config.temp0,
    ]
    .into_iter()
    .flatten()
    {
        let _ = fs::create_dir_all(root);
    }
}

/// Normalize a guest path for prefix matching: '\' -> '/', no trailing
/// '/', leading '/' added. Returns None for an empty path.
fn normalize_guest_path(guest: &str) -> Option<String> {
    let s = guest.replace('\\', "/");
    let s = s.trim_end_matches('/');
    if s.is_empty() {
        return if guest.is_empty() { None } else { Some("/".to_string()) };
    }
    Some(if s.starts_with('/') { s.to_string() } else { format!("/{s}") })
}

/// Resolve "." and ".." inside a mount-relative path, clamping at the
/// mount root so the result can never climb above it.
fn normalize_relative(rel: &str) -> Vec<&str> {
    let mut out: Vec<&str> = Vec::new();
    for seg in rel.split(['/', '\\']) {
        match seg {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            s => out.push(s),
        }
    }
    out
}

fn starts_with_ic(s: &str, prefix: &str) -> bool {
    s.len() >= prefix.len()
        && s.is_char_boundary(prefix.len())
        && s[..prefix.len()].eq_ignore_ascii_case(prefix)
}

/// If `guest` names `mount` or something below it (ASCII case-insensitive),
/// return the remainder after the mount point.
fn strip_mount<'a>(guest: &'a str, mount: &str) -> Option<&'a str> {
    if guest.eq_ignore_ascii_case(mount) {
        return Some("");
    }
    if starts_with_ic(guest, mount) && guest.as_bytes().get(mount.len()) == Some(&b'/') {
        return Some(&guest[mount.len() + 1..]);
    }
    None
}

/// Join normalized segments onto `root` and verify the result stays inside
/// it, including through symlinks planted inside the mount.
/// Host name of `seg` inside `dir`, matched without regard to ASCII case.
/// The console file system is case-insensitive while host dumps usually
/// store lower-case names; an exact match always wins.
fn find_case_insensitive(dir: &Path, seg: &str) -> Option<String> {
    let mut found = None;
    for entry in fs::read_dir(dir).ok()?.flatten() {
        let name = entry.file_name();
        let name = name.to_str()?;
        if name.eq_ignore_ascii_case(seg) {
            if found.is_some() {
                return None; // ambiguous: several names differ only by case
            }
            found = Some(name.to_string());
        }
    }
    found
}

fn combine_within(st: &mut FsState, root: &Path, rel: &str) -> Option<PathBuf> {
    let mut current = root.to_path_buf();
    let mut walking = true;
    for seg in normalize_relative(rel) {
        if !walking {
            current.push(seg);
            continue;
        }
        let mut next = current.join(seg);
        if !st.verified_components.contains(&next) && fs::symlink_metadata(&next).is_err() {
            if let Some(real) = find_case_insensitive(&current, seg) {
                next = current.join(real);
            }
        }
        current = next;
        if st.verified_components.contains(&current) {
            continue;
        }
        match fs::symlink_metadata(&current) {
            Ok(m) if m.file_type().is_symlink() => {
                // A link is only followed when its target stays in the mount.
                let canon_root = fs::canonicalize(root).ok()?;
                match fs::canonicalize(&current) {
                    Ok(target) if target.starts_with(&canon_root) => {
                        st.verified_components.insert(current.clone());
                    }
                    Ok(_) => {
                        log::warn!("[fs] denied symlink escape {}", current.display());
                        return None;
                    }
                    // A dangling link names itself (lstat, unlink, rename
                    // onto it) as long as its target would stay in the mount.
                    Err(_) if dangling_link_within(&current, &canon_root) => walking = false,
                    Err(_) => {
                        log::warn!("[fs] denied symlink escape {}", current.display());
                        return None;
                    }
                }
            }
            Ok(_) => {
                st.verified_components.insert(current.clone());
            }
            // Not created yet (O_CREAT / mkdir target): nothing to follow.
            Err(e) if e.kind() == io::ErrorKind::NotFound => walking = false,
            Err(e) if e.raw_os_error() == Some(libc::ENOTDIR) => walking = false,
            Err(_) => return None,
        }
    }
    current.starts_with(root).then_some(current)
}

/// True when the (missing) target of the link at `link` lies lexically
/// inside `canon_root`, judged from the deepest existing ancestor of the
/// target.
fn dangling_link_within(link: &Path, canon_root: &Path) -> bool {
    let Ok(target) = fs::read_link(link) else { return false };
    let base = link.parent().map(Path::to_path_buf).unwrap_or_default();
    let full = if target.is_absolute() { target } else { base.join(target) };
    let mut existing = full.as_path();
    let mut rest = Vec::new();
    loop {
        if let Ok(canon) = fs::canonicalize(existing) {
            let mut resolved = canon;
            for seg in rest.iter().rev() {
                match seg {
                    std::path::Component::ParentDir => {
                        resolved.pop();
                    }
                    std::path::Component::Normal(n) => resolved.push(n),
                    _ => {}
                }
            }
            return resolved.starts_with(canon_root);
        }
        let Some(parent) = existing.parent() else { return false };
        if let Some(last) = existing.components().next_back() {
            rest.push(last);
        }
        existing = parent;
    }
}

/// Translate a guest path to a host path under the current mount table.
/// Returns None for paths outside every mount (default deny).
/// Inside an image mounted at `/app0` the file is first copied to the
/// image's host cache (see [`App0Image`]).
pub fn resolve_guest_path(guest: &str) -> Option<PathBuf> {
    if let Some((img, rel)) = image_target(guest) {
        return img.materialize(&rel);
    }
    with_state(|st| resolve_locked(st, guest))
}

/// Like [`resolve_guest_path`] but never copies out of an image (callers
/// that only check permissions or modify files).
fn resolve_no_copy(guest: &str) -> Option<PathBuf> {
    if let Some((img, rel)) = image_target(guest) {
        return Some(img.cache.join(rel));
    }
    with_state(|st| resolve_locked(st, guest))
}

fn resolve_locked(st: &mut FsState, guest: &str) -> Option<PathBuf> {
    if guest.trim().is_empty() {
        return None;
    }
    if let Some(p) = st.resolved.get(guest) {
        return Some(p.clone());
    }
    ensure_writable_roots(st);
    let resolved = resolve_uncached(st, guest)?;
    if st.resolved.len() < RESOLVED_CACHE_LIMIT {
        st.resolved.insert(guest.to_string(), resolved.clone());
    }
    Some(resolved)
}

fn resolve_uncached(st: &mut FsState, guest: &str) -> Option<PathBuf> {
    // Registered mounts own their prefix; a denial there is final.
    if let Some(norm) = normalize_guest_path(guest) {
        let best = st
            .config
            .registered
            .iter()
            .filter(|(m, _)| strip_mount(&norm, m).is_some())
            .max_by_key(|(m, _)| m.len())
            .cloned();
        if let Some((mount, root)) = best {
            let rel = strip_mount(&norm, &mount).unwrap_or("").to_string();
            return combine_within(st, &root, &rel);
        }
    }

    let builtin: [(&[&str], Option<PathBuf>); 5] = [
        (&["/devlog/app", "devlog/app"], st.config.devlog_app.clone()),
        (&["/temp0"], st.config.temp0.clone()),
        (&["/download0", "download0"], st.config.download0.clone()),
        (&["/hostapp", "hostapp"], st.config.hostapp.clone()),
        (&["/data"], st.config.data.clone()),
    ];
    for (prefixes, root) in builtin {
        for prefix in prefixes {
            if let Some(rest) = strip_mount(guest, prefix) {
                let root = root?;
                return combine_within(st, &root, rest);
            }
        }
    }

    let rel = app0_rel(guest)?;
    let app0 = st.config.app0.clone()?;
    combine_within(st, &app0, rel)
}

/// The part of `guest` below `/app0`, for every spelling that maps there:
/// `/app0/...`, `app0/...`, `$/...`, `/contents/...` and relative paths.
fn app0_rel(guest: &str) -> Option<&str> {
    if guest == "$" || guest == "$/" || guest == "$\\" {
        return Some("");
    }
    if let Some(rest) = guest.strip_prefix("$/").or_else(|| guest.strip_prefix("$\\")) {
        return Some(rest);
    }
    for prefix in ["/app0", "app0"] {
        if let Some(rest) = strip_mount(guest, prefix) {
            return Some(rest);
        }
    }
    if guest.eq_ignore_ascii_case("/contents") || starts_with_ic(guest, "/contents/") {
        return Some(&guest[1..]);
    }
    if !guest.starts_with('/') && !guest.starts_with('\\') {
        return Some(guest);
    }
    None
}

/// Mount prefixes owned by the writable per-title roots (kept in step
/// with the table in `resolve_uncached`).
const WRITABLE_PREFIXES: [&str; 8] =
    ["/devlog/app", "devlog/app", "/temp0", "/download0", "download0", "/hostapp", "hostapp", "/data"];

/// When `/app0` is served by an image and `guest` lies inside it: the
/// image and the normalized path relative to its root.
fn image_rel(st: &FsState, guest: &str) -> Option<(Arc<App0Image>, String)> {
    let img = st.image.clone()?;
    if guest.trim().is_empty() {
        return None;
    }
    if let Some(norm) = normalize_guest_path(guest) {
        if st.config.registered.iter().any(|(m, _)| strip_mount(&norm, m).is_some()) {
            return None;
        }
    }
    if WRITABLE_PREFIXES.iter().any(|p| strip_mount(guest, p).is_some()) {
        return None;
    }
    let rel = normalize_relative(app0_rel(guest)?).join("/");
    Some((img, rel))
}

fn image_target(guest: &str) -> Option<(Arc<App0Image>, String)> {
    with_state(|st| image_rel(st, guest))
}

/// Writes are refused anywhere inside the game image.
pub(crate) fn is_read_only(guest: &str, host: Option<&Path>) -> bool {
    with_state(|st| {
        if image_rel(st, guest).is_some() {
            return true;
        }
        if let (Some(img), Some(h)) = (&st.image, host) {
            if h.starts_with(&img.cache) {
                return true;
            }
        }
        if st.config.writable_app0 {
            return false;
        }
        let guest_hit = normalize_guest_path(guest).is_some_and(|n| strip_mount(&n, "/app0").is_some());
        let host_hit = match (host, &st.config.app0) {
            (Some(h), Some(app0)) => h.starts_with(app0),
            _ => false,
        };
        guest_hit || host_hit
    })
}

// ---------------------------------------------------------------------------
// Read-only image at /app0
// ---------------------------------------------------------------------------

/// A package's PFS tree mounted read-only at `/app0`.
///
/// Guest file calls (open/read/stat/getdents/APR/stdio) read the image
/// directly. Host code that needs a real path (the module loader, FFmpeg,
/// trophy files) gets a copy of just that file in `cache`, written on first
/// use ([`resolve_guest_path`]).
pub struct App0Image {
    pfs: Arc<ps5pkg::Pfs>,
    cache: PathBuf,
    materialize_lock: Mutex<()>,
}

impl App0Image {
    fn lookup(&self, rel: &str) -> Option<ps5pkg::NodeId> {
        self.pfs.lookup(rel)
    }

    fn node(&self, id: ps5pkg::NodeId) -> &ps5pkg::Node {
        self.pfs.node(id)
    }

    /// Path shown in logs and used to seed inode numbers.
    fn display_path(&self, id: ps5pkg::NodeId) -> PathBuf {
        PathBuf::from(format!("pfs:/{}", self.pfs.path(id)))
    }

    /// Directory listing as the guest sees it ("." and ".." first, then
    /// the children ordered case-insensitively).
    fn listing(&self, id: ps5pkg::NodeId) -> Vec<(String, bool)> {
        let mut out = vec![(".".to_string(), false), ("..".to_string(), false)];
        out.extend(self.node(id).children.iter().map(|&c| (self.node(c).name.clone(), !self.node(c).is_dir)));
        out
    }

    /// Packed dirent stream of an image directory (same layout as a host
    /// directory's).
    fn directory_stream(&self, id: ps5pkg::NodeId) -> Vec<u8> {
        pack_dirents(&self.listing(id))
    }

    fn stat(&self, id: ps5pkg::NodeId) -> [u8; KERNEL_STAT_SIZE] {
        let n = self.node(id);
        let t = UNIX_EPOCH + std::time::Duration::from_secs(n.mtime.max(0) as u64);
        // A directory's size is the length of its packed dirent stream.
        let size = if n.is_dir { self.directory_stream(id).len() as i64 } else { n.size as i64 };
        build_kernel_stat(n.is_dir, size, t, t, t, &self.display_path(id).to_string_lossy())
    }

    /// Host copy of `rel`: directories are created empty, files copied once
    /// (kept while the size matches). A missing path yields where it would
    /// be, which does not exist.
    fn materialize(&self, rel: &str) -> Option<PathBuf> {
        let Some(id) = self.lookup(rel) else {
            return Some(self.cache.join(rel));
        };
        let out = self.cache.join(self.pfs.path(id));
        let n = self.node(id);
        let _guard = self.materialize_lock.lock().unwrap_or_else(|p| p.into_inner());
        if n.is_dir {
            let _ = fs::create_dir_all(&out);
            return Some(out);
        }
        if fs::metadata(&out).is_ok_and(|m| m.is_file() && m.len() == n.size) {
            return Some(out);
        }
        if let Err(e) = self.copy_out(id, &out) {
            log::warn!("[fs] cannot copy {} out of the /app0 image: {e}", out.display());
            return None;
        }
        log::info!("[fs] copied /app0/{} out of the image ({} bytes)", self.pfs.path(id), n.size);
        Some(out)
    }

    fn copy_out(&self, id: ps5pkg::NodeId, out: &Path) -> io::Result<()> {
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        let data = self.pfs.open_file(id).map_err(|e| io::Error::other(e.to_string()))?;
        let mut tmp = out.as_os_str().to_owned();
        tmp.push(".part~");
        let tmp = PathBuf::from(tmp);
        let mut f = File::create(&tmp)?;
        let mut buf = vec![0u8; 4 << 20];
        let mut off = 0u64;
        while off < data.len() {
            let n = data.read_at(&mut buf, off)?;
            if n == 0 {
                break;
            }
            f.write_all(&buf[..n])?;
            off += n as u64;
        }
        drop(f);
        fs::rename(&tmp, out)
    }
}

/// Serve `/app0` from a package's PFS tree, read-only. `title_id` names
/// the per-title writable roots and the copy cache; `source_id` identifies
/// the image (the cache is cleared when it changes). Copies `eboot.bin`,
/// `sce_sys/` and `sce_module/` to the cache right away (the loader and the
/// system services read them by host path) and returns the cache
/// directory, which stands in for the game folder on the host side.
pub fn mount_app0_image(pfs: Arc<ps5pkg::Pfs>, title_id: &str, source_id: &str) -> io::Result<PathBuf> {
    let cache = with_state(|st| {
        st.title_id = sanitize_title_id(title_id);
        title_dir(st).join("app0-cache")
    });
    let marker = cache.with_extension("source");
    if fs::read_to_string(&marker).ok().as_deref() != Some(source_id) {
        let _ = fs::remove_dir_all(&cache);
    }
    fs::create_dir_all(&cache)?;
    fs::write(&marker, source_id)?;
    let img = Arc::new(App0Image { pfs, cache: cache.clone(), materialize_lock: Mutex::new(()) });
    with_state(|st| {
        st.config.app0 = Some(cache.clone());
        st.config.writable_app0 = false;
        st.image = Some(img.clone());
        reset_writable_roots(st);
        invalidate_all_locked(st);
    });
    let root = img.pfs.root();
    let mut prefetch: Vec<ps5pkg::NodeId> =
        ["eboot.bin", "sce_sys", "sce_module"].iter().filter_map(|n| img.pfs.child(root, n)).collect();
    while let Some(id) = prefetch.pop() {
        let n = img.node(id);
        if n.is_dir {
            prefetch.extend(n.children.iter().copied());
        } else if img.materialize(&img.pfs.path(id)).is_none() {
            return Err(io::Error::other(format!("cannot copy /app0/{} to {}", img.pfs.path(id), cache.display())));
        }
    }
    log::info!(
        "[fs] /app0 -> read-only image ({} entries, title {title_id}), host copies in {}",
        img.pfs.node_count(),
        cache.display()
    );
    Ok(cache)
}

/// True when `guest` names something inside an image mounted at `/app0`
/// (whether or not it exists).
pub fn in_app0_image(guest: &str) -> bool {
    image_target(guest).is_some()
}

/// Unmount the `/app0` image (tests).
#[cfg(test)]
pub(crate) fn unmount_app0_image() {
    with_state(|st| {
        st.image = None;
        st.config.app0 = None;
        invalidate_all_locked(st);
    });
}

/// A readable file: on the host, or inside the image mounted at `/app0`.
pub enum FileSource {
    Host(File),
    Image(ps5pkg::FileData),
}

/// Writing to an image file is writing to a descriptor not open for it.
fn not_writable() -> io::Error {
    io::Error::from_raw_os_error(libc::EBADF)
}

impl FileSource {
    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        match self {
            FileSource::Host(f) => f.read_at(buf, offset),
            FileSource::Image(d) => d.read_at(buf, offset),
        }
    }
    pub fn len(&self) -> io::Result<u64> {
        match self {
            FileSource::Host(f) => Ok(f.metadata()?.len()),
            FileSource::Image(d) => Ok(d.len()),
        }
    }
    pub fn host(&self) -> Option<&File> {
        match self {
            FileSource::Host(f) => Some(f),
            FileSource::Image(_) => None,
        }
    }
    pub fn is_image(&self) -> bool {
        matches!(self, FileSource::Image(_))
    }
    /// Sequential (or O_APPEND) write; host files only.
    pub fn write_all(&self, data: &[u8]) -> io::Result<()> {
        let mut f = self.host().ok_or_else(not_writable)?;
        f.write_all(data)
    }
    pub fn write_all_at(&self, data: &[u8], offset: u64) -> io::Result<()> {
        self.host().ok_or_else(not_writable)?.write_all_at(data, offset)
    }
}

/// Open a guest file for reading from whichever backend holds it.
/// None when the path is outside every mount or does not exist.
pub fn open_guest_file(guest: &str) -> Option<io::Result<FileSource>> {
    if let Some((img, rel)) = image_target(guest) {
        let id = img.lookup(&rel)?;
        if img.node(id).is_dir {
            return Some(Err(io::Error::from_raw_os_error(libc::EISDIR)));
        }
        return Some(img.pfs.open_file(id).map(FileSource::Image).map_err(|e| io::Error::other(e.to_string())));
    }
    let host = resolve_guest_path(guest)?;
    if !host.exists() {
        return None;
    }
    Some(File::open(host).map(FileSource::Host))
}

// ---- negative stat cache (paths known to be absent) ----

const STAT_CACHE_PREFIXES: [&str; 5] = ["/app0", "/hostapp", "/devlog/app", "/temp0", "/download0"];

fn negative_stat_key(guest: &str) -> Option<String> {
    let n = normalize_guest_path(guest)?;
    STAT_CACHE_PREFIXES
        .iter()
        .any(|p| strip_mount(&n, p).is_some())
        .then_some(n)
}

fn negative_stat_cached(key: &str) -> bool {
    with_state(|st| st.negative_stat.contains(key))
}

fn add_negative_stat(guest: &str) {
    if let Some(k) = negative_stat_key(guest) {
        with_state(|st| st.negative_stat.insert(k));
    }
}

fn invalidate_negative_stat_ancestors(guest: &str) {
    let Some(mut cur) = normalize_guest_path(guest) else {
        return;
    };
    with_state(|st| {
        loop {
            st.negative_stat.remove(&cur);
            match cur.rfind('/') {
                Some(i) if i > 0 => cur.truncate(i),
                _ => break,
            }
        }
        st.negative_stat.remove("/");
    });
}

// ---------------------------------------------------------------------------
// Descriptor table
// ---------------------------------------------------------------------------

const O_ACCMODE: u64 = 0x3;
const O_WRONLY: u64 = 0x1;
const O_RDWR: u64 = 0x2;
const O_APPEND: u64 = 0x8;
const O_CREAT: u64 = 0x200;
const O_TRUNC: u64 = 0x400;
const O_EXCL: u64 = 0x800;
const O_DIRECTORY: u64 = 0x2_0000;
const O_CLOEXEC: u64 = 0x10_0000;

const SEEK_SET: i32 = 0;
const SEEK_CUR: i32 = 1;
const SEEK_END: i32 = 2;

struct OpenFile {
    file: FileSource,
    /// Host path, or `pfs:/...` for a file of the `/app0` image.
    host: PathBuf,
    /// Sequential position of an image file (host files use the OS one).
    pos: Mutex<u64>,
    /// Stat of an image file (fixed: the image is read-only).
    image_stat: Option<[u8; KERNEL_STAT_SIZE]>,
}

impl OpenFile {
    fn host(file: File, host: PathBuf) -> Self {
        Self { file: FileSource::Host(file), host, pos: Mutex::new(0), image_stat: None }
    }

    fn read_seq(&self, buf: &mut [u8]) -> io::Result<usize> {
        match &self.file {
            FileSource::Host(f) => (&*f).read(buf),
            FileSource::Image(d) => {
                let mut pos = self.pos.lock().unwrap_or_else(|p| p.into_inner());
                let n = d.read_at(buf, *pos)?;
                *pos += n as u64;
                Ok(n)
            }
        }
    }

    /// Current sequential position.
    fn position(&self) -> io::Result<u64> {
        match &self.file {
            FileSource::Host(f) => (&*f).stream_position(),
            FileSource::Image(_) => Ok(*self.pos.lock().unwrap_or_else(|p| p.into_inner())),
        }
    }

    fn seek(&self, to: SeekFrom) -> io::Result<u64> {
        match &self.file {
            FileSource::Host(f) => (&*f).seek(to),
            FileSource::Image(d) => {
                let mut pos = self.pos.lock().unwrap_or_else(|p| p.into_inner());
                let target = match to {
                    SeekFrom::Start(o) => o as i128,
                    SeekFrom::Current(o) => *pos as i128 + o as i128,
                    SeekFrom::End(o) => d.len() as i128 + o as i128,
                };
                if !(0..=i64::MAX as i128).contains(&target) {
                    return Err(io::Error::from_raw_os_error(libc::EINVAL));
                }
                *pos = target as u64;
                Ok(*pos)
            }
        }
    }
}

/// An open directory is a byte stream of packed dirent blocks: reads,
/// seeks, getdirentries and fstat all see the same bytes.
struct OpenDir {
    /// The absolute guest path it was opened by (the start of `*at` lookups).
    guest: String,
    host: PathBuf,
    dirents: Vec<u8>,
    offset: u64,
    /// Directory of the `/app0` image.
    image: Option<(Arc<App0Image>, ps5pkg::NodeId)>,
}

#[derive(Clone)]
enum Handle {
    File(Arc<OpenFile>),
    Dir(Arc<Mutex<OpenDir>>),
    /// `/dev/random` or `/dev/urandom`: reads return fresh random bytes.
    Random,
    /// One end of a pipe().
    Pipe(Arc<pipe::PipeEnd>),
}

/// The kernel's random character devices.
pub(crate) fn is_random_device(guest: &str) -> bool {
    guest == "/dev/random" || guest == "/dev/urandom"
}

/// Fill `len` guest bytes at `buf` from the host CSPRNG.
fn read_random(ctx: &CpuContext, buf: u64, len: usize) -> FsResult<i64> {
    let n = transfer_to_guest(ctx, buf, len, |chunk, _| {
        crate::misc_services::fill_random(chunk);
        Ok(chunk.len())
    })?;
    Ok(n as i64)
}

struct FdTable {
    map: HashMap<i32, Handle>,
    /// Numbers held by objects living outside this table (kernel sockets,
    /// kqueues, JIT shared memory): never handed out while held.
    reserved: HashSet<i32>,
    /// Descriptors carrying FD_CLOEXEC (a per-descriptor flag: duplicates
    /// start without it).
    cloexec: HashSet<i32>,
    /// Access mode and status flags (O_NONBLOCK, O_APPEND, ...) of each
    /// table entry, shared by its duplicates as they share the open file.
    status: HashMap<i32, Arc<Mutex<u32>>>,
}

impl FdTable {
    /// Bind `fd` to `handle` with its descriptor flags cleared.
    fn bind(&mut self, fd: i32, handle: Handle) {
        self.reserved.remove(&fd);
        self.map.insert(fd, handle);
        self.cloexec.remove(&fd);
        self.status.insert(fd, Arc::new(Mutex::new(0)));
    }

    /// Make `fd` another descriptor of `from`'s open file (handle and
    /// status flags shared, FD_CLOEXEC clear).
    fn bind_duplicate(&mut self, fd: i32, from: i32) -> Option<()> {
        let handle = self.map.get(&from)?.clone();
        let status = self.status.get(&from).cloned().unwrap_or_else(|| Arc::new(Mutex::new(0)));
        self.bind(fd, handle);
        self.status.insert(fd, status);
        Some(())
    }

    /// The lowest number at or above `from` that names nothing, as FreeBSD
    /// allocates descriptors; 0-2 are the standard streams.
    fn lowest_free(&self, from: i32) -> i32 {
        let mut fd = from.max(3);
        while self.map.contains_key(&fd) || self.reserved.contains(&fd) {
            fd += 1;
        }
        fd
    }
}

static FDS: LazyLock<Mutex<FdTable>> =
    // The standard streams are always open.
    LazyLock::new(|| {
        Mutex::new(FdTable { map: HashMap::new(), reserved: (0..=2).collect(), cloexec: HashSet::new(), status: HashMap::new() })
    });

fn fds() -> std::sync::MutexGuard<'static, FdTable> {
    FDS.lock().unwrap_or_else(|p| p.into_inner())
}

fn alloc_fd(handle: Handle) -> i32 {
    let mut t = fds();
    let fd = t.lowest_free(3);
    t.bind(fd, handle);
    fd
}

/// Record O_CLOEXEC, the access mode and the status flags of a freshly
/// opened descriptor.
fn set_open_cloexec(fd: i32, flags: u64) {
    let mut t = fds();
    if flags & O_CLOEXEC != 0 {
        t.cloexec.insert(fd);
    }
    if let Some(st) = t.status.get(&fd) {
        *st.lock().unwrap_or_else(|p| p.into_inner()) = (flags & (O_ACCMODE | STATUS_OPEN_FLAGS)) as u32;
    }
}

/// Set the access mode and status flags of a table entry.
pub(super) fn set_status(fd: i32, value: u32) {
    if let Some(st) = fds().status.get(&fd) {
        *st.lock().unwrap_or_else(|p| p.into_inner()) = value;
    }
}

const O_NONBLOCK_: u64 = 0x4;
const O_ASYNC: u64 = 0x40;
const O_FSYNC: u64 = 0x80;
const O_DIRECT: u64 = 0x1_0000;
/// The status flags open() records and F_SETFL may change.
const STATUS_OPEN_FLAGS: u64 = O_NONBLOCK_ | O_APPEND | O_ASYNC | O_FSYNC | O_DIRECT;

/// Reserve a descriptor number for an object living outside this table
/// (kernel sockets, kqueues, JIT shared memory): the lowest free number,
/// shared with files. It stays taken until `release_fd`.
pub(crate) fn reserve_fd() -> i32 {
    let mut t = fds();
    let fd = t.lowest_free(3);
    t.reserved.insert(fd);
    t.cloexec.remove(&fd);
    fd
}

/// Give back a number taken by `reserve_fd` once its object is closed.
pub(crate) fn release_fd(fd: i32) {
    let mut t = fds();
    if t.reserved.remove(&fd) {
        t.cloexec.remove(&fd);
    }
}

/// Set or clear FD_CLOEXEC of an open descriptor (table entry or reserved
/// number); false when `fd` is neither.
pub(crate) fn set_cloexec(fd: i32, on: bool) -> bool {
    let mut t = fds();
    if !t.map.contains_key(&fd) && !t.reserved.contains(&fd) {
        return false;
    }
    if on {
        t.cloexec.insert(fd);
    } else {
        t.cloexec.remove(&fd);
    }
    true
}

fn get_handle(fd: i32) -> Option<Handle> {
    fds().map.get(&fd).cloned()
}

/// Whether `fd` is an open file-system descriptor (file, directory or
/// device).
pub(crate) fn fd_is_open(fd: i32) -> bool {
    get_handle(fd).is_some()
}

/// open(path, O_RDWR | O_CREAT | O_EXCL), as mkstemp creates its file:
/// the new descriptor or the errno.
pub(crate) fn create_exclusive(ctx: &CpuContext, path_addr: u64) -> FsResult<i64> {
    open_core(ctx, path_addr, O_RDWR | O_CREAT | O_EXCL)
}

/// close(fd) for other modules' tests.
#[cfg(test)]
pub(crate) fn close_fd(fd: i32) -> FsResult<i64> {
    close_core(fd)
}

fn get_file(fd: i32) -> FsResult<Arc<OpenFile>> {
    match get_handle(fd) {
        Some(Handle::File(f)) => Ok(f),
        Some(Handle::Dir(_)) => Err(EISDIR),
        Some(Handle::Random) => Err(EINVAL),
        Some(Handle::Pipe(_)) => Err(ESPIPE),
        None => Err(EBADF),
    }
}

/// Whether the descriptor's source is a regular file: a host file is one
/// on the metadata, an image source always is, and a directory, the random
/// device and a pipe never are. None when the descriptor is not open.
/// FreeBSD's sendfile takes a regular file (or a shared-memory object,
/// which the host also backs with one) and nothing else.
pub(crate) fn descriptor_is_regular(fd: i32) -> Option<bool> {
    match get_handle(fd) {
        Some(Handle::File(f)) => match &f.file {
            FileSource::Host(file) => Some(file.metadata().map(|m| m.is_file()).unwrap_or(false)),
            FileSource::Image(_) => Some(true),
        },
        Some(Handle::Dir(_) | Handle::Random | Handle::Pipe(_)) => Some(false),
        None => None,
    }
}

fn is_stdio(fd: i32) -> bool {
    (0..=2).contains(&fd)
}

/// Directory listing as the guest sees it: "." and ".." first, then the
/// children ordered case-insensitively; (name, is a regular file).
fn dir_entries(host: &Path) -> io::Result<Vec<(String, bool)>> {
    let mut names: Vec<(String, bool)> = fs::read_dir(host)?
        .filter_map(|e| e.ok())
        .map(|e| {
            let is_file = !e.path().is_dir();
            (e.file_name().to_string_lossy().into_owned(), is_file)
        })
        .filter(|(n, _)| !n.is_empty())
        .collect();
    names.sort_by_key(|(n, _)| n.to_ascii_lowercase());
    let mut out = vec![(".".to_string(), false), ("..".to_string(), false)];
    out.extend(names);
    Ok(out)
}

/// Directory stream block size.
const DIR_BLOCK_SIZE: u64 = 512;

/// Jenkins one-at-a-time hash: the dirent file number of a name.
pub fn name_hash(key: &[u8]) -> u32 {
    let mut h: u32 = 0;
    for &b in key {
        h = h.wrapping_add(b as u32);
        h = h.wrapping_add(h << 10);
        h ^= h >> 6;
    }
    h = h.wrapping_add(h << 3);
    h ^= h >> 11;
    h.wrapping_add(h << 15)
}

/// Pack directory entries into 512-byte blocks. Record: fileno u32,
/// reclen u16, type u8, namlen u8, name, NUL, padded to 4 bytes; a record
/// never crosses a block and the last record of a block absorbs its tail.
pub fn pack_dirents(entries: &[(String, bool)]) -> Vec<u8> {
    let mut d: Vec<u8> = Vec::new();
    let mut offset = 0usize;
    let mut last_reclen = 0usize;
    for (name, is_file) in entries {
        let name = &name.as_bytes()[..name.len().min(255)];
        let reclen = (8 + name.len() + 1 + 3) & !3;
        if offset + reclen > d.len() {
            if !d.is_empty() {
                let tail = (d.len() - offset) as u16;
                let cur = u16::from_le_bytes([d[last_reclen], d[last_reclen + 1]]);
                d[last_reclen..last_reclen + 2].copy_from_slice(&(cur + tail).to_le_bytes());
            }
            offset = d.len();
            d.resize(offset + DIR_BLOCK_SIZE as usize, 0);
        }
        d[offset..offset + 4].copy_from_slice(&name_hash(name).to_le_bytes());
        d[offset + 4..offset + 6].copy_from_slice(&(reclen as u16).to_le_bytes());
        d[offset + 6] = if *is_file { DT_REG } else { DT_DIR };
        d[offset + 7] = name.len() as u8;
        d[offset + 8..offset + 8 + name.len()].copy_from_slice(name);
        d[offset + 8 + name.len()] = 0;
        last_reclen = offset + 4;
        offset += reclen;
    }
    if !d.is_empty() {
        let tail = (d.len() - offset) as u16;
        let cur = u16::from_le_bytes([d[last_reclen], d[last_reclen + 1]]);
        d[last_reclen..last_reclen + 2].copy_from_slice(&(cur + tail).to_le_bytes());
    }
    d
}

/// Packed directory stream of a host directory.
fn directory_stream(host: &Path) -> FsResult<Vec<u8>> {
    dir_entries(host).map(|e| pack_dirents(&e)).map_err(|e| host_errno(&e))
}

/// `base + offset` as a stream position: EINVAL below 0, EOVERFLOW past
/// i64::MAX.
fn seek_position(base: u64, offset: i64) -> FsResult<u64> {
    if offset >= 0 {
        if base > (i64::MAX - offset) as u64 {
            return Err(EOVERFLOW);
        }
        Ok(base + offset as u64)
    } else {
        let distance = offset.unsigned_abs();
        if base < distance {
            return Err(EINVAL);
        }
        let pos = base - distance;
        if pos > i64::MAX as u64 { Err(EOVERFLOW) } else { Ok(pos) }
    }
}

/// Open flags the console refuses with EINVAL before it looks at the path:
/// access mode 3, and O_CREAT | O_EXCL | O_DIRECTORY together.
pub(crate) fn invalid_open_flags(flags: u64) -> bool {
    flags & O_ACCMODE == O_ACCMODE || flags & (O_CREAT | O_EXCL | O_DIRECTORY) == O_CREAT | O_EXCL | O_DIRECTORY
}

fn open_core(ctx: &CpuContext, path_addr: u64, flags: u64) -> FsResult<i64> {
    if invalid_open_flags(flags) {
        return Err(EINVAL);
    }
    let guest = read_guest_path(ctx, path_addr)?;
    if is_random_device(&guest) {
        if flags & O_DIRECTORY != 0 {
            return Err(ENOTDIR);
        }
        let fd = alloc_fd(Handle::Random);
        set_open_cloexec(fd, flags);
        log::info!("[fs] open device '{}' flags={:#x} fd={}", guest, flags, fd);
        return Ok(fd as i64);
    }
    let access = flags & O_ACCMODE;
    let write = access == O_WRONLY || access & O_RDWR != 0;
    let read = access & O_RDWR != 0 || access == 0;
    if flags & O_TRUNC != 0 && !write {
        return Err(EACCES);
    }
    let mutating = flags & (O_WRONLY | O_RDWR | O_CREAT | O_TRUNC | O_APPEND) != 0;
    if let Some((img, rel)) = image_target(&guest) {
        return open_image(&guest, flags, mutating, &img, &rel);
    }
    let Some(host) = resolve_guest_path(&guest) else {
        log::info!("[fs] open '{}' flags={:#x} -> unresolved path", guest, flags);
        return Err(ENOENT);
    };
    if mutating && is_read_only(&guest, Some(&host)) {
        log::info!("[fs] open '{}' flags={:#x} -> read-only mount", guest, flags);
        return Err(EACCES);
    }

    let is_dir = host.is_dir();
    if flags & O_DIRECTORY != 0 || is_dir {
        if !is_dir {
            log::info!("[fs] open '{}' (dir) -> {} missing", guest, host.display());
            // A file is ENOTDIR; a failed lookup reports its own error
            // (ENOENT, ELOOP for a symbolic link loop, ...).
            return Err(match fs::metadata(&host) {
                Ok(_) => ENOTDIR,
                Err(e) => host_errno(&e),
            });
        }
        // O_CREAT | O_EXCL fails on anything that exists, directories included.
        if flags & (O_CREAT | O_EXCL) == O_CREAT | O_EXCL {
            return Err(EEXIST);
        }
        if write || flags & (O_CREAT | O_TRUNC | O_APPEND) != 0 {
            return Err(EISDIR);
        }
        let dirents = directory_stream(&host)?;
        let fd = alloc_fd(Handle::Dir(Arc::new(Mutex::new(OpenDir {
            guest: guest.clone(),
            host: host.clone(),
            dirents,
            offset: 0,
            image: None,
        }))));
        set_open_cloexec(fd, flags);
        log::info!("[fs] open dir '{}' -> {} fd={}", guest, host.display(), fd);
        return Ok(fd as i64);
    }

    // Creating a file also creates its missing parent directories.
    let devlog = strip_mount(&guest, "/devlog/app").is_some() || strip_mount(&guest, "devlog/app").is_some();
    if flags & O_CREAT != 0 || devlog {
        if let Some(parent) = host.parent() {
            let _ = fs::create_dir_all(parent);
        }
    }

    let mut extra = libc::O_CLOEXEC;
    if flags & O_CREAT != 0 {
        extra |= libc::O_CREAT;
        if flags & O_EXCL != 0 {
            extra |= libc::O_EXCL;
        }
    }
    if flags & O_TRUNC != 0 && write {
        extra |= libc::O_TRUNC;
    }
    if flags & O_APPEND != 0 {
        extra |= libc::O_APPEND;
    }
    let file = fs::OpenOptions::new()
        .read(read)
        .write(write)
        .custom_flags(extra)
        .mode(0o666)
        .open(&host)
        .map_err(|e| {
            log::info!("[fs] open '{}' -> {} failed: {}", guest, host.display(), e);
            host_errno(&e)
        })?;
    if mutating {
        invalidate_negative_stat_ancestors(&guest);
        invalidate_apr_size(&host);
        record_written(&host);
    }
    let fd = alloc_fd(Handle::File(Arc::new(OpenFile::host(file, host.clone()))));
    set_open_cloexec(fd, flags);
    log::info!("[fs] open '{}' -> {} flags={:#x} fd={}", guest, host.display(), flags, fd);
    Ok(fd as i64)
}

/// open() inside the read-only `/app0` image.
fn open_image(guest: &str, flags: u64, mutating: bool, img: &Arc<App0Image>, rel: &str) -> FsResult<i64> {
    if mutating {
        log::info!("[fs] open '{}' flags={:#x} -> read-only image", guest, flags);
        return Err(EACCES);
    }
    let Some(id) = img.lookup(rel) else {
        log::info!("[fs] open '{}' -> not in the /app0 image", guest);
        return Err(ENOENT);
    };
    let shown = img.display_path(id);
    let handle = if img.node(id).is_dir {
        Handle::Dir(Arc::new(Mutex::new(OpenDir {
            guest: guest.to_string(),
            host: shown.clone(),
            dirents: img.directory_stream(id),
            offset: 0,
            image: Some((img.clone(), id)),
        })))
    } else {
        if flags & O_DIRECTORY != 0 {
            return Err(ENOTDIR);
        }
        let data = img.pfs.open_file(id).map_err(|e| {
            log::warn!("[fs] open '{}' -> image read failed: {e}", guest);
            EIO
        })?;
        Handle::File(Arc::new(OpenFile {
            file: FileSource::Image(data),
            host: shown.clone(),
            pos: Mutex::new(0),
            image_stat: Some(img.stat(id)),
        }))
    };
    let fd = alloc_fd(handle);
    set_open_cloexec(fd, flags);
    log::info!("[fs] open '{}' -> {} flags={:#x} fd={}", guest, shown.display(), flags, fd);
    Ok(fd as i64)
}

fn close_core(fd: i32) -> FsResult<i64> {
    if is_stdio(fd) {
        return Ok(0);
    }
    if crate::net::kernel_socket_close(fd) {
        return Ok(0);
    }
    if crate::memory::jit::close(fd) {
        return Ok(0);
    }
    if crate::sync::kqueue::close(fd) {
        crate::sync::equeue::descriptor_closed(fd);
        return Ok(0);
    }
    let removed = {
        let mut t = fds();
        t.cloexec.remove(&fd);
        t.status.remove(&fd);
        t.map.remove(&fd)
    };
    match removed {
        Some(_) => {
            // Read, write and vnode registrations die with the descriptor.
            crate::sync::equeue::descriptor_closed(fd);
            Ok(0)
        }
        None => Err(EBADF),
    }
}

/// True when `fd` names anything open: stdio, a kernel socket, a kqueue or
/// an entry of the descriptor table.
pub(crate) fn descriptor_open(fd: i32) -> bool {
    is_stdio(fd)
        || crate::net::is_kernel_socket(fd)
        || crate::sync::kqueue::is_kqueue(fd)
        || crate::memory::jit::is_jit(fd)
        || get_handle(fd).is_some()
}

fn read_core(ctx: &CpuContext, fd: i32, buf: u64, len: u64) -> FsResult<i64> {
    if len > i32::MAX as u64 {
        return Err(EINVAL);
    }
    let requested = len as usize;
    if requested > 0 && buf == 0 {
        return Err(EFAULT);
    }
    // A zero-length read transfers nothing but still needs an open descriptor.
    if requested == 0 {
        return if descriptor_open(fd) { Ok(0) } else { Err(EBADF) };
    }
    if fd == 0 {
        return Ok(0);
    }
    if let Some(r) = crate::net::kernel_socket_read(ctx, fd, buf, requested) {
        return r;
    }
    match get_handle(fd) {
        Some(Handle::Random) => return read_random(ctx, buf, requested),
        Some(Handle::Dir(d)) => {
            let mut d = d.lock().unwrap_or_else(|p| p.into_inner());
            return read_dir_stream(ctx, &mut d, buf, requested as u64);
        }
        Some(Handle::Pipe(p)) => return pipe::read(ctx, &p, buf, requested),
        _ => {}
    }
    let f = get_file(fd)?;
    let n = transfer_to_guest(ctx, buf, requested, |chunk, _| f.read_seq(chunk))?;
    Ok(n as i64)
}

/// Copy up to `count` stream bytes from the current position.
fn read_dir_stream(ctx: &CpuContext, d: &mut OpenDir, buf: u64, count: u64) -> FsResult<i64> {
    let len = d.dirents.len() as u64;
    let remaining = len.saturating_sub(d.offset);
    let n = count.min(remaining);
    if n != 0 {
        let start = d.offset as usize;
        write_guest(ctx, buf, &d.dirents[start..start + n as usize])?;
        d.offset += n;
    }
    Ok(n as i64)
}

/// Write raw bytes to a host stdio descriptor.
fn write_host_stdio(fd: i32, data: &[u8]) {
    let mut off = 0;
    while off < data.len() {
        let rc = unsafe { libc::write(fd, data[off..].as_ptr() as *const libc::c_void, data.len() - off) };
        if rc <= 0 {
            break;
        }
        off += rc as usize;
    }
}

/// Writes that synced their file because of O_FSYNC (for tests).
static SYNCHRONOUS_WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// After a write to file `f` through `fd`: with O_FSYNC set on the open
/// file (open flag or F_SETFL, shared by its duplicates) the write reaches
/// the disk before the call returns, as FreeBSD's IO_SYNC writes do.
fn sync_synchronous_write(fd: i32, f: &OpenFile) -> FsResult<()> {
    let synchronous = fds().status.get(&fd).is_some_and(|st| *st.lock().unwrap_or_else(|p| p.into_inner()) & O_FSYNC as u32 != 0);
    if !synchronous {
        return Ok(());
    }
    if let Some(file) = f.file.host() {
        file.sync_all().map_err(|e| host_errno(&e))?;
    }
    SYNCHRONOUS_WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

fn write_core(ctx: &CpuContext, fd: i32, buf: u64, len: u64) -> FsResult<i64> {
    let n = write_unsynced(ctx, fd, buf, len)?;
    if let Some(Handle::File(f)) = get_handle(fd) {
        sync_synchronous_write(fd, &f)?;
    }
    Ok(n)
}

/// write() without the O_FSYNC sync.
fn write_unsynced(ctx: &CpuContext, fd: i32, buf: u64, len: u64) -> FsResult<i64> {
    let requested = len.min(i32::MAX as u64) as usize;
    if requested > 0 && buf == 0 {
        return Err(EFAULT);
    }
    if fd == 1 || fd == 2 {
        let n = transfer_from_guest(ctx, buf, requested, |data, _| {
            write_host_stdio(fd, data);
            Ok(())
        })?;
        return Ok(n as i64);
    }
    if let Some(r) = crate::net::kernel_socket_write(ctx, fd, buf, requested) {
        return r;
    }
    let f = match get_handle(fd) {
        Some(Handle::File(f)) => f,
        // Writes to the random devices are accepted and discarded.
        Some(Handle::Random) => return Ok(requested as i64),
        Some(Handle::Pipe(p)) => return pipe::write(ctx, &p, buf, requested),
        _ => return Err(EBADF),
    };
    let n = transfer_from_guest(ctx, buf, requested, |data, _| f.file.write_all(data))?;
    Ok(n as i64)
}

fn pread_core(ctx: &CpuContext, fd: i32, buf: u64, len: u64, offset: i64) -> FsResult<i64> {
    let requested = len.min(i32::MAX as u64) as usize;
    if requested > 0 && buf == 0 {
        return Err(EFAULT);
    }
    if offset < 0 {
        return Err(EINVAL);
    }
    if let Some(Handle::Random) = get_handle(fd) {
        return read_random(ctx, buf, requested);
    }
    if crate::net::is_kernel_socket(fd) {
        return Err(ESPIPE);
    }
    let f = get_file(fd)?;
    if requested == 0 {
        return Ok(0);
    }
    let n = transfer_to_guest(ctx, buf, requested, |chunk, done| f.file.read_at(chunk, offset as u64 + done))?;
    Ok(n as i64)
}

/// Up to `len` bytes of file `fd` at `offset` (sendfile's source): EBADF
/// for a closed descriptor, EISDIR / ESPIPE / EINVAL as pread reports
/// them.
pub(crate) fn pread_bytes(fd: i32, len: usize, offset: u64) -> Result<Vec<u8>, i32> {
    if crate::net::is_kernel_socket(fd) {
        return Err(ESPIPE);
    }
    let f = get_file(fd)?;
    let mut buf = vec![0u8; len];
    let n = loop {
        match f.file.read_at(&mut buf, offset) {
            Ok(n) => break n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(host_errno(&e)),
        }
    };
    buf.truncate(n);
    Ok(buf)
}

fn pwrite_core(ctx: &CpuContext, fd: i32, buf: u64, len: u64, offset: i64) -> FsResult<i64> {
    let n = pwrite_unsynced(ctx, fd, buf, len, offset)?;
    sync_synchronous_write(fd, &*get_file(fd)?)?;
    Ok(n)
}

/// pwrite() without the O_FSYNC sync.
fn pwrite_unsynced(ctx: &CpuContext, fd: i32, buf: u64, len: u64, offset: i64) -> FsResult<i64> {
    let requested = len.min(i32::MAX as u64) as usize;
    if requested > 0 && buf == 0 {
        return Err(EFAULT);
    }
    if offset < 0 {
        return Err(EINVAL);
    }
    if crate::net::is_kernel_socket(fd) {
        return Err(ESPIPE);
    }
    let f = get_file(fd)?;
    if requested == 0 {
        return Ok(0);
    }
    let n = transfer_from_guest(ctx, buf, requested, |data, done| f.file.write_all_at(data, offset as u64 + done))?;
    Ok(n as i64)
}

/// Validate an iovec array: (base, len) pairs and their total.
fn read_iovecs(ctx: &CpuContext, iov: u64, iovcnt: i32, offset: i64) -> FsResult<(Vec<(u64, u64)>, u64)> {
    const MAX_IOV: i32 = 1024;
    if !(0..=MAX_IOV).contains(&iovcnt) || offset < 0 {
        return Err(EINVAL);
    }
    if iov == 0 && iovcnt != 0 {
        return Err(EFAULT);
    }
    let mut out = Vec::with_capacity(iovcnt as usize);
    let mut total = 0u64;
    for i in 0..iovcnt as u64 {
        let raw = read_guest(ctx, iov + i * 16, 16)?;
        let base = u64::from_le_bytes(raw[0..8].try_into().unwrap());
        let len = u64::from_le_bytes(raw[8..16].try_into().unwrap());
        if len > i32::MAX as u64 - total {
            return Err(EINVAL);
        }
        if base == 0 && len != 0 {
            return Err(EFAULT);
        }
        total += len;
        out.push((base, len));
    }
    Ok((out, total))
}

/// readv(fd, iov, iovcnt): scatter read from the current position. On a
/// kernel socket it is one recv() spread over the buffers, so a stream's
/// short read and a datagram's boundary are kept; on a file the buffers are
/// filled in order until one is not filled completely.
fn readv_core(ctx: &CpuContext, fd: i32, iov: u64, iovcnt: i32) -> FsResult<i64> {
    let (bufs, total) = read_iovecs(ctx, iov, iovcnt, 0)?;
    if crate::net::is_kernel_socket(fd) {
        if total == 0 {
            return Ok(0);
        }
        let data = crate::net::kernel_socket_recv_bytes(fd, total as usize).ok_or(EBADF)??;
        let mut copied = 0usize;
        for (base, len) in bufs {
            if copied == data.len() {
                break;
            }
            let n = (len as usize).min(data.len() - copied);
            write_guest(ctx, base, &data[copied..copied + n])?;
            copied += n;
        }
        return Ok(data.len() as i64);
    }
    if total == 0 {
        return if descriptor_open(fd) { Ok(0) } else { Err(EBADF) };
    }
    let mut done = 0u64;
    for (base, len) in bufs {
        if len == 0 {
            continue;
        }
        let n = match read_core(ctx, fd, base, len) {
            Ok(n) => n as u64,
            Err(e) if done == 0 => return Err(e),
            Err(_) => break,
        };
        done += n;
        if n < len {
            break;
        }
    }
    Ok(done as i64)
}

/// writev(fd, iov, iovcnt): gather write at the current position. On a
/// kernel socket it is one send() of the gathered bytes (one datagram).
fn writev_core(ctx: &CpuContext, fd: i32, iov: u64, iovcnt: i32) -> FsResult<i64> {
    let (bufs, total) = read_iovecs(ctx, iov, iovcnt, 0)?;
    if crate::net::is_kernel_socket(fd) {
        let mut data = Vec::with_capacity(total as usize);
        for (base, len) in bufs {
            if len != 0 {
                data.extend_from_slice(&read_guest(ctx, base, len as usize)?);
            }
        }
        return crate::net::kernel_socket_send_bytes(fd, &data).ok_or(EBADF)?;
    }
    if total == 0 {
        return if descriptor_open(fd) { Ok(0) } else { Err(EBADF) };
    }
    let mut done = 0u64;
    for (base, len) in bufs {
        if len == 0 {
            continue;
        }
        let n = match write_unsynced(ctx, fd, base, len) {
            Ok(n) => n as u64,
            Err(e) if done == 0 => return Err(e),
            Err(_) => break,
        };
        done += n;
        if n < len {
            break;
        }
    }
    // One sync for the whole gathered write.
    if let Some(Handle::File(f)) = get_handle(fd) {
        sync_synchronous_write(fd, &f)?;
    }
    Ok(done as i64)
}

/// preadv(fd, iov, iovcnt, offset): scatter read at `offset`, the file
/// position is unchanged.
fn preadv_core(ctx: &CpuContext, fd: i32, iov: u64, iovcnt: i32, offset: i64) -> FsResult<i64> {
    let (bufs, total) = read_iovecs(ctx, iov, iovcnt, offset)?;
    if crate::net::is_kernel_socket(fd) {
        return Err(ESPIPE);
    }
    let handle = get_handle(fd).ok_or(EBADF)?;
    if total == 0 {
        if let Handle::Dir(_) = handle {
            return Err(EISDIR);
        }
        return Ok(0);
    }
    let mut done = 0u64;
    for (base, len) in bufs {
        if len == 0 {
            continue;
        }
        let n = match &handle {
            Handle::Random => read_random(ctx, base, len as usize)? as u64,
            Handle::Dir(_) => return Err(EISDIR),
            Handle::Pipe(_) => return Err(ESPIPE),
            Handle::File(_) => pread_core(ctx, fd, base, len, offset + done as i64)? as u64,
        };
        done += n;
        if n < len {
            break;
        }
    }
    Ok(done as i64)
}

/// pwritev(fd, iov, iovcnt, offset): gather write at `offset`.
fn pwritev_core(ctx: &CpuContext, fd: i32, iov: u64, iovcnt: i32, offset: i64) -> FsResult<i64> {
    let (bufs, total) = read_iovecs(ctx, iov, iovcnt, offset)?;
    if crate::net::is_kernel_socket(fd) {
        return Err(ESPIPE);
    }
    match get_handle(fd).ok_or(EBADF)? {
        Handle::Dir(_) => return Err(EISDIR),
        Handle::Random => return Err(EINVAL),
        Handle::Pipe(_) => return Err(ESPIPE),
        Handle::File(_) => {}
    }
    if total == 0 {
        return Ok(0);
    }
    if offset as u64 > i64::MAX as u64 - total {
        return Err(EFBIG);
    }
    let mut done = 0u64;
    for (base, len) in bufs {
        if len == 0 {
            continue;
        }
        let n = pwrite_unsynced(ctx, fd, base, len, offset + done as i64)? as u64;
        done += n;
        if n < len {
            break;
        }
    }
    sync_synchronous_write(fd, &*get_file(fd)?)?;
    Ok(done as i64)
}

const LOCK_SH: i32 = 1;
const LOCK_EX: i32 = 2;
const LOCK_NB: i32 = 4;
const LOCK_UN: i32 = 8;

/// The lock a flock() operation asks for, decoded as FreeBSD does:
/// LOCK_UN wins over LOCK_EX, which wins over LOCK_SH; other bits are
/// ignored and an operation naming none of the three is EBADF.
fn flock_mode(operation: i32) -> FsResult<i32> {
    if operation & LOCK_UN != 0 {
        Ok(LOCK_UN)
    } else if operation & LOCK_EX != 0 {
        Ok(LOCK_EX)
    } else if operation & LOCK_SH != 0 {
        Ok(LOCK_SH)
    } else {
        Err(EBADF)
    }
}

/// flock(fd, operation): BSD advisory locks owned by the open file (shared
/// by its duplicates, released when the last of them closes). Host files
/// take the host lock; a contended LOCK_NB request is EWOULDBLOCK and a
/// blocking one waits in slices so signals and shutdown get through.
fn flock_core(fd: i32, operation: i32) -> FsResult<i64> {
    // The descriptor first (EBADF), then a pipe or socket (EOPNOTSUPP), then
    // the operation (as measured on a PS5).
    let handle = get_handle(fd);
    match &handle {
        Some(Handle::Pipe(_)) => return Err(EOPNOTSUPP),
        Some(_) => {}
        None if is_stdio(fd) => {}
        None if crate::net::is_kernel_socket(fd) => return Err(EOPNOTSUPP),
        None => return Err(EBADF),
    }
    let mode = flock_mode(operation)?;
    let file = match handle {
        Some(Handle::File(f)) => f,
        // Image files, directories, devices and the standard streams have no
        // host lock to take.
        _ => return Ok(0),
    };
    let Some(host) = file.file.host() else { return Ok(0) };
    let raw = crate::host::fs::as_raw_fd(host);
    let host_op = match mode {
        LOCK_UN => libc::LOCK_UN,
        LOCK_EX => libc::LOCK_EX,
        _ => libc::LOCK_SH,
    };
    loop {
        if unsafe { libc::flock(raw, host_op | libc::LOCK_NB) } == 0 {
            return Ok(0);
        }
        let e = io::Error::last_os_error();
        match e.raw_os_error() {
            Some(libc::EINTR) => continue,
            Some(libc::EWOULDBLOCK) if operation & LOCK_NB != 0 => return Err(EWOULDBLOCK),
            Some(libc::EWOULDBLOCK) => crate::signals::sleep(std::time::Duration::from_millis(1)),
            _ => return Err(host_errno(&e)),
        }
    }
}

fn lseek_core(fd: i32, offset: i64, whence: i32) -> FsResult<i64> {
    match get_handle(fd) {
        Some(Handle::File(f)) => {
            let base = match whence {
                SEEK_SET => 0,
                SEEK_CUR => f.position().map_err(|e| host_errno(&e))?,
                SEEK_END => f.file.len().map_err(|e| host_errno(&e))?,
                _ => return Err(EINVAL),
            };
            let pos = seek_position(base, offset)?;
            f.seek(SeekFrom::Start(pos)).map(|p| p as i64).map_err(|e| host_errno(&e))
        }
        Some(Handle::Dir(d)) => {
            let mut d = d.lock().unwrap_or_else(|p| p.into_inner());
            let base = match whence {
                SEEK_SET => 0,
                SEEK_CUR => d.offset,
                SEEK_END => d.dirents.len() as u64,
                _ => return Err(EINVAL),
            };
            d.offset = seek_position(base, offset)?;
            Ok(d.offset as i64)
        }
        Some(Handle::Random) | Some(Handle::Pipe(_)) => Err(ESPIPE),
        None => Err(EBADF),
    }
}

// ---------------------------------------------------------------------------
// stat
// ---------------------------------------------------------------------------

/// Size of the guest `struct stat` (FreeBSD amd64 layout).
pub const KERNEL_STAT_SIZE: usize = 120;
const STAT_MODE_DIRECTORY: u16 = 0o040777;
const STAT_MODE_REGULAR: u16 = 0o100777;
const STAT_MODE_CHARDEV: u16 = 0o020644;
const STAT_MODE_SYMLINK: u16 = 0o120777;

/// FNV-1a over UTF-8 bytes; used for inode and dirent file numbers.
pub fn fnv1a32(bytes: &[u8]) -> u32 {
    let mut h: u32 = 2_166_136_261;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(16_777_619);
    }
    h
}

fn timespec(t: SystemTime) -> (i64, i64) {
    match t.duration_since(UNIX_EPOCH) {
        Ok(d) => (d.as_secs() as i64, d.subsec_nanos() as i64),
        Err(e) => {
            let d = e.duration();
            if d.subsec_nanos() == 0 {
                (-(d.as_secs() as i64), 0)
            } else {
                (-(d.as_secs() as i64) - 1, 1_000_000_000 - d.subsec_nanos() as i64)
            }
        }
    }
}

/// Serialize a guest `struct stat`:
/// dev u32 @0, ino u32 @4, mode u16 @8, nlink u16 @10, uid @12, gid @16,
/// rdev @20, atim @24, mtim @40, ctim @56, size i64 @72, blocks i64 @80,
/// blksize u32 @88, flags @92, gen @96, lspare @100, birthtim @104.
pub fn build_kernel_stat(
    is_dir: bool,
    size: i64,
    atime: SystemTime,
    mtime: SystemTime,
    birth: SystemTime,
    inode_seed: &str,
) -> [u8; KERNEL_STAT_SIZE] {
    let mut b = [0u8; KERNEL_STAT_SIZE];
    let put_ts = |b: &mut [u8; KERNEL_STAT_SIZE], off: usize, t: SystemTime| {
        let (s, ns) = timespec(t);
        b[off..off + 8].copy_from_slice(&s.to_le_bytes());
        b[off + 8..off + 16].copy_from_slice(&ns.to_le_bytes());
    };
    b[4..8].copy_from_slice(&fnv1a32(inode_seed.as_bytes()).to_le_bytes());
    let mode = if is_dir { STAT_MODE_DIRECTORY } else { STAT_MODE_REGULAR };
    b[8..10].copy_from_slice(&mode.to_le_bytes());
    b[10..12].copy_from_slice(&1u16.to_le_bytes());
    put_ts(&mut b, 24, atime);
    put_ts(&mut b, 40, mtime);
    put_ts(&mut b, 56, mtime);
    b[72..80].copy_from_slice(&size.to_le_bytes());
    // Directory sizes are whole stream blocks.
    let blocks: i64 = if is_dir { size / 512 } else { (size + 511) / 512 };
    b[80..88].copy_from_slice(&blocks.to_le_bytes());
    b[88..92].copy_from_slice(&512u32.to_le_bytes());
    put_ts(&mut b, 104, birth);
    b
}

/// A character device: size 0, 512-byte blocks, timestamps of now.
fn device_stat(seed: &str) -> [u8; KERNEL_STAT_SIZE] {
    let now = SystemTime::now();
    let mut b = build_kernel_stat(false, 0, now, now, now, seed);
    b[8..10].copy_from_slice(&STAT_MODE_CHARDEV.to_le_bytes());
    b
}

/// Stat of a host file or directory; a directory's size is the length of
/// its packed dirent stream (`dir_size`, computed when None).
fn stat_from_metadata(m: &Metadata, host: &Path, dir_size: Option<u64>) -> [u8; KERNEL_STAT_SIZE] {
    let seed = &host.to_string_lossy();
    let is_dir = m.is_dir();
    let size = if is_dir {
        dir_size.unwrap_or_else(|| directory_stream(host).map_or(0, |d| d.len() as u64)) as i64
    } else {
        m.len() as i64
    };
    let mtime = m.modified().unwrap_or(UNIX_EPOCH);
    let atime = m.accessed().unwrap_or(mtime);
    let birth = m.created().unwrap_or(mtime);
    build_kernel_stat(is_dir, size, atime, mtime, birth, seed)
}

/// Metadata of an existing regular file or directory.
fn host_metadata(host: &Path) -> FsResult<Metadata> {
    let m = fs::metadata(host).map_err(|e| host_errno(&e))?;
    if m.is_dir() || m.is_file() {
        Ok(m)
    } else {
        Err(ENOENT)
    }
}

/// A path ending in '/' names a directory: anything else is ENOTDIR.
fn has_trailing_slash(guest: &str) -> bool {
    guest.len() > 1 && (guest.ends_with('/') || guest.ends_with('\\'))
}

fn stat_core(ctx: &CpuContext, path_addr: u64, stat_addr: u64) -> FsResult<i64> {
    if path_addr == 0 || stat_addr == 0 {
        return Err(EFAULT);
    }
    let guest = read_guest_path(ctx, path_addr)?;
    let slash = has_trailing_slash(&guest);
    if is_random_device(&guest) {
        write_guest(ctx, stat_addr, &device_stat(&guest))?;
        return Ok(0);
    }
    if let Some((img, rel)) = image_target(&guest) {
        let id = img.lookup(&rel).ok_or(ENOENT)?;
        if slash && !img.node(id).is_dir {
            return Err(ENOTDIR);
        }
        write_guest(ctx, stat_addr, &img.stat(id))?;
        return Ok(0);
    }
    let key = negative_stat_key(&guest);
    if key.as_deref().is_some_and(negative_stat_cached) {
        return Err(ENOENT);
    }
    let meta = resolve_guest_path(&guest).ok_or(ENOENT).and_then(|h| host_metadata(&h).map(|m| (h, m)));
    match meta {
        Ok((_, m)) if slash && !m.is_dir() => Err(ENOTDIR),
        Ok((host, m)) => {
            write_guest(ctx, stat_addr, &stat_from_metadata(&m, &host, None))?;
            if let Some(k) = key {
                with_state(|st| st.negative_stat.remove(&k));
            }
            log::debug!("[fs] stat '{}' -> {}", guest, host.display());
            Ok(0)
        }
        Err(e) => {
            if e == ENOENT {
                if let Some(k) = key {
                    with_state(|st| st.negative_stat.insert(k));
                }
            }
            log::debug!("[fs] stat '{}' -> errno {}", guest, e);
            Err(e)
        }
    }
}

fn fstat_core(ctx: &CpuContext, fd: i32, stat_addr: u64) -> FsResult<i64> {
    if stat_addr == 0 {
        return Err(EFAULT);
    }
    let bytes = match get_handle(fd) {
        Some(Handle::File(f)) => match (&f.file, f.image_stat) {
            (FileSource::Host(file), _) => {
                let m = file.metadata().map_err(|e| host_errno(&e))?;
                stat_from_metadata(&m, &f.host, None)
            }
            (FileSource::Image(_), stat) => stat.ok_or(EBADF)?,
        },
        Some(Handle::Dir(d)) => {
            let (host, size, image) = {
                let d = d.lock().unwrap_or_else(|p| p.into_inner());
                (d.host.clone(), d.dirents.len() as u64, d.image.clone())
            };
            match image {
                Some((img, id)) => img.stat(id),
                None => {
                    let m = host_metadata(&host)?;
                    stat_from_metadata(&m, &host, Some(size))
                }
            }
        }
        Some(Handle::Random) => device_stat(&format!("random:{fd}")),
        Some(Handle::Pipe(_)) => pipe::stat(fd),
        None if crate::net::is_kernel_socket(fd) => {
            // A socket: S_IFSOCK, the queued bytes and the send buffer
            // size; everything else zero.
            let st = crate::net::kernel_socket_stat(fd).ok_or(EBADF)?;
            let mut b = [0u8; KERNEL_STAT_SIZE];
            b[8..10].copy_from_slice(&st.mode.to_le_bytes());
            b[72..80].copy_from_slice(&st.size.to_le_bytes());
            b[88..92].copy_from_slice(&st.block_size.to_le_bytes());
            b
        }
        None if is_stdio(fd) => {
            let now = SystemTime::now();
            build_kernel_stat(false, 0, now, now, now, &format!("stdio:{fd}"))
        }
        None => return Err(EBADF),
    };
    write_guest(ctx, stat_addr, &bytes)?;
    Ok(0)
}

/// lstat(path, sb): like stat, but a symbolic link reports itself (type
/// S_IFLNK, size = length of the target) instead of what it points to.
fn lstat_core(ctx: &CpuContext, path_addr: u64, stat_addr: u64) -> FsResult<i64> {
    if path_addr == 0 || stat_addr == 0 {
        return Err(EFAULT);
    }
    let guest = read_guest_path(ctx, path_addr)?;
    // Devices and the read-only image hold no links; a trailing slash
    // follows the link.
    if guest.is_empty() || is_random_device(&guest) || in_app0_image(&guest) || has_trailing_slash(&guest) {
        return stat_core(ctx, path_addr, stat_addr);
    }
    let host = resolve_no_copy(&guest).ok_or(ENOENT)?;
    let m = fs::symlink_metadata(&host).map_err(|e| host_errno(&e))?;
    if !m.file_type().is_symlink() {
        return stat_core(ctx, path_addr, stat_addr);
    }
    let target = fs::read_link(&host).map_err(|e| host_errno(&e))?;
    let mtime = m.modified().unwrap_or(UNIX_EPOCH);
    let atime = m.accessed().unwrap_or(mtime);
    let birth = m.created().unwrap_or(mtime);
    let size = target.as_os_str().len() as i64;
    let mut b = build_kernel_stat(false, size, atime, mtime, birth, &host.to_string_lossy());
    b[8..10].copy_from_slice(&STAT_MODE_SYMLINK.to_le_bytes());
    write_guest(ctx, stat_addr, &b)?;
    Ok(0)
}

/// readlink(path, buf, size): the link target, truncated to `size`, not
/// NUL-terminated. EINVAL for something that is not a link.
fn readlink_core(ctx: &CpuContext, path_addr: u64, buf: u64, size: u64) -> FsResult<i64> {
    // The size is checked before the path (measured on a PS5).
    if size > i64::MAX as u64 {
        return Err(EINVAL);
    }
    if path_addr == 0 {
        return Err(EFAULT);
    }
    let guest = read_guest_path(ctx, path_addr)?;
    if guest.is_empty() {
        return Err(ENOENT);
    }
    if let Some((img, rel)) = image_target(&guest) {
        return Err(if img.lookup(&rel).is_some() { EINVAL } else { ENOENT });
    }
    let host = resolve_no_copy(&guest).ok_or(ENOENT)?;
    let m = fs::symlink_metadata(&host).map_err(|e| host_errno(&e))?;
    if !m.file_type().is_symlink() {
        return Err(EINVAL);
    }
    let target = fs::read_link(&host).map_err(|e| host_errno(&e))?;
    let bytes = target.as_os_str().as_encoded_bytes();
    let count = (size as usize).min(bytes.len());
    if count != 0 && buf == 0 {
        return Err(EFAULT);
    }
    write_guest(ctx, buf, &bytes[..count])?;
    Ok(count as i64)
}

/// FreeBSD `struct statfs` (version STATFS_VERSION, 472 bytes).
const KERNEL_STATFS_SIZE: usize = 472;
const STATFS_VERSION: u32 = 0x2003_0518;
const MNT_RDONLY: u64 = 0x1;
const MNT_NOEXEC: u64 = 0x4;
const MNT_NOSUID: u64 = 0x8;

/// Serialize the volume `vfs` describes: version @0, flags @8, bsize @16,
/// iosize @24, blocks @32, bfree @40, bavail @48, files @56, ffree @64,
/// namemax @184; the names stay empty.
fn build_statfs(vfs: &libc::statvfs, read_only: bool) -> [u8; KERNEL_STATFS_SIZE] {
    let mut b = [0u8; KERNEL_STATFS_SIZE];
    let mut flags = 0u64;
    if read_only || vfs.f_flag & libc::ST_RDONLY != 0 {
        flags |= MNT_RDONLY;
    }
    if vfs.f_flag & libc::ST_NOSUID != 0 {
        flags |= MNT_NOSUID;
    }
    if vfs.f_flag & libc::ST_NOEXEC != 0 {
        flags |= MNT_NOEXEC;
    }
    let bsize = if vfs.f_frsize != 0 { vfs.f_frsize } else { vfs.f_bsize } as u64;
    b[0..4].copy_from_slice(&STATFS_VERSION.to_le_bytes());
    b[8..16].copy_from_slice(&flags.to_le_bytes());
    b[16..24].copy_from_slice(&bsize.to_le_bytes());
    b[24..32].copy_from_slice(&(vfs.f_bsize as u64).to_le_bytes());
    b[32..40].copy_from_slice(&(vfs.f_blocks as u64).to_le_bytes());
    b[40..48].copy_from_slice(&(vfs.f_bfree as u64).to_le_bytes());
    b[48..56].copy_from_slice(&(vfs.f_bavail as i64).to_le_bytes());
    b[56..64].copy_from_slice(&(vfs.f_files as u64).to_le_bytes());
    b[64..72].copy_from_slice(&(vfs.f_favail as i64).to_le_bytes());
    b[184..188].copy_from_slice(&(vfs.f_namemax as u32).to_le_bytes());
    b
}

fn host_statvfs_path(path: &Path) -> FsResult<libc::statvfs> {
    let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| EINVAL)?;
    let mut vfs: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut vfs) } != 0 {
        return Err(host_errno(&io::Error::last_os_error()));
    }
    Ok(vfs)
}

fn host_statvfs_fd(fd: i32) -> FsResult<libc::statvfs> {
    let mut vfs: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstatvfs(fd, &mut vfs) } != 0 {
        return Err(host_errno(&io::Error::last_os_error()));
    }
    Ok(vfs)
}

/// _fstatfs(fd, buf): the volume holding the descriptor's file. Sockets
/// are EINVAL; nothing is written on failure. The `/app0` image reports
/// the volume of its host copy cache, read-only.
fn fstatfs_core(ctx: &CpuContext, fd: i32, buf: u64) -> FsResult<i64> {
    if crate::net::is_kernel_socket(fd) {
        return Err(EINVAL);
    }
    let image_cache = || with_state(|st| st.image.as_ref().map(|img| img.cache.clone())).ok_or(EBADF);
    let (vfs, read_only) = match get_handle(fd) {
        Some(Handle::File(f)) => match f.file.host() {
            Some(file) => (host_statvfs_fd(crate::host::fs::as_raw_fd(file))?, is_read_only("", Some(&f.host))),
            None => (host_statvfs_path(&image_cache()?)?, true),
        },
        Some(Handle::Dir(d)) => {
            let (host, image) = {
                let d = d.lock().unwrap_or_else(|p| p.into_inner());
                (d.host.clone(), d.image.is_some())
            };
            if image {
                (host_statvfs_path(&image_cache()?)?, true)
            } else {
                (host_statvfs_path(&host)?, is_read_only("", Some(&host)))
            }
        }
        Some(Handle::Random) => (host_statvfs_path(Path::new("/dev"))?, false),
        // A pipe lives on no mounted file system.
        Some(Handle::Pipe(_)) => return Err(EINVAL),
        None if is_stdio(fd) => {
            let mut st: libc::stat = unsafe { std::mem::zeroed() };
            if unsafe { libc::fstat(fd, &mut st) } != 0 {
                return Err(EBADF);
            }
            // Pipes and sockets live on no volume.
            let kind = st.st_mode & libc::S_IFMT;
            if kind == libc::S_IFIFO || kind == libc::S_IFSOCK {
                return Err(EINVAL);
            }
            (host_statvfs_fd(fd)?, false)
        }
        None => return Err(EBADF),
    };
    if buf == 0 {
        return Err(EFAULT);
    }
    write_guest(ctx, buf, &build_statfs(&vfs, read_only))?;
    Ok(0)
}

// ---------------------------------------------------------------------------
// Directory entries
// ---------------------------------------------------------------------------

const DT_DIR: u8 = 4;
const DT_REG: u8 = 8;

/// getdirentries(fd, buf, nbytes, basep): whole 512-byte blocks from a
/// block-aligned stream position; `*basep` gets that position.
fn getdirentries_core(ctx: &CpuContext, fd: i32, buf: u64, nbytes: u64, basep: u64) -> FsResult<i64> {
    let nbytes = nbytes as i32;
    let dir = match get_handle(fd) {
        Some(Handle::Dir(d)) => Some(d),
        Some(_) => None,
        None => return Err(EBADF),
    };
    if buf == 0 {
        return Err(EFAULT);
    }
    let Some(dir) = dir else { return Err(EINVAL) };
    if nbytes < DIR_BLOCK_SIZE as i32 {
        return Err(EINVAL);
    }
    let mut d = dir.lock().unwrap_or_else(|p| p.into_inner());
    if d.offset > d.dirents.len() as u64 || d.offset % DIR_BLOCK_SIZE != 0 {
        return Err(EINVAL);
    }
    if basep != 0 {
        write_u64(ctx, basep, d.offset)?;
    }
    read_dir_stream(ctx, &mut d, buf, nbytes as u64 & !(DIR_BLOCK_SIZE - 1))
}

// ---------------------------------------------------------------------------
// Path mutations
// ---------------------------------------------------------------------------

/// A guest path argument: NULL is EFAULT (an empty path then fails to
/// resolve, ENOENT).
fn guest_path_arg(ctx: &CpuContext, addr: u64) -> FsResult<String> {
    read_guest_path(ctx, addr)
}

/// FreeBSD FIONBIO (`_IOW('f', 126, int)`), FIONREAD (`_IOR('f', 127,
/// int)`), FIOCLEX and FIONCLEX (`_IO('f', 1)`, `_IO('f', 2)`).
const FIONBIO: u32 = 0x8004_667E;
const FIONREAD: u32 = 0x4004_667F;
const FIOCLEX: u32 = 0x2000_6601;
const FIONCLEX: u32 = 0x2000_6602;
/// IOC_IN: the request carries data in from the caller.
const IOC_IN: u32 = 0x8000_0000;
/// The terminal ioctl group ('t').
const TERMINAL_GROUP: u32 = b't' as u32;
const ENOTTY: i32 = 25;

/// A request the console's answer is unknown for: logged, ENOTTY.
fn ioctl_unsupported(fd: i32, request: u32, what: &str) -> FsResult<i64> {
    log::error!("[fs] ioctl {request:#010x} on {what} (fd {fd}) is not supported");
    Err(ENOTTY)
}

/// Change the status flags of table entry `fd` under `mask` to `value`
/// (F_SETFL, FIONBIO). O_NONBLOCK and O_APPEND reach the host descriptor
/// of files and pipes; O_FSYNC makes every later write to a file sync it
/// ([`sync_synchronous_write`]); O_ASYNC on a pipe or device is not
/// supported (logged, EINVAL).
fn change_status(fd: i32, mask: u32, value: u32) -> FsResult<i64> {
    let (handle, status) = {
        let t = fds();
        (t.map.get(&fd).cloned().ok_or(EBADF)?, t.status.get(&fd).cloned().ok_or(EBADF)?)
    };
    let mut st = status.lock().unwrap_or_else(|p| p.into_inner());
    let next = (*st & !mask) | (value & mask);
    let changed = *st ^ next;
    let file = matches!(handle, Handle::File(_));
    if changed & O_ASYNC as u32 != 0 && !file && !matches!(handle, Handle::Dir(_)) {
        log::error!("[fs] fcntl: O_ASYNC on fd {fd} (not a file) is not supported");
        return Err(EINVAL);
    }
    let host = match &handle {
        Handle::File(f) => f.file.host().map(crate::host::fs::as_raw_fd),
        Handle::Pipe(p) => Some(p.raw()),
        _ => None,
    };
    if let Some(raw) = host.filter(|_| changed & (O_NONBLOCK_ | O_APPEND) as u32 != 0) {
        // SAFETY: F_GETFL / F_SETFL on a descriptor the handle keeps open.
        let flags = unsafe { libc::fcntl(raw, libc::F_GETFL, 0) };
        let mut wanted = flags & !(libc::O_NONBLOCK | libc::O_APPEND);
        if next & O_NONBLOCK_ as u32 != 0 {
            wanted |= libc::O_NONBLOCK;
        }
        if next & O_APPEND as u32 != 0 {
            wanted |= libc::O_APPEND;
        }
        if flags < 0 || unsafe { libc::fcntl(raw, libc::F_SETFL, wanted) } < 0 {
            return Err(host_errno(&io::Error::last_os_error()));
        }
    }
    *st = next;
    Ok(0)
}

/// ioctl(fd, request, arg) / _ioctl. A request carrying data with a NULL
/// argument is EFAULT, then a descriptor that is not open EBADF.
/// FIOCLEX / FIONCLEX set or clear FD_CLOEXEC on any descriptor. FIONBIO
/// sets or clears non-blocking mode (sockets, files, pipes, devices;
/// accepted without effect on JIT shared memory). FIONREAD
/// stores the readable byte count: queued bytes of a socket or pipe read
/// end (0 for a write end), the rest of a file; a random device is
/// ENOTTY. Terminal requests are ENOTTY; other requests are not supported
/// (logged, ENOTTY).
fn ioctl_core(ctx: &CpuContext, fd: i32, request: u64, arg: u64) -> FsResult<i64> {
    const IOC_OUT: u32 = 0x4000_0000;
    const IOC_VOID: u32 = 0x2000_0000;
    let request = request as u32;
    let length = (request >> 16) & 0x1FFF;
    // A request FreeBSD's ioctl cannot decode is ENOTTY before the
    // descriptor is looked at: no direction, an out request with no size,
    // or a void request with a size other than an int.
    if request & (IOC_IN | IOC_OUT | IOC_VOID) == 0
        || (request & IOC_OUT != 0 && length == 0)
        || (request & IOC_VOID != 0 && length != 0 && length != 4)
    {
        return Err(ENOTTY);
    }
    if request & IOC_IN != 0 && length != 0 && arg == 0 {
        return Err(EFAULT);
    }
    if fd < 0 || !descriptor_open(fd) {
        return Err(EBADF);
    }
    if request == FIOCLEX || request == FIONCLEX {
        return if set_cloexec(fd, request == FIOCLEX) { Ok(0) } else { Err(EBADF) };
    }
    if (request >> 8) & 0xFF == TERMINAL_GROUP {
        return Err(ENOTTY);
    }
    if request != FIONBIO && request != FIONREAD {
        return ioctl_unsupported(fd, request, "a descriptor");
    }
    let socket = crate::net::is_kernel_socket(fd);
    let handle = get_handle(fd);
    if request == FIONBIO {
        let on = ctx.read_u32(arg).ok_or(EFAULT)? != 0;
        if socket {
            return crate::net::kernel_socket_set_nonblocking(fd, on).ok_or(EBADF)?.map(|()| 0);
        }
        if handle.is_some() {
            return change_status(fd, O_NONBLOCK_ as u32, if on { O_NONBLOCK_ as u32 } else { 0 });
        }
        if crate::memory::jit::is_jit(fd) {
            return Ok(0);
        }
        return ioctl_unsupported(fd, request, if crate::sync::kqueue::is_kqueue(fd) { "a kqueue" } else { "a standard stream" });
    }
    let queued = match handle {
        _ if socket => crate::net::kernel_socket_queued(fd).ok_or(EBADF)??,
        Some(Handle::File(f)) => {
            let len = match (f.file.host(), f.image_stat) {
                (Some(file), _) => file.metadata().map_err(|e| host_errno(&e))?.len(),
                (None, Some(stat)) => u64::from_le_bytes(stat[72..80].try_into().unwrap()),
                (None, None) => 0,
            };
            let pos = lseek_core(fd, 0, SEEK_CUR)? as u64;
            len.saturating_sub(pos).min(i32::MAX as u64) as i64
        }
        Some(Handle::Pipe(p)) if p.write => 0,
        Some(Handle::Pipe(p)) => {
            let mut n: libc::c_int = 0;
            // SAFETY: FIONREAD stores one int.
            if unsafe { libc::ioctl(p.raw(), libc::FIONREAD, &mut n) } < 0 {
                return Err(host_errno(&io::Error::last_os_error()));
            }
            n as i64
        }
        Some(Handle::Random) => return Err(ENOTTY),
        Some(Handle::Dir(_)) => return ioctl_unsupported(fd, request, "a directory"),
        None if crate::sync::kqueue::is_kqueue(fd) || crate::memory::jit::is_jit(fd) => return Err(ENOTTY),
        None => return ioctl_unsupported(fd, request, "a standard stream"),
    };
    if arg == 0 {
        return Err(EFAULT);
    }
    if ctx.write_u32(arg, queued as u32) { Ok(0) } else { Err(EFAULT) }
}

/// link(existing, new): a hard link. NULL paths are EFAULT, empty ones
/// ENOENT; a missing source ENOENT, a directory source EPERM, an existing
/// destination EEXIST, a missing destination directory ENOENT, another
/// volume EXDEV.
fn link_core(ctx: &CpuContext, existing: u64, created: u64) -> FsResult<i64> {
    if existing == 0 || created == 0 {
        return Err(EFAULT);
    }
    let (from, to) = (read_guest_path(ctx, existing)?, read_guest_path(ctx, created)?);
    if from.is_empty() || to.is_empty() {
        return Err(ENOENT);
    }
    let (from_host, to_host) = (resolve_no_copy(&from), resolve_no_copy(&to));
    if is_read_only(&to, to_host.as_deref()) {
        return Err(EACCES);
    }
    let (from_host, to_host) = (from_host.ok_or(ENOENT)?, to_host.ok_or(ENOENT)?);
    match fs::symlink_metadata(&from_host) {
        Ok(m) if m.is_dir() => return Err(EPERM),
        Ok(_) => {}
        Err(e) => return Err(host_errno(&e)),
    }
    if fs::symlink_metadata(&to_host).is_ok() {
        return Err(EEXIST);
    }
    fs::hard_link(&from_host, &to_host).map_err(|e| host_errno(&e))?;
    invalidate_negative_stat_ancestors(&to);
    record_written(&to_host);
    Ok(0)
}

/// symlink(target, new): a symbolic link holding `target` as written.
/// NULL paths are EFAULT, empty ones ENOENT, an existing name EEXIST, a
/// missing directory ENOENT. A relative target is stored as it is; an
/// absolute guest target has no host meaning, so it is EPERM (a file
/// system without that kind of link).
fn symlink_core(ctx: &CpuContext, target_addr: u64, created: u64) -> FsResult<i64> {
    if target_addr == 0 || created == 0 {
        return Err(EFAULT);
    }
    // The target is stored as written, not looked up.
    let target = read_guest_cstr(ctx, target_addr, MAX_GUEST_PATH).map(|b| String::from_utf8_lossy(&b).into_owned())?;
    let to = read_guest_path(ctx, created)?;
    if target.is_empty() || to.is_empty() {
        return Err(ENOENT);
    }
    if target.starts_with('/') {
        log::error!("[fs] symlink('{target}', '{to}'): absolute targets are not supported");
        return Err(EPERM);
    }
    // A `..` component could lead the host out of the guest's mount.
    if target.split('/').any(|c| c == "..") {
        log::error!("[fs] symlink('{target}', '{to}'): parent-directory components are not supported");
        return Err(EPERM);
    }
    let to_host = resolve_no_copy(&to);
    if is_read_only(&to, to_host.as_deref()) {
        return Err(EACCES);
    }
    let to_host = to_host.ok_or(ENOENT)?;
    if fs::symlink_metadata(&to_host).is_ok() {
        return Err(EEXIST);
    }
    #[cfg(unix)]
    std::os::unix::fs::symlink(&target, &to_host).map_err(|e| host_errno(&e))?;
    #[cfg(windows)]
    std::os::windows::fs::symlink_file(&target, &to_host).map_err(|e| host_errno(&e))?;
    invalidate_negative_stat_ancestors(&to);
    record_written(&to_host);
    Ok(0)
}

fn unlink_core(ctx: &CpuContext, path_addr: u64) -> FsResult<i64> {
    let guest = guest_path_arg(ctx, path_addr)?;
    let host = resolve_no_copy(&guest);
    if is_read_only(&guest, host.as_deref()) {
        return Err(EACCES);
    }
    let host = host.ok_or(ENOENT)?;
    // unlink removes a symbolic link itself, never what it points to, and
    // refuses a directory with EPERM.
    match fs::symlink_metadata(&host) {
        Ok(m) if m.is_dir() => return Err(EPERM),
        Ok(_) => {}
        Err(e) => {
            let errno = host_errno(&e);
            if errno == ENOENT {
                add_negative_stat(&guest);
            }
            return Err(errno);
        }
    }
    fs::remove_file(&host).map_err(|e| host_errno(&e))?;
    invalidate_negative_stat_ancestors(&guest);
    invalidate_apr_size(&host);
    add_negative_stat(&guest);
    record_written(&host);
    log::info!("[fs] unlink '{}' -> {}", guest, host.display());
    Ok(0)
}

fn mkdir_core(ctx: &CpuContext, path_addr: u64) -> FsResult<i64> {
    let guest = guest_path_arg(ctx, path_addr)?;
    if guest.is_empty() {
        return Err(ENOENT);
    }
    if guest == "/" {
        return Err(EEXIST);
    }
    let host = resolve_no_copy(&guest);
    if is_read_only(&guest, host.as_deref()) {
        return Err(EACCES);
    }
    let host = host.ok_or(ENOENT)?;
    // An existing entry is EEXIST; a lookup that fails other than with
    // "not found" (a file used as a directory, ...) reports its error.
    match fs::metadata(&host) {
        Ok(_) => return Err(EEXIST),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(host_errno(&e)),
    }
    match host.parent().map(fs::metadata) {
        Some(Ok(m)) if m.is_dir() => {}
        Some(Ok(_)) => return Err(ENOTDIR),
        Some(Err(e)) => return Err(host_errno(&e)),
        None => return Err(ENOENT),
    }
    fs::create_dir(&host).map_err(|e| host_errno(&e))?;
    invalidate_negative_stat_ancestors(&guest);
    record_written(&host);
    log::info!("[fs] mkdir '{}' -> {}", guest, host.display());
    Ok(0)
}

/// The last component of `path`, trailing separators ignored, is ".".
fn final_component_is_dot(path: &str) -> bool {
    let trimmed = path.trim_end_matches('/');
    trimmed.rsplit('/').next() == Some(".")
}

fn rmdir_core(ctx: &CpuContext, path_addr: u64) -> FsResult<i64> {
    let guest = guest_path_arg(ctx, path_addr)?;
    let host = resolve_no_copy(&guest);
    if is_read_only(&guest, host.as_deref()) {
        return Err(EACCES);
    }
    let host = host.ok_or(ENOENT)?;
    // A file used as a directory anywhere in the path is ENOTDIR, a
    // missing entry ENOENT (the host lookup tells them apart).
    match fs::metadata(&host) {
        Ok(m) if m.is_dir() => {}
        Ok(_) => return Err(ENOTDIR),
        Err(e) => {
            let errno = host_errno(&e);
            if errno == ENOENT {
                add_negative_stat(&guest);
            }
            return Err(errno);
        }
    }
    // FreeBSD refuses to remove a directory named by a final ".".
    if final_component_is_dot(&guest) {
        return Err(EINVAL);
    }
    fs::remove_dir(&host).map_err(|e| host_errno(&e))?;
    invalidate_negative_stat_ancestors(&guest);
    add_negative_stat(&guest);
    record_written(&host);
    Ok(0)
}

/// rename(from, to) for both the libkernel and the libc exports: NULL
/// paths are EFAULT, empty ones ENOENT, host failures keep their errno.
pub(crate) fn rename_core(ctx: &CpuContext, from_addr: u64, to_addr: u64) -> FsResult<i64> {
    // The console's order (measured on a PS5): the source argument, a
    // source ending in a dot component, then the destination argument; a
    // destination error is reported only for a source that exists.
    let from_raw = read_guest_cstr(ctx, from_addr, MAX_GUEST_PATH)?;
    if from_raw.is_empty() {
        return Err(ENOENT);
    }
    if let Some(e) = rename_dot_error(from_addr, &from_raw) {
        return Err(e);
    }
    let to_error = match read_guest_cstr(ctx, to_addr, MAX_GUEST_PATH) {
        Err(e) => Some(e),
        Ok(raw) if raw.is_empty() => Some(ENOENT),
        Ok(raw) => rename_dot_error(to_addr, &raw),
    };
    let from = read_guest_path(ctx, from_addr)?;
    if let Some(e) = to_error {
        let host = resolve_no_copy(&from).ok_or(ENOENT)?;
        fs::symlink_metadata(&host).map_err(|e| host_errno(&e))?;
        return Err(e);
    }
    let to = read_guest_path(ctx, to_addr)?;
    let from_host = resolve_no_copy(&from);
    let to_host = resolve_no_copy(&to);
    if is_read_only(&from, from_host.as_deref()) || is_read_only(&to, to_host.as_deref()) {
        return Err(EACCES);
    }
    let (from_host, to_host) = (from_host.ok_or(ENOENT)?, to_host.ok_or(ENOENT)?);
    // Linux reports a non-empty target directory as EEXIST on some file
    // systems (xfs); the console's kernel always reports ENOTEMPTY.
    fs::rename(&from_host, &to_host).map_err(|e| match host_errno(&e) {
        EEXIST => ENOTEMPTY,
        errno => errno,
    })?;
    invalidate_negative_stat_ancestors(&to);
    add_negative_stat(&from);
    invalidate_apr_size(&from_host);
    invalidate_apr_size(&to_host);
    record_written(&from_host);
    record_written(&to_host);
    log::info!("[fs] rename '{}' -> '{}'", from, to);
    Ok(0)
}

/// POSIX rename (shared by the libkernel and libc exports): -1 and errno
/// on failure.
pub(crate) fn posix_rename(ctx: &CpuContext, from_addr: u64, to_addr: u64) -> HleResult {
    let r = rename_core(ctx, from_addr, to_addr);
    posix(ctx, r)
}

fn truncate_core(ctx: &CpuContext, path_addr: u64, length: i64) -> FsResult<i64> {
    if length < 0 {
        return Err(EINVAL);
    }
    let guest = guest_path_arg(ctx, path_addr)?;
    let host = resolve_no_copy(&guest);
    if is_read_only(&guest, host.as_deref()) {
        return Err(EACCES);
    }
    let host = host.ok_or(ENOENT)?;
    let f = fs::OpenOptions::new().write(true).open(&host).map_err(|e| host_errno(&e))?;
    f.set_len(length as u64).map_err(|e| host_errno(&e))?;
    invalidate_apr_size(&host);
    record_written(&host);
    Ok(0)
}

/// ftruncate(fd, length): EINVAL for a negative length, a descriptor not
/// open for writing, a directory, a device or a socket; EBADF if closed.
fn ftruncate_core(fd: i32, length: i64) -> FsResult<i64> {
    if length < 0 {
        return Err(EINVAL);
    }
    let f = match get_handle(fd) {
        Some(Handle::File(f)) => f,
        Some(_) => return Err(EINVAL),
        None if crate::net::is_kernel_socket(fd) => return Err(EINVAL),
        None => return Err(EBADF),
    };
    // Without write access the host reports EINVAL, as FreeBSD does.
    f.file.host().ok_or(EINVAL)?.set_len(length as u64).map_err(|e| host_errno(&e))?;
    invalidate_apr_size(&f.host);
    record_written(&f.host);
    Ok(0)
}

fn fsync_core(fd: i32) -> FsResult<i64> {
    match get_handle(fd) {
        // A host flush failure is EIO.
        Some(Handle::File(f)) => match f.file.host() {
            Some(file) => file.sync_all().map(|_| 0).map_err(|_| EIO),
            None => Ok(0),
        },
        Some(Handle::Dir(_)) | Some(Handle::Random) => Ok(0),
        // Like a socket, a pipe has no vnode to sync.
        Some(Handle::Pipe(_)) => Err(EINVAL),
        None if is_stdio(fd) => Ok(0),
        None if crate::net::is_kernel_socket(fd) => Err(EINVAL),
        None => Err(EBADF),
    }
}

/// fdatasync(fd): flush the file's data; a host failure is reported.
/// Sockets are EINVAL, closed descriptors EBADF.
fn fdatasync_core(fd: i32) -> FsResult<i64> {
    match get_handle(fd) {
        Some(Handle::File(f)) => match f.file.host() {
            Some(file) => file.sync_data().map(|_| 0).map_err(|e| host_errno(&e)),
            None => Ok(0),
        },
        Some(Handle::Dir(_)) | Some(Handle::Random) => Ok(0),
        Some(Handle::Pipe(_)) => Err(EINVAL),
        None if is_stdio(fd) => Ok(0),
        None if crate::net::is_kernel_socket(fd) => Err(EINVAL),
        None => Err(EBADF),
    }
}

// ---------------------------------------------------------------------------
// sync: paths written by the guest
// ---------------------------------------------------------------------------

/// Host paths the guest created, opened for writing, renamed, truncated or
/// removed since the last sync.
static WRITTEN_PATHS: LazyLock<Mutex<HashSet<PathBuf>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

fn record_written(host: &Path) {
    WRITTEN_PATHS.lock().unwrap_or_else(|p| p.into_inner()).insert(host.to_path_buf());
}

/// Flush one host path (file or directory); a path that is gone or cannot
/// be opened has nothing left to flush.
fn flush_host_path(path: &Path) {
    if let Ok(f) = fs::OpenOptions::new().read(true).custom_flags(libc::O_NONBLOCK).open(path) {
        let _ = f.sync_all();
    }
}

/// sync() / sceKernelSync(): flush the open host files and every path the
/// guest changed since the last sync together with its parent directory,
/// instead of the whole host (which can stall on unrelated I/O).
pub(crate) fn sync_written_paths() {
    let files: Vec<Arc<OpenFile>> = fds()
        .map
        .values()
        .filter_map(|h| match h {
            Handle::File(f) => Some(f.clone()),
            _ => None,
        })
        .collect();
    for f in &files {
        if let Some(file) = f.file.host() {
            let _ = file.sync_all();
        }
    }
    let paths = std::mem::take(&mut *WRITTEN_PATHS.lock().unwrap_or_else(|p| p.into_inner()));
    let mut flushed = HashSet::new();
    for path in paths {
        if flushed.insert(path.clone()) {
            flush_host_path(&path);
        }
        if let Some(parent) = path.parent() {
            if flushed.insert(parent.to_path_buf()) {
                flush_host_path(parent);
            }
        }
    }
}

fn chmod_core(ctx: &CpuContext, path_addr: u64, mode: u32) -> FsResult<i64> {
    let guest = guest_path_arg(ctx, path_addr)?;
    let host = resolve_no_copy(&guest);
    if is_read_only(&guest, host.as_deref()) {
        return Err(EACCES);
    }
    let host = host.ok_or(ENOENT)?;
    if !host.exists() {
        add_negative_stat(&guest);
        return Err(ENOENT);
    }
    // Guest permission bits are accepted but not applied to host files.
    log::debug!("[fs] chmod '{}' mode={:#o}", guest, mode);
    Ok(0)
}

fn fchmod_core(fd: i32, mode: u32) -> FsResult<i64> {
    let (host, image) = match get_handle(fd) {
        Some(Handle::File(f)) => (f.host.clone(), f.file.is_image()),
        Some(Handle::Dir(d)) => {
            let d = d.lock().unwrap_or_else(|p| p.into_inner());
            (d.host.clone(), d.image.is_some())
        }
        // Device nodes keep their fixed permissions.
        Some(Handle::Random) => return Ok(0),
        Some(Handle::Pipe(_)) => return Err(EINVAL),
        None if crate::net::is_kernel_socket(fd) => return Err(EINVAL),
        None => return Err(EBADF),
    };
    if image || is_read_only("", Some(&host)) {
        return Err(EROFS);
    }
    // Guest permission bits are accepted but not applied to host files.
    log::debug!("[fs] fchmod fd={} mode={:#o}", fd, mode);
    Ok(0)
}

/// Read a guest `struct timeval[2]` (access, modification) into host form;
/// a NULL pointer means "now" for both.
fn read_timevals(ctx: &CpuContext, times_addr: u64) -> FsResult<Option<[libc::timeval; 2]>> {
    if times_addr == 0 {
        return Ok(None);
    }
    let mut tv = [libc::timeval { tv_sec: 0, tv_usec: 0 }; 2];
    for (i, t) in tv.iter_mut().enumerate() {
        let base = times_addr + i as u64 * 16;
        let sec = ctx.read_u64(base).ok_or(EFAULT)? as i64;
        let usec = ctx.read_u64(base + 8).ok_or(EFAULT)? as i64;
        if !(0..1_000_000).contains(&usec) {
            return Err(EINVAL);
        }
        *t = libc::timeval { tv_sec: sec, tv_usec: usec };
    }
    Ok(Some(tv))
}

/// Apply `utimes` to a host path.
fn host_utimes(host: &Path, tv: Option<&[libc::timeval; 2]>) -> FsResult<i64> {
    let c = std::ffi::CString::new(host.as_os_str().as_encoded_bytes()).map_err(|_| EINVAL)?;
    let p = tv.map_or(std::ptr::null(), |t| t.as_ptr());
    if unsafe { libc::utimes(c.as_ptr(), p) } != 0 {
        return Err(host_errno(&io::Error::last_os_error()));
    }
    Ok(0)
}

/// utimes(path, const timeval times[2]): NULL times means "now".
fn utimes_core(ctx: &CpuContext, path_addr: u64, times_addr: u64) -> FsResult<i64> {
    // The times are checked before the path (measured on a PS5).
    let tv = read_timevals(ctx, times_addr)?;
    let host = utimes_target(ctx, path_addr)?;
    host_utimes(&host, tv.as_ref())
}

/// The writable host path a utimes-style call changes.
fn utimes_target(ctx: &CpuContext, path_addr: u64) -> FsResult<PathBuf> {
    let guest = guest_path_arg(ctx, path_addr)?;
    let host = resolve_no_copy(&guest);
    if is_read_only(&guest, host.as_deref()) {
        return Err(EROFS);
    }
    let host = host.ok_or(ENOENT)?;
    if !host.exists() {
        return Err(ENOENT);
    }
    Ok(host)
}

/// utime(path, const struct utimbuf* times): utimbuf is {time_t actime,
/// time_t modtime}, whole seconds; NULL times means "now". As utimes with
/// zero microseconds.
fn utime_core(ctx: &CpuContext, path_addr: u64, times_addr: u64) -> FsResult<i64> {
    let host = utimes_target(ctx, path_addr)?;
    let tv = if times_addr == 0 {
        None
    } else {
        let actime = ctx.read_u64(times_addr).ok_or(EFAULT)? as i64;
        let modtime = ctx.read_u64(times_addr + 8).ok_or(EFAULT)? as i64;
        Some([libc::timeval { tv_sec: actime, tv_usec: 0 }, libc::timeval { tv_sec: modtime, tv_usec: 0 }])
    };
    host_utimes(&host, tv.as_ref())
}

/// futimes(fd, const timeval times[2]): NULL times means "now".
fn futimes_core(ctx: &CpuContext, fd: i32, times_addr: u64) -> FsResult<i64> {
    let tv = read_timevals(ctx, times_addr)?;
    match get_handle(fd) {
        Some(Handle::File(f)) => {
            let Some(file) = f.file.host().filter(|_| !is_read_only("", Some(&f.host))) else {
                return Err(EROFS);
            };
            let p = tv.as_ref().map_or(std::ptr::null(), |t| t.as_ptr());
            if unsafe { libc::futimes(crate::host::fs::as_raw_fd(file), p) } != 0 {
                return Err(host_errno(&io::Error::last_os_error()));
            }
            Ok(0)
        }
        Some(Handle::Dir(d)) => {
            let (host, image) = {
                let d = d.lock().unwrap_or_else(|p| p.into_inner());
                (d.host.clone(), d.image.is_some())
            };
            if image || is_read_only("", Some(&host)) {
                return Err(EROFS);
            }
            host_utimes(&host, tv.as_ref())
        }
        // Device timestamps are not tracked.
        Some(Handle::Random) => Ok(0),
        Some(Handle::Pipe(_)) => Err(EINVAL),
        None if crate::net::is_kernel_socket(fd) => Err(EINVAL),
        None => Err(EBADF),
    }
}

static REACHABILITY_MISSES: AtomicU64 = AtomicU64::new(0);

fn check_reachability_core(ctx: &CpuContext, path_addr: u64) -> FsResult<i64> {
    if path_addr == 0 {
        return Err(EINVAL);
    }
    let guest = guest_path_arg(ctx, path_addr)?;
    if is_random_device(&guest) {
        return Ok(0);
    }
    if let Some((img, rel)) = image_target(&guest) {
        return img.lookup(&rel).map(|_| 0).ok_or(ENOENT);
    }
    let host = resolve_guest_path(&guest);
    if host.as_deref().is_some_and(Path::exists) {
        return Ok(0);
    }
    let miss = REACHABILITY_MISSES.fetch_add(1, Ordering::Relaxed) + 1;
    if miss <= 16 || miss.is_power_of_two() {
        log::info!(
            "[fs] check_reachability miss #{} guest='{}' host={:?}",
            miss,
            guest,
            host
        );
    }
    Err(ENOENT)
}

// ---------------------------------------------------------------------------
// Descriptor duplication, fcntl, poll/select
// ---------------------------------------------------------------------------

const F_DUPFD: u64 = 0;
const F_GETFD: u64 = 1;
const F_SETFD: u64 = 2;
const FD_CLOEXEC: u64 = 1;

fn dup_core(fd: i32) -> FsResult<i64> {
    let mut t = fds();
    if !t.map.contains_key(&fd) {
        return Err(EBADF);
    }
    let new = t.lowest_free(3);
    t.bind_duplicate(new, fd).ok_or(EBADF)?;
    Ok(new as i64)
}

/// dup2(old, new): `new` becomes another descriptor of `old`'s open file
/// (sharing its offset and flock lock, without FD_CLOEXEC); whatever `new`
/// named before is closed first. dup2(fd, fd) only checks that fd is open.
fn dup2_core(old: i32, new: i32) -> FsResult<i64> {
    if old < 0 || new < 0 {
        return Err(EBADF);
    }
    if get_handle(old).is_none() {
        return Err(EBADF);
    }
    if old == new {
        return Ok(new as i64);
    }
    // Replacing an open descriptor closes it, kernel sockets, kqueues and
    // JIT shared memory included.
    if !crate::net::kernel_socket_close(new) && crate::sync::kqueue::close(new) {
        crate::sync::equeue::descriptor_closed(new);
    }
    crate::memory::jit::close(new);
    fds().bind_duplicate(new, old).ok_or(EBADF)?;
    Ok(new as i64)
}

fn fcntl_core(fd: i32, cmd: u64, arg: u64) -> FsResult<i64> {
    const F_GETFL: u64 = 3;
    const F_SETFL: u64 = 4;
    const F_GETOWN: u64 = 5;
    const F_SETOWN: u64 = 6;
    const F_DUP2FD: u64 = 10;
    const F_DUPFD_CLOEXEC: u64 = 17;
    const F_DUP2FD_CLOEXEC: u64 = 18;
    match cmd {
        F_DUPFD | F_DUPFD_CLOEXEC => {
            if fd < 0 {
                return Err(EBADF);
            }
            if crate::net::is_kernel_socket(fd) || crate::sync::kqueue::is_kqueue(fd) || crate::memory::jit::is_jit(fd) {
                log::error!("[fs] fcntl F_DUPFD of fd {fd} (not in the file table) is not supported");
                return Err(EINVAL);
            }
            let mut t = fds();
            if !t.map.contains_key(&fd) {
                return Err(EBADF);
            }
            if (arg as i32) < 0 {
                return Err(EINVAL);
            }
            let new = t.lowest_free(arg as i32);
            t.bind_duplicate(new, fd).ok_or(EBADF)?;
            if cmd == F_DUPFD_CLOEXEC {
                t.cloexec.insert(new);
            }
            Ok(new as i64)
        }
        F_DUP2FD | F_DUP2FD_CLOEXEC => {
            let new = dup2_core(fd, arg as i32)?;
            if cmd == F_DUP2FD_CLOEXEC {
                set_cloexec(new as i32, true);
            }
            Ok(new)
        }
        F_GETFD => {
            let t = fds();
            if !t.map.contains_key(&fd) && !t.reserved.contains(&fd) {
                return Err(EBADF);
            }
            Ok(t.cloexec.contains(&fd) as i64)
        }
        F_SETFD => {
            if set_cloexec(fd, arg & FD_CLOEXEC != 0) { Ok(0) } else { Err(EBADF) }
        }
        _ => {
            if let Some(r) = crate::net::kernel_socket_fcntl(fd, cmd, arg) {
                return r;
            }
            if !descriptor_open(fd) {
                return Err(EBADF);
            }
            let status = fds().status.get(&fd).cloned();
            match (cmd, status) {
                (F_GETFL, Some(st)) => Ok(*st.lock().unwrap_or_else(|p| p.into_inner()) as i64),
                (F_SETFL, Some(_)) => change_status(fd, STATUS_OPEN_FLAGS as u32, arg as u32),
                (F_GETFL | F_SETFL | F_GETOWN | F_SETOWN, _) => {
                    log::error!("[fs] fcntl {cmd} on fd {fd} is not supported");
                    Err(EINVAL)
                }
                _ => Err(EINVAL),
            }
        }
    }
}

const POLLIN: i16 = 0x0001;
const POLLOUT: i16 = 0x0004;
/// Guest POLLWRBAND; the host uses 0x200 for it and 0x100 for POLLWRNORM.
const POLLWRBAND: i16 = 0x0100;

fn poll_events_to_host(events: i16) -> i16 {
    (events & 0xFF) | if events & POLLWRBAND != 0 { libc::POLLWRBAND } else { 0 }
}

fn poll_events_from_host(revents: i16) -> i16 {
    let mut r = revents & 0xFF;
    if revents & libc::POLLWRNORM != 0 {
        r |= POLLOUT;
    }
    if revents & libc::POLLWRBAND != 0 {
        r |= POLLWRBAND;
    }
    r
}

const POLLPRI: i16 = 0x0002;
const POLLERR: i16 = 0x0008;
const POLLHUP: i16 = 0x0010;
const POLLNVAL: i16 = 0x0020;
const POLLRDNORM: i16 = 0x0040;
/// Every event bit a guest poll may request.
const POLL_ACCEPTED: i16 = 0x01FF;
/// What a regular file or directory always is: readable and writable.
const POLL_VNODE_READY: i16 = POLLIN | POLLRDNORM | POLLOUT;

/// Keeps the host descriptor of a polled guest descriptor open.
#[allow(dead_code)]
enum PollKeep {
    Kernel(crate::net::PollHandle),
    Net(crate::net::NetPollHandle),
    Pipe(Arc<pipe::PipeEnd>),
}

/// How poll or select learns the readiness of one descriptor.
enum Waitable {
    /// A host descriptor (a socket or a pipe end) polled for the events.
    Host { fd: i32, pipe: bool },
    /// Known without waiting: these of the requested events are ready.
    Ready(i16),
    /// A kqueue: readable while it holds triggered events.
    Kqueue(i32),
    /// Nothing is open on the descriptor.
    NotOpen,
}

/// Classify `fd` for poll/select with the guest `events` requested; host
/// descriptors are kept open in `keep`. Sockets and pipes are polled on the
/// host; a stream socket without a connection never becomes ready (as on
/// FreeBSD); regular files and directories are always readable and
/// writable, and so is JIT shared memory; the random devices are readable
/// (or whatever else was asked); descriptor 0 is never ready and 1 and 2
/// are writable, as for a title without a console. A P2P libSceNet socket
/// is EOPNOTSUPP.
fn waitable(fd: i32, events: i16, keep: &mut Vec<PollKeep>) -> Result<Waitable, i32> {
    if crate::net::is_kernel_socket(fd) {
        return Ok(match crate::net::kernel_socket_poll_handle(fd) {
            Some(h) => {
                let raw = h.fd();
                keep.push(PollKeep::Kernel(h));
                Waitable::Host { fd: raw, pipe: false }
            }
            None => Waitable::Ready(0),
        });
    }
    match get_handle(fd) {
        Some(Handle::File(_)) | Some(Handle::Dir(_)) => return Ok(Waitable::Ready(events & POLL_VNODE_READY)),
        Some(Handle::Random) => {
            let readable = events & (POLLIN | POLLRDNORM);
            return Ok(Waitable::Ready(if readable != 0 { readable } else { events }));
        }
        Some(Handle::Pipe(p)) => {
            let raw = p.raw();
            keep.push(PollKeep::Pipe(p));
            return Ok(Waitable::Host { fd: raw, pipe: true });
        }
        None => {}
    }
    if crate::sync::kqueue::is_kqueue(fd) {
        return Ok(Waitable::Kqueue(fd));
    }
    if crate::memory::jit::is_jit(fd) {
        return Ok(Waitable::Ready(events & POLL_VNODE_READY));
    }
    if is_stdio(fd) {
        return Ok(Waitable::Ready(if fd == 0 { 0 } else { events & POLLOUT }));
    }
    if let Some(h) = crate::net::net_socket_poll_handle(fd) {
        let h = h?;
        let raw = h.fd();
        keep.push(PollKeep::Net(h));
        return Ok(Waitable::Host { fd: raw, pipe: false });
    }
    Ok(Waitable::NotOpen)
}

/// Guest revents of a host poll result for the requested `events`: the
/// requested readiness plus error, hang-up and invalid. A pipe reports a
/// gone peer as a hang-up (readable, no longer writable), as FreeBSD does.
fn host_revents(events: i16, host: i16, pipe: bool) -> i16 {
    let mut r = poll_events_from_host(host) & (events | POLLERR | POLLHUP | POLLNVAL);
    if pipe && r & POLLERR != 0 {
        r = (r & !POLLERR) | POLLHUP;
    }
    if pipe && r & POLLHUP != 0 {
        r = (r & !POLLOUT) | (events & (POLLIN | POLLRDNORM));
    }
    // A hang-up is never reported together with writability (FreeBSD's
    // pollscan clears POLLOUT whenever it reports POLLHUP); Linux reports
    // both for a stream socket shut in both directions.
    if r & POLLHUP != 0 {
        r &= !POLLOUT;
    }
    r
}

/// Poll `host` once for at most `slice_ms`; EINTR counts as nothing ready.
fn poll_host_slice(host: &mut [libc::pollfd], slice_ms: i32) -> FsResult<i32> {
    for p in host.iter_mut() {
        p.revents = 0;
    }
    // SAFETY: a valid pollfd array; the keepers hold the descriptors open.
    let n = unsafe { libc::poll(host.as_mut_ptr(), host.len() as libc::nfds_t, slice_ms) };
    if n < 0 {
        let e = io::Error::last_os_error();
        if e.raw_os_error() == Some(libc::EINTR) {
            return Ok(0);
        }
        return Err(host_errno(&e));
    }
    Ok(n)
}

/// Wait in slices of at most 10 ms until `ready` says so or the deadline
/// (None: never) passes, letting guest signals and shutdown through.
/// `ready` polls with the slice it is given.
fn wait_sliced(deadline: Option<std::time::Instant>, mut ready: impl FnMut(i32) -> FsResult<bool>) -> FsResult<()> {
    loop {
        let slice = match deadline {
            None => 10,
            Some(d) => {
                let left = d.saturating_duration_since(std::time::Instant::now());
                (left.as_micros().div_ceil(1000) as i32).min(10)
            }
        };
        if ready(slice)? || deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(());
        }
        crate::signals::dispatch_pending();
        ps5vm::shutdown::park_if_requested();
    }
}

/// poll(fds, nfds, timeout_ms) as FreeBSD's: a negative descriptor is
/// skipped, one that is not open reports POLLNVAL; see `waitable` for the
/// rest. A timeout below -1 is EINVAL, -1 waits forever (even with nothing
/// to poll), and event bits FreeBSD does not define are refused.
fn poll_core(ctx: &CpuContext, fds_addr: u64, count: u32, timeout: i32) -> FsResult<i64> {
    if timeout < -1 {
        return Err(EINVAL);
    }
    if count != 0 && fds_addr == 0 {
        return Err(EFAULT);
    }
    let mut records = Vec::with_capacity(count.min(4096) as usize);
    for i in 0..count.min(4096) as u64 {
        records.push(read_guest(ctx, fds_addr + i * 8, 8)?);
    }
    let fd_of = |rec: &[u8]| i32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]);
    let events_of = |rec: &[u8]| i16::from_le_bytes([rec[4], rec[5]]);
    let mut keep = Vec::new();
    let mut waits = Vec::with_capacity(records.len());
    let mut host: Vec<libc::pollfd> = Vec::new();
    for rec in &records {
        let (fd, events) = (fd_of(rec), events_of(rec));
        if fd < 0 {
            waits.push(None);
            continue;
        }
        if events & !POLL_ACCEPTED != 0 {
            log::error!("[fs] poll: event bits {events:#x} on fd {fd} are not supported");
            return Err(EINVAL);
        }
        let w = waitable(fd, events, &mut keep)?;
        if let Waitable::Host { fd: raw, .. } = w {
            host.push(libc::pollfd { fd: raw, events: poll_events_to_host(events), revents: 0 });
        }
        waits.push(Some(w));
    }
    let deadline = (timeout >= 0).then(|| std::time::Instant::now() + std::time::Duration::from_millis(timeout as u64));
    let mut revents = vec![0i16; records.len()];
    let mut ready = 0i64;
    wait_sliced(deadline, |slice| {
        let fixed = waits.iter().zip(&records).any(|(w, rec)| match w {
            Some(Waitable::Ready(r)) => *r != 0,
            Some(Waitable::NotOpen) => true,
            Some(Waitable::Kqueue(kq)) => events_of(rec) & (POLLIN | POLLRDNORM) != 0 && crate::sync::kqueue::has_pending(*kq),
            _ => false,
        });
        if !host.is_empty() {
            poll_host_slice(&mut host, if fixed { 0 } else { slice })?;
        } else if !fixed && slice > 0 {
            crate::signals::sleep(std::time::Duration::from_millis(slice as u64));
        }
        let mut hosts = host.iter();
        ready = 0;
        for ((w, rec), out) in waits.iter().zip(&records).zip(revents.iter_mut()) {
            let events = events_of(rec);
            *out = match w {
                None => 0,
                Some(Waitable::Ready(r)) => *r,
                Some(Waitable::NotOpen) => POLLNVAL,
                Some(Waitable::Kqueue(kq)) => {
                    if crate::sync::kqueue::has_pending(*kq) { events & (POLLIN | POLLRDNORM) } else { 0 }
                }
                Some(Waitable::Host { pipe, .. }) => host_revents(events, hosts.next().map_or(0, |p| p.revents), *pipe),
            };
            ready += (*out != 0) as i64;
        }
        Ok(ready > 0)
    })?;
    for (i, (mut rec, r)) in records.into_iter().zip(revents).enumerate() {
        rec[6..8].copy_from_slice(&r.to_le_bytes());
        write_guest(ctx, fds_addr + i as u64 * 8, &rec)?;
    }
    Ok(ready)
}

pub(crate) fn cb_poll(ctx: &mut CpuContext) -> HleResult {
    let r = poll_core(ctx, ctx.rdi(), ctx.rsi() as u32, ctx.rdx() as i32);
    posix(ctx, r)
}

/// FreeBSD FD_SETSIZE: an fd_set is 1024 bits in 64-bit words.
const FD_SETSIZE: usize = 1024;

/// select(nfds, readfds, writefds, exceptfds, timeout) as FreeBSD's:
/// returns the number of set bits left in the three sets, which keep only
/// the ready descriptors. Readiness is poll's (`waitable`): readable on
/// data, end of stream or error, writable on space or error, exceptional on
/// out-of-band data. A set descriptor that is not open is EBADF; nfds
/// outside 0..=FD_SETSIZE is EINVAL. The timeout is a timeval
/// {i64 sec, i64 usec}: invalid fields are EINVAL, one too large to add to
/// the clock waits forever, NULL waits forever.
fn select_core(ctx: &CpuContext, nfds: i32, sets: [u64; 3], timeout: u64) -> FsResult<i64> {
    if !(0..=FD_SETSIZE as i32).contains(&nfds) {
        return Err(EINVAL);
    }
    let n = nfds as usize;
    let words = n.div_ceil(64);
    let mut bits = [vec![0u64; words], vec![0u64; words], vec![0u64; words]];
    for (set, addr) in bits.iter_mut().zip(sets) {
        if addr == 0 {
            continue;
        }
        let raw = read_guest(ctx, addr, words * 8)?;
        for (w, b) in set.iter_mut().zip(raw.chunks_exact(8)) {
            *w = u64::from_le_bytes(b.try_into().unwrap());
        }
        // Bits at or past nfds are not examined (and come back clear).
        if n % 64 != 0 {
            set[words - 1] &= (1u64 << (n % 64)) - 1;
        }
    }
    let deadline = if timeout == 0 {
        None
    } else {
        let raw = read_guest(ctx, timeout, 16)?;
        let sec = i64::from_le_bytes(raw[0..8].try_into().unwrap());
        let usec = i64::from_le_bytes(raw[8..16].try_into().unwrap());
        if sec < 0 || !(0..1_000_000).contains(&usec) {
            return Err(EINVAL);
        }
        let wait = std::time::Duration::new(sec as u64, usec as u32 * 1000);
        std::time::Instant::now().checked_add(wait)
    };
    let has = |set: &Vec<u64>, fd: usize| set[fd / 64] >> (fd % 64) & 1 != 0;
    let mut keep = Vec::new();
    let mut host: Vec<libc::pollfd> = Vec::new();
    let mut waits: Vec<(usize, i16, Waitable)> = Vec::new();
    for fd in 0..n {
        let mut events = 0;
        for (k, flag) in [POLLIN, POLLOUT, POLLPRI].into_iter().enumerate() {
            if has(&bits[k], fd) {
                events |= flag;
            }
        }
        if events == 0 {
            continue;
        }
        let w = waitable(fd as i32, events, &mut keep)?;
        match w {
            Waitable::NotOpen => return Err(EBADF),
            Waitable::Host { fd: raw, .. } => host.push(libc::pollfd { fd: raw, events: poll_events_to_host(events), revents: 0 }),
            _ => {}
        }
        waits.push((fd, events, w));
    }
    // (read, write, except) readiness of every waited descriptor.
    let mut results = vec![(false, false, false); waits.len()];
    let mut count = 0i64;
    wait_sliced(deadline, |slice| {
        let fixed = waits.iter().any(|(_, events, w)| match w {
            Waitable::Ready(r) => *r != 0,
            Waitable::Kqueue(kq) => events & POLLIN != 0 && crate::sync::kqueue::has_pending(*kq),
            _ => false,
        });
        if !host.is_empty() {
            poll_host_slice(&mut host, if fixed { 0 } else { slice })?;
        } else if !fixed && slice > 0 {
            crate::signals::sleep(std::time::Duration::from_millis(slice as u64));
        }
        let mut hosts = host.iter();
        count = 0;
        for ((_, events, w), out) in waits.iter().zip(results.iter_mut()) {
            let revents = match w {
                Waitable::Ready(r) => *r,
                Waitable::Kqueue(kq) => {
                    if crate::sync::kqueue::has_pending(*kq) { events & POLLIN } else { 0 }
                }
                Waitable::Host { pipe, .. } => host_revents(*events, hosts.next().map_or(0, |p| p.revents), *pipe),
                Waitable::NotOpen => 0,
            };
            *out = (
                events & POLLIN != 0 && revents & (POLLIN | POLLHUP | POLLERR) != 0,
                events & POLLOUT != 0 && revents & (POLLOUT | POLLERR) != 0,
                events & POLLPRI != 0 && revents & POLLPRI != 0,
            );
            count += out.0 as i64 + out.1 as i64 + out.2 as i64;
        }
        Ok(count > 0)
    })?;
    let mut out = [vec![0u64; words], vec![0u64; words], vec![0u64; words]];
    for ((fd, _, _), (rd, wr, ex)) in waits.iter().zip(&results) {
        for (k, ready) in [*rd, *wr, *ex].into_iter().enumerate() {
            if ready {
                out[k][fd / 64] |= 1 << (fd % 64);
            }
        }
    }
    for (set, addr) in out.iter().zip(sets) {
        if addr != 0 {
            let bytes: Vec<u8> = set.iter().flat_map(|w| w.to_le_bytes()).collect();
            write_guest(ctx, addr, &bytes)?;
        }
    }
    Ok(count)
}

fn cb_select(ctx: &mut CpuContext) -> HleResult {
    let r = select_core(ctx, ctx.rdi() as i32, [ctx.rsi(), ctx.rdx(), ctx.rcx()], ctx.r8());
    posix(ctx, r)
}

// ---------------------------------------------------------------------------
// AIO (requests complete synchronously at submission)
// ---------------------------------------------------------------------------

const AIO_INIT_PARAM_SIZE: usize = 0x3C;
const AIO_STATE_SUBMITTED: i32 = 1;
const AIO_STATE_PROCESSING: i32 = 2;
const AIO_STATE_COMPLETED: i32 = 3;
const AIO_STATE_ABORTED: i32 = 4;
/// Set once a terminal state has been reported by a poll or wait.
const AIO_STATE_NOTIFIED: i32 = 0x10000;
/// Request ids cycle through 1..AIO_MAX_QUEUE-1.
const AIO_MAX_QUEUE: usize = 512;
const AIO_MAX_REQUESTS: i32 = 128;
const AIO_RW_REQUEST_SIZE: u64 = 40;
const AIO_WAIT_AND: u32 = 1;
const AIO_WAIT_OR: u32 = 2;

/// State per request id (0: free).
static AIO_STATE: [AtomicI32; AIO_MAX_QUEUE] = [const { AtomicI32::new(0) }; AIO_MAX_QUEUE];
static AIO_NEXT_ID: AtomicU32 = AtomicU32::new(1);

fn aio_new_id() -> usize {
    let seq = AIO_NEXT_ID.fetch_add(1, Ordering::Relaxed);
    ((seq - 1) as usize % (AIO_MAX_QUEUE - 1)) + 1
}

fn aio_valid_id(id: i32) -> bool {
    id > 0 && (id as usize) < AIO_MAX_QUEUE
}

/// State of a request; with `notify` a terminal state is marked reported.
fn aio_get_state(id: i32, notify: bool) -> i32 {
    if !aio_valid_id(id) {
        return sce_error(ESRCH) as i32;
    }
    let slot = &AIO_STATE[id as usize];
    let mut current = slot.load(Ordering::Acquire);
    while notify && (current == AIO_STATE_COMPLETED || current == AIO_STATE_ABORTED) {
        match slot.compare_exchange_weak(current, current | AIO_STATE_NOTIFIED, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => break,
            Err(v) => current = v,
        }
    }
    if current == 0 { sce_error(ESRCH) as i32 } else { current }
}

/// One decoded `SceKernelAioRWRequest`.
#[derive(Clone, Copy)]
struct AioRequest {
    offset: i64,
    nbyte: u64,
    buf: u64,
    result: u64,
    fd: i32,
}

fn read_aio_request(ctx: &CpuContext, req: u64) -> FsResult<AioRequest> {
    let raw = read_guest(ctx, req, AIO_RW_REQUEST_SIZE as usize)?;
    let rd = |o: usize| u64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
    let fd = i32::from_le_bytes(raw[32..36].try_into().unwrap());
    Ok(AioRequest { offset: rd(0) as i64, nbyte: rd(8), buf: rd(16), result: rd(24), fd })
}

/// Run one request and store its result: the byte count and COMPLETED, or
/// the kernel error (EINVAL, EBADF, EFAULT, ...) and ABORTED. Returns the
/// request's final state.
fn aio_request(ctx: &CpuContext, r: AioRequest, write: bool) -> FsResult<i32> {
    write_u32(ctx, r.result + 8, AIO_STATE_SUBMITTED as u32)?;
    let done = if write {
        pwrite_core(ctx, r.fd, r.buf, r.nbyte, r.offset)
    } else {
        pread_core(ctx, r.fd, r.buf, r.nbyte, r.offset)
    };
    let (value, state) = match done {
        Ok(n) => (n, AIO_STATE_COMPLETED),
        Err(e) => (sce_error(e), AIO_STATE_ABORTED),
    };
    write_u64(ctx, r.result, value as u64)?;
    write_u32(ctx, r.result + 8, state as u32)?;
    Ok(state)
}

/// Submit `size` requests. `multiple` hands out one id per request
/// (written as an array), each ending in its request's state; otherwise one
/// id covers the batch and ends ABORTED when any request failed. Every
/// request needs a result pointer before any of them runs.
fn aio_submit(ctx: &CpuContext, write: bool, multiple: bool) -> HleResult {
    let (reqs, size, out_id) = (ctx.rdi(), ctx.rsi() as i32, ctx.rcx());
    if reqs == 0 || out_id == 0 {
        return sce_error(EFAULT);
    }
    if size <= 0 || size > AIO_MAX_REQUESTS {
        return sce_error(EINVAL);
    }
    let mut requests = Vec::with_capacity(size as usize);
    for i in 0..size as u64 {
        match read_aio_request(ctx, reqs + i * AIO_RW_REQUEST_SIZE) {
            Ok(r) if r.result != 0 => requests.push(r),
            _ => return sce_error(EFAULT),
        }
    }
    let batch_id = (!multiple).then(aio_new_id);
    if let Some(id) = batch_id {
        AIO_STATE[id].store(AIO_STATE_PROCESSING, Ordering::Release);
    }
    let mut batch_state = AIO_STATE_COMPLETED;
    for (i, r) in requests.into_iter().enumerate() {
        let id = batch_id.unwrap_or_else(aio_new_id);
        if multiple {
            AIO_STATE[id].store(AIO_STATE_PROCESSING, Ordering::Release);
        }
        let state = match aio_request(ctx, r, write) {
            Ok(s) => s,
            Err(e) => {
                AIO_STATE[id].store(AIO_STATE_ABORTED, Ordering::Release);
                return sce_error(e);
            }
        };
        if state == AIO_STATE_ABORTED {
            batch_state = AIO_STATE_ABORTED;
        }
        if multiple {
            AIO_STATE[id].store(state, Ordering::Release);
            if write_u32(ctx, out_id + i as u64 * 4, id as u32).is_err() {
                return sce_error(EFAULT);
            }
        }
    }
    if let Some(id) = batch_id {
        AIO_STATE[id].store(batch_state, Ordering::Release);
        if write_u32(ctx, out_id, id as u32).is_err() {
            return sce_error(EFAULT);
        }
    }
    0
}

fn read_ids(ctx: &CpuContext, ids: u64, num: i32) -> FsResult<Vec<i32>> {
    let raw = read_guest(ctx, ids, num as usize * 4)?;
    Ok(raw.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect())
}

fn write_states(ctx: &CpuContext, states: u64, values: &[i32]) -> FsResult<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_guest(ctx, states, &bytes)
}

fn batch_args(ctx: &CpuContext) -> Result<(Vec<i32>, u64), HleResult> {
    let (ids, num, out) = (ctx.rdi(), ctx.rsi() as i32, ctx.rdx());
    if ids == 0 || out == 0 {
        return Err(sce_error(EFAULT));
    }
    if num <= 0 || num > AIO_MAX_REQUESTS {
        return Err(sce_error(EINVAL));
    }
    Ok((read_ids(ctx, ids, num).map_err(sce_error)?, out))
}

/// sceKernelAioPollRequest(id, state*)
fn cb_aio_poll_request(ctx: &mut CpuContext) -> HleResult {
    let (id, out) = (ctx.rdi() as i32, ctx.rsi());
    if out == 0 {
        return sce_error(EFAULT);
    }
    sce(write_u32(ctx, out, aio_get_state(id, true) as u32).map(|_| 0))
}

/// sceKernelAioPollRequests(ids*, num, states*)
fn cb_aio_poll_requests(ctx: &mut CpuContext) -> HleResult {
    let (ids, out) = match batch_args(ctx) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let states: Vec<i32> = ids.iter().map(|&id| aio_get_state(id, true)).collect();
    sce(write_states(ctx, out, &states).map(|_| 0))
}

/// Wait for requests: AND = all terminal, OR = any newly completed. The
/// remaining time is written back to `usec`.
fn aio_wait(ctx: &CpuContext, ids: &[i32], states_out: u64, mode: u32, usec_ptr: u64) -> HleResult {
    let usec = if usec_ptr != 0 {
        match ctx.read_u32(usec_ptr) {
            Some(v) => Some(v as u64),
            None => return sce_error(EFAULT),
        }
    } else {
        None
    };
    let collect = |notify: bool| {
        let mut finished = 0;
        let mut newly = false;
        let states: Vec<i32> = ids
            .iter()
            .map(|&id| {
                let s = aio_get_state(id, notify);
                finished += (s != AIO_STATE_PROCESSING) as usize;
                newly |= s == AIO_STATE_COMPLETED || s == AIO_STATE_ABORTED;
                s
            })
            .collect();
        (finished == ids.len() || (mode == AIO_WAIT_OR && newly), states)
    };
    let start = Instant::now();
    loop {
        let (complete, _) = collect(false);
        let elapsed = start.elapsed().as_micros() as u64;
        let timed_out = usec.is_some_and(|u| elapsed >= u);
        if complete || timed_out {
            let (done, states) = collect(true);
            if done {
                if write_states(ctx, states_out, &states).is_err() {
                    return sce_error(EFAULT);
                }
                if let Some(u) = usec {
                    ctx.write_u32(usec_ptr, u.saturating_sub(elapsed) as u32);
                }
                return 0;
            }
            if timed_out {
                return sce_error(ETIMEDOUT);
            }
        }
        crate::signals::sleep(std::time::Duration::from_micros(10));
    }
}

/// sceKernelAioWaitRequests(ids*, num, states*, mode, usec*)
fn cb_aio_wait_requests(ctx: &mut CpuContext) -> HleResult {
    let (ids, out) = match batch_args(ctx) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mode = ctx.rcx() as u32;
    if ids.len() != 1 && mode != AIO_WAIT_AND && mode != AIO_WAIT_OR {
        return sce_error(EINVAL);
    }
    aio_wait(ctx, &ids, out, mode, ctx.r8())
}

/// sceKernelAioWaitRequest(id, state*, usec*)
fn cb_aio_wait_request(ctx: &mut CpuContext) -> HleResult {
    let (id, out, usec) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    if out == 0 {
        return sce_error(EFAULT);
    }
    aio_wait(ctx, &[id], out, AIO_WAIT_AND, usec)
}

/// Free a terminal request: OK, EBUSY while running, ESRCH if unknown.
fn aio_delete(id: i32) -> i32 {
    if !aio_valid_id(id) {
        return sce_error(ESRCH) as i32;
    }
    let slot = &AIO_STATE[id as usize];
    let mut current = slot.load(Ordering::Acquire);
    loop {
        let state = current & !AIO_STATE_NOTIFIED;
        if state != AIO_STATE_COMPLETED && state != AIO_STATE_ABORTED {
            return if current == 0 { sce_error(ESRCH) as i32 } else { sce_error(EBUSY) as i32 };
        }
        match slot.compare_exchange_weak(current, 0, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return 0,
            Err(v) => current = v,
        }
    }
}

/// sceKernelAioDeleteRequest(id, ret*)
fn cb_aio_delete_request(ctx: &mut CpuContext) -> HleResult {
    let (id, out) = (ctx.rdi() as i32, ctx.rsi());
    if out == 0 {
        return sce_error(EFAULT);
    }
    sce(write_u32(ctx, out, aio_delete(id) as u32).map(|_| 0))
}

/// sceKernelAioDeleteRequests(ids*, num, rets*)
fn cb_aio_delete_requests(ctx: &mut CpuContext) -> HleResult {
    let (ids, out) = match batch_args(ctx) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let rets: Vec<i32> = ids.iter().map(|&id| aio_delete(id)).collect();
    sce(write_states(ctx, out, &rets).map(|_| 0))
}

/// sceKernelAioCancelRequest(id, state*): requests run at submission, so
/// there is nothing left to cancel; report the current state.
fn cb_aio_cancel_request(ctx: &mut CpuContext) -> HleResult {
    let (id, out) = (ctx.rdi() as i32, ctx.rsi());
    if out == 0 {
        return sce_error(EFAULT);
    }
    sce(write_u32(ctx, out, aio_get_state(id, false) as u32).map(|_| 0))
}

/// sceKernelAioCancelRequests(int32 ids[], num, int32 states[]): requests
/// run to completion when submitted, so each reports its state as
/// sceKernelAioCancelRequest does; id 0 reports PROCESSING. Any other
/// invalid id fails the whole call with EINVAL before a state is written.
fn cb_aio_cancel_requests(ctx: &mut CpuContext) -> HleResult {
    let (ids, num, states) = (ctx.rdi(), ctx.rsi() as i32, ctx.rdx());
    if ids == 0 || states == 0 {
        return sce_error(EFAULT);
    }
    if num < 0 {
        return sce_error(EINVAL);
    }
    if num > AIO_MAX_REQUESTS {
        log::error!("[fs] sceKernelAioCancelRequests: {num} ids in one batch (more than 128) is not modelled");
        return sce_error(EINVAL);
    }
    let Some(list): Option<Vec<i32>> = (0..num as u64).map(|i| ctx.read_u32(ids + i * 4).map(|v| v as i32)).collect() else {
        return sce_error(EFAULT);
    };
    if list.iter().any(|&id| id != 0 && !aio_valid_id(id)) {
        return sce_error(EINVAL);
    }
    for (i, &id) in list.iter().enumerate() {
        let state = if id == 0 { AIO_STATE_PROCESSING } else { aio_get_state(id, false) };
        if write_u32(ctx, states + i as u64 * 4, state as u32).is_err() {
            return sce_error(EFAULT);
        }
    }
    0
}

/// SceKernelAioSchedulingParam limits.
const AIO_SCHEDULING_WINDOW_MAX: i32 = 128;
const AIO_DELAYED_COUNT_MAX: i32 = 128;
const AIO_SPLIT_SIZE_MAX: u32 = 0x100_0000;

/// sceKernelAioSetParam(param*, window, delayed, enableSplit, splitSize,
/// splitChunkSize): fills the five fields; values outside the documented
/// ranges are refused.
fn cb_aio_set_param(ctx: &mut CpuContext) -> HleResult {
    let (p, window, delayed, split, size, chunk) =
        (ctx.rdi(), ctx.rsi() as i32, ctx.rdx() as i32, ctx.rcx() as u32, ctx.r8() as u32, ctx.r9() as u32);
    if p == 0 {
        log::error!("[fs] sceKernelAioSetParam: null param");
        return sce_error(EFAULT);
    }
    let ok = (0..=AIO_SCHEDULING_WINDOW_MAX).contains(&window)
        && (0..=AIO_DELAYED_COUNT_MAX).contains(&delayed)
        && split <= 1
        && size <= AIO_SPLIT_SIZE_MAX
        && chunk <= AIO_SPLIT_SIZE_MAX;
    if !ok {
        log::error!("[fs] sceKernelAioSetParam({window}, {delayed}, {split}, {size:#x}, {chunk:#x}): outside the modelled ranges");
        return sce_error(EINVAL);
    }
    let mut b = [0u8; 20];
    for (i, v) in [window as u32, delayed as u32, split, size, chunk].iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    sce(write_guest(ctx, p, &b).map(|_| 0))
}

fn cb_aio_initialize_param(ctx: &mut CpuContext) -> HleResult {
    let p = ctx.rdi();
    if p == 0 {
        return sce_error(EINVAL);
    }
    sce(write_guest(ctx, p, &[0u8; AIO_INIT_PARAM_SIZE]).map(|_| 0))
}

fn cb_aio_initialize_impl(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 || (ctx.rsi() as i32) < AIO_INIT_PARAM_SIZE as i32 {
        return sce_error(EINVAL);
    }
    0
}

fn cb_aio_submit_read(ctx: &mut CpuContext) -> HleResult {
    aio_submit(ctx, false, false)
}
fn cb_aio_submit_read_multiple(ctx: &mut CpuContext) -> HleResult {
    aio_submit(ctx, false, true)
}
fn cb_aio_submit_write(ctx: &mut CpuContext) -> HleResult {
    aio_submit(ctx, true, false)
}
fn cb_aio_submit_write_multiple(ctx: &mut CpuContext) -> HleResult {
    aio_submit(ctx, true, true)
}

// ---------------------------------------------------------------------------
// APR file ids
// ---------------------------------------------------------------------------

/// Guest paths are limited to PATH_MAX bytes including the terminator;
/// with a prefix, prefix and path share that buffer.
const APR_PATH_MAX: usize = 1024;
/// Most paths one resolve call accepts.
const APR_MAX_PATHS: u64 = 1024;
/// Size a resolve reports for a directory.
const APR_DIRECTORY_RESOLVE_SIZE: u64 = 0x10000;
/// Id written for a path that failed to resolve.
const APR_INVALID_ID: u32 = u32::MAX;

/// One resolved guest path. Ids are handed out once per distinct guest
/// path string (the full, concatenated path) from a process-wide counter,
/// so two different files never share an id.
#[derive(Clone)]
struct AprResolved {
    id: u32,
    is_dir: bool,
    /// Size reported by the resolve (directories report 64 KiB).
    size: u64,
}

struct AprFile {
    host: PathBuf,
    /// Size reported by sceKernelAprGetFileSize (0 for directories).
    size: u64,
    /// The entry of the `/app0` image this id names (then `host` is unused).
    image: Option<(Arc<App0Image>, ps5pkg::NodeId)>,
}

struct AprRegistry {
    by_path: HashMap<String, AprResolved>,
    files: HashMap<u32, AprFile>,
    /// Absolute host path -> guest paths resolved to it (size refresh).
    by_host: HashMap<PathBuf, Vec<String>>,
    /// Guest paths inside the immutable game image known to be missing.
    missing: HashSet<String>,
    next_id: u32,
}

static APR: LazyLock<Mutex<AprRegistry>> = LazyLock::new(|| {
    Mutex::new(AprRegistry {
        by_path: HashMap::new(),
        files: HashMap::new(),
        by_host: HashMap::new(),
        missing: HashSet::new(),
        next_id: 1,
    })
});

fn apr() -> std::sync::MutexGuard<'static, AprRegistry> {
    APR.lock().unwrap_or_else(|p| p.into_inner())
}

/// Host path behind an APR file id (None for ids inside the `/app0` image).
pub fn apr_host_path(id: u32) -> Option<PathBuf> {
    apr().files.get(&id).filter(|f| f.image.is_none()).map(|f| f.host.clone())
}

/// Contents of an APR file id inside the `/app0` image.
pub fn apr_image_file(id: u32) -> Option<ps5pkg::FileData> {
    let (img, node) = apr().files.get(&id)?.image.clone()?;
    img.pfs.open_file(node).ok()
}

/// A file changed on the host: refresh the sizes cached for its ids (the
/// ids themselves stay stable).
fn invalidate_apr_size(host: &Path) {
    let key = absolute(host);
    let mut reg = apr();
    let Some(paths) = reg.by_host.get(&key).cloned() else { return };
    let meta = fs::metadata(&key).ok();
    for path in paths {
        let Some(r) = reg.by_path.get_mut(&path) else { continue };
        match &meta {
            Some(m) if m.is_file() => r.size = m.len(),
            Some(m) if m.is_dir() => r.size = APR_DIRECTORY_RESOLVE_SIZE,
            // Gone: the next resolve of this path looks it up again.
            _ => {
                reg.by_path.remove(&path);
                continue;
            }
        }
        let (id, size) = (r.id, if r.is_dir { 0 } else { r.size });
        if let Some(f) = reg.files.get_mut(&id) {
            f.size = size;
        }
    }
}

/// Resolve one concatenated guest path: (id, resolve size) or an errno.
fn apr_resolve_one(guest: &str) -> Result<(u32, u64), i32> {
    if guest.is_empty() {
        return Err(EINVAL);
    }
    {
        let reg = apr();
        if let Some(r) = reg.by_path.get(guest) {
            return Ok((r.id, r.size));
        }
        if reg.missing.contains(guest) {
            return Err(ENOENT);
        }
    }
    let missing = || {
        log::info!("[fs] APR resolve missing path: {guest}");
        // The game image is immutable: remember misses inside it.
        if is_read_only(guest, None) {
            apr().missing.insert(guest.to_string());
        }
        ENOENT
    };
    let (host, is_dir, len, image) = if let Some((img, rel)) = image_target(guest) {
        let id = img.lookup(&rel).ok_or_else(missing)?;
        let n = img.node(id);
        (img.display_path(id), n.is_dir, n.size, Some((img.clone(), id)))
    } else {
        let host = resolve_guest_path(guest);
        let meta = host.as_deref().and_then(|h| fs::metadata(h).ok()).filter(|m| m.is_dir() || m.is_file());
        let (Some(host), Some(meta)) = (host, meta) else {
            return Err(missing());
        };
        (host, meta.is_dir(), meta.len(), None)
    };
    let size = if is_dir { APR_DIRECTORY_RESOLVE_SIZE } else { len };
    let mut reg = apr();
    // Another thread may have resolved the same path meanwhile.
    if let Some(r) = reg.by_path.get(guest) {
        return Ok((r.id, r.size));
    }
    if reg.next_id == APR_INVALID_ID {
        return Err(ENFILE);
    }
    let id = reg.next_id;
    reg.next_id += 1;
    if image.is_none() {
        reg.by_host.entry(absolute(&host)).or_default().push(guest.to_string());
    }
    reg.files.insert(id, AprFile { host, size: if is_dir { 0 } else { size }, image });
    reg.by_path.insert(guest.to_string(), AprResolved { id, is_dir, size });
    Ok((id, size))
}

/// Resolve a guest path to its APR file id (tests of the APR consumers).
#[cfg(test)]
pub(crate) fn apr_resolve_path(guest: &str) -> Option<u32> {
    apr_resolve_one(guest).ok().map(|(id, _)| id)
}

/// Read a NUL-terminated string that must fit, terminator included, in
/// `room` bytes: EFAULT when unreadable, ENAMETOOLONG when it does not fit.
fn read_apr_string(ctx: &CpuContext, addr: u64, room: usize) -> FsResult<Vec<u8>> {
    if addr == 0 {
        return Err(EFAULT);
    }
    if room == 0 {
        return Err(ENAMETOOLONG);
    }
    let bytes = ctx.read_cstr_bytes(addr, room).ok_or(EFAULT)?;
    if bytes.len() >= room {
        return Err(ENAMETOOLONG);
    }
    Ok(bytes)
}

struct AprResolveArgs {
    /// Guest address of the prefix (0 = none).
    prefix: u64,
    list: u64,
    count: u64,
    ids: u64,
    sizes: u64,
    error_index: u64,
    /// ForEach variants: per-path result array (`int*`). Failures are
    /// reported there and the walk continues; the call returns the number
    /// of resolved paths.
    results: u64,
}

/// Shared body of the ResolveFilepaths* exports; errors are reported as -1
/// with errno.
fn apr_resolve(ctx: &CpuContext, a: AprResolveArgs) -> HleResult {
    let r = apr_resolve_core(ctx, &a);
    posix(ctx, r)
}

fn apr_resolve_core(ctx: &CpuContext, a: &AprResolveArgs) -> FsResult<i64> {
    if a.list == 0 || a.count == 0 || a.count > APR_MAX_PATHS || (a.ids == 0 && a.sizes == 0 && a.results == 0) {
        return Err(EINVAL);
    }
    // The prefix is concatenated literally, without a separator.
    let prefix = if a.prefix != 0 { read_apr_string(ctx, a.prefix, APR_PATH_MAX)? } else { Vec::new() };
    let mut resolved = 0i64;
    for i in 0..a.count {
        let r = ctx.read_u64(a.list + i * 8).ok_or(EFAULT).and_then(|p| {
            let path = read_apr_string(ctx, p, APR_PATH_MAX - prefix.len())?;
            let mut full = prefix.clone();
            full.extend_from_slice(&path);
            apr_resolve_one(&String::from_utf8_lossy(&full))
        });
        if a.results != 0 {
            let code = match r {
                Ok(_) => 0,
                Err(e) => sce_error(e) as u32,
            };
            write_u32(ctx, a.results + i * 4, code)?;
        }
        match r {
            Ok((id, size)) => {
                log::debug!("[fs] apr resolve #{i} -> id={id:#x} size={size:#x}");
                if a.ids != 0 {
                    write_u32(ctx, a.ids + i * 4, id)?;
                }
                if a.sizes != 0 {
                    write_u64(ctx, a.sizes + i * 8, size)?;
                }
                resolved += 1;
            }
            Err(e) => {
                if a.ids != 0 {
                    write_u32(ctx, a.ids + i * 4, APR_INVALID_ID)?;
                }
                if a.sizes != 0 {
                    write_u64(ctx, a.sizes + i * 8, 0)?;
                }
                if a.error_index != 0 {
                    write_u32(ctx, a.error_index, i as u32)?;
                }
                if a.results == 0 {
                    return Err(e);
                }
            }
        }
    }
    Ok(if a.results != 0 { resolved } else { 0 })
}

fn apr_args(prefix: u64, list: u64, count: u64, ids: u64, sizes: u64, error_index: u64, results: u64) -> AprResolveArgs {
    AprResolveArgs { prefix, list, count: count as u32 as u64, ids, sizes, error_index, results }
}

/// (paths, count, ids*, errorIndex*)
fn cb_apr_resolve_ids(ctx: &mut CpuContext) -> HleResult {
    apr_resolve(ctx, apr_args(0, ctx.rdi(), ctx.rsi(), ctx.rdx(), 0, ctx.rcx(), 0))
}

/// (paths, count, ids*, sizes*, errorIndex*)
fn cb_apr_resolve_ids_and_sizes(ctx: &mut CpuContext) -> HleResult {
    apr_resolve(ctx, apr_args(0, ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8(), 0))
}

/// (prefix, paths, count, ids*, errorIndex*)
fn cb_apr_resolve_with_prefix_ids(ctx: &mut CpuContext) -> HleResult {
    apr_resolve(ctx, apr_args(ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), 0, ctx.r8(), 0))
}

/// (prefix, paths, count, ids*, sizes*, errorIndex*)
fn cb_apr_resolve_with_prefix_ids_and_sizes(ctx: &mut CpuContext) -> HleResult {
    apr_resolve(ctx, apr_args(ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8(), ctx.r9(), 0))
}

/// (paths, count, ids*, results*) -> resolved count
fn cb_apr_resolve_ids_for_each(ctx: &mut CpuContext) -> HleResult {
    apr_resolve(ctx, apr_args(0, ctx.rdi(), ctx.rsi(), ctx.rdx(), 0, 0, ctx.rcx()))
}

/// (paths, count, ids*, sizes*, results*) -> resolved count
fn cb_apr_resolve_ids_and_sizes_for_each(ctx: &mut CpuContext) -> HleResult {
    apr_resolve(ctx, apr_args(0, ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), 0, ctx.r8()))
}

/// (prefix, paths, count, ids*, results*) -> resolved count
fn cb_apr_resolve_with_prefix_ids_for_each(ctx: &mut CpuContext) -> HleResult {
    apr_resolve(ctx, apr_args(ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), 0, 0, ctx.r8()))
}

/// (prefix, paths, count, ids*, sizes*, results*) -> resolved count
fn cb_apr_resolve_with_prefix_ids_and_sizes_for_each(ctx: &mut CpuContext) -> HleResult {
    apr_resolve(ctx, apr_args(ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8(), 0, ctx.r9()))
}

/// Stat of an APR file: mode 0777 plus the type, directories empty.
fn apr_stat(host: &Path) -> FsResult<[u8; KERNEL_STAT_SIZE]> {
    let m = host_metadata(host)?;
    let mut b = stat_from_metadata(&m, host, Some(0));
    if m.is_dir() {
        b[72..80].copy_from_slice(&0i64.to_le_bytes());
        b[80..88].copy_from_slice(&0i64.to_le_bytes());
        b[88..92].copy_from_slice(&512u32.to_le_bytes());
    }
    Ok(b)
}

/// (id, SceKernelStat*)
fn cb_apr_get_file_stat(ctx: &mut CpuContext) -> HleResult {
    let (id, stat) = (ctx.rdi() as u32, ctx.rsi());
    let r = (|| {
        if stat == 0 {
            return Err(EINVAL);
        }
        let entry = apr().files.get(&id).map(|f| (f.host.clone(), f.image.clone()));
        let bytes = match entry {
            Some((_, Some((img, node)))) => {
                let mut b = img.stat(node);
                if img.node(node).is_dir {
                    b[72..80].copy_from_slice(&0i64.to_le_bytes());
                    b[80..88].copy_from_slice(&0i64.to_le_bytes());
                    b[88..92].copy_from_slice(&512u32.to_le_bytes());
                }
                b
            }
            Some((host, None)) => apr_stat(&host)?,
            None => {
                log::info!("[fs] APR stat failed for unknown file id: {id:#010x}");
                return Err(ENOENT);
            }
        };
        write_guest(ctx, stat, &bytes)?;
        Ok(0)
    })();
    posix(ctx, r)
}

/// (id, uint64_t* size): the size cached by the resolve.
fn cb_apr_get_file_size(ctx: &mut CpuContext) -> HleResult {
    let (id, out) = (ctx.rdi() as u32, ctx.rsi());
    let r = (|| {
        if out == 0 {
            return Err(EINVAL);
        }
        let Some(size) = apr().files.get(&id).map(|f| f.size) else {
            log::info!("[fs] APR size failed for unknown file id: {id:#010x}");
            return Err(ENOENT);
        };
        write_u64(ctx, out, size)?;
        Ok(0)
    })();
    posix(ctx, r)
}

// ---------------------------------------------------------------------------
// Export callbacks
// ---------------------------------------------------------------------------

fn fd_arg(ctx: &CpuContext) -> i32 {
    ctx.rdi() as i32
}

fn cb_sce_open(ctx: &mut CpuContext) -> HleResult {
    sce(open_core(ctx, ctx.rdi(), ctx.rsi()))
}
fn cb_posix_open(ctx: &mut CpuContext) -> HleResult {
    let r = open_core(ctx, ctx.rdi(), ctx.rsi());
    posix(ctx, r)
}
/// creat(path, mode): open(path, O_WRONLY | O_CREAT | O_TRUNC, mode).
fn cb_posix_creat(ctx: &mut CpuContext) -> HleResult {
    let r = open_core(ctx, ctx.rdi(), O_WRONLY | O_CREAT | O_TRUNC);
    posix(ctx, r)
}
fn cb_sce_close(ctx: &mut CpuContext) -> HleResult {
    sce(close_core(fd_arg(ctx)))
}
fn cb_posix_close(ctx: &mut CpuContext) -> HleResult {
    let r = close_core(fd_arg(ctx));
    posix(ctx, r)
}
fn cb_sce_read(ctx: &mut CpuContext) -> HleResult {
    sce(read_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx()))
}
fn cb_posix_read(ctx: &mut CpuContext) -> HleResult {
    let r = read_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx());
    posix(ctx, r)
}
fn cb_sce_write(ctx: &mut CpuContext) -> HleResult {
    sce(write_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx()))
}
fn cb_posix_write(ctx: &mut CpuContext) -> HleResult {
    let r = write_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx());
    posix(ctx, r)
}
fn cb_sce_pread(ctx: &mut CpuContext) -> HleResult {
    sce(pread_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx(), ctx.rcx() as i64))
}
fn cb_posix_pread(ctx: &mut CpuContext) -> HleResult {
    let r = pread_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx(), ctx.rcx() as i64);
    posix(ctx, r)
}
fn cb_sce_pwrite(ctx: &mut CpuContext) -> HleResult {
    sce(pwrite_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx(), ctx.rcx() as i64))
}
fn cb_posix_pwrite(ctx: &mut CpuContext) -> HleResult {
    let r = pwrite_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx(), ctx.rcx() as i64);
    posix(ctx, r)
}
fn cb_sce_readv(ctx: &mut CpuContext) -> HleResult {
    sce(readv_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx() as i32))
}
fn cb_sce_writev(ctx: &mut CpuContext) -> HleResult {
    sce(writev_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx() as i32))
}
fn cb_posix_readv(ctx: &mut CpuContext) -> HleResult {
    let r = readv_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx() as i32);
    posix(ctx, r)
}
fn cb_posix_writev(ctx: &mut CpuContext) -> HleResult {
    let r = writev_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx() as i32);
    posix(ctx, r)
}
fn cb_sce_preadv(ctx: &mut CpuContext) -> HleResult {
    sce(preadv_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx() as i32, ctx.rcx() as i64))
}
fn cb_sce_pwritev(ctx: &mut CpuContext) -> HleResult {
    sce(pwritev_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx() as i32, ctx.rcx() as i64))
}
fn cb_posix_preadv(ctx: &mut CpuContext) -> HleResult {
    let r = preadv_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx() as i32, ctx.rcx() as i64);
    posix(ctx, r)
}
fn cb_posix_pwritev(ctx: &mut CpuContext) -> HleResult {
    let r = pwritev_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx() as i32, ctx.rcx() as i64);
    posix(ctx, r)
}
fn cb_posix_flock(ctx: &mut CpuContext) -> HleResult {
    let r = flock_core(fd_arg(ctx), ctx.rsi() as i32);
    posix(ctx, r)
}
fn cb_sce_lseek(ctx: &mut CpuContext) -> HleResult {
    sce(lseek_core(fd_arg(ctx), ctx.rsi() as i64, ctx.rdx() as i32))
}
fn cb_posix_lseek(ctx: &mut CpuContext) -> HleResult {
    let r = lseek_core(fd_arg(ctx), ctx.rsi() as i64, ctx.rdx() as i32);
    posix(ctx, r)
}
fn cb_sce_stat(ctx: &mut CpuContext) -> HleResult {
    sce(stat_core(ctx, ctx.rdi(), ctx.rsi()))
}
fn cb_posix_stat(ctx: &mut CpuContext) -> HleResult {
    let r = stat_core(ctx, ctx.rdi(), ctx.rsi());
    posix(ctx, r)
}
fn cb_sce_fstat(ctx: &mut CpuContext) -> HleResult {
    sce(fstat_core(ctx, fd_arg(ctx), ctx.rsi()))
}
fn cb_posix_fstat(ctx: &mut CpuContext) -> HleResult {
    let r = fstat_core(ctx, fd_arg(ctx), ctx.rsi());
    posix(ctx, r)
}
fn cb_sce_getdents(ctx: &mut CpuContext) -> HleResult {
    sce(getdirentries_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx(), 0))
}
fn cb_sce_getdirentries(ctx: &mut CpuContext) -> HleResult {
    sce(getdirentries_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx(), ctx.rcx()))
}
fn cb_sce_mkdir(ctx: &mut CpuContext) -> HleResult {
    sce(mkdir_core(ctx, ctx.rdi()))
}
fn cb_sce_rmdir(ctx: &mut CpuContext) -> HleResult {
    sce(rmdir_core(ctx, ctx.rdi()))
}
fn cb_posix_mkdir(ctx: &mut CpuContext) -> HleResult {
    let r = mkdir_core(ctx, ctx.rdi());
    posix(ctx, r)
}
fn cb_posix_rmdir(ctx: &mut CpuContext) -> HleResult {
    let r = rmdir_core(ctx, ctx.rdi());
    posix(ctx, r)
}
fn cb_sce_unlink(ctx: &mut CpuContext) -> HleResult {
    sce(unlink_core(ctx, ctx.rdi()))
}
fn cb_posix_unlink(ctx: &mut CpuContext) -> HleResult {
    let r = unlink_core(ctx, ctx.rdi());
    posix(ctx, r)
}
fn cb_sce_rename(ctx: &mut CpuContext) -> HleResult {
    sce(rename_core(ctx, ctx.rdi(), ctx.rsi()))
}
fn cb_posix_rename(ctx: &mut CpuContext) -> HleResult {
    posix_rename(ctx, ctx.rdi(), ctx.rsi())
}
fn cb_sce_fsync(ctx: &mut CpuContext) -> HleResult {
    sce(fsync_core(fd_arg(ctx)))
}
fn cb_posix_fsync(ctx: &mut CpuContext) -> HleResult {
    let r = fsync_core(fd_arg(ctx));
    posix(ctx, r)
}
fn cb_posix_sync(_ctx: &mut CpuContext) -> HleResult {
    sync_written_paths();
    0
}
fn cb_sce_ftruncate(ctx: &mut CpuContext) -> HleResult {
    sce(ftruncate_core(fd_arg(ctx), ctx.rsi() as i64))
}
fn cb_posix_ftruncate(ctx: &mut CpuContext) -> HleResult {
    let r = ftruncate_core(fd_arg(ctx), ctx.rsi() as i64);
    posix(ctx, r)
}
fn cb_sce_truncate(ctx: &mut CpuContext) -> HleResult {
    sce(truncate_core(ctx, ctx.rdi(), ctx.rsi() as i64))
}
fn cb_posix_truncate(ctx: &mut CpuContext) -> HleResult {
    let r = truncate_core(ctx, ctx.rdi(), ctx.rsi() as i64);
    posix(ctx, r)
}
fn cb_sce_chmod(ctx: &mut CpuContext) -> HleResult {
    sce(chmod_core(ctx, ctx.rdi(), ctx.rsi() as u32))
}
fn cb_sce_fchmod(ctx: &mut CpuContext) -> HleResult {
    sce(fchmod_core(ctx.rdi() as i32, ctx.rsi() as u32))
}
fn cb_posix_chmod(ctx: &mut CpuContext) -> HleResult {
    let r = chmod_core(ctx, ctx.rdi(), ctx.rsi() as u32);
    posix(ctx, r)
}
fn cb_posix_fchmod(ctx: &mut CpuContext) -> HleResult {
    let r = fchmod_core(fd_arg(ctx), ctx.rsi() as u32);
    posix(ctx, r)
}

/// fchown(fd, uid, gid): the guest runs as uid/gid 0, so owner 0 or -1
/// ("unchanged") is accepted on an open descriptor; any other owner is EPERM.
fn fchown_core(fd: i32, uid: u32, gid: u32) -> FsResult<i64> {
    if get_handle(fd).is_none() && !crate::net::is_kernel_socket(fd) {
        return Err(EBADF);
    }
    let ok = |v: u32| v == 0 || v == u32::MAX;
    if !ok(uid) || !ok(gid) {
        log::error!("[fs] fchown fd={fd} uid={uid} gid={gid}: only owner 0 exists");
        return Err(EPERM);
    }
    Ok(0)
}
fn cb_posix_fchown(ctx: &mut CpuContext) -> HleResult {
    let r = fchown_core(fd_arg(ctx), ctx.rsi() as u32, ctx.rdx() as u32);
    posix(ctx, r)
}
fn cb_sce_utimes(ctx: &mut CpuContext) -> HleResult {
    sce(utimes_core(ctx, ctx.rdi(), ctx.rsi()))
}
fn cb_posix_utimes(ctx: &mut CpuContext) -> HleResult {
    let r = utimes_core(ctx, ctx.rdi(), ctx.rsi());
    posix(ctx, r)
}
/// PATH_MAX of the guest (FreeBSD).
const REALPATH_MAX: usize = 1024;

/// Whether the guest path `guest` names a directory (Err: its errno).
fn guest_path_is_dir(guest: &str) -> FsResult<bool> {
    if let Some((img, rel)) = image_target(guest) {
        let id = img.lookup(&rel).ok_or(ENOENT)?;
        return Ok(img.node(id).is_dir);
    }
    let host = resolve_no_copy(guest).ok_or(ENOENT)?;
    fs::metadata(&host).map(|m| m.is_dir()).map_err(|e| host_errno(&e))
}

/// Lexical realpath of `path` against the root (the guest working
/// directory): "." and ".." are folded, separators may be '/' or '\\',
/// every component must exist and every one but the last be a directory.
/// Err holds the errno and the path resolved up to the failure.
fn realpath_resolve(path: &str) -> Result<String, (i32, String)> {
    let is_separator = |c: char| c == '/' || c == '\\';
    let mut left = path;
    let mut result = String::from("/");
    if left.starts_with(is_separator) {
        left = &left[1..];
    }
    if left.len() >= REALPATH_MAX {
        return Err((ENAMETOOLONG, result));
    }
    while !left.is_empty() {
        let (token, more) = match left.find(is_separator) {
            Some(i) => {
                let token = &left[..i];
                left = &left[i + 1..];
                (token, true)
            }
            None => (std::mem::take(&mut left), false),
        };
        if result.len() > 1 && result.len() + 1 >= REALPATH_MAX {
            return Err((ENAMETOOLONG, result));
        }
        match token {
            "" | "." => continue,
            ".." => {
                if result.len() > 1 {
                    let slash = result.rfind('/').unwrap_or(0);
                    result.truncate(slash.max(1));
                }
                continue;
            }
            _ => {}
        }
        if result.len() > 1 {
            result.push('/');
        }
        result.push_str(token);
        if result.len() >= REALPATH_MAX {
            return Err((ENAMETOOLONG, result));
        }
        match guest_path_is_dir(&result) {
            Ok(is_dir) if more && !is_dir => return Err((ENOTDIR, result)),
            Ok(_) => {}
            Err(e) if e == ENOENT || e == ENOTDIR => return Err((ENOENT, result)),
            Err(e) => return Err((e, result)),
        }
    }
    Ok(result)
}

/// realpath(path, resolved): the resolved path in `resolved`, or in a
/// new PATH_MAX block of the HLE heap when it is NULL. On failure errno is
/// set and a caller's buffer holds the path resolved up to the failure.
fn cb_realpath(ctx: &mut CpuContext) -> HleResult {
    let (path_addr, resolved) = (ctx.rdi(), ctx.rsi());
    if path_addr == 0 {
        set_errno(ctx, EINVAL);
        return 0;
    }
    // realpath bounds and reports every step itself (realpath_resolve).
    let path = match read_guest_cstr(ctx, path_addr, MAX_GUEST_PATH).map(|b| at::absolute(path_addr, String::from_utf8_lossy(&b).into_owned())) {
        Ok(p) => p,
        Err(e) => {
            set_errno(ctx, e);
            return 0;
        }
    };
    if path.is_empty() {
        set_errno(ctx, ENOENT);
        return 0;
    }
    let text = match realpath_resolve(&path) {
        Ok(t) => t,
        Err((errno, trouble)) => {
            if resolved != 0 {
                let mut b = trouble.into_bytes();
                b.truncate(REALPATH_MAX - 1);
                b.push(0);
                let _ = write_guest(ctx, resolved, &b);
            }
            set_errno(ctx, errno);
            return 0;
        }
    };
    let out = if resolved != 0 { resolved } else { crate::libc::heap_block(ctx, REALPATH_MAX as u64) };
    if out == 0 {
        set_errno(ctx, ENOMEM);
        return 0;
    }
    let mut b = text.into_bytes();
    b.push(0);
    match write_guest(ctx, out, &b) {
        Ok(()) => out as HleResult,
        Err(e) => {
            set_errno(ctx, e);
            0
        }
    }
}

/// utime(path, times): errno is left alone on success.
fn cb_posix_utime(ctx: &mut CpuContext) -> HleResult {
    let r = utime_core(ctx, ctx.rdi(), ctx.rsi());
    posix(ctx, r)
}
fn cb_sce_futimes(ctx: &mut CpuContext) -> HleResult {
    sce(futimes_core(ctx, fd_arg(ctx), ctx.rsi()))
}
fn cb_posix_futimes(ctx: &mut CpuContext) -> HleResult {
    let r = futimes_core(ctx, fd_arg(ctx), ctx.rsi());
    posix(ctx, r)
}
fn cb_sce_check_reachability(ctx: &mut CpuContext) -> HleResult {
    sce(check_reachability_core(ctx, ctx.rdi()))
}
fn cb_posix_dup(ctx: &mut CpuContext) -> HleResult {
    let r = dup_core(fd_arg(ctx));
    posix(ctx, r)
}
fn cb_posix_dup2(ctx: &mut CpuContext) -> HleResult {
    let r = dup2_core(fd_arg(ctx), ctx.rsi() as i32);
    posix(ctx, r)
}
fn cb_sce_fcntl(ctx: &mut CpuContext) -> HleResult {
    let r = fcntl_with_locks(ctx, fd_arg(ctx), ctx.rsi() as u32 as u64, ctx.rdx());
    sce(r)
}
fn cb_posix_fcntl(ctx: &mut CpuContext) -> HleResult {
    let r = fcntl_with_locks(ctx, fd_arg(ctx), ctx.rsi() as u32 as u64, ctx.rdx());
    posix(ctx, r)
}

/// fcntl with the record-lock commands, which read and write a guest
/// `struct flock`.
fn fcntl_with_locks(ctx: &CpuContext, fd: i32, cmd: u64, arg: u64) -> FsResult<i64> {
    match cmd {
        7..=9 => record_lock(ctx, fd, cmd + 4, arg, true),
        11..=14 => record_lock(ctx, fd, cmd, arg, false),
        _ => fcntl_core(fd, cmd, arg),
    }
}

/// Advisory record locks (F_GETLK 11, F_SETLK 12, F_SETLKW 13,
/// F_SETLK_REMOTE 14; 7..9 are the old commands with the 24-byte struct
/// without l_sysid). The guest is one process and a process never conflicts
/// with its own locks (FreeBSD lf_blocks), so a set always succeeds and a
/// get reports F_UNLCK, after FreeBSD's checks of the request.
///
/// Guest `struct flock`: i64 l_start @0, i64 l_len @8, i32 l_pid @16,
/// i16 l_type @20, i16 l_whence @22, i32 l_sysid @24.
fn record_lock(ctx: &CpuContext, fd: i32, cmd: u64, arg: u64, old: bool) -> FsResult<i64> {
    const F_GETLK: u64 = 11;
    const F_RDLCK: i16 = 1;
    const F_UNLCK: i16 = 2;
    const F_WRLCK: i16 = 3;
    if arg == 0 {
        return Err(EFAULT);
    }
    let size = if old { 24 } else { 32 };
    let mut fl = ctx.read_bytes(arg, size).ok_or(EFAULT)?;
    match get_handle(fd) {
        Some(Handle::File(_)) | Some(Handle::Dir(_)) => {}
        _ => return Err(EBADF),
    }
    let mut start = i64::from_le_bytes(fl[0..8].try_into().unwrap());
    let len = i64::from_le_bytes(fl[8..16].try_into().unwrap());
    let ty = i16::from_le_bytes([fl[20], fl[21]]);
    let whence = i16::from_le_bytes([fl[22], fl[23]]) as i32;
    if !matches!(ty, F_RDLCK | F_UNLCK | F_WRLCK) {
        return Err(EINVAL);
    }
    if whence == SEEK_CUR {
        let off = lseek_core(fd, 0, SEEK_CUR)?;
        if start > 0 && off > i64::MAX - start {
            return Err(EOVERFLOW);
        }
        start += off;
    }
    if cmd != F_GETLK && ty != F_UNLCK {
        let access = fds().status.get(&fd).map(|st| *st.lock().unwrap_or_else(|p| p.into_inner()) as u64 & O_ACCMODE);
        match (ty, access) {
            (F_RDLCK, Some(O_WRONLY)) | (F_WRLCK, Some(0)) => return Err(EBADF),
            _ => {}
        }
    }
    match whence {
        SEEK_SET | SEEK_CUR => {}
        SEEK_END => {
            let size = match get_handle(fd) {
                Some(Handle::File(f)) => f.file.len().map_err(|e| host_errno(&e))? as i64,
                _ => 0,
            };
            if start > 0 && size > i64::MAX - start {
                return Err(EOVERFLOW);
            }
            start += size;
        }
        _ => return Err(EINVAL),
    }
    if start < 0 || (len < 0 && (start == 0 || start + len < 0)) {
        return Err(EINVAL);
    }
    if len > 0 && len - 1 > i64::MAX - start {
        return Err(EOVERFLOW);
    }
    if cmd != F_GETLK {
        return Ok(0);
    }
    fl[0..8].copy_from_slice(&start.to_le_bytes());
    fl[20..22].copy_from_slice(&F_UNLCK.to_le_bytes());
    if ctx.write_bytes(arg, &fl) { Ok(0) } else { Err(EFAULT) }
}
fn cb_sce_fdatasync(ctx: &mut CpuContext) -> HleResult {
    sce(fdatasync_core(fd_arg(ctx)))
}
fn cb_posix_fdatasync(ctx: &mut CpuContext) -> HleResult {
    let r = fdatasync_core(fd_arg(ctx));
    posix(ctx, r)
}
fn cb_posix_lstat(ctx: &mut CpuContext) -> HleResult {
    let r = lstat_core(ctx, ctx.rdi(), ctx.rsi());
    posix(ctx, r)
}
fn cb_posix_ioctl(ctx: &mut CpuContext) -> HleResult {
    let r = ioctl_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx());
    posix(ctx, r)
}
fn cb_posix_link(ctx: &mut CpuContext) -> HleResult {
    let r = link_core(ctx, ctx.rdi(), ctx.rsi());
    posix(ctx, r)
}
fn cb_posix_symlink(ctx: &mut CpuContext) -> HleResult {
    let r = symlink_core(ctx, ctx.rdi(), ctx.rsi());
    posix(ctx, r)
}
fn cb_posix_readlink(ctx: &mut CpuContext) -> HleResult {
    let r = readlink_core(ctx, ctx.rdi(), ctx.rsi(), ctx.rdx());
    posix(ctx, r)
}
fn cb_posix_fstatfs(ctx: &mut CpuContext) -> HleResult {
    let r = fstatfs_core(ctx, fd_arg(ctx), ctx.rsi());
    posix(ctx, r)
}
fn cb_posix_getdents(ctx: &mut CpuContext) -> HleResult {
    let r = getdirentries_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx(), 0);
    posix(ctx, r)
}
fn cb_posix_getdirentries(ctx: &mut CpuContext) -> HleResult {
    let r = getdirentries_core(ctx, fd_arg(ctx), ctx.rsi(), ctx.rdx(), ctx.rcx());
    posix(ctx, r)
}
fn cb_posix_shm_open(ctx: &mut CpuContext) -> HleResult {
    let r = shm_open_core(ctx, ctx.rdi(), ctx.rsi());
    posix(ctx, r)
}

// ---------------------------------------------------------------------------
// shm_open
// ---------------------------------------------------------------------------

/// `SHM_ANON`: a new anonymous object nobody else can open.
const SHM_ANON: u64 = 1;

/// Named shared memory objects: name -> backing memfd (kept for the life of
/// the process; each open gets its own open file over it).
static SHM_OBJECTS: LazyLock<Mutex<HashMap<Vec<u8>, File>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn memfd(name: &str) -> FsResult<File> {
    crate::host::fs::memfd(name).map_err(|e| host_errno(&e))
}

/// shm_open(path, flags, mode): a descriptor over a shared memory object,
/// sized with ftruncate and accessed with read/write/pread/pwrite. Access
/// must be O_RDONLY or O_RDWR and only O_CREAT, O_EXCL, O_TRUNC and
/// O_CLOEXEC may be added (EINVAL otherwise); an anonymous object needs
/// O_RDWR. O_TRUNC empties a named object opened O_RDWR.
fn shm_open_core(ctx: &CpuContext, path: u64, flags: u64) -> FsResult<i64> {
    let access = flags & O_ACCMODE;
    if access != 0 && access != O_RDWR {
        return Err(EINVAL);
    }
    if flags & !(O_ACCMODE | O_CREAT | O_EXCL | O_TRUNC | O_CLOEXEC) != 0 {
        return Err(EINVAL);
    }
    let writable = access == O_RDWR;
    let (file, shown) = if path == SHM_ANON {
        if !writable {
            return Err(EINVAL);
        }
        (memfd("shm_open anonymous")?, PathBuf::from("shm:anonymous"))
    } else {
        let name = read_guest_cstr(ctx, path, MAX_GUEST_PATH)?;
        let mut objects = SHM_OBJECTS.lock().unwrap_or_else(|p| p.into_inner());
        if objects.contains_key(&name) {
            if flags & (O_CREAT | O_EXCL) == O_CREAT | O_EXCL {
                return Err(EEXIST);
            }
        } else {
            if flags & O_CREAT == 0 {
                return Err(ENOENT);
            }
            objects.insert(name.clone(), memfd("shm_open")?);
        }
        // A fresh open of the object, with its own access mode and offset.
        let file = crate::host::fs::reopen(&objects[&name], writable).map_err(|e| host_errno(&e))?;
        if writable && flags & O_TRUNC != 0 {
            file.set_len(0).map_err(|e| host_errno(&e))?;
        }
        (file, PathBuf::from(format!("shm:{}", String::from_utf8_lossy(&name))))
    };
    let fd = alloc_fd(Handle::File(Arc::new(OpenFile::host(file, shown))));
    set_open_cloexec(fd, flags);
    Ok(fd as i64)
}

// ---------------------------------------------------------------------------
// Sandbox
// ---------------------------------------------------------------------------

/// Name of the directory under which a title's sandbox shows system files
/// (`/<word>/common/...`). Nothing is mounted there.
const SANDBOX_RANDOM_WORD: &[u8] = b"gkbHtXc3Wq\0";
static SANDBOX_WORD_ADDR: Mutex<u64> = Mutex::new(0);

/// sceKernelGetFsSandboxRandomWord(): the same NUL-terminated word on every
/// call and thread.
fn cb_get_fs_sandbox_random_word(ctx: &mut CpuContext) -> HleResult {
    let mut addr = SANDBOX_WORD_ADDR.lock().unwrap_or_else(|p| p.into_inner());
    if *addr == 0 {
        let Some(a) = crate::pthread::alloc_guest_block(ctx, SANDBOX_RANDOM_WORD.len() as u64) else {
            return 0;
        };
        if !ctx.write_bytes(a, SANDBOX_RANDOM_WORD) {
            return 0;
        }
        *addr = a;
    }
    *addr as HleResult
}

/// (nid, name, library, callback) for every export in this module.
const EXPORTS: &[(&str, &str, &str, HleCallback)] = &[
    // open / close
    ("1G3lF1Gg1k8", "sceKernelOpen", "libKernel", cb_sce_open),
    ("wuCroIGjt2g", "open", "libKernel", cb_posix_open),
    ("5AMS0IntU8I", "creat", "libKernel", cb_posix_creat),
    ("6c3rCVE-fTU", "_open", "libKernel", cb_posix_open),
    ("UK2Tl2DWUns", "sceKernelClose", "libKernel", cb_sce_close),
    ("bY-PO6JhzhQ", "close", "libKernel", cb_posix_close),
    ("-Jp7F+pXxNg", "pipe", "libKernel", cb_posix_pipe),
    ("NNtFaKJbPt0", "_close", "libKernel", cb_posix_close),
    // read / write / seek
    ("Cg4srZ6TKbU", "sceKernelRead", "libKernel", cb_sce_read),
    ("AqBioC2vF3I", "read", "libKernel", cb_posix_read),
    ("DRuBt2pvICk", "_read", "libKernel", cb_posix_read),
    ("4wSze92BhLI", "sceKernelWrite", "libKernel", cb_sce_write),
    ("FN4gaPmuFV8", "write", "libKernel", cb_posix_write),
    ("FxVZqBAA7ks", "_write", "libKernel", cb_posix_write),
    ("+r3rMFwItV4", "sceKernelPread", "libKernel", cb_sce_pread),
    ("ezv-RSBNKqI", "pread", "libKernel", cb_posix_pread),
    ("nKWi-N2HBV4", "sceKernelPwrite", "libKernel", cb_sce_pwrite),
    ("C2kJ-byS5rM", "pwrite", "libKernel", cb_posix_pwrite),
    ("QqxBetgJH+g", "sceKernelReadv", "libKernel", cb_sce_readv),
    ("kAt6VDbHmro", "sceKernelWritev", "libKernel", cb_sce_writev),
    ("I7ImcLds-uU", "readv", "libKernel", cb_posix_readv),
    ("Z2aKdxzS4KE", "writev", "libKernel", cb_posix_writev),
    ("+WRlkKjZvag", "_readv", "libKernel", cb_posix_readv),
    ("YSHRBRLn2pI", "_writev", "libKernel", cb_posix_writev),
    ("yTj62I7kw4s", "sceKernelPreadv", "libKernel", cb_sce_preadv),
    ("mBd4AfLP+u8", "sceKernelPwritev", "libKernel", cb_sce_pwritev),
    ("ZaRzaapAZwM", "preadv", "libKernel", cb_posix_preadv),
    ("FCcmRZhWtOk", "pwritev", "libKernel", cb_posix_pwritev),
    ("9eMlfusH4sU", "flock", "libKernel", cb_posix_flock),
    ("oib76F-12fk", "sceKernelLseek", "libKernel", cb_sce_lseek),
    ("Oy6IpwgtYOk", "lseek", "libKernel", cb_posix_lseek),
    // stat / directories
    ("eV9wAD2riIA", "sceKernelStat", "libKernel", cb_sce_stat),
    ("E6ao34wPw+U", "stat", "libc", cb_posix_stat),
    ("kBwCPsYX-m4", "sceKernelFstat", "libKernel", cb_sce_fstat),
    ("mqQMh1zPPT8", "fstat", "libc", cb_posix_fstat),
    ("A0O5kF5x4LQ", "_fstat", "libKernel", cb_posix_fstat),
    ("j2AIqSqJP0w", "sceKernelGetdents", "libKernel", cb_sce_getdents),
    ("taRWhTJFTgE", "sceKernelGetdirentries", "libKernel", cb_sce_getdirentries),
    ("2G6i6hMIUUY", "getdents", "libKernel", cb_posix_getdents),
    ("f09KvIPy-QY", "getdirentries", "libKernel", cb_posix_getdirentries),
    ("DRGXpDDh8Ng", "lstat", "libKernel", cb_posix_lstat),
    ("pd02UI9TbOA", "readlink", "libkernel_sys", cb_posix_readlink),
    ("Pd7KtJ1gybk", "link", "libkernel_sys", cb_posix_link),
    ("PfccT7qURYE", "ioctl", "libKernel", cb_posix_ioctl),
    ("wW+k21cmbwQ", "_ioctl", "libKernel", cb_posix_ioctl),
    ("waIF+5fE5HQ", "symlink", "libkernel_sys", cb_posix_symlink),
    ("17Mfe1B3X6U", "_fstatfs", "libKernel", cb_posix_fstatfs),
    ("+rWFNYiNaEM", "fstatfs", "libkernel_sys", cb_posix_fstatfs),
    ("QuJYZ2KVGGQ", "shm_open", "libKernel", cb_posix_shm_open),
    ("JGfTMBOdUJo", "sceKernelGetFsSandboxRandomWord", "libKernel", cb_get_fs_sandbox_random_word),
    ("1-LFLmRFxxM", "sceKernelMkdir", "libKernel", cb_sce_mkdir),
    ("naInUjYt3so", "sceKernelRmdir", "libKernel", cb_sce_rmdir),
    ("JGMio+21L4c", "mkdir", "libKernel", cb_posix_mkdir),
    ("c7ZnT7V1B98", "rmdir", "libKernel", cb_posix_rmdir),
    // path mutations
    ("AUXVxWeJU-A", "sceKernelUnlink", "libKernel", cb_sce_unlink),
    ("VAzswvTOCzI", "unlink", "libKernel", cb_posix_unlink),
    ("52NcYU9+lEo", "sceKernelRename", "libKernel", cb_sce_rename),
    ("NN01qLRhiqU", "rename", "libKernel", cb_posix_rename),
    ("WlyEA-sLDf0", "sceKernelTruncate", "libKernel", cb_sce_truncate),
    ("ayrtszI7GBg", "truncate", "libKernel", cb_posix_truncate),
    ("VW3TVZiM4-E", "sceKernelFtruncate", "libKernel", cb_sce_ftruncate),
    ("ih4CD9-gghM", "ftruncate", "libKernel", cb_posix_ftruncate),
    ("fgIsQ10xYVA", "sceKernelChmod", "libKernel", cb_sce_chmod),
    ("uWyW3v98sU4", "sceKernelCheckReachability", "libKernel", cb_sce_check_reachability),
    ("z0dtnPxYgtg", "chmod", "libKernel", cb_posix_chmod),
    ("UtszJWHrDcA", "sceKernelFchmod", "libKernel", cb_sce_fchmod),
    ("n01yNbQO5W4", "fchmod", "libKernel", cb_posix_fchmod),
    ("ruyD9QkP8dI", "fchown", "libkernel_sys", cb_posix_fchown),
    ("0Cq8ipKr9n0", "sceKernelUtimes", "libKernel", cb_sce_utimes),
    ("GDuV00CHrUg", "utimes", "libKernel", cb_posix_utimes),
    ("NLq2b1jOaN0", "sceKernelFutimes", "libKernel", cb_sce_futimes),
    ("+0EDo7YzcoU", "futimes", "libKernel", cb_posix_futimes),
    // sync
    ("fTx66l5iWIA", "sceKernelFsync", "libKernel", cb_sce_fsync),
    ("juWbTNM+8hw", "fsync", "libKernel", cb_posix_fsync),
    ("30Rh4ixbKy4", "sceKernelFdatasync", "libKernel", cb_sce_fdatasync),
    ("KIbJFQ0I1Cg", "fdatasync", "libKernel", cb_posix_fdatasync),
    ("Y2OqwJQ3lr8", "sync", "libKernel", cb_posix_sync),
    // descriptors
    ("iiQjzvfWDq0", "dup", "libKernel", cb_posix_dup),
    ("wdUufa9g-D8", "dup2", "libKernel", cb_posix_dup2),
    ("8nY19bKoiZk", "fcntl", "libKernel", cb_posix_fcntl),
    ("t0fXUzq61Z4", "_fcntl", "libKernel", cb_posix_fcntl),
    ("SoZkxZkCHaw", "sceKernelFcntl", "libKernel", cb_sce_fcntl),
    ("ku7D4q1Y9PI", "poll", "libKernel", cb_poll),
    ("T8fER+tIGgk", "select", "libKernel", cb_select),
    // AIO
    ("nu4a0-arQis", "sceKernelAioInitializeParam", "libKernel", cb_aio_initialize_param),
    ("vYU8P9Td2Zo", "sceKernelAioInitializeImpl", "libKernel", cb_aio_initialize_impl),
    ("HgX7+AORI58", "sceKernelAioSubmitReadCommands", "libKernel", cb_aio_submit_read),
    ("lXT0m3P-vs4", "sceKernelAioSubmitReadCommandsMultiple", "libKernel", cb_aio_submit_read_multiple),
    ("XQ8C8y+de+E", "sceKernelAioSubmitWriteCommands", "libKernel", cb_aio_submit_write),
    ("xT3Cpz0yh6Y", "sceKernelAioSubmitWriteCommandsMultiple", "libKernel", cb_aio_submit_write_multiple),
    ("o7O4z3jwKzo", "sceKernelAioPollRequests", "libKernel", cb_aio_poll_requests),
    ("2pOuoWoCxdk", "sceKernelAioPollRequest", "libKernel", cb_aio_poll_request),
    ("lgK+oIWkJyA", "sceKernelAioWaitRequests", "libKernel", cb_aio_wait_requests),
    ("KOF-oJbQVvc", "sceKernelAioWaitRequest", "libKernel", cb_aio_wait_request),
    ("fR521KIGgb8", "sceKernelAioCancelRequest", "libKernel", cb_aio_cancel_request),
    ("3Lca1XBrQdY", "sceKernelAioCancelRequests", "libKernel", cb_aio_cancel_requests),
    ("9WK-vhNXimw", "sceKernelAioSetParam", "libKernel", cb_aio_set_param),
    ("5TgME6AYty4", "sceKernelAioDeleteRequest", "libKernel", cb_aio_delete_request),
    ("Ft3EtsZzAoY", "sceKernelAioDeleteRequests", "libKernel", cb_aio_delete_requests),
    // APR
    ("gEpBkcwxUjw", "sceKernelAprResolveFilepathsToIdsAndFileSizes", "libKernel", cb_apr_resolve_ids_and_sizes),
    (
        "w5fcCG+t31g",
        "sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizes",
        "libKernel",
        cb_apr_resolve_with_prefix_ids_and_sizes,
    ),
    ("WT-5NKy42fw", "sceKernelAprResolveFilepathsToIds", "libKernel", cb_apr_resolve_ids),
    ("i3HWvW35jao", "sceKernelAprResolveFilepathsWithPrefixToIds", "libKernel", cb_apr_resolve_with_prefix_ids),
    ("eYAh2vlCY-U", "sceKernelAprResolveFilepathsToIdsForEach", "libKernel", cb_apr_resolve_ids_for_each),
    (
        "QzB4O+bJQyA",
        "sceKernelAprResolveFilepathsToIdsAndFileSizesForEach",
        "libKernel",
        cb_apr_resolve_ids_and_sizes_for_each,
    ),
    (
        "VB-BtuIW8Xc",
        "sceKernelAprResolveFilepathsWithPrefixToIdsForEach",
        "libKernel",
        cb_apr_resolve_with_prefix_ids_for_each,
    ),
    (
        "C+Khtbbx2g8",
        "sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizesForEach",
        "libKernel",
        cb_apr_resolve_with_prefix_ids_and_sizes_for_each,
    ),
    ("ApkYaHb8Sek", "sceKernelAprGetFileStat", "libKernel", cb_apr_get_file_stat),
    ("WvEu7yl3Ivg", "sceKernelAprGetFileSize", "libKernel", cb_apr_get_file_size),
];

/// Register the file system exports. Must run after the older kernel
/// registrations so these implementations replace their stubs.
pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, library, cb) in EXPORTS.iter().chain(at::AT_EXPORTS) {
        reg.register_fn(nid, name, library, cb);
    }
    // The sceSp* wrapper of libSceSpSysCallWrapper forwards to select.
    reg.register_fn("YsLBZ7qPu20", "sceSpSelect", "libSceSpSysCallWrapper", cb_select);
    for &(nid, name, library, cb) in LIBC_FALLBACK_EXPORTS {
        reg.register_fallback_fn(nid, name, library, cb);
    }
}

/// libc file functions over the kernel's, used when no loaded libc module
/// exports them.
const LIBC_FALLBACK_EXPORTS: &[(&str, &str, &str, HleCallback)] = &[
    ("1nTKA7pN1jw", "utime", "libSceLibcInternal", cb_posix_utime),
    ("vhtcIgZG-Lk", "realpath", "libSceLibcInternal", cb_realpath),
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Held by tests that point /app0 at their own directory.
#[cfg(test)]
pub(crate) static APP0_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(all(test, unix))]
mod ported_tests;

pub(crate) mod pipe;
mod at;

/// Test hooks for other modules: open a host file or directory straight
/// into the descriptor table, and the core read/write/close calls.
#[cfg(test)]
pub(crate) mod test_hooks {
    use super::*;

    pub(crate) fn open_host_file(path: &Path) -> i32 {
        let f = fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path).unwrap();
        alloc_fd(Handle::File(Arc::new(OpenFile::host(f, path.to_path_buf()))))
    }

    #[cfg_attr(windows, allow(dead_code))]
    pub(crate) fn open_host_dir(path: &Path) -> i32 {
        alloc_fd(Handle::Dir(Arc::new(Mutex::new(OpenDir {
            guest: path.to_string_lossy().into_owned(),
            host: path.to_path_buf(),
            dirents: Vec::new(),
            offset: 0,
            image: None,
        }))))
    }

    pub(crate) fn write(ctx: &CpuContext, fd: i32, buf: u64, len: u64) -> Result<i64, i32> {
        write_core(ctx, fd, buf, len)
    }

    pub(crate) fn read(ctx: &CpuContext, fd: i32, buf: u64, len: u64) -> Result<i64, i32> {
        read_core(ctx, fd, buf, len)
    }

    pub(crate) fn close(fd: i32) -> Result<i64, i32> {
        close_core(fd)
    }

    pub(crate) fn pipe(ctx: &CpuContext, out: u64) -> (i32, i32) {
        assert_eq!(pipe::pipe_core(ctx, out), Ok(0));
        (ctx.read_u32(out).unwrap() as i32, ctx.read_u32(out + 4).unwrap() as i32)
    }

    /// A host file opened as open(path, flags) records its flags.
    pub(crate) fn open_host_file_with(path: &Path, flags: u64) -> i32 {
        let fd = open_host_file(path);
        set_open_cloexec(fd, flags);
        fd
    }

    pub(crate) fn fcntl(fd: i32, cmd: u64, arg: u64) -> FsResult<i64> {
        fcntl_core(fd, cmd, arg)
    }

    pub(crate) fn ioctl(ctx: &CpuContext, fd: i32, request: u64, arg: u64) -> FsResult<i64> {
        ioctl_core(ctx, fd, request, arg)
    }

    pub(crate) fn fstat(ctx: &CpuContext, fd: i32, out: u64) -> FsResult<i64> {
        fstat_core(ctx, fd, out)
    }

    pub(crate) fn select(ctx: &CpuContext, nfds: i32, sets: [u64; 3], timeout: u64) -> FsResult<i64> {
        select_core(ctx, nfds, sets, timeout)
    }

    pub(crate) fn poll(ctx: &CpuContext, fds: u64, count: u32, timeout: i32) -> FsResult<i64> {
        poll_core(ctx, fds, count, timeout)
    }
}

/// pipe(int fds[2])
fn cb_posix_pipe(ctx: &mut CpuContext) -> HleResult {
    let r = pipe::pipe_core(ctx, ctx.rdi());
    posix(ctx, r)
}

/// What a vnode watch (EVFILT_VNODE, sceKernelAddFileEvent) of a
/// descriptor follows.
#[cfg_attr(windows, allow(dead_code))]
pub(crate) enum WatchTarget {
    /// A host file: its descriptor, valid while the table entry lives.
    File(crate::host::fs::RawFd),
    /// A host directory.
    Dir(PathBuf),
    /// A file or directory of the read-only `/app0` image: it never changes.
    Image,
    Fifo,
    Device,
}

pub(crate) fn watch_target(fd: i32) -> Option<WatchTarget> {
    Some(match get_handle(fd)? {
        Handle::File(f) => match &f.file {
            FileSource::Host(h) => WatchTarget::File(crate::host::fs::as_raw_fd(h)),
            FileSource::Image(_) => WatchTarget::Image,
        },
        Handle::Dir(d) => {
            let d = d.lock().unwrap_or_else(|p| p.into_inner());
            if d.image.is_some() { WatchTarget::Image } else { WatchTarget::Dir(d.host.clone()) }
        }
        Handle::Pipe(_) => WatchTarget::Fifo,
        Handle::Random => WatchTarget::Device,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn temp_root(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("rps5e-fs-test-{}-{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn state_with(config: MountConfig) -> FsState {
        FsState {
            config,
            title_id: "TEST00000".into(),
            temp0_reset: true,
            resolved: HashMap::new(),
            verified_components: HashSet::new(),
            negative_stat: HashSet::new(),
            image: None,
        }
    }

    #[test]
    #[cfg(unix)]
    fn case_insensitive_component_lookup() {
        let dir = std::env::temp_dir().join(format!("rps5e-ci-{}", std::process::id()));
        fs::create_dir_all(dir.join("prx")).unwrap();
        fs::write(dir.join("prx/akdelay.prx"), b"x").unwrap();
        let mut st = state_with(MountConfig::default());
        let p = combine_within(&mut st, &dir, "Prx/AkDelay.prx").unwrap();
        assert_eq!(p, dir.join("prx/akdelay.prx"));
        let missing = combine_within(&mut st, &dir, "prx/new.txt").unwrap();
        assert_eq!(missing, dir.join("prx/new.txt"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn nids_match_names() {
        // NID = base64(rev(sha1(name || salt)[..8])) with '+' and '-'.
        for &(nid, name, _, _) in EXPORTS.iter().chain(LIBC_FALLBACK_EXPORTS) {
            assert_eq!(ps5nid::compute_nid(name).replace('_', "-"), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn relative_normalization_clamps_at_root() {
        assert_eq!(normalize_relative("a/./b/../c"), vec!["a", "c"]);
        assert_eq!(normalize_relative("../../../Shpc/x.pak"), vec!["Shpc", "x.pak"]);
        assert_eq!(normalize_relative("a\\b//c/"), vec!["a", "b", "c"]);
        assert!(normalize_relative("..").is_empty());
    }

    #[test]
    fn guest_path_normalization() {
        assert_eq!(normalize_guest_path("app0\\x\\"), Some("/app0/x".into()));
        assert_eq!(normalize_guest_path("/"), Some("/".into()));
        assert_eq!(normalize_guest_path(""), None);
        assert_eq!(strip_mount("/APP0/Data", "/app0"), Some("Data"));
        assert_eq!(strip_mount("/app0x", "/app0"), None);
        assert_eq!(strip_mount("/app0", "/app0"), Some(""));
    }

    #[test]
    fn resolves_builtin_mounts() {
        let root = temp_root("mounts");
        let app0 = root.join("game");
        fs::create_dir_all(app0.join("shpc/content")).unwrap();
        let mut st = state_with(MountConfig {
            app0: Some(app0.clone()),
            temp0: Some(root.join("temp0")),
            download0: Some(root.join("dl")),
            hostapp: Some(root.join("hostapp")),
            devlog_app: Some(root.join("devlog")),
            data: Some(root.join("data")),
            ..Default::default()
        });
        let r = |st: &mut FsState, g: &str| resolve_uncached(st, g);
        assert_eq!(r(&mut st, "/app0/shpc/content"), Some(app0.join("shpc/content")));
        assert_eq!(r(&mut st, "/APP0/shpc"), Some(app0.join("shpc")));
        assert_eq!(r(&mut st, "app0/shpc"), Some(app0.join("shpc")));
        assert_eq!(r(&mut st, "$/shpc"), Some(app0.join("shpc")));
        assert_eq!(r(&mut st, "$"), Some(app0.clone()));
        assert_eq!(r(&mut st, "/app0"), Some(app0.clone()));
        assert_eq!(r(&mut st, "/contents/a.bin"), Some(app0.join("contents/a.bin")));
        assert_eq!(r(&mut st, "shpc/x"), Some(app0.join("shpc/x")));
        assert_eq!(r(&mut st, "/temp0/log.txt"), Some(root.join("temp0/log.txt")));
        assert_eq!(r(&mut st, "/download0/a"), Some(root.join("dl/a")));
        assert_eq!(r(&mut st, "/hostapp/a"), Some(root.join("hostapp/a")));
        assert_eq!(r(&mut st, "/devlog/app/x"), Some(root.join("devlog/x")));
        assert_eq!(r(&mut st, "/data/x"), Some(root.join("data/x")));
        // Traversal is clamped at the mount root.
        assert_eq!(r(&mut st, "/app0/../../etc/passwd"), Some(app0.join("etc/passwd")));
        assert_eq!(r(&mut st, "../../../shpc"), Some(app0.join("shpc")));
        // Unknown absolute paths are denied.
        assert_eq!(r(&mut st, "/etc/passwd"), None);
        assert_eq!(r(&mut st, "/temp0x/a"), None);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    #[cfg(unix)]
    fn registered_mounts_win_and_deny_symlink_escape() {
        let root = temp_root("registered");
        let save = root.join("save");
        fs::create_dir_all(&save).unwrap();
        std::os::unix::fs::symlink("/", save.join("escape")).unwrap();
        fs::create_dir_all(save.join("inner")).unwrap();
        std::os::unix::fs::symlink(save.join("inner"), save.join("ok")).unwrap();
        let mut st = state_with(MountConfig {
            registered: vec![("/savedata0".into(), save.clone())],
            ..Default::default()
        });
        assert_eq!(resolve_uncached(&mut st, "/savedata0/a.sav"), Some(save.join("a.sav")));
        assert_eq!(resolve_uncached(&mut st, "/SAVEDATA0"), Some(save.clone()));
        assert_eq!(resolve_uncached(&mut st, "/savedata0/escape/etc/passwd"), None);
        assert_eq!(resolve_uncached(&mut st, "/savedata0/ok/f"), Some(save.join("ok/f")));
        assert_eq!(resolve_uncached(&mut st, "/savedata1/a"), None);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    #[cfg(unix)]
    fn stat_layout() {
        let t = UNIX_EPOCH + std::time::Duration::new(1_700_000_000, 123);
        let b = build_kernel_stat(false, 1000, t, t, t, "/x");
        assert_eq!(b.len(), 120);
        assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), fnv1a32(b"/x"));
        assert_eq!(u16::from_le_bytes(b[8..10].try_into().unwrap()), 0o100777);
        assert_eq!(u16::from_le_bytes(b[10..12].try_into().unwrap()), 1);
        assert_eq!(i64::from_le_bytes(b[24..32].try_into().unwrap()), 1_700_000_000);
        assert_eq!(i64::from_le_bytes(b[32..40].try_into().unwrap()), 123);
        assert_eq!(i64::from_le_bytes(b[56..64].try_into().unwrap()), 1_700_000_000);
        assert_eq!(i64::from_le_bytes(b[72..80].try_into().unwrap()), 1000);
        assert_eq!(i64::from_le_bytes(b[80..88].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(b[88..92].try_into().unwrap()), 512);
        assert_eq!(i64::from_le_bytes(b[104..112].try_into().unwrap()), 1_700_000_000);
        let d = build_kernel_stat(true, 1024, t, t, t, "/d");
        assert_eq!(u16::from_le_bytes(d[8..10].try_into().unwrap()), 0o040777);
        assert_eq!(i64::from_le_bytes(d[80..88].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(d[88..92].try_into().unwrap()), 512);
    }

    #[test]
    fn dirent_layout() {
        let entries: Vec<(String, bool)> =
            [(".", false), ("..", false), ("Content", false), ("a.pak", true)].iter().map(|(n, f)| (n.to_string(), *f)).collect();
        let b = pack_dirents(&entries);
        assert_eq!(b.len(), 512, "one padded block");
        // "." -> reclen 12
        assert_eq!(u32::from_le_bytes(b[0..4].try_into().unwrap()), name_hash(b"."));
        assert_eq!(u16::from_le_bytes(b[4..6].try_into().unwrap()), 12);
        assert_eq!(b[6], DT_DIR);
        assert_eq!(b[7], 1);
        assert_eq!(&b[8..10], b".\0");
        // "Content" starts after "." (12) and ".." (12): reclen (8+7+1+3)&!3 = 16
        let c = 24;
        assert_eq!(u16::from_le_bytes(b[c + 4..c + 6].try_into().unwrap()), 16);
        assert_eq!(b[c + 7], 7);
        assert_eq!(&b[c + 8..c + 15], b"Content");
        // The last record absorbs the rest of the block.
        let p = c + 16;
        assert_eq!(b[p + 6], DT_REG);
        assert_eq!(u16::from_le_bytes(b[p + 4..p + 6].try_into().unwrap()) as usize, 512 - p);
        // Records never cross blocks: 48 names of 28-byte records need 3 blocks.
        let many: Vec<(String, bool)> = (0..48).map(|i| (format!("directory-entry-{i:02}"), true)).collect();
        let d = pack_dirents(&many);
        assert_eq!(d.len() % 512, 0);
        for block in d.chunks(512) {
            let mut pos = 0;
            while pos < 512 {
                let reclen = u16::from_le_bytes([block[pos + 4], block[pos + 5]]) as usize;
                let namlen = block[pos + 7] as usize;
                assert!(reclen >= 9 + namlen && pos + reclen <= 512 && block[pos + 8 + namlen] == 0);
                pos += reclen;
            }
        }
        // Jenkins one-at-a-time.
        assert_eq!(name_hash(b""), 0);
        assert_eq!(name_hash(b"a"), 0xca2e9442);
    }

    /// APR ids keep the native host path: a game installed under a
    /// non-ASCII folder resolves, stats, sizes and reads through its id.
    #[test]
    fn apr_non_ascii_host_folder() {
        let root = temp_root("apr-unicode").join("Инсталляторы").join("[ps5]");
        fs::create_dir_all(root.join("content/paks")).unwrap();
        fs::write(root.join("content/paks/global.utoc"), b"UTOC").unwrap();
        let _app0 = APP0_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        set_app0_root(&root);
        let (id, size) = apr_resolve_one("/app0/content/paks/global.utoc").expect("resolve");
        assert_eq!(size, 4);
        let host = apr_host_path(id).expect("host path for id");
        assert!(host.to_string_lossy().contains("Инсталляторы"));
        assert_eq!(fs::read(&host).unwrap(), b"UTOC");
        assert!(apr_stat(&host).is_ok());
        assert_eq!(apr().files.get(&id).map(|f| f.size), Some(4));
    }

    /// APR path resolution: literal prefix concatenation, stable unique ids
    /// (also for paths whose 31-bit FNV-1a hashes collide), ForEach
    /// results, cached sizes and PATH_MAX limits.
    #[test]
    fn apr_paths() {
        let root = temp_root("apr-paths");
        fs::write(root.join("apr.dat"), b"APR").unwrap();
        let _app0 = APP0_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        set_app0_root(&root);

        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C62_0000_0000;
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let ctx = CpuContext::new(&vm);
        // Guest layout: strings from BASE, path list, ids, sizes, results, error index.
        let (list, ids, sizes, results, error_index) = (BASE + 0x8000, BASE + 0x9000, BASE + 0x9100, BASE + 0x9200, BASE + 0x9300);
        let mut next = BASE;
        let mut put = |ctx: &CpuContext, bytes: &[u8]| {
            let at = next;
            ctx.vm.write(at, bytes).unwrap();
            next += (bytes.len() as u64 + 16) & !15;
            at
        };
        let cstr = |s: &str| {
            let mut b = s.as_bytes().to_vec();
            b.push(0);
            b
        };
        let set_list = |ctx: &CpuContext, ptrs: &[u64]| {
            for (i, p) in ptrs.iter().enumerate() {
                ctx.write_u64(list + i as u64 * 8, *p);
            }
        };
        let resolve = |ctx: &CpuContext, prefix: u64, count: u64, ids: u64, sizes: u64| {
            apr_resolve_core(ctx, &apr_args(prefix, list, count, ids, sizes, error_index, 0))
        };
        let resolve_each = |ctx: &CpuContext, prefix: u64, count: u64| {
            apr_resolve_core(ctx, &apr_args(prefix, list, count, ids, sizes, 0, results))
        };

        let mut expected = None;
        for (prefix, path) in [("", "/app0/apr.dat"), ("/app0/", "apr.dat"), ("/", "app0/apr.dat"), ("/app", "0/apr.dat")] {
            let (p, q) = (put(&ctx, &cstr(prefix)), put(&ctx, &cstr(path)));
            set_list(&ctx, &[q]);
            ctx.write_u32(ids, APR_INVALID_ID);
            ctx.write_u32(error_index, APR_INVALID_ID);
            assert_eq!(resolve(&ctx, p, 1, ids, sizes), Ok(0), "prefix '{prefix}' path '{path}'");
            let id = ctx.read_u32(ids).unwrap();
            assert_ne!(id, APR_INVALID_ID);
            assert_eq!(ctx.read_u64(sizes), Some(3));
            assert_eq!(*expected.get_or_insert(id), id, "equivalent APR paths return the same id");
        }
        let expected = expected.unwrap();
        // The exported entry point: success is 0.
        let mut c = CpuContext::new(&vm);
        let empty = put(&ctx, b"\0");
        let full = put(&ctx, &cstr("/app0/apr.dat"));
        set_list(&ctx, &[full]);
        for (r, v) in [
            (crate::context::CpuRegister::Rdi, empty),
            (crate::context::CpuRegister::Rsi, list),
            (crate::context::CpuRegister::Rdx, 1),
            (crate::context::CpuRegister::Rcx, ids),
            (crate::context::CpuRegister::R8, sizes),
            (crate::context::CpuRegister::R9, error_index),
        ] {
            c.set(r, v);
        }
        assert_eq!(cb_apr_resolve_with_prefix_ids_and_sizes(&mut c), 0);

        // ForEach reports the missing path and continues to the valid file.
        let (missing, valid) = (put(&ctx, &cstr("/app0/missing.dat")), put(&ctx, &cstr("/app0/apr.dat")));
        set_list(&ctx, &[missing, valid]);
        ctx.write_u64(sizes, 1);
        ctx.write_u64(sizes + 8, 1);
        assert_eq!(resolve_each(&ctx, empty, 2), Ok(1));
        assert_eq!(ctx.read_u32(results), Some(sce_error(ENOENT) as u32));
        assert_eq!(ctx.read_u32(results + 4), Some(0));
        assert_eq!(ctx.read_u32(ids), Some(APR_INVALID_ID));
        assert_eq!(ctx.read_u32(ids + 4), Some(expected));
        assert_eq!(ctx.read_u64(sizes), Some(0));
        assert_eq!(ctx.read_u64(sizes + 8), Some(3));

        // These paths share the old 31-bit FNV-1a id (0x122d1544).
        let collision = ["perf-audit/test_00015ddf.bin", "perf-audit/test_000389b8.bin"];
        for p in collision {
            assert_eq!(fnv1a32(format!("/app0/{p}").as_bytes()) & 0x7FFF_FFFF, 0x122d1544);
        }
        fs::create_dir(root.join("perf-audit")).unwrap();
        fs::write(root.join(collision[0]), b"APR").unwrap();
        fs::write(root.join(collision[1]), b"OTHER").unwrap();
        let app0 = put(&ctx, &cstr("/app0/"));
        let (c0, c1) = (put(&ctx, &cstr(collision[0])), put(&ctx, &cstr(collision[1])));
        set_list(&ctx, &[c0, c1]);
        assert_eq!(resolve(&ctx, app0, 2, ids, sizes), Ok(0));
        let collision_ids = [ctx.read_u32(ids).unwrap(), ctx.read_u32(ids + 4).unwrap()];
        assert_ne!(collision_ids[0], collision_ids[1]);
        assert!(!collision_ids.contains(&APR_INVALID_ID));
        let collision_sizes = [3u64, 5];
        assert_eq!(ctx.read_u64(sizes), Some(3));
        assert_eq!(ctx.read_u64(sizes + 8), Some(5));
        // Re-resolution preserves each file's id, host path and cached size.
        let stat = BASE + 0xA000;
        for (i, p) in [c0, c1].into_iter().enumerate() {
            set_list(&ctx, &[p]);
            assert_eq!(resolve(&ctx, app0, 1, ids, 0), Ok(0));
            assert_eq!(ctx.read_u32(ids), Some(collision_ids[i]));
            let mut c = CpuContext::new(&vm);
            c.set(crate::context::CpuRegister::Rdi, collision_ids[i] as u64);
            c.set(crate::context::CpuRegister::Rsi, stat);
            assert_eq!(cb_apr_get_file_stat(&mut c), 0);
            assert_eq!(ctx.read_u64(stat + 72), Some(collision_sizes[i]));
            c.set(crate::context::CpuRegister::Rsi, stat + 0x100);
            assert_eq!(cb_apr_get_file_size(&mut c), 0);
            assert_eq!(ctx.read_u64(stat + 0x100), Some(collision_sizes[i]));
            assert_eq!(apr_host_path(collision_ids[i]), Some(root.join(collision[i])));
        }

        // PATH_MAX includes the terminator; components stay below NAME_MAX.
        let mut longest = String::from("/app0/");
        for _ in 0..3 {
            longest.push_str(&"a".repeat(254));
            longest.push('/');
        }
        longest.push_str(&"b".repeat(1023 - longest.len()));
        let long = put(&ctx, &cstr(&longest));
        set_list(&ctx, &[long]);
        ctx.write_u32(error_index, APR_INVALID_ID);
        assert_eq!(resolve(&ctx, empty, 1, ids, sizes), Err(ENOENT), "a 1023-byte path fits PATH_MAX");
        assert_eq!(ctx.read_u32(error_index), Some(0));
        let slash = put(&ctx, b"/\0");
        assert_eq!(resolve(&ctx, slash, 1, ids, sizes), Err(ENAMETOOLONG), "prefix + path exceed PATH_MAX");
        let unterminated = put(&ctx, &[b'/'; 1024]);
        put(&ctx, b"/");
        set_list(&ctx, &[unterminated]);
        assert_eq!(resolve(&ctx, empty, 1, ids, sizes), Err(ENAMETOOLONG), "unterminated path");
        let short = put(&ctx, &cstr("apr.dat"));
        set_list(&ctx, &[short]);
        assert_eq!(resolve(&ctx, unterminated, 1, ids, sizes), Err(ENAMETOOLONG), "unterminated prefix");

        // Argument checks.
        assert_eq!(resolve(&ctx, 0, 0, ids, sizes), Err(EINVAL));
        assert_eq!(resolve(&ctx, 0, APR_MAX_PATHS + 1, ids, sizes), Err(EINVAL));
        assert_eq!(resolve(&ctx, 0, 1, 0, 0), Err(EINVAL));
        let mut c = CpuContext::new(&vm);
        c.set(crate::context::CpuRegister::Rdi, 0x7FFF_FFFF);
        c.set(crate::context::CpuRegister::Rsi, stat);
        assert_eq!(cb_apr_get_file_size(&mut c), -1);
        c.set(crate::context::CpuRegister::Rsi, 0);
        assert_eq!(cb_apr_get_file_size(&mut c), -1);
        let _ = vm.unmap(BASE, 0x10000);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn title_id_parsing() {
        assert_eq!(parse_title_id(r#"{ "a": 1, "titleId" : "PPSA10112", "x": 2 }"#), Some("PPSA10112".into()));
        assert_eq!(sanitize_title_id(" ppsa-1 2 "), "PPSA-1_2");
        assert_eq!(sanitize_title_id(""), "UNKNOWN");
    }

    #[test]
    fn host_errno_mapping() {
        assert_eq!(host_errno(&io::Error::from_raw_os_error(libc::ENOTEMPTY)), ENOTEMPTY);
        assert_eq!(host_errno(&io::Error::from_raw_os_error(libc::ENOENT)), ENOENT);
        assert_eq!(sce_error(ENOENT), 0x8002_0002u32 as i32 as i64);
    }

    /// Drive the exports through real guest memory: open/read/seek/fstat,
    /// getdents, read-only /app0, writable /temp0, AIO and APR resolve.
    #[test]
    fn random_devices() {
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A71_0000_0000;
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let buf = BASE + 0x1000;
        for dev in ["/dev/urandom\0", "/dev/random\0"] {
            ctx.vm.write(BASE, dev.as_bytes()).unwrap();
            ctx.set(Rdi, BASE);
            ctx.set(Rsi, 0);
            let fd = cb_sce_open(&mut ctx);
            assert!(fd > 2, "open {dev} returned {fd:#x}");
            ctx.set(Rdi, fd as u64);
            ctx.set(Rsi, buf);
            ctx.set(Rdx, 256);
            assert_eq!(cb_sce_read(&mut ctx), 256);
            assert_ne!(ctx.read_bytes(buf, 256).unwrap(), vec![0u8; 256]);
            // Positioned and vector reads ignore the offset and fill the buffers.
            assert_eq!(pread_core(&ctx, fd as i32, buf + 0x200, 16, 1000), Ok(16));
            ctx.write_u64(buf + 0x300, buf + 0x400);
            ctx.write_u64(buf + 0x308, 16);
            ctx.write_u64(buf + 0x310, buf + 0x410);
            ctx.write_u64(buf + 0x318, 16);
            assert_eq!(preadv_core(&ctx, fd as i32, buf + 0x300, 2, 1000), Ok(32));
            assert_ne!(ctx.read_bytes(buf + 0x400, 16), ctx.read_bytes(buf + 0x410, 16));
            ctx.set(Rdi, fd as u64);
            ctx.set(Rdx, 0);
            assert_eq!(cb_sce_lseek(&mut ctx), sce_error(ESPIPE));
            ctx.set(Rsi, buf);
            assert_eq!(cb_sce_fstat(&mut ctx), 0);
            // A character device: S_IFCHR | 0644, empty, 512-byte blocks.
            let mode = u16::from_le_bytes(ctx.read_bytes(buf + 8, 2).unwrap().try_into().unwrap());
            assert_eq!(mode & 0o170000, 0o020000);
            assert_eq!(mode, 0o020644);
            assert_eq!(ctx.read_u64(buf + 72), Some(0));
            assert_eq!(ctx.read_u64(buf + 80), Some(0));
            assert_eq!(ctx.read_u32(buf + 88), Some(512));
            assert_eq!(cb_sce_close(&mut ctx), 0);
            ctx.set(Rdi, BASE);
            ctx.set(Rsi, buf);
            assert_eq!(cb_sce_stat(&mut ctx), 0);
            assert_eq!(ctx.read_bytes(buf + 8, 2).unwrap(), STAT_MODE_CHARDEV.to_le_bytes());
        }
        let _ = vm.unmap(BASE, 0x10000);
    }

    /// F_GETFD/F_SETFD keep a per-descriptor close-on-exec flag, O_CLOEXEC
    /// sets it at open, unknown commands fail with EINVAL and stale or
    /// invalid descriptors with EBADF.
    #[test]
    fn file_descriptor_flags() {
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C61_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let fcntl = |ctx: &mut CpuContext, fd: i64, cmd: i64, arg: u64| {
            ctx.set(Rdi, fd as u64);
            ctx.set(Rsi, cmd as u64);
            ctx.set(Rdx, arg);
            cb_posix_fcntl(ctx)
        };
        assert_eq!(fcntl(&mut ctx, i32::MIN as i64, F_GETFD as i64, 0), -1);
        assert_eq!(fcntl_core(i32::MIN, F_GETFD, 0), Err(EBADF));
        for dev in ["/dev/urandom\0", "/dev/random\0"] {
            ctx.vm.write(BASE, dev.as_bytes()).unwrap();
            ctx.set(Rdi, BASE);
            ctx.set(Rsi, 0);
            let fd = cb_sce_open(&mut ctx);
            assert!(fd >= 3, "open {dev} returned {fd:#x}");
            assert_eq!(fcntl(&mut ctx, fd, 1, 0), 0);
            assert_eq!(fcntl(&mut ctx, fd, 2, 1), 0);
            assert_eq!(fcntl(&mut ctx, fd, 1, 0), 1);
            assert_eq!(fcntl(&mut ctx, fd, 2, 0), 0);
            assert_eq!(fcntl(&mut ctx, fd, 1, 0), 0);
            assert_eq!(fcntl(&mut ctx, fd, -1, 0), -1);
            assert_eq!(fcntl_core(fd as i32, u32::MAX as u64, 0), Err(EINVAL));
            ctx.set(Rdi, fd as u64);
            assert_eq!(cb_sce_close(&mut ctx), 0);
            assert_eq!(fcntl(&mut ctx, fd, 1, 0), -1);
            assert_eq!(fcntl_core(fd as i32, F_GETFD, 0), Err(EBADF));
            ctx.set(Rdi, BASE);
            ctx.set(Rsi, O_CLOEXEC);
            let cloexec_fd = cb_sce_open(&mut ctx);
            assert!(cloexec_fd >= 3);
            assert_eq!(fcntl(&mut ctx, cloexec_fd, 1, 0), 1);
            // A duplicate starts without the flag.
            let dup = fcntl(&mut ctx, cloexec_fd, F_DUPFD as i64, 0);
            assert!(dup >= 3);
            assert_eq!(fcntl(&mut ctx, dup, 1, 0), 0);
            for fd in [dup, cloexec_fd] {
                ctx.set(Rdi, fd as u64);
                assert_eq!(cb_sce_close(&mut ctx), 0);
            }
        }
        let _ = vm.unmap(BASE, 0x1000);
    }

    #[test]
    fn end_to_end_exports() {
        let root = temp_root("e2e");
        let game = root.join("game");
        fs::create_dir_all(game.join("sce_sys")).unwrap();
        fs::create_dir_all(game.join("data")).unwrap();
        fs::write(game.join("sce_sys/param.json"), r#"{"titleId": "TEST12345"}"#).unwrap();
        fs::write(game.join("data/a.txt"), b"hello world").unwrap();
        let _app0 = APP0_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::set_var("RPS5E_DATA_DIR", root.join("user"));
        set_app0_root(&game);

        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A5A_0000_0000;
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let path = |ctx: &CpuContext, s: &str| {
            let mut b = s.as_bytes().to_vec();
            b.push(0);
            ctx.vm.write(BASE, &b).unwrap();
            BASE
        };
        let buf = BASE + 0x1000;
        let set = |ctx: &mut CpuContext, args: &[u64]| {
            use crate::context::CpuRegister::*;
            for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
                ctx.set(*r, *v);
            }
        };

        // open + read + lseek + fstat
        let p = path(&ctx, "/app0/data/a.txt");
        set(&mut ctx, &[p, 0, 0]);
        let fd = cb_sce_open(&mut ctx);
        assert!(fd > 2, "open returned {fd:#x}");
        set(&mut ctx, &[fd as u64, buf, 5]);
        assert_eq!(cb_sce_read(&mut ctx), 5);
        assert_eq!(&ctx.read_bytes(buf, 5).unwrap(), b"hello");
        set(&mut ctx, &[fd as u64, 0, SEEK_END as u64]);
        assert_eq!(cb_sce_lseek(&mut ctx), 11);
        set(&mut ctx, &[fd as u64, buf]);
        assert_eq!(cb_sce_fstat(&mut ctx), 0);
        assert_eq!(i64::from_le_bytes(ctx.read_bytes(buf + 72, 8).unwrap().try_into().unwrap()), 11);
        set(&mut ctx, &[fd as u64, buf, 5, 6]);
        assert_eq!(cb_sce_pread(&mut ctx), 5);
        assert_eq!(&ctx.read_bytes(buf, 5).unwrap(), b"world");
        set(&mut ctx, &[fd as u64]);
        assert_eq!(cb_sce_close(&mut ctx), 0);
        assert_eq!(cb_sce_close(&mut ctx), sce_error(EBADF));

        // /app0 is read-only; unknown absolute paths are denied
        let p = path(&ctx, "/app0/data/new.txt");
        set(&mut ctx, &[p, O_WRONLY | O_CREAT, 0o644]);
        assert_eq!(cb_sce_open(&mut ctx), sce_error(EACCES));
        assert_eq!(cb_posix_open(&mut ctx), -1);
        let p = path(&ctx, "/etc/passwd");
        set(&mut ctx, &[p, 0]);
        assert_eq!(cb_sce_open(&mut ctx), sce_error(ENOENT));

        // stat + negative cache
        let p = path(&ctx, "/app0/data");
        set(&mut ctx, &[p, buf]);
        assert_eq!(cb_sce_stat(&mut ctx), 0);
        assert_eq!(u16::from_le_bytes(ctx.read_bytes(buf + 8, 2).unwrap().try_into().unwrap()), 0o040777);
        let p = path(&ctx, "/app0/missing");
        set(&mut ctx, &[p, buf]);
        assert_eq!(cb_sce_stat(&mut ctx), sce_error(ENOENT));
        assert_eq!(cb_sce_stat(&mut ctx), sce_error(ENOENT));

        // directory enumeration
        let p = path(&ctx, "/app0/data");
        set(&mut ctx, &[p, O_DIRECTORY]);
        let dfd = cb_sce_open(&mut ctx);
        assert!(dfd > 2);
        set(&mut ctx, &[dfd as u64, buf, 4096, buf + 0x800]);
        let n = cb_sce_getdirentries(&mut ctx);
        assert_eq!(n, 512);
        assert_eq!(&ctx.read_bytes(buf + 24 + 8, 5).unwrap(), b"a.txt");
        assert_eq!(cb_sce_getdirentries(&mut ctx), 0);

        // writable /temp0: create, write, rename, unlink, mkdir/rmdir
        let p = path(&ctx, "/temp0/dir/log.txt");
        set(&mut ctx, &[p, O_RDWR | O_CREAT | O_TRUNC]);
        let wfd = cb_sce_open(&mut ctx);
        assert!(wfd > 2, "temp0 open {wfd:#x}");
        ctx.vm.write(buf, b"abc").unwrap();
        set(&mut ctx, &[wfd as u64, buf, 3]);
        assert_eq!(cb_posix_write(&mut ctx), 3);
        let temp0 = root.join("user/TEST12345/temp0");
        assert_eq!(fs::read(temp0.join("dir/log.txt")).unwrap(), b"abc");

        // AIO read of the temp file
        let req = buf + 0x100;
        let res = buf + 0x200;
        let mut r = Vec::new();
        r.extend_from_slice(&1i64.to_le_bytes());
        r.extend_from_slice(&2u64.to_le_bytes());
        r.extend_from_slice(&(buf + 0x300).to_le_bytes());
        r.extend_from_slice(&res.to_le_bytes());
        r.extend_from_slice(&(wfd as i32).to_le_bytes());
        r.extend_from_slice(&[0u8; 4]);
        ctx.vm.write(req, &r).unwrap();
        set(&mut ctx, &[req, 1, 0, buf + 0x400]);
        assert_eq!(cb_aio_submit_read(&mut ctx), 0);
        assert_eq!(&ctx.read_bytes(buf + 0x300, 2).unwrap(), b"bc");
        assert_eq!(ctx.read_u64(res), Some(2));
        let id = ctx.read_u32(buf + 0x400).unwrap();
        set(&mut ctx, &[id as u64, buf + 0x410, 0]);
        assert_eq!(cb_aio_wait_request(&mut ctx), 0);
        assert_eq!(ctx.read_u32(buf + 0x410), Some(AIO_STATE_COMPLETED as u32));

        set(&mut ctx, &[wfd as u64]);
        assert_eq!(cb_posix_close(&mut ctx), 0);
        ctx.vm.write(BASE, b"/temp0/dir/log.txt\0").unwrap();
        ctx.vm.write(BASE + 0x40, b"/temp0/dir/moved.txt\0").unwrap();
        set(&mut ctx, &[BASE, BASE + 0x40]);
        assert_eq!(cb_sce_rename(&mut ctx), 0);
        assert!(temp0.join("dir/moved.txt").is_file());
        let p = path(&ctx, "/temp0/dir/moved.txt");
        set(&mut ctx, &[p]);
        assert_eq!(cb_sce_unlink(&mut ctx), 0);
        let p = path(&ctx, "/temp0/dir");
        set(&mut ctx, &[p]);
        assert_eq!(cb_sce_rmdir(&mut ctx), 0);
        assert_eq!(cb_sce_rmdir(&mut ctx), sce_error(ENOENT));
        assert_eq!(cb_sce_mkdir(&mut ctx), 0);
        assert_eq!(cb_sce_mkdir(&mut ctx), sce_error(EEXIST));

        // APR resolve: unique ids, sizes are file sizes
        ctx.vm.write(BASE, b"/app0/data/a.txt\0").unwrap();
        ctx.write_u64(BASE + 0x80, BASE);
        set(&mut ctx, &[BASE + 0x80, 1, buf, buf + 0x10, buf + 0x20]);
        assert_eq!(cb_apr_resolve_ids_and_sizes(&mut ctx), 0);
        let id = ctx.read_u32(buf).unwrap();
        assert_ne!(id, APR_INVALID_ID);
        assert_eq!(ctx.read_u64(buf + 0x10), Some(11));
        set(&mut ctx, &[id as u64, buf + 0x30]);
        assert_eq!(cb_apr_get_file_size(&mut ctx), 0);
        assert_eq!(ctx.read_u64(buf + 0x30), Some(11));

        // APR ForEach: per-path results, returns the resolved count
        ctx.vm.write(BASE + 0x40, b"/app0/data/none.bin\0").unwrap();
        ctx.write_u64(BASE + 0x88, BASE + 0x40);
        set(&mut ctx, &[BASE + 0x80, 2, buf, buf + 0x10, buf + 0x20]);
        assert_eq!(cb_apr_resolve_ids_and_sizes_for_each(&mut ctx), 1);
        assert_eq!(ctx.read_u32(buf), Some(id));
        assert_eq!(ctx.read_u32(buf + 4), Some(u32::MAX));
        assert_eq!(ctx.read_u64(buf + 0x18), Some(0));
        assert_eq!(ctx.read_u32(buf + 0x20), Some(0));
        assert_eq!(ctx.read_u32(buf + 0x24), Some(sce_error(ENOENT) as u32));
        // Without a result array the walk stops at the missing path.
        set(&mut ctx, &[BASE + 0x80, 2, buf, 0]);
        assert_eq!(cb_apr_resolve_ids_for_each(&mut ctx), -1);
        set(&mut ctx, &[BASE + 0x80, 2, 0, 0]);
        assert_eq!(cb_apr_resolve_ids_for_each(&mut ctx), -1);
        // prefix variant: the prefix is concatenated literally
        ctx.vm.write(BASE + 0x100, b"/app0/data/\0").unwrap();
        ctx.vm.write(BASE + 0x140, b"a.txt\0").unwrap();
        ctx.write_u64(BASE + 0x180, BASE + 0x140);
        set(&mut ctx, &[BASE + 0x100, BASE + 0x180, 1, buf, buf + 0x40]);
        assert_eq!(cb_apr_resolve_with_prefix_ids(&mut ctx), 0);
        assert_eq!(ctx.read_u32(buf), Some(id));
        ctx.vm.write(BASE + 0x100, b"/app0/data\0").unwrap();
        set(&mut ctx, &[BASE + 0x100, BASE + 0x180, 1, buf, buf + 0x40]);
        assert_eq!(cb_apr_resolve_with_prefix_ids(&mut ctx), -1);
        assert_eq!(ctx.read_u32(buf + 0x40), Some(0));
        // an empty path is invalid; a NULL entry is a fault
        ctx.vm.write(BASE + 0x140, b"\0").unwrap();
        set(&mut ctx, &[BASE + 0x180, 1, buf, buf + 0x40]);
        assert_eq!(cb_apr_resolve_ids(&mut ctx), -1);
        let args = |list| apr_args(0, list, 1, buf, 0, buf + 0x40, 0);
        assert_eq!(apr_resolve_core(&ctx, &args(BASE + 0x180)), Err(EINVAL));
        ctx.write_u64(BASE + 0x180, 0);
        set(&mut ctx, &[BASE + 0x180, 1, buf, buf + 0x40]);
        assert_eq!(cb_apr_resolve_ids(&mut ctx), -1);
        assert_eq!(apr_resolve_core(&ctx, &args(BASE + 0x180)), Err(EFAULT));

        // utimes applies host times; /app0 is read-only
        let p = path(&ctx, "/temp0/dir");
        ctx.write_u64(buf, 1_000_000);
        ctx.write_u64(buf + 8, 0);
        ctx.write_u64(buf + 16, 2_000_000);
        ctx.write_u64(buf + 24, 0);
        set(&mut ctx, &[p, buf]);
        assert_eq!(cb_sce_utimes(&mut ctx), 0);
        let mtime = fs::metadata(temp0.join("dir")).unwrap().modified().unwrap();
        assert_eq!(mtime.duration_since(UNIX_EPOCH).unwrap().as_secs(), 2_000_000);
        let p = path(&ctx, "/app0/data/a.txt");
        set(&mut ctx, &[p, 0]);
        assert_eq!(cb_sce_utimes(&mut ctx), sce_error(EROFS));
        let p = path(&ctx, "/app0/data/a.txt");
        set(&mut ctx, &[p, 0, 0]);
        let rfd = cb_sce_open(&mut ctx);
        set(&mut ctx, &[rfd as u64, 0o644]);
        assert_eq!(cb_sce_fchmod(&mut ctx), sce_error(EROFS));
        set(&mut ctx, &[9999, 0o644]);
        assert_eq!(cb_sce_fchmod(&mut ctx), sce_error(EBADF));
        // POSIX names report failures as -1 (errno), never as kernel codes.
        assert_eq!(cb_posix_fchmod(&mut ctx), -1);
        set(&mut ctx, &[rfd as u64, 0]);
        assert_eq!(cb_sce_futimes(&mut ctx), sce_error(EROFS));
        assert_eq!(cb_posix_futimes(&mut ctx), -1);
        let p = path(&ctx, "/app0/data/a.txt");
        set(&mut ctx, &[p, 0o644]);
        assert_eq!(cb_posix_chmod(&mut ctx), -1);
        set(&mut ctx, &[p, 0]);
        assert_eq!(cb_posix_utimes(&mut ctx), -1);
        set(&mut ctx, &[rfd as u64]);
        assert_eq!(cb_sce_close(&mut ctx), 0);

        // futimes on a writable file; chmod is accepted without touching the host mode.
        let p = path(&ctx, "/temp0/dir/f.txt");
        set(&mut ctx, &[p, (O_CREAT | O_RDWR) as u64, 0o666]);
        let wfd = cb_sce_open(&mut ctx);
        assert!(wfd > 2, "open returned {wfd:#x}");
        let host_mode = fs::metadata(temp0.join("dir/f.txt")).unwrap().permissions();
        set(&mut ctx, &[p, 0o400]);
        assert_eq!(cb_posix_chmod(&mut ctx), 0);
        assert_eq!(fs::metadata(temp0.join("dir/f.txt")).unwrap().permissions(), host_mode);
        set(&mut ctx, &[wfd as u64, 0o400]);
        assert_eq!(cb_posix_fchmod(&mut ctx), 0);
        ctx.write_u64(buf, 3_000_000);
        ctx.write_u64(buf + 8, 0);
        ctx.write_u64(buf + 16, 4_000_000);
        ctx.write_u64(buf + 24, 0);
        set(&mut ctx, &[wfd as u64, buf]);
        assert_eq!(cb_sce_futimes(&mut ctx), 0);
        let mtime = fs::metadata(temp0.join("dir/f.txt")).unwrap().modified().unwrap();
        assert_eq!(mtime.duration_since(UNIX_EPOCH).unwrap().as_secs(), 4_000_000);
        ctx.write_u64(buf + 8, 1_000_000);
        assert_eq!(cb_sce_futimes(&mut ctx), sce_error(EINVAL));
        set(&mut ctx, &[wfd as u64]);
        assert_eq!(cb_sce_close(&mut ctx), 0);
        set(&mut ctx, &[wfd as u64, 0]);
        assert_eq!(cb_sce_futimes(&mut ctx), sce_error(EBADF));

        let _ = vm.unmap(BASE, 0x10000);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn package_image_at_app0() {
        let _app0 = APP0_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let root = temp_root("pkgimg");
        std::env::set_var("RPS5E_DATA_DIR", root.join("user"));
        let mut b = ps5pkg::build::PfsBuilder::new()
            .file("eboot.bin", b"\x7FELF")
            .file("sce_sys/param.json", br#"{"titleId":"PPSA99999"}"#)
            .file("Data/A.txt", b"hello world")
            .file("Data/sub/b.bin", &[7u8; 5000])
            .compressed_file("Data/packed.txt", &b"pfsc ".repeat(3000));
        b.signed = true;
        let img = root.join("game.ffpkg");
        fs::write(&img, b.build()).unwrap();
        let pkg = ps5pkg::Package::open(&img).unwrap();
        let cache = mount_app0_image(pkg.fs(), "PPSA99999", "test-image-1").unwrap();
        assert!(cache.starts_with(root.join("user/PPSA99999")));
        // The loader's files are copied out up front; game data is not.
        assert_eq!(fs::read(cache.join("eboot.bin")).unwrap(), b"\x7FELF");
        assert!(cache.join("sce_sys/param.json").is_file());
        assert!(!cache.join("Data/A.txt").exists());
        // Binding /app0 to the cache (what the loader does) keeps the image.
        set_app0_root(&cache);
        assert!(in_app0_image("/app0/Data") && in_app0_image("Data/A.txt") && !in_app0_image("/temp0/x"));

        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A6B_0000_0000;
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let path = |ctx: &CpuContext, s: &str| {
            let mut v = s.as_bytes().to_vec();
            v.push(0);
            ctx.vm.write(BASE, &v).unwrap();
            BASE
        };
        let buf = BASE + 0x1000;
        let set = |ctx: &mut CpuContext, args: &[u64]| {
            use crate::context::CpuRegister::*;
            for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
                ctx.set(*r, *v);
            }
        };

        // open (case-insensitive) + read + lseek + fstat + pread
        let p = path(&ctx, "/app0/DATA/a.TXT");
        set(&mut ctx, &[p, 0, 0]);
        let fd = cb_sce_open(&mut ctx);
        assert!(fd > 2, "open returned {fd:#x}");
        set(&mut ctx, &[fd as u64, buf, 5]);
        assert_eq!(cb_sce_read(&mut ctx), 5);
        assert_eq!(&ctx.read_bytes(buf, 5).unwrap(), b"hello");
        assert_eq!(cb_sce_read(&mut ctx), 5);
        assert_eq!(&ctx.read_bytes(buf, 5).unwrap(), b" worl");
        set(&mut ctx, &[fd as u64, 0, SEEK_END as u64]);
        assert_eq!(cb_sce_lseek(&mut ctx), 11);
        set(&mut ctx, &[fd as u64, (-3i64) as u64, SEEK_CUR as u64]);
        assert_eq!(cb_sce_lseek(&mut ctx), 8);
        set(&mut ctx, &[fd as u64, buf, 100]);
        assert_eq!(cb_sce_read(&mut ctx), 3);
        set(&mut ctx, &[fd as u64, buf]);
        assert_eq!(cb_sce_fstat(&mut ctx), 0);
        assert_eq!(i64::from_le_bytes(ctx.read_bytes(buf + 72, 8).unwrap().try_into().unwrap()), 11);
        set(&mut ctx, &[fd as u64, buf, 5, 6]);
        assert_eq!(cb_sce_pread(&mut ctx), 5);
        assert_eq!(&ctx.read_bytes(buf, 5).unwrap(), b"world");
        set(&mut ctx, &[fd as u64, buf, 1]);
        assert_eq!(cb_sce_write(&mut ctx), sce_error(EBADF));
        set(&mut ctx, &[fd as u64]);
        assert_eq!(cb_sce_close(&mut ctx), 0);

        // Relative paths and the PFSC-compressed file.
        let p = path(&ctx, "Data/packed.txt");
        set(&mut ctx, &[p, buf]);
        assert_eq!(cb_sce_stat(&mut ctx), 0);
        assert_eq!(i64::from_le_bytes(ctx.read_bytes(buf + 72, 8).unwrap().try_into().unwrap()), 15000);
        set(&mut ctx, &[p, 0, 0]);
        let fd = cb_sce_open(&mut ctx);
        set(&mut ctx, &[fd as u64, buf, 10, 14990]);
        assert_eq!(cb_sce_pread(&mut ctx), 10);
        assert_eq!(&ctx.read_bytes(buf, 10).unwrap(), b"pfsc pfsc ");
        set(&mut ctx, &[fd as u64]);
        assert_eq!(cb_sce_close(&mut ctx), 0);

        // Everything in the image is read-only.
        let p = path(&ctx, "/app0/Data/new.txt");
        set(&mut ctx, &[p, O_WRONLY | O_CREAT, 0o644]);
        assert_eq!(cb_sce_open(&mut ctx), sce_error(EACCES));
        let p = path(&ctx, "/app0/Data/A.txt");
        set(&mut ctx, &[p, O_RDWR]);
        assert_eq!(cb_sce_open(&mut ctx), sce_error(EACCES));
        set(&mut ctx, &[p]);
        assert_eq!(cb_sce_unlink(&mut ctx), sce_error(EACCES));
        let p = path(&ctx, "/app0/NewDir");
        set(&mut ctx, &[p, 0o755]);
        assert_eq!(cb_sce_mkdir(&mut ctx), sce_error(EACCES));
        assert!(!cache.join("Data/A.txt").exists(), "permission checks copy nothing");

        // stat / reachability
        let p = path(&ctx, "/app0/data");
        set(&mut ctx, &[p, buf]);
        assert_eq!(cb_sce_stat(&mut ctx), 0);
        assert_eq!(u16::from_le_bytes(ctx.read_bytes(buf + 8, 2).unwrap().try_into().unwrap()), 0o040777);
        set(&mut ctx, &[p]);
        assert_eq!(cb_sce_check_reachability(&mut ctx), 0);
        let p = path(&ctx, "/app0/missing");
        set(&mut ctx, &[p, buf]);
        assert_eq!(cb_sce_stat(&mut ctx), sce_error(ENOENT));
        set(&mut ctx, &[p]);
        assert_eq!(cb_sce_check_reachability(&mut ctx), sce_error(ENOENT));

        // Directory enumeration: ".", "..", "A.txt", "packed.txt", "sub".
        let p = path(&ctx, "/app0/Data");
        set(&mut ctx, &[p, O_DIRECTORY]);
        let dfd = cb_sce_open(&mut ctx);
        assert!(dfd > 2);
        set(&mut ctx, &[dfd as u64, buf, 4096, buf + 0x800]);
        let n = cb_sce_getdirentries(&mut ctx) as usize;
        let raw = ctx.read_bytes(buf, n).unwrap();
        let mut names = Vec::new();
        let mut off = 0;
        while off < raw.len() {
            let reclen = u16::from_le_bytes([raw[off + 4], raw[off + 5]]) as usize;
            let (kind, nl) = (raw[off + 6], raw[off + 7] as usize);
            names.push((String::from_utf8_lossy(&raw[off + 8..off + 8 + nl]).into_owned(), kind));
            off += reclen;
        }
        let want: Vec<(String, u8)> = [(".", DT_DIR), ("..", DT_DIR), ("A.txt", DT_REG), ("packed.txt", DT_REG), ("sub", DT_DIR)]
            .iter()
            .map(|(n, k)| (n.to_string(), *k))
            .collect();
        assert_eq!(names, want);
        set(&mut ctx, &[dfd as u64, buf]);
        assert_eq!(cb_sce_fstat(&mut ctx), 0);
        set(&mut ctx, &[dfd as u64]);
        assert_eq!(cb_sce_close(&mut ctx), 0);

        // APR ids read straight from the image.
        ctx.vm.write(BASE, b"/app0/Data/sub/b.bin\0").unwrap();
        ctx.write_u64(BASE + 0x80, BASE);
        set(&mut ctx, &[BASE + 0x80, 1, buf, buf + 0x10, buf + 0x20]);
        assert_eq!(cb_apr_resolve_ids_and_sizes(&mut ctx), 0);
        let id = ctx.read_u32(buf).unwrap();
        assert_eq!(ctx.read_u64(buf + 0x10), Some(5000));
        assert!(apr_host_path(id).is_none());
        assert_eq!(apr_image_file(id).unwrap().read_all().unwrap(), vec![7u8; 5000]);
        set(&mut ctx, &[id as u64, buf + 0x40]);
        assert_eq!(cb_apr_get_file_stat(&mut ctx), 0);

        // Readers outside the descriptor table.
        let mut s = String::new();
        let f = open_guest_file("/app0/data/a.txt").unwrap().unwrap();
        let mut b11 = [0u8; 11];
        assert_eq!(f.read_at(&mut b11, 0).unwrap(), 11);
        s.push_str(std::str::from_utf8(&b11).unwrap());
        assert_eq!(s, "hello world");
        assert!(open_guest_file("/app0/nope").is_none());
        // Host-path consumers get a copy of just the file they ask for.
        let copied = resolve_guest_path("/app0/data/a.txt").unwrap();
        assert_eq!(copied, cache.join("Data/A.txt"));
        assert_eq!(fs::read(&copied).unwrap(), b"hello world");
        assert!(!resolve_guest_path("/app0/nope").unwrap().exists());
        // The writable roots still work.
        assert!(resolve_guest_path("/temp0").unwrap().starts_with(root.join("user/PPSA99999")));

        unmount_app0_image();
        let _ = vm.unmap(BASE, 0x10000);
        let _ = fs::remove_dir_all(&root);
    }
}
