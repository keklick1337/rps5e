// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libc stdio `FILE` family.
//!
//! `fopen` hands the guest a pointer to a zeroed 0x100-byte object in guest
//! memory (so native helpers that peek into a `FILE`, such as the lock
//! routines, find readable memory); the stream state itself lives on the
//! host, keyed by that address. Streams are backed by host files reached
//! through the guest path translation of the file system module, with a
//! 4 KiB read/write buffer whose flushing follows the `setvbuf` mode.
//!
//! `FILE` pointers that were not produced here (the shipped libc's
//! `_Stdin`/`_Stdout`/`_Stderr` objects) are treated as the console: writes
//! go to the host log as `[stdout]` / `[stderr]` lines, reads fail.
//!
//! Pointer-returning calls return NULL and set errno; int-returning calls
//! return -1 (EOF) on failure.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use crate::context::CpuContext;
use crate::filesystem::errno::*;
use crate::registry::{HleCallback, HleRegistry, HleResult};
use std::collections::HashMap;
use crate::filesystem::FileSource;
use std::fs::OpenOptions;
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

const MAX_PATH_LENGTH: usize = 4096;
const MAX_MODE_LENGTH: usize = 16;
const MAX_TEXT_LENGTH: usize = 1 << 20;
const IO_CHUNK: usize = 1 << 20;

/// Size of the guest-visible `FILE` object.
pub const GUEST_FILE_OBJECT_SIZE: u64 = 0x100;
/// SDK `fpos_t`: an 8-byte offset followed by a 16-byte `_Mbstatet`.
pub const SDK_FPOS_SIZE: usize = 24;

/// `setvbuf` modes.
pub const IOFBF: i32 = 0;
pub const IOLBF: i32 = 1;
pub const IONBF: i32 = 2;

const SEEK_SET: i32 = 0;
const SEEK_CUR: i32 = 1;
const SEEK_END: i32 = 2;

/// Host-side stream buffer size.
const STREAM_BUFFER: usize = 4096;

const EOF: HleResult = -1;

// ---------------------------------------------------------------------------
// Mode parsing
// ---------------------------------------------------------------------------

/// How an `fopen` mode string opens the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenMode {
    pub read: bool,
    pub write: bool,
    /// Create the file when missing.
    pub create: bool,
    /// Truncate an existing file.
    pub truncate: bool,
    /// Every write goes to the end of the file.
    pub append: bool,
    /// 'x': fail with EEXIST when the file already exists.
    pub exclusive: bool,
}

/// Parse an `fopen` mode as FreeBSD's `__sflags` does: the access
/// character r/w/a, then any of 'b', '+', 'x', 'e' and 'v' up to the first
/// other character, which ends the mode without failing ("rt" reads, and
/// "rt+" still only reads). An unknown access character, or 'x' on a
/// read-only stream, is invalid.
pub fn parse_mode(mode: &str) -> Option<OpenMode> {
    let bytes = mode.as_bytes();
    let flags: Vec<u8> = bytes.get(1..)?.iter().copied().take_while(|c| b"b+xev".contains(c)).collect();
    let plus = flags.contains(&b'+');
    let exclusive = flags.contains(&b'x');
    let m = match bytes.first()? {
        b'r' if exclusive && !plus => return None,
        b'r' => OpenMode { read: true, write: plus, create: false, truncate: false, append: false, exclusive },
        b'w' => OpenMode { read: plus, write: true, create: true, truncate: true, append: false, exclusive },
        b'a' => OpenMode { read: plus, write: true, create: true, truncate: false, append: true, exclusive },
        _ => return None,
    };
    Some(m)
}

// ---------------------------------------------------------------------------
// Buffered host stream
// ---------------------------------------------------------------------------

struct Stream {
    file: FileSource,
    mode: OpenMode,
    /// Logical position seen by the guest (includes pending writes).
    pos: u64,
    /// Read-ahead cache holding the file bytes at `rbuf_off..`.
    rbuf: Vec<u8>,
    rbuf_off: u64,
    /// Pending writes starting at `wstart`.
    wbuf: Vec<u8>,
    wstart: u64,
    /// The read cache holds a pushed-back byte that differs from the file.
    rbuf_dirty: bool,
}

fn not_permitted() -> io::Error {
    io::Error::from_raw_os_error(libc::EBADF)
}

impl Stream {
    fn new(file: FileSource, mode: OpenMode) -> io::Result<Self> {
        let pos = if mode.append { file.len()? } else { 0 };
        Ok(Self { file, mode, pos, rbuf: Vec::new(), rbuf_off: 0, wbuf: Vec::new(), wstart: 0, rbuf_dirty: false })
    }

    fn flush_writes(&mut self) -> io::Result<()> {
        if self.wbuf.is_empty() {
            return Ok(());
        }
        let data = std::mem::take(&mut self.wbuf);
        if self.mode.append {
            // O_APPEND: the kernel places the bytes at the current end.
            self.file.write_all(&data)?;
            self.pos = self.file.len()?;
        } else {
            self.file.write_all_at(&data, self.wstart)?;
        }
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_writes()
    }

    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if !self.mode.read {
            return Err(not_permitted());
        }
        self.flush_writes()?;
        let mut done = 0usize;
        while done < out.len() {
            let cached_end = self.rbuf_off + self.rbuf.len() as u64;
            if self.pos >= self.rbuf_off && self.pos < cached_end {
                let from = (self.pos - self.rbuf_off) as usize;
                let n = (self.rbuf.len() - from).min(out.len() - done);
                out[done..done + n].copy_from_slice(&self.rbuf[from..from + n]);
                done += n;
                self.pos += n as u64;
                continue;
            }
            let want = out.len() - done;
            if want >= STREAM_BUFFER {
                let n = self.file.read_at(&mut out[done..], self.pos)?;
                if n == 0 {
                    break;
                }
                done += n;
                self.pos += n as u64;
                continue;
            }
            self.rbuf.resize(STREAM_BUFFER, 0);
            let n = self.file.read_at(&mut self.rbuf, self.pos)?;
            self.rbuf.truncate(n);
            self.rbuf_off = self.pos;
            if n == 0 {
                break;
            }
        }
        Ok(done)
    }

    fn read_byte(&mut self) -> io::Result<Option<u8>> {
        let mut b = [0u8; 1];
        Ok((self.read(&mut b)? == 1).then_some(b[0]))
    }

    fn write(&mut self, data: &[u8]) -> io::Result<()> {
        if !self.mode.write {
            return Err(not_permitted());
        }
        if data.is_empty() {
            return Ok(());
        }
        self.rbuf.clear();
        self.rbuf_dirty = false;
        if self.wbuf.is_empty() {
            self.wstart = self.pos;
        }
        self.wbuf.extend_from_slice(data);
        self.pos += data.len() as u64;
        if self.wbuf.len() >= STREAM_BUFFER {
            self.flush_writes()?;
        }
        Ok(())
    }

    fn seek(&mut self, offset: i64, whence: i32) -> io::Result<u64> {
        self.flush_writes()?;
        if self.rbuf_dirty {
            // Repositioning discards pushed-back bytes.
            self.rbuf.clear();
            self.rbuf_dirty = false;
        }
        let base = match whence {
            SEEK_SET => 0i128,
            SEEK_CUR => self.pos as i128,
            SEEK_END => self.file.len()? as i128,
            _ => return Err(io::Error::from_raw_os_error(libc::EINVAL)),
        };
        let target = base + offset as i128;
        if !(0..=i64::MAX as i128).contains(&target) {
            return Err(io::Error::from_raw_os_error(libc::EINVAL));
        }
        self.pos = target as u64;
        Ok(self.pos)
    }

    /// Push `b` back so the next read returns it (ungetc). The file itself
    /// is unchanged; the byte lives in the read cache just before `pos`.
    fn unread(&mut self, b: u8) -> io::Result<bool> {
        if !self.mode.read || self.pos == 0 {
            return Ok(false);
        }
        self.flush_writes()?;
        let p = self.pos - 1;
        let cached_end = self.rbuf_off + self.rbuf.len() as u64;
        if p >= self.rbuf_off && p < cached_end {
            let slot = &mut self.rbuf[(p - self.rbuf_off) as usize];
            if *slot != b {
                *slot = b;
                self.rbuf_dirty = true;
            }
        } else if self.pos == self.rbuf_off && !self.rbuf.is_empty() {
            self.rbuf.insert(0, b);
            self.rbuf_off = p;
            self.rbuf_dirty = true;
        } else {
            self.rbuf = vec![b];
            self.rbuf_off = p;
            self.rbuf_dirty = true;
        }
        self.pos = p;
        Ok(true)
    }

    fn len(&mut self) -> io::Result<u64> {
        self.flush_writes()?;
        Ok(self.file.len()?)
    }
}

struct StdioFile {
    stream: Stream,
    closed: bool,
    error: bool,
    buffering: i32,
}

impl StdioFile {
    /// Flush according to the buffering mode after `payload` was written.
    fn flush_for_mode(&mut self, payload: &[u8]) -> io::Result<()> {
        if self.buffering == IONBF || (self.buffering == IOLBF && payload.contains(&b'\n')) {
            self.stream.flush()?;
        }
        Ok(())
    }

    fn write_buffered(&mut self, payload: &[u8]) -> io::Result<()> {
        let r = self.stream.write(payload).and_then(|_| self.flush_for_mode(payload));
        if r.is_err() {
            self.error = true;
        }
        r
    }
}

type FileRef = Arc<Mutex<StdioFile>>;

static FILES: Mutex<Option<HashMap<u64, FileRef>>> = Mutex::new(None);

fn with_files<R>(f: impl FnOnce(&mut HashMap<u64, FileRef>) -> R) -> R {
    let mut g = FILES.lock().unwrap_or_else(|p| p.into_inner());
    f(g.get_or_insert_with(HashMap::new))
}

fn lookup(handle: u64) -> Option<FileRef> {
    if handle == 0 {
        return None;
    }
    with_files(|m| m.get(&handle).cloned())
}

fn lock(f: &FileRef) -> std::sync::MutexGuard<'_, StdioFile> {
    f.lock().unwrap_or_else(|p| p.into_inner())
}

// ---------------------------------------------------------------------------
// Console streams
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Console {
    Stdout,
    Stderr,
}

static CONSOLE_LINES: Mutex<(Vec<u8>, Vec<u8>)> = Mutex::new((Vec::new(), Vec::new()));
static STDERR_OBJECT: OnceLock<u64> = OnceLock::new();

/// Address of the shipped libc's `_Stderr` object, once modules are loaded.
fn stderr_object() -> Option<u64> {
    if let Some(&a) = STDERR_OBJECT.get() {
        return Some(a);
    }
    let a = crate::modules::resolve_symbol_any("_Stderr").filter(|&a| a != 0)?;
    Some(*STDERR_OBJECT.get_or_init(|| a))
}

/// Which console an unknown `FILE*` stands for. A NULL stream is stderr
/// (the C runtime reports its own failures there), the libc `_Stderr`
/// object is stderr, everything else is stdout.
fn console_for(handle: u64) -> Console {
    if handle == 0 || stderr_object() == Some(handle) {
        Console::Stderr
    } else {
        Console::Stdout
    }
}

fn emit_line(which: Console, line: &[u8]) {
    let text = String::from_utf8_lossy(line);
    let text = text.trim_end_matches('\r');
    match which {
        Console::Stdout => log::info!("[stdout] {}", text),
        Console::Stderr => log::info!("[stderr] {}", text),
    }
}

/// Append console output; complete lines are logged immediately.
fn console_write(which: Console, data: &[u8]) {
    let mut g = CONSOLE_LINES.lock().unwrap_or_else(|p| p.into_inner());
    let buf = match which {
        Console::Stdout => &mut g.0,
        Console::Stderr => &mut g.1,
    };
    buf.extend_from_slice(data);
    while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
        let line: Vec<u8> = buf.drain(..=nl).collect();
        emit_line(which, &line[..line.len() - 1]);
    }
    if buf.len() >= MAX_TEXT_LENGTH {
        let line = std::mem::take(buf);
        emit_line(which, &line);
    }
}

/// Log any partial console line.
fn console_flush(which: Console) {
    let mut g = CONSOLE_LINES.lock().unwrap_or_else(|p| p.into_inner());
    let buf = match which {
        Console::Stdout => &mut g.0,
        Console::Stderr => &mut g.1,
    };
    if !buf.is_empty() {
        let line = std::mem::take(buf);
        emit_line(which, &line);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read a NUL-terminated string of at most `max` bytes (terminator not
/// included) page by page, so the read never crosses into an unmapped page
/// past the terminator.
fn read_cstr(ctx: &CpuContext, addr: u64, max: usize) -> Option<Vec<u8>> {
    if addr == 0 {
        return None;
    }
    let mut out = Vec::new();
    let mut cur = addr;
    loop {
        let page_left = (0x1000 - (cur & 0xFFF)) as usize;
        let n = page_left.min(max + 1 - out.len());
        let mut buf = vec![0u8; n];
        ctx.vm.read(cur, &mut buf).ok()?;
        if let Some(p) = buf.iter().position(|&b| b == 0) {
            out.extend_from_slice(&buf[..p]);
            return Some(out);
        }
        out.extend_from_slice(&buf);
        if out.len() > max {
            return None;
        }
        cur += n as u64;
    }
}

fn read_string(ctx: &CpuContext, addr: u64, max: usize) -> Option<String> {
    read_cstr(ctx, addr, max).map(|b| String::from_utf8_lossy(&b).into_owned())
}

/// NULL return with errno.
fn null_with_errno(ctx: &CpuContext, errno: i32) -> HleResult {
    crate::kernel::set_errno(ctx, errno);
    0
}

/// -1 return with errno.
fn eof_with_errno(ctx: &CpuContext, errno: i32) -> HleResult {
    crate::kernel::set_errno(ctx, errno);
    EOF
}

/// errno for a failed host open.
/// The guest errno of a failed open, as open(2) reports it (ENOTDIR,
/// EISDIR, ELOOP and ENAMETOOLONG kept).
fn open_errno(e: &io::Error) -> i32 {
    crate::filesystem::io_errno(e)
}

fn open_host(host: &Path, mode: OpenMode) -> io::Result<Stream> {
    if mode.write {
        if let Some(parent) = host.parent() {
            if !parent.as_os_str().is_empty() {
                // A regular file on the way is ENOTDIR, as for open(2).
                if parent.ancestors().any(|a| a.symlink_metadata().is_ok_and(|m| m.is_file())) {
                    return Err(io::Error::from_raw_os_error(libc::ENOTDIR));
                }
                std::fs::create_dir_all(parent)?;
            }
        }
    }
    if host.is_dir() {
        // A directory opened for writing is EISDIR (FreeBSD open(2)).
        return Err(io::Error::from_raw_os_error(if mode.write { libc::EISDIR } else { libc::EACCES }));
    }
    let mut options = OpenOptions::new();
    options.read(mode.read).write(mode.write && !mode.append).append(mode.append);
    if mode.create && mode.exclusive {
        options.create_new(true);
    } else {
        options.create(mode.create).truncate(mode.truncate);
    }
    let file = options.open(host)?;
    Stream::new(FileSource::Host(file), mode)
}

/// Shared path/mode validation of fopen and freopen. Returns the opened
/// stream or the errno to report.
fn open_guest(ctx: &CpuContext, path_addr: u64, mode_addr: u64) -> Result<(String, String, Stream), i32> {
    let (Some(guest), Some(mode)) = (
        read_string(ctx, path_addr, MAX_PATH_LENGTH),
        read_string(ctx, mode_addr, MAX_MODE_LENGTH),
    ) else {
        return Err(EFAULT);
    };
    if guest.is_empty() {
        return Err(ENOENT);
    }
    let Some(parsed) = parse_mode(&mode) else {
        return Err(EINVAL);
    };
    // The random devices read from the host's CSPRNG.
    if crate::filesystem::is_random_device(&guest) {
        if parsed.write {
            return Err(EACCES);
        }
        return match open_host(Path::new("/dev/urandom"), parsed) {
            Ok(s) => Ok((guest, mode, s)),
            Err(e) => Err(open_errno(&e)),
        };
    }
    // Inside a package image mounted at /app0: read straight from it.
    if crate::filesystem::in_app0_image(&guest) {
        if parsed.write {
            log::debug!("[stdio] fopen('{guest}', '{mode}') -> EROFS (read-only image)");
            return Err(EROFS);
        }
        let opened = crate::filesystem::open_guest_file(&guest).ok_or(ENOENT)?;
        return match opened.and_then(|f| Stream::new(f, parsed)) {
            Ok(s) => Ok((guest, mode, s)),
            Err(e) => Err(open_errno(&e)),
        };
    }
    let Some(host) = crate::filesystem::resolve_guest_path(&guest) else {
        return Err(ENOENT);
    };
    if parsed.write && crate::filesystem::is_read_only(&guest, Some(&host)) {
        log::debug!("[stdio] fopen('{guest}', '{mode}') -> EROFS (read-only mount)");
        return Err(EROFS);
    }
    match open_host(&host, parsed) {
        Ok(s) => Ok((guest, mode, s)),
        Err(e) => {
            log::debug!("[stdio] fopen('{guest}', '{mode}') host={} failed: {e}", host.display());
            Err(open_errno(&e))
        }
    }
}

// ---------------------------------------------------------------------------
// Open / close
// ---------------------------------------------------------------------------

/// Open a guest path as a new FILE; returns the handle or the errno.
fn fopen_core(ctx: &CpuContext, path_addr: u64, mode_addr: u64) -> Result<u64, i32> {
    if path_addr == 0 || mode_addr == 0 {
        return Err(EFAULT);
    }
    let (guest, mode, stream) = match open_guest(ctx, path_addr, mode_addr) {
        Ok(v) => v,
        Err(e) => {
            if e != EFAULT {
                let g = read_string(ctx, path_addr, MAX_PATH_LENGTH).unwrap_or_default();
                log::info!("[stdio] fopen('{g}') failed errno={e}");
            }
            return Err(e);
        }
    };
    let Some(handle) = crate::pthread::alloc_guest_block(ctx, GUEST_FILE_OBJECT_SIZE) else {
        return Err(ENOMEM);
    };
    let file = StdioFile { stream, closed: false, error: false, buffering: IOFBF };
    with_files(|m| m.insert(handle, Arc::new(Mutex::new(file))));
    log::info!("[stdio] fopen('{guest}', '{mode}') -> {handle:#x}");
    Ok(handle)
}

/// fopen(path, mode) -> FILE*
fn cb_fopen(ctx: &mut CpuContext) -> HleResult {
    match fopen_core(ctx, ctx.rdi(), ctx.rsi()) {
        Ok(h) => h as i64,
        Err(e) => null_with_errno(ctx, e),
    }
}

/// fopen_s(FILE** result, path, mode) -> 0 / errno
fn cb_fopen_s(ctx: &mut CpuContext) -> HleResult {
    let (out, path_addr, mode_addr) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if out == 0 || path_addr == 0 || mode_addr == 0 {
        if out != 0 {
            ctx.write_u64(out, 0);
        }
        return EINVAL as HleResult;
    }
    match fopen_core(ctx, path_addr, mode_addr) {
        Ok(h) => {
            if ctx.write_u64(out, h) { 0 } else { EFAULT as HleResult }
        }
        Err(e) => {
            ctx.write_u64(out, 0);
            e as HleResult
        }
    }
}

/// freopen(path, mode, stream) -> stream / NULL
fn cb_freopen(ctx: &mut CpuContext) -> HleResult {
    match freopen_core(ctx, ctx.rdi(), ctx.rsi(), ctx.rdx()) {
        Ok(handle) => handle as i64,
        Err(e) => null_with_errno(ctx, e),
    }
}

/// Reopen stream `handle` on `path_addr` with `mode_addr`: the stream, or
/// the errno freopen reports.
fn freopen_core(ctx: &CpuContext, path_addr: u64, mode_addr: u64, handle: u64) -> Result<u64, i32> {
    if path_addr == 0
        || mode_addr == 0
        || read_cstr(ctx, path_addr, MAX_PATH_LENGTH).is_none()
        || read_cstr(ctx, mode_addr, MAX_MODE_LENGTH).is_none()
    {
        return Err(EFAULT);
    }
    if handle == 0 {
        return Err(EBADF);
    }
    // Validate the arguments in the same order as fopen before touching
    // the stream.
    let guest = read_string(ctx, path_addr, MAX_PATH_LENGTH).unwrap_or_default();
    let mode = read_string(ctx, mode_addr, MAX_MODE_LENGTH).unwrap_or_default();
    if guest.is_empty() {
        return Err(ENOENT);
    }
    if parse_mode(&mode).is_none() {
        return Err(EINVAL);
    }
    let file = lookup(handle).ok_or(EBADF)?;
    let (_, _, stream) = open_guest(ctx, path_addr, mode_addr)?;
    let mut f = lock(&file);
    if f.closed {
        return Err(EBADF);
    }
    let _ = f.stream.flush();
    f.stream = stream;
    f.buffering = IOFBF;
    f.error = false;
    log::info!("[stdio] freopen('{guest}', '{mode}') -> {handle:#x}");
    Ok(handle)
}

/// freopen_s(FILE** result, path, mode, stream) -> errno (C11 K.3.5.2.2):
/// a null result, mode or stream is EINVAL, with `*result` cleared when
/// it exists; otherwise freopen's outcome, the stream or null in
/// `*result` and 0 or freopen's errno returned.
fn cb_freopen_s(ctx: &mut CpuContext) -> HleResult {
    let (out, path_addr, mode_addr, handle) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    if out == 0 {
        return EINVAL as HleResult;
    }
    ctx.write_u64(out, 0);
    if mode_addr == 0 || handle == 0 {
        return EINVAL as HleResult;
    }
    match freopen_core(ctx, path_addr, mode_addr, handle) {
        Ok(h) => {
            ctx.write_u64(out, h);
            0
        }
        Err(e) => {
            crate::kernel::set_errno(ctx, e);
            e as HleResult
        }
    }
}

/// fclose(stream) -> 0 / EOF
fn cb_fclose(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    let Some(file) = lookup(handle) else { return EOF };
    let mut f = lock(&file);
    if f.closed {
        return EOF;
    }
    let removed = with_files(|m| match m.get(&handle) {
        Some(cur) if Arc::ptr_eq(cur, &file) => m.remove(&handle).is_some(),
        _ => false,
    });
    if !removed {
        return EOF;
    }
    f.closed = true;
    match f.stream.flush() {
        Ok(()) => 0,
        Err(_) => EOF,
    }
}

// ---------------------------------------------------------------------------
// Positioning
// ---------------------------------------------------------------------------

/// fseek(stream, offset, whence) -> 0 / -1
fn cb_fseek(ctx: &mut CpuContext) -> HleResult {
    let (handle, offset, whence) = (ctx.rdi(), ctx.rsi() as i64, ctx.rdx() as i32);
    let Some(file) = lookup(handle) else { return EOF };
    if !(SEEK_SET..=SEEK_END).contains(&whence) {
        return EOF;
    }
    let mut f = lock(&file);
    if f.closed {
        return EOF;
    }
    match f.stream.seek(offset, whence) {
        Ok(_) => 0,
        Err(_) => EOF,
    }
}

/// ftell(stream) -> offset / -1
fn cb_ftell(ctx: &mut CpuContext) -> HleResult {
    let Some(file) = lookup(ctx.rdi()) else { return EOF };
    let f = lock(&file);
    if f.closed {
        return EOF;
    }
    f.stream.pos as i64
}

/// Encode an SDK `fpos_t`.
pub fn encode_fpos(offset: u64) -> [u8; SDK_FPOS_SIZE] {
    let mut b = [0u8; SDK_FPOS_SIZE];
    b[..8].copy_from_slice(&offset.to_le_bytes());
    b
}

/// Offset stored in an SDK `fpos_t`.
pub fn decode_fpos(b: &[u8; SDK_FPOS_SIZE]) -> i64 {
    i64::from_le_bytes(b[..8].try_into().unwrap())
}

/// fgetpos(stream, fpos_t*) -> 0 / -1
fn cb_fgetpos(ctx: &mut CpuContext) -> HleResult {
    let (handle, pos_addr) = (ctx.rdi(), ctx.rsi());
    let Some(file) = lookup(handle) else { return eof_with_errno(ctx, EBADF) };
    if pos_addr == 0 {
        return eof_with_errno(ctx, EFAULT);
    }
    let f = lock(&file);
    if f.closed {
        return eof_with_errno(ctx, EBADF);
    }
    if ctx.vm.write(pos_addr, &encode_fpos(f.stream.pos)).is_err() {
        return eof_with_errno(ctx, EFAULT);
    }
    0
}

/// fsetpos(stream, const fpos_t*) -> 0 / -1
fn cb_fsetpos(ctx: &mut CpuContext) -> HleResult {
    let (handle, pos_addr) = (ctx.rdi(), ctx.rsi());
    let Some(file) = lookup(handle) else { return eof_with_errno(ctx, EBADF) };
    let mut raw = [0u8; SDK_FPOS_SIZE];
    if pos_addr == 0 || ctx.vm.read(pos_addr, &mut raw).is_err() {
        return eof_with_errno(ctx, EFAULT);
    }
    let offset = decode_fpos(&raw);
    if offset < 0 {
        return eof_with_errno(ctx, EINVAL);
    }
    let mut f = lock(&file);
    if f.closed {
        return eof_with_errno(ctx, EBADF);
    }
    match f.stream.seek(offset, SEEK_SET) {
        Ok(_) => 0,
        Err(_) => eof_with_errno(ctx, EIO),
    }
}

/// rewind(stream): seek to the start (dropping pushed-back bytes) and clear
/// the error indicator whether or not the seek worked; errno changes only
/// when it failed. The end-of-file indicator follows the position.
fn cb_rewind(ctx: &mut CpuContext) -> HleResult {
    if let Some(file) = lookup(ctx.rdi()) {
        let mut f = lock(&file);
        if !f.closed {
            let r = f.stream.seek(0, SEEK_SET);
            f.error = false;
            if let Err(e) = r {
                drop(f);
                crate::kernel::set_errno(ctx, crate::filesystem::io_errno(&e));
            }
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Buffering / state
// ---------------------------------------------------------------------------

/// setvbuf(stream, buf, mode, size) -> 0 / -1
fn cb_setvbuf(ctx: &mut CpuContext) -> HleResult {
    let (handle, buf, mode, size) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as i32, ctx.rcx());
    let Some(file) = lookup(handle) else { return eof_with_errno(ctx, EBADF) };
    if !matches!(mode, IOFBF | IOLBF | IONBF) {
        return eof_with_errno(ctx, EINVAL);
    }
    if buf != 0 && size != 0 {
        if size - 1 > u64::MAX - buf {
            return eof_with_errno(ctx, EINVAL);
        }
        let mut probe = [0u8; 1];
        if ctx.vm.read(buf, &mut probe).is_err() || ctx.vm.read(buf + size - 1, &mut probe).is_err() {
            return eof_with_errno(ctx, EFAULT);
        }
    }
    let mut f = lock(&file);
    if f.closed {
        return eof_with_errno(ctx, EBADF);
    }
    if f.stream.flush().is_err() {
        return eof_with_errno(ctx, EIO);
    }
    f.buffering = mode;
    0
}

/// fflush(stream) -> 0 / EOF; fflush(NULL) flushes every stream.
fn cb_fflush(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    if handle == 0 {
        let all: Vec<FileRef> = with_files(|m| m.values().cloned().collect());
        let mut failed = false;
        for file in all {
            let mut f = lock(&file);
            if !f.closed && f.stream.flush().is_err() {
                f.error = true;
                failed = true;
            }
        }
        console_flush(Console::Stdout);
        console_flush(Console::Stderr);
        return if failed { EOF } else { 0 };
    }
    match lookup(handle) {
        Some(file) => {
            let mut f = lock(&file);
            if f.closed {
                return EOF;
            }
            if f.stream.flush().is_err() {
                f.error = true;
                return EOF;
            }
            0
        }
        None => {
            console_flush(console_for(handle));
            0
        }
    }
}

/// feof(stream): the position has reached the end of a readable stream.
fn cb_feof(ctx: &mut CpuContext) -> HleResult {
    let Some(file) = lookup(ctx.rdi()) else { return 0 };
    let mut f = lock(&file);
    if f.closed || !f.stream.mode.read {
        return 0;
    }
    let pos = f.stream.pos;
    match f.stream.len() {
        Ok(len) => (pos >= len) as i64,
        Err(_) => 0,
    }
}

/// ferror(stream): a host I/O operation on the stream has failed.
fn cb_ferror(ctx: &mut CpuContext) -> HleResult {
    match lookup(ctx.rdi()) {
        Some(file) => lock(&file).error as i64,
        None => 0,
    }
}

/// clearerr(stream): clears the error indicator. The end-of-file indicator
/// is derived from the position (see `feof`), so there is nothing to clear.
fn cb_clearerr(ctx: &mut CpuContext) -> HleResult {
    if let Some(file) = lookup(ctx.rdi()) {
        lock(&file).error = false;
    }
    0
}

/// fileno(stream): the descriptor of the stream's host file. A stream that
/// has none (a file of a package image) or a closed one gives -1 with EBADF.
fn cb_fileno(ctx: &mut CpuContext) -> HleResult {
    let host = lookup(ctx.rdi()).and_then(|file| {
        let f = lock(&file);
        match (&f.closed, &f.stream.file) {
            (false, FileSource::Host(h)) => Some(crate::host::fs::as_raw_fd(h)),
            _ => None,
        }
    });
    match host {
        Some(fd) => fd as i64,
        None => {
            crate::kernel::set_errno(ctx, EBADF);
            -1
        }
    }
}

/// fileno_unlocked(stream): fileno without errno for a null or closed
/// stream.
fn cb_fileno_unlocked(ctx: &mut CpuContext) -> HleResult {
    lookup(ctx.rdi())
        .and_then(|file| {
            let f = lock(&file);
            match (&f.closed, &f.stream.file) {
                (false, FileSource::Host(h)) => Some(crate::host::fs::as_raw_fd(h) as i64),
                _ => None,
            }
        })
        .unwrap_or(-1)
}

/// getw(stream): one int as stored in memory (little-endian); -1 when fewer
/// than four bytes remain.
fn cb_getw(ctx: &mut CpuContext) -> HleResult {
    let Some(file) = lookup(ctx.rdi()) else { return EOF };
    let mut f = lock(&file);
    if f.closed {
        return EOF;
    }
    let mut word = [0u8; 4];
    let mut got = 0;
    while got < 4 {
        match f.stream.read(&mut word[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(_) => {
                f.error = true;
                break;
            }
        }
    }
    if got < 4 {
        return EOF;
    }
    i32::from_le_bytes(word) as i64
}

/// putw(w, stream): the int's four bytes, low byte first; 0 or -1.
fn cb_putw(ctx: &mut CpuContext) -> HleResult {
    let (w, handle) = (ctx.rdi() as u32, ctx.rsi());
    let Some(file) = lookup(handle) else { return eof_with_errno(ctx, EBADF) };
    let mut f = lock(&file);
    if f.closed {
        drop(f);
        return eof_with_errno(ctx, EBADF);
    }
    if f.write_buffered(&w.to_le_bytes()).is_ok() { 0 } else { EOF }
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// fread(dst, size, count, stream) -> items read
fn cb_fread(ctx: &mut CpuContext) -> HleResult {
    let (dst, size, count, handle) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    if size == 0 || count == 0 {
        return 0;
    }
    if dst == 0 {
        return 0;
    }
    let Some(file) = lookup(handle) else { return 0 };
    let total = size.saturating_mul(count);
    let mut f = lock(&file);
    if f.closed {
        return 0;
    }
    let mut buf = vec![0u8; (total.min(IO_CHUNK as u64)) as usize];
    let mut done = 0u64;
    while done < total {
        let want = (total - done).min(buf.len() as u64) as usize;
        let n = match f.stream.read(&mut buf[..want]) {
            Ok(n) => n,
            Err(_) => {
                f.error = true;
                break;
            }
        };
        if n == 0 {
            break;
        }
        if ctx.vm.write(dst + done, &buf[..n]).is_err() {
            break;
        }
        done += n as u64;
    }
    (done / size) as i64
}

/// fgets(dst, n, stream) -> dst / NULL
fn cb_fgets(ctx: &mut CpuContext) -> HleResult {
    let (dst, max, handle) = (ctx.rdi(), ctx.rsi() as i32, ctx.rdx());
    if dst == 0 || max <= 0 {
        return 0;
    }
    let Some(file) = lookup(handle) else { return 0 };
    let mut f = lock(&file);
    if f.closed {
        return 0;
    }
    let mut line = Vec::new();
    while line.len() < (max - 1) as usize {
        match f.stream.read_byte() {
            Ok(Some(b)) => {
                line.push(b);
                if b == b'\n' {
                    break;
                }
            }
            Ok(None) => break,
            Err(_) => {
                f.error = true;
                break;
            }
        }
    }
    if line.is_empty() {
        return 0;
    }
    line.push(0);
    if ctx.vm.write(dst, &line).is_err() {
        return 0;
    }
    dst as i64
}

/// fgetc(stream) / getc(stream) -> byte / EOF
fn cb_fgetc(ctx: &mut CpuContext) -> HleResult {
    let Some(file) = lookup(ctx.rdi()) else { return EOF };
    let mut f = lock(&file);
    if f.closed {
        return EOF;
    }
    match f.stream.read_byte() {
        Ok(Some(b)) => b as i64,
        Ok(None) => EOF,
        Err(_) => {
            f.error = true;
            EOF
        }
    }
}

/// ungetc(c, stream) -> c / EOF
fn cb_ungetc(ctx: &mut CpuContext) -> HleResult {
    let (c, handle) = (ctx.rdi() as i32, ctx.rsi());
    if c == EOF as i32 {
        return EOF;
    }
    let Some(file) = lookup(handle) else { return EOF };
    let mut f = lock(&file);
    if f.closed {
        return EOF;
    }
    match f.stream.unread(c as u8) {
        Ok(true) => (c as u8) as i64,
        Ok(false) => EOF,
        Err(_) => {
            f.error = true;
            EOF
        }
    }
}

/// fscanf(stream, format, ...) -> items assigned / EOF
fn cb_fscanf(ctx: &mut CpuContext) -> HleResult {
    let (handle, format) = (ctx.rdi(), ctx.rsi());
    let mut args = crate::printf::RegArgsCursor::new(ctx, 2);
    fscanf_core(ctx, handle, format, &mut args, false)
}

/// fscanf_s(stream, format, ...): fscanf under the `_s` rules (a capacity
/// after every %c, %s and %[ target; a null stream, format or target is
/// EOF).
fn cb_fscanf_s(ctx: &mut CpuContext) -> HleResult {
    let (handle, format) = (ctx.rdi(), ctx.rsi());
    if handle == 0 || format == 0 {
        return EOF;
    }
    let mut args = crate::printf::RegArgsCursor::new(ctx, 2);
    fscanf_core(ctx, handle, format, &mut args, true)
}

/// vfscanf_s(stream, format, va_list): [`cb_fscanf_s`] over a va_list.
fn cb_vfscanf_s(ctx: &mut CpuContext) -> HleResult {
    let (handle, format, va_list) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if handle == 0 || format == 0 {
        return EOF;
    }
    let Some(mut args) = crate::printf::VaListCursor::new(ctx, va_list) else { return EOF };
    fscanf_core(ctx, handle, format, &mut args, true)
}

fn fscanf_core(ctx: &CpuContext, handle: u64, format: u64, args: &mut dyn crate::printf::ArgSource, secure: bool) -> HleResult {
    let Some(format) = read_cstr(ctx, format, MAX_TEXT_LENGTH) else {
        return eof_with_errno(ctx, EINVAL);
    };
    // Console streams have no input.
    let Some(file) = lookup(handle) else { return EOF };
    let mut f = lock(&file);
    if f.closed || !f.stream.mode.read {
        return eof_with_errno(ctx, EBADF);
    }
    let mut failed = false;
    let (n, leftover) = {
        let stream = &mut f.stream;
        let mut src = || match stream.read_byte() {
            Ok(b) => b.map(u16::from),
            Err(_) => {
                failed = true;
                None
            }
        };
        let format: Vec<u16> = format.iter().map(|&b| b as u16).collect();
        let opts = crate::scanf::Options { secure, wide_input: false };
        crate::scanf::scan_units(ctx, &format, &mut src, args, opts)
    };
    if failed {
        f.error = true;
    }
    if let Some(u) = leftover {
        let _ = f.stream.unread(u as u8);
    }
    n
}

// ---------------------------------------------------------------------------
// FILE locks (_Lockfilelock / _Unlockfilelock)
// ---------------------------------------------------------------------------

/// Recursive per-FILE lock owned by a guest thread. Works for any FILE
/// pointer, including the shipped libc's console objects.
struct FileLock {
    state: Mutex<(u64, u32)>,
    cv: std::sync::Condvar,
}

static FILE_LOCKS: Mutex<Option<HashMap<u64, Arc<FileLock>>>> = Mutex::new(None);

fn file_lock(handle: u64) -> Arc<FileLock> {
    let mut g = FILE_LOCKS.lock().unwrap_or_else(|p| p.into_inner());
    g.get_or_insert_with(HashMap::new)
        .entry(handle)
        .or_insert_with(|| Arc::new(FileLock { state: Mutex::new((0, 0)), cv: std::sync::Condvar::new() }))
        .clone()
}

/// Owner identity of the calling thread (host thread id when the caller
/// is not a guest thread).
fn lock_owner() -> u64 {
    match crate::pthread::current_handle() {
        0 => (unsafe { libc::pthread_self() } as u64) | (1 << 63),
        h => h,
    }
}

/// _Lockfilelock(FILE*) / flockfile(FILE*)
fn cb_lockfilelock(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    if handle == 0 {
        return 0;
    }
    let l = file_lock(handle);
    let me = lock_owner();
    let mut s = l.state.lock().unwrap_or_else(|p| p.into_inner());
    while s.1 > 0 && s.0 != me {
        s = crate::signals::wait_cv(&l.state, &l.cv, s, None).0;
    }
    s.0 = me;
    s.1 += 1;
    0
}

/// _Unlockfilelock(FILE*) / funlockfile(FILE*)
fn cb_unlockfilelock(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    if handle == 0 {
        return 0;
    }
    let l = file_lock(handle);
    let me = lock_owner();
    let mut s = l.state.lock().unwrap_or_else(|p| p.into_inner());
    if s.1 == 0 || s.0 != me {
        log::debug!("[stdio] _Unlockfilelock({handle:#x}) by a non-owner");
        return 0;
    }
    s.1 -= 1;
    if s.1 == 0 {
        s.0 = 0;
        drop(s);
        l.cv.notify_one();
    }
    0
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Write `payload` to a stream: a known FILE goes through its buffer, any
/// other pointer is a console stream. Returns false on an I/O failure or a
/// closed stream.
fn write_stream(handle: u64, payload: &[u8]) -> bool {
    match lookup(handle) {
        Some(file) => {
            let mut f = lock(&file);
            !f.closed && f.write_buffered(payload).is_ok()
        }
        None => {
            console_write(console_for(handle), payload);
            true
        }
    }
}

/// fputc(c, stream) -> c / EOF
fn cb_fputc(ctx: &mut CpuContext) -> HleResult {
    let (c, handle) = (ctx.rdi() as u8, ctx.rsi());
    if write_stream(handle, &[c]) {
        c as i64
    } else {
        EOF
    }
}

/// Append the UTF-8 form of one code point. A lone surrogate keeps its
/// three-byte form, as the reference's encoder writes it.
fn push_utf8(out: &mut Vec<u8>, code: u32) {
    if code < 0x80 {
        out.push(code as u8);
    } else if code < 0x800 {
        out.push(0xC0 | (code >> 6) as u8);
        out.push(0x80 | (code & 0x3F) as u8);
    } else if code < 0x1_0000 {
        out.push(0xE0 | (code >> 12) as u8);
        out.push(0x80 | ((code >> 6) & 0x3F) as u8);
        out.push(0x80 | (code & 0x3F) as u8);
    } else {
        out.push(0xF0 | (code >> 18) as u8);
        out.push(0x80 | ((code >> 12) & 0x3F) as u8);
        out.push(0x80 | ((code >> 6) & 0x3F) as u8);
        out.push(0x80 | (code & 0x3F) as u8);
    }
}

/// UTF-8 of a run of wide characters. The target's wchar_t is one 16-bit
/// UTF-16 unit; a high surrogate followed by a low one is one code point.
fn wide_to_utf8(units: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(units.len());
    let mut i = 0;
    while i < units.len() {
        let mut code = units[i] as u32;
        if (0xD800..0xDC00).contains(&code) && i + 1 < units.len() && (0xDC00..0xE000).contains(&(units[i + 1] as u32)) {
            code = 0x1_0000 + ((code - 0xD800) << 10) + (units[i + 1] as u32 - 0xDC00);
            i += 1;
        }
        i += 1;
        push_utf8(&mut out, code);
    }
    out
}

/// fputwc(wc, stream) -> wc / EOF. Only the low 16 bits of the argument are
/// the character; it is written as UTF-8.
fn cb_fputwc(ctx: &mut CpuContext) -> HleResult {
    let (unit, handle) = (ctx.rdi() as u16, ctx.rsi());
    if write_stream(handle, &wide_to_utf8(&[unit])) {
        unit as i64
    } else {
        EOF
    }
}

/// NUL-terminated UTF-16 string at `addr` without its terminator. None on a
/// null pointer, a fault, or when no NUL comes within `max` units.
fn read_wstr(ctx: &CpuContext, addr: u64, max: usize) -> Option<Vec<u16>> {
    if addr == 0 {
        return None;
    }
    let mut out = Vec::new();
    loop {
        let mut b = [0u8; 2];
        ctx.vm.read(addr + 2 * out.len() as u64, &mut b).ok()?;
        match u16::from_le_bytes(b) {
            0 => return Some(out),
            _ if out.len() >= max => return None,
            unit => out.push(unit),
        }
    }
}

/// fputws(str, stream) -> number of wide characters written / EOF. The
/// string is written as UTF-8; the count is the one the reference returns.
fn cb_fputws(ctx: &mut CpuContext) -> HleResult {
    let (addr, handle) = (ctx.rdi(), ctx.rsi());
    let Some(units) = read_wstr(ctx, addr, MAX_TEXT_LENGTH) else { return EOF };
    if write_stream(handle, &wide_to_utf8(&units)) {
        units.len() as i64
    } else {
        EOF
    }
}

/// fwrite(src, size, count, stream) -> items written
fn cb_fwrite(ctx: &mut CpuContext) -> HleResult {
    let (src, size, count, handle) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    if size == 0 || count == 0 || src == 0 {
        return 0;
    }
    let total = size.saturating_mul(count);
    let file = lookup(handle);
    let mut buf = vec![0u8; (total.min(IO_CHUNK as u64)) as usize];
    let mut done = 0u64;
    while done < total {
        let want = (total - done).min(buf.len() as u64) as usize;
        if ctx.vm.read(src + done, &mut buf[..want]).is_err() {
            break;
        }
        match &file {
            Some(file) => {
                let mut f = lock(file);
                if f.closed || f.write_buffered(&buf[..want]).is_err() {
                    break;
                }
            }
            None => console_write(console_for(handle), &buf[..want]),
        }
        done += want as u64;
    }
    (done / size) as i64
}

/// fputs(str, stream) -> bytes written / EOF
fn cb_fputs(ctx: &mut CpuContext) -> HleResult {
    let (text_addr, handle) = (ctx.rdi(), ctx.rsi());
    if text_addr == 0 {
        return EOF;
    }
    let Some(text) = read_cstr(ctx, text_addr, MAX_TEXT_LENGTH) else { return EOF };
    if write_stream(handle, &text) {
        text.len() as i64
    } else {
        EOF
    }
}

fn write_rendered(handle: u64, rendered: &[u8]) -> HleResult {
    if write_stream(handle, rendered) {
        rendered.len() as i64
    } else {
        EOF
    }
}

/// fprintf(stream, format, ...) -> bytes written / -1
fn cb_fprintf(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    let Some(format) = ctx.read_cstr_bytes(ctx.rsi(), MAX_PATH_LENGTH) else { return EOF };
    let rendered = {
        let mut args = crate::printf::RegArgsCursor::new(ctx, 2);
        crate::printf::format_bytes(ctx, &format, &mut args)
    };
    write_rendered(handle, &rendered)
}

/// fprintf_s(stream, format, ...) -> bytes written / -1: fprintf under the
/// `_s` rules (C11 K.3.5.3.1). A null stream or format, any %n and a null
/// string argument return -1 and write nothing, the output being formatted
/// completely first.
fn cb_fprintf_s(ctx: &mut CpuContext) -> HleResult {
    let (handle, format) = (ctx.rdi(), ctx.rsi());
    let mut args = crate::printf::RegArgsCursor::new(ctx, 2);
    fprintf_s_core(ctx, handle, format, Some(&mut args))
}

/// vfprintf_s(stream, format, va_list): [`cb_fprintf_s`] over a va_list.
fn cb_vfprintf_s(ctx: &mut CpuContext) -> HleResult {
    let (handle, format, va_list) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let mut cursor = crate::printf::VaListCursor::new(ctx, va_list);
    fprintf_s_core(ctx, handle, format, cursor.as_mut().map(|c| c as &mut dyn crate::printf::ArgSource))
}

/// The `_s` printf into stream `handle`.
fn fprintf_s_core(ctx: &CpuContext, handle: u64, format: u64, args: Option<&mut dyn crate::printf::ArgSource>) -> HleResult {
    if handle == 0 || format == 0 {
        return EOF;
    }
    let (Some(format), Some(args)) = (ctx.read_cstr_bytes(format, MAX_PATH_LENGTH), args) else { return EOF };
    let Some(rendered) = crate::printf::format_bytes_secure(ctx, &format, args) else { return EOF };
    write_rendered(handle, &rendered)
}

/// vfprintf(stream, format, va_list) -> bytes written / -1
fn cb_vfprintf(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    let Some(format) = ctx.read_cstr_bytes(ctx.rsi(), MAX_PATH_LENGTH) else { return EOF };
    let rendered = match crate::printf::VaListCursor::new(ctx, ctx.rdx()) {
        Some(mut cursor) => crate::printf::format_bytes(ctx, &format, &mut cursor),
        None => format,
    };
    write_rendered(handle, &rendered)
}

// ---------------------------------------------------------------------------
// Dinkumware ctype tables
// ---------------------------------------------------------------------------

/// The tables cover c in [-128, 255]; the guest pointer addresses c == 0.
const CTYPE_LOWER_BOUND: i32 = -128;
const CTYPE_UPPER_BOUND: i32 = 255;
const CTYPE_ENTRIES: usize = (CTYPE_UPPER_BOUND - CTYPE_LOWER_BOUND + 1) as usize;
const CTYPE_ZERO_OFFSET: u64 = (-CTYPE_LOWER_BOUND as u64) * 2;

/// Dinkumware ctype bits.
pub mod ctype {
    pub const XD: u16 = 0x001; // hex digit
    pub const UP: u16 = 0x002; // upper case
    pub const SP: u16 = 0x004; // ' '
    pub const PU: u16 = 0x008; // punctuation
    pub const LO: u16 = 0x010; // lower case
    pub const DI: u16 = 0x020; // decimal digit
    pub const CN: u16 = 0x040; // '\t' '\n' '\v' '\f' '\r'
    pub const BB: u16 = 0x080; // control characters (0x00-0x1F, 0x7F)
    pub const XB: u16 = 0x400; // '\t', the blank besides ' ' (SP)
}

/// Classification bits of one character (0 outside ASCII), as the
/// console's table has them: every control character is BB, the spacing
/// controls also CN, ' ' only SP and '\t' CN | BB | XB, so `isblank`
/// (SP | XB) answers 0x04 for ' ' and 0x400 for '\t'.
pub fn ctype_flags(c: i32) -> u16 {
    use ctype::*;
    if !(0..=0x7F).contains(&c) {
        return 0;
    }
    let ch = c as u8;
    let upper = ch.is_ascii_uppercase();
    let lower = ch.is_ascii_lowercase();
    let digit = ch.is_ascii_digit();
    let alnum = upper || lower || digit;
    let ctrl_space = (0x09..=0x0D).contains(&ch);
    let ctrl = ch <= 0x1F || ch == 0x7F;
    let print = (0x20..=0x7E).contains(&ch);
    let mut f = 0;
    if upper {
        f |= UP;
    }
    if lower {
        f |= LO;
    }
    if digit {
        f |= DI;
    }
    if ch.is_ascii_hexdigit() {
        f |= XD;
    }
    if ch == b' ' {
        f |= SP;
    }
    if ch == b'\t' {
        f |= XB;
    }
    if ctrl_space {
        f |= CN;
    }
    if print && !alnum && ch != b' ' {
        f |= PU;
    }
    if ctrl {
        f |= BB;
    }
    f
}

#[derive(Clone, Copy)]
enum Table {
    Ctype,
    ToLower,
    ToUpper,
}

/// Table entry for character `c`.
fn table_entry(t: Table, c: i32) -> u16 {
    match t {
        Table::Ctype => ctype_flags(c),
        Table::ToLower if (b'A' as i32..=b'Z' as i32).contains(&c) => (c + 32) as u16,
        Table::ToUpper if (b'a' as i32..=b'z' as i32).contains(&c) => (c - 32) as u16,
        _ => c as i16 as u16,
    }
}

fn build_table(t: Table) -> Vec<u8> {
    (0..CTYPE_ENTRIES)
        .flat_map(|i| table_entry(t, i as i32 + CTYPE_LOWER_BOUND).to_le_bytes())
        .collect()
}

static TABLES: Mutex<[u64; 3]> = Mutex::new([0; 3]);

/// Guest pointer to the c == 0 entry of a table, built on first use.
fn table_pointer(ctx: &CpuContext, t: Table) -> HleResult {
    let mut g = TABLES.lock().unwrap_or_else(|p| p.into_inner());
    let slot = &mut g[t as usize];
    if *slot == 0 {
        let bytes = build_table(t);
        let Some(base) = crate::pthread::alloc_guest_block(ctx, bytes.len() as u64) else {
            return 0;
        };
        if !ctx.write_bytes(base, &bytes) {
            return 0;
        }
        *slot = base + CTYPE_ZERO_OFFSET;
    }
    *slot as i64
}

/// _Getpctype() -> const short* (classification table)
fn cb_getpctype(ctx: &mut CpuContext) -> HleResult {
    table_pointer(ctx, Table::Ctype)
}

/// _Getptolower() -> const short* (tolower table)
fn cb_getptolower(ctx: &mut CpuContext) -> HleResult {
    table_pointer(ctx, Table::ToLower)
}

/// _Getptoupper() -> const short* (toupper table)
fn cb_getptoupper(ctx: &mut CpuContext) -> HleResult {
    table_pointer(ctx, Table::ToUpper)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("xeYO4u7uyJ0", "fopen", cb_fopen),
    ("gkWgn0p1AfU", "freopen", cb_freopen),
    ("uodLYyUip20", "fclose", cb_fclose),
    ("rQFVBXp-Cxg", "fseek", cb_fseek),
    ("Qazy8LmXTvw", "ftell", cb_ftell),
    ("SHlt7EhOtqA", "fgetpos", cb_fgetpos),
    ("7PkSz+qnTto", "fsetpos", cb_fsetpos),
    ("3QIPIh-GDjw", "rewind", cb_rewind),
    ("QMFyLoqNxIg", "setvbuf", cb_setvbuf),
    ("MUjC4lbHrK4", "fflush", cb_fflush),
    ("LxcEU+ICu8U", "feof", cb_feof),
    ("AHxyhN96dy4", "ferror", cb_ferror),
    ("St9nbxSoezk", "clearerr", cb_clearerr),
    ("Fm-dmyywH9Q", "fileno", cb_fileno),
    ("kxm0z4T5mMI", "fileno_unlocked", cb_fileno_unlocked),
    ("cYNk9M+7YkY", "clearerr_unlocked", cb_clearerr),
    ("NuydofHcR1w", "feof_unlocked", cb_feof),
    ("yxbGzBQC5xA", "ferror_unlocked", cb_ferror),
    ("AoLA2MRWJvc", "getw", cb_getw),
    ("DwcWtj3tSPA", "putw", cb_putw),
    ("lbB+UlZqVG0", "fread", cb_fread),
    ("KdP-nULpuGw", "fgets", cb_fgets),
    ("AEuF3F2f8TA", "fgetc", cb_fgetc),
    ("8Q60JLJ6Rv4", "getc", cb_fgetc),
    // The unlocked forms: the stream's own lock (flockfile) is never taken
    // by the locked forms either, so they behave the same.
    ("5tM252Rs2fc", "getc_unlocked", cb_fgetc),
    ("aZK8lNei-Qw", "fputc", cb_fputc),
    ("H-QeERgWuTM", "putc_unlocked", cb_fputc),
    ("1QJWxoB6pCo", "fputwc", cb_fputwc),
    ("-7nRJFXMxnM", "fputws", cb_fputws),
    ("MpxhMh8QFro", "fwrite", cb_fwrite),
    ("QrZZdJ8XsX0", "fputs", cb_fputs),
    ("fffwELXNVFA", "fprintf", cb_fprintf),
    ("pDBDcY6uLSA", "vfprintf", cb_vfprintf),
    ("-LFO7jhD5CE", "ungetc", cb_ungetc),
    ("npLpPTaSuHg", "fscanf", cb_fscanf),
    ("vZkmJmvqueY", "_Lockfilelock", cb_lockfilelock),
    ("0x7rx8TKy2Y", "_Unlockfilelock", cb_unlockfilelock),
    // The POSIX names of the same recursive per-FILE lock: a FILE here is
    // an HLE stream, which the shipped libc's own lock code cannot read.
    ("hGljHZEfF0U", "flockfile", cb_lockfilelock),
    ("vAc9y8UQ31o", "funlockfile", cb_unlockfilelock),
    ("NL836gOLANs", "fopen_s", cb_fopen_s),
    ("NdvAi34vV3g", "freopen_s", cb_freopen_s),
    ("-e-F9HjUFp8", "fprintf_s", cb_fprintf_s),
    ("GhTZtaodo7o", "vfprintf_s", cb_vfprintf_s),
    ("vj2WUi2LrfE", "fscanf_s", cb_fscanf_s),
    ("JjPXy-HX5dY", "vfscanf_s", cb_vfscanf_s),
];

/// Used only when no loaded libc exports its own tables.
const FALLBACK_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("sUP1hBaouOw", "_Getpctype", cb_getpctype),
    ("1uJgoVq3bQU", "_Getptolower", cb_getptolower),
    ("rcQCUr0EaRU", "_Getptoupper", cb_getptoupper),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, "libc", cb);
    }
    for &(nid, name, cb) in FALLBACK_EXPORTS {
        reg.register_fallback_fn(nid, name, "libc", cb);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn temp_root(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("rps5e-stdio-test-{}-{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS.iter().chain(FALLBACK_EXPORTS) {
            assert_eq!(ps5nid::compute_nid(name).replace('_', "-"), nid, "NID mismatch for {name}");
        }
    }

    /// fopen("/dev/urandom", "rb") streams read the host CSPRNG.
    #[test]
    fn random_device_stream_reads() {
        assert!(crate::filesystem::is_random_device("/dev/urandom") && crate::filesystem::is_random_device("/dev/random"));
        let mut s = open_host(Path::new("/dev/urandom"), parse_mode("rb").unwrap()).unwrap();
        let mut a = [0u8; 64];
        let mut b = [0u8; 64];
        assert_eq!(s.read(&mut a).unwrap(), 64);
        assert_eq!(s.read(&mut b).unwrap(), 64);
        assert!(a != [0; 64] && a != b);
    }

    #[test]
    fn mode_parsing() {
        let r = parse_mode("r").unwrap();
        assert!(r.read && !r.write && !r.create && !r.truncate && !r.append);
        let rb = parse_mode("rb+").unwrap();
        assert!(rb.read && rb.write && !rb.create && !rb.truncate);
        let w = parse_mode("wb").unwrap();
        assert!(!w.read && w.write && w.create && w.truncate && !w.append);
        let wp = parse_mode("w+").unwrap();
        assert!(wp.read && wp.write && wp.truncate);
        let a = parse_mode("a").unwrap();
        assert!(!a.read && a.write && a.create && !a.truncate && a.append);
        let ap = parse_mode("a+b").unwrap();
        assert!(ap.read && ap.write && ap.append);
        assert!(parse_mode("").is_none());
        assert!(parse_mode("x").is_none());
        assert!(parse_mode("+r").is_none());
        // FreeBSD __sflags: the flags end at the first other character.
        for m in ["rt", "rbv", "re"] {
            let p = parse_mode(m).unwrap();
            assert!(p.read && !p.write, "{m}");
        }
        let rt_plus = parse_mode("rt+").unwrap();
        assert!(rt_plus.read && !rt_plus.write, "the '+' after 't' is not read");
        assert!(parse_mode("r+t").unwrap().write);
        assert!(parse_mode("wt").unwrap().truncate && parse_mode("w+e").unwrap().read && parse_mode("ae").unwrap().append);
        for m in ["q", "tr", "rx", "rbx"] {
            assert!(parse_mode(m).is_none(), "{m}");
        }
        let wx = parse_mode("wx").unwrap();
        assert!(wx.write && wx.create && wx.exclusive);
        assert!(parse_mode("r+x").unwrap().exclusive);
    }

    /// fopen modes against real files: 'x' refuses an existing file with
    /// EEXIST and creates a missing one; invalid modes give EINVAL.
    #[test]
    #[cfg(unix)]
    fn fopen_exclusive_and_invalid_modes() {
        let root = temp_root("modes");
        let mount = root.join("mnt");
        fs::create_dir_all(&mount).unwrap();
        fs::write(mount.join("bytes"), b"A").unwrap();
        assert!(crate::filesystem::register_guest_path_mount("/stdiomode0", &mount));
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E61_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let ctx = CpuContext::new(&vm);
        let s = |off: u64, text: &str| {
            let mut b = text.as_bytes().to_vec();
            b.push(0);
            ctx.vm.write(BASE + off, &b).unwrap();
            BASE + off
        };
        let path = s(0, "/stdiomode0/bytes");
        for mode in ["", "q", "tr", "rx", "rbx"] {
            assert_eq!(fopen_core(&ctx, path, s(0x100, mode)), Err(EINVAL), "{mode:?}");
        }
        assert_eq!(fopen_core(&ctx, path, s(0x100, "wx")), Err(EEXIST));
        assert_eq!(fs::read(mount.join("bytes")).unwrap(), b"A", "the refused 'wx' kept the file");
        // Open errors as open(2) reports them: a directory for writing is
        // EISDIR, a path through a regular file ENOTDIR, a symbolic link
        // loop ELOOP.
        fs::create_dir_all(mount.join("dir")).unwrap();
        for mode in ["w", "a", "r+", "w+", "a+"] {
            assert_eq!(fopen_core(&ctx, s(0x200, "/stdiomode0/dir"), s(0x100, mode)), Err(21), "{mode}");
        }
        for mode in ["r", "w"] {
            assert_eq!(fopen_core(&ctx, s(0x200, "/stdiomode0/bytes/child"), s(0x100, mode)), Err(20), "{mode}");
        }
        std::os::unix::fs::symlink("loop", mount.join("loop")).unwrap();
        assert_eq!(fopen_core(&ctx, s(0x200, "/stdiomode0/loop"), s(0x100, "r")), Err(62));
        fs::remove_file(mount.join("bytes")).unwrap();
        assert!(fopen_core(&ctx, path, s(0x100, "wx")).is_ok());
        assert!(mount.join("bytes").exists());
        crate::filesystem::unregister_guest_path_mount("/stdiomode0");
        vm.unmap(BASE, 0x1000).unwrap();
        let _ = fs::remove_dir_all(&root);
    }

    /// fputwc writes the UTF-8 form of one UTF-16 unit (a lone surrogate in
    /// its three-byte form) and returns the unit; only the low 16 bits of
    /// the argument are the character. A read-only stream fails with EOF.
    #[test]
    fn fputwc_writes_utf8_units() {
        use crate::context::CpuRegister::*;
        let root = temp_root("fputwc");
        let mount = root.join("mnt");
        fs::create_dir_all(&mount).unwrap();
        assert!(crate::filesystem::register_guest_path_mount("/stdiowide0", &mount));
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E70_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let s = |ctx: &CpuContext, off: u64, text: &str| {
            let mut b = text.as_bytes().to_vec();
            b.push(0);
            ctx.vm.write(BASE + off, &b).unwrap();
            BASE + off
        };
        let set = |ctx: &mut CpuContext, args: &[u64]| {
            for (r, v) in [Rdi, Rsi, Rdx, Rcx].iter().zip(args) {
                ctx.set(*r, *v);
            }
        };
        let path = s(&ctx, 0, "/stdiowide0/w.txt");
        let mode_w = s(&ctx, 0x100, "w");
        let mode_r = s(&ctx, 0x110, "r");
        set(&mut ctx, &[path, mode_w]);
        let f = cb_fopen(&mut ctx) as u64;
        assert_ne!(f, 0);
        set(&mut ctx, &[b'A' as u64, f]);
        assert_eq!(cb_fputwc(&mut ctx), 0x41);
        set(&mut ctx, &[0xE9, f]);
        assert_eq!(cb_fputwc(&mut ctx), 0xE9, "U+00E9 is two bytes");
        set(&mut ctx, &[0x1_20AC, f]);
        assert_eq!(cb_fputwc(&mut ctx), 0x20AC, "only the low 16 bits are the character");
        set(&mut ctx, &[0xD800, f]);
        assert_eq!(cb_fputwc(&mut ctx), 0xD800, "a lone surrogate is written in its three-byte form");
        set(&mut ctx, &[0xDEAD_0000_0042, f]);
        assert_eq!(cb_fputwc(&mut ctx), 0x42);
        set(&mut ctx, &[f]);
        assert_eq!(cb_fclose(&mut ctx), 0);
        assert_eq!(fs::read(mount.join("w.txt")).unwrap(), b"A\xC3\xA9\xE2\x82\xAC\xED\xA0\x80B");

        set(&mut ctx, &[path, mode_r]);
        let r = cb_fopen(&mut ctx) as u64;
        assert_ne!(r, 0);
        set(&mut ctx, &[b'x' as u64, r]);
        assert_eq!(cb_fputwc(&mut ctx), EOF, "a read-only stream refuses the write");
        set(&mut ctx, &[r]);
        assert_eq!(cb_fclose(&mut ctx), 0);

        crate::filesystem::unregister_guest_path_mount("/stdiowide0");
        let _ = vm.unmap(BASE, 0x1000);
        let _ = fs::remove_dir_all(&root);
    }

    /// fputws writes the UTF-8 of a NUL-terminated UTF-16 string and returns
    /// its unit count (a surrogate pair counts as two). An empty string
    /// writes nothing; a null string and a read-only stream fail with EOF.
    #[test]
    fn fputws_writes_utf8_string() {
        use crate::context::CpuRegister::*;
        let root = temp_root("fputws");
        let mount = root.join("mnt");
        fs::create_dir_all(&mount).unwrap();
        assert!(crate::filesystem::register_guest_path_mount("/stdiowide1", &mount));
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E71_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let s = |ctx: &CpuContext, off: u64, text: &str| {
            let mut b = text.as_bytes().to_vec();
            b.push(0);
            ctx.vm.write(BASE + off, &b).unwrap();
            BASE + off
        };
        let put_wide = |ctx: &CpuContext, off: u64, units: &[u16]| {
            let mut b: Vec<u8> = units.iter().flat_map(|u| u.to_le_bytes()).collect();
            b.extend_from_slice(&[0, 0]);
            ctx.vm.write(BASE + off, &b).unwrap();
            BASE + off
        };
        let set = |ctx: &mut CpuContext, args: &[u64]| {
            for (r, v) in [Rdi, Rsi, Rdx, Rcx].iter().zip(args) {
                ctx.set(*r, *v);
            }
        };
        let path = s(&ctx, 0, "/stdiowide1/ws.txt");
        let mode_w = s(&ctx, 0x100, "w");
        let mode_r = s(&ctx, 0x110, "r");
        // "B", U+00E9, U+20AC and U+1F600 (a surrogate pair): 5 units, 10 bytes.
        let text = put_wide(&ctx, 0x200, &[0x42, 0xE9, 0x20AC, 0xD83D, 0xDE00]);
        let empty = put_wide(&ctx, 0x300, &[]);
        set(&mut ctx, &[path, mode_w]);
        let f = cb_fopen(&mut ctx) as u64;
        assert_ne!(f, 0);
        set(&mut ctx, &[text, f]);
        assert_eq!(cb_fputws(&mut ctx), 5);
        set(&mut ctx, &[empty, f]);
        assert_eq!(cb_fputws(&mut ctx), 0, "an empty string writes nothing");
        set(&mut ctx, &[0, f]);
        assert_eq!(cb_fputws(&mut ctx), EOF, "a null string fails");
        set(&mut ctx, &[f]);
        assert_eq!(cb_fclose(&mut ctx), 0);
        assert_eq!(fs::read(mount.join("ws.txt")).unwrap(), b"B\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80");

        set(&mut ctx, &[path, mode_r]);
        let r = cb_fopen(&mut ctx) as u64;
        assert_ne!(r, 0);
        set(&mut ctx, &[text, r]);
        assert_eq!(cb_fputws(&mut ctx), EOF, "a read-only stream refuses the write");
        set(&mut ctx, &[r]);
        assert_eq!(cb_fclose(&mut ctx), 0);

        crate::filesystem::unregister_guest_path_mount("/stdiowide1");
        let _ = vm.unmap(BASE, 0x1000);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn utf8_of_wide_units() {
        assert_eq!(wide_to_utf8(&[0x41, 0xE9, 0x20AC]), "A\u{e9}\u{20ac}".as_bytes());
        // A surrogate pair is one code point; a lone surrogate is three bytes.
        assert_eq!(wide_to_utf8(&[0xD83D, 0xDE00]), "\u{1F600}".as_bytes());
        assert_eq!(wide_to_utf8(&[0xDE00, 0x41]), b"\xED\xB8\x80A");
        assert_eq!(wide_to_utf8(&[0xD800]), b"\xED\xA0\x80");
        assert_eq!(wide_to_utf8(&[]), b"");
    }

    #[test]
    fn fpos_layout() {
        let b = encode_fpos(0x1122_3344_5566_7788);
        assert_eq!(b.len(), 24);
        assert_eq!(&b[..8], &0x1122_3344_5566_7788u64.to_le_bytes());
        assert!(b[8..].iter().all(|&x| x == 0));
        assert_eq!(decode_fpos(&b), 0x1122_3344_5566_7788);
    }

    #[test]
    fn ctype_tables() {
        use ctype::*;
        assert_eq!(ctype_flags(b'a' as i32), LO | XD);
        assert_eq!(ctype_flags(b'G' as i32), UP);
        assert_eq!(ctype_flags(b'7' as i32), DI | XD);
        // The console's table (measured class bits).
        assert_eq!(ctype_flags(b' ' as i32), 0x04);
        assert_eq!(ctype_flags(b'\t' as i32), 0x4C0);
        assert_eq!(ctype_flags(b'\n' as i32), 0xC0);
        assert_eq!(ctype_flags(b'A' as i32), 0x03);
        assert_eq!(ctype_flags(b'0' as i32), 0x21);
        assert_eq!(ctype_flags(0x01), BB);
        assert_eq!(ctype_flags(0x7F), BB);
        for c in -1..=255 {
            let mut bits = 0;
            if (0..=255).contains(&c) {
                let ch = c as u8;
                if ch.is_ascii_hexdigit() {
                    bits |= 0x01;
                }
                if ch.is_ascii_uppercase() {
                    bits |= 0x02;
                }
                if ch == b' ' {
                    bits |= 0x04;
                }
                if (33..=126).contains(&ch) && !ch.is_ascii_alphanumeric() {
                    bits |= 0x08;
                }
                if ch.is_ascii_lowercase() {
                    bits |= 0x10;
                }
                if ch.is_ascii_digit() {
                    bits |= 0x20;
                }
                if (b'\t'..=b'\r').contains(&ch) {
                    bits |= 0x40;
                }
                if ch < 32 || ch == 127 {
                    bits |= 0x80;
                }
                if ch == b'\t' {
                    bits |= 0x400;
                }
            }
            assert_eq!(ctype_flags(c), bits, "{c}");
        }
        assert_eq!(ctype_flags(b'!' as i32), PU);
        assert_eq!(ctype_flags(0x80), 0);
        assert_eq!(ctype_flags(-1), 0);
        let t = build_table(Table::ToLower);
        let at = |c: i32| i16::from_le_bytes([t[((c + 128) * 2) as usize], t[((c + 128) * 2 + 1) as usize]]);
        assert_eq!(at(b'Q' as i32), b'q' as i16);
        assert_eq!(at(b'q' as i32), b'q' as i16);
        assert_eq!(at(-1), -1);
        assert_eq!(table_entry(Table::ToUpper, b'z' as i32), b'Z' as u16);
    }

    #[test]
    fn fopen_s_fscanf_ungetc_and_locks() {
        use crate::context::CpuRegister::*;
        let root = temp_root("scan");
        let mount = root.join("mnt");
        fs::create_dir_all(&mount).unwrap();
        fs::write(mount.join("in.txt"), b"12 abc 3.5\nX").unwrap();
        assert!(crate::filesystem::register_guest_path_mount("/stdioscan0", &mount));

        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A5C_0000_0000;
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let s = |ctx: &CpuContext, off: u64, text: &str| {
            let mut b = text.as_bytes().to_vec();
            b.push(0);
            ctx.vm.write(BASE + off, &b).unwrap();
            BASE + off
        };
        let set = |ctx: &mut CpuContext, args: &[u64]| {
            for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
                ctx.set(*r, *v);
            }
        };
        let out = BASE + 0x800;

        // fopen_s: errno-style result, FILE* through the out pointer.
        let p = s(&ctx, 0, "/stdioscan0/in.txt");
        let m = s(&ctx, 0x100, "r");
        set(&mut ctx, &[out, p, m]);
        assert_eq!(cb_fopen_s(&mut ctx), 0);
        let f = ctx.read_u64(out).unwrap();
        assert_ne!(f, 0);
        let missing = s(&ctx, 0x200, "/stdioscan0/none.txt");
        set(&mut ctx, &[out + 8, missing, m]);
        assert_eq!(cb_fopen_s(&mut ctx), ENOENT as i64);
        assert_eq!(ctx.read_u64(out + 8), Some(0));
        set(&mut ctx, &[0, p, m]);
        assert_eq!(cb_fopen_s(&mut ctx), EINVAL as i64);

        // fscanf leaves the delimiter unread.
        let fmt = s(&ctx, 0x300, "%d %s");
        set(&mut ctx, &[f, fmt, BASE + 0x400, BASE + 0x410]);
        assert_eq!(cb_fscanf(&mut ctx), 2);
        assert_eq!(ctx.read_u32(BASE + 0x400), Some(12));
        assert_eq!(ctx.read_cstring(BASE + 0x410, 8).as_deref(), Some("abc"));
        set(&mut ctx, &[f]);
        assert_eq!(cb_ftell(&mut ctx), 6);
        assert_eq!(cb_fgetc(&mut ctx), b' ' as i64);

        // ungetc of a different byte, then fseek discards it.
        set(&mut ctx, &[b'Z' as u64, f]);
        assert_eq!(cb_ungetc(&mut ctx), b'Z' as i64);
        set(&mut ctx, &[f]);
        assert_eq!(cb_ftell(&mut ctx), 6);
        assert_eq!(cb_fgetc(&mut ctx), b'Z' as i64);
        set(&mut ctx, &[b'Q' as u64, f]);
        assert_eq!(cb_ungetc(&mut ctx), b'Q' as i64);
        set(&mut ctx, &[f, 6, SEEK_SET as u64]);
        assert_eq!(cb_fseek(&mut ctx), 0);
        set(&mut ctx, &[f]);
        assert_eq!(cb_fgetc(&mut ctx), b' ' as i64);
        set(&mut ctx, &[EOF as u64, f]);
        assert_eq!(cb_ungetc(&mut ctx), EOF);

        let fmt = s(&ctx, 0x300, "%f %c%d");
        set(&mut ctx, &[f, fmt, BASE + 0x400, BASE + 0x410, BASE + 0x420]);
        assert_eq!(cb_fscanf(&mut ctx), 2);
        assert_eq!(ctx.read_f32(BASE + 0x400), Some(3.5));
        assert_eq!(ctx.read_bytes(BASE + 0x410, 1).unwrap(), b"X");
        set(&mut ctx, &[f, fmt, BASE + 0x400, BASE + 0x410, BASE + 0x420]);
        assert_eq!(cb_fscanf(&mut ctx), EOF);

        // Recursive FILE lock.
        set(&mut ctx, &[f]);
        assert_eq!(cb_lockfilelock(&mut ctx), 0);
        assert_eq!(cb_lockfilelock(&mut ctx), 0);
        assert_eq!(file_lock(f).state.lock().unwrap().1, 2);
        assert_eq!(cb_unlockfilelock(&mut ctx), 0);
        assert_eq!(cb_unlockfilelock(&mut ctx), 0);
        assert_eq!(file_lock(f).state.lock().unwrap().1, 0);

        set(&mut ctx, &[f]);
        assert_eq!(cb_fclose(&mut ctx), 0);
        let _ = fs::remove_dir_all(&root);
    }

    /// The Annex K stream functions: fprintf_s refuses a null stream or
    /// format, %n and a null string without writing anything; fscanf_s takes
    /// a capacity after string targets; freopen_s reports through its
    /// result pointer.
    #[test]
    fn annex_k_stream_printf_scanf_and_freopen() {
        use crate::context::CpuRegister::*;
        let root = temp_root("annexk");
        let mount = root.join("mnt");
        fs::create_dir_all(&mount).unwrap();
        fs::write(mount.join("other.txt"), b"7 xyz").unwrap();
        assert!(crate::filesystem::register_guest_path_mount("/stdioannexk0", &mount));
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E7D_0000_0000;
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let s = |ctx: &CpuContext, off: u64, text: &str| {
            let mut b = text.as_bytes().to_vec();
            b.push(0);
            ctx.vm.write(BASE + off, &b).unwrap();
            BASE + off
        };
        let set = |ctx: &mut CpuContext, args: &[u64]| {
            for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
                ctx.set(*r, *v);
            }
        };
        let out = BASE + 0x800;
        let path = s(&ctx, 0, "/stdioannexk0/out.txt");
        let mode = s(&ctx, 0x100, "w+");
        set(&mut ctx, &[out, path, mode]);
        assert_eq!(cb_fopen_s(&mut ctx), 0);
        let f = ctx.read_u64(out).unwrap();

        let word = s(&ctx, 0x200, "word");
        let fmt = s(&ctx, 0x300, "%d %s\n");
        set(&mut ctx, &[f, fmt, 42, word]);
        assert_eq!(cb_fprintf_s(&mut ctx), 8);
        set(&mut ctx, &[0, fmt, 42, word]);
        assert_eq!(cb_fprintf_s(&mut ctx), EOF);
        set(&mut ctx, &[f, 0]);
        assert_eq!(cb_fprintf_s(&mut ctx), EOF);
        set(&mut ctx, &[f, fmt, 42, 0]);
        assert_eq!(cb_fprintf_s(&mut ctx), EOF, "a null string");
        let counted = s(&ctx, 0x380, "ab%n");
        set(&mut ctx, &[f, counted, BASE + 0x900]);
        assert_eq!(cb_fprintf_s(&mut ctx), EOF, "%n");
        // The plain form still stores the count.
        set(&mut ctx, &[f, counted, BASE + 0x900]);
        assert_eq!(cb_fprintf(&mut ctx), 2);
        assert_eq!(ctx.read_u32(BASE + 0x900), Some(2));

        set(&mut ctx, &[f]);
        assert_eq!(cb_rewind(&mut ctx), 0);
        let scan = s(&ctx, 0x400, "%d %s");
        set(&mut ctx, &[f, scan, BASE + 0xA00, BASE + 0xA10, 16]);
        assert_eq!(cb_fscanf_s(&mut ctx), 2);
        assert_eq!(ctx.read_u32(BASE + 0xA00), Some(42));
        assert_eq!(ctx.read_cstring(BASE + 0xA10, 16).as_deref(), Some("word"));
        set(&mut ctx, &[0, scan]);
        assert_eq!(cb_fscanf_s(&mut ctx), EOF);
        set(&mut ctx, &[f, 0]);
        assert_eq!(cb_fscanf_s(&mut ctx), EOF);
        set(&mut ctx, &[f]);
        assert_eq!(cb_fflush(&mut ctx), 0);
        assert_eq!(fs::read(mount.join("out.txt")).unwrap(), b"42 word\nab");

        // freopen_s
        let read = s(&ctx, 0x500, "r");
        let other = s(&ctx, 0x600, "/stdioannexk0/other.txt");
        set(&mut ctx, &[0, other, read, f]);
        assert_eq!(cb_freopen_s(&mut ctx), EINVAL as i64);
        ctx.write_u64(out, 0xAAAA);
        set(&mut ctx, &[out, other, 0, f]);
        assert_eq!(cb_freopen_s(&mut ctx), EINVAL as i64);
        assert_eq!(ctx.read_u64(out), Some(0));
        ctx.write_u64(out, 0xAAAA);
        set(&mut ctx, &[out, other, read, 0]);
        assert_eq!(cb_freopen_s(&mut ctx), EINVAL as i64);
        assert_eq!(ctx.read_u64(out), Some(0));
        set(&mut ctx, &[out, other, read, f]);
        assert_eq!(cb_freopen_s(&mut ctx), 0);
        assert_eq!(ctx.read_u64(out), Some(f));
        set(&mut ctx, &[f, scan, BASE + 0xA00, BASE + 0xA10, 16]);
        assert_eq!(cb_fscanf_s(&mut ctx), 2);
        assert_eq!(ctx.read_u32(BASE + 0xA00), Some(7));
        let missing = s(&ctx, 0x700, "/stdioannexk0/none.txt");
        ctx.write_u64(out, 0xAAAA);
        set(&mut ctx, &[out, missing, read, f]);
        assert_eq!(cb_freopen_s(&mut ctx), ENOENT as i64);
        assert_eq!(ctx.read_u64(out), Some(0));

        set(&mut ctx, &[f]);
        cb_fclose(&mut ctx);
        crate::filesystem::unregister_guest_path_mount("/stdioannexk0");
        let _ = fs::remove_dir_all(&root);
    }

    /// flockfile is recursive and keeps other threads out until the last
    /// funlockfile.
    #[test]
    fn flockfile_excludes_other_threads_until_released() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let flock = EXPORTS.iter().find(|e| e.1 == "flockfile").unwrap().2;
        let funlock = EXPORTS.iter().find(|e| e.1 == "funlockfile").unwrap().2;
        let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
        const STREAM: u64 = 0x5F86_0000_1000;
        let mut ctx = CpuContext::new(vm);
        ctx.set(crate::context::CpuRegister::Rdi, STREAM);
        assert_eq!(flock(&mut ctx), 0);
        assert_eq!(flock(&mut ctx), 0);
        let locked = Arc::new(AtomicBool::new(false));
        let seen = locked.clone();
        let other = std::thread::spawn(move || {
            let mut ctx = CpuContext::new(vm);
            ctx.set(crate::context::CpuRegister::Rdi, STREAM);
            assert_eq!(flock(&mut ctx), 0);
            seen.store(true, Ordering::SeqCst);
            assert_eq!(funlock(&mut ctx), 0);
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!locked.load(Ordering::SeqCst));
        assert_eq!(funlock(&mut ctx), 0);
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!locked.load(Ordering::SeqCst), "still held once");
        assert_eq!(funlock(&mut ctx), 0);
        other.join().unwrap();
        assert!(locked.load(Ordering::SeqCst));
        assert_eq!(file_lock(STREAM).state.lock().unwrap().1, 0);
    }

    /// The NIDs of clearerr and fileno are the ones the libraries use.
    #[test]
    fn clearerr_and_fileno_nids_match_their_names() {
        assert_eq!(ps5nid::compute_nid("clearerr"), "St9nbxSoezk");
        assert_eq!(ps5nid::compute_nid("fileno"), "Fm-dmyywH9Q");
        let mut reg = HleRegistry::new();
        register(&mut reg);
        assert!(reg.is_implemented("St9nbxSoezk"), "clearerr is still the catalog stub");
        assert!(reg.is_implemented("Fm-dmyywH9Q"), "fileno is still the catalog stub");
    }

    /// clearerr clears the error indicator and keeps the position; fileno
    /// gives the host descriptor of an open host file and -1 for a closed one.
    #[test]
    fn clearerr_clears_the_error_and_fileno_gives_the_descriptor() {
        use crate::context::CpuRegister::*;
        let root = temp_root("clearerr");
        assert!(crate::filesystem::register_guest_path_mount("/stdioclearerr0", &root));
        let root2 = root.clone();
        std::thread::spawn(move || {
            let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
            const BASE: u64 = 0x6A20_0000_0000;
            vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
            let mut ctx = CpuContext::new(vm);
            let errno = crate::kernel::bind_test_errno(&ctx);
            let set = |ctx: &mut CpuContext, args: &[u64]| {
                for (r, v) in [Rdi, Rsi, Rdx, Rcx].iter().zip(args) {
                    ctx.set(*r, *v);
                }
            };
            ctx.write_bytes(BASE, b"/stdioclearerr0/stream.bin\0");
            ctx.write_bytes(BASE + 0x100, b"w+\0");
            ctx.write_bytes(BASE + 0x200, b"ab");
            set(&mut ctx, &[BASE, BASE + 0x100]);
            let f = cb_fopen(&mut ctx) as u64;
            assert_ne!(f, 0);
            set(&mut ctx, &[BASE + 0x200, 1, 2, f]);
            assert_eq!(cb_fwrite(&mut ctx), 2);
            set(&mut ctx, &[f]);
            cb_rewind(&mut ctx);
            assert_eq!(cb_fgetc(&mut ctx), b'a' as i64);
            lock(&lookup(f).unwrap()).error = true;
            assert_eq!(cb_ferror(&mut ctx), 1);
            assert_eq!(cb_clearerr(&mut ctx), 0);
            assert_eq!(cb_ferror(&mut ctx), 0, "clearerr clears the error indicator");
            assert_eq!(cb_ftell(&mut ctx), 1, "clearerr keeps the position");
            set(&mut ctx, &[f]);
            let fd = cb_fileno(&mut ctx);
            assert!(fd >= 0, "a host file has a descriptor, got {fd}");
            ctx.write_u32(errno, 0);
            assert_eq!(cb_fclose(&mut ctx), 0);
            set(&mut ctx, &[f]);
            assert_eq!(cb_fileno(&mut ctx), -1, "a closed stream has no descriptor");
            assert_eq!(ctx.read_u32(errno), Some(9));
            set(&mut ctx, &[0]);
            assert_eq!(cb_fileno(&mut ctx), -1, "a null stream has no descriptor");
            assert_eq!(ctx.read_u32(errno), Some(9));
            ctx.write_u32(errno, 0);
            assert_eq!(cb_fileno_unlocked(&mut ctx), -1);
            assert_eq!(ctx.read_u32(errno), Some(0), "fileno_unlocked keeps errno");

            // putw / getw: four bytes, low byte first; a short file is -1.
            ctx.write_bytes(BASE, b"/stdioclearerr0/words.bin\0");
            set(&mut ctx, &[BASE, BASE + 0x100]);
            let f = cb_fopen(&mut ctx) as u64;
            assert_ne!(f, 0);
            set(&mut ctx, &[0x1234_5678, f]);
            assert_eq!(cb_putw(&mut ctx), 0);
            set(&mut ctx, &[f]);
            cb_rewind(&mut ctx);
            assert_eq!(cb_getw(&mut ctx), 0x1234_5678);
            assert_eq!(cb_getw(&mut ctx), -1, "end of file");
            assert_eq!(cb_fileno_unlocked(&mut ctx), cb_fileno(&mut ctx));
            assert_eq!(cb_fclose(&mut ctx), 0);
            assert_eq!(fs::read(root2.join("words.bin")).unwrap(), [0x78, 0x56, 0x34, 0x12]);
            set(&mut ctx, &[1, f]);
            assert_eq!(cb_putw(&mut ctx), -1, "closed stream");
            assert_eq!(ctx.read_u32(errno), Some(9));
        })
        .join()
        .unwrap();
        crate::filesystem::unregister_guest_path_mount("/stdioclearerr0");
        let _ = fs::remove_dir_all(&root);
    }

    /// Drive the exports through real guest memory.
    /// rewind: back to the start with errno kept, a pushed-back byte
    /// dropped, and the end-of-file and error indicators cleared.
    #[test]
    fn rewind_clears_the_indicators() {
        use crate::context::CpuRegister::*;
        let root = temp_root("rewind");
        assert!(crate::filesystem::register_guest_path_mount("/stdiorewind0", &root));
        std::thread::spawn(|| {
            let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
            const BASE: u64 = 0x6A18_0000_0000;
            vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
            let mut ctx = CpuContext::new(vm);
            let errno = crate::kernel::bind_test_errno(&ctx);
            let set = |ctx: &mut CpuContext, args: &[u64]| {
                for (r, v) in [Rdi, Rsi, Rdx, Rcx].iter().zip(args) {
                    ctx.set(*r, *v);
                }
            };
            ctx.write_bytes(BASE, b"/stdiorewind0/stream.bin\0");
            ctx.write_bytes(BASE + 0x100, b"w+\0");
            ctx.write_bytes(BASE + 0x200, b"ab");
            set(&mut ctx, &[BASE, BASE + 0x100]);
            let f = cb_fopen(&mut ctx) as u64;
            assert_ne!(f, 0);
            set(&mut ctx, &[BASE + 0x200, 1, 2, f]);
            assert_eq!(cb_fwrite(&mut ctx), 2);
            ctx.write_u32(errno, 7);
            set(&mut ctx, &[f]);
            cb_rewind(&mut ctx);
            assert_eq!(ctx.read_u32(errno), Some(7));
            assert_eq!(cb_ftell(&mut ctx), 0);
            assert_eq!(cb_fgetc(&mut ctx), b'a' as i64);
            set(&mut ctx, &[b'z' as u64, f]);
            assert_eq!(cb_ungetc(&mut ctx), b'z' as i64);
            set(&mut ctx, &[f]);
            cb_rewind(&mut ctx);
            assert_eq!(cb_fgetc(&mut ctx), b'a' as i64);
            assert_eq!(cb_fgetc(&mut ctx), b'b' as i64);
            assert_eq!(cb_fgetc(&mut ctx), EOF);
            assert_eq!(cb_feof(&mut ctx), 1);
            lock(&lookup(f).unwrap()).error = true;
            assert_eq!(cb_ferror(&mut ctx), 1);
            cb_rewind(&mut ctx);
            assert_eq!((cb_feof(&mut ctx), cb_ferror(&mut ctx)), (0, 0));
            assert_eq!(cb_fgetc(&mut ctx), b'a' as i64);
            assert_eq!(ctx.read_u32(errno), Some(7));
            assert_eq!(cb_fclose(&mut ctx), 0);
        })
        .join()
        .unwrap();
        crate::filesystem::unregister_guest_path_mount("/stdiorewind0");
        let _ = fs::remove_dir_all(&root);
    }

    /// getc_unlocked and putc_unlocked read and write bytes like getc and
    /// fputc, set the end-of-file indicator and work while the calling
    /// thread holds the stream lock.
    #[test]
    fn unlocked_character_io() {
        use crate::context::CpuRegister::*;
        let root = temp_root("unlocked");
        assert!(crate::filesystem::register_guest_path_mount("/stdiounlocked0", &root));
        std::thread::spawn(|| {
            let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
            const BASE: u64 = 0x7C46_0000_0000;
            vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
            let mut ctx = CpuContext::new(vm);
            crate::kernel::bind_test_errno(&ctx);
            let set = |ctx: &mut CpuContext, args: &[u64]| {
                for (r, v) in [Rdi, Rsi, Rdx, Rcx].iter().zip(args) {
                    ctx.set(*r, *v);
                }
            };
            ctx.write_bytes(BASE, b"/stdiounlocked0/bytes.bin\0");
            ctx.write_bytes(BASE + 0x100, b"w+\0");
            set(&mut ctx, &[BASE, BASE + 0x100]);
            let f = cb_fopen(&mut ctx) as u64;
            assert_ne!(f, 0);
            set(&mut ctx, &[f]);
            cb_lockfilelock(&mut ctx);
            for b in [b'x', 0xFF] {
                set(&mut ctx, &[b as u64, f]);
                assert_eq!(cb_fputc(&mut ctx), b as i64, "putc_unlocked returns the byte");
            }
            set(&mut ctx, &[f]);
            cb_rewind(&mut ctx);
            assert_eq!(cb_fgetc(&mut ctx), b'x' as i64);
            assert_eq!(cb_fgetc(&mut ctx), 0xFF, "bytes read unsigned");
            assert_eq!(cb_fgetc(&mut ctx), EOF);
            assert_eq!((cb_feof(&mut ctx), cb_ferror(&mut ctx)), (1, 0));
            cb_unlockfilelock(&mut ctx);
            assert_eq!(cb_fclose(&mut ctx), 0);
        })
        .join()
        .unwrap();
        crate::filesystem::unregister_guest_path_mount("/stdiounlocked0");
        let _ = fs::remove_dir_all(&root);
        // Registered as implementations (never bound to a native libc,
        // whose FILE layout differs from the HLE streams).
        for (name, cb) in [("getc_unlocked", cb_fgetc as HleCallback), ("putc_unlocked", cb_fputc as HleCallback)] {
            assert!(EXPORTS.iter().any(|e| e.1 == name && e.2 as usize == cb as usize), "{name}");
        }
    }

    #[test]
    fn end_to_end_file_io() {
        use crate::context::CpuRegister::*;
        let root = temp_root("e2e");
        let mount = root.join("mnt");
        fs::create_dir_all(&mount).unwrap();
        fs::write(mount.join("in.txt"), b"line one\nline two\nend").unwrap();
        assert!(crate::filesystem::register_guest_path_mount("/stdiotest0", &mount));

        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A5B_0000_0000;
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let s = |ctx: &CpuContext, off: u64, text: &str| {
            let mut b = text.as_bytes().to_vec();
            b.push(0);
            ctx.vm.write(BASE + off, &b).unwrap();
            BASE + off
        };
        let set = |ctx: &mut CpuContext, args: &[u64]| {
            for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
                ctx.set(*r, *v);
            }
        };
        let buf = BASE + 0x1000;

        // fopen for reading; the FILE object lives in zeroed guest memory.
        let p = s(&ctx, 0, "/stdiotest0/in.txt");
        let m = s(&ctx, 0x100, "rb");
        set(&mut ctx, &[p, m]);
        let f = cb_fopen(&mut ctx) as u64;
        assert_ne!(f, 0);
        assert_eq!(ctx.read_bytes(f, GUEST_FILE_OBJECT_SIZE as usize).unwrap(), vec![0u8; 0x100]);

        // fgets line by line
        set(&mut ctx, &[buf, 64, f]);
        assert_eq!(cb_fgets(&mut ctx) as u64, buf);
        assert_eq!(ctx.read_bytes(buf, 10).unwrap(), b"line one\n\0");
        set(&mut ctx, &[buf, 5, f]);
        assert_eq!(cb_fgets(&mut ctx) as u64, buf);
        assert_eq!(ctx.read_bytes(buf, 5).unwrap(), b"line\0");
        set(&mut ctx, &[f]);
        assert_eq!(cb_ftell(&mut ctx), 13);
        assert_eq!(cb_fgetc(&mut ctx), b' ' as i64);

        // fgetpos / fsetpos
        set(&mut ctx, &[f, buf + 0x200]);
        assert_eq!(cb_fgetpos(&mut ctx), 0);
        let raw: [u8; 24] = ctx.read_bytes(buf + 0x200, 24).unwrap().try_into().unwrap();
        assert_eq!(decode_fpos(&raw), 14);

        // fseek to the end; feof follows the position.
        set(&mut ctx, &[f, (-3i64) as u64, SEEK_END as u64]);
        assert_eq!(cb_fseek(&mut ctx), 0);
        set(&mut ctx, &[f]);
        assert_eq!(cb_feof(&mut ctx), 0);
        set(&mut ctx, &[buf, 1, 16, f]);
        assert_eq!(cb_fread(&mut ctx), 3);
        assert_eq!(ctx.read_bytes(buf, 3).unwrap(), b"end");
        set(&mut ctx, &[f]);
        assert_eq!(cb_feof(&mut ctx), 1);
        assert_eq!(cb_fgetc(&mut ctx), EOF);
        set(&mut ctx, &[buf, 64, f]);
        assert_eq!(cb_fgets(&mut ctx), 0);
        set(&mut ctx, &[f, 0, 7]);
        assert_eq!(cb_fseek(&mut ctx), EOF);
        set(&mut ctx, &[f, (-100i64) as u64, SEEK_CUR as u64]);
        assert_eq!(cb_fseek(&mut ctx), EOF);

        // fsetpos back, fread whole items
        set(&mut ctx, &[f, buf + 0x200]);
        assert_eq!(cb_fsetpos(&mut ctx), 0);
        set(&mut ctx, &[buf, 4, 2, f]);
        assert_eq!(cb_fread(&mut ctx), 1, "7 bytes left: one whole 4-byte item");
        assert_eq!(ctx.read_bytes(buf, 7).unwrap(), b"two\nend");
        set(&mut ctx, &[f]);
        assert_eq!(cb_ferror(&mut ctx), 0);
        cb_rewind(&mut ctx);
        assert_eq!(cb_ftell(&mut ctx), 0);

        // writes to a read-only stream fail
        set(&mut ctx, &[b'x' as u64, f]);
        assert_eq!(cb_fputc(&mut ctx), EOF);

        set(&mut ctx, &[f]);
        assert_eq!(cb_fclose(&mut ctx), 0);
        assert_eq!(cb_fclose(&mut ctx), EOF);
        assert_eq!(cb_ftell(&mut ctx), EOF);

        // missing file / bad mode / unknown mount
        let p = s(&ctx, 0, "/stdiotest0/missing.txt");
        set(&mut ctx, &[p, m]);
        assert_eq!(cb_fopen(&mut ctx), 0);
        let bad = s(&ctx, 0x100, "q");
        set(&mut ctx, &[p, bad]);
        assert_eq!(cb_fopen(&mut ctx), 0);
        let p = s(&ctx, 0, "/no/such/mount");
        set(&mut ctx, &[p, m]);
        assert_eq!(cb_fopen(&mut ctx), 0);

        // write + line buffering + fprintf + read back with w+
        let p = s(&ctx, 0, "/stdiotest0/sub/out.txt");
        let m = s(&ctx, 0x100, "w+");
        set(&mut ctx, &[p, m]);
        let w = cb_fopen(&mut ctx) as u64;
        assert_ne!(w, 0);
        set(&mut ctx, &[w, 0, IOLBF as u64, 0]);
        assert_eq!(cb_setvbuf(&mut ctx), 0);
        set(&mut ctx, &[w, 0, 9, 0]);
        assert_eq!(cb_setvbuf(&mut ctx), EOF);
        let host = mount.join("sub/out.txt");
        let text = s(&ctx, 0x200, "abc");
        set(&mut ctx, &[text, w]);
        assert_eq!(cb_fputs(&mut ctx), 3);
        assert_eq!(fs::read(&host).unwrap(), b"", "line-buffered: no newline yet");
        let fmt = s(&ctx, 0x300, "-%d-%s\n");
        let arg = s(&ctx, 0x400, "xy");
        set(&mut ctx, &[w, fmt, 42, arg]);
        assert_eq!(cb_fprintf(&mut ctx), 7);
        assert_eq!(fs::read(&host).unwrap(), b"abc-42-xy\n");
        ctx.vm.write(buf, b"ZZZZ").unwrap();
        set(&mut ctx, &[buf, 2, 2, w]);
        assert_eq!(cb_fwrite(&mut ctx), 2);
        set(&mut ctx, &[b'!' as u64, w]);
        assert_eq!(cb_fputc(&mut ctx), b'!' as i64);
        set(&mut ctx, &[w]);
        assert_eq!(cb_ftell(&mut ctx), 15);
        assert_eq!(cb_fflush(&mut ctx), 0);
        assert_eq!(fs::read(&host).unwrap(), b"abc-42-xy\nZZZZ!");
        cb_rewind(&mut ctx);
        set(&mut ctx, &[buf, 64, w]);
        assert_eq!(cb_fgets(&mut ctx) as u64, buf);
        assert_eq!(ctx.read_bytes(buf, 11).unwrap(), b"abc-42-xy\n\0");
        // overwrite in the middle after a seek
        set(&mut ctx, &[w, 1, SEEK_SET as u64]);
        assert_eq!(cb_fseek(&mut ctx), 0);
        set(&mut ctx, &[b'B' as u64, w]);
        cb_fputc(&mut ctx);
        set(&mut ctx, &[w]);
        assert_eq!(cb_fclose(&mut ctx), 0);
        assert_eq!(fs::read(&host).unwrap(), b"aBc-42-xy\nZZZZ!");

        // append mode always writes at the end
        let m = s(&ctx, 0x100, "a");
        set(&mut ctx, &[p, m]);
        let a = cb_fopen(&mut ctx) as u64;
        assert_ne!(a, 0);
        set(&mut ctx, &[a]);
        assert_eq!(cb_ftell(&mut ctx), 15);
        set(&mut ctx, &[a, 0, SEEK_SET as u64]);
        assert_eq!(cb_fseek(&mut ctx), 0);
        let text = s(&ctx, 0x200, "++");
        set(&mut ctx, &[text, a]);
        assert_eq!(cb_fputs(&mut ctx), 2);
        set(&mut ctx, &[a]);
        assert_eq!(cb_fclose(&mut ctx), 0);
        assert_eq!(fs::read(&host).unwrap(), b"aBc-42-xy\nZZZZ!++");

        // freopen swaps the stream behind the same FILE*
        let m = s(&ctx, 0x100, "r");
        let p1 = s(&ctx, 0, "/stdiotest0/in.txt");
        set(&mut ctx, &[p1, m]);
        let r = cb_fopen(&mut ctx) as u64;
        let p2 = s(&ctx, 0x500, "/stdiotest0/sub/out.txt");
        set(&mut ctx, &[p2, m, r]);
        assert_eq!(cb_freopen(&mut ctx) as u64, r);
        set(&mut ctx, &[r]);
        assert_eq!(cb_fgetc(&mut ctx), b'a' as i64);
        set(&mut ctx, &[p2, m, 0x1234]);
        assert_eq!(cb_freopen(&mut ctx), 0);
        set(&mut ctx, &[r]);
        assert_eq!(cb_fclose(&mut ctx), 0);

        // console streams: unknown FILE* pointers accept writes
        let text = s(&ctx, 0x200, "hello console\n");
        set(&mut ctx, &[text, 0xDEAD_0000]);
        assert_eq!(cb_fputs(&mut ctx), 14);
        set(&mut ctx, &[0xDEAD_0000]);
        assert_eq!(cb_fgetc(&mut ctx), EOF);

        crate::filesystem::unregister_guest_path_mount("/stdiotest0");
        let _ = vm.unmap(BASE, 0x10000);
        let _ = fs::remove_dir_all(&root);
    }
}
