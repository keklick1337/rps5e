// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! pipe(): a host pipe behind two descriptors of the table. Reads block
//! until data or end of file, writes to a pipe without a reader fail with
//! EPIPE (the host never raises SIGPIPE), fstat reports a FIFO.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use super::*;
use crate::host::fs::{AsRawFd, FromRawFd, OwnedFd};

/// One end of a host pipe.
pub(crate) struct PipeEnd {
    fd: OwnedFd,
    /// The write end (the read end otherwise).
    pub(crate) write: bool,
}

impl PipeEnd {
    pub(super) fn raw(&self) -> i32 {
        self.fd.as_raw_fd()
    }

    pub(super) fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        if self.write {
            return Err(io::Error::from_raw_os_error(libc::EBADF));
        }
        // SAFETY: `buf` is a live mutable slice of that length.
        let n = unsafe { libc::read(self.raw(), buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
        if n < 0 { Err(io::Error::last_os_error()) } else { Ok(n as usize) }
    }

    /// Write all of `data` (a pipe write blocks until it fits), with SIGPIPE
    /// held off for the calling thread.
    pub(super) fn write_all(&self, data: &[u8]) -> io::Result<()> {
        if !self.write {
            return Err(io::Error::from_raw_os_error(libc::EBADF));
        }
        let _guard = SigpipeBlocked::new();
        let mut off = 0;
        while off < data.len() {
            // SAFETY: the remaining bytes of `data`.
            let n = unsafe { libc::write(self.raw(), data[off..].as_ptr() as *const libc::c_void, data.len() - off) };
            if n < 0 {
                let e = io::Error::last_os_error();
                if e.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(e);
            }
            off += n as usize;
        }
        Ok(())
    }
}

/// SIGPIPE blocked for the calling thread; a SIGPIPE raised meanwhile is
/// consumed before the old mask comes back.
struct SigpipeBlocked {
    old: libc::sigset_t,
}

impl SigpipeBlocked {
    fn new() -> Self {
        // SAFETY: plain signal-mask calls on locals.
        unsafe {
            let mut set: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut set);
            libc::sigaddset(&mut set, libc::SIGPIPE);
            let mut old: libc::sigset_t = std::mem::zeroed();
            libc::pthread_sigmask(libc::SIG_BLOCK, &set, &mut old);
            Self { old }
        }
    }
}

impl Drop for SigpipeBlocked {
    fn drop(&mut self) {
        // SAFETY: plain signal calls on locals.
        unsafe {
            if libc::sigismember(&self.old, libc::SIGPIPE) == 0 {
                let mut set: libc::sigset_t = std::mem::zeroed();
                libc::sigemptyset(&mut set);
                libc::sigaddset(&mut set, libc::SIGPIPE);
                let zero = libc::timespec { tv_sec: 0, tv_nsec: 0 };
                while libc::sigtimedwait(&set, std::ptr::null_mut(), &zero) == libc::SIGPIPE {}
            }
            libc::pthread_sigmask(libc::SIG_SETMASK, &self.old, std::ptr::null_mut());
        }
    }
}

/// pipe(int fds[2]): fds[0] reads, fds[1] writes.
pub(super) fn pipe_core(ctx: &CpuContext, out: u64) -> FsResult<i64> {
    if out == 0 || ctx.read_bytes(out, 8).is_none() {
        return Err(EFAULT);
    }
    let mut native = [0i32; 2];
    // SAFETY: `native` holds two ints.
    if unsafe { libc::pipe2(native.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        return Err(host_errno(&io::Error::last_os_error()));
    }
    // SAFETY: pipe2 returned two fresh descriptors that nothing else owns.
    let ends = native.map(|fd| unsafe { OwnedFd::from_raw_fd(fd) });
    let [r, w] = ends;
    let rfd = alloc_fd(Handle::Pipe(Arc::new(PipeEnd { fd: r, write: false })));
    let wfd = alloc_fd(Handle::Pipe(Arc::new(PipeEnd { fd: w, write: true })));
    set_status(wfd, O_WRONLY as u32);
    let mut b = [0u8; 8];
    b[..4].copy_from_slice(&rfd.to_le_bytes());
    b[4..].copy_from_slice(&wfd.to_le_bytes());
    if write_guest(ctx, out, &b).is_err() {
        let _ = close_core(rfd);
        let _ = close_core(wfd);
        return Err(EFAULT);
    }
    log::debug!("[fs] pipe -> {rfd}, {wfd}");
    Ok(0)
}

pub(super) fn read(ctx: &CpuContext, p: &PipeEnd, buf: u64, requested: usize) -> FsResult<i64> {
    // One read: whatever the pipe holds (blocking until something or EOF).
    let mut chunk = vec![0u8; requested.min(IO_CHUNK)];
    let n = loop {
        match p.read(&mut chunk) {
            Ok(n) => break n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(host_errno(&e)),
        }
    };
    write_guest(ctx, buf, &chunk[..n])?;
    Ok(n as i64)
}

pub(super) fn write(ctx: &CpuContext, p: &PipeEnd, buf: u64, requested: usize) -> FsResult<i64> {
    let n = transfer_from_guest(ctx, buf, requested, |data, _| p.write_all(data))?;
    Ok(n as i64)
}

/// Stat of a pipe end: a FIFO owned by the user, size 0.
pub(super) fn stat(fd: i32) -> [u8; KERNEL_STAT_SIZE] {
    const STAT_MODE_FIFO: u16 = 0o010600;
    let now = SystemTime::now();
    let mut b = build_kernel_stat(false, 0, now, now, now, &format!("pipe:{fd}"));
    b[8..10].copy_from_slice(&STAT_MODE_FIFO.to_le_bytes());
    b
}

/// Bytes waiting in the read end `fd` and whether every writer is gone;
/// None when `fd` is not the read end of a pipe.
pub(crate) fn read_state(fd: i32) -> Option<(u64, bool)> {
    let Some(Handle::Pipe(p)) = get_handle(fd) else { return None };
    if p.write {
        return None;
    }
    let mut n: libc::c_int = 0;
    // SAFETY: FIONREAD stores one int.
    let available = if unsafe { libc::ioctl(p.raw(), libc::FIONREAD, &mut n) } == 0 { n.max(0) as u64 } else { 0 };
    let mut entry = libc::pollfd { fd: p.raw(), events: libc::POLLIN, revents: 0 };
    // SAFETY: one valid pollfd.
    let hangup = unsafe { libc::poll(&mut entry, 1, 0) } > 0 && entry.revents & libc::POLLHUP != 0;
    Some((available, hangup))
}

/// The write end `fd` of a pipe: (free buffer space, read end closed,
/// a write is taken now); None when `fd` is not the write end of a pipe.
/// The free space is the host pipe's capacity less what is queued, 0 once
/// the read end is closed.
pub(crate) fn write_state(fd: i32) -> Option<(u64, bool, bool)> {
    let Some(Handle::Pipe(p)) = get_handle(fd) else { return None };
    if !p.write {
        return None;
    }
    let mut entry = libc::pollfd { fd: p.raw(), events: libc::POLLOUT, revents: 0 };
    let mut queued: libc::c_int = 0;
    // SAFETY: one valid pollfd; FIONREAD stores one int; F_GETPIPE_SZ takes
    // no argument.
    let (polled, counted, capacity) = unsafe {
        (libc::poll(&mut entry, 1, 0), libc::ioctl(p.raw(), libc::FIONREAD, &mut queued), libc::fcntl(p.raw(), libc::F_GETPIPE_SZ, 0))
    };
    if polled < 0 || counted != 0 || capacity < 0 {
        log::error!("[fs] querying the write end of pipe {fd} failed: {}", io::Error::last_os_error());
        return None;
    }
    let eof = entry.revents & libc::POLLERR != 0;
    let writable = entry.revents & libc::POLLOUT != 0;
    let free = if eof { 0 } else { (capacity as i64 - queued as i64).max(0) as u64 };
    Some((free, eof, writable))
}

/// True when `fd` is either end of a pipe.
pub(crate) fn is_pipe(fd: i32) -> bool {
    matches!(get_handle(fd), Some(Handle::Pipe(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps5vm::{MemProtect, VirtualMemory};

    const BASE: u64 = 0x7E50_0000_0000;

    /// pipe(): bytes written come out of the read end, a closed writer
    /// gives end of file, a closed reader gives EPIPE, fstat says FIFO,
    /// and the readiness the event queues see follows the data.
    #[test]
    fn pipe_transfers_and_reports_state() {
        let vm = VirtualMemory::new();
        vm.map(BASE, 0x1000, MemProtect::RW).unwrap();
        let ctx = CpuContext::new(&vm);
        assert_eq!(pipe_core(&ctx, 0), Err(EFAULT));
        assert_eq!(pipe_core(&ctx, BASE), Ok(0));
        let (r, w) = (ctx.read_u32(BASE).unwrap() as i32, ctx.read_u32(BASE + 4).unwrap() as i32);
        assert!(r > 2 && w > 2 && r != w && is_pipe(r) && is_pipe(w));
        assert_eq!(read_state(r), Some((0, false)));
        assert_eq!(read_state(w), None);
        ctx.write_bytes(BASE + 0x100, b"abc");
        assert_eq!(write_core(&ctx, w, BASE + 0x100, 3), Ok(3));
        assert_eq!(read_state(r), Some((3, false)));
        assert_eq!(write_core(&ctx, r, BASE + 0x100, 3), Err(EBADF));
        assert_eq!(read_core(&ctx, w, BASE + 0x200, 3), Err(EBADF));
        assert_eq!(read_core(&ctx, r, BASE + 0x200, 16), Ok(3));
        assert_eq!(ctx.read_bytes(BASE + 0x200, 3).unwrap(), b"abc");
        assert_eq!(&stat(r)[8..10], &0o010600u16.to_le_bytes());
        assert_eq!(close_core(w), Ok(0));
        assert_eq!(read_state(r), Some((0, true)));
        assert_eq!(read_core(&ctx, r, BASE + 0x200, 16), Ok(0));
        assert_eq!(close_core(r), Ok(0));
        assert!(!is_pipe(r));

        assert_eq!(pipe_core(&ctx, BASE), Ok(0));
        let (r, w) = (ctx.read_u32(BASE).unwrap() as i32, ctx.read_u32(BASE + 4).unwrap() as i32);
        assert_eq!(close_core(r), Ok(0));
        assert_eq!(write_core(&ctx, w, BASE + 0x100, 3), Err(errno::EPIPE));
        assert_eq!(close_core(w), Ok(0));
    }
}
