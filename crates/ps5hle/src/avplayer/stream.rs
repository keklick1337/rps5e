// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Byte source behind an AvPlayer movie, with AVIO read/seek semantics.
//!
//! A movie is read either from the host file system (the guest path mapped
//! through the mounts) or through the title's file replacement callbacks.
//! Guest callbacks may only run on a guest thread inside an export, so while
//! the demux worker owns the movie its reads are posted as requests and
//! performed by the next export call of that player (`GuestFile::service`);
//! on the guest thread (open, seek, start) they are performed directly.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use super::{arena_alloc, arena_free, guest_call, FileReplacement};
use std::fs::File;
use crate::host::fs::FileExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// -MKTAG('E','O','F',' ')
pub const AVERROR_EOF: i32 = -0x2046_4F45;
/// -MKTAG('E','X','I','T')
pub const AVERROR_EXIT: i32 = -0x5449_5845;
pub const AVERROR_EIO: i32 = -libc::EIO;
pub const AVERROR_EINVAL: i32 = -libc::EINVAL;
pub const AVSEEK_SIZE: i32 = 0x10000;
pub const AVSEEK_FORCE: i32 = 0x20000;
pub const SEEK_SET: i32 = 0;
pub const SEEK_CUR: i32 = 1;
pub const SEEK_END: i32 = 2;

/// AVIO buffer for host files.
pub const NATIVE_BUFFER: usize = 64 * 1024;
/// AVIO buffer (and guest scratch) for reads through guest callbacks: one
/// request per export call, so requests are large.
pub const GUEST_BUFFER: usize = 1 << 20;

enum Backing {
    Native(File),
    Guest(Arc<GuestFile>),
}

pub struct FileStreamer {
    backing: Backing,
    pos: u64,
    size: u64,
}

impl FileStreamer {
    /// Open a host file.
    pub fn open_native(host: &std::path::Path) -> Option<Self> {
        let file = File::open(host).ok()?;
        let size = file.metadata().ok()?.len();
        Self::checked(Backing::Native(file), size)
    }

    /// Open `path` through the title's file replacement callbacks.
    pub(super) fn open_guest(file: FileReplacement, path: &str) -> Option<Self> {
        if file.close == 0 || file.read_offset == 0 || file.size == 0 {
            return None;
        }
        let g = GuestFile::open(file, path)?;
        let size = guest_call(file.size, [file.object, 0, 0, 0, 0, 0]);
        Self::checked(Backing::Guest(Arc::new(g)), size)
    }

    fn checked(backing: Backing, size: u64) -> Option<Self> {
        // The callbacks' file stays open until the streamer drops.
        let s = Self { backing, pos: 0, size };
        (size != 0 && size <= i64::MAX as u64).then_some(s)
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn buffer_size(&self) -> usize {
        match self.backing {
            Backing::Native(_) => NATIVE_BUFFER,
            Backing::Guest(_) => GUEST_BUFFER,
        }
    }

    pub fn guest(&self) -> Option<&Arc<GuestFile>> {
        match &self.backing {
            Backing::Guest(g) => Some(g),
            Backing::Native(_) => None,
        }
    }

    /// AVIO read_packet.
    pub fn read(&mut self, buf: &mut [u8]) -> i32 {
        if self.pos >= self.size {
            return AVERROR_EOF;
        }
        let len = (buf.len() as u64).min(self.size - self.pos).min(i32::MAX as u64) as usize;
        let buf = &mut buf[..len];
        let read = match &self.backing {
            Backing::Native(f) => {
                let mut done = 0;
                while done < len {
                    match f.read_at(&mut buf[done..], self.pos + done as u64) {
                        Ok(0) => break,
                        Ok(n) => done += n,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                        Err(_) => break,
                    }
                }
                done as i32
            }
            Backing::Guest(g) => g.read(buf, self.pos),
        };
        if read <= 0 || read as usize > len {
            return if read < 0 { read } else { AVERROR_EIO };
        }
        self.pos += read as u64;
        read
    }

    /// AVIO seek.
    pub fn seek(&mut self, offset: i64, whence: i32) -> i64 {
        if whence & AVSEEK_SIZE != 0 {
            return self.size as i64;
        }
        let base = match whence & !AVSEEK_FORCE {
            SEEK_SET => 0,
            SEEK_CUR => self.pos,
            SEEK_END => self.size,
            _ => return AVERROR_EINVAL as i64,
        };
        let distance = offset.unsigned_abs();
        if (offset < 0 && distance > base) || (offset >= 0 && distance > i64::MAX as u64 - base) {
            return AVERROR_EINVAL as i64;
        }
        let position = if offset < 0 { base - distance } else { base + distance };
        self.pos = position;
        position as i64
    }
}

impl Drop for FileStreamer {
    fn drop(&mut self) {
        if let Backing::Guest(g) = &self.backing {
            g.close();
        }
    }
}

#[derive(Clone, Copy)]
struct Request {
    buf: usize,
    len: usize,
    offset: u64,
}

enum Slot {
    Idle,
    Pending(Request),
    Busy,
    Done(i32),
}

struct Channel {
    slot: Slot,
    cancelled: bool,
}

/// A file opened through the title's callbacks.
pub struct GuestFile {
    file: FileReplacement,
    scratch: u64,
    remote: AtomicBool,
    closed: AtomicBool,
    chan: Mutex<Channel>,
    cv: Condvar,
}

impl GuestFile {
    fn open(file: FileReplacement, path: &str) -> Option<Self> {
        let scratch = arena_alloc(GUEST_BUFFER as u64)?;
        let g = Self {
            file,
            scratch,
            remote: AtomicBool::new(false),
            closed: AtomicBool::new(true),
            chan: Mutex::new(Channel { slot: Slot::Idle, cancelled: false }),
            cv: Condvar::new(),
        };
        let name = path.as_bytes();
        if name.len() >= GUEST_BUFFER {
            return None;
        }
        // SAFETY: the scratch block is ours and GUEST_BUFFER bytes long.
        unsafe {
            std::ptr::copy_nonoverlapping(name.as_ptr(), scratch as *mut u8, name.len());
            *(scratch as *mut u8).add(name.len()) = 0;
        }
        let r = guest_call(file.open, [file.object, scratch, 0, 0, 0, 0]) as i32;
        if r < 0 {
            log::warn!("[avplayer] file replacement open({path}) failed: {r:#x}");
            return None;
        }
        g.closed.store(false, Ordering::SeqCst);
        Some(g)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Channel> {
        self.chan.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Read on the calling (guest) thread.
    fn read_direct(&self, buf: &mut [u8], offset: u64) -> i32 {
        let len = buf.len().min(GUEST_BUFFER);
        let f = self.file;
        let n = guest_call(f.read_offset, [f.object, self.scratch, offset, len as u64, 0, 0]) as i32;
        if n > 0 && n as usize <= len {
            // SAFETY: the callback wrote n bytes into our scratch block.
            unsafe { std::ptr::copy_nonoverlapping(self.scratch as *const u8, buf.as_mut_ptr(), n as usize) };
        }
        n
    }

    fn read(&self, buf: &mut [u8], offset: u64) -> i32 {
        if !self.remote.load(Ordering::Acquire) {
            return self.read_direct(buf, offset);
        }
        let mut c = self.lock();
        if c.cancelled {
            return AVERROR_EXIT;
        }
        c.slot = Slot::Pending(Request { buf: buf.as_mut_ptr() as usize, len: buf.len().min(GUEST_BUFFER), offset });
        loop {
            match c.slot {
                Slot::Done(n) => {
                    c.slot = Slot::Idle;
                    return n;
                }
                Slot::Pending(_) if c.cancelled => {
                    c.slot = Slot::Idle;
                    return AVERROR_EXIT;
                }
                _ => c = self.cv.wait(c).unwrap_or_else(|e| e.into_inner()),
            }
        }
    }

    /// Perform a read the demux worker is waiting for (guest thread, no
    /// player lock held).
    pub fn service(&self) {
        let req = {
            let mut c = self.lock();
            match c.slot {
                Slot::Pending(r) => {
                    c.slot = Slot::Busy;
                    r
                }
                _ => return,
            }
        };
        // SAFETY: the worker blocks until Done, keeping its buffer alive.
        let buf = unsafe { std::slice::from_raw_parts_mut(req.buf as *mut u8, req.len) };
        let n = self.read_direct(buf, req.offset);
        self.lock().slot = Slot::Done(n);
        self.cv.notify_all();
    }

    /// Worker reads go through `service` from now on.
    pub fn set_remote(&self, remote: bool) {
        self.lock().cancelled = false;
        self.remote.store(remote, Ordering::Release);
    }

    /// Fail pending and future worker reads (playback is stopping).
    pub fn cancel(&self) {
        self.lock().cancelled = true;
        self.cv.notify_all();
    }

    fn close(&self) {
        if !self.closed.swap(true, Ordering::SeqCst) {
            guest_call(self.file.close, [self.file.object, 0, 0, 0, 0, 0]);
        }
    }
}

impl Drop for GuestFile {
    fn drop(&mut self) {
        arena_free(self.scratch);
    }
}
