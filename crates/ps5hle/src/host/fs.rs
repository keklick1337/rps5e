// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! File extension traits with one surface on every host: positioned reads
//! and writes, POSIX open flags and byte views of OS strings. On Linux
//! these are the standard library's Unix traits.

#[cfg(unix)]
pub use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
pub use std::os::unix::fs::{FileExt, OpenOptionsExt};

#[cfg(windows)]
pub use self::win::*;

use std::fs::File;
use std::io;

/// A host descriptor: a file descriptor on Linux, a handle value on
/// Windows (handles fit in 32 bits there).
pub type RawFd = i32;

#[cfg(unix)]
pub use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

/// An owned descriptor of the POSIX layer (a socket or a pipe end over a
/// socket), closed on drop.
#[cfg(windows)]
#[derive(Debug)]
pub struct OwnedFd(RawFd);

#[cfg(windows)]
impl Drop for OwnedFd {
    fn drop(&mut self) {
        // SAFETY: our own descriptor.
        unsafe { crate::host::c::close(self.0) };
    }
}

#[cfg(windows)]
pub trait AsRawFd {
    fn as_raw_fd(&self) -> RawFd;
}

#[cfg(windows)]
impl AsRawFd for OwnedFd {
    fn as_raw_fd(&self) -> RawFd {
        self.0
    }
}

#[cfg(windows)]
pub trait FromRawFd {
    /// # Safety
    /// `fd` must be an open descriptor nobody else owns.
    unsafe fn from_raw_fd(fd: RawFd) -> Self;
}

#[cfg(windows)]
impl FromRawFd for OwnedFd {
    unsafe fn from_raw_fd(fd: RawFd) -> Self {
        OwnedFd(fd)
    }
}

/// The host descriptor of an open file.
#[cfg(unix)]
pub fn as_raw_fd(f: &File) -> RawFd {
    std::os::fd::AsRawFd::as_raw_fd(f)
}

#[cfg(windows)]
pub fn as_raw_fd(f: &File) -> RawFd {
    std::os::windows::io::AsRawHandle::as_raw_handle(f) as isize as RawFd
}

/// Take ownership of a host descriptor.
///
/// # Safety
/// `fd` must be an open descriptor nobody else owns.
#[cfg(unix)]
pub unsafe fn from_raw_fd(fd: RawFd) -> File {
    std::os::fd::FromRawFd::from_raw_fd(fd)
}

#[cfg(windows)]
pub unsafe fn from_raw_fd(fd: RawFd) -> File {
    std::os::windows::io::FromRawHandle::from_raw_handle(fd as isize as std::os::windows::io::RawHandle)
}

/// An anonymous memory-backed file (`memfd_create`). On Windows a
/// temporary file deleted on close, mostly kept in the cache.
#[cfg(unix)]
pub fn memfd(name: &str) -> io::Result<File> {
    let c = std::ffi::CString::new(name).map_err(|_| io::Error::from_raw_os_error(libc::EINVAL))?;
    // SAFETY: plain syscall; the descriptor is owned by the new File.
    let fd = unsafe { libc::memfd_create(c.as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a fresh descriptor.
    Ok(unsafe { from_raw_fd(fd) })
}

#[cfg(windows)]
pub fn memfd(name: &str) -> io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt as _;
    use std::sync::atomic::{AtomicU64, Ordering};
    const FILE_ATTRIBUTE_TEMPORARY: u32 = 0x100;
    const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x0400_0000;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let clean: String = name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    let path = std::env::temp_dir().join(format!("rps5e-{}-{}-{clean}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    let mut opts = std::fs::OpenOptions::new();
    opts.read(true)
        .write(true)
        .create_new(true)
        .share_mode(7)
        // Read, write and execute: shared memory may hold code.
        .access_mode(0x8000_0000 | 0x4000_0000 | 0x2000_0000)
        .attributes(FILE_ATTRIBUTE_TEMPORARY);
    std::os::windows::fs::OpenOptionsExt::custom_flags(&mut opts, FILE_FLAG_DELETE_ON_CLOSE);
    opts.open(path)
}

/// A new open file description of `f` (its own offset), read-only or
/// read/write.
#[cfg(unix)]
pub fn reopen(f: &File, writable: bool) -> io::Result<File> {
    std::fs::OpenOptions::new().read(true).write(writable).open(format!("/proc/self/fd/{}", as_raw_fd(f)))
}

#[cfg(windows)]
pub fn reopen(f: &File, writable: bool) -> io::Result<File> {
    use windows_sys::Win32::Storage::FileSystem::{ReOpenFile, FILE_GENERIC_READ, FILE_GENERIC_WRITE, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE};
    let access = FILE_GENERIC_READ | if writable { FILE_GENERIC_WRITE } else { 0 };
    let h = std::os::windows::io::AsRawHandle::as_raw_handle(f);
    // SAFETY: a handle we own; the new handle is owned by the new File.
    let n = unsafe { ReOpenFile(h as _, access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, 0) };
    if n == windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a fresh handle.
    Ok(unsafe { std::os::windows::io::FromRawHandle::from_raw_handle(n as _) })
}

#[cfg(windows)]
mod win {
    use std::ffi::OsStr;
    use std::fs::{File, OpenOptions};
    use std::io;

    /// Positioned reads and writes (`pread`/`pwrite`). Every access through
    /// this trait names its offset; the file cursor is not meaningful.
    pub trait FileExt {
        fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize>;
        fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize>;

        fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
            while !buf.is_empty() {
                match self.read_at(buf, offset) {
                    Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                    Ok(n) => {
                        buf = &mut buf[n..];
                        offset += n as u64;
                    }
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            }
            Ok(())
        }

        fn write_all_at(&self, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
            while !buf.is_empty() {
                match self.write_at(buf, offset) {
                    Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                    Ok(n) => {
                        buf = &buf[n..];
                        offset += n as u64;
                    }
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            }
            Ok(())
        }
    }

    impl FileExt for File {
        fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
            match std::os::windows::fs::FileExt::seek_read(self, buf, offset) {
                // Reading at or past the end reports EOF as zero bytes.
                Err(e) if e.raw_os_error() == Some(38) => Ok(0),
                r => r,
            }
        }

        fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
            std::os::windows::fs::FileExt::seek_write(self, buf, offset)
        }
    }

    /// POSIX `open` flags on `OpenOptions`.
    pub trait OpenOptionsExt {
        /// Permission bits: Windows files have none to set.
        fn mode(&mut self, mode: u32) -> &mut Self;
        /// `O_CREAT`, `O_EXCL`, `O_TRUNC` and `O_APPEND` (C runtime values);
        /// other flags are ignored.
        fn custom_flags(&mut self, flags: i32) -> &mut Self;
    }

    impl OpenOptionsExt for OpenOptions {
        fn mode(&mut self, _mode: u32) -> &mut Self {
            self
        }

        fn custom_flags(&mut self, flags: i32) -> &mut Self {
            if flags & libc::O_CREAT != 0 {
                if flags & libc::O_EXCL != 0 {
                    self.create_new(true);
                } else {
                    self.create(true);
                }
            }
            if flags & libc::O_TRUNC != 0 {
                self.truncate(true);
            }
            if flags & libc::O_APPEND != 0 {
                self.append(true);
            }
            self
        }
    }

    /// Byte views of OS strings. Windows paths are WTF-8 here; bytes that
    /// are not UTF-8 become U+FFFD.
    pub trait OsStrExt {
        fn as_bytes(&self) -> &[u8];
        fn from_bytes(bytes: &[u8]) -> &Self;
    }

    impl OsStrExt for OsStr {
        fn as_bytes(&self) -> &[u8] {
            self.as_encoded_bytes()
        }

        fn from_bytes(bytes: &[u8]) -> &OsStr {
            match std::str::from_utf8(bytes) {
                Ok(s) => OsStr::new(s),
                Err(_) => OsStr::new("\u{FFFD}"),
            }
        }
    }
}
