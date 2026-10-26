// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Host operating-system layer of the HLE modules.
//!
//! `c` is the C library surface the modules call as `libc::…`: the `libc`
//! crate on Linux, a POSIX layer over the C runtime and Winsock on Windows
//! (each module imports it as `libc` there).

pub mod fs;

#[cfg(unix)]
pub use ::libc as c;

#[cfg(windows)]
pub mod c_windows;
#[cfg(windows)]
pub use c_windows as c;

/// The C runtime `errno` of a host I/O error. On Windows, errors of the
/// file system come as Windows error codes and are translated; errors of
/// the POSIX layer already carry an `errno`.
#[cfg(windows)]
pub fn errno_of(e: &std::io::Error) -> Option<i32> {
    use ::libc::*;
    let code = e.raw_os_error()?;
    Some(match code {
        3 | 123 | 161 => ENOENT,   // PATH_NOT_FOUND, INVALID_NAME, BAD_PATHNAME
        5 => EACCES,               // ACCESS_DENIED
        6 => EBADF,                // INVALID_HANDLE
        80 | 183 => EEXIST,        // FILE_EXISTS, ALREADY_EXISTS
        87 => EINVAL,              // INVALID_PARAMETER
        112 | 39 => ENOSPC,        // DISK_FULL, HANDLE_DISK_FULL
        145 => ENOTEMPTY,          // DIR_NOT_EMPTY
        206 => ENAMETOOLONG,       // FILENAME_EXCED_RANGE
        267 => ENOTDIR,            // DIRECTORY
        1920 | 1921 => ELOOP,      // CANT_ACCESS_FILE, CANT_RESOLVE_FILENAME
        19 => EROFS,               // WRITE_PROTECT
        109 | 232 => EPIPE,        // BROKEN_PIPE, NO_DATA
        33 | 158 => EAGAIN,        // LOCK_VIOLATION, NOT_LOCKED
        1224 => EBUSY,             // USER_MAPPED_FILE
        other => other,
    })
}

/// The C runtime `errno` of a host I/O error.
#[cfg(unix)]
pub fn errno_of(e: &std::io::Error) -> Option<i32> {
    e.raw_os_error()
}
